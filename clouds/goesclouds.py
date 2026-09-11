"""Reusable functions for calibrating human-judged CTIO cloud reports
against GOES-13 satellite brightness.


Layering:

    Configuration   module constants + a GoesCloudsConfig dataclass
    Layer 1: LOAD   disk/cache -> DataFrame      (no computation)
    Layer 2: COMPUTE  DataFrame -> DataFrame     (pure, no I/O)
    Layer 3: PLOT   DataFrame -> Figure          (no computation)
    Layer 4: WRITE  DataFrame -> disk artifacts
    Layer 5: ACQUIRE  network (opt-in, never called implicitly)

Nothing in this module touches the network, reads credentials, or performs
any computation at import time.

The "notebook" referred to in the comments and docstrings is a prototype
notebooks found in prototype/cloud_satellite.ipynb.

"""

import datetime
import functools
import glob
import gzip
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from astropy.coordinates import EarthLocation
from astropy.time import Time
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor, QuantileRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SATELLITE_NAME = "GOES-13"          # SDS inventory spelling
SATELLITE_NAME_MCFETCH = "GOES13"   # mcfetch spelling / filename prefix

CUTOUT_CENTER_PIX = 128
CUTOUT_SIZE_PIX = "256+256"

DEFAULT_WINDOW_SIZE = 4             # notebook-compatible 4x4 window
DEFAULT_BAND = 4
CORRECTION_BAND = 2
ALL_BANDS = (2, 4, 6)
MIN_CLEAR_SAMPLES_FOR_FIT = 2

SAMPLES_PER_CLOUD_LEVEL = 99
SAMPLE_YEAR_RANGE = (2013, 2016)     # inclusive
SAMPLE_RNG_SEED = 6563

MISSING_CLOUDS_SENTINEL = 9
CLOUD_LEVELS = range(0, 9)
OBSERVABLE_CUT = 2.5                # <= 2.5 eighths counts as observable
STAT_NAMES = ("min", "mean", "max", "25%", "50%", "75%", "std", "IQR", "range")
SATELLITE_CLOUDY_STAT_COLUMNS = ("mean", "std", "min", "25%", "50%", "75%", "max")

DEFAULT_DATA_DIR = "data"
DEFAULT_INVENTORY_CACHE = "sds_inventories.h5"
DEFAULT_REPORTS_PATH = "clouds_ctio_blanco.h5"
DEFAULT_NIGHT_EVENTS = "night_events.h5"

ALLOW_DOWNLOAD = False              # global kill switch; see Layer 5
SDS_ACCESS_KEY_PATH = os.environ.get(
    "MCFETCH_ACCESS_KEY_PATH",
    os.path.expanduser("~/McFETCH_access_key"),
)

MODULE_DIR = Path(__file__).parent


def _resolve(path):
    """Resolve a path relative to this module's directory, not the CWD.

    This lets the module be imported from a notebook started elsewhere and
    still find its data files.
    """
    path = Path(path)
    return path if path.is_absolute() else MODULE_DIR / path


@dataclass(frozen=True)
class GoesCloudsConfig:
    """Convenience bundle of parameters for callers who want to vary several
    at once. Not required: every function also takes plain keyword arguments
    with the same defaults as this class.
    """

    data_dir: str = DEFAULT_DATA_DIR
    band: int = DEFAULT_BAND
    window_size: int = DEFAULT_WINDOW_SIZE
    inventory_cache: str = DEFAULT_INVENTORY_CACHE
    samples_per_level: int = SAMPLES_PER_CLOUD_LEVEL
    year_range: tuple = SAMPLE_YEAR_RANGE
    seed: int = SAMPLE_RNG_SEED


DEFAULT_CONFIG = GoesCloudsConfig()


class DownloadNotAllowedError(RuntimeError):
    """Raised when a network fetch would be required but is not allowed.

    Either pass ``allow_download=True`` explicitly, or set the module-level
    ``goesclouds.ALLOW_DOWNLOAD = True`` kill switch.
    """


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def configure_notebook_logging(level=logging.INFO):
    """Attach a StreamHandler for interactive use from a notebook.

    Library code must not configure logging on its own (no handler is
    attached at import time); this is the one place a handler is added, and
    it is meant to be called explicitly from notebook cells.
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _window_slice(window_size, center=CUTOUT_CENTER_PIX):
    """Return the pixel slice for a square window of ``window_size`` centered
    on ``center``.

    ``window_size=4`` reproduces the notebook's historical (off-by-one) 4x4
    window exactly: ``slice(126, 130)``. ``window_size=5`` gives the
    symmetric 5x5 window the original docstring intended: ``slice(126, 131)``.
    Changing the window changes every downstream
    eighths estimate, so callers must opt in explicitly.
    """
    lo = center - window_size // 2
    return slice(lo, lo + window_size)


# ---------------------------------------------------------------------------
# Layer 1: LOAD
# ---------------------------------------------------------------------------


def quarter_directory(year, month, sday, quarter, data_dir=DEFAULT_DATA_DIR):
    """Path to the on-disk directory holding one night-quarter's images.

    The single source of truth for the
    ``data/satellite_{year}-{month:02d}-{sday:02d}_q{quarter}`` naming
    convention -- ``sday`` (the night's start date), never ``eday``.
    """
    return _resolve(data_dir) / f"satellite_{year}-{month:02d}-{sday:02d}_q{quarter}"


def _mjd_to_iso(mjd_array):
    """Vectorized MJD -> 19-char ISO-8601 UTC string conversion.

    Replaces the notebook's ``Time(mjd, ...).iso[:19]`` applied per-row
    (slow over 69652 rows) with a single vectorized call.
    """
    return Time(np.asarray(mjd_array, dtype=float), format="mjd", scale="utc").iso.astype(
        "<U19"
    )


def load_quarter_reports(
    reports_path=DEFAULT_REPORTS_PATH,
    night_events_path=DEFAULT_NIGHT_EVENTS,
):
    """Build a table of human cloud reports with start/end times per quarter.

    Combines two pre-built tables:

    * ``night_events_path`` (default ``night_events.h5``): for each night,
      the modified Julian date (MJD) of the *center* of each quarter (1-4)
      and the duration of a quarter.
    * ``reports_path`` (default ``clouds_ctio_blanco.h5``): the human-judged
      cloud level (in eighths, 0-8) reported for each quarter of each night.

    Returns
    -------
    pandas.DataFrame
        ``quarter_reports``, indexed by (year, month, sday, quarter), with
        columns ``clouds``, ``date``, ``eday``, ``source``, ``start_iso``,
        ``end_iso``. ``sday`` is the night's start day.
    """
    wide_night_events = pd.read_hdf(_resolve(night_events_path))

    night_events = pd.wide_to_long(
        wide_night_events.rename(
            columns={"q1_mjd": "q1", "q2_mjd": "q2", "q3_mjd": "q3", "q4_mjd": "q4"}
        )[["night", "quarter_duration", "q1", "q2", "q3", "q4"]].reset_index(),
        stubnames=["q"],
        i=["year", "month", "sday"],
        j="quarter",
    ).rename(columns={"q": "center_mjd"})

    night_events["start_mjd"] = (
        night_events["center_mjd"] - night_events["quarter_duration"] / 2
    )
    night_events["end_mjd"] = (
        night_events["center_mjd"] + night_events["quarter_duration"] / 2
    )
    night_events = night_events.reset_index().set_index(
        ["year", "month", "sday", "quarter"]
    )
    night_events["start_iso"] = _mjd_to_iso(night_events["start_mjd"].to_numpy())
    night_events["end_iso"] = _mjd_to_iso(night_events["end_mjd"].to_numpy())

    quarter_reports = (
        pd.read_hdf(_resolve(reports_path))
        .reset_index(drop=True)
        .set_index(["year", "month", "sday", "quarter"])
    )

    # Assignment (not .join) mirrors the notebook exactly: it aligns on the
    # shared MultiIndex, leaving NaN for any quarter_reports row absent from
    # night_events.
    quarter_reports["start_iso"] = night_events["start_iso"]
    quarter_reports["end_iso"] = night_events["end_iso"]

    return quarter_reports


def select_sample_quarters(
    quarter_reports,
    samples_per_level=SAMPLES_PER_CLOUD_LEVEL,
    year_range=SAMPLE_YEAR_RANGE,
    seed=SAMPLE_RNG_SEED,
):
    """Draw a stratified calibration sample, fixed count per cloud level.

    Clear nights vastly outnumber cloudy ones, so a plain random sample
    would contain very few examples of heavy cloud cover. This draws
    exactly ``samples_per_level`` quarters for each cloud level 0-8.

    The iteration order over cloud levels (ascending 0->8) and the single
    shared `numpy.random.default_rng(seed)` instance are part of the
    contract: changing either changes which quarters are drawn, and the
    on-disk image cache (`clouds/data/`) only covers the quarters drawn by
    this exact procedure.

    Returns
    -------
    pandas.DataFrame
        ``samples_per_level * 9`` rows (891 by default), same columns as
        ``quarter_reports``, sorted by index.
    """
    rng = np.random.default_rng(seed)
    start_year, end_year = year_range

    level_samples = []
    for cloud_level in CLOUD_LEVELS:
        level_quarters = quarter_reports.query(
            f"(clouds == {cloud_level}) and ({start_year} <= year <= {end_year})"
        ).sample(samples_per_level, random_state=rng)
        level_samples.append(level_quarters)

    return pd.concat(level_samples).sort_index()


def select_missing_quarters(
    quarter_reports, year=2015, sentinel=MISSING_CLOUDS_SENTINEL
):
    """Select quarters whose human cloud report is missing (sentinel value).

    Parameters
    ----------
    quarter_reports : pandas.DataFrame
        Table produced by `load_quarter_reports`.
    year : int or None
        Restrict to this year; ``None`` means all years.
    sentinel : int
        The missing-data sentinel value in the ``clouds`` column.

    Returns
    -------
    pandas.DataFrame
        Rows of ``quarter_reports`` where ``clouds == sentinel`` (and
        ``year == year`` if given).
    """
    mask = quarter_reports["clouds"] == sentinel
    if year is not None:
        mask &= quarter_reports.index.get_level_values("year") == year
    return quarter_reports[mask]


def _ctio_lat_lon_for_sds():
    """(lat, lon) rounded to whole degrees, as SDS/mcfetch expect.

    Both services want whole-degree lat/lon in degrees *west* for
    longitude, whereas astropy reports degrees *east*, so the longitude is
    negated here. For CTIO this is ``(-30, 71)``.

    Calls `EarthLocation.of_site("CTIO")` directly -- astropy memoizes the
    site registry itself (network-on-first-use, then a process- and
    disk-level cache), and this function is never called at import time.
    """
    location = EarthLocation.of_site("CTIO")
    lat = int(np.round(location.lat.deg))
    lon = int(np.round(-1 * location.lon.deg))
    return lat, lon


def load_inventory_cache(cache_path=DEFAULT_INVENTORY_CACHE):
    """Inflate the gzipped inventory cache to disk if not already present.

    Note: unlike other Layer 1 functions, this does not return a
    DataFrame -- it only ensures ``cache_path`` exists on disk, ready for
    `query_sds_inventory` to read from. Gzip compresses far better than
    HDF5's own compression, which is why only the ``.gz`` is committed to
    git. Called lazily by inventory lookups, never at import time.
    """
    cache_path = _resolve(cache_path)
    gz_path = Path(str(cache_path) + ".gz")
    if not cache_path.exists() and gz_path.exists():
        with gzip.open(gz_path, "rb") as compressed_file:
            with open(cache_path, "wb") as plain_file:
                plain_file.write(compressed_file.read())


_INVENTORY_URL_BASE = "https://inventory.ssec.wisc.edu/inventory/assets/python/query.py"
_CACHE_KEY_UNSAFE_CHARS = re.compile("[" + re.escape(":/?=.- &%") + "]")


def _inventory_url(start_time, end_time):
    lat, lon = _ctio_lat_lon_for_sds()
    encoded_start = urllib.parse.quote(start_time)
    encoded_end = urllib.parse.quote(end_time)
    return (
        f"{_INVENTORY_URL_BASE}?start_time={encoded_start}&end_time={encoded_end}"
        f"&satellite={SATELLITE_NAME}&advanced_lat={lat}&advanced_lon={lon}"
        "&output=csv"
    )


def _inventory_cache_key(inventory_url):
    """Sanitize a query URL into a filesystem/HDF5-safe cache key.

    Keep this derivation *exactly* as in the notebook -- the 186 keys
    already committed in sds_inventories.h5.gz depend on it. See T24.
    """
    return _CACHE_KEY_UNSAFE_CHARS.sub("_", inventory_url)


def query_sds_inventory(
    start_time,
    end_time,
    cache_path=DEFAULT_INVENTORY_CACHE,
    allow_download=None,
    force_query=False,
):
    """Look up what GOES-13 images exist near CTIO in a UTC time range.

    Cache-first: results are keyed on a sanitized version of the full query
    URL (see `_inventory_cache_key`) and stored in an HDF5 file. A network
    query only happens if the result is not already cached (or
    ``force_query=True``) *and* downloads are allowed --
    ``force_query=True`` does not bypass the download guard.

    Parameters
    ----------
    start_time, end_time : str
        UTC timestamps, e.g. ``"2015-08-15 00:00:00"``.
    cache_path : str or None
        HDF5 file used to cache results. If None, always query (and do not
        cache the result).
    allow_download : bool or None
        Whether a network query is permitted on a cache miss. ``None``
        (default) falls back to the module-level `ALLOW_DOWNLOAD` switch.
    force_query : bool
        If True, ignore any cached result and re-query the service (still
        subject to ``allow_download``).

    Returns
    -------
    pandas.DataFrame
        One row per matching image, as returned by the SDS inventory
        service (columns include at least ``tstamp``, ``coverage``,
        ``schedule``).

    Raises
    ------
    DownloadNotAllowedError
        If a network query would be required but is not allowed.
    """
    if allow_download is None:
        allow_download = ALLOW_DOWNLOAD

    inventory_url = _inventory_url(start_time, end_time)
    cache_key = _inventory_cache_key(inventory_url)

    resolved_cache_path = _resolve(cache_path) if cache_path is not None else None
    need_query = force_query or (resolved_cache_path is None)

    if not need_query:
        load_inventory_cache(cache_path)
        try:
            inventory = pd.read_hdf(resolved_cache_path, cache_key)
            logger.debug(
                f"Read cached inventory for {start_time} to {end_time} "
                f"from {resolved_cache_path} (key={cache_key})"
            )
            return inventory
        except (FileNotFoundError, KeyError):
            need_query = True

    if not allow_download:
        raise DownloadNotAllowedError(
            f"No cached inventory for {start_time} to {end_time} "
            f"(key={cache_key}), and downloads are not allowed. "
            "Pass allow_download=True or set goesclouds.ALLOW_DOWNLOAD = True."
        )

    logger.debug(f"Querying inventory for {start_time} to {end_time}")
    with urlopen(inventory_url) as response:
        inventory = pd.read_csv(response)
    logger.info(
        f"Complete query of inventory for {start_time} to {end_time} "
        f"from {inventory_url}"
    )
    if resolved_cache_path is not None:
        inventory.to_hdf(resolved_cache_path, key=cache_key)

    return inventory


_TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{6})(?=\.nc$)")


def extract_band_samples(
    quarters,
    band=DEFAULT_BAND,
    window_size=DEFAULT_WINDOW_SIZE,
    data_dir=DEFAULT_DATA_DIR,
    time_as_datetime=True,
    max_timestamp_offset_days=2,
):
    """Read the pixels nearest CTIO out of every downloaded image in a band.

    The only function in this module that reads the image cache. For each
    quarter in ``quarters``, globs ``{quarter_dir}/GOES13_{band}_*.nc``, reads
    the center window from each readable, in-window file, and emits one row
    per pixel per image.

    Parameters
    ----------
    quarters : pandas.DataFrame
        Indexed by (year, month, sday, quarter); must have a ``clouds``
        column, and, unless ``max_timestamp_offset_days`` is None,
        ``start_iso``/``end_iso`` columns (as produced by
        `load_quarter_reports`).
    band : int
        The GOES imager band to extract.
    window_size : int
        Side length of the square pixel window; see `_window_slice`.
    data_dir : str
        Root directory containing the per-quarter image subdirectories.
    time_as_datetime : bool
        If False, the ``time`` index level is cast to int64 microseconds
        since epoch instead of kept as ``datetime64`` -- this reproduces the
        historical (buggy) behavior of the notebook's blanket
        ``.astype(int)`` for bit-exact comparison.
    max_timestamp_offset_days : float or None
        Files whose filename timestamp falls further than this many days
        outside the quarter's ``[start_iso, end_iso]`` window are skipped
        with a warning (removes stray mis-filed images).
        None disables the check, exactly reproducing the notebook's
        behavior of trusting every file found in a quarter's directory.

    Returns
    -------
    pandas.DataFrame
        Indexed by (year, month, sday, quarter, time, pixel), with columns
        ``clouds`` and ``band{band}`` (one row per extracted pixel, per
        image). Quarters with no matching files contribute no rows.
    """
    pix_slice = _window_slice(window_size)
    offset = (
        pd.Timedelta(days=max_timestamp_offset_days)
        if max_timestamp_offset_days is not None
        else None
    )

    value_column = f"band{band}"
    columns = {
        "year": [],
        "month": [],
        "sday": [],
        "quarter": [],
        "time": [],
        "pixel": [],
        "clouds": [],
        value_column: [],
    }

    for _, quarter_row in quarters.reset_index().iterrows():
        dirname = quarter_directory(
            quarter_row.year,
            quarter_row.month,
            quarter_row.sday,
            quarter_row.quarter,
            data_dir=data_dir,
        )

        if offset is not None:
            start_time = pd.Timestamp(quarter_row.start_iso)
            end_time = pd.Timestamp(quarter_row.end_iso)

        fnames = sorted(
            glob.glob(str(dirname / f"{SATELLITE_NAME_MCFETCH}_{band}_*.nc"))
        )
        for fname in fnames:
            match = _TIMESTAMP_PATTERN.search(fname)
            if match is None:
                logger.warning(f"could not parse timestamp from {fname}")
                continue
            image_time = pd.Timestamp(match.group(1))

            if offset is not None and not (
                start_time - offset <= image_time <= end_time + offset
            ):
                logger.warning(
                    f"skipping {fname}: timestamp {image_time} is more than "
                    f"{max_timestamp_offset_days} day(s) outside the quarter "
                    f"window [{start_time}, {end_time}]"
                )
                continue

            try:
                netcdf_data = scipy.io.netcdf_file(fname, "r", mmap=False)
            except (OSError, TypeError, ValueError):
                logger.warning(f"could not read {fname}")
                continue

            try:
                image = netcdf_data.variables["data"].data[0]
                for dimension in (0, 1):
                    assert image.shape[dimension] == 2 * CUTOUT_CENTER_PIX
                pixel_window = image[pix_slice, pix_slice]
                values = pixel_window.flatten().astype(float)
            finally:
                netcdf_data.close()

            n_pixels = len(values)
            columns["year"].extend([quarter_row.year] * n_pixels)
            columns["month"].extend([quarter_row.month] * n_pixels)
            columns["sday"].extend([quarter_row.sday] * n_pixels)
            columns["quarter"].extend([quarter_row.quarter] * n_pixels)
            columns["time"].extend([image_time] * n_pixels)
            columns["pixel"].extend(range(n_pixels))
            columns["clouds"].extend([quarter_row.clouds] * n_pixels)
            columns[value_column].extend(values)

    result = pd.DataFrame(
        {
            "year": pd.array(columns["year"], dtype="int64"),
            "month": pd.array(columns["month"], dtype="int64"),
            "sday": pd.array(columns["sday"], dtype="int64"),
            "quarter": pd.array(columns["quarter"], dtype="int64"),
            "time": pd.to_datetime(columns["time"]),
            "pixel": pd.array(columns["pixel"], dtype="int64"),
            "clouds": pd.array(columns["clouds"], dtype="int64"),
            value_column: pd.array(columns[value_column], dtype="float64"),
        }
    )

    if not time_as_datetime:
        result["time"] = result["time"].astype("int64")

    return result.set_index(["year", "month", "sday", "quarter", "time", "pixel"])


def load_multiband_samples(
    quarters,
    bands=ALL_BANDS,
    window_size=DEFAULT_WINDOW_SIZE,
    data_dir=DEFAULT_DATA_DIR,
):
    """Extract and join per-band pixel samples into one multiband frame.

    Calls `extract_band_samples` once per band in ``bands`` and combines the
    results, aligned on the full (year, month, sday, quarter, time, pixel)
    index. Because band-2 timestamps are a strict subset of band-4's.
    This correctly leaves NaN wherever a band's image is
    absent for a given timestamp, rather than dropping or duplicating rows
    -- replaces the notebook's fragile
    ``pd.concat(...); loc[:, ~columns.duplicated()]`` idiom.

    Returns
    -------
    pandas.DataFrame
        ``multiband_samples``, indexed by
        (pixel, year, month, sday, quarter, time) -- ``pixel`` first, so
        ``multiband_samples.loc[pixel_id]`` selects one pixel's samples
        across all bands. MultiIndex columns: ``("human", "clouds")`` and
        ``(band, "value")`` for each band in ``bands``.
    """
    band_frames = {
        band: extract_band_samples(
            quarters, band=band, window_size=window_size, data_dir=data_dir
        )
        for band in bands
    }

    clouds = None
    value_columns = {}
    for band, frame in band_frames.items():
        clouds = frame["clouds"] if clouds is None else clouds.combine_first(frame["clouds"])
        value_columns[(band, "value")] = frame[f"band{band}"]

    multiband = pd.DataFrame(value_columns)
    multiband[("human", "clouds")] = clouds
    multiband = multiband[[("human", "clouds")] + [(band, "value") for band in bands]]

    index_names = multiband.index.names
    pixel_first_order = ["pixel"] + [name for name in index_names if name != "pixel"]
    return multiband.reorder_levels(pixel_first_order, axis=0)


# ---------------------------------------------------------------------------
# Layer 2: COMPUTE
# ---------------------------------------------------------------------------


def _quarter_levels(index_names):
    """Index level names identifying a quarter (drops ``time``/``pixel``)."""
    return [name for name in index_names if name not in ("time", "pixel")]


def compute_sample_stats(band_samples, value_column=None):
    """Collapse per-pixel samples down to one row of statistics per quarter.

    Groups by quarter (every index level except ``time``/``pixel``) plus the
    ``clouds`` column, so each (quarter, human-cloud-level) combination
    contributes one row of ``describe()`` statistics, plus derived ``range``
    and ``IQR`` columns.

    Parameters
    ----------
    band_samples : pandas.DataFrame
        As returned by `extract_band_samples` or `as_value_frame`: indexed by
        quarter levels plus ``time``/``pixel``, with a ``clouds`` column and
        exactly one other (numeric) column.
    value_column : str or None
        Name of the column to summarize. ``None`` means "the single
        non-``clouds`` column", which lets this function work unchanged on
        both ``band{N}`` frames and a ``value`` frame from `as_value_frame`.

    Returns
    -------
    pandas.DataFrame
        Indexed by the quarter levels, with a ``clouds`` column restored and
        statistics columns (``count``, ``mean``, ``std``, ``min``, ``25%``,
        ``50%``, ``75%``, ``max``, ``range``, ``IQR``).
    """
    if value_column is None:
        (value_column,) = [c for c in band_samples.columns if c != "clouds"]

    group_levels = _quarter_levels(band_samples.index.names)
    stats = (
        band_samples.groupby(group_levels + ["clouds"])[[value_column]]
        .describe()
        .droplevel(0, axis=1)
        .reset_index(level="clouds")
    )
    stats["range"] = stats["max"] - stats["min"]
    stats["IQR"] = stats["75%"] - stats["25%"]
    return stats


def fraction_greater(reference_value, values):
    """Fraction of `values` that are strictly greater than `reference_value`.

    Used as a `functools.partial`-bound groupby aggregator in
    `compute_by_quarter`.
    """
    return (values > reference_value).mean()


def reference_quantile(samples, value_column, reference_level=4):
    """The `reference_level / 8` quantile of `value_column` at that cloud level.

    At a quarter reported as `reference_level` eighths cloudy, that fraction
    of pixels should read as "cloudy" (dark) and the rest as "clear"
    (bright), so the `reference_level / 8` quantile of brightness among
    samples at that level is the brightness boundary between them. The
    default `reference_level=4` is the halfway case, so this is the median
    -- the physically-motivated "50% cloudy" threshold this module was
    originally built around.

    The returned value is a plain calibration constant -- callers are
    responsible for passing it back into `compute_by_quarter`, rather than
    it being stashed as module state.
    """
    at_level = samples.loc[samples["clouds"] == reference_level, value_column]
    if at_level.empty:
        raise ValueError(
            f"no samples at cloud level {reference_level} to compute a "
            "reference quantile from"
        )
    return int(at_level.quantile(reference_level / 8))


def estimate_eighths(fraction_above_reference):
    """Map a fraction-above-reference-brightness to estimated cloud eighths.

    ``round((1 - fraction) * 8)``: more pixels above the reference
    brightness means *fewer* eighths of cloud (clear sky reads bright in
    these GOES-13 IR bands).
    """
    fraction_above_reference = np.asarray(fraction_above_reference, dtype=float)
    return np.round((1 - fraction_above_reference) * 8).astype(int)


def optimize_cloudcut_fraction(band_samples, value_column, obs_cut=None):
    """Find the brightness threshold that matches the human non-observable fraction.

    Returns the brightness threshold for which the fraction of quarters
    estimated as non-observable (``estimated_eighths > obs_cut``) is closest
    to the fraction of quarters human-judged as non-observable
    (``clouds > obs_cut``).

    Starts at `reference_quantile` and walks through the sorted unique
    brightness values in whichever direction reduces the error, stopping as
    soon as the error stops improving. This is efficient because raising the
    threshold increases ``sat_nonobs_frac`` monotonically (more pixels read
    as cloudy), so the error surface is unimodal and a linear scan from the
    starting point suffices.

    Complements `reference_quantile`: where that function picks the threshold
    from a physical model (the level-``reference_level`` quantile), this one
    picks it empirically by matching observable rates across the calibration
    sample. The returned value can be passed directly to `compute_by_quarter`.

    Parameters
    ----------
    band_samples : pandas.DataFrame
        As returned by `extract_band_samples` or `as_value_frame`: indexed by
        quarter levels plus ``time``/``pixel``, with a ``clouds`` column and
        a numeric brightness column named ``value_column``.
    value_column : str
        Name of the brightness column to threshold.
    obs_cut : float or None
        Threshold (in eighths) for the binary observable/non-observable split.
        ``None`` defaults to `OBSERVABLE_CUT`.

    Returns
    -------
    int
        The brightness threshold minimising
        ``|fraction_estimated_nonobservable − fraction_human_nonobservable|``.
    """
    obs_cut = OBSERVABLE_CUT if obs_cut is None else obs_cut
    candidates = np.sort(band_samples[value_column].dropna().unique())

    start_idx = int(np.clip(
        np.searchsorted(candidates, reference_quantile(band_samples, value_column)),
        0, len(candidates) - 1,
    ))

    # compute_by_quarter carries human clouds through unchanged; the starting
    # call also gives us the per-quarter cloud levels for every subsequent call.
    ref = compute_by_quarter(band_samples, value_column, int(candidates[start_idx]))
    valid = ref["clouds"] != MISSING_CLOUDS_SENTINEL
    human_nonobs_frac = (ref.loc[valid, "clouds"] > obs_cut).mean()

    def _compute_satellite_frac(idx):
        bq = compute_by_quarter(band_samples, value_column, int(candidates[idx]))
        return (bq.loc[valid, "estimated_eighths"] > obs_cut).mean()

    satellite_frac = _compute_satellite_frac(start_idx)
    best_diff = abs(satellite_frac - human_nonobs_frac)
    best_cloudcut = int(candidates[start_idx])
    direction = 1 if satellite_frac < human_nonobs_frac else -1

    idx = start_idx + direction
    while 0 <= idx < len(candidates):
        satellite_frac = _compute_satellite_frac(idx)
        diff = abs(satellite_frac - human_nonobs_frac)
        if diff >= best_diff:
            break
        best_diff = diff
        best_cloudcut = int(candidates[idx])
        idx += direction

    return best_cloudcut


def optimize_cloudcut_obs(band_samples, value_column, obs_cut=None, candidates=None):
    """Find the brightness threshold that maximises per-quarter observability agreement.

    Returns the cloudcut for which the number of quarters where the
    satellite-estimated observability (``estimated_eighths <= obs_cut``)
    agrees with the human-judged observability (``clouds <= obs_cut``) is
    greatest. Ties are broken by taking the lowest threshold found.

    Unlike `optimize_cloudcut_fraction`, which matches aggregate observable
    *rates*, this function maximises the per-quarter binary agreement count
    (``n_match_obs`` in `compare_estimates`). Because that count is not
    guaranteed to be monotonic in the threshold -- each crossing flips one
    quarter's estimate, gaining or losing a match depending on that quarter's
    human rating -- all unique brightness values are evaluated.

    Parameters
    ----------
    band_samples : pandas.DataFrame
        As returned by `extract_band_samples` or `as_value_frame`.
    value_column : str
        Name of the brightness column to threshold.
    obs_cut : float or None
        Threshold (in eighths) for the binary observable/non-observable split.
        ``None`` defaults to `OBSERVABLE_CUT`.
    candidates : np.ndarray
        Candidate cloudcuts to test

    Returns
    -------
    int
        The brightness threshold maximising per-quarter observability matches.
    """
    obs_cut = OBSERVABLE_CUT if obs_cut is None else obs_cut
    if candidates is None:
        candidates = np.sort(band_samples[value_column].dropna().unique())

    ref = compute_by_quarter(band_samples, value_column, int(candidates[0]))
    valid = ref["clouds"] != MISSING_CLOUDS_SENTINEL
    human_observable = ref.loc[valid, "clouds"] <= obs_cut

    best_cloudcut = int(candidates[0])
    best_matches = -1
    for n, cloudcut in enumerate(candidates):
        bq = compute_by_quarter(band_samples, value_column, int(cloudcut))
        satellite_observable = bq.loc[valid, "estimated_eighths"] <= obs_cut
        matches = (human_observable == satellite_observable).sum()
        if matches > best_matches:
            best_matches = matches
            best_cloudcut = int(cloudcut)
            # print(f"{n}/{len(candidates)}: {best_cloudcut} ({best_matches} matches)")

    return best_cloudcut


def optimize_cloudcut_eighths(band_samples, value_column, candidates=None):
    """Find the brightness threshold that minimises RMS error in cloud eighths.

    Returns the cloudcut for which the root-mean-square difference between
    satellite-estimated eighths and human-judged eighths (excluding sentinel
    quarters) is smallest. All unique brightness values are evaluated because
    the RMS is not guaranteed to be monotonic: as the threshold rises,
    estimated eighths increase quarter by quarter, each step improving the
    fit where the human estimate is high and worsening it where it is low.

    The returned value can be passed directly to `compute_by_quarter`.

    Parameters
    ----------
    band_samples : pandas.DataFrame
        As returned by `extract_band_samples` or `as_value_frame`.
    value_column : str
        Name of the brightness column to threshold.
    candidates : np.ndarray
        Candidate cloudcuts to test
    
    Returns
    -------
    int
        The brightness threshold minimising the per-quarter RMS eighths error.
    """
    if candidates is None:
        candidates = np.sort(band_samples[value_column].dropna().unique())

    ref = compute_by_quarter(band_samples, value_column, int(candidates[0]))
    valid = ref["clouds"] != MISSING_CLOUDS_SENTINEL
    human_eighths = ref.loc[valid, "clouds"].to_numpy(dtype=float)

    best_cloudcut = int(candidates[0])
    best_rms = float("inf")
    for cloudcut in candidates:
        bq = compute_by_quarter(band_samples, value_column, int(cloudcut))
        satellite_eighths = bq.loc[valid, "estimated_eighths"].to_numpy(dtype=float)
        rms = np.sqrt(np.mean((human_eighths - satellite_eighths) ** 2))
        if rms < best_rms:
            best_rms = rms
            best_cloudcut = int(cloudcut)

    return best_cloudcut


def compute_by_quarter(band_samples, value_column, reference_median_value):
    """Estimate cloud eighths per quarter from per-pixel brightness samples.

    For each quarter, computes the fraction of pixels brighter than
    `reference_median_value` and converts that fraction to estimated cloud
    eighths via `estimate_eighths`. Does not mutate `band_samples`.

    Parameters
    ----------
    band_samples : pandas.DataFrame
        As returned by `extract_band_samples` or `as_value_frame`.
    value_column : str
        Name of the brightness column to compare against the reference.
    reference_median_value : int
        The "50% cloudy" brightness threshold, e.g. from `reference_quantile`.

    Returns
    -------
    pandas.DataFrame
        Indexed by the quarter levels, with columns ``clouds``,
        ``fraction_above_cloudcut``, ``estimated_eighths``.
    """
    group_levels = _quarter_levels(band_samples.index.names)
    working = pd.DataFrame(
        {
            "clouds": band_samples["clouds"],
            "_minus_reference": band_samples[value_column] - reference_median_value,
        }
    )

    by_quarter = (
        working.groupby(group_levels + ["clouds"])["_minus_reference"]
        .agg(functools.partial(fraction_greater, 0))
        .rename("fraction_above_cloudcut")
        .reset_index()
    )
    by_quarter["estimated_eighths"] = estimate_eighths(by_quarter["fraction_above_cloudcut"])
    return by_quarter.set_index(group_levels)


def fit_clear_sky_model(
    multiband_samples,
    x_band=CORRECTION_BAND,
    y_band=DEFAULT_BAND,
    per_pixel=True,
    quantile=0.5,
    estimator=None,
):
    """Fit a robust linear model of ``y_band`` brightness against ``x_band``.

    Fits on clear (``clouds == 0``), complete (both bands finite) samples
    only. With ``per_pixel=True`` (the default), fits one model per pixel,
    since different pixels can have different clear-sky brightness; with
    ``per_pixel=False`` fits a single model pooling all pixels together.

    A group with fewer than `MIN_CLEAR_SAMPLES_FOR_FIT` clear/complete
    samples is degenerate: it is logged and stored as ``None`` rather than
    attempted, and any other fit failure (``ValueError``/``LinAlgError``) is
    caught, logged, and also stored as ``None``.

    Parameters
    ----------
    multiband_samples : pandas.DataFrame
        As returned by `load_multiband_samples`: indexed by (pixel, ...),
        with MultiIndex columns ``("human", "clouds")`` and
        ``(band, "value")`` for each band.
    x_band, y_band : int
        Predictor and target bands.
    per_pixel : bool
        Fit one model per pixel (keyed by pixel id) if True, else one model
        pooling all pixels (keyed by ``None``).
    quantile : float
        Passed to the default per-pixel estimator, `QuantileRegressor`.
    estimator : sklearn regressor or None
        An unfitted, duck-typed sklearn regressor to clone and fit for each
        group. ``None`` means `QuantileRegressor(quantile=quantile)` for
        ``per_pixel=True`` or `HuberRegressor(epsilon=1.0)` for
        ``per_pixel=False`` -- matching what the notebook actually ran.

    Returns
    -------
    dict
        Maps pixel id (or ``None`` if ``per_pixel=False``) to a fitted
        regressor, or ``None`` for a degenerate group.
    """
    x_col = (x_band, "value")
    y_col = (y_band, "value")
    clouds_col = ("human", "clouds")

    clear_complete = (
        (multiband_samples[clouds_col] == 0)
        & np.isfinite(multiband_samples[x_col])
        & np.isfinite(multiband_samples[y_col])
    )

    if per_pixel:
        pixel_level = multiband_samples.index.get_level_values("pixel")
        groups = {
            pixel_id: clear_complete & (pixel_level == pixel_id)
            for pixel_id in pixel_level.unique()
        }
    else:
        groups = {None: clear_complete}

    def new_estimator():
        if estimator is not None:
            return clone(estimator)
        return QuantileRegressor(quantile=quantile) if per_pixel else HuberRegressor(epsilon=1.0)

    models = {}
    for key, mask in groups.items():
        if mask.sum() < MIN_CLEAR_SAMPLES_FOR_FIT:
            logger.warning(
                f"fit_clear_sky_model: fewer than {MIN_CLEAR_SAMPLES_FOR_FIT} "
                f"clear samples for group {key!r}; skipping fit"
            )
            models[key] = None
            continue
        try:
            model = new_estimator()
            model.fit(multiband_samples.loc[mask, [x_col]], multiband_samples.loc[mask, y_col])
            models[key] = model
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.warning(f"fit_clear_sky_model: degenerate fit for group {key!r}: {err}")
            models[key] = None

    return models


def apply_clear_sky_correction(samples, models, x_band=CORRECTION_BAND, y_band=DEFAULT_BAND):
    """Correct ``y_band`` brightness for its clear-sky relationship to ``x_band``.

    Predicts ``y_band`` from ``x_band`` using the per-group models from
    `fit_clear_sky_model` (keyed by pixel id, or by ``None`` for a single
    pooled model) and subtracts the prediction from the observed value.
    Returns a copy; does not mutate `samples`.
    Replaces the notebook's two divergent per-pixel/aggregate correction
    loops with one function.

    Parameters
    ----------
    samples : pandas.DataFrame
        A multiband-shaped frame (from `load_multiband_samples`, or a
        missing-quarter frame with the same column shape).
    models : dict
        As returned by `fit_clear_sky_model`. A pixel (or ``None``) whose
        model is ``None`` gets ``NaN`` predictions and correction.
    x_band, y_band : int
        Must match the bands `models` was fit with.

    Returns
    -------
    pandas.DataFrame
        `samples` plus two columns: ``(x_band, "pred{y_band}")`` (the
        clear-sky prediction) and ``("4corr", "value")`` (observed ``y_band``
        minus the prediction).
    """
    x_col = (x_band, "value")
    y_col = (y_band, "value")
    pred_col = (x_band, f"pred{y_band}")

    corrected = samples.copy()
    corrected[pred_col] = np.nan

    if None in models:
        pixel_level = None
    else:
        pixel_level = corrected.index.get_level_values("pixel")

    for key, model in models.items():
        if model is None:
            continue
        mask = np.isfinite(corrected[x_col])
        if pixel_level is not None:
            mask = mask & (pixel_level == key)
        if not mask.any():
            continue
        corrected.loc[mask, pred_col] = model.predict(corrected.loc[mask, [x_col]])

    corrected[("4corr", "value")] = corrected[y_col] - corrected[pred_col]
    return corrected


def as_value_frame(samples, column_key, clouds_column=("human", "clouds")):
    """Flatten one value column of a MultiIndex-column frame to ``clouds``/``value``.

    Used to convert a `load_multiband_samples`/`apply_clear_sky_correction`
    frame into the flat two-column shape that `compute_sample_stats`,
    `reference_quantile`, and `compute_by_quarter` expect. Keeps the full
    index (including ``pixel``) unchanged; drops rows where the selected
    value column is NaN.

    Parameters
    ----------
    samples : pandas.DataFrame
        A MultiIndex-column frame, e.g. ``multiband_samples`` or the output
        of `apply_clear_sky_correction`.
    column_key : tuple
        The MultiIndex column to use as ``value``, e.g. ``("4corr", "value")``
        or ``(4, "value")``.
    clouds_column : tuple
        The MultiIndex column to use as ``clouds``.

    Returns
    -------
    pandas.DataFrame
        Flat columns ``clouds`` and ``value``, same index as `samples` minus
        rows where ``value`` is NaN.
    """
    result = pd.DataFrame(
        {"clouds": samples[clouds_column], "value": samples[column_key]}
    )
    return result.dropna(subset=["value"])


def compare_estimates(
    samples_by_quarter,
    estimators,
    human_column=("human", "clouds"),
    observable_cut=OBSERVABLE_CUT,
):
    """Build the ``match_stats`` table comparing estimators to human judgment.

    The notebook builds this table with two divergent code paths: for the
    raw per-band cuts, both the human judgment and the estimate live as
    columns of one shared MultiIndex-column frame (`samples_by_quarter`);
    for the clear-sky-corrected estimate, both live as flat columns of an
    entirely separate `compute_by_quarter` output. This function collapses
    both into one code path by accepting either shape for each entry in
    `estimators`:

    - a `compute_by_quarter`-shaped `pandas.DataFrame` (flat ``clouds`` and
      ``estimated_eighths`` columns) -- used self-contained, ignoring
      `samples_by_quarter`;
    - a bare `pandas.Series` of estimated eighths, or a column key into
      `samples_by_quarter` -- paired with `samples_by_quarter[human_column]`
      as the human judgment.

    Sentinel-9 and NaN rows (in either the human or estimated column) are
    excluded before scoring, via an inner join followed by a sentinel filter.

    Parameters
    ----------
    samples_by_quarter : pandas.DataFrame
        Provides the human judgment (and index alignment) for any
        `estimators` entry that is not already a self-contained
        `compute_by_quarter` frame.
    estimators : dict
        Maps estimator name (e.g. ``"cut on 4"``) to one of the three shapes
        described above.
    human_column : tuple or str
        Column key for the human judgment within `samples_by_quarter`.
    observable_cut : float
        Threshold (in eighths) for the binary "is it observable" question;
        `n_match_obs` counts quarters where estimate and human agree on
        which side of this cut they fall.

    Returns
    -------
    pandas.DataFrame
        Indexed by estimator name, with columns ``n``, ``n_exact``,
        ``n_match_obs``, ``human_mean_eighths``, ``estimated_mean_eighths``,
        ``frac_match_obs``. The notebook calls the mean-eighths columns
        ``human_obs_frac``/``estimated_obs_frac`` despite them not being
        fractions of anything observable; renamed here so the name matches
        the computation.
    """
    rows = {}
    for name, estimator in estimators.items():
        if isinstance(estimator, pd.DataFrame):
            human = estimator["clouds"]
            estimated = estimator["estimated_eighths"]
        elif isinstance(estimator, pd.Series):
            human = samples_by_quarter[human_column]
            estimated = estimator
        else:
            human = samples_by_quarter[human_column]
            estimated = samples_by_quarter[estimator]

        combined = pd.DataFrame({"human": human, "estimated": estimated}).dropna()
        combined = combined[combined["human"] != MISSING_CLOUDS_SENTINEL]

        n = len(combined)
        n_exact = int((combined["human"] == combined["estimated"]).sum())
        human_observable = combined["human"] <= observable_cut
        estimated_observable = combined["estimated"] <= observable_cut
        n_match_obs = int((human_observable == estimated_observable).sum())

        rows[name] = {
            "n": n,
            "n_exact": n_exact,
            "n_match_obs": n_match_obs,
            "human_mean_eighths": combined["human"].mean(),
            "estimated_mean_eighths": combined["estimated"].mean(),
            "frac_match_obs": n_match_obs / n if n else np.nan,
        }

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Layer 3: PLOT
# ---------------------------------------------------------------------------
#
# Every function here takes already-computed DataFrames, draws into an
# optional caller-supplied `fig`/`ax`, and returns a `matplotlib.figure.Figure`.
# No computation, no I/O.


def plot_sample_stats(sample_stats, stat_names=STAT_NAMES, band=None, title=None, fig=None):
    """3x3 grid of boxplots of each statistic in `sample_stats`, by `clouds`.

    `sample_stats` is a `compute_sample_stats` output (flat columns, a
    `clouds` column). Mirrors the notebook's `plot_band_sample_stats`
    (cells 20, 36, 52, 76).
    """
    if fig is None:
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    else:
        axes = np.array(fig.axes)
    for ax, stat_name in zip(axes.flatten(), stat_names):
        sample_stats.boxplot(stat_name, by="clouds", color="blue", ax=ax)
    if title is None:
        band_label = f"band {band} " if band is not None else ""
        title = f"GOES-13 {band_label}pixel statistics vs. human-judged cloud level"
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_by_quarter(by_quarter, band=None, fig=None):
    """Two panels: fraction-above-cutoff boxplot, and estimate-vs-human 2-D histogram.

    `by_quarter` is a `compute_by_quarter` output. Mirrors the notebook's
    `plot_band_by_quarter` (cells 24, 38, 54).
    """
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axes = np.array(fig.axes)
    band_label = f"Band-{band} brightness" if band is not None else "Brightness"

    ax = axes[0]
    by_quarter.boxplot("fraction_above_cloudcut", by="clouds", ax=ax)
    ax.set_title("Boxplot for fraction of GOES pixels by human estimate")
    ax.set_ylabel("Fraction of pixels brighter than the cloudy cutoff")
    ax.set_xlabel("Human estimated cloud fraction (eighths)")

    ax = axes[1]
    h = ax.hist2d(
        by_quarter["estimated_eighths"],
        by_quarter["clouds"],
        bins=np.arange(-0.5, 9.5, 1),
    )
    ax.set_xlabel(f"{band_label}-based estimated cloud level (eighths)")
    ax.set_ylabel("Human-judged cloud level (eighths)")
    ax.set_title(f"GOES-13 {band_label.lower()} cloud estimate vs. human judgment")
    fig.colorbar(h[3], ax=ax, label="count")

    fig.suptitle("")
    return fig


def plot_norm_by_quarter(by_quarter, fig=None):
    """Three-panel variant of `plot_by_quarter` for a clear-sky-corrected estimator.

    `by_quarter` is a `compute_by_quarter` output built from a corrected
    value column (e.g. via `as_value_frame` on `("4corr", "value")`).
    Mirrors the notebook's `plot_norm_by_quarter` (cell 79).
    """
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        axes = np.array(fig.axes)

    ax = axes[0]
    by_quarter.boxplot("fraction_above_cloudcut", by="clouds", ax=ax)
    ax.set_title("Boxplot for fraction of GOES pixels by human estimate")
    ax.set_ylabel("Fraction of pixels brighter than the cloudy cutoff")
    ax.set_xlabel("Human estimated cloud fraction (eighths)")

    ax = axes[1]
    by_quarter.boxplot("clouds", by="estimated_eighths", ax=ax)
    ax.set_title("Boxplot for fraction of GOES pixels by human estimate")
    ax.set_xlabel("Bands 2 & 4 brightness-based estimated cloud level (8ths)")
    ax.set_ylabel("Human estimated cloud fraction (eighths)")

    ax = axes[2]
    h = ax.hist2d(
        by_quarter["estimated_eighths"],
        by_quarter["clouds"],
        bins=np.arange(-0.5, 9.5, 1),
    )
    ax.set_xlabel("Bands 2 & 4 brightness-based estimated cloud level (8ths)")
    ax.set_ylabel("Human-judged cloud level (8ths)")
    ax.set_title("Bands 2 & 4 cloud vs. human judgment")
    fig.colorbar(h[3], ax=ax, label="count")

    fig.suptitle("")
    return fig


def plot_band_comparison(
    sample_stats,
    stat_name,
    ax,
    x_band=DEFAULT_BAND,
    y_band=CORRECTION_BAND,
    cmap=cc.m_kr,
    clear_color="black",
    clear_mask=None,
):
    """Scatter one statistic of `x_band` against `y_band`, colored by human clouds.

    `sample_stats` is a multi-band, MultiIndex-column frame: `(band, stat)`
    for each band plus `("human", "clouds")` (assembled by the caller from
    per-band `compute_sample_stats` outputs). If
    `clear_mask` is given, those rows are overplotted opaque in `clear_color`
    and everything else is drawn at reduced alpha.
    """
    alpha = 1.0 if clear_mask is None else 0.3
    sample_stats.plot.scatter(
        (x_band, stat_name), (y_band, stat_name), c=("human", "clouds"),
        cmap=cmap, alpha=alpha, ax=ax,
    )
    if clear_mask is not None:
        sample_stats.loc[clear_mask, :].plot.scatter(
            (x_band, stat_name), (y_band, stat_name), c=clear_color, ax=ax,
        )

    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Human estimated eighths cloud-cover")
    ax.set_title(stat_name)
    ax.set_xlabel(f"band {x_band}")
    ax.set_ylabel(f"band {y_band}")
    return ax.figure


def plot_clear_sky_model(
    samples, value_key, x_band=CORRECTION_BAND, y_band=DEFAULT_BAND, fig=None, cloudcut=None
):
    """Three-panel clear-sky correction diagnostic.

    `samples` must already carry the columns `apply_clear_sky_correction`
    adds for this `x_band`/`y_band` pair -- `(x_band, "pred{y_band}")` and
    `("4corr", value_key)` -- plus the observed `(y_band, value_key)` and
    `("human", "clouds")` columns. `value_key` names the observed-value
    column: `"value"` for a per-sample frame (notebook cell 72), or a
    statistic name such as `"mean"` for a per-quarter `sample_stats` frame
    fed through the same correction (cell 65).

    Panels: predicted vs. observed with the `y=x` line; residual vs.
    observed colored by human clouds; boxplot of residual by human clouds.
    """
    observed_col = (y_band, value_key)
    pred_col = (x_band, f"pred{y_band}")
    corr_col = ("4corr", value_key)

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        axes = np.array(fig.axes)

    clear_mask = samples.loc[:, ('human', 'clouds')] == 0
        
    cmap = cc.cm['isolum']
    alpha = 0.1
    ax = axes[0]
    samples.plot.scatter(observed_col, pred_col, c=("human", "clouds"), s=1, alpha=alpha, cmap=cmap, ax=ax)
    samples.loc[clear_mask, :].plot.scatter(observed_col, pred_col, c='black', s=1, alpha=alpha, ax=ax)
    ax.plot(samples[observed_col], samples[observed_col], color="red")
    ax.set_xlabel(f"band {y_band} {value_key}")
    ax.set_ylabel(f"clear prediction of band {y_band} {value_key} using band {x_band}")
    ax.collections[0].colorbar.remove()

    ax = axes[1]
    samples.plot.scatter(observed_col, corr_col, c=("human", "clouds"), s=1, alpha=alpha, cmap=cmap, ax=ax)
    samples.loc[clear_mask, :].plot.scatter(observed_col, corr_col, c='black', s=1, alpha=alpha, ax=ax)
    ax.set_xlabel(f"band {y_band} {value_key}")
    ax.set_ylabel(f"band {y_band} {value_key} minus clear prediction using band {x_band}")
    ax.axhline(y=0, color='red')
    if cloudcut is not None:
        ax.axhline(y=cloudcut, color='blue')
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Human estimated eighths cloud-cover")

    ax = axes[2]
    corr_df = pd.DataFrame({
        "4corr": samples[corr_col],
        "clouds": samples[("human", "clouds")],
    })
    corr_df.boxplot("4corr", by="clouds", flierprops={'marker': '.', 'alpha': 0.1}, whis=(5, 95), ax=ax)
    ax.set_ylabel(f"band {y_band} {value_key} minus clear prediction using band {x_band}")
    ax.set_title("")

    fig.suptitle("")
    return fig


def plot_per_pixel_models(multiband_samples, models, x_band=DEFAULT_BAND, y_band=CORRECTION_BAND, fig=None):
    """Grid of per-pixel scatter plots with each pixel's fitted clear-sky relation.

    The grid shape is derived from the actual number of pixels. `models` is a
    `fit_clear_sky_model` output keyed by pixel id; pixels with no model
    (`None`) are plotted without an overlaid fit.
    """
    pixel_level = multiband_samples.index.get_level_values("pixel")
    pixel_ids = np.sort(pixel_level.unique())
    n_pixels = len(pixel_ids)
    n_cols = int(np.ceil(np.sqrt(n_pixels)))
    n_rows = int(np.ceil(n_pixels / n_cols))

    if fig is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    else:
        axes = np.array(fig.axes).reshape(n_rows, n_cols)
    flat_axes = axes.flatten()

    clear_mask_all = multiband_samples[("human", "clouds")] == 0
    for band in (x_band, y_band):
        clear_mask_all = clear_mask_all & np.isfinite(multiband_samples[(band, "value")])

    for ax, pixel_id in zip(flat_axes, pixel_ids):
        this_pixel = pixel_level == pixel_id
        pixel_samples = multiband_samples.loc[this_pixel, :]
        pixel_clear = clear_mask_all.loc[this_pixel]

        pixel_samples.plot.scatter(
            (x_band, "value"), (y_band, "value"), c=("human", "clouds"),
            s=1, cmap=cc.cm["isolum"], alpha=0.5, ax=ax,
        )
        pixel_samples.loc[pixel_clear, :].plot.scatter(
            (x_band, "value"), (y_band, "value"), s=1, c="black", ax=ax,
        )

        model = models.get(pixel_id)
        if model is not None:
            slope = model.coef_[0]
            intercept = model.intercept_
            ys = np.array(ax.get_ylim())
            ax.plot(model.predict(ys.reshape(-1, 1)), ys, c='red')

        colorbar = ax.collections[0].colorbar
        colorbar.set_label("Human estimated eighths cloud-cover")
        ax.set_title(f"Pixel {pixel_id}")
        ax.set_xlabel(f"band {x_band}")
        ax.set_ylabel(f"band {y_band}")

    for ax in flat_axes[n_pixels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_estimate_histogram(by_quarter, column="estimated_eighths", ax=None):
    """Histogram of estimated eighths, e.g. for the missing-quarter estimate.

    Mirrors the notebook's missing-quarter histograms (cells 27, 42, 59, 86).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    by_quarter[column].hist(bins=np.arange(-0.5, 9.5, 1), ax=ax)
    ax.set_title(f"Quarters estimated from {column}")
    ax.set_xlabel("Estimated eighths")
    ax.set_ylabel("# quarters of nights")
    return fig


def plot_estimate_agreement(x_estimates, y_estimates, x_label, y_label, ax=None):
    """2-D histogram comparing two estimators' eighths on their shared quarters.

    `x_estimates`/`y_estimates` are `pandas.Series` of estimated eighths;
    they are aligned on their (shared) index before plotting, and rows
    missing from either are dropped. Mirrors the notebook's cross-band and
    cross-estimator agreement plots (cells 45, 46, 91).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    combined = pd.DataFrame({"x": x_estimates, "y": y_estimates}).dropna()
    h = ax.hist2d(combined["x"], combined["y"], bins=np.arange(-0.5, 9.5, 1))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(h[3], ax=ax, label="count")
    return fig


# ---------------------------------------------------------------------------
# Layer 4: WRITE
# ---------------------------------------------------------------------------


def write_band_estimates(by_quarter, path, columns=None):
    """Write a `compute_by_quarter` frame as a tab-separated text file.

    Replaces the notebook's repeated ``reset_index().to_csv(..., sep="\\t")``
    calls for ``satellite_clouds_band{N}_2015.txt``-style files.

    `by_quarter`'s own ``clouds`` column is the *human* judgment carried
    through from the input samples -- for a missing-quarters frame this is
    the sentinel 9, not useful. To write estimated eighths under a
    ``clouds`` header (as the notebook did for its missing-quarter exports),
    rename the column first, e.g.
    ``by_quarter.rename(columns={"estimated_eighths": "clouds"})``, or pass
    an explicit `columns` list naming ``estimated_eighths`` directly.

    Parameters
    ----------
    by_quarter : pandas.DataFrame
        As returned by `compute_by_quarter`.
    path : str or Path
        Output file path.
    columns : list of str or None
        Columns to write, after `by_quarter.reset_index()`. ``None`` means
        ``["year", "month", "sday", "quarter", "clouds", "fraction_above_cloudcut"]``.

    Returns
    -------
    Path
        `path`, as a `pathlib.Path`.
    """
    if columns is None:
        columns = ["year", "month", "sday", "quarter", "clouds", "fraction_above_cloudcut"]

    by_quarter.reset_index()[columns].to_csv(path, sep="\t", index=False)
    return Path(path)


def write_satellite_cloudy(by_quarter, path, cloudy_threshold=8, stats=None):
    """Write the downstream-compatible ``satellite_cloudy.txt``

    Selects quarters whose ``estimated_eighths`` is at or above
    `cloudy_threshold`, and writes them with statistic columns ``mean std
    min 25% 50% 75% max`` (filled with NaN if `stats` is not supplied).

    ``fill_missing_clouds.ipynb`` reads this file and unconditionally
    assigns ``clouds = 8`` and ``source = 'satellite'`` to every listed
    quarter -- the statistic columns are informational only. The contract
    that matters downstream is: the four index columns are a list of
    quarters to mark as fully clouded.

    Parameters
    ----------
    by_quarter : pandas.DataFrame
        As returned by `compute_by_quarter`.
    path : str or Path
        Output file path.
    cloudy_threshold : int
        Minimum ``estimated_eighths`` to include a quarter.
    stats : pandas.DataFrame or None
        A `compute_sample_stats`-shaped frame, indexed by the same quarter
        levels as `by_quarter`, supplying the statistic columns.

    Returns
    -------
    Path
        `path`, as a `pathlib.Path`.
    """
    cloudy_index = by_quarter.index[by_quarter["estimated_eighths"] >= cloudy_threshold]

    if stats is not None:
        stat_columns = stats.reindex(cloudy_index)[list(SATELLITE_CLOUDY_STAT_COLUMNS)]
    else:
        stat_columns = pd.DataFrame(
            np.nan, index=cloudy_index, columns=list(SATELLITE_CLOUDY_STAT_COLUMNS)
        )

    stat_columns.reset_index().to_csv(path, sep="\t", index=False)
    return Path(path)


# ---------------------------------------------------------------------------
# Layer 5: ACQUIRE (opt-in network)
# ---------------------------------------------------------------------------
#
# Nothing in this section runs at import time or as a side effect of any
# other layer. A caller must construct a `McfetchClient` and call one of its
# methods to touch the network.

_MCFETCH_URL_BASE = "https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch"
_QUOTA_EXCEEDED_MESSAGE = "ERROR: Account over Daily Quota"


class McfetchClient:
    """Client for the SSEC "mcfetch" GOES-13 image-cutout service.

    Preserves the notebook's URL/filename construction and quota-detection
    logic exactly, since the existing on-disk cache
    (`clouds/data/`) depends on the filename convention and
    `extract_band_samples` globs for it. Fixes two defects along the way:
    the access key is read lazily on first actual use, not at import time
    (§4.6), and exhausted-quota dates are tracked per instance, not in a
    module-level global (§4.4).
    """

    def __init__(
        self,
        access_key_path=SDS_ACCESS_KEY_PATH,
        data_dir=DEFAULT_DATA_DIR,
        allow_download=None,
    ):
        self.access_key_path = access_key_path
        self.data_dir = data_dir
        self.allow_download = allow_download
        self._access_key = None
        self._exhausted_dates = set()

    def _resolved_allow_download(self):
        return ALLOW_DOWNLOAD if self.allow_download is None else self.allow_download

    @property
    def access_key(self):
        """The mcfetch access key, read from `access_key_path` on first use."""
        if self._access_key is None:
            self._access_key = Path(self.access_key_path).expanduser().read_text().strip()
        return self._access_key

    def download_image(self, tstamp, band, directory):
        """Download a single GOES-13 band image cutout centered near CTIO.

        If the target file already exists on disk, the download is skipped
        (returning the existing path) without touching the network or the
        allow-download guard.

        Parameters
        ----------
        tstamp : str
            Timestamp of the desired image, ``"YYYY-MM-DD HH:MM:SS"`` (the
            format returned in the SDS inventory's ``tstamp`` column).
        band : int
            GOES imager band number.
        directory : str or Path
            Directory to save the downloaded file in; created if absent.

        Returns
        -------
        Path
            Path to the downloaded (or already-cached) NetCDF file.

        Raises
        ------
        DownloadNotAllowedError
            If downloads are not allowed (checked before any file or
            network access, and before the access key is ever read).
        RuntimeError
            If today's quota is already known to be exhausted, or the
            service returns an error response.
        """
        fname_tstamp = tstamp.replace(" ", "T").replace(":", "")
        directory = Path(directory)
        fname = directory / f"{SATELLITE_NAME_MCFETCH}_{band}_{fname_tstamp}.nc"
        if fname.exists():
            return fname

        if not self._resolved_allow_download():
            raise DownloadNotAllowedError(
                f"{fname} is not cached, and downloads are not allowed. "
                "Pass allow_download=True or set goesclouds.ALLOW_DOWNLOAD = True."
            )

        today = datetime.date.today().isoformat()
        if today in self._exhausted_dates:
            raise RuntimeError(
                f"Skipping, quota for today ({today}) already known to be exceeded."
            )

        lat, lon = _ctio_lat_lon_for_sds()
        # mcfetch wants lat/lon joined with an explicit sign: '+' for
        # positive longitude, nothing extra for negative (the number
        # already has a '-' sign).
        lat_lon_str = f"{lat}+{lon}" if lon > 0 else f"{lat}{lon}"

        image_date, image_time = tstamp.split()
        image_date = image_date.replace("-", "")

        url = (
            f"{_MCFETCH_URL_BASE}?dkey={self.access_key}&satellite={SATELLITE_NAME_MCFETCH}"
            f"&output=NETCDF&lat={lat_lon_str}&size={CUTOUT_SIZE_PIX}"
            f"&date={image_date}&time={image_time}&coverage=SH&band={band}"
        )

        directory.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, fname)

        if fname.stat().st_size < 100:
            content = fname.read_text()
            if content.startswith("ERROR"):
                logger.debug(f"{fname} content: {content}")
                if content.strip() == _QUOTA_EXCEEDED_MESSAGE:
                    self._exhausted_dates.add(today)
                    fname.unlink()
                raise RuntimeError(content)

        return fname

    def fetch_quarter_images(self, quarter_reports, date_str, quarter, band=DEFAULT_BAND):
        """Download every inventoried image for one night-quarter.

        Looks up the quarter's start/end time from `quarter_reports`,
        queries the SDS inventory for that range, and downloads every
        southern-hemisphere ("SH") image found into the directory
        `quarter_directory` names for this quarter.

        Parameters
        ----------
        quarter_reports : pandas.DataFrame
            A `load_quarter_reports`-shaped table (or any subset of it,
            e.g. `select_sample_quarters`'s output) with ``start_iso``/
            ``end_iso`` columns.
        date_str : str
            Local calendar date of the start of the night, ``"YYYY-MM-DD"``.
        quarter : int
            Quarter of the night, 1-4.
        band : int
            GOES imager band to download.

        Returns
        -------
        list of Path
            Paths to the downloaded (or already-cached) NetCDF files.
        """
        obs_datetime = Time(date_str).datetime
        year, month, sday = obs_datetime.year, obs_datetime.month, obs_datetime.day
        directory = quarter_directory(year, month, sday, quarter, data_dir=self.data_dir)

        start_iso, end_iso = quarter_reports.loc[
            (year, month, sday, quarter), ["start_iso", "end_iso"]
        ]
        sh_inventory = query_sds_inventory(
            start_iso, end_iso, allow_download=self._resolved_allow_download()
        ).query('coverage == "SH"')

        return [
            self.download_image(record.tstamp, band, directory=directory)
            for _, record in sh_inventory.iterrows()
        ]

    def download_quarters(self, quarters, band, skip_existing=True):
        """Download images for every quarter in `quarters`.

        Mirrors the notebook's `download_sampled_images`: continues past
        quarters with no inventory entries, logs and continues past other
        errors, and stops early once today's quota is known to be
        exhausted.

        Parameters
        ----------
        quarters : pandas.DataFrame
            A `load_quarter_reports`-shaped table (e.g.
            `select_sample_quarters`'s or `select_missing_quarters`'s
            output), indexed by (year, month, sday, quarter).
        band : int
            GOES imager band to download.
        skip_existing : bool
            If True, skip a quarter entirely (no inventory query) when its
            directory already has at least one cached image for this band.

        Returns
        -------
        list of Path
            Paths to all downloaded (or already-cached) NetCDF files.
        """
        today = datetime.date.today().isoformat()
        fnames = []
        for index_key, _ in quarters.iterrows():
            if today in self._exhausted_dates:
                logger.error(
                    "McFETCH quota filled for today: skipping remaining download attempts."
                )
                break

            year, month, sday, quarter = index_key
            directory = quarter_directory(year, month, sday, quarter, data_dir=self.data_dir)
            if skip_existing and directory.exists() and any(
                directory.glob(f"{SATELLITE_NAME_MCFETCH}_{band}_*.nc")
            ):
                continue

            date_str = f"{year}-{month:02d}-{sday:02d}"
            try:
                fnames.extend(
                    self.fetch_quarter_images(quarters, date_str, quarter, band=band)
                )
            except pd.errors.EmptyDataError:
                continue
            except RuntimeError as runtime_error:
                logger.error(f"Error fetching quarter: {runtime_error}")
                if today in self._exhausted_dates:
                    logger.error(
                        "McFETCH quota filled for today: skipping remaining download attempts."
                    )
                    break

        return fnames
