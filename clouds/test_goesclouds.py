"""Unit tests for goesclouds.py.

Run with:
    /media/psf/aisandbox/goesclouds/.venv/bin/python -m unittest test_goesclouds -v

No test in this suite touches the network. Tests that need the real
gitignored image cache under clouds/data/ are guarded with
@unittest.skipUnless(Path("data").is_dir(), ...) so the suite still passes
on a fresh clone.
"""

import calendar
import datetime
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

import goesclouds as gc

CUTOUT_SIDE = 2 * gc.CUTOUT_CENTER_PIX  # 256


def _embed_window(values, window_size=gc.DEFAULT_WINDOW_SIZE):
    """Build a full 256x256 image with known ``values`` at the center window
    and zeros elsewhere, so `extract_band_samples` output can be checked
    against exact known values.
    """
    image = np.zeros((CUTOUT_SIDE, CUTOUT_SIDE), dtype=float)
    sl = gc._window_slice(window_size)
    image[sl, sl] = np.asarray(values, dtype=float).reshape(window_size, window_size)
    return image


def _make_netcdf(path, values):
    """Write a minimal single-band GOES cutout NetCDF file for testing.

    ``values`` must be a (256, 256) array; it becomes the sole time-slice of
    the ``data`` variable. ``lat``/``lon`` are filled with placeholder zeros
    since extract_band_samples never reads them.
    """
    values = np.asarray(values, dtype=">f4")
    assert values.shape == (CUTOUT_SIDE, CUTOUT_SIDE), values.shape
    with scipy.io.netcdf_file(str(path), "w") as f:
        f.createDimension("xc", CUTOUT_SIDE)
        f.createDimension("yc", CUTOUT_SIDE)
        f.createDimension("time", 1)
        data = f.createVariable("data", "f4", ("time", "xc", "yc"))
        data[:] = values[np.newaxis, :, :]
        lat = f.createVariable("lat", "f4", ("xc", "yc"))
        lat[:] = np.zeros((CUTOUT_SIDE, CUTOUT_SIDE), dtype=">f4")
        lon = f.createVariable("lon", "f4", ("xc", "yc"))
        lon[:] = np.zeros((CUTOUT_SIDE, CUTOUT_SIDE), dtype=">f4")


class TestWindowSlice(unittest.TestCase):
    """T1: _window_slice."""

    def test_default_window_size_matches_notebook(self):
        self.assertEqual(gc._window_slice(4), slice(126, 130))

    def test_symmetric_window_size(self):
        self.assertEqual(gc._window_slice(5), slice(126, 131))

    def test_pixel_count_matches_window_size_squared(self):
        image = np.arange(256 * 256).reshape(256, 256)
        for window_size in (2, 3, 4, 5, 7):
            sl = gc._window_slice(window_size)
            extracted = image[sl, sl]
            self.assertEqual(extracted.size, window_size**2)


class TestNoNetworkOnImport(unittest.TestCase):
    """T11: importing goesclouds must not touch the network."""

    def test_import_does_not_call_urlopen_or_urlretrieve(self):
        # Save the original module to restore it after the test
        original_module = sys.modules.get("goesclouds")

        def _raise(*args, **kwargs):
            raise AssertionError("network call attempted at import time")

        # Temporarily remove goesclouds from sys.modules and re-import with mocks
        sys.modules.pop("goesclouds", None)

        with mock.patch("urllib.request.urlopen", side_effect=_raise), mock.patch(
            "urllib.request.urlretrieve", side_effect=_raise
        ):
            import goesclouds  # noqa: F401 -- re-imported fresh to test import-time behavior

        # Restore the original module so subsequent tests use the same module
        # that test_goesclouds imports at module level
        sys.modules["goesclouds"] = original_module


class TestMakeNetcdfFixture(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.nc"
            values = np.arange(CUTOUT_SIDE * CUTOUT_SIDE, dtype=">f4").reshape(
                CUTOUT_SIDE, CUTOUT_SIDE
            )
            _make_netcdf(path, values)
            with scipy.io.netcdf_file(str(path), "r", mmap=False) as f:
                data = f.variables["data"].data[0]
                np.testing.assert_array_equal(data, values)


class TestQuarterDirectory(unittest.TestCase):
    """T10: quarter_directory."""

    def test_zero_padding_and_suffix(self):
        path = gc.quarter_directory(2015, 1, 2, 3, data_dir="data")
        self.assertEqual(path.name, "satellite_2015-01-02_q3")

    def test_uses_sday_not_eday(self):
        # sday=31 is the last day of the month; quarter_directory must use
        # sday and must not derive anything from an eday-like value.
        path = gc.quarter_directory(2015, 1, 31, 1, data_dir="data")
        self.assertEqual(path.name, "satellite_2015-01-31_q1")

    @unittest.skipUnless(
        (Path(__file__).parent / "data").is_dir(), "image cache not present"
    )
    def test_matches_real_directory_name(self):
        real_dirs = [
            p.name for p in (Path(__file__).parent / "data").iterdir() if p.is_dir()
        ]
        sample = next(name for name in real_dirs if name.startswith("satellite_"))
        _, date_part, quarter_part = sample.split("_")
        year, month, sday = (int(x) for x in date_part.split("-"))
        quarter = int(quarter_part[1:])
        constructed = gc.quarter_directory(year, month, sday, quarter, data_dir="data").name
        self.assertEqual(constructed, sample)


class TestLoadQuarterReports(unittest.TestCase):
    """T9, T9b, T9c: load_quarter_reports, against the real tracked
    clouds_ctio_blanco.h5 / night_events.h5 (not gitignored, no image cache
    needed -- these tests are not skip-guarded).
    """

    @classmethod
    def setUpClass(cls):
        cls.quarter_reports = gc.load_quarter_reports()

    def test_index_and_iso_shape(self):
        qr = self.quarter_reports
        self.assertEqual(qr.index.names, ["year", "month", "sday", "quarter"])
        self.assertTrue((qr["start_iso"].str.len() == 19).all())
        self.assertTrue((qr["end_iso"].str.len() == 19).all())
        pd.to_datetime(qr["start_iso"])  # must parse without error
        pd.to_datetime(qr["end_iso"])

    def test_start_before_end_and_matches_quarter_duration(self):
        qr = self.quarter_reports
        start = pd.to_datetime(qr["start_iso"])
        end = pd.to_datetime(qr["end_iso"])
        self.assertTrue((start < end).all())

        night_events = pd.read_hdf(gc._resolve(gc.DEFAULT_NIGHT_EVENTS))
        duration_days = (end - start).dt.total_seconds() / 86400
        expected_days = night_events["quarter_duration"].reindex(
            qr.index.droplevel("quarter")
        )
        expected_days.index = qr.index
        # atol allows for the 1-second truncation when start_iso/end_iso are
        # rounded to 19-char strings.
        np.testing.assert_allclose(
            duration_days.to_numpy(), expected_days.to_numpy(), atol=2 / 86400
        )

    def test_sday_is_start_day(self):
        # T9b: date.dt.day == sday for every row; date.dt.day == eday only
        # for a small number of rows (it is not, in general, true).
        qr = self.quarter_reports
        date = pd.to_datetime(qr["date"])
        sday = qr.index.get_level_values("sday")
        eday = qr["eday"]
        self.assertTrue((date.dt.day.to_numpy() == sday.to_numpy()).all())
        n_equal_eday = int((date.dt.day.to_numpy() == eday.to_numpy()).sum())
        self.assertLess(n_equal_eday, 10)

    def test_eday_rollover(self):
        # T9c: eday == sday + 1 for the great majority of nights; where not,
        # eday == 1 and sday is the last day of the month, with a documented
        # single real-data anomaly tolerated.
        qr = self.quarter_reports
        nights = (
            qr.reset_index()[["year", "month", "sday", "eday"]]
            .drop_duplicates(subset=["year", "month", "sday"])
        )
        normal = nights["eday"] == nights["sday"] + 1
        self.assertGreater(normal.sum() / len(nights), 0.9)

        abnormal = nights[~normal]
        last_of_month = abnormal.apply(
            lambda row: row["sday"] == calendar.monthrange(row["year"], row["month"])[1],
            axis=1,
        )
        fits_pattern = (abnormal["eday"] == 1) & last_of_month
        self.assertLessEqual((~fits_pattern).sum(), 1)


class TestSelectSampleQuarters(unittest.TestCase):
    """T7: sample reproducibility -- protects the on-disk image cache's
    validity, since it only covers the quarters this exact draw selects.
    """

    # Computed once from the real, tracked clouds_ctio_blanco.h5 /
    # night_events.h5 with the current SAMPLE_RNG_SEED; a change here means
    # the cache under clouds/data/ no longer matches select_sample_quarters.
    EXPECTED_DIGEST = "7783be0aa8e9ffe232353525ac8ce2074d5814acf0df574587dec3b03625392e"

    @classmethod
    def setUpClass(cls):
        cls.quarter_reports = gc.load_quarter_reports()

    def test_counts_and_year_range(self):
        sample = gc.select_sample_quarters(self.quarter_reports)
        self.assertEqual(len(sample), 891)
        counts = sample["clouds"].value_counts()
        self.assertEqual(set(counts.index), set(range(9)))
        self.assertTrue((counts == 99).all())
        years = sample.index.get_level_values("year")
        self.assertTrue(years.min() >= 2013)
        self.assertTrue(years.max() <= 2016)

    def test_index_digest_matches_expected(self):
        sample = gc.select_sample_quarters(self.quarter_reports)
        idx_frame = sample.index.to_frame(index=False)
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(idx_frame, index=False).to_numpy().tobytes()
        ).hexdigest()
        self.assertEqual(digest, self.EXPECTED_DIGEST)


class TestSelectMissingQuarters(unittest.TestCase):
    """T8: select_missing_quarters."""

    @classmethod
    def setUpClass(cls):
        cls.quarter_reports = gc.load_quarter_reports()

    def test_missing_2015(self):
        missing = gc.select_missing_quarters(self.quarter_reports, year=2015)
        self.assertEqual(len(missing), 180)
        self.assertTrue((missing["clouds"] == gc.MISSING_CLOUDS_SENTINEL).all())
        self.assertTrue((missing.index.get_level_values("year") == 2015).all())


class TestQuerySdsInventory(unittest.TestCase):
    """T12, T24: query_sds_inventory and the inventory cache, against the
    real, tracked sds_inventories.h5.gz (not gitignored, no image cache
    needed -- not skip-guarded).
    """

    @classmethod
    def setUpClass(cls):
        gc.load_inventory_cache()
        with pd.HDFStore(str(gc._resolve(gc.DEFAULT_INVENTORY_CACHE))) as store:
            cls.real_keys = {k.lstrip("/") for k in store.keys()}
        cls.quarter_reports = gc.load_quarter_reports()

    def test_ctio_lat_lon(self):
        self.assertEqual(gc._ctio_lat_lon_for_sds(), (-30, 71))

    def test_cache_key_matches_a_real_committed_key(self):
        # T24: guards the §6.1 sanitization regex against the real cache.
        missing = gc.select_missing_quarters(self.quarter_reports, year=2015)
        n_matches = 0
        for _, row in missing.reset_index().iterrows():
            url = gc._inventory_url(row.start_iso, row.end_iso)
            key = gc._inventory_cache_key(url)
            if key in self.real_keys:
                n_matches += 1
        self.assertGreater(n_matches, 0)

    def test_cache_hit_returns_without_download(self):
        missing = gc.select_missing_quarters(self.quarter_reports, year=2015)
        for _, row in missing.reset_index().iterrows():
            url = gc._inventory_url(row.start_iso, row.end_iso)
            key = gc._inventory_cache_key(url)
            if key in self.real_keys:
                result = gc.query_sds_inventory(
                    row.start_iso, row.end_iso, allow_download=False
                )
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result), 0)
                return
        self.fail("no missing-2015 quarter matched a real cached inventory key")

    def test_cache_miss_without_allow_download_raises(self):
        # T12: download guard -- a cache miss with allow_download=False must
        # raise, and must not touch the network.
        def _raise(*args, **kwargs):
            raise AssertionError("network call attempted despite allow_download=False")

        with mock.patch("urllib.request.urlopen", side_effect=_raise):
            with self.assertRaises(gc.DownloadNotAllowedError):
                gc.query_sds_inventory(
                    "1900-01-01 00:00:00", "1900-01-01 01:00:00", allow_download=False
                )

    def test_force_query_does_not_bypass_download_guard(self):
        def _raise(*args, **kwargs):
            raise AssertionError("network call attempted despite allow_download=False")

        with mock.patch("urllib.request.urlopen", side_effect=_raise):
            with self.assertRaises(gc.DownloadNotAllowedError):
                gc.query_sds_inventory(
                    "1900-01-01 00:00:00",
                    "1900-01-01 01:00:00",
                    allow_download=False,
                    force_query=True,
                )


class TestExtractBandSamples(unittest.TestCase):
    """T2, T3, T4, T10b, T13 (partial): extract_band_samples."""

    def test_shape_and_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = gc.quarter_directory(2015, 1, 2, 3, data_dir=tmpdir)
            dirname.mkdir(parents=True)
            values_a = np.arange(16).reshape(4, 4)
            values_b = np.arange(16, 32).reshape(4, 4)
            _make_netcdf(
                dirname / "GOES13_4_2015-01-02T010000.nc", _embed_window(values_a)
            )
            _make_netcdf(
                dirname / "GOES13_4_2015-01-02T020000.nc", _embed_window(values_b)
            )

            quarters = pd.DataFrame(
                {
                    "year": [2015],
                    "month": [1],
                    "sday": [2],
                    "quarter": [3],
                    "clouds": [5],
                    "start_iso": ["2015-01-02T00:00:00"],
                    "end_iso": ["2015-01-02T06:00:00"],
                }
            ).set_index(["year", "month", "sday", "quarter"])

            result = gc.extract_band_samples(quarters, band=4, data_dir=tmpdir)

            self.assertEqual(
                result.index.names,
                ["year", "month", "sday", "quarter", "time", "pixel"],
            )
            self.assertEqual(len(result), 32)
            self.assertEqual(sorted(result.index.get_level_values("pixel").unique()), list(range(16)))

            time_a = pd.Timestamp("2015-01-02T01:00:00")
            time_b = pd.Timestamp("2015-01-02T02:00:00")
            got_a = result.xs(time_a, level="time")["band4"].to_numpy()
            got_b = result.xs(time_b, level="time")["band4"].to_numpy()
            np.testing.assert_array_equal(got_a, values_a.flatten())
            np.testing.assert_array_equal(got_b, values_b.flatten())
            self.assertTrue((result["clouds"] == 5).all())

    def test_unreadable_file_is_skipped_others_intact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            good_dir = gc.quarter_directory(2015, 1, 2, 1, data_dir=tmpdir)
            good_dir.mkdir(parents=True)
            _make_netcdf(
                good_dir / "GOES13_4_2015-01-02T010000.nc",
                _embed_window(np.arange(16).reshape(4, 4)),
            )
            (good_dir / "GOES13_4_2015-01-02T020000.nc").write_bytes(b"not a netcdf file")

            empty_dir = gc.quarter_directory(2015, 1, 3, 1, data_dir=tmpdir)
            empty_dir.mkdir(parents=True)

            quarters = pd.DataFrame(
                {
                    "year": [2015, 2015],
                    "month": [1, 1],
                    "sday": [2, 3],
                    "quarter": [1, 1],
                    "clouds": [0, 0],
                    "start_iso": ["2015-01-02T00:00:00", "2015-01-03T00:00:00"],
                    "end_iso": ["2015-01-02T06:00:00", "2015-01-03T06:00:00"],
                }
            ).set_index(["year", "month", "sday", "quarter"])

            with self.assertLogs(gc.logger, level="WARNING"):
                result = gc.extract_band_samples(quarters, band=4, data_dir=tmpdir)

            self.assertEqual(len(result), 16)

    def test_timestamp_parsing_and_time_as_datetime_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = gc.quarter_directory(2015, 1, 2, 3, data_dir=tmpdir)
            dirname.mkdir(parents=True)
            _make_netcdf(
                dirname / "GOES13_4_2015-01-02T011904.nc",
                _embed_window(np.arange(16).reshape(4, 4)),
            )
            quarters = pd.DataFrame(
                {
                    "year": [2015],
                    "month": [1],
                    "sday": [2],
                    "quarter": [3],
                    "clouds": [0],
                    "start_iso": ["2015-01-02T00:00:00"],
                    "end_iso": ["2015-01-02T06:00:00"],
                }
            ).set_index(["year", "month", "sday", "quarter"])

            result = gc.extract_band_samples(quarters, band=4, data_dir=tmpdir)
            expected_time = pd.Timestamp("2015-01-02T01:19:04")
            self.assertEqual(result.index.get_level_values("time")[0], expected_time)

            result_int = gc.extract_band_samples(
                quarters, band=4, data_dir=tmpdir, time_as_datetime=False
            )
            expected_int = pd.to_datetime([expected_time]).astype("int64")[0]
            self.assertEqual(result_int.index.get_level_values("time")[0], expected_int)

    def test_utc_offset_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = gc.quarter_directory(2015, 1, 3, 1, data_dir=tmpdir)
            dirname.mkdir(parents=True)
            _make_netcdf(
                dirname / "GOES13_4_2015-01-03T013000.nc",
                _embed_window(np.arange(16).reshape(4, 4)),
            )
            _make_netcdf(
                dirname / "GOES13_4_2015-01-11T013000.nc",
                _embed_window(np.arange(16, 32).reshape(4, 4)),
            )
            quarters = pd.DataFrame(
                {
                    "year": [2015],
                    "month": [1],
                    "sday": [3],
                    "quarter": [1],
                    "clouds": [0],
                    "start_iso": ["2015-01-03T00:00:00"],
                    "end_iso": ["2015-01-03T06:00:00"],
                }
            ).set_index(["year", "month", "sday", "quarter"])

            with self.assertLogs(gc.logger, level="WARNING"):
                result = gc.extract_band_samples(
                    quarters, band=4, data_dir=tmpdir, max_timestamp_offset_days=2
                )
            self.assertEqual(len(result), 16)

            result_unfiltered = gc.extract_band_samples(
                quarters, band=4, data_dir=tmpdir, max_timestamp_offset_days=None
            )
            self.assertEqual(len(result_unfiltered), 32)

    def test_purity_does_not_mutate_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = gc.quarter_directory(2015, 1, 2, 3, data_dir=tmpdir)
            dirname.mkdir(parents=True)
            _make_netcdf(
                dirname / "GOES13_4_2015-01-02T010000.nc",
                _embed_window(np.arange(16).reshape(4, 4)),
            )
            quarters = pd.DataFrame(
                {
                    "year": [2015],
                    "month": [1],
                    "sday": [2],
                    "quarter": [3],
                    "clouds": [0],
                    "start_iso": ["2015-01-02T00:00:00"],
                    "end_iso": ["2015-01-02T06:00:00"],
                }
            ).set_index(["year", "month", "sday", "quarter"])
            before = quarters.copy(deep=True)

            gc.extract_band_samples(quarters, band=4, data_dir=tmpdir)

            pd.testing.assert_frame_equal(quarters, before)


class TestLoadMultibandSamples(unittest.TestCase):
    """T19: multiband join."""

    def test_partial_band_coverage_produces_nan_not_dropped_or_duplicated_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = gc.quarter_directory(2015, 1, 2, 3, data_dir=tmpdir)
            dirname.mkdir(parents=True)
            # Band 4 has two timestamps; band 2 only has the first -- mirrors
            # the real 3747/3958 band-2/band-4 split (§3.3).
            _make_netcdf(
                dirname / "GOES13_4_2015-01-02T010000.nc",
                _embed_window(np.arange(16).reshape(4, 4)),
            )
            _make_netcdf(
                dirname / "GOES13_4_2015-01-02T020000.nc",
                _embed_window(np.arange(16, 32).reshape(4, 4)),
            )
            _make_netcdf(
                dirname / "GOES13_2_2015-01-02T010000.nc",
                _embed_window(np.arange(100, 116).reshape(4, 4)),
            )

            quarters = pd.DataFrame(
                {
                    "year": [2015],
                    "month": [1],
                    "sday": [2],
                    "quarter": [3],
                    "clouds": [5],
                    "start_iso": ["2015-01-02T00:00:00"],
                    "end_iso": ["2015-01-02T06:00:00"],
                }
            ).set_index(["year", "month", "sday", "quarter"])

            multiband = gc.load_multiband_samples(quarters, bands=(2, 4), data_dir=tmpdir)

            self.assertEqual(
                multiband.index.names,
                ["pixel", "year", "month", "sday", "quarter", "time"],
            )
            self.assertEqual(
                set(multiband.columns), {("human", "clouds"), (2, "value"), (4, "value")}
            )
            # No duplicate rows, no dropped rows: 2 timestamps * 16 pixels.
            self.assertEqual(len(multiband), 32)
            self.assertFalse(multiband.index.duplicated().any())
            self.assertEqual(multiband[(4, "value")].isna().sum(), 0)
            self.assertEqual(multiband[(2, "value")].isna().sum(), 16)
            self.assertTrue((multiband[("human", "clouds")] == 5).all())

            time_a = pd.Timestamp("2015-01-02T01:00:00")
            time_b = pd.Timestamp("2015-01-02T02:00:00")
            band2_at_a = multiband.xs(time_a, level="time")[(2, "value")].to_numpy()
            band2_at_b = multiband.xs(time_b, level="time")[(2, "value")].to_numpy()
            np.testing.assert_array_equal(sorted(band2_at_a), list(range(100, 116)))
            self.assertTrue(np.isnan(band2_at_b).all())


def _make_band_samples():
    """Synthetic band-4 samples: 3 quarters, 3 cloud levels, 5 pixels each.

    ``band4 = 100 * clouds + pixel``, so at ``clouds=4`` the values are
    400..404 (median 402) and at ``clouds=0``/``clouds=8`` they bracket it.
    """
    rows = []
    for q in range(3):
        for clouds in (0, 4, 8):
            for pixel in range(5):
                rows.append(
                    dict(
                        year=2015,
                        month=1,
                        sday=1 + q,
                        quarter=1,
                        time=pd.Timestamp("2015-01-01") + pd.Timedelta(hours=pixel),
                        pixel=pixel,
                        clouds=clouds,
                        band4=100.0 * clouds + pixel,
                    )
                )
    return pd.DataFrame(rows).set_index(
        ["year", "month", "sday", "quarter", "time", "pixel"]
    )


class TestEstimateEighths(unittest.TestCase):
    """T5."""

    def test_known_values(self):
        self.assertEqual(gc.estimate_eighths(1.0), 0)
        self.assertEqual(gc.estimate_eighths(0.0), 8)
        self.assertEqual(gc.estimate_eighths(0.5), 4)

    def test_monotone_and_bounded(self):
        fractions = np.linspace(0, 1, 17)
        eighths = gc.estimate_eighths(fractions)
        self.assertTrue(np.issubdtype(eighths.dtype, np.integer))
        self.assertTrue((eighths >= 0).all() and (eighths <= 8).all())
        # Non-increasing as fraction-above-reference increases.
        self.assertTrue((np.diff(eighths) <= 0).all())


class TestFractionGreater(unittest.TestCase):
    """T6."""

    def test_strict_inequality_and_extremes(self):
        values = pd.Series([1, 2, 3, 4, 5])
        self.assertEqual(gc.fraction_greater(3, values), 0.4)
        self.assertEqual(gc.fraction_greater(0, values), 1.0)
        self.assertEqual(gc.fraction_greater(10, values), 0.0)
        # A value equal to the reference does not count as "greater".
        self.assertEqual(gc.fraction_greater(5, values), 0.0)


class TestComputeSampleStats(unittest.TestCase):
    """T14."""

    def test_derived_columns_and_clouds_survives(self):
        band_samples = _make_band_samples()
        stats = gc.compute_sample_stats(band_samples)

        self.assertEqual(list(stats.index.names), ["year", "month", "sday", "quarter"])
        self.assertIn("clouds", stats.columns)
        np.testing.assert_allclose(stats["range"], stats["max"] - stats["min"])
        np.testing.assert_allclose(stats["IQR"], stats["75%"] - stats["25%"])

    def test_works_for_value_column_name_too(self):
        # `as_value_frame` output uses "value" instead of "band{N}"; the
        # default `value_column=None` must pick it up unchanged.
        band_samples = _make_band_samples().rename(columns={"band4": "value"})
        stats = gc.compute_sample_stats(band_samples)
        self.assertIn("clouds", stats.columns)
        np.testing.assert_allclose(stats["range"], stats["max"] - stats["min"])


class TestReferenceQuantile(unittest.TestCase):
    """T15."""

    def test_default_level_is_the_median(self):
        band_samples = _make_band_samples()
        self.assertEqual(gc.reference_quantile(band_samples, "band4"), 402)

    def test_non_median_level_uses_level_over_8_quantile(self):
        # clouds=8 -> quantile 8/8=1.0 (max) of band4 values 800..804.
        # clouds=0 -> quantile 0/8=0.0 (min) of band4 values 0..4.
        band_samples = _make_band_samples()
        self.assertEqual(
            gc.reference_quantile(band_samples, "band4", reference_level=8), 804
        )
        self.assertEqual(
            gc.reference_quantile(band_samples, "band4", reference_level=0), 0
        )

    def test_raises_when_level_absent(self):
        band_samples = _make_band_samples()
        with self.assertRaises(ValueError):
            gc.reference_quantile(band_samples, "band4", reference_level=99)


class TestComputeByQuarter(unittest.TestCase):
    """T13 (partial: purity of compute_by_quarter), plus shape/value checks."""

    def test_does_not_mutate_input(self):
        band_samples = _make_band_samples()
        before = band_samples.copy()
        gc.compute_by_quarter(band_samples, "band4", reference_median_value=402)
        pd.testing.assert_frame_equal(band_samples, before)

    def test_shape_and_values(self):
        band_samples = _make_band_samples()
        by_quarter = gc.compute_by_quarter(band_samples, "band4", reference_median_value=402)

        self.assertEqual(list(by_quarter.index.names), ["year", "month", "sday", "quarter"])
        self.assertEqual(
            list(by_quarter.columns), ["clouds", "fraction_above_cloudcut", "estimated_eighths"]
        )
        # 9 (quarter, cloud-level) groups: 3 sdays * 3 cloud levels.
        self.assertEqual(len(by_quarter), 9)

        at_clouds_4 = by_quarter[by_quarter["clouds"] == 4]
        # band4 values at clouds=4 are 400..404; 2 of 5 (403, 404) exceed 402.
        np.testing.assert_allclose(at_clouds_4["fraction_above_cloudcut"], 0.4)
        np.testing.assert_array_equal(at_clouds_4["estimated_eighths"], 5)


def _make_multiband_samples(n_pixels=3, degenerate_pixel=2):
    """Synthetic multiband frame: clear samples follow band4 = 3*band2 + 100,
    cloudy samples are darkened outliers off that line. ``degenerate_pixel``
    (if not None) is left with only one clear sample.
    """
    rows = []
    for pixel in range(n_pixels):
        for i in range(20):
            clouds = 0 if i < 15 else 8
            band2 = 100.0 + i + pixel
            band4 = 3 * band2 + 100 if clouds == 0 else 3 * band2 + 100 - 500
            rows.append(
                dict(
                    pixel=pixel,
                    year=2015,
                    month=1,
                    sday=1,
                    quarter=1,
                    time=pd.Timestamp("2015-01-01") + pd.Timedelta(minutes=i),
                    clouds=clouds,
                    band2=band2,
                    band4=band4,
                )
            )
    if degenerate_pixel is not None:
        rows = [r for r in rows if not (r["pixel"] == degenerate_pixel and r["clouds"] == 0)]
        rows.append(
            dict(
                pixel=degenerate_pixel,
                year=2015,
                month=1,
                sday=1,
                quarter=1,
                time=pd.Timestamp("2015-01-01T00:00:00"),
                clouds=0,
                band2=150.0,
                band4=550.0,
            )
        )
    df = pd.DataFrame(rows).set_index(
        ["pixel", "year", "month", "sday", "quarter", "time"]
    )
    return pd.DataFrame(
        {
            ("human", "clouds"): df["clouds"],
            (2, "value"): df["band2"],
            (4, "value"): df["band4"],
        }
    )


class TestFitClearSkyModel(unittest.TestCase):
    """T16, T17."""

    def test_degenerate_pixel_yields_none_and_logs(self):
        multiband = _make_multiband_samples(n_pixels=3, degenerate_pixel=2)
        with self.assertLogs("goesclouds", level="WARNING"):
            models = gc.fit_clear_sky_model(multiband, x_band=2, y_band=4, per_pixel=True)
        self.assertIsNone(models[2])
        self.assertIsNotNone(models[0])
        self.assertIsNotNone(models[1])

    def test_recovers_known_slope_and_intercept(self):
        multiband = _make_multiband_samples(n_pixels=2, degenerate_pixel=None)
        models = gc.fit_clear_sky_model(multiband, x_band=2, y_band=4, per_pixel=True)
        for model in models.values():
            self.assertIsNotNone(model)
            self.assertAlmostEqual(model.coef_[0], 3.0, places=6)
            self.assertAlmostEqual(model.intercept_, 100.0, places=6)


class TestApplyClearSkyCorrection(unittest.TestCase):
    """T18, T13 (partial: purity of apply_clear_sky_correction)."""

    def test_does_not_mutate_input(self):
        multiband = _make_multiband_samples()
        before = multiband.copy()
        models = gc.fit_clear_sky_model(multiband, x_band=2, y_band=4, per_pixel=True)
        gc.apply_clear_sky_correction(multiband, models, x_band=2, y_band=4)
        pd.testing.assert_frame_equal(multiband, before)

    def test_clear_near_zero_cloudy_negative_degenerate_nan(self):
        multiband = _make_multiband_samples(n_pixels=3, degenerate_pixel=2)
        models = gc.fit_clear_sky_model(multiband, x_band=2, y_band=4, per_pixel=True)
        corrected = gc.apply_clear_sky_correction(multiband, models, x_band=2, y_band=4)

        pixel_level = corrected.index.get_level_values("pixel")
        clouds = corrected[("human", "clouds")]
        corr = corrected[("4corr", "value")]

        fitted_pixels = pixel_level.isin([0, 1])
        clear = clouds == 0
        cloudy = clouds == 8

        np.testing.assert_allclose(
            corr[clear & fitted_pixels].to_numpy(), 0.0, atol=1e-6
        )
        self.assertTrue((corr[cloudy & fitted_pixels] < 0).all())
        self.assertTrue(corr[pixel_level == 2].isna().all())


class TestAsValueFrame(unittest.TestCase):
    """Supports T16-T18's usage pattern; not separately numbered."""

    def test_flattens_and_drops_nan_keeps_index(self):
        multiband = _make_multiband_samples(n_pixels=3, degenerate_pixel=2)
        models = gc.fit_clear_sky_model(multiband, x_band=2, y_band=4, per_pixel=True)
        corrected = gc.apply_clear_sky_correction(multiband, models, x_band=2, y_band=4)

        value_frame = gc.as_value_frame(corrected, ("4corr", "value"))

        self.assertEqual(list(value_frame.columns), ["clouds", "value"])
        self.assertEqual(value_frame.index.names, corrected.index.names)
        self.assertFalse(value_frame["value"].isna().any())
        self.assertLess(len(value_frame), len(corrected))


class TestCompareEstimates(unittest.TestCase):
    """T20."""

    @classmethod
    def setUpClass(cls):
        # Hand-built confusion matrix, observable_cut=2.5 (the default):
        #   q0: human=0 est=0  exact match, both observable      -> obs match
        #   q1: human=2 est=3  not exact, human obs / est not    -> obs mismatch
        #   q2: human=4 est=4  exact match, both not observable  -> obs match
        #   q3: human=6 est=6  exact match, both not observable  -> obs match
        #   q4: human=8 est=8  exact match, both not observable  -> obs match
        #   q5: human=9 (sentinel) -- excluded
        #   q6: human=3 est=NaN -- excluded
        idx = pd.RangeIndex(7, name="q")
        cls.samples_by_quarter = pd.DataFrame(
            {
                ("human", "clouds"): [0, 2, 4, 6, 8, 9, 3],
                (4, "estimated_eighths"): [0, 3, 4, 6, 8, 1, np.nan],
            },
            index=idx,
        )

    def _check_hand_computed(self, result):
        row = result.loc["cut on 4"]
        self.assertEqual(row["n"], 5)
        self.assertEqual(row["n_exact"], 4)
        self.assertEqual(row["n_match_obs"], 4)
        self.assertAlmostEqual(row["frac_match_obs"], 4 / 5)
        self.assertAlmostEqual(row["human_mean_eighths"], np.mean([0, 2, 4, 6, 8]))
        self.assertAlmostEqual(row["estimated_mean_eighths"], np.mean([0, 3, 4, 6, 8]))

    def test_column_key_form_excludes_sentinel_and_nan(self):
        result = gc.compare_estimates(
            self.samples_by_quarter, {"cut on 4": (4, "estimated_eighths")}
        )
        self._check_hand_computed(result)

    def test_series_form_matches_column_key_form(self):
        by_key = gc.compare_estimates(
            self.samples_by_quarter, {"cut on 4": (4, "estimated_eighths")}
        )
        by_series = gc.compare_estimates(
            self.samples_by_quarter,
            {"cut on 4": self.samples_by_quarter[(4, "estimated_eighths")]},
        )
        pd.testing.assert_frame_equal(by_key, by_series)

    def test_self_contained_frame_form_matches(self):
        by_key = gc.compare_estimates(
            self.samples_by_quarter, {"cut on 4": (4, "estimated_eighths")}
        )
        frame = pd.DataFrame(
            {
                "clouds": [0, 2, 4, 6, 8, 9, 3],
                "estimated_eighths": [0, 3, 4, 6, 8, 1, np.nan],
            },
            index=pd.RangeIndex(7, name="q"),
        )
        by_frame = gc.compare_estimates(self.samples_by_quarter, {"cut on 4": frame})
        pd.testing.assert_frame_equal(by_key, by_frame)


def _make_write_test_by_quarter():
    idx = pd.MultiIndex.from_tuples(
        [(2015, 1, 1, 1), (2015, 1, 2, 3), (2015, 1, 3, 2)],
        names=["year", "month", "sday", "quarter"],
    )
    return pd.DataFrame(
        {
            "clouds": [9, 9, 9],
            "fraction_above_cloudcut": [0.1, 0.8, 0.5],
            "estimated_eighths": [7, 1, 4],
        },
        index=idx,
    )


class TestWriteBandEstimates(unittest.TestCase):
    """T21."""

    def test_round_trip_and_header(self):
        by_quarter = _make_write_test_by_quarter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.txt"
            returned = gc.write_band_estimates(by_quarter, path)
            self.assertEqual(returned, Path(path))

            with open(path) as f:
                header = f.readline()
            self.assertEqual(
                header, "year\tmonth\tsday\tquarter\tclouds\tfraction_above_cloudcut\n"
            )

            reread = pd.read_csv(path, sep="\t")
            expected = by_quarter.reset_index()[
                ["year", "month", "sday", "quarter", "clouds", "fraction_above_cloudcut"]
            ]
            pd.testing.assert_frame_equal(reread, expected)

    def test_custom_columns(self):
        by_quarter = _make_write_test_by_quarter()
        columns = ["year", "month", "sday", "quarter", "estimated_eighths"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.txt"
            gc.write_band_estimates(by_quarter, path, columns=columns)
            reread = pd.read_csv(path, sep="\t")
            self.assertEqual(list(reread.columns), columns)


class TestWriteSatelliteCloudy(unittest.TestCase):
    """T22."""

    def test_header_is_the_11_documented_columns(self):
        by_quarter = _make_write_test_by_quarter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "satellite_cloudy.txt"
            returned = gc.write_satellite_cloudy(by_quarter, path, cloudy_threshold=5)
            self.assertEqual(returned, Path(path))
            with open(path) as f:
                header = f.readline()
            self.assertEqual(
                header,
                "year\tmonth\tsday\tquarter\tmean\tstd\tmin\t25%\t50%\t75%\tmax\n",
            )

    def test_filters_by_threshold_and_fills_stats(self):
        by_quarter = _make_write_test_by_quarter()
        idx = by_quarter.index
        stats = pd.DataFrame(
            {
                "clouds": [0, 0, 0],
                "count": [10, 10, 10],
                "mean": [1.0, 2.0, 3.0],
                "std": [0.1, 0.2, 0.3],
                "min": [0.5, 1.5, 2.5],
                "25%": [0.7, 1.7, 2.7],
                "50%": [1.0, 2.0, 3.0],
                "75%": [1.3, 2.3, 3.3],
                "max": [1.5, 2.5, 3.5],
                "range": [1, 1, 1],
                "IQR": [0.6, 0.6, 0.6],
            },
            index=idx,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "satellite_cloudy.txt"
            # Only the (2015, 1, 1, 1) row has estimated_eighths >= 5.
            gc.write_satellite_cloudy(by_quarter, path, cloudy_threshold=5, stats=stats)
            reread = pd.read_csv(
                path, sep="\t", index_col=["year", "month", "sday", "quarter"]
            )
            self.assertEqual(len(reread), 1)
            self.assertEqual(reread.index[0], (2015, 1, 1, 1))
            self.assertAlmostEqual(reread.loc[(2015, 1, 1, 1), "mean"], 1.0)

    def test_no_stats_fills_nan(self):
        by_quarter = _make_write_test_by_quarter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "satellite_cloudy.txt"
            gc.write_satellite_cloudy(by_quarter, path, cloudy_threshold=5, stats=None)
            reread = pd.read_csv(path, sep="\t")
            self.assertEqual(len(reread), 1)
            self.assertTrue(reread["mean"].isna().all())


def _make_sample_stats_multiband(bands=(2, 4)):
    """Multi-band, MultiIndex-column `sample_stats`: `(band, stat)` columns
    plus `("human", "clouds")`, assembled the way goesclouds.md §7.4 does --
    per-band `compute_sample_stats` + `pd.concat`.
    """
    band_samples = _make_band_samples().rename(columns={"band4": "value"})
    human_clouds = None
    frames = {}
    for band in bands:
        stats = gc.compute_sample_stats(band_samples, value_column="value")
        human_clouds = stats["clouds"]
        del stats["clouds"]
        stats.columns = pd.MultiIndex.from_product([[band], stats.columns])
        frames[band] = stats
    combined = pd.concat(frames.values(), axis=1)
    combined[("human", "clouds")] = human_clouds
    return combined


class TestPlots(unittest.TestCase):
    """T23: every plot function returns a Figure and does not raise."""

    def tearDown(self):
        plt.close("all")

    def test_plot_sample_stats(self):
        stats = gc.compute_sample_stats(_make_band_samples())
        fig = gc.plot_sample_stats(stats, band=4)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_by_quarter(self):
        band_samples = _make_band_samples()
        by_quarter = gc.compute_by_quarter(band_samples, "band4", reference_median_value=402)
        fig = gc.plot_by_quarter(by_quarter, band=4)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_norm_by_quarter(self):
        band_samples = _make_band_samples()
        by_quarter = gc.compute_by_quarter(band_samples, "band4", reference_median_value=402)
        fig = gc.plot_norm_by_quarter(by_quarter)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_estimate_histogram(self):
        band_samples = _make_band_samples()
        by_quarter = gc.compute_by_quarter(band_samples, "band4", reference_median_value=402)
        fig = gc.plot_estimate_histogram(by_quarter)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_estimate_agreement(self):
        band_samples = _make_band_samples()
        by_quarter = gc.compute_by_quarter(band_samples, "band4", reference_median_value=402)
        fig = gc.plot_estimate_agreement(
            by_quarter["estimated_eighths"], by_quarter["estimated_eighths"], "x", "y"
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_band_comparison(self):
        sample_stats = _make_sample_stats_multiband()
        clear_mask = sample_stats[("human", "clouds")] == 0
        fig, ax = plt.subplots()
        returned = gc.plot_band_comparison(
            sample_stats, "mean", ax, x_band=4, y_band=2, clear_mask=clear_mask
        )
        self.assertIs(returned, fig)

    def test_plot_clear_sky_model(self):
        multiband = _make_multiband_samples(n_pixels=2, degenerate_pixel=None)
        models = gc.fit_clear_sky_model(multiband, x_band=2, y_band=4, per_pixel=True)
        corrected = gc.apply_clear_sky_correction(multiband, models, x_band=2, y_band=4)
        fig = gc.plot_clear_sky_model(corrected, "value", x_band=2, y_band=4)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_per_pixel_models(self):
        multiband = _make_multiband_samples(n_pixels=3, degenerate_pixel=2)
        models = gc.fit_clear_sky_model(multiband, x_band=2, y_band=4, per_pixel=True)
        corrected = gc.apply_clear_sky_correction(multiband, models, x_band=2, y_band=4)
        fig = gc.plot_per_pixel_models(corrected, models, x_band=4, y_band=2)
        self.assertIsInstance(fig, plt.Figure)
        # All 3 pixels must be plotted, not just int(sqrt(3))**2 == 1 of them
        # (goesclouds.md §4.2/§6.3's non-square-count fix).
        pixel_titles = [ax.get_title() for ax in fig.axes if ax.get_title().startswith("Pixel")]
        self.assertEqual(len(pixel_titles), 3)


class TestMcfetchClient(unittest.TestCase):
    """Supplementary tests for the Layer 5 McfetchClient (Chunk 11), beyond
    the numbered T1-T24 plan: allow_download guard, quota-string handling,
    and exact URL/filename construction. No test here touches the network.
    """

    def test_allow_download_false_raises_before_key_read_or_network(self):
        def _raise(*args, **kwargs):
            raise AssertionError("network call attempted despite allow_download=False")

        with tempfile.TemporaryDirectory() as tmpdir:
            client = gc.McfetchClient(
                access_key_path=Path(tmpdir) / "nonexistent_key",
                data_dir=tmpdir,
                allow_download=False,
            )
            with mock.patch("goesclouds.urlretrieve", side_effect=_raise):
                with self.assertRaises(gc.DownloadNotAllowedError):
                    client.download_image(
                        "2015-01-02 03:04:05", band=4, directory=Path(tmpdir) / "q"
                    )
            # The access key must never have been touched.
            self.assertIsNone(client._access_key)

    def test_quota_string_marks_exhausted_and_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "key"
            key_path.write_text("testkey123")
            client = gc.McfetchClient(
                access_key_path=key_path, data_dir=tmpdir, allow_download=True
            )
            directory = Path(tmpdir) / "q"

            def _write_quota_error(url, fname):
                Path(fname).write_text("ERROR: Account over Daily Quota")

            with mock.patch("goesclouds.urlretrieve", side_effect=_write_quota_error) as mock_retrieve:
                with self.assertRaises(RuntimeError):
                    client.download_image("2015-01-02 03:04:05", band=4, directory=directory)
                self.assertEqual(mock_retrieve.call_count, 1)
                today = datetime.date.today().isoformat()
                self.assertIn(today, client._exhausted_dates)
                # The truncated file must have been removed.
                fname = directory / "GOES13_4_2015-01-02T030405.nc"
                self.assertFalse(fname.exists())

                # A second attempt (different timestamp) must short-circuit
                # without calling urlretrieve again.
                with self.assertRaises(RuntimeError):
                    client.download_image("2015-01-02 06:07:08", band=4, directory=directory)
                self.assertEqual(mock_retrieve.call_count, 1)

    def test_url_and_filename_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "key"
            key_path.write_text("testkey123")
            client = gc.McfetchClient(
                access_key_path=key_path, data_dir=tmpdir, allow_download=True
            )
            directory = Path(tmpdir) / "q"
            captured = {}

            def _capture(url, fname):
                captured["url"] = url
                captured["fname"] = Path(fname)
                Path(fname).write_text("not-an-error, pretend NetCDF content padding" * 3)

            with mock.patch("goesclouds.urlretrieve", side_effect=_capture):
                result = client.download_image(
                    "2015-01-02 03:04:05", band=4, directory=directory
                )

            expected_fname = directory / "GOES13_4_2015-01-02T030405.nc"
            self.assertEqual(result, expected_fname)
            self.assertEqual(captured["fname"], expected_fname)
            expected_url = (
                "https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch"
                "?dkey=testkey123&satellite=GOES13&output=NETCDF"
                "&lat=-30+71&size=256+256"
                "&date=20150102&time=03:04:05&coverage=SH&band=4"
            )
            self.assertEqual(captured["url"], expected_url)


if __name__ == "__main__":
    unittest.main()
