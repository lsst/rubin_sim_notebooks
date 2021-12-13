import sys
import os
from os.path import splitext, basename
from argparse import ArgumentParser

import numpy as np

from rubin_sim import maf


def make_open_shutter_bundle(run_name):
    """Create a MetricBundle to compute open shutter time

    Parameters
    ----------
    run_name : `str`
        the run name

    Returns
    -------
    bundle : `rubin_sim.maf.metricBundles.MetricBundle`
        the instance of MetricBundle
    """
    constraint = ""
    plotDict = {}
    slicer = maf.UniSlicer()
    metric = maf.OpenShutterFractionMetric(
        slewTimeCol="slewTime",
        expTimeCol="visitExposureTime",
        visitTimeCol="visitTime",
    )
    summary_metrics = [maf.IdentityMetric()]
    bundle = maf.MetricBundle(
        metric,
        slicer,
        constraint,
        summaryMetrics=summary_metrics,
        runName=run_name,
        plotDict=plotDict,
    )

    return bundle


def make_depth_map_bundle(run_name, band, footprint_area):
    """Create a MetricBundle to compute open shutter time

    Parameters
    ----------
    run_name : `str`
        the run name
    band : `str`
        the filter
    footprint_area : `float`
        the area of the full footprint, in square degrees

    Returns
    -------
    bundle : `rubin_sim.maf.metricBundles.MetricBundle`
        the instance of MetricBundle
    """
    constraint = f"filter = '{band}'"
    plotDict = {}
    slicer = maf.HealpixSlicer(nside=64)
    metric = maf.Coaddm5Metric()

    # Summary stats for depth over best footprint area
    summary_metrics = [
        maf.AreaSummaryMetric(
            area=footprint_area,
            reduce_func=np.min,
            decreasing=True,
            metricName="Min",
        ),
        maf.AreaSummaryMetric(
            area=footprint_area,
            reduce_func=np.median,
            decreasing=True,
            metricName="Median",
        ),
        maf.AreaSummaryMetric(
            area=footprint_area,
            reduce_func=np.max,
            decreasing=True,
            metricName="Max",
        ),
    ]

    bundle = maf.MetricBundle(
        metric,
        slicer,
        constraint,
        summaryMetrics=summary_metrics,
        runName=run_name,
        plotDict=plotDict,
    )

    return bundle


def make_airmass_bundle(run_name):
    """Create a MetricBundle to compute the distribution of airmasses.

    Parameters
    ----------
    run_name : `str`
        the run name

    Returns
    -------
    bundle : `rubin_sim.maf.metricBundles.MetricBundle`
        the instance of MetricBundle
    """
    constraint = ""
    plotDict = {}
    slicer = maf.OneDSlicer(
        sliceColName="airmass", binMin=1.0, binMax=2.5, binsize=0.05
    )
    metric = maf.CountMetric(col="airmass")

    # produces list of metrics with mean, median, RMS, etc.
    summary_metrics = maf.extendedSummary()

    bundle = maf.MetricBundle(
        metric,
        slicer,
        constraint,
        summaryMetrics=summary_metrics,
        runName=run_name,
        plotDict=plotDict,
    )
    return bundle


def sample_batch(run_name="opsim", bands=("g", "i"), footprint_area=18000):
    """Generate an example set of metrics.

    Parameters
    ----------
    run_name : `str`
        the run name
    bands : `str`
        the filters for which to create depth maps
    footprint_area: `float`
        the reference footprint area, in square degrees

    Returns
    -------
    bundle_dict : `dict` [`str`, `rubin_sim.maf.metricBundles.MetricBundle`]
        A dictionary of the metric bundles in the batch.
    """
    metric_bundles = []

    # Open shutter fraction on each night
    bundle = make_open_shutter_bundle(run_name)
    metric_bundles.append(bundle)

    # Depth map by filter
    for band in bands:
        bundle = make_depth_map_bundle(run_name, band, footprint_area)
        metric_bundles.append(bundle)

    # Hour Angle distribution
    bundle = make_airmass_bundle(run_name)
    metric_bundles.append(bundle)

    # Turn our list of metric bundles into a dictionary
    bundle_dict = maf.metricBundles.makeBundlesDictFromList(metric_bundles)

    return bundle_dict


def compute_metrics(
    opsim_runs, opsim_run_fnames, data_dir, batch_factory, batch_name=""
):
    """Run batches of metrics on a collection of runs.

    Parameters
    ----------
    opsim_runs : `iterable` ['str']
        A collection of name for opsim runs
    opsim_run_fnames : `dict` [`str`, `str`]
        Keys are run names, values of the file name of the opsim output.
    data_dir : `str`
        Directory into which to make subdirectories with output
    batch_factory : `Callable`
        A function that takes a run name and returns bundle dict.
    batch_name : `str`
        The name of the batch

    Returns
    -------
    batches : `dict` [`str`, `dict`]
        A dictionary for dictionaries of MAF bundles with the generated
        metrics, slices, etc.
    """
    batches = {}

    for run_name in opsim_runs:
        opsim_db = maf.OpsimDatabase(opsim_run_fnames[run_name])

        # Follow the opsim team practice and make separate
        # out_dir for each run, and put a results database
        # there.
        out_dir = os.path.join(data_dir, run_name, batch_name)
        results_db = maf.ResultsDb(outDir=out_dir)

        this_batch = batch_factory(run_name)
        this_batch.update(maf.filtersPerNight(runName=run_name))
        bundle_group = maf.MetricBundleGroup(
            this_batch, dbObj=opsim_db, outDir=out_dir, resultsDb=results_db
        )
        bundle_group.runAll()
        bundle_group.plotAll()
        results_db.close()
        maf.writeConfigs(opsim_db, out_dir)
        opsim_db.close()

        batches[run_name] = this_batch

    return batches


def main(*args, **kwargs):
    """Driver for running an example metric bundle group.

    Parameters
    ----------
    *args : `str`
        Passed to `argparse.ArgumentParser.parse_args`
    **kwargs :
        Passed to `argparse.ArgumentParser.parse_args`

    If no parameters are passed, `parse_args` takes them from the
    command line. They are useful here to support testing.
    """
    parser = ArgumentParser(description="Run sample MAF batches in bulk")
    parser.add_argument(
        "out_dir", type=str, help="base output directory for results"
    )
    parser.add_argument(
        "in_files", type=str, nargs="*", help="opsim files to process"
    )
    args = parser.parse_args(*args, **kwargs)

    out_dir = args.out_dir
    in_files = args.in_files

    # Extract run names from the opsim database file names
    opsim_runs = [splitext(basename(fn))[0] for fn in args.in_files]
    opsim_run_fnames = dict(zip(opsim_runs, in_files))

    # Actually compute the metrics
    compute_metrics(
        opsim_runs, opsim_run_fnames, out_dir, sample_batch, "sample"
    )

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
