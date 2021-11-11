# MAF Tutorial Notebooks

These tutorial notebooks show basic features of MAF, the Rubin Observatory Metrics Analysis Framework for analyzing survey simulations of survey scheduling.

Questions are welcome at [community.lsst.org/c/sci/survey-strategy](https://community.lsst.org/c/sci/survey-strategy) and the [#sims_operations slack channel](https://lsstc.slack.com/archives/C2LTWTP5J).

Previous MAF tutorials, including not only the [tutorial notebooks in sims_MAF-contrib](https://github.com/LSST-nonproject/sims_maf_contrib/tree/master/tutorials) but also those by  Weixiang Yu, Gordon Richards, and Will Clarkson in their [LSST_OpSim](https://github.com/RichardsGroup/LSST_OpSim) repository, inspired the material to be included here. Stylistic elements of these notebooks were guided by the DP0.1 notebooks developed by Melissa Graham and the Rubin Observatory Community Engagement Team.

| Notebook | Section | Description |
|---|---|---|
| [01_Introduction_to_MAF.ipynb](01_Introduction_to_MAF.ipynb) | 1. Basic Concepts |  |
| | 1.1 Prerequisites | Links to tutorials on assumed background, including how to use `jupyter` notebooks. |
| | 1.2 Essential elements |  A table of basic MAF architectural elements, and descriptions of what the are and what they are for. |
| | 1.3 A typical MAF workflow | A list of steps typically followed in the use of MAF. | 
| | 2. Notebook preliminaries | Preparation of the notebook, including instructions for installing MAF and importing needed modules. |
| | 3. A detailed example | Make a histogram of airmass of visits in g |
| | 3.1 Computing the metric | A walk throught the steps is computing a metric an a simulation. | 
| | 3.2 Making a plot | Shows how to plot a mertric with the example of a histogram of numbers of visits in g. | 
| | 3.3 Using metric values and slice points directly | Shows how to get access to the metric data itself as `numpy.array`s for further use by arbitrary python tools. |
| | 3.4 Saved results | Shows where MAF saves the computed metrics, and how to load them back again. |
| | 3.5 Plot customization | Shows how to customize plots created by MAF.| 
| | 4. Comparing two simulations | Shows how to compute metrics for a 2nd simulation, and plot a metric for two simulations on the same plot. | 
| | 5. A metric that maps the sky | Shows how to compute and plot metrics that create maps over the sky. |
| | 6. Plot customization after initial plotting | Shows how to customize MAF figures "after the fact," without recomputing the metrics themselves. |
| | 7. Summary statistics | Introduces the concept of MAF summary statistics, and shows how to compute and get access to them. | 
| | 8. Working with multiple metrics on the same simulation at once | Gives an example of computation of multiple metrics. | 
| | 9. Finding available database columns with which to express constraints, slices, and metrics | Shows how to list simulation data available to MAF. |
| | 10. Finding available metrics and slicers | Shows how to get help on metrics and slicers already part of the MAF |
| [02_Writing_Metrics.ipynb](02_Writing_Metrics.ipynb) | 1 Notebook Preparation | Imports and other necessary preparation for the notebook |
| | 2. Understanding how metrics are called | Text describing how metrics code gets colled within MAF |
| | 3. Exploring existing metrics | A link to [reference documentation on existing MAF metrics](https://rubin-sim.lsst.io/rs_maf/metricList.html), plus some tricks for exploring the code of these metrics from within jupyter notebooks |
| | 4. An exploratory metric | A test metric that shows how metric code gets called by MAF, what parameters it gets passed, and what it returns |
| | 5. A Workbook for developing metrics | A link to a [metric development notebook](../science/New%20Metric%20Workbook.ipynb) for template for developing new metrics. |
| | 6. Applying a metric to the whole set of visits | Demonstration of how to write a metric that doesn't use slices, but instead operates on all visits at once. |
| | 7. General metrics | Discussion of metrics that have general applicability, and examples of such metrics already provided by MAF. |
| | 8. Metric values | Discussion of metrics that do not return scalars, but rather arbitrary python objects, and an example implementation of such a metric. |
| | 9. Turning complex metrics into scalars with "reduce" functions | A description of "reduce" functions of MAF metrics, which transform metric values with non-scalar values into differenc metrics with scalar values; and an example implementation of such. |
| | 10. Metrics for use with spatial slicers | Discussion of implementing metrics that create maps of values over position on the sky. | 
| | 11. What columns and stackers can I use? | Shows how to list input values that are available to metrics. |
