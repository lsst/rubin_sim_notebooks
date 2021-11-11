# MAF Tutorial Notebooks

These tutorial notebooks show basic features of MAF, the Rubin Observatory Metrics Analysis Framework for analyzing survey simulations of survey scheduling.

Questions are welcome at [community.lsst.org/c/sci/survey-strategy](https://community.lsst.org/c/sci/survey-strategy) and the [#sims_operations slack channel](https://lsstc.slack.com/archives/C2LTWTP5J).

Previous MAF tutorials, including not only the [tutorial notebooks in sims_MAF-contrib](https://github.com/LSST-nonproject/sims_maf_contrib/tree/master/tutorials) but also those by  Weixiang Yu, Gordon Richards, and Will Clarkson in their [LSST_OpSim](https://github.com/RichardsGroup/LSST_OpSim) repository, inspired the material to be included here. Stylistic elements of these notebooks were guided by the DP0.1 notebooks developed by Melissa Graham and the Rubin Observatory Community Engagement Team.

| Notebook | Section | Description |
|---|---|---|
| 01_Introduction_to_MAF.ipynb | 1 Basic Concepts |  |
| | 1.1 Prerequisites | Links to tutorials on assumed background, including how to use `jupyter` notebooks. |
| | 1.2 Essential elements |  A table of basic MAF architectural elements, and descriptions of what the are and what they are for. |
| | 1.3 A typical MAF workflow | A list of steps typically followed in the use of MAF. | 
| | 2 Notebook preliminaries | Preparation of the notebook, including instructions for installing MAF and importing needed modules. |
| | 3 Example 1: A histogram of airmass of visits in g | |
| | 3.1 Computing the metric | A walk throught the steps is computing a metric an a simulation. | 
| | 3.2 Making a plot | Shows how to plot a mertric with the example of a histogram of numbers of visits in g. | 
| | 3.3 Using metric values and slice points directly | Shows how to get access to the metric data itself as `numpy.array`s for further use by arbitrary python tools. |
| | 3.4 Saved results | Shows where MAF saves the computed metrics, and how to load them back again. |
| | 3.5 Plot customization | Shows how to customize plots created by MAF.| 
| | 4 Example 2: Comparing two simulations | Shows how to compute metrics for a 2nd simulation, and plot a metric for two simulations on the same plot. | 
| | 5 Example 3: Mapping depth in r | Shows how to compute and plot metrics that create maps over the sky. |
| | 6 Plot customization after initial plotting | Shows how to customize MAF figures "after the fact," without recomputing the metrics themselves. |
| | 7 Summary statistics | Introduces the concept of MAF summary statistics, and shows how to compute and get access to them. | 
| | 8 Working with multiple metrics on the same simulation at once | Gives an example of computation of multiple metrics. | 
| | 9 Finding available database columns with which to express constraints, slices, and metrics | Shows how to list simulation data available to MAF. |
| | 10 Finding available metrics and slicers | Shows how to get help on metrics and slicers already part of the MAF |
