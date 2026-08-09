.. 2022-08-03: created new

********************
Aggregate Overview
********************

This chapter explains how ``aggregate`` works. It opens with a working session that introduces the DecL language and the main objects, then covers the :class:`Underwriter` and its recipe library, the four computation pipelines (aggregate, reinsurance, portfolio, and P&L), the automatic grid sizer, numerical artifacts that look like bugs and are not, the layout of the ``info`` string, what changed in 1.0, and how the library is tested.

.. toctree::
   :maxdepth: 3

   2_aggregate_overview/intro-20min
   2_aggregate_overview/underwriter
   2_aggregate_overview/pipeline-aggregate
   2_aggregate_overview/pipeline-reinsurance
   2_aggregate_overview/pipeline-portfolio
   2_aggregate_overview/pipeline-pnl
   2_aggregate_overview/bucket-selection
   2_aggregate_overview/numerical_issues
   2_aggregate_overview/info-strings
   2_aggregate_overview/features
   2_aggregate_overview/tests
