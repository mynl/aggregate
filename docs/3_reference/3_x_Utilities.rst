Utilities, Moments, Constants, and Config
=========================================

The shared support modules: moment accumulation, FFT and display helpers, the
constant / type definitions, and the user-editable settings.

Moments
-------

:mod:`aggregate.moments` accumulates moments across mixture components and
converts between central, non-central, and (mean, CV, skew) representations.
:class:`~aggregate.moments.MomentAggregator` feeds the ``stats_df`` of every
:class:`~aggregate.Aggregate` / :class:`~aggregate.Portfolio`;
:class:`~aggregate.moments.MomentWrangler` converts a single set of moments.

.. currentmodule:: aggregate.moments

.. autosummary::

   MomentAggregator
   MomentWrangler
   xsden_to_mwrangler
   xsden_to_meancv
   xsden_to_meancvskew

.. autoclass:: aggregate.moments.MomentAggregator

.. autoclass:: aggregate.moments.MomentWrangler

.. automodule:: aggregate.moments
   :exclude-members: MomentAggregator, MomentWrangler

Utilities
---------

:mod:`aggregate.utilities` holds the FFT helpers (``ft`` / ``ift``), the
bucket-rounding rule (``round_bucket``), the noise and display helpers
(``remove_fuzz``, ``qd``, ``mv``), the occurrence exceeding probability curve
(``oep``), the equal-tail window measured off a realized pmf
(``balanced_window``), the shared validation formatter (``explain_validation``),
and Kaplan-Meier estimators.

.. currentmodule:: aggregate.utilities

.. autosummary::

   ft
   ift
   round_bucket
   remove_fuzz
   qd
   mv
   oep
   subsets
   nice_multiple
   balanced_window
   kaplan_meier
   kaplan_meier_np
   agg_help
   explain_validation
   introspect
   silence_warnings

.. automodule:: aggregate.utilities

Constants and types
-------------------

:mod:`aggregate.constants` is a dependency-free leaf holding the validation
flag enum (:class:`~aggregate.constants.Validation`), the defective-distribution
and infinite-variance exception types, the reinsurance view labels, and the
plotting figure defaults.

.. automodule:: aggregate.constants

Configuration
-------------

:mod:`aggregate.config` defines the user-editable settings (discretization,
validation thresholds, labels, …) and the load / reload cascade. See the
:doc:`Aggregate Overview <../2_Aggregate_Overview>` for the file locations and
override order.

.. automodule:: aggregate.config
