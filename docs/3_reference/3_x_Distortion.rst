Distortion
==========

:class:`~aggregate.Distortion` (in :mod:`aggregate.spectral`) is the risk-measure
engine: a distortion :math:`g` is an increasing concave function on
:math:`[0, 1]` that re-weights the survival function to produce a coherent
(spectral) risk measure. The standard families — TVaR, Wang, proportional
hazard (PH), dual, bi-TVaR, and the constant cost-of-capital (CCoC) "bent"
distortion — are all constructed from one :class:`Distortion` class.

The module also exposes the calibration and Choquet-pricing helpers that the
:class:`~aggregate.Aggregate` / :class:`~aggregate.Portfolio` pricing methods
build on.

.. currentmodule:: aggregate.spectral

.. autosummary::

   Distortion
   approx_ccoc
   choquet_weights
   ChoquetWeights
   tvar_weights
   p_to_parameters
   consistent_distortions
   convex_distortion
   bagged_distortion
   convex_example

Distortion class
----------------

.. autoclass:: aggregate.spectral.Distortion

Spectral functions
------------------

.. automodule:: aggregate.spectral
   :exclude-members: Distortion
