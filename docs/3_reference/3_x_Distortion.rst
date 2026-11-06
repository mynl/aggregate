Distortion
==========

:class:`~aggregate.Distortion` (in :mod:`aggregate.spectral`) is the risk-measure
engine: a distortion :math:`g` is an increasing concave function on
:math:`[0, 1]` that re-weights the survival function to produce a coherent
(spectral) risk measure. The standard families (TVaR, Wang, proportional
hazard (PH), dual, bi-TVaR, and the constant cost-of-capital (CCoC) "bent"
distortion) are all constructed from one :class:`Distortion` class.

The module also exposes the calibration and Choquet-pricing helpers that the
:class:`~aggregate.Aggregate` / :class:`~aggregate.Portfolio` pricing methods
build on.

.. currentmodule:: aggregate.spectral

.. autosummary::

   Distortion
   TsallisDistortion
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

Research distortion families
----------------------------

``cll`` (capped loglinear), ``clin`` (capped linear), ``lep`` (leverage equivalent pricing) and ``ly`` (linear yield) are research families. They are built by the same :class:`Distortion` class and called the same way as the production families, ``Distortion('cll', b=0.9)`` in Python and ``distortion D cll 0.9`` in DecL. They are absent from the default :meth:`Distortion.available_distortions` list and from the default calibration set, both of which are ``ccoc``, ``ph``, ``wang``, ``dual`` and ``tvar``. ``available_distortions(pricing=True, strict=False)`` and ``available_distortions(pricing=False)`` include them, and ``calibrate_distortions(..., names=['cll'])`` calibrates one.

The Tsallis distortion family
-----------------------------

``tsallis`` is the Beta(:math:`\alpha`, 1) mixture of TVaRs, :math:`g_\alpha(u) = (\alpha u - u^\alpha)/(\alpha - 1)` with :math:`g_1(u) = u(1 - \log u)`, built as ``Distortion('tsallis', alpha=3)`` in Python and ``distortion D tsallis 3`` in DecL. Its Gini index is :math:`1/(\alpha+1)`, it runs from the maximum at :math:`\alpha \to 0` to the mean at :math:`\alpha \to \infty`, and :math:`\alpha = 2` is ``dual(2)``. Its distorted expectation is the mean plus the cumulative residual Tsallis entropy of order :math:`\alpha`. It is a production family with one parameter and no mass, listed between ``wang`` and ``dual`` in ``available_distortions(pricing=True, strict=False)``, and like the research families it is absent from the default list and the default calibration set: ``calibrate_distortions(..., names=['tsallis'])`` calibrates it. See :class:`TsallisDistortion`.

Distortions with a given Gini index
-----------------------------------

The Gini index :math:`2\int_0^1 g - 1` of a distortion is its :attr:`~Distortion.gini`, twice the margin it puts on a standard uniform, from 0 for the mean to 1 for the maximum. :meth:`Distortion.from_gini` builds the member of a one-parameter family with a given index, ``Distortion.from_gini('tsallis', 0.25)`` for :math:`\alpha = 3`, for ``ccoc``, ``ph``, ``wang``, ``tsallis``, ``dual`` and ``tvar``. A family with more than one parameter raises ``ValueError``. :func:`p_to_parameters` and :func:`consistent_distortions` give the five standard families at one index, and the six with ``extended=True``.

Spectral functions
------------------

.. automodule:: aggregate.spectral
   :exclude-members: Distortion
