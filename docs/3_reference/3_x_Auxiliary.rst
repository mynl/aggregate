Auxiliary Modules
=================

The ``aggregate.extensions`` package was removed at 1.0.0a12. Its useful
contents were promoted to the top-level modules below (``ft``, ``tweedie``,
``pentagon``), absorbed into :mod:`aggregate.pedagogy` (paper / blog figures),
or migrated to the companion PMIR project. Except where noted, these modules are
reached by **submodule import** — they are intentionally *not* re-exported at
the top level (``from aggregate.tweedie import Tweedie``, etc.).

Bivariate aggregates
--------------------

:mod:`aggregate.bivariate` builds the joint law of two aggregates via a copula
and a 2D FFT — for reinsurer-vs-cedent dependency, clash, and joint-tail
analysis.

.. automodule:: aggregate.bivariate

Copulas
-------

:mod:`aggregate.copula` supplies the dependence structures
:mod:`aggregate.bivariate` couples per-claim severities with. The taxonomy
follows the house ``Base<Kind>`` convention (:class:`~aggregate.copula.Copula`
and its ``CopulaClayton`` / ``CopulaFGM`` / ``CopulaGumbel`` /
``CopulaIndependent`` / ``CopulaNormal`` / ``CopulaShuffle`` subclasses).

.. automodule:: aggregate.copula

Tail classification
-------------------

:mod:`aggregate.tail` provides the ordered tail-thickness classification
(bounded → thin → subexponential → heavy) for frequencies, severities, and
aggregates. It backs the bucket/window sizer's "thick tail" gates and the
``tail_df`` / tail-narrative surface.

.. currentmodule:: aggregate.tail

.. autosummary::

   TailClass
   classify_frequency
   classify_severity
   combine
   aggregate_tail_info
   is_thick
   severity_support
   concentration
   build_tail_rows
   tail_frame
   describe_row
   explain_rows

.. automodule:: aggregate.tail

Pentagon
--------

:mod:`aggregate.pentagon` is the accounting authority for the
:math:`(L, P, M, a, Q)` pricing identities — the algebra relating loss,
premium, margin, assets, and capital (and the derived ``lr`` / ``coc`` / ``pq``
ratios).

.. automodule:: aggregate.pentagon

Fourier transform support
-------------------------

:mod:`aggregate.ft` (:class:`~aggregate.ft.FourierTools`) performs direct
numerical inversion of characteristic functions — an independent cross-check on
the FFT convolution path.

.. automodule:: aggregate.ft

Tweedie
-------

:mod:`aggregate.tweedie` provides Tweedie exponential-dispersion models: a
frozen scipy-like distribution plus ``tweedie_convert`` /
``tweedie_density`` for translating between the reproductive and additive
parameterizations.

.. automodule:: aggregate.tweedie

Correlation (Iman–Conover)
--------------------------

:mod:`aggregate.iman_conover` induces a target rank correlation across a set of
marginal samples — the rank-reordering used by the portfolio sample path.

.. automodule:: aggregate.iman_conover

Random number generator
-----------------------

:mod:`aggregate.random_agg` holds the single global NumPy ``Generator`` used by
every stochastic path, so a seeded run is reproducible end to end.

.. automodule:: aggregate.random_agg

Pedagogy
--------

:mod:`aggregate.pedagogy` regenerates the figures and exhibits cited in the
documentation, papers, and blogs. It is **not** part of the core API and is
never imported on the ``import aggregate`` path.

.. automodule:: aggregate.pedagogy
