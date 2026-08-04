Internal Architecture
=====================

This page documents the **internal** module layout created by the 1.0.0a90 to
a95 god-module refactor. None of these modules is part of the public API: they
are reached only through the :class:`~aggregate.Aggregate` /
:class:`~aggregate.Portfolio` classes and the public façades. The structure is
documented here because it is the map of how the library is actually organized.

The pattern
-----------

Two god modules of roughly 5,000 to 10,000 lines (``distributions.py``,
``portfolio.py``) were split behind thin **re-export façades**. The public
classes still import and behave exactly as before; their *bodies* were factored
out two ways:

1. **Concern modules**: a single cross-cutting concern (reinsurance, bucket /
   window sizing, validation, pricing) extracted as a *leaf* module of free
   functions that **take the object** (``agg`` / ``port``) and never import
   ``_aggregate`` / ``_portfolio`` (so there is no cycle, and both classes can
   share one module). The class methods are one-line delegators.
2. **Subsystems**: the ``Portfolio`` body split by *how the joint loss
   distribution is built* (density / sample / common).

The **two-file rule** is the retrospective check on each seam: a routine change
to one concern should land in one file. If it forces editing the class *and* the
concern module together every time, the seam is wrong.

Shared concern modules
----------------------

Reinsurance (Aggregate-only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._reinsurance
   :private-members:

Bucket and window sizing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._bucket_window
   :private-members:

Validation
~~~~~~~~~~

``explain_validation`` (the shared formatter) is documented on the
:doc:`Utilities <3_x_Utilities>` page, where it is re-exported as public.

.. automodule:: aggregate._validation
   :private-members:
   :exclude-members: explain_validation

Pricing
~~~~~~~

.. automodule:: aggregate._pricing
   :private-members:

The insurance builders over the P&L kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The domain-agnostic group-ledger kernel (``Leg`` / ``Group`` / ``PnL`` /
``stack_marginal_pnls``, re-exported at the top level) is public and documented
on the :doc:`Profit and Loss <3_x_PnL>` page. The builders below are the
internal half: they translate insurance programs into sources plus signed
groups, so ``pnl`` is the consolidated net position (always one group, for
every program shape, including reinstatements) and ``xpnl`` the multi-group
step walk (a plain engine gets the trivial one-step walk). The assembly
classifies the occurrence tier {none | guaranteed-cost | reinstatements} and
the aggregate tier {none | guaranteed-cost | feature} independently and
dispatches on the pair; the wrapped engine stays reachable via ``pnl.engine``.
See ``dev/done/plan-yapnl.md``, ``dev/done/plan-pnl-consolidated-xpnl-walk.md``
and ``dev/done/plan-pnl-faces-punchlist.md``.

.. automodule:: aggregate._pnl_builders

Shared mixins
-------------

Three surfaces are shared by classes with no common base
(:class:`~aggregate.Aggregate`, :class:`~aggregate.Portfolio`,
:class:`~aggregate.Severity`, :class:`~aggregate.Distortion`, ``PnL``,
``Copula``), so they are mixins rather than base classes. Following the house
convention they take the ``<Role>Mixin`` suffix, and none defines ``__init__``:
each stays transparent to the host's ``super()`` chain, and a
:class:`~aggregate._labeled.LabeledMixin` host calls ``self._init_labels(...)``
explicitly when it is ready.

DecL round trip
~~~~~~~~~~~~~~~

.. automodule:: aggregate._program

Labels
~~~~~~

.. automodule:: aggregate._labeled

Help and discovery
~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._help

Compute leaves
--------------

The FFT convolution kernel and the shared discrete-grid value type: pure,
testable functions with no dependence on the orchestrating classes.

Aggregate compute kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._aggregate_compute

Out-of-core bivariate compute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The disk-backed sibling of the compute kernel, used by the ``massive``
bivariate path (the ``massive`` extra; ``zarr`` on disk rather than the grid in
memory).

.. automodule:: aggregate._aggregate_compute_massive

Renewal counts
~~~~~~~~~~~~~~

The Sparre-Andersen renewal count distribution behind the DecL
``years`` / ``wait`` / ``dwait`` frequency clauses, and the cepstral
Wiener-Hopf factorization behind ``Aggregate.wiener_hopf``.

.. automodule:: aggregate._renewal

Grid distribution value type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._grid_distribution

Portfolio subsystems
--------------------

The ``Portfolio`` joint-distribution machinery, split by construction method
(see :doc:`3_x_Portfolio`).

Density (independence) path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._portfolio_density

Sample (dependence) path
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._portfolio_sample

Common exeqa numerics
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._portfolio_common

Plotting subsystem
------------------

:mod:`aggregate.plots` is the library's single matplotlib boundary: importing
:mod:`aggregate` does not import matplotlib, which is pulled in lazily on first
plot. Each class's ``.plot()`` is a thin stub into a per-class compositor.

.. automodule:: aggregate.plots
