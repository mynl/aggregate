Internal Architecture
=====================

This page documents the **internal** module layout created by the 1.0.0a90–a95
god-module refactor. None of these modules is part of the public API — they are
reached only through the :class:`~aggregate.Aggregate` / :class:`~aggregate.Portfolio`
classes and the public façades — but the structure is documented here because
it is the map of how the library is actually organized.

The pattern
-----------

Two ~5–10k-line god modules (``distributions.py``, ``portfolio.py``) were split
behind thin **re-export façades**. The public classes still import and behave
exactly as before; their *bodies* were factored out two ways:

1. **Concern modules** — a single cross-cutting concern (reinsurance, bucket /
   window sizing, validation, pricing) extracted as a *leaf* module of free
   functions that **take the object** (``agg`` / ``port``) and never import
   ``_aggregate`` / ``_portfolio`` (so there is no cycle, and both classes can
   share one module). The class methods are one-line delegators.
2. **Subsystems** — the ``Portfolio`` body split by *how the joint loss
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

The P&L kernel and its insurance builders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The domain-agnostic group-ledger kernel (``Leg`` / ``Group`` / ``PnL`` /
``stack_marginal_pnls``, re-exported at the top level) and the DecL builders
that translate insurance programs into sources plus signed groups. See
``dev/done/plan-yapnl.md``.

.. automodule:: aggregate._pnl

.. automodule:: aggregate._pnl_builders

Compute leaves
--------------

The FFT convolution kernel and the shared discrete-grid value type — pure,
testable functions with no dependence on the orchestrating classes.

Aggregate compute kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aggregate._aggregate_compute

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
