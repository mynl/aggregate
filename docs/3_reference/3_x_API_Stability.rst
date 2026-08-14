API Stability
=============

What ``aggregate`` promises about each part of its public surface, and what it deliberately does not.

The library has two tiers. Almost everything is **stable**, and changes there follow the usual rules. Two modules are **provisional** in the sense of :pep:`411`, and change there is expected.

The stable core
---------------

These carry the normal promise: from 1.0 onward a documented name keeps its meaning, and a breaking change waits for a major release and is preceded by a deprecation period.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Surface
     - What is covered
   * - :class:`~aggregate.Aggregate`
     - Construction, the public frames (``density_df``, ``summary_df``, ``stats_df``, ``tail_df``, ``reins_stats_df``, ...), the distribution methods (``pdf``, ``cdf``, ``sf``, ``q``, ``tvar``, ...), pricing and validation
   * - :class:`~aggregate.Portfolio`
     - The same, plus allocation, calibration and the diagnostic frames
   * - :class:`~aggregate.PnL`
     - The ledger surface: ``summary_df``, ``economic_df``, ``economic_ratios_df``, ``walk_df``, ``evaluation_df``, ``stats_df``, ``tail_df``, ``evaluate``
   * - :class:`~aggregate.Severity`, :class:`~aggregate.Frequency`
     - The component distributions
   * - :class:`~aggregate.Distortion`
     - The distortion families, calibration and pricing
   * - :class:`~aggregate.Underwriter`, :func:`~aggregate.build`, :func:`~aggregate.qd`
     - The top-level entry points and the knowledge base
   * - The DecL grammar
     - ``src/aggregate/decl.lark``. A program that parses at 1.0 keeps parsing and keeps its meaning
   * - :class:`~aggregate.bivariate.BivariateAggregate`
     - The joint surface

Internal modules, the underscore-prefixed concern modules documented in :doc:`3_x_Internal_Architecture`, are not part of the supported API and never were. They are documented because they are the map of how the library is organized.

.. _provisional-modules:

Provisional modules
-------------------

.. warning::

   :mod:`aggregate.charts` and :mod:`aggregate.exhibits` are **provisional** in the sense of :pep:`411`. They are **not part of the 1.0 API contract**. Their APIs may change in a minor release, with no deprecation period, and code that depends on them may need editing at any minor version.

Both modules were added during the 1.0 cycle as additive side projects, and both ship with 1.0 in a deliberately unfinished state.

**Why they are public anyway.** The alternative was an underscore prefix, which hides them, and hidden code gets no feedback. These modules exist to carry business knowledge that would otherwise leak into every client that draws a chart or lays out a table, so the question of whether they carry the right knowledge can only be answered by people using them. Use them, and report what does not fit. That feedback is how they graduate to stable in a later minor release, which is exactly the path :pep:`411` describes.

**Why they cannot destabilize the release.** Their dependencies point inward: both import from the core, and the core does not import them. Nothing in either module touches an existing class, so no amount of churn there can reach :class:`~aggregate.Aggregate` or :class:`~aggregate.Portfolio` or DecL, and 1.0 ships whether or not either module is finished.

There is exactly one edge into a pre-existing package, and it is worth stating precisely. :mod:`aggregate.plots` gained :func:`~aggregate.plots.plot_chartdoc`, the matplotlib renderer for a chart document, which imports :mod:`aggregate.charts.ir`. That is a new public function in an old package; nothing else in ``plots`` depends on it. Separately, converting a bespoke plot to an emitter plus the generic renderer does edit existing plotting code, one chart at a time, each conversion gated by a before-and-after image diff, and each deferrable past 1.0.

What "provisional" covers
~~~~~~~~~~~~~~~~~~~~~~~~~

For :mod:`aggregate.charts`: the chart IR schema (:class:`~aggregate.charts.ir.ChartDoc` and every dataclass beside it), the field vocabularies, the canonical form and therefore the document hash, the registry, and the set of charts that exist. ``CHART_IR_VERSION`` is how a consumer detects a schema change: pin it and check it. Note that "frozen" on those dataclasses means instances are immutable, and version 1 being closed to additions is a rule about changing the schema in an orderly way. Neither is a promise across releases.

For :mod:`aggregate.exhibits`: exhibit names, block structure, captions, row flags, the format sheets (both their contents and the schema they are written in), the :class:`~aggregate.exhibits.Perspective` vocabulary, and the shape of what :func:`~aggregate.exhibits.build_exhibit` returns.

Explicitly post-1.0, and not gating the release: conversion of the charts the app does not use, full convergence on matplotlib rendering the IR rather than drawing bespoke figures, the ``INSURED`` and ``REINSURER`` perspectives, and any exhibit meta-language.

One thing that is *not* a stability signal: ``greater_tables`` is a plain dependency rather than an optional extra, chosen so the exhibit surface never raises ``ImportError`` on a supported interpreter. That is a fact about installation. It does not move :mod:`aggregate.exhibits` into the 1.0 contract.

Reading a version number
------------------------

Releases are ``MAJOR.MINOR.PATCH``. Against the **stable core**, a patch release fixes bugs, a minor release adds surface and does not break what is documented, and a breaking change waits for a major release after a deprecation period. Against the **provisional modules**, a minor release may break anything.

Numerical results are a separate question from API stability. A correctness fix changes numbers within any release, and such changes are called out in the entry that makes them (``CHANGELOG.md`` in the source distribution) rather than held for a major version. A number that was wrong is not an interface to be preserved.
