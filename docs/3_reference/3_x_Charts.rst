Charts (provisional)
====================

.. warning::

   :mod:`aggregate.charts` is **provisional** in the sense of :pep:`411`: it is **not part of the 1.0 API contract**, and its API may change in a minor release with no deprecation period. It is public, and deliberately so, because feedback is how it graduates to stable. See :doc:`3_x_API_Stability`.

Chart semantics as data. Where a table's meaning is a greater_tables ``TableDoc``, a chart's meaning is a :class:`~aggregate.charts.ir.ChartDoc`: which series, on which axes, at which scales, with which meaningful marks. Color, font, hover, sizing and theme are the renderer's business and have no fields in the IR at all.

An **emitter** reads public frames and ``GridDistribution`` accessors and returns a document. A **renderer** realizes one: :func:`~aggregate.plots.plot_chartdoc` for matplotlib, the app's generic ECharts adapter for the browser. The package never imports matplotlib, and ``tests/test_plots_boundary.py`` enforces that.

Reach it by submodule import; nothing is star exported from the package root::

    from aggregate import build, charts
    a = build('agg Book 100 claims sev lognorm 100 cv 2 poisson')
    charts.available_charts(a)
    doc = charts.chart_severity(a)

Emitters register in ``CHARTS`` as ``functools.singledispatch`` generics, one per chart name, each with an optional availability predicate. :func:`~aggregate.charts.available_charts` derives capability from the dispatch registries plus those predicates, so it cannot go stale.

The registry
------------

.. currentmodule:: aggregate.charts

.. autosummary::

   available_charts
   register_chart
   CHARTS
   chart_distortion
   chart_joint_surface
   chart_reins
   chart_severity

.. automodule:: aggregate.charts

The chart IR
------------

The document schema. ``CHART_IR_VERSION`` is how a consumer detects a schema change; pin it and check it. The dataclasses are frozen, meaning instances are immutable and safe to share, cache and hash, which is a property of an instance and not a promise across releases.

:func:`~aggregate.charts.ir.canonical_dict` and :func:`~aggregate.charts.ir.canonical_json` give a deterministic serialization, and :func:`~aggregate.charts.ir.doc_hash` the first twelve hex characters of its sha256. The same document gives the same bytes and the same hash on any machine on any run, which is what lets a client cache on an ETag.

.. currentmodule:: aggregate.charts.ir

.. autosummary::

   ChartDoc
   Panel
   ChartSeries
   ChartAxis
   Mark
   SurfaceData
   ChartCapabilityError
   canonical_dict
   canonical_json
   doc_hash
   stamp
   CHART_IR_VERSION
   SUPPORT_KINDS

.. automodule:: aggregate.charts.ir

Rendering
---------

.. currentmodule:: aggregate.plots

:func:`~aggregate.plots.plot_chartdoc` is the generic matplotlib renderer: it draws any document the schema can express, choosing stems, steps or a line from each series' declared support and the room each atom gets. A renderer asked strictly for a panel kind it cannot realize raises :class:`~aggregate.charts.ir.ChartCapabilityError` rather than approximating silently; matplotlib and a 3-D surface is the live case, where the honest non-strict answer is a labeled 2-D projection.

.. autosummary::

   plot_chartdoc
