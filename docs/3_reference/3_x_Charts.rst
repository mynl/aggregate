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

A document also declares the **readings** each quantity admits, which is a fact about the quantity and not about the drawing: a log reading of a heavy tail is meaningful, a log reading of a distortion's unit square is not. :attr:`~aggregate.charts.ir.ChartAxis.scales` lists the scales an axis may be read on and :attr:`~aggregate.charts.ir.ChartAxis.full_range` the extent it may be zoomed out to, both alongside the default reading rather than replacing it; :attr:`~aggregate.charts.ir.ChartAxis.reciprocal_of` pairs a probability axis with its return-period reading, computed by the map in ``meta['return_period_map']`` (see :data:`~aggregate.charts.ir.RETURN_PERIOD_MAPS`); and :attr:`~aggregate.charts.ir.Panel.kinds` lists the forms a panel may take, so a joint density read flat or in relief is one document declaring two realizations rather than two charts to keep in step. A reader that ignores all four draws the default reading, which is correct and complete, which is why they landed without a version bump.

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
   RETURN_PERIOD_MAPS

.. automodule:: aggregate.charts.ir

Rendering
---------

.. currentmodule:: aggregate.plots

:func:`~aggregate.plots.plot_chartdoc` is the generic matplotlib renderer: it draws any document the schema can express, choosing stems, steps or a line from each series' declared support and the room each atom gets. A renderer asked strictly for a panel kind it cannot realize raises :class:`~aggregate.charts.ir.ChartCapabilityError` rather than approximating silently; matplotlib and a 3-D surface is the live case, where the honest non-strict answer is a labeled 2-D projection, and a panel offering a realization the renderer does draw natively gets that one instead, with nothing to confess.

Its ``log``, ``full_range``, ``return_period`` and ``kind`` switches select among the readings a document declares. Each acts on every axis or panel that declares the reading and on no other, which is the same surfacing rule the browser applies to its control strip, so a document that declares nothing draws its one reading whatever it is asked for and a caller never has to know which chart it is holding.

.. autosummary::

   plot_chartdoc
