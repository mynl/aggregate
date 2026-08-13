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

:func:`~aggregate.charts.build_chart_doc` is the entry point: it resolves the registry entry, checks availability, dispatches on the type and stamps the content hash, so an emitter is left with nothing to do but the semantics. It is a module function taking the object as an argument, mirroring ``exhibits.build_exhibit``, so no first-class class gains a method. :func:`~aggregate.charts.available_charts` answers what *can* be drawn for an object, and :func:`~aggregate.charts.primary_chart` which of those is the object's **own** picture: an aggregate's severity is a component of it and its reinsurance is a view of it, so neither is what a landing page should draw.

.. autosummary::

   build_chart_doc
   available_charts
   primary_chart
   register_chart
   CHARTS
   ChartEntry
   chart_agg
   chart_distortion
   chart_joint_surface
   chart_pnl
   chart_reins
   chart_severity

.. automodule:: aggregate.charts

The chart IR
------------

The document schema. ``CHART_IR_VERSION`` is how a consumer detects a schema change; pin it and check it. The dataclasses are frozen, meaning instances are immutable and safe to share, cache and hash, which is a property of an instance and not a promise across releases.

:func:`~aggregate.charts.ir.canonical_dict` and :func:`~aggregate.charts.ir.canonical_json` give a deterministic serialization, and :func:`~aggregate.charts.ir.doc_hash` the first twelve hex characters of its sha256. The same document gives the same bytes and the same hash on any machine on any run, which is what lets a client cache on an ETag.

:func:`~aggregate.charts.ir.load_chart_doc` is the way back, so a fetched document draws through :func:`~aggregate.plots.plot_chartdoc` like any other. The round trip is exact, ``doc_hash(load_chart_doc(canonical_dict(doc))) == doc.hash``, and it is the library's to own rather than each client's: the canonical form omits every field equal to its default, so a hand written reader is this build's default table written down a second time, and the reader is also the one place a wire document's ``ir_version`` is negotiated.

A document also declares the **readings** each quantity admits, which is a fact about the quantity and not about the drawing: a log reading of a heavy tail is meaningful, a log reading of a distortion's unit square is not. :attr:`~aggregate.charts.ir.ChartAxis.scales` lists the scales an axis may be read on and :attr:`~aggregate.charts.ir.ChartAxis.full_range` the extent it may be zoomed out to, both alongside the default reading rather than replacing it; :attr:`~aggregate.charts.ir.ChartAxis.reciprocal_of` pairs a probability axis with its return-period reading, computed by the map in ``meta['return_period_map']`` (see :data:`~aggregate.charts.ir.RETURN_PERIOD_MAPS`); :attr:`~aggregate.charts.ir.ChartAxis.complement_of` pairs it with its reflected reading, the map ``v`` to ``1 - v``, which turns a non-exceeding probability into the exceedance and so a quantile function into the survival function; and :attr:`~aggregate.charts.ir.Panel.kinds` lists the forms a panel may take, so a joint density read flat or in relief is one document declaring two realizations rather than two charts to keep in step. A reader that ignores all five draws the default reading, which is correct and complete, which is why they landed without a version bump.

Both pairings are declared the same way, an undrawn axis in ``axes`` naming the drawn one it is an alternative reading of, and each carries its own label, scales and window. That is why the reflected reading is a paired axis rather than a flag: a survival axis is log readable where the non-exceeding probability it reflects is not, and only a separate axis has anywhere to say so. An axis carries at most one pointer, because the two readings compose in the renderer rather than by chaining declarations: ``complement(v) = reciprocal(1 - v)``, so an axis already read reflected takes the plain reciprocal for its return period.

A grid panel's data is a :class:`~aggregate.charts.ir.SurfaceData`, and it carries its two lattices as an origin, a step and a count rather than as coordinate arrays. That is normative and the reason is not size: an aggregate lives on a lattice by construction and a power-of-two block reduction leaves one, so sending ``x0 + i * dx`` sends a derived quantity and invites a reader to wonder whether it might not be uniform this time. Everything a consumer does off the grid divides by a constant step, and against arrays that division is an assumption the format permits an emitter to violate. Beside the lattices the surface says what a coordinate names (:data:`~aggregate.charts.ir.SURFACE_EDGES`), what the grid is a reduction *of* (the fine bucket size and the block factor per axis), what window was kept and what share of the mass is in it, the exact marginals on the display lattice, and the means from the fine lattice, which is the reference a consumer checks its own arithmetic against rather than integrating the picture.

The z values travel twice: as the plain nested array, and as a :class:`~aggregate.charts.ir.SurfaceZBlock`, base64 of little-endian bytes under a declared dtype (:data:`~aggregate.charts.ir.SURFACE_DTYPES`), built and read with :func:`~aggregate.charts.ir.encode_z_block` and :func:`~aggregate.charts.ir.decode_z_block`. Naming the dtype is what lets the default change without a format change. The default is float32 and not float64, which points the other way from intuition: the low mantissa bits of an FFT-built density are genuine digits that no compressor touches, so a float64 payload measures about twice the size of the JSON text it replaces.

Every human-facing string in a document is plain text, never markup in any renderer's language, and carries **both** forms: :attr:`~aggregate.charts.ir.ChartDoc.tex` is a total lookup from the plain string to its typeset form, a plain word mapping to itself. The analogy is alt text in HTML: you write both because they serve different consumers, and you do not make one consumer guess. matplotlib reads the typeset form, the browser reads the plain one, and neither derives one from the other. A missing entry is an emitter bug, so emitters build the map with :func:`~aggregate.charts.ir.complete_tex`, which fills the identities, and the contract is checked as a set difference against :func:`~aggregate.charts.ir.human_strings`. The plain form does not have to be ASCII: Unicode carries most actuarial labels honestly, and the dual distortion ``ǧ(s)`` is the working example.

.. currentmodule:: aggregate.charts.ir

.. autosummary::

   ChartDoc
   Panel
   ChartSeries
   ChartAxis
   Mark
   SurfaceData
   SurfaceZBlock
   ChartCapabilityError
   canonical_dict
   canonical_json
   load_chart_doc
   complete_tex
   human_strings
   doc_hash
   encode_z_block
   decode_z_block
   stamp
   CHART_IR_VERSION
   SUPPORT_KINDS
   SURFACE_DTYPES
   SURFACE_EDGES
   RETURN_PERIOD_MAPS

.. automodule:: aggregate.charts.ir

Rendering
---------

.. currentmodule:: aggregate.plots

:func:`~aggregate.plots.plot_chartdoc` is the generic matplotlib renderer: it draws any document the schema can express, choosing stems, steps or a line from each series' declared support and the room each atom gets. A renderer asked strictly for a panel kind it cannot realize raises :class:`~aggregate.charts.ir.ChartCapabilityError` rather than approximating silently; matplotlib and a 3-D surface is the live case, where the honest non-strict answer is a labeled 2-D projection, and a panel offering a realization the renderer does draw natively gets that one instead, with nothing to confess.

Its ``log``, ``full_range``, ``return_period`` and ``kind`` switches select among the readings a document declares. Each acts on every axis or panel that declares the reading and on no other, which is the same surfacing rule the browser applies to its control strip, so a document that declares nothing draws its one reading whatever it is asked for and a caller never has to know which chart it is holding.

.. autosummary::

   plot_chartdoc
