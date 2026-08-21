.. _pipeline exhibits and charts:

The Exhibit and Chart Publishing Pipeline
=========================================

This section describes the publishing surface: how a computed object becomes a document a consumer can render. It is the companion to :ref:`pipeline aggregate`, :ref:`pipeline portfolio`, :ref:`pipeline reinsurance` and :ref:`pipeline pnl`, and it consumes what all four produce. Those four sections are about getting the numbers right. This one is about who is allowed to say what the numbers mean, and the answer is that the library says it, once, in a form no particular renderer owns.

Two packages implement it, :mod:`aggregate.exhibits` for tables and :mod:`aggregate.charts` for pictures. Both are **provisional in the sense of** :pep:`411` and outside the 1.0 API contract; see :doc:`/3_reference/3_x_API_Stability`. The API details are in :doc:`/3_reference/3_x_Exhibits` and :doc:`/3_reference/3_x_Charts`. What follows is the pipeline and the catalogue: what is published, by whom, and with what options.

.. _exhibits charts one shape:

One shape, twice
----------------

The two packages are deliberately the same machine.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Stage
     - Exhibits
     - Charts
   * - Registry
     - ``EXHIBITS``, name to (generic function, availability predicate)
     - ``CHARTS``, name to ``ChartEntry(emitter, predicate, primary)``
   * - Dispatch
     - ``functools.singledispatch`` on the object's type
     - ``functools.singledispatch`` on the object's type
   * - Capability
     - ``exhibits.available_exhibits(obj)``
     - ``charts.available_charts(obj)``, plus ``charts.primary_chart(obj)``
   * - Build
     - ``exhibits.build_exhibit(obj, name, perspective)``
     - ``charts.build_chart_doc(obj, name, **options)``
   * - Document
     - a greater_tables ``TableDoc`` per block
     - one :class:`~aggregate.charts.ir.ChartDoc`
   * - Content hash
     - ``Exhibit.hash``, sha256 over the block hashes, 12 hex
     - ``ChartDoc.hash``, sha256 of the canonical JSON, 12 hex

Three properties matter more than the mechanics.

**Capability is derived, never declared twice.** Whether an object can serve an exhibit is read off the dispatch registry (an MRO hit means a builder exists) and the exhibit's own availability predicate. Nothing maintains a second list that could disagree with the first, so a registration added by app or user code shows up in ``available_exhibits`` the moment it is made, and an exhibit that cannot be built is never offered.

**Dependencies point inward.** Both packages import from the core; the core does not import them. There is exactly one edge the other way, :func:`~aggregate.plots.plot_chartdoc` in :mod:`aggregate.plots`, which is a new public function in an old package that nothing else in that package depends on. ``aggregate.charts`` never imports matplotlib, and ``tests/test_plots_boundary.py`` enforces it.

**The document is the deliverable.** An exhibit block is presentation-ready table IR and a chart document is chart semantics as data. Neither carries a figure, an axes object, a style, a color or a font. matplotlib renders one of them and a browser renders the other, from the same bytes, and neither is privileged. pandas appears only on the way in, as the frame stage, which is itself useful on its own to a caller who wants the numbers rather than a table.

.. _exhibits pipeline:

Exhibits
--------

Exhibits defined
~~~~~~~~~~~~~~~~~~~

An exhibit is a titled envelope over one or more **blocks**, each block one frame rendered as a greater_tables ``TableDoc`` with its caption, formats and row flags. Most exhibits carry one block. A few carry two or three, because the thing being reported is genuinely more than one table: reinsurance is a layering store and a stage summary, and a P&L ratio view separates amounts from ratios so that no column mixes two units.

The placement test for what belongs here: if deleting the web app would destroy knowledge an actuary would want in a notebook, that knowledge belongs in the library. A caption saying what a frame is, emphasis on the rows that failed a validation gate, and the decision to drop raw noncentral moments from a business view are all knowledge, not decoration.

Two stages, and only the second costs anything. :func:`~aggregate.exhibits.exhibit_frames` is pure pandas and returns ``(block_name, frame, spec_kwargs)`` triples. :func:`~aggregate.exhibits.build_exhibit` converts those to IR, and is the only place greater_tables is imported, at the point of use.

How each column reads comes from the **format sheets**, two YAML files shipped with the package and applied by :func:`~aggregate.exhibits.build_exhibit` after relabeling: ``formats-raw.yaml`` for the default reading of every named column and ``formats-insurer.yaml`` for the entries where the business reading differs. They are overridable from ``~/.aggregate`` and the working directory, nearest winning, the same rule a user ``.agg`` database follows. See :doc:`/3_reference/3_x_Exhibits`.

Every served block carries **both** its formatted strings and its raw values, so a cell arrives as ``{'text': '17.50', 'raw': 17.5000001}``. The formatted string is this library's reading of a number and the raw value is the number, and a document that ships only the reading cannot be sorted numerically, downloaded at full precision, or drawn interactively at all. It costs about 1.5 times the payload, measured across every exhibit on an ``Aggregate`` and a ``PnL``, and it is a library default rather than a caller option because no consumer can put back what the document threw away. How many rows to ship **is** the caller's question, ``build_exhibit(..., max_rows=...)``, and a truncated block says so in its own notes.

Perspective, and the INSURER default rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~aggregate.exhibits.Perspective` names who is reading. The enum has four members, ``RAW``, ``INSURED``, ``INSURER`` and ``REINSURER``, and exactly two are implemented. ``RAW`` serves the frame with no business translation, though since 1.0.0a226 it does carry a caption saying what the frame is and the column formats for units a frame cannot carry itself. ``INSURER`` is the seller of insurance and buyer of reinsurance, and is where the business reading lives: what the frame *means*, over ``RAW``'s what it *is*. ``INSURED`` and ``REINSURER`` are declared vocabulary so the enum does not churn when their implementations arrive; asking for either raises rather than guessing.

The governing rule is that **INSURER equals RAW unless an override is registered for that (exhibit, class) pair**. The rule keeps the generic path total, so a new exhibit is useful the moment its raw registration exists, and it makes the last column of the catalogue below the complete list of business translation that exists today. A blank there means the two perspectives serve identical bytes.

What is published
~~~~~~~~~~~~~~~~~

.. Provenance of the table below. Transcribed from ``dev/summary-exhibits-and-charts.md`` section 1, and verified against the live registries at 1.0.0a248 by introspection, not from memory. Three code sites are the source of truth: the passthrough manifest at the foot of ``src/aggregate/exhibits/__init__.py`` (name, title, source frame, classes, caption), the ``EXHIBITS`` registry and its availability predicates in ``exhibits/_core.py``, and the per-class ``insurer`` overrides in ``exhibits/_aggregate.py``, ``_portfolio.py``, ``_pnl.py``, ``_bivariate.py`` and ``_distortion.py``. Check any row in one line with ``[n for n, _ in exhibits.available_exhibits(obj)]``. To update: edit the summary in ``dev/`` first, since it is the reference snapshot the app team reads, then re-transcribe here. It is a ``list-table`` and not a grid or simple table precisely so it can be edited in place: a grid table has to be redrawn to change one cell, which is why the ones in this documentation go stale.

.. list-table:: The exhibit registry
   :header-rows: 1
   :widths: 12 11 20 17 12 28

   * - Exhibit
     - Title
     - Published by
     - Blocks (source frames)
     - Available when
     - INSURER changes
   * - ``summary``
     - Summary
     - Aggregate, Portfolio, PnL, Distortion, BivariateAggregate
     - 1: ``summary_df``
     - always
     - Aggregate, Portfolio: caption, total and subtotal row flags, measure formats
   * - ``tail``
     - Return periods
     - Aggregate, Portfolio
     - 1: ``tail_df``
     - updated
     - both: caption, 1-in-200 and 1-in-250 row emphasis
   * - ``stats``
     - Statistics
     - Aggregate, Portfolio, PnL, Distortion, BivariateAggregate
     - 1: ``stats_df``
     - always
     - Aggregate, Portfolio, PnL: drops the raw noncentral rows ``ex1``, ``ex2``, ``ex3`` (26 rows becomes 23), caption
   * - ``validation``
     - Validation
     - Aggregate, Portfolio, PnL, Distortion, BivariateAggregate
     - 1: ``validation_df``
     - always
     - Aggregate, Portfolio, Distortion, BivariateAggregate: caption, emphasis on rows failing the object's ``validation_eps`` gate
   * - ``reins``
     - Reinsurance
     - Aggregate, Portfolio
     - RAW 2: ``reins_stats_df``, ``reins_summary_df``
     - cedes and updated
     - Aggregate restructures into 3 blocks, ``reins_layer_terms`` and ``reins_layer_moments`` with layers down the rows, then the summary; Portfolio (no layer axis) drops the raw noncentral rows and flags the summary rows
   * - ``economic``
     - Economics
     - PnL
     - 1: ``economic_df``
     - always
     - **abbreviates to 4 columns**, ``EX``, ``SD``, ``CV`` and the adverse tail state (``κ01``, or ``P01`` on a marginal ladder); caption switches on whether the ladder is a kappa scenario or a marginal ``P`` ladder, ledger row flags, measure formats
   * - ``economic_ratios``
     - Economic ratios
     - PnL
     - RAW 2: ``economic_ratios_df``, ``legs_df``
     - always
     - **restructures into 3 blocks**: ``amounts`` (P, L, E, M), ``ratios`` (LR, ER, CR, ``E_*``, shares), ``legs``, so no column mixes two units
   * - ``economic_waterfall``
     - Economic waterfall
     - PnL
     - 2: ``walk_df``, ``evaluation_df``
     - multi-step walk (``_tower``)
     - none
   * - ``dependency``
     - Dependency
     - BivariateAggregate
     - 2: ``dependency_df``, ``axis_support_df``
     - updated
     - none
   * - ``bs_window``
     - Grid sizing
     - Aggregate, Portfolio, BivariateAggregate
     - 1: ``bs_window_df``
     - updated
     - none
   * - ``sharpen``
     - Grid probe
     - Aggregate, Portfolio
     - 1 raw, 2 insurer: ``score_grid``, ``sharpen_df``
     - probed (``sharpen_df`` present)
     - the score grid, which is ``score`` unstacked by ``d_log2``
   * - ``tail_behavior``
     - Tail behavior
     - Aggregate, Portfolio
     - 1: ``tail_behavior_df``
     - updated
     - none

Read by class, which is the question a landing page actually asks:

Aggregate
    ``summary``, ``tail``, ``stats``, ``validation``, ``bs_window``, ``tail_behavior``, plus ``reins`` when it cedes and ``sharpen`` once the grid probe has run. Six to eight.
Portfolio
    The same, per unit plus the total.
PnL
    ``summary``, ``stats``, ``validation``, ``economic``, ``economic_ratios``, plus ``economic_waterfall`` on a walk.
BivariateAggregate
    ``summary``, ``stats``, ``validation``, ``dependency``, ``bs_window``.
Distortion
    ``summary``, ``stats``, ``validation``.
Severity
    None. It publishes a chart but no exhibit, which is a gap rather than a decision.

Adding a new Exhibit
~~~~~~~~~~~~~~~~~~~~~

Adding an exhibit is one of three sizes, and the first is the common case.

A passthrough over an existing frame is a **manifest line**, :func:`~aggregate.exhibits.register_simple_exhibit` at the foot of ``exhibits/__init__.py``, giving the name, title, source frame attribute, the classes, an availability predicate and a caption. It registers no insurer override, so both perspectives serve the same table with no further code. Calling it twice for one name extends the exhibit to more classes rather than replacing it, which is also how one exhibit carries a different caption per class: a single sentence cannot describe ``summary_df`` on an ``Aggregate`` (count risk, severity, total loss) and on a ``PnL`` (consideration, obligation, margin) at once, and one that tries is worse than none.

A new business reading of an existing exhibit is a **function in the class module**, ``<exhibit>.insurer.register(Cls)``, taking the raw blocks and returning translated ones. Only that (exhibit, class) pair changes.

A wholly new multi-block exhibit declares its generic function in ``_core`` and registers builders in the class modules.

.. _charts pipeline:

Charts
------

Chart documents defined
~~~~~~~~~~~~~~~~~~~~~~~~

A chart emitter returns a :class:`~aggregate.charts.ir.ChartDoc`: axes, panels, series, marks and chart-level ``meta``, carrying semantics only. The boundary rule is semantics against realization, not data against display. Log or linear is statistical meaning, so ``scale`` lives on the axis; a sequential color ramp is presentation, so it has no field at all. There is no renderer passthrough of any kind: a need the schema cannot express changes the schema visibly, or the chart stays bespoke.

Documents are deterministic. The canonical form gives the same bytes and the same 12 hex hash on any machine on any run, which is what lets a client cache on an ETag. ``CHART_IR_VERSION`` is how a consumer detects a schema change; the current version is 2.

Declared readings
~~~~~~~~~~~~~~~~~

Where an exhibit has a perspective, a chart has **declared readings**, and this is the idea that makes one generic renderer possible.

An axis names every scale it may honestly be read on (``scales``) and whether a zoom-out exists (``full_range`` alongside ``suggested_range``). A probability axis may name a paired return-period axis (``reciprocal_of``), computed by the map in ``meta['return_period_map']``, and a paired reflected axis (``complement_of``), the map ``v`` to ``1 - v``: a non-exceeding probability reflected is the exceedance, so the quantile function drawn against it is the survival function, and reflecting both axes of a unit square gives the dual distortion. A paired axis carries its own label, scales and window, which is how the survival axis says it is log readable where the non-exceeding probability is not. A panel may declare itself ``invertible``, and may name what it is called inverted, since a Lee diagram with its axes exchanged is the distribution function and naming that is the library's job. A panel may offer more than one realization (``kinds``), so a joint density read flat or in relief is one document declaring two readings rather than two chart entries to keep in step by hand.

The two paired readings compose without a special case, because ``complement(v) = reciprocal(1 - v)``: the 'complement' map *is* reflect-then-reciprocal, so an axis already read reflected takes the plain reciprocal. On a loss that redraws the return-period curve the switch draws by itself; on a signed result, whose map is the reciprocal because the adverse tail is the low one, it reads the upside tail's return period instead of the shortfall's.

Which readings a quantity admits is a fact about the quantity and not about the drawing. A log reading of a heavy tail is meaningful; a log reading of a distortion's unit square is not. The renderer's switches therefore act on every axis or panel that declares the reading and on nothing else, so a document that declares nothing draws its one reading whatever it is asked for, and a caller never needs to know which chart it is holding.

Chart by chart
~~~~~~~~~~~~~~

Chart-level facts, one line each. Options are semantic arguments to the emitter, never renderer settings.

``agg``
    On Aggregate, and its primary chart. Available when updated. Option ``xmax``. Entry point :meth:`Aggregate.plot` (``xmax``, ``log``, ``full_range``, ``reflect``, ``return_period``, ``invert``).
``port``
    On Portfolio, primary. Updated. Option ``xmax``. Entry point :meth:`Portfolio.plot` (``xmax``, ``log``, ``full_range``).
``pnl``
    On PnL, primary. Needs a result. No options. Entry point :meth:`PnL.plot` (``log``, ``full_range``, ``reflect``, ``return_period``, ``invert``).
``severity``
    On Severity, primary. Always. Option ``n``, default 512 quantile-spaced points. Entry point :meth:`Severity.plot` (``n``, ``log``, ``full_range``, ``reflect``, ``return_period``, ``invert``).
``reins``
    On Aggregate, primary for nothing, because it is a view of a book rather than the book's own picture. Needs an occurrence program. No options. Entry point :meth:`Aggregate.reins_occ_plot` (``log``, ``full_range``, ``reflect``, ``return_period``, ``invert``).
``distortion``
    On Distortion, primary. Always. Option ``dual``. Entry point :meth:`Distortion.plot` (``dual``, ``reflect``, ``ax``).
``envelope``
    On Bounds, primary. Always. Options ``n_resamples`` (bracketing curves inside the band, each carrying its weight as ``ChartSeries.value``) and ``n`` (curve points, default 1001). Entry point ``Bounds.plot_envelope(n_resamples, reflect)``.
``joint_surface``
    On BivariateAggregate, primary. Needs the in-memory joint density. Options ``window`` (default 4: **draw** ``q(1e-4)`` to ``q(1 - 1e-4)`` of each marginal, measured on the fine lattice before the reduction). The window selects the block factor and says which part of the grid is the subject; it does not crop what is served, since a consumer forming a conditional off a cropped grid would normalize it by the visible mass, which is a different object whose mean moves whenever the window does. ``detail`` (default 128 cells **across the window**, a ceiling reached by a power-of-two block sum, mass preservingly) therefore bounds the drawn cells rather than the emitted axis, which runs the whole lattice at that step. ``encoding`` (default ``f32b64``; ``json`` for the plain arrays alone). The surface block carries both lattices as origin, step and count, the fine bucket size and block factor each was reduced from, the exact marginals, the fine-lattice means and the realized window as a sub-rectangle of the lattice. **No class method yet**: reach it through ``charts.build_chart_doc`` and :func:`~aggregate.plots.plot_chartdoc`.

Panel by panel
~~~~~~~~~~~~~~

.. Provenance of the table below. Transcribed from ``dev/summary-exhibits-and-charts.md`` section 2, and verified against the emitters at 1.0.0a248 by introspection, not from memory. The source of truth is one module per chart, ``src/aggregate/charts/_emit_aggregate.py``, ``_emit_portfolio.py``, ``_emit_pnl.py``, ``_emit_severity.py``, ``_emit_reins.py``, ``_emit_distortion.py``, ``_emit_bounds.py`` and ``_emit_bivariate.py``, each ending in its own ``register_chart`` call; the window, survival-depth and float-dust semantics the two-panel charts share live in ``charts/_two_panel.py``. Dump any row in one line with ``doc = charts.build_chart_doc(obj, name)`` then reading ``doc.panels``, ``doc.axes``, ``doc.series`` and ``doc.marks``. To update: edit the summary in ``dev/`` first, then re-transcribe here. A ``list-table`` again, for the same reason: a grid table cannot be edited without redrawing it.

.. list-table:: What each chart draws
   :header-rows: 1
   :widths: 12 22 20 20 14 22

   * - Chart / panel
     - x axis
     - y axis
     - Series
     - Marks
     - Readings offered
   * - ``agg`` / density
     - ``outcome``, Loss, currency, linear or log, window ``q(0.001) or 0 .. q(0.999)`` padded 2%, full is the whole grid
     - ``mass``, Probability mass, density, linear or log, ``(0, peak)``, no zoom-out
     - Aggregate and Severity, role ``density``, atomic
     - mean, full weight
     - log x, log y, full x
   * - ``agg`` / lee
     - ``p``, Non-exceeding probability, ``(0, 1)``, paired with ``return_period`` (log, ``1 .. 1e9``) and with ``survival``, Exceeding probability, linear or log, ``(0, 1)``
     - ``outcome``, shared with the density panel
     - Aggregate and Severity, role ``cdf``
     - none
     - log y, full y, reflect, return period, invert to "Distribution function"
   * - ``port`` / density
     - ``outcome``, Loss, currency, linear or log, window, full
     - ``mass``, linear or log, ``(0, top)``
     - one per unit (role ``unit``) then Total (role ``total``) last, so the book draws on top; each on its own native grid
     - mean
     - log x, log y, full x
   * - ``port`` / kappa
     - ``outcome``, shared
     - ``kappa``, ``E[Xi | X = x]``, currency, linear or log, window is the loss window, full to the last kept point
     - the same names again, role ``unit`` or ``total``, support continuous; the Total curve **is** the diagonal
     - none
     - log x, log y, full x, full y; ``aspect='equal'`` is semantic
   * - ``pnl`` / density
     - ``outcome``, P&L, currency, **linear only** (signed), window not anchored at 0, full
     - ``mass``, linear or log
     - one series, the result name
     - break even at 0, mean
     - log y, full x
   * - ``pnl`` / lee
     - ``p``, paired with ``return_period`` and ``survival``
     - ``outcome``, shared, linear only, full
     - the same series, role ``cdf``
     - break even (horizontal)
     - full y, reflect, return period, invert
   * - ``severity`` / density
     - ``loss``, currency, linear or log, ``0 .. isf(0.001)`` padded, full
     - ``pdf``, or Probability mass for a law with no density, linear or log, no window
     - one series; continuous unless the law is atomic, in which case the ordinate is read from the jumps of the cdf
     - none, deliberately
     - log x, log y, full x
   * - ``severity`` / lee
     - ``p``, paired with ``return_period`` and ``survival``
     - ``loss``, shared, linear or log, full
     - the same series, role ``cdf``
     - none
     - log y, full y, reflect, return period, invert
   * - ``reins`` / occurrence
     - ``claim``, Loss per claim, currency, **linear only**, window is the occurrence limit padded 2%, full is the grid
     - ``sev_density``, Occurrence density, **log only**, no window
     - Gross, Ceded, Net, roles ``gross``, ``ceded``, ``net``, net drawn last
     - none
     - full x only
   * - ``reins`` / aggregate
     - ``p``, paired with ``return_period`` and ``survival``
     - ``annual``, Aggregate loss, currency, linear or log, window from the gross curve, full
     - Gross, Ceded, Net again, as quantile curves
     - none
     - log y, full y, reflect, return period, invert
   * - ``distortion`` / square
     - ``s``, probability, linear, ``(0, 1)``, paired with ``s_complement``, ``1 - s``
     - ``g(s)``, probability, linear, ``(0, 1)``, paired with ``g_complement``, ``1 - g(s)``
     - the distortion, the dual (optional), then identity
     - none
     - reflect, which draws the dual; nothing else, since the unit square **is** the window and ``aspect='equal'`` is the whole point
   * - ``envelope`` / cloud
     - ``s``, ``(0, 1)``, paired with ``s_complement``
     - ``g(s)``, ``(0, 1)``, paired with ``g_complement``
     - Envelope (a ``y2`` band series, so the series is the region), ``n_resamples`` BiTVaR curves each carrying its weight, identity
     - none
     - reflect, which draws the envelope of the duals; equal aspect
   * - ``envelope`` / calibrated
     - ``s``, shared
     - ``g(s)``, shared
     - the band again, then CCoC, ``TVaR(p*)``, PH, Wang, Dual, then Avg extreme, then identity. **Panel omitted** where nothing is calibrated
     - none
     - reflect, which draws the envelope of the duals; equal aspect
   * - ``joint_surface`` / joint
     - ``x0``, resolved component label, currency
     - ``x1``, resolved component label, currency; z axis ``z``, density, linear or log
     - one ``SurfaceData``, role ``joint``, values are display-cell masses
     - none
     - log z. ``kinds`` declares only ``'surface'`` today, so the renderer's ``kind`` switch has nothing to choose between

Rendering
~~~~~~~~~

:func:`~aggregate.plots.plot_chartdoc` is the generic matplotlib renderer, and it draws any document the schema can express. Its six switches, ``log``, ``full_range``, ``reflect``, ``return_period``, ``invert`` and ``kind``, select among the readings a document declares, each acting on every axis or panel that declares the reading and on no other. The class ``plot`` methods listed above are thin: they build the document and render it, so ``a.plot(log=True)`` and ``plot_chartdoc(build_chart_doc(a, 'agg'), log=True)`` are the same call.

The renderer chooses stems, steps or a plain line from each series' declared ``support`` plus the room each atom gets, which is why the IR carries the support and not the drawing. Asked strictly for a panel kind it cannot realize, it raises :class:`~aggregate.charts.ir.ChartCapabilityError` rather than approximating silently. matplotlib and a 3-D surface is the live case: the honest non-strict answer is a labeled 2-D projection, and the title says so.

.. _exhibits charts no column:

More details
-------------

Two panels, one axis, except once.
    On ``agg``, ``pnl`` and ``severity`` the outcome axis is a single :class:`~aggregate.charts.ir.ChartAxis` referenced as the density panel's x and the Lee panel's y, so a window set on it moves both. On ``port`` both panels take it as x. ``reins`` is the deliberate exception: its two panels share nothing, because a per-claim loss and an annual aggregate are different quantities and one window across both would claim they were the same.
Three floors, all measured rather than chosen.
    ``LOG_FLOOR = 1e-15`` turns float dust into ``None`` gaps that a renderer must break the line at, never bridge. ``SURVIVAL_FLOOR = 1e-9`` is the deepest survival worth a panel, and is what puts the return-period axis top at 1e9. ``KAPPA_FLOOR = 1e-14`` on the portfolio kappa panel is a decade above the dust floor because kappa divides by ``p_total``, and the residual of the sum-to-diagonal identity is what measured the cliff.
Support is a fact about the law.
    ``support='atomic'`` says the points carry the whole distribution and there is nothing between them, which in this library is the normal case, because a discretized aggregate **is** the distribution rather than an approximation to some continuous ideal. ``'continuous'`` says the points sample a function that exists everywhere between them, which is a distortion, a kappa curve, and a severity that has a density.
Three panels became two, twice.
    The old aggregate compositor drew density, log density and Lee. The log density is not a third reading of the book, so it became a declared reading of the first, and the cdf panel came back as the Lee panel's declared inversion. The bounds compositor split five calibrated distortions across two panels by order of addition; all five now sit on one band.
What is missing.
    Severity publishes no exhibit. ``joint_surface`` has no class method. ``INSURED`` and ``REINSURER`` are vocabulary with no implementation. The joint panel declares one realization where the schema and the renderer both support a heatmap and surface toggle.

.. _exhibits charts notes:

Notes
------

The library owns meaning, the app owns arrangement.
    A caption, a row flag, a dropped raw moment and a declared reading are all knowledge an actuary would want in a notebook, so they live here. A color, a font and a hover template are not.
Capability is derived, not declared.
    ``available_exhibits`` and ``available_charts`` read the dispatch registries and the predicates. There is no second list to fall out of step, which is why open registration by app or user code is safe.
INSURER equals RAW unless an override is registered.
    That is what makes the last column of the exhibit catalogue the complete inventory of business translation, and what makes a new exhibit useful the moment its raw registration exists.
The document carries both readings of every number, and both forms of every string.
    A block ships formatted text beside raw values; a chart document ships plain text beside its typeset form. In each case the two serve different consumers, and deriving one from the other at the far end is guesswork.
A declared reading is a fact about the quantity.
    Declaring log on an axis asserts that a log reading of that quantity is honest. That is why one renderer switch can act on every chart at once without knowing which chart it holds.
Semantics only, with no escape hatch.
    There is no ``extra_mpl_kwargs`` and no renderer passthrough anywhere in the chart IR. A need the schema cannot express changes the schema visibly, or the chart stays bespoke. The rule is what keeps two independent renderers agreeing on what a picture means.
