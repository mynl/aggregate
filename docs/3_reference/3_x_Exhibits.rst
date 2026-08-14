Exhibits (provisional)
======================

.. warning::

   :mod:`aggregate.exhibits` is **provisional** in the sense of :pep:`411`: it is **not part of the 1.0 API contract**, and its API may change in a minor release with no deprecation period. Exhibit names, block structure, captions, row flags and the :class:`~aggregate.exhibits.Perspective` vocabulary may all move. It is public, and deliberately so, because feedback is how it graduates to stable. See :doc:`3_x_API_Stability`.

The library owns meaning, the app owns arrangement. An **exhibit** is a short list of presentation-ready tables derived from the first-class frames (``summary_df``, ``tail_df``, ``economic_df``, and the rest), carrying the business knowledge that would otherwise leak into every client: captions, row emphasis, raw moment drops, relabeling. The placement test: if deleting the web app would destroy knowledge an actuary would want in a notebook, that knowledge belongs here.

Reach it by submodule import, following the :class:`~aggregate.tweedie.Tweedie` and :class:`~aggregate.pentagon.Pentagon` precedent::

    from aggregate import build, exhibits
    a = build('agg Book 100 claims sev lognorm 100 cv 2 poisson')
    exhibits.available_exhibits(a)
    ex = exhibits.summary(a, 'insurer')

Perspectives
------------

A :class:`~aggregate.exhibits.Perspective` names who is looking at the numbers. ``RAW`` serves the underlying frame with no business translation: no row emphasis, no dropped rows, no rearrangement, though it does carry a caption saying what the frame is and the column formats for units the frame cannot carry itself. ``INSURER`` is where the business reading lives, saying what the frame *means* over ``RAW``'s what it *is*.

The default rule is that **INSURER equals RAW unless an override is registered** for the (exhibit, type) pair, which keeps the generic path total: a new exhibit is useful the moment its raw registration exists.

``INSURED`` and ``REINSURER`` are declared in the enum with no registrations, so that adding them later does not churn it. Declared is not promised: the reinsurer semantics review may yet rename or respell a member.

Two stages
----------

:func:`~aggregate.exhibits.exhibit_frames` is pure pandas and useful on its own to a caller who wants the numbers rather than a table. :func:`~aggregate.exhibits.build_exhibit` is the conversion to greater_tables IR. Only the second imports greater_tables, at the point of use, so ``import aggregate`` and ``import aggregate.exhibits`` both stay free of it.

Registration is open. App or user code may register a new type with ``summary.register(MyType)``, supplying a frames builder ``f(obj) -> [(block_name, DataFrame, spec_kwargs), ...]``.

Format sheets
-------------

How a served column reads is declared in two YAML **format sheets** shipped with the package, not in the code that builds the block. ``formats-raw.yaml`` gives the default reading of every named column, keyed by column label; ``formats-insurer.yaml`` is an overlay holding only the entries where the business reading differs, so absent means same. A sheet's ``styles`` section names a reading once (``money``, ``ratio``, ``probability``, ``residual``) and every column that wears it points at the name; four style names are greater_tables' own semantic column tags (``ratio``, ``year``, ``date``, ``raw``), and a column pointing at one of those is stamped with the tag as well as the format.

Because the sheet is keyed by label rather than by frame, it is also a registry of the column vocabulary: an entry asserts that a word means one thing across the package. Where one label really does mean two things, an ``exhibits`` section scopes an entry to a single exhibit, which is what keeps ``P`` reading as premium on the pricing tables and as the probability ladder on ``tail``.

**Overriding follows the ``.agg`` rule.** The loader reads the shipped sheet, then the same file name in ``~/.aggregate``, then the working directory, nearest winning and merging per key, so a one-line local sheet changes one reading and inherits the rest::

    # ./formats-raw.yaml
    styles:
      money: 'si'          # every amount in SI notation, this session

Precedence, low to high: greater_tables' dtype and tag inference, the shipped raw sheet, the insurer overlay under that perspective, the user and working-directory sheets, then an explicit ``formatters`` entry in a frames builder, which always wins. Values may be format sugar, an int, a mapping of ``FormatSpec`` fields, or a style name, and never a callable: a callable never reaches the IR, so a sheet can only say things a client can re-render.

.. currentmodule:: aggregate.exhibits

.. autosummary::

   Perspective
   Exhibit
   available_exhibits
   exhibit_frames
   build_exhibit
   register_simple_exhibit
   summary
   tail
   stats
   validation
   reins
   economic
   economic_ratios
   economic_waterfall
   dependency
   EXHIBITS
   CAPITAL_ANCHOR_PERIODS
   RAW_MOMENT_MEASURES
   format_sheet
   FormatSheet
   reload_format_sheets
   sheet_paths

.. automodule:: aggregate.exhibits

The machinery
-------------

.. automodule:: aggregate.exhibits._core

.. automodule:: aggregate.exhibits._formats
