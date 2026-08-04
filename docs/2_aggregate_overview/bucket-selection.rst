
.. _num bucket selection:

Automatic Grid Selection (``bs``, ``log2``, ``x_min``)
======================================================

.. note::

   This page documents the automatic grid sizer, ``Aggregate._bs_window``, and
   the after-the-fact probe,
   :meth:`~aggregate.distributions.Aggregate.sharpen`. Severity discretization
   (how a single distribution is laid on a lattice) is covered separately; this
   page covers the *grid* the aggregate is computed on. The sizer is as built at
   ``1.0.0a59`` with the tail-aware work of a64 and a65 on top, and ``sharpen``
   as built at a192 to a195. Both frames are available: the curated public
   ``bs_window_df`` (also on ``Portfolio``) and the complete private
   ``_bs_window_df``, with the narrative ``bs_description`` /
   ``bs_explanation``. The plan of record is
   **``dev/done/plan-univariate-bucket.md``** (the "1A-bucket" plan), whose
   tail-intelligence and reporting work is summarized under
   :ref:`bs open planned` below.

The problem
-----------

``aggregate`` computes an aggregate by FFT on a uniform lattice of
:math:`N = 2^{\mathrm{log2}}` buckets of width ``bs``, with the first bucket at
the origin ``x_min``. Every downstream quantity -- the density, quantiles,
TVaR, distortion prices, allocations -- is read off that lattice, so the grid
choice is the single most consequential numerical decision in the library.
Three numbers must be fixed before the FFT runs:

``bs``
    the bucket width (resolution), constrained to an exact **binary fraction**
    (a power of two, possibly negative -- ``..., 1/4, 1/2, 1, 2, 4, ...``) by
    ``round_bucket`` so grid arithmetic is exact. Too coarse and the bulk is
    under-resolved (the mean drifts); too fine and the grid cannot reach the
    tail.

``log2``
    :math:`\log_2 N`, the bucket count (extent, given ``bs``). Memory and time
    grow like :math:`2^{\mathrm{log2}}`, so this is a hard budget. In practice
    ``log2`` is almost always *supplied* -- by the caller, a hint, or the
    default (16) -- so the sizer's real freedom is ``bs`` and ``x_min`` *given*
    a ``log2`` budget (or an upper bound on it).

``x_min``
    the origin. ``0`` for an ordinary non-negative aggregate; moved off ``0``
    for a *concentrated* book whose mass sits far from ``0``; negative for a
    genuinely signed aggregate.

The tension is fundamental: at a fixed ``log2`` budget, the grid spans
``N * bs``, so **resolution and extent trade off directly**. The sizer's job is
to spend the budget where the mass and the priced tail actually live.

Two failure modes: aliasing and off-grid loss
----------------------------------------------

The lattice is finite and the FFT convolution is periodic. Two distinct
failures follow, and they have **different urgencies**.

Aggregate aliasing (the right tail, a *refinement*)
    For a non-negative severity the discretized severity is bounded on
    ``[0, N*bs]``, but the *aggregate* -- a sum of many claims -- has support
    beyond ``N*bs``. The circular convolution folds that excess right-tail mass
    back to low buckets (or, with zero padding, clips it). The result is a
    sliver of misplaced mass (often :math:`10^{-7}` or less) and a small mean
    error; the bulk is intact. This is the ``ALIASING`` / deficit regime: a
    *quality* issue, fixable by more extent, never catastrophic.

Off-grid mass loss from a mis-placed ``x_min`` (a signed book's *correctness* problem)
    Any severity (or aggregate) mass outside the grid ``[x_min, x_min + N*bs]``
    is simply **dropped** -- it is discretized and the off-grid probability is
    lost. The severity does **not** wrap. For a non-negative book this barely
    matters: ``x_min = 0`` is forced and the upper edge is *implicit* at
    ``N * bs``, so only a thin right-tail sliver can fall off (the aliasing case
    above) -- you never compute ``x_max`` at all. For a **signed** severity the
    sizer must *explicitly set* ``x_min`` (a negative origin), and the
    three-moment window can place it grossly wrong: a heavy-left severity like
    ``100 - lognorm 10 cv 2.5`` has *positive* aggregate skew (the ``+100^3``
    term dominates), so the moment window sits near ``[-749, 3557]`` while the
    true left reach is ~``-110000``. About half the mass falls below ``x_min``
    and is lost (a measured ~47%), and the law is garbage. Setting ``x_min`` to
    cover a signed severity's reach is therefore **non-negotiable**: the sizer
    will coarsen ``bs`` to do it, because *coarse-but-correct beats
    fine-but-garbage*.

This asymmetry -- positive heavy = refinement (a sliver off the implicit top),
signed = correctness (``x_min`` must be set to cover the reach) -- drives the
single-big-jump floor below.

The candidate methods
---------------------

The sizer runs up to four sizing **methods** plus the single-big-jump
**floor**, records each as a row of ``_bs_window_df`` (inspectable, whether or
not it is chosen), then selects one. Each method proposes a window
``[x_lo, x_hi]`` and a grid ``(x_min, bs, log2)`` derived from it; the realized
power-of-two grid is reported in the ``used`` row.

``moment`` -- always present
    The legacy three-moment method-of-moments window. For a **non-negative**
    aggregate it is one-sided: ``x_lo = 0`` and ``x_hi`` is the
    ``1 - 10**-window_nines`` quantile from a shifted-lognormal / shifted-gamma
    fit to the analytic ``(mean, cv, skew)`` (:func:`_estimate_agg_percentile`,
    the maximum over the two fits and a mean-plus-:math:`z\sigma` normal). It
    therefore **does not look at the left tail** -- the origin is pinned at
    ``0``. For a **signed** aggregate it is two-sided
    (:func:`estimate_agg_window`), placing both edges. If ``bs`` is pinned by
    the caller the window is just the realized extent ``[0, N*bs]``. Fallbacks:
    no finite variance (e.g. an unlimited Pareto) defers to
    :meth:`recommend_bucket`; a degenerate (zero-sd) book collapses to the
    mean.

``exact_discrete`` -- when the aggregate lives on an integer lattice
    A ``dfreq`` / ``fixed`` frequency crossed with ``dsev`` / ``dhistogram``
    severities on an integer lattice has *finite, exactly computable* support.
    The method returns ``[A_lo, A_hi]`` with
    ``A_hi = max(N_max*s_max, N_min*s_max)`` (and the mirror for ``A_lo``,
    including ``0`` when ``0`` is a count atom), ``bs = 1``, and the *minimal*
    ``log2`` that holds it (never more than the cap). It is exact, so it has top
    selection priority -- **but only when its support is reachable** (the guard
    below).

    *Reachability guard.* The combinatorial support is exact but can be a gross
    *overstatement*: for a large count the extreme corner ``A_hi = N*s_max`` needs
    *every* one of ``N`` claims to land on the single largest atom, an
    astronomically improbable event, so the mass sits nowhere near ``A_hi`` (a
    near-normal aggregate whose support is vast but whose spread the CLT
    concentrates to a few ``sd``). Sizing the grid to that fictional support then
    forces ``bs`` far past the lattice step to fit the cap, and the coarse ``bs``
    **aliases the severity** (the reported case: ``dfreq[1000000] dsev[-1000 20]``
    coarsened ``bs`` from ``1`` to ``20000`` and the estimated ``sd`` came out
    ``6.3x`` too big). So each corner carries a ``log10`` *attainment probability*

    .. math::

        \log_{10} P(\text{corner})
          = \log_{10} P(N = N_{ach}) + N_{ach}\,\log_{10} P(X = s_{ext}),

    where ``N_ach`` is the count that *realizes* that corner -- ``N_max`` for the
    outer extreme (``s_max > 0`` above, ``s_min < 0`` below), ``N_min`` for the
    inner extreme (a positive ``s_min``, a negative ``s_max``); the low bound of a
    non-negative book is the *fewest* claims of the smallest atom, far likelier
    than ``N_max`` of it. A corner pinned at ``0`` (a ``0`` severity atom, or the
    empty sum when ``0`` is a count atom) is always reachable. The method keeps
    its top priority only when **at least one corner is reachable**
    (``max(logp_lo, logp_hi) >= exact_discrete_reach_logp``, default ``-20``);
    when **both** corners fall below the floor the support is fictional and the
    row is marked inapplicable -- recorded in ``_bs_window_df`` for inspection
    (``coverage = 'support unreachable (rejected)'``, the two ``logp`` in
    ``note``) but not selected, so the sizer falls through to ``bounded_small`` /
    ``moment``. A genuine small-count discrete book has corners near
    ``10**-3``..``10**-6`` and is untouched (byte-stable); an asymmetric book with
    one reachable corner (e.g. a ``0`` lower bound) keeps its exact support.

``bounded_small`` -- when every severity is bounded
    Uses the hard support bound ``[min(0, N_hi*s_min), N_hi*s_max]`` from a high
    frequency quantile ``N_hi`` (analytic frequency moments). Tight for a small
    claim count; for a large count the law of large numbers concentrates the
    aggregate well inside it, so it is selected **only when no wider than
    ``1.5x`` the moment window**.

``windowed`` -- the left-tail-aware placement for a *concentrated* book
    A concentrated aggregate -- ``actual_cv < 1/z`` with
    ``z = norm.isf(10**-window_nines)``, equivalently a lower window edge that
    clears ``0`` -- has its whole mass band sitting far above the origin. This
    method computes a *two-sided* window ``[w_lo, w_hi]``
    (:func:`estimate_agg_window`), so it **does** look at the left tail and lift
    ``x_min`` off the floor, reclaiming the empty ``[0, w_lo)`` region. The
    aggregate is computed far from ``0`` via a *benign* FFT wrap: the finished
    density is relabelled by a modular ``np.roll`` of ``round(x_min/bs)``
    buckets, which is exact for random frequency (it carries no ``N*s`` shift).
    Eligibility is deliberately narrow: auto origin only
    (``x_min`` not pinned), non-signed, non-affine, finite positive sd, the
    conservative ``concentrated`` flag from the tail report (``actual_cv <
    CONCENTRATION_CV``, ~0.1; a64), and no *occurrence* reinsurance (whose
    severity rides the output grid). A **severity-fit guard** then marks the row
    applicable only when a single occurrence fits the windowed extent
    ``[0, N*bs]``. Since a64, a thick-right / thin-left concentrated band floors
    its upper edge by ``sbj_hi`` *before* the guard, which grows the extent
    enough that a single heavy occurrence fits -- so the former heavy-*positive*
    "Regime B" book (severity overflows a tight band) is reclaimed rather than
    falling back to the 0-based grid. The guard still rejects the ``fixed``-1 /
    ``approximate`` trap (a single thin severity already at the aggregate mean).

``sbj`` -- the single-big-jump extent floor
    Not a standalone sizing of the grid but a **floor on the selected window's
    extent**, recorded as its own row for inspection. For a subexponential
    severity the aggregate's far tail is one big claim on an otherwise typical
    bulk, ``P(S > x) ~ E[N] * P(X > x)``. To cover the aggregate to ``p*`` the
    *severity* is probed at the ``E[N]``-adjusted depth
    ``p** = 1 - (1 - p*)/E[N]`` (one claim reaches there; the other ``E[N]-1``
    are typical -- *not* ``N * q_X``, which assumes every claim is huge), and
    the extent floored at

    .. math::

        \mathrm{sbj\_hi} = E[S] - \mu_X + q_X(p^{**}),

    mirrored on the low side for a signed severity. ``p**`` deepens with
    ``E[N]`` and would underflow double precision for a large count, so
    ``1 - p**`` is floored at ``sbj_tail_floor`` (and ``q_X`` is capped by any
    finite policy limit).

Selection and combination
-------------------------

The methods combine in a fixed order.

1. **Pick the base window.** ``exact_discrete`` if present **and reachable**
   (at least one corner's ``log10`` attainment probability ``>=
   exact_discrete_reach_logp``); else ``bounded_small`` when it is no wider than
   ``1.5x`` the moment window; else ``moment``. A ``dfreq`` / ``fixed`` x ``dsev``
   book whose combinatorial support is *fictional* (both corners below the floor,
   e.g. a large fixed count) therefore skips its exact support and sizes on the
   moment window -- avoiding the ``bs`` coarsening that would alias the severity.

2. **Let ``windowed`` override.** If the windowed row *applies* (a single
   severity fits its extent) and is **no coarser** than the base pick
   (``bs_windowed <= bs_base``), it wins -- trading the empty space below the
   band for a centred placement. The ``<=`` (not ``<``) lets a band that clears
   ``0`` win on *placement* alone, even when ``bs`` is unchanged.

3. **Apply the single-big-jump floor** to the selected ``moment`` / ``windowed``
   window (only when ``bs`` is free -- a pinned ``bs`` is the caller's grid).
   The two regimes diverge here:

   *Signed (correctness).* The grid is widened to cover ``[sbj_lo, sbj_hi]``
   unconditionally. The bulk ``bs`` is kept if the reach fits the ``log2``
   budget; otherwise ``bs`` is **coarsened** within the cap until it does. No
   signed mass is left off-grid.

   *Positive (refinement).* The window is extended up to ``sbj_hi`` **only when
   it fits at the bulk ``bs`` within the requested ``log2``** (so a generous
   ``log2`` is captured fully and finely). If it does not fit, the moment
   window is kept -- clipping a tiny far tail beats coarsening the bulk to
   uselessness -- and (since a64) a visible ``DefectiveDistributionWarning``
   reports the reach, the estimated clipped mass, and the ``log2`` that would
   capture it (also stashed in ``Aggregate._bs_clip``). The ``log2`` budget
   (explicit, hinted, or the default) is **never** silently grown, and a pinned
   ``bs`` is never coarsened. (This positive floor is gated on a thick right tail
   from the tail report; for a thin tail the moment window already covers it.)

4. **Balance the padding** (windowed selection, auto origin, non-affine). A
   windowed band is first placed band-bottom (all power-of-two slack above it);
   the slack is then redistributed **tail-aware** (a64): an asymmetric band puts
   ~3/4 of the slack on the thick side (``WINDOW_SLACK_THICK`` -- so a thick-right
   / thin-left book pushes it right, and the mirror for thin-right / thick-left),
   while a symmetric band (both tails thick or both thin) centres with the
   loss/payoff convention skew (``window_pad_skew``) as a tie-breaker.

5. **Reflect / shift for ``pnl``** (affine aggregates). The loss FFT runs on the
   non-negative grid sized above; the finished aggregate is reflected and/or
   shifted, and a tight two-sided P&L *display* window is computed from the
   affine moments. The ``used`` row reports that realized P&L grid.

The ``used`` row is the realized power-of-two grid:
``x_max = x_min + 2**log2 * bs`` (so a 701-point support padded to 1024 reads
``x_max`` at the grid top).

Convention orientation (loss vs payoff)
---------------------------------------

A windowed placement is oriented by the aggregate's sign convention -- the
``value_type`` role (``loss`` = "more is worse", ``payoff`` = "more is
better"), read from the boolean ``_is_loss_value`` (never the label string),
overridable per call via ``update(window_convention=...)``. The convention sets

* **per-edge coverage** -- the *protected* tail (the upper edge for a loss, the
  lower for a payoff) is covered deep at ``window_nines`` nines to avoid
  clipping the priced tail; the *cheap* tail is trimmed shallow at
  ``window_nines_trim``;

* **padding skew** -- *only as a tie-breaker for a symmetric band* (a64):
  ``f = 0.5 - window_pad_skew`` for a loss (more room on the priced right),
  ``0.5 + window_pad_skew`` for a payoff (mirror). An asymmetric band ignores
  the convention and skews to the thick tail instead.

So a loss and its mirror-image payoff place their bands as reflections of one
another *when the tails are symmetric*; an asymmetric (thick-right / thin-left)
book skews to the thick side under both readings (the per-edge coverage still
follows the convention).

Controls
--------

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Setting (``[discretization]``)
     - Default
     - Role
   * - ``bucket_sizing_p``
     - ``0.99999``
     - percentile fed to :meth:`recommend_bucket` (the infinite-variance
       fallback). ``>1`` reads as nines.
   * - ``window_nines``
     - ``12``
     - coverage of the protected window edge (and the moment window).
   * - ``window_nines_trim``
     - ``6``
     - coverage of the cheap (trimmed) edge of a windowed book.
   * - ``window_pad_skew``
     - ``0.1``
     - fraction the padding is skewed toward the priced tail.
   * - ``sbj_tail_floor``
     - ``1e-14``
     - deepest severity probe depth ``1 - p**`` for the single-big-jump floor
       (guards ``q_X(p**) -> inf`` for a large ``E[N]``).
   * - ``exact_discrete_reach_logp``
     - ``-20``
     - ``log10`` attainment-probability floor for the ``exact_discrete`` support
       corners. When both corners fall below it the exact support is fictional
       (unreachable extremes) and the method is rejected in favour of the moment
       window. Well below floating-point noise, well above a genuine small-count
       discrete book.

Module constant ``WINDOW_LOG2_GROWTH = 4`` bounds how far a *windowed*
integer-lattice book may grow ``log2`` past the cap to preserve an exact ``bs``
(so a high-mean ``dsev`` keeps ``bs = 1`` rather than coarsening off its
lattice). It never applies to continuous severities.

Inspecting a choice
-------------------

After ``update`` (or ``build``), ``a._bs_window_df`` holds one row per method
plus ``sbj`` and ``used``. Columns: ``applies`` (eligible / severity fits),
``x_min`` / ``x_max`` (the method's window), ``W`` (width), ``bs``, ``log2``,
``coverage`` (the nines used), ``note`` (provenance, e.g. ``; sbj floor`` when
the floor bound), and ``selected``. Reading it answers, at a glance, *why* a
grid was chosen and what the alternatives were:

.. code-block:: text

    >>> a = build('agg T5 5000 claims sev lognorm 100 cv 2 poisson')
    >>> a._bs_window_df[['applies','x_min','x_max','bs','log2','selected']]
              applies     x_min        x_max    bs  log2  selected
    moment       True       0.0     633908.8  10.0    16      True
    windowed    False  433225.0     633908.8   5.0    16     False
    sbj          True       0.0    1234033.2  20.0    16     False
    used         True       0.0     655360.0  10.0    16     False

Here the heavy positive book selects ``moment`` and clips: the realized top
(``655360``) sits below the single-big-jump reach (``1234033``). The ``sbj``
row shows the grid that *would* capture the tail (``bs=20`` at ``log2=16``); the
``windowed`` row shows the left-tail lift (``x_min=433225``) it cannot yet use
because at its fine ``bs=5`` a single severity overflows the band.

The ``applies`` / ``selected`` / ``used`` columns capture the *outcome* but not
the full *decision journey* -- e.g. *why* ``windowed`` did not apply, by how
much the ``sbj`` floor bound, or how much mass the selected grid clips. The
1A-bucket plan extends the private frame with that journey (extra rows/columns
as needed) and adds the curated public ``bs_window_df`` and narrative
``bs_description`` / ``bs_explanation`` (below).

.. _bs open planned:

Tail-aware sizing and reporting (landed; ``dev/done/plan-univariate-bucket.md``)
--------------------------------------------------------------------------------

**A first-class tail report (support + per-side tail class).** *(Structured
frame landed in a60, schema revised in a61; the narrative extension and the
wiring below are still open.)* The sizer's decisions all turn on tail shape, but
it currently reads only the ``bounded`` bit of :mod:`aggregate.tail`. The report
gives, per layer -- the **frequency**, each **severity** mixture component (then
blended into the combined severity), and the **aggregate** -- the **structural
support** (``min`` / ``max``, the smallest / largest *attainable* value, ``+-inf``
at an unbounded end) and the **tail class on each side** (``left_tail`` /
``right_tail`` ``in {bounded, super-exponential, exponential, subexponential,
power-law}``): a finite support end is ``bounded`` (a hard boundary, no tail), an
infinite end carries the family decay rung. ``bounded`` is the derived
``min``-and-``max``-finite. The aggregate row adds a *conservative* concentration
flag (``cv < CONCENTRATION_CV``, ``0.1``) and ``concentration_p = Phi(mean/sd)``,
the probability the band clears 0. The aggregate's per-side classes follow from
the count and combined-severity support through the ``pnl`` affine: a positive
book is ``bounded`` on the left (hard floor at 0); a signed ``ssev``/``dsev``
mirrors the combined severity's left; an affine ``pnl`` mirrors the loss's
*right* tail onto its left and caps the right at the premium. The sizer's
thick/thin is the derived :func:`~aggregate.tail.is_thick` of the relevant side
(thick ⇔ subexponential-or-heavier). It is exposed as the spec-only
:attr:`Aggregate.tail_df` (a row per layer, indexed by ``component``), and as the
narrative ``tail_description`` (short, three aligned lines, also in ``info()``) /
``tail_explanation`` (verbose, the per-component story with the single-big-jump
mechanism and concentration), both built from the same row list so frame and
prose never drift, with an ANSI ``color=True`` option that emphasises thick tails
on a TTY. ``Severity`` and ``Frequency`` carry one-line ``tail_description``
too.

**Wire the sizer to the tail report (landed a64).** The ``sbj`` floor is the
computational twin of the single-big-jump principle the classifier names, so it
is gated on a thick (SUBEXPONENTIAL-or-heavier) tail via ``_loss_tail_classes``
(a no-op for lighter classes, but explicit). A POWER_LAW / infinite-variance tail
has **no finite deep quantile to size to**, so there is no basis to guess ``bs``:
when ``bs`` is not supplied, the sizer raises
:class:`~aggregate.constants.InfiniteVarianceError` (a ``ValueError``) rather than
inventing a grid -- the user must pass an explicit ``bs`` (a86; this replaced the
earlier reachable-bulk fallback, which sized the bulk from the severity's actual
quantile and warned). An explicit ``bs`` pins the grid and builds normally.
The classifier carries a left-tail rung and classifies non-family severities
**structurally** (base family + limit / splice / attachment), with no numeric
density estimator -- a genuinely unknown *and* unlimited family is treated
conservatively as thick.

**Asymmetric window for concentrated heavy-positive books (landed a64).** ``T5``
above is the motivating case: its ideal grid is roughly ``[400k, 1.2M]`` -- the
``windowed`` left-tail lift (``x_min`` off the floor, **trustworthy precisely
because** the lognormal aggregate's left tail is thin: a large-deviation
"conspiracy of many", no subexponential reach on the left) combined with the
``sbj`` right reach and a ``bs`` coarse enough that a single severity fits. This
is realized by flooring the ``windowed`` upper edge by ``sbj_hi`` (which grows
the severity-discretisation extent enough that a single heavy occurrence fits --
reclaiming the former "Regime B"), gating the left lift on a thin-left
determination from the tail report, and relaxing the selection gate so a windowed
grid wins when it captures a reach the 0-based pick clips (not only when it is no
coarser).

**Tail-aware padding (landed a64).** The fixed ``window_pad_skew`` slack split is
replaced by a tail-driven rule: an asymmetric band puts ~3/4 of the slack on the
thick side (``WINDOW_SLACK_THICK``); a symmetric band centres, with the
loss/payoff convention demoted to a tie-breaker.

**``bs_description`` / ``bs_explanation`` -- a narrative of the grid choice
(landed a65).** Grid selection is the #1 numerical decision, it is subtle, and it
is a frequent source of user confusion, so in addition to the (private, complete)
``_bs_window_df`` and the (public, culled) ``bs_window_df`` there are read-only
``bs_description`` (the short summary -- winning method, ``(bs, log2, x_min)``,
grid top, any clip) and ``bs_explanation`` (the verbose prose -- the aggregate
tail one-liner, which methods applied and why the winner won, how to widen a
clipped tail) text properties, with ANSI-coloured variants in the module
functions ``bs_describe`` / ``bs_explain`` -- the same narrative technique
destined for the validation report. ``Portfolio`` carries ``bs_window_df`` and
``bs_description`` already; its full windowed combine (1P,
``plan-bucket-window-2.md``) inherits the rest when it lands.

.. _bs sharpen:

Auditing the choice afterwards: ``sharpen``
-------------------------------------------

Everything above **chooses** a grid, and it does so from the analytic moments,
before any FFT has run. :meth:`~aggregate.distributions.Aggregate.sharpen`
**audits** that choice after the fact: it re-updates the object on neighbouring
``(bs, log2)`` cells, scores each against the analytic moments, and moves when
the win is worth having. It is on :class:`~aggregate.distributions.Aggregate`
and :class:`~aggregate.portfolio.Portfolio`, and landed over a192 to a195.

The two halves answer different questions. The sizer asks *"given only what I
can compute in closed form, where should the grid go?"*, with no realized
density to look at. ``sharpen`` asks *"now that it has run, was that right?"*,
with the estimated moments, the realized mass and every warning the update
raised in hand.

The score
~~~~~~~~~

``validation_score`` is six terms, severity and aggregate mean, CV and
skewness, each read from the canonical ``stats_df['error']`` and divided by
**its own** validation tolerance: ``eps`` for a mean, ``10 eps`` for a CV,
``100 eps`` for a skewness, exactly the multiples ``valid_aggregate`` applies.
They combine in a power mean, Euclidean (``power=2``) by default; ``power=1``
averages, ``power=inf`` reports the worst single term.

Dividing by the tolerance is what makes the number useful. **The units are
tolerance**, so ``score <= 1`` means the object passes validation and ``1`` sits
exactly on the pass boundary. Where ``valid`` says whether a line was crossed,
this says by how far, which is what makes two grids comparable. A consequence
worth knowing: the score does not move when ``validation_eps`` is changed.

A term is live only when its *theoretical* value is finite and above the noise
floor, the same test validation itself applies: a symmetric severity has no
skewness to validate against, a deterministic one no CV. The theoretical moments
do not depend on the grid, so the live set is stable as ``bs`` and ``log2`` vary,
which is precisely what makes scores comparable across a probe. Averaging rather
than summing then keeps objects with different numbers of live terms on one
scale.

The probe
~~~~~~~~~

One row per ``log2``, and within each row a **line search** out from the current
bucket: ``bs`` is doubled until the score stops improving, then halved likewise,
capped each way by ``bs_limit``. The rows come out ragged, which is why
``sharpen_df`` is a tidy frame rather than a matrix; cells never visited read
``NaN``.

The line search is well posed because the score is single-troughed in ``bs`` at
fixed ``log2``. With extent ``W = bs * 2**log2``, a coarser bucket buys extent
and loses resolution, so the two error families trade off and there is one
turning point. Stopping at the first cell that fails to improve assumes exactly
that. A walk that instead runs out of ``bs_limit`` while still improving records
so in its ``note``, a different fact from turning: a re-run will keep going.

``log2`` and ``log2 - 1`` are searched up front. ``log2 + 1`` is searched **only
if neither reaches the target**, since it costs twice as much per cell and under
the selection rule below it is usually not consulted.

The geometry is the point. A row is constant *resolution*; an anti-diagonal is
constant *extent*. Cells sharing an extent differ only in resolution, so reading
the two directions together says whether the grid is **extent-limited**, widen
it, or **resolution-limited**, refine it. That is also why the bucket steps are a
strict factor of two rather than rungs of the ``round_bucket`` ladder, whose
1.25 and 1.6 factors are too fine to move the error and out of step with ``log2``
plus or minus one.

Every cell runs inside its own guard: one that raises is recorded as ``NaN`` with
the exception text in its ``note``, and the sweep completes.

Selection: thrift, gated on soundness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rule is *never make the caller pay more than they already are*: take the best
score among the cells that **do not grow** ``log2``. Reducing ``log2`` is a bonus
rather than a goal, so it is picked up by the tie rule rather than chased. Cells
within ``SHARPEN_FALLBACK_SLACK`` (``1.25``) of the best count as tied; ties
break on ``log2`` first, so a free memory saving is taken when the score is
genuinely a wash, then on ``|d_bs|``, so a competitive centre wins over an
equally good move and the grid is not churned for nothing. ``log2`` grows by one
**only** when nothing at the current size or smaller reaches ``good_enough`` and
something at ``log2 + 1`` does. If nothing anywhere reaches the target, the best
cell overall is taken, still thrift-ordered, so the descent starts and a re-run
continues it.

``good_enough`` (default ``0.5``) is the only judgment knob, and it does two
things. As a **probe gate**: a grid already at or under it, and sound, is left
alone and nothing is run at all. As a **growth trigger**: it is the target that
``log2 + 1`` has to reach. A score is never negative, so ``good_enough=0`` reads
as "probe everything, never grow", which is why there is no separate ``force``
argument.

**A grid that loses mass off its top end is disqualified, however well it
scores** (a195). This is a *gate*, not a term in the score, for two reasons. A
moment score cannot see lost mass and never will: far-tail mass is negligible for
the first three moments and decisive for tail pricing, so the two measures are
orthogonal and the gate is the only thing that catches it. And it could not be a
penalty even if you wanted one, because every term in the score divides by its
own tolerance and a deficit has no tolerance to divide by; its cost is
qualitative rather than a matter of degree. Mass that runs off the top is
**dropped, not wrapped**, so the realized law sums to less than one and forwards
``S = 1 - cumsum`` differs from backwards ``S`` by exactly the missing amount:
two correct-looking pricing routes disagree in the tail. The threshold is the
library's own ``VALIDATION_NOISE``, the level at which
:class:`~aggregate.constants.DefectiveDistributionWarning` already fires, so a
cell rejected here is exactly one that would warn when you used it. Because
soundness sits outside the thrift ordering, it composes: a deficit becomes a
reason to **grow** ``log2``, which is its cure. Three columns carry it,
``deficit``, ``defective`` and ``warns``, and the narrative reports how many
better-scoring cells the gate threw out.

Discrete severity: the bucket is pinned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When every severity atom is a whole number of buckets the discretization is
already exact and no bucket change can help: a coarser bucket scatters the atoms
off their own values, a finer one only wastes grid. The bucket is therefore
**pinned** and only ``log2`` is probed, which is what a failing discrete object
actually needs, its problem being extent. The test is the gcd of the integer
severity atoms (``Aggregate._severity_lattice``, taken across units for a
``Portfolio``); it declines as soon as any component is continuous or off the
lattice.

Probe controls
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Argument
     - Default
     - Role
   * - ``bs``, ``log2``
     - current grid
     - centre of the probe. An explicit centre that is not where the object sits
       moves it there first, so the centre row describes a real state.
   * - ``good_enough``
     - ``0.5``
     - target score, in tolerance units. Probe gate and growth trigger, as above.
       ``0`` means probe everything and never grow.
   * - ``log2_cap``
     - ``20``
     - ``log2`` is never grown past this.
   * - ``bs_limit``
     - ``16``
     - largest factor by which the line search may multiply or divide ``bs``.
       A power of two, so ``16`` permits four doublings each way and a grid wrong
       by two orders of magnitude is reachable in one call.
   * - ``power``
     - ``2``
     - norm exponent used to combine the six terms.
   * - ``execute``
     - ``True``
     - ``True`` leaves the object on the chosen cell; ``False`` restores the
       original grid, so the probe is pure diagnosis.

Module constants: ``SHARPEN_LOG2_FLOOR = 8`` is the lowest ``log2`` the probe
will drop to (below it the comparison stops meaning anything),
``SHARPEN_FALLBACK_SLACK = 1.25`` is the tie band, and ``SHARPEN_BS_LIMIT = 16``
is the default ``bs_limit``. The tie band is deliberately tight: a genuine
improvement here is orders of magnitude, not percent.

Comparability is enforced by harvesting the settings that must not drift.
``update`` stamps ``sev_calc`` / ``discretization_calc`` / ``normalize`` /
``padding`` from *its arguments*, so a bare ``update(log2=..., bs=...)`` would
silently reset all four. They are read off the object once, up front, and passed
to every cell: the object comes back as it was, and the cells differ in ``bs``
and ``log2`` and in nothing else.

Inspecting a probe
~~~~~~~~~~~~~~~~~~

``sharpen_df`` is one row per cell, indexed by the offsets ``(d_bs, d_log2)``, so
the picture is one unstack away. Take a book auto-sized to ``bs = 2``,
``log2 = 16``, force it onto a grid far too small to hold it, and probe:

.. code-block:: text

    >>> a = build('agg Sharp 100 claims sev lognorm 100 cv 2 poisson')
    >>> a.validation_score
    0.3911267630053992
    >>> a.update(log2=14, bs=1/8)          # far too small
    >>> a.validation_score
    3493.9786647146843
    >>> a.sharpen()
    >>> a.sharpen_df.score.unstack('d_log2').round(3)
    d_log2        -1         0         1
    d_bs
    -1      4064.189  3816.641  3493.969
     0      3816.654  3493.979  3097.981
     1      3493.998  3098.005   907.703
     2      3098.052   907.704    28.736
     3       907.705    28.736     5.486
     4        28.734     5.486     1.606

Down a column is constant grid size, so it is the resolution question; along an
anti-diagonal is constant extent. Here the improvement runs up the ``d_bs`` axis
and does not turn, so the book is resolution-starved in the wrong direction: the
bucket was far too fine for the extent it had to cover. The winner is the
top-right corner, and it sits on the ``bs_limit`` still improving, which
``sharpen_description`` says out loud:

.. code-block:: text

    >>> print(a.sharpen_description)
    Sharpen: 18 cells in 0.35s, best score 1.61 vs 3.49e+03 at the centre, target 0.5. The winner sits on the bs_limit and was still improving. Moved: bs 1/8 to 2, log2 14 to 15.

Re-running re-centres the probe on the new grid and continues from there. The
soundness gate is visible in the same frame, alongside the score:

.. code-block:: text

    >>> a.sharpen_df[['bs', 'log2', 'score', 'deficit', 'defective', 'selected']].tail(4)
                  bs  log2      score       deficit  defective  selected
    d_bs d_log2
    3     1      1.0    15   5.486182  4.077595e-05       True     False
    4    -1      2.0    13  28.733585  1.101324e-02       True     False
          0      2.0    14   5.485980  4.078025e-05       True     False
          1      2.0    15   1.605666  5.448738e-07       True      True

Full column list: ``bs``, ``log2``, ``extent``, ``x_min``, ``score``, the six
per-term ``u_sev_mean`` .. ``u_agg_skew``, ``aliasing`` (the agg-mean over
sev-mean error ratio, recorded alongside the score but deliberately not part of
it, being the specific signature of ``bs`` too small), ``deficit``,
``defective``, ``validation`` (the one-line verdict), ``warnings`` / ``warns``,
``seconds``, ``selected`` and ``note``. ``sharpen_description`` and
``sharpen_explanation`` are the short and long narrative forms, in the same style
as ``bs_description`` / ``bs_explanation`` above.

The discrete case shows the pinned bucket. ``good_enough=0`` forces a probe on an
object that is already exact:

.. code-block:: text

    >>> d = build('agg D6 dfreq [3] dsev [1:6]')
    >>> d.sharpen(good_enough=0)
    >>> print(d.sharpen_description)
    Sharpen: 2 cells in 0.02s (discrete: bucket pinned, grid size only), best score 1.92e-12 vs 1.92e-12 at the centre, target 0. Kept bs 1, log2 5.

Reach, and what it does not cover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``update(..., sharpen=True)`` opts a single update into auto-sharpening. It is
**off by default and not turned on by** ``build``, because a probe costs eight or
more extra updates.

There is no ``BivariateAggregate.sharpen``: the probe is quadratic in the joint
grid. There is deliberately no ``PnL.sharpen`` either, since it would leave a
built ledger sitting on a stale grid; a P&L sharpens through its engine instead.

A :class:`~aggregate.portfolio.Portfolio` scores the **total only**, so a well
resolved total can still hide a poorly resolved unit. ``sharpen_explanation``
says so rather than leaving it to be discovered.
