
.. _num bucket selection:

Automatic Grid Selection (``bs``, ``log2``, ``x_min``)
======================================================

.. note::

   Draft for the technical guides (destined for
   ``docs/5_technical_guides/``). It documents the automatic grid sizer
   :meth:`aggregate.distributions.Aggregate._bs_window` as built at
   ``1.0.0a59``. The companion page :ref:`num how agg reps a dist` covers the
   *severity* discretization (how a single distribution is laid on a lattice);
   this page covers the *grid* the aggregate is computed on. This document is
   kept current by **``dev/plan-univariate-bucket.md``** (the "1A-bucket" plan),
   which owns the tail-intelligence and reporting work flagged under
   :ref:`bs open planned` below. The inspection frame is currently
   ``Aggregate._bs_window_df``; the plan adds a curated public
   ``Aggregate.bs_window_df`` (and ``Portfolio.bs_window_df``) and narrative
   ``bs_description`` / ``bs_explanation``.

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
    selection priority.

``bounded_small`` -- when every severity is bounded
    Uses the hard support bound ``[min(0, N_hi*s_min), N_hi*s_max]`` from a high
    frequency quantile ``N_hi`` (analytic frequency moments). Tight for a small
    claim count; for a large count the law of large numbers concentrates the
    aggregate well inside it, so it is selected **only when no wider than
    ``1.5x`` the moment window**.

``windowed`` -- the left-tail-aware placement for a *concentrated* book
    A concentrated aggregate -- ``agg_cv < 1/z`` with
    ``z = norm.isf(10**-window_nines)``, equivalently a lower window edge that
    clears ``0`` -- has its whole mass band sitting far above the origin. This
    method computes a *two-sided* window ``[w_lo, w_hi]``
    (:func:`estimate_agg_window`), so it **does** look at the left tail and lift
    ``x_min`` off the floor, reclaiming the empty ``[0, w_lo)`` region. The
    aggregate is computed far from ``0`` via a *benign* FFT wrap: the finished
    density is relabelled by a modular ``np.roll`` of ``round(x_min/bs)``
    buckets, which is exact for random frequency (it carries no ``N*s`` shift).
    Eligibility is deliberately narrow: auto origin only
    (``x_min`` not pinned), non-signed, non-affine, finite positive sd, and no
    *occurrence* reinsurance (whose severity rides the output grid). A
    **severity-fit guard** then marks the row applicable only when a single
    occurrence fits the windowed extent ``[0, N*bs]`` -- otherwise the benign
    wrap is invalid (the ``fixed``-1 / ``approximate`` trap, and the heavy
    *positive* "Regime B" book whose severity overflows a tight band).

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

1. **Pick the base window.** ``exact_discrete`` if present; else
   ``bounded_small`` when it is no wider than ``1.5x`` the moment window; else
   ``moment``.

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
   uselessness -- and a ``logger.info`` reports the reach and the ``log2`` that
   would capture it. The ``log2`` budget (explicit, hinted, or the default) is
   **never** silently grown, and a pinned ``bs`` is never coarsened.

4. **Balance the padding** (windowed selection, auto origin, non-affine). A
   windowed band is first placed band-bottom (all power-of-two slack above it);
   the slack is then redistributed a *fixed* fraction ``f = 0.5 -/+
   window_pad_skew`` below the band, so the band sits sensibly in the grid.
   *Planned* (see below): make the split **tail-aware** -- the slack should
   follow the tails, so a thick-right / thin-left book pushes it right (and the
   mirror for thin-right / thick-left), while a book with balanced tails shares
   it evenly -- rather than a single fixed convention skew.

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

* **padding skew** -- ``f = 0.5 - window_pad_skew`` for a loss (more room on the
  priced right), ``0.5 + window_pad_skew`` for a payoff (mirror).

So a loss and its mirror-image payoff place their bands as reflections of one
another. (Under the planned tail-aware padding, the convention skew becomes a
*tie-breaker* applied on top of the tail-driven split.)

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

Open / planned (owned by ``plan-univariate-bucket.md``)
-------------------------------------------------------

**A first-class tail report (freq / sev / agg, thick / thin per side).**
*(Structured frame landed in a60; the narrative extension and the wiring below
are still open.)* The sizer's decisions all turn on tail shape, but it currently
reads only the ``bounded`` bit of :mod:`aggregate.tail`. The report is a
structured, five-field summary -- ``min``, ``max``, ``bounded``, **left**
thick/thin, **right** thick/thin -- plus a *conservative* concentration flag
(``cv < CONCENTRATION_CV``, ``0.1``) for the **frequency**, each **severity**
mixture component (then blended into the combined severity), and the
**aggregate**. Frequencies are ``>= 0`` so their left tail is always thin; the
aggregate left tail is thin for a positive book and inherits the severity's left
jump for a signed one (a reflected ``ssev``/``dsev`` mirrors the combined
severity's left; an affine ``pnl`` mirrors the loss's *right* tail). The decision
basis is ``{left thick/thin, right thick/thin, concentrated}`` (the finer rung /
``alpha`` is kept underneath only for tail-quantile *magnitude*). It is exposed
now as the spec-only structured :attr:`Aggregate.tail_df` (a row per layer,
indexed by ``component``); the narrative ``tail_description`` (short) /
``tail_explanation`` (verbose) extension to the layered content, with the
ANSI-colour option that renders well in JupyterLab, follows in
``[tail-narrative]``.

**Wire the sizer to the tail report.** The ``sbj`` floor is the computational
twin of the single-big-jump principle the classifier already names. Gate it on a
SUBEXPONENTIAL-or-heavier severity (a no-op for lighter classes, but explicit);
for a POWER_LAW severity take the right-tail quantile from the exact tail law
``q_X(p) \propto (1-p)^{-1/\alpha}`` rather than the three-moment shifted fit --
both more accurate and the principled handling of the infinite-variance case
that today falls back to :meth:`recommend_bucket`. A power-law / infinite-variance
tail has no finite deep quantile to size to, so the plan does not chase one: it
sizes the reachable bulk, **accepts the truncation without normalising**, and
warns (exact below the truncation, deficit reported). The classifier is right-tail
only today; the plan adds a left-tail rung and classifies non-family severities
**structurally** (base family + limit / splice / attachment), with no numeric
density estimator -- a genuinely unknown *and* unlimited family is treated
conservatively as thick.

**Asymmetric window for concentrated heavy-positive books.** ``T5`` above is the
motivating case: its ideal grid is roughly ``[400k, 1.2M]`` -- the ``windowed``
left-tail lift (``x_min`` off the floor, **trustworthy precisely because** the
lognormal aggregate's left tail is thin: a large-deviation "conspiracy of many",
no subexponential reach on the left) combined with the ``sbj`` right reach and a
``bs`` coarse enough that a single severity fits. Realizing it means flooring the
``windowed`` upper edge by ``sbj_hi``, *gating the left lift on a thin-left
determination from the tail report*, and relaxing the selection gate so a
windowed grid wins when it captures a reach the 0-based pick clips (not only when
it is no coarser).

**Tail-aware padding.** Replace the fixed ``window_pad_skew`` slack split with
the tail-driven rule in step 4 (slack follows the tails; convention skew becomes
a tie-breaker).

**``bs_description`` / ``bs_explanation`` -- a narrative of the grid choice.**
Grid selection is the #1 numerical decision, it is subtle, and it is a frequent
source of user confusion, so in addition to the (private, complete)
``_bs_window_df`` and the (public, culled) ``bs_window_df`` the plan adds
read-only ``bs_description`` (the short summary) and ``bs_explanation`` (the
verbose prose) text properties -- the same narrative technique planned for the
forthcoming validation report. ``Portfolio`` picks up the same surfaces when the
windowed combine (1P, ``plan-bucket-window-2.md``) lands.
