
.. _num bucket selection:

Automatic Grid Selection (``bs``, ``log2``, ``x_min``)
======================================================

.. note::

   Draft for the technical guides (destined for
   ``docs/5_technical_guides/``). It documents the automatic grid sizer
   :meth:`aggregate.distributions.Aggregate._bs_window` as built at
   ``1.0.0a59``. The companion page :ref:`num how agg reps a dist` covers the
   *severity* discretization (how a single distribution is laid on a lattice);
   this page covers the *grid* the aggregate is computed on. The inspection
   frame is currently ``Aggregate._bs_window_df`` and is slated to become the
   public ``Aggregate.bs_window_df`` / ``Portfolio.bs_window_df``.

The problem
-----------

``aggregate`` computes an aggregate by FFT on a uniform lattice of
:math:`N = 2^{\mathrm{log2}}` buckets of width ``bs``, with the first bucket at
the origin ``x_min``. Every downstream quantity -- the density, quantiles,
TVaR, distortion prices, allocations -- is read off that lattice, so the grid
choice is the single most consequential numerical decision in the library.
Three numbers must be fixed before the FFT runs:

``bs``
    the bucket width (resolution). Too coarse and the bulk is under-resolved
    (the mean drifts); too fine and the grid cannot reach the tail.

    >>>> bs is constrained to be an exact binary fraction.

``log2``
    :math:`\log_2 N`, the bucket count (extent, given ``bs``). Memory and time
    grow like :math:`2^{\mathrm{log2}}`, so this is a hard budget.

``x_min``
    the origin. ``0`` for an ordinary non-negative aggregate; moved off
    ``0`` for a *concentrated* book whose mass sits far from ``0``; negative
    for a genuinely signed aggregate.

The tension is fundamental: at a fixed ``log2`` budget, the grid spans
``N * bs``, so **resolution and extent trade off directly**. The sizer's job is
to spend the budget where the mass and the priced tail actually live.

Two ways the FFT goes wrong
---------------------------

The grid is *periodic*: the FFT convolution is circular, and the severity is
laid on the same wrap-around buffer. Two distinct failures follow, and they
have **different urgencies**.

Aggregate aliasing (the right tail, a *refinement*)
    For a non-negative severity the discretized severity is bounded on
    ``[0, N*bs]``, but the *aggregate* -- a sum of many claims -- has support
    beyond ``N*bs``. The circular convolution folds that excess right-tail mass
    back to low buckets (or, with zero padding, clips it). The result is a
    sliver of misplaced mass (often :math:`10^{-7}` or less) and a small mean
    error; the bulk is intact. This is the ``ALIASING`` / deficit regime: a
    *quality* issue, fixable by more extent, never catastrophic.

Severity wrap (the left tail of a signed sev, a *correctness* problem)
    A **signed** severity (declared ``ssev``, or ``dsev`` with a negative atom,
    e.g. ``100 - lognorm 10 cv 2.5``) reaches below ``0``. It is discretized on
    the *same* periodic buffer, so if its negative reach exceeds the grid width
    the severity itself wraps -- negative mass lands at the *top* of the grid --
    and the whole convolution is built on a corrupted severity. This is not a
    tail-accuracy issue: the entire law is garbage (a measured ~47% mass loss
    on the example above). Covering a signed severity's reach is therefore
    **non-negotiable**: the sizer will coarsen ``bs`` to do it, because
    *coarse-but-correct beats fine-but-garbage*.

    >>>> I don't think this is correct. it is just discretized and off-grid
    probability is lost. sev does not wrap. The agg statement is correct. The
    difference is that with positive sev you don't compute x_max, it is implicit
    equal to N * bs (if you start at 0). Here we SET x_min.


This asymmetry -- positive heavy = refinement, signed = correctness -- drives
the single-big-jump floor below.

The candidate methods
---------------------

The sizer runs up to four sizing **methods** plus the single-big-jump
**floor**, records each as a row of ``_bs_window_df`` (inspectable, whether or
not it is chosen), then selects one. Each method proposes a window
``[x_lo, x_hi]`` and a grid ``(x_min, bs, log2)`` derived from it; the realized
power-of-two grid is reported in the ``used`` row.

>>>> layout that log2 is almost always given - or at least an upper bound.

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
   budget; otherwise ``bs`` is **coarsened** within the cap until it does. The
   severity can never wrap.

   *Positive (refinement).* The window is extended up to ``sbj_hi`` **only when
   it fits at the bulk ``bs`` within the requested ``log2``** (so a generous
   ``log2`` is captured fully and finely). If it does not fit, the moment
   window is kept -- clipping a tiny far tail beats coarsening the bulk to
   uselessness -- and a ``logger.info`` reports the reach and the ``log2`` that
   would capture it. The ``log2`` budget (explicit, hinted, or the default) is
   **never** silently grown, and a pinned ``bs`` is never coarsened.

4. **Balance the padding** (windowed selection, auto origin, non-affine). A
   windowed band is first placed band-bottom (all power-of-two slack above it);
   the slack is then redistributed, a fraction ``f = 0.5 -/+ window_pad_skew``
   below the band, so the band sits sensibly in the grid. The sign of the skew
   follows the convention (below).

5. **Reflect / shift for ``pnl``** (affine aggregates). The loss FFT runs on the
   non-negative grid sized above; the finished aggregate is reflected and/or
   shifted, and a tight two-sided P&L *display* window is computed from the
   affine moments. The ``used`` row reports that realized P&L grid.

The ``used`` row is the realized power-of-two grid:
``x_max = x_min + 2**log2 * bs`` (so a 701-point support padded to 1024 reads
``x_max`` at the grid top).

>>>> This step should be improved. The "extra" should be shared hi/low to
improve the look of the picture. If thick right/thin left it all goes right
and vice versa, but if left right tails are balanced the "extra" should be
shared too.

>>>> The decision journey should be readable from _bs_window_df. Add extra
rows if needed. (there may be that detailed private version and we add a
public property that culls it down a bit.) Or do you feel that is already
covered by "applies" and "used" cols?

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
another.

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


Open / planned
--------------

**Asymmetric window for concentrated heavy-positive books.** The ``T5`` example
above is the motivating case: its ideal grid combines the ``windowed`` left-tail
lift (``x_min`` off the floor -- justified because a positive severity's
aggregate *left* tail is thin, a large-deviation "conspiracy of many", well
estimated by the existing moment lower quantile, **no** subexponential reach on
the left) with the ``sbj`` right reach (the heavy *right* tail) and a ``bs``
coarse enough that a single severity fits. Realizing it means flooring the
``windowed`` upper edge by ``sbj_hi`` and relaxing the selection gate so a
windowed grid wins when it *captures a reach the 0-based pick clips* (not only
when it is no coarser). Pending implementation and a full byte-stability check.

>>>> yes, this is a really good example - but the answer should be 400-1.2M ish.
>>>> we need distinguish that windowed left is a good number cos lognormal
thin left tail?


**Wire the sizer to the tail classifier.** The module :mod:`aggregate.tail`
(:attr:`Aggregate.tail_class`) is the single source of truth for tail shape --
the ordered scale ``BOUNDED < SUPER_EXPONENTIAL < EXPONENTIAL < SUBEXPONENTIAL
< POWER_LAW``, a power-law index ``alpha``, and ``infinite_variance`` /
``infinite_mean`` flags -- and its ``tail_explanation`` already states the very
single-big-jump principle the ``sbj`` floor implements. Today the sizer reads
only ``bounded`` (for ``bounded_small``); the heavier signal is unused. The
``sbj`` floor should be *gated* on a SUBEXPONENTIAL-or-heavier severity (a no-op
for lighter classes, but explicit), and for a POWER_LAW severity the right-tail
quantile ``q_X(p**)`` should come from the exact tail law
``q_X(p) \propto (1-p)^{-1/\alpha}`` rather than the three-moment shifted fit --
which is both more accurate and the principled handling of the
infinite-variance case that currently falls back to :meth:`recommend_bucket`.
The classifier is right-tail only (Phase 1, family lookup); the *left*-tail
heaviness of a signed severity is not yet classified, so the signed ``sbj`` path
reads the exact ``sev.ppf`` directly. A future Phase-2 numeric estimator
(mean-excess / log-log-survival slope) and a left-tail rung would let the signed
path be classifier-informed too.


>>>> we need a good tail report: both as text and more usable internally in bs
selection. It must track

freq:
    min value (ususally but not always eg dfreq or zt poisson) 0
    max value or inf
    bounded: T/F (from max/min)
    left tail:  freq >=0 so must be thin left
    right tail: our usual classification

sev:
    same five -> these are per mix component and then blended at the Aggregate level

agg:
    derived five

Seems Frequency, Severity, and Aggregate all need the tail report.

Thus: internal representation and a text property narrative report. (with that nice
ansi?) formatting output option we have in format program for some color and emphasis
that comes out real nice in JLab.

THEN we need, in addition to _bs_window_df, another text property bs_story or similar
that is a narrative description of the bs choice. As you say in the intro this is the
#1 most important numerical decision. it is very complicated and subtle. it causes a
lot of user confusion. this will be a major selling point. we will use a similar technique
for the updated validation report (coming soon :-)). 
