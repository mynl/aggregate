.. _num bucket selection:

Automatic Grid Selection (``bs``, ``log2``, ``x_min``)
======================================================

This section describes how ``aggregate`` picks the lattice an aggregate is computed on. It covers the two ways a grid can go wrong, the tail report the sizer reads, the candidate sizing methods and how one is selected, the settings that control them, and the narrative and frames that explain a choice after the fact. It closes with :meth:`~aggregate.distributions.Aggregate.sharpen`, which audits the choice once the FFT has run. Severity discretization, how a single distribution is laid on a lattice, is covered in :ref:`agg pipeline grid`; this section is about the grid the aggregate is computed on.

The problem
-----------

``aggregate`` computes an aggregate by FFT on a uniform lattice of :math:`N = 2^{\mathrm{log2}}` buckets of width ``bs``, with the first bucket at the origin ``x_min``. Every downstream quantity, the density, quantiles, TVaR, distortion prices, allocations, is read off that lattice, so the grid choice is the single most consequential numerical decision in the library. Three numbers must be fixed before the FFT runs:

``bs``
    the bucket width (resolution), constrained by ``round_bucket`` to an exact **binary fraction**, a power of two, possibly negative: ``..., 1/4, 1/2, 1, 2, 4, ...``. Grid arithmetic is then exact. Too coarse and the bulk is under-resolved, so the mean drifts; too fine and the grid cannot reach the tail.
``log2``
    :math:`\log_2 N`, the bucket count, which fixes the extent given ``bs``. Memory and time grow like :math:`2^{\mathrm{log2}}`, so this is a hard budget. In practice ``log2`` is almost always supplied, by the caller, a hint, or the default of 16, so the sizer's real freedom is ``bs`` and ``x_min`` given a ``log2`` budget, or given an upper bound on it.
``x_min``
    the origin. It is ``0`` for an ordinary non-negative aggregate, moved off ``0`` for a concentrated book whose mass sits far from ``0``, and negative for a genuinely signed aggregate.

The tension is fundamental: at a fixed ``log2`` budget the grid spans ``N * bs``, so resolution and extent trade off directly. The sizer's job is to spend the budget where the mass and the priced tail actually live.

Three failure modes: aliasing, off-grid loss, and a grid that cannot work
-------------------------------------------------------------------------

The lattice is finite and the FFT convolution is periodic. Two failures follow from that, and they have different urgencies. A third is different in kind: it is a property of the severity, and no choice of grid within the budget fixes it.

Aggregate aliasing, a refinement problem in the right tail
    For a non-negative severity the discretized severity is bounded on ``[0, N*bs]``, but the aggregate, a sum of many claims, has support beyond ``N*bs``. The circular convolution folds that excess right-tail mass back to low buckets, or, with zero padding, clips it. The two outcomes are worth separating, because the library now reports them separately. Wrap **conserves** mass and relocates it, so the realized law still sums to 1 while the mean moves: that is ``ALIASING``, and since ``padding=1`` runs the FFT on a doubled grid and discards the top half, real wrap needs the aggregate to reach past **twice** the grid, which makes it rare by construction. Clipping **drops** mass, so the law sums to less than 1: that is the pmf deficit and ``DEFECTIVE``. Either way the result is usually a sliver, often :math:`10^{-7}` or less, and a small mean error, with the bulk intact. It is a quality issue, fixable by more extent, and never catastrophic.

Off-grid mass loss from a mis-placed ``x_min``, a correctness problem for a signed book
    Any severity or aggregate mass outside the grid ``[x_min, x_min + N*bs]`` is simply dropped: it is discretized and the off-grid probability is lost. The severity does not wrap. For a non-negative book this barely matters, because ``x_min = 0`` is forced and the upper edge is implicit at ``N * bs``, so only a thin right-tail sliver can fall off, which is the aliasing case above; you never compute ``x_max`` at all. For a signed severity the sizer must explicitly set ``x_min``, a negative origin, and the three-moment window can place it grossly wrong. A heavy-left severity like ``100 - lognorm 10 cv 2.5`` has positive aggregate skewness, because the :math:`+100^3` term dominates, so the moment window sits near :math:`[-749, 3557]` while the true left reach is about :math:`-110{,}000`. About half the mass falls below ``x_min`` and is lost, a measured 47%, and the law is garbage. Setting ``x_min`` to cover a signed severity's reach is therefore non-negotiable: the sizer will coarsen ``bs`` to do it, because coarse-but-correct beats fine-but-garbage.

A severity no grid in the budget can resolve, a modeling problem
    A thick enough severity furnishes its mean so far above its median that the resolution its body needs and the reach its tail needs cannot both fit in :math:`2^{\mathrm{log2}}` buckets, at any ``bs``. Refining ``bs`` shortens the reach and coarsening it loses the body, so there is nothing to trade. This is not a numerical accident and not something the sizer can fix; it is a statement about the model, and the fix is an occurrence limit. It is measured and reported as described in :ref:`bs feasibility`.

That asymmetry between the first two drives the single-big-jump floor described in :ref:`bs methods`. Positive and heavy is a refinement question, a sliver off the implicit top; signed is a correctness question, where ``x_min`` must be set to cover the reach.

The tail report
---------------

Every decision the sizer makes turns on tail shape, which it reads from :mod:`aggregate.tail` rather than guessing. The report gives, per layer, meaning the frequency, each severity mixture component (then blended into the combined severity), and the aggregate:

* the **structural support**, ``min`` and ``max``, the smallest and largest attainable value, with :math:`\pm\infty` at an unbounded end;
* the **tail class on each side**, ``left_tail`` and ``right_tail``, drawn from the ladder ``bounded < super-exponential < exponential < subexponential < power-law``. A finite support end is ``bounded``, a hard boundary with no tail; an infinite end carries the family's decay rung. The derived ``bounded`` flag is ``min`` and ``max`` both finite.

The aggregate row adds a conservative concentration flag, ``cv < CONCENTRATION_CV`` (0.1), and ``concentration_p``, the probability the band clears 0. The aggregate's per-side classes follow from the count and the combined-severity support through the ``pnl`` affine: a positive book is ``bounded`` on the left, a hard floor at 0; a signed ``ssev`` or ``dsev`` mirrors the combined severity's left; and an affine ``pnl`` mirrors the loss's right tail onto its left and caps the right at the premium. The sizer's thick-against-thin test is the derived :func:`~aggregate.tail.is_thick` of the relevant side, where thick means subexponential or heavier.

Classification is structural, from the base family plus any limit, splice or attachment, with no numeric density estimator. A genuinely unknown and unlimited family is treated conservatively as thick. The report is exposed as the spec-only :attr:`Aggregate.tail_df`, one row per layer indexed by component, and as the narrative ``tail_description`` (short, three aligned lines, also in ``info()``) and ``tail_explanation`` (verbose, the per-component story with the single-big-jump mechanism and concentration). Both are built from the same row list, so frame and prose never drift, and an ANSI ``color=True`` option emphasizes thick tails on a TTY. :class:`Severity` and :class:`Frequency` carry a one-line ``tail_description`` too.

.. _bs methods:

The candidate methods
---------------------

The sizer runs up to four sizing methods plus the single-big-jump floor, records each as a row of ``_bs_window_df``, inspectable whether or not it is chosen, then selects one. Each method proposes a window ``[x_lo, x_hi]`` and a grid ``(x_min, bs, log2)`` derived from it; the realized power-of-two grid is reported in the ``used`` row.

``moment``, always present
    The legacy three-moment method-of-moments window. For a non-negative aggregate it is one-sided: ``x_lo = 0``, and ``x_hi`` is the :math:`1 - 10^{-\mathrm{window\_nines}}` quantile from a shifted-lognormal or shifted-gamma fit to the analytic ``(mean, cv, skew)`` (:func:`_estimate_agg_percentile`, the maximum over the two fits and a mean plus :math:`z\sigma` normal). It therefore does not look at the left tail; the origin is pinned at ``0``. For a signed aggregate it is two-sided (:func:`estimate_agg_window`), placing both edges. If ``bs`` is pinned by the caller the window is just the realized extent ``[0, N*bs]``. A degenerate book with zero standard deviation collapses to the mean.

    A power-law or infinite-variance tail has no finite deep quantile to size to, so there is no basis on which to guess ``bs``. When ``bs`` is not supplied the sizer raises :class:`~aggregate.constants.InfiniteVarianceError`, a ``ValueError``, rather than inventing a grid: the user must pass an explicit ``bs``. An explicit ``bs`` pins the grid and the object builds normally.

``exact_discrete``, when the aggregate lives on an integer lattice
    A ``dfreq`` or ``fixed`` frequency crossed with ``dsev`` or ``dhistogram`` severities on an integer lattice has finite, exactly computable support. The method returns ``[A_lo, A_hi]`` with ``A_hi = max(N_max*s_max, N_min*s_max)``, and the mirror for ``A_lo``, including ``0`` when ``0`` is a count atom, together with ``bs = 1`` and the minimal ``log2`` that holds it, never more than the cap. It is exact, so it has top selection priority, but only when its support is reachable.

    The reachability guard exists because the combinatorial support, though exact, can be a gross overstatement. For a large count the extreme corner ``A_hi = N*s_max`` needs every one of ``N`` claims to land on the single largest atom, an astronomically improbable event, so the mass sits nowhere near ``A_hi``: a near-normal aggregate whose support is vast but whose spread the central limit theorem concentrates to a few standard deviations. Sizing the grid to that fictional support then forces ``bs`` far past the lattice step to fit the cap, and the coarse ``bs`` aliases the severity. In the reported case ``dfreq[1000000] dsev[-1000 20]`` coarsened ``bs`` from 1 to 20,000 and the estimated standard deviation came out 6.3 times too big.

    Each corner therefore carries a :math:`\log_{10}` attainment probability

    .. math::

        \log_{10} P(\text{corner})
          = \log_{10} P(N = N_{ach}) + N_{ach}\,\log_{10} P(X = s_{ext}),

    where :math:`N_{ach}` is the count that realizes that corner: :math:`N_{max}` for the outer extreme (``s_max > 0`` above, ``s_min < 0`` below) and :math:`N_{min}` for the inner extreme (a positive ``s_min``, a negative ``s_max``). The low bound of a non-negative book is the fewest claims of the smallest atom, far likelier than :math:`N_{max}` of it. A corner pinned at ``0``, from a ``0`` severity atom or the empty sum when ``0`` is a count atom, is always reachable. The method keeps its top priority only when at least one corner is reachable, that is ``max(logp_lo, logp_hi) >= exact_discrete_reach_logp``, by default ``-20``. When both corners fall below the floor the support is fictional and the row is marked inapplicable, recorded in ``_bs_window_df`` for inspection with ``coverage = 'support unreachable (rejected)'`` and the two ``logp`` values in ``note``, but not selected, so the sizer falls through to ``bounded_small`` or ``moment``. A genuine small-count discrete book has corners near :math:`10^{-3}` to :math:`10^{-6}` and is untouched, byte for byte; an asymmetric book with one reachable corner, such as a ``0`` lower bound, keeps its exact support.

``bounded_small``, when every severity is bounded
    Uses the hard support bound ``[min(0, N_hi*s_min), N_hi*s_max]`` from a high frequency quantile ``N_hi``, taken from the analytic frequency moments. It is tight for a small claim count; for a large count the law of large numbers concentrates the aggregate well inside it, so it is selected only when it is no wider than 1.5 times the moment window.

``windowed``, the left-tail-aware placement for a concentrated book
    A concentrated aggregate, meaning ``actual_cv < 1/z`` with ``z = norm.isf(10**-window_nines)``, equivalently one whose lower window edge clears ``0``, has its whole mass band sitting far above the origin. This method computes a two-sided window ``[w_lo, w_hi]`` (:func:`estimate_agg_window`), so it does look at the left tail and lift ``x_min`` off the floor, reclaiming the empty ``[0, w_lo)`` region. The aggregate is computed far from ``0`` through a benign FFT wrap: the finished density is relabelled by a modular ``np.roll`` of ``round(x_min/bs)`` buckets, which is exact for random frequency because it carries no ``N*s`` shift.

    Eligibility is deliberately narrow: automatic origin only, so ``x_min`` is not pinned; non-signed; non-affine; finite positive standard deviation; the conservative ``concentrated`` flag from the tail report; and no occurrence reinsurance, whose severity rides the output grid. A severity-fit guard then marks the row applicable only when a single occurrence fits the windowed extent ``[0, N*bs]``.

    A thick-right, thin-left concentrated band floors its upper edge by ``sbj_hi`` before the guard runs, which grows the extent enough that a single heavy occurrence fits. That is what reclaims the heavy-positive book whose severity would otherwise overflow a tight band. The left lift is trustworthy precisely because such a book's aggregate left tail is thin: reaching low is a large-deviation conspiracy of many small claims, with no subexponential reach on the left. The guard still rejects the ``fixed``-1 and ``approximate`` trap, a single thin severity already at the aggregate mean.

``sbj``, the single-big-jump extent floor
    Not a standalone sizing of the grid but a floor on the selected window's extent, recorded as its own row for inspection. For a subexponential severity the aggregate's far tail is one big claim on an otherwise typical bulk, :math:`P(S > x) \sim \mathsf E[N] P(X > x)`. To cover the aggregate to :math:`p^*` the severity is probed at the :math:`\mathsf E[N]`-adjusted depth :math:`p^{**} = 1 - (1 - p^*)/\mathsf E[N]`, so one claim reaches there and the other :math:`\mathsf E[N] - 1` are typical. It is not :math:`N q_X`, which would assume every claim is huge. The extent is then floored at

    .. math::

        \mathrm{sbj\_hi} = \mathsf E[S] - \mu_X + q_X(p^{**}),

    mirrored on the low side for a signed severity. :math:`p^{**}` deepens with :math:`\mathsf E[N]` and would underflow double precision for a large count, so :math:`1 - p^{**}` is floored at ``sbj_tail_floor``, and :math:`q_X` is capped by any finite policy limit. The floor is the computational twin of the single-big-jump principle the tail classifier names, so it is gated on a thick (subexponential or heavier) tail through ``_loss_tail_classes``. It is a no-op for lighter classes, but explicitly so.

.. _bs selection:

Selection and combination
-------------------------

The methods combine in a fixed order.

1. **Pick the base window.** Take ``exact_discrete`` if present and reachable, meaning at least one corner's :math:`\log_{10}` attainment probability is at or above ``exact_discrete_reach_logp``; else ``bounded_small`` when it is no wider than 1.5 times the moment window; else ``moment``. A ``dfreq`` or ``fixed`` crossed with ``dsev`` book whose combinatorial support is fictional, both corners below the floor, as with a large fixed count, therefore skips its exact support and sizes on the moment window, avoiding the ``bs`` coarsening that would alias the severity.

2. **Let ``windowed`` override.** If the windowed row applies, meaning a single severity fits its extent, and it captures a reach the base pick would clip, or it is no coarser than the base pick (``bs_windowed <= bs_base``), it wins, trading the empty space below the band for a centered placement. The comparison is ``<=`` rather than ``<`` so that a band clearing ``0`` can win on placement alone, even when ``bs`` is unchanged.

3. **Apply the single-big-jump floor** to the selected ``moment`` or ``windowed`` window, and only when ``bs`` is free, since a pinned ``bs`` is the caller's grid. The two regimes diverge here.

   For a signed book this is correctness. The grid is widened to cover ``[sbj_lo, sbj_hi]`` unconditionally. The bulk ``bs`` is kept if the reach fits the ``log2`` budget; otherwise ``bs`` is coarsened within the cap until it does. No signed mass is left off-grid.

   For a positive book this is refinement. The window is extended up to ``sbj_hi`` only when it fits at the bulk ``bs`` within the requested ``log2``, so a generous ``log2`` is captured fully and finely. If it does not fit, the moment window is kept, because clipping a tiny far tail beats coarsening the bulk to uselessness, and a visible :class:`~aggregate.constants.DefectiveDistributionWarning` reports the reach, the estimated clipped mass, and the ``log2`` that would capture it. The same information is stashed in ``Aggregate._bs_clip``. The ``log2`` budget, whether explicit, hinted or default, is never silently grown, and a pinned ``bs`` is never coarsened.

4. **Balance the padding**, for a windowed selection with an automatic origin and a non-affine aggregate. A windowed band is first placed band-bottom, with all the power-of-two slack above it. The slack is then redistributed by tail shape: an asymmetric band puts about three quarters of the slack on the thick side (``WINDOW_SLACK_THICK``), so a thick-right, thin-left book pushes it right and the mirror for thin-right, thick-left, while a symmetric band, both tails thick or both thin, centers with the loss or payoff convention skew (``window_pad_skew``) as the tie-breaker.

5. **Reflect or shift for ``pnl``**, for affine aggregates. The loss FFT runs on the non-negative grid sized above; the finished aggregate is reflected and shifted, and a tight two-sided P&L display window is computed from the affine moments. The ``used`` row reports that realized P&L grid.

The ``used`` row is the realized power-of-two grid, ``x_max = x_min + 2**log2 * bs``, so a 701-point support padded to 1024 reads ``x_max`` at the grid top.

Convention orientation, loss against payoff
-------------------------------------------

A windowed placement is oriented by the aggregate's sign convention, the ``value_type`` role, where ``loss`` means more is worse and ``payoff`` means more is better. It is read from the boolean ``_is_loss_value``, never from the label string, and is overridable per call through ``update(window_convention=...)``. The convention sets two things.

Per-edge coverage
    The protected tail, the upper edge for a loss and the lower for a payoff, is covered deep at ``window_nines`` nines to avoid clipping the priced tail. The cheap tail is trimmed shallow at ``window_nines_trim``.
Padding skew
    Only as a tie-breaker for a symmetric band: :math:`f = 0.5 - \mathrm{window\_pad\_skew}` for a loss, giving more room on the priced right, and :math:`0.5 + \mathrm{window\_pad\_skew}` for a payoff. An asymmetric band ignores the convention and skews to the thick tail instead.

A loss and its mirror-image payoff therefore place their bands as reflections of one another when the tails are symmetric. An asymmetric, thick-right and thin-left book skews to the thick side under both readings, while the per-edge coverage still follows the convention.

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
     - tail probability for the moment and bounded windows. A value above 1
       reads as nines.
   * - ``window_nines``
     - ``12``
     - coverage of the protected window edge, and of the moment window.
   * - ``window_nines_trim``
     - ``6``
     - coverage of the cheap, trimmed edge of a windowed book.
   * - ``window_pad_skew``
     - ``0.1``
     - fraction the padding is skewed toward the priced tail.
   * - ``sbj_tail_floor``
     - ``1e-14``
     - deepest severity probe depth :math:`1 - p^{**}` for the single-big-jump
       floor. Guards :math:`q_X(p^{**}) \to \infty` for a large
       :math:`\mathsf E[N]`.
   * - ``exact_discrete_reach_logp``
     - ``-20``
     - :math:`\log_{10}` attainment-probability floor for the
       ``exact_discrete`` support corners. When both corners fall below it the
       exact support is fictional, its extremes unreachable, and the method is
       rejected in favor of the moment window. Well below floating-point
       noise, and well above a genuine small-count discrete book.

The module constant ``WINDOW_LOG2_GROWTH = 4`` bounds how far a windowed integer-lattice book may grow ``log2`` past the cap to preserve an exact ``bs``, so a high-mean ``dsev`` keeps ``bs = 1`` rather than coarsening off its lattice. It never applies to continuous severities.

.. _bs feasibility:

Feasibility: when no bucket size works
--------------------------------------

Everything above is a choice among grids. This is the question of whether any grid in the budget will do, and it is answered separately, after the choice is made, because the answer does not depend on which method won.

Start from what discretization actually costs the **mean**. Under the ``round`` scheme bucket zero collects every loss below ``bs/2`` and places it at exactly ``0``, so the mean supplied there is lost outright; above the grid top the mean is lost to truncation. Those are the only two first-order losses. Both are naturally measured under the **size biased** law :math:`P_1`, defined by :math:`dP_1/dP = y/\mathsf E[Y]`, which weights by the loss amount: the question is not how much *probability* sits below half a bucket but how much *mean* does. Its cdf comes straight out of quantities the library already computes exactly,

.. math::

    F_1(t) = \frac{1}{\mathsf E[Y]}\int_0^t y\,dF(y) = \frac{\mathrm{LEV}(t) - t\,S(t)}{\mathsf E[Y]},

because :math:`\mathrm{LEV}(t) = \mathsf E[\min(Y,t)]` is that same integral plus the censored block :math:`t\,S(t)`. So a mean accurate to relative :math:`\delta` needs

.. math::

    \frac{bs}{2} \le q_{\delta/2}(P_1) \qquad\text{and}\qquad n \cdot bs \ge q_{1-\delta/2}(P_1),

and dividing one by the other gives the requirement on the bucket count alone, with ``bs`` eliminated:

.. math::

    \log_2 n \ge \log_2 \frac{q_{1-\delta/2}(P_1)}{2\,q_{\delta/2}(P_1)}.

For an **unlimited lognormal** this has a closed form, and the striking part is what drops out of it. :math:`P_1` is again lognormal, :math:`LN(\mu + \sigma^2, \sigma)`, so the quantile ratio is :math:`e^{2z\sigma}` with :math:`z = \Phi^{-1}(1-\delta/2)`, the mean cancels entirely, and only :math:`\sigma` survives:

.. math::

    \log_2 n \ge \frac{2z\sigma}{\log 2} - 1 \approx 11.23\,\sigma - 1 \qquad (\delta = 10^{-4}).

Built on exactly the grid the formula prescribes, the realized error lands on the target:

.. list-table::
   :header-rows: 1
   :widths: 34 14 20 24

   * - severity
     - :math:`\sigma`
     - ``log2`` required
     - realized mean error
   * - ``lognorm 200 cv 2``
     - 1.269
     - 13.2
     - 2.0e-6
   * - ``lognorm 200 cv 5``
     - 1.805
     - 19.3
     - 2.6e-5
   * - ``lognorm 200 cv 10``
     - 2.148
     - 23.1
     - 1.9e-5
   * - ``lognorm 8.501 cv 14.624``
     - 2.317
     - 25.0
     - 1.8e-4

Against a working ``log2 = 16`` the practical boundary sits near :math:`\sigma = 1.51`, a CV around 3. Below it an unlimited lognormal is routine; above it no bucket size works, and a finer one is worse. The connection to Mandelbrot's moment localization argument, which explains why this is a property of the lognormal rather than an accident of this grid, is written up in the monograph.

**A limit is the fix, and thickness alone is not the problem.** ``N.US.Hurricane``, ``1e12 xs 0 sev exp(19.595) * lognorm 2.581``, has :math:`\sigma = 2.581`, thicker than the cat model in the table, and is perfectly feasible: the limit truncates :math:`P_1` so its upper quantile is the limit itself, the required ``log2`` is 15.7, and a 16 grid holds it. Its chosen ``bs`` is still ten times coarser than the body wants, which is why its mean is off by 0.089%, but that is a **coarse** grid, not an impossible one. The distinction is the whole point of the reading.

The reading is computed after selection and stashed on the aggregate as ``_bs_feasibility``, beside ``_bs_clip`` and ``_bs_snap``. It carries ``bs_max`` (the coarsest bucket whose bottom bucket does not swallow more than :math:`\delta/2` of the mean), ``reach`` (how far the grid must extend to keep the same share at the top), ``log2_required``, ``lost_at_zero`` (the share of the mean the first half bucket takes, which is the number that makes the situation legible), and the grid it is measured against. ``log2_required`` is **not** ``_bs_window_df``'s ``log2_need`` column: that one is the exponent a candidate *window* needs at its own ``bs``, while this is the exponent the *severity* needs before its mean can be reproduced at all, and no choice of ``bs`` moves it. For a severity mixture the binding component, the one with the smallest ``bs_max``, is the one reported, because the shared grid has to serve all of them.

``bs_description`` gains a clause and ``bs_explanation`` the full account whenever ``log2_required`` exceeds the realized ``log2`` by more than half an exponent:

.. ipython:: python
    :okwarning:

    from aggregate import build
    cat = build('agg CatDemo 1.74 claims sev lognorm 8.501 cv 14.624 poisson')
    print(cat.bs_description)
    print(cat.bs_explanation)

The reading is exact or absent, never estimated. It is available for the severity kinds whose partial expected values are computed in closed form, ``lognorm``, ``gamma``, ``pareto`` and ``expon``, which covers mixed exponentials, and it reports nothing for a discrete or histogram severity (exact on its own lattice, so the question does not arise), a signed or reflected law, or any other continuous family. Not knowing is not the same as knowing the grid is fine, so a missing reading claims nothing in either direction.

Deliberately **not** done, on the author's ruling of 2026-08-21: the sizer is not capped at the resolution requirement, because buying reach is the decision and it stays; and one-moment local moment matching, which would spread each bucket's mass across its two bounding lattice points and make the discretized mean exact at any ``bs``, is refused for a gross severity. It would repair the reported number while leaving the model's mean furnished by a region no one has an opinion about, which is worse than an obvious error because it is silent. The existing use of mean-preserving scatter for reinsurance rebucketing stands, where the grid is forced and the means have to work.

Inspecting a choice
-------------------

After ``update``, or after ``build``, ``a._bs_window_df`` holds one row per method plus ``sbj`` and ``used``. The columns are ``applies`` (eligible, the severity fits, and the window is finite), ``x_min`` and ``x_max`` (the method's window), ``W`` (width), ``bs``, ``log2``, ``coverage`` (the nines used), ``note`` (provenance, for example ``; sbj floor`` when the floor bound, or ``; non-finite window (rejected)`` for a method whose window had an infinite edge), and ``selected``. Reading it answers, at a glance, why a grid was chosen and what the alternatives were:

.. ipython:: python
    :okwarning:

    from aggregate import build
    a = build('agg T5 5000 claims sev lognorm 100 cv 2 poisson')
    a._bs_window_df[['applies', 'x_min', 'x_max', 'bs', 'log2', 'selected']]

``T5`` is the motivating case for the tail-aware work. Its ideal grid is roughly ``[400k, 1.2M]``, and it gets there: ``windowed`` applies and wins, lifting ``x_min`` off the floor and taking ``bs = 20``, coarse enough that a single heavy occurrence fits the band, while the upper edge is floored by the single-big-jump reach so the far tail is captured rather than clipped. The ``moment`` row shows the 0-based alternative, which at ``bs = 10`` would have topped out at 655,360, well below the ``sbj`` reach of 1,234,033, and clipped.

The narrative properties say the same thing in prose. ``bs_description`` is the short summary, giving the winning method, ``(bs, log2, x_min)``, the grid top, any clip, and any feasibility shortfall; ``bs_explanation`` is the verbose form, giving the aggregate tail one-liner, which methods applied and why the winner won, how to widen a clipped tail, and the full account when the severity cannot be reproduced on the grid at all (:ref:`bs feasibility`).

.. ipython:: python
    :okwarning:

    print(a.bs_description)

Grid selection is the most consequential numerical decision in the library, it is subtle, and it is a frequent source of user confusion, which is why it carries both a complete private frame and a culled public one (``bs_window_df``, also on :class:`Portfolio`) alongside the prose. The module functions ``bs_describe`` and ``bs_explain`` give ANSI-colored variants.

.. _bs sharpen:

Auditing the choice afterwards with ``sharpen``
-----------------------------------------------

Everything above chooses a grid, and it does so from the analytic moments, before any FFT has run. :meth:`~aggregate.distributions.Aggregate.sharpen` audits that choice after the fact: it re-updates the object on neighboring ``(bs, log2)`` cells, scores each against the analytic moments, and moves when the win is worth having. It is on :class:`~aggregate.distributions.Aggregate` and :class:`~aggregate.portfolio.Portfolio`.

The two halves answer different questions. The sizer asks "given only what I can compute in closed form, where should the grid go?", with no realized density to look at. ``sharpen`` asks "now that it has run, was that right?", with the estimated moments, the realized mass and every warning the update raised in hand.

The score
~~~~~~~~~

``validation_score`` is six terms, severity and aggregate mean, CV and skewness, each read from the canonical ``stats_df['error']`` and divided by its own validation tolerance: ``eps`` for a mean, ``10 eps`` for a CV, ``100 eps`` for a skewness, exactly the multiples ``valid_aggregate`` applies. They combine in a power mean, Euclidean (``power=2``) by default; ``power=1`` averages and ``power=inf`` reports the worst single term.

Dividing by the tolerance is what makes the number useful. The units are tolerance, so ``score <= 1`` means the object passes validation and ``1`` sits exactly on the pass boundary. Where ``valid`` says whether a line was crossed, this says by how far, which is what makes two grids comparable. A consequence worth knowing: the score does not move when ``validation_eps`` is changed.

A term is live only when its theoretical value is finite and above the noise floor, the same test validation itself applies: a symmetric severity has no skewness to validate against, and a deterministic one no CV. The theoretical moments do not depend on the grid, so the live set is stable as ``bs`` and ``log2`` vary, which is precisely what makes scores comparable across a probe. Averaging rather than summing then keeps objects with different numbers of live terms on one scale.

The probe
~~~~~~~~~

One row per ``log2``, and within each row a line search out from the current bucket: ``bs`` is doubled until the score stops improving, then halved likewise, capped each way by ``bs_limit``. The rows come out ragged, which is why ``sharpen_df`` is a tidy frame rather than a matrix; cells never visited read ``NaN``.

The line search is well posed because the score is single-troughed in ``bs`` at fixed ``log2``. With extent ``W = bs * 2**log2``, a coarser bucket buys extent and loses resolution, so the two error families trade off and there is one turning point. Stopping at the first cell that fails to improve assumes exactly that. A walk that instead runs out of ``bs_limit`` while still improving records so in its ``note``, a different fact from turning: a re-run will keep going.

``log2`` and ``log2 - 1`` are searched up front. ``log2 + 1`` is searched only if neither reaches the target, since it costs twice as much per cell and under the selection rule below it is usually not consulted.

The geometry is the point. A row is constant resolution; an anti-diagonal is constant extent. Cells sharing an extent differ only in resolution, so reading the two directions together says whether the grid is extent-limited, and should be widened, or resolution-limited, and should be refined. That is also why the bucket steps are a strict factor of two rather than rungs of the ``round_bucket`` ladder, whose 1.25 and 1.6 factors are too fine to move the error and out of step with ``log2`` plus or minus one.

Every cell runs inside its own guard: one that raises is recorded as ``NaN`` with the exception text in its ``note``, and the sweep completes.

Selection: thrift, gated on soundness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rule is never to make the caller pay more than they already are: take the best score among the cells that do not grow ``log2``. Reducing ``log2`` is a bonus rather than a goal, so it is picked up by the tie rule rather than chased. Cells within ``SHARPEN_FALLBACK_SLACK`` (1.25) of the best count as tied; ties break on ``log2`` first, so a free memory saving is taken when the score is genuinely a wash, then on :math:`|d\_bs|`, so a competitive center wins over an equally good move and the grid is not churned for nothing. ``log2`` grows by one only when nothing at the current size or smaller reaches ``good_enough`` and something at ``log2 + 1`` does. If nothing anywhere reaches the target, the best cell overall is taken, still thrift-ordered, so the descent starts and a re-run continues it.

``good_enough`` (default 0.5) is the only judgment knob, and it does two things. As a probe gate, a grid already at or under it, and sound, is left alone and nothing is run at all. As a growth trigger, it is the target that ``log2 + 1`` has to reach. A score is never negative, so ``good_enough=0`` reads as "probe everything, never grow", which is why there is no separate ``force`` argument.

A grid that loses mass off its top end is disqualified, however well it scores. Soundness is a gate, not a term in the score, for two reasons. A moment score cannot see lost mass and never will: far-tail mass is negligible for the first three moments and decisive for tail pricing, so the two measures are orthogonal and the gate is the only thing that catches it. And it could not be a penalty even if you wanted one, because every term in the score divides by its own tolerance and a deficit has no tolerance to divide by; its cost is qualitative rather than a matter of degree. Mass that runs off the top is dropped, not wrapped, so the realized law sums to less than one and forwards ``S = 1 - cumsum`` differs from backwards ``S`` by exactly the missing amount: two correct-looking pricing routes disagree in the tail. The threshold is ``VALIDATION_NOISE``, deliberately tighter than the ``DEFICIT_MATERIALITY`` floor at which :class:`~aggregate.constants.DefectiveDistributionWarning` fires, because the two answer different questions. Choosing among candidate grids, losing no mass at all is free to insist on, so the probe insists. Interrupting the reader is not free, so the warning waits for a deficit large enough to move a price. Because soundness sits outside the thrift ordering, it composes: a deficit becomes a reason to grow ``log2``, which is its cure. Three columns carry it, ``deficit``, ``defective`` and ``warns``, and the narrative reports how many better-scoring cells the gate threw out.

Discrete severity: the bucket is pinned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When every severity atom is a whole number of buckets the discretization is already exact and no bucket change can help: a coarser bucket scatters the atoms off their own values, a finer one only wastes grid. The bucket is therefore pinned and only ``log2`` is probed, which is what a failing discrete object actually needs, its problem being extent. The test is the greatest common divisor of the integer severity atoms (``Aggregate._severity_lattice``, taken across units for a :class:`Portfolio`); it declines as soon as any component is continuous or off the lattice.

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
     - center of the probe. An explicit center that is not where the object
       sits moves it there first, so the center row describes a real state.
   * - ``good_enough``
     - ``0.5``
     - target score, in tolerance units. Probe gate and growth trigger, as
       set out under selection. ``0`` means probe everything and never grow.
   * - ``log2_cap``
     - ``20``
     - ``log2`` is never grown past this.
   * - ``bs_limit``
     - ``16``
     - largest factor by which the line search may multiply or divide ``bs``.
       A power of two, so ``16`` permits four doublings each way and a grid
       wrong by two orders of magnitude is reachable in one call.
   * - ``power``
     - ``2``
     - norm exponent used to combine the six terms.
   * - ``execute``
     - ``True``
     - ``True`` leaves the object on the chosen cell; ``False`` restores the
       original grid, so the probe is pure diagnosis.

The module constants are ``SHARPEN_LOG2_FLOOR = 8``, the lowest ``log2`` the probe will drop to, below which the comparison stops meaning anything; ``SHARPEN_FALLBACK_SLACK = 1.25``, the tie band; and ``SHARPEN_BS_LIMIT = 16``, the default ``bs_limit``. The tie band is deliberately tight: a genuine improvement here is orders of magnitude, not percent.

Comparability is enforced by harvesting the settings that must not drift. ``update`` stamps ``sev_calc``, ``discretization_calc``, ``normalize`` and ``padding`` from its arguments, so a bare ``update(log2=..., bs=...)`` would silently reset all four. They are read off the object once, up front, and passed to every cell: the object comes back as it was, and the cells differ in ``bs`` and ``log2`` and in nothing else.

Inspecting a probe
~~~~~~~~~~~~~~~~~~

``sharpen_df`` is one row per cell, indexed by the offsets ``(d_bs, d_log2)``, so the picture is one unstack away. Take a book auto-sized to ``bs = 2``, ``log2 = 16``, force it onto a grid far too small to hold it, and probe:

.. ipython:: python
    :okwarning:

    a = build('agg Sharp 100 claims sev lognorm 100 cv 2 poisson')
    a.validation_score
    a.update(log2=14, bs=1/8)          # far too small
    a.validation_score
    a.sharpen()
    a.sharpen_df.score.unstack('d_log2').round(3)

Down a column is constant grid size, so it is the resolution question; along an anti-diagonal is constant extent. Here the improvement runs up the ``d_bs`` axis and does not turn, so the book is resolution-starved in the wrong direction: the bucket was far too fine for the extent it had to cover. The winner is the top-right corner, and it sits on the ``bs_limit`` still improving, which ``sharpen_description`` says out loud:

.. ipython:: python
    :okwarning:

    print(a.sharpen_description)

Re-running re-centers the probe on the new grid and continues from there. The soundness gate is visible in the same frame, alongside the score:

.. ipython:: python
    :okwarning:

    cols = ['bs', 'log2', 'score', 'deficit', 'defective', 'selected']
    a.sharpen_df[cols].tail(4)

The full column list is ``bs``, ``log2``, ``extent``, ``x_min``, ``score``, the six per-term ``u_sev_mean`` through ``u_agg_skew``, ``aliasing`` (the aggregate-mean over severity-mean error ratio, recorded alongside the score but deliberately not part of it, being the specific signature of too small a ``bs``), ``deficit``, ``defective``, ``validation`` (the one-line verdict), ``warnings`` and ``warns``, ``seconds``, ``selected`` and ``note``. ``sharpen_description`` and ``sharpen_explanation`` are the short and long narrative forms, in the same style as ``bs_description`` and ``bs_explanation`` above.

The discrete case shows the pinned bucket. Setting ``good_enough=0`` forces a probe on an object that is already exact:

.. ipython:: python
    :okwarning:

    d = build('agg D6 dfreq [3] dsev [1:6]')
    d.sharpen(good_enough=0)
    print(d.sharpen_description)

Reach, and what it does not cover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``update(..., sharpen=True)`` opts a single update into auto-sharpening. It is off by default and not turned on by ``build``, because a probe costs eight or more extra updates.

There is no ``BivariateAggregate.sharpen``: the probe is quadratic in the joint grid. There is deliberately no ``PnL.sharpen`` either, since it would leave a built ledger sitting on a stale grid; a P&L sharpens through its engine instead.

A :class:`~aggregate.portfolio.Portfolio` scores the total only, so a well resolved total can still hide a poorly resolved unit. ``sharpen_explanation`` says so rather than leaving it to be discovered.
