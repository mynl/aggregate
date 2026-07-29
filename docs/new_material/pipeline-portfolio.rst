.. _pipeline portfolio:

####################################
The Portfolio Computation Pipeline
####################################

*Draft for the developer documentation. Describes the* ``Portfolio`` *pipeline
in* ``portfolio.py`` *as of the* ``REFACTOR`` *branch: construction, update, the
* ``add_exa`` *kappa engine, the distortion / augmented-df machinery, the lifted
vs linear natural allocation, and the "switcheroo". Distils the journey-of-
discovery comments, ends with recommendations, and proposes a before/after
consistency harness for the big dataframes. Line numbers are approximate;
method names are the stable anchors.*

.. contents::
   :local:
   :depth: 2


********
Overview
********

A :class:`Portfolio` is a collection of :class:`Aggregate` units plus the
machinery to (a) combine them into a portfolio total under the independence
copula, (b) compute conditional expectations :math:`\kappa_i(a)=\mathsf
E[X_i\mid X=a]` (the "kappas") that drive capital allocation, (c) apply spectral
distortions to price and allocate, (d) **generate multivariate samples** —
optionally with an induced rank-correlation by reordering (Iman–Conover
"shuffle") — and (e) run the **switcheroo**: rebuild the object around an
empirical sample's kappas (below). The pipeline::

    Portfolio.__init__            collect units, build stats_df (theoretical)
        │
        ▼
    update(log2, bs)
        │
        ├─ per unit: agg.update_work(xs)        each unit's FFT (+ its reinsurance)
        ├─ p_total = ift(Π ft(unitᵢ))           combine under independence copula
        ├─ ft_nots[line] = Π_{j≠line} ft(j)     "all but this line" FTs
        ├─ remove_fuzz()
        ├─ add_exa(density_df, ft_nots)          THE KAPPA ENGINE → exeqa_*, exa_*, …
        └─ empirical total moments → stats_df[empirical, error]
        │
        ▼
    calibrate_distortion(s)        solve distortion shape to hit a premium/COC target
        │
        ▼
    apply_distortion / augmented_df            gS, gp_total, exag_*, T.*/M.* columns
        │                                       (cached per distortion name)
        ▼
    price / analyze_distortion(s) / pricing_at  pricing readouts (lifted | linear)

**The switcheroo (a big idea, not a side path).** A Portfolio can be rebuilt
around an empirical multivariate **sample**: instead of carrying the full joint
sample everywhere, ``create_from_sample`` / ``add_exa_sample`` **replace the
density frame with one whose kappas** :math:`E[X_i\mid X=a]` **are estimated
directly from the sample**. The sample's entire dependency structure that
matters for allocation *is* the kappa, so once you have it, every downstream
calculation — allocation, distortion pricing, the whole ``augmented_df``
machinery — runs unchanged on the sample-based object. That is the win: a
high-dimensional sample collapses to a one-dimensional family of conditional
means, and "all manner of things simplify". Mechanics in Phase 4.

One genuinely *non-core* path:

* **add_exa_details / add_eta_mu** — EPD and second-priority (η-μ) columns.
  **Called only from** ``pedagogy.py`` (the ``plot_twelve`` figure), never from
  ``update``.


***********************
Phase 1 — Construction
***********************

``__init__`` (≈ 187–337)
========================

Accepts a heterogeneous ``spec_list``: :class:`Aggregate` objects, spec dicts,
``(kind, spec)`` tuples, names to look up in an :class:`Underwriter`, or a
single :class:`pandas.DataFrame` of samples. Each resolved unit is appended to
``agg_list``/``line_names``, **set as an attribute** (``self.<line> = agg``), and
its theoretical moments accumulated into a ``MomentAggregator``.

* ``line_names`` may not contain ``total`` (reserved).
* DataFrame input is stored as ``sample_df`` and the units become discrete
  histogram aggs (rounded to 8 dp "so close values don't merge" — i.e. respect
  the ``2**-30 ≈ 1e-9`` discrete-grid resolution).
* ``_build_stats_df`` mirrors ``Aggregate.stats_df`` minus the ``independent``
  column (there is no portfolio-level mixed/independent distinction; a
  portfolio's theoretical column is named **``total``**, not ``mixed``).

As with Aggregate, **no FFT happens in construction**.


***********************
Phase 2 — Update
***********************

``update`` (≈ 1347–1494)
========================

#. Resolve ``bs`` (``best_bucket`` → binary-fraction rounding when ``bs=0``).
#. ``xs = linspace(0, N·bs, N, endpoint=False)``.
#. For each unit, call ``agg.update_work(xs, …)`` (which also applies that
   unit's reinsurance) and collect ``agg.ftagg_density`` and ``agg.agg_density``.
#. **Combine** under independence: ``ft_all = Π ftagg_densityᵢ`` and
   ``p_total = real(ift(ft_all))``.
#. **"not-line" FTs**: ``ft_nots[line] = ft_all / ft_line`` (or rebuilt by
   product when ``ft_line`` has zeros) — the FT of "everything except this
   line", needed by the kappa engine.
#. ``remove_fuzz`` (if enabled), then ``add_exa``.
#. Empirical total moments by **plain summation** ``Σ p·xᵏ`` (explicitly *no*
   tail-mass correction, "to match the legacy PEG regression baseline"), written
   to ``stats_df['empirical']`` / ``error`` via ``_write_empirical_stats``.

``add_exa`` — the kappa engine (≈ 1870–2117)
============================================

This is the heart of the Portfolio. For each line it computes (all on the
``density_df`` grid):

**The money calculation** — :math:`\kappa_i(a)=\mathsf E[X_i\mid X=a]`::

    exeqa_i = real( ift( ft(loss · p_i) · ft(p_not_i) ) ) / p_total

i.e. the conditional expectation falls straight out of one FFT pair because
``X_i`` and ``X_{-i}`` are independent. Where ``p_total < eps`` the estimate is
unreliable and set to 0 ("JUNE 25: this makes a difference!").

From ``exeqa_i`` it derives the full allocation column family:

* ``exlea_i`` = E[Xᵢ | X≤a]  (cumsum of ``exeqa·p_total`` / F)
* ``exgta_i`` = E[Xᵢ | X>a]
* ``exi_xgta_i`` = E[Xᵢ/X | X>a]  (reverse-cumsum; the tail-fill is "the John
  Major problem" — the last bucket knows nothing beyond the array)
* ``exa_i`` = E[Xᵢ(a)] = ``(S · exi_xgta_i).shift(1).cumsum()·bs`` — the
  **natural-allocation loss** to line *i* at assets *a*.

Totals: ``exa_total = (S.shift(1).cumsum)·bs``, ``S = 1 - F`` forwards.

A recurring fragility is the ``loss_max`` blanking heuristic: where
``exlea_total > loss`` (impossible, hence numerical), the values up to
``loss_max + mult·bs`` are blanked, with ``mult ∈ {1,10,100}`` chosen by frame
length (``<1100 / <15000 / else``). Two ``# TODO What is this crap?`` comments
mark it.

``add_exa_details`` / ``add_eta_mu`` (≈ 2130–2289) — **plot_twelve-only**
=========================================================================

Adds EPD columns (``epd_0/1/2_*``), the η-μ "not-line densities" (``ημ_*`` and
their ``exeqa_ημ_*`` / ``exa_ημ_*`` derivatives, for second-priority /
equal-priority analysis), and reimbursement-effectiveness columns
(``e1xi_1gta_*``). The EPD interpolation functions were already removed in the
refactor. The **only** caller is ``pedagogy.plot_twelve`` (via its helper, which
needs the ``M.*`` marginal columns and the ``eta_mu=True`` derivatives). It is
**not** kept as-is for that: the job is to determine *exactly* which columns
``plot_twelve`` actually consumes — it is **not** everything produced here — and
compute only those; the rest goes. See Recommendations.


***************************************
Phase 3 — Distortions & pricing
***************************************

Calibration
===========

* ``calibrate_distortion(name, …)`` resolves the asset level, premium target,
  and the ``S`` vector over ``[0, a]`` (truncating at the first zero, recording
  ``ess_sup`` and logging "Mass issues" if the support ends early), asserts
  ``S`` is strictly positive and weakly decreasing, then **dispatches to the
  ``Distortion`` subclass** whose ``calibrate`` runs the Newton iteration. The
  per-distortion math now lives in ``spectral.py``, not here.
* ``calibrate_distortions(coc, p|a)`` inverts COC → LR → premium and calibrates
  the standard set ``[ccoc, ph, wang, dual, tvar]``, storing them on
  ``self.distortions`` and an audit ``distortion_df``.

``apply_distortion`` / ``_build_augmented`` (≈ 2476–2930)
=========================================================

``apply_distortion`` is a thin **cache** keyed by distortion name
(``_augmented_dfs``), invalidated on ``update``. The work is ``_build_augmented``,
a pure builder that returns a copy of ``density_df`` augmented with:

* ``gS = g(S)``, ``gp_total = -diff(gS, prepend=1)`` — the **risk-adjusted
  probabilities**.
* per line, the risk-adjusted allocation premium
  ``exag_i = (exi_xgtag_i · gS).shift(1).cumsum()·bs`` where
  ``exi_xgtag_i`` is the distorted analogue of ``exi_xgta_i`` (tail-filled with
  ``last_x`` so tail mass is not lost).
* ``T.*`` (ground-up / "total") and ``M.*`` (marginal, per-unit-width-layer)
  columns: ``T.M = exag - exa``, ``T.Q`` from the layer ROE, etc. The
  **critical insight** (kept as a comment) is that *layer ROEs are equal across
  all lines by law-invariance*, with ``g'(1)`` (L'Hôpital) giving ``ROE(1)``.
* ``efficient=True`` (default) computes only the ``T.M``/``T.Q`` needed for
  pricing; ``efficient=False`` also builds the full ``M.*`` marginals.

Two numerically important steps: ``S<0`` is clamped to 0 (logged), and the frame
is **truncated** at the last index where ``|Σ exeqa_i − loss| / loss < 1e-4``
(the ``exeqa_err`` test) — beyond that point the conditional expectations are
unreliable, so augmented_df is shortened.

**Assessment (is it OK / duplication / better way?).** Mostly sound, but **yes,
there is duplication — and a likely bug.** The ``efficient`` and full branches
each recompute the same total-level block (``exag_total``, ``M.M_total``,
``M.Q_total``, ``M.ROE_total``, ``roe_zero``); the comment at the top of the
efficient branch even says "duplicated and edited code from below". Worse, the
two branches use **different fallback ROEs** where ``M.Q_total == 0``
(``gS == 1``): the efficient branch sets ``gprime1 = g_prime(1)`` while the full
branch sets ``gprime1 = 1/g_prime(1) − 1``. The L'Hôpital limit of the layer ROE
``(gS−S)/(1−gS)`` as ``S→1`` is ``(1−g'(1))/g'(1) = 1/g'(1) − 1`` — so the **full
branch is correct and the efficient (default!) branch looks wrong** in the
``gS == 1`` tail, where it can produce a different ``T.Q``. **Better way:**
compute ``gS`` / ``gp_total``, the per-line ``exag`` / ``exi_xgtag``, and the
total-level block (with the *one* correct ``gprime1``) **once**; let
``efficient`` gate only the *extra* per-line ``M.*`` marginals and the η-μ
columns. That removes the duplication and the discrepancy in a single move.
(This is reasoning, not yet a test — verify against a mass distortion before
changing, and it is a prime case for the before/after harness.)

Lifted vs linear natural allocation
===================================

This is the conceptual crux and the part most in need of a settled v1.0 story.

* **Lifted** (``price(..., allocation='lifted')``, the current default for
  legacy reasons). Reads the allocation straight from the augmented_df row at
  ``a_reg``: per line ``L=exa_i``, ``P=exag_i``, ``M=T.M_i``, ``Q=T.Q_i``. The
  premium uses the **risk-adjusted** weights ``gp_total`` all the way out into
  the tail. With a distortion that has a **mass at zero** (e.g. CCoC) on an
  **unbounded** distribution, essentially all the distortion weight lands on the
  single last bucket — highly unstable and grid-dependent.

* **Linear** (``allocation='linear'``). Collapses *all states ≥ a* to the asset
  level using **objective** probabilities, then prices the layers below ``a``::

      exp_loss   = ((ps·bs)/loss · exeqa)[::-1].cumsum()[::-1]
      alloc_prem = ((gps·bs)/loss · exeqa)[::-1].cumsum()[::-1]
      capital    = (alloc_prem − exp_loss) · rcoc

  The tail-collapse replaces ``exeqa`` at ``a`` with ``a · exi_xgta(a-bs)`` when
  ``sf(a) > 1 - Σp`` (there is genuine tail beyond ``a``). Because the collapse
  uses **objective** (not risk-adjusted) probabilities, it sidesteps applying the
  distortion far out in the tail — *much* more stable, and the author's preferred
  approach (PIR, "linear natural allocation").

**One-sweep status.** The allocation across **all lines** is already a single
vectorised reverse sweep — ``[::-1].cumsum()[::-1]`` runs over the whole
``exeqa`` frame (every line column at once). What is **not** shared is the loop
over *distortions*: ``exp_loss`` (the objective-probability leg) is
distortion-independent yet recomputed for each distortion — **hoist it out** of
the loop. With that, linear costs essentially the same as lifted, so there is
**no material performance reason to prefer lifted**; linear can be the default
for free. The only remaining difference is the tail treatment — which is exactly
the unbounded-+-mass question.

**v1.0 decisions** (see Recommendations / the portfolio plan): **linear is the
default**; lifted stays available as an option; a *lifted* NA on an *unbounded*
distribution with a *mass* distortion is **refused** ("say NO!") rather than
returning an unstable answer; and the current "ugly" mass/tail-support code is
replaced by that refusal. Boundedness is taken from the spec where provable, with
a user ``bounded`` certification as the override/shortcut.

Readouts
========

* ``pricing_at(dist, p|a)`` — per-line pricing row at one asset level (warms the
  cache; ``Q_total = a − exag_total`` exactly). **The canonical column order is
  the pentagon: amounts then ratios —** ``L M P Q a | LR PQ ROE`` (``LR = L/P``,
  ``PQ = P/Q``, ``a = P + Q``; see ``pentagon.py``). The current
  ``['L','LR','M','P','PQ','Q','ROE']`` order (and the missing ``a``) is wrong
  and should be fixed; this row *is* the canonical ``PricingResult`` shape, used
  everywhere.
* ``analyze_distortion`` / ``analyze_distortions`` — single / multi-distortion
  exhibits built on ``pricing_at``.
* ``price`` — the lifted/linear engine above, returning a ``PricingResult``. Its
  *lifted* branch re-derives the per-line rows itself instead of delegating to
  ``pricing_at`` (duplication). Once ``pricing_at`` takes a ``linear`` /
  ``lifted`` argument, **delete ``price`` if it is truly redundant** (see
  Recommendations).
* ``spectral.Distortion.price`` (single distribution, no allocation) currently
  returns a bare namedtuple and defaults ``S_calculation='backwards'``. It should
  (a) return a ``PricingResult`` (the pentagon row) for consistency, and (b)
  default to **forwards** (offering backwards as an option), with its docstring
  updated. Genuine tension to preserve: the current docstring argues backwards is
  more reliable for *thin-tailed* risks (forward ``1−cumsum`` loses tail detail),
  so keep backwards available — just not the default.


********************************************
Phase 4 — Sample path (the "switcheroo")
********************************************

For working from an empirical multivariate **sample** rather than parametric
units:

* ``add_exa_sample(sample)`` aligns the sample total to the bucket grid
  (``round(total/bs)·bs``), rescales the per-line losses to preserve the total,
  groups by total to form ``exeqa_i = mean(Xᵢ | total)`` and ``p_total =
  Σ`` sample probabilities, then re-runs the ``add_exa`` derivations on that
  density.
* ``create_from_sample(name, sample_df, bs, log2)`` builds a Portfolio from the
  sample, ``update``-s it (independent FFT density), **archives** that as
  ``independent_density_df`` / ``independent_stats_df``, then **swaps in** the
  sample-based ``density_df`` (the *switcheroo*) and recomputes the empirical
  total moments. ``sample_compare`` / ``sample_density_compare`` diff the
  pre/post views.
* ``swap_density_df(new_df)`` (EXPERIMENTAL) lets you inject marginal densities
  directly into a shell Portfolio with the right line names, recompute the
  total + ``ft_nots``, and run ``add_exa`` — sidestepping all Aggregate
  construction. Duplicates the ``ft_nots`` loop from ``update``.

``make_comonotonic_allocations`` (Denuit et al. 2025) post-processes the kappas
into a comonotonic, convex-order improvement (numba-jitted ``_work``).


********************************************
"Journey of discovery" — distilled
********************************************

* **Forward S, post-FFT** (the T. S. Eliot "and the end of all our exploring…"
  comment in ``add_exa``): after years of trying clever backward survival
  cumulations, the conclusion is ``S = 1 - F``. The precision loss is immaterial
  *because you have already been through FFTs*.
* **The money calculation**: ``exeqa_i`` via one FFT pair, exploiting
  independence of ``Xᵢ`` and ``X_{-i}``.
* **``exeqa = 0`` where ``p_total < eps``** ("JUNE 25: this makes a difference!")
  — do not trust conditional expectations where there is no probability.
* **``S_calculation='forwards'`` for thick tails** ("July 2020, COVID-Trump
  madness") — recompute ``S = 1 - p_total.cumsum()`` in ``_build_augmented``.
* **Law-invariant layer ROE**: all lines share the same layer ROE; the L'Hôpital
  limit of ``(gS−S)/(1−gS)`` as ``S→1`` is ``ROE(1) = 1/g'(1) − 1``. This is what
  makes per-line ``Q`` allocation well-defined. (Note: the efficient branch's
  ``g'(1)`` fallback disagrees with this — see the Phase 3 assessment.)
* **Lifted → linear**: the move from the lifted natural allocation (distortion
  weights into the tail; unstable with a mass) to the linear natural allocation
  (objective-probability tail collapse; stable) is the single most important
  conceptual evolution in this file.
* **"The John Major problem"**: the last bucket has no information about what
  lies beyond the array, so reverse-cumulative tail fills must be set explicitly.


***************
Recommendations
***************

Keep
====

* The **kappa engine** (``exeqa_i`` via FFT) — elegant and correct.
* The **augmented_df cache** keyed by distortion, invalidated on ``update``.
* The **linear natural allocation** and its objective-probability tail collapse.
* Delegation of distortion calibration math to ``spectral.py`` subclasses.
* The independence-copula combine and ``ft_nots`` reuse (avoids double FFTs).

Fix / reconcile  *(decisions, 2026-05-28)*
==========================================

#. **Empirical-moment convention vs Aggregate — DECIDED: adopt Aggregate's.**
   Replace ``Portfolio.update``'s plain ``Σ p·xᵏ`` (the "PEG baseline" sum, no
   de-fuzz) with ``xsden_to_mwrangler`` on a **de-fuzzed copy** — exactly the
   Aggregate treatment. Coordinate the resulting baseline shift deliberately
   (harness). No-brainer.

#. **``_write_empirical_stats`` sev-block inversion — DECIDED: delete, same
   time.** It reconstructs each unit's ``(ex1,ex2,ex3)`` from ``(mean,cv,skew)``
   *because Aggregate used not to store empirical sev raw moments* — it does now
   (``stats_df['empirical'][('sev','ex1..3')]``). Read them directly. Fold into
   the change above.

#. **``pricing_at`` order — DECIDED: the pentagon.** Reorder to amounts-then-
   ratios ``L M P Q a | LR PQ ROE`` (and **include** ``a``). This row is the
   canonical ``PricingResult`` shape used everywhere (``pentagon.py``).

#. **``price`` vs ``pricing_at`` — DECIDED.** Give ``pricing_at`` a
   ``linear`` / ``lifted`` argument (default **linear**); if ``price`` is then
   truly duplicative, **delete it**.

#. **``S`` conventions — DECIDED: forwards default, offer backwards
   consistently.** ``add_exa`` / ``_build_augmented`` / ``add_exa_sample`` /
   ``spectral.Distortion.price`` should all default to **forwards** and accept a
   ``backwards`` option uniformly (backwards genuinely helps thin tails — keep it
   reachable, see Open). ``Distortion.price`` also returns a ``PricingResult`` and
   gets its docstring updated.

#. **``valid`` — DECIDED: all from ``stats_df``** (single source of truth),
   mean/aliasing included (same as Aggregate).

#. **No magic numbers — DECIDED.** Replace ``eps**3`` / ``10×`` / ``exeqa_err <
   1e-4`` / ``|ft| < 1e-10`` with named constants, shared as far as possible.
   **The ``loss_max`` length-bucketed ``mult ∈ {1,10,100}`` machinery is
   "utterly horrible" and must be rooted out thoughtfully** — replace with a
   principled rule (e.g. blank where ``F < k·eps``), not a frame-length hack.

#. **``_build_augmented`` duplication + ROE-fallback bug** (see Phase 3
   assessment): compute the total-level block once with the correct
   ``gprime1 = 1/g'(1) − 1``; let ``efficient`` gate only the extra columns.

#. **Pandas: adopt Copy-on-Write.** The repo pins ``pandas>=2.1`` but does **not**
   enable CoW (``pd.options.mode.copy_on_write``). CoW is the default in pandas
   3.0 and removes a class of chained-assignment ``FutureWarning``s (plausibly
   the bulk of the 77 warnings) while cutting defensive copies. Turn it on, fix
   the fallout (these wide ``df['x'] = …`` builders are mostly CoW-clean already),
   and we are future-proofed. (``applymap`` is already gone; ``.map`` is used.)

Optimise
========

* **``add_exa`` per-line loop** is the cost center (cumsums + an FFT pair per
  line). ``ft_nots`` already avoids the double FFT; the remaining per-line column
  algebra could be vectorised across lines.
* **Hoist the distortion-independent ``exp_loss``** out of the linear-allocation
  distortion loop (computed once, not per distortion).
* **Drop unneeded ``augmented_df`` columns — DECIDED: happy to.** With
  ``efficient=True`` already trimming, drop intermediate ``M.*`` columns the
  readouts never consume.

Remove (or quarantine to pedagogy)
==================================

* **``add_exa_details`` / ``add_eta_mu`` and the entire EPD + η-μ column family**
  — pedagogy-only. Move to ``pedagogy.py`` (or gate behind an explicit opt-in)
  so the core ``density_df`` is lean. This is a large simplification.
* **Journey-of-discovery comment blocks** and the many commented-out alternative
  calculations, once their pith (above) is captured here.
* Re-evaluate **``swap_density_df``** (marked EXPERIMENTAL, "is this actually
  worth doing?") — either promote it to a supported sample/empirical entry point
  or drop it.

Decide for v1.0  *(decided 2026-05-28)*
=======================================

* **Default allocation — DECIDED: ``linear``.** Lifted is **not** dropped — it
  stays available as an option — but ``linear`` becomes the default (flip the
  commented ``DeprecationWarning``).
* **Refuse lifted-NA on unbounded + mass — DECIDED.** When
  ``allocation='lifted'`` and the distortion has a mass and the support is
  unbounded, **raise** with a clear message pointing at ``linear`` (replacing the
  "ugly" mass/tail-support code with a clean refusal).
* **Boundedness — DECIDED: detect from spec, with a user override.** Carry a
  ``bounded`` flag. Auto-detect from the spec where provable (see the note
  below — it *is* tractable, contrary to the earlier "ill-posed" framing), and
  let the user **certify** ``bounded=True`` to override. For v1 a pure
  certification shortcut (default ``bounded=False``) is acceptable if detection
  slips, but detection is cheap enough to do.


**********************************************************
Defective / unbounded / mass — the hard cases (note)
**********************************************************

* **Defective** (``Σp<1``) is handled by *not normalising*, forward ``S``, and
  letting ``S`` plateau at the deficit. ``calibrate_distortion`` already trims
  ``S`` at the first zero and records ``ess_sup``.
* **Mass distortions** (CCoC etc.) put weight on the survival function's jump.
  On a **bounded** support this is fine. On an **unbounded** support under the
  *lifted* NA the mass lands on the last bucket → unstable. The *linear* NA
  avoids this by collapsing the tail with objective probabilities.
* **Detect boundedness from the spec, not the density.** Inferring it from a
  bucketed density is ill-posed — ``agg 100 claims dsev[10000]`` is unbounded yet
  has positive mass only on multiples of 10000, so "tail probabilities > 0" fails
  (structural zeros everywhere else). But the **spec** answers it cleanly:

  ``aggregate bounded ⟺ frequency bounded AND per-claim severity bounded``

  - **frequency bounded** is a ``freq_name`` lookup: ``fixed`` / ``bernoulli`` /
    ``binomial`` / empirical ``dfreq`` are bounded; ``poisson`` / ``negbin`` /
    ``geometric`` / ``logarithmic`` and all mixed-Poisson families are
    **unbounded** (``N`` can be arbitrarily large).
  - **severity bounded** if there is a finite ``exp_limit`` *or* finite
    ``sev_ub`` *or* the severity is discrete/finite-support (``dsev``) *or* an
    inherently bounded family (beta, uniform).

  This correctly classifies the ``dsev[10000]`` example as **unbounded** — not
  because of the severity (a single point mass, bounded) but because the
  ``poisson`` frequency is unbounded. Caveats: mixtures (bounded ∧ unbounded
  components), and the gap between *mathematical* unboundedness and *numerical*
  danger (a thin Poisson×bounded-sev tail is mathematically unbounded yet
  perfectly safe for lifted) — so a spec rule is **conservative** (it may flag
  safe cases). That is the reason for the user ``bounded=True`` override: detect
  the clear cases, let the user certify the rest.


***********************************************************
Proposal — before/after consistency harness (big DFs)
***********************************************************

The under-the-hood refactor must not silently change numbers in the large
dataframes (``density_df``, ``augmented_df``, ``stats_df``) or the headline risk
measures. Because the computation is **deterministic** (no RNG), we can capture a
golden baseline from the current ``REFACTOR`` HEAD and assert against it.

Design
======

#. **Corpus — small on purpose.** Because we are checking **max-digit
   precision** of a deterministic computation (not statistical coverage), a
   handful suffices: **~3–5 aggregates composing 2 Portfolios**. The aggregates
   should between them cover the code paths — e.g. a fixed-1 unit *and* the
   equivalent ``dfreq[1]`` (shortcut vs full FFT), a symmetric discrete
   (``dsev[1:6]``, fuzz-sensitive), a thick-tailed lognormal/Pareto, and one with
   occurrence + aggregate **reinsurance** (and a ``normalize=False`` defective
   variant). Compose them into **two Portfolios**: one parametric multi-line
   (PEG-like) and one **sample-built** (the switcheroo). Reuse ``test_suite.agg``
   lines where possible.

#. **Fixed grid + versions.** Pin ``(log2, bs)`` per case (do **not** rely on the
   recommender, which may itself be refactored). Record ``numpy`` / ``scipy``
   versions in the baseline manifest — FFT and ``interp1d`` can move by ULPs
   across releases, which matters at a 1e-12 tolerance.

#. **What to snapshot.** For each case, after ``update``:

   * ``stats_df`` and ``describe`` (full frames);
   * ``density_df.filter(regex='p_|exeqa_')`` — a good start, and we **definitely
     want some of the probabilities** in the snapshot (plus ``loss`` / ``S``);
   * then apply **two distortions — one with a mass (CCoC) and one without
     (e.g. PH/Wang)** — and snapshot the post-distortion columns: ``gS`` /
     ``gp_total`` and the ``exag_*`` (risk-adjusted premium) columns, under both
     ``lifted`` and ``linear``;
   * scalar readouts: ``q(p)`` / ``tvar(p)`` at a few ``p``; the ``price`` /
     ``analyze_distortions`` result frames.

#. **Storage.** Parquet/feather for the frames (preserves float64 exactly),
   plus a small JSON manifest (program text, grid, versions, column list, and a
   content hash). Exclude volatile fields (``last_update``, object ``repr``s).

#. **Comparison — initial goal: VERY VERY close.** A pytest that loads the
   baseline and, per (case, frame, column), asserts ``np.allclose`` at a **tight**
   tolerance and we see how far that holds. Start with one strict tier
   (``rtol≈1e-12``) for the pure deterministic FFT/cumsum quantities (densities,
   ``S``, ``exa``) and only *introduce* a looser tier (``rtol≈1e-8``) where a
   genuine ``interp1d`` / reinsurance step / distortion-root-solve forces it —
   rather than starting loose. On failure, report the **first** divergent
   ``(case, frame, column, index)`` with max abs/rel diff — a targeted signal,
   not a wall of numbers.

#. **Workflow.** ``--regenerate-baseline`` flag (or an env var) to recapture;
   otherwise read-only. Capture the golden **now**, before the refactor starts,
   tagged to the commit, so the refactor can proceed freely and any drift is
   caught immediately.

This is intentionally a *characterisation* test (lock in current behaviour),
distinct from the existing correctness tests (theoretical-vs-empirical
validation). Where the refactor *intends* to change a number (e.g. unifying the
empirical-moment convention), the baseline is updated deliberately, in the same
commit, with the change called out.
