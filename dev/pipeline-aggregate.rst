.. _pipeline aggregate:

####################################
The Aggregate Computation Pipeline
####################################

*Draft for the developer documentation. Describes the* ``Aggregate`` *pipeline
in* ``distributions.py`` *as of the* ``REFACTOR`` *branch, distils the
"journey of discovery" comments into their load-bearing conclusions, and ends
with recommendations for the under-the-hood pass. Line numbers are approximate
and will drift; method names are stable anchors.*

.. contents::
   :local:
   :depth: 2


********
Overview
********

An :class:`Aggregate` turns a frequency model and one or more severity models
into a discrete approximation of the compound distribution
:math:`A = X_1 + \cdots + X_N` using the FFT algorithm of Mildenhall (2024).
The object has two clearly separated phases:

#. **Construction** (``__init__``): parse/broadcast the exposure and severity
   specification into a list of :class:`Severity` objects, compute *theoretical*
   (closed-form) moments, and scaffold ``stats_df``. **No discretisation or FFT
   happens here.** Construction is cheap and grid-independent.

#. **Update** (``update`` → ``update_work``): choose a grid, discretise the
   severity, run the FFT convolution, apply reinsurance, and compute *empirical*
   moments from the resulting density. This is the expensive, grid-dependent
   phase and can be re-run at different ``(log2, bs)`` without rebuilding.

The single end-to-end data flow::

    build(DecL)                       parse to a spec dict
        │
        ▼
    Aggregate.__init__                broadcast → self.sevs[], theoretical moments,
        │                             stats_df[comp_*, mixed, independent]
        ▼
    update(log2, bs)                  pick grid (recommend_bucket → round_bucket)
        │
        ▼
    update_work(xs)
        │
        ├─ discretize()               severity CDF/SF → PMF per component (beds)
        ├─ Σ wᵢ · bedᵢ                 claim-count-weighted severity → sev_density
        ├─ picks()                    optional layer-loss-pick reweighting
        ├─ apply_occ_reins()          per-occurrence reinsurance on sev_density
        ├─ _freq_sev_convolution()    z=ft(sev); ftagg=PGF(z); agg=ift(ftagg)
        ├─ apply_agg_reins()          aggregate reinsurance on agg_density
        └─ empirical moments          de-fuzzed copy → stats_df[empirical, error]
        │
        ▼
    density_df  (lazy)                p_total, F, S, lev/exa, exlea, exgta, exeqa
        │
        ▼
    valid / q / tvar / price / plot   risk measures and pricing


***********************
Phase 1 — Construction
***********************

``__init__`` (≈ lines 2029–2442)
================================

The constructor is the most intricate part of the class (~400 lines). Its job
is to turn a flat spec — possibly a *limit profile* (a vector of layers) and/or
a *mixed severity* (a weighted list of distributions) — into:

* ``self.sevs`` — a flat ``np.array`` of :class:`Severity` objects, one per
  *component* (a component is one (exposure × severity) cell);
* ``self.en`` / ``self.attachment`` / ``self.limit`` — per-component vectors;
* ``self.n`` — total expected claim count;
* ``self.stats_df`` — the canonical moment frame (see below);
* headline scalars ``agg_m/cv/skew``, ``sev_m/cv/skew`` (theoretical).

Spec capture
------------

The very first thing ``__init__`` does is snapshot its own arguments via
``inspect.getargvalues(frame)`` into ``self._spec`` (with ``self``/``frame``/
``get_value`` popped). This is what ``spec`` / ``json`` / persistence round-trip
against, and why argument names *are* the serialisation format — renaming a
constructor parameter is a breaking change to saved specs.

Frequency by composition
------------------------

``self.frequency = Frequency(freq_name, ...)`` dispatches through
``Frequency.__new__`` to the right ``Frequency<Kind>`` subclass. An
``Aggregate`` *has-a* frequency; it is not one. The frequency object exposes
``freq_moms(n)`` (moments) and ``freq_pgf(n, z)`` (the PGF applied in the FFT).

The two broadcasting arms
-------------------------

There are two structurally different paths, chosen by
``if np.sum(sev_wt) == len(sev_wt)`` (i.e. *all weights are 1*):

* **Limit-profile arm** ("all weights 1"). Exposure terms and severity terms are
  ``np.broadcast_arrays``-ed *together* and zipped once. Each row is an
  independent component. Used for a vector of layers sharing one severity shape,
  or a single severity.

* **Mixture-product arm** ("weights ≠ 1"). Exposure terms and severity terms are
  broadcast *separately* and combined as an outer product. A single ground-up
  mixed severity is built first (``gup_sevs``) so that, for an excess layer, the
  mixture weights can be re-derived from each component's survival at the
  attachment (``w1 = wt * sf(attach); w /= w.sum()``). Then per-exposure-row the
  *actual* layered severities (``actual_sevs``) are built and recorded.

Both arms funnel each component through ``_record_component`` (writes the
``comp_<r>`` column of ``stats_df`` and accumulates into the shared
``MomentAggregator``), so the two arms cannot drift on *what* a component
records — only on *how* components are enumerated.

Exposure resolution. Per component, the "claim count trumps loss" rule applies:
if ``en > 0`` then ``el = en · E[X]``; elif ``el > 0`` then ``en = el / E[X]``;
premium/loss-ratio are back-solved similarly; an empirical-frequency sentinel
``en < 0`` resolves to ``Σ freq_a·freq_b``. A mixture component whose layer mean
underflows to ``nan`` is silently replaced by a degenerate zero severity
(``dhistogram`` at 0) so the FFT still has something well-formed to convolve.

``stats_df`` — the single source of truth
=========================================

Shape: ``MultiIndex(component, measure)`` rows × ``[comp_0 … comp_k, mixed,
independent, empirical, error]`` columns, ``dtype=object`` (so the
``('meta','name')`` string can sit alongside floats).

* **rows**: ``meta`` (name, limit, attachment, el, prem, lr, sevcv_param,
  mix_cv, wt) + ``freq`` / ``sev`` / ``agg`` × ``(ex1, ex2, ex3, mean, cv,
  skew)``.
* **comp_i**: this component's contribution.
* **mixed**: portfolio-of-one totals *with* frequency mixing
  (``remix=True``) — the headline theoretical moments.
* **independent**: totals *without* the shared mixing distribution
  (``remix=False``) — the freq-independent comparison.
* **empirical**: filled by ``update_work`` from the FFT output.
* **error**: noise-aware relative error of ``empirical`` vs ``mixed``.

Everything else (``describe``, ``valid``, the headline scalars) reads *from*
``stats_df``; it is the canonical store. The legacy ``audit_df`` / ``report_df``
/ ``statistics*`` surface has been removed.


***********************************
Phase 2 — Grid choice & discretise
***********************************

``update`` → ``recommend_bucket`` → ``round_bucket``
====================================================

``update(log2, bs=0)`` is the convenience wrapper. With ``bs=0`` it calls
``recommend_bucket(log2, p)`` then ``round_bucket`` and builds
``xs = arange(0, 2**log2) · bs``.

``recommend_bucket`` sizes the grid as
``max(limit/N, percentile_estimate/N)`` where the percentile is estimated
analytically from ``(agg_m, agg_cv, agg_skew)`` (no FFT yet). Thick tails need
``p`` pushed toward ``1 - 1e-8``; an infinite limit forces that automatically.

**The exact-binary ``bs`` invariant.** ``round_bucket`` is the codification of
the author's hard-won rule: *only use bucket sizes with an exact binary
representation.* For ``bs < 1`` it returns ``1 / 2**k`` (the largest power-of-two
fraction ≤ the raw estimate); for ``bs > 1`` it snaps to 1/2/5/10·10ⁿ. Exact
binary ``bs`` means ``loss = i·bs`` is representable, ``a/bs`` is an exact
integer, and ``snap``/index lookups are exact — eliminating a whole class of
"off-by-one-bucket" and "key not found" bugs. This is an *invariant the rest of
the code relies on*, not a cosmetic nicety.

``discretize`` (≈ 2629–2710)
============================

Converts each continuous/​discrete :class:`Severity` into a PMF on the bucket
grid. Two orthogonal knobs:

* ``sev_calc`` — bucket placement:

  - ``discrete`` / ``round`` (**default, recommended**): mass at :math:`b_i`
    represents :math:`\Pr(b_i - bs/2 < X \le b_i + bs/2)` — the buckets are the
    *exact* discrete support. ``adj_xs`` are the midpoints; the array is treated
    as fully discrete downstream.
  - ``forward`` / ``continuous``: :math:`\Pr(b_{i-1} < X \le b_i)`; no shift.
  - ``backward``: shifted the other way.
  - ``moment``: raises ``NotImplementedError`` ("Embrechts says it is not worth
    it").

* ``discretization_calc`` — how each bucket probability is computed:

  - ``survival`` (**default**): backward difference of the SF — most accurate in
    the right tail (the tail we care about).
  - ``distribution``: forward difference of the CDF — accurate in the left tail.
  - ``both``: the elementwise ``max`` of the two, getting the best of each
    (underflow makes the wrong one ~0).

Two structural choices encode the defective-distribution philosophy:

#. ``adj_xs[0] = -np.inf`` always: the first bucket absorbs all mass at and
   below zero (severity may have real support).
#. The right end uses ``xs[-1] + bs/2`` (one bucket past the top), **not**
   ``np.inf``. Extending to ``inf`` would guarantee probabilities summing to 1
   but dumps a visible mass into the last bucket; truncating one bucket past the
   top avoids the spurious mass at the cost of summing slightly short — which
   ``normalize`` then optionally rescales. *This is deliberate: a clean tail is
   worth a controlled, measurable deficit.* See "Defective distributions" below.


*****************************
Phase 3 — Severity assembly
*****************************

``update_work`` (≈ 2740–2885) assembles the claim-count-weighted severity:

``wts = freq_ex1 / Σ freq_ex1`` (per-component frequency weights, read straight
from ``stats_df``), then ``sev_density = Σ bedᵢ · wᵢ``. Optional ``picks``
reweighting follows. ``force_severity='yes'`` is the plot-only early exit (no
FFT).

Severity is the *only* thing that is optionally ``normalize``-d. The aggregate
is never normalised.


****************************
Phase 4 — The FFT core
****************************

``_freq_sev_convolution(padding)`` (≈ 2886–2941) is the whole algorithm in five
lines::

    z              = ft(self.sev_density, padding)        # severity chf
    self.ftagg_density = self.frequency.freq_pgf(self.n, z)   # apply PGF
    self.agg_density   = real(ift(self.ftagg_density, padding))  # back to PMF

with two shortcuts:

* **zero-risk** (``n == 0``): unit mass at 0; FFT run only to give
  ``ftagg_density`` the right shape (needed if the agg lives in a portfolio).
* **fixed-frequency-of-1** (``Σen == 1 and freq=='fixed'``): the aggregate *is*
  the severity — skip the inverse FFT entirely. (This is the
  "``agg A 1 claim … fixed``" path; the general ``dfreq[1]`` path does the full
  round trip and so carries sub-eps FFT fuzz the shortcut path does not.)

``padding`` doubles (1) or quadruples (2) the working vector to mitigate FFT
aliasing/wrap-around. ``ftagg_density`` is retained because Portfolio multiplies
unit FTs to combine them under the independence copula.

Reinsurance brackets the FFT
============================

* ``apply_occ_reins`` runs **before** the convolution, on ``sev_density``: it
  builds ceder/netter step functions (``make_ceder_netter``), re-grids the
  net/ceded severities onto the bucket grid via ``interp1d`` of the grouped CDF,
  and replaces ``sev_density`` with the requested (net/ceded) view. Gross is
  cached as ``sev_density_gross``.
* ``apply_agg_reins`` runs **after**, on ``agg_density``, identically, and then
  *re-FFTs* the result to keep ``ftagg_density`` consistent.

The FFT core itself is reinsurance-agnostic.


***************************************
Phase 5 — Empirical moments & density
***************************************

Empirical moments (end of ``update_work``)
==========================================

Both severity and aggregate raw moments are computed once through
``xsden_to_mwrangler`` and the ``(mean, cv, skew)`` derived from the *same*
:class:`MomentWrangler`, so ``ex123`` and ``mcvsk`` are mutually consistent and
written to ``stats_df['empirical']``. ``error`` = ``_noise_aware_rel_error``
vs ``mixed``.

**The de-fuzzed-copy subtlety** (recently fixed, worth preserving). The raw
inverse-FFT density carries sub-machine-eps fuzz in essentially every bucket.
In the plain mass sum this cancels (mass is conserved), but raw moments weight
each bucket by ``xᵏ``, so on a wide grid the far-tail fuzz is amplified by
``x³`` and corrupts the empirical *skew* — a symmetric die's skew drifts from
~1e-15 to ~1e-4 as ``log2`` grows. The fix computes moments from a throwaway
``agg_clean = where(|agg_density| < eps, 0, agg_density)`` while leaving
``self.agg_density`` raw (consistent with ``ftagg_density``). It is safe because
the FFT is exact up to rounding and the exact aggregate has no negative density
even under aliasing, so every stray value is genuinely small.

``density_df`` (lazy property, ≈ 1668–1759)
===========================================

Built on first access after ``update``; cached in ``_density_df``. Columns are
documented in the property docstring. Key construction facts:

* ``p_total`` = ``agg_density`` **after** ``remove_fuzz`` (zero ``|x|<eps``) —
  the *curated* view, distinct from the raw ``agg_density`` member.
* ``p_sev`` is added *after* remove_fuzz so the carefully-computed severity is
  not flushed away.
* ``S = 1 - p_total.cumsum()`` ("S is best computed forwards", 2021-01-28).
* ``lev/exa = S.shift(1).cumsum()·bs`` (left-Riemann LEV); ``exlea``, ``exgta``
  derived; ``e`` is the constant ``est_m`` column; ``exeqa = loss`` (since
  ``E[X|X=a]=a``).
* Several columns are *deliberate duplicates* (``p==p_total``, ``exa==lev``,
  ``exeqa==loss``) that exist purely so a unit can be regex-inlined into a
  Portfolio. This is a Portfolio concern leaking into Aggregate (see
  Recommendations).


*********************
Phase 6 — Validation
*********************

``valid`` (property, ≈ 2946–3048) returns a :class:`Validation` flag. It:

* short-circuits to ``REINSURANCE`` (cannot validate) or ``NOT_UPDATED``
  (``empirical`` agg-mean is NaN);
* flags ``SEV_MEAN`` / ``AGG_MEAN`` when ``|Err E[X]| > eps`` (``eps =
  validation_eps = 1e-4``);
* flags ``ALIASING`` when the agg-mean error is both non-trivial (``> eps³``)
  and ``> 10×`` the sev-mean error — the signature of FFT wrap-around / too-small
  ``bs``;
* flags CV / skew via ``np.isclose(est, theo, rtol=10·eps / 100·eps,
  atol=VALIDATION_NOISE)`` **only when** ``|theo| > VALIDATION_NOISE`` — a
  theoretically-zero skew/CV (symmetric or deterministic severity) is skipped,
  because the FFT's estimate of a zero higher moment is grid-dependent noise
  with no meaningful relative error.

Philosophy: a pass means "not unreasonable" (fail-to-reject the null), not
"correct". Type-1 error (rejecting a good model) is preferred to Type-2.


********************************************
"Journey of discovery" — distilled
********************************************

The narrative comments in the source encode real conclusions. Distilled, so the
comments themselves can be deleted:

* **Forward S.** Post-FFT, ``S = 1 - F`` (forwards) is correct and simplest; the
  precision argument for backward cumulation does not apply once you have been
  through an FFT.
* **No mass at the top bucket.** Discretise to ``xs[-1]+bs`` and normalise,
  rather than extending to ``inf`` — a clean tail beats a guaranteed-unit sum.
* **Protect the severity.** ``p_sev`` is attached *after* ``remove_fuzz`` so
  exact severity work is not zeroed.
* **Fixed-1 shortcut.** Frequency ≡ 1 ⇒ aggregate ≡ severity; skip the inverse
  FFT for speed and accuracy.
* **Moment matching not implemented** on purpose ("Embrechts says it is not
  worth it").


***************
Recommendations
***************

Keep (working well; do not disturb lightly)
===========================================

* The **construction / update split** and the grid-independent constructor.
* **``stats_df`` as single source of truth** with the
  ``mixed``/``independent``/``empirical``/``error`` column model.
* The **5-line FFT core** and its two shortcuts.
* The **exact-binary ``bs`` invariant** in ``round_bucket``.
* The **de-fuzzed-copy** empirical-moment computation (leave ``agg_density``
  raw).
* **Not normalising the aggregate** and the deliberate defective-friendly
  discretisation.

Fix / reconcile
===============

#. **Redundant ``est_*`` writes in the reinsurance methods.**
   ``apply_occ_reins`` sets ``est_sev_*`` and ``apply_agg_reins`` sets
   ``est_*`` via ``xsden_to_meancv(skew)`` — but ``update_work`` *unconditionally
   recomputes both afterwards* through ``xsden_to_mwrangler`` (and, for the agg,
   the de-fuzzed copy). In the normal update path the reinsurance writes are dead
   and use a *different, non-de-fuzzed* helper. Drop them, or have ``update_work``
   skip recomputation when reinsurance already did it. Pick one path.

#. **Empirical-moment convention mismatch with Portfolio.** Aggregate uses
   ``xsden_to_mwrangler`` (places tail-mass deficit at ``xs[-1]+bs``);
   ``Portfolio.update`` uses a *plain* ``Σ p·xᵏ`` "to match the legacy PEG
   baseline". The two should agree on one convention. This intersects the
   existing ``moments.py`` tail-mass TODO and matters for the before/after
   harness.

#. **Mixed-source ``valid``.** Mean/aliasing read from ``describe`` (which has
   already been snapped and noise-adjusted) while CV/skew read from raw
   ``stats_df``. Read everything from ``stats_df`` for one consistent source;
   ``describe`` is then purely a display concern.

#. **Magic numbers in ``valid``.** The ALIASING test's ``eps**3`` floor and
   ``10×`` ratio are heuristics chosen against specific failing docs examples.
   Re-express in terms of ``VALIDATION_NOISE`` and a named ratio constant.

#. **``reinsurance_df`` re-implements the FFT** (including the fixed-1 shortcut)
   to recompute gross/ceded/net aggregates. Factor the convolution into a small
   private helper and call it from both places.

Optimise
========

* **``stats_df`` dtype=object** forces ``.astype(float)`` casts on every read and
  repeated ``[c for c in columns if c.startswith('comp_')]`` scans. Consider a
  numeric float frame plus a tiny separate meta store, or cache the comp-column
  list.
* **Double Severity construction** in the mixture arm (``gup_sevs`` then
  ``actual_sevs`` per exposure row) is ``O(layers × components)``. Fine for small
  profiles; revisit only if large limit profiles become a use case.

Remove
======

* **``aggregate_keys``** (class attribute, ≈ 1639) — defined, never referenced
  anywhere in ``src``. Dead.
* The **journey-of-discovery comment blocks** and large commented-out
  alternatives, once their pith (above) is in the docs.
* **``rescale``'s ``safe_scale`` / ``get_value``** defensive wrapping exists only
  because the parser hands back Python ``list``s instead of ``np.ndarray``
  (there is a standing ``TODO have parser return numpy arrays not lists``).
  Fixing the parser output type removes the defensive code in several places.

Open questions / defer
======================

* **LEV tail-mass convention.** ``density_df.lev`` (left-Riemann
  ``S.shift(1).cumsum()·bs``) and the empirical-moment tail-mass placement
  should be checked for mutual consistency — same family as the ``moments.py``
  ``xsden_to_meancv`` vs ``xsden_to_meancvskew`` TODO.


********************************
Defective distributions (note)
********************************

A *defective* distribution has ``Σ p < 1``. In this library that is a feature,
not an error, and the pipeline is built to leave it alone:

* Severity may be normalised; the **aggregate is never normalised**.
* ``S`` is computed forwards (``1 - cumsum``), so for a defective aggregate
  ``S`` plateaus at the deficit instead of being forced to 0 — exactly what
  pricing needs (forward vs backward survival-function calculations agree).
* The empirical-moment helper places the deficit mass at the implied maximum
  ``xs[-1]+bs`` and logs at INFO when the deficit exceeds ``VALIDATION_NOISE``
  (so genuine defectiveness is distinguished from fp-dust-off-1).

The open detection problem — *is a bucketed distribution genuinely unbounded?* —
is **not** an Aggregate concern in isolation, but it surfaces in Portfolio
pricing (distortion mass on an unbounded tail). It is discussed in the Portfolio
document; the likely answer is "inspect the spec, not the density".
