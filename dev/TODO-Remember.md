# TODO / Remember

> Working list of pending work and things to think about, post v1.0 core-compute
> refactor. As items here move into the codebase, they migrate to PROGRESS.md
> (which expands as this shrinks).
>
> **Last updated: 2026-06-02** — after the `negative-x-agg` cycle
> (1.0.0a21). Current version 1.0.0a21.

---

## Deep dives (understand before changing)

1. **Portfolio update numerical mechanics** — the full `Portfolio.update` →
   `add_exa` split. Why this shape (per-line agg via `Aggregate.update_work`,
   then FFT recombine for `p_total`, then `add_exa` for the conditional
   expectations)? Trace every column written by `add_exa` and confirm the
   numerical convention is consistent end-to-end (forwards `S`, the
   `shift(-1, fill_value=...)` tail handling, the `loss_max` blanking
   heuristic with `mult ∈ {1, 10, 100}` — a hangover from the legacy code,
   probably wants a principled `F < k·eps` rule per portfolio plan D11).
   Goal: a docstring-grade understanding so the next bug or extension lands
   safely.

2. **Bounds numerical mechanics.** The `Bounds` class (IME 2022 methodology,
   `bounds.py`) — fixed 513-point binary `s_grid`, distortion clouds, weight
   frames. Today's `Bounds` works in isolation but `Portfolio.pricing_bounds`
   is `NotImplementedError` because aligning it to the new `s_grid` is
   non-obvious (see item below). Read end-to-end so the pricing-bounds rewrite
   can proceed from understanding, not pattern-matching.

---

## Parked (post-v1.0) — from `dev/done/plan-meta.md`

3. **Aggregate FI-1: negative `xs` / windowed FFT.** ✅ DONE (1.0.0a21).
   `dev/plan-negative-x-agg.md` implemented: signed severity (dsev auto-signs;
   continuous via `update(signed=True)`), output window (`update(x_min=...)`,
   auto via `x_min=None` + `estimate_agg_window`), `value_type` member, signed
   reporting. 789 tests pass; default path byte-for-byte. **Discovery:** plan §4
   wrongly assumed signed severity was free — the Severity layering clamps
   `x<0→0` and `validate_discrete_distribution` clamped negative dsev atoms;
   both fixed. **Still open:** the Portfolio half — `dev/plan-negative-x-port.md`
   (DRAFT, **now ready to refresh** from what we learned: combine + density_df
   column audit + pricing/`value_type` consumption). **Do this before the
   multivariate plan.** Deferred follow-ups: two-sided deficit split; `ft.py`
   recentering helpers → call the core path (+ equivalence test); re-home
   `estimate_agg_window` to `utilities.py` and unify with `bivariate.size_axis`;
   occ-reins on a signed severity grid; DecL keyword for `signed`/`value_type`.

4. **Aggregate FI-2: integrated aliasing + movable window.** ✅ DONE with FI-1
   (the output window `update(x_min=...)` / `x_min=None`, a movable
   possibly-negative window placed by a single roll on the padded FFT buffer).

5. **Portfolio FI-1: `Portfolio.pricing_bounds` rewrite.** Raises
   `NotImplementedError` as of 1.0.0a11. The old wiring assumed the dense
   `density_df.S` was the s-grid; new `Bounds` uses a fixed 513-point binary
   `s_grid` and the per-unit `exeqa_*` columns need to be interpolated onto
   it. Defer until the design is settled. **Author wants periodic reminders.**

6. **Portfolio FI-2: negative `xs` at the portfolio-combine level.** Easier
   half of #3 — combining already-computed unit aggregates is a deterministic
   sum, so a constant shift de-shifts cleanly. Doable without solving the
   within-unit random-frequency problem. **Now split into two plans:**
   - **Combine half (next):** `dev/plan-negative-x-port.md` (REFRESHED rev 1) —
     signed `loss`/`p_total`/`p_{line}`/`F`/`S` + signed VaR/TVaR. Key refresh
     insight: the FFT product is already origin-at-0 (no per-unit roll), but the
     current `ift(ft_all)` truncates away the wrapped negative mass — the
     Portfolio must mirror Aggregate's F2 step (full length-M `irfft` + one roll).
     Compose per-unit `_bs_window` (no back doors).
   - **Pricing half (deferred):** `dev/plan-portfolio-neg-x-pricing.md` (DRAFT) —
     **⚠ the `add_exa` column audit and especially the price/`exeqa` column**
     (`portfolio.py:2122`), distortion pricing, and `value_type` consumption.
     Discovery-driven; must be *checked*, not assumed. Signed books warn+fall back
     to F/S-only until this lands.

7. **Switcheroo harness case.** Add a `Port.Sample` case (hand-built or
   seeded sample) to the baseline once Portfolio sample work next surfaces,
   to keep the kappa-replacement path under regression.

---

## Docs / packaging

8. **New README** for the v1.0 launch — currently `README.rst` reads as a
   running release-notes draft for the refactor iterations. Rewrite for the
   stable v1.0 audience: what `aggregate` is, who it's for, install + the
   one-liner DecL example, pointers to docs.

9. **New CHANGELOG file** — extract the existing `README.rst` iteration notes
   into a proper `CHANGELOG.rst` (or `.md`) keyed by version, so the README
   can shed its release-notes role.

10. **Docs intro for v1.0.** A short orienting page at the top of the Sphinx
    docs that explains the v1.0 shift (linear default allocation, bounded
    detection, `allocation_method` member, forwards-`S` default, pentagon
    pricing columns, `DefectiveDistributionWarning`). Replace any legacy
    framing.

10b. **Reinsurance case-study docs need a per-layer rewrite (from a19).**
    The `reins-reporting` cycle removed the per-layer `reinsurance_audit_df` /
    `reinsurance_occ_layer_df`. Three `docs/2_user_guides/problems/*.rst` case
    studies that built per-layer / per-agg-limit tables against published
    references were left as **migration notes + per-layer-build recipes**, not
    working exhibits: `0x0_bahnemann.rst` (Table 6.4 ILF table),
    `0x0_enterprise_risk_analysis.rst` (per-unit ceded LR summary),
    `0x0_other_misc.rst` (Wang tower layer exhibit — note `make_table` below it
    already does the real work from `density_df.lev`).
    **Update (a19 punch-up):** `reins_stats_df` is now a per-layer layering
    frame (cols `Gross`/`layer.k`/`Ceded`, meta share/limit/attach rows), so
    these exhibits can be rebuilt against it **directly** — e.g.
    `a.reins_stats_df['occ']` gives the occurrence layering — rather than
    looping single-layer builds. Still needs a docs build to verify rendered
    numbers vs the published references. (`0x0_loss_models.rst` and
    `0x0_loss_data_analytics.rst` were converted faithfully and run.)

11. **Docs reference SLY-era grammar.** `docs/4_agg_language_reference/`
    describes the grammar in SLY's `@_` form. Should `include`
    `aggregate/decl.lark` directly, or call
    `aggregate.parser.grammar(add_to_doc=True)` which writes
    `docs/4_agg_language_reference/ref_include.rst`.

12. **Docstring style sweep.** `iman_conover.py` and `moments.py` use Sphinx
    `:param x:` style with many empty parameter slots; the rest of the
    codebase is NumPy style (per CLAUDE.md). Convert in one dedicated pass
    (~25 docstrings in moments + iman_conover, plus pockets elsewhere).
    Public surface first, private helpers second.

13. **`pedagogy.py` migrations.** Move the remaining figure generators in:
    `ft.py`'s `poisson_example`, `fft_wrapping_illustration`,
    `recentering_convolution`, `recentering_convolution_example`;
    `tweedie.py`'s `tweedie_illustration`. Goal: keep `ft.py` / `tweedie.py`
    focused on the API.

---

## Deferred designs

14. **DecL colorization** — `dev/tentative-plan-decl-colorization.md`. Design
    ready but parked (2026-05-27): IPython tracebacks don't call
    `_repr_html_`, so the payoff is mostly Sphinx-docs identity. Wait for a
    clearer use case.

15. **Plot severity when it is outside the aggregate output window.** With the
    negative-x output window (1.0.0a21), the severity (on `xs_sev`, near 0) and
    the aggregate (windowed, possibly far from 0, e.g. high claim count) no
    longer share a grid, so `sev_density_df` can fall entirely outside
    `[x_min, x_max]`. `info` now warns when this happens and the severity panel
    is sourced from `sev_density_df`; **open question:** how to *render* the
    severity in `plot` when its range doesn't overlap the aggregate axis — a
    separate inset/panel, a broken axis, or its own figure. Pended 2026-06-03.

16. **CONSIDER: support-aware window bounds.** Every severity exposes hard
    support endpoints via `fz.support()` (e.g. `ssev lognorm 100 cv .2 - 50`
    has lower bound −50; a `beta` has a finite upper bound). The window
    estimator uses the *upper* bound in `bounded_small` but not the *lower* one,
    and not at all for **finite-frequency** cases. For `fixed`/`dfreq`/
    `binomial`/`bernoulli` the aggregate support is *exactly*
    `[N·loc, N·ub]` (sign-aware) — a generalisation of `exact_discrete` to
    continuous bounded/shifted severities, giving an exact window instead of a
    MoM guess (e.g. `fixed 2 claims ssev lognorm − 50` has agg min exactly
    −100). For Poisson/unbounded frequency the severity floor is only a
    per-claim sanity bound (more claims → more extreme), so MoM stays right
    there. Add a `support_bounds` row / refinement to `_bs_window` keyed off
    `fz.support()`. Medium priority; pended 2026-06-03.

17. **DecL `constant - dist` (`numbers MINUS sev`).** Negative scale now works
    (`-1 * lognorm + 100` ⇒ `100 − lognorm`, via `sev_reflect`), but the
    natural notation `100 - lognorm` is still a parse error (the grammar only
    allows `sev1 MINUS numbers`, a *shift after* the dist). Add a rule
    `sev2: numbers MINUS sev1 -> sev_reflect_shift` so `100 - lognorm` desugars
    to the reflected form. Watch the unary-minus ambiguity that bit the old SLY
    parser; the `NUMBER` terminal already absorbs a leading minus, so the new
    rule must not shadow `sev1: numbers TIMES sev0`. Pended 2026-06-03.

---

## Done since this file opened

*(append as items land)*

- (2026-05-30) `xsden_to_meancv` / `xsden_to_meancvskew` tail-mass
  inconsistency — resolved as a side effect of meta.3 / D8: both now route
  through `xsden_to_mwrangler`, which places the tail mass at `xs[-1] + bs`
  identically. Was item #8 on the CLAUDE.md TODO list.
- (2026-05-31) **`reins-buckets` (1.0.0a18).** `Aggregate.reins_bucket`
  switch (`'linear'`/`'nearest'`) + vectorized `_rebucket_to_grid` scatter
  replacing the old groupby→interp1d-CDF→diff scheme; `_validate_reins_layers`
  hard-errors on out-of-order/overlapping layers. Plan in `dev/done/`.
- (2026-06-02) **`reins-reporting` (1.0.0a19).** Rationalized reinsurance
  reporting: `reins_density_df` (renamed, consistent columns); `reins_stats_df`
  — a **per-layer** layering frame (cols `(view, layer)`; occ layers
  conditional, meta share/limit/attach/pr_attach/pr_detach/pr_loss/lol/output
  rows); `reins_describe` — a per-stage **economic view** (8 `describe` cols;
  `EX/CV/Sk` = gross/subject reference held constant, `Est` = per-view output,
  `Change` = validation/impact; always unconditional; lowercase index), fed by
  the internal per-stage/view/basis EX-vs-Est `_reins_view_stats`; plus the
  Portfolio trio (end-to-end gcn). `REINS_LABEL_*` constants. Removed the
  per-layer audit/report/occ_layer objects and the `F_*` columns. Plan +
  post-plan addendum in `dev/done/reins-reporting.md`; surface documented in
  `dev/pipeline-reinsurance.rst`. Left a docs follow-up (item 10b) for the
  per-layer case studies.
- (2026-06-02) **`reins-bivariate` (1.0.0a20).** Joint (ceded, net) occurrence
  aggregate via 2D FFT: `Aggregate.occ_bivariate(...)` → `BivariateDistribution`
  (new `aggregate/bivariate.py`). Gross severity mass placed at `(c(X), n(X))`
  on `c+n=X`; joint density `= iFFT2(freq_pgf(n, FFT2(S)))` (empirical-freq PGF
  handled by ravel/reshape). Auto per-axis sizing (`size_axis`) + `reins_bucket`
  scatter (`scatter_bivariate`); `bs_*`/`log2_*` overrides. Marginals reproduce
  the univariate occ ceded/net aggregates, anti-diagonal reproduces gross.
  Occurrence only. Plan in `dev/done/reins-bivariate.md`;
  `tests/test_reins_bivariate.py` (31). **Open:** Portfolio `occ_bivariate`
  (2D convolution across independent units) is sketched but not implemented;
  offset/non-zero-floor windows deferred.
  **Successor design (2026-06-02): `dev/plan-multivariate.md`** — generalises
  `occ_bivariate` to a DecL-declared `multivariate` / `netceded` facility
  (`MultivariateAggregate`/`MultivariatePortfolio`); two joint-severity builders
  (outer-product for separate lines, comonotone scatter for ceded/net) over one
  ND-FFT backbone; depends on negative-x (signed axes). Iterate after negative-x.
