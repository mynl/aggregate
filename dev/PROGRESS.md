# Refactor PROGRESS

> Status of the v1.0 **core-compute refactor** of `distributions.py`
> (Aggregate) and `portfolio.py` (Portfolio). Thin status layer — the
> detail lives in the per-step plans (now in `dev/done/`) and the
> release notes in `README.rst`. This file is the resume point if the
> conversation context is lost.
>
> **Last updated: 2026-06-04** — v1.0 core-compute refactor closed at
> meta.8 (2026-05-30); eleven post-refactor cycles landed since (a18–a28:
> reinsurance, negative-x / P&L, `pnl`, `multivariate`, notes/hints,
> discrete severity, distortion DecL, dsev bucketing). Current version
> **1.0.0a28**.
> **Ground truth for code state is `git log`, not this file.** Run
> `git log --oneline -30` when resuming to see what landed.

---

## Phase

**v1.0 core-compute refactor: complete.** All eight meta-steps landed
between 2026-05-29 and 2026-05-30. 701 pytest pass. Both baselines
(harness + PEG) regenerated where numbers moved deliberately; pinned
to the meta.6 capture commit otherwise. The four refactor plans
(`plan-meta.md`, `plan-aggregate-refactor.md`, `plan-portfolio-refactor.md`,
`plan-baseline-harness.md`) have moved to `dev/done/`.

What remains is in `dev/TODO.md` — pre-ship work and post-v1.0 ideas.
As that file shrinks, this one expands.

Immediately pre-refactor: **1.0.0a17** (numerical noise cleanup —
`agg_density` noise, tighter cv/skew noise detection and reporting,
better `MomentWrangler` use; the `xsden_to_meancv*` tail-mass
inconsistency was then resolved in meta.3 when both routed through
`xsden_to_mwrangler`).

---

## What landed (meta-by-meta)

| Step | Date | Headline |
|---|---|---|
| meta.0 pre-flight | 2026-05-29 | env synced; `agg 1.0.0a17`, numpy 2.4.5, scipy 1.17.1, pandas 3.0.3, py 3.14.3 |
| meta.1 harness baseline | 2026-05-29 | `e2d5390`; 10 cases, 65 parquets, manifest pinned at `cfcd6ae` |
| meta.2 pandas CoW | 2026-05-29 | conditional `copy_on_write` switch for pandas 2.x (3.0+ already on by default) |
| meta.3 shared stats hygiene | 2026-05-29 | all-float `stats_df`; `valid` SSoT; Portfolio empirical-moment convention adopts `xsden_to_mwrangler` (D4/D8); `e{e}.m{m}` component naming; baseline + PEG regenerated |
| meta.4 aggregate reins reporting | 2026-05-29 | `describe` becomes Subject / Net-or-Ceded-or-After / Change with denser `EX`/`CV`/`Sk` headings; staged `stats_df` rows populated; validate-subject |
| meta.5 aggregate cleanups + S unification | 2026-05-30 | `_fft_aggregate` helper; redundant `est_*` writes deleted; forwards `S` default in `Distortion.price`; `DefectiveDistributionWarning`; `aggregate_keys` + journey-of-discovery comments scrubbed |
| meta.6 portfolio pricing/allocation | 2026-05-30 | linear default + lifted refused on unbounded+mass; `allocation_method` member; `bounded` property; `pricing_at` pentagon order `L M P Q a \| LR PQ ROE`; `_build_augmented` de-dup + ROE-fallback fix (`1/g'(1) − 1`); `exp_loss` hoisted out of linear loop; both baselines regenerated |
| meta.7 portfolio cleanup | 2026-05-30 | `add_exa_details` slimmed to EPD + reimbursement diagnostics (eta-mu surface deleted); `swap_density_df` promoted to standalone function; journey comments scrubbed; `add_exa`'s `ft_nots` argument now required |
| meta.8 parser + perf | 2026-05-30 | `so`/`po` true synonyms — number sets meaning (`%` → share, bare → `amount/limit`); `_PercentNumber` marker; 4 new corpus cases `J.Re18a..d`; mixture-arm `gup_sevs` skipped when no exposure has positive attachment |

701 pytest pass at meta.8 close. Test count moved 693 → 701 from the
four new `J.Re18a..d` corpus lines (× parse + snapshot = 8 tests).

---

## Post-refactor cycles

Standalone cycles after the core-compute refactor. Each is its own `a*`
bump (one commit per iteration); finished plans live in `dev/done/`, and
the full release notes are the matching `README.rst` sections.

| Cycle | Ver | Date | Headline |
|---|---|---|---|
| reins-buckets | 1.0.0a18 | 2026-05-31 | `Aggregate.reins_bucket` switch (`'linear'` default / `'nearest'`); `_apply_reins_work` rebucket core rewritten as a vectorized `np.add.at` scatter (`_rebucket_to_grid`), replacing the old groupby→interp1d-CDF→diff scheme; `make_ceder_netter` hard-errors on out-of-order/overlapping layers (`_validate_reins_layers`); `Re.Both` baseline regenerated; `tests/test_reins_buckets.py`; DecL `ReBucket` |
| reins-reporting | 1.0.0a19 | 2026-06-01 | Rationalized reinsurance reporting (Aggregate + Portfolio). `reinsurance_df` → `reins_density_df` (consistent columns; `p_agg_gross_occ→p_agg_gross`, old `p_agg_gross→p_agg_subject`); new **per-layer** `reins_stats_df` (cols `(view, layer)`, `view`∈`occ|agg` = `Gross`/per-layer `layer.k`/`Ceded`/`Net`; **occ layers conditional** — freq `n·P(>attach)`, sev ÷ P(attach), agg = layer FFT, so layer agg means sum to `Ceded`; `Ceded`/`Net` unconditional with `Ceded`+`Net` sev = `Gross`; agg block leaves freq/sev NaN; meta rows `share/limit/attach/pr_attach/pr_detach/pr_loss/lol/output` — Gross = cc-weighted policy terms, occ Ceded = placed-sum limit/min attach, `lol`=loss-on-line, `output` 0/1 marks each stage's output and replaces a separate agg `Subject` column); `reins_describe` (per-stage **economic view**, same 8 cols as `describe`: `EX/CV/Sk` = the leading-view theoretic reference — `gross` for occ, `subject` for agg — held constant down each component; `Est *` = per-view model output; `Change = (Est−ref)/ref` reads as the lead row's validation/rebucketing error and the ceded/net rows' cession impact. Always **unconditional** — gross-row `Est` freq left NaN to mirror `describe`, ceded/net freq carry the unconditional mean `E[N]` only so `freq·sev=agg`; `view`/`component` index labels lowercase; fed by internal per-stage/view/basis EX-vs-Est `_reins_view_stats`); new Portfolio trio (end-to-end gcn via independent-FFT convolution; total block uses gross reference, `Change`=programme impact). Reins labels centralised as `REINS_LABEL_*`; `describe` reins view leads **Gross** (was Subject), output col **Net/Ceded/Output** (mixed; was After); occ-before-agg + gross/ceded/net ordering canonical. Removed `reinsurance_audit_df`/`reinsurance_report_df`/`reinsurance_occ_layer_df`, the `occ_reins_df`/`agg_reins_df` members, `_reins_audit_df_work`, and the `F_*` engine columns. `tests/test_reins_reporting.py` (33); DecL section Y; `Re.Both` describe baseline regenerated (Gross/Output relabel) |

746 pytest pass at reins-reporting close (701 → 746: reins-buckets +
`test_reins_reporting.py` (33) + the meta/label/per-layer/economic-view
punch-ups). The new per-layer `reins_stats_df` (a `Gross | layer.k | Ceded |
Net` layering frame, occ layers conditional) supersedes the removed
`reinsurance_audit_df` / `reinsurance_occ_layer_df`, so the
`docs/.../problems/*.rst` case studies left as migration notes (bahnemann ILF
table, ERA/other_misc layer exhibits) can be rebuilt against it directly (still
needs a docs build to verify — TODO.md item 16).

| reins-bivariate | 1.0.0a20 | 2026-06-02 | Joint (ceded, net) occurrence aggregate via 2D FFT. New `Aggregate.occ_bivariate(...)` → `BivariateDistribution` (new submodule `aggregate/bivariate.py`, submodule access only). Places gross severity mass at `(c(X), n(X))` on the line `c+n=X` to build a bivariate severity `S`; joint aggregate density `= iFFT2(freq_pgf(n, FFT2(S)))` — the univariate `_fft_aggregate` with 1D→2D transforms, valid because `freq_pgf` is elementwise (empirical-freq matmul PGF handled by ravel/reshape). Occurrence only (agg-cover bivariate is degenerate). Per-axis auto bucket/window sizing from the univariate `p_agg_ceded_occ`/`p_agg_net_occ` margins (`size_axis`), 2D scatter via the active `reins_bucket` scheme (`scatter_bivariate`, linear preserves both marginal means), `bs_*`/`log2_*` overrides; zero-risk/fixed-1 shortcuts mirror `_fft_aggregate`. `BivariateDistribution`: `.marginals()`, `.moments()` (mixed `E[C^i N^j]`), `.corr()` (positive — random count couples C,N), `.contour()`, reprs. Validation: marginals reproduce the univariate occ ceded/net aggregates (means exact via linear scatter; cv matches at matched grid `bs=self.bs` — auto-sizing is finer/more accurate), anti-diagonal `C+N` reproduces gross. `tests/test_reins_bivariate.py` (31); DecL section Z (`BV.*`) |

777 pytest pass at reins-bivariate close (746 → 777: `test_reins_bivariate.py`
(31)). A key finding: linear rebucketing adds `bs²·f(1-f)` to a severity's
second moment, so on a coarse model grid (bucket comparable to a small ceded
mean) the *univariate* ceded cv is inflated; the bivariate's auto-sized finer
ceded axis is the more accurate one. The rigorous "marginals == univariate"
identity is therefore asserted at the matched grid.

| Cycle | Ver | Date | Headline |
|---|---|---|---|
| negative-x Aggregate | 1.0.0a21 | 2026-06-02 | Signed (P&L) severity + movable output window. `ssev` (continuous, never clamps), auto-signing negative `dsev` atoms; `update(x_min=...)` window placed by a single roll on the padded FFT buffer, auto two-sided window from analytic moments via `estimate_agg_window`; three-method bucket/window sizing recorded in `Aggregate._bs_window_df` (exact_discrete > bounded_small > moment); severity reporting moved to `sev_density_df` on its own `xs_sev` grid; `value_type` member (`loss`/`payoff`, pricing-layer only). Default 0-based path byte-for-byte unchanged. `dev/done/plan-negative-x-agg.md`; `tests/test_negative_x.py` (28) |
| signed Portfolio combine | 1.0.0a22 | 2026-06-02 | Portfolio combine on signed support. Units driven on their **own** signed windows sharing only `bs`/`log2`/`padding` (FFT product is origin-at-0; truncating `ift` replaced by full `irfft` + roll so `p_total` conserves mass); coarsen-to-fit `Portfolio._bs_window` + `_bs_window_df`; signed F/S/VaR/TVaR/plot; gated behind `Portfolio._signed()` so non-signed books byte-identical. Pricing (`add_exa`) deferred to `dev/plan-portfolio-neg-x-pricing.md` — signed books warn + fall back to F/S-only. Accepted residual: a fine-lattice unit beside a wide one gets a coarse shared `bs` (deficit surfaced per-unit; proper fix = multi-resolution decimation, parked). DecL `c - dist` severity (`numbers MINUS sev1`). `dev/done/plan-negative-x-port.md`; `tests/test_negative_x_port.py` (14) |
| `pnl` keyword | 1.0.0a23 | 2026-06-03 | Premium-minus-loss as a first-class sibling of `agg`: `pnl NAME <prem> prem - <lr\|claims\|loss> …`. Built on the aggregate affine primitive (`_apply_agg_affine` grid relabel after `update_work`; analytic moments; no new numerics); premium vectorises like exposure (total only); tight mass-centred two-sided display window; `value_type='payoff'`. General fix: signed `describe` shows **SD not CV** (CV blows up as mean→0). `dev/done/plan-pnl-premium.md`; `tests/test_pnl.py` (15) |
| `multivariate` | 1.0.0a24 | 2026-06-03 | DecL-declared copula-coupled bivariate aggregates: two `agg`/`pnl` components' per-claim severities coupled by `aggregate.copula.Copula` (registry: normal/gumbel/clayton/fgm/independent, natural dependence parameters) via discrete Sklar rectangle mass, accumulated by a shared frequency over the `rfft2` backbone. `pnl` axes as per-axis affine after the FFT. `MultivariateAggregate` (new `aggregate/multivariate.py`) subsumes `bivariate.py`; `occ_bivariate` now returns a `netceded`-mode `MultivariateAggregate` with the full reporting surface. Deferred: `t` copula, ≥3-variate `rfftn`, `MultivariatePortfolio`. `dev/done/plan-multivariate.md`; `tests/test_multivariate.py` (37) |
| notes/hints | 1.0.0a25 | 2026-06-03 | `note{}` becomes pure text; build settings move to a dedicated `hints{key=value; …}` clause (allowed wherever `note` is). Caller kwargs always override hints (incl. `recommend_p`); unknown/duplicate/malformed hints warn, never crash; settings-looking notes get a one-time deprecation warning. Corpora migrated. `dev/done/plan-note-parse.md` |
| discrete severity `fz` | 1.0.0a26 | 2026-06-03 | Honest discrete severity: `SeverityDHistogram`/`SeverityFixed` back `fz` with `_DiscreteRV` (exact step cdf/sf/ppf/isf/support), replacing the `rv_histogram` epsilon-sliver hack. All discrete moments (unlimited, limited, layered) now exact finite sums; aggregate density bit-for-bit unchanged (half-bucket sampling never hits an atom); snapshots re-captured to exact values; `max_log2` unused. `dev/done/plan-discrete-severity-fz.md` |
| distortion DecL | 1.0.0a27 | 2026-06-04 | Distortion DecL is a flat number list: `distortion NAME kind n1 n2 …`. Parser's hand-maintained `_distortion_spec` table deleted; each `Distortion` subclass declares `decl_params` and `Distortion.decl_spec` does the mapping — adding a kind no longer touches the parser. `ccoc` takes return `r`; vector/combinator kinds error clearly in flat form; bracketed form removed. No plan file — README a27 + `tests/test_distortion_decl.py` |
| dsev bucketing | 1.0.0a28 | 2026-06-04 | New `dsev_bucket` setting (mirrors `reins_bucket`): `'linear'` **default** splits off-grid discrete atoms across bracketing buckets (mean-preserving); `'nearest'` is the historical snap. On-grid atoms unchanged; layered discrete severities still discretize via cdf-difference (behave as nearest). Two baseline cases re-captured at the float floor. `dev/done/plan-bucket-dhist.md` |

947 tests collected at a28 (777 at a20 → 947: negative-x 28 + portfolio 14 +
pnl 15 + multivariate 37 + the notes/hints, discrete-severity, distortion, and
dsev-bucket cycles' cases and corpus lines).

---

## Working files (all in `dev/`)

| File | Role |
|---|---|
| `PROGRESS.md` | This file — what landed |
| `TODO.md` | What's pending: pre-ship work (features in flight, windows/plotting, numerics deep dives, docs/packaging) and post-v1.0 ideas (was `TODO-Remember.md`) |
| `pipeline-aggregate.rst` | Aggregate current-state description (was the read-end input to the plans; keep as the algorithmic reference) |
| `pipeline-portfolio.rst` | Portfolio current-state description (same role) |
| `pipeline-reinsurance.rst` | Reinsurance reporting surface, object-by-object — rewritten to the live a19 end state (the 3 public objects + private `_reins_view_stats`, Portfolio trio); pre-refactor inventory dropped |
| `plan-portfolio-neg-x-pricing.md` | DRAFT — the deferred Portfolio pricing half of negative-x (`add_exa` column audit, `value_type` consumption) |
| `tentative-plan-decl-colorization.md` | Parked — IPython tracebacks don't call `_repr_html_` (see TODO.md) |
| `done/plan-meta.md` | The cross-module sequencing — every step done |
| `done/plan-aggregate-refactor.md` | Aggregate decisions (D1–D18) + work items |
| `done/plan-portfolio-refactor.md` | Portfolio decisions (D1–D17) + work items |
| `done/plan-baseline-harness.md` | Before/after harness + DecL corpus |
| `done/reins-buckets.md` | Reins rebucketing switch + layer validation (1.0.0a18) |
| `done/reins-reporting.md` | Rationalized reins reporting, Aggregate + Portfolio (1.0.0a19) |
| `done/reins-bivariate.md` | Joint (ceded, net) occurrence aggregate via 2D FFT (1.0.0a20) |
| `done/plan-negative-x-agg.md` | Signed severity + output window, Aggregate half (1.0.0a21) |
| `done/plan-negative-x-port.md` | Signed Portfolio combine (1.0.0a22) |
| `done/plan-pnl-premium.md` | `pnl` keyword — premium-minus-loss (1.0.0a23) |
| `done/plan-multivariate.md` | Copula-coupled `multivariate` keyword, Stage 1 (1.0.0a24) |
| `done/plan-note-parse.md` | `note{}` pure text / `hints{}` settings (1.0.0a25) |
| `done/plan-discrete-severity-fz.md` | `_DiscreteRV` honest discrete severity (1.0.0a26) |
| `done/spectral-quartet.md` | Distortion info/describe/stats_df/density_df quartet (pre-meta, 2026-05-26) |
| `done/plan-bucket-dhist.md` | `dsev_bucket` linear/nearest discretization (1.0.0a28) |
| `done/plan-tail-thickness.md` | Tail-thickness classifier (`tail.py`) — implementation in flight in the working tree, uncommitted |
| `done/plan-A-aggregate-style.md` etc. | Earlier completed plans (pre-meta) |

Convention reminder: when a plan is finished, move it to `dev/done/`.

---

## Key findings preserved from the read

These were the "before-you-touch-anything" insights from the initial pipeline
read. Recording them here so the *why* isn't lost as the plans archive.

- **Portfolio vs Aggregate empirical-moment divergence** (pre-meta.3):
  Portfolio used plain `Σ p·xᵏ`, Aggregate used `xsden_to_mwrangler` on a
  de-fuzzed copy. Unified in meta.3.
- **`_build_augmented` duplication + efficient-branch ROE-fallback bug**
  (pre-meta.6): the default (efficient) branch used `g'(1)` where the
  L'Hôpital limit is `1/g'(1) − 1`. Disagreed with the full branch on the
  right edge. Fixed in meta.6.
- **`aggregate_keys` class attribute** was dead — deleted in meta.5.
- **`add_exa_details`** was `plot_twelve`-only and `plot_twelve` doesn't
  actually consume its eta-mu output. Slimmed in meta.7.
- **Boundedness is decidable from the spec** (frequency ∈ {fixed, bernoulli,
  binomial, empirical} ∧ all severities bounded). Wired in meta.6 with a
  certify-override.
- **PIR case-study reproduction path:** `pip install aggregate==0.30.1` in
  an isolated env. PMIR is forward-looking and does **not** reproduce PIR
  exhibits — do not point users at it for that purpose.

---

## Open decisions

None on the v1.0 core compute. The negative-`xs` / windowed-FFT family
landed in a21–a24 (the Portfolio *pricing* half remains —
`plan-portfolio-neg-x-pricing.md`). New design questions live in
`TODO.md` (most notably `Portfolio.pricing_bounds` alignment to the new
513-point `s_grid`).
