# Refactor PROGRESS

> Status of the v1.0 **core-compute refactor** of `distributions.py`
> (Aggregate) and `portfolio.py` (Portfolio). Thin status layer — the
> detail lives in the per-step plans (now in `dev/done/`) and the
> release notes in `README.rst`. This file is the resume point if the
> conversation context is lost.
>
> **Last updated: 2026-05-30** — v1.0 core-compute refactor closed at
> meta.8. **Ground truth for code state is `git log`, not this file.**
> Run `git log --oneline -30` when resuming to see what landed.

---

## Phase

**v1.0 core-compute refactor: complete.** All eight meta-steps landed
between 2026-05-29 and 2026-05-30. 701 pytest pass. Both baselines
(harness + PEG) regenerated where numbers moved deliberately; pinned
to the meta.6 capture commit otherwise. The four refactor plans
(`plan-meta.md`, `plan-aggregate-refactor.md`, `plan-portfolio-refactor.md`,
`plan-baseline-harness.md`) have moved to `dev/done/`.

What remains is in `dev/TODO-Remember.md` — parked enhancements,
docs/packaging follow-ups, and deep-dive intentions. As that file
shrinks, this one expands.

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

## Working files (all in `dev/`)

| File | Role |
|---|---|
| `PROGRESS.md` | This file — what landed |
| `TODO-Remember.md` | What's pending: parked enhancements, docs/packaging, deep dives |
| `pipeline-aggregate.rst` | Aggregate current-state description (was the read-end input to the plans; keep as the algorithmic reference) |
| `pipeline-portfolio.rst` | Portfolio current-state description (same role) |
| `tentative-plan-decl-colorization.md` | Parked — IPython tracebacks don't call `_repr_html_` (see TODO-Remember) |
| `done/plan-meta.md` | The cross-module sequencing — every step done |
| `done/plan-aggregate-refactor.md` | Aggregate decisions (D1–D18) + work items |
| `done/plan-portfolio-refactor.md` | Portfolio decisions (D1–D17) + work items |
| `done/plan-baseline-harness.md` | Before/after harness + DecL corpus |
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

None on the v1.0 core compute. New design questions live in
`TODO-Remember.md` (most notably `Portfolio.pricing_bounds` alignment to
the new 513-point `s_grid`, and the negative-`xs` / windowed-FFT family).
