
# Plan — meta sequencing (v1.0 core-compute refactor)

> **Status: ready to execute (2026-05-29).** All design decisions in the
> per-module plans are closed. This document is the single execution thread:
> each `meta.N` is one coherent chunk of work that points into the per-module
> plans for detail. **Resume point**: run the next `meta.N` not yet done.
>
> **Companion docs (do not duplicate here):**
> - `plan-aggregate-refactor.md` — D1–D18, Groups 1–4 (aggregate detail)
> - `plan-portfolio-refactor.md` — D1–D17, Groups 1–4 (portfolio detail)
> - `plan-baseline-harness.md` — DecL corpus, snapshot format, failure protocol
> - `pipeline-aggregate.rst` / `pipeline-portfolio.rst` — current-state docs
>
> **Working rule.** Each `meta.N` commits whatever it produces (code +, where
> applicable, a regenerated baseline). The harness runs after every step;
> failures stop the step and report (per `plan-baseline-harness.md` §6).

---

## meta.0 — Pre-flight (one-time, before meta.1)

* Worktree on `REFACTOR`, environment synced (`uv sync --extra dev`).
* `UV_LINK_MODE=copy` is already set in `.claude/settings.local.json`; outside
  the harness, run `$env:UV_LINK_MODE = "copy"` in PowerShell.
* `git log --oneline -20` to see what has already landed on the author's side.

No code change.

---

## meta.1 — Capture the harness baseline

**Goal.** Lock in the current numbers from `REFACTOR` HEAD so every subsequent
step has a deterministic safety net.

**Plan ref.** `plan-baseline-harness.md` end-to-end. Corpus is final
(6 aggregates + 3 portfolios; no switcheroo; no calibration snapshot; failure
protocol = run all, report all).

**Output.** `tests/baseline/{corpus.py, capture.py}`, `tests/baseline/data/`
parquets + `manifest.json` (records `aggregate`/`numpy`/`scipy`/`pandas`
versions and the capture commit SHA), and `tests/test_baseline.py`.

**Done when.** Capture completes; comparison harness passes against the just-
captured snapshot (the trivial pass), committed.

---

## meta.2 — Pandas Copy-on-Write (library-wide, one switch)

**Goal.** Flip CoW on globally; fix the fallout once.

**Plan refs.** Aggregate **D18**, Portfolio **D13**.

**The switch.** Pandas 3.0 has CoW on as an unchangeable default — and the
option `pd.options.mode.copy_on_write` is a deprecated no-op there
(emits `Pandas4Warning`, removed in pandas 4.0). The library minimum is
pandas 2.1, so we still need to opt in on the 2.x line. Conditional:

```python
import pandas as _pd
if int(_pd.__version__.split(".", 1)[0]) < 3:
    _pd.options.mode.copy_on_write = True
del _pd
```

Goes at the top of `src/aggregate/__init__.py`, before any submodule import.

**The fallout.** With pandas 3.0 already in use during meta.1 capture, the
fallout has already been quietly absorbed by recent commits (e.g.
`401c0c5 Distortion.price ... CoW pandas issue in portfolio`). Full pytest
sweep after the switch: **693 passed, no CoW-related warnings**.

**Done when.** Harness passes (no number should move from CoW alone); test
suite clean of CoW-related warnings.

---

## meta.3 — Shared stats / validation hygiene

**Goal.** Aggregate and Portfolio get the same all-float `stats_df`, the same
SSoT `valid`, the same named constants, and the same `xsden_to_mwrangler`
moment convention. Done in lockstep because the two modules read each other's
`stats_df` shape.

**Plan refs.**
* Aggregate **Group 2** (2.1–2.4) and **1.2 schema + 3.8 component renaming**.
  Specifically: drop `('meta','name')` (D3); keep `comp_*` but rename to
  `e{e}.m{m}` (D16); add staged columns scaffold (D5, NaN-filled — fully
  populated in meta.4); `valid` reads from `stats_df` only (D6); magic numbers
  → `VALIDATION_NOISE`-based constants (D7); Portfolio moments adopt
  `xsden_to_mwrangler` on a de-fuzzed copy (D8).
* Portfolio **Group 2** (2.1–2.4) — empirical convention (D4), delete sev-
  block inversion (D5), `valid` SSoT (D10), magic numbers + drop the
  `mult ∈ {1,10,100}` length-bucketed machinery (D11).

**Numbers move here** (Portfolio empirical-moment convention). Regenerate the
baseline in the same commit with a clear message: "meta.3: Portfolio adopts
Aggregate moment convention — baseline regenerated."

**Done when.** Harness passes against the regenerated baseline.

---

## meta.4 — Aggregate Group 1: reinsurance reporting

**Goal.** The visible reins-reporting payoff. `describe` columns become
Subject / Net-or-Ceded-or-After / Change with denser `EX`/`CV`/`Sk` headings.
Subject is validated under the hood. Staged `stats_df` rows populated.

**Plan refs.** Aggregate **Group 1**: 1.1 (describe relabel + denser headings),
1.2 (staged empirical writes — schema scaffold from meta.3 is now filled in),
1.3 (validate-subject; reuses meta.5 FFT helper if done together, otherwise
inlines), 1.4 (`Subject` display name).

**Numbers do not move** on a no-reinsurance case (the new `Change` column
arithmetically equals the old `Err` column when after-reins == subject). They
**do** become non-trivial on the `Re.Both` reinsurance harness case — these
are *new* columns, not changed numbers, so the baseline gains snapshot
entries rather than diverging. Regenerate the baseline to absorb the new
columns.

**Done when.** Harness passes against the regenerated baseline; doctests
matching `E[X]` / `CV(X)` headings updated to the new `EX`/`CV` forms.

---

## meta.5 — Local aggregate cleanups + S unification

**Goal.** The aggregate-only items not coupled to meta.4: the FFT helper, the
journey-comment / `aggregate_keys` deletions, the redundant `est_*` writes in
the reins methods, and the cross-site S-convention flip.

**Plan refs.** Aggregate **3.2** (FFT helper extracted from
`reinsurance_df`/`_freq_sev_convolution`), **3.3** (redundant `est_*` writes
deleted, D13), **3.4** (forwards S default everywhere, D17 — `Distortion.price`
flips backwards→forwards, `DefectiveDistributionWarning` emitted on
`Σp < 1 − VALIDATION_NOISE`), **3.6** (delete `aggregate_keys`, D9), **3.7**
(delete journey-of-discovery comments, D10).

**Numbers move on the `Def.Pareto` case** for distortion-related columns
(`Distortion.price` flips its default); the deficit warning surfaces. Confirm
the magnitude of the move equals the deficit and regenerate the baseline
deliberately.

**Done when.** Harness passes against the regenerated baseline; the warning
fires once per defective case at user-call-site (`stacklevel=2`).

---

## meta.6 — Portfolio Group 1: pricing & allocation through-line

**Goal.** Pentagon `pricing_at`, `_build_augmented` de-dup + ROE-fallback fix,
`allocation_method` member, linear default, refuse lifted on
unbounded+mass, hoist `exp_loss`, drop unused columns.

**Plan refs.** Portfolio **Group 1**: 1.1 (linear default, D1), 1.2 (refuse
lifted on unbounded+mass, D2), 1.3 (boundedness from spec + certify override,
D3), 1.4 (pentagon order `L M P Q a | LR PQ ROE`, D6), 1.4b
(`allocation_method` member with cache invalidation, D7), 1.5
(`_build_augmented` de-dup + correct `gprime1 = 1/g'(1) − 1`, D12), 1.6
(`Distortion.price` → `PricingResult`, D9 — this folds into meta.5's flip if
both modules are touched together), 1.7 (drop unneeded `augmented_df` cols +
hoist `exp_loss`, D14/D16).

**Numbers move on the mass-distortion (CCoC) cases in the tail** because of
the ROE-fallback fix (efficient branch was using `g'(1)` where the L'Hôpital
limit is `1/g'(1) − 1`). Verify the direction of the move against a known
mass distortion before regenerating; document the change in the commit.

**Done when.** Harness passes against the regenerated baseline. Lifted on
`Port.CNC` with CCoC raises (per D2); lifted on `Port.Bodoff` with CCoC works
(bounded ⇒ stable).

---

## meta.7 — Portfolio + cross-module cleanup

**Goal.** Remaining cleanups: `add_exa_details` slim → pedagogy,
`swap_density_df` → standalone function, journey-comment deletion.

**Plan refs.** Portfolio **Group 3**: 3.2 (`add_exa_details` audit + slim to
exactly what `pedagogy.plot_twelve` consumes, D15), 3.3 (`swap_density_df` →
standalone function recomputing stats via `xsden_…`, D17), 3.5 (delete
journey-of-discovery comments).

**Numbers should not move** (these are pure-structure changes); harness as
regression check.

**Done when.** Harness passes; `pedagogy.plot_twelve` still produces its
figure unchanged.

---

## meta.8 — Independents: parser (so/po) and aggregate perf

**Goal.** Items that do not block the through-line.

**Plan refs.**
* Aggregate **Group 4** — `so`/`po` parser disambiguation; add corpus cases
  for both forms of both keywords (4.1).
* Aggregate **3.5** — double-Severity-construction perf in the mixture arm
  (low priority).

**Done when.** Parser tests pass on the new corpus cases; perf item closed or
deferred with rationale.

---

## Parked for after v1.0

* **Aggregate FI-1** — negative `xs` / profit-loss + windowed FFT.
* **Aggregate FI-2** — integrated aliasing / movable window of interest.
* **Portfolio FI-1** — `Portfolio.pricing_bounds` rewrite (currently
  `NotImplementedError`). *Author wants periodic reminders.*
* **Portfolio FI-2** — negative `xs` at the portfolio-combine level.
* **Switcheroo harness case** — add to baseline when Portfolio sample work
  next surfaces.

---

## Status

| Step | Status | Commit / note |
|------|--------|---------------|
| meta.0 pre-flight | **done** | 2026-05-29; agg 1.0.0a17, numpy 2.4.5, scipy 1.17.1, pandas 3.0.3, py 3.14.3 |
| meta.1 harness baseline | **done** | `e2d5390`; 10 cases, 65 parquets, manifest pinned at `cfcd6ae`. Pandas 3.0 already has CoW on by default — meta.2 mostly explicit-switch + sanity. |
| meta.2 CoW switch | **done** | conditional switch in `__init__.py` (no-op on pandas 3.0+ which already has CoW on); 693 pytest pass, no CoW warnings |
| meta.3 shared stats hygiene | **done** | 2026-05-29; D3+D5+D7+D16 (agg) + D4+D5+D10+D11-named-constants (port); baseline + PEG regenerated; 693 pytest pass. The Portfolio `mult ∈ {1,10,100}` structural cleanup (D11 second half) deferred to a later step — it's a number-mover that wants its own commit. |
| meta.4 aggregate reins reporting | **done** | 2026-05-29; describe relabel (Subject / Net-Ceded-After / Change, denser EX/CV/Sk); 1.2 staged stats_df writes (after_occ + occ_impact/agg_impact + gross_empirical populated); 1.3 validate-subject (`stats_df['error']` is gross_empirical vs mixed; `valid` ORs REINSURANCE + subject flags; `explain_validation` surfaces "reinsurance; subject ..."); baseline regenerated (Re.Both case carries the new column layout); 693 pytest pass. |
| meta.5 aggregate cleanups + S unification | ready | baseline moves (`Def.Pareto` under D17) |
| meta.6 portfolio pricing/allocation | ready | baseline moves (ROE-fallback fix) |
| meta.7 portfolio cleanup | ready | no number movement expected |
| meta.8 parser + perf | ready | independent |

Update the table as each step lands; the per-module plans are the detail.
