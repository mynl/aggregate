# Refactor PROGRESS

> Living status for the v1.0 **core-compute refactor** of `distributions.py`
> (Aggregate) and `portfolio.py` (Portfolio). Thin status layer — the detail
> lives in the plan docs; this file is the resume point if the conversation is
> lost. **Update after each completed phase.**
>
> **Last updated: 2026-05-29.**
> **Ground truth for code state is `git log`, not this file** — the author
> commits frequently in their own terminal (379 commits on `REFACTOR`, 53 since
> 2026-05-17). When resuming, run `git log --oneline -20` to see what actually
> landed.

---

## Phase

**Planning: complete. All decisions closed. Execution: ready to start at meta.1
(harness baseline).**

The full read + design discussion is done and written down. The cross-module
sequencing for execution lives in `plan-meta.md` — use that as the resume point.
A few plan items have already been started in the author's own commits (e.g.
`Distortion.price` forwards/backwards comment + a pandas-CoW fix in portfolio —
commit `401c0c5`). The bulk of the coding is not yet done.

---

## The working set (all in `dev/`)

| File | Role |
|---|---|
| `pipeline-aggregate.rst` | Aggregate current-state description + recommendations (→ docs) |
| `pipeline-portfolio.rst` | Portfolio current-state description + recommendations (→ docs) |
| `plan-aggregate-refactor.md` | Aggregate decisions (D1–D18; all closed) + grouped work items + sequencing + Future Ideas |
| `plan-portfolio-refactor.md` | Portfolio decisions (D1–D17; all closed) + grouped work items + sequencing + Future Ideas |
| `plan-baseline-harness.md` | Before/after harness + concrete DecL corpus + failure protocol (§9 all closed) |
| `plan-meta.md` | Cross-module sequencing — the **resume point** for execution |
| `PROGRESS.md` | This file |

Completed-plan convention: move a plan to `dev/done/` when its work is finished.

---

## Done

- Read both pipelines end-to-end; wrote the two `pipeline-*.rst` descriptions
  (with the journey-of-discovery comments distilled).
- Captured every decision in the two refactor plans + the harness plan; the
  author reviewed and refined all three.
- Key findings recorded: Portfolio vs Aggregate empirical-moment divergence;
  `_build_augmented` duplication + the efficient-branch ROE-fallback bug
  (`g'(1)` vs the correct `1/g'(1) − 1`); `aggregate_keys` dead;
  `add_exa_details` is `plot_twelve`-only; boundedness IS decidable from the spec.

---

## Next steps

See **`dev/plan-meta.md`** for the full cross-module sequencing (meta.1 …
meta.7). The plan-meta doc references the per-module plans by step ID, so the
resume point is always "run the next meta.* that isn't done."

Headline order: **harness → CoW → shared stats hygiene → aggregate reins
reporting → portfolio pricing/allocation → cleanup → parser (so/po) + perf.**

---

## Open decisions

None. All design decisions closed as of 2026-05-29:

- **Aggregate O5** → **D17** (decided): forwards default everywhere; emit
  `DefectiveDistributionWarning` when `Σp<1`; both options offered uniformly.
- **Harness §3 / §9** → all closed (grids reviewed in `hacks/harness.ipynb`;
  `Def.Pareto` α=1.5 form chosen; no switcheroo or calibration snapshot in v1).

## Future ideas (parked)

- Negative `xs` (profit/loss + convolve) — aggregate `plan §6 FI-1`; easy at the
  portfolio-combine level, hard within a unit (random `N`). Author wants periodic
  nudges.
- Integrated aliasing / movable window of interest — aggregate `plan §6 FI-2`.
- `Portfolio.pricing_bounds` rewrite (currently `NotImplementedError`) — portfolio
  `plan §6 FI-1`. **Author asked to be reminded about this one.**
- **Switcheroo harness case** — parked at harness §9.3 (2026-05-29). Add a
  `Port.Sample` case (hand-built or seeded sample) to the baseline once
  Portfolio sample work next surfaces. Remind the author at that time.
