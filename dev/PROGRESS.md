# Refactor PROGRESS

> Living status for the v1.0 **core-compute refactor** of `distributions.py`
> (Aggregate) and `portfolio.py` (Portfolio). Thin status layer — the detail
> lives in the plan docs; this file is the resume point if the conversation is
> lost. **Update after each completed phase.**
>
> **Last updated: 2026-05-28.**
> **Ground truth for code state is `git log`, not this file** — the author
> commits frequently in their own terminal (379 commits on `REFACTOR`, 53 since
> 2026-05-17). When resuming, run `git log --oneline -20` to see what actually
> landed.

---

## Phase

**Planning: complete. Execution: just beginning (author-driven).**

The full read + design discussion is done and written down. A few plan items
have already been started in the author's own commits (e.g.
`Distortion.price` forwards/backwards comment + a pandas-CoW fix in portfolio —
commit `401c0c5`). The bulk of the coding is not yet done.

---

## The working set (all in `dev/`)

| File | Role |
|---|---|
| `pipeline-aggregate.rst` | Aggregate current-state description + recommendations (→ docs) |
| `pipeline-portfolio.rst` | Portfolio current-state description + recommendations (→ docs) |
| `plan-aggregate-refactor.md` | Aggregate decisions (D1–D15; **O5 open**) + grouped work items + sequencing + Future Ideas |
| `plan-portfolio-refactor.md` | Portfolio decisions (D1–D17) + grouped work items + sequencing + Future Ideas |
| `plan-baseline-harness.md` | Before/after harness + **the concrete DecL corpus** (§3 grids to confirm; §9 open Qs) |
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

## Next steps (combined sequencing from the two plans)

1. **Harness first** — execute `plan-baseline-harness.md`: confirm the §3 DecL
   corpus + grids, then capture the deterministic golden baseline from current
   `REFACTOR` HEAD (covers both modules). Nothing else should start until this
   exists.
2. **Library-wide, do once:** adopt **pandas Copy-on-Write** (aggregate D-/
   portfolio D13) — coordinate across both modules; the author has already begun.
3. **Stats / validation hygiene (both modules together):** all-float `stats_df`
   (drop the `name` row), `valid` reads only from `stats_df`, name the magic
   numbers, and adopt the de-fuzzed `xsden_to_mwrangler` moment convention in
   Portfolio (this **moves the baseline** — regenerate deliberately).
4. **Aggregate reins reporting** (plan-aggregate Group 1): `describe`
   Subject/Net-or-Ceded-or-After/Change; staged `stats_df`; validate-subject.
5. **Portfolio pricing/allocation** (plan-portfolio Group 1): `_build_augmented`
   de-dup + ROE fix (verify against a mass distortion under the harness);
   pentagon `pricing_at`; `allocation_method` member; linear default + refuse
   lifted-on-unbounded+mass; `Distortion.price` → PricingResult + forwards.
6. **Cleanup:** `add_exa_details` slim→pedagogy, `swap_density_df`→function,
   S-convention unification, delete journey comments, delete `aggregate_keys`.

(Full per-item detail + file lists are in the plan docs — do not duplicate here.)

---

## Open decisions

- **Aggregate O5** — reconcile forwards/backwards `S` across `density_df`,
  `_build_augmented`, `add_exa_sample`, `Distortion.price`. *Investigate.*
- **Harness §3** — confirm grids for `Tail.LN` / `Re.Both` / `Def.LN`.
- **Harness §9** — pareto vs lognorm-cv6 for the no-2nd-moment path;
  hand-built vs seeded-then-committed switcheroo sample; whether to add a
  separate looser-tolerance `calibrate_distortions` snapshot.

## Future ideas (parked)

- Negative `xs` (profit/loss + convolve) — aggregate `plan §6 FI-1`; easy at the
  portfolio-combine level, hard within a unit (random `N`). Author wants periodic
  nudges.
- Integrated aliasing / movable window of interest — aggregate `plan §6 FI-2`.
- `Portfolio.pricing_bounds` rewrite (currently `NotImplementedError`) — portfolio
  `plan §6 FI-1`. **Author asked to be reminded about this one.**
