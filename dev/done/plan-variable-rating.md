> **SUPERSEDED 2026-06-29** — moved to `dev/done`; superseded by `dev/plan-bivariate-legs.md` (see its consolidation note for why and what was subsumed). Executed a118–a120.

# Plan — Phase 3: variable rating (retro, swing, slide, profit commission, corridor)

> **Status: DRAFT — not executed.** Third of three integrated plans (Phase 1 → Phase 2
> reinstatements → **Phase 3**). Adds the five remaining variable-rating features. **No new
> engine work** — every feature is a `ContractTerms` object supplying a φ, run through Phase 2's
> `pushforward_1d`/`pushforward_2d` (basis selects the path), writing one Phase-1 leg. Shared
> design in **`plan-variable-rating-appendix.md`** (legs §1, dimensionality §2, per-layer
> economics §3, loss basis §4, naming §5); this plan does not repeat it.

These are all the same object as reinstatement premium: a deterministic function of realized
loss, pushed forward over the loss distribution, making one leg stochastic.

---

## Decisions locked (author, 2026-06-27; naming refined 2026-06-28)

0. **One variable feature per program — no stacking (author, 2026-06-28).** A
   program carries **at most one** variable-rating feature (one of retro / swing /
   slide / pc / corridor), full stop. This supersedes the per-layer composition
   matrix (appendix §3) for Phase 3: there is no swing+slide, no stacked aggregate
   features, so the LR denominator for slide/pc/corridor is always the deterministic
   (fixed) ceded premium and φ stays a pure 1-arg map. Validation collapses to a
   single "more than one variable feature → error" check. Reinstatement (Phase 2)
   remains its own already-shipped path.

1. **Each feature is a `ContractTerms` subclass** owning its φ, loss basis (appendix §4), and
   target leg. The layer's **basis** (occurrence/aggregate) selects 1-D vs 2-D pushforward
   (appendix §2); the feature never chooses dimension. A thin `ContractTerms` base **is**
   introduced (author, 2026-06-28); `ReinstatementTerms` is refactored under it. The base holds
   class-level `target_leg` / `loss_basis` metadata, one abstract vectorized `phi`, and the shared
   finite/monotone validation utilities (generalized from `_validate_callable`). Reinstatement is
   the one **two-map** feature (`reinstatement_premium` = premium leg via `phi`; `recovery` = the
   annual-cap loss transform, left as-is); the base stays thin so it is not over-fit to the
   single-leg features.

   **Spec-key naming (author, 2026-06-28; supersedes the `[decl]` flat `*_terms` keys).** The four
   layer-decorating features follow the shipped per-layer convention `{which}_reins_{slot}` with
   `which ∈ {occ, agg}` (parallel to `occ_reins_premium` / `occ_reins_cede` / `occ_reins_reinst`,
   no `_terms` suffix): `occ_reins_swing` / `agg_reins_swing`, `occ_reins_slide` /
   `agg_reins_slide`, `occ_reins_pc` / `agg_reins_pc`, `occ_reins_corridor` / `agg_reins_corridor`.
   They are per-layer lists holding the raw DecL params, emitted by `_split_reins` (`parser.py`)
   only when a layer carries the slot; the `*Terms` dataclass is assembled at analysis time (same
   pattern as `occ_reins_reinst`). Rationale: the tier selects 1-D vs 2-D (appendix §2) and agg
   features stack, so the spec must record *which* tier and index *by layer* — a flat scalar key
   cannot. **`retro_terms` stays flat** — it is the account-level rating clause with no occ/agg
   dimension, the one genuine exception.
2. **`slide` is expressed as `(commission at loss-ratio)` anchor pairs**, piecewise-linear
   between, flat outside the end anchors — **no explicit min/max** (they are the end anchors,
   author-confirmed): `slide 45% at 60% and 25% at 70% and 19% at 80%`.
3. **`retro` lives in the rating clause** (account level), not a reins clause — it varies
   **gross** premium as a function of net account loss. All other features decorate a reins
   layer (appendix §3).
4. **`corridor` reuses `po`/`xs`** as a loss-ratio layer; **`pc`** uses `pc <pct> after <pct>`.
   The per-layer mutual-exclusion / composition rules are appendix §3.

---

## What already exists after Phases 1–2 (do not rebuild)

- **`[engine]`** (Phase 2): `pushforward_1d(gross_density, φ)`, `pushforward_2d(joint, φ)`,
  `transformed_moments` — generic over a vectorized φ, returning a `GridDistribution` leg.
- **Legs model + leg-iterating `gcn_df` builder** (Phase 1 `[legs]`/`[pnl-share]`): filling a
  stochastic leg is additive, no table reshape.
- **Ceded-premium clause `deposit|rol|rate`, `cede`, expenses** (Phase 1): `swing` replaces the
  fixed premium clause; `slide` replaces `cede`; both write legs Phase 1 already defined.
- **`ReinstatementAnalysis` + the `(L,R)` joint + `occ_bivariate`** (Phase 2): the 2-D source
  every occurrence-basis feature reuses; aggregate-basis features use the 1-D gross density.
- **The `[one-variable-occurrence-layer]` rule** (appendix §2, first applied in Phase 2): at
  most one occurrence layer may carry any variable feature.

---

## Workstreams

### `[terms]` — the five `ContractTerms` subclasses (`reinstatement.py` or a sibling module)

Frozen dataclasses, vectorized φ, no FFT code. Each: validation (finite/monotone where
required), `from_*` alt constructors where useful, and a `target_leg` + `loss_basis` (appendix §4).

- **`RetroTerms`** — `prem = clip(basic + lcm·L, minimum, maximum)`, `L` = net account loss.
  Keywords `retro basic <b> lcm <m> min <lo> max <hi>` (`min` = `basic`-floor if omitted;
  `max = ∞`/`premium` if omitted; basic = additive, lcm = multiplier).
- **`SwingTerms`** — same shape as retro but ceded premium as a function of **ceded loss**;
  `swing` replaces `deposit|rol|rate`. Reuse the retro `basic/lcm/min/max` machinery.
- **`SlideTerms`** — ceding commission as a decreasing piecewise-linear function of the **ceded
  LR**, from `(commission, LR)` anchor pairs (decision 2); flat outside the ends. Writes the
  **expense** leg (credit). Validate anchors monotone in LR.
- **`ProfitCommissionTerms`** — `PC = (p·(1 − LR − allowance))₊` of premium, `pc <p> after
  <allowance>`. Writes the **expense** leg (credit).
- **`CorridorTerms`** — cedant retains a loss-ratio corridor; ceded loss `A' = A − corridor(A)`.
  Keywords `corridor <share> po <hi_LR> xs <lo_LR>`. Writes the **ceded-loss (obligation)** leg
  (a comonotone transform of `A`).

**Worked-example discipline** (reinstatement §7 lesson): every feature ships a hand-checked,
sign-correct worked example *before* it becomes a test fixture — especially retro/swing
(collar arithmetic) and slide (anchor interpolation).

### `[decl]` — grammar + transformer + per-layer validation

`decl.lark` / `parser.py`. New terminals `RETRO`, `SWING`, `BASIC`, `LCM`, `MIN`, `MAX`,
`SLIDE`, `PC`, `AFTER`, `CORRIDOR` (appendix §5; reuse `at`/`and`/`po`/`xs`/`premium`/`loss`).
`retro` decorates the **rating clause**; the rest decorate a reins layer. New spec keys:
account-level `retro_terms` (flat); per-layer `{which}_reins_swing` / `_slide` / `_pc` /
`_corridor` for `which ∈ {occ, agg}` via `_split_reins` (decision 1, naming-refined).

**Per-layer validation matrix (appendix §3)** — implement as one check over a layer's filled
slots: one premium mechanism (`deposit|rol|rate|swing`), `cede` xor `slide`, `pc` optional,
`corridor` optional; cross-layer `[one-variable-occurrence-layer]`; `rate` needs gross premium;
`rol` needs a limit. Clear, specific error messages with the offending layer.

### `[analysis]` — wire features through the engine

Each feature, at build/exhibit time: pick `pushforward_1d` (aggregate basis) or `pushforward_2d`
(occurrence basis, reusing the cached `(L,R)` joint), apply φ, write the leg, let the Phase-1
builder render the GCN. `retro` pushes the **gross** density (its loss basis is net-of-inuring
account loss — 2-D iff the inuring reins is occurrence). Reuse the Phase-2 lazy-build + cache.

### `[exhibit]` — first-class surface (reuse, do not reinvent)

Features surface through the same `PnL`/analysis members built in Phase 2 (`gcn_df`,
`summary_df`, `validation_df`, `tail_df`, `plot`, `_repr_html_`, `qd`). Add per-feature
`reins_description`/`reins_explanation` (treaty narrative: the φ, basis, collar/anchors).
Update `dev/FEATURES.csv` for the new `*Terms` classes and rerun the introspection cross-check.

---

## Implementation order

1. **`[terms]`** — the five dataclasses + validation + corrected worked examples; pure, fast to
   test, no engine.
2. **`[analysis]`** wiring, simplest first: **swing** (closest to RP) → **slide** → **pc** →
   **corridor** (obligation-leg path) → **retro** (gross-premium, rating-clause, parallel).
3. **`[decl]`** grammar + transformer + the per-layer matrix **last** (it only sugars the Python
   terms).
4. **`[exhibit]`** trimmings + `FEATURES.csv`.

## Tests

New: `tests/test_variable_rating_terms.py` (φ correctness, corrected worked examples, collar/
anchor edge cases), `tests/test_variable_rating_analysis.py` (1-D vs 2-D pushforward agreement
on aggregate vs occurrence basis; GCN identities with stochastic premium/expense/loss legs;
Monte-Carlo cross-check). Per-layer **negative** cases for every appendix-§3 error
(two premium mechanisms; `cede`+`slide`; second variable occurrence layer; `rate` with no
premium). DecL lines into `src/aggregate/agg/test_decl.agg` with hand-written spec assertions
(SLY-snapshot wrinkle). **Regression bar:** existing `test_suite.agg` still parses and
snapshot-matches; `uv run pytest` green.

## Housekeeping

Plan-based change → bump `1.0.0a*`; `CHANGELOG.md` section (variable rating: retro, swing,
slide, profit commission, corridor; all via the shared pushforward engine). `dev/TODO.md`
entry; move to `dev/done/` at close. This is intended to be the **last** feature plan for v1.0.
Docs: reinsurance case-study page (ties to `D7`) is a follow-up; keep refs in lockstep, do
**not** build the doc tree in the loop.
