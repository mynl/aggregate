# Plan: `approximate` — method-of-moments aggregates via DecL keyword (Option C)

## Goal

Reinstate the method-of-moments shortcut for high-frequency aggregates, but as an
**opt-in, design-time DecL keyword** — you declare it, you know exactly what you
get. Instead of an FFT freq×sev convolution, the aggregate's density is a
**shifted gamma / shifted lognormal fitted to the aggregate's first three
moments**.

```
agg Big 1e6 claims sev lognorm 100 cv 2 poisson approximate sgamma
```

## The mechanism (the whole trick — no special compute path)

`approximate` does **not** add a branch to the convolution. It **rewrites the
spec at construction** to an ordinary exact aggregate:

```
agg Big 1 claim sev <<fitted shifted gamma/lognormal>> fixed note{<original program>}
```

i.e. **fixed frequency = 1 claim**, **severity = the MoM-fitted continuous
distribution**, with the original DecL program preserved in a `note{...}`. After
substitution the object is a completely ordinary aggregate: its density is just
the fitted severity, and **everything downstream — `density_df`, `ftagg_density`,
validation, and crucially the Portfolio combine — works with zero special-casing.**
There is nothing to do after construction.

This is why Option C is clean: the directive is design-time and self-documenting,
the portfolio case is automatic (a member built from DecL carries its own
substitution), and the compute core stays untouched.

## Scope decisions (settled with author)

- **Opt-in at design time** via the DecL keyword (and the matching constructor
  parameter). Never a default; no claim-count auto-trigger.
- **`approximate = exact | sgamma | slognorm`** (`exact` is the inert default).
- **occ-reinsurance + approximate → error.** Per-occurrence reinsurance acts on
  the severity *before* the freq convolution, which the MoM fit bypasses, so the
  combination is meaningless. Reject it (a parse/build-time error with a clear
  message). **Aggregate reinsurance is fine** — it applies to the finished
  aggregate density and rides along unchanged.
- **Works for positive *and* negative skew.** The shifted fits must handle
  reflected cases (`−lognorm` etc.); verify/extend the fitters (see below).

## Changes

### 1. Grammar (`decl.lark`)
- New terminal `APPROXIMATE.2` (with the standard keyword negative-lookahead),
  and add `approximate` and `approx` to the `ID` exclusion list so it can't be a name.
- New clause:
  ```
  approx_clause: APPROXIMATE ID   -> approx_set     // ID ∈ {exact, sgamma, slognorm}
               |                  -> approx_none
  ```
  (Use `ID` for the kind and validate the three allowed words in the transformer,
  to avoid three more terminals.)
- Insert `approx_clause` into the `agg_out` alternatives (between `agg_reins` and
  `trailer`), and `pnl_out` if approximate-on-pnl is to be allowed — **recommend
  NOT** on `pnl` for now (signed support; out of scope, below). This is the
  "bigger surface area" — each `agg_out_*` alternative gains the clause and its
  transformer method gains one positional arg.

### 2. Transformer (`parser.py`)
- `approx_set` → `{'approximate': <kind>}` (validate kind; raise on unknown);
  `approx_none` → `{'approximate': 'exact'}`.
- Merge `approximate` into the emitted spec dict (a normal spec key, like any
  other clause).
- **Semantic check:** if `approximate != 'exact'` **and** `occ_reins` present,
  raise a clear DecL error ("approximate is incompatible with occurrence
  reinsurance"). (Transformer-level semantic validation, surfaced as a parse-time
  error.)
- Append the new programs to `src/aggregate/agg/test_decl.agg` (DecL-sync rule).

### 3. `Aggregate.__init__` — the substitution (`distributions.py`)
- New constructor parameter `approximate='exact'`, threaded from the spec by
  `build`/`Underwriter` (verify the spec→constructor passthrough).
- When `approximate != 'exact'`:
  1. Compute the **theoretical aggregate moments** `(m, cv, skew)` of the
     requested freq×sev aggregate — the constructor already builds these (the
     `statistics`/`MomentWrangler` path). This is the one ordering subtlety:
     moments must be computed from the *original* freq/sev **before** the swap.
  2. Belt-and-suspenders reject occ-reins here too since constructor may be called
     directly.
  3. Fit via the existing `sgamma_fit` / `sln_fit` (reuse
     `approximate_from_mcvsk`, which already emits the fitted severity in several
     `output=` forms — take the `Severity`/`sev_kwargs` form).
  4. **Swap the internal objects:** frequency → fixed 1; severity → the fitted
     continuous `Severity`; keep `agg_reins`.
  5. Record the original program text in `note` (append, don't clobber an
     existing note).
  - Then ordinary construction/update proceeds — it is now an exact 1-claim agg.
- `describe`/`info` should make the approximation visible (the `note` carries the
  original program; consider an explicit "approximate: sgamma" line).

### 4. Negative-skew fits (`sln_fit` / `sgamma_fit`)
The legacy path logged "Negative skewness, ignoring and fitting unshifted
distribution" — i.e. it did **not** truly support negative skew. Verify current
behavior and, if needed, extend the shifted fitters to reflect (fit the shifted
dist to `−X` and flip), so `approximate` is correct for left-skewed aggregates.
Author's note: "easy extension / maybe already done" — confirm first.

## Verification

1. **Correctness vs exact.** For a moderate aggregate, build it both ways
   (`approximate exact` vs `sgamma`/`slognorm`) and assert the approximate
   `describe` moments match the exact aggregate's first three moments to fit
   tolerance (MoM matches mean/cv/skew by construction).
2. **Portfolio combine.** A `port` with an `approximate sgamma` member updates
   with no special handling and conserves mass; the member contributes its fitted
   density to the total. (This is the headline requirement — confirm it "just
   works" via the spec substitution.)
3. **Negative skew** — a left-skewed aggregate fits without the legacy
   "ignoring" fallback (after step 4).
4. **occ-reins rejection** — `… occurrence net of … approximate sgamma` raises a
   clear error; `aggregate net of … approximate sgamma` succeeds.
5. **`note` round-trip** — the original program is recoverable from the built
   object.
6. **`uv run pytest`** green, including the new `test_decl.agg` lines.
7. **freeze/check** — existing objects unaffected (no current test_suite program
   uses `approximate`); all-match expected.

## Housekeeping (standing rules)
- Bump `pyproject.toml` `1.0.0a*`.
- `CHANGELOG.md`: new `approximate exact|sgamma|slognorm` DecL keyword — MoM
  shifted gamma/lognormal aggregates via spec substitution; original program kept
  in `note{}`; incompatible with occurrence reinsurance.
- `dev/TODO.md`: mark the approximate item done with the version, mark F1 done in
  table, amend item to just Ability to approximate.
- `src/aggregate/agg/test_decl.agg`: add the example programs.
- Regenerate `ref_include.rst` per the usual recipe (grammar changed) — see
  [[project_ref_include_regen]].
- Move this plan to `dev/done/plan-approximate.md` on landing.

## Open questions for the author
1. **`approximate` on `pnl`?** Recommend excluding for now (signed support;
   the MoM shifted fits assume a non-negative thick right tail). ANSWER: you
   approximate the aggregate part, then the premium - is a trivial adjustment.
   I.e., the approximation applies to the loss part and the program is rewritten
   `pnl 1000 prem - <<new approx spec>>`
2. **Surface in `info`** `info` for Aggregates reports Approximation exact|slognorm|sgamma.

## Out of scope
- The convolution core (untouched — that's the point).
- The existing `Aggregate.approximate()` *method* (returns a separate object) —
  stays as-is; this keyword is the in-place spec-substitution route.

---

## Execution notes (landed 1.0.0a47)

Executed after a review pass that surfaced two scope decisions to the author:

- **Negative skew → full support (author chose).** Handled by reflecting at the
  *approximate-helper* level, **reusing `sln_fit`/`sgamma_fit` unchanged** rather
  than editing those fitters. The fit is performed on the reflected aggregate
  `-A` (right-skewed) and mapped back via the existing `sev_reflect` machinery
  (`Y = sev_loc − X`, base built at loc 0), so the freeze-checked windowing code
  that *also* calls those fitters is untouched. Verified to match the exact
  aggregate's (m, cv, skew) to ~7 sig figs for both `sgamma` and `slognorm`.
- **Grammar surface → narrow (author chose).** `approx_clause` added only to
  `agg_out_full`, `agg_out_dfreq`, `pnl_out_full`, `pnl_out_dfreq` — not to the
  `tweedie`/`rename`/`builtin` forms (already-specific or lookups). Less
  ambiguity risk, less transformer churn.

Implementation refinements vs the plan text:

- **Substitution mechanism (plan step 3).** Implemented as *early-detect at the
  top of `__init__`* → build a throwaway exact copy from `self._spec` to read the
  analytic moments (no FFT) → **overwrite the local construction variables**
  (freq→fixed-1, sev→fitted) *before* the freq/sev setup runs. The object is then
  a literal ordinary 1-claim aggregate — no mid-constructor object teardown. The
  original spec was already captured into `self._spec`, so persistence round-trips.
- **Signedness test (the subtle bug).** A fitted shifted gamma can have a very
  negative `loc` while *all* its mass sits on the positive axis (the beta example:
  `loc=-15,115`, mass at `+15,873`). The first cut keyed signing on `shift < 0`
  and corrupted that case (wrong mean/cv/skew). Fixed: the severity is marked
  `sev_signed` only when the fit's **low quantile** (`ppf(1e-8)`) is below zero —
  i.e. when there is genuinely mass to keep below 0 — so the common positive case
  stays on the cheap 0-based grid.
- **Symmetric limit.** `|skew| ≤ 1e-3` returns a **normal** (the common limit of
  both shifted fits), signed only if its left tail dips below zero.
- **`note`.** The full original program isn't available in `__init__` (it is set
  on `self.program` *after* construction and round-trips via `to_agg`), so the
  note carries a synthesized one-line fit summary instead.

Verification: full suite **1113 passed** (+14 in `tests/test_approximate.py`);
freeze/check **all 146 objects within 1e-12**; `ruff` clean; `ref_include.rst`
regenerated (grammar changed); doc rebuild pending per CLAUDE.md.
