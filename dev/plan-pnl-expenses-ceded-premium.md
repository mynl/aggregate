# Plan — Phase 1: P&L expenses + ceded premium / ceding commission (deterministic)

> **Status: DRAFT — not executed.** First of three integrated plans (Phase 1 → Phase 2
> reinstatements → Phase 3 variable rating). Establishes the **legs model**, the final GCN
> exhibit shape, gross **expenses**, the **ceded-premium** clause, and **ceding commission** —
> all **deterministic** (no pushforward). Nothing here is stochastic; that is what makes it
> low-risk and a clean foundation. Shared design in **`plan-variable-rating-appendix.md`**
> (legs model §1, per-layer economics §3, naming §5); this plan does not repeat it.

The headline identity: **`UW = Premium − Loss − Expenses`**, doubly additive across
Gross/Ceded/Net. The acceptance index (AI) reads the net UW result unchanged.

---

## Decisions locked (author, 2026-06-27)

1. **Expenses are explicit in all three forms** (no inferred base):
   `25% premium expenses` / `25% loss expenses` (variable, base named) / `2000 fixed expenses`
   (fixed). `expense` and `expenses` both accepted. Optional; absent ⇒ `0`.
2. **Ceding commission (`cede`) books in the expense column** as a credit (negative expense),
   **not** netted against consideration. Net expense `= gross expense − commission − PC`. This
   is what lets slide / PC later make the expense leg stochastic (appendix §1).
3. **Ceded premium is one of `deposit | rol | rate`** per layer (mutually exclusive), entered
   **gross of commission**. `deposit` currency; `rol %` (× share × limit, needs a limit);
   `rate %` (of gross premium, run-time error if none). **No `min` synonym for `deposit`.**
4. **GCN columns become `Premium | Loss | Expense | UW`** (signed, columns add to UW),
   replacing `PnL`'s current `Consideration | Obligation | Margin` where Obligation was loss
   only. Rows and columns add in expectation.

---

## What already exists (ground-truthed — do not rebuild)

- **`PnL`** (`_pnl.py:33`) with `gcn_df` (`:270`, the doubly-additive table) and `summary_df`
  (`:225`, SD-not-CV near break-even at `:231`). This is the table to extend with an Expense
  column and the leg model.
- **The `pnl` output kind** already parses `name <numbers> premium - <loss> … occ_reins freq`
  and builds a `PnL(agg, consideration=...)`. Gross premium and the loss leg are wired.
- **DecL reins clauses** carry pure loss structure today: `occ_reins`/`agg_reins` as
  `(share, limit, attach)` tuples + `occ_kind`/`agg_kind`. **No per-layer premium or expense
  concept exists** — both are introduced here.
- **`exp_premium`/`exp_lr`** on `Aggregate` already express exposure; the `pnl` consideration
  reuses them. Expense is a new, parallel exposure quantity.

---

## Workstreams

### `[legs]` — the legs data model (the foundation; appendix §1)

`_pnl.py`. Introduce the leg as the internal representation behind `gcn_df`:

- A small `Leg` record: `perspective` (`gross`/`ceded`/`net`), `kind` (`consideration`/
  `obligation`, with an `expense` sub-tag on obligations), `value` (scalar **or**, later, a
  `GridDistribution`). Phase 1 only ever sets scalars; the distribution branch is stubbed so
  Phase 2/3 fill it without reshaping.
- `gcn_df` is rebuilt to **iterate legs** into the `Premium | Loss | Expense | UW` frame
  (decision 4). Net legs are derived (`gross ⊕ ceded`). Columns sum to UW; rows sum to net.
- **Keep `PnL`'s public surface stable**: `gcn_df`/`summary_df` names and shapes unchanged
  except for the added Expense column; this is the `[pnl-share]` builder that Phase 2/3 reuse.

### `[expense]` — gross expenses through the GCN

- `Aggregate`/`PnL` gain an expense quantity: fixed (`E_G` currency), or variable as a fraction
  of **premium** or **loss** (the base is stored explicitly, decision 1). Vet the attribute name
  against the existing surface (CLAUDE.md): store the *amount/spec* as a noun (`expense_spec` /
  `gross_expense`), no method shadow.
- The expense leg is a **gross obligation**; net expense `= E_G − C − PC` (commission/PC credits
  land here, decision 2). In Phase 1 `C` is the deterministic `cede`; `PC` is absent.
- `summary_df` gains an expense row and a combined-ratio line (`(L + E)/P`) so the exhibit reads
  as a real underwriting result.
- **AI unchanged**: it consumes the net UW distribution exactly as before (author-confirmed).

### `[ceded-premium]` — `deposit | rol | rate` + `cede`

**Grammar (`decl.lark`).** New optional **layer-premium** decorator on a `reins_clause`:
`deposit <currency>` | `rol <pct>` | `rate <pct>`, and an optional `cede <pct>`. Exactly one
premium form; `cede` only with a premium present. New terminals `DEPOSIT`, `ROL`, `RATE`,
`CEDE` (appendix §5). Watch lexer priority (`rate`/`min` against existing tokens; `cede`
against the `ceded` direction keyword).

**Transformer (`parser.py`).** New spec keys (Lark spec keys are the sanctioned carve-out, but
spell them out): `reins_premium` (resolved to currency where possible; `rate` deferred to
compute when it needs gross premium), `reins_premium_basis` (`deposit`/`rol`/`rate`),
`reins_cede` (commission fraction). Validate: exactly one premium form per layer; `cede`
requires a premium; `rol` requires a finite limit.

**Compute (`_pnl.py`/`_reinsurance.py`).** Resolve each layer's ceded premium to currency
(`rol → share·rol·limit`; `rate → rate·P_G`), sum across the tower into the ceded-consideration
leg; resolve commission `C = Σ cede·CededPrem_layer` into the ceded expense credit. All scalar.

### `[pnl-share]` — make the table builder leg-generic and reusable

Extract the `gcn_df`/`summary_df` construction (signed-additive frame, SD-not-CV margin rule)
into helpers keyed on the leg list, so Phase 2's `ReinstatementAnalysis` and Phase 3's features
call the **same** builder. Surgical; `PnL` public surface unchanged. This is the single change
that makes Phases 2–3 purely additive.

---

## Naming (vetted, CLAUDE.md hard rule)

New on the surface, checked for collision: `gross_expense`/`expense_spec`, `reins_premium`,
`reins_premium_basis`, `reins_cede`, the `Leg` record fields. `reins_*` stays canonical
(memory `project_reins_canonical`). Stored values are nouns; no `__init__` attribute shadows a
method. `cede` flagged against `ceded` for the lexer pass.

---

## Implementation order

1. **`[legs]`** — leg record + rebuild `gcn_df` on it (scalars only), Expense column. Snapshot
   the existing exhibit numbers first; assert unchanged for the no-expense/no-cede case.
2. **`[expense]`** — wire gross expense (all three forms) + combined ratio; AI regression green.
3. **`[ceded-premium]`** — grammar + transformer + compute for `deposit|rol|rate` + `cede`.
4. **`[pnl-share]`** — factor the builder for Phase 2/3 reuse.

## Tests

New: `tests/test_pnl_expenses.py`, `tests/test_pnl_ceded_premium.py`. Cover: the three expense
forms (premium-based, loss-based, fixed) and the absent default; GCN rows/columns add with
expense + commission; net expense `= E_G − C`; `deposit`/`rol`/`rate` equivalence where they
should coincide; `rate` with no gross premium → clear run-time error; `cede` without premium →
error; `rol` without limit → error. Add the DecL lines to `src/aggregate/agg/test_decl.agg`
(keep-in-sync rule) with **hand-written spec assertions** (the SLY snapshot cannot cover new
lines — Phase 2's wrinkle applies). **Regression bar:** every existing `test_suite.agg` line
still parses and snapshot-matches (grammar addition is purely additive); `uv run pytest` green.

## Housekeeping

Plan-based change → bump `1.0.0a*` in `pyproject.toml`; add a `CHANGELOG.md` section (P&L
expenses; ceded-premium `deposit|rol|rate`; ceding commission `cede`; legs model; GCN gains an
Expense column). Add a `dev/TODO.md` entry; move this plan to `dev/done/` at close. Docs: a
follow-up; keep `:meth:`/`:class:` refs in lockstep but do **not** build the doc tree in the
iteration loop.
