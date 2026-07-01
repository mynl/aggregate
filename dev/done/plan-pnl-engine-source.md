# Plan — [PnL-Engine-Source]: a P&L wraps a complete engine (no bastards)

Status: **DONE** (`1.0.0a125`, 2026-07-01). The load-bearing structural refactor.
Followed a123 `[PnL-Exhibits]` and a124 `[DecL-Labels]` (landed first, as planned).

**Execution notes / deviations from the draft below:**
- Two draft claims were wrong (verified in code): `Aggregate.exp_premium` was **not**
  retained — added `self.exp_premium`/`self.exp_lr` in `__init__` (needed for
  Portfolio accumulation + `inherit`); and the SLY snapshot has **zero** pnl
  programs, so no re-baseline was needed.
- Inline engine = **any** valid aggregate form (full/dfreq/tweedie/rename), not just
  `full` — `agg_body` factored across all `agg_out` alternatives.
- The unparser (`decl_writer._render_pnl`) also needed rewriting (a pre-existing gap
  that broke `test_decl_unparser`) — now emits the engine-wrapped form, `inherit
  premium`, and `less port.NAME`.
- Deferred as `NotImplementedError`: `retro` over a reinsured engine; loss-basis
  expense as stochastic `rate·loss` in the exploded tower (decision 9 fallback);
  expenses on a port-sourced P&L.

## The problem — the `pnl` grammar is a fork of the `agg` grammar

Today (`decl.lark`):

```
agg_out: AGG name exposures        layers sev_clause occ_reins freq agg_reins approx_clause orientation trailer
pnl_out: PNL name pnl_premium LESS pnl_exposures layers sev_clause occ_reins freq agg_reins approx_clause expense_clause trailer
```

`pnl_out` **hand-copies** the agg body (`layers sev_clause occ_reins freq
agg_reins`) *and* **tears** the clean `premium at lr` exposure (`agg`, line 433)
into two half-clauses — `pnl_premium` (`10000 premium`) plus a bare
`pnl_exposures` (`85% lr`). That fork is the "half-baked agg inside a pnl"
bastardization. This plan deletes it: **a P&L wraps a complete stochastic engine**
(an `agg`, or a reference to a stored `agg` / `port`) and reads its loss out —
"what comes out of the sausage maker."

The engine-agnostic API from a122/a123 (`create_pnl(source, …)` over an opaque
slot) is exactly what makes this cheap: a `Portfolio` is *just another engine*, so
this **also subsumes** the `[Portfolio-of-PnL]` TODO item (no bespoke `PortPnL` —
you point a `pnl` at a `port`).

## Decisions (from the design dialogue — all settled)

> **Resolved at execution (2026-07-01), superseding the stale grammar sketch below:**
> * **Labels** — a124 already landed `display_label: AS label` (`as "Friendly
>   Label"`, ID or quoted string) as a per-object slot after each internal `name`.
>   Nothing to decide: the factored `agg_source_inline` keeps `AGG name
>   display_label agg_body`, the wrapping `pnl`/`xpnl` keeps its own label, and
>   both flow into exhibit column headers.
> * **Inline engine = any valid aggregate.** A pnl is engine-agnostic — it only
>   reads the outcome distribution (`agg.density_df.p_total`). So the embedded
>   `agg NAME …` may be *any* inline agg form (full, dfreq, tweedie, rename), not
>   just `full`; plus stored `agg.NAME` / `port.NAME`. `agg_body` is factored to
>   span **all** `agg_out` alternatives minus the top-level trailer. Decision 1
>   (named, complete engine — no anonymous body) is unchanged.

1. **No bastards — breaking.** The `less` source is a *complete* engine only:
   `agg name <body>` (inline, named — *any* agg form, see resolved note above),
   `agg.NAME`, or `port.NAME`. **No anonymous inline body.** Every existing
   `pnl X 10000 premium less 85% lr sev … poisson` is rewritten to wrap a real
   `agg`.
2. **Double-`less` for expenses** — `pnl NAME <premium> less <engine> less
   <expenses>`. The second `less` (a reserved keyword) is a **hard anchor** that
   dissolves the nested-trailer ambiguity: the embedded engine's tail cannot cross
   a `less`. Reads as English ("premium, less the book, less the expenses"). The
   second `less` clause is optional (omit when no expenses).
3. **`inherit premium` — explicit, no magic.** Premium is mandatory: a number,
   `inherit premium` (copy the engine's premium), or `retro … premium`. `inherit`
   reads `Aggregate.exp_premium` (already retained — verified) and is a **build
   error** if the engine has none (`… less agg B 1000 loss …` → no premium).
4. **Two independent premiums is a feature, not a bug.** The engine's premium is a
   *sizing / exposure* input (`10000 premium at 65% lr` → E[loss] 6500); the P&L's
   premium is the *consideration* (what you book). `pnl X 12000 premium less agg B
   10000 premium at 65% lr …` books 12 000 over a book sized at 10 000 — rate
   adequacy, on purpose. `inherit` is the convenience that says "book the technical
   premium."
5. **`xpnl` (exploded) → returns the tower.** Same shape as `pnl`, but returns the
   `PnLTower` with the tower's exhibits (Gross / net-occ / net-agg; expenses
   stepped down by each layer's commission). This is consistent with always-PnL:
   the *keyword* announces the shape, so it is not a domain-specific return-type
   wart. `xpnl X …` ≈ `pnl X …`.tower.
6. **`xpnl` + `port` → error** (too ambiguous — the port total hides its units, so
   there is nothing to explode).
7. **Economics stay in reins clauses, resolved by the wrapper.** `deposit` /
   `cede` / `rol` / `rate` have no place on a gross agg book; they live in reins
   clauses, and the wrapping `pnl` / `xpnl` resolves them exactly as the `pnl`
   factory does today. The **standalone-agg guard is unchanged** (a bare `agg …
   cede …` still errors). Whole-book terms use the existing degenerate idiom
   (`ceded to inf xs 0 deposit …`). *No new "economics on a gross agg" surface.*
8. **`port.NAME` uses the net-net portfolio total** and sees nothing inside; a
   port-sourced P&L is inherently the *plain* case. `Portfolio` must **accumulate
   its units' premium** (it does not today — verified) and expose it for `inherit`.
9. **Loss-basis expense passes through as stochastic `rate·loss`** in the exploded
   tower too (reversing the a123 GCN-scalar carve-out). If the per-tier commission
   stepping makes this hard, raise **`NotImplementedError`** and revisit (a user
   would realistically fold `%loss` into loss directly; the split is a
   reporting nicety).
10. **`retro … premium less agg.X`** works when `X` has no reinsurance (retro reads
    the net output, never peeks inside — the current clean 1-D case). `xpnl` +
    `retro` over a *reinsured* engine is the deferred **retro-plus-reins** case →
    stays `NotImplementedError`.

## Grammar sketch

```
pnl_out:  PNL  name pnl_premium LESS agg_source (LESS expense_clause)? trailer
xpnl_out: XPNL name pnl_premium LESS agg_source (LESS expense_clause)? trailer

agg_source: AGG name agg_body     -> agg_source_inline    // named, complete
          | BUILTIN_AGG           -> agg_source_ref_agg   // agg.NAME (existing terminal)
          | BUILTIN_PORT          -> agg_source_ref_port  // port.NAME (NEW terminal)

// factor the shared body out of agg_out so both reuse it
agg_body: exposures layers sev_clause occ_reins freq agg_reins approx_clause orientation
agg_out:  AGG name agg_body trailer

pnl_premium: numbers PREMIUM      -> pnl_premium_fixed
           | INHERIT PREMIUM      -> pnl_premium_inherit
           | RETRO collar PREMIUM -> pnl_premium_retro

BUILTIN_PORT.3: /port\.[a-zA-Z][a-zA-Z0-9._:~\-]*/
INHERIT.2:      /inherit(?![a-zA-Z0-9._:~\-])/    // new reserved word
XPNL.2:         /xpnl(?![a-zA-Z0-9._:~\-])/       // new top-level keyword
```

Notes:
* **`agg_body` factored out** of `agg_out` and reused by `agg_source_inline` — the
  refactor's core move. The embedded agg keeps its `AGG name` prefix but **drops
  its own `trailer`** (the wrapping `pnl`/`xpnl` owns the trailer/notes) — this,
  plus double-`less`, is what removes the nested-tail ambiguity.
* `agg_out` (top-level) keeps its trailer; only the *embedded* form omits it.
  Factor as `agg_body` (no trailer) + a top-level `agg_out: AGG name agg_body trailer`.
* Reserve `inherit`, `xpnl`; add `BUILTIN_PORT`. Grep the suite / KB for units
  named `inherit` / `xpnl`.

## Implementation (Python — small, given the API)

The heavy lifting is grammar + sweep; the runtime is a thin wiring job.

* **`underwriter` factory / `_snapshot_pnl`:**
  * `agg_source_inline` → build the inner `Aggregate` from `agg_body` (reuse the
    existing agg build path); `agg_source_ref_agg` / `_ref_port` → look up the
    stored object (`self._knowledge` / built cache).
  * `pnl_premium_inherit` → read `engine.exp_premium` (agg) or the accumulated
    port premium; **error if 0/absent**.
  * `pnl` → `create_pnl(engine, consideration=premium, obligation=net-loss, …)`
    (today's plain / gcn snapshot paths, unchanged in spirit).
  * `xpnl` → `gcn_tower_from_aggregate(engine, …)` returning the `PnLTower`
    (its `gcn_df` is the headline); `xpnl` + port → raise.
  * The **deferred-snapshot** pattern already fits ("build engine, then snapshot").
* **`Portfolio`:** accumulate `sum(unit.exp_premium)` → a `premium` / `exp_premium`
  attribute + expose the total-loss density as the `create_pnl` source
  (`density_df` already exists). Vet the name against the Portfolio surface.
* **`Aggregate.exp_premium`** already retained — no change (verified line 1513 /
  used as the `make_pnl` gross-premium default line 1165).
* Loss-basis expense stochastic in the tower per-tier, or `NotImplementedError`
  (decision 9). `retro` + reinsured engine → `NotImplementedError` (decision 10).

## The sweep (most of the labor)

Breaking-change rewrite of **every** `pnl` program to `pnl NAME <prem> less agg
NAME <body> [less <exp>]`:

* `src/aggregate/agg/test_suite.agg`, `src/aggregate/agg/test_decl.agg` (every
  `pnl …` line) — and the parametrized tests that consume them.
* `tests/test_pnl*.py`, `tests/test_reinstatement_decl.py`,
  `tests/test_variable_rating_decl.py`, `tests/test_create_pnl.py` DecL strings.
* `dev/regen_features.py` build programs.
* Docs (`.rst` / `.qmd`) DecL examples — keep in lockstep, author rebuilds.
* **Re-baseline the SLY snapshot** (`tests/data/expected_specs.json`) — the `pnl`
  spec shape changes materially.

Delegate the mechanical rewrites to parallel subagents (as in a122/a123), each
with the before/after form and an instruction to preserve intent.

## Phases

1. **[Engine-Refs]** — `BUILTIN_PORT` terminal + `agg.NAME` / `port.NAME`
   resolution in `agg_source`; `Portfolio` premium accumulation + total-loss
   source. (Testable in isolation: `pnl X 100 premium less agg.STORED`.)
2. **[Pnl-Body-Refactor]** — factor `agg_body`; rewrite `pnl_out` to
   `less <agg_source> less <expenses>`; drop the embedded trailer; `inherit`
   keyword. The breaking grammar change + snapshot re-baseline.
3. **[Xpnl]** — `xpnl` kind → `PnLTower`; `xpnl` + port error.
4. **[Economics-Expense]** — wrapper economics threading over the embedded engine;
   loss-expense stochastic-or-NYI; `retro` + agg validation / retro-plus-reins NYI.
5. **[Sweep]** — rewrite all programs, re-baseline the snapshot, docs.

## Testing / housekeeping

* New `tests/test_pnl_engine_source.py` (inline agg, `agg.NAME`, `port.NAME`,
  `inherit premium` + no-premium error, two-premiums feature, `xpnl` tower +
  port-error, retro-plus-reins NYI). Append DecL programs to `test_decl.agg`.
* Version bump, `CHANGELOG.md` `[PnL-Engine-Source]` (call out the breaking `pnl`
  syntax + `[Portfolio-of-PnL]` subsumption), `dev/TODO.md` (mark item 3
  subsumed), `dev/FEATURES.csv` (new `xpnl`; `Portfolio.premium`).
* Full suite green + `dev/regen_features.py` audit clean before close.

## Risks

* **Snapshot churn + program sweep** — broad and mechanical; the main cost. Scope
  up front (above), delegate, verify green.
* **Grammar refactor** — the `agg_body` factoring + embedded-trailer drop is where
  Earley ambiguity would surface; double-`less` is the mitigation. Exercise
  `test_suite.agg` hard after phase 2.
* **`Portfolio` premium accumulation** — genuinely new (verified absent today);
  keep it a plain sum of unit `exp_premium`, no distribution.
* **Behavioral relaxations to confirm in review:** none on the standalone agg
  (guard unchanged, decision 7); the only new *errors* are `inherit`-without-
  premium, `xpnl`+port, and retro-plus-reins NYI.

## Recommended order — labels first, then engine-source

`[DecL-Labels]` → **then** `[PnL-Engine-Source]`. Labels are additive, low-risk,
and independent of this refactor's skeleton; landing them first means the big
breaking **program sweep here is written once against final syntax** — most
concretely, `[DecL-Labels]` changes the *expense* grammar (grouping) and this plan
*relocates* expenses (second `less`), so doing labels first lets the sweep write
every expense program in final grouped form once instead of twice. (Mild
preference, not load-bearing — the two are grammar-independent — but it saves a
rewrite pass on the expense programs.)
