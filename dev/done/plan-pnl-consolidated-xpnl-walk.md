# [PnL-Consolidated-XPnL-Walk] — pnl consolidates, xpnl walks

**Status: DONE — executed 2026-07-04 as `1.0.0a135` (Phase 1) +
`1.0.0a136` (Phases 2–5); see `CHANGELOG.md` and `dev/TODO.md`
[PnL-Consolidated-XPnL-Walk].**

**Was: READY (rev 4) — all decisions settled by the author 2026-07-04;
no open flags. Resolved along the way: net-premium label qualification OK;
NO XPnL class (xpnl is a recipe, not a type); two bumps OK; ladder renames
`P01` / `κ01…κ99` as **pure renames** — every PnL frame is **payoff sign
convention, LEFT tail bad**, always ([Decision-Kappa-Outcome-Direction]).
Execute on "go" after the fresh-session review.**
Follows [PnL-Punchups-01] (`1.0.0a134`, `dev/done/plan-pnl-punchups-01.md`).
Grew out of the occ-tower session of 2026-07-04: the occ guaranteed-cost
`pnl` flattened the walk into a single group with the occ deposit booked
under Obligation (card read "gross premium, higher loss" — arithmetic footed,
presentation misled), and plain `agg`s rejected premium clauses outright.

## The design (settled)

Two objects, two questions, no conditional shapes:

* **`pnl` — "what is my position?"** The consolidated net-in-to-net-out
  view. Always one group; always the flat three-row card. Ceded economics
  are **netted out and not shown** — the pnl is the P&L of *what the object
  does*, and a reinsured aggregate's default output is its net.
* **`xpnl` — "how did I get there?"** The walk. Always the step tower:
  gross → each cover → Total, with per-step results, running nets, and the
  closing impact. **xpnl is a DecL construct — a way of building, not a new
  thing**: it feeds the entirely generic group-ledger tower the kernel
  already has, returning a plain multi-group :class:`PnL` that carries the
  exploded `(Step, View)` card and `(Step, View, Line)` stats sheet from
  a134. No new class; the repr and `.groups` say what it is.

The demo narrative *is* the spec: build the gross agg; overlay occ / agg
reinsurance and see the loss impact; ask "what about P&L?" → `pnl` (gross
program → gross view; reinsured program → net view); ask "what is the
impact, stepping down through occ and agg?" → `xpnl` on the same engine.

## Settled decisions (author, 2026-07-04)

- **[Decision-PnL-Is-Consolidated].** `pnl` always returns a single-group
  P&L. The a125/a129 "premium clause promotes to the Gross/Ceded/Net group
  ledger" behavior is **removed from pnl** (it moves to xpnl). Card:
  `Consideration` = **net premium** (gross − ceded premiums + commissions),
  `Obligation` = net loss + the pnl's own expenses, `Margin`. Strictly you
  cannot net consideration against obligation — the gross/ceded-split
  accounting card (supports GAAP / STAT / IFRS reporting) is the *ideal* —
  but "summary" means summary: start simple; the detail is one `xpnl` away.
  The split card is **pended as `accounting_summary_df`** (author-named) —
  recorded as **[Accounting-Summary-DF]** in `dev/TODO.md` *now*, not at
  execution, so it cannot fall through the cracks.
- **[Decision-XPnL-Is-A-Recipe].** `xpnl` is a DecL construct — a way of
  building, not a new thing per se. It returns a plain **multi-group
  :class:`PnL`** (the generic group-ledger tower, fully built at a134),
  replacing today's bare 4-row `stack_marginal_pnls` DataFrame. No `XPnL`
  class, no factory flag — just an instance whose repr / `.groups` /
  exhibits say what it is. Breaking return-type change, pre-1.0, accepted.
- **[Decision-2D-Is-Computation-Only].** Occ **variable** features
  (reinstatements today; future occ swings etc.) are 2-D **even for pnl** —
  the net itself is joint-valued (`net = P − D − h(R) − L + A(·)` is a
  function of the (L, R) joint no 1-D marginal carries) — but 2-D is only
  *how the net is computed*, never *what is shown*. Retro and the agg-basis
  variable features (swing/slide/pc/corridor) stay 1-D. **Consistency over
  grandfathering**: reinstatement / variable-rating programs return the
  consolidated pnl too; their tower is `xpnl`.
- **[Decision-Kappa-Off-The-2D-Net].** Since the sheets compute and show
  kappas, the 2-D consolidated pnl must **retain the joint as its source**
  with the legs as (``is2d``) functions on it — never pre-collapse the net
  to a 1-D pushforward and wrap it (that would discard the atoms the
  conditioning needs). The net GD falls out of the kernel's own exact
  groupby; the scenario columns condition on the 2-D net automatically.
- **[2D-Deferred] — scope note (author, 2026-07-04).** The 2-D routes are
  **not implemented by this plan** — the two decisions above settle the
  design on paper; building it is a follow-up once the details are figured.
  Transitional carve-out, accepted: reinstatement / occ-variable programs
  keep their *current* behavior (the per-atom tower) while GC `pnl`
  consolidates. **[Gross-Anchored-Kappa-Insight]** for that follow-up: the
  variable-feature map ``(G, C) -> (G, C')`` is a per-atom function
  application (no re-gridding, ever), and conditioning on **G** is the
  canonical Portfolio-``exeqa`` idiom — parts (ceded, net) of their total:
  ``E[C'|G=g] + E[G−C'|G=g] = g`` foots automatically, and G-slices are
  **axis-aligned** (a row average: trivial in memory, one-pass even on the
  massive route — may largely dissolve [Massive-Kappa-Second-Sweep] for
  these exhibits). One decision remains there: the walk's ladder anchor —
  gross-anchored ("what does the tower do in a gross event of size g") vs
  the a134 result-anchored states — both exact, different questions.
  **Author refinement (2026-07-04): the anchor choice mostly dissolves** —
  the gross *result* is (almost always, for moral-hazard reasons) strictly
  antitone in gross loss, so its level sets *are* the G-slices relabeled:
  ``E[. | gross result = r] == E[. | G = g(r)]``, same axis-aligned row
  average, only the traversal order differing between the two views. The
  column order is settled by [Decision-Kappa-Outcome-Direction]: always
  outcome-ordered (payoff convention, left tail bad); the G-anchored
  reading is those columns in reverse. The "(FLW)" exceptions are exactly the a134
  switcheroo family — retro collars (flat result segments pool G-values
  into one level set; cells become pooled averages, still exact) and
  hump-shaped features — all living in the deferred 2-D follow-up. Within
  this plan's GC scope the antitonicity holds *exactly*.
- **[Decision-Kappa-Shared-Source-Rule].** Scenario (kappa) percentile
  columns exist **exactly when the ledger shares one source**: `pnl`
  (single source, trivially) and 2-D-sourced `xpnl` towers get them; a
  marginal-stitched GC `xpnl` has no joint, so its ladder stays **marginal**
  with the same documented flag as the massive route. One sentence of docs,
  no special cases.
- **[Decision-Ladder-Column-Names]** (author, 2026-07-04). Rename the
  percentile columns on the PnL frames (PnL-only — Aggregate / Portfolio
  cards use `p0.01`-style headers; harmonizing those is
  [Reporting-Guidelines] territory, not this plan):
  * `summary_df` (the card): `P1` → **`P01`** (zero-padded pair with
    `P99`; `Median` unchanged). Marginal quantiles, as at a134.
  * `stats_df` / `scaled_stats_df` **scenario** columns: `P01…P99` →
    **`κ01 … κ99`** — they are kappas (conditional expectations of each
    row given the anchor at the indicated percentile), and the author
    writes about them as such. "A slight shift, close enough."
  * **The header signals the semantics**: marginal ladders — the stitched
    GC `xpnl`, the massive route, `stack_marginal_pnls` — keep plain
    `P01…P99` headers. A `κ` column *means* conditioning happened; the
    [Decision-Kappa-Shared-Source-Rule] becomes visible on the sheet
    itself, and the a134 "marginal, flagged in the docstring" caveat gains
    an on-sheet signature.
  * Mechanics: `_pct_label` grows the zero-pad + a `κ` twin
    (`f'κ{q*100:02.0f}'`, `.3g` fallback for fractional points);
    `_CARD_COLS` / `_stat_names` updated; the a134 test `_PCOLS` lists
    sweep with it. Unicode `κ` in pandas headers / `qd` output is fine;
    note in the docstring that positional / `df.filter(like=…)` access
    avoids typing the glyph.
  * **[Decision-Kappa-Outcome-Direction] — settled (author, 2026-07-04).**
    κ columns are ordered by **OUTCOME**: P&Ls are **always** in payoff
    sign convention, **left tail bad** — `κ01` = the adverse state, `κ99`
    = the favorable one. This keeps the a134 result-anchored ordering
    exactly: the rename `P01…P99 → κ01…κ99` is a **pure relabel, no
    reversal**. `κp` = the conditional expectation of each row given the
    *result* lands at its p-quantile. (This is deliberately **reversed
    from the usual actuarial loss view** where the right tail is bad —
    the author's earlier "given the gross loss is at the indicated
    percentile" phrasing was a mis-speak; a G-anchored *reading* remains
    available under antitonicity as the same columns traversed in reverse,
    but the columns themselves are outcome-ordered.) The card `P01/P99`
    follow the identical payoff convention (marginal quantiles of each
    row's signed distribution, left tail adverse) — one direction rule
    across every PnL frame, stated once in the docs.
- **[Decision-Total-Step-Stays-Total].** The grand step's index key stays
  the structural `'Total'`; the object's `label` is the **title** (the `qd`
  P&L branch already prints the labeled repr line above the card — no index
  churn). Base step label ← the engine agg's label (e.g. `"Gross Book1"`),
  fallback `'gross'`; cover step labels ← the reins `as` labels (already
  working at a134), fallback `'ceded occ'` / `'ceded agg'`.

## New public names (vetted 2026-07-04 — no collisions)

- `IgnoredDecLClauseWarning` (class, `constants.py`, subclass of
  `UserWarning`, alongside `DefectiveDistributionWarning`).
- `PnL.construction_description` / `PnL.construction_explanation`
  (properties; the established short/long narrative pair — matches the
  `bs_description` / `bs_explanation` convention).
- `accounting_summary_df` (**pended**, [Accounting-Summary-DF] — named now
  so the deferred split card has its canonical name; not built in this
  plan).
- ~~`XPnL`~~ struck at rev 2: xpnl returns a plain multi-group `PnL`.

---

## Phase 1 — [Reins-Economics-On-Agg-Ignore-Warn]

Plain `agg`s accept every reinsurance decoration everywhere; a **pure
aggregate ignores what it cannot use and says so**. The workflow this
unblocks: build the agg standalone (fully decorated with its economics), get
it working, then fold it into a `pnl` / `xpnl` by reference — the
**knowledge-injection** route.

- **Grammar: no change.** The clauses already parse in the shared agg body
  (the a125 engine reuse); the hard stop is one `ValueError` in
  `Underwriter._factory` for `kind == 'agg'` plus `Aggregate(**spec)`
  choking on the feature keys.
- **Factory**: for `kind == 'agg'`, strip the non-loss knowledge keys off a
  **copy** before `Aggregate(**spec)` — `occ_reins_premium`,
  `occ_reins_cede`, `agg_reins_premium`, `agg_reins_cede`,
  `occ_reins_reinst`, `agg_reins_{swing,slide,pc,corridor}` (+ `_layer`),
  `retro_terms` — and emit **one** `IgnoredDecLClauseWarning` naming the
  ignored clauses and the remedy ("fold into a `pnl`/`xpnl` to activate").
  **Mutation-aliasing guard**: `parsed.spec` *is* the knowledge-base entry —
  filter a copy, never pop the stored dict (a pop would silently destroy the
  knowledge the whole feature exists to retain).
- **Knowledge injection works with no new plumbing**: `agg.NAME` refs
  deep-copy the stored spec and merge into the pnl spec
  (`parser.agg_source_ref_agg` → `_pnl_spec`), and
  `_resolve_reins_economics` already resolves the keys at pnl build time.
  `rate 30%` correctly resolves against the *pnl's* premium — which is
  exactly why the bare agg must ignore it (nothing to rate against).
- **Round-trip**: verify `decl_writer` renders the retained clauses from an
  `agg` context (they already round-trip under `pnl`); add the new
  agg-with-economics programs to `src/aggregate/agg/decl-testers.agg`.
- Tests: `test_premium_clause_on_plain_agg_errors` flips raises → warns;
  new: warning names the clauses; the built Aggregate is byte-identical to
  the undecorated build; `pnl X <prem> less agg.NAME` resolves the injected
  economics (deposit / rol / rate / cede, and the feature keys once Phases
  2–3 land).

Phase 1 is self-contained and lands first (it unblocks the demo flow even
before the pnl/xpnl split).

---

## Phase 2 — [PnL-Consolidated]

`build('pnl …')` always returns a **single-group** :class:`PnL`.

- **Source** = the deepest observable the net needs: the final net marginal
  (`reins_density_df`'s deepest net) for GC programs; **the retained 2-D
  joint** (`occ_bivariate`, the same joint the analysis builds) for occ
  variable features, with the legs as `is2d` functions on it —
  [Decision-Kappa-Off-The-2D-Net]: never pre-collapse; the net GD is the
  kernel's own exact groupby of the per-atom result, and the kappa columns
  condition on the 2-D net for free.
- **Legs**: one net-premium consideration leg (constant for GC; stochastic
  for reinstatements — `P − D − h(R)` rides the joint); the engine's net
  loss; the pnl's own expense legs (unchanged — expenses belong to the pnl,
  not the cessions). Commissions received fold into the net premium.
- The GCN promotion path inside `build_gcn_pnl` (occ constants over
  net-of-occ, the agg-cession buy group) is **removed from the pnl route**;
  its logic migrates to Phase 3. `pnl.economics` stays (the resolution
  record); `pnl.analysis` attachment (reinstatement / variable-rating
  drill-down) stays.
- Card reads: `Consideration` 9000 / `Obligation` −(net loss + expenses) /
  `Margin` — the acceptance example below. Kappa columns always available
  (single source).

## Phase 3 — [XPnL-Walk-Recipe] (with [GC-Tower-Marginal-Stitch])

`build('xpnl …')` is a **recipe over the existing kernel**: it assembles the
generic `Group` / `Leg` walk and returns a plain multi-group :class:`PnL` —
nothing beyond what a134 already built:

```
gross        (sell)  premium, gross loss, expenses     ← step label = engine label
<Occ Cover>  (buy)   −ceded premium, +recovery, +commission
<Agg Cover>  (buy)   −ceded premium, +recovery, +commission
Total                grand rows + Impact
```

- **GC programs: marginal-stitched, no joint** ([GC-Tower-Marginal-Stitch]).
  Every row of a guaranteed-cost walk is an affine transform of an exact
  engine marginal (`reins_density_df`: gross, ceded-occ, net-occ, ceded-agg,
  net-net). Build the rows **gd-backed** — the kernel already supports rows
  carrying their own GridDistribution + exact mean/sd without shared atoms
  (the massive one-sweep idiom, `_EvaluatedLeg(gd=…, exact_mean=…,
  exact_sd=…)`). Derived rows (step results, running nets, grand rows) read
  the engine's own marginals — never cross-row sums (which don't exist
  without atoms). Means foot by linearity; SDs/percentiles exact per row.
  **No hybrid**: any stitched tower is all-marginal
  ([Decision-Kappa-Shared-Source-Rule]) even where a sub-chain (the agg
  cover on net-occ) could be per-atom — simplicity and explainability win.
- **2-D programs** (occ variable features): the tower is per-atom over the
  joint — this is today's reinstatement ledger, re-homed under `xpnl`.
  Kappa columns exact.
- The kernel needs a **stitched construction mode** (groups whose rows are
  supplied as gd-backed entries + a declared footing contract) — an internal
  extension of the sweep-backed path, not a new public surface.
- `build_xpnl_stack` (the 4-row onion) **retires**; `stack_marginal_pnls`
  **stays** (generic public no-joint assembler, independently useful).
  Massive-source `xpnl` is out of scope (note in docstring; the massive
  ledger keeps its current pnl route).
- Labels per [Decision-Total-Step-Stays-Total].

## Phase 4 — [Construction-Introspection]

`construction_description` (one paragraph: route, source, group count) and
`construction_explanation` (the full story) on every built P&L, **recorded by
the builders at construction** — they alone know the why. The explanation
states, in order: the engine and its clauses; the economics resolution
(`deposit 2000 → pc_occ = 2000`; `rate 30% × 12000 → 3600`); which source
each row/step reads (`gross: reins_density_df['p_agg_gross']`; `Agg Cover
recovery: f(x) per atom on net-occ`; …); the booking signs; whether the
ladder is scenario or marginal and **why**; any clauses ignored (shares its
wording with the `IgnoredDecLClauseWarning` text — one source of truth). The
closing block is the **executable replay**: the literal
`PnL(name=…, source=…, groups=[Group(…), …])` call that reproduces the
object (the kernel already retains `_group_specs` and the source reference).
Hand-built kernel P&Ls get a minimal generic narrative so the properties are
never absent.

## Phase 5 — [Docs-And-Education] + test migration

- "Reading the P&L sheets" (the `_pnl.py` module docstring, a134) gains the
  pnl-vs-xpnl paragraph (position vs walk) and the kappa shared-source rule;
  builder docstrings rewritten in lockstep. Author rebuilds docs manually
  (standing rule).
- **Test migration** (the second re-migration for the reins suites —
  consistency over grandfathering, priced in):
  - `test_pnl.py` GCN tests + `test_pnl_ceded_premium.py` promotion tests →
    split: consolidated-pnl asserts (net premium, flat card, no tower) +
    xpnl tower asserts (the a134 3-level keys survive nearly verbatim).
  - `test_reinstatement_decl.py` / `test_variable_rating_decl.py` exhibit
    tests → pnl consolidated + xpnl tower.
  - `test_pnl_engine_source.py::test_xpnl_*` → multi-group `PnL` return +
    tower keys.
  - New: Phase-1 warn/inject suite; stitched-tower correctness (each row's
    mean/sd equals its engine marginal; EX foots; ladder marginal +
    flagged); 2-D pnl net equals the analysis's net pushforward;
    `construction_explanation` smoke (mentions route, source, ignored
    clauses; replay block parses).
- **Acceptance case** (pins this session's motivating example — the `Cat`
  program: 12000 premium, occ 75% po 2750 xs 250 deposit 2000 "Occ Cover",
  agg 80% po 1000 xs 7000 deposit 1000 "Agg Cover", three expense groups):
  - `pnl`: card `Consideration = 9000` / `Obligation = −(net-net loss +
    expenses)` / `Margin` foots. **No** "gross premium next to higher loss".
  - `xpnl`: steps `Gross Book1` / `Occ Cover` / `Agg Cover` / `Total`; occ
    recovery row mean = the engine's ceded-occ marginal mean; running nets =
    the engine's net marginals; EX column foots; ladder marginal, flagged.

---

## Housekeeping on execution

- **Two bumps**: Phase 1 as its own version (self-contained, independently
  valuable); Phases 2–5 as one version. CHANGELOG section each.
- `dev/TODO.md`: mark this plan; **[Accounting-Summary-DF]** already pended
  there (added 2026-07-04 at plan rev 2, ahead of execution); reconcile
  **[Massive-Kappa-Second-Sweep]** (unchanged — still the massive pnl
  route); note `build_xpnl_stack` retirement.
- `dev/FEATURES.csv`: PnL cell notes for `summary_df`/`stats_df`
  (consolidated vs exploded by group count), `construction_*` additions;
  re-run `dev/regen_features.py`.
- `decl-testers.agg`: agg-with-economics + xpnl programs (must round-trip).
- Gate: full fast suite per phase; `-m 'slow or not slow'` at each bump.

## Flags — resolved by the author (2026-07-04, rev 2)

1. **[Flag-Net-Premium-Leg-Label]** — **OK as proposed**: default
   `'net premium'`; a declared label qualifies as `'<label> (net)'` (the
   `'loss (net occ)'` precedent); untouched when nothing is ceded.
2. **[Flag-XPnL-Class-Home]** — **resolved the other way: no class.** xpnl
   is a way of building, not a new thing — it returns a plain multi-group
   `PnL` instance ([Decision-XPnL-Is-A-Recipe]).
3. **[Flag-Two-Bump-Split]** — **OK**: Phase 1 lands as its own bump.
