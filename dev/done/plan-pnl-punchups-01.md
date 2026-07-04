# [PnL-Punchups-01] — kappa scenario percentiles, summary card, view labels

**Status: READY (rev 2) — all decisions settled by the author 2026-07-04; no
coding yet, execute on "go".**
Follows [PnL-Generic-Final] (a129–a131, `dev/done/plan-yapnl.md`) and
[PnL-Stats-View-MultiIndex] (landed under a132, no bump). Executes as **one
version bump** covering all phases.

## Settled decisions (author, 2026-07-04)

- **[Decision-Thin-Slice-Rule] = always exact.** Kappa cells condition on the
  exact result-value slice — no fallback band, no co-TVaR variant. This works
  cleanly for independent / FFT-constructed sources; where it looks jumpy
  (switcheroo — non-monotone results) that is an **education opportunity,
  documented, not engineered away**.
- **[Decision-Stats-Step-Level] = three-level `(Step, View, Line)` on
  towers** (multi-group); single-group stays two-level `(View, Line)`. Kills
  the a132 qualified-string lines (`'base total'`, `'Net through cover'`) —
  the qualifier becomes the level.
- **[Decision-Summary-Single-Index] = flat card single / `(Step, View)`
  blocks for towers.** Exact structural mirror of the existing house pair:
  `Aggregate.summary_df` (flat fixed rows) vs `Portfolio.summary_df`
  (`(unit, X)` block per unit + a `total` block). Verified against
  `_portfolio.py`.
- **[Decision-View-Labels-Scope] = defaults only.** `Consideration /
  Obligation / Margin` ship as-is; **document the serve-time rename recipe**
  (`df.rename({'Obligation': 'Loss & LAE'}, level='View')`) rather than
  building a surface. Revisit later if a real deck demands persistence; the
  `label_map['views']` storage home exists, so retrofitting costs nothing
  extra. No DecL changes in this plan (⇒ no grammar / snapshot /
  `decl-testers.agg` work).
- **[Decision-Card-Percentiles-Stay-Marginal].** The summary card posts
  **marginal quantiles**, not kappa cells: the card answers "how big is each
  total" (range feel) and is thereby *deliberately different in kind* from
  the stats sheet (alignment/footing). Two consequences, both chosen:
  1. Card percentile cells **do not foot** down the card (marginal quantiles
     never add) — a **teaching moment**, stated plainly in the docstring and
     docs, twin to the switcheroo note.
  2. The card is its own small computation, not a column slice of
     `stats_df` (retracts the draft's one-source-of-truth claim for the
     ladder columns; the EX/Scaled/SD/CV/Skew cells still read the same
     grand rows).

No new public names are introduced anywhere in this plan (`summary_df` /
`stats_df` / `scaled_stats_df` keep their names with changed semantics), so
there is no name-collision vetting to do.

---

## Phase 1 — [Kappa-Scenario-Percentiles]

### The rule

Percentile columns are **states, not per-row quantiles**, anchored on the
**grand result** (the bottom line — never per-group results). For ladder
point `q`:

```
cell(row, Pq) = E[ row | result == x_q ],   x_q = grand-result gd.q(q)
```

This is the library's kappa function E[Xᵢ | X = x] — Portfolio's `exeqa_*`
columns (`_portfolio_common.py`) applied to the ledger. Properties:

- **Columns foot exactly** (linearity of conditional expectation): legs →
  totals → result adds down the sheet to `x_q` in every column. The
  Margin/Total cell is automatically its own marginal VaR
  (E[result | result = x] = x) — no special case.
- **Direction is uniform**: a column is one state ordered by how good it is
  for the holder; small p is bad, full stop. The motivating retro bug
  (premium P1 low next to loss P1 high — an impossible state) is fixed: the
  retro premium correctly shows *high* in the bad columns.
- Derived rows (group totals, results, running nets, total impact) get the
  identical conditioning and stay mutually consistent automatically.

`EX / SD / CV / Skew` are row properties and **stay marginal**. Only the
ladder columns change meaning. Per-row marginal quantiles remain one line
away via `density_df[row].q(p)` — no parallel exhibit.

### Column naming

Keep the `P1 … P99` headers — the column *is* the result's Pq state. The
docstring says plainly: cells are conditional means given the result lands
at its q-quantile; only Margin rows are literal quantiles of themselves.

### Implementation (atoms route)

- The ladder computation moves from per-row `stat_vector()` to a
  **PnL-level** pass (conditioning needs the grand-result values shared
  across rows): compute `x_q` per ladder point off the grand-result GD, form
  the exact atom slice `result == x_q`, take probability-weighted means of
  each row's signed values over the slice. Moment columns keep the existing
  per-row path.
- 1-D source: every leg and the result are functions on the same atoms —
  exact groupby by result value; where the result is monotone in the source
  the cell is pointwise deterministic ("evaluate every leg at the state").
- 2-D joint: same exact groupby over the joint atoms — genuine conditional
  spread, still exact.
- Non-monotone results (author's "switcheroo": slides, swings, humps): the
  conditional mean over the level set is exact and well-defined; the
  *interpretation* is subtler and **explicitly deferred** — docstring +
  docs education note, nothing engineered.
- Degenerate case: a constant grand result (fully hedged) makes the
  conditioning event everything ⇒ every cell = its EX. Well-defined; note
  in docstring.
- `scaled_stats_df`: conditional means scale linearly — the existing
  divide-through logic is untouched.
- `stack_marginal_pnls`: stays marginal by construction (independent
  perspectives share no joint, there is nothing to condition on) —
  docstring note only.
- **Massive one-sweep route: deferred.** Conditioning needs the joint per
  atom and the result quantiles before indicator-weighted means can
  accumulate — a second sweep. The massive `stats_df` keeps marginal
  ladders with a docstring flag; tracked as **[Massive-Kappa-Second-Sweep]**
  in `dev/TODO.md`.
- Fix the `PERCENTILE_LADDER` module comment (the card no longer "reads
  P1/Median/P99 off" the stats ladder — decoupled by
  [Decision-Card-Percentiles-Stay-Marginal]).

### Tests (phase gate: affected suites green)

- Footing: every ladder column sums legs → totals → result to the
  Margin/Total cell within `VALIDATION_NOISE`; both roles; 1-D and 2-D
  sources.
- Margin rows equal the grand result's marginal `gd.q(p)` exactly.
- Buy/sell reflection: the bought position's scenario columns are the
  negated, p↔(1−p)-reflected columns of the sold one.
- Retro acceptance case (loss-sensitive premium): the premium cell
  **increases** toward the bad columns — the motivating bug pinned forever.
- 1-D monotone case: cells equal direct evaluation of each leg at the state.
- Switcheroo smoke: non-monotone result (e.g. a slide hump) — cells exact
  level-set means, footing still holds.
- Massive route: ladder unchanged (marginal), flag present.

---

## Phase 2 — [Stats-Tower-Step-Level]

Multi-group `stats_df` / `scaled_stats_df` index becomes three-level
`(Step, View, Line)`; content is unchanged (the tower is already fully in
the frame — per-step legs, totals, results, running nets, grand rows):

```
('gross',     'Consideration', 'premium')      # legs, as declared
('gross',     'Obligation',    'loss')
('gross',     'Obligation',    'Total')        # step total (when >1 leg)
('gross',     'Margin',        'Total')        # the step result
('ceded occ', 'Consideration', 'ceded premium')
('ceded occ', 'Margin',        'Total')
('ceded occ', 'Margin',        'Net')          # running net through step
('Total',     'Consideration', 'Total')        # grand rows close the sheet
('Total',     'Obligation',    'Total')
('Total',     'Margin',        'Total')
('Total',     'Margin',        'Impact')       # grand result vs first step
```

- Step level values = group labels, in ledger order; the grand block's step
  is `'Total'`.
- Qualified strings die: `'<group> total'` → `(step, view, 'Total')`;
  `'Net through <g>'` → `(step, 'Margin', 'Net')`; `'<g> result'` →
  `(step, 'Margin', 'Total')`; `'total impact'` → `('Total', 'Margin',
  'Impact')`.
- Single-group frames stay two-level `(View, Line)` exactly as landed at
  a132 (a degenerate step level is noise; mirrors flat-Aggregate vs
  block-Portfolio).
- The **flat plan labels stay the canonical row keys** everywhere else
  (`_rows`, `density_df`, `validation_df`, sweep result keys) — this level
  restructure is presentation-only, same as a132.
- `_view_index()` grows the multi-group branch; `_VIEW_DEFAULTS` unchanged.

Tests: rewrite `test_stats_df_multiindex_multigroup` (a132) to the 3-level
tuples; assert single-group shape unchanged; `scaled_stats_df` index equals
`stats_df` index.

---

## Phase 3 — [Summary-Fixed-Card]

### Single P&L: the fixed card

- **Rows (fixed, flat index):** `['Consideration', 'Obligation', 'Margin']`
  — read from the grand references (`_grand_cons` / `_grand_obl` /
  `_grand_result`), which always exist regardless of leg count. The card
  never varies with the ledger.
- **Columns:** `EX / Scaled / SD / CV / Skew / P1 / Median / P99`
  (unchanged set). Percentiles are **marginal quantiles of each card row's
  own distribution** ([Decision-Card-Percentiles-Stay-Marginal]) — note the
  grand consideration / obligation rows need their GDs, which the atoms
  route already has; see massive note below.
- With scale = premium the `Scaled` column reads as a combined-ratio
  decomposition: `1.00 / −(loss & expense ratio) / margin ratio`.

### Tower: per-step card stack, `(Step, View)`

One block per step + a closing `Total` block — the Portfolio mirror:

```
('gross',     'Consideration' | 'Obligation' | 'Margin')
('ceded occ', 'Consideration' | 'Obligation' | 'Margin' | 'Net')
('Total',     'Consideration' | 'Obligation' | 'Margin' | 'Impact')
```

`Margin` per step = the group result (step delta); `Net` = running net
(omitted on the first step, where net = margin); `Impact` = grand result −
first step's result. Rows scale with steps, never with legs — the
fixed-shape contract.

### Notes

- **Massive route:** the single-group grand cons/obl references are
  currently mean-only `SimpleNamespace` stubs (no GD ⇒ no marginal
  percentiles). Options: add the two grand rows as sweep entries (one sweep,
  two more keys — cheap), or post `NaN` percentiles on those cards. Pick at
  execution; lean **add the sweep entries** (a card with holes isn't a
  card).
- `qd(pnl)` keeps printing `summary_df` — now genuinely compact on towers.
- `result_name` still names the flat ledger key only; card and stats display
  `Margin` / `'Total'` regardless. Docstring note.
- Docstring states plainly: card percentiles are marginal and **do not
  foot** (teaching moment); the footing sheet is `stats_df`.

### Test migration (the real cost of this phase)

Dozens of asserts across `test_pnl.py`, `test_pnl_expenses.py`,
`test_pnl_ceded_premium.py`, `test_create_pnl.py`, `test_decl_labels.py`,
`test_pnl_engine_source.py` read **leg rows** off `summary_df.loc['loss',
'EX']`. Migration rule: leg-level asserts move to `stats_df` with
`(View, Line)` / `(Step, View, Line)` keys; card-level asserts use the new
fixed keys. Mechanical, touch-every-file. New tests: fixed 3-row single
card; tower block shape; Scaled economics; card percentile = the row GD's
marginal quantile; card/stats percentile cells *differ* on a dependent-leg
case (pins the two meanings apart).

---

## Phase 4 — [Docs-And-Education]

- Docstrings updated in lockstep (Phases 1–3 carry their own).
- A short docs subsection "Reading the P&L sheets": card vs stats (range vs
  alignment), scenario columns as states, why marginal percentiles don't
  add, the switcheroo caveat, the serve-time view-rename recipe
  (`df.rename(..., level='View')`).
- Author rebuilds docs manually outside the loop (standing rule); `.rst`
  edits kept in lockstep, rebuild flagged pending.

---

## Housekeeping on execution

- One version bump (`1.0.0a1xx`) + CHANGELOG section covering all phases;
  `dev/TODO.md`: mark this plan's entry, add
  **[Massive-Kappa-Second-Sweep]**; move this plan to `dev/done/` on
  landing.
- `dev/FEATURES.csv` re-audit (`dev/regen_features.py`) — `summary_df` /
  `stats_df` semantics change on PnL.
- No DecL / grammar / snapshot / `decl-testers.agg` work (view labels
  settled as defaults-only).
- Gate: full fast suite green per phase; `-m 'slow or not slow'` full gate
  at the close.
