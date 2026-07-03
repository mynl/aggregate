# Plan — [PnL-Generic-Final] (a.k.a. yapnl — yet another PnL plan, the last one)

Status: **DONE** — landed `1.0.0a129`–`1.0.0a131` (2026-07-03), all green
(fast suite + slow gate + the `dev/FEATURES.csv` audit). Execution notes:
phases [Kernel-Group-Ledger] + [Builders-Plain-GCN-Port] (and the `xpnl`
slice of [Xpnl-Onion-2D]) landed together as `a129` — the single shared
`PnL` class couples them, so separate green checkpoints were not achievable;
[Builders-Variable-Features] = `a130`; the massive one-sweep route +
closeout = `a131`. The ledger row template is one shared plan
(`_ledger_plan`) consumed by both evaluation routes. Judgment calls flagged
for author review in the run summary: the occ guaranteed-cost booking (occ
ceded premium as an obligation constant in the retained sell group), the
default group / leg label vocabulary (`ceded occ` / `ceded agg`,
`<group> premium / recovery / commission`), `xpnl` returning the bare
DataFrame, `make_pnl(net=)` retired, and loss-basis expense degrading to a
constant over a net marginal.

Original status: **READY** (2026-07-03, revised round 4 after review). All
design decisions settled with the author; the decision log at the end records
each one (including the one vetoed item). This is the last structural step
before beta: make `PnL` *actually* generic, consolidate the four divergent
construction paths, and pin down the 2-D case.

Prerequisite landed: **[DecL-Labels-Everywhere]** shipped at `1.0.0a128`
(`dev/done/plan-labels.md`). `PnL` is a `LabeledMixin`
(`_labeled.py`) — the new kernel keeps that: `PnL(LabeledMixin)`, labels
initialized via `self._init_labels(display_label=…, label_map=…)`, giving
`display_name` / `labels` / `use_labels` / `renamer` for free. **Leg labels
are the ledger row keys directly** (the a124 convention — for legs, label =
handle; there is no separate handle/`renamer` indirection at the leg level,
and ledger ordering is declaration order, never label text).

## The model

A P&L is a **probability space plus named accounting functionals**:

* a **source** — the stochastic generator `g` (1-D or one shared 2-D joint);
* an ordered list of **groups**, each a mini-P&L: a label, a `role`
  (`'sell'` / `'buy'`), and ordered lists of consideration / obligation
  **legs** (`Leg(label, func, ...)`) — every leg an actual cash flow;
* everything else **derived**: per-leg exact GridDistributions, per-group
  totals and result, running nets, step deltas (the group result *is* the
  step delta), the grand total, signed additive exhibits.

All the domain magic happens at the **caller** level (defining the source and
the leg functions); inside `PnL` everything is generic. DecL (`pnl` / `xpnl`)
stays deliberately domain-specific — one caller among many, translating
insurance programs into sources + groups. Insurance is a caller, not a
special case.

## The API (kernel, `_pnl.py`)

```python
Leg(label, func, bs=0, is2d=False)
    # label first — reads like the exhibit row it becomes
    # func: a constant, a vectorized f(x) (1-D), or f(l, r) (is2d=True)
    # bs=0 -> exact irregular GD (group atoms by value, sum probs)
    # bs>0 -> mean-preserving rebucket onto a regular grid (audited)

Group(label, role, consideration=[Leg...], obligation=[Leg...])
    # role in {'sell', 'buy'} books the group in the holder's ledger:
    # sell -> +consideration, -obligation; buy -> the contra. A cession is
    # declared conceptually (consideration = ceded premium, obligation =
    # recovery + commission) and the buy role books it -premium/+recovery.

PnL(*, name, source, groups, scale=None, result_name='result',
    display_label=None)
PnL(*, name, role, source, consideration, obligation, scale=None,
    result_name='result', display_label=None)     # sugar: exactly one Group
pnl_a + pnl_b       # same source required; concatenates the group ledgers
```

* `source`: `GridDistribution` | `Aggregate` | `(values, probs)` [1-D];
  `BivariateDistribution` | `MassiveBivariateDistribution` [2-D] (the exact
  class names in `bivariate.py`). No `probs=` override (transition artifact,
  removed).
* Component shorthand: `{label: func}` ≡ `[Leg(label, func), ...]` in
  declaration order — the simple case reads as simply as before.
* `create_pnl` / `create_pnl_tower` / `PnLTower` **retire** (breaking;
  one canonical name). `create_pnl` existed to morph an `Aggregate` into a
  P&L; that role survives as the `Aggregate.make_pnl` sugar.

### Evaluation invariants

* **Per-atom, then group.** Every leg func evaluates over the source atoms;
  totals, results, running nets are per-atom partial sums of the signed
  legs. The result is never arithmetic on marginal GDs — the covariance
  rides for free ("means add, SDs don't" holds on-sheet automatically).
* **[One-2D-Source]**: any number of `is2d` legs, all reading the single
  shared joint; `is2d` over a 1-D source is an error (and a 2-D source with
  1-D legs is fine — they read axis 0). Only one latent dimension exists;
  the "one 2-D feature" rule is enforced by the builders, where features are
  known (two occ-level features → a clear error, the n-d short-circuit).
* **Exact by default**: `bs=0` legs are exact irregular GDs (GDs need no
  equal spacing — they never touch the FFT machinery). `bs>0` legs rebucket
  via the shared pushforward machinery and feed `validation_df`.
* **Massive source**: `bs` required per leg; the whole leg set + subtotal
  functions evaluate in **one** `MassiveBivariateDistribution.pushforward`
  band sweep (its `bs` parameter takes an array matched to the function dict
  in iteration order — per-leg `bs` needs no API change), each derived row
  pushed as its own signed-sum function (never a sum of bucketed legs) — the
  a126 dict-pushforward contract, reused.

### Exhibits — one row ledger, three views

Rows = the ledger, columns = metrics (ledgers grow long). Per group:

```
consid_1 … consid_n
total consideration        (only if n > 1)
obl_1 … obl_m
total obligation           (only if m > 1)
result                     (the group's signed net = its step delta)
```

For a multi-group ledger, a running-net row (`net through <group>`) follows
each group after the first, and the sheet closes with the grand rows: total
consideration, total obligation, **result = sum of the group results**, and
the total impact (overall result vs the first group's result). All derived —
with signed groups there is nothing left to declare, so there is no
`Subtotal` / `Total` marker anywhere.

* **`summary_df`** — the headline: ledger rows × (EX, Scaled, SD, CV, Skew,
  P1, Median, P99). Signed cash flows; the EX column **adds down the sheet**
  to the result rows.
* **`stats_df`** — same rows, full metric set (EX, SD, CV, Skew + the
  `PERCENTILE_LADDER`), currency units.
* **`scaled_stats_df`** — **the stats of `X / scale`**: EX / SD /
  percentiles divide by the committed scale; CV and Skew are scale-invariant
  and pass through unchanged. Every cell defined; the `_UNSCALABLE_STATS`
  NaN machinery is deleted.
* **`density_df`** — ordered `{row label: GridDistribution}` (legs, group
  results, grand result); no stapled shared grid.
* **`validation_df`** — Est-vs-EX audit for every `bs>0` leg (linear scheme
  matches means, so it reads *very* close — but it is visible). No reaching
  through to the source's own validation, which may not exist.
* **`margin_df` retires** — its metrics-as-rows orientation is superseded by
  `stats_df` (rows = ledger, columns = metrics). One canonical frame per
  view; no alias kept.
* Signed orientation throughout ([Signed-Exhibits-Role-Orientation]): each
  row's GD is built on the signed values, so `Pq(−X) = −P(1−q)(X)` puts the
  adverse tail where the reader expects for both roles, automatically.
* `mean / sd / cv / skew / q / var / cdf / sf / prob_loss / plot` delegate
  to the grand result GD; `evaluate()` (the Cherny–Madan panel) survives
  with its current constraint (single obligation over a regular-`bs`
  source), reading the grand obligation total.

### What survives of the tower

Only the **marginal** (no-joint) case: plain guaranteed-cost perspectives,
each a separate `PnL` over its own `reins_density_df` marginal — no shared
atoms, so no per-atom groups; means add, SDs / percentiles are per-column.
A small assembler covers it (working name `stack_marginal_pnls` — a module
function producing the perspective × stats frame + impact deltas; vet the
name at execution). `PnLTower` the class retires.

## The builders (`_pnl_builders.py`, new)

All insurance semantics live here (and in the DecL grammar); the kernel
never sees the words gross / ceded / net. Moved out of `_pnl.py`:
`gcn_tower_from_aggregate` (rebuilt as the gcn ledger/stack builders),
`resolve_expense`, `_resolve_expense_split`, `_gcn_magnitudes`, the
expense-group normalizers. `Underwriter._snapshot_pnl` becomes a thin
dispatcher; `Aggregate.make_pnl` stays as object sugar delegating here.

Label plumbing (fixes the observed bugs):

* premium `as` label → the consideration leg label (works today, keep);
* engine `as` label → the **loss leg label** (today stripped and discarded —
  `spec.pop('engine_display_label')`; default `'loss'` when absent);
* expense-group `as` labels → one obligation leg per group in **every**
  path; `_resolve_expense_split` is the only expense resolver feeding legs —
  loss-basis expense is stochastic `rate·x` everywhere (the a123/a125
  scalar carve-out closes). `resolve_expense` (deterministic total) may
  survive privately for scalar needs but never feeds a leg;
* reins-layer `as` labels → cession group / leg labels.

### `pnl` vs `xpnl` routing

Not a design decision — the direct consequence of [Group-Ledger] +
[Cash-Flow-Legs-Default]: the builders always book the parts into `Group`s,
so the `PnL` a program produces *is* its group ledger. (A separately flagged
"[Pnl-Returns-The-Ledger]" refinement was vetoed as redundant — see the
decision log.)

* **`pnl`** returns the `PnL` built per the source-selection table below —
  for a program with cessions that is a multi-group ledger over the deepest
  source the cash flows are jointly measurable on. The grand result *is*
  the net, so nothing is lost vs today's collapsed face; the cession groups
  show as labeled rows.
* **`xpnl`** returns the **marginal perspective stack** (across
  `reins_density_df` marginals: gross / net-occ / net-agg, each perspective
  its own one-group `PnL`, assembled by `stack_marginal_pnls`). Requires
  reinsurance economics; `xpnl` over a plain engine or a `port` stays an
  error.
* The analyses (`ReinstatementAnalysis` / `VariableRatingAnalysis`) demote
  to **builders + domain extras** (terms objects, ceder / recovery maps,
  `validation_df`, `tail_df`, `plot`); their bespoke `summary_df` /
  `stats_df` / `distributions` / `_leg_functions` retire. `PnL._waterfall`
  (the type union) dies; the domain object rides as `PnL.analysis` for
  drill-down only — no exhibit is forwarded from it.

### Source selection (the 2-D story, spelled out)

| program | source | ledger |
|---|---|---|
| gross book (no reins) | engine 1-D density | one `sell` group |
| retro | gross 1-D | one `sell` group, premium leg = `retro.phi` |
| agg-only reins / agg-level feature (swing, slide, pc, corridor) | **gross** 1-D marginal | multi-group: gross `sell` + cession `buy` group(s), all legs `f(x)` — exact per-atom waterfall |
| occ guaranteed-cost (± agg) | net-of-occ (or net) 1-D marginal | per-atom groups for everything measurable there (agg cession = `f(net-occ x)`); deterministic premiums as constants; the gross perspective via `xpnl` |
| occ-level feature (reinstatements; future occ swing) | the **(L, R) joint** | multi-group with `is2d` legs; a subsequent agg cover is `g(l − A(r))` — same joint, no new dimension |
| two occ-level features | — | **error**: needs a 3-D joint; not supported |

## Example usage — how DecL maps to the machinery

Domain-specific illustration (the builders' output, sketched as kernel
calls). Engine severity shorthand below: `SEV = sev 10.808 * lognorm 1.75`
(mean-50 lognormal, σ = 1.75, 10.808 = 50·exp(−σ²/2)); `FREQ = mixed gamma
0.25`; all wrapped engines are complete `agg` forms per a125.

### 1 — plain gross book (the acceptance pair, first program)

```
pnl GrossPNL as "Gross Book PNL"
    inherit premium as "Gross Premium"
    less agg A as "Gross Loss" 10000 premium at 85% lr sev lognorm 50 cv 3 poisson
    less 5% loss expense as LAE
         100 fixed expense and 10% premium expense as "Fixed & Acq Exp"
```

```python
PnL(name='GrossPNL', display_label='Gross Book PNL', source=engine_A,
    role='sell',                                   # single-group sugar
    consideration=[Leg('Gross Premium', 10000.0)], # inherit -> exp_premium
    obligation=[Leg('Gross Loss', lambda x: x),    # engine `as` label
                Leg('LAE', lambda x: 0.05 * x),    # stochastic, SD > 0
                Leg('Fixed & Acq Exp', 100 + 0.10 * 10000)])
```

`summary_df` rows: Gross Premium, Gross Loss, LAE, Fixed & Acq Exp, total
obligation, result — every row a signed cash flow, EX column footing to the
result.

### 2 — retro rating (the acceptance pair, second program)

```
pnl RetroPNL as "Retro Rated Gross PNL"
    retro basic 2000 lcm 1.1 min 8000 max 14000 premium as "Retro Premium"
    less agg A as "Gross Loss" 10000 premium at 85% lr sev lognorm 50 cv 3 poisson
    less 5% loss expense as LAE
         100 fixed expense as "Fixed Exp" 10% premium expense as "Acq Exp"
```

```python
retro = RetroTerms(basic=2000, lcm=1.1, min=8000, max=14000)
PnL(name='RetroPNL', display_label='Retro Rated Gross PNL', source=engine_A,
    role='sell',
    consideration=[Leg('Retro Premium', retro.phi)],   # stochastic premium
    obligation=[Leg('Gross Loss', lambda x: x),
                Leg('LAE', lambda x: 0.05 * x),        # stays stochastic
                Leg('Fixed Exp', 100.0),
                Leg('Acq Exp', 0.10 * base_premium)])
```

**Same shape as example 1** — that identity is the acceptance test. Today
this program morphs to hard-coded `net premium` / `net loss` / `net expense`
with LAE collapsed to a deterministic scalar.

### 3 — aggregate cover with ceded premium and commission (two-group ledger)

```
pnl AggFixedCostwCeded 10000 premium
    less agg B 10000 premium at 75% lr 10000 xs 0 SEV FREQ
         aggregate net of 5000 xs 10000 deposit 500 cede 10%
    less 20% premium expense
```

```python
g = agg_ceder                        # recovery map of the 5000 xs 10000 layer
PnL(name='AggFixedCostwCeded', source=gross_marginal,   # 1-D gross density
    groups=[
      Group('Gross', 'sell',
            consideration=[Leg('Premium', 10000.0)],
            obligation=[Leg('Gross Loss', lambda x: x),
                        Leg('Premium Expense', 0.20 * 10000)]),
      Group('Agg XL', 'buy',
            consideration=[Leg('Agg Ceded Premium', 500.0)],
            obligation=[Leg('Agg Recovery', g),
                        Leg('Ceding Commission', 50.0)]),
    ])
```

The payoff row set: Gross legs, gross result; Agg XL legs booked contra
(−500 premium, +recovery, +commission), the group result = the cession's
step delta; running net; grand result = net position — all per-atom exact,
covariance included. This *is* the old Gross/Ceded/Net waterfall, as rows.

### 4 — occurrence reinstatements (the 2-D case)

```
pnl OccReinstatements 10000 premium
    less agg C 10000 premium at 75% lr 10000 xs 0 SEV
         occurrence net of 24500 xs 500 deposit 1000 reinstatements [1] FREQ
    less 20% premium expense
```

```python
src = occ_joint                       # (L, R): gross loss x unlimited recovery
A, h, D = terms.recovery, terms.reinstatement_premium, terms.deposit
PnL(name='OccReinstatements', source=src,
    groups=[
      Group('Gross', 'sell',
            consideration=[Leg('Premium', 10000.0)],
            obligation=[Leg('Gross Loss', lambda l, r: l, is2d=True),
                        Leg('Premium Expense', 2000.0)]),
      Group('Occ XL', 'buy',
            consideration=[Leg('Ceded Premium',
                               lambda l, r: D + h(r), is2d=True)],
            obligation=[Leg('Occ Recovery', lambda l, r: A(r), is2d=True)]),
    ])
```

Multiple `is2d` legs, one shared joint ([One-2D-Source]). A subsequent
`aggregate net of …` adds a third `buy` group with legs
`lambda l, r: g(max(l - A(r), 0))` — same joint, still 2-D.

### 5 — the four agg-level variable-rating features (1-D throughout)

All decorate the agg layer of example 3's engine; each swaps exactly one leg
for a `terms.phi`-driven map (1-D functions of gross `x`; `g` = layer
recovery map):

| DecL feature clause | group | changed leg |
|---|---|---|
| `swing basic 0 lcm 1.1 min 300 max 1000` | `Group('Agg XL (swing)', 'buy')` | consideration `Leg('Swing Premium', lambda x: swing.phi(g(x)))` — stochastic ceded premium |
| `deposit 1500 slide 45% at 60% and 25% at 70% and 19% at 80%` | `Group('Agg XL (slide)', 'buy')` | obligation gains `Leg('Sliding Commission', slide_map)` (deposit stays a constant consideration; `slide_map` = `slide.phi` over the layer loss ratio) |
| `deposit 1500 pc 50% after 10%` | `Group('Agg XL (pc)', 'buy')` | obligation gains `Leg('Profit Commission', pc_map)` (`pc.phi` over the layer loss ratio) |
| `deposit 500 corridor 50% po 30% xs 20%` | `Group('Agg XL (corridor)', 'buy')` | obligation's recovery becomes `Leg('Agg Recovery (corridor)', corridor_map)` — the corridor-adjusted ceder |

With retro (example 2) and reinstatements (example 4) that covers all six
variable features. Note the uniformity: **a feature never changes the
machinery — it changes one leg's function.**

### 6 — `xpnl` (guaranteed-cost onion across marginals)

```
xpnl Onion 10000 premium
    less agg D 10000 premium at 75% lr 10000 xs 0 SEV
         occurrence net of 24500 xs 500 deposit 1000 FREQ
         aggregate net of 5000 xs 10000 deposit 200
    less 20% premium expense
```

No joint exists (guaranteed cost — the cessions are not functions of one
observable), so this is the **marginal stack**: one-group `PnL`s over
`p_agg_gross` / `p_agg_ceded_occ` / `p_agg_net_occ` / `p_agg_ceded` /
`p_agg_net`, assembled by `stack_marginal_pnls` into the perspective × stats
frame with impact deltas (means add; SDs / percentiles per column). The
collapsed `pnl` form of the same program is a per-atom ledger over the
net-of-occ marginal: constants for the ceded premiums, the agg cession as a
real `f(x)` group, the occ cession visible only through `xpnl`.

## Phases

1. **[Kernel-Group-Ledger]** — `Leg` / `Group` / the new `PnL` signature +
   `+` composition; `{label: func}` shorthand; [One-2D-Source] validation;
   per-leg `bs` (exact / rebucket); signed exhibits + percentile
   orientation; the three-view exhibit trio (`summary_df` / `stats_df` /
   `scaled_stats_df`) + `validation_df`; `margin_df` retired; keep
   `PnL(LabeledMixin)` + `_init_labels`; retire `create_pnl` /
   `create_pnl_tower` / `PnLTower` (keep `stack_marginal_pnls`); move the
   insurance helpers out to `_pnl_builders.py` (relocation only).
2. **[Builders-Plain-GCN-Port]** — `build_plain_pnl`, the gcn ledger /
   marginal-stack builders, `build_port_pnl`: cash-flow legs, engine-label →
   loss-leg label, expense groups uniform (port expenses un-NYI), the
   `pnl` / `xpnl` routing above; `_snapshot_pnl` thinned; `make_pnl`
   delegates.
3. **[Builders-Variable-Features]** — the six features as one-changed-leg
   builders (examples 2, 4, 5); acceptance pair passes; analyses demoted
   (bespoke exhibits deleted, domain extras kept); `_waterfall` dies;
   `.analysis` reference-only.
4. **[Xpnl-Onion-2D]** — `xpnl` = marginal stack; reinstatement /
   subsequent-agg joint ledgers; second-occ-feature hard error;
   massive-source one-sweep smoke.
5. **[Sweep-Tests-Docs]** — `test_decl.agg` sync (append the example
   programs above), `dev/FEATURES.csv` + introspection cross-check,
   CHANGELOG + version bump, `.rst`/`.qmd` lockstep (author rebuilds),
   `dev/TODO.md`.

**Each phase independently green** — a phase that breaks a test updates that
test *within the phase* (Phase 1 rewrites `tests/test_create_pnl.py` and the
`create_pnl` call sites in `test_pnl_*.py` when it retires `create_pnl`;
Phase 3 rebases the analyses' tests when it demotes them). Phase 5 is the
housekeeping sweep, not deferred test repair. Version bump per house rule.

## Testing

* **Acceptance pair** (examples 1–2): identical exhibit shape; declared
  labels on every row; LAE `SD > 0` in both; EX column foots to the result.
* **Ledger algebra**: group result rows = step deltas of the running net;
  grand result = sum of group results; `scaled_stats_df` = stats of
  `X / scale` exactly (CV / Skew rows identical to `stats_df`).
* **Role orientation**: the same position built `sell` vs `buy` flips every
  row sign and its percentile ladder consistently (`Pq(−X) = −P(1−q)(X)`).
* **Cross-checks**: kernel exact moments vs the analyses' old exact stats
  (captured as fixtures before demotion); reinstatement ledger mean-check
  `E[Ceded Premium] = D + E[h(R)]`; example-3 waterfall vs the current gcn
  numbers.
* **2-D validation**: `is2d` over a 1-D source errors; second occ feature
  errors; subsequent-agg reinstatement program round-trips.
* **`validation_df`**: `bs>0` legs report Est-vs-EX at `VALIDATION_NOISE`
  scale; `bs=0` legs absent.
* **Massive**: small zarr-backed joint → ledger in one sweep;
  `mean(result) == sum(signed leg means)` to `VALIDATION_NOISE`.

## Risks / notes

* **Breaking surface** (pre-beta, deliberate): `create_pnl` /
  `create_pnl_tower` / `PnLTower` / `margin_df` gone; analyses' exhibits
  gone; `pnl.analysis` narrowed; a reins program's `pnl` exhibit row set
  grows (cession groups as rows — downstream row-label consumers rebase
  once). One migration note in CHANGELOG.
* The degenerate-1-D (agg feature) and joint (occ feature) paths must
  produce the **same leg labels** for the same DecL clauses — write the
  label resolution once, in the builders.
* Reinstatement `Scaled` denominator stays the committed
  `gross − deposit − pc_agg` (moves into the builder).
* `evaluate()` reads the grand obligation total; its regular-`bs` constraint
  is unchanged — revisit post-beta if needed.

## Decision log (all settled with the author, 2026-07-03)

* **[One-2D-Source]** — many `is2d` legs over ONE shared joint; two distinct
  2-D sources is the error ("you just can't have two different 2d sources");
  builder-level guard for two occ features.
* **[Exact-Legs-By-Default]** — `bs=0` exact irregular; `bs>0` (mainly 2-D)
  rebuckets; GDs need no equal spacing.
* **[Analyses-Demoted-To-Builders]** — "no bespoke exhibits anywhere — all
  generic and use the DecL `as` keyword to appropriately label."
* **[PnL-Public-Constructor]** — `PnL(...)` is the entry; `create_pnl`
  retired (its Aggregate-morphing job lives on as `make_pnl`); param
  `source`; `probs=` removed.
* **[PnL-Module-Location]** — REVISED: `_pnl.py` stays (house pattern =
  underscore implementation + public re-export; `distributions.py` /
  `portfolio.py` are pure facades); builders in `_pnl_builders.py`;
  first-class via re-export + docs, no rename.
* **[Leg-Spec-Object]** — exported `Leg`, **label first**
  (`Leg('LAE', lambda x: 0.05 * x)`); ordered lists canonical;
  `{label: func}` shorthand kept; internal evaluated leg renamed
  (`_EvaluatedLeg`).
* **[Signed-Exhibits-Role-Orientation]** — role drives signs and percentile
  orientation; EX adds down the sheet ("all accounting should work like
  that").
* **[Cash-Flow-Legs-Default]** — kernel neutral; DecL builders always book
  the parts; "net" anythings are results, never construction primitives.
* **[Leg-Audit-Validation]** — `bs>0` legs feed `PnL.validation_df`; no
  reach-through to the source's validation.
* **[Group-Ledger]** — "LOVE IT — yes group is right": waterfall = list of
  `Group`s (each a mini-P&L, role per group — contra booking via `role`, no
  `credit` flag); unit template rows; totals / running nets / step deltas /
  grand total all derived (group result = step delta; `Total` automatic);
  rows = ledger, columns = metrics; `stats_df` + `scaled_stats_df` (stats of
  `X / scale`) + `summary_df` as one row-ledger, three views; `+`
  composition; `PnLTower` retires, `stack_marginal_pnls` survives for the
  no-joint case.
* **[Pnl-Returns-The-Ledger]** — **VETOED** (author, 2026-07-03): an
  LLM-introduced "refinement" flag, dropped as redundant — what `pnl` /
  `xpnl` return follows from [Group-Ledger] + [Cash-Flow-Legs-Default] and
  is spelled out in the routing section; no separate decision exists. Do
  not reintroduce.

**Future (out of scope, keep buildable):** [Layer-Peeling-Shorthand] — a
DecL shorthand adding reins layers one at a time (top-down / bottom-up), one
group per layer, each group's result the layer's marginal impact.
