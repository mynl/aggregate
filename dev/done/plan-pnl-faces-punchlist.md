# [PnL-Faces-Punchlist]

> **CLOSED 2026-07-27 — executed in full.** Archived from `dev/PLAN-A.md`;
> renamed to the house form (CLAUDE.md forbids cryptic labels — the author's
> session handle for this plan was **PLAN-A**, and that name still greps here).
>
> **"Any remaining bugs?" — no.** Every defect and design gap in the inventory
> below was fixed and is under test: `tests/test_composition_matrix.py` plus the
> four `test_pnl*` suites run green (78 passed, 2026-07-27), and the two-tier
> classifier is live at `underwriter.py:1119`. Three items survive and are all
> carried in `dev/TODO.md`, so nothing is lost by closing this file:
> **[Walk-Step-Default-Labels]** (still needs the author's format pick — the
> fallback is still `'ceded occ'` / `'ceded agg'`, `_pnl_builders.py:573`,
> `:904`, `:1147`), **[Consolidated-LAE-Off-Source]**, and
> **[Aggregate-Summary-DF-Useless]**.

Status: EXECUTED 2026-07-05 — Phase 0 (audit) + Phase 1 [One-Classifier-Fix]
= `1.0.0a138`, Phase 2 [Consolidated-Reinstatement-PnL] = `1.0.0a139`,
Phases 3–4 = `1.0.0a140`, author-feedback addendum = `1.0.0a141`.

## Addendum (author feedback on a140, executed as a141)

1. **[Single-Block-One-Step-Walk]** — the one-step walk renders just its
   `Gross` block (no grand rows, no zero impact); `force_tower` became
   presentation-only.
2. **[Kappa-Walks]** — every DecL `xpnl` walk converted from the marginal
   stitch to **per-atom** construction so the ladder is the footing
   scenario (κ) pass: gross-marginal atoms when no occ program, the
   occurrence `(gross, ceded)` joint (occ_bivariate) when one exists.
   [GC-Tower-Marginal-Stitch] retired from the builders (the kernel
   `stitched_rows` mode stays as the no-joint seam for a future
   massive-source xpnl, covered by a direct kernel test). Trade-off
   accepted: occ-bearing walk EX values are joint-grid accurate
   (consolidated `pnl` keeps the exact marginals); walks cost one 2-D FFT;
   `CoarseJointGridWarning` guards these joints too. Impact row = true
   per-atom difference now; LAE stochastic on all walks.
3. **[All-Gross-Loss-Renames]** — step fallback `'gross'`→`'Gross'`; grand
   step key `'Total'`→`'All'` everywhere (incl. massive towers); default
   loss leg `'loss'`→`'Loss'` (`'Loss (net)'`). ONE ITEM DEFERRED pending an author format pick:
[Walk-Step-Default-Labels] — undeclared cover steps still read
`'ceded occ'` / `'ceded agg'`; the proposal is the DecL layer descriptor
(one layer: `'occ 4750 xs 250'`, `'agg 95% po 100 xs 100'`; several layers:
keep the generic name). It renames Step keys and derived plan rows
(`'<label> result'`, `'net through <label>'`) in every undeclared program,
so the format deserves a deliberate pick rather than a mid-flight default.
Author's session handle: PLAN-A.
Date: 2026-07-05. Source: pnl/xpnl exploration sessions (14-example sweep,
reinstatement mechanics investigation + MC adjudication, CatBook composition
example).

## Author decisions recorded (2026-07-05)

1. **[Decision-Zero-Premium-Warns]** — yes: a cession with no premium clause in
   a `pnl`/`xpnl` builds at zero ceded premium and warns once.
2. **[Decision-Log-Consolidated-LAE-Off-Source]** — the GC consolidated face
   books loss-basis LAE deterministically (`rate * E[gross loss]`) because its
   source is the net marginal and the gross loss is not measurable on it. Logged
   as a known problem (see Logged problems below), not fixed in this plan.
3. **[Decision-Engine-Attribute-Name]** — `engine` approved (sibling of
   `engine_label`; collision sweep still due at draft-execution time).
4. **[Decision-Grid-Adequacy-Threshold]** — warn when a treaty kink region spans
   fewer than 20 buckets of the reinstatement joint.
5. **[Decision-Grid-Knobs-Scope]** — clarified: the coarse-grid concern is the
   2-D reinstatement joint ONLY (common bs ~40 vs engine bs ~1). The 1-D
   variable features (swing/slide/pc/corridor/retro) evaluate on the engine's
   own fine grid — no new grid, no knobs needed. Mechanism: document the API
   recipe (`agg.reinstatement_analysis(bs=, log2_x=, log2_y=)`), no grammar
   change. (Confirmed by author framing; revisit only if a 2-D feature family
   grows.)
6. **[Decision-XPnL-Plain-Is-One-Step-Walk]** — `xpnl` over a plain engine
   returns the trivial one-step walk (gross → Total), not an error. "Boring but
   OK."

## Issue inventory

### Defects (wrong or inconsistent behavior)

* **[XPnL-Zero-Premium-Cessions]** — classification keys on economics presence
  (`underwriter.py` `'gcn' if econ is not None else 'plain'`), not reinsurance
  presence. Reinsurance with no premium clause ⇒ `xpnl` raises
  NotImplementedError; `pnl` silently takes the plain face (`consideration` /
  `loss` labels, `economics=None`) instead of consolidated (`net premium` /
  `loss (net)`). One root cause, two symptoms.

* **[Var-Feature-Composed-With-Occ-Program]** — **the meat-and-potatoes defect**
  (CatBook example: GC occ program + swing-rated agg cover, engine referenced
  into an `xpnl`). The `var_feat` branch (`underwriter.py` ~1104–1126) pops only
  `agg_reins` and assumes the inner engine's density is the GROSS aggregate —
  but with an inuring occ program the engine's density is **net-of-occ**. Both
  variable-rating builders use `source=agg` and thread only the agg-side
  economics (`econ['pc_agg']` / `econ['c_agg']`); the occ side (`pc_occ`,
  `c_occ`) is resolved and then dropped on the floor. Consequences:
  - **walk**: the gross step books the true gross premium against the
    *net-of-occ* loss (labeled with the declared gross-loss label — silently
    mislabeled); the ceded occ step (premium / commission / recovery) is
    entirely missing.
  - **consolidated**: the net-premium map omits `− pc_occ + c_occ` (a constant
    error, −6,187.5 in the CatBook example); the net-loss leg `x − g(x)` is
    actually correct (the agg treaty's subject IS net-of-occ — right by
    accident).
  - Silent in both faces. The engine object itself (plain `agg` build) is fine.

* **[Consolidated-Reinstatement-PnL]** — `pnl` with reinstatements returns the
  2-D tower, identical to `xpnl` (the deliberate [2D-Deferred] carve-out, now
  superseded by the author's design statement: `pnl` = consolidated COM view
  sourced from the bivariate). Mechanics of the tower itself verified correct
  (MC-adjudicated 2026-07-05); only the face shape is wrong.

* **[Reinstatements-Dropped-By-Feature-Branch]** — CONFIRMED (CatBookReinst
  pair, 2026-07-05): in the `_factory` elif chain (`underwriter.py:1081/1104/1127`)
  the `var_feat` branch precedes the `reinst` branch, and `occ_reins_reinst` is
  popped from the spec at :1063 before either. A program with BOTH a
  reinstatements clause on the occ layer and a feature on the agg cover takes
  the var branch and the reinstatements clause is silently discarded — the occ
  recovery's annual cap evaporates. Demonstrated: occ `4750 xs 250 rate 55% no
  reinstatements` + swing agg cover; heavy tail (pareto 1.2) makes the annual
  cap bind in 64% of years, so E[occ recovery] is 4,031 capped vs 6,873
  uncapped. Combined with [Var-Feature-Composed-With-Occ-Program] (net-as-gross
  + dropped occ economics), the walk's bottom line flips sign: true net margin
  ~ −1,344 vs reported +8,331. Same root cause family: the single-kind
  classification.

### Design gaps (numbers right, surface wrong)

* **[Engine-Reference-On-PnL]** — consolidated builder passes a marginal GD as
  source and keeps NO reference to the wrapped Aggregate; gross passes the
  Aggregate itself as source. Rule: simplest source in (GD / bivariate), plus an
  `engine` attribute on every DecL-assembled P&L.
* **[Walk-Step-Default-Labels]** — undeclared steps default to
  `'ceded occ'`/`'ceded agg'`; prefer the layer descriptor
  (`'occ 4900 xs 100'`). Related: plain `pnl` books the inherited premium line
  as `consideration` while the walk's gross step calls it `premium` — unify.
* **[Reinst-Joint-Grid-Adequacy]** — the reinstatement joint's common bs puts a
  Jensen-type O(bs) bias on kinked legs (0.25% on the CatBook-scale example's
  agg recovery at 25 buckets across the layer; a narrow layer would be garbage).
  Warn under 20 buckets (decision 4); document the API knobs (decision 5).

### Logged problems (not scheduled in this plan)

* **[Consolidated-LAE-Off-Source]** — GC consolidated face cannot book
  stochastic loss-basis LAE (source = net marginal; gross loss unmeasurable on
  it). Candidate future fixes: a (gross, net) bivariate source, or a stitched
  extra row off the gross marginal. Decision 2: logged, deferred.
* **[Aggregate-Summary-DF-Useless]** — `Aggregate.summary_df` on a decorated
  engine (CatBook `ob4`) judged USELESS by the author; needs a redesign. Related
  but not identical to [Accounting-Summary-DF] in dev/TODO.md — reconcile when
  scoped.
* **PnL walk summary card** (`ob5.summary_df`) also flagged wrong by the author
  ("we know that and are fixing it") — covered by the phases below + the
  summary-card work above.

## Phases (execution order)

### Phase 0 — [Composition-Matrix-Audit] — DONE 2026-07-05

**Variable-feature sweep (no occ program): ALL CLEAN.** 15/15 exact checks
pass — every booked EX for swing / slide / pc / corridor / retro, both faces,
matches an independently implemented documented formula (clip-affine, interp
anchors, share*(1-LR-allow)+, corridor LR reduction) computed directly over
the engine density; end-to-end MC (400k sims) z = +0.9 on the swing premium
and corridor recovery. The features themselves need no fixes.

**Matrix enumeration results** (signals: gross-step loss vs true gross mean;
occ step presence; reinstatement cap honored):

| occ \ agg | none | GC | feature |
|---|---|---|---|
| none   | pnl OK; xpnl NIE (fix: 1-step walk) | OK / OK | OK / OK |
| GC     | OK / OK | OK / OK | **pnl: net prem misses occ constants; xpnl: GROSSLOSS_BAD + OCC_STEP_MISSING** |
| reinst | OK / OK (cap_ok) | OK / OK (cap_ok) | **both faces: GROSSLOSS_BAD + CAP_BAD (197 vs 743) — var branch wins, reinstatements dropped** |

Zero-premium variants: `(gc0, none)` and `(none, gc0)` → pnl takes the plain
face (`consideration`/`loss` labels, `economics=None`); xpnl NIE. Confirms
[XPnL-Zero-Premium-Cessions].

**Implementation notes settled by the audit + code read (Phase 1 design):**

* Classifier: `occ_tier = reinst|gc|none` (reinst = `occ_reins_reinst`
  popped at `underwriter.py:1063`), `agg_tier = feat|gc|none`; retro keeps its
  separate head-clause guard. Zero-premium: a side with layers but no
  `<side>_reins_premium` key resolves to zero ceded premium and warns
  (`ZeroPremiumCessionWarning`, new in constants). Exclusions: the
  feature-decorated agg side (swing supplies premium; slide/pc/corridor
  require deposit — existing errors keep working); reinstatements still
  require an occ base premium (existing error preserved).
* `(gc, feat)` consolidated: `_build_variable_consolidated` premium maps gain
  the constants `- pc_occ + c_occ`; source stays the engine (net-of-occ
  atoms — correct subject); `economics` attached.
* `(gc, feat)` walk: new stitched builder — gross step (`p_agg_gross`
  affine), ceded occ GC step (constants + `p_agg_ceded_occ`; running net
  `p_agg_net_occ`), feature step + Total as **pushforwards of the net-occ
  marginal** via a new `_fn_marginal_entry(agg, perspective, fn, label)`
  (exact atoms: map, sort, collapse duplicates — no re-gridding; exact
  mean/sd off raw atoms). Factor `build_xpnl_walk`'s desc→entries assembly
  into a shared `_stitch_ledger` helper; desc rows become tagged
  `('const', b) | ('affine', persp, a, b) | ('fn', persp, f)`. Marginal (P)
  ladder; LAE deterministic (all-marginal tower).
* `(reinst, feat)`: reorder so the reinst branch wins; thread the feature
  terms into `ReinstatementAnalysis` (new `agg_feature_terms=`): the three
  agg-tier maps become closures — recovery `REC` (corridor-adjusted where
  applicable), premium `P = phi_swing(REC_raw)` or constant, stochastic
  commission `K = phi_slide_pc(REC/P_C)*P_C` folded INTO the uw legs (a
  stochastic commission cannot ride the scalar expense-shift path; the
  scalar `agg_commission` stays 0 then). `build_reinstatement_pnl` mirrors
  the maps as group legs. Face = tower for both (until Phase 2).
* One-step walk (plain xpnl): `_ledger_plan(groups, result_name,
  force_grand=True)` emits qualified labels + grand rows for a single group;
  Total duplicates gross, Impact = 0 delta.
* Test churn: `test_xpnl_over_plain_engine_not_implemented` flips to
  one-step-walk assertions; zero-premium tests in `test_pnl_ceded_premium`
  flip from plain-face to consolidated-face expectations; new
  `tests/test_composition_matrix.py` carries the matrix acceptance
  (CatBook, CatBookReinst, NoPrem pairs).

### Phase 1 — [One-Classifier-Fix] (the author's "one fix, soon")

Replace the single-kind elif chain with a **two-tier classifier** — occ tier
`{none | GC | reinstatements}` x agg tier `{none | GC | feature}` — and a
dispatch table in which every cell is either a correct builder or (only if a
cell is consciously deferred) a LOUD NotImplementedError. Never silent. Retro
stays the separate head-clause case with its existing no-reinsurance guard.

| occ \ agg          | none            | GC              | feature                       |
|--------------------|-----------------|-----------------|-------------------------------|
| none               | plain (works)   | gcn (works)     | var (works today)             |
| GC                 | gcn (works)     | gcn (works)     | **composed 1-D** (new, below) |
| reinstatements     | reins (works)   | reins (works)   | **reins + feature map** (new) |

Bundled into the same slice (all one "pathway" fix):

* **[XPnL-Zero-Premium-Cessions]**: classify GC on reinsurance presence;
  missing premium clause ⇒ zero ceded premium + one warning (decision 1).
* **[Decision-XPnL-Plain-Is-One-Step-Walk]**: `xpnl` over a plain engine ⇒
  trivial one-step walk (decision 6).
* **(GC occ, feature agg) — composed 1-D builder**
  ([Var-Feature-Composed-With-Occ-Program]): the feature's subject is
  net-of-occ; everything is a pushforward of the ONE `p_agg_net_occ` marginal
  plus GC occ constants, so both faces are exact.
  - consolidated: net premium map gains the occ constants —
    `P_G − pc_occ + c_occ − φ(g(x)) + C_agg`; loss stays `−(x − g(x))`;
    `economics` gains the occ side; source = net-of-occ marginal GD.
  - walk: gross step from `p_agg_gross` (true gross loss; stitched row), ceded
    occ GC step (constant premium/commission + recovery from
    `p_agg_ceded_occ`, running net from `p_agg_net_occ`), ceded agg feature
    step (function legs over the net-occ atoms: `g(x)`, `−φ(g(x))`), Total
    (functions of net-occ). Mixed stitched + per-atom rows — `stitched_rows`
    already accepts per-row `(gd, mean, sd)`. Engine-drift caveat as the GC
    walk (separate FFTs, rel ~1e-8).
* **(reinstatements occ, feature agg) — feature map on the reinstatement
  route's agg tier** ([Reinstatements-Dropped-By-Feature-Branch] fix): the agg
  tier already books over the (L, R) joint via
  `g_rec(l, r) = g(max(l − A(r), 0))`; the feature premium replaces the
  constant `pc_agg` with the deterministic map `φ(g_rec(l, r))` — still a
  pushforward of the same joint. Face = the 2-D tower for both pnl and xpnl
  (consistent with the reinstatement carve-out until the
  consolidated-reinstatement phase below). If this
  cell slips the slice, it gets the loud error, never the var branch.

Acceptance:
* NetAggNoPrem / xNetAggNoPrem pair (zero-premium).
* CatBook `ob4`/`ob5` (GC occ + swing agg): occ step present with premium
  −6,875 / commission +687.5, gross step's loss = the true gross,
  consolidated net premium = 12,500 − 6,875 + 687.5 − E[φ(g)]; cross-face
  means agree.
* CatBookReinst pair (occ `rate 55% no reinstatements` ± swing agg): with the
  agg cover present, the occ tier must still route through the reinstatement
  terms — occ recovery EX ≈ 4,031 (capped), never 6,873 (uncapped); gross
  step loss = −10,000; the with/without-agg-cover walks agree on every shared
  row.

### Phase 2 — [Consolidated-Reinstatement-PnL]

Face split in `build_reinstatement_pnl` mirroring `build_variable_pnl(walk=)`:
walk keeps the tower; consolidated books one sell group of 2-D legs over the
same joint — `net premium (l,r) = P_G − D − h(r) − pc_agg + commissions`,
`loss (net) (l,r) = −(l − A(r) − g_rec(l,r))`, expenses (loss-basis LAE CAN
stay stochastic here — axis 0 carries gross loss; note vs
[Consolidated-LAE-Off-Source]). Routing: `kind == 'reins'`, `is_tower=False`.
Acceptance: verified numbers from the 2026-07-05 MC session (net premium EX
3,114.2; loss (net) EX −3,641.8); `pnl.mean == xpnl` grand total EXACTLY (one
shared joint — no drift tolerance). Closes [2D-Deferred]; revisit
[Gross-Anchored-Kappa-Insight].

### Phase 3 — [Engine-Reference-On-PnL] + label polish

`engine` attribute on every DecL-assembled P&L (collision sweep first, per the
naming standing order); plain/gross builder normalized to a GD source; walk
step labels default to the layer descriptor; unify `consideration`/`premium`.

### Phase 4 — [Reinst-Joint-Grid-Adequacy]

Definite warning from `reinstatement_analysis` when a treaty kink region spans
< 20 buckets of the joint (decision 4); document the API pass-through recipe
(decision 5 — no grammar change); point `bs_description` at the remedy.

## Housekeeping per standing rules

Each phase: tests, version bump, CHANGELOG section, dev/TODO.md reconciliation
([2D-Deferred] closes at Phase 2), FEATURES.csv where the surface changes,
decl-testers.agg for any new DecL programs. Author commits.
