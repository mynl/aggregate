# Plan: aggregate `rate` nets the inuring occurrence premium; commissions move to the obligation side

Date: 2026-09-29. Status: executed 2026-09-29 as a359 (`[Agg-Rate-Nets-Inuring-Occ]`) and a360 (`[Commission-Obligation-Side]`); see the execution log at the end.

Two related defects in the PnL reinsurance economics, found via the Capstone
program (an occurrence XOL tower inuring to an aggregate quota share). They are
independent fixes and land as two version bumps so the second is verifiable as
margin-preserving on its own.

- `[Agg-Rate-Nets-Inuring-Occ]` (bump 1, a358 at time of writing): an
  aggregate-side `rate` clause prices off the stated gross premium, ignoring
  the premium ceded to an inuring occurrence program. It should price off the
  subject premium, net of the occurrence cession.
- `[Commission-Obligation-Side]` (bump 2, a359 at time of writing): the
  consolidated `pnl` face folds ceding commissions into the net premium
  consideration leg, while every `xpnl` walk face books them as obligation
  legs of the buy groups. The two faces therefore report different net premium
  and net obligation totals (margins agree). Harmonize on the walk convention:
  commissions are obligation-side flows, and the consolidated net premium
  becomes `gross - ceded premiums` with no commission term.

If other work lands first, renumber the bumps; one bump per commit as usual.

## Background and reproduction

Test program (DecL), the Capstone example:

```decl
xpnl Capstone.PnL
  derive premium
  less
    agg Capstone.FullProgram
      [2000 5000 10000 3000 1000] premium as GWP at [0.95 0.95 0.9 0.85 0.75] lr
      [500 1000 5000 10000 5000] xs [0 0 0 0 1000]
      sev 151.63266492815836 * lognorm 1
        picks [500 1000 2000 5000 10000] [14900 2250 1250 500 50]
      occurrence net of
        50% po 500 xs 500 deposit 3000 as "500x500" and
        90% po 1000 xs 1000 deposit 1750 as "1x1" and
        95% po 3000 xs 2000 rol 0.25 as "3x2" and
        5000 xs 5000 rol 0.03 as "5x5"
      mixed gamma 0.125
      aggregate net of
        75% po inf xs 0 rate 1 cede 0.325 as QS
  less
    0.25 premium expenses as "G&A"
  peel bottom-up
```

`derive premium` grosses the technical premium 21000 up for the 25% premium
expense: gross 28000. The occurrence tower resolves to ceded premium
`pc_occ = 1500 + 1575 + 712.5 + 150 = 3937.5`.

**Defect 1.** The QS books `pc_agg = 0.75 x 1.0 x 28000 = 21000`: the `rate`
base is the stated gross. The occurrence program inures to the aggregate
cover, so the QS subject premium is the net position:
`0.75 x 1.0 x (28000 - 3937.5) = 18046.875`, and the ceding commission is
`0.325 x 18046.875 = 5865.234375`. The loss side already inures correctly
(the QS recovery evaluates on the net-of-occ joint: expected recovery
`12131.25 = 0.75 x 16175`); only the premium base is wrong.

Root cause: `Underwriter._resolve_reins_economics`
(`src/aggregate/underwriter.py`, `side()` closure, the `rate` branch at
line 1418) uses `pg` (the stated or derived gross premium) as the `rate` base
for both the occ and the agg side.

**Defect 2.** At the current tree the two faces agree on the margin
distribution exactly, but split it differently:

| face | Consideration | Obligation | Margin |
|---|---|---|---|
| `pnl` (consolidated) | 9887.50 | -11043.75 | -1156.25 |
| `xpnl` All row | 3062.50 | -4218.75 | -1156.25 |

The difference is exactly `c_agg = 6825`. `build_consolidated_pnl`
(`src/aggregate/_pnl_builders.py:579`) computes
`net_premium = p_gross - pc_occ - pc_agg + c_occ + c_agg`, folding commissions
into the consideration leg. Every walk builder instead books the commission as
an obligation leg of the buy group (`kind='commission'`), where the buy role's
sign makes it a positive receipt. So "net premium" reads two different numbers
depending on the face, which is wrong: the faces must agree on the
consideration total, not just the margin.

## Decisions (settled during investigation, approved by the author)

1. **The aggregate-side `rate` base is `gross - pc_occ`**: subject premium net
   of the inuring occurrence ceded premium. The occurrence commission does NOT
   add back (commission is an expense reimbursement, not premium). The
   occurrence-side `rate` base stays the stated gross (occ is first in the
   tower).
2. **With occurrence reinstatements**, the occurrence ceded premium is
   stochastic (`D + h(R)`). The netting amount is the constant deposit, the
   resolved `econ['pc_occ']`. A stochastic rate base is out of scope.
3. **No intra-tier netting.** Several aggregate layers with `rate` clauses all
   use the same net-of-occ base; an aggregate layer does not net the premium
   of an aggregate layer below it. Out of scope; note it in the docstring.
4. **Commissions are obligation-side flows on every face.** The consolidated
   faces stop folding them into net premium; the walk faces are already
   correct and do not change. Rationale: matches the walk convention at all
   seven booking sites; the stochastic commissions (slide, profit commission)
   can never fold into a constant premium leg, so the walk convention has to
   survive anyway; and it matches the accounting reading, where ceding
   commission is an expense offset, not premium.
5. **One commission leg per consolidated face**, labeled `'commission'`,
   `kind='commission'`, carrying the total (constant plus stochastic parts).
   The consolidated card deliberately nets ceded economics out, so it does not
   get per-cover commission legs; the per-cover split is the walk's job. Add
   `'commission'` to the `taken` label set passed to `_expense_legs`.
6. **Sign mechanic.** The consolidated face is a single `sell` group and sell
   books obligations negative, while a commission is received. The leg's
   magnitude is therefore written negated (constant `-(c_occ + c_agg)`, or a
   negated callable), with a code comment saying why, so the signed row books
   and displays positive, matching the walk's commission rows. Alternative
   considered and rejected: adding a buy group to the consolidated face, which
   breaks the single-group contract of [Decision-PnL-Is-Consolidated].

## Step 1 `[Agg-Rate-Nets-Inuring-Occ]`

### Code

`src/aggregate/underwriter.py`, `_resolve_reins_economics`:

- Give the `side()` closure a `rate_base` parameter; the `rate` branch becomes
  `layer_pc = share * val * rate_base`.
- Call `side('occ', pg)` first (it already runs first), then
  `side('agg', pg - pc_occ)`.
- Update the docstring (lines 1360 to 1389): state the per-side base, the
  no-commission-add-back decision, the reinstatement deposit convention, and
  the no-intra-tier-netting scope note.

No other code changes: the consolidated, walk, peel, feature-deposit, and
reinstatement builders all read the resolved `econ` dict, so the corrected
`pc_agg` / `c_agg` flow through, including the chart structure emitter
(`charts/_emit_structure.py`), which renders per-layer economics from the same
dict.

### Tests

- Existing `rate` pins are unaffected (none has an occurrence program):
  `tests/test_pnl_ceded_premium.py` lines 41 and 52,
  `tests/test_agg_ignored_clauses.py` line 110.
- Add to `tests/test_pnl_ceded_premium.py`: a program with an occurrence
  cover priced by `deposit` or `rol` plus an aggregate cover priced by
  `rate`, asserting `pc_agg == rate x share x (gross - pc_occ)` and
  `c_agg == cede x pc_agg`; and a companion assert that `deposit` / `rol`
  aggregate clauses are unchanged by the presence of an occ program.

### Docs

- `docs/4_agg_language_reference/ref_include.rst` line 378: "``rate`` a
  fraction of gross premium" becomes a statement of the subject premium,
  net of an inuring occurrence program on the aggregate side.
- Grammar comments in `decl.lark` if they state the base (grep).

### Acceptance

On the Capstone program: `economics` shows `pc_occ 3937.5`,
`pc_agg 18046.875`, `c_agg 5865.234375`. Consolidated net premium (still
commission-inclusive at this step) `28000 - 3937.5 - 18046.875 + 5865.234375
= 11880.859375`; expected margin `837.109375` on both faces.

Bump to a358, one commit: code, tests, docstring, `ref_include.rst`,
`CHANGELOG.md` entry, `pyproject.toml`.

## Step 2 `[Commission-Obligation-Side]`

### Code, all in `src/aggregate/_pnl_builders.py`

Three consolidated faces fold commissions into net premium; each drops the
commission from the consideration and gains one obligation commission leg per
decision 5, with the negated magnitude per decision 6.

1. `build_consolidated_pnl` (guaranteed cost, line 579):
   `net_premium = p_gross - pc_occ - pc_agg`; commission leg constant
   `c_occ + c_agg` when nonzero. Update the docstring (lines 550 to 552), the
   `_construction_description` (lines 600 to 601), and
   `_consolidated_explanation` (lines 618 to 622).
2. `_build_variable_consolidated` (line 1458): `occ_shift` becomes `-pc_occ`
   only. Swing: drop `+ C` from the premium lambda. Slide and profit
   commission (`tl == 'expense'`): the net premium becomes the constant
   `P_G - pc_occ - P_C` and the stochastic credit
   `terms.phi(g_ceder(x) / P_C) * P_C` moves to the commission leg (plus the
   constant `c_occ`). Corridor: net premium `P_G - pc_occ - P_C`; commission
   leg `c_occ + C`. Update the docstring (lines 1466 to 1468). The retro
   branch has no cession and does not change.
3. The reinstatements consolidated face (line 1874): drop
   `+ c_occ + comm(l, r)` / `+ c_occ + c_agg` from `net_prem`; commission leg
   is the constant `c_occ + c_agg`, or the 2-D map `c_occ + comm(l, r)` when
   the aggregate tier carries a slide or profit commission. Update the
   docstring (lines 1885 to 1887) and the construction strings (lines 1930
   to 1938).

Also update the module docstring (line 18) and sweep the remaining formula
statements: `rg -n "commissions" src/aggregate dev docs` and fix every
statement of "gross - ceded premiums + commissions".

The walk builders, `PnL` itself, `_ledger_economics`, and the `economics`
dict do not change.

### Tests

Pins of the fold-in convention to update:

- `tests/test_pnl_ceded_premium.py`: module docstring (line 6);
  `test_cede_books_commission_into_the_net_premium` (line 76,
  `5000 - 160 + 40` becomes `5000 - 160` plus a `'commission'` leg with
  `EX == 40`); `test_both_sides_split_and_walk_ledger` (line 145,
  `5000 - 5 - 160 + 32` becomes `5000 - 5 - 160` plus the commission leg).
  The walk-side asserts (line 80 and the walk half of the both-sides test)
  are unchanged.
- `tests/test_composition_matrix.py`,
  `test_gc_feat_consolidated_books_occ_constants` (line 70): the hand
  pushforward drops `+ c_occ` from the net premium and gains a commission leg
  assert with `EX == pytest.approx(c_occ)`.
- `tests/test_variable_rating_decl.py`: `test_slide_build` (line 107) and
  `test_pc_build` (line 124) flip from `net premium SD > 0` to
  `net premium SD == 0` plus `commission SD > 0` on the consolidated face.
  `test_swing_build`, `test_swing_terms_scale_by_placement_share`, and
  `test_corridor_build` have no cede clause and are unchanged; audit them
  anyway.
- `tests/test_reinstatement_decl.py` line 364 has no cede; audit, expect
  unchanged. `tests/test_pnl_consolidated_walk.py` pins (lines 45, 181) have
  no cede; audit, expect unchanged.
- Add the cross-face invariant test: for a guaranteed-cost program with
  `cede` on both sides, the consolidated `summary_df` Consideration and
  Obligation totals equal the walk's All-row totals exactly, and the margins
  agree (this is the test that would have caught the defect).

### Acceptance

- On the Capstone program, both faces report Consideration `6015.625`
  (`28000 - 3937.5 - 18046.875`), a commission row showing `+5865.234375`,
  and margin `837.109375`, identical to step 1's margin: this step is
  margin-preserving by construction, and that invariant is the check.
- `test_variable_rating_decl.py` and `test_reinstatement_decl.py` margins
  (`est_m`, the `p.est_m == x.est_m` asserts) are unchanged from before the
  step.

Bump to a359, one commit: code, tests, docstring sweep, `CHANGELOG.md` entry,
`pyproject.toml`.

## Out of scope

- Intra-aggregate-tier premium netting (an agg layer netting the premium of
  an agg layer below it).
- A stochastic rate base under occurrence reinstatements (the deposit is the
  netting amount).
- Any change to the walk faces or to the `economics` dict schema.
- `docs/` rebuild (author runs it outside the loop; keep `.rst` edits in
  lockstep and note the pending rebuild in the run summary).

## Verification procedure

Per change: tier 1 testmon loop
(`pytest -n0 --dist no --testmon-forceselect`). Before each bump: tier 2
(`uv run pytest`). This touches premium arithmetic, so finish with tier 3 plus
the numerics gate
(`uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`) at the final
bump.

## Execution log (2026-09-29)

Both steps landed as planned; the Capstone acceptance numbers matched exactly
at each step (step 1: `pc_occ 3937.5`, `pc_agg 18046.875`, `c_agg
5865.234375`, consolidated net premium `11880.859375`, margin `837.109375`;
step 2: both faces Consideration `6015.625`, Obligation `-5178.515625`,
commission row `+5865.234375` on the walk's QS group, margin `837.109375`,
identical to step 1). Divergences, all small and recorded here:

- **Bumps renumbered a358/a359 to a359/a360**: `[PnL-Waterfall-Labels]`
  landed as a358 first. The plan anticipated this.
- **The no-aggregate reinstatement branch also moved its `c_occ`**: step 2
  item 3 listed only the aggregate-tier formulas, but the `agg_recovery is
  None` branch of `_build_reinstatement_consolidated` carried `+ c_occ` in
  its net premium too. Decision 4 (commissions are obligation-side on every
  face) covers it, so it gained the same commission-leg treatment.
- **Test renamed**: `test_cede_books_commission_into_the_net_premium` became
  `test_cede_books_commission_as_obligation_leg`; the old name stated the
  convention this step removed.
- **Cross-face invariant tolerances**: the plan said the consolidated totals
  equal the walk's All row "exactly". That holds exactly for the
  consideration totals always, and for everything in the agg-only case (the
  walk rides the exact gross marginal). With an occurrence program the walk
  rides the (gross, ceded) joint, so the obligation and margin totals agree
  to joint-grid accuracy; the test pins those at `rel=5e-3`, the tolerance
  the pre-existing `test_walk_means_add_down_the_sheet` margin pin already
  uses.
- **`ref_include.rst` is generated**: the language-reference edit is the
  `decl.lark` grammar comment plus a regeneration via
  `aggregate.parser.grammar(add_to_doc=True)`, not a hand edit of the
  `.rst`.
- **No `dev/TODO.md` entry existed** for this work, so there was nothing to
  tick.
