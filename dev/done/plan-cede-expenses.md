# Plan [Cede-Contra-Expense]: ceding commission folds into the expense column

> **Status: EXECUTED at `1.0.0a304`, 2026-08-18.** All five phases, in one
> commit, on the author's ruling that ceding commission is contra expense the
> same way a cession recovery is contra loss. No grammar change, as drafted.
> Execution notes at the foot.

## Finding (investigated 2026-08-18)

The API's Economics, Ratios exhibit (`economic_ratios`, serving
`PnL.economic_ratios_df`) reports amounts `P / L / E / C / M`. The split comes
from `_RATIO_BUCKET` in `_pnl.py:756`: `'recovery'` already buckets into `L`
(contra loss) but `'commission'` gets its own `C`. That asymmetry is the whole
issue: statutory convention nets ceding commission received against acquisition
expense, exactly parallel to recoveries against loss.

**The posited justification was searched for and not found.** The originating
plan (`dev/done/plan-pnl-ratio-frame.md`, a185) records no economic reason for
a separate `C`; the closest things on record are the pentagon-spelling note
("`E` / `C` / `ER` / `CR` extend it", a naming remark) and the format sheet's
"a column carries one unit" comment (both are money, so it does not apply).
The only rationale that can be reconstructed is keeping a visible gross
expense ratio (`ER = E / P` un-netted), and it does not hold up: commission
legs only arise on cession blocks, so the Gross row's `ER` is identical either
way and the gross reading survives where it belongs. **The library's own
earlier design agrees with the ruling**: the a116-era GCN waterfall booked
"per-tier commission credits in the GCN Expense section" (CHANGELOG), so the
separate `C` introduced at a185 is the anomaly, not the tradition.

## What changes, numerically

With commission bucketed into `E` (signs per `_block_amounts`, gross
direction, so a cession's commission enters negative, a credit):

- `M == P - L - E` becomes the row identity (was `P - L - E - C`).
- `CR` is **numerically invariant**: it already summed `(L + E + C) / P`,
  now `(L + E) / P` over the folded `E`. Same for `E_CR`.
- `LR`, `E_LR`, `P_share`, `M_share`: unchanged.
- `ER` (and `E_ER`) change meaning where commission exists: on a cession
  block `ER` reads as the commission rate with conventional sign, and on the
  `All` row it becomes the statutory net-of-commission expense ratio.
- No information is lost: `legs_df` keeps `kind='commission'` itemized per
  leg, and that frame is the documented place to recover any split the ratio
  frame does not carry.

A hand-built sell-group leg tagged `'commission'` (agent commission paid)
folds into `E` as acquisition expense, which is also the correct reading.

## Edits

1. **[Core]** `src/aggregate/_pnl.py`:
   `_RATIO_BUCKET['commission'] = 'E'`; `_RATIO_AMOUNTS` drops `'C'`;
   `_RATIO_COLS` drops `'C'`; `economic_ratios_df` drops `C` from the row dict,
   `CR` numerator becomes `ell + e`, `E_CR` numerator `v['L'] + v['E']`.
   Docstrings follow (`_block_amounts`, `economic_ratios_df`, the
   `_RATIO_AMOUNTS` / `_RATIO_COLS` comments): `E` absorbs ceding commission
   as contra expense, mirroring `L` absorbing recoveries. While touching the
   `LEG_KINDS` comment (`_pnl.py:134`), correct it: it calls `'commission'` a
   consideration-side flow on a `buy` group, but every builder declares
   commission legs on the **obligation** side (verified by grep of
   `_pnl_builders.py`). `LEG_KINDS` itself is unchanged: `'commission'` stays
   a valid kind, reported by `legs_df`.
2. **[Exhibits]** `src/aggregate/exhibits/_pnl.py`: `_AMOUNT_COLS` drops
   `'C'`; the amounts caption becomes "Premium, loss and expense per block ...
   `M = P - L - E` holds exactly. Loss absorbs cession recoveries and any
   unclassified obligation leg; expense absorbs ceding commission, a contra
   expense." Ratio block and legs block unchanged.
3. **[Formats]** `src/aggregate/formats/formats-raw.yaml`: delete the
   `C: money` entry and reword the adjacent comment (verified: no other frame
   in the library carries a bare `C` column).
4. **[Tests]** Update the identity assertions to `P - L - E`:
   `tests/test_create_pnl.py:328` and `:344` (drop the `row['C']` check),
   `tests/test_exhibits.py:795`, `tests/test_pnl_peel.py:700`. Add one
   commission-specific case (a GCN pnl with `cede`): the cession row's `E` is
   minus the commission, the `All` row's `ER` nets it, and `CR` equals its
   pre-fold value.
5. **[Docs-Hygiene]** `.rst` in lockstep, rebuild pending for the author:
   `docs/2_aggregate_overview/pipeline-pnl.rst:360` (column list and
   identity), `docs/2_aggregate_overview/features.rst:2229`, plus a scan of
   `3_x_Exhibits.rst` and `pipeline-exhibits-and-charts.rst` for the column
   roster. CHANGELOG section, version bump, one-line commit.

## API side: nothing

Per the purist ruling the app draws what it is served. The Economics, Ratios
pane loads the exhibit envelope and the narrower amounts block flows through;
grep of `aggregate_api` found no client-side reference to the `C` column.
No round-note ask, no API phase.

## Acceptance

- `M == P - L - E` on every block of a peeled two-tier walk and on `All`.
- `CR` and `E_CR` byte for byte unchanged on every ledger (the fold is
  invariant for them); `LR` and shares unchanged everywhere; `ER` moves only
  on blocks that carry commission legs.
- `legs_df` unchanged, `economic_df` (the ledger sheet) unchanged: this is a
  ratio-frame and exhibit-presentation change only.


## Execution notes (`1.0.0a304`)

Every claim the plan made was checked against the code before executing, and
all of them held: the `_RATIO_BUCKET` asymmetry, the absence of any recorded
justification for `C`, the a116 GCN waterfall precedent for booking commission
credits in the Expense section, all thirteen commission legs being declared on
the obligation side, no other frame in the library carrying a bare `C` column,
and the API holding no client-side reference to it (the Economics, Ratios pane
loads the exhibit envelope generically through `loadExhibitLeaf`).

The acceptance criteria were verified numerically on the `TWO_EACH` peeled walk
before and after, one block at a time. `M == P - L - E` foots to `0.0` exactly;
`CR`, `E_CR`, `LR`, `E_LR`, `P`, `L`, `M`, `P_share` and `M_share` are equal
element for element to their pre-fold values; `E` and `ER` move on exactly the
three blocks that contain the commission leg (`occ 100 xs 100`, its tier
subtotal, and `All`), the cession block reading `ER = 0.2`, its `cede` rate, and
`All` reading `-0.013953`, the expense ratio net of commission.

Four departures from the plan as drafted.

1. **Eight test sites, not four.** The plan named `test_create_pnl.py:328` and
   `:344`, `test_exhibits.py:795` and `test_pnl_peel.py:700`. It missed
   `test_create_pnl.py:354` and `test_pnl_peel.py:686` (both iterate
   `('P', 'L', 'E', 'C', 'M')`), `test_exhibits.py:783` (the amounts column
   set), and `test_exhibit_formats.py:76`, whose money-vocabulary list carried
   `C` and which fails with a bare `KeyError` once the format entry goes.
2. **`tests/data/exhibit_snapshots.json` had to be regenerated**, which the
   plan did not anticipate. Six canonical-snapshot cases cover
   `economic_ratios` across two perspectives and three objects. The diff is
   exactly what it should be: the `C` column, its header and its cells leaving,
   the trailing column keys renumbering from `c5`, and the new caption.
   Regenerated with `tests/capture_exhibit_snapshots.py`.
3. **`docs/2_aggregate_overview/features.rst` was deliberately NOT edited**,
   against the plan's phase 5. That page is governed by `dev/task-features.md`,
   which makes runs additive and repair-only and forbids rewriting existing
   prose, and its sections are release history: the one holding the identity is
   headed "Ratios in their own frame: ``ratio_df`` and ``legs_df`` (a185,
   a186)" and still spells the frame `ratio_df`, unchanged through the a204
   rename. Retro-editing it would be the first time the house did that. This
   change belongs there as a new section on the next task run. Noted separately
   for that run: the two `qd(tower.ratio_df)` / `qd(retro.ratio_df[...])` code
   blocks on that page have been broken since a204 and are a legitimate repair
   target, independent of this plan.
4. **`dev/FEATURES.csv`** was updated too, which the plan did not list: the
   `economic_ratios_df` row's notes described the `P/L/E/C` roster and the
   `M == P-L-E-C` identity.

One thing corrected in passing, as the plan asked: the `LEG_KINDS` comment
called `'commission'` a consideration-side flow on a `buy` group. It is an
obligation-side leg in every builder, which is precisely why it enters the
amounts negated and reads as a credit.
