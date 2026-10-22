# Plan: annual aggregate deductible and aggregate limit on an occurrence layer

`[Occurrence-Annual-Terms]`, two steps, two minor bumps: `[Annual-Terms-DecL]` and
`[Annual-Terms-PnL]`. Drafted 2026-10-22 at 1.7.1 (`3f342d3`). **Status: final, all rulings made by
the author on 2026-10-22; not executed.**

## Goal

DecL accepts an annual aggregate deductible and an aggregate limit on an occurrence layer:

```
occurrence net of 90% po 100 xs 100 aad 50
occurrence net of 750 xs 250 agglimit 500
occurrence net of 650 xs 350 aad 50 agglimit 500
```

and a `pnl` / `xpnl` produces the cedant's net position for such a program: gross loss `L` less the
layer's recovery after the annual terms. The net of an excess of loss treaty with an AAD or an
annual cap is a common structure, and today it can only be built by hand.

## Background

### The contract

A per occurrence layer `y xs a` on claims `X_1, ..., X_N` recovers `R = sum_i min(y, (X_i - a)+)` a
year. An annual aggregate deductible `d` and aggregate limit `m` apply to that annual recovery, so the
reinsurer pays

```
A(R) = min((R - d)+, m)
```

and the cedant's net is `L - A(R)`, with `L` the gross annual loss. Amounts are stated **for 100%
of the layer** and the placement share `s` applies after (ruled 2026-10-22): the reinsurer pays
`s * min((R100 - d)+, m) = min((R - s d)+, s m)` where `R = s R100` is the shared recovery the
engine already computes. `m` may be unbounded (AAD alone) and `d` may be zero (limit alone).

### Why the cedant's net needs the joint

The reinsurer's side `A(R)` is a function of `R` alone, a one dimensional object. The cedant's net
`L - A(R)` depends on `L` and `R` together; they are dependent but neither is a function of the other,
because the random claim count decouples them. The joint law of `(L, R)` exists already:
`Aggregate.occ_bivariate(views=('gross', 'ceded'))`, a 2-D FFT, is the source of every occurrence
`xpnl` walk and of the reinstatement builders.

### What DecL does today

- `occurrence ceded to 750 xs 250` followed by `aggregate net of 50 xs 0` gives `(R - 50)+`, the
  reinsurer's side, correctly. There is no way to state the cedant's net.
- `occurrence net of 750 xs 250` followed by `aggregate net of 50 xs 0` applies the deductible to
  the net of occurrence loss `L - R`, a different contract. The aggregate tier of an occurrence
  program always acts on what the occurrence tier leaves the cedant
  (`_build_variable_walk_occ`, subject `max(L - C, 0)`).
- A loss corridor is the same provision as an AAD when it sits on the right cover:
  `corridor 100% po d/P xs 0` on a cover recovering `R` pays `(R - d)+`. But DecL allows a corridor
  only on an aggregate layer (`parser.py` line 1434, "supported on aggregate reinsurance only in this
  release"), so it cannot reach `R`; and one corridor is one band, so it gives an AAD or a limit,
  never both.
- Reinstatements already put an annual map on an occurrence layer: `ReinstatementTerms.recovery`
  is `A(R) = min(R, Y)` with `Y = (m+1) s y`, and the builders `build_reinstatement_source` /
  `build_reinstatement_pnl` / `_build_reinstatement_consolidated` in `_pnl_builders.py` run the
  cedant's walk and consolidated faces on the `(L, R)` joint through that map. The AAD and
  aggregate limit are the same map with a deductible: `A(R) = min((R - d)+, m)`.
- In a plain `agg`, every economics or feature clause is ignored with an
  `IgnoredDecLClauseWarning` (`_strip_ignored_economics` and `_AGG_IGNORED_ECONOMICS_KEYS` in
  `underwriter.py`, [Reins-Economics-On-Agg-Ignore-Warn]), including the loss-changing ones: an
  `agg` with `occurrence ceded to 700 xs 0 deposit 1172 no reinstatements` reports a ceded mean of
  900, not the capped value. The stored program keeps the full declaration, so
  `xpnl X 6000 premium less agg.NAME` re-injects it; this build-then-reference workflow was
  confirmed on 1.7.1 with reinstatements and `cede`.

A worked example of the whole calculation, by hand with `PnL` / `Group` / `Leg` and
`CorridorTerms` over the joint, is section 5 of
`D:\projects\hacks\individual-pieces\occurrence-layer-aggregate-deductible.qmd`. Its numbers are the
acceptance target for step 2.

## Rulings

All made by the author, 2026-10-22.

1. **Plain `agg`, `occurrence ceded to`:** the annual terms are **honored**. The ceded aggregate is
   `A(R)`, exact in one dimension.
2. **Plain `agg`, `occurrence net of`:** the annual terms are **ignored with a warning**, like every
   other loss-changing clause in a plain `agg` (reinstatements, corridor). The reason is the
   workflow: build the full `agg` program, then refer to it by name in a `pnl` / `xpnl`, which uses
   everything.
3. **Reinstatements outside a P&L:** unchanged, ignored with a warning, for the same reason.
   Corridors likewise unchanged.
4. **Amounts are 100% values**, prorated by the placement share.
5. **The annual terms cannot coexist with reinstatements** on a layer. They **can** coexist with
   `cede` (a ceding commission) and with the ceded premium clause (`deposit` / `rol` / `rate`).

6. **The keyword for the aggregate limit is `agglimit`**, a solid compound like `netceded`,
   `grossceded`, `xpnl` and `nttagg`. `aal` was out (it means average annual loss in the industry).
   `agg limit` (two words) is ruled out by the grammar: `agg` starts a unit inside a `port` body, so
   `... 750 xs 250 agg limit 500 ...` can parse as a new unit named `limit`. `agg-limit` would lex as
   one token, but DecL has no hyphenated keyword and this would have been the first.
7. **Coexistence with an aggregate-tier variable feature** (swing / slide / pc / corridor on an
   `aggregate` clause of the same program) is **allowed, exactly as with reinstatements**. The
   reinstatement builders already compose an occurrence annual map with one aggregate-tier feature
   over the same joint (`agg_tier_maps`, [Var-Feature-Composed-With-Occ-Program]), so the annual terms
   inherit it at no cost. The occurrence tier carries at most one of {reinstatements, annual terms},
   the aggregate tier at most one feature.
8. **The asymmetry on `ceded to` is accepted.** Ruling 1 honors `agglimit` on a ceded occurrence
   layer while ruling 3 ignores the reinstatement cap there, so `750 xs 250 agglimit 1500` and
   `750 xs 250 reinstatements 1 free` give different ceded distributions in a plain `agg`, though the
   annual limit is the same. Step 1 records it in `dev/TODO.md` as
   `[Reinstatement-Cap-On-Ceded-Agg]`: honor the reinstatement cap on `ceded to` (premium still
   ignored).
9. **Premium in a P&L is optional**, as for a guaranteed-cost layer; a missing premium clause books
   zero with the existing `ZeroPremiumCessionWarning`. (Reinstatements require one, because their
   premium is defined relative to it; the annual terms have no such dependence.)

Also adopted, because they match reinstatements: the annual terms require a
**single occurrence layer** (the 2-D joint is the engine ceiling; per-layer annual terms on a
multi-layer program need one dimension per layer), appear **at most once**, and are **refused on an
aggregate layer**, where the layer itself is the annual deductible and limit (`m xs d`).

## Step `[Annual-Terms-DecL]`: grammar, parse, and the plain `agg`

### Change

- **Grammar** (`src/aggregate/decl.lark`): two keywords, `AAD.2` and `AGGLIMIT.2`, each with the standard lookahead
  `(?![a-zA-Z0-9._:~\-])`, added to the `ID` reserved-word lookahead. A new optional slot in
  `reins_clause` directly after `reins_layer`, before the premium:

  ```
  reins_clause: reins_layer reins_annual reins_premium reins_cede reins_reinst reins_variable as_label
  reins_annual: AAD expr AGGLIMIT expr   -> reins_annual_both
              | AAD expr                 -> reins_annual_aad
              | AGGLIMIT expr            -> reins_annual_limit
              |                          -> reins_annual_none
  ```

  The slot follows the layer because the terms are part of the loss definition, not the economics,
  so `750 xs 250 aad 50 agglimit 500 deposit 100 cede 10% as Layer` reads in loss then money order.
  Add a comment block in the house style of the neighboring rules.
- **Transformer** (`src/aggregate/parser.py`, the reins clause assembly near line 1410): emit
  `occ_reins_aad` and `occ_reins_agglimit` (100% amounts, floats, `inf` for an absent limit, `0` for
  an absent deductible) when present. Validate at parse time, with messages in the style of the
  existing reinstatement checks (line 1449 on): occurrence only; single occurrence layer; at most
  one annual clause; not with `reinstatements`; `aad >= 0`, `agglimit > 0`, both finite except the
  implied unbounded limit.
- **Plain `agg`, `ceded to`** (`underwriter.py`, the plain-`agg` route): rewrite on a **copy** of
  the spec, so the stored program keeps the declaration. Drop the two keys and add an aggregate tier
  `agg_reins = [(1.0, s * m, s * d)]`, `agg_kind = 'ceded to'`, labeled from the terms (for example
  `AAD 50, agglimit 500`). The aggregate stage then **is** the annual layer, so `reins_summary_df`,
  `reins_density_df`, `reins_description` and the rest report it with no new code. A `ceded to`
  program that also carries its own `aggregate` clause raises: the two would need sequential
  aggregate stages, which the engine does not have, and the remedy (write the annual terms into the
  aggregate clause) is one line.
- **Plain `agg`, `net of`:** add the two keys to `_AGG_IGNORED_ECONOMICS_KEYS` and a
  `(('occ_reins_aad', 'occ_reins_agglimit'), 'aad / agglimit')` entry to `_IGNORED_CLAUSE_NAMES`.
  The existing warning text and remedy (fold into a `pnl` by reference) apply unchanged. The `ceded
  to` rewrite above runs before the strip, so only `net of` reaches it.
- **`bvagg`** and portfolio units follow the same two routes (the strip already serves the `bvagg`
  context). Check `_build_pnl_from_port` and make a portfolio P&L with annual terms do whatever it
  does with reinstatements today (refuse, if that is current behavior), and record which.
- **DecL word list** (`src/aggregate/decl_words`, [DecL-Word-List]), the grammar reference regen, the
  cheat sheet (`cheat-sheets/DecL_Cheat_Sheet.tex`, the reinsurance block), and
  `src/aggregate/agg/decl-testers.agg` (one line per form, including a share and each error).
- **Reserved-word vetting** before adding the keywords: `rg -w 'aad|agglimit'` over `src/`, `docs/`,
  `tests/`, the `.agg` corpora and the reproductions book's `reproductions.agg`. Nothing matched in
  `src/` on 2026-10-22.

### Tests (`tests/test_occurrence_annual_terms.py`, new)

- The three forms in the goal parse, with and without a share, `po` amount form included, to the
  expected `occ_reins_aad` / `occ_reins_agglimit` values.
- Each refusal raises with its message: aggregate layer, two occurrence layers, with
  `reinstatements`, negative deductible, zero limit, repeated clause.
- **`ceded to` is exact:** `agg A ... occurrence ceded to 90% po 750 xs 250 aad 50 agglimit 500`
  has `agg_density` equal, array for array, to
  `agg B ... occurrence ceded to 90% po 750 xs 250 aggregate ceded to 450 xs 45`.
- **`net of` is ignored:** the density equals the program without the terms, and one
  `IgnoredDecLClauseWarning` names `aad / agglimit`.
- The stored program of the `net of` build still contains the terms (the re-injection contract).
- Snapshot: `_test_suite.agg` is untouched, so `tests/data/expected_specs.json` must not move; a
  diff there is a finding.

### Acceptance

Tier 2 green; the tests above; the grammar reference regenerated; the word list current.

## Step `[Annual-Terms-PnL]`: the cedant's net in `pnl` / `xpnl`

### Change

- **Terms** (`src/aggregate/contract_terms.py`): a frozen dataclass `AnnualLayerTerms(ContractTerms)`
  in the existing `<Kind>Terms` family, with `target_leg = 'ceded_loss'`,
  `loss_basis = 'occurrence_recovery'`, and fields `limit` (the occurrence limit `s * y`, the name
  and meaning the builders and `check_joint_grid_adequacy` already read off a terms object),
  `annual_deductible` (`s * d`), `annual_limit` (`s * m`, `inf` allowed) and `deposit`. Methods
  `recovery(R) = min((R - annual_deductible)+, annual_limit)`, `reinstatement_premium(R) = 0`
  (vectorized zeros), `phi` aliasing `reinstatement_premium` as `ReinstatementTerms` does. **Name
  hazard:** do not call the aggregate limit `limit`; that attribute is the occurrence fill width in
  every builder. Add it to `__all__` and to the module docstring's feature list (it becomes the
  seventh `ContractTerms` shape, and the docstring's "six features" text changes).
- **Underwriter** (the `pnl` branch near line 1714): pop `occ_reins_aad` / `occ_reins_agglimit`
  before the inner `Aggregate` is built, exactly as `occ_reins_reinst` is popped, so the inner engine
  is the plain occurrence program. When present, build `AnnualLayerTerms` with
  `limit = s * y`, the 100% amounts times `s`, `deposit = econ['pc_occ']`, and route through the
  same branch as reinstatements (`inner.reinstatement_terms`, `kind: 'reins'` recipe, the aggregate
  feature terms per ruling 7). Generalize the branch's error texts from "reinstatements" to "the
  occurrence annual terms" where they now apply to both. Per ruling 9, do not require the base
  premium clause for `AnnualLayerTerms`.
- **Builders** (`src/aggregate/_pnl_builders.py`): no structural change; they already read only
  `recovery`, `reinstatement_premium`, `deposit`, `limit`, `phi` and `target_leg`. Two edits:
  `check_joint_grid_adequacy` also checks the annual kink scales (`annual_deductible` and a finite
  `annual_limit`) against the axis-1 bucket, and the ledger labels name the terms (for example
  `occ 750 xs 250 aad 50` for the recovery row) through `_tier_label` / `_first_reins_label`.
  With `reinstatement_premium` identically zero, the reinstatement premium row should not appear;
  confirm how the builders treat `rates == ()` and match it.
- **Docs:** `docs/3_reference/3_x_PnL.rst` and the reinstatements guide gain a short section on the
  annual terms (the map, the share convention, the coexistence rules); docs are pending a rebuild.

### Tests (added to `tests/test_occurrence_annual_terms.py`)

- **Against reinstatements, exactly.** On one joint, `750 xs 250 deposit P agglimit 1500` and
  `750 xs 250 deposit P reinstatements 1 free` give identical recovery and net legs (same `A(R)`,
  both premiums flat); `agglimit 750` matches `no reinstatements`.
- **Against the pushforward.** The recovery leg's `EX` equals
  `B.transformed_moments(lambda l, r: np.minimum(np.maximum(r - s*d, 0), s*m))['mean']` on the
  builder's own joint, and the net loss leg equals `E[L]` less that.
- **Faces agree.** `pnl` and `xpnl` give the same net margin mean and SD, as for reinstatements.
- **By reference.** `xpnl X P premium less agg.NAME` over the `net of` program from step 1 equals
  the inline `xpnl`.
- **With `cede`.** The commission leg is present and equals `cede * deposit`.
- **Composition.** Annual terms plus an aggregate-tier `pc` build (ruling 7) and the `pc` leg reads
  the net of everything subject, as with reinstatements.

### Acceptance

- The hand-built ledger in section 5 of the occurrence-layer note is reproduced by
  `xpnl Cedant 2000 premium less agg GrossBook 2000 premium at 70% lr 1000 xs 0 sev 120 * lomax 2.5
  occurrence net of 750 xs 250 deposit 326.389307 aad 50 mixed gamma .185`: net margin `EX` 460.70
  and the layer recovery `EX` 187.08, to the joint's grid accuracy (the note uses a bucket of 5; pass
  the same `bs` to compare SDs).
- Tier 3 and the numerics gate green.

## Files

Step 1: `decl.lark`, `parser.py`, `underwriter.py`, `decl_words`, the grammar reference, the cheat
sheet, `decl-testers.agg`, the new test file, and the `[Reinstatement-Cap-On-Ceded-Agg]` entry in
`dev/TODO.md`. Step 2: `contract_terms.py`, `underwriter.py`, `_pnl_builders.py`, the docs pages,
the test file. Both: `__version__`, `CHANGELOG.md` (one
paragraph each), `dev/TODO.md`, `dev/FEATURES.csv` via `dev/regen_features.py`; this plan moves to
`dev/done/` in the commit that lands step 1.

## Follow-ups, out of scope

- `[Reinstatement-Cap-On-Ceded-Agg]`: honoring the reinstatement cap on `ceded to` in a plain `agg`
  (ruling 8).
- Section 5 of the occurrence-layer note can then use the DecL form in place of the hand-built
  ledger (outside the repository).
- Annual terms on more than one occurrence layer needs a joint of dimension one more than the
  number of decorated layers; not planned.
