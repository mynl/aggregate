# [Decommission-Analysis-Classes] — strip `VariableRatingAnalysis` & `ReinstatementAnalysis`

**Status:** ✅ DONE (`1.0.0a144`). Executed with one deviation from the written
test plan: the `*_decl.py` regression suites depend on `.analysis` far more
heavily than the plan's line-strip assumed, so (with author sign-off) their
number/behavior assertions were **rehomed** onto the PnL's own surface —
terms → `p.engine.reinstatement_terms` / `p.engine.variable_terms`, leg
stochasticity → `stats_df` leg SDs, exact means → `p._source.transformed_moments`
off the `(L, R)` joint — rather than kept untouched. Also dropped the now-dead
`variable_gross_expense` assignment (its only reader was the deleted method).

## [Why] rationale

The generic PnL (a signed group ledger over `GridDistribution`s, assembled in
`_pnl_builders.py`) made both analysis classes redundant. The domain math already
lives on the *terms* objects — `ContractTerms.phi` (swing/slide/pc/corridor),
`ReinstatementTerms.recovery`/`reinstatement_premium` — and the source
construction lives on `Aggregate.occ_bivariate`. The builders read *that*
directly; the analysis objects are constructed, attached as `pnl.analysis`, and
(for variable rating) never read at all.

Established facts (from the two investigations, this session):
- `build_variable_pnl` and its helpers (`_feature_maps`, `_build_variable_*`)
  **never touch `VariableRatingAnalysis`** — they rebuild every leg from
  `agg.variable_*` + `terms.phi`. Deleting the class changes nothing.
- `build_reinstatement_pnl` / `_build_reinstatement_consolidated` **do** read the
  reinstatement analysis, but only nine plain `__init__` attributes plus one
  method (`_agg_tier_maps`, 40 lines of pure glue over four values). The
  `.source` joint is built by `occ_bivariate`, not the class. Everything else
  the class offers (`validation_df`, `tail_df`, `summary_df`, `info`, `plot`,
  `reins_explanation`, the ~16 leg pushforwards) is drill-down no PnL exhibit
  consumes.
- Neither class is re-exported (submodule-access-only); no top-level API break.

Note (parked, not this plan): occurrence-*basis* variable features
(swing/slide/pc/corridor on the cession) are 2-D and **not implemented** (they
error today). Reinstatements are the only occurrence-basis stochastic feature
built. When occ-basis variable rating lands, it belongs in the builders +
`occ_bivariate` (the 2-D path), **not** a revived analysis class.

## [Invariant] the safety net

**Decommissioning must change zero PnL numbers.** The builder paths are unchanged;
we only stop constructing/reading a redundant object and relocate one helper. The
regression guard is the existing DecL number assertions in
`test_variable_rating_decl.py` and `test_reinstatement_decl.py` — every
`p.stats_df` / `p.economics` / leg-mean assertion must pass **untouched**. Any
diff there means the refactor changed behavior and is wrong. Run the full suite
(`-m 'slow or not slow'`, includes the bivariate reinstatement suite) at each
phase boundary.

## [Keep-Delete-Boundary]

### `src/aggregate/variable_rating.py` — DELETE the whole file
Nothing in it survives: `VariableRatingAnalysis` (53-363), `_layer_ceder`
(40-50), `SUMMARY_PERCENTILES` (37). Its only import is `ContractTerms` (stays).

### `src/aggregate/reinstatement.py` — DELETE THE WHOLE FILE (relocate 3 keepers first)
Reinstatements don't warrant their own module ([Decisions] #4). Relocate the three
keepers to where they belong, then delete the file:
- `ReinstatementTerms` (93-370) → **fold into `contract_terms.py`** alongside its
  `ContractTerms` siblings (RetroTerms / SwingTerms / SlideTerms /
  ProfitCommissionTerms / CorridorTerms). It already belongs there:
  `contract_terms.py:9` lists it as family and `:90` notes its validation was
  "generalized from ReinstatementTerms." Add to `contract_terms.__all__`.
- `check_joint_grid_adequacy` (49-90) + `JOINT_KINK_MIN_BUCKETS` (46) → **move to
  `_pnl_builders.py`** (its only post-decommission consumers: `_pnl_builders.py:533,849`;
  the third caller, `_aggregate.py:1240`, is inside the deleted method).
- `SUMMARY_PERCENTILES` (38) → **delete** (only the deleted method used it, via
  `_aggregate.py:1208`).
- **DELETE:** `class ReinstatementAnalysis` (373-964), then the now-empty file.
- Repoint importers/refs: `underwriter.py:1183` (import), `parser.py:1368`
  (comment), `_aggregate.py:1164,1179,1194` (docstrings), and the doc cross-ref at
  `constants.py:188` (→ new home of `JOINT_KINK_MIN_BUCKETS`).

### `src/aggregate/plots/_aggregate.py` — DELETE `plot_reinstatement` (259-…)
Its only caller is `ReinstatementAnalysis.plot`.

## [Phase-1-Var-Analysis-Strip] delete VariableRatingAnalysis (pure removal)

Nothing consumes it, so this is a clean excision.

1. `underwriter.py:1756-1766` (`kind == 'var'` dispatch): delete
   `analysis = inner.variable_rating_analysis()` (1758) and
   `face.analysis = analysis` (1765). Keep the `variable_gross_expense` set (1756)
   and the `build_variable_pnl(...)` call — the builder is the real engine.
   `face.engine = inner` (1766) stays.
2. `_aggregate.py:1118-1154`: delete `Aggregate.variable_rating_analysis`
   (judgment call [JC-Public-Methods] — this is a public method).
3. Delete `src/aggregate/variable_rating.py`.
4. Tests: delete `test_variable_rating_analysis.py` (8 fns, tests the class
   directly); in `test_variable_rating_decl.py` remove the
   `isinstance(p.analysis, VariableRatingAnalysis)` assertions (lines 59, 173) and
   the import (27) — **keep every number/behavior assertion** (they test the
   builder path).
5. `test_composition_matrix.py`: update expected-type entries naming
   `'VariableRatingAnalysis'` (243, 245) — see [Phase-4].

Gate: full suite green (numbers unchanged).

## [Phase-2-Reins-Analysis-Relocate] move the two load-bearing pieces off the class

Before the class can go, relocate what the builder genuinely needs. Two things:

**(a) `_agg_tier_maps` → free function.** Move `reinstatement.py:560-599` to a
module-level `agg_tier_maps(terms, agg_recovery, agg_ceded_premium,
agg_feature_terms) -> (rec, prem, comm)` (home: `_pnl_builders.py`, next to its
two callers, or on `ReinstatementTerms`). It needs exactly those four values and
nothing self-referential — a mechanical lift. Update the two call sites
(`_pnl_builders.py:972,1049`) to pass the four values (all already in scope on
the dispatch) instead of `analysis._agg_tier_maps()`.

**(b) The source (`.source` joint) + ceder construction.** The joint,
`agg_recovery` ceder, and grid check currently live in
`Aggregate.reinstatement_analysis` (1235-1251). Relocate that orchestration to a
helper the builder calls — proposed `build_reinstatement_source(agg, terms,
gross_premium, agg_reins, bs, log2_x, log2_y) -> (joint, agg_recovery,
joint_aggregate)` in `_pnl_builders.py` (or a slim private `Aggregate` method).
It does: validate the single `occ_reins` layer (1209-1218), build
`occ_bivariate(views=('gross','ceded'), …)` (1235-1236), `check_joint_grid_adequacy`
(1240-1243), build the `agg_recovery` ceder via `_reinsurance.make_ceder_netter`
(1248-1251). Pure move; identical output.

After (a)+(b), the reins dispatch (`underwriter.py:1778-1792`) calls
`build_reinstatement_pnl(inner, source=…, terms=…, gross_premium=…, econ=…,
agg_feature_terms=…, gross_expense=…, occ_commission=…, agg_commission=…,
agg_ceded_premium=…, agg_recovery=…)` directly — the same nine values it used to
hang on the analysis object, now passed as args.

3. Rewrite `build_reinstatement_pnl` (892-1015) and
   `_build_reinstatement_consolidated` (1018-1090): replace every `analysis.<attr>`
   read (enumerated at 945-947, 960-961, 967-973, 980-981, 988; 1039-1049, 1065,
   1071) with the corresponding parameter, and `analysis._agg_tier_maps()` with
   `agg_tier_maps(...)`. Change the signatures from `(agg, analysis, …)` to take
   the explicit inputs. Drop `pnl.analysis = analysis` (991, 1068) — see
   [JC-Analysis-Attr].

Gate: full suite green — **especially** `test_reinstatement_decl.py` (25 fns) and
the bivariate reinstatement suite. Numbers must be identical.

## [Phase-3-Reins-Analysis-Delete] remove the class, its exhibits, and the module

1. Relocate the three keepers ([Keep-Delete-Boundary]): fold `ReinstatementTerms`
   into `contract_terms.py`; move `check_joint_grid_adequacy` +
   `JOINT_KINK_MIN_BUCKETS` into `_pnl_builders.py`. Repoint every importer/ref
   (`underwriter.py:1183`, `_aggregate.py`, `constants.py:188` doc ref, the
   Phase-2 source helper's call site).
2. Delete `class ReinstatementAnalysis`, then the now-empty `reinstatement.py`.
3. Delete `plot_reinstatement` (`plots/_aggregate.py:259-…`).
4. `utilities.py` `qd`: remove the `isinstance(x, ReinstatementAnalysis)` branch
   (320, 346-353). The PnL has its own `qd` branch; **xpnl is the drill-down**.
5. Delete `Aggregate.reinstatement_analysis` (`_aggregate.py:1156-1259`); its
   source-building moved to the Phase-2 helper.
6. Remove the `pnl.analysis` attribute + docstring (`_pnl.py:770, 765-769`);
   confirm the `pnl.analysis = …` assignments (`_pnl_builders.py:991,1068`) were
   already dropped in Phase 2.
7. Tests: delete `test_reinstatement_analysis.py` (14 fns) and
   `test_reinstatement_exhibit.py` (8 fns, the display surface being removed); in
   `test_reinstatement_decl.py` strip the `.analysis`-type assertions (128, 178,
   246) and import (20), keep the numbers. KEEP untouched:
   `test_reinstatement_terms.py`, `test_reinstatement_pushforward.py`,
   `test_variable_rating_terms.py`.

## [Phase-4-Tests-Docs-Features] curation & housekeeping

1. `dev/FEATURES.csv`: drop columns 10 (`ReinstatementAnalysis`) and 11
   (`VariableRatingAnalysis`); delete now-orphaned rows
   `variable_rating_analysis` (179) and `reinstatement_analysis` (178) if the
   methods are removed; clear the `Y` marks those columns held on shared rows
   (`terms`, `gross_premium`, `source`, `tail_df`, `validation_df`,
   `reins_explanation`, `info`, `plot`, `figure`, `scale`, `bs_*`, `density`).
   Bump the `# table-version` stamp.
2. `dev/regen_features.py`: remove the two objects from `build_objects`
   (82-88), and the two entries from `CLASS_COLS`/`SHORT` (53-58). Re-run the
   audit → expect 0 mismatch / 0 stale.
3. `test_composition_matrix.py`: update the expected-type table (147, 243-249)
   to the post-decommission reality (the composed cells return `PnL`; the
   analysis-type column is dropped).
4. (`pnl.analysis` attribute + `_pnl.py` docstring removed in Phase-3 step 6.)
   FEATURES row `state,analysis` (165) drops with it.
5. Cookbook: `docs/cookbook/_05_03_reinstatements.qmd` and `_05_04_variable_rating.qmd`
   still reference `.analysis` / `reins_explanation` in TODO prose — repoint to
   the PnL's own exhibits when those pages are fleshed (not blocking).
6. `dev/TODO.md`: mark the decommission done; move this plan to `dev/done/`.

## [Decisions] — resolved with author (this session)

1. **[JC-Public-Methods] → remove both.** Delete `Aggregate.variable_rating_analysis`
   and `Aggregate.reinstatement_analysis`; the DecL path is the only entry.
2. **[JC-Analysis-Attr] → remove it.** `pnl.analysis` goes entirely (attribute,
   `_pnl.py` docstring, FEATURES row 165). No `None` placeholder.
3. **[JC-Reins-Exhibits] → drop all.** `validation_df`, `tail_df`, `summary_df`,
   `info`, `plot`, `reins_explanation` all go with the class. **No** standalone
   pushforward-audit function kept — **xpnl is the drill-down**, and
   `check_joint_grid_adequacy` (KEPT, moved to `_pnl_builders.py`) stays the grid
   guard.
4. **[JC-Module] → delete `reinstatement.py` entirely.** No `reinstatement_terms.py`
   rename: fold `ReinstatementTerms` into `contract_terms.py` (with its siblings —
   it was only ever stranded there), move `check_joint_grid_adequacy` +
   `JOINT_KINK_MIN_BUCKETS` to `_pnl_builders.py`, delete the file.

## [Housekeeping]
Plan-based code change → bump `pyproject.toml` `1.0.0a*`, add a `CHANGELOG.md`
section, bump the FEATURES.csv `# table-version` stamp. Commit at phase
boundaries (Phase 1 is independently shippable; Phases 2-3 land together).
