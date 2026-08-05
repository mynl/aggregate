# Plan [Exhibits-Module]: business exhibits, stats frames to greater_tables IR

> **Status: REVISED 2026-08-05 after the author's review of the executed phases.** Phases 1 to 4a are SHIPPED (`1.0.0a200`, `a201`, `a203`; app `1.0.0a39`). This revision folds in the review decisions: the PnL frame rename, the package split, the economic exhibits including the waterfall, and the app's envelope-only consolidation. Purely additive to the core (exhibits import from core, core never imports exhibits), provisional at 1.0. Companion plan: `dev/plan-chart-ir.md`, executing in parallel. Both derive from the author's design notes on the aggregate to aLL (aggregate_api) interface.

## Principle and placement test

aggregate owns meaning, the app owns arrangement. Test for placement: if deleting the web app would destroy knowledge an actuary would want in a notebook, that knowledge is in the wrong place. Today it leaks: aggregate_api carries ROW_FLAGS and FORMATS (`tables.py:113`, `:132`), `_drop_raw_moments` (`objects.py:1407`), exhibit titles and captions (`main.js:646-655`), and the reinsSeries survival accumulation. The FCC stats frames are raw materials; the business translation that makes them resonate belongs in the library. Exhibits are allowed to be domain specific; core aggregate stays domain agnostic.

**Layout never moves into the library** (author, 2026-08-05, confirming the founding principle against the temptation below). The app's tab shape maps almost one to one onto the exhibit names, which is evidence the vocabulary is right, not a reason to let the library hint at grouping. No `group` or `tab` field on `Exhibit`, now or later.

**Scope of the business exhibits: show, do not exhaust** (author, 2026-08-05). There will always be more economics worth exhibiting, and most of it is a business decision this library should not be making for its users. The library ships two or three exemplary business exhibits, done properly, and the open registration is how everyone else builds their own. `economic_waterfall` is the one flagship the author is prepared to specify.

## Decisions taken

Numbered decisions from 2026-08-04 unless dated otherwise.

1. GT dependency: optional extra `exhibits = ["greater_tables>=6.0.0a8"]` with lazy import inside the IR conversion step. `import aggregate` and `import aggregate.exhibits` never touch greater_tables; a clear ImportError names the extra when missing. **Status 2026-08-05:** the extra is written but commented in `pyproject.toml`, because greater_tables 6 is not yet on PyPI (5.3.0 is the latest published) and an active unresolvable extra breaks `uv sync --all-extras`. GT 6 publishes shortly; uncomment verbatim on the day, no other change needed.
2. The perspective enum is named `Perspective`, kwarg `perspective=`, URL `?perspective=`. This avoids the three existing senses of "view" (the reins frame column levels, the app's REINS_VIEWS, the app's table-view toggle). Values are RAW, INSURED, INSURER, REINSURER (author, 2026-08-05, replacing the earlier RAW/BUYER/SELLER: buyer and seller are relative and confusing, since the insured buys insurance, the insurer sells insurance while buying reinsurance, and the reinsurer sells reinsurance while buying retro).
3. RAW serializes as a TableDoc like every other perspective: one client path, uniform ETag, MultiIndex structure survives.
4. Signature default is `Perspective.RAW`: passthrough always works, translation is opt in.
5. v1.0 implements RAW and INSURER only. INSURER defaults to RAW unless an override is registered for the (exhibit, type) pair. The override is custom per exhibit and unbounded in scope: some are identity (no override at all), some thin, and some extensively different from the raw frame. INSURED and REINSURER stay in the enum as stable vocabulary with no 1.0 registrations.
6. **Signs stay as booked** (author, 2026-08-05). Signed values throughout, never flipped to positive magnitudes for readability. It takes getting used to and is the easiest long run, and it is the only way the ledger and the kappa ladder foot. greater_tables stamps `neg` cell flags automatically, so the renderer can style the sign without the library lying about it.
7. **The API always serves the wrapped envelope, never a bare TableDoc** (author, 2026-08-05). One client path for every table in the app. This retires `GET /frame/{which}?format=ir`, whose whole job is what an exhibit at RAW perspective already does.
8. **Three consumer-facing API paths, each with one job** (author, 2026-08-05): bulk raw JSON for the large frames (the densities, paginated, feeding the grid and the charts), exhibit envelopes for everything curated, and CSV for non-aLL consumers who want raw values rather than a presentation. The CSV path is kept deliberately and permanently.
9. **`exhibits` becomes a package mirroring `plots/`** (author, 2026-08-05). Already overdue: the single module hit 950 lines, past the 800-line trigger this plan set for itself, before the PnL work and the waterfall land.
10. **Empty frames are legal and render** (author, 2026-08-05). Verified: `gt.build(pd.DataFrame())` produces a valid 0 by 0 document and a frame with columns but no rows keeps its columns. The library returns an empty frame rather than None or a raise where a frame does not apply, and the insurer view explains the emptiness in its caption. The app's `frame_document` refuses empties ("nothing to render"), which is one more reason it retires under decision 7.

## Background facts

greater_tables 6.0.0a8 IR is `TableDoc` (frozen pydantic, `ir_version` 1, JSON Schema, `canonical_dict` / `canonical_json` / `doc_hash` / `stamp`; byte deterministic). GT has no "block" concept, the unit is one whole table, so a multi table exhibit is a list of TableDocs. `build(df, TableSpec) -> TableDoc` is the conversion entry; TableSpec carries caption, notes, formatters (sugar strings such as `',.1%'`), row_flags, cell_flags, include_raw, max_rows, sparsify. Row and cell flag vocabulary is `total`, `subtotal`, `emphasis`, `muted`, plus the automatic `neg`. The app already renders TableDocs end to end (server `tables.py: frame_document`, ETag = doc hash, client walker `renderTable` plus `irToGridInput`). The library side has the FCC contract (`constants.py:96-151`), `dev/FEATURES.csv`, `dev/reporting-guidelines.md`, and LabeledMixin relabeling at the serve step.

## Library design: the `aggregate/exhibits/` package

Not star imported in `__init__.py` (the Tweedie / Pentagon / pedagogy precedent); users write `from aggregate import exhibits`. The package split (decision 9) mirrors `plots/` exactly, and the public import path is unchanged by it.

```
exhibits/__init__.py      machinery + public surface (Perspective, Exhibit, registry,
                          available_exhibits, exhibit_frames, build_exhibit,
                          register_simple_exhibit, shared flag helpers)
exhibits/_aggregate.py    Aggregate registrations
exhibits/_portfolio.py    Portfolio registrations
exhibits/_pnl.py          PnL registrations, including the economic family
exhibits/_bivariate.py    BivariateAggregate registrations
exhibits/_distortion.py   Distortion registrations
```

Public surface, all reachable as `aggregate.exhibits.X`:

- `Perspective(Enum)`: `RAW`, `INSURED`, `INSURER`, `REINSURER`. Only RAW and INSURER are implemented at 1.0 (decision 5).
- `Exhibit` frozen dataclass: `ir_blocks` (TableDocs), `name`, `title`, `perspective`, `meta`. `to_payload() -> dict`; `hash` property (sha256 over the concatenated block `doc_hash` values, first 12 hex) for ETags.
- Generic exhibit functions via `functools.singledispatch`, one canonical implementation per (exhibit, perspective) pair. Signature `f(obj, perspective=Perspective.RAW) -> Exhibit`. Each carries two open registries: `f.register` / `f.frames` for the RAW frames builder, and `f.insurer` for the per (exhibit, type) override hook whose default is identity. The base raises NotImplementedError naming the type. Registration is open: app or user code may register new types, which is how users build their own economics (see the scope note above).
- `exhibit_frames(obj, name, perspective) -> list[tuple[str, DataFrame, dict]]`: the pure pandas frame stage returning (block name, translated frame, TableSpec kwargs). Testable and usable without GT installed. This is the layer to review a translation.
- `build_exhibit(obj, name, perspective) -> Exhibit`: registry lookup, frame stage, then GT `build(df, TableSpec(**kw))` per block. This and `to_payload` are the only lazy import sites.
- `EXHIBITS: dict[name, (generic_fn, perspectives_fn)]` where `perspectives_fn(obj) -> list[Perspective]` is the per object availability predicate. `available_exhibits(obj) -> list[tuple[str, list[Perspective]]]` derives from the singledispatch registries (MRO hit) plus the predicates, so it cannot go stale. Needs no GT import.
- **`register_simple_exhibit(name, title, frame_attr, classes, *, predicate=None)`** (new, author ask 2026-08-05): declare a passthrough exhibit over one named frame in a single line, generating the exhibit function, the registry entry and the per class frames builders together. It registers no insurer override, so INSURER equals RAW automatically by the default rule, which is exactly the "raw, and insurer the same" behavior wanted for the diagnostic frames. Example: `register_simple_exhibit('bs_window', 'Grid sizing', 'bs_window_df', [Aggregate, Portfolio, BivariateAggregate])`.

All served frames pass through LabeledMixin `_relabel` (honoring `use_labels` and `renamer`); titles use `_title_name` ("{label} ({name})").

## The PnL frame rename ([PnL-Economic-Frames], breaking)

Author decision 2026-08-05, arising from the review finding that `stats` and `pnl_ledger` produced **byte-identical documents** on a PnL (same hash, same single block). Two exhibit names for one frame is a synonym, against the one-canonical-name rule.

The root cause is that `stats_df` means four different things across the FCC contract. On Aggregate and Portfolio it is the (component, measure) by view moment store; on Distortion a column of D_g statistics; on bvagg a (basis, stat) by axis table; on PnL it is not a statistics frame at all but the ledger sheet with the kappa scenario ladder. The FCC audit can only check that the name exists, never that it means the same thing.

The rename:

| now | becomes | note |
|---|---|---|
| `PnL.stats_df` (the ledger) | `PnL.economic_df` | names what it is |
| `PnL.ratio_df` | `PnL.economic_ratios_df` | not a view of `economic_df`: it needs `Leg.kind` to split expense from commission, and the per atom vectors for the means-of-ratios columns, neither of which survives into the ledger sheet |
| absent | `PnL.stats_df` delegating to `self.engine.stats_df` | makes `stats_df` mean one thing everywhere: the moment store of a book |
| `PnL.validation_df` | **unchanged** | author correction 2026-08-05: PnL's own leg rebucketing audit stays, it is a real check on the P&L's own construction and does not delegate |

`PnL.engine` is `None` on a hand-built kernel P&L (`_pnl.py:858`), so `stats_df` returns an empty `pd.DataFrame()` there, per decision 10. Likewise `validation_df` is legitimately empty when every leg is exact (only `bs > 0` legs are audited), which the insurer caption states in words rather than rendering a blank table.

"Economic" already appears in the codebase as prose, naming the Gross / Ceded / Net presentation in the reins frames. No identifier collides (`economic_df`, `economic_ratios_df` are both free). The overlap is accepted deliberately: both senses are the accounting reading of a frame, and the family will grow.

This is a breaking change on a public surface, so it lands as **its own labelled version bump ahead of everything else**, with `dev/FEATURES.csv`, the `qd` PnL branch, `docs/`, and the app's `_CSV_FRAMES` / route resolvers moved in the same commit.

## The economic exhibits

Three, all on PnL, all in `exhibits/_pnl.py`. Frame names and exhibit names mirror each other, restoring the property that holds everywhere else (`summary_df` to `summary`, `tail_df` to `tail`, `dependency_df` to `dependency`).

- **`economic`**: the ledger, source `economic_df`. RAW passthrough. INSURER adds the caption explaining the kappa semantics (scenario states, not per row quantiles; the ladder foots; `κ01` is the adverse state under the payoff convention; plain `P` headers mean no shared atoms and a marginal, non-footing ladder) and row flags (grand result `total`, group and tier results `subtotal`, running nets `muted`).
- **`economic_ratios`**: sources `economic_ratios_df` plus `legs_df`. RAW passes both through. INSURER splits into pure blocks per the reporting guideline, one unit per column: amounts (P, L, E, C, M, signed in the gross direction, captioned that `M == P - L - E - C` foots identically), ratios (LR, ER, CR, E_LR, E_ER, E_CR, P_share, M_share, declared as `ratio_cols` so they render as percentages, captioned on ratios-of-means versus means-of-ratios and when the two part company), and legs unchanged.
- **`economic_waterfall`**: the flagship. Specified below.

### `economic_waterfall` specification (author, 2026-08-05)

The walk from gross margin, through the amount ceded at each layer, to net margin, with the margin evaluated at every stop. **Everything it needs is already computed**; the exhibit is arithmetic over existing columns, which is why it lives wholly in the exhibit layer and adds no frame columns.

One row per step (each group, each tier subtotal, the closing net). Columns:

| column | definition | source |
|---|---|---|
| Premium spent | the step's P over the gross block's P | `economic_ratios_df.P_share` |
| Margin spent | the step's M over the gross block's M | `economic_ratios_df.M_share` |
| CR | combined ratio at the step | `economic_ratios_df.CR` |
| M / SD(M) | margin over the standard deviation of the step's own margin | `economic_df.SD` on the step's margin row |
| M / 100yr standalone | `M / -M_100`, where `M_100` is the step's own 1-in-100 margin | `density_df[row].q(0.01)` |
| M / 100yr diversified | `M / -M_100`, where `M_100` is the step's margin **conditional on the whole book's 1-in-100** | `economic_df` column `κ01` |

The diversified column is the point of the exhibit: `κ01` is `E[row | grand result at its 1st percentile]`, so the diversified walk **foots down the sheet exactly**, while the standalone column does not (standalone tail measures do not add). Showing the two side by side is the diversification benefit, made visible, per layer.

Three constraints, all settled by the author 2026-08-05:

1. **The ratio is `M / -M_100`.** `M_100` is a bad outcome and therefore negative, so `-M_100` **is the capital you need to inject** at that return period, and the ratio reads directly as margin over required capital, positive for a sound book. Not the literal `M / M_100`, which would come out negative. The same definition serves both bases; only the source of `M_100` changes.
2. **The kappa ladder is not always present, and blanks when absent.** When a ledger has no shared atoms (the massive one-sweep route, the stitched guaranteed-cost tower) the columns fall back to marginal `P01…P99` and no conditioning ever happened. The diversified column blanks there, exactly as the frequency percentiles blank in `summary_df`: the value is not missing, it is undefined on that route, and the caption says so. The standalone column always works.
3. **It needs a tower with steps.** The exhibit is for an `xpnl` walk; a single group P&L has one margin row and no walk to draw. Availability gates on the tower, the way `reins` gates on a cession, and the chip grays out rather than showing a one row waterfall.

Orientation follows decision 6 (signed as booked): a purchased layer's margin is negative by construction and stays that way, so the walk foots. `1-in-100` is `p = 0.01` for a payoff, per `period_to_p`'s downside mapping, which is the same `p` the `κ01` header names.

## Exhibit inventory (revised)

| name | source frame(s) | kinds | insurer treatment |
|---|---|---|---|
| summary | `summary_df` | all five | caption, total / subtotal flags on agg and port; identity elsewhere |
| tail | `tail_df` | agg, port | caption, 1-in-200 and 1-in-250 anchors emphasized, portfolio total flagged |
| stats | `stats_df` | all five (PnL via the engine, post rename) | drops raw noncentral moments (ex1/ex2/ex3) on agg and port; identity elsewhere |
| validation | `validation_df` | all five | emphasis on failing rows (Validation flags on agg and port, `Pass == False` on distortion and bvagg); PnL caption explains an empty audit |
| reins | `reins_stats_df` + `reins_summary_df` | agg, port when ceding | raw moment drop, captions, portfolio total flags |
| dependency | `dependency_df` + `axis_support_df` | bvagg | none registered, none needed |
| economic | `economic_df` | pnl | kappa captions, footing rules, ledger row flags |
| economic_ratios | `economic_ratios_df` + `legs_df` | pnl | three pure blocks, ratio columns declared |
| economic_waterfall | derived from `economic_df` and `economic_ratios_df` | pnl with a tower | the exhibit **is** the translation; RAW serves the same table |
| bs_window | `bs_window_df` | agg, port, bvagg | none (via `register_simple_exhibit`) |
| tail_behavior | `tail_behavior_df` | agg, port | none (via `register_simple_exhibit`) |

Not exhibits, deliberately: `density_df`, `unit_density_df`, `reins_density_df`, `sev_density_df`. These are bulk paginated data feeding the grid and the charts, and they stay on the raw JSON path (decision 8).

`economic_waterfall` is the one exhibit whose RAW and INSURER agree while still being a full translation, because the exhibit itself is the business object; there is no underlying frame to pass through.

## App API surface (aggregate_api)

Landed at app `1.0.0a39`: `GET /v1/objects/{oid}/exhibits` (a passthrough of `available_exhibits`) and `GET /v1/objects/{oid}/exhibit/{name}?perspective=raw|insurer` (the envelope `{name, title, perspective, meta, blocks, hash}`, ETag from the exhibit hash, 304 on If-None-Match, 404 listing the capability set, 400 on an unsupported perspective). New library exhibits appear with zero endpoint changes.

Still to do, per decisions 7 and 8: retire `GET /frame/{which}?format=ir` in favor of exhibit envelopes; retire the `xxx_df` JSON routes for frames an exhibit now covers, keeping the bulk density routes; keep `/frame/{which}.csv` permanently for non-aLL consumers; delete `_drop_raw_moments` and the migrated ROW_FLAGS, FORMATS and caption literals once the client reads envelopes.

`dev/scripts/check-exhibits.py` (app, landed a39) sweeps every exhibit across every kind and asserts the capability and envelope routes agree, that envelopes are byte deterministic, and that the insurer blocks match the frame routes row for row while both exist. It is the regression net for the consolidation and belongs in CI.

## Menu and the perspective switch

The page shape is app content and authored, never derived from the library (see the principle section). The author's intended shape, 2026-08-05:

**summary | economics | reinsurance | price/evaluate | more**

Each tab carries both plots and exhibits. `more` holds the diagnostics: stats, validation, tail behavior, grid sizing. **The page takes a single raw / insurer switch at the top**, so the perspective is page level state and every exhibit on the page changes together. That works uniformly because INSURER is total by the default rule: every exhibit answers at both perspectives, whether or not it registers an override.

Note for planning: `price/evaluate` has no exhibits to serve at 1.0. Pricing exhibits are deferred (see below), so that tab carries plots only until the upstream asks land.

The house rule holds throughout: never hide, gray out. A chip whose capability is absent renders grayed and disabled with an explanatory title.

## Tests

Library, `tests/test_exhibits.py`: one small object per kind via `build()`; `available_exhibits` shapes and predicates; frame stage structure via `exhibit_frames` (indexes, dropped rows, relabeling honored), all GT free. Under `pytest.importorskip('greater_tables')`: `build_exhibit` returns TableDocs, and committed `canonical_dict` JSON snapshots per (exhibit, perspective, kind) guard both the translation and IR drift, regenerated by `tests/capture_exhibit_snapshots.py`. The snapshot file drives the case list, and a coverage test asserts it matches the live registry. `tests/test_plots_boundary.py` carries the two lazy-import guards. Fixture programs live in `decl-testers.agg` section EX. `dev/FEATURES.csv` and `dev/TODO.md` stay current.

The waterfall needs numeric tests beyond the snapshots: that the diversified column foots down the sheet (it must, by the kappa construction) and that the standalone column does not, which is the exhibit's whole thesis stated as an assertion.

## Phases and cadence

Phases 1 to 4a are shipped. Each remaining library phase bumps `1.0.0aNNN` with a CHANGELOG section and a one line commit `[<Label>] aNNN: summary`; app phases follow the app's own conventions. Version numbers interleave with the parallel `[Chart-IR]` plan, so check `git log` and the CHANGELOG before claiming one.

1. ~~**[Exhibits-Scaffold]**~~ SHIPPED `a200`.
2. ~~**[Exhibits-Stats-Validation]**~~ SHIPPED `a200`.
3. ~~**[Exhibits-Reins-Insurer]**~~ SHIPPED `a201`.
4. ~~**[Exhibits-PnL-Translation] raw stage**~~ SHIPPED `a203` (`pnl_ledger`, `pnl_ratios` as RAW passthroughs; both superseded by phase 6 below).
5. ~~**[PnL-Economic-Frames]**~~ SHIPPED `a204` (library, breaking): the rename table above, plus `stats_df` delegating to the engine with the empty frame fallback. Landed with `_pnl.py`, `qd`, the exhibit registrations, `FEATURES.csv`, docs, 148 test references and the app's `_CSV_FRAMES` in one commit.
6. ~~**[Exhibits-Package-Split]**~~ SHIPPED `a205` (library): `exhibits.py` became `exhibits/` mirroring `plots/` (`_core.py` plus one module per class plus the manifest in `__init__.py`); `register_simple_exhibit` added and seven hand-written passthrough builders collapsed into five manifest lines; `pnl_ledger` and `pnl_ratios` renamed `economic` and `economic_ratios`; `bs_window` and `tail_behavior` added. Snapshots 78. `check-exhibits.py` now reads the exhibit list from the library.
7. ~~**[Exhibits-Economic-Insurer]**~~ SHIPPED `a206` (library): the `economic` and `economic_ratios` insurer treatments per the section above, plus `MEASURE_FORMATS` (`CV` as `.1%`, `Skew` fixed) applied wherever a measure is a column. Open question 2 is now half answered: the formats work on the summary card and the ledger, and remain impossible on the moment store.
8. ~~**[Exhibits-Waterfall]**~~ SHIPPED `a207` (library): `economic_waterfall` to the specification above, in two blocks (`walk` currency, `evaluation` ratios), tower gated, diversified column blanking on the stitched route, capital as `M / -M_100` blanking where no capital is called for. The footing thesis is asserted in the tests. **GATE STILL OPEN:** the author reviews the first rendered output and rules on the two questions in the open list below. Base case is in, iterate from here.
9. **[Exhibits-App-Consolidation]** (app): the client reads envelopes; `applyKindGating` and the hardcoded NA tables read the capability response; the page gains the raw / insurer switch; `/frame/{which}?format=ir` and the covered `xxx_df` routes retire; `_drop_raw_moments`, ROW_FLAGS, FORMATS and the `main.js` caption literals are deleted; `check-exhibits.py` goes into CI. The CSV route and the bulk density routes stay.

## Considered and declined: per-step grossnet joints for peel kappas (author, 2026-08-05)

A peeled multi-layer occurrence walk serves a **marginal** `P` ladder rather than a scenario `κ` ladder, because the stitched route supplies each row from its own engine marginal with no shared atoms. It is worth recording exactly why, because the obvious fix works and is still not worth doing.

The kappa ladder does **not** need the simultaneous joint law of every row. Each cell is `E[row_i | anchor = x_q]`, a pairwise question about `(row_i, anchor)`, and the columns foot by linearity of conditional expectation: if `T = L1 + L2` then `E[T | A] = E[L1 | A] + E[L2 | A]` however each term was obtained, so long as every cell conditions on the same event. So pairwise joints against the anchor are sufficient, and a `grossnet` bivariate per step supplies them: axis 0 is the position net of layers `1..n`, axis 1 the position net of layers `1..m` with `m > n`, so the difference between the axes is that step's own cession.

**Declined on compute, not on correctness.** A `k` layer peel means `k` bivariate builds, each the size of the single joint a one-layer program already pays for. That is a large amount of calculation for a presentational gain, so the peel keeps its marginal ladder and the `economic_waterfall` blanks its diversified column there, saying so in the caption. Revisit only if the bivariate build gets much cheaper.

Related and already true: an **aggregate** cover needs no joint at all (it is a function of the gross aggregate, so the source stays a 1-D `GridDistribution`), while an **occurrence** cover needs the `(gross, ceded)` bivariate because the ceded-occurrence aggregate is not a function of the gross aggregate. Both serve `κ`.

## Explicitly deferred

INSURED and REINSURER implementations (the reinsurer semantics review is that work's opening gate: does the reinsurer see Ceded relabeled as its gross, does Net render at all, does cede and assume swap on a P&L); the retro perspective; pricing exhibits, which are parameterized by distortion calibration and computed in POST routes and so do not fit the parameter free GET envelope, parked until the app's upstream asks land (a `density=` or `basis=` kwarg on `calibrate_distortions`, and a public `GridDistribution` export); bounds and `bs_window` beyond the simple passthrough; an exhibit meta language, to be extracted post 1.0 only if a declarative pattern emerges with the hand written exhibits as its test cases; narrative describe and explain chips; Sphinx pages beyond docstrings.

## Decisions taken, second round (author, 2026-08-05)

11. **The waterfall gate is CLOSED.** Shipped shape stands: cession rows keep the risk metrics, and no amounts-ceded column is added. Good enough; iterate later if a real book says otherwise.
12. **`tail` keeps its insurer treatment.** The 1-in-200 and 1-in-250 anchors stay migrated into the library.
13. **CV and SD are BOTH kept; CV is blanked where it is meaningless.** The author reads risk in loss CV, which is why the column earns its place, and CV is useless on a margin, which is why `-288.1%` was wrong to print. So neither dropping the column nor reformatting it is right: the fix is per cell. On the ledger's INSURER view the `CV` cell blanks on `Margin` rows and is served everywhere else, and `SD` is served everywhere. This generalizes the existing `_cv_or_nan` rule (which blanks CV only when the mean is near zero relative to SD) to the case that rule misses, a mean that is merely *signed*.
14. **The `stats_df` meta block gets split out under INSURER.** The frame stacks eight `meta` parameter rows (limit, attachment, el, prem, lr, sevcv_param, mix_cv, wt) on top of the moment rows, in one column space, mixing currency, ratios and weights down a single column. That is what the perspective mechanism is for. The INSURER stats view serves **two blocks**, exposure parameters and moments, instead of one 26 row stack. NOTE: this improves the reading but does **not** solve the per measure format problem, which needs measures to become columns; see the punch list.
15. **`aggregate.exhibits` gets Sphinx autodoc pages.** It is a public package and `docs/3_reference` autodocs `aggregate.plots` already; exhibits should not be the hole.
16. **No `EX.Simple` in `decl-testers.agg`.** The monograph example stands on its own; the corpus does not need it.

## Punch list (library, all small)

1. **CV blanking on ledger margin rows** (decision 13).
2. **The `stats_df` meta / moments block split** (decision 14).
3. **Sphinx autodoc pages for `aggregate.exhibits`** (decision 15).
4. **Uncomment the `exhibits` extra** the day greater_tables 6 publishes to PyPI. One line, already written.
5. **Switch `Skew` from `.3f` to `.3g`** once greater_tables ships a `g` format kind. The author has asked for it upstream; `MEASURE_FORMATS` is the single place to change.
6. **Per measure formats on the moment store**, still genuinely open. Formats are per column in greater_tables and the store runs measures down a column, so this needs either a pivot to `(component, view) x measure` inside the INSURER override, or a row format concept upstream, or a decision to leave the store unformatted. Decision 14's block split does not address it.

