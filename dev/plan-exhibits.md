# Plan [Exhibits-Module]: business exhibits, stats frames to greater_tables IR

> **Status: DRAFT, approved 2026-08-04 for phased execution.** New `aggregate/exhibits.py`, purely additive (imports from core, core never imports it), provisional at 1.0. Companion plan: `dev/plan-chart-ir.md`. Both derive from the author's design notes on the aggregate to aLL (aggregate_api) interface.

## Principle and placement test

aggregate owns meaning, the app owns arrangement. Test for placement: if deleting the web app would destroy knowledge an actuary would want in a notebook, that knowledge is in the wrong place. Today it leaks: aggregate_api carries ROW_FLAGS and FORMATS (`tables.py:113`, `:132`), `_drop_raw_moments` (`objects.py:1407`), exhibit titles and captions (`main.js:646-655`), and the reinsSeries survival accumulation. The FCC stats frames are raw materials; the business translation that makes them resonate (especially `PnL.stats_df` and parts of `reins_stats_df`) belongs in the library. Exhibits are allowed to be domain specific; core aggregate stays domain agnostic.

## Decisions taken (author, 2026-08-04)

1. GT dependency: new optional extra `exhibits = ["greater_tables>=6.0.0a8"]` with lazy import inside the IR conversion step. `import aggregate` and `import aggregate.exhibits` never touch greater_tables; a clear ImportError names the extra when missing. GT must publish to PyPI before aggregate ships the extra (it is path installed alpha today); until then the extra is documented as requiring the sibling checkout, as aggregate_api already does.
2. The perspective enum is named `Perspective`, kwarg `perspective=`, URL `?perspective=`. This avoids the three existing senses of "view" (the reins frame column levels, the app's REINS_VIEWS, the app's table-view toggle). Values are RAW, INSURED, INSURER, REINSURER (author, 2026-08-05, replacing the earlier RAW/BUYER/SELLER: buyer and seller are relative and confusing, since the insured buys insurance, the insurer sells insurance while buying reinsurance, and the reinsurer sells reinsurance while buying retro).
3. RAW serializes as a TableDoc like every other perspective: one client path, uniform ETag, MultiIndex structure survives.
4. Signature default is `Perspective.RAW`: passthrough always works, translation is opt in.
5. v1.0 implements RAW and INSURER only (author, 2026-08-05). INSURER defaults to RAW unless an override is registered for the (exhibit, type) pair. The override is custom per exhibit and unbounded in scope: some are identity (no override at all), some thin (renaming a column or two), and some extensively different from the raw frame, the xpnl tower ledger above all. Which treatment each exhibit gets is decided case by case with the author once the infrastructure is in place. INSURED and REINSURER stay in the enum as stable vocabulary with no 1.0 registrations.

## Background facts

greater_tables 6.0.0a8 IR is `TableDoc` (frozen pydantic, `ir_version` 1, JSON Schema, `canonical_dict` / `canonical_json` / `doc_hash` / `stamp`; byte deterministic). GT has no "block" concept, the unit is one whole table, so a multi table exhibit is a list of TableDocs. `build(df, TableSpec) -> TableDoc` is the conversion entry; TableSpec carries caption, notes, formatters (sugar strings such as `',.1%'`), row_flags, cell_flags, include_raw, max_rows, sparsify. The app already renders TableDocs end to end (server `tables.py: frame_document`, ETag = doc hash, client walker `renderTable` plus `irToGridInput`). The library side has the FCC contract (`constants.py:96-151`), `dev/FEATURES.csv`, `dev/reporting-guidelines.md`, LabeledMixin relabeling at the serve step, and no singledispatch precedent (the informal dispatch is `utilities.qd`'s isinstance chain).

## Library design: src/aggregate/exhibits.py (new, single module)

Not star imported in `__init__.py` (same precedent as Tweedie, Pentagon, pedagogy); users write `from aggregate import exhibits`. Names vetted: `Exhibit`, `Perspective`, `stats`, `reins`, `available_exhibits` are free; a star exported `tail` would shadow the `aggregate.tail` submodule binding, which the non star decision moots. Split into a package post 1.0 only if it grows past roughly 800 lines.

Public surface, all inside `aggregate.exhibits`:

- `Perspective(Enum)`: `RAW` (pass the underlying frame through, essentially GT(df) to IR), `INSURED` (the policyholder, buyer of insurance), `INSURER` (seller of insurance and buyer of reinsurance; the cedent, the object holder on the reins exhibits), `REINSURER` (seller of reinsurance; sign and label flips relative to the insurer; retro, where the reinsurer buys, is parked). Only RAW and INSURER are implemented at 1.0 (decision 5); INSURED and REINSURER are vocabulary only until their implementations land post 1.0.
- `Exhibit` frozen dataclass: `ir_blocks: list` (TableDocs), `name`, `title`, `perspective`, `meta: dict` (captions per block, source frame names). `to_payload() -> dict` (canonical_dict per block plus envelope fields); `hash` property (sha256 over concatenated block doc_hash values, first 12 hex) for ETags.
- Generic exhibit functions via `functools.singledispatch`, one canonical implementation per (exhibit, perspective) pair, living here and only here: `summary`, `tail`, `stats`, `validation`, `reins`, `pnl_ledger`, `pnl_ratios`, `dependency`. Signature `f(obj, perspective=Perspective.RAW) -> Exhibit`. The base raises NotImplementedError naming the type. Registration is open: app or user code may register new types. When REINSURER lands post 1.0 it shares the insurer implementation with an internal sign and label switch, never a parallel copy.
- `exhibit_frames(obj, name, perspective) -> list[tuple[str, DataFrame, dict]]`: the pure pandas frame stage returning (block name, translated frame, TableSpec kwargs). Testable and usable without GT installed.
- `build_exhibit(obj, name, perspective) -> Exhibit`: registry lookup, frame stage, then GT `build(df, TableSpec(**kw))` per block. This and `to_payload` are the only lazy import sites.
- `EXHIBITS: dict[name, (generic_fn, perspectives_fn)]` where `perspectives_fn(obj) -> list[Perspective]` is the per object predicate. At 1.0 the predicate gates exhibit availability (reins only when `obj.occ_reins or obj.agg_reins`) and the perspectives list is `[RAW, INSURER]` for every available exhibit, since INSURER always works via the default rule below; post 1.0, REINSURER appears only where a flip is defined. `available_exhibits(obj) -> list[tuple[str, list[Perspective]]]` derives from the singledispatch registries (MRO hit) plus the predicates, so it cannot go stale. Needs no GT import.

The INSURER default rule (author, 2026-08-05): **INSURER equals RAW unless an override is registered** for the (exhibit, type) pair. Mechanically, the insurer path serves the raw frames passed through a per (exhibit, type) override hook whose default is identity. The hook is a full frame translation, not a styling knob: light cases rename a column or two, heavy cases rebuild the presentation entirely (the xpnl tower ledger is the expected heavy case, per the original design notes' "esp. true of xpnl stats_df"). The inventory's "knowledge migrating in" column is the starting point for each override (the raw moment drop, the row flags, the caption text, renames), not a bound on it; the actual treatment is settled case by case at each phase, with the author, once the scaffold exists. The rule keeps the generic path total, makes every override an explicit reviewable delta on the raw frame, and means a new exhibit is useful the moment its raw registration exists.

All served frames pass through LabeledMixin `_relabel` (honoring `use_labels` and `renamer`); titles use `_title_name` ("{label} ({name})"). Plain Python plus pandas; no exhibit meta language. If a declarative pattern emerges after several exhibits are written, extract it post 1.0 with the hand written ones as test cases.

## Exhibit inventory (initial; author confirms at review)

| name | source frame(s) | perspectives | kinds | knowledge migrating in |
|---|---|---|---|---|
| summary | summary_df | raw, insurer | agg, port, pnl, distortion, bvagg (Severity is near first class and exempt; verify its surface at scaffold time) | "Summary" title; caption including Freq percentiles blank by design; `_summary_flags` (total, Agg subtotal) from app tables.py |
| tail | tail_df | raw, insurer | agg, port | capital anchor caption; `_tail_flags` (emphasis at T in (200, 250), total row) |
| stats | stats_df | raw, insurer | all five FCCs | insurer drops ex1/ex2/ex3 (raw keeps all 26 rows); measure formats |
| validation | validation_df | raw, insurer | all five FCCs | insurer adds emphasis on failing rows |
| reins | reins_stats_df + reins_summary_df (two blocks) | raw, insurer | agg, port when reinsurance present | raw moment drop; summary flags |
| pnl_ledger | PnL.stats_df | raw, insurer | pnl | the flagship translation: kappa scenario ladder captions, footing rules, Side sign presentation |
| pnl_ratios | PnL.ratio_df + legs_df | raw, insurer | pnl | arranges the "raw materials, not a card" frame (`_pnl.py:1776`) into the LR/ER/CR card |
| dependency | dependency_df + axis_support_df | raw, insurer | bvagg | none today |

The perspectives column is the 1.0 surface: raw plus insurer everywhere, per decision 5, with the insurer column of the table realized as overrides on the raw frame. INSURED and REINSURER register nothing at 1.0; they are in the vocabulary now so the enum does not churn when their implementations arrive. The reinsurer semantics questions (does the reinsurer see Ceded relabeled as its gross; does Net render at all; the cede and assume swap on pnl) travel with that deferred work as its opening review gate. A retro perspective (the reinsurer as buyer of retrocession) is parked behind even that.

Deferred from the inventory: pricing and pentagon exhibits (parameterized by distortion calibration and computed in POST routes, they do not fit the parameter free GET envelope; park until the app's upstream asks land: a `density=`/`basis=` kwarg on `calibrate_distortions` and a public `GridDistribution` export), bounds, bs_window.

## App API surface (aggregate_api)

Two routes in `routes/objects.py`, reusing `_locked_entry` and the `frame_document` ETag pattern (`objects.py:1533-1539`):

- `GET /v1/objects/{oid}/exhibits`: passthrough of `available_exhibits(entry.obj)` as `{"exhibits": [{"name", "title", "perspectives"}]}`. No per kind tables in the route.
- `GET /v1/objects/{oid}/exhibit/{name}?perspective=raw|insurer` (the enum has four values; 1.0 serves these two, and anything unserved is a 400 like any other unsupported perspective): the envelope `{name, title, perspective, meta, blocks: [TableDoc canonical dicts]}`. ETag is the Exhibit hash; 304 on If-None-Match; 404 listing the capability set on unknown name; 400 on unsupported perspective.

The client renders blocks with existing machinery (mountTable static walker plus irToGridInput grid), unchanged. New library exhibits appear with zero endpoint changes. Pages (exhibit plus plot combinations) are app content: a page composes (exhibit_name, perspective) and (plot_name, options) chips; layout, grid, tabs live app side. Narrative text derived from the object is a library chip (Exhibit.meta captions); text about the page is app content.

## Menu from capability; gray out reconciled

The app house rule (index.html:160-163) is never hide, gray out. Reconciliation: the page's chip set per kind is app content and fixed; a chip whose capability is absent renders grayed and disabled with an explanatory title (the Bounds tab pattern). `applyKindGating` and the hardcoded NA tables in main.js are rewritten to read the capability response; `has_reins` gating collapses into it. Exhibits present in capability but unknown to the page layout appear in a generated list under the More tab: chrome the app never knew was never on the menu to hide.

## Tests

Library, `tests/test_exhibits.py`: one small object per kind via `build()`; `available_exhibits` shapes and predicates (no reins perspectives without reinsurance); frame stage structure via `exhibit_frames` (indexes, dropped rows, relabeling honored), all GT free. Under `pytest.importorskip('greater_tables')`: `build_exhibit` returns TableDocs, and committed canonical_dict JSON snapshots per (exhibit, perspective, kind) guard both the translation and IR drift. `tests/test_plots_boundary.py` gains two assertions: importing aggregate does not import greater_tables; importing aggregate.exhibits imports neither matplotlib nor greater_tables. `dev/FEATURES.csv` gains the exhibits surface; `dev/TODO.md` kept current.

App: `tests/test_objects.py` capability plus envelope contract (ETag, 304, 404, 400, byte determinism); new `dev/scripts/check-exhibits.py` (sibling of check-frames.py) asserting exhibit blocks agree with frame routes where they cover the same frame; `capture_fixtures.py` captures envelope payloads.

## Phases and cadence

Each library phase bumps `1.0.0aNNN` with a CHANGELOG section and a one line commit `[Exhibits-Module] aNNN: summary`; app phases follow the app's own conventions. From phase 2 on, each exhibit's insurer treatment (identity, thin override, or full reshape) is decided with the author at its phase, not scoped in advance; the phase descriptions below name the expected starting overrides only.

1. **[Exhibits-Scaffold]** (library): the module with Perspective, Exhibit, registry, available_exhibits, exhibit_frames, build_exhibit; summary and tail for Aggregate and Portfolio; the `exhibits` extra in pyproject; boundary tests.
2. **[Exhibits-Stats-Validation]** (library): stats and validation across the five FCCs; dependency for bvagg; the raw moment drop semantics move in (the app keeps its copy until the app phase).
3. **[Exhibits-Reins-Insurer]** (library): reins, raw plus the insurer overrides (raw moment drop, summary flags, captions).
4. **[Exhibits-PnL-Translation]** (library): pnl_ledger and pnl_ratios, raw plus insurer. The tower (xpnl built) ledger is the expected heavy insurer override, extensively different from the raw frame. GATE: author reviews the PnL business framing (captions, footing rules, Side sign presentation, the tower reshape) before merge.
5. **[Exhibits-App-Endpoint]** (app): the two routes; menu from capability; delete migrated ROW_FLAGS and FORMATS entries and the main.js title and caption literals for migrated exhibits (pricing FORMATS stay); check-exhibits.py in CI.
6. **[Exhibits-App-Cleanup]** (app, after a soak): retire `_drop_raw_moments` once the stats and reins frame routes re-point at exhibit raw and insurer, or explicitly keep the frame routes as the raw CSV path forever (decide during the endpoint phase).

## Explicitly deferred

INSURED and REINSURER implementations (the reinsurer semantics review is that work's opening gate; see the inventory note); the retro perspective; pricing exhibits; exhibit meta language; narrative describe and explain chips; GT PyPI publication (external prerequisite); Sphinx pages beyond docstrings.
