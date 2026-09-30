# plan-a376: the plugin loader and the extension surface

**Status:** ruled 2026-09-30 (section 10). Stage 1 [Plugin-Loader] landed at `1.0.0a376`; stages 2 to 4 not started. Execution log in section 11.
**Repos touched:** `aggregate` (this one), `aggregate-api`, and a new `aggregate-relativity`.
**Version:** `1.0.0a376` for the library step. The api repo bumps on its own line when its section lands.

---

## 1. Goal

Let third-party packages contribute **charts and exhibits** to `aggregate` and have them appear in the `aggregate-api` SPA, without widening the 1.0 API contract and without any experimental code living inside the library.

The shape: a package named `aggregate-<something>` declares an entry point, registers into the chart and exhibit registries that already exist, and its output lands under one new **Lab** tab in the app. Several such packages may be installed at once. Installing one adds its leaves; uninstalling removes them; no central list is maintained anywhere.

This is the answer to a tension that looks like a contradiction and is not. "Lock the API down" and "allow extras" are the same design act: you freeze the **consumption** surface that users code against, and you publish a separate, explicitly unstable **extension** surface that plugin authors code against. Naming the second one is most of the work.

---

## 2. Current behavior

### 2.1 What already exists, and it is most of this

The registries are already open, by explicit prior design.

* `aggregate/charts/__init__.py` holds `CHARTS`, a dict of `name -> ChartEntry(emitter, predicate, primary)`, populated through `register_chart(name, emitter, predicate=None, primary=None)`. Its docstring says "Registration is open: emitter modules in this package populate it at import, and **app or user code may add entries**." `register_chart` raises `ValueError` on a duplicate name.
* `aggregate/exhibits/_core.py` holds `EXHIBITS`, a dict of `name -> (fn, perspectives_fn)` where `fn` is a `functools.singledispatch` generic. Its module docstring says "Registration is open: app or user code may register new types with `summary.register(MyType)`." `register_simple_exhibit(name, title, frame_attr, classes, ...)` declares a passthrough exhibit over one named frame in a single call.
* `available_charts(obj)` and `available_exhibits(obj)` derive capability from those registries plus each entry's predicate, so capability "cannot go stale" as registrations are added.

The api transport is already generic.

* `GET /v1/objects/{oid}/chart/{name}` resolves names through `available_charts`. Its docstring: "a new library emitter appears here with zero endpoint changes; an unknown or unavailable name is a 404 carrying the capability set."
* `GET /v1/objects/{oid}/exhibit/{name}` does the same through `available_exhibits`.
* `aggregate_api/capability.py` is a pure passthrough of both, and its module docstring states the governing invariant: "capability is derived, never declared twice."

### 2.2 What is missing

Exactly three things.

1. **Discovery.** Nothing gets a third party's `register_chart` call to run inside the api server process. There is no loader and no entry point group. `aggregate`'s `pyproject.toml` declares one entry point today, a Pygments lexer, and nothing consumes entry points.
2. **Provenance.** `capability.py` reports a flat list of names. Nothing says which came from the library and which from a plugin, so the app cannot badge them or separate them.
3. **A place in the UI.** `web/src/nav.js` holds `NAV_GROUPS`, six groups with authored leaves. Its docstring is explicit that this is deliberate: "The skeleton is editorial and the leaves are derived. Which groups exist, what they are called and what order they sit in is a judgment about how insurance work proceeds, so the app authors it." A leaf gated `exhibit: 'foo'` lights from `available_exhibits`, but **the leaf itself must be authored**, so a plugin's registration reaches the api and stops there.

### 2.3 The prior art this must not repeat

`extensions/` was a subpackage inside the library holding exactly this kind of material. It was deleted at 1.0.0a12 and its contents either promoted (`pentagon`, `ft`, `tweedie`), absorbed into `pedagogy.py`, or moved to the separate PMIR package. The lesson is recorded in `CLAUDE.md`. An `aggregate.contrib` bag would rebuild it under a new name: experimental code inside the library becomes the library's dependency surface, its test burden and its stability question, and a locked API shipped beside a `contrib` bag is theater. **Plugins live in their own distributions. This is not negotiable and the plan should be rejected if it drifts back.**

### 2.4 The motivating case, and the template

> **Provenance note.** This material was authored on `X:`, which is wiped nightly. It was copied to **`V:\dev\aggregate-relativity\reference\`** on 2026-09-30 and **that copy is the only one**. Read it there. Every path below names the durable copy; the `X:` original is gone.

`reference/gini_decl.py` (604 lines, plus `gini_views.py`) takes an extended P&L DecL program and produces a one-page PNG: a relativity matrix of `gini_p` across peel steps and distortion families, beside a distortion spectrum. It runs **out of process**, POSTing DecL to `http://127.0.0.1:19456/v1`, then parsing the returned `greater_tables` IR documents back into DataFrames with a local `ir_to_df` helper that has to undo rowspan sparsification.

**The author considers this file complete and is happy with its output, so it is the template, not a sketch.** The port is a re-housing of working code whose answers are already accepted, which is what makes the golden-output check in section 9 meaningful: any number that moves is a porting bug, not a design choice. Read the file before writing any of `aggregate-relativity`.

One porting detail to settle on the way in. The relativity arithmetic exists **twice** in that file: as the `rel_row`, `act_row` and `msd_pos` closures inside `render_view45`, and again at module level inside `actuals_frame`. That is harmless in a script that runs both in one pass and unacceptable in a plugin, where the exhibit and the chart are served by separate requests and must not be able to disagree. Lift it to one set of module-level functions in `derive.py` and have both callers use it.

That `ir_to_df` function is the tell. It rebuilds numbers the object already holds by parsing a presentation document. It is a workaround for a process boundary, not a design, and it is the first thing this plan deletes. In process, its two HTTP calls are two attributes:

| `reference/gini_decl.py` | In process |
|---|---|
| `POST /objects/{oid}/pricing/evaluate`, take the `gini_p` column | `pnl.evaluate().evaluation_df['gini_p']`, a `MultiIndex` on `(Step, distortion)` |
| `GET /objects/{oid}/exhibit/economic_waterfall`, take the block carrying `MSD` | `pnl.evaluation_df`, indexed by `Step`, columns `Premium spent`, `Margin spent`, `CR`, `MSD`, `SA CoC`, `Div CoC` |

**Read that table twice before implementing.** There are two different frames called `evaluation_df` in the P&L surface and this plan needs both: `PnL.evaluation_df` is a **property** (`_pnl.py` line 2439) holding the ratio walk, and `PnL.evaluate()` is a **method** (`_pnl.py` line 2660) returning an `EvaluationResult` whose own `.evaluation_df` is the gini panel. In it the first is the variable named `walk` and the second is the variable named `gini`.

---

## 3. Non-goals

State these plainly so they are not discovered halfway through.

* **[Documents-Not-Interactions]** A plugin contributes charts and exhibits. It contributes **no app behavior**: no forms, no input controls, nothing resembling the Quick QS button or the pricing form. Those are the `flag` leaf kind in `capability.py`, they require code in the SPA, and they stay the app's. This boundary is the entire reason the plan is cheap; widening it turns a week into a quarter.
* **No plugin hot reload.** Entry points are read once at process start. A new plugin needs a server restart. `uvicorn --reload` covers the experiment loop.
* **No auto-load on `import aggregate`.** See decision 4 below.
* **No chart meta-language.** `aggregate/charts/__init__.py` explicitly parks that post-1.0 and this plan does not unpark it.
* **No sandboxing.** A plugin is Python running in the server process with full privileges. That is acceptable because the api binds `127.0.0.1` and the plugins are the author's own. It is written down here so that a future hosted deployment knows it has a decision to make.

---

## 4. Design decisions

Settled with the author before drafting. Each is recorded with its reason because the reasons are what a reviewer needs to push back on.

1. **Entry points, plus an environment variable on-ramp.** The packaged route (`[project.entry-points."aggregate.plugins"]`) is the destination; an `AGGREGATE_PLUGINS` variable naming importable modules is the quick-hit experiment route. Both are the same loader and cost the same lines. The packaged route is the standard mechanism (pytest, Sphinx, Pygments, Datasette all use it) and brings dependency management and versioning; the variable brings zero ceremony, which is what an experiment actually wants at 11pm.
2. **Provenance is captured at load, never parsed from names.** The loader snapshots the `CHARTS` and `EXHIBITS` key sets before and after each plugin's `register()` and records the difference. No naming convention is load bearing, so a plugin may name its chart `relativity` rather than `gini.relativity`, and the app's badge reads a recorded fact.
3. **Cross-plugin name reuse is refused, not merged.** `register_chart` already raises on a duplicate name. `register_simple_exhibit` deliberately does the opposite: its docstring says calling it twice for one name "extends the existing exhibit to more classes rather than replacing it," which is correct inside the library's own manifest (that is how one exhibit carries a different caption per class) and **wrong across a trust boundary**, where it would silently merge two unrelated plugins into one exhibit. The loader therefore polices this itself: it refuses a plugin that reuses an exhibit name already owned by the library or by another plugin. `register_simple_exhibit`'s own behavior is untouched.
4. **The library does not auto-load plugins on import.** `import aggregate` stays deterministic; the **host** calls `aggregate.plugins.load()`. Reasons: `build()` must be reproducible, so a notebook's results should be a function of the notebook's own text and not of what happens to be installed; and the pytest suite must not change behavior because a sibling package was synced. The api server calls `load()` in `create_app()` behind a settings flag, which is the right place for the decision because a server is a deployment and a deployment may declare what it trusts.
5. **A failing plugin is recorded, not fatal.** A plugin whose import or `register()` raises is caught, its traceback stored, and loading continues. One broken experiment must never take down `build()` or the server.
6. **Lab membership is fixed at load; lit-ness varies by object.** The set of leaves under Lab comes from `loaded_plugins()`, which is process-wide and stable, so the tab does not reshuffle as you build different objects. Whether each leaf is live comes from the per-object capability payload, exactly like every other tab. A leaf that cannot serve the current object is drawn dark with a `why`, matching the `nav.js` house rule that nothing is hidden.
7. **No plugins loaded means no Lab tab at all.** A permanently empty tab on a stock install is noise, and the absent thing here is an uninstalled package rather than a library capability, so the no-hiding rule does not apply.
8. **Leaf order is plugin name alphabetically, then registration order within a plugin.** Stable across installs, and it does not depend on the order `importlib.metadata` happens to return distributions in.
9. **The plugin owns its leaf label, hint and `why`, explicitly, through a presentation manifest.** Not through new fields on `ChartEntry` or `Exhibit`: those are IR concerns, a nav `hint` is an app concern, and mixing them puts UI strings in the library's chart registry forever. The manifest is carried by `aggregate.plugins` and surfaced through `loaded_plugins()`, so the plugin still authors every string while the registries stay clean.
10. **`aggregate-<thing>` for the distribution, `aggregate_<thing>` for the import package**, matching `aggregate-api` / `aggregate_api`. Not an `agg-` prefix: `agg` is a DecL keyword and `aggregate/agg/` is the DecL file directory.
11. **The tab is `Lab`, and its key is `lab`.** Label and key agree because the group is new. The `reinsurance` key under the `Re` label is a divergence that exists only because the key predated the rename, and there is no reason to start a second one. The key lands in `data-tab`, the pane id, the stored view state and any shared link, so it is fixed from the first commit. `Lab` over `Extras` because it says experimental rather than miscellaneous, which is the honest description of what is under it.
12. **The plugin renders nothing itself. The PNG deliverable goes away.** A plugin emits a `ChartDoc` and the existing renderers draw it: the SPA's ECharts adapter in the browser, `aggregate.plots.plot_chartdoc` for a figure on disk. No plugin-local matplotlib code, so matplotlib is not a dependency of `aggregate-relativity` at all. Where the generic renderer cannot yet draw what the chart means, the fix is to the renderer or to the IR, not a private escape hatch: a plugin that draws its own figures is the second rendering pipeline this architecture exists to avoid, and the two would diverge within a month. The consequence for `reference/gini_decl.py` is that `render_view45`, the palette block, the `mpl.rcParams` block, `_tidy` and `_save` are all **deleted** rather than ported; what survives from them is the semantic content (which series, which scale, which reference line, which color role) expressed as `ChartDoc` fields.
13. **The plugin manifest rides on `GET /v1/meta`, not a `/v1/plugins` route.** `api.meta()` is already an unconditional boot fetch (`web/src/main.js` line 4024) on the very path that must resolve before the tab strip can know whether `Lab` exists, and the route's own docstring already describes its job as serving "runtime config ... so the SPA can configure its form widgets". A dedicated route would add a second boot round trip, or a `Promise.all` refactor of a two-line boot path, to carry a handful of strings. The real cost of this choice, stated so it can be watched: meta accretes, and it already holds `log2_cap`, `build_timeout` and three versions. So the manifest on meta carries name, version, leaves and a **one-line** error string only; a failed plugin's traceback goes to the server log. Split to `/v1/plugins` if and when the payload outgrows that, not before.

---

## 5. The extension surface

Add a third section to `docs/3_reference/3_x_API_Stability.rst`, after "Provisional modules". It names, in one table, what a plugin author codes against:

* `aggregate.plugins.load`, `loaded_plugins`, and the manifest dataclass
* `aggregate.charts.register_chart` and the `ChartDoc` dataclass family
* `aggregate.exhibits.register_simple_exhibit` and the per-exhibit `.register(Cls)` / `.insurer.register(Cls)` hooks
* the public frames of the first-class classes, which are already stable

And what it deliberately does not cover: anything underscore-prefixed. A plugin that reaches into `_pnl` internals is an `aggregate.contrib` bag with extra steps, and the lock is theater again. If a plugin needs something the public frames do not expose, that is an upstream request against `aggregate`, which is the same rule `aggregate-api`'s `CLAUDE.md` already sets for itself.

The honest stability statement: the extension surface sits **inside** the provisional tier. `charts` and `exhibits` are PEP 411 provisional and may break in a minor release with no deprecation period, so anything registering into them inherits that. The point of naming the surface is not to promise stability. It is to tell a plugin author exactly what their blast radius is, and to make the feedback loop explicit: the stability doc already says "Use them, and report what does not fit," and a real out-of-tree plugin is the best evidence available about whether the IR carries the right knowledge.

---

## 6. Implementation

### [Plugin-Loader] `aggregate`, new `src/aggregate/plugins.py`

One new module, roughly 150 lines with docstrings. No existing library module changes except `__init__.py` (to expose the submodule, not to star-export) and the stability doc.

Public surface:

* `PluginLeaf(name, kind, label, hint, why)` frozen dataclass. `kind` is `'chart'` or `'exhibit'`; `name` is the registry key; the other three are the app-facing strings the plugin authors. Decision 9.
* `LoadedPlugin(name, version, source, leaves, charts, exhibits, error)`. `source` is `'entry_point'` or `'env'`. `charts` and `exhibits` are the registry keys this plugin actually added, captured by the before-and-after snapshot of decision 2. `error` is `None` or the recorded traceback of decision 5.
* `load(*, allow=None)` runs discovery once and is idempotent. It walks `importlib.metadata.entry_points(group='aggregate.plugins')`, then the comma-separated module names in `AGGREGATE_PLUGINS`. `AGGREGATE_NO_PLUGINS=1` short-circuits it entirely. `allow` is an optional allowlist of plugin names, threaded through from the api's settings, for the hosted case non-goal 5 anticipates.
* `loaded_plugins()` returns the `LoadedPlugin` list, including failures.

Each entry point resolves to a zero-argument `register()` callable. It performs its `register_chart` and `register_simple_exhibit` calls and returns its list of `PluginLeaf`. The loader validates that every returned leaf names a registry key the snapshot says this plugin actually added, which catches a plugin that describes a leaf it forgot to register (or, worse, describes someone else's).

Name policing, decision 3: before running a plugin, snapshot the key sets; after, diff them. A chart collision has already raised inside `register_chart`, so it arrives as a recorded failure. An exhibit collision does not raise, so the loader detects it by checking the plugin's declared exhibit names against the pre-snapshot and rejects the plugin with a clear message. Rejection means the plugin's `LoadedPlugin` carries an `error` and is reported; **it does not attempt to unwind partial registrations**, since the registries have no removal API and adding one for this is not worth it. Say so in the docstring: a colliding plugin leaves the process in a state that warrants a restart once the collision is fixed.

Tests, `tests/test_plugins.py`: a fixture plugin module registering one chart and one exhibit against a toy class; load via `AGGREGATE_PLUGINS`; assert the registry gained exactly those keys, that `loaded_plugins()` attributes them correctly, that a raising plugin is recorded and does not propagate, that an exhibit name collision is refused, that a leaf naming an unregistered key is refused, and that `AGGREGATE_NO_PLUGINS` wins. The suite itself must never auto-load, which decision 4 gives for free; add an assertion that a bare `import aggregate` leaves `CHARTS` at its library-only size.

### [Lab-Tab] `aggregate-api`

**Transcribe this section into `dev/plan-NNNN-extras-tab.md` in that repo at implementation time**, per its own one-plan-per-step rule. Its version bumps on its own line.

Backend:

* `app.py`: `create_app()` calls `aggregate.plugins.load(allow=...)` behind a new `AGGAPI_PLUGINS_ENABLED` setting (default on locally) and a `AGGAPI_PLUGINS_ALLOW` allowlist, both in `config.py`.
* `capability.py`: `exhibits_for` and `charts_for` each gain a `provenance` field per entry, `'core'` or the plugin name, read from `loaded_plugins()`. This stays a derived passthrough and declares nothing twice, which is the module's own stated invariant.
* `routes/meta.py`: `GET /v1/meta` reports the loaded plugins, their versions, their leaves and any load failures. The About panel shows them. A plugin that failed to load must be visible somewhere a human looks, or a silent empty tab becomes a debugging session.

Frontend:

* `web/src/nav.js`: one new group, `lab`, label **Lab**, drawn last. Its `leaves` are not authored. `capsFromResponse` gains the provenance split, and a new pure function builds the Lab leaf list from the meta payload's plugin manifest: every `PluginLeaf`, ordered by decision 8, minus any name an authored leaf in the other six groups already claims. The group is omitted entirely when the manifest is empty (decision 7).
* `web/src/main.js`: the loader registry is keyed `group:leaf`, which a dynamic group cannot pre-populate. Add **one** generic loader for the `lab:*` key space that dispatches on the leaf's `kind` and reuses the existing exhibit-envelope pane and the existing `charts/mount.js` ChartDoc path. No new rendering code: a plugin's chart is a `ChartDoc` and the ECharts adapter already draws those.
* `index.html`: the six tabs are **static markup**, so a seventh appearing from JS after the meta fetch resolves is a layout shift on every page load. Carry a **hidden `lab` slot in the markup** that JS reveals when the manifest is non-empty, rather than appending a tab. This also keeps `check-nav.mjs`'s premise intact: the strip still lists every group the app knows about, and the manifest decides only visibility.
* `dev/scripts/check-nav.mjs`: currently asserts `NAV_GROUPS` key order matches the static `<ul class="out-tabs">` in `index.html`. Teach it that `lab` is the one group whose leaves are dynamic: assert the strip carries the hidden slot in the right position, and add the new invariant that no Lab leaf name duplicates an authored leaf.

### [Reference-Plugin] new repo `V:\dev\aggregate-relativity`

Not `X:`, which is wiped nightly and is outside the four sanctioned drives in any case. Its overwrite quirk, which `reference/gini_decl.py`'s own `_save` comment documents, is one more reason the code never belonged there.

Layout mirrors `aggregate-api`: `src/aggregate_relativity/`, `tests/`, `dev/`, `CHANGELOG.md`, `pyproject.toml` with `aggregate` as a path source for co-development.

* `derive.py`: `classify_positions`, `actuals_frame` and the relativity arithmetic, lifted from `reference/gini_decl.py` and re-pointed at `pnl.evaluate().evaluation_df` and `pnl.evaluation_df` per the table in section 2.4. Pure pandas over public frames, no HTTP, no matplotlib, independently testable. `ir_to_df`, `_block_with`, `fetch_frames` and `_call` are all deleted.
* `register.py`: the `register()` entry point. Registers the exhibit and (in step two, see sequencing) the chart, and returns its `PluginLeaf` list.
* `pyproject.toml`: `[project.entry-points."aggregate.plugins"] relativity = "aggregate_relativity.register:register"`.
* `reference/`: the approved original and its output, carried into the repo as the porting spec and the golden data. Not packaged, not imported, and its `data/*.csv` must be **exempted from the usual csv gitignore** so the golden files are actually tracked. They are outputs of the two synthetic DecL programs in the script (`CAPSTONE`, `PNL_LOSS_RATIO`), so they are fixtures and not real data, and the data-firewall rule is satisfied.

**Registry keys, vetted per the `CLAUDE.md` naming-vet rule.** `rg -i relativity src/aggregate` returns nothing, so the name is free across the library: no method, property, exhibit key, chart key, DecL keyword or spec key collides. Package, entry point and document therefore all read `relativity` and agree with each other, which is the happy case and worth keeping. Use `relativity` for the exhibit and the chart while they are one page each; if the chart later splits, `relativity_matrix` and `relativity_spectrum`.

The name this plugin must **not** take is `evaluate`, which was the working title through an earlier draft. The library already has a `pricing_evaluate` exhibit, a `PnL.evaluate()` method and an `EvaluationResult`, and the plugin consumes all three. A registry key names the *document*, not the machinery it was computed from, and `relativity` says what the page shows.

---

## 7. Sequencing

Deliberately staged so each stage is independently useful.

1. **[Plugin-Loader]** alone, with the fixture plugin and its tests. Library bumps to `1.0.0a376`. Nothing user-visible changes.
2. **[Reference-Plugin] exhibit only.** `actuals_frame` behind `register_simple_exhibit`. Pure tables, needs **zero** chart IR work, and proves the whole discovery and provenance path end to end.
3. **[Lab-Tab]** in the api, against that one real exhibit. Now the gini numbers are in the SPA.
4. **[Reference-Plugin] chart**, which forces the IR question in section 8 with a working consumer in hand.

Stages 1 to 3 deliver the gini numbers in the app before the IR question is settled at all, which is the 90%-the-easy-way reading of this.

---

## 8. The known gap: a categorical matrix panel

`reference/gini_decl.py`'s right-hand panel, the distortion spectrum, is a clean `xy` panel and maps onto the current IR today. The left-hand relativity matrix does **not**, and this is worth stating precisely because it is the one piece of real library work the plan uncovers.

`PANEL_KINDS` in `charts/ir.py` includes `'heatmap'`, but the data structure behind it, `SurfaceData`, is built for **numeric lattices**: `x0/dx/nx`, `y0/dy/ny`, `bs`, `k`, mass-preserving block reduction, exact marginals, `window`. It is the bivariate joint surface's carrier. The gini matrix is a different animal: named categorical rows (gross book, each layer, each tier package, final net) against named categorical columns (four distortion families plus three point estimates), **two texts per cell** (the multiple, and the actual in parentheses beneath), and a diverging color scale with a hard neutral band at plus or minus five percent.

So stage 4 needs either a new `'matrix'` panel kind or a categorical mode on `'heatmap'`, carrying string tick labels and an optional per-cell annotation layer. **That is a library change to a provisional module, driven by an out-of-tree consumer, which is exactly the feedback loop `3_x_API_Stability.rst` declares the provisional tier to exist for.** It is evidence for this plan rather than against it, but it is real work and it is why the chart is stage 4 and not stage 1.

Do not attempt it speculatively. Build stages 1 to 3, look at what the evaluate plugin actually needs, and design the panel kind against one working case.

---

## 9. Acceptance checks

* `uv run pytest` green, including a new `tests/test_plugins.py`.
* `python -c "import aggregate; print(len(aggregate.charts.CHARTS))"` returns the library-only count with a plugin installed, proving decision 4.
* `AGGREGATE_PLUGINS=aggregate_relativity.register` plus `aggregate.plugins.load()` makes `available_exhibits(pnl)` include the plugin's exhibit and leaves every other object's capability unchanged.
* `AGGREGATE_NO_PLUGINS=1` suppresses it.
* A plugin raising inside `register()` leaves `build()` working and appears in `loaded_plugins()` with a traceback.
* With the api running and the plugin installed: a Lab tab appears, its leaf is live on a built extended P&L, dark with the plugin's own `why` on an `Aggregate`, and `GET /v1/meta` names the plugin and its version.
* With the plugin uninstalled: no Lab tab, and every existing `check-nav.mjs` assertion still passes.
* The relativity numbers computed in process match the approved `reference/data/gini_view45_actuals*.csv` files to within floating-point noise. This is the regression that says the rewrite off `ir_to_df` did not change any answer.

---

## 10. Questions settled, and what is left

The three questions this plan opened in draft were answered by the author on 2026-09-30 and are now decisions 11, 12 and 13 in section 4: the tab is **Lab** with key `lab`; the plugin **renders nothing itself** and the PNG deliverable goes away; the manifest rides on **`GET /v1/meta`** with one-line errors only.

Nothing is blocking. Two things a reviewer should push on if they disagree, both cheap to change before stage 1 and expensive after:

1. **Decision 4, no auto-load on import.** It makes a notebook user call `aggregate.plugins.load()` explicitly, which is friction, and pytest's precedent runs the other way (it auto-loads and offers `-p no:`). The plan takes reproducibility over convenience because `build()` is a numerical entry point and its answers should not depend on what is installed. If that reads as too strict, the alternative is auto-load plus a prominent kill switch, and the loader is the same either way.
2. **Decision 13, the manifest on meta.** Watch the payload. The moment a plugin's manifest wants more than name, version, leaves and a one-line error, split it to `/v1/plugins` rather than growing meta.

---

## 11. Execution log

### Stage 1 [Plugin-Loader], landed `1.0.0a376` on 2026-09-30

Stage 1 only. Stages 2 to 4 are unstarted: [Reference-Plugin] wants a repo that
does not exist yet at `V:\dev\aggregate-relativity` (only its `reference/` copy
is there), and [Lab-Tab] is `aggregate-api`'s, to be transcribed into that
repo's own plan per section 6. **The plan therefore stays in `dev/`.**

Files: new `src/aggregate/plugins.py`, `tests/test_plugins.py`,
`tests/plugin_fixtures/` (six modules), `docs/3_reference/3_x_Plugins.rst`;
edited `src/aggregate/__init__.py`, `src/aggregate/config.py`,
`docs/3_reference/3_x_API_Stability.rst`, `docs/3_Reference.rst`.

### Facts checked against the code before starting

Every claim in sections 2.1 and 2.2 holds. `CHARTS` is 13 entries and `EXHIBITS`
18; `register_chart` raises on a duplicate (`charts/__init__.py:110`);
`register_simple_exhibit` extends rather than replaces (`exhibits/_core.py:1001`,
docstring confirms); `pyproject.toml` declares only the Pygments lexer entry
point and consumes none. `rg` for `plugin` across `src/aggregate` returned
nothing, so `plugins`, `PluginLeaf`, `LoadedPlugin`, `load`, `loaded_plugins`
and `reset_plugins` are all free. `import aggregate` pulls in neither `charts`
nor `exhibits`, which is the property that lets `__init__` bind the submodule.

One drift, stage 4 material: section 2.4 puts `PnL.evaluate` at `_pnl.py` line
2660; it is at 2636. `PnL.evaluation_df` at 2439 is exact, and the distinction
between the two frames the table warns about is real.

### Divergences

1. **[Env-Allow-List-Exemption], work the plan does not cover.**
   `aggregate.config` validates every `AGGREGATE_*` variable against `_ENV_MAP`
   and warns `Unknown environment variable ...; ignored.` on anything else
   (`config.py:519` before the edit). `AGGREGATE_PLUGINS` and
   `AGGREGATE_NO_PLUGINS` both tripped it, so every settings load with a plugin
   configured would have emitted spurious noise on the one path a host is
   guaranteed to run. Fixed by a new `config._ENV_IGNORED` frozenset holding
   those two beside the existing `AGGREGATE_CONFIG` special case, which is the
   same idea already there for the same reason: a variable this module does not
   own. Covered by
   `test_plugins.py::test_config_does_not_warn_about_the_plugin_variables`.
   Reading the two variables through `config` instead was considered and
   rejected: `load()` must work before a settings object exists, and neither
   value changes a numerical answer.

2. **Acceptance check 2 is tested as an invariant, not as a count.** Section 9
   asks that `len(aggregate.charts.CHARTS)` return the library-only count with a
   plugin installed. Asserting the literal 13 would fail on the next chart
   anyone adds, for no gain, so the test asserts the property instead: in a
   subprocess with `AGGREGATE_PLUGINS` set, `import aggregate` leaves
   `loaded_plugins()` empty and neither fixture key in either registry. The
   count itself was checked by hand and is 13 with the plugin configured, the
   same as without.

3. **`load` takes `env` and `force` beyond the plan's `load(*, allow=None)`.**
   `env=` mirrors `config.load_settings(env=...)`, the house idiom, and keeps
   the tests from mutating `os.environ`. `force=` is what makes an idempotent
   loader testable at all. Both keyword-only.

4. **`reset_plugins()` added to the public surface.** The tests need to re-run
   discovery, and a host that reconfigures does too. Its docstring is explicit
   that it clears this module's record only: neither registry has a removal API,
   so a second `load()` of the same plugin records a duplicate-name failure.

5. **Three small additions to the dataclasses.** `LoadedPlugin.ok`, so callers
   filter without comparing to `None`. `PluginLeaf.__post_init__` validating
   `kind` against `LEAF_KINDS`, which is what keeps non-goal
   [Documents-Not-Interactions] from being lost to a typo. And the module
   constants `ENTRY_POINT_GROUP`, `ENV_PLUGINS`, `ENV_NO_PLUGINS`, `LEAF_KINDS`
   are public, so the api repo names them rather than re-spelling the strings.

6. **Two failure modes beyond section 6's list**, both recorded rather than
   raised, by decision 5's rule: an environment module with no callable
   `register`, and a `register()` returning something that is not a
   `PluginLeaf`.

7. **`docs/3_reference/3_x_Plugins.rst` added**, and entered in the
   `docs/3_Reference.rst` toctree. Section 5 asked only for a section in
   `3_x_API_Stability.rst`, which is there as `.. _extension-surface:`, but its
   `:func:`aggregate.plugins.load`` cross-references do not resolve without an
   `automodule` somewhere. Docs are not rebuilt in the loop, per `CLAUDE.md`;
   the build is pending.

### One finding for stage 4

A plugin registering a **chart** needs a `singledispatch` generic whose base
raises `NotImplementedError`, because that is what `available_charts` reads. The
library has a factory for exactly this, `charts._emitter_base`, and it is
private, while section 5 says in as many words that the extension surface covers
no underscore-prefixed name. So every chart-contributing plugin either hand-rolls
five lines (what `tests/plugin_fixtures/_toy.py` demonstrates, deliberately) or
reaches for a private name the doc just forbade. Promoting a public
`charts.chart_emitter(name)` was **not** done here: it widens the public surface,
the plan did not ask for it, and stage 4 is when a real chart-contributing plugin
exists to judge it against. Decide it there.

### Verification

* `.venv/Scripts/python.exe -m pytest tests/test_plugins.py -n0 --dist no` —
  25 passed.
* `.venv/Scripts/python.exe -m pytest -m 'slow or not slow'` — **5478 passed**,
  651 warnings, 130 s. The gate.
* `ruff check` clean on every file touched. The 49 findings `ruff check src`
  reports are pre-existing and in other modules.
* Section 9's five library-side checks all run by hand and pass: the
  library-only registry with a plugin configured, `load()` lighting the
  plugin's exhibit for its own type while leaving a built `Aggregate`'s
  capability unchanged, `AGGREGATE_NO_PLUGINS` suppressing, and a raising
  plugin leaving `build()` working with its traceback in `loaded_plugins()`.
  The remaining three checks are the api and plugin repos' and wait on stages
  2 to 4.
