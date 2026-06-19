# config — user-editable settings file for aggregate

New feature; bumps the `a*` version. Standalone (no dependency on the tail /
reins plans). Goal: lift the "secret bits" currently hard-coded in
`constants.py` and at call sites (`log2`, default database, `reins_bucket` /
`dsev_bucket`, validation tolerances, plot styling) into a single, discoverable,
hand-editable **TOML** file under `~/.aggregate/`, layered as an override on the
shipped defaults — without violating the project's no-magic principle.

**PHASE 2: SHIPPED (2026-06-19, v1.0.0a83).** Wired the surviving numerics floors
(`aliasing_ratio`, `exeqa_noise_floor`, `deficit_materiality`) plus the stranded
window/bivariate sizing knobs into config, and **dropped the dead `FT_NOISE_FLOOR`**.
Exact constant list and file edits in the Phase 2 section below. (Matplotlib
graphics into config was considered and **dropped** — see
`dev/done/plans-considered-and-rejected.md`.) Both phases complete; plan retired
to `dev/done/`.

> **STATUS: PHASE 1 SHIPPED (2026-06-05, v1.0.0a30).** `config.py` +
> annotated `data/config.default.toml` landed; `requires-python>=3.11` + stdlib
> `tomllib`; `WL` deleted (→ `logger.warning`); full `recommend_p →
> bucket_sizing_p` rename (no alias); `[build]` / `[discretization]` /
> `[validation].eps`+`.noise` / `[multivariate].window_nines` wired through
> `get_settings()`; the 10-vs-16 `log2` split fixed (bare `Underwriter()` and
> module `build` both read `[build]`); `Underwriter` gains a `config` info line,
> `show_settings()`, `write_default_config()`, `reload_settings()`; new
> `tests/test_config.py` (16 cases); full suite green (1009 passed). **One
> deliberate deviation from the slim-everything wording:** `constants.py` still
> holds the **plotting** constants (`FIG_*`, `FONT_SIZE`, …) and the
> **numerics-pending floors** (`ALIASING_RATIO`, `EXEQA_NOISE_FLOOR`,
> `FT_NOISE_FLOOR`). The plotting ones now stay there **permanently** — moving
> graphics into config was considered and rejected (they are used in
> default-argument expressions, and the surface duplicates matplotlib's own
> `mplstyle`/`rcParams`; see `dev/done/plans-considered-and-rejected.md`). The
> floors awaited the numerics review (numerics-0…4, a55–a80 — **now complete**).
> **Phase 2 (scoped 2026-06-19):** wire `aliasing_ratio`, `exeqa_noise_floor`,
> and the post-Phase-1 `deficit_materiality` into `[validation]`; **drop**
> `FT_NOISE_FLOOR` (dead — the experimental `min|ft|` switch was abandoned with
> *zero accuracy benefit*, see `dev/audit-numerics-2-findings.md`); and gather
> the stranded window-sizing trio (`window_log2_growth`, `window_slack_thick`,
> `concentration_cv` → `[discretization]`) plus `min_axis_log2` → `[bivariate]`.
> This plan stays in `dev/` until Phase 2 ships.
>
> **STATUS (original): NOT STARTED — plan for review.** Author confirmed (2026-06-04):
> TOML format; location `~/.aggregate/` via `Path.home()` (option A — **no
> `platformdirs`**); precedence defaults→file→env→kwargs; never auto-create;
> ship an all-commented template you uncomment to activate; `Underwriter.info`
> reports the active config; bundled reference databases and the default
> mplstyle stay in the package (user overrides live in `~/.aggregate`).
> **(2026-06-05) Python decision:** bump `requires-python` to **`>=3.11`** and
> use stdlib `tomllib` directly — **no `tomli` backport, no new dependency.**
> **(2026-06-05) Refinements:** all tunable defaults *move into* `config.py`
> (the magic-numbers "junk drawer" goes away); **path names** also move to
> `config.py`. `constants.py` is **slimmed to a dependency-free leaf** holding
> `Validation`, `DefectiveDistributionWarning`, and `REINS_LABEL_*` (5 strings),
> **retained for a hard architectural reason, not legacy compat:** the two
> *types* are imported by low-level modules (`utilities` imports `Validation`,
> `spectral` uses `DefectiveDistributionWarning`) that sit *below*
> `distributions`/`portfolio`, so they cannot move up without import cycles, and
> a `Flag` enum + `Warning` class do not belong in the settings module. The five
> `REINS_LABEL_*` **stay constants (not config)**: they are structural MultiIndex
> column/axis keys referenced *by literal* in `multivariate.py` (`['Ceded',
> 'Net']`), in plot labels, and asserted as literals across the reins tests, so
> making them user-settable would be a leaky abstraction that breaks
> `.loc[..., ('occ','Gross')]` lookups (they also keep the `constants.py` name
> honest). **`WL` is deleted** (legacy custom log level 25, never registered) —
> its 12 `logger.log(WL, …)` calls become `logger.warning(…)`. **No back-compat
> re-exports** — internal import sites are repointed (pre-1.0, clean break).
> Section `[bucketing]` →
> **`[discretization]`** (absorbs the renamed percentile + `window_nines`).
> `RECOMMEND_P`/`recommend_p` → **`bucket_sizing_p` everywhere, no alias**
> (constant, the `build`/`build_many`/`update` kwarg, the `hints{}` key, regexes,
> docs) — breaking, fine pre-1.0. `WINDOW_NINES` (TODO'd for exposure) joins
> config as `[discretization] window_nines`; the `multivariate._WINDOW_NINES`
> copy (same value, duplicated only to avoid a heavy import — which config, a
> leaf, removes) becomes its **own first-class** `[multivariate] window_nines`
> (independently tunable, since the 2-D per-axis window may want fewer nines for
> memory). Singleton: lazy +
> cached on first use, plus explicit `reload_settings()`. The shipped template
> annotates each key with valid options / sensible range. Note: `noise` is an
> absolute dust floor (1e-12), **not** a coverage tolerance — the 12-nines
> *coverage* is `window_nines`; the two coincidentally both involve 12.

## Context

There is **no** config or environment-variable handling in the codebase today
(verified: no `getenv` / `tomllib` / `configparser` anywhere). Every tunable is
a literal: `constants.py` holds the numbers, and the call sites bake in their
own (e.g. `Underwriter.__init__` defaults `log2=10` while the module-level
`build = Underwriter(databases='test_suite', update=True, log2=16)` at
`underwriter.py:1086` hard-codes `log2=16` — a live inconsistency this work
removes by giving both a single config-backed default).

The directory split already exists and is correct — keep it:
- `default_dir` = `Path(files('aggregate')) / 'agg'` — **shipped, read-only,
  versioned** reference databases (`test_suite.agg`, `test_suite2.agg`,
  `test_decl.agg`). Refresh on `pip install -U`.
- `user_dir` = `Path.home() / '.aggregate'` (`USER_DIR_NAME`) — **user-owned,
  persistent** content. The new config file and any user style override join the
  user's own `*.agg` here.

## Decisions locked with the author

- **Format: TOML.** Read with stdlib **`tomllib`** (requires Python ≥ 3.11 —
  see Files; no third-party reader). No TOML *writer* needed at runtime — the
  template is a static shipped file we copy, so reader-only.
- **Location: `Path.home() / '.aggregate' / 'config.toml'`.** Option A — a
  single hidden dotfolder, identical on every OS (already works on Windows).
  **No `platformdirs`** (it is a third-party dependency whose only purpose is
  OS-native paths like `~/.config/aggregate`, which contradicts the chosen
  `.aggregate` name; switching to it later is a one-line change in `config.py`).
- **Precedence (low → high):** `constants.py` defaults → `config.toml` →
  `AGGREGATE_*` environment variables → explicit `build(...)` / constructor
  kwargs. The top "caller wins" rung is the same rule `hints{}` already
  documents, so a config file is just a new layer underneath — it will feel
  native.
- **Never auto-created.** The library writes nothing to disk on import. Two
  fresh installs with no file → identical behaviour.
- **Shipped template is fully commented.** `write_default_config()` (explicit)
  copies an annotated `config.default.toml` into `~/.aggregate/`; every line is
  `# key = value`, so the file is inert until a line is uncommented — "install +
  write template" still equals pure defaults. Uncomment to activate.
- **Discoverable, not silent (the anti-magic requirement):**
  - `Underwriter.info` / `__repr__` gains a `config:` line —
    `~/.aggregate/config.toml (loaded, N overrides)` or `(none — defaults)`.
  - `Underwriter.show_settings()` prints every setting **and its source**
    (`default` / `config` / `env` / `kwarg`).
  - Escape hatches: `AGGREGATE_CONFIG=/path` relocates the file;
    `AGGREGATE_CONFIG=none` ignores it (reproducible runs).
  - Unknown keys in the file **warn loudly** (mirrors the unknown-`hints{}` key
    behaviour), they do not silently no-op.
- **Bundled databases & default style stay in the package.** The default
  `aggregate.mplstyle` remains package data loaded via `importlib.resources`
  (`style.py:31`). (A config-level `[plotting] style` override was considered and
  **rejected** — users restyle via matplotlib's native `mplstyle`/`rcParams`; see
  `dev/done/plans-considered-and-rejected.md`.) The shipped-template vs. active-file
  symmetry still holds for the **config** file and databases (default_dir vs.
  user_dir).

## Settings inventory

The file is sectioned to mirror the mental model. The default *values* now live
as the `config.py` dataclass field defaults (their single source of truth); the
"former home" column records where each literal lived before this work.

| Section | Key | Default | Former home |
|---|---|---|---|
| `[build]` | `log2` | `16` | `__init__`=10 / module `build`=16 (unify → 16) |
| `[build]` | `bs` | `0` (auto) | `build(bs=0)` |
| `[build]` | `padding` | `1` | `update` kwarg |
| `[build]` | `normalize` | `true` | `update` kwarg |
| `[build]` | `databases` | `["test_suite"]` | module `build` literal |
| `[build]` | `update` | `true` | module `build` literal |
| `[discretization]` | `reins_bucket` | `"linear"` | `REINS_BUCKET_DEFAULT` |
| `[discretization]` | `dsev_bucket` | `"linear"` | `DSEV_BUCKET_DEFAULT` |
| `[discretization]` | `bucket_sizing_p` | `0.99999` | `RECOMMEND_P` (renamed `BUCKET_SIZING_P`) |
| `[discretization]` | `window_nines` | `12` | `WINDOW_NINES` (1-D aggregate output window) |
| `[validation]` | `eps` | `1e-4` | `VALIDATION_EPS` |
| `[validation]` | `noise` | `1e-12` | `VALIDATION_NOISE` (dust floor, **not** a coverage) |
| `[validation]` | `aliasing_ratio` | `10` | `ALIASING_RATIO` |
| `[validation]` | `exeqa_noise_floor` | `1e-4` | `EXEQA_NOISE_FLOOR` |
| `[validation]` | `deficit_materiality` | `1e-4` | `DEFICIT_MATERIALITY` (added post-plan; 1e-4 value flagged for review, numerics-3) |
| `[discretization]` | `window_log2_growth` | `4` | `WINDOW_LOG2_GROWTH` (`distributions.py`; window family) |
| `[discretization]` | `window_slack_thick` | `0.75` | `WINDOW_SLACK_THICK` (`distributions.py`; window family) |
| `[discretization]` | `concentration_cv` | `0.1` | `CONCENTRATION_CV` (`tail.py`; windowing-eligibility gate) |
| `[bivariate]` | `min_axis_log2` | `4` | `_MIN_AXIS_LOG2` (`bivariate.py`; sibling of `total_log2`) |
| `[bivariate]` | `window_nines` | `9` | `bivariate._WINDOW_NINES` (per-axis 2-D window; was import-avoidance dup) |

*`FT_NOISE_FLOOR` (formerly `1e-10`) is **dropped**, not migrated — it has no live
call site (the experimental `min|ft|` switch was abandoned with zero accuracy
benefit, `dev/audit-numerics-2-findings.md`). `copula._PPF_CLIP` and
`spectral._DISTORTION_DENSITY_N` stay module-internal (implementation details,
not user knobs). Fields already shipped via sibling plans — `window_nines_trim`,
`window_pad_skew`, `sbj_tail_floor` (discretization), `total_log2` (bivariate),
the `[labels]` section — are in `config.py` already and are not re-listed here.*

*(A `[plotting]` section — `fig_w`/`fig_h`, `font_size`, colours, mplstyle override —
was considered and **rejected**; those constants stay in `constants.py`. See
`dev/done/plans-considered-and-rejected.md`.)*

## Module design — `src/aggregate/config.py` (new, leaf module)

Duck-typed leaf, imported by `underwriter` / `distributions` /
`multivariate`; imports only stdlib (and, for the non-tunable enums/labels, may
import `constants`) so there is no cycle.

**Ownership split (refined).** `config.py` owns (a) every tunable default *value*
as a dataclass field default — single source of truth — and (b) the **path
names** `USER_DIR_NAME` / `PACKAGE_DATA_DIR` / `TEST_SUITE_FILENAME` (it is the
"where things live" authority; `user_dir()` already needs the first).
`constants.py` is slimmed to a **dependency-free leaf**: `Validation`,
`DefectiveDistributionWarning`, `REINS_LABEL_*` (5 strings). **Why it survives
(not legacy):** `utilities.py` imports `Validation` and `spectral.py` uses
`DefectiveDistributionWarning`, and both modules are imported *by*
`distributions`/`portfolio` — so relocating those types upward would create
cycles (`utilities → distributions → utilities`, `spectral → distributions →
spectral`); a `Flag` enum + `Warning` subclass also do not belong in a settings
module. `REINS_LABEL_*` stay here (structural MultiIndex keys referenced by
literal in `multivariate.py` / plot labels / tests — not user-settable), which
also keeps the `constants.py` name honest. `WL` is **deleted** (legacy custom
level → `logger.warning`). **No back-compat re-exports**: the internal sites that
read tunables from `constants` today — `moments` (`VALIDATION_NOISE`), `underwriter`
(`VALIDATION_EPS`/`bucket_sizing_p`), `distributions`/`portfolio`
(`ALIASING_RATIO`, `*_BUCKET_DEFAULT`, `WINDOW_NINES`, …) — are repointed to
`config`/`get_settings()`. Name left as `constants.py` per author (a `types.py`
rename would mystify).

- **Nested frozen dataclasses** `BuildSettings`, `DiscretizationSettings`,
  `ValidationSettings`, `MultivariateSettings`, `PlottingSettings`, composed into
  `Settings`. Access reads `settings.build.log2`,
  `settings.discretization.dsev_bucket`, `settings.multivariate.window_nines`,
  etc. (`MultivariateSettings` starts with just `window_nines`; the axis
  sizing / coverage knobs land here once the multivariate tuning settles them —
  see `dev/plan-multivariate-punchup.md`.)
- `user_dir() -> Path` — `Path.home() / USER_DIR_NAME`; **not** mkdir'd on read
  (only `write_default_config` creates it). (Underwriter's existing `user_dir`
  property delegates here so there is one definition.)
- `config_path() -> Path | None` — honours `AGGREGATE_CONFIG` (`none` → `None`,
  a path → that path, unset → `user_dir()/'config.toml'`).
- `load_settings(*, path=..., env=...) -> Settings` — pure function applying the
  cascade defaults → file → env. (kwargs are the highest layer but they live at
  the `build` / constructor call sites, so `load_settings` covers the lower
  three.) Records a per-field **source map** for `show_settings`.
- **Resolution: lazy singleton + explicit reload.** `get_settings() -> Settings`
  builds the singleton on **first use** and caches it (config read **once per
  session** — never silently changing mid-run, which is the reproducible, no-
  magic behaviour). `reload_settings() -> Settings` re-reads file+env **and**
  refreshes the module-level `build` Underwriter; it is the escape hatch for
  tests and for "I just edited the file". Documented contract: *to change config
  after import, set env before import, pass kwargs, or call `reload_settings()`.*
  (Note: because the module-level `build` is constructed at import, "first use" ≈
  import for normal sessions; making `build` itself lazy via PEP 562
  `__getattr__` is a later option if programmatic-after-import config is wanted.)
- **Env allow-list** with typed coercion: `AGGREGATE_LOG2`, `AGGREGATE_BS`,
  `AGGREGATE_DATABASES` (comma-sep), `AGGREGATE_REINS_BUCKET`,
  `AGGREGATE_DSEV_BUCKET`, `AGGREGATE_VALIDATION_EPS`, plus `AGGREGATE_CONFIG`
  (handled in `config_path`). Unknown `AGGREGATE_*` warn.
- `write_default_config(path=None, *, force=False) -> Path` — copies bundled
  `data/config.default.toml` to `~/.aggregate/config.toml`; refuses to clobber an
  existing file unless `force` (returns the path written).
- `describe_settings(settings) -> list[tuple[key, value, source]]` — backs
  `Underwriter.show_settings()`.
- Unknown **file** keys → `warnings.warn` (one per key), value dropped.

**Shipped template** (`data/config.default.toml`): every line commented out and
**annotated with valid options / sensible range**, e.g. `legend_font` lists the
matplotlib size words, `dsev_bucket`/`reins_bucket` show `linear|nearest`,
`bucket_sizing_p` notes the `>1 ⇒ nines` convention, `log2` notes the memory
cost, `noise` notes the `1e-12..1e-14` band.

## Files

- `src/aggregate/config.py` — **new**: the dataclasses (with the tunable default
  *values*), cascade loader, lazy singleton + `reload_settings`, env allow-list,
  `write_default_config`, `describe_settings`.
- `src/aggregate/data/config.default.toml` — **new**: fully-commented annotated
  template (the canonical reference; also what `write_default_config` copies).
- `src/aggregate/constants.py` — **slim**: keep only `Validation`,
  `DefectiveDistributionWarning`, `REINS_LABEL_*`; **delete** every tunable
  literal, **the path names** (→ `config.py`), and **`WL`**; **no re-exports**.
  Update `__all__`.
- **`WL` removal (sweep):** replace the 12 `logger.log(WL, …)` calls with
  `logger.warning(…)` in `underwriter.py` (×3), `distributions.py` (×7),
  `portfolio.py` (×2); drop the `WL` import from all three.
- `pyproject.toml` — bump `requires-python` to `>=3.11` and **drop the
  `Programming Language :: Python :: 3.10` classifier** (enables direct
  `import tomllib`, no new dependency); add `data/*.toml` to
  `tool.setuptools.package-data`; version bump.
- `src/aggregate/underwriter.py` — `__init__` reads `get_settings().build.*`
  for `log2` / `databases` / `update` defaults; rewrite module-level `build`
  (`:1086`) to take its log2/databases/update from settings (kills the 10-vs-16
  split); `info` gains the `config:` line; add `show_settings()`,
  `write_default_config()`, `reload_settings()` (thin delegates to `config.py`);
  `user_dir`/`default_dir` read the path names from `config`. **Full
  `recommend_p` → `bucket_sizing_p` rename** in `build`/`build_many`/`update`
  signatures, the `hints{}` key, `_resolve_hints`, and the `_NOTE_SETTINGS_RE`
  regex (default source = `settings.discretization.bucket_sizing_p`). No alias.
- `src/aggregate/distributions.py` — `Aggregate.__init__` bucket defaults read
  `get_settings().discretization.*` instead of the bare `*_BUCKET_DEFAULT`
  constants; `validation_eps` default reads `settings.validation.eps`;
  `WINDOW_NINES` use reads `settings.discretization.window_nines`. Rename the
  `recommend_p` param + `RECOMMEND_P` references to `bucket_sizing_p` /
  `BUCKET_SIZING_P` (`update`, `recommend_bucket`'s call path, `_bs_window`,
  `estimate_agg_window`).
- `src/aggregate/portfolio.py` — same `recommend_p` → `bucket_sizing_p` rename
  across `best_bucket` / `_bs_window` / `update`; read `ALIASING_RATIO` etc. from
  `get_settings()`.
- `src/aggregate/moments.py` — read `VALIDATION_NOISE` from `config` (it imports
  it today).
- `src/aggregate/multivariate.py` — drop the duplicated `_WINDOW_NINES = 12`;
  read its **own** `settings.multivariate.window_nines` (first-class field,
  default 12, independently tunable). The leaf `config` import replaces the
  import-avoidance hack.

*(`src/aggregate/style.py` is untouched — the `[plotting]`/mplstyle-override work
was rejected; see `dev/done/plans-considered-and-rejected.md`.)*

## Phases

**Phase 1 — config core + headline settings (this iteration).**
`config.py` (dataclasses owning the tunable defaults, cascade defaults→file→env,
lazy singleton + `reload_settings`, env allow-list, unknown-key warnings),
bundled annotated `config.default.toml`, `requires-python>=3.11` + stdlib
`tomllib`, slim `constants.py` (drop tunables/paths/`WL`, no re-exports), wire `[build]`
(`log2` / `databases` / `update`), `[discretization]` (`reins_bucket` /
`dsev_bucket` / `bucket_sizing_p` / `window_nines`, incl. the `RECOMMEND_P →
BUCKET_SIZING_P` rename), `[multivariate].window_nines` (its own field,
replacing the `_WINDOW_NINES` dup), and `[validation].eps` + `.noise` through
settings, **fix the log2 split**,
`Underwriter.info` config line, `show_settings()`, `write_default_config()`.
Tests. (The numerics-pending floors wait for Phase 2; graphics dropped entirely.)

**Phase 2 — surviving numerics floors + stranded sizing knobs (scoped 2026-06-19).**
The numerics review (numerics-0…4, a55–a80) is complete, so the floor values are
settled. Wiring follows the **established pattern**: a module-level UPPERCASE
constant captured from `get_settings()` at import — exactly as `distributions.py:68`
(`WINDOW_NINES = get_settings().discretization.window_nines`), `moments.py:15`,
and `bivariate.py:73` already do. So the existing **call sites are untouched**;
only the *source line* flips from a `constants.py` literal to a `get_settings()`
read, and the literal is deleted from `constants.py`.

Changes:

- **Drop `FT_NOISE_FLOOR`.** Dead since numerics-2: the experimental
  `min|ft| < FT_NOISE_FLOOR` switch was abandoned with zero accuracy benefit
  (`dev/audit-numerics-2-findings.md`); grep finds no call site. Remove the
  constant, its `__all__` entry, and its comment block from `constants.py`.

- **`[validation]` += `aliasing_ratio` (10), `exeqa_noise_floor` (1e-4),
  `deficit_materiality` (1e-4).** Add the three fields to `ValidationSettings`
  (with docstrings). Repoint by module-level capture (names unchanged):
  - `distributions.py` — `ALIASING_RATIO = get_settings().validation.aliasing_ratio`
    (call site `:5650` / log string `:5651` untouched).
  - `portfolio.py` — `ALIASING_RATIO` and `EXEQA_NOISE_FLOOR` captured the same
    way (call sites `:2846`/`:2847`, `:3999` untouched).
  - `spectral.py` — `DEFICIT_MATERIALITY = get_settings().validation.deficit_materiality`
    (call sites `:219`/`:225`/`:236` untouched).
  - Delete `ALIASING_RATIO`, `EXEQA_NOISE_FLOOR`, `DEFICIT_MATERIALITY` (and the
    already-dropped `FT_NOISE_FLOOR`) from `constants.py` + `__all__`.
  - *`deficit_materiality`'s 1e-4 value is an open author item (numerics-3 flags
    it for review). Config exposure is what makes revisiting it a one-line edit.*

- **`[discretization]` += `window_log2_growth` (4), `window_slack_thick` (0.75),
  `concentration_cv` (0.1).** These window-sizing knobs sit literally between
  fields already in `[discretization]` (`distributions.py:75`/`:93`, `tail.py:79`).
  Add to `DiscretizationSettings`; `distributions.py` `WINDOW_LOG2_GROWTH` /
  `WINDOW_SLACK_THICK` and `tail.py` `CONCENTRATION_CV` become `get_settings()`
  captures. (`CONCENTRATION_CV` is the windowing-eligibility CV gate — same
  machinery — so it lands in `[discretization]`, not `[validation]`.)

- **`[bivariate]` += `min_axis_log2` (4).** Add to `BivariateSettings`;
  `bivariate.py:79` `_MIN_AXIS_LOG2` becomes a `get_settings()` capture (sibling
  of `total_log2` / `window_nines` already there).

- **Template + tests.** Add the new keys to `data/config.default.toml` (commented,
  annotated with range/units, matching the existing template style); extend
  `tests/test_config.py` with an override case for one field per touched section
  (`validation.aliasing_ratio`, `discretization.window_slack_thick`,
  `bivariate.min_axis_log2`) and assert `FT_NOISE_FLOOR` is gone from
  `aggregate.constants`.

- **Left in `constants.py` on purpose:** plotting constants (config rejected),
  `Validation` / `DefectiveDistributionWarning` / `DefectiveDistributionError`
  (types), `REINS_LABEL_*` (structural keys), the `INFO_*` display convention.

- **Left module-internal (not config):** `copula._PPF_CLIP` (masked scratch
  guard) and `spectral._DISTORTION_DENSITY_N` (plot-density resolution) —
  implementation details, not user knobs.

- Version bump + `CHANGELOG.md` entry per the standing rules; README breaking-change
  note (`FT_NOISE_FLOOR` removed; the four migrated `constants` names moved to
  `aggregate.config`). *(Graphics/`[plotting]` was considered and dropped —
  `dev/done/plans-considered-and-rejected.md`.)*

## Verification

- New `tests/test_config.py`:
  - **defaults** when no file and no env (every field equals the `config.py`
    dataclass default; sources all `default`);
  - **`reload_settings()`** picks up a newly-written file and refreshes the
    module `build`;
  - **file overrides** default (write a temp `config.toml`, `load_settings`);
  - **env overrides file** (`AGGREGATE_LOG2=18` beats a file `log2=12`);
  - **kwargs override env** end-to-end (`build('…', log2=…)` wins);
  - **`AGGREGATE_CONFIG=none`** ignores an existing file (reproducibility);
  - **`AGGREGATE_CONFIG=/path`** relocates;
  - **unknown file key warns** and is dropped (assert the warning);
  - **`write_default_config`** writes a file that loads to *zero* overrides
    (all-commented == defaults) and refuses to clobber without `force`;
  - **`show_settings`** reports the correct source per field;
  - **log2 unified**: module `build.log2 == get_settings().build.log2`.
- **Import integrity**: `import aggregate` and `from aggregate import build`
  succeed (no cycle from the constants slim-down); `aggregate.constants` exposes
  only `Validation`, `DefectiveDistributionWarning`, `REINS_LABEL_*`; `rg` finds
  zero remaining `recommend_p` / `RECOMMEND_P` references (outside changelog/docs)
  and zero `WL` references.
- Existing suite stays green (993+), especially `bounded` / bucket / validation
  paths now reading settings; run with `UV_LINK_MODE=copy uv run pytest`.
- Smoke: `Underwriter().info` shows `config: (none — defaults)`; after
  `write_default_config()` + uncommenting `log2`, a fresh `Underwriter().info`
  shows `(loaded, 1 override)`.

## Close-out

- Bump `pyproject.toml` to the next `1.0.0a*`.
- README.rst bullet: user config file (`~/.aggregate/config.toml`), the
  precedence cascade, `write_default_config()` / `show_settings()` /
  `reload_settings()`, the `Underwriter.info` config line, and the resolved
  `log2` default. **Call out the breaking changes**: minimum Python now 3.11;
  `recommend_p` renamed to `bucket_sizing_p` (no alias); the tunable
  `aggregate.constants` names moved to `aggregate.config` (no re-export).
- Docs: a short "Configuration" page under the user guide (note pending a manual
  rebuild per CLAUDE.md; do not build docs in the loop).
- Move this plan to `dev/done/` when Phase 1 ships.

## Open / watch

- **Python ≥ 3.11 (DECIDED 2026-06-05).** `requires-python` bumps to `>=3.11`
  so stdlib `tomllib` is used directly — no `tomli` backport, no new dependency.
  Drops 3.10 support; the 3.10 classifier is removed. (Note for the close-out:
  call this out in the README as a min-Python bump, not just a feature add.)
- **Singleton resolution (DECIDED 2026-06-05).** Lazy singleton cached on first
  use + explicit `reload_settings()`; config is read once per session.
  `AGGREGATE_*` env must be set before `import aggregate` (or use kwargs /
  `reload_settings()`). Making `build` itself lazy (PEP 562) is a deferred option
  if programmatic-after-import config is ever wanted.
- **Ownership / `constants.py` fate (DECIDED 2026-06-05).** All tunables + path
  names → `config.py`. `constants.py` is **retained** (name kept) holding
  `Validation`, `DefectiveDistributionWarning`, `REINS_LABEL_*` — the two types
  can't move up (cycle with `utilities`/`spectral`), and the labels are
  structural keys, not settings. **`WL` deleted** → `logger.warning`. **No
  re-exports** (clean pre-1.0 break).
- **`recommend_p` rename (DECIDED 2026-06-05).** Full rename to `bucket_sizing_p`
  everywhere — constant `BUCKET_SIZING_P`, the public kwarg, the `hints{}` key,
  regexes, docs. **No alias.** Breaking, acceptable pre-1.0; called out in README.
- **`databases` default churn.** Shipping `databases = ["test_suite"]` preserves
  today's `build` behaviour. If the author later wants all three bundled files
  by default, it becomes a one-line config edit (`databases = ["default"]`) —
  no code change. Flagged, not decided here.
- **Numerics floors (RESOLVED 2026-06-19).** The numerics review (numerics-0…4,
  a55–a80) settled the values, so Phase 2 wires `aliasing_ratio`,
  `exeqa_noise_floor`, and `deficit_materiality`; `ft_noise_floor` is **dropped**
  (dead, no call site). The 1e-4 `deficit_materiality` level stays an open author
  item (numerics-3) but is now a config edit, not a code change.
- **Stranded sizing knobs (DECIDED 2026-06-19).** Gather the window-sizing trio
  `window_log2_growth` / `window_slack_thick` / `concentration_cv` →
  `[discretization]` and `min_axis_log2` → `[bivariate]` (they were half-migrated
  siblings of already-config'd fields). `copula._PPF_CLIP` and
  `spectral._DISTORTION_DENSITY_N` stay module-internal.
