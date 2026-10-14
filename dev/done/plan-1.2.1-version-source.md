# Plan [version-source]: every package reports its version live, so no bump needs a sync

**One plan, four identical copies.** It lives in all four repositories under
the file name each repository's next version gives it:

| repository | path | this step's version |
|---|---|---|
| aggregate (LIB) | `V:\worktrees\aggregate_REFACTOR\dev\plan-1.2.1-version-source.md` | 1.2.0 to **1.2.1** (fix) |
| aggregate-relativity | `V:\dev\aggregate-relativity\dev\plan-0.10.0-version-source.md` | 0.9.0 to **0.10.0** (its rule: every change takes the minor) |
| aggregate-ntt | `V:\dev\aggregate-ntt\dev\plan-1.0.0a5-version-source.md` | 1.0.0a4 to **1.0.0a5** (its alpha line) |
| aggregate-api | `V:\dev\aggregate-api\dev\plan-1.9.1-version-source.md` | 1.9.0 to **1.9.1** (fix) |

The **aggregate-api copy is canonical.** If review changes the text, the change
is made there and copied to the other three, so all four stay byte identical.
Each repository executes only its own step below, but every reader gets the
whole picture. Steps run in the order given, each a separate bump and commit in
its own repository.

## Goal

A version bump edits one line in one file and changes nothing in any
environment. Every place that reports a version (the api's `/v1/meta`, its
About panel, the plugin manifest, audit records) reads the live value, so an
editable install is never stale and no bump needs `uv sync`.

## The problem

All four are installed **editable** in the api's development environment. An
editable install has two parts: a pointer to `src/`, so code edits are live,
and a `*.dist-info` folder written **once, at install time**, recording the
version. `importlib.metadata.version()` reads that folder, so after a bump it
keeps returning the old number until the package is reinstalled.

The api reads its own version and `aggregate`'s that way. So its bump skill
ends with `uv sync --extra dev`, purely to rewrite that folder. But `uv sync`
also makes the environment match `uv.lock` **exactly**: it uninstalls the
relativity plugin (never a dependency, by design), puts the released
`aggregate` back over the worktree, and drops `aggregate-ntt`. Every bump
therefore wipes the co-development environment, and it is rebuilt from notes
kept in three places (the api `CLAUDE.md`, its bump skill, a memory file) plus
`V:\dev\RELEASE-PROCESS.md`. A bump on 2026-10-14 showed the cost: the restore
silently moved `aggregate` from 1.0.1 to the worktree's 1.2.0 as a side effect.

The survey, as of 2026-10-14:

| package | version written in | `__version__` from | state |
|---|---|---|---|
| aggregate | `pyproject.toml` | `importlib.metadata` | stale in an editable install after a bump |
| aggregate-relativity | `pyproject.toml` **and** a literal in `__init__.py` | the literal | **drifted**: 0.9.0 against `'0.8.0'`, so `test_register.py`'s `plugin.version == __version__` is red |
| aggregate-ntt | `Cargo.toml` (`1.0.0-alpha.4`) **and** `pyproject.toml` (`1.0.0a4`) | compiled in from Cargo | two copies kept equal by hand |
| aggregate-api | `pyproject.toml` | `importlib.metadata` | stale until a sync |

The house rules already require a single-sourced version. Three of the four
break that rule, and the fourth reads its single source the stale way.

## The rule, for all four

1. **One written source.**
   * Pure Python (setuptools, all three): `__version__ = "x.y.z"` as a
     **plain string literal** in `src/<package>/__init__.py`. `pyproject.toml`
     drops `version =`, adds `dynamic = ["version"]` to `[project]`, and adds
     `[tool.setuptools.dynamic] version = { attr = "<package>.__version__" }`.
     The literal matters: setuptools then reads it statically, without
     importing the package (importing `aggregate` pulls in pandas).
   * Rust (maturin): `Cargo.toml` is the source. `pyproject.toml` drops
     `version =` and adds `dynamic = ["version"]`; maturin converts
     `1.0.0-alpha.5` to `1.0.0a5` for the wheel.
2. **Read live, never from metadata.** Runtime code reports
   `<package>.__version__`. `importlib.metadata.version()` is not used at
   runtime for any of these four. (A wheel installed into a clean environment,
   as `scripts/release-check.ps1` does, has correct metadata by construction,
   so it may keep reading metadata.)
3. **uv must notice the file.** uv caches a local project's metadata keyed on
   `pyproject.toml`; with a dynamic version it has to watch the file the
   version lives in too, or `uv lock` and `uv run` can carry a stale version.
   Each setuptools project adds
   `[tool.uv] cache-keys = [{ file = "pyproject.toml" }, { file = "src/<package>/__init__.py" }]`;
   ntt adds `Cargo.toml` alongside its existing keys, if any.
4. **A bump touches no environment.** Edit the one source, update
   `CHANGELOG.md`, relock if the project keeps a lock, commit. A `uv sync` is
   for a **dependency** change only.

## Steps

### [library] aggregate, 1.2.1

* `src/aggregate/__init__.py`: replace `__version__ = _pkg_version("aggregate")`
  (and its import) with the literal `__version__ = "1.2.1"`.
  `pyproject.toml` per rule 1 and rule 3.
* `src/aggregate/plugins.py`: a plugin's reported version prefers the live
  `__version__` of the plugin's top-level package, falling back to
  distribution metadata only when the package has none. Today it reads
  metadata (`_distribution_version`, and `dist.version` on the entry-point
  path), so an editable plugin reports its install-time version.
* Version-bump skill and `CLAUDE.md`: name `src/aggregate/__init__.py` as the
  one place the version lives, and remove any sync or reinstall whose only job
  was refreshing the version.
* Tests: a plugin whose package declares `__version__` reports that, and one
  without falls back to metadata.
* Check: `uv build` produces `aggregate-1.2.1-*.whl`.

### [relativity-plugin] aggregate-relativity, 0.10.0

Depends on [library] only for the plugin version to read live; it is correct
either way.

* `src/aggregate_relativity/__init__.py`: `__version__ = "0.10.0"`.
  `pyproject.toml` per rule 1 and rule 3.
* `CLAUDE.md`: replace "the version lives in **two** places that must agree"
  with the single source.
* Check: `test_register.py` is green again (it is red today on the drift).
  `uv build` produces `aggregate_relativity-0.10.0-*.whl`.

### [ntt-engine] aggregate-ntt, 1.0.0a5

* `Cargo.toml` to `1.0.0-alpha.5`; `pyproject.toml` per rule 1 (Rust) and
  rule 3.
* `python/aggregate_ntt/__init__.py`: `__version__` becomes the **PEP 440**
  spelling of `engine_version()` (`1.0.0-alpha.5` to `1.0.0a5`, with `-beta.`
  to `b` and `-rc.` to `rc`), so the api's About panel, `aggregate`'s audit
  records and `pip` all show one form. `engine_version()` keeps the raw Cargo
  string. Both still come from the compiled binary, so a stale build stays
  visible, which is the reason the engine compiles its version in.
* `CLAUDE.md`: replace the "two files and they must not drift" paragraph with
  the single source.
* Tests: `__version__` equals the PEP 440 form of `engine_version()` and, in a
  fresh build, `importlib.metadata.version("aggregate-ntt")`.

### [api] aggregate-api, 1.9.1

Runs last, since it reads the other three.

* `src/aggregate_api/__init__.py`: `__version__ = "1.9.1"`. `pyproject.toml`
  per rule 1 and rule 3.
* Every runtime `importlib.metadata` read of a version becomes a `__version__`
  read: `app.py` (FastAPI `version=`), `routes/meta.py` (`/v1/health`,
  `/v1/meta`, the `greater-tables` asset ETag), `routes/status.py`,
  `routes/decl.py` (the 404 text). For `aggregate` this is correct on the
  released 1.0.1 as well, whose `__version__` is its metadata.
* Version-bump skill: delete section 8 (the re-sync) and the paragraph in
  section 7 that leans on it. `CLAUDE.md`: delete "Re-sync after a version
  bump, or `/v1/meta` lies", and rewrite the co-development notes to say a sync
  happens only on a dependency change, followed by the restore.
  `V:\dev\RELEASE-PROCESS.md`, the "Traps" line: drop "every bump runs one" and
  "reports a stale version until you re-sync".
* Memory `relativity-plugin-pruned-by-sync.md`: reword to "after a dependency
  sync", since bumps no longer sync.
* Tests: `/v1/meta` and `/v1/health` report `aggregate_api.__version__` and
  `aggregate.__version__`.
* Check: `uv build` produces `aggregate_api-1.9.1-*.whl`, and
  `.\scripts\release-check.ps1` passes (the wheel path, where metadata and
  `__version__` must agree).

## Acceptance, end to end

After all four land, with the co-development environment restored once:

1. `/v1/meta` reports api 1.9.1 and aggregate 1.2.1, and the About plugins line
   reports relativity 0.10.0; `aggregate-ntt` shows `1.0.0a5` when installed.
2. **The real test, done once and reverted:** change the api's `__version__`
   to `1.9.2`, restart the server without any `uv` command, and `/v1/meta`
   says `1.9.2`. The relativity plugin and the worktree `aggregate` are still
   installed (`uv pip list` unchanged).
3. No document in any of the four repositories, nor `RELEASE-PROCESS.md`,
   still tells a bump to sync.

## Hazards for whoever executes

* **Uncommitted work.** On 2026-10-14: LIB has `dev/TODO.md` modified and an
  untracked `dev/plan-1.3.0-ntt-engine-status.md`; relativity has
  `src/aggregate_relativity/chart.py` modified; ntt has an untracked
  `uv.lock`. Stage by path only. Never sweep the author's work into a step's
  commit.
* **A concurrent LIB session** may be editing the library. Confirm with the
  author before starting [library].
* **The api's step changes its bump skill mid-flight.** Run the [api] bump by
  the *new* procedure (no sync), which is itself the first live test.
* Never push, in any repository.

## Out of scope

* `greater-tables`, a fifth repository with the same pattern question. The api
  reads `greater_tables.__version__` after this, which is correct today
  because it is not installed editable.
* A declared co-development environment (a dev-only project listing the
  editable siblings, so a dependency sync restores rather than wipes). This
  plan removes the sync from every bump; that one would fix the rare
  dependency sync. Decide after living with this.
* Whether a plain `uv run` (without `--no-sync`) also reverts an editable
  `aggregate` to the locked release. Worth checking, and it belongs with the
  item above.
