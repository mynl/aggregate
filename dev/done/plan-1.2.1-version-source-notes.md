# Execution notes, [library] step of plan-1.2.1-version-source

Executed 2026-10-14 as 1.2.1. The plan text is shared byte identical with aggregate-api (canonical), aggregate-relativity and aggregate-ntt; these notes are this repository's alone.

## Divergences

- **Plugin version read after resolving, from `sys.modules`.** `plugins._live_version` runs in `_run` once `register()` has resolved, so the plugin's top-level package is already imported. It reads `sys.modules` and never imports anything itself. A plugin whose `register()` fails to resolve still gets the live attribute if its package imported, else the metadata version read at discovery.
- **The module path comes from `EntryPoint.module`** on the packaged route and from the environment string on the other. A stub entry point without `module` falls back to metadata, which keeps `test_entry_point_route_is_discovered_and_versioned` unchanged.
- **One parametrized test covers both cases** the plan names (`test_plugin_version_prefers_the_package_attribute`): a package declaring `__version__` reports it, one without falls back to the distribution's `9.9.9`.
- **No `dev/TODO.md` item existed** for this plan, so none was ticked. The file carries the author's uncommitted edits and is left out of the commit entirely.
- **The execute-plan skill** also named the `pyproject.toml` version line (its section 2 grep and section 6), so it was updated beside version-bump and `CLAUDE.md`.
- **`uv.lock`** drops `version = "1.2.0"` from the root package, as uv does for a dynamic version, and reorders one `extras` list. Both are uv's own rewrite.

## Checks

- `uv build --wheel` produced `aggregate-1.2.1-py3-none-any.whl`.
- In `.venv`, `aggregate.__version__` and `importlib.metadata.version("aggregate")` both read `1.2.1` (uv rebuilt the editable install on the cache key).
