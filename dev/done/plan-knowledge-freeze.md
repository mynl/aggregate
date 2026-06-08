# Plan: promote the knowledge-freeze regression harness to `scripts/`

## Motivation

The freeze/check harness prototyped in `hacks/aggregate_hacks.py` is more than
scratch: it snapshots every `agg`/`port` object's `describe` and filtered
`density_df` to parquet, then later rebuilds and verifies them to a tight
tolerance. That is a real maintenance tool — freeze before a refactor, check
after — and is directly useful for the N-track core-compute rewrite. Promote a
clean, dependency-free version into the tracked tree so other contributors (and
future-me) can use it.

## Decisions (settled)

- **Location / name:** `scripts/freeze_knowledge.py` (new top-level `scripts/`
  dir).
- **No third-party deps:** CLI via stdlib `argparse`, not `click`. Runnable as
  `uv run python scripts/freeze_knowledge.py ...` with no overlay.
- **Function-first:** the module exposes importable `freeze(...)` and
  `check(...)` functions (the real API, for Jupyter use); `argparse main()` is a
  thin wrapper over them.
- **Default root = OS temp:** `Path(tempfile.gettempdir()) / "aggregate_freeze"
  / <YYYY-MM-DD>`. Machine-agnostic, no repo pollution. Override with
  `--root` / `root=` for durable storage. The hardcoded `T:\tmp\...` default is
  dropped.
- **No committed baseline:** the parquet output is never tracked (filtered
  `density_df` is ~65k rows × ~146 objects — too much binary for git). This is a
  *local* aid, not a committed-baseline CI test.
- **Personal uber version stays untracked.** `hacks/aggregate_hacks.py` keeps
  its `click.Group` + `uber.plugins` entry point; its commands import and call
  the tracked `freeze`/`check`. `click` lives only in `hacks/` — never in the
  repo. (Handled outside this plan; noted here only so the tracked module's
  function API stays import-friendly.)

## Public API of `scripts/freeze_knowledge.py`

```python
def freeze(databases=("test_suite",), root=None, atol=1e-12) -> dict:
    """Build every agg/port program in `databases` and snapshot frames to
    parquet under `root/<today>/`. `root=None` -> OS temp default.
    Returns the manifest dict (also written as _manifest.json)."""

def check(directory, atol=None) -> list:
    """Rebuild the objects recorded in `directory/_manifest.json` and compare
    recomputed frames to the frozen parquet. `atol=None` -> manifest's atol.
    Returns the list of failing object names (empty == all match)."""
```

CLI: `freeze_knowledge.py freeze [-d NAME ...] [--root DIR] [--atol F]` and
`freeze_knowledge.py check DIRECTORY [--atol F]`. The `check` subcommand exits
non-zero on any mismatch.

Behaviour is otherwise identical to the validated `hacks` prototype:
- knowledge filtered to kinds `agg`, `port`;
- per object: `describe` + `density_df.filter(regex='p_|exeqa_')`;
- files `<kind>_<sanitized name>_describe.parquet` / `..._density.parquet`;
- `_manifest.json` records created-time, aggregate version, database list,
  atol, the per-object program text + filenames, and any skipped objects;
- objects that fail to build standalone are skipped-with-warning, not fatal;
- comparison via `pandas.testing.assert_frame_equal(rtol=0, atol=...)`.

## Files

| File | Change | Touches `src/`? |
|------|--------|:---:|
| `scripts/freeze_knowledge.py` | **new** — argparse port of the prototype, function-first, temp default | no |
| `scripts/README.md` | **new** — short usage (CLI + Jupyter), points at this tool | no |
| `pyproject.toml` | version bump `1.0.0a37` → `1.0.0a38` | no (metadata) |
| `CHANGELOG.md` | new `## 1.0.0a38` section describing the tool | no (docs) |
| `dev/TODO.md` | add a `T4` row to the Priorities table (Tests track); refresh "Last updated" | no (docs) |
| `dev/plan-knowledge-freeze.md` | this file → move to `dev/done/` on landing | no |

**No file under `src/aggregate/` or `tests/` is modified.** The library is
untouched; this is purely additive tooling + housekeeping metadata.

## TODO.md entry (proposed row)

Add under the Tests track in the Priorities & dependencies table:

```
| ✅ | T4 | Knowledge-freeze regression harness (scripts/freeze_knowledge.py) | B | — | N* (it verifies them) |
```

(Marked done on landing. Phase `[B]` — a maintenance aid, does not block the
alpha→beta cut. Listed parallel-safe with the N-track since its whole purpose is
to catch drift there.)

## CHANGELOG.md entry (draft)

```
## 1.0.0a38

### New: knowledge-freeze regression harness (`scripts/freeze_knowledge.py`)

A standalone, dependency-free tool to snapshot and verify knowledge-base
outputs across refactors. `freeze` builds every agg/port program with default
parameters and writes each object's `describe` and filtered `density_df`
(`p_*`/`exeqa_*` columns) to parquet, plus a `_manifest.json` recording the
exact program text, database list, and library version. `check` rebuilds from
the manifest and verifies the recomputed frames match the snapshot to a tight
absolute tolerance (default 1e-12). Importable `freeze()` / `check()` functions
for Jupyter; argparse CLI for the shell. Default output goes to an OS-temp
dir; pass `--root` for durable storage. Output parquet is never committed.
```

## Execution order

1. Write `scripts/freeze_knowledge.py` (argparse port of the prototype; temp
   default; importable functions) and `scripts/README.md`.
2. Smoke-test: `uv run python scripts/freeze_knowledge.py freeze -d test_suite2`
   then `check` the produced dir → expect all-match. (Writes to OS temp.)
3. Bump `pyproject.toml` to `1.0.0a38`; add the `CHANGELOG.md` section.
4. Add the `T4` row + refresh "Last updated" in `dev/TODO.md`.
5. Move this plan to `dev/done/plan-knowledge-freeze.md`.

## Out of scope / explicitly not doing

- No committed parquet baseline; no pytest integration; no CI wiring.
- No change to the library, the test suite, or any DecL grammar.
- The personal uber/click wrapper and `uber-agg.ps1` launcher stay untracked in
  `hacks/` and at repo root respectively.
