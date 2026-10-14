# Plan 1.3.0: a public, warning-free engine status for aggregate.ntt

(Version is a placeholder; roll into whichever release carries the other ntt
work.)

## Goal

Let a caller ask whether the compiled engine would be **used**, not merely
whether it imports, without touching private names or triggering the
protocol-mismatch warning.

## Why

`aggregate_api` 1.9.0 shows `aggregate-ntt <version>` in its About panel by
importing `aggregate_ntt` itself. That cannot tell "installed and used" from
"installed and ignored": the protocol check lives in `_resolve_engine` and
`_ENGINE_PROTOCOL`, both private, and calling `_resolve_engine('auto')` fires
the once-per-process `UserWarning` as a side effect.

## Change

Add to `aggregate/ntt.py`, exported:

```python
def engine_status() -> dict:
    """What 'auto' would pick, and why. Never warns, never raises."""
```

returning

| key | value |
|---|---|
| `engine` | `'rust'` or `'python'`, what `ntt_engine='auto'` resolves to |
| `installed` | `bool`, whether `aggregate_ntt` imported |
| `engine_version` | `aggregate_ntt.__version__` or `None` |
| `engine_protocol` | its `PROTOCOL_VERSION` or `None` |
| `library_protocol` | `_ENGINE_PROTOCOL` |
| `reason` | `None` when used, else one line: `'not installed'` or `'protocol 1, library speaks 2'` |

Implement by sharing the decision with `_resolve_engine` (factor out the pure
part) so the two cannot disagree; leave the warning in `_resolve_engine`.

## Acceptance

Tests with the existing `_FakeEngine`: not installed, matching protocol,
mismatched protocol. Each asserts the dict and that no warning is emitted
(`warnings.catch_warnings` + `simplefilter('error')`).

## Consumer

`aggregate_api` will add `ntt_status` to `/v1/meta` from this when present
(feature-detected with `hasattr`, so the 1.0.x floor still works), and the
About row reads `1.0.0a4` or `1.0.0a4, ignored (protocol 1, library 2)`.

## Execution log

Executed 2026-10-14 at **1.3.0**, one bump, one commit. Gate: tier 3, `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'`. Not numerics-touching, so no `RuntimeWarning` gate.

**As built.** `_auto_engine()` is the pure half, returning `(engine, reason)`; `_resolve_engine` calls it and returns `'rust'` when it says so, keeping its own raise and warn branches for the two fallback cases (the mismatch branch lost one indent level, behavior unchanged). `engine_status()` wraps `_auto_engine()` in the six-key dict and is added to `__all__`. On this machine it reports `engine 'rust'`, `engine_version '1.0.0-alpha.4'`, protocol 2 on both sides.

**Divergences.**

- **Version.** The plan called its number a placeholder. Took 1.3.0, the filename's number, which `dev/plan-1.4.0-nttagg-arithmetic-exact-chart.md` already assumes lands first.
- **Mismatch fixture.** The test uses `_FakeEngine(99)`, matching the existing resolution tests, so its asserted reason is `'protocol 99, library speaks 2'` rather than the plan's illustrative `protocol 1`.
- **Two checks beyond the three asked for.** The mismatch test also asserts that `_PROTOCOL_WARNED` stays `False` and that a following `_resolve_engine('auto')` still warns, since a status call must not spend the warning. A fourth test, `test_engine_status_agrees_with_resolution`, asserts `engine_status()['engine'] == _resolve_engine('auto')` for the real installed engine, none, a matching fake and a mismatched fake.
- **No `dev/TODO.md` tick.** No tracked item existed for this plan, and the file holds the author's uncommitted edits, so it was left out of the commit.
- **No `dev-files.md` row.** The control list was retired on 2026-09-25.
