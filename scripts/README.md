# scripts

Standalone maintenance tools for `aggregate`. Dependency-free (stdlib +
`aggregate` + `pandas`/`pyarrow`, all already in the project env).

## `freeze_knowledge.py` — knowledge-base regression harness

Snapshot every `agg`/`port` object's `describe` and filtered `density_df`
(`p_*`/`exeqa_*` columns) to parquet, then later rebuild and verify them to a
tight tolerance. The workflow is **freeze before a refactor, check after** — it
catches numerical drift in the FFT / allocation core.

### From Jupyter (recommended — no import overhead)

```python
from freeze_knowledge import freeze_knowledge, check_knowledge

m = freeze_knowledge(root=r"T:\tmp\aggregate_freeze")   # -> manifest dict
# ... do a refactor, restart kernel ...
failures = check_knowledge(m["directory"])              # [] == all match
```

* `freeze_knowledge(databases=("test_suite",), root=None, atol=1e-12)` returns
  the manifest dict (its `"directory"` key is the date folder it wrote).
* `check_knowledge(directory, atol=None)` returns the list of failing object
  names; `atol=None` reuses the tolerance stored at freeze time.

### From the shell

```powershell
uv run python scripts\freeze_knowledge.py freeze [-d test_suite ...] [--root DIR] [--atol 1e-12]
uv run python scripts\freeze_knowledge.py check  <DIR> [--atol 1e-12]
```

`check` exits non-zero on any mismatch.

### Notes

* **Default root** is `<OS temp>/aggregate_freeze/<YYYY-MM-DD>/`. The OS may
  sweep temp, so pass `--root` / `root=` for snapshots that must survive across
  days.
* `-d/--database` is repeatable; the default `test_suite` reproduces the
  imported `build`.
* Objects that can't be built standalone (e.g. measure-zero splices) are skipped
  with a warning and listed in the manifest.
* The parquet output is **not** tracked in git — this is a local aid, not a
  committed-baseline test.
