# Plan: exponential tilting as a pedagogy function (Option B)

## Goal

Bring back exponential tilting (Grübel & Hermesmeier; Embrechts–Frei) **purely to
illustrate it** in the numerical-methods docs — it is never used operationally.
The production convolution core stays tilt-free; a self-contained **pedagogy
function** reproduces the doc example.

## Why a pedagogy function (not the core)

The core `ft`/`ift`/`update_work` dropped the `tilt`/`tilt_vector` arguments in
the refactor and the convolution is clean. Tilting is a teaching device about
aliasing control, so it belongs in `pedagogy.py` (not `ft.py` — CLAUDE.md / H5
explicitly want demo/figure generators *out* of `ft.py`). One function, zero core
impact.

## The doc example it must cover

`docs/2_user_guides/problems/010_gh_example.rst` — a Levy-severity aliasing demo.
Current (broken) code:

```python
a = build('agg L 20 claim sev levy poisson', update=False)
bs = 1
a.update(log2=16, bs=bs, padding=2, normalize=False, tilt_vector=None)   # "accurate"
df = a.density_df.loc[[1, 10, 100, 1000], ['p_total']] / a.bs
...
log2 = 10
for tilt in [None, 1/1024, 5/1024, 25/1024]:
    a.update(log2=log2, bs=bs, padding=0, normalize=False, tilt_vector=tilt)
    df[f'tilt {tilt:.4f}'] = a.density_df.loc[[1,10,100,1000], ['p_total']] / a.bs
```

Two breakages: (a) the `tilt_vector=` kwarg no longer exists; (b) the "accurate"
line only breaks on `tilt_vector=None` — `normalize=False` is still supported by
`update`, so dropping `tilt_vector=None` fixes that line on its own.

## The function (`pedagogy.py`)

A standalone tilted single-aggregate convolution, self-contained (its own
tilted rfft/irfft), parameterised exactly enough for the example:

```python
def tilted_aggregate_density(agg, *, log2, bs, padding=0, tilt=None,
                             normalize=False):
    """Aggregate density via FFT with optional exponential tilting (pedagogy).

    Illustrates Grübel–Hermesmeier tilting for aliasing control (Embrechts–Frei
    recommend tilt*N ≤ 20). NOT part of the production update path -- a teaching
    helper for the numerical-methods docs.

    Parameters
    ----------
    agg : Aggregate or str
        A built (or buildable) aggregate; its discretized severity and frequency
        pgf are used.
    log2, bs : grid parameters.
    padding : int, default 0.
    tilt : float or None
        Tilt amount theta; the tilt vector is exp(-theta * arange(N)). None = no
        tilt. (theta*N <= 20 recommended.)
    normalize : bool, default False
        Match the doc (which compares un-normalized tail probabilities).

    Returns
    -------
    pandas.Series
        Aggregate density indexed by loss (so the doc can read p_total/bs at
        chosen x).
    """
```

Implementation sketch: build the agg with `update=False` (or accept a built one),
discretize the severity on the `(log2, bs)` grid, then apply the **classic tilted
transform** — `z*tilt → rfft → freq_pgf → irfft → /tilt` (padding handled in the
rfft length), exactly the old `ft`/`ift` tilt logic, kept local to this function.
A tiny `tilt_vector(theta, N)` helper (`exp(-theta*arange(N))`) can sit alongside.

Optionally provide a one-call exhibit builder
`gh_tilting_exhibit(...) -> DataFrame` that assembles the whole comparison
(accurate column + tilt sweep), so the rst is a single call — recommended, it
keeps the doc tidy and the logic tested in one place.

## Doc rewrite

`010_gh_example.rst`:
- "accurate" line → ordinary `a.update(log2=16, bs=bs, padding=2,
  normalize=False)` (drop `tilt_vector=None`).
- the tilt loop → call `tilted_aggregate_density(a, log2=10, bs=bs, padding=0,
  tilt=tilt)` (or the single `gh_tilting_exhibit(...)` call).
- Keep the narrative (tilt reduces aliasing; `theta*N ≤ 20`).
- RST source only; do not touch `docs/_build/`.

## Verification
1. **Reproduces the example** — the tilted columns show the expected aliasing
   reduction at x=1,10,100,1000 vs the log2=16 accurate column (eyeball against
   the prior published table; values should track the old output).
2. **Tilt = None path** equals an ordinary (untilted) convolution at the same
   grid.
3. **`uv run pytest`** green; add a small test asserting the function runs and
   `tilt=None` matches the plain density.
4. No core/​`freeze_knowledge` impact (production path untouched).

## Housekeeping (standing rules)
- Bump `pyproject.toml` `1.0.0a*` (new pedagogy function + doc).
- `CHANGELOG.md`: "Added `pedagogy.tilted_aggregate_density` (Grübel tilting
  illustration); rewired the GH numerical-methods example. Production convolution
  remains tilt-free."
- `dev/TODO.md`: note the tilting illustration done, it is F2. 
- Note the doc rebuild is pending (per CLAUDE.md, not run in the loop).
- Move this plan to `dev/done/plan-gh-tilt.md` on landing.

## Out of scope
- Restoring `tilt` in the core `ft`/`ift`/`update_work` (rejected — keep the core
  clean).
- Any operational/default tilting. Padding remains the production aliasing control.

---

## Execution notes (landed 1.0.0a46)

Executed as written; the recommended `gh_tilting_exhibit` builder was included.

1. **Function shape.** `tilted_aggregate_density(agg, *, log2, bs, padding=0,
   tilt=None, normalize=False)` calls `agg.update(...)` purely to discretize the
   severity and set up the frequency PGF on the target grid, then runs the
   tilted transform locally (`sfft.rfft`/`irfft`, no change to the core `ft`/`ift`).
   It mirrors the pre-refactor tilt logic exactly: `tilt_vector(θ, N) =
   exp(-θ·arange(N))`, forward `z·tilt`, inverse `/tilt`. Verified
   `tilt=None` reproduces the plain `update` `agg_density` to **0.0** abs diff.
2. **`gh_tilting_exhibit`** assembles accurate (log2=16, padding=2) + closed-form
   `exact` (factored into `_gh_levy_exact`, the Levy-stability conditional sum) +
   the tilt sweep, so the rst is two `qd(...)` calls. Reproduces the published
   table (heaviest tilt θ·N=25 column matches the accurate column to ~1e-3).
3. **Doc** `010_gh_example.rst` rewired to the helpers; the broken
   `tilt_vector=None` kwargs removed. RST source only — **doc build pending**
   (per CLAUDE.md, not run in the loop).
4. **Tests** `tests/test_pedagogy_tilt.py` (4 cases): tilt vector shape,
   `tilt=None` == plain convolution, aliasing reduction at x=1, exhibit table
   coherence. DecL program mirrored in `test_decl.agg` (new section GH).
5. **No core impact** — freeze/check: all 146 knowledge-base objects unchanged.

Pre-existing, untouched: a `ruff` F841 (`b` assigned-but-unused at
`pedagogy.py:155`, inside `plot_similar_risks_graphs`) predates this work and is
left alone (unrelated to tilting).
