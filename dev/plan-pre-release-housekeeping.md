# Plan: `cll` out of the default distortion list, three broken `pedagogy.py` functions repaired, and the CHANGELOG split

`[Pre-2.0-Housekeeping]`. Drafted 2026-10-31 against aggregate 1.43.1 (`03aaec6`). Three steps:
`[Research-Distortions]` (one bump), `[Pedagogy-Repairs]` (one bump), and `[Changelog-Split]` (no
bump; the author commits). Agreed by the author on 2026-10-30
(`D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`, items 2, 3 and 5). On 2026-10-31 the
author asked for a plan "even if very easy", so that every agreed item has a file. **Status: ready
to run.** Approving this plan is the author's word on the CHANGELOG split, which item 5 of the
control document left waiting for it.

No other open plan touches this code, so the plan can run at any time before the release. It is
ninth in the control document's order. The one constraint on timing is in `[Changelog-Split]`: the
author commits that step before any later bump, because a bump stages all of `CHANGELOG.md`.

## Goal

| Item | Today | After |
|:--|:--|:--|
| `Distortion.available_distortions()` | `('ccoc', 'ph', 'wang', 'dual', 'tvar', 'cll')` | the same without `cll`, equal to `DEFAULT_CALIBRATION_DISTORTIONS` |
| Research families (`cll`, `clin`, `lep`, `ly`) in the docs | listed with no distinction | a docs heading and a `Notes` paragraph in the `Distortion` docstring; no namespace |
| `pedagogy.similar_risks_example()` | `TypeError` from `Bounds.plot_envelope(axs=...)` | runs and returns its frame |
| `pedagogy.plot_similar_risks_graphs(..., p_reg=0.9)` | `AttributeError`, `Bounds._tvar_x_func` no longer exists | runs, using TVaR of the new portfolio under the bound |
| `ClassicalPremium.calibrate` and `price(method='Fischer')` | `AttributeError`, `self.p` is never set | run, with a keyword `fischer_power`, default 2 |
| `CHANGELOG.md` | 12,972 lines | about 535 lines and a pointer; the rest in `dev/done/CHANGELOG-before-1.0.md` |

## Census of `pedagogy.py`

The Using notes (`dev/issues-arising-from-using-notes.md`, line 50) count two broken functions but
do not identify them, and the control document identifies only `similar_risks_example`. To find the
other, every public function in the module was called at 1.43.1 on an input its docstring allows,
and attribute use on names holding a `Portfolio`, `Aggregate`, or `Bounds` was scanned statically.
Three functions fail. `similar_risks_example` passes arguments that no longer exist, and
`plot_similar_risks_graphs` calls a method that no longer exists, so these two are probably the
pair in the note. `ClassicalPremium` uses an attribute that is never set in this file or in the
copy at `D:/Projects/aggregate/notes/math/resources/pedagogy.py`, and the plan repairs it too. Since
all three are repaired, the plan holds whichever pair the note refers to.

The other functions ran. `plot_ruin_surplus_paths` and `natural_scale` need the `PZTest` portfolio
(`D:/Projects/aggregate/monograph/posts/2_technical_guides/2_x_pk.qmd`, which uses both), `plot_twelve`
needs a distortion applied first, and `chf_fourier3d_figure` needs `plotly`, as it reports. A census
of calls reaches only the branches its inputs take, as the `p_reg` branch below shows, since only the
static scan found it. A keyword argument passed on a path the census did not reach would still be
undetected.

## Steps

### `[Research-Distortions]`

**1. `cll` leaves the default list.** Set `strict_pricing = False` on `CLLDistortion`
(`spectral.py` 3683), with a one-line comment giving the reason: a research family, kept out of the
default list by the author on 2026-10-30. `cll` is the only research family this affects, because
`clin`, `lep`, and `ly` are already excluded by `has_mass_default`.

| Call | Today | After |
|:--|:--|:--|
| `available_distortions()` | `ccoc ph wang dual tvar cll` | `ccoc ph wang dual tvar` |
| `available_distortions(pricing=True, strict=False)` | also `bitvar wtdtvar clin lep ly` | unchanged, `cll` stays |
| `available_distortions(pricing=False)` | every documented kind | unchanged, so the DecL word list in `decl_words` is unchanged |

`Distortion('cll', ...)`, `distortion D cll 1.5` and an explicit calibration of `cll` are unchanged,
and `tests/test_distortion_calibrate.py` keeps its explicit `cll` case. Calibration tables do not
move, since their default is `DEFAULT_CALIBRATION_DISTORTIONS` (`_pricing.py` 26), which already
omits `cll`. Inside the library the default list reaches only `distortions_from_params`
(`spectral.py` 1793), which has no caller in `src`, `tests`, `docs`, `V:/dev/aggregate-api`, or
`V:/dev/PMIR`. The API and PMIR have no use of `available_distortions` either, so there is no ripple.

**2. Text beside the flag.** The comment at `spectral.py` 586 lists the conditions for
`strict_pricing`, and `cll` meets them, so add that the flag also marks the family as a production
choice. Rewrite the docstring of `available_distortions` (905) in NumPy style, stating the three
lists above. Remove "Called by `Portfolio.calibrate_distortions`" from the docstring of
`distortions_from_params`, which is no longer true.

**3. Docs.** Add a heading "Research distortion families" to `docs/3_reference/3_x_Distortion.rst`,
between "Distortion class" and "Spectral functions", and a `Notes` paragraph with the same facts to
the `Distortion` class docstring. The text: `cll`, `clin`, `lep`, and `ly` are built by the same
`Distortion` class and called the same way, and they are absent from the default
`available_distortions()` and from the default calibration set. The namespace in the original TODO
entry was rejected on 2026-10-30, because changing the spelling would break DecL and Python callers
for no benefit. As usual, the docs build is left to the author.

**Acceptance.** New tests in `tests/test_distortion_construction.py`:
`available_distortions() == DEFAULT_CALIBRATION_DISTORTIONS`; `cll` is in
`available_distortions(pricing=True, strict=False)` and in `available_distortions(pricing=False)`.
Tier 1 on `tests/test_distortion_*.py` and `tests/test_decl_words.py`, then tier 2, then tier 3 at
the bump. The prototype, a scratch monkeypatch of the flag, gave the five-family list and left `cll`
in both of the other lists. The CHANGELOG entry states the change of default, and
`[Distortion-Production-vs-Research]` is deleted from `dev/TODO.md`.

### `[Pedagogy-Repairs]`

**1. `similar_risks_example` (`pedagogy.py` 244).** Four edits, and the rest of the function stands:

- Line 257, `p_base.tvar(0.2, 'interp')` becomes `p_base.tvar(0.2)`. `kind` is ignored today, with a
  logged notice that it will be removed.
- Lines 265 to 273 become `bounds.plot_envelope(n_resamples=0)`. `plot_envelope` now takes only
  `n_resamples` and `reflect`, and draws two panels with the calibrated distortions on the second
  (`dev/done/plan-chart-ir.md`), so the axes array, the `distortions=` split, and the legend and title
  loops all go.
- Line 281, `p_new.plot(figsize=(6, 4))` becomes `p_new.plot()`.
- In the docstring, `similar_risks_graphs_sa` becomes `plot_similar_risks_graphs`.

**2. `plot_similar_risks_graphs`, the `p_reg < 1` branch (`pedagogy.py` 159 to 164).** The branch
calls `Bounds._tvar_x_func`, which no longer exists, computes `b = pnew.q(p_reg)` without using it,
and has a comment marking the bound as unfinished. The intended quantity is TVaR of the new portfolio
limited at the regulatory asset level $a = q_{p_{reg}}(X)$:

$$
\mathrm{TVaR}_p(X \wedge a) =
\begin{cases}
\mathrm{TVaR}_p(X) - \frac{(1 - F(a))\,(\mathrm{TVaR}_{F(a)}(X) - a)}{1 - p} & p < F(a) \\
a & p \ge F(a)
\end{cases}
$$

which `Bounds._tvar_x_a` already implements. The `else` branch is unchanged, so `p_reg=1`, the default
and the path `similar_risks_example` takes, gives the same numbers as today. The code becomes:

```python
    if p_reg < 1:
        # TVaR of min(X_new, a) at the regulatory asset level a = q(p_reg)
        btemp = Bounds(pnew, premium=prem, a=pnew.q(p_reg))
        tvar1 = {p: float(btemp._tvar_x_a(p)) for p in bounds.p_knots}
    else:
        tvar1 = {p: float(pnew.tvar(p)) for p in bounds.p_knots}
```

**3. `ClassicalPremium`, the Fischer premium (`pedagogy.py` 700).** `price(method='Fischer')` uses
`self.p`, which nothing sets. `calibrate` loops over every classical method, so it raises for any
input, not only for a Fischer request. `distribution`, `prices` with other methods, and `illustrate`
are unaffected, which is why `plot_ruin_surplus_paths` and the existing
`tests/test_unit_density.py` case pass. The formula is

$$
\mathsf{P}X + c\,\bigl(\mathsf{P}[(X - \mathsf{P}X)_+^{\,r}]\bigr)^{1/r}.
$$

Add a keyword-only `fischer_power=2` to `__init__`, stored as `self.fischer_power`, and use it at line
813. The default is 2 because power 1 reproduces the `Dutch` method exactly, so only a power above 1
adds a distinct principle, and 2 gives the upper semideviation. The name is new: `p` is already the
probability column of `df` and the keyword of `illustrate`, and `fischer_power` has no match in
`src/aggregate`. Document the keyword and the formula in the class docstring.

**Acceptance.** New file `tests/test_pedagogy_repairs.py`, with the two `similar_risks_example`
programs (`UNIF`, `BETA`) and the `ClassicalPremium` fixture appended to
`src/aggregate/agg/decl-testers.agg`. Figures on the `Agg` backend, closed after each test. Measured
with a scratch copy of the module:

| Check | Expected |
|:--|:--|
| `similar_risks_example()` | a frame of shape (11913, 7), `t1` from 0.635566 to 0.650011, three figures, about 0.5 s |
| `plot_similar_risks_graphs(..., p_reg=1)` | matches the repository today: `t1` from 0.635566 to 0.650011 |
| `plot_similar_risks_graphs(..., p_reg=0.9)` | `a` = 0.876953 and maximum `t1` = 0.628715, below the unbounded 0.650011 |
| `Bounds(p_new, premium=prem, a=0.876953125)._tvar_x_a(0.5)` | 0.73640109, equal to a direct sum over the pmf of $\min(x, a)$ |
| `calibrate('gross', 'Limit1', 8000)` on `0.1 claims 1000000 xs 0 sev lognorm 50000 cv 10 poisson` | nine methods, each `price` within 1e-3 relative of 8,000 |
| `price(0.5, ..., 'Fischer')` with `fischer_power=1` | equal to `price(0.5, ..., 'Dutch')` |

The tolerance on `calibrate` is relative because `Exponential` and `Esscher` land at 7,999.57 and
7,999.63: `newton`'s default tolerance is absolute and their parameters are of order 1e-6. Tier 1 on
the new file and `tests/test_unit_density.py`, then tier 2, then tier 3 at the bump. The bump rewrites
`[Plot-Argument-Warts]` in `dev/TODO.md` to what remains, `Distortion.plot_affine(ax=)`.

### `[Changelog-Split]`

**The move.** Move everything from the heading `## 1.0.0a401` to the end of `CHANGELOG.md` into
`dev/done/CHANGELOG-before-1.0.md`, unchanged, under a short header stating where it came from.
Find the start by that heading, not by line number: at 1.43.1 it is line 534 of 12,972, so 12,439
lines move, and each bump moves the line. Work on bytes so nothing is re-encoded (the file is LF).
`CHANGELOG.md` keeps its title, the "API stability" section, every entry from 1.0.0 up, and one
pointer at the foot under a rule:

```
---

Entries before 1.0.0 are in `dev/done/CHANGELOG-before-1.0.md`.
```

**What depends on the file.** No test reads it. `Get-ChangelogSection` in
`V:/worktrees/release-aggregate.ps1` reads only the section for the version being released, and the
`version-bump` skill inserts at the top of the entries. `README.md` 49 links to the file as a whole.
Comments citing old entries (`tests/test_approximate.py` 305, `tests/test_numerics3_distortion.py`
14) stay, as the pointer resolves them.

**Text to correct.** Three places refer to entries that move, or to a location that is wrong:

- `docs/2_aggregate_overview/features.rst` 15 to 16 and 3871: this chapter covers the `1.0.0a*`
  series, whose source of record becomes `dev/done/CHANGELOG-before-1.0.md`.
- `dev/task-features.md`: a one-line note near the top with the same pointer.
- `docs/3_reference/3_x_API_Stability.rst` 105: "`CHANGELOG.md` in the source distribution" is wrong
  today, since `MANIFEST.in` omits the file and the 1.11.0 sdist does not contain it. It becomes
  "`CHANGELOG.md` in the repository".

**Order and the commit.** This step runs last and has no bump, and under the project's rule the
author commits it. **The author must commit it before any later bump.** A bump stages all of
`CHANGELOG.md`, so while the split is uncommitted, the next bump would sweep 12,439 deleted lines into
an unrelated commit. The run summary states the same warning, and the step is not run while a parallel
session holds a bump in progress.

**Acceptance.** Kept lines (excluding the pointer) followed by moved lines (excluding the header)
equal `git show HEAD:CHANGELOG.md` byte for byte. `CHANGELOG.md` has exactly one heading for the
current version. `git status --short` shows only the files named in this step.

## Not in this plan

- `Distortion.plot_affine(ax=)` ignoring `ax`: the rest of `[Plot-Argument-Warts]`, not a `pedagogy.py` function.
- Moving `pedagogy.py` out of the package, or the research classes out of `spectral.py`: both withdrawn on 2026-10-30.
- `docs/5_technical_guides/5_x_pk.rst`, cited by `pedagogy.py` docstrings: that directory no longer
  exists in `docs`, and the page is now `2_x_pk.qmd` in the monograph. A docstring sweep is its own item.
- `natural_scale` printing a caught `IndexError` on `PZTest`: it still returns its table.
- A Using page: no statement, class or method is added, and `D:/Projects/Using/aggregate/distortions.qmd` needs no change (checked at drafting).

## Execution log

(filled at execution: version per step, the measured table, divergences)
