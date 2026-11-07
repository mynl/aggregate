# Plan: exact pointwise functions for an ELT severity with secondary uncertainty

`[ELT-Severity-Functions]`. Drafted 2026-11-06 against aggregate 2.0.0 (commit `1fe40eb8`) and eltq 2.14.0. **Status: ruled 2026-11-06; ready to execute once eltq 2.15.0 is on PyPI.** One pre-release bump (`2.1.0aN`). **Prerequisite:** the eltq half, `V:/dev/eltq/dev/plan-density-arrays.md` (`[Eltq-Density-Arrays]`), executed as eltq 2.15.0 and published by the author. **Supersedes** the draft `[ELT-OEP]` (`dev/plan-elt-oep.md`, deleted 2026-11-06 unexecuted), whose curve and `oep` branch are no longer needed.

## Goal

The event severity of an ELT6 table without `no2u` (`SeverityElt`) answers `sf`, `cdf`, `isf`, `ppf`, `pdf` and `stats` like any other `Severity`, exactly, and computes nothing until one of them is called. `oep`, `Aggregate.sev`, `severity_error_analysis` and the approximation chart then work on such an aggregate with no ELT-specific branch.

## Current behavior, 2.0.0

- `SeverityElt._build` (`_severity.py:4319`) sets `self.fz = _EltLaw(model.moments.max_support)`. `_EltLaw` (`_severity.py:4253`) supplies `support()` and makes `cdf`, `sf`, `pdf`, `ppf`, `isf` and `stats` raise `NotImplementedError`. The base `Severity` binds those methods into its layered wrappers at construction (`_severity.py:2882` to `2886`) without calling them.
- So `oep(agg, p)` raises from `agg.sev.isf`, as do `Aggregate.sev`, `severity_error_analysis` (`_aggregate.py:7694`, which evaluates `s.sf` and `s.pdf`) and the approximation chart (`charts/_emit_approximation.py:203`, `agg.sev.sf(edges)`). Mean-only tables (ELT3, ELT4, ELT6 under `no2u`) lower to `dhistogram`, whose functions are exact step functions, and already work.
- **No default path calls these methods**, checked 2026-11-06 against the `[ELT-Severity-Protocol]` audit (`dev/done/plan-elt-uncertainty.md`): `SeverityElt._compute_attachment_probs` overrides the base call of `fz.sf(0)`; discretization takes the `quantize` branch; `severity_feasibility` returns `None` for any non-scipy severity; the two in-build direct calls, `fz.ppf` in `_severity_low_estimate` (`_aggregate.py:7237`) and `fz.sf` in `_clipped_mass_estimate` (`_aggregate.py:7372`, reached from `_bucket_window.py:1545`), run only for a signed severity and a thick right tail, and an ELT severity is neither. No `except NotImplementedError` in `src/aggregate` is keyed to the refusal.
- eltq's `tail_moments_arrays` (already wrapped by `SeverityElt.tail_measures`) returns $R(t) = \sum_j\lambda_j\mathsf P(X_j > t)$, strict inequality, every row exact by its law (eltq spec section 4B), for edges $t \ge 0$. Measured 2026-11-06 at 30 edges: 8 ms, 12 ms and 90 ms on $10^4$, $10^5$ and $10^6$ synthetic rows, agreeing with a SciPy beta mixture to $7\times10^{-16}$ relative.

## The change

### [ELT-Pointwise-Law] `_EltLaw` answers

`_EltLaw(model)` replaces `_EltLaw(upper)`: it keeps a reference to the `EltModel` and reads `columns()`, $\Lambda$ (`moments.total_rate`), the support $B$ (`moments.max_support`) and the raw moments from it on each call. Nothing is computed or cached at construction, and every method calls `SeverityElt.check_grid_supported` first, so a shared table under the Burke rule raises the existing `[ELT-Burke-Total]` `NotImplementedError`. With $x$ an array (scalars broadcast) and the eltq calls made with `mean_is_loss_rate=False, variance_policy='bernoulli'`, as every other eltq call here:

- **`sf(x)`** $= R(x)/\Lambda$ from `tail_moments_arrays(edge=x)` for $0 \le x < B$; 1 for $x < 0$; 0 for $x \ge B$; NaN for NaN. Distinct values only are sent to eltq.
- **`cdf(x)`** $= 1 - $ `sf(x)`, right-continuous because $R$ is strict. The Notes say that the relative accuracy of a very small `cdf` is limited by the subtraction, as for any law reached through its survival function.
- **`isf(s)`**, vectorized bisection in $\log x$ on the exact `sf`: $s \le 0$ gives $B$; $s \ge \mathrm{sf}(0)$ (all the mass at zero is below it) gives 0; otherwise the bracket comes from one call on a geometric ladder of 64 edges from $10^{-30}B$ to $B$, and the bisection runs to $10^{-15}$ relative, about 45 calls of `tail_moments_arrays` on the unresolved points. It returns the upper end, the smallest bracketed $x$ with $\mathrm{sf}(x) \le s$, so where an atom's step jumps the target it returns the atom's loss and the read-back `sf` is below $s$, the same convention as a scipy severity with a limit.
- **`ppf(q)`** $=$ `isf(1 - q)`.
- **`pdf(x)`** $= \nu_c(x)/\Lambda$ from eltq's new `density_arrays` (the eltq half), the density of the continuous part with atoms excluded; 0 outside $(0, B)$.
- **`stats(moments='mv')`** from the model's raw severity moments $C_r/\Lambda$, $r \le 3$, in the scipy form; kurtosis NaN.
- Docstrings: `_EltLaw` and `SeverityElt` drop "no pointwise distribution function" and say what each method computes, from which eltq call, its cost order (rows times points), and that the grid still comes only from `quantize`.

### [ELT-OEP-Unchanged] `oep` and the other callers

- `oep` (`utilities.py:771`) needs no code change: it calls `agg.sev.isf` and `agg.sev.sf`. Its docstring gains one sentence on ELT tables. The draft's residual window below `min_loss` disappears, since the exact `sf` has no lower reach.
- `Aggregate.sev`, `severity_error_analysis` and the approximation chart need no code change; see ruling `[ELT-Grid-Scale-Callers]` for their cost.
- `ELTQ_MIN_VERSION` (`_loss_table.py:63`) becomes `(2, 15, 0)` and the `elt` extra (`pyproject.toml:118`) `eltq>=2.15`, with the comment above the constant naming `density_arrays` as the reason.

## Rulings

All by the author, 2026-11-06.

- **Approach.** Exact functions through eltq, with `density_arrays` added to eltq first.
- **[ELT-Grid-Scale-Callers]** Accepted: `severity_error_analysis` and the approximation chart evaluate the severity at every grid node, of the order of minutes at $2^{16}$ nodes and $10^6$ rows, and stay unrestricted because both are explicit diagnostics. The execution log records the measured times.
- **[ELT-Draft-OEP-Plan]** The superseded draft was deleted unexecuted; its measurements and its KOG-blog downstream note are carried here.
- **[Eltq-Release-For-Aggregate]** eltq goes straight to the release 2.15.0, which the author publishes (the eltq half).

## Tests

`tests/test_elt_severity_functions.py`, with the `eltq` fixture of `tests/test_elt_uncertainty.py` (`pytest.importorskip('eltq', minversion=...)`) and a `burke` fixture monkeypatching `_loss_table.SHARED_EVENT_RULE`:

1. **Exact `sf`.** On `EventLossTable.SecondaryUncertainty` (`library.agg:1197`, rows inline) and a seeded 1,000-row synthetic table, `sf` against the SciPy mixture $\sum_j\lambda_j\,\mathrm{beta.sf}(x/M_j;\alpha_j,\beta_j)/\Lambda$ (rows classified as eltq does), relative $10^{-13}$.
2. **Edges.** `sf` is 1 below 0 and 0 at and above $B$; `sf(0)` is $1 - \rho_0/\Lambda$; `cdf` at an atom row's loss includes the atom.
3. **Inverse.** `sf(isf(s)) = s` within $10^{-12}$ relative off atoms; at an atom, `isf` returns its loss within $10^{-14}$ relative and `sf(isf(s)) < s`; `isf(0) = B`, `isf(1) = 0`, `ppf(q) == isf(1 - q)`.
4. **Density.** `pdf` against the SciPy mixture density, relative $10^{-12}$ away from poles; 0 outside $(0, B)$.
5. **Stats.** Against the model moments.
6. **Burke.** Every method raises `NotImplementedError` matching `[ELT-Burke-Total]`.
7. **Shared roll-up.** The shared table `COMONOTONE` of `test_elt_uncertainty.py` matches the one-row table of its rolled row at several points, for every method.
8. **No overhead.** With every `_EltLaw` pointwise method monkeypatched to raise, `build` and `update` of an ELT6 aggregate, `qd`, `info` and the validation report all succeed.
9. **`oep`.** On the six-row table, `oep(a, 1/T)` for T in 5, 10, 25, 100, 1,000 and 5,000 against brentq on the SciPy mixture, $10^{-10}$ relative; the achieved `oep` equals `p` within $10^{-12}$; the result does not need `update`; `freq=2 * a.n` matches brentq at the doubled rate; a table with an atom row and an inadmissible row returns the atom's loss where its step jumps the target, with achieved `oep` below `p`; the updated object's grid reading is within `bs / 2` of `oep` at every T.
10. **Callers.** `Aggregate.sev` returns five working functions; `severity_error_analysis` and the approximation chart return finite results on the six-row table.
11. **Mean-only unchanged.** `no2u` on the same table takes the `dhistogram` path and `oep` returns 2.0.0's frame.

## Acceptance check

- The tests pass; tier 2 passes; tier 3 passes at the bump with `-W error::RuntimeWarning`.
- On `EventLossTable.SecondaryUncertainty`, `qd(oep(a, [0.1, 0.01, 0.001]))` prints losses 16.531, 190.864 and 506.394 (the draft's values from eltq's curve, which agreed with the exact mixture to $2.6\times10^{-8}$), achieved `oep` equal to `p`, and annual return periods 10, 100 and 1,000.
- The execution log records the times at 30 points for `sf`, `isf` and `pdf` on $10^4$, $10^5$ and $10^6$ synthetic rows, and of `severity_error_analysis` on $10^5$ rows.

## Files

- `src/aggregate/_severity.py`: `_EltLaw`, `SeverityElt._build` and docstrings.
- `src/aggregate/utilities.py`: the `oep` docstring.
- `src/aggregate/_loss_table.py`: `ELTQ_MIN_VERSION`; `pyproject.toml`: the `elt` extra.
- `tests/test_elt_severity_functions.py` (new).
- `docs/2_aggregate_overview/features.rst`: one sentence in the OEP section.
- Using pages: an "An event loss table" section in `D:/Projects/Using/aggregate/oep.qmd`, and a line in `elt.qmd` that the severity functions work (read `D:/Projects/Using/CLAUDE.md` first; render with `refresh.ps1 aggregate oep`; the deploy is the author's).
- `CHANGELOG.md`, `dev/TODO.md` (the `[DecL-Loss-Tables]` entry), this plan moved to `dev/done/`.

## Names vetted

No new public name in aggregate: the six methods already exist on `_EltLaw`, and change from raising to answering. The new eltq name `density_arrays` is vetted in the eltq half.

## Downstream

- The KOG-blog post `posts/notes/2026-11-03-Catastrophe-Modeling-with-aggregate` says `oep` raises on a table with secondary uncertainty and defines `oep_point` in its place. Afterwards its points table can call `oep(cat, [1/T for T in PERIODS])` and the sentence and helper go. That edit is the author's, in that repository.

## Execution log

Executed 2026-11-07 from aggregate 2.1.0a5 (commit `a0f90a94`) as 2.1.0a6, against eltq 2.15.0 from PyPI. The review found every line reference of the plan unchanged at 2.1.0a5 and no `except NotImplementedError` keyed to the refusal; a probe recording calls into `_EltLaw` across eleven ELT programs (build, `qd`, `info`) found none.

### Environment

- `uv lock --upgrade-package eltq` moved `uv.lock` from 2.14.0 to 2.15.0. The first `uv sync --all-extras --inexact` failed part way because a running JupyterLab kernel held `.venv/Lib/site-packages/eltq/_eltq.pyd`, leaving eltq half removed; the loaded file was renamed out of the package, the sync rerun, the empty `eltq-2.14.0.dist-info` remnant removed, and the renamed file deleted once the author stopped JupyterLab. `aggregate-ntt` survived the `--inexact` sync.

### Divergences

- **The Burke check.** The plan has each method call `SeverityElt.check_grid_supported`; `_EltLaw` holds the model, not the severity, so the body of that method moved to a module function `_check_elt_grid_supported(model, owner)` that both call, with the message unchanged. `support()` does not check, because a Burke table builds with `update=False` and its sizing reads the support.
- **[ELT-Pointwise-Law] details the plan left open.** `isf` brackets with one call on the 63 ladder edges below $B$ and takes $\mathrm{sf}(B) = 0$ without a call; a target between $\mathrm{sf}(10^{-30}B)$ and `sf(0)` returns $10^{-30}B$, as documented in its Notes. The bisection stops at $10^{-15}$ relative or when the geometric midpoint equals an end, and sends distinct midpoints only; 30 targets took 52 or 53 eltq calls. `stats` takes any of `m`, `v`, `s`, `k` and returns one value for one letter, as scipy does. `sf`, `pdf` and `isf` return a numpy scalar for a scalar input and keep an array's shape.
- **Test 8 records calls as well as raising.** A raising stub alone would pass if a caller swallowed the error in an `except Exception`, as `_clipped_mass_estimate` does around `fz.sf`; the stubs also append to a list that must stay empty. Run on the six-row table, the atom table and the comonotone shared table, each through `build`, `update`, `qd`, `info`, `validation_df` and the loss table checks.
- **Test 1's oracle** classifies rows as eltq's `reference._classify_rows` does (read, not imported): an atom when $M$, $\sigma$ or $\mu$ is zero or $\mu = M$, Bernoulli on $\{0, M\}$ when $(\sigma/\mu)^2 \ge (M - \mu)/\mu$, beta otherwise. The synthetic tables are generated by seed in the test (a `tmp_path` CSV for 1,000 rows), 5% atom and 5% Bernoulli rows unless the test asks for beta rows only.
- **Test 9's grid reading** is `a.q_sev(1 - s)` after `a.update()`, within `bs / 2` (times $1 + 10^{-9}$) of the `oep` loss at every return period.

### Measurements

At 30 points or targets, on seeded synthetic tables (5% atom and 5% Bernoulli rows, maxima from 20 to 5,000), after one warm call, all threads:

| rows | build (`update=False`) | `sf` | `pdf` | `isf` (eltq calls) |
|---|---|---|---|---|
| $10^4$ | 0.88 s | 0.009 s | 0.002 s | 0.15 s (53) |
| $10^5$ | 0.32 s | 0.014 s | 0.004 s | 0.28 s (52) |
| $10^6$ | 3.4 s | 0.11 s | 0.038 s | 2.4 s (53) |

`severity_error_analysis` on $10^5$ rows at the default grid ($2^{16}$ nodes at `bs` 8, top 524,288 above $B = 5{,}000$): 0.14 s, since only the nodes inside $(0, B)$ reach eltq. With the grid forced inside the support (`bs` 1/16, every node below $B$) it does not finish: see the first finding below. On the library's large-exposure tables (`LT.Large1`, `LT.Large2`, `LT.Remote`, `LT.RollupLarge`) at their default grids it runs in under 0.03 s.

### Gates

- Tier 1: the new file, 20 passed; the files touching ELT, `oep`, the approximation charts and the layer pdf, 2,558 passed.
- Tier 3 with the numerics gate, `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`: 11,709 passed, 3 skipped, 2 min 54 s.
- Acceptance: `oep` on `EventLossTable.SecondaryUncertainty` at 0.1, 0.01 and 0.001 gives 16.531372, 190.864261 and 506.393554, achieved `oep` equal to `p`, annual return periods 10, 100 and 1,000 (asserted in `test_oep`).

### Findings left for the author

- **`severity_error_analysis` on a grid that ends inside the support.** Its tail term integrates `s.sf` from the truncation point by `quad` in doublings and raises `ValueError: Questionable integral numeric convergence` when `quad`'s error estimate exceeds $10^{-8}$ absolute. On $10^5$ and $10^6$ beta-only rows with `bs` 1/16 and $2^{16}$ nodes it raised after 4.3 s and 37 s ($5.4\times10^{-5}$ and $1.7\times10^{-5}$), the time being `quad`'s scalar calls, each an eltq pass over every row. The default ELT grid passes the tail check and the integral is then negligible, so only a user-forced short grid meets it. An exact tail term exists, $\int_t^\infty S = D_1(t, t)/\Lambda$ from `tail_moments_arrays`, but it would be the ELT-specific branch this plan rules out.
- **The approximation charts on a degenerate loss window.** `chart_approximation` and `chart_approximation_tails` raise `ValueError: axis 'outcome' declares full_range with no suggested_range` when `loss_window` returns `None` because `q(0.999) = 0`. Not ELT-specific: `agg X.Analog 0.1 claims sev lognorm 100 cv 100 poisson` raises the same. On `LT.Large2` and `LT.Remote` it was hidden behind the old `NotImplementedError` from `agg.sev.sf`.
- **A quantile below the ladder floor.** `LT.Large2` (`[[1 .1 100 1e9 10000 0]]`, $\alpha \approx 10^{-4}$) has 99.4% of its severity below $10^{-21}$, so `isf(0.01)` returns $10^{-30}B = 10^{-21}$ with `sf` 0.0061, and `oep(a, 0.01)` reports that loss with an annual return period of 1,630. The true quantile underflows a double (about $10^{-10^4}$), so the loss is zero in money either way and the achieved `oep` column states the shortfall, the convention for an atom.
