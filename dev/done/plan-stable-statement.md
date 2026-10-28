---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: the `stable` statement and the `Stable` parameter class

`[Stable-Statement]`. Drafted 2026-10-28 against aggregate 1.18.2. **Status:
executed in 1.23.0 on 2026-10-28, with the PMIR commit; see the execution
log.** One minor bump in aggregate, plus one unversioned commit
in PMIR. Depends on `[Tweedie-Statement]` (`dev/done/plan-tweedie-statement.md`, 1.22.0), which
adds the `conversion` statement kind this plan reuses. The aLL display is
`[Status-Bar-Text-Answers]` in `V:/dev/aggregate-api/dev/plan-status-bar-text-answers.md`;
computing a stable distribution on a grid is
`[Characteristic-Function-Engine]` (`dev/plan-characteristic-function-engine.md`,
parked).

## Goal

Three deliverables.

1. **`aggregate.stable.Stable`**, a class for $\alpha$-stable parameters in Nolan's
   S0 and S1 parameterizations [@Nolan2020], with the closure algebra (sums of
   independent stable variables with a common $\alpha$, scaling, negation,
   shifting), the closed-form characteristic function, the mean, the tail
   constants, and `scipy`-backed `pdf`, `cdf`, `ppf` and `rvs` whose tail limits
   are documented. The class is moved from PMIR's
   `D:/Projects/PMIR/Python/pmir_package/src/pmir_legacy/stable.py`, without the
   claim count development methods, which stay in PMIR.
2. **A bare `stable` statement**, on the `conversion` kind:

   ```
   stable 1.5 1 2 0                          # alpha beta scale loc, S1
   stable 1.5 1 2 -2 s0                      # the same law in S0
   stable 1000 claims sev 100 * lomax 1.5    # stable limit of a Pareto book
   ```

   each returning a `Stable` whose string gives both parameterizations and the
   facts a reader most often looks up:

   ```
   stable 1.5 1 2 0 = stable 1.5 1 2 -2 s0, mean 0, Pr(X>mean) 0.333333, Pr(X>x) ~ 1.12838 x^-1.5
   ```

3. **PMIR**: the legacy module gains a note that the class has moved and a
   `FutureWarning` on import, and its broken import is fixed so the warning can
   be seen. Nothing in PMIR is deleted.

## Rulings

Author, 2026-10-28:

1. **The parameter algebra moves into aggregate**, as a class reached only by
   submodule import, `from aggregate.stable import Stable`, as `Tweedie` is.
2. **PMIR keeps its copy for backwards compatibility**, with a note that it has
   moved and a warning when the module is imported.
3. **Motivation, in order:** teaching (demystifying stable laws and their
   parameterizations, and the generalized central limit theorem as something a
   reader can run in DecL), then the PMIR emergence application.
4. **Closed form only in this plan.** Accurate distribution values in the far
   tail, and a stable distribution built on a grid, are
   `[Characteristic-Function-Engine]`.

Proposed in drafting, to confirm at review:

5. **S1 is the default parameterization**, with `s0` or `s1` accepted as a
   trailing word. S1 is scipy's default and the one in which sums and scaling are
   clean for every $\alpha$; S0 is continuous in $\alpha$ and preferred by Nolan
   for numerical work and fitting. The string always shows both.
6. **The limit-theorem form is restricted to $1<\alpha<2$**, where the
   approximating stable has exactly the mean and the right tail of the compound.
   $\alpha \le 1$ is deferred.
7. **`stable` becomes a reserved word**, so it can no longer be a name. On
   2026-10-28, `rg -w stable` over `src/aggregate/agg` and `tests` found it only
   inside a `note{}` (`decl-testers.agg:427`) and in unrelated test docstrings.
8. **The engine plan's object form takes its own keyword** (proposed there:
   `chf NAME stable ...`), so a bare `stable` statement stays a calculator and
   `stable NAME ...` is left unused.

## Current behavior, 1.18.2

- **aggregate has no stable parameter class.** Stable laws appear in three
  places. `Tweedie` covers the totally skewed members ($p<0$, $p>2$, $p=\infty$),
  and `Tweedie.fz` (`src/aggregate/tweedie.py:572`) builds them with
  `ss.levy_stable` and $\beta=\pm1$. `make_levy_chf(alpha, beta)`
  (`src/aggregate/ft.py:761`) returns the standard S1 characteristic function
  with unit scale; at $\alpha=1$ it evaluates $\log 0$ at $t=0$ inside
  `np.where` and emits a `RuntimeWarning`. `sev levy` (scipy `levy`, the
  $\alpha=1/2$, $\beta=1$ law) is usable as a severity (`decl-testers.agg:427`).
- **PMIR's `stable.py`** (March 2025, 491 lines) defines `Stable(a, b, c=1, d=0,
  param='S1')` wrapping `ss.levy_stable`, with `S0_to_S1`, `__add__`, `__sub__`,
  `__neg__`, `__rmul__` and `scalar_mult`, `pdf`, `cdf`, `ppf`, `isf`, `plot`,
  the Doob model methods `zeta_from_pattern`, `ay_counts` and `simulate_ay`, and
  the module functions `unwrap`, `defuzz` and `bivariate_from_matrix`. The algebra
  is correct in S1: locations add for every $\alpha$, and scaling has the extra
  term at $\alpha=1$. The commented-out location adjustments in `__add__` are the
  S0 rule and are rightly unused.
- **The PMIR module does not import.** It imports `estimate_agg_percentile` from
  `aggregate`, which no longer exports it; the function is now the private
  `_estimate_agg_percentile` in `aggregate._bucket_window`, with the same
  arguments, and still accepts a pair of probabilities. PMIR's
  `pmir-legacy-state-of-play.md` lists the module as failing to import and
  imported by nothing. A warning placed in the module would never be seen until
  the import is fixed.
- **scipy's stable distribution is unreliable in the far tail** (scipy 1.17.1,
  measured 2026-10-28, $\alpha=1.5$, $\beta=1$, S1, unit scale). `sf(659)` is
  $2.358\times10^{-5}$ and agrees with the tail asymptotic, yet `ppf(1 - 1e-6)`
  returns 659 rather than about 5430, and `sf(5430)` returns 0.0 against an
  asymptotic $9.97\times10^{-7}$. `pdf` at 2,000 points takes 0.67 s. The
  parameterization is a class attribute, `ss.levy_stable.parameterization`
  (default `'S1'`), shared by every caller in the process.

## Mathematics

**Parameterizations.** Write $X \sim S(\alpha, \beta, \gamma, \delta_1; 1)$ for S1,
$0<\alpha\le2$, $-1\le\beta\le1$, $\gamma>0$. For $\alpha \ne 1$,

$$\log \mathsf{P} e^{itX} = -\gamma^\alpha|t|^\alpha\bigl(1 - i\beta\,\mathrm{sign}(t)\tan(\pi\alpha/2)\bigr) + i\delta_1 t,$$

and for $\alpha=1$ the bracket is $1 + i\beta(2/\pi)\,\mathrm{sign}(t)\log|t|$ with
$|t|$ in place of $|t|^\alpha$. The S0 location is
$\delta_0 = \delta_1 + \beta\gamma\tan(\pi\alpha/2)$ for $\alpha\ne1$ and
$\delta_0 = \delta_1 + (2/\pi)\beta\gamma\log\gamma$ for $\alpha=1$; the other
three parameters agree [@Nolan2020, Def. 1.3 to 1.7]. At $\alpha=2$ the law is
normal with variance $2\gamma^2$ and $\beta$ has no effect.

**Closure, in S1** [@Nolan2020, Prop. 1.3 and 1.4]. For independent
$X_j \sim S(\alpha, \beta_j, \gamma_j, \delta_j; 1)$, the sum has
$\gamma^\alpha = \gamma_1^\alpha+\gamma_2^\alpha$,
$\beta = (\beta_1\gamma_1^\alpha+\beta_2\gamma_2^\alpha)/\gamma^\alpha$ and
$\delta = \delta_1+\delta_2$. For $a\ne0$, $aX \sim S(\alpha, \mathrm{sign}(a)\beta,
|a|\gamma, a\delta_1; 1)$ when $\alpha\ne1$, with location
$a\delta_1 - (2/\pi)\beta\gamma a\log|a|$ when $\alpha=1$.

**Facts shown in the string.** The mean is $\delta_1$ for $\alpha>1$ and does not
exist for $\alpha\le1$. For $\alpha\ne1$, $X-\delta_1$ is strictly stable and

$$\Pr(X>\delta_1) = \tfrac12 + \frac{\arctan\bigl(\beta\tan(\pi\alpha/2)\bigr)}{\pi\alpha},$$

which is $1-1/\alpha$ when $\beta=1$ and $1<\alpha<2$: the mean of a spectrally
positive stable with $\alpha=1.5$ is its 2/3 quantile. For $\alpha<2$ the tails
are $\Pr(X>x)\sim(1+\beta)C_\alpha\gamma^\alpha x^{-\alpha}$ and
$\Pr(X<-x)\sim(1-\beta)C_\alpha\gamma^\alpha x^{-\alpha}$, with
$C_\alpha = \Gamma(\alpha)\sin(\pi\alpha/2)/\pi$ [@Nolan2020, Thm. 1.2].

**The generalized central limit theorem for a Pareto book.** Let the count be
Poisson with mean $\lambda$ and the severity have
$\Pr(X>x) = \theta^\alpha x^{-\alpha}(1+o(1))$ with $1<\alpha<2$. The compound has
Lévy measure $\lambda F(dx)$, whose tail is $\lambda\theta^\alpha x^{-\alpha}$, and
for large $\lambda$ the aggregate is approximately

$$S \approx S\bigl(\alpha,\ 1,\ \gamma,\ \lambda\mathsf{P}X;\ 1\bigr), \qquad
\gamma^\alpha = \lambda\theta^\alpha\,\Gamma(1-\alpha)\cos(\pi\alpha/2)$$

[@Feller1971, ch. XVII]. Both factors $\Gamma(1-\alpha)$ and $\cos(\pi\alpha/2)$
are negative on $(1,2)$, so the product is positive. By the reflection formula
$\Gamma(\alpha)\Gamma(1-\alpha) = \pi/\sin(\pi\alpha)$, the approximating stable has
right tail constant $2C_\alpha\gamma^\alpha = \lambda\theta^\alpha$, so it matches
the compound's mean and right tail exactly. Its left tail decays faster than any
power. For a `pareto` or `lomax` severity with shape $\alpha$, scale $\theta$ and
any location, the tail is $\theta^\alpha x^{-\alpha}(1+o(1))$; the location changes
only the mean.

Worked example: `stable 1000 claims sev 100 * lomax 1.5` has $\lambda=1000$,
$\theta=100$ and $\mathsf{P}X = \theta/(\alpha-1) = 200$, so
$\gamma = 18452.701486$, $\delta_1 = 200000$, $\delta_0 = 181547.30$, and the
right tail constant is $10^6 = \lambda\theta^\alpha$ (computed 2026-10-28).
`sev 100 * pareto 1.5 - 100` is the same law and gives the same answer.

## Design

### [Stable-Class] `aggregate.stable.Stable`

A new leaf module, `src/aggregate/stable.py`, importing only numpy and scipy, so
the parser can import it lazily. It is not re-exported at the top level.

```python
class Stable:
    def __init__(self, alpha, beta, scale=1.0, loc=0.0, parameterization='S1'): ...
```

- Attributes `alpha`, `beta`, `scale`, `loc`, with `loc` always the S1 location;
  `loc0` (property) is the S0 location. A `parameterization='S0'` input converts
  `loc` on construction. Scalars only.
- Static methods `s0_to_s1(alpha, beta, scale, loc0)` and
  `s1_to_s0(alpha, beta, scale, loc1)`, each returning the other location.
- `chf(t)`, the closed form above, guarded with `np.errstate` so $t=0$ at
  $\alpha=1$ emits no warning.
- Properties `mean` (for $\alpha\le1$: `np.inf` when $\beta=1$, `-np.inf` when
  $\beta=-1$, otherwise `np.nan`), `prob_above_loc` (the formula above; `np.nan`
  at $\alpha=1$ unless $\beta=0$, where it is $1/2$), `tail_constant` (right) and
  `left_tail_constant`.
- `fz`, built lazily as `ss.levy_stable` in whichever parameterization the scipy
  class attribute currently names, never assigning that attribute; `ss.norm` at
  $\alpha=2$. `pdf`, `cdf`, `sf`, `ppf`, `isf`, `rvs` delegate to it. The
  docstring `Notes` record the far-tail measurements above, and that the
  characteristic-function engine is the accurate route.
- The algebra from PMIR: `__add__` and `__radd__` (another `Stable` with the same
  $\alpha$, or a number, which shifts; `__radd__` makes `sum()` work), `__sub__`,
  `__neg__`, `__mul__` and `__rmul__` by a real number.
- Class method `from_compound_pareto(frequency, alpha, theta, sev_mean)`, the
  limit theorem above, refusing $\alpha\notin(1,2)$.
- `plot(kind='density' or 'quantile')`, from PMIR's `plot`, on scipy.
- `to_decl()`, returning `stable <alpha> <beta> <scale> <loc>`; `__repr__`,
  `Stable(alpha=1.5, beta=1, scale=2, loc=0)`; `__str__`, the one-line answer,
  ASCII only, numbers formatted `.6g`, parts joined as in the Goal. When the mean
  does not exist the string says `mean infinite` or `mean undefined` and labels
  the probability `Pr(X>loc)` instead of `Pr(X>mean)`, and at
  $\alpha=2$ it shows `normal, variance` in place of the tail.
- The class docstring includes a table of the three notations: Nolan
  $(\alpha, \beta, \gamma, \delta)$, PMIR `(a, b, c, d)`, and scipy
  `(alpha, beta, scale, loc)`.

### [Stable-Statement-Grammar] the grammar

In `src/aggregate/decl.lark`, a new terminal
`STABLE.2: /stable(?![a-zA-Z0-9._:~\-])/`, the word `stable` added to the `ID`
exclusion alternation (line 1047, in alphabetical position), and two
`conversion_out` alternatives:

```
              | STABLE numberl            -> conversion_stable
              | STABLE numberl ID         -> conversion_stable_param
              | STABLE expr CLAIMS SEV sev -> conversion_stable_compound
```

`decl_words` reads the reserved list from the `ID` alternation, so the Pygments
lexer and every client that colors DecL pick the word up with no further edit.
`src/aggregate/parser_errors.py` gains `"STABLE": "'stable'"`. `CLAIMS` is a
keyword, so `STABLE expr CLAIMS` and `STABLE numberl ID` cannot collide.

### [Stable-Statement-Transformer] the transformer

- `conversion_stable` and `conversion_stable_param` accept two, three or four
  numbers (`alpha beta [scale [loc]]`) and an optional `s0` or `s1`
  (case-insensitive); anything else raises
  `ValueError("stable: the statement is 'stable alpha beta [scale [loc]] [s0|s1]', ...")`.
  Range errors raised by `Stable.__init__` include the parameter name.
- `conversion_stable_compound` resolves the severity dict exactly as an `agg`
  does, with `Severity(**sev)`, and takes from the frozen scipy object its name
  (`pareto` or `lomax`), shape $\alpha\in(1,2)$, scale $\theta$ and mean `fz.mean()`. Refused
  with a message that includes what was found: any other name, `sev_reflect`,
  `sev_wt`, splices, picks, and `sev_conditional`. Verify at execution that
  `Severity(**sev).fz` is populated for both names in each spelling (shape and
  scale, and mean and shape).
- Each returns `("conversion", "stable", Stable(...))`.

## Steps

### [Stable-Statement], one minor bump

1. `[Stable-Class]`, `[Stable-Statement-Grammar]` and
   `[Stable-Statement-Transformer]` as above.
2. Tests (below).
3. Grammar reference: regenerate `docs/4_agg_language_reference/ref_include.rst`
   with `parser.grammar(add_to_doc=True)`.
4. Documentation: extend the "Parameter Conversions" section of
   `docs/4_dec_Language_Reference.rst` (added by `[Tweedie-Statement]`; retitle it
   "Parameter Conversions (tweedie, stable)") with the three forms, both worked
   examples, and the scipy far-tail caveat. Add `aggregate.stable` to the API
   reference beside `aggregate.tweedie` (locate with `rg -l "aggregate.tweedie" docs`).
5. aLL help: in `src/aggregate/data/decl-help/card.html`, box 10, the line
   `stable 1000 claims sev 100 * lomax 1.5` the stable limit of a Pareto book.
6. Release: `__version__`, `CHANGELOG.md` (the reserved word is a breaking change
   for any program that used `stable` as a name; say so), `dev/TODO.md`,
   `dev/FEATURES.csv` via `dev/regen_features.py`; move this plan to `dev/done/`.

### [PMIR-Stable-Moved], one unversioned commit in PMIR

After the aggregate bump, in `D:/Projects/PMIR` (legacy package rule: fixes
only):

1. `Python/pmir_package/src/pmir_legacy/stable.py`: a paragraph in the module
   docstring stating that the `Stable` parameter class moved to aggregate as
   `aggregate.stable.Stable` at aggregate `<version>`, that this copy is kept for
   backwards compatibility and is not maintained, and that the claim count
   development methods (`zeta_from_pattern`, `ay_counts`, `simulate_ay`) exist
   only here.
2. The import fix: remove `estimate_agg_percentile` from the top-level
   `from aggregate import (...)` and import it inside `bivariate_from_matrix`
   as `from aggregate._bucket_window import _estimate_agg_percentile as estimate_agg_percentile`.
   Check that the module then imports; the functions themselves are not
   re-tested.
3. After the imports, `warnings.warn("pmir_legacy.stable is superseded by
   aggregate.stable.Stable; this copy is kept for backwards compatibility only.",
   FutureWarning, stacklevel=2)`. `FutureWarning` rather than
   `DeprecationWarning` because Python shows it by default.
4. `pmir-legacy-state-of-play.md`: the `stable` row and the "Stable
   distributions" paragraph record that the module imports with a warning and
   that the class is now in aggregate.
5. Commit, subject `mark pmir_legacy.stable as moved to aggregate.stable`, and
   push.

## Files

| Area | Files |
|---|---|
| Class | `src/aggregate/stable.py` (new) |
| Grammar | `src/aggregate/decl.lark` |
| Parser | `src/aggregate/parser.py` (three conversion methods), `src/aggregate/parser_errors.py` |
| Help | `src/aggregate/data/decl-help/card.html` |
| Tests | `tests/test_stable.py` (new) |
| Documentation | `docs/4_dec_Language_Reference.rst`, `docs/4_agg_language_reference/ref_include.rst`, the API reference page that lists `aggregate.tweedie` |
| Release | `src/aggregate/__init__.py`, `CHANGELOG.md`, `dev/TODO.md`, `dev/FEATURES.csv` |
| PMIR | `Python/pmir_package/src/pmir_legacy/stable.py`, `pmir-legacy-state-of-play.md` |

## Names checked

On 2026-10-28, `rg` over `src/aggregate` found no class `Stable`, no module
`stable`, no attribute `stable` on the package, and no use of
`from_compound_pareto`, `prob_above_loc`, `tail_constant`, `left_tail_constant`,
`s0_to_s1`, `s1_to_s0`, `conversion_stable`, or the terminal `STABLE`. The
aggregate-api maps built objects to kinds by class name (`_KIND_OF_CLASS`), so
the name `Stable` must stay unique among the library's built classes. Recheck at
execution.

## Acceptance and tests

In `tests/test_stable.py`, all oracles exact or closed form except where scipy is
named:

- **Closure by characteristic function.** For $\alpha\in\{0.7, 1, 1.5, 1.9\}$,
  mixed $\beta$, scales and locations, on a grid of $t$ including 0:
  $\varphi_{X_1+X_2} = \varphi_{X_1}\varphi_{X_2}$, $\varphi_{aX}(t) = \varphi_X(at)$
  for $a\in\{-2.5, 0.3, 4\}$, and $\varphi_{-X}(t) = \varphi_X(-t)$, to
  `rtol=1e-12`. These pin the $\alpha=1$ logarithm terms.
- **Parameterizations.** S0 to S1 to S0 is the identity; `Stable(..., 'S0')`
  equals `Stable(..., 'S1')` with the converted location; `ss.levy_stable.parameterization`
  is unchanged after `fz` is built.
- **scipy in the body.** `cdf` at five points between the 1% and 99% quantiles
  agrees with an independent `ss.levy_stable` call for $\alpha\in\{0.8, 1.5\}$.
- **Tail constant.** For $\alpha=1.5$, $\beta=1$, `tail_constant * 100**-1.5`
  agrees with `fz.sf(100)` to `rtol=1e-4` (measured agreement $4\times10^{-6}$).
- **Limit theorem.** `stable 1000 claims sev 100 * lomax 1.5` gives scale
  18452.701486 (`rtol=1e-9`) and location 200000, equals the
  `sev 100 * pareto 1.5 - 100` spelling, and its right tail constant equals
  $\lambda\theta^\alpha = 10^6$ to `rtol=1e-12`.
- **Statement.** `build('stable 1.5 1 2 0')` is a `Stable` with `loc0 == -2` to
  `rtol=1e-12`; `stable 1.5 1 2 -2 s0` equals it; the two- and three-number forms
  default scale 1 and location 0; `str` is ASCII and each spelling in it builds
  back to parameters within `rtol=1e-5`; `prob_above_loc` is $1/3$ for
  $\alpha=1.5$, $\beta=1$.
- **Refusals.** $\alpha\notin(0,2]$, $|\beta|>1$, scale $\le0$, one or five
  numbers, an unknown parameterization word, and in the compound form
  $\alpha\notin(1,2)$, a lognormal severity, a mixture and a reflected severity,
  each raise `ValueError` that states the problem.
- **Reserved word.** `agg stable 1 claims dsev [1] fixed` is a parse error.
- **Nothing stored.** The recipe base is unchanged by a `stable` statement.

## Release gates

Edit loop: `tests/test_stable.py`, `tests/test_tweedie.py`,
`tests/test_decl_parser.py`. At the bump: the full fast suite, then
`uv run pytest -m 'slow or not slow' -W error::RuntimeWarning` (new numerics),
with `UV_PROJECT_ENVIRONMENT=.venv`. Docs are not built.

## Deferred

- `[Characteristic-Function-Engine]`: accurate densities and far-tail quantiles,
  and a stable distribution on a grid, in
  `dev/plan-characteristic-function-engine.md`.
- The limit theorem for $\alpha\le1$, other regularly varying severities, and a
  negative binomial count (whose mixing changes the limit).
- PMIR's emergence application on aggregate's `Stable`: `zeta_from_pattern` uses
  `ppf` at the claim frequency, which is in the body, so scipy suffices there; the
  simulated triangles want the engine's accurate `cdf`.
- Teaching material: the education modules
  `D:/projects/aggregate/education/modules/03-the-single-big-jump` and
  `04-normal-body-subexponential-tail` are natural homes for the limit-theorem
  statement beside `agg X 1000 claims sev 100 * lomax 1.5 poisson`.
- Replacing `ft.make_levy_chf` with `Stable.chf`, and checking `Tweedie.fz` for
  $p>2$ and $p<0$ against `Stable`: engine plan.

## Execution log

**1.23.0, 2026-10-28.** Reviewed and executed in the same session as
`[Tweedie-Statement]` (1.22.0), against that version. The proposals marked "to
confirm at review" (5, S1 the default; 6, the limit theorem for
`1 < alpha < 2` only; 7, `stable` reserved; 8, the engine's own keyword) were
confirmed as written. A recheck of the names found no collision; `rg -w stable`
over the corpora still finds the word only in the `GH.PoissonLevy` note.
Measured before writing `fz`: a frozen `ss.levy_stable` copies the class
attribute's parameterization when it is created (scipy 1.17.1), and
`Severity(**sev)` builds both Pareto spellings of the worked example with mean
200 and no warning.

Divergences:

1. **`agg.sublime-syntax` gains `stable`** in its keyword list and its `ID`
   exclusion. `tests/test_grammar_sync.py` checks both against the grammar,
   and the plan did not list the file.
2. **The compound form refuses non-default values of `sev_wt`, `sev_lb`,
   `sev_ub`, `sev_reflect` and `sev_conditional`, not their presence**, as in
   `[Tweedie-Statement]` divergence 2, because the `sev` rule always sets the
   first three. A shift (`sev_loc`) is allowed. Vector parameters and a
   nonpositive count are refused too. The tail index is checked in the
   transformer, before `fz.mean()` is read, since the mean is infinite for
   `alpha <= 1`; `from_compound_pareto` keeps its own check for Python callers.
   `Severity` is built from the `sev_*` keys only, so the trailer of a stored
   `sev.NAME` is dropped. The shape is `fz.args[0]`, or the scipy shape
   keyword (`c` for lomax, `b` for pareto) when it is passed by name.
3. **The string shows each power tail whose constant is positive**:
   `Pr(X>x) ~ ...` when `beta > -1` and `Pr(X<-x) ~ ...` when `beta < 1`, so
   `beta = -1` shows the left tail rather than `Pr(X>x) ~ 0`. The
   `Pr(X>...)` term is left out when it is `nan` (`alpha = 1`, `beta != 0`).
   A mean of `-inf` reads `mean negatively infinite`.
4. **Locations within `1e-12 * scale` of zero are shown as 0 in the string.**
   `tan(3 pi / 4)` is `-1.0000000000000002`, so `stable 1.5 1 2 -2 s0` stores
   an S1 location of `4.4e-16`, which `.6g` would show. `repr` and `to_decl`
   stay exact.
5. **At `alpha = 2`, `beta` is set to 0 and `fz` is the normal with standard
   deviation `sqrt(2) * scale`**, variance `2 gamma^2` as the plan states.
   PMIR's copy builds `ss.norm(scale=c)`, variance `c^2`, which is the wrong
   normal; left for the author, since the PMIR step is a note, an import fix
   and a warning.
6. **Small additions to the algebra:** `__rsub__` (so `5 - X` works), scaling
   by 0 is refused (a point mass is not a stable law), and operations with a
   non-number return `NotImplemented`. `plot` takes `kind='density'` or
   `'quantile'` only (PMIR also accepted `'lee'`, `'dist'` and
   `'distribution'`) and returns the Axes.
7. **`dev/TODO.md` is unchanged.** It deletes finished items rather than
   ticking them, so there was no entry to close, and the deferred items are in
   this plan's Deferred section and the parked engine plan.
8. **The package docstring** (`src/aggregate/__init__.py`) names `Stable`
   among the submodule-only classes.

Tests: `tests/test_stable.py` 65 passed, also under `-W error::RuntimeWarning`.
Edit loop `tests/test_stable.py tests/test_tweedie.py tests/test_decl_parser.py
tests/test_decl_unparser.py tests/test_grammar_ambiguity.py
tests/test_grammar_sync.py tests/test_underwriter.py
tests/test_parser_errors_integration.py`: 1,945 passed;
`tests/test_decl_words.py` 260 passed, with `stable` in the `reserved` group.
The snapshot test passed without a re-capture. `dev/FEATURES.csv` regenerated
with no change. The grammar reference regenerated with additions and the `ID`
line only. Gate, `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not
slow' -W error::RuntimeWarning`: 7,014 passed in 154 s.

**PMIR.** 9. **Two commits, not one:** `Python/pmir_package` is its own
repository (`mynl/pmir_package`), ignored by the PMIR repository, so
`stable.py` is committed there and `pmir-legacy-state-of-play.md` in PMIR,
each with the subject `mark pmir_legacy.stable as moved to aggregate.stable`.
10. **`pmir_package` bumps to 3.0.1** (`pyproject.toml` and the history in
`pmir_legacy/__init__.py`), since that repository versions its fixes (its 2.1.0
was an import fix); the plan's "unversioned" was written for the PMIR
repository. The module was checked to import, with the `FutureWarning`, under
this repository's `.venv` with `joblib` added from a scratch directory, and
`_estimate_agg_percentile` was checked to accept a pair of probabilities. The
functions themselves were not re-tested, as the plan says.

## References

::: {#refs}
:::
