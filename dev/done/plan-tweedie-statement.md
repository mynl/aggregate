---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: the `tweedie` statement, a calculator for Tweedie parameters

`[Tweedie-Statement]`. Drafted 2026-10-28 against aggregate 1.18.2. **Status:
executed in 1.22.0 on 2026-10-28; see the execution log.** One minor bump.
First of four related plans:

| Plan | Repository | Relation |
|---|---|---|
| `dev/plan-tweedie-statement.md` (this one) | aggregate | adds the `conversion` statement kind and the `tweedie` statement |
| `dev/plan-stable-statement.md` | aggregate | adds the `stable` statement on the same kind, and the `Stable` class |
| `dev/plan-status-bar-text-answers.md` | aggregate-api (`V:/dev/aggregate-api`) | shows a conversion in the aLL status bar, Excel and Sheets |
| `dev/plan-characteristic-function-engine.md` | aggregate | parked: distributions computed from a characteristic function |

Recommended order: this plan, then the api plan straight after it (until the api
plan lands, aLL refuses the new statement as an unsupported kind), then the
stable plan.

## Goal

A program consisting of a bare `tweedie` statement computes the equivalent
spellings of a Tweedie distribution with $1<p<2$, without building anything. It
is the clause from inside an `agg`, standing alone, and it is an answer rather
than a declaration, like a bare expression `(2+2)`. Three input forms are
accepted, and every form returns the same `TweedieConversion` object:

```
tweedie 1.5 100 2                        # p, mean, dispersion (a GLM's p and phi)
tweedie 1.5 100 cv 0.447214              # p, mean, coefficient of variation
tweedie 10 claims sev gamma 10 cv 1      # Poisson count and gamma severity
```

each giving, as its string,

```
tweedie 1.5 100 2 = tweedie 1.5 100 cv 0.447214 = tweedie 10 claims sev gamma 10 cv 1, Pr(X=0) 4.54e-05
```

Each spelling is legal input, so any part of the answer can be pasted back. The
`agg` body clause gains the coefficient of variation form as well, so
`agg X tweedie 1.5 100 cv 0.447214` builds and the unparser renders it as declared.

## Rulings

Author, 2026-10-28:

1. **Three input forms.** The coefficient of variation form is included: by hand,
   actuaries think in p, mean and CV, and the dispersion is mainly a GLM output.
2. **The `agg` body clause accepts the CV form too**, so the bare statement stays
   the same clause as the one inside an `agg`. The declared CV is recorded in a
   separate private key, `_tweedie_cv`, beside `_tweedie`; the public
   `TweedieParameters` is unchanged.
3. **A bare statement is an answer.** Nothing is stored in the recipe base, it has
   no name and no trailer (`note{}` and the rest), and `build` returns the object
   directly, exactly as for an expression.
4. **The statement kind is generic**, `'conversion'`, so the stable statement and
   any later calculator (lognormal or gamma mean and CV, say) reuse it without new
   plumbing. None of the later calculators is in scope.

Proposed in drafting, to confirm at review:

5. **The compound form uses the `agg` body's own `claims` and `sev` clauses**,
   `tweedie 10 claims sev gamma 10 cv 1`, rather than the shorter
   `tweedie 10 claims gamma 10 cv 1` discussed on 2026-10-28. With `sev` the
   statement is the `agg` body with `agg NAME` and `poisson` removed, every gamma
   spelling an `agg` accepts works (`sev gamma 10 cv 1`, `sev 10 * gamma 1`), and
   the stable statement needs `sev` anyway, because there the meaning of `pareto`
   must be DecL's. The cost is one word.
6. **Numbers in the string are shown to six significant figures.** The object
   holds full-precision floats; pasting a displayed spelling back reproduces the
   parameters to about six figures.

## Current behavior, 1.18.2

- `tweedie <p> <mean> <dispersion>` exists only inside an `agg` body:
  `agg_body` alternative `TWEEDIE expr expr expr` (`src/aggregate/decl.lark:85`),
  transformed by `agg_body_tweedie` (`src/aggregate/parser.py:756`). The
  transformer rejects $p \notin (1,2)$, expands through `tweedie_convert`
  (`src/aggregate/tweedie.py:48`) to a Poisson count and gamma severity, and
  records `_tweedie = TweedieParameters(p, mean, dispersion)` as provenance.
  `_render_tweedie` (`src/aggregate/decl_writer.py:937`) renders the clause
  from that key. `Aggregate.as_tweedie()` (`src/aggregate/_aggregate.py:6441`)
  recognizes any bare compound Poisson gamma. See `dev/done/plan-tweedie.md`
  (`[Tweedie-Round-Trip]`).
- A program that starts with `tweedie` is a parse error.
- `Tweedie.from_p_mean_cv` (`tweedie.py:315`) already converts a coefficient of
  variation to a dispersion.
- `tweedie_convert` returns a `pandas.Series` indexed by Greek letters; printing
  it on a cp1252 console raises `UnicodeEncodeError` (measured 2026-10-28). The
  new object must be ASCII throughout.
- A bare expression is the top-level alternative `expr -> answer_expr`, returning
  `("expr", text, value)`. The underwriter treats `'expr'` as an answer at four
  code sites and documents it at three more: `_factory` builds nothing and
  returns the value (`src/aggregate/underwriter.py:2081`); `_interpret_program`
  does not store it (`:2695`); the preview (`:2872`) and the test-file
  interpreter (`:3519`) do not add it to the scratch base; docstrings at `:667`,
  `:745`, `:2320`. The kind lists in the parser docstrings
  (`parser.py:3096`, `:3115`) and in `src/aggregate/recipe.py:52` include it.
- `build_many` updates only objects that are instances of the updatable classes,
  so an answer object is never updated.

## Mathematics

The reproductive Tweedie $\mathrm{Tw}_p(\mu, \sigma^2)$ has mean $\mu$ and
variance $\sigma^2\mu^p$ [@Jorgensen1997]. For $1<p<2$ it is a compound Poisson
distribution with gamma severity:

$$\lambda = \frac{\mu^{2-p}}{(2-p)\sigma^2}, \qquad
\alpha = \frac{2-p}{p-1}, \qquad m = \frac{\mu}{\lambda}, \qquad
\nu_X = \alpha^{-1/2},$$

where $\lambda$ is the expected claim count, $\alpha$ the gamma shape, and $m$ and
$\nu_X$ the severity mean and coefficient of variation. The Tweedie coefficient
of variation is $\nu = \sigma\mu^{p/2-1}$, so the CV form sets
$\sigma^2 = \nu^2\mu^{2-p}$. In the other direction, from $(\lambda, m, \nu_X)$:
$\alpha = \nu_X^{-2}$, $p = (\alpha+2)/(\alpha+1)$, $\mu = \lambda m$ and
$\sigma^2 = \lambda m^2(1+\nu_X^2)/\mu^p$. The probability of no loss is
$e^{-\lambda}$. All of these formulas are already implemented in `tweedie_convert`.

Worked example: $p=1.5$, $\mu=100$, $\sigma^2=2$ gives $\lambda=10$, $\alpha=1$,
$m=10$, $\nu_X=1$, $\nu=0.447214$ and $\Pr(X=0)=e^{-10}=4.54\times10^{-5}$.
The compound variance check is $\lambda m^2(1+\nu_X^2) = 2000 = \sigma^2\mu^p$.

## Design

### [Conversion-Kind] the statement kind

The parser returns `("conversion", <family>, <object>)`, where the family is
`'tweedie'` here and `'stable'` in the stable plan, and the object is the
computed answer. A module constant in `src/aggregate/underwriter.py`,

```python
#: Statement kinds that are answers rather than declarations: built by
#: returning the parsed value, never stored in the recipe base.
ANSWER_KINDS = frozenset({'expr', 'conversion'})
```

replaces the four `== 'expr'` / `!= 'expr'` tests with membership tests. The
docstrings that list the kinds gain `'conversion'`. `build` returns the object, as
it returns an expression's float.

### [Tweedie-Statement-Grammar] the grammar

In `src/aggregate/decl.lark`:

```
answer: ...
      | conversion_out   -> answer_conversion

conversion_out: TWEEDIE expr expr expr        -> conversion_tweedie_dispersion
              | TWEEDIE expr expr CV expr     -> conversion_tweedie_cv
              | TWEEDIE expr CLAIMS SEV sev   -> conversion_tweedie_compound
```

and one new `agg_body` alternative beside the existing one:

```
        | TWEEDIE expr expr CV expr        -> agg_body_tweedie_cv
```

`TWEEDIE`, `CV`, `CLAIMS` and `SEV` are existing terminals, already reserved and
already mapped in `src/aggregate/parser_errors.py`. No other `answer` alternative
starts with `TWEEDIE`, and the three forms differ in their second or fourth
token, so no ambiguity is introduced. The `conversion_out` rule is written with
room for the stable plan's alternatives.

### [Tweedie-Statement-Transformer] the transformer

- `conversion_tweedie_dispersion` and `conversion_tweedie_cv` check
  $1<p<2$ with the message `agg_body_tweedie` uses today (factor it into one
  helper, `_check_tweedie_p`, called by all four tweedie methods), and require a
  positive mean, dispersion and CV.
- `conversion_tweedie_compound` takes the severity dict the `sev` rule produces
  and accepts a plain gamma only: `sev_name == 'gamma'`, scalar parameters, and
  none of `sev_loc` (nonzero), `sev_reflect`, `sev_wt`, `sev_lb`, `sev_ub`, the
  picks keys, or `sev_conditional`. Anything else raises
  `ValueError("tweedie: the compound form needs a plain gamma severity, ...")`
  stating what was found. With `sev_mean` present (the `gamma 10 cv 1` spelling,
  already multiplied by any scale) the mean and CV are taken directly; otherwise
  $m = \alpha\cdot$`sev_scale` and $\nu_X = \alpha^{-1/2}$ from `sev_a`. The count
  must be positive.
- `answer_conversion` returns `("conversion", "tweedie", obj)`.
- `agg_body_tweedie_cv` computes the dispersion from the CV and returns what
  `agg_body_tweedie` returns, plus `"_tweedie_cv": cv`.

### [Tweedie-Conversion-Object] `TweedieConversion`

In `src/aggregate/tweedie.py`, beside `TweedieParameters`, and added to
`__all__` (so it is top-level public, as `TweedieParameters` is):

```python
class TweedieConversion(namedtuple('TweedieConversion',
        ['p', 'mean', 'dispersion', 'cv', 'frequency', 'sev_m', 'sev_cv', 'p0'])):
    @classmethod
    def from_dispersion(cls, p, mean, dispersion): ...
    @classmethod
    def from_cv(cls, p, mean, cv): ...
    @classmethod
    def from_compound(cls, frequency, sev_m, sev_cv): ...
    def __str__(self): ...
```

Field names follow the module's existing vocabulary (`frequency`, `sev_m`,
`sev_cv`, `p0` as in `Tweedie` and `tweedie_convert`). The constructors call
`tweedie_convert`. `__str__` returns the three spellings joined by ` = `, then
`, Pr(X=0) ` and the probability, numbers formatted `.6g`, ASCII only. The
separator is not `;`, because `;` ends a DecL statement and a pasted string would
then parse as several programs. `as_tweedie()` still returns
`TweedieParameters`; the docstring of `TweedieConversion` shows the one-liner
`TweedieConversion.from_dispersion(*a.as_tweedie())`.

### [Tweedie-Clause-CV] the coefficient of variation form inside an `agg`

`Aggregate.__init__` gains `_tweedie_cv=None` after `_tweedie`, documented as
private provenance, and dropped from `self._spec` when `None` by the rule that
already drops `_tweedie` (`_aggregate.py:2819`), so spec hashes are unchanged for
every aggregate not written in the CV form. `_render_tweedie` renders
`tweedie <p> <mean> cv <cv>` when `_tweedie_cv` is set, from the declared CV, so
there is no rounding from converting the dispersion back. The pnl paths need no
code: `agg_source_inline` (`parser.py:996`) and the engine merge (`parser.py:1060`)
copy every key not on their metadata skip list, and `_render_pnl` calls
`_render_tweedie` (`decl_writer.py:1145`). Update the comments there that name
`_tweedie` alone. The constructor-parameter test in `decl_writer.py:1423` already
excludes private names.

## Steps

### [Tweedie-Statement], one minor bump

1. `[Conversion-Kind]`, `[Tweedie-Statement-Grammar]`,
   `[Tweedie-Statement-Transformer]`, `[Tweedie-Conversion-Object]` and
   `[Tweedie-Clause-CV]` as above.
2. Tests (below).
3. Grammar reference: regenerate `docs/4_agg_language_reference/ref_include.rst`
   with `parser.grammar(add_to_doc=True)`.
4. Documentation: a section "Parameter Conversions (tweedie)" in
   `docs/4_dec_Language_Reference.rst`, after "Numeric Expressions" (line 86),
   with the three forms, the worked example, and the statement that a conversion
   is an answer that is not stored. The stable plan extends the same section.
5. aLL help: in `src/aggregate/data/decl-help/card.html`, box 10 ("The library,
   and more"), one line: `tweedie 1.5 100 2` converts between Tweedie
   parameterizations.
6. Corpus: an `agg` in the CV form in `src/aggregate/agg/decl-testers.agg`, near
   the existing tweedie examples. Bare statements go in tests only, not in `.agg`
   files, which declare objects.
7. Release: `__version__`, `CHANGELOG.md`, `dev/TODO.md` (a `[Tweedie-Statement]`
   entry closed at this version; mention `[Power-Variance-Family]` as the open
   item that lifts the $1<p<2$ restriction, now in
   `dev/plan-characteristic-function-engine.md`), `dev/FEATURES.csv` via
   `dev/regen_features.py`; move this plan to `dev/done/`.

## Files

| Area | Files |
|---|---|
| Grammar | `src/aggregate/decl.lark` |
| Parser | `src/aggregate/parser.py` (four tweedie methods, `answer_conversion`, `_check_tweedie_p`, docstrings at :3096 and :3115) |
| Underwriter | `src/aggregate/underwriter.py` (`ANSWER_KINDS`; :2081, :2695, :2872, :3519; docstrings :667, :745, :2320), `src/aggregate/recipe.py` (:52) |
| Tweedie | `src/aggregate/tweedie.py` |
| Aggregate | `src/aggregate/_aggregate.py` (signature :2634, parameter docs :2786, spec pop :2819) |
| Writer | `src/aggregate/decl_writer.py` (`_render_tweedie`, comments near :1143) |
| Help | `src/aggregate/data/decl-help/card.html` |
| Tests | `tests/test_tweedie.py` |
| Corpus | `src/aggregate/agg/decl-testers.agg` |
| Documentation | `docs/4_dec_Language_Reference.rst`, `docs/4_agg_language_reference/ref_include.rst` |
| Release | `src/aggregate/__init__.py`, `CHANGELOG.md`, `dev/TODO.md`, `dev/FEATURES.csv` |

## Names checked

On 2026-10-28, `rg` over `src/aggregate` found no use of `TweedieConversion`,
`ANSWER_KINDS`, `_tweedie_cv`, `_check_tweedie_p`, `conversion_out`,
`answer_conversion`, `agg_body_tweedie_cv`, or the kind string `'conversion'`.
`from_dispersion`, `from_cv` and `from_compound` are classmethods of a new class
and shadow nothing. Recheck at execution.

## Acceptance and tests

In `tests/test_tweedie.py`:

- **Worked example.** `build('tweedie 1.5 100 2')` returns a `TweedieConversion`
  with `frequency == 10`, `sev_m == 10`, `sev_cv == 1`, `cv` equal to
  `0.447213595499958` and `p0 == exp(-10)`, all to `rtol=1e-12`.
- **Three forms agree.** The dispersion, CV and compound forms of the same
  distribution give equal fields to `rtol=1e-12`, for $p$ in
  $\{1.005, 1.5, 1.995\}$.
- **Every gamma spelling.** `sev gamma 10 cv 1`, `sev 10 * gamma 1` and
  `sev 5 * gamma 20 cv 1` in the compound form give the same answer.
- **Refusals.** A lognormal severity, a shifted gamma (`sev gamma 10 cv 1 + 5`),
  a gamma mixture, $p$ outside $(1,2)$ in both reproductive forms, and a zero
  count each raise `ValueError` that states the form, never `ZeroDivisionError`.
- **String.** ASCII only; each of the three spellings in `str(obj)` builds through
  the statement to fields within `rtol=1e-5` of the original.
- **Nothing stored.** The recipe base is unchanged by `build('tweedie 1.5 100 2')`;
  `build_many` returns one `Recipe` of kind `'conversion'`.
- **Expressions unchanged.** `build('(2+2)') == 4`.
- **Agreement with recognition.** For `agg X 10 claims sev gamma 10 cv 1 poisson`,
  `as_tweedie()` matches the calculator's `(p, mean, dispersion)` to `rtol=1e-12`.
- **CV clause.** `agg X tweedie 1.5 100 cv 0.447213595499958` builds; `pprogram`
  contains that clause verbatim; the parse of the rendered text is a fixed point;
  `as_tweedie().dispersion` is 2 to `rtol=1e-12`; the clause inside a `pnl`
  is rendered in the CV form.
- **Hashes stable.** `_spec_hash()` of a dispersion-form tweedie and of a
  non-tweedie aggregate are unchanged by this work.
- **Snapshot.** `test_spec_matches_snapshot` passes without re-capture: the
  grammar change adds alternatives and moves no existing line. A line that moves
  is a finding.

## Release gates

Edit loop: `tests/test_tweedie.py`, `tests/test_decl_parser.py`,
`tests/test_decl_unparser.py`. At the bump: the full fast suite, then
`uv run pytest -m 'slow or not slow'`, with `UV_PROJECT_ENVIRONMENT=.venv`. Docs
are not built.

## Deferred

- `[Status-Bar-Text-Answers]`, the aLL, Excel and Sheets display, in
  `V:/dev/aggregate-api/dev/plan-status-bar-text-answers.md`.
- `[Power-Variance-Family]`: $p$ outside $(1,2)$, which has no frequency and
  severity form; see `dev/plan-characteristic-function-engine.md`.
- Further calculators on the `conversion` kind (lognormal and gamma mean and CV,
  negative binomial), not proposed.
- Teaching material that can use the statement: the monograph page
  `D:/projects/aggregate/monograph/posts/3_user_guides/DecL/_100_tweedie.qmd` and
  the education module `D:/projects/aggregate/education/modules/08-the-tweedie-connection`.
  (`dev/done/plan-tweedie.md` gives the monograph's former location,
  `C:/s/AI/aggregate-monograph`.)

## Execution log

**1.22.0, 2026-10-28.** Reviewed and executed in one session against 1.21.0.
The proposals marked "to confirm at review" (5, the compound form with `claims
sev`, and 6, six significant figures) were confirmed as written. The cited line
numbers had drifted by at most a few lines (`as_tweedie` is at
`_aggregate.py:6449`); every symbol and snippet matched.

Divergences:

1. **The "every gamma spelling" test uses `sev 0.5 * gamma 20 cv 1`, not
   `sev 5 * gamma 20 cv 1`.** The scale multiplies the mean, so the plan's
   spelling has mean 100 and is a different law from `gamma 10 cv 1` (measured:
   `fz.mean()` is 100).
2. **The compound form refuses non-default values of `sev_wt`, `sev_lb`,
   `sev_ub`, `sev_loc`, `sev_reflect` and `sev_conditional`, not their
   presence.** The `sev` rule always sets `sev_wt = 1`, `sev_lb = 0` and
   `sev_ub = inf` (from `weights_none` and `splice_none`), so refusing the keys
   would refuse every gamma. Any key outside the gamma parameters, the defaults
   and the trailer of a stored `sev.NAME` (`note`, `tags`, `hints`, `label`) is
   refused, which covers picks, `xps` and a second shape parameter. Vector
   parameters and a nonpositive shape, count, mean or CV are refused too.
3. **`build_many` skips the answer kinds by kind.** Without it, a
   `TweedieConversion` (neither a `numbers.Number` nor an updatable class) logs
   "Unexpected: build produced a ..." on every build. The plan's claim that an
   answer is never updated was right; the warning was not covered.
4. **`_check_tweedie_p` is wrapped by `_check_tweedie`**, which adds the
   positivity checks for the mean, dispersion and CV and names the form in the
   message. All four reproductive tweedie methods call it. The p message now
   names both reproductive forms.
5. **The string shows `Pr(X=0) 4.53999e-05`, not the Goal's `4.54e-05`**: ruling 6
   (six significant figures) applied to every number. The docs and the
   docstring example show the actual string.
6. **`dev/TODO.md` gets an open `[Power-Variance-Family]` entry rather than a
   closed `[Tweedie-Statement]` one.** The file's convention since 2026-10-07
   deletes finished items instead of ticking them, and `[Power-Variance-Family]`
   was only in the archive.
7. **Hash stability is pinned by value.** `_spec_hash()` of
   `agg TWH1 tweedie 1.5 100 2` (`618DBE6E`) and of
   `agg TWH2 10 claims sev lognorm 50 cv 1 poisson` (`F0160E2C`) were measured
   at HEAD (1.21.0, from a `git archive` of `src`) and in the working tree, and
   agree; the test pins both.
8. **The corpus section is new, `TWS.`, at the end of `decl-testers.agg`**:
   the corpus had no tweedie examples to sit near. It carries a dispersion form,
   a CV form and a CV form `pnl` engine.

Tests: `tests/test_tweedie.py` 70 passed (31 new cases). Edit loop
`tests/test_tweedie.py tests/test_decl_parser.py tests/test_decl_unparser.py
tests/test_grammar_ambiguity.py tests/test_grammar_sync.py
tests/test_underwriter.py`: 1,868 passed. The snapshot test passed without a
re-capture. `dev/FEATURES.csv` regenerated with no change. The grammar
reference regenerated with additions only. Gate,
`UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'`: 6,942
passed in 153 s.

## References

::: {#refs}
:::
