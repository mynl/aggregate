# Plan 1.4.0: exact arithmetic in nttagg probabilities, and an exact tail chart

Two independent steps, two version bumps, two commits:

- **[NTT-Rational-Arithmetic]** `1.4.0`: nttagg probability lists accept
  parenthesized arithmetic (`+ - * /`, `**`), evaluated exactly as `Fraction`.
- **[NTT-Exact-Tail-Chart]** `1.5.0`: a new chart, `exact_tail`, that draws the
  exact answer as base-10 logarithms, so the deep tail the exact computation
  exists for is visible at last.

The versions assume `dev/plan-1.3.0-ntt-engine-status.md` lands first. If it
has not, take the next free minor at execution time; the steps do not depend on
it or on each other.

Companion plans in `V:\dev\aggregate-api\dev\`: `plan-1.13.0-exact-tail-leaf.md`
puts the chart on a More leaf in aLL and is written against the contract in
step 2 below, so **step 2's names are fixed**: change them here and there
together. `plan-1.12.0-raw-display-large-grids.md` is unrelated to this plan.

## Background

`nttagg` declares a bounded discrete compound that the library computes
**exactly**, as rationals, through the number-theoretic transform, alongside the
ordinary float FFT answer on the same integer lattice. The class is
`AggregateNTT` (`src/aggregate/ntt.py:1481`), a subclass of `Aggregate`.
Probabilities are *text exact*: the parser reads token text, so `0.3` is exactly
`3/10` and `a/b` is `Fraction(a, b)`. The exact answer is stored as integer
numerators `_ntt_numerators` over one common denominator `_ntt_denominator`.
`density_df` carries `p_exact_float` (the exact value rounded to a double, which
underflows to 0 below about 1e-308) and `p_exact_log10` (computed from the
integers, so it stays readable at 1e-462 and beyond).

---

## Step 1 [NTT-Rational-Arithmetic]

### Goal

Let the author write probabilities as exact expressions, for example

```
nttagg X dfreq [1:3] dsev [1:3] [1/2 (1/2-1/(13*11*(19**3))) (1/(13*11*(19**3)))]
```

which is `[1/2 980835/1961674 1/980837]`, summing to exactly 1.

### Current behavior

`decl.lark:579-589`: a probability is `NTT_NUMBER` or `NTT_NUMBER DIVIDE
NTT_NUMBER`, nothing else. The example above fails to lex at the first `(`:

```
DecL parse error at line 1, column 38: Unexpected character '('.
Expected: a number (nttagg: nonnegative, no percent, no inf)
```

`agg` accepts the same text because its `dprobs` go through `numberl -> expr`,
whose parenthesized arithmetic (`decl.lark:760-785`, the "paren island")
evaluates in binary64 at parse time. nttagg cannot reuse that path: the
transformer's `NUMBER` terminal callback (`parser.py:472`) converts every
`NUMBER` to a float before any rule callback runs, which is why nttagg has its
own `NTT_NUMBER` terminal (`decl.lark:907`, no sign, no percent, no inf).

### Change

**Grammar** (`decl.lark`, replacing `ntt_rat` at lines 588-589; `ntt_ratl`
unchanged). Same shape as `agg`'s ladder, so one rule holds in both kinds: a
bare item sees a literal, `/`, `**` / `^` and parentheses; `+`, `-` and `*` are
legal only inside parentheses. That keeps a whitespace-separated list with a
single reading. Unary minus is admitted inside parentheses and right after
`**`, which is what makes `2**-3` and `(1--1/2)` work; an item can still never
*start* with a sign, so the "nonnegative" error text stays true.

```
?ntt_rat: ntt_rat DIVIDE ntt_power          -> ntt_divide
        | ntt_power

?ntt_power: ntt_atom EXPONENT ntt_unary     -> ntt_exponent
          | ntt_atom

ntt_atom: NTT_NUMBER                        -> ntt_number
        | "(" ntt_sum ")"                   -> ntt_parens

?ntt_sum: ntt_sum PLUS ntt_product          -> ntt_add
        | ntt_sum MINUS ntt_product         -> ntt_subtract
        | ntt_product

?ntt_product: ntt_product TIMES ntt_unary   -> ntt_multiply
            | ntt_product DIVIDE ntt_unary  -> ntt_divide
            | ntt_unary

?ntt_unary: MINUS ntt_unary                 -> ntt_negate
          | ntt_power
```

Precedence and associativity are Python's: `-2**2` is `-4`, `2**-2**2` is
`2**(-(2**2))`, `1/2/3` is `(1/2)/3`, and `**` is right associative. Update the
section comment above `nttagg_out` (`decl.lark:552-568`) to describe the
arithmetic.

**Transformer** (`parser.py`, replacing `ntt_rat_number` and `ntt_rat_fraction`
at lines 2134-2158). Every callback returns a `Fraction`; none touches a float.

| callback | body |
|---|---|
| `ntt_number` | `Fraction(token text without '_')`, the current `ntt_rat_number` body |
| `ntt_parens` | `c[0]` |
| `ntt_add` / `ntt_subtract` / `ntt_multiply` | `a + b`, `a - b`, `a * b` |
| `ntt_divide` | `a / b`; if `b == 0` raise `ValueError(f'nttagg probability {a}/{b} has a zero denominator')` |
| `ntt_negate` | `-a` |
| `ntt_exponent` | see below |

`ntt_exponent(base, power)`:

1. `power.denominator != 1` raises `ValueError('nttagg exponent must be a whole
   number, got {power}')`. A fractional power of a rational is in general
   irrational, so there is no exact answer to give.
2. Size guard: with `n = power.numerator`, refuse when `abs(n) *
   max(base.numerator.bit_length(), base.denominator.bit_length())` exceeds
   `NTT_EXPONENT_MAX_BITS`, a new constant `2 ** 20` in `constants.py` beside
   `NTT_MAX_SUPPORT` (line 178). The message names the expression size and the
   cap. This turns a typo like `10**10**10` into an error instead of a hang.
3. `base == 0 and n < 0` raises the zero-denominator message.
4. Return `base ** n`, which `Fraction` computes exactly for an `int` power.

Exactness is enforced by construction: there is no `exp`, no float, no
transcendental anywhere in the island.

Note on names: the callbacks are deliberately **not** shared with `agg`'s
`atom_add` / `atom_divide` / `atom_exponent` (`parser.py:2669-2697`). Those
would compute correctly on `Fraction` for most operators, but `atom_exponent`
silently returns a float for a fractional power, and `atom_exp` is
transcendental. Separate `ntt_*` callbacks keep the "nttagg never touches a
float" invariant local and checkable. Names vetted with `rg`: none of
`ntt_number`, `ntt_parens`, `ntt_add`, `ntt_subtract`, `ntt_multiply`,
`ntt_divide`, `ntt_negate`, `ntt_exponent`, `NTT_EXPONENT_MAX_BITS` exists today.

**Behavior changes to record.** `a/b` with decimal operands is now accepted
(`0.5/0.25` is 2, which the existing sum check then rejects as a probability),
and the message "nttagg a/b probabilities take integers" disappears. The
zero-denominator message keeps its wording, so `1/0` reads exactly as today.

**Writer.** No change. `decl_writer._fmt_ntt_rat` (`decl_writer.py:1204`)
prints reduced fractions, so the reformatted program is exact but the
expression is gone: the example reformats to `[1/2 980835/1961674 1/980837]`.
That is the same trade `agg` makes when it prints the evaluated float.

**Errors.** Callback `ValueError`s surface as plain `ValueError` messages without
a column, exactly as the existing nttagg checks do today (verified:
`nttagg X dfreq [1] dsev [1 2] [1/0 1/3]` raises `ValueError: nttagg probability
1/0 has a zero denominator`). No change to `parser_errors.py`: `PLUS`, `MINUS`,
`TIMES`, `DIVIDE`, `EXPONENT`, `LPAREN` and `RPAREN` already have readable names
there (lines 74-82).

### Files

- `src/aggregate/decl.lark`: the block above, plus the section comment.
- `src/aggregate/parser.py`: the callbacks; delete `ntt_rat_number`,
  `ntt_rat_fraction`.
- `src/aggregate/constants.py`: `NTT_EXPONENT_MAX_BITS`.
- `docs/4_agg_language_reference/ref_include.rst`: regenerate with
  `uv run python -c "from aggregate.parser import grammar; grammar(add_to_doc=True)"`.
- `docs/4_dec_Language_Reference.rst:120`: after "an explicit ``a/b`` is the
  fraction itself", add one sentence: parenthesized arithmetic (`+ - * /`, and
  `**` with a whole-number power) is evaluated exactly, with the example above.
- `src/aggregate/agg/_test_suite.agg`: one line in the NTT section (after line
  355), `nttagg NTT.Arithmetic dfreq [1:3] dsev [1:3] [1/2
  (1/2-1/(13*11*(19**3))) (1/(13*11*(19**3)))] note{exact arithmetic inside
  parentheses}`, then regenerate the snapshot with `uv run python
  tests/capture_spec_snapshot.py`. The diff must be exactly the new line's two
  entries; any other moved line is a finding.
- `tests/test_ntt.py`: tests below; add the example program to `NTT_PROGRAMS`
  (line ~320) so `test_nttagg_writer_round_trip` covers it.

### Tests (`tests/test_ntt.py`, beside `test_nttagg_parses_text_exact`)

- The example parses to `[Fraction(1, 2), Fraction(980835, 1961674),
  Fraction(1, 980837)]`.
- Each operator and precedence rule: `(1/2+1/4)`, `(1-1/3)`, `(2*1/8)`,
  `(1/2)/2`, `1/2/2`, `2**-3` as `1/8`, `(-2**2+5)` as `1`, `2**-2**2` as
  `1/16`, nested parentheses.
- Decimal and scientific literals stay exact inside expressions: `(0.1+0.2)` is
  exactly `3/10`, which `agg` cannot say.
- Rejections, each matched on its message: `2**(1/2)` (whole number), `(1/0)`
  and `0**-1` (zero denominator), `10**10**10` (size cap), and a leading sign
  `[-1/2 ...]` still a parse error.
- The existing `test_nttagg_rejects_bad_probabilities` keeps passing unchanged.

### Acceptance

- The example builds: `build('nttagg X dfreq [1:3] dsev [1:3] [1/2
  (1/2-1/(13*11*(19**3))) (1/(13*11*(19**3)))]')` returns an `AggregateNTT`
  whose spec holds the three fractions above, and `pprogram` shows them reduced.
- Spec snapshot diff is exactly the new corpus line.
- Tier 2 (`uv run pytest`) green, then tier 3 (`uv run pytest -m 'slow or not
  slow'`) at the bump.

---

## Step 2 [NTT-Exact-Tail-Chart]

### Goal

A chart that shows the exact answer all the way down: the base-10 logarithm of
the exact mass and of the exact survival function, with the float FFT answer
beside it so the reader sees where the float runs into noise and the exact line
keeps going.

### Current behavior

`AggregateNTT` registers no chart of its own, so it inherits `agg`
(`charts/_emit_aggregate.py:192-251`), which reads `density_df.p_total`, the
float FFT answer (line 229). `a.plot()` (`_aggregate.py:5899`) and aLL's
Overview Plot both draw that document. The exact numbers never reach a chart.
Drawing `p_exact_float` on a log axis would not fix it: it underflows to 0 below
about 1e-308, and a chart document carries doubles. Only a document whose
ordinate **is** the logarithm can show a bucket at 1e-462.

### Contract (consumed by the aLL plan; do not rename without updating it)

- Registry name **`exact_tail`**; emitter **`chart_exact_tail`** in a new module
  **`src/aggregate/charts/_emit_ntt.py`**, registered with
  `@chart_exact_tail.register(AggregateNTT)` and
  `register_chart('exact_tail', chart_exact_tail, predicate=_updated)`, where
  `_updated(a)` is `a._ntt_numerators is not None`. **Not primary**: Overview
  Plot stays `agg`, which is the right picture of the body of the distribution.
  Import it at the bottom of `charts/__init__.py` with the others and add it to
  `__all__`.
- **Two new axis units** in `charts/ir.py` `AXIS_UNITS` (line 170), mirroring
  the existing `'density'` / `'probability'` pair that renderers already
  dispatch on: **`'log10_density'`** (the base-10 logarithm of a mass) and
  **`'log10_probability'`** (the base-10 logarithm of a cumulative or
  exceedance probability). Values are exponents: a tick at -400 means 1e-400.
  Document both in the comment above `AXIS_UNITS`.
- **`CHART_IR_VERSION` stays at 4 (decided by the author, 2026-10-14).** By the rule
  in `ir.py:88-97` the case is arguable: a reader that ignores
  `'log10_density'` draws the mass series as steps or a line, which bridges
  omitted exact zeros, and that is drawing something wrong. But the version is
  global, stamped on every document, and aLL's adapter refuses anything above
  4, so a bump makes **every** chart undrawable in any aLL deployed against
  this library before its adapter moves. The new units appear only in
  `exact_tail` documents, which a reader receives only by asking for that name,
  and the two readers that ask (`plot_chartdoc` here, the aLL leaf in its
  companion plan) learn the units in the same release. Not bumping therefore
  draws nothing wrong in any shipped combination; bumping risks blanking all
  charts. Record this reasoning as a paragraph in the `CHART_IR_VERSION`
  comment, as the precedent for "a unit used only by a chart a reader must
  request by name".
- **Axes**:
  - `outcome`: label 'Loss', unit 'currency', scales `('linear', 'log')`,
    `suggested_range = (0, top)` where `top` is the largest loss with positive
    exact mass, `full_range = (0, xs[-1])`. The window is the **whole support**,
    not `agg`'s quantile crop, because the tail is the point.
  - `log10_mass`: label 'log10 probability mass', unit 'log10_density', scales
    `('linear',)`, `suggested_range = (lo, 0)`.
  - `log10_survival`: label 'log10 exceedance probability', unit
    'log10_probability', scales `('linear',)`, `suggested_range = (lo, 0)`.
  - `lo` is the smallest finite value in the exact series of that panel, rounded
    down to a multiple of 10.
- **Panels**: `mass` (kind 'xy', x `outcome`, y `log10_mass`, title 'Exact
  probability mass') and `survival` (kind 'xy', x `outcome`, y
  `log10_survival`, title 'Exact survival function'). Both read on x (the
  default `read_axis`).
- **Series**, two per panel, built with `lattice_payload(x, 1.0)` so a
  contiguous run ships as a lattice and anything with holes falls back to
  explicit x on its own:
  - `mass` / 'Exact', role 'density': `density_df.p_exact_log10` at the buckets
    where it is finite. Exact zeros, interior ones included, are **omitted**.
  - `mass` / 'Float FFT', role 'density': `log10(p_total)` where `p_total > 0`.
  - `survival` / 'Exact', role 'survival': the exact survival below, finite
    entries only. S is positive for every loss below `top` and exactly 0 from
    `top` on, so this is a gap-free prefix.
  - `survival` / 'Float FFT', role 'survival': `log10(density_df.S)` where
    `S > 0`.
- **No marks.** The comparison series already shows the float's noise floor.
- **Drawing rule**, for every renderer: a `'log10_density'` series is drawn as
  **markers only** (no line, no stems, no steps). A line would bridge omitted
  exact zeros and claim mass where there is none, and a log axis has no floor to
  drop a stem to. A `'log10_probability'` series draws exactly as
  `'probability'` does today (right-continuous steps when there is room, else a
  line).

### Change

1. **Exact survival** on `AggregateNTT`, private, beside
   `_exact_log10_on_grid` (`ntt.py:1754`):
   `_exact_survival_log10_on_grid()` returns, per grid point `x_i`,
   `log10(sum_{j > i} H_j) - log10(D)`, or `-inf` where the sum is 0. One
   reverse cumulative pass over `_exact_numerators_on_grid()` in Python
   integers, then `math.log10` per bucket exactly as the mass version does.
   Complete because `update` guarantees the grid covers the whole support
   (`ntt.py:1665-1669`). The NumPy docstring's `Notes` give the formula, the
   cost (one big-integer addition per bucket), and the ~1e-13 relative error in
   the logarithm inherited from `math.log10` of an integer. Name vetted: no
   `_exact_survival` exists.
2. **Emitter** `charts/_emit_ntt.py`, pure numpy and pandas, no matplotlib,
   following `_emit_aggregate.py`'s structure and module docstring style.
   Build the `ChartDoc` and pass it through `complete_tex` with `typeset={'log10
   probability mass': r'$\log_{10}$ probability mass', 'log10 exceedance
   probability': r'$\log_{10}$ exceedance probability'}`.
3. **Units** in `charts/ir.py` as above.
4. **matplotlib renderer** `plots/_chartdoc.py`, `_draw_atomic` (line 447):
   before the `'probability'` branch at line 489, a `'log10_density'` y unit
   draws `ax.plot(x, y, ls='none', marker='.', ms=...)` with a marker size that
   stays legible at 65,536 points, and the `'probability'` test becomes `unit
   in ('probability', 'log10_probability')`. Tick labels stay plain numbers; the
   axis label says log10.
5. **Docs**: `AggregateNTT` class docstring gains one bullet naming the chart
   and the two-line way to draw it in Jupyter, `from aggregate.charts import
   build_chart_doc` / `from aggregate.plots import plot_chartdoc` then
   `plot_chartdoc(build_chart_doc(a, 'exact_tail'))`.
   `docs/3_reference/3_x_Charts.rst` gains one sentence after the `structure`
   paragraph (line 46) naming `exact_tail` and the two log10 units.

### Files

- `src/aggregate/charts/_emit_ntt.py` (new), `src/aggregate/charts/__init__.py`,
  `src/aggregate/charts/ir.py`
- `src/aggregate/ntt.py` (survival helper, class docstring)
- `src/aggregate/plots/_chartdoc.py`
- `docs/3_reference/3_x_Charts.rst`
- `tests/test_chart_exact_tail.py` (new), `tests/test_charts_ir.py`

### Tests

New `tests/test_chart_exact_tail.py`:

- `'exact_tail' in available_charts(nttagg)`; it is absent for a plain `agg`;
  `primary_chart(nttagg) == 'agg'`; `build_chart_doc` raises `ValueError` before
  `update()`.
- **Deep value**: `Wilson2016a.Wilson` from `agg/reproductions.agg:1151` (two
  claims, severity mass 1e-20 at 3). The exact mass series reads `-40` at loss 6
  and `log10(2e-25)` at loss 5, to 1e-10; the exact survival reads `-40` at
  loss 5 and has no point at 6 or beyond.
- **Interior zeros**: `nttagg T dfreq [0 1 2] [1/4 1/2 1/4] dsev [1 2 10]
  [3/10 1/2 1/5]` has zero mass at losses 5 to 9 (and elsewhere); the exact
  mass series has no point at any of them and therefore ships explicit x.
- **Exactness of the survival**: on `nttagg T dfreq [1:6] dsev [1:6]`, every
  survival point equals `math.log10` of the `Fraction` suffix sum of `p_exact`
  to 1e-12, and the series is nonincreasing.
- Wire round trip (`load_chart_doc(canonical_dict(doc))` reproduces the hash)
  and `plot_chartdoc` draws it without error.

`tests/test_charts_ir.py`: the tex sweep at lines 370-396 ends with `assert
swept == set(CHARTS) - {'envelope'}`, so adding a chart **fails it** until an
nttagg joins its object list. Add `build('nttagg IR.TexN dfreq [1:3] dsev
[1:3]')` there, and to the wire round-trip sweep at line ~874.

### Acceptance

- The Wilson test above passes, which is the whole point in one number: the
  float answer cannot represent 1e-40 against a body of order 1, and the chart
  shows it.
- On `ExactDiceOfDice` (`agg/library.agg:1544`), record the chart build time and
  the `canonical_json` size beside the `agg` chart's for the same object, in
  this plan's Results section. Expect the same order of size: four series of at
  most one grid each, against `agg`'s four.
- Tier 2 green; tier 3 at the bump. `plot_chartdoc` output eyeballed once in
  Jupyter on `Wilson2016a.Wilson` and `ExactDiceOfDice`.

---

## Release hygiene (each step, in its own bump commit)

- Version: the literal `__version__` in `src/aggregate/__init__.py`.
- `CHANGELOG.md`: one paragraph per version, bold `**[Label] what landed**`
  lede, facts only (new names, the relaxed `a/b` rule for step 1; the chart
  name and the two units for step 2).
- `dev/TODO.md`: add each label as done with its version and this plan.
- `dev/FEATURES.csv` via `dev/regen_features.py`; commit any diff.
- Step 1 only: grammar reference regen and spec snapshot regen, as above.
- Commit subject: `[NTT-Rational-Arithmetic] 1.4.0: exact arithmetic in nttagg
  probability lists` and `[NTT-Exact-Tail-Chart] 1.5.0: exact_tail chart with
  log10 mass and survival`. One line, no body, no trailers.
- Docs are not built in the loop; the commit notes they await a rebuild.
- The author moves this plan to `dev/done/` when they say it is done.

## Out of scope

- Arithmetic in **outcomes**. They are nonnegative integers and ranges cover
  them.
- Keeping the source expression through reformatting (the spec would have to
  carry text).
- A convenience plotting method on `AggregateNTT`; the two-line form above
  serves until a name is vetted and wanted.
- An exact survival column in `density_df`; the helper is private to the chart
  for now.
- Any change to `agg`'s own parenthesized arithmetic, which stays binary64.

## Results

(Filled in at execution: timings and document sizes from step 2's acceptance.)

## Execution log

Executed 2026-10-14 against 1.3.0 (`42c52e8`). Line numbers in the plan predate the 1.3.0 bump, which added about 80 lines to `ntt.py` (`AggregateNTT` is now at 1561, `_exact_log10_on_grid` at 1834, the lattice check in `update` at 1746); every named symbol was where the plan said, with the spelling it gave.

### Step 1 [NTT-Rational-Arithmetic], 1.4.0

Grammar and callbacks exactly as specified. Every case in the plan's test list was probed before the tests were written and came out as the plan says.

**Divergences.**

- **The power's zero-denominator message** reads `nttagg probability 0**-1 has a zero denominator`. The plan said "the zero-denominator message", whose `a/b` form does not describe a power; the test matches on `zero denominator` either way.
- **The leading-sign error** reads `DecL parse error at line 1, column 32: Unexpected '-1'.` with no `Expected:` line. That is byte identical to HEAD (checked by parsing the same text with HEAD's `decl.lark` swapped in), so nothing changed; the test matches on `DecL parse error`.
- **Two extra operator cases**, `2^-3` (the caret spelling of `EXPONENT`) and `(3e-1*1)` (a scientific literal inside an expression), and the acceptance check written as a test, `test_nttagg_arithmetic_builds`.
- **Not mirrored into `decl-testers.agg`.** The project's execute-plan skill asks that a new DecL program used in a pytest case be appended there. It has no NTT section, the existing `tests/test_ntt.py` programs are not mirrored either, and this plan names `_test_suite.agg` as the corpus home, so the rule was not applied. Worth a ruling if NTT programs should get a section there.
- **Snapshot diff** is one new key, `NTT.Arithmetic`, which both parametrized cases read; nothing else moved. `dev/FEATURES.csv` regenerated with no diff.
- **`dev/TODO.md`** holds the author's uncommitted edits. Only the new `[NTT-Rational-Arithmetic]` entry was staged (the HEAD blob plus that hunk, written to the index directly); the author's edits stay unstaged in the working copy.
- **This plan file** was untracked. It is committed in place, with this log, in the step 1 commit. It stays in `dev/`, not `dev/done/`, because the plan says the author moves it.
