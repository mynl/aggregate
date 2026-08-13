# plan-math-expressions: [DecL-Paren-Arithmetic] full arithmetic inside parentheses

> **Status: EXECUTED at 1.0.0a268, 2026-08-13.** Landed as planned, with two
> notes. (1) The version is a268, not a267: a parallel session committed
> `[BS-Window-Formats]` as a267 while this was in flight, so the bump took the
> next free number. (2) Section 6 invariant 1 came out exactly as predicted:
> the snapshot regen moved only the two new `_test_suite.agg` lines
> (`E.TSev19`, `F.Expos06`) and nothing else. (3) One naming correction:
> decision 5 calls the percent-marker class `_PctFloat`; the class in
> `parser.py` is `_PercentNumber`. The behavior described is exactly right and
> its own docstring says so, only the name was wrong. Everything below is the
> plan as approved; nothing in it was relitigated.
>
> **Original status: READY FOR EXECUTION.** Drafted 2026-08-13 in the oversight session;
> the design was reviewed and approved by the author the same day. Every
> decision in section 2 is a settled ruling: execute, do not relitigate. LIB
> only, no API half, no symlink. One version bump. The regression bar is
> section 6: every existing program parses to a byte-identical spec, proven by
> an empty snapshot diff on the existing corpus.

---

## 1. What this adds and why

DecL already has a real expression sub-language: `?expr` / `?term` / `?factor`
/ `atom` (`decl.lark`, the "Expression atoms" section near line 662) supporting
`/`, `**` and `^`, `exp`, and parentheses, evaluated eagerly in the transformer
(`parser.py`, the `atom_*` methods near line 2331). What it lacks is `+`, `-`
and `*` at expression level, so a user cannot write a computed exposure, a
grossed-up premium, or a lognormal mean normalizer inline. Target example
(author's, corrected to current keywords):

```
agg TEST (4 + 3*2) claims (100_000/(1-.25)) premium 1000 xs 0
    sev (exp(-1 * .4**2/2)) * lognorm .4 poisson
```

`(4 + 3*2) claims` is the exposure head, `(100_000/(1-.25)) premium` rides the
FYI premium suffix (`[FYI-Premium-Exposure-Head]`, 1.0.0a266), and the `sev`
multiplier is the mean normalizer `exp(-sigma^2/2)`.

**The rule in one sentence: `+`, `-` and `*` become legal only inside
parentheses; bare expressions keep exactly today's grammar (`/`, `**`, `^`,
`exp`, parens).**

The paren gate is load bearing, not a convenience. `PLUS`, `MINUS` and `TIMES`
already carry meaning at other levels (`sev2` shift arithmetic, `sev1` scaling,
the whole `builtin_agg` algebra), and `NUMBER` absorbs a glued leading minus
(priority 2, `decl.lark` near line 797), so admitting these operators into bare
`expr` creates genuine Earley ambiguities (`2 * 3 * agg.X` gets two parses) and
destabilizes vector lexing (`[1 -2]`). Inside parentheses none of that exists:
a paren holds exactly one expression, never a list, so the lexing
`NUMBER NUMBER` has no derivation there and the dynamic lexer's only viable
reading of `(1-.25)` or `(4 -3)` is the subtraction. All five operator
terminals plus `EXP` already exist in the grammar and already carry entries in
`parser_errors._TERMINAL_LABELS`, so this change adds **rules only**: no new
terminals, no lexer edits, no `ID` lookahead edits, no keyword mirrors to
update.

### A finding to carry into the docs: the glued-minus sign trap

`exp(-.4**2/2)` parses **today**, but to `exp(+0.08)`, not the intended
`exp(-0.08)`: `NUMBER` absorbs the minus, so `-.4` is one token and `-.4**2`
is `(-0.4)**2 = +0.16`. Python reads `-.4**2` as `-(0.4**2)` because unary
minus binds looser than `**`; DecL's token-level minus binds tighter. This
plan does NOT change that (decision 2). It documents it, and the new operators
supply the natural spellings for a negated computed value: `(-1 * x)` or
`(0 - x)`.

---

## 2. Settled design decisions (author rulings, 2026-08-13)

1. **[Paren-Gate]** `+`, `-`, `*` are legal only inside parentheses. Bare
   `expr` is untouched. Rejecting the bare-level alternative is final; the
   ambiguity analysis is in section 1.
2. **[No-Unary-Minus]** No unary minus production. `NUMBER` keeps its glued
   minus and its priority. Rationale: a unary rule makes `(-3)` lex two ways
   and makes the value of `(-3**2)` depend on which lexing wins, exactly what
   `tests/test_grammar_ambiguity.py` exists to keep out. The binding quirk is
   documented instead (section 8).
3. **[Evaluate-And-Canonicalize]** Expressions evaluate at parse time, as the
   existing `atom_*` methods do. Specs carry plain floats. `spec_to_decl` and
   `format_program` change **not at all**: the decompiled canonical text shows
   the evaluated literal (`(4 + 3*2) claims` renders as `10 claims`). This is
   the already-documented "canonical, not verbatim" contract
   (`decl_writer.py` module docstring, which names `exp(.5)` collapsing to its
   float). No source-text side channel, no shadow spec keys. The `_tweedie`
   precedent does not apply: that preserved semantic intent being overwritten,
   this is arithmetic sugar and the number is the meaning.
4. **[Exp-Unchanged]** `exp` stays exactly as is (`EXP factor`). It composes
   with the new levels for free.
5. **[Percent-Degrades]** `_PctFloat` (`parser.py` near line 348) survives
   literal use only, by existing design; arithmetic returns plain floats, so a
   computed value in the `po` position reads as an absolute amount. Keep, and
   add one doc sentence (section 8).

---

## 3. Grammar edit (`src/aggregate/decl.lark`)

Replace the paren interior with a new sum/product ladder reachable **only**
from `atom`. The existing `expr` / `term` / `factor` rules do not move:

```lark
?expr: term

?term: term DIVIDE factor       -> atom_divide
     | factor

?factor: EXP factor             -> atom_exp
       | atom EXPONENT factor   -> atom_exponent
       | atom

atom: NUMBER                    -> atom_number
    | "(" sum ")"               -> atom_parens

// Full arithmetic, reachable ONLY through the parens above (the paren
// island). Bare expressions stay on the term/factor ladder: adding these
// operators there would collide with severity shift/scale arithmetic and
// the builtin_agg algebra, and would destabilize vector lexing.
?sum: sum PLUS product          -> atom_add
    | sum MINUS product         -> atom_subtract
    | product

?product: product TIMES factor    -> atom_multiply
        | product DIVIDE factor   -> atom_divide
        | factor
```

Notes for the implementer:

- The `atom_divide` alias appears on two rules deliberately, so both dispatch
  to the one transformer method (child shape `[left, token, right]` is
  identical). If Lark rejects the duplicate alias (it should not), give the
  product-level division its own alias and have it delegate.
- Update the precedence comment block heading the section. The ladder becomes:
  `()` then `exp` then `**` `^` then `*` `/` then `+` `-`; `**` stays
  right-associative, everything else left-associative; bare expressions keep
  only `/`, `**`, `exp`, parens.
- `@` (`INHOMOG_MULTIPLY`), `=`, `:` and all keyword terminals are untouched.
- `^` works inside parens automatically (it is the same `EXPONENT` terminal).
- Because `?sum` and `?product` inline single children, `(2/3)` produces the
  identical post-transform tree it does today, which is why the snapshot diff
  comes out empty (section 6).

## 4. Transformer edit (`src/aggregate/parser.py`)

Three one-line methods next to the existing `atom_*` group (near line 2331),
matching their style (the section comment carries the documentation, as now):

```python
def atom_add(self, c):
    a, _, b = c
    return a + b

def atom_subtract(self, c):
    a, _, b = c
    return a - b

def atom_multiply(self, c):
    a, _, b = c
    return a * b
```

Extend the section comment to note the paren gate and that `_PctFloat`
degrades to plain float under any arithmetic (existing, intended behavior).

## 5. Writer and round trip: zero code change

Nothing in `decl_writer.py` changes. The transformer hands it floats, exactly
as with `exp(.5)` today; `_fmt_num` renders `repr`, the shortest literal that
re-parses to the same value, so `(100_000/(1-.25))` decompiles as
`133333.33333333334` and the idempotence invariant `f(f(f(x))) == f(x)` holds
with the collapse happening on the first application. Two small text touches
only:

- Extend the "Canonical, not verbatim" docstring list with one new example,
  `(4 + 3*2)` versus its evaluated float.
- The consequence worth naming in the CHANGELOG: `format_program` run over
  source text replaces a user's formula with its evaluated literal. That is
  the documented contract, accepted by ruling (decision 3). If the app ever
  wants formula-preserving reformatting of user source, that is a token-level
  text tool, a separate ask, never a spec change.

## 6. Invariants: what must not move

1. **Every existing program parses to a byte-identical spec.** Regenerate
   `tests/data/expected_specs.json` via
   `uv run python tests/capture_spec_snapshot.py` and read the diff: it must
   be empty apart from any new corpus lines added by this plan. A line you
   did not add that moves is a stop-and-investigate finding.
2. **Vector lexing is unchanged.** `[1 -2]` remains the two-element list;
   `[1 - 2]` remains a parse error. Both follow from bare `expr` never seeing
   `MINUS`; both get pinned by tests anyway.
3. **The gate holds.** Bare `4 + 3*2 claims` remains a parse error.
4. **No new terminals.** `test_grammar_sync.py` passes with zero edits to
   `AggLexer`, `agg.sublime-syntax`, or `_TERMINAL_LABELS`.
5. **DecL is stable tier** (LIB CHANGELOG preamble). This change is purely
   additive syntax; invariant 1 is the proof.

## 7. Tests

- **T1 [Snapshot-Empty-Diff]** Invariant 1, executed and the diff read
  deliberately.
- **T2 [Paren-Ambiguity-Guard]** Extend `tests/test_grammar_ambiguity.py`
  with paren-math programs asserting exactly one parse, including at least:
  `(4 + 3*2)`, `(1-.25)`, `(1 -2)`, `(4 -3)`, `(2 - -3)`, `(-.4**2)`,
  `(0 - .4**2/2)`, `(2**3**2)`, `(2+3*4)`, `((2+3)*4)`, `(2**3)` versus
  `(2*3)`, `(50% + 1)`, and embedded uses: `poisson (2+1)`,
  `(500*2) xs (0+0)`, a range `[(1+0):(3*2)]`.
- **T3 [Evaluation-Pins]** Parse-level value assertions: `(4 + 3*2)` is 10;
  `(100_000/(1-.25))` is 400000/3; `(2**3**2)` is 512 (right-associative);
  `(2+3*4)` is 14 and `((2+3)*4)` is 20; `(1 -2)` is -1; `(2 - -3)` is 5;
  `(50% + 1)` is 1.5; `exp(-1 * .4**2/2)` equals `math.exp(-0.08)`; and the
  documented quirk `(-.4**2)` equals +0.16.
- **T4 [Gate-Negatives]** Parse errors, asserted: bare `4 + 3*2 claims`,
  `[1 - 2]`, `(1 2)` (parens are not lists), `(1 +)`. For the last, eyeball
  the `ErrorReport` text once: the expected-token list should read sensibly
  from the existing `_TERMINAL_LABELS` entries.
- **T5 [Unparser-Collapse]** New cases in `tests/test_decl_unparser.py`
  pinning that a program written with paren math decompiles to the evaluated
  literals and that the canonical text is a fixed point.
- **T6 [Corpus-Lines]** Add example lines to the shipped corpus that
  `tests/test_decl_parser.py` parametrizes (check which of
  `src/aggregate/agg/*.agg` feeds it; the files are `_test_suite.agg`,
  `_test_suite2.agg`, `decl-testers.agg`, `library.agg`). Each new line
  automatically gains the parses test, the snapshot pair, and the
  `test_grammar_sync` colorer corpus sweep (zero `Token.Error`).

Run tier 2 (`uv run pytest`) before declaring done and tier 3
(`uv run pytest -m 'slow or not slow'`) at the bump, per house rules.

## 8. Docs

- **Grammar listing**: regenerate
  `docs/4_agg_language_reference/ref_include.rst` via
  `aggregate.parser.grammar(add_to_doc=True)` (helper near `parser.py:2439`).
- **Language reference prose** (`docs/4_dec_Language_Reference.rst`, plus
  wherever the operator set is described; grep the docs for `exp(` to find
  the spots, `2_aggregate_overview/underwriter.rst` and `features.rst` are
  candidates): state the operator set, the precedence ladder, and the paren
  gate, then the three documented behaviors:
  1. A minus glued to a literal is part of the literal and binds tighter than
     `**`: `(-.4**2)` is +0.16. Negate a computed value with `(-1 * x)` or
     `(0 - x)`.
  2. Parentheses hold one expression, never a list: `(1 -2)` is -1. Brackets
     remain the list syntax and are unchanged: `[1 -2]` is the two-element
     list.
  3. A percentage literal keeps its `%` reading only when used literally;
     any arithmetic degrades it to a plain number, which matters in the `po`
     placement position.
- Docs are edited in lockstep but **not rebuilt** in the verification cycle
  (house rule); note the pending rebuild in the CHANGELOG entry.

## 9. Ripples and the oversight record

- **API / `decl-keywords.json` (oversight agreement 6): nothing owed.** No
  new keywords or terminals exist for the editor mirror to pick up; `+ - * ( )`
  are punctuation to the highlighter. State this explicitly in the CHANGELOG
  entry so the oversight loop can close the check without a round note.
- The app gains the syntax transparently through the same parser; completion,
  highlighting and the wire contracts do not move. No round-note item.
- `dev/FEATURES.csv` tracks class capabilities, not grammar syntax; expect no
  regen. If `dev/regen_features.py` disagrees, read why before committing.

## 10. Cadence and acceptance

One version bump, next `1.0.0a*` (read `pyproject.toml` at execution time; do
not trust any remembered number). The bump commit is one coherent unit per
house rules: grammar + transformer + tests + snapshot regen + grammar-listing
regen + docs prose + CHANGELOG section + `dev/TODO.md` line + this plan moved
to `dev/done/`. Commit subject, one line:

```
[DecL-Paren-Arithmetic] aNNN: full arithmetic inside parentheses in DecL expressions
```

CHANGELOG section sketch (the real description lives there):

- DecL: `+`, `-`, `*` join `/`, `**`, `^`, `exp` inside parentheses, in every
  numeric slot (exposures, FYI premium, layers, freq parameters, reins
  clauses, collars, ranges). Bare expressions unchanged; parentheses required
  for the new operators by design.
- Round trip: expressions evaluate at parse time; canonical text shows the
  evaluated literal (existing "canonical, not verbatim" contract; zero writer
  changes).
- Documented: the glued-minus binding quirk and its `(-1 * x)` / `(0 - x)`
  idioms; parens are one expression, never a list; percent degrades under
  arithmetic.
- No new terminals or keywords; API `decl-keywords.json` check: nothing owed.

Acceptance criteria, measured:

1. Snapshot diff on the pre-existing corpus is empty (T1).
2. The target example in section 1 builds, with `en = 10`,
   `exp_premium = 133333.33333333334`, and `sev_scale = exp(-0.08)`.
3. All T2 programs parse unambiguously; all T4 programs fail to parse.
4. `test_grammar_sync.py` green with zero mirror edits.
5. Unparser fixed point holds on the new cases (T5).
6. Full suite green at tier 2; tier 3 at the bump.
