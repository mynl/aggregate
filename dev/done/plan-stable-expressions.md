---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: named stable and tweedie statements, and arithmetic on stable references

`[Stable-Expressions]`. Drafted 2026-10-30 against aggregate 1.40.0. **Status: done, 1.42.0
and 1.43.0 with aggregate-api 1.39.0; see the Execution log.** Two minor bumps in aggregate, `[Named-Conversion-Statements]` and
`[Stable-Arithmetic]`, and one in aggregate-api, `[aLL-Named-Laws]`. Versions are
assigned at execution, because `dev/plan-severity-parameter-count.md` may land first and
a 1.41.0 was in progress in this worktree when the plan was drafted.

## Goal

The class `aggregate.stable.Stable` computes the parameters of combinations of stable
laws: sums of independent laws with a common $\alpha$, scaling, and shifting. DecL
exposes only its constructor, so a reader of the language cannot reach the algebra that
is the reason for the class. After this plan a DecL session can do the following, one
statement at a time:

```
stable x0 1.5 0
stable x1 1.5 0.85
stable x2 1.5 -0.5

3 * stable.x0                    # Stable(alpha=1.5, beta=0, scale=3, loc=0)
stable.x1 + stable.x2            # Stable(alpha=1.5, beta=0.175, scale=1.5874010519681994, loc=0)
stable.x0 + 2 * stable.x1 + 3    # Stable(alpha=1.5, beta=0.6279767062808198, scale=2.4472608147714756, loc=3)

stable z stable.x0 + 2 * stable.x1 + 3    # the same combination, stored as stable.z
```

The right-hand comments are the values the Python algebra returns today (measured
2026-10-30), and they become the acceptance values for the DecL forms. The statement
`tweedie` gains the same name, label and reference, and no arithmetic.

## Rulings

Author, 2026-10-30:

1. **No variables.** `x0 = stable 1.5 0` is rejected. The existing statement head
   declares the name: `stable x0 1.5 0`.
2. **References stay dotted**: `stable.x0`, as for `agg.X` and `sev.X`.
3. **`tweedie` is treated analogously for the name and label.**
4. **aLL stays one statement per build.** It is an educational device, and a buffer of
   several statements is out of scope.
5. **The space around a subtraction is accepted**, with a hint in the error message.
6. **A `chfagg` does not take these references yet.** Recorded as
   `[Chfagg-From-Stable-Expression]` in `dev/TODO.md` and pended, though logically it
   should follow.

Proposed in drafting, confirmed by the author at review, 2026-10-30; these are rulings:

7. **A combination can be named**: `stable z <expression>`. Without this form a result
   cannot be used in a later statement, and the demand for `=` would return. The stored
   spec holds the four computed numbers, and the expression stays in `program`.
8. **Expressions take references only, never an inline `stable alpha beta`.** See
   `[Inline-Literals-Rejected]` under Alternatives.
9. **Unary minus is included**: `-stable.x0`, since `Stable.__neg__` exists and the
   alternative `-1 * stable.x0` is a needless spelling.
10. **`Stable` gains `__truediv__`**, so `stable.x0 / 3` and `x0 / 3` in Python agree.
11. **`[Stable-Library-Laws]`**: `library.agg` gains the four stable laws with classical
    names. With one statement per build, these give aLL something to combine at once
    (`stable.Holtsmark + 2 * stable.Holtsmark`), and they are the only stable laws an
    Excel client can reference, since that client resolves library entries only. The
    step can be dropped without affecting the rest.
12. **`[Ambiguous-Bare-Name-Message]`**: named stable laws make it likelier that one name
    sits under two kinds, so the misleading parse error for that case is fixed here.

Added by the author at review, 2026-10-30:

13. **`[Stable-Independent-Copies]`**: `t @ stable.x` is in `[Stable-Arithmetic]`, the law
    of the sum of $t$ independent copies, or the Levy process at time $t$. It reuses the
    `@` of `builtin_agg`. With $\varphi^t$ as the characteristic function, the scale is
    $t^{1/\alpha}\gamma$, the location is $t\delta$ and the skewness is unchanged, at every
    $\alpha$ including 1, where a sum has no logarithm term. Refused for $t \le 0$. In
    Python it is `Stable.__rmatmul__`.

## Current behavior, 1.40.0

All measured 2026-10-30.

- **The conversion statements.** `stable alpha beta [scale [loc]] [s0|s1]`,
  `stable <count> claims sev <pareto or lomax>`, `tweedie p mean dispersion`,
  `tweedie p mean cv c` and `tweedie <count> claims sev <gamma>` are alternatives of
  `conversion_out` in `src/aggregate/decl.lark`. Each returns
  `("conversion", family, object)`. `ANSWER_KINDS` in `src/aggregate/underwriter.py`
  holds `'expr'` and `'conversion'`, and `_interpret_program` does not store those, so a
  conversion has no name, no trailer and no entry in the recipe base.
- **The name slot is free.** `stable x0 1.5 0` is a parse error at `x0`, and so is
  `tweedie x0 1.5 10 2`. Ruling 8 of `dev/done/plan-stable-statement.md` kept this slot
  unused on purpose. The present plan assigns it, which that ruling allows.
- **References.** A dotted reference is a priority 3 terminal (`BUILTIN_AGG`,
  `BUILTIN_SEV`, `BUILTIN_DIST`, `BUILTIN_PORT`, `BUILTIN_CHFAGG`), resolved in the
  transformer by `Underwriter._safe_lookup`, which returns a deep copy of the stored
  spec. Inside a statement the dotted form is always required: `2 * ProbeA` and
  `agg X ProbeA` are parse errors even for a name unique across kinds.
- **A bare name works only as a whole program.** `build('ProbeA')` looks the name up
  first (`_build_work`). For a name under two kinds, `recipe` raises `KeyError`
  (ambiguous), `_build_work` catches `LookupError` and falls through to the parser, and
  the reader sees `Unexpected 'ProbeB'`. `preview` has the same branch.
- **The minus sign in a reference.** `-` is a legal name character, so
  `agg.ProbeA-1` lexes as one reference and fails with
  `no recipe named 'ProbeA-1' of kind 'agg'`. The forms `agg.ProbeA - 1` and
  `agg.ProbeA -1` work.
- **Statements.** A single newline is whitespace. Statements are separated by a blank
  line or a line-final `;`. `build()` refuses a program with several statements
  (`build() expects a single output, got 3; use build_many() for batched programs.`).
- **aLL.** `collapse_program` in `V:/dev/aggregate-api/src/aggregate_api/routes/objects.py`
  flattens a buffer to one line, so a build is one statement. `sessions.py` keeps one
  recipe base per session, a fork of the library's. `_is_conversion` tests
  `statements[-1].kind == "conversion"` and returns a `kind: "text"` answer holding
  `str(obj)`; anything else not in `SUPPORTED_KINDS` is refused.
- **`TweedieConversion` has `__slots__ = ()`**, so an instance accepts no attribute,
  including `program`. `Stable` has no name or label.

## Mathematics

**Stable laws, in Nolan's S1 parameterization** [@Nolan2020, Prop. 1.3 and 1.4]. For
independent $X_j \sim S(\alpha, \beta_j, \gamma_j, \delta_j; 1)$, the sum has
$\gamma^\alpha = \gamma_1^\alpha + \gamma_2^\alpha$,
$\beta = (\beta_1\gamma_1^\alpha + \beta_2\gamma_2^\alpha)/\gamma^\alpha$ and
$\delta = \delta_1 + \delta_2$. For $a \ne 0$, $aX$ has skewness
$\mathrm{sign}(a)\beta$, scale $|a|\gamma$ and location $a\delta_1$, less
$(2/\pi)\beta\gamma a\log|a|$ at $\alpha = 1$. `Stable` implements these rules and
`tests/test_stable.py` pins them against the characteristic function.

**The operations act on laws, not on random variables.** Each occurrence of a reference
in an expression is a fresh independent copy. Hence `stable.x + stable.x` is not
`2 * stable.x`: for $\alpha = 3/2$ the first has scale $2^{2/3} = 1.5874$ and the second
has scale 2, and that difference is the stability property itself. Likewise
`stable.x - stable.x` is a symmetric stable law with scale 1.5874, not zero. The user
documentation has to state this plainly, and the notation is one reason for keeping the
dotted form (see `[Variables-Rejected]`).

**Combining limit laws of two units.** With `A` the stable limit of 1000 claims with
severity `100 * lomax 1.5` and `B` that of 400 claims with severity `250 * lomax 1.5`,
`A + B` has $\beta = 1$, mean 400,000 and scale 34721.690825521684, and its right tail
constant is $\sum\lambda\theta^\alpha = 2{,}581{,}138.83$ (measured to 13 figures), which
gives a closed-form oracle. `A - B` has $\beta = -0.22514822655441366$ and location 0.

**Tweedie laws are not closed in the same way** [@Jorgensen1997]. Write
$Y \sim Tw_p(\mu, \phi)$ with $1 < p < 2$. It is compound Poisson with count
$\lambda = \mu^{2-p}/((2-p)\phi)$ and a gamma severity of shape $(2-p)/(p-1)$ and scale
$\theta = \phi(p-1)\mu^{p-1}$. Scaling is always closed, $cY \sim Tw_p(c\mu, c^{2-p}\phi)$,
because $\theta$ scales by $c$ and $\lambda$ is unchanged. A sum of independent laws is
Tweedie only when both have the same gamma severity, which means the same $p$ and the
same $\phi\mu^{p-1}$. A shift leaves the family. Check: `tweedie 1.5 10 2` and
`tweedie 1.5 40 1` have $\phi\mu^{1/2} = 6.3246$ and both have gamma mean 3.16228. Since
$+$ would be refused for almost every pair of laws, Tweedie gets no arithmetic here.

## Alternatives considered

**`[Variables-Rejected]` bare variables and `=`.** Three independent reasons. First, the
prefix is what keeps the parse context free: `agg.X + 3` shifts a severity, `stable.x + 3`
shifts a location, and the terminal selects the production before any lookup. A bare
`x + 3` would need the symbol table during the parse, which is the typedef problem in C.
`ID` also already appears in about 39 lines of the grammar for family words, approximation
kinds and `s0`, so `stable 1.5 0 s0` would become a question. Second, the notation
misleads: `x + x` looks like $2x$ in every programming language, whereas `stable.x + stable.x`
looks like two draws from a named law, which is the correct reading. Third, `x0 = stable 1.5 0`
adds no capability over `stable x0 1.5 0`, and it invites `alpha = 1.5`, reassignment,
scoping and dependency order, none of which a declarative language wants or the writer
can round-trip. Ruling 7 gives what `=` would have given. Recorded in
`dev/done/plans-considered-and-rejected.md`.

**`[Inline-Literals-Rejected]` `3 * stable 1.5 0`.** Tempting because a `chfagg` already
writes `c * stable alpha beta + b`. Measured on a scratch grammar, admitting
`STABLE numberl` as an atom makes `stable 1.5 -0.5` ambiguous (a skewness of $-0.5$, or a
shift of a one-parameter law) and likewise `stable 1.5 0.5 -3`, because `NUMBER` absorbs a
glued leading minus. The first of those is an ordinary statement, so the cost is too high.
A law has to be named to enter arithmetic.

**`[Stable-Kind-As-Conversion-Rejected]` storing under the kind `conversion`.** One new
kind would cover both families, but `ANSWER_KINDS` means "never stored", and `stable.x`
and `tweedie.x` need `(kind, name)` keys that match their prefixes. `_safe_lookup` also
checks that the entry's kind matches the prefix.

## Design

### [Grammar] `src/aggregate/decl.lark`

These rules were tested on a scratch copy of the grammar (2026-10-30): 44 statements
covering every form in this plan and the existing ones parsed with no `_ambig` node under
`ambiguity='explicit'`, and all 1,165 statements in the shipped `.agg` files gave trees
identical to the current grammar, with the same 3 entries in `KNOWN_AMBIGUOUS`.

```
stable_out: STABLE name as_label numberl trailer                 -> stable_out_numbers
          | STABLE name as_label numberl ID trailer              -> stable_out_param
          | STABLE name as_label expr CLAIMS SEV sev trailer     -> stable_out_compound
          | STABLE name as_label stable_sum trailer              -> stable_out_expression

tweedie_out: TWEEDIE name as_label expr expr expr trailer        -> tweedie_out_dispersion
           | TWEEDIE name as_label expr expr CV expr trailer     -> tweedie_out_cv
           | TWEEDIE name as_label expr CLAIMS SEV sev trailer   -> tweedie_out_compound

?stable_sum: stable_sum PLUS stable_term     -> stable_add
           | stable_sum MINUS stable_term    -> stable_subtract
           | stable_sum PLUS expr            -> stable_shift_up
           | stable_sum MINUS expr           -> stable_shift_down
           | expr PLUS stable_term           -> stable_number_plus
           | expr MINUS stable_term          -> stable_number_minus
           | stable_term

?stable_term: expr TIMES stable_term         -> stable_scaled
            | stable_factor

?stable_factor: stable_factor TIMES expr     -> stable_times
              | stable_factor DIVIDE expr    -> stable_divide
              | MINUS stable_atom            -> stable_negate
              | stable_atom

?stable_atom: BUILTIN_STABLE                 -> stable_ref
            | "(" stable_sum ")"             -> stable_parens
```

The scratch test used plain rules, and the `?` prefixes above (the form used by `?term`
and `?factor`) change only the tree shape. Rerun the ambiguity check after adding them.

- `answer` gains `stable_out -> answer_stable`, `tweedie_out -> answer_tweedie`,
  `stable_sum -> answer_stable_sum` and `BUILTIN_TWEEDIE -> answer_tweedie_ref`. In the
  first bump, where `stable_sum` does not exist yet, the alternative is
  `BUILTIN_STABLE -> answer_stable_ref`, and the second bump replaces it.
- New terminals beside the others: `BUILTIN_STABLE.3: /stable\.[a-zA-Z][a-zA-Z0-9._:~\-]*/`
  and the same for `BUILTIN_TWEEDIE`.
- The `ID` terminal's first lookahead gains `stable\.|tweedie\.`, as `chfagg\.` has.
- The comment over `conversion_out` describes a conversion as having no name and no
  trailer. That description remains true of `conversion_out`, and the comment gains a
  sentence pointing at `stable_out` and `tweedie_out`.
- Expressions accept what `expr` accepts on the numeric side, so `1/2 * stable.x0` and
  `(1/2) * stable.x0` work, and `(1+2) * stable.x0` needs its parentheses, as everywhere
  in DecL.

`conversion_stable` and the other conversion alternatives are untouched, and a stable
literal never appears in `stable_atom`.

### [Stored-Specs] the two new kinds

`stable` and `tweedie` become recipe kinds, keyed `('stable', 'x0')` and `('tweedie', 't0')`.
The stored spec is canonical and numeric, whichever form was written:

```python
{"name": "x0", "alpha": 1.5, "beta": 0.0, "scale": 1.0, "loc": 0.0,   # S1, always
 **as_label, **trailer}
{"name": "t0", "p": 1.5, "mean": 10.0, "dispersion": 2.0, **as_label, **trailer}
```

So `stable x0 1.5 1 2 -2 s0`, `stable b 1000 claims sev 100 * lomax 1.5` and
`stable z stable.x0 + 1` all store four S1 numbers, and `program` keeps what was written.
A reference resolves at parse time, as every dotted reference other than `sev agg.NAME`
does: redefining `x0` later does not change a stored `z`. The law is validated in the
transformer, so a bad parameter fails at parse, with the messages `Stable.__init__` and
`_check_tweedie` raise today.

### [Named-Objects] `Stable` and `TweedieConversion`

Both become `LabeledMixin` hosts (`src/aggregate/_labeled.py`), so `name`, `label` and
`_title_name` behave as on every other DecL object. The module `_labeled.py` imports only
`types`, so `stable.py` stays effectively a leaf; update its module docstring.

- `Stable.__init__` gains keyword arguments `name=None, label=None`, after the existing
  five, and calls `self._init_labels(label=label)`. New class attribute `program = ''`.
- `Stable.from_spec(spec)`, classmethod, picks `alpha`, `beta`, `scale`, `loc`, `name`
  and `label` and ignores the trailer keys. Used by `_factory` and by the `stable_ref`
  callback.
- `Stable._label_default` returns the short spelling, `stable 1.5 0 1 0`, because the
  resolved label is never blank and an anonymous law has no handle.
- **Arithmetic returns an anonymous law.** The operators already build a new `Stable`
  from four numbers. A lone `stable.x0` keeps its name and label, since `from_spec`
  supplies them. `stable z stable.x0` stores a copy under `z`.
- `Stable.__repr__` adds `name=` and `label=` when set, so it remains constructor-valid.
  `Stable.to_decl` includes the name and label when set, and is unchanged otherwise.
- `Stable.__str__` is unchanged for an anonymous law (`test_statement_string_worked_example`
  pins it). A named law leads with `_title_name`:
  `x0: stable 1.5 0 1 0 = stable 1.5 0 1 0 s0, mean 0, Pr(X>mean) 0.5, Pr(X>x) ~ 0.199471 x^-1.5, Pr(X<-x) ~ 0.199471 x^-1.5`.
  Each spelling after the colon still builds back. With a label the prefix is
  `My law (x0): `.
- `TweedieConversion` drops `__slots__ = ()`, mixes in `LabeledMixin`, and gains
  `__new__`, which sets `self.name = None` and calls `self._init_labels()`; `from_spec(spec)`,
  which calls `from_dispersion` and then sets the name and label; `_label_default`; class
  attribute `program = ''`; and the same `_title_name` prefix in `__str__`.
- `Stable.__truediv__(self, other)` is `self * (1 / other)` for a real nonzero `other`,
  and division by zero raises the existing point mass `ValueError`.

Measured with a throwaway namedtuple (2026-10-30): the mixin and a `__new__` work on a
namedtuple subclass, and tuple unpacking, `len`, `deepcopy` and `pickle` all survive with
the name and label kept. Equality stays over the tuple's fields, so it ignores the name,
which is the right behavior: two names for one law compare equal. **One accepted gap:**
`_make` and `_replace` bypass `__new__`, so `label` on such a result raises
`AttributeError`. Nothing in `src/aggregate` calls either on this class.

### [Underwriter] `src/aggregate/underwriter.py`

- `_factory` gains branches for `'stable'` and `'tweedie'`: `from_spec`, then
  `obj.program = program`.
- `build_many`'s `no_update` tuple gains `Stable` and `TweedieConversion`, as its comment
  directs. `_build_work`'s bare-name branch calls `.update` unconditionally when `update`
  is true, but its only caller passes `update=False`, so no guard is needed.
- `_KIND_WRITE_ORDER` gains `'stable': 0, 'tweedie': 0`. Their canonical text contains
  only numbers, so they depend on nothing.
- Docstrings that enumerate the kinds: the class docstring of `Underwriter`,
  `add_recipe`, and `Recipe.kind` in `recipe.py`; `UnderwritingParser.parse` in `parser.py`.

### [Reference-Messages] the two error messages

**`[Minus-In-Reference-Hint]`.** In `_safe_lookup`, when the lookup raises `RecipeNotFound`
and the name contains `-`, re-raise a `RecipeNotFound` with the same `kind` and `name` and
this message:

```
no recipe named 'x1-stable.x2' of kind 'stable'. A name may contain a minus sign, so
stable.x1-stable.x2 is one name. To subtract, write spaces: stable.x1 - stable.x2
```

The suggestion splits at the first `-`. It fires for every kind, since every reference
form passes through `_safe_lookup`, and a dashed name that exists is unaffected. It
reaches aLL with no change there, since `_missing_entry_detail` uses `exc.message` (unless
`defining_database` finds the name in another file, which is the more useful message).

**`[Ambiguous-Bare-Name-Message]`.** `_build_work` and `preview` change
`except (LookupError, TypeError)` to `except (RecipeNotFound, TypeError)`, so the ambiguous
`KeyError` propagates instead of falling through to the parser. A program cannot be an
ambiguous name unless it is one bare word, so nothing else is affected. The message becomes:

```
'x0' is ambiguous across kinds (agg, stable); write agg.x0 or stable.x0, or pass kind=
to choose. Shipped library names are unique across kinds.
```

`recipe()` builds this message, so `uw['x0']` improves too. aggregate-api's build route
already catches `KeyError` and serves `args[0]` as a 422; confirm that its `_preview` call
sits inside a handler that does the same.

### [Writer] `src/aggregate/decl_writer.py`

`_render_stable` and `_render_tweedie`, on the model of `_render_chfagg`, added to
`_KIND_RENDERERS`:

```python
_Block(f'stable {name}{_render_label(spec.get("label"))}',
       [' '.join(_fmt_num(spec[k]) for k in ('alpha', 'beta', 'scale', 'loc')),
        *_render_trailer(spec, trailer)])
```

`_fmt_num` uses `repr`, so the text re-parses to the same floats, and a negative location
renders glued (`stable x 1.5 1 2 -3`), which `numberl` takes as the number.

### [Mirrors] the keyword lists

`stable` and `tweedie` are already reserved, so `decl_words` and the reserved-word
alternation are unchanged. `parser_errors._TERMINAL_LABELS` gains
`"BUILTIN_STABLE": "a builtin stable law (stable.X)"` and
`"BUILTIN_TWEEDIE": "a builtin tweedie law (tweedie.X)"`. `agg.sublime-syntax` gains the
two prefixes at the lines holding `chfagg\.` (106, 109 and 161 today).
`tests/test_grammar_sync.py` holds both mirrors to the grammar, and it tokenizes every
`.agg` file, so it also covers `decl_pygments.py`.

### [Stable-Library-Laws] `src/aggregate/agg/library.agg`

Four entries in the "Numerical character" section, beside `chfagg StableLaw`, with
`tags{topic:numerics, role:intermediate}` as it has, and each with a `note{}`:

```
stable Cauchy 1 0        # the symmetric law at alpha = 1
stable Levy 0.5 1        # the first passage time of Brownian motion
stable Landau 1 1        # energy loss in physics; totally skewed at alpha = 1
stable Holtsmark 1.5 0   # the gravitational field of random stars
```

`library.agg` has no entry under these names (checked 2026-10-30), and
`tests/test_agg_libraries.py` enforces uniqueness across kinds. These entries enter
`tests/test_library_entries.py::test_every_library_entry_builds`, which needs no edit:
`_check_built` asserts the object exists and audits only the kinds in `VALIDATED_KINDS`,
`agg` and `port`. They also enter the Examples menu in aLL, which is why
`[aLL-Named-Laws]` follows the first bump directly.

## Steps

### [Named-Conversion-Statements], one minor bump

1. `[Grammar]`: `stable_out` without the expression alternative, `tweedie_out`, the two
   terminals, the `ID` lookahead, and whole-statement `stable.NAME` and `tweedie.NAME`.
2. `[Named-Objects]`, `[Stored-Specs]`, `[Underwriter]`, `[Writer]`, `[Mirrors]`, except
   `__truediv__`.
3. `[Ambiguous-Bare-Name-Message]`.
4. `[Stable-Library-Laws]`.
5. Tests below. Corpus: named statements of both kinds in `_test_suite.agg` and
   `decl-testers.agg`, then `uv run python tests/capture_spec_snapshot.py`, and read the
   diff, which should show additions only. `tests/test_decl_parser.py::VALID_KINDS` and
   `tests/test_grammar_ambiguity.py::_NAME_RE` gain the two kinds. A conversion stays out
   of the corpus, since its spec is an object and not JSON.
6. Docs: in `docs/4_dec_Language_Reference.rst`, retitle "Parameter Conversions (tweedie,
   stable)" to "Parameter Conversions and Named Laws (tweedie, stable)", and revise its
   opening sentence on a conversion having no name. Add the named forms.
   Regenerate `docs/4_agg_language_reference/ref_include.rst` with
   `parser.grammar(add_to_doc=True)`. Keep the `.rst` in lockstep and do not build.
7. Release: `__version__`, one `CHANGELOG.md` paragraph (the prefixes `stable.` and
   `tweedie.` can no longer begin a name), `dev/FEATURES.csv` via `dev/regen_features.py`,
   and move this plan to `dev/done/` in this commit. Commit subject:
   `[Named-Conversion-Statements] X.Y.0: stable NAME and tweedie NAME store a law, referenced as stable.NAME and tweedie.NAME`.

### [aLL-Named-Laws], one bump in aggregate-api, directly after the first library bump

In `V:/dev/aggregate-api`, whose working tree held uncommitted edits on 2026-10-30
(`dev/TODO.md`, `tests/conftest.py`, `tests/test_capability.py`, `tests/test_objects.py`).
Check `git status` first and commit only this step's own hunks, or stop and ask.

1. `_is_conversion` in `src/aggregate_api/routes/objects.py` tests
   `kind in ("conversion", "stable", "tweedie")`, with its docstring and the comment at
   the call site updated to match. Under an older library no statement has either kind,
   so the edit is harmless, and the `aggregate>=1.11.0,<2` floor stays. `SUPPORTED_KINDS`
   stays unchanged, since no object manifest is served for these.
2. Confirm that a named statement registers in the session's fork through the build path,
   since `_interpret_program` stores it and a text answer is never cached. `_register`
   runs only on a cache hit, so it is not reached. It skips only `expr`, and if it were
   ever reached with a conversion the fork's `recipes` frame would raise `AttributeError`
   (measured). Change its test to `statement.kind in ANSWER_KINDS` while here.
3. `tests/test_examples.py`: `heads` gains `stable` and `tweedie`. That test reddens when
   the library adds entries of a new head, as it did at 1.30.0.
4. `tests/test_objects.py`, beside the conversion test at line 472: `stable x0 1.5 0`
   returns text and later `stable.x0 + 1` in the same session returns text; another
   session gets the 422 `recipe_not_found`; `tweedie t0 1.5 10 2` returns text;
   `stable.Holtsmark + 1` works under a client limited to library references, and
   `stable.x0` is refused there with `session_reference`.
5. Confirm which `aggregate` the service's environment resolves before running these,
   since the tests need this repository's first bump. Bump, CHANGELOG, commit and push
   under that repository's rules.

### [Stable-Arithmetic], one minor bump

1. `[Grammar]`: the `stable_sum` family, `stable_out_expression`, and `answer` taking
   `stable_sum` in place of `BUILTIN_STABLE`.
2. Transformer callbacks, each applying `Stable`'s own operator, with
   `answer_stable_sum` returning `("conversion", "stable", law)`. `stable_ref` is
   `Stable.from_spec(self.safe_lookup(str(c[0])))`, and `BUILTIN_STABLE` gets a terminal
   method beside `BUILTIN_SEV`.
3. `Stable.__truediv__`, and a `Notes` paragraph in the class docstring on independent
   copies, with the $2^{2/3}$ against 2 example.
4. `[Minus-In-Reference-Hint]`.
5. Tests below, and an unambiguity test: a parser with `ambiguity='explicit'` over the
   list of new forms finds no `_ambig`.
6. Docs: a subsection of the reference on arithmetic with the three worked expressions,
   the `A + B` example, the warning on independent copies (`stable.x + stable.x` is not
   `2 * stable.x`), the space around a subtraction, and the requirement of a common
   $\alpha$. `src/aggregate/data/decl-help/card.html`: `stable x 1.5 0.85` and
   `stable.x + 2 * stable.y + 3` in the same box as the two existing lines. Regenerate
   `ref_include.rst`.
7. Release: `__version__`, one `CHANGELOG.md` paragraph, `dev/FEATURES.csv`, and a note in
   the moved plan's execution log. Commit subject:
   `[Stable-Arithmetic] X.Y.0: stable.NAME references take +, -, * and /, and stable NAME <expression> stores the result`.

## Files

| Area | Files |
|---|---|
| Grammar | `src/aggregate/decl.lark` |
| Parser | `src/aggregate/parser.py`, `src/aggregate/parser_errors.py` |
| Classes | `src/aggregate/stable.py`, `src/aggregate/tweedie.py` |
| Recipe base | `src/aggregate/underwriter.py`, `src/aggregate/recipe.py` |
| Writer | `src/aggregate/decl_writer.py` |
| Mirrors | `agg.sublime-syntax` |
| Corpora | `src/aggregate/agg/library.agg`, `_test_suite.agg`, `decl-testers.agg` |
| Help | `src/aggregate/data/decl-help/card.html` |
| Tests | `tests/test_stable.py`, `tests/test_tweedie.py`, `tests/test_decl_parser.py`, `tests/test_grammar_ambiguity.py`, `tests/test_underwriter.py`, `tests/data/expected_specs.json` |
| Documentation | `docs/4_dec_Language_Reference.rst`, `docs/4_agg_language_reference/ref_include.rst` |
| Release | `src/aggregate/__init__.py`, `CHANGELOG.md`, `dev/FEATURES.csv`, `dev/TODO.md` |
| aLL | `V:/dev/aggregate-api`: `src/aggregate_api/routes/objects.py`, `tests/test_examples.py`, `tests/test_objects.py`, release files |

## Names checked

On 2026-10-30 in `src/aggregate`: `Stable` has no `name`, `label`, `program`, `from_spec`
or `__truediv__`. `TweedieConversion` has the fields `p`, `mean`, `dispersion`, `cv`,
`frequency`, `sev_m`, `sev_cv` and `p0`, none of which collides with the new attributes or
with the mixin's `label`, `labels`, `label_map`, `renamer` and `use_labels`. `decl.lark`
has no rule or terminal `stable_out`, `tweedie_out`, `stable_sum`, `stable_term`,
`stable_factor`, `stable_atom`, `BUILTIN_STABLE` or `BUILTIN_TWEEDIE`. The `.agg` corpora
have no name beginning `stable.` or `tweedie.`. The kinds `'stable'` and `'tweedie'` are
new as recipe kinds; `"stable"` and `"tweedie"` also appear as the family value in a
`chfagg` spec and as the `name` of a conversion answer, which are different fields.
Recheck at execution.

## Acceptance and tests

**Named statements**, in `tests/test_stable.py` and `tests/test_tweedie.py`:

- Every form stores under `(kind, name)`: `stable x0 1.5 0`, with scale and location, with
  `s0`, with `as "My law"`, with a trailer, and the compound form; the three `tweedie`
  forms. Each returned object has the right `name`, `label` and `program`.
- `stable x0 1.5 1 2 -2 s0` stores the S1 location, and agrees with the anonymous form.
- `build('x0')`, `build('stable.x0')` and `tweedie.t0` return the named object; each
  parameter agrees to `rtol=1e-15` after a store and rebuild.
- **Unparser fixed point**: `spec_to_decl` of each stored spec re-parses to the same spec,
  and `to_agg` then `load` round-trips into a fresh fork.
- Anonymous statements are unchanged: `str` of `stable 1.5 1 2 0` is byte-identical,
  nothing is stored, and the kind is `conversion`. `tests/test_stable.py::test_statement_stores_nothing`
  and `test_statement_string_worked_example` still pass, unedited.
- `TweedieConversion` remains an 8-tuple: unpacking, `==` and `_fields` are unchanged, and
  `TweedieConversion.from_dispersion(*a.as_tweedie())` still works.
- `2 * tweedie.t0` is a parse error. So are `stable.x0 + tweedie.t0` and `3 * stable 1.5 0`.
- A name in a fork is not visible in its parent.
- `preview('stable.x0 + 1')` reports `ResolvedReference('stable', 'x0', ...)`, and
  registers nothing.
- Ambiguous bare name: with `agg x` and `stable x` held, `build('x')` and `preview('x')`
  raise `KeyError` with the new message, and no parse error appears.

**Arithmetic**, in `tests/test_stable.py`, with exact values from the Goal:

- The three expressions in the Goal return those parameters to `rtol=1e-14`, and agree
  with the Python operators on `Stable`.
- Every operator form against Python: `stable.x * 3`, `stable.x / 3`, `3 + stable.x`,
  `3 - stable.x`, `stable.x - 3`, `stable.x -3`, `-stable.x`, `stable.x - -stable.y`,
  `2 * 3 * stable.x`, `2 * stable.x * 3`, `2 * (stable.x + stable.y) - 1`,
  `(1/2) * stable.x`, `1/2 * stable.x`.
- **Characteristic function oracle** at $\alpha \in \{0.7, 1, 1.5\}$: a DecL combination
  has $\varphi$ equal to the product of the terms' $\varphi$ at scaled arguments, to
  `rtol=1e-12`, which pins the $\alpha = 1$ logarithm through the DecL path.
- **Independent copies**: `stable.x + stable.x` has scale $2^{1/\alpha}$ and `2 * stable.x`
  has scale 2; `stable.x - stable.x` has location 0 and $\beta = 0$.
- **Limit law oracle**: with `A` and `B` from Mathematics, `stable.A + stable.B` has
  $\beta = 1$, mean 400,000, and `tail_constant` equal to $\sum\lambda\theta^\alpha$ to
  `rtol=1e-12`.
- A result is anonymous, is not stored, and has kind `conversion`.
- `stable z stable.x0 + 2 * stable.x1 + 3` stores four numbers with `program` as written;
  redefining `x0` leaves `z` unchanged; `stable z as Combo stable.x - stable.y note{net}`
  keeps its label and note.
- Refusals with stated messages: differing $\alpha$, `0 * stable.x`, `stable.x / 0`, and
  `stable.nope` (`RecipeNotFound`).
- **Minus hint**: `stable.x1-stable.x2` and `agg.A-1` raise `RecipeNotFound` whose message
  contains the spaced spelling, with `kind` and `name` unchanged; a real name with a dash
  still resolves.

## Release gates

Edit loop: `tests/test_stable.py tests/test_tweedie.py tests/test_decl_parser.py
tests/test_decl_unparser.py tests/test_grammar_ambiguity.py tests/test_grammar_sync.py
tests/test_underwriter.py tests/test_parser_errors_integration.py tests/test_decl_words.py
tests/test_library_entries.py tests/test_agg_libraries.py`. At each bump, with
`UV_PROJECT_ENVIRONMENT=.venv`: the full suite, `uv run pytest -m 'slow or not slow'`.
`KNOWN_AMBIGUOUS` must not change. Docs are not built.

## Deferred

- **`[Chfagg-From-Stable-Expression]`**: `chfagg C stable.x0 + 2 * stable.x1 + 3`, which
  puts a combination's parameters on a grid and closes the loop from parameters to
  density. Pended by the author, in `dev/TODO.md`.
- Arithmetic on Tweedie laws, for scaling and for sums with equal $\phi\mu^{p-1}$.
- A sum with differing $\alpha$, which is not stable but has a closed-form characteristic
  function, and is therefore a `chfagg` question.
- `[Stable-Conversion-Chfagg-Spelling]` in `dev/TODO.md` is unaffected.
- A `tweedie` reference inside an `agg` body, or `tweedie z tweedie.t0` as a rename.
- Several statements in one aLL buffer (ruling 4).

## Execution log

Reviewed 2026-10-30 at 1.41.0, so the bumps are 1.42.0, aggregate-api 1.39.0 and 1.43.0. On a
scratch copy of the grammar, 65 statements have no `_ambig` node except the case below, 13
forms meant to be refused are refused, all 1,165 shipped statements give identical trees with
the same 3 ambiguous entries, and 733 statements parse in the same time within noise.

### Divergences

- **`[Writer]`: a default scale and location are omitted.** The plan's snippet renders all
  four numbers. `tests/test_agg_libraries.py::test_library_is_written_in_the_canonical_layout`
  requires each library entry to equal its canonical rendering, so the planned library text
  `stable Cauchy 1 0` could not stand. `_render_stable` omits a location of 0, and then a scale
  of 1, as `_render_chf_body` omits a unit scale and a zero shift. The spec is unchanged and the
  text re-parses to it.
- **`[Writer]`: the renderers are `_render_stable_law` and `_render_tweedie_law`.** The name
  `_render_tweedie` is already taken in `decl_writer.py`, by the renderer of the `tweedie` clause
  of an `agg`. The plan's name shadowed it and failed 678 tests at once.
- **`Stable.__truediv__`: division by zero has its own message.** The plan reuses the point
  mass error, but $X/0$ is not a point mass. It is still a `ValueError`.
- **`[Ambiguous-Bare-Name-Message]`: dotted spellings only for kinds that have one.** `pnl`,
  `xpnl`, `bvagg` and `nttagg` have no dotted reference, so the message offers `kind=` alone
  for those.
- **`[aLL-Named-Laws]`: `_register` tests the literal `("expr", "conversion")`.** The plan
  imports `ANSWER_KINDS`, which arrived in `aggregate` 1.22.0, and the floor stays 1.11.0.
- **`[Mirrors]`: two lines of `agg.sublime-syntax`, not three.** Only 109 and 161 hold
  `chfagg\.`; line 106 is the `port.` rule.
- **`[Stable-Independent-Copies]`**, ruling 13, is an addition to the plan by the author.

- **`[aLL-Named-Laws]`: the arithmetic tests are gated on the library.** The plan puts
  `stable.x0 + 1` and `stable.Holtsmark + 1` in the aggregate-api step, which runs before
  `[Stable-Arithmetic]` exists. They skip until `Stable.__truediv__` appears, as the conversion
  tests in that file skip before 1.22.0 and 1.23.0, and they were run after 1.43.0 landed.

### Findings fixed at execution

- `_did_you_mean` treated every upper-case terminal as a keyword, so `2 * stable 1 2` drew
  `Did you mean: builtin_stable?`, since `stable` is close enough to that terminal's name. It
  now excludes `BUILTIN_*`, which no one can write. No plan step covered this.

### Accepted findings

- `stable inf 1.5 0` has two parses, since `inf` is both a `NUMBER` and an `ID`. `resolve`
  takes the number, by terminal priority, and $\alpha = \infty$ is refused. A law cannot be
  named `inf` positionally, which is no loss.
- The `execute-plan` skill cites `V:/worktrees/dev-files.md` and
  `web/src/decl-keywords.json`, neither of which exists now, so those steps do not apply.

## References

::: {#refs}
:::
