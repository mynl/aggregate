# Plan: one DecL word list, derived in aggregate and read by every colorizer

`[DecL-Word-List]`. Drafted 2026-10-19 at 1.5.3 (`d85f0c2`) at Steve's request, from the
house-style review (ruling there: "the single word list lives in aggregate"). Not yet ruled.

This is the library half of a set of four plans with the same file name, each stating its own side:

| Repository | Plan | Lands |
|---|---|---|
| aggregate | this file | first, then released to PyPI |
| aggregate-api (aLL) | `V:\dev\aggregate-api\dev\plan-decl-word-list.md` | after this is on PyPI |
| aggregate-asp (aSP) | `V:\dev\aggregate-asp\dev\plan-decl-word-list.md` | after the api's endpoint |
| house-style | `D:\Projects\house-style\dev\plan-decl-word-list.md` | after this is on PyPI |

## Goal

Every program that colors DecL takes its words from one list. That list is derived from
`decl.lark` wherever the grammar holds the words, and kept in one module, tested, wherever it does
not. Adding a keyword to the grammar then colors it everywhere, with no copy to update by hand.

## Current behavior

DecL words are written down in six places. Two of them are in this repository:

| Where | What | Kept |
|---|---|---|
| `decl.lark` | the language: 74 keyword terminals, `FREQ.2` (11 frequency names, line 902), and the `ID` terminal's exclusion alternation (line 955), which is the flat list of all 90 reserved words | the source of truth |
| `decl_pygments.py` `AggLexer` | the Pygments lexer for the docs, `decl_writer._colorize` and the pygments entry point | by hand: "a hand-written mirror of `decl.lark`" (its docstring) |
| `parser_errors._TERMINAL_LABELS` | friendly names for parse errors, and the api's completion fallback | by hand, test-checked |
| `agg.sublime-syntax` | the Sublime Text syntax | by hand, test-checked |
| aggregate-api `web/src/decl-keywords.json` | aLL's editor coloring and offline completion | by hand; its TODO `[Keyword-Mirror-Gaps]` lists 11 missing words |
| aggregate-asp `js/decl.js` `declHTML` | aSP's export box | one inline regex of 33 words |
| house-style `_extensions/house/decl.xml` | Quarto and pandoc coloring | generated from AggLexer's words and aLL's pool |

`tests/test_grammar_sync.py` already holds the extraction: `grammar_reserved_words()` (lines
70 to 84) reads the `ID` exclusion list, and the module checks AggLexer, `_TERMINAL_LABELS` and
`agg.sublime-syntax` against the grammar. Its docstring says aLL's file "lives in a separate
repository and is not checked here", which is the gap.

Two groups of words are not in the grammar at all:

- **Severity distribution names** arrive as a plain `ID` (`sev0: ids numbers ...`, lines 523 to
  527) and are resolved by `getattr(scipy.stats, name)` in `SeverityScipy._build`
  (`_severity.py:2323`), with `dhistogram`, `chistogram` and `cantor` special-cased in
  `_classify_sev`. AggLexer's hand list (`decl_pygments.py:181` to `209`), grouped by parameter
  count, is the only enumeration anywhere.
- **Distortion names** are a plain `ID` too (`distortion_out: DISTORTION name ID numberl`, line
  45). The registry knows them: `Distortion.available_distortions(pricing=False)`
  (`spectral.py:905`) returns every documented kind: `ccoc ph wang dual tvar bitvar wtdtvar beta
  cll clin lep ly`. AggLexer has no distortion names. `dev/done/plan-colorizer-resync.md` left
  them out "to avoid a third hand-maintained vocabulary"; a derived list does not have that cost.

## The change

### [DecL-Word-List] one version, the next free minor (1.6.0 if nothing lands first)

- **A new module, `src/aggregate/decl_words.py`, with one public function, `decl_words()`.** It
  returns a dict of sorted tuples, cached after the first call:

  | Group | Source |
  |---|---|
  | `reserved` | `decl.lark`, the `ID` exclusion alternation: all 90 reserved words |
  | `frequencies` | `decl.lark`, the `FREQ.2` alternation |
  | `clauses` | `decl.lark`, the `word\{` terminals: `note`, `hints`, `tags` (the test helper `grammar_brace_clauses()`, moved likewise) |
  | `constants` | `decl.lark`, the word in `NUMBER.2`: `inf` |
  | `declarations` | `dfreq dsev dbvsev dwait nttagg`, checked to be reserved words |
  | `severities` | a dict of name to parameter count, the table moved out of AggLexer, with `dhistogram`, `chistogram` and `cantor` |
  | `mixing` | the mixing distributions AggLexer accepts after `mixed` (`decl_pygments.py:147`), moved likewise |
  | `distortions` | `Distortion.available_distortions(pricing=False)` |

  `grammar_reserved_words()` moves from the test file into this module as the private helper
  that reads `decl.lark` (already package data), and the tests import it from there. The
  module's docstring Notes say which groups come from the grammar and which are kept here, and why.
- **One word is in two groups: `beta`** is a scipy severity, a mixing distribution and a
  distortion kind (checked against the registry on 2026-10-19). A colorizer reading words without
  their grammatical context cannot tell `sev beta` from `distortion beta`. The rule, stated in the
  docstring for every consumer, is aLL's existing one: a word in any other group is a keyword, and
  only a word found solely in `distortions` is an atom.
- **`python -m aggregate.decl_words` prints the list as JSON**, with the package version beside
  the groups, for consumers that are not Python: a web build, or a script.
- **AggLexer builds its `words()` rules from `decl_words()`.** Its token types do not change: the
  severity table's parameter counts give today's `Name.Function`, `Name.Class` and
  `Name.Namespace` split. Its hand lists go. The lexer still does not color distortion names; the
  docs' coloring is unchanged, which keeps this plan to one behavior.
- **No new package data.** The list is derived at run time from `decl.lark`, which already ships,
  so nothing generated is committed and nothing can go stale.
- **Out of scope:** `_TERMINAL_LABELS` (labels, not just words) and `agg.sublime-syntax`. Both stay
  hand-kept and test-checked as today.

Names vetted at planning time: `decl_words`, `decl_words.py` and `aggregate.decl_words` collide
with nothing in `src/aggregate` (checked with `rg -w`).

## Files

- `src/aggregate/decl_words.py` (new), `src/aggregate/decl_pygments.py`
- `tests/test_grammar_sync.py`, and a new `tests/test_decl_words.py`
- `src/aggregate/__init__.py` (`__version__`), `CHANGELOG.md`, `dev/TODO.md`

## Acceptance

- `uv run pytest`, then `uv run pytest -m 'slow or not slow'` at the bump.
- **AggLexer is unchanged in behavior.** Before and after, its token stream over `library.agg`,
  `decl-testers.agg`, `_test_suite.agg` and `_test_suite2.agg` is identical, token for token.
- `decl_words()['reserved']` equals `grammar_reserved_words()` as the test computed it before the
  move, all 90 words.
- Every name in `severities` resolves: `getattr(scipy.stats, name)` succeeds, or the name is one of
  the three that `_classify_sev` special-cases.
- Every name in `distortions` builds a `Distortion` of that kind.
- `python -m aggregate.decl_words` prints JSON that parses, and its groups equal `decl_words()`.

## Consumers, after release

- **aggregate-api** serves the list at a new `GET /v1/decl/words`, beside the existing
  `GET /v1/decl/grammar`. aLL's web build writes it into the bundle and retires
  `decl-keywords.json`.
- **aggregate-asp** fetches the list from that route and drops its inline regex.
- **house-style** generates `decl.xml` from `decl_words()` alone.

Each of these is the other repository's own plan. This one changes nothing outside aggregate.

## Open questions for Steve

1. **Words aLL colors that the grammar does not reserve.** Computed on 2026-10-19 against the
   grammar, AggLexer and the distortion registry, aLL's pool holds 25 such words, after `inf` and
   `note` (which the new groups cover): `between blend build constant convex fast in infinity
   kurtosis linear loc mean p0 p1 phasefit proportional samples scale shape share skewness slow
   update weibull wt`. Some are values the transformer checks after parsing (`phasefit` is an
   approximation kind, an ordinary `ID` to the grammar). Some are completion hints. Five sit in
   aLL's distortion group but are not kinds in the registry: `proportional linear constant convex
   blend`. The recommendation is that the grammar and the
   registry decide what is colored, so these stop being colored, and aLL's completion keeps
   whatever it finds useful. This is decided in the aggregate-api plan but follows from this one.
2. **Distortion names in the docs.** AggLexer could color them as Pygments `Name.Builtin`, as aLL
   and the house do. That would be a visible change to the Sphinx docs, so it is left out unless
   you want it.
