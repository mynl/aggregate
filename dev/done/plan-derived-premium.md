# Plan: derive premium (gross the engine premium up for expenses)

> **Status: EXECUTED at `1.0.0a270`, 2026-08-13.** Design approved by the
> author the same day. A new DecL premium head for `pnl` and `xpnl`,
> `derive premium`, plus a flip of `pnl_program` to emit it. Paired with
> the API repo's `dev/plan-pnl-button.md`, which consumes the flip.
>
> Execution notes, four divergences from the text below. (1) The version is
> a270, not a269: `[Chart-Reflected-Reading]` took a269 mid flight. (2) The
> worked example as drafted is not grammatical: an `as` label closes an
> expense **group** and `and` joins terms **within** one, so a labeled term
> cannot be followed by `and`. The executed spelling juxtaposes labeled
> groups (`10 fixed expense as "App Fee" 2 fixed expense as "Admin fee"
> 5% premium expense as "TLF" ...`); the sums, and so the derived premium,
> are identical. (3) The features page capability table ends at a195 and
> recent features use version headed subsections, so a "Derived premium
> (a270)" subsection was added instead of a table row. (4)
> `docs/4_dec_Language_Reference.rst` needed no hand edit: it carries no
> premium head prose, and the regenerated grammar listing covers it.

---

## Motivation

`inherit premium` copies the engine's technical premium T, and the `less`
clause then deducts expenses from it, so expenses eat the risk load: the
expected underwriting result is T less expenses less expected loss. The
a268 `[DecL-Paren-Arithmetic]` CHANGELOG entry already names the hand
written workaround, `(100_000/(1-.25)) premium`. `derive premium`
automates it. T is read as a technical, risk loaded premium and grossed up
so that premium net of expenses returns exactly T, leaving the engine risk
load as the expected underwriting result.

## Specification

Grammar: `derive premium [as <label>]`, a fourth `pnl_premium` alternative
beside the fixed amount, `inherit`, and `retro` heads. Shared by `pnl` and
`xpnl`; mutually exclusive with `retro` by construction (they are sibling
alternatives).

Formula. With engine premium T, fixed expense total F (every fixed term in
the `less` clause, summed) and premium expense ratio r (every premium
term, summed, combined ratio style):

    P = (T + F) / (1 - r)

Then P less its own expenses, r * P + F, equals T exactly. Grouping and
`as` labels do not affect the sums; terms accumulate across groups.

Worked example. Engine premium 100 and

    less 10 fixed expense as "App Fee" and 2 fixed expense as "Admin fee"
    and 5% premium expense as "TLF" and 15% premium expense as "Commission"
    and 7% premium expense as "G&A"

gives P = (100 + 12) / (1 - 0.27).

Errors, all build time `ValueError` whose message names the fix:

1. The engine has no premium (same trigger as `inherit premium`).
2. Any loss basis expense. Losses are not reliably known by inspection, so
   the gross up is not defined. The fix is fixed or premium expenses, or
   `inherit premium` with the loss expense left in place.
3. Premium expense ratios totalling 1 or more: no finite premium grosses
   that up.

Edge rulings (author, 2026-08-13):

- No expense clause is legal and gives P = T, identical to inherit.
- Port engines are supported, parallel to inherit (both resolution sites).
- Vector premiums sum exactly as inherit sums them; `derive` reads the
  same merged `exp_premium`, including an FYI premium (a266).
- No rounding, matching the inherit head (compare
  `test_pnl_program_does_not_round_an_inherited_premium`).

## Design, dependency order

New public names, vetted against the existing surface at planning time:
`DERIVE_PREMIUM` (`parser.py` sentinel singleton, no collision) and
`derive_consideration` (`_pnl_builders.py` helper, no collision). `derive`
is unused as an identifier in every shipped corpus (verified by grep
2026-08-13); reserving it is technically breaking and the CHANGELOG states
it, per the a249 `[Single-Placement-Keyword]` model.

1. `src/aggregate/decl.lark`. `DERIVE.2` terminal beside `INHERIT.2`
   (line 767, same negative lookahead shape); `derive` added to the `ID`
   exclusion alternation (line 843, the authoritative reserved word list);
   fourth alternative `| DERIVE PREMIUM as_label -> pnl_premium_derive` in
   `pnl_premium` (lines 181 to 183); extend the head comment (lines 175 to
   180) with the derive reading.
2. `src/aggregate/parser.py`. `_DerivePremium` sentinel class and
   `DERIVE_PREMIUM` singleton beside `INHERIT_PREMIUM` (lines 124 to 140),
   added to `__all__` (lines 59 to 60); transformer method
   `pnl_premium_derive` mirroring `pnl_premium_inherit` (lines 832 to
   844). Spec representation: `spec['consideration'] is DERIVE_PREMIUM`;
   no new spec key.
3. `src/aggregate/_pnl_builders.py`. Helper beside `resolve_expense`
   (lines 173 to 197), suggested signature
   `derive_consideration(expense_spec, technical, name)`, reusing
   `_normalize_expense_groups` (lines 119 to 144) so legacy spec shapes
   are honored. It accumulates (F, r) over the fixed and premium terms,
   raises on a loss basis and on r >= 1, and returns
   `(technical + F) / (1 - r)`. Keeping it next to `resolve_expense`
   keeps the forward map and its fixed point adjacent.
4. `src/aggregate/underwriter.py`. Derive branch beside the inherit
   resolution (lines 1184 to 1185) calling a new
   `_derive_agg_premium(name, spec, expense_spec)` modelled on
   `_inherit_agg_premium` (lines 1933 to 1951): extract T the same way,
   erroring if the engine has none, then delegate to the builders helper.
   Resolution must stay ahead of `_resolve_reins_economics` (line 1222)
   so ceded premium rates see the derived gross. Port path: matching
   branch in `_snapshot_pnl`, `port_plain` case (lines 2027 to 2040); the
   recipe already carries `expense_spec` (line 1981).
5. `src/aggregate/decl_writer.py`. `elif consideration is DERIVE_PREMIUM`
   branch in the `_render_pnl` premium head (lines 926 to 937) emitting
   `derive premium`. Required, not optional: the sentinel breaks
   `_fmt_seq`. Round trip is then free because the writer emits from the
   unresolved spec.
6. Keyword mirrors, all enforced by `tests/test_grammar_sync.py`:
   `parser_errors._TERMINAL_LABELS` entry
   `"DERIVE": "'derive' (gross the engine premium up for expenses)"` near
   line 136 (the quoted token shape matters: the API's server side
   completion parses it and drops descriptive only labels);
   `decl_pygments.py` keyword tuple (lines 264 to 284);
   `agg.sublime-syntax` keyword alternation (line 116) and ID lookahead
   (line 158).
7. `pnl_program` flip. `src/aggregate/_program.py` `_pnl_consideration`
   (lines 596 to 648) returns `DERIVE_PREMIUM` instead of
   `INHERIT_PREMIUM` when the engine carries premium; the `expense_ratio`
   clause emission (line 786) is unchanged. **Behavior change, bold in
   the CHANGELOG**: with the default 0.25 expense ratio the emitted head
   now resolves to T/0.75 rather than T. The loss ratio sized head
   (engine without premium) and its rounding are untouched.

## Tests

- New `tests/test_pnl_derive_premium.py`: parse puts the sentinel in the
  spec; the worked example builds to (100 + 12)/0.73; multiple fixed terms
  add; multiple premium terms add; the invariant, derived premium net of
  `resolve_expense(...)` equals T; a loss expense raises; no engine
  premium raises; r >= 1 raises; an empty expense clause equals inherit;
  a port engine derives; `xpnl` derives; the FYI premium is read (mirror
  `test_fyi_premium.py` lines 86 to 89).
- `tests/test_derived_programs.py`: retitle and update
  `test_pnl_program_inherits_an_engine_premium` (lines 261 to 267) and the
  no rounding test (lines 310 to 315) to the new keyword.
- Corpus: new section in `src/aggregate/agg/decl-testers.agg` following
  the ES. and EXP. header shape (banner, `# DRV. Derived premium
  (1.0.0aNNN)`, prose naming the bracketed label, the plan file and the
  test file, statements with `note{}` trailers). The refusals are
  underwriter time, so no `X.` parse error fixtures are owed.
- Regenerate `tests/data/expected_specs.json` via
  `uv run python tests/capture_spec_snapshot.py`; read the diff
  deliberately.
- `tests/test_grammar_ambiguity.py`: `KNOWN_AMBIGUOUS` must not grow (a
  distinct leading keyword should not fork the forest).

## Docs

- `docs/2_aggregate_overview/pipeline-pnl.rst` line 79: the sentence
  enumerating three premium head forms becomes four.
- `docs/2_aggregate_overview/features.rst`: feature table row near line
  151 and a subsection with an ipython example near line 1788.
- `docs/4_dec_Language_Reference.rst`: prose reference entry.
- Regenerate `docs/4_agg_language_reference/ref_include.rst` via
  `python -m aggregate.parser`, same commit as the grammar edit.
- `cheat-sheets/DecL_Cheat_Sheet.tex`, a hand maintained surface.

## Release hygiene

One bump to the next free `1.0.0a*`. CHANGELOG section `[Derived-Premium]`
covering the keyword, the reserved word note, the `pnl_program` behavior
change in bold, the a268 motivating case, and (a249 convention) naming the
web app's `decl-keywords.json` as needing the mirror edit, pointing at the
API's `dev/plan-pnl-button.md`. Update `dev/TODO.md`; regenerate
`dev/FEATURES.csv` via `uv run python dev/regen_features.py`; move this
plan to `dev/done/`; one line commit `[Derived-Premium] aNNN: derive
premium grosses the engine premium up for expenses`. Test tiers per house
rules: testmon in the edit loop, `uv run pytest` before declaring done,
the `slow or not slow` sweep at the bump.

## API ripple

The API repo's `dev/plan-pnl-button.md` (written 2026-08-13) picks the
flip up: the SPA PnL button posts an empty body and echoes
`pnl_program(loss_ratio=0.70, expense_ratio=0.25)`, so it emits `derive
premium ... less 0.25 premium expenses` with zero app code. App side items
there: two docstring rewords, two test assertion updates, and a new `pnl`
group in `decl-keywords.json` carrying `derive` and the rest of the pnl
clause vocabulary.
