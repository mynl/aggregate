# Plan — hygiene 3 (rolling collector)

> **Status: landed in 1.0.0a52** (Items 1, 3, 4). Item 2 was deferred and moved
> to `dev/TODO.md` Track H (H9) — see its section below. This was a rolling list
> of nits and gnats added over time (the author says *"add to hygiene plan: …"*),
> executed as one batch.
>
> **Batch rules**
> - **One version bump for the whole group** — a single `1.0.0a*` increment in
>   `pyproject.toml` covering every item, not one per item.
> - **One `CHANGELOG.md` section** for the batch, bulleting each item.
> - Update `dev/TODO.md` for anything that lands or shifts; move this plan to
>   `dev/done/` once the batch is executed.
> - Each item is small and independent unless noted. Items that touch the frozen
>   numeric baseline say so explicitly; the default expectation is **no number
>   movement** (these are correctness/robustness/ergonomics nits).
> - Run `uv run pytest` green before committing; add a focused regression test per
>   item where it is cheap.

---

## Item 1 — allow Python-style `_` digit separators in numbers

**Goal.** Accept underscores as digit-group separators in DecL numbers, exactly as
Python does:

```
agg BIG 10_000_000 claims dsev [-1 3] poisson
```

**Where.** The single `NUMBER` terminal in `src/aggregate/decl.lark` (line 389):

```lark
NUMBER.2: /-?(\d+\.?\d*|\d*\.\d+)([eE][+\-]?\d+)?%?|-?inf/
```

**Fix.** Replace each digit run `\d+` / `\d*` with an underscore-aware run that
forbids leading, trailing, and doubled underscores — `\d(?:_?\d)*` — so the lexer
itself rejects `_1`, `1_`, `1__0`, `1_.0` (matching Python). Sketch:

```lark
NUMBER.2: /-?(\d(?:_?\d)*\.?(?:\d(?:_?\d)*)?|\.\d(?:_?\d)*)([eE][+\-]?\d(?:_?\d)*)?%?|-?inf/
```

**Transformer.** Python's `float('10_000_000')` and `int('1_000')` **already**
accept underscores, so the numeric coercion in `parser.py` should need no change
*provided* it goes through `float`/`int` (and the `%`-suffix / `_PercentNumber`
stripping runs as today). **Verify** the conversion path does not regex-split or
hand-parse digits; if it does, strip `_` before converting.

**Tests.** Add to `test_decl.agg` (and a pytest case): `10_000_000`, `1_000.5`,
`1_0e3` parse and equal their underscore-free forms; `_1`, `1_`, `1__0` raise a
parse error. No number movement (grammar-only, equal values).

---

## Item 2 — every public DataFrame member initialized to `None` — **MOVED OUT**

**Deferred at a52; moved to `dev/TODO.md` Track H as H9.** Review found the item
as written was largely either already-satisfied or aimed at members that do not
exist: the scope list named several non-members (`reins_audit_df`, public
`reins_df` / `report_df` / `statistics_df` / `bs_window_df`); the real properties
(`density_df`, `sev_density_df`) already raise an *informative*
`ValueError('Update … first')` rather than `AttributeError`, `reins_*_df` already
return `None`, and `augmented_df` is a parameterised method that cannot return
`None`. The genuine, narrow version (enumerate the real members per class; decide
informative-`ValueError`-vs-`None`) is tracked as **H9** for separate scoping.

---

## Item 3 — fix array-ambiguous truth test `if not self.sevs:`

**Goal.** `Aggregate._sev_label` (`src/aggregate/distributions.py:2145`) does:

```python
if not self.sevs:
```

which raises `ValueError: The truth value of an array with more than one element
is ambiguous` when `self.sevs` is array-like.

**Fix.** Use the safe idiom already used elsewhere in the file (lines 4233, 6171,
6206):

```python
if self.sevs is None or len(self.sevs) == 0:
```

**Sweep.** Grep `distributions.py` (and `portfolio.py`, `spectral.py`) for other
`if [not] self.<attr>:` truthiness tests on attributes that may be list/ndarray —
candidates include `sevs`, `en`, `sev_wt`, `xs`, and any density/array member —
and convert each to an explicit `is None` / `len(...)` / `.size` test. Leave scalar
and genuine-bool attributes alone.

**Tests.** A regression that builds an aggregate whose `self.sevs` is array-like
(the configuration that triggered the report) and calls `_sev_label` /
`tail_description` without error. No number movement (control-flow guard only).

---

## Item 4 — `of` as a synonym for `po` / `so` in reinsurance

**Goal.** Accept `of` as a natural-language share indicator in a reinsurance
clause, so this reads cleanly:

```
agg B 1 claim 10000 xs 0 sev lognorm 120 cv 1.5 occurrence net of 90% of 6000 xs 4000 poisson
```

**Grammar check (done — it is safe, no ambiguity).**
- `OF` is **already** a terminal (`decl.lark:369`); the dynamic lexer already maps
  any standalone `of` to `OF`. This is a **rule-only change, no new/changed
  terminal**, so lexing is untouched.
- Structural `OF` occurs only at the **head** of a reins clause —
  `OCCURRENCE/AGGREGATE NET OF reins_list` (`decl.lark:154–159`). It never recurs
  inside `reins_list` (which is `reins_clause` joined by `AND`, or a `tower`).
  So adding an `OF` alternative to `reins_clause` cannot collide with `NET OF`:
  the head `OF` is consumed first, and a clause's `OF` always sits *after* its
  leading `expr`. Token stream `… NET OF | 90% OF 6000 XS 4000` parses cleanly,
  and `expr OF expr XS expr` does not overlap `expr XS expr` (`reins_clause_xs`).

**Fix.**
1. `decl.lark` `reins_clause` (lines 166–168): add a third alternative
   ```lark
   | expr OF expr XS expr        -> reins_clause_of
   ```
2. `parser.py`: add `reins_clause_of` mirroring `reins_clause_share`
   (parser.py:692) — `so`/`po`/`of` are all synonyms, and the `%`-vs-bare rule
   (`_PercentNumber` → share directly; bare → `amount / limit`) decides the
   meaning. `90% of …` therefore equals `90% so …`. (No "suspiciously small"
   warning needed; `of` reads as a share.)
3. Regenerate `ref_include.rst` (grammar listing) — see
   [[project_ref_include_regen]]: `python -m aggregate.parser` writes to the
   `src/` path under the src layout, so copy the output to the real `docs/` path.

**Tests.** Add to `test_decl.agg` (section J, with the existing `J.Re18*` so/po
cases) and a pytest assertion: `90% of 6000 xs 4000` yields the same
`(share, limit, attach)` as `90% so 6000 xs 4000` and `90% po 6000 xs 4000`.
Grammar changed, but no existing program uses `of` as a share indicator, so no
number movement on the knowledge base.

---

<!-- Append new items below as "## Item N — …" when the author says
     "add to hygiene plan: …". Keep one version bump for the whole group. -->
