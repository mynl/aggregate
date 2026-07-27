# Plan — `[Program-Mixin]`: the shared DecL round-trip surface

**Shipped `1.0.0a154`.** Third mixin, after `LabeledMixin` (a128) and
`HelpMixin` (a150).

## Why

`pprogram` / `pprogram_html` were **ten near-identical properties across five
classes and five files** — `Aggregate` (`_aggregate.py:4165`), `Portfolio`
(`_portfolio.py:2405`), `Severity` (`_severity.py:1199`), `PnL`
(`_pnl.py:1849`), `BivariateAggregate` (`bivariate.py:2391`) — each a
three-line delegation to `decl_writer.format_program(self.program, fmt=...)`.
`pprogram_html` was byte-identical on `Aggregate`/`Portfolio` and again on
`PnL`/`BivariateAggregate`. The only real variation was a redundant
`if self.program` guard on `Severity` and whether the import was deferred.

The trigger was a capability request, not tidiness: **a `pprogram` that omits
`note{...}` / `hints{...}`**. Every host hard-coded the render options in its
own property body, so a new render axis cost an edit per class. Collapsing the
copies is what made the axis affordable.

## Why a narrow mixin, not one "FCC mixin"

The shared surfaces have **different host sets**. Membership is the whole
design; a single class throws it away.

| shared surface | hosts | delivery |
|---|---|---|
| `help` | 11 (every FEATURES column) + `Underwriter` | `HelpMixin`, a150 |
| `label` / `labels` / `renamer` / `use_labels` | 5–6 (+`Copula`) | `LabeledMixin`, a128 |
| `program` / `pprogram` / `pprogram_html` | 5 → **6** | `ProgramMixin`, a154 |
| `info` | 10 (all but `GridDistribution`) | **rejected**, below |

One mixin over all four would give `pprogram` to `Frequency` and `info` to
`GridDistribution`. Also: `HelpMixin` is stateless; the program surface is
stateful. And "utilities" is not a role, so a grab-bag mixin cannot be named
under the house `<Role>Mixin` rule.

### `InfoMixin` — examined and rejected

The raw copy count (ten `info` properties, the same five-sentence docstring
pasted each time) looks like the bigger prize. Reading the bodies says
otherwise: per class the genuinely shared code is **two lines** — the
`info_row` comprehension and the `'\n'.join`. Everything else is payload.
`Aggregate.info` (`_aggregate.py:2518-2565`) is 48 lines of `updated` guards,
premium / loss-ratio derivation and 23 hand-built rows; `Frequency.info`
computes moments off `base_mean` rather than `en` or zm/zt double-applies.
Even the footers diverge — `Aggregate` appends tail rows + `bounded` + `id`,
`Severity` tail + `bounded`, `Frequency` tail only.

So an `InfoMixin` would collapse ≈20 lines and force ten hosts through an
`_info_rows()` hook to do it — the `plan-split-distributions.md:302` failure
mode, *"mixins relocate lines without reducing coupling."* What is shared is
the **contract**, not the code, and it already has two homes:
`dev/info-strings.rst` and `tests/test_fcc_surface.py`. Contrast `pprogram`,
where the shared fraction is the entire body.

*(Also examined and left alone: `validation_explanation` is byte-identical on
`Aggregate`/`Portfolio`; so is `spec_ex`; `_repr_html_` shares a docstring and
tail on the two and wants a third host, `[PnL-Repr-HTML]`. Each is a two-host
duplicate — below the bar that justified `HelpMixin`'s nine.)*

## What shipped

**1. `trailer` axis on `format_program`** (`decl_writer.py`).

```python
format_program(spec_or_text, *, fmt='text', layout='spread',
               trailer=True, width=None)
```

One flag, because `note{...}` and `hints{...}` are one grammar construct
(`decl.lark:480`) emitted by one function (`_render_trailer`). Threaded
explicitly through the nine renderers rather than stripping keys from a copied
spec — no hidden knowledge of which keys nest where, so a portfolio's units and
a bivariate's components are covered by construction.

`spec_to_decl` is untouched (terse, trailer-on): it backs `to_agg` and the
round-trip snapshot contract.

**2. `ProgramMixin`** (`src/aggregate/_program.py`, new) — `program` (empty
class-level default), `format_program(fmt=, layout=, trailer=)`, `pprogram`,
`pprogram_html`. No `__init__`, per house idiom.

**3. Six hosts.** `Aggregate`, `Portfolio`, `PnL`, `Severity`,
`BivariateAggregate`, and — the hole this closed — `Distortion`.

**4. `Distortion` gained the surface.** It is DecL-creatable (the writer has a
`'distortion'` kind renderer; `underwriter.py:1286` stamps `obj.program`) but
`spectral.py` declared neither `program` nor `pprogram`. Same shape of hole
`HelpMixin` closed for `Frequency` / `GridDistribution`.

**5. `Portfolio.nice_program` retired** — a `textwrap.fill` over the *raw*
program, method not property, non-NumPy docstring, zero call sites. The
`from textwrap import fill` import went with it.

**6. `Portfolio.note` gap fixed** — the parser produced `note` for a `port`
spec and `_render_port` rendered it, but the class never stored one, so a
portfolio note could be written and never read back.

## Decisions (confirmed with the author)

- **One `trailer=` flag**, not separate `note=` / `hints=`. One grammar
  construct, one switch, one code path.
- **A bound `format_program(...)` method**, not properties-only. Same canonical
  name as the free function it wraps, mirroring `HelpMixin.help` over
  `utilities.agg_help`. Without it the mixin would remove ten copies but buy no
  ergonomics — the variant would still need a module-level import.
- All four adjacent items taken (`Distortion` host, `nice_program`,
  `Portfolio.note`, incidentals logged).

## Carve-outs — recorded so a future reader does not "finish the job"

- **`Frequency`, `GridDistribution`, `Bounds`, `AllocationBounds`,
  `PricingBounds` are not hosts** — not DecL-creatable, no `program`. This is
  what makes `ProgramMixin` narrower than `HelpMixin`, and
  `tests/test_fcc_surface.py::test_program_mixin_hosts_and_non_hosts` asserts
  it in both directions.
- **`Underwriter` is not a host** — it *renders* programs (`to_agg`,
  `_entry_to_decl`); it is not itself a declaration.
- **No `InfoMixin`.**

## Guardrail: this is not the rejected `ReprMixin`

`plans-considered-and-rejected.md:10-43` rejected a display-mode `ReprMixin`
(2026-07-27) as "not worth the effort": it bought *keystrokes, not capability*,
through a mode flag resolved three layers away. This is the opposite —
`trailer=False` is a capability that did not exist, and it is an explicit
call-time argument. No config field, no env var, no module global, no instance
mode.

## Verification performed

- **Byte-for-byte regression at defaults:** all 1204 renders of every statement
  in every shipped `.agg` file, in both layouts, identical before and after.
- **Round-trip:** `format_program(p)` re-parses to the source spec exactly;
  `trailer=False` re-parses to `{**source, 'note': '', 'hints': ''}` — nothing
  else moves.
- **Semantic `!` survives `trailer=False`** (zero-modified mean pin,
  unconditional severity).
- New corpus entries `FCC.Note` / `FCC.RT` / `FCC.Pin` / `FCC.NoteP` in
  `decl-testers.agg` §HINTS, covered by the existing unparser round-trip sweep.
- `dev/regen_features.py`: 0 failures, 0 undocumented capabilities;
  undocumented attributes 90 → 88 (`note` / `hints` now documented rows).
- Full suite `-m 'slow or not slow'`: **2636 passed**.
