# plan-derived-programs: a program that reproduces a derived object

Status: **executed** at `1.0.0a213`. Both open questions went the way the
recommendations below argued: `sharpen_program` stayed a property, and the loss
ratio and expense ratio stayed in the signature. One case the plan did not
name turned up in execution and is recorded under Open.

Three new members of the DecL round-trip surface, each answering the same question about a different derivation: what is the program that would build this? Sharpening a grid, wrapping an object in a P&L, and adding a cession all produce a new object, and today none of them produces the text that reproduces it.

Split out of `aggregate_api/dev/plan-loss-lab-navigation.md`, which is the immediate consumer and is parked until the plots workstream settles. This plan does not depend on that one and touches neither plots nor charts, so it can land whenever it suits.

## Why

The round-trip surface already says what an object is: `program` is the statement the parser received, `pprogram` is what the parser understood, and `format_program(trailer=...)` renders either with the metadata clauses in or out. That covers objects you declared. It says nothing about objects you arrived at.

Three ways of arriving are common enough to deserve text.

You call `sharpen()`, it moves the grid, and the knowledge of which grid now lives only in the live object. Reopening the notebook tomorrow rebuilds on the automatic choice, and the audit has to run again. A `hints{log2=...; bs=...}` clause is exactly the durable form of that answer, and the machinery to honor it already exists: `hints` is an allow-listed build clause resolved caller-wins in `underwriter.py:229`.

You have an aggregate and want to see it as a P&L. The DecL is `pnl NAME <premium> less <engine> less <expenses>`, the engine can be inlined verbatim (`agg_source_inline`, `decl.lark:152`), and yet writing it out by hand means knowing that the trailer belongs to the wrapping `pnl` rather than to the engine, and knowing what to do when the engine carries no premium.

You have an aggregate and want to cede a layer. The clause cannot be appended, because an occurrence cession sits before the frequency clause and an aggregate cession at the end. Anyone splicing text rather than specs gets this wrong.

Each is a small piece of grammar knowledge that currently has to be re-derived by every caller. The library holds the grammar, so the library should answer.

## What lands

Three names, each joining an existing family rather than starting one.

### `sharpen_program`

A property on `Aggregate` and `Portfolio`, populated by `sharpen()` alongside `sharpen_df`, `sharpen_description` and `sharpen_explanation`. It holds the object's program with the outcome of the last probe merged into its trailer.

The grid moved, so it carries `hints{log2=...; bs=...}` and nothing else, because the hints are the record.

The grid was confirmed, so it carries `note{sharpen: grid confirmed, no change}` and no hints. No hints deliberately: pinning a grid the automatic selector would have picked anyway adds noise to a program someone is going to read and share, and implies the selector is not trusted. The note is the record that the audit ran, which is what saves running it twice.

Empty before `sharpen()` has been called, exactly as `sharpen_df` is.

### `pnl_program`

A method on `Aggregate` and `Portfolio`, since it takes arguments:

```python
a.pnl_program(loss_ratio=0.70, expense_ratio=0.25)
```

Returns `pnl NAME_PnL <premium> less <engine> less 25% premium expense`. The engine is the object's own body inlined, stripped of its trailer, which the wrapping `pnl` owns. Premium is `inherit premium` when the engine carries one, and otherwise a fixed amount computed as expected loss divided by `loss_ratio`. Passing `expense_ratio=0` omits the expense clause rather than writing a zero.

### `reins_program`

A method on `Aggregate`, taking the cession fragment:

```python
a.reins_program('occurrence net of 500 xs 500')
```

Returns a self-contained program with the clause in its correct slot, derived by mutating the spec and re-rendering, never by string surgery.

**Self-contained, and this is the load-bearing choice.** `agg NEW agg.OLD occurrence net of ...` is grammatical (`agg_body_rename`, `decl.lark:85`) and is the obvious first idea, but `agg.OLD` resolves only against an `Underwriter`'s knowledge base. Requiring the caller to have registered the object first makes the returned text non-portable: it builds in the session that made it and nowhere else. For the app that is worse still, since a shared server would be writing every user's builds into one shared knowledge base, with the name collisions and unbounded growth that implies. Text that carries its own engine has neither problem.

## Naming, vetted

The house rule that `approximate` broke says vet at planning time, so this is the check rather than a promise to check.

`rg` across `src/aggregate` and `tests` finds no method, property, attribute assignment, DecL keyword or spec key named `sharpen_program`, `pnl_program` or `reins_program`. All three are free.

Each also joins a family that already exists, which is why these won over the first candidates (`sharpened_program`, `reinsured_program`, `derive_program`). `sharpen_df` / `sharpen_description` / `sharpen_explanation` make `sharpen_program` read as the fourth thing a probe produces. `reins_df` / `reins_stats_df` / `reins_description` make `reins_program` the obvious spelling, and `reins` stays canonical on the Python surface, per a41, whatever the app calls the tab. `pnl_recipe` and `pnl_units` do the same for `pnl_program`.

**One wart to accept or reject.** `sharpen_program` is a property while the other two are methods, because the first is the residue of a probe you already ran and the others take arguments. Within one naming pattern that is a mild inconsistency. The alternative is `sharpen_program()` as a method for uniformity, at the cost of breaking step with `sharpen_df` beside it. Recommendation is to keep the property and take the wart, since the sharpen family is the stronger pull.

## Where the code goes

Shared machinery in `_program.py`, thin members on the host classes, which is the pattern `sharpen` itself already follows: the work in `_bucket_window.sharpen`, three-line delegations on `Aggregate` and `Portfolio`.

`_program.py` currently holds `ProgramMixin` and nothing else, and the mixin is the right neighborhood: all three of these are the mixin's question asked about a derived object. They are not mixin *members*, though, because the mixin's six hosts include `Severity`, `Distortion` and `BivariateAggregate`, none of which can answer any of the three. A member that raises on four of six hosts is a member in the wrong place.

So: a private helper in `_program.py` that parses a program to a spec, applies a mutation, and renders through `decl_writer`, plus the three public members on the classes that can honestly answer.

## Two constraints that shape all three

**The trailer is merged, never appended.** A spec holds one `note` and one `hints`. Appending a second clause produces text whose second value silently wins or loses depending on the transformer, so each mutation merges keys into whatever is already there. A program that arrives with `hints{bs=1/32}` and gets sharpened to a different bucket comes back with the bucket replaced and any other hint key untouched.

**The render must ask for the trailer.** `format_program` defaults to `trailer=False`, on the reasoning that formatting a program is usually about the math rather than the metadata. These three are the exception: the trailer is the entire payload for `sharpen_program`, so the render names the clauses it needs rather than taking the default.

## Tests

Round-trip is the whole contract, so that is what gets asserted. For each function, the derived text parses, builds, and produces the object it claims to: the sharpened program builds at the sharpened grid, the P&L program builds a `PnL` whose engine matches, the reinsured program builds an aggregate with the expected cession.

Cases that need explicit coverage because they are where a naive implementation breaks: a program that already carries `note{}`, one that already carries `hints{}`, one carrying both, an engine with no premium going through the P&L wrap, an occurrence cession and an aggregate cession through the reins splice (different slots), and a sharpen that finds no improvement.

House rule from `CLAUDE.md`: every DecL program embedded in a pytest case is mirrored into `src/aggregate/agg/decl-testers.agg` under its matching section, and must round-trip. These cases produce DecL, so the outputs belong there as well as the inputs.

## Housekeeping

One version bump, one commit, carrying the code, the `pyproject.toml` bump, the `CHANGELOG.md` section, a `dev/TODO.md` entry marked done, and this plan moved to `dev/done/`.

## Open

The property-versus-method wart above, to accept or reject. **Resolved as recommended**: `sharpen_program` is a property, the other two are methods.

Whether `pnl_program`'s defaults belong in the signature or in `constants.py`. The 0.70 loss ratio and 0.25 expense ratio are conventions rather than facts, and a convention that appears in a signature is harder to find later than one with a name. Recommendation is the signature, since they are per-call choices and a caller reading the docstring sees them immediately, but it is worth a moment's thought. **Resolved as recommended**: both live in the signature.

## Found in execution

**A third sharpen outcome.** The plan names two, moved and confirmed. A probe run under `execute=False` is a third: it finds a better cell and does not take it, so the object still sits on its original grid. Pinning the winner would describe an object that does not exist, and writing "grid confirmed" would be false, so it carries a note recording the recommendation and no hints.

**`Portfolio.pnl_program` cannot be self-contained.** The grammar's `agg_source` admits an inline `agg`, a stored `agg.NAME` or a stored `port.NAME`, and there is no inline portfolio engine, so the portfolio form references (`less port.NAME`) where the aggregate form inlines. The two-statement alternative (the `port` declaration followed by the `pnl`) was rejected: `build` takes exactly one top-level output, so the returned text would no longer be something you can hand to `build`. `Aggregate.pnl_program` and `reins_program` are self-contained as planned.

**A cession clause per tier, and a sequence for both.** `reins_program` takes a string or an iterable of them, one per tier, since composing two tiers otherwise means a `build` round trip between the calls. A clause is authoritative for its own tier and leaves the other alone.

**`approximate` and an occurrence cession.** The parser rejects the combination, so `reins_program` rejects it too rather than returning text that will not build.

**Pre-existing, found by the full-suite gate and left alone.** `tests/test_agg_libraries.py::test_library_is_written_in_the_canonical_layout` fails at `a212` as well as here, on three `library.agg` entries (`USXOLTower` writes `90% po` where the writer renders `90% so`; `USHurr` writes `less agg.USXOLTower` where the writer inlines the engine; `BodoffFour` has a double space in `sev   4 * expon`). Two are cosmetic and one is a deliberate authorial reference, so this is the author's call, not a rider on this plan.
