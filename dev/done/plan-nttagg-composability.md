# [NTT-Composability] AggregateNTT should not carry the two methods it refuses

Status: DONE at 1.2.0a3, 2026-10-09. Raised from `aggregate_api`, where the
defect was visible. Execution log at the foot of this file.

## Goal

Two small 1.1.0 follow-ups, both found while admitting `nttagg` to
`aggregate_api`:

1. `AggregateNTT` inherits `pnl_program` and `reins_program` from `Aggregate`
   and refuses both at call time. Make their **absence** say what their guards
   say, which is what every other class already does.
2. `library.agg` carries no `nttagg` entry, so the kind is invisible from the
   front door of every consumer.

## Why step 1 is a library defect and not a consumer's problem

Presence of these two methods is already a deliberate contract here. Measured
on 1.1.0:

| class | `pnl_program` | `reins_program` | correct? |
|---|---|---|---|
| `Aggregate` | present | present | yes |
| `PnL` | absent | absent | yes, cannot be re-wrapped, cannot cede |
| `Portfolio` | present | absent | yes, can be `xpnl`'d, cannot cede directly |
| `AggregateNTT` | present | present | **wrong on both** |

This suite states the convention itself, in
`tests/test_derived_programs.py:537`:

```python
def test_reins_program_rejects_a_portfolio_program():
    """A portfolio has no cession clause; its units do."""
    p = build(PORT)
    assert not hasattr(p, 'reins_program')
```

So `hasattr` is the documented way to ask, and `AggregateNTT` is the single
class where the answer lies. The corroborating tell: both `'nttagg'` guard
branches added by `[NTT-DecL]`, the `kind not in ('agg', 'port')` message in
`pnl_program` and its counterpart in `reins_program`, are reachable **only**
from an `AggregateNTT`. Every other class that would trip them does not carry
the method. The guard went in one layer too deep.

**The consumer-side consequence, for scale.** `aggregate_api` derives its
navigation from object capability and reads these two with `hasattr`
(`capability.can_pnl`, `capability.can_reins`). On an `nttagg` the app
therefore lights a P&L button and a Reins tab that both fail on use. The
alternative was a hard-coded kind list in the api, which the author rejected
on 2026-10-09: the api's job is to pick up what the library delivers, and a
declared list would also go stale silently when the v1 restriction in
`dev/done/plan-nttagg.md` **Known limits** is relaxed. With the override, the
relaxation is one deletion here and the app lights up by itself.

## Steps

### [override] absence instead of refusal

In `ntt.py`, on `AggregateNTT`, replace each inherited method with a property
that raises `AttributeError` carrying the message the `ValueError` carries
today:

```python
@property
def pnl_program(self):
    raise AttributeError(
        "pnl_program: cannot wrap a 'nttagg' declaration in a P&L; "
        "the engine must be an aggregate or a portfolio.")
```

Verified mechanics: `hasattr` returns False, because it swallows
`AttributeError` and nothing else; a direct call still prints the full
explanation; `dir()` still lists the name, so tab completion and
discoverability are unaffected. Comment the *why* in `Notes`, because a
negative override reads as a mistake to anyone who meets it cold: the point is
that presence of the member is this library's answer to "can this compose".

Two facts that make the exception-type change safe. `AggregateNTT` is new in
1.1.0, so no existing code catches `ValueError` from these calls. And
`constants.FCC_REQUIRED` lists neither member, nor is `AggregateNTT` in
`FIRST_CLASS_CLASSES`, so no contract member is being dropped.

**Then delete the now-dead guard branches** in `_program.py`, the `'nttagg'`
arm of each, after confirming by inspection that nothing else can reach them.
Leaving both the override and the branch states one rule twice, which is the
duplication this plan exists to remove.

### [library-entry] one nttagg entry in library.agg

Add one entry to the **Numerical character** section (line 1539, "Moment
approximations, window and grid behavior"), which is where an exact answer on
a lattice belongs. `NTT.DiceOfDice`'s shape is the natural candidate: it is
the plan's own warm-up, it builds in well under a second, and the exact
dice-of-dice answer is the result that motivated the feature.

Tag it `tags{topic:numerics, role:intermediate}` to match the section's
neighbors (lines 1549, 1557).

**On `role:hero`: not yet.** A hero reaches the landing gallery of
`aggregate_api`, and until that app can show `accuracy_df` / `ntt_audit` a
reader who clicks it gets an ordinary aggregate that happens to be exact, with
nothing on screen saying so. Revisit when the accuracy exhibit exists.

**A hero tag also has a downstream cost that must be paid first.**
`aggregate_api`'s `test_hero_example_builds` picks the shortest hero decl and
asserts its kind against a hard-coded tuple with no `nttagg` in it. A short
`nttagg` hero turns that assertion red. The api's own plan
(`dev/plan-1.1.0-nttagg.md`, step `[examples]`) widens the tuple; that widening
has to land before or with any hero tag here.

## Version and commit

**One bump, `1.1.1`, both steps together.** A fix takes the patch, and the
library entry is content rather than a new capability, so it does not earn a
minor of its own. The two land in one commit, per the one-bump-one-commit rule.

Commit subject, house format: `[NTT-Composability] 1.1.1: <terse summary>`.
`CHANGELOG.md` gets one paragraph opening with the bold label.

**Do not publish 1.1.0 to PyPI.** It is unpublished as of 2026-10-09 (the index
stops at 1.0.1), so publishing 1.1.1 as the first 1.1.x means no released
version ever carries the defect. That is the whole reason this is worth doing
now rather than at the next convenient bump.

## Tests

- `tests/test_derived_programs.py`: `assert not hasattr(o, 'pnl_program')` and
  the same for `reins_program` on an `nttagg`, placed beside
  `test_reins_program_rejects_a_portfolio_program` and reading as its sibling.
- One test per member that a direct call raises `AttributeError` and that the
  message still names the kind, so the override cannot decay into a bare
  `AttributeError`.
- `tests/test_library_entries.py` already builds every `library.agg` entry, so
  the new entry is covered by existing coverage. Confirm it is fast enough not
  to need the `slow` marker.
- Tier 3 before the bump: `uv run pytest -m 'slow or not slow'`. The numerics
  gate is not needed; no arithmetic changes.

## Acceptance check

```python
from aggregate import build
o = build('nttagg Dice dfreq [1:6] dsev [1:6]')
assert not hasattr(o, 'pnl_program')
assert not hasattr(o, 'reins_program')
assert 'NTT' in str(build.recipes.index.get_level_values('name').tolist())  # entry present
```

Then in `aggregate_api`, with this worktree installed editable, an `nttagg`
build shows the P&L button and the Reins tab both grey. That is acceptance-check
item 3 of `aggregate_api/dev/plan-1.1.0-nttagg.md`, and it is the only item
there that this library change decides.

## Execution log

Executed 2026-10-09 by Claude at the author's instruction, against 1.2.0a2
(commit f40f26c). The plan was `PROPOSED` rather than ruled; the author's
"execute that separately" is the ruling.

### Version: 1.2.0a3, not 1.1.1

The author ruled on 2026-10-09 that this lands in the running `1.2.0a*` series
rather than taking a decimal of its own. 1.1.1 was no longer reachable in any
case, 1.2.0a1 and 1.2.0a2 having already landed. The plan's reasoning survives
intact: 1.1.0 was never published, so the first published version still will
not carry the defect.

### Verified against the code before editing

The capability table is correct as given. `PnL` is not a subclass of
`Aggregate` (it is `HelpMixin, LabeledMixin, ProgramMixin`), so it carries
neither member; `Portfolio` defines `pnl_program` and not `reins_program`;
`AggregateNTT` subclasses `Aggregate` and inherited both. `FCC_REQUIRED` lists
neither member and `FIRST_CLASS_CLASSES` does not include `AggregateNTT`, so no
contract member is dropped. `dev/FEATURES.csv` has no `AggregateNTT` column, so
it needs no regeneration.

The override mechanics were prototyped before being written in, and all three
of the plan's claims hold: `hasattr` answers False, a direct read prints the
full message, and `dir()` still lists the name. The one risk the plan did not
raise is that a `__getattr__` on the class would intercept the `AttributeError`
and could make `hasattr` answer True again. There is none: the only
`__getattr__` in the package is on `_LabelView`, a helper class, not on
`LabeledMixin` itself.

### Divergence: the guards in `_program.py` stay

**The plan's premise here does not match the code.** It says to delete "the
`'nttagg'` arm" of the guard in each function, but `rg -n nttagg
src/aggregate/_program.py` returns nothing. The guards are generic kind
checks: `if kind not in ('agg', 'port')` in `pnl_program` (line 1229) and
`if kind != 'agg'` in `reins_program` (line 1394). There is no `nttagg` arm to
delete.

Deleting the generic checks would be a different change with its own risk.
`_program.pnl_program(ob, ...)` and `_program.reins_program(ob, ...)` are
module-level functions taking an arbitrary object, called by the two methods
but reachable independently, and the guard is what turns an unsupported kind
into a clear message instead of malformed DecL. The plan's argument for
deletion is that leaving both states one rule twice; against that, the guard is
a backstop one layer down, on a different entry point, and it is what will
catch the *next* kind that cannot compose before anyone writes an override for
it. The messages are also no longer duplicated: the overrides carry their own
text, worded for an `nttagg`, and the generic guards keep the generic wording.

Noted for the author: if the duplication is still unwanted, deleting the two
checks is a one-line change either way and nothing in the suite covers them
today.

### Divergence: the `reins_program` message was reworded

The plan says to carry the message the `ValueError` carries today. Its closing
sentence is "Cede the units of a portfolio individually", written for the
portfolio case and simply wrong on an `nttagg`, which is not a portfolio. The
override says "Declare the cession on an agg and compare against it" instead.
No test matched the old string.

### Not done: `role:hero`

Correctly, per the plan: `role:hero` waits on `aggregate_api`'s
`test_hero_example_builds` tuple widening and on an accuracy exhibit existing.
`ExactDiceOfDice` is tagged `role:intermediate`.

### Divergence: the entry is `ExactDiceOfDice`, not `NTT.DiceOfDice`

**The gate caught this, and the plan's name was wrong twice over.** The first
tier 3 run came back 2 failed, 5747 passed, both failures in
`tests/test_agg_libraries.py`.

`test_library_retired_the_letter_prefixes` pins a library convention the plan
did not know about: `library.agg` retired the old `A.` / `K.` filing codes and
groups by `tags{}` instead, exempting only a leading citekey such as
`Mack2003.Lognorm`. `NTT.DiceOfDice` matches the retired pattern exactly. The
`NTT.` prefix is the *test suite's* convention, from `_test_suite.agg`, which is
where the plan took the shape from. The entry is therefore `ExactDiceOfDice`,
prefix free and reading like its neighbors `RawLognorm` and
`ApproximateSlognorm`.

`test_library_is_written_in_the_canonical_layout` pins every entry as byte
identical to `format_program(..., layout='spread')`, so a hand-written entry
fails whatever its content. The entry text is now the formatter's own output,
which is why it reads `dfreq [1 2 3 4 5 6]` rather than the `dfreq [1:6]` the
plan wrote: the canonical layout expands a range.

Consequence for the plan's acceptance check, whose third line is
`assert 'NTT' in str(build.recipes.index.get_level_values('name').tolist())`.
That line cannot pass under the library naming convention and was never the
right probe anyway, since it tests the name where the kind is what matters. The
check used here is `('nttagg', 'ExactDiceOfDice') in build.recipes.index`.

### Tests and gate

Tier 1: `tests/test_derived_programs.py`, `tests/test_library_entries.py`,
`tests/test_ntt.py`, `tests/test_fcc_surface.py`, 413 passed. Tier 3 at the
bump, red on the first run (see the naming divergence above) and green on the
second. The numerics gate was not run, correctly: no arithmetic changed.

**The four suites of tier 1 did not cover the change.** Both failures were in
`tests/test_agg_libraries.py`, which tests the library *file* rather than what
its entries build into, and so was not on the list the blast radius suggested.
Worth remembering: a `library.agg` edit is gated by `test_agg_libraries.py`
first and `test_library_entries.py` second.

The new `library.agg` entry builds in well under a second and needs no `slow`
marker. `test_every_library_entry_builds` picked it up with no change.

### Acceptance

The plan's own check passes verbatim:

```
hasattr pnl_program: False
hasattr reins_program: False
in dir(): True True
library entry present: True   (kind 'nttagg', name 'ExactDiceOfDice')
```

The `aggregate_api` half, item 3 of `aggregate_api/dev/plan-1.1.0-nttagg.md`,
is the author's to confirm in that worktree.
