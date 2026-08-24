# Plan: [Picks-Grid-And-Reference-Trailers]

Status: drafted 2026-08-24 from that day's bug investigation, against `1.0.0a316`.
The author ruled the one design question the same day: a hard error for picks
attachments off the grid, no snapping, and one plan covering both defects. No
code written yet.

## 1. The reported crash

The program

```
agg LayerPicks
  1 claim
  sev lognorm 100 cv 2 picks [100 200 500] [45 20 25]
  fixed
```

dies with `KeyError: np.float64(100.0)` from `_picks_work`
(`_aggregate.py:262`), which reads the survival function with an exact float
label lookup, `density.loc[x, 'S']`. The windowed sizer chooses `bs=8`,
`log2=16` for this severity (a defensible grid: the update validates
NOT_UNREASONABLE, empirical severity mean 99.996 against 100, cv 2.0002
against 2), and 100/8 = 12.5, so the attachment is not a grid point and the
lookup raises the raw pandas error.

The constraint is legacy, not a picks regression: the lookup line predates the
move to `src/`. It never fired historically because the legacy moment sizer
chose small dyadic buckets (about 1/32 here), and every round number is a
multiple of a small power of two. The windowed sizer's coarse, tail covering
buckets expose it. `DW.Picks` in `decl-testers.agg` still passes by luck: it
draws `bs=2`, which divides 1000, 5000 and 10000.

The contrast that frames the fix: the identical tower as reinsurance,
`occurrence ceded to tower [0 100 200 500]`, runs at the same `bs=8` and gives
the sensible 52.5 / 17.7 / 17.7 layer picture, because `apply_reins_work`
(`_reinsurance.py:182`) evaluates piecewise linear ceder and netter functions
at every grid point and rebuckets the off grid results back onto the lattice.
Reinsurance never needs exact alignment; picks is the one consumer that does.

Verified workaround, kept as a test: `build(..., bs=4)` (any bucket dividing
the attachments) builds the program and hits the first pick at exactly 45.000.

## 2. The second defect, found checking the shipped examples

A builtin aggregate reference drops the referenced entry's stored trailer.
`build('agg.MED.WithPicks')` crashes with the same KeyError even though the
library entry pins `hints{bs=125; log2=18}`, under which 100000/125 = 800 is
exactly on grid.

Mechanism: `trailer()` (`parser.py:2132`) always seeds
`{"note": "", "hints": ""}`, and `agg_out_builtin` (`parser.py:628`) merges
`{**bagg, **agg_reins, **trailer}`, where `bagg` is the deep copied stored
spec. The seeded empty strings overwrite the stored note and hints. Tags
survive because the trailer dict has no tags default; that asymmetry marks the
clobber as accidental rather than designed.

Blast radius: `library.agg` carries 16 `hints{}` trailers and every one is
silently ignored on a dotted reference build. `agg.MED.Exposure` builds at
`bs=40000` instead of its pinned 125, quietly, on the wrong grid. Only
`MED.WithPicks` is loud about it, because picks hard fails off grid. Three
paths are NOT affected, which is why nothing noticed:

1. The bare name path, `build('MED.WithPicks')`, resolves the stored hints
   correctly (builds at `bs=125`, note intact). `test_library_entries.py`
   builds entries this way, so the suite stays green.
2. The severity reference path, `sev agg.X`, reads the stored hints explicitly
   and requires them (`underwriter.py:1806` and the hygiene message at
   `underwriter.py:1908`).
3. An explicit outer trailer wins correctly today:
   `agg X agg.MED.WithPicks hints{bs=125; log2=18}` builds fine. Only the
   *absent* outer trailer clobbers.

## 3. The ruling

Author, 2026-08-24: off grid picks are a hard error, not a snap. Snapping
silently moves a stated layer boundary (100 becomes 96 or 104 at `bs=8`),
which changes meaning; the purist stance is that the user restates the grid,
not that the library restates the tower. The sizer stays picks blind: an off
grid result under auto sizing is an error naming a compatible bucket, not an
input to `_bs_window`. Precedent for an error that computes its own fix:
`_sev_ref_hygiene_message` (`underwriter.py:1908`).

## 4. Phases

Order is A then B. After A alone, `agg.MED.WithPicks` builds end to end; B
turns the remaining raw KeyError class into a named error. Each phase bumps
the version and lands as its own commit, per the house rule.

### Phase A [Reference-Trailer-Preservation]

In `agg_out_builtin`, merge only the trailer keys that carry a value:

```python
{**bagg, **agg_reins, **{k: v for k, v in trailer.items() if v}}
```

Outer wins where written, stored value survives where not. Every other
`**trailer` splat site in `parser.py` builds its spec fresh from program text,
where the seeded empties are the correct defaults; all are untouched. Port
units written as builtin references ride through `agg_out_builtin`, so the
same edit covers them.

Accepted limitation: if the grammar admits an empty clause such as `note{}`,
writing it on a reference no longer blanks the stored note. There is no known
use for that spelling.

Tests, in `tests/` near the existing recipe and library coverage:

- a dotted reference preserves the stored trailer: build `agg.MED.Exposure`
  and assert `bs == 125`, `log2 == 18`, and a non empty `spec['note']`;
- the outer trailer still overrides: `agg X agg.MED.Exposure hints{bs=250;
  log2=17}` yields `bs == 250`;
- `agg.MED.WithPicks` builds end to end (this also exercises picks on a non
  dyadic pinned grid, 125).

Prefer a self contained scratch recipe over the MED names if the test harness
makes registering one easy, to decouple from library churn; otherwise pin on
MED and accept the coupling, which `test_library_entries.py` already has.

Snapshot: re-run `tests/capture_spec_snapshot.py` and read the diff. The
corpus `_test_suite.agg` contains only `sev agg.X` severity references (three
of them), none through `agg_out_builtin`, so the expectation is **no
movement**; any moved line is a finding.

CHANGELOG, one paragraph, and it carries a moved number warning: any program
referencing a hinted library entry by `agg.NAME` now builds on the entry's
pinned grid, so grids and densities move for those programs, deliberately.

### Phase B [Picks-Off-Grid-Error]

At the top of `_picks_work` (`_aggregate.py:203`), before any frame work,
validate the attachments against the realized grid:

- each attachment maps to an integral index: with `i = (a - xs[0]) / bs`,
  reject when `abs(i - round(i)) > 1e-8 * max(1.0, abs(i))`;
- each attachment lies inside the window: `a <= xs[-1]`. This check also
  covers an infinite attachment, which today reaches the same raw KeyError
  through the unguarded fifth element of the `layers.loc` row assignment.

On failure raise `ValueError` listing every offending attachment, the realized
`bs` and window top, and a computed suggestion: the largest `b = bs / 2**j`
(j >= 0, floor around `bs / 2**20`) with every attachment an integral multiple
of `b`, when one exists (for the motivating program, `bs=8` suggests 4). When
none exists in range, the message says to pin `bs` to a common divisor of the
attachments instead, and names `hints{bs=...}` as the DecL spelling for a
library entry, with the MED entries as the model. Validating in `_picks_work`
rather than in `Aggregate.picks` gates every caller, including direct use.

Then convert the three label lookups to positional access using the validated
indices: the two `density.loc[0:x-bs, ...]` prefix aggregations become
`iloc[:i]` slices and `density.loc[x, 'S']` becomes `.iloc[i]`. This is
hardening rather than a second bug fix: exact float labels are one
representation drift away from a spurious KeyError on a non dyadic pinned
grid (`bs=0.1` works today by rounding luck), and after validation the index
is the honest coordinate.

Docstrings: the constraint goes in the Notes of both `Aggregate.picks`
(`_aggregate.py:4341`) and `_picks_work`, stating that attachments must lie on
the realized grid, that the update raises a ValueError naming a compatible
bucket, and why reinsurance towers are exempt (they rebucket; picks defines
the layers the reweighting is solved on, so a boundary inside a bucket has no
faithful reading).

Docs: one sentence where picks is documented (locate by grep across
`docs/2_aggregate_overview` and `docs/4_agg_language_reference` at execution;
`.rst` edits in lockstep, build deferred to the author per house rule).

Tests:

- pinned `bs=8` with `picks [100 200 500] [45 20 25]` raises ValueError and
  the message names `100`, the bucket, and the suggestion 4 (pin the bucket in
  the test so auto sizing drift can never silently skip the case);
- pinned `bs=4` builds and the bottom layer pick lands at 45.000 (verified by
  hand 2026-08-24, gross mean moves to 102.03 as picks intend);
- the existing `test_layer_picks_reproduces_every_pick` passes unchanged;
- a non dyadic pinned grid still builds: `bs=125` via the phase A reference
  test, plus `bs=0.1` inline with attachments [100 200 500].

CHANGELOG, one paragraph: a program whose picks attachments miss the grid now
gets a ValueError naming the offending attachment and a compatible bucket,
where it previously died with a raw pandas KeyError. Programs that built
before build identically; no numbers move.

## 5. What moves

- Phase A: grids (hence all downstream numbers) for programs that reference a
  hinted library entry by `agg.NAME`. Each such move is a correction to the
  grid the entry's author pinned. No test fixture in the repo is known to be
  affected; confirm at execution.
- Phase B: nothing that builds today changes; a class of crashes becomes a
  named error.
- API repo: no action owed. No grammar change (`decl-keywords.json`
  untouched), the new error is at update time and rides the existing error
  path, and no exhibit or chart registry moves.

## 6. What this plan does not touch

The windowed bucket sizer. Making `_bs_window` picks aware (constraining `bs`
to divide the attachments) was considered and set aside: it couples the sizer
to a niche clause, cannot help non dyadic attachment sets anyway, and the
error message now hands the user the same bucket the sizer would have had to
find. The `a[0] > 0` precondition documented on `_picks_work` is also left as
is; making it a checked error can ride along in phase B if it falls out
naturally, but it is not scope.

## 7. Decisions

None open. The author ruled the hard error and the single plan on 2026-08-24.
Two accepted limitations are recorded above: an explicit empty trailer clause
on a reference becomes a no-op (phase A), and the sizer stays picks blind
(section 3).

## 8. Execution log

Executed 2026-08-24 against a moving tree: the session opened at `1.0.0a316` and the author landed `a317` plus two `library.agg` edits while phase A was in hand, so the phases are `a318` and `a319` rather than the `a317`/`a318` the plan assumed.

### Phase A, `[Reference-Trailer-Preservation]`, `1.0.0a318`

The edit is the plan's, a filtered merge in `agg_out_builtin`, with a `Notes` docstring on the method recording why the trailer is seeded and what the filter is protecting. `agg.MED.WithPicks` builds end to end at its pinned `bs=125`, and the snapshot regenerated to a zero diff exactly as section 4 predicted.

Divergences:

1. **The plan's outer-override test exercises a different rule than the one being fixed.** Section 4 phase A writes it as `agg X agg.MED.Exposure hints{bs=250; log2=17}`. That spelling is the *rename* form, `agg_body_rename` wrapped by `agg_out_named`, and never reaches `agg_out_builtin`. The override is tested on the true builtin path instead, `agg.NAME hints{...}`, which is the statement shape the fix touches.

2. **Tests are built on a scratch recipe, not the MED names**, taking the escape clause the plan offers. `build.fork()` (a302) registers `AE.Trailer.RefSource` in a private recipe base, which decouples the semantics from library churn; the author was actively renaming `library.agg` entries during execution, so the coupling the plan was willing to accept would have been a live hazard. The one MED case kept is the end to end `agg.MED.WithPicks` build, which is the reported defect itself.

3. **A new file, `tests/test_reference_trailer.py`, rather than additions to `test_library_entries.py`**, which carried the author's uncommitted edits throughout.

4. **Each test takes its own fork.** Not tidiness: a reference carrying an explicit outer `hints{}` legitimately stores those hints back, so a second case reading the same name in one session would see the first case's grid. That is `[Session-Build-Clobbers-The-Trailer]`, a separate open item.

5. **Only the source entry is mirrored into `decl-testers.agg`.** The reference statements are not, because the unparser resolves `agg.NAME` back to the full spec, so they are not round-trip fixed points and `test_roundtrip` would fail on them (`[Unparser-Reference-Gaps]`).

Findings the plan did not anticipate:

6. **The clobber had a second life, and phase A closes it too.** The reference's spec was re-registered under the same name, so a blanked trailer was written *back* into the recipe base and every later build of that entry by bare name in the same session also came out on the wrong grid. The plan's section 2 claim that the bare name path resolves correctly holds only in a session where no reference was built first. Pinned by `test_building_a_reference_leaves_the_stored_entry_alone`.

7. **The rename form still drops the stored trailer, and is left alone.** Section 4 phase A states that every other `**trailer` splat site builds its spec fresh from program text. That does not hold for `agg_out_named` when its body came from `agg_body_rename`, which is a copy of a stored spec: `agg X agg.SRC` yields the auto sized grid and an empty note. Arguably correct for `note`, which describes the source rather than the copy, and hard to defend for `hints`. Out of the plan's scope and left for the author, recorded in the `a318` CHANGELOG entry so it is not silently absorbed.

Verification. Tier 3, `pytest -m 'slow or not slow'`, gave 24 failed and 4937 passed. Twenty are the pre-existing library reorganization failures recorded before execution began; the other four were checked individually with the change reverted, and three reproduce without it while `test_mv_explain_flags_clipped_book` passes serially and failed only under xdist memory pressure, the documented bivariate grouping hazard. No failure is attributable to this change. The suite could not be brought to green because the author's `library.agg` rename was in flight in the working tree.

### Phase B, `[Picks-Off-Grid-Error]`, `1.0.0a319`

The guard is the plan's, moved into two named helpers above `_picks_work` rather than written inline: `_picks_compatible_bucket` computes the suggestion and `_picks_grid_indices` validates and returns the indices the integrals then use. The plan's tolerance, `abs(i - round(i)) > 1e-8 * max(1.0, abs(i))`, and its search floor of `bs / 2**20` are both as specified, named `_PICKS_GRID_RTOL` and `_PICKS_BUCKET_HALVINGS`. On the motivating program the message names 100 and 500, omits 200 (which is a multiple of 8), and suggests `bs=4`, which is what section 4 predicted.

Divergences:

8. **The prefix slices are anchored on the position of label zero, not on position zero.** The plan converts `density.loc[0:x-bs, ...]` to `iloc[:i]`. Those agree only when `xs[0] == 0`, because label slicing starts at the *label* 0 wherever that sits. Severity grids do start at zero, so the plan's form would have been correct in practice, but the code computes `zero_index` from `xs[0]` so the replacement is faithful on any grid rather than faithful by assumption. Verified bit for bit: across the seven layer tower of `tests/test_picks.py` all three integrals agree with the label form to `0.0` absolute difference.

9. **`x * density.loc[x, 'S'] if x < np.inf else 0.0` loses its conditional.** The guard now rejects an infinite attachment, so the branch is unreachable. It had never worked anyway: the same line's second lookup, `density.loc[x, 'S']`, raised on `inf` before the conditional could help.

10. **Two failure kinds get two sentences, not one list.** The plan folds the window check into the same message. Off grid and above the window want different advice, a finer bucket against a longer grid, so the message names them separately and only computes a bucket suggestion for the off grid case.

11. **The documentation sentence went to `docs/2_aggregate_overview/pipeline-aggregate.rst`**, at the pipeline step that performs the reweighting. The plan expected to locate it by grep across `2_aggregate_overview` and `4_agg_language_reference`; the language reference page is generated from the grammar and is not hand edited, and no prose section documents the picks clause itself.

12. **`DW.PicksOffGrid` is mirrored into `decl-testers.agg` as a parse-only fixture.** It is the first corpus entry that deliberately raises on build under auto sizing. That is consistent with the file, which already does not load clean, and the corpus tests parse and round-trip rather than build; the comment beside it says so, so a future build-everything test knows it is deliberate.

Verification. `tests/test_picks.py` grows six cases, all pinning `bs` so auto sizer drift cannot silently stop exercising the guard, and the three existing cases pass unchanged. Snapshot regenerated to a zero diff.
