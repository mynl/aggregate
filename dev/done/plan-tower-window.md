# Plan: the tower's loss window, and a log reading for it

Status: **ruled and ready to execute**, 2026-09-25. The scope rulings below are
decisions taken with the author after eyeballing the shipped chart, not
proposals. Follows `dev/done/plan-reins-structure.md` (a349 to a351), which
built the `structure` chart; this plan changes only the loss axis it is drawn
against. Companion, not blocking: `aggregate-api` needs a small change to draw
a tower on log, described at the foot.

## Goal

Three defects the author found on first use of the shipped chart, all of them
in one function:

1. The loss axis runs **below zero**. Reinsurance applies to losses, which
   cannot be negative, and a tower whose axis is labeled `-2,000` says
   otherwise.
2. The gross and per-occurrence towers **stop short of the known limit**. A
   `100 xs 0` policy draws as a slab cropped at the 99.9th percentile with an
   open top, which asserts less cover than was bought.
3. A layer can be an unreadable **sliver**. `500 xs 500` against a `10000 xs 0`
   gross block is five percent of the panel height, and a real program is a
   dozen such layers.

The first two are the window. The third is not: a balanced program is layered
in a roughly geometric progression, `250 xs 250`, `500 xs 500`,
`1000 xs 1000`, `3000 xs 2000`, `5000 xs 5000`, whose bands are close to equal
height **on a log scale**. So the loss axis gains a log reading and the reader
chooses.

## Current behavior

`charts/_emit_structure.py::_stage_window` is the whole of it:

```python
hi = top
lo = 0.0
if reference is not None:
    window = loss_window(reference.q, 0.0)
    if window is not None:
        lo = min(lo, window[0])
        hi = max(hi, window[1])
if unlimited:
    hi = hi * (1.0 + HEADROOM) if hi > 0 else 1.0
return pad_window(lo, hi) or (lo, max(hi, lo + 1.0))
```

and the axis it feeds declares the same pair twice:

```python
axes.append(ChartAxis(
    id=loss_id, label=STAGE_LOSS[stage], unit='currency',
    suggested_range=window, full_range=window))
```

**The negative floor is a double padding.** `loss_window` returns a window that
has *already* been through `pad_window`, so its lower end is
`-WINDOW_PAD * q(0.999)` rather than zero; `lo = min(0.0, window[0])` picks
that up, and `pad_window` then pads the whole thing again. On
`agg 5 claims 100 xs 0 sev lognorm 10 cv .75 occurrence net of 15 xs 5 poisson`
the emitted range is `(-2.5723125, 65.6191875)`, and with
`WINDOW_PAD = 0.02` and `q(0.999)` of the gross severity near `63.0` that
lower end is `-0.02 * q * (2 + 2 * 0.02)`, the pad applied twice. Nothing about
the padding is wrong in itself; padding *below a quantity's own lower bound*
is.

**The top is a quantile of the law, not the contract.** On that same program
the severity is limited at 100 and the window stops at 65.6, so `_gross_block`
computes `cropped = np.isfinite(limit) and attach + limit > top` as true and
draws the policy slab open topped at 65.6.

**Declaring the same pair twice disarms the consumer's own guard.** The SPA's
`axisOption` clamps `suggested_range` to `full_range` before rounding, and its
docstring says it exists for precisely this ("an outcome axis whose data starts
at zero arrives as `[-173.18, 8832.18]` ... rounding that outward turns 173
units of padding into a 2,000 unit margin of empty axis below zero, labeled
`-2,000`, on a quantity that cannot be negative"). It cannot fire while the two
ranges are identical.

**The loss axis declares `('linear',)`**, so neither renderer offers a log
reading of it and there is no escape from a sliver.

## Scope rulings (settled with the author, do not reopen)

- **The window reaches the known limit, unconditionally.** Not capped at some
  multiple of the tower top, and no fallback to the quantile slice when the
  limit is far above the program. The gross column is the policy, and a broker
  slide draws it whole. The sliver this can produce is what the log reading is
  for.
- **Log is an option, not the default.** The axis declares both scales and the
  reader presses. The SPA holds panel readings per browser, so one press
  sticks.
- **`tail_behavior_df` is the source of the bounds.** The author named it ("the
  agg object has a report on its bounds"), it already composes both of the two
  ways a severity can be bounded, and it is already public: `aggregate-api`
  serves it behind the More / Tail behavior leaf. No new accessor.
- **No `CHART_IR_VERSION` bump.** `scales`, `suggested_range` and `full_range`
  are existing fields at their existing meanings. A reader that ignores the new
  `scales` entry draws the default linear reading correctly, which is the
  standing test for whether the version moves.
- **The Lee companion's default reading is out of scope.** Whether the curve
  beside a tower should arrive on the return-period reading rather than on the
  probability one is a real question and is deferred; it is noted in
  `aggregate-api`'s own follow-up.

## Design

### The support bounds

New private helper beside `_reference`:

```python
def _support(agg, stage):
    """``(lo, hi)`` of the quantity one stage's tower bands."""
```

`tail_behavior_df` carries `min` and `max` per component, and works on an
un-updated object, which matters because the structure chart is available
before `update()`. Verified against `1.0.0a351`:

| program | rows | occ `max` | agg `max` |
|---|---|---|---|
| `100 xs 0 sev lognorm 10 cv .75`, poisson | `comp0` | 100.0 | inf |
| `sev 40 * uniform`, poisson | `comp0` | 40.0 | inf |
| `[50 200] xs [0 0]` two components | `comp0`, `comp1` | 200.0 | inf |
| `dfreq [1 2 3] sev 40 * uniform` | `comp0` | 40.0 | **120.0** |
| `sev lognorm 10 cv .75`, poisson | `comp0` | inf | inf |

- **The occurrence stage** is read against the gross severity
  (`_reference` uses `p_sev_gross`), so its bound is the **max over the
  `comp*` rows**. Not the `severity` row, which is absent when there is one
  component, and not `severity (net occ)`, which is the net rather than the
  gross. Both spellings of a bounded severity land here: an explicit `xs`
  clause (`comp0` reads `capped at 100` in its `note`) and a distribution
  bounded in itself (`40 * uniform`), which is what the author asked for.
- **The aggregate stage** is the `aggregate` row. It is `inf` under any
  unbounded frequency, which is the author's "the agg should obvs go higher"
  falling out of one rule rather than needing a case: the aggregate acquires a
  ceiling only when the frequency is bounded too, and the fourth row above
  shows it doing exactly that at `3 * 40 = 120`.
- Defensive: a missing frame or a missing row answers `(0.0, inf)`, which is
  today's behavior.

### The window rule

`_stage_window` takes the support and applies it at both ends:

```
lo = support_lo                      # 0.0 for a loss, and never padded past
hi = support_hi                      if finite
   = union(top, q(0.999) slice)      otherwise, as today,
                                     times (1 + HEADROOM) if the top layer
                                     is unlimited
```

The union of tower and law survives **only in the unbounded case**, where it is
still the right answer for the reason its docstring gives: taking the law alone
crops a high layer that rarely attaches out of its own picture, and taking the
tower alone puts a modest program against a book whose tail runs far past it.
Where the support is finite there is nothing to trade off, because the contract
answers.

Padding applies to the **top only**. `pad_window` pads symmetrically, so this
plan does not call it on the pair; pad `hi` and leave `lo` alone.

**The blocks take the unpadded top, the axis takes the padded one.**
`_gross_block` and `_tower_blocks` are both handed `window[1]` today and use it
as the ceiling a cropped or unlimited block is drawn to. Handing them the
padded number would end the top retention two percent above the policy limit,
which is invisible once the axis is clamped but is a block asserting cover that
does not exist. Keep the two apart: `_stage_window` returns the pair the axis
is built from, and the block builders take the bare `hi`.

### The two ranges

```python
suggested_range = (lo, hi_padded)
full_range      = (lo, support_hi)   only when support_hi is finite
```

Omitted rather than faked when the support top is infinite, because there is no
honest number to put there and an axis declaring no extent is a thing both
renderers already handle.

This is what makes the padding harmless: a consumer clamps the suggested window
to the declared extent, so the drawn axis lands exactly on `100` rather than on
`102` and the policy limit is a tick rather than a number near one. The
matplotlib renderer's `_axis_window` should take the same clamp if it does not
already; check before assuming.

### The log scale

```python
ChartAxis(id=loss_id, ..., scales=('linear', 'log'), ...)
```

`AXIS_SCALES` already admits both and `ChartAxis.__post_init__` validates only
membership, with no constraint tying a scale to a unit, so this is a one-word
change. `scales` is in `_ALWAYS`, so every structure document's canonical bytes
and hash move with it; that is expected and is the only fallout.

Two consequences that are **not** free, both about the bottom of the axis:

- **A block that starts at zero has no bottom on a log axis.** The retention
  block runs `0` to the first attachment and the gross slab runs `0` to the
  limit. `_render_tower_panel` already computes a decade floor and passes it to
  `_apply_axis`, so the axis is fine; what needs checking is that a rectangle
  drawn from `0` is clamped to that floor rather than sent to `-inf`.
- **A boundary tick at zero has no position.** `_render_tower_panel` promotes
  the panel's marks to y ticks verbatim, and an aggregate program written
  `20 xs 0` emits a mark at `0`. Drop non-positive ticks under a log reading.

## Files

- `src/aggregate/charts/_emit_structure.py`: `_support`, the rewritten
  `_stage_window`, the two ranges and `scales` on the loss axis.
- `src/aggregate/plots/_chartdoc.py`: the two log-bottom cases above, in
  `_render_tower_panel`.
- `tests/test_chart_structure.py`: see acceptance. Existing assertions about
  the emitted window move with the rule.
- `tests/` image baseline: `structure.png` regenerates. Read the diff
  deliberately rather than accepting it; a closed gross slab and an axis
  starting at zero are what should have changed, and nothing else.
- `dev/TODO.md`: the `[Reinsurance-Structure-Diagrams]` row (#45) is marked
  done at a349 to a351. Add this work to it rather than reopening it.
- `CHANGELOG.md` and the version, per house rules. No `docs/` change: the IR
  version does not move. Check `docs/3_reference/3_x_Charts.rst` for prose
  about the structure chart's window that would now be wrong.

## Acceptance

- **No loss axis starts below zero**, on every program in the existing
  structure test set. Assert on the emitted `suggested_range[0]`, which is the
  defect as the reader sees it.
- `100 xs 0 sev lognorm 10 cv .75` occurrence program: the occurrence loss axis
  has `full_range == (0.0, 100.0)`, and the gross block is `y1 == 100.0` with
  `open_top` **false**. That block is open topped today and its closing is the
  visible half of the fix.
- `sev 40 * uniform` with no limit clause: the same, at 40. This is the case an
  `exp_limit` read alone would miss.
- `dfreq [1 2 3] sev 40 * uniform aggregate net of 30 xs 30`: the **aggregate**
  loss axis reaches 120, because a bounded frequency over a bounded severity is
  bounded. The occurrence axis, were there an occurrence stage, would stop at
  40.
- An unbounded severity under Poisson: no `full_range` on either loss axis, and
  the suggested window is the union rule's answer, unchanged from a351 except
  that it starts at zero.
- Both loss axes declare `('linear', 'log')`, on every tier.
- `canonical_json` round trips and `doc_hash` is stable across a reload, as
  before. Do **not** assert hash equality against a pre-change document: the
  bytes move, by design.
- Image-gated matplotlib render of one bounded program on `log='y'`, alongside
  the existing `structure.png`: no rectangle collapses to the bottom of the
  frame and no tick is drawn at zero.

## The companion change in `aggregate-api`

Not blocking and not this plan's, recorded so the other side is not surprised.

The SPA's tower realizer already resolves `axisScale(yAxis, view.logY)` and
already passes a decade floor to `axisOption`, and `readings()` offers the
`log y` button off the axis' declared `scales`, so the control appears with no
app change at all. What it does not yet do is clamp a block's own rectangle to
that floor: `renderItem` sends `apiRef.coord([b.x0, b.y0])` with `b.y0 == 0`
for the retention and the gross slab. It also needs the same non-positive tick
rule as matplotlib, since it realizes the boundary marks as
`axisLabel.customValues`.

Two other items are already queued on that side and are mentioned only so the
sequencing is visible: a zigzag top on an `open_top` block, which is worth
having **after** this plan lands (today almost every gross slab is open topped
because the window crops it, and a zigzag would fire on all of them), and
dropping the reading group from the Lee companion panels.

## Execution log

Executed 2026-09-26 against `1.0.0a351`. Two phases, `[Tower-Loss-Window]` at
a352 (the emitter) and `[Tower-Log-Reading]` at a353 (the renderer). Every plan
fact was verified against the code before starting and all of them held,
including the quoted `(-2.5723125, 65.6191875)` window and all five rows of the
`tail_behavior_df` table.

### Rulings taken before starting

Two questions were put to the author and answered, both taking the
recommendation.

1. **The `_axis_window` clamp is skipped.** The plan asked for the
   suggested-to-full clamp in the matplotlib renderer "if it does not already";
   it does not. Adding it is global, and the `agg` chart's `outcome` axis is
   suggested `(-213.62, 10894.62)` against full `(0.0, 65535.0)`, so the clamp
   would have moved `tests/data/chartdoc_baselines/agg.png`, an approved picture
   outside this plan's scope. It also buys almost nothing here, because
   `_apply_axis` adds `axes.ymargin` *after* the window, so a clamped `(0, 100)`
   still draws the frame to about `(-5, 105)`, and a tower panel labels no tick
   there since its ticks come from the marks. The SPA's own `axisOption` clamp
   is where the plan's quoted rationale lives, and it now fires because the two
   ranges differ.
2. **Label fit is made log aware**, which is a third change in
   `_render_tower_panel` beyond the two the plan names. See the a353 notes.

### Divergences

- **`lo` is `min(0, support_lo)`, not `support_lo`.** A positive lower support
  bound is real: `dsev [10 20 30]` reports `comp0 min == 10.0`. Taking that as
  the floor would have cropped the retention band, which runs from zero to the
  first attachment whatever the severity's support. `min(0, support_lo)` is `0`
  for every loss and still honors a genuinely signed severity, which is the
  plan's stated intent. Pinned by
  `test_a_positive_lower_bound_does_not_crop_the_retention`.
- **The window top is `max(support_hi, top)`, not `support_hi` alone.** A tower
  can be written above its own severity's ceiling (`occurrence net of 200 xs 0`
  on a severity capped at 100), and a window that cropped it would draw a
  program the object does not have, which is the invariant
  `test_the_window_always_contains_the_whole_tower` already asserts. Pinned by
  `test_a_tower_written_above_its_own_ceiling_is_not_cropped`.
- **`_gross_block` takes the support too, which the plan's Files section did not
  name.** Its acceptance requires `sev 40 * uniform` to draw closed at 40, and
  the slab's `open_top` was computed from `exp_limit` alone, which is `inf`
  there. The rule is now that either bound closes the slab, `ceiling =
  min(attach + limit, support_hi)`, and it is open only when nothing bounds it
  or when what bounds it sits above the drawn window. This also closes the
  aggregate slab on `dfreq [1 2 3] sev 40 * uniform` at 120.
- **Layer blocks keep their declared `open_top` and were deliberately left
  alone.** An `inf xs 60` layer under a support of 100 now draws to 100 but
  still declares an open top. The band is the *contract*, and the contract is
  unlimited; the gross slab is different because it is the subject quantity
  itself, which the support genuinely bounds. The plan is silent here and the
  API's planned zigzag on `open_top` reads the same way.
- **`CS.Gapped` is exercised on the declared tier only in the whole-set
  invariants.** Building it walks `reins_stats_df` into its `0% po` layer,
  whose ceded second moment lands as float noise below zero, so
  `moments.mcvsk` takes the square root of a negative number and the layer
  reports `cv` and `skew` as `NaN` beside a mean of `-1.8e-11`. That is a real
  pre-existing defect with nothing to do with the loss window: no test had
  built this program before, so it was latent. It is tracked as
  `[Zero-Share-Layer-Moments]` in `dev/TODO.md` rather than fixed here, and the
  exclusion keeps `-W error::RuntimeWarning` green. The window rule is tier
  independent on this program (its support is bounded at 100 either way), so
  nothing about the window goes unchecked.
- **The a350 `CS.*` programs are still absent from `decl-testers.agg`.** The
  corpus header says each pytest DecL program is mirrored there; the structure
  chart's original fixtures were not. This plan's four new programs were added
  under a new `TW.` section, and the a350 gap is left for the author rather than
  backfilled here.

### Acceptance, measured

| criterion | result |
|---|---|
| no loss axis starts below zero, every program, both tiers | `suggested_range[0] == 0.0` throughout |
| `100 xs 0 sev lognorm 10 cv .75` occurrence | `full_range == (0.0, 100.0)`, slab `y1 == 100.0`, `open_top` false |
| `sev 40 * uniform`, no limit clause | `full_range == (0.0, 40.0)`, slab `y1 == 40.0`, `open_top` false |
| `dfreq [1 2 3] sev 40 * uniform aggregate net of 30 xs 30` | aggregate `full_range == (0.0, 120.0)` |
| unbounded severity under Poisson | `full_range is None` on both loss axes, union rule's top, starting at zero |
| both loss axes declare `('linear', 'log')` | every program, every tier |
| `canonical_json` round trips, `doc_hash` stable across a reload | unchanged, by the existing tests |

### a353, the renderer, and what the log reading turned out to need

The plan named two log cases in `_render_tower_panel`. Both were real and both
are done: a block starting at zero is clamped to the decade floor, and a
non-positive boundary mark is dropped rather than placed. The author's ruling
added a third, label fit measured in decades rather than in loss units. Drawing
it then surfaced two more, neither anticipated and both blocking a correct
picture rather than cosmetic.

- **`_decade_floor` returned the smallest positive value itself when that value
  was an exact power of ten.** Its docstring already said "a round decade
  **under** the smallest thing actually drawn", so this is the documented
  intent rather than a new rule. It bites the tower immediately: the gross
  panel holds one slab from 0 to 100, whose only positive coordinate is 100, so
  the floor came back as 100 and the slab collapsed to the top of the frame,
  which is the plan's own acceptance criterion failing. Now
  `10 ** (ceil(log10(v)) - 1)`, unchanged wherever the smallest value is not an
  exact decade, which is every continuous series.
- **The floor was computed per panel, and it has to be per quantity axis.** A
  structure document puts the gross slab, the tower and the Lee curve on one
  loss axis and means them as one reading: the faint boundary rules exist to
  carry a number across. Each panel saw only its own content, so the three
  disagreed, and the picture showed a slab whose foot floated a decade above
  the tower beside it. It also broke the clamp: `sharey` is on for a
  single-axis document and takes the last panel's limits for all of them, so a
  panel drawn early clamped its blocks to a bottom the figure did not end up
  with. `_tower_floors` now computes one floor per axis before anything is
  drawn, and both renderers take it.

  **Pooled over the blocks, not over everything drawn against the axis.** This
  is a judgment, and it is the one place the picture could reasonably have gone
  another way. A Lee curve on a loss axis runs down to the first positive grid
  point, three or four decades under the program; a floor taken from that opens
  those decades under every tower and squeezes the bands back into slivers,
  which is the exact pathology the log reading exists to cure and which
  `_decade_floor`'s own docstring warns about. So the tower's breaks set the
  scale and the curve runs off the bottom of the frame, as a curve on a log
  loss axis always does. If the author wants the curve whole instead, the
  change is one line in `_tower_floors`.

### Left for the author

- **Look at `tests/data/chartdoc_baselines/structure_log.png`.** It is a new
  approved picture and nothing but review approves it. `structure.png` also
  moved at a352, and the deliberate differences there are a closed gross slab
  and an occurrence axis reaching 100; the `5 xs 0` retention band lost its
  terms line to the taller window, which is the sliver the log reading answers.
- **`[Zero-Share-Layer-Moments]`**, filed in `dev/TODO.md`. A `0% po` layer
  makes `reins_stats_df` report `cv` and `skew` as NaN. Pre-existing, latent
  until a test built the gapped program.
- **The a350 `CS.*` programs are still missing from `decl-testers.agg`**, whose
  header says every pytest DecL program is mirrored there. This plan's four were
  added; the earlier ones were not backfilled.
- **`docs/3_reference/3_x_Charts.rst` was checked and needs nothing.** Its
  paragraph on the structure chart says nothing about the window, so no prose
  went stale. A sentence about the log reading would be an addition rather than
  a correction, and is left to the author.
- **The companion change in `aggregate-api` is unblocked**, and the plan's own
  description of it still holds: the SPA needs the same zero clamp on a block's
  rectangle and the same non-positive tick rule. Worth adding from this side's
  experience: it also needs the floor to be one number per axis rather than one
  per panel, or its tower and Lee panels will disagree the way matplotlib's did.
