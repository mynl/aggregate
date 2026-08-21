# Plan [Chart-2D-Punchups]: what the library owes the 2-D control strip

Written 2026-08-21 against `aggregate` `1.0.0a308`, from the app-side plan
`aggregate_api\dev\plan-2d-punchups.md`. That plan is canonical for the whole
change and holds the evidence, the app parts A1 to A6 and the clutter count.
This is the LIB half stated as requirements against the library, the
`plan-chart-reflect` arrangement: a copy in each repo rather than a symlink,
because each side owns its own half and neither reads as a diff of the other.

**Status: EXECUTED at `1.0.0a314`, 2026-08-21.** L1, L2 and L3 all landed; see
the execution log at the foot of this file for the seven divergences, three of
which are author rulings taken during the review. The API half has not started.
The status this plan was executed from: specified and ruled, with the author
dispositioning both open questions on 2026-08-21, so the numbers below are
rulings rather than proposals.

**Order of work.** LIB first and in full, then the app. The app half is written
to be correct before and after these parts, but its verification cannot
distinguish "the declared window is honored" from "the declared window was
never wrong" until L1 has moved one. So these land, `aggregate` bumps, the API
runs `uv sync --extra dev`, and only then does the app half start.

## Why the library is being asked at all

The app draws what it is served and holds no windows of its own, so a
return-period axis that reads badly is a question about what the axis declares.
Two of the three parts below are the emitter declaring something it already
knows and had decided not to publish. None of them is a new capability: `scales`,
`suggested_range` and `full_range` are all shipped `ChartAxis` fields, and every
change here is a different value in an existing one.

What the app cannot do, and why these cannot be worked around downstream: the
window a return-period ladder should end at and the extent an unlimited program
reaches are both properties of the distribution, known in the emitter and
nowhere else. Inventing either app side would be the app building meaning, which
the purist ruling forbids.

## L1. `return_period` stops defaulting to log, and declares its full extent

Two emitters, the same axis: `src\aggregate\charts\_emit_aggregate.py:138` and
`src\aggregate\charts\_emit_reins.py:156`. Both currently read

```python
ChartAxis(id='return_period', label='Return period',
          unit='return_period', scale='log', reciprocal_of='p',
          suggested_range=(1.0, float(round(1.0 / SURVIVAL_FLOOR)))),
```

and become

```python
ChartAxis(id='return_period', label='Return period',
          unit='return_period', scales=('linear', 'log'),
          reciprocal_of='p',
          suggested_range=(1.0, 10000.0),
          full_range=(1.0, float(round(1.0 / SURVIVAL_FLOOR)))),
```

Three changes, and all three are needed together.

**`scales=('linear', 'log')` replaces `scale='log'`.** Author decision 5: the
log default is ugly. A single declared scale is also the app's signal that there
is no reading to offer, so the log button never appears on this axis and the
reader cannot choose either way.

**`suggested_range` comes down to `(1.0, 10000.0)`.** Dropping the log default
alone is actively worse than leaving it: nine decades read linearly is a curve
pinned to the left edge. The window has to come down with the default. The
number is the author's ruling of 2026-08-21, and the instruction behind it is
that the axis follow the button: press `full range` and it shows the full
range, leave it alone and it shows 1 to 10,000. So the two readings are the two
declared windows and neither compromises for the other. 1-in-10,000 clears the
1-in-100 and 1-in-200 anchors ruled on elsewhere by two decades.

**`full_range` takes the old window,** so the deep tail stays reachable. This is
the field that lets the default come down without losing anything: past
`SURVIVAL_FLOOR` the curve is a line of float dust, which is worth a button and
is not worth the default reading.

What it buys downstream, and the reason the app half then carries no special
case for this axis at all: the return period becomes an ordinary axis. It draws
in its ladder, `full range` opens it to `1e9`, `log x` or `log y` makes the
opened tail readable, and the app's `MAX_RETURN_PERIOD` constant retires from a
mechanism to a backstop for documents that declare nothing.

## L2. The `reins` occurrence panel gets two honest axes

`src\aggregate\charts\_emit_reins.py:136` and `:143`.

**`sev_density` gains linear.** `scale='log'` stays as the default reading and
`scales=('linear', 'log')` joins it. The comment at `:140` currently argues "a
layered severity is a spike and a tail, and the linear reading of it is a spike
and nothing else, so there is no second reading to offer". Author decision 7 is
that this is a bug rather than a design choice: the linear reading is a reading,
the reader can see for themselves that it is a spike, and declaring one scale
removes the control rather than the temptation. Rewrite the comment to say log
is the default because the linear reading is usually a spike.

**`claim` gains log and an unconditional full extent.** Today:

```python
ChartAxis(id='claim', label='Loss per claim', unit='currency',
          suggested_range=claim,
          full_range=None if claim is None
          else (min(0.0, float(x[0])), float(x[-1]))),
```

On an **unlimited** program `_claim_window` returns `None`, so `full_range` is
written `None` and the panel loses `full range` as well as `log`, though the
full extent is knowable either way. This is `aggregate_api\dev\api-punchlist.md`
item 12, raised and now answered.

The complication is the `ChartAxis.__post_init__` guard: `full_range` requires
`suggested_range`, so the unlimited case needs a suggested window before it can
publish an extent at all. **Author ruling 2026-08-21: it suggests the extent.**

```python
extent = (min(0.0, float(x[0])), float(x[-1]))
ChartAxis(id='claim', label='Loss per claim', unit='currency',
          scales=('linear', 'log'),
          suggested_range=claim or extent,
          full_range=extent),
```

A limited program is untouched: `claim` is a window, so the suggestion is the
computed one exactly as today. An unlimited one draws its whole extent by
default, and `full range` on it becomes a button that changes nothing, which is
the honest reading rather than a defect. There is no crop to undo, and a control
present and idle says that more clearly than a control that is missing.

Together these turn the occurrence group from one offered button into three.

## L3. Does the mass axis declare its full extent? (open, not blocking)

`_emit_aggregate.outcome_doc:121` declines `full_range` on the ordinate, with
the comment "(0, the peak) already IS the whole extent, and a zoom-out button on
it would do nothing." True of the aggregate alone, and false once the severity
companion overtops it, which is exactly what `ordinate_top` and
`COMPANION_HEADROOM` exist to handle. So the emitter already computes the honest
extent, `max(peak, sev_peak)`, and then declines to publish it.

Declaring `full_range=(0.0, float(max(peak, sev_peak)))` would let the app half
delete a fallback and make `full range` mean one thing on every axis. **Not
ruled and not required.** The app's A3 keeps its data-extent fallback either
way, because punchlist item 7's log rule needs it for any document that still
declines. Left here so it is not lost rather than because anything waits on it.

## What moves downstream when these land

- **Every chart ETag changes**, since three axis declarations move. The API
  re-captures `dev\fixtures\charts.json` as its A6, after these land and not
  before.
- **The API needs `uv sync --extra dev`** with its server stopped, or
  `importlib.metadata` keeps reporting the old `aggregate` version and the moved
  axes never arrive. The standing version-skew trap.
- **Nothing in the wire format changes.** No new field, no `CHART_IR_VERSION`
  bump, no reader change: three existing fields take different values.
- The app's `MAX_RETURN_PERIOD` (`chartdoc-to-echarts.js:66`) keeps pointing at
  `_chartdoc.MAX_RETURN_PERIOD` and keeps its note. It becomes a backstop rather
  than the mechanism, so the constant stays and its comment gains a sentence
  saying so.

## Verification, LIB side

1. `pytest` clean, and the chart tests in particular: `return_period` losing its
   `scale` and gaining `full_range` touches whatever asserts on the axis tuple.
2. `doc_hash(load_chart_doc(canonical_dict(doc))) == doc.hash` still holds for
   `agg` and `reins`, the round-6 item-6 contract, since both documents move.
3. Build an **unlimited** occurrence program and confirm its `claim` axis now
   carries `scales`, a `suggested_range` and a `full_range`, where before it
   carried none of the three. This is the case the ruling exists for and the one
   no limited program exercises.
4. `.plot()` on an `Aggregate` and on a reinsurance program still draws through
   `plot_chartdoc(build_chart_doc(...))` unchanged. The return period is not a
   default reading in either, so the library's own drawing should not move; if
   it does, the emitter change reached further than intended.

## Cadence

One version bump, one `CHANGELOG.md` section under `[Chart-2D-Punchups]`, one
commit with a one line subject, this plan to `dev\done\`, the matching
`dev\TODO.md` entry ticked. The API's own bump, section and commit are separate
and come after, which is the whole point of the order of work.

## Execution log, 2026-08-21, `1.0.0a314`

Executed in one bump, as the cadence section specifies. All of L1, all of L2, and L3, which was ruled in during the review rather than left open. Seven divergences, recorded here at the moment each was made.

**Divergence 1: three emitters, not two.** L1 names `_emit_aggregate.py:138` and `_emit_reins.py:156`. `_emit_severity.py:156` carried the same declaration character for character, and the app plan's own clutter table gives `sev` a return period button, so leaving it behind would have left one document reading differently from the other three with nothing the app could do about it. Author ruled 2026-08-21: move all three. `_emit_aggregate.outcome_doc` serves both `agg` and `pnl`, so three edits cover four documents.

**Divergence 2: L3 landed.** The plan records it as open, not ruled, not required. Author ruled it in on 2026-08-21 during this review. The app's A3 keeps its data-extent fallback regardless, per the plan, so nothing app side changes as a result.

**Divergence 3: the extent is read off the series, not written at the declaration site.** L3 as drafted writes `full_range=(0.0, float(max(peak, sev_peak)))`, but `peak` and `sev_peak` are locals of `_agg` and the declaration lives in `outcome_doc`, which also serves `pnl`. Passing a second argument would have made the extent a caller's statement when it is one rule over whatever series arrive. `outcome_doc` now reads the tallest drawn mass in the loop it already runs over `subject` and `companion`, so the suggestion and the extent cannot drift apart. `ordinate_top` stays an argument, because whether to crop is genuinely the caller's decision.

**Divergence 4: `RETURN_PERIOD_TOP` rather than the literal.** The plan writes `10000.0` inline, at what it thought were two sites and turned out to be three. The number now lives in `charts/_two_panel.py` beside `SURVIVAL_FLOOR`, which is the module that exists "so the five emitters cannot drift apart". Its docstring carries the reasoning, and says explicitly that it is a different thing from the renderer's `plots._chartdoc.MAX_RETURN_PERIOD`, which is the backstop for a document declaring no window at all.

**Divergence 5: the local is `claim_extent`, not `extent`.** `_reins` already has `annual`, `claim` and `tops` in scope, and a bare `extent` says nothing about which of them it is the extent of.

**Divergence 6: verification item 4 is wrong, and the library's own drawing does move.** The item reasons that the return period is not a default reading, so `.plot()` should be unchanged. That holds for `.plot()` bare and not for `.plot(return_period=True)`, which is a shipped and tested call: `plots/_chartdoc._axis_scale` takes an axis' drawn scale from its declaration unless `log=True` is passed, and `_axis_window` takes its window from `suggested_range` unless `full_range=True` is. So the library's return-period picture moves from log over nine decades to linear on the ladder, and `a.plot(return_period=True, log=True, full_range=True)` is what recovers the old one. Eleven tests pinned the old default and were updated to state the new one, in `tests/test_plot_return_period.py`, `test_chart_agg.py`, `test_chart_reins.py`, `test_chart_severity.py` and `test_chart_pnl.py`; each gained an assertion that `log=True` still reaches the log reading, which is the half that has to keep working. Author confirmed 2026-08-21 that linear by default with log on offer is the goal, applied uniformly.

**Divergence 7: verification item 2 needs a stamped document.** `doc_hash(load_chart_doc(canonical_dict(doc))) == doc.hash` holds only after `stamp()`, since `ChartDoc.hash` is `None` until then. Checked on stamped `agg`, `reins` and `sev` documents: all three round trip. The contract is unmoved, and `tests/test_charts_ir.py::test_load_chart_doc_round_trips_every_emitted_document` covers it.

### Verification, as the plan asks for it

1. `.venv/Scripts/python.exe -m pytest -m 'slow or not slow'`, clean. The chart and plot modules were run first on their own and are clean at 190 passed.
2. The round trip holds for `agg`, `reins` and `sev`, on stamped documents (divergence 7).
3. An unlimited occurrence program's `claim` axis now carries `scales=('linear', 'log')`, a `suggested_range` and a `full_range`, where before it carried none of the three. Covered by `tests/test_chart_reins.py::test_an_unlimited_program_suggests_its_own_extent`, which also asserts `_claim_window` really does decline on that program, so the test cannot pass for the wrong reason.
4. `.plot()` bare is unchanged on every class. `.plot(return_period=True)` is changed on purpose, per divergence 6.

### Left for the author

`tests/test_chart_reins.py`'s module docstring says its DecL programs mirror the `CH.Reins*` entries in `src/aggregate/agg/decl-testers.agg`. No such entries exist, and none of the chart modules' programs are in that file, so the standing "append new pytest DecL programs there" rule had no section to append the new unlimited program to. Adding one program to a file missing the other five would be worse than leaving it. The fix is either a chart section holding all of them or a corrected docstring, and which one is the author's call.
