# note: what aLL needs from `aggregate.charts`, and in what order

> Written from the app side, `aggregate_api/dev/plan-plot-ir-api.md`, 2026-08-09.
> Everything here is consumer-side information that is **not** derivable from
> `dev/plan-chart-ir.md`, `dev/chart-inventory.md` or the `[Chart-IR]` entry in
> `dev/TODO.md`. It does not restate the conversion plan; it records a decision
> the app made that changes the plan's priorities, and asks four schema
> questions that only the consumer can raise.

## The decision that changes things

The app is going **purist** about IR-based charts. There will be no app-side
chart builder at all: `twoPanelData` and its four argument builders, the
distortion builder, and the client-side reinsurance survival accumulation are
deleted in one commit, not retired chart by chart. Where the library publishes
no chart for an object, the app greys the leaf and its pane says "not yet
implemented".

The author's reason, worth passing on because it applies upstream too: a chart
that appears to work, but works because the old pathway is still wired, teaches
everyone the wrong thing about where the project stands.

**Consequence for this plan's ordering.** `dev/TODO.md` currently reads "Next:
agg, pnl, port, bvagg heatmap in plan order". Under the purist switch, Overview
Plot is blank for an aggregate, a portfolio and a pnl from the moment the app
lands its half. Overview Plot is the landing demo, so `chart_agg` stops being
one item in a queue and becomes **the critical path for the whole app**.

Requested order, and the reasoning rather than a decree:

1. **`chart_agg`.** Unblocks the landing demo. Nothing else in the app is worth
   looking at while this is missing.
2. **`chart_port`.** The second thing anyone builds, and the one the demo flow
   goes to next.
3. **`chart_pnl`.** Reached through the Economics tab, so a blank there costs
   less than a blank on Overview.
4. **The bivariate heatmap** sits with the app's surface workstream and is not
   on this critical path. See question 2 below, which changes what it even is.

The toggle declarations (question 1) are wanted **with or before `chart_agg`**,
because the app's control strip is rebuilt from the document in the same commit
that adopts it, and building it twice is the waste this whole exercise exists to
avoid.

## Four schema questions

### 1. Toggle declarations: what the consumer needs them to say

The app is rebuilding its control strip to read from the document rather than
from a per-kind array. Three families, settled with the author:

1. **log or linear**, per axis.
2. **full or zoomed**, per axis.
3. **probability or return period**, the reciprocal reading of a survival axis.

**The surfacing rule the app will apply**, which the emitter should know so it
does not over-declare: a control appears if **any** axis in the document
declares it, and then applies to **every** axis that declares it. One "log y"
button for a two-panel chart, not one per panel. So declaring log on an axis is
a statement that a log reading of that axis is meaningful **and** that it is
reasonable for it to move at the same moment its siblings do.

Two of the three look like promotions of hooks that already exist rather than
new fields, which is offered as an observation and not a design:

* Log or linear generalizes `meta['z_log_ok']` from the surface pilot into a
  per-axis declaration of the scales an axis may be read on, with `scale`
  staying as the default reading. The app's adapter currently special-cases
  `z_log_ok` and would stop.
* Full or zoomed is half-expressed already: `suggested_range` **is** the zoom
  after a211. What is missing is the statement that the full data extent is a
  legitimate alternative reading, which is not true everywhere. A distortion's
  unit square has no meaningful "full x", so a blanket "the full extent is
  always available" would put a dead button on that chart.
* Probability or return period is J1, resolved as semantic and drafted as
  `ChartAxis.reciprocal_of`. Present means the app offers the control.

The app will ignore a declared reading it does not recognize rather than
guessing, so adding a fourth family later is safe from the consumer's side.

### 2. `heatmap | surface` is now a control, which makes it a schema question

The app is promoting the bivariate heatmap from hidden WebGL fallback to an
explicit option button beside the axis readings. The reader picks the reading
instead of discovering which one their hardware allowed, and a machine without
WebGL gets one option greyed with a reason.

That makes it a question the library has to answer, because the two are the same
data:

**Does one `ChartDoc` declare two realizable forms of its grid panel, or are
`joint_surface` and a heatmap two registry entries?**

The app's preference is **one document, two declared realizations**, for two
reasons. A control that redraws beats one that refetches, and the two entries
would have to be kept in step by hand forever, since they are the same block
summed grid with the same axes and the same labels. It also generalizes: the
same declaration would let the matplotlib renderer stop stamping "(projection)"
as a degradation and instead pick the realization the document says is available,
which is a nicer version of the capability pattern than degrade-and-confess.

If the answer is two entries, the app can live with it; the control just becomes
a refetch and the note is that they must not drift.

### 3. Is `ChartDoc.tex` total?

a209 added the plain-text naming rule and the `ChartDoc.tex` lookup. The author's
ruling app side is that **the document carries a TeX form and a plain-text form
of every label, and neither renderer derives one from the other**.

The consumer question is whether the lookup is **total or partial**. If a label
can arrive with a plain form and no TeX entry, the app has to invent a fallback,
and inventing a fallback is the app deriving semantics, which is the thing this
architecture exists to stop. Either guarantee is workable as long as it is
stated: total, or partial with an explicit "no TeX for this one" that the app
can render as plain without guessing.

Related and low cost: the plain form does not have to be ASCII. `ǧ(s)` in
`constants.py` is the right instinct and Unicode carries most actuarial labels
honestly (Greek, sub and superscript digits, `𝔼`, `≤`, `√`). The app may well
never load a math renderer, in which case the plain form is what everyone sees,
so it is worth spending a little care on.

### 4. Does the document carry panel arrangement?

The app is committing to "whatever the library publishes is the chart, panel
count included". Its current geometry does not honor that: `twoPanelBox` and
`squareBox` hardcode two panels side by side or one square, per kind.

Rebuilding that from the document needs to know what the document says about
**arrangement**, not just panel count. `plot_severity` is four panels as
`'AB\nCD'` and `plot_aggregate` is three as `'ABC'`, so this is not hypothetical
even inside the 1.0 scope. Equal aspect is already a panel-level field per
inventory item D8, which covers the square cases. What is unclear from here is
whether a consumer is expected to read rows and columns off the document, or to
lay panels out itself from their order and their aspect flags.

Either answer is fine. The app just needs to know which, before it writes the
layout that replaces the two hardcoded boxes.

## One request that is outside the current scope

**`chart_bounds_envelope`**, and it now has a concrete cost attached rather than
being a nice-to-have.

`dev/plan-chart-ir.md` lists the bounds plots as bespoke and out of 1.0 scope,
which was right when nothing depended on them. The app is deleting its
server-rendered matplotlib route (`plotting.py`, `GET /objects/{id}/plot`) and
moving image export client side. After that, **the Bounds tab is the only reason
matplotlib remains a dependency of the api at all**: `aggregate_api/bounds.py`
renders `plot_envelope` to SVG server side and ships the bytes.

So this is not a request to reprioritize it into 1.0. It is a note that the day
it lands, an entire rendering pipeline, a dependency, and a class of "why does
the server need a display" questions leave the app in one commit. Worth knowing
when it is weighed against other post-1.0 work.

Also still open and already logged app side: **`chart_reins` for `Portfolio`**.
It is registered for `Aggregate` alone, so the Reinsurance Plot leaf greys on a
reinsured portfolio. The app's TODO records it as blocked on
`dev/plan-loss-lab-round-3.md` phase A.

## What the app is not asking for

Recorded so it is not inferred from the above.

* No renderer passthrough, no `extra_echarts_kwargs`, nothing of that shape. The
  guardrail in `plan-chart-ir.md` is right and the app wants it kept. A need the
  document cannot express is a schema change or the chart stays bespoke.
* No app-side vote on panel content. If the Overview should be density plus
  exceedance rather than density, log density and Lee, that is decided in the
  emitter. `dev/chart-inventory.md` row 1 already documents the app's two-panel
  semantics column by column and is the right input to that decision.
* No hurry on the `plot_twelve` panels or the pedagogy charts. They were
  inventoried for schema pressure and nothing in the app waits on them.
