# Note from `aggregate_api`, round 6

> **Answered upstream 2026-08-11, `a251` through `a256`.** Five of the six
> items are closed and the two live regressions are gone, on a sync:
>
> | Item | Closed by |
> |---|---|
> | 1. Turn the `reins_stats_df` block over | `a256` `[Reins-Insurer-Orientation]`, Aggregate INSURER, three blocks |
> | 2. Publish the wider bucket window frame | `a254` `[BS-Window-Diagnostics]`, plus the `level_0` bug and three more like it |
> | 3. Register a `sharpen` exhibit | `a255` `[Sharpen-Exhibit]`, formats carried up as `exhibits.SHARPEN_FORMATS` |
> | 4. An exhibit for a computed pricing result | **OPEN, design settled 2026-08-12.** Answered in principle (`[Pricing-Keyed-On-Result]`: on the result objects, no `inputs=` channel); the Pricing pane design discussion happened and the pricing leaves are now planned end to end in `dev/plan-pricing-exhibits.md` (canonical copy in the API repo, symlinked here). The bounds leaves stay with `dev/plan-exhibit-official-channels.md` phase 10 |
> | 5. `_pnl_consideration` rounding | `a251` `[PnL-Consideration-Rounding]`; `_round_pnl_premium` deletes |
> | 6. Give a served chart document a way home | `a252` `[Chart-Doc-Reader]`, `load_chart_doc` |
>
> Separately, not asked here but relevant: `a250` fixed
> `calibrate_distortions(reins_view=...)`, so the `_BasisView` shim and
> `_REINS_BASES` are deletable on the same sync.

> **Updated 2026-08-10, after a71.** The author's ruling since this was written:
> **there are no exceptions.** The app draws every published exhibit as
> published, so items 1 and 2 below are no longer "the app does its own thing
> instead"; the app has stopped, and what those items describe is now **what a
> reader sees on screen**. Both are live regressions in the app's output until
> they land upstream, and both were accepted knowingly on that basis. Item 3 is
> unchanged. The api-side workarounds named in 1 and 2 are already deleted.

Four asks, all of one shape: **the app should never build a table document from a
pandas frame it fetched.** The library owns what a frame means, so it should own
the exhibit; where it does, the app draws what it is given and holds no formats,
no captions and no row emphasis of its own.

Round 5's ask 1 landed better than it was asked (raw values became a property of
a served block rather than a caller option, `1.0.0a246`), and a68 / a70 moved
five leaves onto the exhibit route on the back of it. These four are what is
left. Each names exactly what the app does instead today and what it would
delete.

Round 5's **ask 2** is now done on the api side and can be dropped from that
note, or kept: see item 5.

**Item 6 is a different shape from the rest, and the first on the charts side.**
Nothing is built twice and no reader sees the wrong thing. It is an asymmetry
between the two registries: the exhibit envelope reads back from the wire and
the chart document does not.

---

## 1. Turn the `reins_stats_df` block over, or let a caller turn it

**Blocks:** Reinsurance / Stats.

The `reins` exhibit's first block is the layering analysis with **measures down
the stub and layers across the columns**: stub `component | measure`, columns
`Gross | layer.1 | Ceded | Net` under an `occ` header level, 17 rows.

That is right for the library. A layer is a natural column of an analysis, and
the frame is built once for every consumer.

**That is what the Reinsurance / Stats pane now shows**, one 17-row table. From
round 3 to a71 the api transposed and split it (`_reins_stats_transposed`, now
deleted) into two tables:

- **Layer terms**: `view | layer | share | limit | attach | pr_attach |
  pr_detach | pr_loss | lol | output`, 4 rows. The layer's own contract.
- **Moments**: `view | layer` then `freq | sev | agg` each with `mean | cv |
  skew`, 4 rows. What the layer does to the three distributions.

Both run gross, ceded, net **down the rows**, because that is the comparison a
reinsurance reader makes and the eye makes it down a column rather than across
a row. The two are also different kinds of thing (a contract and a consequence)
and reading them as one table means reading a column header that changes meaning
half way down.

**The ask.** Publish the layering analysis in that orientation and that split,
as two blocks of the `reins` exhibit, under the insurer perspective. Raw can
keep the current orientation; that is exactly the distinction a perspective is
for.

**Already deleted at a71**, ahead of this landing: `_reins_stats_transposed`,
the two frame routes, their `tables.FORMATS` entries and `loadReinsStats`. The
leaf is `loadReinsExhibit(0, ...)` beside the Summary at index 1. So there is no
app-side work waiting on this, only the reading.

---

## 2. Publish the wider bucket-window frame

**Blocks:** More / Window.

The `bs_window` exhibit serves the public `bs_window_df`:

    level_0 | applies | selected | x_min | x_max | bs | log2 | log2_need
            | clipped | note

**That is what the More / Window pane now shows.** Until a71 the api served the
private `_bs_window_df` in preference, because it is two columns wider:

    applies | x_min | x_max | W | bs | log2 | coverage | note | selected
            | log2_need | clipped

`W = x_max - x_min` is the window's width and `coverage` is what fraction of the
distribution it holds. Those two are the pane's whole diagnostic value: the leaf
exists to answer "is this grid big enough", and the width and the coverage are
the answer. **A reader now has a list of candidate windows and no way to compare
them**, which is the cost of the ruling and was accepted.

**The ask.** Either add `W` and `coverage` to the published frame, or register
the exhibit against the private one. The stray `level_0` in the head is
presumably an unnamed index level and looks like a second, smaller bug.

**Already deleted at a71**: the route's preference for the private attribute and
the `bs_window_df` entry in `tables.FORMATS`. The leaf is `loadExhibitLeaf`, and
`/v1/objects/{id}/bs_window_df` serves the published frame, so the route and the
exhibit cannot disagree.

---

## 3. Register a `sharpen` exhibit

**Blocks:** More / Sharpen.

There is no exhibit for the bucket probe's audit at all, so this leaf takes the
frame route for `sharpen_score` and `sharpen_df` and the api writes both ledes:

- **Score grid**: every cell the probe walked, as steps in bs down and steps in
  log2 across. Blank cells are where it stopped.
- **Every cell**: the same walk with its working, one row per grid tried.

Those are the api's sentences about the library's own search, which is exactly
the kind of second opinion this whole exercise is removing. The probe knows what
it scored and why; it should say so.

**The ask.** A `sharpen` exhibit, two blocks, present when the object carries an
audit (which is the `has_sharpen` flag the api already computes).
`register_simple_exhibit` looks like it makes this close to a one-liner per
block; the registry is eleven entries today and this would be the twelfth.

**What deletes here:** the two ledes, the two frame fetches, and
`loadSharpenAudit` becomes `loadExhibitLeaf`. It is also the **last** thing in
`tables.FORMATS` that describes a library frame: `sharpen_df` and
`sharpen_score` are the only two entries left that are not computed from the
reader's own input, so this ask empties the table of second opinions entirely.

---

## 4. An exhibit for a computed pricing result

**Blocks:** Pricing / Determine and Evaluate, Bounds / Bounds, Pricing and
Allocation. Five leaves, and the last pandas in the api's table pipeline.

These are different in kind from the other three and the ask is a design
question rather than a registration. They are not keyed on the object: a pricing
pentagon is a function of the object **and** what the reader typed (a
probability or an asset level, a target, a distortion family, a premium), so
there is no `exhibit_frames(obj)` that could produce them.

So the api computes them and builds the documents itself, in `pricing.py` and
`bounds.py`, through `greater_tables` directly, holding a `FORMATS` table for
`price`, `stat_LR`, `stat_P`, `stat_PQ`, `stat_ROE` and the bounds frames. Every
format decision in there is the api's guess at what the library means by a loss
ratio or a return on capital.

**The ask, put as a question.** Is there a shape for "an exhibit computed from
this object plus these inputs"? Something like

    build_exhibit(obj, 'price', perspective=..., inputs={'p': 0.99, 'coc': 0.15})

returning the same envelope shape, so the api passes the form values through and
draws the answer. If that is wrong for the library, the alternative is narrower:
publish the **formats** for these frames so the api stops guessing at them, even
while it keeps assembling the documents.

Naming the deliberate exception is worth as much as closing it. If the answer is
"these stay app-built", the api will record that as settled rather than as a
ticket, which is the state the author objected to finding them in.

---

## 5. `_pnl_consideration` rounding: done downstream, still yours

Round 5's ask 2 was to round the P&L premium, which
`aggregate._program._pnl_consideration` returns as `est_m / loss_ratio`
unrounded, so `pnl_program` writes `1428.5840984231345 premium` into a program
the reader is meant to read, keep and edit.

The api now rounds it in `post_pnl` (`_round_pnl_premium`): no decimals above
100, two at or below, anchored at the head of the program so the
`0.25 premium expenses` trailer is untouched. That is a string rewrite of DecL
in a service, which is not where it belongs.

**The ask stands**, in `_pnl_consideration`, where the number is produced. The
api's version is idempotent, so the day this lands upstream it rounds an
already-round number and can delete.

---

## 6. Give a served chart document a way home

**Charts:** every chart leaf, and the `charts` registry as a whole.

The two registries are the same shape end to end. Discover, build, serve:

    exhibits.available_exhibits(obj)     charts.available_charts(obj)
    exhibits.build_exhibit(obj, name)    charts.build_chart_doc(obj, name)
    Exhibit.to_payload()                 charts.canonical_dict(doc)

**The tables side reads back.** `TableDoc` is a pydantic model, so a served
block reconstructs from its own wire form:

    blocks = [gt.TableDoc.model_validate(b) for b in payload['blocks']]

Hash for hash, and `greater_tables.render_html` draws the reconstruction
without knowing that an `Aggregate` ever existed. That is what makes the
envelope a document rather than a dump.

**The charts side does not.** `canonical_dict` and `canonical_json` go out and
nothing reads them back in. The obvious guess fails:

    ChartDoc(**canonical_dict(doc))
    AttributeError: 'dict' object has no attribute 'id'

`panels`, `axes`, `series` and `marks` arrive as plain dicts and `__post_init__`
validates them as dataclasses. So a consumer holding a fetched chart document
cannot hand it to `aggregate.plots.plot_chartdoc`, which is the library's own
renderer and already generic over any `ChartDoc`, until it has rebuilt the
document itself.

**The ask.** `load_chart_doc(d)` in `charts/ir.py` beside `canonical_dict`,
exported from `charts/__init__.py`, with the round trip as its contract:

    doc_hash(load_chart_doc(canonical_dict(doc))) == doc.hash

It is about a dozen lines: rebuild the four tuples of dataclasses, rebuild
`SurfaceData` where a series carries one, pass `meta`, `tex`, `ir_version`,
`generator` and `hash` through. Checked here against all three shapes the
registry produces today, each through a genuine `json.loads(json.dumps(...))`:
`agg` and `reins` on an `Aggregate` (four and six x/y series, lattice
coordinates on one axis of each), and `joint_surface` on a
`BivariateAggregate` (one nested `SurfaceData`). All three come back hash for
hash and redraw through `plot_chartdoc`.

**Why upstream rather than once per client.** Three reasons, none of them
convenience.

`canonical_dict` drops any field equal to its default, so a reader is a
statement about what those defaults are. Panel `read_axis` and `aspect` never
reach the wire, axis `kind` never reaches it, and `invertible` appears only on
the panels where it is true. A client writing its own reader is writing the
library's default table down a second time, and a default that moves upstream
then moves silently on the far side.

`ir_version` is validated in `ChartDoc.__post_init__`, which makes the reader
the one place a wire document's version is negotiated. A client assembling
dataclasses by hand may pass the field through or may not, and the one that does
not is the one that will accept a version 3 document without noticing.

`SurfaceData` is the only nested payload today, so a reader that forgets it
fails on exactly the bivariate documents, which are both the largest and the
least often exercised.

**Scale is the argument for taking the hash seriously.** The `agg` chart is
5.3 MB of canonical JSON, most of it the cdf series' explicit coordinates. A
quiet coercion in a hand-written reader, a tuple that arrived as a list or a
float that arrived as a string, is not something anyone catches by looking at
the picture. `doc_hash` catches it, and a library-owned reader is what makes
the hash mean anything on the far side of the wire.

**Not asked for:** a renderer, or any change to what is emitted.
`canonical_dict` is already complete and already deterministic. Only the way
back is missing.

---

## What is not asked for

The two density frames stay on the frame route on both sides. `density_df` and
`reins_density_df` are thousands of rows even binned to a display grid, so they
are the interactive grid's permanently and a table document of one would be a
document nobody reads. That is agreed, not outstanding.
