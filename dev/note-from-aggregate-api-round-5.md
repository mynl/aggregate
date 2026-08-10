# Four asks from `aggregate_api`, round 5 (2026-08-09)

From the app side, for review and dispatch by whoever is working in this tree.
All four came out of `aggregate_api`'s `dev/plan-ui-round-5.md`, where the app
side of each is written up. Each is small; two of them **block** app phases,
which is why they are being raised before that work starts rather than
alongside it.

Nothing here is a bug report against a released behavior. Three are gaps the app
found by being the first consumer of a 1.0 surface, and one is a formatting
convention that reads wrong in a program a person is looking at.

Order below is by how much they block, not by size.

---

## 1. `build_exhibit` needs a way to pass `TableSpec` kwargs through

**Blocks** the app's a66. Highest value of the four: it closes three separate
complaints on the app's punch list at once.

`aggregate/exhibits/_core.py:462`:

```python
for block_name, df, kw in blocks:
    ir_blocks.append(gt.build(df, gt.TableSpec(**kw)))
```

`kw` is whatever the frame stage declared (caption, row_flags, formatters). There
is no way for a caller to add to it, and the app needs to add three things:

* **`include_raw=list(df.columns)`**, without which the blocks cannot be rendered
  by `greater_tables`' own `irToGridInput`. That function throws on any
  non-string data column with no raw value, which is every float and bool column
  in every exhibit. Consequence today: in the app's interactive table view,
  **every exhibit-route pane fails to draw** and shows a "Table renderer
  unavailable" message. Five panes, since a62.
* **`max_rows`**, so a long exhibit truncates the way the app's frame documents
  already do rather than shipping the whole frame.
* **`formatters={}` plus a wide `float_format`**, which is how the app serves its
  "full precision" reading. Setting `float_format` alone does nothing when an
  explicit `formatters` mapping is present, which is why both are needed.

**Suggested shape.** A `spec_extra` parameter on `build_exhibit`, merged over each
block's own `kw`, and accepting either a dict or a callable taking the frame, so
the per block `include_raw=list(df.columns)` can be computed:

```python
def build_exhibit(obj, name, perspective=Perspective.RAW, spec_extra=None):
    ...
    for block_name, df, kw in blocks:
        extra = spec_extra(df, kw) if callable(spec_extra) else (spec_extra or {})
        ir_blocks.append(gt.build(df, gt.TableSpec(**{**kw, **extra})))
```

Merge order matters and the one above is the proposal: **caller wins**, because
the caller is asking for presentation carriage (raw values, row cap, float
format) rather than for semantics, and the frame stage owns semantics through
`caption`, `row_flags` and `formatters`. If you would rather the exhibit's own
kwargs win, say so and the app will name only keys the exhibits never set.

**Why not do it app side.** The app can already reach `exhibit_frames()`, which
is public and returns plain `(name, df, kwargs)` triples, and could build its own
blocks. It would then have to reproduce `build_exhibit`'s envelope: the title
(`f'{fn.title}: {obj._title_name}'`, and `_title_name` is private), the meta
dict, and the `hash` that the app serves as an ETag. That is a fork of a contract
rather than a use of it, so the ask is here instead.

**Compatibility.** Purely additive; no existing caller changes.

---

## 2. `_pnl_consideration` should round the premium it sizes

**Blocks** the app's a65, but only cosmetically: the phase can ship without it.

`_program.py:640` returns `e_loss / loss_ratio`, and `decl_writer._fmt_num`
renders a float with `repr`, which is the shortest string that round trips. So
the premium the library writes into a P&L program is the full float:

```
>>> print(build('agg VT 100 claims 1000 xs 0 sev lognorm 50 cv 2 poisson').pnl_program())
pnl VT_PnL
  7037.883281186453 premium
  ...
```

The author's report is a smaller case, `17.500000000000018`, and the complaint is
that this is a program a person reads and edits, so it should carry a number a
person would write. `repr` is right for `_fmt_num` in general (it guarantees the
text re-parses to the same value) and this is not a `_fmt_num` change; it is
about what number goes in.

**Suggested fix**: round to six significant figures in `_pnl_consideration`,
before the value ever reaches the writer. `aggregate_api`'s `routes/objects.py`
has the same three lines as `_snap`, at three figures, for the reinsurance quick
edit form, and the reasoning there transfers: a number in a program is something
someone writes down.

**One documentation consequence, and it should land in the same edit.** The
docstring currently promises that expected loss is read off the computed density
"so the P&L's realized loss ratio is exactly the one asked for". After rounding
it is exact to six figures rather than to the bit. That is the right trade and it
should say so rather than quietly stop being true.

**Open for you to rule on**: six figures, or a different convention. The app has
no opinion beyond "not seventeen".

---

## 3. `PnL` has no `value_type`

Not blocking; the app degrades honestly without it.

The app's status strip reports the sign convention an object is read on, between
`log2` and `mean`. It reads `value_type`, which `Aggregate` and `Portfolio` both
expose off `_is_loss_value` (`_aggregate.py:3284`, `_portfolio.py:1271`). `PnL`
exposes neither the property nor `_is_loss_value`:

```python
>>> getattr(build(a.pnl_program()), 'value_type', 'MISSING')
'MISSING'
```

A P&L is the one first class kind that genuinely reads on the **payoff**
convention, and the library says so in its own prose: the insurer caption on the
`economic` exhibit explains that the kappa columns run "payoff convention, left
tail bad, and a loss sensitive premium correctly reads high there".

So the app was in the position of either printing nothing on the kind where the
convention matters most, or inferring `'payoff'` from the class name.

**The author has ruled that it is not an inference**, 2026-08-10: "PnLs are
ALWAYS payoff; that is implied by the name, profit (positive) and loss
(negative)." So the app now prints `payoff` for a P&L on that authority, and this
ask drops from "blocking" to "tidying": the app is asserting a fact about a
library class from outside it, which is the wrong side of the wall even when the
fact is right.

**Ask**: `PnL.value_type` as a constant property returning the payoff label
through the same `value_type_label` helper the other two use, so the string is
one string and the assertion lives where the class does. The app deletes its
special case the day it lands.

---

## 4. The `agg` chart's density panel carries no 1-in-100 mark

Not blocking. Cosmetic, and the app cannot fix it without inventing data.

The published `agg` chart document carries four marks:

| panel | mark | at | faint |
|---|---|---|---|
| density | `mean` | 4926.5 | no |
| density | `1-in-200` | 7916.0 | no |
| lee | `1-in-100` | 0.99 | yes |
| lee | `1-in-250` | 0.996 | yes |

The author's ask is for the density panel, the one they look at, to carry the
mean, the 1-in-100 and the 1-in-200 together, with the labels reading down the
left and right of their lines. The app owns the labels and the sides and is
fixing those. It cannot add the 1-in-100 line: the marks are the document's
statement about the picture and inventing one app side would be the app deciding
what is worth marking, which is the whole point of the marks living upstream.

**Ask**: the density panel carries the mean, the 1-in-100 and the 1-in-200.

**And 1-in-250 comes off the chart, both panels.** The author's ruling,
2026-08-10: "just 1-in-200, sorry, I couldn't remember what we decided on 200
against 250." So the Lee panel's `1-in-250` mark becomes `1-in-200` and the two
panels agree on one anchor. Scoped to the **chart** marks: this says nothing
about `CAPITAL_ANCHOR_PERIODS`, which the `tail` exhibit emphasizes and which is
a table with room for both.

The app draws whatever arrives, and puts the mean and the 1-in-100 to the left of
their lines with the anchor to the right, per the same author's layout ruling.

---

## Not asks, recorded so they are not raised twice

Two things the app found that are **not** requests, because the app is handling
them or they are already known.

* **`complete()` is a static keyword pool, not a grammar walk.** The api's
  `completion.py` says so in its own docstring: `parse_interactive` is LALR only
  and DecL is Earley with a dynamic lexer, so "what can follow here" is not
  askable. The app is not asking for it this round; it is going to stop pretending
  the endpoint is context aware. If a grammar aware completion is ever wanted it
  is a real piece of work (a parallel LALR grammar, or token by token stepping)
  and it should be planned as one.
* **`_TERMINAL_LABELS` entries carry a gloss inside the label**, so
  `'after' (profit-commission allowance)` is one string, and the api's `_label`
  does `.strip("'")` on it, which cannot reach the interior quote. That is the
  **api's** bug and the api is fixing it, by splitting the quoted token from the
  gloss. Mentioned only in case the same table is being read somewhere else the
  same way.
