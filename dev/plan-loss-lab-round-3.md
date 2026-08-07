# Plan [Loss-Lab-Round-3]: what the app needs from the library

> **Status: not started.** Written 2026-08-07, split out of
> `aggregate_api/dev/plan-ui-round-3.md`, which is the immediate consumer and
> holds the api half of the same round. Current version `1.0.0a217`.

The Loss Lab app had its first end-to-end run and the author wrote a punch list.
Most of it is the app's own, and stays there. This doc holds the part that is
the library's, plus four adjacent upstream items exploration turned up while
tracing the rest. Phase A is the one the app is blocked on.

The table below opens both docs, so this one reads without the other. Items
marked `api` are listed for context only and are not this plan's work.

## The round at a glance

| code | problem and impact | side | effort |
|---|---|---|---|
| `wrap` | derived DecL returns on one line; needs spread layout, and the `hints{}` trailer must ride along or Sharpen stops reproducing | api | M |
| `ledger` | exhibit prose is unbounded inside the table's horizontal scroll box so it runs off screen; exhibits need vertical air between them | api | L |
| `lollipop` | every density draws as steps; the library's own ladder says stem and dot under 40 visible atoms | api | M |
| `parse` | a syntax error renders in the Overview pane while the strip says only "build failed" | api | M |
| `tooltip` | greyed main tabs and sub-tabs explain themselves by different rules, and a main tab collapses to a generic reason | api | L |
| `blue` | five buttons and the emacs switch are Bootstrap blue; the house accent is a custom token Bootstrap never sees | api | L |
| `recap` | the name and kind under the status strip repeat the status strip | api | L |
| `tabsize` | the main tab strip is too tall and its label is the one size off the type scale | api | L |
| `hints` | the history keys are undocumented on the action row | api | L |
| `emacs` | the emacs switch belongs in the hamburger as a checked item, not on the action row | api | L |
| `order` | tab order puts Economics second, and the order is hardcoded in two files with nothing checking they agree | api | L |
| `status` | the strip carries no log2, repeats the kind on line two, and the sharpen note overwrites the timing line | api | M |
| `reflines` | the ref-lines toggle covers the mean and one anchor; it should cover mean, 1-in-100 and 1-in-200 | api | M |
| `label` | the 1-in-200 label sits on top of its own rule | api | L |
| `twin` | the second right-hand axis is redundant against S vs RP, and its width is what re-lays-out the left panel | api | L |
| `ticks` | a padded explicit min and max put a six-significant-digit label at each end of every x axis | api | L |
| `severity` | a severity pdf is a sampled ordinate, not bucket mass, so drawing it as steps is wrong | api | L |
| `cede` | the cession row's label, button text, layout and spacing | api | L |
| `shortcut` | "occurrence net of" is long to type and the box forgets what you last ceded | api | M |
| `rubric` | the reinsurance plot's basis buttons explain themselves only in hover titles | api | L |
| `transpose` | reins stats reads measures down and layers across, wanted the other way, and carries no formats | api | M |
| `replot` | Re to Plot greys out on a reinsured portfolio because the chart is registered for Aggregate only | agg | M |
| `split` | "calibrate on" stacks three visual languages: house-red toggles, Bootstrap grey radios, a blue button | api | L |
| `assets` | `ReinsPriceRequest` takes only `p`, so the assets anchor 422s on a reinsured object | api | L |
| `decimals` | money columns are declared or inferred as integers, so a pricing table shows whole numbers only | api | L |
| `evaluate` | Evaluate offers no gross, net occurrence or net choice | agg then api | M |
| `cession` | nothing prices a cession with a distortion; the library has no glue between reinsurance and distortions | agg | M |
| `params` | an aggregate with no reinsurance gets pentagon rows and no distortion parameter table | api | L |
| `alloc` | reinsuring one unit of a portfolio silently drops every per-unit allocation table | agg then api | M |
| `grey` | the calibration basis row vanishes without reinsurance instead of greying, against house style | api | L |
| `panels` | the bounds envelope returns one panel where the library draws three | api | M |
| `units` | PricingBounds on a portfolio should default to every unit, with the box naming one | api | L |
| `behavior` | `tail_behavior_df` has no leaf | api | L |
| `window` | x_max and W print unformatted, and nothing says the window is `[x_min, x_max]` of width W | agg and api | L |
| `narrative` | narrative sections come out alphabetical by attribute stem | api | L |
| `sharpen` | `sharpen_df` is shown nowhere, so the grid audit is invisible | api | M |
| `caption` | passthrough exhibits return no kwargs, so raw frames arrive with no caption at all | agg | M |
| `writer` | the api re-implements the dependency-ordered `.agg` writer and its order disagrees with the library's | agg | L |
| `kinds` | PnL and BivariateAggregate carry old semantics under the new `summary_df` / `tail_df` names and have no `tail_df` | agg | M |
| `width` | `format_program(width=)` is accepted and ignored, a documented lie | agg | L |

## Why these are library work and not the app's

Three of the six trace to the same root. Pricing, evaluation and reinsurance
charting all assume the object's *own* density is the only one worth asking
about, which is true right up to the moment a program cedes. Then there are
three densities, gross, ceded and net, the object holds one of them, and every
public entry point silently answers about that one.

The app has already worked around this once. `aggregate_api/src/aggregate_api/pricing.py:267-321`
defines a `_BasisView` shim whose entire job is to point
`calibrate_distortions` at a density it was not designed to accept. It exists
because there is no keyword for it. That shim is the api holding library
knowledge, which the house rule says not to do, and it is why a reinsured
Portfolio currently loses its allocation tables altogether: the reinsurance
pricing path cannot call `analyze_distortions` at all, so it returns none.

Phase A closes the hole and lets the shim be deleted. It is the only phase the
app is blocked on, so it goes first.

## Phase A: a basis keyword on the pricing surface

**What lands.** A keyword, `basis` or `density`, on the three entry points that
currently read `self`'s density with no way to say otherwise:

* `calibrate_distortions`, implementation `src/aggregate/_pricing.py:515`,
  fronted at `_portfolio.py:2942` and `_aggregate.py:6251`
* `analyze_distortions`, `_portfolio.py:3715`
* `evaluate`

The vocabulary should be the one the app already speaks and the reinsurance
frames already carry: gross, net of occurrence, net. `_reinsurance.reins_density_df`
(`src/aggregate/_reinsurance.py:345`) has `p_agg_gross`, `p_agg_ceded`,
`p_agg_net`, `p_agg_ceded_occ` and `p_agg_net_occ` on an Aggregate, and a
Portfolio's (`_portfolio.py:1622-1640`) has only gross, ceded and net, so the
accepted set is object-dependent and the keyword must reject what an object
cannot answer rather than quietly returning the wrong one. The app has a bug of
exactly this shape today, offering three basis buttons where a Portfolio has two.

**What to watch.** `_portfolio.py:1637-1640` already warns that the three
portfolio marginals are separate distributions and do not satisfy
gross = net (+) ceded. Whatever the keyword does, it must not imply they do.

**Why the app cannot do this.** It has tried. The result is `_BasisView`, which
reaches past the public surface, and the api's own comment says the hole is the
reason it exists.

**Unblocks.** `evaluate` and `alloc` in the api plan's phase 5.

## Phase B: `cession`, a distortion-priced cession

`grep -n 'distortion\|Distortion' src/aggregate/_reinsurance.py` returns nothing.
The library computes ceded amounts thoroughly and prices ceded premium from the
DecL clause (`underwriter.py:938`, `_resolve_reins_economics`, which returns
`pc_occ`, `pc_agg`, `c_occ`, `c_agg` and per-layer breakdowns at `:1009-1012`),
but nothing prices a cession with a distortion.

**What lands.** A method on `Aggregate`, with the Portfolio twin if it falls out
cheaply, that walks `reins_density_df`'s ceded columns through
`Distortion.price` (`src/aggregate/spectral.py:1875`) and returns a
`(stage, view)` by `(el, bid, ask, margin)` frame. `Distortion.price` already
takes any pmf `Series` and an asset level, and `p_agg_ceded` and
`p_agg_ceded_occ` are exactly such series, so the pieces are all present and
only the glue is missing. Roughly forty to sixty lines plus tests.

**Why here and not in the app.** The stage selection is model knowledge, and
`Portfolio._reins_unit_views` (`_portfolio.py:1594`) already encodes it.
Reimplementing that in the api would duplicate a judgment about which stages
exist for a given program.

**Also, while in the file.** `reins_audit_df` does not exist and has not for
some time, but the name survives in stale docstrings at `_reinsurance.py:183`,
`:270`, `:306` and `_aggregate.py:4000`, `:4015`, `:4028`. Anything grepping for
it, including a future agent, will conclude the frame is there. The live names
are `reins_density_df`, `reins_stats_df` and `reins_summary_df`.

## Phase C: `replot`, a reinsurance chart for a Portfolio

`chart_reins` is registered for `Aggregate` alone: the emitter at
`src/aggregate/charts/_emit_reins.py:91` carries
`@chart_reins.register(Aggregate)`, and the availability gate `_has_cession`
(`:73-75`) reads `occ_reins` and `agg_reins` off the object, which a Portfolio
does not have in that form. So `available_charts` never answers `'reins'` for a
reinsured Portfolio.

The app is the visible casualty. Its Reinsurance Plot leaf is the only one gated
on the chart registry rather than the exhibit registry
(`aggregate_api/web/src/nav.js:108-113`), so it greys out and reads as
unbuilt. It is built.

**What lands.** A Portfolio emitter, plus a portfolio path in `_cession_stages`
(`_emit_reins.py:63-70`). Note the basis set differs: a Portfolio's
`reins_density_df` carries gross, ceded and net but no `p_agg_net_occ`
(`_portfolio.py:1632`), so the three-way basis choice the Aggregate chart offers
collapses to fewer options and the emitter must say which it has rather than
assume.

## Phase D: `caption`, and the prose half of `window`

**`caption`.** `register_simple_exhibit` returns `{}` for its frame kwargs
(`src/aggregate/exhibits/_core.py:716-717`), and captions are lifted from
exactly those kwargs at `_core.py:415-419`. So every passthrough exhibit,
`bs_window`, `stats`, `summary`, `validation`, `tail` and `economic` under the
raw perspective, arrives with no caption at all. The insurer-perspective
builders do carry them (`exhibits/_pnl.py:104-124`, `exhibits/_portfolio.py:24-76`,
`exhibits/_aggregate.py:25-76`), which is why the gap is easy to miss.

Consequence downstream: the app has grown its own competing prose for the
Overview exhibits (`aggregate_api/web/src/main.js:916-936`) because the library
sends none, so a frame's description now has three possible sources and they can
disagree. Give the passthrough exhibits captions, and column formats while
there, and the app can delete its copies.

**`window` prose.** `x_min`, `x_max` and `W` format with `:g` in
`_bucket_window.py:311`, `:397`, `:404` and `_portfolio.py:907-908`. That is
where the unreadable number in the app's Narrative pane comes from. The frame
half of the same item is the api's, since the app can set its own column formats.
The prose is only fixable here.

## Phase E: housekeeping

Three small ones, batched.

**`writer`.** `Underwriter.to_agg` (`underwriter.py:2515`) writes a file and
returns a `Path`, orders by `_KIND_WRITE_ORDER` (`:110`), and renders through
the terse `spec_to_decl` (`:2618`). The app needs the same thing as a string in
spread layout, so it has reimplemented all of it:
`aggregate_api/src/aggregate_api/routes/objects.py:807` defines its own
`_KIND_ORDER` and `:860-890` redoes the walk. The two orders disagree. The
library has no `pnl` key, so a P&L sorts last by fallback, and it puts
`distortion` before `agg` where the app puts it last.

Add `'pnl'` to `_KIND_WRITE_ORDER`, give `to_agg` a `layout=`, and add a
string-returning sibling that `to_agg` writes. Then the api deletes its copy and
there is one ordering in the world. Around thirty lines plus tests.

**`width`.** `format_program(spec_or_text, *, fmt, layout, trailer, width)` at
`decl_writer.py:1230` accepts `width` and ignores it, documented as reserved at
`:1277-1279`. Either implement it in `_render_spread` or remove the parameter.
The app's `wrap` item does not need it, `layout='spread'` is what it wants, so
this is about the signature not lying rather than about a missing feature.

**`kinds`.** The deferred half of `dev/done/plan-summary-tail-tables.md`. The
rename executed for `Aggregate` and `Portfolio` at `1.0.0a113`, hard cut, no
aliases. `PnL` (`_pnl.py:1407`, `:1969`) and `BivariateAggregate`
(`bivariate.py:2113`, `:2418`) still carry the *old* semantics under the *new*
names and neither has `tail_df`, so a consumer holding four kinds has to
dispatch on kind to know what `summary_df` means. The app does exactly that
today. Give both the current contract.

## Verification

* `uv run pytest` after every phase.
* Phase A additionally: build a program with an occurrence cession and one with
  an aggregate cession, and check that calibrating on each basis gives different
  distortion parameters and that asking a Portfolio for net of occurrence raises
  rather than silently answering about net.
* Phase B: check the priced cession against the clause-driven ceded premium from
  `_resolve_reins_economics` for a program where both exist. They measure
  different things and should not agree, but the expected loss leg should tie to
  `reins_stats_df`.
* Phase C: `available_charts` on a reinsured Portfolio answers `['reins']`, and
  the emitted document renders.
* After each phase, in `aggregate_api`: `uv sync --extra dev`, then confirm
  `/v1/meta` reports the new `aggregate_version`. Without the re-sync the app
  reads the old library and the phase looks like it failed.

Each phase bumps `1.0.0a*` and commits as one line,
`[Loss-Lab-Round-3] aNNN: <terse summary>`, with the `CHANGELOG.md` section
carrying the detail. This doc moves to `dev/done/` when phase E lands.
