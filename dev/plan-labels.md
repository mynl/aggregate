# Plan — [DecL-Labels-Everywhere]: labels as namespaces, and their exhibit integration

Status: **DONE (Phases 0–4)** — landed `1.0.0a128`, all green (2143 tests + the
`dev/FEATURES.csv` audit). Phase 0 (`LabeledMixin` + object model), Phases 1–3
(exposure / severity-clause / layer interior labels + grammar + unparser +
round-trip), Phase 3b (Distortion D6 realignment), **Phase 4** (Portfolio exhibit
`renamer` sweep: `summary_df`, unit-density frames, `analyze_distortion(s)` pricing
frames, `plot` legend; `use_labels` switch; `unit_renamer` heuristic retired).
Narrowed during execution: `unit_renamer` had **no internal callers** and PnL legs
already carry their labels from a124, so Phase 4 was Portfolio-only. The interior
labels (exposure/layer/severity) are stored + introspectable via `a.labels`; no
dedicated per-unit interior-label breakdown exhibit yet (a natural follow-on).
**Phases 5–6 (S4 mixture-component / S5 frequency labels) remain deferred (D3).**

Original status: **READY** (2026-07-03; D1–D6 settled; delivery via a shared
`LabeledMixin` — D7). Follow-on to `dev/done/plan-decl-labels.md`
(`[DecL-Labels]`, landed `1.0.0a124`). That pass delivered *object-level* labels
and three special sites; this pass is the two things it left on the table, both
of which the author explicitly asked for:

1. **Broaden** — a label on every *sub-object* site inside a body (exposure,
   layer/limit, inline severity clause, per-mixture-component severity,
   frequency), not just the object head.
2. **Deepen** — actually *route exhibits through labels*. Today the labels exist
   on the objects but the dataframes/plots/towers still key off the bareword
   `name` handle. This is the real payoff and the "controlled blast radius."

Pure-presentation, as before: **no computed value changes.** Labels land in
attributes, dict keys, dataframe index/column headers, and plot text — never in
the FFT.

---

## The idea, stated precisely: a label is a namespace, not a synonym

The author's framing — *"labels are like Python namespaces, a honking good idea,
do more of them"* — is the right one, and it has a concrete architectural
consequence. A DecL program is a **tree** of named things:

```
port  "Big Portfolio"
└── agg  "Big Book"                     (a unit)
    ├── exposure   "GWP 2026"           (the sizing basis)
    ├── sev        "ISO Mixed Exp B"    (the ground-up severity)
    │   ├── component "ISO 5M"          (mixture leg)
    │   └── component "ISO 10M"         (mixture leg)
    ├── occ_reins  "Layer 1"            (a cession)
    └── agg_reins  "Aggregate Program"  (a cession)
```

Each node has a **handle** (the ID-shaped bareword — the identity, the reference
target, the dict key) and an optional **label** (the human string). A namespace
is exactly this: a stable key with a presentation name, and names that *qualify*
as you descend (`Big Book / Layer 1`, not a flat `Layer 1` that collides with
every other program's first cession). The a124 split (`name` = handle,
`display_label` = label; `_description`/`_explanation` precedent, **not** a
synonym) already got the object level right. Broadening = giving the same split
to the interior nodes; deepening = teaching exhibits to walk the tree and read
labels instead of handles.

**Design consequence:** the interior nodes are *not Python objects* — an
`Aggregate` has no `Layer` class, no `Exposure` class. So their labels cannot
live as `self.display_label` on a child object. They need a **single structured
home** on the object. That is where "do more namespaces" pays off literally: one
`labels` accessor per object, not a dozen scattered `self.*_label` attributes.

## Recommended architecture — a `labels` namespace object (object-model first)

Per the author's steer ("start from the new `__init__` and work backwards"),
fix the object model first, then spec, then grammar, then exhibits.

**Add one attribute** — `self.label_map: dict` — populated in `__init__` from the
spec, plus **one accessor** `self.labels` returning a small read-only namespace
view (`SimpleNamespace`-like) so exhibit code reads `a.labels.severity`,
`a.labels.occ_reins[0]`, `a.labels.exposure`, with the handle as the fallback
value so a consumer *always* gets a printable string. Object-level
`display_label`/`display_name` stay exactly as they are (no churn); `labels`
is the *interior* store plus a mirror of the object label at `labels.self`.

```python
# Aggregate.__init__ (sketch — names vetted against surface in Phase 0)
self.label_map = spec.get('label_map', {})     # {'severity': 'ISO...', 'occ_reins': {0: 'Layer 1'}, ...}
# self.labels -> namespace view, missing keys fall back to the handle/default
```

Why a map+accessor rather than N flat attributes:

* **One place to look.** Every exhibit consumer calls the same resolver; adding a
  new labelable site later doesn't add a new attribute to remember.
* **Composable/qualifiable.** A resolver can prefix the parent's label when an
  exhibit wants the qualified form (`Big Book / Layer 1`) — impossible if the
  child label is a bare string with no parent link.
* **Vetting.** Only **one** new public name to clear against the existing surface
  (`label_map`, `labels`) instead of five (`sev_label`, `exposure_label`, …).
  Both must be `rg`-checked in Phase 0 (house rule — a stored value shadowing a
  method is exactly the `self.approximate` trap).
* **Introspection / FEATURES.csv.** `labels` is one documented member.

Fallback/default rule (so exhibits are never blank): resolver returns, in order,
the explicit label → a *derived* default (`"5M xs 5M"` for a layer, the dist name
for a severity, `"GWP"` for a premium exposure) → the handle. The derived default
is the same idea a124 used for expense groups ("premium expense" etc.).

## Delivery — one `LabeledMixin`, not five copies (D7, settled)

The five labelable classes share **no common base** — `Distortion`, `PnL`,
`Aggregate` are bare, `Portfolio(object)`, and `Severity(ss.rv_continuous)`
extends scipy (untouchable). Identical label surface over unrelated hierarchies is
the textbook mixin case, so the whole surface lives in **one `LabeledMixin`**:

* **Mixin provides:** `display_label` storage; `display_name` resolved property
  (label → derived default → handle); `label_map`; the `labels` namespace
  accessor; the `renamer` property (+ cached `_renamer`); `_use_labels` and the
  `use_labels` property whose setter invalidates `_renamer`.
* **Each class supplies the specifics** via a small hook: its handle (`name`), how
  it fills `label_map` from its spec, and the axis its `renamer` serves
  (`Portfolio` → unit handles, `PnL` → leg handles, `Aggregate` → sub-parts) —
  e.g. an overridable `_label_handles()` the generic `renamer` walks.
* **Explicit init, not cooperative `super().__init__()`.** `Severity` sits on a
  heavy scipy base and the code already reads `getattr(self, 'display_label',
  None)` defensively (`_severity.py:1312`) — init ordering is fiddly. The mixin
  exposes `self._init_labels(display_label=…, label_map=…)` that each `__init__`
  calls when ready; no reliance on MRO chaining. `class Severity(LabeledMixin,
  ss.rv_continuous)`.
* **Distortion subclasses come for free** — mix into the `Distortion` base and the
  `Base<Kind>` subclasses inherit. Label state must ride each class's
  `__reduce__` / `decl_spec` tuples so pickle / round-trip survive.
* **Naming:** `LabeledMixin`. The house `Base<Kind>` rule is for sibling
  *taxonomies*; a mixin is a different idiom, `…Mixin` suffix is the clear signal.
  Codebase's first mixin — record the convention in CLAUDE.md's naming section.
* **Not mixed into `Bounds`** (not DecL-created).

## Per-object on/off switch — `use_labels` (D5, settled)

Labels are **on by default per object**; a dev turns them off with a setter.
**No module-level global** (the author's clarification: "there is no GLOBAL
use_labels — each object has a `_use_labels` defaulting to true").

* `self._use_labels = True` set in `__init__`.
* a `use_labels` **property** — getter returns `self._use_labels`; setter assigns
  *and* invalidates the object's cached `_renamer`. A property (not a bare
  attribute) precisely so "do more than flip a bool" (cache invalidation now,
  whatever later) has a home — the author's forward-looking note.
* **Serve sites** gate on `obj.use_labels`: `True` → apply the `renamer` in the
  final `df.rename()`; `False` → serve the raw handle frame (debug / join view).

This supersedes the earlier "opt-in `labels=True`" framing of D2: the label view
is the **default**; `use_labels=False` is the opt-*out* to raw handles. The
"handle stays the DataFrame key" half of D2 is unchanged — the raw frame is always
handle-keyed; the switch only governs whether the served view is relabeled.

## Distortion — rip out the half-baked naming, adopt labels (D6, settled)

Distortion's label story is SNAFU and gets fixed **here and now** (author's call):
the current `display_name` **attribute** (which held the label, defaulted to `''`,
and messed up repr and other display paths) is **removed**, and Distortion adopts
the a124 convention exactly like every other class:

| | handle | optional label | resolved property |
|---|---|---|---|
| agg / port / sev / pnl (a124) | `name` | `display_label` | `display_name` |
| **Distortion (today — broken)** | `_name` | `display_name` (attr) | `name` (property) |
| **Distortion (target)** | `name` | `display_label` | `display_name` (property) |

Target semantics: `name` = the handle (kind key or given name); `display_label` =
the optional explicit label (DecL `as`, or `None`); `display_name` property =
`display_label` → auto-pretty (`'PH(0.9)'` / `'TVaR(0.3)'`, kept as the *derived
default* layer) → `name`. The useful auto-pretty strings survive — they just move
from a mis-named mandatory attribute to the derived-default rung of the fallback.

Work: the ~20 factory sites that pass `display_name=f'…'` become the auto-pretty
default feeding `display_name`; the `name` property that returned
`display_name or _name` is deleted; `decl_spec` / `__reduce__` tuples updated;
**audit every repr / exhibit / plot site that read the old `display_name`** (the
"other things" it broke) and repoint at the resolved `display_name` property.
Grammar: add `display_label` to the three `distortion_out` alternatives (after
`name`), same last-child rule. The class-level `Distortion.renamer`
(`{kind: long_name}` catalog) is a *different* axis and stays untouched.

## Current state — what lands where today (a124)

| Site | Grammar today | Lands as | Status |
|---|---|---|---|
| object `agg`/`pnl`/`sev`/`port` | `name display_label` | `self.display_label` → `display_name` | ✅ done |
| pnl premium head | `pnl_premium display_label` | `_premium_label` → consideration leg key | ✅ done |
| expense group | `expense_group … display_label` | obligation leg key | ✅ done |
| reins cession | `reins_clause … display_label` | `margin_df` perspective label | ✅ done |

The `display_label` non-terminal (`AS label`, `label: ID | STRING`) is already a
reusable child — **the grammar machinery is built.** Broadening reuses it.

## New sites the author's example asks for

`agg A as XXX  premium 10000 as XXXX at 65% lr  10000 xs 0  sev … as "ISO Mixed
Exponential B"  occurrence net of 200 xs 200 as "Layer 1"  poisson  aggregate net
of 2000 xs 3000 as "Aggregate Program"`

| # | Site | DecL sketch | Home in `label_map` | Value | Grammar blast radius |
|---|---|---|---|---|---|
| S1 | **exposure / consideration** | `premium 10000 as "GWP 2026" at 65% lr` | `exposure` | high (the sizing basis is a named thing on every exhibit) | low — one child on 4 `exposures_*` alts |
| S2 | **occurrence limit / layer** | `10000 xs 0 as "Working Layer"` | `layer` | medium | **watch** — `layers` feeds `sev` context; label must sit where it can't be eaten by the `numbers` that follow |
| S3 | **inline severity clause** | `sev lognorm 100 cv 2 as "ISO ME B"` | `severity` | high | low — one child on `sev_clause_*` |
| S4 | **per-mixture-component sev** | `sev [lognorm gamma] … as ["A" "B"]` or per-leg | `severity_components[i]` | medium (the a124 deferral) | **high** — invasive to the severity mini-language; keep separate/optional |
| S5 | **frequency** | `poisson as "Cat frequency"` | `frequency` | low | low, but freq productions are left-recursive (`freq ZM`, `freq ZT`) — label only on the base freq |

Grades drive phasing: S1/S3 are cheap and high-value; S2 needs a careful anchor;
S4 is its own mini-project (do last, opt-in); S5 is low-value (defer unless free).

## Grammar mechanics & blast-radius containment

The reusable child already exists; broadening is **mechanical but wide**. Each
new site is the same four-step edit, and *that repetition is the risk* — every
production that gains a `display_label` child shifts the child tuple the
transformer unpacks, and every one must be mirrored in the **unparser**
(`decl_writer.py`) or round-trip tests fail.

Containment rules (make the sweep boring and greppable):

1. **Always append `display_label` as the *last* child** of a production, exactly
   as `reins_clause` does — so the transformer edit is a uniform "unpack one more
   trailing item, `label = display_label.get('display_label')`" and never
   reorders existing children.
2. **Anchor against the following token.** The one real hazard is S2: `10000 xs 0
   as … sev …` — the `as` must attach to the *layer*, and the delimited/keyword
   `as` can't be confused with the `numbers` grid that a bare label-less layer is
   followed by. Because `display_label` starts with the reserved `AS` token and
   layers are followed by `sev_clause` (which starts with `sev`/`ssev`/`dsev` or a
   builtin), the prefixes are disjoint → Earley-safe. **Verify with an explicit
   ambiguity test**, not by inspection.
3. **One spec key per site**, gathered into `label_map` by the *object*
   transformer method (e.g. `agg_out_named` assembles `label_map` from the
   `exposure`/`severity`/`layer` fragments its children already return), so the
   fragment methods stay small and the assembly is in one place.
4. **Snapshot re-baseline** (`tests/data/expected_specs.json`) once, after all
   grammar edits, for the new spec keys.
5. **Unparser parity** — `decl_writer.py` emits `as "…"` for each populated label.
   The round-trip test (`build(str(a)) == a` shape) is the guardrail that the
   sweep is complete.

## Exhibit integration — the deep half (the real payoff)

Today the labels sit on objects but the human-facing tables/plots still print the
handle. Grep confirms the consumers key off `.name`:

* `PnLTower` builds columns from `leg.name` (`_pnl.py` ~824–835).
* `PnL.margin_df` / `summary_df` index off `l.name` (`_pnl.py` ~439–558).
* `Portfolio` statistics/report/density frames and plot legends/titles use unit
  `.name`.
* `Aggregate`/`Severity` repr and plot titles use `display_name` already, but
  their component/layer breakdowns don't exist yet as labeled rows.

**Mechanism — a serve-time `renamer`, the established idiom (author's steer).**
The codebase already serves labels this exact way: `Portfolio.unit_renamer`
(`_portfolio.py:3426`) is a cached `{handle: display}` dict applied as a final
`df.rename(columns=…)`; `pedagogy._short_renamer`, `tweedie._renamer`,
`spectral._distortion_names_` are the same shape. So the label system's job is to
**produce that dict**, and each object exposes a `renamer` property keyed to the
axis its exhibits serve (`Portfolio` → unit handles, `PnL`/`PnLTower` → leg
handles, `Aggregate` → its own sub-parts). Exhibit code applies it as one
`df.rename(index=/columns=…)` at the serve step — no per-builder `x.name` surgery.

**`unit_renamer` is replaced outright, not extended (author's call).** It was
*guessing* nicer names from the handle (`title()`, `.`/`:` → space, `X1`→ TeX
subscript). Labels give the explicit name, so the heuristic is deleted. The
fallback chain shrinks to three layers: **explicit label → derived default
(`"5M xs 5M"` / dist name / `"GWP"`) → handle**. `renamer` is sourced purely from
`label_map`; `unit_renamer` becomes (or is superseded by) `renamer`.

**Display convention (D2, settled).** Label-only for narrative/plot text; keep the
handle as the DataFrame index/key, and expose the label view via an opt-in
`labels=True` display mode (or a parallel lookup). Nothing downstream that joins
on the handle breaks — this preserves the a124 promise ("labels are data, compute
untouched") at the exhibit layer too. No `label (handle)` clutter in cells, no
MultiIndex reshaping.

**Ordering invariant (D4, settled).** Ordering keys off the **handle** (or
declaration order — `unit_names` is `append`-built, `total` last; the only sort in
the exhibits, the per-distortion frame at `_portfolio.py:3067`, sorts the *handle*
index), **never the label.** This is guaranteed by the mechanism, not just
convention: because `renamer` is applied as the final `df.rename()` and `rename`
preserves order, relabeling happens *after* any sort on an already-ordered axis —
a label change physically cannot reorder an exhibit. Corollary: exhibit code must
never sort *after* relabeling (i.e. never sort by label text).

## Phases (all carry descriptive labels per house rule)

0. **[Labels-Object-Model]** — `rg`-vet `label_map` / `labels` / `renamer` /
   `use_labels` against the whole surface; add `self.label_map` + `labels`
   accessor + `renamer` property + `self._use_labels = True` + the `use_labels`
   property (setter with `_renamer` cache-invalidation) to `Aggregate`,
   `Portfolio`, `PnL`, `Severity`, `Distortion` — **via a shared mixin (see
   below), not copy-paste.** No grammar yet; object-level `display_label`
   untouched. This is the "new `__init__`" the author asked to start from.
1. **[Labels-Exposure]** (S1) — `display_label` on the four `exposures_*` alts →
   `label_map['exposure']`. High value, low risk.
2. **[Labels-Severity-Clause]** (S3) — `display_label` on `sev_clause_*` →
   `label_map['severity']`. (Distinct from the standalone `sev NAME as …` object
   label a124 shipped — this is the *inline* clause inside an `agg` body.)
3. **[Labels-Layer]** (S2) — `display_label` on `layers_xs` (+ tower?) →
   `label_map['layer']`; **ambiguity test** vs the following `sev_clause`.
3b. **[Labels-Distortion]** — reconcile Distortion's inverted `display_name` /
   `name` naming to the a124 convention (D6), add `display_label` to the three
   `distortion_out` grammar alternatives, and give `Distortion` the `labels` /
   `use_labels` members (leaf object: its label is consumed by the Portfolio
   pricing exhibits' `renamer`). Keep the class-level `renamer` kind-catalog.
4. **[Labels-Exhibit-Sweep]** — the deep half: add the `renamer` property (from
   `label_map`) to each object; apply it as a final `df.rename()` in `PnLTower`,
   `PnL.margin_df` / `summary_df`, `Portfolio` frames + plot legends/titles, and
   `Aggregate`/`Severity` breakdowns. **Delete `unit_renamer`'s heuristic** and
   repoint its callers at `renamer`. Land the D2 (label view) + D4 (ordering)
   conventions.
5. **[Labels-Mixture-Components]** (S4) — **deferred (D3), not this pass.** The
   a124 deferral: per-component labels inside a weighted severity. Opt-in, its own
   mini-language change; stays the tracked `dev/TODO.md` item.
6. **[Labels-Frequency]** (S5) — **deferred (D3), not this pass.** Low value.

Phases 1–3 are independent and could land in any order / together; Phase 4
depends on 0 and benefits from 1–3 being in. Phase 0 gates everything. This pass
executes **0–4**; 5–6 are recorded for the next pass.

## Testing / housekeeping (standing rules)

* Extend `tests/test_decl_labels.py`: a label on each new site, the S2 ambiguity
  case, the derived-default fallbacks, and an **exhibit** assertion (a labeled
  `margin_df`/tower column carries the label). Append the DecL programs to
  `src/aggregate/agg/test_decl.agg` under the matching section (house rule).
* Re-baseline `tests/data/expected_specs.json` once after the grammar edits.
* Round-trip: `decl_writer` emits every new label; `build(str(x))` shape-stable.
* Version bump (`1.0.0a*`), `CHANGELOG.md` `[DecL-Labels-Everywhere]` section,
  `dev/TODO.md` (close the a124 mixture-component deferral when Phase 5 lands),
  `dev/FEATURES.csv` for the new `labels` / `label_map` member + resolver.
* Docs: extend the DecL-reference `as` note with the new sites + a short "labels in
  exhibits" subsection (author rebuilds; keep `.rst`/`.qmd` refs in lockstep, do
  not build in the loop).

## Risks

* **Wide but shallow.** No single edit is hard; the risk is an *incomplete sweep*
  — a production gains a label the unparser forgets, or an exhibit site still
  prints the handle. The round-trip test + a "grep for `.name` at presentation
  sites" checklist are the mitigations.
* **S2 layer ambiguity** — the one genuine grammar hazard; gated behind an
  explicit ambiguity test before it lands.
* **S4 mixture mini-language** — invasive; quarantined to its own late phase so it
  can slip without blocking the high-value S1/S3/exhibit work.
* **D2 handle-visibility** — get the convention right *before* the exhibit sweep,
  or the sweep gets redone. Hence it's an Open Decision, not an implementation
  detail.

## Decisions (settled with the author, 2026-07-03)

* **D1 — Object model → `labels` namespace + `label_map`.** One structured store
  `self.label_map` populated in `__init__`, plus a read-only `self.labels`
  accessor (`a.labels.exposure`, `a.labels.occ_reins[0]`). Object-level
  `display_label`/`display_name` unchanged. One public name-pair to `rg`-vet in
  Phase 0. This is the "honking good namespace" form.
* **D2 — Exhibit display → label text, handle stays the key.** Human-facing text
  (columns, plot legends/titles, narrative) shows the **label only**; the
  DataFrame index/key stays the bareword **handle** so joins/references/round-trip
  are untouched. An opt-in `labels=True` display mode surfaces the label view.
  Preserves the a124 promise ("labels are data, compute untouched") at the exhibit
  layer. No `label (handle)` clutter in tables, no MultiIndex reshaping.
* **D4 — Ordering keys off the handle, never the label.** Exhibits keep
  declaration order (or handle-sort where they sort at all); the label rides in via
  a final `df.rename()`, which preserves order, so relabeling can never reorder.
  `renamer` (label-sourced) **replaces** `unit_renamer` outright — the old
  heuristic name-guessing is deleted, not extended.
* **D5 — `use_labels` switch → per-object only, no global (settled).** Each object
  sets `self._use_labels = True` in `__init__`; a `use_labels` property setter
  flips it and invalidates the cached `_renamer`. No `constants.USE_LABELS`.
* **D6 — Distortion naming → rip out and realign (settled).** Delete Distortion's
  half-baked `display_name` attribute (it broke repr + other display paths); adopt
  `name` = handle, `display_label` = optional label, `display_name` = resolved
  property, with the auto-pretty `'PH(0.9)'` strings demoted to the derived-default
  rung. Audit every old-`display_name` reader and repoint at the property.
* **D7 — Delivery via one `LabeledMixin` (settled).** The label surface lives in a
  single mixin (the five classes share no base; `Severity` extends scipy), mixed
  into `Aggregate` / `Portfolio` / `PnL` / `Severity` / `Distortion` with an
  explicit `self._init_labels(...)` per `__init__` (no MRO chaining). First mixin
  in the codebase — note the convention in CLAUDE.md.
* **D3 — Scope → high-value core now.** In-scope: **S1** exposure, **S3** inline
  severity clause, **S2** layer, and the **Phase 4 exhibit sweep**. **Deferred:**
  **S4** mixture-component labels (invasive severity mini-language — remains the
  open a124 `dev/TODO.md` item) and **S5** frequency labels (low value). Phases 5
  and 6 below are therefore *out of this pass* — kept in the doc as the tracked
  next step, not executed now.
