# Plan — hygiene 4 (rolling collector)

> **Status: accumulating — do not execute yet.** Rolling list of nits and gnats
> added over time (the author says *"add to hygiene plan: …"*). Executed **as one
> batch** when the group is large enough and the author green-lights it.
>
> **Batch rules**
> - **One version bump for the whole group** — a single `1.0.0a*` increment in
>   `pyproject.toml` covering every item, not one per item.
> - **One `CHANGELOG.md` section** for the batch, bulleting each item.
> - Update `dev/TODO.md` for anything that lands or shifts; move this plan to
>   `dev/done/` once executed.
> - Each item is small and independent unless noted. Items that touch the frozen
>   numeric baseline say so explicitly; default is **no number movement**.
> - `uv run pytest` green before committing; focused regression test per item
>   where cheap.

---

## Item 1 — `Portfolio.value_type` derived from its units (mixed = construction error)

**Goal.** Give `Portfolio` a `value_type` member, mirroring `Aggregate.value_type`
(`'loss'` / `'payoff'`), **derived** from the units it is built from rather than
passed in. This is a dependency of Item 2 (the Portfolio `info` block adds a
`value_type` row, §2.4); Item 2 only *renders* whatever this item produces.

### 1.1 Derivation rule

`Portfolio` holds its constituent aggregates in `self.agg_list` (each an
`Aggregate` with a `.value_type` of `'loss'` or `'payoff'`). The portfolio's
`value_type` is the **unanimous** `value_type` of its units:

- All units `'loss'`  → portfolio `value_type == 'loss'`.
- All units `'payoff'` → portfolio `value_type == 'payoff'`.
- **Mixed** (`'loss'` and `'payoff'` both present) → **error at construction**
  (see §1.3). A portfolio mixing loss and payoff units has no coherent
  "more is worse / more is better" sign convention, so it must be rejected up
  front rather than silently picking one.

The conceptual mapping the author gave: an `agg` carries a **loss** convention;
a `pnl` (premium-net-of-loss / payoff) carries a **payoff** convention. The
portfolio just inherits whichever its members agree on.

### 1.2 Implementation

- Compute the value_type once during `Portfolio.__init__`, **after** `agg_list`
  is populated (portfolio.py:281 area), by reducing over
  `{a.value_type for a in self.agg_list}`.
- Store it on a private `self._value_type` and expose a read-only
  `value_type` **property** (no setter — it is derived, not user-set, unlike the
  `Aggregate` setter). Match the `Aggregate.value_type` docstring style: note it
  is **inert for the distribution** (density / moments / quantiles / allocation
  do not depend on it) and is consumed only by distortion / pricing, where a
  `'payoff'` portfolio is negated / the dual distortion applied.
- **Empty portfolio** (no units): pick the neutral default `'loss'` (matches the
  `Aggregate` default) so the member is always well-defined and the `info` row
  never renders `n/a`.

### 1.3 Construction error on mixed units

Raise a `ValueError` (consistent with the `Aggregate.value_type` setter's
`ValueError`) with a tight, definite message naming the offenders, e.g.:

```
Portfolio '{name}': mixed value_type across units — loss: [a, b], payoff: [c].
A portfolio cannot mix loss and payoff units.
```

Keep it single-line-per-clause and concrete (which units are which), per the
tight-output house style. Raise it where the reduction happens in `__init__`,
before any expensive `update`, so bad books fail fast.

### 1.4 Tests

- Homogeneous loss book → `value_type == 'loss'`.
- Homogeneous payoff book (`pnl` units) → `value_type == 'payoff'`.
- Mixed book → `pytest.raises(ValueError)` at construction, message names the
  offending units.
- Empty portfolio → `value_type == 'loss'`.
- The `info` block contains a `value_type` row (ties into Item 2; can live in the
  Item 2 info test once both land). Add DecL cases to
  `src/aggregate/agg/test_decl.agg` under the matching section if a `port`
  program is used to drive the test.

No frozen numeric baseline moves — `value_type` is a sign-convention label, inert
for the distribution.

---

## Item 2 — consistent `info` strings across `Aggregate` / `Portfolio` / `Distortion` (+ a reference `.rst`)

**Goal.** All three classes emit a fixed-layout plain-text `info` blob: every
instance of a class lays out identically, and the three classes are **as
consistent as their content allows**. Today `Aggregate` is good, `Portfolio` is
middling, `Distortion` is poor (different formatting convention, parenthetical
kind, no canonical param block). This unifies them and writes a reference doc.

Two deliverables: (1) the harmonized `info` methods; (2) a new
`dev/info-strings.rst` (destined for the docs) cataloguing every row of every
class and the full set of possible values where applicable.

### 2.1 Formatting convention (shared)

`Aggregate`/`Portfolio` already use a **left-aligned label padded to a fixed
width (~25 cols), no colon, value follows**, e.g. `severity distribution    …`.
`Distortion._compute_info` instead uses a 2-space indent + colons + varying
widths (`  kind:           ph  (…)`). **Harmonize Distortion onto the
Aggregate/Portfolio convention** (no indent, no colons, same label column), so a
`Distortion` info blob is visually a sibling of the other two. Factor the
label/value formatter into one shared helper if convenient.

**Core principle — no conditional rows. JUST SHOW THE VALUES.** Every row appears
**every time**, in the same order, for every instance of the class. There is **no
`bs>0` gating**, no "shown only when signed", no "shown only for pnl", no
appear/disappear blocks of any kind. A value that is not yet available (e.g. the
object has not been `update`d, or a row is N/A for this object — `premium` on a
non-`pnl` aggregate) renders as a fixed placeholder (`n/a`), but the **row is
always present**. This is what makes the layout truly consistent: two objects of
the same class always produce the same set of lines in the same order.

### 2.2 Union of rows (current → target)

| Row | Aggregate | Portfolio | Distortion | Target |
|---|---|---|---|---|
| `<class> object name` | ✓ | ✓ | `Distortion: {name}` (off-pattern) | all three, same pattern → `distortion object name   {name}` |
| classification / structural | claim count, frequency, severity, approximate | aggregate objects, allocation_method | kind, kind name, value, shape name, other params, mu({0}), mu({1}), interior atoms, gini_p, area | per-class (see layouts) |
| grid / window block | bs, log2, padding, sev_calc, dsev_bucket, normalize, x_min, x_max | bs, log2, padding, sev_calc, normalize, x_min, x_max | — | **always shown** (no `bs>0` gate); same order; `n/a` until updated |
| tail | 3 tail lines (near end) | `tail_description` (currently near **top**) | — | both: tail near **end** |
| `bounded` | **missing** | present **near top** | — | both: in footer (below tail) |
| `value_type` | present (mid-grid) | **missing** | — | both: footer (Portfolio adds it — dependency) |
| `last_update` | **missing** (not added) | `last update` | — | Portfolio only (keep); Aggregate omit |
| `id` | **missing** | `hash` (rename) | `id` (via `id()`) | all three, labelled `id` |

**Remaining asymmetry** (only one): Portfolio shows `last_update`, Aggregate does
not (Aggregate's `id` is a display-only spec hash, no stored timestamp). Both show
`value_type`; `id`, `bounded`, and the shared grid/window/tail blocks line up.
Adding Portfolio `value_type` is **Item 1** and is a dependency of this item.

### 2.3 Target layout — Aggregate

Every row always present, in this order (no gating; `n/a` where unavailable):

```
aggregate object name    {name}
value_type               {loss | payoff}
claim count              {n:.3f}  # note format 
frequency distribution   {freq}
severity distribution    {sev | N components}
approximate              {exact | sgamma | slognorm}
bs                       {bs}
log2                     {log2}
padding                  {padding}
sev_calc                 {sev_calc}
dsev_bucket              {linear | nearest}
normalize                {True | False}
x_min                    {x_min}
x_max                    {x_max}
premium                  {premium | n/a}
expected loss            {expected loss | n/a}  # from numerics; n/a if not updated
loss ratio               {loss ratio | n/a} # 0.0% format
validation_eps           {validation_eps}
reinsurance              {none | occurrence | aggregate | occurrence and aggregate}
occurrence reinsurance   {description | none}
aggregate reinsurance    {description | none}
validation               {explain_validation}
tail (frequency)         {…}
tail (severity)          {…}
tail (aggregate)         {…}
bounded                  {True | False}
id                       {hash}
```

Notes:
- `approximate` is **one row**: `exact`, or `sgamma`/`slognorm` ; **omit** the fitted
  family + params summarised inline (no separate conditional continuation line).
- `x_min`/`x_max` are the realized output window — **replaces** the old
  conditional `window`/`signed severity`/`severity window` block (those rows are
  dropped; just show the two window edges).
- `premium`,  `loss ratio`, `P(loss)` are populated only when premium is known (eg for a `pnl` or when the decl gives premium); otherwise `n/a`. They are **always present**.
- The three `tail (…)` lines come from `_tail.describe_lines` as today, just
  always emitted.

### 2.4 Target layout — Portfolio

```
portfolio object name    {name}
value_type               {loss | payoff}        # ADDED (Item 1; derivation per that work)
aggregate objects        {k}
allocation_method        {linear | lifted}
bs                       {bs}
log2                     {log2}
padding                  {padding}
sev_calc                 {sev_calc}
normalize                {True | False}
x_min                    {x_min}
x_max                    {x_max}
premium                  {premium | n/a}
expected loss            {expected loss | n/a}  # from numerics; n/a if not updated
loss ratio               {loss ratio | n/a} # 0.0% format
tail (frequency)         {…}
tail (severity)          {…}
tail (aggregate)         {…}
bounded                  {True | False}           # MOVED below tail (was near top)
id                       {hash hex}                # RENAMED from "hash"
```

Notes:
- `x_min`/`x_max` are the realized grid window — **replaces** the old conditional
  `signed window` row (always shown now).
- `tail_description` moves from near the top to **near the end** (matching
  Aggregate); `bounded` sits just below it in the footer.

### 2.5 Target layout — Distortion

```
distortion object name   {name}                    # was "Distortion: {name}"
kind                     {abbrev e.g. ph}           # no parenthetical
kind name                {spelled out e.g. proportional hazard}
shape                    {primary shape value | n/a}
shape name               {param_name from the subclass | n/a}
other params             {none | name=val, …}
weights mean             {True / False } # atom at p=0 — mean-component weight}
weights max              {True / False } # mu atom at p=1 — max/ess-sup-component weight
interior atoms           {True | False}             # Bool (not a count)
gini_p                   {2∫g−1 | n/a}
area                     {∫g = (gini_p+1)/2 | n/a}
id                       {id()}
```

Notes:
- **No `display name` row. No `strict-pricing` row. No update time** (a distortion
  is intrinsic — there is no `update`).
- `weights mean` / `weights max` are the Kusuoka measure atoms at `p=0` / `p=1`
  (renamed from `mu({0})` / `mu({1})`, per the author).
- `interior atoms` is the **boolean** from `_kusuoka_summary` (does `μ` have any
  atom in the open interval) — not a count.
- **Param block** uses the existing single-source machinery (`param_name` +
  `decl_params`, a27/a34): `value` = the primary `param_name`'s value (`n/a` if
  `param_name is None`); `shape name` = the `param_name` string; `other params` =
  the remaining `decl_params` as `name=value` (or `none`). Multi-knot / combo
  kinds (`wtdtvar`, `bitvar`, `minimum`, `mixture`) render their knot vectors /
  member names compactly into `other params` so the row set stays fixed.
- `gini_p` may be `NaN` for multi-knot kinds — render as `n/a`; `area` likewise.

### 2.6 The `id` row (all three, label `id`)

Display-only, computed per class (no new stored attributes):
- **Aggregate** — a short hex hash of the canonical spec, e.g. first 8 hex of a
  hash of `json.dumps(self._spec, sort_keys=True, default=str)`. (Author chose
  *reuse existing / display-only*: **no `last_update`, no new hash attribute**.)
- **Portfolio** — the existing `hash_rep_at_last_update`, just relabelled `id`
  (keep the `:x` hex format).
- **Distortion** — the existing `self.id()`.

Match hex formatting across the three so the row reads consistently.

### 2.7 Reference doc — `dev/info-strings.rst`

New `.rst` (destined for `docs/`, not built in the loop) documenting the `info`
contract:
- One section per class, listing **every row in order**: label, meaning, value
  source, and the **placeholder shown when the value is unavailable** (`n/a`).
  (Every row is always present — there is no "when shown" axis.)
- **Full enumerations of possible values** where applicable, read from the code:
  - Aggregate: `approximate ∈ {exact, sgamma, slognorm}`; `value_type ∈ {loss,
    payoff}`; `dsev_bucket ∈ {linear, nearest}`; `normalize ∈ {True, False}`;
    `sev_calc` values; reinsurance-kind labels; tail classes
    (`bounded < super-exponential < exponential < subexponential < power-law`) +
    log-concavity flag; frequency / severity family names.
  - Portfolio: `allocation_method` values; `bounded`; tail classes; grid fields.
  - Distortion: `kind` registry (abbrev → spelled-out long name) and each kind's
    `shape name` / `other params`; `gini_p` / `area`; Kusuoka atoms; `id`.
- A short note that the three share one label-formatting convention.

### 2.8 Resolved decisions & remaining notes

Resolved (author):
- **No conditional rows** — every row always shown, `n/a` when unavailable (2.1).
- Distortion `mu({0})`/`mu({1})` → **`weights mean`** / **`weights max`**.
- Distortion: **drop `display name`**, **drop `strict-pricing`**, **no update time**.
- Distortion `interior atoms` is a **Bool**, not a count.
- Aggregate: **no `last_update`**; `id` is a display-only spec hash (2.6).
- Portfolio: `value_type` **is** shown (added via Item 1).

Remaining (minor; resolve in review or during execution):
- **Placeholder spelling.** Use `n/a` uniformly for an unavailable value (vs blank
  or `—`). *(Default: `n/a`.)*
- **Portfolio `value_type` derivation** is owned by Item 1 — this item just
  renders whatever that produces; don't re-decide the derivation here.

### 2.9 Tests & housekeeping note

`info` is a display string — **no frozen numeric baseline moves**. Add a test that
each class's `info` contains the expected fixed rows in order (construct one of
each, assert the header + footer rows and that Distortion uses the shared
formatting). Update any doctest/guide snippet pinning the old Distortion format or
the old Portfolio `hash`/bounded position. The `.rst` is new; flag docs pending
rebuild.

---

## Item 3 — `stats_df` `prem` / `lr` meta rows populated completely (GROSS)

**Goal.** The `('meta', 'prem')` and `('meta', 'lr')` rows of `stats_df` (and the
matching `el`-derived loss ratio) should be filled in **for every premium-bearing
construct**, not just the exposure-clause `agg … prem at … lr` form. Today:

- `agg MM 100 prem at 65% lr …` → `prem` / `lr` populated (premium flows through
  the exposure clause: `exp_premium` / `exp_lr`, distributions.py:3632-3635).
- `pnl MM 100 prem - …` → `prem` / `lr` **blank/unfilled**. The `pnl` path with a
  full `sev` clause routes premium through `agg_premium` (parser.py:421), and only
  the bare-`lr` shorthand (`_pnl_lr`) backfills `exp_premium` / `exp_lr`
  (parser.py:415-419). So the meta rows never get the premium.

Fix: backfill the `prem` / `lr` meta rows from `agg_premium` whenever the
exposure clause did not already supply a premium, so the `pnl` form reads the same
as the `agg` form.

### 3.1 GROSS (before reinsurance)

When the aggregate has occurrence and/or aggregate reinsurance, `prem` and `lr`
must show the **GROSS** figures — premium and loss ratio computed on the loss
**before** any reinsurance.

This falls out naturally from *where* the meta block lives: the `prem` / `el` /
`lr` rows are written in `Aggregate.__init__` from the theoretical severity
moments (distributions.py:3808-3830), **before** `update_work` applies
reinsurance. So `el` there is already the gross expected loss and `lr = el / prem`
is already a gross loss ratio. The fix only adds the missing premium; it must
**preserve** the gross basis (do not recompute `lr` against a net/ceded loss).
Call this out explicitly in the implementation and in a comment so a later edit
doesn't "helpfully" switch it to net.

### 3.2 Implementation sketch

- In the `__init__` totals block (distributions.py:3808-3830), after `tot_prem` /
  `tot_loss` are summed from the component meta rows, if `tot_prem == 0` (no
  exposure-clause premium) **and** `self._agg_premium` is not `None`, set
  `tot_prem = sum(agg_premium)` (the same total-premium reduction the `pnl`
  affine shift uses, parser.py:420) and recompute `lr = tot_loss / tot_prem` on
  the **gross** `tot_loss`.
- Write the backfilled `tot_prem` / `lr` into the `mixed` and `independent`
  meta columns (and, if it reads better, distribute across components by the
  existing `wt` weights — decide in execution; the total is the must-have).
- Leave the exposure-clause path untouched (it already populates these).

### 3.3 Tests & baseline

- `pnl MM 100 prem - …` (with and without a stated `lr`) → `stats_df.loc[('meta',
  'prem'), 'mixed']` equals the stated premium; `('meta', 'lr')` equals
  `el / prem`.
- `agg MM 100 prem at 65% lr …` unchanged (regression guard).
- A `pnl` **with reinsurance** → `prem` / `lr` are the **gross** values (assert
  `lr` matches gross `el / prem`, not the net loss).
- Add the DecL programs to `src/aggregate/agg/test_decl.agg` under the matching
  section.
- **Baseline note:** `stats_df` is a captured baseline frame
  (`tests/baseline/`). Filling previously-blank `prem` / `lr` cells **changes the
  captured `stats_df`** for affected `pnl` lines → the baseline snapshot needs
  regenerating for those cases. This is an intended, explainable change (blank →
  correct gross premium/lr), not numeric drift in the distribution itself — note
  it in the CHANGELOG. No `describe` / density / risk-measure numbers move.

---

## Item 4 — `value_type` labels (`'loss'` / `'payoff'`) configurable

**Goal.** The two `value_type` label strings — currently the hardcoded literals
`'loss'` and `'payoff'` — should come from **config**, with `'loss'` / `'payoff'`
as the shipped defaults but user-customizable. (Author note: *not happy with
`'payoff'`* — so the default itself is likely to change later; routing it through
config means that future rename is a one-line default change, not a scatter-edit.)

### 4.1 Where the literals live today

The pair is hardcoded at three sites (rg `'payoff'`):
- `distributions.py:3495` — `__init__` coercion
  (`value_type if value_type in ('loss', 'payoff') else 'loss'`).
- `distributions.py:4411-4415` — the `value_type` setter validation /
  `ValueError` message.
- `parser.py:424` — `_attach_pnl` sets `spec["value_type"] = "payoff"`.

There is **no pricing-side `== 'payoff'` branch yet** (the negate / dual-distortion
consumer is still downstream/future per the `value_type` docstring). Doing this
*now*, before that branch is written, is the cheap moment — the semantic consumer
can be built against the config from the start instead of hardcoding the literal.

### 4.2 Config plumbing

Follow the existing `config.py` dataclass pattern (cf. `DiscretizationSettings`):
- Add a frozen settings section — e.g. `LabelSettings` with two fields,
  `loss: str = 'loss'` and `payoff: str = 'payoff'` — and register it in
  `_SECTIONS`. (Section name `labels` — or fold into a broader `display`/`naming`
  section if one is wanted; decide in execution. One section is fine.)
- Add it to `data/config.default.toml` (the annotated, fully-commented template
  `write_default_config` ships), with a comment that these are display /
  semantic-convention labels and that changing them renames the values everywhere
  (`info`, `stats_df`, the `value_type` setter's accepted inputs).
- Optionally add `AGGREGATE_*` env entries to `_ENV_MAP`
  (`AGGREGATE_VALUE_TYPE_LOSS` / `_PAYOFF`) for parity with the other tunables —
  low priority, include only if cheap.

### 4.3 Design decision (RESOLVED) — store the canonical role, map to label at the edges

`value_type` is really a **two-valued semantic role**: *loss convention* ("more
is worse", the default) vs *payoff convention* ("more is better"). The label text
is just how that role is displayed / spelled in the DSL.

**Decided (author):** store the role internally as a **private boolean
`_is_loss_value`** (the spec member; `True` = loss convention, `False` = payoff
convention). `loss` is the anchor on purpose — it is the unambiguous, never-going-
to-change pole for actuaries, whereas the *payoff* label is the one the author may
rename. The boolean **is never reconfigured**; only the displayed/accepted label
text comes from config. Resolve role ↔ label only at the boundaries:
- **store** — `Aggregate` (and `Portfolio`, Item 1) carry `_is_loss_value`; this
  is the canonical spec field. Nothing else stores the label string.
- **display** — `info` row (Item 2), `stats_df`, any repr render
  `settings.labels.loss if self._is_loss_value else settings.labels.payoff`.
- **parse / set** — the `value_type` setter and `__init__` accept *either* the
  canonical token (`'loss'`/`'payoff'`) *or* the currently-configured label, map
  it to the boolean, and reject anything outside the configured pair. The public
  `value_type` getter returns the configured **label** (back-compatible string),
  computed from `_is_loss_value`.
- **future pricing consumer** — branches on `_is_loss_value`, never on label text,
  so the eventual `'payoff'` rename (or any user relabel) cannot silently break
  the negate / dual-distortion logic.

This keeps configurability entirely out of the math: a relabel changes only the
printed word. (The rejected alternative — store the literal configured string and
compare against `settings.labels.payoff` everywhere — couples pricing to a
user-tunable string and mis-routes existing objects across a mid-session relabel.)

### 4.4 Interaction with Items 1 & 2

- **Item 1** (`Portfolio.value_type`) derives and exposes a `value_type`; it must
  report the **configured label**, and its mixed-units error message should use
  the configured labels too.
- **Item 2** (`info` strings) renders the `value_type` row for Aggregate and
  Portfolio and enumerates `value_type ∈ {loss, payoff}` in `info-strings.rst` —
  that enumeration must read the **configured** pair, not hardcode the words.
- Ordering: this is config plumbing and can land with or just before Items 1/2.
  Whichever lands first, the others consume `settings…labels`, not literals. Note
  the dependency in the batch so the three stay consistent.

### 4.5 Tests & housekeeping

- Default config → `value_type` reads `'loss'` / `'payoff'` exactly as today
  (regression; `test_pnl.py` / `test_negative_x.py` keep passing unchanged).
- Override the labels in a temp config (or via `reload_settings` with a patched
  settings) → a `pnl` reports the custom payoff label in `info` / `stats_df`, and
  the setter accepts the custom label and rejects anything outside the configured
  pair.
- Canonical role is stable under relabel: an object built as payoff keeps
  `_is_loss_value is False` after the payoff label is changed, and still routes
  through the payoff / "more is better" role (guards the §4.3 invariant — the
  boolean does not track the label string).
- No frozen numeric baseline moves — labels are display / semantic tags, inert for
  the distribution. Update `describe_settings` / config docs to list the new
  section; flag docs pending rebuild.

---

<!-- Append new items below as "## Item N — …" when the author says
     "add to hygiene plan: …". Keep one version bump for the whole group. -->
