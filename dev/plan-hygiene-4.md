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

## Item 1 — consistent `info` strings across `Aggregate` / `Portfolio` / `Distortion` (+ a reference `.rst`)

**Goal.** All three classes emit a fixed-layout plain-text `info` blob: every
instance of a class lays out identically, and the three classes are **as
consistent as their content allows**. Today `Aggregate` is good, `Portfolio` is
middling, `Distortion` is poor (different formatting convention, parenthetical
kind, no canonical param block). This unifies them and writes a reference doc.

Two deliverables: (1) the harmonized `info` methods; (2) a new
`dev/info-strings.rst` (destined for the docs) cataloguing every row of every
class and the full set of possible values where applicable.

### 1.1 Formatting convention (shared)

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

### 1.2 Union of rows (current → target)

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
Adding Portfolio `value_type` is in-progress and is a dependency of this item.

### 1.3 Target layout — Aggregate

Every row always present, in this order (no gating; `n/a` where unavailable):

```
aggregate object name    {name}
value_type               {loss | payoff}
claim count              {n}
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
E[loss]                  {E[loss]}
loss ratio               {loss ratio | n/a}
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

### 1.4 Target layout — Portfolio

```
portfolio object name    {name}
value_type               {loss | payoff}        # ADDED (dependency; derivation per that work)
aggregate objects        {k}
allocation_method        {linear | lifted}
bs                       {bs}
log2                     {log2}
padding                  {padding}
sev_calc                 {sev_calc}
normalize                {True | False}
x_min                    {x_min}
x_max                    {x_max}
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

### 1.5 Target layout — Distortion

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

### 1.6 The `id` row (all three, label `id`)

Display-only, computed per class (no new stored attributes):
- **Aggregate** — a short hex hash of the canonical spec, e.g. first 8 hex of a
  hash of `json.dumps(self._spec, sort_keys=True, default=str)`. (Author chose
  *reuse existing / display-only*: **no `last_update`, no new hash attribute**.)
- **Portfolio** — the existing `hash_rep_at_last_update`, just relabelled `id`
  (keep the `:x` hex format).
- **Distortion** — the existing `self.id()`.

Match hex formatting across the three so the row reads consistently.

### 1.7 Reference doc — `dev/info-strings.rst`

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

### 1.8 Resolved decisions & remaining notes

Resolved (author):
- **No conditional rows** — every row always shown, `n/a` when unavailable (1.1).
- Distortion `mu({0})`/`mu({1})` → **`weights mean`** / **`weights max`**.
- Distortion: **drop `display name`**, **drop `strict-pricing`**, **no update time**.
- Distortion `interior atoms` is a **Bool**, not a count.
- Aggregate: **no `last_update`**; `id` is a display-only spec hash (1.6).
- Portfolio: `value_type` **is** shown (added via the in-progress dependency).

Remaining (minor; resolve in review or during execution):
- **Placeholder spelling.** Use `n/a` uniformly for an unavailable value (vs blank
  or `—`). *(Default: `n/a`.)*
- **Portfolio `value_type` derivation** is owned by the separate in-progress
  value_type work — this item just renders whatever that produces; don't re-decide
  the derivation here.

### 1.9 Tests & housekeeping note

`info` is a display string — **no frozen numeric baseline moves**. Add a test that
each class's `info` contains the expected fixed rows in order (construct one of
each, assert the header + footer rows and that Distortion uses the shared
formatting). Update any doctest/guide snippet pinning the old Distortion format or
the old Portfolio `hash`/bounded position. The `.rst` is new; flag docs pending
rebuild.

---

<!-- Append new items below as "## Item N — …" when the author says
     "add to hygiene plan: …". Keep one version bump for the whole group. -->
