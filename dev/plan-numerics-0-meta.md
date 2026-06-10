# Numerics rationalization — meta-plan

> **What this is.** The plan for the plans. It does not write code. It fixes the
> target architecture, carves the work into executable plans, sets the steering
> rules they all obey, and reconciles them with what's already drafted/tracked.
> Source material: the four notes in `../math/docs`
> (`shifted-calc-method.md`, `choquet-calc-method.md`,
> `portfolio-calc-methods.md`, `density_df-in-a-windowed-world.md`), the current
> code, and TODO items **N1/N2/N3** + the former `plan-portfolio-neg-x-pricing`
> draft (removed; recoverable in git history) and the `plan-window-port-bv` draft
> (was untracked, so **not** in git — its content survives only in numerics-4's
> Scope section). Both folded in here.

---

## 1. The target architecture (one picture)

The four notes describe one coherent end state. Three owners, no overlap:

```
Aggregate / unit          owns native-grid unit pmfs        (p on its own window)
Portfolio (total grid)    owns conditional allocation       (kappa, alpha, beta, exa, exag)
Distortion / spectral     owns distorted total weights      (gp = g(T)-g(S), rho)
a pentagon formatter      owns L / M / P / Q / a / LR / PQ / ROE
```

The computational core is **much smaller than today's column surface**. These are
the **canonical column names we keep and use everywhere** (per-line suffix `{line}`;
the "for total loss `X`" part is silent, so `exi_xgta` reads `E[X_i/X | X>a]`):

| Math | Column (canonical) | Meaning |
|---|---|---|
| `p_k`, `S=P(X>x)`, `T=S+p` | `p_total`, `S`, (`T`) | exact discrete **total atom table** |
| `kappa_i(x)=E[X_i∣X=x]` | `exeqa_{line}` | conditional line mean (shifted-support FFT) |
| `s_i(x)=kappa_i/x` | `exi_xeqa_{line}` | point share (positive-loss guarded) |
| `alpha_i(a)=E[X_i/X∣X>a]` | `exi_xgta_{line}` | **objective** tail share |
| `gp_k=g(T_k)-g(S_k)` | `gp_total` | distorted atom weights (one Choquet helper) |
| `beta_i(a)=E_g[X_i/X∣X>a]` | `exi_xgtag_{line}` | **distorted** tail share |
| `L_i(a)=E[X_i(a)]` (cum loss) | `exa_{line}` | objective equal-priority allocation |
| `P_i(a)=E_g[X_i(a)]` (cum premium) | `exag_{line}` | distorted (priced) allocation |

`exi_xgta` **is** alpha and `exi_xgtag` **is** beta — those are the names used
throughout the plans. The two allocation columns are **one symbol apart** (the
*only* linear-vs-lifted difference):

```
exag_i = cumsum_k(exeqa_i · gp_total)  +  x · gS · TAIL_i
   linear:  TAIL_i = exi_xgta_i    (alpha)
   lifted:  TAIL_i = exi_xgtag_i   (beta)
```

The layer densities `S·exi_xgta` and `gS·exi_xgtag` (the old `M.L_*` / `M.P_*`) live
only in the diagnostic frame. Everything else — `exlea/exgta`, the ratios, the
pentagon — is a *view* derived from the table above. The single biggest move is to
stop computing values with the zero-origin idiom `cumsum(S) * bs` and compute them as
`dot(x, weights)` / direct prefix sums that **carry the `x0` term**. That one
change is what simultaneously (a) makes signed/windowed supports correct and (b)
collapses linear-vs-lifted and the three price surfaces into one engine.

### What the user has decided (overrides the notes)

- **Kill `T.*` and `M.*` columns.** The `portfolio-calc` note recommends *keeping*
  them as code-names plus a `human_renamer`. We are not doing that — they go.
  Replace the handful of real consumers (`pricing_at`, `pentagon_at`,
  `price(lifted)`, `pentagon.py`) with explicit per-line `L/M/P/Q` columns; give
  `plot_twelve` the layer curves through an explicit diagnostic frame.
- **Kill the entire EPD / second-priority family** — `e2pri_*`, the portfolio
  `epd_0_* / epd_1_* / e1xi_1gta_*` (`add_exa_details`, already **no callers**),
  **and** `Aggregate.density_df['epd']` (cheap to recompute, unused). All gone.
  (`e2pri_*` no longer exists in the tree; it was 0.30.1-era.)
- **linear and lifted resolved at `apply_distortion`; no difference downstream.**
- **`plot_twelve` is not the schema authority.** It is a consumer; it sources what
  it needs from a diagnostic frame and native unit pmfs.
- **No compatibility crutches for downstream.** We do not keep a column, idiom, or
  primitive *only* because a downstream consumer reads it — we revise the consumer
  to the new design. `AllocationBounds`, `pentagon.py`, `pricing_at`/`pentagon_at`,
  and `plot_twelve` all move to the new surface. Clean final design, no bridge layer.

---

## 2. The decomposition — four plans

The user's instinct (a possible early `p_unit` step, then *objective*, then
*distortion*) is right. Refined, with the reasoning for each boundary. Plan files
are named to sort as a block:

| Plan file | Name | Scope | Distortion? |
|---|---|---|---|
| `plan-numerics-1-unit-density` | Unit-density decoupling | accessors + migrate display readers off `p_{unit}` | no |
| `plan-numerics-2-objective` | Objective spine | shifted-support kappa + direct-sum objective columns; drop `p_{unit}` write & dead EPD; signed objective falls out | no |
| `plan-numerics-3-distortion` | Distortion spine | one Choquet helper; unified linear/lifted at `apply_distortion`; `T.*`/`M.*` gone; Aggregate-side alignment; `plot_twelve` adapter; `value_type` axis | yes |
| `plan-numerics-4-windowed-combine` | Windowed combine | windowed portfolio combine + bivariate per-axis windowing, on the kappa-ready base | (reuses 2/3) |

**Why numerics-1 is worth splitting out.** It is pure-additive accessors plus
mechanical reader migration — low risk, independently testable, and it shrinks
numerics-2's blast radius to *only* the kappa compute. It is exactly "set up the
mechanism to grab `p_unit` from the underlying Aggregate objects." It does **not**
stop writing `p_{unit}` (the kappa path in `add_exa` still reads it); it just
ensures nothing *else* does.

**Why kappa + p_unit-removal must live together in numerics-2.** `add_exa` builds
kappa as `ft(df.loss * df[p_i]) · ft_nots[i] / p_total` — it *needs* `p_{unit}` on
the total grid. You cannot remove the `p_{unit}` write until kappa is computed the
shifted way (native `m_i = (a_i + r·bs)·p_i`). So the final removal is the
*consequence* of the shifted-kappa switch and belongs in the same plan, not earlier.

**Why negative-x is not its own plan anymore.** The former
`plan-portfolio-neg-x-pricing` draft and these notes are two views of the same N2/N3
work. The direct-sum / exact-discrete formulas the notes *mandate* are *precisely*
what removes the `loss ≥ 0` assumptions. So signed support dissolves into numerics-2
(objective half) and numerics-3 (distortion + `value_type` half). That draft is
**superseded**; its one durable asset — the discipline of a *measured*
column-by-column audit against a brute-force reference — is carried in as a
numerics-2/3 deliverable. (The `value_type` DecL *keyword* and the "what does capital
allocation mean on a straddling P&L" semantics stay deferred.)

### numerics-1 — Unit-density decoupling

- **Add** `Portfolio.unit_density(unit, view='agg')`, `unit_density_df(view='agg')`,
  `aligned_unit_density_df(grid='total'|'union'|'zero')` — sourcing from each
  `agg.density_df` on its native grid. (`Aggregate` already models the split:
  `density_df` vs `sev_density_df`.)
- **Migrate display/non-kappa readers** off `density_df[p_{unit}]`: `percentiles`,
  `plot`, `_limits`, `reins_density_df` display, `sample_density_compare`,
  `var_dict` unit quantiles, and `pedagogy` density panels.
- **Keep** writing `p_{unit}` for now (kappa still needs it). Mark legacy.
- **Tests:** `unit_density(u).sum()` == represented mass; disjoint-window book's
  `unit_density_df` rows match each unit's own `density_df`; legacy
  `aligned_unit_density_df(grid='total')` reproduces the old columns.

### numerics-2 — Objective spine (no distortion)

- **Shifted-support kappa** (`shifted-calc-method.md`): stash per-unit
  `origin_i, p_i_native, ft_p_i, ft_xp_i` in `update`; build `kappa_i = ifft(ft_xp_i ·
  ft_not_i)/p_total_shifted` with **prefix/suffix products** for `ft_not_i` (spectral
  division only behind a nonzero-bin test); rebase onto the total output window.
- **Direct-sum objective columns** (`portfolio-calc` core formulas): `exlea/exgta/
  exi_xgta/exa` from `kappa` + `p_total` by prefix sums that carry `x0`; retire
  `cumsum(S)*bs`, `df.loc[0:loss_max]`, the `mult∈{1,10,100}` heuristic, and replace
  with explicit `F<tol / S<tol / |x|<tol` guards.
- **Drop the `p_{unit}` write** from `update` once kappa no longer reads it; delete
  `add_exa_details` and the whole EPD family.
- **Aggregate-side objective columns** in the same pass: `density_df`'s
  `lev/exa/exlea/exgta` to direct sums that carry `x0` (windowed-safe), and **remove
  the `epd` column**. (Objective Aggregate work rides here; the Aggregate *distortion*
  surface, `apply_distortion/price/exag`, is numerics-3.)
- **Signed objective falls out**: re-enable the signed `add_exa` path (remove the
  warn+fallback) for the *objective* columns; the equal-priority `kappa/loss` ratio
  stays behind a positive-loss guard.
- **Invariants / tests** (small exact finite laws): `Σ_i kappa_i(x) == x` on support;
  `Σ_i exa_i == exa_total`; disjoint-window two-unit book correct; `e_i` from native
  pmf reconciles `Σ kappa_i·p_total`; shifted kappa matches brute-force convolution
  incl. negative origins; **full non-negative regression at 1e-14 on key columns**
  (captured baselines; derived columns spot-checked, D5) — this is a pure refactor
  for the legacy case.

### numerics-3 — Distortion spine

- **One Choquet helper** in `spectral.py` (`choquet-calc-method.md`): exact discrete
  `S/T`, `gp = g(T)-g(S)`, `rho = dot(x, gp)`, `rho(X∧a) = dot(min(x,a), gp)`; deficit
  policy (clean fuzz → measure deficit → raise if material); returns `S,T,gS,gp`.
  Built on / replacing the existing `Distortion.price`(`method='dx'/'ds'`) and
  `make_q`. **Bounded/mass gatekeeping stays in Aggregate/Portfolio** (Distortion
  can't know boundedness; the caller certifies).
- **Unified linear/lifted builder** at `apply_distortion`: both emit the *same*
  column schema; the only difference is the collapsed-tail share — **linear uses
  `alpha`, lifted uses `beta`** (`portfolio-calc` two-line formula). `price` reads
  rows for *both*; the separate `_collapsed_exeqa` linear pricing engine is deleted.
  **`AllocationBounds` moves onto the unified builder too** (D2) — `_collapsed_exeqa`
  is absorbed/retired, not kept as a downstream primitive.
- **`T.*`/`M.*` removed**: explicit per-line `L/M/P/Q` columns by direct sums;
  migrate `pricing_at`, `pentagon_at`, `price(lifted)`, `pentagon.py`.
- **Cache key** widened to `(name, view, value_type, S_calculation, allocation)` —
  the `value_type` slot keys on the canonical role flag `_is_loss_value`, not the
  configurable label string (hygiene-4 Item 4); `efficient` removed entirely (D7 — a
  non-issue once diagnostics move out of the core frame).
- **Aggregate distortion surface**: `Aggregate.apply_distortion/price` (`exag`) route
  through the helper; the six price surfaces must agree on the total premium (the
  `choquet` note's consistency target). (Aggregate *objective* columns were done in
  numerics-2.)
- **`plot_twelve` adapter**: an explicit `allocation_diagnostics(distortion,
  surface=...)` frame (layer curves `S·alpha`, `gS·beta`, layer margin) + native unit
  pmfs; remove the efficient→full cache-pop hack.
- **`value_type` as the second pricing axis** (D3): `view ∈ {ask, bid}` and the
  value-type **role** (loss vs payoff convention) are **orthogonal** inputs; each
  independently toggles `g ↔ g_dual`, composing by XOR — use the dual iff
  `(view==bid) XOR (payoff role)`. **Branch on the canonical role flag
  `_is_loss_value`** (hygiene-4 Item 4 — `payoff role == not _is_loss_value`),
  **never on the label string**: the `loss`/`payoff` labels are user-configurable
  (hygiene-4 Item 4), so a literal `value_type == 'payoff'` comparison would couple
  pricing to a tunable string and mis-route after a relabel. No separate
  negate-the-variable path: the exact-discrete `rho = dot(x, gp)` engine is
  orientation-agnostic, so the role only selects the effective `g`. DecL keyword for
  `value_type` deferred.

  | | loss | payoff |
  |---|---|---|
  | **ask** | `g` | `g_dual` |
  | **bid** | `g_dual` | `g` |
- **Invariants / tests:** identity distortion ⇒ `beta==alpha`, `premium==loss`,
  margin≈0; one-line port: line==total; linear & lifted agree at/above max support and
  differ predictably below; mass-on-unbounded raises in the builder (not just
  `price`); all six surfaces agree; signed P&L prices as a signed variable.

---

## 3. Steering rules (binding on all plans)

1. **The bucketed law is an exact discrete atom table, not a curve.** Primary object
   is `(x_k, p_k)`. Compute values as `dot(x, weights)` / direct prefix sums that
   **carry `x0`**. Never reintroduce `cumsum(S)*bs` or `loss[0]==0` assumptions. The
   layer `gS·dx` form survives **only** as a reconciliation `assert` — with **one
   deliberate exception**: the on-demand per-line capital `Q_i(a)` in numerics-3
   deliverable 4 *is* layer-based, because capital genuinely is allocated by layer
   (the layer-ROE construction); that is sanctioned there, not a rule violation.
2. **Three owners, no overlap** (the picture in §1). Portfolio never reimplements a
   distorted integral; Distortion never knows line names / equal priority / boundedness.
3. **Total-grid vs native-grid line is hard.** `Portfolio.density_df` carries
   total-grid columns *only* (`p_total, F, S, exeqa_*=kappa, exa_*, exag_*`). Unit pmfs
   live on the `Aggregate` and are reached through accessors. Any unit-pmf-on-total-grid
   view is explicitly named a display artifact and **never** feeds compute.
4. **linear/lifted is a tail-share choice (`alpha` vs `beta`), decided once at
   `apply_distortion`.** Everything downstream reads columns and cannot tell which
   produced them.
5. **Gatekeeping (bounded / mass / material deficit) lives in Aggregate/Portfolio.**
   Distortion computes unconditionally; the caller that knows the support certifies.
6. **`kappa/loss` (equal-priority share) stays behind a positive-loss guard.** On
   signed/zero-crossing grids, price the signed variable directly via `dot(kappa, gp)`;
   never silently divide by `loss`.
7. **Pedagogy consumes, never dictates.** `plot_twelve` (and friends) get an explicit
   diagnostic frame + native unit pmfs; no `efficient=False` columns or `p_{unit}` in
   the core pricing frame, no cache mutation hacks.
8. **Tests are small exact finite laws with hand-checkable invariants** (each note
   ships its list — bake them as asserts), **plus** a legacy non-negative regression
   at **1e-14 relative tolerance** (D5). fp drift from changed order of operations is
   expected wherever `cumsum(S)*bs` becomes a direct sum; byte equality applies only
   where the computation path is untouched (e.g. `p_total`). **Key columns are `p_*`
   and `exeqa_*`** — everything else is derived from them — so key columns get the
   full baseline gate and derived columns get spot checks, not full byte capture.
   For the legacy zero-origin case these are refactors, not behavior changes.
9. **Noise floors stay tight** (1e-12…1e-14, per house rule), not loose 1e-8; the math
   is essentially exact.
10. **Breaking changes are expected and fine** (`T.*`/`M.*`/`epd_*`/`p_{unit}` removal)
    — pre-1.0 alpha — but each is called out in `CHANGELOG.md` and every internal /
    pedagogy consumer is migrated *in the same plan*.
11. **House rules:** one version bump per plan; `CHANGELOG.md` at iteration close;
    `dev/TODO.md` (N1/N2/N3 + W-track) updated and plan moved to `dev/done/`; US
    spelling; `reins` canonical; mirror any new DecL into `test_decl.agg`; **do not
    build docs in the loop** (keep `.rst` `:meth:`/`:class:` refs in lockstep, note
    pending rebuild). For *removed/renamed columns* in the docs (`T.*`/`M.*`/`epd`/
    `p_{unit}`): **fix where unambiguous** — replacements derivable by simple math
    from the new columns — and **flag the leftover unfixable kernel** for the author
    rather than guessing.

---

## 4. Roadblocks & gotchas (assessment: safe, but eyes open)

You're right that there's no architectural roadblock — the notes are internally
consistent and the signed combine already proves the hard FFT mechanics. The moving
parts that *will* bite if unmanaged:

- **G1 — `update` must persist per-unit state for kappa.** `ft_xp_i` (FFT of the
  first-moment density `(a_i+r·bs)·p_i`) is **not stored today** and cannot be
  reliably reconstructed after rebasing. The signed combine already drives units on
  their own windows, so origins/native pmfs are reachable; we add `ft_xp_i`. This is
  the one genuinely new piece of machinery. Lifecycle settled (D6): transient —
  captured at combine, consumed by `add_exa` in the same `update` call, freed before
  `update` returns.
- **G2 — `ft_not_i`: prefer prefix/suffix products** over `ft_all/ft_line` spectral
  division (symmetric severities zero FFT bins). Keep division behind a nonzero test.
- **G3 — deficit policy on windowed/unbounded supports.** A finite window leaves
  `1 - Σp` of mass at unknown loss; forwards-`S` parks it at the top atom, backwards-`S`
  zeroes it. For Choquet pricing a *material* deficit is an economic error, not float
  dust → clean fuzz, measure, raise if material. Interacts with numerics-4.
- **G4 — `_collapsed_exeqa` is shared with `AllocationBounds`.** Resolved (D2): bounds
  moves onto the unified builder; `_collapsed_exeqa` is absorbed/retired. The tail-
  collapse math becomes one owner inside the builder — verify `AllocationBounds`
  (convex-hull slicing) reproduces its current results against the captured baseline.
- **G5 — `T.*`/`M.*` removal touches `pentagon.py` + `pricing_at`/`pentagon_at`/
  `price(lifted)`.** Migrate in lockstep within numerics-3 or the pentagon read breaks.
- **G6 — mass-distortion-on-unbounded guard must move *down*** into the builder; today
  it only fires in `price(lifted)`, so `apply_distortion`/`exag_total` can still build
  an unstable frame.
- **G7 — sampling/switcheroo cluster reads `p_{unit}` too** (found in the
  pre-execution review): `Portfolio.sample`, `swap_density_df` (method + module
  twin), `add_exa_sample`, `make_awkward`. Its *design* hasn't been considered and
  is **deferred to its own future plan** — numerics-1 leaves it untouched (the
  write stays), numerics-2 deliverable 4 gives it only the minimal mechanical
  re-source onto the accessors so dropping the write doesn't break pytest. Do not
  let either plan grow a sampling redesign.

---

## 5. Reconciliation with existing plans & TODO

- **TODO N1/N2/N3** *are* this work. N1 (combine) is largely landed (signed path).
  **numerics-2 = N2** (update: kappa / exa / `add_exa` + neg-x objective half).
  **numerics-3 = N3** (apply-distortion calcs: columns, masses, `plot_twelve`, +
  neg-x distortion half). numerics-1 is a new, smaller pre-step under N2; numerics-4
  is the W2/M-track. Update the N-track rows on landing.

Both sibling drafts have been **folded into this program and removed** — they are not
separate live dependencies. (`plan-portfolio-neg-x-pricing` was tracked, so recoverable
in git history; `plan-window-port-bv` was untracked and is gone — its content survives
only in numerics-4.)

- **hygiene-4 is a live upstream dependency of numerics-3** (not folded in — it
  ships first, separately). It delivers: `Portfolio.value_type` derived from its
  units with the mixed-book raise (Item 1, = D8); the canonical role flag
  `_is_loss_value` on `Aggregate`/`Portfolio`; and user-configurable `loss`/`payoff`
  labels (Item 4). numerics-3 **consumes** all three — it builds the `view×value_type`
  pricing axis on `_is_loss_value` and does **not** re-implement the Portfolio
  derivation or the mixed-book error. Sequencing: land hygiene-4 before numerics-3.

- **`plan-portfolio-neg-x-pricing`** (was tracked) — folded into numerics-2
  (signed objective columns) + numerics-3 (signed / `value_type` distortion pricing).
  Its one durable asset — *measure the columns against a brute-force reference, don't
  guess* — is carried as the Step-0 audit in both plans.
- **`plan-window-port-bv`** (was untracked) — folded into numerics-4 (windowed
  *combine* + bivariate per-axis windowing + the P0 occ-reins × windowing and
  mixed-book decisions), re-based onto the kappa-on-total-grid path numerics-2
  establishes (per-unit origins handled there).

---

## 6. Recommended sequence

```
plan-numerics-1-unit-density   (accessors + reader migration; low risk, no compute change)
  └─ plan-numerics-2-objective    (shifted kappa + direct-sum objective; drop p_unit write; kill EPD; Aggregate objective cols)
       └─ plan-numerics-3-distortion  (Choquet helper + unified linear/lifted incl. bounds; kill T./M.; Aggregate distortion; plot_twelve adapter; view×value_type)
            └─ plan-numerics-4-windowed-combine  (supersedes the former window-port-bv draft) — now on a kappa-ready base
```

Each plan is independently shippable with green `pytest`, a version bump, and a
legacy regression at 1e-14 on key columns (D5; byte-equal where the computation
path is untouched).

**Decisions settled (D1–D4):** D1 — `p_{unit}` write removal lands wherever easiest in
sequencing (gone by end of numerics-2 regardless). D2 — `AllocationBounds` moves onto the
unified builder; no primitive kept for downstream. D3 — `view` and `value_type` are two
orthogonal axes, each toggling `g/g_dual` (XOR); one orientation-agnostic dot engine.
D4 — the whole EPD/`e2pri` family removed, **including** `Aggregate.density_df['epd']`.

**Decisions settled (D5–D8, second review round):** D5 — the regression gate is
**1e-14 relative** on the key columns `p_*`/`exeqa_*` (everything else is derived;
spot-check it); byte equality only where the computation path is unchanged — fp
drift from changed order of operations is expected and fine. D6 — per-unit FT state
(`ft_p_i`, `ft_xp_i`) is **transient within `update`**: computed with the unit at
combine time, consumed by `add_exa` immediately afterwards, padded FT arrays freed
before `update` returns; only scalars (origins) and native pmfs persist. D7 — **no
persistent per-line `Q` column** (nothing consumes the full curve): line capital is
computed on demand at the requested `a` via the layer-ROE construction inside
`pricing_at`/`pentagon_at`/`price`; **`efficient` is removed entirely** (the methods
it gated go away); **`S_calculation` survives only as the deficit-parking direction**
— on a clean law forwards/backwards agree to tol (asserted), and a material
difference means the law is defective and the deficit policy governs. D8 —
Portfolio `value_type` (settled, and **now implemented in hygiene-4 Item 1** — a
dependency of numerics-3, not a numerics deliverable): a Portfolio's units must be
consistent and the Portfolio must know its level; **homogeneous books adopt the
units' common level** (default `loss`); **mixed books raise** with a clear error —
no magic, no invisible reversal — and the user redeclares the inconsistent units
explicitly (reflect / `pnl` at declaration). The canonical internal storage is the
role flag `_is_loss_value` on both `Aggregate` and `Portfolio` (hygiene-4 Item 4);
numerics-3 **consumes** it and never re-derives the derivation or the mixed-book
error.
