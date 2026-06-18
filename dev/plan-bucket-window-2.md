# Plan bucket-window-2 — symmetric, convention-aware output windowing

> **STATUS (2026-06-17).** **1A** (Aggregate sizer) landed a58 and **1A-fix**
> (single-big-jump extent floor) landed a59 — see the §1A and §1A-fix sections
> below. The further *univariate* bucket work — a first-class thick/thin tail
> report (freq/sev/agg), wiring it into `bs`/`x_min` selection (incl. the
> `windowed`↔`sbj` asymmetric window and tail-aware padding), and making the
> decision legible (public `bs_window_df`, narrative `bs_description` /
> `bs_explanation`) — landed in **`dev/done/plan-univariate-bucket.md`** (the
> "1A-bucket" plan, complete a62–a65), which owns `dev/bucket-selection.rst`.
> **1P** (the
> Portfolio windowed combine) remains open *here*; we return to it after
> 1A-bucket, and it inherits 1A-bucket's reporting surfaces (`tail_df`,
> `tail_description`/`tail_explanation`, `bs_window_df`,
> `bs_description`/`bs_explanation`) at the `Portfolio` level.

> **What this is.** A **round-1 design exploration**, not an execution plan. It
> diagnoses the current window/`bs` asymmetry, fixes the design requirements,
> proposes methods, and surfaces the decisions that rounds 2–3 must settle. **It
> writes no code.** Successor to `dev/done/plan-bucket-window.md` (the a51
> non-zero-window work); feeds `dev/plan-numerics-4-windowed-combine.md`.
>
> The author expects 2–3 planning rounds. This is round 1: frame the methods and
> the open questions; do not finalize. Decisions are tagged **[Q1]…** for the
> next round.

---

## 1. The problem (with numbers)

Under the **loss** sign convention (positive = loss = bad) the interesting tail
is the **right** (quantiles `p → 1`: VaR, TVaR, capital). Under the **payoff**
convention it is the **left**. The current sizer is one-sided *and asymmetric
the wrong way for loss*: for a non-signed aggregate it forces `x_min = 0` and
estimates only an upper edge `x_hi` via `_estimate_agg_percentile`, then sizes
`bs` so the grid top ≈ `x_hi`. Measured today (a57):

| book | cv | grid | mass band | empty below | empty above | tail clip? | method |
|---|---|---|---|---|---|---|---|
| 1. `250 claims … lognorm 100 cv 1.5` | 0.10 | `[0, 65 535]` | `[10 147, 46 991]` | **15%** | 28% | no | moment |
| 3. `5000 claims … lognorm 100 cv 2` | 0.03 | `[0, 655 350]` | `[399 640, 655 350]` | **61%** | 0% | **yes, 3.9e-7** | moment |
| 4. `50 000 claims … lognorm 100 cv 1` | 0.006 | `[4.78M, 5.44M]` | `[4.78M, 5.23M]` | 0% | 31% | no | **windowed** |

Three distinct failure shapes:

- **Case 1** — mass clears 0 (`cv 0.10 < 1/z = 0.142`), but the grid still
  starts at 0: 15% wasted below, and the upper tail (the part we price) is
  squeezed against the grid top with 28% slack dumped *above* it.
- **Case 3 — the smoking gun.** 61% of the grid is empty *below* the mass, and
  the right tail we actually care about is **clipped** (3.9e-7 mass in the top
  bucket, far above the 1e-12 noise floor — a real loss). The windowed method
  *did* compute a finer two-sided grid (`bs 5`, `x_min ≈ 407k`) but **rejected
  it**: the lognormal-cv-2 severity's own 1e-12 quantile (~337k) exceeds the
  windowed extent, so the benign-wrap technique can't apply (see §5).
- **Case 4** — the one case windowing rescues today; even here the 31% slack is
  all dumped *above* the mass, so the log-density plot sits left-of-centre.

Root causes, precisely:

1. **The non-signed `moment` method is one-sided** (`_bs_window`,
   `distributions.py` ~6548): `x_lo = 0` always; only `x_hi` is estimated. It
   never asks where the *left* edge of the mass is.
2. **The `windowed` two-sided method is gated too narrowly**: it fires only when
   `agg_cv < 1/z` *and* it yields a **strictly finer `bs`** (`rows['windowed']
   ['bs'] < rows[selected]['bs']`). So it rescues only the extreme case 4 and
   abandons the broad middle (cases 1, 3) on the wasteful one-sided grid.
3. **Padding is one-sided.** `_size` puts the origin at `floor(x_lo/bs)·bs` and
   extends *up*; the power-of-2 slack `2**log2·bs − W` always lands **above**
   `x_max`. Nothing balances it around the band.

---

## 2. Design requirements (author, restated precisely)

- **R1 — symmetric estimation.** Estimate the left and right mass edges with the
  *same* method, so the window brackets the mass instead of always floor-ing at
  0. (The distribution's own skew still makes the two tails different *lengths*;
  "symmetric" means the *estimator* is symmetric, not the window.)
- **R2 — balanced padding.** Once `[x_min, x_max]` and a natural (binary-
  fraction) `bs` are chosen, the slack `2**log2·bs − W` is split on **both
  sides** of the band, not all above.
- **R3 — keep the obvious exact bounds.** Where the support max/min is known
  exactly — `dfreq × dsev` lattice, bounded severity — use it (already done:
  `_exact_discrete_window`, `_bounded_severity_window`); the new placement logic
  layers on top.
- **R4 — slight convention skew.** Bias the window/padding *gently* toward the
  tail that matters: loss → right, payoff → left. "Slightly, but not too much."
- **R5 — the log-density plot should look balanced** — mass centred, not jammed
  against an edge.

A non-requirement worth stating: the window need not be symmetric in *width*. A
right-skewed loss aggregate properly has a short left edge and a long right
edge; R1 is about using one estimator for both, R4 about a small deliberate lean
on top.

---

## 3. The unifying idea

Replace the special-case `windowed` override with **one two-sided sizer used for
every aggregate**, then clamp and place:

```
x_lo_raw, x_hi_raw  =  two_sided_window(m, sd, skew, p_lo, p_hi)   # R1
x_min  =  max(physical_floor, x_lo_raw)        # loss floor = 0 (or detected support min)
x_max  =  x_hi_raw
W      =  x_max - x_min
bs     =  natural bs (lattice, pinned, or round_bucket(W / 2**log2))
slack  =  2**log2 · bs - W
origin =  snap_to_bs(x_min - f · slack)        # R2 + R4: balance, gently skewed
```

The crucial observation that bounds the blast radius: **for an ordinary
spread-out loss book the two-sided estimator already returns `x_lo_raw ≤ 0`**
(the aggregate has real density near 0), so `x_min` clamps to 0 and `slack` has
no room below — **identical to today**. The new behaviour engages *only* when
the mass genuinely clears 0 (`x_lo_raw > 0`, i.e. `cv < 1/z`) — exactly cases 1,
3, 4. So "symmetric for everyone" and "byte-stable for ordinary books" are the
same policy, not a tension. The change targets precisely the books that are
broken today.

This also **subsumes** the current `signed`, `windowed`, and `pnl`-affine paths:
they are all "two-sided window, place in grid, relabel by the origin roll." The
signed combine already proves the FFT mechanics (`_fft_aggregate`'s modular
`np.roll`); this plan makes the *sizing* uniform so the rest follows.

---

## 4. Methods menu (for round 2)

### 4a. The two-sided estimator (R1)

`estimate_agg_window(m, sd, skew, p)` already returns both edges (shifted-
lognormal/gamma fits, normal fallback, reflect for negative skew). Candidate
unifier. Open points:

- **Per-edge coverage `(p_lo, p_hi)` [Q1].** Today both edges share one `p`
  (`1 - 1e-WINDOW_NINES`). R4 wants the *protected* tail deeper than the
  *unprotected* one: loss → `p_hi` deep (e.g. `1 - 1e-13`, avoid clipping),
  `p_lo` shallow (e.g. `1 - 1e-6`, the left tail is cheap to give up). This is
  the cleanest place to implement the convention skew — it changes *coverage*,
  which is what actually prevents the case-3 clip.
- **Conservative vs fitted edge [Q2].** `_estimate_agg_percentile` takes the
  **max** of {sln, sgamma, normal+3sd} — a safe *upper* bound (anti-clip);
  `estimate_agg_window` takes min-low/max-high of the fits. Proposal: on the
  *protected* edge take the conservative max (anti-clip); on the unprotected
  edge take the tighter fit (anti-waste). This folds the two existing estimators
  into one convention-aware two-sided function.
- **The lower clamp / from-0 tradeoff [Q3] — the big one.** Trimming `x_min`
  above 0 *gives up the `[0, x_min)` axis*: low-attachment aggregate-layer
  evaluations (`E[(A−d)+]` at small `d`), the from-0 severity overlay, the
  from-0 plot. Today's a51 design accepts this for the extreme windowed case and
  documents the `x_min=0` escape hatch. Generalizing to all `cv < 1/z` books
  widens it. Options:
  - (a) accept it (mass is what matters; escape hatch exists);
  - (b) a convention-aware *asymmetric clamp*: for **loss**, keep `x_min` closer
    to 0 (cheap insurance for layers — we care less about the precise left tail
    anyway), trimming only the deep-empty region (e.g. clamp at
    `min(x_lo_raw, m − k·sd)` with a generous `k`); for **payoff**, mirror on the
    right. This makes R4's "skew" *protective* (which axis you keep), not merely
    cosmetic — arguably the most principled reading of the requirement.
  - (c) keep `x_min` at 0 whenever a low aggregate-layer structure is detected.

### 4b. Padding placement (R2, R4, R5)

Given `[x_min, x_max]`, `bs`, `slack = 2**log2·bs − W`: place
`origin = snap_to_bs(x_min − f·slack)`, `f ∈ [0,1]`.

- `f = 0` → today (all above). `f = 0.5` → centred (R5). Convention skew (R4):
  loss → `f < 0.5` (more room above/right, where the tail lives); payoff →
  `f > 0.5`. **[Q4]** the magnitude — propose a small default, e.g.
  `f = 0.5 ∓ 0.1`, behind a config knob.
- **Floor interaction:** never let `origin < physical_floor` (no negative-loss
  padding). When `x_min` is clamped at 0, `f` is inert (no room below) — so
  balancing only acts on books that already cleared 0, consistent with §3.
- **Wrap safety bonus:** centring the band leaves margin on *both* sides of the
  period, so the benign FFT wrap is *safer* than today (band currently sits at
  the bottom with margin only above). Confirm in round 2.
- **Purely-cosmetic-but-not-numerically-inert note:** balancing moves the grid
  *index*, so any baseline keyed on raw index moves, even though the represented
  law on physical support is identical. numerics-2/3 made `exa/lev/kappa`
  origin-carrying, so the *math* is invariant; the regression compare must be on
  physical support, not raw index (see §6).

### 4c. The exact/bounded rows (R3)

`_exact_discrete_window` and `_bounded_severity_window` give hard `[A_lo, A_hi]`.
These bypass the MoM estimator (use the true support) but **still flow through
the new padding placement** (§4b) — place the exact support in the grid with
balanced, gently-skewed slack, lattice-snapped. No change to the bounds
themselves.

---

## 5. The heavy-severity limitation (must be explicit)

Case 3 exposes a hard boundary the symmetric window **cannot** cross. Moving
`x_min > 0` for a non-signed book *requires* the benign-wrap compute: the
severity is laid into the period-`P` FFT buffer on a **0-based** grid `[0, N·bs]`
and the finished aggregate is relabeled up by `round(x_min/bs)`. That is valid
only if a **single severity fits the window** (`_severity_high_estimate <
extent`); otherwise the *severity* discretization itself aliases. A heavy-tailed
severity (lognormal cv 2: 1e-12 quantile ~337k) can exceed the windowed extent,
so the guard (correctly) refuses — and the book is stuck on the 0-based grid,
wasting 61% below and clipping above.

So there are **two regimes**, and round 2 must treat them separately:

- **Regime A — benign-wrap-feasible** (severity fits: concentrated light-/
  bounded-/discrete-severity books, high frequency). The §3 symmetric window +
  balanced padding **fully applies**. Cases 1 (borderline) and 4.
- **Regime B — heavy-severity, single-big-jump-driven** (case 3). Origin cannot
  shift. The symmetric window does **not** help; the remedy is *coverage/clip
  management on the 0-based grid* — extend `x_hi` coverage (deeper `p_hi`, or
  grow `log2`) so the priced right tail is not clipped, and accept the empty
  lower region as the price of the 0-containing severity invariant. **[Q5]** how
  to detect Regime B (the severity-fit guard already does) and what its policy
  is (clip-avoidance via coverage vs. coarser `bs`; whether to warn). A genuine
  severity-windowing fix (windowing the severity grid too) is a far larger change
  and **out of scope** — flag only.

This split is the main reason the work needs rounds: R1–R5 are clean for Regime
A; Regime B is a different lever (coverage, not placement).

---

## 6. Blast radius & regression strategy

- **Ordinary books (`cv ≳ 0.142`)** — `x_lo_raw ≤ 0`, clamp to 0, no slack below
  → **byte-identical** to today. The numerics-2/3 1e-14 regression gate holds
  unchanged for the legacy bulk. Confirm the cv partition is clean.
- **Concentrated Regime-A books (`cv < 0.142`, severity fits)** — move onto the
  symmetric window. Baselines move; each move is an *improvement* (less waste, no
  clip, centred plot). Re-capture, and **assert the move is an improvement**
  (mass conserved, no top/bottom-bucket clipping above noise, moments match
  analytic).
- **Regime-B books** — `bs`/coverage may change (anti-clip), origin stays 0.
  Smaller, targeted baseline moves.
- **Compare on physical support, not raw index** (§4b): the represented law at
  each physical `x` is invariant to padding placement; a naive index-keyed diff
  will report spurious churn. The freeze harness (`scripts/freeze_knowledge.py`,
  `scripts/bucket_baseline.py`) compare may need a physical-support alignment
  step — **[Q6]**.
- **Phasing option [Q7]:** land §4b (balanced padding for already-windowed
  books) first — tiny blast radius — then §4a (general two-sided trimming).
  Or gate the generalization behind the existing "strictly finer bs" until
  validated, then relax. Round 2 picks.

---

## 7. Integration with numerics-4 (the point of doing this now)

`plan-numerics-4-windowed-combine` needs exactly the primitive this plan builds:

- **Part A (windowed portfolio combine).** `best_window` sizes the shared grid
  from per-unit window *widths* and origins: `W_tot = Σ W_k`,
  `x_min_tot = Σ x_min_k` (both add under convolution). Once *every* unit reports
  a two-sided `[x_min_k, x_max_k]` (not `[0, x_hi_k]`), the signed/windowed
  combine path becomes the **general** path — a real simplification, not a new
  branch. This plan is what makes per-unit windows uniform. **[Q8]** confirm the
  origin-additivity and that balanced per-unit padding doesn't double-count slack
  in the sum (likely: combine on *windows*, pad *once* at the total).
- **Part B (bivariate per-axis windowing).** The 2-D severity grid is
  unaffordable; each axis must window to its mass band. A clean symmetric 1-D
  window sizer **is** the per-axis primitive (`multivariate.size_axis`). This
  plan is the 1-D enabler, exactly as a51 was for the original windowing.
- **Convention axis.** numerics-3 already carries `_is_loss_value` and the
  `view × value_type` pricing axis. R4's convention skew should branch on the
  **same** `_is_loss_value` flag (never the label string — house rule), so
  sizing and pricing agree on orientation. **[Q9]** does the window convention
  come from the aggregate's `value_type`, or is it an independent `update`
  argument? (Default: derive from `value_type`; allow override.)

Sequencing proposal: **plan-bucket-window-2 lands before numerics-4's Part A**,
which then consumes its uniform per-unit windows. It may even be folded in as
numerics-4 Part 0. Round 2 decides standalone-vs-folded.

---

## 8. Open questions (round-2 agenda)

- **[Q1]** Per-edge coverage `(p_lo, p_hi)` as the convention-skew lever — values
  and config surface (extend `[discretization]` with a skew/asymmetry knob?).
- **[Q2]** Conservative (anti-clip) vs fitted (anti-waste) edge per side; fold
  `_estimate_agg_percentile` + `estimate_agg_window` into one estimator?
- **[Q3]** The lower-clamp / from-0-axis tradeoff — accept, asymmetric-clamp
  (protective skew), or layer-aware. **This most shapes the design.**
- **[Q4]** Padding fraction `f` default and skew magnitude ("not too much").
- **[Q5]** Regime-B (heavy severity) detection + policy (coverage/clip, not
  placement); warn or silent.
- **[Q6]** Regression compare on physical support; harness changes.
- **[Q7]** Phasing: padding-first vs trim-first vs gated-then-relaxed.
- **[Q8]** numerics-4 combine: origin additivity + pad-once-at-total.
- **[Q9]** Window convention source: `value_type` vs explicit `update` arg.
- **[Q10]** Does `bucket_sizing_p` / `WINDOW_NINES` stay one global, or split into
  per-edge / per-convention settings? (Touches config + `show_settings`.)

## 9. Out of scope

- **Severity-grid windowing** (the Regime-B "real" fix) — the severity must
  contain physical 0; moving *its* origin is a far larger change. Flag only.
- The bivariate solver itself (numerics-4 Part B owns it; this is its 1-D
  enabler).
- Pricing/allocation semantics on windowed/subset grids (numerics-2/3 own the
  origin-carrying math; already landed).
- Re-deriving `_fft_aggregate`'s wrap mechanics — they exist and are proven.

## 10. Files likely touched (when it reaches execution)

- `src/aggregate/distributions.py` — `estimate_agg_window` (per-edge coverage,
  conservative-edge option), `_bs_window` (`_size`/`_row` padding placement, drop
  the narrow `windowed` gate for the general two-sided sizer, convention skew),
  `_severity_high_estimate` (Regime-B detection already here).
- `src/aggregate/config.py` — skew/asymmetry knob(s) [Q1/Q4/Q10].
- `src/aggregate/portfolio.py` — `best_window` consumes uniform per-unit windows
  (numerics-4 boundary).
- `tests/test_bucket_sizing.py` — symmetric-placement, balanced-padding,
  convention-skew, Regime-B clip-avoidance, byte-stable-for-ordinary asserts.

---

*Round 1 ends here. The single most design-shaping decision is **[Q3]** (the
lower-clamp / from-0 tradeoff), closely followed by **[Q5]** (what to do about
heavy-severity Regime-B books, which the symmetric window provably cannot fix).
Recommend the next round opens on those two.*

---

# Round 2 — finalized for Step 1 (1A Aggregate, 1P Portfolio)

> **Decisions locked with the author (2026-06-16).** This round narrows scope to
> **Step 1: univariate symmetric windowing with signs**, in two parts — **1A**
> (Aggregate) and **1P** (Portfolio combine). Porting 1A to `multivariate.py`
> (Part B) and the rest of net-ceded are *later* sessions. Portfolios are not
> relevant to `multivariate` (it works on Aggregate margins / the netceded
> joint), so 1P is its own deliverable, not part of the multivariate port.

## Decisions (resolves the round-1 agenda)

- **[Q3] — "looks right" placement, bounds-first, resolution-preserving.** The
  governing criterion is that the window *looks right against the log-density /
  pmf plot*. Concretely, a **priority ladder**:
  1. **Honor hard/natural bounds** when present and they fit: physical floor
     (`0` for loss), upper bound (`premium = agg_shift` for a P&L), exact lattice
     (`_exact_discrete_window`), bounded severity (`_bounded_severity_window`).
     A both-bounded distribution takes *both* bounds **iff there is space**.
  2. **Hard constraint: never sacrifice severity `bs` resolution** to snap to a
     bound. If a bound only fits by coarsening `bs` below the severity lattice /
     chosen resolution, keep `bs` and either grow `log2` (within `grow_cap`) or
     decline the snap. "Do not lose `bs` resolution on the sev."
  3. Where no hard bound applies, place the **two-sided MoM window**
     (`estimate_agg_window`) clamped at the physical floor, then balance padding.
  Supersedes round-1's (a)/(b)/(c): it is "protective" (R4) *and* bound-aware,
  but the deciding test is visual balance + resolution, not a fixed `k·sd` clamp.
- **[Q5] — heavy severity is non-windowable: don't try, maximize space, honor
  natural bounds.** When the single severity does not fit the window
  (`_severity_high_estimate ≥ extent` — the existing guard), do **not** move the
  origin. Instead: keep the floor-anchored grid, **maximize coverage to kill the
  clip** (deepen `p_hi` / grow `log2`), and still **respect a natural upper
  bound** when one exists (a `premium − Pareto` P&L is bounded above at premium;
  honoring it looks right, ignoring it looks wrong). One `logger.info`, **no
  warning** (expected, not defective).
- **[Q9] — convention from `value_type`.** Window skew branches on
  `self._is_loss_value` (never the label string — house rule). New optional
  `update(window_convention=...)` override; default derives from `value_type`.
- **[Q1/Q4] — skew levers.** Implement R4 as (i) **per-edge coverage**
  `(p_lo, p_hi)` — the lever that actually prevents the case-3 clip: protected
  edge deep (anti-clip), unprotected edge shallow (anti-waste); plus (ii) a small
  **padding fraction** `f = 0.5 ∓ Δ` (loss → more room right; payoff → left),
  both behind config knobs (`[discretization]`).
- **[Q7] — phasing.** Unified two-sided placed window is the single sizer, but
  validated **gated-then-relaxed**: prove byte-stability for ordinary books via
  the freeze/regression harness before removing the legacy 0-based path.

## 1A — Aggregate symmetric windowing (`_bs_window`, `distributions.py`)

The one-sided `moment` row (`x_lo = 0` always, `:6549`) and the narrowly-gated
`windowed` row (`:6605`, `:6642`) **collapse into one two-sided placed window**:

1. **Convention** ← `self._is_loss_value` (+ `window_convention` override).
2. **Edges** `[x_lo_raw, x_hi_raw]`: natural/exact bounds first (ladder step 1),
   else `estimate_agg_window(m, sd, skew, p_lo, p_hi)` with per-edge coverage.
3. **Clamp to physical bound:** loss → `x_min = max(0, x_lo_raw)`; payoff →
   `x_max = min(premium, x_hi_raw)`.
4. **`bs` (resolution-preserving):** pinned `bs_in`, else severity lattice, else
   `round_bucket(W / 2**log2)`. Never coarsen below the lattice to honor a bound
   (ladder step 2).
5. **Balanced padding:** `slack = 2**log2·bs − W`;
   `origin = snap_to_bs(x_min − f·slack)`, `f = 0.5 ∓ Δ`, clamped so
   `origin ≥ floor`. When `x_min` clamps to the floor (ordinary book) there is no
   room below → `f` inert → **byte-identical to today**.
6. **Regime-B branch (Q5):** if origin would clear the floor but the severity
   doesn't fit (`_severity_high_estimate ≥ extent`) → floor-anchored grid,
   anti-clip coverage, honor natural upper bound, one `logger.info`. Generalizes
   today's "reject windowed row" into an explicit branch.
7. **Selection:** `exact_discrete` > `bounded_small` (if tighter) > the unified
   two-sided window; the separate `windowed` override is gone.

**Blast radius / regression:** ordinary loss books (`cv ≳ 0.142`) →
`x_lo_raw ≤ 0` → clamp to 0 → `f` inert → identical grid. Keep the numerics-2/3
1e-14 gate; **assert the `cv` partition is clean** (every byte-stable book has
`x_lo_raw ≤ 0`). Concentrated Regime-A books move onto the symmetric window —
re-capture baselines and **assert the move is an improvement** (mass conserved,
no top/bottom-bucket clip above the noise floor, moments match analytic).

## 1P — Portfolio windowed combine (`best_window` / `update`, `portfolio.py`)

Once every unit reports a two-sided `[x_min_k, x_max_k]` (from 1A), the shared
grid is the **general** path, not a special branch:

- `x_min_tot = Σ x_min_k` (origins add under convolution), floored; `W_tot =
  Σ W_k`; `bs` from summed **widths**; **pad once at the total** (no per-unit
  slack double-count — Q8).
- Generalize `best_window`'s hard-coded `x_min = 0` non-signed branch; route the
  windowed non-signed book through the existing signed / roll-combine path.
- Drive each unit with its **explicit phase-1 origin** (not `x_min='auto'`) so a
  pinned shared `bs` doesn't fail to re-fire the gate.
- **Mixed books:** window the total when it clears 0; off-window unit *views*
  read empty/native via `unit_density` (no wrapped garbage in any core column).
- **Convention:** portfolio `value_type` (a54) drives the total's skew, as 1A.

**Cross-checks:** `Σ_i kappa_i(x) == x` on the windowed combine (numerics-2
anchor, now on the windowed grid); marginals/moments match the per-unit windowed
aggregates; legacy non-windowed books byte-stable.

**Reporting parity (consumes 1A-bucket).** When 1P lands, `Portfolio` gains the
same surfaces 1A-bucket builds for `Aggregate`: a portfolio-level `tail_df` +
`tail_description`/`tail_explanation` (combining the unit tail reports), a curated
public `bs_window_df` (the combine's per-unit → shared-grid journey), and
`bs_description`/`bs_explanation` for the shared grid. Build these on the same
helpers, not parallel implementations.

## Config / files (Step 1)

- `src/aggregate/distributions.py` — `_bs_window` (unified two-sided sizer,
  bounds ladder, balanced padding, Regime-B branch), `estimate_agg_window`
  (per-edge `p_lo, p_hi`), `update` (`window_convention` arg).
- `src/aggregate/config.py` / `data/config.default.toml` — `[discretization]`
  per-edge coverage + padding-`f` skew knobs.
- `src/aggregate/portfolio.py` — `best_window` / `update` windowed routing (1P).
- `tests/test_bucket_sizing.py` — symmetric placement, balanced padding,
  convention skew, Regime-B anti-clip, bounds-snap, byte-stable-for-ordinary,
  `Σ kappa_i == x` windowed combine.

## Out of scope for Step 1 (later sessions)

- Bivariate per-axis windowing (numerics-4 Part B) and `multivariate` copula /
  netceded sizing reconciliation (`plan-multivariate-punchup.md`).
- Occ-reins × windowing (numerics-4 P0).
- The from-0 severity overlay on a windowed grid (W3).

---

# 1A-fix — single-big-jump extent floor (heavy / signed severities)

> **Locked with the author (2026-06-17). Lands before 1P** (a windowed
> portfolio combine inherits per-unit extents, so the per-unit extent must be
> right first). Same method as 1A (`_bs_window`); reuses `_severity_high_estimate`.

## The problem (two faces of one root cause)

The 3-moment MoM window (`estimate_agg_window` / `_estimate_agg_percentile`)
**underestimates a tail the first three moments do not capture.** Two cases
found by testing a58:

- **T5 — heavy sev, high freq** (`5000 claims lognorm 100 cv 2`): MoM upper edge
  634k, true 1e-12 reach **797k** → ~3.9e-7 of the priced right tail clips off
  the grid top. Mild (this was deferred as W7).
- **LNSFixed — signed sev, moments lie** (`10 claims 100 - lognorm 10 cv 2.5`):
  the severity is bounded above at 100 with a long left tail to ~−74k, but the
  aggregate's skew is **+0.21** (positive — `E[X³]` is dominated by the `+100³`
  term) so the MoM window is `[−749, 3557]`. The severity's −74k reach is ~9×
  the whole grid width → the FFT **aliases** (not just clips) → **47% of mass
  lost**, garbage law. Catastrophic. The exact kurtosis is **737** vs the
  3-moment fit's **3.08** — a 240× tell that the moments are lying.

Root cause is shared: **moment-based sizing is blind to a heavy tail.** Mild on
a positive sev (skew still flags heaviness; T5), catastrophic on a signed/
reflected sev (skew can be positive while the tail is heavy-left; LNS).

## The principle — single big jump (subexponential EVT)

For a subexponential severity the aggregate's far tail is one big claim on a
typical bulk: `P(S > x) ≈ E[N]·P(X > x)`. So:

- to cover the aggregate to `p*`, probe the **severity** at
  `p** = 1 − (1 − p*)/E[N]` (one claim must reach there; the other `N−1` are
  typical) — **not** `N × q_X` (that assumes *every* claim is huge: wildly
  over-sizes);
- the single-big-jump extent is **`ES − μ_X + q_X(p**)`** (replace one typical
  claim by one big one), mirrored on the low side for a signed sev.

## The rule — unconditional extent floor on the selected window

Resolution (`bs`) stays sized from the **bulk / window** (today's logic). Only
the **extent** is floored by the single big jump, per side:

```
sbj_hi = ES − μ_X + q_X_hi(p**)          # one big claim up
sbj_lo = ES − μ_X + q_X_lo(p**)          # one big claim down (signed sev; else 0)
grid must cover [ min(window_lo, sbj_lo), max(window_hi, sbj_hi) ]
```

- **Extent floor on the *selected* window's edges**, NOT a standalone `[0, sbj]`
  sizing. Critical: a *windowed* book keeps its narrow band + fine `bs`; SBJ only
  nudges its band edges if one big claim pokes past them (usually it does not —
  for a concentrated high-freq light-sev book `q_X` is small).
- **Unconditional / self-activating.** SBJ binds only when the tail is heavier
  than the window already covers. Light / thin / bounded / concentrated books
  have `sbj ≤ window` → **byte-stable**. Genuinely heavy non-windowed books
  coarsen `bs` (measured: T5 `bs 10→20`, the 5-claim heavy `ORD 5→10`) — the
  *safe* direction (tail capture), and within the author's tolerance (bs=25 was
  acceptable for T5).
- **Signed anti-alias.** For a signed sev additionally enforce grid **width**
  `N·bs ≥ |severity reach|` (anchor `sbj_lo` on `min(sbj_lo, q_X_lo)`), so the
  severity cannot wrap the FFT buffer (the LNS failure mode).

## The guard (author's point)

`p**` deepens with `E[N]`; at `E[N]=5000`, `1 − p** = 2e-16` (past double
precision) → `q_X(p**) → ∞` for an unbounded sev. So:

- **Floor `1 − p**` at `1e-14`** (≈ `WINDOW_NINES + 2` nines) — the deepest the
  severity is numerically meaningful; config knob.
- **Cap `q_X(p**)` by the severity limit** when finite (`_severity_high_estimate`
  already does this).
- **Infinite-variance / unlimited Pareto** (no finite `p**` quantile): fall back
  to the existing limit / `recommend_bucket` path (already handled).

## Kurtosis diagnostic — signed severities only

A *positive* severity encodes a heavy right tail in its **skew**, which the MoM
already consumes — it cannot be blindsided. The skew-lies pathology needs the
reflection (bounded-above + heavy-left), i.e. a **signed** sev. So compute the
analytic aggregate kurtosis (compound formula from the 4th severity moment) and,
**for signed severities only**, flag via `explain_validation` when it exceeds the
fitted-window kurtosis by a factor (e.g. ≥ 10×). Diagnostic, not a sizing gate —
SBJ already sizes correctly; this just tells the user the moments were untrustworthy.

## Mechanics (`_bs_window`)

1. `_single_big_jump_window(p_star)` helper → `(sbj_lo, sbj_hi)` from `ES`,
   `μ_X`, `_severity_high_estimate(p**)` (upper) and the signed `sev.ppf` (lower),
   with the `1e-14` guard and limit cap.
2. After the selected window `[x_lo, x_hi]` is chosen (moment / windowed /
   bounded / exact), floor its extent: `x_hi ← max(x_hi, sbj_hi)`,
   `x_lo ← min(x_lo, sbj_lo)`; resize `bs`/`log2` from the floored extent if it
   grew (keep the bulk `bs` when SBJ does not bind).
3. Signed: enforce width ≥ `|severity reach|`.
4. Record an `sbj` row (or columns) in `_bs_window_df` for inspectability.
5. Signed kurtosis diagnostic into validation.

## Blast radius / regression

- Light / thin / bounded / concentrated books: `sbj ≤ window` → **byte-stable**
  (the numerics-2/3 1e-14 gate and `test_suite.agg` snapshot hold).
- Heavy non-windowed books: `bs` coarsens (safe direction); re-snapshot, assert
  mass conserved + no top/bottom clip above noise + moments match analytic.
- Signed heavy-left books (LNS): from 47% mass loss → mass ~1; assert no alias.

## Files

- `src/aggregate/distributions.py` — `_single_big_jump_window` helper,
  `_bs_window` extent floor + signed width guard, analytic agg kurtosis + signed
  `explain_validation` diagnostic.
- `src/aggregate/config.py` / `data/config.default.toml` — `[discretization]`
  SBJ probe-depth floor (nines) + kurtosis-flag factor.
- `tests/test_bucket_sizing.py` — T5 no-clip, LNS mass-recovered/no-alias,
  byte-stable-for-light, signed kurtosis warning, big-`E[N]` guard.

## Out of scope (still)

- A genuine 4-moment severity fit (NIG / GH). SBJ suffices; a 4-parameter fit is
  a larger modeling change.
- Multivariate per-axis SBJ (numerics-4 consumes this 1-D primitive).

Version bump a58 → a59 on landing.

## As built (landed a59) — three corrections to this draft

Implementation surfaced three things the draft above got wrong; the landed code
follows these, and they supersede the draft where they conflict.

1. **Signed vs positive are different *urgencies*, not one symmetric floor.**
   - *Signed* (the `100 - lognorm` case) is a **correctness** bug: the severity
     discretises on the same N-bucket grid, so if its negative reach exceeds the
     grid width the FFT **wraps** and corrupts the whole law (47% mass loss).
     The grid therefore **always** covers `[sbj_lo, sbj_hi]` — keeping the bulk
     `bs` when it fits the log2 budget, else **coarsening `bs`** within the cap.
     Aliasing is fixed at any log2.
   - *Positive* (the `5000 cv 2` case) is only a **refinement**: a heavy MoM
     window under-reaches, clipping a tiny far tail. The window extends to the
     jump **only when it fits at the bulk `bs` within the requested `log2`**;
     otherwise the MoM window is kept (clipping beats wrecking the bulk). A
     material clip is still flagged by the existing `DefectiveDistribution`.

2. **`log2` is honored — no silent growth.** The draft's "grow `log2` to keep
   the bulk `bs`" violates the caller/hint contract (`test_hints` pins an
   explicit/hinted `log2`) *and* the author's no-magic preference. The floor
   never grows `log2` past the request (explicit, hinted, or the default 16) and
   never coarsens a pinned `bs`. A genuinely heavy book that needs more grid is
   the user's call to raise `log2` (the validation/deficit warning tells them).

3. **The kurtosis diagnostic is dropped.** The "exact kurtosis ≈ 737 vs fit
   3.08" tell was an artifact: 737 is the *empirical* kurtosis of the
   already-aliased (47%-corrupted) distribution. The **true-law** compound
   kurtosis of the signed case is ~4.8 (excess ~1.8), so a kurtosis-vs-fit test
   never fires. With aliasing fixed at source and a positive-tail clip flagged by
   `DefectiveDistribution`, no separate diagnostic is needed. `_severity_low_estimate`
   and the `sbj` row in `_bs_window_df` were kept; the kurtosis machinery was not
   built.

---

# Round 3 — 1P reconciled with 1A-bucket (2026-06-17)

> **Why this round.** 1A-bucket (`dev/done/plan-univariate-bucket.md`, a62–a65)
> turned `_bs_window` into a full *tail-aware* sizer (layered tail report,
> single-big-jump extent floor, concentration gating, honest truncation,
> clip → `DefectiveDistributionWarning`, the public `bs_window_df` /
> `bs_description` / `bs_explanation`). Re-reviewing §1P against the landed code:
> the **skeleton holds** — general path, additive origins, pad-once, convention
> skew, reporting parity — and more of it is pre-built than the round-2 text
> implies. But the **span combine** and the **clip/reporting machinery** need
> updating before 1P executes. This round records the deltas; it writes no code.

## What the combine actually does today (lineage)

- **`best_bucket`** (legacy, *DELETE BEFORE BETA*, `:1818`): root-sum-square of
  per-unit *recommended buckets*, `bs = (Σ bs_k²)^0.5`. Combines **bucket sizes**,
  and scales the wrong way — `k` identical units give `round_bucket(b·√k)`, so
  *adding units coarsens the grid*.
- **`best_window`** (a49, live, `:1852`): `resolution = min_k bs_k`;
  `span = (Σ W_k)/N`; `bs = round_bucket(max(resolution, span))`; `x_min = 0`
  non-signed (Plan A) / `Σ`-floored signed. So the live code does a **linear sum
  of window *widths*** for span. (The old RMS was on `bs`; the current code adds
  `W`.)

## Why `Σ W_k` overstates (the author's intuition, made precise)

Two independent "sum vs proper combine" errors:

1. **Diversification.** The bulk band is a `√variance` quantity, so widths
   combine by **root-sum-square** `√(Σσ_k²)`, *not* `Σσ_k`. For `k` iid units
   `Σσ_k = √k · √(Σσ_k²)` — the linear sum overstates the bulk by `√k`. This is
   the *same* RMS-vs-linear error `best_window` was built to fix on `bs`; it
   merely moved from `bs` onto `W` (over-correcting from one extreme to the
   other on a different quantity).
2. **Single big jump.** The portfolio far tail is *one* big claim riding the
   combined bulk: `P(S_tot > x) ≈ Σ_k E[N_k]·P(X_k > x)`. The extent floor is
   therefore **max-type** (the heaviest unit's jump on the *total* bulk mean),
   not `Σ` of per-unit floors. Now that each `W_k` already embeds its own a59
   SBJ extent, `Σ W_k` **double-counts every unit's tail allowance**.

Concretely, for non-signed loss units (`x_min_k = 0`, `W_k ≈ μ_k + z·σ_k +
sbj_k`): `Σ W_k = Σμ_k + z·Σσ_k + Σ sbj_k`, whereas the true total support is
`≈ Σμ_k + z·√(Σσ_k²) + (one sbj)`. The overstatement is exactly the two gaps
`z·(Σσ_k − √Σσ_k²)` and `(Σ sbj_k − one sbj)`.

**The caveat that keeps `Σ W_k` honest:** it is a *correct upper bound* on the
true support width — support adds (`(max−min)` of a sum `= Σ(max−min)`), exact
for bounded books — hence a **guaranteed no-wrap span**. That is *why* it was
safe, and why the FFT doubling-padding currently absorbs its looseness. So the
change is **tighten-with-validation**, not fix-a-bug: we trade guaranteed-safe-
but-coarse for tight-but-must-be-proven-no-wrap.

## The reconciled combine (target)

Mirror 1A's **bulk / extent split** at the portfolio level. The bulk is sized by
**Portfolio MM**, *not* by combining per-unit windows — neither by sum nor by
RMS (see the next subsection for why both are wrong).

- **`bs` / bulk from Portfolio MM.** Feed the exact total moments — already on the
  object as `agg_m`, `agg_sd = agg_m·agg_cv`, `agg_skew` (`portfolio.py:356-358`,
  the analytic compound moments of the sum; cumulants add under independence) —
  straight into the *same* `estimate_agg_window(m, sd, skew, p)` path 1A uses.
  **There is no width-combine step.** Per-unit widths enter *only* as the
  resolution floor `min_k bs_k` (a unit's own lattice must survive — the "don't
  lose sev `bs`" hard constraint, now portfolio-wide), never the span.
- **One portfolio-level SBJ extent floor — the look-through.** The subexponential
  tail of a sum is the *sum of the tails*, dominated by the heaviest unit:
  `P(S_tot > x) ≈ Σ_k E[N_k]·P(X_k > x)`. The single-big-jump scenario is one big
  claim in unit `k` on the *typical* bulk of everything else, which works out to

  ```
  sbj_hi_port = agg_m + max_k ( sbj_hi_k − ES_k )      # MAX, not Σ
  ```

  where each `sbj_hi_k = a._single_big_jump_window(p_star)[1]` is the unit's own
  SBJ called with the **portfolio** `p_star` — each unit forms its own
  `p**_k = 1 − (1 − p_star)/E[N_k]` from *its* frequency, and `sbj_hi_k − ES_k =
  q_{X_k}(p**_k) − μ_{X_k}` is that unit's "jump excess" over a typical claim. So
  the look-through *reuses* `_single_big_jump_window` per unit and takes the max
  excess on top of the combined bulk mean — never `Σ sbj_k`. Lower edge mirrors
  for signed. Self-activating, inert-when-not-binding exactly as in 1A: a thin /
  bounded / well-diversified total has `sbj_hi_port ≤ MM bulk window`, so the
  floor does not move the grid.
  *Refinement:* `max_k` is tight when one unit dominates; when the `tail_class`
  driver list names ≥ 2 comparably-heavy units, escalate to the pooled root-find
  `Σ_k E[N_k](1 − F_{X_k}(x)) = 1 − p_star`.
- **Origin — windowing is enabled for non-signed totals, not excluded.** The live
  `best_window` hardcodes `x_min = 0` (Plan A). 1P **generalizes** this: a
  non-signed total whose mass clears 0 (the high-frequency / tiny-cv case —
  `Poisson(100000)` and friends) gets a **windowed origin** (Plan B), routed
  through the existing signed roll-combine path, exactly as 1A windows the single
  aggregate. Origins add for signed; pad once at the total.
- **`bs` discipline — carry raw, round once.** `round_bucket` snaps *up* (≤ 2×).
  Rounding per-unit in phase 1 *and* again at the combine compounds to **~4×**
  worst-case coarsening. So: carry the **raw** (un-rounded) per-unit needs up from
  phase 1; `round_bucket` a **single time** at the top. The wrinkle is the
  resolution floor must stay lattice-aware (a discrete unit's `bs` must divide its
  integer lattice), so the rule is: round the **span** term once at the top, keep
  the resolution floor as the finest per-unit *lattice* value, and snap the final
  `bs` to a multiple of that floor so the two stay commensurate. Caps worst-case
  coarsening at ~2×, and since the algo's **only** failure direction is fidelity
  loss (it coarsens rather than truncates — the SBJ floor + honored `log2` protect
  the tail, clipping is always flagged), this directly shrinks the worst case.

### The RMS-of-windows view (kept, inspectable — not the live span)

The windows that propagate up from the units are **whatever each Agg selected**
(exact lattice, bounded-sev, MM, or SBJ-floored) — *not* necessarily MM windows —
so their RMS is a genuinely independent estimate worth carrying. In a strict
normal world with a common tail-to-SD converter `k` (`w_i = k·σ_i`), the window
of the sum is exactly `√(Σ w_i²)` — the `k` cancels (`k·√(Σ(w_i/k)²) = √(Σ w_i²)`),
so RMS-of-windows, *not* the linear `Σ w_i`, is the right normal-approx combine.
It runs **too high but conservative** in practice for one reason: the per-unit
windows bake in each unit's *skew* (deep `k_i` from the sln/sgamma MM fit), while
the sum **de-skews by CLT** and reaches its tail at a shallower `k_sum`. So
`RMS(w_i) ≈ k̄_i·σ_tot ≥ k_sum·σ_tot = ` the MM window.

Therefore: **`m_tot ± RMS(w_i)` is wired in as a standing candidate row in
`bs_window_df`**, alongside the MM window and the SBJ floor — it is *not* selected
as the live span, but the gap **`Port-MM − RMS(w_i)` reads directly as the
skewness/diversification adjustment** the combine is making. If it never fires
(never tighter than MM, never needed as a fallback) we drop it later; until then
it is the cheapest possible sanity check on the MM window, visible to the user.

## Reuse, don't reimplement (single source of truth)

Run the *total* through the same tail-aware machinery rather than a parallel
heuristic:

- `_loss_tail_classes` / `tail_class` on the combined book (`tail_class` is
  already worst-of; feed it combined moments so honest-truncation and
  concentration apply to the total too).
- A portfolio `_bs_clip` record + honest-truncation deficit handling, so a
  clipped *total* warns **once** and `bs_explanation` has something to read.
- **Finish reporting parity:** add Portfolio `bs_explanation` and a `tail_df`
  frame; reconcile the curated `bs_window_df` columns with Aggregate's
  (`clipped` / `log2_need`). `tail_description` / `tail_explanation`,
  `bs_window_df`, `bs_description` already shipped (a65) — the per-unit → shared
  journey rows exist; only these two surfaces and the clip columns are missing.
- **Quiet the phase-1 pre-pass.** `best_window` now runs the full sizer per unit
  (`:1922`), which can emit `DefectiveDistributionWarning` / `logger.info`.
  Dedupe so `update()` warns once — mirror the Aggregate deficit/clip gate.

## `x_min` policy — our algo decides, back-compat is not a ship gate (locked 2026-06-18)

**v1.0 ships with our algo's `x_min`, even where it disagrees with prior
versions.** The live non-signed `x_min = 0` is partly a back-compat choice
(reproduce every pre-1.0 grid); we **drop that as a requirement**. The windowed
origin from 1P is the shipped behavior.

Crucially, this retires only *one* of the two stability claims the earlier rounds
conflated:

- **Back-compat with old published grids — dropped.** No assert that a non-signed
  total reproduces its legacy 0-based grid. `x_min = 0` survives **only as an
  opt-in reconciliation switch** (run a book both ways to explain a difference),
  never as a gate.
- **numerics-2/3 origin-invariance — kept, non-negotiable.** `Σ_i kappa_i(x) ==
  x`, moments, and mass conservation must hold on *whatever* grid our algo picks.
  This is correctness (the origin-carrying `exa`/`lev`/`kappa` math), not
  back-compat — it is unaffected by, and in fact the guarantee that *licenses*,
  moving `x_min`.

So the review asserts flip from "reproduce the legacy grid" to "the new grid is
**correct**," with `x_min = 0` as an optional cross-check, not a pass condition.

## Best process for review (how to validate the reconciled combine)

The combine has been *wrong in two opposite directions* (legacy RMS-on-`bs`
coarsens; current linear-sum-on-`W` overstates), so the review pins the new MM
rule against an ordering and proves no-wrap. The two reference windows are the
ones now carried as inspectable rows in `bs_window_df`, so most of this is read
off the frame, not recomputed in tests:

1. **The ordering assert (the spine).** The three window widths satisfy

   ```
   MM window  ≤  RMS(w_i)  ≤  Σ w_i
   (live)        (conservative)   (linear, way over)
   ```

   `MM ≤ RMS` because the sum de-skews (`k_sum ≤ k̄_i`); `RMS ≤ Σ` by quadrature.
   Assert `MM ≤ RMS(w_i)` on every exemplar — if MM ever exceeds RMS the total
   moments and the per-unit fits disagree and it is a **bug**, not a tuning miss.
   `Port-MM − RMS(w_i)` *is* the skewness/diversification adjustment; eyeball it
   per book.
2. **Correctness assert (replaces the old "reproduce the legacy grid").** On the
   grid *our algo picks* (not the legacy one — see the `x_min` policy above):
   mass conserved, moments match analytic, `Σ_i kappa_i(x) == x` (the numerics-2/3
   origin-invariance, non-negotiable). `x_min = 0` is an **optional** cross-check
   to *explain* a difference, never a pass condition.
3. **Improvement assert (the moved books).** No top/bottom-bucket clip above the
   noise floor; the bulk is centered, not jammed against an edge; marginals/moments
   match the per-unit windowed aggregates; the realized grid is at least as good as
   (and usually finer than) the legacy 0-based one.
4. **Compare on physical support, not raw index** (§4b) — padding placement
   moves the index; the represented law at each physical `x` does not.
5. **No-wrap proof.** Assert the realized span clears the true support (the SBJ
   floor guarantees the tail; `RMS(w_i)` is the conservative bound the doubling-
   padding always covers). A span exceeding `Σ w_i` would be the bug signal.
6. **Exemplar spread — drive from the `agg` database.** Build the review/test
   portfolios from the real example programs in `src/aggregate/agg` (the
   `test_suite.agg` / `test_decl.agg` Portfolio entries), not just synthetic
   one-offs, and append any new 1P cases to `test_decl.agg` under the matching
   section (house rule). Cover heavy-unit, well-diversified-iid, mixed-sign, and
   all-discrete portfolios. The **diversified-iid** book is the headline: legacy
   RMS-on-`bs` coarsens, `Σ W_k` overstates by `√k`, and Portfolio MM should sit
   `√k` tighter — the cleanest demonstration the combine is finally right.

## Files / version (Round 3)

Doc-only amendment — **no version bump, no CHANGELOG** (no behavior change). When
1P *executes*: `portfolio.py` (`best_window` reconciled to Portfolio MM + the SBJ
look-through, the inspectable `RMS(w_i)` row, a `_single_big_jump_window`,
`_bs_clip`, `bs_explanation` / `tail_df`), reusing the `distributions.py` /
`tail.py` helpers rather than re-implementing; asserts in
`tests/test_bucket_sizing.py` per the review process above. Bump on landing.

**Drive-by to fix on the way in:** the comment at `portfolio.py:2081-2082` claims
non-signed books "keep the legacy `best_bucket` path exactly" — **stale and
wrong**; the code at `:2092` calls `best_window`. Correct the comment when this
block is touched.
