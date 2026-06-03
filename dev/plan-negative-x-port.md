# Plan: negative-support at the **Portfolio** level — combine + `density_df` (p_total / F / S)

> **Status:** REFRESHED (rev 1) from the landed Aggregate work
> (`plan-negative-x-agg.md`, implemented at 1.0.0a21). Scope deliberately
> narrowed per author: **this iteration delivers signed-support *combination* and
> the `loss` / `p_total` / `p_{line}` / `F` / `S` columns only.** All pricing /
> allocation machinery (`add_exa`, `exa_*`, `exeqa_*`, `lev`, `epd`, distortion
> pricing, `value_type`) is **split out** into
> [`plan-portfolio-neg-x-pricing.md`](plan-portfolio-neg-x-pricing.md) and is
> **out of scope here**.
>
> This is **TODO #6** (the combine half). Depends on the Aggregate-level
> machinery now in place: per-unit signed grids, the `i0` severity offset, the
> origin-at-0 FFT convention, `Aggregate._bs_window` (and its helpers
> `_exact_discrete_window` / `_bounded_severity_window` / `_severity_lattice`),
> `estimate_agg_window`, and `WINDOW_NINES`.

---

## 1. Scope

**In scope (this iteration):**

1. **Combine independent signed-support unit aggregates** onto a common signed
   portfolio grid (TODO #6).
2. **Correct `loss` (index), `p_total`, every `p_{line}`, `F`, `S`** on signed
   support — and therefore the quantile / VaR / TVaR functions that read them.

**Explicitly deferred** → `plan-portfolio-neg-x-pricing.md`:
`add_exa` and everything it writes (`exa_*`, `exeqa_*`, `lev`, `epd`, the
`loss_max` blanking heuristic, tail `shift(-1)` fills), distortion / `Distortion.price`
behaviour on signed support, and consumption of the `value_type ∈ {loss, payoff}`
member. **In this iteration a signed portfolio routes through the `add_exa=False`
branch** (F/S only); requesting `add_exa=True` on a signed portfolio warns and
falls back (see §4).

---

## 2. The key fact we learned at the Aggregate level

The single most important thing the Aggregate work clarified — and the draft of
this plan did **not** account for:

### 2a. The FFT product is already in the origin-at-0 convention — no per-unit roll

In `Aggregate._fft_aggregate` (`distributions.py:4181`) the signed path lays the
severity into a length-`M = N<<padding` buffer with **physical 0 at index 0** and
the `i0` negative buckets **wrapped to the top of `M`**. The stored
`ftagg_density` is the FFT of *that* buffer. Crucially the output density `a` (the
length-`M` `irfft`) is **always in the origin-at-0 convention regardless of the
unit's `i0` or its own output window** — `i0` only governs how the severity is
laid into the buffer, not where the answer sits.

Therefore, because the portfolio total is a **deterministic sum** of independent
units, the convolution of origin-at-0 densities is itself origin-at-0:

```
ft_all   = Π_k  agg_k.ftagg_density            # product in Fourier space (as today)
a_total  = irfft(ft_all, M)                     # length-M, origin-at-0, negatives wrapped to top
```

No per-unit roll is needed before the product. This is the "easy half": constant
per-unit shifts de-shift cleanly under convolution.

### 2b. ...but a single window roll IS needed, and the current path truncates it away ⚠

The current `update` does `p_total = np.real(ift(ft_all, padding))`. `ift`
truncates the `irfft` to the first `N` samples — which **silently drops the
wrapped negative mass** living at the top of `M`. For non-negative books that
top region is ~0, so today's code is correct. For a signed book it loses the
entire negative tail.

The fix mirrors Aggregate's **F2 step** exactly: do the full-length `irfft`, then
**one** roll to place the chosen origin at output index 0, then keep `N`:

```
j0_tot   = round(x_min_tot / bs)                # negative when x_min_tot < 0
p_total  = np.roll(irfft(ft_all, M), -j0_tot)[:N]   # signed window [x_min_tot, x_min_tot + N*bs)
```

The same applies to every **per-line marginal**. A unit's `agg.agg_density` on a
0-based grid is *already truncated* (`[:N]` of its own length-`M` buffer), so for
a signed portfolio the per-line columns must be rebuilt from the **full** FFT:

```
p_line   = np.roll(irfft(agg_k.ftagg_density, M), -j0_tot)[:N]
```

and the index becomes the signed grid `loss = x_min_tot + bs*arange(N)`.

### 2c. Drive each unit on its **own signed grid**, not a 0-based grid ⚠ (revised from rev 1)

The rev-1 draft said to drive units on a shared 0-based grid and roll only the
total. Tracking down residual-risk #2 shows that is **wrong for the unit
objects**: a signed unit driven with `x_min = 0` has `j0 = 0`, so its
`agg.agg_density` is the *truncated* `[:N]` of the length-`M` buffer with the
**negative tail dropped**. That then poisons the unit object three ways
(`distributions.py`):

- **false deficit warning** — `deficit = 1 - sum(agg_density)` (`:4084`) sees the
  dropped negative mass and fires `DefectiveDistributionWarning` spuriously;
- **corrupted per-unit moments** — `est_m/est_cv/est_skew` come from
  `xsden_to_mwrangler(self.xs, agg_clean)` (`:4054–4058`) on the truncated density;
- **wrong per-unit `density_df` / `describe` / `plot`** built from it.

**The fix — and it's strictly better for instrumentation:** drive each unit on
**its own signed window** `[x_min_k, x_min_k + N·bs)` (via the unit's own
`update`/`_bs_window`, sharing the portfolio's `bs`/`log2`/`padding`). Then each
unit object is fully correct and independently inspectable (`describe`, `plot`,
moments). **Crucially this does not break the combine:** `ftagg_density` is
origin-at-0 **regardless of the unit's `x_min`** — the output roll `-j0` is applied
to the *density* `a`, never to `ftagg` (`:4258` vs `:4262`). So the units' FFTs
still multiply correctly (§2a), and the portfolio still does the single F2 roll
(§2b) for the **total** and rebuilds each **per-line** column from
`ftagg_density` onto the common portfolio grid.

So there is **no 0-based drive grid**. Two grids still exist, but they are:
- per-unit signed grids `[x_min_k, …)` — what the unit objects live on;
- the portfolio signed grid `[x_min_tot, …)` — what `density_df` lives on, reached
  by the single roll. No double-windowing, because the unit's own roll lands its
  *density* on its own grid while the portfolio reads its *un-rolled* `ftagg`.

---

## 3. Portfolio combine on signed support

### 3.0 Hard constraint: byte-identical for non-signed books

The non-signed path must reproduce today's results **exactly**. The clean way to
guarantee that is a single gate: `Portfolio._signed()` (mirroring
`Aggregate._signed()`, = `any(agg._signed() for agg in self.agg_list)`). **When
not signed, every new code path is bypassed** — bs is `best_bucket` exactly as
today, `log2` is the caller's value untouched, the grid is `linspace(0, MAXL, N)`,
`p_total = ift(ft_all)`, and per-line columns stay `= agg.agg_density`. No new
estimate runs. All the machinery below is reached **only** when `_signed()` is
true. (This is why I do *not* fold a window-fit coarsening into the non-signed bs:
it could perturb existing numbers.)

### 3.1 Window/bucket policy — a thin signed-aware wrapper on `best_bucket`

Per author Q4: **a thin wrapper on `best_bucket`, not a replacement.** Add
**`Portfolio._bs_window(log2, bs_in, recommend_p)`** that:

- **non-signed** → returns `(best_bucket(log2, recommend_p), log2, 0.0)` — pure
  pass-through (§3.0).
- **signed**:
  - **x_min_tot, x_max_tot** — author Q1: **start with the sum of unit windows**
    (`x_min_tot = Σ x_min_k`, `x_max_tot = Σ x_max_k`), exact-and-additive for a
    sum, slightly conservative. Each `(x_min_k, x_max_k)` comes from the unit's
    own `Aggregate._bs_window` (already run, or run a sizing pass). Flagged as the
    starting policy that **may need enhancement** (e.g. a portfolio-level
    `estimate_agg_window` when the conservative sum wastes too many buckets).
  - **bs — take the COARSER** (author Q2). The summed support `W_tot = x_max_tot −
    x_min_tot` is wider than any unit's, but `N = 2**log2` is capped, so we need
    `N·bs ≥ W_tot` or the convolution aliases/wraps. Two candidates:
    `bs_best = best_bucket(log2, recommend_p)` (the RMS rule, itself already
    `≥ max_k bs_k`) and the fit floor `bs_fit = W_tot / N`. Take
    `bs = round_bucket(max(bs_best, bs_fit))` — the **coarser**, to buy the space
    and avoid aliasing. (Integer-lattice exactness `bs = 1` is kept **only** if
    the summed lattice support fits: `W_tot ≤ N`; otherwise it coarsens and
    exactness is lost — note it, don't force it.)
  - **log2** stays a **cap** (caller's value or 16); we fill it (do not shrink
    for signed — we generally *want* the buckets to hold the wider support).
  - **moment cross-check (widen-only)** — after the FFT, `update` computes
    `est_m/est_sd/est_skew` (`portfolio.py:1735`). Feed to `estimate_agg_window`;
    if the realized window clips mass beyond `WINDOW_NINES`, emit the same deficit
    warning as the Aggregate path. We don't resize mid-update; the analytic sum is
    primary.

Returns `(bs, log2, x_min_tot)`, mirroring `Aggregate._bs_window`'s contract.

`best_bucket` itself is **kept** (back-compat for callers/docs); `_bs_window` calls
it. (Resolves Q4.)

### 3.1a `Portfolio._bs_window_df` — unit-indexed, with a `used` row (author Q3)

Mirror `Aggregate._bs_window_df`'s **idiom** but swap *method* rows for *unit*
rows. `Aggregate._bs_window_df` is indexed by method (`moment`, `exact_discrete`,
`bounded_small`) + a `used` row, columns `x_min, x_max, W, bs, log2, coverage,
note, selected` (`distributions.py:5527`). The Portfolio version:

- **index** = one row **per unit** (the unit's selected window from its own
  `_bs_window`: `x_min_k, x_max_k, W_k, bs_k, log2_k`, plus which method that unit
  picked in `coverage`/`note`) **+ a final `used` row** = the realized portfolio
  grid (`x_min_tot, x_max_tot = x_min_tot + N·bs, W = N·bs, bs, log2`).
- This matches the Portfolio convention of per-unit rows/columns with a
  total/used summary (cf. `stats_df`/`describe`, which carry per-unit + `total`),
  whereas the Aggregate breaks *itself* down by method. Same spirit, right axis
  for each class.

### 3.1b Sizing order — two phases (resolves the bs/window chicken-and-egg)

`bs` must be coarse enough to hold the **summed** support, but the summed support
isn't known until units are sized — and units can't be sized until `bs` is fixed.
Resolve with a cheap **analytic pre-pass** (no FFT):

1. **Pre-pass (analytic):** for each unit get its natural signed window
   `(x_min_k, x_max_k)` and width `range_k = x_max_k − x_min_k` from the unit's own
   `_bs_window` / `estimate_agg_window` on its theoretical moments (no FFT needed).
2. **Choose the shared grid:** `W_tot = Σ range_k` (the sum's support fits in
   `W_tot`); `bs = round_bucket(max(best_bucket(log2), W_tot / N))` (the coarser,
   §3.1); `log2` = cap.
3. **Drive units** on the shared `bs`/`log2`/`padding` with `x_min='auto'` — with
   the now-coarse `bs` each unit's range easily fits `N·bs`, so no per-unit
   deficit. Read back each unit's realised `x_min_k` (post `bs`-snap).
4. **Combine origin:** `x_min_tot = Σ x_min_k` (realised), `j0_tot =
   round(x_min_tot/bs)`.

### 3.2 `update` changes (`portfolio.py:1616`)

Everything here is inside the `if self._signed():` branch (§3.0); the existing
code is the `else`.

- **Drive units on their OWN signed grids** (§2c), *not* a 0-based grid: each
  `agg.update(log2=log2, bs=bs, padding=padding, x_min='auto', …)` so the unit
  object is internally correct (no false deficit, right moments, right
  `describe`/`plot`). Routing through the unit's `update` (→ its `_bs_window`)
  keeps the no-back-doors property. The portfolio **present grid**
  `xs = x_min_tot + bs*arange(N)` (signed) becomes `density_df.index`/`loss`.
- **FFT product** loop (`:1692–1702`) keeps multiplying the units' full-`M`
  `ftagg_density` (origin-at-0 regardless of each unit's own `x_min`, §2a/§2c).
  Replace the final `p_total = ift(ft_all)` with the **F2 present step** (§2b):
  `p_total = np.roll(irfft(ft_all, M), -j0_tot)[:N]`, `M = N<<padding`.
- **Per-line columns:** do **not** use `agg.agg_density` (it's on the unit's own
  grid, and is the truncated `[:N]`). Rebuild each on the portfolio grid from the
  un-rolled FFT: `p_{line} = np.roll(irfft(agg_k.ftagg_density, M), -j0_tot)[:N]`.
  (No per-unit normalization needed: each unit's full-`M` density is a proper pmf
  summing to 1, so the convolution and each marginal sum to 1.)
- `F = cumsum(p_total)`, `S = 1 - F` — **correct as-is on the signed grid**
  (label-indexed). `make_var_tvar` (`utilities.py:498`) is **verified
  index-agnostic** — it uses `ser.index` as the value axis (incl.
  `tvar_unconditional = (ser*ser.index)…`), no `clip(0)`/`abs`, and the caller
  filters `p_total > 0`; so signed VaR/TVaR/cdf work directly off the monotone
  signed grid (`portfolio.py:1371–1412`).
- Empirical-moment block (`:1735–1744`) already weights by `loss` values, so the
  signed grid yields the correct (possibly negative-mean, left-skew) portfolio
  moments with no change (`xsden_to_mwrangler` weights by the signed `x`).
- `remove_fuzz` (`:1719`) runs after the roll — fine (threshold is machine-eps;
  does **not** touch `ftagg_density`, which is read raw from `_fft_aggregate`).

### 3.3 `ft_nots` and `add_exa`

`ft_nots` (`:1704–1717`) is **only** consumed by `add_exa`. We don't call
`add_exa` for signed books this iteration, so **skip `ft_nots` construction when
signed** (pure overhead otherwise). See §4.

### 3.4 `build_many` routing (no back doors)

**Confirmed back door:** the Portfolio branch of `build_many`
(`underwriter.py:644–660`) pre-computes `bs_ = answer.object.best_bucket(log2_,
recommend_p)` and passes it into `update`. For signed books this bypasses the
coarsen-to-fit logic. **Fix:** move the sizing into `Portfolio.update` (its
`bs == 0` branch already calls `best_bucket`; reroute it through
`self._bs_window`) and have `build_many` pass `bs` through (`0` ⇒ auto) without
the pre-compute — exactly the Aggregate `build_many` fix we already made.

### 3.5 Instrumentation — `_limits` / `plot` / `describe` (author priority)

**`plot` is currently broken on signed support and must be fixed in this
iteration.** `Portfolio.plot` (`:2076`) takes its x-range from
`self._limits('range')` (`:2027`), which returns `f(q(0.999)) = [-0.02·q999,
1.02·q999]` — a *tiny* negative lower bound, so a P&L's whole negative tail is
clipped off the axis. This is the portfolio twin of the Aggregate "x-axis messed
up" bug.

- **Fix `_limits`** by mirroring the proven Aggregate fix
  (`distributions.py:5072–5083`): in the `range` branch, when signed
  (`self.density_df.index[0] < 0`) return a **two-sided** quantile range
  `lo = q(1-p)`, `hi = q(p)`, `pad = 0.02·(hi-lo)`, → `[lo-pad, hi+pad]`; else the
  current `f(hi)`. Use `p = 0.999` (linear) / `0.99999` (log), as Aggregate does.
- With `_limits` signed-aware, `plot` itself works unchanged: it plots the
  `p_[A-Za-z]` columns (total + per-line) against the signed index. (`add_exa`
  columns are pricing-scope and absent this iteration, so the density panels are
  exactly what we want to see.) **The whole point of this iteration is being able
  to *see* the combined P&L**, so verify the plot visually on a straddling book.
- **`describe` is signed-safe** (`:933`): it reads `stats_df` — analytic `total`
  moments + empirical from the signed `xsden_to_mwrangler` — and
  `_noise_aware_rel_error` (`moments.py:579`) degrades to **absolute** error when
  `|ref| ≤ VALIDATION_NOISE`, so a mean-zero P&L doesn't blow up the error column;
  `_snap_noise` likewise. The per-unit blocks (`a._describe()`) are correct
  **because** units are driven on their own signed grids (§2c) — another payoff of
  that design choice. *Watch (low risk):* a small-but-nonzero mean
  (`VALIDATION_NOISE < |mean| ≪ sd`) yields an honestly large relative error;
  acceptable, but eyeball it.
- Also add the realised signed window to `Portfolio.info` (mirroring the Aggregate
  `info` window block) so the grid is self-describing.

---

## 4. `density_df` scope for THIS iteration (p_total / F / S only)

Per author: make `p_total` / `F` / `S` (and `loss`, `p_{line}`) correct on signed
support **now**; pend the pricing/allocation columns.

- When the portfolio is **signed** (`any(agg._signed() for agg in agg_list)` —
  add `Portfolio._signed()` mirroring `Aggregate._signed()`), force the
  `add_exa=False` code path (`portfolio.py:1724–1727`): write only `F`/`S`.
- If the caller passed `add_exa=True` on a signed portfolio, **warn** (plain:
  "pricing/allocation columns are not yet available on signed (P&L) support; see
  plan-portfolio-neg-x-pricing.md") and fall back to F/S. Prefer warn+fallback
  over raising so signed portfolios remain usable for distributional work.
- Everything pricing-related is enumerated and audited in the pricing plan; do
  **not** attempt partial fixes to `add_exa` here.

---

## 5. Testing (this iteration)

`tests/test_negative_x.py` (extend) / a new `tests/test_negative_x_port.py`:

- **Combine mean/var/skew:** a 2-unit signed portfolio (e.g. two `ssev` P&L
  lines, or `dsev` lines with negative atoms) — `est_m / est_sd / est_skew`
  match the analytic sum (means add; variances add under independence; third
  central moments add).
- **Mass conservation:** `p_total.sum() ≈ 1` and no truncated negative tail
  (the §2b fix) — assert the min/max of the support window bracket all mass to
  `WINDOW_NINES`.
- **Per-line alignment:** `Σ_line p_{line}`-implied mean equals `est_m`; each
  `p_{line}` integrates to 1 on the signed grid.
- **Signed quantiles:** `q(0.5)`, a low and a high quantile are negative where
  expected; `var`/`tvar` consistent with the 1-D `Aggregate` of the pooled risk
  where that identity holds (independent same-severity units).
- **Non-negative regression:** an existing non-negative portfolio produces a
  byte-identical `density_df` p_total/F/S (the `x_min_tot == 0` fast path) — guard
  against drift.
- **add_exa fallback:** `add_exa=True` on a signed portfolio warns and yields
  F/S-only without raising.
- **Per-unit objects correct (§2c):** after a signed portfolio `update`, each unit
  aggregate is itself valid — no `DefectiveDistributionWarning`, `est_m`/skew match
  its theoretical signed moments, its own `density_df`/`describe` are on its signed
  grid.
- **Instrumentation:** `_limits('range')` returns a two-sided range bracketing the
  negative tail on a signed book (and the current `f(hi)` unchanged on a
  non-negative book); `plot` renders the negative tail (smoke test it doesn't
  clip); `describe` produces finite error columns for a mean-near-zero P&L.
- Sync any DecL programs into `src/aggregate/agg/test_decl.agg` (a `PortPnL`
  section).

`uv run pytest` green (`UV_LINK_MODE=copy`).

---

## 6. Files

- `src/aggregate/portfolio.py` — add `Portfolio._signed()`, `Portfolio._bs_window`
  (+ `_bs_window_df`); two-phase sizing + drive units on own signed grids (§2c,
  §3.1b); F2 present step for `p_total` / `p_{line}`; signed `add_exa` fallback;
  skip `ft_nots` when signed; **`_limits` two-sided signed branch (the `plot`
  fix)**; signed window line in `info`. `best_bucket` kept (thin wrapper).
- `src/aggregate/underwriter.py` — `build_many` Portfolio branch: drop the
  `best_bucket` pre-compute, pass `bs` through (§3.4).
- `tests/test_negative_x_port.py` (new) + `src/aggregate/agg/test_decl.agg`
  (`PortPnL` section).
- `pyproject.toml` / `README.rst` — bump to the next `1.0.0a*`, release bullet.
- `dev/TODO-Remember.md` — close the combine half of #6; leave the pricing half
  pointing at the pricing plan.

---

## 7. Confidence & residual subtleties

**High confidence (verified against the code):**

- The FFT product is origin-at-0 with no per-unit roll (§2a) — `ftagg_density` is
  the full length-`M` FFT in both the default and signed paths
  (`distributions.py:4234`, `:4258`); the signed path's output is origin-at-0
  regardless of each unit's `i0`.
- The truncation bug (§2b): `ift` / `agg.agg_density` drop the wrapped negative
  mass; the F2 full-`irfft`+roll fix is the exact pattern already proven in
  `Aggregate._fft_aggregate` (`:4259–4262`).
- `F`/`S`/`cdf`/`q`/`var`/`tvar` read the `loss` index directly and the rolled
  grid is monotone ascending → signed quantiles work (`portfolio.py:1371–1412`).
- bs is **coarser** (§3.1), and the `_signed()` gate guarantees byte-identical
  non-signed behaviour (§3.0).
- `_bs_window_df` target layout matches the live `Aggregate` one
  (`distributions.py:5527–5538`).

**The six residual risks — all traced to the code, verdicts below:**

1. **`make_var_tvar` / `xsden_to_mwrangler` on signed input — ✅ RESOLVED, clean.**
   `make_var_tvar` (`utilities.py:498`) uses `ser.index` as the value axis
   throughout (`tvar_unconditional = (ser*ser.index)[::-1].cumsum()…`), asserts
   only unique + monotonic-increasing index, no `clip(0)`/`abs`; the caller filters
   `p_total > 0`. `xsden_to_mwrangler` weights by signed `x`. Signed
   VaR/TVaR/cdf/moments work as-is. No code change needed.
2. **Per-unit `ftagg_density` integrity — ✅ RESOLVED, but it FORCED a design
   change.** `ftagg_density` is set raw from `_fft_aggregate` (`:4313`) and is
   **never** touched by `remove_fuzz` (the fuzz-clean is a throwaway copy for
   moments, `:4054`; `density_df.p_total` gets its own separate clean). **But**
   driving a signed unit on a 0-based grid truncates its `agg_density` and trips
   the deficit warning (`:4084`) + corrupts unit moments (`:4054–4058`). → **§2c:
   drive units on their own signed grids.** Resolved by design, not a residual.
3. **`padding` consistency — ✅ RESOLVED.** Portfolio drives every unit with
   `self.padding` and shares `N`, so all `ftagg_density` are length `M//2+1`,
   `M = N<<padding`; `ft_all` is `M//2+1`, `irfft(ft_all, M)` is length `M`.
   Consistent; just keep the portfolio's `M` definition aligned with the units'.
4. **bs forced coarser strips exactness — ✅ understood, surfaced not silenced.**
   A lattice/fine-`bs` unit coarsened by the shared grid loses exactness; with §2c
   the unit's **own** deficit/validation warning fires correctly, so it's visible
   per-unit. The combine isn't exact when ranges are wide; documented.
5. **`build_many` back door — ⚠ CONFIRMED, fix specified (§3.4).** `build_many`
   pre-computes `bs_ = best_bucket(...)` (`underwriter.py:655`) and passes it in.
   Reroute through `Portfolio.update` → `_bs_window`; pass `bs` through.
6. **`describe` / `plot` on signed grid — ⚠ `plot` BROKEN, `describe` OK (§3.5).**
   `plot` clips the negative tail via `_limits('range') = f(q(0.999))` — must
   mirror the Aggregate `_limits` two-sided signed branch (`distributions.py:
   5072–5083`). `describe` is signed-safe: reads `stats_df`, and
   `_noise_aware_rel_error` (`moments.py:579`) degrades to absolute error near 0,
   so a mean-zero P&L doesn't explode the error column. Both fixed/verified this
   iteration (instrumentation is a stated priority).

**Honest overall:** the *core convolution + windowing* is solid (proven Aggregate
F2 pattern lifted up; product-is-origin-at-0 confirmed; `ftagg` origin-invariant
to the unit's `x_min` confirmed at `:4258`/`:4262`). The §2c own-signed-grid drive
removes the biggest sharp edge and *improves* instrumentation. The genuinely
heuristic part remains the *window/bs sizing policy* (§3.1/§3.1b) — sum-of-windows
+ coarsen-to-fit is a sound start but most likely to need a second pass, guarded by
the widen-only moment cross-check.

## 8. Open questions (resolve during implementation)

1. **Window policy** — sum-of-unit-windows is the agreed start (author Q1); add a
   portfolio-level `estimate_agg_window` enhancement only if the conservative sum
   wastes too many buckets in practice.
2. **Exactness vs fit** — how loudly to warn when a lattice/fine-`bs` unit is
   coarsened by the shared grid (residual risk #4).
3. Whether `value_type` gets a DecL keyword — deferred to the pricing plan.
