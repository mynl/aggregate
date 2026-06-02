# Plan: negative-support at the **Portfolio** level — combine, `density_df` audit, pricing

> **Status:** DRAFT — **to be refreshed after [`plan-negative-x-agg.md`](plan-negative-x-agg.md)
> lands.** This is the second half of the split negative-x work. It is
> intentionally lighter than the Aggregate plan: several decisions here depend on
> what we learn implementing signed support at the `Aggregate` level (which
> `density_df` columns actually break, how `value_type` is consumed, etc.). Do
> **not** execute this until the Aggregate work is merged and validated.
>
> Depends on: the Aggregate plan's signed grid, `x_min`/`x_max`/`i0` offsets, the
> `estimate_agg_window` helper (in `utilities.py`), and the `value_type ∈ {loss,
> payoff}` member. This is **TODO #6** (and the pricing/`density_df` follow-ups).

---

## 1. Scope

Three pieces, in dependency order:

1. **Portfolio combine on signed support** — sum signed-support unit aggregates on
   a common window (the "easy half" of negative-x, TODO #6).
2. **The `Portfolio.density_df` column audit** — make *every* column correct on
   signed support, not just `p_total`/`F`/`S`. **Author flag: remind periodically;
   the price column especially must be looked at.**
3. **Distortion / pricing consumption of `value_type`** — apply the actuarial
   (loss) convention; handle `payoff` objects; the harder semantic questions on
   signed support.

---

## 2. Portfolio combine on signed support (TODO #6)

Combining already-computed unit aggregates is a **deterministic sum**, so a
constant per-unit shift de-shifts cleanly — none of the within-unit random-`N`
difficulty arises at the combine level. Plan:

- Each unit aggregate carries its own window `[x_min_k, x_max_k]` and offset from
  the Aggregate work.
- Choose a **common signed grid** for the portfolio: `x_min_tot = Σ x_min_k`
  (lower edge of the sum) up to `x_max_tot = Σ x_max_k`, at a shared `bs`; size
  `N` so the convolved support fits (`W_tot < N·bs`, reusing the alias-free
  condition and `estimate_agg_window` at the portfolio level).
- Roll each unit into the shared FFT order, multiply in Fourier space exactly as
  today (`portfolio.py:1698–1702`), roll the result back onto the signed grid.
- `Portfolio.update` (`portfolio.py:1616`) grows an `x_min`/window analogous to
  `Aggregate.update`; the grid construction at `:1686–1690` (`xs = linspace(0,
  MAXL, N)`) becomes signed.

**Grounding to confirm at refresh:** `best_bucket` (portfolio bucket
recommendation), `ft_nots` construction (`:1708`), and the `add_exa` call
(`:1722`) all assume the 0-based grid.

---

## 3. The `Portfolio.density_df` column audit ⚠

**Author flag (periodic reminder): signed support requires revisiting ALL columns
of `Portfolio.density_df`.** Today many are derived assuming non-negative loss.
The audit must enumerate every column written by `update` and `add_exa`
(`portfolio.py:2122`) and classify each as: *(a) already valid on signed support*
(label-indexed cumulative quantities — `F`, `S`, `p_*`), *(b) needs the signed
grid but otherwise fine*, or *(c) genuinely assumes `loss ≥ 0` and needs
re-derivation*.

- **Price / pricing columns are the priority** — the author expects the fix to be
  straightforward but it **must** be checked, not assumed.
- Candidates to scrutinise: `exa_*` (conditional expectations / allocation),
  `exeqa_*`, `lev`, `epd`, the `loss_max` blanking heuristic, anything using
  `shift(-1, fill_value=...)` on the tail, and any `loss`-weighted sum that
  presumes positivity.
- Tracked in `dev/TODO-Remember.md` (item 6).

Deliverable: a column-by-column table (column → assumption → signed-support
verdict → fix), produced **as discovery** during the refresh once we can run a
signed-support Portfolio.

---

## 4. Distortion / pricing and `value_type`

The Aggregate plan tracks `value_type ∈ {loss, payoff}` (default `loss`) but
leaves it **inert**. Here we make it bite at the pricing layer:

- Distortions assume the **actuarial loss** orientation ("more is worse").
- A `payoff` object (asset / "more is better") must be **negated** (or the
  distortion's dual applied) before pricing, then results mapped back.
- Open semantic questions (to settle at refresh, possibly a research note): what
  TVaR / distortion premium / capital allocation *mean* on a P&L; whether the
  lifted-natural-allocation admissibility (`bounded` + mass) story changes when
  the support straddles 0; how `Distortion.price` forwards/backwards `S` behaves
  on signed support.

This is the most open part and may spin out its own plan.

---

## 5. Testing (sketch — firm up at refresh)

- **Combine mean/var:** two P&L units combine to the analytic mean/variance of the
  sum on an aligned signed grid.
- **Independence sanity:** portfolio total of independent signed units matches the
  1D `Aggregate` of the pooled risk where that identity holds.
- **`density_df` audit regressions:** once columns are fixed, snapshot the
  signed-support `density_df` for a small P&L portfolio.
- **Pricing:** a `payoff`-typed object prices as the negated `loss` object
  (consistency check of the `value_type` handling).

---

## 6. Open questions (to resolve at refresh, post-Aggregate)

1. Portfolio window/grid policy — sum-of-unit-windows vs. a fresh portfolio-level
   `estimate_agg_window`; how `best_bucket` adapts.
2. The `density_df` column verdicts (discovery-driven — §3).
3. Pricing semantics on signed support and the `value_type` dual (§4) — keep here
   or spin out a dedicated pricing plan?
4. Whether `value_type` gets a DecL keyword.
5. Version (a follow-on `1.0.0a*` after the Aggregate cycle).
