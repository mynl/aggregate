# Plan — user-facing `summary_df` + `tail_df`; rename the QA/behavior frames

> **Status: DRAFT — not executed.** Applies to `Aggregate`, `Portfolio`,
> `PnL`, and (where noted) `BivariateAggregate`. Goal: turn the daily-driver
> display frames from *validation artifacts* into *risk views* a practitioner
> reads at a glance, and free up the two best names (`summary_df`, `tail_df`)
> for them.

---

## Why

`summary_df` today is a **validation** frame: Freq/Sev/Agg × `EX | Est EX | Err
EX | CV | Est CV | Err CV | Sk | Est Sk`. It grew up to prove the FFT reproduced
the analytic moments — "if the first three moments match, the aggregate is *not
unreasonable*." That is a QA story, not a user experience.

`tail_df` today is the **tail-behavior classifier** (`min/max` support,
`left_tail/right_tail` class, `bounded`, `concentrated`, power-law `alpha`). Also
diagnostic, not a risk view.

A risk manager / actuary / quant asks three things of a loss distribution:
*what is it made of, how bad does it get, and how much capital does that cost.*
None of those is answered by the current frames. This plan delivers them and
renames the diagnostics out of the way.

## Naming map

| New name | Content | Was |
|---|---|---|
| `summary_df` | at-a-glance moments + key percentiles, Freq/Sev/Agg | (new) |
| `tail_df` | return-period / exceedance table (VaR, TVaR, …) | (new) |
| `validation_df` | the moment-vs-estimate error table | old `summary_df` |
| `tail_behavior_df` | support + per-side tail class + concentration | old `tail_df` |

Keep `tail_df` / `summary_df` as **deprecated aliases** for one release if any
internal caller still wants the old payload — but prefer migrating callers (see
*Migration* below) since the names are being repurposed, not retired.

---

## Table 1 — `summary_df` (the first impression)

The compound-model identity made legible. **Index = `Freq` / `Sev` / `Agg`**
(the `X` index, as today). The Freq×Sev = Agg split *is* the wow: each row
answers a different question — count risk, single-claim severity, total loss —
and the percentiles trace where the tail comes from (a heavy `Agg` skew you can
see is inherited from `Sev`).

**Columns:** `E[X] | SD | CV | Skew | p0.01 | p0.50 | p0.99`

- `SD` and `CV` are **both always present** (stable column layout). `CV =
  SD / E[X]` is populated only when `|E[X]| >= tol`; for signed / near-break-even
  P&L it is left blank (NaN), since CV is meaningless near a zero mean. SD never
  blanks. Footnote: `CV = SD/E[X]; blank when |mean| < tol`.
- `Skew` is well defined even at mean 0, so it stays for signed objects.
- Percentiles: `Agg` via `self.q`; `Sev` via `sev_density` / `fz.ppf` (already on
  the grid — mixtures included). **`Freq` percentiles are blank** (the engine
  never materializes the count distribution — see *Component percentiles* below).
  Freq moments stay PGF-exact, so the Freq row is still informative; only the
  three percentile cells are empty (as the old validation frame had them).

**Portfolio:** MultiIndex `(unit, X)` per unit, plus a `total` block carrying
the `Agg` row only (a portfolio has no single Freq/Sev). Emphasize the `Agg` /
`total` rows in `_repr_html_` (bold) — that's the "what's my number" line.

## Table 2 — `tail_df` (the centerpiece)

The return-period / exceedance table — the language of reinsurance submissions,
cat-model output, and Solvency II / rating-agency capital.

**Index = return period** `T` (default ladder `2, 5, 10, 25, 50, 100, 200, 250,
500, 1000`; overridable via `tail_df(periods=...)`). Highlight **1-in-200**
(99.5%, Solvency II) **and 1-in-250** (99.6%, US capital-adequacy / rating
anchor) in `_repr_html_`.

**Columns:** `p | VaR | TVaR | xsVaR | VaR/Mean`

- `p` non-exceedance probability for the row.
- `VaR = q(p)` — the quoted number.
- `TVaR = tvar(p)` — the priced number; **adjacent to VaR** so the VaR↔TVaR gap
  (tail-fatness) reads at a glance.
- `xsVaR = VaR − E[X]` — excess of VaR over expected, i.e. capital. Named with
  the library's own `xs` token; the bridge to the Price tab.
- `VaR/Mean` — leverage; turns dollars into intuition.

Column order keeps the two *measures* (VaR, TVaR) adjacent, then the two
*VaR-derived* columns. If symmetric TVaR columns (`xsTVaR`, `TVaR/Mean`) are
wanted later, switch to a two-block layout (`VaR | xsVaR | VaR/Mean | TVaR |
xsTVaR | TVaR/Mean`) — author's call.

`E[X]` carried in the frame caption / `.attrs`, not as a phantom row.

**Loss vs payoff convention** (reuse `_is_loss_value`, per
`plan-plot-return-period.md`): loss objects map `T = 1/(1−p)` (right tail);
payoff / PnL objects map `T = 1/p` so the table reads off the **downside**. A
signed object's `tail_df` is therefore a downside-capital table, not a
nonsense upper-tail one.

**Portfolio:** add a leading index level `unit` → `(unit, T)`, each unit's
aggregate tail plus `total`. (Per-unit *contribution* to the total tail — TVaR
allocation — is allocation/pricing territory; leave it to the Price path, not
here.)

`tail_df` is **Agg-only** (no Freq/Sev rows): tail risk is a property of the
total, which keeps the two tables cleanly complementary — `summary_df` = "made
of", `tail_df` = "how bad".

---

## Differentiators to surface in display

- **Exact, not simulated.** These tail numbers (incl. 1-in-1000 TVaR) come from
  the FFT grid, not Monte Carlo — no wobble, milliseconds. Worth a caption.
- Highlight the **1-in-200 / 99.5%** row (Solvency II economic capital, rating
  anchor) — an eye goes straight to it.

## Component percentiles (Sev free; Freq blank, with an opt-in path)

**Severity — nothing to build.** After `update()` the severity density is already
materialized on the grid (`sev_density` / `sev_density_df`, `_aggregate.py:873`),
and `Severity.fz` carries a layer/mixture-aware `ppf`. So Sev-row percentiles come
from `cumsum(sev_density)` (which *is* the `wts` mixture, so mixtures are free) or
`fz.ppf`. No synthesis.

**Frequency — blank, full stop.** The engine carries frequency only as a **PGF**
(`freq_pgf`), applied in the Fourier domain — it never materializes a count
distribution (no stored scipy frozen dist, even for Poisson). This is the real
asymmetry: *agg and sev densities live on the grid; frequency does not, by
design.* The old validation frame left Freq percentiles blank for exactly this
reason. `summary_df` does the same: the Freq-row `p0.01/p0.50/p0.99` are **blank**;
the row still carries `E[X]/SD/CV/Skew`, PGF-exact from `stats_df` — which is what
that row is *for* (count volatility). `summary_df` builds **nothing** to fill them.

The `summary_df` docstring states this and points the way out:

> *Freq-row percentiles are blank: frequency is represented by its PGF, not a
> materialized distribution. Use `agg.create_frequency()` to get the count
> distribution as a first-class Aggregate, then `.q(...)` / `.tvar(...)` on it.*

Materializing the count distribution on demand is a separate, orthogonal feature
— `Aggregate.create_frequency()` / `Portfolio.create_frequency()` via the
`dsev [1]` trick. It has **no dependency** on this plan and is tracked in
`dev/plan-create-frequency.md`.

## Where

- `_aggregate.py`: new `summary_df`, `tail_df` properties; rename existing
  `summary_df`→`validation_df` (keep `_describe`), `tail_df`→`tail_behavior_df`.
- `_portfolio.py`: same four, with the `unit` index level; reuse
  `Aggregate._describe` plumbing where shared.
- `_pnl.py`, `bivariate.py`: mirror; honor the payoff convention in `tail_df`.
- `tail.py`: the behavior-frame renderer is unchanged; only the property name
  that exposes it moves.

## Migration

Internal callers of the **old** `tail_df` (behavior) must move to
`tail_behavior_df`:

- `_bucket_window.py:367` — `agg.tail_df.loc['aggregate']`
- `_portfolio.py:747, 827` — `self.tail_df`, `a.tail_df.loc['aggregate']`
- `bivariate.py:1696` — `self.tail_df`
- `_aggregate.py:489, 508` — `tail_description` / `tail_explanation` docrefs

Internal callers of the **old** `summary_df` (validation): `_repr_html_`, `qd`,
the API's summary route — repoint to `validation_df` *and* decide which frame the
default display shows (proposal: `summary_df` becomes the headline; `qd` shows
`summary_df` then `tail_df`; `validation_df` is available but no longer the lead).

## Downstream (aggregate_api)

The web playground's **Summary** tab becomes `summary_df`; add a **Tail** view
for `tail_df`; demote `validation_df` + raw `info` under **More ▾**. (Tracked
separately in the api repo — this plan only delivers the library frames.)

## Risks / open questions

1. **Component percentiles** — Sev is free (`sev_density` / `fz.ppf`); Freq is
   blank by design (no materialized count dist). No risk here; the on-demand
   count distribution is `create_frequency`, a separate plan.
2. **Return-period ladder for thin/short support.** A near-Gaussian or bounded
   book may have `q(0.999)` at or near the support max — fine, but the `×Mean`
   column stays sensible; verify no divide-by-tiny on a near-zero mean (reuse
   the CV `tol` guard).
3. **Alias lifetime.** One release of deprecated `tail_df`/`summary_df` aliases,
   or hard cut with the migration above? Author's call.

## Acceptance

- `agg.summary_df`, `agg.tail_df`, `agg.validation_df`, `agg.tail_behavior_df`
  all present and documented (NumPy-style); Portfolio/PnL/Bivariate analogues.
- Signed object: `summary_df` CV column shows `—`; `tail_df` reads the downside.
- No internal caller references the *old* meaning of `summary_df`/`tail_df`.
- Version bump + CHANGELOG entry.
