# Plan — return-period x-axis for quantile (Lee) plots

> **Status: DRAFT — not executed.** Not PnL-specific (applies to `Aggregate`,
> `Portfolio`, and `PnL` plots). A deliberate exercise of the layered plotting
> framework (`aggregate.plots`). Goal: a quantile/Lee panel option that plots the
> outcome against **log return period** instead of the non-exceedance
> probability `p`.

---

## What

The Lee/quantile panel currently plots outcome `q(p)` against `p` on a linear
axis. Add a mode that plots against the **return period** `T` on a **log x**:

- **loss convention** (`_is_loss_value` True): `T = 1 / (1 − p)` — large `p`
  (the right/large-loss tail) maps to large `T`; log-x spreads the tail so you
  *see* the rare big losses.
- **payoff convention** (`_is_loss_value` False, incl. every `PnL` net): the bad
  outcomes sit at **small `p`** (low payoff / a P&L loss), so `T = 1 / p` —
  log-x spreads the small-`p` end where the bad tail lives.

`p < 1` throughout (and `p > 0` for the payoff branch); clip the endpoint atoms.

## API

A plot argument, default `'linear'`, alternative `'return'`:

```python
obj.plot(..., quantile_x='linear')   # today's behavior (x = p)
obj.plot(..., quantile_x='return')   # x = return period, log scale
```

(Name TBD — `quantile_x` proposed; the author's phrasing was "argument to
plot(...), default 'linear', alternative 'return'". `logx` follows from
`'return'`.) Thread it through the Layer-2 compositors (`plot_aggregate`,
`plot_pnl`, `plot_portfolio`) into the Layer-1 quantile worker.

## Where (layering)

- **Layer 1** (`plots/_quantile.py`, `plot_quantile`): owns the transform. Given
  `(p, outcome)` and the convention, compute `T` and draw with `set_xscale('log')`
  when `quantile_x='return'`; label the axis "Return period". The
  loss/payoff branch keys off the object's `_is_loss_value` (passed in).
- **Layer 2** (compositors): pass `quantile_x` and the convention through; no
  domain math.

So the transform lives in exactly one worker and every class's `.plot()` inherits
it — a clean test of the framework's separation.

## Confirmed details (author)

- **Axis ticks: powers of ten.** Label the log return-period axis at `1, 10,
  100, 1000, …` via a log locator (a `LogLocator` / decade ticks), with
  "Return period" as the axis label.
- **`PnL` uses the payoff branch to show the left (loss) tail.** A `PnL` net is
  payoff, so `T = 1/p` — this spreads the small-`p` end, i.e. the **left / loss
  tail** of the P&L, which is what you want to see. (A GCN overlay shares this
  branch across all legs.)
- **Drop/clip the saturating endpoint to stay finite.** Where `p → 1` (loss
  branch) or `p → 0` (payoff branch) the return period diverges; drop/clip that
  terminal atom so `T` and the axis stay finite.

## Remaining open detail

- Does `'return'` also apply to the distribution panel, or only the Lee/quantile
  panel? (Lean: Lee panel only — that's where return period is the natural
  reading. Confirm at build time.)

## Tests

A smoke test per class: `plot(quantile_x='return')` runs, the Lee axis is log,
and the mapping matches `1/(1−p)` (loss) / `1/p` (payoff). No regression to the
default `'linear'` path.
