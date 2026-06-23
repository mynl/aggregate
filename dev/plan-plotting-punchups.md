# plan-plotting-punchups

Small fixes and polish items for the plotting subsystem (in flight — keep each
item terse for now; flesh out when the surrounding work settles).

## Items

1. **`pnl` aggregates: don't overlay severity.** `agg = freq x sev` is actuary
   framing (they want the severity). `pnl = prem - loss` is UW / finance
   framing — that audience isn't concerned with the per-claim severity, and the
   loss-convention severity overlaid on a payoff-convention aggregate just reads
   as a wrong-sign distraction. So for a `pnl` (affine-active) aggregate, plot
   the aggregate only; no severity overlay. The actuary who wants the severity
   still has `agg.sevs[].plot()`. Gate on `_agg_affine_active()` / `_signed()`
   in `plots/_aggregate.py` (both the discrete and continuous branches).
   Related: document the loss/payoff two-level `pnl` convention (see
   `_signed_severity` vs `_signed`).
