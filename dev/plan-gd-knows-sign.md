# Plan — `GridDistribution` knows its sign (loss vs payoff)

> **Status: DRAFT — not executed.** Make orientation (loss vs payoff) an
> intrinsic, immutable property of a `GridDistribution` (GD), so the rare
> "which side is bad?" operations — pricing and quantile/return-period plotting
> — consume a self-describing object instead of a side-channel `is_loss_value=`
> flag threaded through every call.

---

## Why

A `GridDistribution` represents a random variable `X` — a cash flow *to the
object*. The holder knows whether `X = 1` means "I pay 1" (loss) or "I receive
1" (payoff); that orientation is part of the cash flow's identity, fixed at
construction and never mutated. Today GD carries only `(x, p, bs, name)` and the
orientation lives separately on each holder (`Aggregate`/`Portfolio
._is_loss_value`), so every orientation-dependent operation re-injects it:

- `spectral.Distortion.effective_g(view, is_loss_value=...)` (`spectral.py:772`)
- `_pricing.py:359` `reverse = not obj._is_loss_value`; `:438` `transform = …`
- `_portfolio_common.py:151, 291, 396` thread `port._is_loss_value` into `effective_g`
- the Lee/return-period worker (`plan-plot-return-period.md`) takes
  `is_loss_value=agg._is_loss_value`

**The objective math does not depend on sign.** `cdf`, `sf`, `q`, `var`, `tvar`,
`lev`, `mean` are defined on the rv and are identical for a loss or a payoff —
they stay byte-for-byte unchanged. Orientation is a *separate axis*, consulted
only when an operation asks "give me the **bad** outcome": pricing (which tail
the distortion loads) and the Lee plot (which side the return-period axis
spreads). So the change is purely additive: GD gains a fact it already could
have carried, the kernel is untouched, and `Distortion.price(gd)` /
`plot_quantile(gd)` become unambiguous with no extra argument.

This was settled in design discussion (the three objections — "methods would
ignore the field", "mutable flag needs invalidation", "math-only callers have no
role" — all fall: the kernel is objectively defined so it legitimately doesn't
consult the field; `is_loss_value` is immutable per object; orientation is
always known at construction once the axis is signed).

## What

1. **`GridDistribution` gains an immutable public `is_loss_value`** (matches the
   `_is_loss_value` token on holders; GD exposes it *without* the underscore,
   like its public `x` / `p`). Set at construction, defaulting `True` for the
   sign-agnostic callers that don't care.
2. **The objective accessors are unchanged.** `q`, `var`, `tvar`,
   `tvar_threshold`, `cdf`, `sf`, `pmf`, `mean`, `lev`, `tvar_of_limited` do not
   read `is_loss_value`. (A docstring line states that orientation is metadata
   for the "what's bad" operations, not an input to the kernel.)
3. **The orientation-dependent logic moves onto GD, defined once.** The
   loss/payoff return-period map (`T = 1/(1−p)` loss, `T = 1/p` payoff) and the
   "bad-tail" direction become GD methods, so the Lee worker and the upcoming
   `tail_df` (`plan-summary-tail-tables.md`) consume the *same* accessor instead
   of each re-implementing the branch.

## Where (file by file)

- **`_grid_distribution.py`** — add `is_loss_value=True` to `__init__` and
  `from_series`; store it; surface it in `__repr__`. Add the orientation-aware
  accessor(s): a return-period map `T(p)` and/or a `bad_tail` helper that the
  Lee plot and `tail_df` share. The var/tvar/quantile kernel is untouched.
- **`_aggregate.py`** — `_grid_distribution()` (`:4373`) and
  `_sev_grid_distribution()` (`:4392`) pass the holder's role when building the
  cached GD. The `value_type` setter (`:2459`) must also reset the GD caches
  (`self._dist = None`, `self._sev_dist = None`) so a post-build role change
  rebuilds with the new orientation. (Today only `update()` resets them.)
- **`_portfolio.py`** — `_grid_distribution()` (`:1504`, builds `self._dist` at
  `:1520`) passes `port._is_loss_value`.
- **`bounds.py`** — `GridDistribution.from_series(...)` (`:144`) and the raw
  `GridDistribution(self._x, self._prob)` (`:667`) are pricing-context, so pass
  the orientation in hand (or accept the `True` default where it is a plain
  loss). No behavior change for the default case.
- **`spectral.py` / `_pricing.py` / `_portfolio_common.py`** — where a GD is
  already in hand, read `gd.is_loss_value` instead of threading the flag.
  `effective_g`'s `is_loss_value=` kwarg stays (transitional) but can default
  from the GD; the `_portfolio_common` call sites stop passing it once the GD
  carries it. (Scope this conservatively — see open questions.)
- **`plots/_quantile.py` + compositors** — route the Lee workers to take a GD
  (the holders already build `agg._grid_distribution()` /
  `_sev_grid_distribution()`); orientation and the return-period map come from
  the GD, removing the `is_loss_value=` argument from `plot_aggregate` /
  `plot_severity` / `plot_reins_occ`. **This is the revisit of
  `plan-plot-return-period.md`** (which currently passes `is_loss_value=` from
  the holder).

## Execution order (author)

GD-knows-sign (this plan) → **finish the return-period plot** (revisit
`plan-plot-return-period.md` to consume the GD) → **`summary_df` / `tail_df`**
(`plan-summary-tail-tables.md`, whose `tail_df` reads the same orientation) →
then back to **PnL**. Each is a separate version bump.

## Risks / open questions

1. **Severity-curve orientation in the aggregate Lee panel.** A severity is
   intrinsically a loss, so `_sev_grid_distribution()` carries
   `is_loss_value=True`. But a *payoff* aggregate's Lee panel draws the
   aggregate and its severity on one axis with the **aggregate's** orientation.
   So the panel transform should key off the primary (aggregate) GD, not each
   curve's own flag. Decide: the worker takes a panel-level orientation (from
   the aggregate GD) and the severity curve follows it, vs. each GD self-orients
   (which would split a payoff panel). Lean: panel orientation = aggregate GD.
2. **How far to push the pricing migration now.** Minimal: GD carries the flag
   and the Lee plot consumes it. Fuller: drop the threaded `is_loss_value=` from
   `effective_g` / `_portfolio_common`. Recommend doing the GD field + plot in
   this plan and migrating pricing call sites only where a GD is already the
   argument, leaving `effective_g`'s kwarg as a defaulted shim for one release.
3. **Sample / empirical GDs.** Any GD built from a sample or a bare series
   (`bounds.py`, ad-hoc) defaults to `is_loss_value=True`; confirm no
   payoff-context construction site silently takes the wrong default.
4. **Naming the accessor.** One canonical name for the return-period / bad-tail
   map (e.g. `return_period`), shared by the plot and `tail_df`. Vet against the
   existing GD surface before adding (per CLAUDE.md naming rule).

## Acceptance

- `GridDistribution(x, p, is_loss_value=...)` and `.from_series(..., is_loss_value=...)`;
  `.is_loss_value` is a read-only attribute; `__repr__` shows it.
- The objective kernel (`q`/`var`/`tvar`/`cdf`/`sf`/`lev`/`mean`) is unchanged —
  existing numeric tests pass untouched.
- The loss/payoff return-period map lives on GD and is consumed by the Lee
  worker (and is ready for `tail_df`); no compositor passes `is_loss_value=`.
- Holders thread their role into the GD they build; `value_type` change rebuilds
  the GD cache with the new orientation.
- Version bump + CHANGELOG entry.
