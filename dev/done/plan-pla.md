# Plan — `prob_loss_assets` / `pla` (free choice of capital anchor)

> **Status: SHIPPED 1.0.0a97.** New capability + tests landed; version bumped
> and `CHANGELOG.md` updated. `uv run pytest` green; the frozen baseline is
> unmoved. **Deviation from §2:** the cached `_grid_distribution()` handle's
> `lev` was *wrong* on the `p_total > 0` subset (it drops the empty low buckets
> where `S == 1`, so `E[min(X,a)]` undercounts by the missing slab — e.g.
> `lev(a)=682` vs `exa=999.6`). Rather than read the `exa`/`exa_total` column,
> the handle is now built on the **full** contiguous `bs` grid; the var/tvar
> kernel filters `p>0` internally so `q`/`tvar`/`mean` are byte-identical, while
> `lev`/`cdf`/`sf` are now correct and match `add_exa` to dust. `pla` and
> `price_pentagon_ex` then use `GridDistribution.lev` as the single LEV source,
> as the plan intended. Tests in `tests/test_prob_loss_assets.py`.

---

## 0. Motivation

Today the capital level for pentagon pricing is fixed by **either** a VaR
probability `p` (`a = q(p)`) **or** an asset level `a` — see
`_pricing.price_pentagon` (`_pricing.py:79`), which raises unless *exactly one* of
`p`/`a` is supplied. There is no way to anchor on the **expected loss** `L`
(the limited expected value `E[min(X, a)]`), even though `{p, a, L}` are three
views of the same point on the distribution and any one determines the other two.

`pla` fills that gap: given any **one** of `p`, `L`, `a`, it returns all three.
It is the small, well-tested primitive the pentagon completion then leans on so
the user can be free over the capital anchor.

## 1. The function

**Name:** `prob_loss_assets`, alias `pla`. **Home:** `GridDistribution`
(`_grid_distribution.py`) — it already owns `q`, `cdf`, `lev`, `mean`, the exact
pieces needed. Thin callers on `Aggregate` and `Portfolio` (§2).

**Signature** (all keyword-only after `*`, exactly one non-zero/non-None):

```python
def prob_loss_assets(self, *, p=None, L=None, a=None):
```

Raise `ValueError` if the count of supplied (non-None) args ≠ 1.

**Semantics — given one, determine the other two:**

- **`a` given** (asset level): snap to grid — `p = self.cdf(a)`, then
  `a = self.q(p)` (re-snap so the returned `a` is an exact grid point and
  self-consistent with `p`); `L = self.lev(a)` (the `E[min(X,a)]` / `exa`
  convention, `_grid_distribution.py:279`).
- **`p` given** (probability level): `a = self.q(p)`; `L = self.lev(a)`. (`a` is
  already a grid point from `q`, so no extra snap.)
- **`L` given** (expected-loss level):
  1. **Feasibility check:** require `L < self.mean()` (a LEV is bounded above by
     `E[X]`; `L → E[X]` only as `a → ∞`). Error with a clear message otherwise —
     this is the fragile input (see §3), so the guard must be explicit.
  2. **Root-find `a` with `lev(a) = L`.** `L(a) = ∫₀ᵃ S(x) dx` is increasing with
     `dL/da = S(a)` — a clean Newton step `a ← a − (lev(a) − L)/S(a)` (S(a) from
     `self.sf(a)`). Bisection fallback if a step leaves the support or S(a)
     underflows in the far tail. Bracket: `[0, q(1)]`.
  3. **Snap** the solved `a` to the grid, then `p = self.cdf(a)`,
     `L = self.lev(a)` (recompute on the snapped `a` so the returned triple is
     mutually consistent, not the requested `L`).

**Return:** a small module-level namedtuple, matching the `QuantileFunctions` /
`ChoquetWeights` pattern already in this file (the `results.py` dataclasses are
for heavier Portfolio readouts; a 3-tuple doesn't warrant one):

```python
ProbLossAssets = namedtuple('ProbLossAssets', 'p L a')
```

`pla = prob_loss_assets` as a class-level alias.

## 2. Callers

`Aggregate.prob_loss_assets` / `Portfolio.prob_loss_assets` (+ `pla` alias)
delegate to the underlying grid view, whose `lev` (§1) is the single LEV source.
The stored density column it must agree with differs by class — `exa` on an
`Aggregate`, `exa_total` on a `Portfolio` (the `add_exa` convention, which
`_pricing._limited_ev` matches to dust) — so route through `GridDistribution.lev`
rather than reading either column directly, and reuse whatever grid handle the
post-split objects already hold rather than rebuilding a `GridDistribution`.

## 3. Wire `pla` into pentagon pricing (`price_pentagon_ex`)

**Goal: `price_pentagon_ex` accepts *any* soluble config, errors on the
impossible ones, and warns when the solve used accounting losses that the
distribution does not reconcile.** It is the full-power front door over
`Pentagon.solve` plus the `pla` distributional bridge; `price_pentagon` /
`solve_obj` stay exactly as-is underneath.

**The Pentagon engine is the gatekeeper.** `Pentagon.solve` completes any soluble
triple from `{L, M, P, Q, a, LR, PQ, ROE}` (the `make_possible_pentagons`
enumeration) and raises `ValueError('Insoluble case: ...')` on the rest — e.g.
`{PQ, ROE, LR}` (three ratios, scale-free) cannot pin the level and is rejected.
`_ex` lets that pass through; no config enumeration is re-implemented.

**The distribution supplies exactly one extra equation: `L = lev(a)`.** That is
the whole role of `pla`. How it enters depends on how much accounting the caller
already gave:

1. **Translate the probability spelling.** If `p` is given, `a = q(p)`; `p` is not
   a `Pentagon` variable, so this is the only pre-step. (`a`/`L` are already
   `Pentagon` variables.)
2. **Inject `L = lev(a)` only when the accounting is short one equation.** Count
   the supplied pentagon quantities (after `p → a`):
   - **a capital level is fixed (`a` known, or `L` given as an anchor via
     `a = pla(L=L).a`) and one pricing target is supplied** → inject
     `L = lev(a)` to complete the triple, then solve. This *is* the
     `solve_obj` path; `L` is read from the curve, so it is consistent by
     construction.
   - **three quantities already supplied** (`{P, M, ROE}`, `{a, P, M}`, …) →
     hand straight to `Pentagon.solve`; do **not** inject. `L` is whatever the
     accounting yields.
   - **fewer than three and no capital level determinable** (`{P, ROE}`,
     `{P, M}` with no `p`/`a`/`L`) → genuinely under-determined; `Pentagon.solve`
     raises.
3. **Solve** via the existing engine.
4. **Uniform post-check + warn.** *Always* compute `lev(a_solved)` and compare to
   the solved `L`. They match to bucket tolerance whenever `L` was injected from
   the curve (anchor-on-`p`/`a`/`L`), so those paths stay silent; only an
   accounting-determined `L` can diverge:

   ```
   if abs(L_solved - lev(a_solved)) > tol:
       warn(f'price_pentagon_ex: solved by accounting identities; L = '
            f'{L_solved:.6g} does not match the limited expected loss '
            f'E[min(X,a)] = {lev_a:.6g} at assets a = {a_solved:.6g} '
            f'(gap {L_solved - lev_a:.3g}). The accounting loss ignores the '
            f'limit/default haircut E[(X-a)+].')
   ```

   No mode-tracking is needed — the single post-hoc comparison classifies every
   config. **Mechanism: a Python `warnings.warn(..., UserWarning)` — this is a
   pricing-time advisory, deliberately *not* routed through `explain_validation`
   / the `constants.py` flags.** Threshold: the existing bucket tolerance; one
   tight line per house style.
5. **Report `p`.** Attach `p = cdf(a_solved)` alongside the eight stats (the
   `calibration_df` row at `_pricing.py:207` is the precedent for carrying `p`
   next to the octet).

```python
def price_pentagon_ex(self, *, p=None, a=None, L=None,
                      M=None, P=None, Q=None, LR=None, PQ=None, ROE=None):
    # full Pentagon vocabulary + the p spelling; pla supplies L = lev(a) when
    # the accounting is short one equation, and the post-check warns when an
    # accounting-determined L disagrees with lev(a). solve() rejects insoluble
    # configs (e.g. {PQ, ROE, LR}).
    ...
```

`price_ccoc` (`= price_pentagon(p=p, ROE=ccoc)`) is unaffected — it anchors on
`p`, so its `L` is always the injected `lev(a)` and it never warns.

Rationale for keeping `pla` a **separate, separately-tested** primitive rather
than folding it into the pentagon entry: the `L → (p, a)` root-find is the
**fragile** step (`L` near `E[X]` is ill-conditioned because `dL/da = S(a) → 0`).
A named call makes that failure mode visible — and keeps the `L`-anchor's
re-snapping (so the returned `L` is exactly `lev(a)` at a grid point) in one place.

## 4. Tests

- **Round-trip consistency:** for a fixed grid pick `a₀`; `pla(a=a₀)` →
  `(p, L, a)`; then `pla(p=p)` and `pla(L=L)` return the *same* snapped triple.
- **`L`-anchor accuracy:** solved `a` reproduces `L` to bucket tolerance; Newton
  vs. bisection agree.
- **Feasibility errors:** `L ≥ mean()` raises; zero or >1 anchors raise; each with
  a clear message.
- **Ill-conditioning:** `L` very close to `E[X]` (far tail, `S(a) → 0`) — confirm
  the bisection fallback engages rather than a Newton blow-up.
- **`price_pentagon_ex` — anchor equivalence (silent path):** anchoring on `L`
  (with one pricing target) gives the *same* octet as anchoring on the equivalent
  `a` and the equivalent `p`; the returned `L` equals `lev(a)` at the snapped grid
  point and **no warning fires**. Existing `price_pentagon`/`solve_obj` cases
  unchanged (baseline frozen).
- **`price_pentagon_ex` — accounting solve (warns):** `{P, M, ROE}` (no
  `p`/`a`/`L`) solves via `Pentagon.solve`, deduces `a = P + M/ROE`, reports
  `p = cdf(a)`, and **warns** that `L = P−M` disagrees with `lev(a)` whenever it
  does; construct a case where it agrees (pick `P` so `P−M = lev(a)`) and confirm
  the warning is suppressed.
- **Insoluble configs error:** `{PQ, ROE, LR}` (and other non-enumerated triples)
  raise via `Pentagon.solve`; `_ex` propagates cleanly.
- **Uniform post-check:** an over-determined-but-consistent input (e.g. `{a, P, M}`
  with `P−M` chosen to equal `lev(a)`) stays silent; the same with `P−M ≠ lev(a)`
  warns. Confirms the check is on `L_solved` vs `lev(a_solved)`, not on input mode.
- Both `Aggregate` and `Portfolio` callers covered.

## 5. Out of scope / open questions

- No change to `Pentagon.solve` / `solve_obj` accounting — `_ex` only adds the
  `pla` distributional bridge (`p`/`a`/`L` ↔ grid) and the `p`/`lev(a)` readout;
  the octet completion is unchanged. (The accounting-`L` vs `lev(a)` behavior is
  no longer open — it moved to §3 as a pre-coding decision.)
- Signed / payoff distributions: `lev`/`q` conventions assume the zero-based grid
  (`_grid_distribution.py:288`). Defer signed-anchor behaviour to its own pass
  (cf. `plan-signed-distortion-calibration.md`); note the limitation in the
  docstring.
- Confirm the namedtuple name (`ProbLossAssets`) against the existing surface
  before coding. (`price_pentagon_ex` / `pla` confirmed.)
