# Replacing the epsilon-jump trick in `SeverityDHistogram`

**Status:** SHIPPED in 1.0.0a26 (2026-06-03). `_DiscreteRV` added; the
eps-trick + `max_log2` removed from `SeverityDHistogram._build`; an exact
summed-moments fast-path added to `moms()`. Golden (`severity_layer_golden.json`)
and baseline (`Sym.Dice`, `Port.Bodoff` stats_df/describe) re-captured to record
the now-exact values; aggregate density bit-stable. Full suite green (907).
**Date:** 2026-06-03 (reviewed & revised 2026-06-03 after the note/hints work)

> **Correction discovered during implementation (2026-06-03).** §2's claim that
> "moments never come from `fz` for a discrete severity — `_build` precomputes
> `sev1/2/3`, so the unlimited case already uses the fast-path" is **wrong**.
> `_build` sets `self.limit = min(self.limit, max(xs))`, so a discrete severity's
> `detachment` is always *finite* (= `max(xs)`), never `inf`. The `moms()`
> path-1 fast-path requires `detachment == inf`, so it **never fired for
> discrete** — *every* discrete moment, unlimited included, was computed by
> `_numerical_moms` (quad on the eps-trick `isf`) and returned the trailing-9s
> artifact (mean `3.4999999995` instead of `3.5`). The fix therefore routes
> **all** discrete moments (unlimited, limited, layered) through the new exact
> `_DiscreteRV.layer_moments` summation — see §4a, which became the headline of
> the change, not a footnote. The `max_log2` eps offset also *scales with atom
> magnitude*, so the old hack was least accurate for large-valued atoms
> (visible as the larger `Port.Bodoff` moment shift on re-capture).
>
> Line-number anchors (drifted by the 1.0.0a25 `hints{}` work — search by
> symbol): `SeverityDHistogram._build` ~7791, `_DiscreteRV` inserted ~7777,
> `max_log2` ~60 (now unused), `moms()` dispatch ~7535, `_numerical_moms`
> ~6883.

## 1. The problem

`SeverityDHistogram._build` (distributions.py ~7404-7433) represents a *truly
discrete* severity — point masses at user-supplied loss values `xs` with
probabilities `ps` — by abusing `scipy.stats.rv_histogram`, which is a
*continuous* piecewise-linear-CDF object. To make a continuous object behave
like a step function, the code inserts a tiny synthetic bin to the left of each
atom and pours the whole atom's mass into it:

```python
scale = float(np.max(np.abs(xs)))
d = max_log2(scale if scale > 0 else 1.0)        # smallest float-resolvable step
xss = np.sort(np.hstack((xs - 2 ** -d, xs)))     # an x-eps edge before each atom
pss = np.vstack((ps, np.zeros_like(ps))).reshape((-1,), order='F')[:-1]
self.fz = ss.rv_histogram((pss, xss), density=False)
```

The result is a near-vertical CDF rise of width `2**-d` at each atom. It works,
but it is offensive: the distribution is conceptually discrete, the
representation is continuous, and correctness hinges on `2**-d` being small
enough to be invisible yet large enough to stay representable in float64
(`max_log2`). `SeverityFixed` inherits the same trick (single atom, mass 1);
`SeverityMeta` has an analogous `b1size = 1e-7` hack to pin the mass at zero
(out of scope here but the same smell).

## 2. What `self.fz` actually has to provide (the real contract)

`fz` is *only ever* touched through the scipy frozen-distribution method
surface — there is **no** `isinstance(fz, rv_histogram)` check anywhere in the
codebase (confirmed by grep across `src/`). So the contract is just a set of
duck-typed methods. The full set of call sites and what they need:

| Call site | Methods used on `fz` | Reached for a *discrete* severity? |
|---|---|---|
| `Aggregate.discretize` (3813-3821) | `cdf`, `sf` at bucket edges `xs ± bs/2` | **Yes — the hot path.** This is what feeds the FFT. |
| `Severity._apply_lb_ub` (splice) | wraps `cdf/sf/isf/ppf/pdf` | only if `sev_lb/sev_ub` set |
| `Severity._compute_attachment_probs` | `sf(attachment)`, `sf(detachment)` | only if a layer is present |
| `Severity._apply_layer_attachment` | wraps `cdf/sf/pdf/isf/ppf` | only if a layer is present |
| `Severity.support_description` (signed) | `support()` | signed discrete only |
| `_numerical_moms` (6588-6648) | `sf`, `isf`, `stats('mvs')` | discrete **only when layered** (else fast-path) |
| `_apply_signed` / `moms` | `moment(1..3)`, `stats` | **not** for discrete — `sev1/2/3` precomputed in `_build` |
| `grid sizing` (5353-5354) | `support()` | signed only |

Key facts:

- **Moments never come from `fz` for a discrete severity.** `_build` already
  sets `sev1/sev2/sev3` exactly from `Σ xⁿ p`. The `moms()` fast-path and the
  signed path both use those. So `fz.moment`/`fz.stats` are dead code for
  unlayered discrete severities.
- **The FFT only needs `cdf`/`sf` evaluated at bucket boundaries**, which sit
  at half-bucket offsets `xs ± bs/2` — they never coincide with the atoms. A
  *true* right-continuous step CDF returns bit-identical values there (see §4).
- `pdf` of a discrete law is genuinely ill-defined (0 a.e., δ at atoms). The
  current eps-trick returns `mass / 2**-d` (a huge artifact) only if you happen
  to probe exactly at `atom - 2**-d`; everywhere else it returns 0. In practice
  it is already effectively 0/garbage (see §4 golden values).

## 3. Options considered

### Option A — purpose-built frozen discrete RV (RECOMMENDED)

Introduce a tiny internal class — `_DiscreteRV` (or `FrozenDiscrete`) — that
holds sorted `(xs, ps)` plus cumulative sums and implements the scipy
frozen-distribution surface honestly with step-function semantics:

```python
class _DiscreteRV:
    """Frozen discrete distribution over arbitrary float support.

    Exposes the subset of the scipy frozen-RV interface that Severity
    consumes: cdf, sf, ppf, isf, pdf, support, stats, moment, rvs, mean, var.
    cdf/sf are exact right-continuous step functions; pdf is 0 (a discrete
    law has no density). This replaces the rv_histogram epsilon-jump hack.
    """
    def __init__(self, xs, ps):
        order = np.argsort(xs)
        self.xk = np.asarray(xs, float)[order]
        self.pk = np.asarray(ps, float)[order]
        self.cum = np.cumsum(self.pk)              # P(X <= xk_i)
    def cdf(self, x):  # right-continuous: P(X <= x)
        return self.cum[np.searchsorted(self.xk, x, side='right') - 1] ... # 0 below support
    def sf(self, x):   return 1.0 - self.cdf(x)
    def pdf(self, x):  return np.zeros_like(np.asarray(x, float))
    def ppf(self, q):  # smallest xk with cdf >= q
        ...
    def isf(self, q):  return self.ppf(1.0 - q)
    def support(self): return self.xk[0], self.xk[-1]
    def moment(self, n): return float(np.sum(self.xk**n * self.pk))
    def stats(self, moments='mv'): ...   # from xk, pk
    def rvs(self, size=None, random_state=None):
        return random_state... choice(self.xk, p=self.pk, size=size)
```

`_build` collapses to:

```python
self.fz = _DiscreteRV(xs, ps)
```

(no `max_log2`, no `xss`/`pss` interleaving, no `density=False`). `max_log2`
becomes unused — leave it for now or delete in a follow-up.

**Pros:** honest representation; exact `ppf/isf` (no `49.999999999`
artifacts); exact `support`; clean `pdf = 0`; no float-resolution tightrope;
`SeverityFixed` fixed for free. Edge-vectorization (`cdf` on an array of bucket
edges) is a one-liner via `searchsorted`, which is what `discretize` passes.

**Cons:** ~40 lines of new code to own and test. Must handle scalar **and**
array inputs (the decorators in `_apply_lb_ub` / `_apply_layer_attachment`
sometimes pass 0-d arrays — match the existing broadcasting carefully).

### Option B — `scipy.stats.rv_discrete` — REJECTED

`rv_discrete(values=(xk, pk))` is scipy's native discrete object, but **`xk`
must be integers**. Our atoms are arbitrary floats (`dsev [-2 5]`, `10 50 100
200 500`, fractional values). Mapping atoms → integer indices breaks `cdf(x)`
at real loss values and the affine `loc/scale` cannot encode irregular spacing.
This is almost certainly *why* the original author reached for `rv_histogram`.
Not viable.

### Option C — shrink/relocate the eps trick — REJECTED

Tuning `2**-d` or moving it into a helper is lipstick: still a continuous
object approximating a discrete one, still a float-resolution dependency.

## 4. Why this is "zero blast radius" — and how to prove it

There is already a regression oracle for exactly this:
`tests/capture_severity_golden.py` + `tests/data/severity_layer_golden.json`.
It probes `cdf/sf/pdf/ppf/isf` at fixed points for two `dhistogram` cases
(with layer+attachment, conditional True/False). Current captured values:

```
condTrue  cdf @50 = 0.42857...   sf @50 = 0.57142...   (exact, atom at 50)
          pdf @ all probes = 0.0
          ppf 0.01 = 49.999999999090406   ppf 0.5 = 149.9999999992239
          isf 0.25 = 149.99999999976717   ...
condFalse cdf @25 = 0.30  cdf @50 = 0.60  (exact)
          pdf @ all probes = 0.0   EXCEPT pdf @250 = Infinity (a layered-pdf
                                   delta-at-detachment artifact, not discrete-ness)
          ppf 0.5 = 49.999999999689564 ...
```

Reading these tells the whole story:

1. **`cdf`/`sf` already match a true step function exactly.** `cdf(50)=0.6`
   includes the atom at 50; `sf(50)=0.4` excludes it — precisely
   right-continuous semantics. Option A reproduces these bit-for-bit.
2. **`pdf` is already 0 at every probe.** A true discrete `pdf = 0` matches.
   (The lone `pdf@250 = Infinity` in condFalse comes from the
   `make_layer_attachment_pdf` delta at the detachment, *not* from the
   histogram, so it is unchanged by Option A.)
3. **`ppf`/`isf` are the only things that move**, and they move *toward
   correctness*: `49.999999999090406 → 50.0`, `149.9999999992239 → 150.0`.
   The deltas are ~1e-9 (the eps artifact disappearing).

So "zero blast radius" is achievable with one decision about the golden file:

- **Recommended:** re-capture the golden JSON after the change (the eps
  artifacts in `ppf/isf` become exact integers — a strict improvement worth
  locking in), OR
- assert against the existing file with `ATOL ≥ 1e-8` so the ~1e-9 `ppf/isf`
  shift passes without re-capture.

Either way, the user-visible numbers either stay identical (`cdf/sf/pdf`) or
get *more* correct (`ppf/isf`, `support`, `rvs`). No consumer can tell the
representation changed except by noticing the trailing-9s artifacts vanished.

### Verification checklist before merging
- [ ] Re-run / diff `tests/data/severity_layer_golden.json` (decide: re-capture vs ATOL bump).
- [ ] Full `uv run pytest` — especially every `dsev`/`fixed` line in `test_suite.agg`
      (each becomes a parse + shape-regression case) and the parquet density baselines
      in `tests/baseline/` (FFT path = `cdf/sf` at bucket edges → must be bit-stable).
- [ ] Spot-check a signed discrete severity (`dsev [-2 5]`): `support()` now returns
      exact `(-2, 5)`; confirm grid sizing at 5353-5354 and `support_description` unchanged.
- [ ] Confirm array/0-d-array inputs through the splice + layer decorators
      (`make_conditional_*`, `make_layer_attachment_*`) behave identically.
- [ ] `rvs` now returns exact atoms instead of `atom - U(0, 2**-d)`; grep tests/docs
      for any severity `.rvs(` that assumed jitter (none found in `tests/`; verify docs).

## 4a. The one path that is NOT zero-blast-radius: layered-discrete moments

The "zero blast radius" argument in §4 is airtight for the **density/FFT path**
(`cdf`/`sf` at bucket edges) and for the golden probe points — but it has a
blind spot. For a discrete severity **with a layer**, `Severity.moms()`
(~7535-7615) falls through paths 1 and 2 to path 3, `_numerical_moms`
(comment: *"plus any histogram with a layer"*), which integrates `fz.isf` over
the layer in probability space via `_safe_integrate` → `scipy.integrate.quad`
with `epsrel=1e-6`/`1e-4`, then **rejects the result as `NaN` if the estimated
relative error exceeds `max_rel_error = 1e-3`**.

- Today the eps-trick makes `isf` a *continuous* (piecewise-linear) function —
  quad integrates it cleanly, small error, moment accepted.
- A true `_DiscreteRV.isf` is a *step* function. quad on a discontinuous
  integrand inflates the error estimate and can trip the 1e-3 gate, flipping a
  previously-reported layered-discrete moment to `NaN` ("unreliable"), or
  shifting it by far more than the ~1e-9 §4 claims.

Crucially, **the golden file does not capture `moms()`** — only
`cdf/sf/pdf/ppf/isf` at fixed points — so this regression would slip past the
oracle §4 leans on. (The FFT density itself is unaffected; this is a
validation/`describe`/reporting-moment regression, not a density one.)

**Fix (do it as part of this change, not a follow-up).** A discrete law's
layered moments are *exact in closed form* — no integration needed:

```math
E[X(a,d)^n] = Σ_i  min(d - a, (x_i - a)_+)^n · p_i        (then /P(X>a) if conditional)
```

Two clean ways to wire it:

1. **Histogram-layered fast-path in `moms()`** — before falling to
   `_numerical_moms`, if `self._is_histogram` and a layer/attachment is present,
   sum the layer function over `(xk, pk)` directly (mirrors the
   attachment/detachment/conditional adjustments `_numerical_moms` already
   applies). Most localized; leaves `_numerical_moms` for the scipy zoo.
2. Give `_DiscreteRV` an exact `layer_moments(a, d, n)` helper and call it from
   that fast-path.

Either removes the only genuine risk and makes layered-discrete moments *more*
correct (exact vs quad-on-a-step), consistent with the rest of the change.

**Extend verification accordingly:** add the layered `dhistogram` cases'
`moms()` output to the golden capture (or a dedicated unit test asserting
`moms()` equals the summation formula), so the moments path is oracle-covered
too — not just the point probes.

### Surface `_DiscreteRV` actually has to implement (refined)
The hot, must-be-correct methods are `cdf`, `sf`, `ppf`, `isf`, `pdf`,
`support`. Note `fz.stats(...)` is **effectively dead for discrete**: in
`_numerical_moms` it is guarded by `not severity._is_histogram` (~6921, ~6995),
and the only other caller is the analytic path (lognorm/pareto/gamma/expon),
never a histogram — so `stats`/`moment`/`mean`/`var`/`rvs` on `_DiscreteRV` can
be minimal-but-honest (or omitted) without affecting any live path. Keep `rvs`
honest anyway (returns exact atoms) since it is cheap and removes the
`atom - U(0, 2**-d)` jitter surprise.

## 5. Recommendation

**Adopt Option A.** It is the only approach that represents a discrete law *as*
a discrete law, it deletes the `max_log2` float-resolution dependency, it fixes
`SeverityFixed` for free, and the existing golden harness plus the FFT density
baselines give a precise, mechanical proof of zero blast radius — with the one
honest caveat that `ppf/isf` get *more* accurate (re-capture the golden file to
record that as the new truth).

Scope to a single self-contained change: add `_DiscreteRV`, swap the two lines
in `SeverityDHistogram._build`, **add the histogram-layered moments fast-path
(§4a)**, re-capture goldens (including `moms()` for the layered cases), run the
suite. `SeverityMeta`'s
`b1size = 1e-7` zero-mass hack is the same smell and a natural **follow-up**
(it is a hybrid discrete-at-0 + continuous-elsewhere object, so it needs a
mixed representation, not plain `_DiscreteRV`) — keep it out of this change to
hold the blast radius tight.

## 6. Naming note
Per CLAUDE.md `Base<Kind>` convention this is a helper, not a `Severity`
subclass, so `_DiscreteRV` / `FrozenDiscrete` is fine (it is the `fz`, not the
`Severity`). Keep it private (leading underscore) and near the histogram
classes in `distributions.py`.

## 7. Close-out (do this when the change lands)
- **Bump the version.** Increment `pyproject.toml` to the next `1.0.0a*`
  (currently `1.0.0a25` → `1.0.0a26`). This is a representation change with a
  user-visible accuracy improvement (`ppf`/`isf`/`support` become exact, and
  layered-discrete moments become exact), so it earns a version bump.
- **README.rst bullet.** Add a release-note entry under the new version:
  `SeverityDHistogram`/`SeverityFixed` now use an honest frozen discrete RV
  (`_DiscreteRV`) instead of the `rv_histogram` epsilon-jump hack; exact
  `ppf`/`isf`/`support` and exact layered-discrete moments; `max_log2`
  dependency removed.
- **Re-capture the golden** (`tests/data/severity_layer_golden.json`) and note
  in the commit that the `ppf`/`isf` trailing-9s artifacts are now exact and
  that `moms()` for the layered cases is newly oracle-covered.
- **Docs:** keep any `.rst` mentioning the eps-trick / `max_log2` in lockstep;
  do **not** trigger a full docs build in the loop (CLAUDE.md).
- **Move this file to `dev/done/`** once merged and the suite is green, matching
  the convention used for `plan-multivariate.md` and `plan-note-parse.md`.
