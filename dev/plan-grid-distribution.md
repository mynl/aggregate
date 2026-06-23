# Plan P1 — `GridDistribution` value type (the shared discrete-grid distribution)

> **Status: DRAFT — not executed.** This is the **keystone** plan and goes first;
> see `plan-README.md`. Unlike P3/P4 it is a *pure addition* followed by guarded
> swaps — not a module split — so it stands on its own even if no splitting ever
> happens.
>
> **Release mechanics (CLAUDE.md).** Phase 1 (new primitive + tests) bumps
> `1.0.0a*` and adds a `CHANGELOG.md` section. Each adoption phase is
> behaviour-guarded by `test_baseline.py` (same kernel ⇒ identical numbers; any
> drift is a swap bug). `uv run pytest` green before every commit.

---

## 0. The idea

`Aggregate`, `Portfolio`, `Severity` (discretised grid), `Bounds`, and
`Bivariate` all wrap the same thing: **a discrete distribution on a grid** — a
probability vector `p` over a loss index `x`, plus an (optional) bucket size `bs`.

**Spacing is not assumed.** The probability accessors — `q`, `var`, `tvar`,
`tvar_threshold`, `cdf`, `sf`, `pmf`, `mean`, `lev` — are pure functions of
`(x, p)` **alone**: the `make_var_tvar` kernel builds them from `cumsum(p)` and the
index *values* via `searchsorted`, never dividing by a step width, so they are
already correct on a non-uniform index. Only the **width-dependent** ops — `pdf`
(= mass / width) and `snap` (snap to a regular grid) — need `bs`. So `bs` is
*optional metadata*, required only by the width family, not by the risk measures.
This matters concretely: a discrete severity's atoms (`dsev [1 10 100]`,
`_DiscreteRV`) and a Bounds/`_RiskSource` pmf from a Series are genuinely
non-uniform, and GridDistribution must model them honestly rather than pretend a
single `bs` exists.

Today the *kernel* `make_var_tvar(ser)` is already shared (in `utilities.py:364`),
but the **plumbing around it is duplicated and has drifted**:

- Each class carries its own `self._var_tvar_function` cache attribute, reset to
  `None` in `__init__` and again after `update` (4+ reset sites across two files).
- Each has its own `_make_var_tvar` wrapper, and they **don't match**:
  `Aggregate._make_var_tvar` *returns* the dict (`self._var_tvar_function =
  self._make_var_tvar(ser)`); `Portfolio._make_var_tvar` *mutates*
  `self._var_tvar_function = {}` then fills it.
- `Aggregate` has the pattern **twice**: `_var_tvar_function` (aggregate grid) and
  `_sev_var_tvar_function` (severity grid).
- `Bounds` constructs `make_var_tvar(pd.Series(self._prob, index=self._x))` by hand
  (`bounds.py:753`).
- The public `q`/`var`/`tvar`/`tvar_threshold`/`cdf`/`sf`/`pmf` methods are
  re-implemented on each class on top of that.

`GridDistribution` is a small **read-only value type** holding `(x, p, bs)` that
owns the lazy cache and all the accessors. It depends only on numpy/pandas — a
true **leaf** — so it can land before any split, and adopting it *removes* the
duplicated plumbing, shrinking every consumer before P3/P4 move them.

---

## 1. Design

### 1.1 The kernel moves in

`make_var_tvar`'s body (the interpolation that builds `q_lower`/`q_upper`/`tvar`)
**relocates from `utilities.py` into the new grid-distribution module**, kept as a
module-level pure function (isolated brute-force testing) that `GridDistribution`
wraps. Rationale: once all consumers go through `GridDistribution`, `utilities`
has zero remaining callers of it; the kernel belongs with the type that owns
var/tvar, and this makes `GridDistribution` self-contained (no reach into the
`utilities` grab-bag). **This is a public-surface move** — `make_var_tvar` is in
`utilities.__all__`; sweep the callers (all become adopters here), the
`var_tvar_test_suite`, and docs. Per the no-deprecated-aliases-pre-1.0 rule, move
cleanly, no shim.

### 1.2 Sketch (names provisional — `rg`-vet before fixing)

```python
# aggregate/_grid_distribution.py  — leaf: numpy/pandas only
class GridDistribution:
    """A discrete distribution on a grid: mass ``p`` over index ``x``, with an
    optional bucket ``bs``. Read-only; owns the lazy var/tvar cache.

    The probability accessors make NO equal-spacing assumption. ``bs`` is
    needed only by the width-dependent ops (``pdf``, ``snap``); leave it
    ``None`` for a genuinely non-uniform grid and those raise a clear error."""
    def __init__(self, x, p, bs=None, name=''):
        self._x, self._p, self.bs, self.name = x, p, bs, name
        self._vt = None                      # lazy kernel cache, owned HERE

    @classmethod
    def from_series(cls, ser, bs=None, name=''):  # convenience for Series callers
        return cls(ser.index.values, ser.values, bs, name)

    def _funcs(self):
        if self._vt is None:
            self._vt = _make_var_tvar(pd.Series(self._p, index=self._x))
        return self._vt

    # --- probability accessors: pure functions of (x, p); spacing-agnostic ---
    def q(self, p, kind='lower'): ...
    def var(self, p):  return self.q(p)
    def tvar(self, p, kind=''): ...
    def tvar_threshold(self, p, kind): ...
    def cdf(self, x): ...
    def sf(self, x): ...                 # the S vector calibration consumes
    def pmf(self, x): ...
    def mean(self): ...
    def lev(self, a): ...                # E[min(X, a)] = ∫₀ᵃ S dx (limited EV)
    def limited_tvar(self, p, a): ...    # TVaR_p(min(X, a)); analytic composite of
                                         #   tvar/cdf — O(1), no grid rebuild (Bounds)
    # --- width-dependent: require bs (raise if bs is None) ---
    def pdf(self, x): ...                # mass / width — a continuous reading
    def snap(self, x): ...
    # --- optional transform ---
    def cap(self, a): ...                # new GridDistribution for min(X, a)
```

**`q` already vectorises** (the kernel's `searchsorted` takes an array of `p`), so
`gd.q([0.9, 0.99, 0.999])` gives the exact step-function quantiles at a vector of
levels. That is why there is **no `percentiles` method**: `Portfolio.percentiles`
(the only one in the library, and an *interpolated* table) is deprecated and being
dropped (see P4); resurrecting it as a blessed primitive would re-introduce a
deprecated, semantically-different accessor. Vector-`q` supersedes it.

**`lev(a)` is new and load-bearing** — it is the limited-expected-value datum a
`Distortion` needs to calibrate (see Phase L below). Calibration does **not** move
onto the value type: the dependency runs **GD → Distortion** (the caller passes its
GD to the `Distortion`, which already owns the Newton iteration), so GD stays a leaf
and never imports `Distortion`. `lev` is a width-aware integral
(`bs · Σ_{x<a} S`), so it lives with the probability accessors but degrades to
local `np.diff(x)` widths when `bs is None`.

**`limited_tvar(p, a)` is new and load-bearing for Bounds** — it is `TVaR_p(min(X,
a))`, the **analytic composite** of GD quantities that `Bounds._tvar_x_a`
(`bounds.py:230`) already computes today: `TVaR_p(X) − (1−F(a))(TVaR_{F(a)}(X) −
a)/(1−p)` for `p < F(a)`, else `a`. It reads `tvar`/`cdf` only and is **O(1) — no
grid rebuild** — so it is safe inside Bounds' `p_star` root-find loop. Because the
TVaR of `min(X, a)` is purely a function of the distribution and `(p, a)`, **its
home is `GridDistribution`**; the scalar `a` is an *argument* (a pricing input, not
stored on the GD, which stays cap-agnostic), and `Bounds` delegates to it (Phase 3).

**`cap(a)` is the general transform** (kept, but distinct): `min(X, a)` is itself a
grid distribution (pool all mass ≥ a onto the atom at `a`), so `gd.cap(a)` returns a
fresh `GridDistribution` and *any* accessor on the capped law follows —
`gd.cap(a).tvar(p)` ≡ `gd.limited_tvar(p, a)`, `gd.cap(a).var(p)`, etc. `cap`
rebuilds the grid, so it is the convenience path for one-off capped views, **not**
the root-find hot loop, which uses the analytic `limited_tvar`. (The two share a
test: `cap(a).tvar(p)` cross-checks `limited_tvar(p, a)`.)

**Invalidation becomes trivial:** there is nothing to reset — when the density
changes, the holder builds a *new* `GridDistribution`. The scattered
`self._var_tvar_function = None` lines disappear.

### 1.3 Resolved — relationship to `_DiscreteRV` (share the core, keep two faces)

There is already a `_DiscreteRV(xs, ps)` in the severity code
(`distributions.py:9623`), assigned to `Severity.fz` (9812) **interchangeably with
frozen `scipy.stats` RVs and `ss.rv_histogram`**, with `Severity.moms` calling
`self.fz.layer_moments(...)` (9424).

**Decision: do not merge them into one class; share the cumulative-step kernel and
keep two thin faces.** The genuine overlap is the cumulative core — sorted `x`,
`p`, `cumsum` → `cdf`, `sf`, lower-quantile, `mean` (≈30 lines, identical in
spirit). The honest reasons *not* to force a single class:

1. **`pdf` is contradictory.** `_DiscreteRV.pdf ≡ 0` (a genuine discrete law has no
   density); `GridDistribution.pdf = p / bs` (the grid is a *discretisation of a
   continuous* density). Same arrays, opposite meaning of "what is between the grid
   points." One class cannot have one honest `pdf`. **Resolution: keep `pdf` off the
   shared core** — each face interprets density itself.
2. **`_DiscreteRV` exists to *be* a scipy frozen RV.** Severity dispatches
   polymorphically over `self.fz`, so `_DiscreteRV` must carry scipy names
   (`ppf`/`isf`/`stats`/`moment`/`rvs`/`support`) **and** the `layer_moments`
   extension. Merging would force GD to grow a scipy-compat facade + layered-moment
   machinery it has no other use for, polluting the clean actuarial surface.
3. **Disjoint extras.** GD needs `tvar` + `bs`-density + `snap` + `lev`;
   `_DiscreteRV` needs raw `moment(n)` + `layer_moments` + `rvs`. Neither wants the
   other's load.

So: **`GridDistribution` is the spacing-agnostic cumulative core** (the single
source of truth for the step-function kernel — satisfying "never two discrete-grid
types" in spirit); **`_DiscreteRV` becomes a thin scipy-naming adapter** that holds
(or delegates to) a `GridDistribution` for the cumulative parts and adds only the
scipy/severity-specific bits (`ppf`/`isf` aliases, `pdf≡0`, `moment`/`stats`/`rvs`,
`layer_moments`). Item §0's spacing-agnosticism is what makes this work: a
non-uniform `_DiscreteRV` *is* a valid `GridDistribution` underneath.

> **Distinction the rest of this plan must keep straight.** `Severity` holds *two*
> different grid objects: (a) the **input** severity RV `self.fz` — possibly
> `_DiscreteRV`, non-uniform atoms — which this §1.3 is about; and (b) the
> **discretised output** PMF on the `bs`-grid (uniform) fed to the FFT, which
> Phase 2 below is about. They are separate adoptions; do not conflate them.

### 1.4 Adoption shape (per consumer)

Composition (HAS-A), not inheritance — keeps the cache in one place and adds **no**
inheritance edge between the big classes:

```python
# end of update():
self._dist = GridDistribution(density_df.index, density_df['p_total'].values, bs)
# public methods delegate:
def tvar(self, p, kind=''):  return self._dist.tvar(p, kind)
```

`self._dist` is internal; `agg.tvar(p)` stays the blessed surface (no "two ways").

---

## 2. Phased execution

### Phase 1 — build the primitive (pure addition, bumps version)

- **1.1** Create `aggregate/_grid_distribution.py`; move the `make_var_tvar` body
  in as `_make_var_tvar`; implement `GridDistribution` (spacing-agnostic
  accessors + `lev` + `limited_tvar`; `bs`-guarded `pdf`/`snap`; optional `cap`) +
  `from_series`.
- **1.2** Drop `make_var_tvar` from `utilities.__all__`; relocate
  `var_tvar_test_suite` to target the new home.
- **1.3** New `tests/test_grid_distribution.py`: brute-force / analytic checks on
  the kernel and every accessor (small hand-checkable grids; fair-die style),
  plus `kind='lower'/'upper'`, `tvar_threshold`, `lev` against `∫S`,
  `limited_tvar(p, a)` against a brute-force `TVaR_p(min(X, a))` (and against
  `cap(a).tvar(p)`, and against `Bounds._tvar_x_a` on a shared case), and an
  explicit **non-uniform-grid** case (asserting the probability accessors match a
  direct cumulative computation while `pdf`/`snap` raise when `bs is None`).
- **1.4** (resolution of §1.3, *can land in this phase or as Phase 2's first
  step*) Extract the shared cumulative core; reshape `_DiscreteRV` into the thin
  scipy-naming adapter over a held `GridDistribution`, keeping
  `pdf≡0`/`moment`/`stats`/`rvs`/`layer_moments`. `test_discrete_severity.py`
  must stay green (same numbers — pure internal re-plumbing).

Phase 1 ships alone: a new internal primitive with no consumer change yet (the old
plumbing still works because the kernel is re-exported to its current callers from
the new module during the transition, or callers are swept in 1.3). Bump + changelog.

### Phases 2..n — adopt per consumer (behaviour-guarded; pure swaps)

Each is a standalone commit, `test_baseline.py` green (numbers identical):

- **2 — Severity.** Replace `_sev_var_tvar_function` plumbing; the discretised
  **output** severity grid (the uniform `bs`-grid PMF fed to the FFT — face (b) in
  §1.3, *not* `self.fz`) becomes a `GridDistribution`. (Smallest, lowest-risk
  consumer — prove the pattern here first.)
- **3 — Bounds.** Replace the hand-rolled `make_var_tvar(pd.Series(...))` at
  `bounds.py:753` (and `bounds.py:115`, `744`) with `GridDistribution`. Textbook
  adoption — it is already building the object by hand. **Capped TVaR moves into
  GD.** The capped TVaR `TVaR_p(min(X, a))` (`Bounds._tvar_x_a`, `bounds.py:230`) is
  a pure function of the distribution and `(p, a)` — the analytic composite of
  `tvar(p)`, `tvar(F(a))`, `cdf(a)`, and the scalar `a` — so it **becomes
  `GridDistribution.limited_tvar(p, a)`**, and `Bounds._tvar_x_a` delegates to it,
  passing its asset cap `a`. The scalar `a` is an argument *to* the accessor, not
  stored on the GD (the GD stays cap-agnostic); only the formula moves. This keeps
  the O(1) cost — no `cap()` rebuild — inside Bounds' `p_star` root-find while giving
  the formula a single home. `_resolve_obj` (which today hands Bounds the `tvar`/`F`
  callables, `bounds.py:212`) hands it a `GridDistribution` instead, and `Fb`,
  `tvar_x_p`, `p_star` all read through it.
- **4 — Aggregate.** Replace `_var_tvar_function` *and* the second
  `_sev_var_tvar_function` copy; `q`/`var`/`tvar`/`tvar_threshold`/`cdf`/`sf`/`pmf`
  delegate to `self._dist`. Removes the `= None` resets in `__init__`/`update`.
- **5 — Portfolio.** Same swap; also unifies the *mutate-vs-return*
  `_make_var_tvar` divergence noted in §0.
- **6 — Bivariate.** Expose its marginal/total densities *as* `GridDistribution`
  so `q`/`tvar` on a bivariate marginal come for free.

Order = ascending blast radius (Severity → Bounds → Aggregate → Portfolio →
Bivariate), so the pattern is proven on small consumers before the god classes.

### Phase L — feed distortion calibration from a `GridDistribution` (new capability; bumps version)

This is a *payoff* phase, not a guarded swap: it adds capability, so it bumps the
version and gets a `CHANGELOG.md` entry (and is **not** numerics-frozen in the
swap sense — though the calibrated shapes must match the old path bit-for-bit).

**Dependency direction.** Calibration already lives on `Distortion`
(`spectral.py`) — `Distortion.calibrate` owns the Newton iteration. This phase does
**not** move calibration onto GD; it makes the *data* `Distortion.calibrate`
consumes come from a `GridDistribution`. The caller hands its GD (or the
GD-extracted `S`/`lev`/`bs`) to the `Distortion`; **GD never imports `Distortion`**
(leaf preserved).

**The finding.** `Portfolio.calibrate_distortions` (plural) is the only public
entry the author uses; it computes the asset level `a`, the premium target `P`
(from `coc`), then loops `['ccoc','ph','wang','dual','tvar']` calling the
**singular** `calibrate_distortion` once per name — and the singular *re-resolves
the whole S-vector / `ess_sup` / `el` from `density_df` on every call* even though
`a` and `P` are already fixed. Its entire data dependency is grid-distribution
data: `a = q(p)`, `S = sf` over `[0, a]`, `el = lev(a)`, `bs`, `ess_sup` (first
zero of `S`) — none of it Portfolio-specific *for the total* (per-unit `exa_*`
conditional columns from `add_exa` are genuine Portfolio machinery and stay).

**The change.**
- **Drop the public singular `Portfolio.calibrate_distortion`.** The author never
  calls it; it is at best a private worker. Inline its dispatch (the per-family
  `init_shape` / `param_name` construction → `subclass.calibrate(...)`) into the
  plural's loop, resolving the GD data **once** before the loop instead of per
  name. This also lets the singular's unused `S_column`/`S_calc`/`r0`/`kind`
  option surface (the plural always used defaults) be dropped.
- **Source the calibration data from a `GridDistribution`.** The set-calibration
  helper that loops `['ccoc','ph','wang','dual','tvar']` (resolving `init_shape`/
  `param_name` and calling `Distortion.calibrate`) takes a GD + `(premium_target,
  assets)` and extracts `S`/`el = lev(a)`/`bs`/`ess_sup` from the GD. `Portfolio`
  resolves `a`/`P` from its pentagon state and hands its **total** GD to that helper.
  **Home:** this helper is **flavor (b) of the shared `_pricing` concern** (see
  `plan-README.md` and P3). P1 authors it next to `Distortion` in `spectral.py`
  (where the Newton math it drives already lives); **P3 relocates it into
  `_pricing.py`** when it births that module — the one accepted move-twice, called
  out so it is not a surprise. The per-distortion Newton iteration stays on
  `Distortion` either way; only the set-loop *orchestration* migrates.
- **Free win: `Aggregate.calibrate_distortions`.** Because the helper needs only a
  `GridDistribution` + a premium target, an `Aggregate` (or a unit, or a distorted
  density) can calibrate directly — **no more wrapping in a 1-unit Portfolio.**
  Expose `Aggregate.calibrate_distortions` handing its own GD to the same helper.
  (Calibration to the **total** only; per-unit allocation stays a Portfolio
  concern in `_portfolio`.)

**Tests.** `tests/test_distortion_calibrate.py` currently targets the singular —
re-point it at the plural / the GD worker, pinning identical calibrated shapes
(the Newton iterations are unchanged, only their inputs are sourced differently).

---

## 3. Payoff beyond dedup

- A **Portfolio unit** allocation → a `GridDistribution` ⇒ `unit.tvar(p)` for free.
- A **distorted/augmented** density → a `GridDistribution` ⇒ risk measures on the
  distorted measure for free.
- A **discretised Severity** grid → a `GridDistribution` ⇒ kills Aggregate's second
  copy.
- A **Bounds** envelope source → a `GridDistribution` ⇒ no hand-rolled kernel, and
  the capped TVaR `TVaR_p(min(X, a))` is `gd.limited_tvar(p, a)` (Bounds delegates;
  the formula lives in one place).
- **Distortion calibration on anything** ⇒ `Aggregate.calibrate_distortions`
  without the 1-unit-Portfolio wrap (Phase L), because calibration is a function of
  `(S, lev, bs, premium_target)` — all `GridDistribution` data.

The same small type expresses "an Aggregate as a distribution," "a Portfolio
total," "a unit allocation," "a distorted density," "a discretised severity" —
which is how these are already reasoned about.

---

## 4. Guardrails

- **Leaf discipline.** `_grid_distribution.py` imports only numpy/pandas. If it
  ever needs `distributions`/`portfolio`, the design is wrong — stop.
- **Behaviour-guarded swaps.** Same kernel ⇒ identical numbers; `test_baseline.py`
  must not regenerate on any adoption commit. A diff means the swap is wrong.
- **No "two ways."** `self._dist` is private; the class methods stay the public API
  and merely delegate.
- **Name vetting (CLAUDE.md).** `rg` `GridDistribution` / `_dist` / `from_series` /
  `lev` / `limited_tvar` / `cap` against the existing surface before fixing names;
  confirm no collision with an existing attribute/method on the adopting classes.
  (`lev` is an actuarial term of art — limited expected value — check it is free;
  `limited_tvar` is provisional, chosen to parallel `lev` for the *limited* TVaR —
  confirm the name with the author before fixing it.)
- **`_DiscreteRV` resolved (§1.3):** one cumulative core, two faces — never a
  forced merge (the `pdf` semantics conflict) and never two parallel kernels.
- **Spacing-agnostic invariant:** probability accessors must never divide by `bs`;
  only `pdf`/`snap` (and `lev`'s width step) touch spacing, and they degrade to
  `np.diff(x)` or raise when `bs is None`. A `bs`-divide creeping into `tvar`/`q`
  is a bug.

---

## 5. TODO / sequencing

- This is **P1**, first in the refactor track (`plan-README.md`). Phase 1 is
  independently shippable and valuable on its own.
- P3 (`plan-split-distributions.md`) and P4 (`plan-split-portfolio.md`) assume
  adoption is done, so their `_aggregate`/`_portfolio` already delegate to
  `GridDistribution` and are smaller when moved.
- Settles the old `plan-split` §7.3 "shared-accessor" question: the answer is **a
  held value type**, not a mixin and not bare free functions.
