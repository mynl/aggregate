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
`Bivariate` all wrap the same thing: **a discrete distribution on a fixed grid** —
a probability vector `p` over a loss index `x`, plus bucket size `bs`. Every
downstream read — `q`, `var`, `tvar`, `tvar_threshold`, `cdf`, `sf`, `pmf`,
`percentiles`, `mean`, `snap` — is a pure function of `(x, p, bs)`.

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
    """A discrete distribution on a fixed grid: mass ``p`` over index ``x``,
    bucket ``bs``. Read-only; owns the lazy var/tvar cache."""
    def __init__(self, x, p, bs, name=''):
        self._x, self._p, self.bs, self.name = x, p, bs, name
        self._vt = None                      # lazy kernel cache, owned HERE

    @classmethod
    def from_series(cls, ser, bs, name=''):  # convenience for Series callers
        return cls(ser.index.values, ser.values, bs, name)

    def _funcs(self):
        if self._vt is None:
            self._vt = _make_var_tvar(pd.Series(self._p, index=self._x))
        return self._vt

    def q(self, p, kind='lower'): ...
    def var(self, p):  return self.q(p)
    def tvar(self, p, kind=''): ...
    def tvar_threshold(self, p, kind): ...
    def cdf(self, x): ...
    def sf(self, x): ...
    def pmf(self, x): ...
    def mean(self): ...
    def percentiles(self, pvalues=None): ...
    def snap(self, x): ...
```

**Invalidation becomes trivial:** there is nothing to reset — when the density
changes, the holder builds a *new* `GridDistribution`. The scattered
`self._var_tvar_function = None` lines disappear.

### 1.3 Open question — reconcile with `_DiscreteRV`

There is already a `_DiscreteRV(xs, ps)` in the severity code
(`distributions.py:9623`). **Before writing a new class, decide whether
`GridDistribution` should generalise/absorb `_DiscreteRV`** (add `bs` + the cached
var/tvar) rather than introduce a parallel discrete-grid abstraction. Resolve this
in Phase 1; do not ship two.

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

- **1.1** Resolve §1.3 (`_DiscreteRV` reconciliation): decide generalise-vs-new.
- **1.2** Create `aggregate/_grid_distribution.py`; move the `make_var_tvar` body
  in as `_make_var_tvar`; implement `GridDistribution` + `from_series`.
- **1.3** Drop `make_var_tvar` from `utilities.__all__`; relocate
  `var_tvar_test_suite` to target the new home.
- **1.4** New `tests/test_grid_distribution.py`: brute-force / analytic checks on
  the kernel and every accessor (small hand-checkable grids; fair-die style),
  plus `kind='lower'/'upper'` and `tvar_threshold` edge cases.

Phase 1 ships alone: a new internal primitive with no consumer change yet (the old
plumbing still works because the kernel is re-exported to its current callers from
the new module during the transition, or callers are swept in 1.3). Bump + changelog.

### Phases 2..n — adopt per consumer (behaviour-guarded; pure swaps)

Each is a standalone commit, `test_baseline.py` green (numbers identical):

- **2 — Severity.** Replace `_sev_var_tvar_function` plumbing; the discretised
  severity grid becomes a `GridDistribution`. (Smallest, lowest-risk consumer —
  prove the pattern here first.)
- **3 — Bounds.** Replace the hand-rolled `make_var_tvar(pd.Series(...))` at
  `bounds.py:753` (and `bounds.py:115`, `744`) with `GridDistribution`. Textbook
  adoption — it is already building the object by hand.
- **4 — Aggregate.** Replace `_var_tvar_function` *and* the second
  `_sev_var_tvar_function` copy; `q`/`var`/`tvar`/`tvar_threshold`/`cdf`/`sf`/`pmf`
  delegate to `self._dist`. Removes the `= None` resets in `__init__`/`update`.
- **5 — Portfolio.** Same swap; also unifies the *mutate-vs-return*
  `_make_var_tvar` divergence noted in §0.
- **6 — Bivariate.** Expose its marginal/total densities *as* `GridDistribution`
  so `q`/`tvar` on a bivariate marginal come for free.

Order = ascending blast radius (Severity → Bounds → Aggregate → Portfolio →
Bivariate), so the pattern is proven on small consumers before the god classes.

---

## 3. Payoff beyond dedup

- A **Portfolio unit** allocation → a `GridDistribution` ⇒ `unit.tvar(p)` for free.
- A **distorted/augmented** density → a `GridDistribution` ⇒ risk measures on the
  distorted measure for free.
- A **discretised Severity** grid → a `GridDistribution` ⇒ kills Aggregate's second
  copy.
- A **Bounds** envelope source → a `GridDistribution` ⇒ no hand-rolled kernel.

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
- **Name vetting (CLAUDE.md).** `rg` `GridDistribution` / `_dist` / `from_series`
  against the existing surface before fixing names; confirm no collision with an
  existing attribute/method on the adopting classes.
- **Decide `_DiscreteRV` fate in Phase 1** — never ship two discrete-grid types.

---

## 5. TODO / sequencing

- This is **P1**, first in the refactor track (`plan-README.md`). Phase 1 is
  independently shippable and valuable on its own.
- P3 (`plan-split-distributions.md`) and P4 (`plan-split-portfolio.md`) assume
  adoption is done, so their `_aggregate`/`_portfolio` already delegate to
  `GridDistribution` and are smaller when moved.
- Settles the old `plan-split` §7.3 "shared-accessor" question: the answer is **a
  held value type**, not a mixin and not bare free functions.
