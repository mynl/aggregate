# Plan: `price_pentagon` — price an Aggregate/Portfolio to a target via the Pentagon

## Goal

One easy method on `Aggregate` and `Portfolio`: fix a capital level (`a` or `p`),
give **one** pricing target (premium `P`, cost of capital `roe`/CoC, or loss
ratio `lr` — also `M`, `Q`, `pq`), and have the Pentagon completer fill the full
eight-stat octet. Generalizes the existing, uncalled `Portfolio.price_ccoc`
(which is the `{coc}`-only special case) and exposes the already-built
`Pentagon.solve_obj` as object surface.

This is a thin wrapper — **no new pricing math.** `Pentagon.solve` already
solves every soluble triple; `solve_obj` already reads `L` and `a` off the
object. We add `a=` support and a friendly entry point.

## Steps

### 1. `pentagon.py` — let `solve_obj` take `a` as well as `p`
`solve_obj(self, p, *, …)` currently only reads `a = obj.q(p)`. Add an
alternative asset-level path:

```python
def solve_obj(self, *, p=None, a=None, P=None, M=None, Q=None,
              lr=None, pq=None, roe=None):
    # exactly one of p / a
    if (p is None) == (a is None):
        raise ValueError('pass exactly one of p= or a=')
    a = self.obj.q(p) if a is None else self.obj.snap(a)
    L = ... exa / exa_total at a ...           # (existing branch, unchanged)
    return self.solve(L=L, a=a, P=P, M=M, Q=Q, lr=lr, pq=pq, roe=roe)
```

`{L, a, <one target>}` is soluble for every target (premium/coc/lr/M/Q/pq), so
all headline cases resolve. *(Note the signature change: `p` becomes
keyword-only. `solve_obj` is internal — grep shows no other callers — so this is
safe.)*

### 2. `Aggregate` + `Portfolio` — add `price_pentagon`
Identical tiny method on both (each delegates to `Pentagon`, which already
abstracts the `exa`/`exa_total` difference):

```python
def price_pentagon(self, *, p=None, a=None, P=None, M=None, Q=None,
                   lr=None, pq=None, roe=None):
    """Complete the pricing octet at capital level a (or p) given one target.

    Fix the capital level with exactly one of ``p`` (VaR probability) or ``a``
    (asset level), then supply exactly one pricing target — premium ``P``, cost
    of capital ``roe`` (a.k.a. CoC), loss ratio ``lr`` (also ``M``/``Q``/``pq``).
    Returns the canonical one-row ('total') pentagon DataFrame (columns
    ``PENTAGON_STATS``). No distortion involved — pure accounting completion
    against the object's expected loss at ``a``.
    """
    pent = Pentagon(obj=self)
    pent.solve_obj(p=p, a=a, P=P, M=M, Q=Q, lr=lr, pq=pq, roe=roe)
    return pent.as_frame(line='total')   # or complete_pentagon-style octet
```

Validation: require exactly one capital input and exactly one target (clear
`ValueError` otherwise — mirror the `analyze_distortion` p/a guard wording).

### 3. Fold in `price_ccoc`
`price_ccoc(ccoc, *, p)` is exactly `price_pentagon(p=p, roe=ccoc)`. It's
uncalled and alpha → reimplement it as a one-line delegate (keep the name for
the convenience reading), or drop it. Recommend: keep as a thin alias.

## Verification

1. **`uv run pytest`** — existing suite green.
2. **New tests** (`tests/test_pentagon.py` or extend existing): for a small
   Aggregate and a small Portfolio, at a fixed `a`, check the three headline
   entry points agree with hand arithmetic and round-trip:
   - `price_pentagon(a=a, roe=0.1)` reproduces the old `price_ccoc(0.1, p=…)`
     numbers (premium `(L + roe*a)/(1+roe)`).
   - `price_pentagon(a=a, P=prem)` recovers that same `roe` (`M/Q`).
   - `price_pentagon(a=a, lr=lr0)` gives `P = L/lr0`.
   - `p=` and the equivalent `a=self.q(p)` give identical octets.
3. Quick smoke in a notebook: `qd(a.price_pentagon(p=0.99, roe=0.1))`.

## Housekeeping (standing rules)

- Bump `pyproject.toml` `1.0.0a*`.
- `CHANGELOG.md`: "Added `Aggregate.price_pentagon` / `Portfolio.price_pentagon`
  — complete the pricing octet from a capital level (`a`/`p`) plus one target
  (premium / CoC / loss ratio) via the Pentagon. Generalizes `price_ccoc`;
  `Pentagon.solve_obj` now accepts `a=` as well as `p=`."
- `dev/TODO.md`: one-line under the appropriate track, marked done with version.
- Move this plan to `dev/done/plan-price-pentagon.md` on landing.

## Name (settled)

- **`price_pentagon`** on both classes — chosen to push the Pentagon concept as
  the accounting vocabulary. (`.price` stays the distortion-pricing method.)

## Out of scope

- No change to the distortion-based `.price` / `analyze_distortion(s)` family.
- No new identities in `Pentagon.solve` (already complete).
- `occ`/`agg`-style naming and the reins rename are unrelated.
