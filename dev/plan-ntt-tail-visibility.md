# [NTT-Tail-Visibility] the exact tail is unreadable, and asking for it raises

Status: SPECIFIED and ruled by the author 2026-10-09, written against
1.2.0a5. Two phases, one bump each. Independent of the Rust engine work
(`dev/plan-ntt-rust-integration.md`) and of `dev/plan-ntt-crt.md`.

## The two problems, both found by measurement on 2026-10-09

### 1. `ntt_audit` raises on any large case

`accuracy_df` computes a relative error per bucket as

```python
rel_err = float((Fraction(float(p)) - q) / q)      # ntt.py:1286
```

where `q` is the exact probability and `p` the float FFT density. In the deep
tail `q` is astronomically smaller than the float pipeline's noise, so the
ratio exceeds float range and `float()` raises `OverflowError`. `ntt_audit`
reads `accuracy_df` (`ntt.py:1314`), so **the headline diagnostic property
raises too**. Measured threshold, uniform dice of dice:

| case | smallest nonzero exact | `ntt_audit` |
|---|---|---|
| 50 x 50 | ~1e-86 | ok, `max_rel_err` 2.9e68 |
| 100 x 100 | ~1e-202 | ok, `max_rel_err` 2.1e183 |
| 150 x 150 | ~1e-328 | **OverflowError** |
| 200 x 200 | ~1e-462 | **OverflowError** |

So every case from about 150 x 150 up is affected, which is the whole range
the exact engine exists to serve. It went unnoticed because the test corpus
is small cases.

**And the metric is meaningless even where it does not raise.** A
`max_rel_err` of 2.1e183 is the ratio of float noise to an exact probability
far below representability. It reports nothing about the float pipeline's
accuracy, which is what it exists to measure.

### 2. The tail is invisible, and `density_df` pays 78 s to hide it

`density_df` carries `p_exact`, one reduced `Fraction(numerator, denominator)`
per bucket. Each costs a gcd on a numerator of the denominator's width, about
10,000 bits on the headline case. Measured cost of first access to
`density_df`: **5.8 s at 500 x 500 and 77.7 s at 1000 x 1000.**

Meanwhile `p_exact_float` is `0.0` across the entire deep tail, because the
exact value underflows. So the user pays 78 seconds for Fractions and still
cannot see that a bucket holds 1e-462: the one column that could show it reads
zero, and the one that holds it is unreadable without exact arithmetic.

This matters more than it looks, because of where the Rust engine leaves
things. Of the 219.4 s a 1000 x 1000 `build()` costs today, 186.6 s is the
reconstruction. If the engine lands its target of about 20 s, `build()` falls
to roughly 27 s and `density_df`'s 78 s becomes **74 percent of the wall
clock**. The Fractions would dominate a compiled engine.

## Phase 1, [NTT-Accuracy-Overflow]

### The fix

A relative error is only meaningful where the float pipeline can represent the
exact answer at all. So:

1. **A safe conversion.** `_safe_float(fraction)` returns the float value, or
   `±inf` on `OverflowError`, preserving sign. Applied to `rel_err`, so
   `accuracy_df` never raises. The per-element `try` costs nothing on the
   buckets that do not overflow.
2. **`max_rel_err` gets a principled domain.** Compute it over buckets where
   the exact probability is representable as a nonzero float, that is where
   `p_exact_float > 0`. Those are exactly the buckets where asking "how close
   is the float pipeline" is a question with an answer. Everything below is
   reported by count instead.
3. **A new audit row, `underflow_buckets`:** the number of buckets whose exact
   probability is nonzero but underflows to zero in float. This is the honest
   statement of how much of the support lies beyond double precision, and it
   is information the current audit cannot express at all.
4. **`ntt_audit` stops building the Fractions twice.** It calls
   `_exact_on_grid()` for `exact_zero` (`ntt.py:1316`) in addition to the call
   inside `accuracy_df`. Exact zero is `numerator == 0`, which needs no
   Fraction, so this is a straight halving of the audit's cost.

### Acceptance

- `ntt_audit` and `accuracy_df` return for 150 x 150 and 200 x 200.
- `max_rel_err` is finite and plausible (order 1e-10 or smaller for a
  well-resolved case), not 1e183.
- `underflow_buckets` is 0 for 50 x 50 and large for 200 x 200.
- A regression test at a size that reproduced the raise.

## Phase 2, [NTT-Exact-Log10-Lazy]

### The log10 column, which is the interesting part

`math.log10` accepts an arbitrary-precision integer and returns a full
precision double, so

```python
log10_p = math.log10(numerator) - math.log10(denominator)
```

gives the base-10 magnitude of any coefficient, however small. On the bucket
whose exact probability is about 1e-462 it reads **-467.706**, where
`p_exact_float` reads `0.0`.

Measured: 200,000 of these cost **0.06 s** against 13.9 s for the equivalent
Fractions, **239x cheaper**, so a million cost about 0.3 s. It is the cheapest
column in the frame and the only one that can see the tail.

Add `p_exact_log10` to `density_df`: the exact probability's base-10
logarithm, `-inf` where the probability is exactly zero.

Accuracy, for the `Notes`: `math.log10` on an integer is computed from its
top 53 bits plus its bit length, so the result carries about 1e-13 relative
error in the logarithm, meaning roughly 1e-10 relative error in the implied
probability. That is far better resolution than the float density it
complements, and the exact `Fraction` remains available for anything needing
more.

### Making `p_exact` lazy

`density_df` stops carrying `p_exact` and keeps `p_exact_float` plus the new
`p_exact_log10`, both cheap. The Fractions move to a cached property:

```python
a.p_exact            # Series of reduced Fractions, built on first access
```

Same name, same content, same alignment, one attribute away instead of
inside a frame that is read for other reasons.

**This is a breaking change** to a documented column, stated plainly in the
CHANGELOG with its one-line migration, `a.density_df.p_exact` becomes
`a.p_exact`. It is acceptable because `AggregateNTT` arrived at 1.1.0 on
2026-10-09, two days before this change, and because the blast radius inside
the repository is five assertions in `tests/test_ntt.py` (lines 400 to 402,
445, 513) and three docstring references.

**Name vetted 2026-10-09.** `rg` across `src/aggregate` for `p_exact_log10`,
`_safe_float` and `underflow_buckets` returns nothing. `p_exact` as a property
does not collide: it currently exists only as a DataFrame column label and a
docstring string, never as an attribute or method on the class, so there is no
shadowing hazard of the kind `dev/done/plan-approximate.md` records.

### Acceptance

- `density_df` first access on the 1000 x 1000 case **under 5 s**, against
  77.7 s.
- `p_exact_log10` reads about -467.7 on a bucket whose `p_exact_float` is 0.0.
- `a.p_exact` returns the Fractions, summing to exactly 1, cached on second
  access.
- `density_df` no longer has a `p_exact` column, and the five assertions plus
  the docs are updated in the same commit.

## Files

`src/aggregate/ntt.py` (`_safe_float`, `_exact_on_grid`, `density_df`, the
new `p_exact` property, `accuracy_df`, `ntt_audit`, the module docstring and
the `AggregateNTT` class docstring), `tests/test_ntt.py`,
`docs/4_dec_Language_Reference.rst` if it names the column, `CHANGELOG.md`.

## Explicitly not in this plan

**Speeding up `Fraction` construction itself.** The cost is the gcd, and a
reduced `Fraction` requires it. There is a special case worth noting and not
taking: when the denominator factors into small primes, which it does whenever
the input probabilities are decimals, the reduction is a 2-adic and 5-adic
valuation rather than a general gcd and would be far cheaper. That is a
special case masquerading as an optimization; laziness solves the problem for
every denominator.

**Changing `p_exact_float`.** It is correctly rounded and underflows honestly.
`p_exact_log10` is the companion that makes the underflow legible, not a
replacement.
