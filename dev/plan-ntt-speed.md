# [NTT-Speed] geometric spectral runs, in-place kernels, and a numba engine

Status: RULED AND READY TO EXECUTE. Written 2026-10-09 against 1.1.0 (commit
88f9476); the three open decisions were ruled by the author the same day
(see Rulings below). Target release: 1.2.0 (performance plus one new engine
capability), staged 1.2.0a1 through 1.2.0a3 with 1.2.0 final at the last
stage, one commit per bump.

## Goal

Make large `nttagg` cases fast and make the machine actually work. The
1000 x 1000 uniform dice case (support 10^6, transform length 2^20, about
322 primes of a 9,977 bit CRT modulus) runs about an hour serial today and
about 25 minutes at `ntt_workers=8`, sitting near 7 percent CPU on DOOB's 32
logical cores when serial. Three stacked improvements, each its own stage and
bump: an algorithmic cut to the spectral walk for equal-weight outcome runs
(the uniform dice case collapses from about 1,000 modular array multiplies
per prime to about 70), in-place ufunc kernels that remove the temporaries
(measured 1.7x per operation), and an optional numba engine following the
`spectral.py` precedent that fuses the per-prime pipeline into compiled
nogil kernels with Montgomery arithmetic, which is the route past the memory
bandwidth wall that caps numpy thread scaling at about 2.5x.

## Current behavior and measurements (all 2026-10-09 on DOOB)

`aggregate.ntt.exact_compound` computes `H mod p` per prime: forward NTT of
the severity polynomial, a spectral walk evaluating
`H_hat_r = sum_i F_i * Ds^(Kmax - k_i) * a_hat_r^(k_i) (mod p)` by ascending
square-and-multiply over the frequency outcomes, inverse NTT, then an
incremental pure-Python bigint CRT in the calling thread. `workers=` (from
`build(..., ntt_workers=n)`) maps the per-prime pipeline over a
`ThreadPoolExecutor`; serial is the default.

- One `x * y % p` pass on a 2^20 uint64 array: 2.83 ms; the same with
  `np.multiply(..., out=)` and `np.mod(..., out=)`: 1.67 ms; the multiply
  alone: 1.09 ms. The `%` (integer division) is 62 percent of the cost.
- Dense 400 x 400: 164 s serial at 1.1.0a4, 119 s after the a5 walk fix,
  47 s at `workers=8` (2.5x, then flat: the ufuncs are memory bandwidth
  bound, so more threads do not help and processes would not either).
- Dense `[1:1000]` walk: about 1,000 modmul passes per prime, about 15
  minutes over 322 primes before counting the butterflies; CRT adds about
  4 to 5 minutes; exact numerators hold about 1.25 GB.

The per-prime pipeline was factored into `_residues_mod_prime` at 1.1.0a5,
which is the seam all three stages work behind. The decisive end-to-end
checks (coefficients sum to exactly `D`, nonnegative, zero beyond the
support) and the `exact_compound_poly` oracle are the correctness net; every
stage must leave them untouched and green.

## Rulings (author, 2026-10-09; these are decisions, not proposals)

1. **Default `ntt_workers`: ruled YES.** At the [NTT-Numba-Kernel] stage
   the default changes from serial to `min(8, os.cpu_count())`, because the
   serial default already surprised the author on the first big run;
   `ntt_workers=1` forces serial. Until that stage the default stays
   serial.
2. **Engine selection surface: ruled as proposed.** Auto-select the numba
   engine whenever numba imports (the `spectral.py` precedent is silent
   fallback the other way), with `ntt_engine='numpy'` / `'numba'` to force
   either, plumbed like `ntt_override` / `ntt_workers` through
   `AggregateNTT.update` to `exact_compound(engine=...)`. Name vetted
   2026-10-09: `rg -n "ntt_engine" src` has no hits; the bare word
   `engine` is avoided because it is P&L vocabulary (`pnl` engine,
   `_engine_port`).
3. **Version: ruled 1.2.0** (the numba engine is a capability, not a pure
   tidy), staged a1/a2/a3 with the final rollup at 1.2.0.

## Stage 1, [NTT-Uniform-Geometric] (1.2.0a1)

### Mathematics

Fold the severity denominator into the spectral point: with
`b = a_hat * Ds^(-1) (mod p)` (one vectorized Fermat inverse of the scalar
`Ds`), every term becomes `F_i * Ds^Kmax * b^(k_i)`, so the walk is a
weighted power sum of one array `b`. Group the sorted outcomes into maximal
runs of consecutive integers `u..v` carrying the same cleared weight `W`
(uniform `[1:F]` is a single run with `W = 1`). Each run contributes

    W * Ds^Kmax * b^u * (b^(v - u + 1) - 1) * (b - 1)^(-1)   (mod p)

by the geometric series, which costs `log2` squarings for `b^u` (bridged
from the previous run as today) plus `log2(run length)` for `b^len`, against
`run length` multiplies today. `(b - 1)^(-1)` is one vectorized binary
exponentiation to the power `p - 2` (about 48 array multiplies), computed
once per prime and shared by all runs, and only when some run is long enough
to want it (threshold: run length at least 8; shorter runs keep the existing
walk, so a fully irregular outcome set pays nothing new).

The `b == 1` cells need a mask, where the series is `len (mod p)` instead:
this is not an edge case, it is every case, because `a_hat_0 = A(1) = Ds`
makes `b_0 = 1` at frequency index 0 on every prime, so any test that checks
totals exercises the mask. `b == 0` cells fall out of the same formula
(`0^0 = 1` handles a run starting at outcome 0).

### Implementation

All inside `_residues_mod_prime` in `src/aggregate/ntt.py`: precompute the
run decomposition once in `exact_compound` (it is prime independent) and
pass it down; keep the existing walk as the short-run path. NumPy style
docstring Notes carry the algebra above. No surface change anywhere else.

### Tests (tests/test_ntt.py)

- The existing oracle, closed-form dice, and class tests already pin
  correctness and run the new path (uniform cases are runs).
- New: randomized piecewise-uniform weights (several runs of distinct
  lengths including singletons, gaps, a run starting at outcome 0) against
  `exact_compound_poly`; a case engineered so some `b_r == 1` beyond index
  0 if cheap to construct, otherwise index 0 suffices and says so.
- Measurement recorded in this plan's execution log: dense 400 x 400 and
  the 1000 x 1000 serial walk share, before and after.

Expected effect: the uniform 1000 x 1000 walk drops from about 1,000 to
about 70 modmul passes per prime, roughly 12x on the dominant phase; with
stage 2 and 8 workers the whole case should land in the 3 to 5 minute range.

## Stage 2, [NTT-Inplace-Kernels] (1.2.0a2)

Rewrite the hot array expressions with preallocated scratch and `out=`:
butterflies in `ntt` (scratch `t`: `np.multiply(v, wj, out=t)`,
`np.mod(t, p, out=t)`, then the add/subtract pair, watching that `u` and
`v` are disjoint views so in-place writes cannot alias), the twiddle table
build, the inverse scaling, and the power updates in the spectral walk and
geometric path. Answers are bit-identical (same integer arithmetic, fewer
temporaries); the oracle and closed-form tests are the pin, plus the
existing round-trip and convolution tests. Measured 1.7x on the isolated
modmul; expect 1.4x to 1.6x end to end on the transform phase and a higher
thread-scaling ceiling (less memory traffic per op). Record the measured
numbers in the execution log. No surface change, no new tests beyond a
benchmark note; this stage is deliberately boring.

## Stage 3, [NTT-Numba-Kernel] (1.2.0a3)

### Shape

Optional compiled engine, `spectral.py` precedent (try import numba, fall
back silently at INFO): but unlike `spectral.py`, the fallback is the
existing numpy implementation rather than a no-op decorator, because these
kernels are loop-style and would be abysmal as interpreted Python. The
numpy path therefore remains fully supported and tested forever; the suite
must pass with numba absent (`pytest.importorskip` on the numba-specific
tests). numba is already a declared optional extra, installed by
`uv sync --all-extras`.

### Kernels

One `@njit(cache=True, nogil=True)` function mirroring
`_residues_mod_prime`: forward NTT, spectral evaluation (geometric runs
included), inverse NTT, fused so the working set stays in cache across
phases. Arithmetic in Montgomery form with `R = 2^32`: operands below a
32 bit prime keep every REDC intermediate inside uint64 (`t = a * b <
2^64`, the textbook u64-safe REDC sequence), replacing the hardware
division that is 62 percent of the numpy op cost with multiplies and
shifts. Convert into Montgomery form on entry and out on exit, once per
array. Parallelism stays the existing `ThreadPoolExecutor` across primes
calling nogil kernels (no `prange` inside, so there is no nested
parallelism to tune and `ntt_workers` keeps one meaning). First-call JIT
compile is a few seconds, amortized by `cache=True`.

### Dispatch and surface

`exact_compound(..., engine=None)` where `None` means numba when
importable else numpy, with `'numpy'` / `'numba'` forcing (raise with a
clear message if numba is forced but absent). `AggregateNTT.update` pops
`ntt_engine` and forwards it; `meta` / `ntt_audit` gain an `engine` row so
a run records what produced it. Ruling 1 (workers default becomes
`min(8, os.cpu_count())`, with `ntt_workers=1` forcing serial) lands in
this stage.

### Tests

- Bit-identical residues: the numba and numpy engines must produce equal
  numerators and denominator on the oracle corpus and the 100 x 100 dice
  (exact integer equality, the strongest available check).
- Forced-engine raise when numba is unavailable (monkeypatch the import
  flag).
- The `ntt_audit` engine row.
- Anything above about 5 s marked `slow`.

Expected effect: removes the division bottleneck and the bandwidth-bound
temporaries at once; target at least 3x on the transform phase over stage 2
and visible multi-core scaling (the 7 percent CPU complaint is this stage's
acceptance smell test). Record measurements, including thread scaling at 4,
8, 16 workers, in the execution log.

## Stage 4, [NTT-Speed-Release] (1.2.0)

CHANGELOG rollup entry, `dev/TODO.md` tick, module docstring Notes refreshed
with the new engine story, `dev/FEATURES.csv` stamp, this plan moved to
`dev/done/`, author rebuilds docs if the `aggregate.ntt` page prose changed.
Acceptance: full gate `uv run pytest -m 'slow or not slow'` green, the
numerics gate (`-W error::RuntimeWarning`) green at this numerics-touching
release, and the headline measurement reproduced and recorded: 1000 x 1000
uniform dice end to end, serial and at the default workers, against the
1.1.0 baseline of about an hour.

## Files touched

- `src/aggregate/ntt.py`: all three stages (run decomposition, geometric
  path, scratch buffers, numba kernels, engine dispatch, audit row).
- `tests/test_ntt.py`: new oracle cases, engine-parity tests, markers.
- `pyproject.toml`, `uv.lock`, `CHANGELOG.md`, `dev/TODO.md`,
  `dev/FEATURES.csv` per release hygiene at each bump.
- Nothing in the grammar, parser, writer, or docs pages: the DecL surface
  is unchanged (`ntt_engine` is a build/update kwarg like `ntt_workers`).

## Names vetted (2026-10-09)

`ntt_engine` (update kwarg), `engine` (exact_compound kwarg and meta key,
module-local so the P&L `engine` vocabulary cannot collide), run-helper
`_outcome_runs`, kernel module-level `_residues_mod_prime_numba`: `rg` over
`src/aggregate` finds no collisions with existing callables or attributes
on `AggregateNTT`, its parents, or the module.

## Execution log

Executed 2026-10-09 by Claude against 1.1.0 (commit 88f9476). Review findings
and author rulings are recorded here as they were made.

### Review findings, 2026-10-09 (before any edit)

Every plan fact verified against the code: `_residues_mod_prime` is the seam
described, the butterfly block reads as quoted, `workers=` plumbs as described,
and the stage 1 algebra checks out, including that `a_hat_0 = Ds` forces
`b_0 = 1` on every prime. All four new names (`ntt_engine`, `engine`,
`_outcome_runs`, `_residues_mod_prime_numba`) are collision free; the P&L
`engine` vocabulary is confined to `_pnl.py`.

Four findings were raised, two absorbed and two put to the author.

1. **`Ds` need not be invertible mod `p`** (absorbed, implemented in stage 1).
   The geometric path needs `Ds**(-1)`, which does not exist when `p` divides
   `Ds`; the plan did not cover it, and the 1.1.0 walk is correct there, so the
   unguarded path would have been a correctness regression. Reaching it needs a
   severity denominator carrying a prime factor of the form `c * L + 1` just
   below `2**32`. `_spectral_walk_singular` carries those primes and
   `test_prime_dividing_severity_denominator_stays_exact` constructs one, with
   `Ds = 4294967291` at transform length 2.
2. **The plan's REDC sequence overflows** (recorded, stage 3 only). "Operands
   below a 32 bit prime keep every REDC intermediate inside uint64" is false of
   the naive sequence: with `T < p**2` and `m, p < 2**32`, `T + m * p` reaches
   almost `2**65`, and every prime here sits just under `2**32` because
   `ntt_primes` scans downward from it. The high-word formulation is exact and
   u64 safe: `u = (T >> 32) + ((m * p) >> 32) + (1 if (T & 0xffffffff) else 0)`,
   then one conditional subtract. Whoever builds stage 3 must use it.
3. **Stage 3's payoff is capped by Amdahl** (put to the author). By the plan's
   own numbers the headline case lands at 3 to 5 minutes after stages 1 and 2,
   of which the serial pure-Python CRT is 4 to 5 minutes, so compiling the
   transforms makes the CRT essentially the whole wall clock.
4. **The stage 4 headline measurement is expensive** (put to the author), and
   the 1.1.0 serial baseline of about an hour is already recorded here.

### Author rulings, 2026-10-09

5. **Stage 3 is held pending measurement.** Execute and commit stages 1 and 2,
   record the measurements, then stop and report so the numbers choose between
   `[NTT-Numba-Kernel]` and `[NTT-CRT-ProcessPool]`. This plan therefore stays
   open at `dev/` rather than retiring at 1.2.0; stage 4 is not executed.
6. **Ruling 1 moves to stage 2.** The `ntt_workers` default of
   `min(8, os.cpu_count())` does not depend on the numba engine, so it lands
   with the in-place kernels at 1.2.0a2 rather than waiting on stage 3.
7. **Measurement scope.** The cheap 400 x 400 serial and 8-worker pair at each
   stage, plus one 1000 x 1000 run at the default workers after stage 2. The
   second long serial run is not spent.
8. **`ntt_audit` gains both `engine` and `workers` rows.** `workers` lands in
   stage 2 (`meta` already carried it, unreported); `engine` waits for stage 3.

### Divergences from the plan as written

- `dev/TODO.md` had no `[NTT-Speed]` entry to tick, the only NTT entry being
  `[NTT-Exact-Aggregate]`, done at 1.1.0. A live entry was added under a new
  **In flight** heading at 1.2.0a1.
- The singular-prime guard and `_spectral_walk_singular` are new surface the
  plan did not name (finding 1).
- `Ds**Kmax` is factored out of the whole spectral sum to one closing multiply
  rather than carried per outcome, which the plan's algebra implies but does not
  state. The per-outcome `pow(ds, kmax - k, prime)` Python call disappears with
  it.
- The project `version-bump` skill still says "Recent history: `1.0.0a305`" and
  describes the version line as `1.0.0a*`. Stale since 1.1.0 shipped. Left for
  the author; ruling 3 of this plan governs the numbering used here.

### Stage 1 `[NTT-Uniform-Geometric]`, 1.2.0a1

Measurements on DOOB, 2026-10-09, against the plan's 1.1.0a5 baseline.

| case | before | after |
|---|---|---|
| 1000 x 1000 spectral walk, one prime, L = 2**20 | 8.62 s | 0.23 s (37.5x) |
| 1000 x 1000 walk projected over 321 primes | 46.1 min | 1.2 min |
| dense 400 x 400 end to end, serial | 119 s | 25.1 s (4.7x) |
| dense 400 x 400 end to end, 8 workers | 47 s | 17.2 s |

The 37.5x on the walk beats the plan's estimate of about 12x, because the
closed form also retires the per-outcome `pow(ds, kmax - k, prime)` scalar
work. Forward plus inverse NTT is 0.32 s per prime at `L = 2**20`, so the
1000 x 1000 transform phase is now about 3 minutes serial.

**This already confirms review finding 3.** At 400 x 400 the split is 12.7 s
transform against 12.4 s CRT serial, so the CRT is half the wall clock, and at
8 workers the case only reaches 17.2 s because the serial CRT floors it. The
8-worker run also shows the bandwidth wall the plan predicted: the per-prime
transform spans sum to 129.6 thread-seconds against 12.7 s serial, a 10x
inflation under contention.

## Known limits (unchanged, documented)

The 32 bit prime supply still caps the envelope near transform length 2^22;
Montgomery form does not change the modulus arithmetic, only its speed. The
CRT phase stays serial pure-Python bigints in this plan; process-pool
chunking ([NTT-CRT-ProcessPool] in the 1.1.0 plan's notes) remains future
work and becomes the dominant phase once the transforms are compiled, which
stage 3's measurements will quantify for a later decision.
