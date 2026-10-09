# [NTT-CRT] the CRT reconstruction is now the whole cost, and it is quadratic

Status: PROPOSED 2026-10-09, written against 1.2.0a2 (commit f40f26c). Nothing
implemented. Supersedes the `[NTT-CRT-ProcessPool]` note carried in
`dev/done/plan-nttagg.md`, and replaces the dropped `[NTT-Numba-Kernel]` stage
of `dev/done/plan-ntt-speed.md`. **Two open questions for the author at the
foot of this file.**

## Goal

After `[NTT-Uniform-Geometric]` and `[NTT-Inplace-Kernels]` the transform phase
of an exact `nttagg` computation has essentially disappeared and the Chinese
remainder reconstruction is the whole wall clock: **589.6 s of the 594.0 s** of
the 1000 x 1000 uniform dice case, 99.3 percent. Make that phase fast. The
first stage is algorithmic and serial, and may well be enough on its own; the
second reaches for processes only if it is not.

## Current behavior

`exact_compound` folds one prime at a time into a running reconstruction, in
the calling thread, in pure-Python bigints:

```python
reconstructed, modulus = None, 1
for (prime, _root), (head, seconds) in zip(primes, results):
    if reconstructed is None:
        reconstructed = [int(r) for r in head]
        modulus = prime
    else:
        inverse_mod = pow(modulus % prime, prime - 2, prime)
        reconstructed = [
            current + modulus *
            ((int(r) - current % prime) * inverse_mod % prime)
            for current, r in zip(reconstructed, head)]
        modulus *= prime
```

Two properties matter. The fold is **elementwise across coefficients**:
coefficient `j` depends only on the residues of coefficient `j` over the
primes, plus a scalar sequence shared by all coefficients. And each step
multiplies by `modulus`, which **grows with every prime**, so prime `i` costs
about `i` and the phase is quadratic in the prime count.

## Measurements (2026-10-09 on DOOB, at 1.2.0a2)

Per-prime transform and CRT spans recorded separately, 500 x 500 uniform
(141 primes, `L = 2**18`), the scratch harness in this plan's own history:

| prime block | transform, serial | CRT, serial | transform, 8 workers | CRT, 8 workers | modulus bits |
|---|---|---|---|---|---|
| 1..28 | 0.074 s | 0.084 s | 0.647 s | 0.089 s | 896 |
| 29..56 | 0.076 s | 0.164 s | 1.327 s | 0.161 s | 1,792 |
| 57..84 | 0.074 s | 0.230 s | 1.838 s | 0.233 s | 2,688 |
| 85..112 | 0.073 s | 0.291 s | 2.434 s | 0.301 s | 3,584 |
| 113..140 | 0.074 s | 0.397 s | 3.135 s | 0.395 s | 4,480 |
| first to last | 0.8x | **39.6x** | **28.1x** | 26.8x | |

Serial totals: transform 10.5 s, CRT 33.1 s, wall 43.6 s. At 8 workers:
transform 266.2 thread-seconds, CRT 33.4 s, wall 35.2 s.

Three readings, and they are the whole basis of this plan.

1. **The CRT is quadratic, and that is where the time goes.** Its per-prime
   span grows 39.6x over the run, tracking the modulus bits exactly. The 141
   steps total 33.1 s where a flat first-step cost would be 1.4 s. On the
   1000 x 1000 case the same curve integrates to 589.6 s.
2. **The transform arithmetic does not slow down at all.** Serial spans are
   flat, 0.8x first to last. Nothing in the transform needs attention.
3. **The threads are starved by the GIL, not by the work.** Their spans ramp
   28.1x in lockstep with the CRT ramp, on constant-size arrays. The uint64
   ufuncs release the GIL, but a worker needs it back between ufunc calls, and
   the main thread is holding it for 0.4 s at a stretch doing bigint
   arithmetic. A constant contention component exists too (block 1 is 8.7x
   serial), but the ramp is the CRT. **This is the user-visible symptom that
   opened this plan:** per-prime log lines that start at 0.2 s and reach 4.6 s.

So shortening the CRT fixes the headline number *and* unstarves the transform
threads. One change, both effects.

## Stage 1, [NTT-CRT-ProductTree]

### The idea

Replace the linear fold with divide and conquer. Reconstructing over primes
`p_1..p_n` splits into reconstructing over the first half and the second half
independently, then combining the two partial results with a single CRT step
on their respective moduli. Recursively, that is a balanced binary tree of
depth `log2(n)`.

The win is asymptotic, not just structural. The linear fold does
`sum_i O(i) = O(n**2)` bigint work per coefficient. The tree does `O(n log n)`
multiplications of numbers that are each at most the final size, and Python's
integers use Karatsuba multiplication above roughly 70 digits, which every
partial modulus here exceeds (a 9,977 bit modulus is about 3,000 digits). The
expected gain on the headline case is **several times, to be measured in the
spike below**, and it costs no new dependency, no process, no pickling and no
change to the public surface.

### Memory, and the online form

A naive tree wants every prime's residue vector in hand before it starts, which
for the headline case is 321 vectors of 10^6 uint32, about 1.3 GB on top of the
1.25 GB the answer already holds. Avoid that with the standard **merge-stack**
form: keep a stack of partial reconstructions, each tagged with its modulus and
the number of primes behind it, and after pushing a new prime merge any two
top-of-stack partials of equal weight. The stack holds at most `log2(n)`
partials, the largest being the final answer, so the peak is roughly twice the
answer rather than the answer plus every head. Residue vectors are consumed as
they arrive, exactly as today, so the transform phase and its thread pool are
untouched.

Peak memory does roughly double, which `NTT_MAX_BYTES` and the guardrail
estimate in `AggregateNTT._ntt_sizes` must account for. That is a visible
behavior change on the largest cases and belongs in the CHANGELOG entry.

### Spike first

**Measure before implementing.** A throwaway script reconstructing one
coefficient vector both ways over the 141 primes of the 500 x 500 case, and
over the 321 primes of the headline case, settles the gain in minutes and
decides whether stage 2 is wanted at all. If the tree lands the headline case
under about 100 s, stage 2 is probably not worth building, and the author
should be asked before it is.

### Correctness

The three end-to-end checks are untouched and decisive: coefficients sum to
exactly `D`, all nonnegative, zero beyond the support, with the CRT modulus
exceeding `D`. Beyond those, a direct test that the tree and the linear fold
produce **identical integers** on the oracle corpus and the 100 x 100 dice is
the strongest available pin, in the same spirit as
`test_geometric_path_matches_forced_walk`. Keep the linear fold reachable
behind a module-level constant so that test can force it, as
`_GEOMETRIC_RUN_MIN` does.

### Files

`src/aggregate/ntt.py` (the reconstruction loop in `exact_compound`, the
guardrail estimate, module docstring Notes), `tests/test_ntt.py`,
`src/aggregate/constants.py` if `NTT_MAX_BYTES` moves.

## Stage 2, [NTT-CRT-ProcessPool], only if stage 1 is not enough

The tree's subtrees are independent, so the bottom levels fan out across
processes: give each worker a contiguous slice of the primes, have it return
that slice's partial reconstruction and modulus, and combine the partials in
the parent. Alternatively fan out by coefficient chunk, which parallelizes the
whole phase rather than its lower levels.

**The cost that must be weighed before building this.** A partial
reconstruction for the headline case is hundreds of megabytes of Python
integers, and crossing a process boundary means pickling every one of them.
That may well eat the gain; it is the reason this is stage 2 and conditional
rather than the plan's opening move. Chunking by coefficient is the better
shape if it is built, because each worker then returns a disjoint slice of the
final answer, pickled once.

Not specified further here. If stage 1's measurement says this is wanted, it
earns its own section with its own numbers.

## Explicitly not in this plan

**The numba transform kernel.** `[NTT-Numba-Kernel]`, stage 3 of
`dev/done/plan-ntt-speed.md`, was dropped on 2026-10-09 on the measurement
above: the transform phase is 0.7 percent of the headline wall clock, so
compiling it cannot move the number. Should it ever be revived, two findings
from that plan's review are recorded in its execution log and must be carried
forward: the plan's REDC sequence overflows uint64 at these prime sizes (every
prime sits just under `2**32`, so `T + m * p` reaches almost `2**65`), and the
high-word formulation is the fix.

**The `ntt_workers` default.** `min(8, os.cpu_count())` was ruled at
1.2.0a2. The measurement since shows the gain at `L = 2**20` is *overlap* with
the serial CRT rather than parallel speedup: the transform phase sums to
3638.8 thread-seconds against a serial projection of 126 s, a 29x inflation, so
eight workers buy the overlap at 29 times the CPU. Once stage 1 shortens the
CRT, the overlap is worth less and the right default may be 2 to 4. Revisit at
the close of stage 1 with a measurement, not now.

## Open questions for the author

1. **Is stage 1 alone acceptable as the close of this work?** The
   recommendation here is to spike the product tree, implement it, measure, and
   then decide stage 2 on the number rather than committing to processes now. ==> Agree with two step approach.
2. **Does doubled peak memory on the largest cases need a guardrail change, or
   just a CHANGELOG line?** The merge stack roughly doubles the peak against
   today's 1.25 GB on the headline case. `NTT_MAX_BYTES` currently estimates
   the answer only. ==> this is not a concern; just CHANGELOG line.

## Execution log, stage 1 [NTT-CRT-ProductTree], landed 1.2.0a4

Status: stage 1 **DONE** at 1.2.0a4 on 2026-10-09. Stage 2 and the
`ntt_workers` default are open and sit with the author; see the foot of this
log. The plan stays in `dev/` until that call is made.

### The spike, run before implementing as the plan required

Three reconstructions of the same residue vectors over the real NTT primes of
both cases, timed on a batch of coefficients and projected to the full count.
All three agreed bit for bit with the fold.

| case | primes | fold, projected | tree, projected | gain |
|---|---|---|---|---|
| 500 x 500 | 141 | 31.3 s | 13.4 s | 2.3x |
| 1000 x 1000 | 321 | 534.7 s | 174.6 s | 3.1x |

The projections reproduce the plan's own measured reconstruction spans (33.1 s
and 589.6 s), so the harness was calibrated before anything was implemented.

### Divergences from the plan as written

**The plan's `O(n log n)` claim is wrong, and the gain is 3.1x rather than
"several times".** The review raised this before implementing: Python's
multiplication is Karatsuba above roughly 70 digits as the plan says, but its
division is schoolbook at every size, and the Garner merge needs a reduction
modulo the right-hand modulus whose operands are each half the final size.
Summed over the tree levels that is about `n**2 / 2` against the fold's
`n**2`, which predicts about 2x rather than an asymptotic win. The measured
2.3x at 141 primes and 3.1x at 321 primes match that reading, including the
growth with `n` from the Karatsuba terms. The implementation is unaffected;
only the stated reason for it changes, and the docstring states the measured
figures rather than a complexity class.

**A Barrett merge was spiked and rejected.** Replacing the division with a
precomputed reciprocal and two multiplications, which is the textbook fix for
schoolbook division, measured *slower*: 60.4 us against 53.5 us per
coefficient at 141 primes, 188.6 against 174.6 at 321. The reason is that both
operands of that division are near the same size, so the quotient is a single
digit and CPython's division is already linear in the operand size. Recorded
in the `_crt_merge` docstring so it is not attempted again.

**A hybrid, linear fold at the bottom of the tree, was spiked and rejected.**
The merge stack makes about `2n` passes over the coefficient list against the
fold's `n`, so the bottom merges pay full interpreter overhead for cheap
bigint work. Folding the first `block` primes linearly and treeing above
measured flat to worse: at 321 primes, 182.5 s at block 2 against 174.6 s for
the pure tree, 185.1 s at block 8, 283.3 s at block 32. The plan's merge
stack, unmodified, is both the fastest and the simplest shape.

**Peak memory: CHANGELOG line only, no guardrail change**, per the author's
ruling on open question 2.

### End-to-end measurements, 2026-10-09 on DOOB at 1.2.0a4

Serial runs force the fold through `_CRT_TREE_MIN` on the same build, so these
are one binary against itself.

| case | workers | wall | transform, thread-s | reconstruction |
|---|---|---|---|---|
| 500 x 500, fold | 1 | 43.3 s | 10.5 | 32.8 s |
| 500 x 500, tree | 1 | 24.5 s | 10.4 | 14.0 s |
| 500 x 500, tree | 8 | **16.6 s** | 119.3 | 14.5 s |
| 1000 x 1000, tree | 1 | 312.9 s | 104.8 | 207.3 s |
| 1000 x 1000, tree | 8 | **212.3 s** | 1355.7 | 197.7 s |

Against 1.2.0a3, the 500 x 500 case goes 35.2 s to 16.6 s and the headline
1000 x 1000 case 594.0 s to 212.3 s, a 2.80x gain, with the reconstruction
589.6 s to 197.7 s, 2.98x. The fold run reproduces the plan's baseline to
within a second, which pins the comparison.

Both predicted effects landed. The headline number fell, and the transform
threads unstarved: 500 x 500 transform thread-seconds fall from 266.2 to 119.3
for identical arithmetic, and at the headline size from 3638.8 to 1355.7. The
per-prime log lines no longer climb through a run.

### What sits with the author

**Stage 2, `[NTT-CRT-ProcessPool]`.** The plan set about 100 s on the headline
case as the bar below which stage 2 is not worth building. Stage 1 landed at
212.3 s, above that bar, so the plan's own test does not decide it. The cost
shape has changed: the reconstruction is 197.7 s and the serial transform
104.8 s, so the reconstruction is now 93 percent of the serial wall clock
rather than 99.3 percent, and perfect parallelism over it would floor the
headline case near 105 s. Weigh that against pickling roughly 1.25 GB of
Python integers across a process boundary, which is the reason the plan staged
it second. Chunking by coefficient, as the plan says, is the better shape if
it is built.

**The `ntt_workers` default.** The plan deferred this to the close of stage 1
with a measurement. Taken at the headline size, 1000 x 1000 uniform dice over
321 primes:

| workers | wall | transform, thread-s | reconstruction |
|---|---|---|---|
| 1 | 312.9 s | 104.8 | 207.3 s |
| 2 | 264.1 s | 450.2 | 209.0 s |
| 4 | **216.4 s** | 709.3 | 187.0 s |
| 8 | 212.3 s | 1355.7 | 197.7 s |

**Four workers match eight on wall clock for half the CPU**, 216.4 s against
212.3 s for 709.3 thread-seconds against 1355.7. The gain over serial is
1.45x either way, so the eighth worker buys 2 percent of wall clock for twice
the CPU. The recommendation is to change the default from
`min(8, os.cpu_count())` to `min(4, os.cpu_count())`. It was not changed in
this bump because the plan deferred the question rather than ruling it, and
the default is a visible behavior change that belongs in its own entry. GIL
contention is much reduced from the 29x inflation measured at 1.2.0a2 but has
not gone, at 6.8x on four workers, because the reconstruction still runs in
the calling thread.
