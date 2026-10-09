# [NTT-Rust-Integration] use the compiled engine when it is installed

Status: **PHASE 1 OF 2 LANDED at 1.2.0a8**, 2026-10-09, specified against
1.2.0a5. The reconstruction phase is integrated and measured; see the
execution log at the end of this file. **The transform phase is still
blocked** on `[Rust-Transform]` in `aggregate-ntt`, so this plan stays in
`dev/` rather than `dev/done/`.

This is the `aggregate` half of a **reflected pair**. The other half is
`dev/plan-rust-engine.md` in `V:\dev\aggregate-ntt`, which specifies the
engine itself. Each file states its own side as requirements; each is retired
on its own side when that side lands. Read both before changing the seam.

## Goal

`aggregate-ntt` is a separate, optional, Rust package that computes the two
hot phases of an exact `nttagg` computation. When it is installed, use it.
When it is not, behave exactly as today. The library must never require it,
never depend on it, and never be slower or less correct for its absence.

Stated as the requirement it is: **a user who has never heard of
`aggregate-ntt` must see no change whatsoever.**

## Why there is a separate package at all

Three reasons, all of which keep this library better off, and all of which are
the author's ruling of 2026-10-09 rather than an engineering preference.

`aggregate` stays pure Python. No wheels per platform and interpreter, no Rust
toolchain for contributors, no compiled artifact in the release process of a
library whose value is its mathematics.

The license stays clean. If the engine ever links GMP for its bigint
arithmetic, that LGPL obligation lives in the separate distribution and never
reaches this BSD library. Correspondingly: **never add `gmpy2` or any GMP
binding as a dependency here**, and never vendor one.

And a separate package can be built for one machine with
`-C target-cpu=native`, which a published wheel cannot, and which is most of
the point of compiling.

## Current behavior, for a reader with no context

`aggregate.ntt.exact_compound(freq_outcomes, freq_probs, sev_outcomes,
sev_probs, workers=None)` returns `(numerators, denominator, meta)`, where
`numerators` is a list of Python integers, `P(A = n) = numerators[n] /
denominator` exactly, and `meta` is an audit record. Internally:

1. Validate inputs; clear denominators to integers (`_clear_denominators`).
2. Choose NTT primes whose product exceeds the coefficient bound
   (`ntt_primes`), all below `2**32`.
3. Per prime, transform and evaluate (`_residues_mod_prime`), on a
   `ThreadPoolExecutor` of `min(4, os.cpu_count())` threads.
4. Reconstruct the integers with a balanced product tree built online as a
   merge stack (`_crt_merge`, and the loop in `exact_compound`).
5. Run three decisive checks: coefficients sum to exactly `D`, none negative,
   none nonzero beyond the support.

`AggregateNTT` wraps this behind the DecL `nttagg` kind, adding
`density_df.p_exact`, `accuracy_df`, `stats_df`'s `exact` column, and
`ntt_audit`.

Steps 3 and 4 are what the engine replaces. **Steps 1, 2 and 5 stay here**,
which is the safety argument: the three checks police the compiled engine's
answer exactly as they police this one, and `exact_compound_poly` remains an
independent oracle that uses neither a transform nor the CRT.

## The design

### [Rust-Engine-Discovery]

A module-level optional import in `aggregate/ntt.py`, the same shape numba's
optional paths use in `utilities.py`:

```python
try:
    import aggregate_ntt as _engine
except ImportError:
    _engine = None
```

Then a resolution function, not an inline conditional, because three things
must be decided together: whether the engine exists, whether its protocol
version is one this library knows, and what the caller asked for.

```python
_ENGINE_PROTOCOL = 1   # the seam version this library speaks

def _resolve_engine(requested):
    """Return 'rust' or 'python', or raise if the request cannot be met."""
```

Rules:

- `requested='python'` always gives `'python'`, even with the engine
  installed. This is the bisecting switch, and it must never fail.
- `requested='rust'` gives `'rust'`, or **raises** if the engine is missing or
  its protocol version is unknown. An explicit request that cannot be honored
  is an error, never a silent downgrade.
- `requested='auto'`, the default, gives `'rust'` when the engine is present
  **and** `_engine.PROTOCOL_VERSION == _ENGINE_PROTOCOL`, otherwise
  `'python'`. A protocol mismatch under `'auto'` falls back and warns once; it
  does not raise, because a user who upgraded one package should still get
  answers.

**A protocol mismatch must never compute.** Two separately versioned packages
drift, and the one unacceptable outcome is a silently wrong answer instead of
a loud refusal. This is the single most important rule in this plan.

### [Rust-Engine-Keyword]

`exact_compound` gains `ntt_engine='auto'`, and `AggregateNTT.update` accepts
`ntt_engine=` and passes it through, alongside the existing `ntt_workers` and
`ntt_override`. So:

```python
build('nttagg Dice dfreq [1:100] dsev [1:100]', ntt_engine='python')
```

**Name vetted 2026-10-09.** `rg` for `ntt_engine`, `_engine`, `_resolve_engine`
and `_ENGINE_PROTOCOL` across `src/aggregate` returns nothing. `ntt_engine`
matches the existing `ntt_workers` / `ntt_override` keyword family, so it
reads as one of a set rather than an addition. It is a keyword only, never an
instance attribute, so it cannot shadow a method, which is the failure mode
`CLAUDE.md` records from `plan-approximate.md`.

### [Rust-Engine-Provenance]

**A result must always say what produced it.** `meta` gains three keys and
`ntt_audit` gains the matching rows:

| key | value |
|---|---|
| `engine` | `'python'` or `'rust'` |
| `engine_version` | the engine's own version, or `None` |
| `engine_protocol` | the negotiated protocol version, or `None` |

Without this, two machines disagree about what produced a number and nobody
can tell which. With it, a saved `ntt_audit` is self-describing.

Note `tests/test_ntt.py` already asserts the audit's row set, so that test
moves and its diff must be read deliberately.

### [Rust-Engine-Phases]

Integrate in the order the engine lands them, one bump each.

**Reconstruction first.** Replace the merge-stack loop in `exact_compound`
with a call to `_engine.Reconstructor` when the engine is resolved, keeping
the Python merge stack as the fallback and as the conformance reference. The
engine is a stateful accumulator for a reason: residue vectors are consumed
one at a time as the thread pool produces them, and the one-shot form would
put 1.3 GB across the boundary on top of the 1.25 GB the answer occupies.

`finish()` returns `(limbs, modulus_bytes)`, fixed-width little-endian 64-bit
limbs, one row per coefficient. Convert in bulk:

```python
numerators = [int.from_bytes(row.tobytes(), 'little') for row in limbs]
modulus = int.from_bytes(modulus_bytes, 'little')
```

**Measure that conversion and report it as its own `meta` key.** It is a
million `int.from_bytes` calls and it must not hide inside `crt_seconds`. The
engine plan budgets it under 3 s; if it misses, `[Rust-Query-Object]` over
there is promoted and this phase waits.

**Transform second.** Swap `_residues_mod_prime` for the engine's. Two
consequences to handle rather than discover:

1. **Prime width.** The engine can use 64-bit primes, because Rust has `u128`;
   this library cannot, because numpy has no 128-bit integer. So `ntt_primes`
   grows a width parameter, or the engine reports the primes it chose. Either
   way the pure-Python path keeps 32-bit primes. Decide this when the phase
   starts, with the engine's measurement in hand.
2. **The two engines will then use different prime sets, and that is
   correct.** The reconstructed integers are determined by the mathematics,
   not by which primes carried them, so conformance still holds bit for bit.
   What differs is `primes_used` and `prime_bits_*`. **Conformance tests must
   compare answers, never audit records.** Any test that compares `meta`
   across engines is wrong and will fail for the right reason.

## Correctness

The three end-to-end checks stay in Python and run on the engine's answer
unchanged. Beyond them, the pin is the one used throughout this plan's
predecessors: **the two engines must produce identical integers.**

Add `test_rust_engine_matches_python`, parametrized over the oracle corpus and
the 100 x 100 dice, calling `exact_compound` twice with `ntt_engine='python'`
and `ntt_engine='rust'` and comparing numerators and denominator. Guard it
with `pytest.importorskip('aggregate_ntt')` so the suite is green on a machine
without the engine, which is the normal case and must stay the normal case.

This chains onto what already exists: `test_crt_tree_matches_forced_fold` pins
the product tree against the linear fold, and
`test_geometric_path_matches_forced_walk` pins the closed-form spectral run
against the per-outcome walk. Four reconstructions, all pinned to each other.

**The gate runs twice from now on.** `uv run pytest -m 'slow or not slow'`
with the engine absent is the release gate, because that is what users have.
With it installed is an additional gate the author runs locally. A change that
passes only one of the two is not done.

## Acceptance

- With the engine absent, behavior and timings are **identical** to 1.2.0a5,
  and the full suite is green.
- With the engine present, answers are bit-identical to the pure-Python path
  on the oracle corpus and the 100 x 100 dice.
- `ntt_engine='rust'` raises, clearly, when the engine is missing or its
  protocol is unknown. `ntt_engine='python'` never fails.
- A protocol mismatch under `'auto'` falls back, warns once, and does not
  compute with the engine.
- `ntt_audit` names the engine, its version and the protocol.
- The headline 1000 x 1000 case in low tens of seconds against 212.3 s, which
  is the engine's target and the reason for all of this.

## Files

`src/aggregate/ntt.py` (the optional import, `_ENGINE_PROTOCOL`,
`_resolve_engine`, the `ntt_engine` keyword on `exact_compound` and
`AggregateNTT.update`, the reconstruction and transform call sites, the `meta`
dict, `ntt_audit`, docstring Notes), `tests/test_ntt.py`,
`docs/4_dec_Language_Reference.rst` (the `nttagg` paragraph documents
`ntt_workers`; it gains `ntt_engine`), `README.md` only if the optional
accelerator is worth naming on the front page, which is the author's call at
release.

## Explicitly not in this plan

**Making `aggregate-ntt` a dependency, optional extra or otherwise.** It is
discovered, not declared. An extra would put it in the dependency graph and
invite a resolver to install a compiled package nobody asked for. Document it
instead.

**Changing the pure-Python engine.** It stays, it stays the default, it stays
the conformance reference, and it stays fast enough for any ordinary problem.

**`[NTT-CRT-Gmpy2]`**, stage 3 of `dev/plan-ntt-crt.md`, which was measured at
2.0x and **shelved** on 2026-10-09 in favor of this work. It remains the
fallback if the Rust engine stalls, and it is the reason this plan can afford
to be unhurried.

**`[NTT-CRT-ProcessPool]`**, stage 2 of `dev/plan-ntt-crt.md`, which this
supersedes outright: threads without a GIL are what that stage was trying to
approximate with processes and pickling.

---

## Execution log

### Phase 1, reconstruction, landed at 1.2.0a8 on 2026-10-09

Against `aggregate-ntt` 1.0.0a2, whose `[Rust-Reconstruction]` landed the same
day with `PROTOCOL_VERSION` still 1 and no seam signature change, so the
`_ENGINE_PROTOCOL = 1` pinned above is correct as written.

Everything specified for this phase landed as specified:
`[Rust-Engine-Discovery]`, `[Rust-Engine-Keyword]`, `[Rust-Engine-Provenance]`
and the reconstruction half of `[Rust-Engine-Phases]`. Four divergences, all
small, all recorded here rather than absorbed silently.

**Divergence 1: the mismatch warning is a `UserWarning`, not a
`RuntimeWarning`.** The plan did not name a category. `RuntimeWarning` was
the obvious first choice and is wrong here: `[RuntimeWarning-Census]`
(1.0.0a220) establishes that this library emits no `RuntimeWarning` of its
own, which is what makes `pytest -W error::RuntimeWarning` a meaningful
numerics gate. A mismatched package is not a numerical event and must not
trip it, and `UserWarning` is in any case what the rest of the library uses
to tell the user how a computation was set up. The reasoning is recorded as a
comment at the warning site, because the choice looks arbitrary otherwise.

**Divergence 2: the limb decode happens on the whole array at once.** The
plan's sketch was `[int.from_bytes(row.tobytes(), 'little') for row in
limbs]`. Implemented as one `limbs.tobytes()` followed by fixed-width slices
of that buffer, which is the same integers with one copy instead of a million
small ones. Measured 0.89 s on the headline case against the plan's 3 s
budget, matching the engine's own measurement, so `[Rust-Query-Object]` is
not promoted.

**Divergence 3: `meta` gained a fourth key, not three.** The plan asked for
the decode to be "its own `meta` key" without naming it. It is
`limb_decode_seconds`, `0.0` on the pure-Python path so the audit record has
the same shape on every path, including the degenerate `support == 0` return,
which also reports the resolved engine for the same reason.

**Divergence 4, a finding the plan did not anticipate: two existing tests
would have gone vacuous.** `test_crt_tree_matches_forced_fold` and
`test_crt_tree_matches_forced_fold_dice` force the quadratic fold by raising
`_CRT_TREE_MIN` above the prime count. With the engine installed, `'auto'`
resolves to Rust on both sides of the comparison, `_CRT_TREE_MIN` is never
consulted, and the test passes while checking nothing. Both now pin
`ntt_engine='python'` on both calls. This is the general hazard of adding an
engine switch under a defaulted keyword: **a test that pins one
implementation against another has to say which implementations it means.**

### Acceptance, measured on DOOB

Uniform dice of dice, `exact_compound` end to end, `workers` at its default
of 4, release build installed with `uv pip install V:/dev/aggregate-ntt`.

| case | Python | Rust | ratio |
|---|---|---|---|
| 500 x 500 wall | 18.25 s | 4.31 s | 4.2x |
| 500 x 500 reconstruction | 14.55 s | 0.77 s | 19x |
| 1000 x 1000 wall | 212.3 s (a4) | 39.16 s | 5.4x |
| 1000 x 1000 reconstruction | 187.0 s (a4) | 7.49 s | 25x |
| 1000 x 1000 limb decode | n/a | 0.89 s | budget 3 s |

Answers bit-identical at both sizes, which is the only acceptance that
matters. The acceptance line above asked for "low tens of seconds against
212.3 s" and 39.16 s meets it.

The transform phase's thread-seconds also fell, 66.6 s to 15.9 s at
500 x 500, for the same reason the product tree did this at 1.2.0a4: the
Python reconstruction held the GIL in long stretches and starved the
transform threads. Shortening the reconstruction speeds both phases.

Both gates run, as this plan requires from now on. Engine absent: 5,775
passed, 9 skipped, 155.65 s. Engine present: 5,784 passed, 157.10 s, the nine
being the conformance tests that skip without it.

### Carried forward to phase 2

**The `ntt_workers` default of 4 is now too low on the engine path**, and
this is the most actionable thing phase 1 turned up. It was settled at
1.2.0a5 when the Python reconstruction dominated and extra threads bought
only overlap with it. With the reconstruction at 7.5 s the transform phase is
the whole cost, and it scales: 500 x 500 ran 3.57 s at four threads, 2.64 s
at eight, 1.92 s at sixteen, on a 32-core machine. Raising the default would
also move the pure-Python path, where 4 is still right, so the two paths may
want different defaults. **Left alone deliberately**: it is the author's
call, and the acceptance criterion above requires timings identical to
1.2.0a5 when the engine is absent.

The two consequences the plan already names for phase 2 stand unchanged: the
prime-width question (64-bit primes in Rust against numpy's 64-bit ceiling
here), and the rule that conformance tests compare answers and never audit
records, since the two engines will then legitimately choose different
primes.
