# [NTT-Exact-Aggregate] nttagg: exact compound distributions via the number-theoretic transform

Status: EXECUTED 2026-10-09, released as 1.1.0 (staged 1.1.0a1 through
1.1.0a5, one commit per bump). Divergences recorded in the Execution log at
the end of this file.

## Goal

Add a new DecL kind, `nttagg`, that computes a bounded discrete compound
distribution **exactly**, as rational numbers, using the number-theoretic
transform (NTT) over prime finite fields with Chinese remainder theorem (CRT)
reconstruction. The object also runs the ordinary floating-point FFT pipeline,
so the two answers sit side by side on the same lattice:

- `summary_df`, `validation_df`, `stats_df` as usual (inherited), with
  `stats_df` gaining an `exact` column;
- `density_df` gaining `p_exact` (exact rationals) and `p_exact_float`
  (correctly rounded binary64 of the rational), alongside the usual algorithm
  float `p_total`;
- a new `accuracy_df` comparing the exact and floating answers per bucket;
- a new `ntt_audit` Series recording sizes, primes, timings, and error
  summaries.

The deliverable is the computational answer to Wilson and Keich: Wilson2016a
(pairwise convolutions via FFT, CSDA), Wilson2017 (accurate small tail
probabilities of lattice sums via FFT, JCGS), and Wilson2016b (thesis). Their
aFFT-C controls *relative* error approximately; for bounded lattice compounds
the NTT + CRT route is **exact**, at a quantifiable cost of roughly
`bound_bits / 31` modular transforms. `accuracy_df` and `ntt_audit` are the
raw material for a response note (the note itself is a separate writing
project, out of scope here). The mathematics follows the blog post
`D:\Projects\KOG-blog\posts\notes\2026-08-12-dice-of-dice-addendum\index.qmd`,
which validated the method end to end on the 100 x 100 dice-of-dice example.

## Current behavior

`agg Dice dfreq [1:100] dsev [1:100]` runs the binary64 FFT pipeline. For this
input the default build already lands on the exact lattice (`bs=1`,
`log2=14`), and absolute accuracy is excellent (empirical noise floor around
6e-19). But the exact endpoint probability is `1/100^101 = 1e-202`: the float
answer carries no relative information in the deep tail, returning zero or
noise there. There is no exact path anywhere in the library, and the parser
evaluates every DecL number to binary64 at the token
(`UnderwritingTransformer.NUMBER`, parser.py line ~471), so exact rational
inputs cannot even be expressed today.

## Settled design decisions

These were agreed in conversation on 2026-10-09; do not re-litigate.

1. **Text-exact inputs.** `nttagg` gets its own grammar rules whose
   transformer callbacks keep exact values: a uniform default is exactly
   `1/n`, a decimal literal `0.3` is exactly `3/10` (via `Fraction(token
   text)`, which also handles scientific notation exactly), and an explicit
   `a/b` is `Fraction(a, b)`. Note the happy fact that `float(Fraction(...))`
   is correctly rounded, hence bit-identical to what the ordinary float
   pipeline computes from the same literals, so the inner float Aggregate can
   be driven from the exact spec with no input divergence: the comparison
   isolates pure algorithm error, exactly Wilson's framing.
2. **Integer outcomes only**, both clauses, nonnegative. A non-integer
   outcome raises with a message telling the user to clear denominators and
   rescale the lattice themselves.
3. **Names** (vetted against the existing surface 2026-10-09, no collisions):
   DecL keyword `nttagg`, class `AggregateNTT` (Base<Kind> prefix rule),
   module `ntt.py` (promoted-module precedent: `ft.py`, `tweedie.py`;
   submodule access only, no top-level re-export), columns `p_exact` /
   `p_exact_float`, properties `accuracy_df` / `ntt_audit`, spec keys
   `ntt_freq_outcomes` / `ntt_freq_probs` / `ntt_sev_outcomes` /
   `ntt_sev_probs`, update override kwarg `ntt_override`.
4. **Guardrails**: refuse monster cases up front with an informative estimate;
   override with `ntt_override=True`.
5. **Exact moments feed validation** in v1: `stats_df` gains an `exact`
   column and `validation_df` errors for an `AggregateNTT` reference it.
6. **No new dependencies.** scipy has no number theory and sympy is not a
   dependency; a deterministic Miller-Rabin for n < 2^64 is about fifteen
   lines (fixed witness set, first twelve primes, a published deterministic
   result for this range) and is trivially unit-tested. Adding sympy for
   `nextprime` would be a heavy dependency for fifteen lines.
7. **Parallelism is a measured, later stage**, not v1. The per-prime pipeline
   is embarrassingly parallel and numpy uint64 ufuncs release the GIL, so a
   thread pool helps the transform phase at zero pickling cost; the CRT is
   pure-Python bigint work (GIL-bound) and wants process-pool chunking.
   Implement serial, measure, then decide.

## Mathematics (for the implementer)

Inputs: frequency outcomes `k_1 < ... < k_m` (nonnegative integers) with
probabilities `f_i` (exact rationals summing to 1), severity outcomes
`s_1 < ... < s_r` (nonnegative integers) with probabilities `q_j` (exact
rationals summing to 1).

Let `Df = lcm` of the frequency denominators, `F_i = f_i * Df` (integers);
`Ds = lcm` of the severity denominators, `Q_j = q_j * Ds` (integers);
`Kmax = k_m`; support bound `N = k_m * s_r`.

Severity polynomial `A(z) = sum_j Q_j z^{s_j}`, so `A(1) = Ds`. The aggregate
pgf is `G(z) = sum_i f_i (A(z)/Ds)^{k_i}`. Clearing denominators:

    H(z) = Df * Ds^Kmax * G(z) = sum_i F_i * Ds^{Kmax - k_i} * A(z)^{k_i}

has nonnegative integer coefficients `H_n` with `sum_n H_n = D` where
`D = Df * Ds^Kmax`, and `P(A = n) = H_n / D` exactly. `B = D.bit_length()`
bounds every coefficient and therefore sets the CRT modulus requirement.

Transform length `L = 2^ceil(log2(N + 1))`, which prevents circular aliasing.
For each prime `p ≡ 1 (mod L)`, `p < 2^32`:

1. `a_hat = NTT_p(A)` (coefficient vector zero-padded to length L);
2. spectral evaluation `H_hat_r = sum_i F_i * Ds^{Kmax - k_i} * a_hat_r^{k_i}
   (mod p)`, vectorized over r. Walk the outcomes in ascending order keeping a
   running power array; bridge the gap from `k_{i-1}` to `k_i` by vectorized
   square-and-multiply, so the cost is about `log2(gap)` array
   multiplications per gap for sparse outcome sets and `Kmax` for a dense
   `[1:F]` (plain Horner order);
3. `H mod p = INTT_p(H_hat)`;
4. incremental CRT (as in the blog post): maintain the bigint reconstruction
   array, add `current_modulus * ((residue - current) * current_modulus^{-1}
   mod p)` per entry, stop once the accumulated modulus exceeds `D`.

End-to-end internal checks, always on: `sum_n H_n == D` exactly, all
`H_n >= 0`, `H_n == 0` for `n > N`. These are decisive whole-pipeline checks,
not debug assertions; keep them.

### Why 32-bit primes and a vectorized numpy NTT

With `p < 2^32`, products of two residues fit in uint64, so butterflies, the
spectral evaluation, and the inverse scaling are all full-array numpy
operations (`%` on uint64 arrays); subtraction uses `(u + p - v) % p` to stay
unsigned. The blog post's pure-Python NTT is the correctness reference but is
loop-bound; the numpy version is what makes "really stretch it" feasible.

Twiddle factors: per prime, build the length `L/2` table of powers of the
root of unity `w = g^((p-1)/L)` by log-doubling (`P_{2m} = [P_m, P_m *
P_m[-1] * w]` style, one scalar pow per doubling), then each butterfly stage
takes a strided slice. Bit-reversal permutation indices are computed once per
L and shared across primes.

Prime and generator search, pure Python, trivial cost: scan `p = c * L + 1 <
2^32` from the top down (largest primes first minimizes the count), test with
deterministic Miller-Rabin (witness set: the first twelve primes, which is
deterministic far beyond 2^64), factor `c` by trial division (plus the factor
2) to verify a generator `g` via `g^((p-1)/f) != 1` for each prime factor `f`
of `p - 1`.

### Size envelope and guardrail arithmetic

Prime supply `p ≡ 1 (mod L)`, `p < 2^32`, falls roughly as `1/L`: about
24,000 at L = 2^14; 3,000 at 2^17; 370 at 2^20; 90 at 2^22. Primes needed is
about `B / 31` (accumulate actual prime products until they exceed D).
Reference points:

- 100 x 100 dice: N = 10^4, L = 2^14, B = 672 bits, 22 primes. Well under a
  second vectorized.
- 1000 x 1000: N = 10^6, L = 2^20, B ≈ 10^4 bits, about 330 primes against a
  supply of about 370. Feasible on DOOB; tens of minutes serial, a few GB of
  exact numerators (10^6 integers of 10^4 bits each). This is near the
  practical ceiling and that is intrinsic to exactness, not to the transform.

Guardrail (checked in `AggregateNTT.update` before any heavy work): compute
`N`, `L`, `D`, `B`, primes needed, primes available, and estimated numerator
memory `(N + 1) * B / 8` bytes. Refuse with a message quoting all of these
when primes available are insufficient (hard error, no override possible) or
when the estimate exceeds the soft caps (override with `ntt_override=True`).
Soft cap constants live in `constants.py`: `NTT_MAX_SUPPORT = 2**20` and
`NTT_MAX_BYTES = 8 * 2**30` (8 GB). Going past the 32-bit prime supply would
require 62-bit Montgomery arithmetic in numpy; explicitly out of scope
(flagged and declined 2026-10-09).

## DecL surface

    nttagg DiceOfDice dfreq [1:100] dsev [1:100]                  note{exact dice of dice}
    nttagg Skewed     dfreq [0 1 2] [1/4 1/2 1/4] dsev [1 2 10] [0.3 0.5 0.2]

Grammar (decl.lark):

- new terminal `NTTAGG.2: /nttagg(?![a-zA-Z0-9._:~\-])/`, and `nttagg` added
  to the ID exclusion alternation;
- `answer` gains `| nttagg_out -> answer_nttagg`;
- productions, deliberately parallel to `dfreq`/`dsev` but kept separate so
  the shared float paths are untouched:

      nttagg_out: NTTAGG name as_label ntt_dfreq ntt_dsev trailer
      ntt_dfreq: DFREQ ntt_doutcomes ntt_dprobs
      ntt_dsev: DSEV ntt_doutcomes ntt_dprobs
      ntt_doutcomes: "[" ntt_intl "]"                     -> ntt_doutcomes_list
                   | "[" NUMBER RANGE NUMBER "]"          -> ntt_doutcomes_range
                   | "[" NUMBER RANGE NUMBER RANGE NUMBER "]" -> ntt_doutcomes_range_step
      ntt_dprobs: "[" ntt_ratl "]"   -> ntt_dprobs_list
                |                    -> ntt_dprobs_none
      ntt_ratl: list of ntt_rat;  ntt_intl: list of NUMBER
      ntt_rat: NUMBER                -> ntt_rat_number
             | NUMBER "/" NUMBER     -> ntt_rat_fraction

  The body is exactly `dfreq` then `dsev` then the trailer: no exposures, no
  layers, no reinsurance, no `approximate`, no `!`, no picks. Boundedness is
  guaranteed syntactically. `nttagg` is NOT a `port_item` and cannot appear
  inside a portfolio, a pnl, or a bivariate (v1; relaxing later is a separate
  decision).

Transformer (parser.py): the `ntt_*` callbacks never touch
`_check_vectorizable` or float conversion. They read `Token` **text**:

- outcome tokens must parse as base-10 nonnegative integers (`int(text)`);
  anything else raises `ValueError: nttagg outcomes must be nonnegative
  integers; clear denominators and rescale the lattice yourself` (wording to
  taste, but it must say rescale);
- probability tokens become `Fraction(text)` (handles `0.3`, `3e-2`, `7`)
  or `Fraction(int(num), int(den))` for the `a/b` form; zero or negative
  probabilities and a zero denominator raise;
- `ntt_dprobs_none` yields the exact uniform `[Fraction(1, m)] * m`;
- validations: lengths match, probabilities sum to **exactly** `Fraction(1)`
  (the error message prints the exact shortfall as a fraction), duplicate
  outcomes are combined by summing their probabilities and outcomes sorted
  ascending (mirroring dhistogram semantics);
- result: `('nttagg', name, spec)` with spec keys `name`,
  `ntt_freq_outcomes` (list[int]), `ntt_freq_probs` (list[Fraction]),
  `ntt_sev_outcomes`, `ntt_sev_probs`, plus the usual trailer keys
  (`label`, `label_map`, `note`, `hints`, `tags`).

## New module: `src/aggregate/ntt.py`

Engine functions (pure, no class state, unit-testable in isolation; NumPy
style docstrings with Notes explaining the algebra):

- `is_prime_u64(n)`: deterministic Miller-Rabin, witnesses = first twelve
  primes.
- `ntt_primes(length, bound)`: descending scan of `c * length + 1 < 2**32`;
  returns list of `(prime, root_of_unity)` whose product exceeds `bound`;
  raises (quoting supply found vs needed) if the supply is insufficient.
- `ntt(values, prime, root, inverse=False)`: vectorized radix-2
  transform on a uint64 numpy array; shared bit-reversal index cache keyed by
  length; log-doubled twiddle tables.
- `exact_compound(freq_outcomes, freq_probs, sev_outcomes, sev_probs)`:
  returns `(numerators, denominator, meta)` where `numerators` is a list of
  Python ints of length `N + 1`, `denominator = D`, and `meta` is a dict
  (support, transform length, bound bits, primes used, prime bit range,
  seconds in transform vs CRT phases). Implements the mathematics section,
  including the internal `sum == D` check.
- `exact_compound_poly(...)`: the direct Z[z] reference (repeated sparse
  polynomial multiply-accumulate with bigints, the blog post's moving-window
  generalized to weighted outcomes). Slow and simple; used by tests as the
  independent oracle and available for cross-checks. Public in the module,
  documented as the reference implementation.

Class `AggregateNTT(Aggregate)`, same module:

- `__init__(name, ntt_freq_outcomes, ntt_freq_probs, ntt_sev_outcomes,
  ntt_sev_probs, label=None, label_map=None, note='', hints='', tags=())`:
  stores the exact inputs, derives the standard float spec
  (`freq_name='empirical'`, `freq_a/freq_b`, `sev_name='dhistogram'`,
  `sev_xs/sev_ps`, `exp_en=-1`) via `float(Fraction)`, calls
  `super().__init__`.
- `update(log2=0, bs=0, **kwargs)`: pins the lattice. `bs` must be 0 or 1
  (anything else raises); an explicit `log2` smaller than required raises;
  default computes `log2 = ceil(log2(N + 1))` and lets the parent handle
  padding. Pops `ntt_override` from kwargs, runs the guardrail, calls
  `super().update(...)` for the float pipeline, then runs `exact_compound`
  and attaches results. Logs one line per prime at INFO so long runs show
  progress.
- `density_df`: parent frame plus `p_exact` (object column of reduced
  `Fraction`, `Fraction(0)` off-support) and `p_exact_float`. Built lazily
  with the parent's caching pattern.
- `accuracy_df` property: full-grid frame indexed by loss with columns
  `p_exact_float`, `p_fft` (the algorithm float, `p_total`), `err` (signed,
  computed as `float(Fraction(p_fft) - exact)` so the subtraction is exact
  before rounding, per the blog post's cancellation lesson), `abs_err`,
  `rel_err` (NaN where the exact value is zero).
- `ntt_audit` property: pd.Series with support, transform length, denominator
  bits, primes used, prime bits min/max, transform seconds, CRT seconds, max
  abs error, max rel error on the support, noise floor (max `|p_fft|` over
  exact-zero cells), count of exact-zero cells with nonzero float.
- exact moments: raw moments `sum n^k H_n / D`, k = 1, 2, 3, as Fractions
  (cheap bigint sums); likewise exact severity and frequency moments from the
  input rationals. `stats_df` gains an `exact` column (same row MultiIndex,
  values as floats of the exact rationals); `validation_df` error columns for
  an `AggregateNTT` are computed against `exact` rather than the float
  theoretical. Implementation hook: where `_validation.py` reads
  `stats_df['mixed']` for the comparison baseline, an `AggregateNTT` supplies
  `exact`; keep the change inside the subclass if possible (a small override
  or a baseline-column hook), do not fork `_validation.py` logic.

## Other files touched

- `decl.lark`, `parser.py`: as above.
- `underwriter.py`: `elif kind == 'nttagg':` branch building `AggregateNTT`
  (local import, matching the `bvagg` pattern); `_KIND_WRITE_ORDER` entry.
  The update loop needs no change: `AggregateNTT` is an `Aggregate` and rides
  that branch; its own `update` pins the lattice.
- `decl_writer.py`: render the spec back to DecL. Probabilities render as
  exact fractions `a/b` (bare integer when the denominator is 1); ranges may
  be reconstructed for contiguous uniform outcomes or written as explicit
  lists (writer's existing conventions decide). Round-trip (parse, write,
  parse) must reproduce the spec exactly.
- `constants.py`: `NTT_MAX_SUPPORT`, `NTT_MAX_BYTES`.
- `src/aggregate/agg/_test_suite.agg`: a handful of `nttagg` lines (new
  category), which automatically become parse + snapshot cases; regenerate
  `tests/data/expected_specs.json` via `tests/capture_spec_snapshot.py` and
  read the diff (only the new lines may move). Note: the snapshot is JSON, so
  the capture script must serialize Fractions (as `"a/b"` strings); check and
  extend it if needed.
- Docs: new `.rst` section for `nttagg` and the `ntt` module (API page plus a
  DecL language page entry); grammar-reference regen; **docs build is pending
  the author's manual rebuild per standing rule**, keep edits in lockstep
  only.
- `dev/TODO.md`, `CHANGELOG.md`, `dev/FEATURES.csv` via `dev/regen_features.py`
  at each bump per release hygiene.

## Implementation stages

Each stage is one version bump, one commit, committed by Claude when it
lands. Pre-releases of the 1.1.0 feature: 1.1.0a1, a2, ... with 1.1.0 final
at the last stage.

1. **[NTT-Engine]** (1.1.0a1): `ntt.py` engine functions plus
   `tests/test_ntt.py` engine tests. No DecL, no class yet; the module is
   importable and correct on its own.
2. **[NTT-DecL]** (1.1.0a2): grammar, transformer, `decl_writer`, test-suite
   lines, snapshot regen. `build('nttagg ...')` raises a clear
   not-yet-wired error from the underwriter at this stage (or the stage is
   combined with 3 if the seam is awkward; prefer separate).
3. **[NTT-Class]** (1.1.0a3): `AggregateNTT`, underwriter dispatch,
   `density_df` columns, `accuracy_df`, `ntt_audit`, guardrails, class tests.
4. **[NTT-Exact-Moments]** (1.1.0a4): `stats_df` exact column, validation
   baseline wiring, tests.
5. **[NTT-Parallel]** (1.1.0a5, conditional): measure the 1000 x 1000 case
   first. If the transform phase dominates, add `workers=` to
   `exact_compound` with a `ThreadPoolExecutor` across primes (numpy releases
   the GIL in the uint64 ufuncs; zero pickling). If CRT dominates, chunk the
   coefficient range across a `ProcessPoolExecutor` (Windows spawn cost is
   amortized over minutes-long jobs). Skip the stage entirely if serial is
   acceptable; record the measurement in this plan's done notes either way.
6. **[NTT-Docs-Release]** (1.1.0): `.rst` pages, grammar reference and
   FEATURES regen, TODO entry closed, plan moved to `dev/done/`, CHANGELOG
   final entry. Author rebuilds docs and pushes.

## Tests and acceptance checks

`tests/test_ntt.py` (engine and class; mark anything over ~5 s `slow`):

- **[primality]** `is_prime_u64` against a known prime/composite list
  including Carmichael numbers and 2^61 - 1.
- **[prime-supply]** `ntt_primes(2**14, 2**672)` returns 22 or 23 primes, all
  `≡ 1 mod 2^14`, all verified prime, product exceeding the bound.
- **[transform]** NTT/INTT round trip on random vectors for several primes
  and lengths; NTT-based cyclic convolution equals `np.convolve` (padded) on
  small integer vectors.
- **[oracle]** `exact_compound` equals `exact_compound_poly` on randomized
  small cases (outcomes up to 12, random rational probabilities, gappy and
  duplicate outcomes, frequency outcome 0 included).
- **[dice-closed-form]** the 100 x 100 case: `P(A=1) = 1/10^4`,
  `P(A=100) = 101^99 / 100^101`, `P(A=10000) = 1/100^101`, all as exact
  Fraction equalities; `sum(numerators) == 100^101` exactly.
- **[class]** `build('nttagg Dice dfreq [1:6] dsev [1:6]')` returns an
  `AggregateNTT`; `density_df.p_exact.sum() == 1` exactly; `accuracy_df` and
  `ntt_audit` have the documented columns/rows; `stats_df` has the `exact`
  column and the exact mean equals the known rational.
- **[guardrail]** an over-cap spec raises quoting the estimates; passes with
  `ntt_override=True` (monkeypatch the caps small so the test is fast).
- **[writer-round-trip]** parse, write, re-parse reproduces exact spec keys,
  including `0.3` surviving as `3/10`.
- **[decl-snapshot]** the new `_test_suite.agg` lines parse and match the
  regenerated snapshot.

Overall acceptance: full fast suite green (`uv run pytest`), gate green at
the final bump (`uv run pytest -m 'slow or not slow'`), and the quick smoke:

    from aggregate import build, qd
    a = build('nttagg Dice dfreq [1:100] dsev [1:100]')
    qd(a)                    # usual describe
    a.ntt_audit              # 22ish primes, exact endpoint numerator 1
    a.accuracy_df.loc[10000] # exact 1e-202, fft ~0, rel_err ~1

## Known limits (documented, not bugs)

- Integer lattices only; the user rescales.
- 32-bit prime supply caps the envelope at roughly L = 2^22; beyond that is
  explicitly out of scope (would need 62-bit Montgomery arithmetic in numpy).
- `nttagg` cannot appear in a portfolio, pnl, or bivariate in v1.
- Exact numerator memory is intrinsic: `(N + 1) * B` bits; the guardrail
  message quotes the estimate.

## Execution log

Review completed 2026-10-09 by the executing session; plan facts verified
against the code at 1.0.1 (parser.py:471 NUMBER, decl.lark rules, underwriter
_factory bvagg pattern, _KIND_WRITE_ORDER, _validation.py baseline hook,
Aggregate.update cap semantics). Names vetted with rg: no collisions.
Divergences and additions recorded here at the moment they are made.

- **[Update-Cap-Semantics]** build_many's Aggregate branch substitutes
  `self.log2` (16) for the log2=0 default before calling update, so
  `AggregateNTT.update` never sees 0 from `build`. Resolution: treat the
  incoming log2 as a cap. bs must be 0 or 1 and is pinned to 1; required
  log2 is ceil(log2(N + 1)); log2=0 (direct call) uses required; a cap below
  required raises naming the required value and the escape hatches
  (`build(..., log2=r)` or `hints{log2=r}`). The plan's claim that the update
  loop needs no change stands.
- **[Grammar-Mirrors]** adding the NTTAGG terminal forces three mirrors the
  plan did not list, enforced by tests/test_grammar_sync.py:
  decl_pygments.AggLexer, agg.sublime-syntax, parser_errors._TERMINAL_LABELS.
  Updated in the [NTT-DecL] stage.
- **[Snapshot-Fractions]** tests/test_decl_parser.py needs `nttagg` in
  VALID_KINDS and a Fraction branch in _normalize matching the capture
  script's jsonify (both serialize Fraction as the string "a/b").
- **[Test-Suite-Category]** _test_suite.agg letters A to O, Y, Z are taken and
  P to X are reserved by decl-testers.agg; the nttagg lines go in a named
  section instead of a letter.
- **[Dice-Prime-Count]** the 100 x 100 case needs 21 top-down primes, not the
  22 in the plan (D = 100^101 is 671 bits; 21 primes just under 2^32 carry
  about 672 bits). The 2^672-bound prime-supply test does return 22. The
  closed-form test accepts 21 or 22.
- **[NTT-Number-Terminal]** the transformer's global NUMBER terminal callback
  converts every NUMBER token to float before any rule callback runs, so the
  plan's "read Token text" cannot ride the shared NUMBER terminal. Resolution:
  a dedicated NTT_NUMBER terminal (nonnegative, no percent, no inf) used only
  inside the nttagg productions; its transformer callback returns the token
  text. The Earley dynamic lexer never co-expects NUMBER and NTT_NUMBER, so
  there is no ambiguity. Negative outcomes or probabilities are therefore
  parse errors rather than transformer ValueErrors, which is acceptable.
- **[Dice-Prime-Count-Context]** see [Dice-Prime-Count] above; test accepts
  21 or 22 primes.
- **[Stale-PnLBook-Test]** pre-existing red at HEAD 990a977: the author's
  library cleanup deleted port PnLBook from library.agg but
  tests/test_library_entries.py CANNOT_BUILD still listed it (2 failures).
  Fixed the stale entry in the tree; left UNCOMMITTED for the author (pure
  test maintenance on the author's own commit, not part of this plan).
- **[Accuracy-Raw-Density]** the plan's accuracy_df names p_fft as p_total,
  but density_df applies remove_fuzz (zeroes below ~2.2e-16), which would
  fake the deep-tail noise floor. accuracy_df therefore compares against the
  raw algorithm output self.agg_density; density_df's p_total stays the
  display density. Documented in both docstrings.
- **[Per-Prime-Logging-In-Engine]** the plan placed the one-line-per-prime
  INFO log in AggregateNTT.update; it lives in exact_compound's prime loop
  instead (the natural seam, and direct engine callers get it too).
- **[Build-Grid-Is-Cap-Sized]** via build, the parent estimator uses the full
  log2=16 cap (grid 65536) rather than the compact exact window; correctness
  is unaffected (the lattice is pinned at bs=1 and covers the support), and a
  compact grid is available with an explicit log2. The 100 x 100 case builds
  in about 1 second with 21 primes; the exact endpoint 1/100^101 and the
  float noise floor around 6e-19 reproduce the blog post's numbers.
- **[Audit-Object-Dtype]** ntt_audit stores object dtype so integer rows stay
  integers; the plain pandas repr displays small floats at display precision
  (qd formats it properly). Values are exact; display quirk only.
- **[NTT-Parallel-Measurement]** (1.1.0a5, 2026-10-09, DOOB, 32 logical
  cores). Serial dense-case scaling before the stage: 100x100 total 0.33s
  (transform 0.27 / CRT 0.06, 21 primes, L 2^14); 200x200 5.3s (4.5 / 0.8,
  49 primes, L 2^16); 400x400 164s (151 / 13, 109 primes, L 2^18). The
  transform phase dominates (over 90 percent), so per the plan rule the
  stage proceeds as the thread-pool variant: `workers=` on exact_compound
  (ThreadPoolExecutor across primes, per-prime pipeline factored into
  `_residues_mod_prime`), CRT serial in the calling thread. Two changes
  landed: the square-and-multiply walk no longer squares past the last bit
  of each gap (dense gap-1 cost halves: 400x400 serial 164s to 119s), and
  workers=8 gives 2.5x on top (47s; uint64 `%` ufuncs look memory-bandwidth
  bound, so scaling flattens past ~8 threads). Extrapolated 1000x1000:
  about an hour serial, tens of minutes threaded, matching the plan's
  envelope. `AggregateNTT.update` forwards `ntt_workers=<int>`. Serial
  remains the default.
- **[Docs-Release]** (1.1.0) language-reference section "Exact Compounds
  (nttagg)" added to docs/4_dec_Language_Reference.rst and an "Exact NTT
  compounds" automodule section to docs/3_reference/3_x_Auxiliary.rst;
  grammar reference regenerated at the [NTT-DecL] stage and unchanged since.
  Docs rebuild is pending the author's manual run per standing rule. The
  acceptance smoke ran at the [NTT-Class] stage: qd describe, 21 primes,
  exact endpoint numerator 1 (1/100^101) against an FFT value of 3.4e-20,
  rel_err 3.4e+182.
