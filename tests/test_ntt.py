"""
Engine tests for aggregate.ntt (dev/plan-nttagg.md, [NTT-Engine] stage).

The production path :func:`aggregate.ntt.exact_compound` is checked three
independent ways: against the direct bigint polynomial reference
:func:`aggregate.ntt.exact_compound_poly` on randomized small cases, against
closed-form rational probabilities for the 100 x 100 dice-of-dice example,
and through the always-on internal checks (coefficients sum to the
denominator, nonnegative, zero beyond the support) that every call exercises.
"""

import math
import os
import random
import warnings
from fractions import Fraction

import numpy as np
import pytest

import aggregate.ntt as ntt_module
from aggregate.ntt import (is_prime_u64, ntt, ntt_primes, exact_compound,
                           exact_compound_poly, _clear_denominators,
                           _find_generator, _outcome_runs, _prime_factors,
                           _root_of_unity)


# ---------------------------------------------------------------- [primality]

KNOWN_PRIMES = [2, 3, 5, 7, 31, 97, 65537, 999983, 2 ** 31 - 1, 2 ** 61 - 1,
                4294967291]
# Carmichael numbers are the classic Fermat-test killers; 3215031751 is the
# smallest strong pseudoprime to bases 2, 3, 5 and 7 simultaneously.
KNOWN_COMPOSITES = [0, 1, 4, 9, 561, 1105, 1729, 2465, 2821, 6601, 8911,
                    41041, 62745, 63973, 75361, 101101, 3215031751,
                    2 ** 61 + 1, (2 ** 31 - 1) * (2 ** 29 - 3)]


def test_is_prime_u64_on_known_primes():
    for p in KNOWN_PRIMES:
        assert is_prime_u64(p), p


def test_is_prime_u64_on_known_composites():
    for n in KNOWN_COMPOSITES:
        assert not is_prime_u64(n), n


def test_is_prime_u64_against_sieve():
    sieve = np.ones(10000, dtype=bool)
    sieve[:2] = False
    for i in range(2, 100):
        if sieve[i]:
            sieve[i * i::i] = False
    for n in range(10000):
        assert is_prime_u64(n) == bool(sieve[n]), n


# ------------------------------------------------------------- [prime-supply]

def test_ntt_primes_dice_supply():
    length, bound = 2 ** 14, 2 ** 672
    primes = ntt_primes(length, bound)
    assert len(primes) in (22, 23)
    product = 1
    for p, root in primes:
        assert p % length == 1
        assert p < 2 ** 32
        assert is_prime_u64(p)
        # root is a primitive length-th root of unity
        assert pow(root, length, p) == 1
        assert pow(root, length // 2, p) != 1
        product *= p
    assert product > bound


def test_ntt_primes_insufficient_supply_raises():
    # At length 2**30 only three candidates c * 2**30 + 1 sit below 2**32 and
    # just one (3 * 2**30 + 1) is prime, so a 2**64 bound must exhaust fast.
    with pytest.raises(ValueError, match='insufficient NTT prime supply'):
        ntt_primes(2 ** 30, 2 ** 64)


# ---------------------------------------------------------------- [transform]

@pytest.mark.parametrize('length', [2, 8, 64, 1024])
def test_ntt_round_trip(length):
    rng = np.random.default_rng(length)
    for prime, root in ntt_primes(length, 2 ** 40):
        values = rng.integers(0, prime, size=length, dtype=np.uint64)
        back = ntt(ntt(values, prime, root), prime, root, inverse=True)
        assert np.array_equal(back, values)


def test_ntt_convolution_matches_numpy():
    rng = np.random.default_rng(42)
    for trial in range(5):
        n = int(rng.integers(2, 20))
        x = rng.integers(0, 50, size=n)
        y = rng.integers(0, 50, size=n)
        full = np.convolve(x, y)  # length 2n - 1, max value < 50*50*n < p
        length = 1 << int(len(full) - 1).bit_length()
        (prime, root), = ntt_primes(length, 2 ** 20)[:1]
        xa = np.zeros(length, dtype=np.uint64)
        ya = np.zeros(length, dtype=np.uint64)
        xa[:n] = x
        ya[:n] = y
        prod = ntt(xa, prime, root) * ntt(ya, prime, root) % np.uint64(prime)
        back = ntt(prod, prime, root, inverse=True)
        assert np.array_equal(back[:len(full)], full.astype(np.uint64))


def test_ntt_rejects_bad_root():
    (prime, root), = ntt_primes(8, 2)[:1]
    with pytest.raises(ValueError, match='primitive'):
        ntt(np.arange(8), prime, pow(root, 2, prime))


# ------------------------------------------------------------------- [oracle]

def _random_clause(rng, max_outcome=12):
    """Random (outcomes, probs): gappy, duplicated, exactly summing to 1."""
    m = rng.randint(1, 5)
    outcomes = [rng.randint(0, max_outcome) for _ in range(m)]  # dups allowed
    weights = [rng.randint(1, 9) for _ in range(m)]
    total = sum(weights)
    probs = [Fraction(w, total) for w in weights]
    return outcomes, probs


def _random_run_clause(rng, max_runs=3):
    """Random piecewise-uniform (outcomes, probs) built from maximal runs.

    Run lengths mix singletons, short runs below ``_GEOMETRIC_RUN_MIN`` and
    long runs above it, separated by gaps, so one case exercises the
    geometric closed form, the short-run walk and the gap bridge together.
    """
    outcomes = []
    weights = []
    k = rng.choice([0, 1, 2])
    for _ in range(rng.randint(2, max_runs)):
        run_length = rng.choice([1, 1, 3, 9, 11])
        w = rng.randint(1, 7)
        outcomes.extend(range(k, k + run_length))
        weights.extend([w] * run_length)
        k += run_length + rng.randint(1, 3)
    total = sum(weights)
    return outcomes, [Fraction(w, total) for w in weights]


@pytest.mark.parametrize('seed', range(8))
def test_exact_compound_matches_polynomial_oracle(seed):
    rng = random.Random(seed)
    fo, fp = _random_clause(rng)
    so, sp = _random_clause(rng)
    fast, d_fast, _ = exact_compound(fo, fp, so, sp)
    slow, d_slow, _ = exact_compound_poly(fo, fp, so, sp)
    assert d_fast == d_slow
    assert fast == slow


def test_exact_compound_frequency_zero_mass():
    # P(N=0) = 1/2 puts an atom of exactly 1/2 at aggregate 0.
    numerators, denominator, _ = exact_compound(
        [0, 2], [Fraction(1, 2), Fraction(1, 2)],
        [1, 3], [Fraction(2, 3), Fraction(1, 3)])
    assert Fraction(numerators[0], denominator) == Fraction(1, 2)
    assert sum(numerators) == denominator
    # P(A=2) = 1/2 * (2/3)**2 = 2/9
    assert Fraction(numerators[2], denominator) == Fraction(1, 2) * Fraction(4, 9)


def test_exact_compound_degenerate_support():
    numerators, denominator, meta = exact_compound(
        [0], [Fraction(1)], [1, 2], [Fraction(1, 2), Fraction(1, 2)])
    assert numerators == [denominator]
    assert meta['support'] == 0


def test_exact_compound_validates_inputs():
    with pytest.raises(ValueError, match='rescale the lattice'):
        exact_compound([1.5], [Fraction(1)], [1], [Fraction(1)])
    with pytest.raises(ValueError, match='sum to exactly 1'):
        exact_compound([1], [Fraction(1, 2)], [1], [Fraction(1)])
    with pytest.raises(ValueError, match='positive'):
        exact_compound([1, 2], [Fraction(0), Fraction(1)], [1], [Fraction(1)])


def test_exact_compound_combines_duplicate_outcomes():
    a = exact_compound([1, 1], [Fraction(1, 4), Fraction(3, 4)],
                       [2], [Fraction(1)])
    b = exact_compound([1], [Fraction(1)], [2], [Fraction(1)])
    assert Fraction(a[0][2], a[1]) == Fraction(b[0][2], b[1]) == 1


# -------------------------------------------------------- [dice-closed-form]

def test_dice_of_dice_closed_form():
    """100 x 100 uniform dice of dice: known exact rational probabilities."""
    outcomes = list(range(1, 101))
    probs = [Fraction(1, 100)] * 100
    numerators, denominator, meta = exact_compound(
        outcomes, probs, outcomes, probs)
    assert denominator == 100 ** 101
    assert sum(numerators) == 100 ** 101
    assert Fraction(numerators[1], denominator) == Fraction(1, 10 ** 4)
    assert Fraction(numerators[100], denominator) == \
        Fraction(101 ** 99, 100 ** 101)
    assert Fraction(numerators[10000], denominator) == \
        Fraction(1, 100 ** 101)
    assert meta['support'] == 10000
    assert meta['transform_length'] == 2 ** 14
    # The exact rationals above are the content of this test and they do not
    # depend on which primes carried them. The prime count does: it is about
    # bound_bits / prime_bits, so it halves when the compiled engine draws
    # 63-bit primes. Assert the relationship rather than a literal, or this
    # test fails on a machine that merely has the engine installed.
    expected = -(-meta['bound_bits'] // meta['prime_bits_min'])
    assert meta['primes_used'] in (expected, expected + 1)


# ------------------------------- [geometric] ([NTT-Uniform-Geometric] stage)

def test_outcome_runs_decomposition():
    """Maximal runs: consecutive integers sharing one cleared weight."""
    runs = _outcome_runs([0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                         [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 7])
    assert runs == [(0, 3, 3), (5, 8, 1), (20, 1, 7)]
    # A weight change breaks a run even with no gap in the outcomes.
    assert _outcome_runs([4, 5, 6], [2, 5, 5]) == [(4, 1, 2), (5, 2, 5)]
    # Uniform [1:F] is the single run the geometric path exists for.
    assert _outcome_runs(list(range(1, 51)), [1] * 50) == [(1, 50, 1)]


def test_spectral_walk_degenerate_cell_is_always_present():
    """``b_0 = A(1) / Ds = 1`` on every prime, so the geometric mask is hot.

    The ``b == 1`` cells take the degenerate series ``run_length (mod p)``
    rather than the closed form. Frequency index 0 is such a cell on every
    prime and in every case, because the severity polynomial evaluated at 1
    is exactly its cleared denominator. Constructing a second degenerate
    cell deliberately is not cheap (it asks for a spectral point where
    ``A`` takes a prescribed value), and index 0 makes the mask load
    bearing for every check of the totals, so this pins index 0 only.
    """
    sev_outcomes, sev_probs = [1, 2, 7], [Fraction(1, 6), Fraction(1, 3),
                                          Fraction(1, 2)]
    ds, big_q = _clear_denominators(sev_probs)
    prime, root = ntt_primes(16, 10 ** 6)[0]
    a = np.zeros(16, dtype=np.uint64)
    a[np.array(sev_outcomes)] = [q % prime for q in big_q]
    a_hat = ntt(a, prime, root)
    b = int(a_hat[0]) * pow(ds, prime - 2, prime) % prime
    assert b == 1


@pytest.mark.parametrize('seed', range(6))
def test_exact_compound_piecewise_uniform_matches_oracle(seed):
    """Several runs of distinct lengths, singletons and gaps, against the oracle.

    The frequency clause is built from maximal runs by construction, so the
    geometric path and the short-run walk both fire within one case, and
    seed 0 starts its first run at outcome 0 (where ``b**0 = 1`` carries the
    only surviving term on the ``b == 0`` cells).
    """
    rng = random.Random(1000 + seed)
    fo, fp = _random_run_clause(rng)
    so, sp = _random_clause(rng, max_outcome=6)
    fast, d_fast, _ = exact_compound(fo, fp, so, sp)
    slow, d_slow, _ = exact_compound_poly(fo, fp, so, sp)
    assert d_fast == d_slow
    assert fast == slow


def test_geometric_path_matches_forced_walk(monkeypatch):
    """The closed-form run sum equals the per-outcome walk, exactly.

    Raising ``_GEOMETRIC_RUN_MIN`` above every run length forces the plain
    ascending walk on the same case. Both paths are integer arithmetic in
    the same field, so the numerators must agree bit for bit; this is the
    strongest available check on the geometric series and its mask.
    """
    outcomes = list(range(2, 22)) + [40]
    weights = [2] * 20 + [5]
    total = sum(weights)
    probs = [Fraction(w, total) for w in weights]
    sev_outcomes, sev_probs = [1, 3], [Fraction(1, 4), Fraction(3, 4)]
    geometric = exact_compound(outcomes, probs, sev_outcomes, sev_probs)
    monkeypatch.setattr(ntt_module, '_GEOMETRIC_RUN_MIN', 10 ** 6)
    walked = exact_compound(outcomes, probs, sev_outcomes, sev_probs)
    assert geometric[1] == walked[1]
    assert geometric[0] == walked[0]


def test_prime_dividing_severity_denominator_stays_exact():
    """``Ds`` need not be invertible: the walk covers the primes dividing it.

    The geometric path folds the severity denominator into the spectral
    point through ``Ds**(-1) mod p``, which does not exist when ``p``
    divides ``Ds``. Here the severity denominator is 4294967291, the largest
    prime below ``2**32`` and the first prime chosen at transform length 2,
    so the first CRT prime is exactly a divisor of ``Ds`` and the singular
    walk has to carry that prime's residues.
    """
    big_prime = 4294967291
    sev_probs = [Fraction(1, big_prime), Fraction(big_prime - 1, big_prime)]
    fast, d_fast, meta = exact_compound(
        [0, 1], [Fraction(1, 2), Fraction(1, 2)], [0, 1], sev_probs)
    slow, d_slow, _ = exact_compound_poly(
        [0, 1], [Fraction(1, 2), Fraction(1, 2)], [0, 1], sev_probs)
    assert meta['transform_length'] == 2
    assert d_fast == d_slow
    assert fast == slow
    # P(A = 1) = 1/2 * (p - 1)/p, which the plain walk alone can reach.
    assert Fraction(fast[1], d_fast) == \
        Fraction(1, 2) * Fraction(big_prime - 1, big_prime)


# ----------------------------------------------- [decl] ([NTT-DecL] stage)

NTT_PROGRAMS = [
    'nttagg RT.Dice dfreq [1:6] dsev [1:6] note{exact dice of dice}',
    'nttagg RT.Skewed dfreq [0 1 2] [1/4 1/2 1/4] dsev [1 2 10] [0.3 0.5 0.2]',
    'nttagg RT.Stepped as "Even Counts" dfreq [2:10:2] dsev [5]',
    'nttagg RT.Sci dfreq [1] dsev [1 2] [3e-1 7e-1] tags{exact, ntt}',
]


def _parse(uw, program):
    return uw.parser.parse(uw.lexer.tokenize(program))


def test_nttagg_parses_text_exact(underwriter):
    kind, name, spec = _parse(
        underwriter,
        'nttagg T dfreq [0 1 2] [1/4 1/2 1/4] dsev [1 2 10] [0.3 0.5 0.2]')
    assert kind == 'nttagg' and name == 'T'
    assert spec['ntt_freq_outcomes'] == [0, 1, 2]
    assert spec['ntt_freq_probs'] == [Fraction(1, 4), Fraction(1, 2),
                                      Fraction(1, 4)]
    assert spec['ntt_sev_probs'] == [Fraction(3, 10), Fraction(1, 2),
                                     Fraction(1, 5)]


def test_nttagg_uniform_default_and_ranges(underwriter):
    _, _, spec = _parse(underwriter, 'nttagg T dfreq [1:6] dsev [2:10:2]')
    assert spec['ntt_freq_outcomes'] == [1, 2, 3, 4, 5, 6]
    assert spec['ntt_freq_probs'] == [Fraction(1, 6)] * 6
    assert spec['ntt_sev_outcomes'] == [2, 4, 6, 8, 10]
    assert spec['ntt_sev_probs'] == [Fraction(1, 5)] * 5


def test_nttagg_combines_duplicates_and_sorts(underwriter):
    _, _, spec = _parse(
        underwriter, 'nttagg T dfreq [2 1 1] [1/2 1/4 1/4] dsev [1]')
    assert spec['ntt_freq_outcomes'] == [1, 2]
    assert spec['ntt_freq_probs'] == [Fraction(1, 2), Fraction(1, 2)]


def test_nttagg_rejects_non_integer_outcome(underwriter):
    with pytest.raises(ValueError, match='rescale the lattice'):
        _parse(underwriter, 'nttagg T dfreq [1.5] dsev [1]')


def test_nttagg_rejects_bad_probabilities(underwriter):
    with pytest.raises(ValueError, match='sum to exactly 1'):
        _parse(underwriter, 'nttagg T dfreq [1 2] [1/3 1/3] dsev [1]')
    with pytest.raises(ValueError, match='zero denominator'):
        _parse(underwriter, 'nttagg T dfreq [1] [1/0] dsev [1]')
    with pytest.raises(ValueError, match='outcomes against'):
        _parse(underwriter, 'nttagg T dfreq [1 2] [1/1] dsev [1]')


def test_nttagg_shortfall_message_is_exact(underwriter):
    with pytest.raises(ValueError, match='1/3'):
        _parse(underwriter, 'nttagg T dfreq [1 2] [1/3 1/3] dsev [1]')


@pytest.mark.parametrize('program', NTT_PROGRAMS)
def test_nttagg_writer_round_trip(program, underwriter):
    """parse -> write -> parse reproduces the exact spec, Fractions included."""
    from aggregate.decl_writer import spec_to_decl
    kind, name, spec = _parse(underwriter, program)
    rendered = spec_to_decl(spec, kind, name)
    kind2, name2, spec2 = _parse(underwriter, rendered)
    assert (kind2, name2) == (kind, name)
    assert spec2 == spec
    # idempotent: f(f(x)) == f(x)
    assert spec_to_decl(spec2, kind2, name2) == rendered


def test_nttagg_knowledge_write_read_round_trip(underwriter):
    """The writer output re-parses through the knowledge path (spec_to_decl)."""
    from aggregate.decl_writer import spec_to_decl
    kind, name, spec = _parse(
        underwriter, 'nttagg RT.Store dfreq [1 3] [2/5 3/5] dsev [1:4]')
    assert spec_to_decl(spec, kind, name) == \
        'nttagg RT.Store dfreq [1 3] [2/5 3/5] dsev [1 2 3 4]'


# ---------------------------------------------- [class] ([NTT-Class] stage)

def test_aggregate_ntt_builds_and_is_exact(underwriter):
    from aggregate.ntt import AggregateNTT
    a = underwriter.build('nttagg CT.Dice dfreq [1:6] dsev [1:6]')
    assert isinstance(a, AggregateNTT)
    assert a.bs == 1
    df = a.density_df
    # p_exact moved off the frame at 1.2.0a7; the frame keeps the cheap views.
    assert 'p_exact' not in df.columns
    assert df.p_exact_float.loc[36.0] == float(Fraction(1, 6 ** 7))
    assert df.p_exact_log10.loc[36.0] == pytest.approx(
        math.log10(1 / 6 ** 7))
    exact = a.p_exact
    assert exact.sum() == 1                           # exact Fraction algebra
    assert exact.loc[1.0] == Fraction(1, 36)
    assert exact.loc[36.0] == Fraction(1, 6 ** 7)


def test_aggregate_ntt_accuracy_df_and_audit(underwriter):
    a = underwriter.build('nttagg CT.Dice2 dfreq [1:6] dsev [1:6]')
    adf = a.accuracy_df
    assert list(adf.columns) == ['p_exact_float', 'p_fft', 'err', 'abs_err',
                                 'rel_err']
    assert len(adf) == len(a.xs)
    # the float pipeline is excellent in absolute terms on this lattice
    assert adf.abs_err.max() < 1e-12
    # rel_err is NaN exactly where the exact probability is zero (loss 0)
    assert np.isnan(adf.rel_err.loc[0.0])
    audit = a.ntt_audit
    assert list(audit.index) == [
        'support', 'transform_length', 'denominator_bits', 'primes_used',
        'prime_bits_min', 'prime_bits_max', 'workers', 'engine',
        'engine_version', 'engine_protocol', 'transform_seconds',
        'crt_seconds', 'limb_decode_seconds', 'max_abs_err', 'max_rel_err',
        'rel_err_buckets', 'unresolved_buckets', 'underflow_buckets',
        'noise_floor', 'exact_zero_nonzero_float']
    assert audit['support'] == 36
    assert audit['primes_used'] >= 1
    assert audit['max_abs_err'] == adf.abs_err.max()


def test_aggregate_ntt_update_pins_lattice(underwriter):
    a = underwriter.build('nttagg CT.Pin dfreq [1:6] dsev [1:6]', update=False)
    with pytest.raises(ValueError, match='bs must be 0 or 1'):
        a.update(bs=0.5)
    with pytest.raises(ValueError, match='log2 >= 6'):
        a.update(log2=5)
    a.update()   # log2=0 resolves to the required 6
    assert a.bs == 1
    assert len(a.xs) >= 37


def test_aggregate_ntt_guardrail(underwriter, monkeypatch):
    import aggregate.constants
    monkeypatch.setattr(aggregate.constants, 'NTT_MAX_SUPPORT', 30)
    with pytest.raises(ValueError, match='ntt_override'):
        underwriter.build('nttagg CT.Big dfreq [1:6] dsev [1:6]')
    a = underwriter.build('nttagg CT.Big dfreq [1:6] dsev [1:6]',
                          ntt_override=True)
    assert a.p_exact.sum() == 1


def test_aggregate_ntt_build_default_cap_names_required_log2(underwriter):
    # support 160000 needs log2 18; the underwriter's default cap is 16, so
    # the build refuses up front naming the value to pass. No heavy work runs.
    with pytest.raises(ValueError, match='log2 >= 18'):
        underwriter.build('nttagg CT.TooBig dfreq [1:400] dsev [1:400]')


# -------------------------------- [exact-moments] ([NTT-Exact-Moments] stage)

def test_aggregate_ntt_stats_df_exact_column(underwriter):
    a = underwriter.build('nttagg EM.Dice dfreq [1:6] dsev [1:6]')
    sdf = a.stats_df
    assert 'exact' in sdf.columns
    # E[A] = E[N] E[X] = 3.5 * 3.5 = 49/4 and E[A^2] = 196 exactly
    assert sdf['exact'][('agg', 'mean')] == 12.25
    assert sdf['exact'][('agg', 'ex2')] == 196.0
    assert sdf['exact'][('freq', 'mean')] == 3.5
    assert sdf['exact'][('sev', 'ex2')] == float(Fraction(91, 6))
    # the validation baseline rows carry the exact values
    assert sdf['mixed'][('agg', 'mean')] == 12.25
    assert sdf['mixed'][('agg', 'ex2')] == 196.0


def test_aggregate_ntt_validates_against_exact(underwriter):
    a = underwriter.build('nttagg EM.Val dfreq [0 1 2] [1/4 1/2 1/4] '
                          'dsev [1 2 10] [0.3 0.5 0.2]')
    # zero flags (NOT_UNREASONABLE) means validation passes against the
    # exact baseline now sitting in the mixed column
    assert not a.valid
    vdf = a.validation_df
    assert vdf is not None and len(vdf) == 3
    # exact mean: E[N] = 1, E[X] = 3/10 + 1 + 2 = 33/10
    assert a.stats_df['exact'][('agg', 'mean')] == float(Fraction(33, 10))


def test_aggregate_ntt_degenerate_severity_moments(underwriter):
    a = underwriter.build('nttagg EM.Degenerate dfreq [2] dsev [5]')
    sdf = a.stats_df
    assert sdf['exact'][('agg', 'mean')] == 10.0
    assert sdf['exact'][('sev', 'cv')] == 0.0
    assert np.isnan(sdf['exact'][('sev', 'skew')])


# ------------------------------------- [parallel] ([NTT-Parallel] stage)

def test_exact_compound_workers_match_serial():
    """The thread pool changes wall clock only, never the answer."""
    outcomes = list(range(1, 51))
    probs = [Fraction(1, 50)] * 50
    serial = exact_compound(outcomes, probs, outcomes, probs, workers=1)
    pooled = exact_compound(outcomes, probs, outcomes, probs, workers=4)
    assert serial[0] == pooled[0]
    assert serial[1] == pooled[1]
    assert serial[2]['workers'] == 1
    assert pooled[2]['workers'] == 4
    # the default is the thread pool, not serial
    default = exact_compound(outcomes, probs, outcomes, probs)
    assert default[2]['workers'] == min(4, os.cpu_count() or 1)
    assert default[0] == serial[0]


def test_aggregate_ntt_update_accepts_ntt_workers(underwriter):
    a = underwriter.build('nttagg PW.Dice dfreq [1:6] dsev [1:6]',
                          ntt_workers=2)
    assert a._ntt_meta['workers'] == 2
    assert a.p_exact.sum() == 1


# ----------------------------------------- [crt-tree] ([NTT-CRT] stage 1)

@pytest.mark.parametrize('seed', range(8))
def test_crt_tree_matches_forced_fold(monkeypatch, seed):
    """The product tree equals the linear fold it replaced, bit for bit.

    Raising ``_CRT_TREE_MIN`` above any prime count forces the quadratic
    fold on the same case. Both paths reconstruct the same integers from the
    same residues, so they must agree exactly; this is the strongest check
    on the merge stack, in the spirit of
    :func:`test_geometric_path_matches_forced_walk`.
    """
    rng = random.Random(seed)
    fo, fp = _random_clause(rng)
    so, sp = _random_clause(rng)
    # Both sides pin the Python engine: _CRT_TREE_MIN selects between the two
    # Python shapes, and with the compiled engine installed 'auto' would run
    # neither, making this comparison vacuously true.
    tree = exact_compound(fo, fp, so, sp, ntt_engine='python')
    monkeypatch.setattr(ntt_module, '_CRT_TREE_MIN', 10 ** 6)
    folded = exact_compound(fo, fp, so, sp, ntt_engine='python')
    assert tree[0] == folded[0]
    assert tree[1] == folded[1]


def test_crt_tree_matches_forced_fold_dice(monkeypatch):
    """The same pin at a size where the tree has real depth.

    The 100 x 100 uniform dice case needs 22 primes over 10,001
    coefficients, so the merge stack reaches depth 5 and its final drain
    combines partials of unequal weight, which the small oracle cases above
    do not reach.
    """
    outcomes = list(range(1, 101))
    probs = [Fraction(1, 100)] * 100
    tree = exact_compound(outcomes, probs, outcomes, probs, workers=1,
                          ntt_engine='python')
    monkeypatch.setattr(ntt_module, '_CRT_TREE_MIN', 10 ** 6)
    folded = exact_compound(outcomes, probs, outcomes, probs, workers=1,
                            ntt_engine='python')
    assert tree[0] == folded[0]
    assert tree[1] == folded[1]
    assert tree[2]['primes_used'] > 16


# ---------------------------- [accuracy-overflow] ([NTT-Accuracy-Overflow])

@pytest.mark.parametrize('n,log2', [(150, 16), (200, 16)])
def test_audit_survives_unrepresentable_tail(n, log2, underwriter):
    """``ntt_audit`` reports rather than raising when the tail leaves float.

    A uniform dice of dice at this size has exact probabilities below
    1e-328, so the relative error against one of them, a ratio of float
    noise to an unrepresentable number, has no double value. It used to
    raise ``OverflowError`` out of :attr:`accuracy_df`, which
    :attr:`ntt_audit` reads, so the headline diagnostic failed on every
    large case. Now the ratio reports an infinity and the maximum is taken
    over the resolved buckets only.
    """
    a = underwriter.build(f'nttagg OF{n} dfreq [1:{n}] dsev [1:{n}]',
                          log2=log2)
    audit = a.ntt_audit
    # The headline relative error is finite and modest, not 1e300.
    assert np.isfinite(audit['max_rel_err'])
    assert audit['max_rel_err'] < 10
    assert audit['rel_err_buckets'] > 0
    # Buckets the float pipeline cannot resolve are counted, not averaged in.
    assert audit['unresolved_buckets'] > 0
    assert audit['underflow_buckets'] > 0
    # accuracy_df itself holds infinities there rather than raising.
    adf = a.accuracy_df
    assert np.isinf(adf.rel_err.to_numpy()).any()
    assert not np.isnan(audit['max_rel_err'])


def test_safe_float_reports_infinity_not_overflow():
    """The helper converts, or saturates, but never raises."""
    huge = Fraction(10 ** 400, 1)
    assert ntt_module._safe_float(huge) == np.inf
    assert ntt_module._safe_float(-huge) == -np.inf
    assert ntt_module._safe_float(Fraction(1, 4)) == 0.25
    # Underflow is not overflow: it rounds to zero without complaint.
    assert ntt_module._safe_float(Fraction(1, 10 ** 400)) == 0.0


# ------------------------- [exact-log10-lazy] ([NTT-Exact-Log10-Lazy])

def test_p_exact_log10_sees_below_the_float_floor(underwriter):
    """``p_exact_log10`` is readable where ``p_exact_float`` has underflowed.

    A 200 x 200 uniform dice of dice puts exact probabilities near 1e-462 in
    the tail. Those are not representable as a double, so ``p_exact_float``
    reads ``0.0`` and the exact answer is invisible in the frame. The
    logarithm is formed as ``log10(numerator) - log10(denominator)``, which
    is a float for any coefficient, so it reports the magnitude.
    """
    a = underwriter.build('nttagg LG.Dice dfreq [1:200] dsev [1:200]',
                          log2=16)
    df = a.density_df
    underflowed = (df.p_exact_float == 0) & np.isfinite(df.p_exact_log10)
    assert underflowed.sum() > 0, 'expected an unrepresentable tail'
    # Those buckets carry real mass, far below the double range.
    assert df.p_exact_log10[underflowed].min() < -400
    # Exactly-zero buckets are -inf, never a finite number.
    assert np.isneginf(df.p_exact_log10.loc[0.0])
    # Where the float is a normal number, the two agree.
    normal = df.index[df.p_exact_float > 1e-200][-1]
    assert df.p_exact_log10.loc[normal] == pytest.approx(
        math.log10(float(a.p_exact.loc[normal])), rel=1e-12)
    # In the denormal range they do not, and the logarithm is the accurate
    # one: a double near 1e-323 carries only a few significant bits, so
    # log10(p_exact_float) is wrong in the first decimal place where
    # p_exact_log10, formed from the exact integers, is not. This is the
    # reason the column exists rather than being derived from the float.
    denormal = df.index[(df.p_exact_float > 0)
                        & (df.p_exact_float < 1e-310)][-1]
    exact_log = (math.log10(a.p_exact.loc[denormal].numerator)
                 - math.log10(a.p_exact.loc[denormal].denominator))
    assert df.p_exact_log10.loc[denormal] == pytest.approx(exact_log,
                                                           rel=1e-12)
    via_float = math.log10(float(a.p_exact.loc[denormal]))
    assert abs(via_float - exact_log) > 1e-3


def test_p_exact_log10_matches_the_fractions(underwriter):
    """The cheap views agree with the exact rationals bucket by bucket."""
    a = underwriter.build('nttagg LG.Small dfreq [1:6] dsev [1:6]')
    df = a.density_df
    for loss, q in a.p_exact.items():
        if q == 0:
            assert np.isneginf(df.p_exact_log10.loc[loss])
        else:
            assert df.p_exact_log10.loc[loss] == pytest.approx(
                math.log10(q.numerator) - math.log10(q.denominator))
            assert df.p_exact_float.loc[loss] == float(q)


def test_p_exact_is_cached_and_reset_by_update(underwriter):
    """The Fractions are built once, and discarded when the object updates."""
    a = underwriter.build('nttagg LG.Cache dfreq [1:6] dsev [1:6]')
    first = a.p_exact
    assert a.p_exact is first                  # cached, not rebuilt
    a.update(log2=10, bs=1)
    assert a.p_exact is not first              # discarded on update
    assert a.p_exact.sum() == 1


def test_p_exact_before_update_raises(underwriter):
    """Reading the Fractions before the exact answer exists is an error."""
    a = underwriter.build('nttagg LG.Pending dfreq [1:6] dsev [1:6]',
                          update=False)
    with pytest.raises(ValueError, match='update the object'):
        _ = a.p_exact


# ------------------------ [rust-engine] ([NTT-Rust-Integration] phase 1)
#
# The engine is optional and discovered, so every test here runs on a machine
# without it: the resolution tests fake the module, and the conformance tests
# skip. That is the normal case and it must stay the normal case.


class _FakeEngine:
    """Stand-in for an ``aggregate_ntt`` whose seam has moved."""

    def __init__(self, protocol):
        self.PROTOCOL_VERSION = protocol
        self.__version__ = '9.9.9'


@pytest.fixture
def unwarned():
    """Clear the once-per-process protocol warning latch around a test."""
    saved = ntt_module._PROTOCOL_WARNED
    ntt_module._PROTOCOL_WARNED = False
    yield
    ntt_module._PROTOCOL_WARNED = saved


def test_resolve_engine_python_never_consults_the_engine(monkeypatch):
    """``'python'`` is the bisecting switch, so it cannot fail."""
    for fake in (None, _FakeEngine(ntt_module._ENGINE_PROTOCOL),
                 _FakeEngine(99)):
        monkeypatch.setattr(ntt_module, '_engine', fake)
        assert ntt_module._resolve_engine('python') == 'python'


def test_resolve_engine_auto_falls_back_when_absent(monkeypatch):
    monkeypatch.setattr(ntt_module, '_engine', None)
    assert ntt_module._resolve_engine('auto') == 'python'
    assert ntt_module._resolve_engine() == 'python'


def test_resolve_engine_rust_raises_when_absent(monkeypatch):
    """An explicit request that cannot be met is an error, not a downgrade."""
    monkeypatch.setattr(ntt_module, '_engine', None)
    with pytest.raises(ValueError, match='not installed'):
        ntt_module._resolve_engine('rust')


def test_resolve_engine_uses_a_matching_protocol(monkeypatch):
    monkeypatch.setattr(ntt_module, '_engine',
                        _FakeEngine(ntt_module._ENGINE_PROTOCOL))
    assert ntt_module._resolve_engine('auto') == 'rust'
    assert ntt_module._resolve_engine('rust') == 'rust'


def test_resolve_engine_refuses_an_unknown_protocol(monkeypatch, unwarned):
    """A mismatched seam never computes: it warns under auto, raises under rust.

    The single most important rule of the integration. Two separately
    versioned packages drift, and a silently wrong answer must lose to a
    loud refusal.
    """
    monkeypatch.setattr(ntt_module, '_engine', _FakeEngine(99))
    with pytest.warns(UserWarning, match='seam protocol'):
        assert ntt_module._resolve_engine('auto') == 'python'
    with pytest.raises(ValueError, match='refuses aggregate-ntt 9.9.9'):
        ntt_module._resolve_engine('rust')


def test_resolve_engine_warns_once_per_process(monkeypatch, unwarned):
    """The second mismatched call falls back silently, having already said so."""
    monkeypatch.setattr(ntt_module, '_engine', _FakeEngine(99))
    with pytest.warns(UserWarning):
        ntt_module._resolve_engine('auto')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert ntt_module._resolve_engine('auto') == 'python'


def test_resolve_engine_rejects_an_unknown_request():
    with pytest.raises(ValueError, match="must be 'auto', 'python' or 'rust'"):
        ntt_module._resolve_engine('fortran')


def test_exact_compound_rejects_an_unknown_engine_before_working():
    """The request is validated before the transform phase, not after."""
    with pytest.raises(ValueError, match="must be 'auto', 'python' or 'rust'"):
        exact_compound([1], [Fraction(1)], [1], [Fraction(1)],
                       ntt_engine='fortran')


def test_provenance_keys_are_present_on_every_path():
    """Every answer says what produced it, degenerate support included."""
    for fo, fp, so, sp in (([1, 2], [Fraction(1, 2)] * 2,
                            [1, 2], [Fraction(1, 2)] * 2),
                           ([0], [Fraction(1)], [5], [Fraction(1)])):
        _, _, meta = exact_compound(fo, fp, so, sp, ntt_engine='python')
        assert meta['engine'] == 'python'
        assert meta['engine_version'] is None
        assert meta['engine_protocol'] is None
        assert meta['limb_decode_seconds'] == 0.0


def test_aggregate_ntt_update_accepts_ntt_engine(underwriter):
    """The keyword reaches ``exact_compound`` through ``update``."""
    a = underwriter.build('nttagg EN.Dice dfreq [1:6] dsev [1:6]',
                          ntt_engine='python')
    assert a._ntt_meta['engine'] == 'python'
    assert a.ntt_audit['engine'] == 'python'
    assert a.p_exact.sum() == 1


# ---- conformance, on a machine that has the engine

@pytest.mark.parametrize('seed', range(6))
def test_rust_engine_matches_python(seed):
    """The two engines reconstruct identical integers on the oracle corpus.

    The pin that makes the engine safe to use at all. Nothing here is
    compared approximately: the reconstruction is exact arithmetic, so the
    integers agree bit for bit or the engine is wrong.
    """
    pytest.importorskip('aggregate_ntt')
    rng = random.Random(seed)
    fo, fp = _random_clause(rng)
    so, sp = _random_clause(rng)
    py = exact_compound(fo, fp, so, sp, ntt_engine='python')
    rs = exact_compound(fo, fp, so, sp, ntt_engine='rust')
    assert rs[1] == py[1]
    assert rs[0] == py[0]
    assert rs[2]['engine'] == 'rust'
    assert py[2]['engine'] == 'python'


def test_rust_engine_matches_python_dice():
    """The same pin at a size where the engine's merge stack has real depth.

    The 100 x 100 uniform dice case needs 22 primes over 10,001
    coefficients, so the stack reaches depth 5 and its final drain combines
    partials of unequal weight, which the small oracle cases do not reach.
    """
    pytest.importorskip('aggregate_ntt')
    outcomes = list(range(1, 101))
    probs = [Fraction(1, 100)] * 100
    py = exact_compound(outcomes, probs, outcomes, probs,
                        ntt_engine='python')
    rs = exact_compound(outcomes, probs, outcomes, probs, ntt_engine='rust')
    assert rs[0] == py[0]
    assert rs[1] == py[1]
    # Depth comes from the Python side's 22 primes; the engine reaches the
    # same answer from about half as many, which is the width win and the
    # reason conformance compares answers and never audit records.
    assert py[2]['primes_used'] > 16
    assert rs[2]['primes_used'] == pytest.approx(
        py[2]['primes_used'] / 2, abs=2)
    assert py[2]['prime_bits_max'] == 32
    assert rs[2]['prime_bits_max'] == 63


def test_rust_engine_matches_the_polynomial_oracle():
    """The engine's answer against the reference that shares no machinery.

    ``exact_compound_poly`` uses neither a transform nor the CRT, so this
    checks the engine against something genuinely independent rather than
    against the implementation it replaces.
    """
    pytest.importorskip('aggregate_ntt')
    rng = random.Random(17)
    fo, fp = _random_clause(rng)
    so, sp = _random_clause(rng)
    rs = exact_compound(fo, fp, so, sp, ntt_engine='rust')
    slow = exact_compound_poly(fo, fp, so, sp)
    assert rs[1] == slow[1]
    assert rs[0] == slow[0]


def test_rust_engine_audit_names_itself(underwriter):
    """A saved audit record is self-describing about what computed it."""
    engine = pytest.importorskip('aggregate_ntt')
    a = underwriter.build('nttagg EN.Rust dfreq [1:6] dsev [1:6]',
                          ntt_engine='rust')
    audit = a.ntt_audit
    assert audit['engine'] == 'rust'
    assert audit['engine_version'] == engine.__version__
    assert audit['engine_protocol'] == ntt_module._ENGINE_PROTOCOL
    assert audit['limb_decode_seconds'] >= 0.0
    assert a.p_exact.sum() == 1


# -------------- [prime-width] ([Rust-Engine-Prime-Width], phase 2 contract)


@pytest.mark.parametrize('length', [2, 4, 1024, 2 ** 16])
@pytest.mark.parametrize('max_bits', [32, 64])
def test_root_of_unity_has_exact_order(length, max_bits):
    """``w`` has multiplicative order exactly ``length``, at either width.

    The defining property, checked the way the proof states it: ``w**L == 1``
    with ``w**(L/2) == -1``, which together force the order to be ``L``
    rather than a proper divisor of it.
    """
    for p, w in ntt_primes(length, 2, max_bits=max_bits)[:3]:
        assert (p - 1) % length == 0
        assert pow(w, length, p) == 1
        assert pow(w, length // 2, p) == p - 1


def test_root_of_unity_matches_the_generator_route():
    """The factoring route it replaced agrees about the order.

    The two roots are generally different elements, and both are correct:
    any primitive ``length``-th root of unity gives a valid transform, since
    the forward and inverse use the same one. What must agree is the
    property, so that is what is compared, in the spirit of
    :func:`test_crt_tree_matches_forced_fold`. The generator route is only
    exercised at 32 bits, where factoring ``c`` is cheap, which is the whole
    reason it was replaced.
    """
    length = 1024
    for p, w in ntt_primes(length, 10 ** 40)[:4]:
        c = (p - 1) // length
        classic = pow(_find_generator(p, _prime_factors(c) | {2}),
                      (p - 1) // length, p)
        for candidate in (w, classic):
            assert pow(candidate, length, p) == 1
            assert pow(candidate, length // 2, p) == p - 1
        # And both really do transform: round trip through the pair.
        values = np.arange(length, dtype=np.uint64) % p
        for candidate in (w, classic):
            back = ntt(ntt(values, p, candidate), p, candidate, inverse=True)
            assert np.array_equal(back, values)


def test_root_of_unity_is_deterministic():
    """Two runs agree, so a selection is reproducible across sessions."""
    assert ntt_primes(1024, 10 ** 40) == ntt_primes(1024, 10 ** 40)
    assert ntt_primes(1024, 10 ** 40, max_bits=64) == \
        ntt_primes(1024, 10 ** 40, max_bits=64)


def test_ntt_primes_default_width_is_32_bit():
    """The default is unchanged, which every existing caller relies on."""
    primes = ntt_primes(1024, 10 ** 60)
    assert all(p < 2 ** 32 for p, _ in primes)
    assert {p.bit_length() for p, _ in primes} == {32}


def test_ntt_primes_64_bit_draw():
    """A 64-bit draw is valid, sufficient, and about half as many primes."""
    length, bound = 1024, 10 ** 200
    narrow = ntt_primes(length, bound)
    wide = ntt_primes(length, bound, max_bits=64)
    for p, w in wide:
        assert 2 ** 63 < p < 2 ** 64
        assert is_prime_u64(p)
        assert (p - 1) % length == 0
        assert pow(w, length, p) == 1
        assert pow(w, length // 2, p) == p - 1
    product = 1
    for p, _ in wide:
        product *= p
    assert product > bound
    # Twice the bits per prime, so about half the count.
    assert len(wide) == pytest.approx(len(narrow) / 2, abs=2)


def test_ntt_primes_64_bit_at_the_shortest_length():
    """``length=2`` with 64-bit primes, the shape that stressed the old root.

    Here ``c = (p - 1) / 2`` is near ``2**63``, and the factoring route this
    replaced faced a lottery on its shape, up to minutes for a single prime.
    This must simply return.
    """
    primes = ntt_primes(2, 10 ** 30, max_bits=64)
    assert primes
    for p, w in primes:
        assert 2 ** 63 < p < 2 ** 64
        assert pow(w, 2, p) == 1
        assert w == p - 1          # the primitive square root is -1


@pytest.mark.parametrize('bad', [0, 1, 65, -4, 32.0, True, '32'])
def test_ntt_primes_rejects_a_bad_max_bits(bad):
    with pytest.raises(ValueError, match='max_bits must be an integer'):
        ntt_primes(1024, 10 ** 10, max_bits=bad)


def test_ntt_primes_rejects_a_max_bits_below_the_length():
    """No candidate can exist, so say that rather than reporting no supply."""
    with pytest.raises(ValueError, match='leaves no prime'):
        ntt_primes(1024, 10 ** 10, max_bits=10)


def test_ntt_primes_exhaustion_names_the_engine():
    """Running out of 32-bit primes points at the real remedy.

    The reader's instinct on this message is to blame the transform length.
    The supply is simply exhausted, and the only way past it is 64-bit
    primes, which the pure-Python transform cannot use.
    """
    with pytest.raises(ValueError, match='compiled aggregate-ntt engine'):
        ntt_primes(2 ** 22, 10 ** 2000)
    # The 64-bit draw of the same case succeeds, which is the point.
    assert ntt_primes(2 ** 22, 10 ** 2000, max_bits=64)


def test_ntt_refuses_a_prime_too_wide_for_numpy():
    """The guard that pays for demoting a structural limit to a parameter.

    ``_mulmod`` multiplies two residues in uint64. A 64-bit prime overflows
    that silently and returns plausible nonsense, so the refusal is the only
    thing standing between a wide prime and a wrong answer.
    """
    p, w = ntt_primes(1024, 2, max_bits=64)[0]
    values = np.arange(1024, dtype=np.uint64)
    with pytest.raises(ValueError, match='multiplies two residues'):
        ntt(values, p, w)
    with pytest.raises(ValueError, match='max_bits=64'):
        ntt(values, p, w, inverse=True)


def test_prime_width_follows_the_engine():
    """Each path draws the widest prime it can actually transform with.

    The pure-Python path is capped at 32 bits by ``_mulmod``'s uint64
    products. The engine's Montgomery field needs ``2 * prime`` to fit a
    64-bit word, so it is capped at 63, not 64, and it refuses anything
    above: offering a 64-bit prime would be a loud failure rather than a
    wrong answer, but a failure all the same.
    """
    outcomes = list(range(1, 21))
    probs = [Fraction(1, 20)] * 20
    _, _, slim = exact_compound(outcomes, probs, outcomes, probs,
                                ntt_engine='python')
    assert slim['prime_bits_max'] <= ntt_module._NUMPY_PRIME_MAX_BITS

    pytest.importorskip('aggregate_ntt')
    _, _, wide = exact_compound(outcomes, probs, outcomes, probs,
                                ntt_engine='rust')
    assert wide['prime_bits_max'] == ntt_module._ENGINE_PRIME_MAX_BITS == 63
    assert wide['primes_used'] < slim['primes_used']
