"""
Engine tests for aggregate.ntt (dev/plan-nttagg.md, [NTT-Engine] stage).

The production path :func:`aggregate.ntt.exact_compound` is checked three
independent ways: against the direct bigint polynomial reference
:func:`aggregate.ntt.exact_compound_poly` on randomized small cases, against
closed-form rational probabilities for the 100 x 100 dice-of-dice example,
and through the always-on internal checks (coefficients sum to the
denominator, nonnegative, zero beyond the support) that every call exercises.
"""

import random
from fractions import Fraction

import numpy as np
import pytest

from aggregate.ntt import (is_prime_u64, ntt, ntt_primes, exact_compound,
                           exact_compound_poly)


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
    assert meta['primes_used'] in (21, 22)


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
    assert df.p_exact.sum() == 1                      # exact Fraction algebra
    assert df.p_exact.loc[1.0] == Fraction(1, 36)
    assert df.p_exact.loc[36.0] == Fraction(1, 6 ** 7)
    assert df.p_exact_float.loc[36.0] == float(Fraction(1, 6 ** 7))


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
        'prime_bits_min', 'prime_bits_max', 'transform_seconds',
        'crt_seconds', 'max_abs_err', 'max_rel_err', 'noise_floor',
        'exact_zero_nonzero_float']
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
    assert a.density_df.p_exact.sum() == 1


def test_aggregate_ntt_build_default_cap_names_required_log2(underwriter):
    # support 160000 needs log2 18; the underwriter's default cap is 16, so
    # the build refuses up front naming the value to pass. No heavy work runs.
    with pytest.raises(ValueError, match='log2 >= 18'):
        underwriter.build('nttagg CT.TooBig dfreq [1:400] dsev [1:400]')
