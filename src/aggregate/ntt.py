"""
Exact compound distributions via the number-theoretic transform.

This module computes a bounded discrete compound (aggregate) distribution
exactly, as rational numbers, by running the FFT convolution recipe in prime
finite fields (the number-theoretic transform, NTT) and reconstructing the
integer coefficients with the Chinese remainder theorem (CRT). It is the
computational answer to Wilson and Keich (Wilson2016a, Wilson2017), whose
aFFT-C controls relative error approximately: for bounded lattice compounds
the NTT plus CRT route is exact, at a quantifiable cost of roughly
``bound_bits / 31`` modular transforms.

The engine functions are pure and class free. :func:`exact_compound` is the
production path (vectorized numpy NTT per prime, incremental CRT);
:func:`exact_compound_poly` is the slow direct polynomial reference used by
the tests as an independent oracle. :class:`AggregateNTT` is the class behind
the DecL ``nttagg`` kind: an ordinary Aggregate running the float pipeline
side by side with the exact answer (``density_df.p_exact``, ``accuracy_df``,
``ntt_audit``, and a ``stats_df`` ``exact`` column that validation measures
against). History and design rulings: ``dev/done/plan-nttagg.md``.

Access is by submodule only, matching ``ft.py`` and ``tweedie.py``: there is
no top-level re-export.
"""

import logging
import math
import time
from fractions import Fraction
from math import lcm

import numpy as np
import pandas as pd

from .distributions import Aggregate

__all__ = ['is_prime_u64', 'ntt_primes', 'ntt', 'exact_compound',
           'exact_compound_poly', 'AggregateNTT']

logger = logging.getLogger(__name__)


# First twelve primes: a witness set making Miller-Rabin deterministic for all
# n < 3.3 * 10^24 (Sorenson and Webster 2015), far beyond the 2^64 contract.
_MILLER_RABIN_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

#: Bit-reversal permutation indices, keyed by transform length and shared
#: across primes: the permutation depends only on the length.
_BIT_REVERSAL_CACHE = {}


def is_prime_u64(n):
    """
    Deterministic primality test for ``n < 2**64``.

    Parameters
    ----------
    n : int
        Nonnegative integer to test.

    Returns
    -------
    bool
        True if ``n`` is prime.

    Notes
    -----
    Miller-Rabin with the fixed witness set of the first twelve primes, which
    is deterministic for all ``n < 3.3e24`` (Sorenson and Webster 2015), far
    beyond the ``2**64`` range this module needs. No randomness, no
    dependency: this is the fifteen-line alternative to pulling in sympy for
    ``nextprime``.
    """
    if n < 2:
        return False
    for p in _MILLER_RABIN_WITNESSES:
        if n % p == 0:
            return n == p
    # write n - 1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MILLER_RABIN_WITNESSES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime_factors(n):
    """Set of distinct prime factors of ``n`` by trial division."""
    factors = set()
    while n % 2 == 0:
        factors.add(2)
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors.add(f)
            n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors


def _find_generator(p, factors):
    """
    Smallest generator of the multiplicative group mod prime ``p``.

    ``factors`` is the set of distinct prime factors of ``p - 1``; ``g``
    generates the full group iff ``g**((p-1)/f) != 1 (mod p)`` for every
    prime factor ``f``.
    """
    for g in range(2, p):
        if all(pow(g, (p - 1) // f, p) != 1 for f in factors):
            return g
    raise ValueError(f'no generator found mod {p}; {p} is not prime')


def ntt_primes(length, bound):
    """
    NTT-friendly primes whose product exceeds ``bound``.

    Parameters
    ----------
    length : int
        Transform length, a power of two. Every returned prime satisfies
        ``p ≡ 1 (mod length)`` so a primitive ``length``-th root of unity
        exists mod ``p``.
    bound : int
        The CRT modulus requirement: the product of the returned primes
        strictly exceeds it. Pass the coefficient bound ``D``.

    Returns
    -------
    list of (int, int)
        ``(prime, root)`` pairs, largest primes first, where ``root`` is a
        primitive ``length``-th root of unity mod ``prime``. The list is the
        shortest top-down prefix whose product exceeds ``bound``.

    Raises
    ------
    ValueError
        If the supply of primes ``p = c * length + 1 < 2**32`` is
        insufficient, quoting the supply found against the bound still
        outstanding. The 32-bit ceiling is structural: products of two
        residues must fit in uint64 for the vectorized transform.

    Notes
    -----
    Scans ``p = c * length + 1`` downward from ``2**32`` (largest primes
    first minimizes the count needed). Each candidate is tested with
    :func:`is_prime_u64`; a generator ``g`` of the multiplicative group is
    found by checking ``g**((p-1)/f) != 1`` over the prime factors ``f`` of
    ``p - 1`` (trial division of ``c``, plus the factor 2), and the root
    returned is ``g**((p-1)/length) mod p``.
    """
    if length < 2 or length & (length - 1):
        raise ValueError(f'length must be a power of two >= 2, got {length}')
    primes = []
    product = 1
    for c in range((2 ** 32 - 2) // length, 0, -1):
        p = c * length + 1
        if not is_prime_u64(p):
            continue
        factors = _prime_factors(c) | {2}
        g = _find_generator(p, factors)
        root = pow(g, (p - 1) // length, p)
        primes.append((p, root))
        product *= p
        if product > bound:
            return primes
    raise ValueError(
        f'insufficient NTT prime supply: found {len(primes)} primes '
        f'p ≡ 1 (mod {length}) below 2**32 with product of '
        f'{product.bit_length()} bits, but the bound needs '
        f'{bound.bit_length()} bits. The support is too large for 32-bit '
        f'NTT primes at this transform length.')


def _bit_reversal_indices(length):
    """Bit-reversal permutation for ``length`` (a power of two), cached."""
    indices = _BIT_REVERSAL_CACHE.get(length)
    if indices is None:
        bits = length.bit_length() - 1
        i = np.arange(length, dtype=np.int64)
        rev = np.zeros(length, dtype=np.int64)
        for b in range(bits):
            rev = (rev << 1) | ((i >> b) & 1)
        indices = rev
        _BIT_REVERSAL_CACHE[length] = indices
    return indices


def _twiddle_table(length, w, p):
    """
    Powers ``w**0 .. w**(length/2 - 1) mod p`` as a uint64 array.

    Built by log-doubling: with ``P_m = [w**0, ..., w**(m-1)]`` the next
    block is ``P_m * w**m`` and ``w**m = P_m[-1] * w``, so each doubling is
    one scalar multiply and one full-array multiply. All products of two
    residues below ``2**32`` fit in uint64.
    """
    half = length // 2
    table = np.empty(max(half, 1), dtype=np.uint64)
    table[0] = 1
    size = 1
    while size < half:
        factor = int(table[size - 1]) * w % p  # w**size
        block = min(size, half - size)
        table[size:size + block] = table[:block] * np.uint64(factor) % np.uint64(p)
        size *= 2
    return table


def ntt(values, prime, root, inverse=False):
    """
    Number-theoretic transform of a vector mod a 32-bit prime.

    Parameters
    ----------
    values : array_like
        Length ``L`` (a power of two) of residues; converted to uint64 and
        reduced mod ``prime``.
    prime : int
        Odd prime below ``2**32`` with ``prime ≡ 1 (mod L)``.
    root : int
        Primitive ``L``-th root of unity mod ``prime``, as returned by
        :func:`ntt_primes` for this length.
    inverse : bool, default False
        Apply the inverse transform, including the ``1/L`` scaling.

    Returns
    -------
    numpy.ndarray
        uint64 array of length ``L``: the forward or inverse transform.

    Notes
    -----
    Iterative radix-2 decimation-in-time Cooley-Tukey over ``Z/p``: the usual
    FFT with ``exp(-2*pi*i/L)`` replaced by ``root``. All arithmetic is
    full-array numpy uint64 (products of two residues below ``2**32`` fit in
    uint64); subtraction is ``(u + p - t) % p`` to stay unsigned. The
    bit-reversal permutation is cached per length and shared across primes;
    the twiddle table is rebuilt per call (per prime and direction), a
    negligible log-doubling cost against caching hundreds of megabyte-scale
    tables. The inverse uses ``root**(p-2) mod p`` as its root and scales by
    ``L**(p-2) mod p``.
    """
    a = np.asarray(values, dtype=np.uint64) % np.uint64(prime)
    length = len(a)
    if length & (length - 1):
        raise ValueError(f'transform length must be a power of two, got {length}')
    if length == 1:
        return a.copy()
    if pow(root, length, prime) != 1 or pow(root, length // 2, prime) == 1:
        raise ValueError(
            f'root {root} is not a primitive {length}-th root of unity mod {prime}')
    a = a[_bit_reversal_indices(length)]
    w = pow(root, prime - 2, prime) if inverse else root
    table = _twiddle_table(length, w, prime)
    p64 = np.uint64(prime)
    m = 1
    while m < length:
        stride = length // (2 * m)
        wj = table[::stride][:m]
        b = a.reshape(-1, 2 * m)
        u = b[:, :m]
        v = b[:, m:]
        t = v * wj % p64
        # write v first from the untouched u, then u; the slices are disjoint
        v[:] = (u + p64 - t) % p64
        u[:] = (u + t) % p64
        m *= 2
    if inverse:
        scale = np.uint64(pow(length, prime - 2, prime))
        a = a * scale % p64
    return a


def _validate_exact_inputs(outcomes, probs, clause):
    """
    Validate and normalize one (outcomes, probabilities) clause.

    Outcomes must be nonnegative integers; probabilities positive rationals
    summing to exactly 1. Duplicate outcomes are combined by summing their
    probabilities and the result is sorted ascending, mirroring dhistogram
    semantics. Returns ``(outcomes list[int], probs list[Fraction])``.
    """
    if len(outcomes) != len(probs):
        raise ValueError(
            f'{clause}: {len(outcomes)} outcomes against {len(probs)} '
            f'probabilities')
    if len(outcomes) == 0:
        raise ValueError(f'{clause}: empty outcome list')
    combined = {}
    for x, q in zip(outcomes, probs):
        xi = int(x)
        if xi != x or xi < 0:
            raise ValueError(
                f'{clause}: outcomes must be nonnegative integers, got {x!r}; '
                f'clear denominators and rescale the lattice yourself')
        q = Fraction(q)
        if q <= 0:
            raise ValueError(f'{clause}: probabilities must be positive, got {q}')
        combined[xi] = combined.get(xi, Fraction(0)) + q
    total = sum(combined.values())
    if total != 1:
        raise ValueError(
            f'{clause}: probabilities must sum to exactly 1; '
            f'shortfall is {1 - total}')
    outs = sorted(combined)
    return outs, [combined[x] for x in outs]


def _clear_denominators(probs):
    """Return ``(D, [p * D as int])`` for exact rationals ``probs``."""
    d = lcm(*[p.denominator for p in probs])
    return d, [int(p * d) for p in probs]


# The geometric path pays one shared Fermat inverse per prime, so it only
# earns its keep on runs long enough to amortize it; shorter runs keep the
# per-outcome walk and a fully irregular outcome set pays nothing new.
_GEOMETRIC_RUN_MIN = 8


def _outcome_runs(outcomes, weights):
    """
    Decompose sorted outcomes into maximal equal-weight consecutive runs.

    Parameters
    ----------
    outcomes : sequence of int
        Strictly ascending nonnegative integers, as returned by
        :func:`_validate_exact_inputs`.
    weights : sequence of int
        The cleared integer weight of each outcome, same length.

    Returns
    -------
    list of (int, int, int)
        ``(start, run_length, weight)`` triples in ascending order, covering
        every outcome exactly once. A run is maximal: consecutive integers
        carrying one common weight.

    Notes
    -----
    Prime independent, so it is computed once in :func:`exact_compound` and
    shared by every prime. Uniform ``[1:F]`` collapses to the single triple
    ``(1, F, 1)``, which is the case the geometric spectral path exists for;
    an irregular outcome set degenerates to one singleton triple per
    outcome, recovering the plain walk.
    """
    runs = []
    start = outcomes[0]
    weight = weights[0]
    run_length = 1
    for k, f in zip(outcomes[1:], weights[1:]):
        if k == start + run_length and f == weight:
            run_length += 1
        else:
            runs.append((start, run_length, weight))
            start, weight, run_length = k, f, 1
    runs.append((start, run_length, weight))
    return runs


def _array_pow(base, exponent, prime):
    """
    Elementwise ``base**exponent mod prime`` for a uint64 residue array.

    Notes
    -----
    Vectorized binary exponentiation, ascending over the bits of
    ``exponent``, skipping the final squaring (its result would never be
    used). About ``2 * log2(exponent)`` full-array modular multiplies.
    Products of two residues below ``2**32`` fit in uint64.
    """
    p64 = np.uint64(prime)
    result = None
    current = base
    e = int(exponent)
    first = True
    while e:
        if not first:
            current = current * current % p64
        first = False
        if e & 1:
            result = current.copy() if result is None else result * current % p64
        e >>= 1
    if result is None:
        return np.ones(len(base), dtype=np.uint64)
    return result


def _spectral_walk(a_hat, freq_runs, ds, kmax, prime):
    """
    Spectral evaluation of ``H`` with the severity denominator folded in.

    Parameters
    ----------
    a_hat : numpy.ndarray
        Forward NTT of the cleared severity polynomial, uint64 residues.
    freq_runs : list of (int, int, int)
        Frequency outcome runs from :func:`_outcome_runs`.
    ds : int
        Least common denominator of the severity clause, invertible mod
        ``prime``; the caller checks and routes the divisible case to
        :func:`_spectral_walk_singular`.
    kmax : int
        Largest frequency outcome.
    prime : int
        The working prime.

    Returns
    -------
    numpy.ndarray
        ``H_hat mod prime``, uint64, same length as ``a_hat``.

    Notes
    -----
    The target is

        ``H_hat_r = sum_i F_i * Ds**(Kmax - k_i) * a_hat_r**(k_i)``.

    Folding the denominator into the spectral point, with
    ``b = a_hat * Ds**(-1) (mod p)`` from one scalar Fermat inverse, gives
    ``Ds**Kmax * b**(k_i) = Ds**(Kmax - k_i) * a_hat**(k_i)``, so

        ``H_hat = Ds**Kmax * sum_i F_i * b**(k_i)``,

    a weighted power sum of the single array ``b``, with the scalar
    ``Ds**Kmax`` factored out to one closing multiply (the per-outcome
    ``pow(ds, kmax - k, p)`` of the plain walk disappears with it).

    A maximal run of consecutive outcomes ``u..v`` carrying one weight
    ``W`` then contributes a geometric series,

        ``W * b**u * (b**(v - u + 1) - 1) * (b - 1)**(-1)``,

    costing ``log2(run_length)`` array multiplies for ``b**run_length``
    against ``run_length`` for the walk. The inverse ``(b - 1)**(-1)`` is
    one vectorized binary exponentiation to ``p - 2``, computed lazily and
    shared by every run of the prime. Runs shorter than
    ``_GEOMETRIC_RUN_MIN`` keep the per-outcome walk, which is cheaper than
    the two exponentiations the series needs.

    The ``b == 1`` cells need the degenerate series ``run_length (mod p)``,
    and this is every case rather than an edge case: ``a_hat_0 = A(1) = Ds``
    makes ``b_0 = 1`` at frequency index 0 on every prime, so any check of
    the totals exercises the mask. The ``b == 0`` cells need nothing
    special, including a run starting at outcome 0, where ``b**0 = 1``
    carries the lone surviving term.
    """
    p64 = np.uint64(prime)
    one = np.uint64(1)
    b = a_hat * np.uint64(pow(ds % prime, prime - 2, prime)) % p64
    unit = b == one
    total = np.zeros(len(b), dtype=np.uint64)
    # ``power`` holds ``b**held``; every run bridges from the exponent in
    # hand by vectorized square-and-multiply over the gap.
    power = np.ones(len(b), dtype=np.uint64)
    held = 0
    inv_bm1 = None
    for start, run_length, weight in freq_runs:
        gap = start - held
        if gap:
            base = b
            first = True
            while gap:
                if not first:
                    base = base * base % p64
                first = False
                if gap & 1:
                    power = power * base % p64
                gap >>= 1
            held = start
        w = np.uint64(weight % prime)
        if run_length < _GEOMETRIC_RUN_MIN:
            for j in range(run_length):
                total = (total + w * power % p64) % p64
                if j + 1 < run_length:
                    power = power * b % p64
                    held += 1
        else:
            if inv_bm1 is None:
                inv_bm1 = _array_pow((b + p64 - one) % p64, prime - 2, prime)
            series = (_array_pow(b, run_length, prime) + p64 - one) % p64
            series = series * inv_bm1 % p64
            series[unit] = run_length % prime
            total = (total + w * (power * series % p64) % p64) % p64
    return total * np.uint64(pow(ds, kmax, prime)) % p64


def _spectral_walk_singular(a_hat, freq_runs, ds, kmax, prime):
    """
    Spectral evaluation of ``H`` when ``prime`` divides ``ds``.

    Same contract as :func:`_spectral_walk`: returns ``H_hat mod prime``.

    Notes
    -----
    The geometric path needs ``Ds**(-1) mod p``, which does not exist when
    ``p`` divides the severity denominator ``Ds``. That requires the least
    common denominator of the severity clause to carry a prime factor of
    the form ``c * L + 1`` just below ``2**32``, so it is exotic, but it is
    reachable and the answer must still be right. This is the plain
    ascending walk on ``a_hat`` with the explicit ``Ds**(Kmax - k)``
    coefficient, under which ``Ds ≡ 0`` simply kills every term below the
    largest outcome. Runs are expanded back to their outcomes, costing one
    array multiply each, exactly as before the geometric path existed.
    """
    p64 = np.uint64(prime)
    total = np.zeros(len(a_hat), dtype=np.uint64)
    power = np.ones(len(a_hat), dtype=np.uint64)
    held = 0
    for start, run_length, weight in freq_runs:
        for j in range(run_length):
            k = start + j
            gap = k - held
            base = a_hat
            first = True
            while gap:
                if not first:
                    base = base * base % p64
                first = False
                if gap & 1:
                    power = power * base % p64
                gap >>= 1
            held = k
            coefficient = weight % prime * pow(ds, kmax - k, prime) % prime
            total = (total + np.uint64(coefficient) * power % p64) % p64
    return total


def _residues_mod_prime(prime, root, length, support, sev_index, big_q,
                        freq_runs, ds, kmax):
    """
    One prime's slice of the pipeline: ``H mod p`` on the padded lattice.

    Forward NTT of the severity polynomial, spectral evaluation of ``H``
    over the frequency outcome runs (:func:`_spectral_walk`, or
    :func:`_spectral_walk_singular` on the primes dividing ``ds``), inverse
    NTT. Pure function of its arguments plus read-only numpy work, so it is
    safe to run on a thread pool: the uint64 ufuncs release the GIL, which
    is what makes threads worth having here.

    Returns ``(head, seconds)`` where ``head`` is the uint64 residue vector
    truncated to the support. Raises if any residue beyond the support is
    nonzero (the circular-aliasing check, decisive per prime).
    """
    t0 = time.perf_counter()
    a = np.zeros(length, dtype=np.uint64)
    a[sev_index] = [q % prime for q in big_q]
    a_hat = ntt(a, prime, root)
    walk = _spectral_walk if ds % prime else _spectral_walk_singular
    h_hat = walk(a_hat, freq_runs, ds, kmax, prime)
    residues = ntt(h_hat, prime, root, inverse=True)
    if not np.all(residues[support + 1:] == 0):
        raise AssertionError(
            f'NTT internal check failed: nonzero residue beyond the '
            f'support mod {prime}')
    return residues[:support + 1], time.perf_counter() - t0


def exact_compound(freq_outcomes, freq_probs, sev_outcomes, sev_probs,
                   workers=None):
    """
    Exact bounded discrete compound distribution via NTT and CRT.

    Parameters
    ----------
    freq_outcomes : sequence of int
        Claim-count outcomes, nonnegative integers.
    freq_probs : sequence of Fraction
        Exact claim-count probabilities, summing to exactly 1.
    sev_outcomes : sequence of int
        Severity outcomes, nonnegative integers.
    sev_probs : sequence of Fraction
        Exact severity probabilities, summing to exactly 1.
    workers : int, optional
        Thread count for the per-prime transform phase. ``None`` or 1 runs
        serial. The per-prime pipeline is embarrassingly parallel and the
        uint64 ufuncs release the GIL, so threads parallelize it at zero
        pickling cost; the CRT phase is GIL-bound bigint work and stays
        serial in the calling thread either way (measured at about a tenth
        of the transform cost on dense cases; see dev/plan-nttagg.md,
        [NTT-Parallel]).

    Returns
    -------
    numerators : list of int
        ``H_n`` for ``n = 0 .. N`` where ``N = max(freq) * max(sev)``;
        ``P(A = n) = H_n / denominator`` exactly.
    denominator : int
        ``D = Df * Ds**Kmax`` where ``Df``, ``Ds`` are the least common
        denominators of the two clauses and ``Kmax = max(freq)``.
    meta : dict
        Audit record: ``support`` (``N``), ``transform_length`` (``L``),
        ``bound_bits`` (``D.bit_length()``), ``primes_used``,
        ``prime_bits_min`` / ``prime_bits_max``, ``transform_seconds``,
        ``crt_seconds``.

    Notes
    -----
    With integer weights ``F_i = f_i * Df`` and ``Q_j = q_j * Ds``, severity
    polynomial ``A(z) = sum_j Q_j z**s_j``, the cleared-denominator polynomial

        ``H(z) = Df * Ds**Kmax * G(z) = sum_i F_i * Ds**(Kmax - k_i) * A(z)**k_i``

    has nonnegative integer coefficients summing to ``D``, and ``P(A = n)``
    is ``H_n / D``. The coefficients are recovered exactly by computing
    ``H mod p`` for enough NTT primes ``p ≡ 1 (mod L)`` below ``2**32``
    (product exceeding ``D``) and reconstructing with an incremental CRT.
    Per prime: forward NTT of ``A`` padded to ``L = 2**ceil(log2(N + 1))``
    (which prevents circular aliasing), spectral evaluation of ``H`` in
    ascending outcome order with a running power array (each gap bridged by
    vectorized square-and-multiply, so a sparse outcome set costs about
    ``log2(gap)`` array multiplies per gap), inverse NTT. The frequency
    outcomes are first decomposed into maximal equal-weight consecutive
    runs by :func:`_outcome_runs`, and a long run is summed in closed form
    by :func:`_spectral_walk` at ``log2(run_length)`` array multiplies
    instead of ``run_length``: a uniform claim count is one such run, so the
    dominant phase of a dense case collapses by an order of magnitude.

    Three end-to-end checks always run and raise on failure: the
    reconstructed coefficients sum to exactly ``D``, every coefficient is
    nonnegative, and every coefficient beyond the support is zero. Together
    with the CRT modulus exceeding ``D`` these are decisive for the whole
    pipeline, not debug assertions.

    The direct polynomial reference :func:`exact_compound_poly` computes the
    same answer slowly and independently; the tests compare the two.
    """
    freq_outcomes, freq_probs = _validate_exact_inputs(
        freq_outcomes, freq_probs, 'frequency')
    sev_outcomes, sev_probs = _validate_exact_inputs(
        sev_outcomes, sev_probs, 'severity')

    df, big_f = _clear_denominators(freq_probs)
    ds, big_q = _clear_denominators(sev_probs)
    kmax = freq_outcomes[-1]
    smax = sev_outcomes[-1]
    support = kmax * smax
    denominator = df * ds ** kmax
    bound_bits = denominator.bit_length()

    # Degenerate support: the aggregate is 0 with probability 1 (all mass of
    # one clause sits at outcome 0). No transform needed.
    if support == 0:
        meta = {'support': 0, 'transform_length': 1,
                'bound_bits': bound_bits, 'primes_used': 0,
                'prime_bits_min': 0, 'prime_bits_max': 0,
                'transform_seconds': 0.0, 'crt_seconds': 0.0,
                'workers': 1}
        return [denominator], denominator, meta

    length = 1 << support.bit_length()  # 2**ceil(log2(support + 1))
    primes = ntt_primes(length, denominator)

    # Severity coefficient vector index. The integer weights Q_j can exceed
    # any single prime, so reduction mod p happens per prime in the worker.
    sev_index = np.array(sev_outcomes, dtype=np.int64)

    # The run decomposition is prime independent: compute it once and share.
    freq_runs = _outcome_runs(freq_outcomes, big_f)

    def run(pair):
        prime, root = pair
        return _residues_mod_prime(prime, root, length, support, sev_index,
                                   big_q, freq_runs, ds, kmax)

    pool = None
    if workers is not None and workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(run, primes)
    else:
        results = map(run, primes)

    t_transform = 0.0
    t_crt = 0.0
    # Incremental CRT state: bigint coefficients reconstructed so far and the
    # modulus they are exact mod. Python ints throughout, serial in this
    # thread; this phase is GIL-bound bigint work (see the plan's
    # parallelism stage).
    reconstructed = None
    modulus = 1
    try:
        for i, ((prime, _root), (head, seconds)) in enumerate(
                zip(primes, results)):
            logger.info('NTT prime %d of %d done: %d (%.3fs)',
                        i + 1, len(primes), prime, seconds)
            t_transform += seconds
            t0 = time.perf_counter()
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
            t_crt += time.perf_counter() - t0
    finally:
        if pool is not None:
            pool.shutdown()

    if modulus <= denominator:
        raise AssertionError(
            f'CRT modulus of {modulus.bit_length()} bits does not exceed the '
            f'coefficient bound of {bound_bits} bits')
    if sum(reconstructed) != denominator:
        raise AssertionError(
            'NTT internal check failed: reconstructed coefficients do not '
            'sum to the denominator')
    if min(reconstructed) < 0:
        raise AssertionError(
            'NTT internal check failed: negative reconstructed coefficient')

    prime_bits = [p.bit_length() for p, _ in primes]
    meta = {'support': support, 'transform_length': length,
            'bound_bits': bound_bits, 'primes_used': len(primes),
            'prime_bits_min': min(prime_bits),
            'prime_bits_max': max(prime_bits),
            'transform_seconds': t_transform, 'crt_seconds': t_crt,
            'workers': workers if workers else 1}
    return reconstructed, denominator, meta


def exact_compound_poly(freq_outcomes, freq_probs, sev_outcomes, sev_probs):
    """
    Direct polynomial reference for :func:`exact_compound`.

    Same inputs and same ``(numerators, denominator, meta)`` return contract
    as :func:`exact_compound`, computed by repeated sparse bigint polynomial
    multiply-accumulate in ``Z[z]`` with no transform: the blog post's moving
    window generalized to weighted outcomes. Slow and simple by design; the
    tests use it as the independent oracle, and it is public for ad hoc
    cross-checks.

    Notes
    -----
    Walks ``k = 0 .. Kmax`` keeping the running power ``A(z)**k`` as a dense
    bigint coefficient list; whenever ``k`` is a frequency outcome, adds
    ``F_i * Ds**(Kmax - k) * A(z)**k`` into the accumulator. Everything is
    exact integer arithmetic, so the only check that applies (coefficients
    sum to ``D``) still runs.
    """
    freq_outcomes, freq_probs = _validate_exact_inputs(
        freq_outcomes, freq_probs, 'frequency')
    sev_outcomes, sev_probs = _validate_exact_inputs(
        sev_outcomes, sev_probs, 'severity')

    df, big_f = _clear_denominators(freq_probs)
    ds, big_q = _clear_denominators(sev_probs)
    kmax = freq_outcomes[-1]
    smax = sev_outcomes[-1]
    support = kmax * smax
    denominator = df * ds ** kmax

    severity = [0] * (smax + 1)
    for s, q in zip(sev_outcomes, big_q):
        severity[s] = q

    weight_of = dict(zip(freq_outcomes, big_f))
    accumulator = [0] * (support + 1)
    power = [1]  # A(z)**0
    for k in range(kmax + 1):
        if k in weight_of:
            scale = weight_of[k] * ds ** (kmax - k)
            for n, coefficient in enumerate(power):
                if coefficient:
                    accumulator[n] += scale * coefficient
        if k < kmax:
            # power <- power * severity, plain bigint convolution
            new = [0] * (len(power) + smax)
            for i, ci in enumerate(power):
                if ci:
                    for j, sj in enumerate(severity):
                        if sj:
                            new[i + j] += ci * sj
            power = new

    if sum(accumulator) != denominator:
        raise AssertionError(
            'polynomial reference check failed: coefficients do not sum to '
            'the denominator')
    meta = {'support': support, 'transform_length': 0,
            'bound_bits': denominator.bit_length(), 'primes_used': 0,
            'prime_bits_min': 0, 'prime_bits_max': 0,
            'transform_seconds': 0.0, 'crt_seconds': 0.0,
            'workers': 1}
    return accumulator, denominator, meta


class AggregateNTT(Aggregate):
    """
    Compound distribution computed exactly alongside the float FFT pipeline.

    The DecL ``nttagg`` kind builds this class. It is an ordinary
    :class:`~aggregate.distributions.Aggregate` (empirical frequency,
    ``dhistogram`` severity, driven from the exact spec through correctly
    rounded ``float(Fraction)``, so the float pipeline sees bit-identical
    inputs) that additionally computes the distribution exactly, as rational
    numbers, via :func:`exact_compound`. The two answers sit side by side on
    the same integer lattice:

    - :attr:`density_df` gains ``p_exact`` (reduced :class:`~fractions.Fraction`
      per bucket, ``Fraction(0)`` off support) and ``p_exact_float`` (the
      correctly rounded binary64 of the rational);
    - :attr:`accuracy_df` compares the exact and float answers per bucket;
    - :attr:`ntt_audit` records sizes, primes, timings and error summaries.

    Parameters
    ----------
    name : str
        Object name (the knowledge-base key).
    ntt_freq_outcomes : sequence of int
        Claim-count outcomes, nonnegative integers.
    ntt_freq_probs : sequence of Fraction
        Exact claim-count probabilities summing to exactly 1.
    ntt_sev_outcomes : sequence of int
        Severity outcomes, nonnegative integers.
    ntt_sev_probs : sequence of Fraction
        Exact severity probabilities summing to exactly 1.
    label, label_map, note, hints, tags
        The usual presentation and trailer metadata, passed through to
        :class:`~aggregate.distributions.Aggregate`.

    Notes
    -----
    The float inputs are derived, never stated twice: ``float(Fraction(a, b))``
    is correctly rounded, hence bit-identical to what the float pipeline would
    compute from the same literals, so the exact-vs-float comparison isolates
    pure algorithm error (Wilson and Keich's framing; see the module
    docstring). ``update`` pins the lattice to ``bs=1`` and a transform-sized
    ``log2``; guardrails refuse monster cases up front (override the soft caps
    with ``update(..., ntt_override=True)``; the 32-bit prime supply is a hard
    limit with no override). ``nttagg`` cannot appear inside a portfolio, pnl,
    or bivariate.
    """

    def __init__(self, name, ntt_freq_outcomes=None, ntt_freq_probs=None,
                 ntt_sev_outcomes=None, ntt_sev_probs=None, label=None,
                 label_map=None, note='', hints='', tags=()):
        self.ntt_freq_outcomes, self.ntt_freq_probs = _validate_exact_inputs(
            ntt_freq_outcomes, ntt_freq_probs, 'frequency')
        self.ntt_sev_outcomes, self.ntt_sev_probs = _validate_exact_inputs(
            ntt_sev_outcomes, ntt_sev_probs, 'severity')
        # exact results, filled by update
        self._ntt_numerators = None
        self._ntt_denominator = None
        self._ntt_meta = None
        self._accuracy_df = None
        super().__init__(
            name,
            exp_en=-1,
            freq_name='empirical',
            freq_a=np.array([float(k) for k in self.ntt_freq_outcomes]),
            freq_b=np.array([float(q) for q in self.ntt_freq_probs]),
            sev_name='dhistogram',
            sev_xs=np.array([float(s) for s in self.ntt_sev_outcomes]),
            sev_ps=np.array([float(q) for q in self.ntt_sev_probs]),
            label=label, label_map=label_map, note=note, hints=hints,
            tags=tags)

    def _ntt_sizes(self):
        """Size estimates for the guardrail and the audit, cheap arithmetic.

        Returns a dict with the support ``N``, transform length ``L``,
        denominator ``D`` and its bit length ``B``, the estimated primes
        needed (about ``B / 31``), an analytic estimate of the primes
        available (the supply of ``p = c * L + 1 < 2**32`` thins as
        ``2**32 / (L * ln 2**32)`` by the prime number theorem on the
        arithmetic progression), and the estimated exact-numerator memory
        ``(N + 1) * B / 8`` bytes.
        """
        df, _ = _clear_denominators(self.ntt_freq_probs)
        ds, _ = _clear_denominators(self.ntt_sev_probs)
        kmax = self.ntt_freq_outcomes[-1]
        support = kmax * self.ntt_sev_outcomes[-1]
        length = 1 << max(support.bit_length(), 1)
        denominator = df * ds ** kmax
        bound_bits = denominator.bit_length()
        return {
            'support': support,
            'transform_length': length,
            'bound_bits': bound_bits,
            'primes_needed': -(-bound_bits // 31),
            'primes_available': int(2 ** 32 / (length * math.log(2 ** 32))),
            'bytes_estimate': (support + 1) * bound_bits // 8,
        }

    def update(self, log2=0, bs=0, **kwargs):
        """
        Run the float FFT pipeline, then attach the exact NTT answer.

        Parameters
        ----------
        log2 : int, default 0
            Grid-size cap, as for :meth:`Aggregate.update`. ``0`` uses the
            smallest transform covering the support,
            ``ceil(log2(N + 1))`` where ``N = max(freq) * max(sev)``. A cap
            below that required value raises, naming the value to pass (the
            underwriter's ``build`` supplies its default cap of 16, so a case
            needing more asks for an explicit ``build(..., log2=r)`` or a
            ``hints{log2=r}`` clause).
        bs : {0, 1}, default 0
            The lattice is the integers: anything other than 0 (choose) or 1
            raises, and 0 is pinned to 1.
        kwargs
            ``ntt_override=True`` lifts the soft guardrail caps
            (``NTT_MAX_SUPPORT``, ``NTT_MAX_BYTES`` in ``constants.py``);
            ``ntt_workers=<int>`` threads the per-prime transform phase
            (measured at 2.5x on a dense 400 x 400 case with 8 workers;
            the GIL-bound CRT stays serial); everything else passes to
            :meth:`Aggregate.update`.

        Notes
        -----
        Guardrails run before any heavy work: the soft caps raise with the
        full size estimate (support, transform length, denominator bits,
        primes needed against supply, memory) unless overridden; an
        insufficient 32-bit prime supply raises from :func:`ntt_primes` with
        no override, because past roughly ``L = 2**22`` exactness would need
        62-bit Montgomery arithmetic, which is out of scope. The float
        pipeline and the exact transform share the integer lattice, checked
        after the parent update.
        """
        from .constants import NTT_MAX_BYTES, NTT_MAX_SUPPORT
        override = bool(kwargs.pop('ntt_override', False))
        ntt_workers = kwargs.pop('ntt_workers', None)
        if bs not in (0, 1):
            raise ValueError(
                f'{self.name}: nttagg pins the exact integer lattice, so bs '
                f'must be 0 or 1, got {bs}')
        sizes = self._ntt_sizes()
        required = max(sizes['support'].bit_length(), 1)
        if log2 == 0:
            log2 = required
        elif log2 < required:
            raise ValueError(
                f"{self.name}: the exact support 0..{sizes['support']} needs "
                f"log2 >= {required}, got {log2}. Pass build(..., "
                f"log2={required}) or add hints{{log2={required}}} to the "
                f"declaration.")
        if not override and (sizes['support'] > NTT_MAX_SUPPORT
                             or sizes['bytes_estimate'] > NTT_MAX_BYTES):
            raise ValueError(
                f"{self.name}: refusing a monster exact computation: support "
                f"0..{sizes['support']} (cap {NTT_MAX_SUPPORT}), transform "
                f"length {sizes['transform_length']}, denominator of "
                f"{sizes['bound_bits']} bits, about {sizes['primes_needed']} "
                f"primes against roughly {sizes['primes_available']} "
                f"available, estimated exact-numerator memory "
                f"{sizes['bytes_estimate']:,} bytes (cap {NTT_MAX_BYTES:,}). "
                f"Pass ntt_override=True to proceed anyway.")
        rv = super().update(log2=log2, bs=1, **kwargs)
        if self.bs != 1 or len(self.xs) < sizes['support'] + 1:
            raise ValueError(
                f'{self.name}: the float pipeline left the exact lattice '
                f'(bs={self.bs}, {len(self.xs)} points for support '
                f"0..{sizes['support']}); this is a bug, please report it")
        numerators, denominator, meta = exact_compound(
            self.ntt_freq_outcomes, self.ntt_freq_probs,
            self.ntt_sev_outcomes, self.ntt_sev_probs, workers=ntt_workers)
        self._ntt_numerators = numerators
        self._ntt_denominator = denominator
        self._ntt_meta = meta
        self._accuracy_df = None
        self._attach_exact_moments()
        return rv

    def _attach_exact_moments(self):
        """Add the ``exact`` column to ``stats_df`` and rebase validation.

        The raw moments are exact rational sums: frequency and severity from
        the input rationals, aggregate as ``sum n**k H_n / D`` over the
        reconstructed numerators (cheap bigint work). ``mean`` is ``ex1``;
        ``cv`` and ``skew`` are computed from exact central moments with a
        single float square root at the end, so every entry is the correctly
        rounded float of an exact quantity. The same values then overwrite
        the ``mixed`` theoretical rows: ``mixed`` is defined as the analytic
        moments, the exact rationals are those moments computed without
        rounding, and every validation reader (``explain_validation``,
        ``validation_df``, the score terms) picks up the exact baseline
        through the one column it already reads, with no fork of
        ``_validation.py``.
        """
        components = {
            'freq': [sum(Fraction(x) ** k * q for x, q in
                         zip(self.ntt_freq_outcomes, self.ntt_freq_probs))
                     for k in (1, 2, 3)],
            'sev': [sum(Fraction(x) ** k * q for x, q in
                        zip(self.ntt_sev_outcomes, self.ntt_sev_probs))
                    for k in (1, 2, 3)],
            'agg': [Fraction(sum(n ** k * h for n, h in
                                 enumerate(self._ntt_numerators)),
                             self._ntt_denominator)
                    for k in (1, 2, 3)],
        }
        exact = pd.Series(np.nan, index=self.stats_df.index, dtype=float)
        for comp, (e1, e2, e3) in components.items():
            exact[(comp, 'ex1')] = float(e1)
            exact[(comp, 'ex2')] = float(e2)
            exact[(comp, 'ex3')] = float(e3)
            exact[(comp, 'mean')] = float(e1)
            variance = e2 - e1 * e1
            if e1 != 0 and variance > 0:
                sd = math.sqrt(variance)  # the one float rounding
                exact[(comp, 'cv')] = sd / float(e1)
                exact[(comp, 'skew')] = \
                    float(e3 - 3 * e1 * e2 + 2 * e1 ** 3) / sd ** 3
            elif variance == 0:
                exact[(comp, 'cv')] = 0.0
        self.stats_df['exact'] = exact
        moment_rows = exact.notna()
        self.stats_df.loc[moment_rows, 'mixed'] = exact[moment_rows]

    def _exact_on_grid(self):
        """``p_exact`` as a list of reduced Fractions aligned to ``self.xs``."""
        numerators = self._ntt_numerators
        denominator = self._ntt_denominator
        zero = Fraction(0)
        return [Fraction(numerators[int(x)], denominator)
                if 0 <= int(x) < len(numerators) else zero
                for x in np.asarray(self.xs)]

    @property
    def density_df(self):
        """Parent frame plus the exact columns ``p_exact`` / ``p_exact_float``.

        ``p_exact`` is an object column of reduced :class:`~fractions.Fraction`
        (``Fraction(0)`` off support); ``p_exact_float`` its correctly rounded
        binary64. Attached lazily to the parent's cached frame, so the usual
        caching pattern (rebuilt after each ``update``) carries over. Note
        ``p_total`` here is the display density (fuzz removed); the raw
        algorithm output compared in :attr:`accuracy_df` is ``agg_density``.
        """
        df = Aggregate.density_df.fget(self)
        if self._ntt_numerators is not None and 'p_exact' not in df.columns:
            exact = self._exact_on_grid()
            df['p_exact'] = exact
            df['p_exact_float'] = [float(q) for q in exact]
        return df

    @property
    def accuracy_df(self):
        """Full-grid frame comparing the exact and float answers per bucket.

        Indexed by loss over the whole output grid, columns
        ``p_exact_float``, ``p_fft`` (the raw algorithm float,
        ``agg_density``, before any display fuzz removal), ``err`` (signed,
        computed as ``float(Fraction(p_fft) - exact)`` so the subtraction is
        exact before the single rounding), ``abs_err``, and ``rel_err``
        (exact ratio, NaN where the exact value is zero).
        """
        if self._ntt_numerators is None:
            raise ValueError(
                f'{self.name}: update the object before reading accuracy_df')
        if self._accuracy_df is None:
            exact = self._exact_on_grid()
            p_fft = np.asarray(self.agg_density, dtype=float)
            err = np.array([float(Fraction(float(p)) - q)
                            for p, q in zip(p_fft, exact)])
            rel_err = np.array([float((Fraction(float(p)) - q) / q)
                                if q != 0 else np.nan
                                for p, q in zip(p_fft, exact)])
            self._accuracy_df = pd.DataFrame(
                {'p_exact_float': [float(q) for q in exact],
                 'p_fft': p_fft,
                 'err': err,
                 'abs_err': np.abs(err),
                 'rel_err': rel_err},
                index=pd.Index(np.asarray(self.xs, dtype=float), name='loss'))
        return self._accuracy_df

    @property
    def ntt_audit(self):
        """Audit Series: sizes, primes, timings, and error summaries.

        Rows: ``support``, ``transform_length``, ``denominator_bits``,
        ``primes_used``, ``prime_bits_min`` / ``prime_bits_max``,
        ``transform_seconds``, ``crt_seconds``, ``max_abs_err``,
        ``max_rel_err`` (over buckets with nonzero exact probability),
        ``noise_floor`` (max ``|p_fft|`` over exact-zero buckets), and
        ``exact_zero_nonzero_float`` (count of exact-zero buckets carrying a
        nonzero float). The raw material for a Wilson-and-Keich style
        accuracy exhibit.
        """
        if self._ntt_numerators is None:
            raise ValueError(
                f'{self.name}: update the object before reading ntt_audit')
        adf = self.accuracy_df
        meta = self._ntt_meta
        exact_zero = np.array([q == 0 for q in self._exact_on_grid()])
        p_fft = adf.p_fft.to_numpy()
        noise = np.abs(p_fft[exact_zero])
        max_rel = np.nanmax(np.abs(adf.rel_err.to_numpy())) \
            if np.any(~exact_zero) else np.nan
        return pd.Series(
            {'support': meta['support'],
             'transform_length': meta['transform_length'],
             'denominator_bits': meta['bound_bits'],
             'primes_used': meta['primes_used'],
             'prime_bits_min': meta['prime_bits_min'],
             'prime_bits_max': meta['prime_bits_max'],
             'transform_seconds': meta['transform_seconds'],
             'crt_seconds': meta['crt_seconds'],
             'max_abs_err': adf.abs_err.max(),
             'max_rel_err': max_rel,
             'noise_floor': noise.max() if len(noise) else 0.0,
             'exact_zero_nonzero_float': int(np.count_nonzero(noise))},
            name=self.name, dtype=object)
