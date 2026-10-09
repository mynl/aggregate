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

The engine functions here are pure and class free. :func:`exact_compound` is
the production path (vectorized numpy NTT per prime, incremental CRT);
:func:`exact_compound_poly` is the slow direct polynomial reference used by
the tests as an independent oracle. The :class:`AggregateNTT` class wiring
arrives with the DecL ``nttagg`` kind in later stages of
``dev/plan-nttagg.md``.

Access is by submodule only, matching ``ft.py`` and ``tweedie.py``: there is
no top-level re-export.
"""

import time
from fractions import Fraction
from math import lcm

import numpy as np

__all__ = ['is_prime_u64', 'ntt_primes', 'ntt', 'exact_compound',
           'exact_compound_poly']


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


def exact_compound(freq_outcomes, freq_probs, sev_outcomes, sev_probs):
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
    (which prevents circular aliasing), spectral evaluation of ``H`` walking
    the frequency outcomes in ascending order with a running power array
    (each gap bridged by vectorized square-and-multiply, so a sparse outcome
    set costs about ``log2(gap)`` array multiplies per gap), inverse NTT.

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
                'transform_seconds': 0.0, 'crt_seconds': 0.0}
        return [denominator], denominator, meta

    length = 1 << support.bit_length()  # 2**ceil(log2(support + 1))
    primes = ntt_primes(length, denominator)

    # Severity coefficient vector, zero padded to the transform length. The
    # integer weights Q_j can exceed any single prime, so reduction mod p
    # happens per prime below.
    sev_index = np.array(sev_outcomes, dtype=np.int64)

    t_transform = 0.0
    t_crt = 0.0
    # Incremental CRT state: bigint coefficients reconstructed so far and the
    # modulus they are exact mod. Python ints throughout; this phase is
    # GIL-bound bigint work (see the plan's parallelism stage).
    reconstructed = None
    modulus = 1
    for prime, root in primes:
        t0 = time.perf_counter()
        a = np.zeros(length, dtype=np.uint64)
        a[sev_index] = [q % prime for q in big_q]
        a_hat = ntt(a, prime, root)
        # Spectral evaluation of H walking the outcomes in ascending order
        # with a running power array a_hat**k, each gap bridged by vectorized
        # square-and-multiply.
        p64 = np.uint64(prime)
        h_hat = np.zeros(length, dtype=np.uint64)
        power = np.ones(length, dtype=np.uint64)
        previous_k = 0
        for k, f in zip(freq_outcomes, big_f):
            gap = k - previous_k
            base = a_hat.copy()
            while gap:
                if gap & 1:
                    power = power * base % p64
                base = base * base % p64
                gap >>= 1
            previous_k = k
            coefficient = f % prime * pow(ds, kmax - k, prime) % prime
            h_hat = (h_hat + np.uint64(coefficient) * power % p64) % p64
        residues = ntt(h_hat, prime, root, inverse=True)
        t_transform += time.perf_counter() - t0

        t0 = time.perf_counter()
        if not np.all(residues[support + 1:] == 0):
            raise AssertionError(
                f'NTT internal check failed: nonzero residue beyond the '
                f'support mod {prime}')
        head = residues[:support + 1]
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
            'transform_seconds': t_transform, 'crt_seconds': t_crt}
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
            'transform_seconds': 0.0, 'crt_seconds': 0.0}
    return accumulator, denominator, meta
