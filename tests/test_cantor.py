"""Unit tests for the generalized Cantor distribution.

Covers the module surface of ``aggregate.cantor``: the distribution
:class:`aggregate.cantor.CantorGen` and its frozen instance ``cantor``, and the
three helpers ``cantor_pmf``, ``cantor_bs`` and ``cantor_chf``. The DecL
severity that sits on top of it is tested separately.

The distribution function is checked against an independent oracle, the
correctly rounded ternary implementation in ``hacks/cantor_cdf.py``, whose
scalar path is transcribed below as :func:`oracle_cdf`. It uses exact integer
ternary remainders, so it is right to the last bit, and it works only for the
classical ``c = 1/3``. That is the whole reason it lives here as a fixture
rather than in the library: the shipped implementation is the general-shape
digit iteration, and the oracle exists to hold it honest at the one shape where
an exact answer is cheap.

The two agree **bit for bit** on generic points. They part company only at
exact ternary rationals and at points of the Cantor set itself, where the
general iteration's rescaling by ``1/a`` rounds and can place a boundary point
in the neighboring cylinder. The observed gap there is about ``3e-11``, which
is the Holder bound at work: ``F`` has exponent ``log 2 / log 3`` about 0.631,
so a coordinate perturbed by one ulp moves ``F`` by roughly
``(1e-16) ** 0.631``, about ``1e-10``. See ``dev/done/plan-cantor.md``, the
"Numerical honesty" section.
"""
from __future__ import annotations

from functools import lru_cache
from math import inf, isnan, ldexp, nextafter, ulp

import numpy as np
import pytest

from aggregate.cantor import (CantorGen, cantor, cantor_bs, cantor_chf,
                              cantor_pmf)
from aggregate.cantor import _cantor_raw_moments

# ---------------------------------------------------------------------------
# The oracle: correctly rounded middle-thirds cdf, transcribed from
# hacks/cantor_cdf.py. Exact integer ternary digits, c = 1/3 only.
# ---------------------------------------------------------------------------

_DIGITS = 6
_RADIX = 3 ** _DIGITS
_PREFIX_MASK = (1 << _DIGITS) - 1
_TERMINAL = 1 << _DIGITS
_ROUND_AT = 1 << 53


@lru_cache(maxsize=1)
def _lookup() -> tuple:
    """Encode a six-bit cdf prefix and a termination flag for each cell."""
    entries = []
    for cell in range(_RADIX):
        remainder, prefix, place = cell, 0, _RADIX
        for position in range(_DIGITS):
            place //= 3
            digit, remainder = divmod(remainder, place)
            prefix = (prefix << 1) | int(digit != 0)
            if digit == 1:
                # Pad with zero bits after the first middle-third digit.
                prefix <<= _DIGITS - position - 1
                entries.append(prefix | _TERMINAL)
                break
        else:
            entries.append(prefix)
    return tuple(entries)


def oracle_cdf(x: float) -> float:
    """Correctly rounded middle-thirds Cantor cdf of ``float(x)``.

    Parameters
    ----------
    x : float
        A single value. Exact integer remainders prevent any loss of ternary
        digits near a cylinder boundary, so the answer is the correctly
        rounded cdf of the *represented* float, not of the decimal or rational
        that may have been intended.

    Returns
    -------
    float
    """
    value = float(x)
    if isnan(value):
        return value
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0

    remainder, denominator = value.as_integer_ratio()
    table, prefix, exponent = _lookup(), 0, 0
    while True:
        cell, remainder = divmod(_RADIX * remainder, denominator)
        entry = table[cell]
        prefix = (prefix << _DIGITS) | (entry & _PREFIX_MASK)
        exponent -= _DIGITS
        rounded = float(prefix)
        if entry & _TERMINAL:
            return ldexp(rounded, exponent)
        if prefix >= _ROUND_AT:
            # The unresolved tail is strictly between zero and one unit; only
            # an integer prefix rounded downward at a tie needs fixing.
            if prefix - int(rounded) == ulp(rounded) / 2:
                rounded = nextafter(rounded, inf)
            return ldexp(rounded, exponent)


def _oracle(xs):
    """Vectorize :func:`oracle_cdf` over an array."""
    return np.array([oracle_cdf(v) for v in np.asarray(xs, dtype=float).ravel()])


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

#: Generic points, where the two implementations agree bit for bit.
GENERIC = np.concatenate([
    np.random.default_rng(20260925).random(5000),
    # Midpoints of the removed gaps at every level up to 8, all of which
    # resolve on the first level that sees them.
    np.array([(1 + 2 * j) / (2 * 3 ** k) for k in range(1, 8) for j in range(3 ** k)]),
    # Subnormal and near-underflow magnitudes.
    np.array([5e-324, 1e-320, 1e-300, 1e-100, 1e-30, 1e-10]),
    # Approaches to the upper endpoint.
    1 - np.array([1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-16]),
])

#: Cylinder boundaries: exact ternary rationals and the level-10 Cantor atoms.
#: The general digit iteration rounds here, bounded by the Holder estimate.
BOUNDARY = np.concatenate([
    np.array([j / 3 ** k for k in range(1, 9) for j in range(1, 3 ** k)]),
    cantor_pmf(10)[0],
])

#: The Holder bound on how far a one-ulp coordinate error can move ``F`` at
#: ``c = 1/3``: ``(1e-16) ** (log 2 / log 3)`` is about ``1e-10``, and a few
#: levels of accumulated rounding buy a little more.
HOLDER_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Distribution function
# ---------------------------------------------------------------------------

def test_cdf_matches_the_exact_ternary_oracle_bit_for_bit():
    """On generic points the general iteration is the correctly rounded cdf."""
    mine = cantor.cdf(GENERIC, 1 / 3)
    assert np.array_equal(mine, _oracle(GENERIC))


def test_cdf_matches_the_oracle_within_the_holder_bound_on_boundaries():
    """At cylinder boundaries the two part company, by the Holder estimate.

    A ternary rational sits exactly on a cylinder wall, so a one-ulp rescaling
    error puts it on the other side. ``F`` is Holder continuous with exponent
    ``log 2 / log 3``, which caps the damage well below anything the
    discretization can see.
    """
    gap = np.abs(cantor.cdf(BOUNDARY, 1 / 3) - _oracle(BOUNDARY))
    assert gap.max() < HOLDER_TOLERANCE
    # Not merely bounded: the boundary set really is where they differ.
    assert gap.max() > 0


def test_cdf_is_the_identity_at_c_zero():
    """``c = 0`` is the uniform law, and the digit maps are exact in binary."""
    x = np.concatenate([np.linspace(0, 1, 1025),
                        np.random.default_rng(1).random(5000)])
    assert np.array_equal(cantor.cdf(x, 0.0), x)


def test_cdf_is_symmetric_and_monotone():
    """``F(x) + F(1-x) = 1`` from ``X =d 1 - X``, and ``F`` never decreases."""
    x = np.sort(np.random.default_rng(2).random(20000))
    for c in (0.0, 1 / 3, 0.5, 0.9):
        f = cantor.cdf(x, c)
        assert np.all(np.diff(f) >= 0)
        assert np.allclose(f + cantor.cdf(1 - x, c), 1.0, atol=HOLDER_TOLERANCE)


@pytest.mark.parametrize('c', [0.0, 1 / 3, 0.5, 0.75, 0.9])
def test_cdf_is_one_half_across_the_central_gap(c):
    """The staircase is flat at 1/2 on ``[a, 1-a]``, endpoints included."""
    a = (1 - c) / 2
    xs = np.array([a, 1 - a, (a + 1 - a) / 2])
    assert np.allclose(cantor.cdf(xs, c), 0.5, atol=HOLDER_TOLERANCE)


def test_survival_is_the_complement_and_keeps_relative_accuracy():
    """``sf`` is ``1 - cdf``, computed by the mirrored recursion not subtraction."""
    x = np.random.default_rng(3).random(5000)
    assert np.allclose(cantor.sf(x, 1 / 3), 1 - cantor.cdf(x, 1 / 3), atol=1e-15)
    # Deep in the right tail the complement would have lost every digit; the
    # mirrored recursion keeps them, and symmetry gives the check.
    far = 1 - np.array([1e-6, 1e-9, 1e-12, 1e-15])
    assert np.array_equal(cantor.sf(far, 1 / 3),
                          cantor.cdf(1 - far, 1 / 3))


def test_cdf_outside_the_support():
    """Zero below, one above, nan propagates."""
    assert np.array_equal(cantor.cdf([-1.0, 0.0, 1.0, 2.0], 1 / 3),
                          np.array([0.0, 0.0, 1.0, 1.0]))
    assert np.isnan(cantor.cdf(np.nan, 1 / 3))


# ---------------------------------------------------------------------------
# Quantile function
# ---------------------------------------------------------------------------

def test_quantile_inverts_the_cdf():
    """``F(q(p)) = p`` to the Holder tolerance, which is the achievable bound.

    ``q(p)`` is right to a few ulp, and a coordinate perturbed by an ulp moves
    ``F`` by about ``1e-10``. Asking for more here would be asking ``F`` to be
    Lipschitz, which it is not.
    """
    p = np.random.default_rng(4).random(5000)
    for c in (0.0, 1 / 3, 0.5):
        assert np.allclose(cantor.cdf(cantor.ppf(p, c), c), p,
                           atol=HOLDER_TOLERANCE)


@pytest.mark.parametrize('c', [1 / 3, 0.5, 0.8])
def test_quantile_takes_the_left_endpoint_of_a_gap(c):
    """scipy's convention ``q(p) = inf{x : F(x) >= p}`` at a dyadic ``p``.

    ``F`` is flat at 1/2 across the whole central gap, so ``q(1/2)`` is
    ambiguous up to the gap's width. The infimum convention picks the left
    endpoint ``a``, not the right endpoint ``1-a``.
    """
    a = (1 - c) / 2
    assert cantor.ppf(0.5, c) == pytest.approx(a, abs=1e-15)
    # The same one level down, in each half.
    assert cantor.ppf(0.25, c) == pytest.approx(a * a, abs=1e-15)
    assert cantor.ppf(0.75, c) == pytest.approx(1 - a + a * a, abs=1e-15)


def test_quantile_endpoints_and_the_uniform_case():
    """``q(0) = 0``, ``q(1) = 1``, and ``c = 0`` is the identity."""
    assert cantor.ppf(0.0, 1 / 3) == 0.0
    assert cantor.ppf(1.0, 1 / 3) == 1.0
    p = np.random.default_rng(5).random(2000)
    assert np.allclose(cantor.ppf(p, 0.0), p, atol=1e-15)


def test_inverse_survival_mirrors_the_quantile():
    """``isf(q) = 1 - ppf(q)`` by the reflection symmetry."""
    p = np.random.default_rng(6).random(2000)
    assert np.allclose(cantor.isf(p, 1 / 3), cantor.ppf(1 - p, 1 / 3), atol=1e-15)


def test_rvs_is_exact_inverse_transform_sampling():
    """Sampling rides ``_ppf``, so the sample moments land where they should."""
    sample = cantor(1 / 3).rvs(size=200_000, random_state=11)
    assert sample.min() >= 0.0 and sample.max() <= 1.0
    assert sample.mean() == pytest.approx(0.5, abs=2e-3)
    assert sample.var() == pytest.approx(0.125, abs=2e-3)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('c', [0.0, 1 / 3, 0.5, 0.6, 0.9])
def test_moments_match_the_closed_forms(c):
    """``m1 = 1/2``, ``m2 = 1/(2(1+a))``, ``var = (1-a)/(4(1+a))``."""
    a = (1 - c) / 2
    m = _cantor_raw_moments(2, c)
    assert m[1] == pytest.approx(0.5, rel=1e-14)
    assert m[2] == pytest.approx(1 / (2 * (1 + a)), rel=1e-14)
    mean, var, skew, _ = cantor.stats(c, moments='mvsk')
    assert mean == pytest.approx(0.5, rel=1e-14)
    assert var == pytest.approx((1 - a) / (4 * (1 + a)), rel=1e-14)
    assert skew == 0.0


def test_the_two_landmark_variances():
    """1/8 at the middle thirds, 1/12 at the uniform limit."""
    assert cantor.var(1 / 3) == pytest.approx(0.125, rel=1e-14)
    assert cantor.var(0.0) == pytest.approx(1 / 12, rel=1e-14)


def test_moments_agree_with_the_level_m_atoms():
    """The level-m discretization's moments converge geometrically to the exact.

    ``X_m`` drops the tail beyond level ``m``, which is bounded by ``a ** m``,
    so the mean gap is exactly ``a ** m / 2`` at every level.
    """
    exact = _cantor_raw_moments(2, 1 / 3)
    for m in (4, 8, 12):
        xs, ps = cantor_pmf(m)
        assert np.dot(xs, ps) == pytest.approx((1 - (1 / 3) ** m) / 2, rel=1e-13)
        assert abs(np.dot(xs ** 2, ps) - exact[2]) < 3.0 ** -m


def test_loc_and_scale_flow_through():
    """scipy's frozen machinery gives ``3X + 5`` from ``loc`` and ``scale``."""
    fz = cantor(1 / 3, loc=5, scale=3)
    assert fz.mean() == pytest.approx(6.5, rel=1e-14)
    assert fz.var() == pytest.approx(9 * 0.125, rel=1e-13)
    assert fz.support() == (5.0, 8.0)
    assert fz.cdf(6.5) == pytest.approx(0.5, abs=HOLDER_TOLERANCE)


def test_cv_parameterization_round_trips():
    """``cv**2 = (1-a)/(1+a)`` inverts to ``a = (1-cv**2)/(1+cv**2)``.

    The attainable range is ``[1/sqrt(3), 1)``: the uniform law at one end,
    the fair coin on the endpoints at the other.
    """
    for cv in (1 / np.sqrt(3), 0.7, 0.8, 0.95):
        a = (1 - cv ** 2) / (1 + cv ** 2)
        c = 1 - 2 * a
        mean, var = cantor.stats(c, moments='mv')
        assert np.sqrt(var) / mean == pytest.approx(cv, rel=1e-13)


# ---------------------------------------------------------------------------
# Density, or the lack of one
# ---------------------------------------------------------------------------

def test_there_is_no_density():
    """A singular continuous law has none, and ``nan`` says so."""
    assert np.all(np.isnan(cantor.pdf([0.1, 0.5, 0.9], 1 / 3)))
    assert np.all(np.isnan(cantor(1 / 3, loc=5, scale=3).pdf([5.5, 7.0])))


def test_argcheck_admits_zero_and_rejects_one():
    """``0 <= c < 1``. ``c = 0`` is the uniform law, not a defect."""
    gen = CantorGen(a=0.0, b=1.0, name='cantor', shapes='c')
    assert bool(gen._argcheck(np.float64(0.0)))
    assert bool(gen._argcheck(np.float64(0.999)))
    assert not bool(gen._argcheck(np.float64(1.0)))
    assert not bool(gen._argcheck(np.float64(-0.1)))


# ---------------------------------------------------------------------------
# cantor_pmf, cantor_bs, cantor_chf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('q', [2, 3, 4, 5])
def test_cantor_pmf_is_a_lattice_law_with_equal_masses(q):
    """``2 ** m`` atoms of mass ``2 ** -m``, all on the ``q ** -m`` lattice."""
    m = 6
    xs, ps = cantor_pmf(m, q)
    assert len(xs) == 1 << m
    assert ps.sum() == pytest.approx(1.0, rel=1e-15)
    assert np.all(ps == 2.0 ** -m)
    assert np.all(np.diff(xs) > 0)
    assert xs[0] == 0.0
    # Every atom is an integer multiple of the natural bucket size.
    bs = cantor_bs(m, 1 - 2 / q)
    assert np.allclose(xs / bs, np.round(xs / bs), atol=1e-9)


def test_cantor_pmf_at_q_two_is_the_uniform_lattice():
    """``q = 2`` removes nothing, so the atoms fill the lattice."""
    xs, _ = cantor_pmf(5, 2)
    assert np.allclose(xs, np.arange(32) / 32)


def test_cantor_pmf_rejects_bad_arguments():
    with pytest.raises(ValueError):
        cantor_pmf(-1)
    with pytest.raises(ValueError):
        cantor_pmf(4, 1)
    with pytest.raises(ValueError):
        cantor_pmf(4, 2.5)


@pytest.mark.parametrize('c, q', [(0.0, 2), (1 / 3, 3), (0.5, 4), (0.6, 5)])
def test_cantor_bs_is_the_reciprocal_integer_lattice(c, q):
    """``bs = scale / q ** m`` with ``q = 2/(1-c)``, scale carried through."""
    assert cantor_bs(6, c) == pytest.approx(1.0 / q ** 6, rel=1e-15)
    assert cantor_bs(6, c, scale=7.0) == pytest.approx(7.0 / q ** 6, rel=1e-15)


def test_cantor_bs_landmarks():
    """The two values the shipped library entries carry."""
    assert cantor_bs(10, 1 / 3) == 1 / 59049
    assert cantor_bs(6, 0.5) == 1 / 4096


def test_cantor_bs_refuses_a_shape_with_no_lattice():
    """No integer ``q`` means the level-m atoms share no lattice at all."""
    with pytest.raises(ValueError, match='not an integer'):
        cantor_bs(4, 0.4)
    with pytest.raises(ValueError, match='0 <= c < 1'):
        cantor_bs(4, 1.0)


def test_cantor_chf_against_the_empirical_transform():
    """The product formula reproduces the level-m atoms' transform."""
    t = np.array([0.0, 0.5, 1.0, 5.0, -3.0, 40.0])
    xs, ps = cantor_pmf(14)
    empirical = np.array([np.dot(ps, np.exp(1j * ti * xs)) for ti in t])
    assert np.allclose(cantor_chf(t, 1 / 3), empirical, atol=1e-3)
    assert cantor_chf(np.array([0.0]), 1 / 3)[0] == pytest.approx(1.0)


def test_cantor_chf_does_not_vanish_at_infinity():
    """The classical fact that makes this law a stress test for inversion.

    At ``t = 2 pi 3 ** k`` the first ``k`` cosines are ``cos`` of a multiple of
    ``2 pi``, hence 1, and what is left is the tail product
    ``prod_{i >= 1} cos(2 pi / 3 ** i)``, the same for every ``k``. So
    ``|phi|`` returns to one fixed positive value arbitrarily far out and the
    transform has no decay at all: the Riemann-Lebesgue lemma does not apply,
    because there is no density for it to apply to.
    """
    base = 2 * np.pi
    values = np.abs(cantor_chf(base * 3.0 ** np.arange(1, 8), 1 / 3))
    limit = np.prod(np.cos(2 * np.pi / 3.0 ** np.arange(1, 40)))
    assert np.allclose(values, values[0], rtol=1e-6)
    assert values[0] == pytest.approx(abs(limit), rel=1e-9)
    assert values[0] > 0.3
