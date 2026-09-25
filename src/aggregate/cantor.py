"""The generalized Cantor distribution, a singular continuous severity.

Provides :class:`CantorGen`, a ``scipy.stats`` continuous distribution with a
single shape parameter ``c``, the proportion removed from the middle of each
interval at every step, together with the frozen convenience instance
:data:`cantor`. ``c = 1/3`` is the classical middle-thirds Cantor law, ``c = 0``
is exactly uniform on ``[0, 1]``, and ``c`` approaching 1 degenerates to a fair
coin on the endpoints. For ``c > 0`` the law is **singular continuous**: it has
no atoms, no density, and a distribution function that is the devil's
staircase.

Nothing here is re-exported at the top-level ``aggregate`` namespace, following
the :mod:`aggregate.tweedie` precedent. Reach for it as
``from aggregate.cantor import cantor``. The user-facing surface is the DecL
severity name, which needs no import at all::

    build('agg C 1 claim sev cantor fixed')
    build('agg C 1 claim sev 3 * cantor 0.5 + 5 fixed')

Three helpers accompany the distribution. :func:`cantor_pmf` builds the exact
level-m discretization as an atom-and-mass pair, :func:`cantor_bs` gives the
natural bucket size that lattice implies, and :func:`cantor_chf` evaluates the
characteristic function from its classical infinite product.

Notes
-----
Write ``a = (1 - c) / 2`` for the length of each kept piece. The law is that of

.. math::

    X = (1 - a) \\sum_{i \\ge 1} B_i a^{i-1},
    \\qquad B_i \\ \\text{iid Bernoulli}(1/2),

equivalently the self-similar identity :math:`X \\overset{d}{=} aX + (1-a)B`
with :math:`B` an independent fair coin. Every method here is a direct reading
of that identity: the distribution function reads binary digits off ternary
(or base ``1/a``) positional structure, the quantile function runs the map
backwards, and the moments follow a closed recursion obtained by expanding the
identity binomially.
"""

import logging

import numpy as np
from scipy.special import comb
import scipy.stats as ss

__all__ = ['CantorGen', 'cantor', 'cantor_pmf', 'cantor_bs', 'cantor_chf']

logger = logging.getLogger(__name__)

#: Bits of a float64 significand, plus one guard bit. A staircase evaluation
#: stops once the running digit weight falls below this relative to the value
#: accumulated so far: further digits cannot change the rounded answer.
_SIGNIFICAND_BITS = 54

#: Hard stop on the number of leading zero digits tracked before an answer is
#: declared zero. ``2 ** -1080`` is already below the smallest subnormal, so a
#: value needing more leading digits than this rounds to zero anyway.
_MAX_LEADING = 1080

#: Hard stop on the quantile digit loop, for a shape whose kept fraction is so
#: close to 1/2 that the interval width shrinks slowly. Never reached in
#: practice: the loop exits on relative precision long before.
_MAX_QUANTILE_STEPS = 1200


def _kept_fraction(c):
    """Kept-piece length ``a`` from the removed proportion ``c``.

    Parameters
    ----------
    c : float or array_like
        Proportion removed from the middle of each interval, in ``[0, 1)``.

    Returns
    -------
    float or ndarray
        ``a = (1 - c) / 2``, in ``(0, 1/2]``.
    """
    return (1.0 - np.asarray(c, dtype=float)) / 2.0


def _staircase(x, c, survival=False):
    """Distribution or survival function by positional digit iteration.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate, broadcast against ``c``.
    c : array_like
        Removed proportion, in ``[0, 1)``.
    survival : bool, default False
        Return ``S(x) = 1 - F(x)`` rather than ``F(x)``, computed by the
        mirrored recursion rather than by subtraction, so a small survival
        probability keeps its relative accuracy in the right tail.

    Returns
    -------
    ndarray
        The broadcast-shaped result, in ``[0, 1]``.

    Notes
    -----
    With ``a = (1 - c) / 2`` the law splits self-similarly into a left copy on
    ``[0, a]`` carrying mass 1/2, a gap ``(a, 1-a)`` carrying none, and a right
    copy on ``[1-a, 1]`` carrying the other half. So

    .. math::

        F(x) = \\begin{cases}
            F(x/a)\\,/\\,2 & x < a \\\\
            1/2 & a \\le x \\le 1-a \\\\
            1/2 + F\\big((x - (1-a))/a\\big)\\,/\\,2 & x > 1-a.
        \\end{cases}

    Iterating emits one binary digit of ``F(x)`` per level, and landing in the
    gap emits a final 1 and terminates. The survival function obeys the same
    recursion with the two branches exchanged, which is the ``survival`` flag.

    The accumulator carries the digits in a float ``frac`` together with an
    integer count ``shift`` of leading zeros, and the answer is
    ``ldexp(frac, -shift)``. Halving a weight 54 times from the first nonzero
    digit would lose a result of order ``2 ** -200`` entirely; counting the
    leading zeros separately instead keeps full **relative** accuracy at every
    magnitude, which is what makes the far tail and the near-zero left tail
    trustworthy.

    Digit extraction is exact when ``a = 1/2`` (the uniform case: the maps are
    ``2x`` and ``2x - 1``, both exact in binary floating point). For any other
    shape the rescaling ``x / a`` rounds, so a point sitting within a few ulp
    of a cylinder boundary can be assigned to the neighboring cylinder. That
    costs at most a few ulp in ``F``, far below the sensitivity that matters
    downstream: ``F`` is Holder continuous with exponent
    ``log 2 / log(1/a)`` (about 0.63 at ``c = 1/3``), so a coordinate perturbed
    by ``1e-14`` moves ``F`` by about ``1e-9`` at worst.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    shape = np.broadcast_shapes(x.shape, c.shape)
    xs = np.broadcast_to(x, shape).astype(float).ravel()
    av = _kept_fraction(np.broadcast_to(c, shape)).astype(float).ravel()

    out = np.zeros(xs.size, dtype=float)
    # Outside the support the answer is immediate, and settling it here keeps
    # the loop below from spinning forever on x = 0 or x = 1, neither of which
    # ever lands in a gap.
    out[np.isnan(xs)] = np.nan
    if survival:
        out[xs <= 0.0] = 1.0
    else:
        out[xs >= 1.0] = 1.0
    live = np.flatnonzero((xs > 0.0) & (xs < 1.0))
    if live.size == 0:
        return out.reshape(shape)

    y = xs[live].copy()
    a = av[live].copy()
    frac = np.zeros(live.size, dtype=float)
    shift = np.zeros(live.size, dtype=np.int64)
    # Starting weight 1/2, so the first digit emitted carries 2 ** -(shift + 1).
    weight = np.full(live.size, 0.5, dtype=float)

    while live.size:
        hi = 1.0 - a
        left = y < a
        right = y > hi
        gap = ~(left | right)
        # A 1 digit marks the half the point is NOT in: the far branch for the
        # distribution function, the near branch for the survival function. A
        # gap point sits at the flat, whose value is the 1 digit and nothing
        # more.
        one = (left if survival else right) | gap

        # Leading zeros are counted, not shifted into the significand.
        leading = (frac == 0.0) & ~one
        shift[leading] += 1
        frac[one] += weight[one]
        weight[~leading] *= 0.5

        # Advance the surviving points one level. Both mapped values are
        # formed before either is stored, so the right branch never reads a
        # coordinate the left branch has already rescaled.
        y = np.where(left, y / a, np.where(right, (y - hi) / a, y))

        done = gap | (shift > _MAX_LEADING) | (
            (frac > 0.0) & (weight < np.ldexp(1.0, -_SIGNIFICAND_BITS)))
        if done.any():
            idx = live[done]
            # A value needing more than _MAX_LEADING leading zeros is below the
            # smallest subnormal; ldexp already returns zero there.
            out[idx] = np.ldexp(frac[done], -np.minimum(shift[done], _MAX_LEADING))
            keep = ~done
            live = live[keep]
            y = y[keep]
            a = a[keep]
            frac = frac[keep]
            shift = shift[keep]
            weight = weight[keep]

    return out.reshape(shape)


def _quantile(q, c):
    """Quantile function by the inverse digit map.

    Parameters
    ----------
    q : array_like
        Probabilities in ``[0, 1]``, broadcast against ``c``.
    c : array_like
        Removed proportion, in ``[0, 1)``.

    Returns
    -------
    ndarray
        The broadcast-shaped quantiles, in ``[0, 1]``.

    Notes
    -----
    Inverting the self-similar split gives

    .. math::

        q(p) = \\begin{cases}
            a\\, q(2p) & p \\le 1/2 \\\\
            (1-a) + a\\, q(2p - 1) & p > 1/2,
        \\end{cases}

    run here as a loop that narrows an interval ``[lo, lo + w]`` with
    ``w = a**k`` after ``k`` steps and returns ``lo``. The weak inequality on
    the left branch is what implements the scipy convention
    ``q(p) = inf{x : F(x) >= p}``: at a dyadic ``p`` the staircase is flat
    across a whole gap, and the two binary expansions of ``p`` name the two
    gap endpoints. Taking ``p = 1/2`` left yields ``a``, the **left** endpoint,
    which is the infimum. Reading the terminating expansion instead would
    return ``1 - a``, the right endpoint, and break ``ppf`` against ``cdf``.

    Digit extraction is exact: ``2p`` and ``2p - 1`` are both exact in binary
    floating point for ``p`` in ``[0, 1]`` (the second by Sterbenz), so the
    binary expansion of ``p`` is read without error and only the Horner sum
    over ``a`` powers rounds.
    """
    q = np.asarray(q, dtype=float)
    c = np.asarray(c, dtype=float)
    shape = np.broadcast_shapes(q.shape, c.shape)
    qs = np.broadcast_to(q, shape).astype(float).ravel()
    av = _kept_fraction(np.broadcast_to(c, shape)).astype(float).ravel()

    out = np.full(qs.size, np.nan, dtype=float)
    # The two endpoints are exact and would otherwise cost the loop its whole
    # iteration budget: q = 0 never takes a right branch, so its interval only
    # narrows by underflow.
    out[qs == 0.0] = 0.0
    out[qs == 1.0] = 1.0
    live = np.flatnonzero((qs > 0.0) & (qs < 1.0))
    if live.size == 0:
        return out.reshape(shape)

    p = qs[live].copy()
    a = av[live].copy()
    lo = np.zeros(live.size, dtype=float)
    width = np.ones(live.size, dtype=float)

    for _ in range(_MAX_QUANTILE_STEPS):
        if live.size == 0:
            break
        right = p > 0.5
        lo[right] += (1.0 - a[right]) * width[right]
        p = np.where(right, 2.0 * p - 1.0, 2.0 * p)
        width *= a
        # Stop once the remaining interval cannot move the rounded answer, or
        # once it has underflowed (which is the honest answer for a quantile
        # deep in the left tail: it really is that close to zero).
        done = (width == 0.0) | (width <= lo * np.ldexp(1.0, -_SIGNIFICAND_BITS))
        if done.any():
            out[live[done]] = lo[done]
            keep = ~done
            live = live[keep]
            p = p[keep]
            a = a[keep]
            lo = lo[keep]
            width = width[keep]
    if live.size:
        out[live] = lo

    return out.reshape(shape)


def _cantor_raw_moments(n, c):
    """Raw moments ``m_0 ... m_n`` of the standard Cantor law on ``[0, 1]``.

    Parameters
    ----------
    n : int
        Highest moment required.
    c : float
        Removed proportion, in ``[0, 1)``.

    Returns
    -------
    ndarray
        Array of length ``n + 1`` holding ``m_k = P X**k`` for ``k = 0 .. n``.

    Notes
    -----
    Expanding the self-similar identity :math:`X = aX + (1-a)B` binomially and
    using :math:`P B^j = 1/2` for every :math:`j \\ge 1` gives

    .. math::

        m_n = \\frac{1}{2(1 - a^n)}
              \\sum_{k=0}^{n-1} \\binom{n}{k} a^k m_k (1-a)^{n-k},
        \\qquad m_0 = 1,

    an :math:`O(n^2)` recursion that is exact in the rationals. It yields
    :math:`m_1 = 1/2` and :math:`m_2 = 1/(2(1+a))`, hence variance
    :math:`(1-a)/(4(1+a))`, which is ``1/8`` at ``c = 1/3`` and ``1/12`` at
    ``c = 0``, the uniform value, as it must be.
    """
    a = float((1.0 - c) / 2.0)
    m = np.empty(n + 1, dtype=float)
    m[0] = 1.0
    for k in range(1, n + 1):
        j = np.arange(k)
        terms = comb(k, j) * a ** j * m[:k] * (1.0 - a) ** (k - j)
        m[k] = terms.sum() / (2.0 * (1.0 - a ** k))
    return m


class CantorGen(ss.rv_continuous):
    """The generalized Cantor distribution on ``[0, 1]``.

    One shape parameter, ``c``, the proportion removed from the middle of each
    interval at every step of the construction. ``c = 1/3`` is the classical
    middle-thirds Cantor law, ``c = 0`` is exactly uniform, and ``c`` close to
    1 concentrates on the two endpoints.

    Notes
    -----
    scipy's frozen machinery supplies ``loc`` and ``scale`` in the usual way,
    so ``cantor(1/3, loc=5, scale=3)`` is the law of ``3X + 5`` and has mean
    6.5.

    For ``c > 0`` the law is singular continuous, so :meth:`_pdf` returns
    ``nan`` rather than a number: there is no density to return, and a plot's
    density panel is better left blank than filled with a lie. Nothing on the
    default aggregate pipeline asks for one, since the discretization works by
    differencing the survival function.
    """

    def _argcheck(self, c):
        """Admit ``0 <= c < 1``; ``c = 0`` is the uniform law, not a defect."""
        return (c >= 0) & (c < 1)

    def _cdf(self, x, c):
        """Distribution function, the devil's staircase. See :func:`_staircase`."""
        return _staircase(x, c, survival=False)

    def _sf(self, x, c):
        """Survival function by the mirrored recursion. See :func:`_staircase`."""
        return _staircase(x, c, survival=True)

    def _ppf(self, q, c):
        """Quantile function by the inverse digit map. See :func:`_quantile`."""
        return _quantile(q, c)

    def _isf(self, q, c):
        """Inverse survival function.

        Notes
        -----
        Uses the reflection symmetry :math:`X \\overset{d}{=} 1 - X`, which
        holds for every shape because the two kept pieces are mirror images
        carrying equal mass. So the upper ``q`` point is ``1`` minus the lower
        ``q`` point, and the convention follows along: the infimum convention
        on ``ppf`` becomes the supremum convention here, which is what scipy's
        ``isf`` means.
        """
        return 1.0 - _quantile(q, c)

    def _pdf(self, x, c):
        """``nan`` everywhere: a singular continuous law has no density.

        Notes
        -----
        For ``c > 0`` the Cantor law assigns full mass to a set of Lebesgue
        measure zero and no mass to any single point, so it is absolutely
        continuous with respect to nothing and has neither a density nor a
        probability mass function. Returning ``nan`` says exactly that.
        Returning 0 would be wrong in a way that quietly poisons any
        integration against it, and raising would make an otherwise harmless
        plot call fail. The default aggregate discretization differences the
        survival function and never asks.

        The ``c = 0`` uniform case does have a density, but reporting one only
        there would make the family discontinuous in its own shape parameter.
        Use ``scipy.stats.uniform`` when a density is what is wanted.
        """
        return np.full(np.broadcast_shapes(np.shape(x), np.shape(c)), np.nan)

    def _munp(self, n, c):
        """Exact raw moments from the self-similar recursion.

        Parameters
        ----------
        n : int or array_like of int
            Moment order.
        c : float or array_like
            Removed proportion.

        Returns
        -------
        ndarray
            ``P X**n``, broadcast over ``n`` and ``c``.

        Notes
        -----
        See :func:`_cantor_raw_moments` for the recursion. Every integer
        moment is available, and each is a rational function of ``a``, so this
        never falls back on numerical integration, which for a law with no
        density would have nothing to integrate.
        """
        shape = np.broadcast_shapes(np.shape(n), np.shape(c))
        nf = np.broadcast_to(n, shape).ravel()
        cf = np.broadcast_to(c, shape).ravel()
        out = np.empty(nf.size, dtype=float)
        for i, (ni, ci) in enumerate(zip(nf, cf)):
            out[i] = _cantor_raw_moments(int(ni), float(ci))[int(ni)]
        return out.reshape(shape)

    def _stats(self, c):
        """Mean, variance, skewness and excess kurtosis in closed form.

        Notes
        -----
        Mean is ``1/2`` and variance ``(1-a)/(4(1+a))`` for every shape.
        Skewness is exactly zero by the reflection symmetry
        :math:`X \\overset{d}{=} 1 - X`, so it is returned as zero rather than
        as the difference of two nearly equal third moments. Excess kurtosis
        comes from the exact raw moments, with the mean folded in as 1/2.
        """
        c = np.asarray(c, dtype=float)
        a = _kept_fraction(c)
        var = (1.0 - a) / (4.0 * (1.0 + a))
        # Fourth central moment with mean 1/2 throughout:
        # mu4 = m4 - 4*(1/2)*m3 + 6*(1/4)*m2 - 3*(1/16).
        mu4 = (self._munp(4, c) - 2.0 * self._munp(3, c)
               + 1.5 * self._munp(2, c) - 3.0 / 16.0)
        return (np.full_like(var, 0.5), var, np.zeros_like(var),
                mu4 / var ** 2 - 3.0)


#: Frozen-friendly instance. ``cantor(c)`` freezes a shape, ``cantor(c, loc=,
#: scale=)`` freezes a scaled and shifted one, and ``cantor.cdf(x, c)`` calls
#: it unfrozen. The default shape is not supplied by scipy, so write
#: ``cantor(1/3)`` for the classical law.
cantor = CantorGen(a=0.0, b=1.0, name='cantor', shapes='c')


def cantor_pmf(log2_points, q=3):
    """Atoms and masses of the exact level-m Cantor discretization.

    Parameters
    ----------
    log2_points : int
        The level ``m``. The result has ``2 ** m`` atoms.
    q : int, default 3
        Reciprocal of the kept fraction, ``q = 1/a = 2/(1-c)``. Must be an
        integer at least 2. ``q = 3`` is the classical middle-thirds law
        (``c = 1/3``), ``q = 4`` removes the middle half (``c = 1/2``), and
        ``q = 2`` is the uniform law (``c = 0``).

    Returns
    -------
    (xs, ps) : tuple of ndarray
        Sorted atoms in ``[0, 1]`` and their equal masses ``2 ** -m``.

    Raises
    ------
    ValueError
        If ``log2_points`` is negative or ``q`` is not an integer at least 2.

    Notes
    -----
    Truncating the series at level ``m`` gives
    ``X_m = (1-a) sum_{i<=m} B_i a**(i-1)``, which is **exactly** a discrete
    uniform law on ``2 ** m`` points. When ``1/a = q`` is an integer those
    points are ``(q-1) * j / q**m`` for the integers ``j`` whose base-``q``
    digits are all 0 or 1, so they all sit on the lattice of multiples of
    ``q ** -m``. That is the whole content of :func:`cantor_bs`, and it is why
    a Cantor severity wants a base-``q`` bucket size rather than the usual
    binary fraction.

    ``X_m`` converges to ``X`` geometrically: the discarded tail is bounded by
    ``a ** m``, so moments agree to that order. The atoms are the level-``m``
    left endpoints of the surviving intervals.

    For a shape whose ``1/a`` is not an integer the level-``m`` atoms share no
    common lattice and this construction does not apply; consume the frozen
    distribution directly instead.

    Feed the result to a discrete severity, either as
    ``Severity('dhistogram', sev_xs=xs, sev_ps=ps)`` or through a ``dsev``
    program, when an exactly lattice-valued Cantor approximation is wanted.
    """
    m = int(log2_points)
    if m < 0:
        raise ValueError(f'log2_points must be nonnegative, got {log2_points}')
    if int(q) != q:
        raise ValueError(f'q must be an integer at least 2, got {q}')
    q = int(q)
    if q < 2:
        raise ValueError(f'q must be an integer at least 2, got {q}')

    count = 1 << m
    words = np.zeros(count, dtype=np.int64)
    index = np.arange(count, dtype=np.int64)
    place = 1
    for bit in range(m):
        words += ((index >> bit) & 1) * place
        place *= q
    xs = np.sort((q - 1) * words / float(q ** m))
    ps = np.full(count, np.ldexp(1.0, -m), dtype=float)
    return xs, ps


def cantor_bs(m, c=1.0 / 3.0, scale=1.0):
    """Natural bucket size for a Cantor severity at level ``m``.

    Parameters
    ----------
    m : int
        Level. Finer levels resolve more of the self-similar structure.
    c : float, default 1/3
        Removed proportion. ``2 / (1 - c)`` must be an integer.
    scale : float, default 1.0
        Scale of the severity, so the answer is in the severity's own units.

    Returns
    -------
    float
        ``scale / q ** m`` with ``q = 2 / (1 - c)``.

    Raises
    ------
    ValueError
        If ``c`` is outside ``[0, 1)``, or if ``2 / (1 - c)`` is not an
        integer, in which case no lattice contains the level-``m`` atoms and
        there is no natural bucket size to give.

    Notes
    -----
    The standing house guidance is that ``bs`` should be a binary fraction.
    A Cantor severity is the documented exception. Its level-``m`` atoms are
    ``scale * (q-1) * j / q**m`` (see :func:`cantor_pmf`), so a bucket size of
    ``scale / q**m`` puts every atom exactly on a bucket and makes
    cdf-difference discretization reproduce the exact level-``m`` masses. For
    the classical ``c = 1/3`` that means a **ternary** ``bs = scale / 3**m``,
    which is the rule break; for ``c = 1/2`` it is ``scale / 4**m``, binary
    after all; and for the uniform ``c = 0`` it is ``scale / 2**m``, binary
    trivially.

    A binary ``bs`` is not wrong on a Cantor severity, merely blurry at the
    finest scales: the law is atomless and its distribution function is
    continuous, so cdf differencing converges either way. The natural bucket is
    exact at level ``m``, and it is the choice that makes a picture of the
    self-similarity look like one.

    The bound method :meth:`aggregate._severity.SeverityCantor.natural_bs`
    calls this with the severity's own shape and scale already filled in.
    """
    c = float(c)
    if not (0.0 <= c < 1.0):
        raise ValueError(f'c must satisfy 0 <= c < 1, got {c}')
    q = 2.0 / (1.0 - c)
    if not np.isclose(q, round(q)):
        raise ValueError(
            f'c = {c} gives q = 2/(1-c) = {q}, which is not an integer, so the '
            f'level-m atoms share no lattice and there is no natural bucket '
            f'size. Integer q needs c = 1 - 2/q: 0 (q=2), 1/3 (q=3), 1/2 '
            f'(q=4), 3/5 (q=5), and so on.')
    return float(scale) / float(round(q)) ** int(m)


def cantor_chf(t, c=1.0 / 3.0, tol=1e-17):
    """Characteristic function of the standard Cantor law on ``[0, 1]``.

    Parameters
    ----------
    t : array_like
        Real arguments.
    c : float, default 1/3
        Removed proportion, in ``[0, 1)``.
    tol : float, default 1e-17
        Truncate the product once a factor's argument is below this, where
        ``cos`` is 1 to float precision.

    Returns
    -------
    ndarray of complex
        ``P exp(i t X)``.

    Notes
    -----
    Summing the digit representation term by term, each independent Bernoulli
    contributes ``(1 + exp(i t (1-a) a**(k-1))) / 2``, which factors as a phase
    times a cosine. Collecting the phases geometrically gives

    .. math::

        \\varphi(t) = e^{it/2} \\prod_{k \\ge 1}
                      \\cos\\big((1-a) a^{k-1} t / 2\\big),

    and at ``c = 1/3`` this is the classical
    :math:`e^{it/2} \\prod \\cos(t / 3^k)`. The factors approach 1
    geometrically, so the truncation is cheap and the error is of the order of
    the first omitted argument squared.

    Provided for :class:`aggregate.ft.FourierTools` experiments: the Cantor law
    is the standard example of a distribution whose characteristic function
    does not vanish at infinity, which makes it an honest stress test of direct
    inversion.
    """
    t = np.asarray(t, dtype=float)
    a = float(_kept_fraction(c))
    scale = np.max(np.abs(t)) if t.size else 0.0
    out = np.exp(0.5j * t)
    half = (1.0 - a) / 2.0
    k = 0
    while half * a ** k * scale > tol:
        out = out * np.cos(half * a ** k * t)
        k += 1
    return out
