"""Sparre-Andersen renewal count distribution.

Distribution of ``N(T)``, the number of renewals by time ``T`` for iid
waiting times ``W``, computed by an FFT/Plancherel method with **no inverse
FFTs**: ``{N(T) >= k} = {S_k <= T}``, so ``P(N=k) = F^{*k}(T) - F^{*(k+1)}(T)``,
and each ``F^{*k}(T)`` is a frequency-domain inner product of ``phat^k`` with
the transform of the exponentially tilted indicator of ``[0, T]`` -- two
forward rffts total, then one vector multiply and one dot product per ``k``.
Exponential tilting suppresses circular wrap-around, so the grid need only
modestly exceed ``[0, T]``.

Zero-wait mass (claim clusters) is factored out exactly: with
``p0 = P(W <= 0)`` the count decomposes as ``N = sum_{i<=M+1} G_i - 1`` where
``M`` is the renewal count of the conditional positive-wait law ``W | W > 0``
and ``G ~ Geometric{1,2,...}`` with ``P(G=g) = (1-p0) p0^(g-1)`` -- see
:func:`geometric_batch_compose`. Defective waits (total mass < 1) terminate
the renewal process; the count pmf stays proper.

Ported and generalized (t = 1 -> T, pre-discretized pmf input, exact-lattice
readout, zero/negative/defective mass handling) from the author's ``sparre``
notes library (``sparre/renewal.py``, ``theory.md`` sections 1-5, 7, 11).

A leaf module: numpy / scipy / pandas and the ``discretize_severities``
kernel only; it never imports ``_aggregate`` or the parser.
"""

import logging
import warnings
from fractions import Fraction
from math import ceil, gcd, log, log2 as _log2

import numpy as np
import pandas as pd
import scipy.stats as ss

from ._aggregate_compute import discretize_severities

logger = logging.getLogger(__name__)

__all__ = ['renewal_count_pmf', 'geometric_batch_compose', 'wait_grid',
           'wait_count_pmf']

# Series-tail cutoff: contributions below this are dropped when sizing the
# count support (kmax, geometric tail length). Well under float64 resolution.
TAIL_EPS = 1e-17
# Surface (don't block) unusually long count series; cost is kmax vector
# multiply-adds on the transform grid.
KMAX_WARN = 100_000
# Wait-grid log2 window for the continuous sizing path.
WAIT_LOG2_FLOOR = 16
WAIT_LOG2_CAP = 24


def _kmax_from_pmf(pm, bs, T, z=10.0):
    """Series length for the renewal count from a discretized wait pmf.

    ``kmax = ceil(T/mu + z*sqrt(T*var/mu^3) + z)`` from the renewal CLT
    ``N(T) ~ Normal(T/mu, T*var/mu^3)``, using the pmf's own (conditional)
    moments; for total mass ``q < 1`` (defective or truncated-at-T waits)
    ``P(N >= k) <= q^k`` caps the series at ``log(TAIL_EPS)/log(q)``.

    Parameters
    ----------
    pm : ndarray
        Discretized waiting-time pmf on ``arange(len(pm)) * bs``; may be
        sub-stochastic.
    bs : float
        Wait-grid bucket size.
    T : float
        Horizon (years).
    z : float, default 10.0
        Number of CLT standard deviations of headroom.

    Returns
    -------
    int
        Series length ``kmax >= 1``.
    """
    q = float(np.sum(pm))
    if q <= 0:
        raise ValueError('waiting-time pmf has no positive mass')
    x = np.arange(len(pm)) * bs
    mu = float(x @ pm) / q
    if mu <= 0:
        raise ValueError('waiting-time pmf has zero conditional mean -- '
                         'all wait mass is at 0')
    var = max(float((x * x) @ pm) / q - mu * mu, 0.0)
    kmax = int(ceil(T / mu + z * np.sqrt(T * var / mu ** 3) + z))
    if q < 1.0:
        # defective / truncated tail: P(N >= k) <= q^k
        kmax = min(kmax, int(ceil(log(TAIL_EPS) / log(q))))
    kmax = max(kmax, 1)
    if kmax > KMAX_WARN:
        warnings.warn(
            f'renewal count series length kmax = {kmax} > {KMAX_WARN}; '
            f'the wait law is very short relative to the horizon T = {T}. '
            f'Cost is kmax vector multiply-adds -- still feasible, but slow.')
    return kmax


def renewal_count_pmf(pm, bs, T, *, lattice=False, z=10.0, tilt_total=20.0,
                      kmax=None):
    """PMF of the renewal count ``N(T)`` for an already-discretized wait pmf.

    Parameters
    ----------
    pm : ndarray
        Waiting-time pmf on the grid ``arange(len(pm)) * bs``. May be
        sub-stochastic (defective wait / mass truncated beyond ``T`` --
        for the count by ``T`` the two are equivalent: mass beyond ``T``
        cannot contribute to any ``F^{*k}(T)``). Mass at index 0 is treated
        as genuine zero waits -- factor atoms out first via
        :func:`geometric_batch_compose` for the exact cluster geometry
        (O(h) rounding mass of a continuous law may stay).
    bs : float
        Bucket size ``h``; the grid length ``len(pm) * bs`` must exceed
        ``T`` (25%+ headroom recommended -- the tilt damps wrap-around).
    T : float
        Horizon: count renewals in ``[0, T]``.
    lattice : bool, default False
        Readout convention at ``T``. ``False``: half-bucket endpoint
        correction (continuous waits, restores O(h^2) convergence).
        ``True``: full final bucket (exact-lattice discrete waits -- an
        atom exactly at ``T`` belongs to ``P(S_k <= T)``).
    z : float, default 10.0
        CLT standard deviations for the automatic ``kmax``.
    tilt_total : float, default 20.0
        ``theta * L``: circular wrap ``r`` is damped by ``exp(-r*tilt_total)``.
    kmax : int, optional
        Series length; computed via :func:`_kmax_from_pmf` when omitted.

    Returns
    -------
    k : ndarray of int
        ``0..kmax``.
    pN : ndarray
        ``P(N(T) = k)``.

    Notes
    -----
    Plancherel: for real vectors ``a, b`` of length ``m`` with rffts
    ``ahat, bhat``, ``sum_j a_j b_j = Re(sum_w w_w ahat_w conj(bhat_w))/m``
    with rfft weights ``w = [1, 2, ..., 2, 1]`` (last weight 1 only for even
    ``m``). Here ``a = (tilted pm)^{*k}`` -- never materialized; only its
    transform ``phat^k`` is accumulated -- and ``b`` is the tilted indicator
    of ``[0, T]``, so each ``F^{*k}(T)`` costs one vector multiply and one
    dot product. See ``sparre`` theory.md sections 1-5.
    """
    pm = np.asarray(pm, dtype=float)
    m = len(pm)
    L = m * bs
    if L <= T:
        raise ValueError(f'grid length m*bs = {L} must exceed T = {T}')
    n1 = int(round(T / bs))
    if n1 < 1:
        raise ValueError(f'T = {T} must be at least one bucket bs = {bs}')
    if kmax is None:
        kmax = _kmax_from_pmf(pm, bs, T, z=z)

    # exponential tilt: wrap r is damped by exp(-r * tilt_total)
    theta = tilt_total / L
    x = np.arange(m) * bs
    phat = np.fft.rfft(pm * np.exp(-theta * x))

    # tilted indicator of [0, T]; Plancherel weights for rfft folded in
    ind = np.zeros(m)
    ind[:n1 + 1] = np.exp(theta * x[:n1 + 1])
    if not lattice:
        # The lattice partial sum through bucket j reads F^{*k}(j*h + h/2),
        # so a full last bucket lands at T + h/2 (O(h) bias). Weighting it
        # by 1/2 + (T - n1*h)/h centers the readout exactly at T; the
        # weight is 1/2 when the lattice hits T exactly (n1*h = T).
        ind[n1] *= 0.5 + (T - n1 * bs) / bs
    w = np.full(phat.shape, 2.0)
    w[0] = 1.0
    if m % 2 == 0:
        w[-1] = 1.0
    c = w * np.conj(np.fft.rfft(ind)) / m

    # accumulate phat^k elementwise; each F^{*k}(T) is one dot product
    F = np.empty(kmax + 2)
    F[0] = 1.0
    v = np.ones_like(phat)
    for k in range(1, kmax + 2):
        v *= phat
        F[k] = np.real(v @ c)

    pN = np.clip(-np.diff(F), 0.0, None)   # P(N=k) = F_k - F_{k+1}
    return np.arange(kmax + 1), pN


def geometric_batch_compose(pmf_M, p0):
    """Recompose the count pmf after factoring out zero-wait mass ``p0``.

    Parameters
    ----------
    pmf_M : ndarray
        PMF of ``M``, the renewal count of the conditional positive-wait
        law ``W | W > 0`` (index = count).
    p0 : float
        ``P(W <= 0)`` -- atoms at 0 plus any collapsed negative mass;
        ``0 <= p0 < 1``.

    Returns
    -------
    ndarray
        PMF of the full count ``N`` (index = count), length extended to
        cover the geometric cluster tail to below ``TAIL_EPS``.

    Notes
    -----
    Group the iid wait draws into batches: a maximal run of zero waits
    followed by one positive wait. Batch sizes are iid
    ``G ~ Geometric{1,2,...}``, ``P(G=g) = (1-p0) p0^(g-1)``; the clock
    advances only at each batch's terminal positive wait. With ``M``
    positive arrivals in ``[0, T]``, batches ``1..M`` renew entirely by
    ``T``, and the *zero-run of batch M+1* also rides at epoch
    ``S_M <= T`` (equivalently: leading zeros renew at time 0), so

    .. math:: N = \\sum_{i=1}^{M+1} G_i - 1.

    Dropping the boundary batch (``N = sum_{i<=M} G_i``) is wrong -- check
    ``W ~ dwait [0 1] [.5 .5]``, ``T=1``: exact ``P(N=m) = m/2^{m+1}``.
    Conditioning on ``M = m`` and writing ``n = sum_{i<=m+1} g_i - 1``:

    .. math::

        P(N=n \\mid M=m) = \\binom{n}{m} (1-p0)^{m+1} p0^{n-m},
        \\qquad n \\ge m,

    the ``nbinom(m+1, 1-p0)`` law on ``j = n - m`` (failures before the
    (m+1)-th success), evaluated here via ``scipy.stats.nbinom`` for
    numerical stability. ``p0 = 0`` returns ``pmf_M`` unchanged. The same
    formula holds for defective waits: the zeros preceding the terminating
    draw still count, and their run length is independent of the
    terminator type (late arrival or defect).
    """
    pmf_M = np.asarray(pmf_M, dtype=float)
    if not 0.0 <= p0 < 1.0:
        raise ValueError(f'p0 = {p0} must be in [0, 1)')
    if p0 == 0.0:
        return pmf_M.copy()
    mmax = len(pmf_M) - 1
    # beyond j extra counts the geometric tail is < p0^j: cut below TAIL_EPS
    extra = int(ceil(log(TAIL_EPS) / log(p0)))
    nmax = mmax + extra
    pN = np.zeros(nmax + 1)
    for m_, w_ in enumerate(pmf_M):
        if w_ == 0.0:
            continue
        j = np.arange(nmax - m_ + 1)
        pN[m_:] += w_ * ss.nbinom.pmf(j, m_ + 1, 1.0 - p0)
    return pN


def _fraction_gcd(a, b):
    """gcd of two ``Fraction``s: ``gcd(p1/q1, p2/q2) = gcd(p1 q2, p2 q1)/(q1 q2)``."""
    return Fraction(gcd(a.numerator * b.denominator,
                        b.numerator * a.denominator),
                    a.denominator * b.denominator)


def _lattice_step(atoms, T, max_n1=3 << 22):
    """Exact common lattice step of ``atoms + [T]``, or None.

    Rationalizes each value (``Fraction.limit_denominator(1e9)``), takes the
    fraction gcd, and verifies every value is an integer multiple of the
    step within 1e-9 relative tolerance. Returns None for incommensurable
    (or absurdly fine, ``T/step > max_n1``) supports.
    """
    values = [float(a) for a in atoms if a > 0] + [float(T)]
    try:
        g = Fraction(0)
        for val in values:
            g = _fraction_gcd(g, Fraction(val).limit_denominator(10 ** 9))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    if g <= 0:
        return None
    step = float(g)
    for val in values:
        k = round(val / step)
        if k < 0 or abs(val - k * step) > 1e-9 * max(1.0, abs(val)):
            return None
    n1 = round(T / step)
    if n1 < 1 or n1 > max_n1:
        return None
    return step


def wait_grid(mu, sigma, T, atoms=None, *, z=10.0, tol=1e-9, kappa=64.0,
              hard_atoms=None):
    """Size the wait-discretization grid ``(bs, log2)`` for horizon ``T``.

    Parameters
    ----------
    mu, sigma : float
        A-priori conditional positive-wait mean and standard deviation
        (exact, pre-discretization; used for sizing only).
    T : float
        Horizon (years).
    atoms : array-like, optional
        Discrete wait support (positive atoms). When commensurable with
        ``T`` the exact lattice step overrides the continuous sizing rules.
    z : float, default 10.0
        CLT standard deviations for the a-priori ``kmax`` estimate.
    tol : float, default 1e-9
        Total count-error budget for the accuracy rule ``kmax * h^2 <= tol``
        (per-convolution error is O(h^2) with the half-bucket readout,
        accumulating ~linearly in k).
    kappa : float, default 64.0
        Shape-resolution divisor: ``h <= mu/kappa`` (and ``sigma/kappa``
        when ``sigma >= mu/10``; below that a near-deterministic continuous
        wait would blow up log2 -- truly discrete waits take the
        exact-lattice path).
    hard_atoms : array-like, optional
        Isolated atoms of an otherwise continuous wait law (e.g. the cap
        atom at ``y`` of a layered wait). When commensurable with ``T`` the
        continuous bucket size is refined to ``step / 2**j`` (the coarsest
        such at or below the continuous bound) so every hard atom sits
        exactly on the lattice -- phase alignment only; the grid stays fine
        and the half-bucket readout still applies. Incommensurable atoms
        fall back to the continuous sizing (O(h/2) placement smear).

    Returns
    -------
    bs : float
        Bucket size; ``n1 * bs == T`` exactly (integer ``n1``).
    log2 : int
        Grid size exponent (``m = 2**log2`` buckets, ``m * bs = 4/3 T``
        coverage headroom on the continuous path).
    lattice : bool
        True when the exact-lattice override applies (full-bucket readout).
    bs_df : DataFrame
        One row per sizing constraint (``coverage``, ``shape``,
        ``accuracy``, ``exact_lattice``, ``hard_atom_snap``, ``log2_cap``)
        with the implied ``bs``/``log2``/``n1``, a ``feasible`` flag and a
        ``selected`` marker on the binding constraint; the final choice and
        the a-priori ``kmax`` estimate are in ``bs_df.attrs``.
    """
    if mu <= 0:
        raise ValueError(f'conditional wait mean mu = {mu} must be positive')
    if T <= 0:
        raise ValueError(f'horizon T = {T} must be positive')
    var = sigma * sigma
    kmax_est = int(ceil(T / mu + z * np.sqrt(T * var / mu ** 3) + z))

    def _implied(h):
        """(log2_raw, log2_clamped, n1, bs) implied by a bucket-size bound h."""
        l2_raw = int(ceil(_log2(T * 4.0 / 3.0 / h)))
        l2 = min(max(l2_raw, WAIT_LOG2_FLOOR), WAIT_LOG2_CAP)
        n1_ = int(0.75 * (1 << l2))
        return l2_raw, l2, n1_, T / n1_

    h_shape = mu / kappa
    if sigma >= mu / 10.0:
        h_shape = min(h_shape, sigma / kappa)
    h_acc = np.sqrt(tol / kmax_est)
    h_cand = min(h_shape, h_acc)
    binding = 'shape' if h_shape <= h_acc else 'accuracy'
    l2_raw, l2_final, n1_final, bs_final = _implied(h_cand)
    clamped = l2_raw != l2_final

    step = _lattice_step(atoms, T) if atoms is not None else None
    lattice = step is not None
    if lattice:
        n1_lat = round(T / step)
        # small grids are fine on the exact lattice (no shape/accuracy
        # constraint -- atoms sit exactly on the grid); keep the same 4/3
        # wrap headroom, no 2^16 floor.
        l2_lat = max(int(ceil(_log2(n1_lat * 4.0 / 3.0))), 3)
        if l2_lat > WAIT_LOG2_CAP:
            warnings.warn(
                f'exact wait lattice needs log2 = {l2_lat} > {WAIT_LOG2_CAP}; '
                f'falling back to continuous grid sizing.')
            lattice = False
        else:
            n1_final, l2_final, bs_final = n1_lat, l2_lat, T / n1_lat

    # hard-atom snap: phase-align the fine continuous grid so isolated atoms
    # (a layered wait's cap at y) land exactly on the lattice. bs = step/2^j
    # (coarsest at or below the continuous bound) keeps n1 = T/bs exactly
    # integer since T is a multiple of step.
    snapped = False
    snap_step = None
    if not lattice and hard_atoms is not None and len(hard_atoms) > 0:
        snap_step = _lattice_step(hard_atoms, T)
        if snap_step is not None:
            j = max(int(ceil(_log2(snap_step / h_cand))), 0)
            bs_snap = snap_step / (1 << j)
            n1_snap = round(T / bs_snap)
            l2_snap = max(int(ceil(_log2(n1_snap * 4.0 / 3.0))), 3)
            if l2_snap <= WAIT_LOG2_CAP:
                snapped = True
                bs_final, n1_final, l2_final = bs_snap, n1_snap, l2_snap
            else:
                snap_step = None

    if not lattice and l2_raw > WAIT_LOG2_CAP:
        warnings.warn(
            f'wait grid wants log2 = {l2_raw}, capped at {WAIT_LOG2_CAP}; '
            f'bs = {bs_final:.6g} is coarser than the '
            f'{binding} rule wants ({h_cand:.6g}) -- count accuracy may '
            f'fall short of tol = {tol}.')

    selected = ('exact_lattice' if lattice
                else 'hard_atom_snap' if snapped
                else 'log2_cap' if clamped
                else binding)

    def _row(name, bs_, l2_, n1_, feasible, note):
        return dict(constraint=name, bs=bs_, log2=l2_, n1=n1_,
                    feasible=feasible, selected=name == selected, note=note)

    sh = _implied(h_shape)
    ac = _implied(h_acc)
    rows = [
        _row('coverage', bs_final, l2_final, n1_final, True,
             f'm*bs = 4/3*T = {4 * T / 3:.6g}; lattice hits T exactly'),
        _row('shape', sh[3], sh[1], sh[2], True,
             f'h <= mu/kappa = {mu / kappa:.6g}'
             + (f', sigma/kappa = {sigma / kappa:.6g}'
                if sigma >= mu / 10.0 else ' (sigma floor skipped)')),
        _row('accuracy', ac[3], ac[1], ac[2], True,
             f'h <= sqrt(tol/kmax_est) = {h_acc:.6g}, kmax_est = {kmax_est}'),
        _row('exact_lattice', bs_final if lattice else np.nan,
             l2_final if lattice else np.nan,
             n1_final if lattice else np.nan,
             lattice,
             'atoms and T commensurable' if lattice
             else 'no commensurable discrete support'),
        _row('hard_atom_snap', bs_final if snapped else np.nan,
             l2_final if snapped else np.nan,
             n1_final if snapped else np.nan,
             snapped,
             f'bs = step/2^j, step = {snap_step:.6g}' if snapped
             else 'no hard atoms commensurable with T'),
        _row('log2_cap', bs_final, l2_final, n1_final,
             WAIT_LOG2_FLOOR <= l2_raw <= WAIT_LOG2_CAP,
             f'raw log2 = {l2_raw}, window [{WAIT_LOG2_FLOOR}, '
             f'{WAIT_LOG2_CAP}]'),
    ]
    bs_df = pd.DataFrame(rows).set_index('constraint')
    bs_df.attrs.update(bs=bs_final, log2=l2_final, n1=n1_final,
                       lattice=lattice, snapped=snapped, kmax_est=kmax_est,
                       z=z, tol=tol, kappa=kappa)
    return bs_final, l2_final, lattice, bs_df


def wait_count_pmf(components, weights, T, *, z=10.0, tilt_total=20.0,
                   tol=1e-9, kappa=64.0, grid=None):
    """Count pmf ``P(N(T) = k)`` for a (mixture) waiting-time law.

    The orchestrator: discretize each component on a common grid
    (:func:`discretize_severities`, rounding scheme, no normalization),
    apply the wait post-passes (zero/negative-mass split, defective window
    mask, truncation at ``T``), weight-combine, renormalize by ``1 - p0``,
    run the FFT kernel for the positive-wait count ``M``, and recompose the
    zero-wait clusters exactly (:func:`geometric_batch_compose`).

    Parameters
    ----------
    components : list of (sev, lb, ub, conditional)
        ``sev`` is a :class:`Severity`-like object (``cdf``, ``sf``,
        ``moms()``). ``conditional=True`` components have any splice window
        already applied inside ``sev`` (nothing to do here);
        ``conditional=False`` components are built WITHOUT ``lb/ub`` (the
        severity layer would renormalize) and the window ``[lb, ub]`` is
        masked on the grid -- the escaping mass is the defect (terminating
        renewal process).
    weights : array-like
        Mixture weights. May sum to less than 1: the shortfall is defect
        mass (e.g. a defective ``dwait ... !``).
    T : float
        Horizon (years).
    z, tilt_total, tol, kappa : float
        Passed through to :func:`wait_grid` / :func:`renewal_count_pmf`.
    grid : (bs, log2, closed), optional
        Override the automatic grid sizing (used by ``convergence_check``);
        the third element is the readout convention (closed interval when
        True). ``bs_df`` then records the automatic recommendation, not
        the override.

    Returns
    -------
    k : ndarray of int
        ``0..kmax``.
    pN : ndarray
        ``P(N(T) = k)``.
    info : dict
        Diagnostics: ``bs``, ``log2``, ``n1``, ``lattice``, ``snapped``
        (hard-atom phase alignment -- closed readout, like the exact
        lattice), ``p0``, ``defect``, ``kmax``, ``bs_df``, ``pm`` (the
        combined conditional positive-wait pmf actually fed to the
        kernel).

    Notes
    -----
    Sizing moments are the exact pre-discretization mixture moments
    (``sev.moms()``); for unconditional components they include mass
    outside the window (sizing only -- the pmf itself is windowed
    exactly). ``kmax`` is recomputed from the *discretized* conditional
    moments so splices and truncation are respected.
    """
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    if len(components) != len(weights):
        raise ValueError('components and weights must have equal length')
    total_w = float(weights.sum())
    if total_w > 1 + 1e-12:
        raise ValueError(f'mixture weights sum to {total_w} > 1')

    # exact zero-wait / negative-wait mass per component (pre-discretization)
    just_below_0 = np.nextafter(0.0, -1.0)
    p0s, negs, defects = [], [], []
    for sev, lb, ub, conditional in components:
        if conditional:
            p0s.append(float(sev.cdf(0.0)))
            negs.append(float(sev.cdf(just_below_0)))
            defects.append(0.0)
        else:
            # window applied at grid level below; mass at <= 0 collapses to
            # the batch atom only if the window retains 0
            retains_zero = lb <= 0.0 <= ub
            c0 = float(sev.cdf(0.0))
            p0s.append(c0 if retains_zero else 0.0)
            negs.append(float(sev.cdf(just_below_0)) if retains_zero else 0.0)
            # model-level defect: mass escaping [lb, ub] (boundary at lb
            # kept open below 0 handling aside -- diagnostic only)
            defects.append(max(0.0, 1.0 - (float(sev.cdf(ub))
                                           - float(sev.cdf(np.nextafter(lb, -np.inf))))))
    p0 = float(weights @ np.asarray(p0s))
    neg = float(weights @ np.asarray(negs))
    defect = float(weights @ np.asarray(defects)) + (1.0 - total_w)
    if p0 >= 1.0 - 1e-15 or p0 + defect >= 1.0 - 1e-15:
        raise ValueError(
            f'all waiting-time mass is at or below 0 (P(W<=0) = {p0:.6g}, '
            f'defect = {defect:.6g}) -- the renewal count is undefined')
    if neg > 0:
        warnings.warn(
            f'negative waiting-time mass {neg:.6g} collapsed into the '
            f'zero-wait atom; resulting P(W=0) = {p0:.6g} (a batch of '
            f'simultaneous claims)')

    # a-priori sizing moments of the conditional positive wait
    moms = np.array([list(sev.moms())[:2] for sev, *_ in components],
                    dtype=float)
    q_pos = max(1.0 - p0 - defect, 1e-15)
    m1 = float(weights @ moms[:, 0]) / q_pos
    m2 = float(weights @ moms[:, 1]) / q_pos
    sigma = np.sqrt(max(m2 - m1 * m1, 0.0))

    # exact-lattice detection: all-discrete mixture support
    atoms = None
    if all(getattr(sev, 'sev_kind', '') == 'dhistogram'
           for sev, *_ in components):
        atoms = np.concatenate([np.asarray(sev.fz.xk, dtype=float)
                                for sev, *_ in components])

    # cap atoms of layered components (finite limit with escaping mass) are
    # genuine atoms in an otherwise continuous law -- phase-align the grid
    hard_atoms = [float(sev.limit) for sev, *_ in components
                  if np.isfinite(getattr(sev, 'limit', np.inf))
                  and getattr(sev, 'pdetach', 0.0) > 0.0] or None

    bs, log2, lattice, bs_df = wait_grid(m1, sigma, T, atoms,
                                         z=z, tol=tol, kappa=kappa,
                                         hard_atoms=hard_atoms)
    # readout convention: closed interval [0, T] whenever wait atoms sit
    # exactly on the lattice -- exact-lattice grids AND snapped grids (a
    # layered cap at y): sums S_k landing exactly at T belong to N(T).
    # Half-bucket centering would halve that atom mass (a O(1) error);
    # the closed readout's O(h/2) continuous overshoot is negligible on
    # the fine snapped grid.
    snapped = bool(bs_df.attrs.get('snapped', False))
    closed = lattice or snapped
    if grid is not None:
        bs, log2, closed = grid
        lattice, snapped = closed, False
    m = 1 << log2
    n1 = int(round(T / bs))
    xs = np.arange(m) * bs

    beds = discretize_severities([sev for sev, *_ in components], xs, bs,
                                 sev_calc='discrete',
                                 discretization_calc='survival',
                                 normalize=False)

    pm = np.zeros(m)
    for (sev, lb, ub, conditional), w_, p0_, bed in zip(
            components, weights, p0s, beds):
        bed = np.asarray(bed, dtype=float)
        if not conditional:
            # defective window: zero the buckets outside [lb, ub]; the
            # escaping mass is the defect (never renormalized)
            eps = 1e-12 * max(1.0, abs(lb), abs(ub))
            bed[(xs < lb - eps) | (xs > ub + eps)] = 0.0
        # p0 split: bucket 0 = P(W <= h/2); remove the exact atom-and-below
        # mass, leaving only the O(h) continuous rounding mass in (0, h/2)
        bed[0] = max(bed[0] - p0_, 0.0)
        # mass beyond T cannot contribute to any F^{*k}(T)
        bed[n1 + 1:] = 0.0
        pm += w_ * bed

    if pm.sum() <= 0:
        raise ValueError('no positive waiting-time mass on the grid -- '
                         'cannot form a renewal count')
    # conditional positive-wait law (defect stays as short mass)
    pm = pm / (1.0 - p0)

    kmax = _kmax_from_pmf(pm, bs, T, z=z)
    k, pM = renewal_count_pmf(pm, bs, T, lattice=closed, z=z,
                              tilt_total=tilt_total, kmax=kmax)
    pN = geometric_batch_compose(pM, p0) if p0 > 0 else pM
    k = np.arange(len(pN))

    bs_df.attrs.update(kmax=kmax, p0=p0, defect=defect,
                       est_count_error=kmax * bs * bs)
    info = dict(bs=bs, log2=log2, n1=n1, lattice=lattice, snapped=snapped,
                p0=p0, defect=defect, kmax=kmax, bs_df=bs_df, pm=pm)
    return k, pN, info
