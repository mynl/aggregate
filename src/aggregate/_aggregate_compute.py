"""Pure-function compute core for the aggregate FFT convolution.

Extracted from ``Aggregate`` (Phase 2A) so the FFT-PGF-iFFT kernel is reachable
and testable **without** a full :meth:`Aggregate.update` -- it can be driven
directly with hand-built severity vectors and a frequency PGF, cross-checked
against ``np.convolve`` (fixed frequency) or a direct Panjer/compound sum, and
inspected in a notebook. ``Aggregate`` keeps thin methods that call in here.

A leaf: numpy / scipy.fft / the ``utilities`` FFT helpers only; it never imports
``_aggregate``.
"""

import logging

import numpy as np
import scipy.fft as sfft

from .utilities import ft, ift

logger = logging.getLogger(__name__)


def freq_sev_convolution(sev_density, freq_pgf, n, *, N, bs, i0=0, x_min=0.0,
                         en=None, freq_name='', padding=1):
    """Aggregate density by FFT convolution: ``iFFT(freq_pgf(n, FFT(sev)))``.

    The single source of truth for the FFT-PGF-iFFT core (Mildenhall 2024,
    §2.2). The compound (aggregate) law of ``S = X_1 + ... + X_K`` with random
    claim count ``K`` is ``iFFT( P_K( FFT(sev) ) )`` where ``P_K`` is the
    frequency probability generating function -- valid because the PGF acts
    elementwise on the transformed severity. This function is what
    :meth:`Aggregate._fft_aggregate` delegates to; it takes plain arrays and a
    PGF callable, so it can be exercised directly (see
    ``tests/test_aggregate_compute.py``).

    Parameters
    ----------
    sev_density : np.ndarray
        Discretised severity, length ``N``; ``sev_density[j]`` is the mass at
        physical ``(j - i0) * bs``.
    freq_pgf : callable
        The frequency PGF ``freq_pgf(n, z) -> ndarray``, elementwise in the
        transformed argument ``z`` (e.g. ``Frequency.freq_pgf``).
    n : int
        Grid length passed to the PGF (``self.n``; the FFT length parameter).
    N : int
        Length of the output grid (``len(self.xs)``).
    bs : float
        Bucket size; only used to locate the output-window origin.
    i0 : int, optional
        Number of severity buckets below physical 0 (signed severity). ``0`` on
        the default non-negative grid.
    x_min : float, optional
        Output-window origin; ``0.0`` on the default zero-based grid.
    en : array-like, optional
        Expected claim counts; only consulted for the fixed-frequency,
        single-claim shortcut. ``None`` skips the shortcut.
    freq_name : str, optional
        Frequency kind; the fixed-1 shortcut fires only for ``'fixed'``.
    padding : int, optional
        FFT padding factor passed to :func:`~aggregate.utilities.ft` /
        :func:`~aggregate.utilities.ift`. Default 1.

    Returns
    -------
    agg_density : np.ndarray
        Aggregate density on the output grid (length ``N``).
    ftagg_density : np.ndarray
        FT of the aggregate (padded length); callers that don't need it discard.

    Notes
    -----
    Two paths, selected by whether any offset is active:

    - **Default (``i0 == 0`` and window origin ``x_min == 0``).** The original
      ``ft`` / ``freq_pgf`` / ``ift`` path, byte-for-byte identical to prior
      releases.
    - **Signed / windowed.** Negatives live at the top of the padded length-``M
      = N << padding`` FFT buffer (period ``M*bs``); the severity is laid in with
      physical 0 at index 0 (``i0`` negative buckets wrapped to the top). After
      the FFT the result is *relabelled* onto the output window by a single
      ``np.roll`` of ``-round(x_min/bs)`` and the first ``N`` buckets kept.
      Relabelling a finished, exact array carries no ``N·s`` shift term, so this
      is correct for random as well as fixed frequency. Exact when the aggregate
      support width ``W < M*bs``; a narrower window shows a two-sided deficit.
    """
    j0 = int(round(x_min / bs)) if bs else 0
    if i0 == 0 and j0 == 0:
        # ---- default non-negative, zero-based path (unchanged) ----
        if n == 0:
            out = np.zeros(N)
            out[0] = 1.0
            return out, ft(out, padding)
        z = ft(sev_density, padding)
        ftagg = freq_pgf(n, z)
        if en is not None and np.sum(en) == 1 and freq_name == 'fixed':
            return sev_density.copy(), ftagg
        return np.real(ift(ftagg, padding)), ftagg

    # ---- signed / windowed path (F1 negative-x + F2 output window) ----
    M = N << padding
    if n == 0:
        # Zero-risk: point mass at physical 0, placed at FFT index 0 so the
        # output-window roll below sends it to the correct output bucket.
        a = np.zeros(M)
        a[0] = 1.0
        ftagg = sfft.rfft(a)
    else:
        # Lay the severity into the length-M buffer: physical 0..(N-1-i0)*bs at
        # indices 0..N-1-i0; the i0 negative buckets wrap to the very top of M
        # (indices M-i0..M-1). Equivalent to np.roll(sev, -i0) but into the
        # padded length so the period is M*bs, not N*bs.
        g = np.zeros(M)
        g[:N - i0] = sev_density[i0:]
        if i0:
            g[M - i0:] = sev_density[:i0]
        z = sfft.rfft(g)
        ftagg = freq_pgf(n, z)
        a = sfft.irfft(ftagg, M)
    # F2: relabel onto the output window. Roll so x_min lands at output index 0
    # (j0 may be negative when x_min < 0), then keep the first N.
    agg = np.roll(a, -j0)[:N]
    return agg, ftagg


def evaluate_pgf_polynomial(atoms, weights, z):
    """Evaluate the empirical pgf ``P(z) = sum_i w_i z^(a_i)`` elementwise in ``z``.

    Drop-in replacement for the legacy matrix expression
    ``weights @ np.power(z, atoms.reshape(-1, 1))``, which materializes an
    ``n_atoms x len(z)`` complex matrix (memory-heavy and, for large
    exponents, poorly conditioned -- complex ``np.power`` falls back to
    exp/log). Non-negative-integer supports dispatch on an operation-count
    model; anything else (fractional / negative outcomes, which parse
    today) takes the legacy path verbatim.

    Parameters
    ----------
    atoms : array-like
        Support (the ``freq_a`` outcomes).
    weights : array-like
        Probability masses (the ``freq_b`` vector).
    z : scalar or 1-D array, real or complex
        Evaluation points (e.g. an rfft vector).

    Returns
    -------
    ndarray
        ``P(z)``, 1-D; a scalar ``z`` returns shape ``(1,)`` exactly like
        the legacy matrix expression.

    Notes
    -----
    Cost model with sorted exponents ``k_1 < ... < k_K`` and gaps
    ``d_i = k_i - k_(i-1)`` (``k_0 = 0``): **Horner** over scattered dense
    coefficients ``0..k_max`` costs ``k_max`` vector fused multiply-adds;
    **sorted-gap square-and-multiply** costs
    ``sum_i max(2 ceil(log2 d_i), 1)`` vector multiplies. Both are
    O(len(z)) memory, so compare the two integers and pick the smaller;
    ties go to Horner (better conditioned: a single accumulator, no
    explicit large powers). A dense support ``0..K`` therefore takes
    Horner; a sparse ``[0 1 2 1000]`` takes powers. Squaring is explicit
    (never ``np.power`` on large exponents).
    """
    a = np.asarray(atoms, dtype=float)
    w = np.asarray(weights, dtype=float)
    if a.size == 0 or not (np.all(np.isfinite(a))
                           and np.all(a == np.floor(a))
                           and np.all(a >= 0)
                           and np.all(a <= 2.0 ** 53)):
        # legacy matrix path verbatim
        return w @ np.power(z, a.reshape((a.shape[0], 1)))

    zv = np.atleast_1d(np.asarray(z))
    order = np.argsort(a)
    k = a[order].astype(np.int64)
    wk = w[order]
    kmax = int(k[-1])
    gaps = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), k)))
    pos = gaps[gaps > 0]
    cost_powers = int(np.sum(np.maximum(
        2 * np.ceil(np.log2(pos)), 1))) if len(pos) else 0
    out_dtype = np.result_type(zv.dtype, w.dtype)

    if kmax <= cost_powers:
        # dense Horner: scatter the weights onto coefficients 0..kmax and
        # run one fused multiply-add per degree with a single accumulator
        c = np.zeros(kmax + 1)
        np.add.at(c, k, wk)
        acc = np.full(zv.shape, c[kmax], dtype=out_dtype)
        for j in range(kmax - 1, -1, -1):
            acc *= zv
            if c[j]:
                acc += c[j]
        return acc

    # sparse sorted-gap square-and-multiply: cache z^(2^j) once, then walk
    # the gaps accumulating z^(k_i) incrementally by explicit squarings
    pow2 = [zv.astype(out_dtype)]
    max_gap = int(gaps.max())
    while (1 << len(pow2)) <= max_gap:
        pow2.append(pow2[-1] * pow2[-1])
    zpow = np.ones(zv.shape, dtype=out_dtype)
    acc = np.zeros(zv.shape, dtype=out_dtype)
    for gap, w_i in zip(gaps, wk):
        g = int(gap)
        j = 0
        while g:
            if g & 1:
                zpow = zpow * pow2[j]
            g >>= 1
            j += 1
        acc += w_i * zpow
    return acc


def discretize_severities(sevs, xs, bs, *, i0=0, sev_calc='discrete',
                          discretization_calc='survival', normalize=True,
                          dsev_bucket=None, rebucket=None):
    """Discretize severity distributions onto a fixed grid.

    The pure kernel behind :meth:`Aggregate.discretize`, extracted so other
    consumers (the renewal waiting-time path) can discretize a list of
    :class:`Severity` objects without an :class:`Aggregate` instance.

    Parameters
    ----------
    sevs : iterable of Severity
        Components to discretize. Each must expose ``cdf`` / ``sf`` (and, for
        the linear-scatter branch, ``sev_kind``, ``exp_attachment`` and a
        frozen ``fz`` with ``xk`` / ``pk`` atoms).
    xs : ndarray
        The severity grid (physical 0 at index ``i0``); equally spaced with
        step ``bs``.
    bs : float
        Bucket size.
    i0 : int, default 0
        Number of negative buckets; ``i0 > 0`` flags the signed grid.
    sev_calc : str, default 'discrete'
        ``discrete``/``round`` (half-bucket shift), ``forward``/``continuous``,
        or ``backward`` bucket-edge scheme.
    discretization_calc : str, default 'survival'
        ``survival`` (backward differences of sf, best right-tail),
        ``distribution`` (forward differences of cdf, best left-tail), or
        ``both`` (elementwise max of the two).
    normalize : bool, default True
        Rescale each component to sum to 1. Pass False when short mass is
        intended (e.g. a waiting-time pmf truncated at the horizon).
    dsev_bucket : str, optional
        ``'linear'`` routes unlayered discrete severities through the
        mean-preserving linear scatter (requires ``rebucket``).
    rebucket : callable, optional
        ``rebucket(values, mass, scheme=, origin=)`` implementing the scatter
        (:meth:`Aggregate._rebucket_to_grid`).

    Returns
    -------
    list of ndarray
        One pmf vector per component, aligned to ``xs``.
    """
    signed = i0 > 0

    if sev_calc == 'discrete' or sev_calc == 'round':
        # adj_xs = np.hstack((xs - bs / 2, np.inf))
        # mass at the end undesirable. can be put in with reinsurance layer in spec
        # note the first bucket is negative
        adj_xs = np.hstack((xs - bs / 2, xs[-1] + bs / 2))
    elif sev_calc == 'forward' or sev_calc == 'continuous':
        adj_xs = np.hstack((xs, xs[-1] + bs))
    elif sev_calc == 'backward':
        adj_xs = np.hstack((xs[0] - bs, xs))  # , np.inf))
    elif sev_calc == 'moment':
        raise NotImplementedError(
            'Moment matching discretization not implemented. Embrechts says it is not worth it.')
        #
        # adj_xs = np.hstack((xs, np.inf))
    else:
        raise ValueError(
            f'Invalid parameter {sev_calc} passed to discretize; options are discrete, continuous, or raw.')

    if not signed:
        # Non-negative severity: the first bucket must include all mass at
        # and below 0. Capture the whole left tail from -inf, exactly as
        # before (byte-for-byte unchanged on the default path).
        adj_xs[0] = -np.inf
    # Signed mode: the leftmost bucket is treated identically to every
    # other bucket (a finite ``xs[0] - bs/2`` edge, set above) -- no
    # -inf catch. Any residual mass below it (<= 1e-12 by construction of
    # i0) is dropped and absorbed by the optional renormalisation below.

    # bed = bucketed empirical distribution. A signed Severity carries
    # identity layering, so its own cdf/sf are already un-clamped; an
    # unsigned component keeps the clamp-at-0 layering. So per-component
    # ``sev.cdf`` / ``sev.sf`` are correct on the signed grid with no special
    # casing (an unsigned component simply contributes 0 to negative
    # buckets). Occurrence reinsurance on a signed severity is out of scope
    # (plan §6).
    beds = []
    for sev in sevs:
        if (dsev_bucket == 'linear'
                and rebucket is not None
                and sev.sev_kind in ('dhistogram', 'fixed')
                and sev.exp_attachment is None):
            # Unlayered discrete severity: place the atoms on the grid with
            # the mean-preserving linear scatter, so the discretized first
            # moment equals the exact atom mean. The default cdf-difference
            # path (below) snaps each atom to its nearest bucket
            # (== 'nearest'), biasing the mean by up to bs/2 per atom when
            # atoms are off-grid (empirical samples, non-integer bs). On-grid
            # atoms give f == 0 so this reduces to nearest -- the dice /
            # integer-bs case is unchanged. ``sev.fz`` is the _DiscreteRV
            # holding the validated, sorted atoms (xk) and masses (pk).
            # ``exp_attachment is None`` is the truly-unlayered test (a
            # discrete sev always has a finite ``detachment`` = max atom, so
            # a ``== np.inf`` test never fires). Layered discrete severities
            # fall through to the cdf-diff path (Phase 1; see dsev_bucket).
            # The bed is indexed on ``xs``, so the scatter origin is the
            # severity grid origin ``xs[0]`` (== -i0*bs), NOT the
            # output-window origin ``x_min`` -- the two differ when the
            # output window is forced wider than the atom support (e.g. an
            # explicit ``x_min`` below the smallest atom).
            appx = rebucket(sev.fz.xk, sev.fz.pk, scheme='linear', origin=xs[0])
        elif discretization_calc == 'both':
            # see comments: we rescale each severity...
            appx = np.maximum(np.diff(sev.cdf(adj_xs)), -np.diff(sev.sf(adj_xs)))
        elif discretization_calc == 'survival':
            appx = -np.diff(sev.sf(adj_xs))
        elif discretization_calc == 'distribution':
            appx = np.diff(sev.cdf(adj_xs))
        else:
            raise ValueError(
                f'Invalid options {discretization_calc} to double_diff; options are density, survival or both')
        if normalize:
            # A severity whose support lies entirely off the grid discretizes
            # to all zeros (e.g. mean 50, sd 5, on a grid topping out at 16
            # because bs and log2 were both pinned). Dividing by a zero total
            # replaces a truthful "no mass here" with NaN in every bucket,
            # and NaN survives the FFT, so the whole aggregate comes back
            # NaN with nothing said. Leave the zeros: the resulting 100%
            # deficit is what ``update_work`` reports, and that warning names
            # the fix.
            total = np.sum(appx)
            if total > 0:
                appx = appx / total
            else:
                logger.warning(
                    'discretize | severity %r contributes no mass to this '
                    'grid (support is entirely outside it); leaving it at '
                    'zero rather than normalizing 0/0.',
                    getattr(sev, 'name', '?'))
        beds.append(appx)
    return beds
