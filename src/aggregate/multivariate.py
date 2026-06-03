"""Multivariate (bivariate) aggregate distributions via copula + 2D FFT.

This module is the first-class home of the joint-aggregate machinery. It hosts:

* :class:`MultivariateAggregate` -- the modelled object declared in DecL with
  the ``multivariate`` keyword. Two component ``agg`` / ``pnl`` severity
  factories are coupled per-claim by a :class:`aggregate.copula.Copula`, then
  accumulated by a **shared** outer frequency through a 2D FFT.
* :class:`BivariateDistribution` -- the lightweight joint-density container
  (marginals, mixed moments, correlation, contour). Originally introduced for
  :meth:`aggregate.distributions.Aggregate.occ_bivariate` (the joint law of
  occurrence ceded / net under a reinsurance program); kept here as the shared
  result container.
* :func:`size_axis` / :func:`scatter_bivariate` -- the per-axis sizing and the
  2D rebucketing scatter used by ``occ_bivariate``.

(The ``occ_bivariate`` facility and ``BivariateDistribution`` previously lived
in ``aggregate/bivariate.py``; that module is now a thin back-compat re-export
of the names here.)

Nothing here is re-exported at the top-level package namespace (submodule
access only, per the project layout convention): reach it as
``from aggregate.multivariate import MultivariateAggregate``.

Notes
-----
**The construction.** Each component is an inner :class:`Aggregate` restricted
to a zero-inflated Bernoulli ``dfreq [0 1] [p0 p1]`` form, so its *aggregate*
density is the per-event severity ``g_i = (1 - p_i) delta_0 + p_i f_i`` ("peril
i triggered this event w.p. ``p_i``, with conditional severity ``f_i``"). With
marginal per-claim CDFs ``G_i = cumsum(g_i)`` and a copula ``C``, the joint
per-claim severity is the discrete-Sklar rectangle mass
``S[i,j] = C(G1[i],G2[j]) - C(G1[i-1],G2[j]) - C(G1[i],G2[j-1]) +
C(G1[i-1],G2[j-1])`` (see :meth:`Copula.rectangle_pmf`). The joint aggregate is
then ``density = iFFT2(freq_pgf(N, FFT2(S)))`` -- the ordinary compound-FFT with
the 1D transforms replaced by 2D transforms, valid because ``freq_pgf(n, z)`` is
*elementwise* in ``z``. Marginalising one axis recovers the corresponding
standalone aggregate (the exact validation target), because the zero-frequency
slice of ``FFT2(S)`` along an axis is just ``FFT`` of that axis's marginal
``g_i``.

**P&L axes.** A ``pnl`` component contributes its *loss* severity ``g_i`` to the
copula+FFT exactly like an ``agg`` (the aggregate-level affine never touches the
severity). Its premium becomes a **per-axis affine** -- reflect + shift -- applied
to that tensor axis *after* the 2D FFT (:func:`_affine_axis`), the 1D
``_apply_agg_affine`` relabel lifted to one axis. Because the affine is a pure
per-axis grid relabel it commutes with marginalisation, so the affine axis's
marginal reproduces the standalone ``pnl`` and the loss-loss copula dependence
becomes the correct profit-loss sign once the axis is reflected.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft as sfft

from .constants import FIG_H, FIG_W
from .moments import MomentAggregator, xsden_to_mwrangler
from .utilities import round_bucket

logger = logging.getLogger(__name__)

# Coverage of the per-axis sizing window: 1 - 10**-_WINDOW_NINES per tail.
# Mirrors distributions.WINDOW_NINES; duplicated to avoid importing the heavy
# distributions module at import time (it imports this module's siblings).
_WINDOW_NINES = 12


def size_axis(agg_density, xs, bs_model, bs=None, log2=None,
              default_log2=10, cap_log2=14, quantile=1 - 1e-9):
    """Choose a bucket size and grid length for one bivariate axis.

    The joint grid must cover each aggregate margin's effective support (else
    2D FFT wrap-around aliasing), while staying small enough that the dense 2D
    array is feasible. This reads the effective support from the univariate
    aggregate margin (computed on the model grid ``xs``) and returns a rounded
    bucket size and a power-of-two grid length that covers it.

    Parameters
    ----------
    agg_density : ndarray
        Univariate aggregate density on the model grid (e.g.
        ``reins_density_df['p_agg_ceded_occ']``). Need not be normalised.
    xs : ndarray
        Model grid (``bs_model * arange``); the support of ``agg_density``.
    bs_model : float
        Model bucket size (``xs[1]``); used as a floor for the effective max.
    bs : float, optional
        Explicit bucket-size override. If given it is used verbatim.
    log2 : int, optional
        Explicit log2 grid-length override. If given the grid length is fixed
        at ``1 << log2`` and no coverage growth is performed.
    default_log2 : int, default 10
        Target log2 grid length when ``log2`` is not supplied.
    cap_log2 : int, default 14
        Upper bound on the auto-grown log2 (memory guard).
    quantile : float, default 1 - 1e-9
        Upper quantile of the margin used as the effective support max.

    Returns
    -------
    bs : float
        Axis bucket size.
    log2 : int
        Axis log2 grid length (grid has ``1 << log2`` points).

    Notes
    -----
    When neither override is supplied: the bucket is
    ``round_bucket(vmax / 2**default_log2)`` and ``log2`` is grown from
    ``default_log2`` until ``bs * (2**log2 - 1) >= vmax`` or ``cap_log2`` is
    reached. Supplying ``bs`` alone keeps the rounded count search; supplying
    ``log2`` alone derives ``bs`` from it; supplying both bypasses sizing.
    """
    if bs is not None and log2 is not None:
        return float(bs), int(log2)

    tot = agg_density.sum()
    if tot <= 0:
        # Degenerate margin (no mass): a single-bucket grid at the model scale.
        return (float(bs) if bs is not None else float(bs_model),
                int(log2) if log2 is not None else 1)
    cdf = np.cumsum(agg_density) / tot
    k = int(np.searchsorted(cdf, quantile))
    k = min(k, len(xs) - 1)
    vmax = max(float(xs[k]), float(bs_model))

    if log2 is None:
        log2 = default_log2
    if bs is None:
        bs = round_bucket(vmax / (1 << log2))
    # grow the grid until it covers the effective support, capped
    while bs * ((1 << log2) - 1) < vmax and log2 < cap_log2:
        log2 += 1
    return float(bs), int(log2)


def scatter_bivariate(cv, nv, mass, bs_c, bs_n, n_c, n_n, scheme='linear'):
    """Scatter per-claim mass onto a 2D ``(ceded, net)`` grid.

    The 2D analogue of the reinsurance rebucketing scatter
    (:meth:`Aggregate._rebucket_to_grid`). Each gross claim of mass
    ``mass[k]`` is placed at the off-grid point ``(cv[k], nv[k])`` and
    distributed onto the surrounding grid cells.

    Parameters
    ----------
    cv, nv : ndarray
        Ceded and net loss for each gross grid point (``ceder(xs)`` /
        ``netter(xs)``).
    mass : ndarray
        Gross severity probability mass aligned with ``cv`` / ``nv``.
    bs_c, bs_n : float
        Ceded- and net-axis bucket sizes.
    n_c, n_n : int
        Ceded- and net-axis grid lengths.
    scheme : {'linear', 'nearest'}, default 'linear'
        ``'nearest'`` rounds each point to its closest cell (full mass in one
        cell). ``'linear'`` splits each point's mass bilinearly over the <= 4
        surrounding cells with weights ``(1-fc)(1-fn)``, ``fc(1-fn)``,
        ``(1-fc)fn`` and ``fc fn`` where ``fc``/``fn`` are the fractional grid
        positions; this preserves **both** marginal means exactly.

    Returns
    -------
    ndarray
        Bivariate severity density, shape ``(n_c, n_n)``; total mass equals
        ``mass.sum()``.

    Notes
    -----
    Points at or beyond the top of either axis pile into the edge cell -- the
    same overflow mode as a univariate aggregate deficit. Fractional offsets
    are clipped to ``[0, 1]`` against the clipped index, mirroring the 1D
    scatter so the overflow weight collapses cleanly onto the last bucket.
    """
    out = np.zeros((n_c, n_n))
    sc = np.asarray(cv, dtype=float) / bs_c
    sn = np.asarray(nv, dtype=float) / bs_n
    if scheme == 'nearest':
        ic = np.clip(np.round(sc).astype(int), 0, n_c - 1)
        inn = np.clip(np.round(sn).astype(int), 0, n_n - 1)
        np.add.at(out, (ic, inn), mass)
    else:  # 'linear' -- bilinear split, preserves both marginal first moments
        kc = np.clip(np.floor(sc).astype(int), 0, n_c - 1)
        kn = np.clip(np.floor(sn).astype(int), 0, n_n - 1)
        fc = np.clip(sc - kc, 0.0, 1.0)
        fn = np.clip(sn - kn, 0.0, 1.0)
        kc1 = np.clip(kc + 1, 0, n_c - 1)
        kn1 = np.clip(kn + 1, 0, n_n - 1)
        np.add.at(out, (kc, kn), mass * (1 - fc) * (1 - fn))
        np.add.at(out, (kc1, kn), mass * fc * (1 - fn))
        np.add.at(out, (kc, kn1), mass * (1 - fc) * fn)
        np.add.at(out, (kc1, kn1), mass * fc * fn)
    return out


def build_netceded_joint(agg, bs_ceded=None, bs_net=None,
                         log2_ceded=None, log2_net=None):
    """Joint per-occurrence (ceded, net) density of one reinsured aggregate.

    The net/ceded severity builder for :class:`MultivariateAggregate`'s
    ``netceded`` mode (and the engine behind
    :meth:`aggregate.distributions.Aggregate.occ_bivariate`). Per claim the
    cession map sends a gross loss ``X`` to ``(c(X), n(X))`` on the line
    ``c + n = X``; placing the gross severity mass there (rebucketed by the
    object's :attr:`reins_bucket` scheme) builds a **comonotone** bivariate
    severity ``S``, and the joint aggregate is ``iFFT2(freq_pgf(N, FFT2(S)))``.

    Parameters
    ----------
    agg : Aggregate
        An **updated** aggregate carrying occurrence reinsurance.
    bs_ceded, bs_net : float, optional
        Axis bucket sizes; auto-sized from the univariate occ margins if omitted.
    log2_ceded, log2_net : int, optional
        Axis log2 grid lengths; auto-sized if omitted.

    Returns
    -------
    density : ndarray
        Joint ``(ceded, net)`` aggregate density, shape ``(n_c, n_n)``.
    ceded_grid, net_grid : ndarray
        The two axis grids.
    bs_c, bs_n : float
        The two axis bucket sizes.
    sev2 : ndarray
        The bivariate per-claim severity ``S`` (the comonotone scatter).
    deficit : float
        Tail mass lost beyond the grid.

    Raises
    ------
    ValueError
        If ``agg`` carries no occurrence reinsurance, or has not been updated.

    Notes
    -----
    Mirrors :meth:`Aggregate._fft_aggregate` (zero-risk and fixed-count-1
    shortcuts) with the 1D transforms replaced by 2D, valid because
    ``freq_pgf(n, z)`` is elementwise in ``z`` (handled with ravel/reshape).
    """
    import scipy.fft as _sfft

    if agg.occ_reins is None:
        raise ValueError(
            'netceded requires occurrence reinsurance; none configured.')
    if agg.sev_density_gross is None or agg.occ_ceder is None:
        raise ValueError(
            'netceded requires an updated object (no severity densities '
            'present). Call update() first.')

    rd = agg.reins_density_df
    bs_c, log2_c = size_axis(rd['p_agg_ceded_occ'].to_numpy(), agg.xs,
                             agg.bs, bs=bs_ceded, log2=log2_ceded)
    bs_n, log2_n = size_axis(rd['p_agg_net_occ'].to_numpy(), agg.xs,
                             agg.bs, bs=bs_net, log2=log2_net)
    n_c = 1 << log2_c
    n_n = 1 << log2_n
    ceded_grid = bs_c * np.arange(n_c)
    net_grid = bs_n * np.arange(n_n)

    cv = np.asarray(agg.occ_ceder(agg.xs), dtype=float)
    nv = np.asarray(agg.occ_netter(agg.xs), dtype=float)
    mass = np.asarray(agg.sev_density_gross, dtype=float)
    sev2 = scatter_bivariate(cv, nv, mass, bs_c, bs_n, n_c, n_n,
                             scheme=agg.reins_bucket)

    if agg.n == 0:
        density = np.zeros((n_c, n_n))
        density[0, 0] = 1.0
    elif np.sum(agg.en) == 1 and agg.frequency.freq_name == 'fixed':
        density = sev2.copy()
    else:
        pad = agg.padding
        s_shape = (n_c << pad, n_n << pad)
        z = _sfft.rfft2(sev2, s=s_shape)
        ftagg = agg.frequency.freq_pgf(agg.n, z.ravel()).reshape(z.shape)
        density = np.real(_sfft.irfft2(ftagg, s=s_shape))[:n_c, :n_n]

    density[np.abs(density) < 1e-15] = 0.0
    deficit = float(1.0 - density.sum())
    return density, ceded_grid, net_grid, bs_c, bs_n, sev2, deficit


def _affine_axis(density, axis, bs, n, reflect, shift, m_loss, sd, skew):
    """Relabel one tensor axis of a joint density to ``shift - A`` (the ``pnl`` form).

    Lifts the 1D :meth:`aggregate.distributions.Aggregate._apply_agg_affine`
    grid relabel to a single axis of a multi-dimensional density: reflect (a
    profit is a negative loss) and shift by the premium, placing the result on a
    tight, mass-centred P&L window for that axis. A pure reverse-and-roll of the
    finished density -- no new convolution -- so it commutes with
    marginalisation: the affine axis's marginal becomes the standalone ``pnl``,
    the other axis is untouched.

    Parameters
    ----------
    density : ndarray
        Joint loss density.
    axis : int
        Axis to transform.
    bs : float
        Bucket size of ``axis``.
    n : int
        Length of ``axis`` (``density.shape[axis]``).
    reflect : bool
        Reflect the axis (always ``True`` for ``pnl``; a profit is ``-loss``).
    shift : float
        Premium shift added after reflection.
    m_loss, sd, skew : float
        Analytic **loss** mean, standard deviation and skewness of the axis
        marginal (used to size the tight P&L window via
        :func:`aggregate.distributions.estimate_agg_window`).

    Returns
    -------
    out : ndarray
        Joint density with ``axis`` relabelled onto the P&L window.
    xs_new : ndarray
        New 1D grid for ``axis`` (``pnl_lo + arange(n) * bs``).

    Notes
    -----
    The loss grid origin is 0 (the inner loss aggregate is non-negative), so
    with ``s0 = round(shift / bs)`` and ``j_pnl = round(pnl_lo / bs)`` the map
    ``t = (s0 - j_pnl) - k`` (reflect) is a one-to-one reverse-and-roll; buckets
    outside the window are dropped (a negligible far-tail deficit).
    """
    # local import: estimate_agg_window lives in the heavy distributions module
    from .distributions import estimate_agg_window

    s0 = int(round(shift / bs)) if bs else 0
    m = (shift - m_loss) if reflect else (shift + m_loss)
    sk = -skew if reflect else skew
    p = 1.0 - 10.0 ** -_WINDOW_NINES
    if np.isfinite(sd) and sd > 0:
        try:
            lo, hi, _W = estimate_agg_window(m, sd, sk, p)
        except (ValueError, FloatingPointError):  # pragma: no cover - defensive
            lo, hi = m - 8.0 * sd, m + 8.0 * sd
    else:  # pragma: no cover - defensive (point mass)
        lo, hi = m, m
    if (hi - lo) <= n * bs:
        pnl_lo = float(np.floor(lo / bs) * bs) if bs else float(lo)
    else:
        pnl_lo = (float(np.floor((m - 0.5 * n * bs) / bs) * bs) if bs
                  else float(m - 0.5 * n * bs))
    j_pnl = int(round(pnl_lo / bs)) if bs else 0

    k = np.arange(n)
    t = (s0 - j_pnl) - k if reflect else (s0 - j_pnl) + k
    valid = (t >= 0) & (t < n)

    dm = np.moveaxis(density, axis, 0)
    out = np.zeros_like(dm)
    out[t[valid]] = dm[k[valid]]
    out = np.moveaxis(out, 0, axis)
    xs_new = pnl_lo + np.arange(n, dtype=float) * bs
    return out, xs_new


class MultivariateAggregate:
    """Joint (bivariate) aggregate of two copula-coupled component aggregates.

    Declared in DecL with the ``multivariate`` keyword::

        multivariate Cat 25 claims
            agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
            agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
            copula gumbel 0.4
            mixed gamma .2

    A **shared** outer frequency (here ``25`` events, gamma-mixed Poisson) drives
    two perils; within each event the two per-claim severities are coupled by the
    copula (here gumbel upper-tail dependence), and the per-event trigger
    probabilities come from the inner ``dfreq [0 1] [p0 p1]`` Bernoulli forms.
    The result is the joint law ``(A_Wind, A_Flood)``; either marginal reproduces
    the standalone aggregate for that line.

    Parameters
    ----------
    name : str
        Object name.
    lines : list of tuple
        Exactly two ``('agg', name, spec)`` tuples (the component severity
        factories), as produced by the DecL transformer. A component may be a
        ``pnl`` (its spec then carries ``agg_reflect`` / ``agg_shift``); the
        affine is applied per-axis after the 2D FFT.
    copula : Copula
        The fitted :class:`aggregate.copula.Copula` instance coupling the two
        per-claim severities.
    note : str, optional
        Free-text note.
    hints : str, optional
        Raw ``hints{...}`` build-settings string (``key=value;`` form);
        retained as annotation and consumed by the underwriter build path.
    exp_en, exp_el, exp_premium, exp_lr : float, optional
        Outer (shared-count) exposure, as parsed. ``exp_en`` (a claim count) is
        used directly; otherwise the count is derived from ``exp_el`` and the
        component per-event means.
    freq_name, freq_a, freq_b, freq_zm, freq_p0
        Outer (shared) frequency specification.

    Attributes
    ----------
    lines : list of Aggregate
        The two component **loss** aggregates (the per-event severity factories;
        any ``pnl`` affine is stripped here and reapplied per-axis after the FFT).
    line_names : list of str
        Component names.
    copula : Copula
    frequency : Frequency
        The shared outer frequency.
    en : float
        Expected outer event count.
    density : ndarray or None
        Joint aggregate density, shape ``(n0, n1)``; populated by :meth:`update`.
    axis_xs : list of ndarray
        Per-axis output grids (P&L-relabelled where a component is a ``pnl``).
    bs : list of float
        Per-axis bucket sizes.
    """

    def __init__(self, name, lines=None, copula=None, note='', hints='', mode='copula',
                 nc_agg=None, nc_kwargs=None,
                 exp_en=None, exp_el=None, exp_premium=None, exp_lr=None,
                 freq_name='poisson', freq_a=0.0, freq_b=0.0,
                 freq_zm=False, freq_p0=np.nan, **kwargs):
        # local import avoids a circular import at module load (distributions
        # imports nothing from here at import time, but be defensive).
        from .distributions import Aggregate, Frequency

        self.mode = mode
        self.name = name
        self.note = note
        self.hints = hints
        self.program = ''
        # filled by update() (both modes)
        self.density = None
        self.axis_xs = [None, None]
        self.bs = [None, None]
        self.deficit = np.nan
        self._S = None
        self._sev_xs = [None, None]
        self._marg_theory = [None, None]   # per-axis (mean, sd, skew)
        self.figure = None                 # set by plot()

        if mode == 'netceded':
            self._init_netceded(name, lines, nc_agg, nc_kwargs)
            return

        if lines is None or len(lines) != 2:
            raise ValueError(
                'multivariate (copula) requires exactly two components; '
                f'got {0 if lines is None else len(lines)}')
        self.copula = copula

        self._line_specs = [t[2] for t in lines]
        self.line_names = [t[1] for t in lines]

        # split off any per-axis pnl affine; build the loss twins
        self._affine = []
        loss_specs = []
        for s in self._line_specs:
            reflect = bool(s.get('agg_reflect', False))
            shift = float(s.get('agg_shift', 0.0))
            self._affine.append((reflect, shift))
            ls = {k: v for k, v in s.items()
                  if k not in ('agg_reflect', 'agg_shift',
                               'agg_premium', 'value_type')}
            loss_specs.append(ls)
        self.lines = [Aggregate(**s) for s in loss_specs]

        # shared outer frequency + per-event severity raw moments (theoretical,
        # grid-independent, available at Aggregate.__init__)
        self.frequency = Frequency(freq_name, freq_a, freq_b, freq_zm, freq_p0)
        self.freq_name = freq_name
        self._sev_moms = [self._raw3(a.agg_m, a.agg_sd, a.agg_skew)
                          for a in self.lines]
        self.en = self._resolve_en(exp_en, exp_el, exp_premium, exp_lr)
        self._gs = None

    def _init_netceded(self, name, lines, nc_agg, nc_kwargs):
        """Initialise the ``netceded`` mode: one reinsured aggregate split into
        its joint per-occurrence (ceded, net) law.

        Parameters
        ----------
        name : str
            Object name.
        lines : list of tuple or None
            Either ``None`` (when ``nc_agg`` is supplied directly, the
            :meth:`aggregate.distributions.Aggregate.occ_bivariate` path) or a
            single ``('agg', name, spec)`` tuple (the DecL ``netceded`` path),
            from which the inner aggregate is built.
        nc_agg : Aggregate or None
            A pre-built (already updated) aggregate carrying occurrence
            reinsurance; if given it is used as-is.
        nc_kwargs : dict or None
            Axis-sizing overrides (``bs_ceded`` / ``bs_net`` / ``log2_ceded`` /
            ``log2_net``) forwarded to :func:`build_netceded_joint`.
        """
        from .distributions import Aggregate

        self.copula = None
        self.line_names = ['Ceded', 'Net']
        self._affine = [(False, 0.0), (False, 0.0)]
        self._nc_kwargs = dict(nc_kwargs or {})
        if nc_agg is not None:
            self._nc_agg = nc_agg
            self._nc_built_here = False
        else:
            if not lines:
                raise ValueError('netceded requires one component aggregate.')
            self._nc_agg = Aggregate(**lines[0][2])
            self._nc_built_here = True

    # ------------------------------------------------------------------
    # moment / sizing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw3(m, sd, skew):
        """Raw moments ``(E[X], E[X^2], E[X^3])`` from ``(mean, sd, skew)``."""
        var = float(sd) ** 2
        s2 = var + m * m
        s3 = (skew * sd ** 3 if np.isfinite(skew) else 0.0) + 3 * m * s2 - 2 * m ** 3
        return (float(m), float(s2), float(s3))

    def _resolve_en(self, exp_en, exp_el, exp_premium, exp_lr):
        """Expected outer event count from the parsed exposure clause."""
        if exp_en is not None and np.sum(np.asarray(exp_en, dtype=float)) > 0:
            return float(np.sum(np.asarray(exp_en, dtype=float)))
        # fall back to a loss/premium exposure: events = loss / per-event mean
        el = 0.0
        if exp_el is not None:
            el = float(np.sum(np.asarray(exp_el, dtype=float)))
        elif exp_premium is not None and exp_lr is not None:
            el = float(np.sum(np.asarray(exp_premium, dtype=float)
                              * np.asarray(exp_lr, dtype=float)))
        if el > 0:
            mean_per_event = sum(s[0] for s in self._sev_moms)
            if mean_per_event > 0:
                return el / mean_per_event
        raise ValueError(
            'multivariate: cannot determine the shared event count; supply a '
            'claim count (e.g. "25 claims") on the multivariate statement.')

    def _marginal_moments(self, i):
        """Analytic ``(mean, sd, skew)`` of component ``i``'s **loss** marginal.

        The outer compound of the per-event severity ``g_i`` -- the standalone
        aggregate for that line in loss terms (before any ``pnl`` affine).
        """
        f1, f2, f3 = self.frequency.freq_moms(self.en)
        s1, s2, s3 = self._sev_moms[i]
        a1, a2, a3 = MomentAggregator.agg_from_fs(f1, f2, f3, s1, s2, s3)
        m, cv, skew = MomentAggregator.static_moments_to_mcvsk(a1, a2, a3)
        sd = (cv * m) if np.isfinite(cv) else float(np.sqrt(max(a2 - a1 * a1, 0.0)))
        return float(m), float(sd), float(skew)

    def _size_axis(self, i, default_log2=9, cap_log2=11):
        """Pick ``(bs, log2)`` for axis ``i`` covering its loss marginal support."""
        from .distributions import estimate_agg_window

        m, sd, skew = self._marginal_moments(i)
        p = 1.0 - 10.0 ** -_WINDOW_NINES
        if np.isfinite(sd) and sd > 0:
            try:
                _lo, hi, _W = estimate_agg_window(m, sd, skew, p)
            except (ValueError, FloatingPointError):  # pragma: no cover
                hi = m + 8.0 * sd
            vmax = max(hi, m + 8.0 * sd)
        else:  # pragma: no cover - degenerate
            vmax = 2.0 * m if m > 0 else 1.0
        vmax = max(vmax, 1e-9)
        log2 = default_log2
        bs = round_bucket(vmax / (1 << log2))
        while bs * ((1 << log2) - 1) < vmax and log2 < cap_log2:
            log2 += 1
        return float(bs), int(log2)

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, log2=0, bs=0, padding=1, **kwargs):
        """Build the joint density: size axes, discretise ``g_i``, 2D FFT.

        Parameters
        ----------
        log2 : int, optional
            Per-axis log2 grid length override (applied to both axes). ``0``
            (default) auto-sizes each axis from its loss marginal support.
        bs : float, optional
            Per-axis bucket-size override (applied to both axes). ``0``
            (default) auto-sizes.
        padding : int, default 1
            FFT zero-padding factor per axis (mirrors the 1D aggregate; ``1``
            doubles each axis length for the transform).
        **kwargs
            Ignored (accepted for a uniform ``build`` call signature).

        Notes
        -----
        Each component loss twin is rebuilt on the chosen coarse axis grid so
        ``g_i = agg_density`` is the per-event severity on that grid; the joint
        is then assembled in :meth:`update_work`. In ``netceded`` mode the joint
        is built from the single reinsured aggregate's comonotone scatter
        (:func:`build_netceded_joint`) instead.
        """
        self.padding = int(padding)
        if self.mode == 'netceded':
            return self._update_netceded(log2=log2, bs=bs)
        self._gs = []
        for i, a in enumerate(self.lines):
            if bs and log2:
                bs_i, log2_i = float(bs), int(log2)
            else:
                bs_i, log2_i = self._size_axis(i)
                if log2:
                    log2_i = int(log2)
                if bs:
                    bs_i = float(bs)
            a.update(log2=log2_i, bs=bs_i)
            self.bs[i] = float(a.bs)
            self.axis_xs[i] = np.asarray(a.xs, dtype=float).copy()
            self._gs.append(np.asarray(a.agg_density, dtype=float).copy())
        # severity grids are the (0-based, loss) per-event grids, captured
        # before any pnl affine relabels the aggregate axes in update_work.
        self._sev_xs = [x.copy() for x in self.axis_xs]
        self.update_work()
        return self

    def _update_netceded(self, log2=0, bs=0):
        """Build the joint (ceded, net) density of the single reinsured agg.

        Updates the inner aggregate (only if this object built it -- a
        pre-built agg passed by :meth:`Aggregate.occ_bivariate` must already be
        updated, so the precondition raises propagate), then assembles the joint
        via :func:`build_netceded_joint` and caches the per-axis ceded / net
        theoretical moments from the inner ``reins_stats_df``.
        """
        a = self._nc_agg
        if self._nc_built_here and a.sev_density_gross is None:
            kw = {}
            if log2:
                kw['log2'] = int(log2)
            if bs:
                kw['bs'] = bs
            a.update(**kw)
        density, cg, ng, bs_c, bs_n, sev2, deficit = build_netceded_joint(
            a, **self._nc_kwargs)
        self.density = density
        self.axis_xs = [cg, ng]
        self.bs = [bs_c, bs_n]
        self._S = sev2
        self._sev_xs = [cg, ng]   # severity panel = comonotone (c, n) scatter
        self.deficit = deficit
        self._marg_theory = self._netceded_theory(a)
        return self

    @staticmethod
    def _netceded_theory(agg):
        """Per-axis ``(mean, sd, skew)`` of the ceded / net occ aggregates,
        read from the inner aggregate's :attr:`reins_stats_df`."""
        rs = agg.reins_stats_df
        out = []
        for view in ('Ceded', 'Net'):
            m = float(rs.loc[('agg', 'mean'), ('occ', view)])
            cv = float(rs.loc[('agg', 'cv'), ('occ', view)])
            sk = float(rs.loc[('agg', 'skew'), ('occ', view)])
            out.append((m, cv * m, sk))
        return out

    def update_work(self):
        """Assemble the joint density from the per-event severities ``g_i``.

        Builds the copula joint per-claim severity ``S`` (discrete Sklar), runs
        the 2D compound FFT (``freq_pgf`` applied elementwise to ``rfft2(S)``),
        then applies any per-axis ``pnl`` affine. The zero-risk and fixed-count-1
        shortcuts mirror :meth:`aggregate.distributions.Aggregate._fft_aggregate`.
        """
        g0, g1 = self._gs
        n0, n1 = len(g0), len(g1)
        # marginal CDFs (jump at 0 from the Bernoulli zero-inflation handled
        # automatically as a jump in G)
        G0 = np.cumsum(g0)
        G1 = np.cumsum(g1)
        # joint per-claim severity via the copula (marginals exact by
        # construction); retained for the severity panel of plot().
        S = self.copula.rectangle_pmf(G0, G1)
        self._S = S

        if self.en == 0:
            density = np.zeros((n0, n1))
            density[0, 0] = 1.0
        elif np.sum(self.en) == 1 and self.frequency.freq_name == 'fixed':
            density = S.copy()
        else:
            pad = self.padding
            s_shape = (n0 << pad, n1 << pad)
            z = sfft.rfft2(S, s=s_shape)
            # freq_pgf is mathematically elementwise in z, but the empirical
            # implementation assumes a 1D argument -- flatten, apply, reshape.
            ftagg = self.frequency.freq_pgf(self.en, z.ravel()).reshape(z.shape)
            density = np.real(sfft.irfft2(ftagg, s=s_shape))[:n0, :n1]

        density[np.abs(density) < 1e-15] = 0.0

        # per-axis pnl affine (reflect + premium shift) applied post-FFT
        for i, (reflect, shift) in enumerate(self._affine):
            if reflect or shift != 0.0:
                m, sd, skew = self._marginal_moments(i)
                density, xs_new = _affine_axis(
                    density, i, self.bs[i], density.shape[i],
                    reflect, shift, m, sd, skew)
                self.axis_xs[i] = xs_new

        self.density = density
        self.deficit = float(1.0 - density.sum())
        # cache per-axis loss-marginal theory (pre-affine); describe / stats_df
        # apply any pnl affine on top.
        self._marg_theory = [self._marginal_moments(0), self._marginal_moments(1)]

    # ------------------------------------------------------------------
    # results: marginals, moments, correlation, frames
    # ------------------------------------------------------------------

    def _require_density(self):
        if self.density is None:
            raise ValueError('MultivariateAggregate not updated; call update().')

    @property
    def bivariate(self):
        """A :class:`BivariateDistribution` view of the joint density.

        Reuses the shared container for moments, correlation, contour and the
        HTML repr.
        """
        self._require_density()
        if self.mode == 'netceded':
            en, fname = float(self._nc_agg.n), self._nc_agg.frequency.freq_name
            cop = 'netceded'
        else:
            en, fname, cop = float(self.en), self.freq_name, str(self.copula)
        meta = {'name': self.name, 'en': en, 'freq_name': fname,
                'copula': cop, 'deficit': self.deficit,
                'axis_names': tuple(self.line_names)}
        return BivariateDistribution(self.density, self.axis_xs[0],
                                     self.axis_xs[1], self.bs[0], self.bs[1],
                                     meta)

    def marginals(self):
        """Return the two marginal densities ``(density.sum(1), density.sum(0))``.

        Each reproduces the standalone aggregate for that component (a ``pnl``
        axis reproduces the standalone ``pnl``).
        """
        self._require_density()
        return self.density.sum(axis=1), self.density.sum(axis=0)

    def moments(self, max_order=3):
        """Mixed raw moments ``E[A0^i A1^j]`` (delegates to the bivariate view)."""
        return self.bivariate.moments(max_order)

    def corr(self):
        """Pearson correlation of the two component aggregates.

        Notes
        -----
        This is the **realised output** correlation, which is *not* the copula
        parameter: compounding by the shared frequency attenuates the per-claim
        dependence (and a shared mixing frequency adds common-shock dependence on
        top). Compare with :meth:`Copula.tau` via :meth:`describe`.
        """
        return self.bivariate.corr()

    @property
    def density_df(self):
        """Joint density as a DataFrame (a thin wrapper around ``density``).

        Returns
        -------
        DataFrame
            ``density[i, j]`` with the axis-0 grid as the index and the axis-1
            grid as the columns, labelled by the two component names.
        """
        self._require_density()
        return pd.DataFrame(
            self.density,
            index=pd.Index(self.axis_xs[0], name=self.line_names[0]),
            columns=pd.Index(self.axis_xs[1], name=self.line_names[1]))

    @property
    def stats_df(self):
        """Per-component marginal moments (theoretical vs empirical) + joint block.

        Returns
        -------
        DataFrame
            Columns: the two component names plus ``joint``. Rows: ``mean``,
            ``sd``, ``cv``, ``skew`` per component (theoretical ``T`` and
            empirical ``E`` from the realised marginal), and a joint block
            (``cov``, ``corr``, ``copula_tau``, ``E[A0 A1]``).

        Notes
        -----
        Theoretical moments are the analytic loss-marginal moments, P&L-adjusted
        for a ``pnl`` axis (``mean -> shift - E[A]``, ``sd`` unchanged,
        ``skew -> -skew``). Empirical moments come from the realised marginal
        density on the (possibly P&L-relabelled) axis grid.
        """
        self._require_density()
        m0, m1 = self.marginals()
        cols = {}
        for i, (name, dens) in enumerate(zip(self.line_names, (m0, m1))):
            mt, sdt, skt = self._axis_theory(i)
            cvt = sdt / mt if mt else np.nan
            mw = xsden_to_mwrangler(self.axis_xs[i], dens)
            me, cve, ske = mw.mcvsk
            sde = cve * me if np.isfinite(cve) else np.nan
            cols[name] = pd.Series({
                ('theoretical', 'mean'): mt, ('theoretical', 'sd'): sdt,
                ('theoretical', 'cv'): cvt, ('theoretical', 'skew'): skt,
                ('empirical', 'mean'): me, ('empirical', 'sd'): sde,
                ('empirical', 'cv'): cve, ('empirical', 'skew'): ske,
            })
        # joint block
        mom = self.moments(2).to_numpy()
        tot = mom[0, 0]
        e0, e1 = mom[1, 0] / tot, mom[0, 1] / tot
        cov = mom[1, 1] / tot - e0 * e1
        joint = pd.Series({
            ('theoretical', 'mean'): np.nan, ('theoretical', 'sd'): np.nan,
            ('theoretical', 'cv'): np.nan, ('theoretical', 'skew'): np.nan,
            ('empirical', 'cov'): cov,
            ('empirical', 'corr'): self.corr(),
            ('empirical', 'copula_tau'): (np.nan if self.copula is None
                                          else self.copula.tau()),
            ('empirical', 'E[A0A1]'): mom[1, 1] / tot,
        })
        cols['joint'] = joint
        df = pd.DataFrame(cols)
        df.index = pd.MultiIndex.from_tuples(df.index, names=['basis', 'stat'])
        return df

    def _axis_theory(self, i):
        """Displayed theoretical ``(mean, sd, skew)`` for axis ``i``.

        The cached marginal theory with any ``pnl`` affine applied (copula mode;
        ``netceded`` has no affine so the ceded / net moments pass through)."""
        mt, sdt, skt = self._marg_theory[i]
        reflect, shift = self._affine[i]
        if reflect:
            mt, skt = shift - mt, -skt
        elif shift:
            mt = shift + mt
        return mt, sdt, skt

    @property
    def describe(self):
        """Compact per-component summary frame with the realised dependence.

        Returns
        -------
        DataFrame
            One row per component (``kind``, theoretical ``mean`` / ``sd`` /
            ``cv`` / ``skew``) plus a ``joint`` footer row carrying the realised
            output correlation and (copula mode) the copula's Kendall tau -- the
            realised corr is *not* the copula parameter (compounding attenuates
            it, shared mixing adds common shock). A ``pnl`` component is shown in
            P&L terms (``mean -> premium - E[loss]``, ``skew`` sign flipped); in
            ``netceded`` mode the rows are the Ceded / Net occurrence aggregates.
        """
        self._require_density()
        rows = {}
        cols = ['kind', 'mean', 'sd', 'cv', 'skew', 'corr', 'copula_tau']
        for i, name in enumerate(self.line_names):
            mt, sdt, skt = self._axis_theory(i)
            reflect, shift = self._affine[i]
            kind = 'netceded' if self.mode == 'netceded' else (
                'pnl' if (reflect or shift) else 'agg')
            rows[name] = {'kind': kind, 'mean': mt, 'sd': sdt,
                          'cv': (sdt / mt if mt else np.nan), 'skew': skt,
                          'corr': np.nan, 'copula_tau': np.nan}
        # joint footer: realised output correlation + (copula) Kendall tau
        rows['joint'] = {
            'kind': self.mode if self.copula is None else str(self.copula),
            'mean': np.nan, 'sd': np.nan, 'cv': np.nan, 'skew': np.nan,
            'corr': self.corr(),
            'copula_tau': np.nan if self.copula is None else self.copula.tau()}
        df = pd.DataFrame(rows).T[cols]
        df.index.name = 'component'
        return df

    @property
    def info(self):
        """Multi-line human-readable summary of the multivariate aggregate."""
        self._require_density()
        lines = [f'MultivariateAggregate {self.name!r} ({self.mode})',
                 f'  components   {self.line_names[0]!r} x {self.line_names[1]!r}']
        if self.mode == 'netceded':
            lines.append(f'  source       {self._nc_agg.name!r}: '
                         f'{self._nc_agg.frequency.freq_name}, '
                         f'E[N] = {self._nc_agg.n:.6g}, occurrence reinsurance')
        else:
            lines.append(f'  shared freq  {self.freq_name}, E[N] = {self.en:.6g}')
            lines.append(f'  copula       {self.copula!r}')
        lines.append(
            f'  grid         {self.density.shape[0]} x {self.density.shape[1]} '
            f'(bs = {self.bs[0]:g}, {self.bs[1]:g})')
        for i, name in enumerate(self.line_names):
            reflect, shift = self._affine[i]
            tag = ' [pnl]' if (reflect or shift) else ''
            lo, hi = self.axis_xs[i][0], self.axis_xs[i][-1]
            lines.append(
                f'  axis {i} {name}{tag}: window [{lo:.6g}, {hi:.6g}], '
                f'mass {self.density.sum(axis=1 - i).sum():.6f}')
        tail = '' if self.copula is None else f' (copula tau = {self.copula.tau():.4f})'
        lines.append(f'  correlation  {self.corr():.6f}{tail}')
        lines.append(f'  tail deficit {self.deficit:.2e}')
        return '\n'.join(lines)

    @staticmethod
    def _contourf(ax, xgrid, ygrid, Z, title, xlabel, ylabel, levels, log,
                  **kwargs):
        """Single filled-contour panel of a 2D density on ``(xgrid, ygrid)``."""
        xx, yy = np.meshgrid(xgrid, ygrid)
        z = Z.T   # density indexed [axis0, axis1]; contourf wants Z[row=y, col=x]
        if log:
            pos = z[z > 0]
            floor = pos.min() if pos.size else 1e-300
            z = np.log10(np.maximum(z, floor))
        ax.contourf(xx, yy, z, levels=levels, **kwargs)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        return ax

    def plot(self, axs=None, levels=14, log=False, **kwargs):
        """Two-panel contour plot: per-claim severity (left), aggregate (right).

        Parameters
        ----------
        axs : array of matplotlib Axes, optional
            Two target axes (e.g. ``ax0, ax1 = axs.flat``). A new ``1 x 2``
            figure is created if omitted; the figure is stored on
            :attr:`figure`.
        levels : int, default 14
            Number of contour levels.
        log : bool, default False
            Contour ``log10`` of each density (useful for heavy tails).
        **kwargs
            Passed through to ``Axes.contourf``.

        Returns
        -------
        None
            The drawn axes are reachable via ``self.figure.axes`` (or the
            ``axs`` you passed in). Returning ``None`` -- matching
            :meth:`Aggregate.plot` -- avoids the Jupyter inline backend
            rendering the figure twice (once via its post-execute hook and once
            from a returned matplotlib object).

        Notes
        -----
        The left panel is the joint **per-claim severity** ``S`` (the copula
        coupling -- or the comonotone ``(ceded, net)`` scatter in ``netceded``
        mode -- on the severity grids); the right panel is the joint
        **aggregate** density (on the output grids, P&L-relabelled for any
        ``pnl`` axis).
        """
        self._require_density()
        if axs is None:
            self.figure, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_H),
                                            constrained_layout=True)
        else:
            self.figure = np.asarray(axs).flat[0].figure
        ax0, ax1 = np.asarray(axs).flat[:2]
        n0, n1 = self.line_names
        self._contourf(ax0, self._sev_xs[0], self._sev_xs[1], self._S,
                       'severity', n0, n1, levels, log, **kwargs)
        self._contourf(ax1, self.axis_xs[0], self.axis_xs[1], self.density,
                       'aggregate', n0, n1, levels, log, **kwargs)

    def help(self, regex):
        """Lookup help on methods and properties matching ``regex``.

        Thin wrapper over :func:`aggregate.utilities.agg_help` (prefixed to
        avoid shadowing the builtin ``help``)."""
        from .utilities import agg_help
        agg_help(self, regex)

    def __repr__(self):
        tag = self.mode if self.copula is None else repr(self.copula)
        if self.density is None:
            return (f'MultivariateAggregate(name={self.name!r}, '
                    f'lines={self.line_names!r}, {tag}, not updated)')
        return (f'MultivariateAggregate(name={self.name!r}, '
                f'lines={self.line_names!r}, {tag}, '
                f'shape={self.density.shape}, corr={self.corr():.4f})')

    def _repr_html_(self):
        if self.density is None:
            return f'<pre>{self!r}</pre>'
        return self.bivariate._repr_html_()


class BivariateDistribution(object):
    """Joint distribution of two aggregate quantities on a 2D grid.

    A lightweight container for a 2D density (e.g. the joint occurrence ceded
    ``C`` / net ``N`` returned by
    :meth:`aggregate.distributions.Aggregate.occ_bivariate`, or the joint of two
    copula-coupled components from :class:`MultivariateAggregate`), with
    marginals, mixed moments, correlation, and a contour plot.

    Parameters
    ----------
    density : ndarray
        Joint probability mass, shape ``(len(ceded), len(net))``; entry
        ``[a, b]`` is ``P(C ~ ceded[a], N ~ net[b])``.
    ceded, net : ndarray
        1D axis grids.
    bs_ceded, bs_net : float
        Axis bucket sizes.
    meta : dict, optional
        Provenance (``name``, ``en``, ``freq_name``, ``deficit``, ...).

    Attributes
    ----------
    density, ceded, net, bs_ceded, bs_net, meta
        As constructed.
    """

    def __init__(self, density, ceded, net, bs_ceded, bs_net, meta=None):
        self.density = np.asarray(density, dtype=float)
        self.ceded = np.asarray(ceded, dtype=float)
        self.net = np.asarray(net, dtype=float)
        self.bs_ceded = float(bs_ceded)
        self.bs_net = float(bs_net)
        self.meta = dict(meta) if meta else {}

    def marginals(self):
        """Return the ceded and net marginal densities.

        Returns
        -------
        ceded_density : ndarray
            ``density.sum(axis=1)`` on the ``ceded`` grid.
        net_density : ndarray
            ``density.sum(axis=0)`` on the ``net`` grid.

        Notes
        -----
        These should match the univariate occurrence aggregate margins
        ``reins_density_df['p_agg_ceded_occ']`` and ``['p_agg_net_occ']``
        (rebucketed to the bivariate grids) and provide the primary numerical
        validation of the 2D convolution.
        """
        return self.density.sum(axis=1), self.density.sum(axis=0)

    def moments(self, max_order=3):
        """Mixed raw moments ``E[C^i N^j]`` for ``i, j in 0..max_order``.

        Parameters
        ----------
        max_order : int, default 3
            Highest power taken on each axis.

        Returns
        -------
        DataFrame
            ``(max_order + 1) x (max_order + 1)`` table; row ``i`` / column
            ``j`` is ``E[C^i N^j] = sum density * ceded**i * net**j``. The
            ``[0, 0]`` entry is the total probability (``~1``).

        Notes
        -----
        Computed as ``Ci @ density @ Nj.T`` where ``Ci[i] = ceded**i`` and
        ``Nj[j] = net**j`` -- a vectorised double sum over the grid.
        """
        ci = np.vstack([self.ceded ** i for i in range(max_order + 1)])
        nj = np.vstack([self.net ** j for j in range(max_order + 1)])
        m = ci @ self.density @ nj.T
        return pd.DataFrame(
            m,
            index=pd.Index([f'C^{i}' for i in range(max_order + 1)], name='C'),
            columns=pd.Index([f'N^{j}' for j in range(max_order + 1)], name='N'))

    def corr(self):
        """Pearson correlation of the two axes.

        Returns
        -------
        float
            ``Cov(C, N) / sqrt(Var(C) Var(N))``, in ``[-1, 1]``; ``nan`` if
            either margin is degenerate (zero variance).
        """
        m = self.moments(2).to_numpy()
        tot = m[0, 0]
        e_c = m[1, 0] / tot
        e_n = m[0, 1] / tot
        var_c = m[2, 0] / tot - e_c ** 2
        var_n = m[0, 2] / tot - e_n ** 2
        cov = m[1, 1] / tot - e_c * e_n
        denom = var_c * var_n
        if denom <= 0:
            return np.nan
        return float(cov / np.sqrt(denom))

    def contour(self, ax=None, levels=14, log=False, **kwargs):
        """Filled contour plot of the joint density.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Target axes; a new figure is created if omitted (using the
            project ``FIG_W`` / ``FIG_H`` and constrained layout).
        levels : int, default 14
            Number of contour levels.
        log : bool, default False
            Contour ``log10`` of the density (clipped at the smallest positive
            value) -- useful for the heavy-tailed dependency structure.
        **kwargs
            Passed through to ``Axes.contourf``.

        Returns
        -------
        matplotlib Axes
            The axes drawn on.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H),
                                   constrained_layout=True)
        # density is indexed [ceded, net]; meshgrid wants Z[row=net, col=ceded]
        cc, nn = np.meshgrid(self.ceded, self.net)
        z = self.density.T
        if log:
            pos = z[z > 0]
            floor = pos.min() if pos.size else 1e-300
            z = np.log10(np.maximum(z, floor))
        ax.contourf(cc, nn, z, levels=levels, **kwargs)
        names = self.meta.get('axis_names', ('ceded', 'net'))
        ax.set(xlabel=f'Aggregate {names[0]}', ylabel=f'Aggregate {names[1]}',
               title=f'Joint density\n{self.meta.get("name", "")}')
        return ax

    def _summary(self):
        """Return ``(E[C], E[N], corr, deficit)`` for the repr builders."""
        m = self.moments(1).to_numpy()
        tot = m[0, 0]
        return m[1, 0] / tot, m[0, 1] / tot, self.corr(), self.meta.get('deficit', 0.0)

    def __repr__(self):
        e_c, e_n, rho, deficit = self._summary()
        return (f'BivariateDistribution(name={self.meta.get("name", "")!r}, '
                f'shape={self.density.shape}, '
                f'bs=({self.bs_ceded:g}, {self.bs_net:g}), '
                f'E[C]={e_c:,.4g}, E[N]={e_n:,.4g}, corr={rho:.4f})')

    def _repr_html_(self):
        e_c, e_n, rho, deficit = self._summary()
        names = self.meta.get('axis_names', ('ceded (C)', 'net (N)'))
        rows = [
            ('name', self.meta.get('name', '')),
            ('grid shape', f'{self.density.shape[0]} &times; {self.density.shape[1]}'),
            ('bucket', f'{self.bs_ceded:g}, {self.bs_net:g}'),
            (f'E[{names[0]}]', f'{e_c:,.6g}'),
            (f'E[{names[1]}]', f'{e_n:,.6g}'),
            ('correlation', f'{rho:.6f}'),
            ('tail deficit', f'{deficit:.2e}'),
        ]
        body = ''.join(
            f'<tr><th style="text-align:left">{k}</th><td>{v}</td></tr>'
            for k, v in rows)
        return (f'<table class="aggregate bivariate">'
                f'<caption>BivariateDistribution</caption>{body}</table>')
