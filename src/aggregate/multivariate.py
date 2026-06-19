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
* :func:`_netceded_window_hi` / :func:`scatter_bivariate` -- the (one common bs)
  window measurement and the 2D rebucketing scatter used by ``occ_bivariate``.

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
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft as sfft

from .constants import FIG_H, FIG_W, DefectiveDistributionWarning
from .config import get_settings
from .moments import MomentAggregator, xsden_to_mwrangler
from .utilities import round_bucket, balanced_window

logger = logging.getLogger(__name__)

# Coverage of the per-axis sizing window: 1 - 10**-_WINDOW_NINES per tail.
# First-class multivariate setting (see aggregate.config [multivariate]);
# independent of the 1-D distributions.WINDOW_NINES because the 2-D per-axis
# grid may want fewer nines for memory. Resolved once per session.
_WINDOW_NINES = get_settings().multivariate.window_nines
# Total 2-D grid budget in log2 cells (2**_TOTAL_LOG2 cells), split between the
# two axes by measured support; overridable via update(log2=...). Square-law
# memory lever (see dev/plan-mv.md §5.3).
_TOTAL_LOG2 = get_settings().multivariate.total_log2
# Smallest per-axis log2 the sizer will hand back (keeps a usable grid).
_MIN_AXIS_LOG2 = 4


def _netceded_window_hi(occ_density, xs, prob):
    """Upper window edge of an occurrence-reins margin (``balanced_window``).

    Netceded sizing is **window selection** on the realized occ margins -- a
    column of ``reins_density_df`` (``p_agg_ceded_occ`` / ``p_agg_net_occ``,
    computed in one gross pass) read with the same
    :func:`~aggregate.utilities.balanced_window` primitive as the copula axes.
    Ceded / net are non-negative, so the grid is 0-based and need only reach this
    upper edge.

    Parameters
    ----------
    occ_density : ndarray
        The occ-reins aggregate margin on the gross grid ``xs`` (need not be
        normalised).
    xs : ndarray
        The gross loss grid.
    prob : float
        Discarded equal-tail mass for ``balanced_window``.

    Returns
    -------
    float
        The upper window edge ``q(1 - prob/2)`` of the margin.
    """
    ser = pd.Series(np.asarray(occ_density, dtype=float), index=np.asarray(xs))
    ser = ser[ser > 0]
    if len(ser) == 0:        # degenerate margin (no mass)
        return 0.0
    _lo, hi = balanced_window(ser, prob)
    return float(hi)


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
                         log2_ceded=None, log2_net=None, total_log2=None):
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
        Bucket-size override (a single common ``bs`` for both axes). Default
        (``None``) sizes **one common ``bs`` from the budget** -- the comonotone
        ``(c, n)`` curve is sampled at the gross grid and the linear scatter
        rebuckets it onto the common grid, so the ``bs`` is chosen to fit
        ``2**total_log2`` (``~ sqrt(hi_c*hi_n)/2**(total_log2/2)``), *not* pinned
        to the (often far finer) gross ``bs`` (§5.2).
    log2_ceded, log2_net : int, optional
        Axis log2 grid-length overrides; measured from the realized occ margins
        via :func:`balanced_window` if omitted.
    total_log2 : int, optional
        Total 2-D cell budget. The common ``bs`` is coarsened until the two
        windows fit; if a caller pins ``bs``/``log2`` and they still overflow,
        the wider axis is **clipped** (a reported deficit). ``None`` uses the
        :attr:`MultivariateSettings.total_log2` default.

    Returns
    -------
    density : ndarray
        Joint ``(ceded, net)`` aggregate density, shape ``(n_c, n_n)``.
    ceded_grid, net_grid : ndarray
        The two axis grids.
    bs_c, bs_n : float
        The two axis bucket sizes (both the gross ``bs`` unless overridden).
    sev2 : ndarray
        The bivariate per-claim severity ``S`` (the comonotone scatter).
    deficit : float
        Tail mass lost beyond the grid.
    clipped : bool
        ``True`` if the pinned-``bs`` windows exceeded the budget and an axis was
        clipped.

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
    if total_log2 is None:
        total_log2 = _TOTAL_LOG2

    rd = agg.reins_density_df
    prob = 10.0 ** -_WINDOW_NINES
    # ONE common bs across both axes (the comonotone curve couples them). The
    # window lengths are measured from the realized occ margins (balanced_window;
    # ceded / net are non-negative so the grids are 0-based), and the common bs
    # is sized straight from the budget -- NOT pinned to the gross bs. The gross
    # bs auto-sizes fine to resolve the cession layer (e.g. 0.125), which is far
    # too fine for the 2-D grid; the linear scatter rebuckets the gross-sampled
    # (c, n) points onto whatever common bs the budget affords. Sizing from the
    # budget (not gross) also makes occ_bivariate and the DecL `netceded` form
    # agree regardless of the gross grid each was built on.
    hi_c = _netceded_window_hi(rd['p_agg_ceded_occ'].to_numpy(), agg.xs, prob)
    hi_n = _netceded_window_hi(rd['p_agg_net_occ'].to_numpy(), agg.xs, prob)

    def _cov(hi, bs):
        return max(int(np.ceil(np.log2(hi / bs + 1.0))), _MIN_AXIS_LOG2)

    pinned = bool(bs_ceded or bs_net or (log2_ceded and log2_net))
    if bs_ceded or bs_net:
        bs = float(bs_ceded or bs_net)
    else:
        # finest common bs that fits 2**total_log2: n_c*n_n ~ (hi_c*hi_n)/bs**2,
        # so bs ~ sqrt(hi_c*hi_n) / 2**(total_log2/2). round_bucket up, then fit.
        floor_bs = max((hi_c * hi_n) ** 0.5 / (2.0 ** (0.5 * total_log2)), 1e-12)
        bs = float(round_bucket(floor_bs))
    log2_c = int(log2_ceded) if log2_ceded else _cov(hi_c, bs)
    log2_n = int(log2_net) if log2_net else _cov(hi_n, bs)
    if not pinned:
        guard = 0
        while log2_c + log2_n > total_log2 and guard < 8:
            bs = float(round_bucket(bs * 2))
            log2_c, log2_n = _cov(hi_c, bs), _cov(hi_n, bs)
            guard += 1
    clipped = log2_c + log2_n > total_log2
    if clipped:
        # bs pinned by the caller and still over budget -> clip the wider axis.
        if log2_c >= log2_n:
            log2_c = max(total_log2 - log2_n, _MIN_AXIS_LOG2)
        else:
            log2_n = max(total_log2 - log2_c, _MIN_AXIS_LOG2)
        warnings.warn(
            f'{agg.name}: netceded (ceded, net) windows need more than the budget '
            f'2**{total_log2} at the pinned bs={bs:g}; the wider axis is clipped '
            f'(a tail deficit). Raise update(log2=...) or relax the bs pin.',
            DefectiveDistributionWarning, stacklevel=2)
    bs_c = bs_n = bs
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
    return density, ceded_grid, net_grid, bs_c, bs_n, sev2, deficit, clipped


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
        # the shared-frequency kwargs, retained so a standalone marginal
        # aggregate can be rebuilt for measure-don't-guess axis sizing (§5).
        self._freq_kwargs = dict(freq_name=freq_name, freq_a=freq_a,
                                 freq_b=freq_b, freq_zm=freq_zm, freq_p0=freq_p0)
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

    def _standalone_marginal(self, i):
        """Build the standalone **loss** marginal aggregate for axis ``i``.

        The component-``i`` marginal is the shared outer frequency compounded
        with that component's per-event severity, i.e. an ordinary 1-D
        :class:`Aggregate` with the **shared** frequency and an expected count
        ``en * (per-event trigger mean)``. This is the validation target the
        joint marginal must reproduce (exact for Poisson / mixed-Poisson /
        negative-binomial outer frequencies, where thinning preserves the
        family). It is built but **not updated** here -- the caller sizes it.

        Any ``pnl`` affine is *not* applied: the loss marginal is what the 2-D
        FFT runs on, and the per-axis affine relabels it afterwards
        (:func:`_affine_axis`), so the loss marginal is the correct thing to
        size for both axis kinds.

        Returns
        -------
        Aggregate
            The unupdated standalone loss marginal.
        """
        from .distributions import Aggregate

        spec = {k: v for k, v in self._line_specs[i].items()
                if k.startswith('sev_') or k in ('name', 'note')}
        spec['exp_en'] = float(self.en) * float(self.lines[i].n)
        spec.update(self._freq_kwargs)
        return Aggregate(**spec)

    def _measure_marginal_window(self, i, prob, log2):
        """Equal-tail support window of component ``i``'s realized loss marginal.

        Runs the standalone loss marginal and reads its
        :func:`~aggregate.utilities.balanced_window` at tail probability
        ``prob``. A **signed** marginal needs care: the 1-D sizer places its grid
        origin at the window's lower edge, so all the power-of-2 slack lands
        *above* the band and the cheap lower tail clips -- which biases the
        equal-tail measurement upward (a symmetric axis comes out off-centre). So
        a signed marginal is rebuilt on a *centred* grid (slack both sides, sized
        from the SBJ-aware first build's realized extent so a heavy tail is still
        covered) before measuring. Non-negative axes are bounded at 0 and need no
        recentre.

        Returns
        -------
        (lo, hi, bs) : tuple of float
            The measured window edges and the marginal's bucket size (the latter
            a resolution floor for the width).
        """
        a = self._standalone_marginal(i)
        a.update(log2=log2)
        ser = a.density_df.query('p_total > 0').p_total
        reflect, shift = self._affine[i]
        if a.x_min < 0 and not (reflect or shift):
            nz = ser.index[ser.values > 1e-13]
            mlo, mhi = float(nz.min()), float(nz.max())
            margin = 0.5 * (mhi - mlo)        # slack on both sides
            lo_box, hi_box = mlo - margin, mhi + margin
            bs0 = float(round_bucket((hi_box - lo_box) / (1 << log2)))
            a.update(log2=log2, bs=bs0, x_min=float(np.floor(lo_box / bs0) * bs0))
            ser = a.density_df.query('p_total > 0').p_total
        lo, hi = balanced_window(ser, prob)
        return lo, hi, float(a.bs)

    def _size_axes(self, total_log2, bs_axes, log2_axes):
        """Measure each axis's grid from its realized standalone marginal (§5).

        The bv's privilege is to *measure*, not guess. Each standalone loss
        marginal is run once on a fine grid; an equal-tail
        :func:`~aggregate.utilities.balanced_window` (discarded mass
        ``10**-window_nines``) reads its **support window** directly off the
        realized pmf. The total budget ``2**total_log2`` is then a pure
        *resolution* choice: the per-axis ``log2`` split is allocated so the two
        bucket sizes come out comparable (``L0 - L1 ~ log2(W0 / W1)``), and each
        ``bs`` is fit to its window, ``bs_i = round_bucket(W_i / (2**L_i - 1))``.

        Because the measured window always covers the deep (``window_nines``)
        tail, the budget controls only *resolution*: a smaller budget coarsens
        ``bs`` rather than clipping the support, so the joint mass is conserved
        regardless of budget (copula axes have free ``bs``).

        Parameters
        ----------
        total_log2 : int
            Total log2 cell budget (``2**total_log2`` cells across both axes)
            for the auto split; ignored for an axis with an explicit ``log2``.
        bs_axes : list of (float or None)
            Per-axis explicit ``bs``; ``None`` to measure that axis.
        log2_axes : list of (int or None)
            Per-axis explicit ``log2``; ``None`` to allocate that axis from the
            budget. With one axis pinned the other takes the rest of the budget.

        Returns
        -------
        bss, log2s, x_mins, his : list of float/int/float/float
            Per-axis bucket size, output grid log2, window origin, and window
            top. ``x_min`` is the *measured* lower edge -- negative on a signed
            axis, positive on a non-negative book whose mass lives far from 0
            (no artificial 0-pin), and 0 only when the mass genuinely reaches the
            origin or the axis is a ``pnl`` loss (see below). ``his`` lets the
            caller size the FFT buffer to reach physical 0.
        clipped : bool
            ``True`` if explicit per-axis ``log2`` overrides exceed the budget.
        """
        prob = 10.0 ** -_WINDOW_NINES
        # 1. Measure each marginal's support window off the realized pmf.
        los, his, widths = [], [], []
        for i in range(2):
            lo, hi, bs0 = self._measure_marginal_window(i, prob, total_log2)
            # Use the MEASURED lower edge -- do not pin to 0. The only 0-pin is a
            # pnl axis, whose loss has a known lower bound of 0 *and* whose
            # _affine_axis relabel assumes a 0-based loss grid; there the affine
            # owns the (tight) P&L windowing, so the loss axis stays 0-based.
            reflect, shift = self._affine[i]
            if reflect or shift:
                lo = 0.0
            los.append(lo)
            his.append(hi)
            widths.append(max(hi - lo, bs0))

        # 2. Budget split: only when BOTH axes are fully auto does the total
        #    budget get balanced across them (comparable bs). The moment either
        #    axis is overridden, each axis is sized independently to honour its
        #    override and cover its measured window; a free axis then takes half
        #    the budget.
        def _cover_log2(width, bs):
            return max(int(np.ceil(np.log2(width / bs + 1.0))), _MIN_AXIS_LOG2)

        free = [log2_axes[i] is None and bs_axes[i] is None for i in range(2)]
        if free[0] and free[1]:
            delta = np.log2(widths[0] / widths[1])
            cap = total_log2 - _MIN_AXIS_LOG2
            L0 = int(np.clip(round((total_log2 + delta) / 2.0), _MIN_AXIS_LOG2, cap))
            alloc = [L0, total_log2 - L0]
        else:
            alloc = [total_log2 // 2, total_log2 // 2]

        # 3. Per-axis (bs, log2): a pinned bs grows log2 to cover the window; a
        #    pinned log2 fits bs to the window; pinning both is the user's call.
        #    bs is rounded UP (round_bucket), never to nearest, even though grid
        #    size is at a premium here: the grid length N is a power of two, so a
        #    measured window of (say) width 78 lands on a 128-wide grid whatever
        #    bs is -- rounding bs *down* to the nearest rung cannot tighten that,
        #    it only fails to cover the window (clip) or forces a larger log2
        #    (more memory). The dead space is split symmetrically by the centred
        #    placement below instead.
        bss, log2s, x_mins = [], [], []
        for i in range(2):
            if log2_axes[i] is not None:
                L_i = int(log2_axes[i])
                bs_i = (float(bs_axes[i]) if bs_axes[i] is not None
                        else float(round_bucket(widths[i] / ((1 << L_i) - 1))))
            elif bs_axes[i] is not None:
                bs_i = float(bs_axes[i])
                L_i = _cover_log2(widths[i], bs_i)
            else:
                L_i = alloc[i]
                bs_i = float(round_bucket(widths[i] / ((1 << L_i) - 1)))
            log2s.append(L_i)
            bss.append(bs_i)
            # Centre the measured window in the (power-of-two) grid: split the
            # unavoidable slack ``N*bs - width`` equally either side rather than
            # piling it above (which left a symmetric axis looking off-centre).
            # A non-negative axis is clamped at 0 (no mass below, no point wasting
            # grid there). Snap the origin down to the grid so physical 0 stays
            # an integer index (the signed compound's j0 roll needs that).
            slack = (1 << L_i) * bs_i - (his[i] - los[i])
            xc = los[i] - 0.5 * slack
            if los[i] >= 0:
                xc = max(0.0, xc)
            x_mins.append(float(np.floor(xc / bs_i) * bs_i))

        # Coverage can only fail when BOTH bs and log2 are pinned too small.
        clipped = any(((1 << log2s[i]) - 1) * bss[i] < widths[i] - bss[i]
                      for i in range(2))
        if clipped:
            logger.warning(
                'multivariate %s: pinned (bs, log2) does not cover the measured '
                'window on an axis -- expect a tail deficit.', self.name)
        return bss, log2s, x_mins, his, clipped

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, log2=0, bs=0, padding=1, **kwargs):
        """Build the joint density: size axes, discretise ``g_i``, 2D FFT.

        Parameters
        ----------
        log2 : int or (int, int), optional
            A **scalar** is the *total* 2-D grid budget in log2 cells
            (``2**log2`` cells, split between the axes by measured support);
            ``0`` (default) uses the :attr:`MultivariateSettings.total_log2`
            config value (20). A **2-tuple ``(log2_x, log2_y)``** pins the
            per-axis log2 directly (the budget is then their sum) -- use it to
            explore a split (e.g. ``(11, 9)`` vs ``(10, 10)``). The auto split
            *falls out* of each realized marginal's measured support -- the bv
            measures, it does not guess (``dev/plan-mv.md`` §5). Memory is
            quadratic in the per-axis length, so raise only a little.
        bs : float or (float, float), optional
            A **scalar** is applied to both axes; a **2-tuple ``(bs_x, bs_y)``**
            pins the per-axis bucket size. ``0`` (default) measures each axis's
            ``bs`` from its standalone marginal. (Both ``log2`` and ``bs`` tuple
            forms pass straight through ``build(..., log2=(a, b), bs=(x, y))``.)
        padding : int, default 1
            FFT zero-padding factor per axis (mirrors the 1D aggregate; ``1``
            doubles each axis length for the transform).
        **kwargs
            Ignored (accepted for a uniform ``build`` call signature).

        Notes
        -----
        Axis sizing is *measured*, not guessed: each component's standalone loss
        marginal is run first, an equal-tail :func:`balanced_window` trims each
        realized pmf, and the per-axis ``(bs, log2, x_min)`` is read off the
        measured support (:meth:`_size_axes`). A signed (``ssev``) axis takes a
        two-sided window with a negative origin; :meth:`update_work` then
        compounds it with the same ``i0``/``j0`` wrap-and-roll the 1-D
        :meth:`aggregate.distributions.Aggregate._fft_aggregate` uses, so its
        negative tail no longer wraps. In ``netceded`` mode the joint is built
        from the single reinsured aggregate's comonotone scatter
        (:func:`build_netceded_joint`) instead.
        """
        self.padding = int(padding)
        if self.mode == 'netceded':
            return self._update_netceded(log2=log2, bs=bs)
        # Parse scalar-or-(x, y) sizing args into per-axis overrides + budget.
        # Tuple entries may be None/0 to leave that axis auto.
        if isinstance(log2, (tuple, list)):
            log2_axes = [int(v) if v else None for v in log2]
            total_log2 = sum(int(v) for v in log2 if v) or _TOTAL_LOG2
        else:
            log2_axes = [None, None]
            total_log2 = int(log2) if log2 else _TOTAL_LOG2
        if isinstance(bs, (tuple, list)):
            bs_axes = [float(v) if v else None for v in bs]
        elif bs:
            bs_axes = [float(bs), float(bs)]
        else:
            bs_axes = [None, None]
        bss, log2s, x_mins, his, clipped = self._size_axes(
            total_log2, bs_axes, log2_axes)
        self._clipped = clipped
        self._gs = []
        self._i0 = []                    # per-event severity negative reach (lay-in wrap)
        self._j0 = []                    # output-window origin in buckets (final roll)
        self._mlog2 = []                 # log2 FFT buffer length per axis
        self._nout = [1 << L for L in log2s]
        for i, a in enumerate(self.lines):
            # Build the per-event severity g_i on its OWN natural grid at the
            # measured resolution (x_min='auto' so a signed component keeps its
            # negative buckets). The output window is measured separately from
            # the marginal -- the severity's negative reach (i0) and the output
            # origin (j0) are independent, exactly as in the 1-D _fft_aggregate.
            a.update(log2=log2s[i], bs=bss[i])
            self.bs[i] = float(bss[i])
            self._gs.append(np.asarray(a.agg_density, dtype=float).copy())
            self._i0.append(int(round(-a.x_min / a.bs)) if a.x_min else 0)
            self._j0.append(int(round(x_mins[i] / bss[i])) if x_mins[i] else 0)
            self.axis_xs[i] = (x_mins[i]
                               + np.arange(self._nout[i], dtype=float) * bss[i])
            # The compound is anchored at physical 0 (non-negative severity) or
            # wraps the negatives (signed), so the FFT buffer must reach from
            # min(0, x_min) up to the grid top -- DECOUPLED from the output
            # length so a tight window far from 0 doesn't alias the upper tail.
            grid_top = x_mins[i] + self._nout[i] * bss[i]
            reach = grid_top - min(0.0, x_mins[i])
            need = int(np.ceil(np.log2(max(reach / bss[i], 2.0))))
            self._mlog2.append(max(need, log2s[i]) + self.padding)
        # severity grids are the per-event grids (signed where the component is),
        # captured for the severity panel of plot().
        self._sev_xs = [np.asarray(a.xs, dtype=float).copy() for a in self.lines]
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
        density, cg, ng, bs_c, bs_n, sev2, deficit, clipped = build_netceded_joint(
            a, **self._nc_kwargs)
        self.density = density
        self.axis_xs = [cg, ng]
        self.bs = [bs_c, bs_n]
        self._S = sev2
        self._sev_xs = [cg, ng]   # severity panel = comonotone (c, n) scatter
        self.deficit = deficit
        self._clipped = clipped
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

    @staticmethod
    def _lay_signed_2d(S, i0_0, i0_1, M0, M1):
        """Lay per-claim severity ``S`` into an ``(M0, M1)`` FFT buffer, physical 0 at index 0.

        The 2-D analogue of the signed/windowed lay-in of
        :meth:`aggregate.distributions.Aggregate._fft_aggregate`: per axis the
        ``i0`` negative-physical buckets of the severity (array indices
        ``0..i0-1``) wrap to the top of the padded length ``M`` (period
        ``M*bs``), so the FFT-PGF compound treats physical ``0`` as the additive
        identity on each axis. Reduces to a plain zero-pad when both ``i0`` are 0.
        """
        n0, n1 = S.shape
        out = np.zeros((M0, M1))
        out[:n0 - i0_0, :n1 - i0_1] = S[i0_0:, i0_1:]
        if i0_1:
            out[:n0 - i0_0, M1 - i0_1:] = S[i0_0:, :i0_1]
        if i0_0:
            out[M0 - i0_0:, :n1 - i0_1] = S[:i0_0, i0_1:]
        if i0_0 and i0_1:
            out[M0 - i0_0:, M1 - i0_1:] = S[:i0_0, :i0_1]
        return out

    def update_work(self):
        """Assemble the joint density from the per-event severities ``g_i``.

        Builds the copula joint per-claim severity ``S`` (discrete Sklar), runs
        the 2D compound FFT (``freq_pgf`` applied elementwise to ``rfft2(S)``),
        then applies any per-axis ``pnl`` affine.

        The 2-D lift of the 1-D ``i0`` / ``j0`` machinery
        (:meth:`aggregate.distributions.Aggregate._fft_aggregate`): the
        per-claim severity is laid into the padded buffer with physical 0 at
        index 0 (each axis's ``i0`` negative-severity buckets wrapped to the top,
        :meth:`_lay_signed_2d`), the shared frequency is applied elementwise, and
        the finished density is relabelled onto each axis's output window by an
        ``np.roll`` of ``-j0``. Relabelling a finished array carries no ``N*s``
        shift, so it is correct for random as well as fixed frequency. The FFT
        buffer length ``M`` (:attr:`_mlog2`) is sized to reach physical 0 from
        the window, *decoupled* from the output length ``N`` so a tight window
        far from 0 does not alias. On a mass-at-0, 0-based grid (``i0 == j0 ==
        0``) the lay-in is a plain zero-pad and the roll is a no-op.
        """
        g0, g1 = self._gs
        i0_0, i0_1 = getattr(self, '_i0', [0, 0])
        j0_0, j0_1 = getattr(self, '_j0', [0, 0])
        N0, N1 = getattr(self, '_nout', [len(g0), len(g1)])
        mlog2 = getattr(self, '_mlog2', [int(np.log2(N0)) + self.padding,
                                         int(np.log2(N1)) + self.padding])
        # marginal CDFs in physical order (the copula couples ranks); the
        # Bernoulli zero-inflation is automatic as a jump in G.
        G0 = np.cumsum(g0)
        G1 = np.cumsum(g1)
        # joint per-claim severity via the copula (marginals exact by
        # construction); retained for the severity panel of plot().
        S = self.copula.rectangle_pmf(G0, G1)
        self._S = S

        if self.en == 0:
            density = np.zeros((N0, N1))
            density[-j0_0, -j0_1] = 1.0   # point mass at physical (0, 0)
        else:
            M0, M1 = 1 << mlog2[0], 1 << mlog2[1]
            buf = self._lay_signed_2d(S, i0_0, i0_1, M0, M1)
            z = sfft.rfft2(buf)
            # freq_pgf is mathematically elementwise in z, but the empirical
            # implementation assumes a 1D argument -- flatten, apply, reshape.
            ftagg = self.frequency.freq_pgf(self.en, z.ravel()).reshape(z.shape)
            a = np.real(sfft.irfft2(ftagg, s=(M0, M1)))
            # relabel physical 0 (buffer index 0) onto each axis output window
            a = np.roll(a, (-j0_0, -j0_1), axis=(0, 1))
            density = a[:N0, :N1]

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
