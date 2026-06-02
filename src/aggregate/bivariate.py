"""Joint (ceded, net) occurrence aggregate distributions via 2D FFT.

Provides :class:`BivariateDistribution`, the container returned by
:meth:`aggregate.distributions.Aggregate.occ_bivariate`, holding the joint law
of the aggregate occurrence ceded ``C`` and net ``N`` losses under an
occurrence reinsurance program. Also provides the pure-array helpers
:func:`size_axis` (per-axis bucket / window sizing from a univariate aggregate
margin) and :func:`scatter_bivariate` (the 2D analogue of the reinsurance
rebucketing scatter), which :meth:`Aggregate.occ_bivariate` orchestrates.

``BivariateDistribution`` is reached as
``from aggregate.bivariate import BivariateDistribution``; nothing here is
re-exported at the top-level package namespace (submodule access only, per the
project layout convention).

Notes
-----
The mathematics is the ordinary compound-distribution FFT with the 1D
transforms replaced by 2D transforms. Per claim, the cession map sends a gross
loss ``X`` to the point ``(c(X), n(X))`` on the line ``c + n = X``; placing the
gross severity mass there builds a bivariate severity ``S``. Because the
frequency PGF ``freq_pgf(n, z)`` is *elementwise* in ``z`` (e.g. Poisson
``exp(n(z-1))``), it applies unchanged to the 2D transform ``FFT2(S)``, and the
joint aggregate density is ``iFFT2(freq_pgf(n, FFT2(S)))``. Marginalising the
result over one axis recovers the corresponding univariate occurrence
ceded / net aggregate, which gives exact validation targets.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import FIG_H, FIG_W
from .utilities import round_bucket

logger = logging.getLogger(__name__)


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


class BivariateDistribution(object):
    """Joint distribution of aggregate occurrence ceded ``C`` and net ``N``.

    A lightweight container for the 2D density returned by
    :meth:`aggregate.distributions.Aggregate.occ_bivariate`, with marginals,
    mixed moments, correlation, and a contour plot.

    Parameters
    ----------
    density : ndarray
        Joint probability mass, shape ``(len(ceded), len(net))``; entry
        ``[a, b]`` is ``P(C ~ ceded[a], N ~ net[b])``.
    ceded, net : ndarray
        1D ceded- and net-axis grids (``bs_ceded * arange`` /
        ``bs_net * arange``).
    bs_ceded, bs_net : float
        Ceded- and net-axis bucket sizes.
    meta : dict, optional
        Provenance: ``name``, ``en`` (``E[N]``), ``freq_name``, ``scheme``
        (rebucketing), ``padding``, ``deficit`` (lost tail mass).

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
        """Pearson correlation of aggregate ceded and net.

        Returns
        -------
        float
            ``Cov(C, N) / sqrt(Var(C) Var(N))``, in ``[-1, 1]``; ``nan`` if
            either margin is degenerate (zero variance).

        Notes
        -----
        Occurrence ceded and net are *positively* dependent: a large claim
        count drives both up together. Independence would require a degenerate
        (fixed) claim count.
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
        """Filled contour plot of the joint ``(ceded, net)`` density.

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
        ax.set(xlabel='Aggregate ceded', ylabel='Aggregate net',
               title=f'Joint occurrence ceded / net\n{self.meta.get("name", "")}')
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
        rows = [
            ('name', self.meta.get('name', '')),
            ('grid shape', f'{self.density.shape[0]} &times; {self.density.shape[1]}'),
            ('bucket (ceded, net)', f'{self.bs_ceded:g}, {self.bs_net:g}'),
            ('E[C] (ceded)', f'{e_c:,.6g}'),
            ('E[N] (net)', f'{e_n:,.6g}'),
            ('correlation', f'{rho:.6f}'),
            ('tail deficit', f'{deficit:.2e}'),
        ]
        body = ''.join(
            f'<tr><th style="text-align:left">{k}</th><td>{v}</td></tr>'
            for k, v in rows)
        return (f'<table class="aggregate bivariate">'
                f'<caption>BivariateDistribution</caption>{body}</table>')
