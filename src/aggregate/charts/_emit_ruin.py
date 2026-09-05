"""Ruin chart emitter: sample surplus paths and the psi(u) curve.

The served form of :func:`aggregate.pedagogy.ruin_example`: the left panel
draws about fifty sample surplus paths ``U(t) = u + c t - S(t)`` with the
expected trend, the law-of-the-iterated-logarithm funnel and the rug of
simulated ruin times; the right panel draws the exact probability of
eventual ruin ``psi(u)`` over initial surplus with a marker at the resolved
``(u, psi(u))``. Both read straight off :meth:`Aggregate._ruin_paths`, the
one simulation core the docs figure also consumes, so the two cannot drift
(``dev/plan-pk-tab.md``).

Payload control is mandatory here (``[Ruin-Downsampling]``): a raw path
carries two points per claim over the whole horizon and fifty of them would
repeat the approximation chart's megabyte documents. Each path is decimated
to about ``detail`` points **preserving every interval's running minimum**,
so a dip below zero is never thinned away and a ruined path keeps its exact
ruin point; path coordinates are rounded to six significant figures, which
is far below drawing resolution and roughly halves the byte weight. The
``psi(u)`` curve is sampled at about 256 grid points log-spaced toward the
tail, where the curve is flat in linear ``u``; its values are the exact law
and are not rounded.

The document carries **no marks**: the resolved reading is the one-point
``marker`` series on the psi panel plus the ``meta`` scalars, from which
the pane labels itself with no extra request.

Pure numpy; no matplotlib.
"""

import numpy as np

from .._aggregate import _RUIN_SEED, Aggregate, _resolve_margin
from . import register_chart, _emitter_base
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel, complete_tex

__all__ = ['chart_ruin']

chart_ruin = _emitter_base('ruin')

#: Simulated paths behind the served probability estimate: the teaching-aid
#: sizing ruling (``dev/plan-pk-tab.md`` ruling 4). Not an option, so a
#: deployment cannot be talked into a production-sized simulation through
#: the chart route.
RUIN_SIMS = 1000


def _ruin_available(agg):
    """Availability: an updated grid and a frequency a solver serves."""
    return (getattr(agg, 'agg_density', None) is not None
            and getattr(agg.frequency, 'freq_name', '')
            in ('poisson', 'renewal'))


def _decimate_path(tt, uu, budget):
    """Thin one path to about ``budget`` points, keeping every dip.

    Parameters
    ----------
    tt, uu : ndarray
        The path's claim times and interleaved pre/post surplus values.
    budget : int
        Target point count; a shorter path passes through unchanged.

    Returns
    -------
    (ndarray, ndarray)
        The thinned ``(tt, uu)``.

    Notes
    -----
    The index range is split into ``budget // 2`` chunks and each chunk
    contributes its running-minimum point and its last point. Keeping the
    minimum is the load-bearing half: a dip below zero is the fact the
    picture exists to show, and even thinning must not lose one. Keeping
    the last point means the final point of a ruined path (its exact ruin
    point) always survives.
    """
    n = len(tt)
    if n <= budget:
        return tt, uu
    keep = []
    for chunk in np.array_split(np.arange(n), max(1, budget // 2)):
        keep.append(chunk[np.argmin(uu[chunk])])
        keep.append(chunk[-1])
    idx = np.unique(np.array(keep))
    return tt[idx], uu[idx]


def _sig6(values):
    """Tuple of floats rounded to six significant figures.

    Sample-path coordinates only: six figures sit far below any drawing
    resolution and roughly halve the JSON byte weight. Never applied to
    the exact ``psi(u)`` law.
    """
    return tuple(float(f'{v:.6g}') for v in values)


@chart_ruin.register(Aggregate)
def _ruin(agg, rho=None, lr=None, p=None, u=None, log2=None,
          seed=_RUIN_SEED, n_plot=50, detail=192):
    """Emit the two-panel eventual-ruin chart for an aggregate.

    Parameters
    ----------
    agg : Aggregate
        Must be updated, with a poisson or renewal frequency.
    rho : float, optional
        Margin-to-loss ratio; the premium rate is
        ``c = (1 + rho) E[X] / E[W]``. At most one of ``rho`` or ``lr``;
        with neither, the teaching default ``rho = 0.2``.
    lr : float, optional
        The same margin stated as a loss ratio in ``(0, 1)``.
    p : float, optional
        Probability of eventual default, resolved to an initial surplus
        through the ruin function's capital lookup. At most one of ``p``
        or ``u``; with neither, the teaching default ``p = 0.05``.
    u : float, optional
        Initial surplus directly.
    log2 : int, optional
        Renewal path only: forwarded to :meth:`Aggregate.wiener_hopf`.
    seed : int or None
        rng seed; the fixed default keeps the document hash-stable, and
        ``None`` (the Sample action) draws a fresh seed and reports it in
        ``meta``.
    n_plot : int, default 50
        Sample paths drawn.
    detail : int, default 192
        Per-path point budget for the decimation; see
        :func:`_decimate_path`.

    Returns
    -------
    ChartDoc
        Panel ``paths``: the sample paths (role ``sample``, a ruined
        path carrying its ruin time as the series ``value``), the
        two-point expected trend (role ``mean``), the LIL funnel band
        (role ``band``) and the rug of simulated ruin times (role
        ``rug``, drawn at zero). Panel ``psi``: the exact ``psi(u)``
        curve (role ``survival``, log-capable axis) and the one-point
        ``marker`` series at the resolved ``(u, psi(u))``. ``meta``
        carries the scalars the ``ruin`` exhibit also serves, so the
        pane labels itself from the document alone.
    """
    # bare-call defaults: available_charts promises a drawable chart, so
    # a call with no options must draw the teaching default reading
    if rho is None and lr is None:
        rho = 0.2
    rho = _resolve_margin(rho, lr)
    if p is None and u is None:
        p = 0.05
    rp = agg._ruin_paths(rho, u, p=p, log2=log2, n_sims=RUIN_SIMS,
                         n_plot=n_plot, seed=seed)
    detail = max(16, int(detail))

    series = []
    n_fail = 0
    u_lo = 0.0
    u_hi = rp.u0
    for i, (tt, uu, ruined_at) in enumerate(rp.paths):
        kw = {}
        if ruined_at:
            n_fail += 1
            tt, uu = tt[:ruined_at + 1], uu[:ruined_at + 1]
            kw['value'] = float(tt[-1])          # the ruin time
        tt, uu = _decimate_path(tt, uu, detail)
        u_lo = min(u_lo, float(uu.min()))
        u_hi = max(u_hi, float(uu.max()))
        series.append(ChartSeries(
            name=f'path {i + 1}', role='sample', panel_id='paths',
            x=_sig6(tt), y=_sig6(uu), **kw))

    drift = rp.c - rp.mx / rp.mw
    series.append(ChartSeries(
        name='Expected trend', role='mean', panel_id='paths',
        support='continuous',
        x=(0.0, float(rp.t_plot)),
        y=(float(rp.u0), float(rp.u0 + drift * rp.t_plot))))

    # the funnel is smooth, so a few dozen samples draw it exactly enough
    base = rp.u0 + drift * rp.tl
    idx = np.unique(np.linspace(0, len(rp.tl) - 1, 64).round().astype(int))
    series.append(ChartSeries(
        name='LIL funnel', role='band', panel_id='paths',
        support='continuous',
        x=_sig6(rp.tl[idx]),
        y=_sig6((base - rp.band)[idx]),
        y2=_sig6((base + rp.band)[idx])))

    rt = np.sort(rp.ruin_time[np.isfinite(rp.ruin_time)])
    rt = rt[rt <= rp.t_plot]
    if len(rt) > 256:
        rt = rt[np.unique(np.linspace(0, len(rt) - 1, 256).round()
                          .astype(int))]
    if len(rt):
        series.append(ChartSeries(
            name='Ruin times', role='rug', panel_id='paths',
            x=_sig6(rt), y=(0.0,) * len(rt)))

    # psi(u) on ~256 grid points log-spaced toward the flat tail; the
    # exact law, never rounded
    ruin = rp.rf.ruin
    psi = ruin.to_numpy()
    n = len(ruin)
    i_top = min(int(np.searchsorted(-psi, -1e-5)), n - 1)
    u_top = max(float(ruin.index[i_top]), 1.25 * rp.u0)
    i_max = min(int(round(u_top / agg.bs)), n - 1)
    ii = np.unique(np.concatenate(
        ([0], np.geomspace(1, max(i_max, 1), 255).round()))
        .astype(int))
    ii = ii[ii <= i_max]
    series.append(ChartSeries(
        name='psi(u)', role='survival', panel_id='psi',
        support='continuous',
        x=tuple(float(v) for v in ruin.index[ii]),
        y=tuple(float(v) for v in psi[ii])))
    series.append(ChartSeries(
        name='psi at u', role='marker', panel_id='psi',
        x=(float(rp.u0),), y=(float(rp.psi_u0),)))

    return complete_tex(ChartDoc(
        name='ruin',
        title=f'{agg.label}: probability of eventual ruin',
        axes=(
            ChartAxis(id='time', label='Time', unit='time',
                      suggested_range=(0.0, float(rp.t_plot))),
            ChartAxis(id='surplus', label='Surplus', unit='currency',
                      suggested_range=(u_lo, u_hi)),
            ChartAxis(id='capital', label='Initial surplus',
                      unit='currency',
                      suggested_range=(-u_top / 50, u_top),
                      full_range=(0.0, float(ruin.index[-1]))),
            ChartAxis(id='ruin_probability',
                      label='Probability of eventual ruin',
                      unit='probability', scales=('linear', 'log'),
                      suggested_range=(0.0, 1.0)),
        ),
        panels=(
            Panel(id='paths', kind='xy', x_axis='time', y_axis='surplus',
                  title=f'{len(rp.paths)} sample paths, {n_fail} ruined; '
                        f'simulated {rp.n_ruin} of {rp.n_sims} = '
                        f'{rp.p_sim:.1%}, exact {rp.psi_u0:.1%}'),
            Panel(id='psi', kind='xy', x_axis='capital',
                  y_axis='ruin_probability',
                  title=f'psi({rp.u0:.6g}) = {rp.psi_u0:.2%}'),
        ),
        series=tuple(series),
        meta={
            'rho': float(rp.rho),
            'lr': float(1.0 / (1.0 + rp.rho)),
            'u': float(rp.u0),
            'p': None if p is None else float(p),
            'psi_exact': float(rp.psi_u0),
            'psi_sim': float(rp.p_sim),
            'se_sim': float(rp.se_sim),
            'seed': int(rp.seed),
            'n_sims': int(rp.n_sims),
            'n_plot': int(len(rp.paths)),
            'n_ruin': int(rp.n_ruin),
            'freq_kind': rp.fname,
            'premium_rate': float(rp.c),
            'mean_severity': float(rp.mx),
            'mean_wait': float(rp.mw),
        },
    ))


register_chart('ruin', chart_ruin, predicate=_ruin_available)
