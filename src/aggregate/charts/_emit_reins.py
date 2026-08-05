"""Reinsurance chart emitter: the gross / ceded / net triple.

``chart_reins`` reads ``reins_density_df`` and emits the two-panel exhibit
the app draws today: a density panel and a survival panel over one shared
loss window, three series each.

The frame carries three triples, one per stage of the program, and which
one is drawn is a **semantic option** to the emitter rather than renderer
view state: they answer different questions of different contracts, and
they do not share a y scale in any meaningful sense.

======  ====================================================================
basis   the triple
======  ====================================================================
'sev'   the occurrence program seen per claim (``p_sev_*``)
'occ'   the aggregate before, ceded by, and after the occurrence program
'agg'   the aggregate cover: its subject, its cession, and the net
======  ====================================================================

On the 'agg' triple the first series is **subject**, never relabeled gross:
it equals true gross only when no occurrence program sits underneath it,
and calling it gross wherever one does would misstate the contract.

Survival is accumulated here rather than client side, through
:class:`~aggregate._grid_distribution.GridDistribution`: each column is a
pmf on one grid, so ``sf`` is exact (it agrees with the app's
``1 - cumsum`` to the last bit, verified at conversion). Values at or under
:data:`~aggregate.constants.LOG_FLOOR` are emitted as gaps, because a log
axis cannot place them and drawing them puts a fringe of arithmetic noise
where a reader expects tail.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._aggregate import Aggregate
from .._grid_distribution import GridDistribution
from ..constants import (LOG_FLOOR, REINS_LABEL_CEDED, REINS_LABEL_GROSS,
                         REINS_LABEL_NET, REINS_LABEL_SUBJECT)
from . import register_chart, _emitter_base
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel

__all__ = ['chart_reins']

#: ``basis -> (columns, roles, names)``, in draw order. Net draws last and
#: so on top: the reader's question is what did I keep, and the answer must
#: not be hidden under the subject it came from.
BASES = {
    'sev': (('p_sev_gross', 'p_sev_ceded', 'p_sev_net'),
            ('gross', 'ceded', 'net'),
            (REINS_LABEL_GROSS, REINS_LABEL_CEDED, REINS_LABEL_NET)),
    'occ': (('p_agg_gross', 'p_agg_ceded_occ', 'p_agg_net_occ'),
            ('gross', 'ceded', 'net'),
            (REINS_LABEL_GROSS, REINS_LABEL_CEDED, REINS_LABEL_NET)),
    'agg': (('p_agg_subject', 'p_agg_ceded', 'p_agg_net'),
            ('subject', 'ceded', 'net'),
            (REINS_LABEL_SUBJECT, REINS_LABEL_CEDED, REINS_LABEL_NET)),
}

#: The deepest survival the panel will show, one over the longest return
#: period worth drawing. Past it the curve is a line of float dust.
SURVIVAL_FLOOR = 1e-9

#: Fraction of the window added either side, so the curve is not drawn
#: hard against the frame.
WINDOW_PAD = 0.02

chart_reins = _emitter_base('reins')


def _cession_stages(agg):
    """The stages of ``agg`` that actually cede, as basis keys."""
    stages = []
    if getattr(agg, 'occ_reins', None) is not None:
        stages += ['occ', 'sev']
    if getattr(agg, 'agg_reins', None) is not None:
        stages += ['agg']
    return stages


def _has_cession(agg):
    """Availability gate: some stage of the program cedes something."""
    return bool(_cession_stages(agg))


def _loss_window(gd):
    """The shared x window, from the first (widest) series of the triple.

    Mirrors ``Aggregate._limits`` as the app does: a heavy tail otherwise
    squashes every visible mass into a sliver at the origin. An unsigned
    grid is read from zero, because starting a loss axis at ``q(0.001)``
    hides the mass at and near zero that a discrete book routinely has; a
    signed grid has no such anchor and takes ``q(0.001)``.
    """
    lo = float(gd.q(0.001)) if float(gd.x[0]) < 0 else min(0.0, float(gd.x[0]))
    hi = float(gd.q(0.999))
    if not hi > lo:
        return None
    pad = WINDOW_PAD * (hi - lo)
    return (lo - pad, hi + pad)


def _survival_window(survivals):
    """``[decade under the deepest survival drawn, 1]``.

    Fixing the axis at ``[SURVIVAL_FLOOR, 1]`` would be simpler and wastes
    the panel: a book whose deepest survival is 1e-3 would draw six empty
    decades. Rounding down to a decade keeps the gridlines on round numbers.
    """
    seen = [v for s in survivals for v in s if v is not None and v > LOG_FLOOR]
    lo = max(SURVIVAL_FLOOR, min(seen)) if seen else SURVIVAL_FLOOR
    return (float(10.0 ** np.floor(np.log10(lo))), 1.0)


@chart_reins.register(Aggregate)
def _reins(agg, basis=None):
    """Emit the gross / ceded / net two-panel chart for an aggregate.

    Parameters
    ----------
    agg : Aggregate
        Must carry a cession; :func:`~aggregate.charts.available_charts`
        answers ``['reins']`` exactly when it does.
    basis : str, optional
        Which triple to draw: 'sev', 'occ' or 'agg' (see the module
        docstring). Defaults to 'occ' when the occurrence program cedes,
        otherwise 'agg', so the default is always a triple that carries a
        cession.

    Returns
    -------
    ChartDoc
        Two 'xy' panels sharing one loss axis: 'density' (mass by loss) and
        'tail' (log survival, ``read_axis='y'``, because at a chosen
        survival the answer a reader wants is the loss).

    Raises
    ------
    ValueError
        For an unknown basis, or one whose stage cedes nothing.
    """
    stages = _cession_stages(agg)
    if basis is None:
        basis = 'occ' if 'occ' in stages else 'agg'
    if basis not in BASES:
        raise ValueError(f'unknown reinsurance basis {basis!r}; '
                         f'expected one of {tuple(BASES)}')
    if basis not in stages:
        raise ValueError(
            f'{agg.name!r} has no cession on the {basis!r} basis; '
            f'available: {stages or "none"}')

    df = agg.reins_density_df
    x = df['loss'].to_numpy(dtype=float)
    xs = tuple(float(v) for v in x)
    columns, roles, names = BASES[basis]

    grids = [GridDistribution(x, df[c].to_numpy(dtype=float), bs=agg.bs,
                              name=n)
             for c, n in zip(columns, names)]
    # A gap, not a zero: a log axis cannot place these, and drawing them at
    # the floor would read as a tail that is not there.
    survivals = [tuple(float(v) if v > LOG_FLOOR else None
                       for v in gd.sf(x))
                 for gd in grids]

    series = []
    for gd, role, name in zip(grids, roles, names):
        series.append(ChartSeries(name=name, role=role, panel_id='density',
                                  x=xs, y=tuple(float(v) for v in gd.p)))
    for surv, role, name in zip(survivals, roles, names):
        series.append(ChartSeries(name=name, role=role, panel_id='tail',
                                  x=xs, y=surv))

    return ChartDoc(
        name='reins',
        title=f'{agg.label}: {basis} gross, ceded and net',
        axes=(
            # One axis id referenced by both panels IS the shared window.
            ChartAxis(id='loss', label='Loss', unit='currency',
                      suggested_range=_loss_window(grids[0])),
            ChartAxis(id='density', label='Density', unit='density'),
            ChartAxis(id='survival', label='Survival', unit='probability',
                      scale='log',
                      suggested_range=_survival_window(survivals)),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='loss', y_axis='density',
                  title='Density'),
            Panel(id='tail', kind='xy', x_axis='loss', y_axis='survival',
                  read_axis='y', title='Survival'),
        ),
        series=tuple(series),
        meta={'basis': basis, 'bases_available': tuple(stages)},
    )


register_chart('reins', chart_reins, predicate=_has_cession)
