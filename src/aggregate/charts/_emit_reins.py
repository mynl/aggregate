"""Reinsurance chart emitter: an occurrence program, per claim and in total.

The chart is the occurrence-reinsurance plot: what the program does to a
single claim, and what that does to the year. So the two panels answer two
different questions and share nothing, not even a loss axis, because a
per-claim loss and an annual aggregate are not the same quantity and
drawing them against one window would say they were.

**Left, per claim.** The gross, ceded and net severity as the treaty sees
each claim, from the ``p_sev_*`` triple. Read on log and only on log: a
layered severity puts most of its mass in a spike the linear reading
flattens everything else against, so this axis declares no linear alternative
rather than offering a control that draws a worse picture.

**Right, in total.** The same three distributions for the year, as a Lee
diagram, so a chosen probability reads off as a loss. It is the panel the
readings live on: log on both axes, the return-period reading of its
probability axis, and the inversion that turns it into the distribution
function.

**Aggregate only, at 1.0** (author's decision). A portfolio's units cede on
different stages, so a book-level triple would have to pretend they cede on
the same one, and the aggregate cover is a separate contract with a
separate picture. Both are restorable; neither is guessed at here.

The three curves are separate distributions, not a decomposition: they no
more satisfy ``gross = net (+) ceded`` than a portfolio's marginals do. The
panel draws three laws on one grid, which is what it should show, and it
must not be read as an accounting.

Survival is not accumulated here any more: the right panel is a quantile
curve, and :func:`~aggregate.charts._two_panel.quantile_curve` builds it
from each column's own pmf, trimmed to its support at both ends.

Pure numpy and pandas; no matplotlib.
"""

from .._aggregate import Aggregate
from .._grid_distribution import GridDistribution
from ..constants import (REINS_LABEL_CEDED, REINS_LABEL_GROSS,
                         REINS_LABEL_NET)
from . import register_chart, _emitter_base
from ._payload import collapse_empty_runs, lattice_payload
from ._two_panel import SURVIVAL_FLOOR, loss_window, quantile_curve
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel, complete_tex

__all__ = ['chart_reins']

#: The per-claim triple and the annual one, in draw order. Net draws last
#: and so on top: the reader's question is what did I keep, and the answer
#: must not be hidden under the subject it came from.
SEV_COLUMNS = ('p_sev_gross', 'p_sev_ceded', 'p_sev_net')
AGG_COLUMNS = ('p_agg_gross', 'p_agg_ceded_occ', 'p_agg_net_occ')
ROLES = ('gross', 'ceded', 'net')
NAMES = (REINS_LABEL_GROSS, REINS_LABEL_CEDED, REINS_LABEL_NET)

#: How far below the occurrence limit the per-claim window starts, as a
#: fraction of it. The compositor's ``-l / 50``: a cession is bounded by
#: its limit, so the limit is the window, and the sliver below zero is what
#: keeps a mass at zero off the frame.
LIMIT_PAD = 0.02

chart_reins = _emitter_base('reins')


def _has_occurrence(agg):
    """Availability: there is an occurrence program to draw."""
    return getattr(agg, 'occ_reins', None) is not None


def _claim_window(agg):
    """The per-claim window: the occurrence limit, which bounds the cession.

    Falls back to the drawn support where the program is unlimited, since
    an infinite limit is not a window.
    """
    limit = agg.spec.get('exp_limit', float('inf'))
    limit = float(limit) if not hasattr(limit, '__len__') else float(max(limit))
    if not limit < float('inf'):
        return None
    return (-LIMIT_PAD * limit, limit * (1 + LIMIT_PAD / 2))


@chart_reins.register(Aggregate)
def _reins(agg):
    """Emit the occurrence-reinsurance chart for an aggregate.

    Parameters
    ----------
    agg : Aggregate
        Must carry an occurrence program;
        :func:`~aggregate.charts.available_charts` answers ``'reins'``
        exactly when it does.

    Returns
    -------
    ChartDoc
        Two 'xy' panels with **no shared axis**: 'occurrence' (the gross,
        ceded and net severity per claim, read on log) and 'aggregate' (the
        same three for the year, as a Lee diagram). The aggregate panel
        carries every reading: log on both axes, the paired return period,
        and the inversion to the distribution function.
    """
    df = agg.reins_density_df
    x = df['loss'].to_numpy(dtype=float)
    grid = lattice_payload(x, agg.bs)

    series = []
    for column, role, name in zip(SEV_COLUMNS, ROLES, NAMES):
        mass = df[column].to_numpy(dtype=float)
        drawn_x, drawn_mass = collapse_empty_runs(x, mass)
        series.append(ChartSeries(
            name=name, role=role, panel_id='occurrence',
            y=tuple(float(v) for v in drawn_mass),
            **lattice_payload(drawn_x, agg.bs)))
    tops = []
    for column, role, name in zip(AGG_COLUMNS, ROLES, NAMES):
        p, outcome = quantile_curve(x, df[column].to_numpy(dtype=float))
        tops.append(float(outcome[-1]) if outcome.size else 0.0)
        series.append(ChartSeries(
            name=name, role=role, panel_id='aggregate',
            x=tuple(float(v) for v in p),
            **lattice_payload(outcome, agg.bs, 'y')))

    # The annual window comes from the gross curve, the widest of the
    # three: a cession is bounded by its subject.
    gross = GridDistribution(x, df[AGG_COLUMNS[0]].to_numpy(dtype=float),
                             bs=agg.bs, name=NAMES[0])
    annual = loss_window(gross.q, float(x[0]))
    claim = _claim_window(agg)

    return complete_tex(ChartDoc(
        name='reins',
        title=f'{agg.label}: occurrence program, per claim and in total',
        axes=(
            ChartAxis(id='claim', label='Loss per claim', unit='currency',
                      suggested_range=claim,
                      full_range=None if claim is None
                      else (min(0.0, float(x[0])), float(x[-1]))),
            # Log only. A layered severity is a spike and a tail, and the
            # linear reading of it is a spike and nothing else, so there is
            # no second reading to offer.
            ChartAxis(id='sev_density', label='Occurrence density',
                      unit='density', scale='log'),
            ChartAxis(id='p', label='Non-exceeding probability',
                      unit='probability', suggested_range=(0.0, 1.0)),
            ChartAxis(id='annual', label='Aggregate loss', unit='currency',
                      scales=('linear', 'log'), suggested_range=annual,
                      full_range=(min(0.0, float(x[0])), max(tops))),
            # Not named by any panel: the alternative reading of 'p'.
            ChartAxis(id='return_period', label='Return period',
                      unit='return_period', scale='log', reciprocal_of='p',
                      suggested_range=(1.0,
                                       float(round(1.0 / SURVIVAL_FLOOR)))),
        ),
        panels=(
            Panel(id='occurrence', kind='xy', x_axis='claim',
                  y_axis='sev_density', title='Occurrence'),
            Panel(id='aggregate', kind='xy', x_axis='p', y_axis='annual',
                  invertible=True, title='Aggregate',
                  inverse_title='Distribution function'),
        ),
        series=tuple(series),
        meta={'ordinate': 'mass', 'return_period_map': 'complement'},
    ))


register_chart('reins', chart_reins, predicate=_has_occurrence,
               primary=None)
