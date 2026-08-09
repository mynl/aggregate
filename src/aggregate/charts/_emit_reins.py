"""Reinsurance chart emitter: the gross / ceded / net triple.

``chart_reins`` reads ``reins_density_df`` and emits the two-panel exhibit
the app draws today: a density panel and a survival panel over one shared
loss window, three series each.

The frame carries three triples, one per stage of the program, and which
one is drawn is a **semantic option** to the emitter rather than renderer
view state: they answer different questions of different contracts, and
they do not share a y scale in any meaningful sense.

=======  ===================================================================
basis    the triple
=======  ===================================================================
'sev'    the occurrence program seen per claim (``p_sev_*``)
'occ'    the aggregate before, ceded by, and after the occurrence program
'agg'    the aggregate cover: its subject, its cession, and the net
'total'  a **portfolio's** end-to-end gross, ceded and net
=======  ===================================================================

On the 'agg' triple the first series is **subject**, never relabeled gross:
it equals true gross only when no occurrence program sits underneath it,
and calling it gross wherever one does would misstate the contract.

A :class:`~aggregate.Portfolio` offers 'total' and nothing else. Its
``reins_density_df`` convolves each unit's **end-to-end** view, so there is
no book-wide occurrence stage to draw and no ``p_agg_subject``: units cede
on different stages, and a book-level 'occ' triple would have to pretend
they cede on the same one. 'total' is its own key rather than a reuse of
'agg' because the first series really is gross here, where on 'agg' it is
the subject. The emitter reports what it has in
``meta['bases_available']``, so a client offers the buttons that exist
rather than assuming three.

The three portfolio marginals are separate distributions and do not satisfy
``gross = net (+) ceded`` (see :attr:`Portfolio.reins_density_df`). The
chart draws three laws on one grid, which is exactly what it should show;
it is not a decomposition and the panel must not be read as one.

Survival is accumulated here rather than client side, through
:class:`~aggregate._grid_distribution.GridDistribution`: each column is a
pmf on one grid, so ``sf`` is exact (it agrees with the app's
``1 - cumsum`` to the last bit, verified at conversion). Values at or under
:data:`~aggregate.constants.LOG_FLOOR` are emitted as gaps, because a log
axis cannot place them and drawing them puts a fringe of arithmetic noise
where a reader expects tail.

Pure numpy and pandas; no matplotlib.
"""

from functools import singledispatch

from .._aggregate import Aggregate
from .._grid_distribution import GridDistribution
from .._portfolio import Portfolio
from ..constants import (REINS_LABEL_CEDED, REINS_LABEL_GROSS,
                         REINS_LABEL_NET, REINS_LABEL_SUBJECT)
from . import register_chart, _emitter_base
from ._two_panel import gapped, pad_window, survival_window
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel, complete_tex

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
    'total': (('p_agg_gross', 'p_agg_ceded', 'p_agg_net'),
              ('gross', 'ceded', 'net'),
              (REINS_LABEL_GROSS, REINS_LABEL_CEDED, REINS_LABEL_NET)),
}

chart_reins = _emitter_base('reins')


@singledispatch
def _cession_stages(obj):
    """The bases ``obj`` can actually draw, as :data:`BASES` keys.

    Dispatched rather than sniffed, because the answer is a different fact
    about each class: an aggregate's stages come from its own two
    reinsurance slots, a book's from whether any unit cedes at all.
    """
    return []


@_cession_stages.register(Aggregate)
def _agg_stages(agg):
    stages = []
    if agg.occ_reins is not None:
        stages += ['occ', 'sev']
    if agg.agg_reins is not None:
        stages += ['agg']
    return stages


@_cession_stages.register(Portfolio)
def _port_stages(port):
    # one end-to-end triple, and only when some unit cedes; reins_views is
    # empty exactly then, so the availability question is already answered
    return ['total'] if port.reins_views else []


def _has_cession(obj):
    """Availability gate: some stage of the program cedes something."""
    return bool(_cession_stages(obj))


def _loss_window(gd):
    """The shared x window, from the first (widest) series of the triple.

    Mirrors ``Aggregate._limits`` as the app does: a heavy tail otherwise
    squashes every visible mass into a sliver at the origin. An unsigned
    grid is read from zero, because starting a loss axis at ``q(0.001)``
    hides the mass at and near zero that a discrete book routinely has; a
    signed grid has no such anchor and takes ``q(0.001)``.
    """
    lo = float(gd.q(0.001)) if float(gd.x[0]) < 0 else min(0.0, float(gd.x[0]))
    return pad_window(lo, float(gd.q(0.999)))


def _emit(obj, basis, default):
    """Build the two-panel document for one basis of ``obj``.

    The whole body is class agnostic: it reads ``reins_density_df``, ``bs``
    and ``label``, which an ``Aggregate`` and a ``Portfolio`` both carry.
    Only the basis vocabulary differs, and that arrives resolved.

    Parameters
    ----------
    obj : Aggregate or Portfolio
    basis : str or None
        The requested basis; ``None`` takes ``default``.
    default : str
        The basis to draw when none is asked for. Always one that cedes.

    Returns
    -------
    ChartDoc

    Raises
    ------
    ValueError
        For an unknown basis, or one this object cannot draw.
    """
    stages = _cession_stages(obj)
    if basis is None:
        basis = default
    if basis not in BASES:
        raise ValueError(f'unknown reinsurance basis {basis!r}; '
                         f'expected one of {tuple(BASES)}')
    if basis not in stages:
        raise ValueError(
            f'{obj.name!r} has no cession on the {basis!r} basis; '
            f'available: {stages or "none"}')

    df = obj.reins_density_df
    x = df['loss'].to_numpy(dtype=float)
    xs = tuple(float(v) for v in x)
    columns, roles, names = BASES[basis]

    grids = [GridDistribution(x, df[c].to_numpy(dtype=float), bs=obj.bs,
                              name=n)
             for c, n in zip(columns, names)]
    survivals = [gapped(gd.sf(x)) for gd in grids]

    series = []
    for gd, role, name in zip(grids, roles, names):
        series.append(ChartSeries(name=name, role=role, panel_id='density',
                                  x=xs, y=tuple(float(v) for v in gd.p)))
    for surv, role, name in zip(survivals, roles, names):
        series.append(ChartSeries(name=name, role=role, panel_id='tail',
                                  x=xs, y=surv))

    return complete_tex(ChartDoc(
        name='reins',
        title=f'{obj.label}: {basis} gross, ceded and net',
        axes=(
            # One axis id referenced by both panels IS the shared window.
            ChartAxis(id='loss', label='Loss', unit='currency',
                      suggested_range=_loss_window(grids[0])),
            ChartAxis(id='density', label='Density', unit='density'),
            ChartAxis(id='survival', label='Survival', unit='probability',
                      scale='log',
                      suggested_range=survival_window(survivals)),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='loss', y_axis='density',
                  title='Density'),
            Panel(id='tail', kind='xy', x_axis='loss', y_axis='survival',
                  read_axis='y', title='Survival'),
        ),
        series=tuple(series),
        meta={'basis': basis, 'bases_available': tuple(stages)},
    ))


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
    return _emit(agg, basis, 'occ' if 'occ' in stages else 'agg')


@chart_reins.register(Portfolio)
def _reins_port(port, basis=None):
    """Emit the gross / ceded / net two-panel chart for a book.

    The same two panels the aggregate chart draws, over the portfolio's
    convolved end-to-end marginals. Its one basis is 'total': a book has no
    occurrence stage of its own, because its units cede on different ones.

    Parameters
    ----------
    port : Portfolio
        Must carry a cession on some unit;
        :func:`~aggregate.charts.available_charts` answers ``['reins']``
        exactly when one does.
    basis : str, optional
        Only 'total', which is also the default. Accepted for signature
        parity with the aggregate emitter, and refused by name otherwise
        rather than quietly drawing the one basis that exists.

    Returns
    -------
    ChartDoc
        Two 'xy' panels sharing one loss axis, as for an aggregate.

    Raises
    ------
    ValueError
        For any basis other than 'total'.

    Notes
    -----
    The three series are separate distributions, not a decomposition: they
    no more satisfy ``gross = net (+) ceded`` than the unit-level views do.
    """
    return _emit(port, basis, 'total')


register_chart('reins', chart_reins, predicate=_has_cession)
