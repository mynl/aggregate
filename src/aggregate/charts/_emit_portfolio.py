"""Portfolio chart emitter: the book's mass, and where its losses come from.

Two panels. The first is the aggregate chart's density panel with every
unit on it beside the total, and the compositor's log-density panel is a
declared reading of it rather than a second picture, exactly as on the
aggregate.

The second is the **kappa** panel, and it is the reason a portfolio is not
just an aggregate with more curves. ``exeqa_i`` is
``E[X_i | X = x]``, what unit ``i`` contributes when the book as a whole
lands at ``x``, so reading up from a total loss gives the split that
produced it. Its curves sum to the diagonal by construction, which is the
identity the panel is read against and the one the floor below is measured
by.

**The floor.** Kappa is a conditional expectation, so it *divides* by
``p_total``, and where that mass is arithmetic dust the quotient is noise
rather than allocation. :data:`KAPPA_FLOOR` drops those points. The
constant is not a guess: because the unit curves must sum to ``x`` exactly,
the residual of that identity measures kappa's own error, and on a
two-lognormal book it runs (median, worst)

======================  ==============  ==============
``p_total`` floor       median          worst
======================  ==============  ==============
``1e-12``               ``4.6e-11``     ``4.1e-07``
``1e-14``               ``4.6e-09``     ``3.2e-04``
``1e-15``               ``7.3e-05``     ``1.2e-03``
none                    ``9.9e-05``     ``0.52``
======================  ==============  ==============

The cliff is real rather than gradual, and ``1e-14`` sits a decade above
it. The same numbers come back at ``log2`` 16 and 18, so the floor does not
need to scale with the grid, and on that book kappa stays trustworthy out
to a loss ten times ``q(0.999)``: inside the default window the floor never
bites at all, and it is protecting the zoomed-out reading.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._portfolio import Portfolio
from . import register_chart, _emitter_base
from ._emit_aggregate import CAPITAL_ANCHOR, COMPANION_HEADROOM
from ._payload import collapse_empty_runs, lattice_payload
from ._two_panel import loss_window
from .ir import ChartAxis, ChartDoc, ChartSeries, Mark, Panel, complete_tex

__all__ = ['chart_port']

#: Kappa is read only where the total carries more probability than this.
#: Measured rather than chosen: see the module docstring. A decade above
#: ``constants.LOG_FLOOR``, which is the dust floor for a mass that is only
#: *displayed*; a mass that is divided by needs the extra decade.
KAPPA_FLOOR = 1e-14

#: The kappa axis label, and its typeset form.
KAPPA_LABEL = 'E[Xi | X = x]'
KAPPA_TEX = r'$\mathsf{E}[X_i\mid X=x]$'

#: What the whole book's curve is called on both panels.
TOTAL_NAME = 'Total'

chart_port = _emitter_base('port')


def _updated(port):
    """Availability: there is a realized grid to draw."""
    return getattr(port, 'density_df', None) is not None


def _unit_label(port, agg):
    """The series name for one unit: its resolved label, or its handle."""
    return str(agg.label) if port.use_labels else str(agg.name)


@chart_port.register(Portfolio)
def _port(port, xmax=None):
    """Emit the two-panel mass and kappa chart for a portfolio.

    Parameters
    ----------
    port : Portfolio
        Must be updated.
    xmax : float, optional
        Upper end of the loss window, in place of the computed one, for
        reading two books against one common scale.

    Returns
    -------
    ChartDoc
        Two 'xy' panels over one shared loss axis: 'density' (the mass at
        each grid point, the units and the total) and 'kappa'
        (``E[X_i | X = x]``, what each unit contributes at a given total).
        The loss and mass axes declare a log reading, which is the
        compositor's old second panel.

    Notes
    -----
    Draw order is meaning: the units first and the **total last**, so the
    book sits on top of the parts it is made of. Each unit is drawn on its
    own native grid (``unit_density``), which a windowed book does not
    share with the portfolio's, rather than on a resampled common one.

    The kappa panel's total curve is the diagonal, exactly: ``exeqa_total``
    is ``E[X | X = x] = x``. It is drawn because the unit curves are read
    as a decomposition of it, and seeing them sum to it is the reading.
    """
    df = port.density_df
    x = df.loss.to_numpy(dtype=float)
    total = df.p_total.to_numpy(dtype=float)
    units = [(_unit_label(port, a), a) for a in port.agg_list]

    series, peaks = [], []
    for label, agg in units:
        unit = port.unit_density(agg.name)
        ux = unit.index.to_numpy(dtype=float)
        um = unit.to_numpy(dtype=float)
        peaks.append(float(um.max()))
        drawn_x, drawn_m = collapse_empty_runs(ux, um)
        series.append(ChartSeries(
            name=label, role='unit', panel_id='density',
            y=tuple(float(v) for v in drawn_m),
            **lattice_payload(drawn_x, agg.bs)))
    drawn_x, drawn_m = collapse_empty_runs(x, total)
    series.append(ChartSeries(
        name=TOTAL_NAME, role='total', panel_id='density',
        y=tuple(float(v) for v in drawn_m),
        **lattice_payload(drawn_x, port.bs)))

    # Kappa divides by p_total, so it is read only where that mass is more
    # than arithmetic dust; past the floor the quotient is noise, not
    # allocation.
    keep = total > KAPPA_FLOOR
    kx = x[keep]
    for label, agg in units:
        series.append(ChartSeries(
            name=label, role='unit', panel_id='kappa',
            y=tuple(float(v) for v in df[f'exeqa_{agg.name}'].to_numpy()[keep]),
            support='continuous', **lattice_payload(kx, port.bs)))
    series.append(ChartSeries(
        name=TOTAL_NAME, role='total', panel_id='kappa',
        y=tuple(float(v) for v in df['exeqa_total'].to_numpy()[keep]),
        support='continuous', **lattice_payload(kx, port.bs)))

    window = (loss_window(port.q, x[0]) if xmax is None
              else (min(0.0, float(x[0])), float(xmax)))
    # The total sets the ordinate, widened for a unit that nearly fits: a
    # thin unit peaks well above the book it is part of, and scaling to it
    # flattens the subject. The aggregate chart's rule, one book up.
    peak = float(total.max())
    tallest = max(peaks) if peaks else peak
    ordinate_top = (max(peak, tallest)
                    if tallest <= COMPANION_HEADROOM * peak else peak)
    anchor = float(port.q(1 - 1 / CAPITAL_ANCHOR))

    return complete_tex(ChartDoc(
        name='port',
        title=str(port.label),
        axes=(
            ChartAxis(id='outcome', label='Loss', unit='currency',
                      scales=('linear', 'log'), suggested_range=window,
                      full_range=(min(0.0, float(x[0])), float(x[-1]))),
            ChartAxis(id='mass', label='Probability mass', unit='density',
                      scales=('linear', 'log'),
                      suggested_range=(0.0, ordinate_top)),
            # Kappa is a loss, so it reads on the same kind of scale as the
            # outcome and offers the same log reading: which unit dominates
            # far out in the tail is a log-log question.
            #
            # Its window *is* the loss window, because the unit curves sum
            # to the diagonal: at the right edge of the visible losses the
            # tallest curve on the panel is the total, and it is there.
            # Without saying so the panel would scale to kappa at losses
            # far off the right of the shared window, and squash every
            # curve a reader can actually see into the bottom eighth.
            ChartAxis(id='kappa', label=KAPPA_LABEL, unit='currency',
                      scales=('linear', 'log'),
                      suggested_range=(min(0.0, window[0]), window[1]),
                      full_range=(min(0.0, window[0]), float(kx[-1]))),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='outcome', y_axis='mass',
                  title='Probability mass function'),
            # Equal aspect is semantic here: both axes are losses, and the
            # total's curve is the diagonal, so the reading is each unit's
            # slope against 45 degrees. A stretched box misstates it.
            Panel(id='kappa', kind='xy', x_axis='outcome', y_axis='kappa',
                  aspect='equal', title='Conditional loss by unit'),
        ),
        series=tuple(series),
        marks=(
            Mark(panel_id='density', orient='v', at=float(port.est_m),
                 label='mean', role='mean'),
            Mark(panel_id='density', orient='v', at=anchor,
                 label=f'1-in-{CAPITAL_ANCHOR}', role='capital_anchor'),
            # On the kappa panel the same line is the natural allocation at
            # capital: read up from it and each unit's share is its share of
            # the total loss there.
            Mark(panel_id='kappa', orient='v', at=anchor,
                 label=f'1-in-{CAPITAL_ANCHOR}', role='capital_anchor',
                 faint=True),
        ),
        meta={'ordinate': 'mass', 'kappa_floor': KAPPA_FLOOR,
              'return_period_map': 'complement'},
    ), {KAPPA_LABEL: KAPPA_TEX})


register_chart('port', chart_port, predicate=_updated, primary=Portfolio)
