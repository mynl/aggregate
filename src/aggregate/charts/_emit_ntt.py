"""Exact tail chart emitter: an nttagg's exact answer as base-10 logarithms.

An ``nttagg`` computes its distribution exactly, as rationals, and the reason
to do that is the tail: the buckets where the float FFT answer has dissolved
into rounding noise and the exact one is still a number. The ``agg`` chart
cannot show them, because it draws ``p_total``, the float answer, and drawing
``p_exact_float`` on a log axis would not help either: a chart document carries
doubles, and the double of 1e-400 is 0. Only a document whose ordinate **is**
the logarithm can carry a bucket at 1e-462, so that is what this chart ships,
computed from the exact integers.

Two panels over one outcome axis. The mass panel is the exact probability of
each loss, the survival panel the exact probability of exceeding it, and each
carries the float FFT answer beside the exact one, so the reader sees where the
float runs into its noise floor and the exact line keeps going. The window is
the **whole support**, not ``agg``'s quantile crop, because the tail is the
point.

**Exact zeros are omitted, never drawn.** The logarithm of zero is not a
number, and a bucket the law cannot reach (an interior gap in a lattice book)
has no place on a log10 axis. The mass series therefore has holes, and
:func:`~aggregate.charts._payload.lattice_payload` notices on its own and ships
explicit coordinates. The renderers draw a ``'log10_density'`` series as
markers only, since a line would bridge the holes and claim mass where there is
none, and a log axis has no floor for a stem to stand on.

Pure numpy and pandas; no matplotlib.
"""

import math

import numpy as np

from ..ntt import AggregateNTT
from . import register_chart, _emitter_base
from ._payload import lattice_payload
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel, complete_tex

__all__ = ['chart_exact_tail']

#: How the two log10 axis labels typeset. The plain forms are what a browser
#: reader prints; matplotlib reads these.
TYPESET = {
    'log10 probability mass': r'$\log_{10}$ probability mass',
    'log10 exceedance probability': r'$\log_{10}$ exceedance probability',
}

chart_exact_tail = _emitter_base('exact_tail')


def _updated(a):
    """Availability: the exact answer has been computed (``update`` ran)."""
    return a._ntt_numerators is not None


def _window_floor(values):
    """Bottom of a log10 window: the lowest finite value, down to a multiple of 10.

    Parameters
    ----------
    values : ndarray
        Base-10 exponents, every one finite.

    Returns
    -------
    float
        At most -10, so the window never collapses to a point: a degenerate
        law whose one atom has probability 1 has every exponent at 0.
    """
    if not values.size:
        return -10.0
    return float(min(10 * math.floor(float(values.min()) / 10), -10))


@chart_exact_tail.register(AggregateNTT)
def _exact_tail(a):
    """Emit the exact log10 mass and survival chart for an exact aggregate.

    Parameters
    ----------
    a : AggregateNTT
        Must be updated, so the exact numerators exist.

    Returns
    -------
    ChartDoc
        Two 'xy' panels over one shared outcome axis: 'mass' (the base-10
        logarithm of the probability at each loss) and 'survival' (the base-10
        logarithm of the probability of exceeding it). Each carries an 'Exact'
        series computed from the integers and a 'Float FFT' series computed
        from ``density_df``, the latter only where the float is positive.

    Notes
    -----
    Every ordinate is an exponent: -400 stands for 1e-400. The exact mass is
    ``density_df.p_exact_log10`` and the exact survival is
    ``log10(sum_{j > i} H_j) - log10(D)`` over the integer numerators ``H``
    and common denominator ``D``; both are finite for any positive value
    however small, which is the one thing the float columns cannot promise.

    The exact survival is positive at every loss below ``top``, the largest
    loss with positive exact mass, and exactly zero from ``top`` on, so its
    series is a gap-free prefix of the grid. The exact mass can have interior
    zeros, which are dropped; ``lattice_payload`` then falls back to explicit
    x on its own.
    """
    df = a.density_df
    x = df.loss.to_numpy(dtype=float)
    exact_mass = df.p_exact_log10.to_numpy(dtype=float)
    exact_survival = np.asarray(a._exact_survival_log10_on_grid(), dtype=float)
    float_mass = df.p_total.to_numpy(dtype=float)
    float_survival = df.S.to_numpy(dtype=float)

    def series(name, role, panel_id, keep, values):
        """One series over the kept buckets, its x a lattice where it can be."""
        return ChartSeries(name=name, role=role, panel_id=panel_id,
                           y=tuple(float(v) for v in values),
                           **lattice_payload(x[keep], a.bs))

    exact_mass_keep = np.isfinite(exact_mass)
    exact_survival_keep = np.isfinite(exact_survival)
    # Masked before the logarithm, never after it, so no log10 of zero or of
    # a negative rounding residue is ever formed.
    float_mass_keep = float_mass > 0
    float_survival_keep = float_survival > 0
    exact_mass_drawn = exact_mass[exact_mass_keep]
    exact_survival_drawn = exact_survival[exact_survival_keep]
    top = float(x[exact_mass_keep].max())

    return complete_tex(ChartDoc(
        name='exact_tail',
        title=str(a.label),
        axes=(
            ChartAxis(id='outcome', label='Loss', unit='currency',
                      scales=('linear', 'log'),
                      suggested_range=(0.0, top),
                      full_range=(0.0, float(x[-1]))),
            ChartAxis(id='log10_mass', label='log10 probability mass',
                      unit='log10_density', scales=('linear',),
                      suggested_range=(_window_floor(exact_mass_drawn), 0.0)),
            ChartAxis(id='log10_survival',
                      label='log10 exceedance probability',
                      unit='log10_probability', scales=('linear',),
                      suggested_range=(_window_floor(exact_survival_drawn),
                                       0.0)),
        ),
        panels=(
            Panel(id='mass', kind='xy', x_axis='outcome',
                  y_axis='log10_mass', title='Exact probability mass'),
            Panel(id='survival', kind='xy', x_axis='outcome',
                  y_axis='log10_survival', title='Exact survival function'),
        ),
        series=(
            series('Exact', 'density', 'mass', exact_mass_keep,
                   exact_mass_drawn),
            series('Float FFT', 'density', 'mass', float_mass_keep,
                   np.log10(float_mass[float_mass_keep])),
            series('Exact', 'survival', 'survival', exact_survival_keep,
                   exact_survival_drawn),
            series('Float FFT', 'survival', 'survival', float_survival_keep,
                   np.log10(float_survival[float_survival_keep])),
        ),
    ), TYPESET)


register_chart('exact_tail', chart_exact_tail, predicate=_updated)
