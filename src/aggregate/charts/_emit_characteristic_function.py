"""Mass and Lee chart emitter for a ``chfagg``.

A law computed from its characteristic function is an aggregate's shape over
its own grid: the same two panels, the mass at each grid point and the
quantile function drawn sideways, over one outcome axis. What differs is the
companion. An aggregate draws its severity alongside; a ``chfagg`` has none,
so where the family has a scipy reference law the companion is that law's
bucket masses on the same grid, which is the teaching comparison (the grid
against the closed form) and otherwise there is no companion. A stable law's
reference is scipy's ``levy_stable``, too slow at a grid's size to draw, so
it is left out.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from ..characteristic_function import AggregateCHF
from . import register_chart, _emitter_base
from ._emit_aggregate import COMPANION_HEADROOM, outcome_doc
from ._two_panel import loss_window
from .ir import Mark

__all__ = ['chart_chfagg']

chart_chfagg = _emitter_base('chfagg')


def _updated(obj):
    """Availability: the grid has been computed (``update`` ran)."""
    return obj.agg_density is not None


def _reference_masses(obj):
    """The reference law's bucket masses on the grid, or ``None``.

    Only a scipy frozen law with a vectorized ``cdf`` qualifies, and not
    ``levy_stable``, whose ``cdf`` is numerical integration at every point.
    """
    ref = obj.family.reference
    if ref is None or not hasattr(ref, 'cdf') or hasattr(ref, 'agg_density'):
        return None
    name = getattr(getattr(ref, 'dist', None), 'name', '')
    if name == 'levy_stable':
        return None
    h = obj.bs
    with np.errstate(all='ignore'):
        masses = ref.cdf(obj.xs + h / 2) - ref.cdf(obj.xs - h / 2)
    return np.asarray(masses, dtype=float)


@chart_chfagg.register(AggregateCHF)
def _chfagg(obj):
    """Emit the two-panel mass and Lee chart for a ``chfagg``.

    Parameters
    ----------
    obj : AggregateCHF
        Must be updated.

    Returns
    -------
    ChartDoc
        The two 'xy' panels an aggregate emits, over one outcome axis: the
        mass at each grid point, with the reference law's masses alongside
        where the family has one, and the quantile function drawn sideways.
        The outcome axis declares a log reading only when the grid is
        nonnegative, since a signed window cannot take one.
    """
    df = obj.density_df
    x = df['loss'].to_numpy(dtype=float)
    mass = df['p_total'].to_numpy(dtype=float)
    companion = None
    peak = float(mass.max())
    ordinate_top = peak
    ref = _reference_masses(obj)
    if ref is not None and np.all(np.isfinite(ref)):
        companion = ('Reference', x, ref)
        ref_peak = float(ref.max())
        ordinate_top = max(peak, ref_peak) if ref_peak <= COMPANION_HEADROOM * peak else peak
    marks = [Mark(panel_id='density', orient='v', at=float(obj.est_m),
                  label='mean', role='mean')]
    signed = x[0] < 0
    return outcome_doc(
        'chfagg', str(obj.label), (str(obj.family.name), x, mass), companion,
        window=loss_window(obj.q, float(x[0])),
        full_range=(min(0.0, float(x[0])), float(x[-1])),
        ordinate_top=ordinate_top, marks=marks, step=obj.bs,
        outcome_scales=('linear',) if signed else ('linear', 'log'),
        is_loss_value=obj._is_loss_value)


register_chart('chfagg', chart_chfagg, predicate=_updated, primary=AggregateCHF)
