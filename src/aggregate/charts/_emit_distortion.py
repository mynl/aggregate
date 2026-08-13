"""Distortion chart emitters: g(s) on the unit square.

``chart_distortion`` reads the knot-spliced ``density_df`` grid (the TVaR
kink and BiTVaR/WtdTVaR knots are meaning, so the frame is read rather than
``g`` re-evaluated, exactly as ``plots._distortion`` does) and returns one
equal-aspect 'xy' panel: the distortion curve, optionally its dual, and the
identity diagonal (the risk-neutral reference; the area between g and the
diagonal is the load).

Pure numpy and pandas; no matplotlib.
"""

from ..constants import DISTORTION_DUAL_LABEL, DISTORTION_DUAL_TEX
from ..spectral import Distortion
from . import register_chart, _emitter_base
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel, complete_tex

__all__ = ['chart_distortion']

chart_distortion = _emitter_base('distortion')


@chart_distortion.register(Distortion)
def _distortion(dist, dual=True):
    """Emit the g(s) unit-square chart for a distortion.

    Parameters
    ----------
    dist : Distortion
    dual : bool
        Include the dual distortion curve. The app's exhibit draws g and
        the identity only (``dual=False``); the library compositor's
        default view carries both.

    Returns
    -------
    ChartDoc
        One 'xy' panel with ``aspect='equal'``: the unit square is
        semantic, because concavity is the only thing anyone reads a
        distortion plot to see, and a stretched aspect misrepresents it.
        Series order is g, dual, identity; draw order is meaning (the
        reference draws last, underneath nothing).
    """
    df = dist.density_df
    xs = tuple(float(v) for v in df.index.to_numpy())
    # Continuous, and one of the few things here that is: g is a function
    # of s defined at every s, sampled on the knot-spliced grid. Nothing
    # about a distortion is discretized.
    series = [
        ChartSeries(name=str(dist.label), role='distortion',
                    panel_id='square', x=xs, support='continuous',
                    y=tuple(float(v) for v in df['g'].to_numpy())),
    ]
    if dual:
        # Plain text, per the schema rule: ECharts has no TeX, so the name
        # itself must read anywhere, and the typeset form travels in the
        # document's `tex` map for renderers that can use it.
        series.append(
            ChartSeries(name=DISTORTION_DUAL_LABEL, role='distortion',
                        panel_id='square', x=xs, support='continuous',
                        y=tuple(float(v) for v in df['g_dual'].to_numpy())))
    series.append(
        ChartSeries(name='identity', role='identity', panel_id='square',
                    x=(0.0, 1.0), y=(0.0, 1.0), support='continuous'))
    return complete_tex(
        ChartDoc(
            name='distortion',
            title=str(dist.label),
            axes=(
                ChartAxis(id='s', label='s', unit='probability',
                          suggested_range=(0.0, 1.0)),
                ChartAxis(id='g', label='g(s)', unit='probability',
                          suggested_range=(0.0, 1.0)),
                # Named by no panel: the reflected reading of each axis.
                # Reflecting both gives the dual distortion, so the labels
                # are the literal coordinates rather than s and g-check:
                # the reflected point is (1 - s, 1 - g(s)) for every series
                # on the panel, where naming it the dual asserts an
                # identity that holds only of the g curve. Nothing about
                # the unit square is log readable in either reading, so no
                # scales are offered, and both windows stay (0, 1), which
                # is what keeps the square square under the reflection.
                ChartAxis(id='s_complement', label='1 - s',
                          unit='probability', complement_of='s',
                          suggested_range=(0.0, 1.0)),
                ChartAxis(id='g_complement', label='1 - g(s)',
                          unit='probability', complement_of='g',
                          suggested_range=(0.0, 1.0)),
            ),
            panels=(
                Panel(id='square', kind='xy', x_axis='s', y_axis='g',
                      aspect='equal'),
            ),
            series=tuple(series),
        ),
        {DISTORTION_DUAL_LABEL: DISTORTION_DUAL_TEX} if dual else None,
    )


register_chart('distortion', chart_distortion, primary=Distortion)
