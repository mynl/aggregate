"""Bounds chart emitter: the envelope of admissible distortions.

Every distortion pricing the book to its premium lies between two curves,
and that band is the whole subject: it says what prices are *possible*
before anyone argues about which is right. So the band is drawn twice, once
under the cloud that generates it and once under the calibrated distortions
that have to live inside it, and the reader's eye goes to the same region
both times.

**Two panels, where the compositor drew three.** It split the five
calibrated distortions across its last two panels, ``['ccoc', 'tvar']`` on
one and ``['ph', 'wang', 'dual']`` on the other, which is an accident of
the order they were added rather than a reading anyone wants: the question
is how the five compare, and five curves on one band answers it. The
consolidation is the author's decision, recorded in ``dev/plan-chart-ir.md``.

Both panels are equal-aspect unit squares, because concavity is what a
distortion plot is read for and a stretched box misrepresents it, and
neither declares a log or a full-range reading: the unit square *is* the
window, so a zoom-out button on it would do nothing and a log reading of it
says nothing.

The cloud is the one place a series carries a number of its own. Each curve
is one bracketing BiTVaR, and its weight is a fact a reader asks about
(which brackets carry the price), so it travels as ``ChartSeries.value``
and the renderer decides how to show it.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from ..bounds import Bounds
from . import register_chart, _emitter_base
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel, complete_tex

__all__ = ['chart_envelope']

#: Points along ``[0, 1]`` for a calibrated distortion curve. The cloud
#: brings its own grid (``cloud_df.index``), which is the one the envelope
#: was computed on and so the one the band must be drawn on.
CURVE_POINTS = 1001

#: The calibrated distortions, in the order a reader compares them, with
#: the names they are known by. All five on one band: which of them prices
#: highest where is the question, and it cannot be read off two panels.
CALIBRATED = {'ccoc': 'CCoC', 'tvar': 'TVaR(p*)', 'ph': 'PH',
              'wang': 'Wang', 'dual': 'Dual'}

chart_envelope = _emitter_base('envelope')


def _calibrated(bounds):
    """The calibrated distortions of the object these bounds are for.

    A :class:`Bounds` can be built on an ``Aggregate`` as readily as on a
    ``Portfolio``, and only a calibrated ``Portfolio`` carries a set, so
    this answers the empty dict rather than raising: the envelope is worth
    drawing on its own, and the document simply has one panel then.
    """
    found = getattr(bounds._obj, 'distortions', None) or {}
    return {label: found[key] for key, label in CALIBRATED.items()
            if key in found}


def _band(panel, s, lo, hi):
    """The min/max envelope as one band series over ``s``."""
    return ChartSeries(
        name='Envelope', role='distortion', panel_id=panel,
        x=tuple(float(v) for v in s), y=tuple(float(v) for v in lo),
        y2=tuple(float(v) for v in hi), support='continuous')


def _identity(panel):
    """The risk-neutral diagonal: the price with no load at all."""
    return ChartSeries(name='identity', role='identity', panel_id=panel,
                       x=(0.0, 1.0), y=(0.0, 1.0), support='continuous')


@chart_envelope.register(Bounds)
def _envelope(bounds, n_resamples=0, n=CURVE_POINTS):
    """Emit the envelope chart for a set of pricing bounds.

    Parameters
    ----------
    bounds : Bounds
        Carrying ``cloud_df`` and ``weight_df``.
    n_resamples : int, default 0
        How many bracketing curves to draw inside the band. A semantic
        option: it says how densely to show the set of admissible prices,
        and each curve carries its bracket's weight as its ``value``. Zero
        draws the band alone, which is the compositor's default too.
    n : int, default 1001
        Points along ``[0, 1]`` for each calibrated distortion curve.

    Returns
    -------
    ChartDoc
        One equal-aspect unit square with the envelope band and the cloud,
        and a second with the band again under every calibrated distortion,
        where the object carries any. Panel two is omitted rather than
        drawn empty when it would have nothing to say.

    Notes
    -----
    The band is a ``y2`` series: the series *is* the region between the two
    edges, which is what the envelope means, rather than two curves a
    reader has to associate. The average-extreme curve rides on the
    distortion panel as it does in the compositor, because what it is for
    is comparison against the calibrated set.
    """
    cloud = bounds.cloud_df
    s = cloud.index.to_numpy(dtype=float)
    lo = cloud.min(axis=1).to_numpy(dtype=float)
    hi = cloud.max(axis=1).to_numpy(dtype=float)
    dists = _calibrated(bounds)

    series = [_band('cloud', s, lo, hi)]
    if n_resamples > 0:
        # Brackets that pin the mean (p_lo == 0) are the pricing ones, the
        # same restriction the compositor samples under.
        sample = bounds.weight_df.xs(0, drop_level=False) \
                       .sample(n=n_resamples, replace=True).reset_index()
        for _, row in sample.iterrows():
            key = (row['p_lower'], row['p_upper'])
            series.append(ChartSeries(
                name=f'BiTVaR({key[0]:.4g}, {key[1]:.4g})', role='distortion',
                panel_id='cloud', x=tuple(float(v) for v in s),
                y=tuple(float(v) for v in cloud[key].to_numpy(dtype=float)),
                support='continuous', value=float(row['weight'])))
    series.append(_identity('cloud'))

    panels = [Panel(id='cloud', kind='xy', x_axis='s', y_axis='g',
                    aspect='equal', title='Envelope of admissible prices')]
    if dists:
        grid = np.linspace(0, 1, n)
        panels.append(Panel(id='calibrated', kind='xy', x_axis='s',
                            y_axis='g', aspect='equal',
                            title='Calibrated distortions'))
        series.append(_band('calibrated', s, lo, hi))
        for label, dist in dists.items():
            series.append(ChartSeries(
                name=label, role='distortion', panel_id='calibrated',
                x=tuple(float(v) for v in grid),
                y=tuple(float(v) for v in dist.g(grid)),
                support='continuous'))
        series.append(ChartSeries(
            name='Avg extreme', role='distortion', panel_id='calibrated',
            x=tuple(float(v) for v in s),
            y=tuple(float(v) for v in cloud.mean(axis=1).to_numpy(dtype=float)),
            support='continuous'))
        series.append(_identity('calibrated'))

    return complete_tex(ChartDoc(
        name='envelope',
        title=f'{bounds.name}: pricing bounds at {bounds.premium:.6g}',
        axes=(
            # No scales and no full_range: the unit square is the window,
            # and neither a log reading of it nor a zoom out of it exists.
            ChartAxis(id='s', label='s', unit='probability',
                      suggested_range=(0.0, 1.0)),
            ChartAxis(id='g', label='g(s)', unit='probability',
                      suggested_range=(0.0, 1.0)),
        ),
        panels=tuple(panels),
        series=tuple(series),
        meta={'p_star': float(bounds.p_star),
              'calibrated': tuple(dists),
              'value_label': 'Weight to upper threshold'},
    ))


register_chart('envelope', chart_envelope, primary=Bounds)
