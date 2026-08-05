"""[Chart-Conversions] mpl acceptance gate: ChartDoc renders match baselines.

The before side of each converted chart is its ``plots/`` compositor; the
baseline PNG committed under ``tests/data/chartdoc_baselines/`` is that
compositor's output at gate setup, and the gate is that the generic
renderer realizing the emitted ChartDoc lands pixel near identical on it.
A second test per chart keeps the compositor itself pinned to the baseline,
so drift on either side of the seam is caught, not just on the new one.

Pins confirmed by the author 2026-08-05: ``_PINNED_MPL`` is the version the
baselines were rendered with (tests skip elsewhere rather than fail on font
or hinting differences), and ``_RMS_TOL`` is generous against the measured
conversion residual, which since the axis-label and dual-name pass is 0:
the renderer reproduces the compositor exactly. Regenerate baselines by
running this file with ``--regen``:

    .venv/Scripts/python.exe tests/test_chartdoc_render.py --regen
"""

import sys
from pathlib import Path

import matplotlib
import pytest

# Image tests draw to a file and never to a screen. Without this they
# inherit whatever interactive backend is active, which fails
# intermittently (a Tk toolkit error mid-run) for reasons that have nothing
# to do with what is being compared. ``_regen`` sets it too.
matplotlib.use('Agg')

_PINNED_MPL = '3.10.9'
_RMS_TOL = 2.0
_BASELINES = Path(__file__).parent / 'data' / 'chartdoc_baselines'

pytestmark = pytest.mark.skipif(
    matplotlib.__version__ != _PINNED_MPL,
    reason=f'chartdoc baselines are pinned to matplotlib {_PINNED_MPL} '
           f'(installed: {matplotlib.__version__}); regenerate at gate '
           'confirmation')


def _distortion():
    from aggregate import Distortion
    return Distortion('ph', 0.7)


def _render(fig_or_ax, target):
    from aggregate.plots import plt
    fig = getattr(fig_or_ax, 'figure', fig_or_ax)
    fig.savefig(target, dpi=100)
    plt.close('all')


def _compare(actual, baseline_name):
    from matplotlib.testing.compare import compare_images
    result = compare_images(str(_BASELINES / baseline_name), str(actual),
                            tol=_RMS_TOL)
    assert result is None, result


def test_distortion_chartdoc_matches_baseline(tmp_path):
    """The conversion gate: the rendered ChartDoc reproduces the compositor."""
    from aggregate.charts import chart_distortion
    from aggregate.plots import plot_chartdoc, use
    use()
    target = tmp_path / 'distortion.png'
    _render(plot_chartdoc(chart_distortion(_distortion())), target)
    _compare(target, 'distortion.png')


def test_distortion_compositor_still_matches_baseline(tmp_path):
    """The before side stays pinned too, so the seam cannot drift silently."""
    from aggregate.plots import plot_distortion, use
    use()
    target = tmp_path / 'distortion.png'
    _render(plot_distortion(_distortion()), target)
    _compare(target, 'distortion.png')


def _regen():
    matplotlib.use('Agg')
    from aggregate.plots import plot_distortion, use
    use()
    _BASELINES.mkdir(parents=True, exist_ok=True)
    _render(plot_distortion(_distortion()),
            _BASELINES / 'distortion.png')
    print(f'baselines regenerated under {_BASELINES} '
          f'with matplotlib {matplotlib.__version__}')


if __name__ == '__main__' and '--regen' in sys.argv:
    _regen()
