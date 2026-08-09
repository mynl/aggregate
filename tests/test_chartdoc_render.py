"""[Chart-Conversions] mpl acceptance gate: ChartDoc renders match baselines.

A conversion is licensed by a measurement, not by an argument. Where a
``plots/`` compositor existed, its output at gate setup was the baseline and
the generic renderer had to land pixel near identical on it; passing that is
what licensed deleting the compositor, and once it is gone the baseline is
simply the approved picture, held here so it cannot drift unnoticed.

Two of the baselines have moved deliberately, and both are recorded in
``CHANGELOG.md`` rather than left to be discovered:

* ``distortion`` agreed with its compositor at RMS 0 from ``1.0.0a209``
  through the commit that deleted it. It moved only when the legend restyled
  (smaller, and placed in the emptier upper corner), which is renderer-side
  and applies to every chart.
* ``agg`` never had a matching baseline: ``[Chart-Aggregate]`` deliberately
  changed what an aggregate draws, from three panels to two with the log
  reading declared rather than drawn. The baseline here is the new picture,
  generated after the author reviewed it.

Pins confirmed by the author 2026-08-05: ``_PINNED_MPL`` is the version the
baselines were rendered with (tests skip elsewhere rather than fail on font
or hinting differences), and ``_RMS_TOL`` is generous against the measured
residual. Regenerate by running this file with ``--regen``, which requires
having looked at what changed:

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

_AGG = 'agg CD.Book 100 claims sev lognorm 50 cv 2 poisson'


def _subjects():
    """``name -> ChartDoc``, built fresh so nothing caches across tests."""
    from aggregate import build
    from aggregate.charts import build_chart_doc
    return {
        'distortion': build_chart_doc(build('distortion CD.PH ph 0.7'),
                                      'distortion'),
        'agg': build_chart_doc(build(_AGG), 'agg'),
    }


def _render(doc, target):
    from aggregate.plots import plot_chartdoc, plt, use
    use()
    plot_chartdoc(doc).savefig(target, dpi=100)
    plt.close('all')


def _compare(actual, baseline_name):
    from matplotlib.testing.compare import compare_images
    result = compare_images(str(_BASELINES / baseline_name), str(actual),
                            tol=_RMS_TOL)
    assert result is None, result


@pytest.mark.parametrize('name', ['distortion', 'agg'])
def test_chartdoc_matches_baseline(name, tmp_path):
    """The gate: the rendered document is the picture that was approved."""
    target = tmp_path / f'{name}.png'
    _render(_subjects()[name], target)
    _compare(target, f'{name}.png')


def _regen():
    matplotlib.use('Agg')
    _BASELINES.mkdir(parents=True, exist_ok=True)
    for name, doc in _subjects().items():
        _render(doc, _BASELINES / f'{name}.png')
    print(f'baselines regenerated under {_BASELINES} '
          f'with matplotlib {matplotlib.__version__}')


if __name__ == '__main__' and '--regen' in sys.argv:
    _regen()
