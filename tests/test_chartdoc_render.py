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
* ``structure`` (``[Tower-Renderer]``) had no compositor either: the chart
  is new at 1.0.0a350 and the tower panel kind is new with it, so the
  baseline is the approved picture from the start. It moved once, at
  1.0.0a352, when the loss axis learned to reach the policy limit.
* ``structure_log`` is the same document on ``log='y'``, added at
  1.0.0a353. It is the reading a geometrically layered program is meant to
  be read on, and it gates three things a linear render cannot: a block
  starting at an exact zero drawn from the panel's decade floor rather than
  from negative infinity, a boundary mark at zero dropped rather than
  placed, and one floor per quantity axis, so the gross slab, the tower and
  the curve beside them line up.

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

#: A two-stage program with both towers and both Lee curves, which is every
#: part of the tower panel in one picture: a gross slab, a retention, a
#: placed layer, a co-participation block, boundary ticks in currency and
#: the faint rules carrying each boundary across to the curve.
_STRUCTURE = ('agg CD.Program 5 claims 100 xs 0 sev lognorm 10 cv .75 '
              'occurrence ceded to 15 xs 5 poisson '
              'aggregate net of 20 xs 0')


def _subjects():
    """``name -> ChartDoc``, built fresh so nothing caches across tests."""
    from aggregate import build
    from aggregate.charts import build_chart_doc
    return {
        'distortion': build_chart_doc(build('distortion CD.PH ph 0.7'),
                                      'distortion'),
        'agg': build_chart_doc(build(_AGG), 'agg'),
        'structure': build_chart_doc(build(_STRUCTURE), 'structure',
                                     lee=True),
        'structure_log': build_chart_doc(build(_STRUCTURE), 'structure',
                                         lee=True),
    }


#: Render options per subject, for a baseline that is a *reading* of a
#: document rather than a document of its own.
_OPTIONS = {'structure_log': {'log': 'y'}}


def _render(doc, target, **options):
    from aggregate.plots import plot_chartdoc, plt, use
    use()
    plot_chartdoc(doc, **options).savefig(target, dpi=100)
    plt.close('all')


def _compare(actual, baseline_name):
    from matplotlib.testing.compare import compare_images
    result = compare_images(str(_BASELINES / baseline_name), str(actual),
                            tol=_RMS_TOL)
    assert result is None, result


@pytest.mark.parametrize('name', ['distortion', 'agg', 'structure',
                                  'structure_log'])
def test_chartdoc_matches_baseline(name, tmp_path):
    """The gate: the rendered document is the picture that was approved."""
    target = tmp_path / f'{name}.png'
    _render(_subjects()[name], target, **_OPTIONS.get(name, {}))
    _compare(target, f'{name}.png')


def _regen():
    matplotlib.use('Agg')
    _BASELINES.mkdir(parents=True, exist_ok=True)
    for name, doc in _subjects().items():
        _render(doc, _BASELINES / f'{name}.png', **_OPTIONS.get(name, {}))
    print(f'baselines regenerated under {_BASELINES} '
          f'with matplotlib {matplotlib.__version__}')


if __name__ == '__main__' and '--regen' in sys.argv:
    _regen()
