"""[Chart-Surface-Pilot]: the joint density surface emitter and renderer.

Covers ``charts._emit_bivariate.chart_joint_surface`` (the mass-preserving
display reduction migrated from the app's ``surfaceGrid``, axis labeling
from resolved component labels, determinism) and the generic matplotlib
renderer's capability pattern (2-D projection with a ``(projection)``
stamp; ``ChartCapabilityError`` under ``strict``).
"""

import math

import matplotlib
import numpy as np
import pytest

# The renderer cases draw to memory, never to a screen; without this they
# inherit whatever interactive backend is active and fail intermittently on
# a Tk toolkit error unrelated to the assertion.
matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    ChartCapabilityError, available_charts, canonical_json, doc_hash,
    chart_joint_surface,
)
from aggregate.charts._emit_bivariate import (  # noqa: E402
    DISPLAY_CELLS, _reduce_grid)


@pytest.fixture(scope='module')
def bv():
    return build('''bivariate MV 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        copula gumbel 0.4
        poisson''')


# ------------------------------------------------------------- reduction

def js_surface_grid(density, x0, x1, cells):
    """The app's surfaceGrid loop (surface.js:71-110), transliterated."""
    nx, ny = density.shape
    bx = max(1, math.ceil(nx / cells))
    by = max(1, math.ceil(ny / cells))
    ox = math.ceil(nx / bx)
    oy = math.ceil(ny / by)
    acc = np.zeros((ox, oy))
    for i in range(nx):
        for j in range(ny):
            v = density[i, j]
            if np.isfinite(v):
                acc[i // bx, j // by] += v
    xs = [x0[min((i + 1) * bx - 1, nx - 1)] for i in range(ox)]
    ys = [x1[min((j + 1) * by - 1, ny - 1)] for j in range(oy)]
    return np.array(xs), np.array(ys), acc


def test_reduce_grid_matches_app_algorithm():
    # An awkward shape (prime-ish, short final blocks) plus a NaN cell.
    rng = np.random.default_rng(17)
    density = rng.random((13, 7))
    density[3, 2] = np.nan
    x0 = np.arange(13) * 0.5
    x1 = np.arange(7) * 2.0
    xs, ys, z = _reduce_grid(density, x0, x1, cells=4)
    js_xs, js_ys, js_z = js_surface_grid(density, x0, x1, cells=4)
    np.testing.assert_array_equal(xs, js_xs)
    np.testing.assert_array_equal(ys, js_ys)
    np.testing.assert_allclose(z, js_z, rtol=0, atol=1e-15)


def test_reduce_grid_conserves_mass():
    rng = np.random.default_rng(3)
    density = rng.random((300, 130))
    density /= density.sum()
    _, _, z = _reduce_grid(density, np.arange(300), np.arange(130), 64)
    assert z.shape == (math.ceil(300 / math.ceil(300 / 64)),
                       math.ceil(130 / math.ceil(130 / 64)))
    assert np.isclose(z.sum(), density.sum(), atol=1e-14)


def test_reduce_grid_passthrough_when_small():
    density = np.eye(5)
    xs, ys, z = _reduce_grid(density, np.arange(5), np.arange(5), 128)
    np.testing.assert_array_equal(z, density)
    np.testing.assert_array_equal(xs, np.arange(5))


# --------------------------------------------------------------- emitter

def test_emitter_document_shape(bv):
    doc = chart_joint_surface(bv)
    assert doc.name == 'joint_surface'
    assert [p.kind for p in doc.panels] == ['surface']
    # The log height reading is declared on the z axis itself, which is what
    # retired the meta['z_log_ok'] flag at [Chart-Declared-Readings].
    z_axis, = [a for a in doc.axes if a.id == 'z']
    assert z_axis.scale == 'linear' and z_axis.scales == ('linear', 'log')
    assert 'z_log_ok' not in doc.meta
    surf = doc.series[0].surface
    assert len(surf.x) <= DISPLAY_CELLS
    assert len(surf.y) <= DISPLAY_CELLS
    # mass preservation end to end: display cells sum to the joint's mass
    total = sum(v for row in surf.z for v in row)
    assert np.isclose(total, float(bv.density.sum()), atol=1e-10)
    # right-edge label convention: the last label is the last grid point
    assert surf.x[-1] == float(bv.axis_xs[0][-1])
    assert surf.y[-1] == float(bv.axis_xs[1][-1])
    # axes labeled from the component units (unlabeled: the handles)
    labels = {a.id: a.label for a in doc.axes}
    assert labels['x0'] == 'A' and labels['x1'] == 'B'
    assert labels['z'] == 'density'


def test_emitter_tex_is_total(bv):
    """The joint half of the totality sweep in tests/test_charts_ir.py."""
    from aggregate.charts import human_strings
    doc = chart_joint_surface(bv, display_log2=4)
    assert not set(human_strings(doc)) - set(doc.tex)


def test_emitter_orientation(bv):
    # z[r][c] sits at (x[c], y[r]): row count is len(y), col count len(x).
    doc = chart_joint_surface(bv, display_log2=5)
    surf = doc.series[0].surface
    assert len(surf.x) <= 32 and len(surf.y) <= 32
    assert len(surf.z) == len(surf.y)
    assert len(surf.z[0]) == len(surf.x)


def test_emitter_deterministic(bv):
    a = chart_joint_surface(bv)
    b = chart_joint_surface(bv)
    assert canonical_json(a) == canonical_json(b)
    assert len(doc_hash(a)) == 12


def test_emitter_json_clean(bv):
    # No numpy scalars may leak into the payload: json rejects them.
    payload = canonical_json(chart_joint_surface(bv, display_log2=4))
    assert payload.startswith(b'{')


def test_capability_registry(bv):
    assert available_charts(bv) == ['joint_surface']
    assert available_charts(object()) == []
    with pytest.raises(NotImplementedError, match='joint_surface'):
        chart_joint_surface(3.14)


# -------------------------------------------------------------- renderer

def test_renderer_projection_stamp(bv):
    from aggregate.plots import plot_chartdoc, plt
    doc = chart_joint_surface(bv, display_log2=5)
    fig = plot_chartdoc(doc)
    try:
        ax = fig.axes[0]
        assert ax.get_title().endswith('(projection)')
        assert ax.get_xlabel() == 'A'
        assert ax.get_ylabel() == 'B'
    finally:
        plt.close(fig)


def test_renderer_log_z(bv):
    from aggregate.plots import plot_chartdoc, plt
    doc = chart_joint_surface(bv, display_log2=5)
    fig = plot_chartdoc(doc, log=True)
    plt.close(fig)


def test_renderer_strict_raises(bv):
    from aggregate.plots import plot_chartdoc
    doc = chart_joint_surface(bv, display_log2=4)
    with pytest.raises(ChartCapabilityError, match='projection'):
        plot_chartdoc(doc, strict=True)


def test_renderer_lays_out_one_axes_per_panel():
    # Every panel kind has a realization (xy landed with the distortion
    # conversion) and multi-panel layout landed with the reins triple, so
    # nothing is refused here any more: the capability edges that remain
    # are the surface projection (above) and a kind with no realization.
    from aggregate.charts import ChartAxis, ChartDoc, ChartSeries, Panel
    from aggregate.plots import plot_chartdoc, plt
    axes = (ChartAxis(id='a', label='a'), ChartAxis(id='b', label='b'))
    doc = ChartDoc(
        name='t',
        axes=axes,
        panels=(Panel(id='p', kind='xy', x_axis='a', y_axis='b'),
                Panel(id='q', kind='xy', x_axis='a', y_axis='b')),
        series=(ChartSeries(name='s', role='density', panel_id='p',
                            x=(0.0, 1.0), y=(1.0, 0.0)),))
    fig = plot_chartdoc(doc)
    assert len(fig.axes) == 2
    plt.close('all')
