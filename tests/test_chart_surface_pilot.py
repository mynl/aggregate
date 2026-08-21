"""[Chart-Surface-Pilot]: the joint density surface emitter and renderer.

Covers ``charts._emit_bivariate.chart_joint_surface`` (the window chosen on
the fine lattice, the mass-preserving reduction, the representative-point
coordinate convention, the lattice and moment fields, the encodings,
determinism) and
the generic matplotlib renderer's capability pattern (2-D projection with a
``(projection)`` stamp; ``ChartCapabilityError`` under ``strict``).

Two of the fixtures below are the plan's reference surfaces and are worth
their build cost. ``indep`` is the pathological one: 64 fine cells on x
against 16,384 on y, a Lomax tail that spends most of that axis empty, and a
y bucket 128 times the x bucket, which is what turned a display convention
into a wrong number. ``signed`` has mass either side of zero, which is the
only thing that can fail a snap-to-zero rule.
"""

import numpy as np
import matplotlib
import pytest

# The renderer cases draw to memory, never to a screen; without this they
# inherit whatever interactive backend is active and fail intermittently on
# a Tk toolkit error unrelated to the assertion.
matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    ChartCapabilityError, available_charts, build_chart_doc, canonical_dict,
    canonical_json, chart_joint_surface, decode_z_block, doc_hash,
    encode_z_block, load_chart_doc,
)
from aggregate.charts._emit_bivariate import (  # noqa: E402
    DEFAULT_DETAIL, MIN_CELLS, _axis_plan, _block_factor, _reduce)


@pytest.fixture(scope='module')
def bv():
    return build('''bivariate MV 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        copula gumbel 0.4
        poisson''')


@pytest.fixture(scope='module')
def indep():
    """The plan's ``Indep``: the only surface with independent marginals.

    It is the one with ``dfreq[1]``, one claim, certain, so the two units
    share nothing. The three programs that open ``10 claims ... poisson``
    share the claim count, which makes their units conditionally independent
    given N and dependent without it, whatever the copula says. Any test that
    asserts independence must use this one.
    """
    return build('''bivariate Indep
        dfreq[1]
        agg Tame
          dfreq [1] sev gamma 100 cv .1
        agg Severe
          dfreq [1] sev 50 * lomax 3.1''')


@pytest.fixture(scope='module')
def signed():
    """The plan's ``IndepSigned``: support on both sides of zero."""
    return build('''bivariate IndepSigned
        10 claims
        agg Tame
          dfreq [1] ssev 100 * norm
        agg Severe
          dfreq [1] ssev 150 * norm - 50
        poisson''')


def surface_of(doc):
    return doc.series[0].surface


# ------------------------------------------------------------- reduction

def test_block_factor_is_a_power_of_two_ceiling():
    assert _block_factor(128, 128) == 1
    assert _block_factor(129, 128) == 2
    assert _block_factor(232, 128) == 2      # the plan's Indep y window
    assert _block_factor(2048, 128) == 16
    # a ceiling, never straddled upward: the realized count never exceeds
    # the target, which is what makes the API's cap mean something
    for span in (7, 31, 100, 231, 1000, 4095):
        for detail in (8, 16, 128, 512):
            k = _block_factor(span, detail)
            assert -(-span // k) <= detail
            assert k == 1 or -(-span // (k >> 1)) > detail


def test_reduce_conserves_mass():
    rng = np.random.default_rng(3)
    density = rng.random((256, 128))
    density /= density.sum()
    z = _reduce(_reduce(density, 4, axis=0), 8, axis=1)
    assert z.shape == (64, 16)
    assert np.isclose(z.sum(), density.sum(), atol=1e-14)


def test_reduce_passthrough_when_the_factor_is_one():
    density = np.eye(5)
    np.testing.assert_array_equal(_reduce(density, 1), density)


# ------------------------------------------- 5.1, the block coordinates

@pytest.mark.parametrize('detail', [128, 32, 8])
def test_display_grid_starts_where_the_fine_lattice_does(indep, detail):
    """The whole-grid case: the first block is filed at its own center of mass.

    The first display coordinate sits ``(k - 1) * bs / 2`` above the first
    fine one, which is the mean of the atoms that block covers, so backing
    that offset out recovers the fine lattice exactly. The first convention
    this replaced filed a block covering ``[a, a + k * bs)`` under its *last*
    fine coordinate, so ``Indep``'s y axis, which reduces 128 fine cells into
    one 512-wide bucket, reported a distribution supported from 0 as starting
    at 508; the second filed it under ``a``.
    """
    s = surface_of(chart_joint_surface(indep, window=0, detail=detail))
    assert s.edge == 'mid'
    fine_y = float(indep.axis_xs[1][0])
    assert fine_y == 0.0
    assert s.x0 - (s.k[0] - 1) * s.bs[0] / 2 == float(indep.axis_xs[0][0])
    assert s.y0 - (s.k[1] - 1) * s.bs[1] / 2 == fine_y
    assert s.x[0] == s.x0 and s.y[0] == s.y0
    # and the step really is the declared one, all the way along
    np.testing.assert_allclose(np.diff(s.x), s.dx, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.diff(s.y), s.dy, rtol=0, atol=1e-9)
    assert s.dx == s.bs[0] * s.k[0] and s.dy == s.bs[1] * s.k[1]


def test_display_mean_bias_is_bounded_by_half_a_bucket(indep):
    """What survives the representative point is second order and two-sided.

    A display cell holds ``k`` atoms; labeling it with any single coordinate
    loses their spread. Under the representative point that coordinate is the
    atoms' own mean, so the residual is the deviation of the within-block mass
    from uniform rather than a convention, it takes **either** sign, and half
    a display bucket bounds it whatever the density does inside the block: a
    block's conditional mean lies in ``[a, a + (k - 1) * bs]`` and the
    coordinate is the middle of that span. Neither replaced convention has
    that bound. Both are one-sided and reach a whole bucket, high when the
    coordinate was the block's last fine atom, which also moved the support,
    low when it was the first.

    The whole-grid case is the bound; the second half is the case that
    separates the three conventions, ``Indep``'s y axis at the default window
    where a block is two atoms. What a consumer should read instead of any of
    them is ``moments``, which is exact.
    """
    s = surface_of(chart_joint_surface(indep, window=0))
    z = np.asarray(s.z)
    for axis, coords, step in (('x', np.asarray(s.x), s.dx),
                               ('y', np.asarray(s.y), s.dy)):
        mass = z.sum(0) if axis == 'x' else z.sum(1)
        grid_mean = float((mass * coords).sum())
        fine_mean = s.moments['mean'][0 if axis == 'x' else 1]
        assert abs(grid_mean - fine_mean) < step / 2

    # and the discriminating measurement: on a two-atom block the
    # representative point is an order of magnitude closer than either end of
    # the span, where the residual left is the window's own truncation
    s = surface_of(chart_joint_surface(indep, window=4))
    z = np.asarray(s.z)
    mass = z.sum(1)
    coords = np.asarray(s.y)
    offset = (s.k[1] - 1) * s.bs[1] / 2
    fine_mean = s.moments['mean'][1]
    bias = {
        'representative': float((mass * coords).sum()) - fine_mean,
        'low edge': float((mass * (coords - offset)).sum()) - fine_mean,
        'cell midpoint': float(
            (mass * (coords - offset + s.dy / 2)).sum()) - fine_mean,
    }
    assert s.k[1] == 2 and s.dy == 8.0
    assert abs(bias['representative']) < abs(bias['low edge']) / 10
    assert abs(bias['representative']) < abs(bias['cell midpoint']) / 10


# ------------------------------------------------ 5.2, window then reduce

def test_window_is_taken_before_the_reduction(indep):
    """The measurement that settles the order of the two steps.

    On ``Indep``'s y axis the same ``q(1e-4)`` window is 232 fine cells wide
    taken first, and a handful of 512-wide display cells taken last. Reducing
    232 fine cells to a 128 target leaves 116, which is more than an order of
    magnitude more resolution than cropping the emitted grid could reach, and
    it starts at 0 rather than at 508.
    """
    s = surface_of(chart_joint_surface(indep, window=4, detail=128))
    assert s.ny == 116
    assert s.y0 == (s.k[1] - 1) * s.bs[1] / 2      # the first block, from 0
    assert s.dy == 8.0        # two fine buckets of 4, not 128 of them
    # what the old route could have reached: the whole axis reduced to 128
    # cells first, then cropped to the same data window
    whole = surface_of(chart_joint_surface(indep, window=0, detail=128))
    inside = [v for v in whole.y if v <= s.window['y'][1]]
    assert len(inside) < 10


def test_window_reports_the_mass_it_kept(indep):
    s = surface_of(chart_joint_surface(indep, window=4))
    z = np.asarray(s.z)
    assert s.window['p'] == 4.0
    assert abs(s.window['kept'] - z.sum() / indep.density.sum()) < 1e-9
    assert 0.9996 < s.window['kept'] < 1.0
    # the box the document reports is the outer edge of the outer cells, half
    # a step outside the outer coordinates, which is what 'mid' makes it
    assert s.window['x'] == (s.x0 - s.dx / 2,
                             s.x0 + (s.nx - 1) * s.dx + s.dx / 2)
    assert s.window['y'] == (s.y0 - s.dy / 2,
                             s.y0 + (s.ny - 1) * s.dy + s.dy / 2)


def test_deeper_windows_keep_more_mass(indep):
    kept = [surface_of(chart_joint_surface(indep, window=w)).window['kept']
            for w in (2, 3, 4, 5, 6)]
    assert kept == sorted(kept)
    assert kept[-1] > kept[0]


def test_window_zero_is_the_whole_grid(indep):
    s = surface_of(chart_joint_surface(indep, window=0))
    assert s.x0 - (s.k[0] - 1) * s.bs[0] / 2 == float(indep.axis_xs[0][0])
    assert s.y0 - (s.k[1] - 1) * s.bs[1] / 2 == float(indep.axis_xs[1][0])
    assert s.nx * s.k[0] == len(indep.axis_xs[0])
    assert s.ny * s.k[1] == len(indep.axis_xs[1])
    assert abs(s.window['kept'] - 1.0) < 1e-12


def test_low_edge_snaps_to_zero_on_positive_support(indep):
    """``Indep``'s y is a Lomax from the origin: the window must open there.

    The snap puts the first *fine* cell at the origin; the coordinate then
    names that block's representative point, half a fine span above it.
    """
    s = surface_of(chart_joint_surface(indep, window=4))
    assert s.y0 == (s.k[1] - 1) * s.bs[1] / 2
    # x is a gamma whose fine lattice was measured up from 48, so there is no
    # zero on it to reach and nothing to snap to
    assert s.x0 > 0.0 and float(indep.axis_xs[0][0]) > 0.0


def test_low_edge_does_not_snap_on_signed_support(signed):
    """A genuinely negative lower bound is not a window artifact."""
    s = surface_of(chart_joint_surface(signed, window=4))
    assert s.x0 < 0.0 and s.y0 < 0.0
    assert s.window['x'][0] < 0.0 and s.window['y'][0] < 0.0


@pytest.mark.parametrize('depth', list(range(1, 13)))
def test_no_axis_falls_under_the_cell_floor(indep, signed, depth):
    """A Lomax at depth 12 puts both window edges in one bucket; the floor
    is what keeps a grid there that a consumer can still interpolate on."""
    for bv_ in (indep, signed):
        s = surface_of(chart_joint_surface(bv_, window=depth))
        assert s.nx >= MIN_CELLS and s.ny >= MIN_CELLS
        assert s.nx == len(s.x) and s.ny == len(s.y)


def test_detail_is_honored_as_a_ceiling(indep, signed):
    """Every ordinary target exactly; the floor outranks it near the floor.

    Powers of two do not reach every count, so a target within a factor of
    two of :data:`MIN_CELLS` can have nothing to land on: a 528-cell window
    reaches 9 cells or 5, and asked for 8 it takes 9, because 5 is a grid
    with no spacing to interpolate on. The overshoot is bounded absolutely,
    so it never reaches a payload budget.
    """
    for detail in (16, 64, 128, 512):
        for bv_ in (indep, signed):
            s = surface_of(chart_joint_surface(bv_, window=4, detail=detail))
            assert s.nx <= detail and s.ny <= detail
    for bv_ in (indep, signed):
        s = surface_of(chart_joint_surface(bv_, window=4, detail=MIN_CELLS))
        assert s.nx < 2 * MIN_CELLS and s.ny < 2 * MIN_CELLS


def test_the_crop_is_a_whole_number_of_blocks(indep, signed):
    """An uneven last block breaks the uniform step every interpolation
    downstream divides by, and puts the wide cell where the tail is."""
    for bv_ in (indep, signed):
        for depth in (0, 2, 4, 6):
            for i, xs in enumerate(bv_.axis_xs):
                marg = np.asarray(bv_.marginals[i], dtype=float)
                lo, hi, k = _axis_plan(np.asarray(xs, dtype=float), marg,
                                       float(bv_.bs[i]), depth, DEFAULT_DETAIL)
                assert (hi - lo) % k == 0
                assert 0 <= lo < hi <= len(xs)


def test_detail_below_the_floor_is_refused(indep):
    with pytest.raises(ValueError, match='detail must be at least'):
        chart_joint_surface(indep, detail=4)


# ---------------------------------------- 5.5, the marginals and moments

def test_marginals_are_the_objects_own_not_the_boxs(indep):
    """The exact marginal, cropped on its own axis alone.

    Integrating the windowed joint would give the marginal of a *truncated*
    distribution, short by whatever the other axis' crop threw away. The gap
    is small here and the point is that it is not zero: it is the same error
    that bends kappa by several percent when a consumer works inside the box
    instead of on the whole grid.
    """
    s = surface_of(chart_joint_surface(indep, window=4))
    z = np.asarray(s.z)
    mx = np.asarray(s.marginals['x'])
    my = np.asarray(s.marginals['y'])
    assert len(mx) == s.nx and len(my) == s.ny
    # each is short of one only by its own axis' crop
    assert mx.sum() > s.window['kept'] and my.sum() > s.window['kept']
    assert mx.sum() < 1.0 and my.sum() < 1.0
    # and each dominates the row sum of the boxed joint, cell by cell
    assert np.all(mx >= z.sum(0) - 1e-15)
    assert np.all(my >= z.sum(1) - 1e-15)


def test_moments_come_off_the_fine_lattice(indep):
    s = surface_of(chart_joint_surface(indep, window=4))
    for i in range(2):
        xs = np.asarray(indep.axis_xs[i], dtype=float)
        exact = float(xs @ np.asarray(indep.marginals[i], dtype=float))
        assert abs(s.moments['mean'][i] - exact) < 1e-12
    # unmoved by the window, which is the property that makes it a reference
    deep = surface_of(chart_joint_surface(indep, window=0))
    assert deep.moments == s.moments


def test_deficit_travels(indep):
    s = surface_of(chart_joint_surface(indep, window=4))
    assert s.deficit == float(indep.deficit)


# --------------------------------------------- 5.6, the encoded z block

@pytest.mark.parametrize('dtype,tolerance', [('f32b64', 1.2e-7),
                                             ('f64b64', 0.0),
                                             ('u16log12b64', 2.2e-4)])
def test_z_block_round_trips_to_its_declared_error(indep, dtype, tolerance):
    s = surface_of(chart_joint_surface(indep, window=4, encoding=dtype))
    z = np.asarray(s.z)
    back = decode_z_block(s.z_block).reshape(s.ny, s.nx)
    assert s.z_block.dtype == dtype and s.z_block.order == 'yx'
    live = z > z.max() * 1e-12
    assert np.max(np.abs(back[live] - z[live]) / z[live]) <= tolerance
    # an exact zero decodes exactly, under every dtype: a real joint is 14%
    # to 59% exact zeros and a spurious floor under all of them is mass the
    # distribution does not have
    assert np.all(back[z == 0.0] == 0.0)


def test_u16_carries_its_scale_and_reserves_the_zero_code(indep):
    s = surface_of(chart_joint_surface(indep, window=4,
                                       encoding='u16log12b64'))
    assert s.z_block.decades == 12.0
    assert s.z_block.peak == pytest.approx(np.asarray(s.z).max(), rel=1e-12)


def test_json_encoding_emits_no_block(indep):
    s = surface_of(chart_joint_surface(indep, window=4, encoding='json'))
    assert s.z_block is None
    assert len(s.z) == s.ny and len(s.z[0]) == s.nx


def test_unknown_encoding_is_refused(indep):
    with pytest.raises(ValueError, match='unknown encoding'):
        chart_joint_surface(indep, encoding='f16b64')


def test_encoder_is_endian_independent():
    """Little-endian is written explicitly, so the bytes do not depend on the
    machine that wrote them, which is what keeps the hash portable."""
    block = encode_z_block(np.array([1.0, 2.0], dtype='>f8'), dtype='f64b64')
    np.testing.assert_array_equal(decode_z_block(block), [1.0, 2.0])
    assert block.data == encode_z_block([1.0, 2.0], dtype='f64b64').data


def test_encoded_and_plain_forms_agree(indep):
    """Phase one emits both; a consumer reading either must see one grid."""
    s = surface_of(chart_joint_surface(indep, window=4))
    back = decode_z_block(s.z_block).reshape(s.ny, s.nx)
    np.testing.assert_allclose(back, np.asarray(s.z), rtol=1.2e-7, atol=0)


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
    assert surf.nx <= DEFAULT_DETAIL and surf.ny <= DEFAULT_DETAIL
    # mass preservation over the window: display cells sum to the kept share
    total = sum(v for row in surf.z for v in row)
    assert np.isclose(total,
                      surf.window['kept'] * float(bv.density.sum()),
                      atol=1e-12)
    # axes labeled from the component units (unlabeled: the handles)
    labels = {a.id: a.label for a in doc.axes}
    assert labels['x0'] == 'A' and labels['x1'] == 'B'
    assert labels['z'] == 'density'


def test_emitter_tex_is_total(bv):
    """The joint half of the totality sweep in tests/test_charts_ir.py."""
    from aggregate.charts import human_strings
    doc = chart_joint_surface(bv, detail=16)
    assert not set(human_strings(doc)) - set(doc.tex)


def test_emitter_orientation(bv):
    # z[r][c] sits at (x[c], y[r]): row count is len(y), col count len(x).
    doc = chart_joint_surface(bv, detail=32)
    surf = doc.series[0].surface
    assert surf.nx <= 32 and surf.ny <= 32
    assert len(surf.z) == len(surf.y) == surf.ny
    assert len(surf.z[0]) == len(surf.x) == surf.nx
    # and the encoded block is flat in the same order
    assert len(decode_z_block(surf.z_block)) == surf.nx * surf.ny


def test_emitter_deterministic(bv):
    a = chart_joint_surface(bv)
    b = chart_joint_surface(bv)
    assert canonical_json(a) == canonical_json(b)
    assert len(doc_hash(a)) == 12


def test_emitter_json_clean(bv):
    # No numpy scalars may leak into the payload: json rejects them.
    payload = canonical_json(chart_joint_surface(bv, detail=16))
    assert payload.startswith(b'{')


def test_document_round_trips_through_the_reader(bv):
    """The contract from [Chart-Doc-Reader], now over a nested z block."""
    doc = build_chart_doc(bv, 'joint_surface', detail=32)
    back = load_chart_doc(canonical_dict(doc))
    assert doc_hash(back) == doc.hash
    assert back == doc
    assert back.series[0].surface.z_block == doc.series[0].surface.z_block


def test_capability_registry(bv):
    assert available_charts(bv) == ['joint_surface']
    assert available_charts(object()) == []
    with pytest.raises(NotImplementedError, match='joint_surface'):
        chart_joint_surface(3.14)


def test_build_chart_doc_passes_the_semantic_options(bv):
    doc = build_chart_doc(bv, 'joint_surface', window=2, detail=16,
                          encoding='u16log12b64')
    s = doc.series[0].surface
    assert s.window['p'] == 2.0
    assert s.nx <= 16 and s.ny <= 16
    assert s.z_block.dtype == 'u16log12b64'


# -------------------------------------------------------------- renderer

def test_renderer_projection_stamp(bv):
    from aggregate.plots import plot_chartdoc, plt
    doc = chart_joint_surface(bv, detail=32)
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
    doc = chart_joint_surface(bv, detail=32)
    fig = plot_chartdoc(doc, log=True)
    plt.close(fig)


def test_renderer_strict_raises(bv):
    from aggregate.plots import plot_chartdoc
    doc = chart_joint_surface(bv, detail=16)
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
