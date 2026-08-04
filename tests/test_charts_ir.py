"""Chart IR schema v1: structure, validation, determinism, boundary.

Covers ``aggregate.charts.ir`` (the frozen dataclasses and the canonical
serialization contract) and the ``aggregate.charts`` capability surface.
The determinism tests mirror greater_tables' contract: same document, same
bytes, same hash, on any machine, any run.
"""

import dataclasses
import json
import subprocess
import sys

import pytest

from aggregate import charts
from aggregate.charts import (
    CHART_IR_VERSION, ChartAxis, ChartDoc, ChartSeries, Mark, Panel,
    SurfaceData, available_charts, canonical_dict, canonical_json, doc_hash,
    stamp,
)


def small_xy_doc(**overrides):
    """A minimal two-panel xy document (the two-panel exhibit shape)."""
    kw = dict(
        name='agg_overview',
        title='Dice (Dice)',
        axes=(
            ChartAxis(id='loss', label='loss', unit='currency',
                      suggested_range=(0.0, 20.0)),
            ChartAxis(id='dens', label='density', unit='density'),
            ChartAxis(id='surv', label='S(x)', scale='log',
                      unit='probability'),
            ChartAxis(id='rp', label='return period', scale='log',
                      unit='return_period', reciprocal_of='surv'),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='loss', y_axis='dens',
                  title='Density'),
            Panel(id='tail', kind='xy', x_axis='loss', y_axis='surv',
                  read_axis='y', title='Survival'),
        ),
        series=(
            ChartSeries(name='total', role='density', panel_id='density',
                        x=(1.0, 2.0, 3.0), y=(0.25, 0.5, 0.25)),
            ChartSeries(name='total', role='survival', panel_id='tail',
                        x=(1.0, 2.0, 3.0), y=(0.75, 0.25, None)),
        ),
        marks=(
            Mark(panel_id='density', orient='v', at=2.0, label='mean',
                 role='mean'),
            Mark(panel_id='tail', orient='v', at=3.0, label='1-in-250',
                 role='capital_anchor', faint=True),
        ),
        meta={'kind': 'agg'},
    )
    kw.update(overrides)
    return ChartDoc(**kw)


def small_surface_doc():
    """A minimal surface document (the pilot shape)."""
    surf = SurfaceData(x=(1.0, 2.0), y=(10.0, 20.0, 30.0),
                       z=((0.1, 0.2), (0.3, 0.1), (0.2, 0.1)))
    return ChartDoc(
        name='joint_surface',
        axes=(
            ChartAxis(id='x0', label='Wind', unit='currency'),
            ChartAxis(id='x1', label='Flood', unit='currency'),
            ChartAxis(id='z', label='density', unit='density'),
        ),
        panels=(
            Panel(id='joint', kind='surface', x_axis='x0', y_axis='x1',
                  z_axis='z'),
        ),
        series=(
            ChartSeries(name='joint density', role='joint',
                        panel_id='joint', surface=surf),
        ),
        meta={'z_log_ok': True},
    )


# ---------------------------------------------------------------- structure

def test_frozen():
    doc = small_xy_doc()
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.name = 'other'
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.axes[0].label = 'other'


def test_sequences_coerced_to_tuples():
    doc = ChartDoc(name='t', axes=[ChartAxis(id='a', label='a'),
                                   ChartAxis(id='b', label='b')],
                   panels=[Panel(id='p', kind='xy', x_axis='a', y_axis='b')],
                   series=[ChartSeries(name='s', role='density',
                                       panel_id='p', x=[0, 1], y=[1, 0])])
    assert isinstance(doc.axes, tuple)
    assert isinstance(doc.series[0].x, tuple)


def test_bad_ir_version():
    with pytest.raises(ValueError, match='ir_version'):
        small_xy_doc(ir_version=99)


def test_unknown_panel_kind():
    with pytest.raises(ValueError, match='panel kind'):
        Panel(id='p', kind='pie', x_axis='a', y_axis='b')


def test_unknown_axis_scale():
    with pytest.raises(ValueError, match='scale'):
        ChartAxis(id='a', label='a', scale='sqrt')


def test_surface_panel_requires_z_axis():
    with pytest.raises(ValueError, match='z_axis'):
        Panel(id='p', kind='surface', x_axis='a', y_axis='b')


def test_series_reference_integrity():
    with pytest.raises(ValueError, match='unknown panel'):
        small_xy_doc(series=(
            ChartSeries(name='s', role='density', panel_id='nope',
                        x=(0.0,), y=(1.0,)),))


def test_axis_reference_integrity():
    with pytest.raises(ValueError, match='unknown axis'):
        ChartDoc(name='t',
                 axes=(ChartAxis(id='a', label='a'),),
                 panels=(Panel(id='p', kind='xy', x_axis='a', y_axis='b'),))


def test_duplicate_axis_ids():
    with pytest.raises(ValueError, match='duplicate axis ids'):
        ChartDoc(name='t', axes=(ChartAxis(id='a', label='a'),
                                 ChartAxis(id='a', label='b')))


def test_series_data_must_match_panel_kind():
    surf = SurfaceData(x=(1.0,), y=(1.0,), z=((1.0,),))
    with pytest.raises(ValueError, match='does not match panel kind'):
        small_xy_doc(series=(
            ChartSeries(name='s', role='joint', panel_id='density',
                        surface=surf),))


def test_xy_lengths_must_agree():
    with pytest.raises(ValueError, match='len'):
        ChartSeries(name='s', role='density', panel_id='p',
                    x=(0.0, 1.0), y=(1.0,))


def test_series_needs_exactly_one_payload():
    with pytest.raises(ValueError, match='either x/y or surface'):
        ChartSeries(name='s', role='density', panel_id='p')


def test_surface_data_shape_checked():
    with pytest.raises(ValueError, match='row'):
        SurfaceData(x=(1.0, 2.0), y=(1.0,), z=((1.0,),))
    with pytest.raises(ValueError, match='rows'):
        SurfaceData(x=(1.0,), y=(1.0, 2.0), z=((1.0,),))


def test_mark_orient():
    with pytest.raises(ValueError, match='orient'):
        Mark(panel_id='p', orient='d', at=0.0)


def test_panel_aspect():
    p = Panel(id='p', kind='xy', x_axis='a', y_axis='b', aspect='equal')
    assert p.aspect == 'equal'
    with pytest.raises(ValueError, match='aspect'):
        Panel(id='p', kind='xy', x_axis='a', y_axis='b', aspect='square')


def test_band_series_y2():
    s = ChartSeries(name='margin', role='band', panel_id='p',
                    x=(0.0, 1.0), y=(0.0, 0.5), y2=(0.1, 0.9))
    assert s.y2 == (0.1, 0.9)
    with pytest.raises(ValueError, match='y2'):
        ChartSeries(name='m', role='band', panel_id='p',
                    x=(0.0, 1.0), y=(0.0, 0.5), y2=(0.1,))


# -------------------------------------------------------------- determinism

def test_canonical_json_roundtrip():
    doc = small_xy_doc()
    data = json.loads(canonical_json(doc))
    assert data == canonical_dict(doc)
    assert data['ir_version'] == CHART_IR_VERSION
    assert data['name'] == 'agg_overview'


def test_same_content_same_bytes():
    assert canonical_json(small_xy_doc()) == canonical_json(small_xy_doc())


def test_defaults_omitted():
    # An explicit default and an omitted field canonicalize identically, so
    # adding optional fields later never changes existing hashes.
    a = ChartAxis(id='a', label='a')
    b = ChartAxis(id='a', label='a', scale='linear', kind='value')
    doc_a = ChartDoc(name='t', axes=(a,))
    doc_b = ChartDoc(name='t', axes=(b,))
    assert canonical_json(doc_a) == canonical_json(doc_b)
    assert 'scale' not in canonical_dict(doc_a)['axes'][0]


def test_doc_hash_form():
    h = doc_hash(small_xy_doc())
    assert len(h) == 12
    assert all(c in '0123456789abcdef' for c in h)


def test_stamp_does_not_perturb_hash():
    doc = small_surface_doc()
    stamped = stamp(doc, generator='aggregate test')
    assert stamped.hash == doc_hash(doc)
    assert stamped.generator == 'aggregate test'
    assert doc_hash(stamped) == doc_hash(doc)
    # but the stamped fields do appear in the payload form
    assert json.loads(canonical_json(stamped))['hash'] == stamped.hash


def test_nfc_normalization():
    # 'e' + combining acute vs precomposed U+00E9 must canonicalize alike.
    a = small_xy_doc(title='café')
    b = small_xy_doc(title='café')
    assert canonical_json(a) == canonical_json(b)


def test_nan_rejected():
    doc = small_xy_doc(series=(
        ChartSeries(name='s', role='density', panel_id='density',
                    x=(0.0,), y=(float('nan'),)),))
    with pytest.raises(ValueError):
        canonical_json(doc)


def test_gaps_serialize_as_null():
    doc = small_xy_doc()
    tail = [s for s in canonical_dict(doc)['series']
            if s['role'] == 'survival']
    assert tail[0]['y'][-1] is None


# ---------------------------------------------------------------- capability

def test_available_charts_unknown_type_empty():
    assert available_charts(object()) == []


def test_register_chart_rejects_duplicates():
    emitter = charts._emitter_base('doomed')
    name = 'test_dupe_chart'
    charts.register_chart(name, emitter)
    try:
        with pytest.raises(ValueError, match='already registered'):
            charts.register_chart(name, emitter)
    finally:
        del charts.CHARTS[name]


# ------------------------------------------------------------------ boundary

def test_import_charts_does_not_load_matplotlib():
    """``import aggregate.charts`` must leave matplotlib unloaded."""
    code = (
        "import sys, aggregate.charts; "
        "assert 'matplotlib' not in sys.modules, "
        "'aggregate.charts pulled in matplotlib'"
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
