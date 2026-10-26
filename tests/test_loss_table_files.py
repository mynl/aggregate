"""Loss table file sources ([DecL-Loss-Tables], ``[Loss-Table-Files]``).

``elt "path"`` and ``ylt "path"`` read CSV, TSV, Parquet or Feather through one
alias table. Identifiers stay exact, a ``unit`` column splits a long file, and a
relative path is fixed when the statement is registered: against the ``.agg``
file's directory, or the session's working directory. Fixtures are synthetic,
under ``tests/data/elt`` and ``tests/data/ylt``.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggregate import Aggregate, build
from aggregate._loss_table import LossTableError
from aggregate.underwriter import Underwriter

DATA = Path(__file__).parent / 'data'


def _q(path):
    """A path as a DecL string."""
    return f'"{Path(path).as_posix()}"'


def _law(program):
    a = build(program, update=False)
    return a.spec['sev_xs'], a.spec['sev_ps'], a.n, a.actual_m, a.actual_var


def test_csv_matches_inline():
    f = _law(f'agg LT.FileBasic elt {_q(DATA / "elt/basic.csv")}')
    i = _law('agg LT.InlineBasic elt [[1 .1 100] [2 .1 40] [3 .1 80]]')
    for u, v in zip(f, i):
        np.testing.assert_allclose(u, v)


@pytest.mark.parametrize('suffix', ['parquet', 'feather'])
def test_columnar_matches_csv(tmp_path, suffix):
    df = pd.read_csv(DATA / 'elt/basic.csv')
    out = tmp_path / f'basic.{suffix}'
    getattr(df, f'to_{suffix}')(out)
    a = _law(f'agg LT.Col elt {_q(out)}')
    b = _law(f'agg LT.Csv elt {_q(DATA / "elt/basic.csv")}')
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v)


def test_aliases_units_and_exact_ids():
    """``Event ID``, ``Annual Rate``, ``Mean Loss``, ``Exposure`` and ``Unit``
    resolve, ``Notes`` is ignored, ``007`` stays text, event 1 is shared."""
    a = build(f'agg LT.Long elt {_q(DATA / "elt/units-long.csv")}', update=False)
    meta = a._loss_table_meta
    assert meta['units'] == ['Personal', 'Commercial']
    assert meta['form'] == 'ELT4'
    assert meta['events'] == 3
    assert meta['max_event_exposure'] == 4000
    assert meta['sources'][0]['ignored_columns'] == ['Notes']
    assert a.actual_m == pytest.approx(0.1 * (500 + 100 + 200))


def test_clause_unit_must_agree_with_the_column():
    with pytest.raises(LossTableError, match='also holds'):
        build(f'agg LT.U elt unit Personal {_q(DATA / "elt/units-long.csv")}',
              update=False)


def test_no2u_file():
    a = build(f'agg LT.Six elt no2u {_q(DATA / "elt/six-columns.csv")}',
              update=False)
    assert a.actual_m == pytest.approx(0.1 * 50 + 0.2 * 20)
    assert a._loss_table_meta['sources'][0]['form'] == 'ELT6'
    with pytest.raises(NotImplementedError, match='secondary uncertainty'):
        build(f'agg LT.Six2 elt {_q(DATA / "elt/six-columns.csv")}',
              update=False)


@pytest.mark.parametrize('name, form, mean, var', [
    ('years.csv', 'YLT2', 100, 15000),
    ('losses-only.csv', 'YLT1', 100, 15000),
    ('weighted.tsv', 'YLT3', 90, 12900),
])
def test_ylt_files(name, form, mean, var):
    a = build(f'agg LT.YF ylt {_q(DATA / "ylt" / name)}', update=False)
    assert a._loss_table_meta['form'] == form
    assert a.actual_m == pytest.approx(mean)
    assert a.actual_var == pytest.approx(var)


def test_long_ylt_aligns_units_by_year():
    x, p, *_ = _law(f'agg LT.YL ylt {_q(DATA / "ylt/units-long.csv")}')
    np.testing.assert_allclose(x, [100])


def test_digest_and_reread_on_rebuild(tmp_path):
    f = tmp_path / 'table.csv'
    f.write_text('id,rate,mean\n1,0.1,100\n', encoding='utf-8')
    program = f'agg LT.Digest elt {_q(f)}'
    a = build(program, update=False)
    src = a._loss_table_meta['sources'][0]
    assert src['sha256'] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert src['resolved'] == f.as_posix()
    f.write_text('id,rate,mean\n1,0.1,200\n', encoding='utf-8')
    b = build(program, update=False)
    assert b.actual_m == pytest.approx(20)
    assert b._loss_table_meta['sources'][0]['sha256'] != src['sha256']
    # an existing object keeps its law, and reconstruction reads nothing
    f.unlink()
    assert Aggregate(**a.spec).actual_m == pytest.approx(10)


def test_path_punctuation(tmp_path):
    d = tmp_path / 'dir #1 [a] b'
    d.mkdir()
    (d / 'x.csv').write_text('id,rate,mean\n1,0.1,100\n', encoding='utf-8')
    a = build(f'agg LT.Punct elt "{d.as_posix()}//x.csv"', update=False)
    assert a.actual_m == pytest.approx(10)


def test_session_path_fixed_at_registration(tmp_path, monkeypatch):
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / 't.csv').write_text('id,rate,mean\n1,0.1,100\n', encoding='utf-8')
    (b_dir / 't.csv').write_text('id,rate,mean\n1,0.1,999\n', encoding='utf-8')
    uw = Underwriter(update=False)
    monkeypatch.chdir(a_dir)
    uw('agg LT.Rel elt "t.csv"')
    monkeypatch.chdir(b_dir)
    ref = uw('agg LT.RelRef agg.LT.Rel aggregate net of 1 xs 0')
    assert ref.actual_m == pytest.approx(10)


def test_agg_file_path_is_relative_to_the_file(tmp_path, monkeypatch):
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 't.csv').write_text('id,rate,mean\n1,0.1,100\n', encoding='utf-8')
    (lib / 'book.agg').write_text('agg LT.InFile elt "t.csv"\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    uw = Underwriter(update=False, databases=[])
    uw.load(lib / 'book.agg')
    a = uw('agg LT.FromLib agg.LT.InFile aggregate net of 1 xs 0')
    assert a.actual_m == pytest.approx(10)


def test_export_writes_absolute_paths(tmp_path, monkeypatch):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 't.csv').write_text('id,rate,mean\n1,0.1,100\n', encoding='utf-8')
    monkeypatch.chdir(src)
    uw = Underwriter(update=False, databases=[])
    uw('agg LT.Exported elt "t.csv"')
    (tmp_path / 'out').mkdir()
    out = uw.to_agg(tmp_path / 'out' / 'export.agg', pattern='LT.Exported')
    text = out.read_text(encoding='utf-8')
    assert (src / 't.csv').resolve().as_posix() in text
    monkeypatch.chdir(tmp_path)
    fresh = Underwriter(update=False, databases=[])
    fresh.load(out)
    assert fresh('agg LT.Back agg.LT.Exported aggregate net of 1 xs 0').actual_m \
        == pytest.approx(10)


def test_float_ids_refused(tmp_path):
    out = tmp_path / 'f.parquet'
    pd.DataFrame({'id': [1.0, 2.0], 'rate': [.1, .1], 'mean': [1, 2]}).to_parquet(out)
    with pytest.raises(LossTableError, match='floating point'):
        build(f'agg LT.Float elt {_q(out)}', update=False)


def test_integer_ids_beyond_float_precision(tmp_path):
    out = tmp_path / 'big.parquet'
    pd.DataFrame({'id': np.array([2**53, 2**53 + 1], dtype=np.int64),
                  'unit': ['A', 'B'], 'rate': [.1, .1],
                  'mean': [100., 40.]}).to_parquet(out)
    a = build(f'agg LT.Big elt {_q(out)}', update=False)
    assert a._loss_table_meta['events'] == 2


@pytest.mark.parametrize('text, match', [
    ('id,rate\n1,0.1\n', 'no mean column'),
    ('id,event,rate,mean\n1,1,0.1,5\n', 'both mean event_id'),
    ('id,rate,mean,sdi\n1,0.1,5,1\n', 'max, sdi and sdc'),
    ('id,rate,mean\n,0.1,5\n', 'empty identifier'),
    ('id,rate,mean\n1,0.1,abc\n', 'mean must be finite'),
])
def test_malformed_elt_files(tmp_path, text, match):
    f = tmp_path / 'bad.csv'
    f.write_text(text, encoding='utf-8')
    with pytest.raises(LossTableError, match=match):
        build(f'agg LT.BadFile elt {_q(f)}', update=False)


@pytest.mark.parametrize('text, match', [
    ('loss,prob\n0,.5\n1,.5\n', 'needs a year identifier'),
    ('unit,loss\nA,0\nB,1\n', 'several units needs a year'),
])
def test_malformed_ylt_files(tmp_path, text, match):
    f = tmp_path / 'bad.csv'
    f.write_text(text, encoding='utf-8')
    with pytest.raises(LossTableError, match=match):
        build(f'agg LT.BadYear ylt {_q(f)}', update=False)


def test_unsupported_and_missing_files(tmp_path):
    with pytest.raises(LossTableError, match='unsupported file type'):
        build(f'agg LT.X elt {_q(tmp_path / "t.xlsx")}', update=False)
    with pytest.raises(LossTableError, match='cannot read'):
        build(f'agg LT.X elt {_q(tmp_path / "missing.csv")}', update=False)
