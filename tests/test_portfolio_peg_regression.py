"""PEG Portfolio regression test.

Locks the numerical output of ``calibrate_distortions`` and
``analyze_distortions`` on the two-unit PEG Portfolio against the captured
baseline in ``tests/data/peg_baseline.json``. Every subsequent Portfolio
refactor sub-project must keep these assertions green.

Three tests today; grows to six by the end of Sub-project D when
``pricing_at`` exists:

- ``test_portfolio_moments`` — agg/est moments match baseline (``rtol=1e-10``).
- ``test_calibration_shapes`` — each distortion's ``shape`` matches baseline
  (``rtol=1e-8``) and its calibration residual is within ``1e-5``.
- ``test_pricing`` — every cell of ``analyze_distortions(.995).pricing_df``
  matches the baseline (``rtol=1e-8``).

Notes
-----
The baseline JSON is the contract. If a refactor *intentionally* changes a
number, the right move is to re-capture (``uv run python -m
tests.capture_peg_baseline``) and document the change in the commit. Do not
silently widen tolerances.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.peg import build_peg


BASELINE_PATH = Path(__file__).parent / 'data' / 'peg_baseline.json'
BASELINE = json.loads(BASELINE_PATH.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def peg():
    """Build PEG once, calibrate, and run analyze_distortions at p=0.995.

    Returns ``(port, ad)`` where ``ad`` is the pricing DataFrame extracted
    from the ``AnalyzeDistortionsResult`` (rows MultiIndex
    ``(distortion, stat)``, columns are line names).
    """
    p = BASELINE['meta']['p_calibration']
    coc = BASELINE['meta']['coc_calibration']
    log2 = BASELINE['meta']['log2']
    bs = BASELINE['meta']['bs']
    port = build_peg(update=True, calibrate=True, p=p, coc=coc,
                     log2=log2, bs=bs)
    # ccoc (mass) on the unbounded PEG is refused by the lifted builder
    # (numerics-3 G6); analyze the no-mass quartet, which keeps the
    # pre-change lifted regression lock intact. ccoc pricing coverage
    # lives in the baseline corpus under allocation='linear'.
    no_mass = {k: v for k, v in port.distortions.items()
               if not getattr(v, 'has_mass', False)}
    ad = port.analyze_distortions(p=p, distortions=no_mass).pricing_df
    return port, ad


def test_portfolio_moments(peg):
    """Theoretical and empirical aggregate moments match baseline."""
    port, _ = peg
    for key, expected in BASELINE['portfolio_moments'].items():
        actual = getattr(port, key)
        assert np.isclose(actual, expected, rtol=1e-10), \
            f'{key}: {actual!r} vs baseline {expected!r}'


def test_calibration_shapes(peg):
    """Each calibrated distortion reproduces baseline shape and residual."""
    port, _ = peg
    for name, expected in BASELINE['calibration'].items():
        assert name in port.distortions, f'distortion {name!r} missing from port.distortions'
        d = port.distortions[name]
        assert np.isclose(d.shape, expected['shape'], rtol=1e-8), \
            f'{name}.shape: {d.shape!r} vs baseline {expected["shape"]!r}'
        assert abs(d.error) < 1e-5, \
            f'{name}.error: |{d.error!r}| >= 1e-5'


def test_pricing(peg):
    """Every cell of analyze_distortions(.995).pricing_df matches baseline.

    Indexed as ``ad.loc[(distortion, stat), column]``. Five distortions × 8
    stats × 3 columns = 120 cells; all must match within ``rtol=1e-8``.
    """
    _, ad = peg
    for dname, by_column in BASELINE['pricing'].items():
        if dname == 'ccoc':
            # mass distortion on the unbounded PEG: the lifted builder
            # refuses (numerics-3 G6); rows are no longer produced
            continue
        assert dname in ad.index.get_level_values(0).unique(), \
            f'distortion {dname!r} missing from analyze_distortions output'
        for column, by_stat in by_column.items():
            for stat, expected in by_stat.items():
                actual = ad.loc[(dname, stat), column]
                assert np.isclose(actual, expected, rtol=1e-8), \
                    f'{dname}.{column}.{stat}: {actual!r} vs baseline {expected!r}'


# ---------------------------------------------------------------------------
# price_stand_alone — structure + accounting identities
# ---------------------------------------------------------------------------

def test_price_stand_alone_shape_and_identities(peg):
    """price_stand_alone returns the canonical pentagon readout (stats are the
    columns, one row per entity) and the sum/total rows satisfy the accounting
    identities."""
    port, _ = peg
    p = BASELINE['meta']['p_calibration']
    # first no-mass distortion: ccoc (mass) is refused on the
    # unbounded PEG by Aggregate/Portfolio gatekeeping (numerics-3)
    dname = next(n for n, d in port.distortions.items()
                 if not getattr(d, 'has_mass', False))
    a = port.price_stand_alone(port.distortions[dname], p=p)

    # canonical orientation: 8 pentagon stats are the columns; rows are one per
    # entity (units + total + sum) under a (method, unit) MultiIndex
    assert list(a.index.names) == ['method', 'unit']
    assert list(a.columns) == ['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']

    flat = a.droplevel('method')
    assert len(flat) == len(port.unit_names) + 2
    assert {'sum', 'total'}.issubset(flat.index)
    units = [u for u in flat.index if u not in ('sum', 'total')]
    assert units == port.unit_names

    # amounts add: the sum row is the column-wise total of the stand-alone units
    for stat in ['L', 'M', 'P', 'Q', 'a']:
        assert np.isclose(flat.loc['sum', stat], flat.loc[units, stat].sum(), rtol=1e-10)

    # ratios are re-derived from amounts in EVERY row (not summed)
    for u in flat.index:
        L, M, P, Q, A = (flat.loc[u, s] for s in ['L', 'M', 'P', 'Q', 'a'])
        assert np.isclose(A, P + Q, rtol=1e-10)
        assert np.isclose(flat.loc[u, 'LR'], L / P, rtol=1e-10)
        assert np.isclose(flat.loc[u, 'PQ'], P / Q, rtol=1e-10)
        assert np.isclose(flat.loc[u, 'ROE'], M / Q, rtol=1e-10)

    # the 'total' row is the diversified whole == pricing_at total
    pa = port.pricing_at(port.distortions[dname], p=p).loc['total']
    for stat in ['L', 'M', 'P', 'Q', 'a']:
        assert np.isclose(flat.loc['total', stat], pa[stat], rtol=1e-8)


def test_price_stand_alone_arg_checks(peg):
    """Bad arguments raise the documented exceptions."""
    port, _ = peg
    # first no-mass distortion: ccoc (mass) is refused on the
    # unbounded PEG by Aggregate/Portfolio gatekeeping (numerics-3)
    dname = next(n for n, d in port.distortions.items()
                 if not getattr(d, 'has_mass', False))
    dist = port.distortions[dname]

    with pytest.raises(ValueError):
        port.price_stand_alone(dist, p=1.5)            # p out of range
    with pytest.raises(TypeError):
        port.price_stand_alone(dist, p='0.99')         # p not numeric
    with pytest.raises(TypeError):
        port.price_stand_alone(12345, p=0.99)          # dist wrong type
    with pytest.raises(KeyError):
        port.price_stand_alone('no_such_distortion', p=0.99)


# ---------------------------------------------------------------------------
# distortion_df / calibration_df — de-crufted calibration summary (F10)
# ---------------------------------------------------------------------------

def test_distortion_df_schema(peg):
    """distortion_df is the per-distortion receipt: index 'distortion', columns
    param_name/param/gini_p/area/error, in canonical order."""
    port, _ = peg
    p = BASELINE['meta']['p_calibration']
    coc = BASELINE['meta']['coc_calibration']
    ddf = port.calibrate_distortions(coc=coc, p=p)

    assert ddf.index.name == 'distortion'
    assert list(ddf.index) == ['ccoc', 'ph', 'wang', 'dual', 'tvar']
    assert list(ddf.columns) == ['param_name', 'param', 'error', 'gini_p', 'area']
    assert ddf['param_name'].to_dict() == {
        'ccoc': 'r', 'ph': 'a', 'wang': 'lam', 'dual': 'b', 'tvar': 'p'}
    # area == (gini_p + 1)/2 == int g, exactly (definitional)
    assert np.allclose(ddf['area'], (ddf['gini_p'] + 1) / 2, rtol=0, atol=1e-12)
    # the verified identity: tvar's raw param IS its gini_p (TVaR-equiv level)
    assert np.isclose(ddf.loc['tvar', 'param'], ddf.loc['tvar', 'gini_p'], rtol=1e-9)
    # premium miss is small
    assert (ddf['error'].abs() < 1e-4).all()


def test_calibration_df_self_contained(peg):
    """calibration_df leads with the inputs coc, p then the pentagon octet; the
    ROE == coc self-check holds and the octet trails (iloc[:, -8:])."""
    port, _ = peg
    p = BASELINE['meta']['p_calibration']
    coc = BASELINE['meta']['coc_calibration']
    port.calibrate_distortions(coc=coc, p=p)
    cal = port.calibration_df

    assert len(cal) == 1
    assert list(cal.columns) == [
        'coc', 'p', 'F(a)', 'L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']
    row = cal.iloc[0]
    assert np.isclose(row['coc'], coc)
    assert np.isclose(row['p'], p)
    assert np.isclose(row['ROE'], coc, rtol=1e-6)         # the self-check
    assert np.isclose(row['a'], row['P'] + row['Q'])
    assert list(cal.iloc[:, -8:].columns) == [
        'L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']


def test_distortion_gini_p_attribute(peg):
    """Distortion exposes gini_p (renamed from standard_shape)."""
    port, _ = peg
    port.calibrate_distortions(coc=BASELINE['meta']['coc_calibration'],
                               p=BASELINE['meta']['p_calibration'])
    for dist in port.distortions.values():
        assert hasattr(dist, 'gini_p')
        assert not hasattr(dist, 'standard_shape')
