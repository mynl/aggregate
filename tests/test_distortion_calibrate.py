"""Per-subclass calibration unit tests for ``Distortion.calibrate``.

Sub-project C moved the per-distortion Newton iterations out of
``Portfolio.calibrate_distortion`` and onto the pricing-distortion
subclasses. These tests exercise each subclass's ``calibrate`` method
directly on a synthetic ``S`` vector, asserting that the calibrated
distortion reproduces the target premium when applied to ``S``.

The Portfolio-level numerics are pinned separately by
``test_portfolio_peg_regression.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import Distortion


# --- shared fixtures --------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_S():
    """Linearly decreasing S on a 100-bucket grid with bs=0.1."""
    n, bs = 100, 0.1
    S = np.linspace(0.999, 0.001, n)
    el = float(np.sum(S) * bs)
    return {'S': S, 'bs': bs, 'el': el, 'assets': n * bs}


def _achieved_premium(d, S, bs):
    """Compute ``∫ g(S) dx`` for the calibrated distortion."""
    return float(np.sum(d.g(S)) * bs)


# --- the 5 strict_pricing kinds with no r0 / no mass ------------------------

@pytest.mark.parametrize('kind', ['ph', 'wang', 'dual', 'tvar', 'cll'])
def test_calibrate_no_r0(synthetic_S, kind):
    """ph/wang/dual/tvar/cll: Newton iteration hits the target premium."""
    S = synthetic_S['S']
    bs = synthetic_S['bs']
    el = synthetic_S['el']
    prem = el * 1.20

    subclass = Distortion._registry[kind]
    init = subclass._calibration_init_shape
    if kind == 'cll':
        d = Distortion(name=kind, r0=0.0, b=init)
    else:
        d = Distortion(name=kind, **{subclass.param_name: init})
    d.calibrate(S=S, bs=bs, premium_target=prem)

    assert abs(d.error) < 1e-4, \
        f'{kind} residual {d.error} exceeds tolerance'
    assert abs(_achieved_premium(d, S, bs) - prem) < 1e-3, \
        f'{kind} achieved premium does not match target'
    assert d.premium_target == prem
    assert d.assets == 0.0  # default; not passed here


# --- mass-at-zero kinds: ly, clin, lep --------------------------------------

@pytest.mark.parametrize('kind,r0', [('ly', 0.03), ('clin', 0.03), ('lep', 0.03)])
def test_calibrate_with_r0(synthetic_S, kind, r0):
    """ly/clin/lep: calibration includes a mass-at-zero term proportional
    to ``ess_sup`` and ``r0``."""
    S = synthetic_S['S']
    bs = synthetic_S['bs']
    el = synthetic_S['el']
    ess_sup = 10.0
    prem = el * 1.20 + 0.5  # leave room above EL + mass

    subclass = Distortion._registry[kind]
    init = subclass._calibration_init_shape
    pn = {'ly': 'r', 'clin': 'slope', 'lep': 'r'}[kind]
    d = Distortion(name=kind, r0=r0, **{pn: init})
    d.calibrate(S=S, bs=bs, premium_target=prem, ess_sup=ess_sup)

    assert abs(d.error) < 1e-4
    assert d.premium_target == prem


# --- ccoc: closed form ------------------------------------------------------

def test_calibrate_ccoc(synthetic_S):
    """CCoC is closed-form: ``r = (P - el) / (a - P)``."""
    S = synthetic_S['S']
    bs = synthetic_S['bs']
    el = synthetic_S['el']
    assets = 15.0
    prem = el * 1.20

    d = Distortion(name='ccoc', r=0.25)
    d.calibrate(S=S, bs=bs, premium_target=prem, assets=assets, el=el)

    expected = (prem - el) / (assets - prem)
    assert d.shape == pytest.approx(expected)
    assert d.r == pytest.approx(expected)
    assert d.error == 0.0
    assert d.assets == assets


def test_calibrate_roe_aliases_ccoc():
    """The legacy alias 'roe' resolves to CCoCDistortion."""
    subclass = Distortion._registry.get('ccoc')
    assert Distortion._registry.get('roe') is None  # not registered
    # but Distortion(name='roe') still works via __new__ alias
    d = Distortion(name='roe', r=0.25)
    assert type(d) is subclass


# --- dispatch error paths ---------------------------------------------------

def test_base_calibrate_raises():
    """Non-pricing kinds (e.g. ``minimum``) should NotImplementedError."""
    sub_d1 = Distortion.ph(0.5)
    sub_d2 = Distortion.wang(0.3)
    d = Distortion.minimum([sub_d1, sub_d2])
    with pytest.raises(NotImplementedError):
        d.calibrate(S=np.array([0.5]), bs=1.0, premium_target=1.0)


def test_calibration_init_shape_present_for_pricing_kinds():
    """All migrated pricing kinds set ``_calibration_init_shape`` to a
    valid non-None starting shape."""
    expected = {'ph', 'wang', 'dual', 'tvar', 'ccoc',
                'ly', 'clin', 'lep', 'cll'}
    for kind in expected:
        subclass = Distortion._registry[kind]
        assert subclass._calibration_init_shape is not None, \
            f'{kind} has no _calibration_init_shape'
