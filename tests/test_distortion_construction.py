"""
Construction-surface matrix tests for Distortion.

For each kind covered by the refactor, exercise every supported
construction form:

* ``Distortion(kind, positional)`` — factory + positional shape, where it
  makes sense (scalar kinds only)
* ``Distortion(kind, natural_kw=value)`` — factory + natural kwarg
* ``Subclass(natural_kw=value)`` — direct subclass construction
* ``Distortion.<kind>(...)`` — static convenience factory

For each construction the test asserts:

* ``g(0) == 0`` and ``g(1) == 1`` (distortion endpoint contract)
* ``g_inv(g(x)) ≈ x`` at an interior point (round-trip)
* The natural-name property reads back the expected value
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import Distortion
from aggregate.spectral import (
    PHDistortion, WangDistortion, DualDistortion, TVaRDistortion,
    CCoCDistortion, BiTVaRDistortion, WtdTVaRDistortion,
    BetaDistortion, PowerDistortion,
    CLLDistortion, CLinDistortion, LEPDistortion, LYDistortion,
    MinimumDistortion, MixtureDistortion,
)


# --- helpers ---------------------------------------------------------------

def _check_endpoints(d):
    assert float(d.g(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.g(1.0)) == pytest.approx(1.0, abs=1e-12)


def _check_roundtrip(d, x=0.5):
    gx = float(d.g(x))
    if 0 < gx < 1:
        rt = float(d.g_inv(gx))
        assert rt == pytest.approx(x, abs=1e-6)


# --- scalar-shape kinds (factory positional / factory kwarg / direct / static)

SCALAR_KINDS = [
    ('ph',   'a',   0.7, PHDistortion,   Distortion.ph),
    ('wang', 'lam', 0.3, WangDistortion, Distortion.wang),
    ('dual', 'b',   2.0, DualDistortion, Distortion.dual),
    ('tvar', 'p',   0.5, TVaRDistortion, Distortion.tvar),
]


@pytest.mark.parametrize('kind,pn,val,subclass,static', SCALAR_KINDS,
                         ids=[s[0] for s in SCALAR_KINDS])
def test_scalar_factory_positional(kind, pn, val, subclass, static):
    d = Distortion(kind, val)
    assert isinstance(d, subclass)
    assert getattr(d, pn) == pytest.approx(val)
    _check_endpoints(d)
    _check_roundtrip(d)


@pytest.mark.parametrize('kind,pn,val,subclass,static', SCALAR_KINDS,
                         ids=[s[0] for s in SCALAR_KINDS])
def test_scalar_factory_kwarg(kind, pn, val, subclass, static):
    d = Distortion(kind, **{pn: val})
    assert isinstance(d, subclass)
    assert getattr(d, pn) == pytest.approx(val)
    _check_endpoints(d)


@pytest.mark.parametrize('kind,pn,val,subclass,static', SCALAR_KINDS,
                         ids=[s[0] for s in SCALAR_KINDS])
def test_scalar_direct_subclass(kind, pn, val, subclass, static):
    d = subclass(**{pn: val})
    assert isinstance(d, subclass)
    assert getattr(d, pn) == pytest.approx(val)
    _check_endpoints(d)


@pytest.mark.parametrize('kind,pn,val,subclass,static', SCALAR_KINDS,
                         ids=[s[0] for s in SCALAR_KINDS])
def test_scalar_static_factory(kind, pn, val, subclass, static):
    d = static(val)
    assert isinstance(d, subclass)
    assert getattr(d, pn) == pytest.approx(val)
    _check_endpoints(d)


# --- CCoC: kwarg-only d= or r= ---------------------------------------------

def test_ccoc_via_r():
    d = Distortion('ccoc', r=0.1)
    assert d.r == pytest.approx(0.1)
    assert d.d == pytest.approx(0.1 / 1.1)
    _check_endpoints(d)


def test_ccoc_via_d():
    d = Distortion('ccoc', d=0.1)
    assert d.d == pytest.approx(0.1)
    assert d.r == pytest.approx(0.1 / 0.9)
    _check_endpoints(d)


def test_ccoc_positional_rejected():
    with pytest.raises(TypeError):
        Distortion('ccoc', 0.1)


def test_ccoc_both_rejected():
    with pytest.raises(TypeError):
        Distortion('ccoc', d=0.1, r=0.1)


def test_ccoc_neither_rejected():
    with pytest.raises(TypeError):
        Distortion('ccoc')


def test_ccoc_static_takes_discount():
    """The static factory ``Distortion.ccoc(d)`` takes the discount d."""
    d = Distortion.ccoc(0.1)
    assert d.d == pytest.approx(0.1)


def test_ccoc_direct_subclass():
    d = CCoCDistortion(r=0.1)
    assert d.r == pytest.approx(0.1)
    _check_endpoints(d)


# --- multi-param kinds -----------------------------------------------------

def test_bitvar():
    d = Distortion('bitvar', p0=0.95, p1=0.99, w1=0.5)
    assert d.p0 == 0.95 and d.p1 == 0.99 and d.w1 == 0.5
    _check_endpoints(d)


def test_bitvar_direct():
    d = BiTVaRDistortion(p0=0.95, p1=0.99, w1=0.5)
    assert d.p0 == 0.95 and d.p1 == 0.99 and d.w1 == 0.5
    _check_endpoints(d)


def test_wtdtvar():
    d = Distortion('wtdtvar', ps=[0.5, 0.9], wts=[0.3, 0.7])
    assert np.allclose(d.ps, [0.5, 0.9])
    assert np.allclose(d.wts, [0.3, 0.7])
    _check_endpoints(d)


def test_wtdtvar_normalises_fp_noise():
    """Weights summing to 1 + 1e-12 should be normalised silently."""
    d = Distortion('wtdtvar', ps=[0.5, 0.9], wts=[0.3 + 1e-12, 0.7])
    assert d.wts.sum() == pytest.approx(1.0)


def test_wtdtvar_rejects_wrong_sum():
    with pytest.raises(ValueError, match='sum'):
        Distortion('wtdtvar', ps=[0.5, 0.9], wts=[0.4, 0.4])


def test_wtdtvar_rejects_length_mismatch():
    with pytest.raises(ValueError, match='same length'):
        Distortion('wtdtvar', ps=[0.5, 0.9, 1.0], wts=[0.5, 0.5])


def test_beta():
    d = Distortion('beta', a=0.7, b=1.5)
    assert d.a == 0.7 and d.b == 1.5
    _check_endpoints(d)


def test_power():
    d = Distortion('power', x0=0.01, x1=1.0, alpha=2.0)
    assert d.x0 == 0.01 and d.x1 == 1.0 and d.alpha == 2.0
    _check_endpoints(d)


def test_minimum_kwarg():
    d1 = Distortion.ph(0.5)
    d2 = Distortion.wang(0.3)
    d = Distortion('minimum', distortions=[d1, d2])
    assert len(d.distortions) == 2
    _check_endpoints(d)


def test_minimum_positional():
    """The distortions list may also be passed positionally."""
    d1 = Distortion.ph(0.5)
    d2 = Distortion.wang(0.3)
    d = Distortion('minimum', [d1, d2])
    assert len(d.distortions) == 2
    _check_endpoints(d)


def test_mixture_kwarg():
    d1 = Distortion.ph(0.5)
    d2 = Distortion.wang(0.3)
    d = Distortion('mixture', distortions=[d1, d2], wts=[0.5, 0.5])
    assert len(d.distortions) == 2
    assert np.allclose(d.wts, [0.5, 0.5])
    _check_endpoints(d)


def test_mixture_positional():
    """The distortions list may be positional; ``wts`` is keyword-only."""
    d1 = Distortion.ph(0.5)
    d2 = Distortion.wang(0.3)
    d = Distortion('mixture', [d1, d2], wts=[0.6, 0.4])
    assert len(d.distortions) == 2
    assert np.allclose(d.wts, [0.6, 0.4])
    _check_endpoints(d)


# --- mass-at-zero kinds ----------------------------------------------------

def test_cll():
    d = Distortion('cll', r0=0.05, b=0.9)
    assert d.r0 == 0.05 and d.b == 0.9
    _check_endpoints(d)


def test_clin():
    d = Distortion('clin', r0=0.05, slope=2.0)
    assert d.r0 == 0.05 and d.slope == 2.0
    _check_endpoints(d)


def test_lep():
    d = Distortion('lep', r0=0.03, r=0.15)
    assert d.r0 == 0.03 and d.r == 0.15
    _check_endpoints(d)


def test_ly():
    d = Distortion('ly', r0=0.05, r=1.25)
    assert d.r0 == 0.05 and d.r == 1.25
    _check_endpoints(d)


# --- error paths -----------------------------------------------------------

def test_unknown_kwarg():
    with pytest.raises(TypeError, match='unexpected'):
        Distortion('ph', a=0.7, banana=1)


def test_both_positional_and_natural_kwarg():
    with pytest.raises(TypeError):
        Distortion('ph', 0.5, a=0.7)


# --- roe alias -------------------------------------------------------------

def test_roe_aliases_ccoc():
    d = Distortion('roe', r=0.1)
    assert isinstance(d, CCoCDistortion)


# --- DecL combo distortions (minimum / mixture via dist.X references) -----

def test_decl_minimum_combo():
    """DecL: ``dist MIN minimum dist.A dist.B`` resolves children via the
    knowledge base and builds a MinimumDistortion."""
    from aggregate import build
    build('dist tcc ccoc .25')
    build('dist tdu dual 2.5')
    m = build('dist tMIN minimum dist.tcc dist.tdu')
    from aggregate.spectral import MinimumDistortion
    assert isinstance(m, MinimumDistortion)
    assert len(m.distortions) == 2


def test_decl_mixture_combo_with_weights():
    """DecL: ``dist MIX mixture dist.A dist.B wts [w1 w2]``."""
    from aggregate import build
    build('dist tcc2 ccoc .25')
    build('dist tdu2 dual 2.5')
    mx = build('dist tMIX mixture dist.tcc2 dist.tdu2 wts [0.6 0.4]')
    from aggregate.spectral import MixtureDistortion
    assert isinstance(mx, MixtureDistortion)
    assert np.allclose(mx.wts, [0.6, 0.4])


def test_decl_mixture_combo_uniform_default():
    """DecL: omitting the ``wts`` clause yields uniform weighting."""
    from aggregate import build
    build('dist tcc3 ccoc .25')
    build('dist tdu3 dual 2.5')
    build('dist tph3 ph 0.7')
    mx = build('dist tMIX3 mixture dist.tcc3 dist.tdu3 dist.tph3')
    from aggregate.spectral import MixtureDistortion
    assert isinstance(mx, MixtureDistortion)
    assert len(mx.distortions) == 3
    assert np.allclose(mx.wts, [1/3, 1/3, 1/3])


def test_decl_combo_full_prefix():
    """The full ``distortion.`` prefix also works alongside the short ``dist.``."""
    from aggregate import build
    build('dist tcc4 ccoc .25')
    build('dist tdu4 dual 2.5')
    m = build('dist tMIN4 minimum distortion.tcc4 distortion.tdu4')
    from aggregate.spectral import MinimumDistortion
    assert isinstance(m, MinimumDistortion)


def test_decl_combo_parsed_program_has_object():
    """Regression: ParsedProgram.object is set for combo distortions
    just like for the scalar forms."""
    from aggregate import build
    build('dist tcc5 ccoc .25')
    build('dist tdu5 dual 2.5')
    out = build.build_many('dist tMIN5 minimum dist.tcc5 dist.tdu5')
    assert len(out) == 1
    assert out[0].object is not None
    from aggregate.spectral import MinimumDistortion
    assert isinstance(out[0].object, MinimumDistortion)


# --- property setters re-trigger _build ------------------------------------

def test_property_setter_rebuilds():
    d = Distortion('ph', a=0.7)
    g_at_half_before = float(d.g(0.5))
    d.a = 0.5
    g_at_half_after = float(d.g(0.5))
    assert g_at_half_after != pytest.approx(g_at_half_before)
    assert d.a == 0.5
