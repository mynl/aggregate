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

from aggregate import Distortion, build


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
    d.calibrate(S=S, dx=bs, premium_target=prem)

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
    d.calibrate(S=S, dx=bs, premium_target=prem, ess_sup=ess_sup)

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
    d.calibrate(S=S, dx=bs, premium_target=prem, assets=assets, el=el)

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
        d.calibrate(S=np.array([0.5]), dx=1.0, premium_target=1.0)


def test_calibration_init_shape_present_for_pricing_kinds():
    """All migrated pricing kinds set ``_calibration_init_shape`` to a
    valid non-None starting shape."""
    expected = {'ph', 'wang', 'dual', 'tvar', 'ccoc',
                'ly', 'clin', 'lep', 'cll'}
    for kind in expected:
        subclass = Distortion._registry[kind]
        assert subclass._calibration_init_shape is not None, \
            f'{kind} has no _calibration_init_shape'


# --- Distortion.calibrate_set: the family-loop classmethod (Phase 1c) -------

def test_calibrate_set_matches_individual(synthetic_S):
    """``calibrate_set`` returns one calibrated distortion per name, each
    identical to the individual ``calibrate`` call and hitting the target."""
    S, bs, el = synthetic_S['S'], synthetic_S['bs'], synthetic_S['el']
    assets = synthetic_S['assets']
    prem = el * 1.20

    dset = Distortion.calibrate_set(S=S, dx=bs, premium_target=prem,
                                    assets=assets, el=el)
    assert list(dset) == ['ccoc', 'ph', 'wang', 'dual', 'tvar']
    for name, d in dset.items():
        assert abs(d.error) < 1e-4, f'{name} residual {d.error}'
        assert abs(_achieved_premium(d, S, bs) - prem) < 1e-3

    # set member equals the stand-alone calibrate for a representative kind
    sub = Distortion._registry['ph']
    d_ph = Distortion(name='ph', **{sub.param_name: sub._calibration_init_shape})
    d_ph.calibrate(S=S, dx=bs, premium_target=prem, assets=assets, el=el)
    assert dset['ph'].shape == pytest.approx(d_ph.shape)


def test_calibrate_set_unknown_kind_raises(synthetic_S):
    """A non-calibratable kind in ``names`` raises ValueError."""
    with pytest.raises(ValueError, match='calibrate_set not implemented'):
        Distortion.calibrate_set(S=synthetic_S['S'], dx=synthetic_S['bs'],
                                 premium_target=1.0, names=('minimum',))


# --- Aggregate / Portfolio calibration parity (Phase 1c) --------------------

def test_aggregate_calibrate_distortions_parity():
    """``Aggregate.calibrate_distortions`` matches the legacy one-unit-Portfolio
    path bit-for-bit (same grid), and stores the standard receipts."""
    prog = 'agg Solo 80 claims sev lognorm 50 cv 2 poisson'
    agg = build(prog)
    agg.update(log2=18, bs=1)
    port = build('port Wrap ' + prog)
    port.update(log2=18, bs=1, add_exa=True)

    a = agg.q(0.99)
    add = agg.calibrate_distortions(0.1, a=a).distortion_df
    pdf = port.calibrate_distortions(0.1, a=a).distortion_df

    assert list(add.index) == ['ccoc', 'ph', 'wang', 'dual', 'tvar']
    np.testing.assert_allclose(add['param'].values, pdf['param'].values,
                               rtol=1e-6, atol=1e-8)
    # side effects: calibrated set + receipts stored on the Aggregate
    assert set(agg.distortions) == {'ccoc', 'ph', 'wang', 'dual', 'tvar'}
    assert agg.calibration_df.loc['calibration', 'ROE'] == pytest.approx(0.1)


def test_aggregate_price_ccoc_matches_pentagon():
    """``Aggregate.price_ccoc`` is the cost-of-capital alias of price_pentagon."""
    agg = build('agg Solo 50 claims sev lognorm 40 cv 1.8 poisson')
    agg.update(log2=18, bs=1)
    import pandas.testing as pdt
    pdt.assert_frame_equal(agg.price_ccoc(0.1, p=0.99),
                           agg.price_pentagon(p=0.99, ROE=0.1))


# ---------------------------------------------------------------------------
# Evaluation: the grid-agnostic quadrature and the two new public faces
# [Margin-Acceptability-Evaluate]
# ---------------------------------------------------------------------------

def test_quad_scalar_branch_keeps_the_legacy_summation_order():
    """A scalar ``dx`` must stay bit-for-bit ``np.sum(v) * dx``.

    The whole point of the scalar branch: the classic lattice pricing path is
    unchanged, so ``calibrate_distortions`` keeps its byte-for-byte guarantee.
    ``np.sum(v * dx)`` is mathematically the same and numerically is not.
    """
    v = np.linspace(0.999, 0.001, 5000)
    assert Distortion._quad(v, 0.1) == np.sum(v) * 0.1
    dx = np.full(len(v), 0.1)
    assert Distortion._quad(v, dx) == np.sum(v * dx)


def test_quad_vector_branch_is_exact_on_an_irregular_grid():
    """``sum g(S_i) (z_{i+1} - z_i)`` is the layer integral, not an estimate.

    A three-atom law has a closed-form distorted mean, and the width vector
    reproduces it to machine precision. Rebucketing onto a lattice would match
    the mean and miss this.
    """
    # Z in {0, 1.7, 11.9}, masses .5 / .3 / .2, so S = (.5, .2) on the first
    # two nodes and rho_g(Z) = g(.5) * 1.7 + g(.2) * 10.2.
    z = np.array([0.0, 1.7, 11.9])
    S = np.array([0.5, 0.2])
    d = Distortion('ph', 0.6)
    exact = d.g(0.5) * 1.7 + d.g(0.2) * 10.2
    assert Distortion._quad(d.g(S), np.diff(z)) == pytest.approx(exact,
                                                                 rel=1e-15)


def test_evaluate_on_an_irregular_margin_is_exact():
    """A margin off any lattice still solves rho_g(M) = 0.

    The atoms are deliberately incommensurable, so the margin's support is
    genuinely irregular and the width-vector quadrature is the only exact
    route. The bound is ``_newton_iterate``'s own ``tol`` of 1e-5, which is
    what the solve promises; exactness of the *quadrature* is pinned
    bit-for-bit by ``test_quad_vector_branch_is_exact_on_an_irregular_grid``.
    A rebucketed implementation would miss by orders more on this support.
    """
    from aggregate import PnL
    vals = np.array([0.0, 1.7, 4.3, 11.9])
    probs = np.array([0.45, 0.25, 0.2, 0.1])
    p = PnL(name='irr', source=(vals, probs), role='sell',
            consideration=3.0, obligation=lambda x: x)
    assert p.result.bs is None                    # exact irregular grid
    ev = p.evaluate().evaluation_df.droplevel('Step')
    x, q = np.asarray(p.result.x), np.asarray(p.result.p)
    c = float(x.max())
    z, qz = (c - x)[::-1], q[::-1]
    S = 1.0 - np.cumsum(qz)
    k = min(int(np.flatnonzero(S > 0).max()), len(z) - 2)
    for fam in ('ph', 'wang', 'dual', 'tvar'):
        g = Distortion(fam, ev.loc[fam, 'param']).g
        rho = c - float(np.sum(g(S[:k + 1]) * np.diff(z)[:k + 1]))
        assert rho == pytest.approx(0.0, abs=1e-5)


def test_aggregate_evaluate_defaults_to_exp_premium_and_raises_without():
    a = build('agg AEv 1000 premium at 0.7 lr sev gamma 100 cv 0.5 poisson')
    result = a.evaluate()
    ev = result.evaluation_df
    # the result knows the position it measured, not just the shapes
    assert result.premium == pytest.approx(1000.0)
    assert result.name == 'AEv' and result.reins_view is None
    assert list(ev.index.get_level_values('Step').unique()) == ['AEv']
    assert (ev.status == 'ok').all()
    # an aggregate is an obligation written, so it is read as booked
    assert (ev.role == 'sell').all()
    # the explicit premium reproduces the default
    assert ev.param.to_numpy() == pytest.approx(
        a.evaluate(1000.0).evaluation_df.param.to_numpy())
    bare = build('agg ANoP 100 claims sev gamma 10 cv 1 poisson')
    with pytest.raises(ValueError, match='no premium to evaluate against'):
        bare.evaluate()


def test_aggregate_and_pnl_evaluate_agree():
    """The two faces are one solve: a constant premium is just a margin."""
    a = build('agg AX 1000 premium at 0.7 lr sev gamma 100 cv 0.5 poisson')
    p = build('pnl PX 1000 prem less agg PX_e 1000 prem at 0.7 lr '
              'sev gamma 100 cv 0.5 poisson')
    assert a.evaluate().evaluation_df.param.to_numpy() == pytest.approx(
        p.evaluate().evaluation_df.param.to_numpy(), rel=1e-6)


def test_portfolio_evaluate_total_and_unit_profile():
    """The book survives more stress than either unit: diversification."""
    port = build('port PEv '
                 'agg PU1 1000 premium at 0.65 lr sev gamma 100 cv 0.5 poisson '
                 'agg PU2 500 premium at 0.75 lr sev gamma 50 cv 1.2 poisson')
    total = port.evaluate().evaluation_df.droplevel('Step')
    profile = port.evaluate(unit=['PU1', 'PU2']).evaluation_df
    assert list(profile.index.get_level_values('Step').unique()) == \
        ['PU1', 'PU2']
    assert (profile.role == 'sell').all()      # every unit is written, not bought
    for fam in ('ph', 'wang', 'dual', 'tvar'):
        assert (total.loc[fam, 'gini_p']
                > profile.loc[('PU1', fam), 'gini_p']
                > profile.loc[('PU2', fam), 'gini_p'])
    with pytest.raises(ValueError, match='scalar P is ambiguous'):
        port.evaluate(900.0, unit=['PU1', 'PU2'])


def test_evaluate_margin_rejects_a_loss_oriented_gd():
    """Orientation is not guessed: a loss-valued GD is a caller error."""
    from aggregate._grid_distribution import GridDistribution
    from aggregate._pricing import evaluate_margin
    gd = GridDistribution(np.array([0.0, 1.0]), np.array([0.5, 0.5]),
                          is_loss_value=True)
    with pytest.raises(ValueError, match='payoff orientation'):
        evaluate_margin(gd)
