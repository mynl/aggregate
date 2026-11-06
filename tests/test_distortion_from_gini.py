"""Distortions from a Gini index: ``Distortion.from_gini`` (2.1.0a4).

The cases follow the numbered tests of
``dev/done/plan-tsallis-distortion-and-gini.md`` for ``[Distortion-From-Gini]``:
the round trip and refusals of ``from_gini`` (10), calibration on the uniform
(7), and ``p_to_parameters`` / ``consistent_distortions`` unchanged and
extended (11).
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import Distortion
from aggregate._pricing import (DEFAULT_CALIBRATION_DISTORTIONS,
                                EXTENDED_CALIBRATION_DISTORTIONS)
from aggregate.spectral import consistent_distortions, p_to_parameters, phi_inv

GINIS = [0.1, 0.3, 0.6]
ONE_PARAMETER = EXTENDED_CALIBRATION_DISTORTIONS


# --- 10. from_gini ------------------------------------------------------------

@pytest.mark.parametrize('kind', ONE_PARAMETER)
@pytest.mark.parametrize('gini', [0.01, *GINIS, 0.99])
def test_from_gini_round_trips(kind, gini):
    d = Distortion.from_gini(kind, gini)
    assert d.kind == kind
    assert d.gini == pytest.approx(gini, abs=1e-12)


def test_from_gini_parameters_and_label():
    assert Distortion.from_gini('tsallis', 0.25).alpha == 3
    assert Distortion.from_gini('ccoc', 0.25).d == pytest.approx(0.25)
    assert Distortion.from_gini('roe', 0.25).kind == 'ccoc'
    assert Distortion.from_gini('ph', 0.5, label='half').label == 'half'
    assert Distortion.from_gini('ph', 0.5).label == 'PH(0.333)'


@pytest.mark.parametrize('kind, named', [
    ('bitvar', "('p0', 'p1', 'w1')"), ('wtdtvar', "('ps', 'wts')"),
    ('beta', "('a', 'b')"), ('power', "('x0', 'x1', 'alpha')"),
    ('cll', "('r0', 'b')"), ('clin', "('r0', 'slope')"),
    ('lep', "('r0', 'r')"), ('ly', "('r0', 'r')"),
    ('minimum', 'distortion references'), ('mixture', 'distortion references'),
])
def test_from_gini_refuses_a_family_with_more_than_one_parameter(kind, named):
    with pytest.raises(ValueError, match='one-parameter family') as info:
        Distortion.from_gini(kind, 0.3)
    assert repr(kind) in str(info.value) and named in str(info.value)


@pytest.mark.parametrize('gini', [0, 1, -0.1, 1.5])
def test_from_gini_refuses_an_index_outside_the_open_interval(gini):
    with pytest.raises(ValueError, match=r'open interval \(0, 1\)'):
        Distortion.from_gini('ph', gini)


def test_from_gini_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match='Unknown distortion kind'):
        Distortion.from_gini('frobnicate', 0.3)


# --- 7. calibration on the uniform ---------------------------------------------

@pytest.mark.parametrize('gini', GINIS)
def test_calibrating_on_the_uniform_recovers_from_gini(gini):
    """For a uniform ``U``, ``rho_g(U) = int g = (1 + G)/2``.

    A midpoint grid makes the grid integral of ``S`` exactly 1/2. The
    tolerance is twice the 1e-5 premium tolerance of the Newton families.
    """
    n = 2 ** 16
    dx = 1 / n
    S = 1 - (np.arange(n) + 0.5) * dx
    fitted = Distortion.calibrate_set(S=S, dx=dx, premium_target=(1 + gini) / 2,
                                      assets=1.0, el=0.5, names=ONE_PARAMETER)
    assert list(fitted) == list(ONE_PARAMETER)
    for kind, d in fitted.items():
        target = Distortion.from_gini(kind, gini)
        assert d.gini == pytest.approx(gini, abs=2e-5), kind
        assert d.shape == pytest.approx(target.shape, rel=1e-4), kind


# --- the mixture's closed form ------------------------------------------------

def test_mixture_gini_is_the_weighted_sum_in_closed_form():
    a, b = Distortion('ph', 0.5), Distortion('tsallis', 3)
    mix = Distortion.mixture([a, b], [0.25, 0.75])
    assert mix._describe_closed_form() == {'p_equiv': pytest.approx(0.25 / 3 + 0.75 / 4)}
    assert mix.gini == pytest.approx(0.25 * a.gini + 0.75 * b.gini, abs=1e-15)
    unnormalized = Distortion.mixture([a, b], [0.5, 1.0])
    s = np.linspace(0, 1, 200_001)
    assert unnormalized.gini == pytest.approx(2 * np.trapezoid(unnormalized.g(s), s) - 1,
                                              abs=1e-6)


# --- 11. p_to_parameters and consistent_distortions ---------------------------

def _p_to_parameters_200(p):
    """``p_to_parameters`` as it stood at 2.0.0."""
    return {'ccoc': p, 'ph': (1 - p) / (1 + p), 'wang': 2 ** 0.5 * phi_inv((1 + p) / 2),
            'dual': (1 + p) / (1 - p), 'tvar': p}


@pytest.mark.parametrize('p', [0.125, 0.25, 0.6])
def test_p_to_parameters_is_unchanged(p):
    assert p_to_parameters(p) == _p_to_parameters_200(p)
    extended = p_to_parameters(p, extended=True)
    assert tuple(extended) == EXTENDED_CALIBRATION_DISTORTIONS
    assert extended['tsallis'] == pytest.approx(1 / p - 1)
    assert {k: v for k, v in extended.items() if k != 'tsallis'} == p_to_parameters(p)


@pytest.mark.parametrize('p', [0.125, 0.25, 0.6])
def test_consistent_distortions_are_unchanged(p):
    old = _p_to_parameters_200(p)
    s = np.linspace(0, 1, 1001)
    now = consistent_distortions(p)
    assert tuple(now) == DEFAULT_CALIBRATION_DISTORTIONS
    for k, d in now.items():
        ref = (Distortion('ccoc', r=old[k] / (1 - old[k])) if k == 'ccoc'
               else Distortion(k, old[k]))
        assert d.shape == ref.shape and d.label == ref.label and d.id() == ref.id(), k
        np.testing.assert_array_equal(d.g(s), ref.g(s))
    extended = consistent_distortions(p, extended=True)
    assert tuple(extended) == EXTENDED_CALIBRATION_DISTORTIONS
    assert extended['tsallis'].gini == pytest.approx(p, abs=1e-15)


def test_consistent_tsallis_at_an_eighth_is_order_seven():
    d = consistent_distortions(0.125, extended=True)['tsallis']
    assert d.alpha == 7
    assert float(d.g(0.1)) == pytest.approx((0.7 - 0.1 ** 7) / 6, rel=1e-14)
