"""Regression tests for the Tweedie module.

Lightweight in-regression coverage of the three public surfaces in
``aggregate.tweedie``: the parameter-translator ``tweedie_convert``,
the series-expansion density ``tweedie_density``, and the ``Tweedie``
class. Restricted to the compound-Poisson-gamma regime (1 < p < 2)
where closed-form moments are available from V(μ) = dispersion · μ^p.
Heavy-tail regimes (p > 2, p < 0, Cauchy at p=∞) need Fourier
inversion and tolerance-tuning that doesn't belong in a fast suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import tweedie_convert, tweedie_density
from aggregate.tweedie import Tweedie


# (p, mean, dispersion) — compound-Poisson-gamma regime.
PCASES = [
    (1.5, 10.0, 1.0),
    (1.5, 100.0, 0.5),
    (1.1, 10.0, 1.0),
    (1.9, 10.0, 1.0),
]


@pytest.mark.parametrize("p,mu,disp", PCASES)
def test_tweedie_convert_roundtrip(p, mu, disp):
    """(p, μ, σ²) → (λ, α, β) → (μ, σ²) is the identity."""
    fwd = tweedie_convert(p=p, μ=mu, σ2=disp)
    back = tweedie_convert(λ=fwd["λ"], α=fwd["α"], β=fwd["β"])
    assert np.isclose(back["μ"], mu, rtol=1e-10)
    assert np.isclose(back["σ^2"], disp, rtol=1e-10)
    assert np.isclose(back["p"], p, rtol=1e-10)


@pytest.mark.parametrize("p,mu,disp", PCASES)
def test_tweedie_density_finite_positive(p, mu, disp):
    """Density at x = μ is finite and strictly positive."""
    d = tweedie_density(mu, p=p, μ=mu, σ2=disp)
    assert np.isfinite(d)
    assert d > 0


@pytest.mark.parametrize("p,mu,disp", PCASES)
def test_tweedie_moments_match_definition(p, mu, disp):
    """Tweedie reproductive moments satisfy mean=μ, var = disp · μ^p."""
    tw = Tweedie(p, mean=mu, dispersion=disp)
    m, v, _ = tw.stats()
    assert np.isclose(m, mu, rtol=1e-4)
    assert np.isclose(v, disp * mu**p, rtol=1e-4)


def test_tweedie_dual_involution():
    """``Tweedie(...).dual().dual()`` returns the same reproductive params."""
    tw = Tweedie(1.5, mean=100.0, dispersion=0.5)
    back = tw.dual().dual()
    assert np.isclose(back.mean, tw.mean, rtol=1e-12)
    assert np.isclose(back.dispersion, tw.dispersion, rtol=1e-12)
