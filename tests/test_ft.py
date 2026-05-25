"""Regression tests for the FourierTools direct chf-inversion class."""
from __future__ import annotations

import numpy as np
import scipy.stats as ss

from aggregate.ft import FourierTools


def test_fourier_tools_inverts_normal():
    """Inverting φ(t) = exp(-t²/2) recovers a PMF with the standard-normal moments."""
    chf = lambda t: np.exp(-0.5 * t**2)
    fz = ss.norm()
    ft_obj = FourierTools(chf, fz)
    ft_obj.invert(log2=12, x_min=-8.0, x_max=8.0)
    df = ft_obj.df
    x = df.index.values
    p = df['p'].values
    # PMF sums to 1, mean ~ 0, variance ~ 1.
    assert np.isclose(p.sum(), 1.0, atol=1e-6)
    assert abs(np.sum(x * p)) < 1e-6
    assert np.isclose(np.sum(x**2 * p), 1.0, atol=1e-3)
