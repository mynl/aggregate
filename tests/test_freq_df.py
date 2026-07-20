"""Tests for ``Frequency.freq_df`` and resolved ``Aggregate.en``.

``freq_df`` is an on-demand cached comparison table: the count pmf against
the mean-matched Poisson pmf. Parametric kinds read ``Frequency.en`` (the
unconditional expected count the owning Aggregate stamps at construction)
and invert their pgf; the empirical family uses its exact materialized pmf.
``Aggregate.en`` must hold the resolved per-component claim count -- not
the ``-1`` empirical/renewal spec sentinel.
"""

import numpy as np
import pytest
import scipy.stats as ss

from aggregate import build


def test_dfreq_freq_df_exact():
    a = build('agg D dfreq [0 2] [.5 .5] dsev [1]', update=False)
    df = a.frequency.freq_df
    assert df.index.name == 'n'
    assert list(df.columns) == ['p', 'po_p']
    # sparse support: the hole at n = 1 carries p = 0
    assert np.allclose(df.p.values, [0.5, 0.0, 0.5])
    assert np.allclose(df.po_p.values, ss.poisson.pmf([0, 1, 2], 1.0))


def test_renewal_expon_freq_df_matches_poisson():
    # expon waits => Poisson counts: the two columns coincide (the toy's
    # flagship read); tolerance is the kernel's ~2e-9 tilt noise floor
    a = build('agg R 10 years dsev [1] wait expon', update=False)
    df = a.frequency.freq_df
    assert np.abs(df.p - df.po_p).max() < 1e-7


def test_freq_df_cached_and_on_demand():
    a = build('agg C dfreq [1 2 3] dsev [1]', update=False)
    fr = a.frequency
    # not computed until accessed (cached_property stores in __dict__)
    assert 'freq_df' not in fr.__dict__
    df = fr.freq_df
    assert fr.freq_df is df


def test_freq_df_parametric_poisson():
    # the Aggregate stamps its resolved en onto the frequency; the PGF
    # inversion then reproduces the mean-matched Poisson exactly
    a = build('agg P 10 claims dsev [1] poisson', update=False)
    assert a.frequency.en == pytest.approx(10.0)
    df = a.frequency.freq_df
    assert np.abs(df.p - df.po_p).max() < 1e-12
    assert df.p.sum() == pytest.approx(1.0, abs=1e-12)


def test_freq_df_mixed_gamma_overdispersed():
    # gamma-mixed Poisson (negbin): variance ratio = 1 + cv^2 * n, mean
    # matched to the Poisson column
    a = build('agg G 10 claims dsev [1] mixed gamma 0.5', update=False)
    df = a.frequency.freq_df
    k = np.arange(len(df))
    m1 = float(k @ df.p)
    var = float((k * k) @ df.p) - m1 * m1
    assert m1 == pytest.approx(10.0, abs=1e-9)
    assert var / m1 == pytest.approx(1 + 0.5 ** 2 * 10, abs=1e-9)
    assert df.p.sum() == pytest.approx(1.0, abs=1e-12)


def test_freq_df_standalone_needs_en():
    from aggregate import Frequency
    fr = Frequency('poisson', 0, 0, False, np.nan)
    with pytest.raises(ValueError, match='standalone'):
        fr.freq_df
    fr.en = 3.0
    df = fr.freq_df
    assert np.abs(df.p - df.po_p).max() < 1e-12


def test_freq_df_mean_guard():
    a = build('agg B dfreq [2000] dsev [1]', update=False)
    with pytest.raises(ValueError, match='1000'):
        a.frequency.freq_df
    p = build('agg BP 1500 claims dsev [1] poisson', update=False)
    with pytest.raises(ValueError, match='1000'):
        p.frequency.freq_df


def test_freq_df_fractional_support_raises():
    a = build('agg F dfreq [0.5 1.5] dsev [1]', update=False)
    with pytest.raises(ValueError, match='integer'):
        a.frequency.freq_df


def test_en_resolved_not_sentinel():
    # the -1 empirical/renewal spec sentinel must not leak into Aggregate.en
    r = build('agg R2 10 years sev lognorm 100 cv 1 wait expon',
              update=False)
    assert r.en[0] == pytest.approx(r.n, abs=1e-12)
    d = build('agg D2 dfreq [1 2 3] dsev [1]', update=False)
    assert d.en[0] == pytest.approx(2.0, abs=1e-12)
    # regressions: parametric and mixture-product arms unchanged
    p = build('agg P2 10 claims dsev [1] poisson', update=False)
    assert p.en[0] == pytest.approx(10.0)
    m = build('agg M2 10 claims sev [lognorm gamma] [10 20] cv [1 2] '
              'wts [.6 .4] poisson', update=False)
    assert np.allclose(m.en, [6.0, 4.0])
