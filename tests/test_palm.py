"""Tests for the Palm conditional-mean kernel ([Palm-Kernel], a362).

``freq_pgf_prime`` (the pgf derivative on ``Frequency``) and
``palm_conditional_mean`` (the 1-D shared-event conditional mean in
``_aggregate_compute``) implement the identity

    E[S_c | S_n = s] * f_{S_n}(s) = (nu * w)(s),   FT(w) = P_N'(phi_n),

for compounds sharing one event process. Pinned here: an exact brute-force
enumeration on a tiny discrete compound, the Panjer self-check
(``kappa(s) = s`` when conditioning a book on itself), agreement of the
(a, b, 0) route with the closed-form mixed-Poisson route on the negative
binomial (same law, two parameterizations), the zero-modification layering
(route first, ``c * G'`` wrapper second), numerical-derivative checks on
every supported family, the ``NotImplementedError`` contract outside the
supported set, and the cross-check of Palm kappa against the 2-D
``occ_bivariate`` route on the Tower exhibit fixture. Derivation and scope:
``dev/plan-a362-pnl-punchups.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate.distributions import Frequency
from aggregate._aggregate_compute import palm_conditional_mean


# ---------------------------------------------------------------------------
# freq_pgf_prime: every supported family against a numerical derivative
# ---------------------------------------------------------------------------

# (freq_name, freq_a, freq_b, mean) per supported family. The empirical
# family carries its own count, so its mean argument is ignored.
SUPPORTED = {
    'poisson': (0, 0, 3.5),
    'binomial': (0.4, 0, 2.0),
    'negbin': (2.25, 0, 5.0),
    'geometric': (0, 0, 2.5),
    'fixed': (0, 0, 3.0),
    'bernoulli': (0, 0, 0.3),
    'empirical': (np.array([0, 1, 3]), np.array([0.4, 0.4, 0.2]), 1.0),
    'gamma': (0.5, 0, 5.0),
    'delaporte': (0.5, 0.3, 4.0),
    'ig': (0.6, 0, 4.0),
}


def _make_freq(name, zm=False, p0=np.nan):
    a, b, n = SUPPORTED[name]
    return Frequency(name, a, b, zm, p0), n


@pytest.mark.parametrize('name', sorted(SUPPORTED))
def test_pgf_prime_matches_numerical_derivative(name):
    """Central difference of ``freq_pgf`` pins each family's closed form."""
    fr, n = _make_freq(name)
    h = 1e-6
    z = np.array([0.2, 0.5, 0.9, 1.0])
    got = fr.freq_pgf_prime(n, z)
    num = (fr.freq_pgf(n, z + h) - fr.freq_pgf(n, z - h)) / (2 * h)
    np.testing.assert_allclose(got, num, rtol=1e-5, atol=1e-8)


def test_pgf_prime_at_one_is_the_mean():
    """``P_N'(1) = E[N]``, the first factorial moment, on every family.

    The empirical family carries its own count, so the expected value is its
    own mean rather than the ``n`` passed in.
    """
    for name in sorted(SUPPORTED):
        fr, n = _make_freq(name)
        if name == 'empirical':
            expect = float(np.sum(fr.freq_a * fr.freq_b))
        else:
            expect = n
        got = fr.freq_pgf_prime(n, np.array([1.0]))
        assert np.real(got[0]) == pytest.approx(expect, rel=1e-9), name


def test_route_agreement_negbin_vs_gamma_mixed():
    """The (a, b, 0) identity and the gamma-mixed closed form agree.

    ``negbin`` (variance multiplier ``1 + n nu^2``) and ``gamma`` (mixing CV
    ``nu``) are the same law in two parameterizations; the first takes the
    Panjer route, the second the closed-form mixing route, so their
    agreement checks both implementations at once.
    """
    n, nu = 5.0, 0.5
    negbin = Frequency('negbin', 1 + n * nu ** 2, 0, False, np.nan)
    gamma = Frequency('gamma', nu, 0, False, np.nan)
    # real points inside the disc and complex points on the unit circle
    z = np.concatenate([np.linspace(0.0, 1.0, 5),
                        np.exp(2j * np.pi * np.linspace(0.05, 0.45, 4))])
    np.testing.assert_allclose(negbin.freq_pgf(n, z), gamma.freq_pgf(n, z),
                               rtol=1e-10)
    np.testing.assert_allclose(negbin.freq_pgf_prime(n, z),
                               gamma.freq_pgf_prime(n, z), rtol=1e-10)


def test_zero_modified_poisson_layering():
    """ZM wraps the route: derivative is ``c * G'`` of the base pgf.

    The (a, b, 0) identity holds for the base pgf, not the zero-modified
    one, so the layering order (route first, wrapper second) is what a
    numerical derivative of the *wrapped* pgf pins.
    """
    n = 3.0
    fr = Frequency('poisson', 0, 0, True, 0.5)
    h = 1e-6
    z = np.array([0.2, 0.6, 0.95])
    got = fr.freq_pgf_prime(n, z)
    num = (fr.freq_pgf(n, z + h) - fr.freq_pgf(n, z - h)) / (2 * h)
    np.testing.assert_allclose(got, num, rtol=1e-5)
    # and explicitly: c * base derivative, base = n * exp(n(z-1))
    c = fr._zm_weight(n)
    np.testing.assert_allclose(got, c * n * np.exp(n * (z - 1)), rtol=1e-9)


@pytest.mark.parametrize('name,a,b', [
    ('logarithmic', 0, 0, ),
    ('neymana', 2.0, 0),
    ('pascal', 1.5, 2.0),
])
def test_unsupported_families_raise(name, a, b):
    """Outside the supported set the contract is ``NotImplementedError``.

    ``logarithmic`` is the load-bearing case: it *stores* ``panjer_ab`` but
    is (a, b, 1) (the recursion starts at ``k = 2``), so the (a, b, 0)
    route must not fire for it.
    """
    fr = Frequency(name, a, b, False, np.nan)
    with pytest.raises(NotImplementedError, match='freq_pgf_prime'):
        fr.freq_pgf_prime(3.0, np.array([0.5]))


# ---------------------------------------------------------------------------
# palm_conditional_mean / Aggregate.palm_kappa
# ---------------------------------------------------------------------------

def test_brute_force_tiny_discrete_compound():
    """Exact enumeration of a 3-atom severity under an empirical count.

    ``c`` is a per-claim layer (2 xs 2), ``n_fn = x - c(x)`` the retained
    claim. Enumerating every (count, claim tuple) outcome gives the exact
    joint of ``(S_c, S_n)``; the kernel must reproduce ``E[S_c | S_n = s]``
    to FFT precision on every attained ``s``.
    """
    counts = {0: 0.3, 1: 0.5, 2: 0.2}
    sev = {1.0: 0.5, 2.0: 0.3, 4.0: 0.2}

    def c_fn(x):
        return np.minimum(np.maximum(np.asarray(x, dtype=float) - 2.0, 0.0),
                          2.0)

    def n_fn(x):
        return np.asarray(x, dtype=float) - c_fn(x)

    # exact joint by enumeration (i.i.d. draws, counts up to 2)
    num = {}   # s -> E[S_c ; S_n = s]
    den = {}   # s -> P(S_n = s)
    for k, pk in counts.items():
        if k == 0:
            outcomes = [((), pk)]
        elif k == 1:
            outcomes = [((x,), pk * p) for x, p in sev.items()]
        else:
            outcomes = [((x1, x2), pk * p1 * p2)
                        for x1, p1 in sev.items()
                        for x2, p2 in sev.items()]
        for xs_draw, p in outcomes:
            s_c = float(sum(float(c_fn(x)) for x in xs_draw))
            s_n = float(sum(float(n_fn(x)) for x in xs_draw))
            num[s_n] = num.get(s_n, 0.0) + s_c * p
            den[s_n] = den.get(s_n, 0.0) + p

    a = build('agg PalmTiny dfreq [0 1 2] [0.3 0.5 0.2] '
              'dsev [1 2 4] [0.5 0.3 0.2]')
    kappa, f_n = a.palm_kappa(c_fn, n_fn)
    assert a.bs == 1.0
    for s, mass in den.items():
        i = int(round(s / a.bs))
        assert f_n[i] == pytest.approx(mass, abs=1e-12)
        assert kappa[i] == pytest.approx(num[s] / mass, abs=1e-9)
    # buckets with no attained s carry (numerically) no conditioning mass
    attained = {int(round(s)) for s in den}
    others = [i for i in range(len(f_n)) if i not in attained]
    assert np.abs(f_n[others]).max() < 1e-12


def test_panjer_self_check_kappa_is_identity():
    """``c = n_fn = id``: conditioning a book on itself gives ``kappa(s) = s``.

    The identity is then the integral form of the Panjer recursion; this is
    the closing-row anchor the waterfall tests pin at the P&L level.
    """
    a = build('agg PalmSelf 4 claims sev lognorm 10 cv 1 poisson')
    kappa, f_n = a.palm_kappa(lambda x: x)
    ok = f_n > 1e-9
    assert ok.sum() > 100
    np.testing.assert_allclose(kappa[ok], a.xs[ok], rtol=1e-6, atol=1e-6)


def test_kernel_footing_by_linearity():
    """Per-claim targets that sum to the conditioning subject foot to ``s``.

    ``c(x) + n_fn(x) = x`` conditioning on the gross: the two kappa vectors
    must sum to the identity, cell by cell, because conditional means add.
    """
    a = build('agg PalmFoot 3 claims sev lognorm 5 cv 0.75 '
              'mixed gamma 0.4')

    def c_fn(x):
        x = np.asarray(x, dtype=float)
        return np.minimum(np.maximum(x - 8.0, 0.0), 10.0)

    kappa_c, f_n = a.palm_kappa(c_fn)
    kappa_r, _ = a.palm_kappa(lambda x: np.asarray(x, float) - c_fn(x))
    ok = f_n > 1e-9
    np.testing.assert_allclose(kappa_c[ok] + kappa_r[ok], a.xs[ok],
                               rtol=1e-6, atol=1e-6)


def test_palm_agrees_with_occ_bivariate_on_tower():
    """Palm kappa vs the 2-D route at the net 1-in-100, on the Tower engine.

    The Tower exhibit fixture's diversified column is served off the
    ``occ_bivariate`` (net, ceded) joint on a budget-sized grid; the Palm
    route computes the same conditional mean by 1-D FFT on the fine engine
    grid. Observed gap at implementation (2026-09-29): 1.4% relative at the
    net 0.99 quantile (Palm 521.8 vs 2-D 514.6), all of it the joint's
    budget-grid coarseness; the 2% tolerance absorbs that interpolation,
    not the identity.
    """
    pn = build('xpnl EX.Tower 1000 prem less agg EX.TowerE 1000 prem '
               'at 70% lr sev lognorm 100 cv 2 '
               'occurrence ceded to 500 xs 500 deposit 100 poisson')
    engine = pn.engine
    kappa, f_n = engine.palm_kappa(engine.occ_ceder, engine.occ_netter)
    # net occurrence aggregate 1-in-100 (payoff orientation: high loss)
    cdf = np.cumsum(f_n)
    s_star_idx = int(np.searchsorted(cdf, 0.99))
    s_star = float(engine.xs[s_star_idx])
    # the 2-D route: E[ceded | net = s] read off the (net, ceded) joint
    bv = engine.occ_joint(views=('net', 'ceded'))
    den2 = np.asarray(bv.density)
    xs_net = np.asarray(bv.axis_xs[0], dtype=float)
    xs_ceded = np.asarray(bv.axis_xs[1], dtype=float)
    j = int(np.argmin(np.abs(xs_net - s_star)))
    row = den2[j, :]
    assert row.sum() > 0
    kappa_2d = float((row @ xs_ceded) / row.sum())
    assert kappa[s_star_idx] == pytest.approx(kappa_2d, rel=0.02)


def test_palm_kappa_unsupported_frequency_raises():
    """A build whose family has no pgf derivative surfaces the contract."""
    a = build('agg PalmNoPrime 4 claims dsev [1 2 3] logarithmic')
    with pytest.raises(NotImplementedError, match='freq_pgf_prime'):
        a.palm_kappa(lambda x: x)
