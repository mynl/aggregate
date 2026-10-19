"""Gold-standard tests for allocation and pricing invariants.

``[Allocation-Bounds-Gold-Standard-Tests]``: extend the engine's gold-standard
style (closed forms, no captured snapshots) past the FFT into allocation.

Closed forms
------------
**Poisson thinning.** Units ``A`` and ``B`` are compound Poisson with the same
severity and claim counts ``lambda_A`` and ``lambda_B``. Their sum is compound
Poisson with ``lambda = lambda_A + lambda_B``, and given the total ``X = x``
each claim belongs to ``A`` independently with probability ``lambda_A /
lambda``, so ``E[A | X = x] = x lambda_A / lambda``. Every natural allocation
reads ``E[A | X]`` weighted by the distortion, so ``A`` receives exactly the
fraction ``lambda_A / lambda`` of the loss, the premium and the margin, under
**any** distortion.

**Gamma shapes.** ``A ~ Gamma(alpha_A, theta)`` and ``B ~ Gamma(alpha_B,
theta)`` independent with a common scale give ``E[A | A + B = x] = x alpha_A
/ (alpha_A + alpha_B)``, the same identity reached through a continuous
severity rather than through counting. Discretization limits it to about 1e-7.

**Deterministic unit.** A unit that is a constant ``c`` has ``E[C | X] = c``,
so with assets above every outcome (no default) it is allocated premium
exactly ``c`` and margin exactly zero, whatever the distortion.

**Comonotonic units.** Units ``X_i = w_i X`` are functions of the total, so
``E[X_i | X_total = t] = t w_i / sum(w)`` exactly, and every layer of the
total splits in the same proportion. Each unit therefore receives exactly the
fraction ``w_i / sum(w)`` of L, P, M **and** Q, under any distortion and at any
asset level. Built with ``Portfolio.create_from_sample`` from a fixed sample.

**Identical units.** Independent, identically distributed units are
exchangeable, so each of ``n`` receives exactly ``1 / n`` of L, P, M and Q.

The Q comparisons are absolute, scaled to the asset level, rather than ratios:
a distortion can make total capital exactly zero (``TVaR_0.9`` at ``a =
q(0.9)`` distorts every layer below ``a`` to ``gS = 1``, so ``P = a``), and
then each unit's Q is zero to rounding and a ratio is meaningless.

Invariants
----------
Property tests draw small random discrete programs with ``hypothesis`` and
check what must hold for every distribution: ``q`` is nondecreasing, TVaR is
at least VaR, unit premiums add up to the total, and a cession leaves a net
that is first-order dominated by the gross and priced no higher than it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aggregate import Portfolio, build
from aggregate.spectral import Distortion

DISTORTIONS = [
    Distortion('ph', 0.6),
    Distortion('wang', 0.4),
    Distortion('tvar', 0.9),
    Distortion('dual', 1.8),
]
DIST_IDS = [str(d) for d in DISTORTIONS]
COMONOTONIC_WEIGHTS = (1, 2, 5)


@pytest.fixture(scope='module')
def thinning_port():
    """Two compound Poisson units sharing a severity; lambda_A share 1/4."""
    return build('port GS.Thin '
                 'agg A 2 claims sev gamma 10 cv 1 poisson '
                 'agg B 6 claims sev gamma 10 cv 1 poisson',
                 log2=14, bs=1 / 4)


@pytest.fixture(scope='module')
def gamma_shape_port():
    """Gamma(1, 10) and Gamma(3, 10): common scale, alpha_A share 1/4."""
    return build('port GS.Shape '
                 'agg A 1 claim sev gamma 10 cv 1 fixed '
                 f'agg B 1 claim sev gamma 30 cv {1 / np.sqrt(3)} fixed',
                 log2=14, bs=1 / 16)


@pytest.fixture(scope='module')
def constant_unit_port():
    """Bounded discrete units plus a constant 5; support ends at 12."""
    return build('port GS.Const '
                 'agg A 1 claim dsev [1 2 4] fixed '
                 'agg B 1 claim dsev [0 3] fixed '
                 'agg C 1 claim dsev [5] fixed',
                 log2=6, bs=1)


def _unit_rows(result, dist):
    """The per-unit pricing rows for one distortion, total last."""
    return result.df.xs(str(dist), level='distortion')


@pytest.mark.parametrize('dist', DISTORTIONS, ids=DIST_IDS)
def test_poisson_thinning_allocates_by_claim_count(thinning_port, dist):
    """Loss, premium and margin all split exactly 1/4 to unit A."""
    rows = _unit_rows(thinning_port.price(0.999, dist), dist)
    for stat in ('L', 'P', 'M'):
        assert rows.loc['A', stat] / rows.loc['total', stat] == \
            pytest.approx(0.25, rel=1e-10), stat


def test_poisson_thinning_conditional_expectation(thinning_port):
    """``exeqa_A = x / 4`` pointwise wherever the total has material mass."""
    df = thinning_port.density_df
    mask = (df.p_total > 1e-10) & (df.loss > 0)
    np.testing.assert_allclose(df.exeqa_A[mask], df.loss[mask] / 4, rtol=1e-6)


@pytest.mark.parametrize('dist', DISTORTIONS, ids=DIST_IDS)
def test_gamma_common_scale_allocates_by_shape(gamma_shape_port, dist):
    """The same 1/4 identity through a continuous severity."""
    rows = _unit_rows(gamma_shape_port.price(0.999, dist), dist)
    for stat in ('L', 'P'):
        assert rows.loc['A', stat] / rows.loc['total', stat] == \
            pytest.approx(0.25, rel=1e-5), stat


@pytest.mark.parametrize('dist', DISTORTIONS, ids=DIST_IDS)
def test_constant_unit_priced_at_its_value(constant_unit_port, dist):
    """With assets at the top of the support, ``P_C = 5`` and ``M_C = 0``."""
    top = float(constant_unit_port.apply_distortion(
        dist, allocation='linear').loss.iloc[-1])
    assert top == 12.0
    rows = _unit_rows(constant_unit_port.price(top, dist), dist)
    assert rows.loc['C', 'P'] == pytest.approx(5.0, rel=1e-12)
    assert rows.loc['C', 'M'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('dist', DISTORTIONS, ids=DIST_IDS)
def test_unit_allocations_sum_to_total(thinning_port, dist):
    """Additivity: unit L, P and M sum to the total."""
    rows = _unit_rows(thinning_port.price(0.999, dist), dist)
    units = rows.drop('total')
    for stat in ('L', 'P', 'M'):
        assert units[stat].sum() == pytest.approx(rows.loc['total', stat],
                                                  rel=1e-11), stat


@pytest.fixture(scope='module')
def comonotonic_port():
    """Units ``w_i X`` for ``w = (1, 2, 5)``, ``X`` uniform on ``1, ..., 40``."""
    x = np.arange(1, 41, dtype=float)
    sample = pd.DataFrame({f'U{i}': w * x
                           for i, w in enumerate(COMONOTONIC_WEIGHTS)})
    return Portfolio.create_from_sample('GS.Comonotonic', sample, bs=1, log2=10)


@pytest.fixture(scope='module')
def identical_port():
    """Three independent copies of one compound Poisson unit."""
    unit = '3 claims sev gamma 10 cv 1 poisson'
    return build(f'port GS.Identical agg A {unit} agg B {unit} agg C {unit}',
                 log2=13, bs=1 / 4)


def _assert_shares(rows, shares, rel):
    """Each unit's L, P, M and Q equal its share of the total.

    Absolute tolerance scaled to the asset level, so a total that is zero to
    rounding (see the module docstring on Q) is checked rather than divided by.
    """
    abs_tol = 1e-12 * float(rows.loc['total', 'a'])
    for unit, share in shares.items():
        for stat in ('L', 'P', 'M', 'Q'):
            assert rows.loc[unit, stat] == pytest.approx(
                share * rows.loc['total', stat], rel=rel, abs=abs_tol), \
                (unit, stat)


@pytest.mark.parametrize('p', [0.9, 0.95])
@pytest.mark.parametrize('dist', DISTORTIONS, ids=DIST_IDS)
def test_comonotonic_units_allocate_by_weight(comonotonic_port, dist, p):
    """``w_i X`` receives exactly ``w_i / sum(w)`` of L, P, M and Q."""
    rows = comonotonic_port.pricing_at(dist, p=p, allocation='linear')
    total_w = sum(COMONOTONIC_WEIGHTS)
    _assert_shares(rows, {f'U{i}': w / total_w
                          for i, w in enumerate(COMONOTONIC_WEIGHTS)},
                   rel=1e-12)


@pytest.mark.parametrize('dist', DISTORTIONS, ids=DIST_IDS)
def test_identical_units_share_equally(identical_port, dist):
    """Exchangeable units each receive exactly 1/3 of L, P, M and Q."""
    rows = identical_port.pricing_at(dist, p=0.99, allocation='linear')
    _assert_shares(rows, {u: 1 / 3 for u in 'ABC'}, rel=1e-10)


# ----------------------------------------------------------------------------
# Property tests over random small discrete programs
# ----------------------------------------------------------------------------

#: Each example is an FFT build on a small grid (a few ms), so a modest example
#: count keeps the module fast while still exploring the shape space.
PROPERTY_SETTINGS = settings(
    max_examples=25, deadline=None, derandomize=True,
    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def dsev_clause(draw, max_outcome=12):
    """A ``dsev [x...] [p...]`` clause with distinct positive integer outcomes."""
    xs = draw(st.lists(st.integers(1, max_outcome), min_size=1, max_size=4,
                       unique=True))
    ws = draw(st.lists(st.integers(1, 9), min_size=len(xs), max_size=len(xs)))
    total = sum(ws)
    ps = [w / total for w in ws]
    return (f'dsev [{" ".join(str(x) for x in xs)}] '
            f'[{" ".join(repr(p) for p in ps)}]')


claim_counts = st.sampled_from([0.5, 1, 2, 3])
distortions = st.builds(
    lambda kind, s: {'ph': Distortion('ph', 0.4 + 0.5 * s),
                     'wang': Distortion('wang', 0.1 + 0.6 * s),
                     'tvar': Distortion('tvar', 0.2 + 0.7 * s),
                     'dual': Distortion('dual', 1.2 + 1.5 * s)}[kind],
    st.sampled_from(['ph', 'wang', 'tvar', 'dual']),
    st.floats(0, 1))


def _build_agg(sev, n, extra=''):
    """A compound Poisson aggregate on a grid comfortably wider than its tail."""
    return build(f'agg GS.Prop {n} claims {sev} {extra} poisson',
                 log2=10, bs=1)


@PROPERTY_SETTINGS
@given(sev=dsev_clause(), n=claim_counts,
       ps=st.lists(st.floats(0.01, 0.999), min_size=2, max_size=8))
def test_quantile_is_monotone(sev, n, ps):
    """``q`` is nondecreasing in ``p``."""
    a = _build_agg(sev, n)
    qs = [a.q(p) for p in sorted(ps)]
    assert all(q1 <= q2 for q1, q2 in zip(qs, qs[1:]))


@PROPERTY_SETTINGS
@given(sev=dsev_clause(), n=claim_counts, p=st.floats(0.01, 0.999))
def test_tvar_at_least_var(sev, n, p):
    """``TVaR_p >= VaR_p``, up to rounding in the tail sum."""
    a = _build_agg(sev, n)
    assert a.tvar(p) >= a.q(p) - 1e-9 * max(1.0, abs(a.q(p)))


@PROPERTY_SETTINGS
@given(sev_a=dsev_clause(), sev_b=dsev_clause(), n_a=claim_counts,
       n_b=claim_counts, dist=distortions)
def test_unit_premiums_sum_to_total(sev_a, sev_b, n_a, n_b, dist):
    """Additivity of the natural allocation, for any units and distortion."""
    port = build(f'port GS.PropPort agg A {n_a} claims {sev_a} poisson '
                 f'agg B {n_b} claims {sev_b} poisson', log2=10, bs=1)
    rows = port.price(0.99, dist).df.xs(str(dist), level='distortion')
    total = rows.loc['total', 'P']
    assert rows.drop('total')['P'].sum() == pytest.approx(total, rel=1e-9,
                                                          abs=1e-9)


@PROPERTY_SETTINGS
@given(sev=dsev_clause(), n=claim_counts, dist=distortions,
       attach=st.integers(0, 8), limit=st.integers(1, 8),
       share=st.sampled_from([25, 50, 100]))
def test_net_is_dominated_by_gross(sev, n, dist, attach, limit, share):
    """A cession cannot raise the loss: net ``<=`` gross in distribution.

    Net loss is pointwise at most the gross loss, so its cdf is pointwise at
    least the gross cdf, and any distortion premium of the net is at most that
    of the gross (distortion pricing is monotone).
    """
    a = _build_agg(sev, n, f'occurrence net of {share}% po {limit} xs {attach}')
    rd = a.reins_density_df
    gross, net = rd['p_agg_gross'], rd['p_agg_net_occ']
    assert np.all(net.cumsum().to_numpy() >= gross.cumsum().to_numpy() - 1e-12)
    price_g = dist.price(pd.Series(gross.to_numpy(), index=rd.index)).ask
    price_n = dist.price(pd.Series(net.to_numpy(), index=rd.index)).ask
    assert price_n <= price_g + 1e-9 * max(1.0, abs(price_g))
