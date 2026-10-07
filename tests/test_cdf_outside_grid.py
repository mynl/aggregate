"""``cdf`` / ``sf`` outside the computed grid: 0 where known, nan where not.

The contract these pin, `[Outside-Grid-Is-Nan]`
(``dev/plan-a400-v1-loose-ends.md`` phase 1): below a grid's lower edge the
answer is ``0`` only when the law provably has no mass there, and ``nan``
otherwise. "Otherwise" is a support window placed above the origin, or a signed
law whose two-sided window the sizer positioned from three moments: in both,
mass below the edge was discretized away or never computed, and the grid does
not say how much. Reporting ``0`` there would assert an absence of mass that is
not known.

Before `1.0.0a400` all four interpolators used ``fill_value='extrapolate'``
with ``kind='previous'``, and scipy's ``previous`` has no knot below the first,
so **every** below-grid query returned ``nan``, including the ordinary
non-negative case where ``0`` is simply true. The honest ``nan`` on the windowed
and signed sides was therefore right by accident, which is what these tests
remove: a future change of interpolator or scipy version cannot silently turn
either side into a fabricated number without failing here.
"""

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate.utilities import below_grid_fill

# A limit profile whose default grid windows hard, i.e. its lower edge sits well
# above the origin. Shared with tests/test_windowed_calibrate.py, which guards
# the same fixture property for a different reason.
WINDOWED_AGG = """agg OG.LimitProfile
  [10000 20000 5000] premium at [0.8 0.7 0.5] lr
  [1000 2000 5000] xs 0
  sev lognorm 50 cv 1.5
  poisson"""

WINDOWED_PORT = """port OG.WindowTest
  agg U1 500 claims sev lognorm 50 cv 1.5 poisson
  agg U2 400 claims sev lognorm 60 cv 1.4 poisson"""

PLAIN_AGG = 'agg OG.Plain 10 claims sev lognorm 100 cv 2 poisson'

PLAIN_PORT = """port OG.Plain
  agg P1 5 claims sev lognorm 100 cv 2 poisson
  agg P2 5 claims sev gamma 80 cv 1 poisson"""

# Continuous signed severity: the grid straddles 0 and its bottom is chosen by
# the three-moment sizer, so mass below it is not accounted for.
SIGNED_AGG = 'agg OG.Signed 10 claims ssev 100 * norm poisson'

# Discrete signed severity: a negative atom auto-signs.
SIGNED_DISCRETE = 'agg OG.SignedD dfreq [1] dsev [-10 0 10] [.3 .4 .3]'


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


@pytest.fixture(scope='module')
def plain_agg():
    return build(PLAIN_AGG)


@pytest.fixture(scope='module')
def plain_port():
    return build(PLAIN_PORT)


@pytest.fixture(scope='module')
def windowed_agg():
    return build(WINDOWED_AGG)


@pytest.fixture(scope='module')
def windowed_port():
    return build(WINDOWED_PORT)


def _bottom(ob):
    """The first abscissa of the computed grid, for either class."""
    xs = getattr(ob, 'xs', None)
    return float(xs[0]) if xs is not None else float(ob.density_df.loss.iloc[0])


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bottom, signed, expected', [
    (0.0, False, 0.0),        # the ordinary case: nothing below the origin
    (-0.0, False, 0.0),
    (0.0, True, np.nan),      # signed: the sizer placed the edge
    (-250.0, True, np.nan),
    (13_759.5, False, np.nan),  # windowed above the origin
    (1e-9, False, np.nan),    # even a hair above the origin is a window
])
def test_below_grid_fill_rule(bottom, signed, expected):
    got = below_grid_fill(bottom, signed)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == expected


# ---------------------------------------------------------------------------
# Known: a non-negative law on a grid starting at the origin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['plain_agg', 'plain_port'])
def test_non_negative_reports_zero_below_the_origin(fixture, request):
    ob = request.getfixturevalue(fixture)
    assert _bottom(ob) <= 0.0 and not ob._signed()
    for x in (-1e12, -1e6, -1.0, -1e-12):
        assert ob.cdf(x) == 0.0, f'cdf({x}) should be a known 0'
        assert ob.sf(x) == 1.0, f'sf({x}) should be a known 1'


@pytest.mark.parametrize('fixture', ['plain_agg', 'plain_port'])
def test_non_negative_zero_is_vectorized(fixture, request):
    """Array input must get the same fill, not a scalar-only special case."""
    ob = request.getfixturevalue(fixture)
    got = ob.cdf(np.array([-10.0, -1.0, 0.0]))
    assert got[0] == 0.0 and got[1] == 0.0
    assert got[2] > 0.0


# ---------------------------------------------------------------------------
# Not known: a window placed above the origin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['windowed_agg', 'windowed_port'])
def test_windowed_reports_nan_below_the_window(fixture, request):
    ob = request.getfixturevalue(fixture)
    lo = _bottom(ob)
    assert lo > 0.0, 'fixture must actually window, or this test is vacuous'
    for x in (0.0, 1.0, lo / 2, lo * 0.999):
        assert np.isnan(ob.cdf(x)), f'cdf({x}) must not claim an absence of mass'
        assert np.isnan(ob.sf(x)), f'sf({x}) inherits the nan'


@pytest.mark.parametrize('fixture', ['windowed_agg', 'windowed_port'])
def test_windowed_is_finite_from_the_edge_inward(fixture, request):
    """The nan stops exactly at the lower edge; inside, the law is known."""
    ob = request.getfixturevalue(fixture)
    lo = _bottom(ob)
    assert np.isfinite(ob.cdf(lo))
    assert np.isfinite(ob.cdf(lo * 1.5))


# ---------------------------------------------------------------------------
# Not known: a signed law whose grid bottom the sizer chose
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program', [SIGNED_AGG, SIGNED_DISCRETE])
def test_signed_reports_nan_below_the_grid(program):
    ob = build(program)
    assert ob._signed(), 'fixture must actually be signed'
    lo = _bottom(ob)
    for x in (lo - 1e4, lo - 1.0, lo - 1e-9):
        assert np.isnan(ob.cdf(x)), f'cdf({x}) below a signed grid is unknown'
    assert np.isfinite(ob.cdf(lo))


# ---------------------------------------------------------------------------
# Above the grid: the computed total mass, not a fabricated 1.0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['plain_agg', 'windowed_agg'])
def test_above_the_grid_is_the_computed_total_mass(fixture, request):
    ob = request.getfixturevalue(fixture)
    top = float(ob.agg_density.cumsum()[-1])
    for x in (float(ob.xs[-1]) * 2, 1e15):
        assert ob.cdf(x) == pytest.approx(top, rel=0, abs=0)
    # an unbounded law has not exhausted its mass on the grid, so this is
    # strictly below 1: the point of not filling 1.0
    assert top < 1.0


@pytest.mark.parametrize('fixture', ['plain_port', 'windowed_port'])
def test_above_the_grid_is_the_computed_total_mass_portfolio(fixture, request):
    ob = request.getfixturevalue(fixture)
    top = float(ob.density_df.F.iloc[-1])
    assert ob.cdf(float(ob.density_df.loss.iloc[-1]) * 2) == pytest.approx(
        top, rel=0, abs=0)


# ---------------------------------------------------------------------------
# The fill must not have disturbed any in-grid value
# ---------------------------------------------------------------------------

def test_in_grid_values_match_the_density_frame(plain_agg):
    """cdf at the knots is the cumulative sum, unchanged by the fill."""
    cum = plain_agg.agg_density.cumsum()
    idx = np.linspace(0, len(plain_agg.xs) - 1, 40).astype(int)
    got = plain_agg.cdf(plain_agg.xs[idx])
    assert np.allclose(got, cum[idx], rtol=0, atol=0)


def test_in_grid_values_match_the_density_frame_portfolio(plain_port):
    df = plain_port.density_df
    idx = np.linspace(0, len(df) - 1, 40).astype(int)
    got = plain_port.cdf(df.loss.to_numpy()[idx])
    assert np.allclose(got, df.F.to_numpy()[idx], rtol=0, atol=0)


@pytest.mark.parametrize('fixture', ['plain_agg', 'plain_port'])
def test_q_is_unaffected_by_the_fill(fixture, request):
    """q inverts within the grid and never extrapolates, so it sees no nan."""
    ob = request.getfixturevalue(fixture)
    for p in (0.01, 0.5, 0.99, 0.999):
        assert np.isfinite(ob.q(p))
