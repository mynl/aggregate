"""Exact pointwise functions of an ELT severity ([ELT-Severity-Functions]).

The event severity of an ELT6 table without ``no2u`` answers ``sf``, ``cdf``,
``isf``, ``ppf``, ``pdf`` and ``stats`` exactly through eltq's one-row
``tail_moments_arrays`` and ``density_arrays``, and computes nothing until one
is called. Each is checked against the SciPy mixture of the rows, classified
as eltq classifies them: an atom at the mean when the SD, the mean or the
maximum is degenerate, a Bernoulli two-point law on ``{0, M}`` when
:math:`\\sigma^2 \\ge \\mu (M - \\mu)`, and a beta on ``[0, M]`` otherwise,
with :math:`\\sigma = \\sigma_I + \\sigma_C`. Plan:
``dev/done/plan-elt-severity-functions.md``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss
from scipy.optimize import brentq

from aggregate import _loss_table, _severity, build, oep, qd
from aggregate._loss_table import _check_rows
from aggregate.charts import chart_approximation_tails

#: ``EventLossTable.SecondaryUncertainty`` of ``library.agg``, rows inline.
SIX_ROWS = [[1, 0.1, 5, 50, 4, 2], [2, 0.08, 12, 120, 8, 5],
            [3, 0.05, 25, 250, 15, 10], [4, 0.03, 60, 400, 30, 20],
            [5, 0.02, 120, 800, 50, 40], [6, 0.01, 250, 1500, 100, 80]]
#: An atom at 50, a Bernoulli row on {0, 100} (sd 60 above sqrt(30 * 70))
#: and a beta on [0, 40], so the steps at 50 and 100 sit above the beta.
ATOMS = 'agg SF.Atoms elt [[1 .1 50 100 0 0] [2 .05 30 100 60 0] [3 .05 20 40 5 0]]'
COMONOTONE = ('agg SF.Comonotone elt unit A [[1 .1 50 100 0 60]] '
              'elt unit B [[1 .1 20 100 0 60]]')
METHODS = ('sf', 'cdf', 'isf', 'ppf', 'pdf', 'stats')
RETURN_PERIODS = (5, 10, 25, 100, 1_000, 5_000)


@pytest.fixture
def eltq():
    """The eltq package with ``density_arrays``, or skip the test."""
    return pytest.importorskip('eltq', minversion='2.15')


@pytest.fixture
def burke(monkeypatch):
    """The pended Burke total rule in place of the roll-up."""
    monkeypatch.setattr(_loss_table, 'SHARED_EVENT_RULE', 'burke')


def _program(name, rows):
    """An ELT6 program with the rows inline."""
    body = ' '.join('[' + ' '.join(repr(float(v)) if i else str(int(v))
                                   for i, v in enumerate(r)) + ']'
                    for r in rows)
    return f'agg {name} elt [{body}]'


def _quiet_build(program, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build(program, **kwargs)


def _synthetic(seed, n, atoms=True):
    """A seeded synthetic ELT6 table: beta rows, and atom and Bernoulli rows
    when ``atoms``, with moderate beta parameters."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(1e-4, 0.05, n)
    m = np.exp(rng.uniform(np.log(20), np.log(5_000), n))
    mu = m * rng.uniform(0.02, 0.6, n)
    # sd as a share of the Bernoulli bound sqrt(mu (M - mu))
    share = rng.uniform(0.15, 0.9, n)
    if atoms:
        kind = rng.choice(3, n, p=[0.9, 0.05, 0.05])
        share = np.where(kind == 1, 0.0, np.where(kind == 2, 1.5, share))
    sd = share * np.sqrt(mu * (m - mu))
    split = rng.uniform(0, 1, n)
    return pd.DataFrame({'id': np.arange(1, n + 1), 'rate': rate, 'mean': mu,
                         'max': m, 'sdi': sd * split, 'sdc': sd * (1 - split)})


class Mixture:
    """The SciPy oracle: the rate-weighted mixture of the rows' laws."""

    def __init__(self, rows):
        rows = np.asarray(rows, dtype=float)
        self.rate, self.m, self.mu = rows[:, 1], rows[:, 3], rows[:, 2]
        sd = rows[:, 4] + rows[:, 5]
        self.total = self.rate.sum()
        degenerate = (self.m == 0) | (sd == 0) | (self.mu == 0) | (self.mu == self.m)
        with np.errstate(divide='ignore', invalid='ignore'):
            bernoulli = ~degenerate & ((sd / self.mu) ** 2
                                       >= (self.m - self.mu) / self.mu)
        self.atom, self.two = degenerate, bernoulli
        self.beta = ~degenerate & ~bernoulli
        y = self.mu[self.beta] / self.m[self.beta]
        v = (sd[self.beta] / self.m[self.beta]) ** 2
        nu = y * (1 - y) / v - 1
        self.a, self.b = y * nu, (1 - y) * nu

    def sf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))[:, None]
        r = self.rate
        out = (r[self.atom] * (self.mu[self.atom] > x)).sum(axis=1)
        p_top = self.mu[self.two] / self.m[self.two]
        out = out + (r[self.two] * p_top * (x < self.m[self.two])
                     + r[self.two] * (1 - p_top) * (x < 0)).sum(axis=1)
        mb = self.m[self.beta]
        out = out + (r[self.beta] * ss.beta.sf(x / mb, self.a, self.b)).sum(axis=1)
        return out / self.total

    def pdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))[:, None]
        mb = self.m[self.beta]
        dens = self.rate[self.beta] * ss.beta.pdf(x / mb, self.a, self.b) / mb
        return dens.sum(axis=1) / self.total


def _mixture_of(program):
    """The oracle of a one-unit program with its rows inline."""
    body = program[program.index('[['):]
    rows = [[float(v) for v in r.split()]
            for r in body.strip('[]').split('] [')]
    return Mixture(rows)


# ----------------------------------------------------------------------
# The law against SciPy
# ----------------------------------------------------------------------

def test_exact_sf_six_rows(eltq):
    fz = build(_program('SF.Six', SIX_ROWS), update=False).sevs[0].fz
    x = np.array([0.5, 1, 2.5, 5, 10, 16.5, 40, 49.9, 75, 120, 190.9, 300,
                  506.4, 800, 1000, 1400])
    np.testing.assert_allclose(fz.sf(x), Mixture(SIX_ROWS).sf(x),
                               rtol=1e-13, atol=0)


def test_exact_sf_synthetic(eltq, tmp_path):
    rows = _synthetic(20261107, 1_000)
    path = tmp_path / 'synthetic.csv'
    rows.to_csv(path, index=False)
    fz = build(f'agg SF.Synthetic elt "{path.as_posix()}"', update=False).sevs[0].fz
    mix = Mixture(rows.to_numpy())
    assert mix.atom.sum() > 20 and mix.two.sum() > 20
    x = np.exp(np.linspace(np.log(0.5), np.log(0.95 * rows['max'].max()), 60))
    np.testing.assert_allclose(fz.sf(x), mix.sf(x), rtol=1e-13, atol=0)


def test_edges(eltq):
    a = build(ATOMS, update=False)
    fz, moments = a.sevs[0].fz, a.sevs[0].model.moments
    upper = float(moments.max_support)
    assert upper == 100
    np.testing.assert_array_equal(fz.sf([-1e9, -1, -1e-300]), 1.0)
    np.testing.assert_array_equal(fz.sf([upper, 2 * upper, np.inf]), 0.0)
    assert np.isnan(fz.sf(np.nan))
    assert fz.sf(0.0) == pytest.approx(
        1 - moments.rate_at_zero / moments.total_rate, rel=1e-15)
    # right continuous: the atom at 50 is in the cdf at 50, not below it
    step = 0.1 / 0.2
    assert fz.cdf(50.0) - fz.cdf(50 * (1 - 1e-12)) == pytest.approx(step, rel=1e-12)
    assert fz.sf(50.0) == pytest.approx(_mixture_of(ATOMS).sf(50.0)[0], rel=1e-14)
    # shapes: a scalar gives a numpy scalar, an array keeps its shape
    assert np.ndim(fz.sf(10.0)) == 0
    assert fz.sf(np.ones((2, 3))).shape == (2, 3)


@pytest.mark.parametrize('which', ['six', 'synthetic'])
def test_inverse(eltq, which):
    rows = SIX_ROWS if which == 'six' else _synthetic(7, 300, atoms=False).to_numpy()
    fz = build(_program('SF.Inverse', rows), update=False).sevs[0].fz
    s = np.array([0.9, 0.5, 0.2, 0.1, 0.03, 0.01, 1e-3, 1e-4])
    x = fz.isf(s)
    np.testing.assert_allclose(fz.sf(x), s, rtol=1e-12, atol=0)
    assert fz.isf(0.0) == fz.upper
    assert fz.isf(1.0) == 0.0
    assert np.isnan(fz.isf(np.nan))
    q = np.array([0.1, 0.5, 0.9, 0.99, 0.999])
    np.testing.assert_array_equal(fz.ppf(q), fz.isf(1 - q))


def test_inverse_at_atoms(eltq):
    fz = build(ATOMS, update=False).sevs[0].fz
    # sf is (0.1 + 0.015) / 0.2 on [40, 50), 0.075 on [50, 100), 0 above
    for s, atom in [(0.3, 50.0), (0.5, 50.0), (0.05, 100.0)]:
        x = fz.isf(s)
        assert x == pytest.approx(atom, rel=1e-14)
        assert fz.sf(x) < s


def test_density(eltq, tmp_path):
    rows = _synthetic(11, 400)
    path = tmp_path / 'density.csv'
    rows.to_csv(path, index=False)
    fz = build(f'agg SF.Density elt "{path.as_posix()}"', update=False).sevs[0].fz
    mix = Mixture(rows.to_numpy())
    x = np.exp(np.linspace(np.log(0.7), np.log(0.97 * rows['max'].max()), 80))
    # away from poles: no point within 1e-6 relative of any row's maximum
    far = np.abs(x[:, None] / rows['max'].to_numpy()[None, :] - 1).min(axis=1) > 1e-6
    np.testing.assert_allclose(fz.pdf(x[far]), mix.pdf(x[far]), rtol=1e-12, atol=0)
    six = build(_program('SF.SixPdf', SIX_ROWS), update=False).sevs[0].fz
    y = np.array([0.5, 3, 20, 100, 400, 1200])
    np.testing.assert_allclose(six.pdf(y), Mixture(SIX_ROWS).pdf(y), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(six.pdf([-5, 0, 1500, 2000]), 0.0)
    assert np.isnan(six.pdf(np.nan))


def test_stats(eltq):
    fz = build(_program('SF.Stats', SIX_ROWS), update=False).sevs[0].fz
    m1, m2, m3 = fz.model.moments.severity_moments
    mean, var, skew, kurt = fz.stats('mvsk')
    assert mean == m1
    assert var == pytest.approx(m2 - m1 ** 2, rel=1e-15)
    assert skew == pytest.approx((m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5,
                                 rel=1e-12)
    assert np.isnan(kurt)
    assert fz.stats('m') == m1
    assert fz.stats() == (mean, var)


# ----------------------------------------------------------------------
# Shared events
# ----------------------------------------------------------------------

@pytest.mark.parametrize('method', METHODS)
def test_burke_refuses(eltq, burke, method):
    fz = build(COMONOTONE, update=False).sevs[0].fz
    args = () if method == 'stats' else (0.5,)
    with pytest.raises(NotImplementedError, match=r'\[ELT-Burke-Total\]'):
        getattr(fz, method)(*args)
    assert fz.support() == (0.0, 200.0)


def test_shared_rollup_matches_its_rolled_row(eltq):
    shared = build(COMONOTONE, update=False).sevs[0].fz
    cols = shared.model.columns()
    assert len(cols['rate']) == 1
    row = [[1] + [float(cols[k][0]) for k in
                  ('rate', 'mean_loss', 'max_loss', 'sd_indep', 'sd_corr')]]
    one = build(_program('SF.OneRow', row), update=False).sevs[0].fz
    x = np.array([0.0, 10, 50, 70, 120, 190])
    s = np.array([0.9, 0.5, 0.1, 0.01])
    for method, arg in [('sf', x), ('cdf', x), ('pdf', x), ('isf', s), ('ppf', 1 - s)]:
        np.testing.assert_array_equal(getattr(shared, method)(arg),
                                      getattr(one, method)(arg))
    assert shared.stats('mvs') == one.stats('mvs')


# ----------------------------------------------------------------------
# Nothing on the build path calls the law
# ----------------------------------------------------------------------

def test_no_overhead(eltq, monkeypatch):
    calls = []

    def spy(name):
        def refuse(self, *args, **kwargs):
            calls.append(name)
            raise NotImplementedError(f'_EltLaw.{name} called')
        return refuse

    for name in METHODS:
        monkeypatch.setattr(_severity._EltLaw, name, spy(name))
    for program in (_program('SF.NoCalls', SIX_ROWS), ATOMS, COMONOTONE):
        a = _quiet_build(program)
        a.update(bs=1 / 8, log2=14)
        qd(a)
        str(a.info)
        a.validation_df
        _check_rows(a)
    assert calls == []


# ----------------------------------------------------------------------
# oep and the other callers
# ----------------------------------------------------------------------

def _brentq_loss(mix, s):
    return brentq(lambda x: mix.sf(x)[0] - s, 0, mix.m.max(),
                  xtol=1e-300, rtol=1e-15, maxiter=500)


def test_oep(eltq):
    a = build(_program('SF.Oep', SIX_ROWS), update=False)
    mix = Mixture(SIX_ROWS)
    lam = a.n
    assert lam == pytest.approx(mix.total, rel=1e-15)
    p = np.array([1 / t for t in RETURN_PERIODS])
    df = oep(a, p)
    s = -np.log1p(-p) / lam
    expected = [_brentq_loss(mix, v) for v in s]
    np.testing.assert_allclose(df['loss'], expected, rtol=1e-10)
    np.testing.assert_allclose(df['oep'], p, rtol=1e-12)
    np.testing.assert_allclose(df['annual_return_period'], RETURN_PERIODS, rtol=1e-12)
    # the acceptance values
    np.testing.assert_allclose(oep(a, [0.1, 0.01, 0.001])['loss'],
                               [16.531, 190.864, 506.394], atol=5e-4)
    doubled = oep(a, p, freq=2 * lam)
    s2 = -np.log1p(-p) / (2 * lam)
    np.testing.assert_allclose(doubled['loss'], [_brentq_loss(mix, v) for v in s2],
                               rtol=1e-10)
    # the grid of the updated object reads within half a bucket
    a.update()
    grid = np.array([a.q_sev(1 - v) for v in s])
    assert np.all(np.abs(grid - df['loss'].to_numpy()) <= a.bs / 2 * (1 + 1e-9))


def test_oep_at_atoms(eltq):
    a = build(ATOMS, update=False)
    lam = a.n
    # sf is 0.575 on [40, 50), 0.075 on [50, 100): s = 0.3 lands on the step at 50
    p = -np.expm1(-lam * np.array([0.3, 0.05]))
    df = oep(a, p)
    np.testing.assert_allclose(df['loss'], [50, 100], rtol=1e-14)
    assert np.all(df['oep'].to_numpy() < p)
    assert df['S_sev'].to_numpy()[0] == pytest.approx(0.075, rel=1e-13)


def test_oep_net_of_an_occurrence_program(eltq):
    """The six-row table under 25 xs 25, 50 xs 50 and 90% of 200 xs 100: the
    net point is the program's net function of the gross point, and the grid
    of the updated object agrees to within a bucket."""
    program = 'occurrence net of 25 xs 25 and 50 xs 50 and 90% po 200 xs 100'
    gross = build(_program('SF.Gross', SIX_ROWS), update=False)
    net = _quiet_build(f"{_program('SF.Net', SIX_ROWS)} {program}")
    p = np.array([1 / t for t in RETURN_PERIODS])
    x = oep(gross, p)['loss'].to_numpy()
    df = oep(net, p)
    expected = np.select([x <= 25, x <= 100, x <= 300],
                         [x, 25.0, 25 + 0.1 * (x - 100)], x - 255)
    np.testing.assert_allclose(df['loss'], expected, rtol=1e-13)
    np.testing.assert_allclose(oep(net, 0.01)['loss'], [34.0864], atol=5e-5)
    # 1 in 25 lands in the stretch [25, 100] where the net loss is 25
    flat = (x >= 25) & (x <= 100)
    assert flat.any()
    np.testing.assert_allclose(df['S_sev'][flat], gross.sev.sf(100.0), rtol=1e-13)
    assert (df['oep'][flat] < p[flat]).all()
    sev = net.sev_density_df
    achieved = -np.expm1(-net.n * sev.S_sev.to_numpy())
    grid = np.array([sev.loss.to_numpy()[achieved <= q][0] for q in p])
    assert np.all(np.abs(grid - df['loss'].to_numpy()) <= net.bs)


def test_callers(eltq):
    a = _quiet_build(_program('SF.Callers', SIX_ROWS))
    sev = a.sev
    x = np.array([1.0, 10, 100, 1000])
    for f in (sev.cdf, sev.sf, sev.pdf):
        assert np.all(np.isfinite(f(x)))
    np.testing.assert_allclose(sev.sf(x), Mixture(SIX_ROWS).sf(x), rtol=1e-13)
    assert sev.ppf(0.99) == pytest.approx(sev.isf(0.01), rel=1e-14)
    assert sev.sf(sev.isf(0.01)) == pytest.approx(0.01, rel=1e-12)
    sea = a.severity_error_analysis()
    numeric = sea.select_dtypes('number').drop(columns=['limit'])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    doc = chart_approximation_tails(a)
    implied = [s for s in doc.series if s.name == 'Implied tail']
    assert implied and np.isfinite(np.asarray(implied[0].y, dtype=float)).any()


def test_mean_only_unchanged(eltq):
    a = build('agg SF.No2u elt no2u [[1 0.1 5 50 4 2] [2 0.08 12 120 8 5] '
              '[3 0.05 25 250 15 10] [4 0.03 60 400 30 20] '
              '[5 0.02 120 800 50 40] [6 0.01 250 1500 100 80]]', update=False)
    assert type(a.sevs[0]).__name__ == 'SeverityDHistogram'
    df = oep(a, [0.1, 0.01, 0.001])
    # 2.0.0's frame, captured 2026-11-07 before the change
    expected = pd.DataFrame({
        'loss': [25.0, 120.0, 250.0],
        'S_sev': [0.206896551724138, 0.034482758620689724, 0.0],
        'F_sev': [0.793103448275862, 0.9655172413793103, 1.0],
        'oep': [0.05823546641575132, 0.009950166250831968, 0.0],
        'occurrence_return_period': [16.666666666666657, 99.99999999999979, np.inf],
        'annual_return_period': [17.17166636669237, 100.50083333194424, np.inf],
    }, index=pd.Index([0.1, 0.01, 0.001], name='p'))
    pd.testing.assert_frame_equal(df, expected, rtol=1e-15)
