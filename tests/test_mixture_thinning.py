"""Severity-mixture component moments come from thinning the parent frequency.

[Mixture-Thinning-Moments]. A mixture splits each claim into a component by an
iid latent label, so the component count ``K_i`` is ``Binomial(N, w_i)`` given
the parent count ``N``. Routing the weight through the claim count instead is
correct only when the thinning of ``N`` is the same family with the mean scaled
by ``w``, which holds for Poisson and every mixed Poisson and fails elsewhere.

The spine of this module is :func:`reassembled_variance`: the component
marginals plus the closed-form covariance must rebuild the total variance
exactly, for every frequency family. See
``dev/done/plan-mixture-thinning-moments.md``.
"""

import numpy as np
import pytest

from aggregate import Underwriter
from aggregate.moments import MomentAggregator

thin = MomentAggregator.thin_moments


@pytest.fixture(scope='module')
def uw():
    """A recipe-free underwriter, so these cases cannot be perturbed by the
    shipped ``.agg`` libraries."""
    return Underwriter(update=False)


# ---------------------------------------------------------------------------
# the map itself
# ---------------------------------------------------------------------------

def _poisson_moms(n):
    return n, n * (1 + n), n * (1 + n * (3 + n))


def _binomial_moms(r, p):
    """First three non-central moments of Binomial(r, p)."""
    m1 = r * p
    m2 = r * p * (1 - p) + m1 ** 2
    m3 = r * p * (1 - p) * (1 - 2 * p) + 3 * (r * p) ** 2 * (1 - p) + m1 ** 3
    return m1, m2, m3


@pytest.mark.parametrize('w', [0.05, 0.25, 0.5, 0.9])
def test_thinning_a_poisson_gives_a_poisson(w):
    """Poisson splitting: thinning Poisson(n) by w gives Poisson(wn) exactly."""
    n = 7.5
    assert np.allclose(thin(w, *_poisson_moms(n)), _poisson_moms(w * n))


@pytest.mark.parametrize('w', [0.05, 0.25, 0.5, 0.9])
def test_thinning_a_binomial_scales_the_success_probability(w):
    """Bin(r, p) thinned by w is Bin(r, wp): the trial count is unchanged."""
    r, p = 12, 0.4
    assert np.allclose(thin(w, *_binomial_moms(r, p)), _binomial_moms(r, w * p))


def test_a_fixed_count_thins_to_a_binomial_not_to_a_smaller_fixed_count():
    """The case that motivated the change: N = n has Var 0, but K = Bin(n, w)
    does not, so a fixed frequency cannot carry its mixture weight in its mean."""
    n, w = 5, 0.3
    fixed_moms = (n, n ** 2, n ** 3)
    assert np.allclose(thin(w, *fixed_moms), _binomial_moms(n, w))
    # what the old route would have produced: a degenerate count at w * n
    degenerate = (w * n, (w * n) ** 2, (w * n) ** 3)
    assert not np.allclose(thin(w, *fixed_moms), degenerate)


def test_unit_weight_is_the_identity():
    """A limit profile carries weights of one, where the map must not bite."""
    moms = (3.0, 14.0, 90.0)
    assert np.allclose(thin(1.0, *moms), moms)


def test_zero_weight_gives_the_degenerate_zero_count():
    assert np.allclose(thin(0.0, 3.0, 14.0, 90.0), (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# the reassembly identity
# ---------------------------------------------------------------------------

# one program per frequency family, all with the same two-point severity
# mixture so the expected answers are comparable across rows
MIX = 'sev [1 10] * expon wts [0.9 0.1]'
FREQUENCIES = [
    f'agg T 3 claims {MIX} poisson',
    f'agg T 3 claims {MIX} fixed',
    f'agg T 3 claims {MIX} binomial 0.5',
    f'agg T 0.4 claims {MIX} bernoulli',
    f'agg T 3 claims {MIX} geometric',
    f'agg T 3 claims {MIX} logarithmic',
    f'agg T 3 claims {MIX} neymana 2',
    f'agg T 3 claims {MIX} pascal 0.5 2',
    f'agg T 3 claims {MIX} mixed gamma 0.5',
    f'agg T 3 claims {MIX} mixed delaporte 0.5 0.4',
    f'agg T 3 claims {MIX} mixed sichel 0.5 -0.5',
    f'agg T dfreq[1 2 6] {MIX}',
    f'agg T dfreq[3] {MIX}',
    # zero modification belongs to N, so it is applied once to the row and the
    # thinning then acts on the realized parent. Applying it per component, as
    # the count-carrying route did, shifts the mean once per mixture component.
    f'agg T 3 claims {MIX} poisson zt',
    f'agg T 3 claims {MIX} poisson zt !',
    f'agg T 3 claims {MIX} poisson zm 0.2',
]


def reassembled_variance(a):
    """Rebuild Var(A) from the component marginals plus the closed-form covariance.

    With ``nu = E[N]``, ``v = Var(N)`` and component severity means ``mu_i``,
    ``Cov(A_i, A_j) = w_i w_j mu_i mu_j (v - nu)`` for ``i != j``, so the sum
    of the component variances plus those cross terms is the total variance.
    """
    sd = a.stats_df
    cols = list(a._comp_cols)
    nu = sd.loc[('freq', 'mean'), 'mixed']
    v = (sd.loc[('freq', 'cv'), 'mixed'] * nu) ** 2
    total = sum((sd.loc[('agg', 'cv'), c] * sd.loc[('agg', 'mean'), c]) ** 2
                for c in cols)
    for ci in cols:
        for cj in cols:
            if ci == cj:
                continue
            total += (sd.loc[('meta', 'wt'), ci] * sd.loc[('meta', 'wt'), cj]
                      * sd.loc[('sev', 'mean'), ci] * sd.loc[('sev', 'mean'), cj]
                      * (v - nu))
    return total


@pytest.mark.parametrize('program', FREQUENCIES)
def test_components_reassemble_the_total_variance(uw, program):
    """The identity that says the component breakout describes the same model
    as the ``mixed`` column. Fails for fixed, binomial, empirical, Neyman A and
    Pascal if the mixture weight rides the claim count."""
    a = uw.build(program)
    mixed_mean = a.stats_df.loc[('agg', 'mean'), 'mixed']
    mixed_var = (a.stats_df.loc[('agg', 'cv'), 'mixed'] * mixed_mean) ** 2
    assert reassembled_variance(a) == pytest.approx(mixed_var, rel=1e-12)


@pytest.mark.parametrize('program', FREQUENCIES)
def test_component_counts_are_the_weighted_parent_mean(uw, program):
    """E[K_i] = w_i E[N], so the per-component frequency means sum to E[N] and
    are in the ratio of the mixture weights."""
    a = uw.build(program)
    sd = a.stats_df
    ex1 = np.array([sd.loc[('freq', 'ex1'), c] for c in a._comp_cols])
    assert ex1.sum() == pytest.approx(sd.loc[('freq', 'mean'), 'mixed'], rel=1e-12)
    assert np.allclose(ex1 / ex1.sum(), [0.9, 0.1])


def test_a_mixture_under_a_logarithmic_frequency_builds(uw):
    """The thinned count 0.3 is below the logarithmic minimum mean, so deriving
    a component by re-solving the family raised; thinning the moments does not."""
    a = uw.build(f'agg T 3 claims {MIX} logarithmic')
    assert a.stats_df.loc[('sev', 'mean'), 'mixed'] == pytest.approx(1.9)


# ---------------------------------------------------------------------------
# dfreq and fixed agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n', [1, 2, 5])
def test_dfreq_n_and_n_claims_fixed_agree(uw, n):
    """A one-atom dfreq and a fixed count are the same frequency, so they must
    produce the same stats_df, per component and in total."""
    a = uw.build(f'agg A dfreq[{n}] {MIX}')
    b = uw.build(f'agg B {n} claims {MIX} fixed')
    cols = ['mixed', 'independent', *a._comp_cols]
    # ``meta.mix_cv`` records how the frequency was declared rather than
    # anything about the distribution: it is the mixing CV, which is NaN for a
    # dfreq (freq_a holds the pmf) and 0 for fixed. Everything that describes
    # the distribution must agree.
    rows = [i for i in a.stats_df.index if i != ('meta', 'mix_cv')]
    assert np.allclose(a.stats_df.loc[rows, cols].astype(float).values,
                       b.stats_df.loc[rows, cols].astype(float).values,
                       equal_nan=True)
    assert np.allclose(a.en, b.en)


# ---------------------------------------------------------------------------
# the mixed exponential, end to end
# ---------------------------------------------------------------------------

# ISO/Verisk style commercial auto mixed exponential; library.agg records the
# mean as 13,990 under CommAutoMixedExponentialSev
MED = ('sev [2764 24548 275654 1917469 10000000] * expon '
       'wts [0.824796 0.159065 0.014444 0.001624 7.1e-05]')
MED_MEAN = 13989.979796


@pytest.mark.parametrize('exposure,freq', [('dfreq[1]', ''),
                                           ('1 claim', 'fixed'),
                                           ('1 claim', 'poisson')])
def test_the_mixed_exponential_reports_its_stated_mean(exposure, freq):
    """The severity mean is the weight-average of the component scales, whatever
    carries the claim count. Under dfreq this reported the unweighted *sum*."""
    uw = Underwriter(update=True)
    a = uw.build(f'agg M {exposure} {MED} {freq}', bs=500, log2=18)
    assert a.stats_df.loc[('sev', 'mean'), 'mixed'] == pytest.approx(MED_MEAN)
    # the FFT severity is pooled with the same weights, so it agrees to
    # discretization error rather than to a factor of the component count
    assert a.est_sev_m == pytest.approx(MED_MEAN, rel=1e-3)


def test_the_density_pools_on_the_mixture_weights_not_the_component_count():
    """``update_work`` derives the severity pooling weights from the per-component
    frequency means, which is the mixture weight only once those means carry it."""
    uw = Underwriter(update=True)
    a = uw.build(f'agg M dfreq[1] {MED}', bs=500, log2=18)
    ex1 = np.array([a.stats_df.loc[('freq', 'ex1'), c] for c in a._comp_cols])
    assert np.allclose(ex1 / ex1.sum(),
                       [0.824796, 0.159065, 0.014444, 0.001624, 7.1e-05])
