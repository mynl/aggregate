"""Zero-truncated / zero-modified frequency: the (a, b, 1) class.

Covers ``[ZT-ZM-Frequency-Fix]`` (1.0.0a152), which reparameterized ``zm`` /
``zt`` to the textbook form, and ``[ZT-ZM-Recalibrate-Default]`` (1.0.0a325),
which swapped which reading the DecL marker selects. The exposure clause now
states the **realized** ``E[N]`` by default, so ``n claims`` means n claims,
and the ``!`` marker opts out into the textbook (a, b, 1) form where the
clause is the **un-modified (base)** mean and the reweighting shifts it.

The reference values are computed here from the published formulas rather than
captured from the implementation, so these are genuine cross-checks:

* Klugman, Panjer and Willmot (2012) §6.6 -- the (a, b, 1) construction, and
  Example 6.5 (negative binomial, ``a = 1/3``, ``b = 1/2``).
* Frees (2018), *Loss Data Analytics* ch. 2 -- the zero-modified Poisson table.
* Klugman, Panjer and Willmot (2012) §8.6 -- the deductible map
  :math:`N^L \\to N^P`, cross-checked against the elementary thinning
  identities ``E[N^P] = v E[N^L]`` and
  ``Var(N^P) = v^2 Var(N^L) + v(1 - v) E[N^L]``.
"""

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate.constants import ZeroModifiedExposureWarning
from aggregate.distributions import Frequency


def _freq(name, freq_a=0.0, p0=0.0, base_mean=None):
    """Build a standalone zero-modified frequency with its base mean stamped."""
    fr = Frequency(name, freq_a, 0.0, True, p0)
    if base_mean is not None:
        fr.base_mean = base_mean
        fr.en = fr.modify_mean()
    return fr


# ---------------------------------------------------------------------------
# The (a, b, 1) construction against published tables
# ---------------------------------------------------------------------------

def test_kpw_example_6_5_zero_modified_negbin():
    """Loss Models Example 6.5: ZM negative binomial, ``p0M = 0.6``.

    The example fixes ``a = 1/3``, ``b = 1/2``, i.e. ``beta = 0.5`` and
    ``r = 2.5``; ``aggregate`` parameterizes negbin by the variance multiplier
    ``freq_a = 1 + beta`` and mean ``r beta``.
    """
    fr = _freq('negbin', freq_a=1.5, p0=0.6, base_mean=2.5 * 0.5)
    p = fr.freq_df.p.values
    np.testing.assert_allclose(p[:4],
                               [0.6, 0.189860, 0.110752, 0.055376], atol=5e-7)
    # the un-modified p0 the text quotes
    assert fr._prob_eq_0(1.25) == pytest.approx(0.362887, abs=5e-7)


def test_kpw_example_6_5_zero_truncated_negbin():
    """Same example, zero-truncated (``p0M = 0``)."""
    fr = _freq('negbin', freq_a=1.5, p0=0.0, base_mean=1.25)
    p = fr.freq_df.p.values
    np.testing.assert_allclose(p[:4],
                               [0.0, 0.474651, 0.276880, 0.138440], atol=5e-7)


def test_lda_zero_modified_poisson_table():
    """Loss Data Analytics ch. 2: Poisson(2) modified to ``p0M = 0.6``."""
    fr = _freq('poisson', p0=0.6, base_mean=2.0)
    p = fr.freq_df.p.values
    np.testing.assert_allclose(p[:4], [0.600, 0.125, 0.125, 0.083], atol=5e-4)


@pytest.mark.parametrize('name,freq_a', [
    ('poisson', 0.0), ('negbin', 3.0), ('geometric', 0.0), ('binomial', 0.6),
])
@pytest.mark.parametrize('p0', [0.0, 0.25, 0.5, 0.9])
def test_pgf_matches_the_reweighting_identity(name, freq_a, p0):
    """``G^M(z) = p0M + c (G(z) - p0)`` with ``c = (1 - p0M) / (1 - p0)``."""
    base = 3.0
    plain = Frequency(name, freq_a, 0.0, False, np.nan)
    modified = _freq(name, freq_a=freq_a, p0=p0, base_mean=base)
    z = np.linspace(-0.9, 0.9, 11)
    p0_nat = plain._prob_eq_0(base)
    c = (1 - p0) / (1 - p0_nat)
    np.testing.assert_allclose(modified.freq_pgf(base, z),
                               p0 + c * (plain.freq_pgf(base, z) - p0_nat),
                               rtol=1e-12, atol=1e-14)
    # every raw moment scales by c: mass moved to/from 0 adds nothing to E[N^j]
    np.testing.assert_allclose(np.asarray(modified.freq_moms(base)),
                               c * np.asarray(plain.freq_moms(base)),
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# modify_mean / solve_base_mean
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name,freq_a', [
    ('poisson', 0.0), ('negbin', 3.0), ('geometric', 0.0),
    ('binomial', 0.6), ('logarithmic', 0.0),
])
@pytest.mark.parametrize('p0', [0.0, 0.3, 0.75])
def test_modify_and_solve_round_trip(name, freq_a, p0):
    """``modify_mean`` and ``solve_base_mean`` invert each other."""
    fr = _freq(name, freq_a=freq_a, p0=p0, base_mean=4.0)
    base = fr.solve_base_mean(4.0)
    assert fr.modify_mean(base) == pytest.approx(4.0, rel=1e-10)


def test_zero_truncated_poisson_mean_closed_form():
    """ZT Poisson mean is ``lambda / (1 - exp(-lambda))`` -- at any lambda.

    The pre-a152 solver crashed here for *every* parameterization; a base mean
    below 1 was doubly impossible because the ZT mean cannot go below 1.
    """
    fr = _freq('poisson', p0=0.0, base_mean=0.5)
    assert fr.modify_mean(0.5) == pytest.approx(0.5 / (1 - np.exp(-0.5)))
    for lam in (0.01, 0.5, 1.0, 4.0, 25.0):
        assert fr.modify_mean(lam) == pytest.approx(lam / (1 - np.exp(-lam)))


def test_zero_deflation_direction_works():
    """``p0M`` *below* the natural ``p0`` raises the mean (the broken half)."""
    natural = np.exp(-4.0)
    fr = _freq('poisson', p0=0.5 * natural, base_mean=4.0)
    assert fr.modify_mean(4.0) > 4.0
    assert fr.prob_eq_0 == pytest.approx(0.5 * natural)


def test_infeasible_pin_raises_with_the_bound():
    """A ZT count cannot average below 1; the error says so."""
    fr = _freq('poisson', p0=0.0, base_mean=4.0)
    with pytest.raises(ValueError, match=r'admits only E\[N\] > 1'):
        fr.solve_base_mean(0.5)


def test_prob_eq_0_is_the_modified_value():
    """Zero modification *defines* ``P(N = 0)``; read it straight back."""
    for p0 in (0.0, 0.2, 0.95):
        fr = _freq('poisson', p0=p0, base_mean=4.0)
        assert fr.prob_eq_0 == pytest.approx(p0)


# ---------------------------------------------------------------------------
# DecL: the default pins the mean, ``!`` shifts it
# ---------------------------------------------------------------------------

def test_decl_bang_shifts_the_mean():
    """``4 claims ... poisson zm 0.5 !`` -> base 4, realized E[N] = c * 4."""
    a = build('agg ZMDefault 4 claims dsev [1] poisson zm 0.5 !', log2=12)
    c = 0.5 / (1 - np.exp(-4.0))
    assert a.base_mean == pytest.approx(4.0)
    assert a.n == pytest.approx(c * 4.0)
    assert a.est_m == pytest.approx(c * 4.0, rel=1e-10)
    assert a.density_df.p_total.iloc[0] == pytest.approx(0.5, abs=1e-10)


def test_decl_default_pins_the_mean():
    """``poisson zm 0.5`` -> realized E[N] is exactly the exposure clause."""
    a = build('agg ZMPinned 4 claims dsev [1] poisson zm 0.5', log2=12)
    assert a.n == pytest.approx(4.0, rel=1e-10)
    assert a.base_mean > 4.0
    assert a.frequency.modify_mean(a.base_mean) == pytest.approx(4.0, rel=1e-9)
    assert a.density_df.p_total.iloc[0] == pytest.approx(0.5, abs=1e-10)


def test_decl_zero_truncated_builds_at_every_mean():
    """``zt`` used to raise a NaN solver error for every input."""
    for n, expected in [(0.5, 0.5 / (1 - np.exp(-0.5))),
                        (4.0, 4.0 / (1 - np.exp(-4.0)))]:
        a = build(f'agg ZT{n} {n} claims dsev [1] poisson zt !', log2=12)
        assert a.n == pytest.approx(expected)
        assert a.density_df.p_total.iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_decl_round_trips_the_bang():
    """The writer emits ``!`` so an un-pinned program survives a re-render."""
    a = build('agg ZMRT 4 claims dsev [1] poisson zm 0.5 !', log2=10)
    assert 'zm 0.5 !' in ' '.join(a.pprogram.split())
    plain = build('agg ZMRT2 4 claims dsev [1] poisson zm 0.5', log2=10)
    assert '!' not in plain.pprogram.split('zm')[1]


def test_geometric_zt_is_the_trials_variant():
    """``geometric zt`` is the number-of-trials geometric: support 1, 2, ...,
    mean ``n``, ``p = 1/n``. By memorylessness ``G | G >= 1 = 1 + G'``, so
    zero truncation IS the trials convention (a separate ``geometric !``
    spelling existed only for 1.0.0a335)."""
    a = build('agg GTrials 10 claims dsev [1] geometric zt', log2=10)
    assert a.n == pytest.approx(10.0)
    # support starts at 1 and P(N=1) = p = 1/n (trials pmf p(1-p)^(k-1))
    d = a.density_df.p_total
    assert float(d.iloc[0]) == pytest.approx(0.0, abs=1e-10)
    assert float(d.iloc[1]) == pytest.approx(0.1, rel=1e-8)


# ---------------------------------------------------------------------------
# The monetary-exposure warning
# ---------------------------------------------------------------------------

def test_monetary_exposure_warns_and_names_the_fix():
    """A ``loss`` target is missed when the mean shifts; say so.

    Only reachable through ``!`` since a325: the default pins the target, so
    a reader who did not ask for the base parameterization never sees this.
    """
    with pytest.warns(ZeroModifiedExposureWarning, match=r'Drop the ! '):
        a = build('agg ZMLoss 1000 loss sev lognorm 10 cv 1 poisson zm 0.5 !',
                  log2=16)
    # base count 100 = 1000 / 10, realized halves it
    assert a.base_mean == pytest.approx(100.0)
    assert a.actual_m == pytest.approx(500.0, rel=1e-6)


def test_monetary_exposure_with_bang_hits_the_target_silently():
    with warnings.catch_warnings():
        warnings.simplefilter('error', ZeroModifiedExposureWarning)
        a = build('agg ZMLossPin 1000 loss sev lognorm 10 cv 1 poisson zm 0.5',
                  log2=16)
    assert a.actual_m == pytest.approx(1000.0, rel=1e-6)


def test_claims_exposure_does_not_warn():
    """A count clause is never a money target, so stay quiet either way."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', ZeroModifiedExposureWarning)
        build('agg ZMClaims 4 claims sev lognorm 10 cv 1 poisson zm 0.5', log2=14)
        build('agg ZMClaimsB 4 claims sev lognorm 10 cv 1 poisson zm 0.5 !',
              log2=14)


# ---------------------------------------------------------------------------
# Loss Models §8.6 -- the deductible map
# ---------------------------------------------------------------------------

def test_apply_deductible_lda_5_5_4():
    """Loss Data Analytics 5.5.4: ZM Poisson losses under a deductible.

    ``N^L`` is ZM Poisson with ``lambda = 3``, ``p0M = 0.5``; severity is
    Burr(alpha=3, theta=50, gamma=1) and the deductible is 30. Verified two
    independent ways: the §8.6 parameter map, and the elementary thinning
    identities that any independent-thinning model must satisfy.
    """
    lam, p0m, d = 3.0, 0.5, 30.0
    v = (1.0 / (1.0 + d / 50.0)) ** 3          # Burr survival at the deductible
    assert v == pytest.approx(0.244140625)

    loss_count = _freq('poisson', p0=p0m, base_mean=lam)
    pay_count = loss_count.apply_deductible(v)

    # --- route 1: the shifted parameters -------------------------------
    assert pay_count.base_mean == pytest.approx(v * lam)
    e_np = pay_count.modify_mean()
    m1, m2, _ = pay_count.freq_moms(pay_count.base_mean)
    var_np = m2 - m1 * m1

    # --- route 2: elementary thinning ----------------------------------
    e_nl = loss_count.modify_mean()
    f1, f2, _ = loss_count.freq_moms(lam)
    var_nl = f2 - f1 * f1
    np.testing.assert_allclose(e_np, v * e_nl, rtol=1e-12)
    np.testing.assert_allclose(var_np,
                               v * v * var_nl + v * (1 - v) * e_nl, rtol=1e-12)

    # imposing a deductible always makes "no payment" more likely
    assert pay_count.prob_eq_0 > p0m


def test_apply_deductible_truncated_becomes_modified():
    """Loss Models §8.6: a ZT loss count gives a ZM payment count."""
    zt = _freq('poisson', p0=0.0, base_mean=3.0)
    zm = zt.apply_deductible(0.5)
    assert zm.freq_zm
    assert zm.prob_eq_0 > 0.0


@pytest.mark.parametrize('name,freq_a', [
    ('poisson', 0.0), ('geometric', 0.0), ('negbin', 3.0), ('binomial', 0.6),
])
def test_apply_deductible_thinning_identities(name, freq_a):
    """The two thinning identities hold for every supported family."""
    v = 0.4
    loss_count = _freq(name, freq_a=freq_a, p0=0.3, base_mean=5.0)
    pay_count = loss_count.apply_deductible(v)
    e_nl = loss_count.modify_mean()
    f1, f2, _ = loss_count.freq_moms(5.0)
    var_nl = f2 - f1 * f1
    m1, m2, _ = pay_count.freq_moms(pay_count.base_mean)
    np.testing.assert_allclose(m1, v * e_nl, rtol=1e-10)
    np.testing.assert_allclose(m2 - m1 * m1,
                               v * v * var_nl + v * (1 - v) * e_nl, rtol=1e-10)


def test_apply_deductible_rejects_unsupported_family():
    fr = _freq('logarithmic', p0=0.4, base_mean=3.0)
    with pytest.raises(NotImplementedError, match='apply_deductible'):
        fr.apply_deductible(0.5)


def test_apply_deductible_validates_survival():
    fr = _freq('poisson', p0=0.4, base_mean=3.0)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r'survival must lie in'):
            fr.apply_deductible(bad)
