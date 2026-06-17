"""Tests for the Phase-1 (deterministic) tail-thickness classifier.

Covers the ordered scale and combine rule, the frequency / severity family
lookups (including the param-aware families and power-law alpha extraction),
the aggregate combine result, the derived-``bounded`` equivalence (pre-update),
the certify override, multi-component severities, and the Portfolio worst-of.

The numeric density estimator is Phase 2 and not exercised here.
"""

import numpy as np
import pytest
import scipy.stats as ss

from aggregate import build
from aggregate.tail import (
    TailClass, classify_frequency, classify_severity, combine,
    aggregate_tail_info, is_thick, thickness_label, severity_support,
    CONCENTRATION_CV, _BOUNDED_FREQS, _BOUNDED_SCIPY_SEVS,
)


def _agg(program):
    """Build an Aggregate without updating (tail info is spec-only)."""
    return build(program, update=False)


# ---------------------------------------------------------------------------
# Scale and combine rule.
# ---------------------------------------------------------------------------

def test_ordering():
    assert (TailClass.BOUNDED < TailClass.SUPER_EXPONENTIAL
            < TailClass.EXPONENTIAL < TailClass.SUBEXPONENTIAL
            < TailClass.POWER_LAW)


def test_combine_is_max():
    assert combine(TailClass.SUPER_EXPONENTIAL, TailClass.SUBEXPONENTIAL) == TailClass.SUBEXPONENTIAL
    assert combine(TailClass.EXPONENTIAL, TailClass.BOUNDED) == TailClass.EXPONENTIAL
    assert combine(TailClass.POWER_LAW, TailClass.SUPER_EXPONENTIAL) == TailClass.POWER_LAW


def test_combine_unknown_poisons():
    assert combine(TailClass.UNKNOWN, TailClass.POWER_LAW) == TailClass.UNKNOWN
    assert combine(TailClass.BOUNDED, TailClass.UNKNOWN) == TailClass.UNKNOWN
    # UNKNOWN is off the order: it must not be max()'d in as a real thickness.
    assert combine(TailClass.UNKNOWN, TailClass.BOUNDED) == TailClass.UNKNOWN


def test_label_str():
    assert str(TailClass.POWER_LAW) == 'power-law'
    assert str(TailClass.BOUNDED) == 'bounded'
    assert str(TailClass.UNKNOWN) == 'undetermined'


# ---------------------------------------------------------------------------
# Frequency classification.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program,expected', [
    ('agg F dfreq [3] dsev [1]', TailClass.BOUNDED),                  # fixed
    ('agg F 100 claims sev expon 1 poisson', TailClass.SUPER_EXPONENTIAL),
    ('agg F 100 claims sev expon 1 negbin 0.5', TailClass.EXPONENTIAL),
    ('agg F 100 claims sev expon 1 geometric', TailClass.EXPONENTIAL),
])
def test_classify_frequency(program, expected):
    a = _agg(program)
    assert classify_frequency(a.frequency)[0] == expected


def test_poisson_log_concave():
    a = _agg('agg F 100 claims sev expon 1 poisson')
    cls, lc = classify_frequency(a.frequency)
    assert cls == TailClass.SUPER_EXPONENTIAL and lc is True


# ---------------------------------------------------------------------------
# Severity classification (param-aware) and alpha extraction.
# ---------------------------------------------------------------------------

def test_severity_gamma_log_concave_boundary():
    # gamma shape >= 1 is log-concave; shape < 1 is not.
    a = _agg('agg S 10 claims sev gamma 100 cv 0.5 poisson')   # shape = 1/cv**2 = 4 >= 1
    cls, lc, alpha = classify_severity(a.sevs[0])
    assert cls == TailClass.EXPONENTIAL and lc is True
    b = _agg('agg S 10 claims sev gamma 100 cv 2 poisson')     # shape = 0.25 < 1
    cls, lc, alpha = classify_severity(b.sevs[0])
    assert cls == TailClass.EXPONENTIAL and lc is False


@pytest.mark.parametrize('c,expected,lc', [
    (2.0, TailClass.SUPER_EXPONENTIAL, True),
    (1.0, TailClass.EXPONENTIAL, True),
    (0.5, TailClass.SUBEXPONENTIAL, False),
])
def test_severity_weibull_spans_three_rungs(c, expected, lc):
    a = _agg(f'agg S 10 claims sev 100 * weibull_min {c} poisson')
    cls, got_lc, alpha = classify_severity(a.sevs[0])
    assert cls == expected and got_lc is lc


def test_severity_lognorm_subexponential():
    a = _agg('agg S 10 claims sev lognorm 100 cv 2 poisson')
    cls, lc, alpha = classify_severity(a.sevs[0])
    assert cls == TailClass.SUBEXPONENTIAL and lc is False


def test_severity_pareto_power_law_alpha():
    a = _agg('agg S 10 claims sev 100 * pareto 3 poisson')
    cls, lc, alpha = classify_severity(a.sevs[0])
    assert cls == TailClass.POWER_LAW
    assert alpha == pytest.approx(3.0)


@pytest.mark.parametrize('family,shape,scipy_dist,expected_alpha', [
    ('pareto', 2.5, ss.pareto, 2.5),
    ('lomax', 3.0, ss.lomax, 3.0),
    ('fisk', 4.0, ss.fisk, 4.0),
    ('invgamma', 2.0, ss.invgamma, 2.0),
    ('t', 5.0, ss.t, 5.0),
])
def test_power_law_alpha_matches_scipy(family, shape, scipy_dist, expected_alpha):
    # The reported alpha (from the per-family shape-slot table) must equal both
    # the known exponent and the empirical log-log survival slope of the scipy
    # family, confirming the parameter slot was read correctly.
    a = _agg(f'agg S 10 claims sev 1 * {family} {shape} poisson')
    cls, lc, alpha = classify_severity(a.sevs[0])
    assert cls == TailClass.POWER_LAW
    assert alpha == pytest.approx(expected_alpha)
    # numeric confirmation: -d log S / d log x -> alpha in the tail (moderate x
    # to avoid survival-function underflow to 0).
    x = np.array([1e2, 1e3])
    sf = scipy_dist.sf(x, shape)
    emp_alpha = -(np.log(sf[1]) - np.log(sf[0])) / (np.log(x[1]) - np.log(x[0]))
    assert alpha == pytest.approx(emp_alpha, rel=5e-2)


def test_infinite_variance_mean_flags():
    a = _agg('agg S 10 claims sev 1 * pareto 1.5 poisson')   # alpha 1.5: inf var, finite mean
    ti = a._tail_info()
    assert ti.flags['infinite_variance'] is True
    assert ti.flags['infinite_mean'] is False
    b = _agg('agg S 10 claims sev 1 * pareto 0.8 poisson')   # alpha 0.8: inf mean
    tb = b._tail_info()
    assert tb.flags['infinite_mean'] is True


# ---------------------------------------------------------------------------
# Aggregate combine result.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program,expected', [
    ('agg A 100 claims sev lognorm 100 cv 2 poisson', TailClass.SUBEXPONENTIAL),
    ('agg A 100 claims sev gamma 100 cv 0.5 poisson', TailClass.EXPONENTIAL),
    ('agg A 100 claims sev 100 * pareto 3 poisson', TailClass.POWER_LAW),
    ('agg A dfreq [2] dsev [1 2 3]', TailClass.BOUNDED),
    ('agg A 100 claims sev 100 * uniform poisson', TailClass.SUPER_EXPONENTIAL),  # freq-driven
    ('agg A 100 claims sev expon 100 negbin 0.5', TailClass.EXPONENTIAL),
])
def test_aggregate_combine(program, expected):
    a = _agg(program)
    assert a.tail_class.agg == expected


def test_frequency_driven_aggregate_names_driver():
    # bounded severity, Poisson frequency -> super-exponential, driven by freq.
    a = _agg('agg A 100 claims sev 100 * uniform poisson')
    ti = a._tail_info()
    assert ti.agg == TailClass.SUPER_EXPONENTIAL
    assert ti.driver == 'frequency'


# ---------------------------------------------------------------------------
# Multi-component severity (thickest wins; all-log-concave).
# ---------------------------------------------------------------------------

def test_mixed_severity_takes_thickest():
    a = _agg('agg M 10 claims sev [100 200] * [lognorm pareto] [2 3] wts [.5 .5] poisson')
    assert len(a.sevs) == 2
    ti = a._tail_info()
    assert ti.sev == TailClass.POWER_LAW
    assert ti.agg == TailClass.POWER_LAW
    assert ti.alpha == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Derived bounded equivalence + certify override (the key invariant).
# ---------------------------------------------------------------------------

BOUNDED_CASES = [
    'agg B dfreq [2] dsev [1 2 3]',
    'agg B 10 claims sev 100 * uniform poisson',
    'agg B 10 claims sev 100 * lognorm 1 poisson',     # unbounded
    'agg B 10 claims sev 100 * pareto 3 poisson',       # unbounded
    'agg B 10 claims 1000 xs 0 sev lognorm 100 cv 2 poisson',  # finite layer caps severity
]


@pytest.mark.parametrize('program', BOUNDED_CASES)
def test_bounded_is_derived_and_spec_only(program):
    # Crucially: assert on an un-update()'d object (the lifted-NA guard needs
    # bounded to be resolvable from the spec, before any density exists).
    a = _agg(program)
    assert a.agg_density is None
    assert a.bounded == (a.tail_class.agg == TailClass.BOUNDED)


@pytest.mark.parametrize('program', BOUNDED_CASES)
def test_severity_bounded_is_derived(program):
    a = _agg(program)
    for s in a.sevs:
        assert s.bounded == (classify_severity(s)[0] == TailClass.BOUNDED)


def test_certify_flips_bounded_and_class():
    a = _agg('agg C 10 claims sev 100 * pareto 3 poisson')
    assert a.bounded is False
    assert a.tail_class.agg == TailClass.POWER_LAW
    a.bounded = True
    assert a.bounded is True
    assert a.tail_class.agg == TailClass.BOUNDED
    a.bounded = False
    assert a.bounded is False
    assert a.tail_class.agg == TailClass.POWER_LAW


# ---------------------------------------------------------------------------
# Portfolio worst-of.
# ---------------------------------------------------------------------------

def _port_thin_heavy():
    from aggregate import build
    return build('port PF '
                 'agg thin 10 claims sev gamma 100 cv 0.5 poisson '
                 'agg heavy 10 claims sev 100 * pareto 3 poisson',
                 update=False)


def test_portfolio_worst_of():
    p = _port_thin_heavy()
    assert p.tail_class == TailClass.POWER_LAW
    worst, drivers = p._tail_driver()
    assert drivers == ['heavy']
    assert p.bounded is False


def test_portfolio_bounded_when_all_units_bounded():
    # Both units need bounded frequency AND bounded severity: a Poisson count is
    # unbounded, so dfreq (fixed count) is used with a bounded uniform severity.
    p = build('port PB '
              'agg u1 dfreq [2] dsev [1 2 3] '
              'agg u2 dfreq [5] sev 100 * uniform',
              update=False)
    assert p.tail_class == TailClass.BOUNDED
    assert p.bounded is True


def test_portfolio_certify():
    p = _port_thin_heavy()
    assert p.bounded is False
    p.bounded = True
    assert p.bounded is True
    assert p.tail_class == TailClass.BOUNDED


# ---------------------------------------------------------------------------
# Text surfaces (smoke).
# ---------------------------------------------------------------------------

def test_aggregate_tail_text_smoke():
    a = build('agg A 100 claims sev lognorm 100 cv 2 poisson')
    desc = a.tail_description
    assert 'frequency tail' in desc and 'aggregate tail' in desc
    assert 'subexponential' in desc
    assert 'subexponential' in a.tail_explanation
    assert 'aggregate tail' in a.info        # info integration


def test_portfolio_tail_text_smoke():
    p = _port_thin_heavy()
    assert 'power-law' in p.tail_description
    assert 'heavy' in p.tail_explanation
    assert 'aggregate tail' in p.info


def test_tables_match_legacy():
    # The bounded tables moved to tail.py; the distributions re-export must match.
    from aggregate import distributions as d
    assert d._BOUNDED_FREQS is _BOUNDED_FREQS
    assert d._BOUNDED_SCIPY_SEVS is _BOUNDED_SCIPY_SEVS


# ---------------------------------------------------------------------------
# Layered thick/thin tail report (tail_df).
# ---------------------------------------------------------------------------

def test_thick_thin_cut():
    # thick <=> subexponential-or-heavier; UNKNOWN is conservatively thick.
    assert is_thick(TailClass.SUBEXPONENTIAL) and is_thick(TailClass.POWER_LAW)
    assert is_thick(TailClass.UNKNOWN)
    for r in (TailClass.BOUNDED, TailClass.SUPER_EXPONENTIAL, TailClass.EXPONENTIAL):
        assert not is_thick(r)
    assert thickness_label(TailClass.SUBEXPONENTIAL) == 'thick'
    assert thickness_label(TailClass.EXPONENTIAL) == 'thin'


def test_tail_df_is_spec_only():
    # The whole report is valid before update() (agg_density is None).
    a = _agg('agg A 100 claims sev lognorm 100 cv 2 poisson')
    assert a.agg_density is None
    df = a.tail_df
    assert list(df.index) == ['frequency', 'comp0', 'aggregate']
    assert {'min', 'max', 'left', 'right', 'bounded', 'tail_class',
            'alpha', 'concentrated', 'concentration_p'} <= set(df.columns)


def test_tail_df_layered_rows_and_thickness():
    # The motivating multi-component book: one row per component + combined +
    # freq + aggregate; lognorm cv 3 is thick-right, capped lognorm is bounded.
    a = _agg('agg TAILTEST [20 30 40] claims [inf inf 1000] xs [100 0 0] '
             'sev [gamma lognorm lognorm] [100 100 100] cv [1 3 1.3] mixed gamma .5')
    df = a.tail_df
    assert list(df.index) == ['frequency', 'comp0', 'comp1', 'comp2',
                              'severity', 'aggregate']
    assert df.loc['comp1', 'right'] == 'thick'          # lognorm cv 3
    assert df.loc['comp2', 'bounded']                    # 1000 xs 0 caps it
    assert df.loc['comp2', 'right'] == 'thin'
    assert df.loc['severity', 'right'] == 'thick'        # blend takes thickest
    assert df.loc['aggregate', 'right'] == 'thick'       # single big jump
    # every row's left tail is thin (all support >= 0)
    assert (df['left'] == 'thin').all()


def test_tail_df_capped_heavy_note():
    a = _agg('agg B 10 claims 1000 xs 0 sev lognorm 100 cv 2 poisson')
    note = a.tail_df.loc['comp0', 'note']
    assert 'subexponential base' in note and 'capped at 1,000' in note


def test_tail_df_bounded_no_capped_note():
    # An intrinsically-bounded (atom / uniform) base must NOT read as capped-heavy.
    a = _agg('agg D dfreq [3] dsev [1:6]')
    assert a.tail_df.loc['comp0', 'note'] == ''


def test_tail_df_concentration_conservative():
    # Large-E[N] low-cv book is concentrated; an ordinary book is not. The cut
    # is the conservative CONCENTRATION_CV (0.1), tighter than the legacy ~0.14.
    conc = _agg('agg C 5000 claims sev gamma 100 cv 1 poisson').tail_df.loc['aggregate']
    assert conc['concentrated'] is True or conc['concentrated'] == True   # noqa
    assert conc['concentration_p'] > 1.0 / CONCENTRATION_CV
    ordinary = _agg('agg O 5 claims sev lognorm 100 cv 2 poisson').tail_df.loc['aggregate']
    assert not ordinary['concentrated']


def test_tail_df_power_law_alpha_and_nan_reach():
    # Infinite variance (alpha 1.5): alpha is reported but the MoM reach is nan
    # (honest -- no finite deep quantile to size to).
    a = _agg('agg P 10 claims sev 1 * pareto 1.5 poisson')
    row = a.tail_df.loc['aggregate']
    assert row['tail_class'] == 'power-law'
    assert row['alpha'] == pytest.approx(1.5)
    assert np.isnan(row['min']) and np.isnan(row['max'])


def test_tail_df_signed_reach_is_two_sided():
    # A signed-severity aggregate reports a negative lower reach (not floored 0).
    a = _agg('agg S dfreq [2] dsev [-3 -1 2 5]')
    assert a._signed()
    assert a.tail_df.loc['aggregate', 'min'] < 0.0


def test_severity_support_layered():
    # claim-space support of the layered loss: 0 .. limit (finite) or .. inf.
    a = _agg('agg A 10 claims 1000 xs 0 sev lognorm 100 cv 2 poisson')
    assert severity_support(a.sevs[0]) == (0.0, 1000.0)
    b = _agg('agg B 10 claims sev lognorm 100 cv 2 poisson')
    lo, hi = severity_support(b.sevs[0])
    assert lo == 0.0 and np.isinf(hi)
