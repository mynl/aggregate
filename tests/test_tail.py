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


def test_tail_df_is_spec_only_and_schema():
    # The whole report is valid before update() (agg_density is None).
    a = _agg('agg A 100 claims sev lognorm 100 cv 2 poisson')
    assert a.agg_density is None
    df = a.tail_behavior_df
    assert list(df.index) == ['frequency', 'comp0', 'aggregate']
    assert list(df.columns) == ['family', 'min', 'max', 'left_tail',
                                'right_tail', 'bounded', 'concentrated',
                                'cv', 'note']


def test_tail_df_per_side_tail_classes():
    # The motivating multi-component book: per-side tail classes (not thick/thin),
    # structural support, capped component reads bounded.
    a = _agg('agg TAILTEST [20 30 40] claims [inf inf 1000] xs [100 0 0] '
             'sev [gamma lognorm lognorm] [100 100 100] cv [1 3 1.3] mixed gamma .5')
    df = a.tail_behavior_df
    assert list(df.index) == ['frequency', 'comp0', 'comp1', 'comp2',
                              'severity', 'aggregate']
    assert df.loc['comp1', 'right_tail'] == 'subexponential'   # lognorm cv 3
    assert df.loc['comp2', 'right_tail'] == 'bounded'          # 1000 xs 0 caps it
    assert df.loc['comp2', 'bounded']
    assert df.loc['severity', 'right_tail'] == 'subexponential'  # blend thickest
    assert df.loc['aggregate', 'right_tail'] == 'subexponential'
    # every positive layer is bounded on the left (hard floor at 0)
    assert (df['left_tail'] == 'bounded').all()


def test_tail_df_structural_support_is_exact_when_bounded():
    # Fixed 3 claims x dice [1..6] -> support exactly [3, 18], bounded.
    df = _agg('agg D dfreq [3] dsev [1:6]').tail_behavior_df
    assert df.loc['aggregate', 'min'] == 3.0
    assert df.loc['aggregate', 'max'] == 18.0
    assert df.loc['aggregate', 'bounded']


def test_tail_df_unbounded_support_is_inf():
    # A Poisson x lognorm book is unbounded above; bounded is False.
    df = _agg('agg A 100 claims sev lognorm 100 cv 2 poisson').tail_behavior_df
    row = df.loc['aggregate']
    assert row['min'] == 0.0 and np.isinf(row['max'])
    assert not row['bounded']


def test_tail_df_capped_heavy_note():
    a = _agg('agg B 10 claims 1000 xs 0 sev lognorm 100 cv 2 poisson')
    note = a.tail_behavior_df.loc['comp0', 'note']
    assert 'subexponential base' in note and 'capped at 1,000' in note


def test_tail_df_bounded_no_capped_note():
    # An intrinsically-bounded (atom / uniform) base must NOT read as capped-heavy.
    a = _agg('agg D dfreq [3] dsev [1:6]')
    assert a.tail_behavior_df.loc['comp0', 'note'] == ''


def test_tail_df_concentration_is_cv():
    # cv = sd / mean. A large-E[N] low-cv book is concentrated (cv well below the
    # CONCENTRATION_CV cut ~ 0.1); an ordinary book is not (cv ~ O(1)).
    conc = _agg('agg C 5000 claims sev gamma 100 cv 1 poisson').tail_behavior_df.loc['aggregate']
    assert conc['concentrated']
    assert 0.0 < conc['cv'] < 0.1
    ordinary = _agg('agg O 5 claims sev lognorm 100 cv 2 poisson').tail_behavior_df.loc['aggregate']
    assert not ordinary['concentrated']
    assert ordinary['cv'] > 0.1


def test_tail_df_power_law_note_and_structural_support():
    # Infinite variance (alpha 1.5): right_tail power-law, the note carries the
    # index and the failing moment; support is structural [0, inf] (not a reach).
    row = _agg('agg P 10 claims sev 1 * pareto 1.5 poisson').tail_behavior_df.loc['aggregate']
    assert row['right_tail'] == 'power-law'
    assert 'alpha=1.5' in row['note'] and 'infinite variance' in row['note']
    assert row['min'] == 0.0 and np.isinf(row['max'])


def test_tail_df_signed_support_is_two_sided():
    # A signed-severity aggregate has exact negative structural support.
    df = _agg('agg S dfreq [2] dsev [-3 -1 2 5]').tail_behavior_df
    assert df.loc['aggregate', 'min'] == -6.0   # 2 x (-3)
    assert df.loc['aggregate', 'max'] == 10.0    # 2 x 5


def test_severity_support_layered():
    # claim-space support of the layered loss: 0 .. limit (finite) or .. inf.
    a = _agg('agg A 10 claims 1000 xs 0 sev lognorm 100 cv 2 poisson')
    assert severity_support(a.sevs[0]) == (0.0, 1000.0)
    b = _agg('agg B 10 claims sev lognorm 100 cv 2 poisson')
    lo, hi = severity_support(b.sevs[0])
    assert lo == 0.0 and np.isinf(hi)


def test_tail_narrative_is_layered_and_support_based():
    a = _agg('agg A 100 claims sev lognorm 100 cv 2 poisson')
    desc = a.tail_description
    assert 'frequency tail' in desc and 'aggregate tail' in desc
    assert '[0, inf)' in desc                     # structural support, not reach
    assert 'subexponential right tail' in desc    # per-side class phrasing
    expl = a.tail_explanation
    assert 'single big jump' in expl              # the mechanism
    assert 'cv ~' in expl                         # concentration sentence (cv)


def test_tail_narrative_multi_component_breakdown():
    a = _agg('agg T [20 30 40] claims [inf inf 1000] xs [100 0 0] '
             'sev [gamma lognorm lognorm] [100 100 100] cv [1 3 1.3] mixed gamma .5')
    expl = a.tail_explanation
    assert 'blends 3 components' in expl
    assert 'combined effective severity' in expl


def test_tail_narrative_power_law_reports_moments():
    a = _agg('agg P 10 claims sev 1 * pareto 1.5 poisson')
    assert 'infinite variance' in a.tail_explanation
    assert 'power-law' in a.tail_description


def test_tail_narrative_color_emphasises_thick():
    from aggregate.tail import describe_rows
    a = _agg('agg A 100 claims sev lognorm 100 cv 2 poisson')
    rows = a._tail_rows()
    plain = '\n'.join(describe_rows(rows, color=False))
    colored = '\n'.join(describe_rows(rows, color=True))
    assert '\x1b[' not in plain
    assert '\x1b[1;31m' in colored                # subexponential emphasised


def test_severity_and_frequency_tail_description():
    a = _agg('agg A 100 claims sev lognorm 100 cv 2 poisson')
    assert a.sevs[0].tail_description == 'lognorm, [0, inf), subexponential right tail'
    assert a.frequency.tail_description == 'poisson frequency, super-exponential count'


def test_occ_reins_overlay_row_capped():
    """An unlimited 100% occurrence cession caps the net per-occurrence tail.

    ``occurrence net of inf xs 1000`` cedes everything above 1000, so the
    ``severity (net occ)`` overlay row is bounded at 1000 -- while the aggregate
    row stays GROSS (subexponential), since the sizer ignores occ reinsurance.
    """
    a = _agg('agg A 100 claims sev lognorm 50 cv 1.5 '
             'occurrence net of inf xs 1000 poisson')
    df = a.tail_behavior_df
    assert 'severity (net occ)' in df.index
    net = df.loc['severity (net occ)']
    assert net['right_tail'] == 'bounded'
    assert net['max'] == 1000.0
    assert 'capped at 1000' in str(net['note'])
    # the aggregate is still sized on the gross (heavy) tail
    assert df.loc['aggregate', 'right_tail'] == 'subexponential'


def test_occ_reins_overlay_row_finite_layer_retains_tail():
    """A finite occurrence layer leaves the net per-occurrence tail heavy.

    ``occurrence net of 500 xs 1000`` cedes only a finite slice, so the gross
    tail above 1500 is retained: the overlay row keeps the subexponential right
    tail and reports the tail as retained.
    """
    a = _agg('agg B 100 claims sev lognorm 50 cv 1.5 '
             'occurrence net of 500 xs 1000 poisson')
    net = a.tail_behavior_df.loc['severity (net occ)']
    assert net['right_tail'] == 'subexponential'
    assert 'tail retained' in str(net['note'])


@pytest.mark.parametrize('program,expected,alpha', [
    ('agg A 10 claims sev 100 * fisk 2 poisson', TailClass.POWER_LAW, 2.0),
    ('agg A 10 claims sev 100 * betaprime 2 3 poisson', TailClass.POWER_LAW, 3.0),
    ('agg A 10 claims sev 100 * f 4 6 poisson', TailClass.POWER_LAW, 3.0),       # d2/2
    ('agg A 10 claims sev 100 * invgamma 2 poisson', TailClass.POWER_LAW, 2.0),
    ('agg A 10 claims sev 100 * rayleigh poisson', TailClass.SUPER_EXPONENTIAL, None),
    ('agg A 10 claims sev 100 * maxwell poisson', TailClass.SUPER_EXPONENTIAL, None),
    ('agg A 10 claims sev 100 * wald poisson', TailClass.EXPONENTIAL, None),
    ('agg A 10 claims sev 100 * gibrat poisson', TailClass.SUBEXPONENTIAL, None),
    ('agg A 10 claims sev 100 * gengamma 2 0.5 poisson', TailClass.SUBEXPONENTIAL, None),
    ('agg A 10 claims sev 100 * gengamma 2 1.5 poisson', TailClass.SUPER_EXPONENTIAL, None),
])
def test_expanded_severity_family_tables(program, expected, alpha):
    a = _agg(program)
    cls, lc, got_alpha = classify_severity(a.sevs[0])
    assert cls == expected
    if alpha is not None:
        assert got_alpha == pytest.approx(alpha)


def test_tukeylambda_and_levy_stable_param():
    from aggregate.tail import _family_right_class
    import numpy as np
    # tukeylambda: lambda>0 bounded, =0 exp, <0 power-law(alpha=-1/lambda)
    assert _family_right_class('tukeylambda', 0.5, np.nan)[0] == TailClass.BOUNDED
    assert _family_right_class('tukeylambda', 0.0, np.nan)[0] == TailClass.EXPONENTIAL
    cls, _, al = _family_right_class('tukeylambda', -0.5, np.nan)
    assert cls == TailClass.POWER_LAW and al == pytest.approx(2.0)
    # levy_stable: alpha<2 power-law(index=alpha), alpha==2 super-exp
    cls, _, al = _family_right_class('levy_stable', 1.5, 0.0)
    assert cls == TailClass.POWER_LAW and al == pytest.approx(1.5)
    assert _family_right_class('levy_stable', 2.0, 0.0)[0] == TailClass.SUPER_EXPONENTIAL


def test_finite_support_family_is_bounded_via_fallback():
    # argus has finite scipy support [0,1] but is not in _BOUNDED_SCIPY_SEVS;
    # the fz.support() fallback classifies it bounded (spec-only).
    a = _agg('agg A 10 claims sev 100 * argus 1 poisson')
    assert a.sevs[0].bounded
    assert classify_severity(a.sevs[0])[0] == TailClass.BOUNDED


def test_asymmetric_two_sided_per_side():
    # gumbel_r: super-exponential left, exponential right (different sides).
    from aggregate.tail import _family_sides
    import numpy as np
    left, right, _, _ = _family_sides('gumbel_r', np.nan, np.nan)
    assert left == TailClass.SUPER_EXPONENTIAL and right == TailClass.EXPONENTIAL
    left, right, _, _ = _family_sides('loggamma', np.nan, np.nan)
    assert left == TailClass.EXPONENTIAL and right == TailClass.SUPER_EXPONENTIAL


def test_concentration_helper():
    from aggregate.tail import concentration
    c, cv = concentration(100.0, 10.0)    # m/sd = 10 -> exactly the cut
    assert cv == pytest.approx(0.1)       # cv = sd / m
    assert not c                          # strict z > 1/0.1, so 10 is not in
    c2, cv2 = concentration(0.0, 5.0)     # mean 0 -> cv = inf, not concentrated
    assert np.isinf(cv2) and not c2
    assert concentration(1.0, np.inf) == (None, None)   # infinite variance
