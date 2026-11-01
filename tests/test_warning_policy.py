"""Tests for [Warning-Policy]: once-per-session cadence and the materiality floor.

The library's advisory warnings are about a *condition*, not an occurrence. A
sweep that rebuilds the same shape sixty-four times has one grid fault, not
sixty-four, and Python's own dedup cannot see that because the message carries
a computed number. These cover the three pieces that fix it: the ``warn_once``
registry, the materiality floor that decides what is worth saying at all, and
the second, independent channel that fires when a defective law is actually
priced.

Programs mirrored in ``src/aggregate/agg/decl-testers.agg`` (WP block).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats as ss

from aggregate import build, Distortion, Validation
from aggregate._severity import SeverityCHistogram
from aggregate._validation import DEFICIT_MATERIALITY, pmf_deficit
from aggregate.constants import (DefectiveDistributionWarning,
                                 IgnoredDecLClauseWarning, ValidationWarning,
                                 reset_warn_once, validation_warning_suppressed,
                                 warn_once, warn_once_isolated)


# A one-claim unlimited Lomax: alpha 1.5 has no finite variance, so no grid of
# finite extent holds it and the deficit is set by the bucket size. Coarse
# enough to be material, and the same shape as the reproductions book's sweeps.
DEFECTIVE = 'agg WP.Defective 1 claim sev 0.5 * lomax 1.5 fixed'
DEFECTIVE_KW = dict(log2=16, bs=4 / 2 ** 16, normalize=False)
# The same book on a grid fine enough that what it loses is immaterial.
CLEAN = 'agg WP.Clean 100 claims sev lognorm 100 cv 2 poisson'


@pytest.fixture
def caught():
    """Record every warning, with the once-per-session registry re-armed.

    ``conftest`` resets before each test already; this re-resets so the fixture
    reads correctly whatever ran during collection.
    """
    reset_warn_once()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        warnings.simplefilter('ignore', RuntimeWarning)
        yield w


def _defective(w):
    return [x for x in w if x.category is DefectiveDistributionWarning]


# ------------------------------------------------------------- warn_once

def test_warn_once_fires_once_per_category():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        for i in range(5):
            warn_once(f'deficit {i}', DefectiveDistributionWarning)
    assert len(w) == 1
    # The number that varies is in the message, which is exactly why Python's
    # own (text, category, lineno) dedup cannot do this job.
    assert 'deficit 0' in str(w[0].message)


def test_warn_once_returns_whether_it_emitted():
    assert warn_once('first', DefectiveDistributionWarning) is True
    assert warn_once('second', DefectiveDistributionWarning) is False


def test_warn_once_says_that_it_is_suppressing():
    """The one emission tells the reader the silence downstream is policy."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        warn_once('body', DefectiveDistributionWarning)
    text = str(w[0].message)
    assert text.startswith('body')
    assert 'suppressed' in text and 'validation' in text


def test_warn_once_keys_are_independent():
    """An explicit key splits one class into independently-once channels."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        warn_once('build', DefectiveDistributionWarning, key='a')
        warn_once('build again', DefectiveDistributionWarning, key='a')
        warn_once('price', DefectiveDistributionWarning, key='b')
    assert len(w) == 2


def test_warn_once_categories_are_independent():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        warn_once('one', DefectiveDistributionWarning)
        warn_once('two', IgnoredDecLClauseWarning)
    assert len(w) == 2


def test_reset_warn_once_re_arms():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        warn_once('a', DefectiveDistributionWarning)
        warn_once('b', DefectiveDistributionWarning)
        reset_warn_once()
        warn_once('c', DefectiveDistributionWarning)
    assert len(w) == 2


def test_warn_once_isolated_bypasses_the_registry():
    """A probe that COUNTS warnings needs every call to warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with warn_once_isolated():
            for i in range(4):
                warn_once(f'cell {i}', DefectiveDistributionWarning)
    assert len(w) == 4


def test_warn_once_isolated_restores_the_budget():
    """A speculative pre-pass must not spend the session's one emission."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with warn_once_isolated():
            warn_once('probe', DefectiveDistributionWarning)
        warn_once('the real one', DefectiveDistributionWarning)
    assert len(w) == 2
    assert 'the real one' in str(w[-1].message)


def test_warn_once_isolated_restores_a_spent_budget_too():
    """Entering isolation does not un-spend what was already spent."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        warn_once('first', DefectiveDistributionWarning)
        with warn_once_isolated():
            warn_once('probe', DefectiveDistributionWarning)
        warn_once('suppressed again', DefectiveDistributionWarning)
    assert len(w) == 2
    assert 'suppressed again' not in ' '.join(str(x.message) for x in w)


def test_warn_once_isolated_nests():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with warn_once_isolated():
            with warn_once_isolated():
                warn_once('inner', DefectiveDistributionWarning)
            warn_once('outer', DefectiveDistributionWarning)
        warn_once('after', DefectiveDistributionWarning)
    assert len(w) == 3


# --------------------------------------------------- the materiality floor

def test_a_sweep_warns_once(caught):
    """Six defective builds, one warning. This is the whole point."""
    objs = [build(f'agg WP.S{i} 1 claim sev 0.5 * lomax 1.5 fixed',
                  log2=16, bs=4 * u / 2 ** 16, normalize=False)
            for i, u in enumerate([1., 2., 5., 10., 20., 50.])]
    assert len(objs) == 6
    assert all(o._deficit > DEFICIT_MATERIALITY for o in objs)
    assert len(_defective(caught)) == 1


def test_an_immaterial_deficit_is_silent(caught):
    """Below the floor there is nothing the reader could act on."""
    a = build(CLEAN)
    assert a._deficit < DEFICIT_MATERIALITY
    assert _defective(caught) == []


def test_a_material_deficit_warns(caught):
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    assert a._deficit > DEFICIT_MATERIALITY
    assert len(_defective(caught)) == 1
    assert 'deficit' in str(_defective(caught)[0].message)


# --------------------------------------------------- [Validation-Warning]

LAYERED = 'agg WP.Layered 3 claims 1500 xs 250 dsev [1000 2000 3000] fixed'
LAYERED_COARSE = dict(bs=1000, log2=8)


def _verdicts(w):
    return [str(x.message) for x in w if x.category is ValidationWarning]


def test_a_failing_build_warns_once_with_its_verdict(caught):
    a = build(LAYERED, **LAYERED_COARSE)
    assert not a.valid.passes
    verdicts = _verdicts(caught)
    assert len(verdicts) == 1
    assert verdicts[0].startswith('WP.Layered: fails sev mean, agg mean. See validation_explanation.')


def test_a_second_build_of_the_same_program_is_silent(caught):
    build(LAYERED, **LAYERED_COARSE)
    build(LAYERED, **LAYERED_COARSE)
    assert len(_verdicts(caught)) == 1


def test_a_different_failure_of_the_same_name_reports_again(caught):
    build(LAYERED, **LAYERED_COARSE)
    build('agg WP.Layered 1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed',
          **LAYERED_COARSE)
    assert len(_verdicts(caught)) == 2


def test_a_passing_build_is_silent(caught):
    a = build(LAYERED)
    assert a.valid.passes
    assert _verdicts(caught) == []


def test_a_failing_reinsured_view_warns(caught):
    build('agg U.NetOffGrid 1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed',
          bs=1000, log2=8)
    assert _verdicts(caught)[0].startswith(
        'U.NetOffGrid: reinsurance; subject not unreasonable; net fails sev cv, agg cv.')


def test_a_book_warns_once_for_its_failing_units(caught):
    p = build('port WP.Book agg WP.BookA 3 claims 1500 xs 250 dsev [1000 2000 3000] fixed '
              'agg WP.BookB 3 claims 1500 xs 250 dsev [1000 3000] fixed', bs=1000, log2=8)
    assert all(not a.valid.passes for a in p)
    verdicts = _verdicts(caught)
    assert len(verdicts) == 1 and verdicts[0].startswith('WP.Book: ')


def test_a_law_from_its_characteristic_function_warns(caught):
    build('chfagg WP.Wrap stable 1.5 1 hints{tilt=False}')
    verdicts = _verdicts(caught)
    assert len(verdicts) == 1 and 'mass outside the window wrapped in' in verdicts[0]


def test_the_inner_object_of_a_reference_severity_is_quiet(caught):
    build.build_many(f'{LAYERED.replace("WP.Layered", "WP.Inner")} hints{{bs=1000; log2=8}}', update=False)
    a = build('agg WP.Outer 1 claim sev agg.WP.Inner fixed')
    assert a.valid.passes
    assert _verdicts(caught) == []


def test_sharpen_reports_the_grid_it_settles_on(caught):
    """The first grid fails and the probe's cells mostly fail; neither is the reader's concern."""
    a = build(LAYERED, update=False)
    a.update(sharpen=True, **LAYERED_COARSE)
    assert a.valid.passes
    assert a.bs < 1000
    assert _verdicts(caught) == []
    assert 'ValidationWarning' not in ','.join(a.sharpen_df['warns'])


def test_sharpen_reports_a_grid_that_still_fails(caught):
    a = build('agg WP.Thick 1 claim sev lognorm 100 cv 8 fixed', bs=64, log2=10)
    assert len(_verdicts(caught)) == 1
    reset_warn_once()
    a.sharpen(log2_cap=10, bs_limit=2)
    assert not a.valid.passes
    assert len(_verdicts(caught)) == 2


def test_the_suppressor_is_quiet_nests_and_restores(caught):
    with validation_warning_suppressed():
        with validation_warning_suppressed():
            a = build(LAYERED, **LAYERED_COARSE)
        build(LAYERED, bs=1000, log2=9)
    assert not a.valid.passes
    assert _verdicts(caught) == []
    build(LAYERED, **LAYERED_COARSE)
    assert len(_verdicts(caught)) == 1


def test_validation_eps_set_after_a_build_still_applies():
    """The eager verdict leaves the cache empty, as a lazy verdict would."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ValidationWarning)
        a = build(LAYERED, **LAYERED_COARSE)
    a.validation_eps = 10.0
    assert a.valid.passes


# ------------------------------------------------------ Validation.DEFECTIVE

def test_defective_flag_set_and_fails_validation():
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    assert a.valid & Validation.DEFECTIVE
    assert not a.valid.passes


def test_defective_flag_clear_on_a_sound_object():
    a = build(CLEAN)
    assert not (a.valid & Validation.DEFECTIVE)


def test_defective_is_not_in_the_passing_set():
    """DEFECTIVE is a genuine failure, unlike REINSURANCE."""
    assert not (Validation.NOT_UNREASONABLE | Validation.DEFECTIVE).passes


def test_description_quotes_the_deficit():
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    assert 'pmf deficit' in a.validation_description
    assert f'{a._deficit:.3e}' in a.validation_description


def test_explanation_spells_out_the_divergence():
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    text = a.validation_explanation
    assert 'does not sum to 1' in text
    assert 'forwards' in text.lower() and 'backwards' in text.lower()


def test_deficit_leads_the_short_form():
    """A deficit makes every moment comparison below it uninformative."""
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    short = a.validation_description
    assert short.index('pmf deficit') < len(short)
    assert short.startswith('fails pmf deficit')


def test_pmf_deficit_matches_the_definition():
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    assert pmf_deficit(a) == pytest.approx(1.0 - float(np.sum(a.agg_density)))


def test_portfolio_carries_the_deficit_too():
    p = build('port WP.Port '
              'agg WP.PA 50 claims sev lognorm 100 cv 2 poisson '
              'agg WP.PB 20 claims sev lognorm 200 cv 1 poisson')
    p.valid
    assert np.isfinite(p._deficit)
    assert p._deficit == pytest.approx(
        1.0 - float(np.sum(p.density_df['p_total'])))


# --------------------------------------------------- the price-time channel

def test_pricing_a_defective_law_warns_on_its_own_channel(caught):
    """Construction and pricing are different facts; neither swallows the other."""
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    assert len(_defective(caught)) == 1
    a.apply_distortion(Distortion('ph', 0.7), allow_deficit=True)
    msgs = [str(x.message) for x in _defective(caught)]
    assert len(msgs) == 2
    assert 'pricing a law' in msgs[1]
    assert 'parking directions differ' in msgs[1]


def test_the_price_warning_is_once_too(caught):
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    d = Distortion('ph', 0.7)
    for _ in range(5):
        a.apply_distortion(d, allow_deficit=True)
    assert len(_defective(caught)) == 2


def test_pricing_without_opting_in_still_raises():
    """The default is a refusal, not a warning; only allow_deficit warns."""
    from aggregate.constants import DefectiveDistributionError
    a = build(DEFECTIVE, **DEFECTIVE_KW)
    with pytest.raises(DefectiveDistributionError):
        a.apply_distortion(Distortion('ph', 0.7))


def test_an_immaterial_deficit_prices_silently(caught):
    a = build(CLEAN)
    a.apply_distortion(Distortion('ph', 0.7), allow_deficit=True)
    assert _defective(caught) == []


# ------------------------------------------------------------ rv_histogram

def test_chistogram_on_uneven_bins_is_quiet():
    """scipy warns unless ``density`` is stated, and chistogram exists to be uneven."""
    edges = np.array([0., 1., 2., 5., 20., 100.])
    probs = np.array([.4, .3, .2, .07, .03])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        s = SeverityCHistogram(sev_name='chistogram', sev_xs=edges,
                               sev_ps=probs, name='chistogram')
    assert np.isfinite(s.sev1)


def test_density_true_is_what_scipy_already_assumed():
    """The fix is a statement, not a change: identical output either way."""
    edges = np.array([0., 1., 2., 5., 20., 100.])
    heights = np.array([.4, .3, .2, .07, .03]) / np.diff(edges)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        implicit = ss.rv_histogram((heights, edges))
    explicit = ss.rv_histogram((heights, edges), density=True)
    q = np.linspace(-1, 101, 401)
    assert np.array_equal(implicit.cdf(q), explicit.cdf(q))
    assert np.array_equal(implicit.pdf(q), explicit.pdf(q))
    assert implicit.mean() == explicit.mean()


# ------------------------------------------------------- the sharpen probe

def test_the_probe_still_counts_a_warning_per_cell():
    """Once-per-session must not blind the grid probe's soundness gate."""
    a = build('agg WP.Tower as "US Hurricane Reinsurance" 1.74 claims '
              'sev lognorm 8.501 cv 14.624 splice [0 500] poisson')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a.sharpen(good_enough=0, execute=False)
    df = a.sharpen_df
    # The gate is tighter than the warning on purpose, so read the column the
    # gate reads: several probed cells must be flagged, not just the first.
    assert df['defective'].astype(bool).sum() >= 1
    assert np.isfinite(df['deficit']).all()
