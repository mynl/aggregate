"""Tests for [Sharpen-Grid-Probe]: the ``(bs, log2)`` neighbourhood probe.

``bs_window`` / ``port_best_window`` *choose* a grid from the analytic moments
before any FFT runs. ``sharpen`` *audits* that choice afterwards. These cover
the score, the two gates, the bucket line search, the parsimony rule, state
restoration and the class surface. Programs mirrored in
``src/aggregate/agg/decl-testers.agg`` (SH block).
"""

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate._bucket_window import (SHARPEN_LOG2_FLOOR, _bucket_is_exact,
                                      _fmt_bs)
from aggregate._validation import (SCORE_TERMS, pmf_deficit, validation_score,
                                   validation_score_terms)


# A well-behaved book whose auto-sized grid comfortably clears the target.
GOOD = 'agg SH.Good 100 claims sev lognorm 100 cv 2 poisson'
# The same book forced onto a grid far too small to hold it.
STARVED_BS, STARVED_LOG2 = 0.125, 14
PORT = ('port SH.Port agg SH.A 50 claims sev lognorm 100 cv 2 poisson '
        'agg SH.B 20 claims sev lognorm 200 cv 1 poisson')
# A cat XOL tower: a fine bucket scores best but loses mass off the top of the
# grid, which is the failure a moment score cannot see. bs=1/64 at log2=16
# scores 0.044 with a 2.1e-8 deficit; bs=1/32 scores 0.177 and is sound.
TOWER = ('agg SH.Tower as "US Hurricane Reinsurance" 1.74 claims '
         'sev lognorm 8.501 cv 14.624 splice [0 500] poisson')

# GOOD's auto-sized grid clips ~8e-9 of its tail, so the soundness gate makes
# it probe. CLEAN is thin enough that its grid holds everything.
CLEAN = 'agg SH.Clean 100 claims sev gamma 100 cv 1 poisson'

TERM_NAMES = {f'u_{c}_{m}' for (c, m), _ in SCORE_TERMS}


@pytest.fixture(autouse=True)
def _quiet():
    """Grid probing deliberately visits bad cells; their warnings are expected."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _levels(df, name):
    """Sorted unique values of an index level of ``sharpen_df``."""
    return sorted(set(df.index.get_level_values(name)))


# ---------------------------------------------------------------- the score

def test_score_is_in_tolerance_units():
    """A passing object scores at or under 1: the score IS the validation boundary."""
    a = build(GOOD)
    assert a.valid.passes
    assert 0 < a.validation_score <= 1


def test_score_property_matches_the_worker():
    """The property is the power-2 score, no more and no less."""
    a = build(GOOD)
    assert a.validation_score == pytest.approx(validation_score(a, power=2))


def test_score_terms_are_all_present():
    a = build(GOOD)
    assert set(validation_score_terms(a)) == TERM_NAMES


def test_score_degrades_on_a_starved_grid():
    """Halving the extent the grid can hold makes the score far worse."""
    a = build(GOOD)
    good = a.validation_score
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    assert a.validation_score > 100 * good


def test_score_power_orders_as_a_norm():
    """power=1 <= power=2 <= power=inf, the standard p-norm ordering."""
    a = build(GOOD)
    s1 = validation_score(a, power=1)
    s2 = validation_score(a, power=2)
    si = validation_score(a, power=np.inf)
    assert s1 <= s2 + 1e-12 <= si + 1e-12


def test_zero_skew_term_drops_out():
    """A theoretically symmetric severity has no skew term to validate against.

    Its FFT estimate of a zero higher moment is grid-dependent noise with no
    meaningful relative error, so the term is dropped, exactly as
    ``valid_aggregate`` drops it.
    """
    a = build('agg SH.Sym 1 claim dsev [-1 0 1] fixed')
    assert np.isnan(validation_score_terms(a)['u_sev_skew'])


def test_portfolio_carries_the_score_too():
    p = build(PORT)
    assert np.isfinite(p.validation_score)


# ----------------------------------------------------------- the probe gate

def test_probe_gate_skips_a_good_grid():
    """A grid at or under the target AND holding all its mass is left alone."""
    a = build(CLEAN)
    assert pmf_deficit(a) < 1e-12
    bs0, log20 = a.bs, a.log2
    a.sharpen()
    assert len(a.sharpen_df) == 1
    assert bool(a.sharpen_df.selected.iloc[0])
    assert (a.bs, a.log2) == (bs0, log20)
    assert 'not run' in a.sharpen_description


def test_good_enough_zero_forces_the_probe():
    """``good_enough=0`` never clears, which is how a probe is forced."""
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    assert len(a.sharpen_df) > 1


def test_probe_returns_self_for_chaining():
    a = build(GOOD)
    assert a.sharpen() is a


# ---------------------------------------------------------- the line search

def test_each_log2_row_searches_the_bucket_line():
    """Each row holds the centre bucket plus a contiguous walk out from it."""
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    df = a.sharpen_df
    for j in _levels(df, 'd_log2'):
        ks = sorted(df.xs(j, level='d_log2').index)
        assert 0 in ks and max(ks) >= 1 and min(ks) <= -1
        # contiguous: the search walks outward, it never skips a bucket
        assert ks == list(range(min(ks), max(ks) + 1))


def test_search_stops_at_the_first_cell_that_does_not_improve():
    """Walking out from the centre, every step but the last strictly improved."""
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    df = a.sharpen_df
    for j in _levels(df, 'd_log2'):
        row = df.xs(j, level='d_log2').sort_index()
        for step in (1, -1):
            walk = [row.score.loc[k] for k in
                    range(0, (max(row.index) if step > 0 else min(row.index))
                          + step, step)]
            for before, after in zip(walk, walk[1:-1]):
                assert after < before, f'row {j} step {step}: {walk}'


def test_bs_limit_bounds_the_search():
    """The line search never walks past ``bs_limit`` in either direction."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    a.sharpen(good_enough=0, bs_limit=4, execute=False)
    assert max(abs(k) for k in _levels(a.sharpen_df, 'd_bs')) <= 2


def test_bs_limit_must_be_a_power_of_two():
    a = build(GOOD)
    with pytest.raises(ValueError):
        a.sharpen(good_enough=0, bs_limit=10)


def test_expansion_reaches_further_than_one_step():
    """A badly starved grid walks several buckets in a single call."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    a.sharpen(good_enough=0, execute=False)
    assert max(_levels(a.sharpen_df, 'd_bs')) >= 2


def test_growth_row_is_lazy():
    """``log2 + 1`` is not paid for when a thrifty cell already meets the target.

    It costs twice as much per cell as the current row, and under the selection
    rule it is only ever consulted when nothing affordable reaches the target.
    """
    a = build(GOOD)
    a.update(log2=16, bs=16)          # fails, but a finer bucket fixes it
    a.sharpen(execute=False)
    assert _levels(a.sharpen_df, 'd_log2') == [-1, 0]
    assert not a._sharpen_state['grew']


def test_growth_row_runs_when_nothing_affordable_reaches_the_target():
    a = build(GOOD)
    a.update(log2=13, bs=1 / 2)       # only a bigger grid can reach 0.5
    a.sharpen(execute=False)
    assert 1 in _levels(a.sharpen_df, 'd_log2')
    assert a._sharpen_state['grew']


def test_log2_cap_forbids_growing():
    """No ``log2 + 1`` row once the cap is reached, even when nothing qualifies."""
    a = build(GOOD)
    a.update(log2=16, bs=1 / 8)
    a.sharpen(log2_cap=16, execute=False)
    assert max(_levels(a.sharpen_df, 'd_log2')) == 0
    assert not a._sharpen_state['grew']


def test_log2_floor_drops_the_lower_row():
    """No ``log2 - 1`` row at the floor; the current row always runs."""
    a = build(GOOD)
    a.update(log2=SHARPEN_LOG2_FLOOR, bs=64)
    a.sharpen(good_enough=0, execute=False)
    assert min(_levels(a.sharpen_df, 'd_log2')) == 0


def test_anti_diagonals_share_an_extent():
    """Extent is bs * 2**log2, so d_bs + d_log2 constant means extent constant."""
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    df = a.sharpen_df.reset_index()
    same = df[df.d_bs + df.d_log2 == 0]
    assert len(same) >= 2
    assert np.allclose(same.extent, same.extent.iloc[0])


def test_a_limit_stop_is_recorded_differently_from_a_turn():
    """Running out of ``bs_limit`` while improving is not the same as turning."""
    a = build(GOOD)
    a.update(log2=13, bs=1 / 2)
    a.sharpen(good_enough=0, bs_limit=2, execute=False)
    notes = ' '.join(a.sharpen_df.note.astype(str))
    assert 'bs_limit reached, still improving' in notes


def test_limit_stop_is_surfaced_when_the_winner_sits_on_it():
    a = build(GOOD)
    a.update(log2=13, bs=1 / 2)
    a.sharpen(execute=False)
    win = a.sharpen_df[a.sharpen_df.selected].iloc[0]
    if 'bs_limit' in str(win.note):
        assert 'still improving' in a.sharpen_description


# ---------------------------------------------------- the soundness gate

def test_deficit_is_recorded_for_every_cell():
    """The mass the grid failed to hold, alongside the gate and the warnings."""
    a = build(TOWER)
    a.sharpen(execute=False)
    df = a.sharpen_df
    for c in ('deficit', 'defective', 'warnings', 'warns'):
        assert c in df.columns
    assert df.deficit.notna().all()
    # the gate agrees with the warning the library would raise
    fired = df.warns.astype(str).str.contains('DefectiveDistribution')
    assert (df.defective[fired]).all()


def test_a_better_scoring_defective_cell_is_rejected():
    """The tower case: 0.044 with lost mass loses to 0.177 that holds it all.

    A moment score cannot see a far-tail deficit, so soundness has to be asked
    about separately or the probe hands back a grid that warns at you and on
    which forwards and backwards survival functions disagree.
    """
    a = build(TOWER)
    a.sharpen()
    df = a.sharpen_df
    win = df[df.selected].iloc[0]
    assert not win.defective
    # something did score better, and was thrown out for being defective
    better = df[(df.score < win.score) & np.isfinite(df.score)]
    assert len(better) and better.defective.all()
    assert a.bs == pytest.approx(1 / 32)
    assert 1.0 - float(a.density_df.p_total.sum()) < 1e-12


def test_rejection_is_surfaced_not_silent():
    """Otherwise the frame reads as though the picker ignored its own minimum."""
    a = build(TOWER)
    a.sharpen()
    assert a._sharpen_state['passed_over'] >= 1
    assert 'rejected as defective' in a.sharpen_description
    assert 'gate rather than a term in the score' in a.sharpen_explanation


def test_probe_gate_does_not_wave_through_a_defective_grid():
    """At target but losing mass is not a grid to leave alone."""
    a = build(TOWER)
    a.update(log2=16, bs=1 / 64)
    assert a.validation_score <= 0.5          # the score is happy
    assert 1.0 - float(a.density_df.p_total.sum()) > 1e-12   # the grid is not
    a.sharpen()
    assert a._sharpen_state['ran']
    assert a._sharpen_state['centre_defective']
    assert 1.0 - float(a.density_df.p_total.sum()) < 1e-12
    assert 'loses mass off its top end' in a.sharpen_explanation


def test_growing_log2_cures_a_deficit():
    """Extent is the cure for lost mass, so a deficit can be a reason to grow."""
    a = build(TOWER)
    a.sharpen(good_enough=0.1)
    assert a._sharpen_state['grew']
    assert a.bs == pytest.approx(1 / 64)      # the fine bucket, now affordable
    assert 1.0 - float(a.density_df.p_total.sum()) < 1e-12


def test_deficit_helper_matches_the_realized_mass():
    a = build(TOWER)
    a.update(log2=16, bs=1 / 64)
    assert pmf_deficit(a) == pytest.approx(
        1.0 - float(a.density_df.p_total.sum()), abs=1e-15)
    p = build(PORT)
    assert pmf_deficit(p) == pytest.approx(
        1.0 - float(p.density_df.p_total.sum()), abs=1e-15)


# ----------------------------------------------------- the discrete guard

def test_exact_bucket_is_detected():
    """``bs`` divides every atom, so the discretization is already exact."""
    d = build('agg SH.Dice dfreq [3] dsev [1:6]')
    exact, lattice = _bucket_is_exact(d, d.bs)
    assert exact and lattice == 1.0
    # a bucket that does not divide the lattice is not exact
    assert not _bucket_is_exact(d, 4.0)[0]


def test_continuous_severity_has_no_lattice():
    a = build(GOOD)
    assert _bucket_is_exact(a, a.bs) == (False, None)


def test_discrete_pins_the_bucket_and_probes_log2_only():
    """No bucket change can help an exact grid, so only the grid size is probed."""
    d = build('agg SH.Dice dfreq [3] dsev [1:6]')
    d.sharpen(good_enough=0, execute=False)
    assert _levels(d.sharpen_df, 'd_bs') == [0]
    assert d._sharpen_state['bs_exact']
    assert 'bucket pinned' in d.sharpen_description
    assert 'bs divides every atom' in d.sharpen_explanation


def test_discrete_still_probes_the_grid_size():
    """A failing discrete object fails on extent, which is exactly what log2 fixes."""
    d = build('agg SH.Dice dfreq [3] dsev [1:6]')
    d.sharpen(good_enough=0, execute=False)
    assert len(_levels(d.sharpen_df, 'd_log2')) > 1


def test_finer_than_needed_bucket_is_called_out():
    """Exact but wasteful: atoms on a 5-lattice discretized at bs=1."""
    e = build('agg SH.Five 10 claims dsev [0 5 10 15] poisson')
    e.update(log2=12, bs=1)
    assert _bucket_is_exact(e, 1.0) == (True, 5.0)
    e.sharpen(good_enough=0, execute=False)
    assert 'finer than it needs to be' in e.sharpen_explanation
    assert 'lattice of 5' in e.sharpen_explanation


def test_portfolio_of_discrete_units_takes_the_gcd():
    p = build('port SH.PDisc agg SH.D1 dfreq [2] dsev [2 4 6] '
              'agg SH.D2 dfreq [2] dsev [4 8]')
    exact, lattice = _bucket_is_exact(p, 2.0)
    assert exact and lattice == 2.0


# ------------------------------------------------------------ the move gate

def test_starved_grid_moves_and_improves():
    """A failing grid moves to a better cell and the score falls."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    before = a.validation_score
    a.sharpen()
    assert (a.bs, a.log2) != (STARVED_BS, STARVED_LOG2)
    assert a.validation_score < before


def test_repeated_sharpen_walks_to_a_valid_grid_and_converges():
    """Re-running re-centres the probe, so the descent continues, then stops."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    assert not a.valid.passes
    for _ in range(6):
        a.sharpen()
        if not a._sharpen_state['moved']:
            break
    assert a.valid.passes
    assert not a._sharpen_state['moved']


def test_winner_is_the_best_score_that_does_not_grow_log2():
    """The rule: never make the caller pay more than they already are.

    Not "the smallest log2 that clears the bar", which would take a much worse
    score for a grid saving nobody asked for, and not the outright argmin, which
    would grow the grid for a marginal gain.
    """
    a = build(GOOD)
    a.update(log2=16, bs=16)
    a.sharpen(execute=False)
    df = a.sharpen_df.reset_index()
    win = df[df.selected].iloc[0]
    free = df[(df.d_log2 <= 0) & np.isfinite(df.score) & ~df.defective]
    assert win.d_log2 <= 0
    assert win.score == pytest.approx(free.score.min())


def test_a_free_grid_saving_is_taken_when_the_score_is_a_wash():
    """A smaller log2 wins on a tie, so the bonus is not left on the table."""
    a = build(GOOD)
    a.update(log2=16, bs=16)
    a.sharpen(execute=False)
    df = a.sharpen_df.reset_index()
    win = df[df.selected].iloc[0]
    tied = df[np.isfinite(df.score) & (df.d_log2 <= 0) & ~df.defective
              & (df.score <= win.score * 1.25)]
    assert win.log2 == tied.log2.min()


# ------------------------------------------------------ state / restoration

def test_execute_false_restores_grid_and_locked_settings():
    """``execute=False`` is pure diagnosis: nothing about the object changes.

    Including the four settings ``update_work`` stamps from its own arguments
    (``sev_calc`` / ``discretization_calc`` / ``normalize`` / ``padding``), which
    a bare re-update would silently reset to their defaults.
    """
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS, sev_calc='continuous',
             padding=2, normalize=False)
    before = (a.bs, a.log2, a.sev_calc, a.padding, a.normalize,
              a.discretization_calc)
    a.sharpen(execute=False)
    after = (a.bs, a.log2, a.sev_calc, a.padding, a.normalize,
             a.discretization_calc)
    assert before == after
    assert 'not executed' in a.sharpen_description


def test_locked_settings_survive_an_executed_move():
    """A move re-updates on the new grid but keeps the same update settings."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS, sev_calc='continuous', padding=2)
    a.sharpen()
    assert (a.bs, a.log2) != (STARVED_BS, STARVED_LOG2)
    assert a.sev_calc == 'continuous'
    assert a.padding == 2


def test_explicit_centre_is_honoured():
    """Passing ``bs`` / ``log2`` probes around that cell, not the current one."""
    a = build(GOOD)
    a.sharpen(bs=STARVED_BS, log2=STARVED_LOG2, good_enough=0, execute=False)
    centre = a.sharpen_df.loc[(0, 0)]
    assert centre.bs == pytest.approx(STARVED_BS)
    assert int(centre.log2) == STARVED_LOG2


# --------------------------------------------------------------- robustness

def test_a_failing_cell_does_not_abort_the_sweep(monkeypatch):
    """A cell that raises is recorded as nan plus its note; the sweep completes."""
    a = build(GOOD)
    real = type(a).update_work
    calls = {'n': 0}

    def flaky(self, *args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 3:
            raise RuntimeError('synthetic cell failure')
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(a), 'update_work', flaky)
    a.sharpen(good_enough=0, execute=False)
    df = a.sharpen_df
    bad = df[df.score.isna()]
    assert len(bad) == 1
    assert 'synthetic cell failure' in bad.note.iloc[0]
    # the sweep carried on past it
    assert len(df) > len(bad) + 1


# ------------------------------------------------------------ class surface

def test_frame_is_indexed_by_the_offsets():
    """``(d_bs, d_log2)`` is the index, so the picture is one unstack away."""
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    assert list(a.sharpen_df.index.names) == ['d_bs', 'd_log2']
    picture = a.sharpen_df.score.unstack('d_log2')
    assert picture.loc[0, 0] == pytest.approx(a.validation_score)


def test_frame_and_narrative_surface():
    """The frame carries the documented columns and both narrative halves fire."""
    a = build(GOOD)
    assert a.sharpen_df is None
    assert 'not run' in a.sharpen_description
    assert 'not been sharpened' in a.sharpen_explanation
    a.sharpen(good_enough=0, execute=False)
    cols = set(a.sharpen_df.columns)
    for c in ('bs', 'log2', 'extent', 'x_min', 'score', 'aliasing', 'deficit',
              'defective', 'validation', 'warnings', 'warns', 'seconds',
              'selected', 'note'):
        assert c in cols
    assert TERM_NAMES <= cols
    assert len(a.sharpen_description) > 20
    assert len(a.sharpen_explanation) > 200


def test_description_starts_capitalized():
    a = build(GOOD)
    assert a.sharpen_description.startswith('Sharpen')
    a.sharpen()
    assert a.sharpen_description.startswith('Sharpen')


def test_sub_unit_bs_reads_as_a_binary_fraction():
    """0.125 is 1/8, and reads that way in the narrative."""
    assert _fmt_bs(0.125) == '1/8'
    assert _fmt_bs(1 / 32) == '1/32'
    assert _fmt_bs(2.0) == '2'
    assert _fmt_bs(0.3) == '0.3'          # not a unit fraction, plain format
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    a.sharpen()
    assert '1/8' in a.sharpen_description
    assert '0.125' not in a.sharpen_description


def test_move_phrase_names_only_what_changed():
    """A bucket-only move must not report 'log2 16 to 16'."""
    a = build(GOOD)
    a.sharpen(good_enough=0)
    st, win = a._sharpen_state, a.sharpen_df[a.sharpen_df.selected].iloc[0]
    if int(win.log2) == st['log20'] and win.bs != st['bs0']:
        assert f'log2 {st["log20"]} to {st["log20"]}' not in a.sharpen_description


def test_sharpen_df_is_a_copy():
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    a.sharpen_df.loc[(0, 0), 'score'] = -999
    assert a.sharpen_df.loc[(0, 0), 'score'] != -999


def test_update_sharpen_kwarg_matches_an_explicit_call():
    """``update(sharpen=True)`` lands where ``update()`` then ``sharpen()`` does."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    a.sharpen()
    b = build(GOOD)
    b.update(log2=STARVED_LOG2, bs=STARVED_BS, sharpen=True)
    assert (a.bs, a.log2) == (b.bs, b.log2)


def test_update_default_does_not_sharpen():
    """Off by default: an ordinary update never pays for a probe."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    assert (a.bs, a.log2) == (STARVED_BS, STARVED_LOG2)
    assert a._sharpen_df is None


# ---------------------------------------------------------------- portfolio

def test_portfolio_sharpen_runs_end_to_end():
    """The portfolio probe scores the total and keeps add_exa on the final grid."""
    p = build(PORT)
    p.sharpen(good_enough=0, execute=False)
    assert _levels(p.sharpen_df, 'd_log2') == [-1, 0, 1]
    # probe cells run add_exa=False for speed; the final update restores it
    assert any(c.startswith('exa_') for c in p.density_df.columns)


def test_portfolio_starved_grid_moves():
    p = build(PORT)
    p.update(log2=12, bs=1)
    before = p.validation_score
    p.sharpen()
    assert (p.bs, p.log2) != (1, 12)
    assert p.validation_score < before


def test_portfolio_explanation_states_the_total_only_limitation():
    """The unit blind spot is stated, not left silent."""
    p = build(PORT)
    p.sharpen(good_enough=0, execute=False)
    assert 'total only' in p.sharpen_explanation


# ------------------------------------------------- [Center-Window-Rename]

def test_focus_is_gone_and_center_window_is_the_name():
    a = build(GOOD)
    assert not hasattr(a, 'focus')
    assert callable(a.center_window)
