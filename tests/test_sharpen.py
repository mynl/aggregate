"""Tests for [Sharpen-Grid-Probe]: the ``(bs, log2)`` neighbourhood probe.

``bs_window`` / ``port_best_window`` *choose* a grid from the analytic moments
before any FFT runs. ``sharpen`` *audits* that choice afterwards. These cover
the score, the two gates, the parsimony rule, state restoration and the class
surface. Programs mirrored in ``src/aggregate/agg/decl-testers.agg`` (SH block).
"""

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate._bucket_window import (SHARPEN_LOG2_FLOOR, _SHARPEN_TERMS,
                                      sharpen_score)


# A well-behaved book whose auto-sized grid comfortably clears the target.
GOOD = 'agg SH.Good 100 claims sev lognorm 100 cv 2 poisson'
# The same book forced onto a grid far too small to hold it.
STARVED_BS, STARVED_LOG2 = 0.125, 14
PORT = ('port SH.Port agg SH.A 50 claims sev lognorm 100 cv 2 poisson '
        'agg SH.B 20 claims sev lognorm 200 cv 1 poisson')


@pytest.fixture(autouse=True)
def _quiet():
    """Grid probing deliberately visits bad cells; their warnings are expected."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------- the score

def test_score_is_in_tolerance_units():
    """A passing object scores at or under 1: the score IS the validation boundary."""
    a = build(GOOD)
    score, terms = sharpen_score(a)
    assert a.valid.passes
    assert 0 < score <= 1
    # all six terms present, keyed as documented
    assert set(terms) == {f'u_{c}_{m}' for (c, m), _ in _SHARPEN_TERMS}


def test_score_degrades_on_a_starved_grid():
    """Halving the extent the grid can hold makes the score far worse."""
    a = build(GOOD)
    good, _ = sharpen_score(a)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    bad, _ = sharpen_score(a)
    assert bad > 100 * good


def test_score_power_orders_as_a_norm():
    """power=1 <= power=2 <= power=inf, the standard p-norm ordering."""
    a = build(GOOD)
    s1, _ = sharpen_score(a, power=1)
    s2, _ = sharpen_score(a, power=2)
    si, _ = sharpen_score(a, power=np.inf)
    assert s1 <= s2 + 1e-12 <= si + 1e-12


def test_zero_skew_term_drops_out():
    """A theoretically symmetric severity has no skew term to validate against.

    Its FFT estimate of a zero higher moment is grid-dependent noise with no
    meaningful relative error, so the term is dropped, exactly as
    ``valid_aggregate`` drops it.
    """
    a = build('agg SH.Sym 1 claim dsev [-1 0 1] fixed')
    _, terms = sharpen_score(a)
    assert np.isnan(terms['u_sev_skew'])


# ----------------------------------------------------------- the probe gate

def test_probe_gate_skips_a_good_grid():
    """A grid already at or under the target is left alone and nothing is run."""
    a = build(GOOD)
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
    assert len(a.sharpen_df) == 9


def test_probe_returns_self_for_chaining():
    a = build(GOOD)
    assert a.sharpen() is a


# ---------------------------------------------------------- geometry / cells

def test_probe_geometry_is_the_full_three_by_three():
    """Nine cells: half / same / double bs by one step down / same / up in log2."""
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    df = a.sharpen_df
    assert sorted(df.d_bs.unique()) == [-1, 0, 1]
    assert sorted(df.d_log2.unique()) == [-1, 0, 1]
    # extent is bs * 2**log2, so anti-diagonals share an extent
    same = df[df.d_bs + df.d_log2 == 0]
    assert len(same) == 3
    assert np.allclose(same.extent, same.extent.iloc[0])


def test_log2_cap_drops_the_upper_column():
    """No ``log2 + 1`` column once the cap is reached."""
    a = build(GOOD)
    a.sharpen(good_enough=0, log2_cap=a.log2, execute=False)
    assert sorted(a.sharpen_df.d_log2.unique()) == [-1, 0]
    assert len(a.sharpen_df) == 6


def test_log2_floor_drops_the_lower_column():
    """No ``log2 - 1`` column at the floor."""
    a = build(GOOD)
    a.update(log2=SHARPEN_LOG2_FLOOR, bs=64)
    a.sharpen(good_enough=0, execute=False)
    assert sorted(a.sharpen_df.d_log2.unique()) == [0, 1]


# ------------------------------------------------------------ the move gate

def test_starved_grid_moves_and_improves():
    """A failing grid moves to a better cell and the score falls."""
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    before, _ = sharpen_score(a)
    a.sharpen()
    after, _ = sharpen_score(a)
    assert (a.bs, a.log2) != (STARVED_BS, STARVED_LOG2)
    assert after < before
    win = a.sharpen_df[a.sharpen_df.selected].iloc[0]
    assert (win.d_bs, win.d_log2) != (0, 0)


def test_repeated_sharpen_walks_to_a_valid_grid_and_converges():
    """Re-running re-centres the probe, so the descent continues, then stops.

    This is the whole reason the fallback bar is modest: a single probe cannot
    fix a grid that is orders of magnitude wrong, but re-running must be able to
    start the descent at all.
    """
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    assert not a.valid.passes
    for _ in range(6):
        a.sharpen()
        if not a._sharpen_state['moved']:
            break
    assert a.valid.passes
    # converged: the last round found nothing worth taking
    assert not a._sharpen_state['moved']


def test_parsimony_prefers_the_smallest_qualifying_log2():
    """Among cells meeting the target the smallest log2 wins, not the best score.

    More grid almost always helps a little; growing log2 for that is how a probe
    silently doubles everyone's runtime.
    """
    a = build(GOOD)
    a.update(log2=STARVED_LOG2, bs=STARVED_BS)
    a.sharpen(good_enough=1e9)   # every cell qualifies
    df = a.sharpen_df
    win = df[df.selected].iloc[0]
    assert win.log2 == df.log2.min()


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
    centre = a.sharpen_df[(a.sharpen_df.d_bs == 0) & (a.sharpen_df.d_log2 == 0)]
    assert centre.bs.iloc[0] == pytest.approx(STARVED_BS)
    assert int(centre.log2.iloc[0]) == STARVED_LOG2


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
    assert len(df) == 9
    bad = df[df.score.isna()]
    assert len(bad) == 1
    assert 'synthetic cell failure' in bad.note.iloc[0]


# ------------------------------------------------------------ class surface

def test_frame_and_narrative_surface():
    """The frame carries the documented columns and both narrative halves fire."""
    a = build(GOOD)
    assert a.sharpen_df is None
    assert 'not run' in a.sharpen_description
    assert 'not been sharpened' in a.sharpen_explanation
    a.sharpen(good_enough=0, execute=False)
    cols = set(a.sharpen_df.columns)
    for c in ('d_bs', 'd_log2', 'bs', 'log2', 'extent', 'x_min', 'score',
              'aliasing', 'validation', 'warnings', 'seconds', 'selected',
              'note'):
        assert c in cols
    for (comp, meas), _ in _SHARPEN_TERMS:
        assert f'u_{comp}_{meas}' in cols
    assert len(a.sharpen_description) > 20
    assert len(a.sharpen_explanation) > 200


def test_sharpen_df_is_a_copy():
    a = build(GOOD)
    a.sharpen(good_enough=0, execute=False)
    a.sharpen_df.loc[0, 'score'] = -999
    assert a.sharpen_df.loc[0, 'score'] != -999


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
    assert len(p.sharpen_df) == 9
    # probe cells run add_exa=False for speed; the final update restores it
    assert any(c.startswith('exa_') for c in p.density_df.columns)


def test_portfolio_starved_grid_moves():
    p = build(PORT)
    p.update(log2=12, bs=1)
    before, _ = sharpen_score(p)
    p.sharpen()
    after, _ = sharpen_score(p)
    assert (p.bs, p.log2) != (1, 12)
    assert after < before


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
