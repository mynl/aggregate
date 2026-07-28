"""[Reins-Economics-On-Agg-Ignore-Warn]: a pure aggregate ignores what it
cannot use and says so.

Plain ``agg``s accept every reinsurance decoration -- ceded-premium clauses
(``deposit`` / ``rol`` / ``rate``), ceding commissions (``cede``),
``reinstatements``, the variable-rating features -- build the loss structure
only, and emit one :class:`~aggregate.constants.IgnoredDecLClauseWarning`
naming the ignored clauses. The recipe base keeps the **full** spec, so an
``agg.NAME`` reference inside a ``pnl`` / ``xpnl`` re-injects the economics
(the spec-injection route). See
``dev/plan-pnl-consolidated-xpnl-walk.md`` Phase 1.
"""

import numpy as np
import pytest

from aggregate import Underwriter
from aggregate.constants import IgnoredDecLClauseWarning

_ENGINE = ('100 claims sev lognorm 50 cv 1.5 poisson '
           'aggregate net of 2000 xs 3000')


@pytest.fixture
def uw():
    """A clean underwriter (no databases) with eager update."""
    return Underwriter(databases=None, update=True)


# ----------------------------------------------------------------------
# the warning: one per build, naming every ignored clause
# ----------------------------------------------------------------------
def test_warning_names_the_clauses(uw):
    with pytest.warns(IgnoredDecLClauseWarning) as rec:
        uw(f'agg W1 {_ENGINE} deposit 160 cede 25%')
    ours = [w for w in rec
            if issubclass(w.category, IgnoredDecLClauseWarning)]
    assert len(ours) == 1
    msg = str(ours[0].message)
    assert 'ceded premium' in msg
    assert 'cede' in msg
    assert 'pnl' in msg          # the remedy is named


def test_reinstatements_ignored_with_warning(uw):
    with pytest.warns(IgnoredDecLClauseWarning, match='reinstatements'):
        a = uw('agg W2 100 claims sev lognorm 50 cv 1.5 '
               'occurrence net of 500 xs 500 deposit 150 '
               'reinstatements 1 at 100% poisson')
    assert a.occ_reins is not None


def test_variable_feature_ignored_with_warning(uw):
    # swing supplies the ceded premium itself (no deposit alongside)
    with pytest.warns(IgnoredDecLClauseWarning, match='swing'):
        a = uw(f'agg W3 {_ENGINE} swing basic 500 lcm 0.5')
    assert a.agg_reins is not None


def test_undecorated_agg_does_not_warn(uw, recwarn):
    uw(f'agg W4 {_ENGINE}')
    assert not [w for w in recwarn
                if issubclass(w.category, IgnoredDecLClauseWarning)]


# ----------------------------------------------------------------------
# the built loss structure is identical to the undecorated build
# ----------------------------------------------------------------------
def test_loss_structure_identical_to_undecorated(uw):
    with pytest.warns(IgnoredDecLClauseWarning):
        deco = uw(f'agg W5 {_ENGINE} rol 8% cede 25%', bs=1, log2=16)
    plain = uw(f'agg W6 {_ENGINE}', bs=1, log2=16)
    np.testing.assert_array_equal(
        deco.density_df['p_total'].to_numpy(),
        plain.density_df['p_total'].to_numpy())


# ----------------------------------------------------------------------
# spec injection: the stored spec keeps the economics, and an
# ``agg.NAME`` reference inside a pnl activates them
# ----------------------------------------------------------------------
def test_recipe_retains_the_economics(uw):
    with pytest.warns(IgnoredDecLClauseWarning):
        uw(f'agg W7 {_ENGINE} deposit 160 cede 25%')
    stored = uw[('agg', 'W7')].spec
    assert stored['agg_reins_premium'] == [('deposit', 160)]
    assert stored['agg_reins_cede'] == [0.25]


def test_pnl_by_reference_resolves_injected_economics(uw):
    with pytest.warns(IgnoredDecLClauseWarning):
        uw(f'agg W8 {_ENGINE} deposit 160 cede 25%')
    p = uw('pnl P8 5000 premium less agg.W8')
    assert p.economics['pc_agg'] == pytest.approx(160.0)
    assert p.economics['c_agg'] == pytest.approx(0.25 * 160)


def test_rate_resolves_against_the_pnl_premium(uw):
    # ``rate 30%`` has nothing to rate against on the bare agg -- exactly why
    # it is ignored there -- and resolves against the *pnl's* premium here.
    with pytest.warns(IgnoredDecLClauseWarning):
        uw(f'agg W9 {_ENGINE} rate 30%')
    p = uw('pnl P9 5000 premium less agg.W9')
    assert p.economics['pc_agg'] == pytest.approx(0.30 * 5000)


def test_rebuild_by_name_still_warns(uw):
    # the stored entry keeps the economics, so rebuilding the agg by name
    # ignores them again (the stored spec must not have been mutated).
    with pytest.warns(IgnoredDecLClauseWarning):
        uw(f'agg W10 {_ENGINE} deposit 160')
    with pytest.warns(IgnoredDecLClauseWarning):
        a = uw('agg.W10')
    assert type(a).__name__ == 'Aggregate'
