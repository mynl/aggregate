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

[Reins-Economics-On-Bvagg-Ignore-Warn]: the same clauses reach a ``bvagg``
on its component specs, where both unit-construction routes splat the spec
into ``Aggregate``. Before the fix that raised ``TypeError``; now the factory
filters per unit with the joint's own remedy sentence. See
``dev/plan-punchups-aug-24-LIB.md`` item 1.
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


# ----------------------------------------------------------------------
# [Reins-Economics-On-Bvagg-Ignore-Warn]: the joint's components too
# ----------------------------------------------------------------------
#: A dice joint: exact, tiny, and it exercises the cession layers the netceded
#: mode needs kept. The GCN control on a priced occurrence program is the
#: everyday shape of the bug this covers.
_VIEW_PAIR = ('grossceded agg W11 dfreq [3] dsev [2 4 6 8 10 12] '
              'occurrence net of 6 xs 6 deposit 5')

_COPULA_UNIT = ('bivariate W12 5 claims '
                'agg W12a dfreq [0 1] [.5 .5] dsev [2 4 6 8 10 12] '
                'occurrence net of 6 xs 6 deposit 5 '
                'agg W12b dfreq [0 1] [.5 .5] dsev [1 2 3 4] '
                'copula normal 0.5')


@pytest.fixture
def uwb():
    """A clean underwriter with lazy update, for the joints.

    The bug is a construction-time splat of the unit spec into ``Aggregate``,
    so ``__init__`` is the whole story here and the 2-D FFT is pure cost: a
    joint update runs tens of seconds against a fraction of a second for the
    build. Everything asserted below is set in ``__init__``.
    """
    return Underwriter(databases=None, update=False)


def test_view_pair_unit_builds_and_warns(uwb):
    # the GCN shape: before the fix this raised
    # TypeError: unexpected keyword argument 'occ_reins_premium'
    with pytest.warns(IgnoredDecLClauseWarning) as rec:
        b = uwb(_VIEW_PAIR)
    assert type(b).__name__ == 'BivariateAggregate'
    ours = [w for w in rec
            if issubclass(w.category, IgnoredDecLClauseWarning)]
    assert len(ours) == 1
    msg = str(ours[0].message)
    assert msg.startswith('W11:')          # the unit is named, not the joint
    assert 'ceded premium' in msg
    assert 'loss against loss' in msg      # the joint's remedy, not the agg's
    assert 'pnl' not in msg
    # the loss structure survives: the netceded joint needs the cession layers
    assert b._nc_agg.occ_reins is not None


def test_copula_unit_builds_and_warns(uwb):
    with pytest.warns(IgnoredDecLClauseWarning, match='W12a'):
        b = uwb(_COPULA_UNIT)
    assert type(b).__name__ == 'BivariateAggregate'
    assert b.units[0].occ_reins is not None


def test_bvagg_recipe_retains_the_economics(uwb):
    # the filter works on a COPY: the stored unit spec keeps the economics, so
    # the full declaration survives for anything that reads the recipe back.
    with pytest.warns(IgnoredDecLClauseWarning):
        uwb(_VIEW_PAIR)
    stored = uwb[('bvagg', 'W11')].spec
    assert stored['units'][0][2]['occ_reins_premium'] == [('deposit', 5)]
    with pytest.warns(IgnoredDecLClauseWarning):
        uwb('W11')                          # rebuilt from the stored spec


def test_undecorated_bvagg_does_not_warn(uwb, recwarn):
    uwb('grossceded agg W13 dfreq [3] dsev [2 4 6 8 10 12] '
        'occurrence net of 6 xs 6')
    assert not [w for w in recwarn
                if issubclass(w.category, IgnoredDecLClauseWarning)]
