"""The informational (FYI) premium suffix on the claims / loss exposure heads
([FYI-Premium-Exposure-Head], 1.0.0a266).

``<count> claims <P> premium`` and ``<EL> loss <P> premium`` carry a booked
premium alongside the sizing head, for reconciling a rated book to a modeled
one: the head sizes the law exactly as it did without the suffix, the premium
books ``exp_premium``, and the loss ratio back-fills from the realized expected
loss in ``Aggregate.__init__`` (the same reconciliation the ``premium at lr``
head has always used, run in the other direction; it lands in the
``('meta', 'lr')`` row of ``stats_df``). The renewal head ``T years at r rate``
established the informational-premium concept; this extends it to the ordinary
heads.

The invariant these tests hold is the defining one: an informational clause
NEVER changes the law. The suffix is scalar-only (both the head amount and the
premium), because a vector would broadcast into the component structure and
either change the distribution or misreport per-component premium.

The DecL programs here are mirrored in ``src/aggregate/agg/decl-testers.agg``
section FYI (this module is the canonical source).
"""
import numpy as np
import pytest

from aggregate import build
from aggregate.decl_writer import format_program


TOL = 1e-9


def _meta(a, measure):
    """Total-view meta row (prem / lr) from ``stats_df``."""
    return float(a.stats_df.loc[('meta', measure), 'mixed'])


def test_claims_head_books_premium_and_backfills_lr():
    a = build('agg FyiC 5 claims 20000 premium sev lognorm 1000 cv 2 poisson',
              update=False)
    assert a.n == 5
    assert a.exp_premium == 20000
    # lr back-fills from the realized expected loss: 5 * 1000 / 20000
    assert abs(_meta(a, 'prem') - 20000) < TOL
    assert abs(_meta(a, 'lr') - 0.25) < TOL


def test_loss_head_books_premium_and_backfills_lr():
    a = build('agg FyiL 500 loss 650 premium sev lognorm 100 cv 2 poisson',
              update=False)
    assert abs(float(a.stats_df.loc[('agg', 'mean'), 'mixed']) - 500) < TOL
    assert a.exp_premium == 650
    assert abs(_meta(a, 'lr') - 500 / 650) < TOL


def test_fyi_premium_never_changes_the_law():
    base = build('agg FyiBase 5 claims sev lognorm 100 cv 2 poisson')
    fyi = build('agg FyiSame 5 claims 20000 premium sev lognorm 100 cv 2 poisson')
    assert np.allclose(base.agg_density, fyi.agg_density)
    assert np.allclose(base.sev_density, fyi.sev_density)


def test_fyi_premium_with_layer_clause():
    # after the suffix a layers clause must still parse: premium vs xs anchors
    a = build('agg FyiLay 5 claims 20000 premium 1000 xs 100 '
              'sev lognorm 100 cv 2 poisson', update=False)
    assert a.n == 5
    assert a.exp_premium == 20000
    assert np.unique(a.limit)[0] == 1000
    assert np.unique(a.attachment)[0] == 100


def test_comma_spelling_is_the_same_program():
    a = build('agg FyiComma 5 claims, 20000 premium sev lognorm 100 cv 2 poisson',
              update=False)
    assert a.n == 5
    assert a.exp_premium == 20000


def test_labels_land_on_the_exposure_and_premium_sites():
    a = build('agg FyiLab 5 claims as counts 20000 premium as booked '
              'sev lognorm 100 cv 2 poisson', update=False)
    assert a.label_map.get('exposure') == 'counts'
    assert a.label_map.get('premium') == 'booked'


def test_inherit_premium_reads_the_engine_fyi_premium():
    p = build('pnl FyiPnl inherit premium less agg FyiPnlE 5 claims '
              '20000 premium sev lognorm 100 cv 2 poisson')
    legs = p.legs_df
    booked = float(legs.loc[legs['kind'] == 'premium', 'EX'].iloc[0])
    assert booked == 20000


def test_round_trips_through_the_writer():
    out = format_program(
        'agg FyiRt 5 claims 20000 premium sev lognorm 100 cv 2 poisson')
    assert '5 claims 20000 premium' in out
    lab = format_program('agg FyiRtL 500 loss 650 premium as booked '
                         'sev lognorm 100 cv 2 poisson')
    assert '500 loss 650 premium as booked' in lab


def test_premium_lr_head_still_sizes():
    # the sizing form is untouched: premium WITH lr derives the expected loss
    a = build('agg FyiSize 1000 premium at 0.5 lr sev lognorm 50 cv 0.8 poisson',
              update=False)
    assert abs(float(a.stats_df.loc[('agg', 'mean'), 'mixed']) - 500) < TOL
    assert a.exp_premium == 1000


def test_suffix_plus_at_lr_is_a_parse_error():
    # deliberate: the FYI suffix takes no ``at``, so combining the two premium
    # readings fails loudly at the ``at`` rather than guessing a precedence
    with pytest.raises(ValueError, match="Unexpected 'at'"):
        build('agg FyiBad 5 claims 20000 premium at 0.65 lr '
              'sev lognorm 100 cv 2 poisson', update=False)


def test_vector_fyi_premium_is_refused():
    # a vector premium would broadcast into extra components: changes the law
    with pytest.raises(ValueError, match='FYI premium'):
        build('agg FyiVec 5 claims [100 200] premium '
              'sev lognorm 100 cv 2 poisson', update=False)


def test_vector_head_with_fyi_premium_is_refused():
    # a scalar premium against a vector head repeats per component: misreports
    with pytest.raises(ValueError, match='FYI premium'):
        build('agg FyiVecHead [5 10] claims 20000 premium '
              'sev lognorm [100 200] cv 2 poisson', update=False)
