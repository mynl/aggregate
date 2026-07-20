"""Stage-2 parse tests for the renewal ``years`` / ``wait`` / ``dwait`` DecL.

Exact spec dicts for each surface form, strict-pairing parse failures, and
keyword-boundary checks. Parse-only -- no ``build()``. DecL programs are
mirrored in ``src/aggregate/agg/decl-testers.agg`` (Z.*) and exercised as
suite lines in ``src/aggregate/agg/_test_suite.agg`` (Y.*).
See dev/plan-sparre-a.md [Renewal-Frequency-Wait-Clause].
"""

import numpy as np
import pytest

from aggregate.parser import UnderwritingParser


@pytest.fixture(scope='module')
def parser():
    # safe_lookup / builtin resolution unused by these programs
    return UnderwritingParser(None, None)


def _spec(parser, program):
    kind, name, spec = parser.parse(program)
    assert kind == 'agg'
    return name, spec


def test_plain_wait(parser):
    name, spec = _spec(parser,
                       'agg T1 10 years sev lognorm 100 cv 1 wait expon')
    assert spec['exp_years'] == 10.0
    assert spec['exp_en'] == -1
    assert spec['freq_name'] == 'renewal'
    assert spec['wait_name'] == 'expon'
    assert spec['wait_scale'] == 1.0
    assert 'exp_rate' not in spec
    assert not any(k.startswith('freq_a') for k in spec)


def test_years_at_rate(parser):
    name, spec = _spec(parser,
                       'agg T2 1 year at 500 rate sev lognorm 100 cv 1 '
                       'wait 0.1 * expon')
    assert spec['exp_years'] == 1.0
    assert spec['exp_rate'] == 500.0
    assert spec['exp_premium'] == 500.0
    assert spec['wait_scale'] == pytest.approx(0.1)


def test_dwait_plain(parser):
    name, spec = _spec(parser, 'agg T3 3 years dsev [1] dwait [1]')
    assert spec['wait_name'] == 'dhistogram'
    assert np.array_equal(spec['wait_xs'], [1.0])
    assert np.array_equal(spec['wait_ps'], [1.0])
    assert spec['freq_name'] == 'renewal'
    # the severity clause is untouched
    assert spec['sev_name'] == 'dhistogram'


def test_dwait_cluster_probs(parser):
    name, spec = _spec(parser, 'agg T4 2 years dsev [1] dwait [0 1] [.5 .5]')
    assert np.array_equal(spec['wait_xs'], [0.0, 1.0])
    assert np.array_equal(spec['wait_ps'], [0.5, 0.5])
    assert spec.get('wait_conditional', True)


def test_dwait_defective(parser):
    name, spec = _spec(parser, 'agg T5 2 years dsev [1] dwait [1 2] [.4 .5] !')
    assert spec['wait_conditional'] is False
    assert np.array_equal(spec['wait_ps'], [0.4, 0.5])


def test_dwait_defective_over_one_rejected(parser):
    with pytest.raises(ValueError):
        parser.parse('agg T5x 2 years dsev [1] dwait [1 2] [.6 .5] !')


def test_dwait_renormalizes_with_warning(parser, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        name, spec = _spec(parser,
                           'agg T5y 2 years dsev [1] dwait [1 2] [.4 .4]')
    assert np.allclose(spec['wait_ps'], [0.5, 0.5])
    assert any('renormalizing' in r.message for r in caplog.records)


def test_wait_splice_unconditional(parser):
    name, spec = _spec(parser,
                       'agg T6 2 years sev lognorm 10 cv .3 '
                       'wait expon splice [0 1.1] !')
    assert spec['wait_conditional'] is False
    assert np.asarray(spec['wait_lb']).item() == 0.0
    assert np.asarray(spec['wait_ub']).item() == pytest.approx(1.1)


def test_wait_mixture(parser):
    name, spec = _spec(parser,
                       'agg T7 5 years sev gamma 10 cv .3 '
                       'wait [.6 .4] * expon wts [.5 .5]')
    assert np.allclose(spec['wait_scale'], [0.6, 0.4])
    assert np.allclose(spec['wait_wt'], [0.5, 0.5])


def test_wait_label_site(parser):
    name, spec = _spec(parser,
                       'agg T8 10 years as decade dsev [1] '
                       'dwait [1:3] as slow_waits')
    assert spec['label_map'] == {'exposure': 'decade', 'wait': 'slow_waits'}
    assert np.array_equal(spec['wait_xs'], [1.0, 2.0, 3.0])


def test_wait_rejects_picks(parser):
    with pytest.raises(ValueError, match='picks'):
        parser.parse('agg T9 2 years dsev [1] '
                     'wait lognorm 1 cv .5 picks [0 2] [1 3]')


def test_strict_pairing_years_needs_wait(parser):
    with pytest.raises(ValueError):
        parser.parse('agg B1 2 years sev lognorm 10 cv .3 poisson')


def test_strict_pairing_wait_needs_years(parser):
    with pytest.raises(ValueError):
        parser.parse('agg B2 10 claims sev lognorm 10 cv .3 wait expon')


def test_years_vector_rejected(parser):
    with pytest.raises(ValueError, match='scalar'):
        parser.parse('agg B3 [1 2] years dsev [1] dwait [1]')


def test_yearly_still_valid_id(parser):
    # keyword boundary lookahead: 'yearly' must lex as an ordinary ID
    name, spec = _spec(parser, 'agg yearly 1 claims sev lognorm 10 cv .3 fixed')
    assert name == 'yearly'
    assert spec['freq_name'] == 'fixed'


# ---------------------------------------------------------------------------
# layered waits (wait y xs a <dist>)  [Wait-Clause-Layers]
# ---------------------------------------------------------------------------

def test_wait_layer_spec(parser):
    name, spec = _spec(parser,
                       'agg L1 10 years dsev [1] wait 2 xs 0.25 expon')
    assert spec['wait_limit'] == 2.0
    assert spec['wait_attachment'] == 0.25
    assert spec['wait_name'] == 'expon'
    assert spec.get('wait_conditional', True) is True


def test_wait_layer_unconditional(parser):
    name, spec = _spec(parser,
                       'agg L2 10 years dsev [1] wait 2 xs 0.25 expon !')
    assert spec['wait_conditional'] is False
    assert spec['wait_limit'] == 2.0
    assert spec['wait_attachment'] == 0.25


def test_wait_layer_vector(parser):
    name, spec = _spec(parser, 'agg L3 5 years dsev [1] '
                               'wait [1 2] xs [0 0.5] expon wts [.6 .4]')
    assert np.array_equal(spec['wait_limit'], [1.0, 2.0])
    assert np.array_equal(spec['wait_attachment'], [0.0, 0.5])


def test_wait_layer_label(parser):
    name, spec = _spec(parser, 'agg L4 10 years dsev [1] '
                               'wait 2 xs 0.25 expon as capped')
    assert spec['label_map'] == {'wait': 'capped'}
    assert spec['wait_limit'] == 2.0


def test_wait_layer_splice_rejected(parser):
    with pytest.raises(ValueError, match='splice'):
        parser.parse('agg L5 10 years dsev [1] '
                     'wait 2 xs 0.25 expon splice [0 3]')


def test_wait_layer_dwait_not_grammatical(parser):
    # the layered alternative takes a sev subtree; dwait has no layer form
    with pytest.raises(ValueError):
        parser.parse('agg L6 10 years dsev [1] wait 2 xs 1 dwait [1 2]')
