"""Regression net for ``aggregate.underwriter.Underwriter``.

Pins the observable surface (build / interpret_program / __getitem__ /
read_database) before the refactor lands and as the contract afterwards.
Anything that changes here is an intentional, documented behavior change.
"""

import pytest

from aggregate import build as global_build
from aggregate.distributions import Aggregate, Severity
from aggregate.portfolio import Portfolio
from aggregate.underwriter import Underwriter


# ---------------------------------------------------------------------------
# build() — single-output shape
# ---------------------------------------------------------------------------

def test_build_aggregate_returns_aggregate():
    obj = global_build('agg PhaseZero:Dice dfreq [3] dsev [1:6]')
    assert isinstance(obj, Aggregate)
    assert hasattr(obj, 'density_df')
    assert obj.agg_m > 0


def test_build_severity_returns_severity():
    obj = global_build('sev PhaseZero:S lognorm 100 cv 1')
    assert isinstance(obj, Severity)


def test_build_portfolio_returns_portfolio():
    program = (
        'port PhaseZero:P\n'
        '\tagg A1 1 claim sev lognorm 10 cv 1 fixed\n'
        '\tagg A2 1 claim sev lognorm 20 cv 1 fixed'
    )
    obj = global_build(program)
    assert isinstance(obj, Portfolio)


def test_build_distortion_returns_distortion():
    from aggregate.spectral import Distortion
    obj = global_build('distortion PhaseZero:D ph 0.5')
    assert isinstance(obj, Distortion)


# ---------------------------------------------------------------------------
# build() — multi-output shape (PIN; will be updated when S2 lands)
# ---------------------------------------------------------------------------

def test_build_multi_output_contract():
    """build() raises ValueError for multi-output, directing to build_many."""
    program = (
        'agg PhaseZero:Multi1 1 claim sev lognorm 10 cv 1 fixed\n'
        'agg PhaseZero:Multi2 1 claim sev lognorm 20 cv 1 fixed'
    )
    with pytest.raises(ValueError, match='build_many'):
        global_build(program)


def test_build_many_returns_list():
    """build_many always returns the full list of ParsedProgram, regardless of count."""
    program = (
        'agg PhaseZero:Many1 1 claim sev lognorm 10 cv 1 fixed\n'
        'agg PhaseZero:Many2 1 claim sev lognorm 20 cv 1 fixed'
    )
    rv = global_build.build_many(program)
    assert isinstance(rv, list)
    assert len(rv) == 2
    assert {r.name for r in rv} == {'PhaseZero:Many1', 'PhaseZero:Many2'}


# ---------------------------------------------------------------------------
# __getitem__ / knowledge
# ---------------------------------------------------------------------------

def test_getitem_returns_parsed_program():
    uw = Underwriter()
    uw.interpret_program('sev PhaseZero:One dsev [1]')
    rv = uw['sev', 'PhaseZero:One']
    assert rv.kind == 'sev'
    assert rv.name == 'PhaseZero:One'
    assert rv.program.startswith('sev PhaseZero:One')


def test_interpret_program_returns_list_fills_knowledge():
    uw = Underwriter()
    rv = uw.interpret_program('agg PhaseZero:IP 1 claim sev lognorm 10 cv 1 fixed')
    assert isinstance(rv, list)
    assert len(rv) == 1
    parsed = rv[0]
    assert parsed.kind == 'agg'
    assert parsed.name == 'PhaseZero:IP'
    assert parsed.object is None  # not built yet
    assert ('agg', 'PhaseZero:IP') in uw._knowledge.index


# ---------------------------------------------------------------------------
# database loading
# ---------------------------------------------------------------------------

def test_read_database_populates_knowledge():
    uw = Underwriter()
    uw.read_database('test_suite')
    assert len(uw._knowledge) >= 140


def test_read_databases_is_idempotent():
    """Calling read_databases twice should not error and should not double-populate."""
    uw = Underwriter(databases='test_suite')
    _ = uw.knowledge  # triggers first read
    n1 = len(uw._knowledge)
    uw.read_databases()  # second call
    n2 = len(uw._knowledge)
    assert n1 == n2 >= 140


# ---------------------------------------------------------------------------
# __repr__ — sanity (drops "help" block after S5)
# ---------------------------------------------------------------------------

def test_repr_is_multiline_and_includes_identity():
    uw = Underwriter(name='PhaseZeroTest')
    s = repr(uw)
    assert 'PhaseZeroTest' in s
    assert '\n' in s
