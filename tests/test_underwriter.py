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
    uw._interpret_program('sev PhaseZero:One dsev [1]')
    rv = uw['sev', 'PhaseZero:One']
    assert rv.kind == 'sev'
    assert rv.name == 'PhaseZero:One'
    assert rv.program.startswith('sev PhaseZero:One')


def test_interpret_program_returns_list_fills_knowledge():
    """The private _interpret_program API is exercised here as a regression test —
    it's the parse-only path that fills the knowledge base without constructing objects."""
    uw = Underwriter()
    rv = uw._interpret_program('agg PhaseZero:IP 1 claim sev lognorm 10 cv 1 fixed')
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
# __repr__ — sanity
# ---------------------------------------------------------------------------

def test_repr_is_multiline_and_includes_identity():
    uw = Underwriter(name='PhaseZeroTest')
    s = repr(uw)
    assert 'PhaseZeroTest' in s
    assert '\n' in s
    # no embedded help block
    assert 'build.knowledge' not in s
    assert 'build.qshow' not in s


def test_repr_lazy_load_pending():
    """When databases are configured but not yet loaded, repr should say so."""
    uw = Underwriter(databases='test_suite')
    s = repr(uw)
    assert '0 loaded' in s
    # touch knowledge to trigger load
    _ = uw.knowledge
    s2 = repr(uw)
    assert '0 loaded' not in s2
    assert 'programs' in s2


# ---------------------------------------------------------------------------
# interpret_file — bug fix pin + happy path
# ---------------------------------------------------------------------------

def test_interpret_file_runs_clean():
    """interpret_file() with no args should parse the bundled test_suite.agg cleanly."""
    df = global_build.interpret_file()
    assert df.error.sum() == 0
    assert len(df) >= 140


# ---------------------------------------------------------------------------
# discover — replaces qshow/qlist/show
# ---------------------------------------------------------------------------

def test_discover_default_lists():
    """discover() with no plot/describe is a lightweight DataFrame view."""
    df = global_build.discover('^A\\.')
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    assert 'program' in df.columns
    assert len(df) > 0


def test_discover_empty_regex_lists_all():
    df = global_build.discover()
    assert len(df) > 0


def test_discover_describe_handles_severity():
    """Severity in knowledge should not crash discover(describe=True)."""
    global_build('sev DiscSevTest lognorm 100 cv 1')
    df = global_build.discover('DiscSevTest', describe=True)
    assert 'DiscSevTest' in df.index
    # Severity has theoretical moments but no log2/bs/est_*/valid
    assert df.loc['DiscSevTest', 'agg_m'] is not None
    assert df.loc['DiscSevTest', 'log2'] is None
    assert df.loc['DiscSevTest', 'emp_m'] is None
    assert df.loc['DiscSevTest', 'valid'] is None


def test_discover_describe_handles_distortion():
    """Distortion in knowledge should not crash discover(describe=True)."""
    global_build('distortion DiscDistTest ph 0.3')
    df = global_build.discover('DiscDistTest', describe=True)
    assert 'DiscDistTest' in df.index
    # Distortion has none of the moment fields
    for col in ['log2', 'bs', 'agg_m', 'agg_cv', 'emp_m', 'valid']:
        assert df.loc['DiscDistTest', col] is None


def test_discover_plot_handles_all_kinds():
    """discover(plot=True) must not crash on any kind in the knowledge.

    Pins the regression where Distortion.plot rejected the hardcoded
    figsize=(8, 2.4) we used to pass.
    """
    import matplotlib
    matplotlib.use('Agg')  # headless backend; no figure window
    import matplotlib.pyplot as plt

    global_build('sev DiscPlotSev lognorm 100 cv 1')
    global_build('distortion DiscPlotDist ph 0.3')
    # Distortion path — must not raise
    df = global_build.discover('DiscPlotDist', plot=True)
    assert 'DiscPlotDist' in df.index
    # Severity path
    df = global_build.discover('DiscPlotSev', plot=True)
    assert 'DiscPlotSev' in df.index
    plt.close('all')


# ---------------------------------------------------------------------------
# directory rationalization — user_dir replaces site/case/template
# ---------------------------------------------------------------------------

def test_user_dir_path():
    from pathlib import Path
    uw = Underwriter()
    assert uw.user_dir == Path.home() / '.aggregate'


def test_databases_site_raises_with_migration_hint():
    """Old `databases='site'` should fail loudly, directing the user to 'user'."""
    uw = Underwriter(databases='site')
    with pytest.raises(ValueError, match="'user'"):
        uw.read_databases()


def test_dropped_properties_no_longer_exist():
    uw = Underwriter()
    assert not hasattr(uw, 'site_dir')
    assert not hasattr(uw, 'case_dir')
    assert not hasattr(uw, 'template_dir')
