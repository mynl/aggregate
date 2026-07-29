"""Regression net for ``aggregate.underwriter.Underwriter``.

Pins the observable surface (build / interpret_program / __getitem__ /
load / reload / resolve_databases / available_databases / to_agg) before the
refactor lands and as the contract afterwards. Anything that changes here is an
intentional, documented behavior change.
"""

import logging
from pathlib import Path

import pytest

from aggregate import build as global_build
from aggregate.bivariate import BivariateAggregate
from aggregate.distributions import Aggregate, Severity
from aggregate.portfolio import Portfolio
from aggregate.spectral import Distortion
from aggregate.underwriter import Underwriter


# ---------------------------------------------------------------------------
# build() — single-output shape
# ---------------------------------------------------------------------------

def test_build_aggregate_returns_aggregate():
    obj = global_build('agg PhaseZero:Dice dfreq [3] dsev [1:6]')
    assert isinstance(obj, Aggregate)
    assert hasattr(obj, 'density_df')
    assert obj.actual_m > 0


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
    obj = global_build('distortion PhaseZero:D ph 0.5')
    assert isinstance(obj, Distortion)


# ---------------------------------------------------------------------------
# build() — multi-output shape (PIN; will be updated when S2 lands)
# ---------------------------------------------------------------------------

def test_build_multi_output_contract():
    """build() raises ValueError for multi-output, directing to build_many."""
    program = (
        'agg PhaseZero:Multi1 1 claim sev lognorm 10 cv 1 fixed\n\n'
        'agg PhaseZero:Multi2 1 claim sev lognorm 20 cv 1 fixed'
    )
    with pytest.raises(ValueError, match='build_many'):
        global_build(program)


def test_build_many_returns_list():
    """build_many always returns the full list of Recipe, regardless of count."""
    program = (
        'agg PhaseZero:Many1 1 claim sev lognorm 10 cv 1 fixed\n\n'
        'agg PhaseZero:Many2 1 claim sev lognorm 20 cv 1 fixed'
    )
    rv = global_build.build_many(program)
    assert isinstance(rv, list)
    assert len(rv) == 2
    assert {r.name for r in rv} == {'PhaseZero:Many1', 'PhaseZero:Many2'}


# ---------------------------------------------------------------------------
# build_many's output dispatch: every kind must be finished quietly
# ---------------------------------------------------------------------------

#: One program per object type ``_factory`` can produce, paired with the class
#: build must hand back. The bivariate programs are the ones in
#: ``src/aggregate/agg/decl-testers.agg`` (``MV.Indep``, ``MV.NetCeded``,
#: ``MV.DBVUniform``, ``MV.Clash``), reused verbatim.
_ONE_PER_KIND = [
    ('agg', 'agg PhaseZero:Q.Agg dfreq [1:3] dsev [1:6]', Aggregate),
    ('sev', 'sev PhaseZero:Q.Sev lognorm 10 cv 1.2', Severity),
    ('distortion', 'distortion PhaseZero:Q.D ph 0.5', Distortion),
    ('port', 'port PhaseZero:Q.P agg Q.PA dfreq [1] dsev [1:6] '
             'agg Q.PB dfreq [1] dsev [1:6]', Portfolio),
    # a pnl is its deferred inner engine until _snapshot_pnl runs
    ('pnl', 'pnl PhaseZero:Q.PnL 100 prem less agg PhaseZero:Q.PnL_e '
            '10 claims sev lognorm 8 cv 1.5 poisson', Aggregate),
    ('bvagg copula', 'bivariate MV.Indep 25 claims '
                     'agg A dfreq [0 1] [.5 .5] sev lognorm 50 cv 1.5 '
                     'agg B dfreq [0 1] [.5 .5] sev gamma 50 cv 1.0 poisson',
     BivariateAggregate),
    ('bvagg netceded', 'netceded agg MV.NetCeded 8 claims sev 300 * beta 2 3 '
                       'occurrence net of 0.7 so 60 xs 40 poisson',
     BivariateAggregate),
    ('bvagg dbvsev', 'bivariate MV.DBVUniform 5 claims dbvsev [0 1 2] [0 5 10]',
     BivariateAggregate),
    ('bvagg clash', 'clash MV.Clash 8 5 2 claims sev lognorm 50 cv 1.2 '
                    'sev lognorm 60 cv 1.5 mixed gamma 0.2',
     BivariateAggregate),
]


@pytest.mark.parametrize('kind, program, expected', _ONE_PER_KIND,
                         ids=[k for k, _p, _e in _ONE_PER_KIND])
def test_build_update_false_is_quiet(kind, program, expected, caplog):
    """``update=False`` constructs every kind without complaining.

    ``build_many``'s dispatch used to gate the ``BivariateAggregate`` branch on
    ``update is True`` while the no-op escape hatch listed only
    ``(Aggregate, Portfolio)``, so a bivariate built with ``update=False`` fell
    through to the catch-all and logged ``Unexpected: output kind is ...``. The
    object was fine; the message was not.
    """
    with caplog.at_level(logging.WARNING, logger='aggregate.underwriter'):
        obj = global_build(program, update=False)
    assert isinstance(obj, expected)
    assert not [r for r in caplog.records if 'Unexpected' in r.getMessage()]


def test_build_truthy_update_updates():
    """``update=1`` is honored, not silently dropped.

    The branches used to test ``update is True`` by identity, so any truthy
    value that was not the singleton skipped the update *and* tripped the
    catch-all warning.
    """
    obj = global_build('agg PhaseZero:Q.Truthy dfreq [1:3] dsev [1:6]', update=1)
    assert obj.density_df is not None
    assert obj.est_m > 0


# ---------------------------------------------------------------------------
# expr: a bare expression is an answer, not a declaration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program, expected', [
    ('3', 3.0),
    ('-4', -4.0),
    ('2 ** 3', 8.0),
    ('6 / 2', 3.0),
    ('exp(1)', 2.718281828459045),
])
def test_build_expr_returns_its_value(program, expected):
    """``build('3')`` evaluates to 3.0 rather than raising.

    Only the forms the top-level ``expr`` production accepts are covered.
    ``1 + 2`` and a trailing ``2 * 3`` do not parse there, since ``*`` is the
    severity scale operator, but that is a separate grammar question.
    """
    assert global_build(program) == pytest.approx(expected)


def test_build_expr_is_not_stored():
    """An expression leaves no recipe behind.

    It used to: the entry was written before ``_factory`` raised
    ``Cannot build expr objects``, and the orphan then broke ``to_agg``, whose
    renderer has no ``expr`` case.
    """
    uw = Underwriter(databases=None)
    uw('2 ** 5')
    assert not [k for k in uw._recipes if k[0] == 'expr']


# ---------------------------------------------------------------------------
# __getitem__ / recipe base
# ---------------------------------------------------------------------------

def test_getitem_returns_parsed_program():
    uw = Underwriter()
    uw._interpret_program('sev PhaseZero:One dsev [1]')
    rv = uw['sev', 'PhaseZero:One']
    assert rv.kind == 'sev'
    assert rv.name == 'PhaseZero:One'
    assert rv.program.startswith('sev PhaseZero:One')


def test_interpret_program_returns_list_fills_recipe_base():
    """The private _interpret_program API is exercised here as a regression test —
    it's the parse-only path that fills the recipe base without constructing objects."""
    uw = Underwriter()
    rv = uw._interpret_program('agg PhaseZero:IP 1 claim sev lognorm 10 cv 1 fixed')
    assert isinstance(rv, list)
    assert len(rv) == 1
    parsed = rv[0]
    assert parsed.kind == 'agg'
    assert parsed.name == 'PhaseZero:IP'
    assert parsed.object is None  # not built yet
    assert ('agg', 'PhaseZero:IP') in uw._recipes


# ---------------------------------------------------------------------------
# database loading
# ---------------------------------------------------------------------------

def test_load_populates_recipe_base():
    uw = Underwriter()
    uw.load('_test_suite')
    assert len(uw._recipes) >= 140


def test_load_is_idempotent():
    """The configured lazy load runs once; a second load() is a no-op."""
    uw = Underwriter(databases='_test_suite')
    _ = uw.recipes  # triggers first (configured) read
    n1 = len(uw._recipes)
    assert uw.load() == []  # second call: already loaded
    n2 = len(uw._recipes)
    assert n1 == n2 >= 140


def test_load_missing_literal_raises():
    """An explicit load() of a literal name that does not exist raises."""
    uw = Underwriter(databases=None)
    with pytest.raises(FileNotFoundError):
        uw.load('definitely_not_a_real_database')


def test_load_glob_no_match_warns_not_raises(caplog):
    """A glob that matches nothing warns and loads nothing (no raise)."""
    import logging
    uw = Underwriter(databases=None)
    with caplog.at_level(logging.WARNING, logger='aggregate.underwriter'):
        read = uw.load('zzz_no_such_prefix_*')
    assert read == []
    assert any('matched no files' in r.getMessage() for r in caplog.records)


def test_databases_reports_loaded_paths():
    """`databases` is the list of resolved Paths actually read."""
    uw = Underwriter(databases='_test_suite')
    _ = uw.recipes
    assert len(uw.databases) == 1
    assert isinstance(uw.databases[0], Path)
    assert uw.databases[0].name == '_test_suite.agg'


def test_resolve_databases_matches_load():
    """resolve_databases previews exactly what load reads (for files that exist)."""
    uw = Underwriter(databases=None)
    preview = uw.resolve_databases('_test_suite')
    assert len(preview) == 1 and preview[0].name == '_test_suite.agg'
    read = uw.load('_test_suite')
    assert [p.name for p in read] == [p.name for p in preview]


def test_available_databases_discovers_bundled():
    """available_databases lists the bundled _test_suite among on-disk files."""
    uw = Underwriter(databases=None)
    df = uw.available_databases()
    assert 'name' in df.columns and 'where' in df.columns and 'path' in df.columns
    assert '_test_suite' in set(df['name'])


def test_reload_resets_to_as_created():
    """reload drops in-session builds and ad-hoc loads, restoring the request."""
    uw = Underwriter(databases='_test_suite')
    _ = uw.recipes
    n0 = len(uw._recipes)
    uw.build('agg ReloadMe 1 claim sev lognorm 10 cv 1 fixed', update=False)
    assert ('agg', 'ReloadMe') in uw._recipes
    uw.reload()
    assert ('agg', 'ReloadMe') not in uw._recipes
    assert len(uw._recipes) == n0


def test_source_provenance():
    """Loaded entries carry their file Path; in-session builds carry 'session'."""
    uw = Underwriter(databases='_test_suite')
    _ = uw.recipes
    # pick any loaded entry — its source is the _test_suite file Path
    loaded = next(iter(uw._recipes.values()))
    assert isinstance(loaded.source, Path)
    uw.build('agg SessionSrc 1 claim sev lognorm 10 cv 1 fixed', update=False)
    assert uw._recipes[('agg', 'SessionSrc')].source == 'session'


def test_to_agg_round_trip(tmp_path):
    """to_agg writes session builds; a fresh Underwriter reloads the same specs."""
    uw = Underwriter(databases=None)
    uw.build('agg RT:One 1 claim sev lognorm 10 cv 1 fixed', update=False)
    uw.build('sev RT:Two lognorm 5 cv 0.5', update=False)
    out = uw.to_agg(tmp_path / 'mybook')  # absolute -> used as given
    assert out.exists() and out.suffix == '.agg'

    uw2 = Underwriter(databases=None)
    uw2.load(out)
    assert ('agg', 'RT:One') in uw2._recipes
    assert ('sev', 'RT:Two') in uw2._recipes
    assert (uw2._recipes[('agg', 'RT:One')].spec
            == uw._recipes[('agg', 'RT:One')].spec)


def test_to_agg_kind_filter(tmp_path):
    """to_agg(kind=...) restricts the export to one kind."""
    uw = Underwriter(databases=None)
    uw.build('agg KF:A 1 claim sev lognorm 10 cv 1 fixed', update=False)
    uw.build('sev KF:S lognorm 5 cv 0.5', update=False)
    out = uw.to_agg(tmp_path / 'aggsonly', kind='agg')
    uw2 = Underwriter(databases=None)
    uw2.load(out)
    assert ('agg', 'KF:A') in uw2._recipes
    assert ('sev', 'KF:S') not in uw2._recipes


def test_to_agg_writes_dependencies_first(tmp_path):
    """A named reference must be written before the entry that uses it, so the
    file re-loads sequentially (sev before the agg referencing sev.X)."""
    uw = Underwriter(databases=None)
    # build the agg AFTER the sev so insertion order is sev, agg; the bug was
    # that to_agg re-sorts to (kind, name) and 'agg' < 'sev', inverting them.
    uw.build_many('sev Dep:Sev lognorm 10 cv 1', update=False)
    uw.build('agg Dep:Agg 100 claims sev.Dep:Sev poisson', update=False)
    out = uw.to_agg(tmp_path / 'deps')

    text = out.read_text(encoding='utf-8')
    assert text.index('sev Dep:Sev') < text.index('agg Dep:Agg')

    # and it must actually re-load without a missing-reference error
    uw2 = Underwriter(databases=None)
    uw2.load(out)
    assert ('sev', 'Dep:Sev') in uw2._recipes
    assert ('agg', 'Dep:Agg') in uw2._recipes


def test_to_agg_default_mode_x_raises_on_existing(tmp_path):
    """Default mode 'x' is safe: it refuses to clobber an existing file."""
    uw = Underwriter(databases=None)
    uw.build('agg X:1 1 claim sev lognorm 10 cv 1 fixed', update=False)
    uw.to_agg(tmp_path / 'book')                       # creates
    with pytest.raises(FileExistsError):
        uw.to_agg(tmp_path / 'book')                   # exists -> raise


def test_to_agg_mode_w_overwrites(tmp_path):
    """mode='w' replaces the file's contents."""
    uw = Underwriter(databases=None)
    uw.build('agg W:1 1 claim sev lognorm 10 cv 1 fixed', update=False)
    out = uw.to_agg(tmp_path / 'book')
    uw2 = Underwriter(databases=None)
    uw2.build('agg W:2 1 claim sev lognorm 10 cv 1 fixed', update=False)
    uw2.to_agg(out, mode='w')                          # absolute path, overwrite
    text = out.read_text(encoding='utf-8')
    assert 'W:2' in text and 'W:1' not in text


def test_to_agg_mode_a_appends_dated_block_and_round_trips(tmp_path):
    """mode='a' adds a dated block at the end; a dependency written earlier in
    the file still precedes the appended entry that references it."""
    uw = Underwriter(databases=None)
    uw.build_many('sev A:Sev lognorm 10 cv 1', update=False)
    out = uw.to_agg(tmp_path / 'book', kind='sev')     # write the sev
    uw.build('agg A:Agg 100 claims sev.A:Sev poisson', update=False)
    uw.to_agg(out, kind='agg', mode='a')               # append the agg block

    text = out.read_text(encoding='utf-8')
    assert '# added' in text
    assert text.index('sev A:Sev') < text.index('agg A:Agg')

    uw2 = Underwriter(databases=None)
    uw2.load(out)
    assert ('sev', 'A:Sev') in uw2._recipes
    assert ('agg', 'A:Agg') in uw2._recipes


def test_to_agg_bad_mode_raises(tmp_path):
    uw = Underwriter(databases=None)
    uw.build('agg M:1 1 claim sev lognorm 10 cv 1 fixed', update=False)
    with pytest.raises(ValueError):
        uw.to_agg(tmp_path / 'book', mode='q')


# ---------------------------------------------------------------------------
# __repr__ — sanity
# ---------------------------------------------------------------------------

def test_repr_is_multiline_and_includes_identity():
    uw = Underwriter(name='PhaseZeroTest')
    s = repr(uw)
    assert 'PhaseZeroTest' in s
    assert '\n' in s
    # no embedded help block
    assert 'build.recipes' not in s
    assert 'build.qshow' not in s


def test_repr_lazy_load_pending():
    """When databases are configured but not yet loaded, repr should say so."""
    uw = Underwriter(databases='_test_suite')
    s = repr(uw)
    assert '0 loaded' in s
    # touch recipes to trigger load
    _ = uw.recipes
    s2 = repr(uw)
    assert '0 loaded' not in s2
    assert 'programs' in s2


def test_constructor_is_keyword_only():
    """``Underwriter('_test_suite')`` must fail, not silently name the uw."""
    with pytest.raises(TypeError):
        Underwriter('_test_suite')
    # the keyword form is the supported way to ask for a database
    uw = Underwriter(databases='_test_suite')
    assert uw._request == '_test_suite'


def test_repr_reports_request():
    """repr carries a ``requested`` line distinct from resolved databases."""
    uw = Underwriter(databases='_test_suite')
    assert 'requested          _test_suite' in repr(uw)
    # a bare underwriter requests nothing
    assert 'requested          none' in repr(Underwriter())
    # an iterable request is rendered comma-joined
    uw2 = Underwriter(databases=['_test_suite', 'site'])
    assert 'requested          _test_suite, site' in repr(uw2)


# ---------------------------------------------------------------------------
# interpret_file — bug fix pin + happy path
# ---------------------------------------------------------------------------

def test_interpret_file_runs_clean():
    """interpret_file() with no args should parse the bundled _test_suite.agg cleanly."""
    df = global_build.interpret_file()
    assert df.error.sum() == 0
    assert len(df) >= 140


# ---------------------------------------------------------------------------
# discover — replaces qshow/qlist/show
# ---------------------------------------------------------------------------

def test_discover_default_lists():
    """discover() with no plot/describe is a lightweight DataFrame view."""
    # ``^Curve`` replaces the old ``^A\.``: library.agg retired the
    # single-letter filing prefixes at 1.0.0a159 (grouping is tags{} now).
    df = global_build.discover('^Curve')
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    assert 'program' in df.columns
    assert len(df) > 0


def test_discover_by_tags():
    """``tags=`` narrows: an entry must carry EVERY tag given."""
    heroes = global_build.discover(tags='role:hero')
    assert len(heroes) > 0
    both = global_build.discover(tags='topic:severity, role:reference')
    assert 0 < len(both) <= len(global_build.discover(tags='topic:severity'))
    assert len(global_build.discover(tags='no-such-tag')) == 0


def test_discover_kind_is_the_type_filter():
    """``kind=`` filters by TYPE; ``tags=`` by subject. They compose.

    Tags deliberately never restate the object's kind (see
    ``tests/test_agg_libraries.py``), so these are two independent axes: an
    ``agg`` entry demonstrating a severity form is ``kind='agg'`` and
    ``tags='topic:severity'`` at the same time.
    """
    sevs = global_build.discover(kind='sev')
    topic = global_build.discover(tags='topic:severity')
    both = global_build.discover(kind='agg', tags='topic:severity')
    assert len(sevs) > 0 and len(topic) > 0 and len(both) > 0
    # the subject axis reaches well beyond the `sev` type
    assert len(topic) > len(sevs)


def test_discover_empty_regex_lists_all():
    df = global_build.discover()
    assert len(df) > 0


def test_discover_describe_handles_severity():
    """A Severity recipe should not crash discover(describe=True)."""
    global_build('sev DiscSevTest lognorm 100 cv 1')
    df = global_build.discover('DiscSevTest', describe=True)
    assert 'DiscSevTest' in df.index
    # Severity has theoretical moments but no log2/bs/est_*/valid
    assert df.loc['DiscSevTest', 'actual_m'] is not None
    assert df.loc['DiscSevTest', 'log2'] is None
    assert df.loc['DiscSevTest', 'emp_m'] is None
    assert df.loc['DiscSevTest', 'valid'] is None


def test_discover_describe_handles_distortion():
    """A Distortion recipe should not crash discover(describe=True)."""
    global_build('distortion DiscDistTest ph 0.3')
    df = global_build.discover('DiscDistTest', describe=True)
    assert 'DiscDistTest' in df.index
    # Distortion has none of the moment fields
    for col in ['log2', 'bs', 'actual_m', 'actual_cv', 'emp_m', 'valid']:
        assert df.loc['DiscDistTest', col] is None


def test_discover_plot_handles_all_kinds():
    """discover(plot=True) must not crash on any kind in the recipe base.

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


def test_databases_site_token_no_longer_special(caplog):
    """The removed `'site'` collection token is now just an ordinary (missing)
    filename: the configured lazy load warns and loads nothing, never raises."""
    import logging
    uw = Underwriter(databases='site')
    with caplog.at_level(logging.WARNING, logger='aggregate.underwriter'):
        _ = uw.recipes  # triggers the configured load
    assert len(uw._recipes) == 0
    assert any('site' in r.getMessage() for r in caplog.records)


def test_dropped_properties_no_longer_exist():
    uw = Underwriter()
    assert not hasattr(uw, 'site_dir')
    assert not hasattr(uw, 'case_dir')
    assert not hasattr(uw, 'template_dir')
