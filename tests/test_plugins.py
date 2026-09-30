"""``aggregate.plugins``: discovery, provenance, and the refusals.

What is being tested is a **process-global** mechanism: a plugin's registrations
land in :data:`aggregate.charts.CHARTS` and :data:`aggregate.exhibits.EXHIBITS`,
and neither registry has a removal API. Every test here therefore registers
inside a ``try`` and deletes its keys in the ``finally``, the same idiom
``tests/test_charts_ir.py`` and ``tests/test_exhibits.py`` already use. Reaching
into those dicts is a test's privilege, not a supported operation.

The window matters, not just the cleanup. Under xdist a worker may be handed a
test from another module between two tests of this one, and
``test_charts_ir.py`` asserts a sweep over the *whole* chart registry, so a
fixture plugin left registered across tests would flake it. Registration is kept
inside a single test for that reason: no module-scoped load.

The fixture plugins live in ``tests/plugin_fixtures/`` and are reached through
``AGGREGATE_PLUGINS``, the environment on-ramp, since that route needs no
installed distribution.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from aggregate import build, plugins
from aggregate.charts import CHARTS, available_charts
from aggregate.exhibits import EXHIBITS, available_exhibits, exhibit_frames
from aggregate.plugins import ENV_NO_PLUGINS, ENV_PLUGINS, LoadedPlugin, PluginLeaf

from tests.plugin_fixtures import good as good_fixture
from tests.plugin_fixtures._toy import Toy

REPO_ROOT = Path(__file__).resolve().parents[1]

GOOD = 'tests.plugin_fixtures.good'
RAISES = 'tests.plugin_fixtures.raises'
COLLIDES = 'tests.plugin_fixtures.collides'
REUSES = 'tests.plugin_fixtures.reuses'
UNDECLARED = 'tests.plugin_fixtures.undeclared'


@pytest.fixture(autouse=True)
def _clean_loader():
    """Leave the loader unloaded on both sides of every test in this module."""
    plugins.reset_plugins()
    yield
    plugins.reset_plugins()


def _load(*module_paths, **kwargs):
    """Load the named fixture modules through the environment route."""
    env = {ENV_PLUGINS: ','.join(module_paths)}
    env.update(kwargs.pop('extra_env', {}))
    return plugins.load(env=env, force=True, **kwargs)


@pytest.fixture
def toy():
    return Toy('fixture')


# --- no auto-load ------------------------------------------------------------

def test_bare_import_loads_nothing():
    """``import aggregate`` must not run a plugin, however many are configured.

    A subprocess rather than an in-process check: the assertion is about what
    importing does, and this process imported the library long ago.
    """
    code = (
        'import aggregate;'
        'assert aggregate.plugins.loaded_plugins() == [], '
        '"import aggregate ran discovery";'
        'from aggregate.charts import CHARTS;'
        f'assert {good_fixture.CHART_NAME!r} not in CHARTS, "a plugin registered on import";'
        'from aggregate.exhibits import EXHIBITS;'
        f'assert {good_fixture.EXHIBIT_NAME!r} not in EXHIBITS, "a plugin registered on import"'
    )
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT,
                         capture_output=True, text=True,
                         env={**_subprocess_env(), ENV_PLUGINS: GOOD})
    assert out.returncode == 0, out.stderr


def _subprocess_env():
    """A copy of this process's environment with the plugin variables cleared."""
    import os
    env = dict(os.environ)
    env.pop(ENV_PLUGINS, None)
    env.pop(ENV_NO_PLUGINS, None)
    return env


def test_import_aggregate_does_not_pull_in_charts_or_exhibits():
    """Binding ``plugins`` on the package must not drag either provisional package in.

    ``plugins`` imports both only inside ``load()``, so this is the property
    that lets ``__init__`` bind the submodule at all.
    """
    code = (
        'import sys, aggregate;'
        'assert "aggregate.charts" not in sys.modules, "plugins pulled in charts";'
        'assert "aggregate.exhibits" not in sys.modules, "plugins pulled in exhibits"'
    )
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT,
                         capture_output=True, text=True, env=_subprocess_env())
    assert out.returncode == 0, out.stderr


def test_no_plugins_env_wins():
    """``AGGREGATE_NO_PLUGINS`` suppresses discovery whatever else is configured."""
    loaded = _load(GOOD, extra_env={ENV_NO_PLUGINS: '1'})
    assert loaded == []
    assert good_fixture.CHART_NAME not in CHARTS
    assert good_fixture.EXHIBIT_NAME not in EXHIBITS


@pytest.mark.parametrize('value', ['', '0', 'false', 'off', 'no'])
def test_no_plugins_reads_false_as_off(value):
    """An explicitly falsey kill switch is not a kill switch."""
    try:
        loaded = _load(GOOD, extra_env={ENV_NO_PLUGINS: value})
        assert [p.name for p in loaded] == [GOOD]
    finally:
        _forget_good()


# --- the happy path ----------------------------------------------------------

def _forget_good():
    """Undo the good fixture's registrations. A test's privilege, not an API."""
    CHARTS.pop(good_fixture.CHART_NAME, None)
    EXHIBITS.pop(good_fixture.EXHIBIT_NAME, None)


def test_good_plugin_registers_and_is_attributed(toy):
    """One plugin, one chart, one exhibit, attributed by the load snapshot."""
    before_charts, before_exhibits = set(CHARTS), set(EXHIBITS)
    try:
        loaded = _load(GOOD)
        assert len(loaded) == 1
        plugin, = loaded
        assert isinstance(plugin, LoadedPlugin)
        assert plugin.ok and plugin.error is None
        assert plugin.name == GOOD and plugin.source == 'env'

        # provenance is the before-and-after difference, not a parsed name
        assert plugin.charts == (good_fixture.CHART_NAME,)
        assert plugin.exhibits == (good_fixture.EXHIBIT_NAME,)
        assert set(CHARTS) - before_charts == {good_fixture.CHART_NAME}
        assert set(EXHIBITS) - before_exhibits == {good_fixture.EXHIBIT_NAME}

        # the leaves carry the plugin's own strings, in its own order
        assert [leaf.kind for leaf in plugin.leaves] == ['exhibit', 'chart']
        assert [leaf.label for leaf in plugin.leaves] == ['Toy table', 'Toy chart']
        assert all(leaf.why for leaf in plugin.leaves)

        # capability derives, so the new leaves light for the plugin's own type
        assert good_fixture.EXHIBIT_NAME in [n for n, _ in available_exhibits(toy)]
        assert good_fixture.CHART_NAME in available_charts(toy)

        # and both actually serve
        name, frame, kw = exhibit_frames(toy, good_fixture.EXHIBIT_NAME)[0]
        assert name == 'toy_df' and len(frame) == 2
        assert 'caption' in kw
        doc = CHARTS[good_fixture.CHART_NAME].emitter(toy)
        assert doc.name == good_fixture.CHART_NAME
        assert doc.title == 'Toy: fixture'
    finally:
        _forget_good()
    assert set(CHARTS) == before_charts and set(EXHIBITS) == before_exhibits


def test_a_plugin_does_not_widen_a_real_object(toy):
    """A plugin registered for its own type leaves every other object alone."""
    dice = build('agg PluginDice dfreq [3] dsev [1:6]')
    before_charts = set(available_charts(dice))
    before_exhibits = {n for n, _ in available_exhibits(dice)}
    try:
        assert _load(GOOD)[0].ok
        assert set(available_charts(dice)) == before_charts
        assert {n for n, _ in available_exhibits(dice)} == before_exhibits
    finally:
        _forget_good()


def test_load_is_idempotent():
    """A second ``load`` returns the first one's answer and registers nothing twice."""
    try:
        first = _load(GOOD)
        second = plugins.load()
        assert [p.name for p in second] == [p.name for p in first]
        assert all(p.ok for p in second)
        assert sum(1 for k in CHARTS if k == good_fixture.CHART_NAME) == 1
    finally:
        _forget_good()


def test_loaded_plugins_returns_a_copy():
    """A caller cannot edit the loader's record by editing what it handed back."""
    try:
        _load(GOOD)
        record = plugins.loaded_plugins()
        record.clear()
        assert len(plugins.loaded_plugins()) == 1
    finally:
        _forget_good()


def test_allow_is_an_allowlist():
    """``allow`` admits only the named plugins, for the hosted case."""
    assert _load(GOOD, allow=[]) == []
    assert good_fixture.CHART_NAME not in CHARTS
    try:
        assert [p.name for p in _load(GOOD, allow=[GOOD])] == [GOOD]
    finally:
        _forget_good()


# --- the refusals ------------------------------------------------------------

def test_raising_plugin_is_recorded_not_fatal():
    """A broken experiment is a recorded traceback, and ``build`` still works."""
    loaded = _load(RAISES)
    plugin, = loaded
    assert not plugin.ok
    assert 'this plugin is deliberately broken' in plugin.error
    assert 'Traceback' in plugin.error
    assert plugin.leaves == () and plugin.charts == () and plugin.exhibits == ()
    # the whole point: one bad plugin takes nothing else down
    assert build('agg PluginStillWorks dfreq [1] dsev [1]').est_m == 1


def test_missing_register_callable_is_recorded():
    """A module with no ``register`` is a recorded failure, not an AttributeError."""
    plugin, = _load('tests.plugin_fixtures._toy')
    assert not plugin.ok
    assert 'no callable register()' in plugin.error


def test_unimportable_module_is_recorded():
    """So is a module that is not there at all."""
    plugin, = _load('tests.plugin_fixtures.no_such_module')
    assert not plugin.ok
    assert 'ModuleNotFoundError' in plugin.error


def test_a_library_exhibit_name_is_refused():
    """A plugin may not claim a name the library already owns."""
    plugin, = _load(COLLIDES)
    assert not plugin.ok
    assert 'already registered' in plugin.error
    assert "'summary'" in plugin.error
    assert plugin.leaves == ()


def test_another_plugins_exhibit_name_is_refused():
    """Nor one an earlier plugin owns, which is the case that would merge silently.

    ``register_simple_exhibit`` extends rather than raises, deliberately, so the
    loader is what refuses this. The extension is not unwound: dropping the key
    in the cleanup takes the whole exhibit, ``good``'s registration included.
    """
    try:
        first, second = _load(GOOD, REUSES)
        assert first.name == GOOD and first.ok
        assert second.name == REUSES and not second.ok
        assert 'already registered' in second.error
        assert good_fixture.EXHIBIT_NAME in second.error
        assert second.leaves == ()
    finally:
        _forget_good()


def test_leaf_naming_an_unregistered_key_is_refused():
    """A plugin that describes a document it forgot to register is rejected."""
    plugin, = _load(UNDECLARED)
    assert not plugin.ok
    assert 'did not register' in plugin.error
    assert 'toy_phantom' in plugin.error


def test_two_plugins_load_in_name_order():
    """Discovery order is plugin name alphabetically, whatever the variable says."""
    try:
        loaded = _load(RAISES, GOOD, UNDECLARED)
        assert [p.name for p in loaded] == sorted([RAISES, GOOD, UNDECLARED])
    finally:
        _forget_good()


# --- the leaf dataclass ------------------------------------------------------

def test_plugin_leaf_refuses_an_unknown_kind():
    """A leaf is a chart or an exhibit; a plugin contributes no app behavior."""
    with pytest.raises(ValueError, match='not one of'):
        PluginLeaf(name='x', kind='flag', label='Quick QS')


def test_plugin_leaf_is_frozen():
    leaf = PluginLeaf(name='x', kind='chart', label='X')
    with pytest.raises(AttributeError):
        leaf.label = 'Y'


# --- the environment allow-list ----------------------------------------------

def test_config_does_not_warn_about_the_plugin_variables(recwarn):
    """``AGGREGATE_*`` is an allow-list, and these two are not settings.

    Without the exemption in ``config._ENV_IGNORED`` every settings load with a
    plugin variable set would emit "Unknown environment variable", which is
    noise on the one code path a host is guaranteed to run.
    """
    from aggregate import config
    config.load_settings(env={ENV_PLUGINS: GOOD, ENV_NO_PLUGINS: '1'})
    messages = [str(w.message) for w in recwarn]
    assert not [m for m in messages if 'Unknown environment variable' in m], messages


# --- the packaged route ------------------------------------------------------

def test_entry_point_route_is_discovered_and_versioned(monkeypatch):
    """The packaged route, with a stub entry point standing in for a distribution.

    The environment route the other tests use needs no installed distribution,
    which is why it is the one they use; this covers the branch a real plugin
    actually ships on, including where the version and the plugin name come
    from.
    """
    class _Dist:
        version = '9.9.9'

    class _EntryPoint:
        name = 'toyplugin'
        dist = _Dist()

        def load(self):
            return good_fixture.register

    monkeypatch.setattr(plugins, 'entry_points', lambda group=None: [_EntryPoint()])
    try:
        plugin, = plugins.load(env={}, force=True)
        assert plugin.ok
        assert plugin.name == 'toyplugin' and plugin.source == 'entry_point'
        assert plugin.version == '9.9.9'
        assert plugin.charts == (good_fixture.CHART_NAME,)
        assert plugin.exhibits == (good_fixture.EXHIBIT_NAME,)
    finally:
        _forget_good()


def test_entry_point_wins_over_the_same_name_in_the_environment(monkeypatch):
    """The packaged route is the destination; the variable is the experiment."""
    class _EntryPoint:
        name = GOOD
        dist = None

        def load(self):
            return good_fixture.register

    monkeypatch.setattr(plugins, 'entry_points', lambda group=None: [_EntryPoint()])
    try:
        plugin, = _load(GOOD)
        assert plugin.source == 'entry_point'
    finally:
        _forget_good()
