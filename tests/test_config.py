"""Tests for the user-editable settings layer (:mod:`aggregate.config`).

Covers the defaults -> file -> env -> kwargs cascade, the lazy singleton and
``reload_settings`` refresh of the module ``build``, ``AGGREGATE_CONFIG``
relocation / disabling, unknown-key warnings, and the annotated template.
"""

import warnings

import pytest

from aggregate import config
from aggregate import build as global_build


@pytest.fixture(autouse=True)
def _reset_settings_singleton():
    """Restore the cached settings singleton + module build after each test.

    Tests here mutate process state (env vars via monkeypatch, the cached
    singleton via reload). monkeypatch undoes the env; this fixture rebuilds
    the singleton from the now-clean environment so later tests / the session
    see the normal defaults.
    """
    yield
    config._settings = None
    config.reload_settings()


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

def test_defaults_no_file_no_env():
    s = config.load_settings(path=None, env={})
    assert s.build.log2 == 16
    assert s.build.databases == ('test_suite',)
    assert s.build.update is True
    assert s.discretization.reins_bucket == 'linear'
    assert s.discretization.dsev_bucket == 'linear'
    assert s.discretization.bucket_sizing_p == pytest.approx(0.99999)
    assert s.discretization.window_nines == 12
    assert s.validation.eps == pytest.approx(1e-4)
    assert s.validation.noise == pytest.approx(1e-12)
    assert s.multivariate.window_nines == 12
    # every source is 'default'
    assert all(src == 'default' for _, _, src in config.describe_settings(s))


# ---------------------------------------------------------------------------
# file / env / kwargs cascade
# ---------------------------------------------------------------------------

def test_file_overrides_default(tmp_path):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[build]\nlog2 = 12\n[discretization]\ndsev_bucket = "nearest"\n')
    s = config.load_settings(path=cfg, env={})
    assert s.build.log2 == 12
    assert s.sources['build.log2'] == 'config'
    assert s.discretization.dsev_bucket == 'nearest'
    assert s.sources['discretization.dsev_bucket'] == 'config'
    # untouched key still default
    assert s.build.update is True
    assert s.sources['build.update'] == 'default'


def test_env_overrides_file(tmp_path):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[build]\nlog2 = 12\n')
    s = config.load_settings(path=cfg, env={'AGGREGATE_LOG2': '18'})
    assert s.build.log2 == 18
    assert s.sources['build.log2'] == 'env'


def test_databases_env_is_comma_split():
    s = config.load_settings(path=None, env={'AGGREGATE_DATABASES': 'a, b ,c'})
    assert s.build.databases == ('a', 'b', 'c')
    assert s.sources['build.databases'] == 'env'


def test_kwargs_override_env_end_to_end(monkeypatch):
    """An explicit build(..., log2=) wins over env and config (top of cascade)."""
    monkeypatch.setenv('AGGREGATE_LOG2', '18')
    config._settings = None
    config.reload_settings()
    assert config.get_settings().build.log2 == 18
    a = global_build('agg KwTest 1 claim sev lognorm 10 cv 1 fixed', log2=8)
    assert a.log2 == 8  # the call kwarg beats the configured 18


# ---------------------------------------------------------------------------
# AGGREGATE_CONFIG relocation / disabling
# ---------------------------------------------------------------------------

def test_config_path_none_disables(tmp_path):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[build]\nlog2 = 7\n')
    # explicit path still reads, but AGGREGATE_CONFIG=none means config_path None
    assert config.config_path({'AGGREGATE_CONFIG': 'none'}) is None
    s = config.load_settings(env={'AGGREGATE_CONFIG': 'none'})
    assert s.build.log2 == 16  # file ignored


def test_config_path_relocates(tmp_path):
    cfg = tmp_path / 'elsewhere.toml'
    cfg.write_text('[build]\nlog2 = 9\n')
    env = {'AGGREGATE_CONFIG': str(cfg)}
    assert config.config_path(env) == cfg
    s = config.load_settings(env=env)
    assert s.build.log2 == 9
    assert s.sources['build.log2'] == 'config'


# ---------------------------------------------------------------------------
# unknown keys / sections / env vars warn
# ---------------------------------------------------------------------------

def test_unknown_file_key_warns_and_drops(tmp_path):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[build]\nbogus = 1\nlog2 = 14\n')
    with pytest.warns(UserWarning, match="Unknown key 'bogus'"):
        s = config.load_settings(path=cfg, env={})
    assert not hasattr(s.build, 'bogus')
    assert s.build.log2 == 14  # the valid key still applied


def test_unknown_section_warns(tmp_path):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[nope]\nx = 1\n')
    with pytest.warns(UserWarning, match=r'Unknown section \[nope\]'):
        config.load_settings(path=cfg, env={})


def test_unknown_env_var_warns():
    with pytest.warns(UserWarning, match='AGGREGATE_BOGUS'):
        config.load_settings(path=None, env={'AGGREGATE_BOGUS': '1'})


# ---------------------------------------------------------------------------
# write_default_config / template
# ---------------------------------------------------------------------------

def test_write_default_config_is_all_defaults(tmp_path):
    out = tmp_path / 'config.toml'
    p = config.write_default_config(out)
    assert p == out
    s = config.load_settings(path=p, env={})
    # the shipped template is fully commented -> zero overrides
    assert all(src == 'default' for _, _, src in config.describe_settings(s))


def test_write_default_config_refuses_clobber(tmp_path):
    out = tmp_path / 'config.toml'
    config.write_default_config(out)
    with pytest.raises(FileExistsError):
        config.write_default_config(out)
    # force overwrites
    assert config.write_default_config(out, force=True) == out


# ---------------------------------------------------------------------------
# reload + module build refresh
# ---------------------------------------------------------------------------

def test_reload_refreshes_module_build(tmp_path, monkeypatch):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[build]\nlog2 = 11\n')
    monkeypatch.setenv('AGGREGATE_CONFIG', str(cfg))
    config.reload_settings()
    assert config.get_settings().build.log2 == 11
    assert global_build.log2 == 11  # mutated in place, references see it


# ---------------------------------------------------------------------------
# show_settings / source map
# ---------------------------------------------------------------------------

def test_describe_settings_reports_sources(tmp_path):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[validation]\neps = 1e-3\n')
    s = config.load_settings(path=cfg, env={'AGGREGATE_LOG2': '20'})
    src = {k: source for k, _, source in config.describe_settings(s)}
    assert src['validation.eps'] == 'config'
    assert src['build.log2'] == 'env'
    assert src['build.update'] == 'default'


# ---------------------------------------------------------------------------
# log2 unification (the headline fix)
# ---------------------------------------------------------------------------

def test_module_build_log2_matches_settings():
    assert global_build.log2 == config.get_settings().build.log2


def test_bare_underwriter_uses_configured_log2():
    from aggregate.underwriter import Underwriter
    uw = Underwriter()
    assert uw.log2 == config.get_settings().build.log2
    # explicit None databases still loads nothing
    uw2 = Underwriter(databases=None)
    assert uw2.databases == []
