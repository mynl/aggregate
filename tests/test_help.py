"""Tests for ``agg_help`` / ``.help`` rendering (1.0.0a101).

``.help`` is a side-effecting display helper, so it is tested through ``capsys``
and by monkeypatching the Jupyter probe. Three orthogonal axes: ``lod``
(docstring detail), ``values`` (value / call-result detail, renamed from the old
``output``), and ``fmt`` (render target: ``auto`` / ``text`` / ``ansi`` /
``html``).
"""

import pytest

from aggregate import build
import aggregate.utilities as u


@pytest.fixture(scope='module')
def a():
    return build('agg Dice dfreq [3] dsev [1:6]')


def test_help_text_is_plain(a, capsys):
    a.help('actual_m', fmt='text')
    out = capsys.readouterr().out
    assert 'actual_m' in out                       # the matched name appears
    assert '\x1b[' not in out                   # no ANSI escapes
    assert '<span' not in out                   # no HTML
    assert 'Markdown object' not in out         # not a stray IPython repr


def test_help_values_renamed(a, capsys):
    # the new keyword works ...
    a.help('actual_m', values='none', fmt='text')
    assert 'actual_m' in capsys.readouterr().out
    # ... and the old keyword is gone (rename is real, not aliased)
    with pytest.raises(TypeError):
        a.help('actual_m', output='none', fmt='text')


def test_help_bad_values_raises(a):
    with pytest.raises(ValueError):
        a.help('actual_m', values='lots', fmt='text')


def test_help_ansi_has_escape(a, capsys):
    a.help('actual_m', fmt='ansi')
    assert '\x1b[' in capsys.readouterr().out


def test_help_auto_resolves(a, capsys, monkeypatch):
    # auto delegates to the resolver: ANSI inside Jupyter, plain text outside
    monkeypatch.setattr(u, '_in_jupyter', lambda: True)
    a.help('actual_m', fmt='auto')
    assert '\x1b[' in capsys.readouterr().out

    monkeypatch.setattr(u, '_in_jupyter', lambda: False)
    a.help('actual_m', fmt='auto')
    assert '\x1b[' not in capsys.readouterr().out


def test_help_explicit_wins_over_auto(a, capsys, monkeypatch):
    # an explicit fmt is honored even when _in_jupyter would say otherwise
    monkeypatch.setattr(u, '_in_jupyter', lambda: True)
    a.help('actual_m', fmt='text')
    assert '\x1b[' not in capsys.readouterr().out


def test_help_bad_fmt_raises(a):
    with pytest.raises(ValueError):
        a.help('actual_m', fmt='xml')


def test_help_html_runs(a):
    pytest.importorskip('IPython')
    a.help('actual_m', fmt='html')   # rich display path executes without error
