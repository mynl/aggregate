"""Tests for the ``%%agg`` cell magic (:mod:`aggregate.magics`).

The magic's job is the binding and the volume control, not the language, so
these tests swap the module-level ``build_many`` for a bare
:class:`~aggregate.underwriter.Underwriter`, which loads no ``.agg`` file. The
programs below are small and self-contained, so the shipped library is
irrelevant to them and a broken entry in it cannot fail this file.
"""

import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

from aggregate.underwriter import Underwriter              # noqa: E402

IPython = pytest.importorskip('IPython')

from aggregate import magics                               # noqa: E402


@pytest.fixture(scope='module')
def shell():
    """A live IPython shell with the magics registered."""
    from IPython.core.interactiveshell import InteractiveShell
    sh = InteractiveShell.instance()
    magics.load_ipython_extension(sh)
    return sh


@pytest.fixture(autouse=True)
def _no_library(monkeypatch):
    """Point the magic at an underwriter that has read no database."""
    monkeypatch.setattr(magics, 'build_many', Underwriter().build_many)


def test_single_binds_target_and_decl_name(shell, capsys):
    """One output binds ``a`` and the DecL name, and displays."""
    assert shell.run_cell('%%agg\nagg Dice dfreq [3] dsev [1:6]\n').success
    out = capsys.readouterr().out
    assert 'bound a, Dice' in out
    assert 'Agg' in out                       # the qd table arrived
    assert shell.user_ns['a'] is shell.user_ns['Dice']
    assert shell.user_ns['a'].actual_m == pytest.approx(10.5)


def test_named_target(shell):
    """The positional argument renames the binding."""
    assert shell.run_cell('%%agg book\nagg Dice2 dfreq [3] dsev [1:6]\n').success
    assert shell.user_ns['book'] is shell.user_ns['Dice2']


def test_quiet_reports_names_only(shell, capsys):
    """``-q`` says what it bound and stops."""
    assert shell.run_cell('%%agg -q\nagg Dice3 dfreq [3] dsev [1:6]\n').success
    out = capsys.readouterr().out
    assert out.strip() == 'bound a, Dice3'


def test_silent_prints_nothing(shell, capsys):
    """``-s`` builds and says nothing at all."""
    assert shell.run_cell('%%agg -s\nagg Dice4 dfreq [3] dsev [1:6]\n').success
    assert capsys.readouterr().out == ''
    assert shell.user_ns['Dice4'].actual_m == pytest.approx(10.5)


def test_several_outputs_bind_a_dict(shell, capsys):
    """Several statements bind the target to a dict, plus each name."""
    program = ('%%agg book -q\n'
               'agg A1 dfreq [3] dsev [1:6]\n\n'
               'agg B1 dfreq [2] dsev [1:3]\n\n'
               'distortion D1 ph 0.7\n')
    assert shell.run_cell(program).success
    assert capsys.readouterr().out.strip() == 'bound book, A1, B1, D1'
    assert set(shell.user_ns['book']) == {'A1', 'B1', 'D1'}
    assert shell.user_ns['book']['A1'] is shell.user_ns['A1']
    assert shell.user_ns['D1'].shape == pytest.approx(0.7)


def test_grid_arguments_reach_the_build(shell):
    """``--bs`` is evaluated, so a fraction works, and ``--log2`` passes."""
    program = ('%%agg c -s --bs 1/32 --log2 16\n'
               'agg C1 10 claims 100 xs 0 sev lognorm 20 cv 1.5 poisson;\n')
    assert shell.run_cell(program).success
    assert shell.user_ns['c'].bs == pytest.approx(1 / 32)
    assert shell.user_ns['c'].log2 == 16


def test_plot_draws_a_single_object(shell):
    """``-p`` draws; ``-s -p`` still draws, silence being about text."""
    plt.close('all')
    assert shell.run_cell('%%agg -s -p\nagg Dice5 dfreq [3] dsev [1:6]\n').success
    assert plt.get_fignums(), 'expected a figure'
    plt.close('all')


def test_plot_declines_on_several_outputs(shell, capsys):
    """``-p`` says why it did not draw rather than drawing several."""
    plt.close('all')
    program = ('%%agg -q -p\n'
               'agg A2 dfreq [3] dsev [1:6]\n\n'
               'agg B2 dfreq [2] dsev [1:3]\n')
    assert shell.run_cell(program).success
    assert '--plot draws a single object only' in capsys.readouterr().out
    assert not plt.get_fignums()


def test_import_aggregate_does_not_load_ipython():
    """The magics module imports IPython, so it stays off the import path.

    ``aggregate.utilities`` defers its IPython import because the import costs
    about a second. This module cannot defer one, so ``import aggregate`` must
    not reach it: the load is ``%load_ext aggregate.magics``, on request.
    """
    import subprocess
    import sys
    code = ("import sys, aggregate; "
            "assert 'IPython' not in sys.modules, "
            "'import aggregate pulled in IPython'")
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parse_error_propagates(shell):
    """A bad program fails the cell rather than binding half of it."""
    assert not shell.run_cell('%%agg\nagg Bad1 dfreq [3] dsev\n').success
    assert 'Bad1' not in shell.user_ns
