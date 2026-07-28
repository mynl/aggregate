"""Run every documented library recipe: its Solution, then its Check.

This is the *test* half of the notes-driven describe / test / audit surface.
Each entry in ``library.agg`` that carries a ``doc{{{...}}}`` becomes one test
case: the Solution block is executed, then the Check block runs in the same
namespace and its ``assert``s must hold.

Why it matters: before this, the shipped example library had exactly **one**
assertion against it -- that the file loaded. Nothing in it was ever
``build()``-ed, so an entry could parse cleanly and still produce garbage
moments, or fail validation, with no test signal at all. Now the library's own
stated invariants are the test.

An entry tagged ``slow`` is quarantined behind the ``slow`` marker, matching
the suite-wide fast-by-default policy.

See ``aggregate.recipe`` and ``dev/plan-meta-data.md`` [Recipe-Library].
"""
import matplotlib
matplotlib.use('Agg')          # no display; recipes are free to plot

import matplotlib.pyplot as plt
import pytest

from aggregate import Underwriter


def _documented():
    """(name, kind, is_slow) for every library entry carrying a doc."""
    uw = Underwriter(databases='library')
    uw.load()
    out = []
    for (kind, name), pp in sorted(uw._recipes.items()):
        spec = pp.spec if isinstance(pp.spec, dict) else {}
        if spec.get('doc'):
            out.append((name, kind, 'slow' in (spec.get('tags') or ())))
    return out


DOCUMENTED = _documented()
FAST = [(n, k) for n, k, slow in DOCUMENTED if not slow]
SLOW = [(n, k) for n, k, slow in DOCUMENTED if slow]


@pytest.fixture(scope='module')
def library():
    uw = Underwriter(databases='library')
    uw.load()
    return uw


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _run(uw, name):
    recipe = uw.recipe(name)
    assert recipe.is_runnable, (
        f'{name}: has a doc but no runnable ```python block. A recipe that '
        f'cannot be executed cannot be trusted -- give the Solution code, or '
        f'drop the doc.')
    recipe.run()
    return recipe


@pytest.mark.parametrize('name, kind', FAST,
                         ids=[f'{k}:{n}' for n, k in FAST])
def test_recipe_runs(library, name, kind):
    """The entry's Solution executes and its Check assertions hold."""
    _run(library, name)


@pytest.mark.slow
@pytest.mark.parametrize('name, kind', SLOW,
                         ids=[f'{k}:{n}' for n, k in SLOW] or None)
def test_slow_recipe_runs(library, name, kind):
    """Same, for entries tagged ``slow``."""
    _run(library, name)


@pytest.mark.parametrize('name, kind', FAST + SLOW,
                         ids=[f'{k}:{n}' for n, k in FAST + SLOW])
def test_documented_recipe_asserts_something(library, name, kind):
    """A recipe with a Check section must actually assert.

    A Check block with no ``assert`` states an invariant and tests nothing --
    the exact failure mode this whole mechanism exists to prevent. Caught here
    rather than left for ``build.recipes`` to report, because it is cheap.
    """
    recipe = library.recipe(name)
    if recipe.check:
        assert recipe.n_asserts > 0, (
            f'{name}: has a Check section but no assert statements')


def test_some_entries_are_documented():
    """Guard the guard: an empty parametrization would pass vacuously."""
    assert len(DOCUMENTED) > 0, 'no library entry carries a doc{{{...}}}'


def test_audit_frame_agrees_with_what_was_collected(library):
    """``build.recipes`` and this module must see the same documented set."""
    df = library.recipes
    assert set(df.query('doc').index.get_level_values('name')) == {
        n for n, _k, _s in DOCUMENTED}
