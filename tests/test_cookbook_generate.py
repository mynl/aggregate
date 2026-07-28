"""The cookbook generator: [Cookbook-Generate], ``dev/plan-recipes.md`` Part B.

Recipe pages are **generated** from ``library.agg`` into native Quarto
``{python}`` cells by ``dev/generate_cookbook.py``. Two things must hold, and
neither is obvious from reading either file alone:

1. **The committed pages are not stale.** They are build artifacts that are
   committed (diffable; a docs build should not have to run a generator first),
   so an edit to a ``doc{{{...}}}`` without a re-run would silently ship a page
   that disagrees with the library.
2. **The page and the test run the same program.** The generator's only
   transform is a fence rewrite, so the code Quarto executes must be
   byte-identical to the code ``Recipe.run`` executes. That equivalence is the
   entire justification for generating rather than hand-writing pages; if it
   ever stops holding, the cookbook stops being self-testing.

The generator lives in ``dev/``, which is not a package, so it is loaded by
path.
"""
import importlib.util
from pathlib import Path
import re

import pytest

from aggregate import Underwriter
from aggregate.recipe import _FENCE_RE

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / 'dev' / 'generate_cookbook.py'

#: A cell the generator emitted, as Quarto will see it.
_CELL_RE = re.compile(r'```\{python\}\n(.*?)^```', re.S | re.M)
_SETUP_CELL = '#| echo: false\nfrom _setup import *'


@pytest.fixture(scope='module')
def gen():
    """``dev/generate_cookbook.py`` imported by path."""
    spec = importlib.util.spec_from_file_location('generate_cookbook', GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def library():
    uw = Underwriter(databases='library')
    uw.load()
    return uw


def test_committed_pages_are_up_to_date(gen):
    """Re-running the generator changes nothing (it is idempotent, and current)."""
    pages, _ = gen.build_pages()
    stale = [p.name for p, text in pages.items()
             if not p.exists() or p.read_text(encoding='utf-8') != text]
    assert not stale, (
        f'{stale} differ from what library.agg generates -- run '
        f'`python dev/generate_cookbook.py` and commit the result')


def test_every_documented_recipe_reaches_a_page(gen, library):
    """No silent drop: a doc whose topic maps to no section would vanish."""
    pages, n_recipes = gen.build_pages()
    documented = [n for (_k, n), r in library._recipes.items() if r.doc]
    assert n_recipes == len(documented), (
        'a documented entry carries no topic tag the generator maps to a '
        'section; run the generator and read its UNPLACED lines')
    body = '\n'.join(pages.values())
    for name in documented:
        assert name in body, f'{name} is on no generated page'


def test_generated_code_is_what_the_test_harness_runs(gen, library):
    """The whole justification for generating: one source, two consumers.

    The generator rewrites ```` ```python ```` to ```` ```{python} ```` and
    touches nothing else, so the code inside a generated cell must be exactly
    the code ``Recipe.run`` compiles. ``run()`` concatenates a section's fences
    into one script while the page emits them as separate cells, so the
    comparison is fence by fence.
    """
    pages, _ = gen.build_pages()
    body = '\n'.join(pages.values())
    emitted = {m.group(1).strip() for m in _CELL_RE.finditer(body)} - {_SETUP_CELL}
    checked = 0
    for (_kind, name), entry in library._recipes.items():
        if not entry.doc:
            continue
        r = library.recipe(name)
        for m in _FENCE_RE.finditer(f'{r.solution}\n{r.check}'):
            assert m.group(2).strip() in emitted, (
                f'{name}: a fence the harness executes is not on any page')
            checked += 1
    assert checked > 0, 'no fences compared -- the test would pass vacuously'


def test_a_plotting_recipe_puts_its_figure_inside_the_solution(gen):
    """The defect that motivated generating pages at all.

    The retired runtime ``recipe()`` verb ``exec``'d everything in one cell, and
    the inline backend flushes a figure at **cell end** -- so a plot created in
    the Solution appeared after the Discussion. As separate cells the figure
    lands where it was created. Asserted structurally: the plotting cell must
    precede the Discussion prose that follows it.
    """
    pages, _ = gen.build_pages()
    page = next(t for p, t in pages.items() if 'distortions' in p.name)
    plot_at = page.index('g.plot()')
    discussion_at = page.index('The proportional-hazard distortion is')
    assert plot_at < discussion_at
    # ... and it carries its own Quarto cell option, which is impossible when
    # one cell renders the whole recipe.
    assert '#| fig-cap:' in page


def test_check_blocks_are_executable_cells_not_quoted_prose(gen):
    """A check that renders but does not run would be decoration."""
    pages, _ = gen.build_pages()
    for path, text in pages.items():
        if 'The check' not in text:
            continue
        head = text.index('title="The check"')
        assert '```{python}' in text[head:head + 400], (
            f'{path.name}: the check callout holds no executable cell')
