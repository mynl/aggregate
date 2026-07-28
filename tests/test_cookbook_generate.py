""":mod:`aggregate.cookbook`, the renderer half of the recipe surface.

Recipe pages are **generated** from ``library.agg`` into native Quarto
``{python}`` cells. Two things must hold, and neither is obvious from reading
either file alone:

1. **The committed pages are not stale.** They are build artifacts that are
   committed (diffable; a docs build should not have to run a generator first),
   so an edit to a ``doc{{{...}}}`` without a re-run would silently ship a page
   that disagrees with the library.
2. **The page and the test run the same program.** The only transform is a
   fence rewrite, so the code Quarto executes must be byte-identical to the code
   ``Recipe.run`` executes. That equivalence is the entire justification for
   generating rather than hand-writing pages; if it ever stops holding, the
   cookbook stops being self-testing.

Since 1.0.0a167 the generator is package code (it was ``dev/generate_cookbook.py``,
loaded here by file path), so these import it like anything else.
``dev/generate_cookbook.py`` survives as a thin caller supplying this repo's
library and output directory.
"""
from pathlib import Path
import re

import pytest

from aggregate import Underwriter
from aggregate.cookbook import cookbook_pages, render_recipe, write_cookbook
from aggregate.recipe import _FENCE_RE

REPO_ROOT = Path(__file__).resolve().parent.parent
COOKBOOK = REPO_ROOT / 'docs' / 'cookbook'

#: A cell the generator emitted, as Quarto will see it.
_CELL_RE = re.compile(r'```\{python\}\n(.*?)^```', re.S | re.M)
_SETUP_CELL = '#| echo: false\nfrom _setup import *'


@pytest.fixture(scope='module')
def library():
    uw = Underwriter(databases='library')
    uw.load()
    return uw


@pytest.fixture(scope='module')
def pages(library):
    """``{filename: markdown}`` for the shipped library."""
    return cookbook_pages(library)


def test_committed_pages_are_up_to_date(library):
    """Re-running the generator changes nothing (it is idempotent, and current)."""
    stale, _ = write_cookbook(COOKBOOK, library, check=True)
    assert not stale, (
        f'{stale} differ from what library.agg generates -- run '
        f'`python dev/generate_cookbook.py` and commit the result')


def test_every_documented_recipe_reaches_a_page(pages, library):
    """No silent drop: a doc whose topic maps to no section would vanish."""
    n_recipes = sum(t.count('{#sec-recipe-') for t in pages.values())
    documented = [n for (_k, n), r in library._recipes.items() if r.doc]
    assert n_recipes == len(documented), (
        'a documented entry carries no topic tag the generator maps to a '
        'section; run the generator and read the warning it logs')
    body = '\n'.join(pages.values())
    for name in documented:
        assert name in body, f'{name} is on no generated page'


def test_generated_code_is_what_the_test_harness_runs(pages, library):
    """The whole justification for generating: one source, two consumers.

    The generator rewrites ```` ```python ```` to ```` ```{python} ```` and
    touches nothing else, so the code inside a generated cell must be exactly
    the code ``Recipe.run`` compiles. ``run()`` concatenates a section's fences
    into one script while the page emits them as separate cells, so the
    comparison is fence by fence.
    """
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


def test_a_plotting_recipe_puts_its_figure_inside_the_solution(pages):
    """The defect that motivated generating pages at all.

    The retired runtime ``recipe()`` verb ``exec``'d everything in one cell, and
    the inline backend flushes a figure at **cell end** -- so a plot created in
    the Solution appeared after the Discussion. As separate cells the figure
    lands where it was created. Asserted structurally: the plotting cell must
    precede the Discussion prose that follows it.
    """
    page = next(t for n, t in pages.items() if 'distortions' in n)
    plot_at = page.index('g.plot()')
    discussion_at = page.index('The proportional-hazard distortion is')
    assert plot_at < discussion_at
    # ... and it carries its own Quarto cell option, which is impossible when
    # one cell renders the whole recipe.
    assert '#| fig-cap:' in page


def test_check_blocks_are_executable_cells_not_quoted_prose(pages):
    """A check that renders but does not run would be decoration."""
    for name, text in pages.items():
        if 'The check' not in text:
            continue
        head = text.index('title="The check"')
        assert '```{python}' in text[head:head + 400], (
            f'{name}: the check callout holds no executable cell')


# ----------------------------------------------------------------------
# It works on YOUR library, not just the shipped one ([Cookbook-Promote])
# ----------------------------------------------------------------------
def test_renders_an_arbitrary_library_not_just_the_shipped_one(tmp_path):
    """The reason this is package code rather than a repo script."""
    lib = tmp_path / 'mine.agg'
    lib.write_text(
        'agg MyStructure 5 claims sev lognorm 10 cv 2 poisson '
        'tags{topic:aggregate}\n    doc{{{\n'
        '## Problem\n\nI want my own book documented.\n\n'
        '## Solution\n\n```python\na = build(\'\'\'<<decl>>\'\'\')\n```\n\n'
        '## Check\n\n```python\nassert abs(a.actual_m - 50) < 1e-9\n```\n}}}\n',
        encoding='utf-8')
    uw = Underwriter()
    uw.load(lib)

    out = cookbook_pages(uw)
    assert list(out) == ['_recipes_04_aggregate.qmd']
    page = out['_recipes_04_aggregate.qmd']
    assert 'My structure' in page                  # CamelCase split title
    assert '{#sec-recipe-aggregate-my-structure}' in page
    assert '```{python}' in page and '```python\n' not in page

    written, _ = write_cookbook(tmp_path, uw)
    assert written == ['_recipes_04_aggregate.qmd']
    assert (tmp_path / '_recipes_04_aggregate.qmd').exists()
    # second run is a no-op: generation is idempotent, so --check is a real gate
    assert write_cookbook(tmp_path, uw)[0] == []


def test_a_documented_entry_with_no_mapped_topic_warns(tmp_path, caplog):
    """Silence would read as 'covered' when the recipe is nowhere in the book."""
    lib = tmp_path / 'orphan.agg'
    lib.write_text(
        'agg Orphaned 5 claims sev lognorm 10 cv 2 poisson tags{topic:numerics}'
        '\n    doc{{{\n## Problem\n\nNo section owns topic:numerics.\n}}}\n',
        encoding='utf-8')
    uw = Underwriter()
    uw.load(lib)
    with caplog.at_level('WARNING', logger='aggregate.cookbook'):
        assert cookbook_pages(uw) == {}
    assert 'Orphaned' in caplog.text


def test_render_recipe_alone_needs_no_underwriter():
    """The one-recipe entry point, for a caller building its own page."""
    from aggregate.recipe import parse_doc
    r = parse_doc('## Problem\n\nWhy.\n\n## Solution\n\n```python\nx = 1\n```',
                  name='SoloDemo')
    out = render_recipe(r, slug='demo', level=3)
    assert out.startswith('### Solo demo {#sec-recipe-demo-solo-demo}')
    assert '```{python}\nx = 1\n```' in out
