"""Render a DecL library's recipes as a Quarto cookbook.

The third consumer of the ``doc{{{...}}}`` surface. Its siblings are
:meth:`aggregate.recipe.Recipe.run` (which *executes* a recipe, backing the
pytest harness) and :attr:`aggregate.Underwriter.recipes` (which *audits* one).
This module *renders*, so a library that documents and tests itself also
publishes itself.

That applies to your library, not just the shipped one. Point it at any
:class:`~aggregate.Underwriter` and every entry carrying a doc becomes a page::

    from aggregate import Underwriter
    from aggregate.cookbook import write_cookbook

    uw = Underwriter(databases='my_structures')
    uw.load()
    write_cookbook('book', uw)

or from a shell, over the shipped library::

    python -m aggregate.cookbook docs/cookbook
    python -m aggregate.cookbook docs/cookbook --check   # CI: is it stale?

Why generate rather than render at runtime
------------------------------------------
The predecessor was a ``recipe('X')`` verb that a thin page called: it emitted
``display(Markdown(...))`` for the prose and ``exec``'d the code itself, all
inside one cell. That reimplements a slice of Quarto, and badly. Per-block cell
options (``#| echo``, ``#| fig-cap``, ``#| warning``) are Quarto's, and there
were no Quarto blocks to put them on; worse, the inline backend flushes a figure
that was merely *created* at **cell end**, so a plotting recipe drew its plot
after the Discussion instead of inside the Solution.

Emitting real cells hands all of it back to Quarto: figures land where they are
created, ``#|`` options work per block, a failure is reported as a Quarto cell
error, and ``freeze`` caches at cell granularity.

One source, two consumers
--------------------------
The transform is deliberately thin. Prose passes through verbatim and each
fenced python block has its info string rewritten from ``python`` to
``{python}``. Nothing else. Because the code itself is untouched, the
page Quarto executes and the script
:meth:`~aggregate.recipe.Recipe.run` compiles are byte-identical, which is what
makes "the cookbook and the test suite run the same program" true by
construction rather than by discipline.

It also means a doc can open a fence with ``#| fig-cap: "..."``: Quarto honours
it as a cell option, and ``run`` sees a plain comment.

Output
------
One page per cookbook section, ``_recipes_NN_<slug>.qmd``, grouping entries by
their ``topic:`` tag (:data:`TOPIC_SECTIONS`). Pages are **build artifacts that
should still be committed**: they are diffable, and a docs build should not
have to run a generator first. Each carries a do-not-hand-edit header, and
generation is idempotent, so ``--check`` is a usable CI gate.

Section landing pages and hand-written essay pages are not generated and are
not touched.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

from .recipe import _FENCE_RE

__all__ = ['render_recipe', 'cookbook_pages', 'write_cookbook',
           'TOPIC_SECTIONS', 'TITLE_OVERRIDES']

logger = logging.getLogger(__name__)

#: Cookbook section for each ``topic:`` tag: ``(number, slug)``. The number is
#: the stable ``SS`` of the two-level ``_SS_MM`` page numbering; the slug names
#: the generated file and prefixes every anchor in it. An entry carrying several
#: topic tags is placed in the lowest-numbered section it matches, and the
#: others are logged rather than silently ignored.
#:
#: ``topic:numerics`` is deliberately absent: it marks a property of the
#: computation, not a subject the cookbook has a section for. Section 7
#: (Bounds) has no topic tag yet. Pass your own mapping as ``sections=`` to
#: organize a different library.
TOPIC_SECTIONS = {
    'topic:distortion': (1, 'distortions'),
    'topic:severity': (2, 'severity'),
    'topic:frequency': (3, 'frequency'),
    'topic:aggregate': (4, 'aggregate'),
    'topic:reinsurance': (5, 'reinsurance'),
    'topic:pnl': (6, 'pnl'),
    'topic:portfolio': (8, 'portfolio'),
    'topic:bivariate': (9, 'bivariate'),
}

#: Heading text for an entry whose CamelCase name does not split into a good
#: title. The default (:func:`_title_of`) is fine for most; override here rather
#: than contorting a name that also has to work as a ``build()`` argument. Pass
#: your own as ``titles=``.
TITLE_OVERRIDES = {
    'PHDistortion': 'Proportional-hazard distortion',
    'OccurrenceXOL': 'Occurrence excess of loss',
    'ThreeDice': 'Three dice: an exact discrete aggregate',
}

#: The per-page setup cell. The default supplies exactly the names
#: :meth:`aggregate.recipe.Recipe.namespace` seeds (``aggregate``, ``build``,
#: ``qd``, ``np``, ``pd``), so a page and the pytest harness run the same code
#: against the same names. Override for a book with its own setup module.
DEFAULT_SETUP = 'from _setup import *'

_HEADER = ('<!-- GENERATED from a DecL library by aggregate.cookbook '
           '(do not hand-edit). -->\n'
           '<!-- Edit the entry\'s doc{{{...}}} in the .agg file and '
           're-run the generator. -->\n')

# CamelCase boundaries: lower/digit then upper (``LimitProfile``), and the end
# of an acronym run (``PHDistortion`` splits ``PH|Distortion``).
_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


def _title_of(name, titles):
    """Heading text for an entry: an override, else the CamelCase split."""
    if name in titles:
        return titles[name]
    words = _CAMEL_RE.sub(' ', name).split()
    return ' '.join([words[0]] + [w.lower() for w in words[1:]]) if words else name


def _anchor_of(slug, name):
    """A semantic, number-free anchor: ``sec-recipe-<section>-<name>``.

    Its own ``sec-recipe-`` namespace, so a generated fragment can never collide
    with a hand-written page's anchor while both are in the same book, and a
    cross-reference survives any renumbering.
    """
    kebab = _CAMEL_RE.sub('-', name).lower().replace('.', '-').replace('_', '-')
    return f'sec-recipe-{slug}-{kebab}' if slug else f'sec-recipe-{kebab}'


def _quartoize(markdown):
    """Rewrite ```` ```python ```` fences as ```` ```{python} ```` cells.

    The whole transform. Prose, non-python fences (```` ```text ````, which a
    recipe uses to *show* output rather than produce it) and the code itself
    pass through untouched.
    """
    return _FENCE_RE.sub(
        lambda m: '```{python}\n' + m.group(2).rstrip('\n') + '\n```',
        markdown)


def render_recipe(recipe, *, slug='', level=2, titles=None):
    """Render one :class:`~aggregate.recipe.Recipe` as Quarto markdown.

    Parameters
    ----------
    recipe : aggregate.recipe.Recipe
        An entry carrying a ``doc``. One with no doc renders as a bare heading.
    slug : str, optional
        Section slug, used to namespace the anchor.
    level : int, default 2
        Heading level. ``2`` is a recipe under a section landing page.
    titles : dict, optional
        Name to heading-text overrides; :data:`TITLE_OVERRIDES` by default.

    Returns
    -------
    str
        Heading, Problem, Solution, Discussion, any extra sections, then the
        Check inside a collapsed callout. Sections the recipe omits are skipped.

    Notes
    -----
    The Check is collapsed because it is an appendix to the reading rather than
    part of it, but it is still a live cell: a broken invariant fails the
    render, which is the point of publishing it at all.
    """
    titles = TITLE_OVERRIDES if titles is None else titles
    hashes = '#' * max(1, level)
    out = [f'{hashes} {_title_of(recipe.name, titles)} '
           f'{{#{_anchor_of(slug, recipe.name)}}}\n']
    if recipe.problem:
        out.append(recipe.problem + '\n')
    if recipe.solution:
        out.append(_quartoize(recipe.solution) + '\n')
    if recipe.discussion:
        out.append(recipe.discussion + '\n')
    if recipe.extra:
        out.append(_quartoize(recipe.extra) + '\n')
    if recipe.check:
        out.append('::: {.callout-note collapse="true" title="The check"}\n')
        out.append(_quartoize(recipe.check) + '\n')
        out.append(':::\n')
    return '\n'.join(out)


def _place(uw, sections):
    """Group documented entries by section: ``{(number, slug): [Recipe, ...]}``.

    Warns about any documented entry that no ``topic:`` tag places, and notes
    any secondary topic that went unused. A silently dropped recipe would read
    as "everything is covered" when it is not.
    """
    placed = {}
    for (kind, name), entry in sorted(uw._recipes.items()):
        if not entry.doc:
            continue
        hits = sorted(sections[t] for t in entry.tags if t in sections)
        if not hits:
            logger.warning(
                'cookbook: %s %s carries a doc but no topic tag that maps to a '
                'section (tags %s); it will not appear in the book.',
                kind, name, entry.tags)
            continue
        placed.setdefault(hits[0], []).append(entry)
        for extra in hits[1:]:
            logger.info('cookbook: %s also carries %s; placed in %s.',
                        name, extra[1], hits[0][1])
    return placed


def cookbook_pages(uw=None, *, sections=None, titles=None, setup=DEFAULT_SETUP):
    """Render a whole library: ``{filename: text}``.

    Pure. Touches no filesystem, so it is the piece to call when you want the
    markdown for something other than a file on disk.

    Parameters
    ----------
    uw : aggregate.Underwriter, optional
        The library to render. Defaults to a fresh underwriter over the shipped
        ``library.agg``; pass your own to publish your own structures.
    sections : dict, optional
        ``topic:`` tag to ``(number, slug)``; :data:`TOPIC_SECTIONS` by default.
    titles : dict, optional
        Name to heading-text overrides; :data:`TITLE_OVERRIDES` by default.
    setup : str, default :data:`DEFAULT_SETUP`
        Body of the hidden setup cell placed at the top of every page.

    Returns
    -------
    dict
        ``{filename: markdown}``, one entry per section that has at least one
        documented recipe. Filenames are bare, with no directory part.
    """
    if uw is None:
        from .config import LIBRARY_FILENAME
        from .underwriter import Underwriter
        uw = Underwriter(databases=Path(LIBRARY_FILENAME).stem)
        uw.load()
    sections = TOPIC_SECTIONS if sections is None else sections

    pages = {}
    for (number, slug), recipes in sorted(_place(uw, sections).items()):
        body = '\n'.join(render_recipe(r, slug=slug, titles=titles)
                         for r in recipes)
        pages[f'_recipes_{number:02d}_{slug}.qmd'] = (
            f'{_HEADER}\n```{{python}}\n#| echo: false\n{setup}\n```\n\n{body}')
    return pages


def write_cookbook(out_dir, uw=None, *, check=False, **kwargs):
    """Write (or check) the generated pages under *out_dir*.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Directory to write into. Must exist.
    uw : aggregate.Underwriter, optional
        Passed to :func:`cookbook_pages`.
    check : bool, default False
        Do not write; just report which pages are missing or out of date. The
        CI gate, since generation is idempotent.
    **kwargs
        Passed to :func:`cookbook_pages` (``sections``, ``titles``, ``setup``).

    Returns
    -------
    (stale, pages) : (list of str, dict)
        The filenames that differed (written, unless ``check``), and every page
        rendered. ``stale == []`` means the directory is up to date.
    """
    out_dir = Path(out_dir)
    pages = cookbook_pages(uw, **kwargs)
    stale = []
    for name, text in sorted(pages.items()):
        path = out_dir / name
        if path.exists() and path.read_text(encoding='utf-8') == text:
            continue
        stale.append(name)
        if not check:
            path.write_text(text, encoding='utf-8')
            logger.info('cookbook: wrote %s', path)
    return stale, pages


def main(argv=None):
    """Command line entry: ``python -m aggregate.cookbook <out_dir> [--check]``."""
    ap = argparse.ArgumentParser(
        prog='python -m aggregate.cookbook',
        description='Render a DecL library\'s recipes as Quarto pages.')
    ap.add_argument('out_dir', help='directory to write the pages into')
    ap.add_argument('--database', default=None,
                    help='.agg database to render (default: the shipped library)')
    ap.add_argument('--check', action='store_true',
                    help='write nothing; exit 1 if any page is missing or stale')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    uw = None
    if args.database:
        from .underwriter import Underwriter
        uw = Underwriter(databases=args.database)
        uw.load()

    stale, pages = write_cookbook(args.out_dir, uw, check=args.check)
    for name in sorted(pages):
        mark = ('STALE' if args.check else 'written') if name in stale \
            else 'unchanged'
        print(f'  {mark:9s}  {name}')
    # Every recipe emits exactly one anchor, so this counts them exactly;
    # a heading inside a doc's prose does not carry one.
    n = sum(t.count('{#sec-recipe-') for t in pages.values())
    print(f'\n{n} recipe(s) in {len(pages)} page(s).')
    if args.check and stale:
        print(f'{len(stale)} page(s) out of date; re-run without --check.')
        return 1
    return 0


if __name__ == '__main__':                                  # pragma: no cover
    raise SystemExit(main())
