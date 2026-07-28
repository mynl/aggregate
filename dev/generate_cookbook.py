"""Generate cookbook pages from the shipped library's ``doc{{{...}}}`` recipes.

[Cookbook-Generate], ``dev/plan-recipes.md`` Part B.

Reads ``library.agg`` and, for every entry carrying a doc, emits a Quarto
fragment whose code blocks are **native** ```` ```{python} ```` cells. Run it::

    .venv/Scripts/python.exe dev/generate_cookbook.py          # write
    .venv/Scripts/python.exe dev/generate_cookbook.py --check  # CI: is it stale?

Why generate rather than render at runtime
------------------------------------------
The predecessor was a ``recipe('X')`` verb that a thin page called: it emitted
``display(Markdown(...))`` for the prose and ``exec``'d the code itself, all
inside one cell. That reimplements a slice of Quarto, and badly:

* **per-block cell options cannot exist** — ``#| echo``, ``#| fig-cap``,
  ``#| warning`` are Quarto's, and there are no Quarto blocks to put them on;
* **a figure lands in the wrong place** — the inline backend flushes a figure
  that was merely *created* at **cell end**, so a plotting recipe drew its plot
  after the Discussion instead of inside the Solution.

Emitting real cells hands all of it back to Quarto: figures land where they are
created, ``#|`` options work per block, a failure is reported as a Quarto cell
error, and ``freeze`` caches at cell granularity.

The transform is deliberately thin -- prose passes through verbatim and each
fenced python block has its fence rewritten from ```` ```python ```` to
```` ```{python} ````. Nothing else. Because the code passes through
untouched, a doc can put a ``#| fig-cap: "..."`` line at the top of its fence
and Quarto will honour it, while :meth:`aggregate.recipe.Recipe.run` sees a
plain comment.

What is NOT generated
---------------------
Section landing pages (``_SS_00_*.qmd``) and the essay pages
(``_06_03_ir_modeling.qmd``, ``_02_02_scipy_continuous.qmd``) are hand-written
and stay that way; they cite recipes alongside their own prose. Generated
fragments are additive -- they are included at the end of their section, and
the hand-written five-beat stubs are retired page-by-page as phase 5 proceeds
(``dev/plan-meta-data.md``), with author reaction, not wholesale here.

Generated files are build artifacts but ARE committed: they are diffable, and
a docs build should not require running a generator first. They carry a
do-not-hand-edit header and the generator is idempotent.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from aggregate import Underwriter                       # noqa: E402
from aggregate.recipe import _FENCE_RE                  # noqa: E402

COOKBOOK = REPO_ROOT / 'docs' / 'cookbook'

#: Cookbook section for each ``topic:`` tag: ``(number, slug)``. The number is
#: the stable ``SS`` of ``[Cookbook-Numbering]``; the slug names the generated
#: file and prefixes every anchor in it. An entry carrying several topic tags is
#: placed in the lowest-numbered section it matches (and the others are logged,
#: never silently dropped).
#:
#: ``topic:numerics`` is deliberately absent -- it marks a property of the
#: computation, not a subject the cookbook has a section for. Section 7
#: (Bounds) has no topic tag yet.
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
#: title. The default (:func:`_title_of`) is fine for most; override here
#: rather than contorting a name that has to work as a ``build()`` argument.
TITLE_OVERRIDES = {
    'PHDistortion': 'Proportional-hazard distortion',
    'OccurrenceXOL': 'Occurrence excess of loss',
    'ThreeDice': 'Three dice: an exact discrete aggregate',
}

HEADER = ('<!-- GENERATED from library.agg by dev/generate_cookbook.py '
          '-- do not hand-edit. -->\n'
          '<!-- Edit the entry\'s doc{{{...}}} in '
          'src/aggregate/agg/library.agg and re-run the generator. -->\n')

# CamelCase boundaries: lower/digit -> upper (`LimitProfile`), and the end of an
# acronym run (`PHDistortion` -> `PH|Distortion`, not `PHDistortion`).
_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


def _title_of(name):
    """Heading text for an entry: an override, else the CamelCase split."""
    if name in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[name]
    words = _CAMEL_RE.sub(' ', name).split()
    return ' '.join([words[0]] + [w.lower() for w in words[1:]]) if words else name


def _anchor_of(slug, name):
    """A semantic, number-free anchor: ``#sec-recipe-<section>-<name>``.

    Its own ``sec-recipe-`` namespace, so a generated fragment can never
    collide with a hand-written page's anchor while both are in the book.
    """
    kebab = _CAMEL_RE.sub('-', name).lower().replace('.', '-').replace('_', '-')
    return f'sec-recipe-{slug}-{kebab}'


def _quartoize(markdown):
    """Rewrite ```` ```python ```` fences to ```` ```{python} ```` cells.

    The whole transform. Prose, non-python fences (```` ```text ````) and the
    code itself pass through untouched -- which is what keeps a generated page
    and :meth:`aggregate.recipe.Recipe.run` executing the *same* program.
    """
    return _FENCE_RE.sub(
        lambda m: '```{python}\n' + m.group(2).rstrip('\n') + '\n```',
        markdown)


def _fragment(recipe, slug):
    """One recipe as Quarto markdown: heading, Problem, Solution, ..., Check."""
    out = [f'## {_title_of(recipe.name)} '
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
        # Collapsed by default: the check is the point of the cookbook, but it
        # is an appendix to the reading, not part of it. It still EXECUTES --
        # a broken invariant fails the render.
        out.append('::: {.callout-note collapse="true" title="The check"}\n')
        out.append(_quartoize(recipe.check) + '\n')
        out.append(':::\n')
    return '\n'.join(out)


def _placed(uw):
    """``{(number, slug): [Recipe, ...]}`` for every documented entry.

    Logs any documented entry that no ``topic:`` tag places, and any secondary
    topic that was not used -- a silently dropped recipe would read as
    "everything is covered" when it is not.
    """
    placed, unplaced = {}, []
    for (kind, name), entry in sorted(uw._recipes.items()):
        if not entry.doc:
            continue
        hits = sorted(TOPIC_SECTIONS[t] for t in entry.tags
                      if t in TOPIC_SECTIONS)
        if not hits:
            unplaced.append(f'{kind}:{name} (tags {entry.tags})')
            continue
        placed.setdefault(hits[0], []).append(entry)
        for extra in hits[1:]:
            print(f'  note: {name} also carries {extra[1]}; placed in '
                  f'{hits[0][1]}')
    for u in unplaced:
        print(f'  UNPLACED (no cookbook section for its topic tags): {u}')
    return placed


def build_pages():
    """Render every generated fragment: ``{Path: text}``.

    The per-page setup cell is ``from _setup import *``, which supplies exactly
    the names :meth:`aggregate.recipe.Recipe.namespace` seeds (``build``, ``qd``,
    ``np``, ``pd``, ``aggregate``) -- so the page and the pytest harness run the
    same code against the same names.
    """
    uw = Underwriter(databases='library')
    uw.load()
    pages, n_recipes = {}, 0
    for (number, slug), recipes in sorted(_placed(uw).items()):
        n_recipes += len(recipes)
        body = '\n'.join(_fragment(r, slug) for r in recipes)
        path = COOKBOOK / f'_recipes_{number:02d}_{slug}.qmd'
        pages[path] = (f'{HEADER}\n```{{python}}\n#| echo: false\n'
                       f'from _setup import *\n```\n\n{body}')
    return pages, n_recipes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true',
                    help='exit 1 if any generated page is missing or stale; '
                         'write nothing')
    args = ap.parse_args(argv)

    pages, n_recipes = build_pages()
    stale = []
    for path, text in sorted(pages.items()):
        current = path.read_text(encoding='utf-8') if path.exists() else None
        if current == text:
            print(f'  unchanged  {path.name}')
            continue
        stale.append(path.name)
        if args.check:
            print(f'  STALE      {path.name}')
        else:
            path.write_text(text, encoding='utf-8')
            print(f'  written    {path.name}')

    if args.check and stale:
        print(f'\n{len(stale)} generated page(s) out of date: '
              f'run dev/generate_cookbook.py')
        return 1
    print(f'\n{n_recipes} recipe(s) in {len(pages)} generated page(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
