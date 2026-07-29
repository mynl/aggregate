Cheat Sheets
============

One-page (per class) reference cards summarising every public method and
attribute of the major ``aggregate`` classes, plus a multi-page card for the
DecL language. Each class card groups members into eleven fixed categories so
the same layout reads consistently across classes.

PDFs (rebuild with ``make.ps1``; combined into ``Cheat_Sheets.pdf`` by
``combine.ps1``):

* `DecL <DecL_Cheat_Sheet.pdf>`_ (the language; 4 pages: agg, reinsurance
  economics + pnl/xpnl, port + bivariate, distortion)
* `Underwriter <Underwriter_Cheat_Sheet.pdf>`_
* `Severity <Severity_Cheat_Sheet.pdf>`_
* `Aggregate <Aggregate_Cheat_Sheet.pdf>`_
* `BivariateAggregate <BivariateAggregate_Cheat_Sheet.pdf>`_
* `Portfolio <Portfolio_Cheat_Sheet.pdf>`_
* `PnL <PnL_Cheat_Sheet.pdf>`_
* `Distortion <Distortion_Cheat_Sheet.pdf>`_
* `\*Bounds <Bounds_Cheat_Sheet.pdf>`_ (``Bounds``, ``AllocationBounds``,
  ``PricingBounds`` side by side)

Building
--------

Toolchain is **Tectonic** (not lualatex/xelatex): faster, clearer errors,
auto-fetches fonts/packages, no ``.aux``/``.log`` clutter.

.. code-block:: powershell

    .\make.ps1                 # build all sheets
    .\make.ps1 Distortion      # build only matching sheet(s)
    .\make.ps1 DecL Bounds     # several, substring match
    .\combine.ps1              # merge per-sheet PDFs -> Cheat_Sheets.pdf

``cheat_sheet_macros.tex`` holds shared TikZ styles, the colour palette, the
method/static badges, and the footer. It is ``\input`` by every sheet and is
never compiled on its own. The footer version string is **generated**:
``make.ps1`` reads ``version`` from ``pyproject.toml`` and writes the
(gitignored) ``aggversion.tex`` that the macros file ``\input``\ s, so it
cannot drift.


Updating the cheat sheets (instructions for Claude)
---------------------------------------------------

When asked to "update the cheat sheets", follow this process. It is fully
self-contained -- everything you need is here plus the live API.

**1. Get the live API by introspection.** ``aggregate.utilities.introspect(ob)``
returns a DataFrame classifying every public name of ``ob`` as ``method``,
``property``, ``field``, or ``error`` (an attribute whose access raised -- the
function is hardened so it never aborts; a Portfolio that has not been updated,
for example, raises on ``density``). Columns: ``name, kind, value, type,
signature, help, length``. Run it under the managed env, setting
``UV_LINK_MODE=copy`` (see the project ``CLAUDE.md``):

.. code-block:: python

    from aggregate import build, Underwriter
    from aggregate.distributions import Severity
    from aggregate.spectral import Distortion
    from aggregate.bounds import Bounds, AllocationBounds, PricingBounds
    from aggregate.utilities import introspect

    introspect(Underwriter())
    introspect(Severity('lognorm', sev_mean=100, sev_cv=2))
    a = build('agg A 100 claims sev lognorm 100 cv 2 poisson'); introspect(a)
    p = build('port P agg A 100 claims sev lognorm 50 cv 2 poisson '
              'agg B 50 claims sev gamma 40 cv 1 poisson'); introspect(p)
    introspect(Distortion('ph', 0.5))
    bv = build('bv M 100 claims agg A 1 claim sev lognorm 50 cv 2 fixed '
               'agg B 1 claim sev gamma 40 cv 1 fixed copula gumbel 0.4 poisson')
    introspect(bv); introspect(bv.bivariate)          # and the view object
    introspect(build('pnl P 1000 premium less agg E 700 loss '
                     'sev lognorm 50 cv 2 poisson'))
    introspect(Bounds(a, a.actual_m * 1.15))
    introspect(AllocationBounds(p))
    introspect(PricingBounds(p, p.A))

The ``help`` column is empty for properties. Get their one-line docstrings with
``inspect.getattr_static(cls, name).__doc__`` over ``dir(cls)`` -- that is where
most of the useful parentheticals come from. Diff against the existing ``.tex``
to catch renames and removals; there are usually more than you expect (the a66
to a171 refresh found ``agg_m`` -> ``actual_m``, ``describe`` ->
``validation_df``, ``knowledge`` -> ``recipes``, and a dozen more).

**2. Sort members into the eleven categories** (the section order on every class
card): 1. Specification & creation; 2. Update; 3. Moments; 4. Statistical
functions; 5. Validation; 6. Output dataframes; 7. Reinsurance;
8. Visualization; 9. Risk and pricing; 10. Approximations; 11. Meta. A category
with nothing in it still gets a box -- the empty boxes are informative (they
show what a class does **not** do). Say *why* it is empty in one line rather
than printing a bare *None*; that line is often the most useful thing on the
card.

**3. Write the boxes.** Mark instance methods with ``\m`` (red badge) and
static/class methods with ``\s`` (blue badge); plain fields and properties get
no badge ("fields or properties, used interchangeably"). Put a terse
parenthetical only on non-obvious entries. **Keep Aggregate and Portfolio terse**
(comma-separated lists, minimal comments); Underwriter/Distortion/DecL and the
three new cards may carry more explanation.

**4. Conventions.**

* **Colours** (set near the top of each ``.tex`` via ``\colorlet``): Underwriter
  = ``a`` (red), Severity = ``b`` (orange), Aggregate = ``c`` (green), Distortion
  = ``d`` (blue), Portfolio = ``e`` (navy), BivariateAggregate = ``f`` (teal),
  PnL = ``g`` (plum), \*Bounds = ``h`` (violet). DecL = ``a``. The palette is
  defined in ``cheat_sheet_macros.tex``: ``a``--``e`` come from the logo,
  ``f``--``h`` extend it into the hue gaps the logo leaves open.
* **Layout**: ``\begin{multicols*}{3}`` with two ``\columnbreak``\ s. Note the
  **star**: columns are *not* balanced, they fill to the page height and then
  spill onto a second page.
* **The footer overlays the bottom-right corner**, so the last column's content
  must clear it: put the **Notes** block at the end of the right-hand column,
  above the footer, never stranded mid-page. Move ``\columnbreak``\ s so the
  small boxes absorb slack and the big boxes get their own column.
* The harmless ``Overfull \hbox (2.09pt ...)`` warnings come from the fixed
  minipage width and can be ignored.

**5. Compile and eyeball -- this is not optional.** ``.\make.ps1 <Name>``, then
**read the PDF** (Claude's ``Read`` tool renders PDF pages as images). Two
failures are invisible in the LaTeX log and show up only in the render:

* **A sheet that grew a page.** Check with
  ``pdfinfo <Name>_Cheat_Sheet.pdf | Select-String '^Pages:'`` -- every class
  card is 1 page and DecL is 4. A ``multicols*`` overflow silently spills to a
  new page, or strands the footer alone on one.
* **Notes running under the footer.** The footer is an ``overlay`` node and
  takes no vertical space, so LaTeX will happily typeset text underneath it.

Both are fixed by cutting prose, not by fiddling with the layout: trim the
italic glosses, merge two thin boxes, or move a long note up into the
full-width preamble (where it occupies roughly a third of the lines). Iterate
until the render is clean, then ``.\combine.ps1``.

**6. DecL is grammar-driven, not introspected.** Its single source of truth is
``src/aggregate/decl.lark`` (Lark, Earley). It is **4 pages** (one ``.tex`` with
``\clearpage`` between, ``\makefooter`` on each page):

* **agg** -- the compound-distribution clauses (name and ``as`` labels,
  exposure, limit, severity incl. ``dsev``/``xps``/``picks``/``ssev``,
  frequency, renewal ``years``/``wait``/``dwait``, reinsurance structure,
  ``approximate``, orientation, trailer/vectors/math).
* **reinsurance economics + pnl/xpnl** -- ceded premium (``deposit``/``rol``/
  ``rate``/``cede``), ``reinstatements``, variable rating (``swing``/``slide``/
  ``pc``/``corridor``), the ``pnl``/``xpnl`` declaration, premium heads, engine
  sources, ``expense`` groups.
* **port + bivariate** -- ``port`` units, ``bivariate``/``bv`` with a copula,
  ``dbvsev``, the ``netceded``/``grossceded``/``grossnet`` view pairs, ``clash``.
* **distortion** -- declaration, the kind/parameter table (from each
  ``Distortion`` subclass's ``decl_params``), ``minimum``/``mixture``, usage.

**Parse-check every form before printing it.** Write a throwaway script that
runs ``build(prog, update=False)`` over every syntax line the card claims and
report the failures. The a171 pass caught four wrong claims that had been on the
card since a66: ``xps`` attaches to ``dhistogram``/``chistogram`` and not to a
parametric severity; a reflected severity needs ``ssev``, not ``sev``;
``wtdtvar`` has no DecL number-list form at all; and variable rating is
aggregate-basis only in this release. Vocabularies worth re-checking: the
``FREQ`` terminal in ``decl.lark``; copula kinds and their parameter ranges via
``aggregate.copula.Copula._registry``; distortion kinds via
``Distortion.available_distortions()`` and per-kind ``decl_params`` in
``spectral.py``.

**7. Housekeeping.** The footer version is generated from ``pyproject.toml``, so
nothing to bump by hand. Per the project ``CLAUDE.md`` release rules, a doc-only
cheat-sheet refresh does not itself bump the package version.
