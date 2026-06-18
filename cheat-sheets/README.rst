Cheat Sheets
============

One-page (per class) reference cards summarising every public method and
attribute of the major ``aggregate`` classes, plus a multi-page card for the
DecL language. Each card groups members into eleven fixed categories so the
same layout reads consistently across classes.

PDFs (rebuild with ``make.ps1``; combined into ``Cheat_Sheets.pdf`` by
``combine.ps1``):

* `DecL <DecL_Cheat_Sheet.pdf>`_ (the language; 3 pages: agg+pnl, port+mv+netceded, distortion)
* `Underwriter <Underwriter_Cheat_Sheet.pdf>`_
* `Severity <Severity_Cheat_Sheet.pdf>`_
* `Aggregate <Aggregate_Cheat_Sheet.pdf>`_
* `Portfolio <Portfolio_Cheat_Sheet.pdf>`_
* `Distortion <Distortion_Cheat_Sheet.pdf>`_

Building
--------

Toolchain is **Tectonic** (not lualatex/xelatex): faster, clearer errors,
auto-fetches fonts/packages, no ``.aux``/``.log`` clutter.

.. code-block:: powershell

    .\make.ps1                 # build all sheets
    .\make.ps1 Distortion      # build only matching sheet(s)
    .\combine.ps1              # merge per-sheet PDFs -> Cheat_Sheets.pdf

``cheat_sheet_macros.tex`` holds shared TikZ styles, the colour palette, the
method/static badges, and the footer. It is ``\input`` by every sheet and is
never compiled on its own. **The footer version string is hardcoded** in
``cheat_sheet_macros.tex`` (``\texttt{aggregate v.<version>}``) -- bump it when
the package version changes.


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

    from aggregate import build, Underwriter, Severity, Distortion
    from aggregate.utilities import introspect
    introspect(Underwriter())
    introspect(build('agg A 100 claims sev lognorm 100 cv 2 poisson'))   # Aggregate
    introspect(build('port P agg A 100 claims sev lognorm 50 cv 2 poisson '
                     'agg B 50 claims sev gamma 40 cv 1 poisson'))        # Portfolio
    introspect(Severity('lognorm', sev_mean=100, sev_cv=2))
    introspect(Distortion('ph', 0.5))

Also capture each member's ``signature`` and first docstring line (the ``help``
column, or ``inspect``) so comments are accurate, not guessed. Diff against the
existing ``.tex`` to catch renames/removals.

**2. Sort members into the eleven categories** (the section order on every
card): 1. Specification & creation; 2. Update; 3. Moments; 4. Statistical
functions; 5. Validation; 6. Output dataframes; 7. Reinsurance;
8. Visualization; 9. Risk and pricing; 10. Approximations; 11. Meta. A category
with nothing in it still gets a box reading *None* -- the empty boxes are
informative (they show what a class does **not** do).

**3. Write the boxes.** Mark instance methods with ``\m`` (red badge) and
static/class methods with ``\s`` (blue badge); plain fields and properties get
no badge ("fields or properties, used interchangeably"). Put a terse
parenthetical only on non-obvious entries. **Keep Aggregate and Portfolio terse**
(comma-separated lists, minimal comments); Underwriter/Distortion/DecL may carry
more explanation.

**4. Conventions.**

* **Colours** (set near the top of each ``.tex`` via ``\colorlet``): Underwriter
  = ``a`` (red), Severity = ``b`` (orange), Aggregate = ``c`` (green), Distortion
  = ``d`` (blue), Portfolio = ``e`` (navy). DecL = ``a``. Palette is defined in
  ``cheat_sheet_macros.tex``.
* **Layout**: ``\begin{multicols*}{3}`` with two ``\columnbreak``\ s. The footer
  (``\makefooter``) overlays the bottom-right corner, so the last column's
  content must clear it -- put the **Notes** block at the end of the right-hand
  column, *above* the footer, never stranded mid-page. Move ``\columnbreak``\ s
  so the empty *None* boxes absorb slack and the big boxes get their own column.
* The harmless ``Overfull \hbox (2.09pt ...)`` warnings come from the fixed
  minipage width and can be ignored.

**5. Compile and eyeball.** ``.\make.ps1 <Name>`` then open the PDF (read it --
check for column overflow, footer collisions, and that math/badges render).
Iterate. Then ``.\combine.ps1``.

**6. DecL is grammar-driven, not introspected.** Its single source of truth is
``src/aggregate/decl.lark`` (Lark, Earley). It is **3 pages** (one ``.tex`` with
``\clearpage`` between, ``\makefooter`` on each page):

* **agg + pnl** -- the compound-distribution clauses (name, exposure, limit,
  severity incl. ``dsev``/``xps``/``picks``/``ssev``, frequency, occ/agg
  reinsurance, ``approximate``, ``pnl``, trailer/vectors/math).
* **port + mv + netceded** -- ``port`` units, the ``multivariate``/``mv`` copula
  bivariate, ``netceded``.
* **distortion** -- declaration, kind/parameter table (from each ``Distortion``
  subclass's ``decl_params``), ``minimum``/``mixture`` combinators, usage.

Vocabularies to verify against the code when they may have changed: frequency
names and the ``FREQ`` terminal in ``decl.lark``; copula kinds via
``aggregate.copula.Copula._registry``; distortion kinds via
``Distortion.available_distortions()`` and per-kind ``decl_params`` in
``spectral.py``.

**7. Housekeeping.** Bump the footer version in ``cheat_sheet_macros.tex`` if the
package version moved. Per the project ``CLAUDE.md`` release rules, a doc-only
cheat-sheet refresh does not itself bump ``pyproject.toml``.
