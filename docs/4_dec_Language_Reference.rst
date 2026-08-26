***********************************
**Dec** Language  Reference
***********************************

.. To refresh the grammar listing below, run ``python -m aggregate.parser``,
   which regenerates 4_agg_language_reference/ref_include.rst from decl.lark
   (the single source of truth). The railroad diagram is regenerated separately
   (see hacks/decl_railroad.py).

.. updated 2026-08-13


This section describes how a DecL program is pre-processed, lexed, and parsed according to the grammar specification. It reports the results of interpreting the builtin test suite of programs.

The :doc:`Aggregate Overview <2_Aggregate_Overview>` describes DecL's design and purpose.

Pre-Processing
==============

A program holds one or more statements. Two statements are separated either by
a **blank line** (a line that is empty or only whitespace — the markdown
paragraph model) or by a **semicolon at the end of a line** (the Python model,
so dense one-statement-per-line lists stay legal). Every other newline is just
whitespace, so a single statement may be laid out across as many lines, with
whatever indentation, as you like — a multi-line portfolio needs no trailing
markers, only that its units share one paragraph (no blank line between them).

Python and C++ style ``#`` / ``//`` comments are **transparent**: they never
separate statements. A full-line comment between the clause-lines of one
statement (e.g. a commented-out reinsurance clause) simply vanishes; the lines
around it stay in the same statement. The corollary is that a comment cannot
separate two statements — use a blank line or a ``;``.

Before passing to the lexer, the following pre-processing occurs.

1. Remove full-line ``#`` / ``//`` comments entirely, including their newline,
   so they leave no blank-line ghost
2. Strip trailing (inline) ``#`` / ``//`` comments through end of line
3. Remove ``\n`` inside ``[ ]`` (vectors) that appear from using ``f'{np.linspace(...)}'``
4. Turn a semicolon at the end of a line into a statement break
5. Split into statements on blank lines
6. Flatten each statement: its newlines and indentation collapse to spaces

.. note::

   The earlier ``\`` line-continuation has been removed. A statement now spans
   multiple lines for free, so a stray backslash is a lexer error rather than
   silently ignored.

Lexing
======

The following characters are ignored and may be used freely to improve readability: tab (remaining after pre-processing), colon, comma, and pipe.

Aggregate names must not include underscore. Portfolio names may include underscore. Names can include a period, ``A.Basic.01``.

Lexing is performed by Lark's dynamic, context-sensitive lexer directly from the
terminal definitions in the grammar (``aggregate/decl.lark``). Each terminal —
its regular expression, priority, and any keyword negative-lookahead — is part of
the grammar listing in :ref:`4_dec_Language_Reference:Dec Language Grammar Specification` below, so there
is no separate, hand-maintained token table. Keyword/ID disambiguation is handled
by terminal priorities (keywords priority 2, ``ID`` priority 1, builtin dotted
names priority 3), which is what makes the lexing contextual.

.. note::

   Earlier versions of this page reproduced a SLY ``tokens = {...}`` / ``ID['keyword'] = TOKEN``
   lexer table here. That is obsolete: the migration to Lark (Earley + dynamic
   lexer) in 2026 replaced the SLY remapping trick with terminal priorities, and
   the grammar file is now the single source of truth for both terminals and
   rules.

Numeric Expressions
===================

Every slot that takes a number accepts an arithmetic expression, evaluated at parse time: exposures, the informational premium, layers and attachments, frequency parameters, severity scales and shifts, reinsurance clauses, collars, and ranges.

The operator set splits in two. A **bare** expression, one written without surrounding parentheses, admits division ``/``, exponentiation ``**`` or ``^``, the exponential ``exp(...)``, and parentheses. Addition ``+``, subtraction ``-`` and multiplication ``*`` are legal **only inside parentheses**::

    agg TEST (4 + 3*2) claims (100_000/(1-.25)) premium 1000 xs 0
        sev (exp(-1 * .4**2/2)) * lognorm .4 poisson

Here ``(4 + 3*2)`` is a computed claim count of 10, ``(100_000/(1-.25))`` grosses a premium up for a 25% expense ratio, and ``(exp(-1 * .4**2/2))`` is the lognormal mean normalizer :math:`e^{-\sigma^2/2}` that makes the severity mean 1.

The parenthesis requirement is a design decision, not an oversight. Outside parentheses those three characters already mean something else: ``*`` is the severity scale operator (``10 * lognorm 10 cv .09``) and the portfolio homogeneous multiplier (``2 * agg.MyLine``), ``+`` is the severity shift and the portfolio sum, and ``-`` is the severity negation. Admitting them into bare expressions would make ``2 * 3 * agg.X`` genuinely ambiguous. Inside parentheses none of that applies, because a parenthesis holds exactly one expression.

Precedence runs, tightest first: parentheses, then ``exp``, then ``**`` and ``^``, then ``*`` and ``/``, then ``+`` and ``-``. Exponentiation is right-associative, so ``(2**3**2)`` is 512; everything else is left-associative, so ``(8/4/2)`` is 1.

Three behaviors are worth knowing.

**A minus sign glued to a number is part of that number, and so binds tighter than** ``**``. Writing ``(-.4**2)`` gives +0.16, because ``-.4`` lexes as a single numeric token and the square of a negative is positive. Python reads ``-.4**2`` as -0.16, because there the unary minus binds looser than the exponent. To negate a computed value in DecL, write ``(-1 * x)`` or ``(0 - x)``; the mean normalizer above uses the first form for exactly this reason.

**Parentheses hold one expression, never a list.** ``(1 -2)`` is the subtraction -1. Brackets remain the list syntax and are unaffected: ``[1 -2]`` is still the two-element vector, and ``[1 - 2]`` is a parse error.

**A percentage literal keeps its percent reading only when used literally.** Arithmetic returns a plain number, so ``(50% + 1)`` is 1.5, and a computed value in the ``po`` (part of) placement position reads as an absolute amount rather than a share.

Expressions evaluate as the program parses, so the specification carries plain numbers and the formula is not retained. The canonical text that :func:`aggregate.decl_writer.spec_to_decl` and :func:`aggregate.decl_writer.format_program` produce therefore shows the evaluated literal: the program above decompiles with ``10 claims 133333.33333333334 premium``. This is the long-standing canonical, not verbatim, contract, the same one that collapses ``exp(.5)`` to its float.

Dec Language Grammar Specification
===================================

Here is the full DecL Grammar and a `grammar railroad diagram <_static/diagram.xhtml>`_.

.. The grammar below is regenerated from decl.lark by ``python -m aggregate.parser``.

.. include:: 4_agg_language_reference/ref_include.rst

.. _test suite programs:

Test Suite Programs
===================

The test suite (``aggregate/agg/_test_suite.agg``) is exercised by the pytest
suite — each line of the file becomes its own parametrized test case (parse
check + SLY-snapshot shape check). Run::

    uv run pytest

The full ``.agg`` source:

.. literalinclude:: ../src/aggregate/agg/_test_suite.agg
   :language: agg

To only parse the file from Python::

    from aggregate import build
    filename = build.default_dir / '_test_suite.agg'
    assert filename.exists()

    build.logger_level(30)
    df = build.interpreter_file(filename=filename)

    df.query('error != 0')


Reading Parse Errors
=====================

When ``build()`` fails on a DecL typo, the wrapping :class:`ValueError`
carries a structured :class:`~aggregate.parser_errors.ErrorReport` on
its ``.report`` attribute. The report has 1-indexed line and column,
the source line (windowed to the caret on long inputs), a caret
marker, friendly "expected" labels, and ``difflib``-derived "did you
mean" suggestions.

Some mistakes produce a report that is accurate and still unhelpful,
because the offending token is fine in itself and the problem is the
grammar rule around it. For those, ``report.hint`` carries a sentence
naming the rule. Writing ``500 fixed expense as FE and 15% premium
expense as Comm`` reports ``Unexpected 'and'``, and the hint explains
that ``as`` closes an expense group, so the ``and`` has nothing to join:
drop it to leave two groups (one expense leg each), or move the ``as``
label after the last ``and``-joined term to combine them into one leg.
The hint is ``None`` when no rule matches, and it appears in both
``str(e)`` and ``render()``.

``str(e)`` is the one-line summary — ``DecL parse error at line L,
column C: Unexpected '...'. Did you mean: ...?`` — so the default
Python / Jupyter traceback footer is already useful without any
opt-in. ``e.report.render()`` returns the multi-line block with the
caret-annotated source line; the ``aggregate.underwriter`` logger
emits this automatically at ``ERROR`` level on every failed
``build()``. The three patterns below cover the common use-cases.

**1. Notebook / REPL — show the formatted error before the traceback.**

.. code-block:: python

    try:
        build('agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5')
    except ValueError as e:
        print(e.report.render())     # caret + "Did you mean..."
        raise                        # re-raise to keep the traceback

The rendered output looks like::

    DecL parse error at line 1, column 39:

      agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5
                                            ^^^^

    Unexpected 'mixd'. Did you mean: mixed? Expected: 'mixed', 'occurrence', '/', '+', '-', ...

For long programs (longer than ~80 characters) the source line is
windowed around the caret with ``... `` / `` ...`` markers and
word-boundary snapping, so the caret stays on a single terminal row.

**2. Script that wants the suggestion programmatically.**

.. code-block:: python

    try:
        build(text)
    except ValueError as e:
        if getattr(e, 'report', None) and e.report.suggestions:
            print(f"Did you mean: {e.report.suggestions[0]}?")
        raise

The ``getattr`` guard makes the snippet robust against non-parse
``ValueError`` (e.g. semantic validation failures from the transformer)
which don't carry a ``.report``.

**3. IPython traceback hook — auto-format every parse error in a session.**

For users who want the report rendered automatically in every Jupyter
cell, install a one-shot traceback hook (typically in
``~/.ipython/profile_default/startup/decl_errors.py``):

.. code-block:: python

    from IPython import get_ipython

    def _showtb(self, etype, evalue, tb, **kw):
        report = getattr(evalue, 'report', None)
        if report is not None:
            print(report.render())
        return self._showtraceback_original(etype, evalue, tb, **kw)

    ip = get_ipython()
    ip._showtraceback_original = ip.showtraceback
    ip.showtraceback = _showtb.__get__(ip)

The library deliberately does **not** install this hook on import —
recipe (3) is shown here for power users who want it, and is easy to
undo.

The same report is also written via the ``aggregate.underwriter``
logger at ``ERROR`` level on every failed ``build()``, so a notebook
with logging configured will see the rendered text adjacent to the
traceback without any opt-in.


Parser Implementation
=======================

The parser is built using `Lark <https://lark-parser.readthedocs.io/>`_ with an Earley backend and a dynamic, context-sensitive lexer. The grammar lives in ``aggregate/decl.lark`` and is the single source of truth — the listing in :ref:`4_dec_Language_Reference:Dec Language Grammar Specification` is regenerated from it. Earley dissolves the shift/reduce conflicts the previous SLY (LALR) implementation needed to hand-tune; the dynamic lexer plus tightened ``ID`` rule keep the grammar unambiguous.
