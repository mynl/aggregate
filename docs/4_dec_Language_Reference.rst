***********************************
**Dec** Language  Reference
***********************************

.. to update run: python -m aggregate.parser to refresh 4_agg_langugage_reference/ref_include.rst
   and (alas) manually paste the lexer defs below

.. updated 2022-12-27

.. To view the grammar using a railroad diagram paste the
   specification below into
   the Edit Grammar tab of https://www.bottlecaps.de/rr/ui and then View Diagram.
   (Site diagram uses #DDDDDD as the base color.)


This section describes how a DecL program is pre-processed, lexed, and parsed according to the grammar specification. It reports the results of interpreting the builtin test suite of programs.

The DecL :ref:`introduction <design and purpose>` describes its design and purpose.

Pre-Processing
==============

Programs are processed one line at a time. Before passing to the lexer, the following pre-processing occurs.

1. Remove Python and C++ style  ``#`` or ``//`` comments, through end of line
2. Remove \\n in [ ] (vectors) that appear from  using ``f'{np.linspace(...)}'``
3. Map backslash newline (Python line continuations) to space
4. Replace \\n\\t  with space, to support the tabbed indented Portfolio layout
5. Split on remaining newlines

Lexer Term Definitions
======================

Ignored characters: tab (remaining after pre-processing), colon, comma, and pipe. These characters can be used to improve readability.

Aggregate names must not include underscore. Portfolio names may include underscore. Names can include a period, ``A.Basic.01``.

::

    tokens = {ID, BUILTIN_AGG, BUILTIN_SEV, NOTE,
              SEV, AGG, PORT,
              NUMBER,
              PLUS, MINUS, TIMES, DIVIDE, INHOMOG_MULTIPLY,
              LOSS, PREMIUM, AT, LR, CLAIMS, EXPOSURE, RATE,
              XS, PICKS,
              DISTORTION,
              CV, WEIGHTS, EQUAL_WEIGHT, XPS, SPLICE,
              MIXED, FREQ, TWEEDIE, ZM, ZT,
              NET, OF, CEDED, TO, OCCURRENCE, AGGREGATE, PART_OF, SHARE_OF, TOWER,
              AND,
              EXPONENT, EXP,
              DFREQ, DSEV, RANGE
              }

    ignore = ' \t,\\|'
    literals = {'[', ']', '!', '(', ')'}
    NOTE = r'note\{[^\}]*\}'
    BUILTIN_AGG = r'agg\.[a-zA-Z][a-zA-Z0-9._:~\-]*'
    BUILTIN_SEV = r'sev\.[a-zA-Z][a-zA-Z0-9._:~\-]*'
    FREQ = 'binomial|pascal|poisson|bernoulli|geometric|fixed|neyman(a|A)?|logarithmic|negbin'
    DISTORTION = 'dist(ortion)?'
    # NUMBER absorbs an optional leading minus, an optional ``%`` suffix
    # (percent), and the special values ``inf`` / ``-inf``.
    NUMBER = r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?%?|\-?inf'
    # ID excludes any reserved keyword in standalone form and anything
    # starting with ``agg.`` or ``sev.`` (which are BUILTIN_AGG / BUILTIN_SEV).
    ID = r'[a-zA-Z][\._:~a-zA-Z0-9\-]*'
    EXPONENT = r'\^|\*\*'
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    DIVIDE = '/'
    INHOMOG_MULTIPLY = '@'
    EQUAL_WEIGHT = '='
    RANGE = ':'

    ID['occurrence'] = OCCURRENCE
    ID['aggregate'] = AGGREGATE
    ID['exposure'] = EXPOSURE
    ID['tweedie'] = TWEEDIE
    ID['premium'] = PREMIUM
    ID['tower'] = TOWER
    ID['mixed'] = MIXED
    ID['picks'] = PICKS
    ID['prem'] = PREMIUM
    ID['claims'] = CLAIMS
    ID['splice'] = SPLICE
    ID['ceded'] = CEDED
    ID['claim'] = CLAIMS
    ID['dfreq'] = DFREQ
    ID['dsev'] = DSEV
    ID['loss'] = LOSS
    ID['port'] = PORT
    ID['rate'] = RATE
    ID['net'] = NET
    ID['sev'] = SEV
    ID['agg'] = AGG
    ID['xps'] = XPS
    ID['wts'] = WEIGHTS
    ID['and'] = AND
    ID['exp'] = EXP
    ID['at'] = AT
    ID['cv'] = CV
    ID['lr'] = LR
    ID['xs'] = XS
    ID['of'] = OF
    ID['to'] = TO
    ID['po'] = PART_OF
    ID['so'] = SHARE_OF
    ID['zm'] = ZM
    ID['zt'] = ZT

Dec Language Grammar Specification
===================================

Here is the full DecL Grammar and a `grammar railroad diagram <_static/diagram.xhtml>`_.

.. run python aggregate.parser.py to update this file

.. literalinclude:: 4_agg_language_reference/ref_include.rst

.. _test suite programs:

Test Suite Programs
===================

The test suite (``aggregate/agg/test_suite.agg``) is exercised by the pytest
suite — each line of the file becomes its own parametrized test case (parse
check + SLY-snapshot shape check). Run::

    uv run pytest

The full ``.agg`` source:

.. literalinclude:: ../aggregate/agg/test_suite.agg
   :language: agg

To only parse the file from Python::

    from aggregate import build
    filename = build.default_dir / 'test_suite.agg'
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

The parser is built using `Lark <https://lark-parser.readthedocs.io/>`_ with an Earley backend and a dynamic, context-sensitive lexer. The grammar lives in ``aggregate/decl.lark`` and is the single source of truth — the listing in :ref:`Dec Language Grammar Specification` is regenerated from it. Earley dissolves the shift/reduce conflicts the previous SLY (LALR) implementation needed to hand-tune; the dynamic lexer plus tightened ``ID`` rule keep the grammar unambiguous.


