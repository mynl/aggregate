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
3. Map semicolons to newline
4. Map backslash newline (Python line continuations) to space
5. Replace \\n\\t  with space, to support the tabbed indented Portfolio layout
6. Split on remaining newlines

Lexer Term Definitions
======================

Ignored characters: tab (remaining after pre-processing), colon, comma, and pipe. These characters can be used to improve readability.

Aggregate names must not include underscore. Portfolio names may include underscore. Names can include a period, ``A.Basic.01``.

::

    tokens = {ID, BUILTIN_AGG, BUILTIN_SEV,NOTE,
              SEV, AGG, PORT,
              NUMBER, INFINITY,
              PLUS, MINUS, TIMES, DIVIDE, INHOMOG_MULTIPLY,
              LOSS, PREMIUM, AT, LR, CLAIMS, EXPOSURE, RATE,
              XS, PICKS,
              DISTORTION,
              CV, WEIGHTS, EQUAL_WEIGHT, XPS,
              MIXED, FREQ, TWEEDIE, ZM, ZT,
              NET, OF, CEDED, TO, OCCURRENCE, AGGREGATE, PART_OF, SHARE_OF, TOWER,
              AND,  PERCENT,
              EXPONENT, EXP,
              DFREQ, DSEV, RANGE
              }

    ignore = ' \t,\\|'
    literals = {'[', ']', '!', '(', ')'}
    NOTE = r'note\{[^\}]*\}'  # r'[^\}]+'
    BUILTIN_AGG = r'agg\.[a-zA-Z][a-zA-Z0-9._:~]*'
    BUILTIN_SEV = r'sev\.[a-zA-Z][a-zA-Z0-9._:~]*'
    FREQ = 'binomial|pascal|poisson|bernoulli|geometric|fixed|neyman(a|A)?'
    DISTORTION = 'dist(ortion)?'
    NUMBER = r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?'
    ID = r'[a-zA-Z][\.:~_a-zA-Z0-9]*'
    EXPONENT = r'\^|\*\*'
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    DIVIDE = '/'
    PERCENT = '%'
    INHOMOG_MULTIPLY = '@'
    EQUAL_WEIGHT = '='
    RANGE = ':'

    ID['occurrence'] = OCCURRENCE
    ID['unlimited'] = INFINITY
    ID['aggregate'] = AGGREGATE
    ID['exposure'] = EXPOSURE
    ID['tweedie'] = TWEEDIE
    ID['premium'] = PREMIUM
    ID['tower'] = TOWER
    ID['mixed'] = MIXED
    ID['unlim'] = INFINITY
    ID['picks'] = PICKS
    ID['prem'] = PREMIUM
    ID['claims'] = CLAIMS
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
    ID['inf'] = INFINITY
    ID['and'] = AND
    ID['exp'] = EXP
    ID['wt'] = WEIGHTS
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
    ID['x'] = XS

Dec Language Grammar Specification
===================================

Here is the full DecL Grammar and a `grammar railroad diagram <_static/diagram.xhtml>`_.

.. run python aggregate.parser.py to update this file

.. literalinclude:: 4_agg_language_reference/ref_include.rst

.. _test suite programs:

Test Suite Programs
===================

To run the test suite for HTML output, svg graphics.

.. code-block:: python

    from aggregate.extensions.test_suite import TestSuite
    TestSuite().run('^[A-KNO]', 'All Aggregate Tests', 'svg')


.. literalinclude:: ../aggregate/agg/test_suite.agg
   :language: agg


To only parse::

    from aggregate import build
    filename = build.default_dir / 'test_suite.agg'
    assert filename.exists()

    build.logger_level(30)
    df = build.interpreter_file(filename=filename)

    df.query('error != 0')


``sly`` Parser
==================

The parser is built using the ``sly`` package, https://sly.readthedocs.io/en/latest/sly.html.


