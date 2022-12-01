*****************************
**agg** Language Reference
*****************************

.. To view the grammar using a railroad diagram paste the
   specification below into
   the Edit Grammar tab of https://www.bottlecaps.de/rr/ui and then View Diagram.
   (Site diagram uses #DDDDDD as the base color.)


This section describes how an ``agg`` program is pre-processed, lexed, and parsed according to the grammar specification. It reports the results of interpreting the builtin test suite of NNN programs.

The ``agg`` language :ref:`introduction <design and purpose>` describes its design and purpose.

Pre-Processing
==============

Programs are processed one line at a time. Before passing to the lexer, the following pre-processing occurs.

1. Remove C++ style  // comments, through end of line
2. Remove \\n in [ ] (vectors) that appear from  using ``f'{np.linspace(...)    }'``
3. Semicolons are mapped to newline
4. Backslash (Python sytle line continuations) are mapped to space
5. \\n\\t is replaced with space, supporting the tabbed indented Portfolio layout
6. Split on newlines


Lexer Term Definitions
======================

Ignored characters: tab, colon, comma, and pipe. These characters can be used to improve readability.

Aggregate names must not include underscore. Portfolio names may include underscore. Names can include a period, ``A.Basic.01``.

::


    tokens = {ID, BUILTIN_AGG, BUILTIN_SEV,NOTE,
              SEV, AGG, PORT,
              NUMBER, INFINITY,
              PLUS, MINUS, TIMES, DIVIDE, HOMOG_MULTIPLY, # SCALE_MULTIPLY, LOCATION_ADD,
              LOSS, PREMIUM, AT, LR, CLAIMS,
              XS,
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

    # per manual, need to list longer tokens before shorter ones
    NOTE = r'note\{[^\}]*\}'  # r'[^\}]+'
    BUILTIN_AGG = r'agg\.[a-zA-Z][a-zA-Z0-9_:~]*'
    BUILTIN_SEV = r'sev\.[a-zA-Z][a-zA-Z0-9_:~]*'
    FREQ = 'binomial|pascal|poisson|bernoulli|geometric|fixed' # |empirical'
    DISTORTION = 'dist(ortion)?'

    # number regex including unary minus; need before MINUS else that grabs the minus sign in -3 etc.
    NUMBER = r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?'

    ID = r'[a-zA-Z][\.:~_a-zA-Z0-9]*'
    EXPONENT = r'\^|\*\*'
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    DIVIDE = '/'
    PERCENT = '%'
    HOMOG_MULTIPLY = '@'
    EQUAL_WEIGHT = '='
    RANGE = ':'

    ID['occurrence'] = OCCURRENCE
    ID['unlimited'] = INFINITY
    ID['aggregate'] = AGGREGATE
    ID['tweedie'] = TWEEDIE
    ID['premium'] = PREMIUM
    ID['tower'] = TOWER
    ID['mixed'] = MIXED
    ID['unlim'] = INFINITY
    ID['prem'] = PREMIUM
    ID['claims'] = CLAIMS
    ID['ceded'] = CEDED
    ID['claim'] = CLAIMS
    ID['dfreq'] = DFREQ
    ID['dsev'] = DSEV
    ID['loss'] = LOSS
    ID['port'] = PORT
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

Language Grammar Specification
===============================

Here is the full ```agg``` Language Grammar and a `grammar railroad diagram <_static/diagram.xhtml>`_.

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


Parser References
==================

https://sly.readthedocs.io/en/latest/sly.html


