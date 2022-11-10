*****************************
**agg** Language Reference
*****************************

.. To view the grammar using a railroad diagram paste the specification below into
   the Edit Grammar tab of https://www.bottlecaps.de/rr/ui and then View Diagram.
   (Site diagram uses #DDDDDD as the base color.)

Language Overview
=================



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

Ignored characters: tab, colon, comma, and |. These characters can be used to improve readability.

Aggregate names must not include underscore. Portfolio names may include underscore. Names can include a period, ``A.Basic.01``.

::

    tokens = {ID, BUILTIN_AGG, BUILTIN_SEV,NOTE,
              SEV, AGG, PORT,
              NUMBER, INFINITY,
              PLUS, MINUS, TIMES, DIVIDE, SCALE_MULTIPLY, LOCATION_ADD,
              LOSS, PREMIUM, AT, LR, CLAIMS, SPECIFIED,
              XS,
              CV, WEIGHTS, EQUAL_WEIGHT, XPS,
              MIXED, FREQ, EMPIRICAL, TWEEDIE,
              NET, OF, CEDED, TO, OCCURRENCE, AGGREGATE, PART_OF, SHARE_OF,
              AND, PERCENT,
              EXPONENT, EXP,
              DFREQ, DSEV, RANGE
              }

    ignore = ' \t,\\|'
    literals = {'[', ']', '!', '(', ')'}

    NOTE = r'note\{[^\}]*\}'  # r'[^\}]+'
    BUILTIN_AGG = r'agg\.[a-zA-Z][a-zA-Z0-9_:~]*'
    BUILTIN_SEV = r'sev\.[a-zA-Z][a-zA-Z0-9_:~]*'
    FREQ = r'binomial|poisson|bernoulli|pascal|geometric|fixed'

    # number regex including unary minus; need before MINUS else that grabs the minus sign in -3 etc.
    NUMBER = r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?'

    ID = r'[a-zA-Z][\.:~_a-zA-Z0-9]*'
    EXPONENT = r'\^|\*\*'
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    DIVIDE = r'/'
    PERCENT = '%'
    SCALE_MULTIPLY = r'@'
    LOCATION_ADD = '#'
    EQUAL_WEIGHT = r'='
    RANGE = ':'

    ID['occurrence'] = OCCURRENCE
    ID['unlimited'] = INFINITY
    ID['aggregate'] = AGGREGATE

    ID['dfreq'] = DFREQ
    ID['dsev'] = DSEV

    # ID['part'] = PART
    # ID['share'] = SHARE
    # when using an empirical freq the claim count is specified
    # must use "specified claims" ... sets e_n = -1
    ID['specified'] = SPECIFIED
    ID['empirical'] = EMPIRICAL
    ID['tweedie'] = TWEEDIE
    ID['premium'] = PREMIUM
    ID['mixed'] = MIXED
    ID['unlim'] = INFINITY
    ID['claims'] = CLAIMS
    ID['ceded'] = CEDED
    ID['claim'] = CLAIMS
    ID['loss'] = LOSS
    ID['prem'] = PREMIUM
    ID['port'] = PORT
    ID['net'] = NET
    ID['sev'] = SEV
    ID['agg'] = AGG
    ID['nps'] = EMPIRICAL
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
    ID['x'] = XS



Language Grammar Specification
===============================

Here is the full ```agg``` Language Grammar.
`Grammar railroad diagram <_static/diagram.xhtml>`_.


::

    answer                  ::= sev_out
                             | agg_out
                             | port_out
                             | distortion_out
                             | expr

    distortion_out          ::= DISTORTION name ids expr
                             | DISTORTION name ID expr "[" numberl "]"

    port_out                ::= PORT name note agg_list

    agg_list                ::= agg_list agg_out
                             | agg_out

    agg_out                 ::= AGG name exposures layers sev_clause occ_reins freq agg_reins note
                             | AGG name dfreq layers sev_clause occ_reins agg_reins note
                             | AGG name TWEEDIE expr expr expr note
                             | AGG name builtin_agg note
                             | builtin_agg agg_reins note

    sev_out                 ::= SEV name sev note
                             | SEV name dsev note

    freq                    ::= freq ZM expr
                             | freq ZT
                             | MIXED ID expr expr
                             | MIXED ID expr
                             | FREQ expr expr
                             | FREQ expr
                             | FREQ

    agg_reins               ::= AGGREGATE NET OF reins_list
                             | AGGREGATE CEDED TO reins_list
                             |  %prec LOW

    occ_reins               ::= OCCURRENCE NET OF reins_list
                             | OCCURRENCE CEDED TO reins_list
                             |

    reins_list              ::= reins_list AND reins_clause
                             | reins_clause
                             | tower

    reins_clause            ::= expr XS expr
                             | expr SHARE_OF expr XS expr
                             | expr PART_OF expr XS expr

    sev_clause              ::= SEV sev %prec LOW
                             | dsev
                             | BUILTIN_SEV

    sev                     ::= sev "!"
                             | sev PLUS numbers
                             | sev MINUS numbers
                             | numbers TIMES sev
                             | ids numbers CV numbers weights
                             | ids numbers numbers weights
                             | ids numbers weights
                             | ids xps
                             | ids
                             | BUILTIN_SEV

    xps                     ::= XPS doutcomes dprobs

    dsev                    ::= DSEV doutcomes dprobs

    dfreq                   ::= DFREQ doutcomes dprobs

    doutcomes               ::= "[" numberl "]"
                             | "[" expr RANGE expr "]"
                             | "[" expr RANGE expr RANGE expr "]"

    dprobs                  ::= "[" numberl "]"
                             |

    weights                 ::= WEIGHTS EQUAL_WEIGHT expr
                             | WEIGHTS "[" numberl "]"
                             |

    layers                  ::= numbers XS numbers
                             | tower
                             |

    tower                   ::= TOWER doutcomes

    note                    ::= NOTE
                             |  %prec LOW

    exposures               ::= numbers CLAIMS
                             | numbers LOSS
                             | numbers PREMIUM AT numbers LR

    ids                     ::= "[" idl "]"
                             | ID

    idl                     ::= idl ID
                             | ID

    builtin_agg             ::= expr HOMOG_MULTIPLY builtin_agg
                             | expr TIMES builtin_agg
                             | builtin_agg PLUS expr
                             | builtin_agg MINUS expr
                             | BUILTIN_AGG

    name                    ::= ID

    numbers                 ::= "[" numberl "]"
                             | expr

    numberl                 ::= numberl expr
                             | expr

    expr                    ::= term

    term                    ::= term DIVIDE factor
                             | factor

    factor                  ::= power
                             | "(" term ")"
                             | EXP "(" term ")"

    power                   ::= atom EXPONENT factor
                             | atom

    atom                    ::= NUMBER PERCENT
                             | INFINITY
                             | NUMBER

    FREQ                    ::= 'binomial|poisson|bernoulli|pascal|geometric|fixed'

    BUILTINID               ::= 'sev|agg|port|meta.ID'

    NOTE                    ::= 'note{TEXT}'

    EQUAL_WEIGHT            ::= "="

    AGG                     ::= 'agg'

    AGGREGATE               ::= 'aggregate'

    AND                     ::= 'and'

    AT                      ::= 'at'

    CEDED                   ::= 'ceded'

    CLAIMS                  ::= 'claims|claim'

    CONSTANT                ::= 'constant'

    CV                      ::= 'cv'

    DFREQ                   ::= 'dfreq'

    DSEV                    ::= 'dsev'

    EXP                     ::= 'exp'

    EXPONENT                ::= '^|**'

    HOMOG_MULTIPLY          ::= "@"

    INFINITY                ::= 'inf|unlim|unlimited'

    LOSS                    ::= 'loss'

    LR                      ::= 'lr'

    MIXED                   ::= 'mixed'

    NET                     ::= 'net'

    OCCURRENCE              ::= 'occurrence'

    OF                      ::= 'of'

    PART_OF                 ::= 'po'

    PERCENT                 ::= '%'

    PORT                    ::= 'port'

    PREMIUM                 ::= 'premium|prem'

    SEV                     ::= 'sev'

    SHARE_OF                ::= 'so'

    TO                      ::= 'to'

    WEIGHTS                 ::= 'wts|wt'

    XPS                     ::= 'xps'

    xs                      ::= "xs|x"


Test Suite
==========

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


References
==========

https://sly.readthedocs.io/en/latest/sly.html


