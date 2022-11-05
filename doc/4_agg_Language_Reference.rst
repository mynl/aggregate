=============================
**agg** Language Reference
=============================


Language Overview
=================


Other
-----


To view the grammar using a railroad diagram paste the specification below into
the Edit Grammar tab of https://www.bottlecaps.de/rr/ui and then View Diagram.
(Site diagram uses #DDDDDD as the base color.)


Sample Programs
===============

Simple example
--------------

The following short program replicates Thought Experiemnt 1 from Neil Bodoff's
paper Capital Allocation by Percentile Layer  ::

    port BODOFF1 note{Bodoff Thought Experiment No. 1}
        agg wind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed
        agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed


Other examples
--------------

::

    port Complex~Portfolio~Mixed
        agg LineA  50  claims           sev lognorm 12 cv [2, 3, 4] wt [.3 .5 .2] mixed gamma 0.4
        agg LineB  24  claims 10 x 5    sev lognorm 12 cv [{', '.join([str(i) for i in np.linspace(2,5, 20)])}] wt=20 mixed gamma 0.35
        agg LineC 124  claims 120 x 5   sev lognorm 16 cv 3.4                     mixed gamma 0.45

    port Complex~Portfolio
        agg Line3  50  claims [5 10 15] x 0         sev lognorm 12 cv [1, 2, 3]        mixed gamma 0.25
        agg Line9  24  claims [5 10 15] x 5         sev lognorm 12 cv [1, 2, 3] wt=3   mixed gamma 0.25

    port Portfolio~2
        agg CA 500 prem at .5 lr 15 x 12  sev gamma 12 cv [2 3 4] wt [.3 .5 .2] mixed gamma 0.4
        agg FL 1.7 claims 100 x 5         sev 10000 * pareto 1.3 - 10000        poisson
        agg IL 1e-8 * agg.CMP
        agg OH agg.CMP * 1e-8
        agg NY 500 prem at .5 lr 15 x 12  sev [20 30 40 10] * gamma [9 10 11 12] cv [1 2 3 4] wt =4 mixed gamma 0.4

    sev proda 30000 * lognorm 2
    sev prodc:   50000 * lognorm(3)
    sev weird    50000 * beta(1, 4) + 10000
    sev premsop1 25000 * lognorm 2.3; sev premsop2 35000 * lognorm 2.4;
    sev premsop3 45000 * lognorm 2.8

    agg Agg1     20 claims 10 x 2 sev lognorm 12 cv 0.2 mixed gamma 0.8
    agg Agg2     20 claims 10 x 2 sev 15 * lognorm 2.5  poisson;
    sev premsop1 25000 * lognorm 2.3;
    agg Agg3     20 claims 10 x 2 on 25 * lognorm 2 fixed;

    port MyFirstPortfolio
        agg A1: 50  claims          sev gamma 12 cv .30 (mixed gamma 0.014)
        agg A2: 50  claims 30 xs 10 sev gamma 12 cv .30 (mixed gamma 0.014)
        agg A3: 50  claims          sev gamma 12 cv 1.30 (mixed gamma 0.014)
        agg A4: 50  claims 30 xs 20 sev gamma 12 cv 1.30 (mixed gamma 0.14)
        agg B 15 claims 15 xs 15 sev lognorm 12 cv 1.5 + 2 mixed gamma 4.8
        agg Cat 1.7 claims 25 xs 5  sev 25 * pareto 1.3 0 - 25 poisson
        agg ppa: 1e-8 * agg.PPAL

    port distortionTest
        agg mix    50 claims              [50, 100, 150, 200] xs 0  sev lognorm 12 cv [1,2,3,4]    poisson
        agg low    500 premium at 0.5     5 xs 5                    sev gamma 12 cv .30            mixed gamma 0.2
        agg med    500 premium at 0.5 lr  15 xs 10                  sev gamma 12 cv .30            mixed gamma 0.4
        agg xsa    50  claims             30 xs 10                  sev gamma 12 cv .30            mixed gamma 1.2
        agg hcmp   1e-8 * agg.CMP
        agg ihmp   agg.PPAL * 1e-8



Below are a series of programs illustrating the different ways exposure, frequency and severity can be
broadcast together, several different types of severity and all the different types of severity. ::

        # use to create sev and aggs so can illustrate use of sev. and agg. below
        sev sev1 lognorm 10 cv .3
        agg Agg0 1 claim sev lognorm 10 cv .09 fixed


        agg Agg1  1 claim sev {10*np.exp(-.3**2/2)} @ lognorm .3      fixed note{{sigma=.3 mean=10}}
        agg Agg2  1 claim sev {10*np.exp(-.3**2/2)} @ lognorm .3 # 5  fixed note{{shifted right by 5}}''' \
        '''
        agg Agg3  1 claim sev 10 @ lognorm 0.5 cv .3                  fixed note{mean 0.5 scaled by 10 and cv 0.3}
        agg Agg4  1 claim sev 10 @ lognorm 1 cv .5 + 5                fixed note{shifted right by 5}

        agg Agg5  1 claim sev 10 @ gamma .3                           fixed note{gamma distribution....can use any two parameter scipy.stats distribution plus expon, uniform and normal}
        agg Agg6  1 claim sev 10 @ gamma 1 cv .3 # 5                  fixed note{mean 10 x 1, cv 0.3 shifted right by 5}

        agg Agg7  1 claim sev 2 @ pareto 1.6  # -2                      fixed note{pareto alpha=1.6 lambda=2}
        agg Agg8  1 claim sev 2 @ uniform 5 # 2.5                     fixed note{uniform 2.5 to 12.5}

        agg Agg9  1 claim 10 x  2 sev lognorm 20 cv 1.5               fixed note{10 x 2 layer, 1 claim}
        agg Agg10 10 loss 10 xs 2 sev lognorm 20 cv 1.5               fixed note{10 x 2 layer, total loss 10, derives requency}
        agg Agg11 14 prem at .7    10 x 1 sev lognorm 20 cv 1.5       fixed note{14 prem at .7 lr derive frequency}
        agg Agg11 14 prem at .7 lr 10 x 1 sev lognorm 20 cv 1.5       fixed note{14 prem at .7 lr derive frequency, lr is optional}

        agg Agg12: 14 prem at .7 lr (10 x 1) sev (lognorm 20 cv 1.5)  fixed note{trailing semi and other punct ignored};

        agg Agg13: 1 claim sev 50 @ beta 3 2 # 10 fixed note{scaled and shifted beta, two parameter distribution}
        agg Agg14: 1 claim sev 100 @ expon # 10   fixed note{exponential single parameter, needs scale, optional shift}
        agg Agg15: 1 claim sev 10 @ norm # 50     fixed note{normal is single parameter too, needs scale, optional shift}

        # any scipy.stat distribution taking one parameter can be used; only cts vars supported on R+ make sense
        agg Agg16: 1 claim sev 1 * invgamma 4.07 fixed  note{inverse gamma distribution}

        # mixtures
        agg MixedLine1: 1 claim 25 xs 0 sev lognorm 10                   cv [0.2, 0.4, 0.6, 0.8, 1.0] wts=5             fixed note{equally weighted mixture of 5 lognormals different cvs}
        agg MixedLine2: 1 claim 25 xs 0 sev lognorm [10, 15, 20, 25, 50] cv [0.2, 0.4, 0.6, 0.8, 1.0] wts=5             fixed note{equal weighted mixture of 5 lognormals different cvs and means}
        agg MixedLine3: 1 claim 25 xs 0 sev lognorm 10                   cv [0.2, 0.4, 0.6, 0.8, 1.0] wt [.2, .3, .3, .15, .05]   fixed note{weights scaled to equal 1 if input}

        # limit profile
        agg LimitProfile1: 1 claim [1, 5, 10, 20] xs 0 sev lognorm 10 cv 1.2 wt [.50, .20, .20, .1]   fixed note{maybe input EL by band for wt}
        agg LimitProfile2: 5 claim            20  xs 0 sev lognorm 10 cv 1.2 wt [.50, .20, .20, .1]   fixed note{input EL by band for wt}
        agg LimitProfile3: [10 10 10 10] claims [inf 10 inf 10] xs [0 0 5 5] sev lognorm 10 cv 1.25   fixed note{input counts directly}

        # limits and distribution blend
        agg Blend1 50  claims [5 10 15] x 0         sev lognorm 12 cv [1, 1.5, 3]          fixed note{options all broadcast against one another, 50 claims of each}
        agg Blend2 50  claims [5 10 15] x 0         sev lognorm 12 cv [1, 1.5, 3] wt=3     fixed note{options all broadcast against one another, 50 claims of each}

        agg Blend5cv1  50 claims  5 x 0 sev lognorm 12 cv 1 fixed
        agg Blend10cv1 50 claims 10 x 0 sev lognorm 12 cv 1 fixed
        agg Blend15cv1 50 claims 15 x 0 sev lognorm 12 cv 1 fixed

        agg Blend5cv15  50 claims  5 x 0 sev lognorm 12 cv 1.5 fixed
        agg Blend10cv15 50 claims 10 x 0 sev lognorm 12 cv 1.5 fixed
        agg Blend15cv15 50 claims 15 x 0 sev lognorm 12 cv 1.5 fixed

        # semi colon can be used for newline and backslash works
        agg Blend5cv3  50 claims  5 x 0 sev lognorm 12 cv 3 fixed; agg Blend10cv3 50 claims 10 x 0 sev lognorm 12 cv 3 fixed
        agg Blend15cv3 50 claims 15 x 0 sev \
        lognorm 12 cv 3 fixed

        # not sure if it will broadcast limit profile against severity mixture...
        agg LimitProfile4: [10 30 15 5] claims [inf 10 inf 10] xs [0 0 5 5] sev lognorm 10 cv [1.0, 1.25, 1.5] wts=3  fixed note{input counts directly}

        # the logo
        agg logo 1 claim {np.linspace(10, 250, 20)} xs 0 sev lognorm 100 cv 1 fixed'''

        # empirical distributions
        agg dHist1 1 claim sev dhistogram xps [1, 10, 40] [.5, .3, .2] fixed     note{discrete histogram}
        agg cHist1 1 claim sev chistogram xps [1, 10, 40] [.5, .3, .2] fixed     note{continuous histogram, guessed right hand endpiont}
        agg cHist2 1 claim sev chistogram xps [1 10 40 45] [.5 .3 .2]  fixed     note{continuous histogram, explicit right hand endpoint, don't need commas}
        agg BodoffWind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed   note{examples from Bodoffs paper}
        agg BodoffQuake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed

        # set up fixed sev for future use
        sev One dhistogram xps [1] [1]   note{a certain loss of 1}

        # sev, agg and port: using built in objects [have to exist prior to running program]
        agg ppa:       0.01 * agg.PPAL       note{this is using lmult on aggs, needs a dictionary specification to adjust means}
        agg cautoQS:   1e-5 * agg.CAL        note{lmult is quota share or scale for rmul see below }
        agg cautoClms: agg.CAL * 1e-5        note{rmult adjusts the claim count}

        # scaling works with distributions already made by uw
        agg mdist: 5000 * agg.dHist1

        # frequency options
        agg FreqFixed      10 claims sev sev.One fixed
        agg FreqPoisson    10 claims sev sev.One poisson                   note{Poisson frequency}
        agg FreqBernoulli  .8 claims sev sev.One bernoulli               note{Bernoulli en is frequency }
        agg FreqBinomial   10 claims sev sev.One binomial 0.5
        agg FreqPascal     10 claims sev sev.One pascal .8 3

        # mixed freqs
        agg FreqNegBin     10 claims sev sev.One (mixed gamma 0.65)     note{gamma mixed Poisson = negative binomial}
        agg FreqDelaporte  10 claims sev sev.One mixed delaporte .65 .25
        agg FreqIG         10 claims sev sev.One mixed ig  .65
        agg FreqSichel     10 claims sev sev.One mixed delaporte .65 -0.25
        agg FreqSichel.gamma  10 claims sev sev.One mixed sichel.gamma .65 .25
        agg FreqSichel.ig     10 claims sev sev.One mixed sichel.ig  .65 .25
        agg FreqBeta       10 claims sev sev.One mixed beta .5  4  note{second param is max mix}


Pre-processing
==============

Programs are processed one line at a time. Before passing to the lexer, the following pre-processing occurs.

1. Remove C++ style  // comments, through end of line
2. Remove \\n in [ ] (vectors) that appear from  using ``f'{np.linspace(...)    }'``
3. Semicolons are mapped to newline
4. Backslash (Python sytle line continuations) are mapped to space
5. \\n\\t is replaced with space, supporting the tabbed indented Portfolio layout
6. Split on newlines


Lexer term definitions
======================

Ignored characters: tab, colon, comma, ( ) |. Thus, parenthesis and colons can be used to improve readability.

Aggregate names must not include underscore. Portfolio names may include underscore.

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



Language Specification
======================

The ```agg``` Language Grammar:

::

answer              	::= sev_out
                    	 | agg_out
                    	 | port_out
                    	 | distortion_out
                    	 | expr

distortion_out      	::= DISTORTION name ids expr
                    	 | DISTORTION name ID expr "[" numberl "]"

port_out            	::= PORT name note agg_list

agg_list            	::= agg_list agg_out
                    	 | agg_out

agg_out             	::= AGG name exposures layers sev_clause occ_reins freq agg_reins note
                    	 | AGG name dfreq layers sev_clause occ_reins agg_reins note
                    	 | AGG name TWEEDIE expr expr expr note
                    	 | AGG name builtin_agg note
                    	 | builtin_agg agg_reins note

sev_out             	::= SEV name sev note
                    	 | SEV name dsev note

freq                	::= freq ZM expr
                    	 | freq ZT
                    	 | MIXED ID expr expr
                    	 | MIXED ID expr
                    	 | FREQ expr expr
                    	 | FREQ expr
                    	 | FREQ

agg_reins           	::= AGGREGATE NET OF reins_list
                    	 | AGGREGATE CEDED TO reins_list
                    	 |  %prec LOW

occ_reins           	::= OCCURRENCE NET OF reins_list
                    	 | OCCURRENCE CEDED TO reins_list
                    	 | 

reins_list          	::= reins_list AND reins_clause
                    	 | reins_clause
                    	 | tower

reins_clause        	::= expr XS expr
                    	 | expr SHARE_OF expr XS expr
                    	 | expr PART_OF expr XS expr

sev_clause          	::= SEV sev %prec LOW
                    	 | dsev
                    	 | BUILTIN_SEV

sev                 	::= sev "!"
                    	 | sev PLUS numbers
                    	 | sev MINUS numbers
                    	 | numbers TIMES sev
                    	 | ids numbers CV numbers weights
                    	 | ids numbers numbers weights
                    	 | ids numbers weights
                    	 | ids xps
                    	 | ids
                    	 | BUILTIN_SEV

xps                 	::= XPS doutcomes dprobs

dsev                	::= DSEV doutcomes dprobs

dfreq               	::= DFREQ doutcomes dprobs

doutcomes           	::= "[" numberl "]"
                    	 | "[" expr RANGE expr "]"
                    	 | "[" expr RANGE expr RANGE expr "]"

dprobs              	::= "[" numberl "]"
                    	 | 

weights             	::= WEIGHTS EQUAL_WEIGHT expr
                    	 | WEIGHTS "[" numberl "]"
                    	 | 

layers              	::= numbers XS numbers
                    	 | tower
                    	 | 

tower               	::= TOWER doutcomes

note                	::= NOTE
                    	 |  %prec LOW

exposures           	::= numbers CLAIMS
                    	 | numbers LOSS
                    	 | numbers PREMIUM AT numbers LR

ids                 	::= "[" idl "]"
                    	 | ID

idl                 	::= idl ID
                    	 | ID

builtin_agg         	::= expr HOMOG_MULTIPLY builtin_agg
                    	 | expr TIMES builtin_agg
                    	 | builtin_agg PLUS expr
                    	 | builtin_agg MINUS expr
                    	 | BUILTIN_AGG

name                	::= ID

numbers             	::= "[" numberl "]"
                    	 | expr

numberl             	::= numberl expr
                    	 | expr

expr                	::= term

term                	::= term DIVIDE factor
                    	 | factor

factor              	::= power
                    	 | "(" term ")"
                    	 | EXP "(" term ")"

power               	::= atom EXPONENT factor
                    	 | atom

atom                	::= NUMBER PERCENT
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


References
----------

https://sly.readthedocs.io/en/latest/sly.html


