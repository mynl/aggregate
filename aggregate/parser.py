"""
HACK ON THE ORIGINAL AGG SLY

lexer and parser specification for aggregate
============================================

Overview
    Implements the ``agg`` programming lanaguage

Example program
    The following short program replicates Thought Experiemnt 1 from Neil Bodoff's
    paper Capital Allocation by Percentile Layer

    ::

        port BODOFF1 note{Bodoff Thought Experiment No. 1}
            agg wind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed
            agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed

Preprocessing
    tab or four spaces needed and are replaced with space (program is on one line)


Ignored characters
    colon, comma, ( ) |

To view the grammar using a railroad diagram paste the specification below into
the Edit Grammar tab of https://www.bottlecaps.de/rr/ui and then View Diagram.
(Site diagram uses #DDDDDD as the base color.)

Language Specification
----------------------

The ```agg``` Language Grammar:

::

answer              	::= sev_out
                    	 | agg_out
                    	 | port_out
                    	 | expr

port_out            	::= port_name note agg_list

agg_list            	::= agg_list agg_out
                    	 | agg_out

agg_out             	::= agg_name builtin_aggregate note
                    	 | agg_name exposures layers SEV sev occ_reins freq agg_reins note

sev_out             	::= sev_out sev_name sev note
                    	 | sev_name sev note

freq                	::= MIXED ID expr expr
                    	 | MIXED ID expr
                    	 | EMPIRICAL numbers numbers
                    	 | FREQ expr expr
                    	 | FREQ expr
                    	 | FREQ

agg_reins           	::= NET OF reins_list AGGREGATE
                    	 | CEDED TO reins_list AGGREGATE
                    	 | 

occ_reins           	::= NET OF reins_list OCCURRENCE
                    	 | CEDED TO reins_list OCCURRENCE
                    	 | 

reins_list          	::= reins_list AND reins_clause
                    	 | reins_clause

reins_clause        	::= expr XS expr
                    	 | expr SHARE_OF expr XS expr
                    	 | expr PART_OF expr XS expr

sev                 	::= sev "!"
                    	 | sev LOCATION_ADD numbers
                    	 | numbers SCALE_MULTIPLY sev
                    	 | ids numbers CV numbers weights
                    	 | ids numbers weights
                    	 | ids numbers numbers weights
                    	 | ids xps
                    	 | CONSTANT expr
                    	 | builtinids numbers numbers
                    	 | builtinids

xps                 	::= XPS numbers numbers
                    	 | 

weights             	::= WEIGHTS EQUAL_WEIGHT expr
                    	 | WEIGHTS numbers
                    	 | 

layers              	::= numbers XS numbers
                    	 | 

note                	::= NOTE
                    	 | 

exposures           	::= SPECIFIED CLAIMS
                    	 | numbers CLAIMS
                    	 | numbers LOSS
                    	 | numbers PREMIUM AT numbers LR

builtinids          	::= BUILTINID

ids                 	::= "[" idl "]"
                    	 | ID

idl                 	::= idl ID
                    	 | ID

builtin_aggregate   	::= builtin_aggregate_dist TIMES expr
                    	 | expr TIMES builtin_aggregate_dist
                    	 | builtin_aggregate_dist

builtin_aggregate_dist	::= BUILTINID

sev_name            	::= SEV ID

agg_name            	::= AGG ID

port_name           	::= PORT ID

numbers             	::= "[" numberl "]"
                    	 | expr

numberl             	::= numberl expr
                    	 | expr

expr                	::= expr PLUS expr
                    	 | expr MINUS expr
                    	 | expr TIMES expr
                    	 | expr DIVIDE expr
                    	 | expr EXPONENT expr
                    	 | "(" expr ")"
                    	 | EXP "(" expr ")"
                    	 | expr PERCENT
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

EMPIRICAL               ::= 'empirical'

EXP                     ::= 'exp'

EXPONENT                ::= '^|**'

INFINITY                ::= 'inf|unlim|unlimited'

LOSS                    ::= 'loss'

LR                      ::= 'lr'

MIXED                   ::= 'mixed'

NET                     ::= 'net'

OCCURRENCE              ::= 'occurrence'

OF                      ::= 'of'

PART_OF                 ::= 'po'

PORT                    ::= 'port'

PREMIUM                 ::= 'premium|prem'

SEV                     ::= 'sev'

SHARE_OF                ::= 'so'

SPECIFIED               ::= 'specified'

TO                      ::= 'to'

WEIGHTS                 ::= 'wts|wt'

XPS                     ::= 'xps'

xs                      ::= "xs|x"

PERCENT                 ::= '%'

EXP                     ::= 'exp'

SCALE_MULTIPLY          ::= "@"

LOCATION_ADD            ::= "#"

parser.out parser debug information
-----------------------------------

Lexer term definition
---------------------

::

    tokens = {ID, BUILTINID, NOTE,
              SEV, AGG, PORT,
              PLUS, MINUS, TIMES, NUMBER,
              LOSS, PREMIUM, AT, LR, CLAIMS,
              XS,
              CV, WEIGHTS, EQUAL_WEIGHT, XPS,
              MIXED, FREQ
              }
    ignore = ' \t,\\:\\(\\)|'
    literals = {'[', ']'}

    # per manual, need to list longer tokens before shorter ones
    # NOTE = r'note\{[0-9a-zA-Z,\.\(\)\-=\+!\s]*\}'  # r'[^\}]+'
    NOTE = r'note\{[^\}]*\}'  # r'[^\}]+'
    BUILTINID = r'(sev|agg|port|meta)\.[a-zA-Z][a-zA-Z0-9_]*'
    FREQ = r'binomial|poisson|bernoulli|fixed'
    ID = r'[a-zA-Z][\.a-zA-Z0-9~]*'  # do not allow _ in line names, use ~ instead: BECAUSE p_ etc. makes _ special
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    EQUAL_WEIGHT = r'='
    ID['loss'] = LOSS
    ID['at'] = AT
    ID['cv'] = CV
    ID['premium'] = PREMIUM
    ID['prem'] = PREMIUM
    ID['lr'] = LR
    ID['claims'] = CLAIMS
    ID['claim'] = CLAIMS
    ID['xs'] = XS
    ID['x'] = XS
    ID['wts'] = WEIGHTS
    ID['wt'] = WEIGHTS
    ID['xps'] = XPS
    ID['mixed'] = MIXED
    ID['inf'] = NUMBER
    ID['sev'] = SEV
    ID['on'] = SEV
    ID['agg'] = AGG
    ID['port'] = PORT


Example Code
------------

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

References
----------

https://sly.readthedocs.io/en/latest/sly.html

"""
from sly import Lexer, Parser
import sly
import logging
import numpy as np
import warnings
from numpy import exp
import pandas as pd

logger = logging.getLogger(__name__)


class UnderwritingLexer(Lexer):

    tokens = {ID, BUILTINID, NOTE,
              SEV, AGG, PORT,
              NUMBER, INFINITY,
              PLUS, MINUS, TIMES, DIVIDE, SCALE_MULTIPLY, LOCATION_ADD,
              LOSS, PREMIUM, AT, LR, CLAIMS, SPECIFIED,
              XS,
              CV, WEIGHTS, EQUAL_WEIGHT, XPS, CONSTANT,
              MIXED, FREQ, EMPIRICAL,
              NET, OF, CEDED, TO, OCCURRENCE, AGGREGATE, PART_OF, SHARE_OF,
              AND, PERCENT,
              EXPONENT, EXP
              }

    ignore = ' \t,\\:|'
    literals = {'[', ']', '!', '(', ')'}

    # per manual, need to list longer tokens before shorter ones
    # simple but effective notes
    NOTE = r'note\{[^\}]*\}'  # r'[^\}]+'
    BUILTINID = r'(sev|agg|port|meta)\.[a-zA-Z][a-zA-Z0-9_:~]*'
    FREQ = r'binomial|poisson|bernoulli|pascal|geometric|fixed'

    # number regex including unary minus; need before MINUS else that grabs the minus sign in -3 etc.
    NUMBER = r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?'

    # do not allow _ in line names, use ~ or . or : instead: why: because p_ is used and _ is special
    # on honor system...really need two types of ID, it is OK in a portfolio name
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

    ID['occurrence'] = OCCURRENCE
    ID['unlimited'] = INFINITY
    ID['aggregate'] = AGGREGATE
    # when using an empirical freq the claim count is specified
    # must use "specified claims" ... sets e_n = -1
    # ID['part'] = PART
    # ID['share'] = SHARE
    ID['specified'] = SPECIFIED
    # constant severity
    ID['constant'] = CONSTANT
    # nps freq specification
    ID['empirical'] = EMPIRICAL
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

    # number regex including unary minus; need before MINUS else that grabs the minus sign in -3 etc.
    # @_(r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?')
    # def NUMBER(self, t):
    #     return float(t.value)

    @_(r'\n+')
    def newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        print(f"Illegal character '{t.value[0]:s}'")
        self.index += 1


class UnderwritingParser(Parser):
    # debugfile = 'c:\\temp\\parser.out'
    tokens = UnderwritingLexer.tokens
    precedence = (
        ('left', LOCATION_ADD),
        ('left', SCALE_MULTIPLY),
        ('left', PLUS, MINUS),
        ('left', TIMES, DIVIDE),
        # ('right', UMINUS),
        ('right', EXP),
        ('left', PERCENT),
        ('right', EXPONENT)
    )

    def __init__(self, safe_lookup_function, debug=False):
        self.debug = debug
        self.arg_dict = None
        self.sev_out_dict = None
        self.agg_out_dict = None
        self.port_out_dict = None
        self.reset()
        # instance of uw class to look up severities
        self._safe_lookup = safe_lookup_function

    def logger(self, msg, p):
        if self.debug is False:
            return
        logger.info(f'{msg:15s}')
        return
        nm = p._namemap
        sl = p._slice
        ans = []
        for k, v in nm.items():
            rhs = sl[v]
            if type(rhs) == sly.yacc.YaccSymbol:
                ans.append(f'{k}={rhs.value} (type: {rhs.type})')
            else:
                ans.append(f'{k}={rhs!s}')
        ans = "; ".join(ans)
        logger.info(f'{msg:15s}\n\t{ans}\n')

    def reset(self):
        # TODO Add sev_xs and sev_ps !!
        self.sev_out_dict = {}
        self.agg_out_dict = {}
        self.port_out_dict = {}

    @staticmethod
    def _check_vectorizable(value):
        """
        check the if value can be vectorized

        """
        if isinstance(value, float) or isinstance(value, int) or isinstance(value, np.ndarray):
            return value
        else:
            return np.array(value)

    # final answer exit points ===================================
    @_('sev_out')
    def answer(self, p):
        self.logger(
            f'answer <-- sev_out, created severity {p.sev_out}', p)
        return p.sev_out

    @_('agg_out')
    def answer(self, p):
        self.logger(
            f'answer <-- agg_out, created aggregate {p.agg_out}', p)
        return self.agg_out_dict[p.agg_out]

    @_('port_out')
    def answer(self, p):
        self.logger(f'answer <-- port_out, created portfolio {p.port_out} '
                    f'with {len(self.port_out_dict[p.port_out]["spec"])} aggregates', p)
        return p.port_out

    @_('expr')
    def answer(self, p):
        self.logger(f'answer <-- expr {p.expr} ', p)
        return p.expr

    # building portfolios =======================================
    @_('port_name note agg_list')
    def port_out(self, p):
        self.logger(
            f'port_out <-- port_name note agg_list', p)
        self.port_out_dict[p.port_name] = {'spec': p.agg_list, 'note': p.note}
        return p.port_name

    @_('agg_list agg_out')
    def agg_list(self, p):
        if p.agg_list is None:
            raise ValueError('ODD agg list is empty')
        p.agg_list.append(p.agg_out)
        self.logger(f'agg_list <-- agg_list agg_out', p)
        return p.agg_list

    @_('agg_out')
    def agg_list(self, p):
        self.logger(f'agg_list <-- agg_out', p)
        return [p.agg_out]

    # building aggregates ========================================
    @_('agg_name builtin_aggregate note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- agg_name builtin_aggregate note', p)
        if 'name' in p.builtin_aggregate:
            # otherwise will overwrite the agg name
            del p.builtin_aggregate['name']
        self.agg_out_dict[p.agg_name] = {
            'name': p.agg_name, **p.builtin_aggregate, 'note': p.note}
        return p.agg_name

    @_('agg_name exposures layers SEV sev occ_reins freq agg_reins note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- agg_name exposures layers SEV sev occ_reins freq agg_reins note', p)
        self.agg_out_dict[p.agg_name] = {'name': p.agg_name, **p.exposures, **p.layers, **p.sev,
                                         **p.occ_reins, **p.freq, **p.agg_reins, 'note': p.note}
        return p.agg_name

    # building severities ======================================
    @_('sev_out sev_name sev note')
    def sev_out(self, p):
        self.logger(
            f'sev_out <-- sev_out sev_name sev note', p)
        p.sev['note'] = p.note
        self.sev_out_dict[p.sev_name] = p.sev

    @_('sev_name sev note')
    def sev_out(self, p):
        self.logger(
            f'sev_out <-- sev_name sev note ', p)
        p.sev['note'] = p.note
        self.sev_out_dict[p.sev_name] = p.sev

    # frequency term ==========================================
    # for all frequency distributions claim count is determined by exposure / severity
    # EXCEPT for EMPIRICAL
    # only freq shape parameters need be entered
    # one and two parameter mixing distributions
    # no mixing here, just expr
    @_('MIXED ID expr expr')
    def freq(self, p):
        self.logger(
            f'freq <-- MIXED ID expr expr', p)
        return {'freq_name': p.ID, 'freq_a': p[2], 'freq_b': p[3]}

    @_('MIXED ID expr')
    def freq(self, p):
        self.logger(
            f'freq <-- MIXED ID expr', p)
        return {'freq_name': p.ID, 'freq_a': p.expr}

    @_('EMPIRICAL numbers numbers')
    def freq(self, p):
        self.logger(f'freq <-- EMPIRICAL numbers numbers', p)
        # nps discrete given severity...
        a = self._check_vectorizable(p.numbers0)
        a = np.array(a, dtype=int)
        b = self._check_vectorizable(p.numbers1)
        return {'freq_name': 'empirical', 'freq_a': a, 'freq_b': b}

    @_('FREQ expr expr')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ expr expr', p)
        if p.FREQ != 'pascal':
            warnings.warn(
                f'Illogical choice of frequency {p.FREQ}, expected pascal')
        return {'freq_name': p.FREQ, 'freq_a': p[1], 'freq_b': p[2]}

    # binomial p or TODO inflated poisson
    @_('FREQ expr')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ expr', p)
        if p.FREQ != 'binomial':
            warnings.warn(
                f'Illogical choice of frequency {p.FREQ}, expected binomial')
        return {'freq_name': p.FREQ, 'freq_a': p.expr}

    @_('FREQ')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ (zero param distribution)', p)
        if p.FREQ not in ('poisson', 'bernoulli', 'fixed', 'geometric'):
            logger.error(
                f'Illogical choice for FREQ {p.FREQ}, should be poisson, bernoulli, geometric, or fixed')
        return {'freq_name': p.FREQ}

    # agg reins clause ========================================
    @_('NET OF reins_list AGGREGATE')
    def agg_reins(self, p):
        self.logger(f'agg_reins <-- NET OF reins_list AGGREGATE', p)
        return {'agg_reins': p.reins_list, 'agg_kind': 'net of'}

    @_('CEDED TO reins_list AGGREGATE')
    def agg_reins(self, p):
        self.logger(f'agg_reins <-- NET OF reins_list AGGREGATE', p)
        return {'agg_reins': p.reins_list, 'agg_kind': 'ceded to'}

    # @_('SUBJECT TO reins_list AGGREGATE')
    # def agg_reins(self, p):
    #     # same as CEDED TO
    #     self.logger(f'agg_reins <-- NET OF reins_list AGGREGATE', p)
    #     return {'agg_reins': p.reins_list, 'agg_kind': 'ceded to'}

    @_("")
    def agg_reins(self, p):
        self.logger('agg_reins <-- missing agg reins', p)
        return {}

    # occ reins clause ========================================
    @_('NET OF reins_list OCCURRENCE')
    def occ_reins(self, p):
        self.logger(f'occ_reins <-- NET OF reins_list OCCURRENCE', p)
        return {'occ_reins': p.reins_list, 'occ_kind': 'net of'}

    @_('CEDED TO reins_list OCCURRENCE')
    def occ_reins(self, p):
        self.logger(f'occ_reins <-- NET OF reins_list OCCURRENCE', p)
        return {'occ_reins': p.reins_list, 'occ_kind': 'ceded to'}

    # @_('SUBJECT TO reins_list OCCURRENCE')
    # def occ_reins(self, p):
    #     # same as CEDED TO
    #     self.logger(f'occ_reins <-- NET OF reins_list OCCURRENCE', p)
    #     return {'occ_reins': p.reins_list, 'occ_kind': 'ceded to'}

    @_("")
    def occ_reins(self, p):
        self.logger('occ_reins <-- missing occ reins', p)
        return {}

    # reinsurance clauses  ====================================
    @_('reins_list AND reins_clause')
    def reins_list(self, p):
        self.logger(f'reins_list <-- reins_list AND reins_clause', p)
        p.reins_list.append(p.reins_clause)
        return p.reins_list

    @_('reins_clause')
    def reins_list(self, p):
        self.logger(f'reins_list <-- reins_clause becomes reins_list', p)
        return [p.reins_clause]

    @_('expr XS expr')
    def reins_clause(self, p):
        self.logger(
            f'reins_clause <-- expr XS expr {p[0]} po {p[0]} xs {p[2]}', p)
        if np.isinf(p[0]):
            return (1, p[0], p[2])
        else:
            # y p/o y xs a
            return (p[0], p[0], p[2])

    @_('expr SHARE_OF expr XS expr')
    def reins_clause(self, p):
        self.logger(
            f'reins_clause <-- expr SHARE_OF expr XS expr {p[0]} s/o {p[2]} xs {p[4]}', p)
        # here expr is the proportion...
        if np.isinf(p[2]):
            return (p[0], p[2], p[4])
        else:
            return (p[0] * p[2], p[2], p[4])

    @_('expr PART_OF expr XS expr')
    def reins_clause(self, p):
        self.logger(
            f'reins_clause <-- expr PART_OF expr XS expr {p[0]} p/o {p[2]} xs {p[4]}', p)
        return (p[0], p[2], p[4])

    # severity term ============================================
    @_('sev "!"')
    def sev(self, p):
        self.logger(f'sev <-- conditional flag set', p)
        p.sev['sev_conditional'] = False
        return p.sev

    @_('sev LOCATION_ADD numbers')
    def sev(self, p):
        self.logger(f'sev <-- sev LOCATION_ADD numbers', p)
        p.sev['sev_loc'] = UnderwritingParser._check_vectorizable(
            p.sev.get('sev_loc', 0))
        p.numbers = UnderwritingParser._check_vectorizable(p.numbers)
        p.sev['sev_loc'] += p.numbers
        return p.sev

    # must be sev LOCATION_ADD -number
    # @_('sev MINUS numbers')
    # def sev(self, p):
    #     self.logger(f'sev <-- sev MINUS numbers', p)
    #     p.sev['sev_loc'] = p.sev.get('sev_loc', 0) - p.numbers
    #     return p.sev

    @_('numbers SCALE_MULTIPLY sev')
    def sev(self, p):
        self.logger(f'sev <-- numbers SCALE_MULTIPLY sev', p)
        p.numbers = UnderwritingParser._check_vectorizable(p.numbers)
        if 'sev_mean' in p.sev:
            p.sev['sev_mean'] = UnderwritingParser._check_vectorizable(
                p.sev.get('sev_mean', 0))
            p.sev['sev_mean'] *= p.numbers
        # only scale scale if there is a scale (otherwise you double count)
        if 'sev_scale' in p.sev:
            p.sev['sev_scale'] = UnderwritingParser._check_vectorizable(
                p.sev.get('sev_scale', 0))
            p.sev['sev_scale'] *= p.numbers
        if 'sev_mean' not in p.sev:
            # e.g. Pareto has no mean and it is important to set the scale
            # but if there is a mean it handles the scaling and setting scale will
            # confuse the distribution maker
            p.sev['sev_scale'] = p.numbers
        # if there is a location it needs to scale too --- that's a curious choice!
        if 'sev_loc' in p.sev:
            p.sev['sev_loc'] = UnderwritingParser._check_vectorizable(
                p.sev['sev_loc'])
            p.sev['sev_loc'] *= p.numbers
        return p.sev

    @_('ids numbers CV numbers weights')
    def sev(self, p):
        self.logger(
            f'sev <-- ids numbers CV numbers weights', p)
        return {'sev_name':  p.ids, 'sev_mean':  p[1], 'sev_cv':  p[3], 'sev_wt': p.weights}

    @_('ids numbers weights')
    def sev(self, p):
        self.logger(
            f'sev <-- ids numbers weights', p)
        return {'sev_name': p.ids, 'sev_a':  p[1], 'sev_wt': p.weights}

    @_('ids numbers numbers weights')
    def sev(self, p):
        self.logger(
            f'sev <-- ids numbers numbers weights', p)
        return {'sev_name': p.ids, 'sev_a': p[1], 'sev_b': p[2], 'sev_wt': p.weights}

    # no weights with xps terms
    @_('ids xps')
    def sev(self, p):
        self.logger(f'sev <-- ids xps (ids should be (c|d)histogram', p)
        return {'sev_name': p.ids, **p.xps}

    @_('CONSTANT expr')
    def sev(self, p):
        self.logger(f'sev <-- CONSTANT expr', p)
        # syntactic sugar to specify a constant severity
        return {'sev_name': 'dhistogram', 'sev_xs': [p.expr], 'sev_ps': [1]}

    @_('XPS numbers numbers')
    def xps(self, p):
        self.logger(
            f'xps <-- XPS numbers numbers', p)
        return {'sev_xs':  p[1], 'sev_ps':  p[2]}

    @_('')
    def xps(self, p):
        self.logger('xps <-- missing xps term', p)
        return {}

    @_('WEIGHTS EQUAL_WEIGHT expr')
    def weights(self, p):
        self.logger(
            f'weights <-- WEIGHTS EQUAL_WEIGHTS expr ', p)
        return np.ones(int(p.expr)) / p.expr

    @_('WEIGHTS numbers')
    def weights(self, p):
        self.logger(f'weights <-- WEIGHTS numbers', p)
        return p.numbers

    @_('')
    def weights(self, p):
        self.logger('weights <-- missing weights term', p)
        return 1

    @_('builtinids numbers numbers')
    def sev(self, p):
        self.logger(
            f'sev <-- builtinds numbers numbers (log2={p[1]}, bs={p[2]})', p)
        requested_type = p.builtinids.split('.')[0]
        if requested_type == "meta":
            return {'sev_name': p.builtinids, 'sev_a': p[1], 'sev_b': p[2]}
        else:
            raise ValueError(
                f'Only meta type can be used with arguments, not {p.builtinids}')

    @_('builtinids')
    def sev(self, p):
        self.logger(f'sev <-- builtinds', p)
        # look up ID in uw
        # it is not accepetable to ask for an agg or port here; they need to be accessed through
        # meta. E.g. if you request and agg it will overwrite other (freq) variables defined
        # in the script...
        requested_type = p.builtinids.split('.')[0]
        if requested_type not in ("sev", "meta"):
            raise ValueError(
                f'built in type must be sev or meta, not {p.builtinids}')
        if requested_type == 'meta':
            return {'sev_name': p.builtinids}
        else:
            return self._safe_lookup(p.builtinids)

    # layer terms, optional ===================================
    @_('numbers XS numbers')
    def layers(self, p):
        self.logger(
            f'layers <-- numbers XS numbers', p)
        return {'exp_attachment': p[2], 'exp_limit': p[0]}

    @_('')
    def layers(self, p):
        self.logger('layers <-- missing layer term', p)
        return {}

    # optional note  ==========================================
    @_('NOTE')
    def note(self, p):
        self.logger(f'note <-- NOTE', p)
        return p.NOTE[5:-1]

    @_("")
    def note(self, p):
        self.logger("note <-- missing note term", p)
        return ''

    # exposures term ==========================================
    @_('SPECIFIED CLAIMS')
    def exposures(self, p):
        self.logger(f'exposures <- SPECIFIED CLAIMS', p)
        # a code that needs to be picked up later...
        # ONLY for use with EMPIRICAL/EMPIRICAL claim distribution
        # TO DO INTEGRATE CODE!
        return {'exp_en': -1}

    @_('numbers CLAIMS')
    def exposures(self, p):
        self.logger(f'exposures <-- numbers CLAIMS', p)
        return {'exp_en': p.numbers}

    @_('numbers LOSS')
    def exposures(self, p):
        self.logger(f'exposures <-- numbers LOSS', p)
        return {'exp_el': p.numbers}

    @_('numbers PREMIUM AT numbers LR')
    def exposures(self, p):
        self.logger(
            f'exposures <-- numbers PREMIUM AT numbers LR', p)
        return {'exp_premium': p[0], 'exp_lr': p[3], 'exp_el': p[0] * p[3]}

    # @_('numbers PREMIUM AT numbers')
    # def exposures(self, p):
    #     self.logger(
    #         f'resolving numbers PREMIUM AT numbers to exposures {p[0]} at {p[3]}', p)
    #     return {'exp_premium': p[0], 'exp_lr': p[3], 'exp_el': p[0] * p[3]}

    @_('BUILTINID')
    def builtinids(self, p):
        self.logger(f'buildinids <-- BUILTINID', p)
        return p.BUILTINID  # will always be treated as a list

    @_('"[" idl "]"')
    def ids(self, p):
        self.logger(f'ids <-- [idl]', p)
        return p.idl

    @_('idl ID')
    def idl(self, p):
        s1 = f'idl <-- idl ID'
        p.idl.append(p.ID)
        s1 += f'{p.idl}'
        self.logger(s1, p)
        return p.idl

    @_('ID')
    def idl(self, p):
        ans = [p.ID]
        self.logger(f'idl <-- ID', p)
        return ans

    @_('ID')
    def ids(self, p):
        self.logger(f'ids <-- ID ({p.ID})', p)
        return p.ID

    # elements made from named portfolios ========================
    @_('builtin_aggregate_dist TIMES expr')
    def builtin_aggregate(self, p):
        """  inhomogeneous change of scale """
        self.logger(f'builtin_aggregate <-- builtin_aggregate_dist TIMES expr', p)
        bid = p.builtin_aggregate_dist
        bid['exp_en'] = bid.get('exp_en', 0) * p.expr
        bid['exp_el'] = bid.get('exp_el', 0) * p.expr
        bid['exp_premium'] = bid.get('exp_premium', 0) * p.expr
        return bid

    @_('expr TIMES builtin_aggregate_dist')
    def builtin_aggregate(self, p):
        """homogeneous change of scale """
        self.logger(f'builtin_aggregate <-- expr TIMES builtin_aggregate_dist', p)
        # bid = built_in_dict, want to be careful not to add scale too much
        bid = p.builtin_aggregate_dist  # ? does this need copying. if so do in safelookup!
        if 'sev_mean' in bid:
            bid['sev_mean'] = bid['sev_mean'] * p.expr
        if 'sev_scale' in bid:
            bid['sev_scale'] = bid['sev_scale'] * p.expr
        if 'sev_loc' in bid:
            bid['sev_loc'] = bid['sev_loc'] * p.expr
        bid['exp_attachment'] = bid.get('exp_attachment', 0) * p.expr
        bid['exp_limit'] = bid.get('exp_limit', np.inf) * p.expr
        bid['exp_el'] = bid.get('exp_el', 0) * p.expr
        bid['exp_premium'] = bid.get('exp_premium', 0) * p.expr
        return bid

    @_('builtin_aggregate_dist')
    def builtin_aggregate(self, p):
        self.logger('builtin_aggregate <-- builtin_aggregate_dist', p)
        return p.builtin_aggregate_dist

    @_('BUILTINID')
    def builtin_aggregate_dist(self, p):
        # ensure lookup only happens here
        self.logger(f'builtin_aggregate_dist <-- BUILTINID', p)
        built_in_dict = self._safe_lookup(p.BUILTINID)
        return built_in_dict

    # ids =========================================================
    @_('SEV ID')
    def sev_name(self, p):
        self.logger(f'sev_name <-- SEV ID', p)
        return p.ID

    @_('AGG ID')
    def agg_name(self, p):
        self.logger(f'agg_name <-- AGG ID', p)
        # return {'name': p.ID}
        if p.ID.find('_') >= 0:
            raise ValueError(f'agg names cannot include _, you entered  {p.ID}. '
                             '(sev and port object names can include _.)')
        return p.ID

    @_('PORT ID')
    def port_name(self, p):
        self.logger(f'port_name <-- PORT ID', p)
        # return {'name': p.ID}
        return p.ID

    # vectors of numbers
    @_('"[" numberl "]"')
    def numbers(self, p):
        self.logger(f'numbers <-- [numberl]', p)
        return p.numberl

    @_('numberl expr')
    def numberl(self, p):
        self.logger(f'numberl <-- numberl expr (adding {p.expr} to list {p.numberl})', p)
        p.numberl.append(p.expr)
        return p.numberl

    @_('expr')
    def numberl(self, p):
        self.logger(f'numberl <-- expr', p)
        ans = [p.expr]
        return ans

    @_('expr')
    def numbers(self, p):
        self.logger('numbers <-- expr', p)
        return p.expr

    # implement simple calculator with exponents and exp as a convenience
    @_('expr PLUS expr')
    def expr(self, p):
        self.logger('expr <-- expr + expr', p)
        return p.expr0 + p.expr1

    @_('expr MINUS expr')
    def expr(self, p):
        self.logger('expr <-- expr - expr', p)
        return p.expr0 - p.expr1

    @_('expr TIMES expr')
    def expr(self, p):
        self.logger('expr <-- expr * expr', p)
        return p.expr0 * p.expr1

    @_('expr DIVIDE expr')
    def expr(self, p):
        self.logger('expr <-- expr / expr', p)
        return p.expr0 / p.expr1

    @_('expr EXPONENT expr')
    def expr(self, p):
        self.logger('expr <-- expr ** expr', p)
        return p.expr0 ** p.expr1

    @_('"(" expr ")"')
    def expr(self, p):
        self.logger('expr <-- (expr)', p)
        return p.expr

    # @_('MINUS "(" expr ")" %prec UMINUS')
    # def expr(self, p):
    #     self.logger('expr <-- MINUS(expr)', p)
    #     return -p.expr

    @_('EXP "(" expr ")"')
    def expr(self, p):
        self.logger('expr <-- EXP(expr)', p)
        return exp(p.expr)

    @_('expr PERCENT')
    def expr(self, p):
        self.logger('expr <-- expr PERCENT', p)
        return p.expr / 100

    @_('INFINITY')
    def expr(self, p):
        self.logger(f'expr <-- INFINITY', p)
        return np.inf

    @_('NUMBER')
    def expr(self, p):
        # number regex includes -1 automatically so no need for a UMINUS parse
        self.logger(f'expr <-- NUMBER, {p.NUMBER}', p)
        if p.NUMBER in ('inf', 'unlimited', 'unlim'):
            t = np.inf
        else:
            t = float(p.NUMBER)
        return t

    def error(self, p):
        if p:
            raise ValueError(p)
        else:
            raise ValueError('Unexpected end of file')


def safelookup(val):
    s = f'LOOKUP {val}'
    return {'sev_name': 'BUILTIN', 'sev_a': val}

def run_tests(where='', debug=False):
    """
    Run a bunch of tests
    """
    df = pd.read_csv('C:\\S\\TELOS\\Python\\aggregate_extensions_project\\aggregate2\\agg2_database.csv', index_col=0)
    if where != '':
        df = df.loc[df.index.str.match(where)]

    lexer = UnderwritingLexer()
    parser = UnderwritingParser(safelookup, debug)
    from pprint import PrettyPrinter
    pp = PrettyPrinter().pprint
    ans = {}
    errs = []
    if debug is True:
        for k, v in df.iterrows():
            v = v[0]
            print(v)
            print('='*len(v))

            parser.reset()
            try:
                x = parser.parse(lexer.tokenize(v))
            except ValueError as e:
                print('!!!!!!!!!!!!!!!!!!!!!!!!'*4)
                print('!!!!!!! Value Error !!!!' * 4)
                print('!!!!!!!!!!!!!!!!!!!!!!!!'*4)
                x = e
            if x is not None:
                pp(x)
            else:
                pp(parser.agg_out_dict[k])
        return errs, ans
    else:
        for k, v in df.iterrows():
            parser.reset()
            x = None
            v = v[0]
            try:
                x = parser.parse(lexer.tokenize(v))
            except (ValueError, TypeError) as e:
                errs.append([k, type(e)])
            ans[k] = x
        if len(errs) > 0:
            print(f'Errors reported:')
            pp(errs)
        else:
            print('No errors reported')
        return errs, ans

def run_one(v):
    """
    run single test in debug mode, you enter id as k and program as v
    """
    lexer = UnderwritingLexer()
    parser = UnderwritingParser(safelookup, True)
    from pprint import PrettyPrinter
    pp = PrettyPrinter().pprint
    print(v)
    print('='*len(v))
    parser.reset()
    try:
        x = parser.parse(lexer.tokenize(v))
    except ValueError as e:
        print('!!!!!!!!!!!!!!!!!!!!!!!!'*4)
        print('!!!!!!! Value Error !!!!' * 4)
        print('!!!!!!!!!!!!!!!!!!!!!!!!'*4)
        x = e
    if x is not None:
        pp(x)
    else:
        pp(parser.agg_out_dict[k])


if __name__ == '__main__':
    # print the grammar and add to this file as part of docstring
    # TODO fix comments!

    # may need to put an extra indent for rst to work properly
    # eg %run agg_parser.py to run in Jupyter

    start_string = '''Language Specification
----------------------

The ```agg``` Language Grammar:

::

'''
    end_string = 'parser.out parser debug information'

    with open(__file__, 'r') as f:
        txt = f.read()
    stxt = txt.split('@_')
    ans = {}
    for it in stxt[3:-2]:
        if it.find('# def') >= 0:
            # skip rows with a comment between @_ and def
            pass
        else:
            b = it.split('def')
            b0 = b[0].strip()[2:-2]
            try:
                b1 = b[1].split("(self, p):")[0].strip()
            except:
                print(it)
                exit()
            if b1 in ans:
                ans[b1] += [b0]
            else:
                ans[b1] = [b0]
    s = ''
    for k, v in ans.items():
        s += f'{k:<20s}\t::= {v[0]:<s}\n'
        for rhs in v[1:]:
            s += f'{" "*20}\t | {rhs:<s}\n'
        s += '\n'

    # finally add the language words
    # this is a bit manual, but these shouldnt change much...
    lang_words = '''
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

EMPIRICAL               ::= 'empirical|nps'

EXP                     ::= 'exp'

EXPONENT                ::= '^|**'

INFINITY                ::= 'inf|unlim|unlimited'

LOSS                    ::= 'loss'

LR                      ::= 'lr'

MIXED                   ::= 'mixed'

NET                     ::= 'net'

OCCURRENCE              ::= 'occurrence'

OF                      ::= 'of'

PART_OF                 ::= 'po'

PORT                    ::= 'port'

PREMIUM                 ::= 'premium|prem'

SEV                     ::= 'sev'

SHARE_OF                ::= 'so'

SPECIFIED               ::= 'specified'

TO                      ::= 'to'

WEIGHTS                 ::= 'wts|wt'

XPS                     ::= 'xps'

xs                      ::= "xs|x"

PERCENT                 ::= '%'

EXP                     ::= 'exp'

SCALE_MULTIPLY          ::= "@"

LOCATION_ADD            ::= "#"

'''

# PLUS =                  ::= '+'
# MINUS                   ::= '-'
# TIMES                   ::= '*'
# DIVIDE                  ::=  '/'

    s += lang_words
    print(s)
    # actually put into this file (uncomment)
    st = txt.find(start_string) + len(start_string)
    end = txt.find(end_string)
    txt = txt[0:st] + s + txt[end:]
    with open(__file__, 'w') as f:
        f.write(txt)
