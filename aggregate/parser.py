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

port_out            	::= PORT name note agg_list

agg_list            	::= agg_list agg_out
                    	 | agg_out

agg_out             	::= AGG name builtin_agg note
                    	 | AGG name exposures layers SEV sev occ_reins freq agg_reins note
                    	 | AGG name exposures layers builtin_sev occ_reins freq agg_reins note
                    	 | AGG name exposures layers dsev occ_reins freq agg_reins note
                    	 | AGG name dfreq dsev occ_reins agg_reins note
                    	 | builtin_agg agg_reins note

sev_out             	::= SEV name sev note
                    	 | SEV name dsev note

freq                	::= MIXED ID expr expr
                    	 | MIXED ID expr
                    	 | EMPIRICAL doutcomes dprobs
                    	 | FREQ expr expr
                    	 | FREQ expr
                    	 | FREQ

agg_reins           	::= AGGREGATE NET OF reins_list
                    	 | AGGREGATE CEDED TO reins_list
                    	 | 

occ_reins           	::= OCCURRENCE NET OF reins_list
                    	 | OCCURRENCE CEDED TO reins_list
                    	 | 

reins_list          	::= reins_list AND reins_clause
                    	 | reins_clause

reins_clause        	::= expr XS expr
                    	 | expr SHARE_OF expr XS expr
                    	 | expr PART_OF expr XS expr

sev                 	::= sev "!"
                    	 | sev LOCATION_ADD numbers
                    	 | numbers SCALE_MULTIPLY sev
                    	 | builtin_sev
                    	 | ids numbers CV numbers weights
                    	 | ids numbers weights
                    	 | ids numbers numbers weights
                    	 | ids xps
                    	 | CONSTANT expr

xps                 	::= XPS doutcomes dprobs
                    	 | 

dsev                	::= DSEV doutcomes dprobs

dfreq               	::= DFREQ doutcomes dprobs

doutcomes           	::= "[" numberl "]"
                    	 | "[" expr RANGE expr "]"
                    	 | "[" expr RANGE expr RANGE expr "]"

dprobs              	::= "[" numberl "]"
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

ids                 	::= "[" idl "]"
                    	 | ID

idl                 	::= idl ID
                    	 | ID

builtin_agg         	::= expr TIMES builtin_agg
                    	 | expr SCALE_MULTIPLY builtin_agg
                    	 | builtin_agg LOCATION_ADD expr
                    	 | BUILTIN_AGG

builtin_sev         	::= BUILTIN_SEV

name                	::= ID

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

DFREQ                   ::= 'dfreq'

DSEV                    ::= 'dsev'

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
from numpy import exp
import re
from pathlib import Path

logger = logging.getLogger(__name__)
DEBUGFILE = Path.home() / 'aggregate/parser/parser.out'


class UnderwritingLexer(Lexer):

    tokens = {ID, BUILTIN_AGG, BUILTIN_SEV,NOTE,
              SEV, AGG, PORT,
              NUMBER, INFINITY,
              PLUS, MINUS, TIMES, DIVIDE, SCALE_MULTIPLY, LOCATION_ADD,
              LOSS, PREMIUM, AT, LR, CLAIMS, SPECIFIED,
              XS,
              CV, WEIGHTS, EQUAL_WEIGHT, XPS, #  CONSTANT,
              MIXED, FREQ, EMPIRICAL,
              NET, OF, CEDED, TO, OCCURRENCE, AGGREGATE, PART_OF, SHARE_OF,
              AND, PERCENT,
              EXPONENT, EXP,
              DFREQ, DSEV, RANGE
              }

    ignore = ' \t,\\|'
    literals = {'[', ']', '!', '(', ')'}

    # per manual, need to list longer tokens before shorter ones
    # simple but effective notes
    NOTE = r'note\{[^\}]*\}'  # r'[^\}]+'
    BUILTIN_AGG = r'agg\.[a-zA-Z][a-zA-Z0-9_:~]*'
    BUILTIN_SEV = r'sev\.[a-zA-Z][a-zA-Z0-9_:~]*'
    # PORT_BUILTIN = r'port\.[a-zA-Z][a-zA-Z0-9_:~]*'
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
    RANGE = ':'

    ID['occurrence'] = OCCURRENCE
    ID['unlimited'] = INFINITY
    ID['aggregate'] = AGGREGATE

    ID['dfreq'] = DFREQ
    ID['dsev'] = DSEV

    # when using an empirical freq the claim count is specified
    # must use "specified claims" ... sets e_n = -1
    # ID['part'] = PART
    # ID['share'] = SHARE
    ID['specified'] = SPECIFIED
    # constant severity... now just use dsev [n]
    # ID['constant'] = CONSTANT
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

    @staticmethod
    def preprocess(program):
        """
        Separate preprocessor step so it can be called separately.

        preprocessing:
            remove // comments, through end of line
            remove \n in [] (vectors) e.g. put by f{np.linspace} TODO only works for 1d vectors
            ; mapped to newline
            backslash (line continuation) mapped to space
            split on newlines

        :param program:
        :return:
        """

        # handle \n in vectors; first item is outside, then inside... (multidimensional??)
        out_in = re.split(r'\[|\]', program)
        assert len(out_in) % 2  # must be odd
        odd = [t.replace('\n', ' ') for t in out_in[1::2]]  # replace inside []
        even = out_in[0::2]  # otherwise pass through
        # reassemble
        program = ' '.join([even[0]] + [f'[{o}] {e}' for o, e in zip(odd, even[1:])])

        # remove comments C++-style // xxx (can't have comments with # used for location shift)
        # must replace comments before changing other \ns
        program = re.sub(r'//[^\n]*$', r'\n', program, flags=re.MULTILINE)

        #  preprocessing: line continuation; \n\t or \n____ to space (for port agg element indents),
        # ; to new line, split on new line
        program = program.replace('\\\n', ' '). replace('\n\t', ' ').replace('\n    ', ' ').replace(';', '\n')

        # split program into lines, only accept len > 0
        program = [i.strip() for i in program.split('\n') if len(i.strip()) > 0]
        return program


class UnderwritingParser(Parser):
    # uncomment to write detailed grammar rules
    debugfile = Path.home() / 'aggregate/parser/parser.out'
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
        self.out_dict = None
        self.reset()
        # instance of uw class to look up severities
        self.safe_lookup = safe_lookup_function

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
        self.out_dict = {}

    @staticmethod
    def enhance_debugfile(f_out=''):
        """
        Put links in the parser.out debug file, if DEBUGFILE != ''

        :param f_out: Path or filename of output. If "" then DEBUGFILE.html used.
        :return:
        """

        if DEBUGFILE == '':
            return

        if f_out == '':
            f_out = DEBUGFILE.with_suffix('.html')
        else:
            f_out = Path(fn)

        txt = Path(DEBUGFILE).read_text(encoding='utf-8')
        txt = txt.replace('Grammar:\n', '<h1>Grammar:</h1>\n\n<pre>\n').replace('->', '<-')
        txt = re.sub(r'^Rule ([0-9]+)', r'<div id="rule_\1" />Rule \1', txt, flags=re.MULTILINE)
        txt = re.sub(r'^state ([0-9]+)$', r'<div id="state_\1" /><b>state \1</b>', txt, flags=re.MULTILINE)
        txt = re.sub(r'^    \(([0-9]+)\) ', r'    <a href="#rule_\1">Rule (\1)</a> ', txt, flags=re.MULTILINE)
        txt = re.sub(r'go to state ([0-9]+)', r'go to <a href="#state_\1">state (\1)</a>', txt, flags=re.MULTILINE)
        txt = re.sub(r'using rule ([0-9]+)', r'using <a href="#rule_\1">rule (\1)</a>', txt, flags=re.MULTILINE)
        txt = re.sub(r'in state ([0-9]+)', r'in <a href="#state_\1">state (\1)</a>', txt, flags=re.MULTILINE)

        f_out.write_text(txt + '\n</pre>', encoding='utf-8')

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
        return 'sev', p.sev_out

    @_('agg_out')
    def answer(self, p):
        self.logger(
            f'answer <-- agg_out, created aggregate {p.agg_out}', p)
        return 'agg', p.agg_out

    @_('port_out')
    def answer(self, p):
        self.logger(f'answer <-- port_out, created portfolio {p.port_out} '
                    f'with {len(self.out_dict[("port", p.port_out)]["spec"])} aggregates', p)
        return 'port', p.port_out

    @_('expr')
    def answer(self, p):
        self.logger(f'answer <-- expr {p.expr} ', p)
        return 'expr', p.expr

    # building portfolios =======================================
    @_('PORT name note agg_list')
    def port_out(self, p):
        self.logger(
            f'port_out <-- PORT name note agg_list', p)
        self.out_dict[("port", p.name)] = {'spec': p.agg_list, 'note': p.note}
        # print([self.out_dict[('agg', i)] for i in p.agg_list])
        return p.name

    @_('agg_list agg_out')
    def agg_list(self, p):
        if p.agg_list is None:
            raise ValueError('ODD agg list is empty')
        p.agg_list.append(p.agg_out)
        self.logger(f'agg_list <-- agg_list agg_out', p)
        return p.agg_list

    # building aggregates ========================================
    @_('agg_out')
    def agg_list(self, p):
        self.logger(f'agg_list <-- agg_out', p)
        return [p.agg_out]

    @_('AGG name builtin_agg note')
    def agg_out(self, p):
        # for use when you change the agg and need a new name
        self.logger(
            f'agg_out <-- AGG name builtin_aggregate note', p)
        self.out_dict[("agg", p.name)] = {
            'name': p.name, **p.builtin_agg, 'note': p.note}
        return p.name

    @_('AGG name exposures layers SEV sev occ_reins freq agg_reins note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- AGG name exposures layers SEV sev occ_reins freq agg_reins note', p)
        self.out_dict[("agg", p.name)] = {'name': p.name, **p.exposures, **p.layers, **p.sev,
                                         **p.occ_reins, **p.freq, **p.agg_reins, 'note': p.note}
        return p.name

    @_('AGG name exposures layers builtin_sev occ_reins freq agg_reins note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- AGG name exposures layers builtin_sev occ_reins freq agg_reins note', p)
        self.out_dict[("agg", p.name)] = {'name': p.name, **p.exposures, **p.layers, **p.builtin_sev,
                                         **p.occ_reins, **p.freq, **p.agg_reins, 'note': p.note}
        return p.name

    # separate treatment of dsev forbids 3 @ dsev # 4; that is also not handled for xps sevs
    # though it does parse. But with this separate treatment you do not need the sev keyword
    # if sev <-- dsev then in use you'd need sev dsev [1] which is no net gain.
    @_('AGG name exposures layers dsev occ_reins freq agg_reins note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- AGG name exposures layers dsev occ_reins freq agg_reins note', p)
        self.out_dict[("agg", p.name)] = {'name': p.name, **p.exposures, **p.layers, **p.dsev,
                                         **p.occ_reins, **p.freq, **p.agg_reins, 'note': p.note}
        return p.name

    # AMBIGUOUS: net of 3 x 2 ?? occ or agg? and dfreq [1 2] [2 4], is 2 4 the probs
    # or the start of exposure layering.
    # at_('name dfreq layers SEV sev occ_reins agg_reins note')
    # to the general with dfreq use specified claims and put freq term at the end.
    # for reinstatements modeing with occ re may be useful, hence could have the
    # following but doesn't seem worth it. Encourage specified
    # at_('agg_name dfreq SEV sev occ_reins note')
    # DEF agg_out(self, p):
    #     self.logger(
    #         f'agg_out <-- agg_name dfreq layers SEV sev occ_reins agg_reins note', p)
    #     self.out_dict[("agg", p.agg_name)] = {'name': p.agg_name, **p.dfreq,  **p.sev,
    #                                      **p.occ_reins, **p.agg_reins, 'note': p.note}
    #     return p.agg_name
    #
    # as above, can't have layers and can't have occ and agg without a split
    @_('AGG name dfreq dsev occ_reins agg_reins note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- AGG name dfreq dsev occ_reins note', p)
        self.out_dict[("agg", p.name)] = {'name': p.name, **p.dfreq, **p.dsev,
                                         **p.occ_reins, **p.agg_reins, 'note': p.note}
        return p.name

    @_('builtin_agg agg_reins note')
    def agg_out(self, p):
        # no change to the built in agg, allows agg.A as a legitimate agg (called A)
        self.logger(
            f'agg_out <-- builtin_agg agg_reins note', p)
        # print(p.builtin_agg)
        self.out_dict[("agg", p.builtin_agg['name'])] = {**p.builtin_agg, **p.agg_reins, 'note': p.note}
        return p.builtin_agg['name']

    # building severities =====================================
    @_('SEV name sev note')
    def sev_out(self, p):
        self.logger(
            f'sev_out <-- sev name sev note ', p)
        p.sev['name'] = p.name
        p.sev['note'] = p.note
        self.out_dict[("sev", p.name)] = p.sev
        return p.name

    @_('SEV name dsev note')
    def sev_out(self, p):
        self.logger(
            f'sev_out <-- sev name dsev note ', p)
        p.dsev['name'] = p.name
        p.dsev['note'] = p.note
        self.out_dict[("sev", p.name)] = p.dsev
        return p.name

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

    @_('EMPIRICAL doutcomes dprobs')
    def freq(self, p):
        self.logger(f'freq <-- EMPIRICAL doutcomes dprobs', p)
        # nps discrete given severity...
        a = np.array(p.doutcomes, dtype=int)
        if len(p.dprobs) == 0:
            ps = np.ones_like(a) / len(a)
        else:
            ps = p.dprobs
        return {'freq_name': 'empirical', 'freq_a': a, 'freq_b': ps}

    @_('FREQ expr expr')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ expr expr', p)
        if p.FREQ != 'pascal':
            logger.warn(
                f'Illogical choice of frequency {p.FREQ}, expected pascal')
        return {'freq_name': p.FREQ, 'freq_a': p[1], 'freq_b': p[2]}

    # binomial p or TODO inflated poisson
    @_('FREQ expr')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ expr', p)
        if p.FREQ != 'binomial':
            logger.warn(
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
    @_('AGGREGATE NET OF reins_list')
    def agg_reins(self, p):
        self.logger(f'agg_reins <-- AGGREGATE NET OF reins_list', p)
        return {'agg_reins': p.reins_list, 'agg_kind': 'net of'}

    @_('AGGREGATE CEDED TO reins_list')
    def agg_reins(self, p):
        self.logger(f'agg_reins <--  AGGREGATE CEDED TO reins_list', p)
        return {'agg_reins': p.reins_list, 'agg_kind': 'ceded to'}

    @_("")
    def agg_reins(self, p):
        self.logger('agg_reins <-- missing agg reins', p)
        return {}

    # occ reins clause ========================================
    @_('OCCURRENCE NET OF reins_list')
    def occ_reins(self, p):
        self.logger(f'occ_reins <-- OCCURRENCE NET OF reins_list', p)
        return {'occ_reins': p.reins_list, 'occ_kind': 'net of'}

    @_('OCCURRENCE CEDED TO reins_list')
    def occ_reins(self, p):
        self.logger(f'occ_reins <-- OCCURRENCE CEDED TO reins_list', p)
        return {'occ_reins': p.reins_list, 'occ_kind': 'ceded to'}

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
        # will need to use ugly sev LOCATION_ADD -number
        self.logger(f'sev <-- sev LOCATION_ADD numbers', p)
        p.sev['sev_loc'] = UnderwritingParser._check_vectorizable(
            p.sev.get('sev_loc', 0))
        p.numbers = UnderwritingParser._check_vectorizable(p.numbers)
        p.sev['sev_loc'] += p.numbers
        return p.sev

    @_('numbers SCALE_MULTIPLY sev')
    def sev(self, p):
        self.logger(f'sev <-- numbers SCALE_MULTIPLY sev', p)
        p.numbers = UnderwritingParser._check_vectorizable(p.numbers)
        if 'sev_mean' in p.sev:
            p.sev['sev_mean'] = UnderwritingParser._check_vectorizable(
                p.sev.get('sev_mean', 0))
            p.sev['sev_mean'] *= p.numbers
        # only scale if there is a scale (otherwise you double count)
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

    @_('builtin_sev')
    def sev(self, p):
        # allow sev.NAME to become a sev
        self.logger('sev <-- builtin_sev', p)
        if 'name' in p.builtin_sev:
            # print('NAME IS HERE...', p.builtin_sev)
            del p.builtin_sev['name']
        else:
            # print('WHY IS NAME NOT HERE?', p.builtin_sev)
            pass
        return p.builtin_sev

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

    # @_('CONSTANT expr')
    # def sev(self, p):
    #     self.logger(f'sev <-- CONSTANT expr', p)
    #     # syntactic sugar to specify a constant severity
    #     return {'sev_name': 'dhistogram', 'sev_xs': [p.expr], 'sev_ps': [1]}

    @_('XPS doutcomes dprobs')
    def xps(self, p):
        self.logger('xps <-- XPS doutcomes dprobs', p)
        if len(p.dprobs) == 0:
            ps = np.ones_like(p.doutcomes) / len(p.doutcomes)
        else:
            ps = p.dprobs
        return {'sev_xs':  p.doutcomes, 'sev_ps':  ps}

    @_('')
    def xps(self, p):
        self.logger('xps <-- missing xps term', p)
        return {}

    @_('DSEV doutcomes dprobs')
    def dsev(self, p):
        self.logger('dsev <-- DSEV doutcomes dprobs', p)
        # need to check probs has been populated
        if len(p.dprobs) == 0:
            ps = np.ones_like(p.doutcomes) / len(p.doutcomes)
        else:
            ps = p.dprobs
        return {'sev_name': 'dhistogram', 'sev_xs': p.doutcomes, 'sev_ps': ps}

    @_('DFREQ doutcomes dprobs')
    def dfreq(self, p):
        self.logger('dfreq <-- DFREQ doutcomes dprobs', p)
        # need to check probs has been populated
        if len(p.dprobs) == 0:
            b = np.ones_like(p.doutcomes) / len(p.doutcomes)
        else:
            b = p.dprobs
        return {'freq_name': 'empirical', 'freq_a': p.doutcomes, 'freq_b': b, 'exp_en': -1}

    # never valid for this to be a single number not in []
    @_('"[" numberl "]"')
    def doutcomes(self, p):
        self.logger('doutcomes <-- [numberl]', p)
        a = self._check_vectorizable(p.numberl)
        return a

    @_('"[" expr RANGE expr "]"')
    def doutcomes(self, p):
        self.logger('doutcomes <-- [expr : expr]', p)
        return np.arange(p[1], p[3] + 1)

    @_('"[" expr RANGE expr RANGE expr "]"')
    def doutcomes(self, p):
        self.logger('doutcomes <-- [expr : expr : expr]', p)
        return np.arange(p[1], p[3] + 1, p[5])

    @_('"[" numberl "]"')
    def dprobs(self, p):
        self.logger('dprobs <-- [numberl]', p)
        a = self._check_vectorizable(p.numberl)
        return a

    @_('')
    def dprobs(self, p):
        self.logger('dprobs <-- missing dprobs term', p)
        return []

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
        return {'exp_premium': p[0], 'exp_lr': p[3], 'exp_el': np.array(p[0]) * np.array(p[3])}

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
    @_('expr TIMES builtin_agg')
    def builtin_agg(self, p):
        """  inhomogeneous change of scale """
        self.logger(
            f'builtin_aggregate <-- builtin_agg TIMES expr', p)
        bid = p.builtin_agg.copy()
        bid['name'] += '_i_scaled'
        bid['exp_en'] = bid.get('exp_en', 0) * p.expr
        bid['exp_el'] = bid.get('exp_el', 0) * p.expr
        bid['exp_premium'] = bid.get('exp_premium', 0) * p.expr
        return bid

    @_('expr SCALE_MULTIPLY builtin_agg')
    def builtin_agg(self, p):
        """homogeneous change of scale """
        self.logger('builtin_agg <-- expr TIMES builtin_agg', p)
        # bid = built_in_dict, want to be careful not to add scale too much
        bid = p.builtin_agg.copy()
        bid['name'] += '_h_scaled'
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

    @_('builtin_agg LOCATION_ADD expr')
    def builtin_agg(self, p):
        """
        translation
        :param p:
        :return:
        """
        self.logger('builtin_agg <-- builtin_agg LOCATION_ADD expr', p)
        # bid = built_in_dict, want to be careful not to add scale too much
        bid = p.builtin_agg.copy()
        bid['name'] += '_translated'
        if 'sev_loc' in bid:
            bid['sev_loc'] += p.expr
        else:
            bid['sev_loc'] = p.expr
        return bid

    @_('BUILTIN_AGG')
    def builtin_agg(self, p):
        # ensure lookup only happens here
        self.logger(f'builtin_agg <-- AGG_BUILTIN', p)
        built_in_dict = self.safe_lookup(p.BUILTIN_AGG)
        return built_in_dict

    @_('BUILTIN_SEV')
    def builtin_sev(self, p):
        # ensure lookup only happens here
        # unlike aggs, will never just say sev.A
        # but, can allow  agg A 1 claim sev.B and for that need to distinguish from a sev
        self.logger(f'builtin_agg <-- SEV_BUILTIN ({p.BUILTIN_SEV})', p)
        built_in_dict = self.safe_lookup(p.BUILTIN_SEV)
        return built_in_dict

    # @_('BUILTIN_PORT')
    # def builtin_port(self, p):
    #     # ensure lookup only happens here
    #     self.logger(f'builtin_agg <-- PORT_BUILTIN', p)
    #     built_in_dict = self.safe_lookup(p.PORT_BUILTIN)
    #     return built_in_dict


    # ids =========================================================
    @_('ID')
    def name(self, p):
        self.logger(f'name <-- ID = {p.ID}', p)
        return p.ID

    # vectors of numbers
    @_('"[" numberl "]"')
    def numbers(self, p):
        self.logger(f'numbers <-- [numberl]', p)
        return p.numberl

    @_('numberl expr')
    def numberl(self, p):
        self.logger(
            f'numberl <-- numberl expr (adding {p.expr} to list {p.numberl})', p)
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


def grammar(add_to_doc=False, save_to_fn=''):
    '''
    write the grammar at the top of the file as a docstring
    '''
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

DFREQ                   ::= 'dfreq'

DSEV                    ::= 'dsev'

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
    if add_to_doc is True:
        st = txt.find(start_string) + len(start_string)
        end = txt.find(end_string)
        txt = txt[0:st] + s + txt[end:]
        with open(__file__, 'w') as f:
            f.write(txt)
    if save_to_fn == '':
        save_to_fn = Path.home() / 'aggregate/parser/grammar.md'
    Path(save_to_fn).write_text(s, encoding='utf-8')


if __name__ == '__main__':
    # print the grammar and add to this file as part of docstring
    # TODO fix comments!

    # may need to put an extra indent for rst to work properly
    # eg %run agg_parser.py to run in Jupyter
    grammar(add_to_doc=True)
