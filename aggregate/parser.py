"""
lexer and parser specification for aggregate
============================================

Overview
--------

Ignored colon, comma, ( ) |

A markdown pipe table: ignores pipes and first two rows

Valid program

| name        | expos                 | limit                    | sev                                               | freq              |
|:------------|:----------------------|:-------------------------|:--------------------------------------------------|:------------------|
| big_mixture | 50 claims             | [50, 100, 150, 200] xs 0 | on lognorm 12 cv [1,2,3,4] wts [0.25 .25 .25 .25] | poisson           |
| A1          | 500 premium at 0.5    |                          | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A1          | 500 premium at 0.5 lr |                          | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A2          | 50  claims            | 30 xs 10                 | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A3          | 50  claims            |                          | on gamma 12 cv .30                                | mixed gamma 0.014 |
| A4          | 50  claims            | 30 xs 20                 | on gamma 12 cv .30                                | mixed gamma 0.14  |
| hcmp        | 1000 * uw.cmp         |                          |                                                   |                   |
| incmp       | uw.cmp * 0.001        |                          |                                                   |                   |

Language Specification
----------------------

    ans 		name exposures layers sevs freq
                name builtinagg

    name 		ID   # name for portfolio

    # id list
    idl 		ID
                idl ID

    ids 		ID
                [ idl ]

    numbers 	NUMBER
                [ numberl ]

    numberl 	NUMBER
                numberl NUMBER

    builtin_agg	uw.ID
                NUMBER TIMES builtin_agg
                builtin_agg TIMES NUMBER

    exposures 	numbers LOSS
                numbers CLAIMS
                numbers PREMIUM AT numbers
                numbers PREMIUM AT numbers LR
                numbers PREMIUM numbers LR
                empty

    xps         xps numbers numbers
                empty

    layers		numbers XS numbers
                empty

    # singular sev term...is a list ; in agg it is called agg_sev ==> do not use sevs
    sev 		builtins
                ids numbers numbers WT weights XPS numbers numbers
                ids numbers CV numbers WT weights
                numbers * sev
                sev * numbers
                sev + numbers

    weights 	numbers
                empty

    freq 		POISSON
                FIXED   # number must be defined in expos term!
                ID NUMBER
                ID NUMBER NUMBER
                empty

parser.out parser debug information
-----------------------------------

Grammar:

Rule 0     S' -> ans
Rule 1     ans -> name exposures layers sev_term freq
Rule 2     ans -> name builtin_aggregate
Rule 3     freq -> <empty>
Rule 4     freq -> POISSON
Rule 5     freq -> FIXED
Rule 6     freq -> MIXED ids numbers
Rule 7     freq -> MIXED ids numbers numbers
Rule 8     sev_term -> ON sev
Rule 9     sev -> builtinids
Rule 10    sev -> ids numbers numbers weights xps
Rule 11    sev -> ids numbers weights
Rule 12    sev -> ids numbers CV numbers weights
Rule 13    sev -> numbers TIMES sev  [precedence=left, level=2]
Rule 14    sev -> sev MINUS numbers  [precedence=left, level=1]
Rule 15    sev -> sev PLUS numbers  [precedence=left, level=1]
Rule 16    xps -> <empty>
Rule 17    xps -> XPS numbers numbers
Rule 18    weights -> <empty>
Rule 19    weights -> WEIGHTS numbers
Rule 20    layers -> <empty>
Rule 21    layers -> numbers XS numbers
Rule 22    exposures -> numbers PREMIUM AT numbers
Rule 23    exposures -> numbers PREMIUM AT numbers LR
Rule 24    exposures -> numbers LOSS
Rule 25    exposures -> numbers CLAIMS
Rule 26    builtinids -> BUILTINID
Rule 27    builtinids -> [ builtinidl ]
Rule 28    builtinidl -> BUILTINID
Rule 29    builtinidl -> builtinidl BUILTINID
Rule 30    ids -> ID
Rule 31    ids -> [ idl ]
Rule 32    idl -> ID
Rule 33    idl -> idl ID
Rule 34    numbers -> NUMBER
Rule 35    numbers -> [ numberl ]
Rule 36    numberl -> NUMBER
Rule 37    numberl -> numberl NUMBER
Rule 38    builtin_aggregate -> BUILTINID
Rule 39    builtin_aggregate -> NUMBER TIMES BUILTINID
Rule 40    builtin_aggregate -> BUILTINID TIMES NUMBER
Rule 41    name -> ID

Terminals, with rules where they appear:

AT                   : 22 23
BUILTINID            : 26 28 29 38 39 40
CLAIMS               : 25
CV                   : 12
FIXED                : 5
ID                   : 30 32 33 41
LOSS                 : 24
LR                   : 23
MINUS                : 14
MIXED                : 6 7
NUMBER               : 34 36 37 39 40
ON                   : 8
PLUS                 : 15
POISSON              : 4
PREMIUM              : 22 23
TIMES                : 13 39 40
WEIGHTS              : 19
XPS                  : 17
XS                   : 21
[                    : 27 31 35
]                    : 27 31 35
error                :

Nonterminals, with rules where they appear:

ans                  : 0
builtin_aggregate    : 2
builtinidl           : 27 29
builtinids           : 9
exposures            : 1
freq                 : 1
idl                  : 31 33
ids                  : 6 7 10 11 12
layers               : 1
name                 : 1 2
numberl              : 35 37
numbers              : 6 7 7 10 10 11 12 12 13 14 15 17 17 19 21 21 22 22 23 23 24 25
sev                  : 8 13 14 15
sev_term             : 1
weights              : 10 11 12
xps                  : 10

Lexer term definition
---------------------


    tokens = {ID, PLUS, MINUS, TIMES, NUMBER, CV, LOSS, PREMIUM, AT, LR, CLAIMS, XS, MIXED,
              FIXED, POISSON, BUILTINID, WEIGHTS, XPS, ON}
    ignore = ' \t,\\:\\(\\)|'
    literals = {'[', ']'}

    BUILTINID = r'uw\.[a-zA-Z_][a-zA-Z0-9_]*'
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    ID['loss'] = LOSS
    ID['at'] = AT
    ID['cv'] = CV
    ID['premium'] = PREMIUM
    ID['prem'] = PREMIUM
    ID['lr'] = LR
    ID['on'] = ON
    ID['claims'] = CLAIMS
    ID['claim'] = CLAIMS
    ID['xs'] = XS
    ID['wts'] = WEIGHTS
    ID['xps'] = XPS
    ID['mixed'] = MIXED
    ID['poisson'] = POISSON
    ID['fixed'] = FIXED
    ID['inf'] = NUMBER

Testing Code
------------


    from aggregate.underwriter import Underwriter
    uu = Underwriter(debug=True)

    # test cases

    uw.test_write('myname: 100 premium at .4 lr lognorm 12 .3')
    uw.test_write('myname: 100 premium at .4 lr lognorm 12 cv .3')
    uw.test_write('myname: uw.cmp')
    uw.test_write('mycmp: uw.cmp; my1000cmp: 1000 * uw.cmp; mycmp1000 uw.cmp*0.001')
    uw.test_write('myname:  uw.cmp * 0.000001')

    uw.test_write('''A: 500 premium at 0.5   gamma 12 cv .30 (mixed gamma 0.014)
    A1: 500 premium at 0.5 lr  gamma 12 cv .30 (mixed gamma 0.014)
    A2: 50  claims 30 xs 10  gamma 12 cv .30 (mixed gamma 0.014)
    A3: 50  claims    gamma 12 cv .30 (mixed gamma 0.014)
    A4: 50  claims 30 xs 20  gamma 12 cv .30 (mixed gamma 0.14)
    cmp: 1000 * uw.cmp''')

    uw.test_write(''' B 15 claims 15 xs 15 lognorm 12 cv 1.5 + 2 mixed gamma 4.8
    Cat 1.7 claims 25 xs 5 25 * pareto 1.3 0 - 25 poisson ''')

    uw.test_write('weighted 1000 loss 500 xs 100 lognorm 12 [0.3, 0.5, 0.9] wts [.3333, .3333, .3334] ')


References
----------

https://sly.readthedocs.io/en/latest/sly.html

"""
import pandas as pd
from sly import Lexer, Parser
import logging
import numpy as np


class UnderwritingLexer(Lexer):
    tokens = {ID, PLUS, MINUS, TIMES, NUMBER, CV, LOSS, PREMIUM, AT, LR, CLAIMS, XS, MIXED,
              FIXED, POISSON, BUILTINID, WEIGHTS, XPS, SEV}
    ignore = ' \t,\\:\\(\\)|'
    literals = {'[', ']'}

    BUILTINID = r'uw\.[a-zA-Z_][a-zA-Z0-9_]*'
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    ID['loss'] = LOSS
    ID['at'] = AT
    ID['cv'] = CV
    ID['premium'] = PREMIUM
    ID['prem'] = PREMIUM
    ID['lr'] = LR
    # ID['on'] = ON
    ID['claims'] = CLAIMS
    ID['claim'] = CLAIMS
    ID['xs'] = XS
    ID['x'] = XS
    ID['wts'] = WEIGHTS
    ID['xps'] = XPS
    ID['mixed'] = MIXED
    ID['poisson'] = POISSON
    ID['fixed'] = FIXED
    ID['inf'] = NUMBER
    ID['sev'] = SEV

    @_(r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?')
    def NUMBER(self, t):
        if t.value == 'INF' or t.value == 'inf':
            t.value = np.inf
        else:
            t.value = float(t.value)
        return t

    @_(r'\n+')
    def newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        print(f"Illegal character '{t.value[0]:s}'")
        self.index += 1


class UnderwritingParser(Parser):
    # debugfile = 'parser.out'
    tokens = UnderwritingLexer.tokens
    precedence = (('left', PLUS, MINUS), ('left', TIMES))

    def __init__(self, safe_lookup_function, debug=False):
        self.arg_dict = None
        self.reset()
        self.type = ''
        # instance of uw class to look up severities
        self._safe_lookup = safe_lookup_function
        if debug:
            def _print(s):
                print(s)

            self.p = _print
        else:
            def _print(s):
                pass

            self.p = _print

    def reset(self):
        # TODO pull from Aggregate automatically ...
        # TODO Add sev_xs and sev_ps !!
        # in order to allow for missing terms this must reflect sensible defaults
        self.arg_dict = dict(name="", exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                             sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_scale=0, sev_loc=0, sev_wt=1,
                             freq_name='poisson', freq_a=0, freq_b=0)

    # built from a built in book
    @_('name builtin_aggregate')
    def ans(self, p):
        self.p(f'exiting through builtin_aggregate {p.builtin_aggregate}')
        self.type = 'aggregate'
        logging.info('UnderwritingParser | Exiting through built in aggregate')
        pass

    # standard spec expos [layers] sevs [freq]
    @_('name exposures layers SEV sev freq')
    def ans(self, p):
        self.p(f'exiting through name exposures layers SEV sev freq')
        self.type = 'aggregate'
        logging.info('UnderwritingParser | Exiting through name exposures limit SEV sev freq')
        pass

    # allow just to load a severity...
    @_('SEV name sev')
    def ans(self, p):
        self.p('Exiting through SEV name sev')
        self.type = 'severity'
        logging.info('UnderwritingParser | Exiting through sev')
        pass

    # frequency term ==========================================
    @_('MIXED ids numbers numbers')
    def freq(self, p):
        self.p(f'MIXED ids numbers numbers {p.ids}, {p.numbers}, {p.numbers} two param freq, numbers.1=CVs')
        self.arg_dict['freq_name'] = 'poisson'  # p.ids  TODO once freq dists implemented this needs to change
        self.arg_dict['freq_a'] = p[2]
        self.arg_dict['freq_b'] = p[3]

    @_('MIXED ids numbers')
    def freq(self, p):
        self.p(f'MIXED ids numbers {p.ids}, {p.numbers} single param freq, numbers=CVs')
        self.arg_dict['freq_name'] = 'poisson'  # p.ids  TODO, as above
        self.arg_dict['freq_a'] = p.numbers

    @_('FIXED')
    def freq(self, p):
        self.p('FIXED')
        self.arg_dict['freq_name'] = 'fixed'

    @_('POISSON')
    def freq(self, p):
        self.p('POISSON')
        self.arg_dict['freq_name'] = 'poisson'

    @_('')
    def freq(self, p):
        self.p('missing frequency term')
        return True

    # severity term ============================================
    # @_("ON sev")
    # def sev_term(self, p):
    #     self.p(f'resolving ON sev to sev_term')
    #     return True

    @_('sev PLUS numbers')
    def sev(self, p):
        self.p(f'resolving sev PLUS numbers to sev {p.numbers}')
        # this won't work...
        self.arg_dict['sev_loc'] += p.numbers

    @_('sev MINUS numbers')
    def sev(self, p):
        self.p(f'resolving sev MINUS numbers to sev {p.numbers}')
        self.arg_dict['sev_loc'] -= p.numbers

    @_('numbers TIMES sev')
    def sev(self, p):
        self.p(f'resolving numbers TIMES sev to sev {p.numbers}')
        self.arg_dict['sev_mean'] *= p.numbers
        # only scale scale if there is a scale (otherwise you double count)
        # if self.arg_dict['sev_scale']:
        self.arg_dict['sev_scale'] = p.numbers

    @_('ids numbers CV numbers weights')
    def sev(self, p):
        self.p(f'resolving ids numbers CV numbers sev{p[1]}, {p[3]}')
        self.arg_dict['sev_name'] = p.ids
        self.arg_dict['sev_mean'] = p[1]
        self.arg_dict['sev_cv'] = p[3]
        return True

    @_('ids numbers weights')
    def sev(self, p):
        self.p(f'resolving ids numbers to sev {p[1]} (one param dist)')
        self.arg_dict['sev_name'] = p.ids
        self.arg_dict['sev_a'] = p[1]
        return True

    @_('ids numbers numbers weights xps')
    def sev(self, p):
        self.p(f'resolving ids numbers numbers to sev {p[1]}, {p[2]} (two param dist)')
        self.arg_dict['sev_name'] = p.ids
        self.arg_dict['sev_a'] = p[1]
        self.arg_dict['sev_b'] = p[2]
        return True

    @_('XPS numbers numbers')
    def xps(self, p):
        self.p(f'XPS numbers numbers resolving to xs and ps {p[1]}, {p[2]}')
        self.arg_dict['sev_xs'] = p[1]
        self.arg_dict['sev_ps'] = p[2]
        return True

    @_('')
    def xps(self, p):
        self.p('missing xps term')

    @_('WEIGHTS numbers')
    def weights(self, p):
        self.p(f'WEIGHTS numbers resolving to weights {p.numbers}')
        self.arg_dict['sev_wt'] = p.numbers
        return True

    @_('')
    def weights(self, p):
        self.p('missing weights term')

    @_('builtinids')
    def sev(self, p):
        self.p(f'builtinds to sevs {p.builtinids}')
        # look up ID in uw
        for n in p.builtinids:
            built_in_dict = self._safe_lookup(n, 'severity')
            self.arg_dict.update(built_in_dict)

    # layer terms, optoinal ===================================
    @_('numbers XS numbers')
    def layers(self, p):
        self.p(f'numbers XS numbers to layers {p[0]} xs {p[2]}')
        self.arg_dict['exp_attachment'] = p[2]
        self.arg_dict['exp_limit'] = p[0]
        return True

    @_('')
    def layers(self, p):
        self.p('missing layer term')
        return True

    # exposures term ==========================================
    @_('numbers CLAIMS')
    def exposures(self, p):
        self.p(f'resolving numbers CLAIMS to exposures {p.numbers}')
        self.arg_dict['exp_en'] = p.numbers
        return True

    @_('numbers LOSS')
    def exposures(self, p):
        self.p(f'resolving numbers LOSS to exposures {p.numbers}')
        self.arg_dict['exp_el'] = p.numbers
        return True

    @_('numbers PREMIUM AT numbers LR')
    def exposures(self, p):
        self.p(f'resolving numbers PREMIUM AT numbers LR to exposures {p[0]} at {p[3]}')
        self.arg_dict['exp_premium'] = p[0]
        self.arg_dict['exp_lr'] = p[3]
        self.arg_dict['exp_el'] = p[0] * p[3]
        return True

    @_('numbers PREMIUM AT numbers')
    def exposures(self, p):
        self.p(f'resolving numbers PREMIUM AT numbers to exposures {p[0]} at {p[3]}')
        self.arg_dict['exp_premium'] = p[0]
        self.arg_dict['exp_lr'] = p[3]
        self.arg_dict['exp_el'] = p[0] * p[3]
        return True

    # lists for ids and numbers and builtinids ================================
    @_('"[" builtinidl "]"')
    def builtinids(self, p):
        self.p(f'resolving [builtinidl] to builtinids {p.builtinidl}')
        return p.builtinidl

    @_('builtinidl BUILTINID')
    def builtinidl(self, p):
        s1 = f'resolving builtinidl ID {p.builtinidl}, {p.ID} --> '
        p.builtinidl.append(p.ID)
        s1 += f'{p.builtinidl}'
        self.p(s1)
        return p.builtinidl

    @_('BUILTINID')
    def builtinidl(self, p):
        self.p(f'resolving ID to builtinidl {p.ID} --> {ans}')
        ans = [p.ID]
        return ans

    @_('BUILTINID')
    def builtinids(self, p):
        self.p(f'resolving ID to builtinids {p.ID}')
        return p.ID

    @_('"[" idl "]"')
    def ids(self, p):
        self.p(f'resolving [id1] to ids {p.idl}')
        return p.idl

    @_('idl ID')
    def idl(self, p):
        s1 = f'resolving idl ID {p.idl}, {p.ID} --> '
        p.idl.append(p.ID)
        s1 += f'{p.idl}'
        self.p(s1)
        return p.idl

    @_('ID')
    def idl(self, p):
        ans = [p.ID]
        self.p(f'resolving ID to idl {p.ID} --> {ans}')
        return ans

    @_('ID')
    def ids(self, p):
        self.p(f'resolving ID to ids {p.ID}')
        return p.ID

    @_('"[" numberl "]"')
    def numbers(self, p):
        self.p(f'resolving [number1] to numbers {p.numberl}')
        return p.numberl

    @_('numberl NUMBER')
    def numberl(self, p):
        s1 = f'resolving numberl NUMBER {p.numberl}, {p.NUMBER} --> '
        p.numberl.append(p.NUMBER)
        s1 += f'{p.numberl}'
        self.p(s1)
        return p.numberl

    @_('NUMBER')
    def numberl(self, p):
        ans = [p.NUMBER]
        self.p(f'resolving NUMBER to numberl {p.NUMBER} --> {ans}')
        return ans

    @_('NUMBER')
    def numbers(self, p):
        self.p(f'resolving NUMBER to numbers {p.NUMBER}')
        return p.NUMBER

    # elements made from named portfolios ========================
    @_('BUILTINID TIMES NUMBER')
    def builtin_aggregate(self, p):
        """  inhomogeneous change of scale """
        self.p(f'Lookup BUILTINID {p.BUILTINID} TIMES NUMBER {p.NUMBER}')
        built_in_dict = self._safe_lookup(p.BUILTINID, 'aggregate')
        self.arg_dict.update(built_in_dict)
        self.arg_dict['exp_en'] *= p.NUMBER
        self.arg_dict['exp_el'] *= p.NUMBER
        self.arg_dict['exp_premium'] *= p.NUMBER

    @_('NUMBER TIMES BUILTINID')
    def builtin_aggregate(self, p):
        """
        homogeneous change of scale

        :param p:
        :return:
        """
        self.p(f'NUMBER {p.NUMBER} TIMES Lookup BUILTINID {p.BUILTINID}')
        built_in_dict = self._safe_lookup(p.BUILTINID, 'aggregate')
        self.arg_dict.update(built_in_dict)
        self.arg_dict['sev_mean'] *= p.NUMBER
        self.arg_dict['sev_scale'] *= p.NUMBER
        self.arg_dict['sev_loc'] *= p.NUMBER
        self.arg_dict['exp_attachment'] *= p.NUMBER
        self.arg_dict['exp_limit'] *= p.NUMBER
        self.arg_dict['exp_el'] *= p.NUMBER
        self.arg_dict['exp_premium'] *= p.NUMBER

    @_('BUILTINID')
    def builtin_aggregate(self, p):
        self.p(f'Lookup BUILTINID {p.BUILTINID}')
        built_in_dict = self._safe_lookup(p.BUILTINID, 'aggregate')
        self.arg_dict.update(built_in_dict)

    @_('ID')
    def name(self, p):
        self.p(f'ID resolves to name {p.ID}')
        self.arg_dict['name'] = p.ID

    def error(self, p):
        if p:
            raise ValueError(p)
        else:
            raise ValueError('Unexpected end of file')


