# For historical interest, the journey to sorting out the parser ran as follows:
#
# 0. Base: 12 SR conflicts
# 1. Reinstated UMINUS: 25SR/15RR! Rejected rule
# 2. Removed percent: 25/14
# 3. MINUS "(" expr ")" %prec UMINUS --> MINUS expr %prec UMINUS: 25/14
# 4. protoexpr introduced as the first level of decoding a NUMBER, UMINUS, percent reinstated: 25/14
# 5. UMINUS at expression level: 25/14
# 6. Use same symbol but different precenence for scale and loc: 33/14?!!
# 7. UMINUS at proto_expression level: 25/14
# 8. Python style math (atom, power, factor, term, sum) (retains same symbol for LOC/SCALE): 29SR/NONE!
# 9. Removed percent made no difference....reinstated but made highest priority
# 10. Removed EXP and () 23 SR XXXX (26 with (),)
# 11. Removed SPECIFIED (23)
# 12. Removed name exposures layers builtin_sev; builtin_sev->sev and then use the sev rule (23)
# 13. builtin_sevs are defined by a dictionary...once looked up they are no different from regular sevs, so all special code removed... 23SR
# 14. EXP and () reinstated, 26 SR
# 15. Intrdouced sev_clause (includes sev and dsev) 29SR [if you try dfreq sev_clause you get 39SR] ...going with 39
# 16. Put LOW prec in reduced to 36...
# 17. Issue was driven by cases with optional arguments. Need to give the optional (reduce) case lower weight.
# 18. parameters to severity...
#
# Issue with scalar x RV + const and pulling out the parameters. If you allow 2 + 3 * lognorm it will never
# work with the same character. Hence need @. Similarly for #.
# zero param sevs are a problem too.
#
# Zero parameter severities did not work. YOu must enter at least one parameter, but it is ignored.
#
# Calculator is more bother than it is worth... keep exp, ** and /, but drop everything else (use f strings!)
# Result has SR conflicts but it parses all the test programs

import logging
import numpy as np
from numpy import exp
from pathlib import Path
import re
from sly import Lexer, Parser
import sly

logger = logging.getLogger(__name__)

DEBUGFILE = Path.home() / 'aggregate/parser/parser.out'


class UnderwritingLexer(Lexer):
    """
    Implements the Lexer for the agg language.

    """

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

    # per manual, need to list longer tokens before shorter ones
    # simple but effective notes
    NOTE = r'note\{[^\}]*\}'  # r'[^\}]+'
    BUILTIN_AGG = r'agg\.[a-zA-Z][a-zA-Z0-9._:~]*'
    BUILTIN_SEV = r'sev\.[a-zA-Z][a-zA-Z0-9._:~]*'
    FREQ = 'binomial|pascal|poisson|bernoulli|geometric|fixed|neyman(a|A)?|logarithmic'
    DISTORTION = 'dist(ortion)?'
    # number regex including unary minus; need before MINUS else that grabs the minus sign in -3 etc.
    NUMBER = r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?'
    # NUMBER = r'(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?'

    # do not allow _ in line names, use ~ or . or : instead: why: because p_ is used and _ is special
    # on honor system...really need two types of ID, it is OK in a portfolio name
    ID = r'[a-zA-Z][\._:~a-zA-Z0-9]*'
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

    @_(r'\n+')
    def newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        logger.error(f"Illegal character '{t.value[0]:s}'")
        self.index += 1

    @staticmethod
    def preprocess(program):
        """
        Separate preprocessor step, allowing it to be called separately.
        Preprocessing involves six steps:

        1. Remove // comments, through end of line
        2. Remove \\n in [ ] (vectors) that appear from  using ``f'{np.linspace(...)}'``
        3. Semicolon ; mapped to newline
        4. Backslash (line continuation) mapped to space
        5. \\n\\t is replaced with space, supporting the tabbed indented Portfolio layout
        6. Split on newlines

        :param program:
        :return:
        """

        # handle \n in vectors; first item is outside, then inside... (multidimensional??)
        out_in = re.split(r'\[|\]', program)
        assert len(out_in) % 2  # must be odd
        odd = [t.replace('\n', ' ') for t in out_in[1::2]]  # replace inside []
        even = out_in[0::2]  # otherwise, pass through
        # reassemble
        program = ' '.join([even[0]] + [f'[{o}] {e}' for o, e in zip(odd, even[1:])])

        # remove comments C++-style // or # comments
        # must replace comments before changing other \ns
        program = re.sub(r'(//|#)[^\n]*$', r'\n', program, flags=re.MULTILINE)

        #  preprocessing: line continuation; \n\t or \n____ to space (for port agg element indents),
        # ; to new line, split on new line
        program = program.replace('\\\n', ' '). replace('\n\t', ' ').replace('\n    ', ' ').replace(';', '\n')

        # split program into lines, only accept len > 0
        program = [i.strip() for i in program.split('\n') if len(i.strip()) > 0]
        return program


class UnderwritingParser(Parser):
    """
    Implements the Parser for the agg language.

    """

    # uncomment to write detailed grammar rules
    # debugfile = Path.home() / 'aggregate/parser/parser.out'
    debugfile = None
    tokens = UnderwritingLexer.tokens
    precedence = (
        # LOW is used to force shift in rules like
        ('nonassoc', LOW),
        ('nonassoc', INHOMOG_MULTIPLY),
        ('left', PLUS, MINUS),
        ('left', TIMES, DIVIDE),
        ('right', EXP),
        ('right', EXPONENT),
        ('nonassoc', PERCENT),
    )

    def __init__(self, safe_lookup_function, debug=False):
        self.debug = debug
        # self.reset()
        # instance of uw class to look up severities
        self.safe_lookup = safe_lookup_function

    def logger(self, msg, p):
        if self.debug is False:
            return
        nm = p._namemap
        sl = p._slice
        ans = []
        for k, v in nm.items():
            rhs = sl[v]
            if type(rhs) == sly.yacc.YaccSymbol:
                # ans.append(f'{k}={rhs.value} (type: {rhs.type})')
                ans.append(f'{k}={rhs.value}')
            else:
                # ans.append(f'{k}={rhs!s}')
                pass
        ans = "; ".join(ans)
        logger.info(f'{msg:20s}\t{ans}')
        # logger.info(f'{msg:15s}\n\t{ans}\n')

    @staticmethod
    def enhance_debugfile(f_out=''):
        """
        Put links in the parser.out debug file, if DEBUGFILE != ''.

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
        Check the value can be vectorized.

        """
        if isinstance(value, (float, int, np.ndarray)):
            return value
        else:
            return np.array(value)

    # final answer exit points =================================
    @_('sev_out')
    def answer(self, p):
        self.logger(
            f'answer <-- sev_out, created severity {p.sev_out[1]}', p)
        return p.sev_out

    @_('agg_out')
    def answer(self, p):
        self.logger(
            f'answer <-- agg_out, created aggregate {p.agg_out[1]}', p)
        return p.agg_out

    @_('port_out')
    def answer(self, p):
        self.logger(f'answer <-- port_out, created portfolio {p.port_out[1]}', p)
        return p.port_out

    @_('distortion_out')
    def answer(self, p):
        self.logger(f'answer <-- distortion_out, created distortion {p.distortion_out[1]} ', p)
        return p.distortion_out

    @_('expr')
    def answer(self, p):
        self.logger(f'expr_out <-- expr {p.expr} ', p)
        return 'expr', f'{p.expr}', p.expr

    # making distortions ======================================
    @_('DISTORTION name ID expr')
    def distortion_out(self, p):
        self.logger('distortion_out <-- DISTORTION ID name', p)
        # self.out_dict[("distortion", p.name)] =
        return 'distortion', p.name, {'name': p.ID, 'shape': p.expr }

    @_('DISTORTION name ID expr "[" numberl "]"')
    def distortion_out(self, p):
        self.logger('distortion_out <-- DISTORTION name ID [ numberl ]', p)
        # for bitvars etc. TODO apply edit to ID to check it is bitvar?
        # self.out_dict[('distortion', p.name)] =
        return 'distortion', p.name, {'name': p.ID, 'shape': p.expr, 'df': p.numberl }

    # building portfolios ======================================
    @_('PORT name note agg_list')
    def port_out(self, p):
        self.logger(
            f'port_out <-- PORT name note agg_list', p)
        # self.out_dict[("port", p.name)] =
        return 'port', p.name, {'spec': p.agg_list, 'note': p.note}

    @_('agg_list agg_out')
    def agg_list(self, p):
        self.logger(f'agg_list <-- agg_list, agg_out', p)
        p.agg_list.append(p.agg_out)
        return p.agg_list

    # building aggregates ======================================
    @_('agg_out')
    def agg_list(self, p):
        self.logger(f'agg_list <-- agg_out', p)
        return [p.agg_out]

    # simplify agg out with sev_clause
    @_('AGG name exposures layers sev_clause occ_reins freq agg_reins note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- AGG name exposures layers SEV sev occ_reins freq agg_reins note', p)
        # self.out_dict[("agg", p.name)] =
        return 'agg', p.name, {'name': p.name, **p.exposures, **p.layers, **p.sev_clause,
                                         **p.occ_reins, **p.freq, **p.agg_reins, 'note': p.note}

    @_('AGG name dfreq layers sev_clause occ_reins agg_reins note')
    def agg_out(self, p):
        self.logger(
            f'agg_out <-- AGG name dfreq layers sev_clause occ_reins agg_reins note', p)
        # self.out_dict[("agg", p.name)] =
        return 'agg', p.name, {'name': p.name, **p.dfreq, **p.layers, **p.sev_clause,
                                         **p.occ_reins, **p.agg_reins, 'note': p.note}

    @_('AGG name TWEEDIE expr expr expr note')
    def agg_out(self, p):
        self.logger('agg_out <-- AGG name TWEEDIE expr expr expr note', p)
        # Tweedie distribution in mean, p, sigma^2 (dispersion) format (MUST be mean first!!)
        # variance function is sigma^2 mean^p
        # phi = sigma^2 in Jorgenson p. 127 notation
        # p = (2 + a)/(a + 1) to a = (2 - p)/(p - 1)
        # lambda = mu^(2-p) / ((2-p) sigma^2)
        # beta = lambda alpha / mu

        # if not here then relative import fails when you run the program to pring the grammar
        from .utilities import tweedie_convert
        mu = p[3]
        pp = p[4]
        sig2 = p[5]
        ans = tweedie_convert(p=pp, μ=mu, σ2=sig2)
        alpha = ans['α']
        lam = ans['λ']
        beta = ans['β']
        # originally
        # alpha = (2 - pp) / (pp - 1)
        # lam = mu ** (2 - pp) / ((2 - pp) * sig2)
        # beta = lam * alpha / mu

        dout = {'name': p.name, 'exp_en': lam, 'freq_name': 'poisson',
                'sev_name': 'gamma', 'sev_a': alpha, 'sev_scale': beta,
                'note': f'Tw(p={pp}, μ={mu}, σ^2={sig2}) --> CP(λ={lam:8g}, ga(α={alpha:.8g}, β={beta:.8g}), '
                        f'scale={beta:.8g}'}
        # self.out_dict[('agg', p.name)] = dout
        return 'agg', p.name, dout

    @_('AGG name builtin_agg agg_reins note')
    def agg_out(self, p):
        # for use when you change the agg and/or  want a new name
        self.logger(
            f'agg_out <-- AGG name builtin_aggregate note', p)
        # rename; NOTE!! the code below will overwrite the new name!
        del p.builtin_agg['name']
        return 'agg', p.name, {'name': p.name, **p.builtin_agg, **p.agg_reins, 'note': p.note}

    @_('builtin_agg agg_reins note')
    def agg_out(self, p):
        # no change to the builtin agg, allows agg.A as a legitimate agg (called A)
        self.logger(
            f'agg_out <-- builtin_agg agg_reins note', p)
        # print(p.builtin_agg)
        # self.out_dict[("agg", p.builtin_agg['name'])] =
        return 'agg', p.builtin_agg['name'],  {**p.builtin_agg, **p.agg_reins, 'note': p.note}

    # building severities ======================================
    # difference from sev_clause (below) is sev_out has a name
    @_('SEV name sev note')
    def sev_out(self, p):
        self.logger(
            f'sev_out <-- sev name sev note ', p)
        p.sev['name'] = p.name
        p.sev['note'] = p.note
        # self.out_dict[("sev", p.name)] = p.sev
        return 'sev', p.name, p.sev

    @_('SEV name dsev note')
    def sev_out(self, p):
        self.logger(
            f'sev_out <-- sev name dsev note ', p)
        p.dsev['name'] = p.name
        p.dsev['note'] = p.note
        # self.out_dict[("sev", p.name)] = p.dsev
        return 'sev', p.name, p.dsev

    # frequency term ===========================================
    # for all frequency distributions claim count is determined by exposure / severity
    # EXCEPT for dfreq (and old EMPIRICAL) where it is entered
    # only freq shape parameters need be entered at the end
    # one and two parameter mixing distributions

    @_('freq ZM expr')
    def freq(self, p):
        self.logger('freq <-- freq ZM expr', p)
        f = p.freq
        f['freq_zm'] = True
        f['freq_p0'] = p.expr
        return f

    @_('freq ZT')
    def freq(self, p):
        self.logger('freq <-- freq ZT', p)
        f = p.freq
        f['freq_zm'] = True
        f['freq_p0'] = 0.0
        return f

    @_('MIXED ID expr expr')
    def freq(self, p):
        self.logger(
            f'freq <-- MIXED ID {p.ID} expr expr', p)
        return {'freq_name': p.ID, 'freq_a': p[2], 'freq_b': p[3]}

    @_('MIXED ID expr')
    def freq(self, p):
        self.logger(
            f'freq <-- MIXED ID {p.ID} expr', p)
        return {'freq_name': p.ID, 'freq_a': p.expr}

    @_('FREQ expr expr')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ {p.FREQ} expr expr', p)
        if p.FREQ != 'pascal':
            logger.warning(
                f'Illogical choice of frequency {p.FREQ}, expected pascal')
        return {'freq_name': p.FREQ, 'freq_a': p[1], 'freq_b': p[2]}

    # binomial p
    @_('FREQ expr')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ expr {p.FREQ}', p)
        if p.FREQ not in ['binomial', 'neyman', 'neymana', 'neymanA']:
            logger.warning(
                f'Illogical choice of frequency {p.FREQ}, expected binomial or neyman A')
        return {'freq_name': p.FREQ, 'freq_a': p.expr}

    @_('FREQ')
    def freq(self, p):
        self.logger(
            f'freq <-- FREQ {p.FREQ} (zero param distributions)', p)
        if p.FREQ not in ('poisson', 'bernoulli', 'fixed', 'geometric', 'logarithmic'):
            logger.error(
                f'Illogical choice for FREQ {p.FREQ}, should be poisson, bernoulli, geometric, logarithmic or fixed.')
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

    @_(" %prec LOW")
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

    @_('tower')
    def reins_list(self, p):
        # would be dumb if it only contained one layer
        self.logger(
            f'reins_clause <-- tower', p)
        limit = p.tower[0]
        attach = p.tower[1]
        return [(1.0, l, a) for l, a in zip(limit, attach)]

    @_('expr XS expr')
    def reins_clause(self, p):
        self.logger(
            f'reins_clause <-- expr XS expr {p[0]} xs {p[2]}', p)
        return (1.0, p[0], p[2])

    @_('expr SHARE_OF expr XS expr')
    def reins_clause(self, p):
        self.logger(
            f'reins_clause <-- expr SHARE_OF expr XS expr {p[0]} s/o {p[2]} xs {p[4]}', p)
        # here expr is the proportion...always store as a proportion
        return (p[0], p[2], p[4])

    @_('expr PART_OF expr XS expr')
    def reins_clause(self, p):
        self.logger(
            f'reins_clause <-- expr PART_OF expr XS expr {p[0]} p/o {p[2]} xs {p[4]}', p)
        # here expr is the currency amount of cover
        if p[0] / p[2] < 0.05:
            logger.warning(
                f'Part of clause with proportion {p[0] / p[2]} is suspiciously small. '
                 'Did you mean share of?')
        return (p[0] / p[2], p[2], p[4])

    # severity term ============================================
    @_('SEV sev %prec LOW')
    def sev_clause(self, p):
        return p.sev

    @_('dsev')
    def sev_clause(self, p):
        return p.dsev

    @_('BUILTIN_SEV')
    def sev_clause(self, p):
        # when the builtin does not need adjusting
        self.logger(f'sev_clause <-- BUILTIN_SEV ({p.BUILTIN_SEV})', p)
        built_in_dict = self.safe_lookup(p.BUILTIN_SEV)
        if 'name' in built_in_dict:
            del built_in_dict['name']
        return built_in_dict

    @_('sev picks')
    def sev(self, p):
        self.logger(f'sev <-- sev picks', p)
        return {**p.sev, **p.picks}

    @_('sev "!"')
    def sev(self, p):
        self.logger(f'sev <-- unconditional (conditional=False) flag set', p)
        p.sev['sev_conditional'] = False
        return p.sev

    @_('sev PLUS numbers', 'sev MINUS numbers')
    def sev(self, p):
        self.logger(f'sev <-- sev {p[1]} numbers', p)
        p.sev['sev_loc'] = UnderwritingParser._check_vectorizable(
            p.sev.get('sev_loc', 0))
        sign = 1 if p[1]=='+' else -1
        p_numbers = UnderwritingParser._check_vectorizable(p.numbers)
        p.sev['sev_loc'] += sign * p_numbers
        return p.sev

    @_('numbers TIMES sev')
    def sev(self, p):
        self.logger(f'sev <-- numbers TIMES sev', p)
        p_numbers = UnderwritingParser._check_vectorizable(p.numbers)
        if 'sev_mean' in p.sev:
            p.sev['sev_mean'] = UnderwritingParser._check_vectorizable(
                p.sev.get('sev_mean', 0))
            p.sev['sev_mean'] *= p_numbers
        # only scale if there is a scale (otherwise you double count)
        if 'sev_scale' in p.sev:
            p.sev['sev_scale'] = UnderwritingParser._check_vectorizable(
                p.sev.get('sev_scale', 0))
            p.sev['sev_scale'] *= p_numbers
        if 'sev_mean' not in p.sev:
            # e.g. Pareto has no mean and it is important to set the scale
            # but if there is a mean it handles the scaling and setting scale will
            # confuse the distribution maker
            p.sev['sev_scale'] = p_numbers
        # if there is a location it needs to scale too --- that's a curious choice!
        if 'sev_loc' in p.sev:
            p.sev['sev_loc'] = UnderwritingParser._check_vectorizable(
                p.sev['sev_loc'])
            p.sev['sev_loc'] *= p_numbers
        # logger.error(str(p.sev))
        return p.sev

    @_('ids numbers CV numbers weights')
    def sev(self, p):
        self.logger(
            f'sev <-- ids numbers CV numbers weights', p)
        return {'sev_name':  p.ids, 'sev_mean':  p[1], 'sev_cv':  p[3], 'sev_scale': 1.0, 'sev_wt': p.weights}

    @_('ids numbers numbers weights')
    def sev(self, p):
        self.logger(
            f'sev <-- ids numbers numbers weights', p)
        # two parameters for shape...must specify scale somehow. put in default scale as 1
        return {'sev_name': p.ids, 'sev_a': p[1], 'sev_b': p[2], 'sev_scale': 1.0, 'sev_wt': p.weights}

    @_('ids numbers weights')
    def sev(self, p):
        self.logger(
            f'sev <-- ids numbers weights', p)
        return {'sev_name': p.ids, 'sev_a':  p[1], 'sev_scale': 1.0,  'sev_wt': p.weights}

    # no weights with xps terms
    @_('ids xps')
    def sev(self, p):
        self.logger(f'sev <-- ids xps (ids should be (c|d)histogram) or zero param (xps is none)', p)
        return {'sev_name': p.ids, **p.xps}

    @_('ids')
    def sev(self, p):
        # for norm expon uniform levy, zero parameter severities
        # need to make sure there is a scale
        self.logger(
            f'sev <-- ids, zero parameter severity {p.ids}', p)
        return {'sev_name': p.ids, 'sev_scale': 1.0}

    @_('XPS doutcomes dprobs')
    def xps(self, p):
        self.logger('xps <-- XPS doutcomes dprobs', p)
        if len(p.dprobs) == 0:
            ps = np.ones_like(p.doutcomes) / len(p.doutcomes)
        else:
            ps = p.dprobs
        return {'sev_xs':  p.doutcomes, 'sev_ps':  ps}

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

    @_('PICKS "[" numberl "]" "[" numberl "]"')
    def picks(self, p):
        self.logger('picks <-- PICKS "[" numberl "]" "[" numberl "]"', p)

        return {'sev_pick_attachments': p[2], 'sev_pick_losses': p[5]}

    # never valid for this to be a single number not in [], using this
    # format rather than numbers enforces an actual list
    @_('"[" numberl "]"')
    def doutcomes(self, p):
        self.logger('doutcomes <-- [numberl] (must be a list)', p)
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

    # see note above doutcomes
    @_('"[" numberl "]"')
    def dprobs(self, p):
        self.logger('dprobs <-- [numberl] (must be a list)', p)
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

    # force weights to be a vector
    @_('WEIGHTS "[" numberl "]"')
    def weights(self, p):
        self.logger(f'weights <-- WEIGHTS [numberl]', p)
        return p.numberl

    @_('')
    def weights(self, p):
        self.logger('weights <-- missing weights term', p)
        return 1

    # layer terms, optional ====================================
    @_('numbers XS numbers')
    def layers(self, p):
        self.logger(
            f'layers <-- numbers XS numbers', p)
        return {'exp_attachment': p[2], 'exp_limit': p[0]}

    @_('tower')
    def layers(self, p):
        self.logger(
            f'layers <-- tower', p)
        return {'exp_attachment': p.tower[1], 'exp_limit': p.tower[0]}

    @_('')
    def layers(self, p):
        self.logger('layers <-- missing layer term', p)
        return {}

    @_('TOWER doutcomes')
    def tower(self, p):
        # doutcomes allows a list, range, or range with step
        self.logger(f'tower <-- tower doutcomes', p)
        breaks = p.doutcomes
        # do not want this. it means net == 0 and ceded== gross in total which
        # is rarely what you want. User can put in themselves.
        # if breaks[0] != 0:
        #     breaks = np.hstack((0., breaks))
        # if not np.isinf(breaks[-1]):
        #     breaks = np.hstack((breaks, np.inf))
        limits = np.diff(breaks)
        attach = breaks[:-1]
        # logger.info('\n'.join([f'{x} xs {y}' for x, y in zip(limits, attach)]))
        return [limits, attach]

    # optional note  ===========================================
    @_('NOTE')
    def note(self, p):
        self.logger(f'note <-- NOTE', p)
        return p.NOTE[5:-1]

    @_(" %prec LOW")
    def note(self, p):
        self.logger("note <-- missing note term", p)
        return ''

    # exposures ================================================
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

    @_('numbers EXPOSURE AT numbers RATE')
    def exposures(self, p):
        self.logger(f'exposures <-- numbers EXPOSURE AT numbers RATE', p)
        return {'exp_premium': p[0], 'exp_lr': p[3], 'exp_el': np.array(p[0]) * np.array(p[3])}

    # ID =======================================================
    @_('"[" idl "]"')
    def ids(self, p):
        self.logger(f'ids <-- [idl]', p)
        return p.idl

    @_('idl ID')
    def idl(self, p):
        self.logger(f'idl <-- idl ID ({p.ID})', p)
        p.idl.append(p.ID)
        return p.idl

    @_('ID')
    def idl(self, p):
        self.logger(f'idl <-- ID ({p.ID})', p)
        ans = [p.ID]
        self.logger(f'idl <-- ID', p)
        return ans

    @_('ID')
    def ids(self, p):
        self.logger(f'ids <-- ID ({p.ID})', p)
        return p.ID

    # elements made from named portfolios ========================
    @_('expr INHOMOG_MULTIPLY builtin_agg')
    def builtin_agg(self, p):
        """  inhomogeneous change of scale """
        self.logger(
            f'builtin_agg <-- expr INHOMOG_MULTIPLY builtin_agg', p)
        bid = p.builtin_agg.copy()
        bid['name'] += '_i_scaled'

        bid['exp_en'] = self._check_vectorizable(bid.get('exp_en', 0)) * p.expr
        bid['exp_el'] = self._check_vectorizable(bid.get('exp_el', 0)) * p.expr
        bid['exp_premium'] = self._check_vectorizable(bid.get('exp_premium', 0)) * p.expr
        return bid

    @_('expr TIMES builtin_agg')
    def builtin_agg(self, p):
        """homogeneous change of scale """
        self.logger('builtin_agg <-- expr TIMES builtin_agg', p)
        # bid = built_in_dict, want to be careful not to add scale too much
        bid = p.builtin_agg
        bid['name'] += '_homog_scaled'
        if 'sev_mean' in bid:
            bid['sev_mean'] = self._check_vectorizable(bid['sev_mean']) * p.expr
        if 'sev_scale' in bid:
            bid['sev_scale'] = self._check_vectorizable(bid['sev_scale']) * p.expr
        if 'sev_loc' in bid:
            bid['sev_loc'] = self._check_vectorizable(bid['sev_loc']) * p.expr
        bid['exp_attachment'] = self._check_vectorizable(bid.get('exp_attachment', 0)) * p.expr
        bid['exp_limit'] = self._check_vectorizable(bid.get('exp_limit', np.inf)) * p.expr
        bid['exp_el'] = self._check_vectorizable(bid.get('exp_el', 0)) * p.expr
        bid['exp_premium'] = self._check_vectorizable(bid.get('exp_premium', 0)) * p.expr
        return bid

    @_('builtin_agg PLUS expr', 'builtin_agg MINUS expr')
    def builtin_agg(self, p):
        """
        translation (shift, change location) by expr
        :param p:
        :return:
        """
        self.logger('builtin_agg <-- builtin_agg PLUS expr', p)
        # bid = built_in_dict, want to be careful not to add scale too much
        bid = p.builtin_agg
        bid['name'] += '_shifted'
        sign = 1 if p[1]=="+" else -1
        # TODO make vector addable
        if 'sev_loc' in bid:
            bid['sev_loc'] += sign * p.expr
        else:
            bid['sev_loc'] = sign * p.expr
        return bid

    @_('BUILTIN_AGG')
    def builtin_agg(self, p):
        # ensure lookup only happens here
        self.logger(f'builtin_agg <-- BUILTIN_AGG ({p.BUILTIN_AGG})', p)
        built_in_dict = self.safe_lookup(p.BUILTIN_AGG)
        return built_in_dict

    @_('BUILTIN_SEV')
    def sev(self, p):
        # ensure lookup only happens here
        # unlike aggs, will never just say sev.A
        # usage: agg A 1 claim sev sev.B fixed; a little awkward but not used much
        # leaving it here allos for subsequent scaling and translation
        # if it is directly a sev_clause it cannot be adjusted
        self.logger(f'sev <-- BUILTIN_SEV ({p.BUILTIN_SEV})', p)
        built_in_dict = self.safe_lookup(p.BUILTIN_SEV)
        if 'name' in built_in_dict:
            del built_in_dict['name']
        return built_in_dict

    # ids =========================================================
    @_('ID')
    def name(self, p):
        self.logger(f'name <-- ID = {p.ID}', p)
        return p.ID

    # vectors of numbers ==========================================
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

    @_('term')
    def expr(self, p):
        self.logger('expr <-- term', p)
        return p.term

    @_('term DIVIDE factor')
    def term(self, p):
        self.logger('term <-- term / factor', p)
        return p.term / p.factor

    @_('factor')
    def term(self, p):
        self.logger('term <-- factor', p)
        return p.factor

    @_('power')
    def factor(self, p):
        self.logger('factor <-- power', p)
        return p.power

    @_('atom EXPONENT factor')
    def power(self, p):
        self.logger('power <-- atom EXPONENT factor', p)
        return p.atom ** p.factor

    @_('atom')
    def power(self, p):
        self.logger('power <-- atom', p)
        return p.atom

    @_('NUMBER PERCENT')
    def atom(self, p):
        self.logger('atom <-- atom PERCENT', p)
        return float(p.NUMBER) / 100

    @_('INFINITY')
    def atom(self, p):
        self.logger(f'atom <-- INFINITY', p)
        return np.inf

    @_('NUMBER')
    def atom(self, p):
        self.logger(f'atom <-- NUMBER, {p.NUMBER}', p)
        t = float(p.NUMBER)
        return t

    @_('"(" term ")"')
    def factor(self, p):
        return p.term
    #
    @_('EXP "(" term ")"')
    def factor(self, p):
        return exp(p.term)

    def error(self, p):
        if p:
            raise ValueError(p)
        else:
            raise ValueError('Unexpected end of file')


def grammar(add_to_doc=False, save_to_fn=''):
    """
    Write the grammar at the top of the file as a docstring

    To work with multi-rules enter them on one line, like so::

        @_('builtin_agg PLUS expr', 'builtin_agg MINUS expr')

    :param add_to_doc: add the grammar to the docstring
    :param save_to_fn: save the grammar to a file
    """

    pout = Path(__file__).parent / '../docs/4_agg_language_reference/ref_include.rst'

    # get the grammar from the top of the file
    txt = Path(__file__).read_text(encoding='utf-8')
    stxt = txt.split('@_')
    ans = {}
    # 3:-3 get rid of junk at top and bottom (could change if file changes)
    for it in stxt[3:-3]:
        if it.find('# def') >= 0:
            # skip rows with a comment between @_ and def
            pass
        else:
            b = it.split('def')
            b0 = b[0].strip()[2:-2]
            # check if multirule
            if ', ' in b0:
                b0 = [i.replace("'", '') for i in b0.split(', ')]
            else:
                b0 = [b0]
            try:
                b1 = b[1].split("(self, p):")[0].strip()
            except:
                logger.warning(f'Unexpected multirule behavior {it}')
                exit()
            if b1 in ans:
                ans[b1] += b0
            else:
                ans[b1] = b0
    s = ''
    for k, v in ans.items():
        s += f'{k:<20s}\t::= {v[0]:<s}\n'
        for rhs in v[1:]:
            s += f'{" "*20}\t | {rhs:<s}\n'
        s += '\n'

    # finally add the language words
    # this is a bit manual, but these shouldnt change much...
    # lang_words = '\n\nlanguage words go here\n\n'
    lang_words = '''FREQ                    ::= 'binomial|poisson|bernoulli|pascal|geometric|neymana?|fixed'

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

INHOMOG_MULTIPLY        ::= "@"

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

XS                      ::= "xs|x"

'''

    s += lang_words
    # create for docs in one file (that gets included by rst)
    if add_to_doc is True:
        pout.write_text(s, encoding='utf-8')

    # save to user folder grammar
    if save_to_fn == '':
        save_to_fn = Path.home() / 'aggregate/parser/grammar.md'
    Path(save_to_fn).write_text(s, encoding='utf-8')

    return s

if __name__ == '__main__':
    # print the grammar and add to this file as part of docstring in 41_language_reference.rst

    grammar(add_to_doc=True)
    UnderwritingParser.enhance_debugfile()
