"""
Underwriter Module
==================

Does lots of cool things


"""

import os
from ruamel import yaml
import numpy as np
import collections
from io import StringIO
from textwrap import fill, indent
from IPython.core.display import display
import logging
import pandas as pd
from .distr import Aggregate, Severity
from .port import Portfolio
from .utils import html_title
from sly import Lexer, Parser


class _DataManager(object):
    """
    _DataManager class
    --------------

    Private class handling reading and writing to YAML files for Underwriter class, which
    subclasses _YAML_reader


    """

    def __init__(self):
        """
        info

        """

        if self.dir_name == '':
            self.dir_name = os.path.split(__file__)[0]
            self.dir_name = os.path.join(self.dir_name, 'yaml')
        for k, v in self.databases.items():
            d = dict()
            for fn in v:
                with open(os.path.join(self.dir_name, fn), 'r') as f:
                    d = dict(**d, **yaml.load(f, Loader=yaml.Loader))
            self.__setattr__(k, d)

    def list(self):
        """
        list all available databases

        :return:
        """
        sers = dict()
        for k in self.databases.keys():
            d = sorted(list(self.__getattribute__(k).keys()))
            sers[k.title()] = pd.Series(d, index=range(len(d)), name=k)
        df = pd.DataFrame(data=sers)
        # df.index.name = 'No.'
        df = df.fillna('')
        return df

    def describe(self, item_type='', pretty_print=False):
        """
        more informative version of list
        Pull notes from YAML descriptions for type items

        :return:
        """
        df = pd.DataFrame(columns=['Name', 'Type', 'Severity', 'ESev', 'Sev_a', 'Sev_b',
                                   'EN', 'Freq_a',
                                   'ELoss', 'Notes'])
        df = df.set_index('Name')
        df['ELoss'] = np.maximum(df.ELoss, df.ESev * df.EN)
        if item_type == '':
            item_type = self.databases.keys()
        else:
            item_type = [item_type.lower()]
        for k in item_type:  # self.databases.keys():
            for kk, vv in self.__getattribute__(k).items():
                df.loc[kk, :] = (k, vv.get('sev_name', ''),
                                 vv.get('sev_mean', 0),
                                 vv.get('sev_a', 0),
                                 vv.get('sev_b', 0),
                                 vv.get('exp_en', 0),
                                 vv.get('freq_a', 0),
                                 vv.get('exp_el', 0),
                                 vv.get('note', ''))
        df = df.fillna('')
        if pretty_print:
            for t, egs in df.groupby('Type'):
                html_title(t, 2)
                display(egs.style)
        return df

    def __getitem__(self, item):
        """
        scriptable: try user portfolios, b/in portfolios, line, severity
        to access specifically use severity or line methods

        ORDERING PROBLEM!

        :param item:
        :return:
        """

        for k in self.databases.keys():
            if item in self.__getattribute__(k).keys():
                # stip the s off the name: Books to Book etc.
                return k, self.__getattribute__(k)[item]
        raise LookupError


class Underwriter(_DataManager):
    """
    Underwriter class
    -----------------

    The underwriter class constructs real world examples from stored and user input Lines and Accounts.
    Whereas Examples only produces simple Portfolios and Books, the Underwriter class is more flexible.

    Persisitence to and from YAML managed

    """

    def __init__(self, dir_name=""):
        """


        :param dir_name:
        """
        self.last_spec = None
        self.databases = dict(severity=['severities.yaml'],
                              aggregate=['aggregates.yaml', 'user_aggregates.yaml'],
                              portfolio=['portfolios.yaml', 'user_portfolios.yaml'])
        self.dir_name = dir_name
        _DataManager.__init__(self)

    def __getitem__(self, item):
        """
        handles self[item]
        the result is cast into the right type of object

        :param item:
        :return: Book, Account or Line object
        """
        return _DataManager.__getitem__(self, item)

    def __getattr__(self, item):
        """
        handles self.item

        :param item:
        :return:
        """
        return _DataManager.__getitem__(self, item)

    def get_dict(self, item):
        """
        get an item as dictionary, WITHOUT the type

        :param item:
        :return:
        """
        _type, obj = _DataManager.__getitem__(self, item)
        return obj

    def get_object(self, item):
        """
        get an item as an object of the right type

        :param item:
        :return:
        """
        # _type, obj = _DataManager.__getitem__(self, item)
        # if _type == ''
        # return obj
        pass

    def __call__(self, portfolio_program):
        return self.write(portfolio_program)

    def write(self, portfolio_program, portfolio_name='', update=False, verbose=False, log2=0, bs=0, **kwargs):
        """
        write a pseudo natural language programming spec for a book or (if only one line) an aggregate_project

        e.g. Input
        20  loss 3 x 2 gamma 5 cv 0.30 mixed gamma 0.4
        10  claims 3 x 2 gamma 12 cv 0.30 mixed gamma 1.2
        100  premium at 0.4 3 x 2 4 * lognormal 3 cv 0.8 fixed 1

        See parser for full language spec!

        :param portfolio_program:
        :param portfolio_name:
        :param update:
        :param verbose:
        :param log2:
        :param bs:
        :param kwargs:
        :return:
        """
        logging.info(f'Underwriter.write | Executing program\n{portfolio_program[:500]}\n\n')
        lexer = UnderwritingLexer()
        parser = UnderwritingParser(self)
        portfolio_program = [i.strip() for i in portfolio_program.replace(';', '\n').split('\n') if len(i.strip()) > 0]
        spec_list = []

        for txt in portfolio_program:
            parser.reset()
            try:
                parser.parse(lexer.tokenize(txt))
            except ValueError as e:
                if isinstance(e.args[0], str):
                    print(e)
                    raise e
                else:
                    t = e.args[0].type
                    v = e.args[0].value
                    # l = e.arg_dict[0].lineno
                    i = e.args[0].index
                    txt2 = txt[0:i] + f'>>>' + txt[i:]
                    print(f'Parse error in input "{txt2}"\nValue {v} of type {t} not expected')
                    raise e
            spec_list.append(parser.arg_dict)
        if portfolio_name == '':
            portfolio_name = f'built ins {len(spec_list)}'
        logging.info(f'Underwriter.write | creating Portfolio {portfolio_name} from {portfolio_program}')
        # spec_list is a list of dictionaries that can be passed straight through to creaet a portfolio
        port = Portfolio(portfolio_name, spec_list)
        if update:
            if bs == 0:
                # for log2 = 10
                bs = port.recommend_bucket().iloc[-1, 0]
                if log2 == 0:
                    log2 = 10
                else:
                    bs = bs * 2 ** (10 - log2)
            logging.info(f'Underwriter.write | updating Portfolio {portfolio_name}, log2={10}, bs={bs}')
            port.update(log2=log2, bs=bs, verbose=verbose, **kwargs)
        self.last_spec = spec_list
        return port


def dict_2_string(type_name, dict_in, tab_level=0, sio=None):
    """
    nice formating for str function

    :param type_name:
    :param dict_in:
    :param tab_level:
    :param sio:
    :return:
    """

    if sio is None:
        sio = StringIO()

    keys = sorted(dict_in.keys())
    if 'name' in keys:
        # which it should always be
        nm = dict_in['name']
        sio.write(nm + '\n' + '=' * len(nm) + '\n')
        keys.pop(keys.index('name'))

    sio.write(f'{"type":<20s}{type_name}\n')

    for k in keys:
        v = dict_in[k]
        ks = '\t' * max(0, tab_level - 1) + f'{str(k):<20s}'
        if type(v) == dict:
            # sio.write('\t' * tab_level + ks + '\n')
            dict_2_string(type(v), v, tab_level + 1, sio)
        elif isinstance(v, str):
            if len(v) > 30:
                sio.write('\t' * tab_level + ks + '\n' + indent(fill(v, 30), ' ' * (4 * tab_level + 20)))
            elif len(v) > 0:
                sio.write('\t' * tab_level + ks + v)
            sio.write('\n')
        elif isinstance(v, collections.Iterable):
            sio.write('\t' * tab_level + ks + '\n')
            for vv in v:
                sio.write('\t' * (tab_level + 1) + str(vv) + '\n')
        elif type(v) == int:
            sio.write('\t' * tab_level + f'{ks}\t{v:20d}\n')
        elif type(v) == float:
            if abs(v) < 100:
                sio.write('\t' * tab_level + f'{ks}\t{v:20.5f}\n')
            else:
                sio.write('\t' * tab_level + f'{ks}\t{v:20,.1f}\n')
        else:
            # logging.info(f'Uknown type {type(v)} to dict_2_string')
            sio.write('\t' * tab_level + ks + '\t' + str(v) + '\n')
    return sio.getvalue()


class UnderwritingLexer(Lexer):
    tokens = {ID, PLUS, MINUS, TIMES, NUMBER, CV, LOSS, PREMIUM, AT, LR, CLAIMS, XS, MIXED,
              FIXED, POISSON, BUILTINID}
    ignore = ' \t,\\:\\(\\)'

    BUILTINID = r'uw\.[a-zA-Z_][a-zA-Z0-9_]*'
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    # PERCENT = r'%'
    PLUS = r'\+'
    MINUS = r'\-'
    TIMES = r'\*'
    ID['loss'] = LOSS
    ID['LOSS'] = LOSS
    ID['at'] = AT
    ID['AT'] = AT
    ID['@'] = AT
    ID['cv'] = CV
    ID['CV'] = CV
    ID['premium'] = PREMIUM
    ID['prem'] = PREMIUM
    ID['PREMIUM'] = PREMIUM
    ID['PREM'] = PREMIUM
    ID['WP'] = PREMIUM
    ID['EP'] = PREMIUM
    ID['lr'] = LR
    ID['LR'] = LR
    ID['claims'] = CLAIMS
    ID['claim'] = CLAIMS
    ID['CLAIMS'] = CLAIMS
    ID['CLAIM'] = CLAIMS
    ID['xs'] = XS
    ID['XS'] = XS
    ID['x'] = XS
    ID['X'] = XS
    ID['mixed'] = MIXED
    ID['poisson'] = POISSON
    ID['fixed'] = FIXED
    ID['inf'] = NUMBER
    ID['INF'] = NUMBER

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
    tokens = UnderwritingLexer.tokens
    precedence = (('left', PLUS, MINUS), ('left', TIMES))

    def __init__(self, uw):
        self.arg_dict = None
        self.reset()
        # instance of uw class to look up severities
        self.uw = uw

    def reset(self):
        # TODO pull from Aggregate automatically ...
        # in order to allow for missing terms this must reflect sensible defaults
        self.arg_dict = dict(name="", exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                             sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_scale=0, sev_loc=0,
                             freq_name='poisson', freq_a=0, freq_b=0)

    # built from a built in book
    @_('builtin_aggregate')
    def ans(self, p):
        logging.info('UnderwritingParser | Exiting through built in aggregate')
        pass

    # make freq and limit optional
    @_('name expos sev')
    def ans(self, p):
        logging.info('UnderwritingParser | Exiting through name expos sev')
        pass

    # make freq optional
    @_('name expos limit sev')
    def ans(self, p):
        logging.info('UnderwritingParser | Exiting through name expos limit sev')
        pass

    # make limit optional
    @_('name expos sev freq')
    def ans(self, p):
        logging.info('UnderwritingParser | Exiting through name expos sev freq')
        pass

    # base: expos + limit + sev + freq
    @_('name expos limit sev freq')
    def ans(self, p):
        logging.info('UnderwritingParser | Exiting through name expos limit sev freq')
        pass

    @_('MIXED ID NUMBER NUMBER')
    def freq(self, p):
        self.arg_dict['freq_name'] = 'poisson'  # p.ID  TODO once freq dists implemented this needs to change
        self.arg_dict['freq_a'] = p[2]
        self.arg_dict['freq_b'] = p[3]

    @_('MIXED ID NUMBER')
    def freq(self, p):
        self.arg_dict['freq_name'] = 'poisson'  # p.ID  TODO, as above
        self.arg_dict['freq_a'] = p.NUMBER

    @_('FIXED')
    def freq(self, p):
        self.arg_dict['freq_name'] = 'fixed'

    @_('POISSON')
    def freq(self, p):
        self.arg_dict['freq_name'] = 'poisson'

    @_('sev PLUS NUMBER')
    def sev(self, p):
        self.arg_dict['sev_loc'] += p.NUMBER

    @_('sev MINUS NUMBER')
    def sev(self, p):
        self.arg_dict['sev_loc'] -= p.NUMBER

    @_('NUMBER TIMES sev')
    def sev(self, p):
        self.arg_dict['sev_mean'] *= p.NUMBER
        # set the scale, don't "scale" the scale
        self.arg_dict['sev_scale'] = p.NUMBER

    @_('ID NUMBER CV NUMBER')
    def sev(self, p):
        self.arg_dict['sev_name'] = p.ID
        self.arg_dict['sev_mean'] = p[1]
        self.arg_dict['sev_cv'] = p[3]
        return True

    @_('ID NUMBER')
    def sev(self, p):
        self.arg_dict['sev_name'] = p.ID
        self.arg_dict['sev_a'] = p[1]
        return True

    @_('ID NUMBER NUMBER')
    def sev(self, p):
        self.arg_dict['sev_name'] = p.ID
        self.arg_dict['sev_a'] = p[1]
        self.arg_dict['sev_b'] = p[2]
        return True

    @_('BUILTINID')
    def sev(self, p):
        # look up ID in uw
        self._safe_lookup(p.BUILTINID[3:], 'severity')

    @_('NUMBER XS NUMBER')
    def limit(self, p):
        self.arg_dict['exp_attachment'] = p[2]
        self.arg_dict['exp_limit'] = p[0]
        return True

    @_('NUMBER CLAIMS')
    def expos(self, p):
        self.arg_dict['exp_en'] = p.NUMBER
        return True

    @_('NUMBER LOSS')
    def expos(self, p):
        self.arg_dict['exp_el'] = p.NUMBER
        return True

    @_('NUMBER PREMIUM AT NUMBER LR')
    def expos(self, p):
        self.arg_dict['exp_premium'] = p[0]
        self.arg_dict['exp_lr'] = p[3]
        self.arg_dict['exp_el'] = p[0] * p[3]
        return True

    @_('NUMBER PREMIUM AT NUMBER')
    def expos(self, p):
        self.arg_dict['exp_premium'] = p[0]
        self.arg_dict['exp_lr'] = p[3]
        self.arg_dict['exp_el'] = p[0] * p[3]
        return True

    @_('name BUILTINID TIMES NUMBER')
    def builtin_aggregate(self, p):
        """
        inhomogeneous change of scale

        :param p:
        :return:
        """
        self._safe_lookup(p.BUILTINID[3:], 'aggregate')
        self.arg_dict['exp_en'] *= p.NUMBER
        self.arg_dict['exp_el'] *= p.NUMBER
        self.arg_dict['exp_premium'] *= p.NUMBER

    @_('name NUMBER TIMES BUILTINID')
    def builtin_aggregate(self, p):
        """
        homogeneous change of scale

        :param p:
        :return:
        """
        self._safe_lookup(p.BUILTINID[3:], 'aggregate')
        self.arg_dict['sev_mean'] *= p.NUMBER
        self.arg_dict['sev_scale'] *= p.NUMBER
        self.arg_dict['sev_loc'] *= p.NUMBER
        self.arg_dict['exp_attachment'] *= p.NUMBER
        self.arg_dict['exp_limit'] *= p.NUMBER
        self.arg_dict['exp_el'] *= p.NUMBER
        self.arg_dict['exp_premium'] *= p.NUMBER

    @_('name BUILTINID')
    def builtin_aggregate(self, p):
        # look up ID in uw
        self._safe_lookup(p.BUILTINID[3:], 'aggregate')

    @_('ID')
    def name(self, p):
        self.arg_dict['name'] = p[0]

    def error(self, p):
        if p:
            raise ValueError(p)
        else:
            raise ValueError('Unexpected end of file')

    def _safe_lookup(self, uw_id, expected_type):
        """
        lookup uw_id in uw of expected type and merge safely into self.arg_dict
        delete anme and note if appropriate

        :param uw_id:
        :param expected_type:
        :return:
        """
        _type, builtin_dict = self.uw[uw_id]
        assert _type != 'portfolio'  # this is a problem
        logging.info(f'UnderwritingParser._safe_lookup | retrieved {uw_id} type {type(builtin_dict)}')
        if _type != expected_type:
            print(f'WARNING: type of {uw_id} is  {type(builtin_dict)}, not expected {expected_type}')
        assert _type == expected_type
        # may need to delete various items
        if 'note' in builtin_dict:
            del builtin_dict['note']
        self.arg_dict.update(builtin_dict)
