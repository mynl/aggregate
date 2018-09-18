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
from .port import Portfolio
from .utils import html_title
from .distr import Aggregate, Severity
from .parser import UnderwritingLexer, UnderwritingParser


class Underwriter(object):
    """
    Underwriter class
    -----------------

    The underwriter class constructs real world examples from stored and user input Lines and Accounts.
    Whereas Examples only produces simple Portfolios and Books, the Underwriter class is more flexible.

    Handles persistence
    Is interface into program parser
    Handles safe lookup from database for parser

    Persisitence to and from YAML managed

    """

    def __init__(self, dir_name="", databases=None, store_mode=True, debug=False):
        """

        :param dir_name:
        :param store_mode: add newly created aggregates to the database?
        :param debug: run parser in debug mode?
        """

        self.last_spec = None
        self.store_mode = store_mode
        self.debug = debug
        self.lexer = UnderwritingLexer()
        self.parser = UnderwritingParser(self._safe_lookup, debug)
        # otherwise these are hidden from pyCharm....
        self.severity = None
        self.aggregate = None
        self.portfolio = None
        if databases is None:
            databases = dict(severity=['severities.yaml'], aggregate=['aggregates.yaml', 'user_aggregates.yaml'],
                             portfolio=['portfolios.yaml', 'user_portfolios.yaml'])
        self.databases = databases
        self.dir_name = dir_name
        if self.dir_name == '':
            self.dir_name = os.path.split(__file__)[0]
            self.dir_name = os.path.join(self.dir_name, 'yaml')
        for k, v in self.databases.items():
            d = dict()
            for fn in v:
                with open(os.path.join(self.dir_name, fn), 'r') as f:
                    temp = yaml.load(f, Loader=yaml.Loader)
                    for kt, vt in temp.items():
                        d[kt] = vt
            # port, agg and sev actually set here...
            self.__setattr__(k, d)

    def __getitem__(self, item):
        """
        handles self[item]
        subscriptable: try user portfolios, b/in portfolios, line, severity
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

    # this is evil: it passes unknown things through to write...
    # def __getattr__(self, item):
    #     """
    #     handles self.item and returns an appropriate object
    #
    #     :param item:
    #     :return:
    #     """
    #     # print(f'Underwriter.__getattr__({item}) called')
    #     if item[0] == '_':
    #         # deal with the _ipython_canary_method_should_not_exist_
    #         # print('bailing')
    #         return
    #     else:
    #         return self.write(item)

    def __call__(self, portfolio_program):
        # print(f'Call on underwriter with {portfolio_program[:100]}')
        return self.write(portfolio_program)

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

    def write(self, portfolio_program, update=False, verbose=False, log2=0, bs=0, **kwargs):
        """
        write a pseudo natural language programming spec for a book or (if only one line) an aggregate_project

        e.g. Input
        port my_portfolio
            20  loss 3 x 2 sev gamma 5 cv 0.30 mixed gamma 0.4
            10  claims 3 x 2 sevgamma 12 cv 0.30 mixed gamma 1.2
            100  premium at 0.4 3 x 2 sev 4 * lognormal 3 cv 0.8 fixed 1

        The indents are required...

        See parser for full language spec!

        :param portfolio_program:
        :param update:
        :param verbose:
        :param log2:
        :param bs:
        :param kwargs:
        :return:
        """

        # first see if it is a built in object
        lookup_success = True
        _type = ''
        obj = None
        try:
            _type, obj = self.__getitem__(portfolio_program)
        except LookupError:
            lookup_success = False
            logging.warning(f'underwriter.write | object {portfolio_program[:500]} not found, will process as program')
        if lookup_success:
            logging.info(f'underwriter.write | object {portfolio_program[:500]} found, returning object...')
            if _type == 'aggregate':
                return Aggregate(portfolio_program, **obj)
            elif _type == 'portfolio':
                return Portfolio(portfolio_program, obj['spec'])
            elif _type == 'severity':
                return Severity(**obj)
            else:
                ValueError(f'Cannot build {_type} objects')
            return obj

        # run
        self._runner(portfolio_program)

        # what shall we create? only create if there is one item  port then agg then sev, create in rv
        rv = None
        if len(self.parser.port_out_dict) > 0:
            # create ports
            rv = []
            for k in self.parser.port_out_dict.keys():
                s = Portfolio(k, self.portfolio[k]['spec'])
                if update:
                    if bs == 0:
                        # for log2 = 10
                        bs = s.recommend_bucket().iloc[-1, 0]
                        if log2 == 0:
                            log2 = 10
                        else:
                            bs = bs * 2 ** (10 - log2)
                    logging.info(f'Underwriter.write | updating Portfolio {k}, log2={10}, bs={bs}')
                    s.update(log2=log2, bs=bs, verbose=verbose, **kwargs)
                rv.append(s)

        elif len(self.parser.agg_out_dict) > 0 and rv is None:
            # new aggs, create them
            rv = []
            for k, v in self.parser.sev_out_dict.items():
                s = Aggregate(k, **v)
                rv.append(s)

        elif len(self.parser.sev_out_dict) > 0 and rv is None:
            # sev all sevs
            rv = []
            for v in self.parser.sev_out_dict.values():
                if 'sev_wt' in v:
                    del v['sev_wt']
                s = Severity(**v)
                rv.append(s)

        else:
            print('WARNING: Program did not contain any output...')
            logging.warning(f'Underwriter.write | Program {portfolio_program} did not contain any output...')
        if len(rv) == 1:
            rv = rv[0]
        return rv

    def write_from_file(self, file_name, update=False, verbose=False, log2=0, bs=0, **kwargs):
        """
        read program from file. delegates to write

        :param file_name:
        :param update:
        :param verbose:
        :param log2:
        :param bs:
        :param kwargs:
        :return:
        """
        with open(file_name, 'r', encoding='utf-8') as f:
            portfolio_program = f.read()
        return self.write(portfolio_program, update, verbose, log2, bs, **kwargs)

    def write_test(self, portfolio_program):
        """
        replaced     def test_write(self, portfolio_program):
        fka test_run

        :param portfolio_program:
        :return:
        """
        logging.info(f'Runner.write_test | Executing program\n{portfolio_program[:500]}\n\n')
        self._runner(portfolio_program)
        # for a in ['sev_out_dict', 'agg_out_dict', 'port_out_dict']:
        #     ans = getattr(self.parser, a)
        #     if len(ans) > 0:
        #         _s = f'{len(ans)} {a[0:4]} objects created'
        #         print('\n'+_s)
        #         print('='*len(_s))
        #         for k, v in ans.items():
        #             print(f'{k:<10s}\t{v}')
        ans1 = ans2 = ans3 = None
        if len(self.parser.sev_out_dict) > 0:
            for v in self.parser.sev_out_dict.values():
                Underwriter._add_defaults(v, 'sev')
            ans1 = pd.DataFrame(list(self.parser.sev_out_dict.values()), index=self.parser.sev_out_dict.keys())
        if len(self.parser.agg_out_dict) > 0:
            for v in self.parser.agg_out_dict.values():
                Underwriter._add_defaults(v)
            ans2 = pd.DataFrame(list(self.parser.agg_out_dict.values()), index=self.parser.agg_out_dict.keys())
        if len(self.parser.port_out_dict) > 0:
            ans3 = pd.DataFrame(list(self.parser.port_out_dict.values()), index=self.parser.port_out_dict.keys())
        return ans1, ans2, ans3

    def _runner(self, portfolio_program):
        """
        preprocessing:
            ; mapped to newline
            \ (line continuation) mapped to space
            split on newlines
            parse one line at a time
            PIPE format no longer supported
        error handling and piping through parser

        :param portfolio_program:
        :return:
        """
        # preprocess line continuation and replace ; with new line
        portfolio_program = [i.strip() for i in portfolio_program.replace('\\\n', ' ').
                             replace('\n\t', ' ').replace('\n    ', ' ').replace(';', '\n').
                             split('\n') if len(i.strip()) > 0]
        # portfolio_program = portfolio_program.replace('\\\n', ' ') # .replace(';', '\n')
        self.parser.reset()
        for program_line in portfolio_program:
            # print(program_line)
            try:
                if len(program_line) > 0:
                    self.parser.parse(self.lexer.tokenize(program_line))
            except ValueError as e:
                if isinstance(e.args[0], str):
                    print(e)
                    raise e
                else:
                    t = e.args[0].type; v = e.args[0].value; i = e.args[0].index
                    txt2 = program_line[0:i] + f'>>>' + program_line[i:]
                    print(f'Parse error in input "{txt2}"\nValue {v} of type {t} not expected')
                    raise e
        if self.store_mode:
            # could do this with a loop and getattr but it is too hard to read, so go easy route
            if len(self.parser.sev_out_dict) > 0:
                # for k, v in self.parser.sev_out_dict.items():
                self.severity.update(self.parser.sev_out_dict) # [k] = v
                logging.info(f'Underwriter._runner | saving {self.parser.sev_out_dict.keys()} severity/ies')
            if len(self.parser.agg_out_dict) > 0:
                # for k, v in self.parser.agg_out_dict.items():
                #     self.aggregate[k] = v
                self.aggregate.update(self.parser.agg_out_dict)
                logging.info(f'Underwriter._runner | saving {self.parser.agg_out_dict.keys()} aggregate(s)')
            if len(self.parser.port_out_dict) > 0:
                for k, v in self.parser.port_out_dict.items():
                    # v is a list of aggregate names, these have all been added to the database...
                    logging.info(f'Underwriter._runner | saving {k} portfolio')
                    self.portfolio[k] = {'spec': [self.aggregate[_a] for _a in v], 'arg_dict': {}}
        # can we still do something like this?
        #     self.parser.arg_dict['note'] = txt
        return

    @staticmethod
    def _add_defaults(dict_, kind='agg'):
        """
        add default values to dict_ Leave existing values unchanged

        :param dict_:
        :return:
        """
        defaults = dict(name="", exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                    sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_scale=0, sev_loc=0, sev_wt=1,
                    freq_name='poisson', freq_a=0, freq_b=0)
        if kind == 'agg':
            for k, v in defaults.items():
                if k not in dict_:
                    dict_[k] = v
        elif kind == 'sev':
            for k, v in defaults.items():
                if k[0:3] == 'sev' and k not in dict_ and k != 'sev_wt':
                    dict_[k] = v

    def _safe_lookup(self, uw_id, expected_type):
        """
        lookup uw_id in uw of expected type and merge safely into self.arg_dict
        delete name and note if appropriate

        :param uw_id:
        :param expected_type:
        :return:
        """
        _type = 'not found'
        builtin_dict = None
        try:
            # strip the uw. off here
            _type, builtin_dict = self[uw_id[3:]]
        except LookupError as e:
            print(f'ERROR Looked up {uw_id} found {builtin_dict} of type {_type}, expected {expected_type}')
            raise e
        logging.info(f'Looked up {uw_id} found {builtin_dict} of type {_type}, expected {expected_type}')
        assert _type != 'portfolio'  # this is a problem
        logging.info(f'UnderwritingParser._safe_lookup | retrieved {uw_id} type {type(builtin_dict)}')
        if _type != expected_type:
            print(f'WARNING: type of {uw_id} is  {_type}, not expected {expected_type}')
        assert _type == expected_type
        # may need to delete various items
        # if 'note' in builtin_dict:
        #     del builtin_dict['note']
        return builtin_dict.copy()

    @staticmethod
    def dict_to_agg(d):
        """
        convert a spec dictionary d into an agg language specification
        :param d:
        :return:
        """
        pass


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
