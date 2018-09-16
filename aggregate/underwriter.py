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
from .parser import Runner  # UnderwritingLexer, UnderwritingParser


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
                    # ?
                    # d = dict(**d, **yaml.load(f, Loader=yaml.Loader))
                    # better ?!
                    temp = yaml.load(f, Loader=yaml.Loader)
                    for kt, vt in temp.items():
                        d[kt] = vt
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

    def __init__(self, dir_name="", store_mode=True, debug=False):
        """

        :param dir_name:
        :param store_mode: add newly created aggregates to the database?
        :param debug: run parser in debug mode?
        """
        self.last_spec = None
        self.databases = dict(severity=['severities.yaml'],
                              aggregate=['aggregates.yaml', 'user_aggregates.yaml'],
                              portfolio=['portfolios.yaml', 'user_portfolios.yaml'])
        self.dir_name = dir_name
        self.store_mode = store_mode
        self.runner = Runner(self, debug=debug)
        _DataManager.__init__(self)

    def __getitem__(self, item):
        """
        handles self[item]
        the result is just a dictionary
        this method is used by the Parser etc.

        :param item:
        :return: Book, Account or Line object
        """
        return _DataManager.__getitem__(self, item)

    def __getattr__(self, item):
        """
        handles self.item and returns an appropriate object

        :param item:
        :return:
        """
        return self.write(item)  # _DataManager.__getitem__(self, item)

    # def get_dict(self, item):
    #     """
    #     get an item as dictionary, WITHOUT the type
    #
    #     :param item:
    #     :return:
    #     """
    #     _type, obj = _DataManager.__getitem__(self, item)
    #     return obj

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

        # first see if it is a built in object
        lookup_success = True
        try:
            _type, obj = _DataManager.__getitem__(self, portfolio_program)
        except LookupError:
            lookup_success = False
            print(f'Warning: object {portfolio_program} not found...assuming program...')
        if lookup_success:
            if _type == 'aggregate':
                return Aggregate(portfolio_program, **obj)
            elif _type == 'portfolio':
                return Portfolio(portfolio_program, obj['spec'])
            elif _type == 'severity':
                return Severity(**obj)
            else:
                ValueError(f'Cannot build {_type} objects')
            return obj

        spec_list = self.runner.production_run(portfolio_program)
        # add newly created item to the built in list
        if self.store_mode:
            for a in spec_list:
                self.aggregate[a['name']] = a

        if portfolio_name == '':
            portfolio_name = f'built ins {len(spec_list)}'
        logging.info(f'Underwriter.write | creating Portfolio {portfolio_name} from {portfolio_program[0:20]}')
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

    def test_write(self, portfolio_program):
        """
        parse program in debug mode

        :param portfolio_program:
        :return:
        """
        return self.runner.test_run(portfolio_program)

    def write_from_file(self, file_name, portfolio_name='', update=False, verbose=False, log2=0, bs=0, **kwargs):
        """
        read program from file. delegates to write

        :param file_name:
        :param portfolio_name:
        :param update:
        :param verbose:
        :param log2:
        :param bs:
        :param kwargs:
        :return:
        """
        with open(file_name, 'r', encoding='utf-8') as f:
            portfolio_program = f.read()
        return self.write(portfolio_program, portfolio_name, update, verbose, log2, bs, **kwargs)


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
