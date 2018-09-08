"""
Underwriter Module
==================

Does lots of cool things


"""

import os
from ruamel import yaml
from copy import deepcopy
import numpy as np
import collections
from io import StringIO
from textwrap import fill, indent
from IPython.core.display import display
import matplotlib.pyplot as plt
import logging
import pandas as pd
from .distr import Aggregate, Severity
from .port import Portfolio
from .utils import sensible_jump, html_title

class _DataManager(object):
    """
    _DataManager class
    --------------

    Private class handling reading and writing to YAML files for Underwriter and Example classes, which both
    subclass _YAML_reader


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
            d = list(self.__getattribute__(k).keys())
            sers[k.title()] = pd.Series(d, index=range(len(d)), name=k)
        df = pd.DataFrame(data=sers)
        # df.index.name = 'No.'
        df = df.fillna('')
        return df

    def notes(self, item):
        """
        Pull notes from YAML descriptions for type items

        :param item:
        :return:
        """
        item = item.lower()
        items = list(self.__getattribute__(item).keys())
        notes = [self.__getattribute__(item)[i].get('note', '') for i in items]
        Item = item.title()
        df = pd.DataFrame({Item: items, "Notes": notes})
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
                return k[:-1], self.__getattribute__(k)[item]


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

        self.databases = dict(accounts=['accounts.yaml', 'user_accounts.yaml'],
                              lines=['lines.yaml'],
                              books=['portfolio.yaml', 'user_portfolios.yaml'])
        self.dir_name = dir_name
        _DataManager.__init__(self)

    def __getitem__(self, item):
        """
        handles self[item]
        the result is cast into the right type of object

        :param item:
        :return: Book, Account or Line object
        """
        item_type, obj = _DataManager.__getitem__(self, item)
        if item_type == 'book':
            return Book(item, obj['args'], obj['spec'])
        elif item_type == 'account':
            return Account(**obj)
        elif item_type == 'line':
            return Line(item, **obj)
        else:
            raise ValueError('Idiot, you are in charge, how did you end in this pickle?'
                             f'Invaid type {item_type} to Underwriter getitem')

    def __getattr__(self, item):
        """
        handles self.item

        :param item:
        :return:
        """
        return self.__getitem__(item)


class Line(object):
    """
    Line class
    ----------

    Manages construction of a single line on a notional basis. Contains all the severity information
    needed to create a Severity object and information on ground up frequency (contagion, freq_a,
    freq_b) but not the actual frequency. Frequency is determined by the layer and attachment later.

    No work is done until the line used in an Account. Information is lightweight and stored in a
    dictionary.

        new = other * self; treat as scale change

        scale is a homogeneous change, it adjusts

            - exposure: el, premium
            - sev_mean
            - sev_scale
            - sev_loc
            - limit
            - attachment

        There is no sum of lines: no freq en at this point: TODO: sum could be weighted average of sevs

        self + other ; TODO if other==float then move loc?

    """

    def __init__(self, name='',
                 sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0,
                 sev_xs=None, sev_ps=None, sev_wt=1,
                 freq_name='', freq_a=0, freq_b=0, note=''):
        """

        :param name:
        :param sev_name:
        :param sev_a:
        :param sev_b:
        :param sev_mean:
        :param sev_cv:
        :param sev_loc:
        :param sev_scale:
        :param sev_xs:
        :param sev_ps:
        :param sev_wt:
        :param freq_name:
        :param freq_a:
        :param freq_b:
        :param note:
        """
        self.spec = dict(name=name,
                         sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                         sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                         sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
                         freq_name=freq_name, freq_a=freq_a, freq_b=freq_b)
        self.name = name
        self.note = note

    def __getitem__(self, item):
        return self.spec[item]

    def __getattr__(self, item):
        """
        pass through so items in the spec are treated like they are attributes
        :param item:
        :return:
        """
        if item in self.spec:
            print('pulling form object spec')
            return self.spec[item]
        else:
            # this appears never to get called...
            print('pulling form object get attr')
            return self.__getattribute__(item)

    def __str__(self):
        return dict_2_string(type(self), self.spec)

    def __repr__(self):
        return str(self.spec)

    def __rmul__(self, other):
        """
        new = other * self; treat as scale change

        scale is a homogeneous change, it adjusts

            - exposure: el, premium
            - sev_mean
            - sev_scale
            - sev_loc
            - limit
            - attachment

        :param other:
        :return:
        """

        assert other > 0
        assert isinstance(other, float) or isinstance(other, int)
        other = float(other)

        spec = self._copy_and_adjust(other, ('sev_mean', 'sev_scale', 'sev_loc'))
        spec['name'] = f'{other}.{self.name}'
        return Line(**spec)

    def _copy_and_adjust(self, other, adjust_list):
        """
        performs adjustments for homogeneous and independent scale changes

        :param other:
        :param adjust_list:
        :return:
        """
        d = deepcopy(self.spec)
        for a in adjust_list:
            d[a] *= other

        return d

    def write(self, attachment, limit):
        Severity(self.name, attachment, limit, self.spec['sev_mean'],
                 self.spec['sev_cv'], self.spec['sev_a'], self.spec['sev_b'],
                 self.spec['sev_loc'], self.spec['sev_scale'], self.spec['sev_hxs'],
                 self.spec['sev_hps'], True)


class Account(object):
    """
    Account class
    -------------

    Manages construction of a single account on a notional basis. Contains all the information
    needed to create an Aggregate object but no work is done until the account is "written". Information
    is lightweight and stored in a dictionary. Can read/write from YAML.

    There is only adding of accounts in very restrictive conditions: freq must match, then severities
    are weighted

        new = other * self; treat as scale change

        scale is a homogeneous change, it adjusts

            - exposure: el, premium
            - sev_mean
            - sev_scale
            - sev_loc
            - limit
            - attachment
    """

    def __init__(self, name='', exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                 sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0,
                 sev_xs=None, sev_ps=None, sev_wt=1,
                 freq_name='', freq_a=0, freq_b=0, note=''):

        self.spec = dict(name=name, exp_el=exp_el, exp_premium=exp_premium, exp_lr=exp_lr, exp_en=exp_en,
                         exp_attachment=exp_attachment, exp_limit=exp_limit,
                         sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                         sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                         sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
                         freq_name=freq_name, freq_a=freq_a, freq_b=freq_b)
        self.name = name
        self.note = note

    def __getitem__(self, item):
        return self.spec[item]

    def __str__(self):
        return dict_2_string(type(self), self.spec)

    def __repr__(self):
        return str(self.spec)

    def __add__(self, other):
        """
        Add two Accounts objects: must have matching frequency specs (not enforced?)
        I.e. sev = wtd avg of sevs and freq = sum of freqss

        TODO same severity!

        :param other:
        :return:
        """
        assert isinstance(other, type(self))

        # check fully compatible objects
        compatible_requires = ['freq_name', 'freq_a', 'freq_b']
        compatible_objects = True
        for a in compatible_requires:
            compatible_objects = compatible_objects and self.spec[a] == other.spec[a]
            if not compatible_objects:
                print('Incompatible Accounts: must have same ' + ', '.join(compatible_requires))
                print(f'For attribute {a}: {self.spec[a]} does not equal {other.spec[a]}')

        # now create mixture if severity, limit and attach are different? or just bludegon on?
        easy_add_requires = ['exp_attachment', 'exp_limit',
                             'sev_name', 'sev_a', 'sev_b', 'sev_mean', 'sev_cv', 'sev_loc', 'sev_scale',
                             'sev_xs', 'sev_ps', 'sev_wt']
        easy_add = True
        for a in easy_add_requires:
            easy_add = easy_add and self.spec[a] == other.spec[a]

        if easy_add:
            new_spec = deepcopy(self.spec)
            for a in ['exp_en', 'exp_el', 'exp_premium']:
                new_spec[a] += other.spec[a]
            new_spec['name'] = f'{self.name} + {other.name}'
        else:
            # doh, create mixture
            raise ValueError('Needs mixture, NYI')

        return Account(**new_spec)

    def __rmul__(self, other):
        """
        new = other * self; treat as scale change

        scale is a homogeneous change, it adjust

        :param other:
        :return:
        """

        assert other > 0
        assert isinstance(other, float) or isinstance(other, int)
        other = float(other)
        new_spec = self._copy_and_adjust(other, ('sev_mean', 'sev_scale', 'sev_loc',
                                                 'exp_limit', 'exp_attachment', 'exp_premium', 'exp_el'))
        new_spec['name'] = f'{other}.{self.name}'
        return Account(**new_spec)

    def __mul__(self, other):
        """
        new = self * other, other integer, sum of other independent copies in Levy process sense
        other > 0

        adjusts en and exposure (premium and el)

        :param other:
        :return:
        """

        assert isinstance(other, int) or isinstance(other, float)
        assert other >= 0

        new_spec = self._copy_and_adjust(other, ('exp_premium', 'exp_el', 'exp_en'))
        new_spec['name'] = f'{self.name} to time {other}'
        return Account(**new_spec)

    def _copy_and_adjust(self, other, adjust_list):
        """
        performs adjustments for homogeneous and independent scale changes

        :param other:
        :param adjust_list:
        :return:
        """

        d = deepcopy(self.spec)
        for a in adjust_list:
            d[a] *= other
        return d

    def write(self):
        """
        materialize spec in a full Aggregate class object


        :return:
        """
        return Aggregate(**self.spec)


class Book(object):
    """
    Book class
    ----------

    Manages construction of a "book" of business on a notional basis. Contains all the information
    needed to create a Portfolio object but no work is done until the book is "written". Information
    is lightweight and stored in a dictionary. Can read/write from YAML.


    """

    def __init__(self, name, arg_dict, spec_list):

        self.name = name
        self.arg_dict = arg_dict
        self.spec_list = spec_list
        self.account_list = []
        self.line_names = []
        # actually materialize the specs into Accounts...?WHY?
        for spec in spec_list:
            a = Account(**spec)
            self.account_list.append(a)
            self.line_names.append(spec['name'])

    def __iter__(self):
        """
        make Book iterable: for each x in Book

        :return:
        """
        return iter(self.account_list)

    def __str__(self):
        d = dict(name=self.name)
        for a in self.account_list:
            d[a.name] = a.spec
        return dict_2_string(type(self), d)

    def __repr__(self):
        """
        See portfolio repr

        :return:
        """
        s = [f'{{ "name": "{self.name}"']
        s.append(f'"args":, {str(self.arg_dict)}')
        account_list = []
        for a in self.account_list:
            # references through to the defining spec...as input...
            account_list.append(a.spec.__repr__())
        s.append(f"'spec': [{', '.join(account_list)}]")
        return ', '.join(s) + '}'

    def __add__(self, other):
        """
        Add two book objets INDEPENDENT sum

        TODO same severity!

        :param other:
        :return:
        """
        assert isinstance(other, Book)
        return Book(f'{self.name} + {other.name}', dict(), self.account_list + other.account_list)

    def __rmul__(self, other):
        """
        new = other * self; treat as scale change

        scale is a homogeneous change, it adjusts

            - exposure: el, premium
            - sev_mean
            - sev_scale
            - sev_loc
            - limit
            - attachment

        :param other:
        :return:
        """

        assert other > 0
        assert isinstance(other, float) or isinstance(other, int)
        other = float(other)

        return Book(f'{other}.{self.name}', dict(),
                    self._copy_and_adjust(other, ('sev_mean', 'sev_scale', 'sev_loc',
                                                  'exp_limit', 'exp_attachment', 'exp_premium', 'exp_el')))

    def __mul__(self, other):
        """
        new = self * other, other integer, sum of other independent copies in Levy process sense,
        so other can be fractional, other > 0 required

        adjusts en and exposure (premium and el)

        :param other:
        :return:
        """

        assert isinstance(other, int) or isinstance(other, float)
        assert other >= 0
        return Book(f'{self.name} to time {other}', dict(),
                    self._copy_and_adjust(other, ('exp_premium', 'exp_el', 'exp_en')))

    def _copy_and_adjust(self, other, adjust_list):
        """
        performs adjustments for homogeneous and independent scale changes

        :param other:
        :param adjust_list:
        :return:
        """

        new_spec = []
        for a in self.account_list:
            new_spec.append(deepcopy(a.spec))

        for d in new_spec:
            for a in adjust_list:
                d[a] *= other

        return new_spec

    def write(self, update=False):
        """
        materialize spec in a full Portfolio class object


        :return:
        """
        # need to pass the raw (dictionary) spec_list through, not the account_list = [Account]
        port = Portfolio(self.name, self.spec_list)
        logging.info(f'Book.write | created Portfolio {self.name} from Book')
        if update:
            log2 = self.arg_dict.get('log2', 0)
            bs = self.arg_dict.get('bs', 0)
            if bs and log2:
                logging.info(f'Book.write | updating {self.name}')
                port.update(**self.arg_dict, verbose=False)
            else:
                logging.info(f'Book.write | not missing bs and log2 cannot perform requested update of {self.name}')
        return port


def reporting(port, log2, reporting_level):
    """
    handle various reporting options: most important to appear last

    :param port:
    :param log2:
    :param reporting_level:
    :return:
    """

    if reporting_level >= 3:
        # just plot the densities
        f, axs = plt.subplots(1, 6, figsize=(15, 2.5))
        axiter = iter(axs.flatten())
        port.plot(kind='quick', axiter=axiter)
        port.plot(kind='density', line=port.line_names_ex, axiter=axiter)
        port.plot(kind='density', line=port.line_names_ex, axiter=axiter, legend=False, logy=True)
        plt.tight_layout()

    if reporting_level >= 2:
        jump = sensible_jump(2 ** log2, 10)
        html_title('Line Densities', 1)
        display(port.density_df.filter(regex='^p_[^n]|S|^exa_[^n]|^lev_[^n]').
                query('p_total > 0').iloc[::jump, :])
        html_title('Audit Data', 1)
        display(port.audit_df.filter(regex='^[^n]', axis=0))

    if reporting_level >= 1:
        port.report('quick')
        # html_title('Summary Audit Data', 1)
        # temp = port.audit_df.filter(regex='^Mean|^EmpMean|^CV|^EmpCV')
        # temp['MeanErr'] = temp.EmpMean / temp.Mean - 1
        # temp['CVErr'] = temp.EmpCV / temp.CV - 1
        # temp = temp[['Mean', 'EmpMean', 'MeanErr', 'CV', 'EmpCV', 'CVErr']]
        # display(temp.style.applymap(Example._highlight))

    if reporting_level >= 3:
        html_title('Graphics', 1)


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
            sio.write('\t' * tab_level + ks + '\n')
            print('writing dict', v)
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
                sio.write('\t' * (tab_level + 1) + vv + '\n')
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
