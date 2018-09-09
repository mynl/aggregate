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

        self.databases = dict(curves=['curves.yaml'],
                              accounts=['accounts.yaml', 'user_accounts.yaml'],
                              lines=['lines.yaml'],
                              catlines=['catlines.yaml'],
                              books=['books.yaml', 'user_books.yaml'])
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
            return Account(item, **obj)
        elif item_type == 'catline':
            return CatLine(item, **obj)
        elif item_type == 'line':
            return Line(item, **obj)
        elif item_type == 'curve':
            return Curve(item, **obj)
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


class _ScriptableObject(object):
    """
    object wrapper that implements getitem and getattr on a member collection (the contained_iterable)
    used by Lines etc.
    """

    def __init__(self, spec_dict):
        self.contained_iterable = spec_dict

    def __getattr__(self, item):
        """
        return member using x.member syntax

        :param item:
        :return:
        """
        if item in self.__dict__:
            # print(f'getattr __dict__[{item}] ')
            # print(self.__dict__)
            return self.__dict__[item]
        elif item in self.contained_iterable:
            # print(f'getattr from contained object[{item}]')
            return self.contained_iterable[item]
        else:
            # print(f'item {item} not found in getattr, just returning...')
            return

    def __getitem__(self, item):
        """
        return member using x[item] syntax

        :param item:
        :return:
        """
        # print(f'getitem {item} requested')
        return self.contained_iterable[item]

    def __setitem__(self, key, value):
        self.contained_iterable.__setitem__(key, value)

    # def __setattr__(self, key, value):
    #     #    do not allow to add items to the spec?
    #     if key in self.contained_iterable:
    #         self.contained_iterable.__setitem__(key, value)
    #     else:
    #         object.__setattr__(key, value )

    def __deepcopy__(self, memodict={}):
        # https://www.peterbe.com/plog/must__deepcopy__
        return deepcopy(self.contained_iterable)

    def keys(self):
        """
        in conjunction with getitem allows the object to be passed using **obj
        and for it to expand to a list of kwargs


        :return:
        """
        return self.contained_iterable.keys()

    def values(self):
        return self.contained_iterable.values()

    def items(self):
        return self.contained_iterable.items()

    def __iter__(self):
        """
        make iterable

        :return:
        """
        return iter(self.contained_iterable)


class Curve(_ScriptableObject):
    """
    Curve class
    -----------

    A curve corresponds to the underwriting and actuarial notion of a GROUND UP severity curve.

    It contains all the severity information needed to create a Severity object.

    Curves can be scaled and shifted: new = k * old, new = old + k


    """

    def __init__(self, name="", sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0,
                 sev_loc=0, sev_scale=0, sev_xs=None, sev_ps=None, sev_wt=1, note=''):
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
        :param note:
        """

        self.note = note
        _ScriptableObject.__init__(self, dict(name=name,
                                              sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                                              sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                                              sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt))

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
        return Curve(**spec)

    def write(self):
        return Severity(**self)

    # def __add__(self, other):
    #     """
    #     shift
    #
    #     """
    #     assert isinstance(other, float) or isinstance(other, int)
    #     other = float(other)
    #
    #     """
    #     d = deepcopy(self.contained_iterable)
    #     for a in adjust_list:
    #         d[a] *= other
    #
    #     return d
    #     """
    #     spec = deepcopy(self.contained_iterable)
    #     spec['sev_loc'] += other
    #     return Curve(**self)


class _Aggregate(_ScriptableObject):
    """
    _Aggregate class
    ----------------

    _Aggregate is a helper class, containing all the information
    needed to create an Aggregate object but no work is done until the account is "written".

    Information is lightweight and stored in a dictionary.

    Not created directly but subclasssed by Line, CatLine and Account which check appropriate information
    has been included in their specifications.

    Generally created by an underwriter.

    Can read/write from YAML.

    An Account is also a Line but is guaranteed to contain detailed severity information.

    Methods:

    * new = other * self; treat as homogeneous scale change adjusting
            - exposure: el, premium
            - sev_mean
            - sev_scale
            - sev_loc
            - limit
            - attachment

    * An Account and a CatLine are both Lines but Lines are do not contain enough information to be Accounts or CatLines
    * There is no self * other for inhomogeneous volume change
    * There is no sum of lines: create a Book to add. :

    """

    def __init__(self, name='', exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                 sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0,
                 sev_xs=None, sev_ps=None, sev_wt=1,
                 freq_name='', freq_a=0, freq_b=0, note=''):

        self.note = note
        _ScriptableObject.__init__(self, dict(name=name, exp_el=exp_el, exp_premium=exp_premium, exp_lr=exp_lr, exp_en=exp_en,
                                        exp_attachment=exp_attachment, exp_limit=exp_limit,
                                        sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                                        sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                                        sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
                                        freq_name=freq_name, freq_a=freq_a, freq_b=freq_b))

    def __str__(self):
        return dict_2_string(type(self), self.contained_iterable)

    def __repr__(self):
        return str(self.contained_iterable)

    def __add__(self, other):
        """
        Add two _Aggregate objects: must have matching frequency specs (not enforced?)
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
            compatible_objects = compatible_objects and self.contained_iterable[a] == other.contained_iterable[a]
            if not compatible_objects:
                print('Incompatible Accounts: must have same ' + ', '.join(compatible_requires))
                print(f'For attribute {a}: {self.contained_iterable[a]} does not equal {other.contained_iterable[a]}')

        # now create mixture if severity, limit and attach are different? or just bludegon on?
        easy_add_requires = ['exp_attachment', 'exp_limit',
                             'sev_name', 'sev_a', 'sev_b', 'sev_mean', 'sev_cv', 'sev_loc', 'sev_scale',
                             'sev_xs', 'sev_ps', 'sev_wt']
        easy_add = True
        for a in easy_add_requires:
            easy_add = easy_add and self.contained_iterable[a] == other.contained_iterable[a]

        if easy_add:
            new_spec = deepcopy(self.contained_iterable)
            for a in ['exp_en', 'exp_el', 'exp_premium']:
                new_spec[a] += other.contained_iterable[a]
            new_spec['name'] = f'{self.name} + {other.name}'
        else:
            # doh, create mixture
            raise ValueError('Needs mixture, NYI')

        # the caller has to create the right type of object
        return new_spec

    def __rmul__(self, other):
        """
        new = other * self; treat as homogeneous scale change

        :param other:
        :return:
        """

        assert other > 0
        assert isinstance(other, float) or isinstance(other, int)
        other = float(other)
        new_spec = self._copy_and_adjust(other, ('sev_mean', 'sev_scale', 'sev_loc', 'sev_attachment', 'sev_limit',
                                                 'exp_limit', 'exp_attachment', 'exp_premium', 'exp_el'))
        new_spec['name'] = f'{other}.{self.name}'
        return new_spec

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
        return new_spec

    def _copy_and_adjust(self, other, adjust_list):
        """
        performs adjustments for homogeneous and independent scale changes

        :param other:
        :param adjust_list:
        :return:
        """
        d = deepcopy(self.contained_iterable)
        for a in set(adjust_list).intersection(d.keys()):
            d[a] *= other

        return d

    def write(self):
        """
        materialize contained_iterable in a full Aggregate class object
        TODO: do we need , attachment=0, limit=np.inf):??

        :return:
        """
        # swap ??
        # a0 = self.contained_iterable['attachment']
        # l0 = self.contained_iterable['limit']
        # self.contained_iterable['attachment'] = attachment
        # self.contained_iterable['limit'] = limit
        # agg = Aggregate(**self)
        # self.contained_iterable['attachment'] = a0
        # self.contained_iterable['limit'] = l0

        return Aggregate(**self)


class Line(_Aggregate):
    """
    Line class
    ----------

    A line corresponds to the underwriting and actuarial notion of a line of business. It is designed for
    non-cat lines of business and is not guaranteed to contain meaningful severity information. It should
    not be used with limit and attachment. Often the severity is fixed at a constant. Asmptotically a
    corresponding Account will tend to its Line.

    If detailed severity information is needed use either an Account object or a CatLine object, both
    of which have meaningful severity information.

    Contains information on [ground-up] frequency, in particular contagion = freq_a,
    freq_b) and the frequency information appropriate for the total industry writings in the line.

    May, but not required to also contain premium and/or loss ratio information for the line.

    No work is done until the line used in an Aggregate or Portfolio.

    Information is lightweight and stored in a dictionary.



    """

    def __init__(self, name='',
                 exp_el=0, exp_premium=0, exp_lr=0, exp_en=0,
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
        # TODO: print('Line class checking information provided...')
        self.note = note
        _ScriptableObject.__init__(self, dict(name=name,
                                              exp_el=exp_el, exp_premium=exp_premium, exp_lr=exp_lr, exp_en=exp_en,
                                              sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                                              sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                                              sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
                                              freq_name=freq_name, freq_a=freq_a, freq_b=freq_b))

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
        return Line(**_Aggregate.__rmul__(self, other))
        # assert other > 0
        # assert isinstance(other, float) or isinstance(other, int)
        # other = float(other)
        #
        # spec = self._copy_and_adjust(other, ('sev_mean', 'sev_scale', 'sev_loc'))
        # spec['name'] = f'{other}.{self.name}'
        # return Line(**spec)

    def __add__(self, other):
        # somehow demonstrate this is not acceptable
        raise ValueError('Cannot add lines...')

    def __mul__(self, other):
        # somehow demonstrate this is not acceptable
        raise ValueError('Cannot perform inhomogeneous exposure change on Line objects...')

    # def write(self):
    #     return _Aggregate.write(self)


class Account(_Aggregate):
    """
    Account class
    -------------

    Manages construction of a single account. Contains all the information
    needed to create an Aggregate object but no work is done until the account is "written". Information
    is lightweight and stored in a dictionary. Can read/write from YAML. Generally created by an underwriter.

    An Account is also a Line but is guaranteed to contain detailed severity information.

    There is only adding of accounts in very restrictive conditions: freq must match, then severities
    are weighted

    rmul, add and mul are all valid...all passed through to _Aggregate

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
        # check you are given enough information
        # TODO: print('Account class checking information provided...')
        self.note = note
        _Aggregate.__init__(self,
                            **dict(name=name, exp_el=exp_el, exp_premium=exp_premium, exp_lr=exp_lr, exp_en=exp_en,
                                   exp_attachment=exp_attachment, exp_limit=exp_limit,
                                   sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                                   sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                                   sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
                                   freq_name=freq_name, freq_a=freq_a, freq_b=freq_b))

    def __add__(self, other):
        return Account(**_Aggregate.__add__(self, other))

    def __rmul__(self, other):
        return Account(**_Aggregate.__rmul__(self, other))

    def __mul__(self, other):
        return Account(**_Aggregate.__mul__(self, other))


class CatLine(_Aggregate):
    """
    CatLine class
    -------------

    Manages construction of an industry catastrophe line, e.g. US Wind, US Quake.
    Contains all the information blah...
    needed to create an Aggregate object but no work is done until the account is "written". Information
    is lightweight and stored in a dictionary. Can read/write from YAML. Generally created by an underwriter.

    An Account is also a Line but is guaranteed to contain detailed severity information.

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
        # check you are given enough information
        # TODO: print('CatLine class checking information provided...')
        assert sev_name != ''
        assert np.sum(np.abs(np.array([sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale]))) > 0
        if sev_ps is not None:
            assert np.sum(np.array(sev_ps)) > 0
        assert freq_name != ''
        assert exp_en > 0
        self.note = note
        _Aggregate.__init__(self,
                            **dict(name=name,  exp_el=exp_el, exp_premium=exp_premium, exp_lr=exp_lr, exp_en=exp_en,
                                   exp_attachment=exp_attachment, exp_limit=exp_limit,
                                   sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                                   sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                                   sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
                                   freq_name=freq_name, freq_a=freq_a, freq_b=freq_b))

    def __add__(self, other):
        # somehow demonstrate this is not acceptable
        raise ValueError('Cannot add CatLines...')

    def __mul__(self, other):
        # somehow demonstrate this is not acceptable
        raise ValueError('Cannot perform inhomogeneous exposure change on CatLine objects...')

    def __rmul__(self, other):
        return CatLine(**_Aggregate.__rmul__(self, other))


class Book(_ScriptableObject):
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
        # self.spec_list = spec_list
        account_list = []
        self.line_names = []
        # actually materialize the specs into Accounts  TODO: WHY are you doing this rather than leaving as dict?
        for spec in spec_list:
            if isinstance(spec, Account):
                account_list.append(spec)
            else:
                account_list.append(Account(**spec))
            self.line_names.append(spec['name'])
        _ScriptableObject.__init__(self, account_list)

    def __str__(self):
        d = dict(name=self.name)
        for a in self.contained_iterable:
            d[a.name] = str(a) # .spec_dict
        return dict_2_string(type(self), d)

    def __repr__(self):
        """
        See portfolio repr

        :return:
        """
        # delegate
        # return _ScriptableObject.__repr__(self)
        s = list()
        s.append(f'{{ "name": "{self.name}"')
        s.append(f'"args": {str(self.arg_dict)}')
        account_list = []
        for a in self.contained_iterable:
            # references through to the defining contained_iterable...as input...
            account_list.append(repr(a)) # .__repr__())
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
        return Book(f'{self.name} + {other.name}', dict(), self.contained_iterable + other.contained_iterable)

    def __rmul__(self, other):
        """
        new = other * self; treat as homogeneous scale change

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

        # TODO here and elsewhere need to scale xs for histogram distributions...
        return Book(f'{other:.0f} {self.name}', self.arg_dict,
                    self._copy_and_adjust(other, ('sev_mean', 'sev_scale', 'sev_loc', # 'sev_xs',
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

        new_spec = deepcopy(self.contained_iterable)

        for d in new_spec:
            for a in adjust_list:
                d[a] *= other

        return new_spec

    def write(self, update=False):
        """
        materialize contained_iterable in a full Portfolio class object


        :return:
        """
        # need to pass the raw (dictionary) spec_list through, not the account_list = [Account]
        port = Portfolio(self.name, self.contained_iterable)
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
