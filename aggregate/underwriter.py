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
                return k[:-1], self.__getattribute__(k)[item]
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

        self.databases = dict(curves=['curves.yaml'],
                              blocks=['blocks.yaml', 'user_blocks.yaml'],
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
            return Book(item, obj['arg_dict'], obj['spec'])
        elif item_type == 'block':
            return Block(item, **obj)
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

    def __call__(self, portfolio_program):
        self.write(portfolio_program)

    def write(self, program, name='', return_type='portfolio', update=False, verbose=False, log2=0, bs=0, **kwargs):
        """
        built a book and materialize as a portfolio from built in blocks
        e.g. 0.01 * cmp; 5.5*scs; comm_auto * 9; cmp+cmp;

        :param program:
        :param name:
        :param return_type:
        :param update:
        :param verbose:
        :param log2:
        :param bs:
        :return:
        """
        logging.info(f'Underwriter.write | creating Portfolio {name} from {program}')
        lexer = BuiltInBlockLexer()
        parser = BuiltInBlockParser(self)
        program = [i.strip() for i in program.replace(';', '\n').split('\n') if len(i.strip()) > 0]
        ans = []
        read_builtins = []
        for txt in program:
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
            ans.append(parser.spec)
            read_builtins += parser.objs
        if verbose:
            print(f'Loaded objects {read_builtins}')
        # ans is a list of Blocks (because it pulls from uw and because
        # it uses the + * ops defined by blocks
        # need to make it into a Book / Portfolio, which needs a spec_list
        spec_list = [{k: v for k, v in a.__dict__.items() if k in Aggregate.aggregate_keys}
                     for a in ans]
        return Underwriter._book_or_portfolio(spec_list, name, return_type, update, verbose, log2, bs, **kwargs)

    def easy_script(self, program):
        """
        minimal argument scripts

        :param program:
        :return:
        """
        return self.script(program, 'easy script', 'portfolio', update=True,
                           verbose=False, log2=13, bs=0, remove_fuzz=True, trim_df=True)

    def script(self, program, name='', return_type='portfolio', update=False, verbose=False, log2=0, bs=0, **kwargs):
        """
        write a pseudo natural language programming spec for a book or (if only one line) an aggregate_project

        e.g. Input
        20  loss 3 x 2 gamma 5 cv 0.30 mixed gamma 0.4
        10  claims 3 x 2 gamma 12 cv 0.30 mixed gamma 1.2
        100  premium at 0.4 3 x 2 4 * lognormal 3 cv 0.8 fixed 1

        See parser for full language spec!

        :param program:
        :param name:
        :param return_type:
        :param update:
        :param verbose:
        :param log2:
        :param bs:
        :param kwargs:
        :return:
        """

        logging.info(f'Underwriter.nlp | creating Portfolio {name} from {program}')
        lexer = NLPBizLexer()
        parser = NLPBizParser(self)
        program = [i.strip() for i in program.replace(';', '\n').split('\n') if len(i.strip()) > 0]
        spec_list = []

        for txt in program:
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
        # spec_list is a list of dictionaries that can be passed straight through to creaet a portfolio
        return Underwriter._book_or_portfolio(spec_list, name, return_type, update, verbose, log2, bs, **kwargs)

    @staticmethod
    def _book_or_portfolio(spec_list, name, return_type, update=False,
                           verbose=False, log2=0, bs=0, **kwargs):
        """
        Finish script or write. make book or portfolio as requested. update as requested

        :param spec_list:
        :param name:
        :param return_type:
        :param update:
        :param verbose:
        :param log2:
        :param bs:
        :param kwargs: arguments passed to Portfolio.update
        :return:
        """
        if name == '':
            name = f'built ins {len(spec_list)}'
        return_type = return_type.lower()
        if return_type == 'book':
            port = Book(name, dict(log2=log2, bs=bs), spec_list)
        elif return_type == 'portfolio':
            port = Portfolio(name, spec_list)
            if update:
                if bs == 0:
                    # for log2 = 10
                    bs = port.recommend_bucket().iloc[-1, 0]
                    if log2 == 0:
                        log2 = 10
                    else:
                        bs = bs * 2**(10 - log2)
                logging.info(f'Underwriter.write | updating Portfolio {name}, log2={10}, bs={bs}')
                port.update(log2=log2, bs=bs, verbose=verbose, **kwargs)
        else:
            raise ValueError(f'Inadmissible argument {return_type} passed to write. Expected book or portfolio.')
        return port


class _ScriptableObject(object):
    """
    object wrapper that implements getitem  to convert attributes to items and so is (sub)scriptable

    """

    def __getitem__(self, item):
        """
        return member using x[item] syntax

        :param item:
        :return:
        """
        return self.__getattribute__(item)

    def keys(self):
        """
        in conjunction with getitem allows the object to be passed using **obj
        and for it to expand to a list of kwargs

        :return:
        """
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        """
        make iterable

        :return:
        """
        return iter(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


class Curve(_ScriptableObject):
    """
    Curve class
    -----------

    A curve corresponds to the underwriting and actuarial notion of a GROUND UP severity curve.

    It contains all the severity information needed to create a Severity object.

    Curves can be scaled and shifted: new = k * old, new = old + k


    """

    def __init__(self, name="", sev_name='', sev_a=0., sev_b=0., sev_mean=0., sev_cv=0.,
                 sev_loc=0., sev_scale=0., sev_xs=None, sev_ps=None, sev_wt=1, note=''):
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
        self.name = name
        self.sev_name = sev_name
        self.sev_a = sev_a
        self.sev_b = sev_b
        self.sev_mean = sev_mean
        self.sev_cv = sev_cv
        self.sev_loc = sev_loc
        self.sev_scale = sev_scale
        self.sev_xs = sev_xs
        self.sev_ps = sev_ps
        self.sev_wt = sev_wt

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
        return Curve(name=self.name,
                     sev_name=self.sev_name,
                     sev_a=self.sev_a,
                     sev_b=self.sev_b,
                     sev_mean=self.sev_mean * other,
                     sev_cv=self.sev_cv,
                     sev_loc=self.sev_loc * other,
                     sev_scale=self.sev_scale * other,
                     sev_xs=None if self.sev_xs is None else np.array(self.sev_xs) * other,
                     sev_ps=self.sev_ps,
                     sev_wt=self.sev_wt,
                     note=f'{other} x {self.name}')

    def write(self):
        return Severity(name=self.sev_name,
                        a=self.sev_a,
                        b=self.sev_b,
                        mean=self.sev_mean,
                        cv=self.sev_cv,
                        loc=self.sev_loc,
                        scale=self.sev_scale,
                        hxs=self.sev_xs,
                        hps=self.sev_ps,
                        conditional=True)


class Block(_ScriptableObject):
    """
    Block class
    ----------------

    Block is a helper class, containing all the information
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

    def __init__(self, name='', exp_el=0., exp_premium=0., exp_lr=0., exp_en=0., exp_attachment=0., exp_limit=np.inf,
                 sev_name='', sev_a=0., sev_b=0., sev_mean=0., sev_cv=0., sev_loc=0., sev_scale=0.,
                 sev_xs=None, sev_ps=None, sev_wt=1,
                 freq_name='', freq_a=0., freq_b=0., note=''):

        self.name = name
        self.exp_el = exp_el
        self.exp_premium = exp_premium
        self.exp_lr = exp_lr
        self.exp_en = exp_en
        self.exp_attachment = exp_attachment
        self.exp_limit = exp_limit
        self.sev_name = sev_name
        self.sev_a = sev_a
        self.sev_b = sev_b
        self.sev_mean = sev_mean
        self.sev_cv = sev_cv
        self.sev_loc = sev_loc
        self.sev_scale = sev_scale
        self.sev_xs = sev_xs
        self.sev_ps = sev_ps
        self.sev_wt = sev_wt
        self.freq_name = freq_name
        self.freq_a = freq_a
        self.freq_b = freq_b
        # convenient to sort this out for adding etc.
        if self.exp_premium > 0 and self.exp_el > 0:
            self.exp_lr = self.exp_el / self.exp_premium
        elif self.exp_el > 0 and self.exp_lr > 0:
            self.exp_premium = self.exp_el / self.exp_lr
        elif self.exp_lr > 0 and self.exp_premium > 0:
            self.exp_el = self.exp_lr * self.exp_premium
        self.note = note

    # def __str__(self):
    #     return dict_2_string(type(self), dict(  ))
    #
    # def __repr__(self):
    #     return str(self.SOMETHING)
    #
    def __add__(self, other):
        """
        Add two Block objects: must have matching frequency specs (not enforced?)
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
            compatible_objects = compatible_objects and self.__getattribute__(a) == other.__getattribute__(a)
            if not compatible_objects:
                print('Incompatible Accounts: must have same ' + ', '.join(compatible_requires))
                print(f'For attribute {a}: {self.__getattribute__(a)} does not equal {other.__getattribute__(a)}')

        # now create mixture if severity, limit and attach are different? or just bludegon on?
        easy_add_requires = ['exp_attachment', 'exp_limit',
                             'sev_name', 'sev_a', 'sev_b', 'sev_mean', 'sev_cv', 'sev_loc', 'sev_scale',
                             'sev_xs', 'sev_ps', 'sev_wt', 'freq_name', 'freq_a', 'freq_b']
        easy_add = True
        for a in easy_add_requires:
            easy_add = easy_add and self.__getattribute__(a) == other.__getattribute__(a)

        if easy_add:
            prem = self.exp_premium + other.exp_premium
            loss = self.exp_el + other.exp_el
            if prem > 0:
                lr = loss / prem
            else:
                lr = 0
            return Block(name=f'{self.name} + {other.name}',
                         exp_el=loss,
                         exp_premium=prem,
                         exp_lr=lr,
                         exp_en=self.exp_en + other.exp_en,
                         exp_attachment=self.exp_attachment,
                         exp_limit=self.exp_limit,
                         sev_name=self.sev_name,
                         sev_a=self.sev_a,
                         sev_b=self.sev_b,
                         sev_mean=self.sev_mean,
                         sev_cv=self.sev_cv,
                         sev_loc=self.sev_loc,
                         sev_scale=self.sev_scale,
                         sev_xs=self.sev_xs,
                         sev_ps=self.sev_ps,
                         sev_wt=self.sev_wt,
                         freq_name=self.freq_name,
                         freq_a=self.freq_a,
                         freq_b=self.freq_b,
                         note='')
        else:
            # doh, create mixture
            raise ValueError('Needs mixture, NYI')

    def __rmul__(self, other):
        """
        new = other * self; treat as homogeneous scale change

        :param other:
        :return:
        """

        assert other > 0
        assert isinstance(other, float) or isinstance(other, int)
        other = float(other)
        if self.sev_xs is not None:
            xs = other * np.array(self.sev_xs)
        else:
            xs = None
        return Block(name=f'{other} {self.name}',
                     exp_el=other * self.exp_el,
                     exp_premium=other * self.exp_premium,
                     exp_lr=self.exp_lr,
                     exp_en=self.exp_en,
                     exp_attachment=other * self.exp_attachment,
                     exp_limit=other * self.exp_limit,
                     sev_name=self.sev_name,
                     sev_a=self.sev_a,
                     sev_b=self.sev_b,
                     sev_mean=other * self.sev_mean,
                     sev_cv=self.sev_cv,
                     sev_loc=other * self.sev_loc,
                     sev_scale=other * self.sev_scale,
                     sev_xs=xs,
                     sev_ps=self.sev_ps,
                     sev_wt=self.sev_wt,
                     freq_name=self.freq_name,
                     freq_a=self.freq_a,
                     freq_b=self.freq_b,
                     note='')

    def __mul__(self, other):
        """
        new = self * other, sum of other independent copies in Levy process sense
        other > 0

        adjusts en and exposure (premium and el)

        :param other:
        :return:
        """

        assert isinstance(other, int) or isinstance(other, float)
        assert other >= 0

        return Block(name=f'{self.name}⊕{other}',
                     exp_el=other * self.exp_el,
                     exp_premium=other * self.exp_premium,
                     exp_lr=self.exp_lr,
                     exp_en=other * self.exp_en,
                     exp_attachment=self.exp_attachment,
                     exp_limit=self.exp_limit,
                     sev_name=self.sev_name,
                     sev_a=self.sev_a,
                     sev_b=self.sev_b,
                     sev_mean=other * self.sev_mean,
                     sev_cv=self.sev_cv,
                     sev_loc=other * self.sev_loc,
                     sev_scale=other * self.sev_scale,
                     sev_xs=self.sev_xs,
                     sev_ps=self.sev_ps,
                     sev_wt=self.sev_wt,
                     freq_name=self.freq_name,
                     freq_a=self.freq_a,
                     freq_b=self.freq_b,
                     note='')

    def write(self):
        """
        materialize Block in a full Aggregate class object
        TODO: do we need , attachment=0., limit=np.inf):??

        :return:
        """
        return Aggregate(name=self.name,
                         exp_el=self.exp_el,
                         exp_premium=self.exp_premium,
                         exp_lr=self.exp_lr,
                         exp_en=self.exp_en,
                         exp_attachment=self.exp_attachment,
                         exp_limit=self.exp_limit,
                         sev_name=self.sev_name,
                         sev_a=self.sev_a,
                         sev_b=self.sev_b,
                         sev_mean=self.sev_mean,
                         sev_cv=self.sev_cv,
                         sev_loc=self.sev_loc,
                         sev_scale=self.sev_scale,
                         sev_xs=self.sev_xs,
                         sev_ps=self.sev_ps,
                         sev_wt=self.sev_wt,
                         freq_name=self.freq_name,
                         freq_a=self.freq_a,
                         freq_b=self.freq_b)


class Book(_ScriptableObject):
    """
    Book class
    ----------

    Manages construction of a "book" of business on a notional basis. Contains all the information
    needed to create a Portfolio object but no work is done until the book is "written". Information
    is lightweight and stored in a dictionary. Can read/write from YAML.

    Adding assumes all components are independent...down road may want to revise and look through
    to severity and add groups?

    """

    def __init__(self, name, arg_dict, spec_list):

        self.name = name
        self.arg_dict = arg_dict
        self.spec_list = deepcopy(spec_list)

    def __str__(self):
        d = dict(name=self.name)
        for a in self.spec_list:
            d[a.name] = str(a)
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
        s.append(f'"arg_dict": {str(self.arg_dict)}')
        account_list = []
        for a in self.spec_list:
            account_list.append(repr(a))
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
        return Book(f'{self.name} + {other.name}',
                    dict(),
                    self.spec_list + other.spec_list)

    def write(self, update=False):
        """
        materialize Book in a full Portfolio class object


        :return:
        """
        # need to pass the raw (dictionary) spec_list
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


# def reporting(port, log2, reporting_level):
#     """
#     handle various reporting options: most important to appear last
#
#     :param port:
#     :param log2:
#     :param reporting_level:
#     :return:
#     """
#
#     if reporting_level >= 3:
#         # just plot the densities
#         f, axs = plt.subplots(1, 6, figsize=(15, 2.5))
#         axiter = iter(axs.flatten())
#         port.plot(kind='quick', axiter=axiter)
#         port.plot(kind='density', line=port.line_names_ex, axiter=axiter)
#         port.plot(kind='density', line=port.line_names_ex, axiter=axiter, legend=False, logy=True)
#         plt.tight_layout()
#
#     if reporting_level >= 2:
#         jump = sensible_jump(2 ** log2, 10)
#         html_title('Line Densities', 1)
#         display(port.density_df.filter(regex='^p_[^n]|S|^exa_[^n]|^lev_[^n]').
#                 query('p_total > 0').iloc[::jump, :])
#         html_title('Audit Data', 1)
#         display(port.audit_df.filter(regex='^[^n]', axis=0))
#
#     if reporting_level >= 1:
#         port.report('quick')
#
#     if reporting_level >= 3:
#         html_title('Graphics', 1)


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


class BuiltInBlockLexer(Lexer):
    tokens = {LINE, NUMBER}
    ignore = ' \t'
    literals = {'+', '*'}

    # Tokens
    LINE = r'[a-zA-Z_][a-zA-Z0-9_]*'

    @_(r'\d+\.?\d*([eE](\+|\-)?\d+)?')
    def NUMBER(self, t):
        t.value = float(t.value)
        return t

    @_(r'\n+')
    def newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        print(f"Illegal character '{t.value[0]:s}'")
        self.index += 1


class BuiltInBlockParser(Parser):
    tokens = BuiltInBlockLexer.tokens
    precedence = (('left', '+'), ('left', '*'))

    def __init__(self, uw):
        self.names = uw.blocks
        self.uw = uw
        self.spec = None
        self.objs = None
        self.reset()

    def reset(self):
        self.spec = None
        self.objs = []

    @_('term')
    def empty(self, p):
        # print('appending')
        self.spec = p.term

    @_('term "+" term')
    def term(self, p):
        # print('adding t')
        return p.term0 + p.term1

    @_('number "*" term')
    def term(self, p):
        # print('n x t')
        return p.number * p.term

    @_('term "*" number')
    def term(self, p):
        # print('t x n')
        return p.term * p.number

    @_('NUMBER')
    def number(self, p):
        # print('num')
        return p.NUMBER

    @_('LINE')
    def term(self, p):
        try:
            obj = getattr(self.uw, p.LINE)
            self.objs.append((type(obj), p.LINE))
            logging.info(f'Underwriter.write | Loading object {p.LINE} of type {type(obj)} from uw library')
            return obj
        except LookupError:
            print(f"Undefined name {p.LINE}")
            return 0

    def error(self, p):
        if p:
            raise ValueError(p)
        else:
            raise ValueError('Unexpcted end of file')


class NLPBizLexer(Lexer):
    tokens = {ID, PLUS, MINUS, TIMES, NUMBER, CV, LOSS, PREMIUM, AT, LR, CLAIMS, XS, MIXED,
              FIXED, POISSON, BUILTIN}
    ignore = ' \t,\\:\\(\\)'

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
    ID['builtin'] = BUILTIN
    ID['BUILTIN'] = BUILTIN
    ID['bin'] = BUILTIN
    ID['BIN'] = BUILTIN

    @_(r'\-?(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?')
    def NUMBER(self, t):
        t.value = float(t.value)
        return t

    @_(r'\n+')
    def newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        print(f"Illegal character '{t.value[0]:s}'")
        self.index += 1


class NLPBizParser(Parser):
    tokens = NLPBizLexer.tokens
    precedence = (('left', PLUS, MINUS), ('left', TIMES))

    def __init__(self, uw):
        self.arg_dict = None
        self.reset()
        # instance of uw class to look up severities
        self.uw = uw

    def reset(self):
        # TODO pull from Aggregate automatically ...
        self.arg_dict = dict(name="", exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                             sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_scale=0, sev_loc=0,
                             freq_name='', freq_a=0, freq_b=0)

    @_('name expos sev freq')
    def ans(self, p):
        pass

    @_('name expos limit sev freq')
    def ans(self, p):
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
        self.arg_dict['sev_scale'] = p.NUMBER

    @_('ID NUMBER CV NUMBER')
    def sev(self, p):
        self.arg_dict['sev_name'] = p.ID
        self.arg_dict['sev_mean'] = p[1]
        self.arg_dict['sev_cv'] = p[3]
        return True

    @_('ID NUMBER NUMBER')
    def sev(self, p):
        self.arg_dict['sev_name'] = p.ID
        self.arg_dict['sev_a'] = p[1]
        self.arg_dict['sev_b'] = p[2]
        return True

    @_('BUILTIN ID')
    def sev(self, p):
        # look up ID in uw
        sev_dist = self.uw[p.ID]
        print(f'retrieved {p.ID} type {type(sev_dist)}')
        assert isinstance(sev_dist, Curve)
        self.arg_dict['sev_name'] = sev_dist.sev_name
        self.arg_dict['sev_a'] = sev_dist.sev_a
        self.arg_dict['sev_b'] = sev_dist.sev_b
        self.arg_dict['sev_mean'] = sev_dist.sev_mean
        self.arg_dict['sev_cv'] = sev_dist.sev_cv
        self.arg_dict['sev_loc'] = sev_dist.sev_loc
        self.arg_dict['sev_scale'] = sev_dist.sev_scale
        self.arg_dict['sev_xs'] = sev_dist.sev_xs
        self.arg_dict['sev_ps'] = sev_dist.sev_ps
        self.arg_dict['sev_wt'] = sev_dist.sev_wt
        print(self.arg_dict)

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
        self.arg_dict['exp_loss'] = p[0] * p[3]
        return True

    @_('NUMBER PREMIUM AT NUMBER')
    def expos(self, p):
        self.arg_dict['exp_premium'] = p[0]
        self.arg_dict['exp_lr'] = p[3]
        self.arg_dict['exp_el'] = p[0] * p[3]
        return True

    @_('ID')
    def name(self, p):
        self.arg_dict['name'] = p[0]

    def error(self, p):
        if p:
            raise ValueError(p)
        else:
            raise ValueError('Unexpected end of file')
