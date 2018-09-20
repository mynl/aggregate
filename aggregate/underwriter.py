"""
Underwriter Class
=================

The Underwriter is an easy to use interface into the computational functionality of aggregate.

The Underwriter

* Maintains a default library of severity curves
* Maintains a default library of aggregate distributions corresponding to industry losses in
  major classes of business, total catastrophe losses from major perils, and other useful constructs
* Maintains a default library of portfolios, including several example instances and examples used in
  papers on risk theory (e.g. the Bodoff examples)


The library functions can be listed using

        uw.list()

or, for more detail

        uw.describe()

A given example can be inspected using ``uw['cmp']`` which returns the defintion of the database
object cmp (an aggregate representing industry losses from the line Commercial Multiperil). It can
be created as an Aggregate class using ``ag = uw('cmp')``. The Aggregate class can then be updated,
plotted and various reports run on it. In iPython or Jupyter ``ag`` returns an informative HTML
description.

The real power of Underwriter is access to the agg scripting language (see parser module). The scripting
language allows severities, aggregates and portfolios to be created using more-or-less natural language.
For example

        pf = uw('''
        port MyCompanyBook
            agg LineA 100 claims 100000 xs 0 sev lognorm 30000 cv 1.25
            agg LineB 150 claims 250000 xs 5000 sev lognorm 50000 cv 0.9
            agg Cat 2 claims 100000000 xs 0 sev 500000 * pareto 1.8 - 500000
        ''')

creates a portfolio with three sublines, LineA, LineB and Cat. LineA is 100 (expected) claims, each pulled
from a lognormal distribution with mean of 30000 and coefficient of variation 1.25 within the layer
100000 xs 0 (i.e. limited at 100000). The frequency distribution is Poisson. LineB is similar. Cat is jsut
2 claims from the indicated limit, with severity given by a Pareto distribution with shape parameter 1.8,
scale 500000, shifted left by 500000. This corresponds to the usual Pareto with survival function
S(x) = (lambda / (lambda + x))^1.8, x >= 0.

The portfolio can be approximated using FFTs to convolve the aggregates and add the lines. The severities
are first discretized using a certain bucket-size (bs). The port object has a port.recommend_bucket() to
suggest reasonable buckets:

>> pf.recommend_bucket()

|       |      bs10 |     bs11 |       bs12 |       bs13 |      bs14 |      bs15 |      bs16 |   bs18 |   bs20 |
|:------|----------:|---------:|-----------:|-----------:|----------:|----------:|----------:|-------:|-------:|
| LineA |   3,903     |  1,951     |      976     |      488     |     244     |     122     |      61.0 |   15.2 |    3.8 |
| LineB |   8,983     |  4,491     |  2,245       |  1,122       |     561     |     280     |     140     |   35.1 |    8.8 |
| Cat   |  97,656     | 48,828     | 24,414       | 12,207       | 6,103       | 3,051       | 1,525       |  381     |   95.4 |
| total | 110,543     | 55,271     | 27,635       | 13,817       | 6,908       | 3,454       | 1,727       |  431     |  108     |

The column bsNcorrespond to discretizing with 2**N buckets. The rows show suggested bucket sizes for each
line and in total. For example with N=13 (i.e. 8196 buckets) the suggestion is 13817. It is best the bucket
size is a divisor of any limits or attachment points, so we select 10000.

Updating can then be run as

        bs = 10000
        pf.update(13, bs)
        pf.report('quick')
        pf.plot('density')
        pf.plot('density', logy=True)
        print(pf)

Portfolio name           MyCompanyBook
Theoretic expected loss     10,684,541.2
Actual expected loss        10,657,381.1
Error                          -0.002542
Discretization size                   13
Bucket size                     10000.00
<aggregate.port.Portfolio object at 0x0000023950683CF8>


Etc. etc.

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

    def __init__(self, dir_name="", name='Rory', databases=None, store_mode=True, update=False,
                 verbose=False, log2=10, debug=False):
        """

        :param dir_name:
        :param databases:
        :param store_mode: add newly created aggregates to the database?
        :param update:
        :param log2:
        :param debug: run parser in debug mode?
        """

        self.last_spec = None
        self.name = name
        self.store_mode = store_mode
        self.update = update
        self.log2 = log2
        self.debug = debug
        self.verbose = verbose  # for update
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

    def _repr_html_(self):
        s = [f'<h1>Underwriter {self.name}</h1>']
        s.append(f'Underwriter knows about {len(self.severity)} severities, {len(self.aggregate)} aggregates'
                 f' and {len(self.portfolio)} portfolios<br>')
        for what in ['severity', 'aggregate', 'portfolio']:
            s.append(f'<b>{what.title()}</b>: ')
            s.append(', '.join([k for k in sorted(getattr(self, what).keys())]))
            s.append('<br>')
        s.append(f'<h3>Settings</h3>')
        for k in ['update', 'log2', 'store_mode', 'verbose', 'last_spec']:
            s.append(f'{k}: {getattr(self, k)}; ')
        return '\n'.join(s)

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

    def write(self, portfolio_program, **kwargs):
        """
        write a pseudo natural language programming spec for a book or (if only one line) an aggregate_project

        e.g. Input
        port my_portfolio
            20  loss 3 x 2 sev gamma 5 cv 0.30 mixed gamma 0.4
            10  claims 3 x 2 sevgamma 12 cv 0.30 mixed gamma 1.2
            100  premium at 0.4 3 x 2 sev 4 * lognormal 3 cv 0.8 fixed 1

        The indents are required...

        See parser for full language spec!

        Reasonable kwargs:

            bs
            log2
            verbose
            update overrides class default
            add_exa should port.add_exa add the exa related columns to the output?

        :param portfolio_program:
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
                # TODO, sure this isn't the solution to the double name problem....
                _name = obj.get('name', portfolio_program)
                return Aggregate(_name, **{k: v for k, v in obj.items() if k != 'name'})
            elif _type == 'portfolio':
                return Portfolio(portfolio_program, obj['spec'])
            elif _type == 'severity':
                return Severity(**obj)
            else:
                ValueError(f'Cannot build {_type} objects')
            return obj

        # run
        self._runner(portfolio_program)

        # what / how to do; little awkward: to make easier for user have to strip named update args
        # out of kwargs
        update = kwargs.get('update', self.update)
        if update:
            if 'log2' in kwargs:
                log2 = kwargs.get('log2')
                del kwargs['log2']
            else:
                log2 = self.log2
            if 'bs' in kwargs:
                bs = kwargs.get('bs')
                del kwargs['bs']
            else:
                bs = 0
            if 'verbose' in kwargs:
                verbose = kwargs.get('verbose')
                del kwargs['verbose']
            else:
                verbose = self.verbose
            if 'add_exa' in kwargs:
                add_exa = kwargs.get('add_exa')
                del kwargs['add_exa']
            else:
                add_exa = False

        # what shall we create? only create if there is one item  port then agg then sev, create in rv
        rv = None
        if len(self.parser.port_out_dict) > 0:
            # create ports
            rv = []
            for k in self.parser.port_out_dict.keys():
                s = Portfolio(k, self.portfolio[k]['spec'])
                if update:
                    if bs > 0 and log2 > 0:
                        _bs = bs
                        _log2 = log2
                    else:
                        if bs == 0:  # and log2 > 0
                            # for log2 = 10
                            _bs = s.recommend_bucket().iloc[-1, 0]
                            _log2 = log2  # which must be > 0
                            # adjust bucket size for new actual log2
                            _bs *= 2 ** (10 - _log2)
                        else:  # bs > 0 and log2 = 0 which doesn't really make sense...
                            logging.warning('Underwriter.write | nonsensical options bs > 0 and log2 = 0')
                            _bs = bs
                            _log2 = 10
                    logging.info(f"Underwriter.write | updating Portfolio {k} log2={_log2}, bs={_bs}")
                    s.update(log2=_log2, bs=_bs, verbose=verbose, add_exa=add_exa, **kwargs)
                rv.append(s)

        elif len(self.parser.agg_out_dict) > 0 and rv is None:
            # new aggs, create them
            rv = []
            for k, v in self.parser.agg_out_dict.items():
                # TODO FIX this clusterfuck
                s = Aggregate(k, **{kk: vv for kk, vv in v.items() if kk != 'name'})
                if update:
                    s.easy_update(self.log2)
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
            logging.warning(f'Underwriter.write | Program {portfolio_program} did not contain any output')
        if rv is not None and len(rv) == 1:
            rv = rv[0]
        return rv

    def write_from_file(self, file_name, **kwargs):
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
        return self.write(portfolio_program, **kwargs)

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
            backslash (line continuation) mapped to space
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
    def obj_to_agg(obj):
        """
        convert an object into an agg language specification, used for saving
        :param obj: a dictionary, Aggregate, Severity or Portfolio object
        :return:
        """
        pass


# def dict_2_string(type_name, dict_in, tab_level=0, sio=None):
#     """
#     nice formating for str function
#
#     :param type_name:
#     :param dict_in:
#     :param tab_level:
#     :param sio:
#     :return:
#     """
#
#     if sio is None:
#         sio = StringIO()
#
#     keys = sorted(dict_in.keys())
#     if 'name' in keys:
#         # which it should always be
#         nm = dict_in['name']
#         sio.write(nm + '\n' + '=' * len(nm) + '\n')
#         keys.pop(keys.index('name'))
#
#     sio.write(f'{"type":<20s}{type_name}\n')
#
#     for k in keys:
#         v = dict_in[k]
#         ks = '\t' * max(0, tab_level - 1) + f'{str(k):<20s}'
#         if type(v) == dict:
#             # sio.write('\t' * tab_level + ks + '\n')
#             dict_2_string(type(v), v, tab_level + 1, sio)
#         elif isinstance(v, str):
#             if len(v) > 30:
#                 sio.write('\t' * tab_level + ks + '\n' + indent(fill(v, 30), ' ' * (4 * tab_level + 20)))
#             elif len(v) > 0:
#                 sio.write('\t' * tab_level + ks + v)
#             sio.write('\n')
#         elif isinstance(v, collections.Iterable):
#             sio.write('\t' * tab_level + ks + '\n')
#             for vv in v:
#                 sio.write('\t' * (tab_level + 1) + str(vv) + '\n')
#         elif type(v) == int:
#             sio.write('\t' * tab_level + f'{ks}\t{v:20d}\n')
#         elif type(v) == float:
#             if abs(v) < 100:
#                 sio.write('\t' * tab_level + f'{ks}\t{v:20.5f}\n')
#             else:
#                 sio.write('\t' * tab_level + f'{ks}\t{v:20,.1f}\n')
#         else:
#             # logging.info(f'Uknown type {type(v)} to dict_2_string')
#             sio.write('\t' * tab_level + ks + '\t' + str(v) + '\n')
#     return sio.getvalue()
