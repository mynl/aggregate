"""
=================
Underwriter Class
=================

The Underwriter is an easy to use interface into the computational functionality of aggregate.

The Underwriter
---------------

* Maintains a default library of severity curves
* Maintains a default library of aggregate distributions corresponding to industry losses in
  major classes of business, total catastrophe losses from major perils, and other useful constructs
* Maintains a default library of portfolios, including several example instances and examples used in
  papers on risk theory (e.g. the Bodoff examples)


The library functions can be listed using

::

        uw.list()

or, for more detail

::

        uw.describe()

A given example can be inspected using ``uw['cmp']`` which returns the defintion of the database
object cmp (an aggregate representing industry losses from the line Commercial Multiperil). It can
be created as an Aggregate class using ``ag = uw('cmp')``. The Aggregate class can then be updated,
plotted and various reports run on it. In iPython or Jupyter ``ag`` returns an informative HTML
description.

The real power of Underwriter is access to the agg scripting language (see parser module). The scripting
language allows severities, aggregates and portfolios to be created using more-or-less natural language.
For example

::

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

+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
|       | bs10    | bs11   | bs12   | bs13   | bs14  | bs15  | bs16  | bs18 | bs20 |
+=======+=========+========+========+========+=======+=======+=======+======+======+
| LineA | 3,903   | 1,951  | 976    | 488    | 244   | 122   | 61.0  | 15.2 | 3.8  |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
| LineB | 8,983   | 4,491  | 2,245  | 1,122  | 561   | 280   | 140   | 35.1 | 8.8  |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
| Cat   | 97,656  | 48,828 | 24,414 | 12,207 | 6,103 | 3,051 | 1,525 | 381  | 95.4 |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
| total | 110,543 | 55,271 | 27,635 | 13,817 | 6,908 | 3,454 | 1,727 | 431  | 108  |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+

The column bsNcorrespond to discretizing with 2**N buckets. The rows show suggested bucket sizes for each
line and in total. For example with N=13 (i.e. 8196 buckets) the suggestion is 13817. It is best the bucket
size is a divisor of any limits or attachment points, so we select 10000.

Updating can then be run as

::

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
import numpy as np
from IPython.core.display import display
import logging
import pandas as pd
from collections import Iterable
from .port import Portfolio
from .utils import html_title
from .distr import Aggregate, Severity
from .parser import UnderwritingLexer, UnderwritingParser
import re
import warnings

logger = logging.getLogger('aggregate')

class Underwriter(object):
    """
    The underwriter class constructs real world examples from stored and user input Lines and Accounts.
    Whereas Examples only produces simple Portfolios and Books, the Underwriter class is more flexible.

    Handles persistence
    Is interface into program parser
    Handles safe lookup from database for parser

    Persisitence to and from agg files

    """

    data_types = ['portfolio', 'aggregate', 'severity']

    def __init__(self, dir_name="", name='Rory', databases=[], glob=None, store_mode=True, update=False,
                 verbose=False, log2=10, debug=False, create_all=False):
        """

        :param dir_name:
        :param name:
        :param databases: set equal to None to load the default databases. Faster to load without them.
        :param glob: reference, e.g. to globals(), used to resolve meta.XX references
        :param store_mode: add newly created aggregates to the database?
        :param update:
        :param verbose:
        :param log2:
        :param debug: run parser in debug mode
        :param create_all: by default write only creates portfolios.
        """

        self.last_spec = None
        self.name = name
        self.update = update
        self.log2 = log2
        self.debug = debug
        self.verbose = verbose  # for update
        self.glob = glob
        self.lexer = UnderwritingLexer()
        self.parser = UnderwritingParser(self._safe_lookup, debug)
        # otherwise these are hidden from pyCharm....
        self.severity = {}
        self.aggregate = {}
        self.portfolio = {}
        if databases is None:
            databases = ['site.agg', 'user.agg']
        self.dir_name = dir_name
        if self.dir_name == '':
            self.dir_name = os.path.split(__file__)[0]
            self.dir_name = os.path.join(self.dir_name, 'agg')
        # make sure all database entries are stored:
        self.store_mode = True
        for fn in databases:
            with open(os.path.join(self.dir_name, fn), 'r') as f:
                program = f.read()
            # read in, parse, save to sev/agg/port dictionaries
            self._runner(program)
        # set desired store_mode
        self.store_mode = store_mode
        self.create_all = create_all

    def __getitem__(self, item):
        """
        handles self[item]

        subscriptable: try user portfolios, b/in portfolios, line, severity
        to access specifically use severity or line methods

        :param item:
        :return:
        """
        # much less fancy version:
        obj = self.portfolio.get(item, None)
        if obj is not None:
            logger.info(f'Underwriter.__getitem__ | found {item} of type port')
            return 'port', obj
        obj = self.aggregate.get(item, None)
        if obj is not None:
            logger.info(f'Underwriter.__getitem__ | found {item} of type agg')
            return 'agg', obj
        obj = self.severity.get(item, None)
        if obj is not None:
            logger.info(f'Underwriter.__getitem__ | found {item} of type sev')
            return 'sev', obj
        raise LookupError(f'Item {item} not found in any database')
        # old, clever, generic, unreadable, unnecessary generality
        # for k in self.databases.keys():
        #     if item in self.__getattribute__(k).keys():
        #         # stip the s off the name: Books to Book etc.
        #         return k, self.__getattribute__(k)[item]
        # raise LookupError

    def _repr_html_(self):
        s = [f'<h1>Underwriter {self.name}</h1>']
        s.append(
            f'Underwriter expert in all classes including {len(self.severity)} severities, {len(self.aggregate)} aggregates'
            f' and {len(self.portfolio)} portfolios<br>')
        for what in ['severity', 'aggregate', 'portfolio']:
            s.append(f'<b>{what.title()}</b>: ')
            s.append(', '.join([k for k in sorted(getattr(self, what).keys())]))
            s.append('<br>')
        s.append(f'<h3>Settings</h3>')
        for k in ['update', 'log2', 'store_mode', 'verbose', 'last_spec', 'create_all']:
            s.append(f'<span style="color: red;">{k}</span>: {getattr(self, k)}; ')
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

    def __call__(self, portfolio_program, **kwargs):
        """
        make the Underwriter object callable; pass through to write

        :param portfolio_program:
        :return:
        """
        return self.write(portfolio_program, **kwargs)

    def list(self):
        """
        list all available databases

        :return:
        """
        sers = dict()
        for k in Underwriter.data_types:
            d = sorted(list(self.__getattribute__(k).keys()))
            sers[k.title()] = pd.Series(d, index=range(len(d)), name=k)
        df = pd.DataFrame(data=sers)
        # df.index.name = 'No.'
        df = df.fillna('')
        return df

    def describe(self, item_type='', pretty_print=False):
        """
        more informative version of list
        Pull notes for type items

        :return:
        """

        def deal_with_sequences(x):
            """
            pandas can't have a field set as a sequence
            need to check if x is a sequence and if so return something suitable...

            :param x:
            :return:
            """
            if isinstance(x, Iterable):
                return str(x)
            else:
                return x

        df = pd.DataFrame(columns=['Name', 'Type', 'Severity', 'ESev', 'Sev_a', 'Sev_b',
                                   'EN', 'Freq_a',
                                   'ELoss', 'Notes'])
        df = df.set_index('Name')
        df['ELoss'] = np.maximum(df.ELoss, df.ESev * df.EN)
        if item_type == '':
            item_type = Underwriter.data_types
        else:
            item_type = [item_type.lower()]
        for k in item_type:  # self.databases.keys():
            for kk, vv in self.__getattribute__(k).items():
                _data_fields = [vv.get('sev_name', ''), vv.get('sev_mean', 0), vv.get('sev_a', 0),
                                vv.get('sev_b', 0), vv.get('exp_en', 0), vv.get('freq_a', 0),
                                vv.get('exp_el', 0), vv.get('note', '')]
                try:
                    df.loc[kk, :] = [k] + _data_fields
                except ValueError as e:
                    if e.args[0] == "setting an array element with a sequence":
                        df.loc[kk, :] = [k] + list(map(deal_with_sequences, _data_fields))
                    else:
                        raise e
        df = df.fillna('')
        if pretty_print:
            for t, egs in df.groupby('Type'):
                html_title(t, 2)
                display(egs.style)
        return df

    def parse_portfolio_program(self, portfolio_program, output='spec'):
        """
        Utility routine to parse the program and return the spec suitable to pass to Portfolio to
        create the object.
        Initially just for a single portfolio program (which it checks!)
        No argument of default conniptions

        To write program in testing mode use output='df':

        * dictionary definitions are added to uw but no objects are created
        * returns data frame description of added severity/aggregate/portfolios
        * the dataframe of aggregates can be used to create a portfolio (with all the aggregates) by calling

        ```Portfolio.from_DataFrame(name df)```

        To parse and get dictionary definitions use output='spec'.
        Aggregate and severity objects are also returned though they could be
        accessed directly using wu['name']. May be convenient...we'll see.

        Output has form that an Aggregate can be created from Aggregate(**x['name'])
        etc. which is a bit easier than uw['name'] which returns the type.

        TODO make more robust

        :param portfolio_program:
        :param output:  'spec' output a spec (assumes only one portfolio),
                        or a dictionary {name: spec_list} if multiple
                        'df' or 'dataframe' output as pandas data frame
                        'dict' output as dictionary of pandas data frames (old write_test output)
        :return:
        """

        self._runner(portfolio_program)

        # if globs replace all meta objects with a lookup object
        # copy from code below FRAGILE
        if self.glob is not None:
            for a in list(self.parser.agg_out_dict.values()) + list(self.parser.sev_out_dict.values()):
                if a['sev_name'][0:4] == 'meta':
                    obj_name = a['sev_name'][5:]
                    try:
                        obj = self.glob[obj_name]
                    except NameError as e:
                        print(f'Object {obj_name} passed as a proto-severity cannot be found')
                        raise e
                    a['sev_name'] = obj
                    logger.info(f'Underwriter.write | {a["sev_name"]} ({type(a)} reference to {obj_name} '
                                 f'replaced with object {obj.name} from glob')

        if output == 'spec':
            # expecting a single portfolio for this simple function
            # create the spec list string
            if len(self.parser.port_out_dict) == 1:
                # this behaviour to ensure backwards compatibility
                nm = ""
                spec_list = None
                for nm in self.parser.port_out_dict.keys():
                    # remember the spec comes back as a list of aggs that have been entered into the uw
                    # self[v] = ('agg', dictionary def) of the agg component v of the portfolio
                    spec_list = [self[v][1] for v in self.portfolio[nm]['spec']]
                return nm, spec_list

            elif len(self.parser.port_out_dict) > 1 or \
                    len(self.parser.agg_out_dict) or len(self.parser.sev_out_dict):
                # return dictionary: {pf_name : { name: pf_name, spec_list : [list] }}
                # so that you can call Portfolio(*output[pf_name]) to create pf_name
                # notes are dropped...
                ans = {}
                for nm in self.parser.port_out_dict.keys():
                    # remember the spec comes back as a list of aggs that have been entered into the uw
                    # self[v] = ('agg', dictionary def) of the agg component v of the portfolio
                    spec_list = [self[v][1] for v in self.portfolio[nm]['spec']]
                    ans[nm] = dict(name=nm, spec_list=spec_list)

                for nm in self.parser.agg_out_dict.keys():
                    ans[nm] = self.aggregate[nm]

                for nm in self.parser.sev_out_dict.keys():
                    ans[nm] = self.severity[nm]

                return ans

            else:
                logger.warning(f'Underwriter.parse_portfolio_program | program has no Portfolio outputs. '
                                'Nothing returned. ')
                return

        elif output == 'df' or output.lower() == 'dataframe':
            logger.info(f'Runner.write_test | Executing program\n{portfolio_program[:500]}\n\n')
            ans = {}
            if len(self.parser.sev_out_dict) > 0:
                for v in self.parser.sev_out_dict.values():
                    Underwriter._add_defaults(v, 'sev')
                ans['sev'] = pd.DataFrame(list(self.parser.sev_out_dict.values()),
                                          index=self.parser.sev_out_dict.keys())
            if len(self.parser.agg_out_dict) > 0:
                for v in self.parser.agg_out_dict.values():
                    Underwriter._add_defaults(v)
                ans['agg'] = pd.DataFrame(list(self.parser.agg_out_dict.values()),
                                          index=self.parser.agg_out_dict.keys())
            if len(self.parser.port_out_dict) > 0:
                ans['port'] = pd.DataFrame(list(self.parser.port_out_dict.values()),
                                           index=self.parser.port_out_dict.keys())
            return ans

        else:
            raise ValueError(f'Inadmissible output type {output}  passed to parse_portfolio_program. '
                             'Expecting spec or df/dataframe.')

    def write(self, portfolio_program, **kwargs):
        """
        Write a natural language program. Write carries out the following steps.

        1. Read in the program and cleans it (e.g. punctuation, parens etc. are
        removed and ignored, replace ; with new line etc.)
        2. Parse line by line to create a dictioonary definition of sev, agg or port objects
        3. If glob set, pull in objects
        4. replace sev.name, agg.name and port.name references with their objects
        5. If create_all set, create all objects and return in dictionary. If not set only create the port objects
        6. If update set, update all created objects.

        Sample input

        ::

            port MY_PORTFOLIO
                agg Line1 20  loss 3 x 2 sev gamma 5 cv 0.30 mixed gamma 0.4
                agg Line2 10  claims 3 x 2 sevgamma 12 cv 0.30 mixed gamma 1.2
                agg Line 3100  premium at 0.4 3 x 2 sev 4 * lognormal 3 cv 0.8 fixed 1

        The indents are required...

        See parser for full language spec! See Aggregate class for many examples.

        Reasonable kwargs:

        * bs
        * log2
        * verbose
        * update overrides class default
        * add_exa should port.add_exa add the exa related columns to the output?
        * create_all: create all objects, default just portfolios. You generally
                     don't want to create underlying sevs and aggs in a portfolio.

        :param portfolio_program:
        :param kwargs:
        :return: single created object or dictionary name: object
        """

        # prepare for update
        # what / how to do; little awkward: to make easier for user have to strip named update args
        # out of kwargs
        create_all = kwargs.get('create_all', self.create_all)
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

        # function to handle update madness, use in either script or lookup updats for ports
        def _update(s, k):
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
                        logger.warning('Underwriter.write | nonsensical options bs > 0 and log2 = 0')
                        _bs = bs
                        _log2 = 10
                logger.info(f"Underwriter.write | updating Portfolio {k} log2={_log2}, bs={_bs}")
                s.update(log2=_log2, bs=_bs, verbose=verbose, add_exa=add_exa, **kwargs)

        # first see if it is a built in object
        lookup_success = True
        _type = ''
        obj = None
        try:
            _type, obj = self.__getitem__(portfolio_program)
        except LookupError:
            lookup_success = False
            logger.info(f'underwriter.write | object {portfolio_program[:500]} not found, will process as program')
        if lookup_success:
            logger.info(f'underwriter.write | object {portfolio_program[:500]} found, returning object...')
            if _type == 'agg':
                # TODO, sure this isn't the solution to the double name problem....
                _name = obj.get('name', portfolio_program)
                obj = Aggregate(_name, **{k: v for k, v in obj.items() if k != 'name'})
                if update:
                    obj.easy_update(log2, bs)
                return obj
            elif _type == 'port':
                # actually make the object
                obj = Portfolio(portfolio_program, [self[v][1] for v in obj['spec']])
                _update(obj, portfolio_program)
                return obj
            elif _type == 'sev':
                if 'sev_wt' in obj:
                    del obj['sev_wt']
                return Severity(**obj)
            else:
                ValueError(f'Cannot build {_type} objects')
            return obj

        # if you fall through to here then the portfolio_program did not refer to a built in object
        # run the program
        self._runner(portfolio_program)

        # if globs replace all meta objects with a lookup object
        if self.glob is not None:
            logger.info(f'Underwriter.write | Resolving globals')
            for a in list(self.parser.agg_out_dict.values()) + list(self.parser.sev_out_dict.values()):
                if a['sev_name'][0:4] == 'meta':
                    logger.info(f'Underwriter.write | Resolving {a["sev_name"]}')
                    obj_name = a['sev_name'][5:]
                    try:
                        obj = self.glob[obj_name]
                    except NameError as e:
                        print(f'Object {obj_name} passed as a proto-severity cannot be found')
                        raise e
                    a['sev_name'] = obj
                    logger.info(f'Underwriter.write | {a["sev_name"]} ({type(a)} reference to {obj_name} '
                                 f'replaced with object {obj.name} from glob')
            logger.info(f'Underwriter.write | Done resolving globals')

        # create objects
        # 2019-11: create all objects not just the portfolios if create_all==True
        rv = None
        if len(self.parser.port_out_dict) > 0:
            # create ports
            rv = {}
            for k in self.parser.port_out_dict.keys():
                # remember the spec comes back as a list of aggs that have been entered into the uw
                s = Portfolio(k, [self[v][1] for v in self.portfolio[k]['spec']])
                s.program = 'unknown'
                _update(s, k)
                rv[k] = s
            if len(self.parser.port_out_dict) == 1:
                # only one portfolio so we can set its program
                s.program = portfolio_program

        if len(self.parser.agg_out_dict) > 0 and create_all:
            # new aggs, create them
            if rv is None:
                rv = {}
            for k, v in self.parser.agg_out_dict.items():
                # TODO FIX this clusterfuck
                s = Aggregate(k, **{kk: vv for kk, vv in v.items() if kk != 'name'})
                if update:
                    s.easy_update(self.log2, verbose=verbose)
                rv[k] = s

        if len(self.parser.sev_out_dict) > 0 and create_all:
            # new sevs, create them
            if rv is None:
                rv = {}
            for v in self.parser.sev_out_dict.values():
                if 'sev_wt' in v:
                    del v['sev_wt']
                s = Severity(**v)
                rv[f'sev_{s.__repr__()[38:54]}'] = s

        # report on what has been done
        if rv is None:
            print('WARNING: Program did not contain any output...')
            logger.warning(f'Underwriter.write | Program {portfolio_program} did not contain any output')
        else:
            if len(rv):
                logger.info(f'Underwriter.write | Program created {len(rv)} objects and '
                             f'defined {len(self.parser.port_out_dict)} Portfolio(s), '
                             f'{len(self.parser.agg_out_dict)} Aggregate(s), and '
                             f'{len(self.parser.sev_out_dict)} Severity(ies)')
            if len(rv) == 1:
                rv = rv.popitem()[1]

        # return created objects
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
        write programs in testing mode

        dictionary definitions are added to uw but no objects are created

        returns data frame description of added severity/aggregate/portfolios

        the dataframe of aggregates can be used to create a portfolio (with all the aggregates) by calling

        ```Portfolio.from_DataFrame(name df)```

        TODO rationalize with parse_portfolio_program

        :param portfolio_program:
        :return: dictionary with keys sev agg port and assoicated dataframes
        """
        print('write_test deprecated...use parse_portfolio_porgram with output="dict".')
        raise RuntimeError
        # TODO once sure you don't need this delete!
        # logger.info(f'Runner.write_test | Executing program\n{portfolio_program[:500]}\n\n')
        # self._runner(portfolio_program)
        # ans = {}
        # if len(self.parser.sev_out_dict) > 0:
        #     for v in self.parser.sev_out_dict.values():
        #         Underwriter._add_defaults(v, 'sev')
        #     ans['sev'] = pd.DataFrame(list(self.parser.sev_out_dict.values()), index=self.parser.sev_out_dict.keys())
        # if len(self.parser.agg_out_dict) > 0:
        #     for v in self.parser.agg_out_dict.values():
        #         Underwriter._add_defaults(v)
        #     ans['agg'] = pd.DataFrame(list(self.parser.agg_out_dict.values()), index=self.parser.agg_out_dict.keys())
        # if len(self.parser.port_out_dict) > 0:
        #     ans['port'] = pd.DataFrame(list(self.parser.port_out_dict.values()), index=self.parser.port_out_dict.keys())
        # return ans

    def _runner(self, portfolio_program):
        """
        preprocessing:
            remove \n in [] (vectors) e.g. put by f{np.linspace} TODO only works for 1d vectors
            ; mapped to newline
            backslash (line continuation) mapped to space
            split on newlines
            parse one line at a time
            PIPE format no longer supported

        error handling and piping through parser

        :param portfolio_program:
        :return:
        """
        # Preprocess ---------------------------------------------------------------------
        # handle \n in vectors; first item is outside, then inside... (multidimensional??)
        # remove coments # xxx
        portfolio_program = re.sub(r'\s*#[^\n]*\n', r'\n', portfolio_program)
        out_in = re.split(r'\[|\]', portfolio_program)
        assert len(out_in) % 2  # must be odd
        odd = [t.replace('\n', ' ') for t in out_in[1::2]]  # replace inside []
        even = out_in[0::2]  # otherwise pass through
        # reassemble
        portfolio_program = ' '.join([even[0]] + [f'[{o}] {e}' for o, e in zip(odd, even[1:])])
        # other preprocessing: line continuation; \n\t or \n____ to space (for port indents),
        # ; to new line, split on new line
        portfolio_program = [i.strip() for i in portfolio_program.replace('\\\n', ' ').
            replace('\n\t', ' ').replace('\n    ', ' ').replace(';', '\n').
            split('\n') if len(i.strip()) > 0]

        # Parse      ---------------------------------------------------------------------
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
                    t = e.args[0].type
                    v = e.args[0].value
                    i = e.args[0].index
                    txt2 = program_line[0:i] + f'>>>' + program_line[i:]
                    print(f'Parse error in input "{txt2}"\nValue {v} of type {t} not expected')
                    raise e

        # Post process -------------------------------------------------------------------
        if self.store_mode:
            # could do this with a loop and getattr but it is too hard to read, so go easy route
            if len(self.parser.sev_out_dict) > 0:
                # for k, v in self.parser.sev_out_dict.items():
                self.severity.update(self.parser.sev_out_dict)  # [k] = v
                logger.info(f'Underwriter._runner | saving {self.parser.sev_out_dict.keys()} severity/ies')
            if len(self.parser.agg_out_dict) > 0:
                # for k, v in self.parser.agg_out_dict.items():
                #     self.aggregate[k] = v
                self.aggregate.update(self.parser.agg_out_dict)
                logger.info(f'Underwriter._runner | saving {self.parser.agg_out_dict.keys()} aggregate(s)')
            if len(self.parser.port_out_dict) > 0:
                for k, v in self.parser.port_out_dict.items():
                    # v is a list of aggregate names, these have all been added to the database...
                    logger.info(f'Underwriter._runner | saving {k} portfolio')
                    self.portfolio[k] = {'spec': v['spec'], 'arg_dict': {}}
                    # self.portfolio[k] = {'spec': [self.aggregate[_a] for _a in v], 'arg_dict': {}}
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

    def _safe_lookup(self, full_uw_id):
        """
        lookup uw_id in uw of expected type and merge safely into self.arg_dict
        delete name and note if appropriate

        :param full_uw_id:  type.name format
        :return:
        """

        expected_type, uw_id = full_uw_id.split('.')
        try:
            # lookup in Underwriter
            found_type, found_dict = self[uw_id]
        except LookupError as e:
            print(f'ERROR id {expected_type}.{uw_id} not found')
            raise e
        logger.info(f'UnderwritingParser._safe_lookup | retrieved {uw_id} as type {found_type}')
        if found_type != expected_type:
            raise ValueError(f'Error: type of {uw_id} is  {found_type}, not expected {expected_type}')
        return found_dict.copy()

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
#             # logger.info(f'Uknown type {type(v)} to dict_2_string')
#             sio.write('\t' * tab_level + ks + '\t' + str(v) + '\n')
#     return sio.getvalue()
