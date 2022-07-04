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

import numpy as np
import logging
import pandas as pd
from pathlib import Path
from inspect import signature

from .port import Portfolio
from .distr import Aggregate, Severity
from .parser import UnderwritingLexer, UnderwritingParser

logger = logging.getLogger(__name__)


class Underwriter(object):
    """
    The ``Underwriter`` class manages the creation of Aggregate and Portfolio objects, and
    maintains a database of standard Severity (curves) and Aggregate (unit or line level) objects.
    The ``Underwriter`` knows about all the business that is written!

    * Handles persistence to and from agg files
    * Is interface into program parser
    * Handles safe lookup from database for parser

    Objects have a kind (agg, port, or sev) and a name. E.g. agg MyAgg ... has kind agg and name MyAgg.
    They have a representation as a program. When the program is interpreted it produces a string spec
    that can be used to create the object. The static method factory can create any object from the
    (kind, name, spec, program) quartet, though, strictly, program is not needed.

    The underwriter knowledge is stored in a dataframe indexed by kind and name with columns
    spec and program.

    """

    def __init__(self, databases=None, store_mode=True, update=False,
                 log2=10, debug=False, create_all=False):
        """

        :param dir_name:
        :param name:
        :param databases: if None: nothing loaded; if 'default' or 'default' in databases load the installed
        databases; if 'site' or site in databases, load all site databases (home() / agg, which
        is created it it does not exist); other entires treated as db names in home() / agg are then loaded.
        Databases not in site directory must be fully qualified path names.
        :param store_mode: add newly created aggregates to the uw knowledge?
        :param update:
        :param log2:
        :param debug: run parser in debug mode
        :param create_all: by default write only creates portfolios.
        """

        self.update = update
        if log2 <= 0:
            raise ValueError('log2 must be > 0. The number of buckets used equals 2**log2.')
        self.log2 = log2
        self.debug = debug
        self.lexer = UnderwritingLexer()
        self.parser = UnderwritingParser(self.safe_lookup, debug)
        # stop pyCharm complaining
        # knowledge - accounts and line known to the underwriter
        self._knowledge = pd.DataFrame(columns=['kind', 'name', 'spec', 'program'], dtype=object).set_index(
            ['kind', 'name'])
        if databases == 'all':
            databases = ['default', 'site']
        elif type(databases) == str:
            databases = [databases]
        default_dir = Path(__file__).parent / 'agg'
        site_dir = Path.home() / 'aggregate'

        # TODO Useful?
        self.dir_name = site_dir
        if self.dir_name.exists() is False:
            # check site dir exists
            self.dir_name.mkdir(parents=True, exist_ok=True)

        # make sure all database entries are stored:
        self.store_mode = True
        if databases is None:
            # nothing to do
            databases = []

        if 'default' in databases:
            databases.remove('default')
            for fn in default_dir.glob('*.agg'):
                self._read_db(fn)

        if 'site' in databases:
            databases.remove('site')
            databases += list(site_dir.glob('*.agg'))

        for fn in databases:
            if Path(fn).exists():
                self._read_db(fn)
            elif (site_dir / fn).exists():
                self._read_db(site_dir / fn)
            else:
                logger.warning(f'Database {fn} not found. Ignoring.')

        # set desired store_mode
        self.store_mode = store_mode
        self.create_all = create_all

    def _read_db(self, db_path):
        if not isinstance(db_path, Path):
            db_path = Path(db_path)
        try:
            program = db_path.read_test(encoding='utf-8')
        except FileNotFoundError:
            logger.warning(f'Requested database {db_path.name} not found. Ignoring.')
        # read in, parse, save to sev/agg/port dictionaries
        self.interpret_program(program)

    def __getitem__(self, item):
        """
        handles self[item]

        item = 'Name' for all objects called Name
        item = ("kind", "name") for object of kind kind called name

        subscriptable: try user portfolios, b/in portfolios, line, severity
        to access specifically use severity or line methods

        :param item:
        :return:
        """
        # much less fancy version:
        if not isinstance(item, (str, tuple)):
            raise ValueError(f'item must be a str (name of object) or tuple (kind, name), not {type(item)}.')

        try:
            if type(item) == str:
                rows = self._knowledge.xs(item, axis=0, level=1, drop_level=False)
            elif type(item) == tuple:
                # return a dataframe
                rows = self.loc[[item]]
        except KeyError:
            raise KeyError(f'Item {item} not found.')
        except TypeError:
            # TODO fix this "TypeError: unhashable type: 'slice'"
            logger.error(f'getitem type error issue....looking for {item}')
        else:
            if len(rows) == 1:
                kind, name, spec, prog = rows.reset_index().iloc[0]
                return kind, name, spec, prog
            else:
                raise KeyError(f'Multiple objects matching {item} found. Returning them all.')

    def _repr_html_(self):
        s = [f'<h1>Underwriter Knowledge</h1>',
             f'The underwriter knowledge contains {len(self._knowledge)} items.'
             '<br>',
             self.knowledge.to_html(),
             f'<h3>Settings</h3>']
        # s.append(', '.join([f'{n} ({k})' for (k, n), v in
        #                     sorted(self._knowledge.items())]))
        for k in ['log2', 'update', 'store_mode', 'create_all']:
            s.append(f'<span style="color: red;">{k}</span>: {getattr(self, k)}; ')
        return '\n'.join(s)

    def __call__(self, portfolio_program, **kwargs):
        """
        make the Underwriter object callable; pass through to write

        :param portfolio_program:
        :return:
        """
        return self.write(portfolio_program, **kwargs)

    def factory(self, kind, name, spec, program):
        """
        Create object of kind from spec, a dictionary.
        Creating from uw obviously needs the uw, so this is NOT a staticmethod!

        :param kind:
        :param name:
        :param spec:
        :return:
        """

        if kind == 'agg':
            obj = Aggregate(**spec)
            obj.program = program
        elif kind == 'port':
            # actually make the object; spec = list of aggs
            # print(f'Factory name={name}, spec={spec}')
            agg_list = spec['spec'] # [self[v][1] for v in spec['spec']]
            # print(agg_list)
            obj = Portfolio(name, agg_list, uw=self)
            # # actually make the object previous code...
            # obj = Portfolio(portfolio_program, [self[v][1] for v in obj['spec']])
        elif kind == 'sev':
            if 'sev_wt' in spec:
                spec = spec.copy()
                del spec['sev_wt']
            obj = Severity(**spec)
        else:
            ValueError(f'Cannot build {kind} objects')
        return obj

    @property
    def knowledge(self):
        """
        Return the knowledge as a nice dataframe

        :return:
        """
        return self._knowledge[['program']].style.set_table_styles([
            {
                'selector': 'td',
                'props': 'text-align: left'},
            {
                'selector': 'th.col_heading',
                'props': 'text-align: center;'
            },
            {
                'selector': '.row_heading',
                'props': 'text-align: left;'
            }
        ])

    def list(self):
        """
        List summary of the knowledge

        :return:
        """
        raise NotImplementedError('NYI')
        return
        # sers = dict()
        # for k in Underwriter.data_types:
        #     # d = sorted(list(self.__getattribute__(k).keys()))
        #     d = sorted(list(getattr(self, k).keys()))
        #     sers[k.title()] = pd.Series(d, index=range(len(d)), name=k)
        # df = pd.DataFrame(data=sers)
        # df = df.fillna('')
        # return df

    def describe(self, kinds=None):
        """
        More informative version of list including notes.

        TODO: enhance!

        :return:
        """

        cols = ['Name', 'Type', 'Severity', 'ESev', 'Sev_a', 'Sev_b', 'EN', 'Freq_a', 'ELoss', 'Notes']
        # what they are actually called
        cols_agg = ['sev_name', 'sev_mean', 'sev_a', 'sev_b', 'exp_en', 'freq_a', 'exp_el', 'note']
        defaults = ['', 0, 0, 0, 0, 0, 0, '']
        df = pd.DataFrame(columns=cols)
        df = df.set_index('Name')

        for (kind, name), (spec, program) in self._knowledge.iterrows():
            pass
            # for obj_name, obj_values in getattr(self, item_type).items():
            #     data_fields = [obj_values.get(c, d) for c, d in zip(cols_agg, defaults)]
            #     df.loc[obj_name, :] = [item_type] + data_fields
        return None
        # df = df.fillna('')
        # return df

    def write(self, portfolio_program, log2=0, bs=0, **kwargs):
        """
        Write a natural language program. Write carries out the following steps.

        1. Read in the program and cleans it (e.g. punctuation, parens etc. are
        removed and ignored, replace ; with new line etc.)
        2. Parse line by line to create a dictioonary definition of sev, agg or port objects
        3. replace sev.name, agg.name and port.name references with their objects
        4. If create_all set, create all objects and return in dictionary. If not set only create the port objects
        5. If update set, update all created objects.

        Sample input

        ::

            port MY_PORTFOLIO
                agg Line1 20  loss 3 x 2 sev gamma 5 cv 0.30 mixed gamma 0.4
                agg Line2 10  claims 3 x 2 sevgamma 12 cv 0.30 mixed gamma 1.2
                agg Line 3100  premium at 0.4 3 x 2 sev 4 @ lognormal 3 cv 0.8 fixed 1

        The indents are required...

        See parser for full language spec! See Aggregate class for many examples.

        Reasonable kwargs:

        * bs
        * log2
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
        if 'create_all' in kwargs:
            create_all = kwargs['create_all']
            del kwargs['create_all']
        else:
            create_all = self.create_all

        if 'update' in kwargs:
            update = kwargs['update']
            del kwargs['update']
        else:
            update = self.update

        if update is True and log2 == 0:
            log2 = self.log2

        # first see if portfolio_program refers to a built-in object
        try:
            kind, name, spec, program = self[portfolio_program]  # calls __getitem__
        except (LookupError, TypeError):
            logger.debug(f'underwriter.write | object not found, processing as a program.')
        else:
            logger.debug(f'underwriter.write | {kind} object found.')
            obj = self.factory(kind, name, spec, program)
            if update:
                obj.update(log2, bs, **kwargs)
            return obj

        # if you fall through to here then the portfolio_program did not refer to a built-in object
        # run the program, get the interpreter return value, irv which contains info about all
        # the objects in the program, including aggs within ports
        # kind0, name0 is the answer exit point object - this must always be created
        irv = self.interpret_program(portfolio_program)

        # create objects
        # 2019-11: create all objects not just the portfolios if create_all==True
        # rv = return values
        rv = None
        if len(irv) > 0:
            # create ports
            rv = {}
            # parser.out_dict is indexed by (kind, name) and contains the defining dictionary
            # PrettyPrinter().pprint(self.parser.out_dict)
            for (kind, name), (spec, program) in irv.items():
                # remember the spec comes back as a list of aggs that have been entered into the uw
                if create_all or program != '':
                    obj = self.factory(kind, name, spec, program)
                    if update:
                        obj.update(log2, bs, **kwargs)
                    rv[(kind, name)] = (obj, program)
                else:
                    rv[(kind, name)] = (spec, program)

        # report on what has been done
        if rv is None:
            # print('WARNING: Program did not contain any output...')
            logger.warning(f'Underwriter.write | Program did not contain any output')
        else:
            if len(rv):
                logger.info(f'Underwriter.write | Program created {len(rv)} objects.')
            # handle this in build
            # if len(rv) == 1:
            #     # dict, pop the last (only) element
            #     rv = rv.popitem()[1]

        # return created objects
        return rv

    def write_from_file(self, file_name, log2=0, bs=0, update=False, **kwargs):
        """
        read program from file. delegates to write

        :param file_name:
        :param log2:
        :param bs:
        :param update:
        :param kwargs:
        :return:
        """
        portfolio_program = Path(file_name).read_text(encoding='utf-8')
        return self.write(portfolio_program, log2=log2, bs=bs, update=update, **kwargs)

    def interpret_program(self, portfolio_program):
        """
        Preprocess and then parse one line at a time.

        Error handling through parser.

        Old parse_portfolio_program replaced with build.interpret_one and interpret_test,
        and running write with

        :param portfolio_program:
        :return:
        """

        # Preprocess ---------------------------------------------------------------------
        portfolio_program = UnderwritingLexer.preprocess(portfolio_program)

        # Parse and Postprocess-----------------------------------------------------------
        self.parser.reset()
        program_line_dict = {}
        for program_line in portfolio_program:
            logger.debug(program_line)
            # preprocessor only returns lines of length > 0
            try:
                # parser returns the type and name of the object
                kind0, name0 = self.parser.parse(self.lexer.tokenize(program_line))
            except ValueError as e:
                if isinstance(e.args[0], str):
                    logger.error(e)
                    raise e
                else:
                    t = e.args[0].type
                    v = e.args[0].value
                    i = e.args[0].index
                    txt2 = program_line[0:i] + f'>>>' + program_line[i:]
                    logger.error(f'Parse error in input "{txt2}"\nValue {v} of type {t} not expected')
                    raise e
            else:
                logger.info(f'answer out: {kind0} object {name0} parsed successfully')
                program_line_dict[(kind0, name0)] = program_line

        # now go through self.parser.out_dict and store everything
        # if in store_mode, add the program to uw dictionaries
        # virtually nothing works if you don't store stuff...
        # TODO what is use case for store_mode = False?!
        rv = {}
        if self.store_mode is True and len(self.parser.out_dict) > 0:
            logger.info(f'Adding {len(self.parser.out_dict)} specs to the knowledge.')
            for (kind, name), spec in self.parser.out_dict.items():
                # will not know for aggs within port, for example
                # when create_all is false, only create thigns with a program_line != ''
                program_line = program_line_dict.get((kind, name), '')
                self._knowledge.loc[(kind, name), :] = [spec, program_line]
                logger.info(f'Added {kind}, {name}, {program_line[:20]} to knowledge.')
                rv[(kind, name)] = (self.parser.out_dict[(kind, name)], program_line)
        return rv

    @staticmethod
    def add_defaults(dict_in, kind='agg'):
        """
        add default values to dict_inin. Leave existing values unchanged
        Used to output to a data frame, where you want all columns completed

        :param dict_in:
        :param kind:
        :return:
        """

        print('running add_defaults\n' * 10)

        # use inspect to get the defaults
        # obtain signature
        sig = signature(Aggregate.__init__)

        # self and name --> bound signature
        bs = sig.bind(None, '')
        bs.apply_defaults()
        # remove self
        bs.arguments.pop('self')
        defaults = bs.arguments

        if kind == 'agg':
            defaults.update(dict_in)

        elif kind == 'sev':
            for k, v in defaults.items():
                if k[0:3] == 'sev' and k not in dict_in and k != 'sev_wt':
                    dict_in[k] = v

    def safe_lookup(self, buildinid):
        """
        Lookup buildinid=kind.name in uw to find expected kind and merge safely into self.arg_dict.

        Different from getitem because it splits the item into kind and name and double
        checks you get the expected kind.

        :param buildinid:  a string in kind.name format
        :return:
        """

        kind, name = buildinid.split('.')
        try:
            # lookup in Underwriter
            found_kind, found_name, spec, program = self[name]
        except LookupError as e:
            logger.error(f'ERROR id {kind}.{name} not found in the knowledge.')
            raise e
        logger.debug(f'UnderwritingParser.safe_lookup | retrieved {kind}.{name} as type {found_kind}.{found_name}')
        if found_kind != kind:
            raise ValueError(f'Error: type of {name} is  {found_kind}, not expected {kind}')
        return spec


class Build(object):
    """
    The Build class wraps the Underwriter to provide a more elementary interface. Intelligent auto-update,
    detects discrete distributions and sets ``bs = 1``.

    """
    # set overall debug mode, for parser and port, sev and aggregate object updates
    DEBUG = False
    uw = Underwriter(create_all=False, update=True, debug=DEBUG)

    @classmethod
    def parse(cls, program):
        return cls.uw.parse_portfolio_program(program)

    @classmethod
    def list(cls):
        return cls.uw.list()

    @classmethod
    def describe(cls, item_type=''):
        return cls.uw.describe(item_type, pretty_print=True)

    @classmethod
    def build(cls, program, update=True, create_all=None, log2=-1, bs=0, **kwargs):
        """
        Convenience function to make work easy for the user. Hide uw, updating etc. A single instance, called
        ``build`` is exported from the module.

        :param program:
        :param update: build's update
        :param create_all: for just this run
        :param log2: -1 is default. Figure log2 for discrete and 13 for all others. Inupt value over-rides
        and cancels discrete computation (good for large discrete outcomes where bucket happens to be 1.)
        :param bs:
        :param kwargs: passed to update, e.g., padding
        :return: created object(s)
        """

        if program in ['underwriter', 'uw']:
            return cls.uw

        # options for this run
        create_all0 = cls.uw.create_all
        if create_all is not None:
            cls.uw.create_all = create_all

        # make stuff
        # write will return a dict with keys (kind, name) and value either the object or the spec
        out_dict = cls.uw.write(program, update=False)

        # in this loop bs_ and log2_ are the values actually used for each update; they do not
        # overwrite the input default values
        for (kind, name), (out, program) in out_dict.items():
            if isinstance(out, dict):
                # dict spec output, these objects where not created
                logger.info(f'Object {name} of kind {kind} returned as a spec; no further processing.')
            elif isinstance(out, Aggregate) and update is True:
                d = out.spec
                if d['sev_name'] == 'dhistogram' and log2 == -1:
                    bs_ = 1
                    # how big?
                    if d['freq_name'] == 'fixed':
                        max_loss = np.max(d['sev_xs']) * d['exp_en']
                    elif d['freq_name'] == 'empirical':
                        max_loss = np.max(d['sev_xs']) * max(d['freq_a'])
                    else:
                        # normal approx on count
                        max_loss = np.max(d['sev_xs']) * d['exp_en'] * (1 + 3 * d['exp_en']**0.5)
                    # binaries are 0b111... len-2 * 2 is len - 1
                    log2_ = len(bin(int(max_loss))) - 1
                    logger.info(f'({kind}, {name}): Discrete mode, using bs=1 and log2={log2_}')
                else:
                    if log2 == -1:
                        log2_ = 13
                    else:
                        log2_ = log2
                    if bs == 0:
                        bs_ = out.recommend_bucket(log2_)
                    else:
                        bs_ = bs
                    logger.info(f'({kind}, {name}): Normal mode, using bs={bs_} and log2={log2_}')
                try:
                    out.update(log2=log2_, bs=bs_, debug=cls.DEBUG, **kwargs)
                except ZeroDivisionError as e:
                    logger.error(e)
                except AttributeError as e:
                    logger.error(e)
            elif isinstance(out, Severity):
                # there is no updating for severities
                pass
            elif isinstance(out, Portfolio) and update is True:
                # figure stuff
                if log2 == -1:
                    log2_ = 13
                else:
                    log2_ = log2
                if bs == 0:
                    bs_ = out.best_bucket(log2_)
                else:
                    bs_ = bs
                print(f'updating with {log2}, bs=1/{1/bs_}')
                logger.info(f'({kind}, {name}): bs={bs_} and log2={log2_}')
                out.update(log2=log2_, bs=bs_, remove_fuzz=True, debug=cls.DEBUG, **kwargs)
            else:
                logger.warning(f'WTF??? output kind is {type(out)}. (expr/number?)')
                pass

        # put back defaults
        cls.uw.create_all = create_all0
        if len(out_dict) == 1:
            # only one output...just return that
            # dict, pop the last (only) element (popitem: Remove and return a (key, value) pair as a 2-tuple.)
            out_dict = out_dict.popitem()[1]
            if len(out_dict) == 2:
                out_dict = out_dict[0]
            else:
                print('WTF, investigate...'*5)

        return out_dict

    __call__ = build

    @classmethod
    def interpreter_file(cls, where='',
                         filename='C:\\S\\TELOS\\Python\\aggregate_extensions_project\\aggregate2\\agg2_database.csv'):
        """
        Run a suite of test programs. For detailed analysis, run_one.

        """
        df = pd.read_csv(filename, index_col=0)
        if where != '':
            df = df.loc[df.index.str.match(where)]
        return cls.interpreter_work(df.iterrows()), df

    @classmethod
    def interpreter_one(cls, program):
        """
        Interpret single test in debug mode.
        """

        return cls.interpreter_work(iter([('one off', program)]), debug=True)

    @classmethod
    def interpreter_list(cls, program_list):
        """
        Interpret single test in debug mode.
        """
        return cls.interpreter_work(list(enumerate(program_list)), debug=True)

    @classmethod
    def interpreter_work(cls, iterable, debug=False):
        """
        do all the work for the est, allows into to be marshalled into the tester
        in different ways.

        :return:
        """
        lexer = UnderwritingLexer()
        parser = UnderwritingParser(cls.uw.safe_lookup, debug)
        ans = {}
        errs = 0
        no_errs = 0
        # detect non-trivial change
        f = lambda x, y: '' if x.replace(' ', '') == y.replace(' ', '').replace('\t', '') else y
        for test_name, program in iterable:
            parser.reset()
            if type(program) != str:
                program_in = program[0]
                program = lexer.preprocess(program_in)
            else:
                program_in = program
                program = lexer.preprocess(program_in)
            err = 0
            if len(program) == 1:
                program = program[0]
                try:
                    # print(program)
                    kind, name = parser.parse(lexer.tokenize(program))
                except (ValueError, TypeError) as e:
                    errs += 1
                    err = 1
                    name = test_name
                    kind = program.split()[0]
                    parser.out_dict[(kind, name)] = str(e)
                else:
                    no_errs += 1
                ans[test_name] = [kind, err, name,
                                  parser.out_dict[(kind, name)] if kind != 'expr' else None,
                                  program, f(program, program_in)]
            elif len(program) > 1:
                logger.info(f'{program_in} preprocesses to {len(program)} lines; not processing.')
                ans[test_name] = ['multiline', err, None, None, program, program_in]
            else:
                logger.info(f'{program_in} preprocesses to a blank line; ignoring.')
                ans[test_name] = ['blank', err, None, None, program, program_in]

        # print((f'No errors reported.\n' if errs == 0 else f'{errs} errors reported.\n') +
        #       f'{no_errs} programs parsed successfully.')
        df_out = pd.DataFrame(ans,
                              index=['type', 'error', 'name', 'interpreted', 'program',
                                     'raw_input']).T.sort_values('error', ascending=False)
        return df_out


# exported instance
build = Build()
