import numpy as np
import logging
import pandas as pd
from pathlib import Path
# from inspect import signature

from .portfolio import Portfolio
from .distributions import Aggregate, Severity
from .spectral import Distortion
from .parser import UnderwritingLexer, UnderwritingParser
from .utilities import logger_level, round_bucket

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

    def __init__(self, name='Rory', databases=None, update=False, log2=10, debug=False, create_all=False):
        """
        Create underwriter object.

        :param name: Name of the underwriter object. Default is named after Rory Cline, the best
        underwriter I know.
        :param databases: if None: nothing loaded; if 'default' (installed) or 'site' (user,
        in ~/aggregate/databases) database *.agg files in default or site directory are loaded.
        If 'all' both default and site databases loaded. A string refers to a single database;
        an interable of strings is also valid. See `read_database` for search path.
        :param update: If True objects are updated after being created.
        :param log2: Default log2 value.
        :param debug: run parser in debug mode
        :param create_all: False write only creates Portfolio objects; True creates Aggregate objects
        that are within Portfolios.
        """

        self.name = name
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
        # default_dir is installed by pip and contain installation files
        default_dir = Path(__file__).parent / 'agg'
        # site dir is in Users's home directory and stores their files
        site_dir = Path.home() / 'aggregate/databases'

        self.site_dir = site_dir
        # check site dir exists
        self.site_dir.mkdir(parents=True, exist_ok=True)

        self.default_dir = default_dir

        # make sure all database entries are stored:
        if databases is None:
            # nothing to do
            databases = []

        if 'default' in databases:
            # add all databases in default_dir
            databases.remove('default')
            for fn in default_dir.glob('*.agg'):
                self.read_database(fn)

        if 'site' in databases:
            # add all user databases
            databases.remove('site')
            databases += list(site_dir.glob('*.agg'))

        for fn in databases:
            self.read_database(fn)

        self.create_all = create_all

    def read_database(self, fn):
        """
        read database of curves, aggs and portfolios. These can live in the default directory
        that is part of the instalation or ~/aggregate/

        fn can be a string filename, with or without extension. A .agg extension is
        added if there is no suffix. Search path:

        * in the current dir
        * in site_dir (user)
        * in default_dir (installation)

        :param fn: database file name

        """

        p = Path(fn)
        if p.suffix == '':
            p = p.with_suffix('.agg')
        if p.exists():
            db_path = p
        elif (self.site_dir / p).exists():
            db_path = self.site_dir / fn
        elif (self.default_dir / p).exists():
            db_path = self.default_dir / p
        else:
            logger.error(f'Database {fn} not found. Ignoring.')
            return

        try:
            program = db_path.read_text(encoding='utf-8')
        except:
            logger.error(f'Error reading requested database {db_path.name}. Ignoring.')
        else:
            # read in, parse, save to sev/agg/port dictionaries
            # throw away answer...not creating anything
            self.interpret_program(program)
            logger.info(f'Database {fn} read into knowledge.')

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
                # name == item, any type
                rows = self._knowledge.xs(item, axis=0, level=1, drop_level=False)
            elif type(item) == tuple:
                # return a dataframe row
                rows = self._knowledge.loc[[item]]
        except KeyError:
            raise KeyError(f'Item {item} not found.')
        except TypeError as e:
            # TODO fix this "TypeError: unhashable type: 'slice'"
            raise KeyError(f'getitem TypeError looking for {item}, {e}')
        else:
            if len(rows) == 1:
                kind, name, spec, prog = rows.reset_index().iloc[0]
                return kind, name, spec, prog
            else:
                raise KeyError(f'{len(rows)}>1 objects matching {item} found.')

    def _repr_html_(self):
        s = [f'<p><strong>Underwriter {self.name}: </strong></p>',
             f'Underwriter knowledge contains {len(self._knowledge)} aggregates and porfolios. '
             '<p>'
             'Run <code>build.knowledge</code> for more details on each object and '
             '<code>build.programs</code> for a program listing.' 
             '</p>'
             # '<br>',
             # self.knowledge.to_html(),
             f'<p>Settings: '
             ]
        for k in ['log2', 'update', 'create_all', 'debug']:
            s.append(f'<span style="color: red;">{k}</span>: {getattr(self, k)}; ')
        return '\n'.join(s) + '</p>'

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
            # spec = list of aggs
            agg_list = spec['spec']
            obj = Portfolio(name, agg_list, uw=self)
            obj.program = program
        elif kind == 'sev':
            if 'sev_wt' in spec and spec['sev_wt'] != 1:
                logger.warning(f'Mixed severity cannot be created, returning spec. You had {spec["sev_wt"]}, expected 1')
                obj = None
            else:
                obj = Severity(**spec)
                # ? set outside...
                obj.program = program
        elif kind == 'distortion':
            obj = Distortion(**spec)
            # ? set outside
            obj.program = program
        else:
            ValueError(f'Cannot build {kind} objects')
        return obj

    @property
    def programs(self):
        """
        Return the knowledge as a nice dataframe

        :return:
        """
        bit = self._knowledge[['program']]
        bit['note'] = [s['note'] for s in self._knowledge.spec]
        bit['clean program']= bit.program.str.replace(r'note\{[^}]*\}|[ ]{2,}|\t+', ' ')
        bit = bit.sort_index(ascending=[False, True])
        bit = bit.reset_index(0, drop=False)
        bit = bit[['kind', 'note', 'clean program']].query('`clean program` != ""')
        bit = bit[['kind', 'note', 'clean program']]
        with pd.option_context('display.multi_sparse', False):
            # which seems to be ignored?
            sbit = bit.style.set_table_styles([
            {
                'selector': 'td',
                'props': 'text-align: left'},
            {
                'selector': 'th.col_heading',
                'props': 'text-align: left;'
            },
            {
                'selector': '.row_heading',
                'props': 'text-align: left;'
            }
        ])
        return sbit

    @property
    def knowledge(self):
        return self._knowledge.sort_index()

    def describe(self, kind='agg'):
        """
        TODO this is not rationlized...and as a prop it needs to return something
        More informative version of knowledge showing the agg programs. Only Aggregates that have a program.

        TODO Add severities

        :return:
        """
        from IPython.display import HTML

        if kind == 'all':
            kinds = ['agg', 'port']
        else:
            kinds = [kind]

        if 'port' in kinds:
            cols = ['Name', 'Type', 'Agg1', 'Agg2', 'Agg3', 'Notes']
            df_port = pd.DataFrame(columns=cols)
            df_port = df_port.set_index('Name')
            for (kind, name), (spec, program) in self._knowledge.xs('port', axis=0, level=0, drop_level=False).iterrows():
                aggs = spec['spec'][:3] # the list of agg items
                if len(aggs) == 1:
                    aggs.extend(['', ''])
                elif len(aggs) == 2:
                    aggs.extend([''])
                note = spec['note']
                df_port.loc[name, :] = [kind] + aggs + [note]
            df_port = df_port.sort_index()
            return df_port
            # display(HTML('<h3>Known Portfolios</h3>'))
            # display(df_port)

        if 'agg' in kinds:
            cols = ['Name', 'Type', 'ELoss', 'Severity', 'ESev',  'SevCV', 'Sev_a', 'Sev_b', 'Freq',      'EN',     'Freq_a',  'Notes']
            # what they are actually called
            cols_agg =          ['exp_el', 'sev_name', 'sev_mean', 'sev_cv', 'sev_a', 'sev_b', 'freq_name', 'exp_en', 'freq_a', 'note']
            df_agg = pd.DataFrame(columns=cols)
            df_agg = df_agg.set_index('Name')
            for (kind, name), (spec, program) in self._knowledge.xs('agg', axis=0, level=0, drop_level=False).iterrows():
                if program != '':
                    # if no program then it is part of a Portfolio; don't want to replicate here
                    df_agg.loc[name, :] = [kind] + [spec.get(f, '') for f in cols_agg]
            df_agg = df_agg.sort_index()
            df_agg['ELoss'] = np.where(df_agg.ELoss == '', df_agg.ESev.replace('', 0) * df_agg.EN.replace('', 0), df_agg.ELoss)
            df_agg = df_agg.drop(columns='Sev_b')
            return df_agg
            # display(HTML('<h3>Known Aggregates</h3>'))
            # display(df_agg)

    def write(self, portfolio_program, log2=0, bs=0, create_all=None, update=None, **kwargs):
        """
        Write a natural language program. Write carries out the following steps.

        1. Read in the program and cleans it (e.g. punctuation, parens etc. are
           removed and ignored, replace ; with new line etc.)
        2. Parse line by line to create a dictionary definition of sev, agg or port objects.
        3. Replace sev.name, agg.name and port.name references with their objects.
        4. If create_all set, create all objects and return in dictionary. If not set only create the port objects.
        5. If update set, update all created objects.

        Sample input

        ::

            port MY_PORTFOLIO
                agg Line1 20  loss 3 x 2 sev gamma 5 cv 0.30 mixed gamma 0.4
                agg Line2 10  claims 3 x 2 sevgamma 12 cv 0.30 mixed gamma 1.2
                agg Line 3100  premium at 0.4 3 x 2 sev 4 @ lognormal 3 cv 0.8 fixed 1

        The indents are required if each agg item appears on a new line.

        See parser for full language spec! See Aggregate class for many examples.

        Reasonable kwargs:

        * **bs**
        * **log2**
        * **update** overrides class default
        * **add_exa** should port.add_exa add the exa related columns to the output?
        * **create_all**: create all objects, default just portfolios. You generally
          don't want to create underlying sevs and aggs in a portfolio.

        :param portfolio_program:
        :param create_all: override class default
        :param update: override class default
        :param kwargs: passed to object's update method if update==True
        :return: single created object or dictionary name: object
        """

        # prepare for update
        # what / how to do; little awkward: to make easier for user have to strip named update args
        # out of kwargs
        if create_all is None:
            create_all = self.create_all
        if update is None:
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
            # rationalize return to be the same as parsed programs
            # TODO test this code
            rv = {(kind, name): (obj, program)}
            return rv

        # if you fall through to here then the portfolio_program did not refer to a built-in object
        # run the program, get the interpreter return value, the irv, which contains kind/name->spec,program
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
                # OLD the spec comes back as a list of aggs that have been entered into the uw
                # NEW the spec comes back as a list of dictionary agg specs that are NOT entered into the uw
                if create_all:
                    obj = self.factory(kind, name, spec, program)
                    if obj is not None:
                        # this can fail for named mixed severities, which can only
                        # be created in context of an agg... that behaviour is
                        # useful for named severities though... hence:
                        if update:
                            update = getattr(obj, 'update', None)
                            if update is not None:
                                update(log2, bs, **kwargs)
                        # TODO sort out this cluster
                        rv[(kind, name)] = (obj, program)
                    else:
                        rv[(kind, name)] = (spec, program)
                else:
                    rv[(kind, name)] = (spec, program)

        # report on what has been done
        if rv is None:
            logger.warning(f'Underwriter.write | Program did not contain any output')
        else:
            if len(rv):
                logger.info(f'Underwriter.write | Program created {len(rv)} objects.')

        # return created objects
        return rv

    def write_from_file(self, file_name, log2=0, bs=0, update=False, **kwargs):
        """
        Read program from file. Delegates to write.

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
        Preprocess and then parse a program one line at a time. Each output is
        stored in the Underwriter's knowledge database. No objects are created.

        Error handling through parser.

        :param portfolio_program:
        :return:
        """

        # Preprocess ---------------------------------------------------------------------
        portfolio_program = self.lexer.preprocess(portfolio_program)

        # create return value dictionary
        rv = {}

        # Parse and Postprocess-----------------------------------------------------------
        # self.parser.reset()
        # program_line_dict = {}
        for program_line in portfolio_program:
            logger.debug(program_line)
            # preprocessor only returns lines of length > 0
            try:
                # parser returns the type, name, and spec of the object
                # this is where you can marry up with the program
                kind, name, spec = self.parser.parse(self.lexer.tokenize(program_line))
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
                # store in uw dictionary and create if needed
                logger.info(f'answer out: {kind} object {name} parsed successfully...adding to knowledge')
                self._knowledge.loc[(kind, name), :] = [spec, program_line]
                rv[(kind, name)] = (spec, program_line)

        return rv

    # @staticmethod
    # def add_defaults(dict_in, kind='agg'):
    #     """
    #     add default values to dict_inin. Leave existing values unchanged
    #     Used to output to a data frame, where you want all columns completed
    #
    #     :param dict_in:
    #     :param kind:
    #     :return:
    #     """
    #
    #     print('running add_defaults\n' * 10)
    #
    #     # use inspect to get the defaults
    #     # obtain signature
    #     sig = signature(Aggregate.__init__)
    #
    #     # self and name --> bound signature
    #     bs = sig.bind(None, '')
    #     bs.apply_defaults()
    #     # remove self
    #     bs.arguments.pop('self')
    #     defaults = bs.arguments
    #
    #     if kind == 'agg':
    #         defaults.update(dict_in)
    #
    #     elif kind == 'sev':
    #         for k, v in defaults.items():
    #             if k[0:3] == 'sev' and k not in dict_in and k != 'sev_wt':
    #                 dict_in[k] = v

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
        # don't want to pass back the original
        spec = spec.copy()
        return spec

    @staticmethod
    def logger_level(level):
        """
        Convenience function.
        :param level:
        :return:
        """
        # set global logger_level
        logger_level(level)

    def build(self, program, update=True, create_all=None, log2=-1, bs=0, log_level=None, **kwargs):
        """
        Convenience function to make work easy for the user. Intelligent auto updating.
        Detects discrete distributions and sets ``bs = 1``.

        ``build`` method sets loger level to 30 by default.

        ``__call__`` is set equal to ``build``.

        :param program:
        :param update: build's update
        :param create_all: for just this run
        :param log2: -1 is default. Figure log2 for discrete and 13 for all others. Inupt value over-rides
        and cancels discrete computation (good for large discrete outcomes where bucket happens to be 1.)
        :param bs:
        :param log_level:
        :param kwargs: passed to update, e.g., padding. Note force_severity=True is applied automatically
        :return: created object(s)
        """

        # TODO put back the original logger level
        if log_level is not None:
            self.logger_level(log_level)

        # options for this run
        if create_all is None:
            create_all = self.create_all

        # make stuff
        # write will return a dict with keys (kind, name) and value either the object or the spec
        out_dict = self.write(program, create_all=create_all, update=False, force_severity=True)

        if out_dict is None:
            logger.warning('build produced no output')
            return None

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
                        bs_ = round_bucket(out.recommend_bucket(log2_))
                    else:
                        bs_ = bs
                    logger.info(f'({kind}, {name}): Normal mode, using bs={bs_} and log2={log2_}')
                try:
                    out.update(log2=log2_, bs=bs_, debug=self.debug, force_severity=True, **kwargs)
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
                logger.info(f'updating with {log2}, bs=1/{1/bs_}')
                logger.info(f'({kind}, {name}): bs={bs_} and log2={log2_}')
                out.update(log2=log2_, bs=bs_, remove_fuzz=True, force_severity=True,
                           debug=self.debug, **kwargs)
            else:
                logger.warning(f'Unexpected: output kind is {type(out)}. (expr/number?)')
                pass

        if len(out_dict) == 1:
            # only one output...just return that
            # dict, pop the last (only) element (popitem: Remove and return a (key, value) pair as a 2-tuple.)
            out_dict = out_dict.popitem()[1]
            if len(out_dict) == 2:
                out_dict = out_dict[0]
            else:
                raise ValueError('Weird type coming out of update. Investigate.')
        else:
            # multiple outputs, see if there is just one portfolio...this is not ideal?!
            ports_found = 0
            port = None
            for (kind, name), (ob, program) in out_dict.items():
                if kind == 'port':
                    ports_found += 1
                    port = ob
            if ports_found == 1:
                out_dict = port

        return out_dict

    __call__ = build

    def interpreter_file(self, where='', filename=''):
        """
        Run a suite of test programs. For detailed analysis, run_one.

        """
        if filename == '':
            filename = Path.home() / 'aggregate/tests/test_suite.csv'
        df = pd.read_csv(filename, index_col=0)
        if where != '':
            df = df.loc[df.index.str.match(where)]
        # add One severity
        self.write('sev One dsev [1]')
        return self._interpreter_work(df.iterrows())

    def interpreter_line(self, program, name='one off', debug=True):
        """
        Interpret single line of code  in debug mode.
        name is index of output
        """

        return self._interpreter_work(iter([(name, program)]), debug=debug)

    def interpreter_list(self, program_list):
        """
        Interpret single test in debug mode.
        """
        return self._interpreter_work(list(enumerate(program_list)), debug=True)

    def _interpreter_work(self, iterable, debug=False):
        """
        Do all the work for the test, allows input to be marshalled into the tester
        in different ways. Unlike production interpret_program, runs one line at a time.
        Each line is preprocessed and then run through a clean parser, and the output
        analyzed.

        Last column, program as input is only changed if the preprocessor changes the program

        :return: DataFrame
        """
        lexer = UnderwritingLexer()
        parser = UnderwritingParser(self.safe_lookup, debug)
        ans = {}
        errs = 0
        no_errs = 0
        # detect non-trivial change
        f = lambda x, y: 'same' if x.replace(' ', '') == y.replace(' ', '').replace('\t', '') else y
        for test_name, program in iterable:
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
                    kind, name, spec = parser.parse(lexer.tokenize(program))
                except (ValueError, TypeError) as e:
                    errs += 1
                    err = 1
                    kind = program.split()[0]
                    # get something to say about the error
                    ea = getattr(e, 'args', None)
                    if ea is not None:
                        # t = getattr(ea[0], 'type', ea[0])
                        # v = getattr(ea[0], 'value', ea[0])
                        i = getattr(ea[0], 'index', 0)
                        if type(i) != int:  i = 0
                        # print(i, ea)
                        txt = program[0:i] + f'>>>' + program[i:]
                        name = 'parse error'
                    else:
                        txt = str(e)
                        name = 'other error'
                    spec = txt
                else:
                    no_errs += 1
                ans[test_name] = [kind, err, name, spec, program, f(program, program_in)]
            elif len(program) > 1:
                logger.info(f'{program_in} preprocesses to {len(program)} lines; not processing.')
                logger.info(program)
                ans[test_name] = ['multiline', err, None, None, program, program_in]
            else:
                logger.info(f'{program_in} preprocesses to a blank line; ignoring.')
                ans[test_name] = ['blank', err, None, None, program, program_in]

        df_out = pd.DataFrame(ans,
                              index=['kind', 'error', 'name', 'output', 'preprocessed program',
                                     'program']).T
        df_out.index.name = 'index'
        return df_out


# exported instance
# build = dbuild = None
build = Underwriter(databases='examples', create_all=False, update=True, debug=False, log2=16)
dbuild = Underwriter(name='Debug', create_all=False, update=True, debug=True, log2=13)
