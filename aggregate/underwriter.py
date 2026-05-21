from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .constants import WL, VALIDATION_EPS, RECOMMEND_P
from .portfolio import Portfolio
from .distributions import Aggregate, Severity
from .spectral import Distortion
from .parser import UnderwritingLexer, UnderwritingParser
from .utilities import (round_bucket, qd, show_fig, more, parse_note_ex)

logger = logging.getLogger(__name__)


@dataclass
class ParsedProgram:
    """One DecL declaration after parsing, with optional constructed object.

    Returned by :meth:`Underwriter.interpret_program` and used internally by
    :meth:`Underwriter.factory` / :meth:`Underwriter.build`. The ``object``
    field is ``None`` after parsing and is populated by :meth:`Underwriter.factory`
    once the corresponding Aggregate / Severity / Portfolio / Distortion is built.
    """
    kind: str           # 'agg' | 'sev' | 'port' | 'distortion' | 'expr'
    name: str           # the user-given name (e.g. 'Dice', 'MyBook')
    spec: Any           # dict of kwargs for the constructor
    program: str        # the original DecL source line
    object: Any = None  # the constructed object once factory has run


class Underwriter(object):
    """
    Manage the creation of Aggregate, Severity, Portfolio, and Distortion objects.

    Maintains a database of named DecL declarations (the "knowledge base") and
    exposes the user-facing :meth:`build` entry point that parses a DecL program
    and constructs the corresponding object(s).

    Responsibilities:

    - Persist DecL programs to and from ``.agg`` files.
    - Bridge to the parser (`UnderwritingLexer` / `UnderwritingParser`).
    - Safe lookup of named programs from the knowledge base for the parser.

    Every parsed declaration has a *kind* (one of ``'sev'``, ``'agg'``, ``'port'``,
    ``'distortion'``) and a *name*. Parsing produces a :class:`ParsedProgram`
    holding the kind, name, dict spec, source program, and (once :meth:`factory`
    runs) the constructed object.

    The knowledge base is stored in ``self._knowledge`` — a DataFrame indexed
    by ``(kind, name)`` with columns ``spec`` and ``program``.
    """

    def __init__(self, name='Rory', databases=None, update=False, log2=10, debug=False):
        """
        Create an underwriter object. The underwriter is the interface to the knowledge base
        of the aggregate system. It is the interface to the parser and the interpreter, and
        to the database of curves, portfolios and aggregates.

        :param name: name of underwriter. Defaults to Rory, after Rory Cline, the best underwriter
            I know and a supporter of an analytic approach to underwriting.
        :param databases: name or list of database files to read in on creation. if None: nothing loaded; if
            'default' (installed) or 'site' (user, in ~/aggregate/databases) database \\*.agg files in default or site
            directory are loaded. If 'all' both default and site databases loaded. A string refers to a single database;
            an interable of strings is also valid. See `read_database` for search path.
        :param update: if True, update database files with new objects.
        :param log2: log2 of number of buckets in discrete representation.  10 is 1024 buckets.
        :param debug: if True, print debug messages.
        """

        self.name = name
        self.update = update
        if log2 <= 0:
            raise ValueError(
                'log2 must be > 0. The number of buckets used equals 2**log2.')
        self.log2 = log2
        self.debug = debug
        self._lexer = None
        self._parser = None
        # make sure all database entries are stored; they are read on demand
        self.databases = [] if databases is None else databases

        # do not read in until needed for faster loading
        self._default_dir = None
        self._site_dir = None
        self._case_dir = None
        self._template_dir = None
        self._knowledge = pd.DataFrame(columns=['kind', 'name', 'spec', 'program'], dtype=object).set_index(
            ['kind', 'name'])

        # (removed: was `pd.set_option('display.max_colwidth', 100)`; set in
        # your own session if you want wider DataFrame column display)

    @property
    def lexer(self):
        if self._lexer is None:
            self._lexer = UnderwritingLexer()
        return self._lexer

    @property
    def parser(self):
        if self._parser is None:
            self._parser = UnderwritingParser(self.safe_lookup, self.debug)
        return self._parser

    # The four *_dir properties below are lazily resolved and mkdir on first
    # access — downstream code in extensions/case_studies and the visual
    # test_suite reporter relies on these directories existing.
    @property
    def default_dir(self):
        """Installation directory holding the bundled ``.agg`` databases."""
        if self._default_dir is None:
            self._default_dir = Path(__file__).parent / 'agg'
            self._default_dir.mkdir(parents=True, exist_ok=True)
        return self._default_dir

    @property
    def site_dir(self):
        """User-local directory under ``~/aggregate/databases`` for custom ``.agg`` files."""
        if self._site_dir is None:
            self._site_dir = Path.home() / 'aggregate/databases'
            self._site_dir.mkdir(parents=True, exist_ok=True)
        return self._site_dir

    @property
    def case_dir(self):
        """User-local directory under ``~/aggregate/cases`` (used by case studies)."""
        if self._case_dir is None:
            self._case_dir = Path.home() / 'aggregate/cases'
            self._case_dir.mkdir(parents=True, exist_ok=True)
        return self._case_dir

    @property
    def template_dir(self):
        """Packaged Jinja/HTML template directory."""
        if self._template_dir is None:
            self._template_dir = self.default_dir.parent / 'templates'
            self._template_dir.mkdir(parents=True, exist_ok=True)
        return self._template_dir

    def read_databases(self):
        """
        Resolve ``self.databases`` (the constructor argument) into a list of
        files and read each one into the knowledge base. Does not mutate
        ``self.databases`` — calling this twice is idempotent.
        """
        requested = self.databases
        if not requested:
            return
        if requested == 'all':
            requested = ['default', 'site']
        elif isinstance(requested, str):
            requested = [requested]

        files = []
        for entry in requested:
            if entry == 'default':
                files.extend(self.default_dir.glob('*.agg'))
            elif entry == 'site':
                files.extend(self.site_dir.glob('*.agg'))
            else:
                files.append(entry)

        for fn in files:
            self.read_database(fn)

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
            logger.error('Database %s not found. Ignoring.', fn)
            return

        try:
            program = db_path.read_text(encoding='utf-8')
        except OSError:
            logger.exception('Error reading requested database %s. Ignoring.', db_path.name)
        else:
            # read in, parse, save to sev/agg/port dictionaries
            # throw away answer...not creating anything
            # get rid of cosmetic spaces, but keep newline tabs (2 or more spaces)
            program = re.sub('^  +', '\t', program, flags=re.MULTILINE)
            program = re.sub(' +', ' ', program)
            logger.info('Reading database %s...', fn)
            n = len(self._knowledge)
            self.interpret_program(program)
            n = len(self._knowledge) - n
            logger.info('Database %s read into knowledge, adding %d entries.', fn, n)

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
            raise ValueError(
                f'item must be a str (name of object) or tuple (kind, name), not {type(item)}.')

        assert self.knowledge is not None

        try:
            if type(item) == str:
                # name == item, any type
                rows = self._knowledge.xs(
                    item, axis=0, level=1, drop_level=False)
            elif type(item) == tuple:
                # return a dataframe row
                rows = self._knowledge.loc[[item]]
        except KeyError:
            raise KeyError(f'Item {item} not found.')
        except TypeError as e:
            # TODO fix this "TypeError: unhashable type: 'slice'"
            raise KeyError(f'getitem TypeError looking for {item}, {e}') from e
        else:
            if len(rows) == 1:
                kind, name, spec, program = rows.reset_index().iloc[0]
                return ParsedProgram(kind=kind, name=name, spec=spec, program=program)
            else:
                raise KeyError(
                    f'Error: no unique object found matching {item}. Found {len(rows)} objects.')

    @staticmethod
    def _format_dir(path: Path) -> str:
        """Render an absolute path as ``~/...`` if it lives under the user's home."""
        try:
            return f'~/{path.resolve().relative_to(Path.home())}'
        except ValueError:
            return str(path.resolve())

    def __repr__(self):
        # Count knowledge entries from the cached frame directly — avoid
        # self.knowledge here, which would trigger a database read.
        return (
            f'Underwriter        {self.name}\n'
            f'version            {self.version}\n'
            f'knowledge          {len(self._knowledge)} programs\n'
            f'update             {self.update}\n'
            f'log2               {self.log2}\n'
            f'debug              {self.debug}\n'
            f'validation_eps     {VALIDATION_EPS}\n'
            f'site dir           {self._format_dir(self.site_dir)}\n'
            f'default dir        {self._format_dir(self.default_dir)}'
        )

    def factory(self, parsed):
        """
        Construct the object described by a :class:`ParsedProgram` and attach it.

        Portfolio construction needs ``self`` (it is passed as the ``uw``
        argument), which is why this is not a staticmethod.

        :param parsed: a :class:`ParsedProgram` with ``kind``, ``name``, ``spec``,
            and ``program`` populated; ``object`` is None on input.
        :return: the same ``parsed`` with ``parsed.object`` set to the
            constructed object (or left ``None`` for the named-mixed-severity
            case, which can only be created in the context of an Aggregate).
        """

        kind, name, spec, program = parsed.kind, parsed.name, parsed.spec, parsed.program

        if kind == 'agg':
            obj = Aggregate(**spec)
            obj.program = program
        elif kind == 'port':
            # Portfolio expects name, agg_list, uw. agg_list is a list of specs
            # that can be passed to Aggregate. Drop the leading ('agg', name)
            # from each spec entry returned by the parser.
            agg_list = [k for i, j, k in spec['spec']]
            obj = Portfolio(name, agg_list, uw=self)
            obj.program = program
        elif kind == 'sev':
            if 'sev_wt' in spec and spec['sev_wt'] != 1:
                logger.log(WL,
                           'Mixed severity cannot be created, returning spec. You had %s, expected 1',
                           spec["sev_wt"])
                obj = None
            else:
                obj = Severity(**spec)
                obj.program = program
        elif kind == 'distortion':
            obj = Distortion(**spec)
            obj.program = program
        else:
            raise ValueError(f'Cannot build {kind} objects')

        parsed.object = obj
        return parsed

    @property
    def knowledge(self):
        if len(self._knowledge) == 0 and len(self.databases) > 0:
            # knowledge - accounts and line known to the underwriter
            self.read_databases()
        return self._knowledge.sort_index()[['program', 'spec']]

    @property
    def version(self):
        import aggregate
        return aggregate.__version__

    def test_suite(self):
        f = self.default_dir / 'test_suite.agg'
        txt = f.read_text(encoding='utf-8')
        return txt

    @property
    def test_suite_file(self):
        """
        Return the test_suite filename, or None if it does not exist
        """
        f = self.default_dir / 'test_suite.agg'
        if f.exists():
            return f
        else:
            return None

    def write(self, portfolio_program, log2=0, bs=0, update=None, **kwargs):
        """
        Interpret a DecL program and create the corresponding objects.

        Steps:

        1. Preprocess the program (strip comments, normalize whitespace,
           handle line continuations).
        2. Parse line by line to create a dictionary spec for each sev, agg,
           port, or distortion declaration.
        3. Resolve ``sev.name`` / ``agg.name`` builtin references via the
           knowledge base.
        4. If ``update`` is set, update all created objects.

        Sample input::

            port MY_PORTFOLIO
                agg Line1 20  loss xs 3 xs 2 sev gamma 5 cv 0.30 mixed gamma 0.4
                agg Line2 10  claims xs 3 xs 2 sev gamma 12 cv 0.30 mixed gamma 1.2
                agg Line3 100 premium at 0.4 lr 3 xs 2 sev 4 @ lognormal 3 cv 0.8 fixed 1

        Each ``agg`` clause inside a portfolio must be tab-indented on its own
        line; the preprocessor folds ``\\n\\t agg`` back into a single line for
        the parser.

        See ``aggregate/decl.lark`` for the full grammar and ``Aggregate``
        for many examples.

        :param portfolio_program: a DecL program (str), or the name of a
            previously-built object in the knowledge base.
        :param log2: log2 of the number of buckets for the discrete
            representation; 0 uses ``self.log2``.
        :param bs: bucket size; 0 lets the object recommend one.
        :param update: override the class-level ``self.update`` default.
        :param kwargs: passed to each created object's ``update`` method.
        :return: list of :class:`ParsedProgram` (one per top-level declaration).
        """

        # prepare for update
        # what / how to do; little awkward: to make easier for user have to strip named update args
        # out of kwargs
        if update is None:
            update = self.update

        if update is True and log2 == 0:
            log2 = self.log2

        # first see if portfolio_program refers to a built-in object
        try:
            # calls __getitem__
            answer = self[portfolio_program]
        except (LookupError, TypeError):
            logger.debug('underwriter.write | object not found, processing as a program.')
        else:
            logger.debug('underwriter.write | %s object found.', answer.kind)
            answer = self.factory(answer)
            if update:
                answer.object.update(log2, bs, **kwargs)
            # rationalize return to be the same as parsed programs
            return [answer]

        # if you fall through to here then the portfolio_program did not refer to a built-in object
        # run the program, get the interpreter return value, the irv, which contains kind/name->spec,program
        irv = self.interpret_program(portfolio_program)
        rv = []
        for answer in irv:
            # create objects and update if needed
            answer = self.factory(answer)
            if answer.object is not None:
                # this can fail for named mixed severities, which can only
                # be created in context of an agg... that behaviour is
                # useful for named severities though... hence:
                if update:
                    update_method = getattr(answer.object, 'update', None)
                    if update_method is not None:
                        update_method(log2, bs, **kwargs)
            rv.append(answer)

        # report on what has been done
        if not rv:
            logger.log(WL, 'Program did not contain any output')
        else:
            logger.info('Program created %d objects.', len(rv))

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

        # create return value list
        rv = []

        # Parse and Postprocess-----------------------------------------------------------
        # self.parser.reset()
        # program_line_dict = {}
        for program_line in portfolio_program:
            logger.debug(program_line)
            # preprocessor only returns lines of length > 0
            try:
                # parser returns the type, name, and spec of the object
                # this is where you can marry up with the program
                kind, name, spec = self.parser.parse(
                    self.lexer.tokenize(program_line))
            except ValueError as e:
                if isinstance(e.args[0], str):
                    logger.error(e)
                    raise e
                else:
                    t = e.args[0].type
                    v = e.args[0].value
                    i = e.args[0].index
                    txt2 = program_line[0:i] + '>>>' + program_line[i:]
                    logger.error('Parse error in input "%s"\nValue %s of type %s not expected',
                                 txt2, v, t)
                    raise e
            else:
                # store in uw dictionary and create if needed
                logger.info('answer out: %s object %s parsed successfully...adding to knowledge',
                            kind, name)
                self._knowledge.loc[(kind, name), :] = [spec, program_line]
                rv.append(ParsedProgram(kind=kind, name=name, spec=spec, program=program_line))

        return rv

    def safe_lookup(self, buildinid):
        """
        Lookup buildinid=kind.name in uw to find expected kind and merge safely into self.arg_dict.

        Different from getitem because it splits the item into kind and name and double
        checks you get the expected kind.

        :param buildinid:  a string in kind.name format
        :return:
        """

        # allow for sev.WC.1 name
        kind, *name = buildinid.split('.')
        name = '.'.join(name)
        try:
            # lookup in Underwriter
            parsed = self[(kind, name)]
        except LookupError:
            logger.error('ERROR id %s.%s not found in the knowledge.', kind, name)
            raise
        logger.debug('UnderwritingParser.safe_lookup | retrieved %s.%s as type %s.%s',
                     kind, name, parsed.kind, parsed.name)
        if parsed.kind != kind:
            raise ValueError(f'Error: type of {name} is  {parsed.kind}, not expected {kind}')
        # don't want to pass back the original otherwise changes can be reflected in the knowledge
        return deepcopy(parsed.spec)

    def _build_all(self, program, update=None, log2=0, bs=0, recommend_p=RECOMMEND_P, **kwargs):
        """
        Parse, build, and smart-update — the heavy lifting shared by
        :meth:`build` and :meth:`build_many`.

        Returns the full list of :class:`ParsedProgram` regardless of count.
        Callers enforce their own count contract.
        """
        rv = self.write(program, update=False, force_severity=True)

        if not rv:
            logger.log(WL, 'build produced no output')
            return rv

        if update is None:
            update = self.update

        # in this loop bs_ and log2_ are the values actually used for each
        # update; they do not overwrite the input default values
        for answer in rv:
            if answer.object is None:
                # object not created (named-mixed-severity case)
                logger.info('Object %s of kind %s returned as a spec; no further processing.',
                            answer.name, answer.kind)
            elif isinstance(answer.object, Aggregate) and update is True:
                d = answer.spec
                log2, bs, recommend_p, kwargs = parse_note_ex(
                    d['note'], log2, bs, recommend_p, kwargs)
                if d['sev_name'] == 'dhistogram' and log2 == 0:
                    bs_ = 1
                    # how big?
                    if d['freq_name'] == 'fixed':
                        max_loss = np.max(d['sev_xs']) * d['exp_en']
                    elif d['freq_name'] == 'empirical':
                        max_loss = np.max(d['sev_xs']) * max(d['freq_a'])
                    elif d['freq_name'] == 'bernoulli':
                        max_loss = np.max(d['sev_xs'])
                    else:
                        # normal approx on count
                        max_loss = np.max(d['sev_xs']) * d['exp_en'] * (1 + 3 * d['exp_en'] ** 0.5)
                    # binaries are 0b111... len-2 * 2 is len - 1
                    log2_ = len(bin(int(max_loss))) - 1
                    logger.info('(%s, %s): Discrete mode, using bs=1 and log2=%s',
                                answer.kind, answer.name, log2_)
                else:
                    log2_ = self.log2 if log2 == 0 else log2
                    if bs == 0:
                        bs_ = round_bucket(answer.object.recommend_bucket(log2_, p=recommend_p))
                    else:
                        bs_ = bs
                    logger.info('(%s, %s): Normal mode, using bs=%s (1/%s) and log2=%s',
                                answer.kind, answer.name, bs_, 1 / bs_, log2_)
                try:
                    answer.object.update(
                        log2=log2_, bs=bs_, debug=self.debug, force_severity=True, **kwargs)
                except (ZeroDivisionError, AttributeError) as e:
                    logger.error(e)
            elif isinstance(answer.object, Severity):
                # severities have no update
                pass
            elif isinstance(answer.object, Portfolio) and update is True:
                d = answer.spec
                log2, bs, recommend_p, kwargs = parse_note_ex(
                    d['note'], log2, bs, recommend_p, kwargs)
                if log2 == -1:
                    log2_ = 13
                elif log2 == 0:
                    log2_ = self.log2
                else:
                    log2_ = log2
                if bs == 0:
                    bs_ = answer.object.best_bucket(log2_, recommend_p)
                else:
                    bs_ = bs
                logger.info('(%s, %s): bs=%s and log2=%s', answer.kind, answer.name, bs_, log2_)
                answer.object.update(log2=log2_, bs=bs_, remove_fuzz=True, force_severity=True,
                                     debug=self.debug, **kwargs)
            elif isinstance(answer.object, Distortion):
                pass
            elif isinstance(answer.object, (Aggregate, Portfolio)) and update is False:
                pass
            else:
                logger.warning('Unexpected: output kind is %s. (expr/number?)', type(answer.object))

        return rv

    def build(self, program, update=None, log2=0, bs=0, recommend_p=RECOMMEND_P, **kwargs):
        """
        Parse a single DecL program and return the constructed object.

        ``build`` is the primary user-facing entry point. It parses the
        program, constructs the corresponding object, and smart-updates the
        discrete distribution (detecting discrete severities to pick ``bs=1``,
        otherwise calling :meth:`recommend_bucket`).

        ``__call__`` delegates to ``build``.

        :param program: a DecL program producing exactly one top-level output.
        :param update: override the class-level ``self.update`` default.
        :param log2: 0 (default) estimates log2 for discrete severities and
            uses ``self.log2`` for everything else. A nonzero value overrides
            the discrete-mode estimation.
        :param bs: bucket size; 0 lets the object recommend one.
        :param recommend_p: passed to :meth:`recommend_bucket`; raise (closer
            to 1) for thick-tailed distributions.
        :param kwargs: passed to ``update`` (e.g. ``padding``). ``force_severity=True``
            is always applied.
        :return: the constructed Aggregate / Severity / Portfolio / Distortion.
            For the named-mixed-severity case where the spec cannot be built
            standalone, returns the :class:`ParsedProgram` with ``object=None``.
        :raises ValueError: if the program produces zero or more than one
            top-level output. Use :meth:`build_many` for batched programs.
        """
        rv = self._build_all(program, update=update, log2=log2, bs=bs,
                             recommend_p=recommend_p, **kwargs)
        if len(rv) != 1:
            raise ValueError(
                f'build() expects a single output, got {len(rv)}; '
                f'use build_many() for batched programs.'
            )
        answer = rv[0]
        return answer if answer.object is None else answer.object

    def build_many(self, program, update=None, log2=0, bs=0, recommend_p=RECOMMEND_P, **kwargs):
        """
        Parse a (possibly multi-output) DecL program and return all results.

        Same smart-update logic as :meth:`build`, but returns the full
        ``list[ParsedProgram]`` regardless of count. Use this when a program
        deliberately produces multiple top-level outputs.

        :return: list of :class:`ParsedProgram`, one per top-level output.
        """
        return self._build_all(program, update=update, log2=log2, bs=bs,
                               recommend_p=recommend_p, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def interpret_test_file(self, *, filename='', where=''):
        """
        Run every DecL program in a ``.agg`` or ``.csv`` file through the parser
        without creating output, returning a DataFrame with per-line parse-error info.

        This is for *testing* the interpreter — that is, exercising the lex/parse
        path and reporting which lines fail — not for actually interpreting
        anything into objects. Use :meth:`build` or :meth:`build_many` for that.

        For ``.csv`` files, the first column is used as index. For ``.agg`` files,
        the text is preprocessed (``\\n\\tagg`` folded back into one line, comments
        stripped) and then split on newlines.

        :param filename: a string or Path. Defaults to
            ``~/aggregate/tests/test_suite.csv`` if empty.
        :param where: regex filter on the DataFrame index; '' means all rows.
        :return: DataFrame with columns ``kind, error, name, output,
            preprocessed program, program``.
        """
        if filename == '':
            filename = Path.home() / 'aggregate/tests/test_suite.csv'
        elif isinstance(filename, str):
            filename = Path(filename)
        if filename.suffix == '.csv':
            df = pd.read_csv(filename, index_col=0)
        elif filename.suffix == '.agg':
            txt = filename.read_text(encoding='utf-8')
            stxt = re.sub('\n\tagg', ' agg', txt, flags=re.MULTILINE)
            stxt = [i for i in stxt.split('\n') if len(i) and i[0] != '#']
            df = pd.DataFrame(stxt, columns=['program'])
        else:
            raise ValueError(f'File suffix must be .csv or .agg, not {filename.suffix}')
        if where != '':
            df = df.loc[df.index.str.match(where)]
        # ensure the canonical One severity is present for any sev.One references
        self.write('sev One dsev [1]')

        ans = {}
        # detect a non-trivial change between preprocessed and input program
        def _changed(preprocessed, original):
            return 'same' if preprocessed.replace(' ', '') == original.replace(' ', '').replace('\t', '') else original

        for test_name, program in df.iterrows():
            program_in = program[0] if not isinstance(program, str) else program
            preprocessed = self.lexer.preprocess(program_in)
            err = 0
            if len(preprocessed) == 1:
                line = preprocessed[0]
                try:
                    kind, name, spec = self.parser.parse(self.lexer.tokenize(line))
                except (ValueError, TypeError) as e:
                    err = 1
                    kind = line.split()[0]
                    ea = getattr(e, 'args', None)
                    if ea is not None:
                        i = getattr(ea[0], 'index', 0)
                        if not isinstance(i, int):
                            i = 0
                        spec = line[0:i] + '>>>' + line[i:]
                        name = 'parse error'
                    else:
                        spec = str(e)
                        name = 'other error'
                ans[test_name] = [kind, err, name, spec, line, _changed(line, program_in)]
            elif len(preprocessed) > 1:
                logger.info('%s preprocesses to %d lines; not processing.',
                            program_in, len(preprocessed))
                ans[test_name] = ['multiline', err, None, None, preprocessed, program_in]
            else:
                logger.info('%s preprocesses to a blank line; ignoring.', program_in)
                ans[test_name] = ['blank', err, None, None, preprocessed, program_in]

        df_out = pd.DataFrame(ans, index=['kind', 'error', 'name', 'output',
                                          'preprocessed program', 'program']).T
        df_out.index.name = 'index'
        return df_out

    def run_test_suite(self):
        """Run :meth:`interpret_test_file` on the bundled test suite."""
        df = self.interpret_test_file(filename=self.test_suite_file)
        num_errors = df.error.sum()
        if num_errors != 0:
            logger.error('%d errors in test suite', num_errors)
        return df

    def more(self, regex):
        """
        More information about methods and properties matching regex

        """
        more(self, regex)

    def qlist(self, regex):
        """
        Wrapper for show to just list elements in knowledge that match ``regex``.
        Returns a dataframe.
        """
        return self.show(regex, kind='', plot=False, describe=False, verbose=True)

    def qshow(self, regex, tacit=True):
        """
        Wrapper for show to just show (display) elements in knowledge that match ``regex``.
        No reutrn value if tacit, else returns a dataframe.

        """
        def ff(x):
            fs = '{x:120s}'
            return fs.format(x=x)
        bit = self.show(regex, kind='', plot=False,
                        describe=False)[['program']]
        bit['program'] = bit['program'].str.replace(
            r' note\{[^}]+\}', '').str.replace('  +', ' ')  # , flags=re.MULTILINE)
        # bit['program'] = bit['program'].str.replace(' ( +)', ' ') #, flags=re.MULTILINE)
        # bit['program'] = bit['program'].str.replace(r' note\{[^}]+\}$|  *', ' '   ) #, flags=re.MULTILINE)
        if tacit:
            qd(bit,
               line_width=160, max_colwidth=130, col_space=15, justify='left',
               max_rows=200, formatters={'program': ff})
        else:
            return bit

    def show(self, regex, kind='', plot=True, describe=True, verbose=False, **kwargs):
        """
        Create from knowledge by name or match to name.
        Optionally plot. Returns the created object plus dataframe with more detailed information.
        Allows exploration of preloaded databases.

        Eg ``regex = "A.*[234]`` to run examples named A...2, 3 and 4.

        See ``qshow`` for a wrapper that just returns the matches, with no object
        creation or plotting.

        Examples.
        ::

            from aggregate.utilities import pprint
            # pretty print all prgrams starting A; no object creation
            build.show('^A.*', 'agg', False, False).program.apply(pprint);

            # build and plot A..234
            ans, df = build.show('^A.*')

        :param regex: for filtering name
        :param kind: the kind of object, port, agg, etc.
        :param plot:    if True, plot   (default True)
        :param describe: if True, print the describe dataframe
        :param verbose: if True, return the dataframe and objects; else no return value
        :param kwargs: passed to build for calculation instructions
        :return: dictionary of created objects and DataFrame with info about each.
        """
        # too painful getting the one thing out!
        ans = []

        if kind is None or kind == '':
            df = self.knowledge.droplevel('kind').filter(
                regex=regex, axis=0).copy()
        else:
            df = self.knowledge.loc[kind].filter(regex=regex, axis=0).copy()

        # severity causes an error: no est_m etc.
        if "One" in df.index:
            df = df.drop(index='One')

        if plot is False and describe is False:
            # just act like a filtered listing on knowledge
            return df.sort_values('name')

        # added detail columns
        df['log2'] = 0
        df['bs'] = 0.
        df['agg_m'] = 0.
        df['agg_cv'] = 0.
        df['agg_sd'] = 0.
        df['agg_skew'] = 0.
        df['emp_m'] = 0.
        df['emp_cv'] = 0.
        df['emp_sd'] = 0.
        df['emp_skew'] = 0.
        df['valid'] = False

        for n, row in df.iterrows():
            p = row.program
            try:
                a = self.build(p, **kwargs)
                ans.append(a)
            except NotImplementedError:
                logger.error('skipping %s...element not implemented', n)
            else:
                if describe:
                    pp = getattr(a, 'pprogram', None)
                    if pp is not None:
                        print(pp)
                    qd(a)
                if plot is True:
                    a.plot(figsize=(8, 2.4))
                    # print('\nDensity and Quantiles')
                    print()
                    show_fig(a.figure, format='svg')
                if describe:
                    print('\n')
                df.loc[n, ['log2', 'bs', 'agg_m', 'agg_cv', 'agg_sd', 'agg_skew',
                           'emp_m', 'emp_cv', 'emp_sd', 'emp_skew', 'valid']] = (
                    a.log2, a.bs, a.agg_m, a.agg_cv, a.agg_sd, a.agg_skew, a.est_m, a.est_cv, a.est_sd,
                    a.est_skew, a.explain_validation())
        # if only one item, return it...much easier to use
        if len(ans) == 1:
            # noinspection PyUnboundLocalVariable
            ans = a
        if verbose:
            return ans, df

    def dir(self, pattern=''):
        """
        List all agg databases in site and default directories.
        If entries is True then read them and return named objects.

        :param pattern:  glob pattern for filename; .agg is added

        """

        if pattern == '':
            pattern = '*.agg'
        else:
            pattern += '.agg'

        entries = []

        for dn, d in zip(['site', 'default'], [self.site_dir, self.default_dir]):
            for fn in d.glob(pattern):
                txt = fn.read_text(encoding='utf-8')
                stxt = txt.split('\n')
                for r in stxt:
                    rs = r.split(' ')
                    if rs[0] in ['agg', 'port', 'dist', 'distortion', 'sev']:
                        entries.append([dn, fn.name] + rs[:2])

        ans = pd.DataFrame(entries, columns=[
                           'Directory', 'Database', 'kind', 'name'])
        return ans


# Module-level singleton — the canonical user-facing entry point. Importable
# as `from aggregate import build`. `update=True` is why `build('agg ...')`
# auto-updates the constructed object's discrete distribution; pass
# `update=False` (or override at call time) to disable.
build = Underwriter(databases='test_suite', update=True, debug=False, log2=16)
# uncomment to create debug build, add to __init__.py
# debug_build = Underwriter(name='Debug', update=True, debug=True, log2=16)
