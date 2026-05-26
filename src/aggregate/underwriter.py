from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .constants import (WL, VALIDATION_EPS, RECOMMEND_P,
                        USER_DIR_NAME, PACKAGE_DATA_DIR, TEST_SUITE_FILENAME)
from .portfolio import Portfolio
from .distributions import Aggregate, Severity
from .spectral import Distortion
from .parser import UnderwritingLexer, UnderwritingParser
from .utilities import (round_bucket, qd, agg_help)

logger = logging.getLogger(__name__)


def _parse_note(txt, log2, bs, recommend_p, kwargs):
    """
    Extract build kwargs from a DecL note string and merge with caller-supplied
    defaults. Recognizes ``bs``, ``log2``, ``padding``, ``normalize``, and
    ``recommend_p`` in CSS-style ``key=value;`` form. ``bs`` accepts ``1/32``
    style fractions. ``log2`` and ``bs`` from the note are taken only when the
    caller's value is ``0`` (i.e. unset); ``recommend_p`` from the note always
    wins. Remaining keys are merged into ``kwargs``.

    Returns ``(log2, bs, recommend_p, kwargs)`` ready for ``Aggregate``/``Portfolio``
    update.
    """
    stxt = txt.split(';')
    kw = {}
    for s in stxt:
        parts = s.split('=')
        if len(parts) == 2:
            k = parts[0].strip()
            v = parts[1].strip()
            if re.match('bs|recommend_p', k):
                if re.match(r'(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?/(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?', v):
                    v = eval(v)
                else:
                    v = float(v)
            elif re.match('log2|padding', k):
                v = int(v)
            elif 'normalize':
                v = v == 'True'
            kw[k] = v
    if 'log2' in kw and log2 == 0:
        log2 = kw.pop('log2')
    if 'bs' in kw and bs == 0:
        bs = kw.pop('bs')
    if 'recommend_p' in kw:
        # always take the recommend_p from the note
        recommend_p = kw.pop('recommend_p')
    # rest are passed through
    kwargs.update(kw)
    return log2, bs, recommend_p, kwargs


def _row_stats(a, summary_cols):
    """
    Return a list of summary statistics for ``a`` aligned with ``summary_cols``.

    Per-class capability matrix:

    - :class:`Aggregate` / :class:`Portfolio`: all fields populated (log2, bs,
      theoretical and empirical moments, validation).
    - :class:`Severity`: theoretical moments only via ``a.stats('mvsk')``; no
      discretization, no empirical moments, no validation.
    - :class:`Distortion`: none of these fields apply — all returned as ``None``.

    Inapplicable fields are returned as ``None``; the caller's DataFrame is
    object dtype so this mixes cleanly with numeric values from other rows.
    """
    row = {col: None for col in summary_cols}
    if isinstance(a, (Aggregate, Portfolio)):
        row.update(log2=a.log2, bs=a.bs,
                   agg_m=a.agg_m, agg_cv=a.agg_cv, agg_sd=a.agg_sd, agg_skew=a.agg_skew,
                   emp_m=a.est_m, emp_cv=a.est_cv, emp_sd=a.est_sd, emp_skew=a.est_skew,
                   valid=a.explain_validation())
    elif isinstance(a, Severity):
        # theoretical moments only; severity has no discretization. Use the
        # project's own .moms(), which returns raw moments (E[X], E[X^2], E[X^3]);
        # scipy's .stats('mvsk') currently raises a UFuncTypeError on Severity.
        try:
            ex1, ex2, ex3 = a.moms()
            m = float(ex1)
            var = float(ex2) - m * m
            sd = float(np.sqrt(var)) if var > 0 else 0.0
            cv = (sd / m) if m else None
            if var > 0:
                # central third moment, then standardized skew
                mu3 = float(ex3) - 3 * m * float(ex2) + 2 * m ** 3
                skew = mu3 / sd ** 3
            else:
                skew = None
            row.update(agg_m=m, agg_sd=sd, agg_cv=cv, agg_skew=skew)
        except Exception as e:
            logger.debug('Severity %s: moms unavailable (%s)', getattr(a, 'name', '?'), e)
    # Distortion: nothing applies — all fields left as None
    return [row[col] for col in summary_cols]


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


class CannotBuild(ValueError):
    """Raised by :meth:`Underwriter.build` when a parsed spec produces no top-level object.

    Typically the named-mixed-severity case: a ``sev`` declaration with ``wts``
    can only live inside an :class:`Aggregate`, not standalone. Use
    :meth:`Underwriter.build_many` to receive the :class:`ParsedProgram`
    instead, then inspect ``.spec`` directly.

    Subclass of :class:`ValueError` so existing broad ``except ValueError``
    callers continue to catch it.
    """


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
        self._user_dir = None
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
            self._parser = UnderwritingParser(self._safe_lookup, self.debug)
        return self._parser

    @property
    def default_dir(self):
        """
        Installation directory holding the bundled ``.agg`` databases.

        Read-only package data, located via :func:`importlib.resources.files`.
        List bundled databases::

            list(uw.default_dir.glob('*.agg'))
        """
        if self._default_dir is None:
            self._default_dir = Path(files('aggregate')) / PACKAGE_DATA_DIR
        return self._default_dir

    @property
    def user_dir(self):
        """
        User-local data directory (``~/.aggregate``); mkdir'd on first access.

        Drop your own ``.agg`` databases here and they will be picked up by
        ``Underwriter(databases='all')`` (or ``='user'``) or by
        ``read_database('my_curves')``.  List user databases::

            list(uw.user_dir.glob('*.agg'))
        """
        if self._user_dir is None:
            self._user_dir = Path.home() / USER_DIR_NAME
            self._user_dir.mkdir(parents=True, exist_ok=True)
        return self._user_dir

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
            requested = ['default', 'user']
        elif isinstance(requested, str):
            requested = [requested]

        db_files = []
        for entry in requested:
            if entry == 'default':
                db_files.extend(self.default_dir.glob('*.agg'))
            elif entry == 'user':
                db_files.extend(self.user_dir.glob('*.agg'))
            elif entry == 'site':
                raise ValueError(
                    "databases='site' is not recognized; use 'user' "
                    "(renamed for v1.0; data lives in ~/.aggregate)")
            else:
                db_files.append(entry)

        for fn in db_files:
            self.read_database(fn)

    def read_database(self, fn):
        """
        Read a database of curves, aggs, and portfolios from a ``.agg`` file.

        ``fn`` may be a string filename, with or without extension; a ``.agg``
        extension is added if there is no suffix. Search path:

        * the current directory
        * :attr:`user_dir` (``~/.aggregate``)
        * :attr:`default_dir` (installed)

        :param fn: database file name (with or without ``.agg`` suffix).
        """

        p = Path(fn)
        if p.suffix == '':
            p = p.with_suffix('.agg')
        if p.exists():
            db_path = p
        elif (self.user_dir / p).exists():
            db_path = self.user_dir / p
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
            self._interpret_program(program)
            n = len(self._knowledge) - n
            logger.info('Database %s read into knowledge, adding %d entries.', fn, n)

    def __getitem__(self, item):
        """
        Look up a parsed program in the knowledge base.

        Pure lookup: returns the stored :class:`ParsedProgram` recipe.
        **The ``object`` field is always ``None`` on the returned
        ParsedProgram** — :meth:`__getitem__` does not construct the
        object. Use :meth:`__call__` / :meth:`build` /
        :meth:`build_many` (which run :meth:`_factory` after the lookup)
        when you want a live Aggregate / Severity / Portfolio /
        Distortion instance.

        Parameters
        ----------
        item : str or tuple
            ``'name'`` looks up by name across all kinds (must be unique
            across kinds, else ``KeyError``). ``(kind, name)`` is the
            unambiguous form.

        Returns
        -------
        ParsedProgram
            With ``kind`` / ``name`` / ``spec`` / ``program`` populated
            from the knowledge frame and ``object=None``.

        Raises
        ------
        KeyError
            If the lookup matches zero or more than one entry.

        See Also
        --------
        __call__ : the user-facing entry that also constructs the object.
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
        n = len(self._knowledge)
        if n == 0 and self.databases:
            kn_line = (
                'knowledge          0 loaded '
                '(access .knowledge to read configured database(s))'
            )
        else:
            kn_line = f'knowledge          {n} programs'
        return (
            f'Underwriter        {self.name}\n'
            f'version            {self.version}\n'
            f'{kn_line}\n'
            f'update             {self.update}\n'
            f'log2               {self.log2}\n'
            f'debug              {self.debug}\n'
            f'validation_eps     {VALIDATION_EPS}\n'
            f'user dir           {self._format_dir(self.user_dir)}\n'
            f'default dir        {self._format_dir(self.default_dir)}\n'
            f'browse             call .discover(regex) to list knowledge entries'
        )

    def _factory(self, parsed):
        """
        Internal: construct the object described by a :class:`ParsedProgram`.

        Portfolio construction needs ``self`` (passed as ``uw``), which is why
        this is not a staticmethod.

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

    @property
    def test_suite_file(self):
        """Path to the bundled test suite ``.agg`` file, or ``None`` if not present."""
        f = self.default_dir / TEST_SUITE_FILENAME
        return f if f.exists() else None

    def _build_work(self, portfolio_program, log2=0, bs=0, update=None, **kwargs):
        """
        Internal: parse → factory (without smart-update). Used by :meth:`build_many`.

        Tries a name lookup in the knowledge first; falls back to parsing the
        program. If ``update`` is True, calls each constructed object's
        ``.update(log2, bs, **kwargs)`` with the *literal* log2/bs (no bucket
        inference — that lives in :meth:`build_many`).

        :param portfolio_program: a DecL program (str), or the name of a
            previously-built object in the knowledge base.
        :param log2: passed verbatim to each object's ``update``.
        :param bs: passed verbatim to each object's ``update``.
        :param update: override the class-level ``self.update`` default.
        :param kwargs: passed to each created object's ``update`` method.
        :return: list of :class:`ParsedProgram` (one per top-level declaration).
        """
        if update is None:
            update = self.update
        if update is True and log2 == 0:
            log2 = self.log2

        # first see if portfolio_program refers to a built-in object
        try:
            answer = self[portfolio_program]
        except (LookupError, TypeError):
            logger.debug('underwriter._build_work | object not found, processing as a program.')
        else:
            logger.debug('underwriter._build_work | %s object found.', answer.kind)
            answer = self._factory(answer)
            if update:
                answer.object.update(log2, bs, **kwargs)
            return [answer]

        # not a built-in reference — parse and factory each line
        irv = self._interpret_program(portfolio_program)
        rv = []
        for answer in irv:
            answer = self._factory(answer)
            if answer.object is not None:
                # this can fail for named mixed severities, which can only be
                # created in the context of an agg — that behaviour is useful
                # for named severities, hence:
                if update:
                    update_method = getattr(answer.object, 'update', None)
                    if update_method is not None:
                        update_method(log2, bs, **kwargs)
            rv.append(answer)

        if not rv:
            logger.log(WL, 'Program did not contain any output')
        else:
            logger.info('Program created %d objects.', len(rv))
        return rv

    def _interpret_program(self, portfolio_program):
        """
        Internal: preprocess and parse a program one line at a time, storing
        each parsed spec in the knowledge base. No objects are constructed.

        :param portfolio_program: the DecL program text.
        :return: list of :class:`ParsedProgram` (``object`` is ``None`` for each).
        """
        portfolio_program = self.lexer.preprocess(portfolio_program)
        rv = []
        for program_line in portfolio_program:
            logger.debug(program_line)
            try:
                kind, name, spec = self.parser.parse(self.lexer.tokenize(program_line))
            except ValueError as e:
                if isinstance(e.args[0], str):
                    logger.error(e)
                    raise
                t = e.args[0].type
                v = e.args[0].value
                i = e.args[0].index
                txt2 = program_line[0:i] + '>>>' + program_line[i:]
                logger.error('Parse error in input "%s"\nValue %s of type %s not expected',
                             txt2, v, t)
                raise
            else:
                logger.info('answer out: %s object %s parsed successfully...adding to knowledge',
                            kind, name)
                self._knowledge.loc[(kind, name), :] = [spec, program_line]
                rv.append(ParsedProgram(kind=kind, name=name, spec=spec, program=program_line))
        return rv

    def _safe_lookup(self, buildinid):
        """
        Internal: parser callback that looks up ``kind.name`` in the knowledge
        and returns a deepcopy of the spec.

        Different from :meth:`__getitem__` in that it splits the dotted id into
        ``(kind, name)`` and verifies the resulting entry has the expected
        kind.

        :param buildinid: a string in ``kind.name`` form.
        :return: deep-copied spec dict.
        """
        # allow for sev.WC.1 name
        kind, *name = buildinid.split('.')
        name = '.'.join(name)
        try:
            parsed = self[(kind, name)]
        except LookupError:
            logger.error('ERROR id %s.%s not found in the knowledge.', kind, name)
            raise
        logger.debug('UnderwritingParser.safe_lookup | retrieved %s.%s as type %s.%s',
                     kind, name, parsed.kind, parsed.name)
        if parsed.kind != kind:
            raise ValueError(f'Error: type of {name} is  {parsed.kind}, not expected {kind}')
        # don't want to pass back the original; changes would be reflected in the knowledge
        return deepcopy(parsed.spec)

    def build_many(self, program, update=None, log2=0, bs=0, recommend_p=RECOMMEND_P, **kwargs):
        """
        Parse a (possibly multi-output) DecL program, construct each object, and smart-update.

        Always returns the full ``list[ParsedProgram]`` regardless of count.
        Use :meth:`build` instead when you expect a single output.

        Smart-update logic: discrete severities pick ``bs=1`` with a log2 sized
        to the max possible loss; continuous ones call :meth:`recommend_bucket`;
        portfolios use :meth:`best_bucket`. ``note{}`` hints in the program
        can override these.

        :param program: a DecL program producing one or more top-level outputs.
        :param update: override the class-level ``self.update`` default.
        :param log2: 0 (default) estimates log2 for discrete severities and
            uses ``self.log2`` for everything else.
        :param bs: bucket size; 0 lets the object recommend one.
        :param recommend_p: passed to :meth:`recommend_bucket`; raise (closer
            to 1) for thick-tailed distributions.
        :param kwargs: passed to each ``update`` call. ``force_severity=True``
            is always applied.
        :return: list of :class:`ParsedProgram`, one per top-level output.
        """
        rv = self._build_work(program, update=False, force_severity=True)

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
                log2, bs, recommend_p, kwargs = _parse_note(
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
                log2, bs, recommend_p, kwargs = _parse_note(
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

        Primary user-facing entry point. Calls :meth:`build_many` and unwraps
        the single result. ``__call__`` delegates to ``build``.

        :param program: a DecL program producing exactly one top-level output.
        :param update: override the class-level ``self.update`` default.
        :param log2: 0 (default) estimates log2 for discrete severities and
            uses ``self.log2`` for everything else.
        :param bs: bucket size; 0 lets the object recommend one.
        :param recommend_p: passed to :meth:`recommend_bucket`; raise (closer
            to 1) for thick-tailed distributions.
        :param kwargs: passed to ``update`` (e.g. ``padding``). ``force_severity=True``
            is always applied.
        :return: the constructed Aggregate / Severity / Portfolio / Distortion.
        :raises ValueError: if the program produces zero or more than one
            top-level output. Use :meth:`build_many` for batched programs.
        :raises CannotBuild: if the spec parses but no top-level object can be
            built standalone (e.g. a named mixture severity, which can only
            live inside an :class:`Aggregate`). Use :meth:`build_many` to
            receive the :class:`ParsedProgram` instead.
        """
        rv = self.build_many(program, update=update, log2=log2, bs=bs,
                             recommend_p=recommend_p, **kwargs)
        if len(rv) != 1:
            raise ValueError(
                f'build() expects a single output, got {len(rv)}; '
                f'use build_many() for batched programs.'
            )
        answer = rv[0]
        if answer.object is None:
            raise CannotBuild(
                f'build() could not construct {answer.kind} {answer.name!r}: '
                f'spec parses but cannot be built standalone (typically a '
                f'mixture severity — wrap it in an Aggregate, or use '
                f'build_many() to receive the ParsedProgram).'
            )
        return answer.object

    def __call__(self, *args, **kwargs):
        """
        Build an object from a DecL program or a known name.

        Convenience alias for :meth:`build`: ``build(program)`` is the
        canonical way to turn a DecL string into a single live
        Aggregate / Severity / Portfolio / Distortion. ``program`` can
        be either:

        * a DecL source string (parsed, added to the knowledge base,
          constructed); or
        * the bare name of an entry already in the knowledge base
          (looked up, constructed; the original DecL source is *not*
          re-parsed).

        The lookup branch and the parse branch both end by calling
        :meth:`_factory` on each :class:`ParsedProgram`, so the
        returned object is always live (never ``None``).

        Three access patterns, contrasted
        ---------------------------------

        Given::

            from aggregate import build
            build('dist cc1 ccoc .25')        # registers cc1 in knowledge

        1. **Build by program text** — parse, register, construct,
           return the object::

                cc1 = build('dist cc1m ccoc .25')
                assert cc1 is not None                # the Distortion

        2. **Knowledge lookup, no construction** — returns the recipe
           only; ``object`` is ``None`` because :meth:`__getitem__`
           does not run the factory::

                entry = build['cc1m']                 # ParsedProgram
                assert entry.object is None           # *by design*
                assert entry.spec == {'name': 'ccoc', 'r': 0.25}

        3. **Build by name** — lookup in the knowledge **and**
           construct, just like (1) but with no parsing::

                cc1 = build('cc1')                    # the Distortion
                assert cc1 is not None

        :meth:`build_many` is the batched form of (1) / (3); it returns
        a list of :class:`ParsedProgram` and *does* populate
        ``object`` on each (factory runs as part of the build).

        Rationale
        ---------

        The knowledge base stores DecL specs (small, picklable), not
        live objects. Objects are constructed on demand for two
        reasons: (a) Portfolios need an :class:`Underwriter` reference
        which may differ between sessions, and (b) each ``build('cc1')``
        returns a *fresh* instance so calibration / ``update()``
        mutations don't bleed across callers.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :meth:`build`.

        Returns
        -------
        Aggregate, Severity, Portfolio, or Distortion
            The single constructed object.
        """
        return self.build(*args, **kwargs)

    def interpret_file(self, filename=None, *, where=''):
        """
        Parse every DecL program in a ``.agg`` or ``.csv`` file and return per-line error info.

        Useful for validating a new ``.agg`` file before installing it into
        :attr:`user_dir`. Unlike :meth:`read_database` (which aborts on the first
        parse error), this method collects errors for every line and returns a
        DataFrame with columns ``kind, error, name, output, preprocessed program,
        program``.

        The bundled test suite lives at :attr:`test_suite_file` — call
        ``interpret_file()`` with no arguments to run it.

        For ``.csv`` files, the first column is used as index. For ``.agg`` files,
        the text is preprocessed (``\\n\\tagg`` folded back into one line, comments
        stripped) and then split on newlines.

        :param filename: a string or :class:`Path`. When ``None`` (default),
            uses :attr:`test_suite_file`.
        :param where: regex filter on the DataFrame index; ``''`` means all rows.
        :return: DataFrame with one row per line of the input file.
        """
        if filename is None:
            filename = self.test_suite_file
        elif isinstance(filename, str):
            filename = Path(filename)
        if filename.suffix == '.csv':
            df = pd.read_csv(filename, index_col=0)
        elif filename.suffix == '.agg':
            txt = filename.read_text(encoding='utf-8')
            stxt = re.sub('\n\tagg', ' agg', txt, flags=re.MULTILINE)
            stxt = [i for i in stxt.split('\n') if len(i) and i[0] != '#']
            # Use the program name (second token of '<kind> <name> ...') as the
            # DataFrame index so a `where` regex can filter by name.
            names = [(line.split() + ['?'])[1] for line in stxt]
            df = pd.DataFrame({'program': stxt}, index=names)
        else:
            raise ValueError(f'File suffix must be .csv or .agg, not {filename.suffix}')
        if where != '':
            df = df.loc[df.index.astype(str).str.match(where)]
        # ensure the canonical One severity is present for any sev.One references
        self.build_many('sev One dsev [1]', update=False)

        ans = {}
        # detect a non-trivial change between preprocessed and input program
        def _changed(preprocessed, original):
            return 'same' if preprocessed.replace(' ', '') == original.replace(' ', '').replace('\t', '') else original

        # df has exactly one column (program text). Iterate index + first-column-by-position;
        # `program[0]` on a labeled pandas Series raises in modern pandas.
        for test_name, program_in in zip(df.index, df.iloc[:, 0]):
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
        n_errors = df_out.error.sum()
        if n_errors:
            logger.error('%d parse error(s) in %s', n_errors, filename)
        return df_out

    def help(self, regex):
        """
        Lookup help on methods and properties matching ``regex``.
        """
        agg_help(self, regex)

    def discover(self, regex='', kind='', plot=False, describe=False,
                 return_objects=False, **kwargs):
        """
        Match knowledge entries against ``regex`` (and optional ``kind``);
        optionally build, plot, and describe each.

        Default behavior (``plot=False, describe=False``) is a lightweight
        directory view — just filter the knowledge base by name and return
        the matching DataFrame. Pass ``plot=True`` or ``describe=True`` to
        build each match and visualize/describe it.

        Examples::

            build.discover()                        # all entries
            build.discover('^A\\.')                  # entries whose name starts with "A."
            build.discover('Dice', plot=True)        # build + plot
            build.discover('^B\\.', describe=True)   # build + qd describe

        :param regex: filter on the knowledge index (name); '' matches all.
        :param kind: optional filter ('agg', 'sev', 'port', 'distortion'); '' matches all.
        :param plot: build each match and call its ``.plot()``.
        :param describe: build each match and ``qd()`` its describe table.
        :param return_objects: when building, also return the list of built
            objects alongside the DataFrame.
        :param kwargs: passed to :meth:`build` for each match.
        :return: DataFrame of matches; ``(objects, DataFrame)`` if
            ``return_objects=True``.
        """
        # base frame: optionally restricted to a kind
        base = self.knowledge.droplevel('kind') if not kind else self.knowledge.loc[kind]
        # empty regex means "no filter" — return everything
        df = base.filter(regex=regex, axis=0).copy() if regex else base.copy()

        # asking for the built objects implies we must run the build loop
        do_build = plot or describe or return_objects

        if not do_build:
            # lightweight directory view; format the program column for readability
            bit = df[['program']].copy()
            bit['program'] = (bit['program']
                              .str.replace(r' note\{[^}]+\}', '', regex=True)
                              .str.replace(r' {2,}', ' ', regex=True))
            return bit.sort_index()

        # build + plot/describe path: augment df with summary statistics.
        # Initialize as object dtype to avoid pandas LossySetitemError on the
        # mixed-type .loc assignment below (modern pandas refuses to coerce a
        # non-bool result of explain_validation into a bool column).
        summary_cols = ['log2', 'bs', 'agg_m', 'agg_cv', 'agg_sd', 'agg_skew',
                        'emp_m', 'emp_cv', 'emp_sd', 'emp_skew', 'valid']
        for col in summary_cols:
            df[col] = pd.Series([None] * len(df), index=df.index, dtype=object)

        objects = []
        for n, row in df.iterrows():
            try:
                a = self.build(row.program, **kwargs)
            except NotImplementedError:
                logger.error('skipping %s...element not implemented', n)
                continue
            except CannotBuild as e:
                logger.warning('skipping %s — %s', n, e)
                continue
            objects.append(a)
            if describe:
                pp = getattr(a, 'pprogram', None)
                if pp is not None:
                    print(pp)
                # only Aggregate / Portfolio have a `.describe` table
                if hasattr(a, 'describe'):
                    qd(a)
                else:
                    print(repr(a))
                print('\n')
            if plot:
                # Delegate sizing to each class's own .plot default —
                # Aggregate/Severity/Portfolio/Distortion all set sensible
                # ones, and Distortion.plot in particular does not accept
                # `figsize` (would forward to ax.plot and crash).
                a.plot()
                print()
            df.loc[n, summary_cols] = _row_stats(a, summary_cols)

        if return_objects:
            return (objects[0] if len(objects) == 1 else objects), df
        return df

# Module-level singleton — the canonical user-facing entry point. Importable
# as `from aggregate import build`. `update=True` is why `build('agg ...')`
# auto-updates the constructed object's discrete distribution; pass
# `update=False` (or override at call time) to disable.
build = Underwriter(databases='test_suite', update=True, debug=False, log2=16)
# Sibling entry point for building several objects from one program text.
# Bound to the same singleton so `from aggregate import build_many` returns
# a DataFrame summary across all objects in the input.
build_many = build.build_many
# uncomment to create debug build, add to __init__.py
# debug_build = Underwriter(name='Debug', update=True, debug=True, log2=16)
