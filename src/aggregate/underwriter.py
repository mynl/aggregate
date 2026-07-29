from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from importlib.resources import files
import logging
import numbers
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from ._help import HelpMixin
from .config import (get_settings, reload_settings as _reload_settings,
                     write_default_config as _write_default_config,
                     describe_settings, config_path,
                     USER_DIR_NAME, PACKAGE_DATA_DIR, TEST_SUITE_FILENAME,
                     LIBRARY_FILENAME)
from .portfolio import Portfolio
from .distributions import Aggregate, Severity, PnL, BUCKET_SIZING_P
from .spectral import Distortion
from .constants import IgnoredDecLClauseWarning, ZeroPremiumCessionWarning
from .parser import UnderwritingLexer, UnderwritingParser, INHERIT_PREMIUM
from .recipe import Recipe
from .utilities import (qd, agg_help)

logger = logging.getLogger(__name__)

__all__ = ['Underwriter', 'build', 'build_many', 'CannotBuild']

# Sentinel for Underwriter.__init__ arguments that should fall back to the
# configured defaults (aggregate.config [build]). Distinct from None, which for
# ``databases`` still means "load nothing". The custom __repr__ keeps the
# argument readable in help/signatures (Jupyter ``?``, ``inspect.signature``)
# instead of showing ``<object object at 0x...>``.
class _Unset:
    """Singleton sentinel meaning "use the configured aggregate.config default"."""
    __slots__ = ()

    def __repr__(self):
        return '<config default>'


_UNSET = _Unset()


#: The non-loss spec keys a pure ``agg`` accepts and **ignores** (with an
#: :class:`~aggregate.constants.IgnoredDecLClauseWarning`): reinsurance
#: economics (ceded premium / commission), reinstatement schedules, the
#: variable-rating features and their decorated-layer indices, and the retro
#: rating clause. They need a P&L premium context to activate; a plain
#: Aggregate builds the loss structure only. The recipe base keeps the full
#: spec, so an ``agg.NAME`` reference inside a ``pnl`` / ``xpnl`` re-injects
#: them ([Reins-Economics-On-Agg-Ignore-Warn]).
_AGG_IGNORED_ECONOMICS_KEYS = (
    'occ_reins_premium', 'occ_reins_cede',
    'agg_reins_premium', 'agg_reins_cede',
    'occ_reins_reinst',
    'agg_reins_swing', 'agg_reins_swing_layer',
    'agg_reins_slide', 'agg_reins_slide_layer',
    'agg_reins_pc', 'agg_reins_pc_layer',
    'agg_reins_corridor', 'agg_reins_corridor_layer',
    'retro_terms',
)

#: Human clause names for the ignored-economics warning, in reporting order.
#: Each entry maps the spec keys that signal the clause to its DecL name.
_IGNORED_CLAUSE_NAMES = (
    (('occ_reins_premium', 'agg_reins_premium'),
     'ceded premium (deposit / rol / rate)'),
    (('occ_reins_cede', 'agg_reins_cede'), 'cede'),
    (('occ_reins_reinst',), 'reinstatements'),
    (('agg_reins_swing',), 'swing'),
    (('agg_reins_slide',), 'slide'),
    (('agg_reins_pc',), 'pc'),
    (('agg_reins_corridor',), 'corridor'),
    (('retro_terms',), 'retro'),
)


def ignored_clauses_message(name, spec):
    """The ignored-economics message for a plain ``agg``, or ``''`` if none.

    Names each present-but-unusable clause (ceded premium, cede,
    reinstatements, a variable-rating feature, retro) and the remedy: fold the
    agg into a ``pnl`` / ``xpnl`` by reference to activate the economics. One
    source of truth for the wording -- emitted as an
    :class:`~aggregate.constants.IgnoredDecLClauseWarning` by the factory and
    replayed by ``PnL.construction_explanation``
    ([Reins-Economics-On-Agg-Ignore-Warn]).
    """
    clauses = [label for keys, label in _IGNORED_CLAUSE_NAMES
               if any(k in spec for k in keys)]
    if not clauses:
        return ''
    plural = 's' if len(clauses) > 1 else ''
    return (f"{name}: ignoring the {', '.join(clauses)} clause{plural} -- a "
            "plain 'agg' has no premium context, so only the loss structure "
            f"builds. The stored program keeps the full declaration: fold it "
            f"into a 'pnl' / 'xpnl' by reference (e.g. \"pnl {name}_pnl "
            f"<premium> premium less agg.{name}\") to activate the economics.")


# Write order for to_agg: .agg files load sequentially and named references
# (sev.X, agg.X, dist.X) must resolve as each line is parsed, so a definition
# must precede anything that references it. Severities and distortions come
# before the aggregates that use them; aggregates before portfolios.
_KIND_WRITE_ORDER = {'sev': 0, 'distortion': 1, 'agg': 2, 'bvagg': 3, 'port': 4}


def _entry_to_decl(r):
    """Render one :class:`~aggregate.recipe.Recipe` to canonical DecL for :meth:`Underwriter.to_agg`.

    Uses :func:`aggregate.decl_writer.spec_to_decl` (the parser's inverse) so the
    exported ``.agg`` is canonical and re-loads cleanly. Falls back to the stored
    verbatim ``program`` only for a ``minimum`` / ``mixture`` combinator
    distortion, whose child references are not retained on the spec and so cannot
    be unparsed.
    """
    from .decl_writer import spec_to_decl
    try:
        return spec_to_decl(r.spec, r.kind, r.name)
    except NotImplementedError:
        return r.program


# Allow-list of build/update knobs a ``hints{...}`` clause may set. Anything
# else is warned about and dropped (never crashes the build).
_HINT_KEYS = {
    'log2', 'bs', 'padding', 'normalize', 'bucket_sizing_p',
    'sev_calc', 'discretization_calc', 'force_severity', 'x_min', 'x_max',
}


def _coerce_hint_value(v):
    """Infer a Python type for a ``hints{...}`` value string.

    Tries, in order: ``int``, ``float``, an ``a/b`` fraction (so ``bs=1/64``
    works), the literals ``True`` / ``False``, else the raw stripped string.

    Parameters
    ----------
    v : str
        The right-hand side of a ``key=value`` hint.

    Returns
    -------
    int, float, bool, or str
        The coerced value.
    """
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    m = re.fullmatch(r'([-+0-9.eE]+)\s*/\s*([-+0-9.eE]+)', v)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    if v == 'True':
        return True
    if v == 'False':
        return False
    return v


def _parse_hints(txt):
    """Parse a ``hints{...}`` body (``key=value; key=value``) into a typed dict.

    Value typing is by :func:`_coerce_hint_value` (int / float / ``a/b``
    fraction / bool / str). Unknown keys (not in :data:`_HINT_KEYS`) are warned
    about and dropped; a duplicate key warns and the last value wins; a clause
    that is not exactly one ``key=value`` warns and is skipped. This function
    never raises -- a malformed hint degrades to a warning, not a crash.

    Parameters
    ----------
    txt : str
        The raw inner text of a ``hints{...}`` clause (empty string if none).

    Returns
    -------
    dict
        Recognised settings as ``{key: typed_value}``.
    """
    kw = {}
    for chunk in txt.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.count('=') != 1:
            logger.warning("hints: ignoring malformed clause %r (expected a "
                           "single key=value)", chunk)
            continue
        k, v = (s.strip() for s in chunk.split('='))
        if k not in _HINT_KEYS:
            logger.warning("hints: unknown key %r ignored (known: %s)",
                           k, sorted(_HINT_KEYS))
            continue
        if k in kw:
            logger.warning("hints: duplicate key %r; last value wins", k)
        kw[k] = _coerce_hint_value(v)
    return kw


# Settings keys that used to be (mis)read out of ``note{...}``; used only to
# emit a deprecation warning now that notes are pure text.
_NOTE_SETTINGS_RE = re.compile(r'\b(?:log2|bs|padding|normalize|bucket_sizing_p)\s*=')

#: The DecL trailer keys, in render order. ``note``/``hints`` are always on a
#: spec; ``tags``/``doc`` only when written (the captured spec snapshot compares
#: key sets exactly, so they cannot be unconditional). See dev/plan-meta-data.md.
TRAILER_KEYS = ('note', 'tags', 'hints', 'doc')


def _trailer_meta(spec):
    """Extract the DecL trailer metadata from a spec, skipping absent keys."""
    return {k: spec[k] for k in TRAILER_KEYS if spec.get(k)}


def _resolve_hints(spec, log2, bs, bucket_sizing_p, kwargs):
    """Merge a spec's ``hints{...}`` build settings under *caller-wins* rules.

    Parses ``spec['hints']`` via :func:`_parse_hints`, then fills only the
    settings the caller left at their sentinel default: ``log2`` when ``log2 ==
    0``, ``bs`` when ``bs == 0``, ``bucket_sizing_p`` when it equals
    the configured ``discretization.bucket_sizing_p``. Any remaining recognised keys are
    added to ``kwargs`` with :meth:`dict.setdefault`, so an explicit caller
    value always wins. A deprecation warning fires if the (now pure-text)
    ``note`` still looks like it carries ``key=value`` settings.

    Parameters
    ----------
    spec : dict
        A parsed object spec; reads ``spec['note']`` and ``spec['hints']``.
    log2, bs, bucket_sizing_p : int, float, float
        Caller-supplied build settings (sentinels ``0`` / ``0`` /
        ``BUCKET_SIZING_P`` mean "unset").
    kwargs : dict
        Pass-through update kwargs (mutated in place via ``setdefault``).

    Returns
    -------
    (log2, bs, bucket_sizing_p, kwargs) : tuple
        Ready for ``Aggregate`` / ``Portfolio`` / ``BivariateAggregate``
        update.
    """
    note = spec.get('note', '') or ''
    if _NOTE_SETTINGS_RE.search(note):
        logger.warning(
            "note{...} no longer sets build options; move 'key=value' "
            "settings into a hints{...} clause. The note is now treated as "
            "pure text.")
    hints = _parse_hints(spec.get('hints', '') or '')
    # log2 / bs / bucket_sizing_p are passed explicitly at the update() call, so
    # they must NEVER also leak through ``kwargs`` (that would duplicate the
    # keyword). Take the hint value only when the caller left the sentinel
    # default, then drop the key from the pass-through either way.
    if log2 == 0 and 'log2' in hints:
        log2 = int(hints['log2'])
    if bs == 0 and 'bs' in hints:
        bs = hints['bs']
    if bucket_sizing_p == BUCKET_SIZING_P and 'bucket_sizing_p' in hints:
        bucket_sizing_p = hints['bucket_sizing_p']
    for k in ('log2', 'bs', 'bucket_sizing_p'):
        hints.pop(k, None)
    for k, v in hints.items():
        kwargs.setdefault(k, v)
    return log2, bs, bucket_sizing_p, kwargs


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
                   actual_m=a.actual_m, actual_cv=a.actual_cv,
                   actual_sd=a.actual_sd, actual_skew=a.actual_skew,
                   emp_m=a.est_m, emp_cv=a.est_cv, emp_sd=a.est_sd,
                   emp_skew=a.est_skew,
                   valid=a.validation_description)
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
            row.update(actual_m=m, actual_sd=sd, actual_cv=cv, actual_skew=skew)
        except Exception as e:
            logger.debug('Severity %s: moms unavailable (%s)', getattr(a, 'name', '?'), e)
    # Distortion: nothing applies — all fields left as None
    return [row[col] for col in summary_cols]


class CannotBuild(ValueError):
    """Raised by :meth:`Underwriter.build` when a parsed spec produces no top-level object.

    Typically the named-mixed-severity case: a ``sev`` declaration with ``wts``
    can only live inside an :class:`Aggregate`, not standalone. Use
    :meth:`Underwriter.build_many` to receive the :class:`~aggregate.recipe.Recipe`
    instead, then inspect ``.spec`` directly.

    Subclass of :class:`ValueError` so existing broad ``except ValueError``
    callers continue to catch it.
    """


class Underwriter(HelpMixin):
    """
    Manage the creation of Aggregate, Severity, Portfolio, and Distortion objects.

    Maintains a database of named DecL declarations — the **recipe base** — and
    exposes the user-facing :meth:`build` entry point that parses a DecL program
    and constructs the corresponding object(s).

    Responsibilities:

    - Read DecL programs from ``.agg`` files (:meth:`load`) and write a
      selection back out (:meth:`to_agg`).
    - Bridge to the parser (`UnderwritingLexer` / `UnderwritingParser`).
    - Safe lookup of named programs from the recipe base for the parser.

    Every parsed declaration has a *kind* (one of ``'sev'``, ``'agg'``,
    ``'port'``, ``'bvagg'``, ``'pnl'``, ``'xpnl'``, ``'distortion'``, or
    ``'expr'``) and a *name*. Parsing produces a
    :class:`~aggregate.recipe.Recipe` holding the kind, name, dict spec, source
    program, provenance, its documentation (``note`` / ``tags`` / ``hints`` /
    ``doc``, derived from the spec), and, once :meth:`_factory` runs, the
    constructed object. ``'expr'`` is the odd one out: a bare expression is an
    answer rather than a declaration, so it evaluates to its value and is never
    stored in the recipe base.

    The recipe base is a flat in-memory union, ``(kind, name) -> Recipe``,
    stored in ``self._recipes`` as a plain dict. It is fed by the loaded
    ``.agg`` files and by in-session ``build(...)`` calls; there is no "active"
    database. ``(kind, name)`` is the unique key — a later entry with the same
    key overrides an earlier one (last load / last build wins). Each entry
    carries a ``source`` provenance tag (its originating file ``Path``, or
    ``'session'`` for an in-session build). The :attr:`recipes` property exposes
    this as a ``(kind, name)``-indexed DataFrame, built on demand;
    :meth:`recipe` returns one entry by name.

    The loading surface is: construct with a ``databases=`` request, then
    :meth:`load` (read more), :meth:`resolve_databases` (preview a request),
    :meth:`available_databases` (discover what is on disk), :attr:`databases`
    (the resolved file ``Path``\\ s actually loaded) / :attr:`recipes`
    (inspect), :meth:`to_agg` (save), and :meth:`reload` (reset to as-created).
    """

    def __init__(self, *, name='Rory', databases=None, update=_UNSET, log2=_UNSET, debug=False):
        """
        Create an underwriter object. The underwriter is the interface to the recipe base
        of the aggregate system. It is the interface to the parser and the interpreter, and
        to the database of curves, portfolios and aggregates.

        All arguments are **keyword-only**. This prevents the easy mistake of
        ``Underwriter('examples')``, which previously bound the first
        positional to ``name`` and silently *named* the underwriter after the
        database you meant to load. Use ``Underwriter(databases='examples')``.

        ``update`` and ``log2`` default to the configured values in
        :mod:`aggregate.config` (the ``[build]`` section); pass an explicit
        value to override. This is what unifies the historical 10-vs-16 ``log2``
        split between a bare ``Underwriter`` and the module-level ``build``.
        ``databases`` is **not** config-driven: a bare ``Underwriter()`` loads
        **nothing** (``databases=None``). The module-level ``build`` is the one
        that loads the configured ``build.databases`` (``examples`` by
        default), by passing it explicitly — so ``config.toml`` still controls
        what ``build`` knows, while ad-hoc underwriters start empty.

        :param name: name of underwriter. Defaults to Rory, after Rory Cline, the best underwriter
            I know and a supporter of an analytic approach to underwriting.
        :param databases: the load *request* — what to read on first access.
            ``None`` (the default) loads nothing. ``'default'`` loads the bundled
            ``*.agg`` files, ``'user'`` loads ``~/.aggregate`` ``*.agg`` files,
            ``'all'`` loads both. Any other string is a file path or glob,
            resolved against the search path cwd -> user_dir -> default_dir (see
            :meth:`load`). An iterable of such entries is also valid. The request
            is stored privately as ``self._request``; the resolved files actually
            read appear in :attr:`databases`.
        :param update: if True, update constructed objects. Unset uses the
            configured ``build.update``.
        :param log2: log2 of number of buckets in discrete representation. 10 is
            1024 buckets. Unset uses the configured ``build.log2``.
        :param debug: if True, print debug messages.
        """

        build_settings = get_settings().build
        self.name = name
        self.update = build_settings.update if update is _UNSET else update
        if log2 is _UNSET:
            log2 = build_settings.log2
        if log2 <= 0:
            raise ValueError(
                'log2 must be > 0. The number of buckets used equals 2**log2.')
        self.log2 = log2
        self.debug = debug
        self._lexer = None
        self._parser = None

        # The load request (what to read); resolved + read lazily on first
        # access to .recipes, or eagerly via .load(). Default None -> load
        # nothing (bare underwriters start empty). The module-level ``build``
        # passes the configured ``build.databases`` explicitly (see below).
        self._request = databases

        # do not read in until needed for faster loading
        self._default_dir = None
        self._user_dir = None
        # Recipe base: a flat dict {(kind, name): Recipe}. The DataFrame view
        # is built on demand by the `recipes` property.
        self._recipes: dict[tuple, Recipe] = {}
        # `databases` (public) reports the resolved file Paths actually loaded;
        # `_loaded` is the honest "configured request has been read" flag.
        self.databases: list = []
        self._loaded = False

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
        ``uw.load('my_curves')``.  List user databases::

            list(uw.user_dir.glob('*.agg'))
        """
        if self._user_dir is None:
            self._user_dir = Path.home() / USER_DIR_NAME
            self._user_dir.mkdir(parents=True, exist_ok=True)
        return self._user_dir

    # Glob metacharacters that mark a request entry as a pattern rather than a
    # literal filename.
    _GLOB_CHARS = ('*', '?', '[')

    def _search_dirs(self):
        """The literal/glob search path: cwd, then user_dir, then default_dir."""
        return [Path.cwd(), self.user_dir, self.default_dir]

    def _resolve_database_request(self, request, *, explicit):
        """Normalise a load request into an ordered list of ``.agg`` ``Path``\\ s.

        The single resolver shared by construction, :meth:`load`,
        :meth:`reload`, and :meth:`resolve_databases`. The reserved collection
        names are simply predefined globs, so there is one mechanism rather than
        a special case per token.

        Resolution rules:

        - ``None`` / ``[]`` -> no files.
        - ``'default'`` -> ``<default_dir>/*.agg``; ``'user'`` ->
          ``<user_dir>/*.agg``; ``'all'`` -> both.
        - An entry containing a directory separator (or an absolute path, or a
          leading ``~``) is used as given; the search dirs are not consulted.
        - A **glob** entry (contains ``*``, ``?`` or ``[``) is matched against
          all three search dirs and the matches are **unioned** (cwd included);
          a bare pattern with no suffix globs ``<pattern>.agg``. An empty match
          warns ("load whatever is there").
        - A **literal** entry (no glob metacharacters) gets a ``.agg`` suffix if
          it has none and is resolved **first-match-wins** across the search
          dirs (the nearest file). A missing literal **raises**
          :class:`FileNotFoundError` when ``explicit`` (the caller named one
          file in a :meth:`load` call), else **warns** (a configured entry).

        Parameters
        ----------
        request : None, str, pathlib.Path, or iterable
            The load specification.
        explicit : bool
            True for a user-supplied :meth:`load` request (missing literal
            raises); False for the configured/lazy request (missing literal
            warns).

        Returns
        -------
        list[pathlib.Path]
            Files to read, in resolution order, de-duplicated by resolved path.
        """
        if request is None:
            return []
        if isinstance(request, (str, Path)):
            request = [request]

        paths: list = []
        seen: set = set()

        def _add(p):
            p = Path(p)
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                paths.append(p)

        def _add_glob(pattern_dirs, pattern_name, *, what):
            matches = []
            for d in pattern_dirs:
                matches.extend(sorted(d.glob(pattern_name)))
            if not matches:
                logger.warning('Database request %s matched no files. Ignoring.', what)
            for m in matches:
                _add(m)

        for entry in request:
            entry_str = str(entry)
            # Reserved collection names -> predefined globs.
            if entry_str == 'default':
                _add_glob([self.default_dir], '*.agg', what="'default'")
                continue
            if entry_str == 'user':
                _add_glob([self.user_dir], '*.agg', what="'user'")
                continue
            if entry_str == 'all':
                _add_glob([self.default_dir, self.user_dir], '*.agg', what="'all'")
                continue

            p = Path(entry_str).expanduser()
            is_glob = any(c in entry_str for c in self._GLOB_CHARS)
            has_dir = p.is_absolute() or len(p.parts) > 1
            if p.suffix == '':
                p = p.with_suffix('.agg')

            if has_dir:
                # Used as given; search dirs are not consulted.
                if is_glob:
                    _add_glob([p.parent], p.name, what=repr(entry_str))
                elif p.exists():
                    _add(p)
                elif explicit:
                    raise FileNotFoundError(f'Database {entry_str!r} not found.')
                else:
                    logger.warning('Database %r not found. Ignoring.', entry_str)
            elif is_glob:
                # Glob across the search path; union the matches.
                _add_glob(self._search_dirs(), p.name, what=repr(entry_str))
            else:
                # Literal name: first match wins across the search path.
                found = next((d / p.name for d in self._search_dirs()
                              if (d / p.name).exists()), None)
                if found is not None:
                    _add(found)
                elif explicit:
                    raise FileNotFoundError(
                        f'Database {entry_str!r} not found on the search path '
                        f'(cwd, {self.user_dir}, {self.default_dir}).')
                else:
                    logger.warning('Database %r not found on the search path. Ignoring.',
                                   entry_str)

        return paths

    def load(self, request=None):
        """
        Resolve a request and read the matching ``.agg`` files into the recipe base.

        The single load verb. ``request=None`` reads the **configured** request
        (``self._request``, from the constructor or ``config.build.databases``)
        exactly once — this is the lazy path that :attr:`recipes` triggers on
        first access. A given ``request`` (filename, glob, collection name, or
        list thereof) is resolved and read **additively**: its files are added
        to the recipe base and appended to :attr:`databases`.

        Reading "one or more" files is just a glob or a list. The search path
        and ``.agg`` suffixing rules are documented on
        :meth:`_resolve_database_request`.

        Error policy: a **literal** file named in an explicit ``load(path)`` call
        that does not exist raises :class:`FileNotFoundError`; the configured
        request (``load()``) and any glob that matches nothing only warn.

        Parameters
        ----------
        request : None, str, pathlib.Path, or iterable, optional
            What to load. ``None`` (default) loads the configured request once.

        Returns
        -------
        list[pathlib.Path]
            The files read by this call (empty if the configured request was
            already loaded).
        """
        if request is None:
            if self._loaded:
                return []
            paths = self._resolve_database_request(self._request, explicit=False)
            self._loaded = True
        else:
            paths = self._resolve_database_request(request, explicit=True)

        read = []
        for p in paths:
            if self._read_file(p):
                read.append(p)
        return read

    def _read_file(self, path):
        """Read and interpret one ``.agg`` file; record it in :attr:`databases`.

        Tags every entry with ``source=path`` (provenance) and appends ``path``
        to :attr:`databases`. Whitespace/continuation handling is the lexer's
        job (:meth:`UnderwritingLexer.preprocess`); this method does no text
        munging of its own.

        Parameters
        ----------
        path : pathlib.Path
            The ``.agg`` file to read.

        Returns
        -------
        bool
            True if the file was read; False if it could not be opened.
        """
        path = Path(path)
        try:
            program = path.read_text(encoding='utf-8')
        except OSError:
            logger.exception('Error reading requested database %s. Ignoring.', path.name)
            return False
        logger.info('Reading database %s...', path)
        n = len(self._recipes)
        self._interpret_program(program, source=path)
        added = len(self._recipes) - n
        logger.info('Database %s read, adding %d recipes.', path.name, added)
        self.databases.append(path)
        if path.name == LIBRARY_FILENAME:
            self._check_library_names_unique(path)
        return True

    def _check_library_names_unique(self, path):
        """Enforce globally-unique names in the shipped library.

        The recipe base is keyed ``(kind, name)``, so ``sev Pareto`` and
        ``agg Pareto`` can legally coexist -- and did, across the three files
        ``library.agg`` replaced. The shipped library gives that up on purpose,
        so ``build('X')`` and ``build.recipe('X')`` always mean the same entry
        and no ``kind=`` disambiguator is ever needed. Checked at load so the
        invariant cannot rot silently as entries are added.

        Scoped to ``library.agg`` only: ``decl-testers.agg`` and
        ``_test_suite.agg`` are regression corpora, not a curated surface, and
        they reuse names across kinds deliberately.
        """
        from collections import Counter
        names = Counter(name for (_kind, name), r in self._recipes.items()
                        if getattr(r, 'source', None) == path)
        dupes = sorted(n for n, c in names.items() if c > 1)
        if dupes:
            raise ValueError(
                f'{path.name}: names must be unique across kinds, but '
                f'{dupes} appear under more than one kind. Rename one, or move '
                f'the entry to a corpus file where reuse is allowed.')

    def reload(self):
        """
        Reset the underwriter to its as-created state and re-read the configured request.

        Clears the recipe base, the :attr:`databases` list, and the
        ``_loaded`` flag, restores the original load request, then re-resolves
        and re-reads it from disk. This **drops** any ad-hoc ``load(...)``-ed
        files and any in-session ``build(...)`` entries — the object returns to
        exactly its constructor-time state. Picks up on-disk edits to the
        configured ``.agg`` files. Backs config reload via
        :func:`_refresh_default_underwriter`.

        Returns
        -------
        list[pathlib.Path]
            The files re-read.
        """
        self._recipes = {}
        self.databases = []
        self._loaded = False
        return self.load()

    def resolve_databases(self, request=None):
        """
        Preview the files a request *would* load, without reading them (dry run).

        Answers "if I ask to load ``xxx``, what will I get?". ``request=None``
        previews the configured request. Resolution is lenient here (a missing
        literal warns and is omitted rather than raising), so it is safe to use
        for exploration; for files that exist it matches what a subsequent
        :meth:`load` reads.

        Parameters
        ----------
        request : None, str, pathlib.Path, or iterable, optional
            The request to preview; ``None`` previews the configured request.

        Returns
        -------
        list[pathlib.Path]
            The files the request resolves to.
        """
        req = self._request if request is None else request
        return self._resolve_database_request(req, explicit=False)

    def available_databases(self):
        """
        Discover the ``.agg`` files present on the search path (cwd / user / default).

        Distinct from :meth:`resolve_databases` ("what *would* this request
        load") — this is "what *could* I load". A discovery view of the files on
        disk, regardless of any request.

        Returns
        -------
        pandas.DataFrame
            One row per ``.agg`` file found, columns ``name`` (file stem),
            ``where`` (``'cwd'`` / ``'user'`` / ``'default'``), and ``path``
            (the full path). The same name may appear in more than one location.
        """
        rows = []
        for where, d in (('cwd', Path.cwd()), ('user', self.user_dir),
                         ('default', self.default_dir)):
            for f in sorted(d.glob('*.agg')):
                rows.append({'name': f.stem, 'where': where, 'path': str(f)})
        return pd.DataFrame(rows, columns=['name', 'where', 'path'])

    def __getitem__(self, item):
        """
        Subscript form of :meth:`recipe`, the canonical lookup.

        ``uw[name]`` is ``uw.recipe(name)``; ``uw[kind, name]`` is
        ``uw.recipe(name, kind)``. Same result, same errors. Reach for the
        subscript when the code around it already reads like a store; reach for
        :meth:`recipe` in prose, when you want ``kind=`` named, or when you want
        the method to be discoverable through :meth:`help` and tab completion.

        Parameters
        ----------
        item : str or tuple
            ``'name'``, or the unambiguous ``(kind, name)`` pair.

        Returns
        -------
        aggregate.recipe.Recipe
            See :meth:`recipe`. ``object`` is ``None``: this is a lookup, not
            a build.

        Raises
        ------
        KeyError
            No such entry, or the name is ambiguous across kinds.

        See Also
        --------
        recipe : the canonical form of this lookup.
        __call__ : the user-facing entry that also constructs the object.
        """
        if isinstance(item, str):
            return self.recipe(item)
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(
                    f'a tuple subscript must be (kind, name), not {len(item)} items.')
            kind, name = item
            return self.recipe(name, kind)
        raise ValueError(
            f'item must be a str (name of object) or tuple (kind, name), not {type(item)}.')

    @staticmethod
    def _format_dir(path: Path) -> str:
        """Render an absolute path as ``~/...`` if it lives under the user's home."""
        try:
            return f'~/{path.resolve().relative_to(Path.home())}'
        except ValueError:
            return str(path.resolve())

    def _format_source(self, source) -> str:
        """Render a recipe's ``source`` provenance for display.

        The dict store tags every entry with its origin: ``'session'`` (an
        in-session build) or the resolved :class:`~pathlib.Path` of the ``.agg``
        file it was read from. Full paths print badly in the :attr:`recipes`
        DataFrame, so collapse them by location:

        - a **built-in** database (under :attr:`default_dir`) shows just its
          name — no directory, no ``.agg`` suffix (e.g. ``test_suite``);
        - a **user** database (under :attr:`user_dir`, ``~/.aggregate``) shows
          ``~/<name>`` with the suffix dropped, keeping any sub-directory
          (e.g. ``~/mylib``, ``~/sub/mylib``);
        - any **other** loaded file shows its full path unchanged.

        Non-path sources (e.g. the ``'session'`` sentinel) pass through as-is.

        Parameters
        ----------
        source : pathlib.Path or str
            The stored provenance value.

        Returns
        -------
        str
            The display form.
        """
        if not isinstance(source, Path):
            return source
        p = source.resolve()
        try:
            p.relative_to(self.default_dir.resolve())
            return p.stem
        except ValueError:
            pass
        try:
            rel = p.relative_to(self.user_dir.resolve())
            return f'~/{rel.with_suffix("")}'
        except ValueError:
            return str(p)

    def _format_request(self) -> str:
        """Render the load *request* (``self._request``) for :meth:`__repr__`.

        The request is what the constructor was *asked* to load (``None``, a
        keyword like ``'all'``, a path/glob, or an iterable of these); the
        resolved files actually read live in :attr:`databases`. ``None`` loads
        nothing, which is the bare-underwriter default.
        """
        req = self._request
        if req is None:
            return 'none (loads nothing)'
        if isinstance(req, str):
            return req
        try:
            return ', '.join(str(r) for r in req)
        except TypeError:
            return str(req)

    def _config_line(self) -> str:
        """One-line summary of the active config file and override counts.

        Reports whether ``~/.aggregate/config.toml`` (or an ``AGGREGATE_CONFIG``
        override) is loaded, plus how many settings differ from the built-in
        defaults via the file and via the environment. Backs the ``config``
        line in :meth:`__repr__`.
        """
        p = config_path()
        rows = describe_settings(get_settings())
        n_cfg = sum(1 for _, _, src in rows if src == 'config')
        n_env = sum(1 for _, _, src in rows if src == 'env')
        if p is None:
            base = '(disabled via AGGREGATE_CONFIG=none)'
        elif Path(p).exists():
            plural = '' if n_cfg == 1 else 's'
            base = f'{self._format_dir(Path(p))} (loaded, {n_cfg} override{plural})'
        else:
            base = '(none - defaults)'
        if n_env:
            plural = '' if n_env == 1 else 's'
            base += f', {n_env} env override{plural}'
        return base

    def __repr__(self):
        # Count entries from the dict directly — avoid self.recipes here, which
        # would trigger a database read. _loaded is the honest flag.
        n = len(self._recipes)
        if not self._loaded and self._request:
            kn_line = (
                'recipes            0 loaded '
                '(access .recipes to read configured database(s))'
            )
        else:
            kn_line = f'recipes            {n} programs'
        return (
            f'Underwriter        {self.name}\n'
            f'version            {self.version}\n'
            f'{kn_line}\n'
            f'requested          {self._format_request()}\n'
            f'update             {self.update}\n'
            f'log2               {self.log2}\n'
            f'debug              {self.debug}\n'
            f'validation_eps     {get_settings().validation.eps}\n'
            f'config             {self._config_line()}\n'
            f'user dir           {self._format_dir(self.user_dir)}\n'
            f'default dir        {self._format_dir(self.default_dir)}\n'
            f'browse             call .discover(regex) to list recipes'
        )

    @staticmethod
    def _resolve_reins_economics(spec, gross_premium):
        """Resolve per-layer ceded-premium / cede clauses to per-side economics.

        Pops the ``occ_reins_premium`` / ``occ_reins_cede`` /
        ``agg_reins_premium`` / ``agg_reins_cede`` keys from ``spec`` (so the
        inner :class:`Aggregate` sees only the loss structure) and resolves each
        layer's premium to currency **at the layer's placement share** -- all
        forms are quoted at 100% placement and scaled down by the fraction
        placed (``share``): ``deposit`` is ``share x amount``, ``rol`` is
        ``share x rol x limit``, ``rate`` is ``share x rate x gross_premium``.
        The ceding commission is ``cede x ceded_premium`` per layer -- a fraction
        of the *placed* premium, so it scales automatically. Returns
        ``{'gross', 'ceded', 'pc_occ', 'pc_agg', 'c_occ', 'c_agg'}`` for the GCN
        view, or ``None`` when no ceded-premium clause is present.

        Parameters
        ----------
        spec : dict
            The pnl spec (mutated: the four reins-economics keys are popped).
        gross_premium : float
            The pnl's stated premium (the gross premium; the ``rate`` base).
        """
        keys = ('occ_reins_premium', 'occ_reins_cede',
                'agg_reins_premium', 'agg_reins_cede')
        if not any(k in spec for k in keys):
            return None
        pg = float(gross_premium)

        def side(which):
            layers = spec.get(f'{which}_reins') or []
            prem = spec.pop(f'{which}_reins_premium', None)
            cede = spec.pop(f'{which}_reins_cede', None)
            pc = comm = 0.0
            if prem is None:
                return 0.0, 0.0
            for i, p in enumerate(prem):
                if p is None:
                    continue
                basis, val = p
                share, limit, _attach = layers[i]
                # All premium forms are quoted at 100% placement and scaled by
                # the fraction placed (``share``). ``rol`` already carries the
                # share in ``share x rol x limit``; ``deposit`` and ``rate`` are
                # 100% figures that need the explicit factor.
                if basis == 'deposit':
                    layer_pc = share * val
                elif basis == 'rol':
                    layer_pc = share * val * limit
                elif basis == 'rate':
                    layer_pc = share * val * pg
                else:                                   # pragma: no cover
                    raise ValueError(f'unknown ceded-premium basis {basis!r}')
                pc += layer_pc
                if cede is not None and cede[i] is not None:
                    comm += cede[i] * layer_pc
            return pc, comm

        pc_occ, c_occ = side('occ')
        pc_agg, c_agg = side('agg')
        return {'gross': pg, 'ceded': pc_occ + pc_agg,
                'pc_occ': pc_occ, 'pc_agg': pc_agg, 'c_occ': c_occ, 'c_agg': c_agg}

    @staticmethod
    def _make_feature_terms(var_feat, var_params, share=1.0):
        """Instantiate the ContractTerms for a parsed variable feature.

        ``var_feat`` is the feature key (``'swing'`` / ``'slide'`` / ``'pc'``
        / ``'corridor'``) popped from the spec, ``var_params`` its parsed
        parameter dict; returns the terms object, or ``None`` when no
        feature is present.

        ``share`` is the placement fraction of the decorated layer. For
        ``swing`` the currency terms -- ``basic`` / ``minimum`` / ``maximum``,
        quoted at 100% placement -- are scaled by it (``lcm``, a dimensionless
        loss multiplier applied to the already-placed ceded loss, is left
        unchanged), so the ceded premium ``clip(share*basic + lcm*A,
        share*min, share*max)`` is the placed figure. The other features are
        percentage-based (loss ratios / commission fractions) and need no
        scaling.
        """
        if var_feat is None:
            return None
        from .contract_terms import (
            CorridorTerms, ProfitCommissionTerms, SlideTerms, SwingTerms)
        if var_feat == 'swing' and share != 1.0:
            var_params = dict(var_params)
            for k in ('basic', 'minimum', 'maximum'):
                if var_params.get(k) is not None:
                    var_params[k] = float(share) * float(var_params[k])
        return {'swing': SwingTerms, 'slide': SlideTerms,
                'pc': ProfitCommissionTerms,
                'corridor': CorridorTerms}[var_feat](**var_params)

    def _factory(self, parsed):
        """
        Internal: construct the object described by a :class:`~aggregate.recipe.Recipe`.

        Portfolio construction needs ``self`` (passed as ``uw``), which is why
        this is not a staticmethod.

        :param parsed: a :class:`~aggregate.recipe.Recipe` with ``kind``, ``name``,
            ``spec``, and ``program`` populated; ``object`` is None on input.
        :return: the same ``parsed`` with ``parsed.object`` set to the
            constructed object (or left ``None`` for the named-mixed-severity
            case, which can only be created in the context of an Aggregate).
        """

        kind, name, spec, program = parsed.kind, parsed.name, parsed.spec, parsed.program

        if kind == 'agg':
            # A pure aggregate ignores what it cannot use and says so: the
            # reinsurance-economics / feature clauses parse everywhere (the
            # shared agg body), but activating them needs a P&L premium
            # context. Filter a COPY -- ``parsed.spec`` *is* the stored
            # recipe's spec, and the retained keys are exactly what an
            # ``agg.NAME`` reference inside a ``pnl`` / ``xpnl`` re-injects
            # ([Reins-Economics-On-Agg-Ignore-Warn]).
            msg = ignored_clauses_message(name, spec)
            if msg:
                warnings.warn(msg, IgnoredDecLClauseWarning, stacklevel=2)
                spec = {k: v for k, v in spec.items()
                        if k not in _AGG_IGNORED_ECONOMICS_KEYS}
            obj = Aggregate(**spec)
            obj.program = program
            # With ``program`` now populated, fold the method-of-moments
            # approximation description into the (pure) user note so the
            # human-readable record names what was approximated and how. The
            # round-trip itself rides on ``program`` (re-parsed on load), not the
            # note, so this never compounds across re-exports.
            if getattr(obj, '_approx_fit', None):
                _desc = obj._approx_description()
                obj.note = f"{obj.note}; {_desc}" if obj.note else _desc
        elif kind in ('pnl', 'xpnl'):
            # A ``pnl`` / ``xpnl`` wraps a complete stochastic engine (an inline
            # ``agg``, an ``agg.NAME`` ref, or a ``port.NAME`` ref) and reads its
            # loss out. For an agg engine the loss structure is merged into
            # ``spec``, so the inner pure-loss Aggregate is built here plus a
            # **recipe**; the eager P&L value object is *snapshotted* from the
            # recipe only after the inner Aggregate is updated (build the engine
            # first, then snapshot -- a PnL retains no engine). ``build_many``
            # calls :meth:`_snapshot_pnl` after the update loop. ``pnl`` returns
            # the consolidated single-group net view; ``xpnl`` the multi-group
            # step walk; a ``port`` engine takes the dedicated path below.
            is_tower = (kind == 'xpnl')
            port_engine = spec.pop('_engine_port', None)
            # The engine's own note (inline agg) is inner-Aggregate presentation
            # metadata, not loss structure -- strip before build. Its ``as``
            # label names the **loss leg** in the P&L ledger, so pop it into
            # the recipe (dev/plan-yapnl.md label plumbing).
            spec.pop('engine_note', None)
            spec.pop('engine_name', None)
            loss_label = spec.pop('engine_label', None)
            consideration = spec.pop('consideration')
            expense_spec = spec.pop('expense_spec', None)
            # Optional consideration display label (the ``as`` clause on the
            # premium head) -> the consideration leg's dict key in the plain P&L.
            consideration_label = spec.pop('consideration_label', None)
            if port_engine is not None:
                obj = self._build_pnl_from_port(
                    name, port_engine, consideration, expense_spec,
                    consideration_label, loss_label, is_tower, program,
                    trailer_meta=_trailer_meta(spec))
                parsed.object = obj
                return parsed
            # ``inherit premium``: resolve the sentinel to the engine's technical
            # premium (the merged ``exp_premium``) before economics see it. A
            # build error if the agg engine carries no premium (decision 3).
            if consideration is INHERIT_PREMIUM:
                consideration = self._inherit_agg_premium(name, spec)
            # Reinstatement schedule (stochastic ceded premium): pop before the
            # inner Aggregate is built (it is not a loss-structure key) and use it
            # below to attach a ReinstatementTerms.
            reinst = spec.pop('occ_reins_reinst', None)
            # Variable rating (Phase 3, decision 0): one of swing / slide / pc /
            # corridor on one aggregate layer. Pop the feature spec key + the
            # decorated layer index before economics / inner build.
            var_feat = var_params = var_layer_idx = None
            for _f in ('swing', 'slide', 'pc', 'corridor'):
                if f'agg_reins_{_f}' in spec:
                    var_feat = _f
                    var_params = spec.pop(f'agg_reins_{_f}')
                    var_layer_idx = spec.pop(f'agg_reins_{_f}_layer')
                    break
            # Retro (Phase 3): the account-level rating clause varying the *gross*
            # premium as a collared affine map of net account loss. Pop the collar.
            retro_collar = spec.pop('retro_terms', None)
            # --------------------------------------------------------------
            # Two-tier classification ([One-Classifier-Fix],
            # dev/done/plan-pnl-faces-punchlist.md):
            # the occurrence tier is {none | gc | reinstatements}, the
            # aggregate tier {none | gc | feature}, classified INDEPENDENTLY
            # -- the old single-kind elif chain let a feature branch shadow
            # the reinstatements clause and mis-scope an inuring occurrence
            # program. Ceded-premium clauses resolve to per-side economics;
            # a side with cession layers but NO premium clause books at zero
            # ceded premium with one warning ([XPnL-Zero-Premium-Cessions])
            # -- reinsurance presence, not economics presence, drives the
            # face. Exempt: a feature-decorated aggregate side (the feature
            # owns its premium slot) and a reinstated occurrence layer
            # (which requires a base premium clause -- error below).
            # --------------------------------------------------------------
            has_occ = bool(spec.get('occ_reins'))
            has_agg = bool(spec.get('agg_reins'))
            occ_noclause = has_occ and 'occ_reins_premium' not in spec
            agg_noclause = has_agg and 'agg_reins_premium' not in spec
            econ = self._resolve_reins_economics(spec, consideration)
            if econ is None and (has_occ or has_agg):
                econ = {'gross': float(consideration), 'ceded': 0.0,
                        'pc_occ': 0.0, 'pc_agg': 0.0,
                        'c_occ': 0.0, 'c_agg': 0.0}
            zero_sides = []
            if occ_noclause and reinst is None:
                zero_sides.append('occurrence')
            if agg_noclause and var_feat is None:
                zero_sides.append('aggregate')
            if zero_sides and retro_collar is None:
                warnings.warn(
                    f"{name}: the {' and '.join(zero_sides)} cession"
                    f"{'s have' if len(zero_sides) > 1 else ' has'} no "
                    "ceded-premium clause (deposit / rol / rate); booking at "
                    "zero ceded premium. Price the cover to silence this.",
                    ZeroPremiumCessionWarning, stacklevel=2)
            if retro_collar is not None:
                # Retro varies the gross premium via the collar map over the
                # gross density. The clean 1-D case is retro with no inuring
                # reinsurance (net account loss = gross loss); retro +
                # reinsurance is a follow-up.
                if var_feat is not None or spec.get('occ_reins') or \
                        spec.get('agg_reins'):
                    raise ValueError(
                        f"{name}: 'retro' with reinsurance (or another variable "
                        "feature) is not yet supported; retro currently requires a "
                        "book with no inuring reinsurance (the 1-D net-account-loss "
                        "= gross-loss case).")
                inner = Aggregate(**spec)
                from .contract_terms import RetroTerms
                inner.variable_terms = RetroTerms(**retro_collar)
                inner.variable_layer = None
                inner.variable_gross_premium = float(consideration)
                inner.variable_ceded_premium = 0.0
                inner.variable_commission = 0.0
                inner._pnl_recipe = {'kind': 'var', 'expense_spec': expense_spec,
                                     'consideration': consideration,
                                     'consideration_label': consideration_label,
                                     'loss_label': loss_label}
            elif reinst is not None:
                # Reinstatements (stochastic ceded premium D + h(R)) route
                # through the joint-sourced reinstatement builders. The base
                # premium D = the layer's resolved occurrence premium
                # (econ['pc_occ']); effective occ limit y = share x limit.
                # This branch precedes the feature branch
                # ([Reinstatements-Dropped-By-Feature-Branch] fix): the occ
                # tier owns the (L, R) joint, and a feature on the aggregate
                # cover rides the SAME joint via the tier maps
                # (``agg_feature_terms`` passed to the builder).
                from .contract_terms import ReinstatementTerms
                if occ_noclause:
                    raise ValueError(
                        f"{name}: 'reinstatements' require a base premium clause "
                        "(deposit / rol / rate) on the occurrence layer.")
                inner = Aggregate(**spec)
                rates = next((r for r in reinst if r is not None), None)
                share, limit, _attach = spec['occ_reins'][0]
                inner.reinstatement_terms = ReinstatementTerms(
                    limit=float(share) * float(limit), rates=rates,
                    deposit=float(econ['pc_occ']))
                inner.reinstatement_gross_premium = float(econ['gross'])
                inner._pnl_recipe = {'kind': 'reins', 'expense_spec': expense_spec,
                                     'econ': econ,
                                     'agg_feature_terms':
                                         self._make_feature_terms(
                                             var_feat, var_params,
                                             share=(spec['agg_reins']
                                                    [var_layer_idx][0]
                                                    if var_feat is not None
                                                    else 1.0)),
                                     'consideration_label': consideration_label,
                                     'loss_label': loss_label}
            elif var_feat is not None:
                # A feature on the aggregate tier over a plain or GC-occ
                # engine. The feature's subject is what the engine emits --
                # the gross aggregate, or the net-of-occurrence aggregate
                # when a guaranteed-cost occurrence program inures
                # ([Var-Feature-Composed-With-Occ-Program]) -- so do NOT
                # apply the agg reinsurance in the inner Aggregate; keep the
                # decorated layer for the analysis ceder. The occ program
                # (if any) stays in the engine; its GC economics ride the
                # recipe ``econ`` into the builders.
                var_layer = spec['agg_reins'][var_layer_idx]
                spec.pop('agg_reins', None)
                spec.pop('agg_kind', None)
                inner = Aggregate(**spec)
                terms = self._make_feature_terms(var_feat, var_params,
                                                 share=var_layer[0])
                inner.variable_terms = terms
                inner.variable_layer = var_layer
                inner.variable_gross_premium = float(consideration)
                inner.variable_ceded_premium = float(econ['pc_agg']) if econ else 0.0
                inner.variable_commission = float(econ['c_agg']) if econ else 0.0
                inner._pnl_recipe = {'kind': 'var', 'expense_spec': expense_spec,
                                     'econ': econ,
                                     'consideration': consideration,
                                     'consideration_label': consideration_label,
                                     'loss_label': loss_label}
            else:
                inner = Aggregate(**spec)
                if getattr(inner, '_approx_fit', None):
                    _desc = inner._approx_description()
                    inner.note = f"{inner.note}; {_desc}" if inner.note else _desc
                inner._pnl_recipe = {
                    'kind': 'gcn' if (has_occ or has_agg) else 'plain',
                    'expense_spec': expense_spec, 'econ': econ,
                    'consideration': consideration,
                    'consideration_label': consideration_label,
                    'loss_label': loss_label}
            # ``xpnl`` returns the multi-group step walk rather than the
            # consolidated single-group PnL; carried on the recipe so
            # :meth:`_snapshot_pnl` selects that face after the inner engine is
            # updated.
            inner._pnl_recipe['is_tower'] = is_tower
            # The pnl statement's own trailer. It is also splatted onto the
            # inner Aggregate (the parser merges the specs), but the PnL face
            # reads it from here so the port-engine path -- where the engine
            # carries the *portfolio's* metadata -- stays correct.
            inner._pnl_recipe['trailer_meta'] = _trailer_meta(spec)
            inner.program = program
            obj = inner
        elif kind == 'bvagg':
            from .bivariate import BivariateAggregate
            obj = BivariateAggregate(**spec)
            obj.program = program
        elif kind == 'port':
            # Portfolio expects name, agg_list, uw. agg_list is a list of specs
            # that can be passed to Aggregate. Drop the leading ('agg', name)
            # from each spec entry returned by the parser.
            pnl_units = [j for i, j, k in spec['spec'] if i == 'pnl']
            if pnl_units:
                raise NotImplementedError(
                    f"{name}: pnl units in a portfolio are not supported "
                    "(book-level P&L is deferred: a loss-sensitive consideration "
                    "must be netted per unit before combining, which loses unit "
                    f"premium attribution). Offending unit(s): "
                    f"{', '.join(pnl_units)}. Build a standalone PnL, or declare "
                    "the unit as a plain agg.")
            agg_list = [k for i, j, k in spec['spec']]
            # The whole trailer, not just the note: a172 [FCC-Contract-Gaps].
            # ``tags`` / ``doc`` / ``hints`` were parsed into the port spec and
            # dropped here, so a portfolio's own trailer was write-only. The
            # hints still *took effect* (``_resolve_hints`` reads the spec), but
            # the object could not report how it had been asked to build.
            obj = Portfolio(name, agg_list, uw=self,
                            label=spec.get('label'),
                            note=spec.get('note', ''),
                            tags=spec.get('tags', ()),
                            doc=spec.get('doc', ''),
                            hints=spec.get('hints', ''))
            obj.program = program
        elif kind == 'sev':
            if 'sev_wt' in spec and spec['sev_wt'] != 1:
                logger.warning(
                    'Mixed severity cannot be created, returning spec. You had %s, expected 1',
                    spec["sev_wt"])
                obj = None
            else:
                obj = Severity(**spec)
                obj.program = program
        elif kind == 'distortion':
            obj = Distortion.from_spec(spec)
            obj.program = program
        elif kind == 'expr':
            # a bare expression is its own answer: the parser has already
            # evaluated it, so the spec is the number. There is no object to
            # construct and nothing is stored in the recipe base.
            obj = spec
        else:
            raise ValueError(f'Cannot build {kind} objects')

        parsed.object = obj
        return parsed

    def add_recipe(self, kind, name, spec, program, source='session'):
        """
        Add (or overwrite) one parsed declaration in the recipe base.

        The single public mutator of the store. ``(kind, name)`` is the unique
        key; an existing entry with the same key is overwritten (last write
        wins). Used by :meth:`_interpret_program` and by test fixtures that need
        to seed the recipe base directly.

        Parameters
        ----------
        kind : str
            One of ``'sev'``, ``'agg'``, ``'port'``, ``'bvagg'``, ``'pnl'``,
            ``'xpnl'``, ``'distortion'``. Note that ``'expr'`` is deliberately
            not stored: a bare expression is an answer, not a declaration, so
            :meth:`_interpret_program` skips this method for it.
        name : str
            The declaration name.
        spec : dict
            The parsed constructor kwargs.
        program : str
            The DecL statement as the parser received it, i.e. one line of
            :meth:`UnderwritingLexer.preprocess` output, not the source text.
        source : pathlib.Path or str, default 'session'
            Provenance: the file the entry came from, or ``'session'`` for an
            in-session build.
        """
        self._recipes[(kind, name)] = Recipe(
            kind=kind, name=name, spec=spec, program=program, source=source)

    #: :attr:`recipes` columns, in display order: identity and audit flags
    #: first, the wide ``program`` / ``spec`` payload last so the frame stays
    #: readable at a terminal.
    _RECIPE_COLUMNS = ('note', 'tags', 'doc', 'problem', 'solution',
                       'discussion', 'check', 'n_asserts', 'source',
                       'program', 'spec')

    def _recipes_frame(self):
        """Build the ``(kind, name)``-indexed DataFrame view of the dict store.

        Columns are :data:`_RECIPE_COLUMNS`, sorted by index. Built on demand so
        the store itself stays a plain dict. Only entries that actually carry a
        ``doc{{{...}}}`` are parsed -- the section flags of an undocumented
        entry are known to be ``False`` without looking.
        """
        if not self._recipes:
            empty = pd.MultiIndex.from_arrays([[], []], names=['kind', 'name'])
            return pd.DataFrame(columns=list(self._RECIPE_COLUMNS), index=empty)
        rows = []
        for (kind, name), r in self._recipes.items():
            documented = bool(r.doc)
            present = set(r.sections) if documented else set()
            rows.append({
                'kind': kind, 'name': name,
                'note': bool(r.note),
                'tags': r.tags,
                'doc': documented,
                **{s: (s in present) for s in ('problem', 'solution',
                                               'discussion', 'check')},
                'n_asserts': r.n_asserts if documented else 0,
                'source': self._format_source(r.source),
                'program': r.program,
                'spec': r.spec,
            })
        df = pd.DataFrame(rows).set_index(['kind', 'name'])
        return df[list(self._RECIPE_COLUMNS)].sort_index()

    @property
    def recipes(self):
        """The recipe base as a ``(kind, name)``-indexed DataFrame (lazy-loaded).

        One row per entry. Reads the configured databases on first access (the
        lazy path), then builds the frame on demand from the dict store. This is
        both the directory ("what do I know about?") and the *audit* half of the
        notes-driven describe / test / audit surface -- it answers "which
        entries carry a recipe, and which of those actually assert anything?"

        Columns
        -------
        note : bool
            Has a one-line ``note{...}`` abstract.
        tags : tuple of str
            The entry's ``tags{...}`` slugs.
        doc : bool
            Has a ``doc{{{...}}}`` body. Most entries do not, by design: a doc
            is for the cookbook-worthy few, a note is the norm.
        problem, solution, discussion, check : bool
            Which canonical sections the doc provides.
        n_asserts : int
            ``assert`` statements in the Check block. **Zero with
            ``check=True`` means the invariant is stated but not tested** --
            the case worth hunting.
        source : str
            Which ``.agg`` file the entry came from (``'session'`` for an
            in-session build).
        program : str
            The DecL statement as the parser received it: one line, comments
            stripped, any ``doc{{{...}}}`` body base64-encoded.
        spec : dict
            The parsed constructor kwargs.

        Examples
        --------
        The flags without the wide payload::

            build.recipes.iloc[:, :9]

        Entries carrying a cookbook recipe::

            build.recipes.query('doc')

        Documented but unchecked::

            build.recipes.query('doc and n_asserts == 0')

        See Also
        --------
        recipe : one entry, by name.
        discover : filter by name / kind / tags, optionally building each match.
        """
        if not self._loaded:
            self.load()
        return self._recipes_frame()

    def recipe(self, name, kind=''):
        """One :class:`~aggregate.recipe.Recipe` from the recipe base, by name.

        Resolves by **name alone**: the recipe base is keyed ``(kind, name)``,
        but a recipe is addressed the way a reader would say it, and the shipped
        library keeps names unique across kinds so ``recipe('X')`` and
        ``build('X')`` always mean the same entry. If a name does happen to
        exist under more than one kind (possible in a hand-assembled recipe
        base), that is an error rather than a silent pick -- pass ``kind=`` to
        disambiguate.

        This is the canonical lookup, and the one place the lookup is
        implemented: ``uw[name]`` and ``uw[kind, name]`` are subscript spellings
        that delegate here, so all three raise the same errors.

        Pure lookup. **The returned Recipe's ``object`` is always ``None``**,
        because this does not construct anything. Use :meth:`__call__` /
        :meth:`build` / :meth:`build_many`, which run :meth:`_factory` after the
        lookup, when you want a live Aggregate / Severity / Portfolio /
        Distortion.

        Parameters
        ----------
        name : str
            The entry name, e.g. ``'LimitProfile'``.
        kind : str, optional
            Restrict to one kind (``'agg'``, ``'port'``, ``'sev'``,
            ``'distortion'``, ...). Only needed for an ambiguous name.

        Returns
        -------
        aggregate.recipe.Recipe
            The entry: spec, program and provenance, plus its note, tags and
            (lazily parsed) doc sections. An entry with no ``doc{{{...}}}``
            yields a Recipe with empty sections rather than an error -- the
            audit reports it as undocumented. ``object`` is ``None``; use
            :meth:`build` to construct.

        Raises
        ------
        KeyError
            No such entry, or the name is ambiguous across kinds.

        See Also
        --------
        recipes : the frame over every entry.
        __getitem__ : the same lookup, spelled as a subscript.
        """
        if not self._loaded:
            self.load()
        # A copy is returned throughout: the caller must not be able to mutate
        # the stored entry (and a fresh copy re-derives its doc / decl caches).
        if kind:
            # The store is keyed (kind, name), so a supplied kind is a direct
            # hit. This is the parser's path via _safe_lookup, so keep it O(1).
            try:
                return replace(self._recipes[(kind, name)])
            except KeyError:
                raise KeyError(f'no recipe named {name!r} of kind {kind!r}') from None
        hits = [k for (k, n) in self._recipes if n == name]
        if not hits:
            raise KeyError(f'no recipe named {name!r}')
        if len(hits) > 1:
            kinds = ', '.join(sorted(hits))
            raise KeyError(
                f'{name!r} is ambiguous across kinds ({kinds}); pass kind= to '
                f'choose. Shipped library names are unique across kinds.')
        return replace(self._recipes[(hits[0], name)])

    @property
    def version(self):
        import aggregate
        return aggregate.__version__

    @property
    def test_suite_file(self):
        """Path to the bundled test suite ``.agg`` file, or ``None`` if not present."""
        f = self.default_dir / TEST_SUITE_FILENAME
        return f if f.exists() else None

    @staticmethod
    def show_settings():
        """Print every resolved setting and where its value came from.

        Each row is ``section.key = value  [source]`` where source is
        ``default`` (built-in), ``config`` (the TOML file), or ``env`` (an
        ``AGGREGATE_*`` variable). This is the discoverable, no-magic view of
        the configuration described in :mod:`aggregate.config`.
        """
        rows = describe_settings(get_settings())
        width = max(len(k) for k, _, _ in rows)
        lines = [f'{k:<{width}} = {v!r}  [{src}]' for k, v, src in rows]
        print('\n'.join(lines))

    @staticmethod
    def write_default_config(path=None, *, force=False):
        """Write the annotated, fully-commented config template to ``~/.aggregate``.

        Thin delegate to :func:`aggregate.config.write_default_config`. The
        written file is inert (all lines commented) until you uncomment a key.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination; defaults to ``~/.aggregate/config.toml``.
        force : bool, default False
            Overwrite an existing file.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        return _write_default_config(path, force=force)

    @staticmethod
    def reload_settings():
        """Re-read the config file / environment and refresh the module ``build``.

        Thin delegate to :func:`aggregate.config.reload_settings`. Use after
        editing ``~/.aggregate/config.toml`` in a live session.

        Returns
        -------
        Settings
            The freshly resolved settings.
        """
        return _reload_settings()

    def _build_work(self, portfolio_program, log2=0, bs=0, update=None, **kwargs):
        """
        Internal: parse → factory (without smart-update). Used by :meth:`build_many`.

        Tries a name lookup in the recipe base first; falls back to parsing the
        program. If ``update`` is True, calls each constructed object's
        ``.update(log2, bs, **kwargs)`` with the *literal* log2/bs (no bucket
        inference — that lives in :meth:`build_many`).

        :param portfolio_program: a DecL program (str), or the name of a
            previously-built object in the recipe base.
        :param log2: passed verbatim to each object's ``update``.
        :param bs: passed verbatim to each object's ``update``.
        :param update: override the class-level ``self.update`` default.
        :param kwargs: passed to each created object's ``update`` method.
        :return: list of :class:`~aggregate.recipe.Recipe` (one per top-level
            declaration).
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
                pnl_recipe = getattr(answer.object, '_pnl_recipe', None)
                if pnl_recipe is not None:
                    prog = answer.object.program
                    answer.object = self._snapshot_pnl(answer.object, pnl_recipe)
                    answer.object.program = prog
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
            logger.warning('Program did not contain any output')
        else:
            logger.info('Program created %d objects.', len(rv))
        return rv

    def _interpret_program(self, portfolio_program, source='session'):
        """
        Internal: preprocess and parse a program one line at a time, storing
        each parsed spec in the recipe base. No objects are constructed.

        :param portfolio_program: the DecL program text.
        :param source: provenance tag for the stored entries — the originating
            file :class:`~pathlib.Path` when reading a database, else
            ``'session'`` for in-session builds.
        :return: list of :class:`~aggregate.recipe.Recipe` (``object`` is
            ``None`` for each). The returned recipes are copies; mutating their
            ``object`` field does not touch the stored ones.
        """
        portfolio_program = self.lexer.preprocess(portfolio_program)
        rv = []
        for program_line in portfolio_program:
            logger.debug(program_line)
            try:
                kind, name, spec = self.parser.parse(self.lexer.tokenize(program_line))
            except ValueError as e:
                report = getattr(e, "report", None)
                if report is not None:
                    # render() starts with "DecL parse error at ..." -- no prefix needed.
                    logger.error("%s", report.render())
                else:
                    # Some other ValueError (e.g. from the transformer) -- log as-is.
                    logger.error(e)
                raise
            else:
                if kind == 'expr':
                    # a bare expression evaluates to a number; it is an answer,
                    # not a declaration, so it is never stored (nothing to look
                    # up by name, and no spec_to_decl renderer for it).
                    logger.info('answer out: expression %s evaluated', name)
                    rv.append(Recipe(kind=kind, name=name, spec=spec,
                                     program=program_line, source=source))
                    continue
                logger.info('answer out: %s object %s parsed successfully...adding a recipe',
                            kind, name)
                self.add_recipe(kind, name, spec, program_line, source=source)
                # Hand back a fresh copy: _build_work / build_many set .object
                # on these, which must not leak into the stored recipe.
                rv.append(replace(self._recipes[(kind, name)]))
        return rv

    def _safe_lookup(self, buildinid):
        """
        Internal: parser callback that looks up ``kind.name`` in the recipe base
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
            logger.error('ERROR id %s.%s not found in the recipe base.', kind, name)
            raise
        logger.debug('UnderwritingParser.safe_lookup | retrieved %s.%s as type %s.%s',
                     kind, name, parsed.kind, parsed.name)
        if parsed.kind != kind:
            raise ValueError(f'Error: type of {name} is  {parsed.kind}, not expected {kind}')
        # don't want to pass back the original; changes would be reflected in the recipe base
        return deepcopy(parsed.spec)

    def build_many(self, program, update=None, log2=0, bs=0, bucket_sizing_p=BUCKET_SIZING_P, **kwargs):
        """
        Parse a (possibly multi-output) DecL program, construct each object, and smart-update.

        Always returns the full ``list[Recipe]`` regardless of count.
        Use :meth:`build` instead when you expect a single output.

        Smart-update logic: discrete severities pick ``bs=1`` with a log2 sized
        to the max possible loss; continuous ones size from the analytic moment
        window; portfolios use :meth:`Portfolio.best_window`. A ``hints{}`` clause
        in the program can override these (explicit ``build()`` kwargs always win).

        :param program: a DecL program producing one or more top-level outputs.
        :param update: override the class-level ``self.update`` default.
        :param log2: 0 (default) estimates log2 for discrete severities and
            uses ``self.log2`` for everything else.
        :param bs: bucket size; 0 lets the object recommend one.
        :param bucket_sizing_p: passed to the bucket / window sizer; raise (closer
            to 1) for thick-tailed distributions.
        :param kwargs: passed to each ``update`` call. ``force_severity=True``
            is always applied.
        :return: list of :class:`~aggregate.recipe.Recipe`, one per top-level output.
        """
        rv = self._build_work(program, update=False, force_severity=True)

        if not rv:
            logger.warning('build produced no output')
            return rv

        if update is None:
            update = self.update
        # tested by truthiness below, so update=1 behaves like update=True
        # rather than silently skipping the update
        update = bool(update)

        # in this loop bs_ and log2_ are the values actually used for each
        # update; they do not overwrite the input default values
        from .bivariate import BivariateAggregate

        # The closed set of objects _factory can produce. Severity, Distortion
        # and a bare numeric expr have no update step; everything in
        # ``updatable`` is finished by the dispatch at the foot of the loop.
        # Register a new output type here, in one place, and give it a branch.
        no_update = (Severity, Distortion, numbers.Number)
        updatable = (Aggregate, Portfolio, BivariateAggregate)

        for answer in rv:
            obj = answer.object
            if obj is None:
                # object not created (named-mixed-severity case)
                logger.info('Object %s of kind %s returned as a spec; no further processing.',
                            answer.name, answer.kind)
                continue
            if isinstance(obj, no_update):
                continue
            if not isinstance(obj, updatable):
                logger.warning('Unexpected: build produced a %s, which build_many '
                               'does not know how to finish.', type(obj))
                continue
            if not update:
                # constructed on purpose without a density; nothing to finish
                continue
            # BivariateAggregate is tested first. It is not an Aggregate today,
            # but the order is load bearing if that ever changes.
            if isinstance(obj, BivariateAggregate):
                # per-axis auto-sizing lives in BivariateAggregate.update;
                # pass log2/bs through (0 => auto), drop agg-only kwargs.
                d = answer.spec
                log2, bs, bucket_sizing_p, kwargs = _resolve_hints(
                    d, log2, bs, bucket_sizing_p, kwargs)
                log2_ = 0 if log2 == 0 else log2
                logger.info('(%s, %s): bivariate update(log2=%s, bs=%s)',
                            answer.kind, answer.name, log2_, bs)
                obj.update(log2=log2_, bs=bs, **kwargs)
            elif isinstance(obj, Aggregate):
                # pnl / xpnl ride here: their deferred engine is an Aggregate
                # until _snapshot_pnl runs below.
                d = answer.spec
                log2, bs, bucket_sizing_p, kwargs = _resolve_hints(
                    d, log2, bs, bucket_sizing_p, kwargs)
                # ``log2`` is a CAP; bucket + window selection is delegated to
                # Aggregate.update / _bs_window (the single source of truth:
                # exact-discrete, bounded, moment, and signed/P&L windows).
                # Pass bs=0 to estimate, or an explicit bs (call or note) to
                # pin the grid. There is deliberately NO separate bucket logic
                # here -- no back doors around the estimator.
                log2_ = self.log2 if log2 == 0 else log2
                logger.info('(%s, %s): update(log2=%s, bs=%s) -> _bs_window',
                            answer.kind, answer.name, log2_, bs)
                try:
                    obj.update(
                        log2=log2_, bs=bs, bucket_sizing_p=bucket_sizing_p,
                        debug=self.debug, force_severity=True, **kwargs)
                except (ZeroDivisionError, AttributeError) as e:
                    logger.error(e)
            else:
                # Portfolio
                d = answer.spec
                log2, bs, bucket_sizing_p, kwargs = _resolve_hints(
                    d, log2, bs, bucket_sizing_p, kwargs)
                if log2 == -1:
                    log2_ = 13
                elif log2 == 0:
                    log2_ = self.log2
                else:
                    log2_ = log2
                # Pass bs through (0 => auto). No back doors: Portfolio.update
                # routes bs==0 through best_window (non-signed) or _bs_window
                # (signed coarsen-to-fit) itself, so do NOT pre-compute the
                # bucket here -- mirrors the Aggregate branch above (plan 3.4).
                logger.info('(%s, %s): bs=%s and log2=%s', answer.kind, answer.name, bs, log2_)
                obj.update(log2=log2_, bs=bs, bucket_sizing_p=bucket_sizing_p,
                           remove_fuzz=True, force_severity=True,
                           debug=self.debug, **kwargs)

        # snapshot any deferred P&L: the inner Aggregate is now updated, so build
        # the eager PnL / marginal-stack / analysis value object from its recipe.
        if update:
            for answer in rv:
                recipe = getattr(answer.object, '_pnl_recipe', None)
                if recipe is not None:
                    prog = answer.object.program
                    answer.object = self._snapshot_pnl(answer.object, recipe)
                    answer.object.program = prog
        return rv

    @staticmethod
    def _inherit_agg_premium(name, spec):
        """Resolve ``inherit premium`` for an agg engine: its technical premium.

        Reads the merged engine's ``exp_premium`` (from a ``premium at lr``
        exposure or a stored agg with premium). A build error if the engine
        carries no premium (a ``claims`` / ``loss`` exposure) -- ``inherit`` then
        has nothing to copy (decision 3).
        """
        prem = spec.get('exp_premium', None)
        total = (float(np.sum(np.asarray(prem, dtype=float)))
                 if prem is not None else 0.0)
        if not total:
            raise ValueError(
                f"{name}: 'inherit premium' but the wrapped engine has no "
                "premium to inherit -- it needs a 'premium at lr' exposure (or a "
                "stored agg / port carrying premium). Give an explicit "
                "'<amount> premium' instead.")
        return total

    def _build_pnl_from_port(self, name, portname, consideration, expense_spec,
                             consideration_label, loss_label, is_tower, program,
                             trailer_meta=None):
        """Build a portfolio-sourced P&L: wrap a ``port.NAME`` engine's total.

        A ``port`` source reads the **net-net portfolio total** and sees nothing
        inside, so a port-sourced P&L is inherently the *plain* case (decision 8).
        The Portfolio is returned as a deferred object carrying a ``port_plain``
        recipe; the :meth:`build_many` update loop updates it, then
        :meth:`_snapshot_pnl` builds the eager :class:`PnL` from its total-loss
        density. ``xpnl`` over a port is rejected (nothing to explode, decision 6).
        """
        if is_tower:
            raise NotImplementedError(
                f"{name}: 'xpnl' over a portfolio is not supported -- the "
                "portfolio total hides its units, so there is nothing to explode. "
                "Use 'pnl' for the net-net book P&L (or 'xpnl' over a single agg "
                "engine).")
        port_pp = self[('port', portname)]
        port_spec = deepcopy(port_pp.spec)
        port_units = [k for _i, _j, k in port_spec['spec']]
        engine = Portfolio(name, port_units, uw=self,
                           label=port_spec.get('label'))
        engine.program = program
        engine._pnl_recipe = {'kind': 'port_plain', 'expense_spec': expense_spec,
                              'consideration': consideration,
                              'consideration_label': consideration_label,
                              'loss_label': loss_label,
                              # The P&L's own trailer, NOT the portfolio's.
                              'trailer_meta': trailer_meta or {}}
        return engine

    @staticmethod
    def _snapshot_pnl(inner, recipe):
        """Snapshot the eager P&L value object from an **updated** inner engine.

        Called by :meth:`build_many` after the inner engine is updated (build
        the engine first, then snapshot). A thin dispatcher over the builders
        in :mod:`aggregate._pnl_builders` -- two faces, two questions
        (``dev/plan-pnl-consolidated-xpnl-walk.md``):

        * ``pnl`` -- the **consolidated** net view, always one group:
          :func:`~aggregate._pnl_builders.build_plain_pnl` (no cessions),
          :func:`~aggregate._pnl_builders.build_consolidated_pnl`
          (guaranteed-cost reinsurance economics), or the consolidated face
          of :func:`~aggregate._pnl_builders.build_variable_pnl`
          (variable-rating features) /
          :func:`~aggregate._pnl_builders.build_reinstatement_pnl`
          (reinstatements: one sell group of 2-D legs over the (L, R)
          joint -- [2D-Deferred] closed).
        * ``xpnl`` -- the **walk**, a multi-group :class:`PnL`:
          :func:`~aggregate._pnl_builders.build_xpnl_walk` (guaranteed-cost,
          marginal-stitched), the walk face of ``build_variable_pnl`` (the
          per-atom two-group ledger, or the stitched three-step walk when a
          GC occurrence program inures), or the reinstatement 2-D tower
          re-homed. A plain engine gets the same one-group ledger presented
          as a **one-step walk** ([Decision-XPnL-Plain-Is-One-Step-Walk]);
          retro (no cover at all) errors.

        The wrapped engine rides on ``pnl.engine`` for drill-down.
        """
        from ._pnl_builders import (build_plain_pnl, build_consolidated_pnl,
                                    build_xpnl_walk, build_variable_pnl,
                                    build_reinstatement_pnl,
                                    build_reinstatement_source)
        kind = recipe['kind']
        if kind == 'port_plain':
            # ``inner`` is an updated Portfolio; wrap its net-net total loss as
            # a plain P&L against the stated (or inherited) consideration. The
            # loss leg is named by the engine label, else the port itself.
            consideration = recipe['consideration']
            if consideration is INHERIT_PREMIUM:
                prem = float(getattr(inner, 'exp_premium', 0.0) or 0.0)
                if not prem:
                    raise ValueError(
                        f"{inner.name}: 'inherit premium' but the wrapped "
                        "portfolio has no accumulated premium (its units carry "
                        "no 'premium at lr' exposure). Give an explicit "
                        "'<amount> premium'.")
                consideration = prem
            dd = inner.density_df
            source = (dd.index.values, dd['p_total'].values)
            loss_label = recipe.get('loss_label') or inner.name
            face = build_plain_pnl(
                source, consideration=float(consideration),
                consideration_label=recipe.get('consideration_label'),
                loss_label=loss_label,
                expense_spec=recipe.get('expense_spec'), name=inner.name)
            face._adopt_engine(inner, recipe)
            return face
        is_tower = recipe.get('is_tower', False)
        if kind == 'var':
            if is_tower and inner.variable_layer is None:
                # retro varies the gross premium of an unreinsured book:
                # there is no cover to step through.
                raise NotImplementedError(
                    f"{inner.name}: 'xpnl' over a retro program is not "
                    'supported -- retro has no cession to walk through. '
                    "Use 'pnl'.")
            # the LEDGER books the expense legs via the split resolver
            # (expense_spec); the builder is the real engine.
            face = build_variable_pnl(
                inner, walk=is_tower, econ=recipe.get('econ'),
                expense_spec=recipe['expense_spec'],
                consideration_label=recipe.get('consideration_label'),
                loss_label=recipe.get('loss_label'), name=inner.name,
                label=inner.label)
            face._adopt_engine(inner, recipe)
            return face
        if kind == 'reins':
            # both faces are per-atom ledgers over the one (L, R) joint:
            # ``xpnl`` the step tower, ``pnl`` the consolidated net view
            # ([Decision-PnL-Is-Consolidated]; [2D-Deferred] closed). A
            # feature on the aggregate cover rides the same joint
            # ([Var-Feature-Composed-With-Occ-Program]); its stochastic
            # commission (slide / pc) lives in the legs, so the scalar
            # ``agg_commission`` is forced to 0 then.
            econ = recipe['econ']
            ft = recipe.get('agg_feature_terms')
            source, terms, gross_premium, agg_recovery = \
                build_reinstatement_source(inner)
            face = build_reinstatement_pnl(
                inner, source=source, terms=terms, gross_premium=gross_premium,
                agg_recovery=agg_recovery,
                agg_ceded_premium=float(econ.get('pc_agg', 0.0)),
                agg_feature_terms=ft,
                occ_commission=float(econ.get('c_occ', 0.0)),
                agg_commission=(0.0 if getattr(ft, 'target_leg', None)
                                == 'expense'
                                else float(econ.get('c_agg', 0.0))),
                walk=is_tower,
                expense_spec=recipe['expense_spec'], gcn_economics=econ,
                consideration_label=recipe.get('consideration_label'),
                loss_label=recipe.get('loss_label'), name=inner.name,
                label=inner.label)
            face._adopt_engine(inner, recipe)
            return face
        if kind == 'gcn':
            econ = recipe['econ']
            if is_tower:
                # ``xpnl`` -> the marginal-stitched step walk (a plain
                # multi-group PnL).
                face = build_xpnl_walk(
                    inner, gross=econ['gross'], ceded=econ['ceded'],
                    gcn_economics=econ,
                    expense_spec=recipe['expense_spec'],
                    consideration_label=recipe.get('consideration_label'),
                    loss_label=recipe.get('loss_label'), name=inner.name,
                    label=inner.label)
            else:
                # ``pnl`` -> the consolidated single-group net view.
                face = build_consolidated_pnl(
                    inner, gross=econ['gross'], ceded=econ['ceded'],
                    gcn_economics=econ, expense_spec=recipe['expense_spec'],
                    consideration_label=recipe.get('consideration_label'),
                    loss_label=recipe.get('loss_label'), name=inner.name,
                    label=inner.label)
            face._adopt_engine(inner, recipe)
            return face
        # kind == 'plain': the consolidated one-group ledger; ``xpnl`` gets
        # the same ledger presented as a one-step walk
        # ([Decision-XPnL-Plain-Is-One-Step-Walk]).
        face = build_plain_pnl(
            inner, consideration=recipe['consideration'],
            consideration_label=recipe.get('consideration_label'),
            loss_label=recipe.get('loss_label'),
            expense_spec=recipe['expense_spec'], name=inner.name,
            label=inner.label, walk=is_tower)
        face._adopt_engine(inner, recipe)
        return face

    def build(self, program, update=None, log2=0, bs=0, bucket_sizing_p=BUCKET_SIZING_P, **kwargs):
        """
        Parse a single DecL program and return the constructed object.

        Primary user-facing entry point. Calls :meth:`build_many` and unwraps
        the single result. ``__call__`` delegates to ``build``.

        :param program: a DecL program producing exactly one top-level output.
        :param update: override the class-level ``self.update`` default.
        :param log2: 0 (default) estimates log2 for discrete severities and
            uses ``self.log2`` for everything else.
        :param bs: bucket size; 0 lets the object recommend one.
        :param bucket_sizing_p: passed to the bucket / window sizer; raise (closer
            to 1) for thick-tailed distributions.
        :param kwargs: passed to ``update`` (e.g. ``padding``). ``force_severity=True``
            is always applied.
        :return: the constructed Aggregate / Severity / Portfolio / Distortion.
        :raises ValueError: if the program produces zero or more than one
            top-level output. Use :meth:`build_many` for batched programs.
        :raises CannotBuild: if the spec parses but no top-level object can be
            built standalone (e.g. a named mixture severity, which can only
            live inside an :class:`Aggregate`). Use :meth:`build_many` to
            receive the :class:`~aggregate.recipe.Recipe` instead.
        """
        rv = self.build_many(program, update=update, log2=log2, bs=bs,
                             bucket_sizing_p=bucket_sizing_p, **kwargs)
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
                f'build_many() to receive the Recipe).'
            )
        return answer.object

    def __call__(self, *args, **kwargs):
        """
        Build an object from a DecL program or a known name.

        Convenience alias for :meth:`build`: ``build(program)`` is the
        canonical way to turn a DecL string into a single live
        Aggregate / Severity / Portfolio / Distortion. ``program`` can
        be either:

        * a DecL source string (parsed, added to the recipe base,
          constructed); or
        * the bare name of an entry already in the recipe base
          (looked up, constructed; the original DecL source is *not*
          re-parsed).

        The lookup branch and the parse branch both end by calling
        :meth:`_factory` on each :class:`~aggregate.recipe.Recipe`, so
        the returned object is always live (never ``None``).

        Three access patterns, contrasted
        ---------------------------------

        Given::

            from aggregate import build
            build('dist cc1 ccoc .25')        # registers cc1 as a recipe

        1. **Build by program text** — parse, register, construct,
           return the object::

                cc1 = build('dist cc1m ccoc .25')
                assert cc1 is not None                # the Distortion

        2. **Recipe lookup, no construction** — returns the recipe
           only; ``object`` is ``None`` because :meth:`recipe`
           does not run the factory::

                entry = build.recipe('cc1m')          # a Recipe
                assert entry.object is None           # *by design*
                assert entry.spec == {'name': 'ccoc', 'r': 0.25}

        3. **Build by name** — lookup in the recipe base **and**
           construct, just like (1) but with no parsing::

                cc1 = build('cc1')                    # the Distortion
                assert cc1 is not None

        :meth:`build_many` is the batched form of (1) / (3); it returns
        a list of :class:`~aggregate.recipe.Recipe` and *does* populate
        ``object`` on each (factory runs as part of the build).

        Rationale
        ---------

        The recipe base stores DecL specs (small, picklable), not
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
                    report = getattr(e, 'report', None)
                    if report is not None:
                        # report.column is 1-indexed; insert >>> at the offending position.
                        i = max(0, report.column - 1)
                        spec = line[0:i] + '>>>' + line[i:]
                        name = 'parse error'
                    else:
                        # Non-parse error (e.g. transformer-level ValueError).
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

    def discover(self, regex='', kind='', tags='', plot=False, describe=False,
                 return_objects=False, **kwargs):
        """
        Match recipes against ``regex`` / ``tags`` (and optional ``kind``);
        optionally build, plot, and describe each.

        Default behavior (``plot=False, describe=False``) is a lightweight
        directory view — just filter the recipe base and return the matching
        DataFrame. Pass ``plot=True`` or ``describe=True`` to build each match
        and visualize/describe it.

        ``kind`` is the **type** filter and ``tags`` the **subject** filter.
        They are independent axes on purpose: library tags never restate an
        entry's own kind (an ``aggregate`` tag on an ``agg`` entry would say
        nothing ``kind='agg'`` does not), so an ``agg`` demonstrating a
        severity form is ``kind='agg'`` *and* ``tags='topic:severity'``.

        Examples::

            build.discover()                          # all entries
            build.discover('Dice')                     # name contains "Dice"
            build.discover(kind='sev')                 # by TYPE
            build.discover(tags='role:hero')           # by SUBJECT/standing
            build.discover(tags='topic:severity role:reference')   # BOTH tags
            build.discover(kind='agg', tags='topic:severity')      # both axes
            build.discover('Dice', plot=True)          # build + plot
            build.discover(tags='topic:distortion', describe=True)

        :param regex: filter on the recipe index (name); '' matches all.
        :param kind: optional filter ('agg', 'sev', 'port', 'distortion'); ''
            matches all. **This is how you filter by type** -- there is no
            type tag.
        :param tags: one or more tag slugs (comma- and/or space-separated). An
            entry matches when it carries **every** tag given, so tags narrow
            rather than widen. '' matches all. Shipped tags are namespaced
            ``topic:`` / ``role:`` / ``check:`` (plus the bare ``slow``).
        :param plot: build each match and call its ``.plot()``.
        :param describe: build each match and ``qd()`` its summary table.
        :param return_objects: when building, also return the list of built
            objects alongside the DataFrame.
        :param kwargs: passed to :meth:`build` for each match.
        :return: DataFrame of matches; ``(objects, DataFrame)`` if
            ``return_objects=True``.
        """
        # base frame: optionally restricted to a kind
        base = self.recipes.droplevel('kind') if not kind else self.recipes.loc[kind]
        # empty regex means "no filter" — return everything
        df = base.filter(regex=regex, axis=0).copy() if regex else base.copy()

        if tags:
            wanted = {t for t in re.split(r'[,\s]+', tags) if t}
            df = df[[wanted <= set(t or ()) for t in df['tags']]]

        # asking for the built objects implies we must run the build loop
        do_build = plot or describe or return_objects

        if not do_build:
            # lightweight directory view; format the program column for readability
            bit = df[['program']].copy()
            bit['program'] = (bit['program']
                              # doc first: its fence would survive the note
                              # pattern and leave a dangling ``}}}``. A stored
                              # program carries the base64 one-liner the
                              # preprocessor produced, not the raw multi-line
                              # body -- match either.
                              .str.replace(r'\s*doc\{\{\{.*?\}\}\}', '',
                                           regex=True, flags=re.S)
                              .str.replace(r' note\{[^}]+\}', '', regex=True)
                              .str.replace(r' tags\{[^}]+\}', '', regex=True)
                              .str.replace(r' hints\{[^}]+\}', '', regex=True)
                              .str.replace(r' {2,}', ' ', regex=True))
            return bit.sort_index()

        # build + plot/describe path: augment df with summary statistics. Trim
        # to the identity columns first -- the eleven statistics are the point
        # here, and carrying the audit flags and the `spec` payload alongside
        # them makes an unreadably wide frame.
        df = df[['source', 'program']].copy()
        # Initialize as object dtype to avoid pandas LossySetitemError on the
        # mixed-type .loc assignment below (modern pandas refuses to coerce a
        # non-bool result of validation_explanation into a bool column).
        summary_cols = ['log2', 'bs', 'actual_m', 'actual_cv', 'actual_sd', 'actual_skew',
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
                # only Aggregate / Portfolio have a `.summary_df` table
                if hasattr(a, 'summary_df'):
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

    def to_agg(self, path, pattern='.*', kind='all', source='session', mode='x'):
        """
        Write a selection of recipes to a ``.agg`` file (pandas-style export).

        Because the recipe base has no "active" database, "save" is an
        explicit export of selected entries to a named file. Each entry already
        stores its ``program`` (DecL source) line, so writing is just emitting
        those lines; the result **re-loads cleanly** via :meth:`load` —
        round-trip correctness is the contract.

        The default call ``uw.to_agg('mybook')`` writes every entry built this
        session (``source='session'``) to ``~/.aggregate/mybook.agg``, ready to
        re-load by name later.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination. An **absolute** path is used as given; otherwise the
            file is written to :attr:`user_dir` (``~/.aggregate``) so it is
            immediately discoverable by :meth:`available_databases` and
            re-loadable by name. A ``.agg`` suffix is added if absent.
        pattern : str, default '.*'
            Regex matched against each entry *name* (the axis :meth:`discover`
            filters on). Default matches all names.
        kind : str, default 'all'
            Filter by kind: ``'all'`` (default), ``'agg'``, ``'sev'``,
            ``'port'``, ``'distortion'``, or ``'bvagg'``. Mirrors
            ``discover(kind=...)``.
        source : str, pathlib.Path, or None, default 'session'
            Provenance filter. Default ``'session'`` exports only the entries
            built this session (not yet in any file). ``'all'`` or ``None``
            ignores provenance and exports every match; a file stem or
            :class:`~pathlib.Path` exports only entries that came from that file
            (re-export / round-trip).
        mode : {'x', 'w', 'a'}, default 'x'
            File-open mode, mirroring Python's open modes:

            * ``'x'`` (default, safe) — create a new file; **raise**
              :class:`FileExistsError` if it already exists.
            * ``'w'`` — overwrite an existing file (logged).
            * ``'a'`` — append the selection as a new block at the end of an
              existing file (a fresh dated comment precedes the block); if the
              file does not exist it is created like ``'w'``.

        Returns
        -------
        pathlib.Path
            The file written.

        Notes
        -----
        Entries are written in dependency order — severities and distortions,
        then aggregates, then portfolios (:data:`_KIND_WRITE_ORDER`) — because
        ``.agg`` files load sequentially and a named reference (``sev.X`` /
        ``agg.X`` / ``dist.X``) must resolve as its line is parsed. (One residual
        case: a *combo* distortion that references other distortions by name is
        only guaranteed to follow them if it sorts after them by name; deep
        distortion chains may still need a manual reorder.)

        ``mode='a'`` orders only the newly appended block; it does not merge or
        re-sort against what is already in the file, so a freshly appended
        definition that an earlier (already-written) entry references would not
        be reordered ahead of it.
        """
        if mode not in ('x', 'w', 'a'):
            raise ValueError(f"mode must be one of 'x', 'w', 'a'; got {mode!r}.")
        # make sure the configured databases are available to filter against
        if not self._loaded:
            self.load()

        name_re = re.compile(pattern)

        def _source_match(entry_source):
            if source in (None, 'all'):
                return True
            if source == 'session':
                return entry_source == 'session'
            # a file stem or Path: match on the file stem (suffix-insensitive)
            want = Path(str(source)).stem
            if isinstance(entry_source, Path):
                return entry_source.stem == want
            return str(entry_source) == str(source)

        # Order by kind dependency-priority, then name, so the file re-loads
        # sequentially: definitions precede the entries that reference them.
        selected = sorted(
            (r for (k, n), r in self._recipes.items()
             if (kind in ('', 'all') or k == kind)
             and name_re.match(n)
             and _source_match(r.source)),
            key=lambda r: (_KIND_WRITE_ORDER.get(r.kind, 99), r.name),
        )

        out = Path(path).expanduser()
        if out.suffix == '':
            out = out.with_suffix('.agg')
        if not out.is_absolute():
            out = self.user_dir / out.name

        stamp = f'{datetime.now():%Y-%m-%d %H:%M:%S}'
        # Blank line between entries: under the blank-line / `;` statement rule a
        # single newline is a continuation, so entries must be paragraph-separated
        # to re-load as distinct statements (a multi-line port stays one block).
        body = '\n\n'.join(_entry_to_decl(r) for r in selected)

        if mode == 'x' and out.exists():
            raise FileExistsError(
                f'{out} already exists; pass mode="w" to overwrite or '
                f'mode="a" to append.')

        if mode == 'a' and out.exists():
            # Append a dated block at the end; leading newline guarantees a
            # clean separation even if the file did not end with one.
            block = (f'\n# added {stamp} — {len(selected)} program(s); '
                     f'pattern={pattern!r}, kind={kind!r}, source={source!r}\n'
                     f'{body}' + ('\n' if body else ''))
            with out.open('a', encoding='utf-8') as fh:
                fh.write(block)
            logger.info('Appended %d program(s) to %s.', len(selected), out)
        else:
            # 'w', 'x' (new), or 'a' on a missing file: a fresh file with the
            # full provenance header.
            if mode == 'w' and out.exists():
                logger.info('Overwriting %s.', out)
            header = (f'# written by aggregate {self.version} on {stamp}\n'
                      f'# {len(selected)} program(s); pattern={pattern!r}, '
                      f'kind={kind!r}, source={source!r}\n')
            out.write_text(header + body + ('\n' if body else ''), encoding='utf-8')
            logger.info('Wrote %d program(s) to %s.', len(selected), out)
        return out

# Module-level singleton — the canonical user-facing entry point. Importable
# as `from aggregate import build`. log2 / update come from aggregate.config
# ([build] section). Unlike a bare ``Underwriter()`` (which loads nothing),
# ``build`` loads the configured ``build.databases`` (``examples`` by
# default) by passing it explicitly — so config.toml still controls what
# ``build`` knows out of the box.
build = Underwriter(databases=list(get_settings().build.databases) or None, debug=False)
# Sibling entry point for building several objects from one program text.
# Bound to the same singleton so `from aggregate import build_many` returns
# a DataFrame summary across all objects in the input.
build_many = build.build_many


def _refresh_default_underwriter():
    """Refresh the module-level ``build`` underwriter from current settings.

    Called by :func:`aggregate.config.reload_settings`. Mutates the existing
    ``build`` object in place (rather than rebinding the name) so that any
    ``from aggregate import build`` references already held by callers see the
    new configured defaults. Cached recipes are cleared so reconfigured
    databases reload lazily on next access.
    """
    s = get_settings().build
    build.update = s.update
    build.log2 = s.log2
    build._request = list(s.databases) or None
    # reset to as-created so the reconfigured request reloads lazily on next access
    build._recipes = {}
    build.databases = []
    build._loaded = False
# uncomment to create debug build, add to __init__.py
# debug_build = Underwriter(name='Debug', update=True, debug=True, log2=16)
