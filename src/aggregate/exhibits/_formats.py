"""``aggregate.exhibits._formats`` -- the format sheets, loaded and resolved.

.. warning::

   **Provisional, in the sense of PEP 411**, along with the rest of
   :mod:`aggregate.exhibits`: not part of the 1.0 API contract, and subject to
   change in a minor release with no deprecation period. See
   :doc:`/3_reference/3_x_API_Stability`.

Where the library keeps its reading of a number. Two YAML sheets ship with
the package, ``formats/formats-raw.yaml`` (the default reading of every named
column) and ``formats/formats-insurer.yaml`` (an **overlay**, holding only the
entries where the business reading differs). They replace seven module level
dicts applied block by block, and they buy two things a dict could not:

* one place to answer "how does a CV read", rather than wherever someone
  remembered to declare it, with room for the comment saying why; and
* a **registry of the column vocabulary**. An entry asserts that a column
  label means one thing across the package, which turns naming drift (two
  frames using one word for different units) into a test failure rather than
  a quiet inconsistency. The sweep that makes it bite is
  ``test_every_served_column_has_a_declared_reading`` in
  ``tests/test_exhibits.py``.

Nothing here reaches the wire differently. Formats travel inside the served
``TableDoc`` per column exactly as before, so this is invisible to every
consumer: it is a change in where the library keeps its own mind.

Resolution, low to high (``dev/plan-formats.md`` decision 3):

1. greater_tables dtype and tag inference, the guess;
2. ``formats-raw.yaml``, the shipped default reading;
3. ``formats-insurer.yaml``, when the perspective is INSURER;
4. the same two file names in ``~/.aggregate`` and then in the working
   directory, nearest winning;
5. an explicit ``formatters`` entry in a frames builder.

Step 4 is the ``.agg`` rule (author, 2026-08-14): overriding a shipped sheet
is the same act as overriding a shipped DecL database, so it is the same
three stops the :class:`~aggregate.Underwriter` resolves a database over
(the working directory, :attr:`~aggregate.Underwriter.user_dir`, then the
shipped :attr:`~aggregate.Underwriter.default_dir`), and it needs no second
mechanism. A stop with no file contributes nothing.

Notes
-----
**Patterns.** A sheet's ``patterns:`` section keys on a regular expression
rather than a label, matched whole against the displayed column label, and
says how a **family** reads: ``e[0-9]+\\.m[0-9]+`` is the analytic mixture
components of the moment store, one per component, so their number is a
property of the program and no list of exact entries can cover them. It also
collapses a cross product, since anything spelled ``<basis> CV`` is a CV.
Nothing of this reaches greater_tables, which looks formats up by exact
label: the expansion happens here, against the block's own columns, and GT
only ever sees the words it will match. Precedence, high to low: a scoped
exact entry, a global exact entry, a scoped pattern, a global pattern. Among
patterns the first match in file order wins, so a narrow pattern belongs
above a broad one, and ``'.*'`` scoped to one exhibit is how that exhibit
says "everything else here reads like this".

**Styles.** A sheet's ``styles:`` section names a reading once (``ratio:
'.1%'``) and every column that wears it points at the name, so the house
ratio precision is one line rather than fourteen. Styles merge across the
layers before column entries are resolved against them, which is what lets
the insurer overlay redefine ``money`` in one line and move every column
that points at it. Four style names are special because greater_tables owns
them as semantic column tags (``ratio``, ``year``, ``date``, ``raw``): a
column pointing at one of those is **stamped with the tag** as well as the
format, so a consumer learns the column's kind and not only its reading.
Style references are one level deep: a style's value is a format, never
another style name.

**Why a sheet cannot hold a callable.** Every value goes through
``greater_tables.engine.formats.parse_sugar`` at load, which accepts sugar,
an int, a mapping of ``FormatSpec`` fields, or a ``FormatSpec``, and nothing
else. A callable formatter never reaches the IR, so a document formatted by
one cannot be re-rendered by a client; a sheet that cannot express one can
only say things a client can act on. Validating at load also means a bad
string raises **once**, naming the file and the key, rather than per cell.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from ..config import user_dir

__all__ = [
    'FORMATS_DIR', 'SHEET_FILENAMES', 'TAG_STYLES', 'FormatSheet',
    'format_sheet', 'reload_format_sheets', 'sheet_paths',
]

#: Subdirectory inside the installed ``aggregate`` package holding the
#: shipped sheets, the sibling of ``agg/`` and for the same reason.
FORMATS_DIR = 'formats'

#: Sheet file name per perspective. One name at every stop of the search
#: path: a user sheet is the same file in a nearer place, exactly as a user
#: ``.agg`` database is.
SHEET_FILENAMES = {'raw': 'formats-raw.yaml',
                   'insurer': 'formats-insurer.yaml'}

#: greater_tables' semantic column tags. A style with one of these names
#: stamps the tag as well as the format, which is how the sheet retired the
#: two hand written ``ratio_cols`` call sites.
TAG_STYLES = ('ratio', 'year', 'date', 'raw')

#: The only top level keys a sheet may carry. Anything else is a typo that
#: would otherwise be silently ignored, which is the failure mode a config
#: file cannot afford.
SECTIONS = ('styles', 'columns', 'patterns', 'exhibits')

#: The keys a scoped (per exhibit) section may carry, the same two kinds of
#: entry as the top level. The structure is uniform on purpose: a scoped
#: section is a sheet in miniature, so learning one teaches the other.
SCOPED_SECTIONS = ('columns', 'patterns')


@dataclass(frozen=True)
class FormatSheet:
    """One perspective's resolved reading of the column vocabulary.

    Attributes
    ----------
    perspective : str
        The perspective value this sheet was resolved for (``'raw'``,
        ``'insurer'``).
    columns : dict
        ``{column label: format}``, styles already resolved to formats.
    tags : dict
        ``{column label: tag name}`` for the columns whose style is one of
        :data:`TAG_STYLES`.
    exhibits : dict
        ``{exhibit name: (columns, tags)}``, the scoped exceptions.
    sources : tuple of pathlib.Path
        The sheet files this was merged from, nearest last. Empty only if
        the package data is missing, which is a broken install.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    perspective: str
    columns: dict
    tags: dict
    patterns: tuple
    exhibits: dict
    sources: tuple

    def labels(self, exhibit=None):
        """Every column label this sheet names **exactly**, one exhibit's view.

        The global vocabulary plus that exhibit's scoped entries. Callers use
        it to work out how a host's relabeling moves the sheet's own keys, so
        patterns are deliberately absent: a pattern is matched against the
        displayed label after relabeling and has no key to move.
        """
        scoped, _tags, _patterns = self.exhibits.get(exhibit, ({}, {}, ()))
        return list(dict.fromkeys([*self.columns, *scoped]))

    def declares(self, label, exhibit=None):
        """True when this sheet has a reading for ``label``, exact or by pattern.

        What the vocabulary sweep asks. An exact entry and a pattern match are
        equally a declaration: the pattern says how a family of labels reads,
        which is a stronger statement than the same reading written out N
        times, not a weaker one.
        """
        scoped, _tags, scoped_patterns = self.exhibits.get(exhibit, ({}, {}, ()))
        if label in scoped or label in self.columns:
            return True
        return any(regex.fullmatch(str(label))
                   for regex, _value, _tag in (*scoped_patterns, *self.patterns))

    def block(self, exhibit=None, rename=None, labels=()):
        """The greater_tables kwargs this sheet contributes to one block.

        Parameters
        ----------
        exhibit : str, optional
            Exhibit registry name; its scoped section (if any) is laid over
            the global entries, replacing both the format and the tag.
        rename : callable, optional
            ``label -> displayed label``, applied to the **exact** keys. The
            sheets are written in the library's own words and greater_tables
            keys on the displayed label, so a served frame that went through a
            ``renamer`` needs its sheet keys to travel with it. Without this a
            renamer touching ``CV`` would silently detach its format, which is
            what the module level dicts did.
        labels : iterable, optional
            The block's own column labels, which is what patterns are matched
            against. They arrive already relabeled, so a pattern needs no
            ``rename`` trip: it is written against what a reader sees.

        Returns
        -------
        (dict, dict)
            The ``formatters`` mapping, and ``{'<tag>_cols': [labels]}`` for
            each tag the sheet stamps. Labels the frame does not carry are
            harmless in both: greater_tables looks formats up by label and
            resolves a selector list against the columns it actually has.

        Notes
        -----
        Precedence inside the sheet, high to low: a scoped exact entry, a
        global exact entry, a scoped pattern, a global pattern. Exact beats
        pattern because a pattern is a rule about a family and an exact entry
        is a statement about one word, and the word is the more specific of
        the two. Among patterns the first match in file order wins, so a
        narrow pattern belongs above a broad one.
        """
        scoped_columns, scoped_tags, scoped_patterns = self.exhibits.get(
            exhibit, ({}, {}, ()))
        columns, tags = {}, {}
        ordered = (*scoped_patterns, *self.patterns)
        for label in labels:
            for regex, value, tag in ordered:
                if regex.fullmatch(str(label)):
                    columns[label] = value
                    if tag is not None:
                        tags[label] = tag
                    break
        exact = dict(self.columns)
        exact_tags = dict(self.tags)
        for label in scoped_columns:
            exact_tags.pop(label, None)
        exact.update(scoped_columns)
        exact_tags.update(scoped_tags)
        if rename is not None:
            exact = {rename(label): value for label, value in exact.items()}
            exact_tags = {rename(label): tag for label, tag in exact_tags.items()}
        for label in exact:
            tags.pop(label, None)      # an exact entry replaces a pattern's tag
        columns.update(exact)
        tags.update(exact_tags)
        selectors = {}
        for label, tag in tags.items():
            selectors.setdefault(f'{tag}_cols', []).append(label)
        return columns, {tag: sorted(labels_, key=str)
                         for tag, labels_ in selectors.items()}


def sheet_paths(kind):
    """The search path for one sheet kind, package first and cwd last.

    Parameters
    ----------
    kind : {'raw', 'insurer'}

    Returns
    -------
    list of pathlib.Path
        Every candidate location, existing or not, in override order: the
        shipped sheet, then ``~/.aggregate``, then the working directory.
        Unlike :attr:`aggregate.Underwriter.user_dir` this never creates the
        user directory; a sheet search should not have a side effect.
    """
    name = SHEET_FILENAMES[kind]
    return [Path(str(files('aggregate'))) / FORMATS_DIR / name,
            user_dir() / name,
            Path.cwd() / name]


def _read(path):
    """Parse one sheet file, or raise ``ValueError`` naming it.

    ``yaml.safe_load`` only: a format sheet is data, and nothing in it should
    be able to construct a Python object.
    """
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'{path}: not valid YAML: {exc}') from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f'{path}: a format sheet is a mapping with the sections '
            f'{", ".join(SECTIONS)}; got {type(data).__name__}')
    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ValueError(
            f'{path}: unknown section(s) {", ".join(map(repr, unknown))}; '
            f'a format sheet carries {", ".join(SECTIONS)}')
    for exhibit, entries in (data.get('exhibits') or {}).items():
        if not isinstance(entries, dict):
            raise ValueError(
                f'{path}: exhibit {exhibit!r} is a mapping of '
                f'{", ".join(SCOPED_SECTIONS)}; got {type(entries).__name__}')
        unknown = sorted(set(entries) - set(SCOPED_SECTIONS))
        if unknown:
            raise ValueError(
                f'{path}: exhibit {exhibit!r} carries '
                f'{", ".join(map(repr, unknown))}; a scoped section holds '
                f'{", ".join(SCOPED_SECTIONS)}, so a column entry sits under '
                '`columns:` rather than directly under the exhibit name')
    return data


def _declaration(kind):
    """Merge one sheet kind across the search path, nearest winning.

    Merging is per key and per section, not whole file, so a user sheet
    holding one line changes that one reading and inherits the rest. The
    ``exhibits`` section merges one level deeper, per exhibit.
    """
    merged = {'styles': {}, 'columns': {}, 'patterns': {}, 'exhibits': {}}
    sources = []
    for path in sheet_paths(kind):
        if not path.is_file():
            continue
        data = _read(path)
        sources.append(path)
        merged['styles'].update(data.get('styles') or {})
        merged['columns'].update(data.get('columns') or {})
        merged['patterns'].update(data.get('patterns') or {})
        for exhibit, entries in (data.get('exhibits') or {}).items():
            scoped = merged['exhibits'].setdefault(
                exhibit, {'columns': {}, 'patterns': {}})
            scoped['columns'].update(entries.get('columns') or {})
            scoped['patterns'].update(entries.get('patterns') or {})
    return merged, sources


def _validate(value, label, sources):
    """Check one resolved entry through greater_tables, or raise.

    Deferred import, twice over: ``_core`` imports this module, so the import
    of its greater_tables helper cannot sit at module scope, and the frame
    stage must stay free of greater_tables in any case. ``parse_sugar`` is a
    declared public name of ``greater_tables.engine.formats``.
    """
    from ._core import _import_greater_tables
    _import_greater_tables()
    from greater_tables.engine.formats import parse_sugar
    where = ', '.join(str(p) for p in sources) or 'the shipped sheets'
    if callable(value):
        raise ValueError(
            f'{where}: entry {label!r} is a callable. A format sheet holds '
            'only readings a client can re-render, and a callable never '
            'reaches the IR.')
    try:
        parse_sugar(value)
    except Exception as exc:
        raise ValueError(
            f'{where}: entry {label!r} is not a format greater_tables '
            f'understands ({value!r}): {exc}') from None


def _resolve(declaration, sources):
    """Resolve style references and validate, one section at a time."""
    styles = declaration['styles']
    for name, value in styles.items():
        _validate(value, f'styles.{name}', sources)

    def dereference(key, value, where):
        """A style name becomes its format, and a tag style stamps its tag."""
        tag = None
        if isinstance(value, str) and value in styles:
            tag = value if value in TAG_STYLES else None
            value = styles[value]
        _validate(value, f'{where}{key}', sources)
        return value, tag

    def resolve_columns(entries, where):
        columns, tags = {}, {}
        for label, value in entries.items():
            value, tag = dereference(label, value, where)
            if tag is not None:
                tags[label] = tag
            columns[label] = value
        return columns, tags

    def resolve_patterns(entries, where):
        out = []
        for pattern, value in entries.items():
            value, tag = dereference(pattern, value, where)
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                where_files = ', '.join(str(p) for p in sources)
                raise ValueError(
                    f'{where_files}: pattern {where}{pattern!r} is not a '
                    f'regular expression: {exc}') from None
            out.append((regex, value, tag))
        return tuple(out)

    columns, tags = resolve_columns(declaration['columns'], '')
    patterns = resolve_patterns(declaration['patterns'], 'patterns.')
    exhibits = {}
    for name, entries in declaration['exhibits'].items():
        where = f'exhibits.{name}.'
        scoped_columns, scoped_tags = resolve_columns(entries['columns'], where)
        exhibits[name] = (scoped_columns, scoped_tags,
                          resolve_patterns(entries['patterns'],
                                           f'{where}patterns.'))
    return columns, tags, patterns, exhibits


@functools.lru_cache(maxsize=None)
def _sheet(perspective, cwd):
    """Build one resolved sheet. Cached on (perspective, working directory).

    ``cwd`` is a cache key rather than an argument because the search path
    ends there: a session that changes directory is looking at a different
    override, and one that does not pays for the parse once.
    """
    declaration, sources = _declaration('raw')
    if perspective == 'insurer':
        overlay, overlay_sources = _declaration('insurer')
        declaration['styles'].update(overlay['styles'])
        declaration['columns'].update(overlay['columns'])
        declaration['patterns'].update(overlay['patterns'])
        for exhibit, entries in overlay['exhibits'].items():
            scoped = declaration['exhibits'].setdefault(
                exhibit, {'columns': {}, 'patterns': {}})
            scoped['columns'].update(entries['columns'])
            scoped['patterns'].update(entries['patterns'])
        sources = sources + overlay_sources
    columns, tags, patterns, exhibits = _resolve(declaration, sources)
    return FormatSheet(perspective=perspective, columns=columns, tags=tags,
                       patterns=patterns, exhibits=exhibits,
                       sources=tuple(sources))


def format_sheet(perspective='raw'):
    """The resolved format sheet for one perspective.

    Parameters
    ----------
    perspective : Perspective or str, default 'raw'
        Only ``insurer`` has an overlay; every other perspective resolves the
        raw sheet, which is the same default rule the exhibits themselves
        follow (INSURER equals RAW unless an override is registered).

    Returns
    -------
    FormatSheet

    Examples
    --------
    ::

        from aggregate.exhibits import format_sheet
        format_sheet('raw').columns['CV']        # '.1%'
        format_sheet('insurer').columns['L']     # ',.2f'

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    value = getattr(perspective, 'value', perspective)
    return _sheet(str(value).lower(), str(Path.cwd()))


def reload_format_sheets():
    """Drop the cached sheets, so an edited sheet is picked up in session.

    The same courtesy :func:`aggregate.config.reload_settings` does for the
    config file, and needed for the same reason: a sheet is a file a user
    edits while the interpreter is running.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    _sheet.cache_clear()
