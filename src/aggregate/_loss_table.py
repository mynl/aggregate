"""Event and year loss tables lowered to ordinary aggregate specs ([DecL-Loss-Tables]).

``agg NAME elt ...`` and ``agg NAME ylt ...`` parse to a spec that records the
sources symbolically, under ``elt_sources`` or ``ylt_sources`` (see
``parser.UnderwritingTransformer``). :func:`resolve` runs at construction, from
``Underwriter._factory``, beside the ``sev agg.NAME`` resolver: it reads each
source, validates the combined table, and returns the spec of an ordinary
``Aggregate`` plus the private provenance key ``_loss_table_meta``. Design and
rulings: ``dev/done/plan-elt.md``.

An ELT lowers to compound Poisson with one count per physical event and the
event's unit losses summed before any occurrence treaty. A YLT lowers to
``dfreq [1] dsev``, each year a whole annual outcome. A deterministic ELT (ELT3,
ELT4, or ELT6 under ``no2u``) lowers to ``dhistogram`` atoms, and neither it
nor a YLT imports eltq. An ELT6 with secondary uncertainty lowers to an
:class:`EltModel` passed as the severity, whose moments come from eltq;
``dev/done/plan-elt-uncertainty.md`` has the design.

A leaf module: it imports only numpy, pandas and the standard library, and
eltq lazily, at the first ELT6 lowering (:func:`import_eltq`).
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import io
from pathlib import Path
import re

import numpy as np
import pandas as pd

#: Spec keys holding the symbolic sources of a loss table body.
SOURCE_KEYS = ('elt_sources', 'ylt_sources')

#: The unit of a source declared without a ``unit`` clause.
DEFAULT_UNIT = 'Total'

#: Relative tolerance for the rates on one event's unit rows to agree. The
#: first row's rate is canonical; rates are never summed or averaged across
#: units.
RATE_RTOL = 1e-12

#: Tolerances for a YLT3 probability column to sum to one, and for two units'
#: probabilities of the same year to agree: ``|a - b| <= PROB_RTOL * b +
#: PROB_ATOL``. Only a residual inside them is normalized away, and it is
#: recorded.
PROB_RTOL = 1e-12
PROB_ATOL = 1e-15

#: Offending rows quoted in one error message.
MAX_EXAMPLES = 5

#: Table form by the number of fields in a row, identifier included.
ELT_FORMS = {3: 'ELT3', 4: 'ELT4', 6: 'ELT6'}
YLT_FORMS = {2: 'YLT2', 3: 'YLT3'}

#: The oldest eltq with every function the ELT6 path calls; 2.13.0 added
#: ``multiunit_moments``.
ELTQ_MIN_VERSION = (2, 13, 0)

#: Relative tolerance on the shared-event moment integration, eltq's
#: ``multiunit_moments`` ``rtol`` on ``C_2`` and ``C_3``.
SHARED_MOMENT_RTOL = 1e-6


class LossTableError(ValueError):
    """An event or year loss table that cannot be built as declared."""


def import_eltq():
    """Import eltq for the secondary uncertainty path, or refuse with guidance.

    Returns
    -------
    module
        The ``eltq`` package, version at least :data:`ELTQ_MIN_VERSION`.

    Raises
    ------
    ImportError
        eltq is missing or older than :data:`ELTQ_MIN_VERSION`. The message
        gives the minimum version, the install command and ``no2u`` as the
        mean-only alternative.

    Notes
    -----
    Called only when an ELT6 source without ``no2u`` is lowered, so parsing,
    ELT3 and ELT4 tables, all-``no2u`` ELT6 tables and YLTs never import eltq.
    Until eltq 2.13 or later is on PyPI it is installed by hand into the
    development environment (``uv pip install`` of a local checkout, a
    maturin build), so it is not a declared dependency.
    """
    need = '.'.join(map(str, ELTQ_MIN_VERSION))
    advice = (f'install eltq {need} or later (pip install "eltq>={need}", or '
              'uv pip install a local eltq checkout), or add no2u to every elt '
              'clause for the mean-only model, in which each event row is an '
              'atom at its mean.')
    try:
        import eltq
    except ImportError as e:
        raise ImportError(
            'Secondary uncertainty (an ELT6 source without no2u) needs the '
            f'eltq package, which is not installed: {advice}') from e
    found = re.match(r'(\d+)\.(\d+)\.(\d+)', str(getattr(eltq, '__version__', '')))
    if found is None or tuple(int(g) for g in found.groups()) < ELTQ_MIN_VERSION:
        raise ImportError(
            'Secondary uncertainty (an ELT6 source without no2u) needs eltq '
            f"{need} or later; found {getattr(eltq, '__version__', 'unknown')}: "
            f'{advice}')
    return eltq


@dataclass(frozen=True, eq=False, repr=False)
class EltModel:
    """The prepared ELT6 table of one aggregate: rows, model moments, digest.

    Built by :func:`_lower_elt` and passed to ``Aggregate`` as ``sev_name``,
    where :class:`~aggregate._severity.SeverityElt` wraps it. Carried by
    reference through ``Aggregate(**spec)`` and ``copy``: the arrays are
    read-only, so a rebuilt object neither rereads a file nor recomputes the
    moments.

    Attributes
    ----------
    event_ids : numpy.ndarray
        Text event identifier of each row (object dtype).
    event_code, unit_code : numpy.ndarray
        Integer event and unit codes of each row; ``units[unit_code]`` names
        the unit.
    units : tuple of str
        Unit names in order of first appearance.
    rate, mean, max, sdi, sdc : numpy.ndarray
        The ELT6 columns as read-only float64 arrays, positive-rate events
        only, stably sorted by event so each event's rows are contiguous. Every
        row of an event carries the event's canonical rate.
    shared : bool
        Whether some event has rows for two or more units.
    moments : eltq.ModelMoments
        The model moments: ``moments_arrays`` for one-row events,
        ``multiunit_moments`` under Burke's coupling for shared events.
    digest : str
        SHA-256 of the canonical row bytes and identifiers.
    eltq_version, eltq_backend : str
        ``eltq.__version__`` and ``eltq.BACKEND`` of the computation.

    Notes
    -----
    The table specifies the Lévy measure
    :math:`\\nu = \\sum_e \\lambda_e \\mathsf P(X_e \\in \\cdot)`, with
    :math:`X_e` the event total. Its mass :math:`\\Lambda` is the Poisson
    rate and :math:`\\nu / \\Lambda` the severity, so the annual cumulants
    are :math:`\\kappa_r = C_r = \\sum_e \\lambda_e \\mathsf P X_e^r`, the
    ``raw_moments`` eltq returns. ``repr`` and ``str`` are compact and
    content addressed, because ``Aggregate._spec_hash`` serializes the spec
    with ``default=str`` and a million-row table must not reach a message.
    """
    event_ids: np.ndarray
    event_code: np.ndarray
    unit_code: np.ndarray
    units: tuple
    rate: np.ndarray
    mean: np.ndarray
    max: np.ndarray
    sdi: np.ndarray
    sdc: np.ndarray
    shared: bool
    moments: object
    digest: str
    eltq_version: str
    eltq_backend: str

    @property
    def rows(self):
        """Number of positive-rate rows."""
        return int(len(self.rate))

    @property
    def events(self):
        """Number of positive-rate events."""
        return int(len(np.unique(self.event_code)))

    def columns(self):
        """The five ELT6 columns as eltq keywords, ``rate`` to ``sd_corr``.

        eltq orders its columns differently from function to function, so
        every call passes them by keyword.
        """
        return dict(rate=self.rate, max_loss=self.max, mean_loss=self.mean,
                    sd_indep=self.sdi, sd_corr=self.sdc)

    def shared_example(self):
        """One shared event and its units, for messages: ``(event, units)``."""
        counts = np.bincount(self.event_code)
        if not (counts > 1).any():
            return None
        code = int(np.flatnonzero(counts > 1)[0])
        rows = self.event_code == code
        return (str(self.event_ids[rows][0]),
                [self.units[u] for u in self.unit_code[rows]])

    def __repr__(self):
        return (f'EltModel(rows={self.rows}, events={self.events}, '
                f'units={len(self.units)}, sha256={self.digest[:8]})')

    __str__ = __repr__


def is_loss_table(spec):
    """Whether ``spec`` is a loss table body that :func:`resolve` lowers.

    Parameters
    ----------
    spec : object
        A parsed spec, or anything else (an ``expr`` spec is a number).

    Returns
    -------
    bool
    """
    return isinstance(spec, dict) and any(k in spec for k in SOURCE_KEYS)


def carries_loss_table(spec):
    """Whether ``spec`` holds an unresolved loss table anywhere inside it.

    Used to refuse a loss table written where nothing resolves it (a
    ``bivariate`` or ``clash`` body), with a clear message.

    Parameters
    ----------
    spec : object
        A parsed spec, walked through nested dicts, lists and tuples.

    Returns
    -------
    bool
    """
    if is_loss_table(spec):
        return True
    if isinstance(spec, dict):
        return any(carries_loss_table(v) for v in spec.values())
    if isinstance(spec, (list, tuple)):
        return any(carries_loss_table(v) for v in spec)
    return False


def resolve(name, spec):
    """Lower a loss table spec to the spec of an ordinary ``Aggregate``.

    Parameters
    ----------
    name : str
        Object name, for messages.
    spec : dict
        A parsed spec carrying ``elt_sources`` or ``ylt_sources``. Not mutated.

    Returns
    -------
    dict
        ``spec`` without the source key, with frequency and severity keys, and
        with ``_loss_table_meta``, the compact audit. The declared sources stay
        on the recipe, which is what ``decl_writer`` renders.

    Raises
    ------
    LossTableError
        Malformed, inconsistent or incomplete input, before anything is built.
    ImportError
        An ELT6 source without ``no2u`` (secondary uncertainty) when eltq is
        missing or too old; see :func:`import_eltq`.

    Notes
    -----
    ELT: let event :math:`e` have rate :math:`\\lambda_e` and unit means
    :math:`\\mu_{eu}`, so its occurrence loss is :math:`m_e = \\sum_u \\mu_{eu}`.
    With :math:`\\Lambda = \\sum_e \\lambda_e` the annual loss is compound
    Poisson with count :math:`\\Lambda` and severity atoms :math:`m_e` of weight
    :math:`\\lambda_e / \\Lambda`, equal losses coalesced. A zero-loss event
    keeps its rate and its zero atom, so the annual zero mass is
    :math:`\\exp[-\\Lambda(1 - p_0)]`. Annual cumulants are
    :math:`\\kappa_r = \\sum_e \\lambda_e m_e^r`.

    ELT6 without ``no2u``: event :math:`e` has a random total :math:`X_e`, a
    beta, two-point or atom law per unit row as eltq classifies it (summed
    under Burke's coupling for a shared event), and the severity is the
    :class:`EltModel` with cumulants :math:`\\kappa_r = \\sum_e \\lambda_e
    \\mathsf P X_e^r`.

    YLT: year :math:`j` has probability :math:`p_j` (:math:`1/n` unless a
    probability column is supplied) and total :math:`a_j = \\sum_u y_{ju}`,
    the units summed within the year before equal totals coalesce. The annual
    law is :math:`\\sum_j p_j \\delta_{a_j}` with population moments; a
    ``dfreq [1]`` count carries it, so no Poisson compounding is added.
    """
    spec = dict(spec)
    elt = spec.pop('elt_sources', None)
    ylt = spec.pop('ylt_sources', None)
    if elt is not None and ylt is not None:
        raise LossTableError(
            f'{name}: an aggregate is declared from elt or from ylt sources, '
            'not both.')
    if elt is not None:
        lowered, meta = _lower_elt(name, elt)
    else:
        _refuse_occurrence(name, spec)
        lowered, meta = _lower_ylt(name, ylt)
    spec.update(lowered)
    spec['_loss_table_meta'] = meta
    return spec


def describe(meta):
    """One-line description of a loss table for ``Aggregate.info``.

    Parameters
    ----------
    meta : dict
        ``_loss_table_meta`` from :func:`resolve`.

    Returns
    -------
    str
        For example ``ELT3, 3 events, 1 unit, no secondary uncertainty``,
        ``ELT6, 3 events, 1 unit, secondary uncertainty`` or ``YLT1, 4 years,
        1 unit, equally likely years``. An ELT6 with a shared event adds
        ``shared events (model moments only)``.
    """
    units = len(meta['units'])
    unit_text = f"{units} unit{'s' if units != 1 else ''}"
    if meta['table'] == 'elt':
        secondary = meta.get('secondary')
        if secondary is None:
            law = 'no secondary uncertainty'
        elif secondary['shared']:
            law = 'secondary uncertainty, shared events (model moments only)'
        else:
            law = 'secondary uncertainty'
        return f"{meta['form']}, {meta['events']:,d} events, {unit_text}, {law}"
    law = ('equally likely years' if meta['probabilities'] == 'uniform'
           else 'supplied year probabilities')
    return f"{meta['form']}, {meta['years']:,d} years, {unit_text}, {law}"


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------

@dataclass
class _Source:
    """One normalized source: identifiers as text, fields as float columns."""
    index: int
    table: str
    unit: str
    unit_label: object
    origin: str
    form: str
    ids: np.ndarray
    fields: np.ndarray
    no2u: bool = False
    implicit_ids: bool = False
    effective_form: str = field(default='')
    file: dict = None

    @property
    def where(self):
        """Source position and unit, for messages."""
        return f'{self.table} source {self.index + 1} (unit {self.unit})'


def _sources(table, index, raw):
    """Normalize one declared source into one source per unit.

    Parameters
    ----------
    table : {'elt', 'ylt'}
    index : int
        Zero-based position among the body's sources.
    raw : dict
        The parsed source dict.

    Returns
    -------
    list of _Source
        One entry, except for a file with a ``unit`` column and no ``unit``
        clause, which gives one entry per unit in order of appearance.
    """
    common = dict(index=index, table=table, unit_label=raw.get('unit_label'),
                  no2u=bool(raw.get('no2u')))
    if 'path' in raw:
        return _file_sources(table, raw, common)
    unit = raw.get('unit') or DEFAULT_UNIT
    if 'values' in raw:
        values = np.asarray(raw['values'], dtype=float)
        ids = np.array([str(i + 1) for i in range(len(values))], dtype=object)
        return [_Source(origin='inline', form='YLT1', ids=ids, unit=unit,
                        fields=values.reshape(-1, 1), implicit_ids=True,
                        **common)]
    rows = raw['rows']
    forms = ELT_FORMS if table == 'elt' else YLT_FORMS
    ids = np.array([r[0] for r in rows], dtype=object)
    fields = np.array([r[1:] for r in rows], dtype=float).reshape(len(rows), -1)
    return [_Source(origin='inline', form=forms[len(rows[0])], ids=ids,
                    unit=unit, fields=fields, **common)]


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------

#: Readers by file extension, ``(pandas reader, keyword arguments)``. A small
#: explicit registry: arbitrary pandas readers, expressions and pickles are not
#: part of the syntax. Delimited text is read entirely as text, so identifiers
#: keep every digit and leading zero; the numeric columns are converted after.
_READERS = {
    '.csv': ('read_csv', {'dtype': str, 'keep_default_na': False,
                          'encoding': 'utf-8-sig'}),
    '.tsv': ('read_csv', {'sep': '\t', 'dtype': str, 'keep_default_na': False,
                          'encoding': 'utf-8-sig'}),
    '.parquet': ('read_parquet', {}),
    '.feather': ('read_feather', {}),
}

#: Header aliases. A header is matched after lowercasing and removing spaces
#: and underscores, so ``Event ID``, ``event_id`` and ``EVENTID`` agree. ``id``
#: is an event in an ELT and a year in a YLT, and ``loss`` is the conditional
#: mean in an ELT (not rate times mean) and the annual loss in a YLT.
_ALIASES = {
    'elt': {
        'event_id': ('id', 'event_id', 'event', 'eventid'),
        'rate': ('rate', 'frequency', 'freq', 'annual_rate'),
        'mean': ('mean', 'mean_loss', 'meanloss', 'loss'),
        'max': ('max', 'max_loss', 'maximum_loss', 'exposure', 'limit'),
        'sdi': ('sdi', 'sd_indep', 'sdind', 'stddevi'),
        'sdc': ('sdc', 'sd_corr', 'sdcor', 'stddevc'),
        'unit': ('unit', 'unit_id'),
    },
    'ylt': {
        'year_id': ('year_id', 'year', 'trial_id', 'trial', 'simulation_id',
                    'sample_id', 'id'),
        'loss': ('loss', 'annual_loss', 'annual_loss_amount'),
        'prob': ('prob', 'probability', 'p'),
        'unit': ('unit', 'unit_id'),
    },
}


def _header_key(text):
    """A header as matched against :data:`_ALIASES`."""
    return re.sub(r'[\s_]+', '', str(text).strip().lower())


def resolve_path(path, base=None):
    """The file a source path names.

    Parameters
    ----------
    path : str
        The path as written in the program.
    base : str, optional
        The directory recorded when the program was registered (see
        :func:`stamp_base`); the current directory when absent.

    Returns
    -------
    pathlib.Path
        ``path`` itself when absolute (after ``~`` expansion), else ``base /
        path``.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(base) / p if base else Path.cwd() / p
    return p


def _read_table(table, raw):
    """Read a source file into a frame with canonical column names.

    Returns
    -------
    (pandas.DataFrame, dict)
        The canonical columns, and the file record for the audit: authored
        and resolved path, size, modification time, SHA-256 digest of the
        bytes read, and any columns ignored.
    """
    where = f"{table} file \"{raw['path']}\""
    path = resolve_path(raw['path'], raw.get('base'))
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise LossTableError(
            f'{where}: unsupported file type {path.suffix!r}; use '
            f"{', '.join(_READERS)}.")
    try:
        data = path.read_bytes()
        stat = path.stat()
    except OSError as e:
        raise LossTableError(f'{where}: cannot read {path} ({e}).') from None
    fn, kwargs = reader
    try:
        df = getattr(pd, fn)(io.BytesIO(data), **kwargs)
    except ImportError as e:
        raise LossTableError(
            f'{where}: reading {path.suffix} files needs an engine that is not '
            f'installed ({e}); pip install pyarrow.') from None
    except Exception as e:
        raise LossTableError(f'{where}: cannot parse {path} ({e}).') from None

    lookup = {_header_key(a): canon
              for canon, aliases in _ALIASES[table].items() for a in aliases}
    rename, ignored = {}, []
    for col in df.columns:
        canon = lookup.get(_header_key(col))
        if canon is None:
            ignored.append(str(col))
        elif canon in rename.values():
            first = next(c for c, v in rename.items() if v == canon)
            raise LossTableError(
                f'{where}: columns {first!r} and {col!r} both mean {canon}.')
        else:
            rename[col] = canon
    df = df[list(rename)].rename(columns=rename)
    record = {'path': raw['path'], 'resolved': path.as_posix(),
              'bytes': int(stat.st_size),
              'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(
                  timespec='seconds'),
              'sha256': hashlib.sha256(data).hexdigest(),
              'ignored_columns': ignored}
    return df, record


def _file_ids(where, column):
    """Exact text identifiers from a file column.

    Integers become decimal text, as an inline integer does; text is kept as
    written, stripped of surrounding space. A floating point column is refused:
    its values may already have lost digits.
    """
    if pd.api.types.is_float_dtype(column):
        raise LossTableError(
            f'{where}: the identifier column holds floating point numbers, '
            'which may already have lost digits; store it as integers or text.')
    if pd.api.types.is_integer_dtype(column):
        return np.array([str(int(v)) for v in column], dtype=object)
    ids = []
    for v in column:
        if not isinstance(v, str):
            raise LossTableError(
                f'{where}: identifier {v!r} is neither an integer nor text.')
        ids.append(v.strip())
    ids = np.array(ids, dtype=object)
    empty = np.array([len(i) == 0 for i in ids], dtype=bool)
    if empty.any():
        rows = np.flatnonzero(empty)[:MAX_EXAMPLES] + 1
        raise LossTableError(
            f"{where}: empty identifier in rows {', '.join(map(str, rows))}.")
    return ids


def _file_sources(table, raw, common):
    """Read one file source and split it by unit.

    The form is set by which loss columns are present, never by the unit or by
    ignored columns: ELT3 ``event_id rate mean``, ELT4 adds ``max``, ELT6 adds
    ``max sdi sdc``; YLT1 ``loss``, YLT2 adds ``year_id``, YLT3 adds ``prob``.
    """
    df, record = _read_table(table, raw)
    where = f"{table} file \"{raw['path']}\""
    cols = set(df.columns)
    if table == 'elt':
        need = ['event_id', 'rate', 'mean']
        missing = [c for c in need if c not in cols]
        if missing:
            raise LossTableError(f"{where}: no {', '.join(missing)} column.")
        sd = {'sdi', 'sdc'} & cols
        if sd and (len(sd) == 1 or 'max' not in cols):
            found = sorted(sd | ({'max'} & cols))
            raise LossTableError(
                f'{where}: secondary uncertainty needs max, sdi and sdc '
                f"columns together; found {', '.join(found)}.")
        value_cols = need[1:] + (['max'] if 'max' in cols else []) + (
            ['sdi', 'sdc'] if sd else [])
        form = {2: 'ELT3', 3: 'ELT4', 5: 'ELT6'}[len(value_cols)]
        id_col = 'event_id'
    else:
        if 'loss' not in cols:
            raise LossTableError(f'{where}: no loss column.')
        if 'prob' in cols and 'year_id' not in cols:
            raise LossTableError(
                f'{where}: a probability column needs a year identifier column.')
        value_cols = ['loss'] + (['prob'] if 'prob' in cols else [])
        id_col = 'year_id' if 'year_id' in cols else None
        form = 'YLT3' if 'prob' in cols else 'YLT2' if id_col else 'YLT1'

    if len(df) == 0:
        raise LossTableError(f'{where} has no rows.')
    fields = np.column_stack([
        pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        for c in value_cols])
    if id_col:
        ids = _file_ids(where, df[id_col])
    else:
        ids = np.array([str(i + 1) for i in range(len(df))], dtype=object)

    clause_unit = raw.get('unit')
    if 'unit' in cols:
        units = df['unit'].astype(str).str.strip().to_numpy(dtype=object)
        if clause_unit is not None and (units != clause_unit).any():
            other = sorted(set(units) - {clause_unit})[:MAX_EXAMPLES]
            raise LossTableError(
                f"{where}: the clause says unit {clause_unit} but the unit "
                f"column also holds {', '.join(other)}.")
        names = list(dict.fromkeys(units))
        if len(names) > 1 and not id_col:
            raise LossTableError(
                f'{where}: a file holding several units needs a year '
                'identifier column to align them.')
    else:
        units = None
        names = [clause_unit or DEFAULT_UNIT]

    out = []
    for name in names:
        mask = slice(None) if units is None else units == name
        out.append(_Source(origin=raw['path'], form=form, ids=ids[mask],
                           fields=fields[mask], unit=name,
                           implicit_ids=id_col is None, file=record, **common))
    return out


def _walk_sources(node, visit):
    """Call ``visit`` on every file source dict inside a parsed spec."""
    if isinstance(node, dict):
        if node.get('table') in ('elt', 'ylt') and 'path' in node:
            visit(node)
        for v in node.values():
            _walk_sources(v, visit)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk_sources(v, visit)


def carries_file_source(spec):
    """Whether ``spec`` holds a file source anywhere inside it."""
    found = []
    _walk_sources(spec, found.append)
    return bool(found)


def stamp_base(spec, base):
    """Record ``base`` on every file source of ``spec`` that has none.

    Called when a parsed statement is registered, so a relative path keeps
    meaning the file it named then: an ``.agg`` file's directory, or the
    working directory of a session. A source merged in from a reference
    already carries its own base and keeps it, which is how the base travels
    through ``agg.NAME``, portfolio units and ``pnl`` engines.

    Parameters
    ----------
    spec : object
        A parsed spec, walked through nested dicts, lists and tuples and
        modified in place.
    base : str or pathlib.Path
        The directory.

    Returns
    -------
    object
        ``spec``.
    """
    base = Path(base).resolve().as_posix()
    _walk_sources(spec, lambda node: node.setdefault('base', base))
    return spec


def with_absolute_paths(spec):
    """A copy of ``spec`` whose relative file paths are made absolute.

    For export to a file in another directory: the relative path written in a
    session or another ``.agg`` file would otherwise name a different file
    beside the export. A source without a recorded base is left as written.
    """
    spec = copy.deepcopy(spec)

    def _absolute(node):
        if node.get('base'):
            node['path'] = resolve_path(node['path'], node['base']).as_posix()

    _walk_sources(spec, _absolute)
    return spec


def _examples(src, mask):
    """Quote up to :data:`MAX_EXAMPLES` rows of ``src`` selected by ``mask``."""
    rows = np.flatnonzero(mask)
    shown = ', '.join(f'row {r + 1} (id {src.ids[r]})'
                      for r in rows[:MAX_EXAMPLES])
    more = f' and {len(rows) - MAX_EXAMPLES} more' if len(rows) > MAX_EXAMPLES else ''
    return shown + more


def _check_nonnegative(name, src, values, what):
    """Refuse a nonfinite or negative entry in one column of a source."""
    bad = ~np.isfinite(values) | (values < 0)
    if bad.any():
        raise LossTableError(
            f'{name}: {src.where}: {what} must be finite and nonnegative; '
            f'see {_examples(src, bad)}.')


def _unit_labels(name, sources):
    """Map each unit to its label, refusing two different explicit labels."""
    labels = {}
    for src in sources:
        if src.unit_label is None:
            labels.setdefault(src.unit, None)
            continue
        if labels.get(src.unit) not in (None, src.unit_label):
            raise LossTableError(
                f'{name}: unit {src.unit} is labeled both '
                f'"{labels[src.unit]}" and "{src.unit_label}".')
        labels[src.unit] = src.unit_label
    return {u: lab for u, lab in labels.items() if lab is not None}


def _source_summary(src):
    """Compact per-source record for the audit."""
    out = {'source': src.index + 1, 'unit': src.unit, 'origin': src.origin,
           'form': src.form, 'rows': int(len(src.ids))}
    if src.table == 'elt':
        out['effective_form'] = src.effective_form
        out['no2u'] = src.no2u
    if src.file:
        out.update({k: src.file[k] for k in
                    ('resolved', 'bytes', 'modified', 'sha256',
                     'ignored_columns')})
    return out


# ----------------------------------------------------------------------
# ELT
# ----------------------------------------------------------------------

def _lower_elt(name, declared):
    """Validate ELT sources and lower them to compound Poisson.

    Returns
    -------
    (dict, dict)
        Frequency and severity spec keys, and the audit record.
    """
    sources = [s for k, raw in enumerate(declared)
               for s in _sources('elt', k, raw)]
    for src in sources:
        if len(src.ids) == 0:
            raise LossTableError(f'{name}: {src.where} has no rows.')
        # validate every supplied field before projecting: no2u does not
        # excuse malformed input
        for j, what in enumerate(('rate', 'mean', 'max', 'sdi', 'sdc')[
                :src.fields.shape[1]]):
            _check_nonnegative(name, src, src.fields[:, j], what)
        if src.fields.shape[1] >= 3:
            over = src.fields[:, 1] > src.fields[:, 2]
            if over.any():
                raise LossTableError(
                    f'{name}: {src.where}: mean exceeds max; see '
                    f'{_examples(src, over)}.')
        src.effective_form = ('ELT4' if src.form == 'ELT6' and src.no2u
                              else src.form)

    effective = {src.effective_form for src in sources}
    if 'ELT6' in effective and len(effective) > 1:
        active = [src.where for src in sources if src.effective_form == 'ELT6']
        raise LossTableError(
            f'{name}: an ELT6 source with secondary uncertainty cannot be '
            'combined with ELT3 or ELT4 sources, including an ELT6 '
            f"converted by no2u. Add no2u to {', '.join(active)} for the "
            'mean-only model.')
    form = ('ELT6' if 'ELT6' in effective
            else 'ELT3' if 'ELT3' in effective else 'ELT4')
    unit_labels = _unit_labels(name, sources)

    ids = np.concatenate([src.ids for src in sources])
    rate = np.concatenate([src.fields[:, 0] for src in sources])
    mean = np.concatenate([src.fields[:, 1] for src in sources])
    owner = np.concatenate([np.full(len(src.ids), k)
                            for k, src in enumerate(sources)])
    row = np.concatenate([np.arange(len(src.ids)) for src in sources])
    units = list(dict.fromkeys(src.unit for src in sources))
    unit_code = np.array([units.index(sources[k].unit) for k in owner])
    event_code, event_ids = pd.factorize(ids)
    n_events = len(event_ids)

    def _quote(mask):
        idx = np.flatnonzero(mask)
        shown = ', '.join(f'{sources[owner[i]].where} row {row[i] + 1} '
                          f'(event {ids[i]})' for i in idx[:MAX_EXAMPLES])
        more = (f' and {len(idx) - MAX_EXAMPLES} more'
                if len(idx) > MAX_EXAMPLES else '')
        return shown + more

    # one row per (event, unit): summing duplicates would double exposure
    key = event_code.astype(np.int64) * len(units) + unit_code
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    dup = counts[inverse] > 1
    if dup.any():
        raise LossTableError(
            f'{name}: an event appears more than once for one unit: '
            f'{_quote(dup)}.')

    # one rate per event, the first row's, never a sum over units
    _, first = np.unique(event_code, return_index=True)
    event_rate = rate[first]
    canon = event_rate[event_code]
    disagree = np.abs(rate - canon) > RATE_RTOL * canon
    if disagree.any():
        raise LossTableError(
            f"{name}: an event's unit rows carry different rates (an event "
            f'has one rate, whatever its units): {_quote(disagree)}.')

    positive = event_rate > 0
    total_rate = float(event_rate[positive].sum())
    if total_rate <= 0:
        raise LossTableError(
            f'{name}: the table has no event with a positive rate.')

    # an event's exposure sums its unit maximums; a combined ELT3 has none,
    # and a partial total would mislead, so it reports no exposure at all
    max_event_exposure = None
    if form != 'ELT3':
        maxes = np.concatenate([src.fields[:, 2] for src in sources])
        exposure = np.bincount(event_code, weights=maxes, minlength=n_events)
        max_event_exposure = float(exposure[positive].max())

    meta = {
        'table': 'elt', 'form': form, 'units': units,
        'unit_labels': unit_labels,
        'sources': [_source_summary(src) for src in sources],
        'rows': int(len(ids)), 'events': int(positive.sum()),
        'events_zero_rate': int((~positive).sum()),
        'rate': total_rate,
        'rate_inexact': int((rate != canon).sum()),
        'max_event_exposure': max_event_exposure,
    }

    if form == 'ELT6':
        fields = np.concatenate([src.fields for src in sources])
        model = _elt_model(name, ids, event_code, unit_code, units, canon,
                           positive[event_code], fields)
        mom = model.moments
        c = [float(x) for x in mom.raw_moments]
        lowered = {'exp_en': float(mom.total_rate), 'freq_name': 'poisson',
                   'sev_name': model}
        meta.update({
            'rate': float(mom.total_rate), 'aal': c[0],
            'severity_atoms': None,
            'moments': _cumulant_moments(c, float(mom.total_rate)),
            'mass_error': None,
            'secondary': {
                'shared': model.shared,
                'beta_rows': int(mom.beta_rows),
                'two_point_rows': int(mom.two_point_rows),
                'atom_rows': int(mom.atom_rows),
                'rate_at_zero': float(mom.rate_at_zero),
                'max_support': float(mom.max_support),
                'method': str(mom.method),
                'achieved_error': [float(e) for e in mom.achieved_error],
                'converged': bool(mom.converged),
                'rtol': None if mom.rtol is None else float(mom.rtol),
                'eltq_version': model.eltq_version,
                'eltq_backend': model.eltq_backend,
            },
        })
        return lowered, meta

    event_loss = np.bincount(event_code, weights=mean, minlength=n_events)
    xs, inv = np.unique(event_loss[positive], return_inverse=True)
    weights = np.bincount(inv, weights=event_rate[positive])
    lowered = {'exp_en': total_rate, 'freq_name': 'poisson',
               'sev_name': 'dhistogram', 'sev_xs': xs,
               'sev_ps': weights / total_rate}
    meta.update({
        'aal': float(event_rate[positive] @ event_loss[positive]),
        'severity_atoms': int(len(xs)),
        'moments': _elt_moments(event_rate[positive], event_loss[positive]),
        'mass_error': abs(float(weights.sum()) / total_rate - 1),
    })
    return lowered, meta


def _elt_model(name, ids, event_code, unit_code, units, canon, keep, fields):
    """Prepare the :class:`EltModel` of an ELT6 table and compute its moments.

    Parameters
    ----------
    name : str
        Object name, for messages.
    ids : numpy.ndarray
        Text event identifier of each row.
    event_code, unit_code : numpy.ndarray
        Integer event and unit code of each row.
    units : list of str
        Unit names; ``units[unit_code]`` names a row's unit.
    canon : numpy.ndarray
        The canonical (first row's) rate of each row's event.
    keep : numpy.ndarray
        Boolean, the rows whose event has a positive rate.
    fields : numpy.ndarray
        The ``rate mean max sdi sdc`` columns, one row per table row.

    Returns
    -------
    EltModel

    Raises
    ------
    LossTableError
        eltq dropped a row of the validated table, or a model moment leaves
        double precision.

    Notes
    -----
    eltq's defaults agree with the parent plan's rulings, and are passed
    explicitly anyway so that a later change of default cannot silently
    change the model: ``variance_policy='bernoulli'`` (an inadmissible
    variance becomes the two-point law on :math:`\\{0, M\\}` with the same
    mean), ``mean_is_loss_rate=False``, and Burke's coupling for a shared
    event with :data:`SHARED_MOMENT_RTOL`. Rows are stably sorted by event
    so each event's rows are contiguous, as ``multiunit_moments`` requires.
    """
    eltq = import_eltq()
    rows = np.flatnonzero(keep)
    rows = rows[np.argsort(event_code[rows], kind='stable')]
    codes, _ = pd.factorize(event_code[rows])

    def _frozen(values, dtype=np.float64):
        out = np.ascontiguousarray(values, dtype=dtype)
        out.setflags(write=False)
        return out

    rate = _frozen(canon[rows])
    mean, mx, sdi, sdc = (_frozen(fields[rows, j]) for j in (1, 2, 3, 4))
    event_ids = _frozen(ids[rows], dtype=object)
    event_code = _frozen(codes, dtype=np.int64)
    unit_code = _frozen(unit_code[rows], dtype=np.int64)
    shared = bool((np.bincount(event_code) > 1).any())
    if shared:
        moments = eltq.multiunit_moments(
            event_id=event_ids,
            unit_id=np.asarray(units, dtype=object)[unit_code],
            rate=rate, mean_loss=mean, sd_indep=sdi, sd_corr=sdc, max_loss=mx,
            coupling='burke', variance_policy='bernoulli',
            rtol=SHARED_MOMENT_RTOL)
    else:
        moments = eltq.moments_arrays(
            rate=rate, max_loss=mx, mean_loss=mean, sd_indep=sdi, sd_corr=sdc,
            mean_is_loss_rate=False, variance_policy='bernoulli',
            event_ids=list(event_ids))
    if moments.dropped_rows:
        found = '; '.join(f'{k}: {v.count} ({", ".join(map(str, v.examples))})'
                          for k, v in moments.diagnostics.items() if v.count)
        raise LossTableError(
            f'{name}: eltq dropped {moments.dropped_rows} row(s) of a '
            'validated table, which is a bug to report, not a smaller '
            f'portfolio: {found or "no diagnostics"}.')
    if any(moments.moment_overflow):
        orders = [str(r + 1) for r, o in enumerate(moments.moment_overflow) if o]
        raise LossTableError(
            f"{name}: the model moment of order {', '.join(orders)} leaves "
            'double precision; rescale the losses (for example to thousands).')

    digest = hashlib.sha256()
    for column in (rate, mean, mx, sdi, sdc, unit_code):
        digest.update(column.tobytes())
    digest.update('\x1f'.join(map(str, event_ids)).encode('utf-8'))
    digest.update('\x1f'.join(units).encode('utf-8'))
    return EltModel(
        event_ids=event_ids, event_code=event_code, unit_code=unit_code,
        units=tuple(units), rate=rate, mean=mean, max=mx, sdi=sdi, sdc=sdc,
        shared=shared, moments=moments, digest=digest.hexdigest(),
        eltq_version=str(eltq.__version__), eltq_backend=str(eltq.BACKEND))


# ----------------------------------------------------------------------
# YLT
# ----------------------------------------------------------------------

def _refuse_occurrence(name, spec):
    """Refuse occurrence terms on a YLT, including those added by reference."""
    def _stated(v):
        return v is not None and not (isinstance(v, (str, list, tuple))
                                      and len(v) == 0)

    if any(k.startswith('occ_') and _stated(v) for k, v in spec.items()):
        raise LossTableError(
            f'{name}: a ylt holds annual outcomes, so occurrence reinsurance '
            'cannot apply: an annual total does not identify its occurrences. '
            "Write annual terms as 'aggregate net of ...', or declare the "
            'events with elt for occurrence terms.')


def _close(a, b):
    """Elementwise agreement under :data:`PROB_RTOL` and :data:`PROB_ATOL`."""
    return np.abs(a - b) <= PROB_RTOL * np.abs(b) + PROB_ATOL


def _lower_ylt(name, declared):
    """Validate YLT sources, align their years and lower to ``dfreq [1] dsev``.

    Returns
    -------
    (dict, dict)
        Frequency and severity spec keys, and the audit record.
    """
    sources = [s for k, raw in enumerate(declared)
               for s in _sources('ylt', k, raw)]
    seen = {}
    for src in sources:
        if src.unit in seen:
            raise LossTableError(
                f'{name}: unit {src.unit} has two ylt sources; each source is '
                'a complete YLT, so give each unit one.')
        seen[src.unit] = src
        if len(src.ids) == 0:
            raise LossTableError(f'{name}: {src.where} has no years.')
        _check_nonnegative(name, src, src.fields[:, 0], 'losses')
        _, inverse, counts = np.unique(src.ids.astype(str), return_inverse=True,
                                       return_counts=True)
        dup = counts[inverse] > 1
        if dup.any():
            raise LossTableError(
                f'{name}: {src.where}: a year appears more than once; see '
                f'{_examples(src, dup)}.')
    unit_labels = _unit_labels(name, sources)

    # align every source to the first one's year order
    ref = sources[0]
    n = len(ref.ids)
    position = {y: j for j, y in enumerate(ref.ids)}
    aligned = []
    for src in sources:
        idx = np.array([position.get(y, -1) for y in src.ids])
        missing = sorted(set(ref.ids) - set(src.ids))
        if len(src.ids) != n or (idx < 0).any() or missing:
            extra = [y for y, j in zip(src.ids, idx) if j < 0]
            detail = '; '.join(part for part in (
                f"missing years {', '.join(missing[:MAX_EXAMPLES])}"
                if missing else '',
                f"years not in {ref.where}: {', '.join(extra[:MAX_EXAMPLES])}"
                if extra else '') if part)
            raise LossTableError(
                f'{name}: every unit supplies the same complete set of years, '
                f'zero-loss years included; {src.where} has {len(src.ids)} '
                f'years against {n} in {ref.where} ({detail or "different years"}).')
        order = np.empty(n, dtype=int)
        order[idx] = np.arange(n)
        aligned.append(order)

    # one canonical probability per year, counted once
    residual = 0.0
    canon = canon_src = None
    for src, order in zip(sources, aligned):
        if src.form != 'YLT3':
            continue
        _check_nonnegative(name, src, src.fields[:, 1], 'probabilities')
        p = src.fields[order, 1]
        total = float(p.sum())
        if total == 0:
            raise LossTableError(f'{name}: {src.where}: every probability is zero.')
        if not _close(np.array(total), np.array(1.0)):
            raise LossTableError(
                f'{name}: {src.where}: the probabilities sum to {total!r}, not '
                'one. Each row is a year with its probability; counts, rates '
                'and weights are not probabilities.')
        p = p / total
        if canon is None:
            canon, canon_src, residual = p, src, total - 1.0
        elif not _close(p, canon).all():
            raise LossTableError(
                f'{name}: {src.where} gives a year a different probability '
                f'from {canon_src.where}; a year has one probability, whatever '
                'its units.')
    uniform = np.full(n, 1.0 / n)
    if canon is None:
        p = uniform
    else:
        p = canon
        if any(src.form != 'YLT3' for src in sources) and not _close(p, uniform).all():
            raise LossTableError(
                f'{name}: a ylt without probabilities makes its years equally '
                'likely, which the supplied probabilities are not. Give every '
                'unit the same probabilities, or none.')

    totals = np.zeros(n)
    for src, order in zip(sources, aligned):
        totals += src.fields[order, 0]
    support = p > 0
    xs, inv = np.unique(totals[support], return_inverse=True)
    ps = np.bincount(inv, weights=p[support])

    if canon is not None:
        form = 'YLT3'
    elif any(not src.implicit_ids for src in sources):
        form = 'YLT2'
    else:
        form = 'YLT1'
    if len(sources) == 1:
        alignment = 'single unit'
    elif all(src.implicit_ids for src in sources):
        alignment = 'position'
    elif not any(src.implicit_ids for src in sources):
        alignment = 'year id'
    else:
        alignment = 'year id and position'
    zero = support & (totals == 0)
    lowered = {'freq_name': 'empirical', 'freq_a': np.array([1.0]),
               'freq_b': np.array([1.0]), 'exp_en': -1,
               'sev_name': 'dhistogram', 'sev_xs': xs, 'sev_ps': ps}
    meta = {
        'table': 'ylt', 'form': form,
        'probabilities': 'uniform' if canon is None else 'supplied',
        'alignment': alignment, 'units': [src.unit for src in sources],
        'unit_labels': unit_labels,
        'sources': [_source_summary(src) for src in sources],
        'years': int(n), 'years_zero_probability': int((~support).sum()),
        'zero_years': int(zero.sum()),
        'zero_probability': float(p[zero].sum()),
        'outcomes': int(len(xs)), 'probability_residual': float(residual),
        'mean': float(p @ totals),
        'moments': _ylt_moments(p, totals),
        'mass_error': abs(float(ps.sum()) - 1),
    }
    return lowered, meta


# ----------------------------------------------------------------------
# Moments of the table, computed from its rows
# ----------------------------------------------------------------------

def _mean_cv_skew(e1, e2, e3):
    """Mean, CV and skewness from the first three raw moments.

    A zero variance gives CV 0 (``nan`` at mean 0) and skewness ``nan``, the
    degenerate conventions; the variance is floored at zero against rounding.
    """
    var = max(e2 - e1 * e1, 0.0)
    sd = var ** 0.5
    cv = sd / e1 if e1 else np.nan
    skew = (e3 - 3 * e1 * e2 + 2 * e1 ** 3) / sd ** 3 if sd > 0 else np.nan
    return {'mean': float(e1), 'cv': float(cv), 'skew': float(skew)}


def _elt_moments(rate, loss):
    """Severity and annual moments of a deterministic ELT, from its events.

    Notes
    -----
    With event rates :math:`\\lambda_e`, occurrence losses :math:`m_e` and
    :math:`C_r = \\sum_e \\lambda_e m_e^r`, the severity raw moments are
    :math:`C_r / \\Lambda` and the annual cumulants of the compound Poisson are
    :math:`\\kappa_r = C_r`, so the annual mean is :math:`C_1`, the CV
    :math:`\\sqrt{C_2} / C_1` and the skewness :math:`C_3 / C_2^{3/2}`. Computed
    from the events before equal losses coalesce, so comparing them with the
    lowered severity checks the coalescing.
    """
    return _cumulant_moments([float(rate @ loss ** r) for r in (1, 2, 3)],
                             float(rate.sum()))


def _cumulant_moments(c, total):
    """Severity and annual mean, CV and skewness from :math:`C_1, C_2, C_3`.

    Parameters
    ----------
    c : list of float
        :math:`C_r = \\sum_e \\lambda_e \\mathsf P X_e^r`, the annual cumulants
        of the compound Poisson.
    total : float
        The total event rate :math:`\\Lambda`.

    Returns
    -------
    dict
        ``{'severity': {...}, 'annual': {...}}``, each with ``mean``, ``cv``
        and ``skew``. The severity raw moments are :math:`C_r / \\Lambda`; the
        annual mean is :math:`C_1`, the CV :math:`\\sqrt{C_2} / C_1` and the
        skewness :math:`C_3 / C_2^{3/2}`.
    """
    sev = _mean_cv_skew(*(x / total for x in c))
    annual = {'mean': c[0],
              'cv': c[1] ** 0.5 / c[0] if c[0] else np.nan,
              'skew': c[2] / c[1] ** 1.5 if c[1] > 0 else np.nan}
    return {'severity': sev, 'annual': annual}


def _ylt_moments(prob, totals):
    """Annual moments of a YLT, from its years, before equal totals coalesce.

    The one synthetic claim is the whole year, so the severity is the annual
    outcome and the two agree. Population moments: the weights are the year
    probabilities, with no small-sample correction.
    """
    m = _mean_cv_skew(*(float(prob @ totals ** r) for r in (1, 2, 3)))
    return {'severity': dict(m), 'annual': dict(m)}


# ----------------------------------------------------------------------
# Report frames
# ----------------------------------------------------------------------

#: Index and columns of the three report frames, also those of their empty
#: forms.
SUMMARY_INDEX = ['part', 'item']
STATS_INDEX = ['component', 'measure']
STATS_COLUMNS = ['table', 'model', 'estimate', 'error']
VALIDATION_COLUMNS = ['check', 'value', 'tolerance', 'status', 'detail']

#: Tolerance on the lowered severity's total weight against the event rate
#: (ELT) or one (YLT): pure rounding, so a few ulps of a sum.
MASS_TOLERANCE = 1e-12

#: A table CV or skewness (dimensionless) at most this in magnitude is zero up
#: to rounding, so its estimate error is absolute, not relative: a symmetric
#: beta row's skewness computes to about 1e-14, not 0.
SHAPE_NOISE = 1e-12


def _row_laws(secondary):
    """The rows by effective law, for example ``2 beta, 1 two-point, 0 atom``."""
    return (f"{secondary['beta_rows']} beta, {secondary['two_point_rows']} "
            f"two-point, {secondary['atom_rows']} atom")


def _grid_rows(agg, add, against, eps):
    """The validation rows of an ELT6 grid, from the last quantization.

    The tail rate and the three moment shares against the ``[ELT-Grid]``
    targets, the quantized rate mass against :math:`\\Lambda`, and eltq's
    secondary mean error relative to :math:`C_1` against ``eps``. Read from
    ``SeverityElt.last_quantization``, so they describe the grid actually
    used; ``not computed`` before ``update`` (and for a shared table, which
    never reaches a grid), never ``pass``.
    """
    q = _last_quantization(agg) if agg.bs > 0 else None
    checks = [('tail rate above edge', 'rate_above', 'rate_target',
               'annual rate of events above the severity grid edge'),
              ('tail mean share', 0, 'share_target',
               'excess of C1 above the top node, moved onto it'),
              ('tail second moment share', 1, 'share_target',
               'excess of C2 above the top node, moved onto it'),
              ('tail third moment share', 2, 'share_target',
               'excess of C3 above the top node, moved onto it')]
    for check, key, target, detail in checks:
        if q is None:
            add(check, np.nan, np.nan, 'not computed', 'needs update')
            continue
        value = q[key] if isinstance(key, str) else q['shares'][key]
        tol = q[target]
        add(check, value, tol, 'pass' if value <= tol else 'warning',
            f"{detail}; edge {q['edge']:,.6g}")
    if q is None:
        add('quantized mass against event rate', np.nan, MASS_TOLERANCE,
            'not computed', 'needs update')
        add('secondary mean error', np.nan, eps, 'not computed', 'needs update')
        return
    against('quantized mass against event rate', q['mass_error'],
            MASS_TOLERANCE, 'eltq severity rate mass against the event rate, '
            'before dividing by it')
    against('secondary mean error', q['relative_mean_error'], eps,
            "eltq's quantized severity mean against the model, relative to "
            'C1: rounding, interpolation and tail together')


def _last_quantization(agg):
    """The ``last_quantization`` record of an ELT severity, or None."""
    for s in (agg.sevs if agg.sevs is not None else []):
        if getattr(s, 'sev_kind', '') == 'elt':
            return s.last_quantization
    return None


def _moment_method(secondary):
    """How eltq computed the model moments, for the summary frame."""
    if secondary['method'] == 'closed_form':
        return 'closed form'
    e2, e3 = secondary['achieved_error']
    return (f'factor integration, achieved error {e2:.2g} (C2) and '
            f'{e3:.2g} (C3)')


def summary_frame(agg):
    """The ``loss_table_summary_df`` of an aggregate.

    Parameters
    ----------
    agg : Aggregate

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(part, item)`` with one ``value`` column. Part ``table``
        describes the combined table, part ``grid`` the selected grid (``n/a``
        before ``update``), and one part per source and unit its origin, form,
        rows and, for a file, the resolved path, size, modification time,
        SHA-256 digest and ignored columns. Empty for an aggregate not
        declared from a loss table.
    """
    meta = agg._loss_table_meta
    rows = []
    if meta:
        def add(part, item, value):
            rows.append((part, item, value))

        add('table', 'kind', meta['table'])
        add('table', 'form', meta['form'])
        add('table', 'units', ', '.join(meta['units']))
        add('table', 'unit labels', ', '.join(
            f'{u}: {lab}' for u, lab in meta['unit_labels'].items()) or 'none')
        add('table', 'sources', len({s['source'] for s in meta['sources']}))
        if meta['table'] == 'elt':
            no2u = any(s['no2u'] and s['form'] == 'ELT6'
                       for s in meta['sources'])
            secondary = meta.get('secondary')
            add('table', 'rows', meta['rows'])
            add('table', 'events', meta['events'])
            add('table', 'events with zero rate', meta['events_zero_rate'])
            add('table', 'event rate', meta['rate'])
            add('table', 'mean annual loss', meta['aal'])
            add('table', 'severity atoms',
                'n/a' if meta['severity_atoms'] is None
                else meta['severity_atoms'])
            add('table', 'largest event exposure',
                'n/a' if meta['max_event_exposure'] is None
                else meta['max_event_exposure'])
            if secondary is None:
                add('table', 'secondary uncertainty',
                    'removed by no2u' if no2u else 'none')
            else:
                add('table', 'secondary uncertainty', _row_laws(secondary))
                add('table', 'largest event support', secondary['max_support'])
                add('table', 'model moments', _moment_method(secondary))
                add('table', 'eltq', f"{secondary['eltq_version']} "
                    f"({secondary['eltq_backend']} backend)")
        else:
            add('table', 'years', meta['years'])
            add('table', 'years with zero probability',
                meta['years_zero_probability'])
            add('table', 'zero-loss years', meta['zero_years'])
            add('table', 'probability of zero', meta['zero_probability'])
            add('table', 'year probabilities',
                'equally likely' if meta['probabilities'] == 'uniform'
                else 'supplied')
            add('table', 'alignment', meta['alignment'])
            add('table', 'annual outcomes', meta['outcomes'])
            add('table', 'mean annual loss', meta['mean'])
        updated = agg.bs > 0
        add('grid', 'bs', agg.bs if updated else 'n/a')
        add('grid', 'log2', agg.log2 if updated else 'n/a')
        if meta.get('secondary') is not None:
            q = _last_quantization(agg) if updated else None
            add('grid', 'severity edge', 'n/a' if q is None else q['edge'])
            add('grid', 'tail rate above edge',
                'n/a' if q is None else q['rate_above'])
        for s in meta['sources']:
            part = f"source {s['source']}, unit {s['unit']}"
            add(part, 'origin', s['origin'])
            add(part, 'form', s['form'])
            if meta['table'] == 'elt':
                add(part, 'effective form', s['effective_form'])
                add(part, 'no2u', s['no2u'])
            add(part, 'rows', s['rows'])
            for key in ('resolved', 'bytes', 'modified', 'sha256'):
                if key in s:
                    add(part, key, s[key])
            if 'ignored_columns' in s:
                add(part, 'ignored columns',
                    ', '.join(s['ignored_columns']) or 'none')
    index = pd.MultiIndex.from_tuples([r[:2] for r in rows],
                                      names=SUMMARY_INDEX)
    return pd.DataFrame({'value': [r[2] for r in rows]}, index=index,
                        dtype=object)


def stats_frame(agg):
    """The ``loss_table_stats_df`` of an aggregate.

    Parameters
    ----------
    agg : Aggregate

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(component, measure)``, components ``severity`` and
        ``annual``, measures ``mean``, ``cv`` and ``skew``, gross of
        reinsurance. ``table`` is computed from the table's rows, ``model`` is
        the aggregate's theoretical value (``stats_df`` column ``mixed``),
        ``estimate`` the value from the discretized density (``stats_df``
        column ``gross_empirical``, ``nan`` before ``update``), and ``error``
        is ``estimate / table - 1``, or ``estimate - table`` where the table
        value is zero (a symmetric skewness, a degenerate CV), or a CV or
        skewness within :data:`SHAPE_NOISE` of zero. Empty for an aggregate
        not declared from a loss table.

    Notes
    -----
    ``table`` and ``model`` are both exact for a deterministic table, but they
    are computed independently, before and after equal losses coalesce, so
    their agreement checks the lowering. ``estimate`` differs from them by the
    ordinary discretization and FFT error that ``validation_df`` reports.

    For an ELT6 with secondary uncertainty, ``table`` holds eltq's model
    moments (the severity from ``severity_moments``, the annual values from
    ``raw_moments``) and ``model`` the same values through ``stats_df``, so
    their agreement checks the plumbing; ``estimate`` against ``table`` then
    measures quantization, overflow and FFT error together.
    """
    meta = agg._loss_table_meta
    if not meta:
        return pd.DataFrame(
            columns=STATS_COLUMNS, dtype=float,
            index=pd.MultiIndex.from_tuples([], names=STATS_INDEX))
    updated = agg.bs > 0
    rows, index = [], []
    for component, key in (('severity', 'sev'), ('annual', 'agg')):
        for measure in ('mean', 'cv', 'skew'):
            table = meta['moments'][component][measure]
            model = float(agg.stats_df.loc[(key, measure), 'mixed'])
            est = (float(agg.stats_df.loc[(key, measure), 'gross_empirical'])
                   if updated else np.nan)
            if not (np.isfinite(table) and np.isfinite(est)):
                error = np.nan
            elif table == 0 or (measure != 'mean'
                                and abs(table) <= SHAPE_NOISE):
                error = est - table
            else:
                error = est / table - 1
            index.append((component, measure))
            rows.append([table, model, est, error])
    return pd.DataFrame(rows, columns=STATS_COLUMNS,
                        index=pd.MultiIndex.from_tuples(index, names=STATS_INDEX))


def validation_frame(agg):
    """The ``loss_table_validation_df`` of an aggregate.

    Parameters
    ----------
    agg : Aggregate

    Returns
    -------
    pandas.DataFrame
        One row per check: ``check``, the measured ``value``, the
        ``tolerance``, a ``status`` of ``pass``, ``warning`` or ``not
        computed`` (a grid check before ``update``, or a measure undefined for
        the table), and a ``detail``. Input that fails a structural check
        raises when the aggregate is built, so those rows record what was
        checked and passed. Empty for an aggregate not declared from a loss
        table.
    """
    meta = agg._loss_table_meta
    rows = []
    if meta:
        eps = agg.validation_eps

        def add(check, value, tolerance, status, detail):
            rows.append([check, value, tolerance, status, detail])

        def against(check, value, tolerance, detail):
            if not np.isfinite(value):
                add(check, value, tolerance, 'not computed', detail)
            else:
                add(check, value, tolerance,
                    'pass' if abs(value) <= tolerance else 'warning', detail)

        if meta['table'] == 'elt':
            projected = sum(s['form'] != s['effective_form']
                            for s in meta['sources'])
            add('form projection', projected, np.nan, 'pass',
                f'{projected} ELT6 source(s) read as ELT4 under no2u; '
                f"combined form {meta['form']}")
            add('one row per event and unit', 0, 0, 'pass',
                'no event repeats within a unit')
            add('event rate agreement', meta['rate_inexact'], RATE_RTOL,
                'pass', 'unit rows whose rate differs from the first row of '
                'the event, within the relative tolerance')
            add('zero-rate events', meta['events_zero_rate'], np.nan, 'pass',
                'excluded from the count and the moments')
            secondary = meta.get('secondary')
            if secondary is None:
                against('severity mass', meta['mass_error'], MASS_TOLERANCE,
                        'lowered severity weight against the event rate')
            else:
                add('row classification', secondary['two_point_rows'], np.nan,
                    'pass', f'{_row_laws(secondary)} rows; a two-point row '
                    'replaces an inadmissible beta variance, keeping the mean')
                if secondary['shared']:
                    error = max(secondary['achieved_error'])
                    add('moment integration', error, secondary['rtol'],
                        'pass' if secondary['converged'] else 'warning',
                        f"{_moment_method(secondary)}; shared events under "
                        "Burke's coupling")
                _grid_rows(agg, add, against, eps)
        else:
            add('year completeness', meta['years'], np.nan, 'pass',
                'every unit lists the same years')
            against('probability sum', meta['probability_residual'],
                    PROB_RTOL + PROB_ATOL,
                    'roundoff normalized away; a material shortfall raises')
            against('severity mass', meta['mass_error'], MASS_TOLERANCE,
                    'lowered outcome probabilities against one')
        stats = stats_frame(agg)
        for component in ('severity', 'annual'):
            table = stats.loc[(component, 'mean'), 'table']
            model = stats.loc[(component, 'mean'), 'model']
            against(f'{component} mean, table against model',
                    model / table - 1 if table else model - table, eps,
                    'eltq model moments against stats_df'
                    if meta.get('secondary') is not None
                    else 'independent sums before and after coalescing')
        for component in ('severity', 'annual'):
            for measure in ('mean', 'cv', 'skew'):
                if agg.bs <= 0:
                    detail = 'needs update'
                elif not np.isfinite(stats.loc[(component, measure), 'table']):
                    detail = 'undefined for this table (zero variance)'
                else:
                    detail = 'discretized density against the table'
                against(f'{component} {measure} estimate',
                        stats.loc[(component, measure), 'error'], eps, detail)
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)
