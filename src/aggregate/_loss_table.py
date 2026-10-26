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
``dfreq [1] dsev``, each year a whole annual outcome. Neither path imports
eltq. Secondary uncertainty, an ELT6 source without ``no2u``, is refused until
``[ELT-Single-Event-Uncertainty]`` lands.

A leaf module: it imports only numpy and pandas.
"""

from dataclasses import dataclass, field

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


class LossTableError(ValueError):
    """An event or year loss table that cannot be built as declared."""


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
    NotImplementedError
        An ELT6 source without ``no2u`` (secondary uncertainty).

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
        For example ``ELT3, 3 events, 1 unit, no secondary uncertainty`` or
        ``YLT1, 4 years, 1 unit, equally likely years``.
    """
    units = len(meta['units'])
    unit_text = f"{units} unit{'s' if units != 1 else ''}"
    if meta['table'] == 'elt':
        return (f"{meta['form']}, {meta['events']:,d} events, {unit_text}, "
                'no secondary uncertainty')
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

    @property
    def where(self):
        """Source position and unit, for messages."""
        return f'{self.table} source {self.index + 1} (unit {self.unit})'


def _source(table, index, raw):
    """Normalize one declared source.

    Parameters
    ----------
    table : {'elt', 'ylt'}
    index : int
        Zero-based position among the body's sources.
    raw : dict
        The parsed source dict.

    Returns
    -------
    _Source
    """
    unit = raw.get('unit') or DEFAULT_UNIT
    common = dict(index=index, table=table, unit=unit,
                  unit_label=raw.get('unit_label'), no2u=bool(raw.get('no2u')))
    if 'path' in raw:
        raise NotImplementedError(
            f"{table} source {index + 1} (unit {unit}): file sources "
            f"(\"{raw['path']}\") are not supported yet; write the table inline.")
    if 'values' in raw:
        values = np.asarray(raw['values'], dtype=float)
        ids = np.array([str(i + 1) for i in range(len(values))], dtype=object)
        return _Source(origin='inline', form='YLT1', ids=ids,
                       fields=values.reshape(-1, 1), implicit_ids=True, **common)
    rows = raw['rows']
    forms = ELT_FORMS if table == 'elt' else YLT_FORMS
    ids = np.array([r[0] for r in rows], dtype=object)
    fields = np.array([r[1:] for r in rows], dtype=float).reshape(len(rows), -1)
    return _Source(origin='inline', form=forms[len(rows[0])], ids=ids,
                   fields=fields, **common)


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
    sources = [_source('elt', k, raw) for k, raw in enumerate(declared)]
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
    if 'ELT6' in effective:
        active = [src.where for src in sources if src.effective_form == 'ELT6']
        if len(effective) > 1:
            raise LossTableError(
                f'{name}: an ELT6 source with secondary uncertainty cannot be '
                'combined with ELT3 or ELT4 sources, including an ELT6 '
                f"converted by no2u. Add no2u to {', '.join(active)} for the "
                'mean-only model.')
        raise NotImplementedError(
            f'{name}: secondary uncertainty (an ELT6 source without no2u) is '
            'not supported yet. Add no2u to every elt clause for the mean-only '
            'model, in which each event row is an atom at its mean.')
    form = 'ELT3' if 'ELT3' in effective else 'ELT4'
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
    event_loss = np.bincount(event_code, weights=mean, minlength=n_events)
    xs, inv = np.unique(event_loss[positive], return_inverse=True)
    weights = np.bincount(inv, weights=event_rate[positive])

    # an event's exposure sums its unit maximums; a combined ELT3 has none,
    # and a partial total would mislead, so it reports no exposure at all
    max_event_exposure = None
    if form == 'ELT4':
        maxes = np.concatenate([src.fields[:, 2] for src in sources])
        exposure = np.bincount(event_code, weights=maxes, minlength=n_events)
        max_event_exposure = float(exposure[positive].max())

    lowered = {'exp_en': total_rate, 'freq_name': 'poisson',
               'sev_name': 'dhistogram', 'sev_xs': xs,
               'sev_ps': weights / total_rate}
    meta = {
        'table': 'elt', 'form': form, 'units': units,
        'unit_labels': unit_labels,
        'sources': [_source_summary(src) for src in sources],
        'rows': int(len(ids)), 'events': int(positive.sum()),
        'events_zero_rate': int((~positive).sum()),
        'rate': total_rate,
        'aal': float(event_rate[positive] @ event_loss[positive]),
        'rate_inexact': int((rate != canon).sum()),
        'severity_atoms': int(len(xs)),
        'max_event_exposure': max_event_exposure,
    }
    return lowered, meta


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
    sources = [_source('ylt', k, raw) for k, raw in enumerate(declared)]
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
    }
    return lowered, meta
