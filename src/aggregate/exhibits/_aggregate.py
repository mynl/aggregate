"""Exhibit treatments for :class:`~aggregate._aggregate.Aggregate`.

Its insurer overrides, and the registrations that are not plain
passthroughs. The passthrough declarations (``summary``, ``stats``,
``validation``, ``tail`` and the diagnostics) are the manifest at the foot of
``__init__.py``; this module is where an Aggregate's *business translation*
is edited.
"""

import pandas as pd

from .._aggregate import Aggregate
from ._core import (
    reins, stats, summary, tail, validation,
    _moment_validation_emphasis, _reins_frames,
    _stats_insurer_moment_store, _summary_flags, _tail_flags,
)

reins.register(Aggregate)(_reins_frames)
stats.insurer.register(Aggregate)(_stats_insurer_moment_store)


@summary.insurer.register(Aggregate)
def _summary_insurer_aggregate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Moments and key percentiles by component (Freq, Sev, Agg). '
               'Percentiles are exact grid values. Frequency percentiles are '
               'blank by design: frequency enters through its PGF and no '
               'count distribution is materialized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_summary_flags(df)))]


@tail.insurer.register(Aggregate)
def _tail_insurer_aggregate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Return period ladder: VaR (the quoted number), TVaR (the '
               'priced number), excess VaR over the mean (capital), and '
               'VaR to mean leverage, exact from the FFT grid. The 1 in 200 '
               '(99.5%, Solvency II) and 1 in 250 (99.6%, US capital '
               'adequacy) anchors are emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_tail_flags(df)))]


@validation.insurer.register(Aggregate)
def _validation_insurer_aggregate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Moment QA: reference vs realized FFT estimate with noise '
               'aware relative errors (the economic Gross / Net / Ceded view '
               'when reinsurance is present). Rows failing validation at the '
               'object\'s validation_eps gate are emphasized.')
    flags = _moment_validation_emphasis(df, lambda unit: obj.valid)
    return [(block_name, df, dict(kw, caption=caption, row_flags=flags))]


#: The layer's own contract, in reading order: what was bought, then how
#: likely it is to be touched, then what it costs. The ``loss`` and ``lol``
#: pairing is the point of the tail of this list, and the pair is deliberate:
#: ``loss`` is the placed figure, so a 50% placed layer shows half the dollars,
#: while ``lol`` is mean over share times limit and so is share independent.
_LAYER_TERMS = ('share', 'limit', 'attach', 'pr_attach', 'pr_detach',
                'loss', 'lol', 'output')

#: Where a contract row is read from when it is not a ``meta`` row. ``loss`` is
#: the row's expected aggregate loss, which the store already carries as
#: ``('agg', 'mean')``; no new frame row is created for it.
_LAYER_TERM_SOURCES = {'loss': ('agg', 'mean')}

#: Contract rows served as integers rather than floats. ``output`` is a flag,
#: 0 or 1, and the meta row is always set (``col`` defaults it to 0.0 and never
#: leaves it ``NaN``), so the cast is total. The served TableDoc carries
#: dtypes, so the grid renders it as an integer with no app work.
_LAYER_TERM_INTEGERS = ('output',)

#: The cover repeated onto the moments block, under its own component group.
#: The same three meta rows the contract block leads with: with them present
#: the block stands alone instead of depending on the reader holding the block
#: above in their head.
_LAYER_COVER = ('share', 'limit', 'attach')

#: What the layer does to the three distributions. Component major, so the
#: three readings of one distribution sit together.
_LAYER_COMPONENTS = ('cover', 'freq', 'sev', 'agg')

#: The measures served per component. Frequency is one number here. The layer
#: count is the gross count thinned by the probability a loss reaches the layer
#: (``MomentAggregator.thin_moments``), so its mean is the reading that
#: matters, and it is the reading that makes the block foot: freq mean times
#: sev mean equals agg mean on every row. Its cv and skew describe the shape of
#: that thinning rather than anything about the cover, so they leave this view;
#: the frame keeps all three and the raw perspective still serves them.
_LAYER_MOMENTS = {
    'cover': _LAYER_COVER,
    'freq': ('mean',),
    'sev': ('mean', 'cv', 'skew'),
    'agg': ('mean', 'cv', 'skew'),
}


def _term_rows(frame):
    """The contract block's rows, in reading order, ``loss`` read off ``agg``.

    Most of the contract is the store's ``meta`` block, but ``loss`` is the
    aggregate mean under the insurer's own vocabulary. Reading it here rather
    than adding a ``meta`` row keeps the store as it is: a view may restate
    what the frame already holds (``[Perspective-May-Restructure]``), and the
    number is the same number.

    Returns a frame indexed by the contract's own flat measure names, ready
    for :func:`_layer_rows` to turn over.
    """
    keep = [(_LAYER_TERM_SOURCES.get(m, ('meta', m)), m) for m in _LAYER_TERMS]
    keep = [(src, m) for src, m in keep if src in frame.index]
    out = frame.reindex(index=[src for src, _m in keep])
    out.index = pd.Index([m for _src, m in keep], name='measure')
    return out


def _moment_rows(frame):
    """The consequence block's rows, in component order, cover group included.

    The cover rows live under ``meta`` in the store and are relabeled into a
    ``cover`` component here, which is a view level restatement of rows the
    frame already carries and not a new frame row
    (``[Perspective-May-Restructure]``). Everything else is read straight off
    its own component.
    """
    sources, labels = [], []
    for component in _LAYER_COMPONENTS:
        for measure in _LAYER_MOMENTS[component]:
            sources.append(('meta' if component == 'cover' else component,
                            measure))
            labels.append((component, measure))
    keep = [(s, lab) for s, lab in zip(sources, labels) if s in frame.index]
    out = frame.reindex(index=[s for s, _lab in keep])
    out.index = pd.MultiIndex.from_tuples(
        [lab for _s, lab in keep], names=frame.index.names)
    return out


def _layer_rows(frame, index):
    """Turn a slice of ``reins_stats_df`` over, layers down the rows.

    The frame is built layers across, which is right for the library: a layer
    is a natural column of an analysis and the frame is built once for every
    consumer. A reinsurance reader compares gross to ceded to net, and the eye
    makes that comparison down a column rather than across a row, so the
    insurer view transposes.

    Rows the slice does not carry at all are dropped rather than served empty,
    which is how a one stage program avoids a block of ``NaN`` for the stage
    it does not have.
    """
    present = [k for k in index if k in frame.index]
    out = frame.reindex(index=present).T
    return out.dropna(axis=0, how='all')


@reins.insurer.register(Aggregate)
def _reins_insurer_aggregate(obj, blocks):
    """Split the layering analysis into the contract and its consequence.

    Two different kinds of thing were reading as one table, so a column header
    changed meaning half way down: what the layer **is** (a share of a limit
    over an attachment, and how likely it is to be reached) and what it
    **does** to the frequency, severity and aggregate distributions. Under
    INSURER they are two blocks, layers down the rows in both.

    Three blocks here against RAW's two, per ``[Perspective-May-Restructure]``.
    Neither new block has a frame behind it, which is exactly what an INSURER
    block is for: both are readings of ``reins_stats_df``, which RAW serves
    whole in its own orientation.
    """
    (_stats_name, stats_frame, stats_kw), \
        (summary_name, summary_frame, summary_kw) = blocks
    has_meta = 'meta' in stats_frame.index.get_level_values('component')
    terms = _layer_rows(_term_rows(stats_frame), _LAYER_TERMS) if has_meta \
        else _layer_rows(stats_frame.iloc[:0], _LAYER_TERMS)
    for column in _LAYER_TERM_INTEGERS:
        if column in terms.columns and terms[column].notna().all():
            terms[column] = terms[column].astype('int64')
    moments = _moment_rows(stats_frame)
    terms_caption = (
        'The contract, layer by layer: the share of a limit over an '
        'attachment, the chance a loss reaches it (pr_attach) and exhausts '
        'it (pr_detach), the expected loss and the loss on line. Of that '
        'pair, loss is the placed figure, so a half placed layer shows half '
        'the dollars, while lol divides by share times limit and reads the '
        'same whatever the share. Gross, ceded and net read down the rows, '
        'because that is the comparison and the eye makes it down a column.')
    moments_caption = (
        'What each layer does to the three distributions. The cover columns '
        'repeat the share, limit and attachment, so the block reads on its '
        'own. Layer frequency is the ground up count thinned by the chance a '
        'loss reaches the layer, which is what makes freq mean times sev mean '
        'come to agg mean on every row. Per layer figures are conditional on '
        'a loss reaching the layer; the Ceded and Net rows are '
        'unconditional. Raw noncentral moments (ex1, ex2, ex3) and the '
        'frequency cv and skew are dropped from this view; the raw '
        'perspective keeps the full store. An aggregate stage leaves '
        'frequency and severity blank, having changed neither.')
    summary_caption = (
        'Per stage cession impact on the eight validation columns: Change '
        'reads as rebucketing error on the leading gross or subject row and '
        'as the percentage impact of the cession on the ceded and net rows.')
    return [
        ('reins_layer_terms', terms,
         dict(stats_kw, caption=terms_caption)),
        ('reins_layer_moments',
         _layer_rows(moments, list(moments.index)),
         dict(stats_kw, caption=moments_caption)),
        (summary_name, summary_frame,
         dict(summary_kw, caption=summary_caption)),
    ]
