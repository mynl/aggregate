"""Exhibit treatments for :class:`~aggregate._aggregate.Aggregate`.

Its insurer overrides, and the registrations that are not plain
passthroughs. The passthrough declarations (``summary``, ``stats``,
``validation``, ``tail`` and the diagnostics) are the manifest at the foot of
``__init__.py``; this module is where an Aggregate's *business translation*
is edited.
"""

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
#: likely it is to be touched, then what it costs per unit of limit.
_LAYER_TERMS = ('share', 'limit', 'attach', 'pr_attach', 'pr_detach',
                'pr_loss', 'lol', 'output')

#: What the layer does to the three distributions. Component major, so the
#: three readings of one distribution sit together.
_LAYER_COMPONENTS = ('freq', 'sev', 'agg')
_LAYER_MOMENTS = ('mean', 'cv', 'skew')


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
    meta = stats_frame.loc['meta'] if 'meta' in \
        stats_frame.index.get_level_values('component') else stats_frame.iloc[:0]
    moments = stats_frame.reindex(index=[
        (c, m) for c in _LAYER_COMPONENTS for m in _LAYER_MOMENTS])
    terms_caption = (
        'The contract, layer by layer: the share of a limit over an '
        'attachment, the chance a loss reaches it (pr_attach) and exhausts '
        'it (pr_detach), and the loss on line. Gross, ceded and net read '
        'down the rows, because that is the comparison and the eye makes it '
        'down a column.')
    moments_caption = (
        'What each layer does to the three distributions, on the same rows '
        'as the contract above. Per layer figures are conditional on a loss '
        'reaching the layer; the Ceded and Net rows are unconditional. Raw '
        'noncentral moments (ex1, ex2, ex3) are dropped from this view; the '
        'raw perspective keeps the full store. An aggregate stage leaves '
        'frequency and severity blank, having changed neither.')
    summary_caption = (
        'Per stage cession impact on the eight validation columns: Change '
        'reads as rebucketing error on the leading gross or subject row and '
        'as the percentage impact of the cession on the ceded and net rows.')
    return [
        ('reins_layer_terms', _layer_rows(meta, _LAYER_TERMS),
         dict(stats_kw, caption=terms_caption)),
        ('reins_layer_moments',
         _layer_rows(moments, list(moments.index)),
         dict(stats_kw, caption=moments_caption)),
        (summary_name, summary_frame,
         dict(summary_kw, caption=summary_caption)),
    ]
