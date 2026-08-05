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
    _drop_raw_moment_rows, _moment_validation_emphasis, _reins_frames,
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


@reins.insurer.register(Aggregate)
def _reins_insurer_aggregate(obj, blocks):
    (stats_name, stats_frame, stats_kw), \
        (summary_name, summary_frame, summary_kw) = blocks
    stats_caption = (
        'Layering analysis by view and layer: per layer columns are '
        'conditional on a loss reaching the layer, the Ceded and Net totals '
        'are unconditional, and the meta block carries attachment and '
        'exhaustion probabilities and loss on line. Raw noncentral moments '
        '(ex1, ex2, ex3) are dropped from this view; the raw perspective '
        'keeps the full store.')
    summary_caption = (
        'Per stage cession impact on the eight validation columns: Change '
        'reads as rebucketing error on the leading gross or subject row and '
        'as the percentage impact of the cession on the ceded and net rows.')
    return [
        (stats_name, _drop_raw_moment_rows(stats_frame),
         dict(stats_kw, caption=stats_caption)),
        (summary_name, summary_frame,
         dict(summary_kw, caption=summary_caption)),
    ]
