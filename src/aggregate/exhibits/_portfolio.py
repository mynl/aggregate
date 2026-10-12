"""Exhibit treatments for :class:`~aggregate._portfolio.Portfolio`.

Its insurer overrides, and the registrations that are not plain
passthroughs. See ``_aggregate.py`` for the shape; the two modules differ
only in what the frames look like, not in how they are declared.
"""

from .._portfolio import Portfolio
from ._core import (
    reins, stats, summary, tail, validation,
    _drop_raw_moment_rows, _moment_validation_emphasis, _reins_frames,
    _reins_summary_flags, _stats_insurer_moment_store, _summary_flags,
    _tail_flags,
)

reins.register(Portfolio)(_reins_frames)
stats.insurer.register(Portfolio)(_stats_insurer_moment_store)


@summary.insurer.register(Portfolio)
def _summary_insurer_portfolio(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Moments and key percentiles per unit plus the portfolio '
               'total; the total block carries the Agg row only (a portfolio '
               'has no single Freq or Sev). Frequency percentiles are blank '
               'by design: frequency enters through its PGF and no count '
               'distribution is materialized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_summary_flags(df)))]


@tail.insurer.register(Portfolio)
def _tail_insurer_portfolio(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Return period ladder per unit plus the portfolio total: VaR, '
               'TVaR, excess VaR over the mean (capital), and VaR to mean '
               'leverage, exact from the FFT grid. The 1 in 200 (99.5%, '
               'Solvency II) and 1 in 250 (99.6%, US capital adequacy) '
               'anchors are emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_tail_flags(df)))]


@validation.insurer.register(Portfolio)
def _validation_insurer_portfolio(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = (
        'Moment-based validation per unit plus the portfolio total: computes '
        'the theoretical mean (EX), CV and skewness (Sk) of each unit\'s '
        'severity and compound distribution from the relationships between '
        'frequency, severity and compound moments, and the total\'s by '
        'combining the units as independent, which is how the FFT convolves '
        'them. The theoretical calculation uses analytic expressions for the '
        'frequency and for lognormal, gamma, exponential and Pareto '
        'severities, and high-accuracy numerical integration for other '
        'severities. The FFT approximation\'s moments (Est) are compared to the '
        'theoretical ones, and Err is the noise-aware relative error between '
        'them. The theoretical calculation is independent of the FFT, so close '
        'agreement is strong evidence the approximation is valid. When any '
        'unit can take negative values, SD replaces CV throughout. Rows failing '
        'a unit\'s validation_eps gate, or the portfolio\'s for the total, are '
        'emphasized. If any unit cedes, every block switches to the cession '
        'view: EX, Est and Err become Subject, Net (or Ceded) and Change, the '
        'effect of the cession, so the table no longer validates: validate '
        'the gross portfolio.')
    by_unit = {a.name: a.valid for a in obj}
    by_unit['total'] = obj.valid
    flags = _moment_validation_emphasis(df, by_unit.get)
    return [(block_name, df, dict(kw, caption=caption, row_flags=flags))]


@reins.insurer.register(Portfolio)
def _reins_insurer_portfolio(obj, blocks):
    (stats_name, stats_frame, stats_kw), \
        (summary_name, summary_frame, summary_kw) = blocks
    stats_caption = (
        'End to end gross, ceded and net aggregate moments per unit plus '
        'the convolved portfolio total. Raw noncentral moments (ex1, ex2, '
        'ex3) are dropped from this view; the raw perspective keeps the '
        'full store.')
    summary_caption = (
        'Per unit cession impact plus the end to end portfolio total: '
        'Change reads as the percentage impact of the reinsurance program '
        'on each moment (0 on the gross reference rows).')
    return [
        (stats_name, _drop_raw_moment_rows(stats_frame),
         dict(stats_kw, caption=stats_caption)),
        (summary_name, summary_frame,
         dict(summary_kw, caption=summary_caption,
              row_flags=_reins_summary_flags(summary_frame))),
    ]
