"""Exhibit treatments for :class:`~aggregate.characteristic_function.AggregateCHF`.

The four generic exhibits, each a passthrough over the frame of the same name,
in ``Aggregate``'s shapes ([CHF-Aggregate-Frames]): ``summary`` (moments and
percentiles by ``Freq``, ``Sev`` and ``Agg``, the first two empty), ``tail``
(the return period ladder), ``stats`` (the moment store) and ``validation``
(the law's moments against the grid's), with ``bs_window`` and
``tail_behavior`` from 1.39.0. The insurer treatments are
``Aggregate``'s: the summary's ``Agg`` row is flagged as the total and the
moment store drops its raw moment rows, and the validation's ``Agg`` row is
emphasized when ``valid`` fails it (1.37.0). A ``chfagg`` has no frequency,
severity, cession or ledger, so none of the other exhibits apply.
"""

from ..characteristic_function import AggregateCHF
from ._core import (register_simple_exhibit, stats, summary, validation,
                    _moment_validation_emphasis, _perspectives_updated,
                    _stats_insurer_moment_store, _summary_flags)

register_simple_exhibit(
    'summary', 'Summary', 'summary_df', [AggregateCHF],
    caption='Headline moments and key percentiles of the law, read off the '
            'grid the characteristic function was inverted onto. The Freq and '
            'Sev rows are empty: a law computed this way has no count and no '
            'severity. Percentiles are exact grid values; a moment the law '
            'does not have reads as a statistic of the window.')

register_simple_exhibit(
    'stats', 'Statistics', 'stats_df', [AggregateCHF],
    caption='Canonical moment store by component and measure: the law\'s '
            'raw and central moments from its cumulants in the theoretical '
            'columns, the grid\'s in empirical, and the relative error between '
            'them. The frequency and severity blocks are empty. A theoretical '
            'moment the law does not have is infinite or undefined, and its '
            'grid value is then a statistic of the window.')

register_simple_exhibit(
    'validation', 'Validation', 'validation_df', [AggregateCHF],
    caption='Moment validation: the law\'s theoretical mean (EX), CV or SD '
            'and skewness (Sk) from its cumulants against the grid\'s (Est), '
            'with the noise-aware relative error (Err). The Freq and Sev rows '
            'are empty. The checks on the inversion itself (coverage, edge '
            'error, truncation, resolution) are rows of info.')

register_simple_exhibit(
    'tail', 'Return periods', 'tail_df', [AggregateCHF],
    caption='Return period ladder read off the grid: VaR (the quoted number), '
            'TVaR (the priced number), excess VaR over the mean (the capital), '
            'and VaR to mean leverage. On a tilted heavy-tailed grid the TVaR '
            'is the window\'s and omits the mass reported above it.')

register_simple_exhibit(
    'bs_window', 'Grid sizing', 'bs_window_df', [AggregateCHF],
    predicate=_perspectives_updated,
    caption="How the grid was chosen: the window edges from the law's "
            'closed-form tails, the tilted bucket cap where it binds, a given '
            'grid, the recommended grid, and the grid used. Coverage is the '
            'fraction of the law the window holds; clipped is the mass a tilted '
            'grid leaves out above its window rather than wrapping in.')

register_simple_exhibit(
    'tail_behavior', 'Tail behavior', 'tail_behavior_df', [AggregateCHF],
    predicate=_perspectives_updated,
    caption='What each tail of the law looks like: its support, the tail class '
            'on each side from its closed forms (a power tail, a support bound, '
            'or the domain of its cumulant generating function), and the '
            'coefficient of variation. The frequency and severity rows are '
            'empty: a law computed this way has neither.')

stats.insurer.register(AggregateCHF)(_stats_insurer_moment_store)


@summary.insurer.register(AggregateCHF)
def _summary_insurer_chf(obj, blocks):
    """The summary with ``Aggregate``'s row flags: the ``Agg`` row is the total."""
    block_name, df, kw = blocks[0]
    return [(block_name, df, dict(kw, row_flags=_summary_flags(df)))]


@validation.insurer.register(AggregateCHF)
def _validation_insurer_chf(obj, blocks):
    """The validation with ``Aggregate``'s emphasis: the ``Agg`` row when :attr:`~AggregateCHF.valid` fails it."""
    block_name, df, kw = blocks[0]
    flags = _moment_validation_emphasis(df, lambda unit: obj.valid)
    return [(block_name, df, dict(kw, row_flags=flags))]
