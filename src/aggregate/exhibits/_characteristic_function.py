"""Exhibit treatments for :class:`~aggregate.characteristic_function.AggregateCHF`.

The four generic exhibits, each a passthrough over the frame of the same name:
``summary`` (one ``Agg`` row of moments and percentiles), ``tail`` (the return
period ladder), ``stats`` (theoretical against grid moments, one block) and
``validation`` (the checks on the grid, one row each). A ``chfagg`` has no
frequency, severity, cession or ledger, so none of the other exhibits apply,
and nothing here registers an insurer override: the frames read the same
under both perspectives.
"""

from ..characteristic_function import AggregateCHF
from ._core import register_simple_exhibit

register_simple_exhibit(
    'summary', 'Summary', 'summary_df', [AggregateCHF],
    caption='Headline moments and key percentiles of the law, read off the '
            'grid the characteristic function was inverted onto. One row: a '
            'law computed this way has no count and no severity. Percentiles '
            'are exact grid values; a moment the law does not have reads as '
            'a statistic of the window.')

register_simple_exhibit(
    'stats', 'Statistics', 'stats_df', [AggregateCHF],
    caption='The theoretical mean, standard deviation, coefficient of '
            'variation and skewness from the cumulants of the law against '
            'the grid\'s, with the relative error between them. A theoretical '
            'moment the law does not have is infinite or undefined, and its '
            'grid value is then a statistic of the window.')

register_simple_exhibit(
    'validation', 'Validation', 'validation_df', [AggregateCHF],
    caption='Checks on the grid: the total mass against what the window '
            'should hold, the most negative bucket, the mass outside the '
            'window against the budget, the modeled edge error and the tilt '
            'of a heavy-tailed law, the frequency truncation, the buckets '
            'per body scale, and each finite moment\'s relative error. A '
            'flagged coverage row carries the grid hint that would cover.')

register_simple_exhibit(
    'tail', 'Return periods', 'tail_df', [AggregateCHF],
    caption='Return period ladder read off the grid: VaR (the quoted number), '
            'TVaR (the priced number), excess VaR over the mean (the capital), '
            'and VaR to mean leverage. On a tilted heavy-tailed grid the TVaR '
            'is the window\'s and omits the mass reported above it.')
