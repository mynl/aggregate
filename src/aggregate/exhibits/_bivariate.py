"""Exhibit treatments for
:class:`~aggregate.bivariate.BivariateAggregate`.

The ``dependency`` exhibit (joint dependence plus per axis support) and the
check-table emphasis its validation frame takes.
"""

from ..bivariate import BivariateAggregate
from ._core import dependency, validation, _check_table_emphasis


@dependency.register(BivariateAggregate)
def _dependency_frames_bivariate(obj):
    """Joint dependence, then the per axis support the marginals realized."""
    return [
        ('dependency_df', obj.dependency_df,
         {'caption': 'How the two axes move together, by level: covariance '
                     'and linear correlation, and Kendall tau, which reads '
                     'the ranks rather than the values and so survives the '
                     'marginals being reshaped.'}),
        ('axis_support_df', obj.axis_support_df,
         {'caption': 'Where each marginal actually has mass, with its '
                     'theoretical moments alongside. The full tail class '
                     'ladder lives on each axis\'s own standalone '
                     'aggregate, not here.'}),
    ]


@validation.insurer.register(BivariateAggregate)
def _validation_insurer_bivariate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Joint grid checks (per axis marginal means and the tail '
               'deficit) with gates and verdicts; failing checks are '
               'emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_check_table_emphasis(df)))]
