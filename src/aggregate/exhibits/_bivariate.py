"""Exhibit treatments for
:class:`~aggregate.bivariate.BivariateAggregate`.

The ``dependency`` exhibit (joint dependence plus per axis support) and the
check-table emphasis its validation frame takes.
"""

from ..bivariate import BivariateAggregate
from ._core import dependency, validation, _check_table_emphasis


@dependency.register(BivariateAggregate)
def _dependency_frames_bivariate(obj):
    return [('dependency_df', obj.dependency_df, {}),
            ('axis_support_df', obj.axis_support_df, {})]


@validation.insurer.register(BivariateAggregate)
def _validation_insurer_bivariate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Joint grid checks (per axis marginal means and the tail '
               'deficit) with gates and verdicts; failing checks are '
               'emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_check_table_emphasis(df)))]
