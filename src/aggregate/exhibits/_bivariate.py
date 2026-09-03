"""Exhibit treatments for
:class:`~aggregate.bivariate.BivariateAggregate`.

The ``dependency`` exhibit (joint dependence plus per axis support) and the
gate-derived emphasis its validation moment audit takes.
"""

from ..bivariate import BivariateAggregate
from ._core import dependency, validation


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
    """Moment audit with row emphasis derived from the private gate checks.

    Mirrors ``_validation_insurer_aggregate``: the served frame is the
    public ``validation_df`` (the moment audit) and the emphasis comes from
    the object's own gates rather than from the frame. A failing marginal
    mean gate emphasizes that component's ``Agg`` row; a failing tail
    deficit emphasizes the ``total`` ``Agg`` row.
    """
    block_name, df, kw = blocks[0]
    checks = obj._gate_checks()
    fail_labels = {obj._resolve_handle_label(name)
                   for name in obj.unit_names
                   if not bool(checks.loc[f'marginal mean {name}', 'Pass'])}
    if not bool(checks.loc['tail deficit', 'Pass']):
        fail_labels.add('total')
    flags = {i: ('emphasis',) for i, (comp, part) in enumerate(df.index)
             if part == 'Agg' and comp in fail_labels}
    caption = ('Moment QA for the joint grid: the reference moment against '
               'the realized marginal and total estimates, with noise aware '
               'relative errors. A component failing its marginal mean gate, '
               'or a tail deficit over the gate, emphasizes the matching Agg '
               'row.')
    return [(block_name, df, dict(kw, caption=caption, row_flags=flags))]
