"""What the shipped emitters are allowed to mark.

One negative guard, held across every registered chart at once, because
the percentile lines came off in one edit and could creep back one
emitter at a time. A mark is a line the document asserts permanently, and
the only readings that earn one are the two a reader cannot get by
hovering: the mean, which is a property of the whole distribution, and
break even, which is where the sign of a signed outcome changes. Every
percentile is a point on a drawn curve, so it is a hover and a readout
strip rather than a line.

The vocabulary keeps ``'capital_anchor'`` (:data:`MARK_ROLES` says what a
mark *may* mean, not what these emitters say), so this asserts on the
emitters rather than on the schema.
"""

import matplotlib
import pytest

matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, build_chart_doc, primary_chart,
)

#: Every drawable kind that carries marks at all, one program each.
_PROGRAMS = [
    'agg CM.Cont 100 claims sev lognorm 50 cv 2 poisson',
    'port CM.P agg A 50 claims sev lognorm 50 cv 1.5 poisson '
    'agg B 30 claims sev lognorm 40 cv 1.2 poisson',
    'pnl CM.Book 1000 premium less agg CM.Loss 100 claims '
    'sev lognorm 5 cv 2 poisson',
]


@pytest.fixture(scope='module', params=_PROGRAMS)
def obj(request):
    return build(request.param)


def test_no_emitter_marks_a_percentile(obj):
    for name in available_charts(obj):
        doc = build_chart_doc(obj, name)
        roles = {m.role for m in doc.marks}
        assert 'capital_anchor' not in roles, name
        assert roles <= {'mean', 'break_even'}, name


def test_the_mean_is_marked_once_where_it_is_drawn(obj):
    """The subtraction left the mean alone: exactly one, on the mass panel.

    Asked of the object's **own** picture, which is what ``primary_chart``
    answers. Reading the first available chart instead was the same thing
    until a book gained a second one at a285, and then it was not.
    """
    doc = build_chart_doc(obj, primary_chart(obj))
    means = [m for m in doc.marks if m.role == 'mean']
    assert [(m.panel_id, m.orient, m.faint) for m in means] == [
        ('density', 'v', False)]
