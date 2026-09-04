"""``chart_approximation``: the five fits and the implied tail.

The teaching chart behind ``approximation_df``
(dev/done/plan-approximate-punchup.md): the realized mass with the family
densities overlaid, and an exceedance panel carrying every law plus the
sub-exponential implied tail ``E[N] * S_X(x)`` off the exact severity
functions.
"""

import json

import numpy as np
import pytest

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, build_chart_doc, canonical_json, chart_approximation,
    primary_chart,
)

_CONT = 'agg CX.Cont 100 claims sev lognorm 50 cv 2 poisson'
_FAMILIES = ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']


@pytest.fixture(scope='module')
def cont():
    return build(_CONT)


def test_registered_and_available(cont):
    assert 'approximation' in available_charts(cont)
    # the aggregate's own picture stays the agg chart
    assert primary_chart(cont) == 'agg'


def test_unavailable_before_update():
    a = build(_CONT, update=False)
    assert 'approximation' not in available_charts(a)


def test_two_panels_and_thirteen_series(cont):
    doc = chart_approximation(cont)
    assert doc.name == 'approximation'
    assert [p.id for p in doc.panels] == ['density', 'tail']
    density = [(s.name, s.role) for s in doc.series
               if s.panel_id == 'density']
    tail = [(s.name, s.role) for s in doc.series if s.panel_id == 'tail']
    assert density == [('Exact', 'density')] + [(f, 'density')
                                                for f in _FAMILIES]
    assert tail == ([('Exact', 'survival')]
                    + [(f, 'survival') for f in _FAMILIES]
                    + [('Implied tail', 'ceiling')])


def test_implied_tail_stays_a_probability(cont):
    """The implied-tail series is trimmed to where E[N] * S(x) <= 1."""
    doc = chart_approximation(cont)
    implied = next(s for s in doc.series if s.name == 'Implied tail')
    assert max(implied.y) <= 1.0
    assert min(implied.y) > 0.0


def test_implied_tail_dominates_deep_in_the_tail(cont):
    """Sub-exponential subject: the exact tail approaches E[N] * S_X(x).

    Where every parametric family gives out, the aggregate's tail is its
    largest claim: ``S_agg(x) ~ E[N] * S_X(x)``. Convergence is slow, so
    the check is same order of magnitude at the deepest common outcome
    drawn (measured ratio ~2 on this fixture at the grid edge).
    """
    doc = chart_approximation(cont)
    exact = next(s for s in doc.series
                 if s.panel_id == 'tail' and s.name == 'Exact')
    implied = next(s for s in doc.series if s.name == 'Implied tail')
    x_common = min(exact.x[-1], implied.x[-1])
    e = np.interp(x_common, exact.x, exact.y)
    i = np.interp(x_common, implied.x, implied.y)
    assert 0.2 * e < i < 5.0 * e


def test_canonical_json_round_trips(cont):
    doc = build_chart_doc(cont, 'approximation')
    json.loads(canonical_json(doc))


def test_signed_subject_drops_log_and_inadmissible_families():
    """A signed (negative-mean) subject keeps a linear outcome axis and
    drops the unshifted positive-support families it cannot admit."""
    a = build('agg CX.S 10 claims ssev -lognorm 10 cv 0.5 poisson')
    doc = build_chart_doc(a, 'approximation')
    outcome = next(ax for ax in doc.axes if ax.id == 'outcome')
    assert outcome.scales == ('linear',)
    density_names = {s.name for s in doc.series if s.panel_id == 'density'}
    assert 'gamma' not in density_names and 'lognorm' not in density_names
    assert {'Exact', 'norm', 'sgamma', 'slognorm'} <= density_names
    json.loads(canonical_json(doc))


def test_left_skew_clamped_subject_serves_all(cont):
    a = build('agg CX.L 1 claim sev 100 * beta 5 1.3 fixed')
    doc = build_chart_doc(a, 'approximation')
    assert len(doc.series) == 13
    json.loads(canonical_json(doc))
