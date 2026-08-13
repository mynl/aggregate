"""[Chart-Portfolio] the book's mass, and where its losses come from.

The compositor's density and log density were one quantity read two ways,
so log is a declared reading of the one panel. The panel that frees up is
the **kappa** panel, which is the reason a book is not an aggregate with
more curves on it: ``exeqa_i`` is ``E[X_i | X = x]``, what each unit
contributes when the book as a whole lands at ``x``.

The floor on that panel is measured rather than chosen, and the test below
measures it the same way: the unit curves sum to ``x`` by construction, so
the residual of that identity is kappa's own error.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, chart_port, human_strings, primary_chart,
)
from aggregate.charts._emit_portfolio import (  # noqa: E402
    KAPPA_FLOOR, KAPPA_LABEL, TOTAL_NAME,
)

_PORT = ('port CX.P agg A 50 claims sev lognorm 50 cv 1.5 poisson '
         'agg B 30 claims sev lognorm 40 cv 1.2 poisson')


@pytest.fixture(scope='module')
def port():
    return build(_PORT)


def axes_of(doc):
    return {a.id: a for a in doc.axes}


def on(doc, panel):
    return [s for s in doc.series if s.panel_id == panel]


def test_two_panels_over_one_loss_axis(port):
    doc = chart_port(port)
    density, kappa = doc.panels
    assert (density.id, kappa.id) == ('density', 'kappa')
    assert density.x_axis == kappa.x_axis == 'outcome'


def test_the_total_draws_last_so_it_sits_on_top(port):
    """Draw order is meaning: the book over the parts it is made of."""
    for panel in ('density', 'kappa'):
        roles = [s.role for s in on(chart_port(port), panel)]
        assert roles == ['unit', 'unit', 'total']


def test_each_unit_is_drawn_on_its_own_grid(port):
    """A windowed book does not share the portfolio's grid, so nothing is
    resampled onto a common one."""
    doc = chart_port(port)
    for series, unit in zip(on(doc, 'density'), port.agg_list):
        drawn = np.asarray(series.x_values)
        native = port.unit_density(unit.name).index.to_numpy()
        assert drawn[0] >= native[0] and drawn[-1] <= native[-1]
        assert set(np.round(drawn, 9)) <= set(np.round(native, 9))


def test_the_log_density_panel_is_a_reading_of_the_first(port):
    ax = axes_of(chart_port(port))
    assert ax['outcome'].scales == ('linear', 'log')
    assert ax['mass'].scales == ('linear', 'log')
    assert ax['kappa'].scales == ('linear', 'log')


def test_kappa_is_the_conditional_expectation(port):
    doc = chart_port(port)
    kappa = on(doc, 'kappa')
    x = np.asarray(kappa[0].x_values)
    units = np.array([s.y_values for s in kappa if s.role == 'unit'])
    total = np.asarray([s for s in kappa if s.role == 'total'][0].y_values)
    # exeqa_total is E[X | X = x] = x, exactly: the diagonal the unit
    # curves are read as a decomposition of
    np.testing.assert_allclose(total, x)
    np.testing.assert_allclose(units.sum(axis=0), x, rtol=1e-3)


def test_the_floor_is_where_kappa_stops_being_trustworthy(port):
    """The measurement the constant was chosen from, run again here.

    Kappa divides by ``p_total``, so its error is visible in the identity
    the curves must satisfy. At the floor the worst residual is parts per
    thousand; with no floor at all it reaches tens of percent.
    """
    df = port.density_df
    loss = df.loss.to_numpy()
    handles = [a.name for a in port.agg_list]
    summed = df[[f'exeqa_{h}' for h in handles]].sum(axis=1).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = np.abs(summed - loss) / np.where(loss > 0, loss, np.nan)
    kept = df.p_total.to_numpy() > KAPPA_FLOOR
    assert np.nanmax(residual[kept]) < 1e-3
    assert np.nanmax(residual) > 1e-2          # the cliff it is protecting


def test_the_kappa_window_is_the_loss_window(port):
    """The curves sum to the diagonal, so the total is the tallest thing
    on the panel and it is at the window's right edge."""
    ax = axes_of(chart_port(port))
    assert ax['kappa'].suggested_range[1] == \
        pytest.approx(ax['outcome'].suggested_range[1])
    assert ax['kappa'].full_range[1] > ax['kappa'].suggested_range[1]


def test_units_are_named_by_their_resolved_label(port):
    names = {s.name for s in on(chart_port(port), 'density')}
    assert names == {'A', 'B', TOTAL_NAME}


def test_the_mean_is_the_only_mark(port):
    """The kappa panel carries none, and nothing replaces the anchor.

    Reading each unit's share at capital is a hover on the kappa curves
    now, not a line the document asserts.
    """
    (mark,) = chart_port(port).marks
    assert (mark.panel_id, mark.role) == ('density', 'mean')
    assert mark.at == pytest.approx(port.est_m)


def test_tex_is_total(port):
    doc = chart_port(port)
    assert not set(human_strings(doc)) - set(doc.tex)
    assert doc.tex[KAPPA_LABEL].startswith('$')     # and kappa has a real one


def test_capability(port):
    assert available_charts(port) == ['port']
    assert primary_chart(port) == 'port'


def test_it_renders(port):
    from aggregate.plots import plt
    plain, log = port.plot(), port.plot(log=True)
    assert len(plain.axes) == 2
    kappa = plain.axes[1]
    assert kappa.get_ylim()[1] < 2 * port.q(0.999)   # not scaled by the tail
    assert log.axes[1].get_xscale() == log.axes[1].get_yscale() == 'log'
    plt.close('all')
