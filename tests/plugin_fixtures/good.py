"""A well-behaved fixture plugin: one chart, one exhibit, two declared leaves."""

from aggregate.charts import ChartAxis, ChartDoc, ChartSeries, Panel, register_chart
from aggregate.exhibits import register_simple_exhibit
from aggregate.plugins import PluginLeaf

from ._toy import Toy, emitter

CHART_NAME = 'toy_plugin_chart'
EXHIBIT_NAME = 'toy_plugin_exhibit'

chart_toy = emitter(CHART_NAME)


@chart_toy.register(Toy)
def _chart_toy(obj, **options):
    """One xy panel over ``toy_df``, the smallest complete chart document."""
    frame = obj.toy_df
    return ChartDoc(
        name=CHART_NAME,
        title=f'Toy: {obj.name}',
        axes=(ChartAxis(id='x', label='a'), ChartAxis(id='y', label='b')),
        panels=(Panel(id='p', kind='xy', x_axis='x', y_axis='y'),),
        series=(ChartSeries(name='b', role='primary', panel_id='p',
                            x=tuple(frame['a']), y=tuple(frame['b'])),),
    )


def register():
    """Register the toy chart and exhibit, and declare their leaves."""
    register_chart(CHART_NAME, chart_toy, primary=Toy)
    register_simple_exhibit(
        EXHIBIT_NAME, 'Toy table', 'toy_df', [Toy],
        caption='Two rows of nothing, so the passthrough path has something to serve.')
    return [
        PluginLeaf(name=EXHIBIT_NAME, kind='exhibit', label='Toy table',
                   hint='the toy frame, passed through',
                   why='needs a Toy; nothing else carries toy_df'),
        PluginLeaf(name=CHART_NAME, kind='chart', label='Toy chart',
                   hint='b against a',
                   why='needs a Toy; nothing else carries toy_df'),
    ]
