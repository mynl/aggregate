"""``aggregate.charts``: chart semantics as data, the chart IR.

The chart-side sibling of the exhibits module: where a table's meaning is a
``TableDoc``, a chart's meaning is a :class:`~aggregate.charts.ir.ChartDoc`.
Emitters here read public frames and ``GridDistribution`` accessors and
return documents; renderers (``aggregate.plots._chartdoc`` for matplotlib,
the app's generic ECharts adapter) realize them. This package never imports
matplotlib; ``tests/test_plots_boundary.py`` enforces the boundary. plots
may import charts, never the reverse.

Users write ``from aggregate import charts``; nothing is star exported from
the package root.

Emitters register in :data:`CHARTS` as ``functools.singledispatch``
generics, one per chart name, with an optional availability predicate;
:func:`available_charts` derives capability from the dispatch registries
plus the predicates, so it cannot go stale.
"""

from functools import singledispatch

from .ir import (
    CHART_IR_VERSION, ChartAxis, ChartCapabilityError, ChartDoc,
    ChartSeries, Mark, Panel, SurfaceData,
    canonical_dict, canonical_json, doc_hash, stamp,
)

__all__ = [
    'CHART_IR_VERSION', 'CHARTS', 'ChartAxis', 'ChartCapabilityError',
    'ChartDoc', 'ChartSeries', 'Mark', 'Panel', 'SurfaceData',
    'available_charts', 'canonical_dict', 'canonical_json', 'doc_hash',
    'register_chart', 'stamp',
]

#: The chart registry: ``name -> (emitter, predicate)``. The emitter is a
#: ``singledispatch`` generic ``f(obj, **semantic_options) -> ChartDoc``
#: whose base raises ``NotImplementedError`` naming the type; the predicate
#: (or None for always) is a per-object gate over and above type dispatch
#: (a reinsurance chart only where reinsurance is present). Registration
#: is open: emitter modules in this package populate it at import, and app
#: or user code may add entries.
CHARTS = {}


def register_chart(name, emitter, predicate=None):
    """Register a chart emitter under ``name``.

    Parameters
    ----------
    name : str
        The chart's registry name, unique ('joint_surface').
    emitter : callable
        A ``functools.singledispatch`` generic ``f(obj, **options) ->
        ChartDoc`` whose base implementation raises ``NotImplementedError``.
    predicate : callable, optional
        ``predicate(obj) -> bool``, an availability gate beyond type
        dispatch. ``None`` means available wherever the type dispatches.
    """
    if name in CHARTS:
        raise ValueError(f'chart {name!r} is already registered')
    CHARTS[name] = (emitter, predicate)


def _dispatches(emitter, obj):
    """True when ``emitter`` has a non-base registration for ``obj``."""
    return emitter.dispatch(type(obj)) is not emitter.registry[object]


def available_charts(obj):
    """Return the chart names available for ``obj``.

    Derived from the singledispatch registries (an MRO hit counts) plus
    each chart's predicate, mirroring the exhibits capability mechanics.

    Parameters
    ----------
    obj : object
        Any object; an unknown type simply yields an empty list.

    Returns
    -------
    list of str
        Registry names, in registration order.
    """
    out = []
    for name, (emitter, predicate) in CHARTS.items():
        if not _dispatches(emitter, obj):
            continue
        if predicate is not None and not predicate(obj):
            continue
        out.append(name)
    return out


def _emitter_base(name):
    """Create the base singledispatch generic for chart ``name``."""
    @singledispatch
    def emit(obj, **options):
        raise NotImplementedError(
            f'chart {name!r} is not implemented for '
            f'{type(obj).__name__}')
    emit.__name__ = f'chart_{name}'
    return emit
