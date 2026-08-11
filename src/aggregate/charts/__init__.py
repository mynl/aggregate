"""``aggregate.charts``: chart semantics as data, the chart IR.

.. warning::

   **Provisional module, in the sense of PEP 411.** ``aggregate.charts`` is
   additive to the 1.0 release and is **not part of the 1.0 API contract**. Its
   API may change in a minor release with no deprecation period, unlike the
   stable core (:class:`~aggregate.Aggregate`, :class:`~aggregate.Portfolio`,
   :class:`~aggregate.PnL`, :class:`~aggregate.Severity`,
   :class:`~aggregate.Distortion`, :class:`~aggregate.Underwriter`,
   :func:`~aggregate.build` and the DecL grammar), which does carry the usual
   promise. The module is deliberately public rather than underscore prefixed:
   use it, and report what does not fit. That feedback is how it graduates to
   stable in a later minor release. See :doc:`/3_reference/3_x_API_Stability`.

The chart-side sibling of the exhibits module: where a table's meaning is a
``TableDoc``, a chart's meaning is a :class:`~aggregate.charts.ir.ChartDoc`.
Emitters here read public frames and ``GridDistribution`` accessors and
return documents; renderers (``aggregate.plots._chartdoc`` for matplotlib,
the app's generic ECharts adapter) realize them. This package never imports
matplotlib; ``tests/test_plots_boundary.py`` enforces the boundary. plots
may import charts, never the reverse.

Users write ``from aggregate import charts``; nothing is star exported from
the package root.

**Dependencies point inward.** This package imports from the core; the core's
pre-existing classes and modules do not import it, so nothing here can
destabilize or delay 1.0. There is exactly one edge into an existing package:
:mod:`aggregate.plots` gained :func:`~aggregate.plots.plot_chartdoc`, the
renderer, which imports :mod:`aggregate.charts.ir`. That is a new public
function in an old package, and nothing else in ``plots`` depends on it. The
one other class of change to existing code is the per-chart conversion of a
bespoke plot to an emitter plus renderer, gated by before-and-after image
diffs (``tests/data/chartdoc_baselines/``) and deferrable chart by chart past
1.0.

Work continues as attention allows and 1.0 ships whether or not it is
finished. Explicitly post-1.0: conversion of the charts the app does not use,
full convergence on matplotlib-renders-the-IR, and any chart meta-language.

Emitters register in :data:`CHARTS` as ``functools.singledispatch``
generics, one per chart name, with an optional availability predicate;
:func:`available_charts` derives capability from the dispatch registries
plus the predicates, so it cannot go stale.
"""

from collections import namedtuple
from functools import singledispatch

from .ir import (
    CHART_IR_VERSION, SUPPORT_KINDS, ChartAxis, ChartCapabilityError,
    ChartDoc, ChartSeries, Mark, Panel, SurfaceData,
    canonical_dict, canonical_json, complete_tex, doc_hash, human_strings,
    load_chart_doc, stamp,
)

__all__ = [
    'CHART_IR_VERSION', 'CHARTS', 'SUPPORT_KINDS', 'ChartAxis',
    'ChartCapabilityError', 'ChartDoc', 'ChartEntry', 'ChartSeries', 'Mark',
    'Panel', 'SurfaceData', 'available_charts', 'build_chart_doc',
    'canonical_dict', 'canonical_json', 'complete_tex', 'doc_hash',
    'human_strings', 'load_chart_doc', 'primary_chart', 'register_chart',
    'stamp',
]

#: One registry entry: the emitter, its availability predicate, and the
#: classes this chart is the *primary* picture of. A namedtuple rather than
#: a bare tuple so a reader of ``CHARTS`` sees what each slot is.
ChartEntry = namedtuple('ChartEntry', 'emitter predicate primary')

#: The chart registry: ``name -> ChartEntry``. The emitter is a
#: ``singledispatch`` generic ``f(obj, **semantic_options) -> ChartDoc``
#: whose base raises ``NotImplementedError`` naming the type; the predicate
#: (or None for always) is a per-object gate over and above type dispatch
#: (a reinsurance chart only where reinsurance is present). Registration
#: is open: emitter modules in this package populate it at import, and app
#: or user code may add entries.
CHARTS = {}


def register_chart(name, emitter, predicate=None, primary=None):
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
    primary : type or tuple of type, optional
        The classes this chart is the *primary* picture of, meaning the one
        a caller with no other information should draw (see
        :func:`primary_chart`). Most charts are primary for nothing:
        reinsurance is a view of a book and severity is a component of an
        aggregate, and neither is the object's own picture.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    if name in CHARTS:
        raise ValueError(f'chart {name!r} is already registered')
    if primary is not None and not isinstance(primary, tuple):
        primary = (primary,)
    CHARTS[name] = ChartEntry(emitter, predicate, primary)


def _dispatches(emitter, obj):
    """True when ``emitter`` has a non-base registration for ``obj``."""
    return emitter.dispatch(type(obj)) is not emitter.registry[object]


def _serves(entry, obj):
    """True when this entry can emit a document for ``obj``."""
    if not _dispatches(entry.emitter, obj):
        return False
    return entry.predicate is None or bool(entry.predicate(obj))


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

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    return [name for name, entry in CHARTS.items() if _serves(entry, obj)]


def primary_chart(obj):
    """Return the name of the chart that is ``obj``'s own picture, or None.

    :func:`available_charts` answers what *can* be drawn, which for an
    aggregate carrying reinsurance is three things. This answers which one
    to draw when nothing else has been asked for, which is the question a
    landing page has and cannot otherwise put to the library: an aggregate's
    own picture is its aggregate chart, its severity is a component of it,
    and its reinsurance is a view of it.

    Parameters
    ----------
    obj : object

    Returns
    -------
    str or None
        None when no registered chart claims ``obj``, including when the
        chart that would claim it is unavailable (an object that has not
        been updated has no picture yet).

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    for name, entry in CHARTS.items():
        if entry.primary and isinstance(obj, entry.primary) \
                and _serves(entry, obj):
            return name
    return None


def build_chart_doc(obj, name, **options):
    """Build the chart document ``name`` for ``obj``, stamped with its hash.

    The sibling of ``exhibits.build_exhibit``, and the one entry point a
    consumer needs: it resolves the registry entry, checks availability,
    dispatches on the type, and stamps the content hash, so an emitter is
    left with nothing to do but the semantics. A module function rather
    than a method on each class, exactly as on the exhibits side, which
    keeps it reachable for a class that owns no chart of its own and adds
    nothing to any instance namespace.

    Parameters
    ----------
    obj : object
        A first-class object (``Aggregate``, ``Portfolio``, ``Severity``,
        ``Distortion``, ...).
    name : str
        Chart registry name.
    **options
        Semantic options for the emitter (``basis`` for reinsurance,
        ``xmax`` for an aggregate). Never renderer options: how a document
        is drawn is the renderer's argument list, not this one.

    Returns
    -------
    ChartDoc
        Stamped: ``hash`` set, ``generator`` naming the producing build.

    Raises
    ------
    KeyError
        For an unknown chart name, listing the registry.
    NotImplementedError
        When no emitter is registered for this type.
    ValueError
        When the chart exists for the type but not for this object (a
        reinsurance chart on a book that cedes nothing).

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    try:
        entry = CHARTS[name]
    except KeyError:
        raise KeyError(f'unknown chart {name!r}; known charts: '
                       + ', '.join(CHARTS)) from None
    if not _dispatches(entry.emitter, obj):
        entry.emitter(obj)      # raises NotImplementedError naming the type
    if entry.predicate is not None and not entry.predicate(obj):
        raise ValueError(
            f'chart {name!r} is not available for {type(obj).__name__} '
            f'{getattr(obj, "name", "")!r}: its availability predicate '
            'failed (missing update() grid, or required structure such as '
            'reinsurance)')
    from .. import __version__
    return stamp(entry.emitter(obj, **options),
                 generator=f'aggregate {__version__}')


def _emitter_base(name):
    """Create the base singledispatch generic for chart ``name``."""
    @singledispatch
    def emit(obj, **options):
        raise NotImplementedError(
            f'chart {name!r} is not implemented for '
            f'{type(obj).__name__}')
    emit.__name__ = f'chart_{name}'
    return emit


# Emitter modules populate the registry at import (each guards its own
# imports; none may touch matplotlib).
from ._emit_aggregate import chart_agg  # noqa: E402
from ._emit_bivariate import chart_joint_surface  # noqa: E402
from ._emit_bounds import chart_envelope  # noqa: E402
from ._emit_distortion import chart_distortion  # noqa: E402
from ._emit_pnl import chart_pnl  # noqa: E402
from ._emit_portfolio import chart_port  # noqa: E402
from ._emit_reins import chart_reins  # noqa: E402
from ._emit_severity import chart_severity  # noqa: E402

__all__ += ['chart_agg', 'chart_distortion', 'chart_envelope',
            'chart_joint_surface', 'chart_pnl', 'chart_reins',
            'chart_port', 'chart_severity']
