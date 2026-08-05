"""``aggregate.exhibits`` -- business exhibits over greater_tables IR.

The library owns meaning, the app owns arrangement. An *exhibit* is a small
list of presentation ready tables (greater_tables ``TableDoc`` IR blocks)
derived from the first class citizen frames (``summary_df``, ``tail_df``,
``economic_df``, ...), carrying the business knowledge that would otherwise
leak into a client: captions, row emphasis, raw moment drops, relabeling. The
test for placement: if deleting the web app would destroy knowledge an actuary
would want in a notebook, that knowledge belongs here.

Purely additive: exhibits import from the core, the core never imports
exhibits. Deliberately NOT star imported in ``aggregate/__init__.py`` (the
``Tweedie`` / ``Pentagon`` precedent). Reach it with ``from aggregate import
exhibits``, then ``exhibits.summary(obj, 'insurer')``.

Layout, mirroring :mod:`aggregate.plots` (``dev/plan-exhibits.md``):

- :mod:`._core` -- the machinery, class agnostic: :class:`Perspective`,
  :class:`Exhibit`, the singledispatch registry, the frame and IR stages, and
  the translation helpers more than one class shares.
- ``_aggregate`` / ``_portfolio`` / ``_pnl`` / ``_bivariate`` / ``_distortion``
  -- one module per class, holding **that class's business translation**. This
  is where you go to edit how an exhibit reads.
- The **manifest** at the foot of this file -- every plain passthrough
  exhibit, one line each, via :func:`register_simple_exhibit`.

Adding an exhibit is one of three sizes. A passthrough over an existing frame
is a manifest line. A new insurer translation of an existing exhibit is a
function in the class module. A wholly new multi block exhibit declares its
generic function in ``_core`` and registers builders in the class module.

Nothing here imports greater_tables. The IR conversion step
(:func:`build_exhibit`, :meth:`Exhibit.to_payload`) imports it lazily and
names the ``exhibits`` extra when it is missing, so ``import aggregate`` and
``import aggregate.exhibits`` both stay free of it.
"""

from .._aggregate import Aggregate
from .._pnl import PnL
from .._portfolio import Portfolio
from ..bivariate import BivariateAggregate
from ..spectral import Distortion

from ._core import (
    CAPITAL_ANCHOR_PERIODS, EXHIBITS, Exhibit, Perspective,
    RAW_MOMENT_MEASURES,
    available_exhibits, build_exhibit, exhibit_frames, register_simple_exhibit,
    dependency, economic, economic_ratios, economic_waterfall, reins, stats,
    summary, tail, validation,
    _perspectives_updated,
)

# Per-class business translation. Imported for their registration side
# effects, which is the whole point: importing the package wires the registry.
from . import _aggregate, _portfolio, _pnl, _bivariate, _distortion  # noqa: F401

__all__ = [
    'Perspective', 'Exhibit', 'EXHIBITS',
    'available_exhibits', 'exhibit_frames', 'build_exhibit',
    'register_simple_exhibit',
    'CAPITAL_ANCHOR_PERIODS', 'RAW_MOMENT_MEASURES',
    'summary', 'tail', 'stats', 'validation', 'reins',
    'economic', 'economic_ratios', 'economic_waterfall', 'dependency',
    'bs_window', 'tail_behavior',
]


# --- the passthrough manifest -----------------------------------------------
# Every exhibit that serves one frame with no business translation, declared
# in one line. These register no insurer override, so INSURER equals RAW by
# the default rule; adding an override in a class module changes only that
# (exhibit, type) pair. Exhibits with a translation are declared in ``_core``
# and registered in their class module instead.

_ALL_FCC = [Aggregate, Portfolio, BivariateAggregate, PnL, Distortion]

register_simple_exhibit('summary', 'Summary', 'summary_df', _ALL_FCC)
register_simple_exhibit('stats', 'Statistics', 'stats_df', _ALL_FCC)
register_simple_exhibit('validation', 'Validation', 'validation_df', _ALL_FCC)
register_simple_exhibit('tail', 'Return periods', 'tail_df',
                        [Aggregate, Portfolio])
register_simple_exhibit('economic', 'Economics', 'economic_df', [PnL])

#: Diagnostics, the app's "More" material. Both need the realized grid.
bs_window = register_simple_exhibit(
    'bs_window', 'Grid sizing', 'bs_window_df',
    [Aggregate, Portfolio, BivariateAggregate],
    predicate=_perspectives_updated)
tail_behavior = register_simple_exhibit(
    'tail_behavior', 'Tail behavior', 'tail_behavior_df',
    [Aggregate, Portfolio], predicate=_perspectives_updated)
