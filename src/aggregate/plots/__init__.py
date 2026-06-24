"""``aggregate.plots`` -- the library-wide plotting subsystem.

The single matplotlib boundary for ``aggregate``. Importing :mod:`aggregate`
never touches matplotlib; the first call to any ``.plot()`` imports this
subpackage (via the function-local imports in the class stubs), and only then
is matplotlib loaded.

Three layers (see ``dev/plan-plots-subsystem.md``):

- **Layer 0** -- :mod:`aggregate.plots._style`: "make the space". Generic
  canvas creators (:func:`make_mosaic`, :func:`make_grid`), the house style
  (:func:`use`, :func:`context`, :func:`rc_params`) and the shared figure
  constants. No domain knowledge.
- **Layer 1** -- ``_density`` / ``_distribution`` / ``_quantile`` / ...:
  content panel workers, by *what* is drawn. Each renders one content type into
  a **provided** ``Axes`` from a public ``<noun>_df`` slice.
- **Layer 2** -- ``_aggregate`` / ``_severity`` / ``_portfolio`` / ...:
  per-class compositors, by *who*. Thin: ask Layer 0 for the canvas, populate
  each ``Axes`` via Layer 1, hold the class's named variants.

The public class ``.plot()`` methods are one-line stubs delegating to the
Layer-2 compositor functions re-exported here.
"""

from ._style import (
    plt, mpl, ticker,
    FIG_W, FIG_H, FONT_SIZE, LEGEND_FONT, PLOT_FACE_COLOR, FIGURE_BG_COLOR,
    use, context, rc_params,
    make_mosaic, make_grid,
)

# Layer 2 compositors (the public entry points the class stubs delegate to).
from ._aggregate import plot_aggregate, plot_pnl, plot_reins_occ
from ._severity import plot_severity
from ._distortion import plot_distortion, plot_distortion_affine
from ._portfolio import plot_portfolio, plot_scatter, plot_sample_compare
from ._bounds import plot_bounds_envelope, plot_bounds_weights, plot_hull_bounds
from ._bivariate import plot_bivariate, plot_bivariate_distribution
from ._fourier import (plot_fourier, plot_fourier_wraps, plot_fourier_simpson,
                       plot_fourier1d)

__all__ = [
    # Layer 0
    'use', 'context', 'rc_params', 'make_mosaic', 'make_grid',
    'FIG_W', 'FIG_H', 'FONT_SIZE', 'LEGEND_FONT',
    'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR',
    # Layer 2 compositors
    'plot_aggregate', 'plot_reins_occ',
    'plot_severity',
    'plot_distortion', 'plot_distortion_affine',
    'plot_portfolio', 'plot_scatter', 'plot_sample_compare',
    'plot_bounds_envelope', 'plot_bounds_weights', 'plot_hull_bounds',
    'plot_bivariate', 'plot_bivariate_distribution',
    'plot_fourier', 'plot_fourier_wraps', 'plot_fourier_simpson',
    'plot_fourier1d',
]
