"""Layer 0 of the plotting subsystem: canvas, style, and shared constants.

"Make the space." Generic figure/axes creation, the house ``mplstyle``, and
the shared figure-size / tick / axis constants. No domain knowledge: nothing
here knows what an :class:`Aggregate` or a :class:`Distortion` is -- it only
makes blank canvases and applies the look.

This module is also the **single matplotlib entry point** for the whole
library. Everything that touches matplotlib lives under :mod:`aggregate.plots`;
importing :mod:`aggregate` itself never loads matplotlib until the first plot.

The house style (``use`` / ``context`` / ``rc_params``) was absorbed here from
the former top-level ``aggregate.style`` module; that module is now a thin
shim re-exporting these three names for backward compatibility.

Notes
-----
The bundled ``.mplstyle`` is located via :mod:`importlib.resources` so it works
for editable installs, wheel installs and zip-imported installs alike. The file
is parsed exactly once (at module import) and stored as :data:`_STYLE_PARAMS`.
"""

from contextlib import contextmanager
from importlib.resources import files, as_file

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from ..constants import (FIG_W, FIG_H, FONT_SIZE, LEGEND_FONT,
                         PLOT_FACE_COLOR, FIGURE_BG_COLOR)

__all__ = [
    'plt', 'mpl', 'ticker',
    'FIG_W', 'FIG_H', 'FONT_SIZE', 'LEGEND_FONT',
    'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR',
    'use', 'context', 'rc_params',
    'make_mosaic', 'make_grid',
]

# ---------------------------------------------------------------------------
# House style (absorbed from the former top-level ``aggregate.style`` module)
# ---------------------------------------------------------------------------

_RESOURCE = files("aggregate").joinpath("data/aggregate.mplstyle")


def _load_rc_params() -> dict:
    """Parse the bundled ``.mplstyle`` once into an ``rcParams`` dict.

    Returns
    -------
    dict
        Mapping of ``rcParams`` key to value, suitable for
        ``mpl.rcParams.update`` or ``mpl.rc_context``.
    """
    with as_file(_RESOURCE) as p:
        return mpl.rc_params_from_file(
            str(p),
            fail_on_error=True,
            use_default_template=False,
        )


_STYLE_PARAMS = _load_rc_params()

# Pandas display options that travel with the style (not expressible in .mplstyle).
_PANDAS_OPTIONS = {
    "display.width": 120,
}


def use(pandas: bool = True) -> None:
    """Apply aggregate's house style globally.

    Intended for notebook, docs and interactive use. Mutates global state
    (``matplotlib.rcParams`` and optionally ``pandas.options``).

    Parameters
    ----------
    pandas : bool, default True
        Also set pandas display options (``display.width``). Pass ``False`` to
        leave pandas configuration untouched -- useful when the caller already
        manages pandas display.
    """
    mpl.rcParams.update(_STYLE_PARAMS)
    if pandas:
        for key, value in _PANDAS_OPTIONS.items():
            pd.set_option(key, value)


@contextmanager
def context(**overrides):
    """Scoped style for server-side / library code.

    Restores prior ``rcParams`` on exit. Does not touch ``pandas.options`` --
    server code should not mutate global pandas state.

    Parameters
    ----------
    **overrides
        Optional ``rcParams`` overrides layered on top of the base style. Used
        by the apiweb to bump ``figure.figsize`` up and ``figure.dpi`` down for
        screen rendering without forking the ``.mplstyle`` file.

    Yields
    ------
    None

    Examples
    --------
    >>> import aggregate.style
    >>> with aggregate.style.context():
    ...     fig, ax = plt.subplots()      # doctest: +SKIP

    >>> overrides = {"figure.figsize": (5.5, 3.5), "figure.dpi": 100}
    >>> with aggregate.style.context(**overrides):
    ...     fig, ax = plt.subplots()      # doctest: +SKIP
    """
    params = {**_STYLE_PARAMS, **overrides}
    with mpl.rc_context(params):
        yield


def rc_params() -> dict:
    """Return a copy of the bundled style as an ``rcParams`` dict.

    Useful for downstream tooling that wants to compose with other styles or
    inspect the values programmatically.

    Returns
    -------
    dict
        Fresh copy of the parsed ``.mplstyle`` -- mutating the returned dict
        does not affect future :func:`use` / :func:`context` calls.
    """
    return dict(_STYLE_PARAMS)


# ---------------------------------------------------------------------------
# Generic canvas creators ("make the space")
# ---------------------------------------------------------------------------


def make_mosaic(layout, *, figsize=None, **kwargs):
    """Create a constrained-layout figure and a mosaic ``Axes`` dict.

    The Layer-0 canvas helper for compositors whose panels are addressed by
    label (``axd['A']``). Thin wrapper over :func:`matplotlib.pyplot.subplot_mosaic`
    that pins the house ``layout='constrained'`` default.

    Parameters
    ----------
    layout : str or list
        Mosaic specification, e.g. ``'ABC'`` or ``'AB\\nCD'``.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to the house grid size
        ``(ncols * FIG_W, nrows * FIG_H)``, where the grid shape is read from
        ``layout`` (newline-separated rows for a string; nested rows for a
        list) -- one panel-sized cell per mosaic position.
    **kwargs
        Forwarded to :func:`matplotlib.pyplot.subplot_mosaic`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axd : dict of str to matplotlib.axes.Axes
    """
    kwargs.setdefault('layout', 'constrained')
    if figsize is None:
        if isinstance(layout, str):
            rows = [r for r in layout.split('\n') if r.strip() != ''] or ['']
            nrows = len(rows)
            ncols = max(len(r) for r in rows)
        else:
            nrows = len(layout)
            ncols = max((len(r) for r in layout), default=1)
        figsize = (ncols * FIG_W, nrows * FIG_H)
    return plt.subplot_mosaic(layout, figsize=figsize, **kwargs)


def make_grid(nrows, ncols, *, figsize=None, squeeze=True, **kwargs):
    """Create a constrained-layout figure and a grid of ``Axes``.

    The Layer-0 canvas helper for compositors whose panels are a simple
    ``nrows x ncols`` grid. Thin wrapper over :func:`matplotlib.pyplot.subplots`
    that pins the house ``constrained_layout=True`` default.

    Parameters
    ----------
    nrows, ncols : int
        Grid shape.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to the house grid size
        ``(ncols * FIG_W, nrows * FIG_H)`` -- one panel-sized cell per grid
        position -- so ``make_grid(1, 3)`` is three panels wide by default.
    squeeze : bool, default True
        Passed through to :func:`matplotlib.pyplot.subplots`; pass ``False`` to
        always get a 2-D ``Axes`` array.
    **kwargs
        Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes or ndarray of Axes
    """
    if figsize is None:
        figsize = (ncols * FIG_W, nrows * FIG_H)
    return plt.subplots(nrows, ncols, figsize=figsize,
                        constrained_layout=True, squeeze=squeeze, **kwargs)
