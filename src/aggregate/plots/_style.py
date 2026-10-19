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
    'use', 'context', 'rc_params', 'fit_page',
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
# The page a document draws library figures for
# ---------------------------------------------------------------------------

#: The page set by :func:`fit_page`, ``[width, max_height]`` in inches. Either
#: may be ``None``; both ``None``, the default, leaves every figure its size.
_PAGE = [None, None]

#: The height, in inches, a fitted figure keeps however far its width shrinks,
#: unless its own height or the page's cap is lower.
_FIT_MIN_HEIGHT = 2.0


def fit_page(width=None, max_height=None):
    """Fit every library figure to a page ``width`` inches wide.

    A document calls this once in its setup, with the width of its text block,
    and every figure aggregate draws afterwards (``Aggregate.plot()``,
    ``Portfolio.plot()``, the canvases :func:`make_grid` and :func:`make_mosaic`
    make, and the rest) is drawn at a size that fits. ``fit_page()`` with no
    arguments clears the page and restores every figure's own size.

    Parameters
    ----------
    width : float, optional
        Page width in inches. A figure wider than this is shrunk to it.
    max_height : float, optional
        Height cap in inches. No figure is drawn taller.

    Raises
    ------
    ValueError
        If either value is given and is not a positive number.

    Notes
    -----
    A figure of width :math:`w` and height :math:`h` is drawn at

    .. math::

        s = \\min(1, W / w), \\qquad
        w' = s w, \\qquad
        h' = \\min\\bigl(\\max(s h, \\min(h, 2)), H\\bigr)

    for page width :math:`W` and cap :math:`H`. The rule only ever shrinks: a
    figure narrower than the page keeps its size. The height falls with the
    width but stops at 2in, because a row of three 3.5in panels scaled to a
    6.3in page would otherwise be 1.5in high, and the cap always wins. On a
    6.3in page a two panel ``Aggregate.plot()`` (7.0 x 2.45in) is drawn at
    6.3 x 2.2in.

    The figure is drawn at the fitted size, not drawn large and shrunk
    afterwards, so its type prints at the size the style names. matplotlib has
    no rcParam for this: ``figure.figsize`` is the size of one figure, which a
    document sets to its own single figure width, while a library figure of two
    or three panels needs the whole width it may take.

    Examples
    --------
    >>> from aggregate.plots import fit_page
    >>> fit_page(6.3, 8.0)     # a 6.3in text block            # doctest: +SKIP
    >>> fit_page()             # back to each figure's own size # doctest: +SKIP
    """
    for name, value in (('width', width), ('max_height', max_height)):
        if value is not None and not value > 0:
            raise ValueError(f'fit_page: {name} must be a positive number of '
                             f'inches or None, not {value!r}')
    _PAGE[:] = [width, max_height]


def _fit_to_page(figsize):
    """The size to draw a figure of ``figsize`` at, given the :func:`fit_page` page.

    Parameters
    ----------
    figsize : tuple of float
        The figure's own ``(width, height)`` in inches.

    Returns
    -------
    tuple of float
        ``figsize`` itself when no page is set, else the fitted size. The rule
        is in :func:`fit_page`.
    """
    width, max_height = _PAGE
    if width is None and max_height is None:
        return figsize
    w, h = figsize
    s = 1.0 if width is None else min(1.0, width / w)
    h_fit = max(h * s, min(h, _FIT_MIN_HEIGHT))
    if max_height is not None:
        h_fit = min(h_fit, max_height)
    return (w * s, h_fit)


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
        list) -- one panel-sized cell per mosaic position. Fitted to the
        page when :func:`fit_page` has set one, whether given or defaulted.
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
    return plt.subplot_mosaic(layout, figsize=_fit_to_page(figsize), **kwargs)


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
        Fitted to the page when :func:`fit_page` has set one, whether given or
        defaulted.
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
    return plt.subplots(nrows, ncols, figsize=_fit_to_page(figsize),
                        constrained_layout=True, squeeze=squeeze, **kwargs)
