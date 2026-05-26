"""House plotting style for ``aggregate``.

Single source of truth for plot styling shared by the docs build, the
``apiweb`` server, and user notebooks. The style itself lives in
``data/aggregate.mplstyle``; this module loads it once at import and
exposes three entry points:

- :func:`use` -- mutate global ``rcParams`` (and optionally pandas options).
  Intended for notebooks and the Sphinx ``conf.py``.
- :func:`context` -- a context manager that applies the style for the
  duration of a ``with`` block, restoring prior ``rcParams`` on exit.
  Intended for server code that must not leak global state across
  requests. Accepts ad-hoc overrides for screen vs. print rendering.
- :func:`rc_params` -- return the parsed style as a plain dict for
  inspection or composition with other styles.

Notes
-----
The bundled ``.mplstyle`` is located via :mod:`importlib.resources` so
it works for editable installs, wheel installs and zip-imported installs
alike. The file is parsed exactly once (at module import) and the
resulting dict is stored as :data:`_STYLE_PARAMS`.
"""

from contextlib import contextmanager
from importlib.resources import files, as_file

import matplotlib as mpl
import pandas as pd

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
        Also set pandas display options (``display.width``). Pass
        ``False`` to leave pandas configuration untouched -- useful when
        the caller already manages pandas display.
    """
    mpl.rcParams.update(_STYLE_PARAMS)
    if pandas:
        for key, value in _PANDAS_OPTIONS.items():
            pd.set_option(key, value)


@contextmanager
def context(**overrides):
    """Scoped style for server-side / library code.

    Restores prior ``rcParams`` on exit. Does not touch
    ``pandas.options`` -- server code should not mutate global pandas
    state.

    Parameters
    ----------
    **overrides
        Optional ``rcParams`` overrides layered on top of the base
        style. Used by the apiweb to bump ``figure.figsize`` up and
        ``figure.dpi`` down for screen rendering without forking the
        ``.mplstyle`` file.

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

    Useful for downstream tooling that wants to compose with other
    styles or inspect the values programmatically.

    Returns
    -------
    dict
        Fresh copy of the parsed ``.mplstyle`` -- mutating the returned
        dict does not affect future :func:`use` / :func:`context` calls.
    """
    return dict(_STYLE_PARAMS)
