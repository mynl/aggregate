"""Backward-compatibility shim for the house plotting style.

The implementation moved into :mod:`aggregate.plots._style` (Layer 0 of the
plotting subsystem) so that every matplotlib import lives under
:mod:`aggregate.plots`. This module re-exports the three public entry points so
existing callers -- ``import aggregate.style; aggregate.style.use()`` in the
docs build and the ``apiweb`` server -- keep working unchanged.

- :func:`use` -- mutate global ``rcParams`` (and optionally pandas options).
- :func:`context` -- a context manager applying the style for a ``with`` block.
- :func:`rc_params` -- return the parsed style as a plain dict.

Importing this module loads matplotlib (it pulls in the plotting subsystem);
importing :mod:`aggregate` itself does not.
"""

from .plots._style import use, context, rc_params

__all__ = ['use', 'context', 'rc_params']
