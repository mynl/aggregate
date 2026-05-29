# coding: utf-8 -*-
"""Top-level package for ``aggregate``.

Each submodule declares its own public surface in ``__all__``; this file
just re-exports those via ``from .module import *``. To change what's
public at the top level, edit ``__all__`` in the source module -- not
this file.

``Tweedie``, ``FourierTools``, ``Pentagon``, and anything in
``pedagogy`` are intentionally NOT re-exported here -- reach them via
submodule import (``from aggregate.tweedie import Tweedie``, etc.).
"""

# Pandas Copy-on-Write: enabled unconditionally for the library. On
# pandas >= 3.0 CoW is the unchangeable default and the option is a
# deprecated no-op (setting it emits a Pandas4Warning), so we only flip
# the switch on the 2.x line where it is a meaningful opt-in.
import pandas as _pd
if int(_pd.__version__.split(".", 1)[0]) < 3:
    _pd.options.mode.copy_on_write = True
del _pd

from .parser        import *  # noqa: F401,F403
from .moments       import *  # noqa: F401,F403
from .iman_conover  import *  # noqa: F401,F403
from .utilities     import *  # noqa: F401,F403
from .spectral      import *  # noqa: F401,F403
from .distributions import *  # noqa: F401,F403
from .portfolio     import *  # noqa: F401,F403
from .underwriter   import *  # noqa: F401,F403
from .bounds        import *  # noqa: F401,F403
from .constants     import *  # noqa: F401,F403
from .random_agg    import *  # noqa: F401,F403
from .decl_pygments import *  # noqa: F401,F403
# tweedie depends on Aggregate / build / qd already being bound on the
# package, so import last.
from .tweedie       import *  # noqa: F401,F403


__docformat__ = 'restructuredtext'
__project__   = 'aggregate'
__author__    = "Stephen J. Mildenhall"
__copyright__ = "2018-2026, Stephen J Mildenhall"
__license__   = "BSD 3-Clause New License"
__email__     = "stephen.j.mildenhall@gmail.com"
__status__    = "beta"
from importlib.metadata import version as _pkg_version
__version__ = _pkg_version("aggregate")


__doc__ = """
:mod:`aggregate` solves insurance, risk management, and actuarial problems using realistic models that
reflect underlying frequency and severity. It makes working with an aggregate (compound) probability
distribution as easy as the lognormal, delivering the speed and accuracy of parametric distributions
to situations that usually require simulation. :mod:`aggregate` includes an expressive language called
DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.
"""
