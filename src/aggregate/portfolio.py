"""Public facade for the ``Portfolio`` subsystem (Plan P4, three-subsystem split).

This module defines no logic of its own. ``Portfolio`` is one public class, but
its body splits along **how the joint loss distribution is built** into three
underscore-prefixed implementation modules behind this facade (the underscore
answers "import from here?" with "no -- import from the package"):

- :mod:`aggregate._portfolio_density` -- the density-based (independence)
  path: the independent-sum FFT combine and the ``add_exa`` / ``exeqa_*``
  kernels.
- :mod:`aggregate._portfolio_sample` -- the sample-based (dependence) path:
  the switcheroo (``swap_density_df``), comonotonic allocations, Iman--Conover.
- :mod:`aggregate._portfolio_common` -- the common exeqa numerics shared by
  both construction paths (augmented df, apply distortion, allocation).

The ``Portfolio`` class itself lives in :mod:`aggregate._portfolio` and consumes
the P3 shared concerns (``_validation`` / ``_bucket_window`` / ``_pricing``).
Every historical import path (``aggregate.Portfolio``,
``aggregate.portfolio.Portfolio``, ``from aggregate.portfolio import <name>``)
keeps working through the re-exports below.
"""
from ._portfolio import *           # noqa: F401,F403  (the Portfolio class + make_awkward)
from ._portfolio_common import *    # noqa: F401,F403
from ._portfolio_sample import *    # noqa: F401,F403

# Names accessed by qualified path / attribute (not part of the public ``*``
# surface). Explicit so the facade exposes the historical attribute set.
from ._portfolio import (  # noqa: F401
    VALIDATION_NOISE, ALIASING_EPS, EXEQA_NOISE_FLOOR,
)
from ._portfolio_common import check01, make_array, convex_points  # noqa: F401
from ._portfolio_sample import make_comonotonic_allocations_work  # noqa: F401

# The public ``*`` surface, unchanged from the pre-split module.
__all__ = ['Portfolio', 'make_awkward', 'make_comonotonic_allocations',
           'swap_density_df']
