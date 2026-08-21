"""Public facade for the distribution classes (Phase 1 kind split).

This module defines no logic of its own. The implementation lives in the
underscore-prefixed kind modules; the underscore answers "import from here?"
with "no -- import from the package". Every historical import path
(``aggregate.Aggregate``, ``aggregate.distributions.Aggregate``,
``from aggregate.distributions import <name>``) keeps working through the
re-exports below.

Internal import order (the only place the intra-``distributions`` ordering
lives): ``_fits`` -> ``_severity`` -> ``_frequency`` -> ``_aggregate``.
"""
from ._fits import *          # noqa: F401,F403
from ._severity import *      # noqa: F401,F403
from ._frequency import *     # noqa: F401,F403
from ._aggregate import *     # noqa: F401,F403

# Names imported by qualified path / attribute by other modules and tests but
# not part of the public ``*`` surface (constants, private helpers). Explicit
# so the facade exposes exactly the historical attribute set.
from ._fits import _approximate_sev_kwargs  # noqa: F401
from ._severity import (  # noqa: F401
    _DiscreteRV, _scalar_bound, validate_discrete_distribution,
)
from ._bucket_window import (  # noqa: F401
    WINDOW_NINES, WINDOW_LOG2_GROWTH, WINDOW_NINES_TRIM, WINDOW_PAD_SKEW,
    WINDOW_SLACK_THICK, BUCKET_SIZING_P, SBJ_TAIL_FLOOR,
    estimate_agg_window, bs_describe, bs_explain,
)
from ._validation import (  # noqa: F401
    VALIDATION_NOISE, ALIASING_EPS, convolution_residual, explain_validation,
)
from ._reinsurance import make_ceder_netter, _validate_reins_layers  # noqa: F401
from ._aggregate import (  # noqa: F401
    value_type_role, value_type_label, max_log2, _flat_col_to_stats_index,
)
from ._pnl import PnL  # noqa: F401
# Back-compat re-export; tail.py is the source of truth (see tests/test_tail.py).
from .tail import TailClass  # noqa: F401

# The public ``*`` surface, unchanged from the pre-split module.
__all__ = [
    'Frequency', 'Severity', 'Aggregate', 'PnL',
    'lognorm_fit', 'sln_fit', 'sgamma_fit', 'gamma_fit', 'beta_fit',
    'invgamma_fit', 'invgauss_fit',
    'lognorm_lev', 'lognorm_approx',
    'approximate_from_mcvsk',
]
