"""Global random number generator used by all stochastic paths in ``aggregate``.

Named ``random_agg`` rather than ``random`` to avoid shadowing the stdlib
``random`` module. ``RANDOM`` is the module-level ``numpy.random.Generator``;
``set_seed`` replaces it with a freshly seeded one for reproducibility.
"""

import numpy as np

__all__ = ['RANDOM', 'set_seed']


# global default RNG for all random numbers
RANDOM = np.random.default_rng(None)


def set_seed(seed):
    """Fix the seed for the global random number generator.

    Replaces :data:`RANDOM` with ``numpy.random.default_rng(seed)``. All
    subsequent calls to stochastic paths in ``aggregate`` (Iman-Conover,
    rearrangement algorithm, etc.) become reproducible.

    Parameters
    ----------
    seed : int or None
        Seed for ``numpy.random.default_rng``. ``None`` yields fresh
        entropy (non-reproducible).
    """
    global RANDOM
    RANDOM = np.random.default_rng(seed)
