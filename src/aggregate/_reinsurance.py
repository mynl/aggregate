"""Reinsurance ceder/netter construction (Aggregate-only concern).

Extracted from ``distributions.py`` / ``_aggregate.py`` (Phase 1b, shared concerns). A leaf/near-leaf: it never imports ``_aggregate``/``_portfolio`` (takes plain data / a distribution object), which is what lets Portfolio reuse it in P4.
"""

import logging
import numpy as np
from scipy import interpolate
from scipy.optimize import NoConvergence  # noqa

logger = logging.getLogger(__name__)


def _validate_reins_layers(reins_list, tol=1e-9):
    """
    Validate that a reinsurance program is entered bottom-up and non-overlapping.

    ``make_ceder_netter`` builds the ceder by walking layers in list order and
    tracking a running height; out-of-order or overlapping layers silently
    produce wrong cessions. Enforce the contract with a hard error here, at the
    single choke point used by every reinsurance path.

    Parameters
    ----------
    reins_list : list of (share, limit, attach)
        Layer tuples; ``limit`` may be ``np.inf`` for an unlimited top layer.
    tol : float
        Tolerance for the non-overlap comparison (absorbs float noise).

    Raises
    ------
    ValueError
        If attachments are not non-decreasing, or two layers overlap.

    Notes
    -----
    Rules over ``[(share, limit, attach), ...]``:

    - attachments must be **non-decreasing**;
    - layers must **not overlap**: ``attach_{i+1} >= attach_i + limit_i - tol``.

    Gaps between layers are allowed -- represent a gap with a zero-share layer
    ``0 po L xs A``. Zero-share entries still must respect ordering. Single-layer
    programs are trivially valid.
    """
    prev_attach = None
    prev_top = None
    for (share, limit, attach) in reins_list:
        if prev_attach is not None:
            if attach < prev_attach - tol:
                raise ValueError(
                    'Reinsurance layers must be entered bottom-up (lowest '
                    f'attachment first); got attachment {attach} after '
                    f'{prev_attach}. Enter layers in ascending order and use '
                    '"0 po L xs A" to represent a gap.')
            if attach < prev_top - tol:
                raise ValueError(
                    f'Reinsurance layers overlap: layer attaching at {attach} '
                    f'starts below the top ({prev_top}) of the preceding layer. '
                    'Layers must be non-overlapping; use "0 po L xs A" to '
                    'represent a gap.')
        prev_attach = attach
        prev_top = attach + (0 if np.isinf(limit) else limit)


def make_ceder_netter(reins_list, debug=False):
    """
    Build the netter and ceder functions. It is applied to occ_reins and agg_reins,
    so should be stand-alone.

    The reinsurance functions are piecewise linear functions from 0 to inf with
    kinks as needed to express the ceded loss as a function of subject (gross) loss.

    The entries in ``reins_list`` are tuples (share of, limit, attach) where share of is the
    percentage share, between 0 and 1.

    For example, if ``reins_list = [(1, 10, 0), (0.5, 30, 20)]`` the program is 10 x 10 and
    15 part of 30 x 20 (share=0.5). This requires nodes at 0, 10, 20, 50, and inf.

    It is easiest to make the ceder function. Ceded loss at subject loss at x equals
    the sum of the limits below x plus the cession to the layer in which x lies. The
    variable ``base`` keeps track of the layer, ``h`` of the sum (height) of lower layers.
    ``xs`` tracks the knot points, ``ys`` the values.

    ::

         Break (xs)   Ceded (ys)
              0            0
             10            0
             20           10
             50           25
            inf           25


    For example:
    ::

        %%sf 1 2

        c, n, x, y = make_ceder_netter([(1, 10, 10), (0.5, 30, 20), (.25, np.inf, 50)], debug=True)

        xs = np.linspace(0,250, 251)
        ys = c(xs)

        ax0.plot(xs, ys)
        ax0.plot(xs, xs, ':C7')
        ax0.set(title='ceded')

        ax1.plot(xs, xs-ys)
        ax1.plot(xs, xs, 'C7:')
        ax1.set(title='net')

    :param reins_list: a list of (share of, limit, attach), e.g., (0.5, 3, 2) means 50% share of 3x2
        or, equivalently, 1.5 part of 3 x 2. It is better to store share rather than part
        because it still works if limit == inf.
    :param debug: if True, return layer function xs and ys in addition to the interpolation functions.
    :return: netter and ceder functions; optionally debug information.
    """
    # hard error on out-of-order / overlapping layers (single choke point)
    _validate_reins_layers(reins_list)
    # poor mans inf
    INF = 1e99
    h = 0
    base = 0
    xs = [0]
    ys = [0]
    for (share, y, a) in reins_list:
        # part of = share of times limit
        if np.isinf(y):
            y = INF
        p = share * y
        if a > base:
            # moved to new layer, write out left-hand knot point
            xs.append(a)
            ys.append(h)
        # increment height
        h += p
        # write out right-hand knot points
        xs.append(a + y)
        ys.append(h)
        # update left-hand end
        base += (a + y)
    # if not at infinity, stay flat from base to end
    if base < INF:
        xs.append(np.inf)
        ys.append(h)
    ceder = interpolate.interp1d(xs, ys)
    netter = lambda x: x - ceder(x)
    if debug:
        return ceder, netter, xs, ys
    else:
        return ceder, netter
