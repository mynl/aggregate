"""The Aggregate compound-distribution class and its support functions.

Extracted from ``distributions.py`` (Phase 1, kind split). Imported through the
``distributions`` facade so every existing import path keeps working.
"""

from collections import namedtuple
from collections.abc import Iterable
import hashlib
import json
import inspect
import logging
import math
import warnings
import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.fft as sfft
from scipy.integrate import quad
import scipy.stats as ss
from scipy import interpolate
from scipy.optimize import brentq, NoConvergence  # noqa
from textwrap import fill
from ._help import HelpMixin
from .constants import (DefectiveDistributionWarning,
                        INFO_NA, info_row,
                        InfiniteVarianceError,
                        ZeroModifiedExposureWarning,
                        warn_once,
                        REINS_LABEL_GROSS, REINS_LABEL_NET,
                        REINS_LABEL_CEDED, REINS_LABEL_OUTPUT)
from .config import get_settings
from .moments import (MomentAggregator, MomentWrangler,
                      xsden_to_mwrangler,
                      xsden_to_meancv, xsden_to_meancvskew,
                      _noise_aware_rel_error, _snap_noise)
from .utilities import (ft, ift,
                        round_bucket,
                        balanced_window,
                        agg_help, remove_fuzz, value_type_role)
from ._grid_distribution import GridDistribution, return_period_map, period_to_p
from ._labeled import LabeledMixin
from .decl_writer import spec_to_decl
from ._program import ProgramMixin
from . import _program
import aggregate.random_agg as ar
from .spectral import choquet_weights
from . import tail as _tail
from .tail import TailClass

from ._fits import (_approximate_sev_kwargs, _sev_kwargs_to_decl,
                    approximate_from_mcvsk)
from ._frequency import Frequency, FrequencyEmpirical, FrequencyRenewal
from ._renewal import ruin_cepstral
from ._severity import Severity
# Phase 1b shared concerns (leaf/near-leaf; never import back into _aggregate).
from ._bucket_window import (
    WINDOW_NINES, WINDOW_LOG2_GROWTH, WINDOW_NINES_TRIM, WINDOW_PAD_SKEW,
    WINDOW_SLACK_THICK, BUCKET_SIZING_P, SBJ_TAIL_FLOOR,
    _estimate_agg_percentile, estimate_agg_window, bs_describe, bs_explain,
)
from . import _bucket_window
from ._validation import (VALIDATION_NOISE, DEFICIT_MATERIALITY,
                          ALIASING_EPS, explain_validation)
from . import _validation
from . import _reinsurance
from ._aggregate_compute import discretize_severities, freq_sev_convolution
from . import _pricing
from .results import EvaluationResult, RuinResult

logger = logging.getLogger(__name__)

__all__ = [
    'Aggregate',
]

#: Default return-period ladder for the summary ``tail_df`` (overridable via
#: ``tail_periods_df(periods=...)``). The Solvency II ``1-in-200`` (99.5%) and US
#: capital-adequacy / rating ``1-in-250`` (99.6%) anchors are both included and
#: highlighted in the HTML rendering.
DEFAULT_RETURN_PERIODS = (2, 5, 10, 25, 50, 100, 200, 250, 500, 1000)

#: Key percentiles carried by the summary ``summary_df`` (low / median / high).
SUMMARY_PERCENTILES = (0.01, 0.50, 0.99)


def _summary_pct_label(p):
    """Column label for a ``summary_df`` percentile: ``P01`` / ``Median`` /
    ``P99`` (0.50 reads as ``Median``; the zero-padded ``P``-headers match the
    :attr:`PnL.summary_df` card, ``.3g`` fallback for fractional points).
    """
    if p == 0.50:
        return 'Median'
    v = p * 100
    return f'P{v:02.0f}' if float(v).is_integer() else f'P{v:.3g}'

#: Relative floor below which ``CV = SD / E[X]`` is left blank in ``summary_df``:
#: the mean is treated as indistinguishable from zero when ``|E[X]| < tol * SD``
#: (a signed / near-break-even position), where ``CV`` is meaningless. ``SD`` is
#: always reported.
CV_MEAN_REL_TOL = 1e-3


def value_type_label(is_loss_value):
    """Render the is-loss boolean role as the configured label string.

    Parameters
    ----------
    is_loss_value : bool
        ``True`` for the loss convention, ``False`` for payoff.

    Returns
    -------
    str
        ``settings.labels.loss`` or ``settings.labels.payoff``.
    """
    labels = get_settings().labels
    return labels.loss if is_loss_value else labels.payoff


def return_period_frame(q, tvar, mean, periods=None):
    """Build a symmetric return-period table from quantile and TVaR functions.

    Shared by :meth:`Aggregate.tail_periods_df`,
    :meth:`Portfolio.tail_periods_df`, :meth:`PnL.tail_periods_df` and
    :meth:`BivariateAggregate.tail_periods_df`, so every return-period table
    in the library reads one implementation.

    Parameters
    ----------
    q, tvar : callable
        ``q(p)`` (VaR) and ``tvar(p)`` (TVaR) at a non-exceedance probability.
    mean : float
        ``E[X]``, the leverage denominator and the ``xsVaR`` reference.
    periods : array_like of float, optional
        Return-period ladder. Defaults to :data:`DEFAULT_RETURN_PERIODS`.

    Returns
    -------
    pandas.DataFrame
        Indexed by non-exceedance probability ``P``; columns ``T | VaR | TVaR
        | xsVaR | VaR/Mean``; ``E[X]`` carried in ``.attrs['mean']``.

    Notes
    -----
    Every rung of the ladder contributes **both** of its probabilities, the
    lower tail ``P = 1 / T`` and the upper tail ``P = 1 - 1 / T``, so the
    index runs symmetrically from ``0.001`` to ``0.999`` on the default
    ladder and one table serves both sign conventions: a loss is read off the
    high ``P`` rows, a payoff off the low ones. The two rungs come from the
    two branches of :func:`period_to_p`, taken together rather than chosen
    between, which is why the frame needs no orientation argument.

    ``T`` is the return period **as the row is read**: ``1 / P`` below the
    median and ``1 / (1 - P)`` above it, which is the rung the row came from
    and the interpretation that matters at that probability. It is a reading
    rather than one formula in ``P``, so a 1-in-200 loss year and a 1-in-200
    shortfall year both label as ``200``, on opposite sides of the table.
    """
    periods = DEFAULT_RETURN_PERIODS if periods is None else periods
    T = np.unique(np.atleast_1d(np.asarray(periods, dtype=float)))
    # Both rungs of every period, then sort into one ascending P ladder and
    # drop the duplicate where the two meet at the median (T = 2).
    p = np.concatenate([period_to_p(T, False), period_to_p(T, True)])
    per = np.concatenate([T, T])
    order = np.argsort(p, kind='stable')
    p, per = p[order], per[order]
    keep = np.ones(p.size, dtype=bool)
    keep[1:] = p[1:] > p[:-1]
    p, per = p[keep], per[keep]
    mean = float(mean)
    var = np.array([q(float(pi)) for pi in p], dtype=float)
    tv = np.array([tvar(float(pi)) for pi in p], dtype=float)
    leverage = (var / mean if abs(mean) > VALIDATION_NOISE
                else np.full_like(var, np.nan))
    # Integer return periods read cleanly as ``200`` not ``200.0``.
    if np.all(np.mod(per, 1.0) == 0.0):
        per = per.astype(int)
    df = pd.DataFrame(
        {
            'T': per,
            'VaR': var,
            'TVaR': tv,
            'xsVaR': var - mean,
            'VaR/Mean': leverage,
        },
        index=pd.Index(p, name='P'),
    )
    df.attrs['mean'] = mean
    return df


#: Column order of ``approximation_df`` / ``approximation_density_df`` after
#: the ``exact`` anchor column: the method-of-moments families, two-moment
#: fits first.
APPROXIMATION_FAMILIES = ('norm', 'gamma', 'lognorm', 'sgamma', 'slognorm')


def _approximation_laws(m, cv, skew, signed_input):
    """One emitted law per family: closed forms with reflect and clamp applied.

    The engine behind :attr:`Aggregate.approximation_df`,
    :attr:`Aggregate.approximation_density_df` and their ``Portfolio`` twins.
    For each family in :data:`APPROXIMATION_FAMILIES` it runs the single fit
    core (:func:`~aggregate._fits._approximate_sev_kwargs`) and wraps the
    result as the law of the **emitted program**: the severity keyword mirrors
    the input (``signed_input``), so an unsigned input clamps any fitted
    sub-zero mass to an atom at 0, exactly as the built surrogate would.

    Parameters
    ----------
    m, cv, skew : float
        The subject's realized aggregate moments (the same moments
        ``approximate`` fits).
    signed_input : bool
        The subject's own signedness (``_signed()``). ``False`` applies the
        plain ``sev`` clamp at 0 to quantiles; the cdf needs no adjustment
        on an unsigned grid because ``G(x)`` for ``x >= 0`` already includes
        the clamp atom.

    Returns
    -------
    dict of str -> dict
        Per family: ``fragment`` (the DecL severity fragment), ``shape`` /
        ``loc`` / ``scale`` (0 where the family has no such parameter, so
        every parameter row is numeric; a failed fit stays ``NaN``),
        ``cdf`` / ``pdf`` / ``ppf`` (vectorized callables of the emitted
        law).

    Notes
    -----
    A reflected (left-skew) fit is the law ``Y = loc - X`` with the base
    ``X`` built at loc 0, so ``F_Y(x) = S_X(loc - x)``,
    ``f_Y(x) = f_X(loc - x)`` and ``q_Y(p) = loc - q_X(1 - p)``. The clamp
    transform is ``Z = max(0, Y)``: on quantiles ``q_Z = max(0, q_Y)``; the
    cdf is unchanged for ``x >= 0`` (the atom at 0 is the mass ``F_Y(0)``);
    the pdf ignores the atom, which is why a clamped density column sums to
    ``1 -`` the clamp mass rather than exactly 1.

    The fit call is guarded with ``np.errstate``: the frame probes **all**
    five families by design, and on a negative-mean (signed) subject the
    unshifted ``lognorm`` / ``gamma`` fits are inadmissible (``log`` of a
    negative mean). Their ``NaN`` parameters are the answer, the column's
    self-describing "no such fit", not a numerical accident to warn about.
    """
    laws = {}
    for kind in APPROXIMATION_FAMILIES:
        with np.errstate(invalid='ignore', divide='ignore'):
            sev = _approximate_sev_kwargs(m, cv, skew, kind)
        reflected = bool(sev.get('sev_reflect', False))
        name = sev['sev_name']
        shape = float(sev['sev_a']) if 'sev_a' in sev else 0.0
        loc = float(sev['sev_loc']) if 'sev_loc' in sev else 0.0
        scale = float(sev['sev_scale'])
        base_loc = 0.0 if reflected else loc
        if name == 'norm':
            base = ss.norm(loc=base_loc, scale=scale)
        elif name == 'lognorm':
            base = ss.lognorm(shape, loc=base_loc, scale=scale)
        else:
            base = ss.gamma(shape, loc=base_loc, scale=scale)
        if reflected:
            def cdf(x, b=base, L=loc):
                return np.asarray(b.sf(L - np.asarray(x, dtype=float)),
                                  dtype=float)

            def pdf(x, b=base, L=loc):
                return np.asarray(b.pdf(L - np.asarray(x, dtype=float)),
                                  dtype=float)

            def ppf(p, b=base, L=loc):
                return np.asarray(
                    L - b.ppf(1.0 - np.asarray(p, dtype=float)), dtype=float)
        else:
            def cdf(x, b=base):
                return np.asarray(b.cdf(x), dtype=float)

            def pdf(x, b=base):
                return np.asarray(b.pdf(x), dtype=float)

            def ppf(p, b=base):
                return np.asarray(b.ppf(p), dtype=float)
        if not signed_input:
            ppf = (lambda p, f=ppf:
                   np.maximum(0.0, f(p)))
        laws[kind] = dict(
            fragment=_sev_kwargs_to_decl(sev, reflected, kind, skew).strip(),
            shape=shape, loc=loc, scale=scale, cdf=cdf, pdf=pdf, ppf=ppf)
    return laws


def _approximation_ladder():
    """The symmetric non-exceedance ladder ``tail_df`` uses.

    Both ``1/T`` and ``1 - 1/T`` per rung of
    :data:`DEFAULT_RETURN_PERIODS`, deduplicated at the median, so ``P``
    runs 0.001 to 0.999 and the quantile block reads beside ``tail_df``.
    """
    T = np.asarray(DEFAULT_RETURN_PERIODS, dtype=float)
    return np.unique(np.concatenate([1.0 / T, 1.0 - 1.0 / T]))


def approximation_frame(m, cv, skew, signed_input, xs, exact_density, q_exact,
                        bs):
    """Build the ``approximation_df`` frame: all five fits against exact.

    Shared by :attr:`Aggregate.approximation_df` and
    :attr:`Portfolio.approximation_df` (the latter on the total), so both
    read one implementation, the ``return_period_frame`` arrangement.

    Parameters
    ----------
    m, cv, skew : float
        The subject's realized aggregate moments (the fit targets).
    signed_input : bool
        The subject's ``_signed()``; decides the mirrored clamp.
    xs : ndarray
        The realized output grid (the total's grid on a portfolio).
    exact_density : ndarray
        The realized aggregate density on ``xs`` (sums to ~1).
    q_exact : callable
        The subject's grid quantile function ``q(p)``.
    bs : float
        Bucket size; each family cumulative is read at the bucket's upper
        half-edge so its discretization matches the exact column's.

    Returns
    -------
    pandas.DataFrame
        Columns ``exact`` then :data:`APPROXIMATION_FAMILIES` (axis named
        ``approximation``); rows a ``(component, measure)`` MultiIndex with
        blocks ``meta`` (the ``shape`` / ``loc`` / ``scale`` parameters, 0
        where a family has no such parameter, ``NaN`` on ``exact``),
        ``stats`` (achieved ``mean`` / ``cv`` / ``skew`` of the emitted
        law, clamp included, and ``ks``, the Kolmogorov distance to the
        realized cumulative), ``quantiles`` (one row per ``P`` of the
        ``tail_df`` ladder) and ``rel err`` (each family quantile as a
        relative error against ``exact`` on the same ladder). Every value
        is a float, so the frame formats as one numeric table; the DecL
        fragment of each fit is available from :func:`_approximation_laws`.

    Notes
    -----
    Everything a family column reports is the law of the **emitted
    program** (mirrored keyword, clamp included): one law per column, no
    mixing of the fit target and the fit result. The achieved moments come
    from the grid-mass evaluation ``diff(G(xs + bs/2), prepend=0)`` (whose
    first element carries any clamp atom at 0), the same cumulative that
    feeds ``ks``, so the ``stats`` and ``quantiles`` blocks cannot
    disagree about which law they describe.

    **The half-bucket edge is load bearing.** The realized density is
    built under the library's ``round`` (centered) convention,
    ``p_k = P(x_k - bs/2 < X <= x_k + bs/2)``, so its running sum means
    ``F(x_k) = P(X <= x_k + bs/2)``. A family cumulative read at the grid
    points themselves, ``G(x_k)``, would assign each bucket's mass to its
    right endpoint, a systematic ``+bs/2`` mean shift against the exact
    column (visible on a coarse grid: a normal fit of a mean-8 book at
    ``bs = 1`` reported mean 8.5). Reading ``G(x_k + bs/2)`` instead is
    exactly the ``round`` discretization of the emitted law, so both
    columns describe the same convention and both cumulatives compared by
    ``ks`` mean ``P(X <= x_k + bs/2)``. The ``rel err`` block reads each
    family quantile against the exact one as ``q_fam / q_exact - 1``; a 0
    exact quantile (a low rung on a loss book with mass at 0) reports
    ``NaN``, since no relative reading exists there.
    """
    xs = np.asarray(xs, dtype=float)
    exact_density = np.asarray(exact_density, dtype=float)
    F = np.cumsum(exact_density)
    edges = xs + 0.5 * float(bs)
    ps = _approximation_ladder()
    laws = _approximation_laws(m, cv, skew, signed_input)

    rows = ([('meta', 'shape'), ('meta', 'loc'), ('meta', 'scale'),
             ('stats', 'mean'), ('stats', 'cv'), ('stats', 'skew'),
             ('stats', 'ks')]
            + [('quantiles', float(p)) for p in ps]
            + [('rel err', float(p)) for p in ps])
    index = pd.MultiIndex.from_tuples(rows, names=('component', 'measure'))

    q_ex = np.asarray([float(q_exact(float(p))) for p in ps])
    data = {'exact': [np.nan, np.nan, np.nan,
                      float(m), float(cv), float(skew), 0.0]
            + list(q_ex) + [0.0] * len(ps)}
    for kind, law in laws.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            G = law['cdf'](edges)
            mass = np.diff(G, prepend=0.0)
            mean_a = float(mass @ xs)
            var_a = float(mass @ (xs - mean_a) ** 2)
            sd_a = math.sqrt(max(var_a, 0.0))
            cv_a = sd_a / mean_a if mean_a != 0 else np.nan
            skew_a = (float(mass @ (xs - mean_a) ** 3) / sd_a ** 3
                      if sd_a > 0 else np.nan)
            ks = float(np.max(np.abs(F - G)))
            quantiles = np.asarray([float(v) for v in law['ppf'](ps)])
            rel_err = np.where(q_ex != 0.0, quantiles / q_ex - 1.0, np.nan)
        data[kind] = ([law['shape'], law['loc'], law['scale'],
                       mean_a, cv_a, skew_a, ks]
                      + list(quantiles) + [float(v) for v in rel_err])
    df = pd.DataFrame(data, index=index)
    df.columns.name = 'approximation'
    return df


def approximation_density_frame(m, cv, skew, signed_input, xs, exact_density,
                                bs):
    """Build the ``approximation_density_df`` plotting feed.

    Parameters
    ----------
    m, cv, skew, signed_input, xs, exact_density
        As :func:`approximation_frame`.
    bs : float
        Bucket size; family densities are expressed as grid mass
        ``pdf(x) * bs`` so the columns overlay the discrete density
        directly.

    Returns
    -------
    pandas.DataFrame
        Indexed by the grid (named ``loss``); column ``exact`` (the
        realized density) then one column per family (axis named
        ``approximation``). A clamped family's column sums to ``1 -`` its
        clamp atom at 0, which the pdf does not carry.
    """
    xs = np.asarray(xs, dtype=float)
    laws = _approximation_laws(m, cv, skew, signed_input)
    data = {'exact': np.asarray(exact_density, dtype=float)}
    with np.errstate(divide='ignore', invalid='ignore'):
        for kind, law in laws.items():
            data[kind] = law['pdf'](xs) * float(bs)
    df = pd.DataFrame(data, index=pd.Index(xs, name='loss'))
    df.columns.name = 'approximation'
    return df


def max_log2(x):
    """
    Return the largest power of two d so that (x + 2**-d) - x == 2**-d, with d <= 30.
    Used in dhistogram severity types to determine the size of the step.
    """
    d = min(30, -np.log2(np.finfo(float).eps) - np.ceil(np.log2(x)) - 1)
    if (x + 2 ** -d) - x != 2 ** -d:
        raise ValueError('max_log2 failed')
    return d


#: Relative tolerance for deciding that an attachment lands on a grid point.
#: The index is at most order 1e6 on a realistic grid, so this leaves a wide
#: margin over float representation error while still catching a genuine miss:
#: the smallest real miss is half a bucket, an index error of 0.5.
_PICKS_GRID_RTOL = 1e-8

#: How far the compatible-bucket search halves the realized bucket. Past
#: ``bs / 2**20`` the suggestion needs more buckets to cover the same window
#: than any grid the sizer would build, so reporting no suggestion is more
#: useful than one that cannot be run.
_PICKS_BUCKET_HALVINGS = 20


def _is_integral(value, rtol=_PICKS_GRID_RTOL):
    """True when ``value`` is a whole number to within a relative tolerance."""
    return abs(value - round(value)) <= rtol * max(1.0, abs(value))


def _picks_compatible_bucket(attachments, bs):
    """The largest ``bs / 2**j`` dividing every attachment, or ``None``.

    Parameters
    ----------
    attachments : array
        Layer attachment points, all finite.
    bs : float
        The realized bucket size.

    Returns
    -------
    float or None
        The coarsest compatible bucket found within
        ``_PICKS_BUCKET_HALVINGS`` halvings, else ``None``.

    Notes
    -----
    Only halvings of the realized bucket are offered. The grid is a power of
    two lattice, so halving keeps every existing grid point and adds the
    midpoints, which is the change a user can make by pinning ``bs`` without
    disturbing anything else about the sizing. An arbitrary common divisor of
    the attachments (their gcd, say) would often be a bucket the sizer would
    never choose and would not divide the window cleanly.
    """
    b = float(bs)
    for _ in range(_PICKS_BUCKET_HALVINGS + 1):
        if all(_is_integral(a / b) for a in attachments):
            return b
        b /= 2.0
    return None


def _picks_grid_indices(attachments, xs, bs):
    """Positional index of each attachment, refusing any that misses the grid.

    Parameters
    ----------
    attachments : array
        Layer attachment points, ascending.
    xs : array
        The realized grid.
    bs : float
        The realized bucket size, ``xs[1] - xs[0]``.

    Returns
    -------
    numpy array of int
        The index into ``xs`` of each attachment.

    Raises
    ------
    ValueError
        If any attachment falls strictly inside a bucket, or above the top of
        the window. The message names every offender, the realized grid, and a
        bucket that would work.

    Notes
    -----
    Picks defines the layers the reweighting is solved on, so unlike a
    reinsurance tower it cannot rebucket. ``apply_reins_work`` evaluates
    piecewise linear ceder and netter functions at every grid point and lands
    the off grid results back on the lattice, which is faithful because the
    contract is a function of the loss. A pick is a *constraint on an integral
    between two boundaries*, so a boundary strictly inside a bucket has no
    faithful reading: the bucket's mass sits at one point and cannot be split
    between the layer below and the layer above without inventing a
    within-bucket distribution. Snapping the boundary to the nearest grid point
    was rejected (author ruling 2026-08-24) because it silently restates the
    tower the user asked for.

    The check is here rather than in :meth:`Aggregate.picks` so that every
    caller is gated, including direct use of this function.
    """
    xs = np.asarray(xs)
    attachments = np.asarray(attachments, dtype=float)
    top = xs[-1]

    positions = (attachments - xs[0]) / bs
    off_grid = [a for a, i in zip(attachments, positions)
                if np.isfinite(a) and a <= top and not _is_integral(i)]
    too_high = [a for a in attachments if not np.isfinite(a) or a > top]

    if off_grid or too_high:
        from ._bucket_window import _fmt_bs                # noqa: PLC0415 local
        parts = []
        if off_grid:
            named = ', '.join(f'{a:,.6g}' for a in off_grid)
            parts.append(
                f'picks: attachment{"s" if len(off_grid) > 1 else ""} {named} '
                f'{"do" if len(off_grid) > 1 else "does"} not lie on the grid, '
                f'which runs from {xs[0]:,.6g} in steps of bs={_fmt_bs(bs)}. '
                f'A layer boundary inside a bucket has no faithful reading, so '
                f'this is refused rather than snapped to the nearest bucket.')
            finite = [a for a in attachments if np.isfinite(a) and a <= top]
            suggestion = _picks_compatible_bucket(finite, bs)
            if suggestion is not None:
                parts.append(
                    f'Rebuild on bs={_fmt_bs(suggestion)}, which divides every '
                    f'attachment: pass bs={_fmt_bs(suggestion)} to build, or '
                    f'write hints{{bs={_fmt_bs(suggestion)}}} on a library '
                    f'entry (the MED entries are the model).')
            else:
                parts.append(
                    'No halving of the bucket down to '
                    f'bs={_fmt_bs(bs / 2 ** _PICKS_BUCKET_HALVINGS)} divides '
                    'every attachment, so pin bs to a common divisor of the '
                    'attachments instead, with hints{bs=...} on a library '
                    'entry.')
        if too_high:
            named = ', '.join('inf' if not np.isfinite(a) else f'{a:,.6g}'
                              for a in too_high)
            parts.append(
                f'picks: attachment{"s" if len(too_high) > 1 else ""} {named} '
                f'{"lie" if len(too_high) > 1 else "lies"} above the top of '
                f'the window, {top:,.6g}. Every attachment must be inside the '
                f'grid; raise log2 or bs to reach it, and note that the top '
                f'layer is capped by the window rather than unlimited.')
        raise ValueError(' '.join(parts))

    return np.rint(positions).astype(int)


def _picks_work(attachments, layer_loss_picks, xs, sev_density, n=1, sf=None, debug=False):
    """
    Adjust the layer unconditional expected losses to target. You need int xf(x)dx, but
    that is fraught when f is a mixed distribution. So we only use the int S version.
    ``fz`` was initially a frozen continuous distribution; but adjusted to sf function
    and dropped need for pdf function.

    See notes for how the parts are defined. Notice that::

        np.allclose(p.layers.v - p.layers.f, p.layers.l - p.layers.e)

    is true.

    **Every attachment must lie on the realized grid**, and inside the window.
    :func:`_picks_grid_indices` checks that first and raises a ``ValueError``
    naming the offenders and a compatible bucket size. A boundary strictly
    inside a bucket has no faithful reading, because that bucket's mass sits at
    one point and cannot be split between the layer below and the layer above;
    snapping it to the nearest grid point would silently restate the tower
    (author ruling 2026-08-24). A reinsurance tower over the same attachments
    is exempt because it rebuckets: a contract is a function of the loss and
    can be evaluated at every grid point, whereas a pick constrains an integral
    between two boundaries that therefore have to exist. Before 1.0.0a319 an
    off grid attachment surfaced as a raw pandas ``KeyError`` from the survival
    lookup. The integrals below are positional for the same reason: after the
    check the index is the honest coordinate, and an exact float label lookup
    is one representation drift from a spurious failure on a non dyadic grid.

    Infeasible picks (a target below the full-limit losses implied by the
    layers above it) produce a negative adjustment weight and hence negative
    adjusted probabilities. That is reported with a ``logger.warning`` naming
    the offending layers, and a second catch-all warning fires whenever the
    final adjusted density carries negative probabilities by any route. The
    adjusted density is still returned so the caller can inspect it. In debug
    mode the exact layer statistics check ``quad``'s error estimate relative
    to the integral's value, not absolutely, since the estimate grows with
    the integration range.

    :param attachments: array of layer attachment points, in ascending order (bottom to top). a[0]>0
    :param layer_loss_picks: Target means. If ``len(layer_loss_picks)==len(attachments)`` then the bottom layer, 0 to a[0],
      is added. Can be input as unconditional layer severity (i.e., :math:`\\mathbb{E}[(X-a)^+\\wedge y]`) or as the
      layer loss pick (i.e., :math:`\\mathbb{E}[(X-a)^+\\wedge y]'times n` where *n* is the number of ground-up (to the
      insurer) claims. Multiplying and dividing by :math:`S(a)` shows this equals conditional severity in the layer
      times the number of claims in the layer.) Actuaries usually estimate the loss pick to the layer in pricing. When
      called from :class:`Aggregate` the number of ground up claims is known.
    :param en: ground-up expected claims. Target is divided by ``en``.
    :param xs: x values for discretization
    :param sev_density: Series of existing severity density from Aggregate.
    :param sf: cdf function for the severity distribution.
    :param debug: if True, return debug information (layers, density with adjusted probs, audit
      of layer expected values.
    """

    # want xs, attachments, and sev_density to be numpy arrays
    xs = np.array(xs)
    attachments = np.array(attachments)
    # target is the unconditional layer expected loss, E[(X-a)^+ ^ y]
    target = np.array(layer_loss_picks) / n
    # print(n, layer_loss_picks, target)
    sev_density = np.array(sev_density)
    # figure bucket size
    bs = xs[1] - xs[0]

    # every attachment must be a grid point, else the layers below are solved
    # on boundaries the grid cannot express; raises naming a compatible bucket
    attachment_index = _picks_grid_indices(attachments, xs, bs)

    # dataframe of adjusted probabilties, starts here
    density = pd.DataFrame({'x': xs, 'p': sev_density}).set_index('x', drop=False)
    fill_value = max(0, 1. - density.p.sum())
    density['S'] = density.p.shift(-1, fill_value=fill_value)[::-1].cumsum()

    # the integrals below run from zero, which is the first row on the usual
    # severity grid; computed rather than assumed so a grid that starts
    # elsewhere keeps the label-slice semantics this replaced
    zero_index = max(0, int(np.rint((0.0 - xs[0]) / bs)))

    # numerical integrals - these match
    layers = pd.DataFrame(columns=['a', 'lev', 'int_fdx', 'aS', 'S'], index=range(1, 1+len(attachments)),
                          dtype=float)
    for i, (x, xi) in enumerate(zip(attachments, attachment_index)):
        # positional, not label: an exact float label lookup is one
        # representation drift from a spurious KeyError, and after the grid
        # check the index is the honest coordinate. iloc[zero_index:xi] is the
        # old loc[0:x-bs], since label slicing includes both endpoints.
        prefix = density.iloc[zero_index:xi]
        ix = prefix['S'].sum() * bs
        ix2 = prefix[['x', 'p']].prod(axis=1).sum()
        s_at_x = density['S'].iloc[xi]
        layers.loc[i+1, :] = [x, ix, ix2, x * s_at_x, s_at_x]

    # prob of loss in layer
    layers['p'] = layers.S.shift(1, fill_value=1) - layers.S
    # unconditional expected loss in layer
    layers['l'] = layers.lev - layers.lev.shift(1, fill_value=0)
    layers.index.name = 'layer'
    # bottom of layer
    layers['a_bottom'] = layers.a.shift(1, fill_value=0)
    # width of layer
    layers['y'] = layers.a - layers.a_bottom
    # e = rectangle to right in int S computation
    layers['e'] = layers.S * layers.y
    # f = rectangle below attachment in int xf computation
    layers['f'] = layers.p * layers.a_bottom
    # these are two versions of m (unconditional)
    # m-bit: int S - e == int xf - f
    layers['m'] = layers.l - layers.e
    # int f dx in layer
    layers['v'] = layers.f + layers.m
    # and conditional vertical loss in layer
    layers['v_c'] = layers.v / layers.p
    layers = layers[['a_bottom', 'a', 'y', 'lev', 'S', 'p', 'l', 'v', 'v_c', 'm', 'e', 'f']]

    # add weights w and offsets=omega, computed from the top layer down
    layers['t'] = target
    layers['w'] = 0.0
    layers['ω'] = 0.0

    # this computation leaves the tail unchanged and uses the same "adjust the curve" method
    # in all layers
    ω = layers.loc[len(layers), 'S']
    for i in layers.index[::-1]:
        layers.loc[i, 'w'] = (layers.loc[i, 't'] - ω * layers.loc[i, 'y']) / layers.loc[i, 'm']
        layers.loc[i, 'ω'] = ω
        ω += layers.loc[i, 'p'] * layers.loc[i, 'w']

    # a negative weight means the pick cannot be reached by scaling the
    # in-layer part of the curve: the target is below the full-limit losses
    # ω y implied by the layers above, the adjusted survival function
    # increases across the layer, and the density goes negative
    infeasible = layers.index[layers['w'] < 0].tolist()
    if infeasible:
        logger.warning(f'Infeasible picks: negative adjustment weight in layer(s) {infeasible}. '
                       'The pick is below the full-limit losses implied by the layers above it, '
                       'so the adjusted severity has negative probabilities. Revise the picks.')

    # adjusted S: bins -> layer number; add in offsets
    density['bin'] = pd.cut(density.x, np.hstack((0, layers.a.values)), include_lowest=True, right=True)
    # layer description returned by cut to layer number in layers
    mapper = {i:j+1 for j, i in enumerate(density.bin.unique())}
    density['layer'] = density.bin.map(mapper.get)

    density['ω'] = density.layer.map(layers.ω).astype(float)
    # S(a_n-1)
    density['Sa'] = density.layer.map(layers.S).astype(float)
    density['w'] = density.layer.map(layers.w).astype(float)

    density['S_adj'] = np.minimum(1, density.ω + (density.S - density.Sa) * density.w)
    # no change in the tail
    density.loc[attachments[-1]:, 'S_adj'] = density.loc[attachments[-1]:, 'S']
    # adj probs as difference of S
    density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
    achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
    # display(achieved)
    if abs(achieved.iloc[0] - target[0]) > 1e-3:
        # issues with hitting 1
        logger.warning(f'achieved[0] = {achieved.iloc[0]} != target[0] = {target[0]}')
        # take top right corner off
        if target[0] > attachments[0]:
            raise ValueError(f'target[0] = {target[0]} > first attachment[0] = {attachments[0]} which is impossible.')
        s0 = 2 * (attachments[0] - target[0]) / (1 - layers.loc[1, 'ω'])
        # snap to index
        s0 = bs * np.round(s0 / bs, 0)
        # convert to probability
        s = attachments[0] - s0
        density.loc[0:s, 'S_adj'] = 1.0
        temp = np.array(density.loc[s+bs:attachments[0]].index)
        wts = (temp - s) / s0
        density.loc[s+bs:attachments[0], 'S_adj'] = 1 - wts + layers.loc[1, 'ω'] * wts
        # update
        density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
        achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
        logger.warning(f'Revised layer 1 achieved = {achieved.iloc[0]}')

    density['diff S'] = density['S'] - density['Sa']

    # catch-all for any route to a negative density (the weight warning above
    # names the cause when a single pick is at fault; interactions with the
    # cap at 1 or the bottom-layer rebuild can also drive p_adj negative)
    min_p_adj = density['p_adj'].min()
    if min_p_adj < -1e-10:
        logger.warning(f'Adjusted severity has negative probabilities (min {min_p_adj:.6g}): '
                       'the layer loss picks are mutually infeasible. Revise the picks.')

    if debug is False:
        return density['p_adj'].values

    # data frame of layer statistics from input density
    exact = None
    if sf is not None:
        logger.warning('sf passed in; computing exact layer statistics')
        exact = pd.DataFrame(columns=['a', 'lev', 'aS', 'S'],
                             index=range(1, 1+len(attachments)), dtype=float)
        for i, x in enumerate(attachments):
            ix = quad(sf, 0, x)
            # quad reports an absolute error estimate. Over a wide range the
            # estimate can exceed any fixed absolute tolerance while the
            # integral itself is large and accurate, so test the error
            # relative to the value.
            assert ix[1] < 1e-6 * max(ix[0], 1e-12), \
                f'quad relative error {ix[1] / max(ix[0], 1e-12):.3g} too large integrating sf to {x}'
            sf_ = sf(x)
            exact.loc[i+1, :] = [x, ix[0], x * sf_ if x < np.inf else 0.0, sf_]

    if exact is None:
        l = layers.l
        ln = 'layers'
    else:
        l = exact.lev - exact.lev.shift(1, fill_value=0)
        ln = 'exact'

    t = pd.concat((l,
                   density.groupby(density.layer.shift(-1)).apply(lambda g: g['S'].sum() * bs),
                   achieved,
                   ), keys=[ln, 'computed', 'adj'], axis=1)
    t.loc['sum'] = t.sum()
    Picks = namedtuple('picks', ['layers', 'exact', 'density', 'audit'])
    return Picks(layers=layers, exact=exact, density=density, audit=t)


def _integral_by_doubling(func, x0, err=1e-8):
    r"""
    Compute :math:`\int_{x_0}^\infty f` as the sum

    .. math::

        \int_{x_0}^\infty f = \sum_{n \ge 0} \int_{2^nx_0}^{2^{n+1}x_0} f

    Caller should check the integral actually converges.

    :param func: function to be integrated.
    :param x0: starting x value
    :param err: desired accuracy: stop when incremental integral is <= err.
    """
    ans = 0.
    counter = 0
    # from to
    f, t = x0, 2 * x0
    last_int = 10
    while last_int > err:
        s = quad(func, f, t)
        if s[1] > err:
            raise ValueError(
                f'Questionable integral numeric convergence, err {s[1]:.4g}\n'
                f'f={f}, t={t}, x0={x0}, counter={counter}')
        last_int = s[0]
        ans += s[0]
        f, t = t, 2 * t
        counter += 1
        if counter > 96:
            raise ValueError(f'counter = {counter} and error = {err}')
    return -ans


# ---------------------------------------------------------------------------
# Stats DataFrame helpers
# ---------------------------------------------------------------------------
# ``Aggregate.stats_df`` is the canonical (component, measure) × view
# DataFrame holding theoretical + empirical moments. ``MomentAggregator``
# emits its per-component / totals statistics as flat names like ``freq_1``,
# ``actual_m``; ``_flat_col_to_stats_index`` maps each to the
# ``(component, measure)`` tuple used by the ``stats_df`` row MultiIndex.

_STATS_META_NAMES = frozenset({
    'name', 'limit', 'attachment', 'el', 'prem', 'lr', 'sevcv_param',
    'mix_cv', 'wt',
})


_STATS_MEASURE_MAP = {'1': 'ex1', '2': 'ex2', '3': 'ex3', 'm': 'mean'}


def _flat_col_to_stats_index(col):
    """Map a flat ``MomentAggregator`` moment name to ``(component, measure)``.

    Examples: ``'freq_1' → ('freq', 'ex1')``, ``'actual_m' → ('agg', 'mean')``,
    ``'limit' → ('meta', 'limit')``.

    Used to bridge the flat moment names emitted by
    :meth:`MomentAggregator.get_fsa_stats` / ``column_names()`` to the
    canonical ``(component, measure)`` MultiIndex used by ``stats_df``.
    """
    if col in _STATS_META_NAMES:
        return ('meta', col)
    comp, _, measure = col.partition('_')
    if comp in ('freq', 'sev', 'agg'):
        return (comp, _STATS_MEASURE_MAP.get(measure, measure))
    raise ValueError(f'Cannot map column {col!r} for stats_df build.')


# Canonical row MultiIndex for ``stats_df`` — written directly in __init__
# (component columns) and the post-loop totals block (``mixed`` /
# ``independent``). All ``meta`` rows up top, then freq/sev/agg moment blocks.
# The frame is all-float: ``self.name`` is already an attribute, no need for
# a ``('meta','name')`` string row that would force ``dtype=object``.
_STATS_ROW_INDEX = pd.MultiIndex.from_tuples(
    [
        ('meta', 'limit'), ('meta', 'attachment'),
        ('meta', 'el'), ('meta', 'prem'), ('meta', 'lr'),
        ('meta', 'sevcv_param'), ('meta', 'mix_cv'), ('meta', 'wt'),
        ('freq', 'ex1'), ('freq', 'ex2'), ('freq', 'ex3'),
        ('freq', 'mean'), ('freq', 'cv'), ('freq', 'skew'),
        ('sev', 'ex1'), ('sev', 'ex2'), ('sev', 'ex3'),
        ('sev', 'mean'), ('sev', 'cv'), ('sev', 'skew'),
        ('agg', 'ex1'), ('agg', 'ex2'), ('agg', 'ex3'),
        ('agg', 'mean'), ('agg', 'cv'), ('agg', 'skew'),
    ],
    names=['component', 'measure'],
)


#: Shared return type of the two eventual-ruin solvers,
#: :meth:`Aggregate.pollaczeck_khinchine` and :meth:`Aggregate.wiener_hopf`.
#: A namedtuple IS a tuple, so legacy positional unpacking
#: ``ruin, find_u, mean, dfi = ...`` keeps working. ``ruin`` is the pd.Series
#: ``psi(u)`` on the u-grid; ``find_u`` the capital-lookup closure; ``mean``
#: the discretized severity mean; ``density`` the method's u-grid density
#: vector -- the integrated-severity (equilibrium, ladder-height) density
#: ``dfi`` for Pollaczeck-Khinchine, the pmf of the all-time maximum for
#: Wiener-Hopf.
RuinFunction = namedtuple('RuinFunction', ['ruin', 'find_u', 'mean', 'density'])


#: Fixed default seed for the ruin simulation (:meth:`Aggregate._ruin_paths`).
#: A constant rather than ``None`` so a served ruin document is hash-stable
#: and cacheable; pass ``seed=None`` explicitly to draw a fresh seed (the
#: Sample action). See ``dev/plan-pk-tab.md`` ruling 2.
_RUIN_SEED = 20260905

#: Return type of :meth:`Aggregate._ruin_paths`, the simulation core shared
#: by :func:`aggregate.pedagogy.ruin_example`, the ``ruin`` exhibit and the
#: ``ruin`` chart. Groups, in order: identity (``fname``, the solver's
#: :data:`RuinFunction`, the seed actually used); the premium and moment
#: scalars; the full-simulation estimate; and the drawable material (the
#: horizon, the sampled paths, the trend line and LIL funnel arrays).
_RuinPaths = namedtuple('_RuinPaths', [
    'fname', 'rf', 'seed',
    'rho', 'u0', 'psi_u0', 'c', 'mx', 'var_x', 'mw', 'var_w', 'mu', 'sd',
    'sigma2',
    'n_sims', 'n_steps', 'n_ruin', 'p_sim', 'se_sim', 'ruin_time',
    't_plot', 'paths', 'tg', 'trend', 'tl', 'band'])


#: Return type of :meth:`Aggregate.sev`, the *exact* continuous severity: the
#: en-weighted mixture of the component :class:`~aggregate.Severity` objects,
#: evaluated from the input distributions rather than from the discretized
#: ``bs``-grid in ``sev_density_df``. ``cdf`` / ``sf`` / ``pdf`` are the forward
#: functions; ``ppf`` / ``isf`` are their inverses, added in 1.0.0a180. Contrast
#: :meth:`Aggregate.q_sev`, which snaps to the grid.
SevFunctions = namedtuple('SevFunctions', ['cdf', 'sf', 'pdf', 'ppf', 'isf'])


def _mixture_inverse(q, mixture_fn, component_invs, increasing):
    """Invert a monotone weighted-mixture cdf or sf by bracketed root finding.

    A weighted mixture has no closed-form inverse, but it is bracketed for free
    by the inverses of its own components, so no search for a starting interval
    is needed.

    Parameters
    ----------
    q : float or array_like
        Probability level(s) to invert. ``0 < q < 1`` for a useful answer;
        values outside are passed through to the component inverses, which
        return the support endpoints or ``nan`` as scipy does.
    mixture_fn : callable
        The weighted mixture function being inverted, ``cdf`` or ``sf``.
    component_invs : list of callable
        The matching per-component inverses, ``ppf`` for a cdf, ``isf`` for an
        sf. Must be the components of ``mixture_fn``.
    increasing : bool
        ``True`` when ``mixture_fn`` is a cdf (nondecreasing), ``False`` for an
        sf (nonincreasing). Sets the sign of the bracket test.

    Returns
    -------
    numpy scalar or ndarray
        The inverse, matching the shape of ``q``. A 0-d input gives a numpy
        scalar, following the :meth:`Severity._unwrap` convention.

    Notes
    -----
    **Why the component inverses bracket the mixture inverse.** Write the
    mixture as :math:`G(x) = \\sum_i w_i G_i(x)` with :math:`w_i > 0`,
    :math:`\\sum_i w_i = 1`, each :math:`G_i` monotone in the same direction.
    Take the cdf case. Let :math:`lo = \\min_i G_i^{-1}(q)`, attained by
    component :math:`j`. Then :math:`G_j(lo) \\ge q`, and every other component
    has :math:`G_k^{-1}(q) \\ge lo`, hence :math:`G_k(lo) \\ge q` as well, so
    :math:`G(lo) \\ge q`. Symmetrically :math:`G(hi) \\le q` at
    :math:`hi = \\max_i G_i^{-1}(q)`. The root therefore lies in ``[lo, hi]``
    and :func:`scipy.optimize.brentq` converges on it. The sf case is the same
    argument with the inequalities reversed.

    A severity whose survival function *jumps* past ``q`` (a limit, a
    discrete component) has no exact root. brentq still converges to the jump
    location, which is the correct lower-quantile answer; the same-sign guard
    below covers the degenerate case where the bracket collapses onto it.
    """
    q = np.asarray(q, dtype=float)

    def _one(qi):
        candidates = np.array([inv(qi) for inv in component_invs], dtype=float)
        lo, hi = float(np.min(candidates)), float(np.max(candidates))
        # all components agree, or the bracket is degenerate / unbounded: no
        # root to find, and brentq would reject the interval
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            return lo
        f_lo = mixture_fn(lo) - qi
        f_hi = mixture_fn(hi) - qi
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        if np.sign(f_lo) == np.sign(f_hi):
            # the mixture stepped clean over qi; the crossing is the endpoint
            # the function has already passed, which is hi for an increasing
            # mixture and lo for a decreasing one
            return hi if increasing else lo
        return float(brentq(lambda x: mixture_fn(x) - qi, lo, hi))

    out = np.array([_one(float(qi)) for qi in q.ravel()], dtype=float).reshape(q.shape)
    return out[()] if out.ndim == 0 else out


def _ruin_find_u(ruin, kind):
    """Capital-lookup closure ``p -> u`` over a decreasing ruin Series.

    Parameters
    ----------
    ruin : pd.Series
        ``psi(u)`` indexed by initial surplus ``u``, decreasing in ``u``.
    kind : str
        ``'index'`` snaps to the grid point; anything else linearly
        interpolates between the bracketing grid points.

    Returns
    -------
    callable
        ``find_u(p)`` returning the initial surplus with eventual ruin
        probability ``p``.
    """
    if kind == 'index':
        def find_u(p):
            idx = len(ruin) - ruin[::-1].searchsorted(p, 'left')
            return ruin.index[idx]
    else:
        def find_u(p):
            below = len(ruin) - ruin[::-1].searchsorted(p, 'left')
            above = below - 1
            q_below = ruin.index[below]
            q_above = ruin.index[above]
            p_below = ruin.iloc[below]
            p_above = ruin.iloc[above]
            return q_below + (p - p_below) / (p_above - p_below) * (q_above - q_below)
    return find_u


def _lundberg_exponent(p_x, xs_x, rho):
    """Adjustment coefficient ``R`` of the compound Poisson surplus process.

    Parameters
    ----------
    p_x : ndarray
        Discretized severity pmf (normalized inside).
    xs_x : ndarray
        The severity outcomes the pmf sits on.
    rho : float
        Margin-to-loss ratio; the premium rate is ``c = (1 + rho) lambda
        E[X]``.

    Returns
    -------
    float or None
        The positive Lundberg root, or ``None`` when no root is bracketed
        under the overflow guard.

    Notes
    -----
    ``R`` solves ``M_X(R) = 1 + (1 + rho) E[X] R`` (the Cramer-Lundberg
    adjustment equation with the Poisson rate cancelled), giving the
    classical bound ``psi(u) <= exp(-R u)``. On the discretized pmf the
    mgf is a finite sum, so a root exists whenever the bracket top
    ``R_max = 700 / max(x)`` (the ``exp`` overflow guard) reaches past it;
    a heavy-tailed book discretized on a wide grid can fail the bracket,
    and ``None`` is the honest answer there rather than a spurious root.
    """
    p = np.asarray(p_x, dtype=float)
    p = p / p.sum()
    x = np.asarray(xs_x, dtype=float)
    mx = float(p @ x)
    x_top = float(x.max())
    if not x_top > 0:
        return None

    def h(r):
        return float(p @ np.exp(r * x)) - 1.0 - (1.0 + rho) * mx * r

    r_max = 700.0 / x_top
    # h(0) = 0 with negative slope -rho E[X]; a usable bracket needs
    # h(r_max) > 0
    if h(r_max) <= 0:
        return None
    return float(brentq(h, 1e-12, r_max))


class Aggregate(HelpMixin, LabeledMixin, ProgramMixin):
    """Compound (aggregate) probability distribution.

    Implements the FFT-based algorithm of Mildenhall (2024): discretize
    severity, FFT, apply frequency PGF, inverse FFT. See
    ``_freq_sev_convolution`` for the five-line core; ``update_work`` for the
    orchestration (severity prep → occurrence reinsurance → convolution →
    aggregate reinsurance → audit). Validation by theoretical-vs-empirical
    moment comparison (paper §4.7) lives in the ``empirical`` and ``error``
    columns of ``stats_df`` written at the end of ``update_work``.

    Construction is via the ``__init__`` arguments below, or — more usually —
    via :func:`build` parsing DecL.

    **Public surface that Portfolio and Bounds depend on.** Three stats
    surfaces — ``info`` for text, ``summary_df`` for the daily risk view,
    and ``stats_df`` for everything else — plus the compute and risk-measure
    surface:

    Stats / display
        - ``info``: one-screen textual summary (frequency, severity, layer,
          grid, validation flag). Not stats.
        - ``summary_df``: 3-row Freq / Sev / Agg at-a-glance moments + key
          percentiles. The daily-driver headline. ``tail_df`` is the companion
          return-period table; ``validation_df`` the moment-error QA frame.
        - ``stats_df``: ``MultiIndex (component, measure)`` × per-component
          / ``mixed`` / ``independent`` / ``empirical`` / ``error``.
          Single source of truth for Aggregate moments — see the property
          for the row / column reference.
        - ``stats_df``: ``MultiIndex (component, measure)`` × per-component
          / ``mixed`` / ``independent`` / ``empirical`` / ``error``.
          Single source of truth for Aggregate moments — see the property
          for the row / column reference.

    Data attributes
        - ``agg_density``: empirical PMF on the bucket grid (set by
          ``update_work``; consumed by Portfolio).
        - ``ftagg_density``: FT of the aggregate density (consumed by
          Portfolio's copula combine).
        - ``density_df``: per-bucket density / CDF / risk-measure frame.
        - ``n``: total frequency.
        - ``name``, ``program``, ``note``: spec metadata.
        - ``bs``, ``log2``, ``xs``: discretization grid.

    Methods
        - ``update``, ``update_work``: trigger / drive the compute.
        - ``q``, ``q_sev``, ``tvar``, ``tvar_sev``, ``cdf``, ``sf``, ``pdf``,
          ``pmf``, ``var_dict``: risk-measure surface.
        - ``sample``: draw from the discretised aggregate.
        - ``price``: distortion-based pricing.
        - ``approximate``, ``entropy_fit``: parametric fits to the FFT output.
        - ``apply_distortion``, ``pollaczeck_khinchine``: distortion / ruin.
        - ``plot``: single plotting entry point.
        - ``snap``, ``picks``, ``unwrap``,
          ``aggregate_error_analysis``, ``severity_error_analysis``: utilities.

    Methods / attributes with a leading underscore are internal —
    ``_init_stats_df``, ``_record_component``, ``_freq_sev_convolution``,
    ``_apply_reins_work``, ``_limits``, ``_html_info_blob``,
    ``_grid_distribution``, … . The legacy ``audit_df`` / ``report_df`` /
    ``report_ser`` / ``statistics`` / ``statistics_df`` /
    ``statistics_total_df`` surface has been removed; consult ``stats_df``
    instead.
    """

    # ================================================================
    # Public read-only properties: spec, density frame, reinsurance frames
    # ================================================================

    @property
    def spec(self):
        """
        Get the dictionary specification, but treat as a read only
        property

        :return:
        """
        return self._spec

    @property
    def spec_ex(self):
        """
        All relevant info.

        :return:
        """
        return {'type': type(self), 'spec': self.spec, 'bs': self.bs, 'log2': self.log2,
                'sevs': len(self.sevs)}

    def _tail_info(self):
        """Build the :class:`~aggregate.tail.TailInfo` for this aggregate.

        Spec-only (frequency name + severity families); touches no computed
        density, so it is valid *before* ``update()``. A ``_certified_bounded``
        override short-circuits to a BOUNDED result so the certify contract and
        the lifted-natural-allocation guard are preserved.

        Returns
        -------
        aggregate.tail.TailInfo
        """
        if getattr(self, '_certified_bounded', False):
            return _tail.TailInfo(
                freq=TailClass.BOUNDED, sev=TailClass.BOUNDED, agg=TailClass.BOUNDED,
                freq_lc=None, sev_lc=None, agg_lc=None, alpha=None,
                driver='certified', flags={'certified': True})
        # ``reference=True``: a severity that came from a ``sev agg.NAME``
        # reference answers for the object it stands for, so a reference to an
        # unbounded aggregate is not bounded however finite its atoms are. This
        # is the reporting reading, and it is what ``bounded`` and the ``p=1``
        # pricing guard want. The bucket sizer takes the other one, through
        # ``_bounded_severity_window``.
        return _tail.aggregate_tail_info(self.frequency, self.sevs, reference=True)

    @property
    def tail_class(self):
        """The ``(freq, sev, agg)`` :class:`~aggregate.tail.TailClass` triple.

        The authoritative tail-thickness computation: frequency and severity
        rungs by deterministic family lookup, the aggregate rung by the
        ``max`` combine rule. ``.bounded`` and the tail text are derived from
        this. Returns a :class:`~aggregate.tail.TailClasses` namedtuple with
        ``.freq`` / ``.sev`` / ``.agg`` fields.
        """
        return self._tail_info().classes

    @property
    def bounded(self) -> bool:
        """Whether the aggregate has bounded support.

        Derived view: ``True`` iff the aggregate tail class is
        :attr:`~aggregate.tail.TailClass.BOUNDED`, i.e. the frequency *and*
        every severity component is bounded. Frequencies in
        :data:`~aggregate.tail._BOUNDED_FREQS` (``fixed``, ``bernoulli``,
        ``binomial``, ``empirical``) are bounded; mixed Poisson / negbin / etc.
        are not. A severity is bounded when it is a histogram, a fixed atom, a
        bounded scipy family, or carries a finite layer ``exp_limit`` or splice
        ``sev_ub``. Conservative: ``False`` whenever boundedness cannot be
        proved from the spec. Set ``self.bounded = True`` to certify (e.g. a
        fat-tailed scipy severity with a large layer cap the heuristic misses).
        """
        return self._tail_info().agg == TailClass.BOUNDED

    @bounded.setter
    def bounded(self, value: bool) -> None:
        if value is not True and value is not False:
            raise ValueError('bounded must be True (certify) or False (reset)')
        self._certified_bounded = bool(value)

    @property
    def tail_description(self) -> str:
        """Three aligned lines summarising the frequency, severity, and aggregate tails.

        Short narrative over the layered :attr:`tail_behavior_df` report -- per-layer
        support and per-side tail class, plus the aggregate concentration. E.g.::

            frequency tail           poisson, count [0, inf), super-exponential right tail
            severity tail            lognorm, [0, inf), subexponential right tail
            aggregate tail           [0, inf), subexponential right tail; not concentrated (cv=1.5)

        The verbose form is :attr:`tail_explanation`; the lines are also appended
        to :meth:`info`.
        """
        return '\n'.join(_tail.describe_rows(self._tail_rows()))

    @property
    def tail_explanation(self) -> str:
        """Verbose prose over the layered tail report (the per-component story).

        Walks the book bottom-up -- frequency, the severity components and their
        blend, the aggregate -- naming the single-big-jump mechanism (or the
        frequency driver) for a thick right tail, any power-law moment failure,
        and the concentration. Derived from :attr:`tail_behavior_df`.
        """
        return _tail.explain_rows(self._tail_rows(), self._tail_info())

    @property
    def bs_window_df(self) -> 'pd.DataFrame':
        """Curated, read-only view of the bucket/window decision (``[bs-reporting]``).

        One row per sizing method that ran (``moment``, ``exact_discrete``,
        ``bounded_small``, ``windowed``, ``sbj``) plus the realized ``used``
        grid, indexed by ``method``, culled to the user-facing columns: whether
        the method ``applies``, whether it was ``selected``, the method window
        (``x_min`` / ``x_max``) and its width ``W``, the grid (``bs`` /
        ``log2``), the log2 the window ``log2_need``s at that ``bs``, what
        fraction of the distribution the window holds (``coverage``), any
        estimated ``clipped`` far-tail mass, and a one-line ``note``.

        Returns ``None`` before :meth:`update`. See :attr:`bs_description` /
        :attr:`bs_explanation` for the narrative.

        Notes
        -----
        **``W`` and ``coverage`` are the point of the frame**, and were missing
        from it until ``1.0.0a254``. The leaf exists to answer "is this grid big
        enough", and the width and the coverage are the answer: without them a
        reader has a list of candidate windows and no way to compare them. They
        were curated out as expert material, which was the wrong call about
        which columns carry the meaning.

        ``coverage`` is a **string** (``'1-1e-12'``,
        ``'E[N]-adj 1-1e-12'``), carrying precision a float cannot and saying
        which of two things was held to that precision.

        **Every column carries its own dtype**, which it did not until
        ``1.0.0a275``: ``applies`` / ``selected`` bool, the window and grid
        columns float, ``log2`` / ``log2_need`` nullable ``Int64`` (an exponent
        reads as an integer, and a method that never ran records none),
        ``coverage`` / ``note`` string. The frame used to be built by
        transposing a method-per-column block, which typed the lot ``object``
        and cost a served table its right alignment and its raw values.
        """
        if self._bs_window_df is None:
            return None
        cols = ['applies', 'selected', 'x_min', 'x_max', 'W', 'bs', 'log2',
                'log2_need', 'coverage', 'clipped', 'note']
        return self._bs_window_df.reindex(columns=cols).copy()

    @property
    def bs_description(self) -> str:
        """One-line summary of the chosen bucket grid (``[bs-reporting]``).

        The winning method and the realized ``(bs, log2, x_min)`` with the grid
        top, plus a clip note when the far tail is truncated. The verbose form is
        :attr:`bs_explanation`; the ANSI-coloured variant is
        ``aggregate.distributions.bs_describe(agg, color=True)``.
        """
        return bs_describe(self)

    @property
    def bs_explanation(self) -> str:
        """Verbose prose explaining the bucket-grid choice (``[bs-reporting]``).

        What the book is (the aggregate tail one-liner), which methods applied
        and which won and why, the realized grid, and any far-tail clip with how
        to widen it. The ANSI-coloured variant is
        ``aggregate.distributions.bs_explain(agg, color=True)``.
        """
        return bs_explain(self)

    def sharpen(self, bs=None, log2=None, *, log2_cap=20,
                bs_limit=_bucket_window.SHARPEN_BS_LIMIT, power=2,
                good_enough=0.5, execute=True):
        """Probe the grid neighbourhood and move to a better ``(bs, log2)``.

        Delegated to :func:`~aggregate._bucket_window.sharpen`, where the score,
        the probe geometry and the selection rule are documented in full.

        :meth:`update` *chooses* a grid from the analytic moments before any FFT
        runs; this *audits* that choice afterwards. A row per ``log2``, and
        within each a line search out from the current bucket: ``bs`` is doubled
        until :attr:`validation_score` stops improving, then halved likewise,
        capped at ``bs_limit`` each way. It then takes **the best score among the
        cells that do not grow** ``log2``, growing by one only when nothing at
        the current size or smaller reaches ``good_enough``. A discrete severity
        whose atoms are a whole number of buckets is already exact, so the bucket
        is pinned and only ``log2`` is probed.

        Populates :attr:`sharpen_df`, :attr:`sharpen_description` and
        :attr:`sharpen_explanation`. Returns ``self``, so the call chains; the
        object is moved **in place**, so ``a = a.sharpen()`` rebinds the same
        object rather than producing a second one.

        **It also pins the outcome onto the object's own text.**
        :attr:`program`, :attr:`note` and :attr:`hints` are rewritten together
        so the three agree and ``build(a.program)`` reproduces the sharpened
        object: a moved grid becomes ``hints{log2=...; bs=...}``, a confirmed
        one a ``note{sharpen: ...}`` and no hints. Probing twice replaces the
        verdict rather than stacking a second one. See
        :func:`aggregate._program.pin_sharpen`.

        Examples
        --------
        ::

            a = build('agg X 100 claims sev lognorm 100 cv 2 poisson')
            a.sharpen()
            print(a.sharpen_description)
            print(a.hints)          # the grid it settled on
            build(a.program)        # rebuilds on that grid
        """
        return _bucket_window.sharpen(
            self, bs, log2, log2_cap=log2_cap, bs_limit=bs_limit, power=power,
            good_enough=good_enough, execute=execute)

    @property
    def sharpen_df(self) -> 'pd.DataFrame':
        """The last :meth:`sharpen` probe, one tidy row per grid cell.

        Columns: the integer offsets ``d_bs`` / ``d_log2`` from the probe centre,
        the realized ``bs`` / ``log2`` / ``extent`` / ``x_min``, the ``score`` and
        its six normalized terms (``u_sev_mean`` through ``u_agg_skew``), the
        ``aliasing`` ratio, the ``validation`` verdict, a ``warnings`` count,
        ``seconds``, the ``selected`` winner, and a ``note`` carrying the
        exception text for any cell that failed.

        Indexed by ``(d_bs, d_log2)``, so the picture is one unstack away::

            a.sharpen_df.score.unstack('d_log2')

        The line search makes the rows ragged, so cells never visited come back
        ``nan``. ``None`` before :meth:`sharpen` runs.
        """
        if self._sharpen_df is None:
            return None
        return self._sharpen_df.copy()

    @property
    def sharpen_description(self) -> str:
        """One-line summary of the last :meth:`sharpen` probe.

        What was scored, what won, and whether the grid moved. The verbose form
        is :attr:`sharpen_explanation`.
        """
        return _bucket_window.sharpen_describe(self)

    @property
    def sharpen_explanation(self) -> str:
        """Verbose prose explaining the last :meth:`sharpen` probe.

        What the score means, how to read the probe as extent-limited versus
        resolution-limited, why ``log2`` was or was not grown, and what changed.
        The short form is :attr:`sharpen_description`.
        """
        return _bucket_window.sharpen_explain(self)

    @property
    def sharpen_program(self) -> str:
        """The program that rebuilds this aggregate on the sharpened grid.

        The fourth thing a probe produces, beside :attr:`sharpen_df`,
        :attr:`sharpen_description` and :attr:`sharpen_explanation`.
        :meth:`sharpen` pins its outcome onto :attr:`program` as it finishes,
        so this is that program **rendered to read**: the ``spread`` layout
        with the trailer left in, since here the trailer is the payload.
        ``hints{log2=...; bs=...}`` when the grid moved, a
        ``note{sharpen: ...}`` and deliberately no hints when it did not.
        ``''`` before :meth:`sharpen` runs.

        Which of the four to reach for: this one for text to read or share,
        :attr:`program` for the one-line stamp, :attr:`hints` for the settings
        alone, :attr:`note` for the verdict alone.

        Delegated to :func:`aggregate._program.sharpen_program`; the three
        outcomes are documented on :func:`aggregate._program.pin_sharpen`.
        """
        return _program.sharpen_program(self)

    def pnl_program(self, loss_ratio=0.70, expense_ratio=0.25, *,
                    net_combined_ratio=None, occ_combined_ratio=None,
                    agg_combined_ratio=None) -> str:
        """The program that wraps this aggregate in a P&L.

        Returns ``pnl NAME_PnL <premium> less <engine> less <expense>``, with
        this object's body inlined as the engine, so the text is self-contained
        and builds anywhere. The premium is ``inherit premium`` when the
        exposure states one, and otherwise expected loss over ``loss_ratio``;
        ``expense_ratio=0`` omits the expense clause.

        ``net_combined_ratio`` additionally **prices the reinsurance**, which
        is what silences the ``ZeroPremiumCessionWarning`` an unpriced cession
        raises: the net book and each cover are priced separately and added,
        then grossed up once for expenses.

        Delegated to :func:`aggregate._program.pnl_program`, where the premium
        rule and the trailer's move up to the wrapping ``pnl`` are documented
        in full, and to :func:`aggregate._program._pnl_technical` for the
        ladder's algebra.

        Parameters
        ----------
        loss_ratio : float, default 0.70
            Sizes the premium, and only when there is none to inherit and
            ``net_combined_ratio`` is ``None``.
        expense_ratio : float, default 0.25
            Gross expense as a fraction of premium.
        net_combined_ratio : float, optional
            Expected net loss over net technical premium. ``None``, the
            default, leaves the function exactly as it was and any cession
            unpriced; a number engages the ladder.
        occ_combined_ratio, agg_combined_ratio : float or sequence, optional
            Each tier's combined ratio, defaulting to ``net_combined_ratio``.
            A sequence gives one value per layer, in declaration order.

        Returns
        -------
        str
            DecL for the wrapping P&L.

        Examples
        --------
        ::

            a = build('agg X 100 claims sev lognorm 100 cv 2 poisson')
            build(a.pnl_program(loss_ratio=0.65))

            r = build('agg Y 10 claims sev lognorm 100 cv 2 '
                      'occurrence net of 100 xs 100 poisson')
            build(r.pnl_program(net_combined_ratio=0.9))
        """
        return _program.pnl_program(self, loss_ratio=loss_ratio,
                                    expense_ratio=expense_ratio,
                                    net_combined_ratio=net_combined_ratio,
                                    occ_combined_ratio=occ_combined_ratio,
                                    agg_combined_ratio=agg_combined_ratio)

    def reins_program(self, cession) -> str:
        """The program that rebuilds this aggregate with ``cession`` added.

        The clause lands in its correct slot, which is the point: an occurrence
        cession sits **before** the frequency clause and an aggregate cession
        after it, so neither can simply be appended to the program text. The
        result is self-contained, with this object's body inlined, so it builds
        in any session with nothing registered first.

        Delegated to :func:`aggregate._program.reins_program`.

        Parameters
        ----------
        cession : str or iterable of str
            One cession clause per tier, each opening with ``occurrence`` or
            ``aggregate``. The clause is authoritative for its own tier and
            leaves the other alone.

        Returns
        -------
        str
            DecL for the reinsured aggregate.

        Examples
        --------
        ::

            a = build('agg X 100 claims sev lognorm 100 cv 2 poisson')
            build(a.reins_program('occurrence net of 500 xs 500'))
        """
        return _program.reins_program(self, cession)

    def _sev_label(self) -> str:
        """Short severity family label for tail text (the family, or ``'N components'``)."""
        if self.sevs is None or len(self.sevs) == 0:
            return ''
        if len(self.sevs) == 1:
            name = getattr(self.sevs[0], 'sev_name', '')
            return name if isinstance(name, str) else 'severity'
        return f'{len(self.sevs)} components'

    @property
    def tail_behavior_df(self) -> pd.DataFrame:
        """The layered tail report -- support and tail class per layer -- as a DataFrame.

        One row per layer, bottom-up -- ``frequency``; one per severity mix
        component (``comp0`` ...); the combined effective ``severity`` (only when
        there is more than one component); and the ``aggregate``. Columns:
        ``min`` / ``max`` **structural support** (``-inf`` / ``inf`` at an
        unbounded end), ``left_tail`` / ``right_tail`` (the per-side tail class --
        ``bounded`` at a finite end, else the family decay rung), ``bounded`` (the
        support finite both ends), and (aggregate row only) the conservative
        ``concentrated`` flag and ``cv = sd / mean``. A
        ``note`` carries power-law ``alpha`` / infinite-moment and capped-base
        annotations.

        Spec-only -- built from the family classifier, the structural support,
        and the pre-computed moments, so it is valid *before* :meth:`update`. The
        numeric grid *reach* lives in the bucket report (``bs_window_df``), not
        here. See :mod:`aggregate.tail` and ``dev/bucket-selection.rst``.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``component``.
        """
        return _tail.tail_frame(self._tail_rows())

    def _tail_rows(self, reference=True):
        """The layered tail report as a list of :class:`~aggregate.tail.TailRow`.

        Single source for :attr:`tail_behavior_df`, :attr:`tail_description`,
        and :attr:`tail_explanation`. Spec-only -- valid before :meth:`update`.

        Parameters
        ----------
        reference : bool, default True
            Whether a severity that came from a ``sev agg.NAME`` reference
            reports the **referenced object's theoretical** tail (the default,
            and what every reporting surface wants) or the tail of the finite
            atoms it materialized to. The bucket sizer asks for ``False``
            through :meth:`_loss_tail_classes`, because the grid is chosen for
            the law that is actually convolved: a certified reference is a
            fully formed ``dsev``, and the outer sizes from it exactly as it
            would from a hand-written one. So the two readings differ on
            purpose. See ``dev/done/plan-agg-port-as-sev.md`` section 4.6.
        """
        freq_min, freq_max, freq_zt = self._frequency_count_support()
        return _tail.build_tail_rows(
            self.frequency, self.sevs,
            freq_min=freq_min, freq_max=freq_max, freq_zero_truncated=freq_zt,
            actual_m=self.actual_m, actual_sd=self.actual_sd,
            occ_reins=self.occ_reins, reference=reference,
        )

    def _frequency_count_support(self):
        """Spec-only ``(min, max, zero_truncated)`` claim-count support.

        The count magnitude lives on the aggregate (the exposure ``n``), not on
        the bare :class:`Frequency`, so it is resolved here. Returns ``(n, n)``
        for a ``fixed`` count, ``(0, 1)`` for ``bernoulli``, the atom range for
        an ``empirical`` count, and ``(0, inf)`` otherwise -- with the third
        element flagging a genuine zero-truncated (``zm``, ``p0 == 0``) count.
        """
        fname = getattr(self.frequency, 'freq_name', '')
        n = float(self.n) if self.n else 0.0
        if fname == 'fixed':
            return n, n, False
        if fname == 'bernoulli':
            return 0.0, 1.0, False
        if fname in ('empirical', 'renewal'):
            atoms = getattr(self.frequency, 'freq_a', None)
            if atoms is not None and len(atoms):
                return float(np.min(atoms)), float(np.max(atoms)), False
        zt = False
        lo = 0.0
        if getattr(self.frequency, 'freq_zm', False):
            p0 = getattr(self.frequency, 'freq_p0', None)
            if p0 is not None and float(p0) == 0.0:
                lo, zt = 1.0, True
        return lo, np.inf, zt

    @property
    def reins_bucket(self) -> str:
        """Rebucketing scheme for reinsurance net/ceded distributions.

        ``'linear'`` (default) splits each off-grid net/ceded value's mass
        across its two bracketing grid buckets, preserving the first moment
        exactly; ``'nearest'`` rounds to the closest bucket (≤ ``bs/2``
        positional bias). Mirrors :meth:`Portfolio.allocation_method`.

        Reinsurance is baked in during :meth:`update`, so a change after
        ``build`` requires a re-``update()`` to take effect. The setter
        clears the cached reinsurance frames so they rebuild on next access.
        """
        return self._reins_bucket

    @reins_bucket.setter
    def reins_bucket(self, value: str) -> None:
        if value not in ('linear', 'nearest'):
            raise ValueError(
                f"reins_bucket must be 'linear' or 'nearest', not {value!r}")
        if value != getattr(self, '_reins_bucket', None):
            self._reins_bucket = value
            self._reins_density_df = None
            self._reins_stats_df = None
            self._reins_view_stats_cache = None
            self._reins_describe = None

    @property
    def dsev_bucket(self) -> str:
        """Scheme for placing discrete-severity atoms onto the model grid.

        Applies to ``dsev`` / ``dhistogram`` / ``fixed`` (point-mass)
        severities; continuous severities are already exact via the
        cdf-difference and ignore this. Sibling of :attr:`reins_bucket`.

        ``'linear'`` (default) splits each off-grid atom's mass across its two
        bracketing grid buckets ``k``, ``k+1`` with weights ``1-f``, ``f``
        (``f = x/bs - k``), so the discretized first moment equals
        ``Σ xₖ pₖ`` **exactly**; ``'nearest'`` snaps each atom to its closest
        bucket (the historical behaviour, with up to ``bs/2`` positional bias).
        On-grid atoms (``f == 0``, e.g. integer atoms with ``bs == 1``) give
        identical results under both schemes.

        Discretization happens in :meth:`update`, so a change after ``build``
        requires a re-``update()`` to take effect. The setter clears the cached
        density frames so they rebuild on next access.

        Notes
        -----
        Phase 1 covers **unlayered** discrete severities (the empirical-sample
        use case). A *layered* discrete severity (``a xs b dsev ...``)
        discretizes via the standard cdf-difference and so behaves as
        ``'nearest'`` regardless of this setting.
        """
        return self._dsev_bucket

    @dsev_bucket.setter
    def dsev_bucket(self, value: str) -> None:
        if value not in ('linear', 'nearest'):
            raise ValueError(
                f"dsev_bucket must be 'linear' or 'nearest', not {value!r}")
        if value != getattr(self, '_dsev_bucket', None):
            self._dsev_bucket = value
            self._density_df = None
            self._sev_density_df = None

    def _severity_in_window(self):
        """Whether the severity support overlaps the output window ``[x_min, x_max]``.

        Returns ``True`` (the common case) when at least one severity bucket
        with positive mass falls inside the aggregate output grid. ``False``
        flags the divergent case (e.g. a high-claim-count aggregate windowed
        far from 0 while the per-claim severity sits near 0), where
        ``density_df.p_sev`` is all zeros and the severity must be read from
        ``sev_density_df`` instead.
        """
        if self.sev_density is None or self.xs_sev is None:
            return True
        nz = self.sev_density > 0
        if not np.any(nz):
            return True
        s_lo = float(self.xs_sev[nz][0])
        s_hi = float(self.xs_sev[nz][-1])
        return not (s_hi < self.x_min or s_lo > self.x_max)

    def _sev_density_on_output_grid(self):
        """Severity density mapped from ``xs_sev`` onto the output grid ``xs``.

        The severity is discretised on ``xs_sev`` (physical 0 at index ``i0``);
        the output grid is ``xs = x_min + bs*arange``. This places each severity
        bucket at its physical location on the output grid, zero where the
        severity falls outside the output window. With no offset (``i0 == 0``
        and ``x_min == 0``) it returns ``self.sev_density`` unchanged.

        Returns
        -------
        np.ndarray
            Severity density aligned to ``self.xs`` (length ``len(xs)``).

        Notes
        -----
        Output index ``k`` (physical ``x_min + k*bs``) corresponds to severity
        index ``k + i0 + j0`` where ``j0 = round(x_min/bs)`` -- because the
        severity bucket ``s`` sits at physical ``(s - i0)*bs`` and we need
        ``(s - i0)*bs == x_min + k*bs``. When the severity is far from the
        output window (e.g. a tight P&L window around a large negative mean) the
        overlap is empty and the result is all zeros -- the severity simply is
        not in the displayed window.
        """
        sev = self.sev_density
        if sev is None:
            return sev
        N = len(self.xs)
        j0 = int(round(self.x_min / self.bs)) if self.bs else 0
        shift = self.i0 + j0
        if shift == 0:
            return sev
        out = np.zeros(N)
        k = np.arange(N)
        src = k + shift
        valid = (src >= 0) & (src < N)
        out[valid] = np.asarray(sev)[src[valid]]
        return out

    @property
    def density(self):
        """The "live" part of :attr:`density_df` — rows with positive total mass.

        Returns ``density_df.query('p_total > 0')``: the actual support of the
        aggregate, dropping the leading and trailing zero-probability buckets of
        the FFT grid. This is usually what you want to *see*. It is recomputed on
        each access (a plain property, not cached) because the underlying frame
        can be rebuilt by ``update``.
        """
        return self.density_df.query('p_total > 0')

    @property
    def density_df(self):
        """Per-bucket density / distribution / risk-measure frame.

        Built lazily on first access after ``update``. Stored in
        ``self._density_df``; treat as read-only.

        Columns (in construction order):

        ================  =====================================  =========================
        Column            Set from                               Read by
        ================  =====================================  =========================
        ``loss``          ``self.xs`` (also the index)           ``plot``, ``q``, bounds
        ``p_total``       ``self.agg_density``                   Portfolio (when Aggregate
                                                                  is in a port), bounds,
                                                                  ``plot``, user code
        ``p``             alias of ``p_total``                   Portfolio API compat
        ``log_p``         ``np.log(p)``                          ``plot`` log scale
        ``F``             ``p.cumsum()``                         ``q``, ``var``, ``tvar``
        ``S``             ``1 - p_total.cumsum()``               ``q``, ``tvar``, ``plot``
        ``lev``           ``cumsum(loss·p) + loss·S``            pricing
        ``exa``           alias of ``lev``                       Portfolio API compat
        ``exlea``         ``cumsum(loss·p) / F``                 pricing
        ``e``             ``self.est_m`` (constant column)       ``exgta``
        ``exgta``         ``(e - cumsum(loss·p)) / S``           pricing
        ``exeqa``         ``loss`` (since ``E[X|X=a] = a``)      Portfolio API compat
        ================  =====================================  =========================

        ``lev``, ``exlea`` and ``exgta`` are direct sums that carry the
        window origin ``x0`` (numerics-2): on a windowed or signed grid
        ``E[X ∧ a] = Σ_{x≤a} x·p + a·S(a)``, never ``cumsum(S)·bs`` (which
        silently assumes the grid starts at 0). Ratio denominators are
        guarded: ``exlea`` (``exgta``) is NaN where ``F`` (``S``) is at or
        below the validation noise floor. The ``epd`` column was removed
        at numerics-2 (no consumers; ``max(0, e - lev)/e`` if needed).

        Duplicated columns (``p == p_total``, ``exa == lev``, ``exeqa == loss``) are
        intentional: Portfolio's ``filter(regex='p_<name>')`` / ``exeqa_*`` /
        ``exa_*`` patterns require these names exist on the unit's frame so the
        unit can be inlined into a portfolio.

        :return: DataFrame indexed by ``loss``, columns as tabulated above.
        """
        if self._density_df is None:
            # really should have one of these anyway...
            if self.agg_density is None:
                raise ValueError('Update Aggregate before asking for density_df')

            # really convenient to have p=p_total to be consistent with Portfolio objects
            self._density_df = pd.DataFrame(dict(loss=np.asarray(self.xs, dtype=float),
                                                 p_total=self.agg_density))
            self._density_df = self._density_df.set_index('loss', drop=False)
            self._density_df['p'] = self._density_df.p_total
            # remove the fuzz, same method as Portfolio.remove_fuzz
            self._density_df = remove_fuzz(self._density_df)

            # Severity columns (p_sev/F_sev/S_sev/log_p_sev) now live on their
            # own native grid in ``sev_density_df`` -- on a windowed/signed grid
            # the severity (near 0) and the aggregate (windowed) no longer share
            # a grid, so forcing the severity onto the aggregate index is at best
            # partial. See ``sev_density_df`` and ``info``'s outside-window note.

            # reindex
            self._density_df = self._density_df.set_index('loss', drop=False)
            # guard log of 0 / fp-fuzz negatives (cosmetic display column only;
            # values unchanged: log(0) = -inf, log(<0) = nan as before).
            with np.errstate(divide='ignore', invalid='ignore'):
                self._density_df['log_p'] = np.log(self._density_df.p)

            # generally acceptable for F, by construction
            self._density_df['F'] = self._density_df.p.cumsum()

            # Update 2021-01-28: S is best computed forwards
            self._density_df['S'] = 1 - self._density_df.p_total.cumsum()

            # LEV and the conditional means by direct sums that carry the
            # window origin (numerics-2): E[X∧a] = Σ_{x≤a} x·p + a·S(a),
            # valid on windowed and signed grids where cumsum(S)·bs is not.
            loss_v = self._density_df['loss'].to_numpy()
            p_v = self._density_df['p_total'].to_numpy()
            F_v = self._density_df['F'].to_numpy()
            S_v = self._density_df['S'].to_numpy()
            cum_xp = np.cumsum(loss_v * p_v)
            self._density_df['lev'] = cum_xp + loss_v * S_v
            self._density_df['exa'] = self._density_df['lev']
            # explicit denominator guards (F/S at or below the validation
            # noise floor cannot support a conditional mean)
            tol = VALIDATION_NOISE
            with np.errstate(divide='ignore', invalid='ignore'):
                exlea = cum_xp / F_v
            exlea[F_v <= tol] = np.nan
            self._density_df['exlea'] = exlea

            # expected value
            self._density_df['e'] = self.est_m  # np.sum(self._density_df.p * self._density_df.loss)
            with np.errstate(divide='ignore', invalid='ignore'):
                exgta = (self.est_m - cum_xp) / S_v
            exgta[S_v <= tol] = np.nan
            self._density_df['exgta'] = exgta
            self._density_df['exeqa'] = self._density_df.loss  # E(X | X=a) = a(!) included for symmetry was exa

        return self._density_df

    @property
    def sev_density_df(self):
        """Per-bucket severity density / distribution on the **severity** grid.

        The severity lives on its own grid ``xs_sev`` (physical 0 at index
        ``i0``), which equals the aggregate grid only on the default 0-based
        case. On a windowed / signed aggregate the severity (near 0) and the
        aggregate (windowed, possibly far from 0) genuinely occupy different
        grids, so the severity reporting columns live here -- indexed by the
        severity's own loss -- rather than in ``density_df``. This frame is
        always correct regardless of the aggregate output window.

        Columns
        -------
        loss : severity grid ``xs_sev`` (also the index).
        p_sev : ``self.sev_density``.
        log_p_sev : ``log(p_sev)`` (``-inf`` at zero buckets).
        F_sev : ``p_sev.cumsum()``.
        S_sev : ``1 - p_sev.cumsum()``.

        Returns
        -------
        DataFrame indexed by severity ``loss``.
        """
        if self._sev_density_df is None:
            if self.sev_density is None:
                raise ValueError('Update Aggregate before asking for sev_density_df')
            df = pd.DataFrame(dict(loss=np.asarray(self.xs_sev, dtype=float),
                                   p_sev=self.sev_density))
            df = df.set_index('loss', drop=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                if df.p_sev.dtype == np.dtype('O'):
                    df['log_p_sev'] = np.nan
                else:
                    df['log_p_sev'] = np.log(df.p_sev)
            df['F_sev'] = df.p_sev.cumsum()
            df['S_sev'] = 1 - df.p_sev.cumsum()
            self._sev_density_df = df
        return self._sev_density_df

    # ================================================================
    # Reinsurance reporting (rationalized; see dev/reins-reporting.md)
    # ================================================================

    @property
    def reins_density_df(self):
        """Per-bucket gross / ceded / net densities under reinsurance.

        One row per model-grid bucket (``loss = k * bs``); the empirical,
        FFT-ready rebucketed densities. Columns are **always present**
        regardless of which stages are configured; a missing stage
        contributes the no-cession values (ceded mass at 0, net = subject).

        Columns
        -------
        loss : float
            The model grid (also the index).
        p_sev_gross, p_sev_ceded, p_sev_net : occurrence-level severity
            views. With no occurrence cover ``p_sev_ceded`` is a point mass
            at 0 and ``p_sev_net == p_sev_gross``.
        p_agg_gross, p_agg_ceded_occ, p_agg_net_occ : aggregate of each
            occurrence severity view. ``p_agg_gross`` is the *true* gross
            aggregate (``_fft_aggregate`` of the gross severity).
        p_agg_subject, p_agg_ceded, p_agg_net : aggregate-cover views.
            ``p_agg_subject`` is the aggregate input to the aggregate cover
            (= aggregate of the requested occurrence output); it equals
            ``p_agg_gross`` only when there is no occurrence cover.

        Notes
        -----
        All aggregate columns route through ``_fft_aggregate`` so the
        zero-risk / fixed-1 shortcuts apply consistently. Renamed from the
        legacy ``reinsurance_df``: ``p_agg_gross_occ -> p_agg_gross`` and the
        old ``p_agg_gross`` (the agg-cover input) ``-> p_agg_subject``.
        Returns ``None`` when no reinsurance is configured.
        """
        return _reinsurance.reins_density_df(self)

    @property
    def reins_views(self):
        """The distributions this aggregate's cession makes available for pricing.

        The accepted values of the ``reins_view=`` keyword on
        :meth:`calibrate_distortions` and :meth:`evaluate`, and the answer to
        "what can I price this on": ``[]`` when nothing cedes,
        ``['gross', 'ceded', 'net']`` with one stage, and those plus
        ``'ceded occ'`` / ``'net occ'`` when a program has both. Mirrors
        :func:`~aggregate.charts.available_charts` and
        :func:`~aggregate.exhibits.available_exhibits`, which answer the same
        shape of question about the same object.

        ``ceded`` and ``net`` are **end to end**, so on an occurrence-only
        program they are the occurrence stage's cession and retention, not the
        empty aggregate-stage columns of :attr:`reins_density_df`.

        Returns
        -------
        list of str

        See Also
        --------
        aggregate._reinsurance.reins_view_columns : the mapping onto columns.
        """
        return list(_reinsurance.reins_view_columns(self))

    def _reins_view_density(self, view):
        """The aggregate density of one named reinsurance view, as a ``Series``.

        The pricing surface's private hook; :attr:`reins_views` is the public
        question. Raises ``ValueError`` for a view this aggregate cannot
        answer, including the no-reinsurance case.
        """
        return _reinsurance.reins_view_density(self, view)

    def reins_price_df(self, distortion=None, *, p=None, a=None, views=None):
        """What a stated risk measure says this cession is worth.

        The fourth ``reins_*`` frame, and the one that crosses over into
        pricing: :attr:`reins_density_df` computes the cession,
        :meth:`~aggregate.spectral.Distortion.price` prices any pmf, and this
        walks one through the other. The DecL ``ceded premium`` clause records
        a price someone agreed; this is the price a distortion implies, and
        the gap between them is the reading worth having.

        See :func:`aggregate._reinsurance.reins_price_df` for the full
        contract. The usual call is after
        :meth:`calibrate_distortions`, with no ``distortion`` argument, which
        prices every view with every calibrated family.

        Parameters
        ----------
        distortion : Distortion, str, dict, or None
            A distortion, a name in :attr:`distortions`, a mapping, or
            ``None`` for the whole calibrated set.
        p : float, optional
            Asset probability; each view resolves its own ``a = q(p)``.
        a : float, optional
            A common asset level. At most one of ``p`` or ``a``; with neither
            the price is unlimited.
        views : sequence of str, optional
            Defaults to all of :attr:`reins_views`.

        Returns
        -------
        pandas.DataFrame
            ``(distortion, view)`` rows by ``a`` / ``el`` / ``bid`` / ``ask``
            / ``margin``.

        Examples
        --------
        ::

            a = build('agg Re 100 claims sev lognorm 50 cv 2 '
                      'occurrence net of 100 xs 100 poisson')
            a.calibrate_distortions(0.10, p=0.999)
            a.reins_price_df().loc[(slice(None), 'ceded'), :]
        """
        return _reinsurance.reins_price_df(self, distortion, p=p, a=a,
                                           views=views)

    def reins_occ_plot(self, log=False, full_range=False, reflect=False,
                       return_period=False, invert=False):
        """Plot the occurrence program: per claim, and in total.

        Parameters
        ----------
        log : bool
            Read the aggregate panel's axes on log. The occurrence panel is
            read on log either way, and offers no other reading.
        full_range : bool
            Show the whole grid rather than the cropped windows.
        reflect : bool
            Read the aggregate panel against the exceeding probability, so
            the three curves are survival functions rather than quantile
            functions of non-exceedance.
        return_period : bool
            Read the aggregate panel against return period rather than
            non-exceedance probability.
        invert : bool
            Exchange the aggregate panel's axes, which draws the
            distribution function the Lee diagram inverts to.

        Returns
        -------
        matplotlib.figure.Figure
            Also stashed on ``self.figure``.

        Notes
        -----
        Draws the document ``charts.chart_reins`` emits. Two panels, and
        they share nothing, not even a loss axis: a per-claim loss and an
        annual aggregate are different quantities, and one window across
        both would say they were the same. The left is the gross, ceded and
        net severity as the treaty sees each claim; the right is the same
        three for the year, as a Lee diagram.

        The three curves are separate distributions and not a
        decomposition: they no more satisfy ``gross = net (+) ceded`` than
        a portfolio's marginals do.
        """
        from .charts import build_chart_doc
        from .plots import plot_chartdoc
        self.figure = plot_chartdoc(
            build_chart_doc(self, 'reins'), log=log, full_range=full_range,
            reflect=reflect, return_period=return_period, invert=invert)
        return self.figure

    def occ_bivariate(self, views=('net', 'ceded'), bs=None,
                      log2_x=None, log2_y=None, total_log2=None,
                      store_dir=None, row_chunk=512, col_chunk=512,
                      keep_transform=False):
        """Joint law of two of the occurrence {gross, ceded, net} aggregates via 2D FFT.

        Computes the *joint* distribution of two aggregate occurrence views --
        e.g. ceded ``C = sum c(X_i)`` and net ``N = sum n(X_i)`` -- where each
        gross claim ``X_i`` is split deterministically by the occurrence cession
        map. The two are **not** deterministic functions of one another -- the
        random claim count decouples them -- so the joint law carries genuine
        information (their correlation, co-moments, reinsurer-vs-cedent
        dependency) beyond the two univariate margins already in
        :attr:`reins_density_df`. The three views satisfy ``ceded + net = gross``,
        so any *two* determine the third; the pick of which two is ``views``.

        Parameters
        ----------
        views : (str, str), default ``('net', 'ceded')``
            The ``(x, y)`` axis view pair, each one of ``'gross'`` / ``'ceded'``
            / ``'net'``. Axis 0 (x) is ``views[0]``, axis 1 (y) is ``views[1]``.
            The default ``('net', 'ceded')`` matches the DecL ``netceded`` form;
            ``('gross', 'ceded')`` matches ``grossceded`` and ``('gross', 'net')``
            matches ``grossnet``.
        bs : float, optional
            Bucket-size override (a single common ``bs`` for both axes).
            Default: the **exact common lattice** when the budget affords it,
            so every per-claim point of the comonotone curve is a cell center
            and the scatter never fires; otherwise one common ``bs`` sized from
            the budget (coarser than the gross bucket, since the gross grid is
            far finer than a 2-D grid affords). Which one was taken reads off
            :attr:`~aggregate.bivariate.BivariateAggregate.bs_explanation`.
        log2_x, log2_y : int, optional
            Axis-0 / axis-1 log2 grid lengths (grid has ``1 << log2`` points).
            Default: measured from the two views' occurrence aggregate margins
            via :func:`~aggregate.utilities.balanced_window`, each axis
            independently, since the two windows of a cession are structurally
            asymmetric (ceded is capped by the cover, gross is not).
        total_log2 : int, optional
            Total 2-D cell budget, ``2**total_log2`` cells. ``None`` uses the
            :attr:`BivariateSettings.total_log2` default (20). Raise it for a
            fine joint, and pair it with ``store_dir`` past the point where the
            joint fits in memory: the two are coupled, since without the first
            no grid is large enough to want the second.
        store_dir : str, optional
            Backing directory for a **massive (disk-backed) build**: the joint
            streams to ``density.zarr`` there and never materializes in RAM.
            Requires the ``massive`` extra.
        row_chunk, col_chunk : int, optional
            Massive-path band and tile sizes (ignored in core).
        keep_transform : bool, optional
            Massive path: keep the staging stores after the build.

        Returns
        -------
        BivariateAggregate
            A first-class joint object in ``netceded`` mode, with the joint
            ``density``, the two axis grids (``axis_xs``), ``marginals`` /
            ``moments`` (``E[X^i Y^j]``) / ``corr`` / ``summary_df`` / ``stats_df``
            / ``info`` and a two-panel ``plot`` (comonotone per-claim severity
            and joint aggregate). Equivalent to the DecL ``netceded`` /
            ``grossceded`` / ``grossnet`` ``<agg>`` prefix forms.

        Raises
        ------
        ValueError
            If the object carries no occurrence reinsurance, has not been
            updated (no severity densities present), or carries a pinned grid
            that cannot be honored inside ``total_log2``. A pinned pair that
            overflows used to clip the wider axis, which is not a tail loss:
            ``occ_bivariate(views=('gross', 'ceded'), bs=0.5)`` on an
            unbounded severity clipped the gross axis to 512 buckets and
            answered off a joint with a deficit of 0.535.

        Notes
        -----
        **Occurrence only.** Any *aggregate* reinsurance on the object is
        ignored: this is the joint law of the per-occurrence ceded / net
        aggregates. (The aggregate-cover bivariate is degenerate -- at the
        aggregate level ceded and net are deterministic functions of the
        aggregate, supported on a curve -- so it is out of scope.)

        **Method.** Per claim, ``(c(X), n(X))`` lies on the line ``c + n = X``.
        Placing the gross severity mass at ``(c(x_k), n(x_k))`` (rebucketed onto
        the 2D grid by the active :attr:`reins_bucket` scheme) builds the
        bivariate severity ``S``. The joint aggregate density is
        ``iFFT2(freq_pgf(n, FFT2(S)))`` -- exactly the univariate
        :meth:`_fft_aggregate` with the 1D transforms replaced by 2D transforms,
        valid because ``freq_pgf(n, z)`` is elementwise in ``z``. The zero-risk
        (``n == 0``) and fixed-count-one shortcuts mirror ``_fft_aggregate``.
        Marginalising the result over one axis recovers the corresponding
        univariate occurrence ceded / net aggregate (exact validation targets;
        see the ``Cross-check`` notes in ``dev/done/reins-bivariate.md``).

        This wraps the object as a :class:`aggregate.bivariate.BivariateAggregate`
        in ``netceded`` mode (the engine is :func:`aggregate.bivariate.build_netceded_joint`).
        """
        from .bivariate import BivariateAggregate

        mv = BivariateAggregate(
            self.name, mode='netceded', nc_agg=self, nc_views=views,
            nc_kwargs=dict(bs=bs, log2_x=log2_x, log2_y=log2_y,
                           total_log2=total_log2))
        # build eagerly so preconditions (occ reins present, object updated)
        # raise here, and the returned object is ready to query.
        mv.update(store_dir=store_dir, row_chunk=row_chunk,
                  col_chunk=col_chunk, keep_transform=keep_transform)
        return mv

    def occ_joint(self, views=('gross', 'ceded'), **sizing):
        """The occurrence joint, built once per sizing and held.

        :meth:`occ_bivariate` behind a memo. Two surfaces want the same joint
        of one object, the natural allocation frame and the kappa chart, and a
        2-D FFT is not something to pay for twice because two callers asked
        the same question (``dev/plan-pricing-natural-allocation.md``,
        decision 6, agreed by the author 2026-08-14).

        Parameters
        ----------
        views : (str, str), default ``('gross', 'ceded')``
            The axis view pair. The default is the pair the allocation and the
            kappa band both condition on, which is why it differs from
            :meth:`occ_bivariate`'s.
        **sizing
            Passed to :meth:`occ_bivariate`, and part of the memo key, so a
            resize is an honest rebuild rather than a stale hit.

        Returns
        -------
        BivariateAggregate

        Notes
        -----
        The cache is cleared by :meth:`update_work`, so a re-updated object
        never answers off a joint built on its old grid. A ``store_dir`` build
        is **not** held: a disk backed joint owns a directory whose lifetime is
        the caller's, and holding a reference to one would quietly keep it
        alive past the point the caller expected to be done with it.
        """
        if sizing.get('store_dir') is not None:
            return self.occ_bivariate(views=views, **sizing)
        key = (tuple(views), tuple(sorted(
            (k, v) for k, v in sizing.items() if v is not None)))
        if key not in self._occ_joints:
            self._occ_joints[key] = self.occ_bivariate(views=views, **sizing)
        return self._occ_joints[key]

    # ----- reinsurance stats: exact (EX) vs rebucketed (Est) -------------

    @staticmethod
    def _reins_moments6_from_raw(e1, e2, e3):
        """``(ex1, ex2, ex3, mean, cv, skew)`` from raw moments."""
        return _reinsurance.reins_moments6_from_raw(e1, e2, e3)

    def _reins_exact_image_raw(self, image_fn, p_subject):
        """Raw moments ``E[g(X)^j]``, ``j=1..3``, of an exact loss image
        ``g = image_fn`` weighted by ``p_subject``: the pre-bucket truth
        ``sum g(xs)^j * p_subject`` with **no** rebucketing scatter.

        Notes
        -----
        Unlike :func:`xsden_to_mwrangler` (the ``Est`` basis), no
        defective-mass tail term is added: that convention places lost mass
        at the implied max loss ``xs[-1] + bs``, which is right for a gross
        aggregate but wrong for a ceded image (where the lost mass maps to the
        capped cession). The EX basis is therefore the literal exact moment of
        the on-grid subject; the EX-vs-Est difference reflects the rebucketing
        scatter (plus, for a gross aggregate carried through an FFT, the grid
        deficit -- negligible on an adequate grid).
        """
        return _reinsurance.reins_exact_image_raw(self, image_fn, p_subject)

    def _reins_agg6_from_sev_raw(self, s1, s2, s3):
        """Compound exact severity raw moments into aggregate
        ``(ex1, ex2, ex3, mean, cv, skew)`` via the frequency."""
        return _reinsurance.reins_agg6_from_sev_raw(self, s1, s2, s3)

    def _reins_density6(self, p):
        """``(ex1, ex2, ex3, mean, cv, skew)`` of a density on the grid."""
        return _reinsurance.reins_density6(self, p)

    @property
    def _reins_view_stats(self):
        """Per-stage reinsurance moments on two bases: exact and rebucketed.

        Internal frame feeding :meth:`reins_summary_df` (which surfaces the
        ``EX`` exact vs ``Est`` rebucketed comparison as its ``Change``
        column). The public per-layer summary is :meth:`reins_stats_df`.
        Shaped like ``stats_df`` but indexed by stage / view / basis instead
        of component columns.

        Rows
        ----
        ``MultiIndex (component, measure)`` with ``component in
        {freq, sev, agg}`` and ``measure in {ex1, ex2, ex3, mean, cv, skew}``.

        Columns
        -------
        ``MultiIndex (stage, view, basis)``:

        * ``stage`` ``occ`` (when ``occ_reins``) and/or ``agg`` (when
          ``agg_reins``);
        * ``view`` ``gross|ceded|net`` for ``occ``; ``subject|ceded|net`` for
          ``agg``;
        * ``basis`` ``EX`` (exact, pre-bucket) or ``Est`` (rebucketed).

        Notes
        -----
        **EX** ("theoretic"). For the occurrence stage the exact severity
        moments are ``E[g(X)^j] = sum g(xs)^j * p_sev_gross`` with ``g`` the
        identity / ``occ_ceder`` / ``occ_netter``; the aggregate row compounds
        those via the frequency. For the aggregate stage the exact aggregate
        moments are ``E[g(S)^j] = sum g(xs)^j * p_agg_subject`` with ``g`` the
        identity / ``agg_ceder`` / ``agg_netter`` (no compounding -- the cover
        acts on the aggregate directly). Frequency: the ``EX`` reference is the
        gross full moments for every view (the count is unchanged by occurrence
        reinsurance). On the ``Est`` (model-output) basis the gross view is left
        ``NaN`` (mirroring ``validation_df``, which never re-estimates the input
        frequency); occ ceded / net carry the *unconditional* mean ``E[N]`` only
        (so ``freq * sev == agg`` per view), cv / skew ``NaN``. The aggregate
        stage has no sev rows and a degenerate freq row (all ``NaN``).
        Columns are ordered occ before agg, views gross/subject, ceded, net,
        and ``EX`` before ``Est``.

        **Est** ("empirical") reads the rebucketed densities from
        :meth:`reins_density_df` through ``xsden_to_mwrangler``. The
        difference EX vs Est isolates the per-stage ``reins_bucket``
        rebucketing error (``linear`` preserves the mean exactly; ``nearest``
        biases it by at most ``bs/2``).

        Lazily built; invalidated by the ``reins_bucket`` setter and on
        ``update``. Returns ``None`` when no reinsurance is configured.
        """
        return _reinsurance.reins_view_stats(self)

    @property
    def reins_stats_df(self):
        """Per-layer reinsurance layering summary (empirical, model-grid).

        A layering analysis with one column per reinsurance layer plus the
        gross book and the ceded / net totals.

        Columns
        -------
        ``MultiIndex (view, layer)`` -- ``view`` is ``occ`` / ``agg``:

        * occurrence: ``Gross`` (the gross book -- always present), then, when
          ``occ_reins``, ``layer.1`` ... ``layer.k`` (one per layer) and
          ``Ceded`` / ``Net`` totals.
        * aggregate (when ``agg_reins``): ``layer.1`` ... ``layer.m``,
          ``Ceded`` / ``Net``. There is **no aggregate ``Subject`` column** --
          the subject (the occurrence output the aggregate cover applies to)
          is the column flagged ``('meta', 'output') == 1`` in the occurrence
          block, or ``Gross`` when there is no occurrence program.

        Rows
        ----
        ``MultiIndex (component, measure)``:

        * ``meta``: ``share`` (proportion covered), ``limit``, ``attach``,
          ``pr_attach`` and ``pr_detach`` -- the **ground-up exposure
          probabilities** that the underlying loss attaches / fully exhausts
          the view, ``P(X > exp_attach + view_attach)`` and ``P(X > exp_attach
          + view_attach + view_limit)``. These come from the **underlying**
          severity (``self.sevs[i].fz``), not the modeled ``sev_density``,
          which is conditional (claims to the policy layer) and would report 0
          at the policy cap. ``pr_detach`` is ``NaN`` for unlimited layers /
          net totals. ``pr_loss`` (``P(aggregate > 0)`` from the column's
          aggregate density), ``lol`` (loss on line = expected layer aggregate
          loss / placed limit), and ``output`` (``0/1`` flag marking each
          stage's output view -- two 1s for an occ+agg program; ``Gross``
          carries the 1 when there is no occurrence program). ``Gross`` carries
          the claim-count-weighted policy ``limit`` / ``attach`` (``share`` 1);
          the occurrence ``Ceded`` total carries the share-placed sum of layer
          limits and the minimum attachment. (The layer ``freq`` ``n'`` uses
          the *conditional* ``P(subject > attach | policy loss)`` -- the model
          count is claims to the policy -- a separate basis from the absolute
          ``pr_attach``.)
        * ``('freq'|'sev'|'agg', ex1|ex2|ex3|mean|cv|skew)`` -- moments
          (``ex1`` duplicates ``mean`` for easy ``filter(regex=...)`` access).

        Notes
        -----
        **Occurrence layers are conditional** on a loss reaching the layer:
        the frequency is the expected penetrating count ``n' = E[N] * P(X >
        attach)`` and the severity is the unconditional layer severity divided
        by ``P(X > attach)`` (conditional given attach), which leaves the layer
        aggregate mean ``n' * sev`` equal to the unconditional ``E[N] *
        E[ceded]``. The ``agg`` row is the column's actual aggregate
        distribution (FFT of the unconditional layer ceded severity), so its
        higher moments and ``pr_loss`` are exact.

        **The occurrence ``Ceded`` / ``Net`` columns are unconditional**
        totals: the same claim count as ``Gross`` and unconditional
        severities, so ``Ceded`` sev + ``Net`` sev == ``Gross`` sev. Aggregate
        layer means sum to the ``Ceded`` aggregate mean.

        **The aggregate block leaves ``freq`` and ``sev`` all ``NaN``** -- a
        cover on the aggregate has no per-claim frequency / severity that
        combine in the usual way.

        Lazily built; invalidated by the ``reins_bucket`` setter and on
        ``update``. Returns ``None`` when no reinsurance is configured.
        """
        return _reinsurance.reins_stats_df(self)

    @property
    def reins_summary_df(self):
        """Per-stage reinsurance summary -- the daily driver.

        Mirrors the **economic view** of :attr:`validation_df`: compare the theoretic
        reference (the leading view -- ``Gross`` for occurrence, ``Subject`` for
        aggregate) against the model output of each view. One block per
        applicable stage; each block is a ``view x component`` table sharing the
        **same eight columns as** :attr:`validation_df`:

        * ``EX`` / ``Est EX`` / ``Change EX`` -- the **theoretic reference** mean
          (constant down each component), the per-view model-output mean, and
          ``(Est - reference) / reference``;
        * ``CV`` / ``Est CV`` / ``Change CV`` -- the same three for the CV;
        * ``Sk`` / ``Est Sk`` -- reference and model-output skew (no change
          column).

        ``Change`` carries two readings off one arithmetic: on the leading
        (Gross / Subject) row the reference and the model output are the same
        view, so it is the **numerical validation / rebucketing error** (~0 under
        ``linear``); on the ceded / net rows it is the **% impact of the
        cession** on that moment. This is the per-view, per-component analogue of
        the single ``Change`` column in :attr:`validation_df`.

        Layout (per the gross/subject convention; ``view`` and ``component``
        labels are lower-case to match the other frames):

        * **Occurrence block** -- leads with ``gross`` (the reference): rows
          ``(gross|ceded|net) x (freq|sev|agg)``.
        * **Aggregate block** -- leads with ``subject`` (the reference): rows
          ``(subject|ceded|net) x (agg)`` (sev not applicable; freq
          degenerate -> ``NaN``).

        Frequency is reported on the ``Est`` (model-output) basis
        *unconditionally* (mean ``E[N]`` only, so ``freq * sev == agg`` within a
        view; cv / skew ``NaN``) -- consistent with :meth:`reins_stats_df`. The
        leading ``gross`` row's ``Est`` frequency is left ``NaN`` to mirror
        :attr:`validation_df` exactly. (Only the per-layer ``layer.k`` columns of
        :meth:`reins_stats_df` are *conditional*; ``reins_summary_df`` is always
        unconditional.)

        Index is ``MultiIndex (stage, view, component)``. Derived from
        :meth:`reins_stats_df` / the internal view-stats frame. Returns ``None``
        when no reinsurance is configured.
        """
        return _reinsurance.reins_summary_df(self)

    def _reins_describe_block(self, stage, views, comps):
        """One :meth:`reins_summary_df` block: theoretic reference vs model output
        by view x component, mirroring the eight-column :attr:`validation_df` layout.

        The ``EX`` / ``CV`` / ``Sk`` columns hold the **theoretic reference** --
        the leading view's exact (pre-bucket) moments: ``Gross`` for the
        occurrence block, ``Subject`` for the aggregate block. They are therefore
        constant down each component (the same reference is compared against
        every view). ``Est *`` is the per-view model output (the rebucketed,
        model-grid moment). ``Change`` is ``(Est - reference) / reference``: on
        the leading (Gross/Subject) row it degenerates to the rebucketing /
        validation error (~0 under ``linear``); on the ceded / net rows it reads
        as the % impact of the cession on that moment.
        """
        return _reinsurance.reins_describe_block(self, stage, views, comps)

    def rescale(self, scale, kind='homog'):
        """
        Return a rescaled Aggregate object - used to compute derivatives.

        All need to be safe multiplies because of array specification there is an array that is not a numpy array

        TODO have parser return numpy arrays not lists!

        :param scale:  amount of scale
        :param kind:  homog of inhomog

        :return:
        """
        spec = self._spec.copy()

        def safe_scale(sc, x):
            """
            if x is a list wrap it

            :param x:
            :param sc:
            :return: sc x
            """

            if type(x) == list:
                return sc * np.array(x)
            else:
                return sc * x

        nm = spec['name']
        spec['name'] = f'{nm}:{kind}:{scale}'
        if kind == 'homog':
            # do NOT scale en... that is inhomog
            # do scale EL etc. to keep the count the same
            spec['exp_el'] = safe_scale(scale, spec['exp_el'])
            spec['exp_premium'] = safe_scale(scale, spec['exp_premium'])
            spec['exp_attachment'] = safe_scale(scale, spec['exp_attachment'])
            spec['exp_limit'] = safe_scale(scale, spec['exp_limit'])
            spec['sev_loc'] = safe_scale(scale, spec['sev_loc'])
            # note: scaling the scale takes care of the mean, so do not double count
            # default is 0. Can't ask if array is...but if array have to deal with it
            if (type(spec['sev_scale']) not in (int, float)) or spec['sev_scale']:
                spec['sev_scale'] = safe_scale(scale, spec['sev_scale'])
            else:
                spec['sev_mean'] = safe_scale(scale, spec['sev_mean'])
            if spec['sev_xs']:
                spec['sev_xs'] = safe_scale(scale, spec['sev_xs'])
        elif kind == 'inhomog':
            # just scale up the volume, including en
            spec['exp_el'] = safe_scale(scale, spec['exp_el'])
            spec['exp_premium'] = safe_scale(scale, spec['exp_premium'])
            spec['exp_en'] = safe_scale(scale, spec['exp_en'])
        else:
            raise ValueError(f'Inadmissible option {kind} passed to rescale, kind should be homog or inhomog.')
        return Aggregate(**spec)

    # ================================================================
    # Construction: __init__ and its component-recording helper
    # ================================================================

    def __init__(self, name, exp_el=0.0, exp_premium=0.0, exp_lr=0.0, exp_en=0.0, exp_attachment=None, exp_limit=np.inf,
                 sev_name='', sev_a=np.nan, sev_b=0.0, sev_mean=0.0, sev_cv=0.0, sev_loc=0.0, sev_scale=0.0,
                 sev_xs=None, sev_ps=None, sev_wt=1.0, sev_lb=0.0, sev_ub=np.inf, sev_conditional=True,
                 sev_signed=False, sev_reflect=False,
                 sev_pick_attachments=None, sev_pick_losses=None,
                 occ_reins=None, occ_kind='', occ_reins_label=None,
                 freq_name='', freq_a=0.0, freq_b=0.0, freq_zm=False, freq_p0=np.nan,
                 freq_pin_mean=True,
                 exp_years=0.0, exp_rate=0.0,
                 wait_name='', wait_a=np.nan, wait_b=0.0, wait_mean=0.0, wait_cv=0.0,
                 wait_loc=0.0, wait_scale=0.0, wait_xs=None, wait_ps=None,
                 wait_wt=1.0, wait_lb=0.0, wait_ub=np.inf, wait_conditional=True,
                 wait_attachment=None, wait_limit=np.inf,
                 agg_reins=None, agg_kind='', agg_reins_label=None,
                 reins_bucket=None, dsev_bucket=None,
                 value_type='loss',
                 approximate='exact',
                 label=None, label_map=None,
                 note='', hints='', tags=(), _tweedie=None):
        """
        The :class:`Aggregate` distribution class manages creation and calculation of aggregate distributions.
        It allows for very flexible creation of Aggregate distributions. Severity
        can express a limit profile, a mixed severity or both. Mixed frequency types share
        a mixing distribution across all broadcast terms to ensure an appropriate inter-
        class correlation.

        :param name:            name of the aggregate
        :param exp_el:          expected loss or vector
        :param exp_premium:     premium volume or vector. With ``exp_lr`` it sizes
                                the expected loss (``exp_el = exp_premium * exp_lr``);
                                alone, alongside a ``claims`` or ``loss`` sizing
                                head, it is informational (the DecL FYI premium,
                                ``5 claims 20000 premium``): the loss ratio
                                back-fills from the realized expected loss and
                                the law is untouched.
        :param exp_lr:          loss ratio or vector  (requires premium)
        :param exp_en:          expected claim count per segment (self.n = total claim count)
        :param exp_attachment:  occurrence attachment; None indicates no limit clause, which is treated different
                                from an attachment of zero.
        :param exp_limit:       occurrence limit
        :param sev_name:        severity name or sev.BUILTIN_SEV or meta.var agg or port or similar or vector or matrix
        :param sev_a:           scipy stats shape parameter
        :param sev_b:           scipy stats shape parameter
        :param sev_mean:        average (unlimited) severity
        :param sev_cv:          unlimited severity coefficient of variation
        :param sev_loc:         scipy stats location parameter
        :param sev_scale:       scipy stats scale parameter
        :param sev_xs:          xs and ps must be provided if sev_name is (c|d)histogram, xs are the bucket break points
        :param sev_ps:          ps are the probability densities within each bucket; if buckets equal size no adjustments needed
        :param sev_wt:          weight for mixed distribution
        :param sev_lb:          lower bound for severity (length of sev_lb must equal length of sev_ub and weights)
        :param sev_ub:          upper bound for severity
        :param sev_conditional: if True, severity is conditional, else unconditional.
        :param sev_signed:      if True the severity is signed (never clamps its
                                negative support; a profit is a negative loss).
                                Set by the ``ssev`` DecL keyword. Orthogonal to
                                ``value_type``. ``dsev`` with a negative atom
                                self-signs regardless of this flag.
        :param sev_pick_attachments:  if not None, a list of attachment points to define picks
        :param sev_pick_losses:  if not None, a list of losses by layer
        :param occ_reins:       layers: share po layer xs attach or XXXX
        :param occ_kind:        ceded to or net of
        :param freq_name:       name of frequency distribution
        :param freq_a:          cv of freq dist mixing distribution
        :param freq_b:          claims per occurrence (delaporte or sig), scale of beta or lambda (Sichel)
        :param freq_zm:         True/False zero modified flag
        :param freq_p0:         if freq_zm, provides the modified value of p0; default is nan
        :param freq_pin_mean:   True (default): solve for the base mean whose realized
                                E[N] matches the exposure clause, so ``n claims`` means
                                n claims. False, the DecL ``!`` after ``zm`` / ``zt``:
                                the clause sets the **un-modified (base)** mean and the
                                (a, b, 1) reweighting shifts it, so the realized E[N] is
                                an output, the textbook parameterization. Ignored unless
                                ``freq_zm``. The default changed at 1.0.0a325
                                ([ZT-ZM-Recalibrate-Default]); before that it was False
                                and the DecL marker meant the opposite.
        :param exp_years:       renewal horizon T (the DecL ``T years`` exposure);
                                requires ``freq_name='renewal'`` and a ``wait_*``
                                law (strict pairing). The claim count is the
                                Sparre-Andersen renewal count N(T) of the wait law.
        :param exp_rate:        informational premium rate (``at NUMBER rate``):
                                sets ``exp_premium = exp_years * exp_rate``; never
                                sizes the count. Stored for writer round-trip.
        :param wait_name:       waiting-time distribution name; the ``wait_*``
                                family mirrors ``sev_*`` exactly (the wait clause
                                reuses the severity mini-language on the time axis)
        :param wait_a:          wait shape parameter (as ``sev_a``)
        :param wait_b:          wait shape parameter (as ``sev_b``)
        :param wait_mean:       wait mean (as ``sev_mean``)
        :param wait_cv:         wait cv (as ``sev_cv``)
        :param wait_loc:        wait location (as ``sev_loc``)
        :param wait_scale:      wait scale (as ``sev_scale``)
        :param wait_xs:         ``dwait`` outcomes (as ``sev_xs``)
        :param wait_ps:         ``dwait`` probabilities; may sum to q < 1 for a
                                defective (terminating) renewal process (``dwait
                                ... !``)
        :param wait_wt:         wait mixture weight (as ``sev_wt``)
        :param wait_lb:         wait splice lower bound (as ``sev_lb``)
        :param wait_ub:         wait splice upper bound (as ``sev_ub``)
        :param wait_conditional: as ``sev_conditional``; False = defective splice,
                                or the unconditional layer when a wait layer is set
        :param wait_attachment: wait layer attachment ``a`` (DecL ``wait y xs a``);
                                as ``exp_attachment`` on the wait law -- conditional
                                ``(W - a | W > a)`` by default, ``min((W - a)+, y)``
                                with ``wait_conditional=False`` (zero waits cluster).
                                Cannot combine with a splice window
        :param wait_limit:      wait layer limit ``y`` -- caps the (layered) wait,
                                an atom at ``y`` of the escaping mass
                                window (mass outside ``[lb, ub]`` terminates the
                                process rather than renormalizing)
        :param agg_reins:       layers
        :param agg_kind:        ceded to or net of
        :param value_type:      ``'loss'`` (default) or ``'payoff'``; the DecL
                                orientation suffix sets ``'payoff'`` (more is
                                better) for an asset-return / direct-payoff
                                primitive. Inert for the distribution, consumed
                                at the pricing layer (the dual distortion). A
                                premium-minus-loss position is the separate
                                :class:`PnL` veneer (see ``make_pnl``), not a
                                value on the aggregate.
        :param approximate:     ``'exact'`` (default), ``'norm'``, ``'lognorm'``,
                                ``'gamma'``, ``'sgamma'`` or ``'slognorm'``.
                                When not ``'exact'``, the freq x sev convolution is
                                replaced at construction by a single continuous
                                severity fitted to the aggregate's moments (method
                                of moments), carried on a fixed frequency of 1.
                                The shifted families ``sgamma`` / ``slognorm``
                                match mean, cv and skew (normal as the symmetric
                                limit, a reflected fit for left skew); ``lognorm``
                                and ``gamma`` match mean and cv only (the declared
                                aggregate's skew is not reproduced); ``norm``
                                matches mean and cv with zero skew. The fitted
                                severity's signedness mirrors the input's: an
                                unsigned (plain ``sev``) input clamps any fitted
                                sub-zero mass to an atom at 0, and the moment
                                drift shows in ``validation_df``. The original
                                program is preserved in
                                ``note``. Incompatible with occurrence reinsurance
                                (which acts pre-convolution); aggregate reinsurance
                                rides along unchanged. Set by the ``approximate``
                                DecL keyword. The chosen kind is exposed on the
                                instance as the **attribute** ``self.approximation``
                                (``''`` when exact, else the kind) -- a distinct name
                                from the ``approximate()`` *method* (the MoM-surrogate
                                factory) so the two do not collide. See
                                dev/done/plan-approximate.md.
        :param label:   optional human display label (the DecL ``as
                                "..."`` clause); ``None`` falls back to ``name``.
                                Presentation only -- repr / exhibit titles prefer
                                it over ``name`` -- never an identity / reference
                                target. See dev/plan-decl-labels.md.
        :param occ_reins_label: optional per-occurrence-layer display labels (the
                                reins-clause ``as`` clause); a list parallel to
                                ``occ_reins`` (entries ``None`` where unlabeled), or
                                ``None``. Pooled into ``label_map['occ_reins']`` as
                                a sparse ``{layer_index: label}`` dict (read via
                                ``a.labels.occ_reins``); names the cession group /
                                leg rows in a P&L ledger.
        :param agg_reins_label: optional per-aggregate-layer display labels,
                                parallel to ``agg_reins`` (see ``occ_reins_label``;
                                pooled into ``label_map['agg_reins']``).
        :param note:            free-text note, from a ``note{...}`` clause
        :param hints:           raw ``hints{...}`` build-settings string
            (``key=value;`` form). Pure annotation here; the underwriter
            parses it on build (caller-supplied ``build()`` kwargs win).
        :param _tweedie:        private provenance, a
            :class:`aggregate.tweedie.TweedieParameters` or ``None``. Set only
            by the parser, when a ``tweedie`` clause (rather than its
            compound-Poisson-gamma expansion) is what the author wrote; read
            only by ``decl_writer`` and :meth:`as_tweedie`. Not part of the
            public constructor contract. It is an argument rather than an
            underwriter-applied stamp because a parsed spec is splatted
            straight into this constructor at eleven call sites and there is no
            choke point to pop it at. See dev/done/plan-tweedie.md.
        """

        # have to be ready for inputs to be in a list, e.g. comes that way from Pandas via Excel
        def get_value(v):
            if isinstance(v, list):
                return v[0]
            else:
                return v

        # class variables
        self.name = get_value(name)
        # for persistence, save the raw called spec via inspect; must call before
        # creating any other local variables.
        frame = inspect.currentframe()
        self._spec = dict(inspect.getargvalues(frame).locals)
        for n in ['frame', 'get_value', 'self']:
            if n in self._spec: self._spec.pop(n)
        # ``_tweedie`` is dropped when unset so adding it did not change
        # ``_spec_hash`` for every object in the library. A real Tweedie keeps
        # it, so ``Aggregate(**a.spec)`` preserves the provenance.
        if self._spec.get('_tweedie') is None:
            self._spec.pop('_tweedie', None)
        self._tweedie = _tweedie

        # Method-of-moments approximation (the ``approximate`` DecL keyword). When
        # not ``'exact'`` the requested freq x sev aggregate is replaced, right
        # here at construction, by a single continuous severity fitted to its
        # first three moments and carried on a fixed frequency of 1. The fit needs
        # the theoretical aggregate moments, so build a throwaway exact copy
        # (analytic moments only -- no FFT) to read them, then rewrite the local
        # construction variables before the frequency/severity setup below. The
        # original spec was just captured into ``self._spec`` (so the object still
        # round-trips), and any ``pnl`` affine / aggregate reinsurance rides along
        # on the rewritten object unchanged. See dev/done/plan-approximate.md.
        #
        # Stored as ``self.approximation`` -- a *noun* attribute -- deliberately
        # NOT ``self.approximate``, which would shadow the ``approximate()``
        # method (the MoM-surrogate factory, the parity-partner of
        # ``Portfolio.approximate``) on every instance. Falsey (``''``) for an
        # exact freq x sev convolution, else the fit kind, so ``if
        # a.approximation:`` reads as "is this object a moment-match surrogate?".
        # The DecL keyword and the ``approximate=`` kwarg / spec key are
        # unchanged; ``_spec`` maps ``'' -> 'exact'`` on the way out (round-trip).
        self.approximation = '' if approximate == 'exact' else approximate
        # Structured record of the method-of-moments fit, populated below when
        # ``approximate != 'exact'``. Kept here (not folded into ``note``) so the
        # human-readable description can be composed *lazily* -- the original
        # ``program`` text is set by the build path only after ``__init__``
        # returns, so it is not yet visible at construction. See
        # ``_approx_description`` and the note finalisation in ``Underwriter``.
        self._approx_fit = None
        if approximate not in ('exact', 'norm', 'lognorm', 'gamma',
                               'sgamma', 'slognorm'):
            raise ValueError(
                f"approximate must be 'exact', 'norm', 'lognorm', 'gamma', "
                f"'sgamma' or 'slognorm', not {approximate!r}")
        if approximate != 'exact':
            if occ_reins is not None:
                raise ValueError(
                    f"{self.name}: approximate is incompatible with occurrence "
                    "reinsurance (the method-of-moments fit bypasses the "
                    "per-occurrence convolution); use aggregate reinsurance instead.")
            _orig = Aggregate(**{**self._spec, 'approximate': 'exact'})
            _m, _cv, _sk = _orig.actual_m, _orig.actual_cv, _orig.actual_skew
            _fit = _approximate_sev_kwargs(_m, _cv, _sk, approximate)
            # frequency -> fixed 1; exposure -> a single deterministic claim
            freq_name, freq_a, freq_b, freq_zm, freq_p0 = 'fixed', 0.0, 0.0, False, np.nan
            exp_en, exp_el, exp_premium, exp_lr = 1, 0.0, 0.0, 0.0
            exp_attachment, exp_limit = None, np.inf
            occ_reins, occ_kind = None, ''
            # severity -> the fitted continuous distribution
            sev_name = _fit['sev_name']
            sev_a = _fit.get('sev_a', np.nan)
            sev_b = 0.0
            sev_mean, sev_cv = 0.0, 0.0
            # the unshifted lognorm / gamma fits carry no loc
            sev_loc = _fit.get('sev_loc', 0.0)
            sev_scale = _fit['sev_scale']
            sev_xs = sev_ps = None
            sev_wt = 1.0
            # The emitted severity type mirrors the INPUT's signedness, not the
            # fit's (author ruling 2026-09-04, dev/done/plan-approximate-punchup.md):
            # a loss aggregate stays a loss, so a fit reaching below zero under
            # an unsigned input clamps its sub-zero tail to an atom at 0. The
            # moment drift the clamp introduces is displayed, not hidden, in
            # validation_df and approximation_df.
            sev_signed = _orig._signed_severity()
            sev_reflect = bool(_fit.get('sev_reflect', False))
            # Record the fit for a lazy, program-aware description (rendered in
            # ``info`` and folded into ``note`` once ``self.program`` is set).
            # ``note`` is deliberately left as the user's pure note here.
            self._approx_fit = dict(
                kind=approximate, sev_name=sev_name, sev_a=sev_a,
                sev_loc=sev_loc, sev_scale=sev_scale, m=_m, cv=_cv, skew=_sk)

        logger.debug(
            f'Aggregate.__init__ | creating new Aggregate {self.name}')
        # Composition: an Aggregate *has* a frequency model, not *is* one.
        # ``Frequency(...)`` dispatches via ``__new__`` to the correct
        # ``Frequency<Kind>`` subclass. A renewal frequency is constructed
        # DIRECTLY (never via factory dispatch) from the wait-law payload.
        if get_value(freq_name) == 'renewal':
            # strict pairing (the grammar enforces this for DecL; this guards
            # programmatic construction)
            if not exp_years or exp_years <= 0:
                raise ValueError(
                    f'{self.name}: a renewal frequency requires exp_years > 0 '
                    f'(the DecL "T years" exposure head)')
            if wait_name == '' or wait_name is None:
                raise ValueError(
                    f'{self.name}: a renewal frequency requires a wait law '
                    f'(the DecL wait/dwait clause)')
            self.frequency = self._build_renewal_frequency(
                exp_years, wait_name, wait_a, wait_b, wait_mean, wait_cv,
                wait_loc, wait_scale, wait_xs, wait_ps, wait_wt, wait_lb,
                wait_ub, wait_conditional, wait_attachment, wait_limit)
        else:
            # the MoM-approximate path rewrites freq_name to 'fixed' but the
            # wait/years locals survive -- exempt it from strict pairing
            if (not getattr(self, '_approx_fit', None)
                    and (exp_years or wait_name)):
                raise ValueError(
                    f'{self.name}: years/wait inputs require '
                    f"freq_name='renewal' (strict pairing), got "
                    f'{get_value(freq_name)!r}')
            self.frequency = Frequency(
                get_value(freq_name), get_value(freq_a), get_value(freq_b),
                get_value(freq_zm), get_value(freq_p0))
        # DecL ``!`` after zm / zt: pin the realized mean to the exposure clause
        # rather than letting the (a, b, 1) reweighting shift it. Only meaningful
        # when the frequency is actually zero modified.
        self._freq_zm = bool(getattr(self.frequency, 'freq_zm', False))
        self._freq_pin_mean = self._freq_zm and bool(get_value(freq_pin_mean))
        # set by the broadcast loops when a component's claim count was derived
        # from a *monetary* exposure clause (loss, premium x lr, exposure x rate)
        # rather than ``n claims``; drives the ZeroModifiedExposureWarning below.
        # ``_zm_requested_loss`` accumulates the money target actually asked for,
        # so the warning can quote it against what was delivered.
        self._zm_monetary_exposure = False
        self._zm_requested_loss = 0.0
        # Spec pass through from constructor arguments
        self.note = note
        #: Tag slugs from the DecL ``tags{...}`` trailer, for grouping and
        #: selection (``Underwriter.discover(tags=...)``). Empty tuple if none.
        self.tags = tuple(tags)
        # Exposure premium / loss ratio, retained for the P&L path: a ``pnl``
        # wrapping this engine reads ``exp_premium`` as the *technical* premium
        # (``inherit premium``), and a Portfolio accumulates it across units. 0.0
        # for a claims / loss exposure or an approximated aggregate (no premium).
        # Held as passed (scalar or per-component vector); consumers reduce with
        # ``np.sum``. See dev/plan-pnl-engine-source.md.
        self.exp_premium = exp_premium
        self.exp_lr = exp_lr
        # Renewal exposure: the horizon T and the informational premium rate
        # (``T years at r rate`` books exp_premium = T*r; the count comes
        # solely from the wait law). 0.0 for every other exposure form.
        self.exp_years = exp_years
        self.exp_rate = exp_rate
        # Raw `hints{...}` settings string; consumed by the underwriter build
        # path (caller-wins merge), retained here for round-tripping / repr.
        self.hints = hints
        self.program = ''  # can be set externally
        self.occ_reins = occ_reins
        self.occ_kind = occ_kind
        self.agg_reins = agg_reins
        self.agg_kind = agg_kind
        # Object-level display label + interior label_map (exposure / layer /
        # severity clause / cessions). Presentation only -- repr / exhibit titles
        # and the exhibit ``renamer`` read these; ``name`` stays the identity /
        # reference handle. See dev/done/plan-labels.md ([DecL-Labels-Everywhere]).
        self._init_labels(label=label, label_map=label_map)
        # The per-layer cession labels arrive as spec lists parallel to
        # ``occ_reins`` / ``agg_reins`` (the unparser round-trips those keys)
        # but POOL into ``label_map`` as sparse ``{layer_index: label}`` dicts
        # -- read ``a.labels.occ_reins[0]`` -- so ``labels`` is the complete
        # interior-label surface: one home, no parallel label attributes.
        for site, layer_labels in (('occ_reins', occ_reins_label),
                                   ('agg_reins', agg_reins_label)):
            if layer_labels:
                d = {i: lbl for i, lbl in enumerate(layer_labels)
                     if lbl is not None}
                if d:
                    self.label_map[site] = d
        self.sev_pick_attachments = sev_pick_attachments
        self.sev_pick_losses = sev_pick_losses

        # Grid + runtime config (set by update / update_work)
        self.figure = None
        self.xs = None
        self.bs = 0
        self.log2 = 0
        self.padding = 0
        # Signed-support / output-window state (set by update_work). The
        # defaults reproduce the non-negative, zero-based grid exactly:
        #   x_min == 0   -> output window starts at 0 (no output roll)
        #   i0 == 0      -> severity has no negative buckets (no input roll)
        #   xs_sev == xs -> severity and output share the grid
        # See dev/plan-negative-x-agg.md (F1 negative-x, F2 output window).
        self.x_min = 0.0
        self.x_max = None
        self.i0 = 0           # index of physical 0 in the severity array
        self.xs_sev = None    # severity discretisation grid (may differ from xs)
        self._bs_window_df = None   # inspectable bucket/window estimator summary
        self._sharpen_df = None     # last sharpen() probe, one row per cell
        self._sharpen_state = None  # last sharpen() decision, for the narrative
        self._bs_clip = None        # structured far-tail clip report (item 6) or None
        self._bs_snap = None        # reference-lattice bs snap report, or None
        self._bs_feasibility = None  # severity-vs-grid feasibility reading, or None
        self._bs_raw = None         # pre-dyadic-round bs (unset for the multi-method agg sizer)
        # F1 opt-in: when True the severity keeps its negative support (the
        # layering clamp ``x<0 -> 0`` is bypassed). Default False preserves the
        # established non-negative behaviour. Opt-in wiring is pending a design
        # decision (DecL keyword vs flag vs dsev auto-detect).
        self._signed_sev = False
        # Sign convention: how the variable is read. Inert for the
        # distribution itself; consumed at the pricing/distortion layer
        # (actuarial loss orientation). ``pnl`` sets ``payoff``. See plan §5.5.
        # Canonical role boolean; the label string is resolved at display time
        # (value_type_label) so a [labels] config relabel never moves the role.
        self._is_loss_value = value_type_role(value_type)
        self.validation_eps = get_settings().validation.eps
        self.sev_calc = ""
        self.discretization_calc = ""
        self.normalize = ""

        # Exposure / mixture outputs (filled by broadcasting below)
        self.en = None   # per-component frequency (e.g. for a limit profile)
        self.n = 0       # total frequency
        self.attachment = None
        self.limit = None
        self.sevs = None

        # Computed densities (set by update_work)
        self.sev_density = None
        self.agg_density = None
        self.ftagg_density = None
        self.fzapprox = None
        self._density_df = None
        self._sev_density_df = None

        # Empirical moment estimates (set by update_work; consumed by q / tvar)
        self.est_m = 0
        self.est_cv = 0
        self.est_sd = 0
        self.est_var = 0
        self.est_skew = 0
        self.est_sev_m = 0
        self.est_sev_cv = 0
        self.est_sev_sd = 0
        self.est_sev_var = 0
        self.est_sev_skew = 0

        # Cached lazy functions (built on demand)
        self._valid = None
        self._deficit = np.nan
        # GridDistribution views over the aggregate and severity grids; own the
        # var/tvar kernel cache. Rebuilt (set None -> lazily) when update runs.
        self._dist = None
        self._sev_dist = None
        self._cdf = None
        self._pdf = None
        self._sev = None

        # Reinsurance state (set by apply_occ_reins / apply_agg_reins).
        # The exact (EX) reporting path reads the ceder/netter step
        # functions retained here; the per-stage reins frames
        # (``reins_density_df``, ``reins_stats_df``, ``reins_summary_df``)
        # are rebuilt lazily and cached in the underscore members below.
        self.occ_netter = None
        self.occ_ceder = None
        self.agg_netter = None
        self.agg_ceder = None
        self.sev_density_ceded = None
        self.sev_density_net = None
        self.sev_density_gross = None
        self.agg_density_ceded = None
        self.agg_density_net = None
        self.agg_density_gross = None
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_view_stats_cache = None
        self._reins_describe = None
        #: Occurrence joints built through :meth:`occ_joint`, keyed by the
        #: sizing that produced them; cleared on every ``update``.
        self._occ_joints = {}
        # rebucketing scheme for reins net/ceded distributions; set the backing
        # field directly (the setter clears the caches just initialised above)
        self._reins_bucket = reins_bucket if reins_bucket is not None else get_settings().discretization.reins_bucket
        if self._reins_bucket not in ('linear', 'nearest'):
            raise ValueError(
                f"reins_bucket must be 'linear' or 'nearest', not {self._reins_bucket!r}")
        # rebucketing scheme for discrete-severity atoms (dsev/dhistogram/fixed);
        # set the backing field directly (no caches to clear at construction)
        self._dsev_bucket = dsev_bucket if dsev_bucket is not None else get_settings().discretization.dsev_bucket
        if self._dsev_bucket not in ('linear', 'nearest'):
            raise ValueError(
                f"dsev_bucket must be 'linear' or 'nearest', not {self._dsev_bucket!r}")

        # ``stats_df`` is pre-created inside each broadcasting arm below once
        # ``n_components`` is known; see ``_init_stats_df``.
        ma = MomentAggregator(self.frequency.freq_moms)

        # overall freq CV with common mixing
        mix_cv = self.frequency.freq_a

        # broadcast arrays: force answers all to be arrays (?why only these items?!)
        if not isinstance(exp_el, Iterable):
            exp_el = np.array([exp_el])
        if not isinstance(sev_wt, Iterable):
            sev_wt = np.array([sev_wt])
        if not isinstance(sev_lb, Iterable):
            sev_lb = np.array([sev_lb])
        if not isinstance(sev_ub, Iterable):
            sev_ub = np.array([sev_ub])

        # counter to label components
        r = 0
        # broadcast together and create container for the severity distributions
        if np.sum(sev_wt) == len(sev_wt):
            # do not perform the exp / sev product, in this case
            # broadcast all exposure and sev terms together
            exp_el, exp_premium, exp_lr, en, attachment, limit, \
                sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, \
                sev_wt, sev_lb, sev_ub = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit,
                                    sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                                    sev_wt, sev_lb, sev_ub)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            all_arrays = zip(exp_el, exp_premium, exp_lr, en, attachment, limit,
                             sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                             sev_wt, sev_lb, sev_ub)
            # writable copy: the loop below writes the RESOLVED per-component
            # count back (broadcast views are read-only; the raw input can
            # hold the empirical/renewal -1 sentinel or 0 for loss-entered
            # exposure), mirroring the product arm's ``self.en[r] = _en0``
            self.en = np.array(en, dtype=float)
            self.attachment = attachment
            self.limit = limit
            # these all have the same length because have been broadcast
            n_components = len(exp_el)
            self._guard_exposure_rows(n_components)
            logger.debug('Aggregate.__init__ | Broadcast/align: exposures + severity = %d exp = '
                         '%d sevs = %d componets', len(exp_el), len(sev_a), n_components)
            self.sevs = np.empty(n_components, dtype=type(Severity))
            # limit-profile arm: weights all 1, single severity per exposure
            # row → mixture component ``m`` is trivially 0; ``e`` indexes the
            # broadcast exposure rows.
            self._init_stats_df([f'e{e_idx}.m0' for e_idx in range(n_components)])

            # perform looping creation of severity distribution
            # in this case wts are all 1, so no need to broadcast
            for _el, _pr, _lr, _en, _at, _y, _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _swt, _slb, _sub in all_arrays:
                assert _swt==1, 'Expect weights all equal to 1'

                # WARNING: note sev_xs and sev_ps are NOT broadcast
                self.sevs[r] = Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                        _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect)
                sev1, sev2, sev3 = self.sevs[r].moms()

                # input claim count trumps input loss
                _monetary = False
                if _en > 0:
                    _el = _en * sev1
                elif _el > 0:
                    _en = _el / sev1
                    _monetary = True
                # neither of these options can be triggered, by a dfreq dsev, for example.

                # for empirical/renewal freq claim count entered as -1;
                # resolved BEFORE the premium/lr reconciliation below so a
                # ``years at rate`` (premium-carrying) renewal records the
                # per-component lr correctly. Regression-neutral: no legacy
                # path pairs _en < 0 with _pr > 0 or _lr > 0 (a dfreq body
                # has no exposure clause).
                if _en < 0:
                    _en = np.sum(self.frequency.freq_a * self.frequency.freq_b)
                    _el = _en * sev1

                # Zero modification / truncation moves the mean: the exposure
                # clause sets the BASE mean and the realized E[N] follows (or,
                # under ``!``, invert so the realized mean meets the clause).
                # Expected loss must track the *realized* count, so it is
                # recomputed here -- before the premium / lr reconciliation.
                if self._freq_zm and _en > 0:
                    if _monetary:
                        self._zm_monetary_exposure = True
                        self._zm_requested_loss += _el
                    _en = self._zm_base_count(_en)
                    _el = self._zm_realized_count(_en) * sev1

                # if premium compute loss ratio, if loss ratio compute premium
                if _pr > 0:
                    _lr = _el / _pr
                elif _lr > 0:
                    _pr = _el / _lr

                # scale for the mix - OK because we have split the exposure and severity components
                _pr *= _swt
                _el *= _swt
                # _lr *= _swt  ?? seems wrong
                _en *= _swt

                # weights are all 1 here, so the thinning map is the identity
                # and each row simply carries its own frequency at its own count
                self._record_component(self._comp_cols[r], ma, _at, _y, _scv,
                                       self.frequency.freq_moms(_en), _en,
                                       _el, _pr, _lr,
                                       mix_cv, sev1, sev2, sev3)
                self.en[r] = _en
                r += 1

        else:
            # perform exp / sev product; but there is only one severity distribution
            # it could be a mixture - in which case we need to convert to en input (not loss)
            # and potentially re-weight for excess covers.
            # broadcast exposure terms (el, epremium, en, lr, attachment, limit) and sev terms (sev_) separately
            # then we take an "outer product" of the two parts...
            exp_el, exp_premium, exp_lr, en, attachment, limit = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit)
            sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt, sev_lb, sev_ub = \
                np.broadcast_arrays(sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                                    sev_wt, sev_lb, sev_ub)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            exp_arrays = [exp_el, exp_premium, exp_lr, en, attachment, limit]
            sev_arrays = [sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_lb, sev_ub]
            n_components = len(exp_el) * len(sev_name)
            self.en = np.empty(n_components, dtype=float)
            self.attachment = np.empty(n_components, dtype=float)
            self.limit = np.empty(n_components, dtype=float)
            # all broadcast arrays have the same length, hence:
            logger.debug(
                f'Aggregate.__init__ | Broadcast/product: exposures x severity = {len(exp_el)} x {len(sev_name)} '
                f'=  {n_components}')
            self.sevs = np.empty(n_components, dtype=type(Severity))
            # mixture-product arm: outer exposure × inner severity-mixture
            # gives a 2-D component grid; labels carry both indices.
            _n_exp = len(exp_el)
            self._guard_exposure_rows(_n_exp)
            _n_mix = len(sev_name)
            self._init_stats_df([
                f'e{e_idx}.m{m_idx}'
                for e_idx in range(_n_exp)
                for m_idx in range(_n_mix)
            ])

            # Ground-up mixture components are needed only to reweight the
            # mixture under an excess-of attachment (the ``sf(_at)`` call
            # below). Skip the constructions entirely when no exposure row
            # carries a positive attachment — saves one ``Severity`` per
            # mixture component on the common no-excess path.
            need_gup = any(
                _at is not None and _at > 0 for _at in attachment
            )
            gup_sevs = []
            if need_gup:
                for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, _swt in zip(*sev_arrays, sev_wt):
                    gup_sevs.append(Severity(_sn, 0, np.inf, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                             _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect))

            # perform looping creation of severity distribution
            for e_idx, (_el, _pr, _lr, _en, _at, _y) in enumerate(zip(*exp_arrays)):
                # adjust weights for excess coverage
                sev_wt0 = sev_wt.copy()
                # attachment can be None, and that needs to percolate through to Severity
                if _at is not None and _at > 0:
                    w1 = sev_wt0 * np.array([s.sf(_at) for s in gup_sevs])
                    sev_wt0 = w1 / w1.sum()

                # store actual sevs in a group (all are also appended to self.sevs) so we can compute the expected value
                # weight still irrelevant; but pull in layer and attaches which must vary for it to be meaningful
                actual_sevs = []
                for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, _swt in zip(*sev_arrays, sev_wt):
                    actual_sevs.append(Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                                _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect))

                # now we need to figure the severity across the mixture for this particular layer and  attach
                moms = []
                for s in actual_sevs:
                    # just return the first moment
                    moms.append(s.moms())

                # component mean (corresponding to the outside loop) can now be computed
                component_mean = (np.nan_to_num(np.array([m[0] for m in moms])) * sev_wt0).sum()

                # figure claim count if not entered, for the group (at this point we have not weighted down)
                # this forces subsequent calcuations to use (correct) en weighting even if premium or loss are
                # entered
                logger.info('%s xs %s, component_mean = %s, %s',
                            _y, _at, component_mean, [m[0] for m in moms])
                if _en == 0:
                    _en = _el / component_mean
                    # monetary exposure clause: a zero modification that shifts
                    # the mean will miss the stated money target (see the
                    # ZeroModifiedExposureWarning raised after the loops)
                    self._zm_monetary_exposure = True
                    self._zm_requested_loss += _el
                elif _en < 0:
                    # empirical / renewal sentinel: the frequency states its own
                    # count, so read it off the pmf rather than the exposure clause
                    _en = float(np.sum(self.frequency.freq_a * self.frequency.freq_b))

                # Resolve this row's frequency ONCE, before the mixture split.
                # A zero modification belongs to N, so it is applied here and
                # not once per component; ``freq_moms`` is already wrapped to
                # return the realized (shifted) moments for the base count.
                _en_base = self._zm_base_count(_en)
                _parent3 = self.frequency.freq_moms(_en_base)
                _row_mean = _parent3[0]

                # for cases where a mixture component has no losses in the layer
                # usually because of underflow.
                zero = None

                # break up the total claim count into parts and add sevs to self.sevs
                # need the first variables for sev statistics
                for m_idx, (_sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, s, _swt, (sev1, sev2, sev3)) in \
                        enumerate(zip(*sev_arrays, actual_sevs, sev_wt0, moms)):

                    # store the severity
                    if np.isnan(sev1):
                        if zero is None:
                            zero = Severity('dhistogram', 0, np.inf, 0, 0, 0, 0, 0, 0, [0], [1], 0, np.inf, 0, False)
                        # replace this component with the zero distribution
                        # ignore the (small) weights that are being ignored
                        self.sevs[r] = zero
                        _sn = 'dhistogram'
                        logger.info('%s xs %s on %s x (%s, %s, %s, %s, %s) + %s '
                                    ' | %s < X le %s '
                                    'component has sev=(%s, %s, %s), '
                                    ' weight = %s; replacing with zero.',
                                    _y, _at, _ssc, _at, _sm, _scv, _sa, _sb, _sloc,
                                    _slb, _sub, sev1, sev2, sev3, _swt)
                        sev1 = sev2 = sev3 = 0.0
                    else:
                        self.sevs[r] = s

                    # realized claim count for the row, figure total loss for
                    # the component. ``_row_mean`` is E[N] after any zero
                    # modification, resolved once above the loop.
                    if _row_mean > 0:
                        _el = _row_mean * sev1
                    else:
                        logger.info('%s xs %s on %s x (%s, %s, %s, %s, %s) + %s '
                                    ' | %s < X le %s has '
                                    '_en = %s. Adjusting el to 0.',
                                    _y, _at, _ssc, _at, _sm, _scv, _sa, _sb, _sloc,
                                    _slb, _sub, _row_mean)
                        _el = 0.

                    # if premium compute loss ratio, if loss ratio compute premium
                    if _pr > 0:
                        _lr = _el / _pr
                    elif _lr > 0:
                        _pr = _el / _lr

                    # scale for the mix - OK because we have split the exposure and severity components
                    _pr0 = _pr * _swt
                    _el0 = _el * _swt
                    # ``en`` carries the BASE count (what ``freq_pgf`` consumes,
                    # via ``frequency.base_mean``), so it splits the row's base
                    # rather than the realized mean. Identical to the thinned
                    # first moment for every unmodified frequency.
                    _en0 = _en_base * _swt

                    # Thin the row's frequency onto this component: K | N is
                    # Binomial(N, w) whatever N is. The weight cannot ride the
                    # claim count instead, because only a Poisson thinning is
                    # recoverable from its mean alone.
                    _freq3 = MomentAggregator.thin_moments(_swt, *_parent3)

                    self._record_component(f'e{e_idx}.m{m_idx}', ma, _at, _y, _scv,
                                           _freq3, _en0, _el0, _pr0, _lr,
                                           mix_cv, sev1, sev2, sev3)

                    self.en[r] = _en0
                    self.attachment[r] = _at
                    self.limit[r] = _y

                    r += 1

        # average exp_limit and exp_attachment — weighted by per-component
        # frequency mean. Sourced from the stats_df columns populated by the
        # broadcast loop above. ``stats_df`` is now all-float, so the casts
        # that this block used to need are gone.
        _comp_cols = self._comp_cols
        _comp_limit = self.stats_df.loc[('meta', 'limit'), _comp_cols]
        _comp_attach = self.stats_df.loc[('meta', 'attachment'), _comp_cols]
        _comp_freq = self.stats_df.loc[('freq', 'ex1'), _comp_cols]
        avg_limit = float(np.sum(_comp_limit * _comp_freq) / ma.tot_freq_1)
        avg_attach = float(np.sum(_comp_attach * _comp_freq) / ma.tot_freq_1)

        # store answer for total
        tot_prem = float(self.stats_df.loc[('meta', 'prem'), _comp_cols].sum())
        tot_loss = float(self.stats_df.loc[('meta', 'el'), _comp_cols].sum())
        # GROSS basis: ``tot_loss`` here is the theoretical loss before
        # ``update_work`` applies any reinsurance, so ``lr`` is a gross loss
        # ratio -- do not recompute it against a net/ceded loss.
        if tot_prem > 0:
            lr = tot_loss / tot_prem
        else:
            lr = np.nan

        # Write the post-loop totals directly into ``stats_df``: per-component
        # weights, then ``mixed`` and ``independent`` columns (theoretical
        # moments + meta).
        freq_ex1 = self.stats_df.loc[('freq', 'ex1'), _comp_cols]
        self.stats_df.loc[('meta', 'wt'), _comp_cols] = (freq_ex1 / ma.tot_freq_1).values
        # mixed and independent totals
        _flat_names = MomentAggregator.column_names()
        for _col, _remix in (('mixed', True), ('independent', False)):
            for _flat, _val in zip(_flat_names, ma.get_fsa_stats(total=True, remix=_remix)):
                self.stats_df.loc[_flat_col_to_stats_index(_flat), _col] = _val
            self.stats_df.loc[('meta', 'limit'), _col] = avg_limit
            self.stats_df.loc[('meta', 'attachment'), _col] = avg_attach
            self.stats_df.loc[('meta', 'sevcv_param'), _col] = 0
            self.stats_df.loc[('meta', 'el'), _col] = tot_loss
            self.stats_df.loc[('meta', 'prem'), _col] = tot_prem
            self.stats_df.loc[('meta', 'lr'), _col] = lr
            self.stats_df.loc[('meta', 'mix_cv'), _col] = (
                float(mix_cv) if np.isscalar(mix_cv) else np.nan
            )
            self.stats_df.loc[('meta', 'wt'), _col] = float(
                self.stats_df.loc[('meta', 'wt'), _comp_cols].sum()
            )

        self.n = ma.tot_freq_1
        # stamp the resolved unconditional claim count onto the frequency:
        # a parametric Frequency is a family until the exposure fixes its
        # mean; this is what freq_df (and any standalone use) reads
        self.frequency.en = float(self.n)
        # ... and the un-modified mean that ``freq_moms`` / ``freq_pgf``
        # actually consume. These differ only under zm / zt, where ``n`` is the
        # realized (shifted) mean and ``base_mean`` is what the exposure clause
        # set. Everything that evaluates the PGF must use ``base_mean``.
        self.frequency.base_mean = float(np.sum(self.en)) if self._freq_zm \
            else float(self.n)
        if (self._freq_zm and self._zm_monetary_exposure
                and not self._freq_pin_mean and self._zm_requested_loss > 0):
            # a money target was stated and the zero modification moved the mean
            # off it; say so, and name the fix. Silent for the ``n claims`` form,
            # where a count in / shifted count out is the documented default.
            _base_n = float(np.sum(self.en))
            warnings.warn(
                f'{self.name}: the exposure clause states a monetary target of '
                f'{self._zm_requested_loss:,.6g}, but zm/zt shifts the mean off '
                f'it: E[N] {_base_n:,.6g} -> {self.n:,.6g} delivers '
                f'{tot_loss:,.6g}. Drop the ! from the zm/zt clause to pin the '
                f'target instead.',
                ZeroModifiedExposureWarning, stacklevel=2)
        # Pull the headline moments off the canonical stats_df mixed column.
        _mixed = self.stats_df['mixed']
        self.actual_m = float(_mixed[('agg', 'mean')])
        self.actual_cv = float(_mixed[('agg', 'cv')])
        self.actual_skew = float(_mixed[('agg', 'skew')])
        # variance and sd come up in exam questions. Derive them directly from
        # the second moment (var = ex2 - mean^2), NOT as mean*cv: at mean 0 the
        # CV is legitimately nan, which would poison sd = mean*cv -> nan for a
        # signed (P&L) aggregate whose sd is perfectly well defined. The clamp
        # absorbs fp dust when a symmetric ex2 - mean^2 lands slightly negative.
        self.actual_var = max(float(_mixed[('agg', 'ex2')]) - self.actual_m ** 2, 0.0)
        self.actual_sd = math.sqrt(self.actual_var)
        # severity exact moments
        self.sev_m = float(_mixed[('sev', 'mean')])
        self.sev_cv = float(_mixed[('sev', 'cv')])
        self.sev_skew = float(_mixed[('sev', 'skew')])
        self.sev_var = max(float(_mixed[('sev', 'ex2')]) - self.sev_m ** 2, 0.0)
        self.sev_sd = math.sqrt(self.sev_var)

    @staticmethod
    def _build_renewal_frequency(exp_years, wait_name, wait_a, wait_b,
                                 wait_mean, wait_cv, wait_loc, wait_scale,
                                 wait_xs, wait_ps, wait_wt, wait_lb, wait_ub,
                                 wait_conditional, wait_attachment=None,
                                 wait_limit=np.inf):
        """Build the :class:`FrequencyRenewal` from the flat ``wait_*`` spec.

        Broadcasts the wait mixture terms (mirroring the severity broadcast,
        but with no exposure product) and constructs one :class:`Severity`
        per component:

        - ``dhistogram`` (``dwait``): the atom probabilities may sum to
          ``q < 1`` (defective ``dwait ... !``); the Severity gets the
          normalized pmf and the shortfall rides the component *weight* --
          the orchestrator treats missing weight as terminating mass.
        - layered (``wait y xs a ...``): the severity layer transform lives
          entirely INSIDE the Severity (``exp_attachment``/``exp_limit``
          with ``sev_conditional`` deciding ``(W-a | W>a) ^ y`` vs
          ``min((W-a)+, y)``); the orchestrator sees an ordinary component
          -- the unconditional zero-wait atom rides ``sev.cdf(0)`` into the
          cluster machinery. Cannot combine with a splice window.
        - conditional splice: ``lb/ub`` pass into the Severity as usual
          (``_apply_lb_ub`` renormalizes -- correct for a conditional law).
        - unconditional splice (``splice ... !``): the Severity is built
          WITHOUT ``lb/ub`` (the layer machinery would renormalize) and the
          window is masked on the wait grid -- the escaping mass is the
          defect. See :func:`aggregate._renewal.wait_count_pmf`.
        """
        bc = [np.atleast_1d(a) for a in np.broadcast_arrays(
            wait_name, wait_a, wait_b, wait_mean, wait_cv, wait_loc,
            wait_scale, wait_wt, wait_lb, wait_ub,
            wait_attachment, wait_limit)]
        components, weights = [], []
        for (_wn, _wa, _wb, _wm, _wcv, _wloc, _wsc, _wwt, _wlb, _wub,
             _watt, _wlim) in zip(*bc):
            _wn = str(_wn)
            layered = _watt is not None or float(_wlim) != np.inf
            if layered and (float(_wlb) != 0.0 or float(_wub) != np.inf):
                raise ValueError(
                    'a wait clause cannot combine a splice window [lb ub] '
                    'with a layer (y xs a); use one or the other')
            if _wn == 'dhistogram':
                if layered:
                    raise ValueError(
                        'layers (y xs a) are not supported on dwait; '
                        'write the clamped outcomes directly')
                ps = np.asarray(wait_ps, dtype=float)
                mass = float(ps.sum())
                sev_w = Severity('dhistogram', None, np.inf,
                                 sev_xs=np.asarray(wait_xs, dtype=float),
                                 sev_ps=ps / mass)
                components.append((sev_w, 0.0, np.inf, True))
                weights.append(float(_wwt) * mass)
            elif layered:
                sev_w = Severity(_wn,
                                 None if _watt is None else float(_watt),
                                 float(_wlim), _wm, _wcv, _wa, _wb,
                                 _wloc, _wsc, None, None, _wwt,
                                 0.0, np.inf, bool(wait_conditional))
                components.append((sev_w, 0.0, np.inf, True))
                weights.append(float(_wwt))
            elif not wait_conditional:
                sev_w = Severity(_wn, None, np.inf, _wm, _wcv, _wa, _wb,
                                 _wloc, _wsc, None, None, _wwt,
                                 0.0, np.inf, True)
                components.append((sev_w, float(_wlb), float(_wub), False))
                weights.append(float(_wwt))
            else:
                sev_w = Severity(_wn, None, np.inf, _wm, _wcv, _wa, _wb,
                                 _wloc, _wsc, None, None, _wwt,
                                 float(_wlb), float(_wub), True)
                components.append((sev_w, float(_wlb), float(_wub), True))
                weights.append(float(_wwt))
        return FrequencyRenewal(components, weights, float(exp_years))

    def _init_stats_df(self, comp_cols):
        """Pre-create the empty ``stats_df`` (NaN-filled).

        Called from each broadcasting arm of ``__init__`` once the per-
        component column labels are known. Columns are:

        * the broadcast components, named ``e{e}.m{m}`` where ``e`` is the
          exposure component (one per ``(claims|premium, limit xs attach)``
          row) and ``m`` is the severity-mixture component (one per weighted
          severity); the limit-profile arm always uses ``m=0``.
        * ``mixed`` / ``independent``: theoretical (subject / gross) totals.
          ``mixed`` is the model: one frequency over the pooled severity, which
          is also what the FFT computes. ``independent`` sums the component
          aggregates as though they were independent. The two agree only for a
          Poisson frequency, which is the one family whose mixture components
          really are independent. The gap between them is exactly the sum of
          the component covariances, which are available in closed form: with
          ``nu = E[N]``, ``v = Var(N)`` and component severity means ``mu_i``,
          ``Cov(A_i, A_j) = w_i w_j mu_i mu_j (v - nu)`` for ``i != j``, so
          ``Var(mixed) = Var(independent) + sum_{i != j} Cov(A_i, A_j)``. The
          sign of ``v - nu`` fixes the sign of the dependence: negative for
          fixed, binomial and Neyman A, zero for Poisson, positive for the
          mixed Poissons. The covariance matrix is not stored, being a matrix
          rather than a column; see
          :meth:`~aggregate.moments.MomentAggregator.thin_moments`.
        * ``empirical``: post-FFT empirical moments (the final, possibly
          after-reinsurance object).
        * ``after_occ``: empirical moments after the occurrence-reinsurance
          stage (populated in meta.4; scaffold here, NaN-filled).
        * ``occ_impact`` / ``agg_impact``: ``after_occ / mixed`` and
          ``empirical / after_occ`` ratios (scaffold).
        * ``gross_empirical``: subject empirical (the reinsurance validation
          hook from §1.3 of the plan; scaffold).
        * ``error``: noise-aware relative error of ``empirical`` vs
          ``mixed``.

        ``_record_component`` writes each component column inside the
        broadcast loop; the post-loop block writes ``mixed`` /
        ``independent``; ``update_work`` writes ``empirical`` and ``error``
        after the FFT; ``after_occ`` / ``occ_impact`` / ``agg_impact`` /
        ``gross_empirical`` are populated in meta.4 (reins reporting).

        ``stats_df`` is now an all-``float64`` frame: ``self.name`` lives on
        the attribute, so no string row is needed.
        """
        self._comp_cols = list(comp_cols)
        cols = self._comp_cols + [
            'mixed', 'independent', 'after_occ', 'empirical',
            'occ_impact', 'agg_impact', 'gross_empirical', 'error',
        ]
        self.stats_df = pd.DataFrame(
            np.nan, index=_STATS_ROW_INDEX, columns=cols, dtype=float,
        )

    @property
    def base_mean(self):
        """The un-modified expected claim count -- the mean ``freq_pgf`` consumes.

        Equals :attr:`n` for every unmodified frequency, so ordinary code need
        not distinguish them. Under ``zm`` / ``zt`` they part company: :attr:`n`
        is the *realized* (shifted) ``E[N]`` that the aggregate actually has,
        while this is the base mean the exposure clause set and the one the
        ``(a, b, 1)`` PGF is parameterized by. Every PGF evaluation -- the FFT
        convolution, the bivariate 2-D compound, the pedagogy transforms --
        must pass this, not ``n``.

        See :meth:`~aggregate.distributions.Frequency.modify_mean` for the map
        between the two.
        """
        base = getattr(self.frequency, 'base_mean', None)
        return float(self.n) if base is None else float(base)

    def _zm_base_count(self, en):
        """Map an exposure-clause claim count to the base mean the frequency consumes.

        Under ``zm`` / ``zt`` the DecL exposure clause states the *realized*
        ``E[N]`` by default, so the base mean is solved for by
        :meth:`~aggregate.distributions.Frequency.solve_base_mean`. The ``!``
        marker (:attr:`_freq_pin_mean` False) flips the reading: the clause
        then states the **un-modified (base)** mean and this is the identity,
        the realized mean being whatever the reweighting makes it.

        Returns ``en`` unchanged for an unmodified frequency, so callers can
        apply it without branching.
        """
        if not self._freq_zm or not en > 0:
            return en
        if self._freq_pin_mean:
            return self.frequency.solve_base_mean(en)
        return en

    def _zm_realized_count(self, base_en):
        """The realized ``E[N]`` for a base claim count.

        The forward (a, b, 1) shift; the identity for an unmodified frequency.
        Expected loss is computed from this rather than from the base count,
        so ``stats_df`` and the accumulated moments describe the same
        distribution.
        """
        if not self._freq_zm or not base_en > 0:
            return base_en
        return self.frequency.modify_mean(base_en)

    def _guard_exposure_rows(self, n_exposure_rows):
        """Refuse an exposure profile an empirical frequency cannot carry.

        An exposure profile of several rows is modeled as one frequency at the
        summed claim count against the pooled severity, which is what the FFT
        computes. A ``dfreq`` (or ``years``) frequency states its own count and
        cannot be asked for a different mean, so the summed-count parent does
        not exist and the program is ill posed.

        This is not the severity-mixture case, which is well posed for every
        frequency: a mixture splits one count rather than summing several, and
        the split is handled by thinning
        (:meth:`~aggregate.moments.MomentAggregator.thin_moments`).

        Parameters
        ----------
        n_exposure_rows : int
            Number of rows in the broadcast exposure clause, before the
            severity-mixture product.

        Raises
        ------
        ValueError
            If there is more than one exposure row and the frequency carries
            its own count.
        """
        if n_exposure_rows > 1 and self.frequency.carries_own_count:
            raise ValueError(
                f'{self.name}: a dfreq or years frequency states its own claim '
                f'count, so it cannot be spread over the {n_exposure_rows} rows '
                f'of an exposure profile. A profile is one frequency at the '
                f'summed count over the pooled severity, and a stated count '
                f'distribution cannot be asked for a different mean. Either '
                f'state the count with an exposure clause and a frequency that '
                f'takes a mean (for example "1 claim ... fixed" or '
                f'"... poisson"), or declare one aggregate per row and combine '
                f'them in a portfolio. A severity mixture ("wts") is unaffected '
                f'and needs no change.')

    def _record_component(self, col, ma, attach, layer, scv, freq3, base, el, prem, lr, mix_cv,
                          sev1, sev2, sev3):
        """Accumulate this component into ``ma`` and write its ``stats_df`` column.

        Called once per component from each of the two broadcasting arms of
        ``__init__``: the limit-profile arm (all weights == 1) and the
        mixture-product arm. Centralises which ``stats_df`` per-component
        column gets written so the two arms cannot drift apart.

        Parameters
        ----------
        col : str
            Per-component column label in ``stats_df``, of the form
            ``e{e}.m{m}`` (exposure × severity-mixture). Limit-profile arm
            uses ``m=0``.
        ma : MomentAggregator
            Accumulator collecting freq, sev, agg moments across all components.
        attach, layer : float
            Per-component attachment and layer height.
        scv : float
            Severity CV parameter for this component.
        freq3 : tuple of float
            The component's first three non-central frequency moments, already
            thinned by the mixture weight by the caller (see
            :meth:`MomentAggregator.thin_moments`). The caller supplies the
            moments rather than a count because only a Poisson thinning is
            recoverable from its mean alone.
        base : float
            This component's share of the exposure row's un-modified claim
            count, which ``ma`` accumulates so ``get_fsa_stats(remix=True)``
            can re-enter ``freq_moms`` at the count the frequency was built
            with. Sums to the row count across the row's components.
        el, prem, lr : float
            Per-component expected loss, premium, loss ratio, already scaled by
            the mixture weight by the caller.
        mix_cv : float
            Overall mixing-distribution CV (constant across rows).
        sev1, sev2, sev3 : float
            First three raw severity moments for this component.
        """
        ma.tot_freq_base += base
        ma.add_fs(*freq3, sev1, sev2, sev3)
        moments = ma.get_fsa_stats(total=False)
        # Write this component's data directly into the canonical ``stats_df``
        # column. Maps MA's flat moment names to the ``(component, measure)``
        # MultiIndex via ``_flat_col_to_stats_index``. ``('meta', 'wt')`` is
        # filled in by the post-loop block (it depends on total frequency).
        # ``mix_cv`` is only meaningful for true mixed-Poisson frequencies; for
        # ``dfreq`` (where ``freq_a`` is the discrete pmf array) it is NaN.
        self.stats_df.loc[('meta', 'limit'), col] = layer
        self.stats_df.loc[('meta', 'attachment'), col] = attach
        self.stats_df.loc[('meta', 'el'), col] = el
        self.stats_df.loc[('meta', 'prem'), col] = prem
        self.stats_df.loc[('meta', 'lr'), col] = lr
        self.stats_df.loc[('meta', 'sevcv_param'), col] = scv
        self.stats_df.loc[('meta', 'mix_cv'), col] = (
            float(mix_cv) if np.isscalar(mix_cv) else np.nan
        )
        for flat, val in zip(MomentAggregator.column_names(), moments):
            self.stats_df.loc[_flat_col_to_stats_index(flat), col] = val

    # ================================================================
    # Repr / info / help — string and HTML representations
    # ================================================================

    # ``label`` / ``_title_name`` come from ``LabeledMixin`` (the shared
    # label surface); ``name`` stays the identity handle. See dev/plan-labels.md.

    def __repr__(self):
        """
        String version of _repr_html_
        :return:
        """
        return f'{self.label}, {super(Aggregate, self).__repr__()}'

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        s = [self.info]
        with pd.option_context('display.width', 200,
                               'display.max_columns', 15,
                               'display.float_format', lambda x: f'{x:,.5g}'):
            # get it on one row
            s.append(str(self.summary_df))
        # s.append(super().__repr__())
        return '\n'.join(s)

    def _approx_description(self):
        """One-line description of the method-of-moments fit, or ``''`` if none.

        Renders the *original* program together with the fitted family and its
        parameters, e.g.::

            <program>  approximated by sgamma: gamma(loc=.., scale=.., a=..), m=.. cv=.. skew=..

        Composed lazily (not at construction) because the original ``program``
        text is assigned by the build path only after ``__init__`` returns; the
        fit itself is captured in :attr:`_approx_fit`. Returns ``''`` for an
        ordinary (``exact``) aggregate, or when no fit was recorded.

        Returns
        -------
        str
        """
        fit = getattr(self, '_approx_fit', None)
        if not fit:
            return ''
        prog = self.pprogram if self.program else self.name
        parts = [f"loc={fit['sev_loc']:.6g}", f"scale={fit['sev_scale']:.6g}"]
        a = fit.get('sev_a')
        if a is not None and np.isfinite(a):
            parts.append(f"a={a:.6g}")
        params = ', '.join(parts)
        return (f"{prog}  approximated by {fit['kind']}: {fit['sev_name']}({params}), "
                f"m={fit['m']:.6g} cv={fit['cv']:.6g} skew={fit['skew']:.6g}")

    def _spec_hash(self):
        """Display-only 8-hex id of the canonical spec.

        Computed on the fly from ``self._spec`` (machine-independent md5,
        first 8 hex, matching ``Distortion.id()``); no stored attribute, no
        timestamp.
        """
        blob = json.dumps(self._spec, sort_keys=True, default=str)
        return hashlib.md5(blob.encode('utf-8')).hexdigest()[:8].upper()

    @property
    def info(self):
        """Fixed-layout multi-line summary string.

        Every row is always present, in the same order, for every
        ``Aggregate``; a value that is not (yet) available -- e.g. the grid
        block before ``update`` -- renders as ``n/a``. The row catalogue and
        value enumerations are documented in ``dev/info-strings.rst``. Shares
        the label/value convention (:func:`aggregate.constants.info_row`)
        with ``Portfolio`` and ``Distortion``.
        """
        updated = self.bs > 0
        n_sev = len(self.sevs)
        if n_sev == 1:
            sv = self.sevs[0]
            sev_desc = sv.tail_description
        else:
            sev_desc = f'{n_sev} components'
        if updated:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1 / self.bs)}'
        else:
            bss = INFO_NA
        # premium / expected loss / loss ratio: populated when a premium is
        # known (the DecL exposure clause states one) and the object is updated.
        # ``P(X=0)`` is the sign-neutral break-even / no-loss atom
        # (:attr:`prob_eq_0`), so it is meaningful on a loss and a payoff alike.
        prem = float(self.stats_df.loc[('meta', 'prem'), 'mixed'])
        e_loss = None
        p_eq_0 = INFO_NA
        if updated and self.agg_density is not None:
            e_loss = float(self.est_m)
            p_eq_0 = f'{self.prob_eq_0:.6g}'
        rows = [
            ('aggregate object name', self.name),
            ('value_type', self.value_type),
            ('claim count', f'{self.n:,.3f}'),
            ('frequency distribution', self.frequency.freq_name),
            ('severity distribution', sev_desc),
            ('approximate', getattr(self, 'approximation', '') or 'exact'),
            ('bs', bss),
            ('log2', self.log2 if updated else INFO_NA),
            ('padding', self.padding if updated else INFO_NA),
            ('sev_calc', self.sev_calc if updated else INFO_NA),
            ('dsev_bucket', self.dsev_bucket),
            ('normalize', self.normalize if updated else INFO_NA),
            ('x_min', f'{self.x_min:,.6g}' if updated else INFO_NA),
            ('x_max', f'{self.x_max:,.6g}'
             if updated and self.x_max is not None else INFO_NA),
            ('premium', f'{prem:,.6g}' if prem > 0 else INFO_NA),
            ('expected loss', f'{e_loss:,.6g}' if e_loss is not None else INFO_NA),
            ('loss ratio', f'{e_loss / prem:.1%}'
             if prem > 0 and e_loss is not None else INFO_NA),
            ('P(X=0)', p_eq_0),
            ('validation_eps', self.validation_eps),
            ('reinsurance', self.reins_kinds.lower()),
            ('occurrence reinsurance', self._reins_description('occ').lower()),
            ('aggregate reinsurance', self._reins_description('agg').lower()),
            ('validation', self.validation_description),
        ]
        s = [info_row(label, value) for label, value in rows]
        # Tail report summary (frequency / severity / aggregate -- support and
        # per-side tail class); spec-only, so available before update.
        s.extend(_tail.describe_rows(self._tail_rows()))
        s.append(info_row('bounded', self.bounded))
        s.append(info_row('id', self._spec_hash()))
        return '\n'.join(s)

    @property
    def validation_score(self):
        """Continuous validation score: how well the grid reproduces theory (float).

        **In units of the validation tolerance**, so ``score <= 1`` means the
        object passes at its own ``validation_eps`` and ``1`` is exactly the pass
        boundary. Where :attr:`valid` says whether a line was crossed, this says
        by how far, which is what makes it comparable across grids and the
        quantity :meth:`sharpen` minimizes.

        Six terms: severity and aggregate mean, CV and skewness, each relative
        error divided by its own tolerance, combined as a power mean with
        ``power=2``. See :func:`~aggregate._validation.validation_score` for the
        other powers and :func:`~aggregate._validation.validation_score_terms`
        for the per-term detail.
        """
        return _validation.validation_score(self)

    @property
    def validation_description(self):
        """
        Short one-line validation verdict (str).

        The terse phrase the ``info`` row and the one-line intro carry:
        ``'not unreasonable'``, ``'fails agg cv'``. The verbose form is
        :attr:`validation_explanation`. Validation is computed if needed.
        """
        return _validation.validation_description(self)

    @property
    def validation_explanation(self):
        """
        Long-narrative explanation of the validation result (str).

        The consistent narrative surface, mirroring ``tail_explanation`` /
        ``bs_explanation``, and the verbose form of
        :attr:`validation_description`. Validation is computed if needed.
        """
        return _validation.validation_explanation(self)

    def _html_info_blob(self):
        """Short HTML intro for ``_repr_html_`` -- identity, grid, and the
        validation result.

        The headline table (``summary_df``) carries the risk view; this blob is
        the one-glance context. The validation result is always stated inline as
        the closing sentence (``Validation: {validation_description}.``) -- a
        clean object reads "not unreasonable", a failing one names the offending
        moment.
        """
        parts = [f'Frequency distribution {self.frequency.freq_name}.']
        n = len(self.sevs)
        if n == 1:
            sv = self.sevs[0]
            parts.append(f'Severity {sv.tail_description}.')
        else:
            parts.append(f'Severity with {n} components.')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            parts.append(f'Updated with bucket size {bss} and log2 = {self.log2}.')
        parts.append(f'Validation: {self.validation_description}.')
        return (f'<h3>Aggregate object: {self._title_name}</h3>\n'
                f'<p>{" ".join(parts)}</p>')

    def _text_info_blob(self) -> str:
        """Short plain-text intro (the text twin of :meth:`_html_info_blob`).

        Object identity, the frequency / severity families, the realised grid,
        and the validation result -- the one-glance context :meth:`qd` prints
        above the headline ``summary_df``. **One line**: the sentences are
        space-joined (not stacked), and the blob closes with
        ``Validation: {validation_description}.`` exactly as the HTML twin
        does, so the two renderings say the same thing.
        """
        s = [f'Aggregate object: {self._title_name}.',
             f'Frequency distribution {self.frequency.freq_name}.']
        n = len(self.sevs)
        if n == 1:
            sv = self.sevs[0]
            s.append(f'Severity {sv.tail_description}.')
        else:
            s.append(f'Mixed severity with {n} components.')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.')
        s.append(f'Validation: {self.validation_description}.')
        return ' '.join(s)

    def _repr_html_(self):
        """HTML view: short intro (with the inline validation result) and the
        ``summary_df`` headline. The ``tail_df`` return-period table is served
        on demand via :meth:`tail_df`, not inlined here.
        """
        fmt = lambda x: f'{x:,.5g}'
        out = [self._html_info_blob(),
               '<h4>Summary</h4>',
               self.summary_df.to_html(float_format=fmt, na_rep='')]
        return '\n'.join(out)

    # ================================================================
    # Discretization, snap, update, FFT convolution
    # The 5-line FFT core (Mildenhall 2024, §2.2) lives in
    # ``_freq_sev_convolution`` below.
    # ================================================================

    @property
    def one_claim(self):
        """Whether the claim count is identically one.

        Returns
        -------
        bool
            ``True`` iff ``N = 1`` with probability one, however the frequency
            happens to be spelled.

        Notes
        -----
        This is the *convolution* gate for the exact severity copy shortcut.
        With exactly one claim the aggregate **is** the (post occurrence
        reinsurance) severity, so ``iFFT(P_N(FFT(sev)))`` is an identity
        executed numerically: it returns the severity plus machine epsilon
        dust. Copying the array instead is exact and cheaper. See
        :func:`aggregate._aggregate_compute.freq_sev_convolution` (1-D) and
        :func:`aggregate.bivariate.netceded_joint_density` (2-D), which share
        this predicate so the two never drift apart.

        Two spellings reach the same law and both answer ``True``:

        - ``1 claim ... fixed``: a
          :class:`~aggregate._frequency.FrequencyFixed` whose component claim
          counts sum to one. A limit profile splitting that single claim over
          several components still qualifies, because ``sev_density`` is then
          the corresponding severity mixture.
        - ``dfreq [1]``: a :class:`~aggregate._frequency.FrequencyEmpirical`
          carrying all its mass on the outcome one.
          :class:`~aggregate._frequency.FrequencyRenewal` is an empirical
          frequency post build, so a renewal count that degenerates to one
          claim qualifies too.

        The empirical test is on the **support**, never on the mean:
        ``dfreq [0 2] [.5 .5]`` has mean one and is emphatically not one
        claim. Zero probability atoms are ignored, since
        :func:`~aggregate._severity.validate_discrete_distribution` makes the
        outcomes distinct and ascending but never drops a zero mass.
        """
        freq = self.frequency
        if freq.freq_name == 'fixed':
            return bool(self.en is not None and np.sum(self.en) == 1)
        if isinstance(freq, FrequencyEmpirical):
            atoms = np.asarray(freq.freq_a, dtype=float)
            probs = np.asarray(freq.freq_b, dtype=float)
            return bool(np.array_equal(atoms[probs > 0], np.array([1.0])))
        return False

    def _signed_severity(self):
        """Whether the aggregate has signed (negative-support) severity.

        Returns ``True`` iff any component severity is signed -- declared with
        the ``ssev`` keyword (continuous) or a ``dsev`` with a negative atom,
        both recorded on ``Severity.signed`` at construction.

        This is the *convolution-grid* gate: it drives the negative-x severity
        layout (``i0``), the two-sided window in :meth:`_bs_window`, and
        ``_signed_sev``. It is **not** the public display/combine gate -- a
        ``pnl`` aggregate has an ordinary non-negative loss severity (so this is
        ``False``) but is signed at the aggregate level via the affine wrapper;
        see :meth:`_signed`.

        Returns
        -------
        bool

        Notes
        -----
        Signedness is a *parse-time property of the severity*, not a runtime
        flag -- which is what lets the analytic moments (and hence the auto
        window) be correct before any FFT, and is why there is **no**
        ``signed=`` override on ``update``: a built ``Severity`` is already
        clamped or not, and flipping it would require a deep rebuild. To change
        signedness, change the declaration (``sev`` / ``dsev`` / ``ssev``).
        See ``dev/plan-negative-x-agg.md``.
        """
        return any(getattr(s, 'signed', False)
                   for s in (self.sevs if self.sevs is not None else []))

    def _signed(self):
        """Whether the aggregate is signed (straddles 0) for display / combine.

        ``True`` when the severity itself reaches below 0
        (:meth:`_signed_severity`) -- a ``ssev`` continuous severity or a
        ``dsev`` with a negative atom. This is the gate read by plotting,
        two-sided quantiles, the signed-aware ``validation_df`` (SD instead of CV),
        and the :class:`Portfolio` combine. (A premium-minus-loss position is
        now the separate :class:`PnL` veneer, not a signed aggregate.)

        Returns
        -------
        bool
        """
        return self._signed_severity()

    def _severity_negative_buckets(self, bs):
        """Number of negative buckets the severity reaches (index of physical 0).

        Returns ``i0`` such that the severity discretisation grid
        ``xs_sev = (arange(N) - i0) * bs`` places physical 0 at index ``i0``
        and covers the severity's left tail. ``0`` for any severity supported
        on ``[0, inf)`` -- which keeps the non-negative path byte-for-byte
        unchanged. For a severity that reaches below 0 (e.g. ``norm``, a
        shifted distribution, or a ``dsev`` with negative atoms) it is the
        number of buckets from 0 down to the severity's effective lower
        endpoint.

        Parameters
        ----------
        bs : float
            Bucket size.

        Returns
        -------
        int
            ``max(0, ceil(-lo / bs))`` where ``lo`` is the smallest effective
            lower endpoint over the severity mixture components. A small
            tolerance keeps floating-point dust at exactly 0 from spuriously
            triggering signed mode.

        Notes
        -----
        ``lo`` is taken as the ``1e-12`` lower quantile of each *signed*
        component (a signed Severity has identity layering, so its ``ppf`` is
        the raw lower quantile), mirroring how the upper grid edge is sized from
        a high quantile. Any mass below the leftmost bucket is dropped and
        absorbed by the (optional) renormalisation in ``discretize`` -- at the
        ``1e-12`` level this is negligible.
        """
        # Only signed severities reach below 0; an unsigned component keeps the
        # clamp-at-0 layering (no negative buckets). i0 == 0 unless the
        # aggregate is signed, so the non-negative path is untouched.
        if not getattr(self, '_signed_sev', False):
            return 0
        if self.sevs is None or len(self.sevs) == 0:
            return 0
        los = []
        for sev in self.sevs:
            if not getattr(sev, 'signed', False):
                continue
            try:
                lo = float(sev.ppf(1e-12))   # identity layering -> raw quantile
            except Exception:  # pragma: no cover - defensive
                lo = 0.0
            if not np.isfinite(lo):
                lo = 0.0
            los.append(lo)
        lo = min(los) if los else 0.0
        if lo >= 0:
            return 0
        i0 = int(np.ceil(-lo / bs - 1e-9))
        # The severity's negative reach must fit inside the grid.
        N = len(self.xs)
        if i0 >= N:
            logger.warning(
                '%s: severity negative reach (%d buckets) exceeds grid size '
                '%d; clipping. Increase log2 or bs for a signed aggregate.',
                self.name, i0, N)
            i0 = N - 1
        return i0

    def discretize(self, sev_calc, discretization_calc, normalize):
        """
        Discretize the severity distributions and weight.

        ``sev_calc`` describes how the severity is discretized. The
        options are discrete=round, forward, backward or moment.

        ``sev_calc='continuous'`` (same as forward, kept for backwards compatibility) is used when
        you think of the resulting distribution as continuous across the buckets
        (which we generally don't). The buckets are not shifted and so :math:`Pr(X=b_i) = Pr( b_{i-1} < X \\le b_i)`.
        Note that :math:`b_{i-1}=-bs/2` is prepended.

        We use the discretized distribution as though it is fully discrete and only takes values at the bucket
        points. Hence, we should use `sev_calc='discrete'`. The buckets are shifted left by half a bucket,
        so :math:`Pr(X=b_i) = Pr( b_i - b/2 < X \\le b_i + b/2)`.

        The other wrinkle is the righthand end of the range. If we extend to np.inf then we ensure we have
        probabilities that sum to 1. But that method introduces a probability mass in the last bucket that
        is often not desirable (we expect to see a smooth continuous distribution, and we get a mass). The
        other alternative is to use endpoint = 1 bucket beyond the last, which avoids this problem but can leave
        the probabilities short. We opt here for the latter and normalize (rescale).

        ``discretization_calc`` controls whether individual probabilities are computed using backward-differences of
        the survival function or forward differences of the distribution function, or both. The former is most
        accurate in the right-tail and the latter for the left-tail of the distribution. We are usually concerned
        with the right-tail, so prefer `survival`. Using `both` takes the greater of the two esimates giving the best
        of both worlds (underflow makes distribution zero in the right-tail and survival zero in the left tail,
        so the maximum gives the best estimate) at the expense of computing time.

        Sensible defaults: sev_calc=discrete, discretization_calc=survival, normalize=True.

        :param sev_calc:  discrete=round, forward, backward, or continuous
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :return:
        """

        # Severity is discretised on ``xs_sev`` (physical 0 at index i0), which
        # equals ``self.xs`` on the default 0-based grid. ``i0 > 0`` means the
        # severity reaches below 0 (signed mode). The math lives in the pure
        # kernel ``discretize_severities`` (``_aggregate_compute``) so other
        # consumers (the renewal waiting-time path) can reuse it.
        xs_sev = self.xs_sev if self.xs_sev is not None else self.xs
        return discretize_severities(
            self.sevs, xs_sev, self.bs, i0=self.i0, sev_calc=sev_calc,
            discretization_calc=discretization_calc, normalize=normalize,
            dsev_bucket=self.dsev_bucket, rebucket=self._rebucket_to_grid)

    def snap(self, x):
        """
        Snap value x to the index of density_df, i.e., as a multiple of self.bs.

        :param x:
        :return:
        """
        ix = self.density_df.index.get_indexer([x], 'nearest')[0]
        return self.density_df.iloc[ix, 0]

    @property
    def prob_eq_0(self):
        """``P(X == 0)`` -- the probability the outcome is exactly zero.

        On demand, from the realised grid, after :meth:`update`. On a **loss**
        object this reads as "no loss" (the ground-up ``P(N = 0)``, plus any
        severity atom at zero and any mass the reinsurance nets to zero); on a
        **payoff** / P&L object as "exactly break even". Shared, sign-neutral
        surface with :attr:`Portfolio.prob_eq_0` and
        :attr:`~aggregate._pnl.PnL.prob_eq_0`.

        Returns
        -------
        float or None
            ``None`` before :meth:`update` (there is no realised grid yet).

        Notes
        -----
        Reads the mass at the zero grid point through the canonical
        :class:`~aggregate._grid_distribution.GridDistribution` (``pmf(0)``) --
        identical to :meth:`pmf` at zero. On a **discrete** book that is the
        exact atom. On a **continuous** severity the zero bucket also absorbs
        every outcome that discretizes to zero (losses below ``bs``, or below
        ``bs / 2`` under the round scheme), so the answer is ``P(N = 0)`` plus
        that sliver -- the usual FFT-grid reading, and it tightens as ``bs``
        shrinks.
        """
        if self.agg_density is None:
            return None
        return float(self._grid_distribution().pmf(0.0))

    @property
    def value_type(self):
        """Sign convention for the variable: ``'loss'`` or ``'payoff'``.

        Records how the aggregate should be read -- actuarial **loss**
        ("more is worse", the default) vs. **payoff** / asset ("more is
        better"). It is **inert for the distribution itself**: density,
        moments, quantiles, deficit and plotting do not depend on it.
        It is consumed only when applying distortions / pricing (specified
        in the Portfolio plan and downstream pricing work), where a
        ``'payoff'`` object is negated / the dual distortion applied.

        The role is stored as a boolean (``_is_loss_value``); this getter
        returns the **configured label** for the role (``[labels]`` in the
        config, defaults ``'loss'`` / ``'payoff'``). The setter accepts the
        canonical tokens or the configured labels. Pricing code must branch
        on ``_is_loss_value``, never on the label text.

        See ``dev/plan-negative-x-agg.md`` §5.5.
        """
        return value_type_label(self._is_loss_value)

    @value_type.setter
    def value_type(self, v):
        self._is_loss_value = value_type_role(v)
        # The cached GridDistributions carry the orientation, so a post-build
        # role change must drop them; the next accessor rebuilds with the new
        # is_loss_value. (Until now only update() reset these caches.)
        self._dist = None
        self._sev_dist = None

    def make_pnl(self, consideration=None, *, gross=None, ceded=None,
                 expense_spec=None, gcn_economics=None, consideration_label=None,
                 loss_label=None):
        """Wrap this aggregate as the risky leg of a :class:`PnL` position.

        Object sugar delegating to the builders in
        :mod:`aggregate._pnl_builders`. Two construction modes, both the
        **consolidated** (single-group) pnl face
        ([Decision-PnL-Is-Consolidated]):

        * **plain** -- ``make_pnl(consideration=C)``: a one-group ``sell``
          ledger over this aggregate's own density (for a reinsurance-bearing
          aggregate that is the *net* density -- what comes out of the
          aggregate).
        * **consolidated reinsurance** -- ``make_pnl(gross=Pg, ceded=Pc)`` on
          a reinsurance-bearing aggregate
          (:func:`~aggregate._pnl_builders.build_consolidated_pnl`): one
          net-premium consideration leg (gross - ceded premiums +
          commissions) against the engine's net loss over its deepest net
          marginal. The per-step walk (gross -> each cover -> Total) is the
          DecL ``xpnl`` face.

        Parameters
        ----------
        consideration : float, array-like, or callable, optional
            The plain-mode consideration, **signed** (``+`` received, ``-``
            paid); a vector sums to one book amount; a callable ``f(x)`` is a
            loss-sensitive consideration applied bucket-wise. Mutually
            exclusive with ``gross``/``ceded``.
        gross, ceded : float, optional
            The gross premium received and ceded premium paid (positive
            magnitudes) for the group-ledger view.
        consideration_label : str, optional
            Name for the premium leg (the DecL premium ``as`` clause);
            defaults to ``'consideration'`` (plain) / ``'premium'`` (ledger).
        loss_label : str, optional
            Name for the loss leg (the DecL engine ``as`` clause); defaults to
            ``'loss'``.

        Returns
        -------
        PnL
            **Always** a :class:`~aggregate.PnL` value object; the ledger form
            carries its resolved cession economics as ``pnl.economics``.

        Notes
        -----
        ``build('pnl NAME C premium less <body>')`` is sugar for
        ``build('agg NAME <body>').make_pnl(consideration=C)``.
        """
        from ._pnl_builders import build_plain_pnl, build_consolidated_pnl
        if gross is not None or ceded is not None:
            if gross is None or ceded is None:
                raise ValueError(
                    'a consolidated reinsurance PnL needs both gross= and '
                    'ceded= premiums.')
            if consideration is not None:
                raise ValueError(
                    'pass either consideration= or gross=/ceded=, not both.')
            if self.agg_reins is None and self.occ_reins is None:
                raise ValueError(
                    'the consolidated reinsurance view requires reinsurance on '
                    'the risky leg; the aggregate carries no occurrence / '
                    'aggregate treaty.')
            face = build_consolidated_pnl(
                self, gross=gross, ceded=ceded, gcn_economics=gcn_economics,
                expense_spec=expense_spec,
                consideration_label=consideration_label,
                loss_label=loss_label, name=self.name,
                label=self.label)
            face.engine = self
            return face
        if consideration is None:
            raise ValueError(
                'PnL needs a consideration= (or gross=/ceded= for the '
                'Gross/Ceded/Net view).')
        face = build_plain_pnl(
            self, consideration=consideration,
            consideration_label=consideration_label, loss_label=loss_label,
            expense_spec=expense_spec, name=self.name,
            label=self.label)
        face.engine = self
        return face

    def update(self, log2=16, bs=0, bucket_sizing_p=BUCKET_SIZING_P, debug=False,
               x_min='auto', x_max=None, window_convention=None, sharpen=False,
               **kwargs):
        """
        Convenience function, delegates to update_work. Avoids having to pass xs.

        :param log2:
        :param bs:
        :param bucket_sizing_p: p value passed to the moment-window bucket sizer. If > 1 converted to 1 - 10**-p.
        :param debug:
        :param x_min: lower edge of the output window. ``'auto'`` (default)
          resolves to ``0`` for an ordinary non-negative aggregate (today's
          behaviour) and to ``None`` (automatic two-sided window) for a signed
          aggregate -- so a P&L declared with ``ssev`` / negative ``dsev`` just
          works from ``build``. ``None`` forces the automatic window; a number
          forces that origin (snapped to a multiple of ``bs``; may be negative).
        :param x_max: upper edge of the output window; informational, the grid
          length is fixed by ``log2``. Currently unused when ``x_min`` is given
          explicitly (the window is ``[x_min, x_min + (2**log2)*bs)``).
        :param window_convention: ``'loss'`` / ``'payoff'`` to override the sign
          convention orienting the automatic windowed placement (per-edge
          coverage and padding skew). ``None`` (default) derives it from
          ``value_type``. See ``dev/plan-bucket-window-2.md`` §1A.
        :param sharpen: opt in to auto-sharpening. ``False`` (default) leaves the
          grid exactly as the estimator chose it. ``True`` runs :meth:`sharpen`
          once the update completes, which probes the eight neighbouring
          ``(bs, log2)`` cells and moves to a better one on a large win. Off by
          default, and *not* turned on by ``build``, because a probe costs eight
          extra updates.
        :param kwargs:  passed through to update
        :return:

        Signedness is carried by the severity declaration (``ssev`` /negative
        ``dsev``), not by an argument here -- see ``_signed``.
        """
        # Unified bucket + window estimator: runs the candidate sizing methods
        # (moment / exact_discrete / bounded_small), records them in the
        # inspectable ``self._bs_window_df``, and selects (see ``_bs_window``).
        # ``x_min='auto'`` lets the selected method choose the origin (0 for a
        # non-negative aggregate, a negative origin only when signed); a number
        # forces it; ``log2`` is a cap. The legacy non-negative ``moment`` path
        # reproduces the legacy 3-moment bucket sizing exactly, so ordinary
        # aggregates are unchanged.
        x_min_arg = None if (isinstance(x_min, str) and x_min == 'auto') else x_min
        bs, log2, x_min = self._bs_window(log2, bs, x_min_arg, bucket_sizing_p,
                                          window_convention=window_convention)
        N = 1 << log2
        # ``x_min`` is the convolution origin chosen by ``_bs_window``: 0 for an
        # ordinary aggregate; a negative origin when the severity is signed
        # (``ssev`` / negative-``dsev``), so the loss convolves on its genuine
        # signed grid.
        xs = x_min + np.arange(0, N, dtype=float) * bs
        rv = self.update_work(xs, debug=debug, x_min=x_min, x_max=x_max,
                              **kwargs)
        if sharpen:
            # Opt-in only. ``sharpen``'s own probe updates pass sharpen=False,
            # so there is no recursion.
            self.sharpen()
        return rv

    def update_work(self, xs, padding=1, sev_calc='discrete',
                    discretization_calc='survival', normalize=True, force_severity=False,
                    reins_bucket=None, dsev_bucket=None, debug=False, x_min=0, x_max=None):
        """
        Compute a discrete approximation to the aggregate density via FFT.

        See discretize for sev_calc, discretization_calc and normalize.

        Empirical-moment note: the aggregate raw moments -- and hence the
        empirical CV/skew shown in ``stats_df`` and ``validation_df`` -- are taken
        from a de-fuzzed *copy* of the FFT density (values below machine
        epsilon zeroed). Without this, sub-eps floating-point fuzz in far-tail
        buckets is amplified by ``x**3`` in the third moment and corrupts the
        empirical skew on wide grids: a symmetric distribution's skew can
        drift from ~1e-15 to ~1e-4 as log2 grows, purely from buckets the
        distribution never reaches. The fuzz is safe to drop because the FFT
        is exact up to rounding and the exact aggregate has no negative density
        even under aliasing, so any stray value is small. ``self.agg_density``
        is deliberately left as the raw FFT output (consistent with
        ``ftagg_density``); only the moment computation sees the cleaned copy.
        See the inline comment at the moment computation for full detail.

        Quick simple test with log2=13 update took 5.69 ms and _eff took 2.11 ms. So quicker
        but not an issue unless you are doing many buckets or aggs.

        :param xs: range of x values used to discretize
        :param padding: for FFT calculation
        :param sev_calc:  discrete=round, forward, backward, or continuous
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :param force_severity: make severities for plotting even when only the aggregate is requested
        :param reins_bucket: optional override of the net/ceded rebucketing scheme
               ('linear' or 'nearest'); defaults to the current ``self.reins_bucket``.
        :param dsev_bucket: optional override of the discrete-severity atom
               placement scheme ('linear' or 'nearest'); defaults to the current
               ``self.dsev_bucket``. See :attr:`dsev_bucket`.
        :param debug: run reinsurance in debug model if True.
        :param x_min: ``None`` requests an automatic two-sided output window
          (NYI in this stage -- treated as the grid origin ``xs[0]``);
          otherwise informational (the grid origin is read from ``xs[0]``).
        :param x_max: informational upper window edge.
        :return:
        """
        self._density_df = None  # invalidate
        self._sev_density_df = None
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_view_stats_cache = None
        self._reins_describe = None
        self._occ_joints = {}
        self._dist = None
        self._sev_dist = None
        self._valid = None
        self._deficit = np.nan
        self.sev_calc = sev_calc
        self.discretization_calc = discretization_calc
        self.normalize = normalize
        self.padding = padding
        if reins_bucket is not None:
            # validating setter; takes effect for the reins applied below
            self.reins_bucket = reins_bucket
        if dsev_bucket is not None:
            # validating setter; takes effect for the discretization below
            self.dsev_bucket = dsev_bucket
        self.xs = xs
        # bs is the grid step; xs[1]-xs[0] (not xs[1]) so a signed/offset grid
        # whose origin xs[0] != 0 still reports the correct bucket size.
        self.bs = xs[1] - xs[0]
        self.log2 = int(np.log2(len(xs)))
        # Output-window origin: the physical value at output index 0. For the
        # default 0-based grid this is 0 and all the offset machinery below is
        # inert (i0 == 0, j0 == 0 -> no rolls; xs_sev == xs).
        self.x_min = float(xs[0])
        self.x_max = float(xs[-1])

        # F1 -- negative-support severity. Resolve the signed opt-in (auto for
        # discrete severities with negative atoms; explicit flag otherwise),
        # then determine the severity's negative reach as a whole number of
        # buckets ``i0`` (the index of physical 0 in the severity array). Zero
        # for any severity supported on [0, inf), so the non-negative path is
        # untouched. The severity is then discretised on its own grid
        # ``xs_sev`` (physical 0 at index i0), which may differ from the output
        # grid ``xs`` (e.g. a tight far-from-0 output window).
        self._signed_sev = self._signed_severity()
        self.i0 = self._severity_negative_buckets(self.bs)
        self.xs_sev = (np.arange(len(xs), dtype=float) - self.i0) * self.bs

        # claim-count weighted severity vector (always computed; FFT is the only path)
        freq_ex1 = self.stats_df.loc[('freq', 'ex1'), self._comp_cols].values
        wts = freq_ex1 / freq_ex1.sum()
        if self.en.sum() == 0:
            self.en = freq_ex1
        self.sev_density = np.zeros_like(xs)
        beds = self.discretize(sev_calc, discretization_calc, normalize)
        for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
            self.sev_density += temp * w

        # adjust for picks if necessary
        if self.sev_pick_attachments is not None:
            logger.warning('Adjusting for picks.')
            self.sev_density = self.picks(self.sev_pick_attachments, self.sev_pick_losses)

        if force_severity == 'yes':
            # only asking for severity (used by plot)
            return

        # deal with per occ reinsurance
        if self.occ_reins is not None:
            if self.sev_density_gross is not None:
                # re-applying reins on an already-updated object: restore
                # gross sev so apply_occ_reins is idempotent
                self.sev_density = self.sev_density_gross
            self.apply_occ_reins(debug)

        self._freq_sev_convolution(padding)
        if self.n > 0:
            # zero-risk case has no aggregate to reinsure
            self.apply_agg_reins(debug)

        # Empirical severity moments from the discretised distribution.
        # Compute the raw moments once and derive (mean, cv, skew) from the
        # same MomentWrangler, so the ex123 rows and the mcvsk values are
        # mutually consistent.
        if self.sev_density is not None:
            # severity lives on xs_sev (== xs on the default grid)
            _mw = xsden_to_mwrangler(self.xs_sev, self.sev_density)
            sev_ex1, sev_ex2, sev_ex3 = _mw.noncentral
            self.est_sev_m, self.est_sev_cv, self.est_sev_skew = _mw.mcvsk
            # var/sd straight off the wrangler (var = central[1]), never via
            # mean*cv -- see the theoretical site above for the mean-0 rationale.
            self.est_sev_var = max(float(_mw.central[1]), 0.0)
            self.est_sev_sd = math.sqrt(self.est_sev_var)
        else:
            sev_ex1 = sev_ex2 = sev_ex3 = np.nan
            self.est_sev_m = np.nan
            self.est_sev_cv = np.nan
            self.est_sev_skew = np.nan
            self.est_sev_var = np.nan
            self.est_sev_sd = np.nan

        # Empirical aggregate moments from the FFT output.
        #
        # WHY a de-fuzzed *copy*: the raw inverse-FFT density carries
        # sub-machine-epsilon "fuzz" (tiny +/- values) in essentially every
        # bucket. In the plain mass sum this cancels (mass is conserved), but
        # the raw moments weight each bucket by ``x**k``, so on a wide grid the
        # far-tail fuzz at large ``x`` is amplified by ``x**3`` and corrupts
        # the empirical skew -- e.g. a symmetric die's skew drifts from ~1e-15
        # to ~1e-4 as log2 grows, purely from fuzz at buckets the distribution
        # never reaches. The fuzz is genuine fp noise: the FFT is exact up to
        # rounding and the exact aggregate has no negative density even under
        # aliasing (aliasing only wraps *positive* mass), so every stray value
        # is small and zeroing ``|x| < eps`` is safe and lossless. We do this
        # on a throwaway copy and deliberately leave ``self.agg_density`` as
        # the raw output (kept consistent with ``ftagg_density``); the curated
        # view ``density_df.p_total`` applies the identical ``remove_fuzz``
        # separately. (We cannot source the moments from ``density_df.p_total``
        # here: building ``density_df`` needs ``est_m``, computed just below.)
        agg_clean = remove_fuzz(self.agg_density)
        _mw = xsden_to_mwrangler(self.xs, agg_clean)
        agg_ex1, agg_ex2, agg_ex3 = _mw.noncentral
        self.est_m, self.est_cv, self.est_skew = _mw.mcvsk
        # var/sd straight off the wrangler (var = central[1]), never via
        # mean*cv -- correct for a mean-0 signed aggregate (see above).
        self.est_var = max(float(_mw.central[1]), 0.0)
        self.est_sd = math.sqrt(self.est_var)

        # Write empirical and error columns into the canonical stats_df.
        # This is the validation showpiece of Mildenhall 2024, §4.7:
        # theoretical (``mixed`` column) vs. empirical (FFT output).
        self.stats_df.loc[('sev', 'ex1'),  'empirical'] = sev_ex1
        self.stats_df.loc[('sev', 'ex2'),  'empirical'] = sev_ex2
        self.stats_df.loc[('sev', 'ex3'),  'empirical'] = sev_ex3
        self.stats_df.loc[('sev', 'mean'), 'empirical'] = self.est_sev_m
        self.stats_df.loc[('sev', 'cv'),   'empirical'] = self.est_sev_cv
        self.stats_df.loc[('sev', 'skew'), 'empirical'] = self.est_sev_skew
        self.stats_df.loc[('agg', 'ex1'),  'empirical'] = agg_ex1
        self.stats_df.loc[('agg', 'ex2'),  'empirical'] = agg_ex2
        self.stats_df.loc[('agg', 'ex3'),  'empirical'] = agg_ex3
        self.stats_df.loc[('agg', 'mean'), 'empirical'] = self.est_m
        self.stats_df.loc[('agg', 'cv'),   'empirical'] = self.est_cv
        self.stats_df.loc[('agg', 'skew'), 'empirical'] = self.est_skew

        # Defective-distribution check. The aggregate FFT loses mass off the
        # right end of the grid when log2 is too small, and a deficit makes
        # forwards and backwards S diverge by exactly that amount in
        # Distortion.price, so one answer silently differs from another
        # downstream.
        #
        # The gate is DEFICIT_MATERIALITY (1e-4), the same floor
        # choquet_weights uses to separate an economic problem from FFT
        # truncation -- not VALIDATION_NOISE (1e-12), which measures
        # arithmetic dust and fires on deficits no one can act on. Every
        # object records its own verdict either way (Validation.DEFECTIVE);
        # this is only about whether to interrupt.
        #
        # warn_once, so a sweep that rebuilds the same shape 64 times reports
        # one fault rather than 64 copies of it.
        #
        # When the sizer already WARNED about a far-tail clip, that warning is
        # the same mass with actionable advice (the exact log2 to raise to),
        # so we do not double-warn here. Note the test is "did the sizer
        # warn", not "is ``_bs_clip`` set": ``_bs_clip`` is recorded for the
        # bs report at any size, and an immaterial estimate there must not
        # silence a material measured deficit here.
        clip = self._bs_clip
        clip_warned = clip is not None and (
            not np.isfinite(clip.get('clipped_mass', np.nan))
            or clip['clipped_mass'] > DEFICIT_MATERIALITY)
        # Recorded on the object here, where it is already in hand, rather
        # than lazily inside ``valid``: the deficit is a fact about the
        # update, and reading it should not depend on having asked for a
        # validation verdict first.
        self._deficit = deficit = _validation.pmf_deficit(self)
        if deficit > DEFICIT_MATERIALITY and not clip_warned:
            warn_once(
                f'{self.name}: aggregate PMF deficit {deficit:.3e} '
                f'(Σp = 1 − {deficit:.3e} < 1); forwards and backwards '
                f'S diverge by the deficit (forwards > backwards).',
                DefectiveDistributionWarning,
                key='defective-construction', stacklevel=3)

        # Staged reinsurance reporting -- §1.2 of the aggregate refactor plan.
        #
        # ``empirical`` above is the final (after-occ + after-agg) realised
        # view. To express the Subject -> after-occ -> after-agg progression
        # we also need the subject (gross) empirical moments and -- when
        # reinsurance is present -- the intermediate after-occ moments.
        # Validation continues to use the subject vs theoretical comparison,
        # which is the only apples-to-apples check available under reins.
        has_occ = self.occ_reins is not None
        has_agg = self.agg_reins is not None

        # Subject severity density: when occ-reins applied, the pre-reins
        # severity is preserved on ``sev_density_gross``; otherwise the
        # current ``sev_density`` IS gross.
        subject_sev = self.sev_density_gross if has_occ else self.sev_density
        # After-occ severity is whatever the occ stage passed along
        # (= ``sev_density`` post ``apply_occ_reins``); identical to subject
        # severity when there is no occ stage.
        after_occ_sev = self.sev_density

        # Subject aggregate: with occ-reins we need one extra FFT of the
        # gross severity (the "validate the subject" hook in §1.3); with no
        # occ-reins the pre-agg-reins density already encodes gross
        # (``agg_density_gross`` when has_agg, else the final
        # ``agg_density``).
        if has_occ:
            subject_agg, _ = self._fft_aggregate(subject_sev, padding)
        elif has_agg:
            subject_agg = self.agg_density_gross
        else:
            subject_agg = self.agg_density

        # After-occ aggregate (pre-agg-reins): when an agg stage exists,
        # ``apply_agg_reins`` stored the pre-stage density in
        # ``agg_density_gross``; with only occ-reins the final
        # ``agg_density`` IS the after-occ density; with no reins there is
        # no separate stage to report.
        if has_agg:
            after_occ_agg = self.agg_density_gross
        elif has_occ:
            after_occ_agg = self.agg_density
        else:
            after_occ_agg = None

        # De-fuzzed moment helper: same |x| < eps zeroing the main
        # empirical block uses (see WHY comment above), wrapped so we can
        # reuse it on subject / after-occ densities.
        def _moments(arr, grid):
            # ``grid`` is xs_sev for severity densities, xs for aggregates
            # (they coincide on the default 0-based grid).
            if arr is None:
                return (np.nan,) * 6
            mw = xsden_to_mwrangler(grid, remove_fuzz(arr))
            return (*mw.noncentral, *mw.mcvsk)

        sub_sev_mom = _moments(subject_sev, self.xs_sev)
        sub_agg_mom = _moments(subject_agg, self.xs)
        self._write_stage_moments('gross_empirical', sub_sev_mom, sub_agg_mom,
                                  copy_freq_from='empirical')

        if has_occ or has_agg:
            aft_sev_mom = _moments(after_occ_sev, self.xs_sev)
            aft_agg_mom = _moments(after_occ_agg, self.xs)
            self._write_stage_moments('after_occ', aft_sev_mom, aft_agg_mom,
                                      copy_freq_from='empirical')

        # Per-stage impact ratios (after / before). 1.0 means no impact;
        # written only when the corresponding stage exists, so consumers can
        # detect "stage absent" by NaN.
        if has_occ:
            self.stats_df['occ_impact'] = (
                self.stats_df['after_occ'] / self.stats_df['mixed'])
        if has_agg:
            self.stats_df['agg_impact'] = (
                self.stats_df['empirical'] / self.stats_df['after_occ'])

        # ``error`` is the SUBJECT validation: gross_empirical vs mixed.
        # With no reinsurance gross_empirical == empirical and this is
        # exactly the legacy theoretical-vs-empirical column. With reins it
        # is the only apples-to-apples check (the after-reins object has no
        # independent theoretical to validate against).
        self.stats_df['error'] = _noise_aware_rel_error(
            self.stats_df['gross_empirical'], self.stats_df['mixed'])

        # invalidate stored functions
        self._cdf = None

    def _fft_aggregate(self, sev_density, padding):
        """Run one FFT convolution: severity density -> aggregate density.

        Single source of truth for the FFT-PGF-iFFT core (Mildenhall 2024,
        §2.2). Used by ``_freq_sev_convolution`` (the main per-update path),
        by ``update_work`` for the subject (gross) aggregate when occ-reins
        is present, and by ``reins_density_df`` to compute gross/ceded/net
        aggregates from the corresponding severities. The zero-risk and
        one-claim shortcuts live here so every caller sees them consistently.

        Parameters
        ----------
        sev_density : np.ndarray
            Discretised severity on ``self.xs_sev`` (gross, net, or ceded);
            ``sev_density[j]`` is the mass at physical ``(j - i0) * bs``.
        padding : int
            FFT padding factor passed to ``ft`` / ``ift``.

        Returns
        -------
        agg_density : np.ndarray
            Aggregate density on the output grid ``self.xs``.
        ftagg_density : np.ndarray
            FT of the aggregate (padded length). Callers that don't need
            this (e.g. ``reins_density_df``) discard it.

        Notes
        -----
        Two paths, selected by whether any offset is active:

        - **Default (``i0 == 0`` and output window origin ``x_min == 0``).**
          The original ``ft`` / ``freq_pgf`` / ``ift`` path, unchanged and
          byte-for-byte identical to prior releases.
        - **Signed / windowed (F1 + F2).** Negatives live at the top of the
          padded length-``M = N << padding`` FFT buffer (period ``M*bs``); the
          severity is laid in with physical 0 at index 0 (``i0`` negative
          buckets wrapped to the top). After the FFT the result is *relabelled*
          onto the output window by a single ``np.roll`` of ``-round(x_min/bs)``
          and the first ``N`` buckets kept. Relabelling a finished, exact array
          carries no ``N·s`` shift term, so this is correct for random as well
          as fixed frequency (the key F2 clarification, plan §2). Exact when the
          aggregate support width ``W < M*bs``; a window narrower than the
          support shows up as a two-sided deficit.
        """
        # Thin wrapper over the extracted pure kernel (Phase 2A); all the
        # self-state it needs is passed explicitly so the core is testable
        # without a full update(). See aggregate._aggregate_compute.
        return freq_sev_convolution(
            sev_density, self.frequency.freq_pgf, self.base_mean,
            N=len(self.xs), bs=self.bs, i0=self.i0, x_min=self.x_min,
            one_claim=self.one_claim, padding=padding)

    def _write_stage_moments(self, col, sev_mom, agg_mom, copy_freq_from=None):
        """Write a moment tuple into a single ``stats_df`` column.

        Helper to keep the staged-empirical writes in ``update_work`` tidy.
        Each moment tuple is the six values ``(ex1, ex2, ex3, mean, cv,
        skew)`` returned by ``xsden_to_mwrangler``.

        Parameters
        ----------
        col : str
            Destination column label (``empirical``, ``after_occ``,
            ``gross_empirical``, ...).
        sev_mom, agg_mom : tuple of float
            Six-tuple raw + central moments for the sev and agg rows.
        copy_freq_from : str or None
            If set, mirror the freq mean/cv/skew rows from another column
            (freq is unchanged by either reinsurance stage).
        """
        _measures = ('ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew')
        for measure, value in zip(_measures, sev_mom):
            self.stats_df.loc[('sev', measure), col] = value
        for measure, value in zip(_measures, agg_mom):
            self.stats_df.loc[('agg', measure), col] = value
        if copy_freq_from is not None:
            src = self.stats_df[copy_freq_from]
            for measure in ('mean', 'cv', 'skew'):
                self.stats_df.loc[('freq', measure), col] = src[('freq', measure)]

    def _freq_sev_convolution(self, padding):
        """Compute the aggregate density by FFT convolution (Mildenhall 2024, §2.2).

        Thin wrapper that routes the (post-occ-reins) ``sev_density`` through
        ``_fft_aggregate`` and writes ``self.agg_density`` /
        ``self.ftagg_density``. The FFT core, zero-risk and one-claim shortcuts
        all live in ``_fft_aggregate``.

        Parameters
        ----------
        padding : int
            Padding factor passed to ``ft`` / ``ift`` to mitigate FFT aliasing
            (see Mildenhall 2024, §2.3.2).

        Notes
        -----
        Per-occurrence reinsurance is applied to ``sev_density`` *before* this
        method is called; aggregate reinsurance is applied to ``agg_density``
        *after*. The FFT here is unaware of either.
        """
        self.agg_density, self.ftagg_density = self._fft_aggregate(
            self.sev_density, padding)

    # ================================================================
    # Validation (paper §4.7), unwrap, picks, freq_pmf
    # ================================================================

    @property
    def valid(self):
        """
        Check if the model appears valid. An answer of True means the model is "not unreasonable".
        It does not guarantee the model is valid. On the other hand,
        False means it is definitely suspect. (The interpretation is similar to the null hypothesis
        in a statistical test).
        Called and reported automatically by qd for Aggregate objects.

        Checks the relative errors (from the canonical ``stats_df``) for:

        * severity mean < eps
        * severity cv < 10 * eps
        * severity skew < 100 * eps (skewness is more difficult to estimate)
        * aggregate mean < eps.
        * the convolution residual < ``ALIASING_EPS``
          (:func:`aggregate._validation.convolution_residual`, the ``ALIASING``
          test), whenever the pmf deficit is dust.
        * aggregate cv < 10 * eps
        * aggregate skew < 100 * esp

        The default uses eps = 1e-4 relative error. This can be changed by
        setting the ``validation_eps`` variable.

        All reads come from ``stats_df`` -- the single source of truth -- not
        ``validation_df`` (display).

        The CV and skew tests are applied only when the theoretical value is
        finite and its magnitude exceeds ``VALIDATION_NOISE`` -- a
        theoretically-zero skew (symmetric severity) or CV (deterministic
        severity) is skipped, because the FFT's empirical estimate of a zero
        higher moment is grid-dependent noise with no meaningful relative
        error. When the test applies, ``np.isclose`` with relative tolerance
        ``10*eps`` (CV) / ``100*eps`` (skew, harder to estimate) measures
        agreement.

        The ALIASING test measures the convolution step directly rather than
        comparing two mean errors, and fires only when the pmf deficit is
        arithmetic dust. Wrap conserves mass while moving the mean; truncation
        drops mass, which DEFECTIVE and AGG_MEAN already own. See
        :func:`aggregate._validation.convolution_residual` and
        ``dev/done/plan-validation-punchup.md``.

        Run with logger level 20 (info) for more information on failures.

        A Type 1 error (rejecting a valid model) is more likely than Type 2 (failing to reject an invalide one).

        :return: True (interpreted as not unreasonable) if all tests are passed, else False.

        """
        return _validation.valid_aggregate(self)

    def unwrap(self, p=1e-7, audit=True):
        """
        Unwrap self created with log2 that is too small to contain the answer.

        :param p: Percentile threshold. The estimated p and 1-p quantiles are
            used to determine the effective support [L, R]. R-L must fit in the
            space available, i.e., R-L <= N * self.bs.
        :param audit: If audit, return comparison of empirical moments of shifted
            answer with a.actual_m etc. analytic moments.
        :return: Unwrap named tuple containing fields y the density as a Series,
            mode of shifting/unwrapping, prob_captured the probability in the
            effective support (which should be close to 1), L, R the boundary of the
            effective support.
        """
        # figure bounds from method of moments estimates
        m, cv, skew = self.actual_m, self.actual_cv, self.actual_skew
        sc = self.bs
        L, R = _estimate_agg_percentile(m, cv, skew, p=(p, 1 - p))
        # snap to grid in both cases (can't use self.snap because outside index!)
        L = int(np.round(L / sc, 0))
        R = int(np.round(R / sc, 0))

        # number of buckets
        N = 1 << self.log2

        # is the request reasonable?
        # enough space condition: R - L <= N
        assert R - L <= N, f'{R=} - {L=} = {R-L=} > {N=}, not enough space'

        # how many "blocks" to the right are we?
        l = L // N
        r = R // N

        # extract aliased density
        y = self.density_df.p_total.values

        # there are now two cases: dist fits within one block or wraps over two
        # if it falls over more than two that is an error
        if l == r:
            # no unwrapping, range lies in one block
            # just shift index to right by correct number of chunks
            # locate correct left hand edge, index created below
            L = (L // N) * N
            # method reporting
            mode = 'Shift only'  # \n{L=}'
        elif l == r - 1:
            # must wrap answer into one block and shift
            # figure location of extreme points as remainders
            rem_r = R % N       # right hand end in fft-wrapped coords
            rem_l = L % N
            # by math this will always be true (see blog post)
            assert rem_l >= rem_r
            # unwrap amount
            roll_forward = N - (rem_l + rem_r) // 2
            y = np.roll(y, roll_forward)
            # shifted index, factoring in unwrap
            L = (L // N + 1) * N - roll_forward
            # method reporting
            mode = 'Shift and wrap'  # \n{roll_forward=}, {L=}'
        else:
            # see blog post
            print(f'Should not occur: {l=}, {r=}')

        # align with index and create answer
        i = np.arange(L, L + N, dtype=float) * sc
        ans = pd.Series(y, index=i)
        # apply scale to L and R now to match ans
        L *= sc
        R *= sc
        # document proportion of probability in selected range
        prob_captured = ans[L:R].sum()
        # package results
        Unwrap = namedtuple('Unwrap', 'y, mode, prob_captured, L, R, audit_df')
        if audit:
            em, ecv, eskew = xsden_to_meancvskew(ans.index, ans)
            audit_df = pd.DataFrame(
                {'m': [m, em],
                 'cv': [cv, ecv],
                 'skew': [skew, eskew]},
                index=['actual', 'rewrapped'])

        else:
            audit_df = None
        ans = Unwrap(ans, mode, prob_captured, L, R, audit_df)
        return ans

    def picks(self, attachments, layer_loss_picks, debug=False):
        """
        Adjust the computed severity to hit picks targets in layers defined by a.
        Delegates work to :func:`_picks_work`. See that function for details.

        Raises
        ------
        ValueError
            If any attachment misses the realized grid or lies above the top of
            the window. The message names the offenders and a bucket size that
            would work.

        Notes
        -----
        **Every attachment must be a grid point.** Picks defines the layers the
        reweighting is solved on, so a boundary strictly inside a bucket has no
        faithful reading: that bucket's mass sits at a single point and cannot
        be divided between the layer below and the layer above without
        inventing a within-bucket distribution. The update therefore raises a
        ``ValueError`` naming a compatible bucket rather than snapping the
        boundary, which would silently restate the tower the caller asked for
        (author ruling 2026-08-24).

        A reinsurance tower over the same attachments is exempt, and the
        contrast is the point. ``apply_reins_work`` evaluates piecewise linear
        ceder and netter functions at every grid point and lands the off grid
        results back on the lattice, which is faithful because a contract is a
        function of the loss and can be evaluated anywhere. A pick is a
        constraint on an integral between two boundaries, so the boundaries
        have to exist on the grid.

        The bucket sizer is deliberately blind to picks: an off grid result
        under auto sizing is an error naming a compatible bucket, not an input
        to ``_bs_window``.
        """
        # always want to work off gross severity
        if self.sev_density_gross is not None:
            logger.info('Using GROSS severity in picks')
            sd = self.sev_density_gross
        else:
            sd = self.sev_density
        return _picks_work(attachments, layer_loss_picks, self.xs, sd, n=self.n,
                          sf=self.sev.sf, debug=debug)

    def freq_pmf(self, log2):
        """
        Return the frequency probability mass function (pmf) computed using 2**log2 buckets.
        Uses self.en to compute the expected frequency. The :class:`Frequency` does not
        know the expected claim count, so this is a method of :class:`Aggregate`.

        """
        n = 1 << log2
        z = np.zeros(n)
        z[1] = 1
        fz = ft(z, 0)
        fz = self.frequency.freq_pgf(self.en, fz)
        dist = ift(fz, 0)
        # remove fuzz -- intentionally ONE-SIDED (zeroes negatives too); a
        # frequency pmf has no legitimate negatives, so this is NOT the shared
        # two-sided ``remove_fuzz`` utility.
        dist[dist < np.finfo(float).eps] = 0
        # ``en`` holds the per-component BASE counts, which is what freq_pgf
        # wants; under zm / zt ``n`` is legitimately the shifted realized mean,
        # so only flag a mismatch when the frequency is unmodified.
        if not self._freq_zm and not np.allclose(self.n, self.en):
            logger.warning('Frequency.pmf | n %s != en %s; using en', self.n, self.en)
        return dist

    # ================================================================
    # Reinsurance application: occ pre-FFT, agg post-FFT
    # ================================================================

    def _rebucket_to_grid(self, values, mass, scheme=None, origin=None):
        """Scatter off-grid ``mass`` at target ``values`` onto the model grid.

        The model grid is ``self.xs == bs * arange`` with ``xs[0] == 0``, so
        the (fractional) grid index of a value ``v`` is ``v / bs``. Used to
        place reinsurance net/ceded values back on the grid after the cession
        map moves them off it, and to place discrete-severity atoms on the grid
        during discretization (see :meth:`discretize`).

        Parameters
        ----------
        values : ndarray
            Target loss values (net/ceded points, or severity atoms).
        mass : ndarray
            Probability mass to redistribute, aligned with ``values``.
        scheme : {'linear', 'nearest'}, optional
            Placement scheme. Defaults to :attr:`reins_bucket` (so the
            reinsurance call sites are unchanged); the discrete-severity call
            site passes :attr:`dsev_bucket`.
        origin : float, optional
            Physical value at output index 0 -- the grid origin used to map a
            value to its (fractional) bucket index ``(v - origin) / bs``.
            Defaults to :attr:`x_min` (the output-window origin), correct for
            the reinsurance call sites. The discrete-severity call site passes
            the *severity* grid origin ``xs_sev[0] == -i0·bs``, which differs
            from ``x_min`` when the output window is forced wider than the
            severity support (e.g. an explicit ``x_min`` below the smallest
            atom).

        Returns
        -------
        ndarray
            Probability vector on ``self.xs`` (length ``self.n``).

        Notes
        -----
        Two schemes:

        - ``'nearest'`` rounds each value to its closest bucket. Full mass
          lands in one bucket, with up to ``bs/2`` positional bias.
        - ``'linear'`` splits each value's mass between its two bracketing
          buckets ``k`` and ``k+1`` with weights ``1-f`` and ``f`` where
          ``f = v/bs - k``. Because ``(1-f)·k·bs + f·(k+1)·bs == v``, the
          first moment is preserved **exactly**; both schemes preserve total
          mass (``Σp == Σmass``).

        Values at or beyond the top of the grid (``xs[-1]``) pile into the
        last bucket -- the same overflow mode as an aggregate deficit, and
        surfaced the same way. Negative targets (should not occur for valid
        cessions) clip into bucket 0.
        """
        bs = self.bs
        n = len(self.xs)
        if origin is None:
            origin = self.x_min
        # Grid index of a value v is (v - origin) / bs; origin == 0 on the
        # default grid recovers the original v / bs.
        scaled = (np.asarray(values, dtype=float) - origin) / bs
        out = np.zeros(n)
        if scheme is None:
            scheme = self.reins_bucket
        if scheme == 'nearest':
            idx = np.clip(np.round(scaled).astype(int), 0, n - 1)
            np.add.at(out, idx, mass)
        else:  # 'linear' -- mass split preserves E[X] exactly
            k = np.clip(np.floor(scaled).astype(int), 0, n - 1)
            f = np.clip(scaled - k, 0.0, 1.0)
            kp1 = np.clip(k + 1, 0, n - 1)
            np.add.at(out, k, mass * (1 - f))
            np.add.at(out, kp1, mass * f)
        return out

    def _apply_reins_work(self, reins_list, base_density, debug=False):
        """
        Actually do the work. Called by apply_reins.
        Only needs self to get limits, which it must guess without q (not computed
        at this stage). Does not need to know if occ or agg reins,
        only that the correct base_density is supplied.

        :param reins_list:
        :param kind: occ or agg, for debug plotting
        :param debug:
        :return: ceder, netter,
        """
        return _reinsurance.apply_reins_work(self, reins_list, base_density, debug)

    def apply_occ_reins(self, debug=False):
        """
        Apply the entire occ reins structure and save output
        For by layer detail see reins_stats_df.
        Makes sev_density_gross, sev_density_net and sev_density_ceded, and updates sev_density to the requested view.

        Not reflected in statistics df.

        :param debug: More verbose.
        :return:
        """
        return _reinsurance.apply_occ_reins(self, debug)

    def apply_agg_reins(self, debug=False, padding=1):
        """
        Apply the entire agg reins structure and save output.
        For by layer detail see reins_stats_df.
        Makes agg_density_gross, agg_density_net and agg_density_ceded, and
        updates agg_density to the requested view.

        Not reflected in statistics df: the post-reins empirical moments
        (``est_*`` and the ``stats_df['empirical']`` column) are written by
        ``update_work`` from the same density updated here.

        :return:
        """
        return _reinsurance.apply_agg_reins(self, debug, padding)

    @property
    def reins_description(self):
        """
        Short narrative description of the reinsurance (str).

        The consistent narrative surface, mirroring ``tail_description`` /
        ``bs_description``: returns the ``kind='both', width=0`` text. For the
        parameterized form (a single kind, or wrapped to a width) use the
        private worker :meth:`_reins_description`.
        """
        return self._reins_description(kind='both', width=0)

    @property
    def reins_explanation(self):
        """
        Long narrative of the reinsurance: the terms, then what they do (str).

        The verbose half of the pair (a172 [FCC-Contract-Gaps]).
        :attr:`reins_description` says what the program *declares*;  this adds
        what the cession *does*, read off :attr:`reins_summary_df`: expected loss
        gross, ceded and net at each stage, and the share of the gross that the
        cession carries.

        Returns
        -------
        str
            ``'No reinsurance.'`` on a clean book, so the caller never has to
            branch on :attr:`reins_kinds` first.

        Notes
        -----
        Occurrence and aggregate are reported as separate stages because they are
        applied in sequence: the aggregate cover attaches to the *subject*, which
        is the occurrence stage's net, not to the gross. Reading a single
        "ceded" number across both stages is the standard way to double count.
        """
        if self.reins_kinds == 'None':
            return 'No reinsurance.'
        # reins_description is not consistently sentence-terminated (the
        # occurrence-only form has no full stop, the aggregate form does), and
        # the Portfolio look-through already works around that with rstrip.
        # Terminate it here rather than changing an asserted string.
        desc = self.reins_description.rstrip('.')
        out = [f'{desc}.']
        df = self.reins_summary_df
        if df is None:
            return ' '.join(out)

        def _agg_ex(stage, view):
            try:
                return float(df.loc[(stage, view, 'agg'), 'Est EX'])
            except KeyError:
                return None

        for stage, base_view, base_word in (('occ', 'gross', 'Gross'),
                                            ('agg', 'subject', 'Subject')):
            base = _agg_ex(stage, base_view)
            ceded, net = _agg_ex(stage, 'ceded'), _agg_ex(stage, 'net')
            if base is None or ceded is None or net is None:
                continue
            share = f'{ceded / base:.1%}' if base else 'n/a'
            where = ('Per occurrence' if stage == 'occ' else 'In the aggregate')
            out.append(f'{where}, {base_word.lower()} expected loss '
                       f'{base:,.6g} splits into {ceded:,.6g} ceded ({share}) '
                       f'and {net:,.6g} net.')
        return ' '.join(out)

    def _reins_description(self, kind='both', width=0):
        """
        Text description of the reinsurance (parameterized worker).

        :param kind: both, occ, or agg
        :param width: width of text for textwrap.fill; omitted if width==0
        """
        return _reinsurance.reins_description(self, kind, width)

    @property
    def reins_kinds(self):
        """Text description of kinds of reinsurance applied.

        Returns
        -------
        str
            One of ``'None'``, ``'Occurrence only'``, ``'Aggregate only'``, or
            ``'Occurrence and aggregate'``.
        """
        return _reinsurance.reins_kinds(self)

    # ================================================================
    # Distortion, ruin theory, plotting
    # ================================================================

    def apply_distortion(self, dist, *, view='ask', S_calculation='forwards',
                         allow_deficit=False):
        r"""
        Apply distortion to the aggregate density; appends ``gS``,
        ``gp_total`` and ``exag`` columns to ``density_df``.

        Routes through the exact-discrete Choquet helper
        (:func:`~aggregate.spectral.choquet_weights`):
        ``exag(a) = rho_g(X ∧ a) = Σ_{x≤a} x·gp + a·g(S(a))``, a direct
        sum carrying the origin -- valid on windowed and signed supports
        (the old ``cumsum(gS)·bs`` idiom assumed a zero-origin grid). The
        effective ``g`` resolves ``view`` × the object's value-type role
        (:meth:`~aggregate.spectral.Distortion.effective_g`), so a
        payoff-role aggregate prices through the dual automatically.

        Parameters
        ----------
        dist : Distortion
            The distortion to apply.
        view : {'ask', 'bid'}
            Pricing view; composes with the value-type role by XOR.
        S_calculation : {'forwards', 'backwards'}
            Deficit-parking direction; see
            :func:`~aggregate.spectral.choquet_weights`.
        allow_deficit : bool
            Explicit truncation policy for a materially defective pmf.
            Default False raises
            :class:`~aggregate.constants.DefectiveDistributionError`.

        Raises
        ------
        ValueError
            For a mass distortion on an unbounded support: the mass lands
            on the last represented bucket, which is a different bounded
            problem, not an approximation. Certify ``self.bounded = True``
            if the support is in fact bounded.
        """
        if self.agg_density is None:
            logger.warning('You must update before applying a distortion ')
            return
        if getattr(dist, 'has_mass', False) and not self.bounded:
            raise ValueError(
                f'mass distortion ({dist.label}) on an unbounded aggregate: '
                f'the mass lands on the last represented bucket, a '
                f'different bounded problem. Certify `bounded = True` if '
                f'the support is in fact bounded.')

        g, _, _ = dist.effective_g(view, is_loss_value=self._is_loss_value)
        x = self.density_df.loss.to_numpy()
        p = self.density_df.p_total.to_numpy()
        w = choquet_weights(x, p, g, S_calculation=S_calculation,
                            allow_deficit=allow_deficit)
        self.density_df['gS'] = w.gS
        self.density_df['gp_total'] = w.gp
        # exag(a) = rho(X ∧ a); the strict-tail gp sum telescopes to g(S(a))
        self.density_df['exag'] = np.cumsum(x * w.gp) + x * w.gS

    def pollaczeck_khinchine(self, rho, cap=0, excess=0, stop_loss=0, kind='index', padding=1):
        """
        Return the Pollaczeck-Khinchine Capital function relating surplus to eventual probability of ruin.
        Requires a Poisson frequency (raises ``ValueError`` otherwise); for a
        renewal (wait-clause) frequency use :meth:`wiener_hopf`.

        See Embrechts, Kluppelberg, Mikosch 1.2, page 28 Formula 1.11

        :param rho: rho = prem / loss - 1 is the margin-to-loss ratio
        :param cap: cap = cap severity at cap, which replaces severity with X | X <= cap
        :param excess:  excess = replace severity with X | X > cap (i.e. no shifting)
        :param stop_loss: stop_loss = apply stop loss reinsurance to cap, so  X > stop_loss replaced
          with Pr(X > stop_loss) mass
        :param kind:
        :param padding: for update (the frequency tends to be high, so more padding may be needed)
        :return: :class:`RuinFunction` named tuple ``(ruin, find_u, mean, density)``:
          ruin vector as pd.Series, function to lookup capitals (no interpolation
          if kind==index; else interp), discretized severity mean, and the
          integrated-severity (equilibrium) density ``dfi``. Unpacks positionally
          like the historical bare tuple.
        """
        fname = getattr(self.frequency, 'freq_name', '')
        if fname != 'poisson':
            hint = (' -- for a renewal (wait-clause) frequency use wiener_hopf'
                    if fname == 'renewal' else '')
            raise ValueError(
                f'pollaczeck_khinchine assumes a Poisson frequency, '
                f'got {fname!r}{hint}')
        if self.sev_density is None:
            raise ValueError("Must recalc before computing Cramer Lundberg distribution.")

        bit = self.sev_density_df.p_sev.copy()
        if cap:
            idx = np.searchsorted(bit.index, cap, 'right')
            bit.iloc[idx:] = 0
            bit = bit / bit.sum()
        elif excess:
            # excess may not be in the index...
            idx = np.searchsorted(bit.index, excess, 'right')
            bit.iloc[:idx] = 0
            bit = bit / bit.sum()
        elif stop_loss:
            idx = np.searchsorted(bit.index, stop_loss, 'left')
            xsprob = bit.iloc[idx + 1:].sum()
            bit.iloc[idx] += xsprob
            bit.iloc[idx + 1:] = 0
        mean = np.sum(bit * bit.index)

        # integrated F function
        fi = bit.shift(-1, fill_value=0)[::-1].cumsum()[::-1].cumsum() * self.bs / mean
        # difference = probability density
        dfi = np.diff(fi, prepend=0)
        # use loc FFT, with wrapping
        fz = ft(dfi, padding)
        mfz = 1 / (1 - fz / (1 + rho))
        f = ift(mfz, padding)
        f = np.real(f) * rho / (1 + rho)
        f = np.cumsum(f)
        ruin = pd.Series(1 - f, index=bit.index)
        find_u = _ruin_find_u(ruin, kind)
        return RuinFunction(ruin, find_u, mean, dfi)

    def _discretize_wait_pmf(self, c, n):
        """Discretize the renewal wait law, scaled by ``c``, on the money grid.

        Rounding pmf of the scaled waiting time ``cW`` on ``n`` buckets of
        width ``self.bs`` -- equivalently the wait law itself on time
        buckets of width ``bs / c``. Bucket 0 holds ``P(W <= h/2)`` (any
        zero-wait atom and collapsed negative mass ride along); bucket
        ``j`` holds ``P((j-1/2) h < W <= (j+1/2) h)`` with ``h = bs/c``;
        mass beyond the last edge is dropped (the caller's grid must be
        long enough for the drop to be negligible).

        Follows :func:`aggregate._renewal.wait_count_pmf` (the renewal
        count discretization) -- same :func:`discretize_severities` call
        and unconditional-window masking -- but with **no** truncation at
        the horizon ``T`` and **no** zero-wait removal: the ruin random
        walk needs the whole law. Used by :meth:`wiener_hopf` and by
        ``pedagogy.ruin_example`` (wait sampling for simulated paths).

        Parameters
        ----------
        c : float
            Premium rate per unit time (the scale factor).
        n : int
            Number of buckets.

        Returns
        -------
        ndarray, length n
            Weighted-combined pmf of ``cW`` on ``arange(n) * bs``. Sums to
            1 less the dropped tail (and any defect, which
            :meth:`wiener_hopf` refuses).
        """
        freq = self.frequency
        weights = np.atleast_1d(np.asarray(freq.wait_weights, dtype=float))
        bsw = self.bs / c
        xsw = np.arange(n, dtype=float) * bsw
        beds = discretize_severities(
            [sev for sev, *_ in freq.wait_components], xsw, bsw,
            sev_calc='discrete', discretization_calc='survival',
            normalize=False)
        fcw = np.zeros(n)
        for (sev, lb, ub, conditional), w_, bed in zip(
                freq.wait_components, weights, beds):
            bed = np.asarray(bed, dtype=float)
            if not conditional:
                # unconditional splice window, masked on the grid (mass-
                # neutral under the wiener_hopf defect guard; keeps the
                # wait_count_pmf pattern)
                eps = 1e-12 * max(1.0, abs(lb), abs(ub))
                bed[(xsw < lb - eps) | (xsw > ub + eps)] = 0.0
            fcw += w_ * bed
        return fcw

    def wiener_hopf(self, rho, kind='index', log2=None):
        r"""
        Eventual ruin probability for a Sparre-Andersen (renewal) model by
        cepstral Wiener-Hopf factorization.

        The renewal counterpart of :meth:`pollaczeck_khinchine`: the
        frequency must be a renewal (``years ... wait ...``) frequency, and
        the premium rate is ``c = (1 + rho) E[X] / E[W]`` -- the loaded
        long-run loss rate, so ``rho`` keeps its PK margin-to-loss meaning.

        Parameters
        ----------
        rho : float
            Margin-to-loss ratio: premium = ``(1 + rho)`` times the expected
            loss rate ``E[X] / E[W]``.
        kind : {'index', 'interpolate'}
            ``find_u`` lookup convention, as in :meth:`pollaczeck_khinchine`:
            ``'index'`` snaps to the grid, anything else interpolates.
        log2 : int, optional
            Half-grid exponent. The wrapped circle has ``2**(log2 + 1)``
            buckets and psi is returned on ``u = arange(2**log2) * bs``.
            Defaults to ``self.log2``, so the u-grid coincides with the
            severity grid (and with the PK u-grid). Must be at least
            ``self.log2``; raise it for heavy-tailed severities (see the
            wrap-around warning).

        Returns
        -------
        RuinFunction
            Named tuple ``(ruin, find_u, mean, density)``: ``psi(u)`` as a
            pd.Series on the u-grid, the capital-lookup closure, the
            discretized severity mean, and the pmf of the all-time maximum
            of the loss walk on the u-grid. Unpacks positionally exactly
            like :meth:`pollaczeck_khinchine`.

        Raises
        ------
        ValueError
            If the frequency is not renewal (for Poisson use
            :meth:`pollaczeck_khinchine`); if the object has not been
            updated; if the severity grid is signed; if the wait law is
            defective (terminating renewal process); or if the net profit
            condition ``E[Y] < 0`` fails on the grid (margin too small).

        Notes
        -----
        Sparre-Andersen surplus ``U(t) = u + ct - sum_{i <= N(t)} X_i`` with
        iid waits ``W``. Ruin can occur only at claim instants, so the
        problem collapses to the random walk with per-claim step
        ``Y = X - cW``: ``psi(u) = P(M > u)`` with ``M`` the all-time
        maximum of the partial sums -- the Lindley / GI-G-1 stationary
        waiting time, finite iff ``E[Y] < 0`` (net profit condition
        ``c E[W] > E[X]``, i.e. ``rho > 0``).

        ``M`` is still a geometric number of iid ascending ladder heights,
        but outside the Poisson world the ladder height law has no closed
        form -- the two pieces of Poisson magic behind PK (ladder heights
        = the equilibrium distribution, ladder epoch probability
        ``1/(1+rho)``) do not survive. Instead the ladder structure is
        extracted numerically from the Wiener-Hopf factorization of the
        walk by the cepstral method: see
        :func:`aggregate._renewal.ruin_cepstral` for the algorithm (Spitzer
        identity, support separation in the cepstrum, ``z = 1``
        regularization).

        Discretization: the severity rides its existing rounding pmf
        (``sev_density_df.p_sev``, exactly as PK); the scaled wait ``cW``
        is discretized by the same rounding scheme via
        :meth:`_discretize_wait_pmf`. Rounding keeps means correct to
        ``O(bs^2)`` and makes the lattice walk aperiodic, so the symbol
        ``1 - phi_Y`` has no unit-circle zeros besides ``z = 1``. The
        kernel's ``z = 1`` patch uses the **grid** mean step (``signed @
        fy``), which also absorbs the (negligible) wait mass dropped
        beyond the grid top. psi is valid for ``u`` below ``2**log2 * bs``;
        ``psi`` at the top of the grid is the wrap-around diagnostic --
        the cepstrum coefficients decay like the severity tail, so a
        heavy-tailed severity needs a wider grid (larger ``log2``) before
        wrap-around contamination sets in, and a warning is issued when
        the top-of-grid psi exceeds 1e-6.

        For exponential waits the model is compound Poisson and this
        method agrees with :meth:`pollaczeck_khinchine` up to
        discretization. See Embrechts, Kluppelberg, Mikosch 1.2 for the
        model and Spitzer (1956) for the identity.
        """
        fname = getattr(self.frequency, 'freq_name', '')
        if fname != 'renewal':
            hint = (' -- for a Poisson frequency use pollaczeck_khinchine'
                    if fname == 'poisson' else '')
            raise ValueError(
                f'wiener_hopf requires a renewal (wait-clause) frequency, '
                f'got {fname!r}{hint}')
        if self.sev_density is None:
            raise ValueError(
                'Must update before computing the Wiener-Hopf ruin function.')
        if self.i0:
            raise ValueError(
                'wiener_hopf requires a nonnegative severity grid '
                '(signed severities unsupported)')
        defect = float(self.frequency.wait_defect or 0.0)
        if defect > 1e-12:
            raise ValueError(
                f'wiener_hopf requires a proper waiting-time law; this one '
                f'is defective (terminating renewal process), '
                f'wait_defect = {defect:.6g}')
        if log2 is None:
            log2 = self.log2
        elif log2 < self.log2:
            raise ValueError(
                f'log2 = {log2} < self.log2 = {self.log2} would truncate '
                f'the severity')
        n = 1 << log2          # usable half grid = u-grid length
        M = 2 * n              # wrapped circle
        bs = self.bs

        # severity: reuse the discretized rounding pmf, exactly as PK
        bit = self.sev_density_df.p_sev
        mean = float(np.sum(bit * bit.index))

        # exact unconditional wait mean; the defect guard guarantees any
        # unconditional window loses no mass, so moms() is the true E[W]
        freq = self.frequency
        weights = np.atleast_1d(np.asarray(freq.wait_weights, dtype=float))
        m1 = np.array([list(sev.moms())[0] for sev, *_ in freq.wait_components])
        mean_wait = float(weights @ m1)
        c = (1.0 + rho) * mean / mean_wait   # premium rate per unit time

        fcw = self._discretize_wait_pmf(c, n)

        # wrapped step pmf of Y = X - cW: X on 0..len(bit)-1, cW reflected
        # into the wrapped negative indices M-1..M-n+1 (index n = M/2 unused)
        X = np.zeros(M)
        X[:len(bit)] = bit.to_numpy()
        Wn = np.zeros(M)
        Wn[0] = fcw[0]
        Wn[M - n + 1:] = fcw[1:][::-1]
        fy = np.real(sfft.ifft(sfft.fft(X) * sfft.fft(Wn)))
        kk = np.arange(M)
        signed = np.where(kk < n, kk, kk - M)
        mean_y = float(signed @ fy)          # bucket units
        if mean_y >= 0:
            raise ValueError(
                f'net profit condition fails on the grid: mean per-claim '
                f'step {mean_y * bs:.6g} >= 0; increase the margin rho')

        pmf_max, psi = ruin_cepstral(fy, mean_y)
        if psi[n - 1] > 1e-6:
            warnings.warn(
                f'wiener_hopf: psi at the top of the grid = {psi[n - 1]:.3g} '
                f'> 1e-6; the all-time maximum has material mass beyond the '
                f'represented range. Re-run with log2 >= {log2 + 1}.')
        ruin = pd.Series(psi[:n], index=np.arange(n, dtype=float) * bs)
        find_u = _ruin_find_u(ruin, kind)
        return RuinFunction(ruin, find_u, mean, pmf_max[:n])

    def _ruin_paths(self, rho, u0=None, *, p=None, log2=None,
                    n_sims=100_000, n_plot=50,
                    n_steps=None, t_plot=None, seed=_RUIN_SEED):
        """Simulation core of the eventual-ruin example: arrays, not a figure.

        Computes the exact eventual-ruin function (dispatching on the
        frequency: Poisson to :meth:`pollaczeck_khinchine`, renewal to
        :meth:`wiener_hopf`; anything else raises), runs the full simulated
        reasonableness check, and samples ``n_plot`` drawable surplus paths.
        :func:`aggregate.pedagogy.ruin_example`, the ``ruin`` exhibit and
        the ``ruin`` chart all consume this one helper, so the docs figure
        and the served documents cannot drift.

        Parameters
        ----------
        rho : float
            Margin-to-loss ratio; the premium rate is
            ``c = (1 + rho) E[X] / E[W]``. Must be positive (net profit).
        u0 : float, optional
            Initial surplus at which the exact and simulated ruin
            probabilities are compared. Exactly one of ``u0`` or ``p``.
        p : float, optional
            Probability of eventual default; resolved to a surplus through
            the ruin function's capital lookup ``find_u(p)`` on the grid.
            Exactly one of ``u0`` or ``p``.
        log2 : int, optional
            Renewal path only: half-grid exponent forwarded to
            :meth:`wiener_hopf` for heavy tails. Raises if supplied with a
            Poisson frequency.
        n_sims : int, default 100_000
            Number of simulated paths for the reasonableness check.
        n_plot : int, default 50
            Number of drawable sample paths returned.
        n_steps : int, optional
            Claims per simulated path (horizon); if None, chosen so the
            remaining drift makes late ruin negligible.
        t_plot : float, optional
            Calendar-time horizon of the drawable paths; if None, derived
            from the exact psi curve (residual ruin beyond it ~ 1e-3).
        seed : int or None, default :data:`_RUIN_SEED`
            rng seed. The default is a fixed constant so a served document
            is hash-stable and cacheable. ``None`` draws a fresh seed,
            which is materialized and reported in the returned ``seed``
            field so the consumer can still say what it used.

        Returns
        -------
        _RuinPaths
            Named tuple of scalars and arrays; see its module comment for
            the field groups. ``paths`` is a tuple of ``(tt, uu,
            ruined_at)`` triples, the interleaved pre/post-claim surplus of
            one drawable path, with ``ruined_at`` the index of its ruin
            point within the horizon (0 when it survives the window).
            ``ruin_time`` is the calendar ruin time per simulated path,
            NaN for survivors.

        Notes
        -----
        The simulation samples the **discretized** severity
        (``sev_density_df.p_sev``) and, on the renewal path, the
        discretized wait law (:meth:`_discretize_wait_pmf`), so the
        simulated model is exactly the model the psi computation prices
        and the sim-vs-exact comparison is apples-to-apples. Ruin can only
        occur at claim instants, so paths are simulated claim by claim; the
        horizon ``n_steps`` is chosen so a surviving path is, with a 3
        sigma margin, deep enough that its residual ruin probability is
        below 1e-5. The funnel is the law of the iterated logarithm under
        the renewal-reward CLT rate ``sigma2 = Var(X - (E[X]/E[W]) W) /
        E[W]``.
        """
        if rho <= 0:
            raise ValueError(
                f'net profit condition requires rho > 0, got {rho}')
        if (u0 is None) == (p is None):
            raise ValueError('exactly one of u0 and p is required')
        if seed is None:
            # the Sample action: draw a real seed so the document can
            # still report what it used (plan-pk-tab ruling 2)
            seed = int(np.random.default_rng().integers(2 ** 31))
        rng = np.random.default_rng(seed)

        # --- exact eventual ruin probability (dispatch on frequency) ------
        fname = getattr(self.frequency, 'freq_name', '')
        if log2 is not None and fname != 'renewal':
            raise ValueError(
                'log2 applies to the renewal (wiener_hopf) path only')
        if fname == 'renewal':
            rf = self.wiener_hopf(rho, kind='index', log2=log2)
        elif fname == 'poisson':
            rf = self.pollaczeck_khinchine(rho, kind='index')
        else:
            raise ValueError(
                f'no eventual-ruin solver for a {fname!r} frequency: need '
                f'poisson (pollaczeck_khinchine) or renewal (wiener_hopf)')
        ruin = rf.ruin
        psi = ruin.to_numpy()
        bs = self.bs
        n = len(ruin)
        if p is not None:
            u0 = float(rf.find_u(p))
        iu = int(round(u0 / bs))
        if iu >= n:
            raise ValueError(
                f'u0 = {u0} lies beyond the represented grid top '
                f'{ruin.index[-1]:.6g}')
        psi_u0 = psi[iu]

        # --- model moments (discretized severity; exact wait mixture) -----
        bit = self.sev_density_df.p_sev
        p_x = bit.to_numpy()
        xs_x = bit.index.to_numpy()
        mx = float(p_x @ xs_x)
        var_x = float(p_x @ xs_x ** 2) - mx * mx
        if fname == 'renewal':
            freq = self.frequency
            ws = np.atleast_1d(np.asarray(freq.wait_weights, dtype=float))
            moms = np.array([list(sev.moms())[:2]
                             for sev, *_ in freq.wait_components], dtype=float)
            mw = float(ws @ moms[:, 0])
            var_w = float(ws @ moms[:, 1]) - mw * mw
        else:
            # poisson rate self.n per year: exponential waits
            mw = 1.0 / self.n
            var_w = mw * mw
        c = (1.0 + rho) * mx / mw            # premium rate per unit time
        # per-claim step Y = X - cW: drift and sd, analytic moments
        mu = c * mw - mx                     # = rho * mx > 0
        sd = np.sqrt(var_x + c * c * var_w)

        # --- samplers: the discretized model, not the continuum -----------
        cdf_x = np.cumsum(p_x / p_x.sum())
        if fname == 'renewal':
            fcw = self._discretize_wait_pmf(c, n)
            tw = np.arange(n, dtype=float) * bs / c    # time units
            cdf_w = np.cumsum(fcw / fcw.sum())

            def sample_w(size):
                return tw[np.searchsorted(cdf_w, rng.random(size))]
        else:
            def sample_w(size):
                return rng.exponential(mw, size)

        def sample_x(size):
            return xs_x[np.searchsorted(cdf_x, rng.random(size))]

        # --- simulation check (ruin can only occur at claim instants) -----
        if n_steps is None:
            # Horizon such that a surviving path is, with ~3 sigma margin,
            # deep enough that its residual ruin probability is < 1e-5:
            # solve |P(Y)| n - 3 sd sqrt(n) = u_safe for n, where u_safe is
            # read off the exact psi just computed.
            i_safe = np.searchsorted(-psi, -1e-5)     # psi is decreasing
            u_safe = max(u0 + i_safe * bs, 10 * mx)
            r = (3 * sd + np.sqrt(9 * sd ** 2 + 4 * mu * u_safe)) / (2 * mu)
            n_steps = min(int(np.ceil(r ** 2)), 50_000)
        ruined = np.zeros(n_sims, dtype=bool)
        ruin_time = np.full(n_sims, np.nan)           # calendar time of ruin
        chunk = max(1, int(2e7) // n_steps)           # cap memory use
        for lo in range(0, n_sims, chunk):
            m = min(chunk, n_sims - lo)
            w = sample_w((m, n_steps))
            x = sample_x((m, n_steps))
            t = np.cumsum(w, axis=1)
            surplus = u0 + c * t - np.cumsum(x, axis=1)
            below = surplus < 0
            hit = below.any(axis=1)
            ruined[lo:lo + m] = hit
            first = np.argmax(below, axis=1)
            ruin_time[lo:lo + m] = np.where(hit, t[np.arange(m), first],
                                            np.nan)
        n_ruin = int(ruined.sum())
        p_sim = n_ruin / n_sims
        se_sim = np.sqrt(p_sim * (1 - p_sim) / n_sims)

        # --- plot horizon -------------------------------------------------
        if t_plot is None:
            # residual ruin beyond the window ~ 1e-3, invisible at n_plot
            # scale
            i3 = np.searchsorted(-psi, -1e-3 * max(psi_u0, 1e-6))
            u3 = max(u0 + i3 * bs, 10 * mx)
            r3 = (3 * sd + np.sqrt(9 * sd ** 2 + 4 * mu * u3)) / (2 * mu)
            t_plot = min(int(np.ceil(r3 ** 2)), n_steps) * mw
        # claims needed to cover t_plot with a fluctuation margin
        n_steps_plot = int(np.ceil(t_plot / mw
                                   + 6 * np.sqrt(t_plot * var_w / mw ** 3)
                                   + 10))

        # --- n_plot drawable sample paths ---------------------------------
        paths = []
        for i in range(n_plot):
            w = sample_w(n_steps_plot)
            x = sample_x(n_steps_plot)
            t = np.cumsum(w)
            u_pre = u0 + c * t - np.concatenate(([0.0], np.cumsum(x)[:-1]))
            u_post = u_pre - x                    # surplus just after claim
            # interleave (pre, post) values at each claim time for the path
            tt = np.repeat(t, 2)
            uu = np.empty(2 * n_steps_plot)
            uu[0::2], uu[1::2] = u_pre, u_post
            tt = np.concatenate(([0.0], tt))
            uu = np.concatenate(([u0], uu))
            hit = np.argmax(uu < 0) if (uu < 0).any() else 0
            # a dip beyond the horizon reads as a survivor of the window
            ruined_at = hit if (hit and tt[hit] <= t_plot) else 0
            paths.append((tt, uu, ruined_at))

        # --- expected trend and LIL funnel --------------------------------
        # renewal-reward CLT rate: sigma2 = Var(X - (PX/PW) W) / PW
        sigma2 = (var_x + (mx / mw) ** 2 * var_w) / mw
        tg = np.linspace(0, t_plot, 400)
        trend = u0 + (c - mx / mw) * tg
        tl = tg[tg > np.e]                        # ln ln t defined
        band = np.sqrt(2 * sigma2 * tl * np.log(np.log(tl)))

        return _RuinPaths(
            fname=fname, rf=rf, seed=seed,
            rho=rho, u0=u0, psi_u0=psi_u0, c=c, mx=mx, var_x=var_x, mw=mw,
            var_w=var_w, mu=mu, sd=sd, sigma2=sigma2,
            n_sims=n_sims, n_steps=n_steps, n_ruin=n_ruin, p_sim=p_sim,
            se_sim=se_sim, ruin_time=ruin_time,
            t_plot=t_plot, paths=tuple(paths), tg=tg, trend=trend, tl=tl,
            band=band)

    def eventual_ruin(self, rho=None, *, lr=None, p=None, u=None, log2=None,
                      n_sims=1000, seed=_RUIN_SEED):
        """Probability of eventual ruin at one capital level: the receipt.

        States a premium margin (``rho`` or ``lr``, exactly one) and a
        capital level (``p`` or ``u``, exactly one), computes the exact
        probability of eventual ruin there (Poisson frequency via
        :meth:`pollaczeck_khinchine`, renewal via :meth:`wiener_hopf`;
        anything else raises), validates it with a capped simulation of the
        same discretized model, and returns the
        :class:`~aggregate.results.RuinResult` the ``ruin`` exhibit
        dispatches on.

        Parameters
        ----------
        rho : float, optional
            Margin-to-loss ratio; the premium rate is
            ``c = (1 + rho) E[X] / E[W]``. Must be positive (net profit).
            Exactly one of ``rho`` or ``lr``.
        lr : float, optional
            The same margin stated as a loss ratio, ``lr = 1 / (1 + rho)``,
            in ``(0, 1)``. Exactly one of ``rho`` or ``lr``.
        p : float, optional
            Probability of eventual default; resolved to an initial surplus
            through the ruin function's capital lookup ``find_u(p)`` on the
            grid. Exactly one of ``p`` or ``u``.
        u : float, optional
            Initial surplus directly. Exactly one of ``p`` or ``u``.
        log2 : int, optional
            Renewal path only: forwarded to :meth:`wiener_hopf` for heavy
            tails.
        n_sims : int, default 1000
            Simulated paths for the reasonableness check. The small default
            follows the teaching-aid sizing ruling (``dev/plan-pk-tab.md``
            ruling 4); raise it for a tighter standard error.
        seed : int or None, default fixed
            rng seed. The fixed default keeps served documents hash-stable;
            ``None`` draws a fresh seed and reports it on the result.

        Returns
        -------
        RuinResult
            Carrying ``ruin_df`` (the one-column stats strip), the resolved
            ``u`` and exact ``psi`` beside the simulated estimate and its
            standard error, the margin both ways, and the seed used.

        Notes
        -----
        On the Poisson path the frame also carries the Lundberg exponent
        ``R`` and the classical bound ``exp(-R u)`` when the adjustment
        equation ``M_X(R) = 1 + (1 + rho) E[X] R`` has a bracketed root on
        the discretized severity (see :func:`_lundberg_exponent`); a book
        whose grid puts the root past the overflow guard simply omits the
        two rows. The simulation samples the discretized severity and wait
        laws, so the simulated model is exactly the model the exact psi
        prices.

        .. versionadded:: 1.0
           Provisional companion to the ``ruin`` exhibit and chart
           (``dev/plan-pk-tab.md``).
        """
        if (rho is None) == (lr is None):
            raise ValueError('exactly one of rho and lr is required')
        if lr is not None:
            if not 0 < lr < 1:
                raise ValueError(f'loss ratio must be in (0, 1), got {lr}')
            rho = 1.0 / lr - 1.0
        if (p is None) == (u is None):
            raise ValueError('exactly one of p and u is required')
        rp = self._ruin_paths(rho, u, p=p, log2=log2, n_sims=n_sims,
                              n_plot=0, seed=seed)
        rows = {
            'frequency kind': rp.fname,
            'safety loading rho': rp.rho,
            'loss ratio': 1.0 / (1.0 + rp.rho),
            'premium rate c': rp.c,
            'mean severity E[X]': rp.mx,
            'mean wait E[W]': rp.mw,
            'initial surplus u': rp.u0,
            'psi(u) exact': rp.psi_u0,
            'psi(u) simulated': rp.p_sim,
            'sim std error': rp.se_sim,
            'sim trials': rp.n_sims,
            'sim ruins': rp.n_ruin,
            'sim horizon (claims)': rp.n_steps,
            'seed': rp.seed,
        }
        if rp.fname == 'poisson':
            bit = self.sev_density_df.p_sev
            r_lund = _lundberg_exponent(bit.to_numpy(),
                                        bit.index.to_numpy(), rp.rho)
            if r_lund is not None:
                rows['Lundberg exponent R'] = r_lund
                rows['Lundberg bound exp(-Ru)'] = np.exp(-r_lund * rp.u0)
        ruin_df = pd.DataFrame({'value': rows})
        return RuinResult(
            ruin_df=ruin_df, rho=rp.rho, lr=lr, p=p, u=rp.u0,
            psi=rp.psi_u0, psi_sim=rp.p_sim, se_sim=rp.se_sim,
            seed=rp.seed, freq_kind=rp.fname, _source=self)

    def plot(self, xmax=None, log=False, full_range=False, reflect=False,
             return_period=False, invert=False):
        """Plot the aggregate and its severity: the mass, and the Lee diagram.

        Parameters
        ----------
        xmax : float, optional
            Upper end of the loss window, in place of the computed one. How
            a gross and a net aggregate are read against one common scale.
        log : bool
            Read both axes of the mass panel on log, which is the log
            density panel this plot used to draw as a third picture.
        full_range : bool
            Show the whole grid rather than the ``q(0.001)`` to ``q(0.999)``
            crop.
        reflect : bool
            Read the Lee panel against the exceeding probability rather
            than the non-exceeding one, so the curve drawn is the survival
            function. With ``invert`` it is ``S(x)`` the usual way round,
            and that axis offers a log reading where the non-exceeding one
            does not.
        return_period : bool
            Read the Lee panel against return period rather than
            non-exceedance probability, which spreads the rare tail so it
            can be read off directly. This was ``quantile_x='return'``.
        invert : bool
            Exchange the Lee panel's axes, which draws the distribution
            function: a quantile function and a cdf are inverses, so it is
            the same pairs read the other way round.

        Returns
        -------
        matplotlib.figure.Figure
            Also stashed on ``self.figure``.

        Notes
        -----
        Draws the chart document ``charts.chart_agg`` emits, through the one
        generic renderer, so this picture and the one a browser draws come
        from a single set of semantic decisions rather than two that must be
        kept in step by hand.

        The two panels are one reading of one book: the loss axis is a
        single axis, the density panel's x and the Lee panel's y, so a
        window moves both. Which drawing each series gets (stems for a small
        discrete book, steps, or a line once a bucket is sub-pixel) is the
        renderer's, from the declared support and the room on screen, which
        is what retired the old discrete and continuous branches.
        """
        from .charts import build_chart_doc
        from .plots import plot_chartdoc
        self.figure = plot_chartdoc(
            build_chart_doc(self, 'agg', xmax=xmax),
            log=log, full_range=full_range, reflect=reflect,
            return_period=return_period, invert=invert)
        return self.figure

    def _limits(self, stat='range', kind='linear', zero_mass='include'):
        """
        Suggest sensible plotting limits for kind=range, density, etc., same as Portfolio.

        Should optionally return a locator for plots?

        Called by ploting routines. Single point of failure!

        Must work without ``q`` function when not yet computed.

        :param stat:  range or density (for y axis)
        :param kind:  linear or log (this is the y-axis, not log of range...that is rarely plotted)
        :param zero_mass:  include exclude, for densities
        :return:
        """

        # fudge l/r factors
        def f(x):
            fl, fr = 0.02, 1.02
            return [-fl * x, fr * x]

        # lower bound for log plots
        eps = 1e-16

        # if not computed
        # GOTCHA: if you call q and it fails because not agg_density then q is set to {}
        # which is not None
        if self.agg_density is None:
            # No FFT output yet; estimate the 0.999 quantile from the theoretical
            # mixed-total agg moments.
            try:
                p999 = _estimate_agg_percentile(self.actual_m, self.actual_cv, self.actual_skew, 0.999)
            except ValueError:
                p999 = np.inf
            return f(p999)

        if stat == 'range':
            p = 0.999 if kind == 'linear' else 0.99999
            hi = self.q(p)
            # Window-aware x-limits keyed on the grid origin (``xs[0]``):
            #  * origin < 0  -- signed P&L: mass can sit anywhere on the real
            #    line, so use a *two-sided* quantile range (``f(hi)`` would be
            #    reversed/clipped when hi < 0);
            #  * origin > 0  -- thin-tailed output window: anchor the left edge
            #    at the realised support minimum, not 0, so the empty
            #    ``[0, x_min]`` band isn't drawn;
            #  * origin == 0 -- ordinary non-negative aggregate: unchanged.
            if self.xs is not None and self.xs[0] < 0:
                lo = self.q(1 - p)
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            if self.xs is not None and self.xs[0] > 0:
                lo = float(self.density['loss'].min())
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            return f(hi)

        elif stat == 'density':
            # for density need to divide by bs
            mx = self.agg_density.max() / self.bs
            mxx0 = self.agg_density[1:].max() / self.bs
            if kind == 'linear':
                if zero_mass == 'include':
                    return f(mx)
                else:
                    return f(mxx0)
            else:
                return [eps, mx * 1.5]
        else:
            # if you fall through to here, wrong args
            raise ValueError('Inadmissible stat/kind passsed, expected range/density and log/linear.')

    # ================================================================
    # Display reports, diagnostics, queries, risk measures, pricing
    # ================================================================

    # ``program`` / ``format_program`` / ``pprogram`` / ``pprogram_html`` come
    # from ``ProgramMixin`` (the shared DecL round-trip surface). See
    # dev/done/plan-program-mixin.md ([Program-Mixin]).

    @staticmethod
    def _count_program(spec, n, name):
        """Render the DecL for a claim-count distribution from a parsed spec.

        Builds a minimal ``agg`` program that keeps the frequency clause verbatim
        and replaces the severity with a point mass at 1 (``dsev [1]``). Because N
        claims each of size 1 sum to N, the resulting aggregate density is exactly
        the claim-count distribution ``P(N = k)``. Shared by
        :meth:`create_frequency` and :meth:`Portfolio.create_frequency`.

        Parameters
        ----------
        spec : dict
            A **raw transformer spec** (``parser.parse(...)[2]``), *not* the
            dense ``Aggregate._spec`` constructor dict.
        n : float
            The resolved total expected count (``Aggregate.n``).
        name : str
            Name for the rendered ``agg``.

        Returns
        -------
        str
            A canonical, single-line DecL ``agg`` program for the count
            distribution.

        Notes
        -----
        Only the *frequency* keys are carried over; severity, layers
        (``exp_limit`` / ``exp_attachment``) and both reinsurance clauses are
        dropped. Those reshape severity-per-claim or the aggregate total but
        never the *number* of claims, and carrying them through against a
        ``dsev [1]`` severity would corrupt the count (e.g.
        ``occurrence net of 50 xs 0`` would net every unit point mass to 0).

        The exposure is collapsed to the resolved expected count ``n`` as
        ``<n> claims`` rather than re-rendering the original exposure clause:
        when the count is *derived* from severity (``500 loss ...``, a
        ``premium at lr`` or limit profile), swapping the severity would change
        the count. ``n`` is the correct total expected count even for profiles
        and mixed frequency. Under ``zm`` / ``zt`` the caller passes the *base*
        mean instead, because the preserved zero-modification clause re-applies
        the shift when the rendered program is rebuilt, and pairs it with an
        explicit ``freq_pin_mean=False`` so the clause reads that number as the
        base mean rather than as the realized one. An empirical
        (``dfreq``) frequency already *is* the
        count distribution, so its outcome/probability vectors are kept as the
        ``dfreq`` head and no ``claims`` exposure is synthesized.

        Frequency mixing / contagion (``mixed gamma c``, ``zm`` / ``zt``, etc.)
        is part of the frequency and is preserved verbatim.
        """
        # Keep only the frequency clause; drop severity, layers and reinsurance.
        new = {'name': name}
        for key in ('freq_name', 'freq_a', 'freq_b', 'freq_zm', 'freq_p0',
                    'freq_pin_mean'):
            if key in spec:
                new[key] = spec[key]
        # An empirical (dfreq) frequency is itself the count distribution and
        # renders as the exposure head; everything else collapses to the
        # resolved expected count.
        if spec.get('freq_name') != 'empirical':
            new['exp_en'] = n
        # Point-mass severity: N unit claims sum to N.
        new['sev_name'] = 'dhistogram'
        new['sev_xs'] = [1.0]
        new['sev_ps'] = [1.0]
        return spec_to_decl(new, kind='agg', name=name)

    @property
    def _renewal_bs_df(self):
        """Wait-grid sizing constraint table for a renewal frequency.

        Delegates to :attr:`FrequencyRenewal._renewal_bs_df` (one row per
        sizing constraint, ``selected`` marking the binding one; final
        grid / kmax / p0 / defect in ``.attrs``). ``None`` for every other
        frequency kind. Mirrors the aggregate ``_bs_window_df`` idiom.
        """
        return getattr(self.frequency, '_renewal_bs_df', None)

    def _frequency_program(self, name):
        """Render the count-distribution DecL by re-parsing :attr:`program`.

        The raw transformer spec is recovered by re-parsing :attr:`program`
        (``spec_to_decl`` wants the transformer spec, not the dense
        ``Aggregate._spec`` constructor dict), then handed to
        :meth:`_count_program` with the resolved :attr:`n`.

        A renewal (wait-clause) frequency emits the **realized** count --
        the materialized ``dfreq [0:kmax] [pN...]`` program built from
        ``self.frequency.freq_a / freq_b`` -- not the wait clause: the
        returned object runs the ordinary dfreq machinery with no
        recomputation of the renewal kernel (the dfreq-conversion view;
        the *model* round-trip via ``spec`` / the writer keeps the wait
        clause).

        Raises
        ------
        ValueError
            If the object was built programmatically and carries no
            :attr:`program` to re-parse.
        """
        if not self.program:
            raise ValueError(
                f'create_frequency requires a DecL program to re-parse; '
                f'aggregate {self.name!r} was built programmatically (empty '
                f'program).')
        from .underwriter import build
        _kind, _name, spec = build.parser.parse(self.program)
        if getattr(self.frequency, 'freq_name', '') == 'renewal':
            spec = {k: v for k, v in spec.items()
                    if not (k.startswith('wait_')
                            or k in ('exp_years', 'exp_rate'))}
            spec['freq_name'] = 'empirical'
            spec['freq_a'] = np.asarray(self.frequency.freq_a)
            spec['freq_b'] = np.asarray(self.frequency.freq_b)
        # base_mean, not n: the rendered program still carries the zm / zt
        # clause, so its exposure number must be the *base* mean -- feeding the
        # realized ``n`` back in would apply the modification a second time.
        # The two coincide for every unmodified frequency.
        #
        # Since a325 the bare clause reads its exposure number as the REALIZED
        # mean, so the child is written in the explicit base-parameterization
        # form (``freq_pin_mean=False``, which the writer renders as the ``!``
        # marker). Without this the child solves for a base mean whose realized
        # mean is the parent's base mean, and the count comes out modified
        # twice: a ``zm 0.3`` parent of 100 claims reported 142.857
        # ([ZT-ZM-Recalibrate-Default]). Harmless for an unmodified frequency,
        # where the writer emits no marker at all.
        if spec.get('freq_zm'):
            spec = dict(spec, freq_pin_mean=False)
        return self._count_program(spec, self.base_mean, name)

    def with_hints(self, **extra):
        """This aggregate's program with its realized grid pinned into ``hints{}``.

        The certification helper for the reference-severity feature. A
        ``sev agg.NAME`` reference requires the referenced declaration to carry
        explicit ``log2`` and ``bs`` hints, so that the severity it stands for
        is pinned by the recipe base rather than by whatever ambient defaults
        happened to be in force. This is how a declaration acquires them: get
        the inner right interactively, then ``build(inner.with_hints())``
        re-registers it with its resolution pinned, turning a candidate inner
        into a certified one.

        Parameters
        ----------
        **extra
            Further ``hints{}`` settings (any key the language allows, e.g.
            ``padding=2``, ``normalize=False``), merged over ``log2``, ``bs``
            and ``normalize``, which come from the object's current state.

        Returns
        -------
        str
            One line of DecL, ready to hand back to ``build``.

        Raises
        ------
        ValueError
            If the object has not been updated, carries no DecL program, or is
            given a hint key DecL does not have.

        Examples
        --------
        >>> from aggregate import build
        >>> a = build('agg WH.Doc 10 claims sev lognorm 50 cv 1 poisson')
        >>> 'hints{' in a.with_hints()
        True

        Notes
        -----
        The existing trailer survives: the clause is merged key by key, so a
        declared ``padding`` stays and only the grid moves.
        """
        return _program.with_hints(self, **extra)

    def as_severity(self, limit=np.inf, attachment=0, conditional=False):
        """Use this aggregate's output loss distribution as a severity.

        The programmatic twin of the DecL ``sev agg.NAME`` reference, and the
        mirror of :meth:`aggregate.portfolio.Portfolio.as_severity`. The
        motivating shape is a compound-of-a-compound: a per-policy aggregate
        becomes the per-policy severity of a book of policies, which is how a
        US personal-auto split limit (100 per claimant, 300 per policy) is
        written.

        Parameters
        ----------
        limit : float, default ``np.inf``
            Layer width applied to the resulting severity.
        attachment : float, default 0
            Layer attachment applied to the resulting severity.
        conditional : bool, default False
            Whether layered moments divide out ``P(X > attachment)``.

        Returns
        -------
        SeverityMeta
            A discrete severity whose atoms are :attr:`agg_density` on
            :attr:`xs`. Under reinsurance that is the **output view** (ceded or
            net), which is the whole point: the object's answer to "what do you
            output" already has the cession baked in.

        Raises
        ------
        ValueError
            If the aggregate has not been updated. The conversion reads the
            distribution the object outputs, and it does not have one yet.

        Notes
        -----
        Nothing is recomputed and nothing on ``self`` is touched. See
        :class:`aggregate.distributions.SeverityMeta` for what the resulting
        severity is (a fully formed discrete severity with exact moments) and
        the DecL reference documentation for the declarative route, which adds
        a hygiene rule this programmatic path cannot enforce: the source must
        already be at the resolution you meant.
        """
        if self.agg_density is None:
            raise ValueError(
                f'{self.name}: update the aggregate before converting it to a '
                'severity -- the conversion reads the distribution it outputs, '
                'and there is not one yet.')
        return Severity(sev_name=self, exp_attachment=attachment,
                        exp_limit=limit, sev_conditional=conditional)

    def create_frequency(self):
        """Materialize this object's claim-count distribution as an ``Aggregate``.

        The engine carries frequency only as a PGF (applied in the Fourier
        domain), so there is no ``q`` / ``tvar`` / ``cdf`` / percentiles for the
        count itself. This builds the marginal count distribution as a
        first-class :class:`Aggregate` (via the ``dsev [1]`` point-mass trick) so
        every inherited method works on the count.

        Returns
        -------
        Aggregate
            A built aggregate named ``f'{self.name}.freq'`` whose aggregate
            density *is* this object's claim-count distribution: ``agg_density[k]
            = P(N = k)``. Use ``.q``, ``.tvar``, ``.cdf``, ``.plot``,
            ``.density_df``, ``.summary_df`` etc. on it directly.

        Examples
        --------
        >>> fa = a.create_frequency()
        >>> fa.q([0.01, 0.5, 0.99])   # count percentiles
        >>> fa.tvar(0.99)             # tail count
        >>> fa.plot()                 # the count distribution, plotted

        Notes
        -----
        Built through the normal ``build`` front door, so grid windowing, ``bs``
        and ``log2`` selection all happen automatically -- a high-mean count
        (large exposure) needs no special handling. The returned object is a
        *snapshot*: rebuild it if the parent changes. See
        :meth:`_frequency_program` for what is kept and dropped from the spec.
        """
        from .underwriter import build
        return build(self._frequency_program(f'{self.name}.freq'))

    def as_tweedie(self):
        """The reproductive Tweedie parameters of this aggregate, or ``None``.

        A Tweedie with ``1 < p < 2`` *is* a compound Poisson distribution with
        gamma severity, so any aggregate of that shape has reproductive
        parameters whether or not it was declared with the ``tweedie`` keyword.
        This reports them.

        Returns
        -------
        TweedieParameters or None
            ``(p, mean, dispersion)``, with ``variance = dispersion * mean ** p``.
            ``None`` when the aggregate is not a plain compound Poisson-gamma.

        Notes
        -----
        Two sources, one answer. When the object was declared as
        ``tweedie <p> <mean> <dispersion>`` the declared triple is returned
        verbatim, off the private ``_tweedie`` provenance key the parser
        records. Otherwise the triple is *derived* from the engine by
        :func:`aggregate.tweedie.tweedie_convert`, running its
        ``(lambda, alpha, beta)`` to ``(p, mu, sigma^2)`` direction over the
        Poisson mean and the gamma shape and scale.

        That second path is the point. ``10.05 claims sev gamma 0.0995 cv
        0.0709 poisson`` is the same distribution as ``tweedie 1.005 1 0.1`` and
        answers here too, so the feature is about the mathematics rather than
        about which spelling was used. The unparser deliberately does *not*
        work this way: it renders from provenance alone, because rewriting an
        aggregate into a spelling its author did not choose is not its job.
        See ``dev/done/plan-tweedie.md``.

        Recognition is refused for anything that is no longer a bare compound
        Poisson-gamma: reinsurance or a layer at either level, a limit or
        attachment on the severity, a mixed or weighted severity, a location
        shift or splice, a reflected severity, a limit profile, a zero-modified
        or truncated frequency, and any method-of-moments ``approximate``
        fit (whose engine is a fitted single severity on a fixed count, not the
        requested compound at all).

        Examples
        --------
        >>> from aggregate import build
        >>> from aggregate.tweedie import Tweedie
        >>> a = build('agg Doc tweedie 1.5 100 0.5')
        >>> a.as_tweedie()
        TweedieParameters(p=1.5, mean=100.0, dispersion=0.5)

        The named tuple splats, which is how you reach the analytic object and
        its exact density::

            tw = Tweedie(*a.as_tweedie())
        """
        from .tweedie import TweedieParameters, tweedie_convert

        if self._tweedie is not None:
            return self._tweedie
        # Structural gates. Each one is a way for the object to stop being a
        # bare compound Poisson-gamma; none is recoverable by reparameterizing.
        if self.approximation:
            return None
        if self.occ_reins is not None or self.agg_reins is not None:
            return None
        if len(self.sevs) != 1 or len(np.atleast_1d(self.en)) != 1:
            return None
        freq = self.frequency
        if getattr(freq, 'freq_name', '') != 'poisson' or freq.freq_zm:
            return None
        sev = self.sevs[0]
        if sev.sev_name != 'gamma' or sev.sev_reflect:
            return None
        if sev.sev_loc != 0 or sev.sev_lb != 0 or not np.isinf(sev.sev_ub):
            return None
        if not np.isinf(sev.limit) or (sev.attachment or 0) != 0:
            return None
        if not np.isclose(sev.sev_wt, 1.0):
            return None
        ans = tweedie_convert(λ=self.n, α=sev.sev_a, β=sev.sev_scale)
        return TweedieParameters(p=float(ans['p']), mean=float(ans['μ']),
                                 dispersion=float(ans['σ^2']))

    @property
    def validation_df(self):
        """Moment-vs-estimate table for Freq / Sev / Agg (the QA view).

        The validation frame: it proves the FFT reproduced the analytic
        moments ("if the first three moments match, the aggregate is *not
        unreasonable*"). Surfaced on demand and in :meth:`qd` when the object
        *fails* validation; the daily-driver headline is :attr:`summary_df`.
        Three-row Freq / Sev / Agg frame.

        Two display modes, same 8-column shape and same column arithmetic:

        * **No reinsurance** -- validation view. Columns are theoretical
          ``EX | Est EX | Err EX | CV | Est CV | Err CV | Sk | Est Sk``.
          ``Err`` is the noise-aware relative error of empirical vs
          theoretical.
        * **With reinsurance** -- economic view. Columns become ``Gross EX
          | <label> EX | Change EX | Gross CV | <label> CV | Change CV |
          Gross Sk | <label> Sk``, where ``Gross`` is the theoretical
          before any cover and ``<label>`` is the model output -- ``Net``
          (all covers net of), ``Ceded`` (all ceded to), or ``Output``
          (mixed, occ and agg passing different kinds). ``Change = (output
          - gross) / gross`` -- arithmetically the same column as ``Err``
          (so the eyeball degenerates cleanly to the validation view when
          reins is absent), but now read as the % change driven by the
          cession. Labels are the ``REINS_LABEL_*`` constants.

        Sources from the canonical ``self.stats_df``: ``mixed`` for
        Gross, ``empirical`` for the realised (model-output) view.
        """
        return self._describe()

    @staticmethod
    def _cv_or_nan(mean, sd):
        """``CV = SD / E[X]``, blanked (``NaN``) when the mean is ~0.

        ``CV`` is meaningless near a zero mean (a signed / near-break-even
        position), so it is left blank when ``|mean| < CV_MEAN_REL_TOL * sd``
        -- the mean is then indistinguishable from zero at the scale of the
        spread. ``SD`` is always reported by the caller; only ``CV`` blanks.
        """
        mean = float(mean)
        sd = float(sd)
        if not (np.isfinite(mean) and np.isfinite(sd)):
            return np.nan
        if abs(mean) < CV_MEAN_REL_TOL * sd:
            return np.nan
        if mean == 0.0:
            return np.nan
        return sd / mean

    @property
    def summary_df(self):
        """At-a-glance risk view -- moments + key percentiles, Freq / Sev / Agg.

        The daily-driver headline (the lead frame in :meth:`qd` and
        :meth:`_repr_html_`). The compound-model identity made legible: each row
        answers a different question -- count risk (``Freq``), single-claim
        severity (``Sev``), total loss (``Agg``) -- and the percentiles trace
        where the tail comes from (a heavy ``Agg`` skew you can see is inherited
        from ``Sev``). The *validation* moment-error table is now
        :attr:`validation_df`; the tail-behavior classifier is
        :attr:`tail_behavior_df`.

        **Index** ``Freq`` / ``Sev`` / ``Agg`` (the ``X`` index).

        **Columns** ``Mean | SD | CV | Skew | P01 | Median | P99``.

        - ``SD`` and ``CV`` are **both always present** (stable layout).
          ``CV = SD / Mean`` is blank when ``|Mean|`` is ~0 relative to ``SD``
          (a signed / near-break-even position -- see :meth:`_cv_or_nan`); ``SD``
          never blanks. ``Skew`` is well defined even at mean 0, so it stays.
        - Moments are the **computed** (realised FFT-grid) moments for the
          ``Sev`` and ``Agg`` rows -- ``Mean`` is the ``est_*`` estimate, the
          same value validation audits against, not the analytic moment. The
          ``Freq`` row stays PGF-exact (the engine never materializes a count
          distribution to estimate from, so analytic *is* the realised value).
          **Before** :meth:`update` there is no grid, so the whole frame falls
          back to the analytic (theoretical) moments from :attr:`stats_df`.
        - Percentiles come from the FFT grid (exact, not simulated): ``Agg`` via
          :meth:`q`, ``Sev`` via :meth:`q_sev` (mixtures included, already on the
          grid). They populate only **after** :meth:`update`.

        **Frequency-row percentiles are blank** by design: frequency is carried
        as a PGF (``freq_pgf``), applied in the Fourier domain -- the engine
        never materializes a count distribution, so there is nothing to take a
        quantile of. The Freq row still carries ``E[X] / SD / CV / Skew``
        (PGF-exact), which is what that row is for (count volatility). To get the
        count distribution as a first-class object, use
        :meth:`create_frequency`, then ``.q(...)`` / ``.tvar(...)`` on it.

        Returns
        -------
        pandas.DataFrame
            Three-row Freq / Sev / Agg frame, ``Mean`` carried in ``.attrs``.
        """
        st = self.stats_df['mixed']
        rows = ['Freq', 'Sev', 'Agg']
        comps = ['freq', 'sev', 'agg']
        updated = self.agg_density is not None
        # Theoretical (analytic) moments: the permanent source for the Freq row
        # (PGF-exact) and the whole-frame fallback before update().
        th_means = [float(st[(c, 'mean')]) for c in comps]
        th_cvs = [float(st[(c, 'cv')]) for c in comps]
        th_sds = [m * cv if np.isfinite(cv) else np.nan
                  for m, cv in zip(th_means, th_cvs)]
        th_skews = [float(st[(c, 'skew')]) for c in comps]
        if updated:
            # Computed (realised FFT-grid) moments for Sev / Agg; Freq stays
            # theoretical (the engine estimates no count distribution).
            means = [th_means[0], self.est_sev_m, self.est_m]
            sds = [th_sds[0], self.est_sev_sd, self.est_sd]
            skews = [th_skews[0], self.est_sev_skew, self.est_skew]
        else:
            means, sds, skews = th_means, th_sds, th_skews
        df = pd.DataFrame(
            {
                'Mean': means,
                'SD': sds,
                'CV': [self._cv_or_nan(m, sd) for m, sd in zip(means, sds)],
                'Skew': skews,
            },
            index=rows,
        )
        df.index.name = 'X'
        # Percentiles from the realised grid (exact, not simulated); Freq blank
        # by design (PGF, no materialized count distribution); pre-update blank.
        pcols = [_summary_pct_label(p) for p in SUMMARY_PERCENTILES]
        for pc in pcols:
            df[pc] = np.nan
        if updated:
            for p, pc in zip(SUMMARY_PERCENTILES, pcols):
                df.loc['Sev', pc] = self.q_sev(p)
                df.loc['Agg', pc] = self.q(p)
        for c in ('Mean', 'SD', 'Skew', *pcols):
            df[c] = _snap_noise(df[c])
        df.attrs['mean'] = means[-1]
        return df

    @property
    def tail_df(self):
        """Return-period / exceedance table on the default ladder (a property).

        The first-class-citizen form of :meth:`tail_periods_df`: no arguments,
        the standard :data:`DEFAULT_RETURN_PERIODS` ladder. Pass your own
        ladder with ``tail_periods_df(periods=...)``.

        Returns
        -------
        pandas.DataFrame or None
            ``None`` before :meth:`update`.
        """
        return self.tail_periods_df()

    def tail_periods_df(self, periods=None):
        """Return-period / exceedance table for the aggregate (the centerpiece).

        The language of reinsurance submissions, cat-model output, and
        Solvency II / rating-agency capital. **Aggregate-only** (tail risk is a
        property of the total), so it complements :attr:`summary_df`
        ("made of") with "how bad does it get". The tail numbers -- including the
        1-in-1000 TVaR -- come from the FFT grid, **exact, not simulated** (no
        Monte-Carlo wobble).

        **Index** the non-exceedance probability ``P``, running symmetrically
        from ``0.001`` to ``0.999`` on the default ladder
        (:data:`DEFAULT_RETURN_PERIODS`; pass ``periods=`` to override): every
        rung contributes both its lower-tail probability ``1 / T`` and its
        upper-tail probability ``1 - 1 / T``. The 1-in-200 (99.5%, Solvency II)
        and 1-in-250 (99.6%, US capital-adequacy / rating) rows are highlighted
        in the HTML rendering, on both sides.

        **Columns** ``T | VaR | TVaR | xsVaR | VaR/Mean``.

        - ``T`` the return period as the row reads: ``1 / P`` below the median,
          ``1 / (1 - P)`` above it.
        - ``VaR = q(P)``: the quoted number.
        - ``TVaR = tvar(P)``: the priced number, adjacent to ``VaR`` so the
          VaR-to-TVaR gap (tail fatness) reads at a glance.
        - ``xsVaR = VaR - E[X]``: capital, the excess of VaR over expected.
        - ``VaR/Mean``: leverage.

        Parameters
        ----------
        periods : array_like of float, optional
            Return-period ladder. Defaults to :data:`DEFAULT_RETURN_PERIODS`.

        Returns
        -------
        pandas.DataFrame or None
            Indexed by non-exceedance probability ``P``; ``E[X]`` carried in
            ``.attrs``. ``None`` before :meth:`update` (the realised grid is
            not yet built).

        Notes
        -----
        The ladder is symmetric in ``P`` and carries no orientation of its
        own: a loss object is read off the upper rows, where the bad outcome
        is a rare large loss, and a payoff object off the lower rows, where it
        is a rare small result. Both readings sit in one table, which is what
        lets a signed position be read from either side. Downside *TVaR* for a
        signed payoff position is refined in the P&L veneer (see
        ``dev/plan-pnl-*``).
        """
        if self.agg_density is None:
            return None
        return return_period_frame(self.q, self.tvar, self.est_m, periods)

    @property
    def approximation_df(self):
        """All five method-of-moments fits against the exact law (a property).

        Columns ``exact`` (the anchor, the realized grid) then
        :data:`APPROXIMATION_FAMILIES`; rows the ``meta`` / ``stats`` /
        ``quantiles`` / ``rel err`` blocks of :func:`approximation_frame`,
        the last two on the ``tail_df`` ladder. On demand, no options,
        always all five families; ``rel err`` reads each family quantile
        as a relative error against ``exact``.

        Returns
        -------
        pandas.DataFrame or None
            ``None`` before :meth:`update`.

        Notes
        -----
        Each family column is the law of the **emitted program** under the
        mirroring rule (`dev/done/plan-approximate-punchup.md`): an unsigned
        (plain ``sev``) subject clamps a fit's sub-zero mass to an atom at
        0, so even a three-parameter (shifted) family can differ from the
        ``exact`` column, which is exactly what the frame teaches; the
        two-parameter ``lognorm`` / ``gamma`` differ in ``skew`` by
        construction, ``norm`` targets zero. ``ks`` is the Kolmogorov
        distance ``sup |F - G|`` on the grid at the bucket convention
        ``F(x_k) = P(X <= x_k)``, the Berry-Esseen quantity.
        """
        if self.agg_density is None:
            return None
        return approximation_frame(
            self.est_m, self.est_cv, self.est_skew, self._signed(),
            self.density_df.loss.to_numpy(dtype=float),
            self.density_df.p_total.to_numpy(dtype=float), self.q, self.bs)

    @property
    def approximation_density_df(self):
        """Grid-mass densities of the five fits beside the exact (a property).

        The plotting feed for the ``approximation`` chart: index the output
        grid, column ``exact`` the realized density, then one column per
        family holding ``pdf(x) * bs`` of the emitted law (reflect and
        mirror-clamp applied), so the columns overlay the discrete density
        directly. See :func:`approximation_density_frame`.

        Returns
        -------
        pandas.DataFrame or None
            ``None`` before :meth:`update`.
        """
        if self.agg_density is None:
            return None
        return approximation_density_frame(
            self.est_m, self.est_cv, self.est_skew, self._signed(),
            self.density_df.loss.to_numpy(dtype=float),
            self.density_df.p_total.to_numpy(dtype=float), self.bs)

    def _describe(self, force_reins_label=None, force_sd=False):
        """Build the ``validation_df`` frame, optionally forced into reins view.

        Parameters
        ----------
        force_reins_label : str or None
            When ``None`` (the default, used by the ``validation_df`` property)
            the column format is chosen from this unit's own reinsurance:
            the economic Gross/Net/Ceded/Output view if a treaty is
            present, else the plain theory/empirical validation view.

            When a non-``None`` label is supplied, the economic view is
            forced and that label is used for the after-reins column,
            regardless of this unit's own cession. ``Portfolio.validation_df``
            passes a portfolio-wide label here so that every unit block —
            including units with no reinsurance — shares one column
            layout and aligns with the ``total`` block.
        force_sd : bool, default False
            Force the **SD** spread trio (instead of CV) even when this unit
            is not itself signed. A signed unit always uses SD; this flag lets
            ``Portfolio.validation_df`` push the whole table into SD when *any* unit
            is signed, so the unit blocks and the ``total`` block share one
            column layout (CV and SD cannot be mixed in one frame).

        Returns
        -------
        pandas.DataFrame
            Three-row Freq / Sev / Agg frame; see :attr:`validation_df`.
        """
        if self._signed() or force_sd:
            return self._describe_signed(force_reins_label)
        st = self.stats_df['mixed']
        rlabel = force_reins_label if force_reins_label is not None \
            else self._reins_after_label()
        df = pd.DataFrame(
            {
                'EX': [st[('freq', 'mean')], st[('sev', 'mean')], st[('agg', 'mean')]],
                'CV': [st[('freq', 'cv')],   st[('sev', 'cv')],   st[('agg', 'cv')]],
                'Sk': [st[('freq', 'skew')], st[('sev', 'skew')], st[('agg', 'skew')]],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'
        emp = self.stats_df['empirical']
        post_update = pd.notna(emp.get(('agg', 'mean'), np.nan))
        if post_update:
            # Realised (after-reins, or = subject if no reins) middle column.
            mid_label = rlabel or 'Est'
            df.loc['Sev', f'{mid_label} EX'] = emp[('sev', 'mean')]
            df.loc['Agg', f'{mid_label} EX'] = emp[('agg', 'mean')]
            change_label = 'Change' if rlabel else 'Err'
            df.loc[:, f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} CV'] = emp[('sev', 'cv')]
            df.loc['Agg', f'{mid_label} CV'] = emp[('agg', 'cv')]
            df.loc[:, f'{change_label} CV'] = _noise_aware_rel_error(
                df[f'{mid_label} CV'], df['CV'])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = emp[('sev', 'skew')]
            df.loc['Agg', f'{mid_label} Sk'] = emp[('agg', 'skew')]
            ordered = [
                'EX', f'{mid_label} EX', f'{change_label} EX',
                'CV', f'{mid_label} CV', f'{change_label} CV',
                'Sk', f'{mid_label} Sk',
            ]
            df = df[ordered]
        # First-column label: under reinsurance the theoretical is the
        # ``Gross`` view (top of step 1, before any cover); without reins keep
        # the legacy ``EX``/``CV``/``Sk`` headings (no rename necessary).
        if rlabel:
            df = df.rename(columns={
                'EX': f'{REINS_LABEL_GROSS} EX',
                'CV': f'{REINS_LABEL_GROSS} CV',
                'Sk': f'{REINS_LABEL_GROSS} Sk'})
        # snap floating-point dust to 0 in moment-value columns for
        # display (e.g. the skew of a symmetric severity); NaN preserved.
        # Change/Err columns retain their numeric dust (they are the
        # validation eyeball).
        for c in df.columns:
            if ' EX' in c or ' CV' in c or ' Sk' in c or c in ('EX', 'CV', 'Sk'):
                if not (c.startswith('Err ') or c.startswith('Change ')):
                    df[c] = _snap_noise(df[c])
        return df

    def _describe_signed(self, force_reins_label=None):
        """``validation_df`` for a signed aggregate -- **SD** trio instead of CV.

        Same 8-column shape and column arithmetic as :meth:`_describe`, but the
        ``CV`` trio is replaced by an ``SD`` trio. The coefficient of variation
        ``CV = sd / mean`` is unstable and meaningless when the mean can be ~0
        (a signed aggregate straddling 0), so for any signed object -- a
        ``ssev`` / negative-``dsev`` aggregate -- the spread is reported as the
        standard deviation, which is finite and informative regardless of the
        mean. (This also cleans up the 1.0.0a22 signed-portfolio validation_df.)

        The Freq / Sev / Agg rows are in their native (signed) frame; the
        theoretical (Gross) column is sourced from the loss ``stats_df``.

        Parameters
        ----------
        force_reins_label : str or None
            As in :meth:`_describe`.

        Returns
        -------
        pandas.DataFrame
        """
        st = self.stats_df['mixed']
        emp = self.stats_df['empirical']
        rlabel = force_reins_label if force_reins_label is not None \
            else self._reins_after_label()

        # Theoretical (loss stats_df). The signed severity already straddles 0,
        # so the moments are in their native frame -- no display transform.
        freq_sd = st[('freq', 'mean')] * st[('freq', 'cv')]
        sev_sd = self.sev_sd
        actual_sd = self.actual_sd
        df = pd.DataFrame(
            {
                'EX': [st[('freq', 'mean')], st[('sev', 'mean')],
                       st[('agg', 'mean')]],
                'SD': [freq_sd, sev_sd, actual_sd],
                'Sk': [st[('freq', 'skew')], st[('sev', 'skew')],
                       st[('agg', 'skew')]],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'
        post_update = pd.notna(emp.get(('agg', 'mean'), np.nan))
        if post_update:
            mid_label = rlabel or 'Est'
            # Empirical: Freq/Sev from the stats_df; Agg from the scalars.
            emp_freq_sd = emp[('freq', 'mean')] * emp[('freq', 'cv')]
            df.loc['Sev', f'{mid_label} EX'] = self.est_sev_m
            df.loc['Agg', f'{mid_label} EX'] = self.est_m
            change_label = 'Change' if rlabel else 'Err'
            df.loc[:, f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} SD'] = self.est_sev_sd
            df.loc['Agg', f'{mid_label} SD'] = self.est_sd
            df.loc['Freq', f'{mid_label} SD'] = emp_freq_sd
            df.loc[:, f'{change_label} SD'] = _noise_aware_rel_error(
                df[f'{mid_label} SD'], df['SD'])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = self.est_sev_skew
            df.loc['Agg', f'{mid_label} Sk'] = self.est_skew
            ordered = [
                'EX', f'{mid_label} EX', f'{change_label} EX',
                'SD', f'{mid_label} SD', f'{change_label} SD',
                'Sk', f'{mid_label} Sk',
            ]
            df = df[ordered]
        if rlabel:
            df = df.rename(columns={
                'EX': f'{REINS_LABEL_GROSS} EX',
                'SD': f'{REINS_LABEL_GROSS} SD',
                'Sk': f'{REINS_LABEL_GROSS} Sk'})
        for c in df.columns:
            if ' EX' in c or ' SD' in c or ' Sk' in c or c in ('EX', 'SD', 'Sk'):
                if not (c.startswith('Err ') or c.startswith('Change ')):
                    df[c] = _snap_noise(df[c])
        return df

    def _reins_after_label(self):
        """Heading for the model-output column in ``validation_df``.

        ``Net`` when every cession passes the net; ``Ceded`` when every
        cession passes the ceded; ``Output`` when occ and agg pass
        different kinds (e.g. ``net of occ then ceded to agg`` -- a mixed
        output). Returns ``None`` when no reinsurance is configured (legacy
        validation-view headings apply).
        """
        return _reinsurance.reins_after_label(self)

    def _severity_lattice(self):
        """Integer-lattice step of the severity, or ``None`` if not on a lattice.

        Returns the gcd of the (integer) severity atoms -- the natural bucket
        size, since an aggregate of lattice-valued severities is itself on that
        lattice **regardless of the frequency** (e.g. Poisson x ``dsev [1:10]``
        is integer-valued, so ``bs=1``). ``None`` for any continuous component.
        This is what lets the window estimator pick a coarse, exact ``bs`` and
        shrink ``log2`` to fit, instead of defaulting to a fine ``bs`` over the
        full ``2**log2`` buckets.

        Returns
        -------
        float or None
            ``gcd`` of the integer atoms (1 for ``dsev [1:n]``; 5 for atoms
            ``[0 5 10]``), or ``None`` if any severity component is continuous /
            non-integer.
        """
        atoms = []
        for s in (self.sevs if self.sevs is not None else []):
            a = getattr(s, 'support_atoms', None)
            if a is None:
                return None
            atoms.append(np.asarray(a, dtype=float))
        if not atoms:
            return None
        allx = np.concatenate(atoms)
        if not np.allclose(allx, np.round(allx), atol=1e-9):
            return None
        ints = np.abs(np.round(allx).astype(np.int64))
        ints = ints[ints != 0]
        if len(ints) == 0:
            return 1.0
        g = int(np.gcd.reduce(ints))
        return float(g) if g > 0 else 1.0

    def _severity_high_estimate(self, p):
        """Upper extent (~``p`` quantile) of the per-occurrence severity mixture.

        Used by ``_bs_window``'s ``windowed`` guard. The windowed method only
        relabels the finished *aggregate*; the severity is still discretised on
        ``xs_sev = [0, N*bs]`` (physical 0 at index 0). So windowing is valid
        only when a single severity fits in the windowed grid extent -- true for
        a genuine compound (many small claims summing far above 0), false for a
        ``fixed``-1 / ``approximate`` object whose one severity already sits at
        the aggregate mean.

        Parameters
        ----------
        p : float
            Coverage (e.g. ``1 - 1e-12``) for the method-of-moments percentile
            on an unbounded severity.

        Returns
        -------
        float
            A (conservative) upper bound on the severity support: the analytic
            method-of-moments high quantile from the severity moments, capped by
            any finite policy/support limit. A **degenerate (point-mass)**
            severity returns its atom location directly (see Notes). ``inf`` if
            it cannot be estimated (forces the windowed guard to fail safe -> no
            windowing).

        Notes
        -----
        The method-of-moments quantile slightly *overstates* a bounded
        severity's reach, which biases the guard toward **not** windowing --
        the safe direction (a false reject merely keeps the legacy 0-based
        grid; a false accept would corrupt the severity discretisation).

        A degenerate severity (standard deviation zero -- a ``dsev [k]`` point
        mass, e.g. the count distribution materialized by
        :meth:`create_frequency`) has no spread, so its skewness is ``0/0 =
        NaN`` and the MoM fit returns NaN -- which would silently fail the
        ``np.isfinite`` windowed guard and force a high-mean concentrated
        aggregate onto the coarse 0-based grid (the *textbook* windowing case,
        defeated). Such a severity's high extent is simply the atom location
        (the mean), returned directly. The test is on the standard deviation
        against :data:`VALIDATION_NOISE`, so a machine-noise ``cv`` still counts
        as a point mass.
        """
        try:
            sev_m = float(self.stats_df['mixed'][('sev', 'mean')])
            sev_cv = float(self.stats_df['mixed'][('sev', 'cv')])
            sev_sk = float(self.stats_df['mixed'][('sev', 'skew')])
            if sev_m * sev_cv <= VALIDATION_NOISE:
                # Degenerate point mass: zero spread, the MoM fit is undefined.
                hi = sev_m
            else:
                hi = float(_estimate_agg_percentile(sev_m, sev_cv, sev_sk, p))
        except (ValueError, KeyError):
            hi = np.inf
        # ``self.limit`` records the layer AS DECLARED. A signed severity never
        # applies it (``Severity._drop_layer_clause`` warns and drops the
        # clause), so the declared limit does not bound the law and must not
        # cap its reach here: on a signed book this estimate sizes the grid,
        # and capping at a layer that was never applied sizes it short.
        lim = (float(self.limit.max())
               if (self.limit is not None and len(self.limit)
                   and not self._signed_severity()) else np.inf)
        return min(hi, lim) if np.isfinite(lim) else hi

    def _severity_low_estimate(self, tail):
        """Lower (``~tail`` quantile) reach of the severity mixture.

        The signed counterpart of :meth:`_severity_high_estimate`: how far
        *below* zero a single occurrence reaches at lower-tail probability
        ``tail``. Used by :meth:`_single_big_jump_window` to size the negative
        extent (and grid width) of a signed aggregate so the severity cannot
        wrap the FFT buffer (the aliasing failure mode).

        Parameters
        ----------
        tail : float
            Lower-tail probability (e.g. ``1e-14``); the reach is ``ppf(tail)``.

        Returns
        -------
        float
            The minimum component ``ppf(tail)`` -- a conservative (most
            negative) bound on the severity's lower support. ``0.0`` for a
            non-signed severity (no negative reach). ``-inf`` if a component
            quantile cannot be evaluated.

        Notes
        -----
        Bracketing the mixture quantile below by the minimum of the component
        quantiles is the safe (wider) direction for grid sizing: it can only
        widen the window, never clip the severity's negative tail.
        """
        if not self._signed_severity() or self.sevs is None:
            return 0.0
        los = []
        for s in self.sevs:
            if not getattr(s, 'signed', False):
                continue
            try:
                los.append(float(s.fz.ppf(tail)))
            except Exception:  # pragma: no cover - defensive
                los.append(-np.inf)
        return float(min(los)) if los else 0.0

    def _loss_tail_classes(self):
        """Loss-space ``(left_tail, right_tail)`` rungs for the bucket sizer.

        The single-big-jump floor and the tail-aware slack split key off the
        *loss* convolution's tail thickness -- the grid is sized on the loss
        FFT. Returns the aggregate's per-side decay rungs from the shared tail
        report (:meth:`_tail_rows`), oriented in loss space.

        Returns
        -------
        (left_tail, right_tail) : tuple of aggregate.tail.TailClass
            The loss-space aggregate decay rungs. Spec-only (valid before
            :meth:`update`); fed to :func:`aggregate.tail.is_thick`.

        Notes
        -----
        ``reference=False``: the grid is sized on the law that is actually
        convolved. A ``sev agg.NAME`` reference materializes to a certified
        finite set of atoms, and the outer sizes from it exactly as it would
        from a hand-written ``dsev``, legitimately treating it as bounded --
        the referenced object's own theoretical tail already drove the
        referenced object's own window choice. The reporting surfaces take the
        other reading; see :meth:`_tail_rows`.
        """
        agg = self._tail_rows(reference=False)[-1]
        return agg.left_tail, agg.right_tail

    def _single_big_jump_window(self, p_star):
        """Single-big-jump extent floor for a heavy / signed severity.

        For a subexponential severity the aggregate's far tail is dominated by a
        single large claim on an otherwise typical bulk:
        ``P(S > x) ~ E[N]·P(X > x)``. So to cover the aggregate to ``p_star``
        the *severity* must be probed at the deeper level
        ``p** = 1 - (1 - p_star)/E[N]`` -- one claim reaches there, the other
        ``E[N]-1`` are typical (this is **not** ``N·q_X``, which would assume
        *every* claim is huge and wildly over-size). The single-big-jump extent
        replaces one typical claim (mean ``mu_X``) on the bulk (aggregate mean
        ``ES``) by one big claim::

            sbj_hi = ES - mu_X + q_X_hi(p**)      # one big claim up
            sbj_lo = ES - mu_X + q_X_lo(p**)      # one big claim down (signed)

        Parameters
        ----------
        p_star : float
            Aggregate coverage to guarantee (e.g. ``1 - 1e-12``); ``> 1`` is
            read as a number of nines.

        Returns
        -------
        (sbj_lo, sbj_hi) : tuple of float, or None
            The single-big-jump window edges (``sbj_lo == 0`` for a non-signed
            severity). ``None`` when ``E[N]``, the severity mean, or the
            ``p**`` quantile is unavailable (e.g. no finite variance) -- the
            caller then keeps the existing window / moment-sizer path.

        Notes
        -----
        ``p**`` deepens with ``E[N]``; at ``E[N]=5000`` and
        ``p_star = 1 - 1e-12``, ``1 - p** = 2e-16`` is past double precision and
        ``q_X(p**) -> inf`` for an unbounded severity. The lower tail
        ``1 - p**`` is therefore floored at ``SBJ_TAIL_FLOOR``
        (``discretization.sbj_tail_floor``) and ``q_X_hi`` is capped by any
        finite policy limit (via :meth:`_severity_high_estimate`).
        """
        en = float(self.n)
        if not (np.isfinite(en) and en >= 1.0):
            return None
        try:
            sev_m = float(self.stats_df['mixed'][('sev', 'mean')])
        except (KeyError, ValueError, TypeError):
            return None
        es = float(self.actual_m)
        if not (np.isfinite(sev_m) and np.isfinite(es)):
            return None
        # p** with the author's numerical-depth guard: 1 - p** = (1-p*)/E[N]
        # floored at SBJ_TAIL_FLOOR so q_X(p**) stays finite for an unbounded
        # severity.
        p_star = float(np.where(p_star > 1, 1.0 - 10.0 ** -p_star, p_star))
        tail = max((1.0 - p_star) / en, SBJ_TAIL_FLOOR)
        p2 = 1.0 - tail
        q_hi = self._severity_high_estimate(p2)
        if not np.isfinite(q_hi):
            return None
        sbj_hi = es - sev_m + q_hi
        if self._signed_severity():
            q_lo = self._severity_low_estimate(tail)
            sbj_lo = (es - sev_m + q_lo) if np.isfinite(q_lo) else q_lo
        else:
            sbj_lo = 0.0
        if not np.isfinite(sbj_lo):
            return None
        return float(sbj_lo), float(sbj_hi)

    def _clipped_mass_estimate(self, grid_top):
        """Estimate the aggregate mass above ``grid_top`` via the single big jump.

        When the far right tail does not fit the grid (item 6), the clipped
        aggregate mass is dominated by the single-big-jump mechanism: a grid top
        of ``grid_top`` corresponds to one big claim of size
        ``grid_top - (ES - mu_X)`` on an otherwise typical bulk, so
        ``P(S > grid_top) ~ E[N]·P(X > grid_top - ES + mu_X)``. Summed (the mix
        is additive in the survival), capped at 1.

        Parameters
        ----------
        grid_top : float
            The realized grid's upper edge.

        Returns
        -------
        float
            The estimated clipped mass in ``[0, 1]``, or ``nan`` if it cannot be
            formed (no finite ``E[N]`` / severity mean).
        """
        en = float(self.n)
        try:
            sev_m = float(self.stats_df['mixed'][('sev', 'mean')])
        except (KeyError, ValueError, TypeError):
            return np.nan
        es = float(self.actual_m)
        if not (np.isfinite(en) and en >= 1.0
                and np.isfinite(sev_m) and np.isfinite(es)):
            return np.nan
        x_claim = grid_top - es + sev_m
        sevs = self.sevs if self.sevs is not None else []
        sf = 0.0
        for s in sevs:
            try:
                sf += float(s.fz.sf(x_claim))
            except Exception:  # pragma: no cover - defensive
                return np.nan
        n_comp = max(len(sevs), 1)
        return float(min(en * sf / n_comp, 1.0))

    def _exact_discrete_window(self):
        """Exact aggregate support for a fully-discrete ``dfreq``/``fixed`` x ``dsev``.

        Returns ``(A_min, A_max, bs_lattice, logp_lo, logp_hi)`` when the
        frequency is discrete-finite (``dfreq`` -> ``empirical``, or ``fixed``)
        **and** every severity component is a discrete histogram on an integer
        lattice; otherwise ``None``. The aggregate then takes values exactly on
        the integer lattice and its support is finite and exactly computable.

        Notes
        -----
        With claim counts ``N`` (atoms, min ``N_min`` max ``N_max``, possibly
        including 0) and severity atoms ``s_min … s_max``:

        - ``A_max = max(N_max·s_max, N_min·s_max)`` (the sum is maximised by the
          largest atom repeated; over ``N`` the extreme is at ``N_max`` if
          ``s_max>0`` else ``N_min``); include ``0`` if ``0`` is a count atom.
        - ``A_min = min(N_max·s_min, N_min·s_min)`` symmetrically; include ``0``
          if ``0`` is a count atom.

        ``bs_lattice`` is 1 for integer atoms (the common case).

        **Corner reachability (``logp_lo`` / ``logp_hi``).** The exact support is
        only a trustworthy grid extent if the aggregate can *attain* its
        extremes. A non-zero corner ``B = N_ach·s_ext`` is reached only when the
        realized count is exactly ``N_ach`` **and** every one of those claims
        lands on the extreme atom ``s_ext``, so

        .. math::

            \\log_{10} P(\\text{corner})
              = \\log_{10} P(N = N_{ach}) + N_{ach}\\,\\log_{10} P(X = s_{ext}),

        where ``N_ach`` is the count that *realizes* that corner -- ``N_max`` for
        the outer extreme (largest positive ``s_max`` / most-negative ``s_min``),
        ``N_min`` for the inner extreme (a positive ``s_min`` / negative
        ``s_max``): the low bound of a non-negative book is the *fewest* claims of
        the smallest atom, far more likely than ``N_max`` of it. A corner pinned
        at ``0`` (an ``s = 0`` atom, or a zero count atom giving the empty sum) is
        always reachable, so its ``logp`` is ``0``. ``P(X = s_ext)`` is the
        (mixture) single-claim probability at the extreme atom, taken as the max
        over components (an upper bound, so the guard never over-rejects). For a
        large fixed/So count both corners underflow (a near-normal aggregate whose
        combinatorial support is vast but whose mass the CLT concentrates); the
        caller (:func:`_bucket_window.bs_window`) rejects ``exact_discrete`` when
        both fall below ``exact_discrete_reach_logp``.
        """
        freq = self.frequency
        if freq.freq_name in ('empirical', 'renewal'):
            # a renewal count IS an empirical frequency post-build (freq_a =
            # 0..kmax, freq_b = pN), so it gets the same exact-discrete window
            n_atoms = np.asarray(freq.freq_a, dtype=float)
            n_probs = np.asarray(freq.freq_b, dtype=float)
            has_zero = bool(np.any(n_atoms == 0))
        elif freq.freq_name == 'fixed':
            n_atoms = np.array([float(self.n)])
            n_probs = np.array([1.0])
            has_zero = (self.n == 0)
        else:
            return None
        if self.sevs is None or len(self.sevs) == 0:
            return None
        atoms = []
        for s in self.sevs:
            a = getattr(s, 'support_atoms', None)
            if a is None:
                return None    # a non-discrete component -> not exact
            atoms.append(np.asarray(a, dtype=float))
        s_all = np.concatenate(atoms)

        def _allint(x):
            return bool(np.allclose(x, np.round(x), atol=1e-9))

        if not (_allint(n_atoms) and _allint(s_all)):
            return None
        s_min, s_max = float(s_all.min()), float(s_all.max())
        n_min, n_max = float(n_atoms.min()), float(n_atoms.max())
        hi = max(n_max * s_max, n_min * s_max)
        lo = min(n_max * s_min, n_min * s_min)
        if has_zero:
            hi = max(hi, 0.0)
            lo = min(lo, 0.0)

        # ---- corner reachability (log10 probabilities) ------------------
        def _freq_logp(n):
            """log10 P(N = n) from the (empirical) count law; 0 for fixed."""
            mask = np.isclose(n_atoms, n)
            pn = float(n_probs[mask].sum()) if mask.any() else 0.0
            return np.log10(pn) if pn > 0.0 else -np.inf

        def _sev_atom_logp(v):
            """log10 (upper bound on) P(one claim == v) across mixture components."""
            pv = 0.0
            for s in self.sevs:
                fz = getattr(s, 'fz', None)
                xk = getattr(fz, 'xk', None)
                pk = getattr(fz, 'pk', None)
                if xk is None or pk is None:
                    continue
                m = np.isclose(np.asarray(xk, dtype=float), v)
                if m.any():
                    pv = max(pv, float(np.asarray(pk, dtype=float)[m].sum()))
            return np.log10(pv) if pv > 0.0 else -np.inf

        def _corner_logp(bound, s_ext, n_ach):
            # a bound pinned at 0 (0 atom / empty sum) is always attainable.
            if bound == 0.0:
                return 0.0
            return _freq_logp(n_ach) + n_ach * _sev_atom_logp(s_ext)

        n_hi = n_max if s_max >= 0 else n_min   # count realizing A_max
        n_lo = n_max if s_min <= 0 else n_min   # count realizing A_min
        logp_hi = _corner_logp(hi, s_max, n_hi)
        logp_lo = _corner_logp(lo, s_min, n_lo)
        return float(lo), float(hi), 1.0, float(logp_lo), float(logp_hi)

    def _bounded_severity_window(self, p):
        """Window for a bounded severity with a (possibly small) claim count.

        Returns ``(A_lo, A_hi)`` using the severity's bounded support and a high
        frequency quantile ``N_hi`` (from the analytic frequency moments), or
        ``None`` if the severity is not bounded. For a small claim count this
        bound is tight; for a large count the law of large numbers concentrates
        the aggregate far inside ``[0, N_hi·s_max]`` and the moment window is
        tighter -- the caller (``_bs_window``) only selects this method when it
        is at least as tight as the moment window.

        Also returns ``None`` when the computed edges are not both finite. A
        severity can pass the structural ``_severity_bounded`` test and still
        produce an infinite edge here: a signed severity reads its lower edge
        off ``fz.support()``, and a spliced or reflected law can report an
        infinite support end that the recorded ``limit`` hides. A window with
        an infinite edge is not a window, and it used to reach the sizer as
        ``int(inf)``; see ``dev/done/plan-signed-bounded-window-overflow.md``
        and the same hazard on the splice path in ``_severity.py``.
        """
        if self.sevs is None or len(self.sevs) == 0:
            return None
        # ``_severity_bounded`` rather than the public ``bounded``, and without
        # ``reference=``: the grid is sized on the atoms actually convolved, so
        # a materialized ``sev agg.NAME`` reference counts as the bounded thing
        # it is here, even though every ``bounded`` surface reports the
        # referenced object's support instead. See ``_tail_info``.
        if not all(_tail._severity_bounded(s) for s in self.sevs):
            return None
        s_his, s_los = [], []
        for s in self.sevs:
            hi = s.limit if np.isfinite(s.limit) else float(s.fz.support()[1])
            lo = float(s.fz.support()[0]) if getattr(s, 'signed', False) else 0.0
            s_his.append(hi)
            s_los.append(lo)
        s_max, s_min = max(s_his), min(s_los)
        if not (np.isfinite(s_max) and np.isfinite(s_min)):
            # Not boundedly windowable, whatever the structural test said:
            # fall back to the moment window rather than hand the sizer an
            # infinite span.
            return None
        # ``freq_moms`` consumes the BASE mean (``n`` is the realized one, which
        # under zm / zt would apply the modification a second time).
        f1, f2, f3 = self.frequency.freq_moms(self.base_mean)
        if self._freq_zm:
            # A zero-modified law is ``c`` times the base law on {1, 2, ...}, so
            # its *reach* is the base distribution's reach. The realized mean and
            # sd are dragged toward zero by the mass at 0 and badly understate the
            # upper quantile -- a 95%-at-zero count would size a 4-bucket grid for
            # a Poisson(4) body. Recover the base moments (``f_j = c * base_j``)
            # and take the quantile from those.
            c = self.frequency._zm_weight(self.base_mean)
            if c > 0:
                f1, f2 = f1 / c, f2 / c
        fsd = float(np.sqrt(max(f2 - f1 * f1, 0.0)))
        zN = ss.norm.isf(1 - p)
        n_hi = f1 + zN * fsd
        return float(min(0.0, n_hi * s_min)), float(n_hi * s_max)

    def _bs_window(self, log2, bs_in, x_min_in, bucket_sizing_p,
                   window_convention=None):
        """Decide ``(bs, log2, x_min)`` for ``update`` and build ``_bs_window_df``.

        Orchestrator delegated to :func:`_bucket_window.bs_window`; see there
        for the full method / selection documentation. Runs the sizing methods,
        populates ``self._bs_window_df``, and returns the chosen grid.
        """
        return _bucket_window.bs_window(self, log2, bs_in, x_min_in,
                                        bucket_sizing_p, window_convention)

    def aggregate_error_analysis(self, log2, bs2_from=None, **kwargs):
        """
        Analysis of aggregate error across a range of bucket sizes. If ``bs2_from
        is None`` size a starting bs from the analytic moment window
        (``estimate_agg_window``) and scan plus/minus 3 doublings. Note: if the
        distribution does not have a second moment, you must enter bs2_from.

        :param log2:
        :param bs2_from: lower bound on bs to use, in log2 terms; estimated from
          the analytic moment window if not input.
        :param kwargs: passed to ``update``

        """
        # copy of self, updating alters the internal state of an object
        cself = Aggregate(**self.spec)

        if bs2_from is None:
            if cself.actual_cv == np.inf:
                raise ValueError('Distribution must have variance to guess bucket size. '
                                 'Input bs2_from')
            # ``recommend_bucket`` retired (W10); size a starting bs from the
            # analytic 3-moment output window (``estimate_agg_window``) instead.
            _, _, _w = estimate_agg_window(
                self.actual_m, self.actual_m * self.actual_cv, self.actual_skew)
            bs = round_bucket(_w / (1 << log2))
            bs2 = int(np.log2(bs))
            bss = 2. ** np.arange(bs2 - 3, bs2 + 4)
        else:
            bss = 2. ** np.arange(bs2_from, bs2_from + 7)

        # analytic aggregate mean
        m = cself.actual_m
        # aggregate analysis
        agg_ans = []
        for bs in bss:
            cself.update(bs=bs, log2=log2, **kwargs)
            agg_ans.append([bs, m, cself.est_m,
                            cself.est_m - m, cself.est_m / m - 1])

        agg_df = pd.DataFrame(agg_ans,
                              columns=['bs', 'actual_m', 'est_m',
                                       'abs_m', 'rel_m', ])
        m = cself.sev_m
        agg_df['rel_h'] = agg_df.bs / 2 / m
        agg_df['rel_total'] = agg_df.rel_h * np.sign(agg_df.rel_m) + agg_df.rel_m
        agg_df = agg_df.set_index('bs')
        agg_df.columns = agg_df.columns.str.split('_', expand=True)
        agg_df.columns.names = ['view', 'stat']
        return agg_df

    def severity_error_analysis(self, sev_calc='round', discretization_calc='survival',
                                normalize=True):
        """
        Analysis of severity component errors, uses the current bs in self.
        Gives detailed, component by component, error analysis of severities.
        Includes discretization error (bs large relative to mean) and
        truncation error (tail integral large).

        Total S shows the aggregate not severity. Generally about self.n * (1 - sum_p)
        (per Feller).

        """
        truncation_point = self.bs * (1 << self.log2)
        wts = self.en / self.n
        beds = self.discretize(sev_calc=sev_calc,
                               discretization_calc=discretization_calc,
                               normalize=normalize)
        sev_ans = []
        total_row = len(self.sevs)
        for i, (s, wt, en, bed) in enumerate(zip(self.sevs, wts, self.en, beds)):
            # exact theoretical sev mean from the canonical stats_df: the
            # per-component column label is ``e{e}.m{m}`` (exposure × sev-
            # mixture), enumerated in order by both broadcasting arms.
            label = self._comp_cols[i]
            m = self.stats_df.loc[('sev', 'ex1'), label]
            if len(self.sevs) == 1:
                i = self.name
            # estimated
            em, _ = xsden_to_meancv(self.xs, bed)
            sev_ans.append([s.long_name,
                            s.limit, s.attachment,
                            truncation_point,
                            s.sf(truncation_point), bed.sum(),
                            wt, en,
                            m, 0,
                            m, em
                            ])
        # the total
        m = self.sev_m
        # attachment is None if the limit clause is missing
        min_attach = np.where(self.attachment==None, 0., self.attachment).min()
        sev_ans.append(['total',
                        self.limit.max(), min_attach,
                        truncation_point,
                        self.sf(truncation_point), self.sev_density_df.p_sev.sum(),
                        1, self.n,
                        m, 0.,
                        m, self.est_sev_m
                        ])

        sev_df = pd.DataFrame(sev_ans,
                              columns=['name',
                                       'limit', 'attachment',
                                       'trunc',
                                       'S', 'sum_p',
                                       'wt', 'en',
                                       'agg_mean', 'agg_wt',
                                       'mean', 'est_mean'
                                       ],
                              index=range(total_row + 1))
        sev_df['agg_mean'] *= sev_df['en']
        sev_df['agg_wt'] = sev_df['agg_mean'] / \
                           sev_df.loc[0:total_row - 1, 'agg_mean'].sum()
        sev_df['abs'] = sev_df['est_mean'] - sev_df['mean']
        sev_df['rel'] = sev_df['abs'] / sev_df['mean']
        sev_df['trunc_error'] = \
            [_integral_by_doubling(s.sf, truncation_point) for s in self.sevs] + \
            [_integral_by_doubling(self.sev.sf, truncation_point)]
        sev_df['rel_trunc_error'] = sev_df.trunc_error / sev_df['mean']
        sev_df['h_error'] = self.bs / 2
        sev_df['rel_h_error'] = self.bs / 2 / sev_df['mean']

        # compute discretization_err_2 (was a separate function in development)
        xs = np.hstack((self.xs - self.bs / 2, self.xs[-1] + self.bs / 2))
        ans = []
        for s in self.sevs:
            # density at xs
            f = s.pdf(xs)
            # derv of f = -S''
            df = np.gradient(f, self.bs)
            # integral to quadratic adjustment term approx to S
            ans.append(np.sum(df) * self.bs ** 3 / 24)
        ans = pd.Series(ans)

        sev_df['h2_adj'] = np.hstack((ans, 0.))
        sev_df.loc[total_row, 'h2_adj'] = \
            sev_df.loc[0:total_row - 1, ['wt', 'h2_adj']].prod(1).sum()
        sev_df['rel_h2_adj'] = sev_df['h2_adj'] / sev_df['mean']

        return sev_df

    def q(self, p, kind='lower'):
        """
        Return quantile function of density_df.p_total.

        Definition 2.1 (Quantiles)
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] ≥ α} is the lower α-quantile of X
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] > α} is the upper α-quantile of X.

        ``kind=='middle'`` has been removed.

        :param p:
        :param kind: 'lower' or 'upper'.
        :return:
        """

        if kind == 'middle' and getattr(self, 'middle_warning', 0) == 0:
            self.middle_warning = 1

        if kind == 'middle':
            kind = 'lower'

        assert kind in ['lower', 'upper'], 'kind must be lower or upper'

        return self._grid_distribution().q(p, kind)

    def _grid_distribution(self):
        """The :class:`GridDistribution` view over the aggregate ``p_total`` grid.

        Lazily built and cached on first use (and after :meth:`update`, which
        resets the cache to ``None``). Built on the **full** contiguous ``bs``
        grid (zero-mass buckets included): the var/tvar kernel filters to the
        positive-mass subset internally (:meth:`GridDistribution._funcs`), so
        ``q``/``tvar`` are unchanged, while the width-summing ``lev`` /
        ``cdf`` / ``sf`` need the full grid to match the ``exa`` / ``add_exa``
        convention -- dropping the empty low buckets (where ``S == 1``) would
        make ``lev`` undercount. (Pre-1.0.0a97 this filtered ``p_total > 0`` up
        front, which silently broke ``lev`` on the subset grid; the filter moved
        inside the kernel where it belongs.)
        """
        if self._dist is None:
            self._dist = GridDistribution.from_series(
                self.density_df.p_total, bs=self.bs, name=self.name,
                is_loss_value=self._is_loss_value)
        return self._dist

    def _sev_grid_distribution(self):
        """The :class:`GridDistribution` view over the severity ``p_sev`` grid.

        Severity has its own grid (``sev_density_df``); this is the discretised
        *output* severity PMF (the uniform ``bs``-grid fed to the FFT), distinct
        from the input ``self.fz``. Lazily built and cached.

        Carries the **aggregate's** orientation (``self._is_loss_value``), not an
        intrinsic loss role: the severity curve shares the aggregate's Lee panel
        and must spread the same tail, so a payoff aggregate draws its severity
        with the payoff convention too. The objective accessors that read this GD
        (:meth:`q_sev`, :meth:`tvar_sev`) are sign-agnostic, so the role only
        affects :meth:`GridDistribution.return_period`.
        """
        if self._sev_dist is None:
            ser = self.sev_density_df.query('p_sev > 0').p_sev
            self._sev_dist = GridDistribution.from_series(
                ser, bs=self.bs, name=f'{self.name} sev',
                is_loss_value=self._is_loss_value)
        return self._sev_dist

    def center_window(self, p=1e-6):
        """Return the central window of ``density_df`` holding ``1 - p`` of the mass.

        A thin, no-recompute re-slicer over the finished aggregate: it runs
        :func:`~aggregate.utilities.balanced_window` on the realized
        ``p_total`` and returns the rows of :attr:`density_df` in the equal-tail
        window ``[q(p/2), q(1 - p/2)]`` -- ``p/2`` of the mass trimmed off each
        tail, ``1 - p`` kept and centred. Useful for tightening the display
        window of any computed aggregate (and the post-calc primitive the
        bivariate axis sizing is built on).

        Parameters
        ----------
        p : float, default 1e-6
            Total discarded tail mass, split equally between the two tails. Must
            satisfy ``0 < p < 1``. The literal discarded mass, not a coverage --
            see :func:`~aggregate.utilities.balanced_window`.

        Returns
        -------
        pandas.DataFrame
            The slice ``density_df.loc[lo:hi]`` (a view onto the existing frame;
            no recomputation). All columns are preserved.

        Notes
        -----
        Edges are snapped to the bucket size ``bs`` so the window aligns with
        the grid. Does not mutate the aggregate -- ``density_df`` is unchanged.
        """
        if self.density_df is None:
            raise ValueError('Must update before calling center_window.')
        ser = self.density_df.query('p_total > 0').p_total
        lo, hi = balanced_window(ser, p, bs=self.bs)
        return self.density_df.loc[lo:hi]

    def q_sev(self, p):
        """
        Compute quantile of severity distribution, returning element in the index.
        Very similar code to q, but only lower quantiles.

        :param p:
        :return:
        """

        return self._sev_grid_distribution().q(p, 'lower')

    def tvar_sev(self, p):
        """
        TVaR of severity - now available for free!

        added June 2023

        Fixed 1.0.0a91: previously this read the *aggregate* var/tvar cache
        (``_var_tvar_function['tvar']`` built from ``p_total``) rather than the
        severity grid, so it returned the aggregate TVaR. It now correctly uses
        the severity grid (``sev_density_df.p_sev``) -- the numbers move.
        """
        return self._sev_grid_distribution().tvar(p)

    def tvar(self, p, kind=''):
        """
        Updated June 2023, 0.13.0

        Compute the tail value at risk at threshold p

        Definition 2.6 (Tail mean and Expected Shortfall)
        Assume E[X−] < ∞. Then
        x¯(α) = TM_α(X) = α^{−1}E[X 1{X≤x(α)}] + x(α) (α − P[X ≤ x(α)])
        is α-tail mean at level α the of X.
        Acerbi and Tasche (2002)

        We are interested in the right hand exceedence [?? note > vs ≥]
        α^{−1}E[X 1{X > x(α)}] + x(α) (P[X ≤ x(α)] − α)

        McNeil etc. p66-70 - this follows from def of ES as an integral
        of the quantile function

        q is exact quantile (most of the time)
        q1 is the smallest index element (bucket multiple) greater than or equal to q

        tvar integral is int_p^1 q(s)ds = int_q^infty xf(x)dx = q + int_q^infty S(x)dx
        we use the last approach. np.trapz approxes the integral. And the missing piece
        between q and q1 approx as a trapezoid too.

        :param p:
        :param kind:
        :return:
        """
        if kind != '' and getattr(self, 'c', None) is None:
            logger.warning('kind is no longer used in TVaR, new method equivalent to kind=tail but much faster. '
                           'Argument kind will be removed in the future.')
            self.c = 1

        if kind == 'inverse':
            logger.warning('kind=inverse called...??!!')

        assert self.density_df is not None, 'Must recompute prior to computing tail value at risk.'

        return self._grid_distribution().tvar(p)

    def sample(self, n, replace=True):
        """
        Draw a sample of n items from the aggregate distribution. Wrapper around
        pd.DataFrame.sample.


        """

        if self.density_df is None:
            raise ValueError('Must update before sampling.')
        return self.density_df[['loss']].sample(n=n, weights=self.density_df.p_total,
                                                replace=replace, random_state=ar.RANDOM,
                                                ignore_index=True)

    @property
    def sev(self):
        """The *exact* continuous severity distribution, as a set of functions.

        The look-through past the discretization: the weighted mixture of the
        component :class:`~aggregate.Severity` objects evaluated from the input
        distributions, not from the ``bs``-grid PMF in ``sev_density_df``. Use
        it whenever the answer should not be quantized to a bucket.

        Returns
        -------
        SevFunctions
            A namedtuple of five callables, each accepting a scalar or an
            array:

            ``cdf(x)``, ``sf(x)``, ``pdf(x)``
                The forward functions.
            ``ppf(p)``, ``isf(s)``
                Their inverses. ``ppf`` is the quantile function indexed by
                non-exceedance probability, ``isf`` by exceedance probability.
                Added in 1.0.0a180.

        See Also
        --------
        Aggregate.q_sev : the same quantile snapped to the ``bs`` grid.

        Notes
        -----
        **Exact versus grid.** :meth:`q_sev` reads the discretized severity PMF
        through a :class:`~aggregate._grid_distribution.GridDistribution`, so it
        can only ever return a lattice point. ``sev.ppf`` calls the underlying
        continuous distribution, so it returns the true quantile. On
        ``agg T1 2 claims sev lognorm 1000 cv 1.31 poisson`` at ``bs = 8``,
        ``sev.ppf(0.99) = 6207.853`` against ``q_sev(0.99) = 6208``. The grid
        answer is the right one when the question is about the computed
        aggregate; the exact one is right when the question is about the input
        severity, as it is for occurrence PML and exceedance curves. See
        :func:`~aggregate.utilities.oep`.

        **Single component.** The five functions are the component
        :class:`~aggregate.Severity` methods directly. ``Severity`` subclasses
        :class:`scipy.stats.rv_continuous` and routes ``ppf`` / ``isf`` through
        its layered overrides, so limits, attachments and splices are all
        respected.

        **Mixture.** The forward functions are the ``wts``-weighted sums of the
        component functions. The inverses have no closed form and are solved by
        :func:`_mixture_inverse`, which brackets the root with the component
        inverses and calls :func:`scipy.optimize.brentq`. All five close over
        the *same* ``wts`` vector, so the inverses invert exactly the mixture
        the forward functions evaluate.

        **Weights.** ``sev_wt`` carries the mixture weights directly. Under
        broadcasting (exposure crossed with severity) the per-component
        ``sev_wt`` are all 1, so their sum is the component count; that case is
        detected and the weights are taken from the per-component expected
        claim counts instead.
        """
        if self._sev is None:
            if len(self.sevs) == 1:
                self._sev = SevFunctions(cdf=self.sevs[0].cdf, sf=self.sevs[0].sf,
                                         pdf=self.sevs[0].pdf, ppf=self.sevs[0].ppf,
                                         isf=self.sevs[0].isf)
            else:
                # multiple severites, needs more work
                wts = np.array([i.sev_wt for i in self.sevs])
                # for non-broadcast weights the sum is n = number of components; rescale
                if wts.sum() == len(self.sevs):
                    wts = self.stats_df.loc[('freq', 'ex1'), self._comp_cols].values
                wts = wts / wts.sum()

                # tried a couple of different approaches here and this is as fast as any
                def _sev_cdf(x):
                    return np.sum([wts[i] * self.sevs[i].cdf(x) for i in range(len(self.sevs))], axis=0)

                def _sev_sf(x):
                    return np.sum([wts[i] * self.sevs[i].sf(x) for i in range(len(self.sevs))], axis=0)

                def _sev_pdf(x):
                    return np.sum([wts[i] * self.sevs[i].pdf(x) for i in range(len(self.sevs))], axis=0)

                # inverses: bracket with the component inverses, then brentq
                def _sev_ppf(p):
                    return _mixture_inverse(p, _sev_cdf, [s.ppf for s in self.sevs],
                                            increasing=True)

                def _sev_isf(s):
                    return _mixture_inverse(s, _sev_sf, [sv.isf for sv in self.sevs],
                                            increasing=False)

                self._sev = SevFunctions(cdf=_sev_cdf, sf=_sev_sf, pdf=_sev_pdf,
                                         ppf=_sev_ppf, isf=_sev_isf)
        return self._sev

    def cdf(self, x, kind='previous'):
        """
        Return cumulative probability distribution at x using kind interpolation.

        2022-10 change: kind introduced; default was linear

        :param x: loss size
        :return:
        """
        if self._cdf is None:
            self._cdf = interpolate.interp1d(self.xs, self.agg_density.cumsum(), kind=kind,
                                             bounds_error=False, fill_value='extrapolate')
        # 0+ converts to float
        return 0. + self._cdf(x)

    def sf(self, x):
        """
        Return survival function using linear interpolation.

        :param x: loss size
        :return:
        """
        return 1 - self.cdf(x)

    def pdf(self, x):
        """
        Probability density function, assuming a continuous approximation of the bucketed density.

        :param x:
        :return:
        """
        if self._pdf is None:
            self._pdf = interpolate.interp1d(self.xs, self.agg_density, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
        return self._pdf(x) / self.bs

    def pmf(self, x):
        """
        Probability mass function, treating aggregate as discrete
        x must be in the index (?)

        """
        if self.density_df is None:
            raise ValueError("Must update before computing probabilities!")

        try:
            return self.density_df.loc[x, 'p_total']
        except KeyError:
            return 0.0
            # raise KeyError(f'Value {x} must be in index for probability mass function.')

    def json(self):
        """
        Write spec to json string.

        :return:
        """
        return json.dumps(self._spec)

    def approximate(self, approx_type='slognorm', output='scipy'):
        """
        Create an approximation to self using method of moments matching.

        Compare to Portfolio.approximate which returns a single sev fixed freq agg, this
        returns a scipy dist by default.

        Use case: exam questions with the normal approximation!

        :param approx_type: norm, lognorm, slognorm (shifted lognormal), gamma, sgamma. If 'all'
            then returns a dictionary of each admissible approx (families that
            cannot be represented for this distribution/output are skipped).
        :param output: scipy - frozen scipy.stats continuous rv object;
          sev_decl - DecL severity fragment (to substitute into an agg; no name)
          sev_kwargs - dictionary of parameters to create Severity
          agg_decl - DecL program ``agg NM 1 claim sev ... fixed``
          agg (or any other string) - Aggregate object built from the agg_decl
          program via ``build``, so it is parser-born: nonempty ``program``,
          decompiles, and lands in the underwriter's knowledge under the fit
          name.
        :return: as above.

        A shifted family (``slognorm`` / ``sgamma``) requested for a symmetric
        distribution degenerates to its normal limit and emits a ``UserWarning``
        (pass ``approx_type='norm'`` to select the normal explicitly). A
        left-skewed distribution is fitted by reflection and renders in DecL
        through the ordinary reflection syntax ``loc - scale * name shape``;
        only ``output='scipy'`` raises ``ValueError`` for it (scipy has no
        frozen reflected rv).

        **The emitted severity keyword mirrors the input.** The DecL and object
        outputs spell the severity ``ssev`` when this aggregate is signed
        (:meth:`_signed`), else plain ``sev``; ``output='sev_kwargs'`` mirrors
        the same rule through ``sev_signed``. A fit whose law reaches below
        zero emitted under plain ``sev`` clamps its sub-zero tail to an atom at
        0 (a loss stays a loss), at the cost of a small moment drift that the
        built surrogate's ``validation_df`` reports.
        """
        # Prefer empirical moments (post-update) over theoretical (pre-update).
        emp = self.stats_df['empirical']
        if pd.notna(emp.get(('agg', 'mean'), np.nan)):
            m, cv, skew = (emp[('agg', 'mean')], emp[('agg', 'cv')], emp[('agg', 'skew')])
        else:
            mixed = self.stats_df['mixed']
            m, cv, skew = (mixed[('agg', 'mean')], mixed[('agg', 'cv')], mixed[('agg', 'skew')])
        note = f'frozen version of {self.name}'

        def _one(kind, warn):
            nm = f'{kind[0:4]}.{self.name[0:5]}'
            return approximate_from_mcvsk(m, cv, skew, nm, f'agg {nm} 1 claim ',
                                          note, kind, output, warn_degenerate=warn,
                                          signed_input=self._signed())

        if approx_type == 'all':
            # Survey: stay quiet about degeneration, and skip a family that
            # cannot be represented for this distribution/output (after the
            # reflected DecL rendering landed, only a reflected fit requested
            # as a frozen scipy rv).
            out = {}
            for kind in ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']:
                try:
                    out[kind] = _one(kind, warn=False)
                except ValueError:
                    continue
            return out
        return _one(approx_type, warn=True)

    def entropy_fit(self, n_moments, tol=1e-10, verbose=False):
        """
        Find the max entropy fit to the aggregate based on n_moments fit.
        The constant is added (sum of probabilities constraint), for two
        moments there are n_const = 3 constrains.

        Based on discussions with, and R code from, Jon Evans

        Run ::

            ans = obj.entropy_fit(2)
            ans['ans_df'].plot()

        to compare the fits.

        :param n_moments: number of moments to match
        :param tol:
        :param verbose:
        :return:
        """
        # sum of probs constraint
        n_constraints = n_moments + 1

        # don't want to mess up the object...
        xs = self.xs.copy()
        p = self.agg_density.copy()
        # de-fuzz before the moment fit (threshold standardized on machine eps)
        p = remove_fuzz(p)
        p = p / np.sum(p)
        p1 = p.copy()

        mtargets = np.zeros(n_constraints)
        for i in range(n_constraints):
            mtargets[i] = np.sum(p)
            p *= xs

        parm1 = np.zeros(n_constraints)
        x = np.array([xs ** i for i in range(n_constraints)])

        probs = np.exp(-x.T @ parm1)
        machieved = x @ probs
        der1 = -(x * probs) @ x.T

        er = 1
        iters = 0
        while er > tol:
            iters += 1
            try:
                parm1 = parm1 - inv(der1) @ (machieved - mtargets)
            except np.linalg.LinAlgError:
                print('Singluar matrix')
                print(der1)
                return None
            probs = np.exp(-x.T @ parm1)
            machieved = x @ probs
            der1 = -(x * probs) @ x.T
            er = (machieved - mtargets).dot(machieved - mtargets)
            if verbose:
                print(f'Error: {er}\nParameter {parm1}')
        ans = pd.DataFrame(dict(xs=xs, agg=p1, fit=probs))
        ans = ans.set_index('xs')
        return dict(params=parm1, machieved=machieved, mtargets=mtargets, ans_df=ans)

    def var_dict(self, p, kind='lower', snap=False):
        """
        Make a dictionary of value at risks for the aggregate, mirrors Portfolio.var_dict.
        Here is just marshals calls to the appropriate var or tvar function.

        Allows the price function to run consistently with Portfolio version.

        Example Use: ::

            for p, arg in zip([.996, .996, .996, .985], ['var', 'lower', 'upper', 'tvar']):
                print(port.var_dict(p, arg,  snap=True))

        :param p:
        :param kind: var (defaults to lower), upper, lower, tvar
        :param snap: snap tvars to index
        :return:
        """
        if kind == 'var': kind = 'lower'
        if kind == 'tvar':
            d = {self.name: self.tvar(p)}
        else:
            d = {self.name: self.q(p, kind)}
        if snap and kind == 'tvar':
            d = {self.name: self.snap(d[self.name])}
        return d

    def price(self, p, g, kind='var'):
        """
        Price using regulatory and pricing g functions, mirroring Portfolio.price.
        Unlike Portfolio, cannot calibrate. Applying specified Distortions only.
        If calibration is needed, embed Aggregate in a one-line Portfolio object.

        Compute E_price (X wedge E_reg(X) ) where E_price uses the pricing distortion and E_reg uses
        the regulatory distortion.

        Regulatory capital distortion is applied on unlimited basis: ``reg_g`` can be:

        * if input < 1 it is a number interpreted as a p value and used to determine VaR capital
        * if input > 1 it is a directly input  capital number
        * d dictionary: Distortion; spec { name = dist name | var, shape=p value a distortion used directly

        ``pricing_g`` is  { name = ph|wang and shape=}, if shape (lr or roe not allowed; require calibration).

        if ly, must include ro in spec

        :param p: a distortion function spec or just a number; if >1 assets, if <1 a prob converted to quantile
        :param kind: var lower upper tvar
        :param g:  pricing distortion function
        :return:
        """
        # Thin delegator to the shared single-distribution pricing concern.
        return _pricing.price(self, p, g, kind)

    def price_pentagon(self, *, p=None, a=None, P=None, M=None, Q=None,
                       LR=None, PQ=None, ROE=None, reins_view=None):
        """Complete the pricing octet at a capital level given one target.

        Fix the capital level with exactly one of ``p`` (a VaR probability,
        ``a = self.q(p)``) or ``a`` (an asset level, snapped to the grid), then
        supply exactly one pricing target -- premium ``P``, cost of capital
        ``ROE`` (a.k.a. CoC), or a loss ratio via ``LR`` (equivalently ``M``,
        ``Q`` or ``PQ``). Returns the canonical one-row (``'total'``) pentagon
        ``DataFrame`` (columns :data:`~aggregate.pentagon.PENTAGON_STATS`).
        The target keywords match the canonical stat names.

        Pure accounting completion against the object's expected loss at the
        chosen capital level -- **no distortion is involved** (contrast
        :meth:`price`, which prices with a :class:`Distortion`). The triple
        ``{L, a, target}`` is solved by :meth:`Pentagon.solve`.

        Parameters
        ----------
        p : float, optional
            VaR probability fixing the capital level; mutually exclusive with ``a``.
        a : float, optional
            Asset level fixing the capital; mutually exclusive with ``p``.
        P, M, Q, LR, PQ, ROE : float, optional
            Exactly one pricing target -- premium, margin, capital, loss ratio,
            premium-to-capital, or cost of capital (``M/Q``).
        reins_view : str, optional
            Which of a cession's distributions to complete the octet on, one
            of :attr:`reins_views`. The default ``None`` is the object's own.
            Both the asset level and the expected loss resolve on the chosen
            view, matching :meth:`calibrate_distortions`.

        Returns
        -------
        pandas.DataFrame
            One ``'total'`` row, eight canonical pentagon columns.

        Raises
        ------
        ValueError
            If not exactly one of ``p``/``a`` is given, or not exactly one
            pricing target is supplied.
        """
        # Thin delegator to the shared single-distribution pricing concern.
        return _pricing.price_pentagon(
            self, p=p, a=a, P=P, M=M, Q=Q, LR=LR, PQ=PQ, ROE=ROE,
            reins_view=reins_view)

    def price_ccoc(self, ccoc, *, p):
        """Price at a constant cost of capital ``ccoc`` and VaR level ``p``.

        No distortion is involved -- a thin alias for
        ``self.price_pentagon(p=p, ROE=ccoc)`` returning the canonical one-row
        (``'total'``) pentagon ``DataFrame``. Parity with
        :meth:`Portfolio.price_ccoc`.
        """
        return _pricing.price_ccoc(self, ccoc, p=p)

    def prob_loss_assets(self, *, p=None, L=None, a=None):
        """Given any one of ``p``, ``L``, ``a``, return the consistent triple.

        Free choice over the capital anchor: pass exactly one of the VaR
        probability ``p``, the limited expected loss ``L = E[min(X, a)]``, or
        the asset level ``a`` -- any one determines the other two. Thin
        delegator to
        :meth:`~aggregate._grid_distribution.GridDistribution.prob_loss_assets`
        over this aggregate's ``p_total`` grid (the single ``lev`` source).

        Returns
        -------
        ProbLossAssets
            Namedtuple ``(p, L, a)``, grid-snapped and mutually consistent
            (``L == lev(a)``, ``p == cdf(a)``).
        """
        return self._grid_distribution().prob_loss_assets(p=p, L=L, a=a)


    def price_pentagon_ex(self, *, p=None, a=None, L=None,
                          M=None, P=None, Q=None, LR=None, PQ=None,
                          ROE=None, reins_view=None):
        """Complete the pricing octet over the full pentagon vocabulary.

        The full-power front door over :meth:`price_pentagon`: free over the
        capital anchor (``p``, ``a``, **or** the limited expected loss ``L``)
        and accepting any soluble pentagon configuration, warning when an
        accounting-determined ``L`` does not reconcile with ``E[min(X, a)]``.
        Thin delegator to the shared single-distribution pricing concern; see
        :func:`aggregate._pricing.price_pentagon_ex` for the full contract.

        Returns
        -------
        pandas.DataFrame
            One ``'total'`` row: a leading ``p`` column then the eight canonical
            pentagon stats.
        """
        return _pricing.price_pentagon_ex(
            self, p=p, a=a, L=L, M=M, P=P, Q=Q, LR=LR, PQ=PQ, ROE=ROE,
            reins_view=reins_view)

    def calibrate_distortions(self, coc=None, *, lr=None, p=None, a=None,
                              kind='lower',
                              names=_pricing.DEFAULT_CALIBRATION_DISTORTIONS,
                              reins_view=None):
        """Calibrate the standard pricing distortion set to a pricing target.

        The ``Aggregate`` counterpart of :meth:`Portfolio.calibrate_distortions`
        -- calibration to **this** distribution (the aggregate is its own
        total), with no per-unit allocation (that stays a ``Portfolio``
        concern). Calibrating directly here means no more wrapping a single
        ``Aggregate`` in a one-unit ``Portfolio`` to obtain a calibrated
        distortion set.

        Parameters
        ----------
        coc : float, optional
            Target cost of capital ``COC = (P - L) / Q``. Exactly one of
            ``coc`` or ``lr``.
        lr : float, optional
            Target loss ratio ``LR = L / P``, converted to a cost of capital
            through the pentagon at the resolved anchor. Exactly one of
            ``coc`` or ``lr``.
        p : float, optional
            Probability at which the calibration applies; converted to an asset
            level via ``self.q(p, kind)``. Exactly one of ``p`` or ``a``.
        a : float, optional
            Asset level; snapped to the grid. Exactly one of ``p`` or ``a``.
        kind : {'lower', 'upper'}, optional
            VaR kind when ``p`` is provided. Default ``'lower'``.
        reins_view : str, optional
            Which of a cession's distributions to calibrate on, one of
            :attr:`reins_views`. The default ``None`` is this aggregate's own
            density, which under a cession is whichever view the program asked
            for: a ``net of`` program holds its net, a ``ceded to`` program
            holds its ceded.

        Returns
        -------
        CalibrationResult
            Carrying ``distortion_df`` (one row per distortion in
            ``[ccoc, ph, wang, dual, tvar]``), the shared one-row
            ``calibration_df`` target, the ``distortions`` themselves, and the
            inputs they were fitted at. The same three are stored on
            ``self.distortion_df`` / ``self.calibration_df`` /
            ``self.distortions`` as before, so nothing that read them moves;
            both frames carry ``attrs['reins_view']``, recording which
            distribution they were fitted to. Same schema as
            :meth:`Portfolio.calibrate_distortions`.

            The return type changed at 1.0.0a259: it was the bare
            ``distortion_df``, which is now an attribute of the result.

        Notes
        -----
        The expected loss anchoring the premium target is computed on the full
        aggregate grid (the ``E[min(X, a)]`` / ``add_exa`` convention), matching
        a one-unit ``Portfolio``'s ``exa_total``.

        ``names`` selects the distortion families to calibrate (default the
        standard set); signed / payoff supports are handled transparently (see
        :func:`aggregate._pricing.calibrate_distortions`).

        Calibrating gross and net separately and differencing the premiums is
        the allowance for reinsurance in the rate. The asset level resolves on
        the chosen view, so ``p=`` holds the threshold fixed across views
        rather than the capital.

        Examples
        --------
        ::

            a = build('agg Re 100 claims sev lognorm 50 cv 2 poisson '
                      'occurrence net of 100 xs 100')
            a.reins_views                                    # ['gross', 'ceded', 'net']
            gross = a.calibrate_distortions(0.10, p=0.999, reins_view='gross')
            net = a.calibrate_distortions(0.10, p=0.999, reins_view='net')
        """
        return _pricing.calibrate_distortions(self, coc, lr=lr, p=p, a=a,
                                              kind=kind, names=names,
                                              reins_view=reins_view)

    def evaluate(self, P=None, *, p=None, a=None, names=None,
                 reins_view=None):
        """Evaluate the position ``P - X``: the breakeven acceptability panel.

        Pricing asks what the obligation is worth; evaluation asks how much
        stress the position you hold survives. Solves, per distortion family,
        for the shape at which the risk-adjusted margin reaches 0. The
        breakeven ``gini_p`` is the family-agnostic acceptability index of
        Cherny and Madan.

        Parameters
        ----------
        P : float, optional
            The premium held against this aggregate. Defaults to
            :attr:`exp_premium`, and raises when that is unset: a position has
            to have a consideration before it can be evaluated.
        p : float, optional
            Asset probability, resolved on the evaluated distribution
            (``a = q(p)``). At most one of ``p`` or ``a``.
        a : float, optional
            Asset level, snapped to the grid. At most one of ``p`` or ``a``;
            with neither, the position is measured against its whole
            distribution, which is the unlimited reading.
        names : sequence of str, optional
            Distortion families. Defaults to
            :data:`~aggregate._pricing.EVAL_FAMILIES` (``ph`` / ``wang`` /
            ``dual`` / ``tvar``) unanchored, and to
            :data:`~aggregate._pricing.EVAL_FAMILIES_ANCHORED`, which adds
            ``ccoc``, when an anchor is given.
        reins_view : str, optional
            Which of a cession's distributions to evaluate, one of
            :attr:`reins_views`. The default ``None`` is this aggregate's own.
            ``P`` is **not** adjusted with the view: evaluating the gross
            distribution against the net premium asks what stress the position
            would survive if the cover failed to respond, which is a question
            worth being able to ask deliberately, and a wrong answer to ask by
            accident. Pass the premium that goes with the view.

        Returns
        -------
        EvaluationResult
            Carrying ``evaluation_df``, the panel: tidy (long) form,
            ``MultiIndex`` rows ``(Step, distortion)`` with ``Step`` this
            aggregate's name, and columns ``role`` / ``param_name`` /
            ``param`` / ``gini_p`` / ``error`` / ``status``. ``role`` is
            always ``'sell'`` here: an aggregate is an obligation written, so
            the position is held the way it is booked. The same shape
            :meth:`PnL.evaluate` and :meth:`Portfolio.evaluate` return, so
            panels concatenate. The result also carries the premium and the
            view the panel was measured on.

            The return type changed at 1.0.0a259: it was the bare panel, which
            is now ``.evaluation_df``. A frame cannot be dispatched on, and the
            evaluate exhibit dispatches on the result.

        Warns
        -----
        DegenerateEvaluationWarning
            When no breakeven level exists: ``P <= E[X]`` (unacceptable at any
            stress) or the position cannot lose (acceptable at every stress).
            Both report ``NaN``.

        Notes
        -----
        **Anchoring is what makes evaluation and calibration comparable.**
        :meth:`calibrate_distortions` solves ``rho_g(min(X, a)) = P`` at the
        asset level it resolved. Evaluating at the same anchor solves the same
        equation, so the panel recovers that calibration's parameters family
        for family, which is the round trip worth having. Unanchored, the
        implicit asset level is the top of the FFT grid, and the two answers
        differ by whatever the tail beyond ``a`` is worth.

        See Also
        --------
        aggregate._pricing.evaluate_margin : the solve and its math.
        calibrate_distortions : the pricing counterpart, given a CoC target.
        """
        if p is not None and a is not None:
            raise ValueError('evaluate: pass at most one of p= or a=.')
        _pricing.guard_unbounded_anchor(self, p, where='evaluate')
        P = self._resolve_evaluation_premium(P)
        s = (self.density_df['p_total'] if reins_view is None
             else self._reins_view_density(reins_view))
        assets, p_val = self._resolve_evaluation_assets(s, p, a)
        panel = _pricing.evaluate_constant_premium(
            s.index.to_numpy(dtype=float), s.to_numpy(dtype=float), self.bs, P,
            assets=assets, names=names)
        step = self.name if reins_view is None else f'{self.name} {reins_view}'
        panel = pd.concat([panel], keys=[step], names=['Step'])
        _pricing.warn_degenerate(panel, self.name)
        default_names = (_pricing.EVAL_FAMILIES if assets is None
                         else _pricing.EVAL_FAMILIES_ANCHORED)
        return EvaluationResult(
            evaluation_df=panel, premium=P, reins_view=reins_view,
            p=p_val, a=assets,
            names=tuple(names if names is not None else default_names),
            _source=self)

    def _resolve_evaluation_assets(self, density, p, a):
        """``(a, p)`` for an evaluation anchor, resolved on the density served.

        A cession makes several distributions available and the anchor belongs
        to the one being evaluated, not to the object's own, so the quantile
        comes off the density the caller's ``reins_view`` selected. Returns
        ``(None, None)`` when neither was given, the unlimited reading.
        """
        if p is None and a is None:
            return None, None
        gd = GridDistribution.from_series(
            density, bs=self.bs, name=self.name,
            is_loss_value=self._is_loss_value)
        if a is None:
            assets = float(gd.q(p))
            return assets, float(p)
        assets = float(gd.snap(a))
        return assets, float(gd.cdf(assets))

    def _resolve_evaluation_premium(self, P):
        """The consideration :meth:`evaluate` measures against: the argument,
        else :attr:`exp_premium`. Raises rather than guessing, since a premium
        of 0 would silently make every position unacceptable."""
        if P is None:
            P = float(np.sum(np.asarray(
                getattr(self, 'exp_premium', 0.0), dtype=float)))
            if P == 0:
                raise ValueError(
                    f'{self.name} has no premium to evaluate against: pass '
                    'P=, or declare one in DecL (e.g. "1000 premium at 0.7 '
                    'lr").')
        return float(P)
