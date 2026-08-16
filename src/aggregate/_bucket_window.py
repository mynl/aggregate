"""Bucket and window sizing for the aggregate FFT grid.

Extracted from ``distributions.py`` / ``_aggregate.py`` (Phase 1b, shared concerns). A leaf/near-leaf: it never imports ``_aggregate``/``_portfolio`` (takes plain data / a distribution object), which is what lets Portfolio reuse it in P4.
"""

import logging
import math
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.optimize import NoConvergence  # noqa
from .config import get_settings
from .constants import (InfiniteVarianceError, DefectiveDistributionWarning,
                        warn_once, warn_once_isolated)
from . import _validation
# Module scope is safe: _program imports only decl_writer, which imports nothing
# from aggregate at module scope, and _program reaches back here for _fmt_bs
# through a deferred import inside the function.
from . import _program
from .utilities import round_bucket, value_type_role
from . import tail as _tail
from ._fits import gamma_fit, lognorm_fit, sgamma_fit, sln_fit

logger = logging.getLogger(__name__)


# Config-backed module constants, resolved once per session from
# aggregate.config (see that module for the cascade and reload semantics).
#
# WINDOW_NINES: probability coverage for the automatic output WINDOW (number of
# nines): the window spans roughly the 10**-WINDOW_NINES .. 1-10**-WINDOW_NINES
# quantiles. Deliberately far tighter than BUCKET_SIZING_P (the bucket-sizing
# percentile) so a P&L / signed aggregate's window captures essentially all the
# mass.
WINDOW_NINES = get_settings().discretization.window_nines


# WINDOW_LOG2_GROWTH: how many powers of 2 the *windowed* sizing may grow log2
# past the requested cap to preserve an exact integer-lattice bs -- e.g. keep
# bs=1 for a high-mean ``dsev`` rather than coarsening to bs=2 and mis-placing
# the atoms at half-integer buckets. Small and bounded because the windowed band
# is provably narrow (a concentrated aggregate, actual_cv < 1/z); it never fires for
# a genuinely wide band, which coarsens bs as before.
WINDOW_LOG2_GROWTH = get_settings().discretization.window_log2_growth


# WINDOW_NINES_TRIM: coverage (number of nines) for the *unprotected* edge of a
# windowed two-sided aggregate -- the cheap tail the sign convention does not
# price (the left edge for a loss, the right for a payoff). Shallower than
# WINDOW_NINES so the window trims dead space on that side. Only affects a book
# whose mass band clears 0 (windowed); ordinary 0-based books are untouched.
WINDOW_NINES_TRIM = get_settings().discretization.window_nines_trim


# WINDOW_PAD_SKEW: padding-balance skew Delta. After a windowed band is placed,
# the power-of-2 slack is split with fraction f = 0.5 -/+ Delta below the band
# (loss -> 0.5 - Delta, more room above/right; payoff -> 0.5 + Delta). 0 centres
# the band; the legacy placement was f = 0 (all slack above the band).
WINDOW_PAD_SKEW = get_settings().discretization.window_pad_skew


# WINDOW_SLACK_THICK: for an ASYMMETRIC windowed band (one tail thick, one thin),
# the fraction of the power-of-2 slack placed on the *thick* side -- the tail
# that needs the room (item 4, the "right look"). The thin side gets the rest
# (1 - WINDOW_SLACK_THICK). Hardwired 3/4; the loss/payoff convention does not
# apply when the tails are asymmetric (the tail shape dictates placement),
# only as a tie-breaker for a symmetric band (WINDOW_PAD_SKEW).
WINDOW_SLACK_THICK = get_settings().discretization.window_slack_thick


# BUCKET_SIZING_P: percentile of the fitted distribution fed to the moment-window
# bucket sizer to size bs (formerly the recommend_bucket percentile). >1 is read as nines.
BUCKET_SIZING_P = get_settings().discretization.bucket_sizing_p


# EXACT_DISCRETE_REACH_LOGP: reachability floor (log10 probability) for the
# exact-discrete support corners. The finite combinatorial support of a
# fully-discrete aggregate is only a trustworthy grid extent when the mass can
# actually reach an extreme; a large fixed/So count makes every corner
# astronomically improbable (a near-normal aggregate whose support is vast but
# whose mass the CLT concentrates), so the exact support overstates the extent
# and coarsening bs to fit it aliases the severity. When both corners fall below
# this floor the method is rejected (see the selection block below).
EXACT_DISCRETE_REACH_LOGP = get_settings().discretization.exact_discrete_reach_logp


# SBJ_TAIL_FLOOR: deepest lower-tail probability (1 - p**) the single-big-jump
# extent floor will probe the severity at. The SBJ adjustment p** = 1 -
# (1-p*)/E[N] deepens with E[N]; this floors 1 - p** so q_X(p**) stays finite
# for an unbounded severity (the author's guard, §1A-fix). Default 1e-14.
SBJ_TAIL_FLOOR = get_settings().discretization.sbj_tail_floor


def _estimate_agg_percentile(m, cv, skew, p=0.999):
    """
    Come up with an estimate of the tail of the distribution based on the three parameter fits, ln and gamma

    Updated Nov 2022 with a way to estimate p based on lognormal results. How far in the
    tail you need to go to get an accurate estimate of the mean. See 2_x_approximation_error
    in the help.

    Retain p param for backwards compatibility.

    :param m:
    :param cv:
    :param skew:
    :param p: if > 1 converted to 1 - 10**-n
    :return:
    """

    # p_estimator = interp1d([0.53294, 0.86894, 1.9418, 7.3211, 22.738, 90.012, 457.14, 2981],
    #                        [3,       4,       5,      7,      8,      9,      11,     12],
    #                        assume_sorted=True, bounds_error=False, fill_value=(3, 13))
    # p = 1 - 10**-p_estimator(cv)

    if np.isinf(cv):
        raise ValueError('Infinite variance passed to estimate_agg_percentile')

    # make vectorizable
    p = np.array(p)
    p = np.where(p > 1, 1 - 10 ** -p, p)

    pn = pl = pg = 0
    if skew <= 0:
        # neither sln nor sgamma works, use a normal
        # for negative skewness the right tail will be thin anyway so normal not outrageous
        fzn = ss.norm(scale=m * cv, loc=m)
        pn = fzn.isf(1 - p)
    elif not np.isinf(skew):
        shift, mu, sigma = sln_fit(m, cv, skew)
        fzl = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
        shift, alpha, theta = sgamma_fit(m, cv, skew)
        fzg = ss.gamma(alpha, scale=theta, loc=shift)
        pl = fzl.isf(1 - p)
        pg = fzg.isf(1 - p)
    else:
        mu, sigma = lognorm_fit(m, cv)
        fzl = ss.lognorm(sigma, scale=np.exp(mu))
        alpha, theta = gamma_fit(m, cv)
        fzg = ss.gamma(alpha, scale=theta)
        pl = fzl.isf(1 - p)
        pg = fzg.isf(1 - p)
    # throw in a mean + 3 sd approx too...
    return np.maximum(np.maximum(pn, pl), np.maximum(pg, m * (1 + ss.norm.isf(1 - p) * cv)))


def estimate_agg_window(m, sd, skew, p=BUCKET_SIZING_P, p_lo=None, p_hi=None):
    """Two-sided output window ``[x_lo, x_hi]`` and width ``W`` for an aggregate.

    The signed counterpart of :func:`_estimate_agg_percentile`: where that
    returns a single upper bound on a non-negative aggregate, this returns
    *both* edges from method-of-moments fits, so a profit/loss aggregate (mean
    possibly < 0, mass possibly far from 0) can be placed on a tight window
    where its mass actually lives. Used by ``Aggregate.update(x_min=None)``.

    Parameters
    ----------
    m, sd, skew : float
        Analytic aggregate mean, **standard deviation** (not cv), and skewness.
        ``m`` may be negative or zero (a net-profit / mean-zero P&L); ``sd`` is
        taken directly so the mean-zero case -- where ``cv`` is undefined --
        works.
    p : float
        Coverage for *both* edges when ``p_lo`` / ``p_hi`` are not given.
        ``p > 1`` is read as ``1 - 10**-p`` (e.g. ``p=6`` -> ``1 - 1e-6``); the
        per-edge tail probability is ``1 - p``.
    p_lo, p_hi : float, optional
        Per-edge coverage. When supplied they override ``p`` on the lower /
        upper edge respectively, so the two tails can be covered to different
        depths -- the convention-skew lever (a loss covers its **upper** edge
        deep to avoid clipping the priced right tail, and trims the cheap lower
        edge shallow; a payoff mirrors). Same ``>1 -> nines`` reading as ``p``.

    Returns
    -------
    (x_lo, x_hi, W) : tuple of float
        Lower edge, upper edge, and width ``W = x_hi - x_lo``. The window is
        **not** forced to be centred on the mean -- for a skewed aggregate the
        two quantiles place it off-centre.

    Notes
    -----
    Three regimes (mirrors ``_estimate_agg_percentile`` but two-sided):

    - **Symmetric** (``|skew|`` below a tolerance, incl. any genuinely
      symmetric or mean-zero P&L): neither shifted-lognormal nor shifted-gamma
      is defined, so use a **normal** approximation for both edges
      ``m - z_lo*sd`` / ``m + z_hi*sd`` with ``z_e = norm.isf(1-p_e)``. This is
      the default fallback, not an afterthought.
    - **Right-skewed** (``skew > 0``): fit shifted lognormal and shifted gamma
      to ``(m, sd, skew)`` and take the **wider** window
      (``min`` of the low quantiles, ``max`` of the high quantiles).
    - **Left-skewed** (``skew < 0``): **reflect** -- fit to ``-A`` (mean
      ``-m``, ``skew -skew > 0``), then map the window back as
      ``[-hi_r, -lo_r]``.

    Falls back to the normal window if the shifted fits fail (e.g. mean ~ 0
    with non-trivial skew). The returned window is widened, if necessary, to
    contain ``m``.
    """
    sd = abs(float(sd))
    if not np.isfinite(sd):
        raise ValueError('Infinite/undefined sd passed to estimate_agg_window')

    def _to_prob(q):
        return float(np.where(q > 1, 1 - 10.0 ** -q, q))

    p = _to_prob(p)
    p_lo = p if p_lo is None else _to_prob(p_lo)
    p_hi = p if p_hi is None else _to_prob(p_hi)
    tail_lo = 1.0 - p_lo
    tail_hi = 1.0 - p_hi
    z_lo = ss.norm.isf(tail_lo)
    z_hi = ss.norm.isf(tail_hi)
    if sd == 0:
        return float(m), float(m), 0.0

    def _normal_window():
        return float(m - z_lo * sd), float(m + z_hi * sd), float((z_lo + z_hi) * sd)

    skew_tol = 1e-3
    if abs(skew) <= skew_tol or not np.isfinite(skew):
        return _normal_window()

    # Reflect for negative skew so the fits always see a right tail. Under the
    # reflection the final lower edge comes from the reflected fit's *high*
    # quantile and vice versa, so the per-edge tails swap with ``s``: the
    # reflected ppf (low) carries the tail that ends up on the final low edge.
    s = 1.0 if skew > 0 else -1.0
    fit_tail_lo = tail_lo if s == 1.0 else tail_hi
    fit_tail_hi = tail_hi if s == 1.0 else tail_lo
    mm = s * m
    sk = s * skew
    cvv = sd / mm if mm != 0 else np.inf
    los, his = [], []
    if np.isfinite(cvv):
        try:
            shift, mu, sigma = sln_fit(mm, cvv, sk)
            fz = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
            los.append(float(fz.ppf(fit_tail_lo)))
            his.append(float(fz.isf(fit_tail_hi)))
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            shift, alpha, theta = sgamma_fit(mm, cvv, sk)
            fz = ss.gamma(alpha, scale=theta, loc=shift)
            los.append(float(fz.ppf(fit_tail_lo)))
            his.append(float(fz.isf(fit_tail_hi)))
        except Exception:  # pragma: no cover - defensive
            pass
    if not his:
        return _normal_window()
    lo_r, hi_r = min(los), max(his)
    if s == 1.0:
        x_lo, x_hi = lo_r, hi_r
    else:  # undo reflection: window of A is the mirror of the window of -A
        x_lo, x_hi = -hi_r, -lo_r
    # ensure the window contains the mean
    x_lo = min(x_lo, m)
    x_hi = max(x_hi, m)
    return float(x_lo), float(x_hi), float(x_hi - x_lo)


# ---------------------------------------------------------------------------
# Bucket-grid narrative ([bs-reporting]) -- the short / verbose explanation of
# the grid choice. Plain by default; ``color=True`` emphasises a far-tail clip
# (lost mass) in bold red, mirroring the tail narrative's ANSI option. The
# ``bs_description`` / ``bs_explanation`` Aggregate properties delegate here.
# ---------------------------------------------------------------------------

_METHOD_BLURB = {
    'moment': 'the 3-moment method-of-moments window',
    'exact_discrete': 'the exact discrete (integer-lattice) support',
    'bounded_small': 'the bounded-severity support window',
    'windowed': 'a two-sided window far from 0 (benign FFT wrap)',
    'sbj': 'the single-big-jump extent',
}


def _bs_grid_top(used) -> float:
    """Realized grid top ``x_min + 2**log2 * bs`` from the ``used`` row."""
    return float(used['x_min']) + (1 << int(used['log2'])) * float(used['bs'])


def bs_describe(agg, *, color: bool = False) -> str:
    """One-line summary of an aggregate's chosen bucket grid (``[bs-reporting]``).

    Parameters
    ----------
    agg : Aggregate
        A *sized* aggregate (``update`` has run, so ``_bs_window_df`` exists).
    color : bool
        Emit ANSI colour (a far-tail clip is bold red). Default plain.

    Returns
    -------
    str
        ``'<method> grid: bs=…, log2=…, x_min=… (x_max=…)'`` plus a clip note when
        the far tail is truncated.
    """
    df = agg._bs_window_df
    if df is None:
        return 'bucket grid not sized yet (call update())'
    sel = df.index[df['selected'].astype(bool)]
    method = sel[0] if len(sel) else 'moment'
    used = df.loc['used']
    top = _bs_grid_top(used)
    text = (f'{method} grid: bs={_fmt_bs(used["bs"])}, log2={int(used["log2"])}, '
            f'window {fmt_window(float(used["x_min"]), top)}')
    clip = agg._bs_clip
    if clip is not None:
        cm = clip.get('clipped_mass', float('nan'))
        cm_txt = f'~{cm:.3g}' if np.isfinite(cm) else 'a sliver'
        msg = f'; clips {cm_txt} of the tail (raise log2 to {int(clip["need_log2"])})'
        if color:
            msg = f'{_tail._ANSI_THICK}{msg}{_tail._ANSI_RESET}'
        text += msg
    return text


def bs_explain(agg, *, color: bool = False) -> str:
    """Verbose prose explaining an aggregate's bucket-grid choice (``[bs-reporting]``).

    The aggregate mirror of :attr:`Portfolio.bs_explanation`, simpler (no unit
    rows): the aggregate tail one-liner and log2; the winning method and its
    **window width** against the candidate methods that applied; any natural
    support bounds and the concentration; the realised ``x_min`` / ``x_max``;
    and a closing "increase log2" suggestion when a far-tail clip occurred.

    Parameters
    ----------
    agg : Aggregate
        A *sized* aggregate.
    color : bool
        Emit ANSI colour (a far-tail clip is bold red). Default plain.

    Returns
    -------
    str

    Notes
    -----
    "Window width" (the realised ``W = x_max - x_min`` and the per-method
    candidates) replaces the older "span" wording. The raw-to-dyadic ``bs``
    sentence the portfolio narrative carries is omitted here: the per-method
    aggregate sizer rounds each candidate independently, so there is no single
    pre-round ``bs`` to report (``agg._bs_raw`` stays ``None``).
    """
    df = agg._bs_window_df
    if df is None:
        return 'Bucket grid not sized yet (call update()).'
    sel = df.index[df['selected'].astype(bool)]
    method = sel[0] if len(sel) else 'moment'
    used = df.loc['used']
    log2 = int(used['log2'])
    bs = float(used['bs'])
    x_min = float(used['x_min'])
    x_max = _bs_grid_top(used)
    W = float(df.loc[method, 'W']) if method in df.index and 'W' in df.columns \
        else x_max - x_min
    agg_line = _tail.describe_row(agg._tail_rows()[-1], color=color)
    parts = [f'The aggregate is {agg_line}. Log2 is {log2}.']

    applied = [i for i in df.index
               if i not in ('used',) and bool(df.loc[i].get('applies'))]
    blurb = _METHOD_BLURB.get(method, method)
    if applied:
        cand_txt = ', '.join(
            f'{i} {fmt_amount(df.loc[i, "W"])}' for i in applied)
        parts.append(
            f'Of the methods that applied ({cand_txt}), the {method} method won '
            f'(window width {fmt_amount(W)}) -- {blurb}.')
    else:
        parts.append(
            f'The {method} method won (window width {fmt_amount(W)}) '
            f'-- {blurb}.')

    raw = getattr(agg, '_bs_raw', None)
    if raw is not None:
        parts.append(
            f'The window produces a raw bs {_fmt_bs(raw)} which dyadically '
            f'rounds to {_fmt_bs(bs)} producing a final window width of '
            f'{fmt_amount(W)}.')

    snap = getattr(agg, '_bs_snap', None)
    if snap is not None:
        parts.append(
            f'The severity comes from {snap["ref"]}, whose atoms sit on a '
            f'{_fmt_bs(snap["d"])} lattice, so the chosen bs '
            f'{_fmt_bs(snap["bs_before"])} was snapped to '
            f'{_fmt_bs(snap["bs_after"])} (= {_fmt_bs(snap["d"])} x '
            f'2^{snap["m"]}) to keep the two grids commensurable, which puts '
            'every inner atom on a bucket edge instead of between two.')

    # natural support bounds + concentration from the aggregate tail row
    try:
        trow = agg.tail_behavior_df.loc['aggregate']
    except Exception:  # pragma: no cover - defensive
        trow = None
    if trow is not None:
        if trow['left_tail'] == 'bounded' and np.isfinite(float(trow['min'])):
            parts.append('It has a natural lower support bound of '
                         f'{fmt_amount(trow["min"])}.')
        if trow['right_tail'] == 'bounded' and np.isfinite(float(trow['max'])):
            parts.append('It has a natural upper support bound of '
                         f'{fmt_amount(trow["max"])}.')
        cv = trow.get('cv')
        if bool(trow.get('concentrated')) and cv is not None and np.isfinite(float(cv)):
            parts.append(f'The distribution is concentrated with a CV of {float(cv):g}.')

    parts.append(f'The recommended x_min is {fmt_amount(x_min)} and the '
                 f'x_max that follows is {fmt_amount(x_max)}, so the window '
                 f'is {fmt_window(x_min, x_max)}.')

    clip = agg._bs_clip
    if clip is not None:
        cm = clip.get('clipped_mass', float('nan'))
        cm_txt = f'~{cm:.3g}' if np.isfinite(cm) else 'a sliver'
        msg = (f'The heavy right tail reaches {fmt_amount(clip["reach"])}, past '
               f'the grid x_max {fmt_amount(x_max)} ({cm_txt} of the mass '
               f'clipped, a reported '
               f'deficit not normalized); the analysis suggests increasing log2 '
               f'to {int(clip["need_log2"])}.')
        if color:
            msg = f'{_tail._ANSI_THICK}{msg}{_tail._ANSI_RESET}'
        parts.append(msg)
    return ' '.join(parts)


# ====================================================================
# Aggregate bucket/window orchestrator (moved from Aggregate._bs_window,
# Phase B1). Sits beside its estimate_agg_window/_estimate_agg_percentile
# leaves; the Aggregate method delegates here.
# ====================================================================

def _integer_ratio(a, b, rel_tol=1e-9):
    """Whether ``a / b`` is a positive integer, to a stated relative tolerance."""
    if not (b and np.isfinite(a) and np.isfinite(b)):
        return False
    r = a / b
    return r >= 1.0 and math.isclose(r, round(r), rel_tol=rel_tol)


def snap_bs_to_reference(agg, b0, pinned):
    """Keep the grid commensurable with a reference severity's own lattice.

    The final query at the end of bucket selection, not an optimization woven
    through the sizer: the winning candidate is already chosen, and this asks
    one question about it. A severity built from a ``sev agg.NAME`` reference
    carries the referenced object's bucket size ``d`` (``reference_bs``, set
    exactly on the resolution path, never inferred from ``np.diff`` of the
    atoms). If the outer grid and ``d`` are incommensurable, every inner atom
    lands between two outer buckets and the mean-preserving scatter spreads
    each of them, for no reason: ``d`` was a free choice and a nearby
    commensurable value costs nothing.

    Parameters
    ----------
    agg : Aggregate
        The aggregate being sized; read for ``sevs`` only.
    b0 : float
        The selected bucket size.
    pinned : bool
        Whether ``b0`` came from an explicit ``bs`` (a call argument or a
        ``hints{bs=…}``) rather than from the estimator.

    Returns
    -------
    (float or None, str)
        The bucket size to use instead of ``b0``, or ``None`` to leave it
        alone, and a note fragment for the winning row.

    Notes
    -----
    The decision table, in order:

    * **no component exposes a** ``d``: unchanged, and this is the
      overwhelmingly common path, so it costs one generator expression;
    * ``d / b0`` **an integer** (the outer grid is finer, so inner atoms land
      exactly on grid points): no change. This also covers an exact discrete
      winner at ``bs = 1`` over an integer-lattice reference;
    * ``b0 / d`` **an integer**: already commensurable, no change. Deliberately
      not forced to a power of two, since the estimator or the user landed on
      an exact multiple and there is nothing to fix;
    * **auto-sized otherwise**: snap to ``d * 2**m`` with
      ``m = max(0, ceil(log2(b0 / d)))``, never below ``d`` (the severity *is*
      the reference's output, so an estimate finer than ``d`` snaps to ``d``
      itself). ``d * 2**m`` is exact in binary floating point, so the snapped
      grid carries no representation fuzz. The caller re-runs ``_size`` with
      it, so origin, span and the log2 need all come from the one kernel;
    * **pinned otherwise**: left alone (no back doors) with a warning naming
      both values and the nearest commensurable bucket on each side.

    Origins are multiples of the outer ``bs``, hence of ``d``, so inner atoms
    stay on the ``d`` sublattice; a signed reference is covered by the existing
    origin flooring in ``_size``.

    The v1 grammar admits at most one reference per aggregate, so at most one
    ``d`` reaches a sizing. More than one is future-proofing for portfolio unit
    sizing: the largest wins, and it warns.
    """
    ds = sorted({float(s.reference_bs)
                 for s in (agg.sevs if agg.sevs is not None else [])
                 if getattr(s, 'reference_bs', None)})
    if not ds:
        return None, ''
    d = ds[-1]
    ref = next((getattr(s, 'reference_id', '') or '?'
                for s in agg.sevs
                if getattr(s, 'reference_bs', None) == d), '?')
    if len(ds) > 1:
        warn_once(
            f'{agg.name}: severity references sit on {len(ds)} different '
            f'lattices ({", ".join(_fmt_bs(x) for x in ds)}); snapping to the '
            f'coarsest, {_fmt_bs(d)}. The finer ones stay off the grid.',
            UserWarning, key=f'sev-ref-lattices-{agg.name}', stacklevel=3)
    if _integer_ratio(d, b0) or _integer_ratio(b0, d):
        return None, ''
    if pinned:
        lo, hi = d * np.floor(b0 / d), d * np.ceil(b0 / d)
        warn_once(
            f'{agg.name}: the pinned bs {_fmt_bs(b0)} is incommensurable with '
            f'{ref}, whose atoms sit on a {_fmt_bs(d)} lattice, so every one '
            f'of them straddles two buckets. The nearest commensurable buckets '
            f'are {_fmt_bs(lo)} and {_fmt_bs(hi)}. The pin is honored as '
            'written.',
            UserWarning, key=f'sev-ref-pinned-{agg.name}', stacklevel=3)
        return None, ''
    m = max(0, int(np.ceil(np.log2(b0 / d))))
    snapped = d * 2.0 ** m
    agg._bs_snap = dict(ref=ref, d=float(d), bs_before=float(b0),
                        bs_after=float(snapped), m=m)
    return snapped, (f'; bs {_fmt_bs(b0)} snapped to {_fmt_bs(snapped)} '
                     f'(= {_fmt_bs(d)} x 2^{m}) to match {ref}')


def bs_window(agg, log2, bs_in, x_min_in, bucket_sizing_p,
               window_convention=None):
    """Decide ``(bs, log2, x_min)`` for ``update`` and build ``_bs_window_df``.

    Runs up to three sizing methods and records each in the expert-
    inspectable ``agg._bs_window_df``, then selects per the documented
    priority. ``log2`` is a **cap** (the input value, default 16): the exact
    discrete method may use fewer buckets but never more; the other methods
    fill the cap. The 0-origin convention is preserved -- ``x_min = 0``
    whenever the support is non-negative; a negative origin is used only for
    a genuinely signed aggregate.

    Parameters
    ----------
    log2 : int
        Bucket-count cap, ``2**log2`` buckets.
    bs_in : float
        ``0`` to estimate the bucket; ``>0`` to force it (honoured, D4).
    x_min_in : float or None
        ``None`` lets the selected method choose the origin; a number forces
        it (snapped to ``bs``, D4).
    bucket_sizing_p : float
        Tail probability for the moment / bounded windows.
    window_convention : {'loss', 'payoff'}, optional
        Override the sign convention that orients the windowed placement
        (per-edge coverage + padding skew). ``None`` (default) derives it
        from ``agg.value_type`` (``_is_loss_value``). See
        ``dev/plan-bucket-window-2.md`` §1A (Q9).

    Returns
    -------
    (bs, log2, x_min) : tuple
        Final grid parameters; ``agg._bs_window_df`` is also populated.

    Methods (rows of ``_bs_window_df``)
    -----------------------------------
    - ``moment`` -- always; the legacy 3-moment sizing for non-negative
      aggregates (the former ``recommend_bucket`` algorithm), two-sided
      ``estimate_agg_window`` for signed ones.
    - ``exact_discrete`` -- ``dfreq``/``fixed`` x ``dsev`` on an integer
      lattice: exact finite support, ``bs=1``, minimal ``log2`` (<= cap).
    - ``bounded_small`` -- bounded severity: ``[0, N_hi·s_max]`` from a high
      frequency quantile; selected only when tighter than ``moment``.
    - ``windowed`` -- non-signed high-mean / thin-spread aggregate
      (``actual_cv < 1/z``): a convention-skewed two-sided window far from 0,
      computed via benign FFT wrap. Auto-origin only; selected when the
      severity fits the windowed extent and it is **no coarser** than the
      0-based pick (it then reclaims the empty space below the band and
      balances the power-of-2 slack). Gives the windowed grid a non-zero
      ``x_min`` (so ``q`` / ``F`` / plots are defined on the window, not
      from 0); pass ``x_min=0`` to force the legacy grid.

    Selection: ``exact_discrete`` > ``bounded_small`` (if tighter) >
    ``moment``; ``windowed`` then overrides when it applies and is no
    coarser. The selected windowed origin is then balance-padded
    (``window_pad_skew``) so the band sits sensibly in the grid (R2/R5).
    """
    N0 = 1 << log2
    agg._bs_clip = None    # cleared each sizing; set only if the tail clips
    agg._bs_snap = None    # ditto; set only if a reference lattice moved bs
    m = agg.actual_m
    try:
        ex2 = float(agg.stats_df['mixed'][('agg', 'ex2')])
        sd = float(np.sqrt(max(ex2 - m * m, 0.0)))
    except Exception:  # pragma: no cover - defensive
        sd = agg.actual_sd
    skew = agg.actual_skew
    # Convolution-grid signedness: the loss FFT is sized on the two-sided grid
    # only when the severity itself is signed (``ssev`` / negative-``dsev``).
    signed = agg._signed_severity()
    # Sign convention orienting the windowed placement: loss -> protect the
    # right (priced) tail and trim/balance toward it; payoff mirrors. From
    # ``value_type`` unless explicitly overridden (Q9; branch on the boolean
    # role, never the label string -- house rule).
    if window_convention is None:
        is_loss = bool(agg._is_loss_value)
    else:
        is_loss = value_type_role(window_convention)
    # Window coverage: WINDOW_NINES nines (default 12) -- far tighter than
    # the legacy bucket p, so the window captures essentially all the mass.
    # Per-edge coverage for the windowed two-sided placement: deep on the
    # protected edge (anti-clip), shallow on the cheap edge (anti-waste).
    p = 1.0 - 10.0 ** -WINDOW_NINES
    p_protect = 1.0 - 10.0 ** -WINDOW_NINES
    p_trim = 1.0 - 10.0 ** -WINDOW_NINES_TRIM
    p_lo_w, p_hi_w = (p_trim, p_protect) if is_loss else (p_protect, p_trim)
    lattice = agg._severity_lattice()

    def _size(x_lo, x_hi, lattice_bs, force_origin=False, grow_cap=None):
        """Origin ``x0`` and grid ``(bs, log2)`` for a window ``[x_lo, x_hi]``.

        Returns ``(x0, bs, log2)`` only -- the *window* edges stay the
        method's own ``[x_lo, x_hi]``; the realized grid extent
        (``x0 + 2**log2 * bs``, which power-of-2 padding makes wider than the
        window) is reported separately in the ``used`` row.

        bs: forced ``bs_in`` if given; else the integer lattice step (so an
        integer-valued aggregate uses ``bs=1`` not a fine fraction); else a
        resolution ``round_bucket(W / 2**cap)``. ``log2`` is shrunk to just
        cover the window (<= cap) when bs is free, keeping the realized grid
        sensible; if the user pinned bs they control the grid so the cap
        ``log2`` is honoured; if the window needs more than the cap, bs is
        coarsened to fit.

        ``force_origin`` makes the origin follow ``x_lo`` (snapped down to a
        multiple of ``bs``) even for a non-signed aggregate -- used by the
        ``windowed`` method, whose mass band sits far from 0 and is computed
        via the benign FFT wrap in ``_fft_aggregate``. The default
        (``signed`` only) keeps the 0-based convention for every legacy
        method.

        ``grow_cap`` (windowed only) is an *absolute* log2 ceiling above the
        requested cap: when an integer lattice is present and the band needs
        more than the cap to hold ``bs = lattice_bs``, grow ``log2`` up to
        ``grow_cap`` rather than coarsening below the lattice (which would
        mis-place the atoms). Only coarsen if even ``grow_cap`` is too small.
        """
        use_origin = signed or force_origin
        W = max(x_hi - x_lo, 0.0)
        if bs_in > 0:
            bs = float(bs_in)
        elif lattice_bs is not None:
            bs = float(lattice_bs)
        else:
            bs = round_bucket(W / N0) if W > 0 else 1.0
        x0 = float(np.floor(x_lo / bs) * bs) if use_origin else 0.0
        span = x_hi - x0
        need = int(np.ceil(np.log2(max(span / bs + 1.0, 1.0)))) if span > 0 else 0
        if bs_in > 0:
            l2 = log2
        elif need <= log2:
            # at least 1 (>= 2 buckets) so a degenerate / point-mass window
            # never collapses to a single bucket.
            l2 = min(log2, max(need, 1))
        elif (grow_cap is not None and lattice_bs is not None
              and need <= grow_cap):
            # windowed lattice case: keep the exact lattice bs and grow log2
            # past the cap (the band is narrow, so a small bounded bump) so
            # an integer-atom severity is not coarsened off its lattice.
            l2 = need
        else:
            bs = round_bucket(span / N0)
            x0 = float(np.floor(x_lo / bs) * bs) if use_origin else 0.0
            l2 = log2
        return x0, float(bs), int(l2)

    def _row(x_lo, x_hi, lattice_bs, coverage, note, force_origin=False,
             grow_cap=None):
        # ``x_max`` is the method's own computed window top (e.g. the exact
        # support max), NOT the padded grid extent -- the ``used`` row shows
        # the realized grid.
        x0, bs_, l2_ = _size(x_lo, x_hi, lattice_bs, force_origin, grow_cap)
        return dict(applies=True, x_min=float(x0), x_max=float(x_hi),
                    W=float(x_hi - x0), bs=bs_, log2=l2_,
                    coverage=coverage, note=note)

    rows = {}

    # ---- moment (always) --------------------------------------------
    if bs_in > 0 and not (signed and np.isfinite(sd)):
        # bs is pinned -> the grid is the user's; the moment "window" is
        # just the realized extent. Avoids estimating a window (and the
        # infinite-variance raise) when we don't need one.
        x_lo, x_hi = 0.0, float(N0 * bs_in)
    elif signed and np.isfinite(sd):
        x_lo, x_hi, _W = estimate_agg_window(m, sd, skew, p)
    else:
        x_lo = 0.0
        try:
            x_hi = float(_estimate_agg_percentile(m, agg.actual_cv, skew, p))
        except ValueError as e:
            # No finite variance (power-law / infinite-variance severity,
            # e.g. pareto shape alpha <= 2) and no explicit ``bs``: there is
            # no finite tail quantile to size the grid to, so there is no
            # basis to guess ``bs``. Refuse to build rather than invent one
            # -- the user must pass an explicit ``bs`` (item 2).
            raise InfiniteVarianceError(
                f'{agg.name}: infinite-variance (power-law) aggregate '
                f'(no finite second moment) -- cannot estimate a bucket '
                f'size. Pass an explicit bs, e.g. build(..., bs=...).'
            ) from e
        if not np.isfinite(x_hi):
            # deterministic (sd ~ 0) or undefined skew (NaN): a few sd above
            # the mean (collapses to the mean for a point mass).
            x_hi = m + 8.0 * sd
    rows['moment'] = _row(x_lo, x_hi, lattice, f'1-1e-{WINDOW_NINES}',
                          '3-moment MoM window')

    # ---- exact_discrete ---------------------------------------------
    ed = agg._exact_discrete_window()
    if ed is not None:
        a_lo, a_hi, bs_lat, logp_lo, logp_hi = ed
        r = _row(a_lo, a_hi, bs_lat, 'exact', 'dfreq/fixed x dsev integer lattice')
        if not np.isclose(r['bs'], bs_lat):
            r['coverage'] = 'support exact, bs coarsened'
        # Reachability guard: the exact support is only trustworthy as a grid
        # extent when the aggregate can actually attain an extreme. Each corner
        # ``N*s`` needs every one of ``N`` claims on the same extreme atom
        # (``logp_lo`` / ``logp_hi``, log10 attainment probability; a corner at 0
        # is always reachable). For a large fixed/So count both corners underflow
        # -- the "exact" support is a gross overstatement whose ``bs`` coarsening
        # aliases the severity -- so mark the row inapplicable (recorded for
        # inspection, not selected -> falls through to bounded_small / moment).
        # An asymmetric book with one reachable corner keeps its exact support.
        reach = max(float(logp_lo), float(logp_hi))
        r['applies'] = bool(reach >= EXACT_DISCRETE_REACH_LOGP)
        r['note'] += f'; logp_lo={logp_lo:.4g}, logp_hi={logp_hi:.4g}'
        if not r['applies']:
            r['coverage'] = 'support unreachable (rejected)'
            logger.info(
                '%s: exact-discrete support [%.6g, %.6g] is unreachable '
                '(log10 P(corner): lo=%.4g, hi=%.4g < %g); using the moment '
                'window instead.', agg.name, a_lo, a_hi, logp_lo, logp_hi,
                EXACT_DISCRETE_REACH_LOGP)
        rows['exact_discrete'] = r

    # ---- bounded_small ----------------------------------------------
    bw = agg._bounded_severity_window(p)
    if bw is not None:
        a_lo, a_hi = bw
        rows['bounded_small'] = _row(a_lo, a_hi, lattice, f'freq 1-1e-{WINDOW_NINES}',
                                     'bounded severity x freq quantile')

    # ---- windowed (non-signed high-mean / thin relative spread) -----
    # A concentrated aggregate -- ``actual_cv = sd/m < 1/z`` with
    # ``z = norm.isf(1e-WINDOW_NINES)`` -- has its whole mass band sitting a
    # long way above 0. Compute it on the two-sided window
    # ``[m - z*sd, m + z*sd]`` (``estimate_agg_window``) far from 0 and let
    # the periodic FFT wrap: ``_fft_aggregate`` lays the severity at period
    # ``M*bs`` and relabels the finished aggregate by ``round(x_min/bs)``
    # (modular ``np.roll``, so a 15M-bucket roll and any period straddle are
    # handled automatically). The relabel carries no ``N*s`` shift, so it is
    # exact for random frequency. Eligibility is deliberately narrow:
    #   - auto origin only (``x_min_in is None``); an explicit ``x_min`` --
    #     including ``x_min=0`` to force the legacy grid back -- keeps the
    #     0-based methods;
    #   - non-signed, non-affine, finite positive sd;
    #   - no *occurrence* reinsurance: the occ-reins severity rebucketing and
    #     ``reins_density_df`` carry the severity on the *output* grid
    #     (``xs == xs_sev``), which windowing breaks (the output window sits
    #     far above the severity grid). Aggregate reinsurance is fine -- it
    #     operates on the aggregate, on the windowed ``xs``/``x_min``.
    #   - the book is **concentrated** -- the tail report's conservative
    #     ``concentrated`` flag (``actual_cv < CONCENTRATION_CV``, i.e. ~0.1),
    #     the single source of truth (item 5). This replaces the looser
    #     geometric ``w_lo > 0`` (~``cv < 0.21``) gate: a band that merely
    #     grazes 0 is no longer windowed, so a borderline book reverts to the
    #     0-based grid. The ``w_lo > 0`` geometry is still required below (the
    #     band must actually clear 0 to be placed), but is now a necessary
    #     condition under the stricter concentration gate, not the gate.
    # The edges use *per-edge* coverage (``p_lo_w, p_hi_w``), deep on the
    # protected tail and shallow on the cheap one, so the convention skews
    # the placement (Q1/Q2). Selection then takes it when the severity fits
    # the windowed extent and it is **no coarser** than the 0-based pick
    # (below) -- reclaiming the empty space below the band even when ``bs``
    # is unchanged; the only quiet fall back is a coarser bucket.
    # Tail report and single-big-jump reach, computed once here so the
    # windowed left-lift (item 3), the thickness gate (item 1) and the SBJ
    # floor below all share them.
    loss_left, loss_right = agg._loss_tail_classes()
    sbj = agg._single_big_jump_window(p)
    conc_flag, _conc_cv = _tail.concentration(m, sd)
    if (x_min_in is None and not signed
            and agg.occ_reins is None
            and np.isfinite(sd) and sd > 0
            and bool(conc_flag)):
        try:
            w_lo, w_hi, _Ww = estimate_agg_window(
                m, sd, skew, p, p_lo=p_lo_w, p_hi=p_hi_w)
        except ValueError:
            w_lo = -1.0  # no finite window (e.g. infinite variance)
        if w_lo > 0:
            # Thin-left-gated upper floor (item 3, the asymmetric window). A
            # thick right tail (subexponential severity) reaches past the
            # moment window; floor the windowed upper edge by the single-big-
            # jump reach so the grid -- and with it the severity discretisation
            # extent ``N*bs`` -- grows enough to (a) capture that tail and
            # (b) let a single heavy occurrence fit the window (the
            # severity-fit guard below then passes where the un-floored
            # window failed -> Regime B is reclaimed). Lifting ``x_min`` off 0
            # is only safe when the *left* tail is thin (no mass below
            # ``w_lo`` to clip); a non-signed aggregate is bounded-left, so
            # this holds, but gate on it explicitly for correctness and for a
            # future signed windowing.
            thin_left = not _tail.is_thick(loss_left)
            w_hi_eff = w_hi
            floored_up = False
            if (thin_left and _tail.is_thick(loss_right)
                    and sbj is not None and sbj[1] > w_hi):
                w_hi_eff = float(sbj[1])
                floored_up = True
            r = _row(w_lo, w_hi_eff, lattice,
                     f'lo 1-1e-{WINDOW_NINES_TRIM if is_loss else WINDOW_NINES}'
                     f' / hi 1-1e-{WINDOW_NINES if is_loss else WINDOW_NINES_TRIM}',
                     'two-sided window, benign FFT wrap',
                     force_origin=True, grow_cap=log2 + WINDOW_LOG2_GROWTH)
            # Severity-fit guard: the severity discretises on [0, N*bs]; a
            # single occurrence must fit the windowed extent or its mass
            # overflows (the fixed-1 / approximate trap). Record the row
            # either way (inspectable) but mark it inapplicable -> not
            # selected -> quiet fall back to the 0-based grid.
            extent = float((1 << int(r['log2'])) * r['bs'])
            sev_hi = agg._severity_high_estimate(p)
            r['applies'] = bool(np.isfinite(sev_hi) and sev_hi < extent)
            r['note'] += f'; sev_hi={sev_hi:.6g}, extent={extent:.6g}'
            if floored_up:
                r['note'] += '; upper floored by sbj'
            rows['windowed'] = r
            if not r['applies']:
                # Regime B (heavy severity): the mass band clears 0 but a
                # single severity overflows the window, so the origin cannot
                # move -- the book keeps the floor-anchored grid (empty space
                # below the band is the price of the 0-containing severity
                # invariant). Discoverability only -- expected, not defective.
                logger.info(
                    '%s: mass band clears 0 (w_lo=%.6g) but severity '
                    'overflows the window (sev_hi=%.6g >= extent=%.6g); '
                    'keeping the 0-based grid (heavy-severity, non-windowable).',
                    agg.name, w_lo, sev_hi, extent)

    # ---- selection (D1/D2) ------------------------------------------
    # bounded_small is selected a bit permissively -- it is a hard support
    # bound, so accept it even when modestly wider (1.5x) than the moment
    # window. With high coverage it is typically the tighter of the two.
    if 'exact_discrete' in rows and rows['exact_discrete']['applies']:
        selected = 'exact_discrete'
    elif ('bounded_small' in rows
          and rows['bounded_small']['W'] <= 1.5 * rows['moment']['W']):
        selected = 'bounded_small'
    else:
        selected = 'moment'
    # windowed overrides the 0-based pick when it is applicable (the
    # severity fits the windowed extent) and EITHER
    #   - it is **no coarser** than that pick (``bs <= sel bs``) -- a band
    #     that clears 0 wins on *placement* alone, reclaiming the empty
    #     ``[0, x_lo)`` region and balancing the slack even when ``bs`` is
    #     unchanged; OR
    #   - the 0-based pick **clips** the single-big-jump reach while the
    #     windowed grid (upper-floored by ``sbj``, item 3) **captures** it --
    #     a coarser windowed bulk ``bs`` is the price of not clipping the
    #     thick right tail (the asymmetric-window reclaim of Regime B).
    # Self-limiting: ``applies`` can only hold when the mass band clears 0
    # *and* a single severity fits the (possibly floored) window.
    win = rows.get('windowed')
    if win is not None and win['applies']:
        sel_top = (float(rows[selected]['x_min'])
                   + (1 << int(rows[selected]['log2'])) * float(rows[selected]['bs']))
        win_top = (float(win['x_min'])
                   + (1 << int(win['log2'])) * float(win['bs']))
        reach = float(sbj[1]) if sbj is not None else float(win['x_max'])
        sel_clips = sel_top < reach
        win_covers = win_top >= reach
        if (win['bs'] <= rows[selected]['bs']
                or (sel_clips and win_covers)):
            selected = 'windowed'

    # ---- single-big-jump extent floor (1A-fix) ----------------------
    # A subexponential / signed severity can carry a far tail the 3-moment
    # window misses (``P(S>x) ~ E[N]·P(X>x)``); the moment window then either
    # clips the priced right tail (heavy positive sev) or -- for a signed sev
    # whose positive skew hides a heavy reflected tail -- lets the severity
    # wrap the FFT buffer (aliasing). Floor the SELECTED window's extent by
    # one big claim on a typical bulk so the grid covers it; the resolution
    # (``bs``) still follows the bulk/window. Self-activating: a light /
    # thin / bounded / concentrated book has ``sbj`` inside the window (the
    # ``max``/``min`` are no-ops) -> byte-stable. Only ``moment`` and
    # ``windowed`` are floored -- ``exact_discrete`` and ``bounded_small``
    # carry hard support bounds the SBJ moment estimate must not widen.
    # (``sbj`` and ``loss_left``/``loss_right`` were computed above, before
    # the windowed block, which now also consumes them.)
    if sbj is not None:
        # Record the grid the single-big-jump extent *alone* implies (sized
        # like any other method row, via ``_size``), so the row is directly
        # comparable to ``moment`` / ``windowed`` and never reads NaN. For a
        # non-signed sev this is the 0-based ``[0, sbj_hi]`` grid; for a
        # signed sev it carries the negative origin.
        sx0, sbs, sl2 = _size(sbj[0], sbj[1], lattice)
        rows['sbj'] = dict(
            applies=True, x_min=float(sx0), x_max=float(sbj[1]),
            W=float(sbj[1] - sx0), bs=float(sbs), log2=int(sl2),
            coverage=f'E[N]-adj 1-1e-{WINDOW_NINES}',
            note='single big jump: ES - mu_X + q_X(p**)')
    else:
        rows['sbj'] = dict(
            applies=False, x_min=np.nan, x_max=np.nan, W=np.nan,
            bs=np.nan, log2=np.nan, coverage=f'E[N]-adj 1-1e-{WINDOW_NINES}',
            note='single big jump: n/a (no finite E[N] / variance)')
    # Thickness gate (item 1): the single-big-jump mechanism only governs a
    # *thick* (subexponential-or-heavier) tail -- the loss aggregate's right
    # tail for a positive sev, its (reflected) left tail for a signed one.
    # For a thin tail the MoM window already covers the reach, so the floor
    # is a no-op; gating on the tail report makes that explicit and cheaper
    # and stops the deep ``p**`` severity quantile firing where it should not.
    if sbj is not None and bs_in <= 0 and selected in ('moment', 'windowed'):
        sbj_lo, sbj_hi = sbj
        win_lo = float(rows[selected]['x_min'])
        win_hi = float(rows[selected]['x_max'])
        keep_bs = float(rows[selected]['bs'])
        sel_l2 = int(rows[selected]['log2'])

        def _apply_floor(x0, hi, bs, l2):
            floored = dict(rows[selected])
            floored.update(x_min=float(x0), x_max=float(hi),
                           W=float(hi - x0), bs=float(bs), log2=int(l2),
                           note=rows[selected]['note'] + '; sbj floor')
            rows[selected] = floored

        if signed and _tail.is_thick(loss_left) and sbj_lo < win_lo:
            # SIGNED -- correctness, non-negotiable. The severity discretises
            # on the same N-bucket grid; if its full negative reach does not
            # fit, the FFT *wraps* and corrupts the whole law (the LNS 47%
            # mass-loss / aliasing failure). So the grid MUST cover
            # [sbj_lo, sbj_hi] at any log2 -- keep the bulk ``bs`` if it fits
            # the (hard) log2 budget, else coarsen ``bs`` to fit. Coarsening
            # the bulk is the lesser evil vs. an aliased law.
            floor_lo = min(win_lo, sbj_lo)
            floor_hi = max(win_hi, sbj_hi)
            x0 = float(np.floor(floor_lo / keep_bs) * keep_bs)
            span = floor_hi - x0
            need = int(np.ceil(np.log2(max(span / keep_bs + 1.0, 1.0))))
            if need <= log2:
                _apply_floor(x0, floor_hi, keep_bs, max(need, sel_l2))
            else:
                bs_f = round_bucket(span / (1 << log2))
                x0_f = float(np.floor(floor_lo / bs_f) * bs_f)
                _apply_floor(x0_f, floor_hi, bs_f, log2)
        elif not signed and _tail.is_thick(loss_right) and sbj_hi > win_hi:
            # POSITIVE -- a refinement, NOT a correctness fix. A heavy
            # unlimited severity's MoM window under-reaches the true tail, so
            # extend the (non-negative) window up to the single big jump --
            # but only when it fits at the bulk ``bs`` within the requested
            # log2 budget (grow ``log2`` up to the cap, no further, no
            # ``bs`` coarsening). If it does not fit, keep the MoM window:
            # clipping a tiny far tail is the lesser evil vs. coarsening the
            # bulk to uselessness (e.g. a 5-claim, mean-50 book whose tail
            # reaches 47k). The DefectiveDistribution warning already tells
            # the user to raise log2 when the clipped mass is material.
            x0 = win_lo if selected == 'windowed' else 0.0
            span = sbj_hi - x0
            need = int(np.ceil(np.log2(max(span / keep_bs + 1.0, 1.0))))
            if need <= log2:
                _apply_floor(x0, sbj_hi, keep_bs, max(need, sel_l2))
            else:
                # Doesn't fit at the bulk ``bs`` within the requested log2;
                # keep the MoM window (clip the far tail) rather than coarsen
                # the bulk. This is a visible warning (item 6), not a silent
                # log: the user should know a heavy tail is clipped and how
                # to widen the grid. The clipped-mass estimate is also stashed
                # in ``agg._bs_clip`` for the validation report / bs report.
                grid_top = x0 + (1 << sel_l2) * keep_bs
                clipped = agg._clipped_mass_estimate(grid_top)
                agg._bs_clip = dict(
                    reach=float(sbj_hi), grid_top=float(grid_top),
                    log2=int(sel_l2), bs=float(keep_bs),
                    need_log2=int(need), clipped_mass=float(clipped))
                cm = (f'~{clipped:.3g} of the aggregate mass'
                      if np.isfinite(clipped) else 'a sliver')
                # ``_bs_clip`` above is recorded whatever the size, because
                # the bs report wants it. The WARNING is gated on the same
                # materiality floor as every other defective report: a
                # clipped mass of 1e-8 is a fact about the grid, not
                # something the reader can act on. A non-finite estimate is
                # warned anyway -- unknown is not the same as small, and the
                # reach test is then the only evidence there is.
                if not np.isfinite(clipped) \
                        or clipped > _validation.DEFICIT_MATERIALITY:
                    warn_once(
                        f'{agg.name}: heavy right tail reaches {sbj_hi:.6g} '
                        f'but the grid top is {grid_top:.6g} at '
                        f'log2={sel_l2}, bs={keep_bs:.6g}; {cm} is clipped. '
                        f'Raise log2 to ~{need} (keeping this bs) to '
                        f'capture it.',
                        DefectiveDistributionWarning,
                        key='defective-construction', stacklevel=3)

    # ---- commensurable grids (reference severities) ------------------
    # The final query of section 4.7 of dev/done/plan-agg-port-as-sev.md: the
    # winner is chosen and any floor applied, so ask once whether the grid sits
    # on the reference's lattice, and re-derive through ``_size`` if it does
    # not. The window is the winner's own current ``[x_min, x_max]``, which is
    # the floored one when the single-big-jump floor fired, so the snap cannot
    # undo it. Snapping only coarsens, so the re-derived ``log2`` need can only
    # fall, and ``_size``'s coarsen-to-fit fallback (which would leave the
    # lattice) is unreachable from here.
    snapped, snap_note = snap_bs_to_reference(
        agg, float(rows[selected]['bs']), bs_in > 0)
    if snapped is not None:
        _sel = rows[selected]
        _forig = (selected == 'windowed')
        _gcap = (log2 + WINDOW_LOG2_GROWTH) if _forig else None
        _x0, _bs, _l2 = _size(float(_sel['x_min']), float(_sel['x_max']),
                              snapped, _forig, _gcap)
        rows[selected] = {**_sel, 'x_min': float(_x0), 'bs': float(_bs),
                          'log2': int(_l2),
                          'W': float(float(_sel['x_max']) - _x0),
                          'note': _sel['note'] + snap_note}

    # ---- realized grid (the ``used`` row) ---------------------------
    sel_bs = float(rows[selected]['bs'])
    sel_l2 = int(rows[selected]['log2'])
    sel_x0 = float(rows[selected]['x_min'])
    if x_min_in is not None:           # explicit origin override (D4)
        sel_x0 = float(round(x_min_in / sel_bs) * sel_bs)
    # ---- tail-aware padding / slack (item 4) ------------------------
    # A windowed band is placed band-bottom by ``_size`` (all power-of-2
    # slack above it). Redistribute the slack so the band sits sensibly in
    # the grid, with the split driven by the **tail report**, not a fixed
    # convention skew. ``f`` is the fraction of slack below the band:
    #   - asymmetric tails (one side thick, one thin) -> ~3/4 of the slack
    #     goes to the *thick* side (the tail that needs room): ``f = 1/4``
    #     for a thick right tail, ``f = 3/4`` for a thick left tail. The
    #     loss/payoff convention does NOT enter here -- the tail shape does.
    #   - symmetric tails (both thick or both thin) -> centre the band, with
    #     the loss/payoff convention demoted to a tie-breaker
    #     (``f = 0.5 -/+ window_pad_skew``: a loss leaves more room on the
    #     priced right, a payoff mirrors).
    # Only the *windowed* row is rebalanced -- ordinary, exact, and bounded
    # rows keep their band-bottom origin (byte-stable); an explicit ``x_min``
    # also pins the origin. The shift is clamped so the grid still covers the
    # window top and never crosses the 0 floor, so the benign FFT wrap stays
    # valid (and is in fact safer -- margin both sides).
    if selected == 'windowed' and x_min_in is None:
        w_lo = float(rows['windowed']['x_min'])   # snapped band-bottom origin
        w_hi = float(rows['windowed']['x_max'])
        N = 1 << sel_l2
        slack = N * sel_bs - (w_hi - w_lo)
        if slack > 0:
            thick_l = _tail.is_thick(loss_left)
            thick_r = _tail.is_thick(loss_right)
            if thick_r and not thick_l:
                f = 1.0 - WINDOW_SLACK_THICK     # thick right: room above
            elif thick_l and not thick_r:
                f = WINDOW_SLACK_THICK           # thick left: room below
            else:                                # symmetric: convention tie-break
                f = 0.5 - WINDOW_PAD_SKEW if is_loss else 0.5 + WINDOW_PAD_SKEW
            target = w_lo - f * slack
            origin = float(np.floor(target / sel_bs) * sel_bs)
            # keep the band: origin in [x_hi - N*bs, w_lo], and >= 0 floor.
            lo_bound = max(0.0, float(np.ceil((w_hi - N * sel_bs) / sel_bs) * sel_bs))
            origin = min(max(origin, lo_bound), w_lo)
            sel_x0 = origin
    grid_x_max = sel_x0 + (1 << sel_l2) * sel_bs

    ret_x0 = sel_x0

    # ``from_dict(orient='index')``, NOT ``pd.DataFrame(rows).T``. The rows mix
    # bool, float, int and str, so the transposed form puts *methods* in the
    # columns, makes every column mixed, and lands the whole block as
    # ``object``; the transpose then carries that dtype across wholesale,
    # because pandas does not re-infer per column on a transpose. Every
    # published column read ``object``, which cost the served table its right
    # alignment and its raw values (a string column carries none, so an
    # interactive grid cannot sort or filter it). Index-oriented construction
    # infers per column instead, which is what the Portfolio sizer has always
    # done (``port_build_bs_window_df`` builds from a list of row dicts).
    df = pd.DataFrame.from_dict(rows, orient='index')
    # Named at construction, so the private frame and the published one agree.
    # Without it the index reaches a served table as a column headed
    # ``level_0``, which names nothing a reader recognizes.
    df.index.name = 'method'
    # the winning method is flagged; the ``used`` row is the realized grid.
    df['selected'] = df.index == selected
    # The ``used`` row converts the selected method's window into the actual
    # power-of-2 grid: x_max = x_min + 2**log2 * bs (so a 701-point support
    # padded to 1024 reads x_max = grid top, W = the full grid width).
    df.loc['used'] = dict(
        applies=True, x_min=sel_x0, x_max=float(grid_x_max),
        W=float(grid_x_max - sel_x0), bs=sel_bs, log2=sel_l2,
        coverage=rows[selected]['coverage'],
        note=f'realized grid ({selected})', selected=False)

    # ---- journey columns (bs-reporting item 1) ----------------------
    # Purely derived reporting -- no effect on the grid. ``log2_need`` is the
    # log2 a method's own window needs at its own ``bs`` (so a row whose
    # ``log2_need > log2`` was capped/coarsened to fit the budget);
    # ``clipped`` carries the estimated far-tail mass dropped, on the ``used``
    # row, when the positive sbj floor clipped (``agg._bs_clip``).
    def _need(r):
        w = float(r['x_max']) - float(r['x_min'])
        b = float(r['bs'])
        if not (np.isfinite(w) and np.isfinite(b) and b > 0 and w > 0):
            return np.nan
        return float(np.ceil(np.log2(w / b + 1.0)))
    df['log2_need'] = df.apply(_need, axis=1)
    df['clipped'] = np.nan
    if agg._bs_clip is not None:
        df.loc['used', 'clipped'] = float(agg._bs_clip.get('clipped_mass', np.nan))
    # The two log2 columns are exponents, so they read as integers. Both can be
    # genuinely absent (a non-applies ``sbj`` row records no grid; ``_need``
    # returns NaN on a degenerate window), hence the nullable ``Int64`` rather
    # than a plain ``int64``: without it inference would give ``int64`` on most
    # books and ``float64`` on the ones with an n/a row, and the published
    # dtype would depend on the book. A float log2 also renders as ``16.00``,
    # which is not what an exponent looks like.
    df['log2'] = df['log2'].astype('Int64')
    df['log2_need'] = df['log2_need'].astype('Int64')
    agg._bs_window_df = df

    return sel_bs, sel_l2, ret_x0


# ====================================================================
# Portfolio bucket/window sizers (moved from Portfolio, Phase B2). The
# Portfolio methods delegate here; they sit beside the Aggregate sizer so
# the two approaches share one module.
# ====================================================================

def port_bs_window_df(port) -> 'pd.DataFrame':
    """Curated, read-only view of the portfolio combine grid (``[bs-reporting]``).

    One row per unit (its selected window), then the four candidate combine
    rows (``mm`` Portfolio MM bulk / ``rms`` RMS-of-windows reference /
    ``sbj`` single-big-jump look-through / ``sum`` legacy linear bound), then
    the realised shared ``used`` grid, indexed by ``source``, culled to the
    user-facing columns (``x_min`` / ``x_max`` / ``W`` / ``bs`` / ``log2`` /
    ``log2_need`` / ``coverage`` / ``clipped`` / ``note``, parity with
    :attr:`Aggregate.bs_window_df`). Returns ``None`` before the grid is
    sized. See :attr:`bs_description` / :attr:`bs_explanation`.

    ``W`` and ``coverage`` joined the published frame at ``1.0.0a254``: the
    width and the coverage are what let a reader compare two candidate
    windows, which is the only thing this frame is for. The two exponent
    columns are nullable ``Int64`` since ``1.0.0a275``, parity with
    :attr:`Aggregate.bs_window_df`.
    """
    df = getattr(port, '_bs_window_df', None)
    if df is None:
        return None
    cols = ['x_min', 'x_max', 'W', 'bs', 'log2', 'log2_need', 'coverage',
            'clipped', 'note']
    return df.reindex(columns=cols).copy()


def port_single_big_jump_window(port, p_star):
    """Portfolio single-big-jump extent floor by look-through to the units.

    The subexponential tail of an independent sum is the *sum* of the unit
    tails, dominated by the heaviest unit:
    ``P(S_tot > x) ~ sum_k E[N_k]*P(X_k > x)``. So the single-big-jump
    scenario is one big claim in some unit ``k`` riding the *typical* bulk
    of everything else -- the portfolio mean ``actual_m`` with one typical
    claim (unit ``k``'s severity mean) replaced by one big one::

        sbj_hi_port = actual_m + max_k ( sbj_hi_k - ES_k )      # MAX, not sum

    where ``sbj_hi_k`` is unit ``k``'s own
    :meth:`Aggregate._single_big_jump_window` upper edge called with the
    **portfolio** ``p_star`` (each unit forms its own
    ``p**_k = 1 - (1 - p_star)/E[N_k]`` from *its* frequency),
    ``ES_k = a.actual_m`` is the unit's aggregate mean, and
    ``sbj_hi_k - ES_k = q_{X_k}(p**_k) - mu_{X_k}`` is that unit's "jump
    excess" over a typical claim. The lower edge mirrors over **signed**
    units only (a non-negative unit reaches no lower than its 0 floor); for
    an all-non-negative book ``sbj_lo = 0``.

    ``max_k`` is tight when one unit dominates the tail; when two or more
    comparably-heavy units drive it, the exact extent is the pooled
    root-find ``sum_k E[N_k](1 - F_{X_k}(x)) = 1 - p_star`` (a documented
    refinement, not yet wired -- the ``max`` is a safe lower bound on the
    true reach for the dominant-unit case and the doubling-padding absorbs
    the rest; see ``dev/plan-bucket-window-2.md`` Round 3).

    Parameters
    ----------
    p_star : float
        Portfolio aggregate coverage (``> 1`` read as a number of nines).

    Returns
    -------
    (sbj_lo, sbj_hi) : tuple of float, or None
        The portfolio single-big-jump window edges, or ``None`` when no
        unit yields a finite SBJ window (no finite ``E[N]`` / variance).
    """
    m = float(port.actual_m)
    if not np.isfinite(m):
        return None
    hi_excess, lo_excess = [], []
    for a in port.agg_list:
        sbj = a._single_big_jump_window(p_star)
        if sbj is None:
            continue
        es_k = float(a.actual_m)
        if not np.isfinite(es_k):
            continue
        hi_excess.append(float(sbj[1]) - es_k)
        if a._signed_severity():
            lo_excess.append(float(sbj[0]) - es_k)
    if not hi_excess and not lo_excess:
        return None
    sbj_hi = m + max(hi_excess) if hi_excess else m
    sbj_lo = m + min(lo_excess) if lo_excess else 0.0
    return float(sbj_lo), float(sbj_hi)


def port_best_window(port, log2=16, bs_in=0, bucket_sizing_p=BUCKET_SIZING_P):
    """Decide the portfolio combine grid by **Portfolio MM** + SBJ look-through.

    Mirrors the per-aggregate :meth:`Aggregate._bs_window` *bulk / extent*
    split at the portfolio level. The bulk is sized from the **exact total
    moments**, never by combining per-unit windows (neither by linear sum --
    which overstates by ``sqrt(k)`` for ``k`` iid units, ignoring
    diversification -- nor by root-sum-square):

    1. **Bulk window from Portfolio MM.** Feed the analytic compound total
       moments (``actual_m``, ``actual_sd = actual_m*actual_cv``, ``actual_skew``; cumulants
       add under independence) straight into the *same*
       :func:`estimate_agg_window` the single-aggregate sizer uses. This
       gives the two-sided ``[mm_lo, mm_hi]`` where the combined mass lives.
    2. **Resolution floor.** Per-unit widths enter *only* as the finest
       bucket ``min_k bs_k`` -- a unit's own lattice must survive ("don't
       lose sev ``bs``", now portfolio-wide), never the span.
    3. **One portfolio SBJ extent floor** (the look-through,
       :meth:`_single_big_jump_window`): ``sbj_hi_port = actual_m + max_k(
       sbj_hi_k - ES_k)`` -- the heaviest unit's one big claim on the
       combined bulk, **max** not sum, so the per-unit a59 extents are not
       double-counted. Self-activating: a thin / well-diversified total has
       ``sbj <= mm`` and the floor does not move the grid.

    The window is ``[x_lo, x_hi] = [min(mm_lo, sbj_lo), max(mm_hi, sbj_hi)]``.

    ``bs`` discipline -- **carry raw, round once**: the MM span term
    ``W_ext / N`` is rounded by ``round_bucket`` a *single* time at the top
    (no per-unit + combine double-round); the resolution floor stays the
    finest per-unit *lattice* value. ``bs = round_bucket(max(min_k bs_k,
    W_ext / N))`` (a pinned ``bs_in > 0`` is honoured verbatim, D4).

    Origin and ``log2``:

    - **non-signed, ordinary** (mass reaches 0, or not concentrated): the
      grid starts at ``x_min = 0`` (Plan A); ``log2`` is **shrunk** to just
      hold ``x_hi`` at ``bs`` (capped) -- a tiny discrete book no longer
      inflates to the cap.
    - **non-signed, concentrated and clear of 0** (the high-frequency /
      tiny-cv case, e.g. ``Poisson(100000)``): the total is **windowed**
      (Plan B), ``x_min = floor(x_lo / bs) * bs > 0``, routed by ``update``
      through the signed roll-combine path. The concentration gate
      (:func:`aggregate.tail.concentration`) is the same one
      :meth:`Aggregate._bs_window` uses, so a merely-grazing book reverts to
      the 0-based grid.
    - **signed** (P&L): the origin is the windowed low edge ``x_lo`` floored;
      ``update`` recomputes the realised origin from the units' post-snap
      ``x_min``. ``log2`` stays at the cap and the span is floored at the
      conservative ``max_k W_k / N`` so no per-unit marginal wraps.

    Parameters
    ----------
    log2 : int
        Bucket-count cap, ``2**log2`` buckets.
    bs_in : float
        ``0`` to estimate the bucket; ``>0`` to force it (honoured).
    bucket_sizing_p : float
        Tail probability for the per-unit moment / bounded windows.

    Returns
    -------
    (bs, log2, x_min) : tuple
        Shared grid parameters. ``x_min > 0`` for a non-signed total signals
        ``update`` to take the windowed roll path (Plan B).
        :attr:`_bs_window_df` (unit rows, the MM / RMS / SBJ / sum candidate
        rows, and a ``used`` row) is also populated.

    See Also
    --------
    _single_big_jump_window : the portfolio SBJ look-through.
    Aggregate._bs_window : the per-unit window estimator consumed here.
    """
    signed = port._signed()
    N_cap = 1 << log2
    p_star = 1.0 - 10.0 ** -WINDOW_NINES

    # ---- phase 1: per-unit natural windows (analytic, no FFT) ---------
    # Each unit sizes itself on its own (0-origin or signed) grid; we read
    # the *selected method* row, not the padded ``used`` row, for the width.
    # Quiet the pre-pass: a unit may emit a clip warning here that the real
    # combine re-issues once (deduped) -- silence the speculative pass.
    # ``warn_once_isolated`` restores the once-per-session budget on exit, so
    # a warning swallowed here does not spend the one emission the real
    # combine is entitled to.
    rows = []
    bs_ks, x_min_ks, W_ks = [], [], []
    with warn_once_isolated(), warnings.catch_warnings():
        warnings.simplefilter('ignore', DefectiveDistributionWarning)
        for a in port.agg_list:
            bs_k, l2_k, x_min_k = a._bs_window(log2, 0, None, bucket_sizing_p)
            wdf = a._bs_window_df
            sel = wdf[wdf['selected']].iloc[0] if 'selected' in wdf.columns \
                else wdf.loc['used']
            sx_min, sx_max = float(sel['x_min']), float(sel['x_max'])
            W_k = max(sx_max - sx_min, 0.0)
            bs_ks.append(float(bs_k))
            x_min_ks.append(sx_min)
            W_ks.append(W_k)
            rows.append(dict(unit=a.name, x_min=sx_min, x_max=sx_max, W=W_k,
                             bs=float(bs_k), log2=int(l2_k),
                             coverage=sel.get('coverage', ''),
                             note=str(sel.get('note', ''))))

    # ---- the bulk window: Portfolio MM (NOT a width-combine) ----------
    m = float(port.actual_m)
    sd = float(port.actual_sd)
    skew = float(port.actual_skew)
    try:
        mm_lo, mm_hi, _W_mm = estimate_agg_window(m, sd, skew, p_star)
    except (ValueError, FloatingPointError):
        # No finite window (e.g. infinite variance): fall back to the
        # conservative linear sum of per-unit widths.
        mm_lo, mm_hi = 0.0, float(sum(W_ks))

    # ---- the portfolio SBJ extent floor (the look-through) ------------
    sbj = port._single_big_jump_window(p_star)
    if sbj is not None:
        sbj_lo, sbj_hi = sbj
    else:
        sbj_lo, sbj_hi = mm_lo, mm_hi

    # ---- the two inspectable reference windows ------------------------
    # RMS(w_i): the normal-approx window combine (the k cancels, so it is
    # ``m +/- sqrt(sum w_i^2)``); runs above the MM window because the
    # per-unit windows bake in skew the sum de-skews away by CLT. Carried as
    # a standing candidate row -- the gap ``MM - RMS`` reads as the
    # skewness/diversification adjustment. ``sum w_i``: the legacy linear
    # span, a guaranteed-no-wrap upper bound on the support width.
    half_rms = float(np.sqrt(sum(w * w for w in W_ks))) / 2.0 if W_ks else 0.0
    rms_lo, rms_hi = m - half_rms, m + half_rms
    W_sum = float(sum(W_ks))

    # ---- the chosen extent --------------------------------------------
    # The SBJ floor only governs the *upper* extent for a non-negative book
    # (its ``sbj_lo`` is the 0 floor, which must not pull the windowed origin
    # down off the bulk); for a signed book it floors the lower edge too.
    x_hi = max(mm_hi, sbj_hi)
    x_lo_raw = min(mm_lo, sbj_lo) if signed else mm_lo
    # Concentration gate (same source as Aggregate._bs_window): only a
    # genuinely concentrated total that clears 0 is windowed (Plan B).
    conc_flag, _conc_cv = _tail.concentration(m, sd)
    window_nonsigned = (not signed and bs_in <= 0 and bool(conc_flag)
                        and x_lo_raw > 0)

    # ---- bs: carry raw, round once ------------------------------------
    resolution = min(bs_ks) if bs_ks else 1.0
    if signed:
        x_lo = x_lo_raw
    elif window_nonsigned:
        x_lo = x_lo_raw                          # windowed origin (Plan B)
    else:
        x_lo = 0.0                               # 0-based (Plan A)
    W_ext = max(x_hi - x_lo, 0.0)
    if bs_in > 0:
        bs = float(bs_in)
        port._bs_raw = None                      # pinned: no rounding to report
    else:
        span = W_ext / N_cap if N_cap else W_ext
        if signed:
            # Wrap safety: every per-unit marginal is driven on the shared
            # grid, so N*bs must hold the widest unit too. Floor the span at
            # ``max_k W_k / N`` (the MM span is usually wider, but guard the
            # one-dominant-unit case).
            span = max(span, (max(W_ks) if W_ks else 0.0) / N_cap)
        raw = float(max(resolution, span))
        port._bs_raw = raw                       # for the bs_explanation narrative
        bs = round_bucket(raw)

    # ---- origin and log2 ----------------------------------------------
    if signed:
        x_min = float(np.floor(x_lo / bs) * bs)
        log2_out = log2
    elif window_nonsigned:
        x_min = float(np.floor(x_lo / bs) * bs)
        if bs_in > 0:
            log2_out = log2
        else:
            need = (int(np.ceil(np.log2((x_hi - x_min) / bs + 1.0)))
                    if x_hi > x_min else 1)
            log2_out = min(log2, max(need, 1))
    else:
        x_min = 0.0
        if bs_in > 0:
            log2_out = log2                       # user pinned the grid
        else:
            need = int(np.ceil(np.log2(x_hi / bs + 1.0))) if x_hi > 0 else 1
            log2_out = min(log2, max(need, 1))

    cand = dict(mm=(mm_lo, mm_hi), rms=(rms_lo, rms_hi),
                sbj=(sbj_lo, sbj_hi), sum=(min(x_lo, 0.0), W_sum + min(x_lo, 0.0)))
    port._build_bs_window_df(rows, bs, log2_out, x_min, cand,
                             resolution, W_ext)
    return bs, log2_out, x_min


def port_build_bs_window_df(port, rows, bs, log2, x_min, cand, resolution, W_ext):
    """Build the unit-indexed bucket/window summary for the combine.

    Mirrors :attr:`Aggregate._bs_window_df`'s idiom but swaps *method*
    rows for *unit* rows -- the Portfolio convention of one row per unit
    plus a summary line (cf. ``stats_df`` / ``summary_df``, which carry
    per-unit columns and a ``total``). Each unit row is that unit's
    selected window; then four **candidate** combine rows -- ``mm`` (the
    Portfolio MM bulk, the live span), ``rms`` (the RMS-of-windows
    normal-approx reference), ``sbj`` (the single-big-jump look-through),
    and ``sum`` (the legacy linear-sum no-wrap bound) -- so the combine's
    journey is inspectable; finally the ``used`` row is the realised shared
    portfolio grid ``[x_min, x_min + 2**log2 * bs)``. The ``mm <= rms <=
    sum`` ordering and the ``mm - rms`` skewness/diversification gap can be
    read straight off the frame.

    Parameters
    ----------
    rows : list of dict
        Per-unit window rows (``unit``/``x_min``/``x_max``/``W``/``bs``/
        ``log2``/``coverage``/``note``).
    bs, log2 : float, int
        The realised shared grid bucket size and log2.
    x_min : float
        The realised portfolio-grid origin.
    cand : dict
        ``{'mm': (lo, hi), 'rms': (lo, hi), 'sbj': (lo, hi),
        'sum': (lo, hi)}`` -- the candidate window edges.
    resolution : float
        The resolution floor ``min_k bs_k``.
    W_ext : float
        The chosen extent width (for the ``used``-row note).
    """
    N = 1 << log2
    cols = ['x_min', 'x_max', 'W', 'bs', 'log2', 'coverage', 'note']
    df = pd.DataFrame(rows).set_index('unit')[cols] if rows \
        else pd.DataFrame(columns=cols)
    cov = f'1-1e-{WINDOW_NINES}'
    cand_note = {
        'mm': 'Portfolio MM bulk (3-moment fit on total) -- the live span',
        'rms': 'RMS-of-windows reference (normal approx); mm-rms = skew adj',
        'sbj': 'single big jump look-through: actual_m + max_k(sbj_hi_k - ES_k)',
        'sum': 'legacy linear sum of widths (guaranteed-no-wrap bound)',
    }
    for key in ('mm', 'rms', 'sbj', 'sum'):
        lo, hi = cand[key]
        df.loc[key] = dict(
            x_min=float(lo), x_max=float(hi), W=float(hi - lo),
            bs=float(bs), log2=int(log2), coverage=cov, note=cand_note[key])
    df.loc['used'] = dict(
        x_min=float(x_min), x_max=float(x_min + N * bs),
        W=float(N * bs), bs=float(bs), log2=int(log2), coverage=cov,
        note=f'realised portfolio grid (resolution={resolution:g}, '
             f'extent={W_ext:g})')

    # ---- journey columns (parity with Aggregate.bs_window_df) ---------
    # ``log2_need`` is the log2 a row's window needs at the shared ``bs``;
    # ``clipped`` (the realised far-tail deficit) is patched by ``update``.
    def _need(r):
        w, b = float(r['x_max']) - float(r['x_min']), float(r['bs'])
        if not (np.isfinite(w) and np.isfinite(b) and b > 0 and w > 0):
            return np.nan
        return float(np.ceil(np.log2(w / b + 1.0)))
    df['log2_need'] = df.apply(_need, axis=1)
    df['clipped'] = np.nan
    # Nullable ``Int64`` for the two exponent columns, parity with the
    # Aggregate frame: this one already infers cleanly (it is built from a
    # list of row dicts, never transposed), but ``log2_need`` is NaN on a
    # degenerate window, so it needs the nullable form all the same.
    df['log2'] = df['log2'].astype('Int64')
    df['log2_need'] = df['log2_need'].astype('Int64')
    # ``unit`` came from the per unit rows and then five rows that are not
    # units were appended under it: four combine candidates and the realized
    # grid. ``source`` is what every row actually answers, namely where this
    # window came from.
    df.index.name = 'source'
    port._bs_window_df = df


def port_bs_window(port, log2, bs_in, bucket_sizing_p=BUCKET_SIZING_P):
    """Decide ``(bs, log2, x_min)`` for the portfolio combine grid.

    Thin forwarder to :meth:`best_window` (the resolution + span combine).
    Retained because :meth:`update`'s signed branch calls it by name; the
    whole decision -- the phase-1 per-unit pre-pass, the
    ``max(resolution, span)`` bucket, the origin/``log2`` choice, and
    building :attr:`_bs_window_df` -- lives in :meth:`best_window`.

    Parameters
    ----------
    log2 : int
        Bucket-count cap, ``2**log2`` buckets.
    bs_in : float
        ``0`` to estimate the bucket; ``>0`` to force it (honoured).
    bucket_sizing_p : float
        Tail probability for the per-unit moment windows.

    Returns
    -------
    (bs, log2, x_min) : tuple
        Shared grid parameters; :attr:`_bs_window_df` is also populated.
    """
    return port.best_window(log2, bs_in, bucket_sizing_p)


# ====================================================================
# [Sharpen-Grid-Probe] -- probing the neighbourhood of the chosen grid.
#
# ``bs_window`` / ``port_best_window`` above CHOOSE a grid from the analytic
# moments, before any FFT runs. ``sharpen`` below AUDITS that choice after the
# fact: it re-updates the object on neighbouring (bs, log2) cells, scores each
# with ``_validation.validation_score``, and moves only when the win is large.
# See ``dev/done/plan-sharpen.md``.
# ====================================================================

#: Lowest ``log2`` the probe will drop to. Below this the grid is too coarse for
#: the comparison to mean anything.
SHARPEN_LOG2_FLOOR = 8


#: How close to the best score a cell must be to count as a tie when no cell
#: reaches the target. Ties are broken by the smallest ``log2``, so a 5% better
#: score never buys a doubled grid. Deliberately tight: a genuine improvement is
#: orders of magnitude here, not percent.
SHARPEN_FALLBACK_SLACK = 1.25


#: Default cap on how far the bucket line search walks: the largest factor by
#: which the probe will multiply or divide ``bs``. 16 allows four doublings each
#: way, so a grid wrong by two orders of magnitude is reachable in one call.
SHARPEN_BS_LIMIT = 16


def _sharpen_is_port(ob):
    """Whether ``ob`` is a Portfolio (duck test, keeps this module a leaf)."""
    return hasattr(ob, 'agg_list')


def _sharpen_aliasing(ob):
    """The aliasing fingerprint ``agg mean error / sev mean error``.

    Recorded alongside the score but deliberately **not** part of it: a ratio is
    the specific signature of ``bs`` too small (the FFT amplifies severity
    discretization error during convolution) and no sum of errors can express
    it. ``nan`` when the severity error is zero.
    """
    err = ob.stats_df['error'].abs()
    sev = float(err.get(('sev', 'mean'), np.nan))
    agg = float(err.get(('agg', 'mean'), np.nan))
    if not np.isfinite(sev) or sev <= 0:
        return np.nan
    return agg / sev


def _sharpen_locked(ob):
    """Harvest the update settings that must stay fixed across the probe.

    ``update_work`` stamps ``sev_calc`` / ``discretization_calc`` / ``normalize``
    / ``padding`` **from its arguments**, whose defaults are ``'discrete'`` /
    ``'survival'`` / ``True`` / ``1``. So a bare ``update(log2=..., bs=...)``
    restores the grid but silently *resets* those four. Harvesting them once up
    front and passing them to every cell is both the fidelity fix, the object
    comes back as it was, and the comparability fix: the cells differ in ``bs``
    and ``log2``, and in nothing else.

    Returns
    -------
    dict
        Keyword arguments accepted by the object's ``update``.
    """
    out = {}
    for name in ('sev_calc', 'discretization_calc', 'normalize', 'padding'):
        v = getattr(ob, name, None)
        # A Portfolio carries '' / None sentinels before its first update.
        if v is not None and v != '':
            out[name] = v
    if _sharpen_is_port(ob):
        out['remove_fuzz'] = bool(getattr(ob, '_remove_fuzz', False))
    else:
        for name in ('reins_bucket', 'dsev_bucket'):
            v = getattr(ob, name, None)
            if v:
                out[name] = v
    return out


def _sharpen_update(ob, bs, log2, locked, *, final):
    """Run one probe cell, or the final update, at ``(bs, log2)``.

    ``force_severity`` and ``add_exa`` are not stamped anywhere, so they are
    handled by rule rather than harvested: probe cells skip both, ``add_exa``
    being the dominant cost of a Portfolio update while contributing nothing to
    the moments, and the final update runs both, the safe superset that matches
    what ``build`` passes.
    """
    kwargs = dict(locked)
    kwargs['force_severity'] = bool(final)
    if _sharpen_is_port(ob):
        kwargs['add_exa'] = bool(final)
    ob.update(log2=log2, bs=bs, sharpen=False, **kwargs)


def _bs_steps(bs_limit):
    """Largest number of doublings the bucket line search may take each way."""
    k = int(round(np.log2(float(bs_limit))))
    if k < 1 or not np.isclose(2.0 ** k, float(bs_limit)):
        raise ValueError(f'bs_limit must be a power of two >= 2, got {bs_limit}')
    return k


def _sharpen_lattice(ob):
    """The integer lattice step of a wholly discrete severity, else ``None``.

    Delegates to :meth:`Aggregate._severity_lattice`, the gcd of the integer
    severity atoms, taking the gcd across units for a Portfolio. ``None`` as soon
    as any component is continuous or off the integer lattice.
    """
    if _sharpen_is_port(ob):
        lats = [a._severity_lattice() for a in getattr(ob, 'agg_list', [])]
        if not lats or any(v is None for v in lats):
            return None
        return float(np.gcd.reduce([int(round(v)) for v in lats]))
    fn = getattr(ob, '_severity_lattice', None)
    return None if fn is None else fn()


def _bucket_is_exact(ob, bs):
    """Whether ``bs`` discretizes a discrete severity exactly.

    ``(exact, lattice)``. Exact when every atom is a whole number of buckets,
    i.e. the lattice gcd is an integer multiple of ``bs``. A *finer* bucket is
    still exact but wastes grid; a coarser one that does not divide the lattice
    scatters the atoms off their own values.
    """
    lattice = _sharpen_lattice(ob)
    if lattice is None or not bs:
        return False, lattice
    ratio = lattice / float(bs)
    return bool(abs(ratio - round(ratio)) < 1e-9), lattice


def _note(row, text):
    """Append to a cell's ``note``, never overwrite it.

    A walk terminates on a cell that failed as readily as on one that merely
    stopped improving, and the exception text is the more valuable of the two
    facts, so the stop reason is added after it rather than in place of it.
    """
    row['note'] = f'{row["note"]}; {text}' if row['note'] else text


def _pick_cell(df):
    """Index of the best cell: sound first, then thrifty, then best score.

    Three preferences, outermost first.

    **Sound.** A cell carrying a genuine deficit is disqualified outright while
    any clean cell survives. This is a gate rather than a term in the score
    because a deficit is a different kind of failure from a moment error: the
    score's terms each divide by their own validation tolerance, a deficit has
    no such tolerance to divide by, and its cost is qualitative, two correct
    looking pricing routes disagreeing by exactly the lost mass. A moment score
    cannot see it either, far tail mass being negligible for the first three
    moments and decisive for tail pricing, so the two measures are orthogonal
    and the gate is the only thing that catches it. The threshold is
    ``VALIDATION_NOISE``, deliberately tighter than the
    ``DEFICIT_MATERIALITY`` floor at which ``DefectiveDistributionWarning``
    fires. The two answer different questions: choosing among candidate grids,
    losing no mass at all is free to insist on, so insist; interrupting the
    user is not free, so reserve it for a deficit large enough to move a
    price.

    **Thrifty.** Cells within :data:`SHARPEN_FALLBACK_SLACK` of the best score
    count as tied, and ties break on ``log2`` first, so a free memory saving is
    taken when the score is genuinely a wash, then on ``|d_bs|``, so a
    competitive centre wins over an equally good move and the grid is not
    churned for nothing.

    Returns ``None`` when every cell failed.
    """
    finite = df[np.isfinite(df['score'])]
    if not len(finite):
        return None
    clean = finite[~finite['defective'].astype(bool)]
    pool = clean if len(clean) else finite
    near = pool[pool['score'] <= pool['score'].min() * SHARPEN_FALLBACK_SLACK]
    near = near.assign(_move=near['d_bs'].abs())
    return near.sort_values(['log2', '_move', 'score']).index[0]


def sharpen(ob, bs=None, log2=None, *, log2_cap=20, bs_limit=SHARPEN_BS_LIMIT,
            power=2, good_enough=0.5, execute=True):
    """Probe the grid neighbourhood and move to a better ``(bs, log2)``.

    ``bs_window`` and ``port_best_window`` *choose* a grid from the analytic
    moments before any FFT runs. This *audits* that choice after the fact: it
    re-updates ``ob`` on neighbouring cells, scores each with
    :func:`~aggregate._validation.validation_score`, and takes the best grid that
    does not cost more than the one you already have. Results land on
    ``ob._sharpen_df``.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        The object to sharpen; updated in place. If it has never been updated it
        is updated once first, to establish the centre cell.
    bs, log2 : float, int, optional
        Centre of the probe. Default the object's current grid.
    log2_cap : int, default 20
        ``log2`` is never grown to more than this.
    bs_limit : float, default 16
        Largest factor by which the line search may multiply or divide ``bs``.
        Must be a power of two, so 16 permits four doublings each way.
    power : float, default 2
        Norm exponent passed to the score.
    good_enough : float, default 0.5
        Target score, in tolerance units, and the only judgment knob. It does two
        things. **Probe gate**: if the current grid already scores at or below
        it, nothing is run at all. **Growth trigger**: ``log2`` is grown only
        when no cell at the current size or smaller reaches it. ``good_enough=0``
        therefore means "probe everything, never grow", since a score is never
        negative; that is why there is no separate ``force`` argument.
    execute : bool, default True
        ``True`` leaves ``ob`` on the chosen cell. ``False`` restores the
        original grid, so the probe is pure diagnosis.

    Returns
    -------
    Aggregate or Portfolio
        ``ob``, so the call chains. The object is updated in place either way;
        the return is a convenience, not a copy.

    Notes
    -----
    **Shape.** A row per ``log2``, and within each row a **line search out from
    the current bucket**: ``bs`` is doubled until the score stops improving, then
    halved until it stops improving, capped at ``bs_limit`` each way. The rows
    are ragged, which is why the frame is tidy rather than a matrix. ``log2``
    and ``log2 - 1`` are searched up front; ``log2 + 1`` is searched **only if
    neither reaches the target**, because it costs twice as much per cell and
    under the selection rule below it is usually not consulted.

    The line search is well posed because the score is single-troughed in ``bs``
    at fixed ``log2``: with extent ``W = bs * 2**log2``, a larger bucket buys
    extent and loses resolution, so the two error families trade off and there is
    one turning point. Stopping at the first worse cell assumes exactly that. A
    walk that runs out of ``bs_limit`` while still improving says so in its
    ``note``, which is a different fact from turning, and means a re-run will
    keep going.

    **Selection.** Take the best score among cells that **do not grow**
    ``log2``. That is the principle: never make the caller pay more than they
    already are. Reducing ``log2`` is a bonus rather than a goal, so it is picked
    up by the tie rule in :func:`_pick_cell` rather than chased. ``log2`` grows
    only when nothing at the current size or smaller reaches ``good_enough`` and
    something at ``log2 + 1`` does. If nothing anywhere reaches the target, the
    best cell overall is taken, still thrift-ordered, so the descent starts and a
    re-run continues it.

    **Discrete severity.** When every severity atom is a whole number of buckets
    the discretization is already exact, and no bucket change can help: coarser
    scatters the atoms off their own values, finer only wastes grid. The bucket
    is pinned and only ``log2`` is probed, which is what a failing discrete
    object actually needs, its problem being extent.

    **Geometry.** Rows are constant *resolution* and anti-diagonals are constant
    *extent*, which is what makes the frame readable: cells sharing an extent
    differ only in resolution, so comparing them says whether the grid is extent
    limited, widen it, or resolution limited, refine it. That is also why the
    bucket steps are a strict factor of two rather than rungs of the
    ``round_bucket`` ladder, which would be factors of 1.25 and 1.6, too fine to
    move the error and out of step with ``log2`` plus or minus one.

    Every cell runs inside its own guard: one that raises is recorded as ``nan``
    with the exception text in its ``note``, and the sweep completes.
    """
    if log2_cap is None:
        log2_cap = 20
    k_max = _bs_steps(bs_limit)
    # Establish the centre. An object that has never been updated has no
    # empirical moments to score, so update it once at the auto-sized grid.
    if not getattr(ob, 'bs', 0):
        ob.update(log2=(log2 or 16), bs=(bs or 0), sharpen=False)
    bs0 = float(bs) if bs else float(ob.bs)
    log20 = int(log2) if log2 else int(ob.log2)
    # An explicit centre that is not where the object sits: move there first, so
    # the centre row describes a real state.
    if float(ob.bs) != bs0 or int(ob.log2) != log20:
        ob.update(log2=log20, bs=bs0, sharpen=False)
    locked = _sharpen_locked(ob)
    x_min0 = float(ob.xs[0]) if getattr(ob, 'xs', None) is not None else 0.0
    bs_exact, lattice = _bucket_is_exact(ob, bs0)

    rows = []

    def _record(d_bs, d_log2, seconds, note='', caught=()):
        terms = _validation.validation_score_terms(ob)
        xs = getattr(ob, 'xs', None)
        deficit = _validation.pmf_deficit(ob)
        row = {'d_bs': d_bs, 'd_log2': d_log2, 'bs': float(ob.bs),
               'log2': int(ob.log2),
               'extent': float(ob.bs) * (1 << int(ob.log2)),
               'x_min': float(xs[0]) if xs is not None else np.nan,
               'score': _validation.combine_score_terms(terms, power)}
        row.update(terms)
        row['aliasing'] = _sharpen_aliasing(ob)
        row['deficit'] = deficit
        # The gate, deliberately tighter than the warning: choosing among
        # candidate grids, losing no mass at all is free to insist on.
        row['defective'] = bool(np.isfinite(deficit)
                                and deficit > _validation.VALIDATION_NOISE)
        row['validation'] = ob.validation_description
        row['warnings'] = len(caught)
        row['warns'] = ','.join(sorted({type(w.message).__name__
                                        for w in caught}))
        row['seconds'] = seconds
        row['selected'] = False
        row['note'] = note
        rows.append(row)
        return row['score']

    def _failed(d_bs, d_log2, bs_c, log2_c, seconds, note):
        row = {'d_bs': d_bs, 'd_log2': d_log2, 'bs': bs_c, 'log2': log2_c,
               'extent': bs_c * (1 << log2_c), 'x_min': np.nan, 'score': np.nan}
        row.update({f'u_{k[0]}_{k[1]}': np.nan
                    for k, _ in _validation.SCORE_TERMS})
        row.update({'aliasing': np.nan, 'deficit': np.nan, 'defective': False,
                    'validation': '', 'warnings': 0, 'warns': '',
                    'seconds': seconds, 'selected': False, 'note': note})
        rows.append(row)
        return np.nan

    def _cell(k, j):
        """Evaluate one cell and return its score (``nan`` if it blew up)."""
        bs_c, log2_c = bs0 * (2.0 ** k), log20 + j
        t0 = time.perf_counter()
        try:
            # ``warn_once_isolated`` makes every cell warn: the probe COUNTS
            # warnings per cell (see ``_record``), so once-per-session
            # semantics would report the fault for the first cell only and
            # score every later one as clean. The budget is restored on exit.
            with warn_once_isolated(), \
                    warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _sharpen_update(ob, bs_c, log2_c, locked, final=False)
            return _record(k, j, time.perf_counter() - t0, caught=caught)
        except Exception as e:          # noqa: BLE001 a cell may legitimately blow up
            logger.info('sharpen: cell (bs x %s, log2 %s) failed: %s',
                        2.0 ** k, log2_c, e)
            return _failed(k, j, bs_c, log2_c, time.perf_counter() - t0,
                           f'{type(e).__name__}: {e}')

    def _search_row(j, anchor):
        """Line search along the bucket axis of one ``log2`` row.

        The anchor cell is already recorded; this walks out from it in both
        directions and stops each walk at the first cell that fails to improve,
        recording on the terminal cell WHY it stopped. Running out of
        ``bs_limit`` while still improving is a different fact from turning, and
        the caller wants to know which happened.
        """
        if bs_exact:
            return                      # the bucket is pinned, nothing to search
        for step in (1, -1):
            prev, k, last = anchor, step, None
            while abs(k) <= k_max:
                score = _cell(k, j)
                last = len(rows) - 1
                # `not (score < prev)` also stops on a nan, which is what a
                # failed cell should do: it is not evidence to keep walking.
                if not (score < prev):
                    _note(rows[last],
                          'stopped: this cell failed' if np.isnan(score)
                          else 'stopped: no better than the previous bucket')
                    break
                prev, k = score, k + step
            else:
                if last is not None:
                    _note(rows[last],
                          'stopped: bs_limit reached, still improving')

    # The centre is already computed: score it in place, for free.
    centre_score = _record(0, 0, 0.0, note='centre (current grid)')

    # Probe gate. A grid that already clears the target is left alone and nothing
    # is run. good_enough=0 never clears, which is how a probe is forced.
    # Soundness is part of the gate: a grid whose moments are fine but whose mass
    # runs off the end is not a grid to wave through, and no moment score will
    # ever say so, so the deficit has to be asked about separately.
    centre_defective = bool(rows[0]['defective'])
    if (np.isfinite(centre_score) and centre_score <= good_enough
            and not centre_defective):
        rows[0]['selected'] = True
        rows[0]['note'] = ('centre (current grid); at or under target and '
                           'sound, probe not run')
        ob._sharpen_df = _sharpen_frame(rows)
        ob._sharpen_state = {'ran': False, 'moved': False, 'power': power,
                             'good_enough': good_enough, 'execute': execute,
                             'centre_score': centre_score, 'bs0': bs0,
                             'log20': log20, 'bs_exact': bs_exact,
                             'lattice': lattice, 'grew': False,
                             'centre_defective': False, 'passed_over': 0}
        _program.pin_sharpen(ob)
        return ob

    # Thrifty rows first: the current grid size and, when there is room below,
    # the smaller one. Neither costs more than the caller is already paying.
    _search_row(0, centre_score)
    if log20 - 1 >= SHARPEN_LOG2_FLOOR:
        _search_row(-1, _cell(0, -1))

    df = pd.DataFrame(rows)
    thrifty = _pick_cell(df)
    grew = False
    if thrifty is None:
        pick = 0
        reason = 'every probed cell failed, so the grid was left alone'
    elif df.loc[thrifty, 'score'] <= good_enough:
        # The rule: best sound score among cells that do not grow log2. Reached
        # the target without growing, so the expensive row is never even run.
        pick = thrifty
        reason = ('best sound score at the current grid size or smaller, and it '
                  'meets the target, so log2 was not grown')
    elif log20 >= log2_cap:
        pick = thrifty
        reason = (f'nothing at the current grid size or smaller meets the '
                  f'target, and log2 is already at the cap of {log2_cap}')
    else:
        # Only now is growing worth pricing: nothing affordable reaches the
        # target, so pay for the log2 + 1 row and see whether it does.
        _search_row(1, _cell(0, 1))
        df = pd.DataFrame(rows)
        growth = df[df['d_log2'] == 1]
        ok_growth = growth[growth['score'] <= good_enough]
        ok_growth = ok_growth[~ok_growth['defective'].astype(bool)]
        if len(ok_growth):
            pick = _pick_cell(ok_growth)
            grew = True
            reason = ('nothing sound at the current grid size or smaller meets '
                      'the target, so log2 grew by one to reach it')
        else:
            # Nothing anywhere reaches the target. Take the best cell overall,
            # still thrift-ordered so growth must be meaningfully better to win.
            # Refusing to move here would leave a known-bad grid in place AND
            # freeze the descent, since a re-run would refuse identically.
            pick = _pick_cell(df)
            if pick is None:
                pick = 0
            grew = bool(df.loc[pick, 'd_log2'] == 1)
            reason = ('no cell anywhere meets the target, so the best available '
                      'step was taken; re-run to continue')
    df.loc[pick, 'selected'] = True
    moved = bool(df.loc[pick, 'd_bs'] or df.loc[pick, 'd_log2'])
    # Better-scoring cells the soundness gate threw out. Without this the frame
    # reads as though the picker ignored its own minimum.
    passed_over = int(((df['score'] < df.loc[pick, 'score'])
                       & df['defective'].astype(bool)).sum())

    # Final update: the winner when executing, otherwise back to the centre.
    # x_min is pinned on the restore so a signed grid returns on its own origin.
    if execute and moved:
        _sharpen_update(ob, float(df.loc[pick, 'bs']),
                        int(df.loc[pick, 'log2']), locked, final=True)
    else:
        kwargs = dict(locked)
        kwargs['force_severity'] = True
        if _sharpen_is_port(ob):
            kwargs['add_exa'] = True
        else:
            kwargs['x_min'] = x_min0
        ob.update(log2=log20, bs=bs0, sharpen=False, **kwargs)

    ob._sharpen_df = _sharpen_frame(df)
    ob._sharpen_state = {'ran': True, 'moved': moved and execute,
                         'power': power, 'good_enough': good_enough,
                         'execute': execute, 'centre_score': centre_score,
                         'bs0': bs0, 'log20': log20, 'reason': reason,
                         'bs_exact': bs_exact, 'lattice': lattice, 'grew': grew,
                         'centre_defective': centre_defective,
                         'passed_over': passed_over}
    # The probe's outcome goes onto the object's own program and trailer, so
    # `program` means the program that BUILDS this object rather than merely the
    # one that built it, and `hints` never half describes a grid it moved.
    _program.pin_sharpen(ob)
    return ob


def _sharpen_frame(rows):
    """Index the probe rows by their offsets, so the picture is one unstack.

    ``sharpen_df.score.unstack('d_log2')`` is the bucket-by-grid-size table; the
    line search makes the rows ragged, so absent cells come back ``nan``.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return df.set_index(['d_bs', 'd_log2']).sort_index()


def _fmt_bs(bs):
    """Format a bucket size, sub-unit values as the binary fractions they are.

    A ``bs`` below 1 is a negative power of two (``round_bucket`` keeps it binary
    exact, and the probe only halves and doubles), and ``1/8`` reads at a glance
    where ``0.125`` does not. Anything that is not a unit fraction falls back to
    plain formatting.
    """
    bs = float(bs)
    if bs >= 1 or bs <= 0:
        return f'{bs:.6g}'
    inv = 1.0 / bs
    if abs(inv - round(inv)) < 1e-9 and round(inv) > 1:
        return f'1/{round(inv)}'
    return f'{bs:.6g}'


def fmt_amount(x):
    """Format a loss-scale number for prose: grouped, never in exponent form.

    ``:g`` is right for a score and wrong for money. A grid whose top is
    12,182,881 prints as ``1.21829e+07``, which a reader has to decode before
    they can compare it to anything, and that is the number the grid
    narratives are mostly made of. Thousands are grouped and the fractional
    part is dropped once it cannot matter at this scale.

    Parameters
    ----------
    x : float

    Returns
    -------
    str
        ``'12,182,881'``, ``'123.457'``, ``'0.000125'``; ``'infinite'`` /
        ``'-infinite'`` for the infinities and ``'n/a'`` for ``nan``, so a
        missing bound reads as missing rather than as a number.
    """
    x = float(x)
    if np.isnan(x):
        return 'n/a'
    if not np.isfinite(x):
        return 'infinite' if x > 0 else '-infinite'
    if x == 0:
        return '0'
    if abs(x) >= 1000:
        return f'{x:,.0f}'
    return f'{x:,.6g}'


def fmt_window(x_min, x_max, W=None):
    """The grid window as the interval it is, with its width.

    Every narrative reported ``x_min`` and ``x_max`` as two loose numbers and
    the width as a third, so nothing on the page said they were the same
    fact. Written once, here.

    Parameters
    ----------
    x_min, x_max : float
        The window ends.
    W : float, optional
        The width; defaults to ``x_max - x_min``.

    Returns
    -------
    str
        ``'[0, 12,182,881] of width 12,182,881'``.
    """
    if W is None:
        W = float(x_max) - float(x_min)
    return (f'[{fmt_amount(x_min)}, {fmt_amount(x_max)}] '
            f'of width {fmt_amount(W)}')


def _move_phrase(st, win):
    """Describe a move naming only what actually changed.

    A line search often moves the bucket at the same grid size, and "log2 16 to
    16" reads as a bug rather than as "unchanged".
    """
    bs_moved = float(win['bs']) != float(st['bs0'])
    log2_moved = int(win['log2']) != int(st['log20'])
    bs_txt = f'bs {_fmt_bs(st["bs0"])} to {_fmt_bs(win["bs"])}'
    log2_txt = f'log2 {st["log20"]} to {int(win["log2"])}'
    if bs_moved and log2_moved:
        return f'{bs_txt}, {log2_txt}'
    if bs_moved:
        return f'{bs_txt}, at log2 {st["log20"]}'
    return f'{log2_txt}, at bs {_fmt_bs(st["bs0"])}'


def _sharpen_fmt(x):
    """Format a score for the narrative, so ``inf`` and ``nan`` read plainly."""
    if x is None:
        return 'n/a'
    if np.isnan(x):
        return 'n/a'
    if not np.isfinite(x):
        return 'infinite'
    return f'{x:.3g}'


def sharpen_describe(ob) -> str:
    """One-line summary of the last :func:`sharpen` probe.

    The verbose form is :func:`sharpen_explain`.
    """
    df = getattr(ob, '_sharpen_df', None)
    st = getattr(ob, '_sharpen_state', None)
    if df is None or st is None:
        return 'Sharpen: not run.'
    target = st['good_enough']
    if not st['ran']:
        return (f'Sharpen: not run, the grid already scores '
                f'{_sharpen_fmt(st["centre_score"])} against a target of '
                f'{target:g}.')
    win = df[df['selected']].iloc[0]
    secs = float(df['seconds'].sum())
    pinned = ' (discrete: bucket pinned, grid size only)' if st['bs_exact'] else ''
    head = (f'Sharpen: {len(df)} cells in {secs:.2f}s{pinned}, best score '
            f'{_sharpen_fmt(win["score"])} vs '
            f'{_sharpen_fmt(st["centre_score"])} at the centre, target '
            f'{target:g}.')
    if st.get('passed_over'):
        n = st['passed_over']
        head += (f' {n} better-scoring cell{"s" if n > 1 else ""} rejected as '
                 f'defective (mass off the end of the grid).')
    if 'bs_limit' in str(win['note']):
        head += ' The winner sits on the bs_limit and was still improving.'
    if win['bs'] == st['bs0'] and int(win['log2']) == st['log20']:
        return f'{head} Kept bs {_fmt_bs(st["bs0"])}, log2 {st["log20"]}.'
    move = _move_phrase(st, win)
    if st['execute']:
        return f'{head} Moved: {move}.'
    return f'{head} Recommends {move}; not executed, grid restored.'


def sharpen_explain(ob) -> str:
    """Verbose prose explaining the last :func:`sharpen` probe.

    The short form is :func:`sharpen_describe`.
    """
    df = getattr(ob, '_sharpen_df', None)
    st = getattr(ob, '_sharpen_state', None)
    if df is None or st is None:
        return ('The grid has not been sharpened. Call sharpen() to probe the '
                'neighbouring (bs, log2) cells and score each one.')
    out = [
        'The sharpen score measures how well the realized grid reproduces the '
        'analytic moments: severity and aggregate mean, CV and skewness, each '
        'relative error divided by its own validation tolerance and combined in '
        f'a power-{st["power"]:g} norm. The units are tolerance, so a score of 1 '
        'sits exactly on the validation pass boundary and smaller is better. '
        f'The target here is {st["good_enough"]:g}.']
    if not st['ran']:
        out.append(
            f'The current grid scores {_sharpen_fmt(st["centre_score"])}, at or '
            'under the target, and holds all of its mass, so no probe was run '
            'and nothing was changed. Pass good_enough=0 to force the probe '
            'regardless.')
        return ' '.join(out)
    win = df[df['selected']].iloc[0]
    if st['bs_exact']:
        out.append(
            'The severity is discrete and bs divides every atom, so the bucket '
            'is already exact and only the grid size was probed. No bucket '
            'change can help: a coarser one scatters the atoms off their own '
            'values, a finer one only wastes grid. What a discrete object fails '
            'on is extent, which is what log2 controls.')
        if st['lattice'] and st['lattice'] > st['bs0']:
            out.append(
                f'Note that bs {_fmt_bs(st["bs0"])} is finer than it needs to '
                f'be: the atoms sit on a lattice of {st["lattice"]:g}, which is '
                'the coarsest exact bucket and would hold the same book on a '
                'shorter grid.')
    else:
        out.append(
            f'{len(df)} cells were scored, a row per log2 and within each a line '
            'search out from the current bucket: bs is doubled until the score '
            'stops improving, then halved likewise. That works because the score '
            'has a single trough in bs, a larger bucket buying extent and losing '
            'resolution, so the rows are ragged and the frame is tidy rather '
            'than a matrix. A walk that runs out of bs_limit while still '
            'improving says so in its note, which is a different fact from '
            'turning. Extent is bs times 2**log2, so cells on an anti-diagonal '
            'share an extent and differ only in resolution. Reading them '
            'together says whether the grid is extent-limited, where the '
            'aggregate mean error runs far above the severity error and the fix '
            'is a wider grid, or resolution-limited, where the severity moments '
            'themselves are poorly reproduced and the fix is a finer bucket.')
    if st.get('centre_defective'):
        out.append(
            'The probe ran even though the score was at or under the target, '
            'because the current grid loses mass off its top end: '
            f'{_sharpen_fmt(df.loc[(0, 0), "deficit"])} of the distribution is '
            'missing. That is not something the score can report, since far '
            'tail mass barely moves the first three moments, and it is not '
            'cosmetic either: forwards and backwards survival functions differ '
            'by exactly the lost mass, so two correct-looking pricing routes '
            'disagree.')
    if st.get('passed_over'):
        out.append(
            f'{st["passed_over"]} cell(s) scored better than the winner and were '
            'rejected anyway, for carrying a genuine deficit. Soundness is a '
            'gate rather than a term in the score: every score term divides by '
            'its own validation tolerance and a deficit has none to divide by, '
            'and the cost of one is qualitative rather than a matter of degree. '
            'The threshold is the one the library already uses, the level at '
            'which a defective-distribution warning fires, so a cell rejected '
            'here is '
            'exactly one that would warn when you used it. The cure for a '
            'deficit is extent, which is why it can also be a reason to grow '
            'log2.')
    out.append(
        'Selection takes the best sound score among the cells that do not grow '
        'log2, '
        'the principle being that a probe should never make you pay more than '
        'you already are. A smaller log2 is a bonus rather than a goal, so it '
        'wins only when the score is a wash. log2 grows by one only when nothing '
        'at the current size or smaller reaches the target and something at the '
        'larger size does, which is why the more expensive row is not even '
        f'computed unless it is needed. Here: {st["reason"]}.')
    if win['bs'] != st['bs0'] or int(win['log2']) != st['log20']:
        verb = ('The grid moved to' if st['execute']
                else 'The recommendation is')
        out.append(
            f'{verb} bs {_fmt_bs(win["bs"])} at log2 {int(win["log2"])}, '
            f'scoring {_sharpen_fmt(win["score"])} against '
            f'{_sharpen_fmt(st["centre_score"])} before.')
        if st['execute']:
            out.append('Re-running sharpen re-centres the probe on the new grid '
                       'and continues from there, should more be available.')
        else:
            out.append('execute=False, so the original grid was restored and '
                       'nothing about the object changed.')
    else:
        out.append('No neighbour was enough better to justify moving, so the '
                   'grid was kept.')
    if _sharpen_is_port(ob):
        out.append(
            'One limitation for a portfolio: the score reads the total only. A '
            'portfolio whose total is well resolved can still hold one unit '
            'that is not, and that will not show here. Check the units '
            'individually with their own valid property.')
    return ' '.join(out)
