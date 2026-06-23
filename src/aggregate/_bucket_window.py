"""Bucket and window sizing for the aggregate FFT grid.

Extracted from ``distributions.py`` / ``_aggregate.py`` (Phase 1b, shared concerns). A leaf/near-leaf: it never imports ``_aggregate``/``_portfolio`` (takes plain data / a distribution object), which is what lets Portfolio reuse it in P4.
"""

import logging
import numpy as np
import scipy.stats as ss
from scipy.optimize import NoConvergence  # noqa
from .config import get_settings
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
# is provably narrow (a concentrated aggregate, agg_cv < 1/z); it never fires for
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


# BUCKET_SIZING_P: percentile of the fitted distribution fed to
# recommend_bucket to size bs (formerly BUCKET_SIZING_P). >1 is read as nines.
BUCKET_SIZING_P = get_settings().discretization.bucket_sizing_p


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
    text = (f'{method} grid: bs={float(used["bs"]):g}, log2={int(used["log2"])}, '
            f'x_min={float(used["x_min"]):g} (x_max={top:g})')
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
        cand_txt = ', '.join(f'{i} {float(df.loc[i, "W"]):g}' for i in applied)
        parts.append(
            f'Of the methods that applied ({cand_txt}), the {method} method won '
            f'(window width {W:g}) -- {blurb}.')
    else:
        parts.append(f'The {method} method won (window width {W:g}) -- {blurb}.')

    raw = getattr(agg, '_bs_raw', None)
    if raw is not None:
        parts.append(
            f'The window produces a raw bs {raw:g} which dyadically rounds to '
            f'{bs:g} producing a final {W:g} window width.')

    # natural support bounds + concentration from the aggregate tail row
    try:
        trow = agg.tail_df.loc['aggregate']
    except Exception:  # pragma: no cover - defensive
        trow = None
    if trow is not None:
        if trow['left_tail'] == 'bounded' and np.isfinite(float(trow['min'])):
            parts.append(f'It has a natural lower support bound of {float(trow["min"]):g}.')
        if trow['right_tail'] == 'bounded' and np.isfinite(float(trow['max'])):
            parts.append(f'It has a natural upper support bound of {float(trow["max"]):g}.')
        cv = trow.get('cv')
        if bool(trow.get('concentrated')) and cv is not None and np.isfinite(float(cv)):
            parts.append(f'The distribution is concentrated with a CV of {float(cv):g}.')

    parts.append(f'The recommended x_min is {x_min:g} resulting in x_max of {x_max:g}.')

    clip = agg._bs_clip
    if clip is not None:
        cm = clip.get('clipped_mass', float('nan'))
        cm_txt = f'~{cm:.3g}' if np.isfinite(cm) else 'a sliver'
        msg = (f'The heavy right tail reaches {float(clip["reach"]):g}, past the '
               f'grid x_max {x_max:g} ({cm_txt} of the mass clipped, a reported '
               f'deficit not normalized); the analysis suggests increasing log2 '
               f'to {int(clip["need_log2"])}.')
        if color:
            msg = f'{_tail._ANSI_THICK}{msg}{_tail._ANSI_RESET}'
        parts.append(msg)
    return ' '.join(parts)
