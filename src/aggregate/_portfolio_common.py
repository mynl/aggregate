"""Common exeqa-based numerics for ``Portfolio`` (Plan P4, Phase 4A.2).

Once *either* construction path -- the independent-sum FFT combine
(:mod:`aggregate._portfolio_density`) or a sample
(:mod:`aggregate._portfolio_sample`) -- has produced an augmented
``density_df`` carrying the ``exeqa_*`` columns, everything downstream
(apply a distortion, allocate the distorted price across units, diagnostics)
is **identical**. That identity is the switcheroo invariant: the allocation
numerics work off ``exeqa`` (conditional expectations), not the raw losses, so
they do not care how the joint distribution was built.

This module holds the free helpers that are agnostic to FFT-vs-sample origin.
The bulk of the exeqa numerics presently live as ``Portfolio`` methods
(``apply_distortion``, ``augmented_df``, the linear/lifted allocation,
``allocation_diagnostics``, ``bodoff``); they are progressively lifted here.
This is the home of **allocation** -- a Portfolio-only concern, *not* shared
with ``Aggregate`` and *not* part of the single-distribution ``_pricing``.
"""

import logging

import numpy as np
from scipy.spatial import ConvexHull

from .config import get_settings
from .spectral import choquet_weights

logger = logging.getLogger(__name__)

# EXEQA_NOISE_FLOOR: the exeqa_err floor below which a bucket's conditional
# decomposition is reliable (the augmented-frame truncation cut). Resolved once
# per session from config; mirrors the capture in ``_portfolio``.
EXEQA_NOISE_FLOOR = get_settings().validation.exeqa_noise_floor

__all__ = []


def check01(s):
    """ add 0 1 at start end """
    if 0 not in s:
        s = np.hstack((0, s))
    if 1 not in s:
        s = np.hstack((s, 1))
    return s


def make_array(s, gs):
    """ convert to np array and pad with 0 1 """
    s = np.array(s)
    gs = np.array(gs)
    s = check01(s)
    gs = check01(gs)
    return np.array((s, gs)).T


def convex_points(s, gs):
    """
    Extract the points that make the convex envelope, including 0 1

    Testers::

        %%sf 1 1 5 5

        s_values, gs_values = [.001,.0011, .002,.003, 0.005, .008, .01], [0.002,.02, .03, .035, 0.036, .045, 0.05]
        s_values, gs_values = [.001, .002,.003, .009, .011, 1],  [0.02, .03, .035, .05, 0.05, 1]
        s_values, gs_values = [.001, .002,.003, .009, .01, 1],  [0.02, .03, .035, .0351, 0.05, 1]
        s_values, gs_values = [0.01, 0.04], [0.03, 0.07]

        points = make_array(s_values, gs_values)
        ax.plot(points[:, 0], points[:, 1], 'x')

        s_values, gs_values = convex_points(s_values, gs_values)
        ax.plot(s_values, gs_values, 'r+')

        ax.set(xlim=[-0.0025, .1], ylim=[-0.0025, .1])

        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=.25)


    """
    points = make_array(s, gs)
    hull = ConvexHull(points)
    hv = hull.vertices[::-1]
    hv = np.roll(hv, -np.argmin(hv))
    return points[hv, :].T


def build_augmented(port, dist, *, view='ask', S_calculation='forwards',
                    allocation='lifted', allow_deficit=False):
    r"""Construct an augmented_df from ``port.density_df`` under ``dist``.

    Pure builder: returns the frame without touching ``port`` (the
    ``Portfolio.apply_distortion`` wrapper writes it into the cache). One
    O(n) sweep serves both allocation methods and **all** asset
    levels; everything is a direct sum over the exact discrete atom
    table that carries the origin ``x0`` (never ``cumsum(S)·bs``):

    .. math::

        exag_i(a) = \sum_{k \le a} \kappa_i(x_k)\,gp_k
                    + a\,g(S(a))\,\mathrm{TAIL}_i(a)

    with ``TAIL = exi_xgtag`` (beta, the distorted tail share) for
    ``allocation='lifted'`` and ``TAIL = exi_xgta`` (alpha, the
    objective tail share) for ``'linear'`` -- the *only* difference
    between the two methods. The distorted atom weights ``gp`` come
    from the one Choquet helper
    (:func:`~aggregate.spectral.choquet_weights`); the effective
    ``g`` resolves ``view`` × the portfolio's value-type role
    (:meth:`~aggregate.spectral.Distortion.effective_g`).

    Notes
    -----
    * **Mass guard (G6).** The lifted tail split integrates ``gp``
      across tail states, so a distortion with a mass on an
      unbounded support puts essentially all tail weight on the last
      represented bucket -- a different bounded problem, refused
      here (not just in ``price``). The linear split and all total
      columns depend on the tail only through ``g(S(a))`` and are
      stable; with a mass on an unbounded support the linear frame
      is built with the beta columns blanked.
    * **Signed support.** The total columns (``gS``, ``gp_total``,
      ``exag_total``) are exact on any signed window. The per-unit
      ``exag_*`` columns require the equal-priority share
      ``kappa/x`` -- not a recovery share on a signed grid
      (steering 6) -- and are left NaN; price signed unit variables
      directly via ``dot(exeqa_i, gp_total)``.
    * The frame is truncated at the last reliable ``exeqa`` row
      (FFT-noise cut, positional); the tail sums feeding beta are
      computed on the full law first.
    """
    if allocation not in ('lifted', 'linear'):
        raise ValueError(
            f"allocation must be 'lifted' or 'linear', not {allocation!r}")
    mass_unbounded = getattr(dist, 'has_mass', False) and not port.bounded
    if allocation == 'lifted' and mass_unbounded:
        raise ValueError(
            f"lifted allocation on an unbounded portfolio with a mass "
            f"distortion ({dist.name}) is unstable on the right edge "
            f"(essentially all the distortion weight lands on the last "
            f"bucket). Use allocation='linear' or certify "
            f"`portfolio.bounded = True` if the support is in fact bounded.")

    df = port.density_df.copy()
    loss = df.loss.to_numpy()
    p_total = df.p_total.to_numpy()
    signed = float(loss[0]) < 0

    g, g_prime, _ = dist.effective_g(view, is_loss_value=port._is_loss_value)
    w = choquet_weights(loss, p_total, g, S_calculation=S_calculation,
                        allow_deficit=allow_deficit)
    gS = w.gS
    gp = w.gp
    df['S'] = w.S
    df['gS'] = gS
    df['gF'] = 1 - gS
    df['gp_total'] = gp

    # exag_total(a) = rho(X ∧ a) at every grid point, carrying the
    # origin; the strict-tail gp sum telescopes to g(S(a)) exactly.
    df['exag_total'] = np.cumsum(loss * gp) + loss * gS

    if signed:
        # equal-priority kappa/x is not a recovery share on a signed
        # grid (steering 6): blank the per-unit distorted allocation;
        # totals above are exact.
        for unit in port.unit_names:
            df[f'exi_xgtag_{unit}'] = np.nan
            df[f'exag_{unit}'] = np.nan
        return df

    # Truncate where the exeqa decomposition breaks down (FFT noise;
    # discrete "gaps" in p_total are ignored — error is only
    # meaningful on support). Positional indexing: no zero-origin
    # assumption. Truncate BEFORE the per-unit sweep: beyond the cut
    # the shares are FFT junk, so the per-unit tail sums close with a
    # collapsed atom at the cut (below) rather than integrating noise.
    lnp = '|'.join(port.unit_names)
    idx_pne0 = df.query(' p_total > 0 ').index
    exeqa_err = np.abs(
        (df.loc[idx_pne0].filter(regex=f'exeqa_({lnp})').sum(axis=1) - df.loc[idx_pne0].loss) /
        df.loc[idx_pne0].loss)
    exeqa_err.iloc[0] = 0
    reliable = exeqa_err[exeqa_err < EXEQA_NOISE_FLOOR]
    if len(reliable):
        # +1 to keep the last reliable row (iloc[:idx] is exclusive)
        idx = int(np.searchsorted(df.index.to_numpy(),
                                  reliable.index[-1], side='left')) + 1
        logger.debug(f'index of max reliable value = {idx}')
        df = df.iloc[:idx]
        loss = loss[:idx]
        gS = gS[:idx]
        gp = gp[:idx]

    # Per-unit distorted tail shares (beta) and allocations. The share
    # columns are exi_xeqa = kappa/x with the origin-row 0 convention
    # (add_exa). The distorted tail mass beyond the cut, g(S_cut) in
    # total, collapses onto the cut row at its share -- the exact
    # analogue of the old fill-value closure; it vanishes when the
    # frame runs to the end of the support (gS_cut == 0).
    gSeq0 = gS == 0
    for unit in port.unit_names:
        share = df[f'exi_xeqa_{unit}'].to_numpy()
        kappa = df[f'exeqa_{unit}'].to_numpy()
        sgp = share * gp
        # strict tail sum Σ_{j>k} share_j gp_j + the collapsed closure
        closure = share[-1] * gS[-1]
        tail_g_share = (np.cumsum(sgp[::-1])[::-1] - sgp) + closure
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = np.where(gSeq0, 0.0, tail_g_share / gS)
        df[f'exi_xgtag_{unit}'] = beta
        tail = beta if allocation == 'lifted' else df[f'exi_xgta_{unit}'].to_numpy()
        if mass_unbounded:
            # linear frame under a mass distortion on an unbounded
            # support: beta inherits the top-bucket artifact -- blank
            # it; the alpha-based allocation below is stable.
            df[f'exi_xgtag_{unit}'] = np.nan
        df[f'exag_{unit}'] = (
            np.cumsum(kappa * gp)
            + np.where(gSeq0, 0.0, loss * gS * tail))
    return df


def unit_capital_at(port, aug, a, dist, *, view='ask', units=None):
    r"""Per-unit allocated capital ``Q_i(a)`` by the layer-ROE construction.

    .. math::

        Q_i(a) = \sum_{k:\,x_k < a}
            \left(gS_k\,\beta_{i,k} - S_k\,\alpha_{i,k}\right)
            \frac{1 - gS_k}{gS_k - S_k}\,\Delta x_k

    -- unit layer margin divided by total layer ROE, integrated. This
    is the one legitimately layer-based quantity (capital *is*
    allocated by layer); it is computed on demand at the requested
    ``a`` (D7: no persistent per-unit ``Q`` column). A ``gS == S``
    layer has zero margin and contributes zero capital (the ratio is
    guarded). The layer margin is taken as the exact first difference
    of the frame's cumulative margin ``exag_i - exa_i``, which equals
    ``(gS·beta - S·alpha)·Δx`` identically on the lifted frame and is
    the frame-consistent alpha-based margin on the linear frame.

    Parameters
    ----------
    port : Portfolio
        Supplies ``unit_names`` and the value-type role.
    aug : pandas.DataFrame
        An augmented frame from :meth:`Portfolio.apply_distortion` (carries
        ``S``/``gS``/``exa_*``/``exag_*``).
    a : float
        Asset level on the grid (callers snap).
    dist : Distortion
        The distortion that built ``aug``; supplies ``g'(1)`` for the
        L'Hôpital ROE limit at ``gS == 1`` layers (the fully
        loss-funded bottom, where unit margins may offset with zero
        total layer capital).
    view : {'ask', 'bid'}
        View used to build ``aug`` (resolves the effective ``g'``).
    units : list of str, optional
        Subset of unit names; default all ``unit_names``.

    Returns
    -------
    dict[str, float]
        ``{unit: Q_i(a)}``.

    Notes
    -----
    Reconciles ``Σ_i Q_i(a) == a - exag_total(a)`` exactly (asserted,
    scale-aware) whenever no layer hit the zero-margin guard; the
    identity carries the origin through ``exag_total``.
    """
    if units is None:
        units = list(port.unit_names)
    loss = aug.loss.to_numpy()
    pos = int(np.searchsorted(loss, a))
    # layers [x_k, x_{k+1}) for k < pos lie below a
    S = aug.S.to_numpy()[:pos]
    gS = aug.gS.to_numpy()[:pos]
    denom = gS - S
    one_minus_gS = 1 - gS
    # reciprocal layer ROE = (1 - gS)/(gS - S). At gS == 1 (fully
    # loss-funded layers, 0/0) use the L'Hôpital limit
    # 1/ROE(1) = g'(1)/(1 - g'(1)); when g'(1) == 1 (identity) the
    # unit margins are zero there and the fill is moot (guarded to 0
    # below). A genuine zero-total-margin layer (gS == S with
    # gS < 1) has zero margin and contributes zero capital.
    _, g_prime, _ = dist.effective_g(view,
                                     is_loss_value=port._is_loss_value)
    with np.errstate(divide='ignore', invalid='ignore'):
        gp1 = float(g_prime(1))
    if np.isnan(gp1):
        # g'(1) undefined (e.g. wang): the fully-loss-funded layers
        # get no capital, matching the legacy skip-NaN cumsum
        inv_fill = 0.0
    elif gp1 == 1:
        inv_fill = np.inf   # identity-like; margins are zero there
    else:
        inv_fill = gp1 / (1 - gp1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            one_minus_gS == 0, inv_fill,
            np.where(denom != 0, one_minus_gS / np.where(denom == 0, 1.0, denom), 0.0))
    out = {}
    for unit in units:
        # layer margin·Δx as the exact first difference of the frame's
        # cumulative margin: for the lifted frame this telescopes to
        # (gS·beta - S·alpha)·Δx identically; for the linear frame it
        # is the frame-consistent (alpha-based) margin, which stays
        # stable under a mass distortion on an unbounded support
        # (where beta is blanked).
        cum_margin = (aug[f'exag_{unit}']
                      - aug[f'exa_{unit}']).to_numpy()
        m_dx = np.diff(cum_margin)[:pos]
        # zero-margin layers contribute zero even when ratio is the
        # inf fill (0·inf guard)
        out[unit] = float(np.sum(np.where(m_dx == 0, 0.0, m_dx * ratio)))
    guarded = (denom == 0) & (one_minus_gS != 0)
    if (len(units) == len(port.unit_names) and pos
            and not guarded.any()):
        total = sum(out.values())
        target = a - float(aug['exag_total'].to_numpy()[pos])
        scale = max(abs(target), abs(a), 1e-30)
        rec_tol = max(1e-9, 64 * pos * np.finfo(float).eps)
        if abs(total - target) > rec_tol * scale:
            logger.warning(
                f'unit capital reconciliation: sum Q_i = {total:.10g} vs '
                f'a - exag_total = {target:.10g} '
                f'(rel {abs(total - target) / scale:.3e})')
    return out


def allocation_diagnostics(port, distortion, *, surface='lifted',
                           view='ask', S_calculation='forwards'):
    r"""Layer-curve diagnostic frame for a distorted portfolio.

    The explicit consumer surface for :func:`pedagogy.plot_twelve`
    and similar exhibits (pedagogy consumes, never dictates --
    steering 7): the core pricing frame no longer carries layer
    diagnostic columns. Sourced from the
    :meth:`Portfolio.apply_distortion` frame for ``surface``.

    Parameters
    ----------
    port : Portfolio
        The portfolio to diagnose.
    distortion : Distortion or str
        Passed through to :meth:`Portfolio.apply_distortion`.
    surface : {'lifted', 'linear'}
        Which allocation surface to diagnose.
    view, S_calculation
        Passed through to :meth:`Portfolio.apply_distortion`.

    Returns
    -------
    pandas.DataFrame
        Indexed like the augmented frame. Carries ``loss``, ``F``,
        ``gF``, ``S``, ``gS``, ``gp_total``; per unit ``exeqa_*``
        (kappa), ``exi_xgta_*`` (alpha), ``exi_xgtag_*`` (beta); and
        the layer curves, per unit and total:

        * ``layer_loss_*`` = ``S·alpha`` (total: ``S``)
        * ``layer_premium_*`` = ``gS·beta`` (total: ``gS``)
        * ``layer_margin_*`` = ``layer_premium - layer_loss``
        * ``layer_capital_*`` = ``layer_margin / layer_roe_total``
          (total: ``1 - gS``)
        * ``cum_margin_*`` = ``exag - exa`` (cumulative margin)
        * ``cum_capital_*`` = integrated layer capital (total: the
          exact ``loss - exag_total``)
        * ``layer_roe_total`` = ``(gS - S)/(1 - gS)`` with the
          L'Hôpital fill ``1/g'(1) - 1`` at ``gS == 1``.
    """
    if isinstance(distortion, str):
        distortion = port.distortions[distortion]
    aug = port.apply_distortion(distortion, view=view,
                                S_calculation=S_calculation,
                                allocation=surface)
    cols = ['loss', 'F', 'S', 'gS', 'gF', 'gp_total']
    cols += [c for unit in port.unit_names_ex
             for c in (f'exeqa_{unit}', f'exi_xgta_{unit}',
                       f'exi_xgtag_{unit}', f'exa_{unit}',
                       f'exag_{unit}')
             if c in aug.columns]
    df = aug[cols].copy()

    loss = aug.loss.to_numpy()
    S = aug.S.to_numpy()
    gS = aug.gS.to_numpy()
    # layer ROE with the L'Hôpital fill at the right end (gS == 1):
    # ROE(1) = lim (gS-S)/(1-gS) = 1/g'(1) - 1; when g'(1) == 0 the
    # limit is +inf (premium 100% loss-funded, no capital) and the
    # layer capital divides to zero.
    _, g_prime, _ = distortion.effective_g(
        view, is_loss_value=port._is_loss_value)
    gp1 = float(g_prime(1))
    roe_fill = np.inf if gp1 == 0 else 1 / gp1 - 1
    mq_total = 1 - gS
    with np.errstate(divide='ignore', invalid='ignore'):
        layer_roe = np.where(mq_total != 0, (gS - S) / mq_total, roe_fill)
    df['layer_roe_total'] = layer_roe
    dx = np.diff(loss)

    def cum_int(layer):
        """Σ_{j<k} layer_j Δx_j -- bottom-up layer integral."""
        return np.concatenate(([0.0], np.cumsum(layer[:-1] * dx)))

    with np.errstate(divide='ignore', invalid='ignore'):
        for unit in port.unit_names:
            alpha = aug[f'exi_xgta_{unit}'].to_numpy()
            beta = aug[f'exi_xgtag_{unit}'].to_numpy()
            ll = S * alpha
            lp = gS * beta
            lm = lp - ll
            df[f'layer_loss_{unit}'] = ll
            df[f'layer_premium_{unit}'] = lp
            df[f'layer_margin_{unit}'] = lm
            lq = np.where(layer_roe != 0, lm / layer_roe, np.nan)
            df[f'layer_capital_{unit}'] = lq
            df[f'cum_margin_{unit}'] = (
                aug[f'exag_{unit}'] - aug[f'exa_{unit}'])
            df[f'cum_capital_{unit}'] = cum_int(np.nan_to_num(lq))
    df['layer_loss_total'] = S
    df['layer_premium_total'] = gS
    df['layer_margin_total'] = gS - S
    df['layer_capital_total'] = mq_total
    df['cum_margin_total'] = aug['exag_total'] - aug['exa_total']
    # exact row identity, preferred over the integrated layer drift
    df['cum_capital_total'] = loss - aug['exag_total'].to_numpy()
    return df


def bodoff(port, *, p=0.99, a=0):
    """
    Determine Bodoff layer asset allocation at asset level a or
    VaR percentile p, one of which must be provided. Uses formula
    14.42 on p. 284 of Pricing Insurance Risk.

    :param port: the Portfolio.
    :param p: VaR percentile
    :param a: asset level
    :return: Bodoff layer asset allocation by unit
    """

    if p == 0 and a == 0:
        raise ValueError('Must provide either p or a')

    if p > 0:
        a = port.q(p)

    ans = port.density_df.filter(regex='exi_xgta_') \
        .loc[:a - port.bs, :].sum() * port.bs
    ans = ans.to_frame().T
    ans.index = [a]
    ans.index.name = 'a'
    ans = ans.drop(columns='exi_xgta_sum')
    ans.columns = [i.replace('exi_xgta_', '') for i in ans.columns]
    return ans
