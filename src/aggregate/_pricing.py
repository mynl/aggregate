"""Single-distribution pricing: pentagon completion and distortion application.

Extracted from ``_aggregate.py`` (Phase 1b, shared concerns). Thin orchestration
over a single distribution-bearing object (``agg`` -- an ``Aggregate``, or in P4
a ``Portfolio`` total): the pentagon algebra lives in ``pentagon.py`` and the
distortion in ``spectral.py``. A near-leaf -- it takes the object as a parameter
and never imports ``_aggregate``/``_portfolio``, which is what lets P4 route
``Portfolio``-total pricing through it. See ``plan-README.md`` for the
pricing/allocation boundary.

The distortion-calibration capability (``Distortion.calibrate_set`` + the
pentagon->target glue) is **not** here yet -- it lands in Phase 1c.
"""

import warnings

import numpy as np
import pandas as pd

from .spectral import Distortion, DISTORTION_DTYPE, VALIDATION_NOISE
from .pentagon import complete_pentagon, Pentagon
from .constants import DegenerateEvaluationWarning

# The standard pricing distortion set calibrated by ``calibrate_distortions``.
DEFAULT_CALIBRATION_DISTORTIONS = ('ccoc', 'ph', 'wang', 'dual', 'tvar')


def price(agg, p, g, kind='var'):
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

    :param agg: the distribution-bearing object (an ``Aggregate``) to price.
    :param p: a distortion function spec or just a number; if >1 assets, if <1 a prob converted to quantile
    :param kind: var lower upper tvar
    :param g:  pricing distortion function
    :return:
    """

    # figure regulatory assets; applied to unlimited losses
    vd = agg.var_dict(p, kind, snap=True)
    a_reg = vd[agg.name]

    # figure pricing distortion
    if isinstance(g, Distortion):
        # just use it
        pass
    else:
        # Distortion spec as dict (may carry note/tags/hints/doc metadata)
        g = Distortion.from_spec(g)

    agg.apply_distortion(g)
    aug_row = agg.density_df.loc[a_reg]

    el = aug_row['exa']
    P = aug_row['exag']
    M = P - el
    Q = a_reg - P

    # one-row canonical pentagon (pentagon.py owns the identities + order)
    df = pd.DataFrame(
        [[el, M, P, Q]], columns=['L', 'M', 'P', 'Q'],
        index=pd.Index([agg.name], name='unit'),
    )
    return complete_pentagon(df)


def price_pentagon(agg, *, p=None, a=None, P=None, M=None, Q=None,
                   LR=None, PQ=None, ROE=None):
    """Complete the pricing octet at a capital level given one target.

    Fix the capital level with exactly one of ``p`` (a VaR probability,
    ``a = agg.q(p)``) or ``a`` (an asset level, snapped to the grid), then
    supply exactly one pricing target -- premium ``P``, cost of capital
    ``ROE`` (a.k.a. CoC), or a loss ratio via ``LR`` (equivalently ``M``,
    ``Q`` or ``PQ``). Returns the canonical one-row (``'total'``) pentagon
    ``DataFrame`` (columns :data:`~aggregate.pentagon.PENTAGON_STATS`).
    The target keywords match the canonical stat names.

    Pure accounting completion against the object's expected loss at the
    chosen capital level -- **no distortion is involved** (contrast
    :func:`price`, which prices with a :class:`Distortion`). The triple
    ``{L, a, target}`` is solved by :meth:`Pentagon.solve`.

    Parameters
    ----------
    agg : Aggregate
        The distribution-bearing object whose expected loss anchors the octet.
    p : float, optional
        VaR probability fixing the capital level; mutually exclusive with ``a``.
    a : float, optional
        Asset level fixing the capital; mutually exclusive with ``p``.
    P, M, Q, LR, PQ, ROE : float, optional
        Exactly one pricing target -- premium, margin, capital, loss ratio,
        premium-to-capital, or cost of capital (``M/Q``).

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
    if (p is None) == (a is None):
        raise ValueError('price_pentagon: pass exactly one of p= or a=')
    targets = {'P': P, 'M': M, 'Q': Q, 'LR': LR, 'PQ': PQ, 'ROE': ROE}
    n_targets = sum(v is not None for v in targets.values())
    if n_targets != 1:
        raise ValueError(
            'price_pentagon: pass exactly one pricing target '
            f'(one of P, M, Q, LR, PQ, ROE); got {n_targets}.')
    pent = Pentagon(obj=agg)
    pent.solve_obj(p=p, a=a, P=P, M=M, Q=Q, lr=LR, pq=PQ, roe=ROE)
    return pent.as_frame(unit='total')


def price_pentagon_ex(agg, *, p=None, a=None, L=None,
                      M=None, P=None, Q=None, LR=None, PQ=None, ROE=None):
    """Complete the pricing octet over the *full* pentagon vocabulary, free over
    the capital anchor.

    The full-power front door to pentagon pricing: it accepts any soluble
    configuration of ``{p, a, L, M, P, Q, LR, PQ, ROE}``, errors on the
    impossible ones, and **warns** when the solve leaned on accounting losses
    that the distribution does not reconcile. ``price_pentagon`` / ``solve_obj``
    are unchanged underneath; this is :meth:`Pentagon.solve` plus the ``prob_loss_assets``
    distributional bridge (``L = lev(a)``) and a ``p`` readout.

    The :class:`~aggregate.pentagon.Pentagon` engine is the gatekeeper:
    :meth:`Pentagon.solve` completes any soluble triple and raises
    ``ValueError`` on the rest (e.g. ``{PQ, ROE, LR}`` -- three scale-free
    ratios -- cannot pin the level). The distribution supplies exactly one extra
    equation, ``L = E[min(X, a)] =`` :meth:`~aggregate._grid_distribution.GridDistribution.lev`,
    injected only when the accounting is one equation short of a soluble triple.

    Parameters
    ----------
    agg : Aggregate or Portfolio
        The distribution-bearing object whose limited expected loss anchors the
        octet (routes through its :meth:`_grid_distribution`).
    p : float, optional
        VaR probability fixing the capital level (``a = q(p)``); not a pentagon
        variable, so it is translated to ``a`` up front. Mutually exclusive with
        ``a``.
    a : float, optional
        Asset level (capital anchor), snapped to the grid.
    L : float, optional
        Limited-expected-loss anchor; ``a`` is root-found from ``lev(a) = L``
        via :meth:`~aggregate._grid_distribution.GridDistribution.prob_loss_assets`.
    M, P, Q, LR, PQ, ROE : float, optional
        Pentagon targets (margin, premium, capital, loss ratio,
        premium-to-capital, cost of capital).

    Returns
    -------
    pandas.DataFrame
        One ``'total'`` row; a leading ``p`` descriptor column followed by the
        eight canonical pentagon stats (mirrors the ``calibration_df`` layout).

    Warns
    -----
    UserWarning
        When the solved ``L`` (e.g. an accounting ``P - M``) disagrees with the
        limited expected loss ``E[min(X, a)]`` at the solved assets, beyond the
        bucket tolerance. A pricing-time advisory only -- deliberately **not**
        routed through ``explain_validation`` / the ``constants.py`` flags.

    Raises
    ------
    ValueError
        If both ``p`` and ``a`` are given, or the configuration is insoluble
        (propagated from :meth:`Pentagon.solve`).

    Notes
    -----
    The injection rule, after translating ``p -> a``: count the supplied
    pentagon quantities. If exactly two are supplied and one anchors a capital
    level (``a`` known, or ``L`` given), inject the curve equation -- resolve
    the consistent ``(p, L, a)`` via ``prob_loss_assets`` -- to complete the triple. If three
    are already supplied, hand straight to :meth:`Pentagon.solve` (``L`` is
    whatever the accounting yields). Fewer than three with no determinable level
    is under-determined and ``solve`` raises.

    The uniform post-check -- ``|L_solved - lev(a_solved)|`` against the bucket
    size -- classifies every config without mode tracking: curve-anchored solves
    (``p``/``a``/``L``) match by construction and stay silent; only an
    accounting-determined ``L`` can diverge and warn. ``price_ccoc`` (anchors on
    ``p``) therefore never warns.

    Signed / payoff distributions are out of scope (``lev``/``q`` assume the
    zero-based grid); see :meth:`~aggregate._grid_distribution.GridDistribution.prob_loss_assets`.
    """
    if p is not None and a is not None:
        raise ValueError('price_pentagon_ex: pass at most one of p= or a=.')
    gd = agg._grid_distribution()

    # 1. translate the probability spelling (p is not a pentagon variable).
    if p is not None:
        a = float(gd.q(p))

    # 2. inject L = lev(a) only when the accounting is one equation short and a
    #    capital level is anchorable.
    quantities = (L, M, P, Q, a, LR, PQ, ROE)
    n = sum(v is not None for v in quantities)
    if n == 2 and (a is not None or L is not None):
        if a is not None:
            res = gd.prob_loss_assets(a=a)
        else:
            res = gd.prob_loss_assets(L=L)
        a, L = res.a, res.L

    # 3. solve via the existing engine (raises on insoluble configs).
    pent = Pentagon(obj=agg)
    pent.solve(L=L, M=M, P=P, Q=Q, a=a, lr=LR, pq=PQ, roe=ROE)

    # 4. uniform post-check + warn: compare the solved L to the curve's lev(a).
    a_solved = float(pent.a)
    L_solved = float(pent.L)
    lev_a = float(gd.lev(a_solved))
    tol = gd.bs if gd.bs is not None else 0.0
    if abs(L_solved - lev_a) > tol:
        warnings.warn(
            f'price_pentagon_ex: solved by accounting identities; L = '
            f'{L_solved:.6g} does not match the limited expected loss '
            f'E[min(X,a)] = {lev_a:.6g} at assets a = {a_solved:.6g} '
            f'(gap {L_solved - lev_a:.3g}). The accounting loss ignores the '
            f'limit/default haircut E[(X-a)+].', UserWarning)

    # 5. report p alongside the octet (calibration_df precedent: p as a leading
    #    descriptor next to the eight stats).
    p_solved = float(gd.cdf(a_solved))
    return complete_pentagon(pd.DataFrame(
        [[p_solved, pent.L, pent.M, pent.P, pent.Q]],
        columns=['p', 'L', 'M', 'P', 'Q'],
        index=pd.Index(['total'], name='unit')))


def price_ccoc(obj, ccoc, *, p):
    """Price ``obj`` (an Aggregate or a Portfolio total) at a constant cost of
    capital ``ccoc`` and VaR level ``p``. No distortion involved -- a thin alias
    for ``obj.price_pentagon(p=p, ROE=ccoc)``, returning the canonical one-row
    (``'total'``) pentagon ``DataFrame``. Delegating to ``obj.price_pentagon``
    keeps each class's own completion semantics.
    """
    return obj.price_pentagon(p=p, ROE=ccoc)


# ---------------------------------------------------------------------------
# Distortion-set calibration (flavor (b): pentagon target -> Distortion).
#
# The pentagon->target glue: resolve the asset level ``a`` and premium target
# ``P`` from the object's distribution and a cost-of-capital input, then hand
# the survival datum to ``Distortion.calibrate_set`` (which owns the family
# loop). Both an ``Aggregate`` and a ``Portfolio`` total calibrate through here
# -- single-distribution pricing, no per-unit allocation (that stays a
# Portfolio concern). ``GridDistribution`` is never imported; the object hands
# its survival/loss data in.
# ---------------------------------------------------------------------------

def _calibration_survival(density, bs, assets):
    """Resolve the calibration survival vector ``S`` over ``[0, assets]`` and the
    essential supremum from a 0-based ``p_total`` density (the full contiguous
    bs-grid). Faithful extraction of the default (``S_calc='cumsum'``) path of
    the former ``Portfolio.calibrate_distortion``; ``S`` is strictly positive
    and weakly decreasing.
    """
    Splus = (1 - density.loc[0:assets].cumsum()).values
    last_non_zero = np.argwhere(Splus)
    ess_sup = 0.0
    if len(last_non_zero) == 0:
        last_non_zero = len(Splus) + 1
    else:
        last_non_zero = last_non_zero.max()
    if last_non_zero + 1 < len(Splus):
        # truncate at first zero; record where the mass runs out
        S = Splus[:last_non_zero + 1]
        ess_sup = density.index[last_non_zero + 1]
    else:
        S = (1 - density.loc[0:assets - bs].cumsum()).values
    assert np.all(S > 0) and np.all(S[:-1] >= S[1:])
    return S, ess_sup


def _limited_ev(density, bs, assets):
    """``E[min(X, assets)] = bs · Σ_{x < assets} S(x)`` on the full contiguous
    bs-grid -- the ``add_exa`` / ``exa_total`` convention. Used for the expected
    loss when the object has no ``exa_total`` column (an ``Aggregate``); matches
    a one-unit ``Portfolio``'s ``exa_total`` to floating-point dust.
    """
    S = 1.0 - density.cumsum()
    return float(bs * S[S.index < assets].sum())


def _canonical_loss_frame(obj):
    """Slide (and, for a payoff, reverse) an object's outcome onto the canonical
    non-negative loss axis on which the layer-integral calibration is exact.

    The layer / Lee form ``∫₀^∞ g(S(x)) dx`` that every subclass ``calibrate``
    integrates is valid only for ``X ≥ 0``. Distortion risk measures are
    translation-equivariant and comonotone-additive, so for a signed outcome the
    correct two-sided measure equals the one-sided integral over the shifted,
    non-negative variable ``Z`` -- with the shift recovered afterwards. The two
    transforms are orthogonal:

    * **reverse** (``not obj._is_loss_value``) -- a payoff is "more is better",
      so its bad tail is on the *left*; the physical reverse ``X -> -X`` puts the
      bad tail back on the right where a concave ``g`` loads it. This is the
      matched half of the ``g_dual`` flip that ``Distortion.effective_g`` applies
      when *pricing* a payoff; the two cancel on price (see ``spectral.py:772``).
    * **shift** ``c = max(0, -min(support))`` -- slide the (optionally reversed)
      loss-convention outcome so its least value carrying mass sits at 0.

    The FFT zero-padding is trimmed (mass below :data:`VALIDATION_NOISE` is dust)
    so ``c`` keys off the genuine support, not the padded grid extent, and the
    returned grid is a clean contiguous ``bs``-lattice starting at 0.

    Parameters
    ----------
    obj : Aggregate or Portfolio
        The distribution-bearing object; reads ``density_df['p_total']``,
        ``_is_loss_value`` and ``bs``.

    Returns
    -------
    (dz, c, reverse) : (pandas.Series, float, bool)
        ``dz`` -- the canonical 0-based loss density on a contiguous ``bs``-grid;
        ``c`` -- the shift applied (``L_canonical = L_rv + c``); ``reverse`` --
        whether the outcome was reversed (payoff role).

    Notes
    -----
    Caller un-shifts the receipt (``L``, ``P``, ``a`` move by ``-c``; ``M``,
    ``Q``, ``coc`` are shift-invariant) so the pentagon reports in the user's
    loss-convention units -- ``L``/``P``/``a`` go negative together only when the
    position is genuinely beneficial (a payoff that is really a profit). The
    stored ``Distortion`` shapes are frame-free.

    A thin wrapper over :func:`_canonical_grid`, which does the same work for a
    bare ``(x, p, bs)`` grid and so also serves an irregular one.
    """
    s = obj.density_df['p_total']
    z, p, c, reverse = _canonical_grid(
        s.index.to_numpy(dtype=float), s.to_numpy(dtype=float),
        obj.bs, obj._is_loss_value)
    return pd.Series(p, index=z), c, reverse


def _canonical_grid(x, p, bs, is_loss_value):
    """The canonical non-negative loss grid for a bare ``(x, p)`` distribution.

    The array-level engine behind :func:`_canonical_loss_frame`; see there for
    what the reverse and the shift are doing and why. Split out so the same
    transform serves an **irregular** grid, which is what a margin pushforward
    generally lands on.

    Parameters
    ----------
    x, p : ndarray
        Sorted outcomes and their masses.
    bs : float or None
        Bucket size when the grid is a regular lattice; ``None`` for an exact
        irregular (atomic) grid.
    is_loss_value : bool
        ``False`` for a payoff, whose bad tail is on the left and which is
        therefore physically reversed.

    Returns
    -------
    (z, p, c, reverse) : (ndarray, ndarray, float, bool)
        The 0-based canonical outcomes, their masses, the shift applied and
        whether the outcome was reversed.

    Notes
    -----
    Both flavors trim the same way, and they must. The end-of-grid mass below
    :data:`VALIDATION_NOISE` is numerical dust wherever it appears -- FFT
    zero-padding on a lattice, and the same padding carried through a
    pushforward on an irregular grid -- so dropping it is what makes ``c`` key
    off the genuine support rather than the grid extent. That matters far more
    for evaluation than for pricing: ``c`` *is* the calibration target, so a
    dust bucket at the top of a padded grid would set a target the position
    cannot meaningfully be measured against.

    Only the final snap differs: a lattice is rounded back onto its ``bs`` grid
    to kill the ``x + c`` floating dust, and an irregular grid keeps its exact
    node positions.
    """
    reverse = not is_loss_value
    if reverse:
        x = -x
        order = np.argsort(x, kind='stable')
        x, p = x[order], p[order]
    # trim padding to the genuine mass support so c keys off real support
    mass = p > VALIDATION_NOISE
    if not mass.any():
        mass = np.ones_like(p, dtype=bool)
    lo = int(np.argmax(mass))
    hi = len(mass) - int(np.argmax(mass[::-1]))
    x, p = x[lo:hi], p[lo:hi]
    c = max(0.0, -float(x[0]))
    if bs is None:
        return x + c, p, c, reverse
    # snap onto a clean bs-lattice from 0 (kills the x + c floating dust)
    return np.round((x + c) / bs) * bs, p, c, reverse


def _calibration_survival_atomic(z, p):
    """Survival vector, node widths and essential supremum for an **irregular**
    0-based grid: the exact quadrature datum for :meth:`Distortion.calibrate`.

    Parameters
    ----------
    z, p : ndarray
        Canonical 0-based outcomes (strictly increasing) and their masses.

    Returns
    -------
    (S, dx, ess_sup) : (ndarray, ndarray, float)
        ``S[i] = P(Z > z[i])`` truncated to its strictly positive prefix,
        ``dx[i] = z[i+1] - z[i]`` aligned to it, and the largest outcome
        carrying mass.

    Notes
    -----
    The survival function of an atomic distribution is a step function: it holds
    ``S[i]`` across ``[z[i], z[i+1])`` and is 0 above the top atom. So

    .. math:: \\int_0^\\infty g(S(x))\\,dx = \\sum_i g(S_i)\\,(z_{i+1} - z_i)

    is the layer integral **exactly**, not a discretization of it. That is the
    whole reason this path exists rather than rebucketing the margin onto a
    lattice, which is mean-preserving and therefore looks right while being
    wrong about the tail the index is reading.

    The lattice twin is :func:`_calibration_survival`, which returns the same
    datum with a scalar width.
    """
    if len(z) < 2:
        raise ValueError(
            'degenerate calibration grid: a single outcome carries all the '
            'mass, so there is no layer to integrate.')
    S_full = 1.0 - np.cumsum(p)
    positive = np.flatnonzero(S_full > 0)
    if len(positive) == 0:
        raise ValueError(
            'degenerate calibration grid: all mass sits at the smallest '
            'outcome, so the distorted integral is identically zero.')
    # S is non-increasing, so its positive part is a prefix ending at k.
    k = min(int(positive.max()), len(z) - 2)
    S = S_full[:k + 1]
    dx = np.diff(z)[:k + 1]
    ess_sup = float(z[k + 1])
    assert np.all(S > 0) and np.all(S[:-1] >= S[1:])
    return S, dx, ess_sup


def _calibration_datum(z, p, bs, assets):
    """Dispatch the calibration datum ``(S, dx, ess_sup)`` on grid regularity.

    ``bs`` given routes to the lattice :func:`_calibration_survival` with a
    scalar width, which keeps the classic pricing path bit-for-bit what it was;
    ``bs`` ``None`` routes to the exact :func:`_calibration_survival_atomic`
    with a width vector.
    """
    if bs is None:
        return _calibration_survival_atomic(z, p)
    S, ess_sup = _calibration_survival(pd.Series(p, index=z), bs, assets)
    return S, bs, ess_sup


def _calibration_frames(dists, coc, p_val, Fa, exa, P, a):
    """Build the ``(distortion_df, calibration_df)`` receipt for a calibrated set
    -- the per-distortion shapes/errors and the shared one-row pentagon target.
    Schema identical to the former ``Portfolio.calibrate_distortions``.
    """
    names = list(dists)
    rows = []
    for dname in names:
        dist = dists[dname]
        # param_name is the family's natural parameter ('a', 'lam', 'b', 'p');
        # ccoc has none -> 'r'. gini_p = 2∫g - 1 (= p_equiv); area = ∫g.
        param_name = getattr(dist, 'param_name', None) or 'r'
        rows.append([param_name, dist.shape, dist.error, dist.gini_p,
                     (dist.gini_p + 1) / 2])
    distortion_df = pd.DataFrame(
        rows,
        columns=['param_name', 'param', 'error', 'gini_p', 'area'],
        index=pd.CategoricalIndex(names, dtype=DISTORTION_DTYPE, name='distortion'),
    )
    calibration_df = complete_pentagon(
        pd.DataFrame([[coc, p_val, Fa, exa, P - exa, P, a - P]],
                     columns=['coc', 'p', 'F(a)', 'L', 'M', 'P', 'Q'],
                     index=pd.Index(['calibration'], name='unit')))
    return distortion_df, calibration_df


def calibrate_distortions(obj, coc, *, p=None, a=None, kind='lower',
                          names=DEFAULT_CALIBRATION_DISTORTIONS):
    """Calibrate the standard pricing distortion set to a cost-of-capital target
    on a single distribution (an ``Aggregate`` or a ``Portfolio`` total).

    Resolves the asset level ``a`` (from ``p`` or given), the expected loss
    ``exa = E[min(X, a)]`` and the premium target ``P`` (from ``coc`` via the
    ROE -> LR -> P inversion), then calls :meth:`Distortion.calibrate_set` once.
    Stores ``obj.distortions`` / ``obj.distortion_df`` / ``obj.calibration_df``
    (mirroring the legacy ``Portfolio`` behaviour) and returns ``distortion_df``.

    The expected loss is read from the object's ``exa_total`` column when it has
    one (a ``Portfolio``, byte-for-byte the legacy value) and otherwise computed
    on the full grid via :func:`_limited_ev` (an ``Aggregate``).

    Signed and payoff supports
    --------------------------
    When the outcome straddles 0 (an ``ssev`` severity, a negative ``dsev`` atom)
    or is a payoff (``pnl``, ``_is_loss_value`` False) the layer integral is run
    on the canonical non-negative loss frame ``Z`` built by
    :func:`_canonical_loss_frame` -- physically reversing a payoff and shifting by
    ``c = max(0, -min(support))``. The subclass ``calibrate`` math stays pure and
    0-based; only this caller does the in-/out-of-frame bookkeeping. The receipt
    is un-shifted back to the caller's loss convention (``L``, ``P``, ``a`` slide
    by ``-c``; ``M``, ``Q``, ``coc`` are shift-invariant), so ``L``/``P``/``a``
    read negative together exactly when the position is net-beneficial. The
    classic ``X >= 0`` loss path (``c = 0``, no reverse) is byte-for-byte
    unchanged. For a signed/payoff object the asset anchor is resolved as the
    lower quantile on ``Z`` (``kind`` is honoured only on the classic path).
    """
    if (p is None) == (a is None):
        raise ValueError(
            'calibrate_distortions requires exactly one of p= (probability) '
            'or a= (asset level).')

    transform = (not obj._is_loss_value
                 or float(obj.density_df.index.min()) < 0)
    if not transform:
        # ---- classic non-negative loss frame (c = 0): legacy path verbatim ---
        if a is None:
            a = obj.q(p, kind)
            p_val = p
        else:
            a = obj.snap(a)
            p_val = obj.cdf(a)
        density = obj.density_df['p_total']
        if 'exa_total' in obj.density_df.columns:
            exa = obj.density_df.loc[a, 'exa_total']
        else:
            exa = _limited_ev(density, obj.bs, a)
        # invert COC -> LR -> P (matches the legacy ROE -> LR -> P path).
        delta = coc / (1 + coc)
        nu = 1 - delta
        P = nu * exa + delta * a
        S, ess_sup = _calibration_survival(density, obj.bs, a)
        dists = Distortion.calibrate_set(
            S=S, dx=obj.bs, premium_target=P, ess_sup=ess_sup, assets=a,
            el=exa, names=names)
        distortion_df, calibration_df = _calibration_frames(
            dists, coc, p_val, obj.cdf(a), exa, P, a)
    else:
        # ---- signed / payoff: calibrate on the canonical loss frame Z --------
        bs = obj.bs
        dz, c, reverse = _canonical_loss_frame(obj)
        cz = dz.cumsum()                       # raw (deficit treated as classic)
        zi = dz.index.to_numpy()
        snap = lambda v: float(zi[int(np.abs(zi - v).argmin())])
        if a is None:
            if not reverse:
                # signed loss: reuse the object's robust quantile, then shift.
                a_z = snap(float(obj.q(p, kind)) + c)
            else:
                # payoff: lower quantile on the reversed law (no classic analog),
                # via a normalized cumsum so the PMF deficit can't skip an atom.
                czn = (dz / dz.sum()).cumsum().to_numpy()
                j = min(int(np.searchsorted(czn, p - VALIDATION_NOISE,
                                            side='left')), len(dz) - 1)
                a_z = float(zi[j])
            p_val = p
        else:
            # caller's a is in loss-convention units; snap a + c onto Z's grid
            a_z = snap(a + c)
            p_val = float(cz.loc[a_z])
        Fa = float(cz.loc[a_z])
        exa_z = _limited_ev(dz, bs, a_z)
        S, ess_sup = _calibration_survival(dz, bs, a_z)
        # in-frame COC -> P inversion (shift-covariant: P, exa, a all carry +c).
        delta = coc / (1 + coc)
        nu = 1 - delta
        P_z = nu * exa_z + delta * a_z
        dists = Distortion.calibrate_set(
            S=S, dx=bs, premium_target=P_z, ess_sup=ess_sup, assets=a_z,
            el=exa_z, names=names)
        # un-shift the receipt into the caller's loss convention.
        distortion_df, calibration_df = _calibration_frames(
            dists, coc, p_val, Fa, exa_z - c, P_z - c, a_z - c)

    obj.distortions = dists
    obj.distortion_df = distortion_df
    obj.calibration_df = calibration_df
    return distortion_df


# ---------------------------------------------------------------------------
# Evaluation (flavor (c): a held position -> its breakeven acceptability).
#
# Pricing asks what an obligation is worth. Evaluation asks how much stress a
# position you already hold survives. The two run the same Newton solve on the
# same layer integral; they differ in what is distorted and what the target is.
# ---------------------------------------------------------------------------

#: Distortion families the acceptability panel reports. ``ccoc`` is excluded:
#: its shape is a cost of capital, a rate rather than a stress level, and its
#: closed form needs an asset level that the acceptability question does not
#: supply.
EVAL_FAMILIES = ('ph', 'wang', 'dual', 'tvar')

#: Sign applied to a position's margin before the solve, by the role it is held
#: at. A ``buy`` position's margin is negative by construction (you pay for
#: cover), so it has no breakeven of its own; the acceptability question about a
#: purchased layer is the **seller's**, and the panel answers it by reading the
#: margin negated.
EVAL_SIGN = {'sell': 1.0, 'buy': -1.0}

#: Fixed column order of the acceptability panel. Tidy (long) form, one row per
#: ``(step, family)``: ``param`` holds each family's own natural parameter, so
#: it is a pure column only at this grain. ``role`` leads because it qualifies
#: the whole row, naming whose position the parameters describe.
#: ``area = (gini_p + 1) / 2`` is not carried, being a restatement of ``gini_p``.
EVAL_COLS = ['role', 'param_name', 'param', 'gini_p', 'error', 'status']


def _eval_sign(role):
    """Validate a position's role and return the sign its margin is read at.

    See :data:`EVAL_SIGN`. Raises rather than defaulting, so a typo surfaces at
    the call site instead of silently evaluating the wrong side of the trade.
    """
    try:
        return EVAL_SIGN[role]
    except (KeyError, TypeError):
        raise ValueError(
            f'role must be one of {", ".join(EVAL_SIGN)}, got {role!r}.'
        ) from None


def _eval_panel(rows, names):
    """Assemble one step's acceptability block, indexed by distortion family."""
    return pd.DataFrame(
        rows, columns=EVAL_COLS,
        index=pd.CategoricalIndex(list(names), dtype=DISTORTION_DTYPE,
                                  name='distortion'))


def no_distribution_panel(status, *, role='sell', names=None):
    """One step's block for a row that carries **no distribution** to distort.

    A ledger row can be a well-defined number without being a random variable:
    a stitched impact row is the difference of two statistics whose sides ride
    different marginals, so it has no law absent a joint. Such a row keeps its
    place in the panel and says why it is ``NaN``, rather than vanishing. It
    still carries its ``role``, so the sheet reports whose position it was even
    where there is nothing to solve.
    """
    _eval_sign(role)
    names = list(EVAL_FAMILIES if names is None else names)
    return _eval_panel(
        [[role, None, np.nan, np.nan, np.nan, status] for _ in names], names)


def evaluate_margin(gd, *, role='sell', names=None):
    """The Cherny and Madan breakeven acceptability panel for one margin.

    Solves, per distortion family, for the shape at which the risk-adjusted
    margin reaches 0: the most stress the position survives. The breakeven
    ``gini_p`` is the family-agnostic acceptability index.

    Parameters
    ----------
    gd : GridDistribution
        The **margin** distribution in payoff orientation (more is better),
        as the position is booked. Its grid may be a regular lattice or exact
        and irregular; the quadrature adapts (:func:`_calibration_datum`).
    role : {'sell', 'buy'}, default 'sell'
        How the position is held. A ``buy`` margin is negated before the solve
        and the panel answers for the seller (:data:`EVAL_SIGN`).
    names : sequence of str, optional
        Distortion families. Defaults to :data:`EVAL_FAMILIES`.

    Returns
    -------
    pandas.DataFrame
        Indexed by distortion family, columns :data:`EVAL_COLS`. A degenerate
        position returns the same shape with ``NaN`` parameters and a ``status``
        naming which case fired; it does **not** warn, leaving the caller to
        aggregate one warning per call (:func:`warn_degenerate`).

    Notes
    -----
    Write ``M`` for the margin and ``D = -M`` for its loss orientation.
    :func:`_canonical_grid` reverses the payoff and shifts by
    ``c = max(0, -min(D)) = max(0, max(M))``, giving ``Z = D + c >= 0``.
    Distortion risk measures are translation-equivariant, so

    .. math:: \\rho_g(M) = 0 \\iff \\rho_g(D) = 0 \\iff \\rho_g(Z) = c

    and **the calibration target is the shift constant**, with no premium term.
    That is the whole point of evaluating the margin rather than pricing an
    obligation against a premium: ``rho_g`` is comonotone-additive but not
    additive, so once the consideration is random (swing rating, a profit
    commission, reinstatement premium) ``rho_g(P) - rho_g(L)`` is not
    ``rho_g(P - L)`` and only the margin's own pushforward answers the question.

    The constant-premium case specializes back exactly. With ``M = P - L``,
    ``c = P - min(L)`` and ``Z = L - min(L)``, so the target is ``P - min(L)``
    and the solve is ``rho_g(L) = P``, the classic price-the-obligation form.

    A breakeven exists precisely when ``E[M] > 0`` (so ``E[Z] < c``) and ``M``
    is negative with positive probability (so ``ess_sup(Z) > c``), which is
    what the two guards below check. The target is then strictly interior and
    the Newton iteration has a root to find.

    Those same guards are why ``role`` exists. A purchased layer fails the
    ``E[M] > 0`` test by construction, since its margin to the buyer is the
    premium paid less the recoveries received. There is no breakeven to find on
    that side and never was; the question worth asking about a layer is what
    stress the **seller's** position survives, which is the buyer's margin
    negated. Reading a ``buy`` row at ``role='buy'`` answers that, so a ceded
    layer prices instead of reporting ``NaN``.
    """
    if gd.is_loss_value:
        raise ValueError(
            'evaluate_margin expects a margin in payoff orientation (more is '
            'better); this GridDistribution is loss-valued. Negate it, or use '
            'evaluate_constant_premium for a premium against a loss.')
    return _evaluate_margin_arrays(
        np.asarray(gd.x, dtype=float), np.asarray(gd.p, dtype=float),
        gd.bs, role=role, names=names)


def evaluate_constant_premium(x, p, bs, premium, *, role='sell', names=None):
    """Acceptability panel for the margin ``premium - X``: the ordinary case.

    A constant consideration against a loss distribution ``X``, which is the
    position an ``Aggregate`` or a ``Portfolio`` total represents. Routed
    through the same solve as every other margin rather than special-cased:
    translation equivariance makes ``rho_g(premium - X) = 0`` and
    ``rho_g(X) = premium`` the same statement, so there is one code path and
    the constant case is simply the one where it collapses.

    Parameters
    ----------
    x, p : ndarray
        The loss outcomes and their masses.
    bs : float or None
        Bucket size, or ``None`` for an irregular grid.
    premium : float
        The constant consideration held against ``X``.
    role : {'sell', 'buy'}, default 'sell'
        How the position is held; see :func:`evaluate_margin`. Writing the
        obligation as a loss already fixes the holder as the one who owes it,
        so ``sell`` is almost always right here.
    names : sequence of str, optional
        Distortion families; defaults to :data:`EVAL_FAMILIES`.

    Returns
    -------
    pandas.DataFrame
        As :func:`evaluate_margin`.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    # M = premium - X, re-sorted ascending (negating reverses the order).
    return _evaluate_margin_arrays((premium - x)[::-1], p[::-1], bs,
                                   role=role, names=names)


def _evaluate_margin_arrays(x, p, bs, *, role='sell', names=None):
    """The solve itself, on a margin's ``(x, p)`` arrays in payoff orientation.
    See :func:`evaluate_margin` for the math and the return contract."""
    if names is None:
        names = EVAL_FAMILIES
    names = list(names)
    # A 'buy' margin is read from the seller's side: negate, then re-sort
    # ascending, since negating reverses the order (:data:`EVAL_SIGN`).
    if _eval_sign(role) < 0:
        x, p = (-x)[::-1], p[::-1]
    mean = float(np.sum(x * p))

    # --- degeneracy: no stress level to solve for ---------------------------
    status = None
    if not (p > 0).any():
        status = 'no mass'
    elif float(p[x < 0].sum()) <= VALIDATION_NOISE:
        status = 'M >= 0 a.s. (arbitrage)'
    elif mean <= 0:
        status = f'E[M] = {mean:,.6g} <= 0'
    if status is not None:
        return _eval_panel(
            [[role, None, np.nan, np.nan, np.nan, status] for _ in names],
            names)

    # --- the solve ----------------------------------------------------------
    z, pz, c, _reverse = _canonical_grid(x, p, bs, is_loss_value=False)
    a_full = float(z[-1])
    S, dx, ess_sup = _calibration_datum(z, pz, bs, a_full)
    # E[min(Z, top)] on the same quadrature; only ``ccoc`` reads it.
    el = float(Distortion._quad(S, dx))
    dists = Distortion.calibrate_set(
        S=S, dx=dx, premium_target=c, ess_sup=ess_sup,
        assets=ess_sup or a_full, el=el, names=names)
    rows = []
    for nm in names:
        d = dists[nm]
        rows.append([role, getattr(d, 'param_name', None) or 'param',
                     d.shape, d.gini_p, d.error, 'ok'])
    return _eval_panel(rows, names)


def warn_degenerate(panel, where=''):
    """Raise one :class:`DegenerateEvaluationWarning` for a whole panel.

    Scans the assembled panel's ``status`` column and warns once, naming every
    degenerate step and its reason. One warning per ``evaluate`` call rather
    than one per row: a reinsurance program booked as its own step reads
    ``E[M] <= 0`` correctly (you pay for cover), and a row-by-row warning would
    make that routine truth into noise.
    """
    bad = panel[panel['status'] != 'ok']
    if bad.empty:
        return
    if isinstance(bad.index, pd.MultiIndex):
        # one reason per step, in ledger order (a step is degenerate for a
        # reason that does not vary by family, so the first row speaks for it)
        first = bad.groupby(level=0, sort=False, observed=True)['status'].first()
        detail = '; '.join(f'{k}: {v}' for k, v in first.items())
    else:
        detail = bad['status'].iloc[0]
    warnings.warn(
        f'no breakeven acceptability level{f" for {where}" if where else ""}, '
        f'reporting NaN ({detail}).', DegenerateEvaluationWarning, stacklevel=3)
