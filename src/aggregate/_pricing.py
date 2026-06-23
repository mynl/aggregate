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

import numpy as np
import pandas as pd

from .spectral import Distortion, DISTORTION_DTYPE
from .pentagon import complete_pentagon, Pentagon

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
        # Distortion spec as dict
        g = Distortion(**g)

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
    """
    if (p is None) == (a is None):
        raise ValueError(
            'calibrate_distortions requires exactly one of p= (probability) '
            'or a= (asset level).')
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
        S=S, bs=obj.bs, premium_target=P, ess_sup=ess_sup, assets=a, el=exa,
        names=names)
    distortion_df, calibration_df = _calibration_frames(
        dists, coc, p_val, obj.cdf(a), exa, P, a)
    obj.distortions = dists
    obj.distortion_df = distortion_df
    obj.calibration_df = calibration_df
    return distortion_df
