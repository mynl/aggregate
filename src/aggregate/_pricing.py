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

import pandas as pd

from .spectral import Distortion
from .pentagon import complete_pentagon, Pentagon


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
