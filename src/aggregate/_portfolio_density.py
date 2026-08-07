"""Density-based (independence) subsystem for ``Portfolio`` (Plan P4, Phase 4A.1).

The independent-sum combine: the unit ``Aggregate`` densities are convolved
(FFT is the mechanic) into the portfolio total, and ``add_exa`` builds the
conditional-expectation (``exeqa_*``) columns of the augmented ``density_df``.
The distinguishing feature is *densities*, not the FFT -- hence the module name.

This module holds the pure-math combine / ``add_exa`` kernels lifted out of the
``Portfolio`` class as free functions (taking the ``port`` object); the methods
become thin callers. The augmented ``density_df`` it produces is consumed by the
exeqa numerics in :mod:`aggregate._portfolio_common`, which are agnostic to
whether the joint was built here or from a sample
(:mod:`aggregate._portfolio_sample`).

A near-leaf: it takes the ``Portfolio`` as a parameter and never imports
``_portfolio``, so there is no import cycle.
"""

import logging
import numpy as np
import pandas as pd

from ._validation import VALIDATION_NOISE
from .utilities import ft, ift

logger = logging.getLogger(__name__)

__all__ = []


def _ft_nots(ft_units):
    """Per-unit "everything except this unit" FT products.

    ``ft_units`` maps unit name to the (padded) rfft of that unit's
    pmf laid into the physical-zero FFT buffer. Returns
    ``(ft_all, ft_nots)`` where ``ft_all`` is the product over all
    units and ``ft_nots[i]`` is the product over ``j != i``.

    Spectral division ``ft_all / ft_i`` is the fast path and is
    per-bin well-conditioned — ``(a·b)/a = b·(1+O(eps))`` — even on
    deeply underflowed spectra (the Step-0 audit measured division
    and prefix/suffix both at ~2e-9 against an untrimmed brute-force
    reference on tight thin-CV units). It fails only on **exactly
    zero** bins (symmetric severities zero bins exactly): those units
    use prefix/suffix partial products instead — ``O(m·M)`` total,
    no division, replacing the legacy ``O(m²·M)`` rebuild.

    Single owner of this construction, shared by ``add_exa`` callers
    (``update`` via ``add_exa`` and :func:`aggregate._portfolio_sample.swap_density_df`).
    """
    names = list(ft_units)
    ft_all = None
    for nm in names:
        ft_all = (np.copy(ft_units[nm]) if ft_all is None
                  else ft_all * ft_units[nm])
    nots = {}
    if len(names) == 1:
        nots[names[0]] = np.ones_like(ft_all)
        return ft_all, nots
    prefix = suffix = None
    for i, nm in enumerate(names):
        f = ft_units[nm]
        if not np.any(f == 0):
            nots[nm] = ft_all / f
            continue
        if prefix is None:
            # prefix[i] = prod(arrs[:i]), suffix[i] = prod(arrs[i:])
            arrs = [ft_units[n2] for n2 in names]
            m = len(arrs)
            prefix = [np.ones_like(ft_all)]
            for k in range(m - 1):
                prefix.append(prefix[-1] * arrs[k])
            suffix = [None] * (m + 1)
            suffix[m] = np.ones_like(ft_all)
            for k in range(m - 1, -1, -1):
                suffix[k] = suffix[k + 1] * arrs[k]
        nots[nm] = prefix[i] * suffix[i + 1]
    return ft_all, nots


def add_exa(port, df, unit_state):
    r"""Add the objective (conditional-expectation) allocation columns to ``df``.

    Per-unit and total: ``exeqa_*`` = ``E[X_i | X=a]`` (kappa),
    ``exlea_*`` = ``E[X_i | X≤a]``, ``exgta_*`` = ``E[X_i | X>a]``,
    ``exi_x_*`` = ``E[X_i / X]``, ``exi_xlea_*`` / ``exi_xgta_*`` /
    ``exi_xeqa_*`` = conditional-share variants (``exi_xgta`` is
    alpha, the objective tail share), ``e_*`` = unconditional mean,
    ``lev_*`` = stand-alone ``E[X_i ∧ a]``, ``exa_*`` = equal-priority
    expected loss allocated to unit ``i``. Also writes ``F``, ``S``,
    ``exa_total``, ``lev_total``.

    Names with a leading ``t`` clash with the ``total`` regex
    anchors — do not use them.

    Parameters
    ----------
    port : Portfolio
        The portfolio whose ``bs`` / ``name`` / ``unit_names`` drive the
        combine. ``Portfolio.add_exa`` is the thin method wrapper.
    df : pandas.DataFrame
        Frame to extend, carrying ``loss`` (the total output
        grid, any snapped origin — zero, positive or negative) and
        ``p_total``. ``update`` passes ``self.density_df``;
        :func:`aggregate._portfolio_sample.swap_density_df` passes its own frame.

    Returns
    -------
    pandas.DataFrame
        A **new** frame: ``df`` with the objective columns appended.
        The columns are attached in one :func:`pandas.concat` rather
        than assigned one at a time, so the result is consolidated
        instead of carrying a block per column. That is why this does
        not extend ``df`` in place; callers must take the return value.
    unit_state : dict[str, dict]
        Per-unit native-grid state captured at combine time (transient
        — the caller frees it after this returns): ``xs`` the unit's
        native loss grid, ``p`` the unit's pmf on it, and ``ft_p`` the
        (padded) rfft of the pmf laid into the physical-zero FFT
        buffer (``Aggregate.ftagg_density`` on the update path).

    Notes
    -----
    Kappa uses the shifted-support method
    (``dev/../math/docs/shifted-calc-method.md``): the numerator is
    ``ift(ft_xp_i · ft_not_i)`` where ``ft_xp_i`` is the FFT of the
    **native first-moment density** ``x · p_i(x)`` scattered into the
    same physical-zero buffer convention as ``ft_p`` — first moments,
    unlike probabilities, cannot be recovered from a rolled vector,
    so they are built from the true physical values. The result is
    relabelled onto the output window by the same single roll the
    combine applies to ``p_total``.

    All cumulative columns are **direct sums that carry the origin**
    (``E[X∧a] = Σ_{x≤a} x·p + a·S(a)``, etc.) — never
    ``cumsum(S)·bs``, which silently assumes the grid starts at 0.
    Ratio denominators carry explicit guards (``F``/``S`` at or below
    the validation noise floor blank the row) replacing the legacy
    ``loss_max`` / ``mult ∈ {1,10,100}`` blanking heuristic.

    On a signed (P&L) grid the equal-priority share ``kappa/x`` is
    not a recovery share, so the share-based columns (``exi_x*_*``,
    ``exa_{unit}``) are left NaN rather than divided through zero;
    the conditional means (``exeqa/exlea/exgta``), ``lev_*`` and the
    total columns are valid on any signed window.
    """
    cut_eps = np.finfo(float).eps
    # Explicit denominator guard (meta steering 1): F or S at or below
    # the validation noise floor cannot support a conditional mean.
    tol = VALIDATION_NOISE
    bs = port.bs
    n_out = len(df)
    loss = df['loss'].to_numpy()
    origin = float(loss[0])
    j0 = int(round(origin / bs))
    signed = origin < 0

    p_total = df['p_total'].to_numpy()
    if not np.all(p_total >= 0):
        n_neg = int((p_total < -cut_eps).sum())
        logger.warning(f'p_total has {n_neg} negative values; NOT setting to zero...')
    sum_p_total = p_total.sum()
    logger.info(f'{port.name}: sum of p_total is 1 - {1 - sum_p_total:12.8e} NOT rescaling.')
    F = np.cumsum(p_total)
    S = 1 - F
    F_small = F <= tol
    S_small = S <= tol

    logger.info(
        f'Portfolio.add_exa | {port.name}: S <= 0 values has length {len(np.argwhere(S <= 0))}')

    # Every column below is accumulated here and attached to ``df`` in a
    # SINGLE concat at the end. Assigning them one at a time inserts a
    # block per column, and a book with enough units pushes the frame past
    # pandas' 100-block threshold, which both fragments the frame and emits
    # a PerformanceWarning into every rendered notebook that builds one.
    # Insertion order is the historical column order, which dict preserves.
    cols = {'F': F, 'S': S}

    # total columns by direct sums carrying the origin
    cum_x = np.cumsum(loss * p_total)
    e_total = np.sum(loss * p_total)
    cols['exa_total'] = cum_x + loss * S
    cols['lev_total'] = cols['exa_total']
    with np.errstate(divide='ignore', invalid='ignore'):
        exlea_total = cum_x / F
        exgta_total = (e_total - cum_x) / S
    exlea_total[F_small] = np.nan
    exgta_total[S_small] = np.nan
    cols['exlea_total'] = exlea_total
    cols['e_total'] = e_total
    cols['exgta_total'] = exgta_total
    cols['exeqa_total'] = loss  # E[X | X=a] = a

    # kappa numerators: not-unit FT products plus the native
    # first-moment FTs, all in the physical-zero buffer convention
    # (transient — freed with unit_state when the caller returns).
    ft_all, ft_nots = _ft_nots(
        {nm: st['ft_p'] for nm, st in unit_state.items()})
    m_buf = 2 * (len(ft_all) - 1)

    for col in port.unit_names:
        st = unit_state[col]
        xs_n = np.asarray(st['xs'], dtype=float)
        p_n = np.asarray(st['p'], dtype=float)

        # exeqa_{unit} = E[X_i | X=a] = ift(ft_xp_i · ft_not_i) / p_total,
        # ft_xp_i from the native physical values (shifted-support method);
        # rebased onto the output window by the same roll as p_total.
        # j_native: signed bucket numbers; the % m_buf wrap places
        # negative-x mass at the top of the FFT buffer (same
        # convention as the signed combine).
        j_native = np.round(xs_n / bs).astype(np.int64)
        buf = np.zeros(m_buf)
        buf[j_native % m_buf] = xs_n * p_n
        num = ift(ft(buf, 0) * ft_nots[col], 0)
        if j0:
            num = np.roll(num, -j0)
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = np.real(num[:n_out]) / p_total
        # p_total ≈ 0 ⇒ exeqa is unreliable; zero it.
        kappa[p_total < cut_eps] = 0.0
        cols[f'exeqa_{col}'] = kappa

        # stand-alone lev_{col} = E[X_i ∧ a] from the native unit pmf:
        # cum_xp[i(a)] + a·(1 − cum_p[i(a)]), valid whether or not the
        # unit window overlaps the total window (a below the window
        # gives a; deficit mass stays in the tail term).
        cum_p_n = np.cumsum(p_n)
        cum_xp_n = np.cumsum(xs_n * p_n)
        pos = np.searchsorted(j_native,
                              np.round(loss / bs).astype(np.int64),
                              side='right') - 1
        inside = pos >= 0
        pos_c = np.maximum(pos, 0)
        cols[f'lev_{col}'] = np.where(
            inside, cum_xp_n[pos_c] + loss * (1 - cum_p_n[pos_c]), loss)

        # e_{col} from the native pmf (the unit's represented mean)
        e_col = float(np.sum(xs_n * p_n))
        cols[f'e_{col}'] = e_col

        # conditional means by direct sums of kappa · p_total
        kp = kappa * p_total
        cum_xi = np.cumsum(kp)
        with np.errstate(divide='ignore', invalid='ignore'):
            exlea = cum_xi / F
            exgta = (e_col - cum_xi) / S
        exlea[F_small] = np.nan
        exgta[S_small] = np.nan
        cols[f'exlea_{col}'] = exlea
        cols[f'exgta_{col}'] = exgta

        if signed:
            # kappa/x is not a recovery share on a signed grid
            # (steering 6): blank rather than divide through zero.
            cols[f'exi_x_{col}'] = np.nan
            cols[f'exi_xlea_{col}'] = np.nan
            cols[f'exi_xgta_{col}'] = np.nan
            cols[f'exi_xeqa_{col}'] = np.nan
            cols[f'exa_{col}'] = np.nan
            continue

        # share s_i(x) = kappa_i(x)/x; the origin row x=0 takes the
        # 0 convention (X=0 ⇒ X_i=0 a.s. on a non-negative book).
        with np.errstate(divide='ignore', invalid='ignore'):
            share = np.where(np.abs(loss) < bs / 2, 0.0, kappa / loss)
        sp = share * p_total
        cols[f'exi_x_{col}'] = np.sum(sp)
        cum_sp = np.cumsum(sp)
        with np.errstate(divide='ignore', invalid='ignore'):
            exi_xlea = cum_sp / F
        exi_xlea[F_small] = 0.0
        cols[f'exi_xlea_{col}'] = exi_xlea

        # tail_share_k = Σ_{j>k} share_j·p_j (reverse cumsum);
        # alpha = exi_xgta = tail_share / S. The last row has no
        # information past the grid: NaN when material tail mass
        # remains, 0 when the support is exhausted.
        rev = np.cumsum(sp[::-1])[::-1]
        tail_share = np.append(rev[1:], 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = tail_share / S
        alpha[S_small] = 0.0
        if S[-1] > tol:
            alpha[-1] = np.nan
        cols[f'exi_xgta_{col}'] = alpha
        cols[f'exi_xeqa_{col}'] = share

        # exa_{col} = E[X_i(a)] = Σ_{x≤a} kappa_i·p + a·tail_share(a)
        # — the direct-sum form of ∫ S·alpha dx, carrying the origin.
        cols[f'exa_{col}'] = cum_xi + loss * tail_share

    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)

    # Sum-of-shares check columns, read back off the concatenated frame.
    for metric in ['exi_xlea_', 'exi_xgta_', 'exi_xeqa_']:
        df[metric + 'sum'] = df.filter(regex=metric + '[^η]').sum(axis=1)

    return df
