"""Sample-based (dependence) subsystem for ``Portfolio`` (Plan P4, Phase 4A.3).

The dependence path: build a Portfolio surface from a *sample* rather than from
the independent-sum FFT combine -- the switcheroo (``swap_density_df``),
comonotonic allocations, and (via :mod:`aggregate.iman_conover`) the
Iman--Conover correlation induction. This is the **answer to "what about
portfolios with correlation?"**; it produces the *same* augmented
``density_df`` / ``exeqa_*`` columns the common exeqa numerics
(:mod:`aggregate._portfolio_common`) consume, so everything downstream is
agnostic to how the joint was built.

**Behaviour-frozen relocation.** This module holds the free functions moved out
of ``portfolio.py``; the substantive review of the sample / switcheroo /
dependence machinery is a separate later plan (see ``dev/plan-split-portfolio.md``
section 4). The ``Portfolio`` methods (``sample``, ``create_from_sample``,
``add_exa_sample``, ``make_comonotonic_allocations``) call into these helpers.
"""

import logging

import numpy as np
import pandas as pd

from .moments import xsden_to_mwrangler
from .utilities import ft, ift
from .iman_conover import iman_conover
import aggregate.random_agg as ar

logger = logging.getLogger(__name__)

# Optional numba acceleration for ``make_comonotonic_allocations_work``.
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

__all__ = ['make_comonotonic_allocations', 'swap_density_df']


@njit
def make_comonotonic_allocations_work(s_grid: np.ndarray, pdf_s: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    Computes a comonotonic convex-order improvement for an allocation matrix.

    Implements the algorithmic convex-order improvement from Theorem 3.1 in
    Denuit et. al.
    Uses a majorization approach based on Lorentz and Shimogaki (1968)
    to flatten monotonicity violations and redistribute mass .

    Reference
    ---------

    Denuit, Michel, et al. "Comonotonicity and Pareto optimality, with application
    to collaborative insurance." Insurance: Mathematics and Economics 120 (2025): 1-16.

    Parameters
    ----------
    s_grid : np.ndarray
        1D array of length M representing the discretized aggregate sum $S$.
    pdf_s : np.ndarray
        1D array of length M containing the probability mass function of $S$.
    kappa : np.ndarray
        2D array of shape (N, M) where N is the number of individual risks and M
        is the length of s_grid. Represents the initial Conditional Mean
        Risk-Sharing (CMRS) allocations $X_i^0 = \\mathsf{E}[X_i | S]$.

    Returns
    -------
    np.ndarray
        2D array of shape (N, M) containing the comonotonic allocations $\\tilde{f}_i(S)$.
    """
    n, m = kappa.shape
    kappa_tilde = np.copy(kappa)

    # Sweep forward through the aggregate states S
    for k in range(1, m):
        # Calculate local slopes to check for monotonicity
        diffs = kappa_tilde[:, k] - kappa_tilde[:, k-1]

        # Identify components where the allocation decreases as S increases
        violators = np.where(diffs < 0)[0]

        if len(violators) > 0:
            non_violators = np.where(diffs >= 0)[0]

            for i in violators:
                p = k - 1
                mass = pdf_s[k]
                weighted_sum = kappa_tilde[i, k] * mass

                # Scan backward to find the pooling index p that restores monotonicity
                # by creating an integral average (lambda_val) that bounds the previous steps
                while p >= 0 and kappa_tilde[i, p] > (weighted_sum / mass if mass > 0 else kappa_tilde[i, k]):
                    weighted_sum += kappa_tilde[i, p] * pdf_s[p]
                    mass += pdf_s[p]
                    p -= 1

                p += 1

                if mass > 0:
                    lambda_val = weighted_sum / mass
                else:
                    lambda_val = kappa_tilde[i, k]

                # delta represents the mass removed from the violator to flatten it
                delta = kappa_tilde[i, p:k + 1] - lambda_val
                kappa_tilde[i, p:k + 1] = lambda_val

                if len(non_violators) > 0:
                    slopes = diffs[non_violators]
                    sum_slopes = np.sum(slopes)

                    # Compute redistribution weights proportional to positive slopes
                    # to prevent non-violators from breaking monotonicity
                    if sum_slopes > 0:
                        alpha = slopes / sum_slopes
                    else:
                        alpha = np.ones(len(non_violators)) / len(non_violators)

                    # Redistribute the removed mass to the non-violating components
                    for idx, j in enumerate(non_violators):
                        kappa_tilde[j, p:k + 1] += delta * alpha[idx]

    return kappa_tilde


# Public alias for module-level callers (``Portfolio.make_comonotonic_allocations``
# is the method counterpart and reuses the underscored ``_work`` name internally
# to avoid clashing).
make_comonotonic_allocations = make_comonotonic_allocations_work


def swap_density_df(port, new_df, padding=1):
    """Swap a Portfolio's ``density_df`` for one with new marginal densities.

    Recombine the per-unit densities (``p_{unit}`` columns) by FFT,
    recompute the ``add_exa`` derivatives, and refresh the empirical
    rows of ``port.stats_df`` via :func:`xsden_to_mwrangler`. The
    swapped object has **no** ``mixed`` / ``independent`` columns —
    those describe a frequency-times-severity decomposition that the
    new densities do not carry — so those entries are blanked.

    Intended use: you have marginal numerical distributions (sample-
    derived, scenario-modelled, or hand-built) and want the Portfolio
    surface (sums, ``exa``, ``exeqa``, distortion pricing) without
    going through the FFT-from-spec path. Typical recipe::

        port0 = build('port Tmpl agg A dfreq[1] dsev[1] agg B dfreq[1] dsev[1]')
        new_df = ...  # frame with index = loss grid and columns p_A, p_B
        swap_density_df(port0, new_df)
        port0.apply_distortion(d)

    Parameters
    ----------
    port : Portfolio
        Target object. ``port.agg_list`` and ``port.unit_names`` set
        which ``p_{unit}`` columns are read from ``new_df``.
    new_df : pandas.DataFrame
        Indexed by the loss grid; carries ``loss`` and one ``p_{unit}``
        column per unit. ``p_total`` is recomputed by FFT convolution.
    padding : int
        FFT padding for the recombination (default 1).
    """
    port.density_df = new_df
    port.log2 = int(np.log2(len(new_df)))
    port.padding = padding
    port.bs = float(new_df['loss'].iloc[1] - new_df['loss'].iloc[0])

    # Recombine via FFT and build the per-unit state ``add_exa`` needs:
    # the user-supplied ``p_{unit}`` columns ARE the native unit pmfs
    # here, on the frame's own (zero-origin) grid.
    xs = port.density_df['loss'].to_numpy()
    ft_all = None
    unit_state = {}
    for agg in port.agg_list:
        raw_nm = agg.name
        p_unit = port.density_df[f'p_{raw_nm}'].to_numpy()
        ft_p = ft(p_unit, padding)
        unit_state[raw_nm] = dict(xs=xs, p=p_unit, ft_p=ft_p)
        if ft_all is None:
            ft_all = np.copy(ft_p)
        else:
            ft_all *= ft_p
    port.density_df['p_total'] = np.real(ift(ft_all, padding))

    port.density_df = port.add_exa(port.density_df, unit_state)
    port._augmented_dfs.clear()

    # Refresh empirical rows of stats_df from the swapped densities.
    # No mixed/independent decomposition exists here — leave those
    # columns at NaN. xsden_to_mwrangler tolerates a deficit if the
    # marginals don't sum to 1.
    xs = port.density_df['loss'].to_numpy()
    for unit in port.unit_names:
        mw = xsden_to_mwrangler(xs, port.density_df[f'p_{unit}'].to_numpy())
        ex1, ex2, ex3 = mw.noncentral
        m, cv, skew = mw.mcvsk
        for measure, value in [('ex1', ex1), ('ex2', ex2), ('ex3', ex3),
                               ('mean', m), ('cv', cv), ('skew', skew)]:
            port.stats_df.loc[('agg', measure), unit] = value
            port.stats_df.loc[('agg', measure), 'empirical'] = np.nan  # no longer valid
    mw_total = xsden_to_mwrangler(xs, port.density_df['p_total'].to_numpy())
    ex1, ex2, ex3 = mw_total.noncentral
    m, cv, skew = mw_total.mcvsk
    for measure, value in [('ex1', ex1), ('ex2', ex2), ('ex3', ex3),
                           ('mean', m), ('cv', cv), ('skew', skew)]:
        port.stats_df.loc[('agg', measure), 'empirical'] = value
        port.stats_df.loc[('agg', measure), 'total'] = value


def sample(port, n, replace=True, desired_correlation=None, keep_total=True):
    """
    Pull multivariate sample. Apply Iman Conover to induce correlation if required.

    """
    df = pd.DataFrame(index=range(n))
    for c in port.unit_names:
        # native unit pmf via the accessor (the p_{unit} columns left
        # density_df at numerics-2); same draw mechanics as before.
        pc = f'p_{c}'
        bit = port.unit_density(c).reset_index()
        df[c] = bit[['loss', pc]].\
                query(f'`{pc}` > 0').\
                sample(n, replace=replace, weights=pc, ignore_index=True, random_state=ar.RANDOM).\
                drop(columns=pc)

    if desired_correlation is not None:
        df = iman_conover(df, desired_correlation)
    else:
        df['total'] = df.sum(axis=1)
        df = df.set_index('total', drop=not keep_total)
    df = df.reset_index(drop=True)
    return df


def add_exa_sample(port, sample, S_calculation='forwards'):
    """
    Computes a version of density_df using sample to compute E[Xi | X].
    Then fill in the other ex.... variables using code from
    Portfolio.add_exa, stripped down to essentials.

    If no p_total is given then samples are assumed equally likely.
    total is added if not given (sum across rows)
    total is then aligned to the bucket size port.bs using (total/bs).round(0)*bs.
    The other loss columns are then scaled so they sum to the adjusted total

    Next, group by total, sum p_total and average the units to create E[Xi|X]

    This sample is merged into a stripped down density_df. Then
    the other ex... columns are added. Excludes eta mu columns.

    Anticipated use: replace density_df with this, invalidate quantile
    function and then compute various allocation metrics.

    The index on the input sample is ignored.

    Formally ``extensions.samples.add_exa_sample``.

    """

    # starter information
    # cut_eps = np.finfo(float).eps
    bs = port.bs

    # working copy
    sample_in = sample.copy()

    if 'total' not in sample:
        # p_total may be in sample
        cols = list(sample.columns)
        if 'p_total' in sample:
            cols.remove('p_total')
        sample_in['total'] = sample_in[cols].sum(axis=1)
    # index may be called total; that causes confusion; throw away input index
    sample_in = sample_in.reset_index(drop=True)

    # want to align the index to that of port.density_df; all multiples of port.bs
    # at the same time, want to scale all elements
    # temp0 gives the multiples of bs for the index; temp is the scaling for
    # all the other columns; temp0 will all be exact
    temp0 = (sample_in.total / bs).round(0) * bs
    temp = (temp0 / sample_in.total).to_numpy().reshape((len(sample_in), 1))
    # re-scale loss samples so they sum to total, need to extract p_total first
    if 'p_total' not in sample_in:
        # equally likely probs
        logger.info('Adding p_total to sample_in')
        # logger.info('Adding p_total to sample_in')
        p_total = 1.0 / len(sample_in)
    else:
        # use input probs
        p_total = sample_in['p_total']

    # re-scale
    sample_in = sample_in * temp
    # exact for total
    sample_in['total'] = temp0
    # and put probs back
    sample_in['p_total'] = p_total

    # Group by X values, aggregate probs and compute E[Xi  | X]
    exeqa_sample = sample_in.groupby(by='total').agg(
        **{f'exeqa_{i}': (i, np.mean) for i in port.unit_names})
    # need to do this after rescaling to get correct (rounded) total values
    probs = sample_in.groupby(by='total').p_total.sum()
    # want all probs to be positive
    probs = np.maximum(0, probs.fillna(0.0))

    # working copy of port's density_df with relevant columns, plus the
    # unit pmfs scattered back onto the total grid for the stand-alone
    # lev calc below. Sample grids are zero-origin, so the aligned view
    # is exact; sampling semantics on a windowed/signed book are
    # deferred to the sampling redesign plan (numerics-2 deliverable 4
    # keeps this path mechanically working only).
    df = port.density_df.filter(
        regex=f'^(loss|e_({port.unit_name_pipe})|(e|p)_total)$').copy()
    df = df.join(port.aligned_unit_density_df(grid='total',
                                              allow_window_mismatch=True))

    # want every value in sample_in.total to be in the index of df
    # this code verifies that has occurred
    # for t in sample_in.total:
    #     try:
    #         df.index.get_loc(t)
    #     except KeyError:
    #         print(f'key error for t={t}')
    #
    # or, if you prefer,
    #
    # test = df[['loss', 'p_total']].merge(sample_in, left_index=True, right_on='total', how='outer', indicator=True)
    # test.groupby('_merge')[['loss']].count()
    #
    # shows nothing right_only.

    # fix p_total and hence S and F
    # fill in these values (note, all this is to get an answer the same
    # shape as df, so it can be swapped in)
    df['p_total'] = probs
    df['p_total'] = df['p_total'].fillna(0.)

    # macro, F, S
    df['F'] = df.p_total.cumsum() # np.cumsum(df.p_total)

    if S_calculation == 'forwards':
        df['S'] = 1 - df.F
    else:
        # add_exa method; you'd think the fill value should be 0, which
        # will be the case when df.p_total sums to 1 (or more)
        df['S'] =  \
            df.p_total.shift(-1, fill_value=min(df.p_total.iloc[-1],
                                                max(0, 1. - (df.p_total.sum()))))[::-1].cumsum()[::-1]

    # this avoids irritations later on
    df.F = np.minimum(df.F, 1)
    df.S = np.minimum(df.S, 1)
    # where is S=0
    Seq0 = (df.S == 0)

    # invalidate quantile functions
    port._dist = None

    # E[X_i | X=a], E(xi eq a)
    # all in one go (outside loop)
    df = pd.merge(df,
                  exeqa_sample,
                  how='left',
                  left_on='loss',
                  right_on='total').fillna(0.0).set_index('loss', drop=False)
    # check exeqa sums to correct total. note this only happens ae, ie when
    # p_total > 0
    assert np.allclose(df.query('p_total > 0').loss,
                       df.query('p_total > 0')[[f'exeqa_{i}' for i in port.unit_names]].sum(axis=1))

    assert df.index.is_unique
    df['exeqa_total'] = df.loss

    # add additional variables via loop
    for col in port.unit_names_ex:
        # ### Additional Variables
        # * exeqa_line = $E(X_i \mid X=a)$
        # * exlea_line = $E(X_i \mid X\le a)$
        # * e_line = $E(X_i)$
        # * exgta_line = $E(X_i \mid X \ge a)$
        # * exi_x_line = $E(X_i / X \mid X = a)$
        # * and similar for le and gt a
        # * exa_line = $E(X_i(a))$
        # * Price based on same constant ROE formula (later we will do $g$s)

        # need the stand alone LEV calc
        # E(min(Xi, a)
        # needs to be shifted down by one for the partial integrals....
        # stemp = 1 - df['p_' + col].cumsum()
        stemp = df['p_' + col].shift(-1, fill_value=min(df['p_' + col].iloc[-1],
                                                        max(0, 1. - (df['p_' + col].sum()))))[::-1].cumsum()[::-1]
        df['lev_' + col] = stemp.shift(1, fill_value=0).cumsum() * port.bs

        # E[X_i | X<= a] temp is used in le and gt calcs
        temp = np.cumsum(df['exeqa_' + col] * df.p_total)
        df['exlea_' + col] = temp / df.F

        # E[X_i | X>a]
        df['exgta_' + col] = (df['e_' + col] - temp) / df.S

        # E[X_i / X | X > a]; guard loss[0]=0 with a patched copy.
        denom = df['loss'].copy()
        denom.iat[0] = 1.0

        df['exi_x_' + col] = np.sum(df['exeqa_' + col] * df.p_total / denom)
        temp_xi_x = np.cumsum(df['exeqa_' + col] * df.p_total / denom)
        df['exi_xlea_' + col] = temp_xi_x / df.F
        df.loc[0, 'exi_xlea_' + col] = 0  # selection, 0/0 problem


        fill_value = np.nan

        assert df.index.is_unique, "Index is not unique!"

        df['exi_xgta_' + col] = ((df[f'exeqa_{col}'] / df.loss *
                                  df.p_total).shift(-1, fill_value=fill_value)[
                                 ::-1].cumsum()) / df.S
        # need this NOT to be nan otherwise exa won't come out correctly
        df.loc[Seq0, 'exi_xgta_' + col] = 0.

        df['exi_xeqa_' + col] = df['exeqa_' + col] / df['loss']
        df.loc[0, 'exi_xeqa_' + col] = 0

        # need the loss cost with equal priority rule
        df[f'exa_{col}'] = (df.S * df['exi_xgta_' + col]).shift(1,
                                                                fill_value=0).cumsum() * port.bs

    # put in totals for the ratios... this is very handy in later use
    for metric in ['exi_xlea_', 'exi_xgta_', 'exi_xeqa_']:
        df[metric + 'sum'] = df.filter(regex=metric).sum(axis=1)

    df = df.set_index('loss', drop=False)
    df.index.name = None
    return df
