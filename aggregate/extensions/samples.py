# code for using a sample, adjusting kappa function
# see UsingSamples.ipynb for runners
# (c) Stephen Mildenhall 2022

import aggregate as agg
from aggregate import build, Bounds, iman_conover, mu_sigma_from_mean_cv, random_corr_matrix
from collections import namedtuple
from functools import lru_cache
import logging
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.sparse import coo_matrix
import scipy.stats as ss


logger = logging.getLogger(__name__)

def qdp(df):
    """
    quick describe with nice percentiles and cv
    """
    d = df.describe()
    d.loc['cv'] = d.loc['std'] / d.loc['mean']
    return d


def make_test_sample(n, means, cvs, desired_correlation=None):
    """
    make a test DataFrame with sample lognormal marginals, given means and cvs

    """

    mus, sigmas = mu_sigma_from_mean_cv(means, cvs)
    c = len(mus)
    names = [chr(65+i) for i in range(c)]
    df = pd.DataFrame(
                     {i: ss.lognorm(s,
                         scale=m*np.exp(-s*s/2)).rvs(n) for m, s, i in
                         zip(means, sigmas, names)
                     })
    if desired_correlation is not None:
        df = iman_conover(df, desired_correlation)
    return df


def portfolio_from_sample(n, means, cvs, rcm_p=1, positive=True, plot=True, log2=16, bs=1, **kwargs):
    """
    Create a sample and portfolio from the sample.
    Apply IC correlation to the sample.

    n = number of rows in sample
    rcm_p controls average magnitude of correlation
    positive = True for just positive correlation
    kwargs = passed through to update (bs=, log2=, remove_fuzz=)
    """
    c = len(means)
    if rcm_p is None:
        rcm = None
    elif isinstance(rcm_p, np.ndarray):
        rcm = rcm_p
    else:
        rcm = random_corr_matrix(c, rcm_p, positive)
    sample = make_test_sample(n, means, cvs, rcm)
    # display(sample.corr())
    if plot is True:
        scatter_matrix(sample, marker='.', s=5, alpha=1, figsize=(10, 10), diagonal='kde');

    # add total etc.
    sample = sample.sort_index()
    sample = sample.reset_index(drop=False)

    # make the Portfolio object
    port = agg.Portfolio('test_from_df', sample)
    port.update(log2=log2, bs=bs, **kwargs)

    df_exa = add_exa_sample(port, sample)
    SampleResult = namedtuple('SampleResult', ['sample', 'port', 'df_exa', 'rcm'])
    ans = SampleResult(sample, port, df_exa, rcm)
    return ans


def add_exa_sample(self, sample):
    """
    Computes a version of density_df using sample to compute E[Xi | X].
    Then fill in the other ex.... variables using code from
    Portfolio.add_exa, stripped down to essentials.

    If no p_total is given then samples are assumed equally likely.
    total is added if not given (sum across rows)
    total is then aligned to the bucket size self.bs using (total/bs).round(0)*bs.
    The other loss columns are then scaled so they sum to the adjusted total

    Next, group by total, sum p_total and average the lines to create E[Xi|X]

    This sample is merged into a stripped down density_df. Then
    the other ex... columns are added. Excludes eta mu columns.

    Anticipated use: replace density_df with this, invalidate quantile
    function and then compute various allocation metrics.

    The index on sample is ignored.

    """

    # starter information
    cut_eps = np.finfo(np.float).eps
    bs = self.bs

    # working copy
    sample_in = sample.copy()

    if 'total' not in sample:
        # p_total may be in sample
        cols = list(sample.columns)
        if 'p_total' in sample:
            cols.remove('p_total')
        sample_in['total'] = sample_in[cols].sum(axis=1)
    # index may be called total; that causes confusion; throw away input index
    sample_in = sample_in.reset_index(drop=True) # .index.name = None

    # want to align the index to that of self.density_df; all multiples of self.bs
    # at the same time, want to scale all elements
    # temp0 gives the multiples of bs for the index; temp is the scaling for
    # all the other columns; temp0 will all be exact
    temp0 = (sample_in.total / bs).round(0) * bs
    temp = (temp0 / sample_in.total).to_numpy().reshape((len(sample_in),1))
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
        **{f'exeqa_{i}': (i, np.mean) for i in self.line_names})
    # need to do this after rescaling to get correct (rounded) total values
    probs = sample_in.groupby(by='total').p_total.sum()
    # want all probs to be positive
    probs = np.maximum(0, probs.fillna(0.0))

    # working copy of self's density_df with relevant columns
    df = self.density_df.filter(
        regex=f'^(loss|(p|e)_({self.line_name_pipe})|(e|p)_total)$').copy()

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
    df['F'] = np.cumsum(df.p_total)
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

    ## TODO fix the means and other stats here?
    ## invalidate quantile functions
    self._linear_quantile_function = None

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
        df.query('p_total > 0')[[f'exeqa_{i}' for i in self.line_names]].sum(axis=1))
    # display(df.query('p_total > 0').head().T)

    # in another situation; here we are before exlea has been added
    # deal with the problem of conditioning on very small probabilities in the left tail
    # loss_max = df[['loss', 'exlea_total']].query('exlea_total > loss').loss.max()
    # if np.isnan(loss_max):
    #     loss_max = 0
    # else:
    #     # was mult * bs
    #     loss_max += bs
    #     logger.warning(f'Small probability fix: loss_max > 0, {loss_max}')
    # # try nan in place of 0             V
    # df.loc[0:loss_max, 'exlea_total'] = np.nan

    assert df.index.is_unique
    df['exeqa_total'] = df.loss

    # add additional variables via loop
    for col in self.line_names_ex:
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
        df['lev_' + col] = stemp.shift(1, fill_value=0).cumsum() * self.bs

        # E[X_i | X<= a] temp is used in le and gt calcs
        temp = np.cumsum(df['exeqa_' + col] * df.p_total)
        df['exlea_' + col] = temp / df.F

        # E[X_i | X>a]
        df['exgta_' + col] = (df['e_' + col] - temp) / df.S

        # E[X_i / X | X > a]  (note=a is trivial!)
        temp = df.loss.iloc[0]  # loss=0, should always be zero
        df.loss.iloc[0] = 1  # avoid divide by zero
        # unconditional E[X_i/X]
        df['exi_x_' + col] = np.sum(
            df['exeqa_' + col] * df.p_total / df.loss)
        temp_xi_x = np.cumsum(df['exeqa_' + col] * df.p_total / df.loss)
        df['exi_xlea_' + col] = temp_xi_x / df.F
        df.loc[0, 'exi_xlea_' + col] = 0  # selection, 0/0 problem
        # more generally F=0 error:                      V
        # df.loc[df.exlea_total == 0, 'exi_xlea_' + col] = 0
        # ?? not an issue for samples; don't have exlea_total anyway??
        # put value back
        df.loss.iloc[0] = temp

        fill_value = np.nan

        assert df.index.is_unique

        df['exi_xgta_' + col] = ((df[f'exeqa_{col}'] / df.loss *
                                  df.p_total).shift(-1, fill_value=fill_value)[
                                 ::-1].cumsum()) / df.S
        # need this NOT to be nan otherwise exa won't come out correctly
        df.loc[Seq0, 'exi_xgta_' + col] = 0.

        df['exi_xeqa_' + col] = df['exeqa_' + col] / df['loss']
        df.loc[0, 'exi_xeqa_' + col] = 0

        # need the loss cost with equal priority rule
        df[f'exa_{col}'] = (df.S * df['exi_xgta_' + col]).shift(1,
            fill_value=0).cumsum() * self.bs

    # put in totals for the ratios... this is very handy in later use
    for metric in ['exi_xlea_', 'exi_xgta_', 'exi_xeqa_']:
        df[metric + 'sum'] = df.filter(regex=metric).sum(axis=1)

    df = df.set_index('loss', drop=False)
    df.index.name = None
    return df


def create_bounds(port, premium, a=0, n_tps=129, s=64):
    """
    THIS FUNCTION IS NOT NEEDED... use add_exa_sample bd = Bounds(df_exa)

    Summarize total and line's E[Xi|X] into data frame
    Set up bounds object

    line = line to which you want allocations...  [eventually, want to do them all...]

    Generic version to work with Portfolio object

    TODO: add asset limit?

    """

    raise ValueError('Routine not needed...why are you calling?')

    # :-6 takes out |total
    tot = port.density_df.filter(regex=f'loss|p_total|exeqa_({port.line_name_pipe})').copy()
    tot = tot.set_index('loss', drop=True)
    tot = tot.rename(columns={ i: i[6:] for i in tot.filter(regex='exeqa_')})

    # p_total names the probabilities here
    tot = tot.groupby('total').agg(
        {**{'p_total': lambda x: np.sum(x)},
        **{l: np.mean for l in port.line_names_ex}}
        )

    # this is unnecessary for Agg generated...
    ptot = pd.DataFrame(index=port.density_df.index)
    ptot.index.name = 'total'
    ptot = ptot.merge(tot, how='left', left_index=True, right_index=True)
    ptot = ptot.fillna(0.)

    bd = Bounds(ptot)
    if a==0:
        a = tot.index[-1]
    bd.tvar_cloud('total', premium=premium, a=a, n_tps=n_tps, s=s)
    p_star = bd.p_star('total', premium)

    return bd, ptot, p_star



def allocation_ranges(self, df_exa, a=np.inf):
    """
    price total and alloca te to other columns using to assets a
    across all calibrated biTVaRs

    TODO: fix when a < inf... need extra part of integral

    self = Bounds object

    df_exa = previously created index=loss X; p_total; E[Xi|X] X...

    :param a:
    :return:
    """

    # compute S
    # S = df_exa.query('p_total > 0').p_total.shift(-1, fill_value=0)[::-1].cumsum()[::-1]
    s = df_exa.S.to_numpy()

    hinges = coo_matrix(np.nan_to_num(np.minimum(1.0, s.reshape(1, len(s)) / (1.0 - self.tps.reshape(len(self.tps), 1))), nan=0.0))

    ml = coo_matrix((1 - self.weight_df.weight, (np.arange(len(self.weight_df)), self.idx[0])),
                    shape=(len(self.weight_df), len(self.tps)))
    mr = coo_matrix((self.weight_df.weight, (np.arange(len(self.weight_df)), self.idx[1])),
                    shape=(len(self.weight_df), len(self.tps)))
    m = ml + mr

    # print(f'm shape = {m.shape}, hinges shape = {hinges.shape}, types {type(m)}, {type(hinges)}')

    gS_cloud = (m @ hinges).toarray()

    # compute gp
    gps = -np.diff(gS_cloud, axis=1, prepend=1)

    # sum products for allocations
    bs = df_exa.index[1]
    bit = df_exa .filter(regex='exeqa_[A-Z]')
    # compute the allocations
    allocs = pd.DataFrame(
            gps @ (bit.to_numpy() * bs),
            columns=[i.replace('exeqa_', 'alloc_') for i in bit.columns],
            index=self.weight_df.index)

    return allocs




def plot_max_min(self, ax):
    ax.fill_between(self.cloud_df.index, self.cloud_df.min(1), self.cloud_df.max(1), facecolor='C7', alpha=.15)
    self.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=0.5, ls='-', c='w')
    self.cloud_df.max(1).plot(ax=ax, label="_nolegend_", lw=0.5, ls='-', c='w')

