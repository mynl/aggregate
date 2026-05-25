"""Iman-Conover dependence imposition and related correlation utilities."""

from collections import namedtuple
from functools import lru_cache
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import multivariate_t

import aggregate.random_agg as ar


logger = logging.getLogger(__name__)


@lru_cache()
def ic_noise(n, d):
    """
    Implements steps 1, 2, 3, 4, 5, and 6
    This is bottleneck function, therefore cache it
    It handles the true-up of the random sample to ensure it is exactly independent
    :param n: row
    :param d: columns
    :return:
    """
    # step 1: make a reference n x d random uncorrelated normal sample
    p = [ss.norm.ppf( x / (n + 1)) for x in range(1, n+1)]
    # mean is zero...but belt and braces
    p = (p - np.mean(p)) / np.std(p)
    # space for answer
    score = np.zeros((n, d))
    # steps 2 and 3
    for j in range(0, score.shape[1]):
        # shuffle each column
        score[:, j] = ar.RANDOM.permutation(p)

    # actual correlation of reference (this will be close to, but not equal to, the identity)
    # @ denotes matrix multiplication
    # step 4 and 5
    E = np.linalg.cholesky((score.T @ score) / n)
    # sample with exact desired correlation
    # step 6
    return score @ np.linalg.inv(E.T)


@lru_cache()
def ic_t_noise(n, d, dof):
    """
    as above using multivariate t distribution noise
    """
    mvt = multivariate_t([0.]*d, 1, df=dof)
    score = mvt.rvs(n)

    # actual correlation of reference (this will be close to, but not equal to, the identity)
    # @ denotes matrix multiplication
    # step 4 and 5
    E = np.linalg.cholesky((score.T @ score) / n)
    # sample with exact desired correlation
    # step 6
    return score @ np.linalg.inv(E.T)


def ic_rank(N):
    """
    rankdata function: assign ranks to data, dealing with ties appropriately
    work by column
    N is a numpy array
    """
    rank = np.zeros((N.shape[0], N.shape[1]))
    for j in range(0, N.shape[1]):
        rank[:, j] = ss.rankdata(N[:, j], method='ordinal')
    return rank.astype(int) - 1


def ic_reorder(ranks, samples):
    """
    put samples into the order determined by ranks
    array is calibrated to the reference distribution
    space for the answer
    """
    rank_samples = np.zeros((samples.shape[0], samples.shape[1]))
    for j in range(0, samples.shape[1]):
        s = np.sort(samples[:, j])
        rank_samples[:, j] = s[ranks[:,j]]
    return rank_samples


def iman_conover(marginals, desired_correlation, dof=0, add_total=True):
    """
    Perform Iman Conover shuffling on input marginals to achieve desired_correlation
    Desired_correlation must be positive definite and of the correct size.
    The result has the same rank correlation as a reference sample with the
    desired linear correlation. Thus, the process relies on linear and rank
    correlation (for the reference and the input sample) being close.

    if dof==0 use normal scores; else you mv t

    Sample code:
    ::

        n = 100
        df = pd.DataFrame({ f'line_{i}': ss.lognorm(.1 + .2*np.random.rand(),
                        scale=10000).rvs(n) for i in range(3)})
        desired = np.matrix([[1, -.3, 0], [-.3, 1, .8], [0, .8, 1]])
        print(desired)
        # check it is a corr matrix
        np.linalg.cholesky(desired)

        df2 = iman_conover(df, desired)
        df2.corr()
        df_scatter(df2)


    Iman Conover Method

    **Make rank order the same as a reference sample with desired correlation structure.**

    Reference sample usually chosen as multivariate normal because it is easy and flexible, but you can use **any** reference, e.g. copula based.

    The old @Risk software used Iman Conover.

    Input: matrix $\\mathbf X$ of marginals and desired correlation matrix $\\mathbf S$

    1. Make one column of scores  $a_i=\\Phi^{-1}(i/(n+1))$ for $i=1,\\dots,n$ and rescale to have standard deviation one.
    1. Copy the scores $r$ times to make the score matrix $\\mathbf M$.
    1. Randomly permute the entries in each column of $\\mathbf M$.
    1. Compute the correlation matrix $n^{-1}\\mathbf M'\\mathbf M$ of the sample scores $\\mathbf M$.
    1. Compute the Choleski decomposition $n^{-1}\\mathbf M^t\\mathbf M=\\mathbf E\\mathbf E^t$ of the score correlation matrix.
    1. Compute $\\mathbf M' = \\mathbf M(\\mathbf E^t)^{-1}$, which is exactly uncorrelated.
    1. Compute the Choleski decomposition $\\mathbf S=\\mathbf C\\mathbf C^t$  of the  desired correlation matrix $\\mathbf S$.
    1. Compute $\\mathbf T=\\mathbf M'\\mathbf C^t$. The matrix $\\mathbf T$ has exactly the desired correlation structure
    1. Let $\\mathbf Y$ be the input matrix $\\mathbf X$ with each column reordered to have exactly the same **rank ordering** as the corresponding column of $\\mathbf T$.

    Relies on the fact that rank (Spearman) and linear (Pearson) correlation are approximately the same.

    """

    n, d = marginals.shape

    # "square root" of "variance"
    # step 7
    C = np.linalg.cholesky(desired_correlation)

    # make a perfectly uncorrelated reference: noise function = steps 1-6; product is step 8 (transposed)
    if dof == 0:
        N = ic_noise(n, d) @ C.T
    else:
        N = ic_t_noise(n, d, dof) @ C.T

    # required ordering of marginals determined by reference sample, step 9
    R = ic_rank(N)

    # re order
    if type(marginals) == np.ndarray:
        shuffled_marginals = ic_reorder(R, marginals)
        df = pd.DataFrame(shuffled_marginals)
    else:
        shuffled_marginals = ic_reorder(R, marginals.to_numpy())
        df = pd.DataFrame(shuffled_marginals, columns=marginals.columns)

    # add total if requested
    if add_total:
        df['total'] = df.sum(axis=1)
        df = df.set_index('total')
        df = df.sort_index(ascending=False)

    return df


def block_iman_conover(unit_losses, intra_unit_corrs, inter_unit_corr, as_frame=False):
    """
    Apply Iman Conover to the unit loss blocks in ``unit_losses`` with correlation matrices in ``intra``.

    Then determine the ordering for the unit totals with correlation ``inter``.

    Re-order each unit, row by row, so that the totals have the desired correlation structure, but
    leaving the intra unit correlation unchanged.

    ``unit_losses = [np.arrays or pd.Series]`` of losses by subunit within units, without totals

    ``len(unit_losses) == len(intra_unit corrs)``

    For simplicity all normal copula; can add other later if required.

    No totals input or output anywhere.

    ``if as_frame`` then a dataframe version returned, for auditing.

    Here is some tester code, using great.test_df to make random unit losses. Vary num_units and
    num_sims as required.

    ::

        def bic_tester(num_units=3, num_sims=10000):
            from aggregate import random_corr_matrix
            # from great import test_df

            # create samples
            R = range(num_units)
            unit_losses = [test_df(num_sims, 3 + i) for i in R]
            totals = [u.sum(1) for u in unit_losses]

            # manual dataframe to check against
            manual = pd.concat(unit_losses + totals, keys=[f'Unit_{i}' for i in R] + ['Total' for i in R], axis=1)

            # for input to method
            unit_losses = [i.to_numpy() for i in unit_losses]
            totals = [i.to_numpy() for i in totals]

            # make corrs
            intra_unit_corrs = [random_corr_matrix(i.shape[1], p=.5, positive=True) for i in unit_losses]
            inter_unit_corr = random_corr_matrix(len(totals), p=1, positive=True)

            # apply method
            bic = block_iman_conover(unit_losses, intra_unit_corrs, inter_unit_corr, True)

            # extract frame answer, put col names back
            bic.frame.columns = manual.columns
            dm = bic.frame

            # achieved corr
            for i, target in zip(dm.columns.levels[0], intra_unit_corrs + [inter_unit_corr]):
                print(i)
                print((dm[i].corr() - target).abs().max().max())
                # print(dm[i].corr() - target)

            # total corr across subunits
            display(dm.drop(columns=['Total']).corr())

            # total corr across subunits
            display(dm.drop(columns=['Total']).corr())

            return manual, bic, intra_unit_corrs, inter_unit_corr

        manual, bic, intra, inter = bic_tester(3, 10000)

    """

    if isinstance(unit_losses, dict):
        unit_losses = unit_losses.values()

    if isinstance(intra_unit_corrs, dict):
        intra_unit_corrs = intra_unit_corrs.values()

    # shuffle unit losses
    # IC returns a dataframe
    unit_losses = [iman_conover(l, c, dof=0, add_total=False).to_numpy() for l, c in zip(unit_losses, intra_unit_corrs)]

    # extract totals
    totals = [l.sum(1) for l in unit_losses]
    totals = np.vstack(totals)

    # apply the interunit correlation to totals: this code copies iman_conover because we want
    # to keep the same ordering matrices

    # block shuffle units; this code is IC by hand to keep track of R and apply to the units
    d, n = totals.shape

    # "square root" of "variance"
    # step 7
    C = np.linalg.cholesky(inter_unit_corr)

    # make a perfectly uncorrelated reference: noise function = steps 1-6; product is step 8 (transposed)
    N = ic_noise(n, d) @ C.T

    # required ordering of marginals determined by reference sample, step 9
    R = ic_rank(N)

    # re-order totals and the corresponding unit losses
    for i, (u, t) in enumerate(zip(unit_losses, totals)):
        r = np.argsort(t)
        unit_losses[i] = u[r]
        totals[i] = t[r]

    # put into the desired IC ordering,
    for i, (u, t, r) in enumerate(zip(unit_losses, totals, R.T)):
        totals[i] = t[r]
        unit_losses[i] = u[r]

    # assembled sample
    combined = np.hstack(unit_losses)

    if as_frame:
        fr = pd.concat((pd.DataFrame(combined), pd.DataFrame(totals.T)), axis=1, keys=['units', 'totals'])

    else:
        fr = None

    BlockImanConover = namedtuple('BlockImanConover', 'totals combined frame')
    ans = BlockImanConover(totals, combined, fr)

    return ans


def rearrangement_algorithm_max_VaR(df, p=0, tau=1e-3, max_n_iter=100):
    """
    Implementation of the Rearragement Algorithm (RA). Determines the worst p-VaR
    rearrangement of the input variables.

    For loss random variables p is usually close to 1.

    Embrechts, Paul, Giovanni Puccetti, and Ludger Ruschendorf, 2013, *Model uncertainty and
    VaR aggregation*, Journal of Banking and Finance 37, 2750-2764.

    **Worst-Case VaR**

    Worst value at risk arrangement of marginals.

    See `Actuarial Review article <https://ar.casact.org/the-re-arrangement-algorithm>`_.

    Worst TVaR / Variance arrangement of bivariate data = pair best with worst, second best with
    second worst, etc., called **countermonotonic** arangement.

    More than 2 marginals: can't *make everything negatively correlated with
    everything else*. If :math:`X` and :math:`Y` are negatively correlated
    and :math:`Y` and :math:`Z` are negatively correlated then :math:`X` and
    :math:`Z` will be positively correlated.

    Next best attempt: make :math:`X` countermonotonic to :math:`Y+Z`,
    :math:`Y` to :math:`X+Z` and :math:`Z` to :math:`X+Y`. Basis of
    **rearrangement algorithm**.

    *The Rearrangement Algorithm*

    1. Randomly permute each column of :math:`X`, the :math:`N\\times d`
       matrix of top :math:`1-p` observations
    2. Loop

       -  Create a new matrix :math:`Y` as follows. For column
          :math:`j=1,\\dots,d`

          -  Create a temporary matrix :math:`V_j` by deleting the
             :math:`j` th column of :math:`X`
          -  Create a column vector :math:`v` whose :math:`i` th element
             equals the sum of the elements in the :math:`i` th row of
             :math:`V_j`
          -  Set the :math:`j` th column of :math:`Y` equal to the
             :math:`j` th column of :math:`X` arranged to have the opposite
             order to :math:`v`, i.e. the largest element in the
             :math:`j` th column of :math:`X` is placed in the row of
             :math:`Y` corresponding to the smallest element in :math:`v`,
             the second largest with second smallest, etc.

       -  Compute :math:`y`, the :math:`N\\times 1` vector with
          :math:`i` th element equal to the sum of the elements in the
          :math:`i` th row of :math:`Y` and let :math:`y^*=\\min(y)` be the
          smallest element of :math:`y` and compute :math:`x^*` from
          :math:`X` similarly
       -  If :math:`y^*-x^* \\ge \\epsilon` then set :math:`X=Y` and repeat
          the loop
       -  If :math:`y^*-x^* < \\epsilon` then break from the loop

    3. The arrangement :math:`Y` is an approximation to the worst
       :math:`\\text{VaR}_p` arrangement of :math:`X`.

    :param df: Input DataFrame containing samples from each marginal. RA will only combine the
        top 1-p proportion of values from each marginal.
    :param p: If ``p==0`` assume df has already truncated to the top p values (for each marginal).
        Otherwise truncate each at the ``int(1-p * len(df))``
    :param tau: simulation tolerance
    :param max_iter: maximum number of iterations to attempt
    :return: the top 1-p values of the rearranged DataFrame
    """

    sorted_marginals = {}

    # worst N shuffled
    if p:
        N = int(np.round((1 - p) * len(df), 0))
    else:
        N = len(df)
    # container for answer
    df_out = pd.DataFrame(columns=df.columns, dtype=float)

    # iterate over each column, sort, truncate (if p>0)
    for m in df:
        sorted_marginals[m] = df[m].sort_values(ascending=False).reset_index(drop=True).iloc[:N]
        df_out[m] = ar.RANDOM.permutation(sorted_marginals[m])

    # change in VaR and last VaR computed, to control looping
    chg_var = max(100, 2 * tau)
    last_var = 2 * chg_var
    # iteration counter for reporting
    n_iter = 0
    while abs(chg_var) > tau:
        for m in df_out:
            # sum all the other columns
            E = df_out.loc[:, df_out.columns != m].sum(axis=1)
            # ranks of sums
            rks = np.array(E.rank(method='first') - 1, dtype=int)
            # make current column counter-monotonic to sum (sorted marginals are in descedending order)
            df_out[m] = sorted_marginals[m].loc[rks].values
        # achieved VaR is minimum value
        v = df_out.sum(axis=1).sort_values(ascending=False).iloc[-1]
        chg_var = last_var - v
        last_var = v
        # reporting and loop control
        n_iter += 1
        if n_iter >= 2:
            logger.info(f'Iteration {n_iter:d}\t{v:5.3e}\tChg\t{chg_var:5.3e}')
        if n_iter > max_n_iter:
            logger.error("ERROR: not converging...breaking")
            break

    df_out['total'] = df_out.sum(axis=1)
    logger.info(f'Ending VaR\t{v:7.5e}\ns lower {df_out.total.min():7.5e}')
    return df_out.sort_values('total')


def make_corr_matrix(vine_spec):
    r"""
    Make a correlation matrix from a vine specification, https://en.wikipedia.org/wiki/Vine_copula.

    A vine spececification is::

        row 0: correl of X0...Xn-1 with X0
        row 1: correl of X1....Xn-1 with X1 given X0
        row 2: correl of X2....Xn-1 with X2 given X0, X1
        etc.

    For example ::

        vs = np.array([[1,.2,.2,.2,.2],
                       [0,1,.3,.3,.3],
                       [0,0,1,.4, .4],
                       [0,0,0,1,.5],
                       [0,0,0,0,1]])
        make_corr_matrix(vs)

    Key fact is the partial correlation forumula

    .. math::

        \rho(X,Y|Z) = \frac{(\rho(X,Y) - \rho(X,Z)\rho(Y,Z))}{\sqrt{(1-\rho(X,Z)^2)(1-\rho(Y,Z)^2)}}

    and therefore

    .. math::

        \rho(X,Y) =  \rho(X,Z)\rho(Y,Z) + \rho(X,Y|Z) \sqrt((1-\rho(XZ)^2)(1-\rho(YZ)^2))

    see https://en.wikipedia.org/wiki/Partial_correlation#Using_recursive_formula.

    """

    A = np.matrix(vine_spec)
    n, m = A.shape
    assert n==m

    for i in range(n - 2, 0, -1):
        for j in range(i + 1, n):
            for k in range(1, i+1):
                # recursive formula
                A[i, j] = A[i - k, i] * A[i - k, j] + A[i, j] * np.sqrt((1 - A[i - k, i] ** 2) * (1 - A[i - k, j] ** 2))

    # fill in (unnecessary but simpler)
    for i in range(n):
        for j in range(i + 1, n):
            A[j, i] = A[i, j]

    return A


def random_corr_matrix(n, p=1, positive=False):
    """
    make a random correlation matrix

    smaller p results in more extreme correlation
    0 < p <= 1

    Eg ::

        rcm = random_corr_matrix(5, .8)
        rcm
        np.linalg.cholesky(rcm)


    positive=True for all entries to be positive

    """

    if positive is True:
        A = ar.RANDOM.random((n, n))**p
    else:
        A = 1 - 2 * ar.RANDOM.random((n, n))**p
    np.fill_diagonal(A, 1)

    return make_corr_matrix(A)
