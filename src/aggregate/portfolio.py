from collections.abc import Iterable
from copy import deepcopy
import json
import logging
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, StrMethodFormatter, MaxNLocator,
                               FixedLocator, FixedFormatter, AutoMinorLocator)
import numpy as np
import pandas as pd
from pandas.io.formats.format import EngFormatter
from pandas.plotting import scatter_matrix
from pathlib import Path
import re
from scipy import interpolate
from scipy.optimize import bisect
from scipy.spatial import ConvexHull
from textwrap import fill
import warnings
from IPython.display import HTML, display

from .constants import *
from .distributions import Aggregate, Severity, _flat_col_to_stats_index, approximate_from_mcvsk
from .results import (AnalyzeDistortionResult, AnalyzeDistortionsResult,
                      PricingBoundsResult, PricingResult)
from .spectral import Distortion, DISTORTION_DTYPE
from .moments import MomentAggregator, MomentWrangler
from .iman_conover import iman_conover
from .utilities import (ft, ift, decl_pprint,
                        subsets, round_bucket,
                        make_var_tvar, agg_help, explain_validation)
import aggregate.random_agg as ar

# Optional numba acceleration for ``make_comonotonic_allocations_work``.
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func


# fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# matplotlib.rcParams['legend.fontsize'] = 'xx-small'
logger = logging.getLogger(__name__)


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


# Canonical column order for pricing exhibits. Used as a pandas
# ``CategoricalDtype`` so ``pricing_at`` results sort consistently.
# Letters: L=loss, LR=loss ratio, M=margin, P=premium, PQ=P/Q (leverage),
# Q=capital/equity, ROE=return on equity. ``a`` (asset level) is not a
# pricing statistic and is not included here.
PRICING_STAT_ORDER = ['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE']
PRICING_STAT_DTYPE = pd.CategoricalDtype(categories=PRICING_STAT_ORDER, ordered=True)


# Canonical row MultiIndex for ``Portfolio.stats_df``. Parallels
# ``aggregate.distributions._STATS_ROW_INDEX`` (meta + freq + sev + agg
# moment blocks). Kept as its own constant so future Portfolio-only
# rows (e.g. between-line copula moments) do not bleed into
# Aggregate's surface.
_PORT_STATS_ROW_INDEX = pd.MultiIndex.from_tuples(
    [
        ('meta', 'name'), ('meta', 'limit'), ('meta', 'attachment'),
        ('meta', 'el'), ('meta', 'prem'), ('meta', 'lr'),
        ('meta', 'sevcv_param'), ('meta', 'mix_cv'), ('meta', 'wt'),
        ('freq', 'ex1'), ('freq', 'ex2'), ('freq', 'ex3'),
        ('freq', 'mean'), ('freq', 'cv'), ('freq', 'skew'),
        ('sev', 'ex1'), ('sev', 'ex2'), ('sev', 'ex3'),
        ('sev', 'mean'), ('sev', 'cv'), ('sev', 'skew'),
        ('agg', 'ex1'), ('agg', 'ex2'), ('agg', 'ex3'),
        ('agg', 'mean'), ('agg', 'cv'), ('agg', 'skew'),
    ],
    names=['component', 'measure'],
)


class Portfolio(object):
    """
    Portfolio creates and manages a portfolio of Aggregate objects each modeling one
    unit of business. Applications include

    - Model a book of insurance
    - Model a large account with several sub lines
    - Model a reinsurance portfolio or large treaty

    """

    def __init__(self, name, spec_list, uw=None):
        """
        Create a new :class:`Portfolio` object.

        :param name: The name of the portfolio. No spaces or underscores.
        :param spec_list: A list of

           1. dictionary: Aggregate object dictionary specifications or
           2. Aggregate: An actual aggregate objects or
           3. tuple (type, dict) as returned by uw['name'] or
           4. string: Names referencing objects in the optionally passed underwriter
           5. a single DataFrame: empirical samples (the total column, if present, is ignored);
              a p_total column is used for probabilities if present

        :returns:  new :class:`Portfolio` object.
        """
        self.name = name
        self.agg_list = []
        self.line_names = []
        self._valid = None
        self.sample_df = None
        logger.debug(f'Portfolio.__init__| creating new Portfolio {self.name}')
        # logger.debug(f'Portfolio.__init__| creating new Portfolio {self.name} at {super(Portfolio, self).__repr__()}')
        ma = MomentAggregator()
        max_limit = 0
        if len(spec_list) == 1 and isinstance(spec_list[0], pd.DataFrame):
            # create from samples...slightly different looping behavior
            logger.info('Creating from sample DataFrame')
            spec_list = spec_list[0]
        if isinstance(spec_list, pd.DataFrame):
            spec_list = spec_list.copy().astype(float)
            if 'p_total' not in spec_list:
                logger.info('Adding p_total column to DataFrame with equal probs')
                spec_list['p_total'] = np.repeat(1 / len(spec_list), len(spec_list))
            # it is helpful to know what sample the object is created
            self.sample_df = spec_list

        for spec in spec_list:
            if isinstance(spec, Aggregate):
                # directly passed in an agg object
                a = spec
                agg_name = spec.name
            elif isinstance(spec, str) and isinstance(spec_list, pd.DataFrame):
                if spec not in ['total', 'p_total']:
                    # hack: issue: close values mess up the discrete distribution
                    # 2**-30 = 9.313225746154785e-10 approx 1e-9, so to ensure we don't
                    # have any merging issues in the discrete distribution, we round
                    # to 8 decimal places.
                    temp = spec_list[[spec, 'p_total']]
                    temp['rounded'] = np.round(temp[spec].astype(float), 8)
                    s = temp.groupby('rounded').sum()
                    a = Aggregate(name=spec,
                                  exp_en=1,
                                  sev_name='dhistogram', sev_xs=s.index.values, sev_ps=s.p_total.values,
                                  freq_name='fixed')
                    agg_name = spec
                else:
                    a = None
            elif isinstance(spec, str):
                # look up object in uw return actual instance
                # uw.build_many(spec, update=False) parses or looks up by name
                # and returns a list[ParsedProgram] with `.object` populated but
                # not smart-updated — Portfolio handles its own update later.
                if uw is None:
                    raise ValueError('Must pass valid Underwriter instance to create aggs by name')
                try:
                    a_out = uw.build_many(spec, update=False)
                except Exception as e:
                    logger.error('Item %s not found in your underwriter', spec)
                    raise e
                # a is a disct (kind, name) -> (obj or spec, program) pair. Portfolios are ?always created so
                # here, spec is the name
                assert len(a_out) == 1
                # remember, the thing you make must be called a as part of the loop
                a = a_out[('agg', spec)][0]
                assert isinstance(a, Aggregate)
                agg_name = a.name
            elif isinstance(spec, tuple):
                # uw returns type, spec
                assert spec[0] == 'agg'
                a = Aggregate(**spec[1])
                agg_name = spec[1]['name']
            elif isinstance(spec, dict):
                a = Aggregate(**spec)
                agg_name = spec['name'][0] if isinstance(spec['name'], list) else spec['name']
            else:
                raise ValueError(f'Invalid type {type(spec)} passed to Portfolio, expect Aggregate, str or dict.')

            if a is not None:
                # deals with total in DataFrame intput mode
                self.agg_list.append(a)
                self.line_names.append(agg_name)
                self.__setattr__(agg_name, a)
                mixed = a.stats_df['mixed']
                ma.add_fs(mixed[('freq', 'ex1')], mixed[('freq', 'ex2')], mixed[('freq', 'ex3')],
                          mixed[('sev',  'ex1')], mixed[('sev',  'ex2')], mixed[('sev',  'ex3')])
                max_limit = max(max_limit, np.max(np.array(a.limit)))

        self.line_names_ex = self.line_names + ['total']
        self.line_name_pipe = "|".join(self.line_names_ex)
        for n in self.line_names:
            # line names cannot equal total
            if n == 'total':
                raise ValueError('Line names cannot equal total, it is reserved for...total')

        # Canonical ``stats_df``: per-unit columns + ``mixed`` +
        # ``empirical`` + ``error``. Mirror of ``Aggregate.stats_df``
        # shape, minus the ``independent`` column (an
        # Aggregate-frequency-mixing concept with no portfolio-level
        # analog). ``empirical`` and ``error`` start NaN; ``update``
        # populates them after the FFT.
        self._build_stats_df(ma, max_limit)
        # future storage
        self.density_df = None
        self.independent_density_df = None
        self._augmented_dfs: dict[str, pd.DataFrame] = {}
        self._last_applied_distortion_name: str | None = None
        self.independent_stats_df = None
        self.padding = 0
        self._var_tvar_function = None
        self._cdf = None
        self._pdf = None
        self.bs = 0
        self.log2 = 0
        self.ex = 0
        self.last_update = 0
        self.hash_rep_at_last_update = ''
        self._distortion = None
        self.sev_calc = ''
        self._remove_fuzz = 0
        self.discretization_calc = ''
        self.normalize = None
        self._line_renamer = None
        self._tm_renamer = None
        # if created by uw it stores the program here
        self.program = ''
        self.distortions = None
        self.distortion_df = None
        self.figure = None

        # for consistency with Aggregates
        self.agg_m = self.stats_df.loc[('agg', 'ex1'), 'total']
        self.agg_cv = self.stats_df.loc[('agg', 'cv'), 'total']
        self.agg_skew = self.stats_df.loc[('agg', 'skew'), 'total']
        # variance and sd come up in exam questions
        self.agg_sd = self.agg_m * self.agg_cv
        self.agg_var = self.agg_sd * self.agg_sd
        # these are set when the object is updated
        self.est_m = self.est_cv = self.est_skew = self.est_sd = self.est_var = 0

        self.validation_eps = VALIDATION_EPS

    def help(self, regex):
        """
        Lookup help on methods and properties matching ``regex``.
        """
        agg_help(self, regex)

    def add_exa_sample(self, sample, S_calculation='forwards'):
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

        The index on the input sample is ignored.

        Formally ``extensions.samples.add_exa_sample``.

        """

        # starter information
        # cut_eps = np.finfo(float).eps
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
        sample_in = sample_in.reset_index(drop=True)

        # want to align the index to that of self.density_df; all multiples of self.bs
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
        self._var_tvar_function = None

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
                                                                    fill_value=0).cumsum() * self.bs

        # put in totals for the ratios... this is very handy in later use
        for metric in ['exi_xlea_', 'exi_xgta_', 'exi_xeqa_']:
            df[metric + 'sum'] = df.filter(regex=metric).sum(axis=1)

        df = df.set_index('loss', drop=False)
        df.index.name = None
        return df

    @staticmethod
    def create_from_sample(name, sample_df, bs, log2=16, **kwargs):
        """
        Create from a multivariate sample, update with bs, execute switcheroo,
        and return new Portfolio object.

        OED: switcheroo, n. a change of position or an exchange, esp. one intended
        to surprise or deceive; a reversal or turn-about; spec. an unexpected change
        or ‘twist’ in a story. Also attributive, reversible, reversed.

        """
        logger.info(f'Creating Porfolio {name} from sample_df. Handles adding total and converting to floats.')
        port = Portfolio(name, sample_df)
        logger.info(f'Updating with bs={bs}, log2={log2}, remove_fuzz=True')
        port.update(bs=bs, log2=log2, remove_fuzz=True, **kwargs)
        # archive the original density_df
        port.independent_density_df = port.density_df.copy()
        # execute switeroo
        logger.info('Creating exa_sample and executing switcheroo')
        port.density_df = port.add_exa_sample(sample_df)
        # update total stats — snapshot the pre-switcheroo stats_df and
        # recompute the empirical agg-total moments from the new density_df.
        logger.info('Updating total statistics (WARNING: these are now empirical)')
        port.independent_stats_df = port.stats_df.copy()
        _t = port.density_df['p_total'] * port.density_df['loss']
        _ex1 = float(np.sum(_t))
        _t *= port.density_df['loss']
        _ex2 = float(np.sum(_t))
        _t *= port.density_df['loss']
        _ex3 = float(np.sum(_t))
        port.est_m, port.est_cv, port.est_skew = (
            MomentAggregator.static_moments_to_mcvsk(_ex1, _ex2, _ex3)
        )
        port.ex = port.est_m
        port.est_sd = port.est_m * port.est_cv
        port.est_var = port.est_sd ** 2
        port._write_empirical_stats(_ex1, _ex2, _ex3)
        # return new created object
        logger.info('Returning new Portfolio object')
        return port

    def sample_compare(self, ax=None):
        """Compare the sample-based portfolio total to the independent
        marginal sum.

        Compares the ``empirical`` agg-total moments from the
        post-switcheroo ``stats_df`` against the pre-switcheroo
        snapshot stored in ``independent_stats_df``.
        """
        if self.independent_density_df is None:
            raise ValueError('No independent_density_df, cannot compare')

        if ax is not None:
            ax.plot(self.independent_density_df.index, self.independent_density_df['S'], lw=1, label='independent')
            ax.plot(self.density_df.index, self.density_df['S'], lw=1, label='sample')
            ax.legend()

        return pd.concat(
            (self.independent_stats_df[['total', 'empirical']],
             self.stats_df[['total', 'empirical']]),
            keys=['independent', 'sample'], axis=1,
        )

    def sample_density_compare(self, fuzz=0):
        """
        Compare from density_df
        """
        bit = pd.concat((self.independent_density_df.filter(regex='p_'),
                         self.density_df.filter(regex='p_total|exeqa_')),
                        axis=1, keys=['independent', 'sample'])
        bit = bit.loc[(bit[('independent', 'p_total')] > fuzz) + (bit[('sample', 'p_total')] > fuzz)]
        bit[('', 'difference')] = bit[('independent', 'p_total')] - bit[('sample', 'p_total')]
        return bit

    def pricing_bounds(self, premium, a=0, p=0, n_tps=512, s=512, kind='interp', slow=False, verbose=250):
        """
        Natural allocation premium ranges by unit, consistent with total
        premium at asset level ``a`` or ``p``.

        PENDING: this method is currently broken on dense portfolios (matmul
        shape mismatch when ``s`` defaults to ``port.density_df.S.values``)
        and was written against the legacy ``Bounds`` API (``tvar_cloud``,
        ``p_star`` as a method). Pending API alignment with the rewritten
        ``Bounds`` class. Raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            'Portfolio.pricing_bounds is pending an update for the new '
            'Bounds API (1.0.0a11). See GitHub issue / CLAUDE.md TODO.')
        from .bounds import Bounds  # noqa: unreachable
        if a == 0:
            assert p > 0, 'Must provide either a or p'
            a = self.q(p)

        # need a -= self.bs?
        S = self.density_df.loc[:a, 'S'].copy()
        # last entry needs to include all remaining losses from a-bs onwards, hence:
        S.iloc[-1] = 0.
        bounds = Bounds(self)
        bounds.add_one = True
        # bounds.tvar_cloud('total', premium, a, n_tps, S.values, kind=kind)
        if s <= 0:
            bounds.tvar_cloud('total', premium, a,  n_tps, S.values, kind=kind)
        else:
            bounds.tvar_cloud('total', premium, a,  n_tps, s, kind=kind)

        # TODO: (hack) pl=1 and s=1 is driving an error NAN - need to replace with 0. But WHY?
        gS = bounds.cloud_df.fillna(0).values.T
        gps = -np.diff(gS, axis=1, prepend=1)
        # sum products for allocations
        deal_losses = self.density_df.filter(regex='exeqa_[A-Z0-9]').loc[:a]
        if self.sf(a) > 0:
            # see notes below in slow method
            logger.info('Adjusting tail of deal_losses')
            deal_losses.iloc[-1] = self.density_df.loc[a-self.bs].filter(regex='exi_xgta_[A-Z]') * a

        # compute the allocations
        allocs = pd.DataFrame(
            gps @ (deal_losses.to_numpy()),
            columns=[i.replace('exeqa_', 'alloc_') for i in deal_losses.columns],
            index=bounds.weight_df.index)

        # this is a good audit: should have max = min
        allocs['total'] = allocs.sum(1)

        allocs = allocs.rename(columns=lambda x: x.replace('alloc_', ''))
        # summary stats
        stats = pd.concat((pd.concat((allocs.min(0), allocs.mean(0), allocs.max(0)),
                                     keys=['min', 'mean', 'max'], axis=0).unstack(1),
                           pd.concat((allocs.idxmin(0), allocs.idxmax(0)),
                                     keys=['min', 'max'], axis=0).unstack(1)),
                          axis=1).fillna('')

        if slow:
            # alternative method (slow)
            logger.info('Calculating extreme natural allocations: mechanical method')
            # probabilities of total loss
            pt = self.density_df.p_total
            S = 1 - np.minimum(1, pt.loc[:a].cumsum())
            # individual deal losses
            deal_losses = self.density_df.loc[:a].filter(regex='exeqa_')
            if self.sf(a) > 0:
                # need to adjust for losses in default region
                # use the linear natural allocation (note price uses the lifted allocation)
                # replace the last row with E[Xi/X | X>=a] a
                # need >=, so pull from the prior row
                # exi_xgta includes a sum column for the total, must exclude that
                deal_losses.iloc[-1] = self.density_df.\
                                           drop(columns='exi_xgta_sum').\
                                           loc[[a-self.bs]].filter(regex='exi_xgta_') * a
            # deal_losses
            # linear natural allocation for each total outcome
            ans = {}
            i = 0
            for _, r in bounds.weight_df.reset_index().iterrows():
                pl, pu, tl, tu, w = r.values
                d = Distortion('bitvar', w, df=[pl, pu])
                # pricing kernel, this in-lines the function rap (that was in extensions.sample)
                gS = np.array(d.g(S))
                z = -np.diff(gS[:-1], prepend=1, append=0)
                ans[(pl, pu)] = z @ deal_losses
                i += 1
                if verbose and i % verbose == 0:
                    logger.info(f'Completed {i} out of {len(bounds.weight_df)} biTVaRs')

            # all pricing ranges: rows = (pl, pu) pairs defining extreme distortion
            # cols = deals
            allocs_slow = pd.concat(ans.values(), keys=ans.keys()).unstack(2)
            # get max/min/mean natural allocation by deal and compare to actual pricing
            comp = {}
            for c in allocs_slow.iloc[:, :-1]:
                col = allocs_slow[c]
                nm = c.split('_')[1]
                comp[nm] = [col.min(), col.mean(), col.max()]

            comp = pd.DataFrame(comp.values(),
                                 index=comp.keys(),
                                 columns=['min', 'mean', 'max'])
        else:
            comp = allocs_slow = None
        # assemble answer
        p_star = bounds.p_star('total', premium, a, kind=kind)
        return PricingBoundsResult(
            bounds=bounds, allocs=allocs, stats=stats, comp=comp,
            allocs_slow=allocs_slow, p_star=p_star)

    @property
    def distortion(self):
        return self._distortion

    def remove_fuzz(self, df=None, eps=0, force=False, log=''):
        """
        remove fuzz at threshold eps. if not passed use np.finfo(float).eps.

        Apply to self.density_df unless df is not None

        Only apply if self.remove_fuzz or force
        :param eps:
        :param df:  apply to dataframe df, default = self.density_df
        :param force: do regardless of self.remove_fuzz
        :return:
        """

        if df is None:
            df = self.density_df
        if eps == 0:
            eps = np.finfo(float).eps

        if self._remove_fuzz or force:
            logger.debug(f'Portfolio.remove_fuzz | Removing fuzz from {self.name} dataframe, caller {log}')
            df[df.select_dtypes(include=['float64']).columns] = \
                df.select_dtypes(include=['float64']).map(lambda x: 0 if abs(x) < eps else x)

    def __repr__(self):
        """
        Goal unmbiguous
        :return:
        """
        # return str(self.to_dict())
        # this messes up when port = self has been enhanced...

        # cannot use ex, etc. because object may not have been updated
        return f'{self.name} at {super().__repr__()}'

    def _repr_html_(self):
        """
        Updated to mimic Aggregate
        """
        s = [f'<h3>Portfolio object: {self.name}</h3>']
        _n = len(self.agg_list)
        _s = "" if _n <= 1 else "s"
        s.append(f'Portfolio contains {_n} aggregate component{_s}.')
        if self.bs > 0:
            s.append(f'Updated with bucket size {self.bs:.6g}, log2 = {self.log2}, validation: {self.explain_validation()}')
        df = self.describe
        return '\n'.join(s) + df.fillna('').to_html()

    def __str__(self):
        """ Default behavior """
        ex = float(self.stats_df.loc[('agg', 'mean'), 'total'])
        empex_raw = self.stats_df.loc[('agg', 'mean'), 'empirical']
        if pd.isna(empex_raw):
            empex = np.nan
            isupdated = False
        else:
            empex = float(empex_raw)
            isupdated = True

        s = [f'Portfolio object         {self.name:s}',
             f'Theoretic expected loss  {ex:,.1f}',
             f'Estimated expected loss  {empex:,.1f}',
             f'Error                    {empex / ex - 1:.6g}'
             ]

        s.append(
             f'Updated                  {isupdated}'
        )

        if self.bs > 0:
            if self.bs > 1:
                s.append(f'bs                       {self.bs}')
            else:
                s.append(f'bs                       1 / {int(1/self.bs)}')
            s.append(f'log2                     {self.log2}')
            s.append(f'validation_eps           {self.validation_eps}')
            s.append(f'padding                  {self.padding}')
            s.append(f'sev_calc                 {self.sev_calc}')
            s.append(f'normalize                {self.normalize}')
            s.append(f'remove_fuzz              {self._remove_fuzz}')
            s.append(f'distortion               {repr(self._distortion)}')

        if isupdated:
            s.append('')
            with pd.option_context('display.width', 140, 'display.float_format', lambda x: f'{x:,.5g}'):
                # get it on one row
                s.append(str(self.describe))
        # s.append(super(Portfolio, self).__repr__())
        return '\n'.join(s)

    def __hash__(self):
        """
        hashing behavior
        :return:
        """
        return hash(repr(self.__dict__))

    def __iter__(self):
        """
        make Portfolio iterable: for each x in Portfolio

        :return:
        """
        return iter(self.agg_list)

    def __getitem__(self, item):
        """
        allow Portfolio[slice] to return bits of agg_list

        :param item:
        :return:
        """
        if type(item) == str:
            return self.agg_list[self.line_names.index(item)]
        return self.agg_list[item]

    @property
    def info(self):
        s = []
        s.append(f'portfolio object name    {self.name}')
        s.append(f'aggregate objects        {len(self.line_names):d}')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1/self.bs)}'
            s.append(f'bs                       {bss}')
            s.append(f'log2                     {self.log2}')
            s.append(f'padding                  {self.padding}')
            s.append(f'sev_calc                 {self.sev_calc}')
            s.append(f'normalize                {self.normalize}')
            s.append(f'last update              {self.last_update}')
            s.append(f'hash                     {self.hash_rep_at_last_update:x}')
        return '\n'.join(s)

    @property
    def describe(self):
        """Theoretic-and-empirical stats. Used in ``_repr_html_``.

        Reads from the canonical ``stats_df``: theoretical moments from
        the ``total`` column, empirical from ``empirical``, errors from
        ``error``. The output shape mirrors ``Aggregate.describe`` — one
        ``Freq``/``Sev``/``Agg`` row block per unit + ``total`` — so
        callers see the same 3-row × 6-col view they always have.
        """
        _total = self.stats_df['total']
        df = pd.DataFrame(
            {
                'E[X]':    [float(_total[('freq', 'ex1')]),
                            float(_total[('sev',  'ex1')]),
                            float(_total[('agg',  'ex1')])],
                'CV(X)':   [float(_total[('freq', 'cv')]),
                            float(_total[('sev',  'cv')]),
                            float(_total[('agg',  'cv')])],
                'Skew(X)': [float(_total[('freq', 'skew')]),
                            float(_total[('sev',  'skew')]),
                            float(_total[('agg',  'skew')])],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'

        # Post-update? Empirical agg moments live in stats_df['empirical'].
        # After the punch-up: portfolio-level sev empirical is also
        # populated (via MomentAggregator off per-unit empirical sev);
        # surface it in the describe table too.
        emp = self.stats_df['empirical']
        emp_agg_m = emp.get(('agg', 'mean'), np.nan)
        if pd.notna(emp_agg_m):
            df.loc['Sev', 'Est E[X]'] = float(emp[('sev', 'mean')])
            df.loc['Agg', 'Est E[X]'] = float(emp_agg_m)
            df['Err E[X]'] = df['Est E[X]'] / df['E[X]'] - 1
            df.loc['Sev', 'Est CV(X)'] = float(emp[('sev', 'cv')])
            df.loc['Agg', 'Est CV(X)'] = float(emp[('agg', 'cv')])
            df['Err CV(X)'] = df['Est CV(X)'] / df['CV(X)'] - 1
            df.loc['Sev', 'Est Skew(X)'] = float(emp[('sev', 'skew')])
            df.loc['Agg', 'Est Skew(X)'] = float(emp[('agg', 'skew')])
            df = df[['E[X]', 'Est E[X]', 'Err E[X]', 'CV(X)', 'Est CV(X)', 'Err CV(X)',
                     'Skew(X)', 'Est Skew(X)']]

        t1 = [a.describe for a in self] + [df]
        t2 = [a.name for a in self] + ['total']
        df = pd.concat(t1, keys=t2, names=['unit', 'X'])
        return df

    @property
    def spec(self):
        """
        Get the dictionary specification.

        :return:
        """
        d = dict()
        d['name'] = self.name
        d['spec_list'] = [a._spec for a in self.agg_list]
        return d

    @property
    def spec_ex(self):
        """
        All relevant info.

        :return:
        """
        return {'type': type(self), 'spec': self.spec, 'bs': self.bs, 'log2': self.log2,
                'aggs': len(self.agg_list)}

    def json(self, stream=None):
        """
        write object as json

        :param    stream:
        :return:  stream or text
        """

        args = dict()
        args["bs"] = self.bs
        args["log2"] = self.log2
        args["padding"] = self.padding
        args["distortion"] = repr(self._distortion)
        args["sev_calc"] = self.sev_calc
        args["remove_fuzz"] = self._remove_fuzz
        args["last_update"] = str(self.last_update)
        args["hash_rep_at_last_update"] = str(self.hash_rep_at_last_update)

        d = self.spec
        d['args'] = args

        logger.debug(f'Portfolio.json| dummping {self.name} to {stream}')
        s = json.dumps(d)  # , default_flow_style=False, indent=4)
        logger.debug(f'Portfolio.json | {s}')
        if stream is None:
            return s
        else:
            return stream.write(s)

    def save(self, filename='', mode='a'):
        """
        persist to json in filename; if none save to user.json

        :param filename:
        :param mode: for file open
        :return:
        """
        if filename == "":
            filename = Path.home() / 'agg/user.json'
            filename.parent.mkdir(parents=True, exist_ok=True)

        with filename.open(mode=mode, encoding='utf-8') as f:
            self.json(stream=f)
            logger.debug(f'Portfolio.save | {self.name} saved to {filename}')

    def __add__(self, other):
        """
        Add two portfolio objects INDEPENDENT sum (down road can look for the same severity...)

        :param other:
        :return:
        """
        assert isinstance(other, Portfolio)
        new_spec = []
        for a in self.agg_list:
            c = deepcopy(a._spec)
            c['name'] = c['name']
            new_spec.append(c)
        for a in other.agg_list:
            c = deepcopy(a._spec)
            c['name'] = c['name']
            new_spec.append(c)

        return Portfolio(f'({self.name}) + ({other.name})', new_spec)

    def __rmul__(self, other):
        """
        new = other * self; treat as scale change

        :param other:
        :return:
        """

        assert other > 0

        new_spec = []
        for a in self.agg_list:
            new_spec.append(deepcopy(a._spec))

        for d in new_spec:
            # d is a dictionary agg spec, need to adjust the severity
            s = d['severity']
            if 'mean' in s:
                s['mean'] *= other
            elif 'scale' in s:
                s['scale'] *= other
            else:
                raise ValueError(f"Cannot adjust s['name'] for scale")

        return Portfolio(f'{other} x {self.name}', new_spec)

    def __mul__(self, other):
        """
        new = self * other, other integer, sum of other independent copies

        :param other:
        :return:
        """

        assert isinstance(other, int)

        new_spec = []
        for a in self.agg_list:
            new_spec.append(deepcopy(a._spec))

        for d in new_spec:
            # d is a dictionary agg spec, need to adjust the frequency
            # TODO better freq dists; deal with Bernoulli where n=log<1
            d['frequency']['n'] *= other

        return Portfolio(f'Sum of {other} copies of {self.name}', new_spec)

    def snap(self, x):
        """
        snap value x to the index of density_df

        :param x:
        :return:
        """
        ix = self.density_df.index.get_indexer([x], 'nearest')[0]
        return self.density_df.iloc[ix, 0]

    def q(self, p, kind='lower'):
        """
        Return quantile function of density_df.p_total.

        Definition 2.1 (Quantiles)
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] ≥ α} is the lower α-quantile of X
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] > α} is the upper α-quantile of X.

        ``kind=='middle'`` has been removed.

        :param p:
        :param kind: 'lower' or 'upper'.
        :return:
        """

        if kind == 'middle':
            # logger.warning(f'kind=middle is deprecated, replacing with kind=lower')
            kind = 'lower'

        assert kind in ['lower', 'upper'], 'kind must be lower or upper'

        if self._var_tvar_function is None:
            # revised June 2023
            ser = self.density_df.query('p_total > 0').p_total
            self._make_var_tvar(ser)

        return self._var_tvar_function[kind](p)

    def _make_var_tvar(self, ser):
        """
        There is no severity version here, so this knows where to store the answer, cf Aggregate version.
        """
        self._var_tvar_function = {}
        qf = make_var_tvar(ser)
        self._var_tvar_function['upper'] = qf.q_upper
        self._var_tvar_function['lower'] = qf.q_lower
        self._var_tvar_function['tvar'] = qf.tvar

    def cdf(self, x):
        """
        distribution function

        :param x:
        :return:
        """
        if self._cdf is None:
            # Dec 2019: kind='linear' --> kind='previous'
            self._cdf = interpolate.interp1d(self.density_df.loss, self.density_df.F, kind='previous',
                                             bounds_error=False, fill_value='extrapolate')
        return 0. + self._cdf(x)

    def sf(self, x):
        """
        survival function

        :param x:
        :return:
        """
        return 1 - self.cdf(x)

    def pdf(self, x):
        """
        probability density function, assuming a continuous approximation of the bucketed density
        :param x:
        :return:
        """
        if self._pdf is None:
            self._pdf = interpolate.interp1d(self.density_df.loss, self.density_df.p_total, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
        return self._pdf(x) / self.bs

    def pmf(self, x):
        """
        Probability mass function, treating aggregate as discrete
        x must be in the index (?)

        """
        if self.density_df is None:
            raise ValueError("Must update before computing probabilities!")

        try:
            return self.density_df.loc[x, 'p_total']
        except KeyError:
            return 0.0
            # raise KeyError(f'Value {x} must be in index for probability mass function.')

    def var(self, p):
        """
        value at risk = alias for quantile function

        :param p:
        :return:
        """
        return self.q(p)

    def tvar(self, p, kind=''):
        """
        Compute the tail value at risk at threshold p. Revised June 2023.

        Really this function returns ES, CVaR, but in modern terminology
        this is called TVaR.

        Definition 2.6 (Tail mean and Expected Shortfall)
        Assume E[X−] < ∞. Then
        x¯(α) = TM_α(X) = α^{−1}E[X 1{X≤x(α)}] + x(α) (α − P[X ≤ x(α)])
        is α-tail mean at level α the of X.
        Acerbi and Tasche (2002)

        McNeil etc. p66-70 - this follows from def of ES as an integral
        of the quantile function


        :param p:
        :param kind: No longer neeed as the new method is exact (equals the old
        tail) and about 1000x faster.
        :return:
        """

        if kind != '' and getattr(self, 'tvar-warning', 0) == 0:
            logger.warning('kind is no longer used in TVaR, new method equivalent to kind=tail but much faster. '
                           'Argument kind will be removed in the future.')
            setattr(self, 'tvar-warning', 1)

        if kind == 'inverse':
            logger.warning('kind=inverse called...??!!')

        assert self.density_df is not None, 'Must recompute prior to computing tail value at risk.'

        if self._var_tvar_function is None:
            # revised June 2023
            ser = self.density_df.query('p_total > 0').p_total
            self._make_var_tvar(ser)

        return self._var_tvar_function['tvar'](p)

    def tvar_threshold(self, p, kind):
        """
        Find the value pt such that TVaR(pt) = VaR(p) using Bisection method.
        Will fail if p=0 because signs are the same.
        """
        # target value
        a = self.q(p, kind)

        if p == 0:
            # mean is mean
            return 0

        def f(p):
            return self.tvar(p) - a
        p1 = bisect(f, 0, 1)
        # loop = 0
        # p1 = max(.1, 1 - 2 * (1 - p))
        # fp1 = f(p1)
        # delta = 1e-5
        # while abs(fp1) > 1e-6 and loop < 20:
        #     df1 = (f(p1 + delta / 2) - f(p1 - delta / 2)) / delta
        #     p1 = p1 - fp1 / df1
        #     fp1 = f(p1)
        #     loop += 1
        # if loop == 20:
        #     raise ValueError(f'Difficulty computing TVaR to match VaR at p={p}; last guess {p1}')
        return p1

    def as_severity(self, limit=np.inf, attachment=0, conditional=False):
        """
        Convert portfolio into a severity without recomputing.

        Throws an error if self not updated.

        :param limit:
        :param attachment:
        :param conditional:
        :return:
        """
        if self.density_df is None:
            raise ValueError('Must update prior to converting to severity')
        return Severity(sev_name=self, sev_a=self.log2, sev_b=self.bs,
                        exp_attachment=attachment, exp_limit=limit, sev_conditional=conditional)

    def approximate(self, approx_type='slognorm', output='scipy'):
        """
        Create an approximation to self using method of moments matching.

        Returns a dictionary specification of the portfolio aggregate_project.
        If updated uses empirical moments, otherwise uses theoretic moments

        :param approx_type: slognorm | sgamma | normal
        :param output: return a dict or agg language specification
        :return:
        """

        if approx_type == 'all':
            return {kind: self.approximate(kind)
                    for kind in ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']}

        emp_mean = self.stats_df.loc[('agg', 'mean'), 'empirical']
        if pd.isna(emp_mean):
            # not updated — use theoretical moments from the mixed column
            m = float(self.stats_df.loc[('agg', 'mean'), 'total'])
            cv = float(self.stats_df.loc[('agg', 'cv'), 'total'])
            skew = float(self.stats_df.loc[('agg', 'skew'), 'total'])
        else:
            # use empirical (post-FFT) moments matched to the computed aggregate
            m = float(emp_mean)
            cv = float(self.stats_df.loc[('agg', 'cv'), 'empirical'])
            skew = float(self.stats_df.loc[('agg', 'skew'), 'empirical'])

        name = f'{approx_type[0:4]}.{self.name[0:5]}'
        agg_str = f'agg {name} 1 claim sev '
        note = f'frozen version of {self.name}'
        return approximate_from_mcvsk(m, cv, skew, name, agg_str, note, approx_type, output)

    def percentiles(self, pvalues=None):
        """
        Per-line percentiles (interpolated) of the FFT-derived
        ``density_df`` distribution.

        :param pvalues: optional vector of log values to use. If None sensible defaults provided
        :return: DataFrame of percentiles indexed by line and log
        """
        df = pd.DataFrame(columns=['line', 'log', 'Agg Quantile'])
        df = df.set_index(['line', 'log'])
        # df.columns.name = 'perspective'
        if pvalues is None:
            pvalues = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.994, 0.995, 0.999, 0.9999]
        for line in self.line_names_ex:
            q_agg = interpolate.interp1d(self.density_df[f'p_{line}'].cumsum(), self.density_df.loss,
                                         kind='linear', bounds_error=False, fill_value='extrapolate')
            for p in pvalues:
                qq = q_agg(p)
                df.loc[(line, p), :] = [float(qq)]
        df = df.unstack(level=1)
        return df

    def recommend_bucket(self):
        """
        Data to help estimate a good bucket size.

        :return:
        """
        df = pd.DataFrame(columns=['line', 'bs10'])
        df = df.set_index('line')
        for a in self.agg_list:
            df.loc[a.name, :] = [a.recommend_bucket(10)]
        df['bs11'] = df['bs10'] / 2
        df['bs12'] = df['bs10'] / 4
        df['bs13'] = df['bs10'] / 8
        df['bs14'] = df['bs10'] / 16
        df['bs15'] = df['bs10'] / 32
        df['bs16'] = df['bs10'] / 64
        df['bs17'] = df['bs10'] / 128
        df['bs18'] = df['bs10'] / 256
        df['bs19'] = df['bs10'] / 515
        df['bs20'] = df['bs10'] / 1024
        df.loc['total', :] = df.sum()
        return df

    def best_bucket(self, log2=16, recommend_p=RECOMMEND_P):
        """
        Recommend the best bucket. Rounded recommended bucket for log2 points.

        TODO: Is this really the best approach?!

        :param log2:
        :param recommend_p:
        :return:
        """

        # bs = sum([a.recommend_bucket(log2, p=recommend_p) for a in self])
        bs = sum([a.recommend_bucket(log2, p=recommend_p) ** 2 for a in self]) ** 0.5

        return round_bucket(bs)

    def update(self, log2, bs, remove_fuzz=False,
               sev_calc='discrete', discretization_calc='survival', normalize=True, padding=1,
               trim_df=False, add_exa=True, force_severity=True, recommend_p=RECOMMEND_P,
               debug=False):
        """

        TODO: currently debug doesn't do anything...

        Create density_df, performs convolution. optionally adds additional information if ``add_exa=True``
        for allocation and priority analysis

        num buckets and max loss from bucket size

        Aggregate reinsurance in parser has replaced the aggregate_cession_function (a function of a Portfolio object
        that adjusts individual line densities; applied after line aggs created but before creating not-lines;
        actual statistics do not reflect impact.) Agg re by unit is now applied in the Aggregate object.

        TODO: consider aggregate covers at the portfolio level...Where in parse - at the top!


        :param log2:
        :param bs: bucket size
        :param remove_fuzz: remove machine noise elements from FFT
        :param sev_calc: how to calculate the severity, discrete (point masses as xs) or continuous (uniform between xs points)
        :param discretization_calc:  survival or distribution (accurate on right or left tails)
        :param normalize: if true, normalize the severity so sum probs = 1. This is generally what you want; but
        :param padding: for fft 1 = double, 2 = quadruple
        :param epds: epd points for priority analysis; if None-> sensible defaults
        :param trim_df: remove unnecessary columns from density_df before returning
        :param add_exa: run add_exa to append additional allocation information needed for pricing; if add_exa also add
            epd info
        :param force_severity: force computation of severities for aggregate components even when approximating
        :param recommend_p: percentile to use for bucket recommendation.
        :param debug: if True, print debug information
        :return:
        """
        self._valid = None # reset valid flag

        if log2 <= 0:
            raise ValueError('log2 must be >= 0')
        self.log2 = log2
        if bs == 0:
            self.bs = self.best_bucket(log2, recommend_p)
            logger.info(f'bs=0 enterered, setting bs={bs:.6g} using self.best_bucket rounded to binary fraction.')
        else:
            self.bs = bs
        self.padding = padding
        self.sev_calc = sev_calc
        self._remove_fuzz = remove_fuzz
        self.discretization_calc = discretization_calc
        self.normalize = normalize

        if self.hash_rep_at_last_update == hash(self):
            # this doesn't work
            logger.warning(f'Nothing has changed since last update at {self.last_update}')
            return

        self._var_tvar_function = None
        # density changes invalidate the augmented_df cache
        self._augmented_dfs = {}
        self._last_applied_distortion_name = None

        ft_line_density = {}
        # line_density = {}
        # not_line_density = {}

        # add the densities
        # num buckets and max loss from bucket size
        N = 1 << log2
        MAXL = N * bs
        xs = np.linspace(0, MAXL, N, endpoint=False)
        # make all the single line aggs
        # note: looks like duplication but will all be references
        # easier for add_exa to have as part of the portfolio module

        # where the answer will live
        self.density_df = pd.DataFrame(index=xs)
        self.density_df['loss'] = xs
        ft_all = None
        for agg in self.agg_list:
            raw_nm = agg.name
            nm = f'p_{agg.name}'
            # agg.update_work handles the reinsurance too
            agg.update_work(xs, self.padding, sev_calc, discretization_calc,
                            normalize, force_severity, debug)

            ft_line_density[raw_nm] = agg.ftagg_density
            self.density_df[nm] = agg.agg_density
            if ft_all is None:
                ft_all = np.copy(ft_line_density[raw_nm])
            else:
                ft_all *= ft_line_density[raw_nm]
        self.density_df['p_total'] = np.real(ift(ft_all, self.padding))
        # ft_line_density['total'] = ft_all

        # make the not self.line_density = sum of all but the given line
        # have the issue here that if you divide and the dist
        # is symmetric then you get a div zero...
        ft_nots = {}
        for line in self.line_names:
            ft_not = np.ones_like(ft_all)
            if np.any(ft_line_density[line] == 0):
                # have to build up
                for not_line in self.line_names:
                    if not_line != line:
                        ft_not *= ft_line_density[not_line]
            else:
                if len(self.line_names) > 1:
                    ft_not = ft_all / ft_line_density[line]
            ft_nots[line] = ft_not

        self.remove_fuzz(log='update')

        # add exa details
        if add_exa:
            self.add_exa(self.density_df, ft_nots=ft_nots)
        else:
            # at least want F and S to get quantile functions
            self.density_df['F'] = np.cumsum(self.density_df.p_total)
            self.density_df['S'] = 1 - self.density_df.F

        # Empirical portfolio-total agg moments straight from the FFT
        # output. Drives the canonical ``stats_df['empirical']`` agg
        # writes and the headline ``est_m`` / ``est_cv`` / ``est_skew``.
        # Plain summation — no tail-mass correction — to match the
        # legacy numerics the PEG regression baseline was captured
        # against.
        _t = self.density_df['p_total'] * self.density_df['loss']
        _ex1 = float(np.sum(_t))
        _t *= self.density_df['loss']
        _ex2 = float(np.sum(_t))
        _t *= self.density_df['loss']
        _ex3 = float(np.sum(_t))
        self.est_m, self.est_cv, self.est_skew = (
            MomentAggregator.static_moments_to_mcvsk(_ex1, _ex2, _ex3)
        )
        self.ex = self.est_m
        self.est_sd = self.est_m * self.est_cv
        self.est_var = self.est_sd ** 2
        self._write_empirical_stats(_ex1, _ex2, _ex3)

        self.last_update = np.datetime64('now')
        self.hash_rep_at_last_update = hash(self)
        if trim_df:
            self.trim_df()
        # invalidate stored functions
        self._var_tvar_function = None
        self._cdf = None

    def _build_stats_df(self, ma, max_limit):
        """Construct ``stats_df`` from the ``MomentAggregator`` after init.

        Adapted from ``Aggregate._init_stats_df`` + post-loop totals
        block. Per-unit columns hold each ``Aggregate.stats_df['mixed']``
        (the unit's view of its own theoretical moments); the ``total``
        column holds portfolio totals from the ``MomentAggregator``
        (``remix=False`` — the running totals across units).

        Portfolio's column is ``total`` rather than Aggregate's ``mixed``
        because there is no portfolio-level mixed-vs-independent analog
        (mixed-vs-independent is an Aggregate-only concept that strips
        a single agg's freq mixing distribution). Columns are therefore:
        per-unit + ``total`` + ``empirical`` + ``error``.

        ``empirical`` and ``error`` are left as NaN here; ``update``
        populates them after the FFT.
        """
        cols = list(self.line_names) + ['total', 'empirical', 'error']
        self.stats_df = pd.DataFrame(
            np.nan, index=_PORT_STATS_ROW_INDEX, columns=cols, dtype=object,
        )

        # Per-unit columns: copy each agg's ``stats_df['mixed']`` into the
        # matching unit column.
        for a in self.agg_list:
            a_mixed = a.stats_df['mixed']
            for idx, val in a_mixed.items():
                if idx in self.stats_df.index:
                    self.stats_df.loc[idx, a.name] = val

        # Portfolio totals: ``total`` = running totals across units
        # (``remix=False``, preserves each agg's freq mixing).
        unit_cols = list(self.line_names)
        _flat_names = MomentAggregator.column_names()

        def _collect(meta_key):
            return [self.stats_df.loc[('meta', meta_key), unit]
                    for unit in unit_cols]

        def _safe_sum(vals):
            out = 0.0
            for v in vals:
                try:
                    out += float(v)
                except (TypeError, ValueError):
                    return np.nan
            return out

        tot_el = _safe_sum(_collect('el'))
        tot_prem = _safe_sum(_collect('prem'))
        tot_lr = (tot_el / tot_prem) if tot_prem else np.nan
        # Attachment: portfolio total only makes sense if all units
        # share the same attachment (commonly 0); else NaN. Limit at
        # portfolio level is the max across units (legacy convention).
        _attaches = _collect('attachment')
        try:
            _attach_floats = [float(v) for v in _attaches]
            tot_attach = _attach_floats[0] if (
                _attach_floats and all(a == _attach_floats[0] for a in _attach_floats)
            ) else np.nan
        except (TypeError, ValueError):
            tot_attach = np.nan

        stats = ma.get_fsa_stats(total=True, remix=False)
        for flat, val in zip(_flat_names, stats):
            self.stats_df.loc[_flat_col_to_stats_index(flat), 'total'] = val
        self.stats_df.loc[('meta', 'name'), 'total'] = self.name
        self.stats_df.loc[('meta', 'limit'), 'total'] = max_limit
        self.stats_df.loc[('meta', 'attachment'), 'total'] = tot_attach
        self.stats_df.loc[('meta', 'el'), 'total'] = tot_el
        self.stats_df.loc[('meta', 'prem'), 'total'] = tot_prem
        self.stats_df.loc[('meta', 'lr'), 'total'] = tot_lr

    def _write_empirical_stats(self, agg_ex1, agg_ex2, agg_ex3):
        """Populate the ``empirical`` and ``error`` columns of ``stats_df``.

        Called from ``update`` (and ``create_from_sample`` after the
        switcheroo) once the FFT has run. Three sources feed it:

        - ``meta`` rows: limit / attachment / prem / el / lr have no
          empirical analog at the portfolio level — they are factual
          (limit, attach) or sums of expected values (prem, el, lr).
          Copy the ``total`` values across with implied ``error = 0``.
        - ``sev`` rows: combine each unit's empirical sev moments
          (``a.stats_df['empirical']``) with the unit's theoretical
          frequency via a fresh ``MomentAggregator``. Yields portfolio-
          level empirical sev mean / cv / skew + raw moments.
        - ``agg`` rows: ``ex1`` / ``ex2`` / ``ex3`` are the raw moments
          ``agg_ex1`` / ``agg_ex2`` / ``agg_ex3`` already computed from
          the portfolio-total FFT density; ``mean`` / ``cv`` / ``skew``
          are ``est_m`` / ``est_cv`` / ``est_skew``.

        ``('freq', *)`` empirical rows stay NaN — frequency is exact (no
        FFT applies to it) and the Aggregate convention is the same.
        """
        # meta: copy from total, error = 0 implicit.
        for meta_key in ('limit', 'attachment', 'el', 'prem', 'lr'):
            self.stats_df.loc[('meta', meta_key), 'empirical'] = (
                self.stats_df.loc[('meta', meta_key), 'total']
            )

        # sev: re-aggregate per-unit (theoretical freq, empirical sev)
        # through a fresh MomentAggregator. Aggregate's stats_df only
        # stores empirical sev mean/cv/skew (not raw moments), so we
        # invert each unit's (m, cv, sk) back to (ex1, ex2, ex3) via
        # MomentWrangler before feeding the aggregator. ``get_fsa_stats``
        # returns 18 entries: freq f1/f2/f3/m/cv/sk, sev s1/s2/s3/m/cv/sk,
        # agg a1/a2/a3/m/cv/sk; we use the sev block (indices 6..11).
        ma_emp = MomentAggregator()
        for a in self.agg_list:
            mixed = a.stats_df['mixed']
            emp = a.stats_df['empirical']
            s_m = float(emp[('sev', 'mean')])
            s_cv = float(emp[('sev', 'cv')])
            s_sk = float(emp[('sev', 'skew')])
            s_sd = s_m * s_cv
            mw = MomentWrangler()
            # (mean, variance, third-central-moment) → noncentral raw moments
            mw.central = (s_m, s_sd * s_sd, s_sk * s_sd ** 3)
            s_ex1, s_ex2, s_ex3 = mw.noncentral
            ma_emp.add_fs(
                float(mixed[('freq', 'ex1')]),
                float(mixed[('freq', 'ex2')]),
                float(mixed[('freq', 'ex3')]),
                s_ex1, s_ex2, s_ex3,
            )
        emp_stats = ma_emp.get_fsa_stats(total=True, remix=False)
        sev_block = [('sev', 'ex1'), ('sev', 'ex2'), ('sev', 'ex3'),
                     ('sev', 'mean'), ('sev', 'cv'), ('sev', 'skew')]
        for idx, val in zip(sev_block, emp_stats[6:12]):
            self.stats_df.loc[idx, 'empirical'] = val

        # agg: raw moments + mean/cv/skew from the FFT density.
        self.stats_df.loc[('agg', 'ex1'),  'empirical'] = agg_ex1
        self.stats_df.loc[('agg', 'ex2'),  'empirical'] = agg_ex2
        self.stats_df.loc[('agg', 'ex3'),  'empirical'] = agg_ex3
        self.stats_df.loc[('agg', 'mean'), 'empirical'] = self.est_m
        self.stats_df.loc[('agg', 'cv'),   'empirical'] = self.est_cv
        self.stats_df.loc[('agg', 'skew'), 'empirical'] = self.est_skew

        # error: relative diff vs the ``total`` column. For meta rows
        # where empirical == total the result is 0 (or NaN where total
        # is NaN, e.g. attach was inconsistent across units).
        self.stats_df['error'] = (
            pd.to_numeric(self.stats_df['empirical'], errors='coerce')
            / pd.to_numeric(self.stats_df['total'], errors='coerce')
            - 1
        )

    @property
    def valid(self):
        """
        Check if the model appears valid. See documentation for Aggregate.valid.

        An answer of True does not guarantee the model is valid, but
        False means it is definitely suspect. (Similar to the null hypothesis in a statistical test).
        Called and reported automatically by qd for Aggregate objects.

        Checks the relative errors (from ``self.describe``) for:

        * severity mean < eps
        * severity cv < 10 * eps
        * severity skew < 100 * eps (skewness is more difficult to estimate)
        * aggregate mean < eps and < 2 * severity mean relative error (larger values
          indicate possibility of aliasing and that ``bs`` is too small).
        * aggregate cv < 10 * eps
        * aggregate skew < 100 * esp

        eps = 1e-3 by default; change in ``validation_eps`` attribute.

        Test only applied for CV and skewness when they are > 0.

        :return: True if all tests are passed, else False.

        """
        if self._valid is not None:
            return self._valid

        rv = Validation.NOT_UNREASONABLE
        if self.density_df is None:
            self._valid = Validation.NOT_UPDATED
            return Validation.NOT_UPDATED

        for a in self.agg_list:
            r = a.valid
            if r & Validation.REINSURANCE:
                logger.info(f'Aggregate {a.name} has reinsurance, validation n/a')
            elif not r:
                logger.info(f'Aggregate {a.name} fails validation')
            rv |= r

        if rv != Validation.NOT_UNREASONABLE:
            logger.info(f'Exiting: Portfolio validation steps skipped due to failed or n/a Aggregate validation')
            self._valid = rv
            return rv
        else:
            logger.info('No Aggregate object fails validation')

        # apply validation to the Portfolio total
        df = self.describe.xs('total', level=0, axis=0).abs()
        try:
            df['Err Skew(X)'] = df['Est Skew(X)'] / df['Skew(X)'] - 1
        except ZeroDivisionError:
            df['Err Skew(X)'] = np.nan
        except TypeError:
            df['Err Skew(X)'] = np.nan
        eps = self.validation_eps
        if df.loc['Sev', 'Err E[X]'] > eps:
            logger.info('FAIL: Portfolio Sev mean error > eps')
            rv |= Validation.SEV_MEAN

        if df.loc['Agg', 'Err E[X]'] > eps:
            logger.info('FAIL: Portfolio Agg mean error > eps')
            rv |= Validation.AGG_MEAN

        if abs(df.loc['Sev', 'Err E[X]']) > 0 and df.loc['Agg', 'Err E[X]'] > 10 * df.loc['Sev', 'Err E[X]']:
            logger.info('FAIL: Agg mean error > 10 * sev error')
            rv |= Validation.ALIASING

        try:
            if np.inf > df.loc['Sev', 'CV(X)'] > 0 and df.loc['Sev', 'Err CV(X)'] > 10 * eps:
                logger.info('FAIL: Portfolio Sev CV error > eps')
                rv |= Validation.SEV_CV

            if np.inf > df.loc['Agg', 'CV(X)'] > 0 and df.loc['Agg', 'Err CV(X)'] > 10 * eps:
                logger.info('FAIL: Portfolio Agg CV error > eps')
                rv |= Validation.AGG_CV

            if np.inf > df.loc['Sev', 'Skew(X)'] > 0 and df.loc['Sev', 'Err Skew(X)'] > 100 * eps:
                logger.info('FAIL: Portfolio Sev skew error > eps')
                rv |= Validation.SEV_SKEW

            if np.inf > df.loc['Agg', 'Skew(X)'] > 0 and df.loc['Agg', 'Err Skew(X)'] > 100 * eps:
                logger.info('FAIL: Portfolio Agg skew error > eps')
                rv |= Validation.AGG_SKEW

        except (TypeError, ZeroDivisionError):
            pass

        if rv == Validation.NOT_UNREASONABLE:
            logger.info('Portfolio does not fail any validation: not unreasonable')
        self._valid = rv
        return rv

    def explain_validation(self):
        """
        Explain the validation result. Can pass in if already calculated.
        """
        return explain_validation(self.valid)

    def trim_df(self):
        """
        Trim out unwanted columns from density_df

        epd used in graphics

        :return:
        """
        self.density_df = self.density_df.drop(
            self.density_df.filter(regex='^e_|^exi_xlea|^[a-z_]+ημ').columns,
            axis=1
        )

    @property
    def pprogram(self):
        """
        pretty print the program to html
        """
        return decl_pprint(self.program, 20, show=False)

    @property
    def pprogram_html(self):
        """
        pretty print the program to html
        """
        return decl_pprint(self.program, 0, html=True, show=False)

    def _limits(self, stat='range', kind='linear', zero_mass='include'):
        """
        Suggest sensible plotting limits for kind=range, density, .. (same as Aggregate).

        Should optionally return a locator for plots?

        Called by ploting routines. Single point of failure!

        Must work without ``q`` function when not yet computed.

        :param stat:  range or density or logy (for log density/survival function...ensure consistency)
        :param kind:  linear or log (this is the y-axis, not log of range...that is rarely plotted)
        :param zero_mass:  include exclude, for densities
        :return:
        """
        # fudge l/r factors
        def f(x):
            fl, fr = 0.02, 1.02
            return [-fl * x, fr * x]

        # lower bound for log plots
        eps = 1e-16

        # if not computed have no business asking for limits
        assert self.density_df is not None

        if stat == 'range':
            if kind == 'linear':
                return f(self.q(0.999))
            else:
                # wider range for log density plots
                return f(self.q(1 - 1e-10))
        elif stat == 'density':
            mx = self.density_df.filter(regex='p_[a-zA-Z]').max().max()
            mxx0 = self.density_df.filter(regex='p_[a-zA-Z]').iloc[1:].max().max()
            if kind == 'linear':
                if zero_mass == 'include':
                    return f(mx)
                else:
                    return f(mxx0)
            else:
                return [eps, mx * 1.5]
        elif stat == 'logy':
            mx = min(1, self.density_df.filter(regex='p_[A-Za-z]').max().max())
            return [1e-12, mx * 2]
        else:
            # if you fall through to here, wrong args
            raise ValueError(f'Inadmissible stat/kind passsed, expected range/density and log/linear.')

    def plot(self, axd=None, figsize=(2 * FIG_W, FIG_H)):
        """
        Defualt plot of density, survival functions (linear and log)

        :param axd: dictionary with plots A and B for density and log density
        :param figsize: figure size used by ``plt.subplot_mosaic`` if ``axd`` is not provided
        :return:
        """

        if axd is None:
            self.figure, axd = plt.subplot_mosaic('AB', figsize=figsize, layout='constrained')

        ax = axd['A']
        xl = self._limits()
        yl = self._limits(stat='density', zero_mass='exclude')
        bit = self.density_df.filter(regex='p_[a-zA-Z]')
        if bit.shape[1] == 3:
            # put total first = Book standard
            bit = bit.iloc[:, [2,0,1]]
        bit.plot(ax=ax, xlim=xl, ylim=yl)
        ax.set(xlabel='Loss', ylabel='Density')
        ax.legend()

        ax = axd['B']
        xl = self._limits(kind='log')
        yl = self._limits(stat='logy')
        bit.plot(ax=ax, logy=True, xlim=xl, ylim=yl)
        ax.set(xlabel='Loss', ylabel='Log density')
        ax.legend().set(visible=False)

        # ax = axd['C']
        # self.density_df.filter(regex='p_[a-zA-Z]')[::-1].cumsum().plot(ax=ax, xlim=xl, logy=True)

    def scatter(self, marker='.', s=5, alpha=1, figsize=(10, 10), diagonal='kde', **kwargs):
        """
        Create a scatter plot of marginals against one another, using pandas.plotting scatter_matrix.

        Designed for use with samples. Plots exeqa columns


        """
        bit = self.density_df.query('p_total > 0').filter(regex='exeqa_[a-zA-Z]')
        ax = scatter_matrix(bit, marker='.', s=5, alpha=1,
                            figsize=(10, 10), diagonal='kde', **kwargs)
        return ax

    def add_exa(self, df, ft_nots=None):
        r"""
        Use fft to add exeqa_XXX = E(X_i | X=a) to each dist

        also add exlea = E(X_i | X <= a) = sum_{x<=a} exa(x)*f(x) where f is for the total
        ie. self.density_df['exlea_attrit'] = np.cumsum( self.density_df.exa_attrit *
        self.density_df.p_total) / self.density_df.F

        and add exgta = E(X_i | X>a) since E(X) = E(X | X<= a)F(a) + E(X | X>a)S(a) we have
        exgta = (ex - exlea F) / S

        and add the actual expected losses (not theoretical) the empirical amount:
        self.density_df['e_attrit'] =  np.sum( self.density_df.p_attrit * self.density_df.loss)

        Mid point adjustment is handled by the example creation routines
        self.density_df.loss = self.density_df.loss - bs/2

        **YOU CANNOT HAVE A LINE with a name starting t!!!**

        See LCA_Examples for original code

        Alternative approach to exa: use UC=unconditional versions of exlea and exi_xgta:

        * exleaUC = np.cumsum(port.density_df['exeqa\_' + col] * port.density_df.p_total)  # unconditional
        * exixgtaUC =np.cumsum(  self.density_df.loc[::-1, 'exeqa\_' + col] / self.density_df.loc[::-1, 'loss']
          * self.density_df.loc[::-1, 'p_total'] )
        * exa = exleaUC + exixgtaUC * self.density_df.loss

        :param df: data frame to add to. Initially add_exa was only called by update and wrote to self.density_df. But now
            it is called by gradient too which writes to gradient_df, so we need to pass in this argument
        :param ft_nots: FFTs of the not lines (computed in gradients) so you don't round trip an FFT; gradients needs
            to recompute all the not lines each time around and it is stilly to do that twice
        """

        # eps is used NOT to do silly things when x is so small F(x)< eps
        # below this percentile you do not try to condition on the event!
        # np.finfo(np.float).eps = 2.2204460492503131e-16
        cut_eps = np.finfo(float).eps

        # get this done
        # defuzz(self.density_df, cut_eps)

        # bucket size
        bs = self.bs  # self.density_df['loss'].iloc[1] - self.density_df['loss'].iloc[0]
        # index has already been reset

        # sum of p_total is so important...we will rescale it...
        if not np.all(df.p_total >= 0):
            # have negative densities...get rid of them
            first_neg = df.query(f'p_total < {-cut_eps}')
            logger.log(WL,
                # f'Portfolio.add_exa | p_total has a negative value starting at {first_neg.head()}; NOT setting to zero...')
                f'p_total has {len(first_neg)} negative values; NOT setting to zero...')
        sum_p_total = df.p_total.sum()
        logger.info(f'{self.name}: sum of p_total is 1 - {1 - sum_p_total:12.8e} NOT rescaling.')
        # df.p_total /= sum_p_total
        df['F'] = np.cumsum(df.p_total)
        # this method is terrible because you lose precision for very small values of S
        # df['S'] = 1 - df.F
        # old method
        # df['S'] = np.hstack((df.p_total.to_numpy()[:0:-1].cumsum()[::-1],
        #                      min(df.p_total.iloc[-1],
        #                          max(0, 1. - (df.p_total.sum())))))
        # which was ugly and not quite right because the it ended  ... pn plast  vs  pn+plast, plast
        # Dec 2020
        # TODO: fix and rationalize with Aggregate.density_df; this can't be right, can it?
        #  Fill value is just S(last point)
        #df['S'] =  \
        #    df.p_total.shift(-1, fill_value=min(df.p_total.iloc[-1], max(0, 1. - (df.p_total.sum()))))[::-1].cumsum()[::-1]

        # Jan 2023 update: use the same method as Aggregate.density_df
        # And the end of all our exploring
        # Will be to arrive where we started
        # And know the place for the first time.
        # T.S. Eliot
        # The loss of accuracy is not germaine in portfolio where you have gone through
        # FFTs
        df['S'] = 1 - df.F

        # get rounding errors, S may not go below zero
        logger.info(
            f'Portfolio.add_exa | {self.name}: S <= 0 values has length {len(np.argwhere((df.S <= 0).to_numpy()))}')

        # E(min(X, a))
        # needs to be shifted down by one for the partial integrals....
        # temp = np.hstack((0, np.array(df.iloc[:-1, :].loc[:, 'S'].cumsum())))
        # df['exa_total'] = temp * bs
        # df['exa_total'] = self.cumintegral(df['S'])
        df['exa_total'] = df.S.shift(1, fill_value=0).cumsum() * self.bs
        df['lev_total'] = df['exa_total']

        # $E(X\wedge a)=\int_0^a tf(t)dt + aS(a)$ therefore exlea
        # (EXpected $X$ given Less than or Equal to **a**)
        # $$=E(X \mid X\le a)=\frac{E(X\wedge a)-aS(a)}{F(a)}$$
        df['exlea_total'] = \
            (df.exa_total - df.loss * df.S) / df.F
        # fix very small values
        # don't pretend you know values!
        # find the largest value where exlea_total > loss, which has to be an error
        # 100 bs is a hack, move a little beyond last problem observation
        # from observation looks about right with 1<<16 buckets
        # TODO What is this crap?
        n_ = df.shape[0]
        if n_ < 1100:
            mult = 1
        elif n_ < 15000:
            mult = 10
        else:
            mult = 100
        loss_max = df[['loss', 'exlea_total']].query(' exlea_total>loss ').loss.max()
        if np.isnan(loss_max):
            loss_max = 0
        else:
            loss_max += mult * bs
        # try nan in place of 0             V
        df.loc[0:loss_max, 'exlea_total'] = np.nan
        # df.loc[df.F < 2 * cut_eps, 'exlea_total'] = df.loc[
        #     df.F < 2*cut_eps, 'loss']

        # if F(x)<very small then E(X | X<x) = x, you are certain to be above the threshold
        # this is more stable than dividing by the very small F(x)
        df['e_total'] = np.sum(df.p_total * df.loss)

        df['exgta_total'] = df.loss + (df.e_total - df.exa_total) / df.S
        df['exeqa_total'] = df.loss  # E(X | X=a) = a(!) included for symmetry was exa

        # where is S=0
        Seq0 = (df.S == 0)

        for col in self.line_names:
            # ### Additional Variables
            #
            # * exeqa_line = $E(X_i \mid X=a)$
            # * exlea_line = $E(X_i \mid X\le a)$
            # * e_line = $E(X_i)$
            # * exgta_line = $E(X_i \mid X \ge a)$
            # * exi_x_line = $E(X_i / X \mid X = a)$
            # * and similar for le and gt a
            # * exa_line = $E(X_i(a))$
            # * Price based on same constant ROE formula (later we will do $g$s)

            #
            #
            # THE MONEY CALCULATION
            # EX_i | X=a, E(xi eq a)
            #
            #
            #
            #
            if ft_nots is None:
                df['exeqa_' + col] = \
                    np.real(self.ift(self.ft(df.loss * df['p_' + col]) *
                                    self.ft(df['ημ_' + col]))) / df.p_total
            else:
                df['exeqa_' + col] = \
                    np.real(self.ift(self.ft(df.loss * df['p_' + col]) *
                                    ft_nots[col])) / df.p_total
            # these are unreliable estimates because p_total=0
            # JUNE 25: this makes a difference!
            df.loc[df.p_total < cut_eps, 'exeqa_' + col] = 0

            # E(X_{i, 2nd priority}(a))
            # need the stand alone LEV calc
            # E(min(Xi, a)
            # needs to be shifted down by one for the partial integrals....
            stemp = 1 - df['p_' + col].cumsum()
            # temp = np.hstack((0, stemp.iloc[:-1].cumsum()))
            # df['lev_' + col] = temp * bs
            df['lev_' + col] = stemp.shift(1, fill_value=0).cumsum() * self.bs

            # EX_i | X<= a; temp is used in le and gt calcs
            temp = np.cumsum(df['exeqa_' + col] * df.p_total)
            df['exlea_' + col] = temp / df.F
            # revised version for small losses: do not know this value
            df.loc[0:loss_max, 'exlea_' + col] = 0  # df.loc[0:loss_max, 'loss']

            # constant value, helpful in calculations
            df['e_' + col] = np.sum(df['p_' + col] * df.loss)

            # EX_i | X>a
            df['exgta_' + col] = (df['e_' + col] - temp) / df.S

            # E{X_i / X | X > a}  (note=a is trivial!)
            temp = df.loss.iloc[0]  # loss
            df.loss.iloc[0] = 1  # avoid divide by zero
            # unconditional E(X_i/X)
            df['exi_x_' + col] = np.sum(
                df['exeqa_' + col] * df.p_total / df.loss)
            temp_xi_x = np.cumsum(df['exeqa_' + col] * df.p_total / df.loss)
            df['exi_xlea_' + col] = temp_xi_x / df.F
            df.loc[0, 'exi_xlea_' + col] = 0  # df.F=0 at zero
            # more generally F=0 error:                      V
            df.loc[df.exlea_total == 0, 'exi_xlea_' + col] = 0

            # put value back
            df.loss.iloc[0] = temp
            # this is so important we will calculate it directly rather than the old:
            # df['exi_xgta_' + col] = (df['exi_x_' + col] - temp_xi_x) / df.S
            # the last value is undefined because we know nothing about what happens beyond our array
            # above that we need a shift: > second to last value will only involve the last row (the John Major problem)
            # hence
            # Nov 2020, consider added fill values using same approach as exi_xgtag_[CHANGE NOT MADE, investigate]
            # fill_value = df[f'exeqa_{col}'].iloc[-1] / df.loss.iloc[-1] * df.S.iloc[-1]
            fill_value = np.nan
            # logger.info(f'exi_xgta_ fill_value = {fill_value}')
            df['exi_xgta_' + col] = ((df[f'exeqa_{col}'] / df.loss *
                                      df.p_total).shift(-1, fill_value=fill_value)[
                                     ::-1].cumsum()) / df.S
            # print(df['exi_xgta_' + col].tail())
            # need this NOT to be nan otherwise exa won't come out correctly
            df.loc[Seq0, 'exi_xgta_' + col] = 0.
            # df['exi_xgta_ημ_' + col] = \
            #     (df['exi_x_ημ_' + col] - temp_xi_x_not) / df.S
            # as for line
            # fill_value = df[f'exeqa_ημ_{col}'].iloc[-1] / df.loss.iloc[-1] * df.S.iloc[-1]

            df['exi_xeqa_' + col] = df['exeqa_' + col] / df['loss']
            df.loc[0, 'exi_xeqa_' + col] = 0

            # need the loss cost with equal priority rule
            # exa_ = E(X_i(a)) = E(X_i | X<= a)F(a) + E(X_i / X| X>a) a S(a)
            #   = exlea F(a) + exixgta * a * S(a)
            # and hence get loss cost for line i
            # df['exa_' + col] = \
            #     df['exlea_' + col] * df.F + df.loss * \
            #     df.S * df['exi_xgta_' + col]
            # df['exa_ημ_' + col] = \
            #     df['exlea_ημ_' + col] * df.F + df.loss * \
            #     df.S * df['exi_xgta_ημ_' + col]
            # alt calc using S: validated the same, going with this as a more direct calc
            df[f'exa_{col}'] = (df.S * df['exi_xgta_' + col]).shift(1, fill_value=0).cumsum() * self.bs

        # put in totals for the ratios... this is very handy in later use
        for metric in ['exi_xlea_', 'exi_xgta_', 'exi_xeqa_']:
            df[metric + 'sum'] = df.filter(regex=metric + '[^η]').sum(axis=1)

    def ft(self, x):
        """
        FT of x with padding applied
        """
        return ft(x, self.padding)

    def ift(self, x):
        """
        IFT of x with padding applied
        """
        return ift(x, self.padding)

    def add_eta_mu(self):
        """ convenience function to just add the eta-mus. """
        self.add_exa_details(self.density_df, eta_mu='only')

    def add_exa_details(self, df, eta_mu=False):
        """
        From add_exa, details for epd functions and eta_mu flavors.

        Note ``eta_mu=True`` is required for ``epd_2`` functions.

        """
        cut_eps = np.finfo(float).eps
        bs = self.bs
        # epds for total on a standalone basis (all that makes sense)
        df['epd_0_total'] = \
            np.maximum(0, (df['e_total'] - df['lev_total'])) / \
            df['e_total']        # E[1/X 1_{X>a}] used for reimbursement effectiveness graph
        # Nov 2020 tweaks
        index_inv = 1.0 / df.loss
        df['e1xi_1gta_total'] = (df['p_total'] * index_inv).shift(-1)[::-1].cumsum()

        # TODO What is this crap?
        n_ = df.shape[0]
        if n_ < 1100:
            mult = 1
        elif n_ < 15000:
            mult = 10
        else:
            mult = 100
        loss_max = df[['loss', 'exlea_total']].query(' exlea_total>loss ').loss.max()
        if np.isnan(loss_max):
            loss_max = 0
        else:
            loss_max += mult * bs
        # where is S=0
        Seq0 = (df.S == 0)

        # will need two decorators for epd functions: these handle swapping the arguments and
        # protecting against value errors
        def minus_arg_wrapper(a_func):
            def new_fun(x):
                try:
                    x = a_func(-x)
                except ValueError:
                    x = 999
                return x

            return new_fun

        def minus_ans_wrapper(a_func):
            def new_fun(x):
                try:
                    x = -a_func(x)
                except ValueError:
                    x = 999
                return x

            return new_fun

        if eta_mu:
            # recrecate the ημ columns (NO padding!)
            ft_line_density = {}
            ft_all = ft(self.density_df.p_total, self.padding)
            for line in self.line_names:
                # create all for inner loop below
                ft_line_density[line] = ft(self.density_df[f'p_{line}'], self.padding)
            for line in self.line_names:
                ft_not = np.ones_like(ft_all)
                # this fails because the ft can contain very small quantites
                # if np.any(ft_line_density[line] == 0):
                # more robust test (tried 2 * np.finfo(float).eps but that failed)
                if np.abs(ft_line_density[line]).min() < 1e-10:
                    # have to build up
                    for not_line in self.line_names:
                        if not_line != line:
                            ft_not *= ft_line_density[not_line]
                else:
                    if len(self.line_names) > 1:
                        ft_not = ft_all / ft_line_density[line]
                self.density_df[f'ημ_{line}'] = np.real(ift(ft_not, self.padding))

        if eta_mu == 'only':
            return

        for col in self.line_names:
            # fill in ημ
            if eta_mu:
                # rarely if ever ussed; not greatly tested
                df['exeqa_ημ_' + col] = \
                    np.real(self.ift(self.ft(df.loss * df['ημ_' + col]) *
                                    self.ft(df['p_' + col]))) / df.p_total
                # these are unreliable estimates because p_total=0 JUNE 25: this makes a difference!
                df.loc[df.p_total < cut_eps, 'exeqa_ημ_' + col] = 0

                df['e2pri_' + col] = \
                    np.real(self.ift(self.ft(df['lev_' + col]) * self.ft(df['ημ_' + col])))

                stemp = 1 - df['ημ_' + col].cumsum()
                # temp = np.hstack((0, stemp.iloc[:-1].cumsum()))
                # df['lev_ημ_' + col] = temp * bs
                df['lev_ημ_' + col] = stemp.shift(1, fill_value=0).cumsum() * self.bs

                temp_not = np.cumsum(df['exeqa_ημ_' + col] * df.p_total)
                df['exlea_ημ_' + col] = temp_not / df.F
                # revised version for small losses: do not know this value
                df.loc[0:loss_max, 'exlea_ημ_' + col] = 0  # df.loc[0:loss_max, 'loss']

                df['e_ημ_' + col] = np.sum(df['ημ_' + col] * df.loss)

                # not version
                df['exi_x_ημ_' + col] = np.sum(
                    df['exeqa_ημ_' + col] * df.p_total / df.loss)
                # as above
                temp_xi_x_not = np.cumsum(
                    df['exeqa_ημ_' + col] * df.p_total / df.loss)
                df['exi_xlea_ημ_' + col] = temp_xi_x_not / df.F
                df.loc[0, 'exi_xlea_ημ_' + col] = 0  # df.F=0 at zero
                # more generally F=0 error:
                df.loc[df.exlea_total == 0, 'exi_xlea_ημ_' + col] = 0

                # per about line 2052 above
                fill_value = np.nan
                df['exi_xgta_ημ_' + col] = ((df[f'exeqa_ημ_{col}'] / df.loss *
                                         df.p_total).shift(-1, fill_value=fill_value)[
                                        ::-1].cumsum()) / df.S
                df.loc[Seq0, 'exi_xgta_ημ_' + col] = 0.

                df['exi_xeqa_ημ_' + col] = df['exeqa_ημ_' + col] / df['loss']
                df.loc[0, 'exi_xeqa_ημ_' + col] = 0

                df['exa_ημ_' + col] = (df.S * df['exi_xgta_ημ_' + col]).shift(1, fill_value=0).cumsum() * self.bs

            # E[1/X 1_{X>a}] used for reimbursement effectiveness graph
            # Nov 2020
            # df['e1xi_1gta_total'] = (df['p_total'] * index_inv).shift(-1)[::-1].cumsum()
            df[f'e1xi_1gta_{col}'] = (df[f'p_{col}'] * index_inv).shift(-1)[::-1].cumsum()

            # epds
            df['epd_0_' + col] = \
                np.maximum(0, (df['e_' + col] - df['lev_' + col])) / \
                df['e_' + col]
            df['epd_1_' + col] = \
                np.maximum(0, (df['e_' + col] - df['exa_' + col])) / \
                df['e_' + col]

            if eta_mu:
                df['epd_2_' + col] = \
                    np.maximum(0, (df['e_' + col] - df['e2pri_' + col])) / \
                    df['e_' + col]
                df['epd_0_ημ_' + col] = \
                    np.maximum(0, (df['e_ημ_' + col] - df['lev_ημ_' + col])) / \
                    df['e_ημ_' + col]
                df['epd_1_ημ_' + col] = \
                    np.maximum(0, (df['e_ημ_' + col] -
                               df['exa_ημ_' + col])) / \
                    df['e_ημ_' + col]

            # EPD interpolation functions removed in the refactor; EPD-based
            # allocation and priority code went with them (Sub-project A).

    def calibrate_distortion(self, name, r0=0.0, df=[0.0, .9], premium_target=0.0,
                             roe=0.0, assets=0.0, p=0.0, kind='lower', S_column='S',
                             S_calc='cumsum'):
        """
        Find a distortion transform to hit a premium target at the given
        asset level.

        Portfolio.calibrate_distortion has been reduced to (a) resolving
        the asset level / premium target / S vector / ``ess_sup`` from the
        Portfolio state and (b) dispatching to the appropriate
        ``Distortion`` subclass, whose ``calibrate`` method runs the
        Newton iteration. The per-distortion math (the Newton ``f``
        closures) now lives on each subclass in :mod:`aggregate.spectral`.

        Parameters
        ----------
        name : str
            Distortion kind (``ph``, ``wang``, ``dual``, ``tvar``,
            ``ccoc`` / ``roe``, ``ly``, ``clin``, ``lep``, ``cll``).
        r0 : float, optional
            Fixed parameter for mass-at-zero kinds.
        df : list, optional
            Kind-specific second parameter (kept on the returned
            distortion for audit; unused by the migrated kinds).
        premium_target : float, optional
            Target premium. If 0, derived from ``roe`` and ``assets``.
        roe : float, optional
            Used to derive ``premium_target`` when not supplied.
        assets : float, optional
            Asset level. If 0, derived from ``p`` via ``self.q``.
        p : float, optional
            Probability used to derive ``assets`` via the quantile.
        kind : str
            Quantile interpolation kind for ``self.q``.
        S_column, S_calc : str
            Which column / method to use to construct ``S``; see existing
            callers.

        Returns
        -------
        Distortion
            Calibrated distortion with ``shape``, ``error``, ``assets``,
            and ``premium_target`` set.
        """
        assert S_calc in ('S', 'cumsum')

        if S_column == 'S':
            if assets == 0:
                assert (p > 0)
                assets = self.q(p, kind)
            el = self.density_df.loc[assets, 'exa_total']
            if premium_target == 0:
                assert (roe > 0)
                premium_target = (el + roe * assets) / (1 + roe)
        else:
            # calibrating to unlimited premium; let code trim S at max loss
            if assets == 0:
                assets = self.density_df.loss.iloc[-1]
            el = self.density_df.loc[assets, 'exa_total']

        # extract S over [0, assets]; integration is inclusive of endpoint
        if S_calc == 'S':
            Splus = self.density_df.loc[0:assets, S_column].values
        else:
            Splus = (1 - self.density_df.loc[0:assets, 'p_total'].cumsum()).values

        last_non_zero = np.argwhere(Splus)
        ess_sup = 0
        if len(last_non_zero) == 0:
            last_non_zero = len(Splus) + 1
        else:
            last_non_zero = last_non_zero.max()
        if last_non_zero + 1 < len(Splus):
            # truncate at first zero
            S = Splus[:last_non_zero + 1]
            ess_sup = self.density_df.index[last_non_zero + 1]
            logger.info(
                'Portfolio.calibrate_distortion | Mass issues in calibrate_distortion...'
                f'{name} at {last_non_zero}, loss = {ess_sup}')
        else:
            if S_calc == 'original':
                S = self.density_df.loc[0:assets - self.bs, S_column].values
            else:
                S = (1 - self.density_df.loc[0:assets - self.bs, 'p_total'].cumsum()).values

        # S must be strictly positive and weakly decreasing
        assert np.all(S > 0) and np.all(S[:-1] >= S[1:])

        # dispatch to the subclass that owns this kind's calibration
        lookup = 'ccoc' if name == 'roe' else name
        subclass = Distortion._registry.get(lookup)
        if subclass is None or subclass._calibration_init_shape is None:
            raise ValueError(
                f'calibrate_distortion not implemented for {name}')
        dist = Distortion(name=name, shape=subclass._calibration_init_shape,
                          r0=r0, df=df)
        dist.calibrate(S=S, bs=self.bs, premium_target=premium_target,
                       ess_sup=ess_sup, assets=assets, el=el)
        return dist

    def calibrate_distortions(self, coc, *, p=None, a=None,
                              r0=0.03, df=5.5, kind='lower'):
        """
        Calibrate the standard pricing distortion set to a cost-of-capital target.

        Parameters
        ----------
        coc : float
            Target cost of capital ``COC = (P - L) / Q``.
        p : float, optional
            Probability at which the calibration applies; converted to asset
            level via ``self.q(p, kind)``. Exactly one of ``p`` or ``a`` must
            be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.
        r0 : float, optional
            ``r0`` parameter for distortions with a minimum rate-on-line
            (``ly``, ``clin``, ``lep``). Default 0.03.
        df : float, optional
            Degrees-of-freedom for ``tt``. Default 5.5.
        kind : {'lower', 'upper'}, optional
            VaR kind when ``p`` is provided. Default ``'lower'``.

        Returns
        -------
        pandas.DataFrame
            Calibration audit table (one row per distortion in
            ``[ccoc, ph, wang, dual, tvar]``) with columns
            ``[S, L, P, PQ, Q, COC, param, std_param, error]`` and
            MultiIndex ``(a, LR, method)``. ``method`` is an ordered
            categorical so sorts produce the canonical distortion order.
            Also stored on ``self.distortion_df``. The calibrated
            distortion objects are stored on ``self.distortions`` keyed
            by name.

        Notes
        -----
        Replaces both the legacy
        ``calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, ...)`` and
        ``calibrate_distortions2(coc, reg_p)``. 
        """
        if (p is None) == (a is None):
            raise ValueError(
                'calibrate_distortions requires exactly one of p= (probability) '
                'or a= (asset level).')
        if a is None:
            a = self.q(p, kind)
        else:
            a = self.snap(a)
        exa, S = self.density_df.loc[a, ['exa_total', 'S']]
        # invert COC -> LR -> P (matches the legacy ROE -> LR -> P path).
        delta = coc / (1 + coc)
        nu = 1 - delta
        P = nu * exa + delta * a
        LR = exa / P
        profit = P - exa
        K = a - P
        d_list = ['ccoc', 'ph', 'wang', 'dual', 'tvar']
        rows = []
        distortions = {}
        for dname in d_list:
            dist = self.calibrate_distortion(
                name=dname, r0=r0, df=df, premium_target=P, assets=a)
            distortions[dname] = dist
            rows.append(
                [S, exa, P, P / K, K, profit / K,
                 dist.shape, dist.standard_shape, dist.error])
        distortion_df = pd.DataFrame(
            rows,
            columns=['S', 'L', 'P', 'PQ', 'Q', 'COC',
                     'param', 'std_param', 'error'],
        )
        distortion_df.index = pd.MultiIndex.from_arrays(
            [[a] * len(d_list),
             [LR] * len(d_list),
             pd.Categorical(d_list, dtype=DISTORTION_DTYPE)],
            names=['a', 'LR', 'method'],
        )
        self.distortion_df = distortion_df
        self.distortions = distortions
        return distortion_df

    def apply_distortion(self, distortion, *, view='ask', S_calculation='forwards', efficient=True):
        """
        Apply ``distortion`` and return the resulting augmented_df.

        Results are cached on ``self._augmented_dfs`` keyed by distortion name. A
        second call with the same distortion is an O(1) dict lookup; the returned
        DataFrame is the same object (``is``-identical) as the prior call.

        Parameters
        ----------
        distortion : Distortion or str
            A ``Distortion`` instance, or the name of a previously calibrated
            distortion (looked up in ``self.distortions``).
        view : {'ask', 'bid'}
            Pricing view. Default 'ask'.
        S_calculation : {'forwards', 'backwards'}
            How to (re)compute the total survival ``S``. Default 'forwards' --
            recompute from ``1 - p_total.cumsum()`` to keep the tail accurate.
        efficient : bool
            If True (the default) compute only the columns needed for pricing
            (T.* series). If False, also build the M.* marginal columns.

        Returns
        -------
        pandas.DataFrame
            The cached ``augmented_df`` for this distortion.

        Notes
        -----
        The actual construction lives in ``_build_augmented``. The cache is
        invalidated whenever ``update`` is called (the underlying density
        changes).
        """
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        name = distortion.name
        if name not in self._augmented_dfs:
            self._augmented_dfs[name] = self._build_augmented(
                distortion, view=view, S_calculation=S_calculation, efficient=efficient)
        self._distortion = distortion
        self._last_applied_distortion_name = name
        return self._augmented_dfs[name]

    def augmented_df(self, distortion):
        """
        Return the cached augmented_df for ``distortion`` (building it on demand).

        Identical to ``apply_distortion(distortion)`` with default kwargs --
        provided as the clean read-side accessor.
        """
        return self.apply_distortion(distortion)

    @property
    def augmented_dfs(self):
        """
        The augmented_df cache as a dict ``{distortion_name: DataFrame}``.

        Read-only view -- mutate via ``apply_distortion`` (insert) or
        ``update`` (clear).
        """
        return self._augmented_dfs

    def pricing_at(self, distortion, *, p=None, a=None):
        """
        Extract per-line pricing readout at probability ``p`` or asset level ``a``.

        Warms the augmented_df cache for ``distortion``, then pulls the
        L/LR/M/P/PQ/Q/ROE row at the requested asset level.

        Parameters
        ----------
        distortion : Distortion or str
            Passed through to ``apply_distortion``.
        p : float, optional
            Probability; converted to asset level via ``self.q(p)``. Exactly
            one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.

        Returns
        -------
        pandas.DataFrame
            Rows indexed by line (units + 'total'), columns
            ``['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE']``.

        Notes
        -----
        Consolidates row-extraction logic that previously lived in ``price``
        (lifted allocation) and ``analyze_distortion``.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'pricing_at requires exactly one of p= (probability) '
                'or a= (asset level).')
        if a is None:
            a = self.q(p)
        else:
            a = self.snap(a)
        aug = self.apply_distortion(distortion)
        if a in aug.index:
            row = aug.loc[a]
        else:
            logger.warning(
                f'pricing_at: asset level {a} not in augmented_df.index; using last row.')
            row = aug.iloc[-1]
        lines = list(self.line_names_ex)
        out = pd.DataFrame(
            index=lines,
            columns=['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE'],
            dtype=float,
        )
        out.index.name = 'line'
        for line in lines:
            L = row[f'exa_{line}']
            P = row[f'exag_{line}']
            M = row[f'T.M_{line}']
            Q = row[f'T.Q_{line}']
            out.loc[line, 'L'] = L
            out.loc[line, 'P'] = P
            out.loc[line, 'M'] = M
            out.loc[line, 'Q'] = Q
        # exact Q_total = a - exag_total beats the layer-by-layer cumsum,
        # which can drift by a few buckets in the tail (matches the legacy
        # ``analyze_distortion`` correction).
        out.loc['total', 'Q'] = a - row['exag_total']
        out['LR'] = out['L'] / out['P']
        out['PQ'] = out['P'] / out['Q']
        out['ROE'] = out['M'] / out['Q']
        # bake the canonical pricing-stat order into the columns
        out.columns = pd.CategoricalIndex(
            out.columns, dtype=PRICING_STAT_DTYPE, name='stat')
        return out

    def _build_augmented(self, dist, *, view='ask', S_calculation='forwards', efficient=True):
        """
        Construct an augmented_df from ``self.density_df`` under ``dist``.

        Pure builder: returns the frame without touching ``self`` (the
        ``apply_distortion`` wrapper writes it into the cache). The body
        is the former ``apply_distortion`` minus the ``df_in`` / ``plots`` /
        ``create_augmented`` branches.
        """

        df = self.density_df.copy()

        # PREVIOUSLY: did not make this adjustment because loss of resolution on small S values
        # however, it doesn't work well for very thick tailed distributions, hence intro of S_calculation
        # July 2020 (during COVID-Trump madness) try this instead
        if S_calculation == 'forwards':
            logger.debug('Using updated S_forwards calculation in apply_distortion! ')
            df['S'] = 1 - df.p_total.cumsum()

        # make g and ginv and other interpolation functions
        if view == 'bid':
            g = dist.g_dual
            g_prime = lambda x: dist.g_prime(1 - x)
        elif view == 'ask':
            g = dist.g
            g_prime = dist.g_prime
        else:
            raise ValueError(f'kind must be bid or ask, not {view}')

        # maybe important that p_total sums to 1
        # this appeared not to make a difference, and introduces an undesirable difference from
        # the original density_df
        # df.loc[df.p_total < 0, :] = 0
        # df['p_total'] = df['p_total'] / df['p_total'].sum()
        # df['F'] = df.p_total.cumsum()
        # df['S'] = 1 - df.F

        # floating point issues: THIS HAPPENS, so needs to be corrected...
        cut_eps = np.finfo(float).eps
        if len(df.loc[df.S < 0, 'S'] > 0):
            logger.log(WL, f"{len(df.loc[df.S < -cut_eps, 'S'] > 0)} negative S < -eps values being set to zero...")
            # logger.log(WL, f"{len(df.loc[df.S < 0, 'S'] > 0)} negative S values being set to zero...")
        df.loc[df.S < 0, 'S'] = 0

        # add the exag and distorted probs
        df['gS'] = g(df.S)
        df['gF'] = 1 - df.gS
        # updated for ability to prepend 0 in newer numpy
        # df['gp_total'] = np.diff(np.hstack((0, df.gF)))
        # also checked these two method give the same result:
        # bit2['gp1'] = -np.diff(bit2.gS, prepend=1)
        # bit2['gp2'] = np.diff(1 - bit2.gS, prepend=0)
        # validated using grt.test_df(): this is correct
        # t = grt.test_df(20,2)
        # t.A = t.A / t.A.sum()
        #
        # t['B'] = t.A.shift(-1, fill_value=0)[::-1].cumsum()
        # t['C'] = -np.diff(t.B, prepend=1)
        # t
        df['gp_total'] = -np.diff(df.gS, prepend=1)
        # weirdness occurs here were 0 appears as -0: get rid of that
        df.loc[df.gp_total==0, 'gp_total'] = 0.0

        # figure out where to truncate df (which turns into augmented_df)
        lnp = '|'.join(self.line_names)
        # in discrete distributions there are "gaps" of impossible values; do not want to worry about these
        idx_pne0 = df.query(' p_total > 0 ').index
        exeqa_err = np.abs(
            (df.loc[idx_pne0].filter(regex=f'exeqa_({lnp})').sum(axis=1) - df.loc[idx_pne0].loss) /
            df.loc[idx_pne0].loss)
        exeqa_err.iloc[0] = 0
        # print(exeqa_err)
        # +1 because when you iloc you lose the last element (two code lines down)
        idx = int(exeqa_err[exeqa_err < 1e-4].index[-1] / self.bs + 1)
        # idx = np.argmax(exeqa_err > 1e-4)
        logger.debug(f'index of max reliable value = {idx}')
        if idx:
            # if exeqa_err > 1e-4 is empty, np.argmax returns zero...do not want to truncate at zero in that case
            df = df.iloc[:idx]
        # where gS==0, which should be empty set
        gSeq0 = (df.gS == 0)
        logger.debug(f'S==0 values: {df.gS.loc[gSeq0]}')
        if idx:
            logger.debug(f'Truncating augmented_df at idx={idx}, loss={idx*self.bs}, len(S==0) = {np.sum(gSeq0)} elements')
            # print(f'Truncating augmented_df at idx={idx}, loss={idx*self.bs}\nS==0 on len(S==0) = {np.sum(gSeq0)} elements')
        else:
            logger.debug(f'augmented_df not truncated (no exeqa error), len(S==0) = {np.sum(gSeq0)} elements')
            # print(f'augmented_df not truncated (no exeqa error\nS==0 on len(S==0) = {np.sum(gSeq0)} elements')

        # S better be decreasing
        if not np.all(df.S.iloc[1:] <= df.S.iloc[:-1].values):
            logger.error('S = density_df.S is not non-increasing...carrying on but you should investigate...')

        for line in self.line_names:
            # avoid double count: going up sum needs to be stepped one back, hence use cumintegral is perfect
            # for <=a cumintegral,  for > a reverse and use cumsum (no step back)
            # UC = unconditional
            # old
            #
            # exleaUC = self.cumintegral(self.density_df[f'exeqa_{line}'] * df.gp_total, 1)
            #
            # correct that increasing integrals need the offset
            # exleaUC1 = np.cumsum(self.density_df[f'exeqa_{line}'] * df.gp_total)
            #
            # old
            # exixgtaUC = np.cumsum(
            #    self.density_df.loc[::-1, f'exeqa_{line}'] / self.density_df.loc[::-1, 'loss'] *
            #    df.loc[::-1, 'gp_total'])
            #
            # or shift...NO should be cumsum for gt
            # exixgtaUC1 = self.cumintegral(
            #     self.density_df.loc[::-1, f'exeqa_{line}'] / self.density_df.loc[::-1, 'loss'] *
            #     df.loc[::-1, 'gp_total'], 1)[::-1]
            #
            # when computed using np.cumsum exixgtaUC is a pd.Series has an index so when it is mult by .loss
            # (which also has an index) it gets re-sorted into ascending order
            # when computed using cumintegral it is a numpy array with no index and so need reversing
            # the difference between UC and UC1 is a shift up by 1.
            #
            # Here is a little tester example to show what goes on
            #
            # test = pd.DataFrame(dict(x=range(20)))
            # test['a'] = 10 * test.x
            # test['y'] = test.x * 3 + 5
            # bit = np.cumsum(test['y'][::-1])
            # test['z'] = bit
            # test['w'] = bit / test.a
            # test
            #            #
            # again, exi_xgtag is super important, so we will compute it bare bones the same way as exi_xgta
            # df[f'exi_xgtag_{line}'] = exixgtaUC / df.gS
            #
            #
            # df['exi_xgtag_' + line] = ((df[f'exeqa_{line}'] / df.loss *
            #                             df.gp_total).shift(-1)[::-1].cumsum()) / df.gp_total.shift(-1)[::-1].cumsum()
            # exa uses S in the denominator...and actually checking values there is a difference between the sum and gS
            #
            # mass hints removed
            # original
            # df['exi_xgtag_' + line] = ((df[f'exeqa_{line}'] / df.loss *
            #             df.gp_total).shift(-1)[::-1].cumsum()) / df.gS
            # df['exi_xgtag_ημ_' + line] = ((df[f'exeqa_ημ_{line}'] / df.loss *
            #                                df.gp_total).shift(-1)[::-1].cumsum()) / df.gS
            # Nov 2020
            # shift[-1] because result is for > a, so the first entry sums from bucket 1...
            # need the denominator to equal the numerator sum of p values
            # the shift up needs to be filled in with the last S value (for the tail) otherwise that probability
            # is lost...hence we need to set fill_values:
            last_gS = df.gS.iloc[-1]
            last_x = df[f'exeqa_{line}'].iloc[-1] / df.loss.iloc[-1] * last_gS
            logger.debug(f'Tail adjustment for {line}: {last_x:.6g}')
            df['exi_xgtag_' + line] = \
                ((df[f'exeqa_{line}'] / df.loss * df.gp_total).
                    shift(-1, fill_value=last_x)[::-1].cumsum()) / df.gS
            # need these to be zero so nan's do not propagate
            df.loc[gSeq0, 'exi_xgtag_' + line] = 0.
            if not efficient:
                last_x = df[f'exeqa_ημ_{line}'].iloc[-1] / df.loss.iloc[-1] * last_gS
                df['exi_xgtag_ημ_' + line] = \
                    ((df[f'exeqa_ημ_{line}'] / df.loss * df.gp_total).
                    shift(-1, fill_value=last_x)[::-1].cumsum()) / df.gS
                df.loc[gSeq0, 'exi_xgtag_ημ_' + line] = 0.
            #
            #
            # following the Audit Vignette this is the way to go:
            # in fact, need to shift both down because cumint (prev just gS, but int beta g...integrands on same
            # basis
            # df[f'exag_{line}'] = (df[f'exi_xgtag_{line}'].shift(1) * df.gS.shift(1)).cumsum() * self.bs
            # df[f'exag_ημ_{line}'] = (df[f'exi_xgtag_ημ_{line}'].shift(1) * df.gS.shift(1)).cumsum() * self.bs
            # np.allclose(
            #    (df[f'exi_xgtag_{line}'].shift(1, fill_value=0) * df.gS.shift(1, fill_value=0)).cumsum() * self.bs,
            #     (df[f'exi_xgtag_{line}'] * df.gS).shift(1, fill_value=0).cumsum() * self.bs)
            # Doh
            df[f'exag_{line}'] = (df[f'exi_xgtag_{line}'] * df.gS).shift(1, fill_value=0).cumsum() * self.bs
            if not efficient:
                df[f'exag_ημ_{line}'] = (df[f'exi_xgtag_ημ_{line}'] * df.gS).shift(1, fill_value=0).cumsum() * self.bs
            # maybe sometime you want this unchecked item?
            # df[f'exleag_1{line}'] = np.cumsum( df[f'exeqa_{line}'] * df.p_total )
            # it makes a difference NOT to divivde by df.gS but to compute the actual weights you are using (you mess
            # around with the last weight)
            #
            #
            #
            # these are all here for debugging...see
            # C:\S\TELOS\spectral_risk_measures_monograph\spreadsheets\[AS_IJW_example.xlsx]
            # df[f'exag1_{line}'] = exleaUC + exixgtaUC1 * self.density_df.loss + mass
            # df[f'exi_xgtag1_{line}'] = exixgtaUC1 / df.gS
            # df[f'exleaUC_{line}'] = exleaUC
            # df[f'exleaUCcs_{line}'] = exleaUCcs
            # df[f'U_{line}'] = exixgtaUC
            # df[f'U1_{line}'] = exixgtaUC1
            # df[f'RAW_{line}'] = self.density_df.loc[::-1, f'exeqa_{line}'] / self.density_df.loc[::-1, 'loss'] * \
            #     df.loc[::-1, 'gp_total']

        if efficient:
            # need to get to T.M and T.Q for pricing... laser in on those...
            # duplicated and edited code from below

            df['exag_total'] = df.gS.shift(1, fill_value=0).cumsum() * self.bs
            df['M.M_total'] = (df.gS - df.S)
            df['M.Q_total'] = (1 - df.gS)
            # hummmm.aliases, but...?
            # df['M.L_total'] = df['S']
            # df['M.P_total'] = df['gS']
            # df['T.L_total'] = df['exa_total']
            # df['T.P_total'] = df['exag_total']
            # critical insight is the layer ROEs are the same for all lines by law invariance
            # lhopital's rule estimate of g'(1) = ROE(1)
            # this could blow up...
            ϵ = 1e-4
            gprime1 = g_prime(1) # (g(1 - ϵ) - (1 - ϵ)) / (1 - g(1 - ϵ))
            df['M.ROE_total'] = np.where(df['M.Q_total']!=0,
                                                df['M.M_total'] / df['M.Q_total'],
                                                gprime1)
            # where is the ROE zero? need to handle separately else Q will blow up
            roe_zero = (df['M.ROE_total'] == 0.0)

            # print(f"g'(0)={gprime1:.5f}\nroe zero vector {roe_zero}")
            for line in self.line_names_ex:
                df[f'T.M_{line}'] = df[f'exag_{line}'] - df[f'exa_{line}']

                mm_l = df[f'T.M_{line}'].diff().shift(-1) / self.bs
                # careful about where ROE==0
                mq_l = mm_l / df['M.ROE_total']
                mq_l.iloc[-1] = 0
                mq_l.loc[roe_zero] = np.nan
                df[f'T.Q_{line}'] = mq_l.shift(1).cumsum() * self.bs
                df.loc[0, f'T.Q_{line}'] = 0

            return df

        # sum of parts: careful not to include the total twice!
        # not used
        # df['exag_sumparts'] = df.filter(regex='^exag_[^η]').sum(axis=1)
        # LEV under distortion g
        # originally
        # df['exag_total'] = self.cumintegral(df['gS'])
        # revised  cumintegral does: v.shift(1, fill_value=0).cumsum() * bs
        df['exag_total'] = df.gS.shift(1, fill_value=0).cumsum() * self.bs
        # df.loc[0, 'exag_total'] = 0

        # comparison of total and sum of parts
        # Dec 2019 added info to compute the total margin and capital allocation by layer
        # [MT].[L LR Q P M]_line: marginal or total (ground-up) loss, loss ratio etc.
        # do NOT divide MARGINAL versions by bs because they are per 1 wide layer
        # df['lookslikemmtotalxx'] = (df.gS - df.S)
        df['M.M_total'] = (df.gS - df.S)
        df['M.Q_total'] = (1 - df.gS)
        # hummmm.aliases, but...?
        df['M.L_total'] = df['S']
        df['M.P_total'] = df['gS']
        # df['T.L_total'] = df['exa_total']
        # df['T.P_total'] = df['exag_total']
        # critical insight is the layer ROEs are the same for all lines by law invariance
        # lhopital's rule estimate of ROE(1) estimates (gs-s)/(1-gs) = g's - 1)/(-g's) =
        # 1 / g's - 1
        # this can be infinite...
        gprime1 = 1 / g_prime(1) - 1 # (g(1 - ϵ) - (1 - ϵ)) / (1 - g(1 - ϵ))
        df['M.ROE_total'] = np.where(df['M.Q_total']!=0,
                                            df['M.M_total'] / df['M.Q_total'],
                                            gprime1)
        # where is the ROE zero? need to handle separately else Q will blow up
        roe_zero = (df['M.ROE_total'] == 0.0)

        # print(f"g'(0)={gprime1:.5f}\nroe zero vector {roe_zero}")
        for line in self.line_names_ex:

            # these are not used
            # df[f'exa_{line}_pcttotal'] = df['exa_' + line] / df.exa_total
            # df[f'exag_{line}_pcttotal'] = df['exag_' + line] / df.exag_total
            # hummm more aliases
            df[f'T.L_{line}'] = df[f'exa_{line}']
            df[f'T.P_{line}'] = df[f'exag_{line}']
            df.loc[0, f'T.P_{line}'] = 0
            # TOTALs = ground up cumulative sums
            # exag is the layer (marginal) premium and exa is the layer (marginal) loss
            df[f'T.LR_{line}'] = df[f'exa_{line}'] / df[f'exag_{line}']
            df[f'T.M_{line}'] = df[f'exag_{line}'] - df[f'exa_{line}']
            df.loc[0, f'T.M_{line}'] = 0
            # MARGINALs
            # MM should be per unit width layer so need to divide by bs
            # prepend=0 satisfies: if
            # d['B'] = d.A.cumsum()
            # d['C'] = np.diff(d.B, prepend=0)
            # then d.C == d.A, which is what you want.
            # note this overwrites M.M_total set above
            # T.M starts at zero, by previous line and sense: no assets ==> no prem or loss
            # old
            # df[f'M.M_{line}'] = np.diff(df[f'T.M_{line}'], prepend=0) / self.bs
            # new:
            df[f'M.M_{line}'] = df[f'T.M_{line}'].diff().shift(-1) / self.bs
            # careful about where ROE==0
            df[f'M.Q_{line}'] = df[f'M.M_{line}'] / df['M.ROE_total']
            df[f'M.Q_{line}'].iloc[-1] = 0
            df.loc[roe_zero, f'M.Q_{line}'] = np.nan
            # WHAT IS THE LAYER AT ZERO? Should it have a price? What is that price?
            # TL and TP at zero are both 1
            if line != 'total':
                df[f'M.L_{line}'] = df[f'exi_xgta_{line}'] * df['S']
                df[f'M.P_{line}'] = df[f'exi_xgtag_{line}'] * df['gS']
            df[f'M.LR_{line}'] = df[f'M.L_{line}'] / df[f'M.P_{line}']
            # for total need to reflect layer width...
            # Jan 2020 added shift down
            df[f'T.Q_{line}'] = df[f'M.Q_{line}'].shift(1).cumsum() * self.bs
            df.loc[0, f'T.Q_{line}'] = 0
            df[f'T.ROE_{line}'] = df[f'T.M_{line}'] / df[f'T.Q_{line}']
            # leverage
            df[f'T.PQ_{line}'] = df[f'T.P_{line}'] / df[f'T.Q_{line}']
            df[f'M.PQ_{line}'] = df[f'M.P_{line}'] / df[f'M.Q_{line}']

        # in order to undo some numerical issues, things will slightly not add up
        # but that appears to be a numerical issue around the calc of exag
        df['T.L_total'] = df['exa_total']
        df['T.P_total'] = df['exag_total']
        df['T.Q_total'] = df.loss - df['exag_total']
        df['T.M_total'] = df['exag_total'] - df['exa_total']
        df['T.PQ_total'] = df['T.P_total'] / df['T.Q_total']
        df['T.LR_total'] = df['T.L_total'] / df['T.P_total']
        df['T.ROE_total'] = df['T.M_total'] / df['T.Q_total']

        return df

    def var_dict(self, p, kind='lower', total='total', snap=False):
        """
        make a dictionary of value at risks for each line and the whole portfolio.

         Returns: {line : var(p, kind)} and includes the total as self.name line

        if p near 1 and epd uses 1-p.

        Example:

            for p, arg in zip([.996, .996, .996, .985, .01], ['var', 'lower', 'upper', 'tvar', 'epd']):
                print(port.var_dict(p, arg,  snap=True))

        :param p:
        :param kind: var (defaults to lower), upper, lower, tvar, epd
        :param total: name for total: total=='name' gives total name self.name
        :param snap: snap tvars to index
        :return:
        """
        if kind == 'var':
            kind = 'lower'

        if kind == 'tvar':
            d = {a.name: a.tvar(p) for a in self.agg_list}
            d['total'] = self.tvar(p)
        else:
            d = {a.name: a.q(p, kind) for a in self.agg_list}
            d['total'] = self.q(p, kind)
        if total != 'total':
            d[self.name] = d['total']
            del d['total']
        if snap and kind == 'tvar':
            d = {k: self.snap(v) for k, v in d.items()}
        return d

    def price(self, p, distortion=None, *, allocation='lifted', view='ask', efficient=True):
        """
        Price total using regulatory capital and pricing distortion functions and allocate to units.

        Compute rho(X wedge q(p)) where rho corresponds to the pricing distortion and q is the
        quantile function for the total. p is input as a probability level and is converted
        to assets using VaR (and hence snapped to the index). If p > 1, it is interpreted as
        an asset level.

        For the linear allocation all states >= a must be collapsed using objective probabilities to
        a single state.

        Do not attempt to use with a weight_df dataframe from Bounds. For that, use the bounds
        object logic directly which is much more efficient.

        :param p: float; if >1 assets if <1 a prob converted to quantile
        :param distortion: a distortion, list or dictionary (name: dist) of distortions. If None then
          ``self.distortions`` dictionary is used.
        :param allocation: 'lifted' (default for legacy reasons) or 'linear': treatment in default scenarios. See PIR.
        :param view: bid or ask
        :param efficient: for apply_distortion, lifted only.
        :return: :class:`PricingResult` dataclass with fields ``df``, ``price``,
            ``price_dict``, ``a_reg``, ``reg_p``.
        """

        # warnings.warn('In 0.13.0 the default allocation will become linear not lifted.', DeprecationWarning)

        assert allocation in ('lifted', 'linear'), "allocation must be 'lifted' or 'linear'"

        if isinstance(distortion, Distortion):
            distortion = {str(distortion): distortion}
        elif isinstance(distortion, list):
            distortion = {str(d): d for d in distortion}
        elif distortion is None:
            assert self.distortions is not None, 'Must pass a distortion or calibrate distortions prior to calling'
            distortion = self.distortions

        # figure regulatory assets; applied to unlimited losses
        if p > 1:
            a_reg = self.snap(p)
            reg_p = self.cdf(a_reg)
        else:
            a_reg = self.q(p)
            reg_p = p

        if allocation == 'lifted':
            dfs = {}
            price = {}
            last_price = 0
            for k, v in distortion.items():
                # this code is unchanged from original
                logger.info(f'Executing for {k}, lifted')
                aug_df = self.apply_distortion(v, view=view, efficient=efficient)
                if a_reg in aug_df.index:
                    aug_row = aug_df.loc[a_reg]
                else:
                    logger.warning('Regulatory assets not in augmented_df. Using last.')
                    aug_row = aug_df.iloc[-1]

                # holder for the answer
                df = pd.DataFrame(columns=['line', 'L', 'P', 'M', 'Q'], dtype=float)
                df.columns.name = 'statistic'
                df = df.set_index('line', drop=True)

                for line in self.line_names_ex:
                    df.loc[line, :] = [aug_row[f'exa_{line}'], aug_row[f'exag_{line}'],
                                       aug_row[f'T.M_{line}'], aug_row[f'T.Q_{line}']]
                df['a'] = df.P + df.Q
                df['LR'] = df.L / df.P
                df['PQ'] = df.P / df.Q
                df['COC'] = df.M / df.Q
                price[k] = last_price = df.loc['total', 'P']
                dfs[k] = df.sort_index()

            df = pd.concat(dfs.values(), keys=dfs.keys(), names=['distortion', 'unit'])

            ans = PricingResult(df, last_price, price, a_reg, reg_p)

        elif allocation == 'linear':
            # code mirrors pricing_bounds
            # slice for extracting
            # sle = slice(self.bs, a_reg)
            sle = slice(0, a_reg)
            S = self.density_df.loc[sle, ['S']].copy()
            loss = self.density_df.loc[sle, ['loss']]
            # deal losses for allocations; do not want to pick up eta mu versions here
            exeqa = self.density_df.filter(regex='exeqa_[^η]').loc[sle]

            # last entry needs to include all remaining losses from a-bs onwards, hence:
            S.loc[a_reg, 'S'] = 0.
            ps = pd.DataFrame(-np.diff(S, prepend=1, axis=0), index=S.index)

            dfs = {}
            price = {}
            last_price = 0
            for k, v in distortion.items():
                logger.info(f'Executing for {k}, linear')
                if view == 'ask':
                    gS = v.g(S)
                else:
                    gS = 1 - v.g(1 - S)
                gS = pd.DataFrame(gS, index=S.index, columns=['S'])
                gps = pd.DataFrame(-np.diff(gS, prepend=1, axis=0), index=S.index)

                if self.sf(a_reg) > (1 - self.density_df.p_total.sum()) and p != 1:
                    # if p==1 is input then by definition sf(a) = 0 even if there are numerical
                    # rounding issues, so this code can be skipped
                    logger.info('Collapsing tail events by replacing exeqa with a * exi_xgta')
                    logger.info(f'\tsf(areg) > 1 - p_total: {self.sf(a_reg):.5g} > '
                          f'{1 - self.density_df.p_total.sum():.5g} code ')
                    # NOTE: this adjustment requires the whole tail; it has been computed in
                    # density_df. However, when you come to risk adjusted version it hasn't
                    # been computed. That's why the code above falls back to apply distortion.
                    # However, for the linear allocation we use the objective weights.
                    # Here is the adjustment.
                    # painful issue here with the naming leading to
                    rner = lambda x: x.replace('exi_xgta_', 'exeqa_')
                    # this regex does not capture the sum column if present
                    exeqa.loc[a_reg, :] = self.density_df.filter(regex='exi_xgta_.+$(?<!exi_xgta_sum)').\
                                            rename(columns=rner).loc[a_reg - self.bs] * a_reg
                    # there is no exi_xgta_total, so that comes out as missing
                    # need to fill in value
                    if np.isnan(exeqa.loc[a_reg, 'exeqa_total']):
                        exeqa.loc[a_reg, 'exeqa_total'] = exeqa.loc[a_reg].fillna(0).sum()
                    # The lifted/natural difference is that here scenarios in the tail are not re-
                    # weighted using risk adjusted probabilities. They are collapsed with objective
                    # probs.

                # these are at the layer level, these compute αS Eq 14.20 p. 372 inside the
                # parenthsis and then the cumsum computes the integral, bottom p. 372 for
                # bar S, and similarly p. 373 for premium and β (Eq 14.23 and integral at
                # bottom of page.
                # note that by construction S(a) = 0 so there is no extra mass at the end
                exp_loss =   ((ps.to_numpy() * self.bs) / loss.to_numpy() * exeqa )[::-1].cumsum()[::-1]
                alloc_prem = ((gps.to_numpy() * self.bs) / loss.to_numpy() * exeqa)[::-1].cumsum()[::-1]
                margin = alloc_prem - exp_loss

                # deal with last row KLUDGE, s=0, coc = gs-s/(1-gs)=0
                # think about what this should be... poss shift?
                # reciprocal cost of capital = capital / margin = 1 - gS / (gS - S)
                rcoc = (1 - gS) / (gS - S)
                # compute 1/roe at s=1
                gprime = v.g_prime(1)
                fv = gprime / (1 - gprime)
                # print(f'computed s=1 capital factor={fv}')
                # if gS-S=0 then gS=S=1 is possible (certain small losses); then fully loss funded, no equity, hence:
                rcoc = rcoc.fillna(fv).shift(1, fill_value=fv)
                # at S=0 also have gS-S=0, could have infinite
                capital = margin * rcoc.values
                # from IPython.display import display as d2
                # d2(pd.concat((S, gS, rcoc, self.density_df.filter(regex='exi_xgta_').loc[sle], margin, capital), axis=1,
                #              keys=['S', 'gS', 'rcoc', 'alpha', 'margin', 'capital']))

                # these are integrals of alpha S and beta gS
                exp_loss_sum = exp_loss.replace([np.inf, -np.inf, np.nan], 0).sum()
                alloc_prem_sum = alloc_prem.replace([np.inf, -np.inf, np.nan], 0).sum()
                capital_sum = capital.replace([np.inf, -np.inf, np.nan], 0).sum()

                df = pd.concat((exp_loss_sum, alloc_prem_sum, capital_sum), axis=1, keys=['L', 'P', 'Q']) . \
                        rename(index=lambda x: x.replace('exeqa_', '')). \
                        sort_index()
                df['M'] = df.P - df.L
                df['LR'] = df.L / df.P
                df['PQ'] = df.P / df.Q
                df['COC'] = df.M / df.Q
                df['a'] = df.P + df.Q
                price[k] = last_price = df.loc['total', 'P']
                df = df[['L', 'P', 'M', 'Q', 'a', 'LR', 'PQ', 'COC']]
                dfs[k] = df

            df = pd.concat(dfs.values(), keys=dfs.keys(), names=['distortion', 'unit'])
            ans = PricingResult(df, last_price, price, a_reg, reg_p)

        return ans

    def price_ccoc(self, p, ccoc):
        """
        Convenience function to price with a constant cost of captial equal ``ccoc``
        at VaR level ``p``. Does not invoke a Distortion. Returns standard DataFrame
        format.

        """
        a = self.q(p)
        el = self.density_df.loc[a, 'exa_total']
        p = (el + ccoc * a) / (1 + ccoc)
        q = a - p
        m = p - el
        df = pd.DataFrame([[el, p, p - el, a - p, a, el / p, p / q, m / q]],
                          columns=['L', 'P', "M", 'Q', 'a', 'LR', 'PQ', 'COC'],
                          index=['total'])
        return df

    def analyze_distortion(self, distortion, *, p=None, a=None, kind='lower'):
        """
        Pricing readout for ``distortion`` at probability ``p`` or asset level ``a``.

        Parameters
        ----------
        distortion : Distortion or str
            A ``Distortion`` instance, or the name of a previously calibrated
            distortion (looked up in ``self.distortions``).
        p : float, optional
            Probability; converted to asset level via ``self.q(p, kind)``.
            Exactly one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.
        kind : {'lower', 'upper'}
            Type of VaR (only relevant when ``p`` is provided).

        Returns
        -------
        AnalyzeDistortionResult
            Holds the per-line pricing DataFrame (from :meth:`pricing_at`)
            and a small audit DataFrame with the total-level calibration
            quantities (a, L, P, M, Q, LR, ROE, dname, dshape).
        """
        if (p is None) == (a is None):
            raise ValueError(
                'analyze_distortion requires exactly one of p= (probability) '
                'or a= (asset level).')
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        if a is None:
            a_cal = self.q(p, kind)
        else:
            a_cal = self.snap(a)
        pricing_df = self.pricing_at(distortion, a=a_cal)
        L = pricing_df.loc['total', 'L']
        P = pricing_df.loc['total', 'P']
        M = pricing_df.loc['total', 'M']
        Q = pricing_df.loc['total', 'Q']
        audit_df = pd.DataFrame(
            {'value': [a_cal, L, P, M, Q,
                       pricing_df.loc['total', 'LR'],
                       pricing_df.loc['total', 'ROE'],
                       distortion.name, distortion.shape]},
            index=['a', 'L', 'P', 'M', 'Q', 'LR', 'ROE', 'dname', 'dshape'],
        )
        return AnalyzeDistortionResult(
            distortion=distortion,
            pricing_df=pricing_df,
            audit_df=audit_df,
        )

    def analyze_distortions(self, *, p=None, a=None, distortions=None):
        """
        Pricing readout for a set of distortions at probability ``p`` or asset ``a``.

        Parameters
        ----------
        p : float, optional
            Probability; converted to asset level via ``self.q(p)``. Exactly
            one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.
        distortions : dict[str, Distortion], optional
            The distortions to analyse. Defaults to ``self.distortions`` (populated
            by :meth:`calibrate_distortions`).

        Returns
        -------
        AnalyzeDistortionsResult
            ``pricing_df`` is the concatenated exhibit with MultiIndex
            ``(distortion, stat)`` on rows and line names on columns;
            ``stat`` runs over ``['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE', 'a']``.
            ``augmented_dfs`` is a snapshot of the cache for the analysed
            distortions.

        Notes
        -----
        Replaces both the legacy ``analyze_distortions(a=0, p=0, ...)`` and
        ``analyze_distortions2(p, dists=None)``. The output shape matches the
        legacy ``analyze_distortions2``: rows are ``(distortion, stat)``,
        columns are line names.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'analyze_distortions requires exactly one of p= (probability) '
                'or a= (asset level).')
        distortions = distortions or self.distortions
        if not distortions:
            raise ValueError(
                'No distortions to analyse. Pass distortions=, or call '
                'calibrate_distortions first.')
        if a is None:
            a_cal = self.q(p)
        else:
            a_cal = self.snap(a)
        per_dist = {}
        for name, d in distortions.items():
            # rows: line, cols: [L, LR, M, P, PQ, Q, ROE] -> transpose so
            # stats are rows and lines are columns. The transpose drops the
            # categorical column dtype, so we work in plain string indices
            # here and reapply the canonical ordering after concat.
            exhibit = self.pricing_at(d, a=a_cal).T
            exhibit.index = exhibit.index.astype(str)
            # 'a' row: P + Q per line, rescaled so totals sum to a_cal.
            a_row = exhibit.loc['P'] + exhibit.loc['Q']
            a_row = a_row * a_cal / a_row['total']
            exhibit.loc['a'] = a_row
            per_dist[name] = exhibit
        pricing_df = pd.concat(
            per_dist.values(),
            keys=per_dist.keys(),
            names=['distortion', 'stat'],
        )
        # bake the canonical distortion order into level 0 of the index
        pricing_df.index = pricing_df.index.set_levels(
            pricing_df.index.levels[0].astype(DISTORTION_DTYPE), level='distortion')
        # snapshot only the distortions analysed
        augmented_dfs = {
            n: self._augmented_dfs[n] for n in distortions if n in self._augmented_dfs
        }
        return AnalyzeDistortionsResult(
            distortions=dict(distortions),
            pricing_df=pricing_df,
            augmented_dfs=augmented_dfs,
        )


    @property
    def line_renamer(self):
        """
        plausible defaults for nicer looking names

        replaces . or : with space and capitalizes (generally don't use . because it messes with
        analyze distortion....

        leaves : alone

        converts X1 to tex

        converts XM1 to tex with minus (for reserves)

        :return:
        """
        def rename(ln):
            # guesser ...
            if ln == 'total':
                return 'Total'
            if ln.find('.') > 0:
                return ln.replace('.', ' ').title()
            if ln.find(':') > 0:
                return ln.replace(':', ' ').title()
            # numbered lines
            ln = re.sub('([A-Z])m([0-9]+)', r'$\1_{-\2}$', ln)
            ln = re.sub('([A-Z])([0-9]+)', r'$\1_{\2}$', ln)
            return ln

        if self._line_renamer is None:
            self._line_renamer = { ln: rename(ln) for ln in self.line_names_ex}

        return self._line_renamer

    @property
    def tm_renamer(self):
        """
        rename exa -> TL, exag -> TP etc.
        :return:
        """
        if self._tm_renamer is None:
            self._tm_renamer = { f'exa_{l}' : f'T.L_{l}' for l in self.line_names_ex}
            self._tm_renamer.update({ f'exag_{l}' : f'T.P_{l}' for l in self.line_names_ex})

        return self._tm_renamer


    def nice_program(self, wrap_col=90):
        """
        return wrapped version of port program
        :return:
        """
        return fill(self.program, wrap_col, subsequent_indent='\t\t', replace_whitespace=False)


    def bodoff(self, *, p=0.99, a=0):
        """
        Determine Bodoff layer asset allocation at asset level a or
        VaR percentile p, one of which must be provided. Uses formula
        14.42 on p. 284 of Pricing Insurance Risk.

        :param p: VaR percentile
        :param a: asset level
        :return: Bodoff layer asset allocation by unit
        """

        if p==0 and a==0:
            raise ValueError('Must provide either p or a')

        if p > 0:
            a = self.q(p)

        ans = self.density_df.filter(regex='exi_xgta_') \
            .loc[:a - self.bs, :].sum() * self.bs
        ans = ans.to_frame().T
        ans.index = [a]
        ans.index.name = 'a'
        ans = ans.drop(columns='exi_xgta_sum')
        ans.columns = [i.replace('exi_xgta_', '') for i in ans.columns]
        return ans

    def sample(self, n, replace=True, desired_correlation=None, keep_total=True):
        """
        Pull multivariate sample. Apply Iman Conover to induce correlation if required.

        """
        df = pd.DataFrame(index=range(n))
        for c in self.line_names:
            pc = f'p_{c}'
            df[c] = self.density_df[['loss', pc]].\
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

    @property
    def unit_names(self):
        # what these should have been called!
        return self.line_names

    @property
    def unit_names_ex(self):
        # what these should have been called!
        return self.line_names_ex

    @property
    def n_units(self):
        return len(self.line_names)

    def make_comonotonic_allocations(self, max_loss=-1):
        """
        Make comonotonic version of kappas using Denuit's alogorithm.

        Pass in upper bound max_loss, or use self.q(1) by default.
        """
        if max_loss <= 0: max_loss = self.q(1)
        df = self.density_df.filter(regex='loss|p_total|exeqa_[^t]').loc[:max_loss]
        assert df is not None, 'Object must be updated to compute allocations.'
        s_grid = df.loss.to_numpy()
        pdf_s = df.p_total.to_numpy()
        bit = df.iloc[:, 2:]
        kappa = bit.T.to_numpy()
        # do the work
        kappa_tilde = make_comonotonic_allocations_work(s_grid, pdf_s, kappa)
        # add to extract and return
        dfnew = df.copy()
        for c, k in zip(bit, kappa_tilde):
            dfnew[f'{c}_t'] = k
        return dfnew

    def swap_density_df(self, new_df, padding=1):
        """
        EXPERIMENTAL FUNCTION
        TODO: Deal with stats? Create as a stand alone function and create the container portfolio too?
        TODO: is this actually worth doing; are empirical distributions really that slow?
        Swap out density_df for a new density_df created from direct input of line densities.
        This sidesteps all Aggregate object creation. The resulting object has invalid stats.
        USE WITH CAUTION. ``self`` must have the right line names.

        Intended use case: you know the marginal densities as numerical distributions and
        want to compute the sums, exas etc. Generally best to create the object ``port0`` as
        a trivial object with the correct line names and then swap out. Example::

            port0 = build('port Test agg A dfreq [1] dsev[1] agg B dfreq [1] dsev [1]')
            port1 = build('port T2 agg A 1 claim sev lognorm 100 cv .3 fixed agg B 1 claim sev gamma 100 cv .3 fixed')
            new_df = port1.density_df.filter(regex=port1.line_name_pipe + '|loss').drop(columns='p_total')
            port0.swap_density_df(new_df)

        port0 now has the same ``density_df`` as ``port1``.
        """

        from aggregate.utilities import ft, ift

        # swap out marginals
        self.density_df = new_df
        self.log2 = int(np.log2(len(new_df)))
        self.padding = padding

        # recompute sums
        ft_all = None
        ft_line_density = {}
        for agg in self.agg_list:
            raw_nm = agg.name
            nm = f'p_{agg.name}'
            ft_line_density[raw_nm] = ft(self.density_df[nm], padding)
            if ft_all is None:
                ft_all = np.copy(ft_line_density[raw_nm])
            else:
                ft_all *= ft_line_density[raw_nm]
        self.density_df['p_total'] = np.real(ift(ft_all, padding))
        ft_nots = {}
        for line in self.line_names:
            ft_not = np.ones_like(ft_all)
            if np.any(ft_line_density[line] == 0):
                # have to build up
                for not_line in self.line_names:
                    if not_line != line:
                        ft_not *= ft_line_density[not_line]
            else:
                if len(self.line_names) > 1:
                    ft_not = ft_all / ft_line_density[line]
            ft_nots[line] = ft_not

        # do add_exa calculation
        self.add_exa(self.density_df, ft_nots)

        # done!

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


def make_awkward(log2, scale=False):
    """
    Decompose a uniform random variable on range(2**log2) into two parts
    using Eamonn Long's base 4 method.

    Usage: ::

        awk = make_awkward(16)
        awk.density_df.filter(regex='p_[ABt]').cumsum().plot()
        awk.density_df.filter(regex='exeqa_[AB]|loss').plot()

    """
    n = 1 << (log2 // 2)
    sc = 1 << log2
    xs = [int(bin(i)[2:], 4) for i in range(n)]
    ys = [2 * i for i in xs]
    ps = [1 / n] * n
    if scale is True:
        xs = np.array(xs) / sc
        ys = np.array(ys) / sc

    A = Aggregate('A', exp_en=1, sev_name='dhistogram', sev_xs=xs, sev_ps=ps,
                      freq_name='empirical', freq_a=np.array([1]), freq_b=np.array([1]))
    B = Aggregate('B', exp_en=1, sev_name='dhistogram', sev_xs=ys, sev_ps=ps,
                      freq_name='empirical', freq_a=np.array([1]), freq_b=np.array([1]))
    awk = Portfolio('awkward', [A, B])
    awk.update(log2, 1/sc if scale else 1, remove_fuzz=True, padding=0)
    return awk


