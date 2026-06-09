from copy import deepcopy
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from pathlib import Path
import re
from scipy import interpolate
from scipy.optimize import bisect
from scipy.spatial import ConvexHull
from textwrap import fill
import warnings

from .constants import (ALIASING_RATIO, DefectiveDistributionWarning,
                        EXEQA_NOISE_FLOOR, FIG_H, FIG_W,
                        REINS_LABEL_OUTPUT, Validation)
from .config import get_settings
from .distributions import (Aggregate, Severity, WINDOW_NINES, BUCKET_SIZING_P,
                            _flat_col_to_stats_index, approximate_from_mcvsk)

# Resolved once per session from config (see aggregate.config). VALIDATION_NOISE
# is the absolute dust floor used throughout validation.
VALIDATION_NOISE = get_settings().validation.noise

__all__ = ['Portfolio', 'make_awkward', 'make_comonotonic_allocations',
           'swap_density_df']
from .results import (AnalyzeDistortionResult, AnalyzeDistortionsResult,
                      PricingResult)
from .spectral import Distortion, DISTORTION_DTYPE
from . import tail as _tail
from .tail import TailClass
from .pentagon import (PENTAGON_STATS, PENTAGON_DTYPE, complete_pentagon,
                       Pentagon)
from .moments import (MomentAggregator, xsden_to_mwrangler,
                      _noise_aware_rel_error, _snap_noise)
from .iman_conover import iman_conover
from .utilities import (ft, ift, decl_pprint,
                        round_bucket,
                        make_var_tvar, agg_help, explain_validation,
                        remove_fuzz as remove_fuzz_util)
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


# Canonical column order/dtype for pricing exhibits — the single source of
# truth lives in ``pentagon.py`` (the accounting authority). These module-level
# aliases preserve the historical names used throughout this file.
PRICING_STAT_ORDER = PENTAGON_STATS
PRICING_STAT_DTYPE = PENTAGON_DTYPE


# Canonical row MultiIndex for ``Portfolio.stats_df``. Parallels
# ``aggregate.distributions._STATS_ROW_INDEX`` (meta + freq + sev + agg
# moment blocks). Kept as its own constant so future Portfolio-only
# rows (e.g. between-line copula moments) do not bleed into
# Aggregate's surface. All-float: ``self.name`` lives on the attribute.
_PORT_STATS_ROW_INDEX = pd.MultiIndex.from_tuples(
    [
        ('meta', 'limit'), ('meta', 'attachment'),
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
        # default natural-allocation method (linear; see ``allocation_method``)
        self._allocation_method: str = 'linear'
        self._certified_bounded: bool = False
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
        # portfolio reinsurance reporting caches (end-to-end gcn); rebuilt
        # lazily, invalidated on update. See dev/reins-reporting.md.
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_describe = None
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
        self.calibration_df = None
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

        self.validation_eps = get_settings().validation.eps

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
        # Same de-fuzzed ``xsden_to_mwrangler`` convention as
        # ``Portfolio.update`` and ``Aggregate.update_work``.
        _xs = port.density_df['loss'].values
        _p = port.density_df['p_total'].values
        _p_clean = remove_fuzz_util(_p)
        _mw = xsden_to_mwrangler(_xs, _p_clean)
        _ex1, _ex2, _ex3 = _mw.noncentral
        port.est_m, port.est_cv, port.est_skew = _mw.mcvsk
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

    def allocation_bounds(self, *, a=0, p=0, units=None, s_floor=1e-14):
        """
        Natural-allocation premium ranges by unit, as a function of the
        total premium.

        Constructs an :class:`~aggregate.bounds.AllocationBounds` object:
        for any total premium P within the feasible range it returns the
        lower/upper bound on each unit's natural allocation over all
        distortions pricing the total to P, together with the achieving
        biTVaR distortions. Construction is P-independent (exact convex
        hulls of the ``(TVaR_p, a_i(p))`` curves); evaluating at a premium
        is a cheap slice.

        With assets specified (``a`` or ``p``) the total is bounded at
        ``X ∧ a``: default states ``X >= a`` collapse to a single atom
        carrying the *linear* natural allocation ``a · E[X_i/X | X >= a]``
        (built by :meth:`_collapsed_exeqa`, shared with
        ``price(allocation='linear')``); the feasible premium range becomes
        ``[E[X ∧ a], a]``. With neither given the total is unbounded.

        Replaces the pre-1.0 ``pricing_bounds`` method (removed at
        1.0.0a36).

        Parameters
        ----------
        a : float, default 0
            Asset level, snapped to the loss grid. ``0`` means
            unspecified.
        p : float, default 0
            Probability level: resolves ``a = q(p)`` when ``a`` is
            unspecified. Both zero gives the unbounded total.
        units : list of str, optional
            Unit names to include. Default: all of ``line_names``.
        s_floor : float, default 1e-14
            Tail-probability floor below which curve vertices are dropped
            as FFT noise; see :class:`~aggregate.bounds.AllocationBounds`.

        Returns
        -------
        AllocationBounds
            Call it (or its ``bounds`` method) with one or more premiums::

                ab = port.allocation_bounds(p=0.995)
                ab.bounds([1200, 1300])    # (P, unit) -> lower/upper/width
                ab.bitvars(1200)           # achieving (p0, p1, w1)
        """
        from .bounds import AllocationBounds
        if a == 0 and p == 0:
            a = np.inf
        elif a == 0:
            a = self.q(p)
        else:
            a = self.snap(a)
        return AllocationBounds(self, a=a, units=units, s_floor=s_floor)

    def pricing_bounds(self, y_sources, *, a=0, p=0, s_floor=1e-14, n_grid=1024):
        """
        Price ranges of other risks consistent with pricing this total.

        Constructs a :class:`~aggregate.bounds.PricingBounds` object with this
        Portfolio's total as the reference risk ``X``: for any premium P
        within the feasible range it returns the lower/upper bound on the
        price of each risk in ``y_sources`` over all distortions pricing the
        total to P, together with the achieving biTVaR distortions.

        With assets specified (``a`` or ``p``) both ``X`` and the ``y_sources``
        are priced capped at ``min(., a)`` — the well-conditioned regime
        (deep-tail vertices of unbounded heavy-tailed risks are FFT noise and
        make the upper bound grid-sensitive; cap, or raise ``s_floor``).

        Parameters
        ----------
        y_sources : source or list/dict of sources
            The risk(s) ``Y`` to price.  Each may be an ``Aggregate``,
            ``Portfolio``, pmf ``Series``, ``'uniform'``, or a TVaR source;
            a dict supplies explicit names.
        a : float, default 0
            Asset level, snapped to the loss grid.  ``0`` means unspecified.
        p : float, default 0
            Probability level: resolves ``a = q(p)`` when ``a`` is
            unspecified.  Both zero gives the unbounded total.
        s_floor : float, default 1e-14
            Tail-probability floor below which curve vertices are dropped;
            see :class:`~aggregate.bounds.PricingBounds`.
        n_grid : int, default 1024
            Fallback p-grid size for all-closed-form sources.

        Returns
        -------
        PricingBounds
            ::

                pb = port.pricing_bounds(layer_agg, p=0.99)
                pb.bounds([1200, 1300])    # (P, risk) -> lower/upper/width
                pb.bitvars(1200)           # achieving (p0, p1, w1)
        """
        from .bounds import PricingBounds
        if a == 0 and p == 0:
            a = np.inf
        elif a == 0:
            a = self.q(p)
        else:
            a = self.snap(a)
        return PricingBounds(self, y_sources, a=a, s_floor=s_floor,
                             n_grid=n_grid)

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
            # ``remove_fuzz`` returns a de-fuzzed copy; write the float columns
            # back *in place* so the mutation reaches ``self.density_df`` (the
            # default ``df``). Rebinding the local ``df`` would silently no-op.
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = remove_fuzz_util(df, eps)[float_cols]

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
    def tail_class(self) -> TailClass:
        """The portfolio's worst-of aggregate :class:`~aggregate.tail.TailClass`.

        Under independence the tail of a sum is governed by the thickest
        summand (subexponential closure; and for the lighter rungs the
        convolution decay rate equals the slowest = thickest), so this is the
        ``max`` over the unit aggregate rungs. ``UNKNOWN`` on any unit poisons
        the result (it is not treated as a thickness). A ``_certified_bounded``
        override forces :attr:`~aggregate.tail.TailClass.BOUNDED`.
        """
        if getattr(self, '_certified_bounded', False):
            return TailClass.BOUNDED
        worst = TailClass.BOUNDED
        for a in self.agg_list:
            c = a.tail_class.agg
            if c == TailClass.UNKNOWN:
                return TailClass.UNKNOWN
            worst = max(worst, c)
        return worst

    @property
    def bounded(self) -> bool:
        """Whether every unit's aggregate has bounded support.

        Derived view: ``True`` iff :attr:`tail_class` is
        :attr:`~aggregate.tail.TailClass.BOUNDED`, i.e. each unit is bounded
        (correct under independence). Set ``self.bounded = True`` to certify the
        portfolio bounded (e.g. when a unit's auto-detection is conservatively
        ``False`` but the modeller knows the support is capped).
        """
        return self.tail_class == TailClass.BOUNDED

    @bounded.setter
    def bounded(self, value: bool) -> None:
        if value is not True and value is not False:
            raise ValueError('bounded must be True (certify) or False (reset)')
        self._certified_bounded = bool(value)

    def _tail_driver(self):
        """Return ``(worst_class, [driver unit name(s)])`` for the portfolio tail.

        The driver unit(s) are those whose aggregate rung equals the worst-of
        rung. Returns an empty driver list when the worst class is UNKNOWN or
        the portfolio is certified bounded.
        """
        worst = self.tail_class
        if worst in (TailClass.UNKNOWN,) or getattr(self, '_certified_bounded', False):
            return worst, []
        drivers = [a.name for a in self.agg_list if a.tail_class.agg == worst]
        return worst, drivers

    @property
    def tail_description(self) -> str:
        """One aligned line: the portfolio's worst-of aggregate tail class.

        E.g. ``aggregate tail           subexponential (driver: B)``. Derived
        from :attr:`tail_class`.
        """
        worst, drivers = self._tail_driver()
        phrase = _tail.tail_class_label(worst)
        if drivers:
            phrase += f' (driver: {", ".join(drivers)})'
        return f'{"aggregate tail":<25}{phrase}'

    @property
    def tail_explanation(self) -> str:
        """Sentence: per-unit aggregate tail classes and the named driver unit(s).

        Derived from each unit's :attr:`Aggregate.tail_class`.
        """
        worst, drivers = self._tail_driver()
        parts = [f'{a.name}: {_tail.tail_class_label(a.tail_class.agg)}'
                 for a in self.agg_list]
        units = '; '.join(parts)
        if worst == TailClass.UNKNOWN:
            return (f'Portfolio aggregate tail undetermined (a unit is an '
                    f'unrecognised or numeric-only family). Units -- {units}.')
        driver_txt = (f', driven by {", ".join(drivers)}' if drivers else '')
        return (f'Portfolio aggregate tail is {_tail.tail_class_label(worst)} '
                f'(worst-of under independence{driver_txt}). Units -- {units}.')

    @property
    def allocation_method(self) -> str:
        """Natural-allocation method: ``'linear'`` (default) or ``'lifted'``.

        Drives :meth:`price` (and downstream readouts) when no explicit
        ``allocation=`` is passed. Set this once on the portfolio rather
        than threading the argument through every call. The setter
        invalidates the ``augmented_df`` cache because the lifted and
        linear frames differ on the right edge.
        """
        return self._allocation_method

    @allocation_method.setter
    def allocation_method(self, value: str) -> None:
        if value not in ('linear', 'lifted'):
            raise ValueError(
                f"allocation_method must be 'linear' or 'lifted', not {value!r}")
        if value != self._allocation_method:
            self._augmented_dfs.clear()
            self._allocation_method = value

    @property
    def info(self):
        s = []
        s.append(f'portfolio object name    {self.name}')
        s.append(f'aggregate objects        {len(self.line_names):d}')
        s.append(f'allocation_method        {self.allocation_method}')
        s.append(f'bounded                  {self.bounded}')
        s.append(self.tail_description)
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1/self.bs)}'
            s.append(f'bs                       {bss}')
            s.append(f'log2                     {self.log2}')
            win = getattr(self, '_signed_window', None)
            if win is not None:
                s.append(f'signed window            [{win[0]:.6g}, {win[1]:.6g})')
            s.append(f'padding                  {self.padding}')
            s.append(f'sev_calc                 {self.sev_calc}')
            s.append(f'normalize                {self.normalize}')
            s.append(f'last update              {self.last_update}')
            s.append(f'hash                     {self.hash_rep_at_last_update:x}')
        return '\n'.join(s)

    def _reins_after_label(self):
        """Portfolio-wide heading for the after-reins column in ``describe``.

        Aggregates the per-unit :meth:`Aggregate._reins_after_label`
        across the book. Returns ``None`` when **no** unit carries
        reinsurance (the legacy theory/empirical validation view
        applies). When exactly one cession kind appears across all
        ceding units returns that label (``Net`` / ``Ceded``);
        otherwise returns ``Output`` — the umbrella label for mixed output —
        so the whole table can share one column layout.
        """
        labels = {lbl for a in self
                  if (lbl := a._reins_after_label()) is not None}
        if not labels:
            return None
        if len(labels) == 1:
            return labels.pop()
        return REINS_LABEL_OUTPUT

    @property
    def describe(self):
        """Theoretic-and-empirical stats. Used in ``_repr_html_``.

        Reads from the canonical ``stats_df``: theoretical moments from
        the ``total`` column, empirical from ``empirical``, errors from
        ``error``. The output shape mirrors ``Aggregate.describe`` — one
        ``Freq``/``Sev``/``Agg`` row block per unit + ``total``.

        Two display modes, chosen at the **portfolio** level so the unit
        blocks and the ``total`` block always share one column layout:

        * **No unit has reinsurance** -- validation view, columns
          ``EX | Est EX | Err EX | CV | Est CV | Err CV | Sk | Est Sk``.
        * **Any unit has reinsurance** -- economic view, columns
          ``Subject EX | <label> EX | Change EX | ...`` where ``<label>``
          is the portfolio-wide :meth:`_reins_after_label`. Every unit —
          including those with no cession — is rendered in this layout
          (forced via ``Aggregate._describe``) so the table aligns.

        The spread column is **CV** normally, but **SD** when the portfolio is
        signed (any unit is a P&L / negative-support ``ssev``/``dsev`` unit),
        since CV is unstable near a zero mean. The choice is portfolio-wide —
        CV and SD cannot be mixed in one frame — so every unit block is forced
        to SD (via ``Aggregate._describe(force_sd=...)``) and the total SD is
        read robustly from the second moment, not ``mean * cv``.
        """
        _total = self.stats_df['total']
        emp = self.stats_df['empirical']
        rlabel = self._reins_after_label()

        # Spread column: CV by default, **SD** when the portfolio is signed.
        # CV = sd / mean is unstable/meaningless when a mean can be ~0 (a P&L
        # unit straddling break-even), so a signed portfolio reports the
        # standard deviation instead. CV and SD cannot be mixed in one frame,
        # so the choice is made ONCE here, at the portfolio level: if ANY unit
        # is signed (``self._signed()``) the whole table -- every unit block
        # and the total -- switches to SD, with the units forced via
        # ``Aggregate._describe(force_sd=...)``. SD is taken robustly from the
        # second moment, ``sqrt(ex2 - mean**2)``, never ``mean * cv``.
        use_sd = self._signed()
        spread = 'SD' if use_sd else 'CV'

        def _theo_spread(comp):
            if not use_sd:
                return float(_total[(comp, 'cv')])
            m = float(_total[(comp, 'mean')])
            return float(np.sqrt(max(float(_total[(comp, 'ex2')]) - m * m, 0.0)))

        def _emp_spread(comp):
            if not use_sd:
                return float(emp[(comp, 'cv')])
            m = float(emp[(comp, 'mean')])
            return float(np.sqrt(max(float(emp[(comp, 'ex2')]) - m * m, 0.0)))

        df = pd.DataFrame(
            {
                'EX': [float(_total[('freq', 'ex1')]),
                       float(_total[('sev',  'ex1')]),
                       float(_total[('agg',  'ex1')])],
                spread: [_theo_spread('freq'),
                         _theo_spread('sev'),
                         _theo_spread('agg')],
                'Sk': [float(_total[('freq', 'skew')]),
                       float(_total[('sev',  'skew')]),
                       float(_total[('agg',  'skew')])],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'

        # Post-update? Empirical agg moments live in stats_df['empirical'].
        # After the punch-up: portfolio-level sev empirical is also
        # populated (via MomentAggregator off per-unit empirical sev);
        # surface it in the describe table too. Under reinsurance the
        # ``total`` column is the (gross) Subject view and ``empirical``
        # the realised after-reins view, exactly mirroring Aggregate.
        emp_agg_m = emp.get(('agg', 'mean'), np.nan)
        if pd.notna(emp_agg_m):
            mid_label = rlabel or 'Est'
            change_label = 'Change' if rlabel else 'Err'
            df.loc['Sev', f'{mid_label} EX'] = float(emp[('sev', 'mean')])
            df.loc['Agg', f'{mid_label} EX'] = float(emp_agg_m)
            df[f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} {spread}'] = _emp_spread('sev')
            df.loc['Agg', f'{mid_label} {spread}'] = _emp_spread('agg')
            df[f'{change_label} {spread}'] = _noise_aware_rel_error(
                df[f'{mid_label} {spread}'], df[spread])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = float(emp[('sev', 'skew')])
            df.loc['Agg', f'{mid_label} Sk'] = float(emp[('agg', 'skew')])
            df = df[['EX', f'{mid_label} EX', f'{change_label} EX',
                     spread, f'{mid_label} {spread}', f'{change_label} {spread}',
                     'Sk', f'{mid_label} Sk']]
        # Subject-column label under reinsurance; plain headings otherwise.
        if rlabel:
            df = df.rename(columns={
                'EX': 'Subject EX', spread: f'Subject {spread}',
                'Sk': 'Subject Sk'})
        # snap floating-point dust to 0 in the moment-value columns for
        # display (e.g. the skew of a symmetric unit); NaN preserved.
        # Change/Err columns retain their numeric dust.
        for c in df.columns:
            if (' EX' in c or ' CV' in c or ' SD' in c or ' Sk' in c
                    or c in ('EX', 'CV', 'SD', 'Sk')) \
                    and not (c.startswith('Err ') or c.startswith('Change ')):
                df[c] = _snap_noise(df[c])

        # Force every unit block into the portfolio-wide layout so the
        # concat aligns (units with no cession render in reins view too
        # when ``rlabel`` is set; unsigned units render in SD view via
        # ``force_sd`` when the portfolio is signed).
        t1 = [a._describe(force_reins_label=rlabel, force_sd=use_sd)
              for a in self] + [df]
        t2 = [a.name for a in self] + ['total']
        df = pd.concat(t1, keys=t2, names=['unit', 'X'])
        return df

    # ================================================================
    # Reinsurance reporting (end-to-end gcn; see dev/reins-reporting.md)
    # ================================================================

    def _reins_unit_views(self, a):
        """End-to-end gross / ceded / net aggregate densities for one unit.

        Returns a dict ``{'gross', 'ceded', 'net'}`` of densities on the
        unit's grid. A non-ceding unit contributes ``gross == net ==
        modeled`` and a point mass at 0 for ``ceded``. For a ceding unit the
        views are read from its :meth:`Aggregate.reins_density_df`: when an
        aggregate cover is present the final aggregate-cover net/ceded; else
        the occurrence net/ceded. The ``ceded`` view of a unit carrying
        *both* covers is the final (aggregate-stage) cession; the per-stage
        chain lives in the unit's own ``reins_describe``.
        """
        zero = np.zeros_like(a.xs, dtype=float)
        zero[0] = 1.0
        rd = a.reins_density_df
        if rd is None:
            modeled = np.asarray(a.agg_density, dtype=float)
            return {'gross': modeled, 'ceded': zero, 'net': modeled}
        g = rd['p_agg_gross'].to_numpy()
        if a.agg_reins is not None:
            return {'gross': g, 'ceded': rd['p_agg_ceded'].to_numpy(),
                    'net': rd['p_agg_net'].to_numpy()}
        if a.occ_reins is not None:
            return {'gross': g, 'ceded': rd['p_agg_ceded_occ'].to_numpy(),
                    'net': rd['p_agg_net_occ'].to_numpy()}
        return {'gross': g, 'ceded': zero, 'net': g}

    @property
    def reins_density_df(self):
        """Portfolio end-to-end gross / ceded / net aggregate densities.

        Convolves the per-unit end-to-end gross / ceded / net aggregate
        marginals (see :meth:`_reins_unit_views`) under the same independent
        FFT machinery used for the portfolio total, producing three
        portfolio-level marginal aggregate distributions. Units without
        reinsurance contribute their modeled density to ``gross`` and ``net``
        and nothing (point mass at 0) to ``ceded``.

        Columns ``loss``, ``p_agg_gross``, ``p_agg_ceded``, ``p_agg_net``
        (always present). Returns ``None`` when **no** unit cedes.

        Notes
        -----
        Assumes unit independence -- consistent with the existing portfolio
        total. The three portfolio marginals are *separate* distributions
        (portfolio gross / ceded / net loss); they no more satisfy
        ``gross = net (+) ceded`` than the unit-level views do.
        """
        if self._reins_after_label() is None:
            return None
        if self._reins_density_df is not None:
            return self._reins_density_df
        xs = self.density_df['loss'].to_numpy()
        ft_gross = ft_ceded = ft_net = None
        for a in self.agg_list:
            uv = self._reins_unit_views(a)
            fg = ft(uv['gross'], self.padding)
            fc = ft(uv['ceded'], self.padding)
            fn = ft(uv['net'], self.padding)
            if ft_gross is None:
                ft_gross, ft_ceded, ft_net = fg, fc, fn
            else:
                ft_gross = ft_gross * fg
                ft_ceded = ft_ceded * fc
                ft_net = ft_net * fn
        df = pd.DataFrame({'loss': xs}, index=pd.Index(xs, name='loss'))
        df['p_agg_gross'] = np.real(ift(ft_gross, self.padding))
        df['p_agg_ceded'] = np.real(ift(ft_ceded, self.padding))
        df['p_agg_net'] = np.real(ift(ft_net, self.padding))
        self._reins_density_df = df
        return self._reins_density_df

    @property
    def reins_stats_df(self):
        """Per-unit and portfolio-total end-to-end gross / ceded / net moments.

        Rows are ``MultiIndex (view, measure)`` with ``view in
        {gross, ceded, net}`` and ``measure in {ex1, ex2, ex3, mean, cv,
        skew}`` (aggregate-level; there is no per-stage split at the portfolio
        level). Columns are the unit names plus ``total``. Unit columns read
        each unit's end-to-end marginals; ``total`` reads the convolved
        portfolio marginals in :meth:`reins_density_df`. Returns ``None`` when
        no unit cedes.
        """
        if self._reins_after_label() is None:
            return None
        if self._reins_stats_df is not None:
            return self._reins_stats_df
        views = ['gross', 'ceded', 'net']
        measures = ['ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew']
        row_index = pd.MultiIndex.from_product(
            [views, measures], names=['view', 'measure'])

        def moments6(xs, p):
            mw = xsden_to_mwrangler(xs, np.asarray(p, dtype=float))
            return (*mw.noncentral, *mw.mcvsk)

        cols = {}
        for a in self.agg_list:
            uv = self._reins_unit_views(a)
            s = pd.Series(np.nan, index=row_index)
            for v in views:
                for m, val in zip(measures, moments6(a.xs, uv[v])):
                    s[(v, m)] = val
            cols[a.name] = s
        rdp = self.reins_density_df
        xs = rdp['loss'].to_numpy()
        s = pd.Series(np.nan, index=row_index)
        for v, col in [('gross', 'p_agg_gross'), ('ceded', 'p_agg_ceded'),
                       ('net', 'p_agg_net')]:
            for m, val in zip(measures, moments6(xs, rdp[col].to_numpy())):
                s[(v, m)] = val
        cols['total'] = s
        self._reins_stats_df = pd.DataFrame(cols)
        return self._reins_stats_df

    @property
    def reins_describe(self):
        """Portfolio end-to-end reinsurance loss summary.

        One block per unit plus a ``total`` block, concatenated with
        ``unit`` / ... keys (the :meth:`describe` assembly pattern). Each
        block is a ``view x component`` table of **mean loss** on the
        eight :meth:`Aggregate.describe` columns (``EX | Est EX | Change EX |
        CV | Est CV | Change CV | Sk | Est Sk``) for the unit's own per-stage
        cession (from :meth:`Aggregate.reins_describe`); units without
        reinsurance are omitted from their own blocks. The ``total`` block is
        the end-to-end gross / ceded / net portfolio aggregate moments from
        :meth:`reins_stats_df`; the ``EX`` / ``CV`` / ``Sk`` reference is the
        gross end-to-end moment held constant down the three views, ``Est`` is
        the per-view output and ``Change = (output - gross) / gross`` reads as
        the % impact of the programme (0 on the gross row -- no exact pre-bucket
        reference exists for the convolved portfolio marginals).

        Total means equal the sum of the unit end-to-end means per view
        (means add under convolution). Returns ``None`` when no unit cedes.
        """
        if self._reins_after_label() is None:
            return None
        if self._reins_describe is not None:
            return self._reins_describe
        blocks = []
        keys = []
        for a in self:
            rdesc = a.reins_describe
            if rdesc is not None:
                blocks.append(rdesc)
                keys.append(a.name)
        # total block: end-to-end gcn, eight columns matching the unit blocks.
        # Reference (EX/CV/Sk) is the gross end-to-end moment, held constant down
        # each view (the economic view of describe): Est is the per-view output
        # and Change = (output - gross) / gross. The gross row compares gross to
        # gross, so its Change is 0 (no exact pre-bucket reference exists for the
        # convolved portfolio marginals); the ceded / net rows read as the %
        # impact of the whole reinsurance programme.
        rs = self.reins_stats_df
        gross_mean = float(rs.loc[('gross', 'mean'), 'total'])
        gross_cv = float(rs.loc[('gross', 'cv'), 'total'])
        gross_sk = float(rs.loc[('gross', 'skew'), 'total'])
        rows = []
        idx = []
        for v in ['gross', 'ceded', 'net']:
            mean = float(rs.loc[(v, 'mean'), 'total'])
            cv = float(rs.loc[(v, 'cv'), 'total'])
            sk = float(rs.loc[(v, 'skew'), 'total'])
            rows.append([
                gross_mean, mean, float(_noise_aware_rel_error(mean, gross_mean)),
                gross_cv, cv, float(_noise_aware_rel_error(cv, gross_cv)),
                gross_sk, sk])
            idx.append(('total', v, 'agg'))
        total_block = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx, names=['stage', 'view', 'component']),
            columns=Aggregate._REINS_DESCRIBE_COLS)
        blocks.append(total_block)
        keys.append('total')
        self._reins_describe = pd.concat(blocks, keys=keys, names=['unit'])
        return self._reins_describe

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
                raise ValueError(f"Cannot adjust {s['name']} for scale")

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

    def best_bucket(self, log2=16, bucket_sizing_p=BUCKET_SIZING_P):
        """Legacy root-sum-square bucket combine. **DELETE BEFORE BETA.**

        Combines the per-unit recommended buckets by root-sum-square, then
        rounds. This is the historical heuristic and is **no longer on the live
        path**: :meth:`best_window` (resolution + span) replaced it at
        1.0.0a49. It is retained only as a side-by-side comparison aid for
        reviewing the new sizer and is slated for removal before the beta.

        The RMS combine scales the wrong way: *k* identical units give
        ``round_bucket(b*sqrt(k))``, i.e. *adding* units *coarsens* the grid
        (the ``round_bucket`` rounding-up was the only thing that ever made it
        "work"). It also ignores the integer lattice entirely, so an all-integer
        discrete portfolio gets a fine continuous ``bs`` (e.g. ``1/4096``) rather
        than ``bs=1``. See :meth:`best_window` for the correct rule.

        Parameters
        ----------
        log2 : int
            ``log2`` of the bucket count (passed through to ``recommend_bucket``).
        bucket_sizing_p : float
            Tail probability for the per-unit moment windows.

        Returns
        -------
        float
            The rounded root-sum-square bucket.
        """
        # DELETE BEFORE BETA -- superseded by best_window (resolution + span).
        # bs = sum([a.recommend_bucket(log2, p=bucket_sizing_p) for a in self])
        bs = sum([a.recommend_bucket(log2, p=bucket_sizing_p) ** 2 for a in self]) ** 0.5

        return round_bucket(bs)

    def best_window(self, log2=16, bs_in=0, bucket_sizing_p=BUCKET_SIZING_P):
        """Decide the portfolio combine grid by *resolution* + *span* (replaces RMS).

        The portfolio-combine grid must satisfy two **independent** constraints;
        the right combine is their **max**, never a root-sum-square:

        1. **Resolution** -- the finest bucket any unit needs,
           ``min_k bs_k``, where ``bs_k`` is each unit's *natural selected*
           ``bs`` from its own :meth:`Aggregate._bs_window` (captured here in a
           phase-1 pre-pass). A finer shared grid is strictly better provided it
           still fits -- there is no "over-cost" from one unit forcing the
           portfolio finer.
        2. **Span fit** -- the summed support must fit ``N = 2**log2`` buckets
           without wrapping: ``W_tot / N``, where ``W_tot = sum_k W_k`` and
           ``W_k`` is the **selected method's** support-window width
           (``x_max - x_min`` of that unit's winning ``_bs_window_df`` row), *not*
           the padded ``used``-row grid extent (``N*bs``, power-of-2 inflated --
           that would needlessly re-coarsen).

        ``bs = round_bucket(max(min_k bs_k, W_tot / N))`` (a pinned ``bs_in > 0``
        is honoured verbatim, D4). The bare ``W_tot / N`` floor relies on the
        FFT ``padding`` (doubling) for headroom -- that reliance is by design.

        Origin and ``log2``:

        - **non-signed** portfolio: the grid starts at ``x_min = 0`` (Plan A
          keeps non-signed origins at 0; non-zero output windows are Plan B). The
          per-unit selected rows carry ``x_min = 0`` and ``x_max`` = the from-0
          extent, so ``W_k = x_max`` is exactly what must fit a 0-based grid.
          ``log2`` is then **shrunk** to just hold ``W_tot`` at the chosen ``bs``
          (``ceil(log2(W_tot/bs + 1))``, capped) -- a tiny all-integer discrete
          book no longer inflates to the ``log2`` cap.
        - **signed** (P&L) portfolio: the origin is the analytic estimate of the
          summed-support min (floored so no per-line marginal wraps); ``update``
          recomputes the realised origin from the units' post-snap ``x_min``.
          ``log2`` is kept at the cap -- the signed origin estimate is built from
          each unit's grid origin (0 for a non-negative unit, e.g. a point mass
          at 12 reads ``x_min = 0``), so it is *not* the true summed-support min
          and the grid must stay wide enough to absorb the offset. Tight signed
          ``log2`` shrinkage is deferred to Plan B (which also fixes the origin).

        Parameters
        ----------
        log2 : int
            Bucket-count cap, ``2**log2`` buckets.
        bs_in : float
            ``0`` to estimate the bucket; ``>0`` to force it (honoured).
        bucket_sizing_p : float
            Tail probability for the per-unit moment / bounded windows.

        Returns
        -------
        (bs, log2, x_min) : tuple
            Shared grid parameters. :attr:`_bs_window_df` (unit rows + a ``used``
            row) is also populated.

        See Also
        --------
        best_bucket : the deprecated RMS combine (kept for comparison).
        Aggregate._bs_window : the per-unit window estimator consumed here.
        """
        signed = self._signed()
        N_cap = 1 << log2

        # ---- phase 1: per-unit natural windows (analytic, no FFT) ---------
        # Each unit sizes itself on its own (0-origin or signed) grid; we read
        # the *selected method* row, not the padded ``used`` row, for the width.
        rows = []
        bs_ks, x_min_ks, W_ks = [], [], []
        for a in self.agg_list:
            bs_k, l2_k, x_min_k = a._bs_window(log2, 0, None, bucket_sizing_p)
            wdf = a._bs_window_df
            sel = wdf[wdf['selected']].iloc[0] if 'selected' in wdf.columns \
                else wdf.loc['used']
            sx_min, sx_max = float(sel['x_min']), float(sel['x_max'])
            W_k = max(sx_max - sx_min, 0.0)
            bs_ks.append(float(bs_k))
            x_min_ks.append(sx_min)
            W_ks.append(W_k)
            rows.append(dict(unit=a.name, x_min=sx_min, x_max=sx_max, W=W_k,
                             bs=float(bs_k), log2=int(l2_k),
                             coverage=sel.get('coverage', ''),
                             note=str(sel.get('note', ''))))

        # ---- the resolution + span combine --------------------------------
        resolution = min(bs_ks) if bs_ks else 1.0
        W_tot = float(sum(W_ks))
        if bs_in > 0:
            bs = float(bs_in)
        else:
            span = W_tot / N_cap if N_cap else W_tot
            bs = round_bucket(max(resolution, span))

        # ---- origin and log2 ----------------------------------------------
        if signed:
            # Analytic origin estimate: the summed-support min, floored so no
            # per-line marginal wraps (a unit whose support starts above the sum
            # of mins would otherwise wrap). update recomputes from realised
            # post-snap unit origins. log2 stays at the cap (see docstring).
            x_min = min(sum(x_min_ks), min(x_min_ks)) if x_min_ks else 0.0
            x_min = float(np.floor(x_min / bs) * bs)
            log2_out = log2
        else:
            x_min = 0.0
            if bs_in > 0:
                log2_out = log2                       # user pinned the grid
            else:
                # Shrink to just hold the summed (0-based) support at ``bs``;
                # never exceed the cap, never below 1 (>= 2 buckets).
                need = int(np.ceil(np.log2(W_tot / bs + 1.0))) if W_tot > 0 else 1
                log2_out = min(log2, max(need, 1))

        self._build_bs_window_df(rows, bs, log2_out, x_min)
        return bs, log2_out, x_min

    def _signed(self):
        """Whether any unit has signed (negative-support) severity.

        The portfolio is *signed* iff at least one component aggregate is
        signed -- declared with ``ssev`` (continuous) or a ``dsev`` with a
        negative atom; see :meth:`Aggregate._signed`. This mirrors the
        Aggregate gate so the non-signed path stays byte-for-byte identical:
        every signed-support code path in :meth:`update` (window coarsening,
        the F2 present roll, the ``add_exa`` fallback) is reached **only**
        when this returns ``True``.

        Returns
        -------
        bool
        """
        return any(a._signed() for a in self.agg_list)

    def _build_bs_window_df(self, rows, bs, log2, x_min):
        """Build the unit-indexed bucket/window summary for a signed combine.

        Mirrors :attr:`Aggregate._bs_window_df`'s idiom but swaps *method*
        rows for *unit* rows -- the Portfolio convention of one row per unit
        plus a summary line (cf. ``stats_df`` / ``describe``, which carry
        per-unit columns and a ``total``). Each unit row is that unit's
        selected signed window; the final ``used`` row is the realised shared
        portfolio grid ``[x_min, x_min + 2**log2 * bs)``.

        Parameters
        ----------
        rows : list of dict
            Per-unit window rows (``unit``/``x_min``/``x_max``/``W``/``bs``/
            ``log2``/``coverage``/``note``); empty for a non-signed portfolio.
        bs, log2 : float, int
            The realised shared grid bucket size and log2.
        x_min : float
            The realised portfolio-grid origin.
        """
        N = 1 << log2
        cols = ['x_min', 'x_max', 'W', 'bs', 'log2', 'coverage', 'note']
        if rows:
            df = pd.DataFrame(rows).set_index('unit')[cols]
        else:
            df = pd.DataFrame(columns=cols)
        df.loc['used'] = dict(
            x_min=float(x_min), x_max=float(x_min + N * bs),
            W=float(N * bs), bs=float(bs), log2=int(log2),
            coverage=f'1-1e-{WINDOW_NINES}', note='realised portfolio grid')
        self._bs_window_df = df

    def _bs_window(self, log2, bs_in, bucket_sizing_p=BUCKET_SIZING_P):
        """Decide ``(bs, log2, x_min)`` for the portfolio combine grid.

        Thin forwarder to :meth:`best_window` (the resolution + span combine).
        Retained because :meth:`update`'s signed branch calls it by name; the
        whole decision -- the phase-1 per-unit pre-pass, the
        ``max(resolution, span)`` bucket, the origin/``log2`` choice, and
        building :attr:`_bs_window_df` -- lives in :meth:`best_window`.

        Parameters
        ----------
        log2 : int
            Bucket-count cap, ``2**log2`` buckets.
        bs_in : float
            ``0`` to estimate the bucket; ``>0`` to force it (honoured).
        bucket_sizing_p : float
            Tail probability for the per-unit moment windows.

        Returns
        -------
        (bs, log2, x_min) : tuple
            Shared grid parameters; :attr:`_bs_window_df` is also populated.
        """
        return self.best_window(log2, bs_in, bucket_sizing_p)

    def update(self, log2, bs, remove_fuzz=False,
               sev_calc='discrete', discretization_calc='survival', normalize=True, padding=1,
               trim_df=False, add_exa=True, force_severity=True, bucket_sizing_p=BUCKET_SIZING_P,
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
        :param bucket_sizing_p: percentile to use for bucket recommendation.
        :param debug: if True, print debug information
        :return:
        """
        self._valid = None # reset valid flag

        if log2 <= 0:
            raise ValueError('log2 must be >= 0')
        self.log2 = log2
        # Signed (P&L) portfolios route the bucket/window through the
        # signed-aware ``_bs_window`` wrapper (coarsen-to-fit); non-signed
        # books keep the legacy ``best_bucket`` path exactly (3.0 gate).
        signed = self._signed()
        if signed:
            bs, log2, _x_min_est = self._bs_window(log2, bs, bucket_sizing_p)
            self.log2 = log2
            self.bs = bs
        elif bs == 0:
            # Non-signed auto-size: resolution + span combine (best_window),
            # which also shrinks log2 to just hold the summed support -- a tiny
            # all-integer discrete book no longer inflates to the log2 cap.
            bs, log2, _x0 = self.best_window(log2, 0, bucket_sizing_p)
            self.log2 = log2
            self.bs = bs
            logger.info(f'bs=0 entered, setting bs={bs:.6g}, log2={log2} via best_window '
                        f'(resolution + span combine).')
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
        # invalidate reinsurance reporting caches
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_describe = None

        ft_line_density = {}

        # Build the grid and the per-line densities, accumulating their
        # product in Fourier space to get ``p_total``.
        N = 1 << log2
        if signed:
            # ---- signed combine on a shared signed grid (plan 2/3) --------
            # Drive each unit on its OWN signed window [x_min_k, ...) sharing
            # the portfolio bs/log2/padding, so the unit object stays
            # internally correct (no false deficit, right moments, right
            # describe/plot -- plan 2c). The combine reads each unit's
            # ftagg_density, which is origin-at-0 *regardless* of the unit's
            # x_min (the output roll hits the density, never ftagg), so the
            # units' FFTs still multiply correctly here.
            ft_all = None
            x_mins = []
            for agg in self.agg_list:
                agg.update(log2=log2, bs=self.bs, padding=self.padding,
                           sev_calc=sev_calc,
                           discretization_calc=discretization_calc,
                           normalize=normalize, force_severity=force_severity,
                           x_min='auto', bucket_sizing_p=bucket_sizing_p, debug=debug)
                ft_line_density[agg.name] = agg.ftagg_density
                x_mins.append(agg.x_min)
                if ft_all is None:
                    ft_all = np.copy(agg.ftagg_density)
                else:
                    ft_all *= agg.ftagg_density
            # Realised portfolio origin: the support min of the independent
            # sum is Sigma x_min_k; floored by min_k x_min_k so no per-line
            # marginal wraps (safety floor on the plan's sum-of-windows).
            # Snapped to bs (each unit x_min is already a multiple of bs, so
            # the sum is too; the round guards fp dust).
            x_min_tot = float(min(sum(x_mins), min(x_mins)))
            x_min_tot = float(np.round(x_min_tot / self.bs) * self.bs)
            j0_tot = int(round(x_min_tot / self.bs))
            xs = x_min_tot + np.arange(N, dtype=float) * self.bs
            self.density_df = pd.DataFrame(index=xs)
            self.density_df['loss'] = xs
            # F2 present step: full-length irfft then a single roll placing
            # x_min_tot at output index 0, keep N. ``ift(., 0)`` returns the
            # whole length-M = N<<padding buffer (origin-at-0, negatives
            # wrapped to the top); the truncating ``ift(., padding)`` used on
            # the non-signed path would silently drop that wrapped tail.
            self.density_df['p_total'] = np.roll(ift(ft_all, 0), -j0_tot)[:N]
            for agg in self.agg_list:
                self.density_df[f'p_{agg.name}'] = np.roll(
                    ift(agg.ftagg_density, 0), -j0_tot)[:N]
            self._signed_window = (x_min_tot, x_min_tot + N * self.bs)
        else:
            # ---- non-signed path (unchanged, byte-for-byte) ---------------
            # Use self.bs (resolved above): build_many now passes bs through
            # as 0 => auto, so the grid must read the resolved bucket, not the
            # raw parameter.
            MAXL = N * self.bs
            xs = np.linspace(0, MAXL, N, endpoint=False)
            self.density_df = pd.DataFrame(index=xs)
            self.density_df['loss'] = xs
            ft_all = None
            for agg in self.agg_list:
                raw_nm = agg.name
                agg.update_work(xs, self.padding, sev_calc, discretization_calc,
                                normalize, force_severity, debug=debug)
                ft_line_density[raw_nm] = agg.ftagg_density
                self.density_df[f'p_{raw_nm}'] = agg.agg_density
                if ft_all is None:
                    ft_all = np.copy(ft_line_density[raw_nm])
                else:
                    ft_all *= ft_line_density[raw_nm]
            self.density_df['p_total'] = np.real(ift(ft_all, self.padding))

        # ``ft_nots[i]`` = FFT of the sum of all lines except ``i`` —
        # needed for ``exeqa_{i}`` in ``add_exa``. The direct division
        # path is faster but unsafe if any FFT bin is exactly zero
        # (symmetric distributions); fall back to building the product.
        # Skipped for signed books -- ``ft_nots`` is consumed only by
        # ``add_exa``, which is deferred to the pricing iteration (plan 3.3).
        ft_nots = {}
        if not signed:
            for line in self.line_names:
                ft_not = np.ones_like(ft_all)
                if np.any(ft_line_density[line] == 0):
                    for not_line in self.line_names:
                        if not_line != line:
                            ft_not *= ft_line_density[not_line]
                elif len(self.line_names) > 1:
                    ft_not = ft_all / ft_line_density[line]
                ft_nots[line] = ft_not

        self.remove_fuzz(log='update')

        # add exa details
        if add_exa and not signed:
            self.add_exa(self.density_df, ft_nots=ft_nots)
        else:
            # at least want F and S to get quantile functions
            if add_exa and signed:
                # Pricing/allocation columns assume a loss>=0 axis; defer to
                # the pricing iteration rather than emit wrong numbers.
                warnings.warn(
                    'pricing/allocation columns (add_exa) are not yet '
                    'available on signed (P&L) support; writing F/S only. '
                    'See dev/plan-portfolio-neg-x-pricing.md.', stacklevel=2)
            self.density_df['F'] = np.cumsum(self.density_df.p_total)
            self.density_df['S'] = 1 - self.density_df.F

        # Mass-conservation check on the signed grid: a window narrower than
        # the summed support shows up as a deficit (the wrapped tail would be
        # truncated by the final [:N]). Surface it loudly, like the Aggregate
        # path, rather than silently lose mass.
        if signed:
            deficit = 1.0 - float(np.sum(self.density_df['p_total']))
            if deficit > VALIDATION_NOISE:
                warnings.warn(
                    f'{self.name}: portfolio PMF deficit {deficit:.3e} '
                    f'(Σp = 1 − {deficit:.3e} < 1); the signed window is '
                    f'narrower than the combined support -- raise log2 or '
                    f'widen the grid.', DefectiveDistributionWarning,
                    stacklevel=2)

        # Empirical portfolio-total agg moments from the FFT output, via
        # ``xsden_to_mwrangler`` on a de-fuzzed copy -- mirrors
        # ``Aggregate.update_work`` so both modules share one moment
        # convention (meta.3 D4/D8). The de-fuzz zeroes |p| < eps before the
        # ``x**k`` weighting, otherwise far-tail fp noise gets amplified to
        # spurious skew on wide grids.
        _xs = self.density_df['loss'].values
        _p = self.density_df['p_total'].values
        _p_clean = remove_fuzz_util(_p)
        _mw = xsden_to_mwrangler(_xs, _p_clean)
        _ex1, _ex2, _ex3 = _mw.noncentral
        self.est_m, self.est_cv, self.est_skew = _mw.mcvsk
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
        cols = list(self.line_names) + [
            'total', 'after_occ', 'empirical',
            'occ_impact', 'agg_impact', 'gross_empirical', 'error',
        ]
        self.stats_df = pd.DataFrame(
            np.nan, index=_PORT_STATS_ROW_INDEX, columns=cols, dtype=float,
        )

        # Per-unit columns: copy each agg's ``stats_df['mixed']`` into the
        # matching unit column. (Both frames are now all-float; the legacy
        # ``meta/name`` row that was excluded by the ``in self.stats_df.index``
        # guard is gone, but the guard is harmless.)
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
            return self.stats_df.loc[('meta', meta_key), unit_cols]

        tot_el = float(_collect('el').sum())
        tot_prem = float(_collect('prem').sum())
        tot_lr = (tot_el / tot_prem) if tot_prem else np.nan
        # Attachment: portfolio total only makes sense if all units
        # share the same attachment (commonly 0); else NaN. Limit at
        # portfolio level is the max across units (legacy convention).
        _attaches = _collect('attachment').values
        if len(_attaches) and np.all(_attaches == _attaches[0]):
            tot_attach = float(_attaches[0])
        else:
            tot_attach = np.nan

        stats = ma.get_fsa_stats(total=True, remix=False)
        for flat, val in zip(_flat_names, stats):
            self.stats_df.loc[_flat_col_to_stats_index(flat), 'total'] = val
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
        # through a fresh MomentAggregator. Aggregate stores empirical sev
        # raw moments directly (``a.stats_df['empirical']`` rows
        # ``('sev','ex1'..'ex3')``), so we read them straight in -- no
        # (mean, cv, skew) → (ex1, ex2, ex3) inversion needed.
        # ``get_fsa_stats`` returns 18 entries: freq f1/f2/f3/m/cv/sk, sev
        # s1/s2/s3/m/cv/sk, agg a1/a2/a3/m/cv/sk; we use the sev block
        # (indices 6..11).
        ma_emp = MomentAggregator()
        for a in self.agg_list:
            mixed = a.stats_df['mixed']
            emp = a.stats_df['empirical']
            ma_emp.add_fs(
                float(mixed[('freq', 'ex1')]),
                float(mixed[('freq', 'ex2')]),
                float(mixed[('freq', 'ex3')]),
                float(emp[('sev', 'ex1')]),
                float(emp[('sev', 'ex2')]),
                float(emp[('sev', 'ex3')]),
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

        # error: noise-aware diff vs the ``total`` column — relative error,
        # falling back to absolute where the theoretical value is ~0 (e.g.
        # the skew of a symmetric unit). For meta rows where empirical ==
        # total the result is 0 (or NaN where total is NaN).
        self.stats_df['error'] = _noise_aware_rel_error(
            self.stats_df['empirical'], self.stats_df['total'])

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

        The CV and skew tests are applied only when the theoretical value is
        finite and its magnitude exceeds ``VALIDATION_NOISE`` -- a
        theoretically-zero skew (symmetric total) or CV is skipped, because
        the FFT's empirical estimate of a zero higher moment is grid-
        dependent noise with no meaningful relative error. When the test
        applies, ``np.isclose`` with relative tolerance ``10*eps`` (CV) /
        ``100*eps`` (skew) measures agreement.

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
            logger.info('Exiting: Portfolio validation steps skipped due to failed or n/a Aggregate validation')
            self._valid = rv
            return rv
        else:
            logger.info('No Aggregate object fails validation')

        # apply validation to the Portfolio total. SSoT: relative errors
        # come straight off ``stats_df['error']`` (noise-aware diff of
        # ``empirical`` vs ``total``) -- no detour through ``describe``.
        err = self.stats_df['error'].abs()
        eps = self.validation_eps
        sev_err_mean = float(err.get(('sev', 'mean'), 0.0))
        agg_err_mean = float(err.get(('agg', 'mean'), 0.0))
        if sev_err_mean > eps:
            logger.info('FAIL: Portfolio Sev mean error > eps')
            rv |= Validation.SEV_MEAN

        if agg_err_mean > eps:
            logger.info('FAIL: Portfolio Agg mean error > eps')
            rv |= Validation.AGG_MEAN

        # Aliasing fingerprint: agg error >> sev error. Silenced under
        # ``VALIDATION_NOISE`` where both are dust.
        if (agg_err_mean > VALIDATION_NOISE
                and sev_err_mean > 0
                and agg_err_mean > ALIASING_RATIO * sev_err_mean):
            logger.info('FAIL: Portfolio Agg mean error > %d * sev error', ALIASING_RATIO)
            rv |= Validation.ALIASING

        # CV and skew: tested only when the theoretical value is meaningfully
        # non-zero (abs(theo) > VALIDATION_NOISE); a theoretically-zero skew
        # (symmetric total) or CV cannot be validated against the FFT's
        # empirical estimate, whose noise floor is grid-dependent. See
        # Aggregate.valid. Read theoretical (``total``) and empirical from the
        # canonical stats_df; isfinite skips undefined moments.
        total = self.stats_df['total']
        emp = self.stats_df['empirical']
        for comp, flag in (('sev', Validation.SEV_CV), ('agg', Validation.AGG_CV)):
            theo = float(total[(comp, 'cv')])
            est = float(emp[(comp, 'cv')])
            if (np.isfinite(theo) and abs(theo) > VALIDATION_NOISE and np.isfinite(est)
                    and not np.isclose(est, theo, rtol=10 * eps, atol=VALIDATION_NOISE)):
                logger.info('FAIL: Portfolio %s CV error > eps', comp)
                rv |= flag
        for comp, flag in (('sev', Validation.SEV_SKEW), ('agg', Validation.AGG_SKEW)):
            theo = float(total[(comp, 'skew')])
            est = float(emp[(comp, 'skew')])
            if (np.isfinite(theo) and abs(theo) > VALIDATION_NOISE and np.isfinite(est)
                    and not np.isclose(est, theo, rtol=100 * eps, atol=VALIDATION_NOISE)):
                logger.info('FAIL: Portfolio %s skew error > eps', comp)
                rv |= flag

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
            p = 0.999 if kind == 'linear' else 1 - 1e-10
            hi = self.q(p)
            # Window-aware x-limits keyed on the grid origin
            # (``density_df.index[0]``); mirrors the Aggregate ``_limits``:
            #  * origin < 0  -- signed (P&L) book: two-sided quantile range
            #    (``f(hi)`` would clip the negative tail or reverse when hi < 0);
            #  * origin > 0  -- thin-tailed output window: anchor the left edge
            #    at the realised support minimum, not 0;
            #  * origin == 0 -- ordinary non-negative book: unchanged.
            if self.density_df.index[0] < 0:
                lo = self.q(1 - p)
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            if self.density_df.index[0] > 0:
                lo = float(self.density['loss'].min())
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            return f(hi)
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
            raise ValueError('Inadmissible stat/kind passsed, expected range/density and log/linear.')

    @property
    def density(self):
        """The "live" part of :attr:`density_df` — rows with positive total mass.

        Returns ``density_df.query('p_total > 0')``: the actual support of the
        portfolio, dropping the leading and trailing zero-probability buckets of
        the FFT grid. This is usually what you want to *see*. It is recomputed on
        each access (a plain property, not cached) because ``density_df`` can be
        reassigned by ``update`` or by sampling.
        """
        return self.density_df.query('p_total > 0')

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

    def add_exa(self, df, ft_nots):
        r"""Add the conditional-expectation columns to ``df``.

        Per-line and total: ``exeqa_*`` = ``E[X_i | X=a]``, ``exlea_*`` =
        ``E[X_i | X≤a]``, ``exgta_*`` = ``E[X_i | X>a]``, ``exi_x_*`` =
        ``E[X_i / X | X=a]``, ``exi_xlea_*`` / ``exi_xgta_*`` =
        conditional-share variants, ``e_*`` = unconditional mean,
        ``lev_*`` = ``E[X_i ∧ a]``, ``exa_*`` = portfolio-allocated
        expected loss to line ``i``. Also writes ``F``, ``S``,
        ``exa_total``, ``lev_total``.

        Names with a leading ``t`` clash with the ``total`` regex
        anchors — do not use them.

        Parameters
        ----------
        df : pandas.DataFrame
            Frame to extend in place. ``update`` passes
            ``self.density_df``; ``gradient`` and ``swap_density_df``
            pass their own frames.
        ft_nots : dict[str, np.ndarray]
            FFTs of the "not-line" densities, one per line — used by
            ``exeqa_{line} = ift(ft(loss · p_i) · ft_nots[i]) / p_total``.
            Always pre-computed by the caller.
        """
        cut_eps = np.finfo(float).eps
        bs = self.bs

        if not np.all(df.p_total >= 0):
            n_neg = (df.p_total < -cut_eps).sum()
            logger.warning(f'p_total has {n_neg} negative values; NOT setting to zero...')
        sum_p_total = df.p_total.sum()
        logger.info(f'{self.name}: sum of p_total is 1 - {1 - sum_p_total:12.8e} NOT rescaling.')
        df['F'] = np.cumsum(df.p_total)
        df['S'] = 1 - df.F

        logger.info(
            f'Portfolio.add_exa | {self.name}: S <= 0 values has length {len(np.argwhere((df.S <= 0).to_numpy()))}')

        df['exa_total'] = df.S.shift(1, fill_value=0).cumsum() * self.bs
        df['lev_total'] = df['exa_total']

        # exlea_total = (E[X∧a] - a·S(a)) / F(a)
        df['exlea_total'] = (df.exa_total - df.loss * df.S) / df.F
        # Blank ``exlea_total`` where ``exlea > loss`` (numerical noise
        # at the small-F end of the grid), plus a small ``mult·bs``
        # safety buffer that scales with grid size. The bucketing
        # ``mult ∈ {1, 10, 100}`` is a heuristic carried over from
        # earlier code; replacement with a principled ``F < k·eps``
        # rule is tracked separately.
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
        df.loc[0:loss_max, 'exlea_total'] = np.nan

        df['e_total'] = np.sum(df.p_total * df.loss)
        df['exgta_total'] = df.loss + (df.e_total - df.exa_total) / df.S
        df['exeqa_total'] = df.loss  # E[X | X=a] = a

        Seq0 = (df.S == 0)

        for col in self.line_names:
            # exeqa_{line} via FFT: E[X_i | X=a] = E[X_i 1_{X=a}] / P(X=a)
            # = ift( ft(loss · p_i) · ft_nots[i] ) / p_total
            df[f'exeqa_{col}'] = (
                np.real(self.ift(self.ft(df.loss * df[f'p_{col}']) *
                                 ft_nots[col])) / df.p_total)
            # p_total ≈ 0 ⇒ exeqa is unreliable; zero it.
            df.loc[df.p_total < cut_eps, f'exeqa_{col}'] = 0

            stemp = 1 - df[f'p_{col}'].cumsum()
            df[f'lev_{col}'] = stemp.shift(1, fill_value=0).cumsum() * self.bs

            temp = np.cumsum(df[f'exeqa_{col}'] * df.p_total)
            df[f'exlea_{col}'] = temp / df.F
            df.loc[0:loss_max, f'exlea_{col}'] = 0

            df[f'e_{col}'] = np.sum(df[f'p_{col}'] * df.loss)
            df[f'exgta_{col}'] = (df[f'e_{col}'] - temp) / df.S

            # exi_x_{col}: guard loss[0]=0 by copying and patching.
            denom = df['loss'].copy()
            denom.iat[0] = 1.0
            df[f'exi_x_{col}'] = np.sum(df[f'exeqa_{col}'] * df.p_total / denom)
            temp_xi_x = np.cumsum(df[f'exeqa_{col}'] * df.p_total / denom)
            df[f'exi_xlea_{col}'] = temp_xi_x / df.F
            df.loc[0, f'exi_xlea_{col}'] = 0
            df.loc[df.exlea_total == 0, f'exi_xlea_{col}'] = 0

            # exi_xgta_{col}: tail-reverse-cumsum of (exeqa_i/loss · p_total),
            # normalised by S. Last value is undefined (we have no information
            # past the grid), so we fill with NaN and zero out the S==0 rows
            # to avoid NaN propagation into exa.
            df[f'exi_xgta_{col}'] = (
                (df[f'exeqa_{col}'] / df.loss * df.p_total)
                .shift(-1, fill_value=np.nan)[::-1].cumsum()) / df.S
            df.loc[Seq0, f'exi_xgta_{col}'] = 0.

            df[f'exi_xeqa_{col}'] = df[f'exeqa_{col}'] / df['loss']
            df.loc[0, f'exi_xeqa_{col}'] = 0

            # exa_{col} = ∫₀ᵃ S(x)·exi_xgta_{col}(x) dx (PIR-style).
            df[f'exa_{col}'] = (df.S * df[f'exi_xgta_{col}']).shift(1, fill_value=0).cumsum() * self.bs

        # Sum-of-shares check columns.
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

    def add_exa_details(self, df):
        """Add EPD (expected policyholder deficit) and reimbursement columns.

        Adds, in-place on ``df``:

        - ``epd_0_total`` and ``epd_0_{line}`` — ``max(0, E[X]-LEV(a))/E[X]``
          (pure stand-alone EPD on the unmodified loss).
        - ``epd_1_{line}`` — ``max(0, E[X]-E[X∧a;X≤a])/E[X]`` using
          ``exa_{line}`` (the in-portfolio allocation of expected loss).
        - ``e1xi_1gta_total`` / ``e1xi_1gta_{line}`` — ``E[1/X · 1_{X>a}]``
          (reimbursement-effectiveness diagnostic).

        Pure diagnostic columns; no core surface consumes them. The
        legacy ``eta_mu`` second-priority / EPD-interpolation branch
        (and the companion ``add_eta_mu`` method) was retired in the
        v1.0 refactor — there are no callers left.
        """
        index_inv = 1.0 / df.loss
        df['epd_0_total'] = \
            np.maximum(0, df['e_total'] - df['lev_total']) / df['e_total']
        df['e1xi_1gta_total'] = (df['p_total'] * index_inv).shift(-1)[::-1].cumsum()
        for col in self.line_names:
            df[f'e1xi_1gta_{col}'] = (df[f'p_{col}'] * index_inv).shift(-1)[::-1].cumsum()
            df[f'epd_0_{col}'] = \
                np.maximum(0, df[f'e_{col}'] - df[f'lev_{col}']) / df[f'e_{col}']
            df[f'epd_1_{col}'] = \
                np.maximum(0, df[f'e_{col}'] - df[f'exa_{col}']) / df[f'e_{col}']

    def calibrate_distortion(self, name, r0=0.05, premium_target=0.0,
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
            Mass-at-zero intercept for ``cll``, ``clin``, ``lep``, ``ly``.
            Ignored by the other kinds. Default 0.05.
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
        init_shape = subclass._calibration_init_shape
        # natural-kwarg construction; the calibration loop then mutates
        # ``self.shape`` in place via ``_newton_iterate``.
        if lookup == 'ccoc':
            dist = Distortion('ccoc', r=init_shape)
        elif lookup in ('cll', 'clin', 'lep', 'ly'):
            pn = subclass.param_name or {
                'cll': 'b', 'clin': 'slope', 'lep': 'r', 'ly': 'r',
            }[lookup]
            dist = Distortion(name=lookup, r0=r0, **{pn: init_shape})
        else:
            pn = subclass.param_name
            dist = Distortion(name=lookup, **{pn: init_shape})
        dist.calibrate(S=S, bs=self.bs, premium_target=premium_target,
                       ess_sup=ess_sup, assets=assets, el=el)
        return dist

    def calibrate_distortions(self, coc, *, p=None, a=None, kind='lower'):
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
        kind : {'lower', 'upper'}, optional
            VaR kind when ``p`` is provided. Default ``'lower'``.

        Returns
        -------
        pandas.DataFrame
            The per-distortion calibration receipt, ``distortion_df`` (also
            stored on ``self.distortion_df``): one row per distortion in
            ``[ccoc, ph, wang, dual, tvar]``, index named ``distortion`` (an
            ordered categorical, canonical sort), columns
            ``[param_name, param, error, gini_p, area]``. ``param`` is the raw
            shape (``param_name`` says what it is per family); ``error`` is the
            premium miss; ``gini_p`` is the comparable normalised shape
            ``= 2∫g−1 = p_equiv`` (TVaR-equivalent level); ``area = (gini_p+1)/2
            = ∫g``.

        Notes
        -----
        The shared calibration *target* — identical across all five rows — is
        not repeated here; it is stored once on ``self.calibration_df`` as a
        one-row frame: the inputs ``coc, p, F(a)`` lead, then the canonical
        pentagon octet ``L, M, P, Q, a, LR, PQ, ROE``. ``ROE`` there equals
        ``coc`` (a free self-check). The calibrated distortion objects are on
        ``self.distortions`` keyed by name.

        Calibration is one-point (one ``coc`` at one ``p``/``a``). This replaces
        both the legacy batch
        ``calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, ...)`` and
        ``calibrate_distortions2(coc, reg_p)``.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'calibrate_distortions requires exactly one of p= (probability) '
                'or a= (asset level).')
        if a is None:
            a = self.q(p, kind)
            p_val = p
        else:
            a = self.snap(a)
            p_val = self.cdf(a)
        exa = self.density_df.loc[a, 'exa_total']
        # invert COC -> LR -> P (matches the legacy ROE -> LR -> P path).
        delta = coc / (1 + coc)
        nu = 1 - delta
        P = nu * exa + delta * a
        d_list = ['ccoc', 'ph', 'wang', 'dual', 'tvar']
        rows = []
        distortions = {}
        for dname in d_list:
            dist = self.calibrate_distortion(
                name=dname, premium_target=P, assets=a)
            distortions[dname] = dist
            # param_name is the family's natural parameter ('a', 'lam', 'b',
            # 'p'); ccoc has none -> 'r'. gini_p = 2*int(g) - 1 (= p_equiv);
            # area = int(g) = (gini_p + 1)/2.
            param_name = getattr(dist, 'param_name', None) or 'r'
            rows.append([param_name, dist.shape, dist.error, dist.gini_p,
                         (dist.gini_p + 1) / 2])
        distortion_df = pd.DataFrame(
            rows,
            columns=['param_name', 'param', 'error', 'gini_p', 'area'],
            index=pd.CategoricalIndex(
                d_list, dtype=DISTORTION_DTYPE, name='distortion'),
        )

        # the shared calibration target, shown once: the inputs coc, p enter as
        # leading descriptor columns and complete_pentagon trails the canonical
        # octet (a is the octet's a, not duplicated as a lead column).
        calibration_df = complete_pentagon(
            pd.DataFrame([[coc, p_val, self.cdf(a), exa, P - exa, P, a - P]],
                         columns=['coc', 'p', 'F(a)', 'L', 'M', 'P', 'Q'],
                         index=pd.Index(['calibration'], name='line')))

        self.distortion_df = distortion_df
        self.calibration_df = calibration_df
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
        """Pentagon pricing readout per line at probability ``p`` or asset ``a``.

        Warms the augmented_df cache for ``distortion`` and pulls the
        ``L M P Q a | LR PQ ROE`` row at the requested asset level.

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
            ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']``. Per-line
            ``a = P + Q`` (allocated assets); on ``total`` it equals the
            requested portfolio asset level.

        Notes
        -----
        Consolidates row-extraction logic that previously lived in ``price``
        and ``analyze_distortion``.
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
            columns=['L', 'M', 'P', 'Q'],
            dtype=float,
        )
        out.index.name = 'line'
        for line in lines:
            out.loc[line, 'L'] = row[f'exa_{line}']
            out.loc[line, 'P'] = row[f'exag_{line}']
            out.loc[line, 'M'] = row[f'T.M_{line}']
            out.loc[line, 'Q'] = row[f'T.Q_{line}']
        # exact total Q = a - exag_total beats the layer-by-layer cumsum,
        # which can drift by a few buckets in the tail.
        out.loc['total', 'Q'] = a - row['exag_total']
        # fill a + ratios and stamp the canonical categorical (pentagon.py)
        out = complete_pentagon(out)
        return out

    def pentagon_at(self, distortion, *, p=None, a=None, line='total'):
        """Single-line pentagon as a :class:`~aggregate.pentagon.Pentagon` object.

        The object-flavored analogue of :meth:`pricing_at`: returns one fully
        solved :class:`Pentagon` (an eight-vector with named attributes and
        provenance) for ``line`` at probability ``p`` or asset level ``a``,
        rather than a DataFrame of all lines. The natural entry point for the
        "complete a partial input" workflow — the returned object carries the
        accounting identities and can be re-solved.

        Parameters
        ----------
        distortion : Distortion or str
            Passed through to :meth:`apply_distortion`.
        p : float, optional
            Probability; converted to asset level via ``self.q(p)``. Exactly
            one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index.
        line : str, default 'total'
            Which unit to read (``'total'`` for the portfolio total).

        Returns
        -------
        Pentagon
            Fully solved, with ``.distortion`` / ``.shape`` provenance attached.

        Notes
        -----
        Reads the same augmented-distortion row as :meth:`pricing_at`; the
        total ``Q`` uses the exact ``a - exag_total`` (matching ``pricing_at``),
        so the two agree.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'pentagon_at requires exactly one of p= (probability) '
                'or a= (asset level).')
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        if a is None:
            a = self.q(p)
        else:
            a = self.snap(a)
        aug = self.apply_distortion(distortion)
        row = aug.loc[a] if a in aug.index else aug.iloc[-1]
        peg = Pentagon(obj=self)
        L = row[f'exa_{line}']
        P = row[f'exag_{line}']
        if line == 'total':
            # exact total Q, matching pricing_at
            Q = a - row['exag_total']
        else:
            Q = row[f'T.Q_{line}']
        # L, P, Q are the three independent amounts; M = P - L, a = P + Q follow.
        peg.solve(L=L, P=P, Q=Q)
        peg.distortion = distortion
        peg.shape = getattr(distortion, 'shape', None)
        return peg

    def _build_augmented(self, dist, *, view='ask', S_calculation='forwards', efficient=True):
        """Construct an augmented_df from ``self.density_df`` under ``dist``.

        Pure builder: returns the frame without touching ``self`` (the
        ``apply_distortion`` wrapper writes it into the cache). The common
        path is shared between the ``efficient`` and full branches; only
        the per-line marginal-margin (``M.*_{line}``) and eta-mu columns
        are gated by ``efficient=False`` (consumed by
        :func:`pedagogy.plot_twelve`).

        Notes
        -----
        The L'Hôpital ROE fallback at the right end (``Q_total==0``,
        ``gS==1``) is ``ROE = 1/g'(1) - 1``: the limit of
        ``(gS-S)/(1-gS)`` as ``S → 1``. Earlier the ``efficient`` branch
        used ``g'(1)``, the full branch used ``1/g'(1) - 1`` — the
        two disagreed at the boundary and ``efficient`` was wrong.
        Unified here.
        """

        df = self.density_df.copy()

        # forwards S keeps the tail accurate (recomputed from p_total cumsum);
        # backwards is the historical default, retained for thin-tailed cases.
        if S_calculation == 'forwards':
            df['S'] = 1 - df.p_total.cumsum()

        if view == 'bid':
            g = dist.g_dual
            g_prime = lambda x: dist.g_prime(1 - x)
        elif view == 'ask':
            g = dist.g
            g_prime = dist.g_prime
        else:
            raise ValueError(f'view must be bid or ask, not {view}')

        # cosmetic floor at zero for residual float noise
        cut_eps = np.finfo(float).eps
        n_neg = (df.S < 0).sum()
        if n_neg:
            n_below_neg_eps = (df.S < -cut_eps).sum()
            logger.warning(f'{n_below_neg_eps} negative S < -eps values being set to zero...')
        df.loc[df.S < 0, 'S'] = 0

        df['gS'] = g(df.S)
        df['gF'] = 1 - df.gS
        df['gp_total'] = -np.diff(df.gS, prepend=1)
        # kill -0 entries from np.diff
        df.loc[df.gp_total == 0, 'gp_total'] = 0.0

        # Truncate where the exeqa decomposition breaks down (discrete "gaps"
        # in p_total are ignored — error is only meaningful on support).
        lnp = '|'.join(self.line_names)
        idx_pne0 = df.query(' p_total > 0 ').index
        exeqa_err = np.abs(
            (df.loc[idx_pne0].filter(regex=f'exeqa_({lnp})').sum(axis=1) - df.loc[idx_pne0].loss) /
            df.loc[idx_pne0].loss)
        exeqa_err.iloc[0] = 0
        # +1 to keep the last reliable row (iloc[:idx] is exclusive)
        idx = int(exeqa_err[exeqa_err < EXEQA_NOISE_FLOOR].index[-1] / self.bs + 1)
        logger.debug(f'index of max reliable value = {idx}')
        if idx:
            df = df.iloc[:idx]
        gSeq0 = (df.gS == 0)
        logger.debug(f'len(S==0) = {np.sum(gSeq0)} elements')

        if not np.all(df.S.iloc[1:] <= df.S.iloc[:-1].values):
            logger.error('S = density_df.S is not non-increasing...carrying on but you should investigate...')

        # Per-line distorted conditional means and ground-up ``exag``.
        # The shift(-1, fill_value=last_x) puts the tail mass on the
        # right of bucket ``a`` so the denominator gS sums to the same
        # numerator weights — without it the last bucket leaks tail
        # probability (Nov 2020 fix).
        for line in self.line_names:
            last_gS = df.gS.iloc[-1]
            last_x = df[f'exeqa_{line}'].iloc[-1] / df.loss.iloc[-1] * last_gS
            df[f'exi_xgtag_{line}'] = (
                (df[f'exeqa_{line}'] / df.loss * df.gp_total)
                .shift(-1, fill_value=last_x)[::-1].cumsum()) / df.gS
            df.loc[gSeq0, f'exi_xgtag_{line}'] = 0.0
            df[f'exag_{line}'] = (
                df[f'exi_xgtag_{line}'] * df.gS).shift(1, fill_value=0).cumsum() * self.bs

        # ---- Total-level block (single source of truth, both branches) ----
        df['exag_total'] = df.gS.shift(1, fill_value=0).cumsum() * self.bs
        df['M.M_total'] = df.gS - df.S
        df['M.Q_total'] = 1 - df.gS
        # Layer ROE is the same on every layer by law invariance; at the
        # right edge ``M.Q_total==0`` so use the L'Hôpital limit
        # ``ROE(1) = lim (gS-S)/(1-gS) = 1/g'(1) - 1``. When ``g'(1)==0``
        # (TVaR beyond its threshold) the limit is ``+∞`` — premium is
        # 100% loss-funded, no capital — and ``M.Q_{line}/inf == 0``.
        gp1 = float(g_prime(1))
        gprime1 = np.inf if gp1 == 0 else 1 / gp1 - 1
        df['M.ROE_total'] = np.where(
            df['M.Q_total'] != 0,
            df['M.M_total'] / df['M.Q_total'],
            gprime1)
        roe_zero = (df['M.ROE_total'] == 0.0)

        # ---- Per-line T.M / T.Q (needed by both branches; consumed by pricing_at) ----
        for line in self.line_names_ex:
            df[f'T.M_{line}'] = df[f'exag_{line}'] - df[f'exa_{line}']
            mm_l = df[f'T.M_{line}'].diff().shift(-1) / self.bs
            mq_l = mm_l / df['M.ROE_total']
            mq_l.iloc[-1] = 0
            mq_l.loc[roe_zero] = np.nan
            df[f'T.Q_{line}'] = mq_l.shift(1).cumsum() * self.bs
            df.loc[0, f'T.Q_{line}'] = 0

        if efficient:
            return df

        # ---- Full diagnostic columns (pedagogy.plot_twelve) ----
        df['M.L_total'] = df['S']
        df['M.P_total'] = df['gS']
        for line in self.line_names_ex:
            df[f'T.L_{line}'] = df[f'exa_{line}']
            df[f'T.P_{line}'] = df[f'exag_{line}']
            df.loc[0, f'T.P_{line}'] = 0
            df[f'T.LR_{line}'] = df[f'exa_{line}'] / df[f'exag_{line}']
            df.loc[0, f'T.M_{line}'] = 0
            df[f'M.M_{line}'] = df[f'T.M_{line}'].diff().shift(-1) / self.bs
            mq = df[f'M.M_{line}'] / df['M.ROE_total']
            mq.iloc[-1] = 0
            mq.loc[roe_zero] = np.nan
            df[f'M.Q_{line}'] = mq
            if line != 'total':
                df[f'M.L_{line}'] = df[f'exi_xgta_{line}'] * df['S']
                df[f'M.P_{line}'] = df[f'exi_xgtag_{line}'] * df['gS']
            df[f'M.LR_{line}'] = df[f'M.L_{line}'] / df[f'M.P_{line}']
            df[f'T.ROE_{line}'] = df[f'T.M_{line}'] / df[f'T.Q_{line}']
            df[f'T.PQ_{line}'] = df[f'T.P_{line}'] / df[f'T.Q_{line}']
            df[f'M.PQ_{line}'] = df[f'M.P_{line}'] / df[f'M.Q_{line}']

        # Recompute totals from definitions to absorb small drift from the
        # per-line cumulative sums in the loop above.
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

    def _collapsed_exeqa(self, a, *, collapse=None):
        """Tail-collapsed slice of ``density_df`` for linear-NA work at assets ``a``.

        The bounded total ``X ∧ a`` keeps the grid rows ``loss < a``
        unchanged and collapses all default states ``X >= a`` into a single
        atom at ``a``. Under the *linear* natural allocation
        (equal-priority proportional sharing in default: unit i receives
        ``a · X_i / X``), the conditional allocation at the collapsed atom
        is

            ``exeqa_i(a) = a · E[X_i / X | X >= a] = a · exi_xgta_i(a - bs)``

        read from the precomputed ``exi_xgta_*`` columns. The atom's
        probability is ``S(a - bs)``: the construction forces ``S(a) = 0``,
        so the whole tail mass — including any PMF deficit — lands in the
        last bucket. ``exeqa_total`` at the atom is filled from the sum of
        parts (there is no ``exi_xgta_total``).

        Single owner of this delicate collapse idiom, shared by
        :meth:`price` (``allocation='linear'``) and
        :class:`~aggregate.bounds.AllocationBounds`.

        Parameters
        ----------
        a : float
            Asset level; must lie on the loss grid (callers snap).
        collapse : bool, optional
            Force (``True``) or suppress (``False``) the exeqa tail
            re-aiming. Default ``None`` collapses iff the tail mass
            ``sf(a)`` exceeds the PMF deficit ``1 - sum(p_total)`` — i.e.,
            there is real mass beyond ``a``, not just FFT leakage.

        Returns
        -------
        S, loss, exeqa, ps : DataFrame
            All indexed by loss on ``[0, a]``: survival (with ``S(a) = 0``
            forced), loss levels, conditional allocations (tail-collapsed
            last row), and the resulting probability masses.
        """
        sle = slice(0, a)
        S = self.density_df.loc[sle, ['S']].copy()
        loss = self.density_df.loc[sle, ['loss']]
        # deal losses for allocations; not eta-mu versions
        exeqa = self.density_df.filter(regex='exeqa_[^η]').loc[sle]

        # last entry collapses all remaining losses from a-bs onwards
        S.loc[a, 'S'] = 0.
        ps = pd.DataFrame(-np.diff(S, prepend=1, axis=0), index=S.index)

        # Tail-collapse on exeqa is distortion-independent: when
        # sf(a) > 1 - sum(p_total) the tail-bucket carries the
        # missing tail mass and exeqa at a has to be re-aimed at
        # a * exi_xgta_{line}.
        if collapse is None:
            collapse = bool(self.sf(a) > (1 - self.density_df.p_total.sum()))
        if collapse:
            logger.info('Collapsing tail events by replacing exeqa with a * exi_xgta')
            rner = lambda x: x.replace('exi_xgta_', 'exeqa_')
            exeqa.loc[a, :] = self.density_df.filter(
                regex='exi_xgta_.+$(?<!exi_xgta_sum)'). \
                rename(columns=rner).loc[a - self.bs] * a
            # there is no exi_xgta_total — fill from the sum of parts
            if np.isnan(exeqa.loc[a, 'exeqa_total']):
                exeqa.loc[a, 'exeqa_total'] = exeqa.loc[a].fillna(0).sum()

        return S, loss, exeqa, ps

    def price(self, p, distortion=None, *, allocation=None, view='ask', efficient=True):
        """Price the total under a distortion and allocate to units.

        ``rho(X ∧ q(p))`` for a single distortion (or a dict / list of
        distortions). ``p`` is a probability if ``p ≤ 1`` (converted to
        assets via VaR and snapped to the index) and an asset level
        otherwise. ``allocation`` defaults to :attr:`allocation_method`
        (``'linear'`` out of the box); pass ``'lifted'`` to override.

        Lifted allocation reads from the risk-adjusted ``augmented_df``
        and is unstable on the right edge for distortions with a mass
        on an unbounded support. In that case ``price`` **refuses**
        and points at ``'linear'`` — linear collapses tail states using
        objective probabilities and stays bounded.

        Parameters
        ----------
        p : float
            VaR probability when ``p <= 1``; asset level when ``p > 1``.
        distortion : Distortion | list | dict | None
            One distortion, several, or ``None`` to use
            :attr:`self.distortions`.
        allocation : {'linear', 'lifted', None}, optional
            ``None`` (default) reads :attr:`allocation_method`.
        view : {'ask', 'bid'}
            Pricing view.
        efficient : bool
            Build only the columns needed for pricing (``augmented_df``
            is faster). Lifted only.

        Returns
        -------
        PricingResult
            Per-line ``df`` (pentagon columns), the total price scalar,
            ``price_dict`` keyed by distortion, and ``a_reg`` / ``reg_p``.
        """
        if allocation is None:
            allocation = self._allocation_method
        if allocation not in ('lifted', 'linear'):
            raise ValueError(
                f"allocation must be 'lifted' or 'linear', not {allocation!r}")

        if isinstance(distortion, Distortion):
            distortion = {str(distortion): distortion}
        elif isinstance(distortion, list):
            distortion = {str(d): d for d in distortion}
        elif distortion is None:
            assert self.distortions is not None, 'Must pass a distortion or calibrate distortions prior to calling'
            distortion = self.distortions

        if allocation == 'lifted' and not self.bounded:
            bad = [name for name, d in distortion.items() if getattr(d, 'has_mass', False)]
            if bad:
                raise ValueError(
                    f"lifted allocation on an unbounded portfolio with a mass distortion "
                    f"is unstable on the right edge (essentially all the distortion weight "
                    f"lands on the last bucket). Distortion(s) with mass: {bad}. "
                    f"Use allocation='linear' (the new default) or certify "
                    f"`portfolio.bounded = True` if the support is in fact bounded.")

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
                logger.info(f'Executing for {k}, lifted')
                aug_df = self.apply_distortion(v, view=view, efficient=efficient)
                if a_reg in aug_df.index:
                    aug_row = aug_df.loc[a_reg]
                else:
                    logger.warning('Regulatory assets not in augmented_df. Using last.')
                    aug_row = aug_df.iloc[-1]

                df = pd.DataFrame(
                    index=pd.Index(list(self.line_names_ex), name='line'),
                    columns=['L', 'M', 'P', 'Q'],
                    dtype=float,
                )
                for line in self.line_names_ex:
                    df.loc[line, 'L'] = aug_row[f'exa_{line}']
                    df.loc[line, 'P'] = aug_row[f'exag_{line}']
                    df.loc[line, 'M'] = aug_row[f'T.M_{line}']
                    df.loc[line, 'Q'] = aug_row[f'T.Q_{line}']
                df = complete_pentagon(df)
                price[k] = last_price = df.loc['total', 'P']
                dfs[k] = df.sort_index()

            df = pd.concat(dfs.values(), keys=dfs.keys(), names=['distortion', 'unit'])

            ans = PricingResult(df, last_price, price, a_reg, reg_p)

        elif allocation == 'linear':
            # Tail-collapsed slice [0, a_reg] — S, loss levels, conditional
            # allocations and masses. The delicate collapse construction
            # lives in _collapsed_exeqa (shared with AllocationBounds);
            # p == 1 suppresses the collapse (a_reg is the essential sup).
            S, loss, exeqa, ps = self._collapsed_exeqa(
                a_reg, collapse=None if p != 1 else False)

            # Distortion-independent expected-loss integral (αS) — hoist
            # so the per-distortion loop only redoes the distortion-
            # dependent alloc_prem / capital.
            # Eq 14.20 (PIR p. 372): row x carries f_x = (p_x · exeqa_x / loss_x) · bs;
            # the reverse-cumsum-then-sum integrates over [0, a_reg].
            exp_loss = ((ps.to_numpy() * self.bs) / loss.to_numpy() * exeqa)[::-1].cumsum()[::-1]
            exp_loss_sum = exp_loss.replace([np.inf, -np.inf, np.nan], 0).sum()

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

                # alloc_prem = β g(S) integral (Eq 14.23)
                alloc_prem = ((gps.to_numpy() * self.bs) / loss.to_numpy() * exeqa)[::-1].cumsum()[::-1]
                margin = alloc_prem - exp_loss

                # reciprocal cost of capital = capital / margin = (1 - gS) / (gS - S);
                # at gS = S = 1 the layer is fully loss-funded, no equity — use the
                # L'Hôpital limit fv = g'(1) / (1 - g'(1)) as the fill.
                rcoc = (1 - gS) / (gS - S)
                gprime = v.g_prime(1)
                fv = gprime / (1 - gprime)
                rcoc = rcoc.fillna(fv).shift(1, fill_value=fv)
                capital = margin * rcoc.values

                alloc_prem_sum = alloc_prem.replace([np.inf, -np.inf, np.nan], 0).sum()
                capital_sum = capital.replace([np.inf, -np.inf, np.nan], 0).sum()

                df = pd.concat(
                    (exp_loss_sum, alloc_prem_sum, capital_sum),
                    axis=1, keys=['L', 'P', 'Q']
                ).rename(index=lambda x: x.replace('exeqa_', '')).sort_index()
                df['M'] = df.P - df.L
                df = complete_pentagon(df)
                price[k] = last_price = df.loc['total', 'P']
                dfs[k] = df

            df = pd.concat(dfs.values(), keys=dfs.keys(), names=['distortion', 'unit'])
            ans = PricingResult(df, last_price, price, a_reg, reg_p)

        return ans

    def price_stand_alone(self, dist, p):
        """
        Price each unit on a stand-alone basis and compare to the diversified whole.

        Every unit is priced *as if it were the only line in the book*: its
        capital standard is its own VaR at level ``p`` (no diversification
        credit), and the distortion ``dist`` is applied to its own loss
        distribution. This is contrasted with the ``total`` column — the whole
        portfolio priced together at the portfolio VaR(``p``) — and with the
        ``sum`` column, the simple sum of the stand-alone parts. The gap between
        ``sum`` and ``total`` is the diversification benefit.

        Each unit's stand-alone row is produced by :meth:`Aggregate.price`
        (a unit is just an :class:`Aggregate`); the ``total`` row by
        :meth:`pricing_at`. Both route through the canonical pentagon
        (:func:`~aggregate.pentagon.complete_pentagon`), so every row carries
        the full octet ``L, M, P, Q, a, LR, PQ, ROE`` derived in one place.

        Parameters
        ----------
        dist : Distortion or str
            The (already calibrated) pricing distortion, or the name of one in
            :attr:`distortions` (populated by :meth:`calibrate_distortions`).
        p : float
            Probability in ``(0, 1)``; the VaR capital standard for each
            stand-alone unit and for the total.

        Returns
        -------
        pandas.DataFrame
            Canonical pentagon orientation: the eight statistics ``L, M, P, Q,
            a, LR, PQ, ROE`` are the columns; rows are one per priced entity —
            the units, plus ``total`` (diversified whole) and ``sum`` (sum of
            the stand-alone parts) — under a ``(method, unit)`` MultiIndex, where
            ``method`` is the distortion's string form. Transpose for the
            traditional stat-down-the-side exhibit (``a.T``), matching
            ``analyze_distortion(...).pricing_df.T``.

        Raises
        ------
        TypeError
            If ``dist`` is neither a :class:`Distortion` nor a string, or ``p``
            is not numeric.
        ValueError
            If ``p`` is not in ``(0, 1)``, or a distortion name is requested
            when no distortions have been calibrated.
        KeyError
            If a distortion name is not found in :attr:`distortions`.

        Notes
        -----
        The amounts (``L, M, P, Q, a``) add across units, so the ``sum`` row is
        their column-wise total; the ratios (``LR, PQ, ROE``) do not add and are
        re-derived from the summed amounts via ``complete_pentagon``.
        """
        # ---- validate p -------------------------------------------------
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError(f'p must be a probability in (0, 1), got {p!r}.')
        if not 0 < p < 1:
            raise ValueError(f'p must be a probability in (0, 1), got {p}.')

        # ---- resolve the distortion ------------------------------------
        if isinstance(dist, str):
            if not self.distortions:
                raise ValueError(
                    f'No calibrated distortions on this Portfolio; cannot look '
                    f'up {dist!r}. Pass a Distortion instance, or call '
                    f'calibrate_distortions(...) first.')
            try:
                dist = self.distortions[dist]
            except KeyError:
                raise KeyError(
                    f'Distortion {dist!r} not found; available: '
                    f'{sorted(self.distortions)}.')
        elif not isinstance(dist, Distortion):
            raise TypeError(
                f'dist must be a Distortion or the name of a calibrated '
                f'distortion, got {type(dist).__name__}.')

        # ---- per-unit stand-alone pentagons (each at its OWN VaR(p)) ----
        parts = pd.concat([ag.price(p, dist) for ag in self.agg_list])

        # ---- the whole book priced together at the portfolio VaR(p) -----
        total = self.pricing_at(dist, p=p).loc[['total']]

        # ---- sum of the stand-alone parts: amounts add, ratios re-derive --
        sop = parts[['L', 'M', 'P', 'Q']].sum().to_frame('sum').T
        sop.index.name = 'line'
        sop = complete_pentagon(sop)

        # ---- assemble in canonical orientation: stats are the columns, one
        # row per entity (units, then total, then sum), tagged by method ----
        exhibit = pd.concat([parts, total, sop])[list(PENTAGON_STATS)]
        exhibit.index.name = 'unit'
        exhibit.columns.name = 'stat'
        return pd.concat({str(dist): exhibit}, names=['method'])

    def price_pentagon(self, *, p=None, a=None, P=None, M=None, Q=None,
                       LR=None, PQ=None, ROE=None):
        """Complete the pricing octet at a capital level given one target.

        Fix the capital level with exactly one of ``p`` (a VaR probability,
        ``a = self.q(p)``) or ``a`` (an asset level, snapped to the grid), then
        supply exactly one pricing target -- premium ``P``, cost of capital
        ``ROE`` (a.k.a. CoC), or a loss ratio via ``LR`` (equivalently ``M``,
        ``Q`` or ``PQ``). Returns the canonical one-row (``'total'``) pentagon
        ``DataFrame`` (columns :data:`~aggregate.pentagon.PENTAGON_STATS`).
        The target keywords match the canonical stat names.

        Pure accounting completion against the portfolio's total expected loss
        at the chosen capital level -- **no distortion is involved** (contrast
        :meth:`price`, which calibrates and applies a :class:`Distortion`). The
        triple ``{L, a, target}`` is solved by :meth:`Pentagon.solve`.
        Generalizes :meth:`price_ccoc` (the cost-of-capital special case).

        Parameters
        ----------
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
        pent = Pentagon(obj=self)
        pent.solve_obj(p=p, a=a, P=P, M=M, Q=Q, lr=LR, pq=PQ, roe=ROE)
        return pent.as_frame(line='total')

    def price_ccoc(self, ccoc, *, p):
        """
        Convenience function to price with a constant cost of captial equal ``ccoc``
        at VaR level ``p``. Does not invoke a Distortion. Returns the standard
        canonical pentagon DataFrame (one ``'total'`` row, columns
        :data:`~aggregate.pentagon.PENTAGON_STATS`).

        Thin alias for :meth:`price_pentagon` with the cost-of-capital target::

            self.price_pentagon(p=p, ROE=ccoc)
        """
        return self.price_pentagon(p=p, ROE=ccoc)

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
            and a one-row ``audit_df`` for the total: descriptor columns
            ``dname``, ``dshape`` first, then the canonical pentagon octet
            (:data:`~aggregate.pentagon.PENTAGON_STATS`) as the trailing eight.
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
        # one-row audit, same orientation as every other readout: descriptors
        # (dname/dshape) lead, the pentagon octet is the trailing [-8:].
        audit_df = pd.DataFrame(
            {'dname': distortion.name, 'dshape': distortion.shape,
             'L': pricing_df.loc['total', 'L'],
             'M': pricing_df.loc['total', 'M'],
             'P': pricing_df.loc['total', 'P'],
             'Q': pricing_df.loc['total', 'Q']},
            index=pd.Index(['total'], name='line'),
        )
        audit_df = complete_pentagon(audit_df)
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
            # pricing_at returns lines × canonical pentagon columns; transpose
            # so stats are rows and lines are columns. The transpose drops the
            # categorical column dtype, so work in plain string labels here and
            # reapply the canonical stat order/dtype after concat.
            exhibit = self.pricing_at(d, a=a_cal).T
            exhibit.index = exhibit.index.astype(str)
            # 'a' row: P + Q per line, rescaled so totals sum to a_cal.
            a_row = exhibit.loc['P'] + exhibit.loc['Q']
            a_row = a_row * a_cal / a_row['total']
            exhibit.loc['a'] = a_row
            # canonical stat order (pentagon.py), trailing octet semantics
            per_dist[name] = exhibit.reindex(PENTAGON_STATS)
        pricing_df = pd.concat(
            per_dist.values(),
            keys=per_dist.keys(),
            names=['distortion', 'stat'],
        )
        # bake the canonical distortion order into level 0 and the canonical
        # stat order/dtype into level 1 of the index (survives the transpose).
        pricing_df.index = pricing_df.index.set_levels(
            pricing_df.index.levels[0].astype(DISTORTION_DTYPE), level='distortion')
        pricing_df.index = pricing_df.index.set_levels(
            pricing_df.index.levels[1].astype(PENTAGON_DTYPE), level='stat')
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
        """Thin shim around :func:`swap_density_df` (module-level function).

        Kept for callers that already use the method form; new code
        should call the standalone function so the dependency on a
        pre-existing Portfolio object is explicit.
        """
        swap_density_df(self, new_df, padding=padding)

def swap_density_df(port, new_df, padding=1):
    """Swap a Portfolio's ``density_df`` for one with new marginal densities.

    Recombine the per-line densities (``p_{line}`` columns) by FFT,
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
        Target object. ``port.agg_list`` and ``port.line_names`` set
        which ``p_{line}`` columns are read from ``new_df``.
    new_df : pandas.DataFrame
        Indexed by the loss grid; carries ``loss`` and one ``p_{line}``
        column per unit. ``p_total`` is recomputed by FFT convolution.
    padding : int
        FFT padding for the recombination (default 1).
    """
    port.density_df = new_df
    port.log2 = int(np.log2(len(new_df)))
    port.padding = padding
    port.bs = float(new_df['loss'].iloc[1] - new_df['loss'].iloc[0])

    # Recombine via FFT — same logic as ``Portfolio.update``'s
    # ``ft_nots`` block; reuse the recipe so the two stay aligned.
    ft_all = None
    ft_line_density = {}
    for agg in port.agg_list:
        raw_nm = agg.name
        ft_line_density[raw_nm] = ft(port.density_df[f'p_{raw_nm}'], padding)
        if ft_all is None:
            ft_all = np.copy(ft_line_density[raw_nm])
        else:
            ft_all *= ft_line_density[raw_nm]
    port.density_df['p_total'] = np.real(ift(ft_all, padding))
    ft_nots = {}
    for line in port.line_names:
        ft_not = np.ones_like(ft_all)
        if np.any(ft_line_density[line] == 0):
            for not_line in port.line_names:
                if not_line != line:
                    ft_not *= ft_line_density[not_line]
        elif len(port.line_names) > 1:
            ft_not = ft_all / ft_line_density[line]
        ft_nots[line] = ft_not

    port.add_exa(port.density_df, ft_nots)
    port._augmented_dfs.clear()

    # Refresh empirical rows of stats_df from the swapped densities.
    # No mixed/independent decomposition exists here — leave those
    # columns at NaN. xsden_to_mwrangler tolerates a deficit if the
    # marginals don't sum to 1.
    xs = port.density_df['loss'].to_numpy()
    for line in port.line_names:
        mw = xsden_to_mwrangler(xs, port.density_df[f'p_{line}'].to_numpy())
        ex1, ex2, ex3 = mw.noncentral
        m, cv, skew = mw.mcvsk
        for measure, value in [('ex1', ex1), ('ex2', ex2), ('ex3', ex3),
                               ('mean', m), ('cv', cv), ('skew', skew)]:
            port.stats_df.loc[('agg', measure), line] = value
            port.stats_df.loc[('agg', measure), 'empirical'] = np.nan  # no longer valid
    mw_total = xsden_to_mwrangler(xs, port.density_df['p_total'].to_numpy())
    ex1, ex2, ex3 = mw_total.noncentral
    m, cv, skew = mw_total.mcvsk
    for measure, value in [('ex1', ex1), ('ex2', ex2), ('ex3', ex3),
                           ('mean', m), ('cv', cv), ('skew', skew)]:
        port.stats_df.loc[('agg', measure), 'empirical'] = value
        port.stats_df.loc[('agg', measure), 'total'] = value


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


