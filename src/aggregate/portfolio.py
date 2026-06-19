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
                        EXEQA_NOISE_FLOOR, FIG_H, FIG_W, INFO_NA, info_row,
                        REINS_LABEL_OUTPUT, Validation)
from .config import get_settings
from .distributions import (Aggregate, Severity, WINDOW_NINES, BUCKET_SIZING_P,
                            _flat_col_to_stats_index, approximate_from_mcvsk,
                            estimate_agg_window, value_type_label)

# Resolved once per session from config (see aggregate.config). VALIDATION_NOISE
# is the absolute dust floor used throughout validation.
VALIDATION_NOISE = get_settings().validation.noise

__all__ = ['Portfolio', 'make_awkward', 'make_comonotonic_allocations',
           'swap_density_df']
from .results import (AnalyzeDistortionResult, AnalyzeDistortionsResult,
                      PricingResult)
from .spectral import Distortion, DISTORTION_DTYPE, choquet_weights
from . import tail as _tail
from .tail import TailClass
from .pentagon import (PENTAGON_STATS, PENTAGON_DTYPE, complete_pentagon,
                       Pentagon)
from .moments import (MomentAggregator, xsden_to_mwrangler,
                      _noise_aware_rel_error, _snap_noise)
from .iman_conover import iman_conover
from .decl_writer import format_program
from .utilities import (ft, ift,
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
# rows (e.g. between-unit copula moments) do not bleed into
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
    - Model a large account with several sub units
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
        self.unit_names = []
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
                self.unit_names.append(agg_name)
                self.__setattr__(agg_name, a)
                mixed = a.stats_df['mixed']
                ma.add_fs(mixed[('freq', 'ex1')], mixed[('freq', 'ex2')], mixed[('freq', 'ex3')],
                          mixed[('sev',  'ex1')], mixed[('sev',  'ex2')], mixed[('sev',  'ex3')])
                max_limit = max(max_limit, np.max(np.array(a.limit)))

        self.unit_names_ex = self.unit_names + ['total']
        self.unit_name_pipe = "|".join(self.unit_names_ex)
        for n in self.unit_names:
            # unit names cannot equal total
            if n == 'total':
                raise ValueError('Line names cannot equal total, it is reserved for...total')

        # value_type: derived, the unanimous sign-convention role of the
        # units. Mixed loss/payoff units have no coherent "more is worse /
        # more is better" reading, so they are rejected here, before any
        # expensive update. Empty portfolio -> neutral default loss.
        roles = {a._is_loss_value for a in self.agg_list}
        if len(roles) == 2:
            loss_label, payoff_label = value_type_label(True), value_type_label(False)
            loss_units = [a.name for a in self.agg_list if a._is_loss_value]
            payoff_units = [a.name for a in self.agg_list if not a._is_loss_value]
            raise ValueError(
                f"Portfolio {self.name!r}: mixed value_type across units -- "
                f"{loss_label}: {loss_units}, {payoff_label}: {payoff_units}. "
                f"A portfolio cannot mix {loss_label} and {payoff_label} units.")
        self._is_loss_value = roles.pop() if roles else True

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
        self._unit_renamer = None
        # if created by uw it stores the program here
        self.program = ''
        self.distortions = None
        self.distortion_df = None
        self.calibration_df = None
        self.figure = None
        # bucket/window reporting (set by best_window / update)
        self._bs_window_df = None
        self._bs_clip = None            # portfolio far-tail clip record or None
        self._combine_x_min = None      # windowed combine origin (Plan B) or None

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
            **{f'exeqa_{i}': (i, np.mean) for i in self.unit_names})
        # need to do this after rescaling to get correct (rounded) total values
        probs = sample_in.groupby(by='total').p_total.sum()
        # want all probs to be positive
        probs = np.maximum(0, probs.fillna(0.0))

        # working copy of self's density_df with relevant columns, plus the
        # unit pmfs scattered back onto the total grid for the stand-alone
        # lev calc below. Sample grids are zero-origin, so the aligned view
        # is exact; sampling semantics on a windowed/signed book are
        # deferred to the sampling redesign plan (numerics-2 deliverable 4
        # keeps this path mechanically working only).
        df = self.density_df.filter(
            regex=f'^(loss|e_({self.unit_name_pipe})|(e|p)_total)$').copy()
        df = df.join(self.aligned_unit_density_df(grid='total',
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
                           df.query('p_total > 0')[[f'exeqa_{i}' for i in self.unit_names]].sum(axis=1))

        assert df.index.is_unique
        df['exeqa_total'] = df.loss

        # add additional variables via loop
        for col in self.unit_names_ex:
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
        (the collapse is built inside ``AllocationBounds`` from the
        ``exi_xgta_*`` columns); the feasible premium range becomes
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
            Unit names to include. Default: all of ``unit_names``.
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
            return self.agg_list[self.unit_names.index(item)]
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
    def bs_window_df(self) -> 'pd.DataFrame':
        """Curated, read-only view of the portfolio combine grid (``[bs-reporting]``).

        One row per unit (its selected window), then the four candidate combine
        rows (``mm`` Portfolio MM bulk / ``rms`` RMS-of-windows reference /
        ``sbj`` single-big-jump look-through / ``sum`` legacy linear bound), then
        the realised shared ``used`` grid -- culled to the user-facing columns
        (``x_min`` / ``x_max`` / ``bs`` / ``log2`` / ``log2_need`` / ``clipped`` /
        ``note``, parity with :attr:`Aggregate.bs_window_df`). The full frame --
        with the ``coverage`` string and window width ``W`` -- stays on the
        private :attr:`_bs_window_df` for experts. Returns ``None`` before the
        grid is sized. See :attr:`bs_description` / :attr:`bs_explanation`.
        """
        df = getattr(self, '_bs_window_df', None)
        if df is None:
            return None
        cols = ['x_min', 'x_max', 'bs', 'log2', 'log2_need', 'clipped', 'note']
        return df.reindex(columns=cols).copy()

    @property
    def bs_description(self) -> str:
        """One-line summary of the shared portfolio combine grid (``[bs-reporting]``).

        The realised ``(bs, log2, x_min)`` and grid top of the resolution + span
        combine (``best_window``); ``'portfolio grid not sized yet'`` before the
        grid is built.
        """
        df = getattr(self, '_bs_window_df', None)
        if df is None or 'used' not in df.index:
            return 'portfolio grid not sized yet (call update())'
        u = df.loc['used']
        top = float(u['x_min']) + (1 << int(u['log2'])) * float(u['bs'])
        txt = (f'portfolio grid: bs={float(u["bs"]):g}, log2={int(u["log2"])}, '
               f'x_min={float(u["x_min"]):g} (top={top:g})')
        clip = getattr(self, '_bs_clip', None)
        if clip is not None:
            cm = clip.get('clipped_mass', float('nan'))
            cm_txt = f'~{cm:.3g}' if np.isfinite(cm) else 'a sliver'
            txt += f'; clips {cm_txt} of the tail (raise log2 to {int(clip["need_log2"])})'
        return txt

    @property
    def bs_explanation(self) -> str:
        """Verbose prose explaining the portfolio combine grid (``[bs-reporting]``).

        Mirrors :attr:`Aggregate.bs_explanation` at the portfolio level: the
        worst-of tail one-liner, the Portfolio MM bulk window vs the inspectable
        ``mm <= rms <= sum`` reference ordering (the ``mm - rms`` gap is the
        skewness/diversification adjustment the combine makes), whether the
        single-big-jump look-through floored the extent, the realised shared
        grid (windowed Plan B or 0-based Plan A), and any far-tail clip with how
        to widen it. ``'portfolio grid not sized yet'`` before :meth:`update`.
        """
        df = getattr(self, '_bs_window_df', None)
        if df is None or 'used' not in df.index:
            return 'Portfolio grid not sized yet (call update()).'
        u = df.loc['used']
        top = float(u['x_min']) + (1 << int(u['log2'])) * float(u['bs'])
        parts = [f'The portfolio is {self.tail_explanation}']
        if {'mm', 'rms', 'sum'}.issubset(df.index):
            mm_w = float(df.loc['mm', 'W'])
            rms_w = float(df.loc['rms', 'W'])
            sum_w = float(df.loc['sum', 'W'])
            order = 'mm <= rms <= sum' if mm_w <= rms_w <= sum_w + 1e-9 \
                else 'mm/rms/sum out of order (check moments)'
            parts.append(
                f'The bulk span is sized by Portfolio MM (width {mm_w:g}); the '
                f'RMS-of-windows reference is {rms_w:g} and the legacy linear '
                f'sum {sum_w:g} ({order}); the mm-rms gap {rms_w - mm_w:g} is '
                f'the skewness/diversification adjustment.')
        if 'sbj' in df.index and float(df.loc['sbj', 'x_max']) > float(df.loc['mm', 'x_max']):
            parts.append(
                f'The single-big-jump look-through floored the upper extent at '
                f'{float(df.loc["sbj", "x_max"]):g} (a heavy unit reaches past '
                f'the MM bulk).')
        placement = ('windowed (Plan B, mass clears 0)' if float(u['x_min']) > 0
                     else '0-based (Plan A)')
        parts.append(
            f'Realised {placement} grid: bs={float(u["bs"]):g}, '
            f'log2={int(u["log2"])}, x_min={float(u["x_min"]):g}, top={top:g}.')
        clip = getattr(self, '_bs_clip', None)
        if clip is not None:
            cm = clip.get('clipped_mass', float('nan'))
            cm_txt = f'~{cm:.3g}' if np.isfinite(cm) else 'a sliver'
            parts.append(
                f'The combined support exceeds the grid top {top:g}: {cm_txt} of '
                f'the mass is clipped (a reported deficit, not normalized) -- '
                f'raise log2 to {int(clip["need_log2"])} to capture it.')
        return ' '.join(parts)

    @property
    def tail_df(self) -> 'pd.DataFrame':
        """Per-unit aggregate tail rows plus a worst-of ``total`` (``[bs-reporting]``).

        One row per unit -- that unit's aggregate-level
        :attr:`Aggregate.tail_df` row (support ``min`` / ``max``, ``left_tail`` /
        ``right_tail`` classes, ``bounded``, ``concentrated`` /
        ``concentration_p``) -- and a ``total`` row carrying the portfolio
        worst-of :attr:`tail_class` and the total-moment concentration. Spec-only
        (valid before :meth:`update`). See :attr:`tail_description` /
        :attr:`tail_explanation` for the narrative.

        Returns
        -------
        pandas.DataFrame
            Indexed by unit name, with a final ``total`` row.
        """
        rows = {a.name: a.tail_df.loc['aggregate'] for a in self.agg_list}
        df = pd.DataFrame(rows).T
        worst = self.tail_class
        conc_flag, conc_p = _tail.concentration(float(self.agg_m), float(self.agg_sd))
        total = pd.Series(INFO_NA, index=df.columns, dtype=object)
        worst_label = _tail.tail_class_label(worst)
        if 'left_tail' in df.columns:
            total['left_tail'] = worst_label
        if 'right_tail' in df.columns:
            total['right_tail'] = worst_label
        if 'bounded' in df.columns:
            total['bounded'] = (worst == TailClass.BOUNDED)
        if 'concentrated' in df.columns:
            total['concentrated'] = bool(conc_flag)
        if 'concentration_p' in df.columns:
            total['concentration_p'] = float(conc_p)
        if 'note' in df.columns:
            total['note'] = 'portfolio worst-of (independence)'
        df.loc['total'] = total
        return df

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
    def value_type(self):
        """Sign convention for the portfolio: ``'loss'`` or ``'payoff'``.

        Derived, not user-set: the unanimous
        :attr:`~aggregate.distributions.Aggregate.value_type` of the units
        (mixed books are rejected at construction; an empty portfolio
        defaults to loss). Read-only — there is no setter, unlike
        ``Aggregate``. It is **inert for the distribution itself**: density,
        moments, quantiles and allocation do not depend on it. It is consumed
        only by distortion / pricing, where a payoff portfolio is negated /
        the dual distortion applied. Returns the configured label for the
        role (``[labels]`` in the config); pricing code must branch on
        ``_is_loss_value``, never on the label text.
        """
        return value_type_label(self._is_loss_value)

    @property
    def info(self):
        """Fixed-layout multi-line summary string.

        Every row is always present, in the same order, for every
        ``Portfolio``; a value that is not (yet) available -- e.g. the grid
        block before ``update`` -- renders as ``n/a``. The row catalogue is
        documented in ``dev/info-strings.rst``. Shares the label/value
        convention (:func:`aggregate.constants.info_row`) with ``Aggregate``
        and ``Distortion``.
        """
        updated = self.bs > 0
        if updated:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1/self.bs)}'
        else:
            bss = INFO_NA
        # Realized grid window, read off the density index (the signed path
        # may start below 0; the non-signed grid starts at 0).
        if self.density_df is not None:
            x_min = f'{float(self.density_df.index[0]):,.6g}'
            x_max = f'{float(self.density_df.index[-1]) + self.bs:,.6g}'
        else:
            x_min = x_max = INFO_NA
        # premium / expected loss / loss ratio: populated when the units
        # carry a premium. Expected loss is empirical (est_m); for a payoff
        # book est_m is E[PnL], so E[loss] = premium - E[PnL].
        prem = float(self.stats_df.loc[('meta', 'prem'), 'total'])
        e_loss = None
        if updated:
            if self._is_loss_value:
                e_loss = float(self.est_m)
            elif prem > 0:
                e_loss = prem - float(self.est_m)
        h = self.hash_rep_at_last_update
        rows = [
            ('portfolio object name', self.name),
            ('value_type', self.value_type),
            ('aggregate objects', f'{len(self.unit_names):d}'),
            ('allocation_method', self.allocation_method),
            ('bs', bss),
            ('log2', self.log2 if updated else INFO_NA),
            ('padding', self.padding if updated else INFO_NA),
            ('sev_calc', self.sev_calc if updated else INFO_NA),
            ('normalize', self.normalize if updated else INFO_NA),
            ('x_min', x_min),
            ('x_max', x_max),
            ('premium', f'{prem:,.6g}' if prem > 0 else INFO_NA),
            ('expected loss', f'{e_loss:,.6g}' if e_loss is not None else INFO_NA),
            ('loss ratio', f'{e_loss / prem:.1%}'
             if prem > 0 and e_loss is not None else INFO_NA),
        ]
        s = [info_row(label, value) for label, value in rows]
        # Footer: tail, bounded, last update, id -- matching Aggregate.info
        # (tail near the end; ``id`` was labelled ``hash``).
        s.append(self.tail_description)
        s.append(info_row('bounded', self.bounded))
        s.append(info_row('last update',
                          self.last_update if self.last_update != 0 else INFO_NA))
        s.append(info_row('id', f'{h:x}' if isinstance(h, int) else INFO_NA))
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

        :param approx_type: norm | lognorm | gamma | slognorm | sgamma | all
        :param output: return a dict or agg language specification
        :return:

        A shifted family (``slognorm`` / ``sgamma``) requested for a symmetric
        portfolio total degenerates to its normal limit and emits a
        ``UserWarning`` (pass ``approx_type='norm'`` for the normal explicitly).
        A left-skewed total is fitted by reflection, which has no native frozen
        ``scipy`` / one-line DecL form; use ``output='sev_kwargs'`` or the
        default Aggregate object for that case. Mirrors
        :meth:`Aggregate.approximate`.
        """
        emp_mean = self.stats_df.loc[('agg', 'mean'), 'empirical']
        if pd.isna(emp_mean):
            # not updated — use theoretical moments from the total column
            m = float(self.stats_df.loc[('agg', 'mean'), 'total'])
            cv = float(self.stats_df.loc[('agg', 'cv'), 'total'])
            skew = float(self.stats_df.loc[('agg', 'skew'), 'total'])
        else:
            # use empirical (post-FFT) moments matched to the computed aggregate
            m = float(emp_mean)
            cv = float(self.stats_df.loc[('agg', 'cv'), 'empirical'])
            skew = float(self.stats_df.loc[('agg', 'skew'), 'empirical'])
        note = f'frozen version of {self.name}'

        def _one(kind, warn):
            nm = f'{kind[0:4]}.{self.name[0:5]}'
            return approximate_from_mcvsk(m, cv, skew, nm, f'agg {nm} 1 claim sev ',
                                          note, kind, output, warn_degenerate=warn)

        if approx_type == 'all':
            # Survey: quiet about degeneration; skip a family that cannot be
            # represented for this distribution/output (e.g. reflected + scipy).
            out = {}
            for kind in ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']:
                try:
                    out[kind] = _one(kind, warn=False)
                except ValueError:
                    continue
            return out
        return _one(approx_type, warn=True)

    def percentiles(self, pvalues=None):
        """
        Per-unit percentiles (interpolated) of the FFT-derived
        ``density_df`` distribution.

        :param pvalues: optional vector of log values to use. If None sensible defaults provided
        :return: DataFrame of percentiles indexed by unit and log
        """
        df = pd.DataFrame(columns=['unit', 'log', 'Agg Quantile'])
        df = df.set_index(['unit', 'log'])
        # df.columns.name = 'perspective'
        if pvalues is None:
            pvalues = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.994, 0.995, 0.999, 0.9999]
        for unit in self.unit_names_ex:
            # total from the portfolio frame; units from their native pmfs
            # (numerics-1) -- deliberately interpolated, unlike the exact
            # step-function q.
            if unit == 'total':
                ser = self.density_df.p_total
            else:
                ser = self.unit_density(unit)
            q_agg = interpolate.interp1d(ser.cumsum(), ser.index,
                                         kind='linear', bounds_error=False, fill_value='extrapolate')
            for p in pvalues:
                qq = q_agg(p)
                df.loc[(unit, p), :] = [float(qq)]
        df = df.unstack(level=1)
        return df

    def recommend_bucket(self):
        """
        Data to help estimate a good bucket size.

        :return:
        """
        df = pd.DataFrame(columns=['unit', 'bs10'])
        df = df.set_index('unit')
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

    def _single_big_jump_window(self, p_star):
        """Portfolio single-big-jump extent floor by look-through to the units.

        The subexponential tail of an independent sum is the *sum* of the unit
        tails, dominated by the heaviest unit:
        ``P(S_tot > x) ~ sum_k E[N_k]*P(X_k > x)``. So the single-big-jump
        scenario is one big claim in some unit ``k`` riding the *typical* bulk
        of everything else -- the portfolio mean ``agg_m`` with one typical
        claim (unit ``k``'s severity mean) replaced by one big one::

            sbj_hi_port = agg_m + max_k ( sbj_hi_k - ES_k )      # MAX, not sum

        where ``sbj_hi_k`` is unit ``k``'s own
        :meth:`Aggregate._single_big_jump_window` upper edge called with the
        **portfolio** ``p_star`` (each unit forms its own
        ``p**_k = 1 - (1 - p_star)/E[N_k]`` from *its* frequency),
        ``ES_k = a.agg_m`` is the unit's aggregate mean, and
        ``sbj_hi_k - ES_k = q_{X_k}(p**_k) - mu_{X_k}`` is that unit's "jump
        excess" over a typical claim. The lower edge mirrors over **signed**
        units only (a non-negative unit reaches no lower than its 0 floor); for
        an all-non-negative book ``sbj_lo = 0``.

        ``max_k`` is tight when one unit dominates the tail; when two or more
        comparably-heavy units drive it, the exact extent is the pooled
        root-find ``sum_k E[N_k](1 - F_{X_k}(x)) = 1 - p_star`` (a documented
        refinement, not yet wired -- the ``max`` is a safe lower bound on the
        true reach for the dominant-unit case and the doubling-padding absorbs
        the rest; see ``dev/plan-bucket-window-2.md`` Round 3).

        Parameters
        ----------
        p_star : float
            Portfolio aggregate coverage (``> 1`` read as a number of nines).

        Returns
        -------
        (sbj_lo, sbj_hi) : tuple of float, or None
            The portfolio single-big-jump window edges, or ``None`` when no
            unit yields a finite SBJ window (no finite ``E[N]`` / variance).
        """
        m = float(self.agg_m)
        if not np.isfinite(m):
            return None
        hi_excess, lo_excess = [], []
        for a in self.agg_list:
            sbj = a._single_big_jump_window(p_star)
            if sbj is None:
                continue
            es_k = float(a.agg_m)
            if not np.isfinite(es_k):
                continue
            hi_excess.append(float(sbj[1]) - es_k)
            if a._signed_severity():
                lo_excess.append(float(sbj[0]) - es_k)
        if not hi_excess and not lo_excess:
            return None
        sbj_hi = m + max(hi_excess) if hi_excess else m
        sbj_lo = m + min(lo_excess) if lo_excess else 0.0
        return float(sbj_lo), float(sbj_hi)

    def best_window(self, log2=16, bs_in=0, bucket_sizing_p=BUCKET_SIZING_P):
        """Decide the portfolio combine grid by **Portfolio MM** + SBJ look-through.

        Mirrors the per-aggregate :meth:`Aggregate._bs_window` *bulk / extent*
        split at the portfolio level. The bulk is sized from the **exact total
        moments**, never by combining per-unit windows (neither by linear sum --
        which overstates by ``sqrt(k)`` for ``k`` iid units, ignoring
        diversification -- nor by root-sum-square):

        1. **Bulk window from Portfolio MM.** Feed the analytic compound total
           moments (``agg_m``, ``agg_sd = agg_m*agg_cv``, ``agg_skew``; cumulants
           add under independence) straight into the *same*
           :func:`estimate_agg_window` the single-aggregate sizer uses. This
           gives the two-sided ``[mm_lo, mm_hi]`` where the combined mass lives.
        2. **Resolution floor.** Per-unit widths enter *only* as the finest
           bucket ``min_k bs_k`` -- a unit's own lattice must survive ("don't
           lose sev ``bs``", now portfolio-wide), never the span.
        3. **One portfolio SBJ extent floor** (the look-through,
           :meth:`_single_big_jump_window`): ``sbj_hi_port = agg_m + max_k(
           sbj_hi_k - ES_k)`` -- the heaviest unit's one big claim on the
           combined bulk, **max** not sum, so the per-unit a59 extents are not
           double-counted. Self-activating: a thin / well-diversified total has
           ``sbj <= mm`` and the floor does not move the grid.

        The window is ``[x_lo, x_hi] = [min(mm_lo, sbj_lo), max(mm_hi, sbj_hi)]``.

        ``bs`` discipline -- **carry raw, round once**: the MM span term
        ``W_ext / N`` is rounded by ``round_bucket`` a *single* time at the top
        (no per-unit + combine double-round); the resolution floor stays the
        finest per-unit *lattice* value. ``bs = round_bucket(max(min_k bs_k,
        W_ext / N))`` (a pinned ``bs_in > 0`` is honoured verbatim, D4).

        Origin and ``log2``:

        - **non-signed, ordinary** (mass reaches 0, or not concentrated): the
          grid starts at ``x_min = 0`` (Plan A); ``log2`` is **shrunk** to just
          hold ``x_hi`` at ``bs`` (capped) -- a tiny discrete book no longer
          inflates to the cap.
        - **non-signed, concentrated and clear of 0** (the high-frequency /
          tiny-cv case, e.g. ``Poisson(100000)``): the total is **windowed**
          (Plan B), ``x_min = floor(x_lo / bs) * bs > 0``, routed by ``update``
          through the signed roll-combine path. The concentration gate
          (:func:`aggregate.tail.concentration`) is the same one
          :meth:`Aggregate._bs_window` uses, so a merely-grazing book reverts to
          the 0-based grid.
        - **signed** (P&L): the origin is the windowed low edge ``x_lo`` floored;
          ``update`` recomputes the realised origin from the units' post-snap
          ``x_min``. ``log2`` stays at the cap and the span is floored at the
          conservative ``max_k W_k / N`` so no per-unit marginal wraps.

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
            Shared grid parameters. ``x_min > 0`` for a non-signed total signals
            ``update`` to take the windowed roll path (Plan B).
            :attr:`_bs_window_df` (unit rows, the MM / RMS / SBJ / sum candidate
            rows, and a ``used`` row) is also populated.

        See Also
        --------
        best_bucket : the deprecated RMS combine (kept for comparison).
        _single_big_jump_window : the portfolio SBJ look-through.
        Aggregate._bs_window : the per-unit window estimator consumed here.
        """
        signed = self._signed()
        N_cap = 1 << log2
        p_star = 1.0 - 10.0 ** -WINDOW_NINES

        # ---- phase 1: per-unit natural windows (analytic, no FFT) ---------
        # Each unit sizes itself on its own (0-origin or signed) grid; we read
        # the *selected method* row, not the padded ``used`` row, for the width.
        # Quiet the pre-pass: a unit may emit a clip warning here that the real
        # combine re-issues once (deduped) -- silence the speculative pass.
        rows = []
        bs_ks, x_min_ks, W_ks = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DefectiveDistributionWarning)
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

        # ---- the bulk window: Portfolio MM (NOT a width-combine) ----------
        m = float(self.agg_m)
        sd = float(self.agg_sd)
        skew = float(self.agg_skew)
        try:
            mm_lo, mm_hi, _W_mm = estimate_agg_window(m, sd, skew, p_star)
        except (ValueError, FloatingPointError):
            # No finite window (e.g. infinite variance): fall back to the
            # conservative linear sum of per-unit widths.
            mm_lo, mm_hi = 0.0, float(sum(W_ks))

        # ---- the portfolio SBJ extent floor (the look-through) ------------
        sbj = self._single_big_jump_window(p_star)
        if sbj is not None:
            sbj_lo, sbj_hi = sbj
        else:
            sbj_lo, sbj_hi = mm_lo, mm_hi

        # ---- the two inspectable reference windows ------------------------
        # RMS(w_i): the normal-approx window combine (the k cancels, so it is
        # ``m +/- sqrt(sum w_i^2)``); runs above the MM window because the
        # per-unit windows bake in skew the sum de-skews away by CLT. Carried as
        # a standing candidate row -- the gap ``MM - RMS`` reads as the
        # skewness/diversification adjustment. ``sum w_i``: the legacy linear
        # span, a guaranteed-no-wrap upper bound on the support width.
        half_rms = float(np.sqrt(sum(w * w for w in W_ks))) / 2.0 if W_ks else 0.0
        rms_lo, rms_hi = m - half_rms, m + half_rms
        W_sum = float(sum(W_ks))

        # ---- the chosen extent --------------------------------------------
        # The SBJ floor only governs the *upper* extent for a non-negative book
        # (its ``sbj_lo`` is the 0 floor, which must not pull the windowed origin
        # down off the bulk); for a signed book it floors the lower edge too.
        x_hi = max(mm_hi, sbj_hi)
        x_lo_raw = min(mm_lo, sbj_lo) if signed else mm_lo
        # Concentration gate (same source as Aggregate._bs_window): only a
        # genuinely concentrated total that clears 0 is windowed (Plan B).
        conc_flag, _conc_p = _tail.concentration(m, sd)
        window_nonsigned = (not signed and bs_in <= 0 and bool(conc_flag)
                            and x_lo_raw > 0)

        # ---- bs: carry raw, round once ------------------------------------
        resolution = min(bs_ks) if bs_ks else 1.0
        if signed:
            x_lo = x_lo_raw
        elif window_nonsigned:
            x_lo = x_lo_raw                          # windowed origin (Plan B)
        else:
            x_lo = 0.0                               # 0-based (Plan A)
        W_ext = max(x_hi - x_lo, 0.0)
        if bs_in > 0:
            bs = float(bs_in)
        else:
            span = W_ext / N_cap if N_cap else W_ext
            if signed:
                # Wrap safety: every per-unit marginal is driven on the shared
                # grid, so N*bs must hold the widest unit too. Floor the span at
                # ``max_k W_k / N`` (the MM span is usually wider, but guard the
                # one-dominant-unit case).
                span = max(span, (max(W_ks) if W_ks else 0.0) / N_cap)
            bs = round_bucket(max(resolution, span))

        # ---- origin and log2 ----------------------------------------------
        if signed:
            x_min = float(np.floor(x_lo / bs) * bs)
            log2_out = log2
        elif window_nonsigned:
            x_min = float(np.floor(x_lo / bs) * bs)
            if bs_in > 0:
                log2_out = log2
            else:
                need = (int(np.ceil(np.log2((x_hi - x_min) / bs + 1.0)))
                        if x_hi > x_min else 1)
                log2_out = min(log2, max(need, 1))
        else:
            x_min = 0.0
            if bs_in > 0:
                log2_out = log2                       # user pinned the grid
            else:
                need = int(np.ceil(np.log2(x_hi / bs + 1.0))) if x_hi > 0 else 1
                log2_out = min(log2, max(need, 1))

        cand = dict(mm=(mm_lo, mm_hi), rms=(rms_lo, rms_hi),
                    sbj=(sbj_lo, sbj_hi), sum=(min(x_lo, 0.0), W_sum + min(x_lo, 0.0)))
        self._build_bs_window_df(rows, bs, log2_out, x_min, cand,
                                 resolution, W_ext)
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

    def _build_bs_window_df(self, rows, bs, log2, x_min, cand, resolution, W_ext):
        """Build the unit-indexed bucket/window summary for the combine.

        Mirrors :attr:`Aggregate._bs_window_df`'s idiom but swaps *method*
        rows for *unit* rows -- the Portfolio convention of one row per unit
        plus a summary line (cf. ``stats_df`` / ``describe``, which carry
        per-unit columns and a ``total``). Each unit row is that unit's
        selected window; then four **candidate** combine rows -- ``mm`` (the
        Portfolio MM bulk, the live span), ``rms`` (the RMS-of-windows
        normal-approx reference), ``sbj`` (the single-big-jump look-through),
        and ``sum`` (the legacy linear-sum no-wrap bound) -- so the combine's
        journey is inspectable; finally the ``used`` row is the realised shared
        portfolio grid ``[x_min, x_min + 2**log2 * bs)``. The ``mm <= rms <=
        sum`` ordering and the ``mm - rms`` skewness/diversification gap can be
        read straight off the frame.

        Parameters
        ----------
        rows : list of dict
            Per-unit window rows (``unit``/``x_min``/``x_max``/``W``/``bs``/
            ``log2``/``coverage``/``note``).
        bs, log2 : float, int
            The realised shared grid bucket size and log2.
        x_min : float
            The realised portfolio-grid origin.
        cand : dict
            ``{'mm': (lo, hi), 'rms': (lo, hi), 'sbj': (lo, hi),
            'sum': (lo, hi)}`` -- the candidate window edges.
        resolution : float
            The resolution floor ``min_k bs_k``.
        W_ext : float
            The chosen extent width (for the ``used``-row note).
        """
        N = 1 << log2
        cols = ['x_min', 'x_max', 'W', 'bs', 'log2', 'coverage', 'note']
        df = pd.DataFrame(rows).set_index('unit')[cols] if rows \
            else pd.DataFrame(columns=cols)
        cov = f'1-1e-{WINDOW_NINES}'
        cand_note = {
            'mm': 'Portfolio MM bulk (3-moment fit on total) -- the live span',
            'rms': 'RMS-of-windows reference (normal approx); mm-rms = skew adj',
            'sbj': 'single big jump look-through: agg_m + max_k(sbj_hi_k - ES_k)',
            'sum': 'legacy linear sum of widths (guaranteed-no-wrap bound)',
        }
        for key in ('mm', 'rms', 'sbj', 'sum'):
            lo, hi = cand[key]
            df.loc[key] = dict(
                x_min=float(lo), x_max=float(hi), W=float(hi - lo),
                bs=float(bs), log2=int(log2), coverage=cov, note=cand_note[key])
        df.loc['used'] = dict(
            x_min=float(x_min), x_max=float(x_min + N * bs),
            W=float(N * bs), bs=float(bs), log2=int(log2), coverage=cov,
            note=f'realised portfolio grid (resolution={resolution:g}, '
                 f'extent={W_ext:g})')

        # ---- journey columns (parity with Aggregate.bs_window_df) ---------
        # ``log2_need`` is the log2 a row's window needs at the shared ``bs``;
        # ``clipped`` (the realised far-tail deficit) is patched by ``update``.
        def _need(r):
            w, b = float(r['x_max']) - float(r['x_min']), float(r['bs'])
            if not (np.isfinite(w) and np.isfinite(b) and b > 0 and w > 0):
                return np.nan
            return float(np.ceil(np.log2(w / b + 1.0)))
        df['log2_need'] = df.apply(_need, axis=1)
        df['clipped'] = np.nan
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
        that adjusts individual unit densities; applied after unit aggs created but before creating not-units;
        actual statistics do not reflect impact.) Agg re by unit is now applied in the Aggregate object.

        TODO: consider aggregate covers at the portfolio level...Where in parse - at the top!


        :param log2:
        :param bs: bucket size
        :param remove_fuzz: remove machine noise elements from FFT
        :param sev_calc: how to calculate the severity, discrete (point masses as xs) or continuous (uniform between xs points)
        :param discretization_calc:  survival or distribution (accurate on right or left tails)
        :param normalize: if true, normalize the severity so sum probs = 1. This is generally what you want; but
        :param padding: for fft 1 = double, 2 = quadruple
        :param trim_df: remove unnecessary columns from density_df before returning
        :param add_exa: run add_exa to append the objective allocation columns needed for pricing
        :param force_severity: force computation of severities for aggregate components even when approximating
        :param bucket_sizing_p: percentile to use for bucket recommendation.
        :param debug: if True, print debug information
        :return:
        """
        self._valid = None # reset valid flag

        if log2 <= 0:
            raise ValueError('log2 must be >= 0')
        self.log2 = log2
        # Grid sizing routes through ``best_window`` (Portfolio MM bulk + SBJ
        # look-through). ``self._combine_x_min`` records the windowed origin for
        # a *non-signed* total whose mass clears 0 (Plan B): when set, ``update``
        # takes the same roll-combine path as a signed book, placing the shared
        # grid at ``x_min > 0`` instead of forcing the legacy 0-based grid.
        signed = self._signed()
        self._combine_x_min = None
        if signed:
            bs, log2, _x_min_est = self._bs_window(log2, bs, bucket_sizing_p)
            self.log2 = log2
            self.bs = bs
        elif bs == 0:
            # Non-signed auto-size: Portfolio MM bulk + SBJ look-through
            # (best_window), which shrinks log2 to just hold the extent -- a tiny
            # discrete book no longer inflates to the log2 cap. A concentrated
            # total that clears 0 is windowed (x_min > 0, Plan B).
            bs, log2, x_min_est = self.best_window(log2, 0, bucket_sizing_p)
            self.log2 = log2
            self.bs = bs
            if x_min_est > 0:
                self._combine_x_min = float(x_min_est)
            logger.info(f'bs=0 entered, setting bs={bs:.6g}, log2={log2}, '
                        f'x_min={x_min_est:.6g} via best_window (Portfolio MM '
                        f'+ SBJ look-through).')
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

        # Per-unit state for the kappa construction in ``add_exa``: the
        # unit's native grid / pmf plus the padded FT of its pmf. Captured
        # at combine time (it cannot be reconstructed after rebasing) and
        # **transient within this call** — freed before ``update`` returns
        # (meta D6); only scalars (``agg.x_min``) and the native pmfs
        # persist, on the Aggregate objects themselves.
        unit_state = {}

        # Build the grid and the per-unit densities, accumulating their
        # product in Fourier space to get ``p_total``.
        N = 1 << log2
        # The roll-combine path serves both a signed (P&L) book and a windowed
        # non-signed total (Plan B, ``self._combine_x_min`` set by best_window).
        use_roll = signed or self._combine_x_min is not None
        if use_roll:
            # ---- roll combine on a shared (signed or windowed) grid --------
            # Drive each unit on its OWN window [x_min_k, ...) sharing the
            # portfolio bs/log2/padding, so the unit object stays internally
            # correct (no false deficit, right moments, right describe/plot --
            # plan 2c). The combine reads each unit's ftagg_density, which is
            # origin-at-0 *regardless* of the unit's x_min (the output roll hits
            # the density, never ftagg), so the units' FFTs still multiply
            # correctly here -- and a heavy unit severity that cannot itself
            # window stays on its own 0-based grid, so the portfolio windowing
            # is not blocked by it (the per-aggregate Regime-B limitation does
            # not bind the combine).
            ft_all = None
            x_mins = []
            for agg in self.agg_list:
                agg.update(log2=log2, bs=self.bs, padding=self.padding,
                           sev_calc=sev_calc,
                           discretization_calc=discretization_calc,
                           normalize=normalize, force_severity=force_severity,
                           x_min='auto', bucket_sizing_p=bucket_sizing_p, debug=debug)
                # de-fuzz the native pmf when requested — same convention
                # as the de-fuzzed density_df the kappa numerator used to
                # read (the first-moment weighting amplifies far-tail dust)
                p_unit = (remove_fuzz_util(agg.agg_density)
                          if self._remove_fuzz else agg.agg_density)
                unit_state[agg.name] = dict(
                    xs=agg.xs, p=p_unit, ft_p=agg.ftagg_density)
                x_mins.append(agg.x_min)
                if ft_all is None:
                    ft_all = np.copy(agg.ftagg_density)
                else:
                    ft_all *= agg.ftagg_density
            # Realised portfolio origin. Signed: the support min of the
            # independent sum is Sigma x_min_k; floored by min_k x_min_k so no
            # per-unit marginal wraps (safety floor on the plan's sum-of-
            # windows). Windowed non-signed (Plan B): the Portfolio MM windowed
            # origin from best_window (each unit sits at its own 0-origin; the
            # *total* mass is what clears 0). Snapped to bs.
            if signed:
                x_min_tot = float(min(sum(x_mins), min(x_mins)))
            else:
                x_min_tot = float(self._combine_x_min)
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
            self._signed_window = (x_min_tot, x_min_tot + N * self.bs)
        else:
            # ---- non-signed path (combine unchanged, byte-for-byte) -------
            # Use self.bs (resolved above): build_many now passes bs through
            # as 0 => auto, so the grid must read the resolved bucket, not the
            # raw parameter.
            MAXL = N * self.bs
            xs = np.linspace(0, MAXL, N, endpoint=False)
            self.density_df = pd.DataFrame(index=xs)
            self.density_df['loss'] = xs
            ft_all = None
            for agg in self.agg_list:
                agg.update_work(xs, self.padding, sev_calc, discretization_calc,
                                normalize, force_severity, debug=debug)
                # de-fuzz as for the signed path above (legacy parity: the
                # kappa numerator read the de-fuzzed p_{unit} columns)
                p_unit = (remove_fuzz_util(agg.agg_density)
                          if self._remove_fuzz else agg.agg_density)
                unit_state[agg.name] = dict(
                    xs=xs, p=p_unit, ft_p=agg.ftagg_density)
                if ft_all is None:
                    ft_all = np.copy(agg.ftagg_density)
                else:
                    ft_all *= agg.ftagg_density
            self.density_df['p_total'] = np.real(ift(ft_all, self.padding))

        self.remove_fuzz(log='update')

        # Objective allocation columns (kappa + direct sums) — now valid on
        # signed (P&L) windows too (the shifted-support kappa carries unit
        # origins; share-based columns are blanked per steering 6).
        if add_exa:
            self.add_exa(self.density_df, unit_state)
        else:
            # at least want F and S to get quantile functions
            self.density_df['F'] = np.cumsum(self.density_df.p_total)
            self.density_df['S'] = 1 - self.density_df.F
        # D6: the padded FT state is transient — drop it before returning.
        del unit_state, ft_all

        # Mass-conservation check on the rolled grid (signed or windowed
        # non-signed): a window narrower than the combined support shows up as a
        # deficit (the wrapped tail would be truncated by the final [:N]).
        # Surface it loudly, like the Aggregate path, and record it as the
        # portfolio far-tail clip for the bs report.
        if use_roll:
            deficit = 1.0 - float(np.sum(self.density_df['p_total']))
            if deficit > VALIDATION_NOISE:
                top = float(x_min_tot + N * self.bs)
                self._bs_clip = dict(
                    reach=float('nan'), grid_top=top, log2=int(log2),
                    bs=float(self.bs), need_log2=int(log2 + 1),
                    clipped_mass=float(deficit))
                if self._bs_window_df is not None \
                        and 'used' in self._bs_window_df.index:
                    self._bs_window_df.loc['used', 'clipped'] = float(deficit)
                kind = 'signed' if signed else 'windowed'
                warnings.warn(
                    f'{self.name}: portfolio PMF deficit {deficit:.3e} '
                    f'(Σp = 1 − {deficit:.3e} < 1); the {kind} window is '
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
        cols = list(self.unit_names) + [
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
        unit_cols = list(self.unit_names)
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

        :return:
        """
        self.density_df = self.density_df.drop(
            self.density_df.filter(regex='^e_|^exi_xlea|^[a-z_]+ημ').columns,
            axis=1
        )

    @property
    def pprogram(self):
        """Canonical DecL program text, rendered from the parsed spec.

        Re-parses :attr:`program` and renders through
        :func:`aggregate.decl_writer.format_program` (the inverse of the
        parser): one ``port`` line then one tab-indented unit per line. Canonical
        rather than verbatim. A portfolio built programmatically (empty
        ``program``) returns ``''``.
        """
        return format_program(self.program, fmt='text')

    @property
    def pprogram_html(self):
        """Syntax-highlighted DecL program for IPython / Jupyter display."""
        return format_program(self.program, fmt='html')

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
            # total + native unit pmfs (numerics-1)
            pmfs = [self.density_df.p_total] + \
                   [self.unit_density(unit) for unit in self.unit_names]
            mx = max(float(s.max()) for s in pmfs)
            mxx0 = max(float(s.iloc[1:].max()) for s in pmfs)
            if kind == 'linear':
                if zero_mass == 'include':
                    return f(mx)
                else:
                    return f(mxx0)
            else:
                return [eps, mx * 1.5]
        elif stat == 'logy':
            pmfs = [self.density_df.p_total] + \
                   [self.unit_density(unit) for unit in self.unit_names]
            mx = min(1, max(float(s.max()) for s in pmfs))
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

    # ================================================================
    # Unit (native-grid) density accessors -- numerics-1.
    # Unit pmfs live on each Aggregate's own grid; these are the only
    # ways to read them from the Portfolio. The legacy
    # ``density_df['p_{unit}']`` columns were removed at numerics-2
    # (dev/done/plan-numerics-2-objective.md).
    # ================================================================

    def unit_density(self, unit, view='agg'):
        """Native-grid pmf of one unit, read from the owning :class:`Aggregate`.

        The unit's pmf lives on the unit's **own** loss grid (which on a
        windowed or signed book differs from the portfolio total grid).
        This accessor is the supported source for per-unit densities; the
        legacy ``density_df['p_{unit}']`` total-grid columns were removed
        at numerics-2 (use :meth:`aligned_unit_density_df` for an
        explicitly-labelled total-grid display view).

        Parameters
        ----------
        unit : str
            Unit name; one of :attr:`unit_names`.
        view : {'agg', 'sev'}
            ``'agg'`` reads the unit's aggregate pmf
            (``Aggregate.density_df.p_total`` on grid ``xs``); ``'sev'``
            reads the discretized severity
            (``Aggregate.sev_density_df.p_sev`` on grid ``xs_sev``).

        Returns
        -------
        pandas.Series
            pmf named ``p_{unit}``, indexed by the unit's native loss grid.
        """
        if unit not in self.unit_names:
            raise KeyError(
                f'unknown unit {unit!r}; expected one of {self.unit_names}')
        agg = self[unit]
        if view == 'agg':
            ser = agg.density_df['p_total'].copy()
        elif view == 'sev':
            ser = agg.sev_density_df['p_sev'].copy()
        else:
            raise ValueError(f"view must be 'agg' or 'sev', not {view!r}")
        ser.name = f'p_{agg.name}'
        return ser

    def unit_density_df(self, view='agg'):
        """Long-form frame of all unit pmfs, each on its native grid.

        One block per unit, concatenated with a ``(unit, loss)``
        MultiIndex. Each block carries the unit's pmf plus the
        window-audit metadata needed to reason about grid alignment.

        Parameters
        ----------
        view : {'agg', 'sev'}
            As in :meth:`unit_density`.

        Returns
        -------
        pandas.DataFrame
            Index ``(unit, loss)``; columns ``unit, loss, p, F, S, bs,
            x_min, x_max, mass`` where ``x_min`` / ``x_max`` are the unit's
            grid edges and ``mass`` is the unit's represented probability
            ``p.sum()`` (less than 1 on a too-narrow window).
        """
        blocks = []
        for agg in self.agg_list:
            if view == 'agg':
                src = agg.density_df
                df = src[['loss', 'p_total', 'F', 'S']].rename(
                    columns={'p_total': 'p'}).copy()
            elif view == 'sev':
                src = agg.sev_density_df
                df = src[['loss', 'p_sev', 'F_sev', 'S_sev']].rename(
                    columns={'p_sev': 'p', 'F_sev': 'F', 'S_sev': 'S'}).copy()
            else:
                raise ValueError(f"view must be 'agg' or 'sev', not {view!r}")
            df['unit'] = agg.name
            df['bs'] = agg.bs
            df['x_min'] = float(df['loss'].iloc[0])
            df['x_max'] = float(df['loss'].iloc[-1])
            df['mass'] = float(df['p'].sum())
            blocks.append(df[['unit', 'loss', 'p', 'F', 'S', 'bs',
                              'x_min', 'x_max', 'mass']])
        out = pd.concat(blocks)
        out = out.set_index(['unit', 'loss'], drop=False)
        return out

    def aligned_unit_density_df(self, grid='total', *,
                                allow_window_mismatch=False):
        """Unit pmfs scattered onto a common grid — a **display adapter**.

        Reindexes each unit's native pmf onto the requested grid. The
        result is a labelled presentation artifact, never compute input:
        on a windowed book a unit's grid can extend beyond the requested
        grid, and the off-grid buckets are silently dropped (the method
        warns; see below).

        Alignment is by bucket number ``round(loss / bs)`` — the Portfolio
        dictates a common ``bs`` to its units, asserted here — so
        physically identical points always land together regardless of how
        each grid's floats were built.

        Parameters
        ----------
        grid : {'total', 'union', 'zero'}
            ``'total'`` — the portfolio ``density_df`` grid (for a legacy
            zero-origin book this reproduces the ``p_{unit}`` columns
            exactly); ``'union'`` — the union of the unit grids (never
            drops a bucket); ``'zero'`` — the bucket lattice from
            ``min(0, lowest unit x_min)`` through the highest unit
            ``x_max`` (always contains the origin).
        allow_window_mismatch : bool, keyword only
            When unit buckets fall outside the requested grid the method
            warns unless this is ``True`` (acknowledging the view is a
            window-clipped artifact).

        Returns
        -------
        pandas.DataFrame
            Indexed by the requested grid's loss values, one ``p_{unit}``
            column per unit; zero where a unit has no bucket.
        """
        bs = self.bs
        for agg in self.agg_list:
            if not np.isclose(agg.bs, bs, rtol=1e-12, atol=0):
                raise ValueError(
                    f'unit {agg.name!r} has bs={agg.bs} != portfolio '
                    f'bs={bs}; it was re-updated off the portfolio grid')
        # per-unit pmfs on the integer bucket lattice
        unit_j = {}
        unit_p = {}
        for agg in self.agg_list:
            ser = agg.density_df['p_total']
            unit_j[agg.name] = np.round(
                ser.index.to_numpy() / bs).astype(np.int64)
            unit_p[agg.name] = ser.to_numpy()
        if grid == 'total':
            target_x = self.density_df.index.to_numpy()
            target_j = np.round(target_x / bs).astype(np.int64)
        elif grid == 'union':
            target_j = None
            for j in unit_j.values():
                target_j = j if target_j is None else np.union1d(target_j, j)
            target_x = target_j * bs
        elif grid == 'zero':
            j_lo = min(0, min(int(j[0]) for j in unit_j.values()))
            j_hi = max(int(j[-1]) for j in unit_j.values())
            target_j = np.arange(j_lo, j_hi + 1, dtype=np.int64)
            target_x = target_j * bs
        else:
            raise ValueError(
                f"grid must be 'total', 'union' or 'zero', not {grid!r}")
        out = pd.DataFrame(index=pd.Index(target_x, name='loss'))
        mismatched = []
        for agg in self.agg_list:
            j, p = unit_j[agg.name], unit_p[agg.name]
            inside = np.isin(j, target_j)
            if not inside.all():
                dropped = float(np.abs(p[~inside]).sum())
                mismatched.append(
                    f'{agg.name} ({int(np.sum(~inside))} buckets, '
                    f'|p| {dropped:.3e} dropped)')
            col = np.zeros(len(target_j))
            col[np.searchsorted(target_j, j[inside])] = p[inside]
            out[f'p_{agg.name}'] = col
        if mismatched and not allow_window_mismatch:
            warnings.warn(
                f'{self.name}: unit window(s) extend beyond the {grid!r} '
                f'grid -- {"; ".join(mismatched)}. This view is a clipped '
                f'display artifact; pass allow_window_mismatch=True to '
                f'acknowledge.', stacklevel=2)
        return out

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
        # total first = Book standard, then each unit on its native grid
        # (numerics-1); on a legacy zero-origin book the grids coincide.
        bit = pd.concat(
            [self.density_df.p_total] +
            [self.unit_density(unit) for unit in self.unit_names], axis=1)
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

    @staticmethod
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
        (``update`` via ``add_exa`` and :func:`swap_density_df`).
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

    def add_exa(self, df, unit_state):
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
        df : pandas.DataFrame
            Frame to extend in place, carrying ``loss`` (the total output
            grid, any snapped origin — zero, positive or negative) and
            ``p_total``. ``update`` passes ``self.density_df``;
            :func:`swap_density_df` passes its own frame.
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
        bs = self.bs
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
        logger.info(f'{self.name}: sum of p_total is 1 - {1 - sum_p_total:12.8e} NOT rescaling.')
        df['F'] = np.cumsum(p_total)
        df['S'] = 1 - df.F
        F = df['F'].to_numpy()
        S = df['S'].to_numpy()
        F_small = F <= tol
        S_small = S <= tol

        logger.info(
            f'Portfolio.add_exa | {self.name}: S <= 0 values has length {len(np.argwhere((df.S <= 0).to_numpy()))}')

        # total columns by direct sums carrying the origin
        cum_x = np.cumsum(loss * p_total)
        e_total = np.sum(loss * p_total)
        df['exa_total'] = cum_x + loss * S
        df['lev_total'] = df['exa_total']
        with np.errstate(divide='ignore', invalid='ignore'):
            exlea_total = cum_x / F
            exgta_total = (e_total - cum_x) / S
        exlea_total[F_small] = np.nan
        exgta_total[S_small] = np.nan
        df['exlea_total'] = exlea_total
        df['e_total'] = e_total
        df['exgta_total'] = exgta_total
        df['exeqa_total'] = loss  # E[X | X=a] = a

        # kappa numerators: not-unit FT products plus the native
        # first-moment FTs, all in the physical-zero buffer convention
        # (transient — freed with unit_state when the caller returns).
        ft_all, ft_nots = self._ft_nots(
            {nm: st['ft_p'] for nm, st in unit_state.items()})
        m_buf = 2 * (len(ft_all) - 1)

        for col in self.unit_names:
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
            df[f'exeqa_{col}'] = kappa

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
            df[f'lev_{col}'] = np.where(
                inside, cum_xp_n[pos_c] + loss * (1 - cum_p_n[pos_c]), loss)

            # e_{col} from the native pmf (the unit's represented mean)
            e_col = float(np.sum(xs_n * p_n))
            df[f'e_{col}'] = e_col

            # conditional means by direct sums of kappa · p_total
            kp = kappa * p_total
            cum_xi = np.cumsum(kp)
            with np.errstate(divide='ignore', invalid='ignore'):
                exlea = cum_xi / F
                exgta = (e_col - cum_xi) / S
            exlea[F_small] = np.nan
            exgta[S_small] = np.nan
            df[f'exlea_{col}'] = exlea
            df[f'exgta_{col}'] = exgta

            if signed:
                # kappa/x is not a recovery share on a signed grid
                # (steering 6): blank rather than divide through zero.
                df[f'exi_x_{col}'] = np.nan
                df[f'exi_xlea_{col}'] = np.nan
                df[f'exi_xgta_{col}'] = np.nan
                df[f'exi_xeqa_{col}'] = np.nan
                df[f'exa_{col}'] = np.nan
                continue

            # share s_i(x) = kappa_i(x)/x; the origin row x=0 takes the
            # 0 convention (X=0 ⇒ X_i=0 a.s. on a non-negative book).
            with np.errstate(divide='ignore', invalid='ignore'):
                share = np.where(np.abs(loss) < bs / 2, 0.0, kappa / loss)
            sp = share * p_total
            df[f'exi_x_{col}'] = np.sum(sp)
            cum_sp = np.cumsum(sp)
            with np.errstate(divide='ignore', invalid='ignore'):
                exi_xlea = cum_sp / F
            exi_xlea[F_small] = 0.0
            df[f'exi_xlea_{col}'] = exi_xlea

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
            df[f'exi_xgta_{col}'] = alpha
            df[f'exi_xeqa_{col}'] = share

            # exa_{col} = E[X_i(a)] = Σ_{x≤a} kappa_i·p + a·tail_share(a)
            # — the direct-sum form of ∫ S·alpha dx, carrying the origin.
            df[f'exa_{col}'] = cum_xi + loss * tail_share

        # Sum-of-shares check columns.
        for metric in ['exi_xlea_', 'exi_xgta_', 'exi_xeqa_']:
            df[metric + 'sum'] = df.filter(regex=metric + '[^η]').sum(axis=1)


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
                         index=pd.Index(['calibration'], name='unit')))

        self.distortion_df = distortion_df
        self.calibration_df = calibration_df
        self.distortions = distortions
        return distortion_df

    def apply_distortion(self, distortion, *, view='ask', S_calculation='forwards',
                         allocation='lifted', allow_deficit=False):
        """
        Apply ``distortion`` and return the resulting augmented_df.

        Results are cached on ``self._augmented_dfs`` keyed by
        ``(name, view, role, S_calculation, allocation)`` so bid/ask,
        loss/payoff, forwards/backwards and linear/lifted frames coexist
        (the ``role`` slot is the canonical value-type flag
        ``_is_loss_value``, never the configurable label string). A second
        call with the same key is an O(1) dict lookup; the returned
        DataFrame is the same object (``is``-identical) as the prior call.

        Parameters
        ----------
        distortion : Distortion or str
            A ``Distortion`` instance, or the name of a previously calibrated
            distortion (looked up in ``self.distortions``).
        view : {'ask', 'bid'}
            Pricing view. Composes with the portfolio's value-type role by
            XOR to select ``g`` or ``g_dual``; see
            :meth:`~aggregate.spectral.Distortion.effective_g`.
        S_calculation : {'forwards', 'backwards'}
            Deficit-parking direction for the exact-discrete tail; see
            :func:`~aggregate.spectral.choquet_weights`. Equivalent on a
            clean (normalized) law.
        allocation : {'lifted', 'linear'}
            Tail-share choice for the per-unit ``exag_*`` columns: lifted
            uses the distorted tail share ``exi_xgtag_*`` (beta), linear
            the objective ``exi_xgta_*`` (alpha). Identical column schema;
            the total columns do not depend on the choice.
        allow_deficit : bool
            Explicit truncation policy for a materially defective total
            (``1 - sum(p_total)`` above the validation noise floor).
            Default False raises ``DefectiveDistributionError``.

        Returns
        -------
        pandas.DataFrame
            The cached ``augmented_df`` for this key.

        Notes
        -----
        The actual construction lives in ``_build_augmented``. The cache is
        invalidated whenever ``update`` is called (the underlying density
        changes).
        """
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        name = distortion.name
        key = (name, view, self._is_loss_value, S_calculation, allocation)
        if key not in self._augmented_dfs:
            self._augmented_dfs[key] = self._build_augmented(
                distortion, view=view, S_calculation=S_calculation,
                allocation=allocation, allow_deficit=allow_deficit)
        self._distortion = distortion
        self._last_applied_distortion_name = name
        return self._augmented_dfs[key]

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
        The augmented_df cache as a dict keyed by
        ``(distortion_name, view, role, S_calculation, allocation)``.

        Read-only view -- mutate via ``apply_distortion`` (insert) or
        ``update`` (clear).
        """
        return self._augmented_dfs

    def pricing_at(self, distortion, *, p=None, a=None, allocation='lifted'):
        """Pentagon pricing readout per unit at probability ``p`` or asset ``a``.

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
        allocation : {'lifted', 'linear'}
            Tail-share choice for the per-unit premium allocation; passed
            through to :meth:`apply_distortion`.

        Returns
        -------
        pandas.DataFrame
            Rows indexed by unit (units + 'total'), columns
            ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']``. Per-unit
            ``a = P + Q`` (allocated assets); on ``total`` it equals the
            requested portfolio asset level.

        Notes
        -----
        ``L = exa``, ``P = exag``, ``M = P - L``; per-unit capital ``Q`` is
        computed on demand by the layer-ROE construction
        (:meth:`_unit_capital_at` -- no persistent per-unit ``Q`` column).
        Total ``Q`` uses the exact row identity ``a - exag_total``.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'pricing_at requires exactly one of p= (probability) '
                'or a= (asset level).')
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        if a is None:
            a = self.q(p)
        else:
            a = self.snap(a)
        aug = self.apply_distortion(distortion, allocation=allocation)
        if a in aug.index:
            row = aug.loc[a]
        else:
            logger.warning(
                f'pricing_at: asset level {a} not in augmented_df.index; using last row.')
            row = aug.iloc[-1]
            a = float(row['loss'])
        units = list(self.unit_names_ex)
        out = pd.DataFrame(
            index=units,
            columns=['L', 'M', 'P', 'Q'],
            dtype=float,
        )
        out.index.name = 'unit'
        unit_q = self._unit_capital_at(aug, a, distortion)
        for unit in self.unit_names:
            out.loc[unit, 'L'] = row[f'exa_{unit}']
            out.loc[unit, 'P'] = row[f'exag_{unit}']
            out.loc[unit, 'Q'] = unit_q[unit]
        out.loc['total', 'L'] = row['exa_total']
        out.loc['total', 'P'] = row['exag_total']
        out['M'] = out.P - out.L
        # exact total Q = a - exag_total beats the layer-by-layer sum,
        # which can drift by a few buckets in the tail.
        out.loc['total', 'Q'] = a - row['exag_total']
        # fill a + ratios and stamp the canonical categorical (pentagon.py)
        out = complete_pentagon(out)
        return out

    def pentagon_at(self, distortion, *, p=None, a=None, unit='total',
                    allocation='lifted'):
        """Single-unit pentagon as a :class:`~aggregate.pentagon.Pentagon` object.

        The object-flavored analogue of :meth:`pricing_at`: returns one fully
        solved :class:`Pentagon` (an eight-vector with named attributes and
        provenance) for ``unit`` at probability ``p`` or asset level ``a``,
        rather than a DataFrame of all units. The natural entry point for the
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
        unit : str, default 'total'
            Which unit to read (``'total'`` for the portfolio total).
        allocation : {'lifted', 'linear'}
            Tail-share choice for the per-unit premium allocation; passed
            through to :meth:`apply_distortion`.

        Returns
        -------
        Pentagon
            Fully solved, with ``.distortion`` / ``.shape`` provenance attached.

        Notes
        -----
        Reads the same augmented-distortion row as :meth:`pricing_at`; the
        total ``Q`` uses the exact ``a - exag_total`` (matching ``pricing_at``),
        per-unit ``Q`` the on-demand layer-ROE construction
        (:meth:`_unit_capital_at`), so the two agree.
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
        aug = self.apply_distortion(distortion, allocation=allocation)
        if a in aug.index:
            row = aug.loc[a]
        else:
            row = aug.iloc[-1]
            a = float(row['loss'])
        peg = Pentagon(obj=self)
        L = row[f'exa_{unit}']
        P = row[f'exag_{unit}']
        if unit == 'total':
            # exact total Q, matching pricing_at
            Q = a - row['exag_total']
        else:
            Q = self._unit_capital_at(aug, a, distortion, units=[unit])[unit]
        # L, P, Q are the three independent amounts; M = P - L, a = P + Q follow.
        peg.solve(L=L, P=P, Q=Q)
        peg.distortion = distortion
        peg.shape = getattr(distortion, 'shape', None)
        return peg

    def _build_augmented(self, dist, *, view='ask', S_calculation='forwards',
                         allocation='lifted', allow_deficit=False):
        r"""Construct an augmented_df from ``self.density_df`` under ``dist``.

        Pure builder: returns the frame without touching ``self`` (the
        ``apply_distortion`` wrapper writes it into the cache). One
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
        mass_unbounded = getattr(dist, 'has_mass', False) and not self.bounded
        if allocation == 'lifted' and mass_unbounded:
            raise ValueError(
                f"lifted allocation on an unbounded portfolio with a mass "
                f"distortion ({dist.name}) is unstable on the right edge "
                f"(essentially all the distortion weight lands on the last "
                f"bucket). Use allocation='linear' or certify "
                f"`portfolio.bounded = True` if the support is in fact bounded.")

        df = self.density_df.copy()
        loss = df.loss.to_numpy()
        p_total = df.p_total.to_numpy()
        signed = float(loss[0]) < 0

        g, g_prime, _ = dist.effective_g(view, is_loss_value=self._is_loss_value)
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
            for unit in self.unit_names:
                df[f'exi_xgtag_{unit}'] = np.nan
                df[f'exag_{unit}'] = np.nan
            return df

        # Truncate where the exeqa decomposition breaks down (FFT noise;
        # discrete "gaps" in p_total are ignored — error is only
        # meaningful on support). Positional indexing: no zero-origin
        # assumption. Truncate BEFORE the per-unit sweep: beyond the cut
        # the shares are FFT junk, so the per-unit tail sums close with a
        # collapsed atom at the cut (below) rather than integrating noise.
        lnp = '|'.join(self.unit_names)
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
        for unit in self.unit_names:
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

    def _unit_capital_at(self, aug, a, dist, *, view='ask', units=None):
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
        aug : pandas.DataFrame
            An augmented frame from :meth:`apply_distortion` (carries
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
            units = list(self.unit_names)
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
                                         is_loss_value=self._is_loss_value)
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
        if (len(units) == len(self.unit_names) and pos
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

    def allocation_diagnostics(self, distortion, *, surface='lifted',
                               view='ask', S_calculation='forwards'):
        r"""Layer-curve diagnostic frame for a distorted portfolio.

        The explicit consumer surface for :func:`pedagogy.plot_twelve`
        and similar exhibits (pedagogy consumes, never dictates --
        steering 7): the core pricing frame no longer carries layer
        diagnostic columns. Sourced from the
        :meth:`apply_distortion` frame for ``surface``.

        Parameters
        ----------
        distortion : Distortion or str
            Passed through to :meth:`apply_distortion`.
        surface : {'lifted', 'linear'}
            Which allocation surface to diagnose.
        view, S_calculation
            Passed through to :meth:`apply_distortion`.

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
            distortion = self.distortions[distortion]
        aug = self.apply_distortion(distortion, view=view,
                                    S_calculation=S_calculation,
                                    allocation=surface)
        cols = ['loss', 'F', 'S', 'gS', 'gF', 'gp_total']
        cols += [c for unit in self.unit_names_ex
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
            view, is_loss_value=self._is_loss_value)
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
            for unit in self.unit_names:
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

    def var_dict(self, p, kind='lower', total='total', snap=False):
        """
        make a dictionary of value at risks for each unit and the whole portfolio.

         Returns: {unit : var(p, kind)} and includes the total as self.name unit

        Example:

            for p, arg in zip([.996, .996, .996, .985], ['var', 'lower', 'upper', 'tvar']):
                print(port.var_dict(p, arg,  snap=True))

        :param p:
        :param kind: var (defaults to lower), upper, lower, tvar
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

    def price(self, p, distortion=None, *, allocation=None, view='ask'):
        """Price the total under a distortion and allocate to units.

        ``rho(X ∧ q(p))`` for a single distortion (or a dict / list of
        distortions). ``p`` is a probability if ``p ≤ 1`` (converted to
        assets via VaR and snapped to the index) and an asset level
        otherwise. ``allocation`` defaults to :attr:`allocation_method`
        (``'linear'`` out of the box); pass ``'lifted'`` to override.

        Both methods read rows of the same unified augmented frame
        (:meth:`apply_distortion`); the only difference is the tail
        share allocating the collapsed default states -- objective
        ``alpha`` (linear) vs distorted ``beta`` (lifted). Lifted is
        unstable on the right edge for distortions with a mass on an
        unbounded support; the builder **refuses** that combination and
        points at ``'linear'``.

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

        Returns
        -------
        PricingResult
            Per-unit ``df`` (pentagon columns), the total price scalar,
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

        # figure regulatory assets; applied to unlimited losses
        if p > 1:
            a_reg = self.snap(p)
            reg_p = self.cdf(a_reg)
        else:
            a_reg = self.q(p)
            reg_p = p

        dfs = {}
        price = {}
        last_price = 0
        for k, v in distortion.items():
            logger.info(f'Executing for {k}, {allocation}')
            aug_df = self.apply_distortion(v, view=view, allocation=allocation)
            if a_reg in aug_df.index:
                aug_row = aug_df.loc[a_reg]
                a_eff = a_reg
            else:
                logger.warning('Regulatory assets not in augmented_df. Using last.')
                aug_row = aug_df.iloc[-1]
                a_eff = float(aug_row['loss'])

            df = pd.DataFrame(
                index=pd.Index(list(self.unit_names_ex), name='unit'),
                columns=['L', 'M', 'P', 'Q'],
                dtype=float,
            )
            unit_q = self._unit_capital_at(aug_df, a_eff, v, view=view)
            for unit in self.unit_names:
                df.loc[unit, 'L'] = aug_row[f'exa_{unit}']
                df.loc[unit, 'P'] = aug_row[f'exag_{unit}']
                df.loc[unit, 'Q'] = unit_q[unit]
            df.loc['total', 'L'] = aug_row['exa_total']
            df.loc['total', 'P'] = aug_row['exag_total']
            df['M'] = df.P - df.L
            # exact total capital, immune to layer-sum drift
            df.loc['total', 'Q'] = a_eff - aug_row['exag_total']
            df = complete_pentagon(df)
            price[k] = last_price = df.loc['total', 'P']
            dfs[k] = df.sort_index()

        df = pd.concat(dfs.values(), keys=dfs.keys(), names=['distortion', 'unit'])
        return PricingResult(df, last_price, price, a_reg, reg_p)

    def price_stand_alone(self, dist, p):
        """
        Price each unit on a stand-alone basis and compare to the diversified whole.

        Every unit is priced *as if it were the only unit in the book*: its
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
        sop.index.name = 'unit'
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
        return pent.as_frame(unit='total')

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
            Holds the per-unit pricing DataFrame (from :meth:`pricing_at`)
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
            index=pd.Index(['total'], name='unit'),
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
            ``(distortion, stat)`` on rows and unit names on columns;
            ``stat`` runs over ``['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE', 'a']``.
            ``augmented_dfs`` is a snapshot of the cache for the analysed
            distortions.

        Notes
        -----
        Replaces both the legacy ``analyze_distortions(a=0, p=0, ...)`` and
        ``analyze_distortions2(p, dists=None)``. The output shape matches the
        legacy ``analyze_distortions2``: rows are ``(distortion, stat)``,
        columns are unit names.

        A mass distortion on an unbounded portfolio cannot build the
        lifted frame (the mass lands on the last represented bucket); such
        members of the sweep are skipped with a ``UserWarning`` -- price
        them explicitly with ``price(..., allocation='linear')``.
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
            # a mass distortion on an unbounded support cannot build the
            # lifted frame (numerics-3 G6); skip it from the sweep with a
            # visible warning rather than failing the whole exhibit.
            if getattr(d, 'has_mass', False) and not self.bounded:
                warnings.warn(
                    f'analyze_distortions: skipping {name} -- mass '
                    f'distortion on an unbounded portfolio (lifted frame '
                    f'refused). Price it explicitly with '
                    f"allocation='linear'.")
                continue
            # pricing_at returns units × canonical pentagon columns; transpose
            # so stats are rows and units are columns. The transpose drops the
            # categorical column dtype, so work in plain string labels here and
            # reapply the canonical stat order/dtype after concat.
            exhibit = self.pricing_at(d, a=a_cal).T
            exhibit.index = exhibit.index.astype(str)
            # 'a' row: P + Q per unit, rescaled so totals sum to a_cal.
            a_row = exhibit.loc['P'] + exhibit.loc['Q']
            a_row = a_row * a_cal / a_row['total']
            exhibit.loc['a'] = a_row
            # canonical stat order (pentagon.py), trailing octet semantics
            per_dist[name] = exhibit.reindex(PENTAGON_STATS)
        if not per_dist:
            raise ValueError(
                'analyze_distortions: nothing to price -- every requested '
                'distortion is a mass distortion on an unbounded portfolio.')
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
        # snapshot only the distortions analysed (default-key frames; the
        # cache key is (name, view, role, S_calculation, allocation))
        augmented_dfs = {
            n: frame for (n_, view_, role_, sc_, alloc_), frame
            in self._augmented_dfs.items()
            for n in distortions
            if n_ == n and view_ == 'ask' and sc_ == 'forwards'
            and alloc_ == 'lifted'
        }
        return AnalyzeDistortionsResult(
            distortions=dict(distortions),
            pricing_df=pricing_df,
            augmented_dfs=augmented_dfs,
        )


    @property
    def unit_renamer(self):
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
            # numbered units
            ln = re.sub('([A-Z])m([0-9]+)', r'$\1_{-\2}$', ln)
            ln = re.sub('([A-Z])([0-9]+)', r'$\1_{\2}$', ln)
            return ln

        if self._unit_renamer is None:
            self._unit_renamer = { ln: rename(ln) for ln in self.unit_names_ex}

        return self._unit_renamer

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
        for c in self.unit_names:
            # native unit pmf via the accessor (the p_{unit} columns left
            # density_df at numerics-2); same draw mechanics as before.
            pc = f'p_{c}'
            bit = self.unit_density(c).reset_index()
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

    @property
    def n_units(self):
        return len(self.unit_names)

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

    port.add_exa(port.density_df, unit_state)
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
        awk.aligned_unit_density_df().cumsum().plot()
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


