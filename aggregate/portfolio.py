from collections import namedtuple
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
from numpy.random import PCG64
import pandas as pd
from pandas.io.formats.format import EngFormatter
from pandas.plotting import scatter_matrix
from pathlib import Path
import re
import scipy.stats as ss
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from textwrap import fill
from IPython.core.display import HTML, display

from .constants import *
from .distributions import Aggregate, Severity
from .spectral import Distortion
from .utilities import ft, \
    ift, sln_fit, sgamma_fit, \
    axiter_factory, AxisManager, html_title, \
    suptitle_and_tight, pprint_ex, \
    MomentAggregator, Answer, subsets, round_bucket, \
    make_mosaic_figure, iman_conover, approximate_work

# fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# matplotlib.rcParams['legend.fontsize'] = 'xx-small'
logger = logging.getLogger(__name__)


class Portfolio(object):
    """
    Portfolio creates and manages a portfolio of Aggregate objects each modeling one
    unit of business. Applications include

    - Model a book of insurance
    - Model a large account with several sub lines
    - Model a reinsurance portfolio or large treaty

    """

    # namer helper classes
    premium_capital_renamer = {
        'Assets': "0. Assets",
        'T.A': '1. Allocated assets',
        'T.P': '2. Market value liability',
        'T.L': '3. Expected incurred loss',
        'T.M': '4. Margin',
        'T.LR': '5. Loss ratio',
        'T.Q': '6. Allocated equity',
        'T.ROE': '7. Cost of allocated equity',
        'T.PQ': '8. Premium to surplus ratio',
        'EPD': '9. Expected pol holder deficit'
    }

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
        logger.debug(f'Portfolio.__init__| creating new Portfolio {self.name}')
        # logger.debug(f'Portfolio.__init__| creating new Portfolio {self.name} at {super(Portfolio, self).__repr__()}')
        ma = MomentAggregator()
        max_limit = 0
        if (len(spec_list) == 1 and isinstance(spec_list[0], pd.DataFrame)):
            # create from samples...slightly different looping behavior
            logger.info('Creating from sample DataFrame')
            spec_list = spec_list[0]
        if isinstance(spec_list, pd.DataFrame):
            if 'p_total' not in spec_list:
                logger.info('Adding p_total column to DataFrame with equal probs')
                spec_list = spec_list.copy()
                spec_list['p_total'] = np.repeat(1 / len(spec_list), len(spec_list))

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
                # note here you could do uw.aggregate[spec] and get the dictionary def
                # or uw.write(spec) to return the already-created (and maybe updated) object
                # we go the latter route...if user wants they can pull off the dict item themselves
                if uw is None:
                    raise ValueError(f'Must pass valid Underwriter instance to create aggs by name')
                try:
                    a_out = uw.write(spec)
                except Exception as e:
                    logger.error(f'Item {spec} not found in your underwriter')
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
                ma.add_fs(a.report_ser[('freq', 'ex1')], a.report_ser[('freq', 'ex2')], a.report_ser[('freq', 'ex3')],
                          a.report_ser[('sev', 'ex1')], a.report_ser[('sev', 'ex2')], a.report_ser[('sev', 'ex3')])
                max_limit = max(max_limit, np.max(np.array(a.limit)))

        self.line_names_ex = self.line_names + ['total']
        self.line_name_pipe = "|".join(self.line_names_ex)
        for n in self.line_names:
            # line names cannot equal total
            if n == 'total':
                raise ValueError('Line names cannot equal total, it is reserved for...total')
        # make a pandas data frame of all the statistics_df
        temp_report = pd.concat([a.report_ser for a in self.agg_list], axis=1)

        # max_limit = np.inf # np.max([np.max(a.get('limit', np.inf)) for a in spec_list])
        temp = pd.DataFrame(ma.stats_series('total', max_limit, 0.999, remix=False))
        self.statistics_df = pd.concat([temp_report, temp], axis=1)
        # future storage
        self.density_df = None
        self.independent_density_df = None
        self.augmented_df = None
        self._epd_2_assets = None
        self._assets_2_epd = None
        self._priority_capital_df = None
        self.priority_analysis_df = None
        self.audit_df = None
        self.independent_audit_df = None
        self.padding = 0
        self.tilt_amount = 0
        self._linear_quantile_function = None
        self._cdf = None
        self._pdf = None
        self._tail_var = None
        self._tail_var2 = None
        self._inverse_tail_var = None
        self.bs = 0
        self.log2 = 0
        self.ex = 0
        self.last_update = 0
        self.hash_rep_at_last_update = ''
        self._distortion = None
        self.sev_calc = ''
        self._remove_fuzz = 0
        self.approx_type = ""
        self.approx_freq_ge = 0
        self.discretization_calc = ''
        self.normalize = None
        # for storing the info about the quantile function
        self.q_temp = None
        self._renamer = None
        self._line_renamer = None
        self._tm_renamer = None
        # if created by uw it stores the program here
        self.program = ''
        self.audit_percentiles = [.9, .95, .99, .996, .999, .9999, 1 - 1e-6]
        self.dists = None
        self.dist_ans = None
        self.figure = None

        # for consistency with Aggregates
        self.agg_m = self.statistics_df.loc[('agg', 'ex1'), 'total']
        self.agg_cv = self.statistics_df.loc[('agg', 'cv'), 'total']
        self.agg_skew = self.statistics_df.loc[('agg', 'skew'), 'total']
        # variance and sd come up in exam questions
        self.agg_sd = self.agg_m * self.agg_cv
        self.agg_var = self.agg_sd * self.agg_sd
        # these are set when the object is updated
        self.est_m = self.est_cv = self.est_skew = self.est_sd = self.est_var = 0

        # enhanced portfolio items
        self.EX_premium_capital = None
        self.last_a = None
        self.EX_multi_premium_capital = None
        self.EX_accounting_economic_balance_sheet = None
        self.validation_eps = VALIDATION_EPS

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
        """
        logger.info(f'Creating Porfolio {name} from sample_df')
        port = Portfolio(name, sample_df)
        logger.info(f'Updating with bs={bs}, log2={log2}, remove_fuzz=True')
        port.update(bs=bs, log2=log2, remove_fuzz=True, **kwargs)
        # archive the original density_df
        port.independent_density_df = port.density_df.copy()
        # execute switeroo
        logger.info('Creating exa_sample and executing switcheroo')
        port.density_df = port.add_exa_sample(sample_df)
        # update total stats
        logger.info('Updating total statistics (WARNING: these are now emprical)')
        port.independent_audit_df = port.audit_df.copy()
        # just update the total stats
        port.make_audit_df(['total'], None)
        # return new created object
        logger.info('Returning new Portfolio object')
        return port

    def sample_compare(self, ax=None):
        """
        Compare the sample sum to the independent sum of the marginals.

        """
        if self.independent_density_df is None:
            raise ValueError('No independent_density_df, cannot compare')

        if ax is not None:
            ax.plot(self.independent_density_df.index, self.independent_density_df['S'], lw=1, label='independent')
            ax.plot(self.density_df.index, self.density_df['S'], lw=1, label='sample')
            ax.legend()

        df = pd.concat((self.independent_audit_df.T, self.audit_df.T),
            keys=['independent', 'sample'], axis=1).xs('total', 1,1, drop_level=False)
        return df

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

    def pricing_bounds(self, premium, a=0, p=0, n_tps=64, kind='tail', slow=False, verbose=250):
        """
        Compute the natural allocation premium ranges by unit consistent with
        total premium at asset level a or p (one of which must be provided).

        Unlike typical case with even s values, this is run at the actual S
        values of the Portfolio.

        Visualize::

            from pandas.plotting import scatter_matrix
            ans = port.pricing_bounds(premium, p=0.98)
            scatter_matrix(ans.allocs, marker='.', s=5, alpha=1,
                           figsize=(10, 10), diagonal='kde' )
        """
        from .bounds import Bounds
        if a == 0:
            assert p > 0, 'Must provide either a or p'
            a = self.q(p)

        # need a -= self.bs?
        S = self.density_df.loc[:a, 'S'].copy()
        # last entry needs to include all remaining losses from a-bs onwards, hence:
        S.iloc[-1] = 0.
        bounds = Bounds(self)
        bounds.tvar_cloud('total', premium, a, n_tps + 1, S.values, kind=kind)

        # TODO: (hack) pl=1 and s=1 is driving an error NAN - need to replace with 0. But WHY?
        gS = bounds.cloud_df.fillna(0).values.T
        gps = -np.diff(gS, axis=1, prepend=1)
        # sum products for allocations
        deal_losses = self.density_df.filter(regex='exeqa_[A-Z]').loc[:a]
        if self.sf(a) > 0:
            # see notes below in slow method
            logger.info('Adjusting tail of deal_losses')
            deal_losses.iloc[-1] = self.density_df.loc[a-self.bs].filter(regex='exi_xgta_[A-Z]') * a

        # compute the allocations
        allocs = pd.DataFrame(
            gps @ (deal_losses.to_numpy() * self.bs),
            columns=[i.replace('exeqa_', 'alloc_') for i in deal_losses.columns],
            index=bounds.weight_df.index)

        # this is a good audit: should have max = min
        allocs['total'] = allocs.sum(1)
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
        ans = namedtuple('pricing_bounds', 'bounds allocs stats comp allocs_slow p_star')
        p_star = bounds.p_star('total', premium, a)
        return ans(bounds, allocs, stats, comp, allocs_slow, p_star)

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
                df.select_dtypes(include=['float64']).applymap(lambda x: 0 if abs(x) < eps else x)

    def __repr__(self):
        """
        Goal unmbiguous
        :return:
        """
        # return str(self.to_dict())
        # this messes up when port = self has been enhanced...

        # cannot use ex, etc. because object may not have been updated
        if self.audit_df is None:
            ex = self.statistics_df.loc[('agg', 'mean'), 'total']
            empex = np.nan
            isupdated = False
        else:
            ex = self.get_stat(stat="Mean")
            empex = self.get_stat()
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
            s.append(f'approx_type              {self.approx_type}')
            s.append(f'approx_freq_ge           {self.approx_freq_ge}')
            s.append(f'distortion               {repr(self._distortion)}')

        if isupdated:
            s.append('')
            with pd.option_context('display.width', 140, 'display.float_format', lambda x: f'{x:,.5g}'):
                # get it on one row
                s.append(str(self.describe))
        # s.append(super(Portfolio, self).__repr__())
        return '\n'.join(s)

    def _repr_html_(self):
        """
        Updated to mimic Aggregate
        """
        s = [f'<h3>Portfolio object: {self.name}</h3>']
        _n = len(self.agg_list)
        _s = "" if _n <= 1 else "s"
        s.append(f'Portfolio contains {_n} aggregate component{_s}.')
        if self.bs > 0:
            s.append(f'Updated with bucket size {self.bs:.6g} and log2 = {self.log2}.')
        df = self.describe
        return '\n'.join(s) + df.fillna('').to_html()

    def __str__(self):
        """ Default behavior """
        return repr(self)

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
    def statistics(self):
        """
        Same as statistics df, to be consistent with Aggregate objects
        :return:
        """
        return self.statistics_df

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
        """
        Theoretic and empirical stats. Used in _repr_html_.
        Leverage Aggregate object stats; same format

        """
        sev_m = self.statistics_df.loc[('sev', 'ex1'), 'total']
        sev_cv = self.statistics_df.loc[('sev', 'cv'), 'total']
        sev_skew = self.statistics_df.loc[('sev', 'skew'), 'total']
        n_m = self.statistics_df.loc[('freq', 'ex1'), 'total']
        n_cv = self.statistics_df.loc[('freq', 'cv'), 'total']
        n_skew = self.statistics_df.loc[('freq', 'skew'), 'total']
        a_m = self.statistics_df.loc[('agg', 'ex1'), 'total']
        a_cv = self.statistics_df.loc[('agg', 'cv'), 'total']
        a_skew = self.statistics_df.loc[('agg', 'skew'), 'total']
        df = pd.DataFrame({'E[X]': [n_m, sev_m, a_m], 'CV(X)': [n_cv, sev_cv, a_cv],
                           'Skew(X)': [n_skew, sev_skew, a_skew]},
                          index=['Freq', 'Sev', 'Agg'])
        df.index.name = 'X'

        if self.audit_df is not None:
            df.loc['Sev', 'Est E[X]'] = np.nan
            df.loc['Agg', 'Est E[X]'] = self.est_m
            df['Err E[X]'] = df['Est E[X]'] / df['E[X]'] - 1
            df.loc['Sev', 'Est CV(X)'] = np.nan
            df.loc['Agg', 'Est CV(X)'] = self.est_cv
            df['Err CV(X)'] = df['Est CV(X)'] / df['CV(X)'] - 1
            df.loc['Sev', 'Est Skew(X)'] = np.nan
            df.loc['Agg', 'Est Skew(X)'] = self.est_skew
            df = df[['E[X]', 'Est E[X]', 'Err E[X]', 'CV(X)', 'Est CV(X)', 'Err CV(X)', 'Skew(X)',
                     'Est Skew(X)']]

        t1 = [a.describe for a in self] + [df]
        t2 = [a.name for a in self] + ['total']
        df = pd.concat(t1, keys=t2, names=['unit', 'X'])
        if self.audit_df is not None:
            # add estimated severity
            sev_est_m = (df.xs('Sev', axis=0, level=1)['Est E[X]'].iloc[:-1].astype(float) *
                df.xs('Freq', axis=0, level=1)['E[X]'].iloc[:-1]).sum() / df.loc[('total', 'Freq'), 'E[X]']
            df.loc[('total', 'Sev'), 'Err E[X]'] = sev_est_m / df.loc[('total', 'Sev'), 'E[X]'] - 1
            df.loc[('total', 'Sev'), 'Est E[X]'] = sev_est_m
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
        args["tilt_amount"] = self.tilt_amount
        args["distortion"] = repr(self._distortion)
        args["sev_calc"] = self.sev_calc
        args["remove_fuzz"] = self._remove_fuzz
        args["approx_type"] = self.approx_type
        args["approx_freq_ge"] = self.approx_freq_ge
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

    def audits(self, kind='all', **kwargs):
        """
        produce audit plots to assess accuracy of outputs.

        Currently only exeqa available

        :param kind:
        :param kwargs: passed to pandas plot, e.g. set xlim
        :return:
        """

        if kind == 'all':
            kind = ['exeqa']

        for k in kind:
            if k == 'exeqa':
                temp = self.density_df.filter(regex='exeqa_.*(?<!total)$').copy()
                temp['sum'] = temp.sum(axis=1)
                temp['err'] = temp['sum'] - temp.index
                f, axs = plt.subplots(1, 2, figsize=(8, 3.75), constrained_layout=True)
                ax = axs.flatten()
                a = temp['err'].abs().plot(logy=True, title=f'Exeqa Sum Error', ax=ax[1], **kwargs)
                a.plot(self.density_df.loss, self.density_df.p_total, label='p_total')
                a.plot(self.density_df.loss, self.density_df.p_total * temp.err, label='prob wtd err')
                a.grid('b')
                a.legend(loc='lower left')

                if 'xlim' in kwargs:
                    kwargs['ylim'] = kwargs['xlim']
                temp.filter(regex='exeqa_.*(?<!total)$|sum').plot(title='exeqa and sum of parts', ax=ax[0],
                                                                  **kwargs).grid('b')
            f.suptitle(f'E[Xi | X=x] vs. Sum of Parts\nbs={self.bs}, log2={self.log2}, padding={self.padding}',
                       fontsize='x-large')
            return f  # for doc maker

    def get_stat(self, line='total', stat='EmpMean'):
        """
        Other analysis suggests that iloc and iat are about same speed but slower than ix

        :param line:
        :param stat:
        :return:
        """
        return self.audit_df.loc[line, stat]

    def q(self, p, kind='lower'):
        """
        return lowest quantile, appropriate for discrete bucketing.
        quantile guaranteed to be in the index
        nearest does not work because you always want to pick rounding up

        Definition 2.1 (Quantiles)
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] ≥ α} is the lower α-quantile of X
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] > α} is the upper α-quantile of X.

        We use the x-notation if the dependence on X is evident, otherwise the q-notion.
        Acerbi and Tasche (2002)

        :param p:
        :param kind: allow upper or lower quantiles
        :return:
        """
        if self._linear_quantile_function is None:
            # revised Dec 2019
            self._linear_quantile_function = {}
            self.q_temp = self.density_df[['loss', 'F']].groupby('F').agg({'loss': np.min})
            self.q_temp.loc[1, 'loss'] = self.q_temp.loss.iloc[-1]
            self.q_temp.loc[0, 'loss'] = 0
            # revised Jan 2020
            # F           loss        loss_s
            # 0.000000    0.0         0.0
            # 0.667617    0.0         4500.0
            # a value here is  V   and ^ which is the same: correct
            # 0.815977    4500.0      5500.0
            # 0.937361	  5500.0   	  9000.0
            # upper and lower only differ at exact values of F where lower is loss and upper is loss_s
            # in between must take the next value for lower and the previous value for next to get the same answer
            self.q_temp = self.q_temp.sort_index()
            # that q_temp left cts, want right continuous:
            self.q_temp['loss_s'] = self.q_temp.loss.shift(-1)
            self.q_temp.iloc[-1, 1] = self.q_temp.iloc[-1, 0]
            # create interp functions
            # old
            # self._linear_quantile_function['upper'] = \
            #     interpolate.interp1d(self.q_temp.index, self.q_temp.loss_s, kind='previous', bounds_error=False,
            #                          fill_value='extrapolate')
            # self._linear_quantile_function['lower'] = \
            #     interpolate.interp1d(self.q_temp.index, self.q_temp.loss, kind='previous', bounds_error=False,
            #                          fill_value='extrapolate')
            # revised
            self._linear_quantile_function['upper'] = \
                interpolate.interp1d(self.q_temp.index, self.q_temp.loss_s, kind='previous', bounds_error=False,
                                     fill_value='extrapolate')
            self._linear_quantile_function['lower'] = \
                interpolate.interp1d(self.q_temp.index, self.q_temp.loss, kind='next', bounds_error=False,
                                     fill_value='extrapolate')
            # change to using loss_s
            self._linear_quantile_function['middle'] = \
                interpolate.interp1d(self.q_temp.index, self.q_temp.loss_s, kind='linear', bounds_error=False,
                                     fill_value='extrapolate')
        l = float(self._linear_quantile_function[kind](p))
        # because we are not interpolating the returned value must (should) be in the index...
        assert kind == 'middle' or l in self.density_df.index
        return l

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

    def tvar(self, p, kind='interp'):
        """
        Compute the tail value at risk at threshold p

        Really this function returns ES

        Definition 2.6 (Tail mean and Expected Shortfall)
        Assume E[X−] < ∞. Then
        x¯(α) = TM_α(X) = α^{−1}E[X 1{X≤x(α)}] + x(α) (α − P[X ≤ x(α)])
        is α-tail mean at level α the of X.
        Acerbi and Tasche (2002)

        We are interested in the right hand exceedence [?? note > vs ≥]
        α^{−1}E[X 1{X > x(α)}] + x(α) (P[X ≤ x(α)] − α)

        McNeil etc. p66-70 - this follows from def of ES as an integral
        of the quantile function


        :param p:
        :param kind:  'interp' = interpolate exgta_total;  'tail' tail integral, 'body' NYI - (ex - body integral)/(1-p)+v
            'inverse' from capital to p using interp method
        :return:
        """
        assert self.density_df is not None

        if kind == 'tail':
            # original
            # _var = self.q(p)
            # ex = self.density_df.loc[_var + self.bs:, ['p_total', 'loss']].product(axis=1).sum()
            # pip = (self.density_df.loc[_var, 'F'] - p) * _var
            # t_var = 1 / (1 - p) * (ex + pip)
            # return t_var
            # revised
            if self._tail_var2 is None:
                self._tail_var2 = self.density_df[['p_total', 'loss']].product(axis=1).iloc[::-1].cumsum().iloc[::-1]
            _var = self.q(p)
            if p >= 1.:
                return _var
            ex = self._tail_var2.loc[_var + self.bs]
            pip = (self.density_df.loc[_var, 'F'] - p) * _var
            t_var = 1 / (1 - p) * (ex + pip)
            return t_var
        elif kind == 'interp':
            # original implementation interpolated
            if self._tail_var is None:
                # make tvar function
                sup = (self.density_df.p_total[::-1] > 0).idxmax()
                if sup == self.density_df.index[-1]:
                    sup = np.inf
                    _x = self.density_df.F
                    _y = self.density_df.exgta_total
                else:
                    _x = self.density_df.F.values[:self.density_df.index.get_loc(sup)]
                    _y = self.density_df.exgta_total.values[:self.density_df.index.get_loc(sup)]
                p0 = self.density_df.at[0., 'F']
                if p0 > 0:
                    ps = np.linspace(0, p0, 200, endpoint=False)
                    tempx = np.hstack((ps, _x))
                    tempy = np.hstack((self.ex / (1-ps), _y))
                    self._tail_var = interpolate.interp1d(tempx, tempy,
                                  kind='linear', bounds_error=False,
                                  fill_value=(self.ex, sup))
                else:
                    self._tail_var = interpolate.interp1d(_x, _y, kind='linear', bounds_error=False,
                                                          fill_value=(self.ex, sup))
            if isinstance(p, (float, np.float64)):
                return float(self._tail_var(p))
            else:
                return self._tail_var(p)
        elif kind == 'inverse':
            if self._inverse_tail_var is None:
                # make tvar function
                self._inverse_tail_var = interpolate.interp1d(self.density_df.exgta_total, self.density_df.F,
                                                      kind='linear', bounds_error=False,
                                                      fill_value='extrapolate')
            if isinstance(p, (int, np.int32, np.int64, float, np.float64)):
                return float(self._inverse_tail_var(p))
            else:
                return self._inverse_tail_var(p)
        else:
            raise ValueError(f'Inadmissible kind passed to tvar; options are interp (default), inverse, or tail')

    def tvar_threshold(self, p, kind):
        """
        Find the value pt such that TVaR(pt) = VaR(p) using numerical Newton Raphson
        """
        a = self.q(p, kind)

        def f(p):
            return self.tvar(p) - a

        loop = 0
        p1 = 1 - 2 * (1 - p)
        fp1 = f(p1)
        delta = 1e-5
        while abs(fp1) > 1e-6 and loop < 10:
            df1 = (f(p1 + delta) - fp1) / delta
            p1 = p1 - fp1 / df1
            fp1 = f(p1)
            loop += 1
        if loop == 10:
            raise ValueError(f'Difficulty computing TVaR to match VaR at p={p}')
        return p1

    def equal_risk_var_tvar(self, p_v, p_t):
        """
        solve for equal risk var and tvar: find pv and pt such that sum of
        individual line VaR/TVaR at pv/pt equals the VaR(p) or TVaR(p_t)

        these won't return elements in the index because you have to interpolate
        hence using kind=middle
        """
        # these two should obviously be the same
        target_v = self.q(p_v, 'middle')
        target_t = self.tvar(p_t)

        def fv(p):
            return sum([float(a.q(p, 'middle')) for a in self]) - target_v

        def ft(p):
            return sum([float(a.tvar(p)) for a in self]) - target_t

        ans = np.zeros(2)
        for i, f in enumerate([fv, ft]):
            p1 = 1 - 2 * (1 - (p_v if i == 0 else p_t))
            fp1 = f(p1)
            loop = 0
            delta = 1e-5
            while abs(fp1) > 1e-6 and loop < 10:
                dfp1 = (f(p1 + delta) - fp1) / delta
                p1 = p1 - fp1 / dfp1
                fp1 = f(p1)
                loop += 1
            if loop == 100:
                raise ValueError(f'Trouble finding equal risk {"TVaR" if i else "VaR"} at p_v={p_v}, p_t={p_t}. '
                                 'No convergence after 100 iterations. ')
            ans[i] = p1
        return ans

    def equal_risk_epd(self, a):
        """
        determine the common epd threshold so sum sa equals a
        """
        def f(p):
            return sum([self.epd_2_assets[(l, 0)](p) for l in self.line_names]) - a
        p1 = self.assets_2_epd[('total', 0)](a)
        fp1 = f(p1)
        loop = 0
        delta = 1e-5
        while abs(fp1) > 1e-6 and loop < 10:
            dfp1 = (f(p1 + delta) - fp1) / delta
            p1 = p1 - fp1 / dfp1
            fp1 = f(p1)
            loop += 1
        if loop == 100:
            raise ValueError(f'Trouble finding equal risk EPD at pe={p1}. No convergence after 100 iterations. ')
        return p1

    def merton_perold(self, p, kind='lower'):
        """
        Compute Merton-Perold capital allocation at VaR(p) capital using VaR as risk measure.

        TODO TVaR version of Merton Perold

        """
        # figure total assets
        a = self.q(p, kind)
        # shorthand abbreviation
        df = self.density_df
        loss = df.loss
        ans = []
        total = 0
        for l in self.line_names:
            q = self.density_df.loss.iloc[np.searchsorted(self.density_df[f'ημ_{l}'].cumsum(), .995, side='right')]
            diff = a - q
            ans.append(diff)
            total += diff
        ans.append(total)
        return ans

    def cotvar(self, p):
        """
        Compute the p co-tvar asset allocation using ISA.
        Asset alloc = exgta = tail expected value, treating TVaR like a pricing variable.

        """
        av = self.q(p)
        return self.density_df.loc[av, [f'exgta_{l}' for l in self.line_names_ex]].values

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

        if self.audit_df is None:
            # not updated
            m = self.statistics_df.loc[('agg', 'mean'), 'total']
            cv = self.statistics_df.loc[('agg', 'cv'), 'total']
            skew = self.statistics_df.loc[('agg', 'skew'), 'total']
        else:
            # use statistics_df matched to computed aggregate_project
            m, cv, skew = self.audit_df.loc['total', ['EmpMean', 'EmpCV', 'EmpSkew']]

        name = f'{approx_type[0:4]}.{self.name[0:5]}'
        agg_str = f'agg {name} 1 claim sev '
        note = f'frozen version of {self.name}'
        return approximate_work(m, cv, skew, name, agg_str, note, approx_type, output)

    fit = approximate

    def collapse(self, approx_type='slognorm'):
        """
        Returns new Portfolio with the fit

        Deprecated...prefer uw.write(self.fit()) to go through the agg language approach.

        :param approx_type: slognorm | sgamma
        :return:
        """
        spec = self.fit(approx_type, output='dict')
        logger.debug(f'Portfolio.collapse | Collapse created new Portfolio with spec {spec}')
        logger.warning(f'Portfolio.collapse | Collapse is deprecated; use fit() instead.')
        return Portfolio(f'Collapsed {self.name}', [spec])

    def percentiles(self, pvalues=None):
        """
        report_ser on percentiles and large losses.
        Uses interpolation, audit_df uses nearest.

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

    def best_bucket(self, log2=16, recommend_p=0.999):
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

    def update(self, log2, bs, approx_freq_ge=100, approx_type='slognorm', remove_fuzz=False,
               sev_calc='discrete', discretization_calc='survival', normalize=True, padding=1, tilt_amount=0,
               trim_df=False, add_exa=True, force_severity=True, recommend_p=0.999, approximation=None,
               debug=False):
        """

        TODO: currently debug doesn't do anything...

        Create density_df, performs convolution. optionally adds additional information if ``add_exa=True``
        for allocation and priority analysis

        tilting: [@Grubel1999]: Computation of Compound Distributions I: Aliasing Errors and Exponential Tilting
        (ASTIN 1999)
        tilt x numbuck < 20 is recommended log. 210
        num buckets and max loss from bucket size

        Aggregate reinsurance in parser has replaced the aggregate_cession_function (a function of a Portfolio object
        that adjusts individual line densities; applied after line aggs created but before creating not-lines;
        actual statistics do not reflect impact.) Agg re by unit is now applied in the Aggregate object.

        TODO: consider aggregate covers at the portfolio level...Where in parse - at the top!


        :param log2:
        :param bs: bucket size
        :param approx_freq_ge: use method of moments if frequency is larger than ``approx_freq_ge``
        :param approx_type: type of method of moments approx to use (slognorm or sgamma)
        :param remove_fuzz: remove machine noise elements from FFT
        :param sev_calc: how to calculate the severity, discrete (point masses as xs) or continuous (uniform between xs points)
        :param discretization_calc:  survival or distribution (accurate on right or left tails)
        :param normalize: if true, normalize the severity so sum probs = 1. This is generally what you want; but
        :param padding: for fft 1 = double, 2 = quadruple
        :param tilt_amount: for tiling methodology - see notes on density for suggested parameters
        :param epds: epd points for priority analysis; if None-> sensible defaults
        :param trim_df: remove unnecessary columns from density_df before returning
        :param add_exa: run add_exa to append additional allocation information needed for pricing; if add_exa also add
            epd info
        :param force_severity: force computation of severities for aggregate components even when approximating
        :param recommend_p: percentile to use for bucket recommendation.
        :param approximation: if not None, use these instructions ('exact')
        :param debug: if True, print debug information
        :return:
        """

        if approximation is not None:
            if approximation == 'exact':
                approx_freq_ge = 1e9

        if log2 <= 0:
            raise ValueError('log2 must be >= 0')
        self.log2 = log2
        if bs == 0:
            self.bs = self.best_bucket(log2, recommend_p)
            logger.info(f'bs=0 enterered, setting bs={bs:.6g} using self.best_bucket rounded to binary fraction.')
        else:
            self.bs = bs
        self.padding = padding
        self.tilt_amount = tilt_amount
        self.approx_type = approx_type
        self.sev_calc = sev_calc
        self._remove_fuzz = remove_fuzz
        self.approx_type = approx_type
        self.approx_freq_ge = approx_freq_ge
        self.discretization_calc = discretization_calc
        self.normalize = normalize

        if self.hash_rep_at_last_update == hash(self):
            # this doesn't work
            logger.warning(f'Nothing has changed since last update at {self.last_update}')
            return

        self._linear_quantile_function = None

        ft_line_density = {}
        # line_density = {}
        # not_line_density = {}

        # add the densities
        # tilting: [@Grubel1999]: Computation of Compound Distributions I: Aliasing Errors and Exponential Tilting
        # (ASTIN 1999)
        # tilt x numbuck < 20 recommended log. 210
        # num buckets and max loss from bucket size
        N = 1 << log2
        MAXL = N * bs
        xs = np.linspace(0, MAXL, N, endpoint=False)
        # make all the single line aggs
        # note: looks like duplication but will all be references
        # easier for add_exa to have as part of the portfolio module
        # tilt
        if self.tilt_amount != 0:
            tilt_vector = np.exp(self.tilt_amount * np.arange(N))
        else:
            tilt_vector = None

        # where the answer will live
        self.density_df = pd.DataFrame(index=xs)
        self.density_df['loss'] = xs
        ft_all = None
        for agg in self.agg_list:
            raw_nm = agg.name
            nm = f'p_{agg.name}'
            # agg.update_work handles the reinsurance too
            agg.update_work(xs, self.padding, tilt_vector, 'exact' if agg.n < approx_freq_ge else approx_type,
                            sev_calc, discretization_calc, normalize, force_severity, debug)

            ft_line_density[raw_nm] = agg.ftagg_density
            self.density_df[nm] = agg.agg_density
            if ft_all is None:
                ft_all = np.copy(ft_line_density[raw_nm])
            else:
                ft_all *= ft_line_density[raw_nm]
        self.density_df['p_total'] = np.real(ift(ft_all, self.padding, tilt_vector))
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

        # make audit statistics_df df
        theoretical_stats = self.statistics_df.T.filter(regex='agg')
        theoretical_stats.columns = ['EX1', 'EX2', 'EX3', 'Mean', 'CV', 'Skew', 'Limit', 'P99.9Est']
        theoretical_stats = theoretical_stats[['Mean', 'CV', 'Skew', 'Limit', 'P99.9Est']]
        self.make_audit_df(columns=self.line_names_ex, theoretical_stats=theoretical_stats)

        # add exa details
        if add_exa:
            self.add_exa(self.density_df, ft_nots=ft_nots)
        else:
            # at least want F and S to get quantile functions
            self.density_df['F'] = np.cumsum(self.density_df.p_total)
            self.density_df['S'] = 1 - self.density_df.F

        self.ex = self.audit_df.loc['total', 'EmpMean']
        # pull out estimated stats to match Aggergate
        self.est_m, self.est_cv, self.est_skew = self.audit_df.loc['total', ['EmpMean', 'EmpCV', 'EmpSkew']]
        self.est_sd = self.est_m * self.est_cv
        self.est_var = self.est_sd ** 2

        self.last_update = np.datetime64('now')
        self.hash_rep_at_last_update = hash(self)
        if trim_df:
            self.trim_df()
        # invalidate stored functions
        self._linear_quantile_function = None
        self.q_temp = None
        self._cdf = None

    def make_audit_df(self, columns, theoretical_stats=None):
        """
        Add or update the audit_df.

        """
        # these are the values set by the audit
        column_names = ['Sum probs', 'EmpMean', 'EmpCV', 'EmpSkew', "EmpKurt", 'EmpEX1', 'EmpEX2', 'EmpEX3'] + \
                        ['P' + str(100 * i) for i in self.audit_percentiles]
        if self.audit_df is None:
            self.audit_df = pd.DataFrame(columns=column_names)
        for col in columns:
            sump = np.sum(self.density_df[f'p_{col}'])
            t = self.density_df[f'p_{col}'] * self.density_df['loss']
            ex1 = np.sum(t)
            t *= self.density_df['loss']
            ex2 = np.sum(t)
            t *= self.density_df['loss']
            ex3 = np.sum(t)
            t *= self.density_df['loss']
            ex4 = np.sum(t)
            m, cv, s = MomentAggregator.static_moments_to_mcvsk(ex1, ex2, ex3)
            # empirical kurtosis
            kurt = (ex4 - 4 * ex3 * ex1 + 6 * ex1 ** 2 * ex2 - 3 * ex1 ** 4) / ((m * cv) ** 4) - 3
            ps = np.zeros((len(self.audit_percentiles)))
            temp = self.density_df[f'p_{col}'].cumsum()
            for i, p in enumerate(self.audit_percentiles):
                ps[i] = (temp > p).idxmax()
            newrow = [sump, m, cv, s, kurt, ex1, ex2, ex3] + list(ps)
            # TODO this is fragile if you request another set of percentiles
            # add the row, then subset it (when called first time you don't have
            # the Err columns)
            self.audit_df.loc[col, column_names] = newrow
        if theoretical_stats is not None and 'Mean' not in self.audit_df.columns:
            # merges on the first five columns: mean, cv, sk, limit, p99.9E
            self.audit_df = pd.concat((theoretical_stats, self.audit_df), axis=1, sort=True)
        try:
            self.audit_df['MeanErr'] = self.audit_df['EmpMean'] / self.audit_df['Mean'] - 1
            self.audit_df['CVErr'] = self.audit_df['EmpCV'] / self.audit_df['CV'] - 1
            self.audit_df['SkewErr'] = self.audit_df['EmpSkew'] / self.audit_df['Skew'] - 1
        except ZeroDivisionError as e:
            raise e

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
        if self.density_df is None:
            return False

        any_false = False
        for a in self.agg_list:
            if not a.valid:
                logger.warning(f'Aggregate {a.name} fails validation')
                any_false = True

        if any_false:
            logger.warning(f'Exiting: Portfolio validation steps skipped due to failed aggregate validation')
            return False
        else:
            logger.info('All aggregate objects are not unreasonable')

        df = self.describe.xs('total', level=0, axis=0).abs()
        try:
            df['Err Skew(X)'] = df['Est Skew(X)'] / df['Skew(X)'] - 1
        except ZeroDivisionError:
            df['Err Skew(X)'] = np.nan
        except TypeError:
            df['Err Skew(X)'] = np.nan
        eps = self.validation_eps
        if df.loc['Sev', 'Err E[X]'] > eps:
            logger.warning('FAIL: Portfolio Sev mean error > eps')
            return False

        if df.loc['Agg', 'Err E[X]'] > eps:
            logger.warning('FAIL: Portfolio Agg mean error > eps')
            return False

        if abs(df.loc['Sev', 'Err E[X]']) > 0 and df.loc['Agg', 'Err E[X]'] > 10 * df.loc['Sev', 'Err E[X]']:
            logger.warning('FAIL: Agg mean error > 10 * sev error')
            return False
        try:
            if np.inf > df.loc['Sev', 'CV(X)'] > 0 and df.loc['Sev', 'Err CV(X)'] > 10 * eps:
                logger.warning('FAIL: Portfolio Sev CV error > eps')
                return False

            if np.inf > df.loc['Agg', 'CV(X)'] > 0 and df.loc['Agg', 'Err CV(X)'] > 10 * eps:
                logger.warning('FAIL: Portfolio Agg CV error > eps')
                return False

            if np.inf > df.loc['Sev', 'Skew(X)'] > 0 and df.loc['Sev', 'Err Skew(X)'] > 100 * eps:
                logger.warning('FAIL: Portfolio Sev skew error > eps')
                return False

            if np.inf > df.loc['Agg', 'Skew(X)'] > 0 and df.loc['Agg', 'Err Skew(X)'] > 100 * eps:
                logger.warning('FAIL: Portfolio Agg skew error > eps')
                return False
        except (TypeError, ZeroDivisionError):
            pass

        logger.info('Portfolio does not fail any validation: not unreasonable')
        return True

    @property
    def priority_capital_df(self):
        if self._priority_capital_df is None:
            # default priority analysis
            # never use these, so movaed out of main update
            # TODO make a property and create on demand
            logger.debug('Adding EPDs in Portfolio.update')
            epds = np.hstack(
                [np.linspace(0.5, 0.1, 4, endpoint=False)] +
                [np.linspace(10 ** -n, 10 ** -(n + 1), 9, endpoint=False) for n in range(1, 7)])
            epds = np.round(epds, 7)
            self._priority_capital_df = pd.DataFrame(index=pd.Index(epds))
            for col in self.line_names:
                for i in range(3):
                    self._priority_capital_df['{:}_{:}'.format(col, i)] = self.epd_2_assets[(col, i)](epds)
                    self._priority_capital_df['{:}_{:}'.format('total', 0)] = self.epd_2_assets[('total', 0)](
                        epds)
                col = 'not ' + col
                for i in range(2):
                    self._priority_capital_df['{:}_{:}'.format(col, i)] = self.epd_2_assets[(col, i)](epds)
            self._priority_capital_df['{:}_{:}'.format('total', 0)] = self.epd_2_assets[('total', 0)](epds)
            self._priority_capital_df.columns = self._priority_capital_df.columns.str.split("_", expand=True)
            self._priority_capital_df.sort_index(axis=1, level=1, inplace=True)
            self._priority_capital_df.sort_index(axis=0, inplace=True)
        return self._priority_capital_df

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

    def gradient(self, epsilon=1 / 128, kind='homog', method='forward', distortion=None, remove_fuzz=True,
                 extra_columns=None, do_swap=True):
        """
        Compute the gradient of various quantities relative to a change in the volume of each
        portfolio component.

        Focus is on the quantities used in rate calculations: S, gS, p_total, exa, exag, exi_xgta, exi_xeqq,
        exeqa, exgta etc.


        homog:


        inhomog:

        :param epsilon: the increment to use; scale is 1+epsilon
        :param kind:    homog[ogeneous] or inhomog: homog computes impact of f((1+epsilon)X_i)-f(X_i). Inhomog
            scales the frequency and recomputes. Note inhomog will have a slight scale issues with
            E[Severity]
        :param method:  forward, central (using epsilon/2) or backwards
        :param distortion: if included derivatives of statistics using the distortion, such as exag are also
            computed
        :param extra_columns: extra columns to compute dervs of. Note there is virtually no overhead of adding additional
            columns
        :param do_swap: force the step to replace line with line+epsilon in all not line2's line2!=line1; whether you need
            this or not depends on what variables you to be differentiated. E.g. if you ask for exa_total only you don't need
            to swap. But if you want exa_A, exa_B you do, otherwise the d/dA exa_B won't be correct.
            TODO: replace with code!
        :return:   DataFrame of gradients and audit_df in an Answer class
        """

        if kind == 'inhomog' or kind[:7] == 'inhomog':
            raise NotImplementedError(f'kind=={kind} not yet implemented')

        if method == 'central':
            raise NotImplementedError(f'method=={method} not yet implemented')

        if method not in ('forward', 'backwards', 'central'):
            raise ValueError('Inadmissible option passed to gradient.')

        if self.tilt_amount:
            raise ValueError('Gradients do not allow tilts')

        # central = run this code forwards and backwards with epsilon / 2 and average?!

        # Forwards or backwards
        if method == 'forward':
            delta = 1 + epsilon
            dx = epsilon
            pm = '+'
        else:
            delta = 1 - epsilon
            dx = -epsilon
            pm = '-'

        # FFT functions for use in exa calculations; padding needs to be consistent with agg
        def loc_ft(x):
            return ft(x, self.padding, None)

        def loc_ift(x):
            return ift(x, self.padding, None)

        # setup (compare self.update)
        xs = self.density_df['loss'].values
        tilt_vector = None

        # (1+e)X computed for each line
        agg_epsilon_df = pd.DataFrame(index=xs)

        # compute the individual line (1+epsilon)X_i and then the revised total
        new_aggs = {}
        for base_agg in self.agg_list:
            agg = base_agg.rescale(delta, kind)
            new_aggs[base_agg.name] = agg
            _a = agg.update_work(xs, self.padding, tilt_vector, 'exact' if agg.n < self.approx_freq_ge else self.approx_type,
                            self.sev_calc, self.discretization_calc)
            agg_epsilon_df[f'p_{agg.name}'] = agg.agg_density
            # the total with the line incremented
            agg_epsilon_df[f'p_total_{agg.name}'] = \
                np.real(self.ift(agg.ftagg_density * loc_ft(self.density_df[f'ημ_{agg.name}'])))

        self.remove_fuzz(df=agg_epsilon_df, force=remove_fuzz, log='gradient')

        percentiles = [0.9, 0.95, 0.99, 0.996, 0.999, 0.9999, 1 - 1e-6]
        audit_df = pd.DataFrame(
            columns=['Sum probs', 'EmpMean', 'EmpCV', 'EmpSkew', 'EmpEX1', 'EmpEX2', 'EmpEX3'] +
                    ['P' + str(100 * i) for i in percentiles])
        # 949 = epsilon 916 Delta
        ep = chr(949)
        D = chr(916)
        for col in agg_epsilon_df.columns:
            sump = np.sum(agg_epsilon_df[col])
            t = agg_epsilon_df[col] * xs
            ex1 = np.sum(t)
            t *= xs
            ex2 = np.sum(t)
            t *= xs
            ex3 = np.sum(t)
            m, cv, s = MomentAggregator.static_moments_to_mcvsk(ex1, ex2, ex3)
            ps = np.zeros((len(percentiles)))
            temp = agg_epsilon_df[col].cumsum()
            for i, p in enumerate(percentiles):
                ps[i] = (temp > p).idxmax()
            audit_df.loc[f'{col[2:]}{pm}{ep}', :] = [sump, m, cv, s, ex1, ex2, ex3] + list(ps)
        for l in self.line_names_ex:
            audit_df.loc[l, :] = self.audit_df.loc[l, :]
        # differences
        for l in self.line_names:
            audit_df.loc[f'{l}{D}', :] = audit_df.loc[f'{l}{pm}{ep}'] - audit_df.loc[l]
            audit_df.loc[f'total_{l}{D}', :] = audit_df.loc[f'total_{l}{pm}{ep}'] - audit_df.loc['total']
        audit_df = audit_df.sort_index()

        # now need to iterate through each line to compute differences
        # variables we want to differentiate
        # note asking for derv of exa_A makes things a lot more complex...see swap function below
        # may want to default to not including that?
        columns_of_interest = ['S'] + [f'exa_{line}' for line in self.line_names_ex]
        if extra_columns:
            columns_of_interest += extra_columns

        # these are the columns add_exa expects
        columns_p_only = ['loss'] + [f'p_{line}' for line in self.line_names_ex] + \
                         [f'ημ_{line}' for line in self.line_names]

        # first, need a base and add exag to coi
        if distortion:
            _x = self.apply_distortion(distortion, create_augmented=False)
            base = _x.augmented_df
            columns_of_interest.extend(['gS'] + [f'exag_{line}' for line in self.line_names_ex])
        else:
            base = self.density_df

        # and then a holder for the answer
        answer = pd.DataFrame(index=pd.Index(xs, name='loss'),
                              columns=pd.MultiIndex.from_arrays(((), ()), names=('partial_wrt', 'line')))
        answer.columns.name = 'derivatives'

        # the exact same as add exa; same padding no tilt
        def ae_ft(x):
            return ft(x, 1, None)

        def swap(adjust_line):
            """

            in the not line swap A for Ae

            E.g. X = A + B + C and adjust_Line = A. Then not A is the same, but for not B and not C you
            need to swap A with A+epsilon. This function accomplishes the swap.

            :param ημ:
            :param A:
            :param Ae:
            :return: collection of all not lines with adjusted adjust_line
            adjusted_not_fft[line] is fft of not_line with adjust_line swapped out for line + epsilon
            """

            # look if there are just two lines then this is easy......but want to see if this works too...
            adjusted_not_fft = {}
            adjust_line_ft = ae_ft(agg_epsilon_df[f'p_{adjust_line}'])
            base_line_ft = ae_ft(base[f'p_{adjust_line}'])
            adj_factor = adjust_line_ft / base_line_ft
            adj_factor[np.logical_and(base_line_ft == 0, adjust_line_ft == 0)] = 0
            n_and = np.sum(np.logical_and(base_line_ft == 0, adjust_line_ft == 0))
            n_or = np.sum(np.logical_or(base_line_ft == 0, adjust_line_ft == 0))
            # TODO sort this out...often not actually the same...
            logger.info(f'SAME? And={n_and} Or={n_or}; Zeros in fft(line) and '
                        'fft(line + epsilon for {adjust_line}.')
            for line in self.line_names:
                if line == adjust_line:
                    # nothing changes, adjust_line not in not adjust_line it doesn't need to change
                    adjusted_not_fft[line] = ae_ft(base[f'ημ_{line}'])
                else:
                    adjusted_not_fft[line] = ae_ft(base[f'ημ_{line}']) * adj_factor
            return adjusted_not_fft

        # finally perform iteration and compute differences
        for line in self.line_names:
            gradient_df = base[columns_p_only].copy()
            gradient_df[f'p_{line}'] = agg_epsilon_df[f'p_{line}']
            gradient_df['p_total'] = agg_epsilon_df[f'p_total_{line}']
            if do_swap:
                # we also need to update ημ_lines whenever it includes line (i.e. always
                # call original add_exa function, operates on gradient_df in-place
                self.add_exa(gradient_df, ft_nots=swap(line))
            else:
                self.add_exa(gradient_df)
            if distortion is not None:
                # apply to line + epsilon
                gradient_df = self.apply_distortion(distortion, df_in=gradient_df, create_augmented=False).augmented_df

            # compute differentials and store answer!
            answer[[(line, i) for i in columns_of_interest]] = (gradient_df[columns_of_interest] -
                                                                base[columns_of_interest]) / dx

        return Answer(gradient=answer, audit=audit_df, new_aggs=new_aggs)

    def report(self, report_list='quick'):
        """

        :param report_list:
        :return:
        """
        full_report_list = ['statistics', 'quick', 'audit', 'priority_capital', 'priority_analysis']
        if report_list == 'all':
            report_list = full_report_list
        for r in full_report_list:
            if r in report_list:
                html_title(f'{r} Report for {self.name}', 1)
                if r == 'priority_capital':
                    if self._priority_capital_df is not None:
                        display(self._priority_capital_df.loc[1e-3:1e-2, :].style)
                    else:
                        html_title(f'Report {r} not generated', 2)
                elif r == 'quick':
                    if self.audit_df is not None:
                        df = self.audit_df[['Mean', 'EmpMean', 'MeanErr', 'CV', 'EmpCV', 'CVErr', 'P99.0']]
                        display(df.style)
                    else:
                        html_title(f'Report {r} not generated', 2)
                else:
                    df = getattr(self, r + '_df', None)
                    if df is not None:
                        try:
                            display(df.style)
                        except ValueError:
                            display(df)
                    else:
                        html_title(f'Report {r} not generated', 2)

    @property
    def report_df(self):
        if self.audit_df is not None:
            summary_sl = (slice(None), ['mean', 'cv', 'skew'])

            bit1 = self.statistics_df.loc[summary_sl, :]
            bit1.index = ['freq_m', 'sev_m', 'agg_m', 'freq_cv', 'sev_cv', 'agg_cv', 'freq_skew', 'sev_skew', 'agg_skew']


            bit2 = self.audit_df[['EmpMean', 'EmpCV', 'EmpSkew', 'EmpKurt', 'P99.0', 'P99.6']].T
            bit2.index = ['agg_emp_m', 'agg_emp_cv', 'agg_emp_skew', 'agg_emp_kurt', 'P99.0_emp', 'P99.6_emp']

            df = pd.concat((bit1, bit2), axis=0)
            df.loc['agg_m_err', :] = df.loc['agg_emp_m'] / df.loc['agg_m'] - 1
            df.loc['agg_cv_err', :] = df.loc['agg_emp_cv'] / df.loc['agg_cv'] - 1
            df.loc['agg_skew_err', :] = df.loc['agg_emp_skew'] / df.loc['agg_skew'] - 1
            df = df.loc[['freq_m', 'freq_cv', 'freq_skew', 'sev_m', 'sev_cv', 'sev_skew',
                         'agg_m', 'agg_emp_m', 'agg_m_err',
                         'agg_cv', 'agg_emp_cv', 'agg_cv_err',
                         'agg_skew', 'agg_emp_skew', 'agg_skew_err',
                         'agg_emp_kurt',
                         'P99.0_emp','P99.6_emp']]
            df.columns.name = 'unit'
            df.index.name = 'statistic'
        else:
            df = None
        return df

    @property
    def pprogram(self):
        """
        pretty print the program to html
        """
        pprint_ex(self.program, 20)

    def limits(self, stat='range', kind='linear', zero_mass='include'):
        """
        Suggest sensible plotting limits for kind=range, density, .. (same as Aggregate).

        Should optionally return a locator for plots?

        Called by ploting routines. Single point of failure!

        Must work without q function when not computed (apply_reins_work for
        occ reins...uses report_ser instead).

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
        :param figsize: arguments passed to make_mosaic_figure if no axd
        :return:
        """

        if axd is None:
            self.figure, axd = make_mosaic_figure('AB', figsize=figsize)

        ax = axd['A']
        xl = self.limits()
        yl = self.limits(stat='density', zero_mass='exclude')
        bit = self.density_df.filter(regex='p_[a-zA-Z]')
        if bit.shape[1] == 3:
            # put total first = Book standard
            bit = bit.iloc[:, [2,0,1]]
        bit.plot(ax=ax, xlim=xl, ylim=yl)
        ax.set(xlabel='Loss', ylabel='Density')
        ax.legend()

        ax = axd['B']
        xl = self.limits(kind='log')
        yl = self.limits(stat='logy')
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

    def plot_old(self, kind='density', line='all', p=0.99, c=0, a=0, axiter=None, figsize=None, height=2,
             aspect=1, **kwargs):
        """
        kind = density
            simple plotting of line density or not line density;
            input single line or list of lines;
            log underscore appended as appropriate

        kind = audit
            Miscellaneous audit graphs

        kind = priority
            LEV EXA, E2Pri and combined plots by line

        kind = quick
            four bar charts of EL etc.

        kind = collateral
            plot to illustrate bivariate density of line vs not line with indicated asset a and capital c

        :param kind: density | audit | priority | quick | collateral
        :param line: lines to use, defaults to all
        :param p:   for graphics audit, x-axis scale has maximum q(p)
        :param c:   collateral amount
        :param a:   asset amount
        :param axiter: optional, pass in to use existing ``axiter``
        :param figsize: arguments passed to axis_factory if no axiter
        :param height:
        :param aspect:
        :param kwargs: passed to pandas plot routines
        :return:
        """
        do_tight = (axiter is None)

        if kind == 'quick':
            if self.audit_df is not None:
                axiter = axiter_factory(axiter, 4, figsize, height, aspect)
            else:
                axiter = axiter_factory(axiter, 3, figsize, height, aspect)

            self.statistics_df.loc[('agg', 'mean')]. \
                sort_index(ascending=True, axis=0). \
                plot(kind='bar', rot=-45, title='Expected Loss', ax=next(axiter))

            self.statistics_df.loc[('agg', 'cv')]. \
                sort_index(ascending=True, axis=0). \
                plot(kind='bar', rot=-45, title='Coeff of Variation', ax=next(axiter))

            self.statistics_df.loc[('agg', 'skew')]. \
                sort_index(ascending=True, axis=0). \
                plot(kind='bar', rot=-45, title='Skewness', ax=next(axiter))

            if self.audit_df is not None:
                self.audit_df['P99.9']. \
                    sort_index(ascending=True, axis=0). \
                    plot(kind='bar', rot=-45, title='99.9th Percentile', ax=next(axiter))

        elif kind == 'density':
            if isinstance(line, str):
                if line == 'all':
                    line = [f'p_{i}' for i in self.line_names_ex]
                else:
                    line = ['p_' + line]
            elif isinstance(line, list):
                line = ['p_' + i if i[0:2] != 'ημ' else i for i in line]
            else:
                raise ValueError
            line = sorted(line)
            if 'subplots' in kwargs and len(line) > 1:
                axiter = axiter_factory(axiter, len(line), figsize, height, aspect)
                ax = axiter.grid(len(line))
            else:
                axiter = axiter_factory(axiter, 1, figsize, height, aspect)
                # want to be able to pass an axis in rather than an axiter...
                if isinstance(axiter, AxisManager):
                    ax = axiter.grid(1)
                else:
                    ax = axiter
            self.density_df[line].sort_index(axis=1). \
                plot(sort_columns=True, ax=ax, **kwargs)
            if 'logy' in kwargs:
                _t = 'log Density'
            else:
                _t = 'Density'
            if 'subplots' in kwargs and isinstance(ax, Iterable):
                for a, l in zip(ax, line):
                    a.set(title=f'{l} {_t}')
                    a.legend().set_visible(False)
            elif isinstance(ax, Iterable):
                for a in ax:
                    a.set(title=f'{_t}')
            else:
                ax.set(title=_t)

        elif kind == 'audit':
            D = self.density_df
            # n_lines = len(self.line_names_ex)
            n_plots = 12  # * n_lines + 8  # assumes that not lines have been taken out!
            axiter = axiter_factory(axiter, n_plots, figsize, height, aspect)

            # make appropriate scales
            density_scale = D.filter(regex='^p_').iloc[1:, :].max().max()
            expected_loss_scale = np.sum(D.loss * D.p_total) * 1.05
            large_loss_scale = (D.p_total.cumsum() > p).idxmax()

            # densities
            temp = D.filter(regex='^p_', axis=1)
            ax = axiter.grid(1)
            temp.plot(ax=ax, ylim=(0, density_scale), xlim=(0, large_loss_scale), title='Densities')

            ax = axiter.grid(1)
            temp.plot(ax=ax, logx=True, ylim=(0, density_scale), title='Densities log/linear')

            ax = axiter.grid(1)
            temp.plot(ax=ax, logy=True, xlim=(0, large_loss_scale), title='Densities linear/log')

            ax = axiter.grid(1)
            temp.plot(ax=ax, logx=True, logy=True, title='Densities log/log')

            # graph of cumulative loss cost and rate of change of cumulative loss cost
            temp = D.filter(regex='^exa_[^η]')
            # need to check exa actually computed
            if temp.shape[1] == 0:
                logger.error('Update exa before audit plot')
                return

            ax = axiter.grid(1)
            temp.plot(legend=True, ax=ax, xlim=(0, large_loss_scale), ylim=(0, expected_loss_scale),
                      title='Loss Cost by Line: $E(X_i(a))$')

            ax = axiter.grid(1)
            temp.diff().plot(legend=True, ax=ax, xlim=(0, large_loss_scale), ylim=(0, D.index[1]),
                             title='Change in Loss Cost by Line: $\\nabla E(X_i(a))$')

            # E(X_i / X | X > a); exi_x_lea_ dropped
            prefix_and_titles = dict(exi_xgta_=r'$E(X_i/X \mid X>a)$',
                                     exeqa_=r'$E(X_i \mid X=a)$',
                                     exlea_=r'$E(X_i \mid X \leq a)$',
                                     exgta_=r'$E(X_i \mid X>a)$')
            for prefix in prefix_and_titles.keys():
                regex = f'^{prefix}[a-zA-Z]'
                ax = axiter.grid(1)
                D.filter(regex=regex).plot(ax=ax, xlim=(0, large_loss_scale))
                if prefix == 'exgta_':
                    ax.set_title(r'$E(X_i \mid X > a)$ by line and total')
                if prefix.find('xi_x') > 0:
                    # these are fractions between zero and 1; plot sum on same axis and adjust limit
                    D.filter(regex=regex).sum(axis=1).plot(ax=ax, label='calced total')
                    ax.set_ylim(-.05, 1.05)  # so you can see if anything weird is going on
                elif prefix == 'exgta_' or prefix == 'exeqa_':
                    # scale same as x axis - so you can see E(X | X=a) is the diagonal ds
                    ax.set_ylim(0, large_loss_scale)
                else:
                    # scale like mean
                    ax.set_ylim(0, expected_loss_scale)
                ax.set_title(prefix_and_titles[prefix])
                ax.legend(frameon=False)

            # Lee diagrams by peril - will fit in the sixth small plot
            ax = next(axiter)
            # force total first
            ax.plot(D.loc[:, 'p_total'].cumsum(), D.loss, label='total')
            for c in D.filter(regex='^p_[^t]').columns:
                ax.plot(D.loc[:, c].cumsum(), D.loss, label=c[2:])
            ax.legend(frameon=False)
            ax.set_title('Lee Diagram')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, large_loss_scale)

        elif kind == 'priority':
            xmax = self.q(p)
            n_lines = len(self.line_names_ex)
            n_plots = 3 + 2 * n_lines
            if axiter is None:
                axiter = axiter_factory(axiter, n_plots, figsize, height, aspect)

            for prefix, fmt in dict(lev_='LEV', exa_=r'$E(X_i\mid X=a)$', e2pri_=r'$E_2(X_i \mid X=a)$').items():
                ax = axiter.grid(1)
                self.density_df.filter(regex=f'{prefix}').plot(ax=ax, xlim=(0, xmax),
                                                               title=fmt)
                ax.set_xlabel('Capital assets')

            for line in self.line_names:
                ax = axiter.grid(1)
                self.density_df.filter(regex=f'(lev|exa|e2pri)_{line}$').plot(ax=ax, xlim=(0, xmax),
                                                                              title=f'{line.title()} by Priority')
                ax.set_xlabel('Capital assets')
            for col in self.line_names_ex:
                ax = axiter.grid(1)
                self.density_df.filter(regex=f'epd_[012]_{col}').plot(ax=ax, xlim=(0, xmax),
                                                                      title=f'{col.title()} EPDs', logy=True)

        elif kind == 'collateral':
            assert line != '' and line != 'all'
            if axiter is None:
                axiter = axiter_factory(axiter, 2, figsize, height, aspect)

            cmap = cm.BuGn
            if a == 0:
                a = self.q(p)
            pline = self.density_df.loc[0:a, f'p_{line}'].values
            notline = self.density_df.loc[0:a, f'ημ_{line}'].values
            xs = self.density_df.loc[0:a, 'loss'].values
            N = pline.shape[0]
            biv = np.matmul(notline.reshape((N, 1)), pline.reshape((1, N)))
            biv = biv  # / np.sum(np.sum(biv))
            for rho in [1, 0.05]:
                ax = next(axiter)
                ax.imshow(biv ** rho, cmap=cmap, origin='lower', extent=[0, xs[-1], 0, xs[-1]],
                          interpolation='nearest', **kwargs)
                cy = a - c
                ax.plot((c, c), (a - c, xs[-1]), 'k', linewidth=0.5)
                ax.plot((0, a), (a, 0), 'k', linewidth=0.5)
                if c > 0:
                    ax.plot((c, xs[-1]), (cy, xs[-1] * (a / c - 1)), 'k', linewidth=0.5)
                ax.plot((0, c, c), (a - c, a - c, 0), c='k', ls='--', linewidth=0.25)
                ax.set_xlim(0, xs[-1])
                ax.set_ylim(0, xs[-1])
                ax.set_xlabel(f'Line {line}')
                ax.set_ylabel(f'Not {line}')

        else:
            logger.error(f'Portfolio.plot | Unknown plot type {kind}')
            raise ValueError(f'Portfolio.plot unknown plot type {kind}')

        if do_tight:
            axiter.tidy()
            suptitle_and_tight(f'{kind.title()} Plots for {self.name.title()}')

    def uat_interpolation_functions(self, a0, e0):
        """
        Perform quick audit of interpolation functions

        :param a0: base assets
        :param e0: base epd
        :return:
        """
        # audit interpolation functions
        temp = pd.DataFrame(columns=['line', 'priority', 'epd', 'a from e', 'assets', 'e from a'])
        e2a = self.epd_2_assets
        a2e = self.assets_2_epd
        for i in range(3):
            for c in self.line_names + ['total'] + ['not ' + i for i in self.line_names]:
                # if i == 0 and c == 'total' or c != 'total':
                if (c, i) in a2e:
                    e = a2e[(c, i)](a0)
                    a = e2a[(c, i)](e0)
                    temp.loc[c + "_" + str(i), :] = (c, i, e, e2a[(c, i)](e), a, a2e[(c, i)](a))
        display(temp.style)

    def add_exa(self, df, ft_nots=None):
        """
        Use fft to add exa_XXX = E(X_i | X=a) to each dist

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

    def ft(self, x, tilt=None):
        """
        FT of x with padding and tilt applied
        """
        return ft(x, self.padding, tilt)

    def ift(self, x, tilt=None):
        """
        IFT of x with padding and tilt applied
        """
        return ift(x, self.padding, tilt)

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
            ft_all = ft(self.density_df.p_total, self.padding, None)
            for line in self.line_names:
                # create all for inner loop below
                ft_line_density[line] = ft(self.density_df[f'p_{line}'], self.padding, None)
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
                self.density_df[f'ημ_{line}'] = np.real(ift(ft_not, self.padding, None))

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

            # epd interpolation functions
            # capital and epd functions: for i = 0 and 1 we want line and not line
            loss_values = df.loss.values
            # only have type 2 when eta_mu is True
            options = [0, 1, 2] if eta_mu else [0, 1]
            for i in options:
                epd_values = -df['epd_{:}_{:}'.format(i, col)].values
                # if np.any(epd_values[1:] <= epd_values[:-1]):
                #     print(i, col)
                #     print( 1e12*(epd_values[1:][epd_values[1:] <= epd_values[:-1]] -
                #       epd_values[:-1][epd_values[1:] <= epd_values[:-1]]))
                # raise ValueError('Need to be sorted ascending')
                self._epd_2_assets[(col, i)] = minus_arg_wrapper(
                    interpolate.interp1d(epd_values, loss_values, kind='linear', assume_sorted=True,
                                         fill_value='extrapolate'))
                self._assets_2_epd[(col, i)] = minus_ans_wrapper(
                    interpolate.interp1d(loss_values, epd_values, kind='linear', assume_sorted=True,
                                         fill_value='extrapolate'))
            if eta_mu:
                for i in [0, 1]:
                    epd_values = -df['epd_{:}_ημ_{:}'.format(i, col)].values
                    self._epd_2_assets[('not ' + col, i)] = minus_arg_wrapper(
                        interpolate.interp1d(epd_values, loss_values, kind='linear', assume_sorted=True,
                                             fill_value='extrapolate'))
                    self._assets_2_epd[('not ' + col, i)] = minus_ans_wrapper(
                        interpolate.interp1d(loss_values, epd_values, kind='linear', assume_sorted=True,
                                             fill_value='extrapolate'))
        epd_values = -df['epd_0_total'].values
        # if np.any(epd_values[1:] <= epd_values[:-1]):
        #     print('total')
        #     print(epd_values[1:][epd_values[1:] <= epd_values[:-1]])
        # raise ValueError('Need to be sorted ascending')
        loss_values = df.loss.values
        self._epd_2_assets[('total', 0)] = minus_arg_wrapper(
            interpolate.interp1d(epd_values, loss_values, kind='linear', assume_sorted=True,
                                 fill_value='extrapolate'))
        self._assets_2_epd[('total', 0)] = minus_ans_wrapper(
            interpolate.interp1d(loss_values, epd_values, kind='linear', assume_sorted=True,
                                 fill_value='extrapolate'))

    @property
    def epd_2_assets(self):
        """
        Make epd to assets and vice versa
        Note that the Merton Perold method requies the eta_mu fields, hence set True
        """
        if self._epd_2_assets is None:
            self._epd_2_assets = {}
            self._assets_2_epd = {}
            self.add_exa_details(self.density_df, eta_mu=True)
        return self._epd_2_assets

    @property
    def assets_2_epd(self):
        if self._assets_2_epd is None:
            self._epd_2_assets = {}
            self._assets_2_epd = {}
            self.add_exa_details(self.density_df, eta_mu=True)
        return self._assets_2_epd

    def calibrate_distortion(self, name, r0=0.0, df=[0.0, .9], premium_target=0.0,
                             roe=0.0, assets=0.0, p=0.0, kind='lower', S_column='S',
                             S_calc='cumsum'):
        """
        Find transform to hit a premium target given assets of ``assets``.
        Fills in the values in ``g_spec`` and returns params and diagnostics...so
        you can use it either way...more convenient


        :param name: name of distortion
        :param r0:   fixed parameter if applicable
        :param df:  t-distribution degrees of freedom
        :param premium_target: target premium
        :param roe:             or ROE
        :param assets: asset level
        :param p:
        :param kind:
        :param S_column: column of density_df to use for calibration (allows routine to be used in other contexts; if
                so used must input a premium_target directly. If assets they are used; else max assets used
        :return:
        """

        # figure assets
        assert S_calc in ('S', 'cumsum')

        if S_column == 'S':
            if assets == 0:
                assert (p > 0)
                assets = self.q(p, kind)

            # figure premium target
            el = self.density_df.loc[assets, 'exa_total']
            if premium_target == 0:
                assert (roe > 0)
                # expected losses with assets
                premium_target = (el + roe * assets) / (1 + roe)
        else:
            # if assets not entered, calibrating to unlimited premium; set assets = max loss and let code trim it
            if assets == 0:
                assets = self.density_df.loss.iloc[-1]
            el = self.density_df.loc[assets, 'exa_total']

        # extract S and trim it: we are doing int from zero to assets
        # integration including ENDpoint is
        if S_calc == 'S':
            Splus = self.density_df.loc[0:assets, S_column].values
        else:
            Splus = (1 - self.density_df.loc[0:assets, 'p_total'].cumsum()).values

        last_non_zero = np.argwhere(Splus)
        ess_sup = 0
        if len(last_non_zero) == 0:
            # no nonzero element
            last_non_zero = len(Splus) + 1
        else:
            last_non_zero = last_non_zero.max()
        # remember length = max index + 1 because zero based
        if last_non_zero + 1 < len(Splus):
            # now you have issues...
            # truncate at first zero; numpy indexing because values
            S = Splus[:last_non_zero + 1]
            ess_sup = self.density_df.index[last_non_zero + 1]
            logger.info(
                'Portfolio.calibrate_distortion | Mass issues in calibrate_distortion...'
                f'{name} at {last_non_zero}, loss = {ess_sup}')
        else:
            # this calc sidesteps the Splus created above....
            if S_calc == 'original':
                S = self.density_df.loc[0:assets - self.bs, S_column].values
            else:
                S = (1 - self.density_df.loc[0:assets - self.bs, 'p_total'].cumsum()).values

        # now all S values should be greater than zero  and it is decreasing
        assert np.all(S > 0) and np.all(S[:-1] >= S[1:])

        if name == 'ph':
            lS = np.log(S)
            shape = 0.95  # starting param

            def f(rho):
                trho = S ** rho
                ex = np.sum(trho) * self.bs
                ex_prime = np.sum(trho * lS) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'wang':
            n = ss.norm()
            shape = 0.95  # starting param

            def f(lam):
                temp = n.ppf(S) + lam
                tlam = n.cdf(temp)
                ex = np.sum(tlam) * self.bs
                ex_prime = np.sum(n.pdf(temp)) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'ly':
            # linear yield model; min rol is ro/(1+ro)
            shape = 1.25  # starting param
            mass = ess_sup * r0 / (1 + r0)

            def f(rk):
                num = r0 + S * (1 + rk)
                den = 1 + r0 + rk * S
                tlam = num / den
                ex = np.sum(tlam) * self.bs + mass
                ex_prime = np.sum(S * (den ** -1 - num / (den ** 2))) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'clin':
            # capped linear, input rf as min rol
            shape = 1
            mass = ess_sup * r0

            def f(r):
                r0_rS = r0 + r * S
                ex = np.sum(np.minimum(1, r0_rS)) * self.bs + mass
                ex_prime = np.sum(np.where(r0_rS < 1, S, 0)) * self.bs
                return ex - premium_target, ex_prime
        elif name in ['roe', 'ccoc']:
            # constant roe
            # TODO Note if you input the roe this is easy!
            shape = 0.25
            def f(r):
                v = 1 / (1 + r)
                d = 1 - v
                mass = ess_sup * d
                r0_rS = d + v * S
                ex = np.sum(np.minimum(1, r0_rS)) * self.bs + mass
                ex_prime = np.sum(np.where(r0_rS < 1, S, 0)) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'lep':
            # layer equivalent pricing
            # params are d=r0/(1+r0) and delta* = r/(1+r)
            d = r0 / (1 + r0)
            shape = 0.25  # starting param
            # these hard to compute variables do not change with the parameters
            rSF = np.sqrt(S * (1 - S))
            mass = ess_sup * d

            def f(r):
                spread = r / (1 + r) - d
                temp = d + (1 - d) * S + spread * rSF
                ex = np.sum(np.minimum(1, temp)) * self.bs + mass
                ex_prime = (1 + r) ** -2 * np.sum(np.where(temp < 1, rSF, 0)) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'tt':
            # wang-t-t ... issue with df, will set equal to 5.5 per Shaun's paper
            # finding that is a reasonable level; user can input alternative
            # TODO bivariate solver for t degrees of freedom?
            # param is shape like normal
            t = ss.t(df)
            shape = 0.95  # starting param

            def f(lam):
                temp = t.ppf(S) + lam
                tlam = t.cdf(temp)
                ex = np.sum(tlam) * self.bs
                ex_prime = np.sum(t.pdf(temp)) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'cll':
            # capped loglinear
            shape = 0.95  # starting parameter
            lS = np.log(S)
            lS[0] = 0
            ea = np.exp(r0)

            def f(b):
                uncapped = ea * S ** b
                ex = np.sum(np.minimum(1, uncapped)) * self.bs
                ex_prime = np.sum(np.where(uncapped < 1, uncapped * lS, 0)) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'dual':
            # dual moment
            # be careful about partial at s=1
            shape = 2.0  # starting parameter
            lS = -np.log(1 - S)  # prob a bunch of zeros...
            lS[S == 1] = 0

            def f(rho):
                temp = (1 - S) ** rho
                trho = 1 - temp
                ex = np.sum(trho) * self.bs
                ex_prime = np.sum(temp * lS) * self.bs
                return ex - premium_target, ex_prime
        elif name == 'tvar':
            # tvar
            shape = 0.9   # starting parameter
            def f(rho):
                temp = np.where(S <= 1-rho, S / (1 - rho), 1)
                temp2 = np.where(S <= 1-rho, S / (1 - rho)**2, 1)
                ex = np.sum(temp) * self.bs
                ex_prime = np.sum(temp2) * self.bs
                return ex - premium_target, ex_prime

        elif name == 'wtdtvar':
            # weighted tvar with fixed p parameters
            shape = 0.5  # starting parameter
            p0, p1 = df
            s = np.array([0., 1-p1, 1-p0, 1.])

            def f(w):
                pt = (1 - p1) / (1 - p0) * (1 - w) + w
                gs = np.array([0.,  pt,   1., 1.])
                g = interp1d(s, gs, kind='linear')
                trho = g(S)
                ex = np.sum(trho) * self.bs
                ex_prime = (np.sum(np.minimum(S / (1 - p1), 1) - np.minimum(S / (1 - p0), 1))) * self.bs
                return ex - premium_target, ex_prime
        else:
            raise ValueError(f'calibrate_distortion not implemented for {name}')

        # numerical solve except for tvar, and roe when premium is known
        if name in ('roe', 'ccoc'):
            assert el and premium_target
            r = (premium_target - el) / (assets - premium_target)
            shape = r
            r0 = 0
            fx = 0
        else:
            i = 0
            fx, fxp = f(shape)
            # print(premium_target)
            # print('dist    iter       error            shape          deriv')
            max_iter = 200 if name == 'tvar' else 50
            while abs(fx) > 1e-5 and i < max_iter:
                # print(f'{name}\t{i: 3d}\t{fx: 8.3f}\t{shape:8.3f}\t{fxp:8.3f}')
                shape = shape - fx / fxp
                fx, fxp = f(shape)
                i += 1
            # print(f'{name}\t{i: 3d}\t{fx+premium_target: 8.3f}\t{shape:8.3f}\t{fxp:8.3f}\n\n')
            if abs(fx) > 1e-5:
                logger.warning(
                    f'Portfolio.calibrate_distortion | Questionable convergenge! {name}, target '
                    f'{premium_target} error {fx}, {i} iterations')

        # build answer
        dist = Distortion(name=name, shape=shape, r0=r0, df=df)
        dist.error = fx
        dist.assets = assets
        dist.premium_target = premium_target
        return dist

    def calibrate_distortions(self, LRs=None, COCs=None, ROEs=None, As=None, Ps=None, kind='lower', r0=0.03, df=5.5,
                              strict=True, S_calc='cumsum'):
        """
        Calibrate assets a to loss ratios LRs and asset levels As (iterables)
        ro for LY, it :math:`ro/(1+ro)` corresponds to a minimum rate online


        :param LRs:  LR or ROEs given
        :param ROEs: ROEs override LRs
        :param COCs: CoCs override LRs, preferred terms to ROE; ROE maintained for backwards compatibility.
        :param As:  Assets or probs given
        :param Ps: probability levels for quantiles
        :param kind:
        :param r0: for distortions that have a min ROL
        :param df: for tt
        :param strict: if=='ordered' then use the book nice ordering else
            if True only use distortions with no mass at zero, otherwise
            use anything reasonable for pricing
        :param S_calc:
        :return:
        """
        if COCs is not None:
            ROEs = COCs

        ans = pd.DataFrame(
            columns=['$a$', 'LR', '$S$', '$\\iota$', '$\\delta$', '$\\nu$', '$EL$', '$P$', 'Levg', '$K$',
                     'ROE', 'param', 'error', 'method'], dtype=float)
        ans = ans.set_index(['$a$', 'LR', 'method'], drop=True)
        dists = {}
        if As is None:
            if Ps is None:
                raise ValueError('Must specify assets or quantile probabilities')
            else:
                As = [self.q(p, kind) for p in Ps]
        for a in As:
            exa, S = self.density_df.loc[a, ['exa_total', 'S']]
            if ROEs is not None:
                # figure loss ratios
                LRs = []
                for r in ROEs:
                    delta = r / (1 + r)
                    nu = 1 - delta
                    prem = nu * exa + delta * a
                    LRs.append(exa / prem)
            for lr in LRs:
                P = exa / lr
                profit = P - exa
                K = a - P
                iota = profit / K
                delta = iota / (1 + iota)
                nu = 1 - delta
                if strict == 'ordered':
                    # d_list = ['roe', 'ph', 'wang', 'dual', 'tvar']
                    d_list = ['ccoc', 'ph', 'wang', 'dual', 'tvar']
                else:
                    d_list = Distortion.available_distortions(pricing=True, strict=strict)

                for dname in d_list:
                    dist = self.calibrate_distortion(name=dname, r0=r0, df=df, premium_target=P, assets=a, S_calc=S_calc)
                    dists[dname] = dist
                    ans.loc[(a, lr, dname), :] = [S, iota, delta, nu, exa, P, P / K, K, profit / K,
                                                  dist.shape, dist.error]
        # very helpful to keep these...
        self.dist_ans = ans
        self.dists = dists
        return ans

    @property
    def distortion_df(self):
        """
        Nicely formatted version of self.dist_ans (that exhibited several bad choices!).

        ROE returned as COC in modern parlance.
        """
        if self.dist_ans is None:
            return None

        df = self.dist_ans.iloc[:, [0,4,5,6,7,8,9,10]].copy()
        df.index.names = ['a', 'LR', 'method']
        df.columns = ['S', 'L', 'P', 'PQ', 'Q', 'COC', 'param', 'error']
        return df

    def apply_distortions(self, dist_dict, As=None, Ps=None, kind='lower', efficient=True):
        """
        Apply a list of distortions, summarize pricing and produce graphical output
        show loss values where  :math:`s_ub > S(loss) > s_lb` by jump

        :param kind:
        :param dist_dict: dictionary of Distortion objects
        :param As: input asset levels to consider OR
        :param Ps: input probs (near 1) converted to assets using ``self.q()``
        :return:
        """
        ans = []
        if As is None:
            As = np.array([float(self.q(p, kind)) for p in Ps])

        for g in dist_dict.values():
            _x = self.apply_distortion(g, efficient=efficient)
            df = _x.augmented_df
            # extract range of S values
            if As[0] in df.index:
                temp = df.loc[As, :].filter(regex=r'^loss|^S|exa[g]?_[^η][\.:~a-zA-Z0-9]*$|exag_sumparts|lr_').copy()
            else:
                logger.warning(f'{As} not in index...max value is {max(df.index)}...selecing that!')
                temp = df.iloc[[-1], :].filter(regex=r'^loss|^S|exa[g]?_[^η][\.:~a-zA-Z0-9]*$|exag_sumparts|lr_').copy()
            # jump = sensible_jump(len(temp), num_assets)
            # temp = temp.loc[::jump, :].copy()
            temp['method'] = g.name
            ans.append(temp)

        ans_table = pd.concat(ans)
        ans_table['return'] = np.round(1 / ans_table.S, 0)

        df2 = ans_table.copy()
        df2 = df2.set_index(['loss', 'method', 'return', 'S'])
        df2.columns = df2.columns.str.split('_', expand=True)
        ans_stacked = pd.DataFrame(df2.stack().stack()).reset_index()
        ans_stacked.columns = ['assets', 'method', 'return', 'S', 'line', 'stat', 'value']

        # figure reasonable max and mins for LR plots
        mn = ans_table.filter(regex='^lr').min().min()
        mn1 = mn
        mx = ans_table.filter(regex='^lr').max().max()
        mn = np.round(mn * 20, 0) / 20
        mx = np.round(mx * 20, 0) / 20
        if mx >= 0.9:
            mx = 1
        if mn <= 0.2:
            mn = 0
        if mn1 < mn:
            mn -= 0.1

        return ans_table, ans_stacked

    def apply_distortion(self, dist, view='ask', plots=None, df_in=None, create_augmented=True,
                         S_calculation='forwards', efficient=True):
        """
        Apply the distortion, make a copy of density_df and append various columns to create augmented_df.

        augmented_df depends on the distortion but only includes variables that work for all asset levels, e.g.

        1. marginal loss, lr, roe etc.
        2. bottom up totals

        Top down total depend on where the "top" is and do not work in general. They are handled in analyze_distortions
        where you explicitly provide a top.

        Does not touch density_df: that is independent of distortions

        Optionally produce graphics of results controlled by plots a list containing none or more of:

        1. basic: exag_sumparts, exag_total df.exa_total
        2. extended: the full original set

        Per 0.11.0: no mass at 0 allowed. If you want to use a distortion with mass at 0 you must use
        a close approximation.


        :type create_augmented: object
        :param dist: agg.Distortion
        :param view: bid or ask price
        :param plots: iterable of plot types
        :param df_in: when called from gradient you want to pass in gradient_df and use that; otherwise use self.density_df
        :param create_augmented: store the output in self.augmented_df
        :param S_calculation: if forwards, recompute S summing p_total forwards...this gets the tail right; the old method was
                backwards, which does not change S
        :param efficient: just compute the bare minimum (T. series, not M. series) and return
        :return: density_df with extra columns appended
        """

        if not efficient:
            # make sure eta-mu columns have been computed
            _ = self.assets_2_epd

        # initially work will "full precision"
        if df_in is None:
            df = self.density_df.copy()
        else:
            df = df_in

        # PREVIOUSLY: did not make this adjustment because loss of resolution on small S values
        # however, it doesn't work well for very thick tailed distributions, hence intro of S_calculation
        # July 2020 (during COVID-Trump madness) try this instead
        if S_calculation == 'forwards':
            logger.debug('Using updated S_forwards calculation in apply_distortion! ')
            df['S'] = 1 - df.p_total.cumsum()

        if type(dist) == str:
            # try looking it up in calibrated distortions
            dist = self.dists[dist]

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
        if not np.alltrue(df.S.iloc[1:] <= df.S.iloc[:-1].values):
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

            if create_augmented:
                self.augmented_df = df
            return Answer(augmented_df=df)

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
        # lhopital's rule estimate of g'(1) = ROE(1)
        gprime1 = g_prime(1) # (g(1 - ϵ) - (1 - ϵ)) / (1 - g(1 - ϵ))
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

        f_distortion = f_byline = f_bylineg = f_exas = None
        if plots == 'all':
            plots = ['basic', 'extended']
        if plots:
            if 'basic' in plots:
                f_distortion, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.plot(df.filter(regex='^exag_[^η]').sum(axis=1), label='Sum of Parts')
                ax.plot(df.exag_total, label='Total')
                ax.plot(df.exa_total, label='Loss')
                ax.legend()
                ax.set_title(f'Mass audit for {dist.name}')
                ax.legend(loc="upper right")
                ax.grid()

            if 'extended' in plots:
                # yet more graphics, but the previous one is the main event
                # truncate for graphics
                max_x = 1.1 * self.q(1 - 1e-6)
                df_plot = df.loc[0:max_x, :]

                f_exas, axs, axiter = AxisManager.make_figure(12, sharex=True)

                ax = next(axiter)
                df_plot.filter(regex='^p_').sort_index(axis=1).plot(ax=ax)
                ax.set_ylim(0, df_plot.filter(regex='p_[^η]').iloc[1:, :].max().max())
                ax.set_title("Densities")
                ax.legend(loc="upper right")
                ax.grid()

                ax = next(axiter)
                df_plot.loc[:, ['p_total', 'gp_total']].plot(ax=ax)
                ax.set_title("Total Density and Distortion")
                ax.legend(loc="upper right")
                ax.grid()

                ax = next(axiter)
                df_plot.loc[:, ['S', 'gS']].plot(ax=ax)
                ax.set_title("S, gS")
                ax.legend(loc="upper right")
                ax.grid()

                # exi_xlea removed
                for prefix in ['exa', 'exag', 'exeqa', 'exgta', 'exi_xeqa', 'exi_xgta']:
                    # look ahead operator: does not match n just as the next char, vs [^n] matches everything except n
                    ax = next(axiter)
                    df_plot.filter(regex=f'^{prefix}_(?!ημ)[a-zA-Z0-9]+$').sort_index(axis=1).plot(ax=ax)
                    ax.set_title(f'{prefix} by line')
                    ax.legend(loc="upper left")
                    ax.grid()
                    if prefix.find('xi_x') > 0:
                        # fix scale for proportions
                        ax.set_ylim(-0.025, 1.025)

                ax = next(axiter)
                # _ = df_plot[[f'exa_{i}' for i in self.line_names]] / df.exa_total
                # _.sort_index(axis=1).plot(ax=ax)
                ax.set_title('Proportion of loss: T.L_line / T.L_total')
                ax.set_ylim(0, 1.05)
                ax.legend(loc='upper left')
                ax.grid()

                ax = next(axiter)
                # _ = df_plot[[f'exag_{i}' for i in self.line_names]] / df.exag_total
                # _.sort_index(axis=1).plot(ax=ax)
                ax.set_title('Proportion of premium: T.P_line / T.P_total')
                # ax.set_ylim(0, 1.05)
                ax.legend(loc='upper left')
                ax.grid()

                ax = next(axiter)
                df_plot.filter(regex='^M.LR_').sort_index(axis=1).plot(ax=ax)
                ax.set_title('LR with the Natural (constant layer ROE) Allocation')
                ax.legend(loc='lower right')
                ax.grid()

                # by line plots
                nl = len(self.line_names_ex)
                f_byline, axs, axiter = AxisManager.make_figure(nl)
                for line in self.line_names:
                    ax = next(axiter)
                    df_plot.filter(regex=f'ex(le|eq|gt)a_{line}').sort_index(axis=1).plot(ax=ax)
                    ax.set_title(f'{line} EXs')
                    # ?
                    ax.set(ylim=[0, self.q(0.999, 'lower')])
                    ax.legend(loc='upper left')
                    ax.grid()
                AxisManager.tidy_up(f_byline, axiter)

                # compare exa with exag for all lines
                f_bylineg, axs, axiter = AxisManager.make_figure(nl)
                for line in self.line_names_ex:
                    ax = next(axiter)
                    df_plot.filter(regex=f'exa[g]?_{line}$').sort_index(axis=1).plot(ax=ax)
                    ax.set_title(f'{line} T.L and T.P')
                    ax.legend(loc='lower right')
                    ax.grid()
                AxisManager.tidy_up(f_bylineg, axiter)

        if create_augmented:
            self.augmented_df = df
            # store associated distortion
            self._distortion = dist

        return Answer(augmented_df=df,
                      f_distortion=f_distortion, f_byline=f_byline, f_bylineg=f_bylineg, f_exas=f_exas)

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
        if kind=='var': kind = 'lower'

        if kind=='tvar':
            d = {a.name: a.tvar(p) for a in self.agg_list}
            d['total'] = self.tvar(p)
        elif kind=='epd':
            if p >= 0.7:
                p = 1 - p
                logger.debug(f'Illogical value of p={1-p:.5g} passed for epd; using {p:.5g}')
            d = {ln: float(self.epd_2_assets[(ln, 0)](p)) for ln in self.line_names_ex }
        else:
            d = {a.name: a.q(p, kind) for a in self.agg_list}
            d['total'] = self.q(p, kind)
        if total != 'total':
            d[self.name] = d['total']
            del d['total']
        if snap and kind in ['tvar', 'epd']:
            d = {k: self.snap(v) for k, v in d.items()}
        return d

    def gamma(self, a=None, p=None, kind='lower', compute_stand_alone=False, axs=None, plot_mode='return'):
        """
        Return the vector gamma_a(x), the conditional layer effectiveness given assets a.
        Assets specified by percentile level and type (you need a in the index)
        gamma can be created with no base and no calibration - it does not depend on a distortion.
        It only depends on total losses.

        Returns the total and by layer versions, see
        "Main Result for Conditional Layer Effectiveness; Piano Diagram" in OneNote

        In total gamma_a(x) = E[ (a ^ X) / X | X > x] is the average rate of reimbursement for losses above x
        given capital a. It satisfies int_0^\infty gamma_a(x) S(x) dx = E[a ^ X]. Notice the integral is to
        infinity, regardless of the amount of capital a.

        By line gamma_{a,i}(x) = E[ E[X_i | X] / X  {(X ^ a) / X} 1_{X>x} ] / E[ {E[X_i | X] / X} 1_{X>x} ].

        The denominator equals total weights. It is the line-i recovery weighted layer effectiveness. It equals
        alpha_i(x) S(x).

        Now we have

        E[X_i(a)] = int_0^infty gamma_{a,i}(x) alpha_i(x) S(x) dx

        Note that you need upper and lower q's in aggs now too.

        Nov 2020: added arguments for plots; revised axes, separate plots by line

        :param a:     input a or p and kind as usual
        :param p:     asset level percentile
        :param kind:  lower or upper
        :param compute_stand_alone: add stand-alone evaluation of gamma
        :param axs:   enough axes; only plot if not None
        :param plot_mode: return or linear scale for y axis
        :return:
        """

        if a is None:
            assert p is not None
            a = self.q(p, kind)
        else:
            p = self.cdf(a)
            ql = self.q(p, 'lower')
            qu = self.q(p, 'upper')
            logger.log(WL,
                f'Input a={a} to gamma; computed p={p:.8g}, lower and upper quantiles are {ql:.8g} and {qu:.8g}')

        # alter in place or return a copy? For now return a copy...
        temp = self.density_df.filter(regex='^(p_|e1xi_1gta_|exi_xgta_|exi_xeqa_|'
                                    'exeqa_)[a-zA-Z]|^S$|^loss$').copy()

        # var dictionaries
        var_dict = self.var_dict(p, kind, 'total')
        extreme_var_dict = self.var_dict(1 - (1 - p) / 100, kind, 'total')

        # rate of payment factor
        min_xa = np.minimum(temp.loss, a) / temp.loss
        temp['min_xa_x'] = min_xa

        # total is a special case
        ln = 'total'
        gam_name = f'gamma_{self.name}_{ln}'
        # unconditional version avoid divide and multiply by a small number
        # exeqa is closest to the raw output...
        temp[f'exi_x1gta_{ln}'] = (temp[f'loss'] * temp.p_total / temp.loss).shift(-1)[::-1].cumsum() * self.bs
        # equals this temp.S?!
        s_ = temp.p_total.shift(-1)[::-1].cumsum() * self.bs
        # print(s_[:-1:-1], temp[f'exi_x1gta_{ln}'].iloc[:-2], temp.S.iloc[:-1] * self.bs)
        t1, t2 = np.allclose(s_[:-1:-1], temp[f'exi_x1gta_{ln}'].iloc[-1]), \
                 np.allclose(temp[f'exi_x1gta_{ln}'].iloc[:-1], temp.S.iloc[:-1] * self.bs)
        logger.debug(f'TEMP: the following should be close: {t1}, {t2} (expect True/True)')

        #                               this V 1.0 is exi_x for total
        temp[gam_name] = (min_xa * 1.0 * temp.p_total).shift(-1)[::-1].cumsum() / \
                         (temp[f'exi_x1gta_{ln}']) * self.bs

        for ln in self.line_names:
            # sa = stand alone; need to figure the sa capital from the agg objects hence var_dict
            if axs is not None or compute_stand_alone:
                a_l = var_dict[ln]
                a_l_ = a_l - self.bs
                xinv = temp[f'e1xi_1gta_{ln}']
                gam_name = f'gamma_{ln}_sa'
                s = 1 - temp[f'p_{ln}'].cumsum()
                temp[f'S_{ln}'] = s
                temp[gam_name] = 0
                temp.loc[0:a_l_, gam_name] = (s[0:a_l_] - s[a_l] + a_l * xinv[a_l]) / s[0:a_l_]
                temp.loc[a_l:, gam_name] = a_l * xinv[a_l:] / s[a_l:]
                temp[gam_name] = temp[gam_name].shift(1)
                # method unreliable in extreme right tail, but now changed plotting, not an issue
                # temp.loc[extreme_var_dict[ln]:, gam_name] = np.nan

            # actual computation for l within portfolio allowing for correlations
            gam_name = f'gamma_{self.name}_{ln}'
            # unconditional version avoid divide and multiply by a small number
            # exeqa is closest to the raw output...
            temp[f'exi_x1gta_{ln}'] = (temp[f'exeqa_{ln}'] * temp.p_total / temp.loss).shift(-1)[::-1].cumsum() * self.bs
            temp[gam_name] = (min_xa * temp[f'exi_xeqa_{ln}'] * temp.p_total).shift(-1)[::-1].cumsum() / \
                             (temp[f'exi_x1gta_{ln}']) * self.bs

        if axs is not None:
            axi = iter(axs.flat)
            nr, nc = axs.shape
            v = self.var_dict(.996, 'lower', 'total')
            ef = EngFormatter(3, True)

            if plot_mode == 'return':
                def transformer(x):
                    # transform on y scale
                    return np.where(x==0, np.nan, 1.0 / (1 - x))
                yscale = 'log'
            else:
                def transformer(x):
                    # transform on y scale
                    return x
                yscale = 'linear'

            # 1/s is re-used
            s1 = 1 / temp.S
            for i, ln in enumerate(self.line_names_ex):
                r, c = i // nc, i % nc
                ax = next(axi)
                if ln != 'total':
                    # line 1/s
                    ls1 = 1/temp[f'S_{ln}']
                    ax.plot(ls1, transformer(temp[f'gamma_{ln}_sa']), c='C1', label=f'SA {ln}')
                    ax.plot(s1, transformer(temp[f'gamma_{self.name}_total']), lw=1, c='C7', label='total')
                    c = 'C1'
                else:
                    ls1 = s1
                    c = 'C0'
                ax.plot(s1, transformer(temp[f'gamma_{self.name}_{ln}']), c='C0', label=f'Pooled {ln}')
                temp['WORST'] = np.maximum(0, 1 - (1 - p) * ls1)
                temp['BEST'] = 1
                temp.loc[v[ln]:, 'BEST'] = v[ln] / temp.loc[v[ln]:, 'loss']
                ax.fill_between(ls1, transformer(temp.WORST), transformer(temp.BEST), lw=.5, ls='--', alpha=.1,
                                color=c, label="Possible range")
                ax.plot(ls1, transformer(temp.BEST), lw=.5, ls='--', alpha=1, c=c)
                ax.plot(ls1, transformer(temp.WORST), lw=.5, ls=':', alpha=1, c=c)
                ax.set(xlim=[1, 1e9], xscale='log', yscale=yscale, xlabel='Loss return period' if r==nr-1 else None,
                        ylabel='Coverage Effectiveness (RP)' if c==0 else None, title=f'{ln}, a={ef(v[ln]).strip()}')
                ax.axvline(1/(1-.996), ls='--', c='C7', lw=.5, label='Capital p')
                ll = ticker.LogLocator(10, numticks=10)
                ax.xaxis.set_major_locator(ll)
                lm = ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks = 10)
                ax.xaxis.set_minor_locator(lm)
                ax.grid(lw=.25)
                ax.legend(loc='upper right')
            # tidy up axes
            try:
                while 1:
                    ax.figure.delaxes(next(axi))
            except StopIteration:
                pass
        temp.drop(columns=['BEST', 'WORST'])
        return Answer(gamma_df=temp.sort_index(axis=1), base=self.name, assets=a, p=p, kind=kind)

    def price(self, p, g, *, allocation='lifted', view='ask', kind='var', efficient=True):
        """
        Price using regulatory and pricing g functions

        Compute E_price (X wedge E_reg(X) ) where E_price uses the pricing distortion and E_reg uses
        the regulatory distortion derived from p. p can be input as a probability level converted
        to assets using `kind`, a level of assets directly (snapped to index).

        Regulatory capital distortion is applied on unlimited basis.

        ``g`` is a dictionary that creates a distortion or the dictionary `{ 'name': distortion_name, 'lr'|'roe':}`.
        The function then calls calibrate distortion to figure the distortion.

        :param p: a distortion function spec or just a number; if >1 assets if <1 a prob converted to quantile
        :param g:  pricing distortion function or dictionary spec or dictionary with distortion name, and lr or roe.
        :param allocation: 'lifted' (default) or 'linear'
        :param view: bid or ask [NOT FULLY INTEGRATED... MUST PASS IN A DISTORTION]
        :param kind: var (default), upper var, tvar, epd; passed to `var_dict`
        :param efficient: for apply_distortion
        :return: PricingResult namedtuple with 'price', 'assets', 'reg_p', 'distortion', 'df'
        """

        assert allocation in ('lifted', 'linear'), "allocation must be 'lifted' or 'linear'"

        # figure regulatory assets; applied to unlimited losses
        if p > 1:
            a_reg = self.snap(p)
        else:
            vd = self.var_dict(p, kind, snap=True)
            a_reg = vd['total']

        # relevant row for all statistics_df
        row = self.density_df.loc[a_reg]

        # figure pricing distortion
        if isinstance(g, Distortion):
            # just use it
            pass
        elif isinstance(g, dict):
            # spec as dict
            prem = 0
            if 'lr' in g:
                # given LR, figure premium
                prem = row['exa_total'] / g['lr']
            elif 'roe' in g:
                # given roe, figure premium
                roe = g['roe']
                delta = roe / (1 + roe)
                prem = row['exa_total'] + delta * (a_reg - row['exa_total'])
            if prem > 0:
                g = self.calibrate_distortion(name=g['name'], premium_target=prem, assets=a_reg)
            else:
                g = Distortion(**g)
        else:
            raise ValueError(f'Inadmissible type {type(g)} passed to price. Expected Distortion or dict.')

        if allocation == 'lifted':
            ans_ad = self.apply_distortion(g, view=view, create_augmented=False, efficient=efficient)
            if a_reg in ans_ad.augmented_df.index:
                aug_row = ans_ad.augmented_df.loc[a_reg]
            else:
                logger.warning('Regulatory assets not in augmented_df. Using last.')
                aug_row = ans_ad.augmented_df.iloc[-1]

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
            df['ROE'] = df.M / df.Q
            price = df.loc['total', 'P']
            reg_p = self.cdf(a_reg)

        elif allocation == 'linear':
            raise NotImplementedError('linear allocation not implemented yet')



        PricingResult = namedtuple('PricingResult', ['price', 'assets', 'reg_p', 'distortion', 'df'])
        ans = PricingResult(price, a_reg, reg_p, g, df)

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

    def analyze_distortions(self, a=0, p=0, kind='lower',  efficient=True,
                            augmented_dfs=None, regex='', add_comps=True):
        """
        Run analyze_distortion on self.dists

        :param a:
        :param p: the percentile of capital that the distortions are calibrated to
        :param kind: var, upper var, tvar, epd
        :param efficient:
        :param augmented_dfs: input pre-computed augmented_dfs (distortions applied)
        :param regex: apply only distortion names matching regex
        :param add_comps: add traditional pricing comps to the answer
        :return:

        """
        import re

        a, p = self.set_a_p(a, p)
        dfs = []
        ans = Answer()
        if augmented_dfs is None:
            augmented_dfs = {}
        ans['augmented_dfs'] = augmented_dfs
        for k, d in self.dists.items():
            if regex == '' or re.match(regex, k):
                if augmented_dfs is not None and k in augmented_dfs:
                    use_self = True
                    self.augmented_df = augmented_dfs[k]
                    logger.info(f'Found {k} in provided augmented_dfs...')
                else:
                    use_self = False
                    logger.info(f'Running distortion {d} through analyze_distortion, p={p}...')
                # first distortion...add the comps...these are same for all dists
                ad_ans = self.analyze_distortion(d, p=p, kind=kind, add_comps=(len(dfs) == 0) and add_comps,
                                                 efficient=efficient, use_self=use_self)
                dfs.append(ad_ans.exhibit)
                ans[f'{k}_exhibit'] = ad_ans.exhibit
                ans[f'{k}_pricing'] = ad_ans.pricing
                if not efficient and k not in augmented_dfs:
                    # remember, augmented_dfs is part of ans
                    augmented_dfs[k] = ad_ans.augmented_df
        ans['comp_df'] = pd.concat(dfs).sort_index()
        return ans

    def analyze_distortion(self, dname, dshape=None, dr0=.025, ddf=5.5, LR=None, ROE=None,
                           p=None, kind='lower', A=None, use_self=False, plot=False,
                           a_max_p=1-1e-8, add_comps=True, efficient=True):
        """

        Graphic and summary DataFrame for one distortion showing results that vary by asset level.
        such as increasing or decreasing cumulative premium.

        Characterized by the need to know an asset level, vs. apply_distortion that produced
        values for all asset levels.

        Returns DataFrame with values upto the input asset level...differentiates from apply_distortion
        graphics that cover the full range.

        analyze_pricing will then zoom in and only look at one asset level for micro-dynamics...

        Logic of arguments:
        ::

            if data_in == 'self' use self.augmented_df; this implies a distortion self.distortion

            else need to build the distortion and apply it
                if dname is a distortion use it
                else built one calibrated to input data

            LR/ROE/a/p:
                if p then a=q(p, kind) else p = MESSY
                if LR then P and ROE; if ROE then Q to P to LR
                these are used to calibrate distortion

            A newly made distortion is run through apply_distortion with no plot

        Logic to determine assets similar to calibrate_distortions.

        Can pass in a pre-calibrated distortion in dname

        Must pass LR or ROE to determine profit

        Must pass p or A to determine assets

        Output is an `Answer` class object containing
        ::

                Answer(augmented_df=deets, trinity_df=df, distortion=dist, fig1=f1 if plot else None,
                      fig2=f2 if plot else None, pricing=pricing, exhibit=exhibit, roe_compare=exhibit2,
                      audit_df=audit_df)

        Originally `example_factory`.


        example_factory_exhibits included:

        do the work to extract the pricing, exhibit and exhibit 2 DataFrames from deets
        Can also accept an ans object with an augmented_df element (how to call from outside)
        POINT: re-run exhibits at different p/a thresholds without recalibrating
        add relevant items to audit_df
        a = q(p) if both given; if not both given derived as usual

        Figures show

        :param dname: name of distortion
        :param dshape:  if input use dshape and dr0 to make the distortion
        :param dr0:
        :param ddf:  r0 and df params for distortion
        :param LR: otherwise use loss ratio and p or a loss ratio
        :param ROE:
        :param p: p value to determine capital.
        :param kind: type of VaR, upper or lower
        :param A:
        :param use_self:  if true use self.augmented and self.distortion...else recompute
        :param plot:
        :param a_max_p: percentile to use to set the right hand end of plots
        :param add_comps: add old-fashioned method comparables (included = True as default to make backwards comp.)
        :param efficient:
        :return: various dataframes in an Answer class object

        """

        # setup: figure what distortion to use, apply if necessary
        if use_self:
            # augmented_df was called deets before, FYI
            augmented_df = self.augmented_df
            if type(dname) == str:
                dist = self.dists[dname]
            elif isinstance(dname, Distortion):
                dist = dname
            else:
                raise ValueError(f'Unexpected dname={dname} passed to analyze_distortion')
            a_cal = self.q(p, kind)
            exa = self.density_df.loc[a_cal, 'exa_total']
            exag = augmented_df.loc[a_cal, 'exag_total']
            K = a_cal - exag
            LR = exa / exag
            ROE = (exag - exa) / K
        else:
            # figure assets a_cal (for calibration) from p or A
            if p:
                # have p
                a_cal = self.q(p, kind)
                exa = self.density_df.loc[a_cal, 'exa_total']
            else:
                # must have A...must be in index
                assert A is not None
                p = self.cdf(A)
                a_cal = self.q(p)
                exa = self.density_df.loc[a_cal, 'exa_total']
                if a_cal != A:
                    logger.info(f'a_cal:=q(p)={a_cal} is not equal to A={A} at p={p}')

            if dshape or isinstance(dname, Distortion):
                # specified distortion, fill in
                if isinstance(dname, Distortion):
                    dist = dname
                else:
                    dist = Distortion(dname, dshape, dr0, ddf)
                _x = self.apply_distortion(dist, create_augmented=False, efficient=efficient)
                augmented_df = _x.augmented_df
                exa = self.density_df.loc[a_cal, 'exa_total']
                exag = augmented_df.loc[a_cal, 'exag_total']
                profit = exag - exa
                K = a_cal - exag
                ROE = profit / K
                LR = exa / exag
            else:
                # figure distortion from LR or ROE and apply
                if LR is None:
                    assert ROE is not None
                    delta = ROE / (1 + ROE)
                    nu = 1 - delta
                    exag = nu * exa + delta * a_cal
                    LR = exa / exag
                else:
                    exag = exa / LR

                profit = exag - exa
                K = a_cal - exag
                ROE = profit / K
                # was wasteful
                # cd = self.calibrate_distortions(LRs=[LR], As=[a_cal], r0=dr0, df=ddf)
                dist = self.calibrate_distortion(dname, r0=dr0, df=ddf, roe=ROE, assets=a_cal)
                _x = self.apply_distortion(dist, create_augmented=False, efficient=efficient)
                augmented_df = _x.augmented_df

        # other helpful audit values
        # keeps track of details of calc for Answer
        audit_df = pd.DataFrame(dict(stat=[p, LR, ROE, a_cal, K, dist.name, dist.shape]),
                                index=['p', 'LR', "ROE", 'a_cal', 'K', 'dname', 'dshape'])
        audit_df.loc['a'] = a_cal
        audit_df.loc['kind'] = kind

        # make the pricing summary DataFrame and exhibit: this just contains info about dist
        # in non-efficient the renamer and existing columns collide and you get duplicates
        # transpose turns into rows...
        pricing = augmented_df.rename(columns=self.tm_renamer).loc[[a_cal]].T. \
                    filter(regex='^(T)\.(L|P|M|Q)_', axis=0).copy()
        pricing = pricing.loc[~pricing.index.duplicated()]
        # put in exact Q rather than add up the parts...more accurate
        pricing.at['T.Q_total', a_cal] = a_cal - exag
        # TODO EVIL! this reorders the lines and so messes up when port has lines not in alpha order
        pricing.index = pricing.index.str.split('_|\.', expand=True)
        pricing = pricing.sort_index()
        pricing = pd.concat((pricing,
                             pricing.xs(('T','P'), drop_level=False).rename(index={'P': 'PQ'}) / pricing.xs(('T', 'Q')).to_numpy(),
                             pricing.xs(('T', 'L'), drop_level=False).rename(index={'L': 'LR'}) / pricing.xs(('T', 'P')).to_numpy(),
                             pricing.xs(('T', 'M'), drop_level=False).rename(index={'M': 'ROE'}) / pricing.xs(('T', 'Q')).to_numpy()
                            ))
        pricing.index.set_names(['Method', 'stat', 'line'], inplace=True)
        pricing = pricing.sort_index()
        # make nicer exhibit
        exhibit = pricing.unstack(2).copy()
        exhibit.columns = exhibit.columns.droplevel(level=0)
        exhibit.loc[('T', 'a'), :] = exhibit.loc[('T', 'P'), :] + exhibit.loc[('T', 'Q'), :]
        exhibit.loc[('T', 'a'), :] = exhibit.loc[('T', 'a'), :] * a_cal / exhibit.at[('T', 'a'), 'total']
        ans = Answer(augmented_df=augmented_df, distortion=dist, audit_df=audit_df, pricing=pricing, exhibit=exhibit)
        if add_comps:
            logger.debug('Adding comps...')
            ans = self.analyze_distortion_add_comps(ans, a_cal, p, kind, ROE)
        if plot:
            ans = self.analyze_distortion_plots(ans, dist, a_cal, p, self.q(a_max_p), ROE, LR)

        # ans['exhibit'] = ans.exhibit.swaplevel(0,1).sort_index()
        ans['exhibit'] = ans.exhibit.rename(index={'T': f'Dist {dist.name}'}).sort_index()
        return ans

    def analyze_distortion_add_comps(self, ans, a_cal, p, kind, ROE):
        """
        make exhibit with comparison to old-fashioned methods: equal risk var/tvar, scaled var/tvar, stand-alone
        var/tvar, merton perold, co-TVaR. Not all of these methods is additive.

        covar method = proportion of variance (additive)

        Other methods could be added, e.g. a volatility method?

        **Note on calculation**

        Each method computes allocated assets a_i (which it calls Q_i) = Li + Mi + Qi
        All methods are constant ROEs for capital
        We have Li in exhibit. Hence:

                L = Li
                P = (Li + ROE ai) / (1 + ROE) = v Li + d ai
                Q = a - P
                M = P - L
                ratios

        In most cases, ai is computed directly, e.g. as a scaled proportion of total assets etc.


        The covariance method is slightly different.

                Mi = vi M, vi = Cov(Xi, X) / Var(X)
                Pi = Li + Mi
                Qi = Mi / ROE
                ai = Pi + Qi

        and sum ai = sum Li + sum Mi + sum Qi = L + M + M/ROE = L + M + Q = a as required. To fit it in the same
        scheme as all other methods we compute qi = Li + Mi + Qi = Li + vi M + vi M / ROE = li + viM(1+1/ROE)
        = Li + vi M/d, d=ROE/(1+ROE)

        :param ans:  answer containing dist and augmented_df elements
        :param a_cal:
        :param p:
        :param kind:
        :param LR:
        :param ROE:
        :return: ans Answer object with updated elements
        """

        exhibit = ans.exhibit.copy()
        # display(exhibit)
        total_margin = exhibit.at[('T', 'M'), 'total']
        logger.debug(f'Total margin = {total_margin}')
        # shut the style police up:
        done = []
        # some reasonable traditional metrics
        # tvar threshold giving the same assets as p on VaR
        logger.debug(f'In analyze_distortion_comps p={p} and a_cal={a_cal}')
        try:
            p_t = self.tvar_threshold(p, kind)
        except (ZeroDivisionError, ValueError) as e:
            logger.warning(f'Error computing p_t threshold for VaR at p={p}')
            logger.warning(str(e))
            p_t = np.nan
        try:
            pv, pt = self.equal_risk_var_tvar(p, p_t)
        except (ZeroDivisionError, ValueError) as e:
            logger.warning(f'Error computing pv, pt equal_risk_var_tvar for VaR, p={p}, kind={kind}, p_t={p_t}')
            logger.warning(str(e))
            pv = np.nan
            pt = np.nan
        try:
            pe = self.assets_2_epd[('total', 0)](a_cal)
            p_e = self.equal_risk_epd(a_cal)
        except Exception as e:
            logger.warning(f'Error {e} calibrating EPDs')
            pe = p_e = 1 - p
        try:
            temp = exhibit.loc[('T', 'L'), :]
            exhibit.loc[('EL', 'Q'), :] = temp / temp['total'] * a_cal
            done.append('EL')
            exhibit.loc[('VaR', 'Q'), self.line_names_ex] = [float(a.q(p, kind)) for a in self] + [self.q(p, kind)]
            done.append('var')
            exhibit.loc[('TVaR', 'Q'), self.line_names_ex] = [float(a.tvar(p_t)) for a in self] + [self.tvar(p_t)]
            done.append('tvar')
            # print(done)
            exhibit.loc[('ScaledVaR', 'Q'), :] = exhibit.loc[('VaR', 'Q'), :]
            exhibit.loc[('ScaledTVaR', 'Q'), :] = exhibit.loc[('TVaR', 'Q'), :]
            exhibit.loc[('ScaledVaR', 'Q'), 'total'] = 0
            exhibit.loc[('ScaledTVaR', 'Q'), 'total'] = 0
            sumvar = exhibit.loc[('ScaledVaR', 'Q'), :].sum()
            sumtvar = exhibit.loc[('ScaledTVaR', 'Q'), :].sum()
            exhibit.loc[('ScaledVaR', 'Q'), :] = \
                exhibit.loc[('ScaledVaR', 'Q'), :] * exhibit.at[('VaR', 'Q'), 'total'] / sumvar
            exhibit.loc[('ScaledTVaR', 'Q'), :] = \
                exhibit.loc[('ScaledTVaR', 'Q'), :] * exhibit.at[('TVaR', 'Q'), 'total'] / sumtvar
            exhibit.at[('ScaledVaR', 'Q'), 'total'] = exhibit.at[('VaR', 'Q'), 'total']
            exhibit.at[('ScaledTVaR', 'Q'), 'total'] = exhibit.at[('TVaR', 'Q'), 'total']
            # these are NOT additive
            exhibit.at[('VaR', 'Q'), 'total'] = sumvar
            exhibit.at[('TVaR', 'Q'), 'total'] = sumtvar
            exhibit.loc[('EqRiskVaR', 'Q'), self.line_names_ex] = [float(a.q(pv, kind)) for a in self] + [self.q(p)]
            done.append('eqvar')
            # print(done)
            exhibit.loc[('EqRiskTVaR', 'Q'), self.line_names_ex] = [float(a.tvar(pt)) for a in self] + [self.tvar(p_t)]
            done.append('eqtvar')
            # print(done)
            exhibit.loc[('MerPer', 'Q'), self.line_names_ex] = self.merton_perold(p)
            done.append('merper')
            # print(done)
            # co-tvar at threshold to match capital
            exhibit.loc[('coTVaR', 'Q'), self.line_names_ex] = self.cotvar(p_t)
            done.append('cotvar')
            # print(done)
            # epd and scaled epd methods at 1-p threshold
            vd = self.var_dict(pe, kind='epd', total='total')
            logger.debug(f'Computing EPD at {pe:.3%} threshold, total = {vd["total"]}')
            exhibit.loc[('EPD', 'Q'), vd.keys()] = vd.values()
            exhibit.loc[('EPD', 'Q'), 'total'] = 0.
            exhibit.loc[('ScaledEPD', 'Q'), :] = exhibit.loc[('EPD', 'Q'), :] * vd['total'] / \
                                                  exhibit.loc[('EPD', 'Q')].sum()
            exhibit.loc[('ScaledEPD', 'Q'), 'total'] = vd['total']
            exhibit.loc[('EPD', 'Q'), 'total'] = exhibit.loc[('EPD', 'Q')].iloc[:-1].sum()
            exhibit.loc[('EqRiskEPD', 'Q'), self.line_names_ex] = [float( self.epd_2_assets[(l, 0)](p_e) )
                                                    for l in self.line_names] + \
                                                 [float(self.epd_2_assets[('total',0)](pe))]
            done.append('epd')
            # print(done)
            # covariance = percent of variance allocation
            # qi = Li + Mi + Qi = Li + vi M + vi M / ROE = li + viM(1+1/ROE) = Li + vi M/d, d=ROE/(1+ROE)
            d = ROE / (1 + ROE)
            vars_ = self.audit_df['EmpEX2'] - self.audit_df['EmpEX1']**2
            vars_ = vars_ / vars_.iloc[-1]
            exhibit.loc[('covar', 'Q'), :] = exhibit.loc[('T', 'L'), :] + vars_ * total_margin / d
            done.append('covar')
        except Exception as e:
            logger.warning('Some general out of bounds error on VaRs and TVaRs, setting all equal to zero. '
                           f'Last completed steps {done[-1] if len(done) else "none"}, '
                           'out of var, tvar, eqvar, eqtvar merper. ')
            logger.warning(f'The built in warning is type {type(e)} saying {e}')
            for c in ['VaR', 'TVaR', 'ScaledVaR', 'ScaledTVaR', 'EqRiskVaR', 'EqRiskTVaR', 'MerPer', 'coTVaR']:
                if c.lower() in done:
                    # these numbers have been computed
                    pass
                else:
                    # put in placeholder
                    exhibit.loc[(c, 'Q'), :] = np.nan

        exhibit = exhibit.sort_index()
        # subtract the premium to get the actual capital
        try:
            for m in ['EL', 'VaR', 'TVaR', 'ScaledVaR', 'ScaledTVaR', 'EqRiskVaR', 'EqRiskTVaR', 'MerPer', 'coTVaR',
                      'EPD', 'ScaledEPD', 'EqRiskEPD', 'covar']:
                # Q as computed above gives assets not equity...so adjust
                # usual calculus: P = (L + ra)/(1+r); Q = a-P, remember Q above is a (sorry)
                exhibit.loc[(m, 'L'), :] = exhibit.loc[('T', 'L'), :]
                exhibit.loc[(m, 'P'), :] = (exhibit.loc[('T', 'L'), :] + ROE * exhibit.loc[(m, 'Q'), :]) / (1 + ROE)
                exhibit.loc[(m, 'Q'), :] -= exhibit.loc[(m, 'P'), :].values
                exhibit.loc[(m, 'M'), :] = exhibit.loc[(m, 'P'), :] - exhibit.loc[('T', 'L'), :].values
                exhibit.loc[(m, 'LR'), :] = exhibit.loc[('T', 'L'), :] / exhibit.loc[(m, 'P'), :]
                exhibit.loc[(m, 'ROE'), :] = exhibit.loc[(m, 'M'), :] / exhibit.loc[(m, 'Q'), :]
                exhibit.loc[(m, 'PQ'), :] = exhibit.loc[(m, 'P'), :] / exhibit.loc[(m, 'Q'), :]
                exhibit.loc[(m, 'a'), :] = exhibit.loc[(m, 'P'), :] + exhibit.loc[(m, 'Q'), :]
        except Exception as e:
            logger.error(f'Exception {e} creating LR, P, Q, ROE, or PQ')
        # print(ans.distortion.name)
        # display(exhibit)
        ans.audit_df.loc['TVaR@'] = p_t
        ans.audit_df.loc['erVaR'] = pv
        ans.audit_df.loc['erTVaR'] = pt
        ans.audit_df.loc['EPD@'] = pe
        ans.audit_df.loc['erEPD'] = p_e
        ans.update(exhibit=exhibit)
        return ans

    def analyze_distortion_plots(self, ans, dist, a_cal, p, a_max, ROE, LR):
        """
        Create plots from an analyze_distortion ans class
        note: this only looks at distortion related items...it doesn't use anything from the comps

        :param ans:
        :param dist:
        :param a_cal:
        :param p:
        :param a_max:
        :param ROE:
        :param LR:
        :return:
        """
        augmented_df = ans.augmented_df

        # top down stats: e.g. T.P[a_cal] - T.P and zero above a_cal
        # these are only going to apply to total...will not bother with by line
        # call this series V.P with v indicating a down arrow and a letter after and close to T
        # hack out space
        for nc in ['L', 'P', 'M', 'Q', 'LR', 'ROE', 'PQ']:
            augmented_df[f'V.{nc}_total'] = 0

        augmented_df.loc[0:a_cal, 'V.L_total'] = \
            augmented_df.at[a_cal, 'T.L_total'] - augmented_df.loc[0:a_cal, 'T.L_total']
        augmented_df.loc[0:a_cal, 'V.P_total'] = \
            augmented_df.at[a_cal, 'T.P_total'] - augmented_df.loc[0:a_cal, 'T.P_total']
        augmented_df.loc[0:a_cal, 'V.M_total'] = \
            augmented_df.at[a_cal, 'T.M_total'] - augmented_df.loc[0:a_cal, 'T.M_total']
        augmented_df.loc[0:a_cal, 'V.Q_total'] = \
            augmented_df.at[a_cal, 'T.Q_total'] - augmented_df.loc[0:a_cal, 'T.Q_total']
        augmented_df.loc[0:a_cal, 'V.LR_total'] = \
            augmented_df.loc[0:a_cal, 'V.L_total'] / augmented_df.loc[0:a_cal, 'V.P_total']
        augmented_df.loc[0:a_cal, 'V.PQ_total'] = \
            augmented_df.loc[0:a_cal, 'V.P_total'] / augmented_df.loc[0:a_cal, 'V.Q_total']
        augmented_df.loc[0:a_cal, 'V.ROE_total'] = \
            augmented_df.loc[0:a_cal, 'V.M_total'] / augmented_df.loc[0:a_cal, 'V.Q_total']

        # bottom up calc is already done: it is the T. series in augmented_df
        # marginal calc also done: M. series

        # f_6_part = f_trinity = f_8_part = f_distortion = f_close = None
        # plot the distortion
        f_distortion, ax = plt.subplots(1, 1)
        dist.plot(ax=ax)

        # six part up and down plot
        def tidy(a, y=True):
            """
            function to tidy up the graphics
            """
            n = 6
            a.set(xlabel='Assets')
            a.xaxis.set_major_locator(FixedLocator([a_cal]))
            ff = f'A={a_cal:,.0f}'
            a.xaxis.set_major_formatter(FixedFormatter([ff]))
            a.xaxis.set_minor_locator(MaxNLocator(n))
            a.xaxis.set_minor_formatter(StrMethodFormatter('{x:,.0f}'))
            if y:
                a.yaxis.set_major_locator(MaxNLocator(n))
                a.yaxis.set_minor_locator(AutoMinorLocator(4))
            # gridlines with various options
            # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
            a.grid(which='major', axis='x', c='cornflowerblue', alpha=1, linewidth=1)
            a.grid(which='major', axis='y', c='lightgrey', alpha=0.5, linewidth=1)
            a.grid(which='minor', axis='x', c='lightgrey', alpha=0.5, linewidth=1)
            a.grid(which='minor', axis='y', c='gainsboro', alpha=0.25, linewidth=0.5)
            # tick marks
            a.tick_params('x', which='major', labelsize=7, length=10, width=0.75, color='cornflowerblue',
                          direction='out')
            a.tick_params('y', which='major', labelsize=7, length=5, width=0.75, color='black',
                          direction='out')
            a.tick_params('both', which='minor', labelsize=7, length=2, width=0.5, color='black',
                          direction='out')

        # plots
        f_6_part, axs = plt.subplots(3, 2, figsize=(8, 10), sharex=True, constrained_layout=True)
        axi = iter(axs.flatten())

        # ONE
        ax = next(axi)
        # df[['Layer Loss', 'Layer Prem', 'Layer Capital']]
        augmented_df.filter(regex='^F|^gS|^S').rename(columns=self.renamer). \
            plot(xlim=[0, a_max], ylim=[-0.025, 1.025], logy=False, title='F, S, gS: marginal premium and loss',
                 ax=ax)
        tidy(ax)
        ax.legend(frameon=True, loc='center right')

        # TWO
        ax = next(axi)
        # df[['Layer Capital', 'Layer Margin']].plot(xlim=xlim, ax=ax)
        augmented_df.filter(regex='^M\.[QM]_total').rename(columns=self.renamer). \
            plot(xlim=[0, a_max], ylim=[-0.05, 1.05], logy=False, title='Marginal Capital and Margin', ax=ax)
        tidy(ax)
        ax.legend(frameon=True, loc='center right')

        # THREE
        ax = next(axi)
        # df[['Premium↓', 'Loss↓', 'Capital↓', 'Assets↓', 'Risk Margin↓']].plot(xlim=xlim, ax=ax)
        augmented_df.filter(regex='^V\.(L|P|Q|M)_total').rename(columns=self.renamer). \
            plot(xlim=[0, a_max], ylim=[0, a_max], logy=False, title=f'Decreasing LPMQ from {a_cal:.0f}',
                 ax=ax)
        (a_cal - augmented_df.loc[:a_cal, 'loss']).plot(ax=ax, label='Assets')
        tidy(ax)
        ax.legend(frameon=True, loc='upper right')

        # FOUR
        ax = next(axi)
        augmented_df.filter(regex='^(T|V|M)\.LR_total').rename(columns=self.renamer). \
            plot(xlim=[0, a_cal * 1.1], ylim=[-0.05, 1.05], ax=ax, title='Increasing, Decreasing and Marginal LRs')
        tidy(ax)
        ax.legend(frameon=True, loc='lower left')

        # FIVE
        ax = next(axi)
        augmented_df.filter(regex='^T\.(L|P|Q|M)_total|loss').rename(columns=self.renamer). \
            plot(xlim=[0, a_max], ylim=[0, a_max], logy=False, title=f'Increasing LPMQ to {a_cal:.0f}',
                 ax=ax)
        tidy(ax)
        ax.legend(frameon=True, loc='upper left')

        # SIX
        # could include leverage?
        ax = next(axi)
        augmented_df.filter(regex='^(M|T|V)\.ROE_(total)?$').rename(columns=self.renamer). \
            plot(xlim=[0, a_max],  # ylim=[-0.05, 1.05],
                 logy=False, title=f'Increasing, Decreasing and Marginal ROE to {a_cal:.0f}',
                 ax=ax)
        # df[['ROE↓', '*ROE↓', 'ROE↑', 'Marginal ROE', ]].plot(xlim=xlim, logy=False, ax=ax, ylim=ylim)
        # df[['ROE↓', 'ROE↑', 'Marginal ROE', 'P:S↓', 'P:S↑']].plot(xlim=xlim, logy=False, ax=a, ylim=[0,_])
        ax.plot([0, a_max], [ROE, ROE], ":", linewidth=2, alpha=0.75, label='Avg ROE')
        # print('plot 6 completed\n' * 6)
        try:
            tidy(ax)
            ax.legend(loc='upper right')

            title = f'{self.name} with {str(dist)} Distortion\nCalibrated to LR={LR:.3f} and p={p:.3f}, ' \
                    f'Assets={a_cal:,.1f}, ROE={ROE:.3f}'
            f_6_part.suptitle(title, fontsize='x-large')
        except Exception as e:
            logger.error(f'Formatting error in last plot...\n{e}\n...continuing')

        # trinity plots
        def tidy2(a, k, xloc=0.25):
            n = 4
            a.xaxis.set_major_locator(MultipleLocator(xloc))
            a.xaxis.set_minor_locator(AutoMinorLocator(4))
            a.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
            a.yaxis.set_major_locator(MaxNLocator(2 * n))
            a.yaxis.set_minor_locator(AutoMinorLocator(4))
            a.grid(which='major', axis='x', c='lightgrey', alpha=0.5, linewidth=1)
            a.grid(which='major', axis='y', c='lightgrey', alpha=0.5, linewidth=1)
            a.grid(which='minor', axis='x', c='gainsboro', alpha=0.25, linewidth=0.5)
            a.grid(which='minor', axis='y', c='gainsboro', alpha=0.25, linewidth=0.5)
            # tick marks
            a.tick_params('both', which='major', labelsize=7, length=4, width=0.75, color='black', direction='out')
            a.tick_params('both', which='minor', labelsize=7, length=2, width=0.5, color='black', direction='out')
            # line to show where capital lies
            a.plot([0, 1], [k, k], linewidth=1, c='black', label='Assets')

        plots_done = []
        try:
            f_trinity, axs = plt.subplots(1, 5, figsize=(8, 3), constrained_layout=True, sharey=True)
            axi = iter(axs.flatten())
            xr = [-0.05, 1.05]

            audit = augmented_df.loc[:a_max, :]

            # ONE
            ax = next(axi)
            ax.plot(audit.gS, audit.loss, label='M.P_total')
            ax.plot(audit.S, audit.loss, label='M.L_total')
            ax.set(xlim=xr, title='Marginal Prem & Loss')
            ax.set(xlabel='Loss = S = Pr(X>a)\nPrem = g(S)', ylabel="Assets, a")
            tidy2(ax, a_cal)
            ax.legend(loc="upper right", frameon=True, edgecolor=None)
            plots_done.append(1)

            # TWO
            ax = next(axi)
            m = audit.F - audit.gF
            ax.plot(m, audit.loss, linewidth=2, label='M')
            ax.set(xlim=-0.01, title='Marginal Margin', xlabel='M = g(S) - S')
            tidy2(ax, a_cal, m.max() * 1.05 / 4)
            plots_done.append(2)

            # THREE
            ax = next(axi)
            ax.plot(1 - audit.gS, audit.loss, label='Q')
            ax.set(xlim=xr, title='Marginal Equity')
            ax.set(xlabel='Q = 1 - g(S)')
            tidy2(ax, a_cal)
            plots_done.append(3)

            # FOUR
            ax = next(axi)
            temp = audit.loc[self.q(1e-5):, :]
            r = (temp.gS - temp.S) / (1 - temp.gS)
            ax.plot(r, temp.loss, linewidth=2, label='ROE')
            ax.set(xlim=-0.05, title='Layer ROE')
            ax.set(xlabel='ROE = M / Q')
            tidy2(ax, a_cal, r.max() * 1.05 / 4)
            plots_done.append(4)

            # FIVE
            ax = next(axi)
            ax.plot(audit.S / audit.gS, audit.loss)
            ax.set(xlim=xr, title='Layer LR')
            ax.set(xlabel='LR = S / g(S)')
            tidy2(ax, a_cal)
            plots_done.append(5)

        except Exception as e:
            logger.error(f'Plotting error in trinity plots\n{e}\nPlots done {plots_done}\n...continuing')

        #
        #
        #
        # from original example_factory_sublines
        try:
            temp = augmented_df.filter(regex='exi_xgtag?_(?!sum)|^S|^gS|^(M|T)\.').copy()
            renamer = self.renamer
            augmented_df.index.name = 'Assets a'
            temp.index.name = 'Assets a'

            f_8_part, axs = plt.subplots(4, 2, figsize=(8, 10), constrained_layout=True, squeeze=False)
            ax = iter(axs.flatten())

            # ONE
            a = (1 - augmented_df.filter(regex='p_').cumsum()).rename(columns=renamer).sort_index(1). \
                plot(ylim=[0, 1], xlim=[0, a_max], title='Survival functions', ax=next(ax))
            a.grid('b')

            # TWO
            a = augmented_df.filter(regex='exi_xgtag?').rename(columns=renamer).sort_index(1). \
                plot(ylim=[0, 1], xlim=[0, a_max], title=r'$\alpha=E[X_i/X | X>a],\beta=E_Q$ by Line', ax=next(ax))
            a.grid('b')

            # THREE total margins
            a = augmented_df.filter(regex=r'^T\.M').rename(columns=renamer).sort_index(1). \
                plot(xlim=[0, a_max], title='Total Margins by Line', ax=next(ax))
            a.grid('b')

            # FOUR marginal margins was dividing by bs end of first line
            # for some reason the last entry in M.M_total can be problematic.
            a = (augmented_df.filter(regex=r'^M\.M').rename(columns=renamer).sort_index(1).iloc[:-1, :].
                 plot(xlim=[0, a_max], title='Marginal Margins by Line', ax=next(ax)))
            a.grid('b')

            # FIVE
            a = augmented_df.filter(regex=r'^M\.Q|gF').rename(columns=renamer).sort_index(1). \
                plot(xlim=[0, a_max], title='Capital = 1-gS = gF', ax=next(ax))
            a.grid('b')
            for _ in a.lines:
                if _.get_label() == 'gF':
                    _.set(linewidth=5, alpha=0.3)
            # recreate legend because changed lines
            a.legend()

            # SIX see apply distortion, line 1890 ROE is in augmented_df
            a = augmented_df.filter(regex='^ROE$|exi_xeqa').rename(columns=renamer).sort_index(1). \
                plot(xlim=[0, a_max], title='M.ROE Total and $E[X_i/X | X=a]$ by line', ax=next(ax))
            a.grid('b')

            # SEVEN improve scale selection
            a = augmented_df.filter(regex='M\.LR').rename(columns=renamer).sort_index(1). \
                plot(ylim=[-.05, 1.5], xlim=[0, a_max], title='Marginal LR',
                     ax=next(ax))
            a.grid('b')

            # EIGHT
            a = augmented_df.filter(regex='T.LR_').rename(columns=renamer).sort_index(1). \
                plot(ylim=[-.05, 1.25], xlim=[0, a_max], title='Increasing Total LR by Line',
                     ax=next(ax))
            a.grid('b')
            a.legend(loc='center right')
        except Exception as e:
            logger.error('Error', e)

        #
        # close up of plot 2
        #
        bit = augmented_df.query(f'loss < {a_max}').filter(regex='exi_xgtag?_.*(?<!sum)$')
        f_close, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax = bit.rename(columns=renamer).plot(ylim=[-0.025, 1.025], ax=ax)
        ax.grid()
        nl = len(self.line_names)
        for i, l in enumerate(ax.lines[nl:]):
            ax.lines[i].set(linewidth=1, linestyle='--')
            l.set(color=ax.lines[i].get_color(), linewidth=2)
        ax.legend(loc='upper left')
        # slightly evil
        ans.update(fig_distortion=f_distortion, fig_six_up_down=f_6_part,
            fig_trinity=f_trinity, fig_eight=f_8_part, fig_close=f_close)

        return ans

    def analysis_priority(self, asset_spec, output='df'):
        """
        Create priority analysis report_ser.
        Can be called multiple times with different ``asset_specs``
        asset_spec either a float used as an epd percentage or a dictionary. Entering an epd percentage
        generates the dictionary

                base = {i: self.epd_2_assets[('not ' + i, 0)](asset_spec) for i in self.line_names}

        :param asset_spec: epd
        :param output: df = pandas data frame; html = nice report, markdown = raw markdown text
        :return:
        """

        ea = self.epd_2_assets
        ae = self.assets_2_epd

        if isinstance(asset_spec, dict):
            base = asset_spec
        else:
            if type(asset_spec) != float:
                raise ValueError("Input dictionary or float = epd target")
            base = {i: ea[('not ' + i, 0)](asset_spec) for i in self.line_names}

        if output == 'df':
            priority_analysis_df = pd.DataFrame(
                columns=['a', 'chg a', 'not_line epd sa @a', 'line epd @a 2pri', 'not_line epd eq pri',
                         'line epd eq pri', 'total epd'],
                index=pd.MultiIndex.from_arrays([[], []], names=['Line', 'Scenario']))
            for col in set(self.line_names).intersection(set(base.keys())):
                notcol = 'not ' + col
                a_base = base[col]
                a = a_base
                e0 = ae[(notcol, 0)](a_base)
                e = e0
                priority_analysis_df.loc[(col, 'base'), :] = (
                    a, a - a_base, e, ae[(col, 2)](a), ae[(notcol, 1)](a), ae[(col, 1)](a), ae[('total', 0)](a))

                a = ea[(col, 2)](e0)
                priority_analysis_df.loc[(col, '2pri line epd = not line sa'), :] = (
                    a, a - a_base, ae[(notcol, 0)](a), ae[(col, 2)](a), ae[(notcol, 1)](a), ae[(col, 1)](a),
                    ae[('total', 0)](a))

                a = ea[(col, 2)](priority_analysis_df.ix[(col, 'base'), 'line epd eq pri'])
                priority_analysis_df.loc[(col, 'thought buying (line 2pri epd = base not line eq pri epd'), :] = (
                    a, a - a_base, ae[(notcol, 0)](a), ae[(col, 2)](a), ae[(notcol, 1)](a), ae[(col, 1)](a),
                    ae[('total', 0)](a))

                a = ea[(notcol, 1)](e0)
                priority_analysis_df.loc[(col, 'fair to not line, not line eq pri epd = base sa epd'), :] = (
                    a, a - a_base, ae[(notcol, 0)](a), ae[(col, 2)](a), ae[(notcol, 1)](a), ae[(col, 1)](a),
                    ae[('total', 0)](a))

                a = ea[(col, 1)](e0)
                priority_analysis_df.loc[(col, 'line eq pri epd = base not line sa'), :] = (
                    a, a - a_base, ae[(notcol, 0)](a), ae[(col, 2)](a), ae[(notcol, 1)](a), ae[(col, 1)](a),
                    ae[('total', 0)](a))

                a = ea[('total', 0)](e0)
                priority_analysis_df.loc[(col, 'total epd = base sa not line epd'), :] = (
                    a, a - a_base, ae[(notcol, 0)](a), ae[(col, 2)](a), ae[(notcol, 1)](a), ae[(col, 1)](a),
                    ae[('total', 0)](a))

            priority_analysis_df['pct chg'] = priority_analysis_df['chg a'] / priority_analysis_df.a
            return priority_analysis_df

        # else HTML or markdown output
        ans = []
        for line in set(self.line_names).intersection(set(base.keys())):
            a = base[line]
            e = ae[(f'not {line}', 0)](a)
            a0 = float(ea[('total', 0)](e))
            eb0a0 = ae[(f'not {line}', 0)](a0)
            eba0 = ae[(f'not {line}', 1)](a0)
            e2a0 = ae[(line, 2)](a0)
            e1a0 = ae[(line, 1)](a0)
            e2 = ae[(line, 2)](a)
            e1 = float(ae[(line, 1)](a))
            a2 = float(ea[(line, 2)](e1))
            af = float(ea[(f'not {line}', 1)](e))
            af2 = float(ea[(line, 1)](e))
            a3 = float(ea[(line, 2)](e))
            a4 = float(ea[(f'not {line}', 1)](e))

            story = f"""
Consider adding **{line}** to the existing portfolio. The existing portfolio has capital {a:,.1f} and and epd of {e:.4g}.

<ul>
<li> If {line} is added as second priority to the existing lines with no increase in capital it has an epd of {e2:.4g}.
<li> If the regulator requires the overall epd be a constant then the firm must increase capital to {a0:,.1f} or by {(a0 / a - 1) * 100:.2f} percent.
    - At the higher capital {line} has an epd of {e2a0:.4g} as second priority and the existing lines have an epd of {eb0a0:.4g} as first priority.
    - The existing and {line} epds under equal priority are {eba0:.4g} and {e1a0:.4g}.
<li> If {line} *thought* it was added at equal priority it would have expected an epd of {e1:.4g}.
  In order to achieve this epd as second priority would require capital of {a2:,.1f}, an increase of {(a2 / a - 1) * 100:.2f} percent.
<li> In order for {line} to have an epd equal to the existing lines as second priority would require capital
  of {a3:,.1f}, and increase of {(a3 / a - 1) * 100:.2f} percent.
<li> In order for {line} to be added at equal priority and for the existing lines to have an unchanged epd requires capital of {af:,.1f}, an
  increase of {(af / a - 1) * 100:.2f} percent.
<li> In order for {line} to be added at equal priority and to have an epd equal to the existing line epd requires capital of {af2:,.1f}, an
  increase of {(af2 / a - 1) * 100:.2f} percent.
<li> In order for the existing lines to have an unchanged epd at equal priority requires capital of {a4:,.1f}, an increase of {(a4 / a - 1) * 100:.2f} percent.
<ul>
"""
            ans.append(story)
        ans = '\n'.join(ans)
        if output == 'html':
            display(HTML(ans))
        else:
            return ans

    def analysis_collateral(self, line, c, a, debug=False):
        """
        E(C(a,c)) expected value of line against not line with collateral c and assets a, c <= a

        :param line: line of business with collateral, analyzed against not line
        :param c: collateral, c <= a required; c=0 reproduces exa, c=a reproduces lev
        :param a: assets, assumed less than the max loss (i.e. within the square)
        :param debug:
        :return:
        """
        assert (c <= a)
        xs = self.density_df['loss'].values
        pline = self.density_df['p_' + line].values
        notline = self.density_df['ημ_' + line].values
        ans = []
        gt, incr, int1, int2, int3 = 0, 0, 0, 0, 0
        c1, c2, c3 = 0, 0, 0
        n_c = int(c / self.bs)
        n_max = len(xs)  # this works as a rhs array[0:n_max] is the whole array, array[n_max] is an error
        err_count = 0
        for loss in np.arange(a + self.bs, 2 * xs.max(), self.bs):
            n_loss = int(loss / self.bs)  # 0...loss INCLUSIVE
            c1 = c / a * loss
            n_c1 = min(n_max, int(c1 / self.bs))
            # need to adjust for trimming when loss > max loss
            # loss indexes...see notes in blue book
            la = max(0, n_loss - (n_max - 1))
            lc = min(n_loss, n_max - 1)
            lb = lc + 1
            if la == 0:
                ld = None
            else:
                ld = la - 1
            try:
                s1 = slice(la, max(la, min(lb, n_c)))
                s2 = slice(max(la, min(lb, n_c)), max(la, min(lb, n_c1)))
                s3 = slice(max(la, min(lb, n_c1)), lb)
                if ld is None:
                    # means go all the way to zero, do not have to worry about n_loss - n_c > 0 being smaller
                    s1r = slice(lc, min(lc, n_loss - n_c), -1)
                    s2r = slice(min(lc, n_loss - n_c), min(lc, n_loss - n_c1), -1)
                    s3r = slice(min(lc, n_loss - n_c1), ld, -1)
                else:
                    s1r = slice(lc, max(ld, min(lc, n_loss - n_c)), -1)
                    s2r = slice(max(ld, min(lc, n_loss - n_c)), max(ld, min(lc, n_loss - n_c1)), -1)
                    s3r = slice(max(ld, min(lc, n_loss - n_c1)), ld, -1)
                int1 = np.sum(xs[s1] * pline[s1] * notline[s1r])
                int2 = c * np.sum(pline[s2] * notline[s2r])
                int3 = a / loss * np.sum(xs[s3] * pline[s3] * notline[s3r])
                ptot = np.sum(pline[s3] * notline[s3r])
            except ValueError as e:
                logger.error(e)
                logger.error(f"Value error: loss={loss}, loss/b={loss / self.bs}, c1={c1}, c1/b={c1 / self.bs}")
                logger.error(f"n_loss {n_loss},  n_c {n_c}, n_c1 {n_c1}")
                logger.error(f'la={la}, lb={lb}, lc={lc}, ld={ld}')
                logger.error('ONE:', *map(len, [xs[s1], pline[s1], notline[s1r]]))
                logger.error('TWO:', *map(len, [pline[s2], notline[s2r]]))
                logger.error('THR:', *map(len, [xs[s3], pline[s3], notline[s3r]]))
                err_count += 1
                if err_count > 3:
                    break
            if n_loss < n_max:
                p = self.density_df.loc[loss, 'p_total']
            else:
                p = np.nan
            incr = (int1 + int2 + int3)
            gt += incr
            c1 += int1
            c2 += int2
            c3 += int3
            if debug:
                ans.append([loss, int1, int2, int3, int3 * loss / a / ptot, ptot, incr, c1, c2, c3, gt, p])
            if incr / gt < 1e-12:
                if debug:
                    logger.info(f'incremental change {incr / gt:12.6f}, breaking')
                break
        exlea = self.density_df.loc[a, 'exlea_' + line]
        exgta = self.density_df.loc[a, 'exgta_' + line]
        exix = self.density_df.loc[a, 'exi_xgta_' + line]
        exeqa = self.density_df.loc[a, 'exeqa_' + line]
        p_total = self.density_df.loc[a, 'p_total']
        F = self.density_df.loc[a, 'F']
        exa = self.density_df.loc[a, 'exa_' + line]
        lev = self.density_df.loc[a, 'lev_' + line]
        df = pd.DataFrame(
            [(line, a, c, p_total, F, gt, a * exix * (1 - F), exeqa, exlea, exgta, exix, exa, gt + exlea * F, lev)],
            columns=['line', 'a', 'c', 'p_total', 'F', 'gt', 'exa_delta', 'exeqa', 'exlea', 'exgta', 'exix', 'exa',
                     'ecac', 'lev'],
        )
        if debug:
            ans = pd.DataFrame(ans,
                               columns=['loss', 'int1', 'int2', 'int3', 'exeqa', 'ptot', 'incr', 'c1', 'c2', 'c3', 'gt',
                                        'log'])
            ans = ans.set_index('loss', drop=True)
            ans.index.name = 'loss'
        else:
            ans = None
        return df, ans

    def uat_differential(self, line):
        """
        Check the numerical and theoretical derivatives of exa agree for given line

        :param line:
        :return:
        """

        test = self.density_df[f'exa_{line}']
        dtest = np.gradient(test, test.index)
        dtest2 = self.density_df[f'exi_xgta_{line}'] * self.density_df.S

        ddtest = np.gradient(dtest)
        ddtest2 = -self.density_df[f'exeqa_{line}'] / self.density_df.loss * self.density_df.p_total

        f, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].plot(test.index, test, label=f'exa_{line}')

        axs[1].plot(test.index, dtest, label='numdiff')
        axs[1].plot(test.index, dtest2, label='xi_xgta S(x)')
        axs[1].legend()

        axs[2].plot(test.index, ddtest, label='numdiff')
        axs[2].plot(test.index, ddtest2, label='-EXi(a)/a')
        axs[2].legend()

    def uat(self, As=None, Ps=[0.98], LRs=[0.965], r0=0.03, num_plots=1, verbose=False):
        """
        Reconcile apply_distortion(s) with price and calibrate


        :type Ps: object
        :param As:   Asset levels
        :param Ps:   probability levels used to determine asset levels using quantile function
        :param LRs:  loss ratios used to determine profitability
        :param r0:   r0 level for distortions
        :param verbose: controls level of output
        :return:
        """

        # figure As
        if As is None:
            As = []
            for p in Ps:
                As.append(self.q(p))

        # 0. Calibrate
        params = self.calibrate_distortions(LRs=LRs, As=As, r0=r0)

        # 1. Apply and compare to calibration
        K = As[0]
        LR = LRs[0]
        idx = (K, LR)
        dd = Distortion.distortions_from_params(params, index=idx, r0=r0)
        if num_plots == 2:
            axiter = axiter_factory(None, len(dd))
        elif num_plots == 3:
            axiter = axiter_factory(None, 30)
        else:
            axiter = None
        table, stacked = self.apply_distortions(dd, As, axiter, num_plots)
        table['lr err'] = table['lr_total'] - LR

        # 2. Price and compare to calibration
        pdfs = []  # pricing data frmes
        for name in Distortion.available_distortions():
            pdf, _ = self.price(reg_g=K, pricing_g=dd[name])
            pdf['dist'] = name
            pdfs.append(pdf)
        p = pd.concat(pdfs)
        p['lr err'] = p['lr'] - LR

        # a from apply, p from price
        a = table.query(f' loss=={K} ')

        # easier tests
        # sum of parts = total
        logger.info(
            f'Portfolio.uat | {self.name} Sum of parts all close to total: '
            f'{np.allclose(a.exag_total, a.exag_sumparts)}')
        logger.info(
            f'Portfolio.uat | {self.name} Sum of parts vs total: '
            f'{np.sum(np.abs(a.exag_total - a.exag_sumparts)):15,.1f}')

        pp = p[['dist', 'exag']]
        pp = pp.pivot(columns='dist').T.loc['exag']
        aa = a.filter(regex='exa|method').set_index('method')

        test = pd.concat((aa, pp), axis=1, sort=True)
        for c in self.line_names_ex:
            test[f'err_{c}'] = test[c] / test[f'exag_{c}'] - 1
        test['err sum/total'] = test['exag_sumparts'] / test['exag_total'] - 1
        test = test[
            [f'{i}{j}' for j in self.line_names_ex for i in ['exag_', '', 'err_']] + ['exag_sumparts', 'err sum/total']]
        lr_err = pd.DataFrame({'applyLR': a.lr_total, 'method': a.method, 'target': LR, 'errs': a.lr_total - LR})
        lr_err = lr_err.reset_index(drop=False).set_index('method')
        lr_err = lr_err.rename(columns={'index': 'a'})
        test = pd.concat((test, lr_err), axis=1, sort=True)
        overall_test = (test.filter(regex='err').abs()).sum().sum()
        if verbose:
            html_title(f'Combined, overall error {overall_test:.3e}')  # (exag=apply)')
            display(test)

        if lr_err.errs.abs().max() > 1e-4:
            logger.error('Portfolio.uat | {self.name} UAT Loss Ratio Error {lr_err.errs.abs().max()}')

        if overall_test < 1e-7:
            logger.info(f'Portfolio.uat | {self.name} UAT All good, total error {overall_test:6.4e}')
        else:
            s = f'{self.name} UAT total error {overall_test:6.4e}'
            logger.error(f'Portfolio.uat | {s}')
            logger.error(f'Portfolio.uat | {s}')
            logger.error(f'Portfolio.uat | {s}')

        return a, p, test, params, dd, table, stacked

    @property
    def renamer(self):
        """
        write a sensible renamer for the columns to use thusly

        self.density_df.rename(columns=renamer)

        write a tex version separately
        Create once per item...assume lines etc. never change

        :return: dictionary that can be used to rename columns
        """
        if self._renamer is None:
            # make it
            # key : (before, after)
            # ημ needs to come last because of epd_nu etc.
            # keep lEV and EX(a) separate because otherwise index has non-unique values
            self._renamer = {}
            meta_namer = dict(p_=('', ' density'),
                              lev_=('LEV[', 'a]'),
                              exag_=('EQ[', '(a)]'),
                              exa_=('E[', '(a)]'),
                              exlea_=('E[', ' | X<=a]'),
                              exgta_=('E[', ' | X>a]'),
                              exeqa_=('E[', ' | X=a]'),
                              e1xi_1gta_=('E[1/', ' 1(X>a)]'),
                              exi_x_=('E[', '/X]'),
                              exi_xgta_sum=('Sum Xi/X gt', ''),
                              # exi_xgta_sum=('Sum of E[Xi/X|X>a]', ''),
                              exi_xeqa_sum=("Sum Xi/X eq", ''),
                              # exi_xeqa_sum=("Sum of E[Xi/X|X=a]", ''),
                              exi_xgta_=('α=E[', '/X | X>a]'),
                              exi_xeqa_=('E[', '/X | X=a]'),
                              exi_xlea_=('E[', '/X | X<=a]'),
                              epd_0_=('EPD(', ') stand alone'),
                              epd_1_=('EPD(', ') within X'),
                              epd_2_=('EPD(', ') second pri'),
                              e2pri_=('E[X', '(a) second pri]'),
                              ημ_=('All but ', ' density')
                              )

            for l in self.density_df.columns:
                if re.search('^ημ_', l):
                    # nu_line -> not line density
                    self._renamer[l] = re.sub('^ημ_([0-9A-Za-z\-_.,]+)', r'not \1 density', l)
                else:
                    l0 = l.replace('ημ_', 'not ')
                    for k, v in meta_namer.items():
                        d1 = l0.find(k)
                        if d1 >= 0:
                            d1 += len(k)
                            b, a = v
                            self._renamer[l] = f'{b}{l0[d1:]}{a}'.replace('total', 'X')
                            break

            # augmented df items
            for l in self.line_names_ex:
                self._renamer[f'exag_{l}'] = f'EQ[{l}(a)]'
                self._renamer[f'exi_xgtag_{l}'] = f'β=EQ[{l}/X | X>a]'
                self._renamer[f'exi_xleag_{l}'] = f'EQ[{l}/X | X<=a]'
                self._renamer[f'e1xi_1gta_{l}'] = f'E[{l}/X 1(X >a)]'
            self._renamer['exag_sumparts'] = 'Sum of EQ[Xi(a)]'

            # more post-processing items
            for pre, m1 in zip(['M', 'T'], ['Marginal', 'Total']):
                for post, m2 in zip(['L', 'P', 'LR', 'Q', 'ROE', "PQ", "M"],
                                    ['Loss', 'Premium', 'Loss Ratio', 'Equity', 'ROE', 'Leverage (P:S)', "Margin"]):
                    self._renamer[f'{pre}.{post}'] = f'{m1} {m2}'
            for line in self.line_names_ex:
                for pre, m1 in zip(['M', 'T'], ['Marginal', 'Total']):
                    for post, m2 in zip(['L', 'P', 'LR', 'Q', 'ROE', "PQ", "M"],
                                        ['Loss', 'Premium', 'Loss Ratio', 'Equity', 'ROE', 'Leverage (P:S)', "Margin"]):
                        self._renamer[f'{pre}.{post}_{line}'] = f'{m1} {m2} {line}'
            self._renamer['A'] = 'Assets'
            # self._renamer['exi/xgta'] = 'α=E[Xi/X | X > a]'
            # self._renamer['exi/xgtag'] = 'β=E_Q[Xi/X | X > a]'

            # gamma files
            for l in self.line_names:
                self._renamer[f'gamma_{l}_sa'] = f"γ {l} stand-alone"
                self._renamer[f'gamma_{self.name}_{l}'] = f"γ {l} part of {self.name}"
                self._renamer[f'p_{l}'] = f'{l} stand-alone density'
                self._renamer[f'S_{l}'] = f'{l} stand-alone survival'
            self._renamer['p_total'] = f'{self.name} total density'
            self._renamer['S_total'] = f'{self.name} total survival'
            self._renamer[f'gamma_{self.name}_total'] = f"γ {self.name} total"

        # for enhanced exhibits --- these are a bit specialized!
        self._renamer['mv'] = "$\\mathit{MVL}(a)$??"
        for orig in self.line_names_ex:
            l = self.line_renamer.get(orig, orig).replace('$', '')
            if orig == 'total':
                self._renamer[f'S'] = f"$S_{{{l}}}(a)$"
            else:
                self._renamer[f'S_{orig}'] = f"$S_{{{l}}}(a)$"
            self._renamer[f'lev_{orig}'] = f"$E[{l}\\wedge a]$"
            self._renamer[f'levg_{orig}'] = f"$\\rho({l}\\wedge a)$"
            self._renamer[f'exa_{orig}'] = f"$E[{l}(a)]$"
            self._renamer[f'exag_{orig}'] = f"$\\rho({l}\\subseteq X^c\\wedge a)$"
            self._renamer[f'ro_da_{orig}'] = "$\\Delta a_{ro}$"
            self._renamer[f'ro_dmv_{orig}'] = "$\\Delta \\mathit{MVL}_{ro}(a)$"
            self._renamer[f'ro_eva_{orig}'] = "$\\mathit{EGL}_{ro}(a)$"
            self._renamer[f'gc_da_{orig}'] = "$\\Delta a_{gc}$"
            self._renamer[f'gc_dmv_{orig}'] = "$\\Delta \\mathit{MVL}_{gc}(a)$"
            self._renamer[f'gc_eva_{orig}'] = "$\\mathit{EGL}_{gc}(a)$"

        return self._renamer

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

    @property
    def stat_renamer(self):
        return dict('')

    def set_a_p(self, a, p):
        """
        sort out arguments for assets and prob level and make them consistent
        neither => set defaults
        a only set p
        p only set a
        both do nothing

        """
        if a == 0 and  p== 0:
            p = 0.995
            a = self.q(p)
            # click to exact
            p = self.cdf(a)
        elif a:
            p = self.cdf(a)
            # snap to index
            a = self.q(p)
        elif p:
            a = self.q(p)
            # click to exact
            p = self.cdf(a)
        return a, float(p)

    @staticmethod
    def from_DataFrame(name, df):
        """
        create portfolio from pandas dataframe
        uses columns with appropriate names

        Can be fed the agg output of uw.write_test( agg_program )

        :param name:
        :param df:
        :return:
        """
        # ...and this is why we love pandas so much
        spec_list = [g.dropna(axis=1).to_dict(orient='records') for n, g in df.groupby('name')]
        spec_list = [i[0] for i in spec_list]
        return Portfolio(name, spec_list)

    @staticmethod
    def from_Excel(name, ffn, sheet_name, **kwargs):
        """
        read in from Excel

        works via a Pandas dataframe; kwargs passed through to pd.read_excel
        drops all blank columns (mostly for auditing purposes)
        delegates to from_dataFrame


        :param name:
        :param ffn: full file name, including path
        :param sheet_name:
        :param kwargs:
        :return:
        """
        df = pd.read_excel(ffn, sheet_name=sheet_name, **kwargs)
        df = df.dropna(axis=1, how='all')
        return Portfolio.from_DataFrame(name, df)

    @staticmethod
    def from_dict_of_aggs(prefix, agg_dict, sub_ports=None, uw=None, bs=0, log2=0,
                          padding=2, **kwargs):
        """
        Create a portfolio from any iterable with values aggregate code snippets

        e.g.  agg_dict = {label: agg_snippet }

        will create all the portfolios specified in subsets, or all if subsets=='all'

        labels for subports are concat of keys in agg_dict, so recommend you use A:, B: etc.
        as the snippet names.  Portfolio names are prefix_[concat element names]

        agg_snippet is line agg blah without the tab or newline

        :param prefix:
        :param agg_dict:
        :param sub_ports:
        :param bs, log2, padding, kwargs: passed through to update; update if bs * log2 > 0
        :return:
        """

        agg_names = list(agg_dict.keys())

        # holder for created portfolios
        ports = Answer()
        if sub_ports == 'all':
            sub_ports = subsets(agg_names)

        for sub_port in sub_ports:
            name = ''.join(sub_port)
            if prefix != '':
                name = f'{prefix}_{name}'
            pgm = f'port {name}\n'
            for l in agg_names:
                if l in sub_port:
                    pgm += f'\t{agg_dict[l]}\n'
            if uw:
                ports[name] = uw.write(pgm)
            else:
                ports[name] = pgm
            if uw and bs * log2 > 0:
                ports[name].update(log2=log2, bs=bs, padding=padding, **kwargs)

        return ports

    # enhanced portfolio methods (see description in class doc string)

    def premium_capital(self, a=0, p=0):
        """
        at a if given else p level of capital

        pricing story allows two asset levels...handle that with a concat

        was premium_capital_detail

        """
        a, p = self.set_a_p(a, p)

        if getattr(self, '_raw_premium_capital', None) is not None and self.last_a == a:
            # done already
            return

        # else recompute

        # story == run off
        # pricing report from adf
        dm = self.augmented_df.filter(regex=f'T.[MPQLROE]+.({self.line_name_pipe})').loc[[a]].T
        dm.index = dm.index.str.split('_', expand=True)
        self._raw_premium_capital = dm.unstack(1)
        self._raw_premium_capital = self._raw_premium_capital.droplevel(axis=1, level=0)  # .drop(index=['Assets'])
        self._raw_premium_capital.loc['T.A', :] = self._raw_premium_capital.loc['T.Q', :] + \
                                                  self._raw_premium_capital.loc['T.P', :]
        self._raw_premium_capital.index.name = 'Item'
        self.EX_premium_capital = self._raw_premium_capital. \
            rename(index=self.premium_capital_renamer, columns=self.line_renamer).sort_index()
        self.last_a = a

    def multi_premium_capital(self, As, keys=None):
        """
        concatenate multiple prem_capital exhibits

        """
        if keys is None:
            keys = [f'a={i:.1f}' for i in As]

        ans = []
        for a in As:
            self.premium_capital(a)
            ans.append(self.EX_premium_capital.copy())
        self.EX_multi_premium_capital = pd.concat(ans, axis=1, keys=keys, names=['Assets', "Line"])

    def accounting_economic_balance_sheet(self, a=0, p=0):
        """
        story version assumes line 0 = reserves and 1 = prospective....other than that identical

        usual a and p rules
        """

        # check for update
        self.premium_capital(a, p)

        aebs = pd.DataFrame(0.0, index=self.line_names_ex + ['Assets', 'Equity'],
                            columns=['Statutory', 'Objective', 'Market', 'Difference'])
        slc = slice(0, len(self.line_names_ex))
        aebs.iloc[slc, 0] = self.audit_df['Mean'].values
        aebs.iloc[slc, 1] = self._raw_premium_capital.loc['T.L']
        aebs.iloc[slc, 2] = self._raw_premium_capital.loc['T.P']
        aebs.loc['Assets', :] = self._raw_premium_capital.loc['T.A', 'total']
        aebs.loc['Equity', :] = aebs.loc['Assets'] - \
                                aebs.loc['total']
        aebs[
            'Difference'] = aebs.Market - aebs.Objective
        # put in accounting order
        aebs = aebs.iloc[
            [-2] + list(range(len(self.line_names) + 1)) + [-1]]
        aebs.index.name = 'Item'
        self.EX_accounting_economic_balance_sheet = aebs.rename(index=self.line_renamer)

    def make_all(self, p=0, a=0, As=None):
        """
        make all exhibits with sensible defaults
        if not entered, paid line is selected as the LAST line

        """
        a, p = self.set_a_p(a, p)

        # exhibits that require a distortion
        if self.distortion is not None:
            self.premium_capital(a=a, p=p)
            if As is not None:
                self.multi_premium_capital(As)
            self.accounting_economic_balance_sheet(a=a, p=p)

    def show_enhanced_exhibits(self, fmt='{:.5g}'):
        """
        show all the exhibits created by enhanced_portfolio methods
        """
        display(HTML(f'<h2>Exhibits for {self.name.replace("_", " ").title()} Portfolio</h2>'))
        for x in dir(self):
            if x[0:3] == 'EX_':
                ob = getattr(self, x)
                if isinstance(ob, pd.DataFrame):
                    # which they all will be...
                    display(HTML(f'<h3>{x[3:].replace("_", " ").title()}</h3>'))
                    display(ob.style.format(fmt, subset=ob.select_dtypes(np.number).columns))
                    display(HTML(f'<hr>'))

    def profit_segment_plot(self, ax, p, line_names, dist_name, colors=None, translations=None):
        """
        Lee diagram for each requested line on a stand-alone basis, loss and risk adj
        premium using the dist_name distortion. Optionally specify colors, using C{n}.
        Optionally specify translations applied to each line. Generally, this applies
        to shift the cat line up by E[non cat] losses to show it overlays the total.

        For a Portfolio with line names CAT and NC::

            port.gross.profit_segment_plot(ax, 0.99999, ['total', 'CAT', 'NC'],
                                'wang', [2,0,1])

        add translation to cat line::

            port.gross.profit_segment_plot(ax, 0.99999, ['total', 'CAT', 'NC'],
                                'wang', [2,0,1], [0, E[NC], 0])


        :param ax: axis on which to render
        :param p:  probability level to set upper and lower y axis limits (p and 1-p quantiles)
        :param line_names:
        :param dist_name:
        :param colors:
        :param translations:
        :return:
        """

        dist = self.dists[dist_name]
        if colors is None:
            colors = range(len(line_names))
        if translations is None:
            translations = [0] * len(line_names)
        for line, cn, translation in zip(line_names, colors, translations):
            c = f'C{cn}'
            f1 = self.density_df[f'p_{line}'].cumsum()
            idx = (f1 < p) * (f1 > 1.0 - p)
            f1 = f1[idx]
            gf = 1 - dist.g(1 - f1)
            x = self.density_df.loss[idx] + translation
            ax.plot(gf, x, '-', c=c, label=f'Risk Adj {line}' if translation == 0 else None)
            ax.plot(f1, x, '--', c=c, label=line if translation == 0 else None)
            if translation == 0:
                ax.fill_betweenx(x, gf, f1, color=c, alpha=0.5)
            else:
                ax.fill_betweenx(x, gf, f1, color=c, edgecolor='black', alpha=0.5)
        # if you plot a small line this needs to start at zer0!
        ax.set(ylim=[0, self.q(p)])
        ax.legend(loc='upper left')

    def natural_profit_segment_plot(self, ax, p, line_names, colors, translations):
        """
        Plot the natural allocations between 1-p and p th percentiles and
        optionally translate line(s).
        Works with augmented_df, no input distortion. User must ensure the
        correct distortion has been applied.

        :param ax:
        :param p:
        :param line_names:
        :param colors:
        :param translations:
        :return:
        """
        lw, up = self.q(1 - p), self.q(p)
        # common extract for all lines
        bit = self.augmented_df.query(f' {lw} <= loss <= {up} ')
        F = bit[f'F']
        gF = bit[f'gF']
        for line, cn, translation in zip(line_names, colors, translations):
            c = f'C{cn}'
            ser = bit[f'exeqa_{line}']
            ax.plot(F, ser, ls='dashed', c=c)
            ax.plot(gF, ser, c=c)
            if translation == 0:
                ax.fill_betweenx(ser + translation, gF, F, color=c, alpha=0.5, label=line)
            else:
                ax.fill_betweenx(ser, gF, F, color=c, alpha=0.5, label=line)
        # see comment above.
        ax.set(ylim=[0, up], title=self.distortion)
        ax.legend(loc='upper left')

    def density_sample(self, n=20, reg="loss|p_|exeqa_"):
        """
        sample of equally likely points from density_df with interesting columns
        reg - regex to select the columns
        """
        ps = np.linspace(0.001, 0.999, n)
        xs = [self.q(i) for i in ps]
        return self.density_df.filter(regex=reg).loc[xs, :].rename(columns=self.renamer)

    def biv_contour_plot(self, fig, ax, min_loss, max_loss, jump,
                         log=True, cmap='Greys', min_density=1e-15, levels=30, lines=None, linecolor='w',
                         colorbar=False, normalize=False, **kwargs):
        """
        Make contour plot of line A vs line B. Assumes port only has two lines.

        Works with an extract density_df.loc[np.arange(min_loss, max_loss, jump), densities]
        (i.e., jump is the stride). Jump = 100 * bs is not bad...just think about how big the outer product will get!

        :param fig:
        :param ax:
        :param min_loss:  the density for each line is sampled at min_loss:max_loss:jump
        :param max_loss:
        :param jump:
        :param log:
        :param cmap:
        :param min_density: smallest density to show on underlying log region; not used if log
        :param levels: number of contours or the actual contours if you like
        :param lines: iterable giving specific values of k to plot X+Y=k
        :param linecolor:
        :param colorbar:  show color bar
        :param normalize: if true replace Z with Z / sum(Z)
        :param kwargs: passed to contourf (e.g., use for corner_mask=False, vmin,vmax)
        :return:
        """
        
        # careful about origin when big prob of zero loss
        npts = np.arange(min_loss, max_loss, jump)
        ps = [f'p_{i}' for i in self.line_names]
        bit = self.density_df.loc[npts, ps]
        n = len(bit)
        Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
        if normalize:
            Z = Z / np.sum(Z)
        X, Y = np.meshgrid(bit.index, bit.index)

        if log:
            z = np.log10(Z)
            mask = np.zeros_like(z)
            mask[z == -np.inf] = True
            mz = np.ma.array(z, mask=mask)
            cp = ax.contourf(X, Y, mz, levels=levels, cmap=cmap, **kwargs)
            # cp = ax.contourf(X, Y, mz, levels=np.linspace(-12, 0, levels), cmap=cmap, **kwargs)
            if colorbar:
                cb = fig.colorbar(cp, fraction=.1, shrink=0.5, aspect=10)
                cb.set_label('Log10(Density)')
        else:
            mask = np.zeros_like(Z)
            mask[Z < min_density] = True
            mz = np.ma.array(Z, mask=mask)
            cp = ax.contourf(X, Y, mz, levels=levels, cmap=cmap, **kwargs)
            if colorbar:
                cb = fig.colorbar(cp)
                cb.set_label('Density')
        # put in X+Y=c lines
        if lines is None:
            lines = np.arange(max_loss / 4, 2 * max_loss, max_loss / 4)
        try:
            for x in lines:
                ax.plot([0, x], [x, 0], lw=.75, c=linecolor, label=f'Sum = {x:,.0f}')
        except:
            pass

        title = (f'Bivariate Log Density Contour Plot\n{self.name.replace("_", " ").title()}'
                 if log
                 else f'Bivariate Density Contour Plot\n{self.name.replace("_", " ")}')

        # previously limits set from min_loss, but that doesn't seem right
        ax.set(xlabel=f'Line {self.line_names[0]}',
               ylabel=f'Line {self.line_names[1]}',
               xlim=[-max_loss / 50, max_loss],
               ylim=[-max_loss / 50, max_loss],
               title=title,
               aspect=1)

        # for post-processing
        self.X = X
        self.Y = Y
        self.Z = Z

    def nice_program(self, wrap_col=90):
        """
        return wrapped version of port program
        :return:
        """
        return fill(self.program, wrap_col, subsequent_indent='\t\t', replace_whitespace=False)

    def short_renamer(self, prefix='', postfix=''):
        if prefix:
            prefix = prefix + '_'
        if postfix:
            postfix = '_' + postfix

        knobble = lambda x: 'Total' if x == 'total' else x

        return {f'{prefix}{i}{postfix}': knobble(i).title() for i in self.line_names_ex}

    def twelve_plot(self, fig, axs, p=0.999, p2=0.9999, xmax=0, ymax2=0, biv_log=True, legend_font=0,
                    contour_scale=10, sort_order=None, kind='two', cmap='viridis'):
        """
        Twelve-up plot for ASTIN paper and book, by rc index:

        Greys for grey color map

        11 density
        12 log density
        13 biv density plot

        21 kappa
        22 alpha (from alpha beta plot 4)
        23 beta (?with alpha)

        row 3 = line A, row 4 = line B from alpha beta four 2
         1 S, gS, aS, bgS

        32 margin
        33 shift margin
        42 cumul margin
        43 natural profit compare

        **Args**

        self = portfolio or enhanced portfolio object
        p control xlim of plots via quantile; used if xmax=0
        p2 controls ylim for 33 and 34: stand alone M and natural M; used if ymax2=0
        biv_log - bivariate plot on log scale
        legend_font - fine tune legend font size if necessary
        sort_order = plot sorts by column and then .iloc[:, sort_order], if None [1,2,0]

        from common_scripts.py

        """

        a11, a12, a13, a21, a22, a23, a31, a32, a33, a41, a42, a43 = axs.flat
        col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lss = ['solid', 'dashed', 'dotted', 'dashdot']

        if sort_order is None:
            sort_order = [1, 2, 0]  # range(len(self.line_names_ex))

        if xmax == 0:
            xmax = self.q(p)
        ymax = xmax

        # density and log density

        temp = self.density_df.filter(regex='p_').rename(columns=self.short_renamer('p')).sort_index(axis=1).loc[:xmax]
        temp = temp.iloc[:, sort_order]
        temp.index.name = 'Loss'
        l1 = temp.plot(ax=a11, lw=1)
        l2 = temp.plot(ax=a12, lw=1, logy=True)
        l1.lines[-1].set(linewidth=1.5)
        l2.lines[-1].set(linewidth=1.5)
        a11.set(title='Density')
        a12.set(title='Log density')
        a11.legend()
        a12.legend()

        # biv den plot
        if kind == 'two':
            # biv den plot
            # min_loss, max_loss, jump = 0, xmax, (2 ** (self.log2 - 8)) * self.bs
            xmax = self.snap(xmax)
            min_loss, max_loss, jump = 0, xmax, self.snap(xmax / 255)
            min_density = 1e-15
            levels = 30
            color_bar = False

            # careful about origin when big prob of zero loss
            # handle for discrete distributions
            ps = [f'p_{i}' for i in self.line_names]
            title = 'Bivariate density'
            query = ' or '.join([f'`p_{i}` > 0' for i in self.line_names])
            if self.density_df.query(query).shape[0] < 512:
                logger.info('Contour plot has few points...going discrete...')
                bit = self.density_df.query(query)
                n = len(bit)
                Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
                X, Y = np.meshgrid(bit.index, bit.index)
                norm = mpl.colors.Normalize(vmin=-10, vmax=np.log10(np.max(Z.flat)))
                cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                mapper = cm.to_rgba
                a13.scatter(x=X.flat, y=Y.flat, s=1000 * Z.flatten(), c=mapper(np.log10(Z.flat)))
                # edgecolor='C2', lw=1, facecolors='none')
                a13.set(xlim=[min_loss - (max_loss - min_loss) / 10, max_loss],
                        ylim=[min_loss - (max_loss - min_loss) / 10, max_loss])

            else:
                npts = np.arange(min_loss, max_loss, jump)
                bit = self.density_df.loc[npts, ps]
                n = len(bit)
                Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
                Z = Z / np.sum(Z)
                X, Y = np.meshgrid(bit.index, bit.index)

                if biv_log:
                    z = np.log10(Z)
                    mask = np.zeros_like(z)
                    mask[z == -np.inf] = True
                    mz = np.ma.array(z, mask=mask)
                    cp = a13.contourf(X, Y, mz, levels=np.linspace(-17, 0, levels), cmap=cmap)
                    if color_bar:
                        cb = fig.colorbar(cp)
                        cb.set_label('Log10(Density)')
                else:
                    mask = np.zeros_like(Z)
                    mask[Z < min_density] = True
                    mz = np.ma.array(Z, mask=mask)
                    cp = a13.contourf(X, Y, mz, levels=levels, cmap=cmap)
                    if color_bar:
                        cb = fig.colorbar(cp)
                        cb.set_label('Density')
                a13.set(xlim=[min_loss, max_loss], ylim=[min_loss, max_loss])

            # put in X+Y=c lines
            lines = np.arange(contour_scale / 4, 2 * contour_scale + 1, contour_scale / 4)
            logger.debug(f'Contour lines based on {contour_scale} gives {lines}')
            for x in lines:
                a13.plot([0, x], [x, 0], ls='solid', lw=.35, c='k', alpha=0.5, label=f'Sum = {x:,.0f}')

            a13.set(xlabel=f'Line {self.line_names[0]}',
                    ylabel=f'Line {self.line_names[1]}',
                    title=title,
                    aspect=1)
        else:
            # kind == 'three' plot survival functions...bivariate plots elsewhere
            l1 = (1 - temp.cumsum()).plot(ax=a13, lw=1)
            l1.lines[-1].set(linewidth=1.5)
            a13.set(title='Survival Function')
            a13.legend()

        # kappa
        bit = self.density_df.loc[:xmax]. \
            filter(regex=f'^exeqa_({self.line_name_pipe})$')
        bit = bit.iloc[:, sort_order]
        bit.rename(columns=self.short_renamer('exeqa')).replace(0, np.nan). \
            sort_index(axis=1).iloc[:, sort_order].plot(ax=a21, lw=1)
        a21.set(title='$\\kappa_i(x)=E[X_i\\mid X=x]$')
        # ugg for ASTIN paper example
        # a21.set(xlim=[0,5], ylim=[0,5])
        a21.set(xlim=[0, xmax], ylim=[0, xmax], aspect='equal')
        a21.legend(loc='upper left')

        # alpha and beta
        aug_df = self.augmented_df
        aug_df.filter(regex=f'exi_xgta_({self.line_name_pipe})'). \
            rename(columns=self.short_renamer('exi_xgta')). \
            sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a22, lw=1)
        for ln, ls in zip(a22.lines, lss[1:]):
            ln.set_linestyle(ls)
        a22.legend()
        a22.set(xlim=[0, xmax], title='$\\alpha_i(x)=E[X_i/X\\mid X>x]$');

        bit = aug_df.query(f'loss < {xmax}').filter(regex=f'exi_xgtag?_({self.line_name_pipe})')
        bit.rename(columns=self.short_renamer('exi_xgtag')). \
            sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a23)
        for i, l in enumerate(a23.lines[len(self.line_names):]):
            if l.get_label()[0:3] == 'exi':
                a23.lines[i].set(linewidth=2, ls=lss[1 + i])
                l.set(color=f'C{i}', linestyle=lss[1 + i], linewidth=1,
                      alpha=.5, label=None)
        a23.legend(loc='upper left');
        a23.set(xlim=[0, xmax], title="$\\beta_i(x)=E_{Q}[X_i/X \\mid X> x]$");

        aug_df.filter(regex='M.M').rename(columns=self.short_renamer('M.M')). \
            sort_index(axis=1).iloc[:, sort_order].plot(ax=a32, lw=1)
        a32.set(xlim=[0, xmax], title='Margin density $M_i(x)$')

        aug_df.filter(regex='T.M').rename(columns=self.short_renamer('T.M')). \
            sort_index(axis=1).iloc[:, sort_order].plot(ax=a42, lw=1)
        a42.set(xlim=[0, xmax], title='Margin $\\bar M_i(x)$');

        # by line S, gS, aS, bgS
        adf = self.augmented_df.loc[:xmax]
        if kind == 'two':
            zipper = zip(range(2), sorted(self.line_names), [a31, a41])
        else:
            zipper = zip(range(3), sorted(self.line_names), [a31, a41, a33])
        for i, line, a in zipper:
            a.plot(adf.loss, adf.S, c=col_list[2], ls=lss[1], lw=1, alpha=0.5, label='$S$')
            a.plot(adf.loss, adf.gS, c=col_list[2], ls=lss[0], lw=1, alpha=0.5, label='$g(S)$')
            a.plot(adf.loss, adf.S * adf[f'exi_xgta_{line}'], c=col_list[i], ls=lss[1], lw=1, label=f'$\\alpha S$ {line}')
            a.plot(adf.loss, adf.gS * adf[f'exi_xgtag_{line}'], c=col_list[i], ls=lss[0], lw=1,
                   label=f"$\\beta g(S)$ {line}")
            a.set(xlim=[0, ymax])

            a.set(title=f'Line = {line}')
            a.legend()
            a.set(xlim=[0, ymax])
            a.legend(loc='upper right')

        alpha = 0.05
        if kind == 'two':
            # three mode this is used for the third line
            # a33 from profit segment plot
            # special ymax for last two plots
            ymax = ymax2 if ymax2 > 0 else self.q(p2)
            # if could have changed
            p2 = self.cdf(ymax)
            for cn, ln in enumerate(sort_order):
                line = sorted(self.line_names_ex)[ln]
                c = col_list[cn]
                s = lss[cn]
                # print(line, s)
                f1 = self.density_df[f'p_{line}'].cumsum()
                idx = (f1 < p2) * (f1 > 1.0 - p2)
                f1 = f1[idx]
                gf = 1 - self.distortion.g(1 - f1)
                x = self.density_df.loss[idx]
                a33.plot(gf, x, c=c, ls=s, lw=1, label=None)
                a33.plot(f1, x, ls=s, c=c, lw=1, label=None)
                # a33.plot(f1, x, ls=lss[1], c=c, lw=1, label=None)
                a33.fill_betweenx(x, gf, f1, color=c, alpha=alpha, label=line.title())
            a33.set(ylim=[0, ymax], title='Stand-alone $M$')
            a33.legend(loc='upper left')  # .set_visible(False)

        # a43 from natural profit segment plot
        # common extract for all lines
        lw, up = self.q(1 - p2), ymax
        bit = self.augmented_df.query(f' {lw} <= loss <= {up} ')
        F = bit[f'F']
        gF = bit[f'gF']
        # x = bit.loss
        for cn, ln in enumerate(sort_order):
            line = sorted(self.line_names_ex)[ln]
            c = col_list[cn]
            s = lss[cn]
            ser = bit[f'exeqa_{line}']
            if kind == 'three':
                a43.plot(1 / (1 - F), ser, lw=1, ls=lss[1], c=c)
                a43.plot(1 / (1 - gF), ser, lw=1, c=c)
                a43.set(xlim=[1, 1e4], xscale='log')
                a43.fill_betweenx(ser, 1 / (1 - gF), 1 / (1 - F), color=c, alpha=alpha, lw=0.5, label=line.title())
            else:
                a43.plot(F, ser, lw=1, ls=s, c=c)
                a43.plot(gF, ser, lw=1, ls=s, c=c)
                a43.fill_betweenx(ser, gF, F, color=c, alpha=alpha, lw=0.5, label=line.title())
        a43.set(ylim=[0, ymax], title='Natural $M$')
        a43.legend(loc='upper left')  # .set_visible(False)

        if legend_font:
            for ax in axs.flat:
                try:
                    if ax is not a13:
                        ax.legend(prop={'size': 7})
                except:
                    pass

    def stand_alone_pricing_work(self, dist, p, kind, roe, S_calc='cumsum'):
        """
        Apply dist to the individual lines of self, with capital standard determined by a, p, kind=VaR, TVaR, etc.
        Return usual data frame with L LR M P PQ  Q ROE, and a

        Dist can be a distortion, traditional, or defaut pricing modes. For latter two you have to input an ROE. ROE
        not required for a distortion.

        :param self: a portfolio object
        :param dist: "traditional", "default", or a distortion (already calibrated)
        :param p: probability level for assets
        :param kind: var (or lower, upper), tvar or epd (note, p should be small for EPD, to pander, if p is large we use 1-p)
        :param roe: for traditional methods input roe

        :return: exhibit is copied and augmented with the stand-alone statistics

        from common_scripts.py
        """
        assert S_calc in ('S', 'cumsum')

        var_dict = self.var_dict(p, kind=kind, total='total', snap=True)
        exhibit = pd.DataFrame(0.0, index=['L', 'LR', 'M', 'P', "PQ", 'Q', 'ROE'], columns=['sop'])

        def tidy_and_write(exhibit, ax, exa, prem):
            """ finish up calculation and store answer """
            roe_ = (prem - exa) / (ax - prem)
            exhibit.loc[['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE'], l] = \
                (exa, exa / prem, prem - exa, prem, prem / (ax - prem), ax - prem, roe_)

        if dist == 'traditional - no default':
            # traditional roe method, no allowance for default
            method = dist
            d = roe / (1 + roe)
            v = 1 - d
            for l in self.line_names_ex:
                ax = var_dict[l]
                # no allowance for default
                exa = self.audit_df.at[l, 'EmpMean']
                prem = v * exa + d * ax
                tidy_and_write(exhibit, ax, exa, prem)
        elif dist == 'traditional':
            # traditional but allowing for default
            method = dist
            d = roe / (1 + roe)
            v = 1 - d
            for l in self.line_names_ex:
                ax = var_dict[l]
                exa = self.density_df.loc[ax, f'lev_{l}']
                prem = v * exa + d * ax
                tidy_and_write(exhibit, ax, exa, prem)
        else:
            # distortion method
            method = f'sa {str(dist)}'
            for l, ag in zip(self.line_names_ex, self.agg_list + [None]):
                # use built in apply distortion
                if ag is None:
                    # total
                    if S_calc == 'S':
                        S = self.density_df.S
                    else:
                        # revised
                        S = (1 - self.density_df['p_total'].cumsum())
                    # some dist return np others don't this converts to numpy...
                    gS = pd.Series(dist.g(S), index=S.index)
                    exag = gS.shift(1, fill_value=0).cumsum() * self.bs
                else:
                    ag.apply_distortion(dist)
                    exag = ag.density_df.exag
                ax = var_dict[l]
                exa = self.density_df.loc[ax, f'lev_{l}']
                prem = exag.loc[ax]
                tidy_and_write(exhibit, ax, exa, prem)

        exhibit.loc['a'] = exhibit.loc['P'] + exhibit.loc['Q']
        exhibit['sop'] = exhibit.filter(regex='[A-Z]').sum(axis=1)
        exhibit.loc['LR', 'sop'] = exhibit.loc['L', 'sop'] / exhibit.loc['P', 'sop']
        exhibit.loc['ROE', 'sop'] = exhibit.loc['M', 'sop'] / (exhibit.loc['a', 'sop'] - exhibit.loc['P', 'sop'])
        exhibit.loc['PQ', 'sop'] = exhibit.loc['P', 'sop'] / exhibit.loc['Q', 'sop']

        exhibit['method'] = method
        exhibit = exhibit.reset_index()
        exhibit = exhibit.set_index(['method', 'index'])
        exhibit.index.names = ['method', 'stat']
        exhibit.columns.name = 'line'
        exhibit = exhibit.sort_index(axis=1)

        return exhibit

    def stand_alone_pricing(self, dist, p=0, kind='var', S_calc='cumsum'):
        """

        Run distortion pricing, use it to determine and ROE and then compute traditional and default
        pricing, then consolidate the answer

        :param self:
        :param roe:
        :param p:
        :param kind:
        :return:

        from common_scripts.py
        """
        assert isinstance(dist, (Distortion, list))
        if type(dist) != list:
            dist = [dist]
        ex1s = []
        for d in dist:
            ex1s.append(self.stand_alone_pricing_work(d, p=p, kind=kind, roe=0, S_calc=S_calc))
            if len(ex1s) == 1:
                roe = ex1s[0].at[(f'sa {str(d)}', 'ROE'), 'total']
        ex2 = self.stand_alone_pricing_work('traditional - no default', p=p, kind=kind, roe=roe, S_calc=S_calc)
        ex3 = self.stand_alone_pricing_work('traditional', p=p, kind=kind, roe=roe, S_calc=S_calc)

        return pd.concat(ex1s + [ex2, ex3])

    def calibrate_blends(self, a, premium, s_values, gs_values=None, spread_values=None, debug=False):
        """
        Input s values and gs values or (market) yield or spread.

        A bond with prob s (small) of default is quoted with a yield (to maturity)
        of r over risk free (e.g., a cat bond spread, or a corporate bond spread
        over the appropriate Treasury). As a discount bond, the price is v = 1 - d.

        B(s) = bid price for 1(U<s) (bond residual value)
        A(s) = ask price for 1(U<s) (insurance policy)

        By no arb A(s) + B(1-s) = 1.
        By definition g(s) = A(s) (using LI so the particular U doesn't matter. Applied
        to U = F(X)).

        Let v = 1 / (1 + r) and d = 1 - v be the usual theory of interest quantities.

        Hence B(1-s) = v = 1 - A(s) = 1 - g(s) and therefore g(s) = 1 - v = d.

        The rate of risk discount δ and risk discount factor (nu) ν are defined so that
        B(1-s) = ν * (1 - s), it is the extra discount applied to the actuarial value that
        is bid for the bond. It is a function of s. Therefore ν = (1 - d) / (1 - s) =
        price of bond / actuarial value of payment.

        Then, g(s) = 1 - B(1-s) = 1 - ν (1 - s) = ν s + δ.

        Thus, if return (i.e., market yield spreads) are input, they convert to
        discount factors to define g points.

        Blend can be defined by extrapolating the last points in a credit curve. If
        that fails, increase the return on the highest s point and fill in with a
        constant return to 1.

        The ROE on the investment is not the promised return, because the latter does not
        allow for default.

        Set up to be a function of the Portfolio = self. Calibrated to hit premium at
        asset level a. a must be in the index.

            a = self.pricing_summary.at['a', kind]
            premium = self.pricing_summary.at['P', kind]

        method = extend or roe

        Input

        blend_d0 is the Book's blend, with roe above the equity point
        blend_d is calibrated to the same premium as the other distortions

        method = extend if f_blend_extend or ccoc
            ccoc = pick and equity point and back into its required roe. Results in a
            poor fit to the calibration data

            extend = extrapolate out the last slope from calibrtion data

        Initially tried interpolating the bond yield curve up, but that doesn't work.
        (The slope is too flat and it interpolates too far. Does not look like
        a blend distortion.)
        Now, adding the next point off the credit yield curve as the "equity"
        point and solving for ROE.

        If debug, returns more output, for diagnostics.

        """
        global logger

        # corresponding gs values to s_values
        if gs_values is None:
            gs_values = 1 - 1 / (1 + np.array(spread_values))

        # figure out the convex hull points out of input s, gs
        s_values, gs_values = convex_points(s_values, gs_values)
        if len(s_values) < 4:
            raise ValueError('Input s,gs points do not generate enough separate points.')

        # calibration prefob to compute rho
        df = self.density_df
        bs = self.bs
        # survival function points
        S = (1 - df.p_total[0:a-bs].cumsum())

        def pricer(g):
            nonlocal S, bs
            return np.sum(g(S)) * bs

        # figure the four extreme values:
        # extrapolate or interp to 0 and l or r end
        # it appears you need to do this extra step to lock in the parameters.
        def make_g(s, gs):
            i = interp1d(s, gs, bounds_error=False, fill_value='extrapolate')
            def f(x):
                return np.minimum(1, i(x))
            return f

        ans = []
        dists = {}
        for left in ['e', '0']:
            for right in ['e', '1']:
                s = s_values.copy()
                gs = gs_values.copy()
                if left == 'e':
                    s = s[1:]
                    gs = gs[1:]
                if right == 'e':
                    s = s[:-1]
                    gs = gs[:-1]
                dists[(left, right)] = make_g(s, gs)
                new_p = pricer(dists[(left, right)])
                ans.append((left, right, new_p, new_p > premium))

        # horrible but small and short...
        df = pd.DataFrame(ans, columns=['left', 'right', 'premium', 'gt'])
        wts = {}
        wdists = {}
        for i, j in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
            pi = df.iat[i, 2]
            pj = df.iat[j, 2]
            if min(pi, pj) <= premium <= max(pi, pj):
                il = df.iat[i, 0]
                ir = df.iat[i, 1]
                jl = df.iat[j, 0]
                jr = df.iat[j, 1]
                w = (premium - pj) / (pi - pj)
                wts[(il, ir, jl, jr)] = w
                # feed into a dummy distortion
                temp = Distortion('ph', .599099)
                temp.name = 'blend'
                # temp.display_name  = f'Extension ({il}, {ir}), ({jl}, {jr})'
                temp.g = lambda x: (
                        w * dists[(il, ir)](x) + (1 - w) * dists[(jl, jr)](x))
                temp.g_inv = None
                wdists[(il, ir, jl, jr)] = temp
        if len(wdists) == 0:
            # failed to extrapolate, but still want a reasonable blend
            logger.warning('Failed to fit blend')
            # TODO placeholder - FIX!
            wdists[0] = Distortion('ph', .599099)
            wdists[0].name = 'blend'
        if debug is True:
            return wdists, df, pricer, dists, wts
        else:
            return wdists

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

    def sample(self, n, replace=True, random_state=None, desired_correlation=None, keep_total=True):
        """
        Pull multivariate sample. Apply Iman Conover to induce correlation if required.

        """
        # bit generator
        bg = PCG64(random_state)
        df = pd.DataFrame(index=range(n))
        for c in self.line_names:
            pc = f'p_{c}'
            df[c] = self.density_df[['loss', pc]].\
                    query(f'`{pc}` > 0').\
                    sample(n, replace=replace, weights=pc, ignore_index=True, random_state=bg).\
                    drop(columns=pc)

        if desired_correlation is not None:
            df = iman_conover(df, desired_correlation)
        else:
            df['total'] = df.sum(axis=1)
            df = df.set_index('total', drop=not keep_total)
        df = df.reset_index(drop=True)
        return df

    resample = sample

    @property
    def unit_names(self):
        # what these should have been called!
        return self.line_names

    @property
    def unit_names_ex(self):
        # what these should have been called!
        return self.line_names_ex


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
