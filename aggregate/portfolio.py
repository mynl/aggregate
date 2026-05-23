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
import pandas as pd
from pandas.io.formats.format import EngFormatter
from pandas.plotting import scatter_matrix
from pathlib import Path
import re
import scipy.stats as ss
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy.spatial import ConvexHull
from textwrap import fill
import warnings
from IPython.display import HTML, display

from .constants import *
from .distributions import Aggregate, Severity
from .spectral import Distortion
from .utilities import (ft, ift, axiter_factory, AxisManager, html_title,
                        suptitle_and_tight, pprint_ex,
                        MomentAggregator, Answer, subsets, round_bucket,
                        make_mosaic_figure, iman_conover, approximate_work,
                        make_var_tvar, agg_help, explain_validation,
                        make_comonotonic_allocations as make_comonotonic_allocations_work)
import aggregate.random_agg as ar


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
        # make a pandas data frame of all the statistics_df
        temp_report = pd.concat(
            [a.stats_df['mixed'].rename(a.name) for a in self.agg_list],
            axis=1,
        )

        # max_limit = np.inf # np.max([np.max(a.get('limit', np.inf)) for a in spec_list])
        temp = pd.DataFrame(ma.stats_series('total', max_limit, 0.999, remix=False))
        self.statistics_df = pd.concat([temp_report, temp], axis=1)
        # future storage
        self.density_df = None
        self.independent_density_df = None
        self.augmented_df = None
        self.audit_df = None
        self.independent_audit_df = None
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
        # update total stats
        logger.info('Updating total statistics (WARNING: these are now empirical)')
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

    def pricing_bounds(self, premium, a=0, p=0, n_tps=512, s=512, kind='interp', slow=False, verbose=250):
        """
        Compute the natural allocation premium ranges by unit consistent with
        total premium at asset level a or p (one of which must be provided).

        Unlike typical case with even s values, this is run at the actual S
        values of the Portfolio.

        Use s<=0 to use S values.

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
        ans = namedtuple('pricing_bounds', 'bounds allocs stats comp allocs_slow p_star')
        p_star = bounds.p_star('total', premium, a, kind=kind)
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
        if self.audit_df is None:
            ex = self.statistics_df.loc[('agg', 'mean'), 'total']
            empex = np.nan
            isupdated = False
        else:
            ex = self.audit_df.loc['total', 'Mean']
            empex = self.audit_df.loc['total', 'EmpMean']
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
        self._var_tvar_function = None
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
        except ZeroDivisionError as e:
            pass
        try:
            self.audit_df['CVErr'] = self.audit_df['EmpCV'] / self.audit_df['CV'] - 1
        except ZeroDivisionError as e:
            pass
        try:
            self.audit_df['SkewErr'] = self.audit_df['EmpSkew'] / self.audit_df['Skew'] - 1
        except ZeroDivisionError as e:
            pass

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
        return pprint_ex(self.program, 20)

    @property
    def pprogram_html(self):
        """
        pretty print the program to html
        """
        return pprint_ex(self.program, 0, html=True)

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
            with np.errstate(divide='ignore', invalid='ignore'):
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
                    f'Portfolio.calibrate_distortion | Questionable convergence for {name} distortion, target '
                    f'{premium_target} error {fx} after {i} iterations')

        # build answer
        dist = Distortion(name=name, shape=shape, r0=r0, df=df)
        dist.error = fx
        dist.assets = assets
        dist.premium_target = premium_target
        return dist

    def calibrate_distortions2(self, coc, reg_p):
        """Simplified calibrate_distortions reflecting how it is used."""
        ans = self.calibrate_distortions(COCs=[coc], Ps=[reg_p])

    def calibrate_distortions(self, LRs=None, COCs=None, ROEs=None, As=None, Ps=None, kind='lower', r0=0.03, df=5.5,
                              strict='ordered', S_calc='cumsum'):
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
                     'ROE', 'param', 'std_param', 'error', 'method'], dtype=float)
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
                                                  dist.shape, dist.standard_shape, dist.error]
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

        df = self.dist_ans.iloc[:, [0,4,5,6,7,8,9,10,11]].copy()
        df.index.names = ['a', 'LR', 'method']
        df.columns = ['S', 'L', 'P', 'PQ', 'Q', 'COC', 'param', 'std_param', 'error']
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
          ``self.dists`` dictionary is used.
        :param allocation: 'lifted' (default for legacy reasons) or 'linear': treatment in default scenarios. See PIR.
        :param view: bid or ask
        :param efficient: for apply_distortion, lifted only.
        :return: PricingResult namedtuple with 'price', 'assets', 'reg_p', 'distortion', 'df'
        """

        # warnings.warn('In 0.13.0 the default allocation will become linear not lifted.', DeprecationWarning)

        assert allocation in ('lifted', 'linear'), "allocation must be 'lifted' or 'linear'"
        PricingResult = namedtuple('PricingResult', ['df', 'price', 'price_dict', 'a_reg', 'reg_p'])

        if isinstance(distortion, Distortion):
            distortion = {str(distortion): distortion}
        elif isinstance(distortion, list):
            distortion = {str(d): d for d in distortion}
        elif distortion is None:
            assert self.dists is not None, 'Must pass a distortion or calibrate distortions prior to calling'
            distortion = self.dists

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
                ans_ad = self.apply_distortion(v, view=view, create_augmented=False, efficient=efficient)
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

    def analyze_distortions2(self, p, dists=None):
        """
        Updated version of analyze_distortions reflecting how it is really used!

        Use dists or self.dists

        Returns only comp_df.
        """
        dfs = {}
        dists = dists or self.dists
        if dists is None:
            raise ValueError('Must pass dists or self must have dists. '
                             'Did you forget to calibrate_distortions?')

        # defaults that we always select
        use_self = add_comps = False
        kind = 'lower'
        efficient = True
        for k, d in dists.items():
            # first distortion...add the comps...these are same for all dists
            ad_ans = self.analyze_distortion(d, p=p, kind=kind, add_comps=add_comps,
                                             efficient=efficient, use_self=use_self,
                                             rename_dists=False)
            dfs[d] = ad_ans.exhibit
        ans = pd.concat(dfs.values(), keys=[str(i) for i in dfs.keys()])   #.sort_index()
        ans = ans.droplevel(1, axis=0)

        # force the same order as input dist
        # ans.index.name = 'distortion'
        # new_index_df = ans.index.to_frame()
        # new_index_df['distortion'] = (
        #     new_index_df['distortion']
        #     .astype(pd.CategoricalDtype(categories=dists.keys(), ordered=True))
        # )
        # # Reassign and sort
        # ans.index = new_index_df.distortion
        # ans = ans.sort_index()
        return ans

    def analyze_distortion(self, dname, dshape=None, dr0=.025, ddf=5.5, LR=None, ROE=None,
                           p=None, kind='lower', A=None, use_self=False, plot=False,
                           a_max_p=1-1e-8, add_comps=False, efficient=True, rename_dists=True):
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
                    filter(regex=r'^(T)\.(L|P|M|Q)_', axis=0).copy()
        pricing = pricing.loc[~pricing.index.duplicated()]
        # put in exact Q rather than add up the parts...more accurate
        pricing.at['T.Q_total', a_cal] = a_cal - exag
        # TODO EVIL! this reorders the lines and so messes up when port has lines not in alpha order
        pricing.index = pricing.index.str.split(r'_|\.', expand=True)
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
        # ``add_comps`` and ``plot`` branches removed in Sub-project A: the
        # backing helpers (analyze_distortion_add_comps and
        # analyze_distortion_plots) consumed deleted Bucket-D allocation
        # methods (cotvar, merton_perold, equal_risk_*, EPD); PMIR always
        # passes add_comps=False and never sets plot=True.
        if rename_dists:
            ans['exhibit'] = ans.exhibit.rename(index={'T': f'Dist {dist.name}'}).sort_index()
        return ans


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


