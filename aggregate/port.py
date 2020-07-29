"""
Purpose
-------

A Portfolio represents a collection of Aggregate objects. Applications include

* Model a book of insurance
* Model a large account with several sub lines
* Model a reinsurance portfolio or large treaty



"""

import collections
import json
import logging
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pypandoc
import scipy.stats as ss
from IPython.core.display import HTML, display
from matplotlib.ticker import MultipleLocator, StrMethodFormatter, MaxNLocator, FixedLocator, \
    FixedFormatter, AutoMinorLocator
from scipy import interpolate
import re

from .distr import Aggregate, CarefulInverse, Severity
from .spectral import Distortion
from .utils import ft, \
    ift, sln_fit, sgamma_fit, \
    axiter_factory, AxisManager, html_title, \
    sensible_jump, suptitle_and_tight, \
    MomentAggregator, Answer, subsets, round_bucket

# fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
matplotlib.rcParams['legend.fontsize'] = 'xx-small'

logger = logging.getLogger('aggregate')


# debug
# info
# warning
# error
# critical

class Portfolio(object):
    """
    Portfolio creates and manages a portfolio of Aggregate objects.

    :param name: the name of the portfolio, no spaces or underscores
    :param spec_list: a list of 1) dictionary: Aggregate object dictionary specifications or
                                2) Aggregate: An actual aggregate objects or
                                3) tuple (type, dict) as returned by uw['name'] or
                                4) string: Names referencing objects in the optionally passed underwriter

    """

    def __init__(self, name, spec_list, uw=None):
        self.name = name
        self.agg_list = []
        self.line_names = []
        logger.info(f'Portfolio.__init__| creating new Portfolio {self.name}')
        # logger.info(f'Portfolio.__init__| creating new Portfolio {self.name} at {super(Portfolio, self).__repr__()}')
        ma = MomentAggregator()
        max_limit = 0
        for spec in spec_list:
            if isinstance(spec, Aggregate):
                # directly passed in an agg object
                a = spec
                agg_name = spec.name
            elif isinstance(spec, str):
                # look up object in uw return actual instance
                # note here you could do uw.aggregate[spec] and get the dictionary def
                # or uw(spec) to return the already-created (and maybe updated) object
                # we go the latter route...if user wants they can pull off the dict item themselves
                if uw is None:
                    raise ValueError(f'Must pass valid Underwriter instance to create aggs by name')
                try:
                    a = uw(spec)
                except e:
                    print(f'Item {spec} not found in your underwriter')
                    raise e
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

            self.agg_list.append(a)
            self.line_names.append(agg_name)
            self.__setattr__(agg_name, a)
            ma.add_fs(a.report_ser[('freq', 'ex1')], a.report_ser[('freq', 'ex2')], a.report_ser[('freq', 'ex3')],
                      a.report_ser[('sev', 'ex1')], a.report_ser[('sev', 'ex2')], a.report_ser[('sev', 'ex3')])
            max_limit = max(max_limit, np.max(np.array(a.limit)))
        self.line_names_ex = self.line_names + ['total']
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
        self.augmented_df = None
        self.epd_2_assets = {}
        self.assets_2_epd = {}
        self.priority_capital_df = None
        self.priority_analysis_df = None
        self.audit_df = None
        self.padding = 0
        self.tilt_amount = 0
        self._linear_quantile_function = None
        self._cdf = None
        self._pdf = None
        self._tail_var = None
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
        # for storing the info about the quantile function
        self.q_temp = None
        self._renamer = None
        self._line_renamer = None
        # if created by uw it stores the program here
        self.program = ''
        self.audit_percentiles = [.9, .95, .99, .995, .999, .9999, 1 - 1e-6]
        self.dists = None
        self.dist_ans = None

    def __str__(self):
        """
        Goal: readability
        :return:
        """
        # cannot use ex, etc. because object may not have been updated
        if self.audit_df is None:
            ex = self.statistics_df.loc[('agg', 'mean'), 'total']
            empex = np.nan
            isupdated = False
        else:
            ex = self.get_stat(stat="Mean")
            empex = self.get_stat()
            isupdated = True
        # df = pd.DataFrame(columns=['Statistic', 'Value'])
        # df = df.set_index('Statistic')
        # df.loc['Portfolio Name', 'Value'] = self.name
        # df.loc['Expected loss', 'Value'] = ex
        # df.loc['Model loss', 'Value'] = empex
        # df.loc['Error', 'Value'] = ex / empex - 1
        # print(df)
        s = f'Portfolio name           {self.name:<15s}\n' \
            f'Theoretic expected loss  {ex:15,.1f}\n' \
            f'Actual expected loss     {empex:15,.1f}\n' \
            f'Error                    {empex / ex - 1:15.6f}\n' \
            f'Discretization size      {self.log2:15d}\n' \
            f'Bucket size              {self.bs:15.2f}\n' \
            f'{object.__repr__(self)}'
        if not isupdated:
            s += '\nNOT UPDATED!'
        return s

    @property
    def distortion(self):
        return self._distortion

    def remove_fuzz(self, df=None, eps=0, force=False, log=''):
        """
        remove fuzz at threshold eps. if not passed use np.finfo(np.float).eps.

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
            eps = np.finfo(np.float).eps

        if self._remove_fuzz or force:
            logger.info(f'CPortfolio.remove_fuzz | Removing fuzz from {self.name} dataframe, caller {log}')
            df.loc[:, df.select_dtypes(include=['float64']).columns] = \
                df.select_dtypes(include=['float64']).applymap(lambda x: 0 if abs(x) < eps else x)

    def __repr__(self):
        """
        Goal unmbiguous
        :return:
        """
        # return str(self.to_dict())
        # this messes up when port = self has been enhanced...
        if isinstance(self, Portfolio):
            s = [super(Portfolio, self).__repr__(), f"{{ 'name': '{self.name}'"]
        else:
            s = [f'Non-Portfolio (enhanced) object {{ "name": "{self.name}"']
        agg_list = [str({k: v for k, v in a.__dict__.items() if k in Aggregate.aggregate_keys})
                    for a in self.agg_list]
        s.append(f"'spec': [{', '.join(agg_list)}]")
        if self.bs > 0:
            s.append(f'"bs": {self.bs}')
            s.append(f'"log2": {self.log2}')
            s.append(f'"padding": {self.padding}')
            s.append(f'"tilt_amount": {self.tilt_amount}')
            s.append(f'"distortion": "{repr(self._distortion)}"')
            s.append(f'"sev_calc": "{self.sev_calc}"')
            s.append(f'"remove_fuzz": {self._remove_fuzz}')
            s.append(f'"approx_type": "{self.approx_type}"')
            s.append(f'"approx_freq_ge": {self.approx_freq_ge}')
        return ', '.join(s) + '}'

    def _repr_html_(self):
        s = [f'<h2>Portfolio object: {self.name}</h2>']
        _n = len(self.agg_list)
        _s = "" if _n <= 1 else "s"
        s.append(f'Portfolio contains {_n} aggregate component{_s}')
        summary_sl = (slice(None), ['mean', 'cv', 'skew'])
        if self.audit_df is not None:
            _df = pd.concat((self.statistics_df.loc[summary_sl, :],
                             self.audit_df[['Mean', 'EmpMean', 'MeanErr', 'CV', 'EmpCV', 'CVErr', 'P99.0']].T),
                            sort=True)
            s.append(_df._repr_html_())
        else:
            s.append(self.statistics_df.loc[summary_sl, :]._repr_html_())
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
        return self.agg_list[item]

    @property
    def audit(self):
        """
        Renamed version of the audit dataframe
        :return:
        """
        if self.audit_df is not None:
            return self.audit_df.rename(columns=self.renamer, index=self.line_renamer).T

    @property
    def density(self):
        """
        Renamed version of the density_df dataframe
        :return:
        """
        if self.density_df is not None:
            return self.density_df.rename(columns=self.renamer)

    @property
    def augmented(self):
        """
        Renamed version of the density_df dataframe
        :return:
        """
        if self.augmented_df is not None:
            return self.augmented_df.rename(columns=self.renamer)

    @property
    def statistics(self):
        """
        Renamed version of the statistics dataframe
        :return:
        """
        return self.statistics_df.rename(columns=self.renamer)

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

        d = dict()
        # original
        # d[self.name] = dict(args=args, spec=[a.spec for a in self.agg_list])
        d['name'] = self.name
        d['args'] = args
        d['spec_list'] = [a._spec for a in self.agg_list]

        logger.info(f'Portfolio.json| dummping {self.name} to {stream}')
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
            # TODO: directory naming
            filename = './agg/user.json'

        with open(filename, mode=mode) as f:
            self.json(stream=f)
            logger.info(f'Portfolio.save | {self.name} saved to {filename}')

    def __add__(self, other):
        """
        Add two portfolio objects INDEPENDENT sum (down road can look for the same severity...)

        TODO same severity!

        :param other:
        :return:
        """
        assert isinstance(other, Portfolio)
        # TODO consider if better naming of L&R sides is in order
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
        return self._cdf(x)

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

    # # make some handy aliases; delete these go strictly with scipy.stats notation
    # def F(self, x):
    #     """
    #     handy alias for distribution, CDF
    #     :param x:
    #     :return:
    #     """
    #     return self.cdf(x)
    #
    # def S(self, x):
    #     """
    #     handy alias for survival function, S
    #     :param x:
    #     :return:
    #     """
    #     return self.sf(x)

    def var(self, p):
        """
        value at risk = alias for quantile function

        :param p:
        :return:
        """
        return self.q(p)

    def tvar(self, p):
        """
        Compute the tail value at risk at threshold p

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
        :return:
        """
        assert self.density_df is not None

        _var = self.q(p)
        # evil floating point issue here... this is XXXX TODO kludge because 13 is not generally applicable
        # if you pick bs to be binary-consistent this error will not occur
        # ex = self.density_df.loc[np.round(_var + self.bs, 13):, ['p_total', 'loss']].product(axis=1).sum()
        ex = self.density_df.loc[_var + self.bs:, ['p_total', 'loss']].product(axis=1).sum()
        pip = (self.density_df.loc[_var, 'F'] - p) * _var
        t_var = 1 / (1 - p) * (ex + pip)
        return t_var
        # original implementation interpolated
        # if self._tail_var is None:
        #     # make tvar function
        #     self._tail_var = interpolate.interp1d(self.density_df.F, self.density_df.exgta_total,
        #                                           kind='linear', bounds_error=False,
        #                                           fill_value='extrapolate')
        # return self._tail_var(p)

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

    def merton_perold(self, p, kind='lower'):
        """
        compute Merton Perold capital allocation at VaR(p) capital using VaR as risk measure
        v = q(p)
        TODO TVaR MERPER
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
        make the p co-tvar asset allocation  using ISA
        Asset alloc = exgta = tail expected value, treating TVaR like a pricing variable
        """
        av = self.q(p)
        return self.density_df.loc[av, [f'exgta_{l}' for l in self.line_names_ex]].values

    def as_severity(self, limit=np.inf, attachment=0, conditional=False):
        """
        convert into a severity without recomputing

        throws error if self not updated

        :param limit:
        :param attachment:
        :param conditional:
        :return:
        """
        if self.density_df is None:
            raise ValueError('Must update prior to converting to severity')
        return Severity(sev_name=self, sev_a=self.log2, sev_b=self.bs,
                        exp_attachment=attachment, exp_limit=limit, sev_conditional=conditional)

    def fit(self, approx_type='slognorm', output='agg'):
        """
        returns a dictionary specification of the portfolio aggregate_project
        if updated uses empirical moments, otherwise uses theoretic moments

        :param approx_type: slognorm | sgamma
        :param output: return a dict or agg language specification
        :return:
        """
        if self.audit_df is None:
            # not updated
            m = self.statistics_df.loc[('agg', 'mean'), 'total']
            cv = self.statistics_df.loc[('agg', 'cv'), 'total']
            skew = self.statistics_df.loc[('agg', 'skew'), 'total']
        else:
            # use statistics_df matched to computed aggregate_project
            m, cv, skew = self.audit_df.loc['total', ['EmpMean', 'EmpCV', 'EmpSkew']]

        name = f'{approx_type[0:4]}~{self.name[0:5]}'
        agg_str = f'agg {name} 1 claim sev '

        if approx_type == 'slognorm':
            shift, mu, sigma = sln_fit(m, cv, skew)
            # self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
            sev = {'sev_name': 'lognorm', 'sev_shape': sigma, 'sev_scale': np.exp(mu), 'sev_loc': shift}
            agg_str += f'{np.exp(mu)} * lognorm {sigma} + {shift} '
        elif approx_type == 'sgamma':
            shift, alpha, theta = sgamma_fit(m, cv, skew)
            # self.fzapprox = ss.gamma(alpha, scale=theta, loc=shift)
            sev = {'sev_name': 'gamma', 'sev_a': alpha, 'sev_scale': theta, 'sev_loc': shift}
            agg_str += f'{theta} * lognorm {alpha} + {shift} '
        else:
            raise ValueError(f'Inadmissible approx_type {approx_type} passed to fit')

        if output == 'agg':
            agg_str += ' fixed'
            return agg_str
        else:
            return {'name': name, 'note': f'frozen version of {self.name}', 'exp_en': 1, **sev, 'freq_name': 'fixed'}

    def collapse(self, approx_type='slognorm'):
        """
        returns new Portfolio with the fit

        TODO: deprecated...prefer uw(self.fit()) to go through the agg language approach

        :param approx_type: slognorm | sgamma
        :return:
        """
        spec = self.fit(approx_type, output='dict')
        logger.debug(f'Portfolio.collapse | Collapse created new Portfolio with spec {spec}')
        logger.warning(f'Portfolio.collapse | Collapse is deprecated; use fit() instead.')
        return Portfolio(f'Collapsed {self.name}', [spec])

    def percentiles(self, pvalues=None):
        """
        report_ser on percentiles and large losses
        uses interpolation, audit_df uses nearest

        :pvalues: optional vector of log values to use. If None sensible defaults provided
        :return: DataFrame of percentiles indexed by line and log
        """
        df = pd.DataFrame(columns=['line', 'log', 'Agg Quantile'])
        df = df.set_index(['line', 'log'])
        # df.columns.name = 'perspective'
        if pvalues is None:
            pvalues = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.994, 0.995, 0.999, 0.9999]
        for line in self.line_names_ex:
            q_agg = interpolate.interp1d(self.density_df.loc[:, f'p_{line}'].cumsum(), self.density_df.loss,
                                         kind='linear', bounds_error=False, fill_value='extrapolate')
            for p in pvalues:
                qq = q_agg(p)
                df.loc[(line, p), :] = [float(qq)]
        df = df.unstack(level=1)
        return df

    def recommend_bucket(self):
        """
        data to help estimate a good bucket size

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

    def best_bucket(self, log2=16):
        bs = sum([a.recommend_bucket(log2) for a in self])
        return round_bucket(bs)

    def update(self, log2, bs, approx_freq_ge=100, approx_type='slognorm', remove_fuzz=False,
               sev_calc='discrete', discretization_calc='survival', normalize=True, padding=1, tilt_amount=0, epds=None,
               trim_df=False, verbose=False, add_exa=True):
        """
        create density_df, performs convolution. optionally adds additional information if ``add_exa=True``
        for allocation and priority analysis

        tilting: [@Grubel1999]: Computation of Compound Distributions I: Aliasing Errors and Exponential Tilting
        (ASTIN 1999)
        tilt x numbuck < 20 is recommended log. 210
        num buckets and max loss from bucket size


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
        :param verbose: level of output
        :param add_exa: run add_exa to append additional allocation information needed for pricing; if add_exa also add
        epd info
        :return:
        """

        self.log2 = log2
        self.bs = bs
        self.padding = padding
        self.tilt_amount = tilt_amount
        self.approx_type = approx_type
        self.sev_calc = sev_calc
        self._remove_fuzz = remove_fuzz
        self.approx_type = approx_type
        self.approx_freq_ge = approx_freq_ge
        self.discretization_calc = discretization_calc

        if self.hash_rep_at_last_update == hash(self):
            print(f'Nothing has changed since last update at {self.last_update}')
            return

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
            _a = agg.update(xs, self.padding, tilt_vector, 'exact' if agg.n < approx_freq_ge else approx_type,
                            sev_calc, discretization_calc, normalize, verbose=verbose)
            if verbose:
                display(_a)
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
            self.density_df[f'ημ_{line}'] = np.real(ift(ft_not, self.padding, tilt_vector))

        self.remove_fuzz(log='update')

        # make audit statistics_df df
        theoretical_stats = self.statistics_df.T.filter(regex='agg')
        theoretical_stats.columns = ['EX1', 'EX2', 'EX3', 'Mean', 'CV', 'Skew', 'Limit', 'P99.9Est']
        theoretical_stats = theoretical_stats[['Mean', 'CV', 'Skew', 'Limit', 'P99.9Est']]
        # self.audit_percentiles = [0.9, 0.95, 0.99, 0.995, 0.996, 0.999, 0.9999, 1 - 1e-6]
        self.audit_df = pd.DataFrame(
            columns=['Sum probs', 'EmpMean', 'EmpCV', 'EmpSkew', "EmpKurt", 'EmpEX1', 'EmpEX2', 'EmpEX3'] +
                    ['P' + str(100 * i) for i in self.audit_percentiles])
        for col in self.line_names_ex:
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
            self.audit_df.loc[col, :] = newrow
        self.audit_df = pd.concat((theoretical_stats, self.audit_df), axis=1, sort=True)
        self.audit_df['MeanErr'] = self.audit_df['EmpMean'] / self.audit_df['Mean'] - 1
        self.audit_df['CVErr'] = self.audit_df['EmpCV'] / self.audit_df['CV'] - 1
        self.audit_df['SkewErr'] = self.audit_df['EmpSkew'] / self.audit_df['Skew'] - 1

        # add exa details
        if add_exa:
            self.add_exa(self.density_df, details=True)
            # default priority analysis
            logger.info('Adding EPDs in Portfolio.update')
            if epds is None:
                epds = np.hstack(
                    [np.linspace(0.5, 0.1, 4, endpoint=False)] +
                    [np.linspace(10 ** -n, 10 ** -(n + 1), 9, endpoint=False) for n in range(1, 7)])
                epds = np.round(epds, 7)
            self.priority_capital_df = pd.DataFrame(index=pd.Index(epds))
            for col in self.line_names:
                for i in range(3):
                    self.priority_capital_df.loc[:, '{:}_{:}'.format(col, i)] = self.epd_2_assets[(col, i)](epds)
                    self.priority_capital_df.loc[:, '{:}_{:}'.format('total', 0)] = self.epd_2_assets[('total', 0)](
                        epds)
                col = 'not ' + col
                for i in range(2):
                    self.priority_capital_df.loc[:, '{:}_{:}'.format(col, i)] = self.epd_2_assets[(col, i)](epds)
            self.priority_capital_df.loc[:, '{:}_{:}'.format('total', 0)] = self.epd_2_assets[('total', 0)](epds)
            self.priority_capital_df.columns = self.priority_capital_df.columns.str.split("_", expand=True)
            self.priority_capital_df.sort_index(axis=1, level=1, inplace=True)
            self.priority_capital_df.sort_index(axis=0, inplace=True)
        else:
            # at least want F and S to get quantile functions
            self.density_df['F'] = np.cumsum(self.density_df.p_total)
            self.density_df['S'] = 1 - self.density_df.F

        self.ex = self.audit_df.loc['total', 'EmpMean']
        self.last_update = np.datetime64('now')
        self.hash_rep_at_last_update = hash(self)
        if trim_df:
            self.trim_df()
        # invalidate stored functions
        self._linear_quantile_function = None
        self.q_temp = None
        self._cdf = None

    def update_efficiently(self, log2, bs, approx_freq_ge=100, approx_type='slognorm',
                           sev_calc='discrete', discretization_calc='survival', normalize=True, padding=1):
        """
        runs stripped down versions of update and add_exa - bare bones
        code copied from those routines and cleaned for comments etc.

        :param log2:
        :param bs:
        :param approx_freq_ge:
        :param approx_type:
        :param remove_fuzz:
        :param sev_calc:
        :param discretization_calc:
        :param padding:
        :return:
        """
        self.log2 = log2
        self.bs = bs
        self.padding = padding
        self.approx_type = approx_type
        self.sev_calc = sev_calc
        self._remove_fuzz = True
        self.approx_type = approx_type
        self.approx_freq_ge = approx_freq_ge
        self.discretization_calc = discretization_calc

        ft_line_density = {}
        N = 1 << log2
        MAXL = N * bs
        xs = np.linspace(0, MAXL, N, endpoint=False)
        # no tilt for efficient mode
        tilt_vector = None

        # where the answer will live
        self.density_df = pd.DataFrame(index=xs)
        self.density_df['loss'] = xs
        ft_all = None
        for agg in self.agg_list:
            raw_nm = agg.name
            nm = f'p_{agg.name}'
            _a = agg.update_efficiently(xs, self.padding, 'exact' if agg.n < approx_freq_ge else approx_type,
                                        sev_calc, discretization_calc, normalize)
            ft_line_density[raw_nm] = agg.ftagg_density
            self.density_df[nm] = agg.agg_density
            if ft_all is None:
                ft_all = np.copy(ft_line_density[raw_nm])
            else:
                ft_all *= ft_line_density[raw_nm]
        self.density_df['p_total'] = np.real(ift(ft_all, self.padding, tilt_vector))

        # make the not self.line_density = sum of all but the given line
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
            self.density_df[f'ημ_{line}'] = np.real(ift(ft_not, self.padding, tilt_vector))
            ft_nots[line] = ft_not

        self.remove_fuzz(log='update_efficiently')

        # no audit statistics_df

        # BEGIN add_exa ================================================================================================
        # add exa details now in-line
        # def add_exa(self, df, details, ft_nots=None):
        # Call is self.add_exa(self.density_df, details=True)

        # name in add_exa, keeps code shorter
        df = self.density_df
        cut_eps = np.finfo(np.float).eps

        # sum of p_total is so important...we will rescale it...
        if not np.all(df.p_total >= 0):
            # have negative densities...get rid of them
            first_neg = np.argwhere((df.p_total < 0).to_numpy()).min()
        sum_p_total = df.p_total.sum()

        df['F'] = np.cumsum(df.p_total)
        df['S'] = np.hstack((df.p_total.to_numpy()[:0:-1].cumsum()[::-1],
                             min(df.p_total.iloc[-1],
                                 max(0, 1. - (df.p_total.sum())))))

        # E(min(X, a))
        df['exa_total'] = self.cumintegral(df['S'])
        df.loc[:, 'lev_total'] = df['exa_total']

        df['exlea_total'] = \
            (df.exa_total - df.loss * df.S) / df.F
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

        df['e_total'] = np.sum(df.p_total * df.loss)
        df['exgta_total'] = df.loss + (df.e_total - df.exa_total) / df.S
        df['exeqa_total'] = df.loss  # E(X | X=a) = a(!) included for symmetry was exa

        # FFT functions for use in exa calculations
        # computing sums so minimal padding required
        def loc_ft(x):
            return ft(x, 1, None)

        def loc_ift(x):
            return ift(x, 1, None)

        # where is S=0
        Seq0 = (df.S == 0)

        for col in self.line_names:
            df['exeqa_' + col] = \
                np.real(loc_ift(loc_ft(df.loss * df['p_' + col]) *
                                ft_nots[col])) / df.p_total
            df.loc[df.p_total < cut_eps, 'exeqa_' + col] = 0
            df['exeqa_ημ_' + col] = \
                np.real(loc_ift(loc_ft(df.loss * df['ημ_' + col]) *
                                loc_ft(df['p_' + col]))) / df.p_total
            df.loc[df.p_total < cut_eps, 'exeqa_ημ_' + col] = 0
            stemp = 1 - df.loc[:, 'p_' + col].cumsum()
            df['lev_' + col] = self.cumintegral(stemp)

            stemp = 1 - df.loc[:, 'ημ_' + col].cumsum()
            df['lev_ημ_' + col] = self.cumintegral(stemp)

            # EX_i | X<= a; temp is used in le and gt calcs
            temp = np.cumsum(df['exeqa_' + col] * df.p_total)
            df['exlea_' + col] = temp / df.F
            df.loc[0:loss_max, 'exlea_' + col] = 0  # df.loc[0:loss_max, 'loss']
            temp_not = np.cumsum(df['exeqa_ημ_' + col] * df.p_total)
            df['exlea_ημ_' + col] = temp_not / df.F
            df.loc[0:loss_max, 'exlea_ημ_' + col] = 0  # df.loc[0:loss_max, 'loss']

            # constant value, helpful in calculations
            # df['e_' + col] = np.sum(df['p_' + col] * df.loss)
            # df['e_ημ_' + col] = np.sum(df['ημ_' + col] * df.loss)
            #
            # df['exgta_' + col] = (df['e_' + col] - temp) / df.S

            # temp = df.loss.iloc[0]  # loss
            # df.loss.iloc[0] = 1  # avoid divide by zero
            #
            # # df['exi_x_' + col] = np.sum(
            # #     df['exeqa_' + col] * df.p_total / df.loss)
            # temp_xi_x = np.cumsum(df['exeqa_' + col] * df.p_total / df.loss)
            # df['exi_xlea_' + col] = temp_xi_x / df.F
            # df.loc[0, 'exi_xlea_' + col] = 0  # df.F=0 at zero
            # # more generally F=0 error:                      V
            # df.loc[df.exlea_total == 0, 'exi_xlea_' + col] = 0
            # # not version
            # df['exi_x_ημ_' + col] = np.sum(
            #     df['exeqa_ημ_' + col] * df.p_total / df.loss)
            # # as above
            # temp_xi_x_not = np.cumsum(
            #     df['exeqa_ημ_' + col] * df.p_total / df.loss)
            # df['exi_xlea_ημ_' + col] = temp_xi_x_not / df.F
            # df.loc[0, 'exi_xlea_ημ_' + col] = 0  # df.F=0 at zero
            # # more generally F=0 error:
            # df.loc[df.exlea_total == 0, 'exi_xlea_ημ_' + col] = 0
            # # put value back
            # df.loss.iloc[0] = temp

            # this is so important we will calculate it directly
            df['exi_xgta_' + col] = ((df[f'exeqa_{col}'] / df.loss *
                                      df.p_total).shift(-1)[
                                     ::-1].cumsum()) / df.S
            # need this NOT to be nan otherwise exa won't come out correctly
            df.loc[Seq0, 'exi_xgta_' + col] = 0.
            df['exi_xgta_ημ_' + col] = ((df[f'exeqa_ημ_{col}'] / df.loss *
                                         df.p_total).shift(-1)[
                                        ::-1].cumsum()) / df.S
            df.loc[Seq0, 'exi_xgta_ημ_' + col] = 0.
            df['exi_xeqa_' + col] = df['exeqa_' + col] / df['loss']
            df.loc[0, 'exi_xeqa_' + col] = 0
            df['exi_xeqa_ημ_' + col] = df['exeqa_ημ_' + col] / df['loss']
            df.loc[0, 'exi_xeqa_ημ_' + col] = 0
            df[f'exa_{col}'] = (df.S * df['exi_xgta_' + col]).shift(1, fill_value=0).cumsum() * self.bs
            df['exa_ημ_' + col] = (df.S * df['exi_xgta_ημ_' + col]).shift(1, fill_value=0).cumsum() * self.bs

        # END add_exa ==================================================================================================

        self.last_update = np.datetime64('now')
        # invalidate stored functions
        self._linear_quantile_function = None
        self.q_temp = None
        self._cdf = None

    def trim_df(self):
        """
        trim out unwanted columns from density_df

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
        to swap. But if you want exa_A, exa_B you do, otherwise the d/dA exa_B won't be correct. TODO: replace with code!
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
            _a = agg.update(xs, self.padding, tilt_vector, 'exact' if agg.n < self.approx_freq_ge else self.approx_type,
                            self.sev_calc, self.discretization_calc, verbose=False)
            agg_epsilon_df[f'p_{agg.name}'] = agg.agg_density
            # the total with the line incremented
            agg_epsilon_df[f'p_total_{agg.name}'] = \
                np.real(loc_ift(agg.ftagg_density * loc_ft(self.density_df[f'ημ_{agg.name}'])))

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
                self.add_exa(gradient_df, details=False, ft_nots=swap(line))
            else:
                self.add_exa(gradient_df, details=False)
            if distortion is not None:
                # apply to line + epsilon
                gradient_df = self.apply_distortion(distortion, df_in=gradient_df, create_augmented=False).augmented_df

            # compute differentials and store answer!
            # print(columns_of_interest)
            # print([(line, i) for i in columns_of_interest])
            # print(type(gradient_df))
            # temp0 = gradient_df[columns_of_interest]
            # temp1 = base[columns_of_interest]
            # temp2 = (temp0 - temp1) / dx
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
                    if self.priority_capital_df is not None:
                        display(self.priority_capital_df.loc[1e-3:1e-2, :].style)
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

    def plot(self, kind='density', line='all', p=0.99, c=0, a=0, axiter=None, figsize=None, height=2,
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
                self.audit_df.loc[:, 'P99.9']. \
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
            self.density_df.loc[:, line].sort_index(axis=1). \
                plot(sort_columns=True, ax=ax, **kwargs)
            if 'logy' in kwargs:
                _t = 'log Density'
            else:
                _t = 'Density'
            if 'subplots' in kwargs and isinstance(ax, collections.Iterable):
                for a, l in zip(ax, line):
                    a.set(title=f'{l} {_t}')
                    a.legend().set_visible(False)
            elif isinstance(ax, collections.Iterable):
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
                print('Update exa before audit plot')
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

    def add_exa(self, df, details, ft_nots=None):
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

        * exleaUC = np.cumsum(port.density_df['exeqa_' + col] * port.density_df.p_total)  # unconditional
        * exixgtaUC =np.cumsum(  self.density_df.loc[::-1, 'exeqa_' + col] / self.density_df.loc[::-1, 'loss']
          * self.density_df.loc[::-1, 'p_total'] )
        * exa = exleaUC + exixgtaUC * self.density_df.loss

        :param df: data frame to add to. Initially add_exa was only called by update and wrote to self.density_df. But now
        it is called by gradient too which writes to gradient_df, so we need to pass in this argument
        :param details: True = include everything; False = do not include junk around epd etc

        :param ft_nots: FFTs of the not lines (computed in gradients) so you don't round trip an FFT; gradients needs
        to recompute all the not lines each time around and it is stilly to do that twice

        """

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

        # eps is used NOT to do silly things when x is so small F(x)< eps
        # below this percentile you do not try to condition on the event!
        # np.finfo(np.float).eps = 2.2204460492503131e-16
        cut_eps = np.finfo(np.float).eps

        # get this done
        # defuzz(self.density_df, cut_eps)

        # bucket size
        bs = self.bs  # self.density_df.loc[:, 'loss'].iloc[1] - self.density_df.loc[:, 'loss'].iloc[0]
        # index has already been reset

        # sum of p_total is so important...we will rescale it...
        if not np.all(df.p_total >= 0):
            # have negative densities...get rid of them
            first_neg = df.query('p_total < 0')  #  np.argwhere((df.p_total < 0).to_numpy()).min()
            logger.warning(
                f'CPortfolio.add_exa | p_total has a negative value starting at {first_neg.head()}; NOT setting to zero...')
            # TODO what does this all mean?!
            # df.p_total.iloc[first_neg:] = 0
        sum_p_total = df.p_total.sum()
        logger.info(f'CPortfolio.add_exa | {self.name}: sum of p_total is 1 - '
                    f'{1 - sum_p_total:12.8e} NOT RESCALING')
        # df.p_total /= sum_p_total
        df['F'] = np.cumsum(df.p_total)
        # and this, ladies and gents, is terrible...
        # df['S'] = 1 - df.F
        df['S'] = np.hstack((df.p_total.to_numpy()[:0:-1].cumsum()[::-1],
                             min(df.p_total.iloc[-1],
                                 max(0, 1. - (df.p_total.sum())))))
        # get rounding errors, S may not go below zero
        logger.info(
            f'CPortfolio.add_exa | {self.name}: S <= 0 values has length {len(np.argwhere((df.S <= 0).to_numpy()))}')

        # E(min(X, a))
        # needs to be shifted down by one for the partial integrals....
        # temp = np.hstack((0, np.array(df.iloc[:-1, :].loc[:, 'S'].cumsum())))
        # df['exa_total'] = temp * bs
        df['exa_total'] = self.cumintegral(df['S'])
        df.loc[:, 'lev_total'] = df['exa_total']

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
        # epds for total on a stand alone basis (all that makes sense)
        df.loc[:, 'epd_0_total'] = \
            np.maximum(0, (df.loc[:, 'e_total'] - df.loc[:, 'lev_total'])) / \
            df.loc[:, 'e_total']
        df['exgta_total'] = df.loss + (df.e_total - df.exa_total) / df.S
        df['exeqa_total'] = df.loss  # E(X | X=a) = a(!) included for symmetry was exa

        # E[1/X 1_{X>a}] used for reimbursement effectiveness graph
        index_inv = 1.0 / np.array(df.index)
        df['e1xi_1gta_total'] = (df['p_total'] * index_inv).iloc[::-1].cumsum()

        # FFT functions for use in exa calculations
        # computing sums so minimal padding required
        def loc_ft(x):
            return ft(x, 1, None)

        def loc_ift(x):
            return ift(x, 1, None)

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
                    np.real(loc_ift(loc_ft(df.loss * df['p_' + col]) *
                                    loc_ft(df['ημ_' + col]))) / df.p_total
            else:
                df['exeqa_' + col] = \
                    np.real(loc_ift(loc_ft(df.loss * df['p_' + col]) *
                                    ft_nots[col])) / df.p_total
            # these are unreliable estimates because p_total=0 JUNE 25: this makes a difference!
            df.loc[df.p_total < cut_eps, 'exeqa_' + col] = 0
            df['exeqa_ημ_' + col] = \
                np.real(loc_ift(loc_ft(df.loss * df['ημ_' + col]) *
                                loc_ft(df['p_' + col]))) / df.p_total
            # these are unreliable estimates because p_total=0 JUNE 25: this makes a difference!
            df.loc[df.p_total < cut_eps, 'exeqa_ημ_' + col] = 0
            # E(X_{i, 2nd priority}(a))
            # need the stand alone LEV calc
            # E(min(Xi, a)
            # needs to be shifted down by one for the partial integrals....
            stemp = 1 - df.loc[:, 'p_' + col].cumsum()
            # temp = np.hstack((0, stemp.iloc[:-1].cumsum()))
            # df['lev_' + col] = temp * bs
            df['lev_' + col] = self.cumintegral(stemp)
            if details:
                df['e2pri_' + col] = \
                    np.real(loc_ift(loc_ft(df['lev_' + col]) * loc_ft(df['ημ_' + col])))
            stemp = 1 - df.loc[:, 'ημ_' + col].cumsum()
            # temp = np.hstack((0, stemp.iloc[:-1].cumsum()))
            # df['lev_ημ_' + col] = temp * bs
            df['lev_ημ_' + col] = self.cumintegral(stemp)

            # EX_i | X<= a; temp is used in le and gt calcs
            temp = np.cumsum(df['exeqa_' + col] * df.p_total)
            df['exlea_' + col] = temp / df.F
            # revised version for small losses: do not know this value
            df.loc[0:loss_max, 'exlea_' + col] = 0  # df.loc[0:loss_max, 'loss']
            temp_not = np.cumsum(df['exeqa_ημ_' + col] * df.p_total)
            df['exlea_ημ_' + col] = temp_not / df.F
            # revised version for small losses: do not know this value
            df.loc[0:loss_max, 'exlea_ημ_' + col] = 0  # df.loc[0:loss_max, 'loss']

            # constant value, helpful in calculations
            df['e_' + col] = np.sum(df['p_' + col] * df.loss)
            df['e_ημ_' + col] = np.sum(df['ημ_' + col] * df.loss)

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
            # put value back
            df.loss.iloc[0] = temp
            # this is so important we will calculate it directly rather than the old:
            # df['exi_xgta_' + col] = (df['exi_x_' + col] - temp_xi_x) / df.S
            # the last value is undefined because we know nothing about what happens beyond our array
            # above that we need a shift: > second to last value will only involve the last row (the John Major problem)
            # hence
            df['exi_xgta_' + col] = ((df[f'exeqa_{col}'] / df.loss *
                                      df.p_total).shift(-1)[
                                     ::-1].cumsum()) / df.S
            # need this NOT to be nan otherwise exa won't come out correctly
            df.loc[Seq0, 'exi_xgta_' + col] = 0.
            # df['exi_xgta_ημ_' + col] = \
            #     (df['exi_x_ημ_' + col] - temp_xi_x_not) / df.S
            # as for line
            df['exi_xgta_ημ_' + col] = ((df[f'exeqa_ημ_{col}'] / df.loss *
                                         df.p_total).shift(-1)[
                                        ::-1].cumsum()) / df.S
            df.loc[Seq0, 'exi_xgta_ημ_' + col] = 0.
            df['exi_xeqa_' + col] = df['exeqa_' + col] / df['loss']
            df.loc[0, 'exi_xeqa_' + col] = 0
            df['exi_xeqa_ημ_' + col] = df['exeqa_ημ_' + col] / df['loss']
            df.loc[0, 'exi_xeqa_ημ_' + col] = 0
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
            df['exa_ημ_' + col] = (df.S * df['exi_xgta_ημ_' + col]).shift(1, fill_value=0).cumsum() * self.bs

            # E[1/X 1_{X>a}] used for reimbursement effectiveness graph
            df[f'e1xi_1gta_{col}'] = (df[f'p_{col}'] * index_inv).iloc[::-1].cumsum()

            if details:
                # epds
                df.loc[:, 'epd_0_' + col] = \
                    np.maximum(0, (df.loc[:, 'e_' + col] - df.loc[:, 'lev_' + col])) / \
                    df.loc[:, 'e_' + col]
                df.loc[:, 'epd_0_ημ_' + col] = \
                    np.maximum(0, (df.loc[:, 'e_ημ_' + col] - df.loc[:, 'lev_ημ_' + col])) / \
                    df.loc[:, 'e_ημ_' + col]
                df.loc[:, 'epd_1_' + col] = \
                    np.maximum(0, (df.loc[:, 'e_' + col] - df.loc[:, 'exa_' + col])) / \
                    df.loc[:, 'e_' + col]
                df.loc[:, 'epd_1_ημ_' + col] = \
                    np.maximum(0, (df.loc[:, 'e_ημ_' + col] -
                                   df.loc[:, 'exa_ημ_' + col])) / \
                    df.loc[:, 'e_ημ_' + col]
                df.loc[:, 'epd_2_' + col] = \
                    np.maximum(0, (df.loc[:, 'e_' + col] - df.loc[:, 'e2pri_' + col])) / \
                    df.loc[:, 'e_' + col]

                # epd interpolation functions
                # capital and epd functions: for i = 0 and 1 we want line and not line
                loss_values = df.loss.values
                for i in [0, 1, 2]:
                    epd_values = -df.loc[:, 'epd_{:}_{:}'.format(i, col)].values
                    # if np.any(epd_values[1:] <= epd_values[:-1]):
                    #     print(i, col)
                    #     print( 1e12*(epd_values[1:][epd_values[1:] <= epd_values[:-1]] -
                    #       epd_values[:-1][epd_values[1:] <= epd_values[:-1]]))
                    # raise ValueError('Need to be sorted ascending')
                    self.epd_2_assets[(col, i)] = minus_arg_wrapper(
                        interpolate.interp1d(epd_values, loss_values, kind='linear', assume_sorted=True,
                                             fill_value='extrapolate'))
                    self.assets_2_epd[(col, i)] = minus_ans_wrapper(
                        interpolate.interp1d(loss_values, epd_values, kind='linear', assume_sorted=True,
                                             fill_value='extrapolate'))
                for i in [0, 1]:
                    epd_values = -df.loc[:, 'epd_{:}_ημ_{:}'.format(i, col)].values
                    self.epd_2_assets[('not ' + col, i)] = minus_arg_wrapper(
                        interpolate.interp1d(epd_values, loss_values, kind='linear', assume_sorted=True,
                                             fill_value='extrapolate'))
                    self.assets_2_epd[('not ' + col, i)] = minus_ans_wrapper(
                        interpolate.interp1d(loss_values, epd_values, kind='linear', assume_sorted=True,
                                             fill_value='extrapolate'))

        # put in totals for the ratios... this is very handy in later use
        for metric in ['exi_xlea_', 'exi_xgta_', 'exi_xeqa_']:
            df[metric + 'sum'] = df.filter(regex=metric + '[^η]').sum(axis=1)

        if details:
            epd_values = -df.loc[:, 'epd_0_total'].values
            # if np.any(epd_values[1:] <= epd_values[:-1]):
            #     print('total')
            #     print(epd_values[1:][epd_values[1:] <= epd_values[:-1]])
            # raise ValueError('Need to be sorted ascending')
            loss_values = df.loss.values
            self.epd_2_assets[('total', 0)] = minus_arg_wrapper(
                interpolate.interp1d(epd_values, loss_values, kind='linear', assume_sorted=True,
                                     fill_value='extrapolate'))
            self.assets_2_epd[('total', 0)] = minus_ans_wrapper(
                interpolate.interp1d(loss_values, epd_values, kind='linear', assume_sorted=True,
                                     fill_value='extrapolate'))

    def calibrate_distortion(self, name, r0=0.0, df=5.5, premium_target=0.0,
                             roe=0.0, assets=0.0, p=0.0, kind='lower', S_column='S'):
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
        if S_column == 'S':
            if assets == 0:
                assert (p > 0)
                assets = self.q(p, kind)

            # figure premium target
            if premium_target == 0:
                assert (roe > 0)
                # expected losses with assets
                el = self.density_df.loc[assets, 'exa_total']
                premium_target = (el + roe * assets) / (1 + roe)
        else:
            # if assets not entered, calibrating to unlimited premium; set assets = max loss and let code trim it
            if assets == 0:
                assets = self.density_df.loss.iloc[-1]

        # extract S and trim it: we are doing int from zero to assets
        # integration including ENDpoint is
        Splus = self.density_df.loc[0:assets, S_column].values
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
            logger.warning(
                'CPortfolio.calibrate_distortion | Mass issues in calibrate_distortion...'
                f'{name} at {last_non_zero}, loss = {ess_sup}')
        else:
            S = self.density_df.loc[0:assets - self.bs, S_column].values

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
            shape = 2.0  # starting parameter
            S = S[S < 1]
            lS = -np.log(1 - S)  # prob a bunch of zeros...

            # lS[0] = 0  # ??

            def f(rho):
                temp = (1 - S) ** rho
                trho = 1 - temp
                ex = np.sum(trho) * self.bs
                ex_prime = np.sum(temp * lS) * self.bs
                return ex - premium_target, ex_prime
        else:
            raise ValueError(f'calibrate_distortion not implemented for {name}')

        # numerical solve
        i = 0
        fx, fxp = f(shape)
        while abs(fx) > 1e-5 and i < 20:
            shape = shape - fx / fxp
            fx, fxp = f(shape)
            i += 1

        if abs(fx) > 1e-5:
            logger.warning(
                f'CPortfolio.calibrate_distortion | Questionable convergenge! {name}, target '
                f'{premium_target} error {fx}, {i} iterations')

        # build answer
        dist = Distortion(name=name, shape=shape, r0=r0, df=df)
        dist.error = fx
        dist.assets = assets
        dist.premium_target = premium_target
        return dist

    def calibrate_distortions(self, LRs=None, ROEs=None, As=None, Ps=None, kind='lower', r0=0.03, df=5.5,
                              strict=True, return_distortions=False):
        """
        Calibrate assets a to loss ratios LRs and asset levels As (iterables)
        ro for LY, it :math:`ro/(1+ro)` corresponds to a minimum rate on line


        :param LRs:  LR or ROEs given
        :param ROEs: ROEs override LRs
        :param As:  Assets or probs given
        :param Ps: probability levels for quantiles
        :param r0: for distortions that have a min ROL
        :param df: for tt
        :param strict: if True only use distortions with no mass at zero, otherwise
                        use anything reasonable for pricing
        :param return_distortions: return the created distortions
        :return:
        """
        ans = pd.DataFrame(
            columns=['$a$', 'LR', '$S$', '$\\iota$', '$\\delta$', '$\\nu$', '$EL$', '$P$', 'Levg', '$K$',
                     'ROE', 'param', 'error', 'method'], dtype=np.float)
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
                for dname in Distortion.available_distortions(pricing=True, strict=strict):
                    dist = self.calibrate_distortion(name=dname, r0=r0, df=df, premium_target=P, assets=a)
                    dists[dname] = dist
                    ans.loc[(a, lr, dname), :] = [S, iota, delta, nu, exa, P, P / K, K, profit / K,
                                                  dist.shape, dist.error]
        # very helpful to keep these...
        self.dist_ans = ans
        self.dists = dists
        if return_distortions:
            logger.warning('Get rid of usage of return_distortions!! Now available as member dfs')
            return ans, dists
        else:
            return ans

    def apply_distortions(self, dist_dict, As=None, Ps=None, kind='lower', axiter=None, num_plots=1):
        """
        Apply a list of distortions, summarize pricing and produce graphical output
        show loss values where  :math:`s_ub > S(loss) > s_lb` by jump

        :param dist_dict: dictionary of Distortion objects
        :param As: input asset levels to consider OR
        :param Ps: input probs (near 1) converted to assets using ``self.q()``
        :param num_plots: 0, 1 or 2
        :return:
        """
        ans = []
        if As is None:
            As = np.array([float(self.q(p, kind)) for p in Ps])

        if num_plots == 2 and axiter is None:
            axiter = axiter_factory(None, len(dist_dict))
        elif num_plots == 3 and axiter is None:
            axiter = axiter_factory(None, 30)
        else:
            pass

        for g in dist_dict.values():
            _x = self.apply_distortion(g, axiter, num_plots)
            df = _x.augmented_df
            # extract range of S values
            temp = df.loc[As, :].filter(regex='^loss|^S|exa[g]?_[^η][\.:~a-zA-Z0-9]*$|exag_sumparts|lr_').copy()
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

        # by line columns=method x capital
        if num_plots >= 1:
            sns.catplot(x='line', y='value', row='return', col='method', height=2.5, kind='bar',
                        data=ans_stacked.query(' stat=="lr" ')).set(ylim=(mn, mx), ylabel='LR')
            sns.catplot(x='method', y='value', row='return', col='line', height=2.5, kind='bar',
                        data=ans_stacked.query(' stat=="lr" ')).set(ylim=(mn, mx), ylabel='LR')
            # sns.factorplot(x='return', y='value', row='line', col='method', size=2.5, kind='bar',
            #                data=ans_stacked.query(' stat=="lr" ')).set(ylim=(mn, mx))

        return ans_table, ans_stacked

    def apply_distortion(self, dist, plots=None, df_in=None, create_augmented=True, mass_hints=None,
                         S_calculation='forwards'):
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

        The issue with distortions that have a mass at 0
        ================================================

        exag is computed as the cumulative integral of beta g(S), which is unaffected by the mass.

        But beta = exi_xgtag is computed as (cumint exeqa / loss * gp_total) / gS which includes the point at infinity
        or sup loss. Since gp_total is computed as a difference of gS it does not see the mass ( and it will
        sum to 1 - mass). Therefore we need to add a term "E[X_i/X | X=sup loss or infty] mass" to the cumint term
        for all loss (upper limits of integration) <= ess sup(X).
        To do so requires an estimate of E[X_i/X | X=sup loss or infty], which is hard because the estimates tend to
        become unstable in the right tail and we only have estimates upto the largest modeled value, not actually
        infinity. In order to circumvent this difficulty introduce the mass_hint variable where you specifically
        input estimates of the values. mass_hint is optional. Make a estimate_exi_x function to come up with plausible
        estimates that are used if no hint is given.

        Note that if the largest loss really is infinite then exag will be infinite when there is a mass.

        It also requires identifying the sup loss, i.e. the largest loss with S>0. np.searchsorted can be used for
        the latter. It needs care but is not really difficult.

        The binding constraint is accurate estimation of E[X_i | X]. We will truncate augmented_df at the point that
        the sum exeqa is < loss - epsilon. In order to compute exeqa you must have S>0, so it will always be the case
        that the tail is "missing" and you need to add the mass, if there is one.

        :type create_augmented: object
        :param dist: agg.Distortion
        :param plots: iterable of plot types
        :param df_in: when called from gradient you want to pass in gradient_df and use that; otherwise use self.density_df
        :param create_augmented: store the output in self.augmented_df
        :param mass_hints: hints for values of E[X_i/X | X=x] as x --> infty (optional; if not entered estimated using
                self.estimate_exi_x(). Mass hints needs to have a get-item method so that mass_hints[line] is the hint
                for line. e.g. it can be a dictionary or pandas Series indexed by line names. mass_hints must sum to
                1 over all lines. Total line obviously excluded.
        :param S_calculation: if forwards, recompute S summing p_total forwards...this gets the tail right; the old method was
                backwards, which does not change S
        :return: density_df with extra columns appended
        """

        # initially work will "full precision"
        if df_in is None:
            df = self.density_df.copy()
        else:
            df = df_in

        # PREVIOUSLY: did not make this adjustment because loss of resolution on small S values
        # however, it doesn't work well for very thick tailed distributions, hence intro of S_calculation
        # July 2020 (COVID-Trump madness) try this instead
        if S_calculation == 'forwards':
            logger.warning('Using updated S_forwards calculation in apply_distortion! ')
            df['S'] = 1 - df.p_total.cumsum()

        # make g and ginv and other interpolation functions
        g, g_inv = dist.g, dist.g_inv

        # maybe important that p_total sums to 1
        # this appeared not to make a difference, and introduces an undesirable difference from
        # the original density_df
        # df.loc[df.p_total < 0, :] = 0
        # df['p_total'] = df['p_total'] / df['p_total'].sum()
        # df['F'] = df.p_total.cumsum()
        # df['S'] = 1 - df.F

        # floating point issues: THIS HAPPENS, so needs to be corrected...
        if len(df.loc[df.S < 0, 'S'] > 0):
            logger.warning(f"{len(df.loc[df.S < 0, 'S'] > 0)} negative S values being set to zero...")
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

        # Impact of mass at zero
        # if total has an ess sup < top of computed range then any integral a > ess sup needs to have
        # the mass added. It only needs to be added to quantities computed as integrals, not g(S(x))
        # which includes it automatically.
        # total_mass is the mass for the total line, it will be apportioned below to the individual lines
        # total_mass  = (mass %) x (ess sup loss), applied when S>0
        # TODO TODO Audit and check reasonable results!

        # figure out where to truncate df (which turns into augmented_df)
        lnp = '|'.join(self.line_names)
        exeqa_err = np.abs(
            (df.filter(regex=f'exeqa_({lnp})').sum(axis=1) - df.loss) / df.loss)
        exeqa_err.iloc[0] = 0
        # print(exeqa_err)
        idx = int(exeqa_err[exeqa_err < 1e-4].index[-1] / self.bs + 1)
        # idx = np.argmax(exeqa_err > 1e-4)
        if idx:
            # if exeqa_err > 1e-4 is empty, np.argmax returns zero...do not want to truncate at zero in that case
            df = df.iloc[:idx, :]
        # where S==0, which should be empty set
        gSeq0 = (df.gS == 0)
        if idx:
            print(f'Truncating augmented_df at idx={idx}, loss={idx*self.bs}\nS==0 on len(S==0) = {np.sum(gSeq0)} elements')
        else:
            print(f'augmented_df not truncated (no exeqa error\nS==0 on len(S==0) = {np.sum(gSeq0)} elements')

        # this should now apply in ALL mass situations...
        total_mass = 0
        if dist.mass:
            S = df.S
            # S better be decreasing
            if not np.alltrue(S.iloc[1:] <= S.iloc[:-1].values):
                logger.error('S = density_df.S is not non-increasing...carrying on but you should investigate...')
            idx_ess_sup = S.to_numpy().nonzero()[0][-1]
            logger.warning(f'Index of ess_sup is {idx_ess_sup}')
            print(f'Index of ess_sup is {idx_ess_sup}')
            total_mass = np.zeros_like(S)
            # locations and amount of mass to be added to computation of exi_xgtag below...
            total_mass[:idx_ess_sup + 1] = dist.mass
            print(total_mass)

        if dist.mass and mass_hints is None:
            mass_hints = pd.Series(index=self.line_names)
            for line in self.line_names:
                logger.warning(f'Individual line={line}')
                print(f'Individual line={line}')
                mass_hints[line] = df[f'exi_xeqa_{line}'].iloc[-1]
                for ii in range(1, max(self.log2 - 4, 0)):
                    avg_xix = df[f'exi_xeqa_{line}'].iloc[idx_ess_sup - (1 << ii):].mean()
                    logger.warning(f'Avg weight last {1 << ii} observations is  = {avg_xix:.5g} vs. last '
                                   f'is {self.density_df[f"exi_xeqa_{line}"].iloc[idx_ess_sup]:.5g}')
                    print(f'Avg weight last {1 << ii} observations is  = {avg_xix:.5g} vs. last '
                                   f'is {self.density_df[f"exi_xeqa_{line}"].iloc[idx_ess_sup]:.5g}')
                logger.warning('You want these values all to be consistent!')
            print(f'Estimated mass_hints is {mass_hints}')

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
            # Treatment of Masses
            # You are integrating down from infinity. You need to add the mass at the first
            # loss level with S>0, or, if S>0 for all x then you add the mass at all levels because
            # int_x^infty fun = int_F(x)^1 fun + (prob=1) * fun(1) (fun=TVaR)
            # The mass is (dist.mass x ess sup).
            # if S>0 but flat and there is a mass then need to include loss X g(S(loss)) term!
            # pick  up right hand places where S is very small (rounding issues...)
            # OLD
            #
            # mass = 0
            # if dist.mass:
            #     logger.error("You cannot use a distortion with a mass at this time...")
            #     # this is John's problem: the last scenario is getting all the weight...
            #     # not clear how accurately this is computed numerically
            #     # this amount is a
            #     mass = total_mass * self.density_df[f'exi_xeqa_{line}'].iloc[idx_ess_sup]
            #     logger.warning(f'Individual line={line} weight from portfolio mass = {mass}')
            #     for ii in range(1, max(self.log2 - 4, 0)):
            #         avg_xix = self.density_df[f'exi_xeqa_{line}'].iloc[idx_ess_sup - (1 << ii):].mean()
            #         logger.warning(f'Avg weight last {1 << ii} observations is  = {avg_xix:.5g} vs. last '
            #                        f'is {self.density_df[f"exi_xeqa_{line}"].iloc[idx_ess_sup]}:.5g')
            #     logger.warning('You want these values all to be consistent!')
            # old
            # mass = dist.mass * self.density_df.loss * self.density_df[f'exi_xeqa_{line}'] * \
            #        np.where(self.density_df.S > 0, 1, 0)

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
            #
            # df[f'exag_{line}'] = exleaUC + exixgtaUC * self.density_df.loss + mass
            # df[f'exag1_{line}'] = exleaUC1 + exixgtaUC * self.density_df.loss + mass
            # is this ever used?
            # df[f'exleag_{line}'] = exleaUC / df.gF
            # df[f'exleag1_{line}'] = exleaUC1 / df.gF
            #
            # again, exi_xgtag is super important, so we will compute it bare bones the same way as exi_xgta
            # df[f'exi_xgtag_{line}'] = exixgtaUC / df.gS
            #
            #
            # df['exi_xgtag_' + line] = ((df[f'exeqa_{line}'] / df.loss *
            #                             df.gp_total).shift(-1)[::-1].cumsum()) / df.gp_total.shift(-1)[::-1].cumsum()
            # exa uses S in the denominator...and actually checking values there is a difference between the sum and gS
            #
            # May 2020 new treatment of masses
            #
            mass = 0
            ημ_mass = 0
            if dist.mass:
                # this is John's problem: the last scenario is getting all the weight...
                # total_mass is a vector [mass,...,mass,0,...] upto the ess sup X
                # only need to add masses up to this point - an issue for bounded variables
                # mass = total_mass * self.density_df[f'exi_xeqa_{line}'].iloc[idx_ess_sup]
                # TODO Is indexing added so it aligns with what it is added to?
                mass = total_mass * mass_hints[line]
                ημ_mass = total_mass * (np.sum(mass_hints) - mass_hints[line])
            # original
            # df['exi_xgtag_' + line] = ((df[f'exeqa_{line}'] / df.loss *
            #             df.gp_total).shift(-1)[::-1].cumsum()) / df.gS
            # df['exi_xgtag_ημ_' + line] = ((df[f'exeqa_ημ_{line}'] / df.loss *
            #                                df.gp_total).shift(-1)[::-1].cumsum()) / df.gS
            df['exi_xgtag_' + line] = \
                ((df[f'exeqa_{line}'] / df.loss * df.gp_total).shift(-1)[::-1].cumsum() + mass) / df.gS
            df['exi_xgtag_ημ_' + line] = \
                ((df[f'exeqa_ημ_{line}'] / df.loss * df.gp_total).shift(-1)[::-1].cumsum() + ημ_mass) / df.gS
            # need these to be zero so nan's do not propogate
            df.loc[gSeq0, 'exi_xgtag_' + line] = 0.
            df.loc[gSeq0, 'exi_xgtag_ημ_' + line] = 0.
            #
            #
            # following the Audit Vignette this is the way to go:
            # in fact, need to shift both down because cumint (prev just gS, but int beta g...integrands on same
            # basis
            df[f'exag_{line}'] = (df[f'exi_xgtag_{line}'].shift(1) * df.gS.shift(1)).cumsum() * self.bs
            df[f'exag_ημ_{line}'] = (df[f'exi_xgtag_ημ_{line}'].shift(1) * df.gS.shift(1)).cumsum() * self.bs
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
        # sum of parts: careful not to include the total twice!
        df['exag_sumparts'] = df.filter(regex='^exag_[^η]').sum(axis=1)
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
        # critical insight is the layer ROEs are the same for all lines (law invariance)
        df['M.ROE_total'] = df['M.M_total'] / df['M.Q_total']
        # where is the ROE zero? need to handle separately else Q will blow up
        roe_zero = (df['M.ROE_total'] == 0.0)
        for line in self.line_names_ex:
            df[f'exa_{line}_pcttotal'] = df['exa_' + line] / df.exa_total
            df[f'exag_{line}_pcttotal'] = df['exag_' + line] / df.exag_total
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
        # TODO investiage why exag calc less accurate than exa calc
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
                ax.plot(df.exag_sumparts, label='Sum of Parts')
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
                df_plot.filter(regex='^exa_.*_pcttotal$').sort_index(axis=1).plot(ax=ax)
                ax.set_title('Proportion of loss: T.L_line / T.L_total')
                ax.set_ylim(0, 1.05)
                ax.legend(loc='upper left')
                ax.grid()

                ax = next(axiter)
                df_plot.filter(regex='^exag_.*_pcttotal$').sort_index(axis=1).plot(ax=ax)
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

    def var_dict(self, p, kind):
        """
        make a dictionary of value at risks for each line and the whole portfolio.

         Returns: {line : var(p, kind)} and includes the total as self.name line

        :param p:
        :param kind:
        :return:
        """
        var_dict = {a.name: a.q(p, kind) for a in self.agg_list}
        var_dict[self.name] = self.q(p, kind)
        return var_dict

    def gamma(self, a=None, p=None, kind='', plot=False, compute_stand_alone=False, three_plot_xlim=-1,
              ylim_zoom=(1, 1e3), extreme_var=1 - 2e-8):
        """
        Return the vector gamma_a(x), the conditional layer effectiveness given assets a.
        Assets specified by percentile level and type (you need a in the index too hard to guess?)
        gamma can be created with no base and no calibration - it does not depend on a distortion.
        It only depends on total losses.

        Returns the total and by layer versions, see
        "Main Result for Conditional Layer Effectiveness; Piano Diagram" in OneNote

        Originally in aggregate_extensions...but only involves one portfolio so should be in agg
        Note that you need upper and lower q's in aggs now too.

        :param a:     input a or p and kind as ususal
        :param p:     asset level percentile
        :param kind:  lower or upper
        :param plot:
        :param compute_stand_alone:
        :param three_plot_xlim:        if >0 do the three plot xlim=[0, loss_threshold] to show details
        :param ylim_zoom: now on a return period, 1/(1-p) basis so > 1
        :param extreme_var:
        :return:
        """

        if a is None:
            assert p is not None
            a = self.q(p, kind)
        else:
            p = self.cdf(a)
            kind = 'lower'
            ql = self.q(p, 'lower')
            qu = self.q(p, 'upper')
            logger.warning(
                f'Input a={a} to gamma; computed p={p:.8g}, lower and upper quantiles are {ql:.8g} and {qu:.8g}')

        # alter in place or return a copy? For now return a copy...
        temp = self.density_df.filter(regex='^p_|^e1xi_1gta_|exi_xgta_|exi_xeqa_|exeqa_|S|loss').copy()

        # var dictionaries
        var_dict = self.var_dict(p, kind)
        extreme_var_dict = self.var_dict(extreme_var, kind)

        # rate of payment factor
        min_xa = np.minimum(temp.loss, a) / temp.loss
        temp['min_xa_x'] = min_xa

        # total is a special case
        ln = 'total'
        gam_name = f'gamma_{self.name}_{ln}'
        # unconditional version avoid divide and multiply by a small number
        # exeqa is closest to the raw output...
        # there maybe an INDEX ISSUE with add_exa...there are cumsums there that should be cumintegrals..
        # exeqa_total = loss of course...
        # ?!! WTF surely this
        temp[f'exi_x1gta_{ln}'] = np.cumsum((temp[f'loss'] * temp.p_total / temp.loss)[::-1]) * self.bs
        # equals this temp.S?!
        s_ = np.cumsum(temp.p_total[::-1]) * self.bs
        print('TEMP: the following should be close: ', np.allclose(s_[::-1], temp[f'exi_x1gta_{ln}']),
                np.allclose(temp[f'exi_x1gta_{ln}'], temp.S * self.bs))
        #                               this V 1.0 is exi_x for total
        temp[gam_name] = np.cumsum((min_xa * 1.0 * temp.p_total)[::-1]) / \
                         (temp[f'exi_x1gta_{ln}']) * self.bs

        for ln in self.line_names:
            # sa = stand alone; need to figure the sa capital from the agg objects hence var_dict
            if compute_stand_alone:
                a_l = var_dict[ln]
                a_l_ = a_l - self.bs
                xinv = temp[f'e1xi_1gta_{ln}'].shift(-1)
                gam_name = f'gamma_{ln}_sa'
                s = 1 - temp[f'p_{ln}'].cumsum()
                temp[f'S_{ln}'] = s
                temp[gam_name] = 0
                temp.loc[0:a_l_, gam_name] = (s[0:a_l_] - s[a_l] + a_l * xinv[a_l]) / s[0:a_l_]
                temp.loc[a_l:, gam_name] = a_l * xinv[a_l:] / s[a_l:]
                temp[gam_name] = temp[gam_name].shift(1)
                # method unreliable in extreme right tail
                temp.loc[extreme_var_dict[ln]:, gam_name] = np.nan

            # actual computation for l within portfolio allowing for correlations
            gam_name = f'gamma_{self.name}_{ln}'
            # unconditional version avoid divide and multiply by a small number
            # exeqa is closest to the raw output...
            # there is an INDEX ISSUE with add_exa...there are cumsums there that should be cumintegrals..
            temp[f'exi_x1gta_{ln}'] = np.cumsum((temp[f'exeqa_{ln}'] * temp.p_total / temp.loss)[::-1]) * self.bs
            temp[gam_name] = np.cumsum((min_xa * temp[f'exi_xeqa_{ln}'] * temp.p_total)[::-1]) / \
                             (temp[f'exi_x1gta_{ln}']) * self.bs

        # other stuff that is easy to ignore but helpful for plotting
        temp['BEST'] = 1
        temp.loc[a:, 'BEST'] = a / temp.loc[a:, 'loss']
        temp['WORST'] = np.maximum(0, 1 - temp.loc[a, 'S'] / temp.S)

        f = spl = None
        if plot:
            renamer = self.renamer
            # rename if doing the second set of plots too
            if three_plot_xlim > 0:
                spl = temp.filter(regex='gamma|BEST|WORST').rename(columns=renamer).sort_index(1). \
                    plot(ylim=[-.05, 1.05], linewidth=1)
            else:
                spl = temp.filter(regex='gamma|BEST|WORST').sort_index(1).plot(ylim=[-.05, 1.05], linewidth=1)
            for ln in spl.lines:
                lbl = ln.get_label()
                if lbl == lbl.upper():
                    ln.set(linewidth=1, linestyle='--', label=lbl.title())
            spl.grid('b')
            if three_plot_xlim > 0:
                spl.set(xlim=[0, three_plot_xlim])
            # update to reflect changes to line styles
            spl.legend()

            # fancy comparison plots
            if three_plot_xlim > 0 and compute_stand_alone:
                temp_ex = temp.query(f'loss < {three_plot_xlim}').filter(regex='gamma_|p_')
                # tester
                # t = grt.test_df(12)
                # t[['D', 'E', 'F']] = t[['A', 'B', 'C']].shift(-1, fill_value=0).iloc[::-1].cumsum()
                temp_ex[[f'S_{l[2:]}' for l in temp_ex.filter(regex='p_').columns]] = \
                    temp_ex.filter(regex='p_').shift(-1, fill_value=0).iloc[::-1].cumsum()

                f, axs = plt.subplots(3, 1, figsize=(8, 9), squeeze=False, constrained_layout=True)
                # ax1 = return period scale, ax2 = linear scale, ax3 = survival functions
                ax1 = axs[0, 0]
                ax2 = axs[1, 0]
                ax3 = axs[2, 0]

                btemp = temp_ex.filter(regex='gamma_').rename(columns=renamer).sort_index(axis=1)
                # first two plots same except for scales
                (1 / (1 - btemp.iloc[1:])).plot(logy=True, ylim=ylim_zoom, ax=ax1)
                btemp.plot(ax=ax2)
                ax1.set(ylim=ylim_zoom, title='Gamma Functions (Return Period Scale)')
                ax2.set(ylim=[0, 1.005], title='Gamma Functions')

                temp_ex.filter(regex='S_').rename(columns=renamer).sort_index(axis=1).plot(ax=ax3)
                # sort out colors
                col_by_line = {}
                for l in ax3.lines:
                    ls = l.get_label().split(' ')
                    col_by_line[ls[0]] = l.get_color()
                l_loc = dict(zip(axs.flatten(), ['upper right', 'lower left', 'upper right']))
                ax3.set(ylim=[0, 1.005], title=f'Survival Functions up to p={self.cdf(three_plot_xlim):.1%} Loss')
                for ax in axs.flatten():
                    # put in asset levels
                    for l in self.line_names + [self.name]:
                        ln = ax.plot([var_dict[l], var_dict[l]], [1 if ax is ax1 else 0, ylim_zoom[1]],
                                     label=f'{l} s/a assets {var_dict[l]:.0f}')
                        ln[0].set(linewidth=1, linestyle=':')
                try:
                    for ax in axs.flatten():
                        for line in ax.lines:
                            # line name and shortened line name
                            ln = line.get_label()
                            lns = ln.split(' ')
                            if ln.find('stand-alone') > 0:
                                # stand alone=> dashed line, color is lns[-2]
                                line.set(linestyle='--')
                            if lns[0] in col_by_line:
                                line.set(color=col_by_line[lns[0]])
                            elif lns[1] in col_by_line:
                                line.set(color=col_by_line[lns[1]])
                        # update legend to reflect line style changes
                        ax.legend(loc=l_loc[ax])
                        ax.grid(which='major', axis='y')
                except Exception as e:
                    logger.error(f'Errors in gamma plotting;, {e}')

        return Answer(augmented_df=temp.sort_index(axis=1), fig_gamma=spl, base=self.name,
                      fig_gamma_three_part=f, assets=a, p=p, kind=kind)

    def price(self, reg_g, pricing_g=None, method='apply_distortion'):
        """
        Price using regulatory and pricing g functions
            Compute E_price (X wedge E_reg(X) ) where E_price uses the pricing distortion and E_reg uses
            the regulatory distortion

            regulatory capital distortion is applied on unlimited basis: ``reg_g`` can be:

            * if input < 1 it is a number interpreted as a p value and used to determine VaR capital
            * if input > 1 it is a directly input  capital number
            * d dictionary: Distortion; spec { name = dist name | var | epd, shape=p value a distortion used directly

            ``pricing_g`` is  { name = ph|wang and shape= or lr= or roe= }, if shape and lr or roe shape is overwritten

            if ly it must include ro in spec

            if lr and roe then lr is used

        :param reg_g: a distortion function spec or just a number; if >1 assets if <1 a prob converted to quantile
        :param pricing_g: spec or CDistortion class or lr= or roe =; must have name= to define spec; if CDist that is
                          used
        :param method: apply_distortion or quick; if quick does an (unaudited) calc here; ad returns more info
        :return:
        """

        # interpolation functions for distribution and inverse distribution
        F = interpolate.interp1d(self.density_df.loss, self.density_df.F, kind='linear',
                                 assume_sorted=True, bounds_error=False, fill_value='extrapolate')
        Finv = interpolate.interp1d(self.density_df.F, self.density_df.loss, kind='nearest',
                                    assume_sorted=True, fill_value='extrapolate', bounds_error=False)

        # figure regulatory assets; applied to unlimited losses
        a_reg_ix = 0
        a_reg = 0
        # note here that a distortion passes through...
        if isinstance(reg_g, float) or isinstance(reg_g, int):
            if reg_g > 1:
                a_reg = reg_g
                a_reg_ix = self.density_df.iloc[
                    self.density_df.index.get_loc(reg_g, 'ffill'), 0]
                # print(f'a_reg {a_reg} and ix {a_reg_ix}')
            else:
                a_reg = a_reg_ix = float(Finv(reg_g))
        elif isinstance(reg_g, dict):
            if reg_g['name'] == 'var':  # must be dictionary
                # given var, nearest interpolation for assets
                a_reg = a_reg_ix = float(Finv(reg_g['shape']))
            elif reg_g['name'] == 'epd':
                a_reg = float(self.epd_2_assets[('total', 0)](reg_g['shape']))
                a_reg_ix = self.density_df.iloc[
                    self.density_df.index.get_loc(a_reg, 'ffill'), 0]
            else:
                reg_g = Distortion(**reg_g)
        elif isinstance(reg_g, Distortion):
            if reg_g.name == 'tvar':
                a_reg = self.tvar(reg_g.shape)
                a_reg_ix = self.density_df.iloc[
                    self.density_df.index.get_loc(a_reg, 'ffill'), 0]
        if a_reg == 0:
            # still need to figure capital
            assert (isinstance(reg_g, Distortion))
            gS = reg_g.g(self.density_df.S)
            a_reg = self.bs * np.sum(gS)
            ix = self.density_df.index.get_loc(a_reg, method='ffill')
            a_reg_ix = self.density_df.index[ix]

        # relevant row for all statistics_df
        row = self.density_df.loc[a_reg_ix, :]

        # figure pricing distortion
        prem = 0
        if isinstance(pricing_g, Distortion):
            # just use it
            pass
        else:
            # spec as dict
            if 'lr' in pricing_g:
                # given LR, figure premium
                prem = row['exa_total'] / pricing_g['lr']
            elif 'roe' in pricing_g:
                # given roe, figure premium
                roe = pricing_g['roe']
                delta = roe / (1 + roe)
                prem = row['exa_total'] + delta * (a_reg - row['exa_total'])
            if prem > 0:
                pricing_g = self.calibrate_distortion(name=pricing_g['name'], premium_target=prem, assets=a_reg_ix)
            else:
                pricing_g = Distortion(**pricing_g)

        if method == 'apply_distortion':
            ans_ad = self.apply_distortion(pricing_g, create_augmented=False)
            aug_row = ans_ad.augmented_df.loc[a_reg_ix, :]

            # holder for the answer
            df = pd.DataFrame(columns=['line', 'a_reg', 'exa', 'exag', 'margin', 'equity'], dtype=float)
            df.columns.name = 'statistic'
            df = df.set_index('line', drop=True)

            for line in self.line_names_ex:
                df.loc[line, :] = [a_reg_ix, aug_row[f'exa_{line}'], aug_row[f'exag_{line}'],
                                   aug_row[f'T.M_{line}'], aug_row[f'T.Q_{line}']]

            df['lr'] = df.exa / df.exag
            df['profit'] = df.exag - df.exa  # duplicate
            # df.loc['total', 'ROE'] = df.loc['total', 'profit'] / (df.loc['total', 'a_reg'] - df.loc['total', 'exag'])
            df.loc['total', 'prDef'] = 1 - float(F(a_reg))
            df['pct_loss'] = df.exa / df.loc['total', 'exa']
            df['pct_prem'] = df.exag / df.loc['total', 'exag']
            df['PQ'] = df.exag / df.equity
            df['ROE'] = df.profit / df.equity

        else:
            # the original method
            # create pricing distortion functions
            g_pri, g_pri_inv = pricing_g.g, pricing_g.g_inv

            # apply pricing distortion to create pricing probs
            # pgS = g_pri(self.density_df.S)
            # pgp_total = -np.diff(np.hstack((0, pgS)))  # adjusted incremental probabilities

            # holder for the answer
            df = pd.DataFrame(columns=['line', 'a_reg', 'exa', 'exag'], dtype=float)
            df.columns.name = 'statistic'
            df = df.set_index('line', drop=True)

            # E_Q((X \wedge a)E(X_i/X|X))
            # W = np.minimum(self.density_df.loss, a_reg_ix)
            # loop through lines and add details
            # the Q measure
            # some g's will return numpy (all except ph in fact return numpy arrays)
            gS = pd.Series(g_pri(self.density_df.S), index=self.density_df.index)
            # make this into a pandas series so the indexing works the same (otherwise it is an np object)
            gp_total = -pd.Series(np.diff(np.hstack((1, gS))), index=self.density_df.index)
            mass = 0
            if pricing_g.has_mass:
                mass = pricing_g.mass
                mass *= a_reg_ix
                logger.info(f'CPortfolio.price | {self.name}, Using mass {mass}')
            for line in self.line_names:
                # int E(Xi/X| X ge x)S = int d/da exa = exa DOES NOT WORK because uses ge x, and that is pre-computed
                # using P and not Q
                # AND have issue of weight at zero applying a capacity charge ??
                # up to a_reg
                # remember loc on df includes RHS
                exag1 = np.sum(self.density_df.loc[0:a_reg_ix - self.bs, f'exeqa_{line}'] *
                               gp_total.loc[0:a_reg_ix - self.bs])
                # note: exi_xeqa = exeqa / loss
                exag2 = np.sum(self.density_df.loc[a_reg_ix:, f'exeqa_{line}'] /
                               self.density_df.loss.loc[a_reg_ix:] * gp_total.loc[a_reg_ix:]) * a_reg_ix
                exag = exag1 + exag2
                if mass > 0:
                    # need average EX_i_X for large X, which is tough to compute
                    lim_xi_x = self.density_df.loc[a_reg_ix, f'exi_xeqa_{line}']
                    exag += lim_xi_x * mass
                # exag = np.sum(W * self.density_df[f'exi_xeqa_{line}'] * pgp_total)
                df.loc[line, :] = [a_reg_ix, row[f'exa_{line}'], exag]

            # total
            line = 'total'
            # if the g fun is degenerate you have the problem of capacity charge
            # so int density does not work. have to use int S
            # apply_distortion uses df['exag_total'] = cumintegral(df['gS'], self.bs)
            # which is the same since it shifts forward
            # old
            # exag = np.sum(g_pri(self.density_df.loc[self.bs:a_reg_ix, 'S'])) * self.bs
            # new
            exag = np.sum(g_pri(self.density_df.loc[0:a_reg_ix - self.bs, 'S'])) * self.bs
            assert (np.isclose(exag, np.sum(gS.loc[0:a_reg_ix - self.bs]) * self.bs))
            df.loc[line, :] = [a_reg_ix, row[f'exa_{line}'], exag]

            # df.loc['sum', :] = df.filter(regex='^[^t]', axis=0).sum()
            df['lr'] = df.exa / df.exag
            df['profit'] = df.exag - df.exa
            df.loc['total', 'ROE'] = df.loc['total', 'profit'] / (df.loc['total', 'a_reg'] - df.loc['total', 'exag'])
            df.loc['total', 'prDef'] = 1 - float(F(a_reg))
            df['pct_loss'] = df.exa / df.loc['total', 'exa']
            df['pct_prem'] = df.exag / df.loc['total', 'exag']
            df['PQ'] = df.exag / df.a_reg
            df['ROE'] = df.profit / (df.a_reg - df.exag)
            # logger.info(f'CPortfolio.price | {self.name} portfolio pricing g {pricing_g}')
            # logger.info(f'CPortfolio.price | Capital sufficient to prob {float(F(a_reg)):7.4f}')
            # logger.info(f'CPortfolio.price | Capital quantization error {(a_reg - a_reg_ix) / a_reg:7.5f}')
            # if prem > 0:
            #     logger.info(f'CPortfolio.price | Premium calculated as {prem:18,.1f}')
            #     logger.info(f'CPortfolio.price | Pricing distortion shape calculated as {pricing_g.shape}')

        return df, pricing_g

    def analyze_distortions(self, dist_list, a=0, p=0, kind='lower', mass_hints=None):
        """
        run analyze_distortion on a range of different distortions...
        distortions must be calibrated to p-var capital level - that is how they are passed into analyze distortions

        even if you only have one dist use ads because it names the outputs better in exhibits

        could keep a lot more output...but let's see how this goes...

        :param dist_list: list of distortion names...will pull form self.dists; these are paramed per self.dist_ans
        :param p: the percentile of capital that the distortions are calibrated to
        :param kind:
        :return:
        """

        a, p = self.set_a_p(a, p)
        print(a,p)
        dfs = []
        ans = Answer()
        for k in dist_list:
            d = self.dists[k]
            logger.info(f'Running distortion {d} through analyze_distortion, p={p}...')
            if len(dfs) == 0:
                # first distortion...add the comps...these are same for all dists
                ad_ans = self.analyze_distortion(d, p=p, kind=kind, add_comps=True, mass_hints=mass_hints)
            else:
                ad_ans = self.analyze_distortion(d, p=p, kind=kind, add_comps=False, mass_hints=mass_hints)
                ad_ans.exhibit = ad_ans.exhibit.drop('Assets', axis=0, level=0)
            dfs.append(ad_ans.exhibit)
            ans[f'{k}_exhibit'] = ad_ans.exhibit
            ans[f'{k}_pricing'] = ad_ans.pricing
        ans['comp_df_full'] = pd.concat(dfs).sort_index()
        # frankly, certain columns just an irritation
        ans['comp_df'] = ans['comp_df_full'].\
            drop(index=['Assets'] + [f'Dist {d} Marginal' for d in self.dists.keys()], level=0).\
            drop(index=['EPD'], level=1)
        return ans

    def analyze_distortion(self, dname, dshape=None, dr0=.025, ddf=5.5, LR=None, ROE=None,
                           p=None, kind='lower', A=None, use_self=False, plot=False,
                           a_max_p=1-1e-8, add_comps=True, mass_hints=None):
        """

        Graphic and summary DataFrame for one distortion showing results that vary by asset levelm
        such as increasing or decreasing cumulative premium.

        Characterized by the need to know an asset level, vs. apply_distortion that produced
        values for all asset levels.

        Returns DataFrame with values upto the input asset level...differentiates from apply_distortion
        graphics that cover the full range.

        analyze_pricing will then zoom in and only look at one asset level for micro-dynamics...

        Logic of arguments:

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
        :param add_comps: add old-fashioned method comparables (included =True as default to make backwards comp.
        :param mass_hints: for analyze_distortion
        :return: various dataframes in an Answer class object

        """

        # for plotting
        a_max = self.q(a_max_p)
        a_min = self.q(1e-4)

        # setup: figure what distortion to use, apply if necessary
        if use_self:
            dist = self.distortion
            # augmented_df was called deets before, FYI
            augmented_df = self.augmented_df
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
                    logger.warning(f'a_cal:=q(p)={a_cal} is not equal to A={A} at p={p}')

            if dshape or isinstance(dname, Distortion):
                # specified distortion, fill in
                if isinstance(dname, Distortion):
                    dist = dname
                else:
                    dist = Distortion(dname, dshape, dr0, ddf)
                _x = self.apply_distortion(dist, create_augmented=False, mass_hints=mass_hints)
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
                _x = self.apply_distortion(dist, create_augmented=False, mass_hints=mass_hints)
                augmented_df = _x.augmented_df

        # deets is now the result of apply_distortion on dist...now turn to the analysis
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

        # bottom up calc is already done: it is the T. series in deets
        # marginal calc also done: M. series

        # other helpful audit values
        # keeps track of details of calc for Answer
        audit_df = pd.DataFrame(dict(stat=[p, LR, ROE, a_cal, K, dist.name, dist.shape]),
                                index=['p', 'LR', "ROE", 'a_cal', 'K', 'dname', 'dshape'])
        audit_df.loc['a'] = a_cal
        audit_df.loc['kind'] = kind

        # make the pricing summary DataFrame and exhibit: this just contains info about dist
        pricing = augmented_df.loc[[a_cal]].T. \
            filter(regex='^(T)\.(L|P|M|Q|LR|ROE|PQ)|exi_xgtag?_[a-zA-Z:]+(?<!sum)$', axis=0).copy()
        #                  ^M here to put marginal back
        pricing.index = pricing.index.str.replace('exi_xgta_(.+)$', r'M_alpha_\1').str. \
            replace('exi_xgtag_(.+)$', r'M_beta_\1').str. \
            split('_|\.', expand=True)
        pricing.index.set_names(['Method', 'stat', 'line'], inplace=True)
        pricing = pricing.sort_index(level=[0, 1, 2])

        dist_name = dist.name
        exhibit = pricing.unstack(2).copy()
        exhibit.columns = exhibit.columns.droplevel(level=0)
        exhibit.loc['Assets', :] = a_cal
        exhibit = exhibit.sort_index()
        ans = Answer(augmented_df=augmented_df, distortion=dist, audit_df=audit_df, pricing=pricing, exhibit=exhibit)
        if add_comps:
            logger.warning('Adding comps...')
            ans = self.analyze_distortion_add_comps(ans, a_cal, p, kind, ROE)
        if plot:
            ans = self.analyze_distortion_plots(ans, dist, a_cal, p, a_max, ROE, LR)
        # last minute re-naming...
        # want to drop the useless M version here too?
        ans['exhibit'] = ans.exhibit.\
            sort_index().\
            rename(index={'T': f'Dist {dist_name}', 'M': f'Dist {dist_name} Marginal'}, level=0)
        return ans

    def analyze_distortion_add_comps(self, ans, a_cal, p, kind, ROE):
        """
        make exhibit with comparison to old-fashioned methods

        :param ans:  answer containing dist and augmented_df elements
        :param a_cal:
        :param p:
        :param kind:
        :param LR:
        :param ROE:
        :return: ans Answer object with updated elements
        """

        exhibit = ans.exhibit.copy()
        # shut the style police up:
        done = []
        # some reasonable traditional metrics
        # tvar threshold giving the same assets as p on VaR
        print(f'In analyze_distortion_comps p={p} and a_cal={a_cal}')
        try:
            p_t = self.tvar_threshold(p, kind)
        except ValueError as e:
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
            done = []
            exhibit.loc[('VaR', 'Q'), :] = [float(a.q(p, kind)) for a in self] + [self.q(p, kind)]
            done.append('var')
            exhibit.loc[('TVaR', 'Q'), :] = [float(a.tvar(p_t)) for a in self] + [self.tvar(p_t)]
            done.append('tvar')
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
            exhibit.loc[('EqRiskVaR', 'Q'), :] = [float(a.q(pv, kind)) for a in self] + [self.q(p)]
            done.append('eqvar')
            exhibit.loc[('EqRiskTVaR', 'Q'), :] = [float(a.tvar(pt)) for a in self] + [self.tvar(p_t)]
            done.append('eqtvar')
            exhibit.loc[('MerPer', 'Q'), :] = self.merton_perold(p)
            done.append('merper')
            # co-tvar at threshold to match capital
            exhibit.loc[('coTVaR', 'Q'), :] = self.cotvar(p_t)
            done.append('cotvar')
        except Exception as e:
            logger.warning('Some general out of bounds error on VaRs and TVaRs, setting all equal to zero. '
                           f'Last completed steps {done[-1] if len(done) else "none"}, out of var, tvar, eqvar, eqtvar merper. ')
            logger.warning(f'The built in warning is type {type(e)} saying {e}')
            for c in ['VaR', 'TVaR', 'ScaledVaR', 'ScaledTVaR', 'EqRiskVaR', 'EqRiskTVaR', 'MerPer', 'coTVaR']:
                if c.lower() in done:
                    # these numbers have been computed
                    pass
                else:
                    # put in placeholder
                    exhibit.loc[(c, 'Q'), :] = np.nan
        # EPD
        row = self.density_df.loc[a_cal, :]
        exhibit.loc[('T', 'EPD'), :] = [row.at[f'epd_{0 if l == "total" else 1}_{l}'] for l in self.line_names_ex]
        exhibit = exhibit.sort_index()
        # subtract the premium to get the actual capital
        try:
            for m in ['VaR', 'TVaR', 'ScaledVaR', 'ScaledTVaR', 'EqRiskVaR', 'EqRiskTVaR', 'MerPer', 'coTVaR']:
                # Q as computed above gives assets not equity...so adjust
                # usual calculus: P = (L + ra)/(1+r); Q = a-P, remember Q above is a (sorry)
                exhibit.loc[(m, 'L'), :] = exhibit.loc[('T', 'L'), :]
                exhibit.loc[(m, 'P'), :] = (exhibit.loc[('T', 'L'), :] + ROE * exhibit.loc[(m, 'Q'), :]) / (1 + ROE)
                exhibit.loc[(m, 'Q'), :] -= exhibit.loc[(m, 'P'), :].values
                exhibit.loc[(m, 'M'), :] = exhibit.loc[(m, 'P'), :] - exhibit.loc[('T', 'L'), :].values
                exhibit.loc[(m, 'LR'), :] = exhibit.loc[('T', 'L'), :] / exhibit.loc[(m, 'P'), :]
                exhibit.loc[(m, 'ROE'), :] = exhibit.loc[(m, 'M'), :] / exhibit.loc[(m, 'Q'), :]
                exhibit.loc[(m, 'PQ'), :] = exhibit.loc[(m, 'P'), :] / exhibit.loc[(m, 'Q'), :]
        except Exception as e:
            print(f'Exception at the last step... {e}')
        # print(ans.distortion.name)
        # display(exhibit)
        ans.audit_df.loc['TVaR@'] = p_t
        ans.audit_df.loc['erVaR'] = pv
        ans.audit_df.loc['erTVaR'] = pt
        ans.update(exhibit=exhibit)
        return ans

    def analyze_distortion_plots(self, ans, dist, a_cal, p, a_max, ROE, LR):
        """
        Create plots fron an analyze_distortion ans class
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
            print(f'Formatting error in last plot...\n{e}\n...continuing')

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
            print(f'Plotting error in trinity plots\n{e}\nPlots done {plots_done}\n...continuing')

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
            print('Error', e)

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

    def top_down(self, distortions, A_or_p):
        """
        DataFrame summary and nice plots showing marginal and average ROE, lr etc. as you write a layer from x to A
        If A=0 A=q(log) is used

        Not integrated into graphics format (plot)

        :param distortions: list or dictionary of CDistortion objects, or a single CDist object
        :param A_or_p: if <1 interpreted as a quantile, otherwise assets
        :return:
        """

        logger.warning('Portfolio.top_down is deprecated. It has been replaced by Portfolio.example_factory.')

        # assert A_or_p > 0
        #
        # if A_or_p < 1:
        #     # call with one arg and interpret as log
        #     A = self.q(A_or_p)
        # else:
        #     A = A_or_p
        #
        # if isinstance(distortions, dict):
        #     list_specs = distortions.values()
        # elif isinstance(distortions, list):
        #     list_specs = distortions
        # else:
        #     list_specs = [distortions]
        #
        # dfs = []
        # for dist in list_specs:
        #     g, g_inv = dist.g, dist.g_inv
        #
        #     S = self.density_df.S
        #     loss = self.density_df.loss
        #
        #     a = A - self.bs  # A-bs for pandas series (includes endpoint), a for numpy indexing; int(A / self.bs)
        #     lossa = loss[0:a]
        #
        #     Sa = S[0:a]
        #     Fa = 1 - Sa
        #     gSa = g(Sa)
        #     premium = np.cumsum(gSa[::-1])[::-1] * self.bs
        #     el = np.cumsum(Sa[::-1])[::-1] * self.bs
        #     capital = A - lossa - premium
        #     risk_margin = premium - el
        #     assets = capital + premium
        #     marg_roe = (gSa - Sa) / (1 - gSa)
        #     lr = el / premium
        #     roe = (premium - el) / capital
        #     leverage = premium / capital
        #     # rp = -np.log(Sa) # return period
        #     marg_lr = Sa / gSa
        #
        #     # sns.set_palette(sns.color_palette("Paired", 4))
        #     df = pd.DataFrame({'$F(x)$': Fa, '$x$': lossa, 'Premium': premium, r'$EL=E(X\wedge x)$': el,
        #                        'Capital': capital, 'Risk Margin': risk_margin, 'Assets': assets, '$S(x)$': Sa,
        #                        '$g(S(x))$': gSa, 'Loss Ratio': lr, 'Marginal LR': marg_lr, 'ROE': roe,
        #                        'Marginal ROE': marg_roe, 'P:S levg': leverage})
        #     df = df.set_index('$F(x)$', drop=True)
        #     df.plot(subplots=True, rot=0, figsize=(14, 4), layout=(-1, 7))
        #     suptitle_and_tight(f'{str(dist)}: Statistics for Layer $x$ to $a$ vs. $F(x)$')
        #     df['distortion'] = dist.name
        #     dfs.append(df)
        # return pd.concat(dfs)

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

            priority_analysis_df.loc[:, 'pct chg'] = priority_analysis_df.loc[:, 'chg a'] / priority_analysis_df.a
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

* If {line} is added as second priority to the existing lines with no increase in capital it has an epd of {e2:.4g}.
* If the regulator requires the overall epd be a constant then the firm must increase capital to {a0:,.1f} or by {(a0 / a - 1) * 100:.2f} percent.
    - At the higher capital {line} has an epd of {e2a0:.4g} as second priority and the existing lines have an epd of {eb0a0:.4g} as first priority.
    - The existing and {line} epds under equal priority are {eba0:.4g} and {e1a0:.4g}.
* If {line} *thought* it was added at equal priority it would have expected an epd of {e1:.4g}.
  In order to achieve this epd as second priority would require capital of {a2:,.1f}, an increase of {(a2 / a - 1) * 100:.2f} percent.
* In order for {line} to have an epd equal to the existing lines as second priority would require capital
  of {a3:,.1f}, and increase of {(a3 / a - 1) * 100:.2f} percent.
* In order for {line} to be added at equal priority and for the existing lines to have an unchanged epd requires capital of {af:,.1f}, an
  increase of {(af / a - 1) * 100:.2f} percent.
* In order for {line} to be added at equal priority and to have an epd equal to the existing line epd requires capital of {af2:,.1f}, an
  increase of {(af2 / a - 1) * 100:.2f} percent.
* In order for the existing lines to have an unchanged epd at equal priority requires capital of {a4:,.1f}, an increase of {(a4 / a - 1) * 100:.2f} percent.
"""
            ans.append(story)
        ans = '\n'.join(ans)
        if output == 'html':
            display(HTML(pypandoc.convert_text(ans, to='html', format='markdown')))
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
        xs = self.density_df.loc[:, 'loss'].values
        pline = self.density_df.loc[:, 'p_' + line].values
        notline = self.density_df.loc[:, 'ημ_' + line].values
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
                print(e)
                print(f"Value error: loss={loss}, loss/b={loss / self.bs}, c1={c1}, c1/b={c1 / self.bs}")
                print(f"n_loss {n_loss},  n_c {n_c}, n_c1 {n_c1}")
                print(f'la={la}, lb={lb}, lc={lc}, ld={ld}')
                print('ONE:', *map(len, [xs[s1], pline[s1], notline[s1r]]))
                print('TWO:', *map(len, [pline[s2], notline[s2r]]))
                print('THR:', *map(len, [xs[s3], pline[s3], notline[s3r]]))
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
        dtest2 = self.density_df.loc[:, f'exi_xgta_{line}'] * self.density_df.S

        ddtest = np.gradient(dtest)
        ddtest2 = -self.density_df.loc[:, f'exeqa_{line}'] / self.density_df.loss * self.density_df.p_total

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
        dd = Distortion.distortions_from_params(params, index=idx, r0=r0, plot=False)
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
            # TODO FIND where this is introduced and remove it!!
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
            # TODO SQUASH...
            self._renamer['exi/xgta'] = 'α=E[Xi/X | X > a]'
            self._renamer['exi/xgtag'] = 'β=E_Q[Xi/X | X > a]'

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

    def set_a_p(self, a, p):
        """
        sort out arguments for assets and prob level and make them consistent
        neither => set defaults
        a only set p
        p only set a
        both do nothing

        """
        if a==0 and p==0:
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

    # def example_factory_sublines(self, data_in, xlim=0):
    #     """
    #     plausible plots for the specified example and a summary table
    #
    #     data_in is augmented_df or Answer object coming out of example_factory
    #
    #     example_factor is total; these exhibits look at subplots...
    #
    #     The line names must be of the form [A-Z]anything and identified by the capital letter
    #
    #     was agg extensions.plot_example
    #
    #     :param data_in: result of running self.apply_distortion()
    #     :param xlim:         for plots
    #     :return:
    #     """
    #
    #     if isinstance(data_in, Answer):
    #         augmented_df = data_in.augmented_df
    #     else:
    #         augmented_df = data_in
    #
    #     temp = augmented_df.filter(regex='exi_xgtag?_(?!sum)|^S|^gS|^(M|T)\.').copy()
    #     # add extra stats
    #     # you have:
    #     # ['M.M_Atame', 'M.M_Dthick', 'M.M_total',
    #     #  'M.Q_Atame', 'M.Q_Dthick', 'M.Q_total',
    #     #  'M.ROE', 'S',
    #     #  'T.LR_Atame', 'T.LR_Dthick', 'T.LR_total',
    #     #  'T.M_Atame', 'T.M_Dthick', 'T.M_total',
    #     #  'T.Q_Atame', 'T.Q_Dthick', 'T.Q_total',
    #     #  'T.ROE_Atame', 'T.ROE_Dthick', 'T.ROE_total',
    #     #  'exi_xgta_Atame', 'exi_xgta_Dthick',
    #     #  'exi_xgtag_Atame', 'exi_xgtag_Dthick',
    #     #  'gS']
    #
    #     # temp['M.LR'] = temp['M.L'] / temp['M.P']
    #     # temp['M.ROE'] = (temp['M.P'] - temp['M.L']) / (1 - temp['M.P'])
    #     # temp['M.M'] = temp['M.P'] - temp['M.L']
    #     for l in self.line_names_ex:
    #         if l != 'total':
    #             temp[f'M.L_{l}'] = temp[f'exi_xgta_{l}'] * temp.S
    #             temp[f'M.P_{l}'] = temp[f'exi_xgtag_{l}'] * temp.gS
    #             # temp[f'M.M.{l}'] = temp[f'exi_xgtag_{l}'] * temp['M.P'] - temp[f'exi_xgta_{l}'] * temp['M.L']
    #             temp[f'beta/alpha.{l}'] = temp[f'exi_xgtag_{l}'] / temp[f'exi_xgta_{l}']
    #         else:
    #             temp[f'M.L_{l}'] = temp.S
    #             temp[f'M.P_{l}'] = temp.gS
    #             # temp[f'M.M.{l}'] = temp['M.P'] - temp['M.L']
    #         temp[f'M.LR_{l}'] = temp[f'M.L_{l}'] / temp[f'M.P_{l}']
    #         # should be zero:
    #         temp[f'Chk L+M=p_{l}'] = temp[f'M.P_{l}'] - temp[f'M.L_{l}'] - temp[f'M.M_{l}']
    #
    #     renamer = self.renamer.copy()
    #     # want to recast these now too...(special)
    #     renamer.update({'S': 'M.L total', 'gS': 'M.P total'})
    #     augmented_df.index.name = 'Assets a'
    #     temp.index.name = 'Assets a'
    #     if xlim == 0:
    #         xlim = self.q(1 - 1e-5)
    #
    #     f, axs = plt.subplots(4, 2, figsize=(8, 10), constrained_layout=True, squeeze=False)
    #     ax = iter(axs.flatten())
    #
    #     # ONE
    #     a = (1 - augmented_df.filter(regex='p_').cumsum()).rename(columns=renamer).sort_index(1). \
    #         plot(ylim=[0, 1], xlim=[0, xlim], title='Survival functions', ax=next(ax))
    #     a.grid('b')
    #
    #     # TWO
    #     a = augmented_df.filter(regex='exi_xgtag?').rename(columns=renamer).sort_index(1). \
    #         plot(ylim=[0, 1], xlim=[0, xlim], title=r'$\alpha=E[X_i/X | X>a],\beta=E_Q$ by Line', ax=next(ax))
    #     a.grid('b')
    #
    #     # THREE total margins
    #     a = augmented_df.filter(regex=r'^T\.M').rename(columns=renamer).sort_index(1). \
    #         plot(xlim=[0, xlim], title='Total Margins by Line', ax=next(ax))
    #     a.grid('b')
    #
    #     # FOUR marginal margins was dividing by bs end of first line
    #     # for some reason the last entry in M.M_total can be problematic.
    #     a = (augmented_df.filter(regex=r'^M\.M').rename(columns=renamer).sort_index(1).iloc[:-1, :].
    #          plot(xlim=[0, xlim], title='Marginal Margins by Line', ax=next(ax)))
    #     a.grid('b')
    #
    #     # FIVE
    #     a = augmented_df.filter(regex=r'^M\.Q|gF').rename(columns=renamer).sort_index(1). \
    #         plot(xlim=[0, xlim], title='Capital = 1-gS = F!', ax=next(ax))
    #     a.grid('b')
    #     for _ in a.lines:
    #         if _.get_label() == 'gF':
    #             _.set(linewidth=5, alpha=0.3)
    #     # recreate legend because changed lines
    #     a.legend()
    #
    #     # SIX see apply distortion, line 1890 ROE is in augmented_df
    #     a = augmented_df.filter(regex='^ROE$|exi_xeqa').rename(columns=renamer).sort_index(1). \
    #         plot(xlim=[0, xlim], title='M.ROE Total and $E[X_i/X | X=a]$ by line', ax=next(ax))
    #     a.grid('b')
    #
    #     # SEVEN improve scale selection
    #     a = temp.filter(regex='beta/alpha\.|M\.LR').rename(columns=renamer).sort_index(1). \
    #         plot(ylim=[-.05, 1.5], xlim=[0, xlim], title='Alpha, Beta and Marginal LR',
    #              ax=next(ax))
    #     a.grid('b')
    #
    #     # EIGHT
    #     a = augmented_df.filter(regex='LR').rename(columns=renamer).sort_index(1). \
    #         plot(ylim=[-.05, 1.25], xlim=[0, xlim], title='Total ↑LR by Line',
    #              ax=next(ax))
    #     a.grid('b')
    #
    #     # return
    #     if isinstance(data_in, Answer):
    #         data_in['subline_summary'] = temp
    #         data_in['fig_eight_plots'] = f
    #     else:
    #         data_in = Answer(summary=temp, fig_eight_by_line=f)
    #     return data_in

    def cumintegral(self, v, bs_override=0):
        """
        cumulative integral of v with buckets size bs

        :param bs_override:
        :param v:
        :return:
        """

        if bs_override != 0:
            bs = bs_override
        else:
            bs = self.bs

        if type(v) == np.ndarray:
            logger.warning('CALLING cumintegral on a numpy array!!\n' * 5)
            return np.hstack((0, v[:-1])).cumsum() * bs
        else:
            # was consistently (and obviously) the same
            # t1 = np.hstack((0, v.values[:-1])).cumsum() * bs
            # t2 = v.shift(1, fill_value=0).cumsum() * bs
            # logger.warning(f'Alternative cumintegral allclose={np.allclose(t1, t2)}')
            return v.shift(1, fill_value=0).cumsum() * bs

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
        spec_list = [g.dropna(axis=1).to_dict(orient='list') for n, g in df.groupby('name')]
        return Portfolio(name, spec_list)

    @staticmethod
    def from_Excel(name, ffn, sheet_name, **kwargs):
        """
        read in from Excel

        works via a Pandas dataframe; kwargs passed through to pd.read_excel
        drops all blank columns (mostly for auditing purposes)


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
                ports[name] = uw(pgm)
            else:
                ports[name] = pgm
            if uw and bs * log2 > 0:
                ports[name].update(bs=bs, log2=log2, padding=padding, **kwargs)

        return ports
