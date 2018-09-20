import scipy.stats as ss
import numpy as np
from scipy.integrate import quad
import pandas as pd
import collections
import logging
from .utils import sln_fit, sgamma_fit, ft, ift, \
    axiter_factory, estimate_agg_percentile, suptitle_and_tight, MomentAggregator, html_title
from .spectral import Distortion
from scipy import interpolate
from scipy.optimize import newton
from IPython.core.display import display


class Aggregate(object):
    """
    Aggregate help placeholder

    """
    aggregate_keys = ['name', 'exp_el', 'exp_premium', 'exp_lr', 'exp_en', 'exp_attachment', 'exp_limit', 'sev_name',
                      'sev_a', 'sev_b', 'sev_mean', 'sev_cv', 'sev_loc', 'sev_scale', 'sev_xs', 'sev_ps',
                      'sev_wt', 'freq_name', 'freq_a', 'freq_b', 'note']

    def __init__(self, name, exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                 sev_name='', sev_a=0, sev_b=0, sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0,
                 sev_xs=None, sev_ps=None, sev_wt=1,
                 freq_name='', freq_a=0, freq_b=0, note=''):
        """

        el -> en
        prem x exp_lr -> el
        x . en -> el
        always have en x and el; may have prem and exp_lr
        if prem then exp_lr computed; if exp_lr then premium computed

        el is determined using np.where(el==0, prem*exp_lr, el)
        if el==0 then el = freq * sev
        assert np.all( el>0 or en>0 )

        call with el (or prem x exp_lr) (or n) expressing a mixture, with the same severity
        call with el expressing lines of business with an array of severities
        call with single el and array of sevs expressing a mixture; [] broken down by weights

        n is the CONDITIONAL claim count
        X is the GROUND UP severity, so X | X > attachment is used and generates n claims

        For fixed or histogram have to separate the parameter so they are not broad cast; otherwise
        you end up with multiple lines when you intend only one

        TODO: later do both, for now assume one or other is a scalar
        call with both that does the cross product... (note the sev array may not be equal sizes)

        :param name:
        :param exp_el:   expected loss or vector or matrix
        :param exp_premium:
        :param exp_lr:  loss ratio
        :param exp_en:  expected claim count per segment (self.n = total claim count)
        :param exp_attachment: occ attachment
        :param exp_limit: occ limit
        :param sev_name: Severity class object or similar or vector or matrix
        :param sev_a:
        :param sev_b:
        :param sev_mean:
        :param sev_cv:
        :param sev_loc:
        :param sev_scale:
        :param sev_xs:  xs and ps must be provided if sev_name is (c|d)histogram or fixed
        :param sev_ps:
        :param sev_wt: weight for mixed distribution
        :param freq_name: name of frequency distribution
        :param freq_a: freq dist shape1 OR CV = sq root contagion
        :param freq_b: freq dist shape2

        """

        assert np.allclose(np.sum(sev_wt), 1)

        # self.spec = dict(name=name, exp_el=exp_el, exp_premium=exp_premium, exp_lr=exp_lr, exp_en=exp_en,
        #                  exp_attachment=exp_attachment, exp_limit=exp_limit,
        #                  sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
        #                  sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
        #                  sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
        #                  freq_name=freq_name, freq_a=freq_a, freq_b=freq_b)

        # have to be ready for inputs to be in a list, e.g. comes that way from Pandas via Excel
        def get_value(v):
            if isinstance(v, list):
                return v[0]
            else:
                return v

        # class variables (mostly)
        self.name = get_value(name)
        self.freq_name = get_value(freq_name)
        self.freq_a = get_value(freq_a)
        self.freq_b = get_value(freq_b)
        self.note = note
        self.sev_density = None
        self.ftagg_density = None
        self.agg_density = None
        self.xs = None
        self.fzapprox = None
        self.n = 0  # total frequency
        self.agg_m, self.agg_cv, self.agg_skew = 0, 0, 0
        self._nearest_quantile_function = None

        # get other variables defined in init
        self.en = None  # this is for a sublayer e.g. for limit profile
        self.n = 0  # this is total frequency
        self.attachment = None
        self.limit = None
        self.scalar_business = None
        self.sev_density = None
        self.fzapprox = None
        self.agg_density = None
        self.ftagg_density = None
        self.xs = None
        self.bs = 0
        self.dh_agg_density = None
        self.dh_sev_density = None
        self.beta_name = ''  # name of the beta function used to create dh distortion
        self.sevs = None
        self.audit_df = None
        self.statistics_df = pd.DataFrame(columns=['name', 'limit', 'attachment', 'sevcv_param', 'el', 'prem', 'lr'] +
                                                  MomentAggregator.column_names(agg_only=False) +
                                                  ['contagion', 'mix_cv'])
        self.statistics_total_df = self.statistics_df.copy()

        ma = MomentAggregator(self.freq_name, self.freq_a, self.freq_b)

        # broadcast arrays: first line forces them all to be arrays
        # TODO this approach forces EITHER a mixture OR multi exposures
        # TODO need to expand to product for general case, however, that should be easy

        if not isinstance(exp_el, collections.Iterable):
            exp_el = np.array([exp_el])

        # pyCharm formatting
        exp_el, exp_premium, exp_lr, self.en, self.attachment, self.limit, sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, \
        sev_scale, sev_wt = \
            np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit, sev_name, sev_a, sev_b,
                                sev_mean, sev_cv,
                                sev_loc, sev_scale, sev_wt)
        # just one overall line/class of business?
        self.scalar_business = \
            np.all(
                list(map(lambda x: len(x) == 1, [exp_el, exp_premium, exp_lr, self.en, self.attachment, self.limit])))

        # holder for the severity distributions
        self.sevs = np.empty(exp_el.shape, dtype=type(Severity))
        exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
        # compute the grand total for approximations
        # overall freq CV with common mixing TODO this is dubious
        c = self.freq_a
        root_c = np.sqrt(self.freq_a)
        # counter
        r = 0
        for _el, _pr, _lr, _en, _at, _y, sn, sa, sb, sm, scv, sloc, ssc, smix in \
                zip(exp_el, exp_premium, exp_lr, self.en, self.attachment, self.limit, sev_name, sev_a, sev_b, sev_mean,
                    sev_cv, sev_loc, sev_scale, sev_wt):

            self.sevs[r] = Severity(sn, _at, _y, sm, scv, sa, sb, sloc, ssc, sev_xs, sev_ps, True)
            sev1, sev2, sev3 = self.sevs[r].moms()

            if _el > 0:
                _en = _el / sev1
            elif _en > 0:
                _el = _en * sev1
            if _pr > 0:
                _lr = _el / _pr
            elif _lr > 0:
                _pr = _el / _lr

            # scale for the mix
            _pr *= smix
            _el *= smix
            _lr *= smix
            _en *= smix

            # accumulate moments
            ma.add_fs(_en, sev1, sev2, sev3)

            # store
            self.statistics_df.loc[r, :] = [self.name, _y, _at, scv, _el, _pr, _lr] + ma.get_fsa_stats(total=False) + [
                c, self.freq_a]
            r += 1

        # average exp_limit and exp_attachment
        avg_limit = np.sum(self.statistics_df.limit * self.statistics_df.freq_1) / ma.tot_freq_1
        avg_attach = np.sum(self.statistics_df.attachment * self.statistics_df.freq_1) / ma.tot_freq_1
        # assert np.allclose(ma.freq_1, self.statistics_df.exp_en)

        # store answer for total
        tot_prem = self.statistics_df.prem.sum()
        tot_loss = self.statistics_df.el.sum()
        if tot_prem > 0:
            lr = tot_loss / tot_prem
        else:
            lr = np.nan
        self.statistics_total_df.loc[f'mixed', :] = [self.name, avg_limit, avg_attach, 0, tot_loss, tot_prem, lr] + \
                                                    ma.get_fsa_stats(total=True, remix=True) + [c, root_c]
        self.statistics_total_df.loc[f'independent', :] = \
            [self.name, avg_limit, avg_attach, 0, tot_loss, tot_prem, lr] + ma.get_fsa_stats(total=True,
                                                                                             remix=False) + [c, root_c]
        self.statistics_df['wt'] = self.statistics_df.freq_1 / ma.tot_freq_1
        self.statistics_total_df['wt'] = self.statistics_df.wt.sum()  # better equal 1.0!
        self.n = ma.tot_freq_1
        self.agg_m = self.statistics_total_df.loc['mixed', 'agg_m']
        self.agg_cv = self.statistics_total_df.loc['mixed', 'agg_cv']
        self.agg_skew = self.statistics_total_df.loc['mixed', 'agg_skew']
        # finally, need a report_ser series for Portfolio to consolidate
        self.report_ser = ma.stats_series(self.name, np.max(self.limit), 0.999, total=True)
        # TODO fill in missing p99                                   ^ pctile

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        # pull out agg statistics_df
        ags = self.statistics_total_df.loc['mixed', :]
        s = f"Aggregate: {self.name}\n\tEN={ags['freq_1']}, CV(N)={ags['freq_cv']:5.3f}\n\t" \
            f"{len(self.sevs)} severit{'ies' if len(self.sevs)>1 else 'y'}, EX={ags['sev_1']:,.1f}, CV(X)={ags['sev_cv']:5.3f}\n\t" \
            f"EA={ags['agg_1']:,.1f}, CV={ags['agg_cv']:5.3f}"
        return s

    def _repr_html_(self):
        s = [f'<h3>Aggregate object: {self.name}</h3>']
        s.append(f'Claim count {self.n:0,.2f}, {self.freq_name} distribution<br>')
        _n = len(self.statistics_df)
        if _n == 1:
            sv = self.sevs[0]
            s.append(f'Severity{sv.long_name} distribution, {sv.limit} xs {sv.attachment}<br>')
        else:
            s.append(f'Severity with {_n} components<br>')
        st = self.statistics_total_df.loc['mixed', :]
        sev_m = st.sev_m
        sev_cv = st.sev_cv
        n_m = st.freq_m
        n_cv = st.freq_cv
        a_m = st.agg_m
        a_cv = st.agg_cv
        _df = pd.DataFrame({'E(X)': [sev_m, n_m, a_m], 'CV(X)': [sev_cv, n_cv, a_cv],
                            'Skew(X)': [None, None, st.agg_skew]}, index=['Sev', 'Freq', 'Agg'])
        _df.index.name = 'X'
        if self.audit_df is not None:
            esev_m = self.audit_df.loc['mixed', 'emp_sev_1']
            esev_cv = self.audit_df.loc['mixed', 'emp_sev_cv']
            ea_m = self.audit_df.loc['mixed', 'emp_agg_1']
            ea_cv = self.audit_df.loc['mixed', 'emp_agg_cv']
            _df.loc['Sev', 'Est E(X)'] = esev_m
            _df.loc['Agg', 'Est E(X)'] = ea_m
            _df.loc[:, 'Err E(X)'] = _df['Est E(X)'] / _df['E(X)'] - 1
            _df.loc['Sev', 'Est CV(X)'] = esev_cv
            _df.loc['Agg', 'Est CV(X)'] = ea_cv
            _df.loc[:, 'Err CV(X)'] = _df['Est CV(X)'] / _df['CV(X)'] - 1
        _df.fillna('')
        return '\n'.join(s) + _df._repr_html_()

    # def __repr__(self):
    #     """
    #     Goal unmbiguous
    #     :return: MUST return a string
    #     """
    #     return str(self.spec)

    def discretize(self, sev_calc, discretization_calc='survival'):
        """


        :param sev_calc:  continuous or discrete or raw (for...)
        :param discretization_calc:  survival, distribution or both
        :return:
        """

        if sev_calc == 'continuous':
            adj_xs = np.hstack((self.xs, np.inf))
        elif sev_calc == 'discrete':
            adj_xs = np.hstack((self.xs - self.bs / 2, np.inf))
        elif sev_calc == 'raw':
            adj_xs = self.xs
        else:
            raise ValueError(
                f'Invalid parameter {sev_calc} passed to double_diff; options are raw, discrete or histogram')

        # bed = bucketed empirical distribution
        beds = []
        for fz in self.sevs:
            if discretization_calc == 'both':
                beds.append(np.maximum(np.diff(fz.cdf(adj_xs)), -np.diff(fz.sf(adj_xs))))
            elif discretization_calc == 'survival':
                beds.append(-np.diff(fz.sf(adj_xs)))
            elif discretization_calc == 'distribution':
                beds.append(np.diff(fz.cdf(adj_xs)))
            else:
                raise ValueError(
                    f'Invalid options {discretization_calc} to double_diff; options are density, survival or both')

        return beds

    def easy_update(self, log2=13, bs=0, **kwargs):
        """
        Convenience function


        :param log2:
        :param bs:
        :param reporting_level:
        :param kwargs:  passed through to update
        :return:
        """
        # guess bucket and update
        if bs == 0:
            bs = self.recommend_bucket(log2)
        self.log2 = log2
        xs = np.arange(0, 1 << log2, dtype=float) * bs
        if 'approximation' not in kwargs:
            kwargs['approximation'] = 'slognorm'
        self.update(xs, **kwargs)

    def update(self, xs, padding=1, tilt_vector=None, approximation='exact', sev_calc='discrete',
               discretization_calc='survival', force_severity=False, verbose=False):
        """
        Compute the density
        :param xs:
        :param padding:
        :param tilt_vector:
        :param approximation:
        :param sev_calc:   discrete = suitable for fft, continuous = for rv_histogram cts version
        :param discretization_calc: use survival, distribution or both (=max(cdf, sf)) which is most accurate calc
        :param force_severity: make severities even if using approximation, for plotting
        :param verbose: make partial plots and return details of all moments
        :return:
        """

        if verbose:
            axm = axiter_factory(None, len(self.sevs) + 2, aspect=1.66)
            df = pd.DataFrame(
                columns=['n', 'limit', 'attachment', 'en', 'emp ex1', 'emp cv', 'sum p_i', 'wt', 'nans', 'max', 'wtmax',
                         'min'])
            df = df.set_index('n')
        else:
            axm = None
            df = None
        audit = None
        r = 0
        aa = al = 0
        self.xs = xs
        self.bs = xs[1]

        # make the severity vector
        # case 1 (all for now) it is the claim count weighted average of the severities
        if approximation == 'exact' or force_severity:
            wts = self.statistics_df.freq_1 / self.statistics_df.freq_1.sum()
            self.sev_density = np.zeros_like(xs)
            beds = self.discretize(sev_calc, discretization_calc)
            for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
                self.sev_density += temp * w
                if verbose:
                    _m = np.sum(self.xs * temp)
                    _cv = np.sqrt(np.sum((self.xs ** 2) * temp) - (_m ** 2)) / _m
                    df.loc[r, :] = [l, a, n, _m, _cv,
                                    temp.sum(),
                                    w, np.sum(np.where(np.isinf(temp), 1, 0)),
                                    temp.max(), w * temp.max(), temp.min()]
                    r += 1
                    next(axm).plot(xs, temp, label='compt', lw=0.5, drawstyle='steps-post')
                    axm.ax.plot(xs, self.sev_density, label='run tot', lw=0.5, drawstyle='steps-post')
                    if np.all(self.limit < np.inf):
                        axm.ax.set(xlim=(0, np.max(self.limit) * 1.1), title=f'{l:,.0f} xs {a:,.0f}\twt={w:.2f}')
                    else:
                        axm.ax.set(title=f'{l:,.0f} xs {a:,.0f}\twt={w:.2f}')
                    axm.ax.legend()
            if verbose:
                next(axm).plot(xs, self.sev_density, lw=0.5, drawstyle='steps-post')
                axm.ax.set_title('occ')
                aa = float(np.sum(df.attachment * df.wt))
                al = float(np.sum(df.limit * df.wt))
                if np.all(self.limit < np.inf):
                    axm.ax.set_xlim(0, np.max(self.limit))
                else:
                    axm.ax.set_xlim(0, xs[-1])
                _m = np.sum(self.xs * np.nan_to_num(self.sev_density))
                _cv = np.sqrt(np.sum(self.xs ** 2 * np.nan_to_num(self.sev_density)) - _m ** 2) / _m
                df.loc["Occ", :] = [al, aa, self.n, _m, _cv,
                                    self.sev_density.sum(),
                                    np.sum(wts), np.sum(np.where(np.isinf(self.sev_density), 1, 0)),
                                    self.sev_density.max(), np.nan, self.sev_density.min()]
        if force_severity:
            return
        if approximation == 'exact':
            if self.freq_name == 'poisson':
                # TODO ignoring contagion! Where are other freq dists!!!
                # convolve for compound Poisson
                # TODO put back!
                if self.n > 100:
                    logging.warning(f' | warning, {self.n} very high claim count ')
                # assert self.n < 100
                self.ftagg_density = np.exp(self.n * (ft(self.sev_density, padding, tilt_vector) - 1))
                self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            elif self.freq_name == 'fixed':
                # fixed count distribution...still need to do convolution
                self.ftagg_density = ft(self.sev_density, padding, tilt_vector) ** self.n
                if self.n == 1:
                    self.agg_density = self.sev_density
                else:
                    self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            elif self.freq_name == 'bernoulli':
                # binomial M_N(t) = p M_X(t) + (1-p) at zero point
                assert ((self.n > 0) and (self.n < 1))
                self.ftagg_density = self.n * ft(self.sev_density, padding, tilt_vector)
                self.ftagg_density += (1 - self.n) * np.ones_like(self.ftagg_density)
                self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            else:
                raise ValueError(f'Inadmissible value for fixed {self.freq_name}'
                                 ' Allowable values are -1 (or bernoulli) 1 (or fixed), missing or 0 (Poisson)')
        else:
            # regardless of request if skew == 0 have to use normal
            if self.agg_skew == 0:
                self.fzapprox = ss.norm(scale=self.agg_m * self.agg_cv, loc=self.agg_m)
            elif approximation == 'slognorm':
                shift, mu, sigma = sln_fit(self.agg_m, self.agg_cv, self.agg_skew)
                self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
            elif approximation == 'sgamma':
                shift, alpha, theta = sgamma_fit(self.agg_m, self.agg_cv, self.agg_skew)
                self.fzapprox = ss.gamma(alpha, scale=theta, loc=shift)
            else:
                raise ValueError(f'Invalid approximation {approximation} option passed to CAgg density. '
                                 'Allowable options are: exact | slogorm | sgamma')

            ps = self.fzapprox.pdf(xs)
            self.agg_density = ps / np.sum(ps)
            self.ftagg_density = ft(self.agg_density, padding, tilt_vector)

        # make a suitable audit_df
        cols = ['name', 'limit', 'attachment', 'el', 'freq_1', 'sev_1', 'agg_m', 'agg_cv', 'agg_skew']
        self.audit_df = pd.concat((self.statistics_df[cols],
                                   self.statistics_total_df.loc[['mixed'], cols]),
                                  axis=0)
        # add empirical stats
        if self.sev_density is not None:
            _m = np.sum(self.xs * np.nan_to_num(self.sev_density))
            _cv = np.sqrt(np.sum(self.xs ** 2 * np.nan_to_num(self.sev_density)) - _m ** 2) / _m
        else:
            _m = np.nan
            _cv = np.nan
        self.audit_df.loc['mixed', 'emp_sev_1'] = _m
        self.audit_df.loc['mixed', 'emp_sev_cv'] = _cv
        _m = np.sum(self.xs * np.nan_to_num(self.agg_density))
        _cv = np.sqrt(np.sum(self.xs ** 2 * np.nan_to_num(self.agg_density)) - _m ** 2) / _m
        self.audit_df.loc['mixed', 'emp_agg_1'] = _m
        self.audit_df.loc['mixed', 'emp_agg_cv'] = _cv

        if verbose:
            ax = next(axm)
            ax.plot(xs, self.agg_density, 'b')
            ax.set(xlim=(0, self.agg_m * (1 + 5 * self.agg_cv)), title='aggregate_project')
            suptitle_and_tight(f'{self.name} severity audit')
            _m = np.sum(self.xs * np.nan_to_num(self.agg_density))
            _cv = np.sqrt(np.sum(self.xs ** 2 * np.nan_to_num(self.agg_density)) - _m ** 2) / _m
            df.loc['Agg', :] = [al, aa, self.n, _m, _cv,
                                self.agg_density.sum(),
                                np.nan, np.sum(np.where(np.isinf(self.agg_density), 1, 0)),
                                self.agg_density.max(), np.nan, self.agg_density.min()]
            audit = pd.concat((df[['limit', 'attachment', 'emp ex1', 'emp cv']],
                               pd.concat((self.statistics_df, self.statistics_total_df))[[
                                   'freq_1', 'sev_1', 'sev_cv']]), sort=True, axis=1)
            # TODO should one be mixed?
            audit.iloc[-1, -1] = self.statistics_total_df.loc['independent', 'agg_cv']
            audit.iloc[-1, -2] = self.statistics_total_df.loc['independent', 'agg_1']
            audit['abs sev err'] = audit.sev_1 - audit['emp ex1']
            audit['rel sev err'] = audit['abs sev err'] / audit['emp ex1']
        return df, audit

    def emp_stats(self):
        """
        report_ser on empirical statistics_df

        :return:
        """

        ex = np.sum(self.xs * self.agg_density)
        ex2 = np.sum(self.xs ** 2 * self.agg_density)
        # ex3 = np.sum(self.xs**3 * self.agg_density)
        v = ex2 - ex * ex
        sd = np.sqrt(v)
        cv = sd / ex
        s1 = pd.Series([ex, sd, cv], index=['mean', 'sd', 'cv'])
        if self.dh_sev_density is not None:
            ex = np.sum(self.xs * self.dh_agg_density)
            ex2 = np.sum(self.xs ** 2 * self.dh_agg_density)
            # ex3 = np.sum(self.xs**3 * self.dh_agg_density)
            v = ex2 - ex * ex
            sd = np.sqrt(v)
            cv = sd / ex
            s2 = pd.Series([ex, sd, cv], index=['mean', 'sd', 'cv'])
            df = pd.DataFrame({'numeric': s1, self.beta_name: s2})
        else:
            df = pd.DataFrame(s1, columns=['numeric'])
        df.loc['mean', 'theory'] = self.statistics_total_df.loc['Agg', 'agg1']
        df.loc['sd', 'theory'] = np.sqrt(self.statistics_total_df.loc['Agg', 'agg2'] -
                                         self.statistics_total_df.loc['Agg', 'agg1'] ** 2)
        df.loc['cv', 'theory'] = self.statistics_total_df.loc['Agg', 'agg_cv']  # report_ser[('agg', 'cv')]
        df['err'] = df['numeric'] / df['theory'] - 1
        return df

    def delbaen_haezendonck_density(self, xs, padding, tilt_vector, beta, beta_name=""):
        """
        Compare the base and Delbaen Haezendonck transformed aggregates

        * beta(x) = alpha + gamma(x)
        * alpha = log(freq' / freq): log of the increase in claim count
        * gamma = log(RND of adjusted severity) = log(tilde f / f)

        Adjustment guarantees a positive loading iff beta is an increasing function
        iff gamma is increasing iff tilde f / f is increasing.
        cf. eqn 3.7 and 3.8

        Note conditions that E(exp(beta(X)) and E(X exp(beta(X)) must both be finite (3.4, 3.5)
        form of beta function described in 2.23 via, 2.16-17 and 2.18

        From examples on last page of paper:

            beta(x) = a ==> adjust frequency by factor of e^a
            beta(x) = log(1 + b(x - E(X)))  ==> variance principle EN(EX + bVar(X))
            beta(x) = ax- logE_P(exp(a x))  ==> Esscher principle

        :param xs:
        :param padding:
        :param tilt_vector:
        :param beta: function R+ to R with appropriate properties or name of prob distortion function
        :param beta_name:
        :return:
        """
        if self.agg_density is None:
            # update
            self.update(xs, padding, tilt_vector, 'exact')
        if isinstance(beta, Distortion):
            # passed in a distortion function
            beta_name = beta.name
            self.dh_sev_density = np.diff(beta.g(np.cumsum(np.hstack((0, self.sev_density)))))
            # expect ex_beta = 1 but allow to pass multiples....
        else:
            self.dh_sev_density = self.sev_density * np.exp(beta.g(xs))
        ex_beta = np.sum(self.dh_sev_density)
        self.dh_sev_density = self.dh_sev_density / ex_beta
        adj_n = ex_beta * self.n
        if self.freq_name == 'poisson':
            # convolve for compound Poisson
            ftagg_density = np.exp(adj_n * (ft(self.dh_sev_density, padding, tilt_vector) - 1))
            self.dh_agg_density = np.real(ift(ftagg_density, padding, tilt_vector))
        else:
            raise ValueError('Must use compound Poisson for DH density')
        self.beta_name = beta_name

    def plot(self, kind='long', axiter=None, aspect=1, figsize=None):
        """
        make a quick plot of fz: computed density and aggregate_project

        :param kind:
        :param aspect:
        :param axiter:
        :param figsize:
       :return:
        """

        if self.agg_density is None:
            print('Cannot plot before update')
            return
        if self.sev_density is None:
            self.update(self.xs, 1, None, sev_calc='discrete', force_severity=True)

        set_tight = (axiter is None)

        if kind == 'long':
            axiter = axiter_factory(axiter, 10, aspect=aspect, figsize=figsize)

            max_lim = min(self.xs[-1], np.max(self.limit)) * 1.05
            if max_lim < 1: max_lim = 1

            next(axiter).plot(self.xs, self.sev_density)  # , drawstyle='steps-post')
            axiter.ax.set(title='Severity', xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.sev_density)
            axiter.ax.set(title='Log Severity')
            if np.sum(self.sev_density == 1) >= 1:
                # sev density is degenerate, 1,0,0,... log scales won't work
                axiter.ax.set(title='Severity Degenerate')
                axiter.ax.set(xlim=(0, max_lim * 2))
            else:
                axiter.ax.set(title='Log Severity')
                axiter.ax.set(title='Log Severity', yscale='log')
                axiter.ax.set(xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.sev_density.cumsum(), drawstyle='steps-post')
            axiter.ax.set(title='Severity Distribution')
            axiter.ax.set(xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.agg_density, label='aggregate_project')
            axiter.ax.plot(self.xs, self.sev_density, lw=0.5, drawstyle='steps-post', label='severity')
            axiter.ax.set(title='Aggregate')
            axiter.ax.legend()

            next(axiter).plot(self.xs, self.agg_density, label='aggregate_project')
            axiter.ax.set(title='Aggregate')

            next(axiter).plot(self.xs, self.agg_density, label='aggregate_project')
            axiter.ax.set(yscale='log', title='Aggregate, log scale')

            F = self.agg_density.cumsum()
            next(axiter).plot(self.xs, 1 - F)
            axiter.ax.set(title='Survival Function')

            next(axiter).plot(self.xs, 1 - F)
            axiter.ax.set(title='Survival Function, log scale', yscale='log')

            next(axiter).plot(1 - F, self.xs, label='aggregate_project')
            axiter.ax.plot(1 - self.sev_density.cumsum(), self.xs, label='severity')
            axiter.ax.set(title='Lee Diagram')
            axiter.ax.legend()

            # figure for extended plotting of return period:
            max_p = F[-1]
            if max_p > 0.9999:
                _n = 10
            else:
                _n = 5
            if max_p >= 1:
                max_p = 1 - 1e-10
            k = (max_p / 0.99) ** (1 / _n)
            extraps = 0.99 * k ** np.arange(_n)
            q = interpolate.interp1d(F, self.xs, kind='linear', fill_value='extrapolate', bounds_error=False)
            ps = np.hstack((np.linspace(0, 1, 100, endpoint=False), extraps))
            qs = q(ps)
            next(axiter).plot(1 / (1 - ps), qs)
            axiter.ax.set(title='Return Period', xscale='log')

        else:  # kind == 'quick':
            if self.dh_agg_density is not None:
                n = 4
            else:
                n = 3

            axiter = axiter_factory(axiter, n, figsize, aspect=aspect)

            F = np.cumsum(self.agg_density)
            mx = np.argmax(F > 1 - 1e-5)
            if mx == 0:
                mx = len(F) + 1
            dh_F = None
            if self.dh_agg_density is not None:
                dh_F = np.cumsum(self.dh_agg_density)
                mx = max(mx, np.argmax(dh_F > 1 - 1e-5))
                dh_F = dh_F[:mx]
            F = F[:mx]

            xs = self.xs[:mx]
            d = self.agg_density[:mx]
            sevF = np.cumsum(self.sev_density)
            sevF = sevF[:mx]
            f = self.sev_density[:mx]

            ax = next(axiter)
            ax.plot(xs, d, label='agg')
            ax.plot(xs, f, label='sev')
            if self.dh_agg_density is not None:
                ax.plot(xs, self.dh_agg_density[:mx], label='dh {:} agg'.format(self.beta_name))
                ax.plot(xs, self.dh_sev_density[:mx], label='dh {:} sev'.format(self.beta_name))
            max_y = min(2 * np.max(d), np.max(f[1:]))
            if max_y > 0:
                ax.set_ylim(0, max_y)
            ax.legend()
            ax.set_title('Density')
            ax = next(axiter)
            ax.plot(xs, d, label='agg')
            ax.plot(xs, f, label='sev')
            if self.dh_agg_density is not None:
                ax.plot(xs, self.dh_agg_density[:mx], label='dh {:} agg'.format(self.beta_name))
                ax.plot(xs, self.dh_sev_density[:mx], label='dh {:} sev'.format(self.beta_name))
            ax.set_yscale('log')
            ax.legend()
            ax.set_title('Log Density')

            ax = next(axiter)
            ax.plot(F, xs, label='Agg')
            ax.plot(sevF, xs, label='Sev')
            if self.dh_agg_density is not None:
                dh_F = np.cumsum(self.dh_agg_density[:mx])
                ax.plot(dh_F, xs, label='dh {:} agg'.format(self.beta_name))
            ax.legend()
            ax.set_title('Lee Diagram')

            if self.dh_agg_density is not None:
                # if dh computed graph comparision
                ax = next(axiter)
                ax.plot(1 - F, 1 - dh_F, label='g(S) vs S')
                ax.plot(1 - F, 1 - F, 'k', linewidth=.5, label=None)

        if set_tight:
            axiter.tidy()
            suptitle_and_tight(f'Aggregate {self.name}')

    def report(self, report_list='quick'):
        """

        :param report_list:
        :return:
        """
        full_report_list = ['statistics', 'quick', 'audit']
        if report_list == 'all':
            report_list = full_report_list

        if 'quick' in report_list:
            html_title(f'{self.name} Quick Report (Theoretic)', 1)
            display(pd.DataFrame(self.report_ser).unstack())

        if 'audit' in report_list:
            if self.audit_df is not None:
                html_title(f'{self.name} Audit Report', 1)
                display(self.audit_df)

        if 'statistics' in report_list:
            if len(self.statistics_df) > 1:
                df = pd.concat((self.statistics_df, self.statistics_total_df), axis=1)
            else:
                df = self.statistics_df
            html_title(f'{self.name} Statistics Report', 1)
            display(df)

    def recommend_bucket(self, log2=10, verbose=False):
        """
        recommend a bucket size given 2**N buckets

        :param N:
        :return:
        """
        N = 1 << log2
        if not verbose:
            moment_est = estimate_agg_percentile(self.agg_m, self.agg_cv, self.agg_skew) / N
            limit_est = self.limit.max() / N
            if limit_est == np.inf:
                limit_est = 0
            logging.info(f'Agg.recommend_bucket | {self.name} moment: {moment_est}, limit {limit_est}')
            return max(moment_est, limit_est)
        else:
            for n in sorted({log2, 16, 13, 10}):
                rb = self.recommend_bucket(n)
                if n == log2:
                    rbr = rb
                print(f'Recommended bucket size with {2**n} buckets: {rb:,.0f}')
            if self.bs != 0:
                print(f'Bucket size set with {2**self.log2} buckets at {self.bs:,.0f}')
            return rbr

    def q(self, p):
        """
        return a quantile using nearest (i.e. will be in the index

        :param p:
        :return:
        """
        if self._nearest_quantile_function is None:
            self._nearest_quantile_function = self.quantile_function("nearest")
        return float(self._nearest_quantile_function(p))

    def quantile_function(self, kind='nearest'):
        """
        return an approximation to the quantile function

        TODO sort out...this isn't right

        :param kind:
        :return:
        """
        q = interpolate.interp1d(self.agg_density.cumsum(), self.xs, kind=kind,
                                 bounds_error=False, fill_value='extrapolate')
        return q


class Severity(ss.rv_continuous):
    """

    A continuous random variable, subclasses ``scipy.statistics_df.rv_continuous``.

    adds layer and attachment to scipy statistics_df continuous random variable class
    overrides

    * cdf
    * pdf
    * isf
    * ppf
    * moments

    Should consider over-riding: sf, **statistics_df** ?munp



    """

    def __init__(self, sev_name, exp_attachment=0, exp_limit=np.inf, sev_mean=0, sev_cv=0, sev_a=0, sev_b=0,
                 sev_loc=0, sev_scale=0, sev_xs=None, sev_ps=None, conditional=True, name='', note=''):
        """

        :param sev_name: scipy statistics_df continuous distribution | (c|d)histogram  cts or discerte | fixed
        :param exp_attachment:
        :param exp_limit:
        :param sev_mean:
        :param sev_cv:
        :param sev_a:
        :param sev_b:
        :param sev_loc:
        :param sev_scale:
        :param sev_xs: for fixed or histogram classes
        :param sev_ps:
        :param conditional: conditional or unconditional; for severities use conditional
        """

        ss.rv_continuous.__init__(self, name=f'{sev_name}[{exp_limit} xs {exp_attachment:,.0f}]')
        self.limit = exp_limit
        self.attachment = exp_attachment
        self.detachment = exp_limit + exp_attachment
        self.fz = None
        self.pattach = 0
        self.pdetach = 0
        self.conditional = conditional
        self.sev_name = sev_name
        self.name = name
        self.long_name = f'{sev_name}[{exp_limit} xs {exp_attachment:,.0f}'
        self.note = note
        self.sev1 = self.sev2 = self.sev3 = None

        # there are two types: if sev_xs and sev_ps provided then fixed/histogram, else scpiy dist
        # allows you to define fixed with just xs=1 (no p)
        if sev_xs is not None:
            if sev_name == 'fixed':
                # fixed is a special case of dhistogram with just one point
                sev_name = 'dhistogram'
                sev_ps = np.array(1)
            assert sev_name[1:] == 'histogram'
            # TODO: make histogram work with exp_limit and exp_attachment; currently they are ignored
            xs, ps = np.broadcast_arrays(np.array(sev_xs), np.array(sev_ps))
            if not np.isclose(np.sum(ps), 1.0):
                logging.error(f'Severity.init | {sev_name} histogram/fixed severity with probs do not sum to 1, '
                              f'{np.sum(ps)}')
            # need to exp_limit distribution
            exp_limit = min(np.min(exp_limit), xs.max())
            if sev_name == 'chistogram':
                # continuous histogram: uniform between xs's
                xss = np.sort(np.hstack((xs, xs[-1] + xs[1])))
                # midpoints
                xsm = (xss[:-1] + xss[1:]) / 2
                self.sev1 = np.sum(xsm * ps)
                self.sev2 = np.sum(xsm ** 2 * ps)
                self.sev3 = np.sum(xsm ** 3 * ps)
                self.fz = ss.rv_histogram((ps, xss))
            elif sev_name == 'dhistogram':
                # discrete histogram: point masses at xs's
                self.sev1 = np.sum(xs * ps)
                self.sev2 = np.sum(xs ** 2 * ps)
                self.sev3 = np.sum(xs ** 3 * ps)
                xss = np.sort(np.hstack((xs - 1e-5, xs)))  # was + but F(x) = Pr(X<=x) so seems shd be to left
                pss = np.vstack((ps, np.zeros_like(ps))).reshape((-1,), order='F')[:-1]
                self.fz = ss.rv_histogram((pss, xss))
            else:
                raise ValueError('Histogram must be chistogram (continuous) or dhistogram (discrete)'
                                 f', you passed {sev_name}')

        elif sev_name in ['norm', 'expon']:
            # distributions with no shape parameters
            #     Normal (and possibly others) does not have a shape parameter
            if sev_loc == 0 and sev_mean > 0:
                sev_loc = sev_mean
            if sev_scale == 0 and sev_cv > 0:
                sev_scale = sev_cv * sev_loc
            gen = getattr(ss, sev_name)
            self.fz = gen(loc=sev_loc, scale=sev_scale)

        elif sev_name in ['beta']:
            # distributions with two shape parameters
            # require specific inputs
            # for Kent examples input sev_scale=maxl, sev_mean=el and sev_cv as input
            #     beta sev_a and sev_b params given expected loss, max loss exposure and sev_cv
            #     Kent E.'s specification. Just used to create the CAgg classes for his examples (in agg.examples)
            #     https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters
            if sev_name == 'beta' and sev_mean > 0 and sev_cv > 0:
                m = sev_mean / sev_scale
                v = m * m * sev_cv * sev_cv
                sev_a = m * (m * (1 - m) / v - 1)
                sev_b = (1 - m) * (m * (1 - m) / v - 1)
                self.fx = ss.beta(sev_a, sev_b, loc=0, scale=sev_scale)
            else:
                gen = getattr(ss, sev_name)
                self.fz = gen(sev_a, sev_b, loc=sev_loc, scale=sev_scale)
        else:
            # distributions with one shape parameter
            if sev_a == 0:  # TODO figuring 0 is invalid shape...
                sev_a, _ = self.cv_to_shape(sev_cv)
            if sev_scale == 0 and sev_mean > 0:
                sev_scale, self.fz = self.mean_to_scale(sev_a, sev_mean)
            else:
                gen = getattr(ss, sev_name)
                self.fz = gen(sev_a, scale=sev_scale, loc=sev_loc)

        if self.detachment == np.inf:
            self.pdetach = 0
        else:
            self.pdetach = self.fz.sf(self.detachment)

        if self.attachment == 0:
            self.pattach = 1
        else:
            self.pattach = self.fz.sf(self.attachment)

        if sev_mean > 0 or sev_cv > 0:
            # if you input a sev_mean or sev_cv check we are close to target
            st = self.fz.stats('mv')
            m = st[0]
            acv = st[1] ** .5 / m  # achieved sev_cv
            if sev_mean > 0 and not np.isclose(sev_mean, m):
                print(f'WARNING target mean {sev_mean} and achieved mean {m} not close')
                # assert (np.isclose(sev_mean, m))
            if sev_cv > 0 and not np.isclose(sev_cv, acv):
                print(f'WARNING target cv {sev_cv} and achieved cv {acv} not close')
                # assert (np.isclose(sev_cv, acv))
            # print('ACHIEVED', sev_mean, sev_cv, m, acv, self.fz.statistics_df(), self._stats())
            logging.info(
                f'Severity.__init__ | parameters {sev_a}, {sev_scale}: target/actual {sev_mean} vs {m};  '
                f'{sev_cv} vs {acv}')

        if exp_limit < np.inf or exp_attachment > 0:
            layer_text = f'[{exp_limit:,.0f}' if exp_limit != np.inf else "Unlimited"
            layer_text += f' xs {exp_attachment:,.0f}]'
        else:
            layer_text = ''
        try:
            self.long_name = f'{name}: {sev_name}({self.fz.arg_dict[0]:.2f}){layer_text}'
        except:
            # 'rv_histogram' object has no attribute 'arg_dict'
            self.long_name = f'{name}: {sev_name}{layer_text}'

        assert self.fz is not None

    def cv_to_shape(self, cv, hint=1):
        """
        create a frozen object of type dist_name with given cv
        dist_name = 'lognorm'
        cv = 0.25

        :param cv:
        :param hint:
        :return:
        """
        # some special cases we can handle:
        if self.sev_name == 'lognorm':
            shape = np.sqrt(np.log(cv*cv + 1))
            fz = ss.lognorm(shape)
            return shape, fz

        if self.sev_name == 'gamma':
            shape = 1 / (cv * cv)
            fz = ss.gamma(shape)
            return shape, fz

        gen = getattr(ss, self.sev_name)

        def f(shape):
            fz0 = gen(shape)
            temp = fz0.stats('mv')
            return cv - temp[1] ** .5 / temp[0]

        try:
            ans = newton(f, hint)
        except RuntimeError:
            logging.error(f'cv_to_shape | error for {self.sev_name}, {cv}')
            ans = np.inf
            return ans, None
        fz = gen(ans)
        return ans, fz

    def mean_to_scale(self, shape, mean):
        """
        adjust scale of fz to have desired mean
        return frozen instance

        :param shape:
        :param mean:
        :return:
        """
        gen = getattr(ss, self.sev_name)
        fz = gen(shape)
        m = fz.stats('m')
        scale = mean / m
        fz = gen(shape, scale=scale)
        return scale, fz

    def __enter__(self):
        """ Support with Severity as f: """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def _pdf(self, x, *args):
        if self.conditional:
            return np.where(x > self.limit, 0,
                            np.where(x == self.limit, np.inf if self.pdetach > 0 else 0,
                                     self.fz.pdf(x + self.attachment) / self.pattach))
        else:
            if self.pattach < 1:
                return np.where(x < 0, 0,
                                np.where(x == 0, np.inf,
                                         np.where(x == self.detachment, np.inf,
                                                  np.where(x > self.detachment, 0,
                                                           self.fz.pdf(x + self.attachment, *args)))))
            else:
                return np.where(x < 0, 0,
                                np.where(x == self.detachment, np.inf,
                                         np.where(x > self.detachment, 0,
                                                  self.fz.pdf(x + self.attachment, *args))))

    def _cdf(self, x, *args):
        if self.conditional:
            return np.where(x > self.limit, 1,
                            np.where(x < 0, 0,
                                     (self.fz.cdf(x + self.attachment) - (1 - self.pattach)) / self.pattach))
        else:
            return np.where(x < 0, 0,
                            np.where(x == 0, 1 - self.pattach,
                                     np.where(x > self.limit, 1,
                                              self.fz.cdf(x + self.attachment, *args))))

    def _sf(self, x, *args):
        if self.conditional:
            return np.where(x > self.limit, 0,
                            np.where(x < 0, 1,
                                     self.fz.sf(x + self.attachment, *args) / self.pattach))
        else:
            return np.where(x < 0, 1,
                            np.where(x == 0, self.pattach,
                                     np.where(x > self.limit, 0,
                                              self.fz.sf(x + self.attachment, *args))))

    def _isf(self, q, *args):
        if self.conditional:
            return np.where(q < self.pdetach / self.pattach, self.limit,
                            self.fz.isf(q * self.pattach) - self.attachment)
        else:
            return np.where(q >= self.pattach, 0,
                            np.where(q < self.pdetach, self.limit,
                                     self.fz.isf(q, *args) - self.attachment))

    def _ppf(self, q, *args):
        if self.conditional:
            return np.where(q > 1 - self.pdetach / self.pattach, self.limit,
                            self.fz.ppf(1 - self.pattach * (1 - q)) - self.attachment)
        else:
            return np.where(q <= 1 - self.pattach, 0,
                            np.where(q > 1 - self.pdetach, self.limit,
                                     self.fz.ppf(q, *args) - self.attachment))

    def _stats(self, *args, **kwds):
        ex1, ex2, ex3 = self.moms()
        var = ex2 - ex1 ** 2
        skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / var ** 1.5
        return np.array([ex1, var, skew, np.nan])

    def _munp(self, n, *args):
        print('wow, called munp')
        pass

    def moms(self):
        """
        moms method
        remember have to use integral of survival function otherwise quad can fail to "see" the distribution
        for unlimited integral of a low sd variable

        :return:
        """

        if self.sev1 is not None:
            # precomputed (fixed and (c|d)histogram classes
            # TODO ignores layer and attach...
            return self.sev1, self.sev2, self.sev3

        median = self.fz.isf(0.5)

        def safe_integrate(f, level):
            ex = quad(f, self.attachment, self.detachment, limit=100, full_output=1)
            if len(ex) == 4:  # 'The integral is probably divergent, or slowly convergent.':
                # TODO just wing it for now
                logging.warning(
                    f'Severity.moms | splitting {self.sev_name} EX^{level} integral for convergence reasons')
                exa = quad(f, self.attachment, median, limit=100, full_output=1)
                exb = quad(f, median, self.detachment, limit=100, full_output=1)
                if len(exa) == 4:
                    logging.warning(f'Severity.moms | left hand split EX^{level} integral returned {exa[-1]}')
                if len(exb) == 4:
                    logging.warning(f'Severity.moms | right hand split EX^{level} integral returned {exb[-1]}')
                ex = (exa[0] + exb[0], exa[1] + exb[1])
            return ex

        ex1 = safe_integrate(lambda x: self.fz.sf(x), 1)
        ex2 = safe_integrate(lambda x: 2 * (x - self.attachment) * self.fz.sf(x), 2)
        ex3 = safe_integrate(lambda x: 3 * (x - self.attachment) ** 2 * self.fz.sf(x), 3)

        # quad returns abs error
        eps = 1e-5
        if not ((ex1[1] / ex1[0] < eps or ex1[1] < 1e-4) and
                (ex2[1] / ex2[0] < eps or ex2[1] < 1e-4) and
                (ex3[1] / ex3[0] < eps or ex3[1] < 1e-6)):
            logging.info(f'Severity.moms | **DOUBTFUL** convergence of integrals, abs errs '
                         f'\t{ex1[1]}\t{ex2[1]}\t{ex3[1]} \trel errors \t{ex1[1]/ex1[0]}\t{ex2[1]/ex2[0]}\t'
                         f'{ex3[1]/ex3[0]}')
            # raise ValueError(f' Severity.moms | doubtful convergence of integrals, abs errs '
            #                  f'{ex1[1]}, {ex2[1]}, {ex3[1]} rel errors {ex1[1]/ex1[0]}, {ex2[1]/ex2[0]}, '
            #                  f'{ex3[1]/ex3[0]}')
        # logging.info(f'Severity.moms | GOOD convergence of integrals, abs errs '
        #                      f'\t{ex1[1]}\t{ex2[1]}\t{ex3[1]} \trel errors \
        #  t{ex1[1]/ex1[0]}\t{ex2[1]/ex2[0]}\t{ex3[1]/ex3[0]}')

        ex1 = ex1[0]
        ex2 = ex2[0]
        ex3 = ex3[0]

        if self.conditional:
            ex1 /= self.pattach
            ex2 /= self.pattach
            ex3 /= self.pattach

        return ex1, ex2, ex3

    def plot(self, N=100, axiter=None):
        """
        quick plot

        :param axiter:
        :param N:
        :return:
        """

        ps = np.linspace(0, 1, N, endpoint=False)
        xs = np.linspace(0, self._isf(1e-4), N)

        do_tight = axiter is None
        axiter = axiter_factory(None, 4)

        ax = next(axiter)
        ys = self._pdf(xs)
        ax.plot(xs, ys)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set_title('PDF')

        ys = self._cdf(xs)
        ax = next(axiter)
        ax.plot(xs, ys, drawstyle='steps-post', lw=1)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set(title='CDF', ylim=(0, 1))

        ys = self._isf(ps)
        ax = next(axiter)
        ax.plot(ps, ys, drawstyle='steps-post', lw=1)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set(title='ISF', xlim=(0, 1))

        ax = next(axiter)
        ax.plot(1 - ps, ys, drawstyle='steps-post', lw=1)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set(title='Lee diagram', xlim=(0, 1))

        if do_tight:
            axiter.tidy()
            suptitle_and_tight(self.long_name)
