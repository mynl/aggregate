"""
Does this do anything?
"""

from scipy.integrate import quad
from copy import deepcopy
from . utils import *


class Agg(object):
    """
    CAgg creates an aggregate distritbution from a frequency/severity specfication.

    Spec = members:
        name
        attachment 0 (occ attachment) TODO why aren't these in the seveirty and have AGG L & A outside?
        limit np.inf (occ limit)
        severity =
            version 1
                dist name lognorm | gamma | histogram | portfolio
                shape param(s)
                scale = 1
                loc = 0
                portfolio = portfolio variable (hummmm, now do you want to deep copy??!)
            version 2
                dist name = scipy.stats distname  | histogram
                mean (unlimited)
                cv
                if histogram supply array of xs and ps
        frequency =
            n = claim count
            contagion, c
            fixed = 1  # makes the distribution fixed type rather than Poisson

    :param spec:
    """

    def __init__(self, spec):

        self.spec = deepcopy(spec)
        self.name = spec['name']

        # occurrence specs
        self.attachment = spec.get('attachment', 0)
        self.limit = spec.get('limit', np.inf)

        # set up severity
        sev = spec['severity']
        self.sev_name = sev['name']
        if sev['name'] == 'histogram':
            xs = np.array(sev['xs'])
            ps = np.array(sev['ps'])
            xss = np.sort(np.hstack((xs, xs + 1e-5)))
            pss = np.vstack((ps, np.zeros_like(ps))).reshape((-1,), order='F')[:-1]
            self.fz = ss.rv_histogram((pss, xss))
        elif sev['name'] == 'frozen':
            # hand in a frozen sev directly
            self.fz = sev['fz']
        elif sev['name'] == 'portfolio':
            port = sev['portfolio']
            # object must have been updated
            # TODO implement auto update...
            assert port.audit_df is not None
            xs = port.density_df.loss.values
            ps = port.density_df.p_total.values
            hist_type = sev.get('type', 'continuous')
            if hist_type == 'continuous':
                xs = np.hstack((xs, xs[-1] + xs[1]))
                self.fz = ss.rv_histogram((ps, xs))
            elif hist_type == 'discrete':
                xss = np.sort(np.hstack((xs, xs + 1e-5)))
                pss = np.vstack((ps, np.zeros_like(ps))).reshape((-1,), order='F')[:-1]
                self.fz = ss.rv_histogram((pss, xss))
            else:
                raise ValueError(f'Unknown type {hist_type} passed to portfolio approximation, ' 
                                 'valid=continuous or discrete')
        elif 'mean' in sev:
            mean = sev['mean']
            cv = sev['cv']
            self.fz, sh, sc = distribution_factory(sev['name'], mean, cv)
            sev['shape'] = sh
            sev['scale'] = sc
        else:
            gen = getattr(ss, sev['name'])
            shape = sev.get('shape', None)
            loc = sev.get('loc', 0)
            scale = sev.get('scale', 1)
            if shape is None:
                self.fz = gen(loc=loc, scale=scale)
            else:
                self.fz = gen(shape, loc=loc, scale=scale)

        # compute the various moments (EX, EX2, EX3 and stats=mcvskew)
        # this is too confusing with big means: quad does not see the function is not zero
        # sev1 = self.fz.expect(lambda x: x - self.attachment, lb=self.attachment,
        #                       ub=self.limit + self.attachment, conditional=True)
        # sev2 = self.fz.expect(lambda x: (x - self.attachment) ** 2, lb=self.attachment,
        #                       ub=self.limit + self.attachment, conditional=True)
        # sev3 = self.fz.expect(lambda x: (x - self.attachment) ** 3, lb=self.attachment,
        #                       ub=self.limit + self.attachment, conditional=True)
        # if limit = inf then get same issue with quad not "seeing" the function and returning incorrectly
        # make a temporary limit for these calculations only
        if self.limit == np.inf:
            m, v = self.fz.stats()  # note mv is default...messes up for histogram_rv
            v = np.sqrt(v)
            p = self.fz.isf(1e-6)
            if v == np.inf:
                temp_limit = max(0, self.attachment + p)
            else:
                temp_limit = max(m + 5 * v + self.attachment, self.attachment + p)
        else:
            temp_limit = self.limit
        sev1 = quad(lambda x: self.fz.sf(x), self.attachment, self.attachment + temp_limit)
        sev2 = quad(lambda x: 2 * x * self.fz.sf(x), self.attachment, self.attachment + temp_limit)
        sev3 = quad(lambda x: 3 * x * x * self.fz.sf(x), self.attachment, self.attachment + temp_limit)
        # should do some checking here...
        sev1 = sev1[0]
        sev2 = sev2[0]
        sev3 = sev3[0]

        # frequency
        freq = spec['frequency']
        self.n = freq['n']
        self.contagion = freq.get('contagion', 0)
        if 'cv' in freq:
            self.contagion = freq['cv'] ** 2
        self.fixed = freq.get('fixed', 0)
        c = self.contagion
        freq1 = self.n
        if self.fixed == 1 or self.fixed == 'fixed':
            # fixed distribution N=self.n certainly
            freq2 = freq1 ** 2
            freq3 = freq1 * freq2
        elif self.fixed == -1 or self.fixed == 'bernoulli':
            # code for bernoulli self.n, E(N^k) = E(N) = self.n
            freq2 = self.n
            freq3 = self.n
        elif c == 0:
            # Poisson
            freq2 = freq1 * (1 + freq1)
            freq3 = freq1 * (1 + freq1 * (3 + freq1))
        else:
            # for gamma alpha, k with density x^alpha e^-kx, EX^n = Gamma(alpha + n) / Gamma(n) k^-n
            # EX = a/k = 1, so a=k
            # EX2 = (a+1)a/k^2 = a^2/k^2 + a/k^2 = (EX)^2 + a/k^2 = 1 + 1/k, hence var = a/k^2
            # if EX=1 and var = c then var = a/k/k = 1/k = c, so k = 1/c
            # then a = 1/c
            # Finally EX3 = (a+2)(a+1)a/k^3 = (c+1)(c+2)
            # Iman Conover paper page 14
            freq2 = freq1 * (1 + freq1 * (1 + c))  # note 1+c = E(G^2)
            freq3 = freq1 * (1 + freq1 * (3 * (1 + c) + freq1 * (1 + c) * (1 + 2 * c)))

        # raw moments of aggregate, not central moments
        agg1 = freq1 * sev1
        agg2 = freq1 * sev2 + (freq2 - freq1) * sev1 ** 2
        agg3 = freq1 * sev3 + freq3 * sev1 ** 3 + 3 * (freq2 - freq1) * sev1 * sev2 + (
                - 3 * freq2 + 2 * freq1) * sev1 ** 3

        sevm, sevcv, sevskew = moments_to_mcvsk(sev1, sev2, sev3)
        freqm, freqcv, freqskew = moments_to_mcvsk(freq1, freq2, freq3)
        # store these: used to make approximations
        self.aggm, self.aggcv, self.aggskew = moments_to_mcvsk(agg1, agg2, agg3)
        p999 = estimate_agg_percentile(self.aggm, self.aggcv, self.aggskew, 0.999)
        self.report = stats_series([self.aggm, self.aggcv, self.aggskew,
                                    freqm, freqcv, freqskew,
                                    sevm, sevcv, sevskew,
                                    agg1, agg2, agg3,
                                    freq1, freq2, freq3,
                                    sev1, sev2, sev3, self.limit, p999], self.name)
        # get other variables defined in init
        self.sev_density = None
        self.fzapprox = None
        self.agg_density = None
        self.ftagg_density = None
        self.xs = None
        self.dh_agg_density = None
        self.dh_sev_density = None
        self.beta_name = ''  # name of the beta function used to create dh distortion

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        s = f"CAgg: {self.name}\n\tEN={self.n}, CV={self.report[('freq', 'cv')]:5.3f}\n\t" \
            f"{self.sev_name} EX={self.report[('sev', 'mean')]:,.0f}, " \
            f"CV={self.report[('sev', 'cv')]:5.3f}\n\t" \
            f"EA={self.report[('agg', 'mean')]:,.0f}, CV={self.report[('agg', 'cv')]:5.3f}"
        return s

    def __repr__(self):
        """
        Goal unmbiguous
        :return: MUST return a string
        """
        return str(self.spec)

    def plot(self, N=100, p=1e-4, axiter=None):
        """
        make a quick plot of fz

        :param axiter:
        :param N:
        :param p:
        :return:
        """
        # for now just severity
        if axiter is None:
            axiter = make_axes(2, (6, 3))

        x0 = self.fz.isf(1 - p)
        if x0 < 0.1:
            x0 = 0
        x1 = self.fz.isf(p)
        xs = np.linspace(x0, x1, N)
        ps = np.linspace(1 / N, 1, N, endpoint=False)
        den = self.fz.pdf(xs)
        qs = self.fz.ppf(ps)
        # plt.figure()
        next(axiter).plot(xs, den)
        next(axiter).plot(ps, qs)
        plt.tight_layout()

    def recommend_bucket(self, N):
        """
        recommend a bucket size given N buckets

        :param N:
        :return:
        """
        moment_est = estimate_agg_percentile(self.aggm, self.aggcv, self.aggskew) / N
        if self.limit == np.inf:
            limit_est = 0
        else:
            limit_est = self.limit / N
        logging.info(f'Agg.recommend_bucket | {self.name} moment: {moment_est}, limit {limit_est}')
        return max(moment_est, limit_est)

    def density(self, xs, padding, tilt_vector, approximation='exact', sev_calc='gradient', force_severity=False):
        """
        Compute the density

        :param xs:
        :param padding:
        :param tilt_vector:
        :param approximation:
        :param sev_calc:   use gradient (of F) or simple kludge (f / sum(f)) to estimate severity
        :param force_severity: make severities even if using approximation, for plotting
        :return:
        """
        bs = xs[1]  # adjustment not logical
        self.xs = xs
        # make the severity vector
        if approximation == 'exact' or force_severity:
            # why not just use pdf / sum(pdf)?
            if sev_calc == 'gradient':
                # no need to divide by bucket size because accounted for in cdf(xs)
                sev = np.gradient(self.fz.cdf(xs))
            elif sev_calc == 'rescale':
                sev = self.fz.pdf(xs)
                sev = sev / np.sum(sev)
            else:
                raise ValueError(f'Inadmissible value {sev_calc} for severity calc; '
                                 'allowed values: gradient or rescale (rescale density)')
            # deal with limit...
            if (self.limit > 0) and (self.limit < np.inf):
                lim_bucket = int(self.limit / bs)
                sev[lim_bucket] = np.sum(sev[lim_bucket:])
                sev[lim_bucket + 1:] = 0
            # kludge
            self.sev_density = sev / np.sum(sev)
        if force_severity:
            return
        if approximation == 'exact':
            if self.fixed == 0:
                # convolve for compound Poisson
                self.ftagg_density = np.exp(self.n * (ft(self.sev_density, padding, tilt_vector) - 1))
                self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            elif self.fixed == 1 or self.fixed == 'fixed':
                # fixed count distribution...still need to do convolution
                self.ftagg_density = ft(self.sev_density, padding, tilt_vector) ** self.n
                if self.n == 1:
                    self.agg_density = self.sev_density
                else:
                    self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            elif self.fixed == -1 or self.fixed == 'bernoulli':
                # binomial M_N(t) = p M_X(t) + (1-p) at zero point
                assert ((self.n > 0) and (self.n < 1))
                self.ftagg_density = self.n * ft(self.sev_density, padding, tilt_vector)
                self.ftagg_density += (1 - self.n) * np.ones_like(self.ftagg_density)
                self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            else:
                raise ValueError(f'Inadmissible value for fixed {self.fixed}'
                                 ' Allowable values are -1 (or bernoulli) 1 (or fixed), missing or 0 (Poisson)')
        else:
            if approximation == 'slognorm':
                shift, mu, sigma = sln_fit(self.aggm, self.aggcv, self.aggskew)
                self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)

            elif approximation == 'sgamma':
                shift, alpha, theta = sgamma_fit(self.aggm, self.aggcv, self.aggskew)
                self.fzapprox = ss.gamma(alpha, scale=theta, loc=shift)
            else:
                raise ValueError(f'Invalid approximation {approximation} option passed to CAgg density. '
                                 'Allowable options are: exact | slogorm | sgamma')

            ps = self.fzapprox.pdf(xs)
            self.agg_density = ps / np.sum(ps)
            self.ftagg_density = ft(self.agg_density, padding, tilt_vector)

    def delbaen_haezendonck_density(self, xs, padding, tilt_vector, beta, beta_name):
        """
        Compare the base and Delbaen Haezendonck transformed aggregates
        $\beta(x) = \alpha + \gamma(x)$.
        alpha = log(freq' / freq): log of the increase in claim count
        gamma = log(RND of adjusted severity) = log(tilde f / f)
        Adjustment guarantees a positive loading iff beta is an increasing function
        iff gamma is increasing iff tilde f / f is increasing.
        cf. eqn 3.7 and 3.8
        Note conditions that E(exp(beta(X)) and E(X exp(beta(X)) must both be finite (3.4, 3.5)
        form of beta function described in 2.23 via, 2.16-17 and 2.18
        From examples on last page of paper:

        *    beta(x) = a ==> adjust frequency by factor of e^a
        *    beta(x) = log(1 + b(x - E(X)))  ==> variance principle EN(EX + bVar(X))
        *    beta(x) = ax- logE_P(exp(a x))  ==> Esscher principle

        :param xs:
        :param padding:
        :param tilt_vector:
        :param beta: function R+ to R with appropriate properties or name of prob distortion function
        :param beta_name:
        :return:
        """
        if self.agg_density is None:
            # update
            self.density(xs, padding, tilt_vector, 'exact')
        if beta_name[0:2] == 'g_':
            # passed in a distortion function
            self.dh_sev_density = np.diff(beta(np.cumsum(np.hstack((0, self.sev_density)))))
            # expect ex_beta = 1 but allow to pass multiples....
        else:
            self.dh_sev_density = self.sev_density * np.exp(beta(xs))
        ex_beta = np.sum(self.dh_sev_density)
        self.dh_sev_density = self.dh_sev_density / ex_beta
        adj_n = ex_beta * self.n
        if self.fixed == 0:
            # convolve for compound Poisson
            ftagg_density = np.exp(adj_n * (ft(self.dh_sev_density, padding, tilt_vector) - 1))
            self.dh_agg_density = np.real(ift(ftagg_density, padding, tilt_vector))
        else:
            raise ValueError('Must use compound Poisson for DH density')
        self.beta_name = beta_name

    def emp_stats(self):
        """
        report on empirical stats


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
            df = pd.DataFrame({'raw': s1, self.beta_name: s2})
        else:
            df = pd.DataFrame(s1, columns=['raw'])
        df.loc['mean', 'theory'] = self.report[('agg', 'mean')]
        df.loc['sd', 'theory'] = self.report[('agg', 'cv')] * self.report[('agg', 'mean')]
        df.loc['cv', 'theory'] = self.report[('agg', 'cv')]
        df['err'] = df['raw'] / df['theory'] - 1
        return df

    def quick_visual(self, axiter=None, figsize=(9, 3)):
        """
        Plot severity and agg, density, distribution and Lee diagram

        :param axiter: iterator for axes
        :param figsize: figure size, only used if axiter is None
        :return:
        """

        if self.dh_agg_density is not None:
            n = 4
        else:
            n = 3

        set_tight = False
        if axiter is None:
            axiter = make_axes(n, figsize)
            set_tight = True

        F = np.cumsum(self.agg_density)
        mx = np.argmax(F > 1 - 1e-5)
        dh_F = None
        if self.dh_agg_density is not None:
            dh_F = np.cumsum(self.dh_agg_density)
            mx = max(mx, np.argmax(dh_F > 1 - 1e-5))
            dh_F = dh_F[:mx]
        F = F[:mx]
        if self.sev_density is None:
            self.density(self.xs, 1, None, sev_calc='rescale', force_severity=True)
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
        ax.set_ylim(0, min(2 * np.max(d), np.max(f[1:])))
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
            plt.tight_layout()
