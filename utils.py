import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
from scipy.optimize import newton
import seaborn as sns
from IPython.core.display import HTML, display
import logging
import itertools

# logging
# TODO better filename!
LOGFILE = 'c:/S/TELOS/python/aggregate/aggregate.log'
logging.basicConfig(filename=LOGFILE,
                    filemode='w',
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    level=logging.DEBUG)
logging.info('aggregate.__init__ | New Aggregate Session started')


# momnent utility functions
def cumulate_moments(m1, m2, m3, n1, n2, n3):
    """
    Moments of sum of indepdendent variables

    :param m1: 1st moment, E(X)
    :param m2: 2nd moment, E(X^2)
    :param m3: 3rd moment, E(X^3)
    :param n1:
    :param n2:
    :param n3:
    :return:
    """
    # figure out moments of the sum
    t1 = m1 + n1
    t2 = m2 + 2 * m1 * n1 + n2
    t3 = m3 + 3 * m2 * n1 + 3 * m1 * n2 + n3
    return t1, t2, t3


def ft(z, padding, tilt):
    """
    fft with padding and tilt
    padding = n makes vector 2^n as long
    n=1 doubles (default)
    n=2 quadruples
    tilt is passed in as the tilting vector or None: easier for the caller to have a single instance

    :param z:
    :param padding: = 1 doubles
    :param tilt: vector of tilt values
    :return:
    """
    locft = np.fft.fft
    if z.shape != (len(z),):
        raise ValueError('FUKYUPY error, wrong shape passed into ft: ' + str(z.shape))
    # tilt
    if tilt is not None:
        zt = z * tilt
    else:
        zt = z
    # pad
    if padding > 0:
        # temp = np.hstack((z, np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)))
        pad_len = zt.shape[0] * ((1 << padding) - 1)
        temp = np.hstack((zt, np.zeros(pad_len)))
    else:
        temp = zt
    # temp = np.hstack((z, np.zeros_like(z)))
    return locft(temp)


def ift(z, padding, tilt):
    """
    ift that strips out padding and adjusts for tilt

    :param z:
    :param padding:
    :param tilt:
    :return:
    """
    locift = np.fft.ifft
    if z.shape != (len(z),):
        raise ValueError('FUKYUPY error, wrong shape passed into ft: ' + str(z.shape))
    temp = locift(z)
    # unpad
    temp = temp[0:]
    if padding != 0:
        temp = temp[0:int(len(temp) / (1 << padding))]
    # untilt
    if tilt is not None:
        temp /= tilt
    return temp
    # return temp[0:int(len(temp) / 2)]


def moments_to_mcvsk(ex1, ex2, ex3):
    """
    returns mean, cv and skewness from non-central moments

    :param ex1:
    :param ex2:
    :param ex3:
    :return:
    """
    m = ex1
    var = ex2 - ex1 ** 2
    sd = np.sqrt(var)
    cv = sd / m
    skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / sd ** 3
    return m, cv, skew


def stats_series(data_list, name):
    """
    combine elements into a reporting series
    handles order, index names etc. in one place

    :param data_list:
    :param name:
    :return:
    """
    idx = pd.MultiIndex.from_arrays(
        [['agg', 'agg', 'agg', 'freq', 'freq', 'freq', 'sev', 'sev', 'sev'] * 2 + ['agg', 'agg'],
         ['mean', 'cv', 'skew'] * 3 + ['ex1', 'ex2', 'ex3'] * 3 + ['limit', 'P99.9e']],
        names=['component', 'measure'])
    return pd.Series(data_list, name=name, index=idx)


def cv_to_shape(dist_name, cv, hint=1):
    """
    create a frozen object of type dist_name with given cv
    dist_name = 'lognorm'
    cv = 0.25

    :param dist_name:
    :param cv:
    :param hint:
    :return:
    """

    gen = getattr(ss, dist_name)

    def f(shape):
        fz0 = gen(shape)
        temp = fz0.stats('mv')
        return cv - temp[1] ** .5 / temp[0]

    try:
        ans = newton(f, hint)
    except RuntimeError:
        logging.error(f'cv_to_shape | error for {dist_name}, {cv}')
        ans = np.inf
        return ans, None
    fz = gen(ans)
    return ans, fz


def mean_to_scale(dist_name, shape, mean):
    """
    adjust scale of fz to have desired mean
    return frozen instance

    :param dist_name:
    :param shape:
    :param mean:
    :return:
    """
    gen = getattr(ss, dist_name)
    fz = gen(shape)
    m = fz.stats('m')
    scale = mean / m
    fz = gen(shape, scale=scale)
    return scale, fz


def sln_fit(m, cv, skew):
    """
    method of moments shifted lognormal fit matching given mean, cv and skewness

    :param m:
    :param cv:
    :param skew:
    :return:
    """
    eta = (((np.sqrt(skew ** 2 + 4)) / 2) + (skew / 2)) ** (1 / 3) - (
            1 / (((np.sqrt(skew ** 2 + 4)) / 2) + (skew / 2)) ** (1 / 3))
    sigma = np.sqrt(np.log(1 + eta ** 2))
    shift = m - cv * m / eta
    mu = np.log(m - shift) - sigma ** 2 / 2
    return shift, mu, sigma


def sgamma_fit(m, cv, skew):
    """
    method of moments shifted gamma fit matching given mean, cv and skewness

    :param m:
    :param cv:
    :param skew:
    :return:
    """
    alpha = 4 / (skew * skew)
    theta = cv * m * skew / 2
    shift = m - alpha * theta
    return shift, alpha, theta


# Distribution factory

def beta_factory(el, maxl, cv):
    """
    beta a and b params given expected loss, max loss exposure and cv
    Kent E.'s specification. Just used to create the CAgg classes for his examples (in agg.examples)
    https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters

    :param el:
    :param maxl:
    :param cv:
    """
    m = el / maxl
    v = m * m * cv * cv
    a = m * (m * (1 - m) / v - 1)
    b = (1 - m) * (m * (1 - m) / v - 1)
    return ss.beta(a, b, loc=0, scale=maxl)


def distribution_factory(dist_name, mean, cv):
    """
    Create a frozen distribution object by name
    Normal (and possibly others) does not have a shape parameter
    figure shape and scale from mean and cv
    corresponds to unlimited severity for now

    E.g.
    fz = dist_factory_ex('lognorm', 1000, 0.25)
    fz = dist_factory_ex('gamma', 1000, 1.25)
    plot_frozen(fz)

    :param dist_name:
    :param mean:
    :param cv:
    :return frozen distribution instance:
    """

    if dist_name in ['norm']:
        #     Create a frozen distribution object by name
        #     Normal (and possibly others) does not have a shape parameter
        scale = cv * mean
        # ss is scipy.stats
        gen = getattr(ss, dist_name)
        fz = gen(loc=mean, scale=scale)
        return fz

    sh, _ = cv_to_shape(dist_name, cv)
    sc, fz = mean_to_scale(dist_name, sh, mean)
    st = fz.stats('mv')
    m = st[0]
    acv = st[1] ** .5 / m  # achieved cv
    assert (np.isclose(mean, m))
    assert (np.isclose(cv, acv))
    return fz, sh, sc


def estimate_agg_percentile(m, cv, skew, p=0.999):
    """
    Come up with an estimate of the tail of the distribution based on the three parameter fits, ln and gamma
    if len(spec) > 3 it is assumed to be a spec, otherwise input m, cv, sk

    If spec passed in also take max with the limit

    :param m:
    :param cv:
    :param skew:
    :param p:
    :return:
    """

    shift, mu, sigma = sln_fit(m, cv, skew)
    fzl = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
    shift, alpha, theta = sgamma_fit(m, cv, skew)
    fzg = ss.gamma(alpha, scale=theta, loc=shift)
    pl = fzl.isf(1 - p)
    pg = fzg.isf(1 - p)
    # throw in a mean + 3 sd approx too...
    return max(pl, pg, m * (1 + ss.norm.isf(1 - p) * cv))


# misc
def plot_frozen(fz, N=100, p=1e-4):
    """
    make a quick plot of fz

    :param fz:
    :param N:
    :param p:
    :return:
    """

    x0 = fz.isf(1 - p)
    if x0 < 0.1:
        x0 = 0
    x1 = fz.isf(p)
    xs = np.linspace(x0, x1, N)
    ps = np.linspace(1 / N, 1, N, endpoint=False)
    den = fz.pdf(xs)
    qs = fz.ppf(ps)
    # plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(xs, den)
    plt.subplot(1, 2, 2)
    plt.plot(ps, qs)
    plt.tight_layout()
    # '{:5.3f}\t{:5.3f}'.format(*[float(i) for i in fz.stats()])
    plt.show()


def axiter_factory(axiter, n, figsize=None, height=2, aspect=1, nr=5):
    """
    axiter = check_axiter(axiter, ...) to allow chaining
    TODO can this be done in the class somehow?

    :param axiter:
    :param n:
    :param figsize:
    :param height:
    :param aspect:
    :param nr:
    :return:
    """
    if axiter is None:
        return AxisManager(n, figsize, height, aspect, nr)
    else:
        return axiter


class AxisManager(object):
    """


    """

    def __init__(self, n, figsize=None, height=2, aspect=1, nr=5):
        """

        :param n:
        :param figsize:
        :param height:
        :param aspect:
        :param nr: number of plots per row
        """

        self.n = n
        self.nr = nr
        self.r, self.c = self.grid_size(n)

        if figsize is None:
            h = self.r * height
            w = self.c * height * aspect
            figsize = (w, h)

        self.f, self.axs = plt.subplots(self.r, self.c, figsize=figsize)
        if n == 1:
            self.ax = self.axs
            self.it = None
        else:
            self.faxs = self.axs.flatten()
            self.it = iter(self.faxs)
            self.ax = None

    def __next__(self):
        if self.n > 1:
            self.ax = next(self.it)
        return self.ax

    def grid_size(self, n, subgrid=False):
        """
        appropriate grid size given class parameters

        :param n:
        :param subgrid: call is for a subgrid, no special treatment for 6 and 8
        :return:
        """
        r = (self.nr - 1 + n) // self.nr
        c = min(n, self.nr)
        if not subgrid:
            if self.nr > 3 and n == 6 and self.nr != 6:
                r = 2
                c = 3
            elif self.nr > 4 and n == 8 and self.nr != 8:
                r = 2
                c = 4
        return r, c

    def dimensions(self):
        return self.r, self.c

    def grid(self, size=0):
        """
        return a block of axes suitable for Pandas
        if size=0 return all the axes

        :param size:
        :return:
        """

        if size == 0:
            return self.faxs
        elif size == 1:
            return self.__next__()
        else:
            # need local sizing
            assert self.n >= size
            r, c = self.grid_size(size, subgrid=True)
            return [self.__next__() for _ in range(c) for _ in range(r)]


def cumintegral(v, bs):
    """
    cumulative integral of v with buckets size bs

    :param v:
    :param bs:
    :return:
    """
    if type(v) == np.ndarray:
        return np.hstack((0, v[:-1])).cumsum() * bs
    else:
        return np.hstack((0, v.values[:-1])).cumsum() * bs


def lognorm_lev(mu, sigma, n, limit):
    """
    return E(min(X, limit)^n) for lognormal using exact calculation
    currently only for n=1, 2

    :param mu:
    :param sigma:
    :param n:
    :param limit:
    :return:
    """
    if limit == -1:
        return np.exp(n * mu + n * n * sigma * sigma / 2)
    else:
        phi = ss.norm.cdf
        ll = np.log(limit)
        sigma2 = sigma * sigma
        phi_l = phi((ll - mu) / sigma)
        phi_l2 = phi((ll - mu - n * sigma2) / sigma)
        unlimited = np.exp(n * mu + n * n * sigma2 / 2)
        return unlimited * phi_l2 + limit ** n * (1 - phi_l)


def html_title(txt, n=1):
    """

    :param txt:
    :param n:
    :return:
    """
    display(HTML('<h{:}> {:}'.format(n, txt.replace("_", " ").title())))


def sensible_jump(n, desired_rows=20):
    """
    return a sensible jump size to output desired_rows given input of n

    :param n:
    :param desired_rows:
    :return:
    """
    if n < desired_rows:
        return 1
    j = int(n / desired_rows)
    return round(j, -len(str(j)) + 1)


def suptitle_and_tight(title, fontsize=14, **kwargs):
    """
    deal with tight layout when there is a suptitle

    :param title:
    :param fontsize:
    :return:
    """
    plt.suptitle(title, fontsize=fontsize, **kwargs)
    plt.tight_layout(rect=[0, 0, 1, 0.97])


def insurability_triangle():
    """
    Illustrate the insurability triangle...

    :return:
    """
    f, axs = plt.subplots(1, 3, figsize=(12, 4))
    it = iter(axs.flatten())
    λs = [1.5, 2, 3, 5, 10, 25, 50, 100]
    sns.set_palette(sns.cubehelix_palette(len(λs)))

    LR = np.linspace(0, 1, 101)
    plt.sca(next(it))
    for λ in λs:
        δ = (1 - LR) / LR / (λ - 1)
        plt.plot(LR, δ, label=f'λ={λ}')
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel('Loss Ratio')
    plt.ylabel('δ Investor Discount Rate')
    plt.title('Profitability vs. Loss Ratio \nBy PML to EL Ratio')

    LR = np.linspace(0, 1, 101)
    plt.sca(next(it))
    for λ in λs:
        levg = np.where(λ * LR > 1, 1 / (λ * LR - 1), 4)  # hide the discontinuity
        plt.plot(LR, levg, label=f'λ={λ}')
    # plt.legend()
    plt.ylim(0, 3)
    plt.xlabel('Loss Ratio')
    plt.ylabel('Premium to Surplus Ratio')
    plt.title('Premium to Surplus Ratio vs. Loss Ratio \nBy PML to EL Ratio')

    δ = np.linspace(0.0, 0.3, 301)
    plt.sca(next(it))
    for λ in λs:
        LR = 1 / (1 + δ * (λ - 1))
        plt.plot(δ, LR, label=f'λ={λ}')
    # plt.legend()
    plt.ylim(0, 1)
    plt.ylabel('Loss Ratio')
    plt.xlabel('Investor Discount Rate')
    plt.title('Loss Ratio vs. Investor Discount Rate \nBy PML to EL Ratio')


def read_log():
    """
    read and return the log file

    :return:
    """

    df = pd.read_csv(LOGFILE, sep='|', header=0, names=['datetime', 'context', 'type', 'routine', 'log'],
                     parse_dates=[0])
    for c in df.select_dtypes(object):
        df[c] = df[c].str.strip()
    df = df.dropna()
    df = df.set_index('datetime')
    return df


class MomentAggregator(object):
    """
    Accumulate moments
    Used by Portfolio and Aggregate (when there are multiple severities)
    makes report df and statistics_df

    Internal variables agg, sev, frqe, tot = running total, 1, 2, 3 = noncentral moments, E(X^k)


    """

    def __init__(self, freq_name, freq_a, freq_b):

        self.freq_name = freq_name
        self.freq_a = freq_a
        self.freq_b = freq_b

        # accumulators
        self.agg_1 = self.agg_2 = self.agg_3 = 0
        self.tot_agg_1 = self.tot_agg_2 = self.tot_agg_3 = 0
        self.freq_1 = self.freq_2 = self.freq_3 = 0
        self.tot_freq_1 = self.tot_freq_2 = self.tot_freq_3 = 0
        self.sev_1 = self.sev_2 = self.sev_3 = 0
        self.tot_sev_1 = self.tot_sev_2 = self.tot_sev_3 = 0

    def freq_moms(self, f1):
        """
        figure the moments for the given frequency distribution

        :param f1:
        :return:
        """

        freq_1 = f1

        if self.freq_name == 'fixed':
            # fixed distribution N=n certainly
            freq_2 = freq_1 ** 2
            freq_3 = freq_1 * freq_2
        elif self.freq_name == 'bernoulli':
            # code for bernoulli n, E(N^k) = E(N) = n
            freq_2 = freq_1
            freq_3 = freq_1
        elif self.freq_name == 'poisson' and self.freq_a == 0:
            # Poisson
            freq_2 = freq_1 * (1 + freq_1)
            freq_3 = freq_1 * (1 + freq_1 * (3 + freq_1))
        else:
            # for gamma alpha, k with density x^alpha e^-kx, EX^n = Gamma(alpha + n) / Gamma(n) k^-n
            # EX = a/k = 1, so a=k
            # EX2 = (a+1)a/k^2 = a^2/k^2 + a/k^2 = (EX)^2 + a/k^2 = 1 + 1/k, hence var = a/k^2
            # if EX=1 and var = c then var = a/k/k = 1/k = c, so k = 1/c
            # then a = 1/c
            # Finally EX3 = (a+2)(a+1)a/k^3 = (c+1)(c+2)
            # Iman Conover paper page 14
            # for now...
            c = self.freq_a * self.freq_a
            freq_2 = freq_1 * (1 + freq_1 * (1 + c))  # note 1+c = E(G^2)
            freq_3 = freq_1 * (1 + freq_1 * (3 * (1 + c) + freq_1 * (1 + c) * (1 + 2 * c)))

        return freq_2, freq_3

    def add_fs(self, f1, s1, s2, s3):
        """
        accumulate new moments defined by f and s
        compute agg for the latest values


        :param f1:
        :param s1:
        :param s2:
        :param s3:
        :return:
        """

        # fill in the frequency moments and store away
        f2, f3 = self.freq_moms(f1)
        self.freq_1 = f1
        self.freq_2 = f2
        self.freq_3 = f3

        # load current sev stats
        self.sev_1 = s1
        self.sev_2 = s2
        self.sev_3 = s3

        # accumulatge frequency
        self.tot_freq_1, self.tot_freq_2, self.tot_freq_3 = \
            self.cumulate_moments(self.tot_freq_1, self.tot_freq_2, self.tot_freq_3, f1, f2, f3)

        # severity
        self.tot_sev_1 = self.tot_sev_1 + f1 * s1
        self.tot_sev_2 = self.tot_sev_2 + f1 * s2
        self.tot_sev_3 = self.tot_sev_3 + f1 * s3

        # aggregate
        self.agg_1, self.agg_2, self.agg_3 = self.agg_from_fs(f1, f2, f3, s1, s2, s3)

        # finally accumulate the aggregate
        self.tot_agg_1, self.tot_agg_2, self.tot_agg_3 = \
            self.cumulate_moments(self.tot_agg_1, self.tot_agg_2, self.tot_agg_3, self.agg_1, self.agg_2, self.agg_3)

    @staticmethod
    def agg_from_fs(f1, f2, f3, s1, s2, s3):
        """
        aggregate moments from freq and sev components


        :param f1:
        :param f2:
        :param f3:
        :param s1:
        :param s2:
        :param s3:
        :return:
        """
        return f1 * s1, \
               f1 * s2 + (f2 - f1) * s1 ** 2, \
               f1 * s3 + f3 * s1 ** 3 + 3 * (f2 - f1) * s1 * s2 + (- 3 * f2 + 2 * f1) * s1 ** 3

    def get_fsa_stats(self, total, remix=False):
        """
        get the current f x s = agg stats and moments
        total = true use total else, current
        remix = true for total only, re-compute freq stats based on total freq 1

        :param total: binary
        :param remix: combine all sevs and recompute the freq moments from total freq
        :return:
        """

        if total:
            if remix:
                # recompute the frequency moments; all local variables
                f1 = self.tot_freq_1
                f2, f3 = self.freq_moms(f1)
                s1, s2, s3 = self.tot_sev_1 / f1, self.tot_sev_2 / f1, self.tot_sev_3 / f1
                a1, a2, a3 = self.agg_from_fs(f1, f2, f3, s1, s2, s3)
                return [f1, f2, f3, *self._moments_to_mcvsk(f1, f2, f3),
                        s1, s2, s3, *self._moments_to_mcvsk(s1, s2, s3),
                        a1, a2, a3, *self._moments_to_mcvsk(a1, a2, a3)]
            else:
                # running total, not adjusting freq cv
                return [self.tot_freq_1, self.tot_freq_2, self.tot_freq_3, *self.moments_to_mcvsk('freq', True),
                        self.tot_sev_1 / self.tot_freq_1, self.tot_sev_2 / self.tot_freq_1,
                        self.tot_sev_3 / self.tot_freq_1, *self.moments_to_mcvsk('sev', True),
                        self.tot_agg_1, self.tot_agg_2, self.tot_agg_3, *self.moments_to_mcvsk('agg', True)]
        else:
            # not total
            return [self.freq_1, self.freq_2, self.freq_3, *self.moments_to_mcvsk('freq', False),
                    self.sev_1, self.sev_2, self.sev_3, *self.moments_to_mcvsk('sev', False),
                    self.agg_1, self.agg_2, self.agg_3, *self.moments_to_mcvsk('agg', False)]

    def moments_to_mcvsk(self, mom_type, total=True):
        """
        convert noncentral moments into mean, cv and skewness
        type = agg | freq | sev | mix
        delegates work

        :param mom_type:
        :param total:
        :return:
        """

        if mom_type == 'agg':
            if total:
                return MomentAggregator._moments_to_mcvsk(self.tot_agg_1, self.tot_agg_2, self.tot_agg_3)
            else:
                return MomentAggregator._moments_to_mcvsk(self.agg_1, self.agg_2, self.agg_3)
        elif mom_type == 'freq':
            if total:
                return MomentAggregator._moments_to_mcvsk(self.tot_freq_1, self.tot_freq_2, self.tot_freq_3)
            else:
                return MomentAggregator._moments_to_mcvsk(self.freq_1, self.freq_2, self.freq_3)
        elif mom_type == 'sev':
            if total:
                return MomentAggregator._moments_to_mcvsk(self.tot_sev_1 / self.tot_freq_1,
                                                          self.tot_sev_2 / self.tot_freq_1,
                                                          self.tot_sev_3 / self.tot_freq_1)
            else:
                return MomentAggregator._moments_to_mcvsk(self.sev_1, self.sev_2, self.sev_3)

    def moments(self, mom_type, total=True):
        """
        vector of the moments; convenience function

        :param mom_type:
        :param total:
        :return:
        """
        if mom_type == 'agg':
            if total:
                return self.tot_agg_1, self.tot_agg_2, self.tot_agg_3
            else:
                return self.agg_1, self.agg_2, self.agg_3
        elif mom_type == 'freq':
            if total:
                return self.tot_freq_1, self.tot_freq_2, self.tot_freq_3
            else:
                return self.freq_1, self.freq_2, self.freq_3
        elif mom_type == 'sev':
            if total:
                return self.tot_sev_1 / self.tot_freq_1, self.tot_sev_2 / self.tot_freq_1, \
                       self.tot_sev_3 / self.tot_freq_1
            else:
                return self.sev_1, self.sev_2, self.sev_3

    @staticmethod
    def cumulate_moments(m1, m2, m3, n1, n2, n3):
        """
        Moments of sum of indepdendent variables

        :param m1: 1st moment, E(X)
        :param m2: 2nd moment, E(X^2)
        :param m3: 3rd moment, E(X^3)
        :param n1:
        :param n2:
        :param n3:
        :return:
        """
        # figure out moments of the sum
        t1 = m1 + n1
        t2 = m2 + 2 * m1 * n1 + n2
        t3 = m3 + 3 * m2 * n1 + 3 * m1 * n2 + n3
        return t1, t2, t3

    @staticmethod
    def column_names(agg_only):
        """
        list of the moment and stats names for f x s = a
        list of the moment and stats names for just agg

        :param agg_only: = True for total r
        :return:
        """

        if agg_only:
            return [i + j for i, j in itertools.product(['agg'], [f'_{i}' for i in range(1, 4)] +
                                                        ['_m', '_cv', '_skew'])]
        else:
            return [i + j for i, j in itertools.product(['freq', 'sev', 'agg'], [f'_{i}' for i in range(1, 4)] +
                                                        ['_m', '_cv', '_skew'])]

    @staticmethod
    def _moments_to_mcvsk(ex1, ex2, ex3):
        """
        returns mean, cv and skewness from non-central moments

        :param ex1:
        :param ex2:
        :param ex3:
        :return:
        """
        m = ex1
        var = ex2 - ex1 ** 2
        sd = np.sqrt(var)
        cv = sd / m
        skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / sd ** 3
        return m, cv, skew

    def stats_series(self, name, limit, p999e, total=False):
        """
        combine elements into a reporting series
        handles order, index names etc. in one place

        :param name: series name
        :param limit:
        :param p999e:
        :param total:
        :return:
        """
        idx = pd.MultiIndex.from_arrays(
            [['agg', 'agg', 'agg', 'freq', 'freq', 'freq', 'sev', 'sev', 'sev'] * 2 + ['agg', 'agg'],
             ['mean', 'cv', 'skew'] * 3 + ['ex1', 'ex2', 'ex3'] * 3 + ['limit', 'P99.9e']],
            names=['component', 'measure'])

        return pd.Series([*self.moments_to_mcvsk('agg', total),
                          *self.moments_to_mcvsk('freq', total),
                          *self.moments_to_mcvsk('sev', total),
                          *self.moments('agg', total),
                          *self.moments('freq', total),
                          *self.moments('sev', total),
                          limit, p999e], name=name, index=idx)
