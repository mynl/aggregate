from cycler import cycler
from functools import lru_cache
from io import StringIO
import itertools
from itertools import product
import logging.handlers
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import re
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.optimize import broyden2, newton_krylov
from scipy.optimize.nonlin import NoConvergence
from scipy.special import kv
from scipy.stats import multivariate_t
# from time import time_ns
from IPython.core.display import HTML, display


logger = logging.getLogger(__name__)


# TODO take out timer stuff
last_time = first_time = 0
timer_active = False


def get_fmts(df):
    """
    reasonable formats for a styler

    :param df:
    :return:
    """
    fmts = {}

    def guess_fmt(nm, sz):
        named_cols = {'Err': '{:6.3e}', 'CV': '{:6.3f}', 'Skew': '{:6.3f}'}
        for n, f in named_cols.items():
            if nm.find(n) >= 0:  # note -1 means not found
                return f
        if abs(sz) < 1:
            return '{:6.3e}'
        elif abs(sz) < 10:
            return '{:6.3f}'
        elif abs(sz) < 1000:
            return '{:6.0f}'
        elif abs(sz) < 1e10:
            return '{:,.0f}'
        else:
            return '{:5.3e}'

    for k, v in df.mean().items():
        fmts[k] = guess_fmt(k, v)
    return fmts


def pprint(txt):
    """
    guess a nice format for an agg program

    :param txt: program text input

    """
    bits = re.split(r'(agg|sev|mixed|poisson|fixed)', txt)
    clean = [re.sub(r'[ ]+', ' ', i.strip()) for i in bits]
    sio = StringIO()
    sio.write(clean[0])
    for agg, exp, sev, sevd, fs, freq in zip(*[clean[i::6] for i in range(1, 7)]):
        nm, *rest = exp.split(' ')
        sio.write(f'\n\t{agg} {nm:^12s} {float(rest[0]):8.1f} {" ".join(rest[1:]):^20s} '
                  f'{sev} {sevd:^25s} {fs:>8s}   {freq}')
    return sio.getvalue()


# moment utility functions
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
    # locft = np.fft.fft
    locft = np.fft.rfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
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
    # locift = np.fft.ifft
    locift = np.fft.irfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
    temp = locift(z)
    # unpad
    # temp = temp[0:]
    if padding != 0:
        temp = temp[0:len(temp) >> padding]
    # untilt
    if tilt is not None:
        temp /= tilt
    return temp
    # return temp[0:int(len(temp) / 2)]


def mu_sigma_from_mean_cv(m, cv):
    """
    lognormal parameters
    """
    cv = np.array(cv)
    m = np.array(m)
    sigma = np.sqrt(np.log(cv*cv + 1))
    mu = np.log(m) - sigma**2 / 2
    return mu, sigma


def sln_fit(m, cv, skew):
    """
    method of moments shifted lognormal fit matching given mean, cv and skewness

    :param m:
    :param cv:
    :param skew:
    :return:
    """
    if skew == 0:
        return -np.inf, np.inf, 0
    else:
        eta = (((np.sqrt(skew ** 2 + 4)) / 2) + (skew / 2)) ** (1 / 3) - (
                1 / (((np.sqrt(skew ** 2 + 4)) / 2) + (skew / 2)) ** (1 / 3))
        sigma = np.sqrt(np.log(1 + eta ** 2))
        shift = m - cv * m / eta
        if shift > m:
            logger.warning(f'utils sln_fit | shift > m, {shift} > {m}, too extreme skew {skew}')
            shift = m - 1e-6
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
    if skew == 0:
        return np.nan, np.inf, 0
    else:
        alpha = 4 / (skew * skew)
        theta = cv * m * skew / 2
        shift = m - alpha * theta
        return shift, alpha, theta


def estimate_agg_percentile(m, cv, skew, p=0.999):
    """
    Come up with an estimate of the tail of the distribution based on the three parameter fits, ln and gamma

    :param m:
    :param cv:
    :param skew:
    :param p:
    :return:
    """

    pn = pl = pg = 0
    if skew <= 0:
        # neither sln nor sgamma works, use a normal
        # for negative skewness the right tail will be thin anyway so normal not outrageous
        fzn = ss.norm(scale=m * cv, loc=m)
        pn = fzn.isf(1 - p)
    else:
        shift, mu, sigma = sln_fit(m, cv, skew)
        fzl = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
        shift, alpha, theta = sgamma_fit(m, cv, skew)
        fzg = ss.gamma(alpha, scale=theta, loc=shift)
        pl = fzl.isf(1 - p)
        pg = fzg.isf(1 - p)
    # throw in a mean + 3 sd approx too...
    return max(pn, pl, pg, m * (1 + ss.norm.isf(1 - p) * cv))


def round_bucket(bs):
    """
    Compute a decent rounded bucket from an input float bs. ::

        if bs > 1 round to 2, 5, 10, ...

        elif bs < 1 find the smallest power of two greater than 1/bs

    Test cases: ::

        test_cases = [1, 1.1, 2, 2.5, 4, 5, 5.5, 8.7, 9.9, 10, 13,
                      15, 20, 50, 100, 99, 101, 200, 250, 400, 457,
                        500, 750, 1000, 2412, 12323, 57000, 119000,
                        1e6, 1e9, 1e12, 1e15, 1e18, 1e21]
        for i in test_cases:
            print(i, round_bucket(i))
        for i in test_cases:
            print(1/i, round_bucket(1/i))

    """
    if bs == 0 or np.isinf(bs):
        raise ValueError(f'Inadmissible value passed to round_bucket, {bs}')

    if bs == 1:
        return bs

    if bs > 1:
        # rounded bs, to an integer
        rbs = np.round(bs, 0)
        if rbs == 1:
            return 2.0
        elif rbs == 2:
            return 2
        elif rbs <= 5:
            return 5.0
        elif rbs <= 10:
            return 10.0
        else:
            rbs = np.round(bs, -int(np.log(bs) / np.log(10)))
            if rbs < bs:
                rbs *= 2
            return rbs

    if bs < 1:
        # inverse bs
        # originally
        # bsi = 1 / bs
        # nbs = 1
        # while nbs < bsi:
        #     nbs <<= 1
        # nbs >>= 1
        # return 1. / nbs
        # same answer but ? clearer and slightly quicker
        x = 1. / bs
        x = bin(int(x))
        x = '0b1' + "0" * (len(x) -3)
        x = int(x[2:], 2)
        return 1./ x


def make_ceder_netter(reins_list, debug=False):
    """
    Build the netter and ceder functions. It is applied to occ_reins and agg_reins,
    so should be stand-alone.
    TODO deal with infinity!; limit is inf then share = 1 as percentage not currency amount
    """
    h = 0
    base = 0
    xs = [0]
    ys = [0]
    for (p, y, a) in reins_list:
        if np.isinf(y):
            raise NotImplementedError('inf not implemented')
        else:
            if a > base:
                xs.append(a)
                ys.append(h)
            h += p
            xs.append(a + y)
            ys.append(h)
            base += (a + y)
    xs.append(np.inf)
    ys.append(h)
    ceder = interp1d(xs, ys)
    netter = lambda x: x - ceder(x)
    if debug:
        return ceder, netter, xs, ys
    else:
        return ceder, netter


# axis management OLD
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
    Manages creation of a grid of axes for plotting. Allows pandas plot and matplotlib to
    plot to same set of axes.

    Always created and managed through axiter_factory function

    :param n:       number of plots in grid
    :param figsize:
    :param height:  height of individual plot
    :param aspect:  aspect ratio of individual plot
    :param nr:      number of plots per row


    """
    __slots__ = ['n', 'nr', 'r', 'ax', 'axs', 'c', 'f', 'faxs', 'it']

    def __init__(self, n, figsize=None, height=2, aspect=1, nr=5):
        self.n = n
        self.nr = nr
        self.r, self.c = self.grid_size(n)

        if figsize is None:
            h = self.r * height
            w = self.c * height * aspect
            # almost all the time will sup_title which scales down by 0.96
            figsize = (w, h / 0.96)

        self.f, self.axs = plt.subplots(self.r, self.c, figsize=figsize)
        if n == 1:
            self.ax = self.axs
            self.it = None
        else:
            # faxs = flattened axes
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
        """
        return dimensions (width and height) of current layout

        :return:
        """
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
            # r, c = self.grid_size(size, subgrid=True)
            return [self.__next__() for _ in range(size)]  # range(c) for _ in range(r)]

    def tidy(self):
        """
        delete unused axes to tidy up a plot

        :return:
        """
        if self.it is not None:
            for ax in self.it:
                self.f.delaxes(ax)

    # methods for a more hands on approach
    @staticmethod
    def good_grid(n, c=4):
        """
        Good layout for n plots
        :param n:
        :return:
        """
        basic = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (2, 4),
                 8: (2, 4), 9: (3, 3), 10: (4, 3), 11: (4, 3), 12: (4, 3), 13: (5, 3), 14: (5, 3),
                 15: (5, 3), 16: (4, 4), 17: (5, 4), 18: (5, 4), 19: (5, 4), 20: (5, 4)}

        if n <= 20:
            r, c = basic[n]
        else:
            r = n // c
            if r * c < n:
                r += 1
        return r, c

    @staticmethod
    def size_figure(r, c, aspect=1.5):
        """
        reasonable figure size for n plots
        :param r:
        :param c:
        :param aspect:
        :return:
        """
        w = min(6, 8 / c)
        h = w / aspect
        tw = w * c
        th = h * r
        if th > 10.5:
            tw = 10.5 / th * tw
            th = 10.5
        return tw, th

    @staticmethod
    def make_figure(n, aspect=1.5, **kwargs):
        """
        make the figure and iterator
        :param n:
        :param aspect:
        :return:
        """

        r, c = AxisManager.good_grid(n)
        w, h = AxisManager.size_figure(r, c, aspect)

        f, axs = plt.subplots(r, c, figsize=(w, h), constrained_layout=True, squeeze=False, **kwargs)
        axi = iter(axs.flatten())
        return f, axs, axi

    @staticmethod
    def print_fig(n, aspect=1.5):
        """
        printout code...to insert (TODO copy to clipboard!)
        :param n:
        :param aspect:
        :return:
        """
        r, c = AxisManager.good_grid(n)
        w, h = AxisManager.size_figure(r, c, aspect)

        l1 = f"f, axs = plt.subplots({r}, {c}, figsize=({w}, {h}), constrained_layout=True, squeeze=False)"
        l2 = "axi = iter(axs.flatten())"
        return '\n'.join([l1, l2])

    @staticmethod
    def tidy_up(f, ax):
        """
        delete unused frames out of a figure
        :param ax:
        :return:
        """
        for a in ax:
            f.delaxes(a)


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


# display related
def html_title(txt, n=1, title_case=True):
    """

    :param txt:
    :param n:
    :param title_case:
    :return:
    """
    if title_case:
        display(HTML('<h{:}> {:}'.format(n, txt.replace("_", " ").title())))
    else:
        display(HTML('<h{:}> {:}'.format(n, txt.replace("_", " "))))


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


def suptitle_and_tight(title, **kwargs):
    """
    deal with tight layout when there is a suptitle

    :param title:
    :return:
    """
    plt.suptitle(title, **kwargs)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])


class MomentAggregator(object):
    """
    Purely accumulates moments
    Used by Portfolio
    Not frequency aware
    makes report_ser df and statistics_df

    Internal variables agg, sev, freq, tot = running total, 1, 2, 3 = noncentral moments, E(X^k)

    :param freq_moms: function of one variable returning first three noncentral moments of the underlying
            frequency distribution

    """
    __slots__ = ['freq_1', 'freq_2', 'freq_3', 'sev_1', 'sev_2', 'sev_3', 'agg_1', 'agg_2', 'agg_3',
                 'tot_freq_1', 'tot_freq_2', 'tot_freq_3',
                 'tot_sev_1', 'tot_sev_2', 'tot_sev_3',
                 'tot_agg_1', 'tot_agg_2', 'tot_agg_3', 'freq_moms'
                 ]

    def __init__(self, freq_moms=None):
        # accumulators
        self.agg_1 = self.agg_2 = self.agg_3 = 0
        self.tot_agg_1 = self.tot_agg_2 = self.tot_agg_3 = 0
        self.freq_1 = self.freq_2 = self.freq_3 = 0
        self.tot_freq_1 = self.tot_freq_2 = self.tot_freq_3 = 0
        self.sev_1 = self.sev_2 = self.sev_3 = 0
        self.tot_sev_1 = self.tot_sev_2 = self.tot_sev_3 = 0
        # function to comptue frequency moments, hence can call add_f1s(...)
        self.freq_moms = freq_moms

    def add_fs(self, f1, f2, f3, s1, s2, s3):
        """
        accumulate new moments defined by f and s

        used by Portfolio

        compute agg for the latest values

        :param f1:
        :param f2:
        :param f3:
        :param s1:
        :param s2:
        :param s3:
        :return:
        """
        self.freq_1 = f1
        self.freq_2 = f2
        self.freq_3 = f3

        # load current sev statistics_df
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

        # aggregate_project
        self.agg_1, self.agg_2, self.agg_3 = self.agg_from_fs(f1, f2, f3, s1, s2, s3)

        # finally accumulate the aggregate_project
        self.tot_agg_1, self.tot_agg_2, self.tot_agg_3 = \
            self.cumulate_moments(self.tot_agg_1, self.tot_agg_2, self.tot_agg_3, self.agg_1, self.agg_2, self.agg_3)

    def add_fs2(self, f1, vf, s1, vs):
        """
        accumulate based on first two moments entered as mean and variance - this
        is how questions are generally written.

        """
        f2 = vf + f1 * f1
        s2 = vs + s1 * s1
        self.add_fs(f1, f2, 0., s1, s2, 0.)

    def add_f1s(self, f1, s1, s2, s3):
        """
        accumulate new moments defined by f1 and s - fills in f2, f3 based on
        stored frequency distribution

        used by Aggregate

        compute agg for the latest values


        :param f1:
        :param s1:
        :param s2:
        :param s3:
        :return:
        """

        # fill in the frequency moments and store away
        f1, f2, f3 = self.freq_moms(f1)
        self.add_fs(f1, f2, f3, s1, s2, s3)

    def get_fsa_stats(self, total, remix=False):
        """
        get the current f x s = agg statistics_df and moments
        total = true use total else, current
        remix = true for total only, re-compute freq statistics_df based on total freq 1

        :param total: binary
        :param remix: combine all sevs and recompute the freq moments from total freq
        :return:
        """

        if total:
            if remix:
                # recompute the frequency moments; all local variables
                f1 = self.tot_freq_1
                f1, f2, f3 = self.freq_moms(f1)
                s1, s2, s3 = self.tot_sev_1 / f1, self.tot_sev_2 / f1, self.tot_sev_3 / f1
                a1, a2, a3 = self.agg_from_fs(f1, f2, f3, s1, s2, s3)
                return [f1, f2, f3, *self.static_moments_to_mcvsk(f1, f2, f3),
                        s1, s2, s3, *self.static_moments_to_mcvsk(s1, s2, s3),
                        a1, a2, a3, *self.static_moments_to_mcvsk(a1, a2, a3)]
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

    @staticmethod
    def factorial_to_noncentral(f1, f2, f3):
        # eg. Panjer Willmot p 29, 2.3.13
        nc2 = f2 + f1
        nc3 = f3 + 3 * f2 + f1
        return nc2, nc3

    @staticmethod
    def agg_from_fs(f1, f2, f3, s1, s2, s3):
        """
        aggregate_project moments from freq and sev components


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

    @staticmethod
    def agg_from_fs2(f1, vf, s1, vs):
        """
        aggregate_project moments from freq and sev ex and var x


        :param f1:
        :param vf:
        :param s1:
        :param vs:
        :return:
        """
        f2 = vf + f1 * f1
        s2 = vs + s1 * s1
        a1, a2, a3 = MomentAggregator.agg_from_fs(f1, f2, 0., s1, s2, 0.)
        mw = MomentWrangler()
        mw.noncentral = a1, a2, a3
        # drop skewness
        return mw.stats[:-1]

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
                return MomentAggregator.static_moments_to_mcvsk(self.tot_agg_1, self.tot_agg_2, self.tot_agg_3)
            else:
                return MomentAggregator.static_moments_to_mcvsk(self.agg_1, self.agg_2, self.agg_3)
        elif mom_type == 'freq':
            if total:
                return MomentAggregator.static_moments_to_mcvsk(self.tot_freq_1, self.tot_freq_2, self.tot_freq_3)
            else:
                return MomentAggregator.static_moments_to_mcvsk(self.freq_1, self.freq_2, self.freq_3)
        elif mom_type == 'sev':
            if total:
                return MomentAggregator.static_moments_to_mcvsk(self.tot_sev_1 / self.tot_freq_1,
                                                                self.tot_sev_2 / self.tot_freq_1,
                                                                self.tot_sev_3 / self.tot_freq_1)
            else:
                return MomentAggregator.static_moments_to_mcvsk(self.sev_1, self.sev_2, self.sev_3)

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
    def static_moments_to_mcvsk(ex1, ex2, ex3):
        """
        returns mean, cv and skewness from non-central moments

        :param ex1:
        :param ex2:
        :param ex3:
        :return:
        """
        m = ex1
        var = ex2 - ex1 ** 2
        # rounding errors...
        if np.allclose(var, 0):
            var = 0
        if var < 0:
            logger.error(f'MomentAggregator.static_moments_to_mcvsk | weird var < 0 = {var}; ex={ex1}, ex2={ex2}')
        sd = np.sqrt(var)
        if m == 0:
            cv = np.nan
            logger.warning('MomentAggregator.static_moments_to_mcvsk | encountered zero mean, called with '
                         f'{ex1}, {ex2}, {ex3}')
        else:
            cv = sd / m
        if sd == 0:
            skew = np.nan
        else:
            skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / sd ** 3
        return m, cv, skew

    @staticmethod
    def column_names():
        """
        list of the moment and statistics_df names for f x s = a

        :return:
        """

        return [i + j for i, j in itertools.product(['freq', 'sev', 'agg'], [f'_{i}' for i in range(1, 4)] +
                                                    ['_m', '_cv', '_skew'])]

    def stats_series(self, name, limit, pvalue, remix):
        """
        combine elements into a reporting series
        handles order, index names etc. in one place

        :param name: series name
        :param limit:
        :param pvalue:
        :param remix: called from Aggregate want remix=True to collect mix terms; from Portfolio remix=False
        :return:
        """
        # TODO: needs to be closer link to column_names()
        idx = pd.MultiIndex.from_arrays([['freq'] * 6 + ['sev'] * 6 + ['agg'] * 8,
                                         (['ex1', 'ex2', 'ex3'] + ['mean', 'cv', 'skew']) * 3 + ['limit', 'P99.9e']],
                                        names=['component', 'measure'])
        all_stats = self.get_fsa_stats(total=True, remix=remix)
        p999e = estimate_agg_percentile(*all_stats[15:18], pvalue)
        return pd.Series([*all_stats, limit, p999e], name=name, index=idx)


class MomentWrangler(object):
    """
    Conversion between central, noncentral and factorial moments

    Input any one and ask for any translation.

    Stores moments as noncentral internally

    """

    __slots__ = ['_central', '_noncentral', '_factorial']

    def __init__(self):
        self._central = None
        self._noncentral = None
        self._factorial = None

    @property
    def central(self):
        return self._central

    @central.setter
    def central(self, value):
        self._central = value
        ex1, ex2, ex3 = value
        # p 43, 1.241
        self._noncentral = (ex1, ex2 + ex1 ** 2, ex3 + 3 * ex2 * ex1 + ex1 ** 3)
        self._make_factorial()

    @property
    def noncentral(self):
        return self._noncentral

    @noncentral.setter
    def noncentral(self, value):
        self._noncentral = value
        self._make_factorial()
        self._make_central()

    @property
    def factorial(self):
        return self._factorial

    @factorial.setter
    def factorial(self, value):
        self._factorial = value
        ex1, ex2, ex3 = value
        # p 44, 1.248
        self._noncentral = (ex1, ex2 + ex1, ex3 + 3 * ex2 + ex1)
        self._make_central()

    @property
    def stats(self):
        m, v, c3 = self._central
        sd = np.sqrt(v)
        if m == 0:
            cv = np.nan
        else:
            cv = sd / m
        if sd == 0:
            skew = np.nan
        else:
            skew = c3 / sd ** 3
        #         return pd.Series((m, v, sd, cv, skew), index=('EX', 'Var(X)', 'SD(X)', 'CV(X)', 'Skew(X)'))
        # shorter names are better
        return pd.Series((m, v, sd, cv, skew), index=('ex', 'var', 'sd', 'cv', 'skew'))

    @property
    def mcvsk(self):
        m, v, c3 = self._central
        sd = np.sqrt(v)
        if m == 0:
            cv = np.nan
        else:
            cv = sd / m
        if sd == 0:
            skew = np.nan
        else:
            skew = c3 / sd ** 3
        return m, cv, skew

    def _make_central(self):
        ex1, ex2, ex3 = self._noncentral
        # p 42, 1.240
        self._central = (ex1, ex2 - ex1 ** 2, ex3 - 3 * ex2 * ex1 + 2 * ex1 ** 3)

    def _make_factorial(self):
        """ add factorial from central """
        ex1, ex2, ex3 = self._noncentral
        # p. 43, 1.245 (below)
        self._factorial = (ex1, ex2 - ex1, ex3 - 3 * ex2 + 2 * ex1)


def xsden_to_meancv(xs, den):
    """
    Compute mean and cv from xs and density.

    Consider adding: np.nan_to_num(den)

    :param xs:
    :param den:
    :return:
    """
    xd = xs * den
    ex1 = np.sum(xd)
    ex2 = np.sum(xd * xs)
    sd = np.sqrt(ex2 - ex1 ** 2)
    if ex1 != 0:
        cv = sd / ex1
    else:
        cv = np.nan
    return ex1, cv


def xsden_to_meancvskew(xs, den):
    """
    Compute mean, cv and skewness from xs and density

        Consider adding: np.nan_to_num(den)

        :param xs:
        :param den:
        :return:
        """
    xd = xs * den
    ex1 = np.sum(xd)
    xd *= xs
    ex2 = np.sum(xd)
    ex3 = np.sum(xd * xs)
    mw = MomentWrangler()
    mw.noncentral = ex1, ex2, ex3
    return mw.mcvsk


def tweedie_convert(*, p=None, μ=None, σ2=None, λ=None, α=None, β=None, m=None, cv=None):
    """
    Translate between Tweedie parameters. Input p, μ, σ2 or λ, α, β or  λ, m, cv. Remaining
    parameters are computed and returned in pandas Series.

    p, μ, σ2 are the reproductive parameters, μ is the mean and the variance equals σ2 μ^p
    λ, α, β are the additive parameters; λα/β is the mean, λα(α + 1) / β^2 is the variance
    λ, m, cv specify the compound Poisson with expected claim count λ and gamma with mean m and cv

    In addition, returns p0, the probability mass at 0.
    """

    if μ is None:
        if α is None:
            # λ, m, cv spec directly as compound Poisson
            assert λ is not None and m is not None and cv is not None
            α = 1 / cv ** 2
            β = α / m
        else:
            # λ, α, β in additive form
            assert λ is not None and α is not None and β is not None
            m = α / β
            cv = α ** -0.5
        p = (2 + α) / (1 + α)
        μ = λ * m
        σ2 = λ * α * (α + 1) / β ** 2 / μ ** p
    else:
        # p, μ, σ2 in reproductive form
        assert p is not None and μ is not None and σ2 is not None
        α = (2 - p) / (p - 1)
        λ = μ**(2-p) / ((2-p) * σ2)
        β = λ * α / μ
        m = α / β
        cv = α ** -0.5

    p0 = np.exp(-λ)
    twcv = np.sqrt(σ2 * μ ** p) / μ
    ans = pd.Series([p, μ, σ2, λ, α, β, twcv, m, cv, p0],
                    index=['p', 'μ', 'σ^2', 'λ', 'α', 'β', 'tw_cv', 'sev_m', 'sev_cv', 'p0'])
    return ans

def power_variance_family():
    """
    Graph to illustrate the power variance exponential family distributions.

    Reference: Jørgensen, Bent. 1997. The theory of dispersion models. CRC Press.
    """
    alpha = np.linspace(-2, 2, 101)
    p = (alpha-2) / (alpha-1)
    alphabar = -(alpha+1)
    f, ax = plt.subplots(figsize=(7,7))
    ax.plot(alphabar, p, lw=3)
    ax.set(ylim=[-5,10])
    # ax.grid(lw=.25, c='b', alpha=.5)
    ax.axhline(1, c='k', lw=1)
    ax.axhline(0, c='k', lw=1)
    ax.axvline(-2, c='k', lw=.5)
    ax.axvline(-3/2, c='k', lw=0.5, ls='--')
    ax.axhline(3, c='k', lw=0.5, ls='--')
    ax.axhline(2, c='r', lw=1)
    ax.axvline(-1, c='r', lw=1)
    ax.set(xlabel='$\\bar\\alpha=-(\\alpha+1)$, base jump density is $x^{\\bar\\alpha}$',
           ylabel='Variance power function $p$, $V(\\mu)=\\phi\\mu^p$')
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(-4,10)))

    def ql(x, y, t, dot=True, rhs=None):
        ax.text(x + .05, y+0.2, t)
        if dot:
            ax.plot(x,y, 'rd', ms=5)
        else:
            if rhs is None:
                rhs = x + 1
            ax.plot([x, rhs], [y, y], lw=3)

    ql(-3, 0, 'Normal')
    ql(-3, 6, 'Stable', False)
    ql(-2, 9, 'Cauchy')
    ql(-2, -2, 'Pos Stable', False)
    ql(-1.5, 3, 'Stab 3/2\nIG')
    ql(-1, 2, 'Gamma')
    ql(1, 1, 'Poisson')
    ql(-1, -2, 'Tweedie', False, 1)
    ax.set(title='Power Variance Exponential Family Distributions');


def frequency_examples(n, ν, f, κ, sichel_case, log2, xmax=500):
    """
    Illustrate different frequency distributions and frequency moment
    calculations.

    sichel_case = gamma | ig | ''

    Sample call: ::

        df, ans = frequency_examples(n=100, ν=0.45, f=0.5, κ=1.25,
                                     sichel_case='', log2=16, xmax=2500)

    :param n: E(N) = expected claim count
    :param ν: CV(mixing) = asymptotic CV of any compound aggregate whose severity has a second moment
    :param f: proportion of certain claims, 0 <= f < 1, higher f corresponds to greater skewnesss
    :param κ: (kappa) claims per occurrence
    :param sichel_case: gamma, ig or ''
    :param xmax:
    """

    def ft(x):
        return np.fft.fft(x)

    def ift(x):
        return np.fft.ifft(x)

    def defuzz(x):
        x[np.abs(x) < 5e-16] = 0
        return x

    def ma(x):
        return list(MomentAggregator.static_moments_to_mcvsk(*x))

    def row(ps):
        moms = [(x ** k * ps).sum() for k in (1, 2, 3)]
        stats = ma(moms)
        return moms + stats + [np.nan]

    def noncentral_n_moms_from_mixing_moms(n, c, g):
        """
        c=Var(G), g=E(G^3) return EN, EN2, EN3
        """
        return [n, n * (1 + (1 + c) * n), n * (1 + n * (3 * (1 + c) + n * g))]

    def asy_skew(g, c):
        """
        asymptotic skewnewss
        """
        return [(g - 3 * c - 1) / c ** 1.5]

    ans = pd.DataFrame(columns=['X', 'type', 'EX', 'EX2', 'EX3', 'mean', 'CV', 'Skew', 'Asym Skew'])
    ans = ans.set_index(['X', 'type'])

    N = 1 << log2
    x = np.arange(N, dtype=float)
    z = np.zeros(N)
    z[1] = 1
    fz = ft(z)

    # build poisson for comparison
    dist = 'poisson'
    kernel = n * (fz - 1)
    p = np.real(ift(np.exp(kernel)))
    p = defuzz(p)

    # for poisson c=0 and g=1 (the "mixing" distribution is G identically equal to 1)
    ans.loc[(dist, 'empirical'), :] = row(p)
    temp = noncentral_n_moms_from_mixing_moms(n, 0, 1)
    ans.loc[(dist, 'theoretical'), :] = temp + ma(temp) + [np.inf]
    ans.loc[(dist, 'diff'), :] = ans.loc[(dist, 'empirical'), :] / ans.loc[(dist, 'theoretical'), :] - 1

    # negative binomial
    # Var(G) = c, E(G)=1 so Var(G) = ν^2 = c
    # wikipedia / scipy stats use a and θ for shape and scale
    dist = 'neg bin'
    c = ν * ν  # contagion
    a = 1 / c
    θ = c
    # E(G^3): skew(G) = skew(G') = γ, so E(G-EG)^3 = γν^3, so EG^3 = γν^3 + 3(1+c) - 2 = γν^3 + 3c + 1
    # for Gamma skew = 2 / sqrt(a) = 2ν
    g = 1 + 3 * c + 2 * c * c
    nb = np.real(ift((1 - θ * kernel) ** -a))
    nb = defuzz(nb)

    ans.loc[(dist, 'empirical'), :] = row(nb)
    # this row is generic: it applies to all distributions
    temp = noncentral_n_moms_from_mixing_moms(n, c, g)
    ans.loc[(dist, 'theoretical'), :] = temp + ma(temp) + asy_skew(g, c)
    ans.loc[(dist, 'diff'), :] = ans.loc[(dist, 'empirical'), :] / ans.loc[(dist, 'theoretical'), :] - 1

    # delaporte G = f + G'
    dist = 'delaporte'
    c = ν * ν  # contagion
    a = ((1 - f) / ν) ** 2
    θ = (1 - f) / a
    g = 2 * ν ** 4 / (1 - f) + 3 * c + 1
    delaporte = np.real(ift(np.exp(f * kernel) * (1 - θ * kernel) ** -a))
    delaporte = defuzz(delaporte)

    ans.loc[(dist, 'empirical'), :] = row(delaporte)
    temp = noncentral_n_moms_from_mixing_moms(n, c, g)
    ans.loc[(dist, 'theoretical'), :] = temp + ma(temp) + asy_skew(g, c)
    ans.loc[(dist, 'diff'), :] = ans.loc[(dist, 'empirical'), :] / ans.loc[(dist, 'theoretical'), :] - 1

    # pig
    dist = 'pig'
    c = ν * ν  # contagion
    μ = c
    λ = 1 / μ
    # our param (λ, μ) --> (λ, λμ) in Mathematica and hence skew = γ = 3 * sqrt(μ) in scip py parameterization
    γ = 3 * np.sqrt(μ)
    g = γ * ν ** 3 + 3 * c + 1
    pig = np.real(ift(np.exp(1 / μ * (1 - np.sqrt(1 - 2 * μ ** 2 * λ * kernel)))))
    pig = defuzz(pig)
    #     print(f'PIG parameters μ={μ} and λ={λ}, g={g}, γ={γ}')
    ans.loc[(dist, 'empirical'), :] = row(pig)
    temp = noncentral_n_moms_from_mixing_moms(n, c, g)
    ans.loc[(dist, 'theoretical'), :] = temp + ma(temp) + asy_skew(g, c)
    ans.loc[(dist, 'diff'), :] = ans.loc[(dist, 'empirical'), :] / ans.loc[(dist, 'theoretical'), :] - 1

    # shifted pig
    dist = 'shifted pig'
    c = ν * ν  # contagion
    μ = c / (1 - f) ** 2
    λ = (1 - f) / μ
    γ = 3 * np.sqrt(μ)
    g = γ * ν ** 3 + 3 * c + 1
    shifted_pig = np.real(ift(np.exp(f * kernel) * np.exp(1 / μ * (1 - np.sqrt(1 - 2 * μ ** 2 * λ * kernel)))))
    shifted_pig = defuzz(shifted_pig)

    ans.loc[(dist, 'empirical'), :] = row(shifted_pig)
    temp = noncentral_n_moms_from_mixing_moms(n, c, g)
    ans.loc[(dist, 'theoretical'), :] = temp + ma(temp) + asy_skew(g, c)
    ans.loc[(dist, 'diff'), :] = ans.loc[(dist, 'empirical'), :] / ans.loc[(dist, 'theoretical'), :] - 1

    #  poisson pascal
    # parameters
    dist = 'poisson pascal'
    # solve for local c to hit overall c=ν^2 value input
    c = (n * ν ** 2 - 1 - κ) / κ
    a = 1 / c
    θ = κ * c
    λ = n / κ  # poisson parameter for number of claims
    pois_pascal = np.real(ift(np.exp(λ * ((1 - θ * (fz - 1)) ** -a - 1))))
    pois_pascal = defuzz(pois_pascal)

    ans.loc[(dist, 'empirical'), :] = row(pois_pascal)
    # moments for the PP are different can't use noncentral__nmoms_from_mixing_moms
    g = κ * λ * (
            2 * c ** 2 * κ ** 2 + 3 * c * κ ** 2 * λ + 3 * c * κ ** 2 + 3 * c * κ + κ ** 2 * λ ** 2 + 3 * κ ** 2 * λ + κ ** 2 + 3 * κ * λ + 3 * κ + 1)
    g2 = n * (
            2 * c ** 2 * κ ** 2 + 3 * c * n * κ + 3 * c * κ ** 2 + 3 * c * κ + n ** 2 + 3 * n * κ + 3 * n + κ ** 2 + 3 * κ + 1)
    assert np.allclose(g, g2)
    temp = [λ * κ, n * (κ * (1 + c + λ) + 1), g]
    ans.loc[(dist, 'theoretical'), :] = temp + ma(temp) + [
        np.nan]  # note: hard to interpret this, but for FIXED cv of NB it tends to zero
    ans.loc[(dist, 'diff'), :] = ans.loc[(dist, 'empirical'), :] / ans.loc[(dist, 'theoretical'), :] - 1

    # sichel: find sichel to increase mixing skewness indicated amount from IG skewness
    # ======
    # three flavors
    # 1. sichel        cv, lambda  solve for beta and mu to hit mean and cv given lambda
    #                     lambda = -0.5 = ig; lambda = 0.5 inverse IG
    # 2. sichel.gamma  cv, certain match moments to negative binomial
    # 3. sichel.ig     cv, certain match moments to shifted ig distribution
    dist = 'sichel'
    c = ν * ν  # contagion
    if sichel_case == 'gamma':
        # sichel_case 2: match delaporte moments
        # G = f + G'; E(G') = 1 - f, SD(G) = SD(G') = ν, skew(G') = skew(G)
        # a = ((1 - f) / ν) ** 2
        # FWIW θ = ν / (1 - f)  # (1 - f) / a
        target = np.array([1, ν, 2 * ν / (1 - f)])  # / np.sqrt(a)])
    elif sichel_case == 'ig':
        # match shifted IG moments
        # μ = (ν / (1 - f)) ** 2
        target = np.array([1, ν, 3.0 * ν / (1 - f)])  # np.sqrt(μ)])
    elif sichel_case == '':
        # input lambda, match mean = 1 and cv (input)
        pass
    else:
        raise ValueError("Idiot")

    add_sichel = True

    if sichel_case in ('gamma', 'ig'):
        # need starting parameters (Panjer and Willmost format
        if sichel_case == 'gamma':
            λ = -0.5
        else:
            λ = -0.5
        μ = 1
        β = ν ** 2

        def f(arrIn):
            """
            calibration function to match target mean, cv and skewness (keeps the scale about the same)
            :param arrIn:
            :return:
            """
            μ, β, λ = arrIn
            # mu and beta are positive...
            μ = np.exp(μ)
            β = np.exp(β)
            ex1, ex2, ex3 = np.array([μ ** r * kv(λ + r, μ / β) / kv(λ, μ / β) for r in (1, 2, 3)])
            sd = np.sqrt(ex2 - ex1 * ex1)
            skew = (ex3 - 3 * ex2 * ex1 + 2 * ex1 ** 3) / (sd ** 3)
            return np.array([ex1, sd, skew]) - target

        try:
            params1 = broyden2(f, (np.log(μ), np.log(β), λ), verbose=False, iter=10000,
                               f_rtol=1e-11)  # , f_rtol=1e-9)  , line_search='wolfe'
            params2 = newton_krylov(f, (np.log(μ), np.log(β), λ), verbose=False, iter=10000,
                                    f_rtol=1e-11)  # , f_rtol=1e-9)  , line_search='wolfe'
            if np.sum((params1 - params2) ** 2) > 0.05:
                print(f'Broyden {params1}\nNewton K {params2}')
                m1 = np.sum(np.abs(params1))
                m2 = np.sum(np.abs(params2))
                if m1 < m2:
                    print(f'selecting Broyden {params1}')
                    params = params1
                else:
                    print(f'selecting Newton K {params2}')
                    params = params2
            else:
                print(f'Two estimates similar, selecting Bry {params1}, {params2}')
                params = params1
        except NoConvergence as e:
            print('ERROR: broyden did not converge')
            print(e)
            add_sichel = False
            raise e

    elif sichel_case == '':
        # input lambda match mean = 1 and cv
        λ = κ  # will actually be freq_b
        # need starting parameters (for lamda = -0.5) Panjer Willmot format
        μ = 1
        β = ν ** 2
        target = np.array([1, ν])

        def f(arrIn):
            """
            calibration function to match target mean = 1 and cv
            :param arrIn:
            :return:
            """
            μ, β = arrIn
            # mu and beta are positive...
            μ = np.exp(μ)
            β = np.exp(β)
            ex1, ex2 = np.array([μ ** r * kv(λ + r, μ / β) / kv(λ, μ / β) for r in (1, 2)])
            sd = np.sqrt(ex2 - ex1 * ex1)
            return np.array([ex1, sd]) - target

        try:
            params = broyden2(f, (np.log(μ), np.log(β)), verbose=False, iter=10000,
                              f_rtol=1e-11)  # , f_rtol=1e-9)  , line_search='wolfe'

        except NoConvergence as e:
            print('ERROR: broyden did not converge')
            print(e)
            add_sichel = False
            raise e

    else:
        raise ValueError("Idiot ")

    # if parameters found...
    if add_sichel:
        if sichel_case == '':
            μ, β = params
        else:
            μ, β, λ = params
        μ, β = np.exp(μ), np.exp(β)
        print(f'Sichel params {(μ, β, λ)}')
        # theoretic noncentral moments of the **mixing** distribution = Factorial of N
        # have calibrated to EG=1 so use same kernel n * (z - 1)
        # compute density
        inner = np.sqrt(1 - 2 * β * kernel)
        sichel = np.real(ift(inner ** (-λ) * kv(λ, μ * inner / β) / kv(λ, μ / β)))
        sichel = defuzz(sichel)
        ans.loc[(dist, 'empirical'), :] = row(sichel)
        mw = MomentWrangler()
        junk = [μ ** r * kv(λ + r, μ / β) / kv(λ, μ / β) for r in (1, 2, 3)]
        g = junk[2]  # non central G
        temp = noncentral_n_moms_from_mixing_moms(n, c, g)
        print('Noncentral N from mixing moms            ', temp)
        mw.factorial = junk
        print('Non central N moms                       ', mw.noncentral)
        print('Empirical central N moms                 ', row(sichel))
        ans.loc[(dist, 'theoretical'), :] = temp + ma(temp) + asy_skew(g, c)
        ans.loc[(dist, 'diff'), :] = ans.loc[(dist, 'empirical'), :] / ans.loc[(dist, 'theoretical'), :] - 1

    # ---------------------------------------------------------------------
    # sum of all errors is small
    print(ans.loc[(slice(None), 'diff'), :].abs().sum().sum())
    # assert ans.loc[(slice(None), 'diff'), :].abs().sum().sum() < 1e-6

    # graphics
    df = pd.DataFrame(dict(x=x, poisson=p, pois_pascal=pois_pascal, negbin=nb, delaporte=delaporte, pig=pig,
                           shifted_pig=shifted_pig, sichel=sichel))
    df = df.query(f' x < {xmax} ')
    df = df.set_index('x')
    axiter = axiter_factory(None, 12, aspect=1.414, nr=4)
    all_dist = ['poisson', 'negbin', 'delaporte', 'pig', 'shifted_pig', 'pois_pascal', 'sichel']
    for vars in [all_dist,
                 ['poisson', 'negbin', 'pig', ],
                 ['poisson', 'negbin', 'delaporte', ],
                 ['poisson', 'pig', 'shifted_pig', 'sichel'],
                 ['poisson', 'negbin', 'pois_pascal'],
                 ['poisson', 'delaporte', 'shifted_pig', 'sichel'],
                 ]:
        # keep the same colors
        # FIX TODO
        print('FIX palette')
        # pal = [sns.color_palette("Paired", 7)[i] for i in [all_dist.index(j) for j in vars]]
        df[vars].plot(kind='line', ax=next(axiter)) #, color=pal)
        axiter.ax.set_xlim(0, 4 * n)
        df[vars].plot(kind='line', logy=True, ax=next(axiter), legend=None) # , color='virdis')
    axiter.tidy()
    display(ans.unstack())
    return df, ans


class Answer(dict):
    # TODO replace with collections.namedtuple? Or at least, stop using it!
    def __init__(self, **kwargs):
        """
        Generic answer wrapping class with plotting

        :param kwargs: key=value to wrap
        """
        super().__init__(kwargs)

    def __getattr__(self, item):
        return self[item]

    def __repr__(self):
        return str(self.list())
        # return super().__repr__()

    def __str__(self):
        return self.list()

    def list(self):
        """ List elements """
        return pd.DataFrame(zip(self.keys(),
                                [self.nice(v) for v in self.values()]), columns=['Item', 'Type'])

    def __str__(self):
        return '\n'.join([f'{i[0]:<20s}\t{i[1]}'
                          for i in zip(self.keys(), [self.nice(v) for v in self.values()])
                          ])
    @staticmethod
    def nice(x):
        """ return a nice rep of x """
        if type(x) in [str, float, int]:
            return x
        else:
            return type(x)

    def summary(self):
        """
        just print out the dataframes: horz or vertical as appropriate
        reasonable styling
        :return:
        """

        for k, v in self.items():
            if isinstance(v, pd.core.frame.DataFrame):
                print(f'\n{k}\n{"=" * len(k)}\n')
                if v.shape[1] > 12:
                    display(v.head(5).T)
                else:
                    display(v.head(10))


def log_test():
    """"
    Issue logs at each level
    """
    print('Issuing five messages...')
    for l, n in zip([logger], ['logger']):
        print(n)
        l.debug('A debug message')
        l.info('A info message')
        l.warning('A warning message')
        l.error('A error message')
        l.critical('A critical message')
        print(f'...done with {n}')
    print('...done')


def logger_level(level=30, name='aggregate', show=False):
    """
    Code from common.py

    Change logger level all loggers containing name
    Changing for EVERY logger is a really bad idea,
    you get the endless debug info out of matplotlib
    find_font, for exapmle.

    FWIW, to list all loggers:

        loggers = [logging.getLogger()]  # get the root logger
        loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        loggers

    :param level:
    :return:
    """

    try:
        logging.basicConfig(format='%(asctime)s.%(msecs)03d|%(lineno)4d|%(levelname)-10s| %(name)s, %(funcName)s|  %(message)-s',
                            datefmt='%M:%S')
        loggers = [logging.getLogger()]  # get the root logger
        loggers = loggers + \
            [logging.getLogger(name)
             for name in logging.root.manager.loggerDict]

        # set the level selectively
        for logger in loggers:
            if logger.name.find(name) >= 0:
                logger.setLevel(level)
        if show:
            for logger in loggers:
                print(logger.name, logger.getEffectiveLevel())
    except ValueError as e:
        raise e


def subsets(x):
    """
    all non empty subsets of x, an interable
    """
    return list(itertools.chain.from_iterable(
        itertools.combinations(x, n) for n in range(len(x) + 1)))[1:]


# new graphics methods
def nice_multiple(mx):
    """
    Suggest a nice multiple for an axis with scale 0 to mx. Used by the MultipleLocator in discrete plots,
    where you want an integer multiple. Return 0 to let matplotlib figure the answer. Real issue is stopping
    multiples like 2.5.

    :param mx:
    :return:
    """
    m = mx / 6
    if m < 0:
        return 0

    m = mx // 6
    m = {3: 2, 4: 5, 6: 5, 7: 5, 8: 10, 9: 10}.get(m, m)
    if m < 10:
        return m

    # punt back to mpl for larger values
    return 0

class GreatFormatter(ticker.ScalarFormatter):
    def __init__(self, sci=True, power_range=(-3, 3), offset=True, mathText=False):
        super().__init__(useOffset=offset, useMathText=mathText)
        self.set_powerlimits(power_range)
        self.set_scientific(sci)

    def _set_order_of_magnitude(self):
        super()._set_order_of_magnitude()
        self.orderOfMagnitude = int(3 * np.floor(self.orderOfMagnitude / 3))


def make_mosaic_figure(mosaic, figsize=None, w=3.5*1.333, h=3.5, xfmt='great', yfmt='great',
                       places=None, power_range=(-3, 3), sep='', unit='', sci=True,
                       mathText=False, offset=True, return_array=False):
    """
    make mosaic of axes
    apply format to xy axes
    default engineering format
    default w x h per subplot

    xfmt='d' for default axis formatting, n=nice, e=engineering, s=scientific, g=great
    great = engineering with power of three exponents

    if return_array then the returns are mor comparable with the old axiter_factory

    """

    if figsize is None:
        sm = mosaic.split('\n')
        nr = len(sm)
        nc = len(sm[0])
        figsize = (w * nc, h * nr)

    f = plt.figure(constrained_layout=True, figsize=figsize)
    axd = f.subplot_mosaic(mosaic)

    for ax in axd.values():
        if xfmt[0] != 'd':
            easy_formatter(ax, which='x', kind=xfmt, places=places,
                           power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText, offset=offset)
        if yfmt[0] != 'default':
            easy_formatter(ax, which='y', kind=yfmt, places=places,
                           power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText, offset=offset)

    if return_array:
        return f, np.array(list(axd.values()))
    else:
        return f, axd


def easy_formatter(ax, which, kind, places=None, power_range=(-3, 3), sep='', unit='', sci=True,
                   mathText=False, offset=True):
    """
    set which (x, y, b, both) to kind = sci, eng, nice
    nice = engineering but uses e-3, e-6 etc.
    see docs for ScalarFormatter and EngFormatter


    """

    def make_fmt(kind, places, power_range, sep, unit):
        if kind == 'sci' or kind[0] == 's':
            fm = ticker.ScalarFormatter()
            fm.set_powerlimits(power_range)
            fm.set_scientific(True)
        elif kind == 'eng' or kind[0] == 'e':
            fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
        elif kind == 'great' or kind[0] == 'g':
            fm = GreatFormatter(
                sci=sci, power_range=power_range, offset=offset, mathText=mathText)
        elif kind == 'nice' or kind[0] == 'n':
            fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
            fm.ENG_PREFIXES = {
                i: f'e{i}' if i else '' for i in range(-24, 25, 3)}
        else:
            raise ValueError(f'Passed {kind}, expected sci or eng')
        return fm

    # what to set
    if which == 'b' or which == 'both':
        which = ['xaxis', 'yaxis']
    elif which == 'x':
        which = ['xaxis']
    else:
        which = ['yaxis']

    for w in which:
        fm = make_fmt(kind, places, power_range, sep, unit)
        getattr(ax, w).set_major_formatter(fm)


# styling - greys verions
def style_df(df):
    """
    Style a df similar to pricinginsurancerisk.com styles.

    graph background color is B4C3DC and figure (paler) background is F1F8F#

    Dropped row lines; bold level0, caption

    :param df:
    :return: styled dataframe

    """

    cell_hover = {
        'selector': 'td:hover',
        'props': [('background-color', '#ffffb3')]
    }
    index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: white; background-color: #777777; '
                 'font-weight:bold; border: 1px solid white; text-transform: capitalize; '
                 'text-align:left;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #DDDDDD; color: black;  border: 1px solid #ffffff;'
    }
    center_heading = {
        'selector': 'th.col_heading',
        'props': 'text-align: center;'
    }
    left_index = {
        'selector': '.row_heading',
        'props': 'text-align: left;'
    }
    td = {
        'selector': 'td',
        'props': f'text-align: right;'
    }
    all_styles = [cell_hover, index_names, headers, center_heading,  left_index, td]
    return df.style.set_table_styles(all_styles)


# styling blue to match graphs...not great
# def style_df(df):
#     """
#     Style a df similar to pricinginsurancerisk.com styles.
#
#     graph background color is B4C3DC and figure (paler) background is F1F8F#
#
#     Dropped row lines; bold level0, caption
#
#     :param df:
#     :return: styled dataframe
#
#     """
#
#     cell_hover = {
#         'selector': 'td:hover',
#         'props': [('background-color', '#ffffb3')]
#     }
#     index_names = {
#         'selector': '.index_name',
#         'props': 'font-style: italic; color: black; background-color: white; '
#                  'font-weight:bold; border: 0px solid #a4b3dc; text-transform: capitalize; '
#                  'text-align:left;'
#     }
#     headers = {
#         'selector': 'th:not(.index_name)',
#         'props': 'background-color: #b4c3dc; color: black;  border: 1px solid #ffffff;'
#     }
#     center_heading = {
#         'selector': 'th.col_heading',
#         'props': 'text-align: center;'
#     }
#     left_index = {
#         'selector': '.row_heading',
#         'props': 'text-align: left;'
#     }
#     td = {
#         'selector': 'td',
#         'props': f'text-align: right; '
#     }
#     nrow = {
#         'selector': 'tr:nth-child(even)',
#         'props': 'background-color: #f1f8fe;'
#     }
#     all_styles = [cell_hover, index_names, headers, center_heading, nrow, left_index, td]
#     return df.style.set_table_styles(all_styles)


def friendly(df):
    """
    Attempt to format df "nicely", in a user-friendly manner. Not designed for big dataframes!

    :param df:
    :return:
    """
    def ur(x):
        # simple ultimate renamer
        if type(x) == str:
            sx = x.split("_")
            if len(sx) > 1:
                x = ' '.join([rn(i) for i in sx])
            else:
                x = rn(x).title()

        return x

    def rn(x):
        # specific renamings...
        return {
            'freq': 'Freqency',
            'sev': 'Severity',
            'agg': 'Aggregate',
            'el': 'Expected Loss',
            'm': 'Mean',
            'cv': 'CV', 'skew': 'Skewness', 'kurt': 'Kurtosis'
        }.get(x,x)

    bit = df.rename(index=ur).rename(columns=ur)

    # style like pricinginsurancerisk.com
    # return style_df(bit).format(lambda x: x if type(x)==str else f'{x:,.3f}')
    return bit

class FigureManager():
    def __init__(self, cycle='c', lw=1.5, color_mode='mono', k=0.8, font_size=12,
                 legend_font='small', default_figsize=(5, 3.5)):
        """
        Another figure/plotter manager: manages cycles for color/black and white
        from Great utilities.py, edited and stripped down
        combined with lessons from MetaReddit on matplotlib options for fonts, background
        colors etc.

        Font size was 9 and legend was x-small

        Create figure with common defaults

        cycle = cws
            c - cycle colors
            w - cycle widths
            s - cycle styles
            o - styles x colors, implies csw and w=single number (produces 8 series)

        lw = default line width or [lws] of length 4

        smaller k overall darker lines; colors are equally spaced between 0 and k
        k=0.8 is a reasonable range for four colors (0, k/3, 2k/3, k)

        https://matplotlib.org/3.1.1/tutorials/intermediate/color_cycle.html

        https://matplotlib.org/3.1.1/users/dflt_style_changes.html#colors-in-default-property-cycle

        https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html

        https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        """

        assert len(cycle) > 0

        # this sets a much smaller base fontsize
        # plt.rcParams.update({'axes.titlesize': 'large'})
        # plt.rcParams.update({'axes.labelsize': 'small'})
        # list(map(plt.rcParams.get, ('axes.titlesize', 'font.size')))
        # everything scales off font size
        plt.rcParams['font.size'] = font_size
        # mpl default is medium
        plt.rcParams['legend.fontsize'] = legend_font
        # see https://matplotlib.org/stable/gallery/color/named_colors.html
        self.plot_face_color = 'lightsteelblue'
        self.figure_bg_color = 'aliceblue'
        # graphics set up
        plt.rcParams["axes.facecolor"] = self.plot_face_color
        # note plt.rc lets you set multiple related properties at once:
        plt.rc('legend', fc=self.plot_face_color, ec=self.plot_face_color)
        # is equivalent to two calls:
        # plt.rcParams["legend.facecolor"] = self.plot_face_color
        # plt.rcParams["legend.edgecolor"] = self.plot_face_color
        plt.rcParams['figure.facecolor'] = self.figure_bg_color

        self.default_figsize = default_figsize
        self.plot_colormap_name = 'cividis'

        # fonts: add some better fonts as earlier defaults
        mpl.rcParams['font.serif'] = ['STIX Two Text', 'Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif',
                                      'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L',
                                      'Utopia', 'ITC Bookman',
                                      'Bookman', 'Nimbus Roman No9 L', 'Times', 'Palatino', 'Charter', 'serif']
        mpl.rcParams['font.sans-serif'] = ['Nirmala UI', 'Myriad Pro', 'Segoe UI', 'DejaVu Sans', 'Bitstream Vera Sans',
                                           'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid',
                                           'Arial',
                                           'sans-serif']
        mpl.rcParams['font.monospace'] = ['Ubuntu Mono', 'QuickType II Mono', 'Cascadia Mono', 'DejaVu Sans Mono',
                                          'Bitstream Vera Sans Mono', 'Computer Modern Typewriter', 'Andale Mono',
                                          'Nimbus Mono L', 'Courier New',
                                          'Courier', 'Fixed', 'Terminal', 'monospace']
        mpl.rcParams['font.family'] = 'serif'
        # or
        # plt.rc('font', family='serif')
        # much nicer math font, default is dejavusans
        mpl.rcParams['mathtext.fontset'] = 'stixsans'

        if color_mode == 'mono':
            # https://stackoverflow.com/questions/20118258/matplotlib-coloring-line-plots-by-iteration-dependent-gray-scale
            # default_colors = ['black', 'grey', 'darkgrey', 'lightgrey']
            default_colors = [(i * k, i * k, i * k) for i in [0, 1 / 3, 2 / 3, 1]]
            default_ls = ['solid', 'dashed', 'dotted', 'dashdot']

        elif color_mode == 'cmap':
            # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            norm = mpl.colors.Normalize(0, 1, clip=True)
            cmappable = mpl.cm.ScalarMappable(
                norm=norm, cmap=self.plot_colormap_name)
            mapper = cmappable.to_rgba
            default_colors = list(map(mapper, np.linspace(0, 1, 10)))
            default_ls = ['solid', 'dashed',
                          'dotted', 'dashdot', (0, (5, 1))] * 2
        else:
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                              '#7f7f7f', '#bcbd22', '#17becf']
            default_ls = ['solid', 'dashed',
                          'dotted', 'dashdot', (0, (5, 1))] * 2

        props = []
        if 'o' in cycle:
            n = len(default_colors) // 2
            if color_mode == 'mono':
                cc = [i[1] for i in product(default_ls, default_colors[::2])]
            else:
                cc = [i[1] for i in product(default_ls, default_colors[:n])]
            lsc = [i[0] for i in product(default_ls, default_colors[:n])]
            props.append(cycler('color', cc))
            props.append(
                cycler('linewidth', [lw] * (len(default_colors) * len(default_ls) // 2)))
            props.append(cycler('linestyle', lsc))
        else:
            if 'c' in cycle:
                props.append(cycler('color', default_colors))
            else:
                props.append(
                    cycler('color', [default_colors[0]] * len(default_ls)))
            if 'w' in cycle:
                if type(lw) == int:
                    props.append(
                        cycler('linewidth', [lw] * len(default_colors)))
                else:
                    props.append(cycler('linewidth', lw))
            if 's' in cycle:
                props.append(cycler('linestyle', default_ls))

        # combine all cyclers
        cprops = props[0]
        for c in props[1:]:
            cprops += c

        mpl.rcParams['axes.prop_cycle'] = cycler(cprops)

    def make_fig(self, nr=1, nc=1, figsize=None, xfmt='great', yfmt='great',
                 places=None, power_range=(-3, 3), sep='', unit='', sci=True,
                 mathText=False, offset=True, **kwargs):
        """

        make grid of axes
        apply format to xy axes

        xfmt='d' for default axis formatting, n=nice, e=engineering, s=scientific, g=great
        great = engineering with power of three exponents

        """

        if figsize is None:
            figsize = self.default_figsize

        f, axs = plt.subplots(nr, nc, figsize=figsize,
                              constrained_layout=True, squeeze=False, **kwargs)
        for ax in axs.flat:
            if xfmt[0] != 'd':
                FigureManager.easy_formatter(ax, which='x', kind=xfmt, places=places,
                                             power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText,
                                             offset=offset)
            if yfmt[0] != 'default':
                FigureManager.easy_formatter(ax, which='y', kind=yfmt, places=places,
                                             power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText,
                                             offset=offset)

        if nr * nc == 1:
            axs = axs[0, 0]

        self.last_fig = f
        return f, axs

    __call__ = make_fig

    @staticmethod
    def easy_formatter(ax, which, kind, places=None, power_range=(-3, 3), sep='', unit='', sci=True,
                       mathText=False, offset=True):
        """
        set which (x, y, b, both) to kind = sci, eng, nice
        nice = engineering but uses e-3, e-6 etc.
        see docs for ScalarFormatter and EngFormatter


        """

        def make_fmt(kind, places, power_range, sep, unit):
            if kind == 'sci' or kind[0] == 's':
                fm = ticker.ScalarFormatter()
                fm.set_powerlimits(power_range)
                fm.set_scientific(True)
            elif kind == 'eng' or kind[0] == 'e':
                fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
            elif kind == 'great' or kind[0] == 'g':
                fm = GreatFormatter(
                    sci=sci, power_range=power_range, offset=offset, mathText=mathText)
            elif kind == 'nice' or kind[0] == 'n':
                fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
                fm.ENG_PREFIXES = {
                    i: f'e{i}' if i else '' for i in range(-24, 25, 3)}
            else:
                raise ValueError(f'Passed {kind}, expected sci or eng')
            return fm

        # what to set
        if which == 'b' or which == 'both':
            which = ['xaxis', 'yaxis']
        elif which == 'x':
            which = ['xaxis']
        else:
            which = ['yaxis']

        for w in which:
            fm = make_fmt(kind, places, power_range, sep, unit)
            getattr(ax, w).set_major_formatter(fm)



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
        score[:, j] = np.random.permutation(p)

    # actual correlation of reference (this will be close to, but not equal to, the identity)
    # @ denotes matrix multiplication
    # step 4 and 5
    E = np.linalg.cholesky((score.T @ score) / n)
    # sample with exact desired correlation
    # step 6
    return score @ np.linalg.inv(E.T)


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
    ranks is calibrated to the reference distribution
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
    The result has the same rand correlation as a reference sample with the
    desired linear correlation. Thus the process relies on linear and rank
    correlation (for the reference and the input sample) being close.

    if dof==0 use normal scores; else you mv t

    Sample code:

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
    if dof==0:
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
    if add_total is True:
        df['total'] = df.sum(axis=1)
        df = df.set_index('total')
        df = df.sort_index(ascending=False)

    return df


def rearrangement_algorithm_max_VaR(df, p=0, tau=1e-3, max_n_iter=100):
    """
    Implementation of the Rearragement Algorithm (RA). Determines the worst p-VaR
    rearrangement of the input variables.

    For loss random variables p is usually close to 1.

    Embrechts, Paul, Giovanni Puccetti, and Ludger Ruschendorf, 2013, Model uncertainty and
    VaR aggregation, Journal of Banking and Finance 37, 2750–2764.

    :param df: Input DataFrame containing samples from each marginal. RA will only combine the
    top 1-p proportion of values from each marginal.
    :param p: If p=0 assume df has already truncated to the top p values (for each marginal).
    Otherwise truncate each at the int(1-p * len(df))
    :param tau: simulation tolerance
    :param max_iter: maximum number of iterations to attempt
    :return:

    ## Worst-Case VaR

    Worst value at risk arrangement of marginals.

    See [Actuarial Review article](https://ar.casact.org/the-re-arrangement-algorithm).

    Worst TVaR / Variance arrangement of bivariate data = pair best with worst, second best with second worst, etc., called **countermonotonic** arangement.

    More than 2 marginals: can't *make everything negatively correlated with everything else*. If $X$ and $Y$ are negatively correlated and $Y$ and $Z$ are negatively correlated then $X$ and $Z$ will be positively correlated.

    Next best attempt: make $X$ countermonotonic to $Y+Z$, $Y$ to $X+Z$ and $Z$ to $X+Y$. Basis of **rearrangement algorithm**.


    ### The Rearrangement Algorithm

    1. Randomly permute each column of $X$, the $N\\times d$ matrix of top $1-p$ observations
    2. Loop
        * Create a new matrix $Y$ as follows. For column $j=1,\\dots,d$
            - Create a temporary matrix $V_j$ by deleting the $j$th column of $X$
            - Create a column vector $v$ whose $i$th element equals the sum of the elements in the $i$th row of $V_j$
            - Set the $j$th column of $Y$ equal to the $j$th column of $X$ arranged to have the opposite order to $v$, i.e. the largest element in the $j$th column of $X$ is placed in the row of $Y$ corresponding to the smallest element in $v$, the second largest with second smallest, etc.
        * Compute $y$, the $N\\times 1$ vector with $i$th element equal to the sum of the elements in the $i$th row of $Y$ and let $y^*=\min(y)$ be the smallest element of $y$ and compute $x^*$ from $X$ similarly
        * If $y^*-x^* \\ge \\epsilon$  then set $X=Y$ and repeat the loop
        * If $y^*-x^* < \\epsilon$  then break from the loop

    3. The arrangement $Y$ is an approximation to the worst $\text{VaR}_p$ arrangement of $X$.

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
        df_out[m] = np.random.permutation(sorted_marginals[m])

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
            print(f'Iteration {n_iter:d}\t{v:5.3e}\tChg\t{chg_var:5.3e}')
        if n_iter > max_n_iter:
            print("ERROR: not converging...breaking")
            break

    df_out['total'] = df_out.sum(axis=1)
    print(f'Ending VaR\t{v:7.5e}\ns lower {df_out.total.min():7.5e}')
    return df_out.sort_values('total')


def make_corr_matrix(vine_spec):
    """
    Make a correlation matrix from a vine specification, https://en.wikipedia.org/wiki/Vine_copula

    Vine spececification is

    row 0: correl of X0...Xn-1 with X0
    row 1: correl of X1....Xn-1 with X1 given X0
    row 2: correl of X2....Xn-1 with X2 given X0, X1
    etc.

    For example ::

        vs = np.array([[1,.2,.2,.2,.2],[0,1,.3,.3,.3], [0,0,1,.4, .4], [0,0,0,1,.5], [0,0,0,0,1]])
        make_corr_matrix(vs)

    Key fact is the partial correlation forumula

        rho(X,Y|Z) = (rho(X,Y) - rho(X,Z)rho(Y,Z)) / sqrt((1-rho(X,Z)^2)(1-rho(Y,Z)^2))

    and therefore

        rho(X,Y) =  rho(X,Z)rho(Y,Z) + rho(X,Y|Z) sqrt((1-rho(XZ)^2)(1-rho(YZ)^2))

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
        A = np.random.rand(n,n)**p
    else:
        A = 1 - 2 * np.random.rand(n,n)**p
    np.fill_diagonal(A, 1)

    return make_corr_matrix(A)