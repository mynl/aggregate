import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
from IPython.core.display import HTML, display
import logging
import logging.handlers
import itertools
# import os
import seaborn as sns
from scipy.special import kv
from scipy.optimize import broyden2, newton_krylov
from scipy.optimize.nonlin import NoConvergence
from io import StringIO
import re
from pathlib import Path

# import warnings

# logging
# see https://docs.python.org/3.7/howto/logging.html
# and https://docs.python.org/3.7/howto/logging-cookbook.html
# LOGFILE = os.path.join(os.path.split(__file__)[0], 'aggregate.log')
LOGFILE = Path.home() / '.agglog/agg.main.logger.log'
# RLOGFILE = Path.home() / '.agglog/r.agg.main.logger.log'
# check it exists
LOGFILE.parent.mkdir(exist_ok=True, parents=True)

# approved method is call (__name__)
logger = logging.getLogger('aggregate')
logger.setLevel(logging.INFO)

# rh = logging.handlers.RotatingFileHandler(RLOGFILE, maxBytes=100000, backupCount=10)
rh = logging.FileHandler(LOGFILE)
rh.setLevel(logging.INFO)
rh_formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)-10s | %(funcName)s (l. %(lineno) 5d) | %(message)s')
rh.setFormatter(rh_formatter)

# to stderr
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
ch_formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)-10s | %(funcName)s (l. %(lineno) 5d) | %(message)s')
ch.setFormatter(ch_formatter)

# add loggers
logger.addHandler(rh)
logger.addHandler(ch)

# dev logger to stderr - all levels
# used for messages during development
# dev_logger = logging.getLogger('aggdev')
# dev_logger.setLevel(logging.DEBUG)
# sh = logging.StreamHandler()
# sh_formatter = logging.Formatter(
#     '%(asctime)s | %(name)s | %(levelname)-10s | %(funcName)s (l. %(lineno)d) | %(message)s')
# sh.setFormatter(sh_formatter)
# dev_logger.addHandler(sh)

logger.info('aggregate_project.__init__ | New Aggregate Session started')


# display
# def qd(df, max_rows=10):
#     """
#     generic quick display of data frame df
#     aware of likely column names with appropriate format
#     for each
#     """
#     if max_rows == -1:
#         max_rows = df.shape[0]
#         if max_rows > 1000:
#             max_rows = 1000
#     display(df.head(max_rows).style.format(get_fmts(df)))


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


def tidy_agg_program(txt):
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
    compute a decent rounded bucket from an input float bs

    if bs > 1 round to 2, 5, 10,

    if bs < 1 find the smallest power of two greater than 1/bs

    """

    if bs >= 1:
        # rounded bs
        rbs = np.round(bs, 0)
        if rbs == 1:
            return 2.0
        elif rbs <= 5:
            return 5.0
        elif rbs <= 10:
            return 10.0
        else:
            return 10 * round_bucket(bs / 10)

    if bs < 1:
        # inverse bs
        bsi = 1 / bs
        nbs = 1
        while nbs < bsi:
            nbs <<= 1
        nbs >>= 1
        return 1. / nbs

# axis management
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
    def good_grid(n):
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
            c = 4
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
    plt.tight_layout(rect=[0, 0, 1, 0.96])


# general nonsense
def insurability_triangle(figsize=(10, 3)):
    """
    Illustrate the insurability triangle...

    ::

        λ = A / L
        ROE = (P-L) / (A - P) = (P/L - 1) / (A/L - P/L)
            = (1/LR - 1) / (λ - 1/LR) = (1 - LR) / (λLR - 1)

    Hence

    ::

        δ = ROE / (1 + ROE) = (1 - LR) / [LR (λ - 1)]

    :return:
    """
    f, axs = plt.subplots(1, 3, figsize=figsize)
    it = iter(axs.flatten())
    λs = [1.5, 2, 3, 5, 10, 25, 50, 100]
    # up to user to manage colors
    # sns.set_palette(sns.cubehelix_palette(len(λs)))

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
    Purely accumulates moments
    Used by Portfolio
    Not frequency aware
    makes report_ser df and statistics_df

    Internal variables agg, sev, frqe, tot = running total, 1, 2, 3 = noncentral moments, E(X^k)

    :param freq_moms: function of one variable returning first three noncentral moments of the underlying frequency distribution

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

    def _make_central(self):
        ex1, ex2, ex3 = self._noncentral
        # p 42, 1.240
        self._central = (ex1, ex2 - ex1 ** 2, ex3 - 3 * ex2 * ex1 + 2 * ex1 ** 3)

    def _make_factorial(self):
        """ add factorial from central """
        ex1, ex2, ex3 = self._noncentral
        # p. 43, 1.245 (below)
        self._factorial = (ex1, ex2 - ex1, ex3 - 3 * ex2 + 2 * ex1)


# class qd(object):
#     """
#     quick display for dictionaries and Pandas dataframes, with some sensible number defaults
#     experimental
#
#     """
#
#     # Set CSS properties for th elements in dataframe
#     th_props = [
#         ('font-size', '11px'),
#         ('text-align', 'center'),
#         ('font-weight', 'bold'),
#         ('color', '#6d6d6d'),
#         ('background-color', '#f7f7f9')
#     ]
#
#     # Set CSS properties for td elements in dataframe
#     td_props = [
#         ('font-size', '10px'),
#         ('text-align', 'left')
#     ]
#
#     # Set table styles
#     styles = [
#         dict(selector="th", props=th_props),
#         dict(selector="td", props=td_props)
#     ]
#
#     cm = sns.light_palette("green", as_cmap=True)
#
#     def __init__(self, d):
#         self.x = d
#
#     def _repr_html_(self):
#         if isinstance(self.x, dict):
#             return pd.DataFrame(self.x, index=[len(self.x)])._repr_html_()
#         if isinstance(self.x, list) or isinstance(self.x, tuple) and len(self.x) == 2:
#             if isinstance(self.x[1], dict):
#                 return f'<h2>{self.x[0]}</h2><br>' + qd(self.x[1])._repr_html_()
#         elif isinstance(self.x, pd.DataFrame):
#             # do a bit of styling
#             num_cols = self.x.select_dtypes(np.number).columns
#             fmt = {}
#             for a, b in zip(self.x.columns, self.x.dtypes):
#                 if np.issubdtype(b, np.number):
#                     m, s = self.x[a].agg([np.mean, np.std])
#                     x = np.abs(m) + 3 * s
#                     if abs(x) > self.x[a].max():
#                         x = self.x[a].max()
#                     if x < 10:
#                         fmt[a] = '{:7.3f}'
#                     elif x < 1000:
#                         fmt[a] = '{:7.1f}'
#                     elif x < 10e6:
#                         fmt[a] = '{:12,.1f}'
#                     else:
#                         fmt[a] = '{:12.3e}'
#                 else:
#                     fmt[a] = '{:}'
#
#             return (self.x.style
#                     .background_gradient(cmap=cm, subset=num_cols)
#                     .highlight_max(subset=num_cols)
#                     #   .set_caption('This is a custom caption.')
#                     .format(fmt)
#                     .set_table_styles(styles))._repr_html_()
#         else:
#             return repr(self.x)


def xsden_to_meancv(xs, den):
    """
    compute mean and cv from xs and density

    consider adding: np.nan_to_num(den)

    :param xs:
    :param den:
    :return:
    """
    ex1 = np.sum(xs * den)
    ex2 = np.sum(xs ** 2 * den)
    sd = np.sqrt(ex2 - ex1 ** 2)
    if ex1 != 0:
        cv = sd / ex1
    else:
        cv = np.nan
    return ex1, cv


def frequency_examples(n, ν, f, κ, sichel_case, log2, xmax=500, **kwds):
    """
    Illustrate different frequency distributions and frequency moment
    calculations.

    sichel_case = gamma | ig | ''

    n = E(N) = expected claim count
    ν = CV(mixing) = asymptotic CV of any compound aggregate whose severity has a second moment
    f = proportion of certain claims, 0 <= f < 1, higher f corresponds to greater skewnesss
    κ = claims per occurrence
    g_mult = adjust EG^3 of Sichel above/below standard PIG
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
        pal = [sns.color_palette("Paired", 7)[i] for i in [all_dist.index(j) for j in vars]]
        df[vars].plot(kind='line', ax=next(axiter), color=pal)
        axiter.ax.set_xlim(0, 4 * n)
        df[vars].plot(kind='line', logy=True, ax=next(axiter), legend=None, color=pal)
    axiter.tidy()
    display(ans.unstack())
    return df, ans


class Answer(dict):
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


def subsets(x):
    """
    all non empty subsets of x, an interable
    """
    return list(itertools.chain.from_iterable(
        itertools.combinations(x, n) for n in range(len(x) + 1)))[1:]