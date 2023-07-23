from collections import namedtuple
from cycler import cycler
import decimal
from functools import lru_cache
from io import BytesIO
import itertools
from itertools import product
import logging.handlers
import logging
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numbers import Number
import numpy as np
import pandas as pd
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
import re
import scipy.stats as ss
import scipy.fft as sft
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import broyden2, newton_krylov, brentq
from scipy.optimize.nonlin import NoConvergence
from scipy.special import kv, binom, loggamma
from scipy.stats import multivariate_t
from IPython.core.display import HTML, Markdown, display, Image as ipImage, SVG as ipSVG

from .constants import *
import aggregate.random_agg as ar


logger = logging.getLogger(__name__)

GCN = namedtuple('GCN', ['gross', 'net', 'ceded'])

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
    Simple text version of pprint with line breaking
    """
    return pprint_ex(txt, split=60)

def pprint_ex(txt, split=0, html=False):
    """
    Try to format an agg program. This is difficult because of dfreq and dsev, optional
    reinsurance, etc. Go for a simple approach of removing unnecessary spacing
    and removing notes. Notes can be accessed from the spec that is always to hand.

    For long programs use split=60 or so, they are split at appropriate points.

    Best to use html = True to get colorization.

    :param txt: program text input
    :param split: if > 0 split lines at this length
    :param html: if True return html (via pygments) , else return text
    """
    ans = []
    # programs come in as multiline
    txt = txt.replace('\n\tagg', ' agg')
    for t in txt.split('\n'):
        clean = re.sub(r'[ \t]+', ' ', t.strip())
        clean = re.sub(r' note\{[^}]*\}', '', clean)
        if split > 0 and len(clean) > split:
            clean = re.sub(
                r' ((dfreq )([0-9]+ )|([0-9]+ )(claims?|premium|loss|exposure)'
                r'|d?sev|dfreq|occurrence|agg|aggregate|wts?|mixed|poisson|fixed)',
                           r'\n  \1', clean)
        if clean[:4] == 'port':
            # put in extra tabs at agg for portfolios
            sc = clean.split('\n')
            clean = sc[0] + '\n' + '\n'.join([i if i[:5] == '  agg' else '  ' + i for i in sc[1:]])
        ans.append(clean)
    ans = '\n'.join(ans)
    if html is True:
        # ans = f'<p><code>{ans}\n</code></p>'
        # notes = re.findall('note\{([^}]*)\}', txt)
        # for i, n in enumerate(notes):
        #     ans += f'<p><small>Note {i+1}. {n}</small><p>'
        # use pygments to colorize
        agg_lex = get_lexer_by_name('agg')
        # remove extra spaces
        txt = re.sub(r'[ \t\n]+', ' ', txt.strip())
        ans = HTML(highlight(txt, agg_lex, HtmlFormatter(style='friendly', full=False)))
    return ans


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
    locft = sft.rfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
    # tilt
    # valeus per https://stackoverflow.com/questions/71706387/finding-fft-gives-keyerror-aligned-pandas
    if tilt is not None:
        zt = z * tilt
    else:
        zt = z
    if type(zt) != np.ndarray:
        zt = zt.to_numpy()
    # padding handled by the ft routine
    # temp = np.hstack((z, np.zeros_like(z)))
    return locft(zt, len(z) << padding)


def ift(z, padding, tilt):
    """
    ift that strips out padding and adjusts for tilt

    :param z:
    :param padding:
    :param tilt:
    :return:
    """
    locift = sft.irfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
    if type(z) != np.ndarray:
        temp = locift(z.to_numpy())
    else:
        temp = locift(z)
    # unpad
    if padding != 0:
        temp = temp[0:len(temp) >> padding]
    # untilt
    if tilt is not None:
        temp /= tilt
    return temp
    # return temp[0:int(len(temp) / 2)]

# def ft(z, padding, tilt):
#     """
#     fft with padding and tilt
#     padding = n makes vector 2^n as long
#     n=1 doubles (default)
#     n=2 quadruples
#     tilt is passed in as the tilting vector or None: easier for the caller to have a single instance
#
#     :param z:
#     :param padding: = 1 doubles
#     :param tilt: vector of tilt values
#     :return:
#     """
#     # locft = np.fft.fft
#     locft = np.fft.rfft
#     if z.shape != (len(z),):
#         raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
#     # tilt
#     if tilt is not None:
#         zt = z * tilt
#     else:
#         zt = z
#     # pad
#     if padding > 0:
#         # temp = np.hstack((z, np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)))
#         pad_len = zt.shape[0] * ((1 << padding) - 1)
#         temp = np.hstack((zt, np.zeros(pad_len)))
#     else:
#         temp = zt
#     # temp = np.hstack((z, np.zeros_like(z)))
#     return locft(temp)
#
#
# def ift(z, padding, tilt):
#     """
#     ift that strips out padding and adjusts for tilt
#
#     :param z:
#     :param padding:
#     :param tilt:
#     :return:
#     """
#     # locift = np.fft.ifft
#     locift = np.fft.irfft
#     if z.shape != (len(z),):
#         raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
#     temp = locift(z)
#     # unpad
#     # temp = temp[0:]
#     if padding != 0:
#         temp = temp[0:len(temp) >> padding]
#     # untilt
#     if tilt is not None:
#         temp /= tilt
#     return temp
#     # return temp[0:int(len(temp) / 2)]


def mu_sigma_from_mean_cv(m, cv):
    """
    lognormal parameters
    """
    cv = np.array(cv)
    m = np.array(m)
    sigma = np.sqrt(np.log(cv*cv + 1))
    mu = np.log(m) - sigma**2 / 2
    return mu, sigma

ln_fit = mu_sigma_from_mean_cv

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
            logger.log(WL, f'utils sln_fit | shift > m, {shift} > {m}, too extreme skew {skew}')
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

def gamma_fit(m, cv):
    """

    """
    alpha = cv**-2
    beta = m / alpha
    return alpha, beta


def approximate_work(m, cv, skew, name, agg_str, note, approx_type, output):
    """
    Does the work for Portfolio.approximate and Aggregate.approximate. See their documentation.

    :param output: scipy - frozen scipy.stats continuous rv object; agg_decl
      sev_decl - DecL program for severity (to substituate into an agg ; no name)
      sev_kwargs - dictionary of parameters to create Severity
      agg_decl - Decl program agg T 1 claim sev_decl fixed
      any other string - created Aggregate object
    """
    if approx_type == 'norm':
        sd = m*cv
        if output == 'scipy':
            return ss.norm(loc=m, scale=sd)
        sev = {'sev_name': 'norm', 'sev_scale': sd, 'sev_loc': m}
        decl = f'{sd} @ norm 1 # {m} '

    elif approx_type == 'lognorm':
        mu, sigma = mu_sigma_from_mean_cv(m, cv)
        sev = {'sev_name': 'lognorm', 'sev_a': sigma, 'sev_scale': np.exp(mu)}
        if output == 'scipy':
            return ss.lognorm(sigma, scale=np.exp(mu))
        decl = f'{np.exp(mu)} * lognorm {sigma} '

    elif approx_type == 'gamma':
        shape = cv ** -2
        scale = m / shape
        if output == 'scipy':
            return ss.gamma(shape, scale=scale)
        sev = {'sev_name': 'gamma', 'sev_a': shape, 'sev_scale': scale}
        decl = f'{scale} * gamma {shape} '

    elif approx_type == 'slognorm':
        shift, mu, sigma = sln_fit(m, cv, skew)
        if output == 'scipy':
            return ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
        sev = {'sev_name': 'lognorm', 'sev_a': sigma, 'sev_scale': np.exp(mu), 'sev_loc': shift}
        decl = f'{np.exp(mu)} * lognorm {sigma} + {shift} '

    elif approx_type == 'sgamma':
        shift, alpha, theta = sgamma_fit(m, cv, skew)
        if output == 'scipy':
            return ss.gamma(alpha, loc=shift, scale=theta)
        sev = {'sev_name': 'gamma', 'sev_a': alpha, 'sev_scale': theta, 'sev_loc': shift}
        decl = f'{theta} * gamma {alpha} + {shift} '

    else:
        raise ValueError(f'Inadmissible approx_type {approx_type} passed to fit')

    if output == 'agg_decl':
        agg_str += decl
        agg_str += ' fixed'
        return agg_str
    elif output == 'sev_kwargs':
        return sev
    elif output == 'sev_decl':
        return decl
    else:
        from . distributions import Aggregate
        return Aggregate(**{'name': name, 'note': note,
                            'exp_en': 1, **sev, 'freq_name': 'fixed'})


def estimate_agg_percentile(m, cv, skew, p=0.999):
    """
    Come up with an estimate of the tail of the distribution based on the three parameter fits, ln and gamma

    Updated Nov 2022 with a way to estimate p based on lognormal results. How far in the
    tail you need to go to get an accurate estimate of the mean. See 2_x_approximation_error
    in the help.

    Retain p param for backwards compatibility.

    :param m:
    :param cv:
    :param skew:
    :param p: if > 1 converted to 1 - 10**-n
    :return:
    """

    # p_estimator = interp1d([0.53294, 0.86894, 1.9418, 7.3211, 22.738, 90.012, 457.14, 2981],
    #                        [3,       4,       5,      7,      8,      9,      11,     12],
    #                        assume_sorted=True, bounds_error=False, fill_value=(3, 13))
    # p = 1 - 10**-p_estimator(cv)

    if np.isinf(cv):
        raise ValueError('Infinite variance passed to estimate_agg_percentile')

    if p > 1:
        p = 1 - 10 ** -p

    pn = pl = pg = 0
    if skew <= 0:
        # neither sln nor sgamma works, use a normal
        # for negative skewness the right tail will be thin anyway so normal not outrageous
        fzn = ss.norm(scale=m * cv, loc=m)
        pn = fzn.isf(1 - p)
    elif not np.isinf(skew):
        shift, mu, sigma = sln_fit(m, cv, skew)
        fzl = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
        shift, alpha, theta = sgamma_fit(m, cv, skew)
        fzg = ss.gamma(alpha, scale=theta, loc=shift)
        pl = fzl.isf(1 - p)
        pg = fzg.isf(1 - p)
    else:
        mu, sigma = ln_fit(m, cv)
        fzl = ss.lognorm(sigma, scale=np.exp(mu))
        alpha, theta = gamma_fit(m, cv)
        fzg = ss.gamma(alpha, scale=theta)
        pl = fzl.isf(1 - p)
        pg = fzg.isf(1 - p)
    # throw in a mean + 3 sd approx too...
    return max(pn, pl, pg, m * (1 + ss.norm.isf(1 - p) * cv))


def round_bucket(bs):
    """
    Compute a decent rounded bucket from an input float ``bs``. ::

        if bs > 1 round to 2, 5, 10, ...

        elif bs < 1 find the smallest power of two greater than bs

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
            rbs = np.round(bs, -int(np.log10(bs)))
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

    The reinsurance functions are piecewise linear functions from 0 to inf which
    kinks as needed to express the ceded loss as a function of subject (gross) loss.

    For example, if ``reins_list = [(1, 10, 0), (0.5, 30, 20)]`` the program is 10 x 10 and
    15 part of 30 x 20 (share=0.5). This requires nodes at 0, 10, 20, 50, and inf.

    It is easiest to make the ceder function. Ceded loss at subject loss at x equals
    the sum of the limits below x plus the cession to the layer in which x lies. The
    variable ``base`` keeps track of the layer, ``h`` of the sum (height) of lower layers.
    ``xs`` tracks the knot points, ``ys`` the values.

    ::

         Break (xs)   Ceded (ys)
              0            0
             10            0
             20           10
             50           25
            inf           25


    For example:
    ::

        %%sf 1 2

        c, n, x, y = make_ceder_netter([(1, 10, 10), (0.5, 30, 20), (.25, np.inf, 50)], debug=True)

        xs = np.linspace(0,250, 251)
        ys = c(xs)

        ax0.plot(xs, ys)
        ax0.plot(xs, xs, ':C7')
        ax0.set(title='ceded')

        ax1.plot(xs, xs-ys)
        ax1.plot(xs, xs, 'C7:')
        ax1.set(title='net')

    :param reins_list: a list of (share of, limit, attach), e.g., (0.5, 3, 2) means 50% share of 3x2
        or, equivalently, 1.5 part of 3 x 2. It is better to store share rather than part
        because it still works if limit == inf.
    :param debug: if True, return layer function xs and ys in addition to the interpolation functions.
    :return: netter and ceder functions; optionally debug information.
    """
    # poor mans inf
    INF = 1e99
    h = 0
    base = 0
    xs = [0]
    ys = [0]
    for (share, y, a) in reins_list:
        # part of = share of times limit
        if np.isinf(y):
            y = INF
        p = share * y
        if a > base:
            # moved to new layer, write out left-hand knot point
            xs.append(a)
            ys.append(h)
        # increment height
        h += p
        # write out right-hand knot points
        xs.append(a + y)
        ys.append(h)
        # update left-hand end
        base += (a + y)
    # if not at infinity, stay flat from base to end
    if base < INF:
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


def lognorm_approx(ser):
    """
    Lognormal approximation to series, index = loss values, values = density.
    """
    m, cv = xsden_to_meancv(ser.index, ser.values)
    mu, sigma = mu_sigma_from_mean_cv(m, cv)
    fz = ss.lognorm(sigma, scale=np.exp(mu))
    return fz


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
            logger.info('MomentAggregator.static_moments_to_mcvsk | encountered zero mean, called with '
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
        try:
            p999e = estimate_agg_percentile(*all_stats[15:18], pvalue)
        except ValueError:
            # if no cv this is a value error
            p999e = np.inf
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

    Note: cannot rely on pd.Series[-1] to work... it depends on the index.
    xs could be an index
    :param xs:
    :param den:
    :return:
    """
    pg = 1 - den.sum()
    xd = xs * den
    if isinstance(xs, np.ndarray):
        xsm = xs[-1]
    elif isinstance(xs, pd.Series):
        xsm = xs.iloc[-1]
    else:
        xsm = np.array(xs)[-1]
    ex1 = np.sum(xd) + pg * xsm
    # logger.log(WL, f'tail mass mean adjustment {pg * xsm}')
    ex2 = np.sum(xd * xs) + pg * xsm ** 2
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
    pg = 1 - den.sum()
    xd = xs * den
    if isinstance(xs, np.ndarray):
        xsm = xs[-1]
        bs = xs[1] - xs[0]
    elif isinstance(xs, pd.Series):
        xsm = xs.iloc[-1]
        bs = xs.iloc[1] - xs.iloc[0]
    else:
        _ = np.array(xs)
        xsm = _[-1]
        bs = _[1] - _[0]
    xsm = xsm + bs
    ex1 = np.sum(xd) + pg * xsm
    # logger.log(WL, f'tail mass mean adjustment {pg * xsm}')
    xd *= xs
    ex2 = np.sum(xd) + pg * xsm ** 2
    ex3 = np.sum(xd * xs) + pg * xsm ** 3
    mw = MomentWrangler()
    mw.noncentral = ex1, ex2, ex3
    return mw.mcvsk


def tweedie_convert(*, p=None, μ=None, σ2=None, λ=None, α=None, β=None, m=None, cv=None):
    """
    Translate between Tweedie parameters. Input p, μ, σ2 or λ, α, β or  λ, m, cv. Remaining
    parameters are computed and returned in pandas Series.

    p, μ, σ2 are the reproductive parameters, μ is the mean and the variance equals σ2 μ^p
    λ, α, β are the additive parameters; λαβ is the mean, λα(α + 1) β^2 is the variance
    (α is the gamma shape and β is the scale).
    λ, m, cv specify the compound Poisson with expected claim count λ and gamma with mean m and cv

    In addition, returns p0, the probability mass at 0.
    """

    if μ is None:
        if α is None:
            # λ, m, cv spec directly as compound Poisson
            assert λ is not None and m is not None and cv is not None
            α = 1 / cv ** 2
            β = m / α
        else:
            # λ, α, β in additive form
            assert λ is not None and α is not None and β is not None
            m = α * β
            cv = α ** -0.5
        p = (2 + α) / (1 + α)
        μ = λ * m
        σ2 = λ * α * (α + 1) * β ** 2 / μ ** p
    else:
        # p, μ, σ2 in reproductive form
        assert p is not None and μ is not None and σ2 is not None
        α = (2 - p) / (p - 1)
        λ = μ**(2-p) / ((2-p) * σ2)
        β = μ / (λ * α)
        m = α * β
        cv = α ** -0.5

    p0 = np.exp(-λ)
    twcv = np.sqrt(σ2 * μ ** p) / μ
    ans = pd.Series([μ, p, σ2, λ, α, β, twcv, m, cv, p0],
                    index=['μ', 'p', 'σ^2', 'λ', 'α', 'β', 'tw_cv', 'sev_m', 'sev_cv', 'p0'])
    return ans


def tweedie_density(x, *, p=None, μ=None, σ2=None, λ=None, α=None, β=None, m=None, cv=None):
    """
    Exact density of Tweedie distribution from series expansion.
    Use any parameterization and convert between them with Tweedie convert.
    Coded for clarity and flexibility not speed. See ``tweedie_convert``
    for parameterization.

    """
    pars = tweedie_convert(p=p, μ=μ, σ2=σ2, λ=λ, α=α, β=β, m=m, cv=cv)
    λ = pars['λ']
    α = pars['α']
    β = pars['β']
    if x == 0:
        return np.exp(-λ)
    # reasonable max n from normal approx to Poisson
    maxn = λ + 4 * λ ** 0.5
    logl = np.log(λ)
    logx = np.log(x)
    logb = np.log(β)
    const = -λ - x / β
    ans = 0.0
    for n in range(1, 2000):
        log_term = (const +
                    + n * logl +
                    + (n * α - 1) * logx +
                    - loggamma(n+1) +
                    - loggamma(n * α) +
                    - n * α * logb)
        ans += np.exp(log_term)
        if n > maxn or (n >λ and log_term < -227):
            break
    return ans


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
    # Maybe not, namedtuples are immutable
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

    def __str__(self):
        return self.list()

    def list(self):
        """ List elements """
        return pd.DataFrame(zip(self.keys(),
                                [self.nice(v) for v in self.values()]),
                            columns=['Item', 'Type']).set_index('Item')

    _repr_html_ = list

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


class LoggerManager():
    def __init__(self, level, name='aggregate'):
        """
        Manage all the aggregate loggers: toggle levels
        Put lm = LoggerManager(10) at the start of a function.
        When it goes out of scope it puts the level back
        where it was.

        TODO: make work on a per logger basis!

        """
        self.level = level
        self.loggers = [v for k, v in logging.root.manager.loggerDict.items()
                        if isinstance(v, logging.Logger) and k.find(name) >= 0 and v.getEffectiveLevel() != level]

        if len(self.loggers) > 0:
            self.base_level = self.loggers[0].getEffectiveLevel()
            for l in self.loggers:
                l.setLevel(level)

    def __del__(self):
        for l in self.loggers:
            logger.info(f'Putting logger level back to {self.base_level} from {self.level}')
            l.setLevel(self.base_level)


def logger_level(level=30, name='aggregate', verbose=False):
    """
    Code from common.py

    Change logger level all loggers containing name
    Changing for EVERY logger is a really bad idea,
    you get the endless debug info out of matplotlib
    find_font, for exapmle.

    FWIW, to list all loggers:
    ::

        loggers = [logging.getLogger()]  # get the root logger
        loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        loggers

    :param level:
    :return:
    """

    try:
        # logging.basicConfig(format='%(asctime)s.%(msecs)03d|%(lineno)4d|%(levelname)-10s| %(name)s.%(funcName)s|  %(message)-s',
        logging.basicConfig(format='line %(lineno)4d|%(levelname)-10s| %(name)s.%(funcName)s|  %(message)-s',
                            datefmt='%M:%S')
        loggers = [logging.getLogger()]  # get the root logger
        loggers = loggers + \
            [logging.getLogger(name)
             for name in logging.root.manager.loggerDict]

        # set the level selectively
        for logger in loggers:
            if logger.name.find(name) >= 0:
                logger.setLevel(level)
        if verbose:
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


def make_mosaic_figure(mosaic, figsize=None, w=FIG_W, h=FIG_H, xfmt='great', yfmt='great',
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


def knobble_fonts():
    """
    Not sure we should get into this...

    See FigureManager in Great or common.py

    https://matplotlib.org/3.1.1/tutorials/intermediate/color_cycle.html

    https://matplotlib.org/3.1.1/users/dflt_style_changes.html#colors-in-default-property-cycle

    https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

    https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html

    https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib

    """

    # this sets a much smaller base fontsize
    # everything scales off font size
    plt.rcParams['font.size'] = FONT_SIZE

    # mpl default is medium
    plt.rcParams['legend.fontsize'] = LEGEND_FONT

    # graphics set up
    plt.rcParams["axes.facecolor"] = PLOT_FACE_COLOR
    # note plt.rc lets you set multiple related properties at once:
    plt.rc('legend', fc=PLOT_FACE_COLOR, ec=PLOT_FACE_COLOR)
    plt.rcParams['figure.facecolor'] = FIGURE_BG_COLOR

    plot_colormap_name = 'cividis'

    # fonts: add some better fonts as earlier defaults
    mpl.rcParams['font.serif'] = ['STIX Two Text', 'Times New Roman', 'DejaVu Serif']
    mpl.rcParams['font.sans-serif'] = ['Nirmala UI', 'Myriad Pro', 'Segoe UI', 'DejaVu Sans']
    mpl.rcParams['font.monospace'] = ['Ubuntu Mono', 'QuickType II Mono', 'Cascadia Mono', 'DejaVu Sans Mono']
    # this matches html output better
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['mathtext.fontset'] = 'stixsans'
    pd.options.display.width = 120

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
            'cv': 'CV', 'skew': 'Skewness', 'kurt': 'Kurtosis',
            'L': 'Exp Loss', 'P': 'Premium', 'PQ': 'Leverage',
            'LR': 'Loss Ratio',
            'M': 'Margin', 'Q': 'Capital', 'a': 'Assets',
            'Roe': 'ROE'
        }.get(x,x)

    bit = df.rename(index=ur).rename(columns=ur)

    # style like pricinginsurancerisk.com
    # return style_df(bit).format(lambda x: x if type(x)==str else f'{x:,.3f}')
    return bit

class FigureManager():
    def __init__(self, cycle='c', lw=1.5, color_mode='mono', k=0.8, font_size=12,
                 legend_font='small', default_figsize=(FIG_W, FIG_H)):
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
        self.plot_face_color = PLOT_FACE_COLOR
        self.figure_bg_color = FIGURE_BG_COLOR

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
        score[:, j] = ar.RANDOM.permutation(p)

    # actual correlation of reference (this will be close to, but not equal to, the identity)
    # @ denotes matrix multiplication
    # step 4 and 5
    E = np.linalg.cholesky((score.T @ score) / n)
    # sample with exact desired correlation
    # step 6
    return score @ np.linalg.inv(E.T)

@lru_cache()
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
    array is calibrated to the reference distribution
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
    The result has the same rank correlation as a reference sample with the
    desired linear correlation. Thus, the process relies on linear and rank
    correlation (for the reference and the input sample) being close.

    if dof==0 use normal scores; else you mv t

    Sample code:
    ::

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
    if dof == 0:
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
    if add_total:
        df['total'] = df.sum(axis=1)
        df = df.set_index('total')
        df = df.sort_index(ascending=False)

    return df


def block_iman_conover(unit_losses, intra_unit_corrs, inter_unit_corr, as_frame=False):
    """
    Apply Iman Conover to the unit loss blocks in ``unit_losses`` with correlation matrices in ``intra``.

    Then determine the ordering for the unit totals with correlation ``inter``.

    Re-order each unit, row by row, so that the totals have the desired correlation structure, but
    leaving the intra unit correlation unchanged.

    ``unit_losses = [np.arrays or pd.Series]`` of losses by subunit within units, without totals

    ``len(unit_losses) == len(intra_unit corrs)``

    For simplicity all normal copula; can add other later if required.

    No totals input or output anywhere.

    ``if as_frame`` then a dataframe version returned, for auditing.

    Here is some tester code, using great.test_df to make random unit losses. Vary num_units and
    num_sims as required.

    ::

        def bic_tester(num_units=3, num_sims=10000):
            from aggregate import random_corr_matrix
            # from great import test_df

            # create samples
            R = range(num_units)
            unit_losses = [test_df(num_sims, 3 + i) for i in R]
            totals = [u.sum(1) for u in unit_losses]

            # manual dataframe to check against
            manual = pd.concat(unit_losses + totals, keys=[f'Unit_{i}' for i in R] + ['Total' for i in R], axis=1)

            # for input to method
            unit_losses = [i.to_numpy() for i in unit_losses]
            totals = [i.to_numpy() for i in totals]

            # make corrs
            intra_unit_corrs = [random_corr_matrix(i.shape[1], p=.5, positive=True) for i in unit_losses]
            inter_unit_corr = random_corr_matrix(len(totals), p=1, positive=True)

            # apply method
            bic = block_iman_conover(unit_losses, intra_unit_corrs, inter_unit_corr, True)

            # extract frame answer, put col names back
            bic.frame.columns = manual.columns
            dm = bic.frame

            # achieved corr
            for i, target in zip(dm.columns.levels[0], intra_unit_corrs + [inter_unit_corr]):
                print(i)
                print((dm[i].corr() - target).abs().max().max())
                # print(dm[i].corr() - target)

            # total corr across subunits
            display(dm.drop(columns=['Total']).corr())

            # total corr across subunits
            display(dm.drop(columns=['Total']).corr())

            return manual, bic, intra_unit_corrs, inter_unit_corr

        manual, bic, intra, inter = bic_tester(3, 10000)

    """

    if isinstance(unit_losses, dict):
        unit_losses = unit_losses.values()

    if isinstance(intra_unit_corrs, dict):
        intra_unit_corrs = intra_unit_corrs.values()

    # shuffle unit losses
    # IC returns a dataframe
    unit_losses = [iman_conover(l, c, dof=0, add_total=False).to_numpy() for l, c in zip(unit_losses, intra_unit_corrs)]

    # extract totals
    totals = [l.sum(1) for l in unit_losses]
    totals = np.vstack(totals)

    # apply the interunit correlation to totals: this code copies iman_conover because we want
    # to keep the same ordering matrices

    # block shuffle units; this code is IC by hand to keep track of R and apply to the units
    d, n = totals.shape

    # "square root" of "variance"
    # step 7
    C = np.linalg.cholesky(inter_unit_corr)

    # make a perfectly uncorrelated reference: noise function = steps 1-6; product is step 8 (transposed)
    N = ic_noise(n, d) @ C.T

    # required ordering of marginals determined by reference sample, step 9
    R = ic_rank(N)

    # re-order totals and the corresponding unit losses
    for i, (u, t) in enumerate(zip(unit_losses, totals)):
        r = np.argsort(t)
        unit_losses[i] = u[r]
        totals[i] = t[r]

    # put into the desired IC ordering,
    for i, (u, t, r) in enumerate(zip(unit_losses, totals, R.T)):
        totals[i] = t[r]
        unit_losses[i] = u[r]

    # assembled sample
    combined = np.hstack(unit_losses)

    if as_frame:
        fr = pd.concat((pd.DataFrame(combined), pd.DataFrame(totals.T)), axis=1, keys=['units', 'totals'])

    else:
        fr = None

    BlockImanConover = namedtuple('BlockImanConover', 'totals combined frame')
    ans = BlockImanConover(totals, combined, fr)

    return ans


def rearrangement_algorithm_max_VaR(df, p=0, tau=1e-3, max_n_iter=100):
    """
    Implementation of the Rearragement Algorithm (RA). Determines the worst p-VaR
    rearrangement of the input variables.

    For loss random variables p is usually close to 1.

    Embrechts, Paul, Giovanni Puccetti, and Ludger Ruschendorf, 2013, *Model uncertainty and
    VaR aggregation*, Journal of Banking and Finance 37, 2750–2764.

    **Worst-Case VaR**

    Worst value at risk arrangement of marginals.

    See `Actuarial Review article <https://ar.casact.org/the-re-arrangement-algorithm>`_.

    Worst TVaR / Variance arrangement of bivariate data = pair best with worst, second best with
    second worst, etc., called **countermonotonic** arangement.

    More than 2 marginals: can’t *make everything negatively correlated with
    everything else*. If :math:`X` and :math:`Y` are negatively correlated
    and :math:`Y` and :math:`Z` are negatively correlated then :math:`X` and
    :math:`Z` will be positively correlated.

    Next best attempt: make :math:`X` countermonotonic to :math:`Y+Z`,
    :math:`Y` to :math:`X+Z` and :math:`Z` to :math:`X+Y`. Basis of
    **rearrangement algorithm**.

    *The Rearrangement Algorithm*

    1. Randomly permute each column of :math:`X`, the :math:`N\\times d`
       matrix of top :math:`1-p` observations
    2. Loop

       -  Create a new matrix :math:`Y` as follows. For column
          :math:`j=1,\\dots,d`

          -  Create a temporary matrix :math:`V_j` by deleting the
             :math:`j`\ th column of :math:`X`
          -  Create a column vector :math:`v` whose :math:`i`\ th element
             equals the sum of the elements in the :math:`i`\ th row of
             :math:`V_j`
          -  Set the :math:`j`\ th column of :math:`Y` equal to the
             :math:`j`\ th column of :math:`X` arranged to have the opposite
             order to :math:`v`, i.e. the largest element in the
             :math:`j`\ th column of :math:`X` is placed in the row of
             :math:`Y` corresponding to the smallest element in :math:`v`,
             the second largest with second smallest, etc.

       -  Compute :math:`y`, the :math:`N\\times 1` vector with
          :math:`i`\ th element equal to the sum of the elements in the
          :math:`i`\ th row of :math:`Y` and let :math:`y^*=\min(y)` be the
          smallest element of :math:`y` and compute :math:`x^*` from
          :math:`X` similarly
       -  If :math:`y^*-x^* \\ge \\epsilon` then set :math:`X=Y` and repeat
          the loop
       -  If :math:`y^*-x^* < \\epsilon` then break from the loop

    3. The arrangement :math:`Y` is an approximation to the worst
       :math:`\text{VaR}_p` arrangement of :math:`X`.

    :param df: Input DataFrame containing samples from each marginal. RA will only combine the
        top 1-p proportion of values from each marginal.
    :param p: If ``p==0`` assume df has already truncated to the top p values (for each marginal).
        Otherwise truncate each at the ``int(1-p * len(df))``
    :param tau: simulation tolerance
    :param max_iter: maximum number of iterations to attempt
    :return: the top 1-p values of the rearranged DataFrame
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
        df_out[m] = ar.RANDOM.permutation(sorted_marginals[m])

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
            logger.info(f'Iteration {n_iter:d}\t{v:5.3e}\tChg\t{chg_var:5.3e}')
        if n_iter > max_n_iter:
            logger.error("ERROR: not converging...breaking")
            break

    df_out['total'] = df_out.sum(axis=1)
    logger.info(f'Ending VaR\t{v:7.5e}\ns lower {df_out.total.min():7.5e}')
    return df_out.sort_values('total')


def make_corr_matrix(vine_spec):
    r"""
    Make a correlation matrix from a vine specification, https://en.wikipedia.org/wiki/Vine_copula.

    A vine spececification is::

        row 0: correl of X0...Xn-1 with X0
        row 1: correl of X1....Xn-1 with X1 given X0
        row 2: correl of X2....Xn-1 with X2 given X0, X1
        etc.

    For example ::

        vs = np.array([[1,.2,.2,.2,.2],
                       [0,1,.3,.3,.3],
                       [0,0,1,.4, .4],
                       [0,0,0,1,.5],
                       [0,0,0,0,1]])
        make_corr_matrix(vs)

    Key fact is the partial correlation forumula

    .. math::

        \rho(X,Y|Z) = \frac{(\rho(X,Y) - \rho(X,Z)\rho(Y,Z))}{\sqrt{(1-\rho(X,Z)^2)(1-\rho(Y,Z)^2)}}

    and therefore

    .. math::

        \rho(X,Y) =  \rho(X,Z)\rho(Y,Z) + \rho(X,Y|Z) \sqrt((1-\rho(XZ)^2)(1-\rho(YZ)^2))

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
        A = ar.RANDOM.random((n, n))**p
    else:
        A = 1 - 2 * ar.RANDOM.random((n, n))**p
    np.fill_diagonal(A, 1)

    return make_corr_matrix(A)


def show_fig(f, format='svg', **kwargs):
    """
    Save a figure so it can be placed precisely in output. Used by Underwriter.show to
    interleaf tables and plots.

    :param f: a plt.Figure
    :param format: svg or png
    :param kwargs: passed to savefig
    """
    bio = BytesIO()
    f.savefig(bio, format=format, **kwargs)
    bio.seek(0)
    if format == 'png':
        display(ipImage(bio.read()))
    elif format == 'svg':
        display(ipSVG(bio.read()))
    else:
        raise ValueError(f'Unknown type {format}')
    plt.close(f)


def partial_e(sev_name, fz, a, n):
    """
    Compute the partial expected value of fz. Computing moments is a bottleneck, so you
    want analytic computation for the most commonly used types.

    Exponential (for mixed exponentials) implemented separate from gamma even though it
    is a special case.

    .. math:

        \int_0^a x^k fz.pdf(x)dx

    for k=0,...,n as a np.array

    To do: beta? weibull? Burr? invgamma, etc.

    :param sev_name: scipy.stats name for distribution
    :param fz: frozen scipy.stats instance
    :param a: double, limit for integral
    :param n: int, power
    :return: partial expected value
    """

    if sev_name not in ['lognorm', 'gamma', 'pareto', 'expon']:
        raise NotImplementedError(f'{sev_name} NYI for analytic moments')

    if a == 0:
        return [0] * (n+1) # for k in range(n+1)]

    if sev_name == 'lognorm':
        m = fz.stats('m')
        sigma = fz.args[0]
        mu = np.log(m) - sigma**2 / 2
        ans = [np.exp(k * mu + (k * sigma)**2 / 2) *
               (ss.norm.cdf((np.log(a) - mu - k * sigma**2)/sigma) if a < np.inf else 1.0)
               for k in range(n+1)]
        return ans

    elif sev_name == 'expon':
        # really needed for MEDs
        # expon is gamma with shape = 1
        scale = fz.stats('m')
        shape = 1.
        lgs = loggamma(shape)
        ans = [scale ** k * np.exp(loggamma(shape + k) - lgs) *
               (ss.gamma(shape + k, scale=scale).cdf(a) if a < np.inf else 1.0)
               for k in range(n + 1)]
        return ans

    elif sev_name == 'gamma':
        shape = fz.args[0]
        scale = fz.stats('m') / shape
        # magic ingredient is the norming constant
        # c = lambda sh: 1 / (scale ** sh * gamma(sh))
        # therefore c(shape)/c(shape+k) = scale**k * gamma(shape + k) / gamma(shape)
        # = scale ** k * exp(loggamma(shape + k) - loggamma(shape)) to avoid errors
        ans = [scale ** k * np.exp(loggamma(shape + k) - loggamma(shape)) *
               (ss.gamma(shape + k, scale=scale).cdf(a) if a < np.inf else 1.0)
               for k in range(n + 1)]
        return ans

    elif sev_name == 'pareto':
        # integrate xf(x) even though nx^n-1 S(x) may be more obvious
        # former fits into the overall scheme
        # a Pareto defined by agg is like so: ss.pareto(2.5, scale=1000, loc=-1000)
        α = fz.args[0]
        λ = fz.kwds['scale']
        loc = fz.kwds.get('loc', 0.0)
        # regular Pareto is scale=lambda, loc=-lambda, so this has no effect
        # single parameter Pareto is scale=lambda, loc=0
        # these formulae for regular pareto, hence
        if λ + loc != 0:
            logger.log(WL, 'Pareto not shifted to x>0 range...using numeric moments.')
            return partial_e_numeric(fz, a, n)
        ans = []
        # will return inf if the Pareto does not have the relevant moments
        # TODO: formula for shape=1,2,3
        for k in range(n + 1):
            b = [α * (-1) ** (k - i) * binom(k, i) * λ ** (k + α - i) *
                 ((λ + a) ** (i - α) - λ ** (i - α)) / (i - α)
                 for i in range(k + 1)]
            ans.append(sum(b))
        return ans


def partial_e_numeric(fz, a, n):
    """
    Simple numerical integration version of partial_e for auditing purposes.

    """
    ans = []
    for k in range(n+1):
        temp = quad(lambda x: x ** k * fz.pdf(x), 0, a)
        if temp[1] > 1e-4:
            logger.warning('Potential convergence issues with numerical integral')
        ans.append(temp[0])
    return ans


def moms_analytic(fz, limit, attachment, n, analytic=True):
    """
    Return moments of :math:`E[(X-attachment)^+ \wedge limit]^m`
    for m = 1,2,...,n.

    To check:
    ::

        # fz = ss.lognorm(1.24)
        fz = ss.gamma(6.234, scale=100)
        # fz = ss.pareto(3.4234, scale=100, loc=-100)

        a1 = moms_analytic(fz, 50, 1234, 3)
        a2 = moms_analytic(fz, 50, 1234, 3, False)
        a1, a2, a1-a2, (a1-a2) / a1


    :param fz: frozen scipy.stats instance
    :param limit: double, limit (layer width)
    :param attachment: double, limit
    :param n: int, power
    :param analytic: if True use analytic formula, else numerical integrals
    """
    # easy
    if limit == 0:
        return np.array([0.] * n)

    # don't know how robust this will be...
    sev_name = str(fz.__dict__['dist']).split('.')[-1].split('_')[0]

    # compute and store the partial_e
    detachment = attachment + limit
    if analytic is True:
        pe_attach = partial_e(sev_name, fz, attachment, n)
        pe_detach = partial_e(sev_name, fz, detachment, n)
    else:
        pe_attach = partial_e_numeric(fz, attachment, n)
        pe_detach = partial_e_numeric(fz, detachment, n)

    ans1 = np.array([sum([(-1) ** (m - k) * binom(m, k) * attachment ** (m - k) * (pe_detach[k] - pe_attach[k])
                          for k in range(m + 1)])
                     for m in range(n + 1)])

    if np.isinf(limit):
        ans2 = np.zeros_like(ans1)
    else:
        ans2 = np.array([limit ** m * fz.sf(detachment) for m in range(n+1)])

    ans = ans1 + ans2

    return ans


def qd(*argv, accuracy=3, align=True, trim=True, **kwargs):
    """
    Endless quest for a robust display format!

    Quick display (qd) a list of objects.
    Dataframes handled in text with reasonable defaults.
    For use in documentation.

    :param: argv: list of objects to print
    :param: accuracy: number of decimal places to display
    :param: align: if True, align columns at decimal point (sEngFormatter)
    :kwargs: passed to pd.DataFrame.to_string for dataframes only. e.g., pass dict of formatters by column.

    """
    from .distributions import Aggregate
    from .portfolio import Portfolio
    # ff = sEngFormatter(accuracy=accuracy - (2 if align else 0), min_prefix=0, max_prefix=12, align=align, trim=trim)
    ff = kwargs.pop('ff', lambda x: f'{x:.5g}')
    # split output
    for x in argv:
        if isinstance(x, (Aggregate, Portfolio)):
            if 'Err CV(X)' in x.describe.columns:
                qd(x.describe.drop(columns=['Err CV(X)']).fillna(''), accuracy=accuracy, **kwargs)
            else:
                # object not updated
                qd(x.describe.fillna(''), accuracy=accuracy, **kwargs)
            bss = 'na' if x.bs == 0 else (f'{x.bs:.0f}' if x.bs >= 1 else f'1/{1/x.bs:.0f}')
            vr = x.explain_validation()
            print(f'log2 = {x.log2}, bandwidth = {bss}, validation: {vr}.')
        elif isinstance(x, pd.DataFrame):
            # 100 line width matches rtd html format
            args = {'line_width': 100,
                    'max_cols': 35,
                    'max_rows': 25,
                    'float_format': ff,
                    # needs to be larger for text output
                    # 'max_colwidth': 10,
                    'sparsify': True,
                    'justify': None
                    }
            args.update(kwargs)
            print()
            print(x.to_string(**args))
            # print(x.to_string(formatters={c: f for c in x.columns}))
        elif isinstance(x, pd.Series):
            args = {'max_rows': 25,
                    'float_format': ff,
                    'name': True
                    }
            args.update(kwargs)
            print()
            print(x.to_string(**args))
        elif isinstance(x, int):
            print(x)
        elif isinstance(x, Number):
            print(ff(x))
        else:
            print(x)


def qdp(df):
    """
    Quick describe with nice percentiles and cv for a dataframe.
    """
    d = df.describe()
    # replace with non-sample sd
    d.loc['std'] = df.std(ddof=0)
    d.loc['cv'] = d.loc['std'] / d.loc['mean']
    return d


def mv(x, y=None):
    """
    Nice display of mean and variance for Aggregate or Portfolios or
    entered values.

    R style function, no return value.

    :param x: Aggregate or Portfolio or float
    :param y: float, if x is a float
    :return: None
    """
    from .distributions import Aggregate
    from .portfolio import Portfolio
    if y is None and isinstance(x, (Aggregate, Portfolio)):
        print(f'mean     = {x.agg_m:.6g}')
        print(f'variance = {x.agg_var:.7g}')
        print(f'std dev  = {x.agg_sd:.6g}')
    else:
        print(f'mean     = {x:.6g}')
        print(f'variance = {y:.7g}')
        print(f'std dev  = {y**.5:.6g}')


class sEngFormatter:
    """
    Formats float values according to engineering format inside a range
    of exponents, and standard scientific notation outside.

    Uses the same number of significant digits throughout.
    Optionally aligns at decimal point. That takes up more horizontal
    space but produces easier to read output.

    Based on matplotlib.ticker.EngFormatter and pandas EngFormatter.
    Converts to scientific notation outside (smaller) range of prefixes.
    Uses same number of significant digits?

    Testers::

        sef1 = sEngFormatter(accuracy=5, min_prefix=0, max_prefix=12, align=True, trim=True)
        sef2 = sEngFormatter(accuracy=5, min_prefix=0, max_prefix=12, align=False, trim=True)
        sef3 = sEngFormatter(accuracy=5, min_prefix=0, max_prefix=12, align=True, trim=False)
        sef4 = sEngFormatter(accuracy=5, min_prefix=0, max_prefix=12, align=False, trim=False)
        test = [1.234 * 10**n for n in range(-20,20)]
        test = [-i for i in test] + test
        for sef in [sef1, sef2, sef3, sef4]:
            print('\n'.join([sef(i) for i in test]))
            print('\n\n')
        print('===============')
        test = [1.234 * 10**n + 3e-16 for n in range(-20,20)]
        test = [-i for i in test] + test
        for sef in [sef1, sef2, sef3, sef4]:
            print('\n'.join([sef(i) for i in test]))

    """

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "μ",
        -3: "m",
        0: " ",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }
    regex = re.compile(r'^([^.]*?)\.([0-9]*?)(0+)([yzafpnμmkMGTPEZY]*)( *)$')

    def __init__(self, accuracy, min_prefix=-6, max_prefix=12, align=True, trim=True):
        self.accuracy = accuracy
        self.align = align
        self.trim = trim
        self.ENG_PREFIXES = {k: v for k, v in sEngFormatter.ENG_PREFIXES.items() if min_prefix <= k <= max_prefix}

    def __call__(self, num):
        """
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:

        :param num: the value to represent
        :type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        :return: engineering formatted string
        """
        dnum = decimal.Decimal(str(num))

        if decimal.Decimal.is_nan(dnum):
            return "NaN"

        if decimal.Decimal.is_infinite(dnum):
            return "inf"

        sign = 1

        if dnum < 0:  # pragma: no cover
            sign = -1
            dnum = -dnum

        if dnum != 0:
            pow10 = decimal.Decimal(int(math.floor(dnum.log10() / 3) * 3))
            # extra accuracy
            if dnum >= 1:
                ex_acc = 2 - (int(dnum.log10()) % 3)
            else:
                ex_acc = abs(int(dnum.log10())) % 3
        else:
            pow10 = decimal.Decimal(0)
            ex_acc = 0
        int_pow10 = int(pow10)

        sci = False
        if pow10 > max(self.ENG_PREFIXES.keys()) or pow10 < min(self.ENG_PREFIXES.keys()):
            sci = True

        if sci:
            if 0.01 <= dnum < 10:
                # in this case need the max ex_acc=2 to get the right number of sig figs
                if self.align:
                    format_str = f"{{dnum: {(7 + self.accuracy + 2)}.{self.accuracy + 2:d}f}} "
                else:
                    format_str = f'{{dnum: .{self.accuracy + 2}f}}'
                formatted = format_str.format(dnum=sign * dnum)
                formatted = self.remove_trailing_zeros(formatted, num, self.accuracy + 2)
            else:
                format_str = f'{{dnum: .{self.accuracy + 2}e}}'
                formatted = format_str.format(dnum=sign * dnum)
        else:
            prefix = self.ENG_PREFIXES.get(int_pow10, ' ')
            mant = sign * dnum / (10**pow10)
            if self.align:
                if self.accuracy + ex_acc == 0:
                    # if .0f then you don't get the period. hence have to add that, width 6 + . = 7
                    format_str = f"{{mant: 6.0f}}.{{prefix}}"
                else:
                    format_str = f"{{mant: {(7+self.accuracy+ex_acc)}.{self.accuracy + ex_acc:d}f}}{{prefix}}"
                formatted = format_str.format(mant=mant, prefix=prefix)
                formatted = self.remove_trailing_zeros(formatted, num, self.accuracy + ex_acc)
            else:
                format_str = f"{{mant: .{self.accuracy:d}f}}{{prefix}}"
                formatted = format_str.format(mant=mant, prefix=prefix)
                formatted = self.remove_trailing_zeros(formatted, num, self.accuracy)
            if ex_acc == 0:
                formatted += '  '
            elif ex_acc == 1:
                formatted += ' '

        return formatted

    def remove_trailing_zeros(self, str_x, x, dps):
        """
        Remove trailing zeros from a string representation ``str_x`` of a number ``x``.
        The number of decimal places is ``dps``. If the number is in scientific notation
        then there is no change.  Eg with dps == 3, 1.2000 becomes 1.2, 1.000 becomes 1,
        but 1.200 when x=1.20000001 is unchanged.

        """
        if self.trim is True:
            if abs(x - np.round(x, dps)) < 5 * np.finfo(float).eps:
                try:
                    return self.regex.sub(self.regex_replace, str_x)
                except TypeError as e:
                    print(e)
                    return str_x
            else:
                return str_x
        else:
            return str_x

    def regex_replace(self, x):
        # x is a match object (1 before period, 2 after period, 3 trailing zeros, 4 prefix, 5 spaces)
        # regex = re.compile(r'^([^.]*?)\.([0-9]*?)(0+)([yzafpnμmkMGTPEZY]*)( *)$')

        # return x  # f'{x[1]}.{x[2]}{x[4]}{" "*(len(x[3])+len(x[5]))}'
        if len(x[4]) == 0:
            # no prefix, kMGT etc.
            return f'{x[1]}.{x[2]}{" "*(len(x[3])+len(x[5]))}'
        else:
            if x[2] == '':
                # turns 2.M into 2.0M
                # return f'{x[1]}.0{x[4]}{" "*(len(x[3])+len(x[5]))}'
                # turns 2.M into 2M
                return f'{x[1]}{x[4]}{" "*(1+len(x[3])+len(x[5]))}'
            else:
                return f'{x[1]}.{x[2]}{x[4]}{" "*(len(x[3])+len(x[5]))}'


def picks_work(attachments, layer_loss_picks, xs, sev_density, n=1, sf=None, debug=False):
    """
    Adjust the layer unconditional expected losses to target. You need int xf(x)dx, but
    that is fraught when f is a mixed distribution. So we only use the int S version.
    ``fz`` was initially a frozen continuous distribution; but adjusted to sf function
    and dropped need for pdf function.

    See notes for how the parts are defined. Notice that::

        np.allclose(p.layers.v - p.layers.f, p.layers.l - p.layers.e)

    is true.

    :param attachments: array of layer attachment points, in ascending order (bottom to top). a[0]>0
    :param layer_loss_picks: Target means. If ``len(layer_loss_picks)==len(attachments)`` then the bottom layer, 0 to a[0],
      is added. Can be input as unconditional layer severity (i.e., :math:`\\mathbb{E}[(X-a)^+\wedge y]`) or as the
      layer loss pick (i.e., :math:`\\mathbb{E}[(X-a)^+\wedge y]'times n` where *n* is the number of ground-up (to the
      insurer) claims. Multiplying and dividing by :math:`S(a)` shows this equals conditional severity in the layer
      times the number of claims in the layer.) Actuaries usually estimate the loss pick to the layer in pricing. When
      called from :class:`Aggregate` the number of ground up claims is known.
    :param en: ground-up expected claims. Target is divided by ``en``.
    :param xs: x values for discretization
    :param sev_density: Series of existing severity density from Aggregate.
    :param sf: cdf function for the severity distribution.
    :param debug: if True, return debug information (layers, density with adjusted probs, audit
      of layer expected values.
    """

    # want xs, attachments, and sev_density to be numpy arrays
    xs = np.array(xs)
    attachments = np.array(attachments)
    # target is the unconditional layer expected loss, E[(X-a)^+ ^ y]
    target = np.array(layer_loss_picks) / n
    # print(n, layer_loss_picks, target)
    sev_density = np.array(sev_density)
    # figure bucket size
    bs = xs[1] - xs[0]

    # dataframe of adjusted probabilties, starts here
    density = pd.DataFrame({'x': xs, 'p': sev_density}).set_index('x', drop=False)
    fill_value = max(0, 1. - density.p.sum())
    density['S'] = density.p.shift(-1, fill_value=fill_value)[::-1].cumsum()

    # numerical integrals - these match
    layers = pd.DataFrame(columns=['a', 'lev', 'int_fdx', 'aS', 'S'], index=range(1, 1+len(attachments)),
                          dtype=float)
    for i, x in enumerate(attachments):
        ix = density.loc[0:x-bs, 'S'].sum() * bs
        ix2 = density.loc[0:x-bs, ['x', 'p']].prod(axis=1).sum()
        layers.loc[i+1, :] = [x, ix, ix2, x * density.loc[x, 'S'] if x < np.inf else 0.0, density.loc[x, 'S']]

    # prob of loss in layer
    layers['p'] = layers.S.shift(1, fill_value=1) - layers.S
    # unconditional expected loss in layer
    layers['l'] = layers.lev - layers.lev.shift(1, fill_value=0)
    layers.index.name = 'layer'
    # bottom of layer
    layers['a_bottom'] = layers.a.shift(1, fill_value=0)
    # width of layer
    layers['y'] = layers.a - layers.a_bottom
    # e = rectangle to right in int S computation
    layers['e'] = layers.S * layers.y
    # f = rectangle below attachment in int xf computation
    layers['f'] = layers.p * layers.a_bottom
    # these are two versions of m (unconditional)
    # m-bit: int S - e == int xf - f
    layers['m'] = layers.l - layers.e
    # int f dx in layer
    layers['v'] = layers.f + layers.m
    # and conditional vertical loss in layer
    layers['v_c'] = layers.v / layers.p
    layers = layers[['a_bottom', 'a', 'y', 'lev', 'S', 'p', 'l', 'v', 'v_c', 'm', 'e', 'f']]

    # add weights w and offsets=omega, computed from the top layer down
    layers['t'] = target
    layers['w'] = 0.0
    layers['ω'] = 0.0

    # this computation leaves the tail unchanged and uses the same "adjust the curve" method
    # in all layers
    ω = layers.loc[len(layers), 'S']
    for i in layers.index[::-1]:
        layers.loc[i, 'w'] = (layers.loc[i, 't'] - ω * layers.loc[i, 'y']) / layers.loc[i, 'm']
        layers.loc[i, 'ω'] = ω
        ω += layers.loc[i, 'p'] * layers.loc[i, 'w']

    # adjusted S: bins -> layer number; add in offsets
    density['bin'] = pd.cut(density.x, np.hstack((0, layers.a.values)), include_lowest=True, right=True)
    # layer description returned by cut to layer number in layers
    mapper = {i:j+1 for j, i in enumerate(density.bin.unique())}
    density['layer'] = density.bin.map(mapper.get)

    density['ω'] = density.layer.map(layers.ω).astype(float)
    # S(a_n-1)
    density['Sa'] = density.layer.map(layers.S).astype(float)
    density['w'] = density.layer.map(layers.w).astype(float)

    density['S_adj'] = np.minimum(1, density.ω + (density.S - density.Sa) * density.w)
    # no change in the tail
    density.loc[attachments[-1]:, 'S_adj'] = density.loc[attachments[-1]:, 'S']
    # adj probs as difference of S
    density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
    achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
    # display(achieved)
    if abs(achieved.iloc[0] - target[0]) > 1e-3:
        # issues with hitting 1
        logger.log(WL, f'achieved[0] = {achieved.iloc[0]} != target[0] = {target[0]}')
        # take top right corner off
        if target[0] > attachments[0]:
            raise ValueError(f'target[0] = {target[0]} > first attachment[0] = {attachments[0]} which is impossible.')
        s0 = 2 * (attachments[0] - target[0]) / (1 - layers.loc[1, 'ω'])
        # snap to index
        s0 = bs * np.round(s0 / bs, 0)
        # convert to probability
        s = attachments[0] - s0
        density.loc[0:s, 'S_adj'] = 1.0
        temp = np.array(density.loc[s+bs:attachments[0]].index)
        wts = (temp - s) / s0
        density.loc[s+bs:attachments[0], 'S_adj'] = 1 - wts + layers.loc[1, 'ω'] * wts
        # update
        density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
        achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
        logger.log(WL, f'Revised layer 1 achieved = {achieved.iloc[0]}')

    density['diff S'] = density['S'] - density['Sa']

    if debug is False:
        return density['p_adj'].values

    # data frame of layer statistics from input density
    exact = None
    if sf is not None:
        logger.log(WL, 'sf passed in; computing exact layer statistics')
        exact = pd.DataFrame(columns=['a', 'lev', 'aS', 'S'],
                             index=range(1, 1+len(attachments)), dtype=float)
        for i, x in enumerate(attachments):
            ix = quad(sf, 0, x)
            # check error is small
            assert ix[1] < 1e-6
            sf_ = sf(x)
            exact.loc[i+1, :] = [x, ix[0], x * sf_ if x < np.inf else 0.0, sf_]

    if exact is None:
        l = layers.l
        ln = 'layers'
    else:
        l = exact.lev - exact.lev.shift(1, fill_value=0)
        ln = 'exact'

    t = pd.concat((l,
                   density.groupby(density.layer.shift(-1)).apply(lambda g: g['S'].sum() * bs),
                   achieved,
                   ), keys=[ln, 'computed', 'adj'], axis=1)
    t.loc['sum'] = t.sum()
    Picks = namedtuple('picks', ['layers', 'exact', 'density', 'audit'])
    return Picks(layers=layers, exact=exact, density=density, audit=t)


def integral_by_doubling(func, x0, err=1e-8):
    r"""
    Compute :math:`\int_{x_0}^\infty f` as the sum

    .. math::

        \int_{x_0}^\infty f = \sum_{n \ge 0} \int_{2^nx_0}^{2^{n+1}x_0} f

    Caller should check the integral actually converges.

    :param func: function to be integrated.
    :param x0: starting x value
    :param err: desired accuracy: stop when incremental integral is <= err.
    """
    ans = 0.
    counter = 0
    # from to
    f, t = x0, 2 * x0
    last_int = 10
    while last_int > err:
        s = quad(func, f, t)
        if s[1] > err:
            raise ValueError(
                f'Questionable integral numeric convergence, err {s[1]:.4g}\n'
                f'f={f}, t={t}, x0={x0}, counter={counter}')
        last_int = s[0]
        ans += s[0]
        f, t = t, 2 * t
        counter += 1
        if counter > 96:
            raise ValueError(f'counter = {counter} and error = {err}')
    return -ans

@lru_cache(maxsize=128)
def logarithmic_theta(mean):
    """
    Solve for theta parameter given mean, see JKK p. 288
    """
    f = lambda x: x / (-np.log(1 - x) * (1 - x)) - mean
    theta = brentq(f, 1e-10, 1-1e-10)
    if not np.allclose(mean, theta / (-np.log(1 - theta) * (1 - theta))):
        print('num method failed')
    else:
        return theta


def make_var_tvar(ser):
    """
    Make var (lower quantile), upper quantile, and tvar functions from a ``pd.Series`` ``ser``, which
    has index given by losses and p_total values.

    ``ser`` must have a unique monotonic increasing index and all p_totals > 0.

    Such a series comes from ``a.density_df.query('p_total > 0').p_total``, for example.

    Tested using numpy vs pd.Series lookup functions, and this version is much
    faster. See ``var_tvar_test_suite`` function below for testers (obviously
    run before this code was integrated).

    Changed in v. 0.13.0

    """

    # audits
    assert ser.index.is_unique, 'index values must be unique'
    assert ser.index.is_monotonic_increasing, 'index values must be increasing'

    # detach from the outside scope
    ser = ser.copy()

    # create needed arrays
    x_np = np.array(ser.index)
    # better not to cumulate array when all elements are equal (because of
    # floating point issues). This does make some difference. 
    if np.all(np.isclose(ser, ser.iloc[0], atol=2**-53)):
        d = 1 / len(ser)
        cser = pd.Series(np.linspace(d, 1, len(ser)), index=ser.index)
    else:
        cser = ser.cumsum()
    cser_F_np = cser.to_numpy()
    # detach the index values
    # cser_idx = pd.Index(cser.values)
    tvar_unconditional = ((ser * ser.index)[::-1].cumsum()[::-1]).to_numpy()

    # these last three are annoyting because np.where does not short circuit
    tvar_unconditional = np.hstack((tvar_unconditional, np.inf, np.inf))
    cser_F_np2 = np.hstack((cser_F_np, 1))
    x_np2l = np.hstack((x_np, x_np[-1]))
    x_np2u = np.hstack((x_np, np.inf))
    # x_max = cser_F_np[-2]

    # tests show this is about 6 times faster than
    # q = interp1d(cser, ser.index, kind='next', bounds_error=False, fill_value=(ser.index.min(), ser.index.max()))
    def q_lower(p):
        nonlocal x_np2l, cser_F_np
        return x_np2l[np.searchsorted(cser_F_np, p, side='left')]

    def q_upper(p):
        nonlocal x_np2u, cser_F_np
        return x_np2u[np.searchsorted(cser_F_np, p, side='right')]

    def tvar(p):
        """
        Vectorized TVaR computation.
        """
        nonlocal cser_F_np, x_np, tvar_unconditional
        if isinstance(p, (float, int)):
            # easy
            if p >= cser_F_np[-2]:
                return x_np[-1]
            else:
                idx = np.searchsorted(cser_F_np, p, side='right')
                return ((cser_F_np[idx] - p) * x_np[idx] + tvar_unconditional[idx + 1]) / (1 - p)
        else:
            # vectorized
            p = np.array(p)
            idx = np.searchsorted(cser_F_np, p, side='right')
            return np.where(idx >= len(cser_F_np) - 1,
                            x_np[-1],
                           ((cser_F_np2[idx] - p) * x_np2u[idx] + tvar_unconditional[idx + 1]) / (1 - p))

    QuantileFunctions = namedtuple("QuantileFUnctions", 'q q_lower var q_upper tvar')
    return QuantileFunctions(q_lower, q_lower, q_lower, q_upper, tvar)


def test_var_tvar(program, bs=0, n_ps=1025, normalize=False, speed_test=False, log2=16):
    """
    Run a test suite of programs against new var and tvar functions compared to
    old aggregate.Aggregate versions

    Suggestion::

        args = [
            ('agg T dfreq [1,2,4,8] dsev [1]', 0, 1025, False),
            ('agg T dfreq [1:8] dsev [2:3]', 0, 1025, False),
            ('agg D dfreq [1:6] dsev [1]', 0, 7, False),
            ('agg T2 10 claims 100 x 0 sev lognorm 10 cv 1 poisson ', 1/64, 1025, False),
            ('agg T2 10 claims sev lognorm 10 cv 4 poisson ', 1/16, 1025, False),
            ('agg T2 10 claims sev lognorm 10 cv 4 mixed gamma 1.2 ', 1/8, 1025, False),
            ('agg Dice dfreq [1] dsev [1:6]', 0, 7, False)
        ]

        from IPython.display import display, Markdown
        for t in args:
            display(Markdown(f'## {t[0]}'))
            bs = t[1]
            test_suite(*t, speed_test=False, log2=16 if bs!= 1/8 else 20)

    Expected output to show that new functions are 3+ orders of magnitude faster, and
    agree with the old functions. q and q_upper agree everywhere except the jumps.

    """

    from aggregate import build, qd

    a = build(program, bs=bs, normalize=normalize, log2=log2)
    qd(a)
    ser = a.density_df.query('p_total > 0').p_total

    print(f'\ntotal probability = {ser.sum():.16f}')
    print()

    qf = make_var_tvar(ser)
    ps = np.linspace(0, 1, n_ps)

    if speed_test:
        pass
        # print('Timeit Tests\n============\n')
        # print('new var')
        # %timeit qf.var(ps)
        # print('agg.var')
        # %timeit [a.q(i) for i in ps]
        # %timeit qf.q_upper(ps)
        # print('new tvar')
        # %timeit qf.tvar(ps)
        # print('agg.tvar')
        # %timeit [a.tvar(i, kind='tail') for i in ps]
        # print()

    # report results
    test_df = pd.DataFrame({
        'q': qf.var(ps),
        'a.q': [a.q(i) for i in ps],
        'q_upper': qf.q_upper(ps),
        'a.q upper': [a.q(i, kind='upper') for i in ps],
        'tvar': qf.tvar(ps),
        'a.tvar tail': [a.tvar(i, kind='tail') for i in ps]
            }, index=ps)

    for tq in ['q != `a.q`', '`a.q upper` != q_upper', 'q != q_upper', 'abs(tvar - `a.tvar tail`) > 1e-12']:
        df = test_df.query(tq)
        display(df.head(10).style.format(precision=5).set_caption(f'{tq} with {len(df)} rows'))

    display(test_df.sample(min(25, len(test_df))).sort_index().style.format(precision=5).set_caption('Whole Dataframe Sample of 25 Values'))

    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    test_df.plot(lw=1, ax=ax, drawstyle='steps-pre')
    for l, ls in zip(ax.lines, ['-', '--', ':', '-.', '-', ':']):
        l.set_linestyle(ls)
    ax.legend()
    ax.set(title=program)


def kaplan_meier(df, loss='loss', closed='closed'):
    """
    Compute Kaplan Meier Product limit estimator based on a sample
    of losses in the dataframe df. For each loss you know the current
    evaluation in column ``loss`` and a 0/1 indicator for open/closed
    in ``closed``.

    The output dataframe has columns

    * index x_i, size of loss
    * open - the number of open events of size x_i (open claim with this size)
    * closed - the number closed at size x_i
    * events - total number of events of size x_i
    * n - number at risk at x_i
    * s - probability of suriviving past x_i = 1 - closed / n
    * pl - cumulative probability of surviving past x_i

    See ipython workbook kaplan_meier.ipynb for a check against lifelines
    and some kaggle data (telco customer churn,
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
    https://towardsdatascience.com/introduction-to-survival-analysis-the-kaplan-meier-estimator-94ec5812a97a

    :param df: dataframe of data
    :param loss: column containing loss amount data
    :param closed: column indicating if the obervation is a closed claim (1) or open (0)
    :return: dataframe as described above
    """

    df = df[[loss, closed]].rename(columns={loss: 'loss', closed: 'closed'}).copy()
    df['open'] = 1 - df.closed
    df = df.sort_values(['loss', 'closed'], ascending=[False, True]).reset_index(drop=True)

    df = df.groupby(['loss', 'closed']).count()
    # c has index loss amount and closed indicator, and column number of observations
    c = df.unstack(1)
    # total number of observables at each loss event size
    c['t'] = c.sum(1)
    # total number at risk at each event size
    c['n'] = c.t[::-1].cumsum()
    # better column names
    c.columns = ['open', 'closed', 'events', 'n']
    #
    c = c.fillna(0)
    # prob of surviving past each observed event size
    c['s'] = 1 -  c.closed / c.n
    # KM product estimator
    c['pl'] = c.s.cumprod()
    return c


def kaplan_meier_np(loss, closed):
    """
    Feeder to kaplan_meier where loss is np array  of loss amounts and
    closed a same sized array of 0=open, 1=closed indicators.
    """
    df = pd.DataFrame({'loss': loss, 'closed': closed})
    return kaplan_meier(df)


def more(self, regex):
    """
    Investigate self for matches to the regex. If callable, try calling with no args, else display.

    """
    for i in dir(self):
        if re.search(regex, i):
            ob = getattr(self, i)
            if not callable(ob):
                display(Markdown(f'### Attribute: {i}\n'))
                display(ob)
            else:
                display(Markdown(f'### Callable: {i}\n'))
                try:
                    print(ob())
                except Exception as e:
                    help(ob)


def parse_note(txt):
    """
    Extract kwargs from txt note. Recognizes bs, log2, padding, normalize, recommend_p.
    CSS format.
    Split on ; and then look for k=v pairs
    bs can be entered as 1/32 etc.

    :param txt: input text
    :return value: dictionary of keyword: typed value
    """

    stxt = txt.split(';')
    ans = {}
    for s in stxt:
        kw = s.split('=')
        if len(kw) == 2:
            k = kw[0].strip()
            v = kw[1].strip()
            if re.match('bs|recommend_p', k):
                if re.match(r'(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?/(\d+\.?\d*|\d*\.\d+)([eE](\+|\-)?\d+)?', v):
                    v = eval(v)
                else:
                    v = float(v)
            elif re.match('log2|padding', k):
                v = int(v)
            elif 'normalize':
                v = v == 'True'
            ans[k] = v
    return ans


def parse_note_ex(txt, log2, bs, recommend_p, kwargs):
    """
    Avoid duplication: this is how the function is used in Underwriter.build.

    """
    kw = parse_note(txt)
    if 'log2' in kw and log2 == 0:
        log2 = kw.pop('log2')
    if 'bs' in kw and bs == 0:
        bs = kw.pop('bs')
    if 'recommend_p' in kw:
        # always take the recommend_p from the note
        recommend_p = kw.pop('recommend_p')
    # rest are passed through
    kwargs.update(kw)
    return log2, bs, recommend_p, kwargs


def introspect(ob):
    """
    Discover the non-private methods and properties of an object ob. Returns
    a pandas DataFrame.
    """
    d = [i for i in dir(ob) if i[0] != '_']
    df = pd.DataFrame({'name': d})
    ans = []
    for i in d:
        g = getattr(ob, i)
        c = callable(g)
        v = ''
        t = ''
        h = ''
        l = 0
        if not c:
            v = str(g)
            t = type(g)
            h = ''
            if isinstance(getattr(ob.__class__, i, None), property):
                c = 'property'
            else:
                c = 'field'
            try:
                l = len(g)
            except:
                pass
        else:
            c = 'method'
            h = g.__doc__
        ans.append([c, v, t, h, l])

    df[['callable', 'value', 'type', 'help', 'length']] = ans
    df = df.sort_values(['callable', 'length', 'name'])
    return df


def explain_validation(rv):
    """
    Explain the validation result rv.
    Don't over report: if you fail CV don't need to be told you fail Skew too.
    """
    if rv == Validation.NOT_UNREASONABLE:
        return "not unreasonable"
    elif rv & Validation.NOT_UPDATED:
        return "n/a, not updated"
    elif rv & Validation.REINSURANCE:
        return "n/a, reinsurance"
    else:
        explanation = 'fails '
        if rv & Validation.SEV_MEAN:
            # explanation += f'sev mean: {ob.sev_m: .4e} vs {ob.est_sev_m: .4e}\n'
            explanation += f'sev mean, '
        if rv & Validation.AGG_MEAN:
            explanation += f'agg mean, '
        if rv & Validation.ALIASING:
            explanation += "agg mean error >> sev, possible aliasing; try larger bs, "
        if not(rv & Validation.SEV_MEAN) and (rv & Validation.SEV_CV):
            # explanation += f'sev cv: {ob.sev_cv: .4e} vs {ob.est_sev_cv: .4e}, '
            explanation += f'sev cv, '
        if not(rv & Validation.AGG_MEAN) and (rv & Validation.AGG_CV):
            explanation += f'agg cv, '
        if not (rv & Validation.SEV_CV) and (rv & Validation.SEV_SKEW):
            # explanation += f'sev skew: {ob.sev_skew: .4e} vs {ob.est_sev_skew: .4e}, '
            explanation += f'sev skew, '
        if not (rv & Validation.AGG_CV) and (rv & Validation.AGG_SKEW):
            explanation += f'agg skew, '
    return explanation[:-2]

