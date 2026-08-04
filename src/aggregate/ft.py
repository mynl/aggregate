"""Direct numerical inversion of characteristic functions.

Provides :class:`FourierTools`, an object-oriented FFT-based inverter for
characteristic functions (used internally by :class:`aggregate.tweedie.Tweedie`
for distributions without closed-form pdfs), plus :func:`make_levy_chf` and
several paper-figure helpers (``poisson_example``, ``fft_wrapping_illustration``,
``recentering_convolution[_example]``).

``FourierTools`` is reached as ``from aggregate.ft import FourierTools``;
nothing here is re-exported at the top-level package namespace.
"""

import logging

import numpy as np
import pandas as pd
from scipy.fft import irfft, rfft, ifft as ift
from numpy import roll

from .constants import FIG_H, FIG_W
from .distributions import Aggregate
from .underwriter import build
from .utilities import qd, remove_fuzz as remove_fuzz_util

logger = logging.getLogger(__name__)


def poisson_example(en, small2):
    """
    Example to show how to compute Po(en) using 1 << small2 buckets.
    For AAS paper. Sample call::

        poisson_example(10**8, 17)

    :param en: mean of Poisson, e.g., 10**8
    :param small2: log2 number of buckets to use in FFT routine, 2**small2 should be
      about 10 * en ** 0.5 to get +/-5 standard deviations around the mean

    """
    from scipy.stats import poisson
    from .plots import plt, ticker

    B = 1 << small2
    z = np.zeros(B); z[1] = 1
    wrap = irfft(np.exp(en * (rfft(z) - 1)))
    k = en // B + 1
    xs = k * B - (B >> 1) +  np.arange(B)
    pmf = roll(wrap, B >> 1)
    df = pd.DataFrame({'x': xs, 'FFT pmf': pmf})
    po = poisson(en)
    df['Exact pmf'] = po.pmf(df.x)
    df = df.set_index('x', drop=True)
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H + 0.3), constrained_layout=True)
    ax0.plot(wrap);
    ax0.set(title=f'Raw FFT-based output, wrapped to [0, {B}]',
            xlabel=f'Wrapped outcome, n mod {B}',
            ylabel='Probability mass, Pr(N=n)');
    df[['FFT pmf', 'Exact pmf']].plot(style=['-', ':'], ax=ax1, logy=True,
        title='Shifted FFT vs exact Poisson probabilities\n(log scale)',
        xlabel='Outcome, n', ylabel='Probability mass, Pr(N=n)');
    ax1.set(ylim=[1e-17, 2 * df['FFT pmf'].max()])
    ax1.yaxis.set_minor_locator(ticker.LogLocator(subs='all'))


def fft_wrapping_illustration(ez=10, en=20, sev_clause='', small2=0, cmap='plasma'):
    """
    Illustrate wrapping by convolving a uniform distribution with mean ez
    en times (if ``ez>0`` or ``sev_clause!=''``) or using the input ``sev_clause``.
    ``sev_clause`` should be a ``dsev`` tailored to ``bs==1`` or just a Poisson
    if ez==1.

    Show in a space just big enough for the severity first and then
    big enough for the full aggregate. Center and right hand plot illustrate how the
    full components are sliced up and combined to the wrapped total.

    If small2 is zero it is taken to be the smallest value to "fit" the severity.

    In Poisson ez==1 mode, small2 equals the size of the small window to use for
    convolution. big2 is estimated to fit the whole distribution.

    (moved from figures.py)

    """
    from .plots import plt, mpl

    fig, axs = plt.subplots(1, 3, figsize=(3 * FIG_W, FIG_H + 0.3), constrained_layout=True)
    ax0, ax1, ax2 = axs.flat

    if ez == 0 or sev_clause != '':
        if sev_clause == '':
            raise ValueError('Must input one of ez>0 or a valid DecL sev_clause')
        sev = build(f'agg Junk 1 claim {sev_clause} fixed')
        ez = sev.sev_m
        q1 = sev.q(1)
        if small2 == 0:
            small2 = int(np.ceil(np.log2(q1)))
        xs = np.hstack((-np.inf, np.arange(1 << small2) + 0.5))
        z = np.diff(sev.sev.cdf(xs))
        # enough space for aggregate
        big2 = int(np.ceil(np.log2(q1 * en)))
    elif ez == 1:
        assert small2, 'Need to input small2 in Poisson mode'
        z = np.zeros(1 << small2)
        z[1] = 1
        # full space
        sigma = np.sqrt(en)
        big2 = int(np.ceil(np.log2(en + 5 * sigma)))
    else:
        # enough space for severity and make sev
        if small2 == 0:
            small2 = int(np.ceil(np.log2(2 * ez)))
        z = np.zeros(1 << small2)
        z[:ez*2] = 1 / ez / 2
        # enough space for aggregate
        big2 = int(np.ceil(np.log2(2 * ez * en)))

    if big2 <= 8:
        ds = 'steps-post'
    else:
        ds = 'default'
    if ez == 1:
        wrapped = irfft( np.exp(en * (rfft(z) - 1)))
        full = irfft( np.exp(en * (rfft(z, 1 << big2) - 1)))
    else:
        wrapped = irfft( rfft(z) ** en )
        full = irfft( rfft(z, 1 << big2) ** en )

    ax0.plot(wrapped, c='C0', drawstyle=ds)
    ax0.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    ax0.set(title=f'Wrapped distribution\nlog2={small2}')
    lm = ax0.get_ylim()
    lm = (-lm[1] / 20, lm[1]* 1.1)
    ax0.set(ylim=lm)

    norm = mpl.colors.Normalize(0, 1, clip=True)
    cmappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper = cmappable.to_rgba
    cc = list(map(mapper, np.linspace(0, 1, 1 << big2-small2)))
    ax1.plot(full, label='Full computation', c='w', alpha=1, lw=3, drawstyle=ds)
    ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(32))
    for n, (s, c) in enumerate(zip(full.reshape((1<<big2-small2, 1<<small2)), cc)):
        ax1.plot(s, c=c, label=f'Part {n}', drawstyle=ds)
        ax1.plot(np.arange((1<<small2) * n, (1<<small2) * (n+1)), s, c=c, lw=2,
                 drawstyle=ds, label=None)
    for n in range(1 << big2-small2):
        ax1.axvline(n * (1 << small2), lw=.25, c='C7')
    for n in range(1 << big2-small2):
        ax1.axvline(n * (1 << small2), lw=.25, c='C7')
    if big2 - small2 <= 3:
        ax1.legend(loc='center right')
    ax1.set(title=f'Full distribution\nlog2={big2}, {1<<big2-small2} components')

    wrapped_from_full = full.reshape((1<<big2-small2, 1<<small2))
    ax2.plot(wrapped_from_full.T, label=None, c='C7', lw=.5, drawstyle=ds)
    ax2.plot(wrapped_from_full.sum(0), lw=3
             , drawstyle=ds, label='Wrapped from full', c='C1')
    ax2.plot(wrapped, lw=1, label='Wrapped', c='C0', drawstyle=ds)
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    ax2.set(title='Wrapping components (grey)\nSums (blue, organge as left)')
    # ax2.legend(loc)
    ax2.set(ylim=lm)

    assert np.allclose(wrapped_from_full.sum(0), wrapped)


def recentering_convolution_example(sev_clause, en, log2, agg_log2=0, bs=1,
                                    freq_clause='poisson', remove_fuzz=False):
    """
    Illustrate how to find the "correct" part of an aggregate and
    recenter it appropriately. Aggregate is::

        agg RecenteringExample en claims dsev xs ps poisson

    **Method**

    #. Compute aggregate with enough space for the supported part of the
       distribution (say, where density > 1e-15 or so)
    #. Subtract the mean by rolling left (negative shift) by mean / bs buckets (mod n)
    #. fft shift = roll (in either direction) by n / 2 buckets, because
       the distribution is centered at zero and has positive and negative parts.
    #. Set appropriate x values to align with the density. Density is from
       mean - n/2 to mean + n/2 - 1 (times bs).

    Reasonable defaults::

        en = 5000
        log2 = 15

        xs = [3, 4, 7, 34]
        ps = [1/8, 1/8, 1/8, 5/8]

    (from hifreq.py)

    :param xs: array of x values for dsev
    :param ps: array of x values for dsev

    """
    from .plots import plt

    df, ag = recentering_convolution(sev_clause, freq_clause, en, log2, bs, remove_fuzz)

    # update the ag object if agg_log2
    if agg_log2 == 0:
        # just plot
        fig, ax0 = plt.subplots(1, 1, figsize=(
            3.5, 2.45), constrained_layout=True)

        df.plot(ax=ax0, c='C0')
        ax0.set(
            title=f'Shifted, centered, re-indexed aggregate\nlog2={log2} buckets')
        bit = None

    elif agg_log2 > 0:
        ag.update(log2=agg_log2, bs=bs, padding=0)
        qd(ag)
        print('-'*80)
        # percentiles - help determining log2 needed for hi freq calculation
        qd(pd.Series([ag.actual_sd, ag.q(0.001), ag.q(0.999),
                      ag.q(0.999999) - ag.q(0.000001)],
                     index=['std dev', 'p001', 'p999', 'range']))
        print('-'*80)

        # merge and compare
        fig, axs = plt.subplots(2, 3, figsize=(
            3 * 3.5, 2 * 2.45), constrained_layout=True)
        ax0, ax1, ax2, ax3, ax4, ax5 = axs.flat

        df.plot(ax=ax0, c='C0')
        ax0.set(
            title=f'Shifted, centered, re-indexed aggregate\nlog2={log2} buckets')
        bit = pd.concat((ag.density_df.p_total, df), axis=1, join='inner')
        bit.plot(ax=ax1)
        ax1.lines[1].set(ls='--')
        ax1.legend(loc='upper right')
        ax1.set(
            title=f'Shifted vs. agg object\n'
                  f'Linear scale; agg object log2 = {ag.log2}')
        bit.plot(ax=ax2, logy=True)
        ax2.lines[1].set(ls='--')
        ax2.legend(loc='upper right')
        ax2.set(title='Shifted vs. agg object\nLog scale')

        abs_error = (bit.p_total - bit.a).abs()
        abs_error.plot(ax=ax4, c='C2', logy=True,
                       title=f'Abs error\n'
                       f'Max {abs_error.max():.5g}; '
                       f'Avg {abs_error.mean():.5g}')
        rel_error = abs_error / bit.p_total
        rel_error = rel_error.loc[bit.p_total > 1e-15]
        rel_error.plot(ax=ax5, c='C2', logy=True,
                       title=f'Rel error p_total > 1e-15\n'
                       f'Max {rel_error.max():.5g}; '
                       f'Avg {rel_error.mean():.5g}')
        ax3.remove()

    else:
        raise ValueError('log2 >= 0!')

    return df, ag, bit


def recentering_convolution(sev_clause, freq_clause, en, log2, bs, remove_fuzz):
    """
    Compute hifreq convol for sev_clause, a DecL severity statement.
    Must have only one component for now.
    Illustrates how to find the "correct" part of an aggregate and
    recenter it appropriately. Aggregate is::

        agg RecenteringExample en claims dsev xs ps poisson

    **Method**

    #. Compute aggregate with enough space for the supported part of the
       distribution (say, where density > 1e-15 or so)
    #. Subtract the mean by rolling left (negative shift) by mean / bs buckets (mod n)
    #. fft shift = roll (in either direction) by n / 2 buckets, because
       the distribution is centered at zero and has positive and negative parts.
    #. Set appropriate x values to align with the density. Density is from
       mean - n/2 to mean + n/2 - 1 (times bs).

    """

    ag = build(f'agg SevEg {en} claims {sev_clause} {freq_clause}', update=False)
    ez = ag.sev_m
    mean_bucket = int(round(ez * en / bs))

    ft_len = 1 << log2
    ag.xs = np.linspace(0, ft_len * bs, ft_len, endpoint=False)
    ag.log2 = log2
    ag.bs = bs
    ag.padding = 0
    z = ag.discretize('round', 'survival', True)
    assert len(z) == 1
    z = z[0]
    fz = rfft(z, ft_len)
    # aggregate by hand
    # fa = np.exp(en*(fz - 1))
    fa = ag.frequency.freq_pgf(en, fz)
    a = irfft(fa)

    # center roll left by ez and 1 << log2-1, former
    # is the mean offset, the latter is fftshift
    a = roll(a, -(mean_bucket % ft_len) + (1 << log2 - 1))

    # set up aligned to the appropriate xs values
    df = pd.DataFrame(
        {'n': np.arange(mean_bucket - (1 << log2 - 1),
                        mean_bucket + (1 << log2 - 1)) * bs,
         'a': a}
    ).set_index('n')
    if remove_fuzz:
        # remove fuzz; this site keeps a looser 2*eps tolerance (preserved via
        # the explicit eps= argument to the shared utility).
        df = remove_fuzz_util(df, eps=2 * np.finfo(float).eps)
    qd(stats(df).T)
    print('-'*80)
    return df, ag


def stats(df):
    ans = {}
    ns = {}
    ns[1] = np.array(df.index)
    ns[2] = ns[1] * ns[1]
    ns[3] = ns[2] * ns[1]
    for c in df:
        ans[c] = [np.sum(ns[i] * df[c]) for i in [1, 2, 3]]
    df = pd.DataFrame(ans, index=[1, 2, 3])
    df.loc['var'] = df.loc[2] - df.loc[1] ** 2
    df.loc['sd'] = df.loc['var'] ** .5
    df.loc['cv'] = df.loc['sd'] / df.loc[1]
    df.loc['skew'] = (df.loc[3] - 3 * df.loc[2] * df.loc[1] +
                      2 * df.loc[1]**3) / df.loc['sd'] ** 3
    return df


class FourierTools:
    """Manual inversion of a ch. f. using FFTs."""

    def __init__(self, chf, fz, scale_mode=True):
        """
        Class version of manual inversion of characteristic function.

        Numerical inversion is split across four methods:

        1. ``invert`` — the FFT inversion itself.
        2. ``compute_exact`` — exact density via ``fz`` (when available).
        3. ``plot`` — graph to compare densities.
        4. ``plot_wraps`` — graph to compute effect of wrapping (aliasing).

        For discrete rvs, x_max is always n - 1 and the bucketr size 1.
        For continuous rvs, it is either input or estimated as a quantile.

        The arguments completely define the distribution of interest. Other class
        functions vary the numerical variables, defining the window and number of
        points used in the FFT routine.

        See BLOG POST.

        :param chf: the characteristic function of the distribution, takes args t;
          routine handles conversion to Fourier Transform and adds loc and scale effects.
          Must use same shape parameters as fz.
        :param fz: the scipy.stats frozen distribution object. Used to compute the exact answer.
          If fz is 'discrete' or 'continuous' or 'mixed' it is a generic distribution
          with no closed form cdf/pdf, e.g. Tweedie. Then you can't compute exact,
          obviously.
        :param scale_mode: if True, the scale parameter from fz is used in the Fourier
          Transform, otherwise it is unadjusted.
        """
        self.chf = chf
        self.fz = fz
        self.scale_mode = scale_mode
        if isinstance(fz, str):
            # extremely limited functionality
            self.params = (0,)  # allows passing as param 1 to the chf
            self.loc = 0.
            self.scale = 1.
            self.discrete = True if fz == 'discrete' else False
            self.distribution_name = 'User defined'
        elif isinstance(fz, Aggregate):
            self.params = (0,)
            self.loc = 0.
            self.scale = 1.
            self.discrete = fz.bs == 1
            self.distribution_name = f'Aggregate({fz.name})'
        elif fz is None:
            # flying blind...do the best we can
            self.params = (0,)
            self.loc = 0.
            self.scale = 1.
            self.discrete = False
            self.distribution_name = 'Unknown'
        else:
            # extract shape params from fz; no longer used
            self.params = fz.args
            # location and scale with default 0, 1
            kwds = fz.kwds
            self.loc = kwds.get('loc', 0.)
            self.scale = kwds.get('scale', 1)
            # is this a discrete or continuous variable?
            self.discrete = True if str(type(fz)).find('discrete') > 0 else False
            self.distribution_name = fz.dist.name
        # slightly ugly state, but this encodes n=2**log2, xmin, xmax, etc.
        self._df = self._df_exact = None
        self._ts = None      # phases (angles) at which ft evaluated , for plotting
        self._fourier = None  # store the sample of ft at self._ts for plotting
        self.last_fig = None
        # from the last run, note: x_range = P
        self.bs = self.x_range = self.x_min = self.x_max = self.log2 = 0
        self.exact_calc = ""

    def __repr__(self):
        """Repr of the object."""
        return f'FourierTools({self.distribution_name}{self.params}, loc={self.loc}, scale={self.scale})'

    def describe(self):
        """More information."""
        return f'{repr(self)}\nn={2**self.log2}, x_min={self.x_min}, x_max={self.x_max:.3g}, bs={self.bs:.3g}'

    @property
    def df(self):
        """Return current state dataframe output."""
        if self._df is not None:
            return self._df
        else:
            raise ValueError('Must run invert first!')

    @property
    def df_exact(self):
        """Return current state dataframe output."""
        if self._df is not None:
            return self._df_exact
        else:
            raise ValueError('Must run compute_exact first!')

    def fourier_transform(self, t):
        """
        Create ft function  by converting ch f to Fourier transform and including scale and loc.

        Recall:
        ft(t)= int f(x)exp(-2πi t x)dx
        chf(t) = int f(x)exp(i t x)dx.
        """
        TWOPI = 6.283185307179586
        t1 = -t * TWOPI
        if self.scale_mode:
            ans = self.chf(t1 * self.scale)
            if self.loc != 0:
                # for some reason ans *= np.exp(-t1 * loc) does not work
                ans = ans * np.exp(t1 * self.loc * 1j)
        else:
            ans = self.chf(t1)
        return ans

    def invert(self, log2, x_min=0, bs=0, x_max=None, s=1e-17):
        """
        Invert a characteristic function using irfft.

        Call with just log2 for positive support, to determine x_max and
        bs from quantiles. Call with bs fixed to a reasonable value to
        deduce x_max = x_min + n bs. Call with x_max to fix the range
        and deduce bs.

        :param x_min: minimum value of range.
        :param bs: bucket size to use, determines x_max
        :param x_max: minimum value of range.
        :param s: survival probability to determine tails
        """
        # number of buckets
        self.log2 = log2
        n = 1 << log2
        if x_min is None:
            x_min = self.fz.ppf(s)
        if bs == 0:
            if self.discrete:
                x_max = x_min + n
            else:
                x_max = self.fz.isf(s) if x_max is None else x_max
        else:
            x_max = x_min + bs * n
        if x_min == 0 and x_max == 0:
            raise ValueError('Must provide x_min < 0 or x_max > 0. Current range is 0 to 0!')

        # spatial range is [x_min, x_max]
        # translate to [0, x_max - x_min]
        x_range = x_max - x_min
        # sampling interval (wavelength) = bs = x_range / n
        # sampling domain, for exact and to "label" the Fourier Transform output
        # sampling interval is bs (small bs means high sampling rate)
        # the highest sampling freq for inverting the FT is 1 / bs = n / x_range
        bs = x_range / n
        if self.discrete and bs != 1:
            logger.warning(f'{bs=}, not the expected 1 for a discrete rv')

        # f_max is 1 / bs
        f_max = 1 / bs
        # f(x) = int_R fhat(t) exp(2πi tx)dt ≈ int_-f_max_f^max_f ...
        # sample the FT; using real fft, only need half the range
        # self._ts = np.arange(n // 2 + 1) * f_max / n
        self._ts = np.linspace(0, 1/2, n // 2 + 1) * f_max
        self._fourier = self.fourier_transform(self._ts)
        probs = irfft(self._fourier)
        if x_min != 0:
            probs = np.roll(probs, -int(x_min / bs))

        # for df index
        self._df = pd.DataFrame({
            'x': np.linspace(x_min, x_max, n, endpoint=False),
            'p': probs}).set_index('x')

        # store for future use
        self.bs, self.x_range, self.x_min, self.x_max = bs, x_range, x_min, x_max

    def invert_simpson(self, *, log2=0, bs=0, x_min=None):
        """
        Add Simpson's method approximation.

        Adds ``simpson`` column to
        ``self.df``. Uses method of Wang and Zhang, "Simpson's rule based FFT
        method to compute densities of stable distribution" (2008). Can no
        longer use the real fft and ifft methods because the input vector is
        not conjugate symmetric about its midpoint.

        Run after self.invert to set parameters or input directly. Defaults
        are None because 0 is a legitimate value for x_min. Convenient to
        state x_min and bs directly. Best to input all parameters if adjusting
        to avoid mismatches. Note x_max gets trumped by n, bs, and x_min.
        """
        # parameters to use for the calculation
        log2 = log2 if log2 > 0 else self.log2
        n = 1 << log2
        x_min = x_min if x_min is not None else self.x_min
        bs = bs if bs > 0 else self.bs
        f_max = 1 / bs
        ks = np.arange(0, n)
        P = n * bs   # period

        t_left = np.linspace(-f_max / 2, f_max / 2, n, endpoint=False)
        t_left = np.roll(t_left, n >> 1)

        if x_min != 0:
            # pre-FFT adjustment
            x_min_adj = np.exp(2 * np.pi * 1j * x_min * t_left)
            # post-FFT adjustment
            post_2 = np.exp(2 * np.pi * 1j * x_min / (2 * P))
            post_3 = np.exp(2 * np.pi * 1j * x_min / P)
        else:
            # no adjustments needed
            x_min_adj = post_2 = post_3 = 1.

        phi_left = self.fourier_transform(t_left)
        # three terms left, middle, right of simpson's rule
        term1 = ift(x_min_adj * phi_left)
        term2 = post_2 * np.exp(np.pi * 1j * ks / n) * ift(
            x_min_adj * self.fourier_transform(t_left + 0.5 / n * f_max))
        term3 = post_3 * np.exp(2 * np.pi * 1j * ks / n) * ift(
            x_min_adj * self.fourier_transform(t_left + 1 / n * f_max))
        # weighted sum
        simpson = (term1 + 4. * term2 + term3) / 6.
        # check imaginary part small
        max_abs = np.abs(np.imag(simpson)).max()
        if max_abs > 1e-10:
            logger.warning(f'Answer has suspiciously large imaginary component {max_abs}')
        # create / update answer dataframe
        if self._df is None or self.bs != bs or self.x_min != x_min:
            # changed scale or location, recreate dataframe from scratch
            logger.info('recreating df')
            self._df = pd.DataFrame({
                'x': np.linspace(x_min, x_min + n * bs, n, endpoint=False),
                'simpson': np.real(simpson)}).set_index('x')
        else:
            # append to existing dataframe
            self.df['simpson'] = np.real(simpson)

    def compute_exact(self, calc='survival', max_points=257):
        """
        Compute exact density using frozen scipy.stats object.

        :param calc: 'density' re-scales pdf, 'survival' uses backward
          differences of sf.
        :param max_points: maximum number of points to use; if more just
          interpolate this many points.
        :param decimate: decimate (take ``::decimate``) input xs to reduce
          number of calls to fz (which may be slow)
        :param min_points: opposite of decimate, ensure exact computed
          with min_points. Useful for example when log2 is "too small"
          to ensure exact distribution is rendered correctly.
        """
        assert calc in ('survival', 'density'), 'calc must be "survival" or "density"'
        self.exact_calc = calc
        assert self._df is not None, 'Must recompute first. Run invert().'
        xs = np.array(self._df.index)
        if len(xs) > max_points and not self.discrete:
            # mostly this is an issue for non-discrete distributions
            xs = np.linspace(xs[0], xs[-1] + self.bs, max_points)
        # if decimate > 1:
        #     xs = xs[::decimate]
        # # if too few points beef up, non-discrete distributions only
        # if len(xs) < min_points and not self.discrete:
        #     xs = np.linspace(xs[0], xs[-1] + self.bs, min_points)
        self.bs_exact = bs = xs[1] - xs[0]
        # discrete dists have pmf not pdf
        if getattr(self.fz, 'pdf', None) is None:
            pdf = self.fz.pmf
        else:
            pdf = self.fz.pdf

        if calc == 'density':
            logger.warning('Best to use survival calc rather than density method.')
            exact = pdf(xs)
            self._df_exact = pd.DataFrame({'x': xs, 'p': exact}).set_index('x')
        elif calc == 'survival':
            xs1 = np.hstack((xs - bs / 2, xs[-1] + bs / 2))
            exact = -np.diff(self.fz.sf(xs1)) /  bs
            self._df_exact = pd.DataFrame({'x': xs, 'p': exact}).set_index('x')
        # self.decimate = decimate
        return self._df_exact

    def plot(self, suptitle='', xlim=None, verbose=True):
        """
        Compare density, log density, and plot amplitude and argument of Fourier transform.

        :param suptitle: super title for the plot.
        """
        from .plots import plot_fourier
        return plot_fourier(self, suptitle=suptitle, xlim=xlim, verbose=verbose)

    def plot_wraps(self, wraps=None, calc='survival', add_tail=False):
        """
        Illustrate wrapping. Only run when fz.pdf is easy to calc, otherwise too slow.

        :param wraps: optional list of wrap values. Eg [-1,1] plots one greater and one
            less than [0, P)
        :param calc: how to estimate the density outside the base range, same ``compute_exact``.
        :param add_tail: plot the shifted exact densities in plots 2 and 4.
        """
        from .plots import plot_fourier_wraps
        return plot_fourier_wraps(self, wraps=wraps, calc=calc, add_tail=add_tail)

    def plot_simpson(self, ylim=1e-16):
        """Plot Simpson's approximation."""
        from .plots import plot_fourier_simpson
        return plot_fourier_simpson(self, ylim=ylim)

    def _plot_fourier1d(self, ax, min_abs=1e-20):
        """Create simple plot of Fourier transform on one axis."""
        from .plots import plot_fourier1d
        return plot_fourier1d(self, ax, min_abs=min_abs)

    def plot_fourier3d(self, scale=True):
        """Three dimensional line plot of the Fourier transform using mayavi."""
        # dont want to make mayavi a required package
        logger.warning('REMEMBER: this routine pops a separate window!')
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('mayavi required, pip install mayavi')

        # Generate data
        t = self._ts
        f_t = self._fourier
        c = np.abs(f_t)  # Use |f(t)| for coloring

        if scale:
            f_t = f_t / c
        f_t[np.isinf(f_t)] = np.nan

        x = np.real(f_t)
        y = np.imag(f_t)
        # scale to 0, 1
        z = t / np.max(t)

        # Create the 3D line plot
        mlab.figure(size=(1000, 800))  # Set figure size
        mlab.plot3d(x, y, z, c, tube_radius=0.005, colormap='viridis')

        # Add axes and labels
        mlab.xlabel('Re(f)')
        mlab.ylabel('Im(f)')
        mlab.zlabel('scaled t')
        mlab.title('Scaled FT' if scale else "FT")
        # Show the plot
        mlab.show()

    def plot_fourier3da(self):
        """Three dimensional line plot of the Fourier transform using plotly."""
        # dont want to make plotly a required package
        try:
            import plotly.graph_objects as go
        except ModuleNotFoundError:
            raise ModuleNotFoundError('plotly required, pip install plotly')
        t = self._ts
        f_t = self._fourier
        x = np.real(f_t)
        y = np.imag(f_t)
        c = np.abs(f_t)

        # normalized plot data
        f_t_rhs = f_t / np.abs(f_t)
        x_rhs = np.real(f_t_rhs)
        y_rhs = np.imag(f_t_rhs)

        # Create the subplot figure
        fig = go.Figure()

        # LHS plot
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=t,
                mode="lines",
                line=dict(
                    color=.5,
                    width=4,
                ),
                name="blue: f(t)",
            )
        )

        # RHS plot
        fig.add_trace(
            go.Scatter3d(
                x=x_rhs,
                y=y_rhs,
                z=t,
                mode="lines",
                line=dict(
                    color=c,
                    colorscale="Plasma",
                    width=4,
                    colorbar=dict(
                        title="|f(t)|",
                        len=0.5,
                        lenmode="fraction",
                    ),
                ),
                name="colored: f(t) / |f(t)|",
            )
        )

        # Update layout for side-by-side 3D plots
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Re(f)", range=[-1, 1]),
                yaxis=dict(title="Im(f)", range=[-1, 1]),
                zaxis=dict(title="t", range=[0, .55]),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1)
            ),
            width=900,  # Wider figure for two plots
            height=800,
            title='Fourier transform and normalized transform.'
        )
        return fig


def make_levy_chf(alpha, beta):
    """Make the ch of stable(alpha, beta) per Nolan book page 5 Def 1.3."""
    assert 0 < alpha < 2, 'alpha must be in (0, 2]'
    if alpha == 1:
        def chf(t):
            return np.where(t==0, 1.,
                            np.exp(- np.abs(t) * (1 + 1j * beta * 2 / np.pi *
                                         np.sign(t) * np.log(np.abs(t)))))
    elif alpha < 2:
        # alpha not = 1
        tan = np.tan(np.pi / 2 * alpha)
        def chf(t):
            return np.exp(- np.abs(t) ** alpha * (1 - 1j * beta * np.sign(t) * tan))
    else:
        # actually normal
        def chf(t):
            return np.exp(-t**2 / 2)
    return chf
