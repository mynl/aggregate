import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.fft import irfft,  rfft, ifft as ift
from numpy import real, imag, roll
from .. import build, qd, Aggregate
from .. constants import FIG_H, FIG_W


# get the logger
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


def ft_invert(log2, chf, frz_generator, params, loc=0, scale=1, xmax=0, xshift=0,
              suptitle='', wraps=None, disc_calc='density'):
    """
    Illustrate "manual" inversion of a characteristic function using irfft, including
    optional scaling and location shift.

    :param params: a list of shape parameters. loc and scale are handled separately.
    :param chf: the characteristic function of the distribution, takes args params, loc, scale, t;
      routine handles conversion to Fourier Transform
    :param frz_generator: the scipy.stats function to create the underlying distribution.
      Used to compute the exact answer. If there is not analytic formula, you can pass in
      a numpy array with the values of the distribution at the points in xs computed by
      some other means (e.g., for the Tweedie distribution).
    :param loc: location paramteter
    :param scale: scale parameter
    :param xmax: if not zero, used to fix xmax. Otherwise, selected as frz.isf(1e-17) to capture the
      full range of theunderlying distribution. For thick tailed distributions, you usually input
      xmax manually. Note bs = xmax / n. If None set equal to n = 1 << log2, for use in
      discrete distributions (forced bucket size of 1).
    :param xshift: if not zero, used to shift the x-axis. The minimum x value equals xshift.
      To center the x-axis, set xshift = -xmax/2. Should be a multiple of bs = xmax / n.
    :param suptitle: optional suptitle for the figure
    :param wraps: optional list of wrap values.
    :param disc_calc: 'density' rescales pdf, 'surival' uses backward differences of sf (to
      match Aggregate class calculation)
    """

    # number of buckets
    n = 1 << log2
    if xmax is None:
        xmax = n

    # make frozen object
    if callable(frz_generator):
        if scale is None:
            # freq dists (Poisson) do not allow scaling
            frz = frz_generator(*params, loc=loc)
            frz.pdf = frz.pmf
            # for subsequent use
            scale = 1
        else:
            frz = frz_generator(*params, loc=loc, scale=scale)

    # spatial upto xmax; used to create exact using the scipy stats object
    # sampling interval (wavelength) = xmax / n
    if xmax == 0:
        xmax = frz.isf(1e-17)
    # sampling domain, for exact and to "label" the Fourier Transform output
    # xs = np.arange(n) * xmax / n
    bs = xmax / n
    xs = np.arange(n) * bs + xshift
    if callable(frz_generator):
        if disc_calc == 'density':
            exact = frz.pdf(xs)
            exact = exact / exact.sum() * (frz.cdf(xs[-1]) - frz.cdf(xs[0]))
        else:
            xs1 = np.hstack((xs - bs / 2, xs[-1] + bs / 2))
            exact = -np.diff(frz.sf(xs1))
    else:
        # pass in values
        exact = frz_generator

    # convert chf to ft including scale and loc effects
    def loc_ft(t):
        nonlocal params, loc, scale
        # ft(t)= int f(x)exp(-2πi t x)dx
        # chf(t) = int f(x)exp(i t x)dx
        t1 = -t * 2 * np.pi
        ans = chf(*params, t1 * scale)
        if loc != 0:
            # for some reason ans *= np.exp(-t1 * loc) does not work
            ans = ans * np.exp(t1 * loc * 1j)
        return ans

    # sampling interval = bs = xmax / n [small bs, high sampling rate]
    # sampling freq is 1 / bs = n / xmax, the highest sampling freq for inverting the FT
    # note xmax = n * bs, so n / xmax = 1 / bs.
    # f(x) = int_R fhat(t) exp(2πi tx)dt ≈ int_-f_max_f^max_f ...
    f_max = n / xmax
    # sample the FT; using real fft, only need half the range
    ts = np.arange(n // 2 + 1) * f_max / n   # ts = np.arange(n // 2 + 1) / xmax
    fx = loc_ft(ts)
    # for debugging
    ft_invert.fx = fx
    ft_invert.ts = ts
    x = irfft(fx)
    if xshift != 0:
        x = np.roll(x, -int(xshift / bs))

    # plotting
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axs.flat

    for ax in axs[0].flat:
        ax.plot(xs, exact, label='exact', lw=1.5)
        ax.plot(xs, x, label='xs irfft', ls=':', lw=1.5)
        ax.legend(fontsize='x-small')
    ax0.set(title='Density', xlabel='Outcome, x')
    # mn = min(np.log10(exact).min(), np.log10(x).min())
    mn0 = np.log10(x).min() * 1.25
    mn = 10 ** np.floor(mn0)
    mx = max(np.log10(exact).max(), np.log10(x).max())
    mx = 10 ** np.ceil(mx)
    if np.isnan(mn):
        mn = 1e-17
    if np.isnan(mx):
        mx = 1
    ax1.set(yscale='log', ylim=[mn, mx], title='Log density', xlabel='Outcome, x')

    # amplitude and phase
    ax2.plot(ts, np.abs(fx), '-', lw=1.5, c='C3')
    ax2.set(title='Amplitude',
            ylabel='|ft|', yscale='log', xlabel='frequency')
    if log2 <= 8:
        ax3.plot(ts, np.cumsum(np.angle(fx)) / (2 * np.pi), '-', marker='.', ms=3, c='C2')
    else:
        ax3.plot(ts, np.cumsum(np.angle(fx)) / (2 * np.pi), c='C2')
    ax3.set(title='Cumulative phase',
            ylabel='cumsum(arg(ft)) / $2\\pi$', xlabel='frequency')

    if suptitle != '':
        fig.suptitle(suptitle)

    if wraps is not None:
        fig2, ax = plt.subplots(1, 1, figsize=(3.5 * 2, 2.45 * 2), constrained_layout=True)
        rt = exact.copy()
        ax.plot(xs, exact, label='exact', lw=1.5)
        ax.plot(xs, x, label='xs irfft', ls=':', lw=1.5)
        for b in wraps:
            xs2 = b * n / f_max + xs
            adj = frz.pdf(xs2)
            adj = adj / np.sum(adj) * (frz.cdf((b + 1) * n / f_max) - frz.cdf(b * n / f_max))
            rt += adj
            ax.plot(xs, rt, label=f'wrap {b}', lw=.5)
        ax.set(yscale='log', ylim=[mn, mx], title='Aliasing analysis',
               xlabel='Outcome, x', ylabel='Log density')
        ax.legend(fontsize='x-small', ncol=2, loc='upper right')

    return pd.DataFrame({'x': xs, 'p': x, 'p_exact': exact}).set_index('x')


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
        ag.update(approximation='exact', log2=agg_log2, bs=bs, padding=0)
        qd(ag)
        print('-'*80)
        # percentiles - help determining log2 needed for hi freq calculation
        qd(pd.Series([ag.agg_sd, ag.q(0.001), ag.q(0.999),
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
    fa = ag.freq_pgf(en, fz)
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
        # remove fuzz
        eps = np.finfo(float).eps
        df.loc[df.a.abs() < 2 * eps, 'a'] = 0
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


class FourierTools(object):
    """Manual inversion of a ch. f. using FFTs."""

    def __init__(self, chf, fz, scale_mode=True):
        """
        Class version of manual inversion of characteristic function.

        Splits functionality of aggregate.extensions.ft_invert into:

        1. Numerical inversion, ``ft_invert``
        2. Computation of actual density
        2. Graph to compare densities
        3. Graph to compute effect of wrapping

        Compared to ``ft_invert`` function, the meaning of x_max and
        xshift are changed.

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
          (Note: ft_invert allowed the class, pre-creation or an array. That option no
          longer allowed.) If fz is 'discrete' or 'continuous' or 'mixed' it is a
          generic distribution with no closed form cdf/pdf, e.g. Tweeedie. Then you
          can't compute exact, obviously.
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
        # for debugging
        # ft_invert.fx = fx
        # ft_invert.self._ts = self._ts
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
        assert self._df is not None, 'Must recompute first. Run ft_invert().'
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
        assert self._df is not None, 'Must recompute first. Run ft_invert() and compute_exact().'
        has_exact = self._df_exact is not None
        if not has_exact:
            logger.warning('No exact! Maybe run compute_exact().')

        # plot four graphs per ft_invert
        if verbose:
            self.last_fig, axs = plt.subplots(2, 3, figsize=(3 * 2.5, 2 * 2), constrained_layout=True)
            ax0, ax1, ax2, ax3, ax4, ax5 = axs.flat
        else:
            self.last_fig, axs = plt.subplots(1, 2, figsize=(2 * 2.5, 1 * 2.5), constrained_layout=True)
            ax0, ax1 = axs.flat

        x = np.array(self._df.index)
        p = self._df.p.values
        b = x[1] - x[0]
        if has_exact:
            xe = np.array(self._df_exact.index)
            pe = self._df_exact.p.values
            be = xe[1] - xe[0]
        else:
            # avoid an error below when no exact
            pe = self.df.p.values
        if xlim is None:
            if x[0] == 0:
                lower = -x[-1] / 25
            else:
                lower = x[0]
            xlim = [lower, x[-1]]

        for ax in [ax0, ax1]:
            ax.plot(x, p / b, label='Fourier', lw=1)
            if has_exact:
                ax.plot(xe, pe, ls='--', c='C3', label='exact', lw=1)
            ax.legend(fontsize='x-small')
        ax0.set(xlim=xlim, title='Density', xlabel='Outcome, x')
        # mn = min(np.log10(exact).min(), np.log10(x).min())
        mn0 = np.log10(p / b).min() * 1.25
        mn = 10 ** np.floor(mn0)
        mx = max(np.log10(pe).max(), np.log10(x).max())
        mx = 10 ** np.ceil(mx)
        if np.isnan(mn):
            mn = 1e-17
        if np.isnan(mx):
            mx = 1
        ax1.set(yscale='log', ylim=[mn, mx], xlim=xlim, title='Log density', xlabel='Outcome, x')
        if not verbose:
            if suptitle != '':
                self.last_fig.suptitle(suptitle)
            return

        # else: verbose mode: full monty with six plots
        if self.discrete and len(self._df) <= 64:
            drawstyle = 'steps-post'
        else:
            drawstyle = 'default'
        for ax in [ax2, ax5]:
            ax.plot(x, np.cumsum(p), label='cdf Fourier', lw=1,
                    c='C0', drawstyle=drawstyle)
            if has_exact:
                ax.plot(xe, np.cumsum(pe * be), label='cdf exact', ls='--', lw=.5,
                    c='C3', drawstyle=drawstyle)
            ax.plot(x, np.cumsum(p[::-1])[::-1], label='sf Fourier', lw=1,
                    c='C2', drawstyle=drawstyle)
            if has_exact:
                ax.plot(xe, np.cumsum(pe[::-1] * be)[::-1], label='sf exact', ls='--', lw=.5,
                    c='C4', drawstyle=drawstyle)

        ax2.set(title='sf and cdf', xlabel='Outcome, x', xlim=xlim)
        ax2.legend()
        ax5.set(yscale='log', title='log sf and cdf', ylim=[mn, 10],
                xlim=xlim, xlabel='Outcome, x')

        # ft on ax3, probably should inline this
        self._plot_fourier1d(ax3)

        # amplitude and phase both on ax4
        ax4r = ax4.twinx()
        ax4.plot(self._ts, np.abs(self._fourier), '-', lw=1.5, c='C4', label='Amplidude')
        # for amplitude, only look at nonzero fts, find index of non-zero (inz) items
        inz = np.abs(self._fourier) > 0
        tnz = self._ts[inz]
        fnz = self._fourier[inz]
        anz = np.angle(fnz)
        uw = np.unwrap(anz)
        if len(self._df) <= 256:
            kw = {'ls': '-', 'marker': '.', 'ms': 3, 'lw': 1}
        else:
            kw = {'lw': 1}
        ax4r.plot(tnz, uw, c='C2', **kw, label='phase, unwrapped')
        kw['lw'] = 0.5
        kw['marker'] = None
        kw['ls'] = ':'
        ax4r.plot(tnz, anz, c='C2', **kw, label='phase, wrapped')
        ax4r.legend(loc='upper right')
        ax4.legend(loc='center right')
        ax4.set(ylabel='Amplitude |ft|', yscale='log', xlabel='frequency', xlim=[-.05, 0.5 / self.bs + 0.05])
        if self.bs == 1:
            ax4.set(xticks=[0, .25, .5])
        ax4r.set(title='Amplitude and phase',
                 ylabel='Phase / 2π', xlabel='frequency / 2π')
        if suptitle != '':
            self.last_fig.suptitle(suptitle)

    def plot_wraps(self, wraps=None, calc='survival', add_tail=False):
        """
        Illustrate wrapping. Only run when fz.pdf is easy to calc, otherwise too slow.

        :param wraps: optional list of wrap values. Eg [-1,1] plots one greater and one
            less than [0, P)
        :param calc: how to estimate the density outside the base range, same ``compute_exact``.
        :param add_tail: plot the shifted exact densities in plots 2 and 4.
        """
        assert self._df is not None, 'Must recompute first. Run invert().'
        assert self._df_exact is not None, "Must run compute_exact() first."
        # extract values
        x = np.array(self._df.index)
        b = x[1] - x[0]
        p = self._df['p'].values / b    # here and below divide by b to convert to a density
        xe = np.array(self._df_exact.index)
        be = xe[1] - xe[0]
        pe = self._df_exact.p.values
        x_range = self.x_max - self.x_min
        rt = pe.copy()

        # duplicated...
        mn0 = np.log10(x).min() * 1.25
        mn = 10 ** np.floor(mn0)
        mx = max(np.log10(p).max(), np.log10(x).max())
        mx = 10 ** np.ceil(mx)
        if np.isnan(mn):
            mn = 1e-17
        if np.isnan(mx):
            mx = 1

        # report answer
        ans = []
        self.last_fig, axs = plt.subplots(2, 2, figsize=(2 * 2.5, 2 * 2.), constrained_layout=True)
        ax0, ax1, ax2, ax3 = axs.flat
        for ax in axs.flat:
            ax.plot(x, p, label='Fourier', lw=2)
        lw = .5
        if calc == 'density':
            if hasattr(self.fz, 'pdf'):
                pdf = self.fz.pdf
            elif hasattr(self.fz, 'pmf'):
                pdf = self.fz.pmf
            else:
                raise ValueError('fz must have pdf or pmf method for density method')
        for i, b_wrap in enumerate(wraps):
            if type(b_wrap) == int:
                bl = f'{b_wrap:d}'
            else:
                bl = f'{b_wrap:.3f}'
            xs2 = b_wrap * x_range + xe
            if calc == 'survival':
                # for computing probs using survival method
                xs2d = np.hstack((xs2 - be / 2, xs2[-1] + be / 2))
                adj = -np.diff(self.fz.sf(xs2d)) / be
                rt += adj
            elif calc == 'density':
                adj = pdf(xs2)
                rt += adj
            else:
                raise ValueError('calc must be "survival" or "density"')
            c = f'C{i+1}'
            ax0.plot(xe, rt, label=bl, lw=lw, c=c)
            ax1.plot(xe, adj, label=bl, lw=lw, c=c)
            if add_tail:
                ax1.plot(xs2, adj, label=bl, lw=lw, c=c, ls=':')
                ax3.plot(xs2, adj, label=bl, lw=lw, c=c, ls=':')
            ax2.plot(xe, rt, label=bl, lw=lw, c=c)
            ax3.plot(xe, adj, label=bl, lw=lw, c=c)
            ans.append([b_wrap, xs2[0], xs2[-1], self.fz.cdf(xs2[-1]), self.fz.cdf(xs2[0]),
                        self.fz.cdf(xs2[-1]) - self.fz.cdf(xs2[0])])
        for ax in axs.flat:
            ax.plot(xe, pe, label='exact', ls='--', lw=1, c='C3')
        ax0.set(yscale='linear',
                title='Cumulative aliasing',
                xlabel='Outcome, x',
                ylabel='Density')
        ax2.set(yscale='log',
                title='Cumulative - log scale',
                xlabel='Outcome, x',
                ylabel='Log density')
        ax1.set(yscale='linear',
                title='Incremental aliasing',
                xlabel='Outcome, x',
                ylabel='Density')
        ax3.set(yscale='log',
                title='Incremental - log scale',
                xlabel='Outcome, x',
                ylabel='Log density')
        ax2.legend(fontsize='x-small', ncol=2)
        if add_tail:
            ax1.legend(fontsize='x-small', ncol=2)
        if add_tail:
            if 0 not in wraps:
                wraps.append(0)
            if 1 not in wraps:
                wraps.append(1)
            wraps.append(max(wraps) + 1)
            wraps = sorted(wraps)
            for b_wrap in wraps:
                ax1.axvline(x[0] + b_wrap * x_range, lw=.25, c='k', ls=':')
                ax3.axvline(x[0] + b_wrap * x_range, lw=.25, c='k', ls=':')
            for ax in [ax1, ax3]:
              ax.set(xticks=self.x_min + np.arange(0, max(wraps), 2) * self.x_max)
              ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        df = pd.DataFrame(ans, columns=['Wrap', 'x0', 'x1', 'Pr(X≤x1)', 'Pr(X≤x0)', 'Pr(X in Wrap)'])
        df = df.set_index('Wrap')
        return df

    def plot_simpson(self, ylim=1e-16):
        """Plot Simpson's approximation."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.25), constrained_layout=True)
        if self._df_exact is not None:
            self.df_exact.p.plot(ax=ax, lw=1, c='C1')
        xs = np.array(self.df.index)
        if 'p' in self.df:
            ax.plot(xs, self.df.p / self.bs, label='basic', c='C0', lw=1)
        ax.plot(xs, self.df.simpson / self.bs, label='simpson', c='C3', ls=':')
        ax.set(yscale='log', ylim=ylim)
        ax.legend()

    def _plot_fourier1d(self, ax, min_abs=1e-20):
        """Create simple plot of Fourier transform on one axis."""
        fhat = self._fourier.copy()
        c = np.abs(fhat)
        num_large = np.sum(c > min_abs)
        fhat = fhat[:num_large]
        c = c[:num_large]

        ax.plot(np.real(fhat), np.imag(fhat), '-o', ms=2, lw=.5, label='ft')
        fhat = fhat[c > 0] / c[c > 0]
        ax.plot(np.real(fhat), np.imag(fhat), '-o', ms=1, lw=.5, label='ft / |ft|')
        lim = [-1.05, 1.05]
        ax.set(xlim=lim, ylim=lim, aspect='equal', xlabel='real(ft)',
               ylabel='imag(ft)', title='Fourier transform')
        ax.legend(loc='upper left')
        ax.axhline(0, c='k', alpha=.5, lw=.5)
        ax.axvline(0, c='k', alpha=.5, lw=.5)

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
    assert 0 < alpha < 2, f'alpha must be in (0, 2]'
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
