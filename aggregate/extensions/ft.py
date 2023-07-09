import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import irfft, fft, ifft, rfft, fftshift
from numpy import real, imag, roll
from .. import build
from .. constants import FIG_H, FIG_W
import matplotlib as mpl
from aggregate import build, qd

def fft_wrapping_illustration(ez=10, en=20, sev_clause='', small2=0):
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
    fig, axs = plt.subplots(1, 3, figsize=(3 * FIG_W, FIG_H), constrained_layout=True)
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
    cmappable = mpl.cm.ScalarMappable(norm=norm, cmap='plasma')
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
      xmax manually. Note bs / xmax / n. If None set equal to n = 1 << log2, for use in
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
            ylabel='cumsum(arg(ft)) / $2\pi$', xlabel='frequency')

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
    fa = ag.mgf(en, fz)
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