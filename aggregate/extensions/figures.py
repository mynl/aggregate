# figures other than from PIR book

import numpy as np
# from numpy import roll
import scipy.stats as ss
from scipy.fft import rfft, irfft # , fftshift, ifftshift, fft, ifft
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
from .. import build, qd
from .. constants import FIG_H, FIG_W, PLOT_FACE_COLOR
from .. spectral import Distortion


def adjusting_layer_losses():
    """
    Figure to illustrate the process of adjusting layer losses.
    TODO: Add reference

    """
    f, ax = plt.subplots(1, 1, figsize=(FIG_H, FIG_W), constrained_layout=True)
    fz = ss.lognorm(.4)
    xs = np.linspace(0, fz.isf(1e-3), 1001, endpoint=False)
    F = fz.cdf(xs)
    ax.plot(F, xs)
    a1 = 1.5
    a0 = 1
    ax.axhline(a1, c='k', lw=.25)
    ax.axhline(a0, c='k', lw=.25)
    p0 = fz.cdf(a0)
    p1 = fz.cdf(a1)
    ax.plot([p0, p0], [0, a0], c='k', lw=.25)
    ax.plot([p1, p1], [0, a1], c='k', lw=.25)
    ax.set_xticks([0, p0, p1, 1])
    ax.set_xticklabels(['0', '$F(a_{i-1})$', '$F(a_{i})$', ''])
    ax.set_yticks([0, a0, a1, 3.5])
    ax.set_yticklabels(['0', '$a_{i-1}$', '$a_{i}$', ''])
    ax.text((p0+p1)/2, a0/2, '$f_i$', ha='center', va='center')
    ax.text((1+p1)/2, 1.25, '$e_i$', ha='center', va='center')
    xx = [.8]
    yy = [1.15]
    ll = ['$m_i$']
    ho = -0.2
    vo = 0.55
    ax.set(xlabel='Nonexceedance probability\n$F(x)$', ylabel='Outcome $x$',
           ylim=[0, 3.5], xlim=[-0.05, 1.05],
           title='Lee diagram\n$x$ vs $F(x)$')
    for x, y, l in zip(xx, yy, ll):
        ax.annotate(text=l,
                    xy=(x, y),
                    xytext=(x + ho, y + vo),
                    arrowprops={'arrowstyle': '->', 'linewidth': .5}
                    )
    return f


def savings_charge():
    """
    Figure to illustrate the insurance savings and expense(charge).

    """
    f, ax = plt.subplots(1, 1, figsize=(FIG_H, FIG_W), constrained_layout=True)
    fz = ss.lognorm(.4)
    xs = np.linspace(0, fz.isf(1e-3), 1001, endpoint=False)
    F = fz.cdf(xs)
    ax.plot(F, xs, lw=3)
    a1 = 1.25  # height of line
    ax.axhline(a1, c='C7', lw=1.5)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, a1, 3.5])
    ax.set_yticklabels(['0', '$r$', ''])
    ax.set(xlabel='Nonexceedance probability', ylabel='Scaled outcome',
           ylim=[0, 3.5], xlim=[0, 1],
           title='Insurance savings and expense')

    xx = [.9, .25]
    yy = [1.05 * a1, .8* a1]
    ll = ['Insurance\ncharge, $\\phi(r)$',
         'Insurance\nsavings, $\\psi(r)$']
    hos = [-0.1, 0.25]
    vos = [1.2, .65]
    ax.text(.5, a1 * .4, '$E[X\\wedge l]$', ha='center', va='center')
    for x, y, l, ho, vo in zip(xx, yy, ll, hos, vos):
        ax.annotate(text=l,
                    xy=(x, y),
                    xytext=(x + ho, y + vo),
                    ha='right',
                    va='bottom',
                    arrowprops={'arrowstyle': '->', 'linewidth': .5}
                    )
    return f


def mixing_convergence(freq_cv, sev_cv, bs=1/64):
    """
    Illustrate convergence of mixed distributions to the mixing distribution.

    """

    # make the two dataframes of distributions
    a = build('agg M '
              '1 claims '
              f'sev gamma 1 cv {sev_cv} '
              f'mixed gamma {freq_cv} ', log2=16, bs=bs, approximation='exact')
    dfnb = a.density_df[['p']].rename(columns={'p': 1})
    assert np.abs(a.est_m / a.agg_m - 1) < 1e-3
    # print("1", a.agg_m, a.est_m, a.est_m / a.agg_m - 1)

    for freq in [2, 5, 10, 20, 50, 100, 200]:
        a = build('agg M '
                  f'{freq} claims '
                  f'sev gamma 1 cv {sev_cv} '
                  f'mixed gamma {freq_cv} ', log2=16, bs=bs, approximation='exact')
        dfnb[freq] = a.density_df[['p']]
        assert np.abs(a.est_m / a.agg_m - 1) < 1e-3
        # print(freq, a.agg_m, a.est_m, a.est_m / a.agg_m - 1)

    a = build('agg M '
              '1 claims '
              f'sev gamma 1 cv {sev_cv} '
              'poisson ', log2=16, bs=bs, approximation='exact')
    dfp = a.density_df[['p']].rename(columns={'p': 1})
    assert np.abs(a.est_m / a.agg_m - 1) < 1e-3
    # print("1", a.agg_m, a.est_m, a.est_m / a.agg_m - 1)

    for freq in [2, 5, 10, 20, 50, 100, 200]:
        a = build('agg M '
                  f'{freq} claims '
                  f'sev gamma 1 cv {sev_cv} '
                  'poisson ', log2=16, bs=bs, approximation='exact')
        dfp[freq] = a.density_df[['p']]
        assert np.abs(a.est_m / a.agg_m - 1) < 1e-3
        # print(freq, a.agg_m, a.est_m, a.est_m / a.agg_m - 1)

    # plotting
    fig, axs = plt.subplots(2, 2, figsize=(
        2 * FIG_W, 2 * FIG_H), constrained_layout=True)
    axi = iter(axs.flat)

    for lbl, df in zip(['Poisson distribution', 'Mixed frequency distribution'], [dfp, dfnb]):
        ax0 = next(axi)
        ax1 = next(axi)
        for c in df:
            ax0.plot(df.index/c, c*df[c],       lw=1, label=str(c))
            ax1.plot(df.index/c, df[c].cumsum(), lw=1, label=str(c))

        if ax0 is axs.flat[0]:
            ax0.legend()
        ax0.set(ylim=[0, 1e-1], xlim=[-0.25, 5])
        ax1.set(ylim=[-0.05, 1.05], xlim=[-0.25, 5])

        ax0.set(ylabel='Density', xlabel='Normalized loss', title=lbl)
        ax1.set(ylabel='Distribution', xlabel='Normalized loss', title=lbl)

    # add convergence to mixing
    alpha = freq_cv**-2
    fz = ss.gamma(alpha, scale=1/alpha)
    ps = np.linspace(0, 5, 501)
    ax = axs.flat[-1]
    ax.plot(ps, fz.cdf(ps), lw=2, alpha=.5, c='k', label='Mixing')
    ax.legend(loc='lower right')


def power_variance_family():
    """
    Graph to illustrate the power variance exponential family distributions.

    Reference: Jørgensen, Bent. 1997. The theory of dispersion models. CRC Press.
    """
    alpha = np.linspace(-2, 2, 101)
    p = (alpha-2) / (alpha-1)
    alphabar = -(alpha+1)
    f, ax = plt.subplots(figsize=(FIG_W * 2, FIG_W * 2))
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


def dual_distortion(dist=None, s=0.3):
    """
    Illustrate how the dual distortion relates to the distortion.

    """

    def setbg(t):
        """ make text boxes opaque and same color as plot background """
        t.set_bbox(dict(facecolor=PLOT_FACE_COLOR, alpha=0.85, edgecolor='none', boxstyle='square,pad=.1'))

    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_W), constrained_layout=True)
    ax0, ax1 = axs.flat

    if dist is None:
        dist = Distortion('ph', 0.4)

    ps = np.linspace(0,1,1001)
    gp = dist.g(ps)
    dp = dist.g_dual(ps)

    ax0.plot(ps, gp, c='C1', label='Premium, $g(s)$')
    ax0.plot(ps, ps, c='C0', label='Loss cost, $s$')
    ax0.legend()
    ax0.axis([-0.025, 1.025, -0.025, 1.025])
    # ax0.set(xlim=[0,1], ylim=[0,1])


    ax1.plot(ps, gp, c='C1', label='$g$')
    ax1.plot(ps, ps, c='C0', label='$s$')
    ax1.plot(ps, dp, c='C2', label=r'Dual, $g\check$')

    ax1.plot([s, s], [dist.g(s), 1],            ls='--', c='C2', alpha=1, linewidth=2.5)
    ax1.plot([s, s], [0, s],                    ls='--', c='C0', alpha=1, linewidth=2.5)
    ax1.plot([s, s], [s, dist.g(s)],            ls='--', c='C1', alpha=1, linewidth=2.5)
    ax1.plot([1-s, 1-s], [0, dist.g_dual(1-s)], ls='--', c='C2', alpha=1, linewidth=2.5)
    t = ax1.text(s * 1.05, (1 + dist.g(s))/2, 'Capital\n$=1-g(s)$', ha='left', va='center')
    setbg(t)
    t = ax1.text(s * .95, (s + dist.g(s))/2, 'Margin\n$=g(s)-s$', ha='right', va='center')
    setbg(t)
    t = ax1.text(s * .95, s/2, 'Loss\n$=s$', ha='right', va='center')
    setbg(t)
    t = ax1.text((1-s) * 1.05, dist.g_dual(1 - s)/2, 'Bid price\n$=g\check(1-s)$', ha='left', va='center')
    setbg(t)

    x1, y1 = s, 1-s
    x2, y2 = 1-s, s

    ax1.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="<->", color="k",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle='bar,angle=90,fraction=-0.3333',
                                ),
                )

    ax1.legend()
    ax1.axis([-0.025, 1.025, -0.025, 1.025])
    # ax1.set(xlim=[0,1], ylim=[0,1])
    axs[0].set(title=None, xlabel='$s$, probability of loss to layer $1_{U<s}$',
               ylabel='Price of layer $1_{U<s}$', aspect='equal')
    axs[1].set(title=None, xlabel='$s$, probability of loss to layer $1_{U<s}$',
               aspect='equal')


def discretization_sev_example(outcomes):
    """
    For AAS paper. Convergence of sevs with smaller bucket size.

    """
    a01 = build(f'agg Num:01 1 claim dsev [{outcomes}] fixed', update=False)
    aex = build(f'agg Num:01e 1 claim dsev [{outcomes}] fixed', update=False)
    aex.update(log2=16, bs=1/2048)
    xlim = aex.limits()
    xlim = (xlim[0], np.round(xlim[1], 0))
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45 + 0.1),
        constrained_layout=True)
    for bs, ax in zip([1, 1/2, 1/4, 1/8], axs.flat):
        for k in ['forward', 'round', 'backward']:
            a01.update(log2=10, bs=bs, sev_calc=k)
            a01.density_df.p_total.cumsum().\
                plot(xlim=xlim, lw=2 if  k=='round' else 1,
                drawstyle='steps-post', ls='--', label=k, ax=ax)
        aex.density_df.p_total.cumsum().\
            plot(xlim=xlim, lw=1, label='exact', ax=ax)
        ax.legend(loc='lower right')
        ax.set(title=f'Bandwidth bs={bs}')
    axs[0,0].set(ylabel='distribution');
    axs[1,0].set(ylabel='distribution');
    # @savefig num_ex1a.png scale=20
    fig.suptitle('Severity by discretization method for different bandwidths');


def discretization_agg_example(outcomes):
    """
    For AAS paper. Convergence of sevs with smaller bucket size.

    """
    a02 = build(f'agg Num:02 4 claims dsev [{outcomes}] poisson', update=False)
    aex = build(f'agg Num:02e 4 claims dsev [{outcomes}] poisson', update=False)
    aex.update(log2=16, bs=1/2048)
    xlim = aex.limits()
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45 + 0.1),
        constrained_layout=True)
    for bs, ax in zip([1, 1/2, 1/4, 1/8], axs.flat):
        for k in ['forward', 'round', 'backward']:
            a02.update(log2=10, bs=bs, sev_calc=k)
            a02.density_df.p_total.cumsum().\
                plot(xlim=xlim, lw=2 if  k=='round' else 1,
               drawstyle='steps-post', label=k, ax=ax)
        aex.density_df.p_total.cumsum().\
            plot(xlim=xlim, lw=1, label='exact', ax=ax)
        ax.legend(loc='lower right')
        ax.set(title=f'Bandwidth bs={bs}')
    # @savefig num_ex1b.png scale=20
    fig.suptitle('Aggregates by discretization method');


def gh_example(en):
    '''
    Code to reproduce GHGrübel and Hermesmeier 1999, Table 1.
    The function ``exact_cdf`` calculates the compound probability
    that :math:`x-1/2 < X \le x+1/2`.
    For AAS paper with en=20.

    '''
    from scipy.stats import levy
    a = build(f'agg L {en} claim sev levy poisson', update=False)
    qd(a)
    bs = 1
    a.update(log2=16, bs=bs, padding=2, normalize=False, tilt_vector=None)
    df = a.density_df.loc[[1, 10, 100, 1000], ['p_total']] / a.bs
    df.columns = ['Agg pad=2']

    def exact_cdf(x):
        nonlocal en
        n = 5 * en
        # poisson freqs
        p = np.zeros(n)
        a = np.zeros(n)
        p[0] = np.exp(-en)
        fz = levy()
        for i in range(1, n):
            p[i] = p[i-1] * en / i
            a[i] = fz.cdf((x+0.5)/i**2) - fz.cdf((x-0.5)/i**2)
        return np.sum(p * a)

    df['True'] = [exact_cdf(i) for i in df.index]

    # other models
    log2 = 10
    for tilt in [None, 1/1024, 5/1024, 25/1024]:
        a.update(log2=log2, bs=bs, padding=0,
                 normalize=False, tilt_vector=tilt)
        if tilt is None:
            tilt = 0
        df[f'Tilt {tilt:.2g}'] = a.density_df.loc[[1, 10, 100, 1000],
                                  ['p_total']]/a.bs
    df.index = [f'{x: 6.0f}' for x in df.index]
    df.index.name = 'x'
    qd(df.iloc[:, [1,0,2,3,4, 5]], ff=lambda x: f'{x:11.3e}')