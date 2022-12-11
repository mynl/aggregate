# standalone figures from PIR book

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import ticker
from .. import build

def fig_4_1():
    """
    Figure 4.1: illustrating quantiles.

    """

    fz = ss.lognorm(.5)
    xs = np.linspace(0,5,501)[1:]
    xsx = np.linspace(0,5,501)[1:]
    xsx[89:149] = xsx[89]
    F = fz.cdf(xsx)

    fig, ax = plt.subplots(1, 1, figsize=(6,4), constrained_layout=True, squeeze=True)

    lt = F < .6
    gt = F > .6

    for f in [lt, gt]:
       if f is lt:
           ax.plot(xs[f], F[f], lw=2, label='Distribution, $F$')
       else:
           ax.plot(xs[f], F[f], lw=2, label=None)

    ax.plot([0,5], [0.6, 0.6], ls='--', c='k', lw=1, label='$p=0.6$')

    p = fz.cdf(xs[89])
    ax.plot([0,5], [p, p], ls='--', lw=1, c='C2', label=f'$p={p:.3f}$')
    ax.set(xlabel='$x$', ylabel='$F(x)$')
    ax.axvline(1.50, lw=0.5)

    xx = 0.75
    pp = fz.cdf(xx)
    ax.plot([0, xx], [pp, pp], ls='-', lw=.5, c='k', label=f'$p={pp:.3f}$')
    ax.plot([xx, xx], [0, pp], ls='-', lw=.5, c='k', label=None)
    ax.legend(loc='lower right')

    p1 = fz.cdf(xs[149])
    x = 1.5
    s = .1
    ax.plot(x,p, 'ok', ms=5, fillstyle='none')
    ax.plot(x,p1, 'ok', ms=5)
    ax.text(x+s, p+s/4, f'$Pr(X<1.5)={p:.3f}$')
    ax.text(x+s, p1-s/4, f'$Pr(X ≤ 1.5)={fz.cdf(1.5):.3f}$')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5));

    return fig


# non-PIR book figures ============================================
def adjusting_layer_losses():
    """
    Figure to illustrate the process of adjusting layer losses.
    TODO: Add reference

    """
    f, ax = plt.subplots(1,1, figsize=(2.45, 3.5), constrained_layout=True)
    fz = ss.lognorm(.4)
    xs = np.linspace(0,fz.isf(1e-3),1001, endpoint=False)
    F = fz.cdf(xs)
    ax.plot(F, xs)
    a1 = 1.5
    a0 = 1
    ax.axhline(a1 , c='k', lw=.25)
    ax.axhline(a0, c='k', lw=.25)
    p0 = fz.cdf(a0)
    p1 = fz.cdf(a1)
    ax.plot([p0, p0], [0, a0] , c='k', lw=.25)
    ax.plot([p1, p1], [0, a1] , c='k', lw=.25)
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
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45), constrained_layout=True)
    axi = iter(axs.flat)

    for lbl, df in zip(['Poisson distribution', 'Mixed frequency distribution'], [dfp, dfnb]):
        ax0 = next(axi)
        ax1 = next(axi)
        for c in df:
            ax0.plot(df.index/c, c*df[c],       lw=1, label=str(c))
            ax1.plot(df.index/c, df[c].cumsum(),lw=1, label=str(c))

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

