# standalone figures from PIR book

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import ticker

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