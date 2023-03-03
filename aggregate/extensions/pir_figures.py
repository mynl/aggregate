# standalone figures and tables from PIR book

from itertools import count
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import scipy.stats as ss
from .. import build
from .. import Distortion
from .. constants import FIG_W, FIG_H, PLOT_FACE_COLOR



def fig_4_1():
    """
    Figure 4.1: illustrating quantiles.

    """

    fz = ss.lognorm(.5)
    xs = np.linspace(0, 5, 501)[1:]
    xsx = np.linspace(0, 5, 501)[1:]
    xsx[89:149] = xsx[89]
    F = fz.cdf(xsx)

    fig, ax = plt.subplots(1, 1, figsize=(
        FIG_W, FIG_H), constrained_layout=True, squeeze=True)

    lt = F < .6
    gt = F > .6

    for f in [lt, gt]:
        if f is lt:
            ax.plot(xs[f], F[f], lw=2, label='Distribution, $F$')
        else:
            ax.plot(xs[f], F[f], lw=2, label=None)

    ax.plot([0, 5], [0.6, 0.6], ls='--', c='k', lw=1, label='$p=0.6$')

    p = fz.cdf(xs[89])
    ax.plot([0, 5], [p, p], ls='--', lw=1, c='C2', label=f'$p={p:.3f}$')
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
    ax.plot(x, p, 'ok', ms=5, fillstyle='none')
    ax.plot(x, p1, 'ok', ms=5)
    ax.text(x + s, p + s / 4, f'$Pr(X<1.5)={p:.3f}$')
    ax.text(x + s, p1 - s / 4, f'$Pr(X ≤ 1.5)={fz.cdf(1.5):.3f}$')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    return fig

def ex49():
    ps = np.ones(10) / 10
    cps = np.hstack((0,np.cumsum(ps)))
    xs = np.array([0,0,1,1,1,2,3, 4,8, 12, 25])
    df = pd.DataFrame({'x': xs[1:], 'p': ps})
    df = pd.DataFrame(df.groupby('x').p.sum())
    df['F'] = df.p.cumsum()
    df = df.reset_index(drop=False)
    return ps, cps, xs, df


def prob_format(axis):
    axis.set_major_formatter(ticker.FuncFormatter(
            lambda x, y: '0' if x==0
            else ('1' if x>=0.999
            else (f'{x:.2f}' if np.allclose(x,0.25) or np.allclose(x, 0.75)
            else f'{x:.1f}'))))


def fig_4_5():
    ps, cps, xs, df = ex49()
    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_W  + .2))
    ax0, ax1 = axs.flat
    ax = ax0
    ax.plot(xs, cps, drawstyle='steps-post')
    ax.plot(xs[1:], cps[1:], 'o')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.yaxis)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(xlim=[-.5, 25.5],
           ylim=[-.025, 1.025],
           title='Distribution function\nright continuous',
           aspect=(26/1.05)/(4.5/3.25)/1.15,
           ylabel='$F(x)$', xlabel='Outcome, $x$')

    ax = ax1
    ax.plot(cps, xs, drawstyle='steps-pre')
    ax.plot(cps[1:], xs[1:], 'o')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.5],
           xlim=[-.025, 1.025],
           title='Lower quantile VaR function\nleft continuous',
           aspect=(4.5/3.25)/(26/1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')

def fig_4_6():
    ps, cps, xs, df = ex49()
    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_W  + .2))
    ax0, ax1 = axs.flat
    ax = ax0
    ax.plot(df.x, df.F, c='C0')
    ax.plot([0,0], [0, df.F.iloc[0]], c='C0')
    ax.plot(df.x, df.F, 'o', c='C0')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.yaxis)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(xlim=[-.5, 25.5], ylim=[-.025, 1.025],
               title='Distribution function\n',
               aspect=(26/1.05)/(3.5/2.45),
               ylabel='$F(x)$', xlabel='Outcome, $x$')

    ax = ax1
    ax.plot(df.F, df.x , c='C0')
    ax.plot([0, df.F.iloc[0]], [0,0], c='C0')
    ax.plot(df.F, df.x, 'o', c='C0')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.5], xlim=[-.025, 1.025],
           title='Lower quantile VaR function\n',
           aspect=(3.5/2.45)/(26/1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')

def fig_4_8():
    ps, cps, xs, df = ex49()

    ad = build(f'agg Empirical 1 claim sev dhistogram xps {df.x.values} {df.p.values} fixed', bs=1)
    xv = np.hstack((1e-10, df.x.values))
    adc = build(f'agg Empirical 1 claim sev chistogram xps {xv} {df.p.values} fixed', bs=1/128)
    qps = np.linspace(0,1,1000, endpoint=True)
    tvar =np.array([ad.tvar(p) for p in qps])
    tvarx =np.array([ad.tvar(p, kind='tail') for p in qps])
    ctvar =np.array([adc.tvar(p) for p in qps])

    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_H, FIG_W  + .3), sharey=True)
    ax0,ax1 = axs.flat

    # discrete
    ax = ax0
    ad.density_df.loss = np.minimum(ad.density_df.loss, 25)

    ad.density_df.plot(y='loss', x='F', drawstyle='steps-pre', ylim=[-1,25.2], xlim=[-0.02,1.02], ax=ax, ls='--', label='Quantile')
    ax.plot(cps[:2], [0,0], ls='--', label='_none_')
    ax.plot(cps[1:], xs[1:], 'o', ms=5, c='C0', label='_none_')
    ax.plot(qps, tvar, c='C0', lw=1, label='TVaR')
    ax.plot(qps, tvarx, c='C3', lw=1, label='TVaR Ex')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.2],
           xlim=[-.025, 1.025],
           title='TVaR and lower quantile VaR,\ndiscrete sample',
           aspect=(4.5/3.25)/(26/1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')
    ax.legend() # .set(visible=False)

    # continuous
    ax = ax1
    adc.density_df.plot(y='loss', x='F', drawstyle='steps-pre', ylim=[-1,25.2], xlim=[-0.02,1.02], ax=ax, ls='--')
    ax.plot(df.F, df.x, 'o', ms=5)

    ax.plot(qps, ctvar, c='C0')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.2],
           xlim=[-.025, 1.025],
           title='TVaR and lower quantile VaR,\ncontinuous sample',
           aspect=(4.5/3.25)/(26/1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')
    ax.legend().set(visible=False)




def fig_10_3(dist=None, s=0.3):
    """
    Figure 10.3 Illustrating distortion functions
    (s, g(s)) with vertical line at s and split loss, premium, margin, and capital labelled
    """
    fig, axs = plt. subplots(1, 2, figsize=(2 * FIG_W, FIG_W), constrained_layout=True)
    if dist is None:
        dist = Distortion('ph', 0.4)

    g = dist.g
    N = 1000
    ps = np.linspace(0, 1, N, endpoint=False)
    gs = g(ps)
    sm = 0.085
    g_s = g(s)
    lbl = str(dist).replace('\n', ' ')

    def setbg(t):
        """ make text boxes opaque and same color as plot background """
        t.set_bbox(dict(facecolor=PLOT_FACE_COLOR, alpha=0.85, edgecolor='none', boxstyle='square,pad=.1'))

    for a in axs.flat:
        a.plot(ps, gs, c='C1', lw=1.5, label='Premium, $g(s)$')
        a.plot(ps, ps, linewidth=1.5, c='C0', alpha=1, label='Loss cost, $s$')
        a.axis([-0.025, 1.025, -0.025, 1.025])
        a.set(aspect='equal', xlabel='$s$', ylabel='$g(s)$',
              title=f'Insurance Statistics\n{lbl}')
        # a.grid(lw=0.25)
    axs[0].legend(loc='upper left')

    # a is the right hand plot
    a.plot([s, s], [0, s],   c='C0', ls='--', alpha=1, linewidth=2.5)
    a.plot([s, s], [s, g_s], c='C1', ls='--', alpha=1, linewidth=2.5)
    a.plot([s, s], [g_s, 1], c='C2', ls='--', alpha=1, linewidth=2.5)
    a.text(s + sm, s / 2, 'Loss $=s$', va='center')
    t = a.text(s + sm, (g_s + s) / 2, 'Margin\n$=g(s)-s$', va='center')
    setbg(t)

    if s > 0.3:
        a.text(s - sm, (1 + g_s) / 2, 'Capital =\n$1-g(s)$', ha='right', va='center')
    else:
        t = a.text(s + sm, (1 + g_s) / 2, 'Capital\n$=1-g(s)$', ha='left', va='center')
        setbg(t)

    delta = 0.02
    p3 = (s + delta, 0)
    p2 = (s + delta, s)
    p1 = (s + delta, dist.g(s))
    p0 = (s + delta, 1)

    p2m = (s + 1.5 * delta, s)
    p1m = (s + 1.5 * delta, dist.g(s))

    # capital
    curlyBrace(a, p0, p1, str_text=None,   int_line_num=2, k_r=0.055, c='k', lw=0.5)
    # margin
    curlyBrace(a, p1m, p2m, str_text=None, int_line_num=2, k_r=0.075, c='k', lw=0.5)
    # loss
    curlyBrace(a, p2, p3, str_text=None,   int_line_num=2, k_r=0.075, c='k', lw=0.5)
    # premium
    g_s = dist.g(s)
    curlyBrace(a, (.625, g_s), (.625, 0), str_text=None, int_line_num=2, k_r=0.0375, c='k', lw=0.5)
    a.text(.625 + sm, g_s / 2, 'Premium\n$=g(s)$', va='center', ha='left')
    # a.plot([0, s], [g_s, g_s], lw=1, c='k')
    a.plot([0, .626], [g_s, g_s], lw=.5, c='k', ls='-')

    for ax in axs.flat:
        ax.set(title=None, xlabel='$s$, probability of loss to layer $1_{U<s}$',
               ylabel='Price of layer $1_{U<s}$', aspect='equal')


def fig_10_5(port=None, dist=None, s=0.3):
    """
    three plot version of previous with more explanation of first picture

    return_period_max = defines extend of yaxis
    return_period_x = capital level to illustrate

    map from s space into loss space
    extended version of ch04_s_gs_loss_premium_capital which
    includes the horizontal bar [ loss ][m][  equity  ]
    plotted on the provided second axis

    Suggested figure set up for extended:

            f = plt.Figure(figsize=(4,3), tight_layout=True)
            a = f.add_axes([0, 100/3+1/27, 1, 2/3], label='a')
            b = f.add_axes([0, 0, 1, 1/3], label='b')

    """

    return_period_max = 100
    return_period_x = 1 / s

    fig, axs = plt.subplots(1, 3, figsize=(3 * FIG_H, FIG_W), constrained_layout=True)
    ax0, ax1, ax2 = axs.flat

    if port is None:
        port = build('port Test agg A 10 claims sev lognorm 50 cv 1 mixed gamma .5', bs=1/16)
    if dist is None:
        dist = build('distortion myph ph 0.4')

    g = dist.g
    K = port.q(1 - 1 / return_period_max)  # 200 year capital
    xs = port.density_df.loss
    S = port.density_df.S
    gS = g(S)
    gS[0] = 1.0

    x = port.q(1 - 1 / return_period_x)
    Fx = port.cdf(x)
    gFx = 1 - g(1-Fx)

    idx = int(port.cdf(K) * len(S))
    lev = np.trapz(S.iloc[:idx], x=xs.iloc[:idx]) + xs[0]
    levg = np.trapz(np.array(gS)[:idx], x=xs.iloc[:idx]) + xs[0]

    dist_name = str(dist).replace('\n', ' ')

    ax0.plot(1-S, xs, lw=1.5, c='C0', label='Loss, $S(x)$')
    ax0.plot(1-gS, xs, lw=1.5, c='C1',
           label=f'Premium $g(S(x))$\nDistortion {dist_name}')

    ax0.plot([Fx, Fx], [0, x], linewidth=0.25, c='C7')
    ax0.plot([gFx, gFx], [0, x], linewidth=0.25, c='C1')
    # ax0.plot([0, Fx], [x, x], linewidth=0.25, c='k')
    ax0.plot([0, gFx], [x, x],  linewidth=2.5, ls='--', c='C2', alpha=1)
    ax0.plot([Fx, 1], [x, x],   linewidth=2.5, ls='--', c='C0', alpha=1)
    ax0.plot([gFx, Fx], [x, x], linewidth=2.5, ls='--', c='C1', alpha=1)

    ax0.set(ylabel='Asset layer', xlabel='Probability of\nnon-exceedance',
          ylim=(0, K), xlim=(0, 1))  # -0.01, 1.01))
    ax0.xaxis.set_ticks([0, gFx,Fx, 1])
    ax0.xaxis.set_ticklabels(['0', '$\\tilde p$', '$p$', '1'])
    ax0.yaxis.set_ticks([0, x, K])
    ax0.yaxis.set_ticklabels(['', '$x$', '$a$'])

    ax0.annotate('Layer\ncapital', ((gFx)/2+0.04, x), ((gFx)/2-0.1+0.04, x +0.3*lev),
                 va='baseline', ha='center', arrowprops={'arrowstyle': '->'})
    ax0.annotate('Layer\nmargin', ((Fx+gFx)/2, x), ((Fx+gFx)/2-0.1, x +0.3*lev),
                 va='baseline', ha='center', arrowprops={'arrowstyle': '->'})
    ax0.annotate('Layer\nloss', ((Fx+1)/2, x), ((Fx+1)/2, x - 0.5*lev),
                 va='baseline', ha='center', arrowprops={'arrowstyle': '->'})

    # middle plot =======================================================================
    ax1.plot(1-S, xs, lw=1.5, c='C0', label='Loss, $S(x)$')
    ax1.plot(1-gS, xs, lw=1.5, c='C1',
           label=f'Premium $g(S(x))$\ndistortion {dist_name}')

    loss_line = [(port.cdf(i), i) for i in np.linspace(K, .01, 200)]
    prem_line = [(1-g(1 - port.cdf(i)), i) for i in np.linspace(K, .01, 200)]

    # top patch
    patches = [Polygon([(0, 0), (0, K), (1 - g(1-port.cdf(K)), K)] + prem_line, True)]
    # bottom
    patches.append(Polygon([(1, 0), (1, K), (port.cdf(K), K)] + loss_line, True))
    # middle
    patches.append(Polygon([(1, 0)] + loss_line[::-1] + prem_line, True))

    # under loss, eq, margin
    p = PatchCollection(patches, alpha=.25, facecolors=['lightsteelblue', 'C0', 'C1' ])
    ax1.add_collection(p)

    ax1.text(0.5, lev / 2, 'Loss', ha='center')
    ax1.text(0.5, (lev + levg) / 2, 'Margin', ha='center')
    ax1.text(0.5, (K + levg) / 2, 'Capital', ha='center')
    ax1.set(ylabel=None, xlabel='Probability of\nnon-exceedance',
            ylim=(0, K), xlim=(0, 1))
    ax1.xaxis.set_ticks([0, 1])
    ax1.xaxis.set_ticklabels(['0', '1'])
    ax1.yaxis.set_ticks(np.linspace(0, K, 1))
    ax1.yaxis.set_ticklabels('')
    ax1.legend().set_visible(False)

    # right hand plot =================================================================
    ax2.bar(0, height=lev, width=1,
          align='edge', alpha=.25)
    ax2.bar(0, height=levg - lev, bottom=lev, width=1,
          align='edge', alpha=.25)
    ax2.axhline(lev, c='C0', lw=1.5)
    ax2.axhline(levg, c='C1', lw=1.5)
    ax2.text(0.5, lev / 2, f'Loss', ha='center', va='center')
    ax2.text(0.5, (lev + levg) / 2, f'Margin', ha='center', va='center')
    ax2.text(0.5, (K + levg) / 2, f'Capital', ha='center')
    ax2.set(xlabel=None)
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_ticks([0, 1])
    ax2.xaxis.set_ticklabels(['0', '1'])
    ax2.set(ylim=[0, K], xlim=[-.0, 1.], xlabel='Traditional\nlayer diagram')


def fig_10_6(port=None, dist=None):
    """
    Same distortion and portfolio as 10_5
    Slight clarification of the diagram vs. book version.

    """
    fig = plt.figure(constrained_layout=True, figsize=(6,4))
    gs = fig.add_gridspec(2, 3)
    gs2 = fig.add_gridspec(1, 1)

    if port is None:
        port = build('port Test agg A 10 claims sev lognorm 50 cv 1 mixed gamma .5', bs=1/16)
    if dist is None:
        dist = build('distortion myph ph 0.45')

    g = dist.g
    ps = np.linspace(0, 1, 400, endpoint=False)
    gps = g(ps)

    ax0 = fig.add_subplot(gs[1,0])
    ax0.set(
        xlabel='Layer $1_{U<s}$', ylabel='$s$, $g(s)$')
    ax0.plot(ps, gps, c='C1',   lw=1.5)
    ax0.plot(ps, ps,  c='C0',   lw=1.5)
    xl = ax0.get_xlim()
    yl = ax0.get_ylim()
    ax0.plot([-1,2], [2,-1], c='C7', ls=':', lw=1)
    ax0.set(xlim=xl, ylim=yl, aspect='equal')
    ax0.text(.75, 0.2, 'reflect', rotation=-45, va='baseline', ha='center')

    ax1 = fig.add_subplot(gs[1,1])
    ax1.set(
        ylabel='Layer $1_{U>p}$', xlabel='$p=1-s$, price',
        aspect='equal')
    ax1.plot(1-gps, 1-ps, c='C1',  lw=1.5)
    ax1.plot(ps,    ps,   c='C0',  lw=1.5)

    q = [port.q(p) for p in ps]
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.set(title='Lee diagram',
            xlabel='Pr(non-exceedance) $p$', ylabel='Asset layer of $X$')
    ax2.plot(1-g(1-ps), q, c='C1',  lw=1.5)
    ax2.plot(ps,        q, c='C0',  lw=1.5)

    for ax in [ax0, ax1, ax2]:
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_ticklabels(['0', '0', '',  '', '', '', '1'])
        if ax is ax2:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2*ax2.get_ylim()[-1]))
        else:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        # ax.grid(lw=.25)

    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="k", lw=0.5)

    ax_top = fig.add_subplot(gs[0,1])
    ax_top.text(0.15, 0.55, "Apply\n$q(p)=F^{-1}(p)$\nto $y$ axis", ha='left',   va='baseline')
    ax_top.text(0.5, 0,    '$(1-g(s), 1-s)$',                           ha='center', va='baseline')
    ax_top.axis('off')

    arrow_apply_q = patches.FancyArrowPatch((0.5, 0.61), (2/3+0.05, 0.85), connectionstyle="angle3", **kw)
    # the background layer
    ax = fig.add_subplot(gs2[0,0])
    ax.axis('off')
    ax.set(xlim=[0,1], ylim=[0,1])
    ax.add_patch(arrow_apply_q)


def natural_scale(port):
    """
    For creating Table 9.15
    """
    margins = np.hstack((np.linspace(.025, .1, 4), np.linspace(.15, .25, 3)))
    roe = .1
    p_defaults = [.01, .05, 0.1, .25]
    df = pd.DataFrame(columns=['limit', 'p_default', 'margin', 'roe', 'exi', 'cvxi', 'lambda', 'u', 'mean_g', 'max_index'], dtype=float)
    limit_dict = {f'Limit{n}': n * 1e6 for n in [1, 5, 10]}
    counter = count(0, 1)
    for line_name in  port.line_names[:3]:
        ag = port[line_name]
        ag_ex = ag.agg_m
        for margin in margins:
            try:
                ruin, find_u, mean, dfi = ag.pollaczeck_khinchine(margin, kind='index', padding=2)
                # ruin, find_u, mean, dfi = ag.cramer_lundberg(margin, kind='interpolate', padding=2)
                # density of integrated distribution
                dfi = pd.Series(dfi, index=ruin.index)
                ex = np.sum(dfi * dfi.index)
                ex2 = np.sum(dfi * dfi.index**2)
                # mean and SD of integrated distibution
                cv = np.sqrt(ex2 - ex*ex) / ex
                mean_g = ex / margin
                for p_default, i in zip(p_defaults, counter):
                    u = find_u(p_default)
                    n_lambda = roe *  u / (margin * ag_ex)
                    df.loc[i] = [limit_dict[line_name], p_default, margin, roe, ex, cv, n_lambda, u, mean_g, ruin.index[-1]]
            except IndexError as e:
                print(e)
    df['u/r'] = df.u / df.margin
    bit = df.set_index(['limit', 'margin', 'p_default'])['lambda'].unstack(1)
    bit.index.names = ['Limit', 'p']
    bit.columns.name = 'Margin'
    return bit


def fig_9_1(port):
    from .case_studies import ClassicalPremium
    port_name = 'gross'
    line_names = ['Limit1', 'Limit10']
    margin = 0.1
    ruins = {}
    find_us = {}
    dfis = {}
    for line_name in line_names:
        ag = port[line_name]
        ruins[line_name], find_us[line_name], mean, dfi = ag.cramer_lundberg(margin, kind='interpolate')
        dfis[line_name] = pd.Series(dfi, index=ruins[line_name].index)
    xmaxs= {'Limit1': 10e6, 'Limit10': 50e6}
    limit_dict = {f'Limit{n}': n * 1e6 for n in [1, 10]}
    n_big_dict ={'Limit1': 10000, 'Limit10': 50000}
    cp = ClassicalPremium({'gross': port}, 110)
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45), constrained_layout=True)
    axi = iter(axs.flat)
    for line_name in line_names:
        ax0 = next(axi)
        ax1 = next(axi)
        ax_ = ax0.twinx()
        xmax = xmaxs[line_name]
        ruins[line_name].index.name = 'Starting capital'
        ruins[line_name].plot(ax=ax0)
        ax0.axhline(1/(1+margin), lw=1)
        ruins[line_name].plot(ax=ax_, ls='--', lw=1)
        ax_.set(ylim=[0.5e-6, 2], ylabel='log probability', yscale='log')
        ax_.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=20))
        ax0.set(xlim=[-xmax/50, xmax], ylim=[-0.05, 1.05], ylabel='Probability of eventual default',
                title=f'Limit {limit_dict[line_name]/1e6:.0f}M, margin {margin}')
        ax_.set(xlim=[-xmax/50, xmax])
        p_default = 0.05
        cp.illustrate(port_name, line_name, ax1, margin, p=p_default, n_big=n_big_dict[line_name], n_sample=100)
        ax1.set(xlabel='Volume or time')


# Module Name : curlyBrace
#
# Author : 高斯羽 博士 (Dr. GAO, Siyu)
#
# Version : 1.0.2
#
# Last Modified : 2019-04-22
#
# This module is basically an Python implementation of the function written Pål Næverlid Sævik
# for MATLAB (link in Reference).
#
# The function "curlyBrace" allows you to plot an optionally annotated curly bracket between
# two points when using matplotlib.
#
# The usual settings for line and fonts in matplotlib also apply.
#
# The function takes the axes scales into account automatically. But when the axes aspect is
# set to "equal", the auto switch should be turned off.
#
# Change Log
# ----------------------
# * **Notable changes:**
#     + Version : 1.0.2
#         - Added considerations for different scaled axes and log scale
#     + Version : 1.0.1
#         - First version.
#
# Reference
# ----------------------
# https://uk.mathworks.com/matlabcentral/fileexchange/38716-curly-brace-annotation
#
# List of functions
# ----------------------
#
# * getAxSize_
# * curlyBrace_

def getAxSize(fig, ax):
    '''
    Get the axes size in pixels.

    Reference: https://uk.mathworks.com/matlabcentral/fileexchange/38716-curly-brace-annotation

    :param fig: matplotlib figure object The of the target axes.
    :param ax: matplotlib axes object The target axes.
    :return: ax_width : float, the axes width in pixels; ax_height : float, the axes height in pixels.

    '''

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width, ax_height = bbox.width, bbox.height
    ax_width *= fig.dpi
    ax_height *= fig.dpi

    return ax_width, ax_height


def curlyBrace(ax, p1, p2, k_r=0.1, bool_auto=True, str_text='', int_line_num=2, fontdict={}, **kwargs):
    '''
    Plot an optionally annotated curly bracket on the given axes of the given figure.

    Note that the brackets are anti-clockwise by default. To reverse the text position, swap
    "p1" and "p2".

    Note that, when the axes aspect is not set to "equal", the axes coordinates need to be
    transformed to screen coordinates, otherwise the arcs may not be seeable.

    **Parameters**

    fig : matplotlib figure object
        The of the target axes.

    ax : matplotlib axes object
        The target axes.

    p1 : two element numeric list
        The coordinates of the starting point.

    p2 : two element numeric list
        The coordinates of the end point.

    k_r : float
        This is the gain controlling how "curvy" and "pointy" (height) the bracket is.

        Note that, if this gain is too big, the bracket would be very strange.

    bool_auto : boolean
        This is a switch controlling wether to use the auto calculation of axes
        scales.

        When the two axes do not have the same aspects, i.e., not "equal" scales,
        this should be turned on, i.e., True.

        When "equal" aspect is used, this should be turned off, i.e., False.

        If you do not set this to False when setting the axes aspect to "equal",
        the bracket will be in funny shape.

        Default = True

    str_text : string
        The annotation text of the bracket. It would displayed at the mid point
        of bracket with the same rotation as the bracket.

        By default, it follows the anti-clockwise convention. To flip it, swap
        the end point and the starting point.

        The appearance of this string can be set by using "fontdict", which follows
        the same syntax as the normal matplotlib syntax for font dictionary.

        Default = empty string (no annotation)

    int_line_num : int
        This argument determines how many lines the string annotation is from the summit
        of the bracket.

        The distance would be affected by the font size, since it basically just a number of
        lines appended to the given string.

        Default = 2

    fontdict : dictionary
        This is font dictionary setting the string annotation. It is the same as normal
        matplotlib font dictionary.

        Default = empty dict

    **kwargs : matplotlib line setting arguments
        This allows the user to set the line arguments using named arguments that are
        the same as in matplotlib.

    **Returns**

    theta : float
        The bracket angle in radians.

    summit : list
        The positions of the bracket summit.

    arc1 : list of lists
        arc1 positions.

    arc2 : list of lists
        arc2 positions.

    arc3 : list of lists
        arc3 positions.

    arc4 : list of lists
        arc4 positions.

    **Reference**

    https://uk.mathworks.com/matlabcentral/fileexchange/38716-curly-brace-annotation
    '''

    fig = ax.get_figure()
    pt1 = [None, None]
    pt2 = [None, None]
    ax_width, ax_height = getAxSize(fig, ax)
    ax_xlim = list(ax.get_xlim())
    ax_ylim = list(ax.get_ylim())

    # log scale consideration
    if 'log' in ax.get_xaxis().get_scale():
        if p1[0] > 0.0:
            pt1[0] = np.log(p1[0])
        elif p1[0] < 0.0:
            pt1[0] = -np.log(abs(p1[0]))
        else:
            pt1[0] = 0.0
        if p2[0] > 0.0:
            pt2[0] = np.log(p2[0])
        elif p2[0] < 0.0:
            pt2[0] = -np.log(abs(p2[0]))
        else:
            pt2[0] = 0
        for i in range(0, len(ax_xlim)):
            if ax_xlim[i] > 0.0:
                ax_xlim[i] = np.log(ax_xlim[i])
            elif ax_xlim[i] < 0.0:
                ax_xlim[i] = -np.log(abs(ax_xlim[i]))
            else:
                ax_xlim[i] = 0.0
    else:
        pt1[0] = p1[0]
        pt2[0] = p2[0]
    if 'log' in ax.get_yaxis().get_scale():
        if p1[1] > 0.0:
            pt1[1] = np.log(p1[1])
        elif p1[1] < 0.0:
            pt1[1] = -np.log(abs(p1[1]))
        else:
            pt1[1] = 0.0
        if p2[1] > 0.0:
            pt2[1] = np.log(p2[1])
        elif p2[1] < 0.0:
            pt2[1] = -np.log(abs(p2[1]))
        else:
            pt2[1] = 0.0
        for i in range(0, len(ax_ylim)):
            if ax_ylim[i] > 0.0:
                ax_ylim[i] = np.log(ax_ylim[i])
            elif ax_ylim[i] < 0.0:
                ax_ylim[i] = -np.log(abs(ax_ylim[i]))
            else:
                ax_ylim[i] = 0.0
    else:
        pt1[1] = p1[1]
        pt2[1] = p2[1]

    # get the ratio of pixels/length
    xscale = ax_width / abs(ax_xlim[1] - ax_xlim[0])
    yscale = ax_height / abs(ax_ylim[1] - ax_ylim[0])

    # this is to deal with 'equal' axes aspects
    if bool_auto:
        pass
    else:
        xscale = 1.0
        yscale = 1.0

    # convert length to pixels,
    # need to minus the lower limit to move the points back to the origin. Then add the limits back on end.
    pt1[0] = (pt1[0] - ax_xlim[0]) * xscale
    pt1[1] = (pt1[1] - ax_ylim[0]) * yscale
    pt2[0] = (pt2[0] - ax_xlim[0]) * xscale
    pt2[1] = (pt2[1] - ax_ylim[0]) * yscale

    # calculate the angle
    theta = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

    # calculate the radius of the arcs
    r = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) * k_r

    # arc1 centre
    x11 = pt1[0] + r * np.cos(theta)
    y11 = pt1[1] + r * np.sin(theta)

    # arc2 centre
    x22 = (pt2[0] + pt1[0]) / 2.0 - 2.0 * r * np.sin(theta) - r * np.cos(theta)
    y22 = (pt2[1] + pt1[1]) / 2.0 + 2.0 * r * np.cos(theta) - r * np.sin(theta)

    # arc3 centre
    x33 = (pt2[0] + pt1[0]) / 2.0 - 2.0 * r * np.sin(theta) + r * np.cos(theta)
    y33 = (pt2[1] + pt1[1]) / 2.0 + 2.0 * r * np.cos(theta) + r * np.sin(theta)

    # arc4 centre
    x44 = pt2[0] - r * np.cos(theta)
    y44 = pt2[1] - r * np.sin(theta)

    # prepare the rotated
    q = np.linspace(theta, theta + np.pi / 2.0, 50)

    # reverse q
    # t = np.flip(q) # this command is not supported by lower version of numpy
    t = q[::-1]

    # arc coordinates
    arc1x = r * np.cos(t + np.pi / 2.0) + x11
    arc1y = r * np.sin(t + np.pi / 2.0) + y11

    arc2x = r * np.cos(q - np.pi / 2.0) + x22
    arc2y = r * np.sin(q - np.pi / 2.0) + y22

    arc3x = r * np.cos(q + np.pi) + x33
    arc3y = r * np.sin(q + np.pi) + y33

    arc4x = r * np.cos(t) + x44
    arc4y = r * np.sin(t) + y44

    # convert back to the axis coordinates
    arc1x = arc1x / xscale + ax_xlim[0]
    arc2x = arc2x / xscale + ax_xlim[0]
    arc3x = arc3x / xscale + ax_xlim[0]
    arc4x = arc4x / xscale + ax_xlim[0]

    arc1y = arc1y / yscale + ax_ylim[0]
    arc2y = arc2y / yscale + ax_ylim[0]
    arc3y = arc3y / yscale + ax_ylim[0]
    arc4y = arc4y / yscale + ax_ylim[0]

    # log scale consideration
    if 'log' in ax.get_xaxis().get_scale():
        for i in range(0, len(arc1x)):
            if arc1x[i] > 0.0:
                arc1x[i] = np.exp(arc1x[i])
            elif arc1x[i] < 0.0:
                arc1x[i] = -np.exp(abs(arc1x[i]))
            else:
                arc1x[i] = 0.0
        for i in range(0, len(arc2x)):
            if arc2x[i] > 0.0:
                arc2x[i] = np.exp(arc2x[i])
            elif arc2x[i] < 0.0:
                arc2x[i] = -np.exp(abs(arc2x[i]))
            else:
                arc2x[i] = 0.0
        for i in range(0, len(arc3x)):
            if arc3x[i] > 0.0:
                arc3x[i] = np.exp(arc3x[i])
            elif arc3x[i] < 0.0:
                arc3x[i] = -np.exp(abs(arc3x[i]))
            else:
                arc3x[i] = 0.0
        for i in range(0, len(arc4x)):
            if arc4x[i] > 0.0:
                arc4x[i] = np.exp(arc4x[i])
            elif arc4x[i] < 0.0:
                arc4x[i] = -np.exp(abs(arc4x[i]))
            else:
                arc4x[i] = 0.0
    else:
        pass
    if 'log' in ax.get_yaxis().get_scale():
        for i in range(0, len(arc1y)):
            if arc1y[i] > 0.0:
                arc1y[i] = np.exp(arc1y[i])
            elif arc1y[i] < 0.0:
                arc1y[i] = -np.exp(abs(arc1y[i]))
            else:
                arc1y[i] = 0.0
        for i in range(0, len(arc2y)):
            if arc2y[i] > 0.0:
                arc2y[i] = np.exp(arc2y[i])
            elif arc2y[i] < 0.0:
                arc2y[i] = -np.exp(abs(arc2y[i]))
            else:
                arc2y[i] = 0.0
        for i in range(0, len(arc3y)):
            if arc3y[i] > 0.0:
                arc3y[i] = np.exp(arc3y[i])
            elif arc3y[i] < 0.0:
                arc3y[i] = -np.exp(abs(arc3y[i]))
            else:
                arc3y[i] = 0.0
        for i in range(0, len(arc4y)):
            if arc4y[i] > 0.0:
                arc4y[i] = np.exp(arc4y[i])
            elif arc4y[i] < 0.0:
                arc4y[i] = -np.exp(abs(arc4y[i]))
            else:
                arc4y[i] = 0.0
    else:
        pass

    # plot arcs
    ax.plot(arc1x, arc1y, **kwargs)
    ax.plot(arc2x, arc2y, **kwargs)
    ax.plot(arc3x, arc3y, **kwargs)
    ax.plot(arc4x, arc4y, **kwargs)

    # plot lines
    ax.plot([arc1x[-1], arc2x[1]], [arc1y[-1], arc2y[1]], **kwargs)
    ax.plot([arc3x[-1], arc4x[1]], [arc3y[-1], arc4y[1]], **kwargs)

    summit = [arc2x[-1], arc2y[-1]]

    if str_text:
        int_line_num = int(int_line_num)
        str_temp = '\n' * int_line_num
        # convert radians to degree and within 0 to 360
        ang = np.degrees(theta) % 360.0
        if (ang >= 0.0) and (ang <= 90.0):
            rotation = ang
            str_text = str_text + str_temp
        if (ang > 90.0) and (ang < 270.0):
            rotation = ang + 180.0
            str_text = str_temp + str_text
        elif (ang >= 270.0) and (ang <= 360.0):
            rotation = ang
            str_text = str_text + str_temp
        else:
            rotation = ang
        ax.axes.text(arc2x[-1], arc2y[-1], str_text, ha='center', va='center', rotation=rotation, fontdict=fontdict)
    else:
        pass

    arc1 = [arc1x, arc1y]
    arc2 = [arc2x, arc2y]
    arc3 = [arc3x, arc3y]
    arc4 = [arc4x, arc4y]

    return theta, summit, arc1, arc2, arc3, arc4
