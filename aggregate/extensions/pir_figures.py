# standalone figures from PIR book

import matplotlib.pyplot as plt
from matplotlib import ticker
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




def fig_10_3():
    """
    Figure 10.3 Illustrating distortion functions
    """
    fig, axs = plt. subplots(1, 2, figsize=(2 * FIG_W, FIG_W), constrained_layout=True)
    d = Distortion('ph', 0.4)
    s = 0.3
    g_loss_margin_equity(axs, d, s)
    for ax in axs.flat:
        ax.set(title=None, xlabel='$s$, probability of loss to layer',
               ylabel='$g(s)$, price of layer $1_{U<s}$', aspect='equal')


def g_loss_margin_equity(axs, dist, s=0.25):
    '''
    (s, g(s)) with vertical line at s and split loss, premium, margin, and capital labelled
    :param a:
    :param dist:
    :param s:
    :return:
    '''
    # s, g(s), premium, loss, margin on [0,1]^2
    # s = 0.25
    g = dist.g
    N = 1000
    ps = np.linspace(0, 1, N, endpoint=False)
    gs = g(ps)
    gs[0] = 1.0
    sm = 0.085
    g_s = g(s)
    lbl = str(dist).replace('\n', ' ')

    def setbg(t):
        t.set_bbox(dict(facecolor=PLOT_FACE_COLOR, alpha=0.85, edgecolor='none', boxstyle='square,pad=.1'))

    for a in axs.flat:
        a.plot(ps[1:], gs[1:], lw=1.5)
        a.plot(ps, ps, linewidth=1.5, c='k', ls='--', alpha=1)
        a.axis([0.0, 1.025, 0.0, 1.025])
        a.set(aspect='equal', xlabel='$s$', ylabel='$g(s)$',
              title=f'Insurance Statistics\n{lbl}')
        # a.grid(lw=0.25)

    # a is the right hand plot
    a.plot([s, s], [0, s], c='k', alpha=0.25, linewidth=2.5)
    a.plot([s, s], [s, g_s], c='k', alpha=.75, linewidth=2.5)
    a.plot([s, s], [g_s, 1], c='k', alpha=0.45, linewidth=2.5)
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
    curlyBrace(a, p0, p1, str_text=None, int_line_num=2, k_r=0.055, c='k', lw=0.5)
    # margin
    curlyBrace(a, p1m, p2m, str_text=None, int_line_num=2, k_r=0.075, c='k', lw=0.5)
    # loss
    curlyBrace(a, p2, p3, str_text=None, int_line_num=2, k_r=0.075, c='k', alpha=0.5, lw=0.5)
    # premium
    g_s = dist.g(s)
    curlyBrace(a, (.625, g_s), (.625, 0), str_text=None, int_line_num=2, k_r=0.0375, c='k', lw=0.5)
    a.text(.625 + sm, g_s / 2, 'Premium\n$=g(s)$', va='center', ha='left')
    # a.plot([0, s], [g_s, g_s], lw=1, c='k')
    a.plot([0, .626], [g_s, g_s], lw=.5, c='k', ls='-')






















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
