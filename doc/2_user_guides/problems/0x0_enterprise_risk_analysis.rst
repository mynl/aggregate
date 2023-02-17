Enterprise Risk Analysis
----------------------------

This section re-analyzes reinsurance structure alternatives introduced in :cite:t:`Brehm2007`,
Enterprise Risk Analysis for Property & Liability Insurance Companies.
This book is the ERM text on the syllabus for CAS Exam Part 7.

Reinsurance Example
~~~~~~~~~~~~~~~~~~~~

This section analyzes the example given in Section 2.5 of Enterprise Risk Analysis.

**Assumptions.** ABCD writes 33M excess property and casualty business.


* ABCD total gross

    - Loss ratio: 69.36%
    - Expense ratio: 23%
    - Combined ratio: 92.36%
    - Margin 2.52M

* Casualty

    - 14M premium
    - 78% expected loss ratio
    - 5M limits
    - 4M xs 1M reinsurance, ceded premium 4.41M

* Property

    - 19M premium
    - 63% expected loss ratio
    - 20M limits
    - 17M xs 3M per risk reinsurance, ceded premium 2.36M
    - 95% share of 24M xs 1M cat reinsurance, ceded premium 1.53M with 1@100%
    - Cat program designed to cover to 250-year event.

* Reinsurance total

    - Average recoveries 5.08M
    - Ceded premium 8.3M
    - Net premium 24.7M

* Mythical alternative program

    - Stop-loss, 20 xs 30, ceded premium 1.98


**Additional assumptions.** There are no details of the stochastic model, so we assume

* Frequency and severity models per DecL below,
* Split the property losses into cat and non-cat by assuming that cat premium
  equals 2M, non-cat premium 17M, at the same loss ratios (this is just a
  split of losses, the by line loss ratios are not used), and
* Cat, non-cat and casualty are independent.
* Free and unlimited reinstatements on the catastrophe protection. See REF for
  a discussion of reinstatements.

Stochastic Models and Baseline Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct the gross and net portfolios. All amounts in millions.

Gross Portfolio
"""""""""""""""""

The 250-year cat PML is printed last, to compare with the 25M program.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd, mv
    import pandas as pd
    import matplotlib.pyplot as plt

    abcd = build('port ABCD '
             'agg Casualty 14.0 premium at 78% lr '
                 '5 xs 0 '
                 'sev lognorm 0.1 cv 10 '
                 'mixed gamma 0.3 '
             'agg PropertyNC 17.0 premium at 63% lr '
                 '25 xs 0 '
                 'sev lognorm [0.1 1] cv [5 10] wts [.7 .3] '
                 'mixed gamma 0.1 '
             'agg PropertyC 2.0 premium at 63% lr '
                 '150 xs 0 '
                 'sev 3 * pareto 2.375 - 3 '
                 'poisson ', bs=1/128, approximation='exact')
    qd(abcd)
    mv(abcd)
    print(abcd['PropertyC'].q(0.996))

Net Portfolio
""""""""""""""

.. ipython:: python
    :okwarning:

    abcd_net = build('port ABCD:Net '
             'agg Casualty 14.0 premium at 78% lr '
                 '5 xs 0 '
                 'sev lognorm 0.1 cv 10 '
                 'occurrence net of 4 xs 1 '
                 'mixed gamma 0.3 '
             'agg PropertyNC 17.0 premium at 63% lr '
                 '25 xs 0 '
                 'sev lognorm [0.1 1] cv [5 10] wts [.7 .3] '
                 'occurrence net of 17 xs 3 '
                 'mixed gamma 0.1 '
             'agg PropertyC 2.0 premium at 63% lr '
                 '150 xs 0 '
                 'sev 3 * pareto 2.375 - 3 '
                 'occurrence net of 24 xs 1 '
                 'poisson ', bs=1/128, approximation='exact')
    qd(abcd_net)
    qd(abcd_net.est_sd)

Ceded Portfolio
""""""""""""""""

.. ipython:: python
    :okwarning:

    abcd_ceded = build('port ABCD:Ceded '
             'agg Casualty 14.0 premium at 78% lr '
                 '5 xs 0 '
                 'sev lognorm 0.1 cv 10 '
                 'occurrence ceded to 4 xs 1 '
                 'mixed gamma 0.3 '
             'agg PropertyNC 17.0 premium at 63% lr '
                 '25 xs 0 '
                 'sev lognorm [0.1 1] cv [5 10] wts [.7 .3] '
                 'occurrence ceded to 17 xs 3 '
                 'mixed gamma 0.1 '
             'agg PropertyC 2.0 premium at 63% lr '
                 '150 xs 0 '
                 'sev 3 * pareto 2.375 - 3 '
                 'occurrence ceded to 24 xs 1 '
                 'poisson ', bs=1/128, approximation='exact')
    qd(abcd_ceded)
    qd(abcd_ceded.est_sd)

Reinsurance Summary
""""""""""""""""""""""

The bottom table shows expected losses, counts, severity, loss ratios and margins
implicit in the given reinsurance structure, pricing, and the gross stochastic model.
The non-cat property reinsurance has the highest ceded loss ratio and the cat program
the lowest.

.. ipython:: python
    :okwarning:

    re_all = pd.concat((a.reinsurance_occ_layer_df for a in abcd_net),
        keys=abcd_net.unit_names, names=['unit', 'share', 'limit', 'attach']); \
    re_all = re_all.drop('gup', axis=0, level=3); \
    qd(re_all, sparsify=False)
    re_summary = re_all.iloc[:, [0, 3, 6, 7]]; \
    re_summary.columns = ['ex', 'cv', 'en', 'severity']; \
    re_summary['premium'] = [4.41, 2.36, 1.53]; \
    re_summary['lr'] = re_summary.ex / re_summary.premium; \
    re_summary['margin'] = re_summary.premium - re_summary.ex; \
    qd(re_summary)

Underwriting Result Distributions
""""""""""""""""""""""""""""""""""

Make the underwriting result distributions, including the proposed stop loss reinsurance (computed by hand).
The dataframe ``compare`` accumulates the gross, ceded, and net probability mass functions. We use these
to determine statistics and to plot.

.. ipython:: python
    :okwarning:

    compare = abcd.density_df[['loss', 'p_total']]; \
    compare.columns = ['loss', 'gross']; \
    compare['gross_uw'] = 33 - compare.loss; \
    compare['net_current'] = abcd_net.density_df.p_total; \
    compare['net_current_uw'] = 33 - 4.41 - 2.36 - 1.53 - compare.loss;
    from aggregate import make_ceder_netter
    compare['net_stoploss'] = abcd.density_df.p_total; \
    c, n = make_ceder_netter([(1, 20, 30)]); \
    compare['nsll'] = n(compare.loss); \
    g = compare.groupby('nsll').net_stoploss.sum(); \
    compare['net_stoploss'] = 0.0; \
    compare.loc[g.index, 'net_stoploss'] = g; \
    compare['net_stoploss_uw'] = 33 - 1.98 - compare.loss;

Comparison with ERA Book Figures
""""""""""""""""""""""""""""""""""

Statistics summary, compare Figure 2.5.2.

.. ipython:: python
    :okwarning:

    from aggregate import MomentWrangler
    from scipy.interpolate import interp1d
    ans = []; cdfs = []
    for xs, den in [(compare.gross_uw, compare.gross), (compare.net_current_uw, compare.net_current),
                     (compare.net_stoploss_uw, compare.net_stoploss)]:
        xd = xs * den
        ex1 = np.sum(xd)
        xd *= xs
        ex2 = np.sum(xd)
        ex3 = np.sum(xd * xs)
        mw = MomentWrangler()
        mw.noncentral = ex1, ex2, ex3
        ans.append(mw)
        cdfs.append(interp1d(den.cumsum(), xs))

    fig_252 = pd.concat([i.stats for i in ans], keys=['Gross', 'Current', 'StopLoss'], axis=1)
    for p in [0.01, 0.99]:
        fig_252.loc[f'q({p})'] = [float(i(p)) for i in cdfs]
    qd(fig_252)


Plot of densities and distributions, compare Figure 2.5.3 and 2.5.4.

.. ipython:: python
    :okwarning:

    fig, axs = plt.subplots(1, 3, figsize=(3 * 3.5, 2.45), constrained_layout=True)
    ax0, ax1, ax2 = axs.flat

    for ax in [ax0, ax1]:
        ax.plot(compare.gross_uw, compare.gross, label='Gross')
        ax.plot(compare.net_current_uw, compare.net_current, label='Net, current')
        yl = ax.get_ylim()
        ax.plot(compare.net_stoploss_uw, compare.net_stoploss, label='Net, stop loss')
        ax.legend(loc='upper left')
        ax.set(xlim=[-45, 30], ylim=yl)
        ax.axvline(0, lw=.25, c='C7')
    ax1.set(yscale='log', ylim=[1e-9, 1], title='Log density'); \
    ax0.set(title='Mixed density/mass function');

    ax2.plot(compare.gross_uw, 1 - compare.gross.cumsum(), label='Gross'); \
    ax2.plot(compare.net_current_uw, 1 - compare.net_current.cumsum(), label='Net, current'); \
    ax2.plot(compare.net_stoploss_uw, 1 - compare.net_stoploss.cumsum(), label='Net, stop loss'); \
    ax2.legend(loc='upper left'); \
    ax2.set(xlim=[-45, 30], ylim=[-0.025, 1.025]);
    @savefig gc253.png
    ax2.axvline(0, lw=.25, c='C7');

Numerical distribution of underwriting results at various return points, compare Figure 2.5.5.
Given there was no information about the stochastic model provided, and the model here is based
on common benchmarks, the agreement between the two distributions is striking.

.. ipython:: python
    :okwarning:

    fig_255 = pd.DataFrame(columns=['Gross', 'Current', 'StopLoss'], dtype=float)

    for p in [.0025, .005, 0.0075, .01, .0125, .015, .0175, .02,
              .04, .06, .08, .1, .12, .14, .16, .18, .2, .22, .24,
              .25, .26, .28, .3, .32, .34, .36, .38, .4, .42, .44,
              .46, .48, .5]:
        fig_255.loc[p] = [float(i(1-p)) for i in cdfs]
    fig_255.index.name = 'p'
    qd(fig_255, float_format=lambda x: f'{x:.3f}', max_rows=len(fig_255))



Modern Analysis
~~~~~~~~~~~~~~~~~~

The first step is to analyze the pricing in the context of needed capital.
Strip expenses out (at 23% across all units) to determine a net (of expenses)
technical premium.

.. ipython:: python
    :okwarning:

    er = 0.23
    df = pd.DataFrame({'unit': ['Casualty', 'PropertyNC', 'PropertyC'],
                              'prem': [14, 17, 2],
                                'gross_loss': [a.est_m for a in abcd]}).set_index('unit')
    df['ceded_prem'] = [4.41, 2.36, 1.53]; \
    df['net_prem'] = df.prem - df.ceded_prem; \
    df['tech_prem'] = df.prem * (1 - er); \
    df['margin'] = df.tech_prem - df.gross_loss; \
    df.loc['Total'] = df.sum(0); \
    df['lr'] = df.gross_loss / df.prem; \
    df['cr'] = df.lr + er; \
    df['tech_lr'] = df.gross_loss / df.tech_prem;
    fp = lambda x: f'{x:.1%}';
    fc = lambda x: f'{x:.2f}'
    qd(df, float_format=fc, formatters={'lr':fp, 'cr': fp, 'tech_lr': fp})

The example does not specify a capital standard. Let's investigate the implied
return on capital at different capital standards. The capital standard is
expressed as a loss percentile. The next calculation produces a table of
returns expressed as a cost of capital (``coc``). It also shows the expected
policyholder deficit.


.. ipython:: python
    :okwarning:

    tech_prem = df.loc['Total', 'tech_prem']; \
    ps = [.99, .995, .996, .999]; \
    As = [abcd.q(p) for  p in ps]; \
    el = abcd.density_df.loc[As, 'exa_total']; \
    margin = tech_prem - el; \
    cocs = margin /  (As - tech_prem); \
    summary = pd.DataFrame({'p': ps, 'a': As, 'prem':tech_prem, 'el': el,
                            'margin': margin, 'tech_lr': el / tech_prem, 'coc': cocs,
                           'epd': (abcd.est_m - el) / abcd.est_m}).set_index('p')
    summary.index = [fp(i) for i in summary.index]; \
    summary.index.name = 'p'; \
    qd(summary, float_format=fc, formatters={'coc': fp, 'tech_lr': fp, 'epd': fp})

Based on this analysis, we assume a 99.5% (200-year) capital standard, which
gives a reasonable 8% return on capital. 200-year capital is also the
Solvency II standard.

From here, the analysis could proceed in many directions. The approach we select is

#. Calibrate a set of distortions to total pricing on a gross basis with the
   200-year capital standard.
#. Analyze the pricing implied by these distortions on the net book and its
   natural allocation by unit.
#. Compare the model value (implied ceded premium) with market reinsurance
   price.

The model value is the *maximum* amount that is consistent with pricing
according to each distortion. Reinsurance cheaper than the model value is
efficient: replacing traditional capital with reinsurance capital lowers the
economic cost of bearing risk.

Calibrate Distortions
""""""""""""""""""""""

Extract the exact cost of capital implied by given gross pricing.


.. ipython:: python
    :okwarning:

    coc = summary.loc['99.5%', 'coc']
    print(coc)

Calibrate distortions to current pricing. Use five one-parameter distortion
families

#. constant cost of capital (CCoC),
#. proportional hazard (PH)
#. Wang,
#. dual, and
#. TVaR.

They are sorted from most tail-centric (expensive for tail risk) to cheapest. See
:cite:t:`PIR`.

The next dataframe shows the asset level and implied loss ratio,
distortion name, survival probability (0.5%), expected loss, premium, premium
to capital leverage (``PQ``), the cost of (return on) capital, the distortion
family parameter, and the parameterization error. The calibrated premium
matches the technical premium.

.. ipython:: python
    :okwarning:

    abcd.calibrate_distortions(ROEs=[coc], Ps=[.995], strict='ordered');
    qd(abcd.distortion_df)

The plot show this effect: COC is fattest on the left for small exceedance
probabilities (high losses), whereas TVaR is fattest on the right.

.. ipython:: python
    :okwarning:

    fig, axs = plt.subplots(1, 5, figsize=(10.0, 2.1), constrained_layout=True)
    for ax, (k, v) in zip(axs.flat, abcd.dists.items()):
        v.plot(ax=ax)
    @savefig gc_dist.png
    fig.suptitle('Comparison of distortion functions giving current market premium in total')

Analyze Implied Pricing
""""""""""""""""""""""""

Apply the distortions to the net portfolio and analyze the resulting pricing
using :meth:`analyze_distortions`, which includes a by-unit margin
allocation. The dataframe ``ans.comp_df`` contains a wealth of other
information; we just focus on the premium. The last row, ``Technical``, shows
market reinsurance pricing.

.. ipython:: python
    :okwarning:

    abcd_net.dists = abcd.dists
    ansn = abcd_net.analyze_distortions(p=0.996, add_comps=False); \
    ans = abcd.analyze_distortions(p=0.996, add_comps=False); \
    bit = pd.concat((ans.comp_df.xs('P', 0, 1), ansn.comp_df.xs('P', 0, 1),
                    ans.comp_df.xs('P', 0, 1) - ansn.comp_df.xs('P', 0, 1)),
                    axis=1, keys=['gross', 'net', 'ceded']); \
    bit = bit.iloc[[0, 2,-1, 1, -2]]; \
    bit.loc['Technical'] = 0.0; \
    bit.loc['Technical', 'gross'] = df.tech_prem.sort_index().values; \
    bit.loc['Technical', 'ceded'] = df.ceded_prem.sort_index().values; \
    bit.loc['Technical', 'net'] = df.net_prem.sort_index().values; \
    qd(bit, sparsify=False, line_width=50)


Compare Model Value and Market Price
"""""""""""""""""""""""""""""""""""""

Focus on the last block above, under ``ceded``. The rows ``Dist ...`` show the
model value of reinsurance according to each distortion. The row
``Technical`` shows the market price. The market suggests to buy when the
value is greater than the price.

The analysis provides a clear answer only for casualty, where the model value
of reinsurance is much lower than the market price for all distortions: don't
buy the reinsurance.

For property cat, CCoC, the most tail-centric distortion, sees a lot of value
in the reinsurance --- hardly surprising. All the other less tail-centric
distortions do not see it as adding value overall (lower value than market
price). The order of the distortions and their assessment of the value of cat
reinsurance are perfectly aligned, as they were for casualty albeit in the
opposite order.

For property non-cat, the PH and Wang distortions see value, the others do
not, though dual is close. This is the most interesting case because the
ranking does not agree with the distortion ordering (as it does for the other
two units). Property non-cat contributes to volatility and tail-risk, and so
is more nuanced. Management often struggles with property risk reinsurance
because tail-centric measures understate the value it provides. Actuaries
stuggle to find analytic methods that capture its management-perceived value.
The range of distortions considered covers the two views well.

In total the program is not seen as good value by any of the distortions. Since
they span the reasonable range of risk preferences, this is a robust result.

Management often cares about more than just tail risk and they generally
rejects the findings from CCoC. Whether or not they see value in reinsurance
is sensitive to their exact risk appetite. These findings are consistent with
the fact that each company tends to structure its reinsurance differently,
tailored to their own risk appetite. Difference in risk appetite have a
material impact on decision making.

Analysis for Stop Loss Reinsurance
"""""""""""""""""""""""""""""""""""

Here is the analysis for the stop loss reinsurance. This analysis is manual,
because the net of stop loss distribution for a :class:`Portfolio` is not
currently built-in. We have to extract the relevant distributions and apply
the distortions, estimate ``a_stoploss`` the net asset requirement at
``p=0.995`` (rounded to be a multiple of ``bs``), determine the net expected
loss and the model value. Recall ``compare.net_stoploss`` is the density of
the net of stop-loss loss outcome. ``S1`` is used to create its survival
function, to which the distortion is applied to determine pricing. ``exa``
and ``exag`` are the objective and risk adjusted losses (model value) given
an asset level ``a``, computed as :math:`\int_0^a S` and :math:`\int_0^a g
(S)` respectively (see PIR REF). We then select the relevant row and assemble
the answer.

.. ipython:: python
    :okwarning:

    S0 = pd.Series(compare.net_stoploss, index=compare.loss); \
    S0.name = 'S'; \
    S1 = S0[::-1].shift(1, fill_value=0).cumsum(); \
    a0 = float(interp1d(S0.cumsum(), S0.index)(0.995)); \
    a_stoploss = abcd.snap(a0); \
    print(f'Net of stoploss assets {a_stoploss:.3f}');
    net_el_stoploss_unlim = (compare.loss * compare.net_stoploss).sum(); \
    net_el_stoploss = (np.minimum(compare.loss, a_stoploss) * compare.net_stoploss).sum(); \
    epd = 1 - net_el_stoploss / net_el_stoploss_unlim; \
    qd(pd.Series([net_el_stoploss_unlim, net_el_stoploss, epd], index=['unlimited net loss', 'net loss limited by assets', 'epd']));
    pricer = S1.to_frame().sort_index();
    for nm, dist in abcd.dists.items():
        pricer[f'{nm}_exa'] = pricer['S'].shift(1, fill_value=0).cumsum() * abcd.bs
        pricer[f'{nm}_gS'] = dist.g(pricer.S)
        pricer[f'{nm}_exag'] = pricer[f'{nm}_gS'].shift(1, fill_value=0).cumsum() * abcd.bs
        pricer = pricer.sort_index()
    pricer = pricer.loc[[a_stoploss]]; \
    pricer.columns = pricer.columns.str.split('_', expand=True); \
    comp = pricer.stack(0).droplevel(0,0); \
    comp.loc['Technical'] = [net_el_stoploss, tech_prem - 1.98, np.nan]; \
    comp['stoploss_value'] = tech_prem - comp.exag; \
    comp = comp.sort_values('stoploss_value', ascending=False); \
    qd(comp)

The output table reveals that the stop loss value is greater than its market
price for the CCoC, PH, and Wang distortions, but less for the dual and TVaR.
Thus, management averse to tail risk regard it as beneficial, but those more
concerned with volatility and body risk do not see it as worthwhile.

A note of caution is in order on this analysis. Stop loss structures are a
broker favorite, but are generally not liked by reinsurers. Aggregate
features are hard to underwrite and price, and the lower premium is not
attractive. A treaty similar to the proposed stop loss would be very hard to
find in the market.


Visualizing Risk
~~~~~~~~~~~~~~~~~~~~

The next figure shows the kappa functions, a handy way to visualize which
units are contributing to total risk across the loss spectrum (see REF). Here
the horizontal axis is total loss. The middle plot shows the reinsurance is
quite effective at lowering the risk from Property NC (green line), but less
effective at altering the risk profile of the other two lines. In particular,
cat (red line) still dominates the tail risk.

.. ipython:: python
    :okwarning:

    fig, axs = plt.subplots(1, 3, figsize=(3 * 3.5, 2.55), constrained_layout=True)

    for ax, a in zip(axs.flat, [abcd, abcd_net, abcd_ceded]):
        mx = a.q(0.9999)
        a.density_df.filter(regex='exeqa_[CPt]').plot(ax=ax,
            xlim=[0, mx], ylim=[0, mx], title=a.name);
        ax.set(xlabel='loss, $x$');
    axs.flat[0].set(ylabel='$E[X_unit | X=x]$');
    @savefig gc_kappa.png
    fig.suptitle('Conditional loss as a function of x for each unit');
