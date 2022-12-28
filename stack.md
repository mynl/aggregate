# TODO stack

Tuesday
=======

* estimating bucket size: put error code into agg *and how into port*?
* All references! REF!
* Cat: put in changes to climate change el impact! (in progress)
* Reins: occ plot function not used in 2.6.6 = is it the right function?; duplication in explanation of new reins function (esp at the end).
* Create a WARNINGS OFF function in utilities (or in build.warnings_off())
* Actuary LEV so far off (student, see below)
* shift and loc with mean and cv (sev in 10 mins)
* How does DecL update?


Xi to EXi|X to... graph progression... also consider scaled versions
---------------------------------------------------------------------

    %%sf 2 3 2.45 3.5
    # mx = p07.q(1-1e-5)
    bite = p07.density_df.filter(regex='exeqa_[ABC]').loc[0:mx]
    bitp = p07.density_df.filter(regex='p_[ABC]').loc[0:mx]

    bitp = bitp.cumsum()
    xs = p07.density_df.loc[0:mx, 'loss']
    ps = p07.density_df.loc[0:mx, 'p_total'].cumsum()

    xl = [-0.025, 1.025]
    # xl = [.98, 1.001]
    # yl = p07.q(0.995)
    # yl = [-yl/50, yl]

    for ax, ce, cp, a in zip(axs.flat, bite, bitp, p07):
        ax.plot(bitp[cp], xs, label='Standalone')
        ax.plot(ps, bite[ce], label='E[Xi | X]')
        ax.axhline(a.agg_m, lw=.5, c='C7', label='Expected')
        ax.set(title=f'{a.name}, EX={a.agg_m:,.0f}', xlim=xl, ylim=yl)
        ax.legend()

    # mx = p07.q(1-1e-6)
    # xl2 = [-mx/50, mx]
    bite = p07.density_df.filter(regex='exeqa_[ABC]')
    bitp = p07.density_df.filter(regex='p_[ABC]').cumsum()
    xs = p07.density_df['loss']
    ps = p07.density_df['p_total'].cumsum()

    for ax, ce, cp, a in zip(axs.flat[3:], bite, bitp, p07):
        ax.plot(xs, xs, label='total')
        ax.plot(xs, bite[ce], label='E[Xi | X]')
        ax.set(title=f'{a.name}, EX={a.agg_m:,.0f}',  ylim=yl, xlabel='loss')
        ax.legend()


Handy Stuff
===========

* `np.newaxis`, x=x[:, np.newaxis]

actuary student example, why is lev so far off analytic answer?

::

        a01 = build('agg Actuary:01 '
                    '2000 premium at 0.675 lr 1000 xs 0 '
                    'sev lognorm 50 cv 1.25 '
                    'poisson', bs=1/8)
        qd(a01)

        qd(a01.sf(2000), a01.sf(2500))
        qd(a01.density_df.loc[[2500], ['F', 'S', 'lev', 'epd']])


    a01.density_df.loc[2500, ['F', 'exlea']].prod() + 2500 * a01.density_df.loc[2500, 'S']

    from aggregate import lognorm_lev, mu_sigma_from_mean_cv

    mu, sigma = mu_sigma_from_mean_cv(a01.agg_m, a01.agg_cv)
    lev = lognorm_lev(mu, sigma, 1, 2500)
    lev_agg = a01.density_df.loc[2500, 'lev']
    default = a01.agg_m - lev
    epd = default / a01.est_m
    default_agg = a01.est_m - lev_agg
    pd.DataFrame((lev, default, lev_agg, default_agg, epd, default_agg / a01.agg_m),
                 index=pd.Index(['Lognorm LEV', 'Lognorm Default', 'Agg LEV',
                 'Agg Default', 'Lognorm EPD', 'Agg EPD'],
                 name='Item'),
                 columns=['Value'])


shift.loc and cv::

    a = build('agg T 1 claim sev 10 * gamma 1 cv 3 + 50 fixed')
    qd(a)
    a.plot()
    sigma = a.sevs[0].fz.args[0]
    (np.exp(sigma**2)-1)**.5, 1/sigma**.5




0. Uncertainty principle for bs: too small not enough space; too large miss features of sev; for given n there is a min possible error
1. qd looks for like columns / column format guesser?
3. ZM and ZT
3. cat paper; match Jewson; ILW pricing?
4. rec bucket - do some testing to determine a good p.
5. Update efficiently - used anywhere?
6. Aggregate.en is unreliable; where/how is it used
7. sev_cdf etc. are unrealiable for pdfs when there are masses
8. with picks the analytic severities are not altered...be careful!
9. with picks you should invalidate / reompute statistics etc.
10. with cession you should recompute statistics? or make clear it is gross/
11. Formatting for MultiIndex with gup (reins_audit_df) messed up because not float
12. Install in Ubuntu
13. Add script for intall from source into readme.rst
14. References!!!!
15. This fails:  s = build('sev LG loggamma 10 cv .5') and for Pareto
16. dot in names fuck up things eg analyze_distortion (seems colon works)
17. Update reins section for latest re object attributes



Development Outline (TODOs) from Ch 6
=========================================

Non-Programming Enhancements
----------------------------
* Add a library of realistic severity curves by line and country
* Add by-line industry aggregate distributions in DecL based on a method of moments fit to historical data

Programming Enhancements
-------------------------

.. * Credit modeling: what is distortion implied by bond credit curve? By cat bond pricing?
.. * Jon Evans note and severity??
.. * Jed note??

.. Short Term
.. -----------
.. * Width of printed output
.. * Understand output for collateral and priority!
.. * Output Levy measure
.. * Funky objects from JacodS? Simple jump examples

.. Medium Term
.. ------------
.. * recommend_bucket function when passed infinity?
.. * More consistent and informative reports and plots (e.g. include severity match in agg)
.. * Convex Hull distortion built from pricing
.. * Delete items easily from the database
.. * Save / load from non-YAML, persist the database; dict to Dec language converter? Get rid of YAML dependence
.. * Using agg as a severity (how!)
.. * Name as a member in dict vs list conniptions (put up with duplication?)

.. Nice to Have Enhancements
.. -------------------------
.. * How to model two reinstatements?

Useful?

.. |appveyor| image:: https://img.shields.io/appveyor/ci/mynl/aggregate/master.svg?maxAge=3600&label=Windows
    :target: https://ci.appveyor.com/project/mynl/aggregate
    :alt: Windows tests (Appveyor)

* Papers
    - Clark re -> re pricing
    - Wang Agg
    - Robertson FFT
    - Bear and Nemlick
    - Clark Cred of Tower
    - Mata and Verheyen
    - Fisher
        + p 17; retro rating formula prem = (B + cL) x T (c=loss conv, B=basic, T=tax); basic = expenses, occ limt; agg cost; savings
        + large ded / sir plans
        + dividend plans
        + p 40 table M charge; ins charge, savings; entry ratio, table M_D, table L
    - Hipf
    - Ludwig property curves
    - COPLFR risk transfer testing
    - Blier-wong Generating function method for the efficient computation of expected allocations
    - Denuit other papers on kappa


Jewson cat...
Brown and Wolfe - estimation of variance of percentile estimates
Corro and Tseng: NCCI 2014 ELFs

6. local moment matching ? for the first few points? jnwts (juice not worth the squeeze)



## Done

* forward and backward computation; discrete -> round method
* pass recommend_p as arg to build
* correct width for qd and what displays in pdf?



## Considered and rejected


AAS Paper!


# Junk Yard

Reproducing a Book Case Study
------------------------------

TODO Code here!

Bodoff’s Examples
-----------------

This section shows how to reproduce Bodoff’s “Thought experiment 1”. He considers a situation of two losses wind, *W*, and earthquake, *Q*, where *W* and *Q* are independent, *W* takes the value 99 with probability 20% and otherwise zero, and *Q* takes the value 100 with probability 5% and otherwise zero. Total losses *Y* = *W* + *Q*. There are four possibilities outcomes.

.. table:: Bodoff Thought Experiment 1

   =================== ===============
   **Event**           **Probability**
   =================== ===============
   No Loss             0.76
   *W* = 99            0.19
   *Q* = 100           0.04
   *W* = 99, *Q* = 100 0.01
   =================== ===============

Here are the ``Aggregate`` programs for the four examples Bodoff considers.

::

   port BODOFF1 note{Bodoff Thought Experiment No. 1}
       agg wind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed
       agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed


   port BODOFF2 note{Bodoff Thought Experiment No. 2}
       agg wind  1 claim sev dhistogram xps [0,  50] [0.80, 0.20] fixed
       agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed


   port BODOFF3 note{Bodoff Thought Experiment No. 3}
       agg wind  1 claim sev dhistogram xps [0,   5] [0.80, 0.20] fixed
       agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed

   port BODOFF4 note{Bodoff Thought Experiment No. 4 (check!)}
       agg a 0.25 claims sev   4 * expon poisson
       agg b 0.05 claims sev  20 * expon poisson
       agg c 0.05 claims sev 100 * expon poisson

