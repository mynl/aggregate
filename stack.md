# TODO stack

Examples introduced!

All references! REF!

Cat: put in changes to climate change el impact! (in progress)
Then reinsurance and 10 mins...

**Details.**

`np.newaxis`, x=x[:, np.newaxis]

Comments then code!!

agg written in one line stand-alone and broken up in build MAKES SURE IT IS PASTEABLE!

One-line it only for graphics?

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


* Extract Python from rst
* reins summaries in re section? [rather than by hand]; describe in DecL section
* scale all graph plots! what are standards?
* constants module in agg?


0. Add exposure / rate keywords for 10 exposure at 1200 rate ? Is that in docs?
0. Uncertainty principle for bs: too small not enough space; too large miss features of sev; for given n there is a min possible error
1.
1. qd looks for like columns / column format guesser?
2. poisson poisson distribution  = neyman type a
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
