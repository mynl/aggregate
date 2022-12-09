
Material Not Used
==================



Parameters for the Simple Discrete Traffic Example
--------------------------------------------------

You observe traffic past a junction over sixteen 1-minute intervals.
There are 20 vehicles in total, or 20/16=1.25 per minute on average.
This gives the **frequency distribution** :math:`N`.


================== =================== =====================
Number of vehicles Number of intervals Probability
================== =================== =====================
0                  4                   4/16=0.25
1                  6                   6/16=0.375
2                  4                   4/16=0.25
3                  2                   2/16=0.125
Total intervals    16
================== =================== =====================

Separately, you also observe the number of occupants per vehicle for
sixteen vehicles, giving the **severity distribution** :math:`X`. The
ability to determine frequency and severity using separate data studies
is a strength of the aggregate method.

=================== ================== =====================
Number of occupants Number of vehicles Probability :math:`X`
=================== ================== =====================
1                   10                 10/16=0.625
2                   3                  3/16=0.1875
3                   0                  0
4                   3                  3/16=0.1875
Total vehicles      16
=================== ================== =====================

The average number of occupants per vehicle equals
1.75=(10+6+12)/16=28/16=1.75.

From here we can build an aggregate distribution :math:`A` of the number of occupants per minute. You should work out the entire distribution of :math:`A` by hand in a spreadsheet!

Aggregate Program for the Traffic Example
-----------------------------------------

Here is the DecL program for the traffic example. The program is one line long. Python automatically concatenates strings within parenthesis; it is split for clarity.

.. code:: agg

   a = build('agg Traffic dfreq [0 1 2 3] [4/16 6/16 4/16 2/16] '
             'dsev [1 2 4] [10/16 3/16 3/16]')

-  ``agg`` is a keyword
-  Traffic is a user-selected label
-  ``dfreq`` is a keyword to specify a discrete probability distribution for the frequency (the number of outcomes). It has the form ``[outcomes] [probabilities]``

   -  ``[0 1 2 3]`` are the outcomes
   -  ``[4/16 6/16 4/16 2/16]`` are the probabilities
   -  Commas are optional
   -  Only division arithmetic is supported

-  ``dsev`` is a keyword to specify the severity (the value of each outcome). It has the same form as ``dfreq``.

   -  ``[1 2 4]`` are the outcomes; there can be gaps
   -  The probabilities are ``[10/16 3/16 3/16]``.

Creating and printing the object yields:

.. ipython:: python
    :okwarning:

    from aggregate import build
    a = build('agg Traffic '
             'dfreq [0 1 2 3] [4/16 6/16 4/16 2/16] '
             'dsev [1 2 4] [10/16 3/16 3/16]')
    a



Simple Aggregate Model Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a given year there can be 1, 2 or 3 events. There is a 50% chance of
1 event, 25% chance of 2, and 25% chance of 3. Each event randomly
causes a loss of 5, 10 or 15, each with equal probability.

1. What is the average annual event frequency?
2. What is the average event severity?
3. What are the average losses each year?
4. What is the coefficient of variation of losses for each year?
5. Create a table showing all possible outcomes from the model
6. What is the probability of an annual loss of 5? How can it occur?
7. What is the probability of an annual loss of 10? How can it occur?
8. What is the highest amount of total losses that can occur in one
   year? What is the chances that occurs?



.. ipython:: python
    :okwarning:

    from aggregate import build
    import pandas as pd
    sam = build('agg SAM dfreq [1 2 3] [.5 .25 .25] dsev [5 10 15]')
    sam.plot()
    @savefig student_sam.png
    print(sam)
    sam.density_df.query('p_total > 0')[['p_total', 'p_sev']]


The largest outcome of 45 has probability 0.25 * (1/3)**3 (1/4) one for count, three outcomes of 50); check accuracy:

.. ipython:: python
    :okwarning:

    a, e = (1/4) * (1/3)**3, sam.pmf(45)
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'], columns=['value'])

It is

A More Complex Aggregate Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a given year there can be 1, 2, 3 or 20 events. There is a 45% chance
of 1 event, 25% chance of 2, 25% chance of 3, and 5% chance of 100
events. Each event randomly causes a loss of 5, 10 or 50, each with
equal probability.

1. What is the average annual event frequency?
2. What are the average losses each year?
3. What is the coefficient of variation of losses for each year?
4. What are the probabilities of each possible outcome?
5. What are the 99 and 99.6 percentiles of aggregate losses?
6. What is the probability of a maximum loss of 1000?

.. ipython:: python
    :okwarning:

    cam = build('agg CAM dfreq [1 2 3 20] [.45 .25 .25 0.05] '
                'dsev [5 10 50] [1/3 1/3 1/3]', log2=11, bs=1)
    cam.plot()
    @savefig student_cam.png
    print(cam)

    # percentiles
    cam.q(0.99), cam.q(0.996), cam.cdf(570)


The largest outcome of 1000 has probability 0.05 * (1/3)**20 (1/4 one for count, three outcomes of 50); check accuracy:

.. ipython:: python
    :okwarning:

    a, e = 0.05 * (1/3)**20, cam.pmf(1000)
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'],
        columns=['value'])

Finally, show the density.

.. ipython:: python
    :okwarning:

    cam.density_df.query('p_total > 0')[['p_total', 'p_sev', 'F', 'S']]



A Dice Roll of Dice Rolls
---------------------------

The aggregate program for one dice roll.

.. ipython:: python
    :okwarning:

    one_dice = build('agg OneDice dfreq [1] dsev [1:6]')
    one_dice.plot()
    @savefig student_onedice.png
    print(one_dice)

The program for two dice rolls, yielding the triangular distribution.

.. ipython:: python
    :okwarning:

    two_dice = build('agg TwoDice dfreq [2] dsev [1:6]')
    two_dice.plot()
    @savefig student_twodice.png
    print(two_dice)
    print(two_dice.density_df.query('p_total > 0')[['loss', 'p_total', 'F']])

The aggregate program  for twelve dice rolls, which is much harder to do by hand! The answer is compared to a moment-matched normal approximation.

.. ipython:: python
    :okwarning:

    import numpy as np
    twelve_dice = build('agg TwelveDice dfreq [12] dsev [1:6]')
    print(twelve_dice)

    fz = twelve_dice.approximate('norm')
    # model dataframe and append normal approx
    df = twelve_dice.density_df[['loss', 'p_total']]
    df['normal'] = np.diff(fz.cdf(df.loss + 0.5), prepend=0)
    print(df) # .iloc[32:52])
    df.drop(columns=['loss']).plot(xlim=[22, 64])
    @savefig student_norm12.png
    pass


Finally, a dice roll of dice rolls: throw a dice, then throw that many die.

.. ipython:: python
    :okwarning:

    dice2 = build('agg Dice2 dfreq [1:6] dsev [1:6]')
    dice2.plot()
    @savefig student_rollroll.png
    dice2


The largest  outcome of 36 has probability 6**-7; check accuracy

.. ipython:: python
    :okwarning:

    a, e = (1/6)**7, dice2.density_df.loc[36, 'p_total']
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'],
        columns=['value'])

Create the same distribution without shorthand notation and using more basic DecL.

.. code:: ipython3

    dice21 = build('agg Dice2b dfreq [1 2 3 4 5 6]  [1/6 1/6 1/6 1/6 1/6 1/6] '
                   ' sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6]')




Main Features
-------------

- Human readable input with the simple DecL
- Built in library of insurance severity curves for both catastrophe and non
  catastrophe lines
- Built in parameterization for most major lines of insurance in the US, making it
  easy to build a "toy company" based on market share by line
- Clear distinction between catastrophe and non-catastrohpe lines
- Use of Fast Fourier Transforms throughout differentiates ``aggregate`` from
  tools based on simulation
- Fast, accurate - no simulations!
- Graphics and summaries using ``pandas`` and ``matplotlib``
- Outputs in ``pandas`` dataframes



Potential Applications
----------------------

- Education

    * Building intuition around how loss distribtions convolve
    * Convergence to the central limit theorem
    * Generalized distributions
    * Compound Poisson distributions
    * Mixed distributiuons
    * Tail behavior based on frequency or severity tail
    * Log concavity properties
    * Uniform, triangular to normal
    * Bernoulli to normal = life insurance
    * $P(A>x)\sim \lambda P(X>x) \sim P(M>x)$ if thick tails
    * Occ vs agg PMLs, body vs. tail. For non-cat lines it is all about correlation; for cat it is all about the tail
    * Effron's theorem
    * FFT exact for "making" Poisson, sum of normals is normal, expnentials is gamma etc.
    * Slow convergence of truncated stable to normal
    * Severity doesn't matter: difference between agg with sev and without for large claim count and stable severity
    * Small large claim split approach...attrit for small; handling without correlation??
    * Compound Poisson: CP(mixed sev) = sum CP(sev0

- Pricing small insurance portfolios on a claim by claim basis
- Analysis of default probabilities
- Allocation of capital and risk charges
- Detailed creation of marginal loss distributions that can then be sampled and used by other simulation software, e.g. to incorporate dependence structures, or in situations where it is necessary to track individual events, e.g. to compute gross, ceded and net bi- and trivariate distributions.




Aggregate and Portfolio object rationalization
----------------------------------------------

Common Features in Aggregate and Portfolio classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Discrete Aggregate Distributions
---------------------------------------

Aggregate Frequency and Severity Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulation algorithm for insurance losses

::

       for i = 1 to 10000
           agg = 0
           simulate number of events n
           for j = 1 to n
               simulate loss amount X
               agg = agg + X
           output agg for simulation i

-  Write :math:`A = X_1 + \cdots X_N`, :math:`X_i` and :math:`N` random and independent, and :math:`X_i` identically distributed
-  Model insured losses via number of claims :math:`N` the **frequency** and the amount :math:`X_i` of each claim, the **severity**


Notes from Portfolio helpstring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add all the enhanced exhibits methods to port.

Methods defined within this function.

From common_scripts.py
June 2022 took out options that needed a jinja template (reserve story etc.)

Added Methods

*Exhibit creators (EX_name)*

    1. DROPPED basic_loss_statistics
    2. DROPPED distortion_information
    3. DROPPED distortion_calibration
    4. premium_capital
    5. multi_premium_capital
    6. accounting_economic_balance_sheet
       compares best estimate, default adjusted, risk adjusted values

    Exhibits 7-9 are for reserving
    DROPPED 7. margin_earned (by year)
    DROPPED 8. year_end_option_analysis (implied stand alone vs pooled analysis)
    Run a distortion and compare allocations

    9. DROPPED compare_allocations creates:

            EX_natural_allocation_summary
            EX_allocated_capital_comparison
            EX_margin_comparison
            EX_return_on_allocated_capital_comparison

*Exhibit Utilities*

    10. make_all
        runs all of 1-9 with sensible defaults
    11. show_enhaned_exhibits
        shows all exhibits, with exhibit title
        uses `self.dir` to find all attributes EX\_
    12. DROPPED qi
        quick info: the basic_loss_stats plus a density plot

*Graphics*

    13. DROPPED 13. density_plot
    14. profit_segment_plot
        plots given lines S, gS and shades the profit segment between the two
        lines plotted on a stand-alone basis; optional transs allows shifting up/down
    15. natural_profit_segment_plot
        plot kappa = EX_i|X against F and gF
        compares the natural allocation with stand-alone pricing
    16. DROPPED 16. alpha_beta_four_plot
        alpha, beta; layer and cumulative margin plots
    17. DROPPED 17. alpha_beta_four_plot2 (for two line portfolios)
        lee and not lee orientations (lee orientation hard to parse)
        S, aS; gS, b gS separately by line
        S, aS, gS, bGS  for each line [these are the most useful plots]
    18. biv_contour_plot
        bivariate plot of marginals with some x+y=constant lines
    19. DROPPED 19. reserve_story_md

*Reserve Template Populators*

    20. DROPPED 20. reserve_runoff_md
    21. DROPPED 21. reserve_two_step_md
    22. nice_program

*Other*

    23. DROPPED 23. show_md
    24. DROPPED 24. report_args
    25. DROPPED 25. save
    26. density_sample: stratified sample from density_df

**Sample Runner** ::

    from common_header import *
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
    import common_scripts as cs

    port = cs.TensePortfolio('''
    port CAS
        agg Thick 5000 loss 100 x 0 sev lognorm 10 cv 20 mixed sig 0.35 0.6
        agg Thin 5000 loss 100 x 0 sev lognorm 10 cv 20 poisson
    ''', dist_name='wang', a=20000, ROE=0.1, log2=16, bs=1, padding=2)

    # port.make_all() will update all exhibits with sensible defaults

    port.premium_capital(a=20000)
    display(port.EX_premium_capital)

    port.multi_premium_capital(As=[15000, 20000, 25000])
    display(port.EX_multi_premium_capital)

    port.accounting_economic_balance_sheet(a=20000)
    display(port.EX_accounting_economic_balance_sheets)

    port.show_enhanced_exhibits()

    port.density_plot(f, ax0, ax1, p=0.999999)

    port.profit_segment_plot(ax, 0.999, ['total', 'Thick', 'Thin'],
                                 [2,0,1,0], [0,0,0], 'ph')

    port.natural_profit_segment_plot(ax, 0.999, ['total', 'Thick', 'Thin'],
                                 [2,0,1,0], [0,0,0])

    port.profit_segment_plot(ax, 0.999, ['Thick', 'Thin'],
                                 [3,4], [0,0], 'wang')

    aug_df = port.augmented_df
    f, axs = smfig(1,2, (10,5), sharey=True)
    a1, a2 = axs.flat
    bigx = 20000
    bit = aug_df.loc[0:, :].filter(regex='exeqa_(T|t)').copy()
    bit.loc[bit.exeqa_Thick==0, ['exeqa_Thick', 'exeqa_Thin']] = np.nan
    bit.rename(columns=port.renamer).sort_index(1).plot(ax=a1)
    a1.set(xlim=[0,bigx], ylim=[0,bigx], xlabel='Total Loss', ylabel="Conditional Line Loss");
    a1.set(aspect='equal', title='Conditional Expectations\\nBy Line')
    port.biv_contour_plot(f, a2, 5, bigx, 100, log=False, cmap='viridis_r', min_density=1e-12)


Editing test_suite.agg
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    p = Path.home() / 's/telos/python/aggregate_project/aggregate/agg/test_suite.agg'
    txt = p.read_text()
    stxt = txt.split('\n')
    def f(x):
     if x:
         m = x.group(0)
         letter = m[1]
         newletter = chr(ord(letter) + 1)
     return f' {newletter}.'
    ans = []
    for l in stxt:
     if len(l)==0 or l[0] == '#':
         ans.append(l)
     else:
         ans.append(re.sub(r' ([A-Z])\.', f, l))
    ans = '\n'.join(ans)
    p.write_text(ans)


Tweedie conniptions

::

     Tweedies with mu=1, p=1.005 and sigma2=0.1, which is close to Poisson

     from aggregate import tweedie_convert
     # three reps, starting with the most interpretable
     p = 1.005
     μ = 1
     σ2 = 0.1
     m0 = tweedie_convert(p=p, μ=μ, σ2=σ2)

     # magic numbers are
     λ = μ**(2-p) / ((2-p) * σ2)
     α = (2 - p) / (p - 1)
     β = μ / (λ * α)
     tw_cv = σ2**.5 * μ**(p/2-1)
     sev_m = α *  β
     sev_cv = α**-0.5

     m1 = tweedie_convert(λ=λ, m=sev_m, cv=sev_cv)
     m2 = tweedie_convert(λ=λ, α=α, β=β)
     assert np.allclose(m0, m1, m2)
     pd.concat((m0, m1, m2), axis=1)
     program = f'''
     agg Tw0 {λ} claims sev gamma {sev_m:.8g} cv {sev_cv} poisson
     agg Tw1 {λ} claims sev {β:.4g} * gamma {α:.4g} poisson
     agg Tw1 tweedie {μ} {p} {σ2}
     '''
     print(program)
     tweedies = build(program)

     for a in tweedies:
         a.object.plot()
         plt.gcf().suptitle(a.program)

Older Examples from Test_suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ## OLDER EXAMPLES
    # was site.agg
    # sev directorsA lognorm 10000000 cv 1.25 note{directors and officers liability class a, sigma=1.25}
    # sev directorsB lognorm 10000000 cv 1.75 note{directors and officers liability class b, sigma=1.75}
    # sev directorsC lognorm 10000000 cv 2.0  note{directors and officers liability class c, sigma=2.00}
    # sev cata pareto 2.1 -1  note{small cat, pareto 2.1}
    # sev catb pareto 1.7 -1  note{moderate cat, pareto 1.7}
    # sev catc pareto 1.4 -1  note{severe cat, pareto 1.4}
    # sev catd pareto 1.1 -1  note{very severe cat, pareto 1.1}
    # sev cate pareto 0.9 -1  note{extreme cat, pareto 0.9}
    # sev liaba lognorm 50 cv 1.0 note{prems ops A, lognormal cv=1.0}
    # sev liabb lognorm 24 cv 1.5 note{prems ops B, lognormal cv=1.5}
    # sev liabc lognorm 50 cv 2.0 note{prems ops C, lognormal cv=2.0}

    # agg Cata                 3.7         claims            sev pareto 2.1 -1          poisson             note{tropical storms and up US wind events}
    # agg Catb                 1.7         claims            sev pareto 1.7 -1          poisson             note{category 1 and up US wind events}
    # agg Catc                 1.3         claims            sev pareto 1.4 -1          poisson             note{category 3 and up US wind events}
    # agg Catd                 0.4         claims            sev pareto 1.1 -1          poisson             note{category 4 and up US wind events}
    # agg Cate                 0.1         claims            sev pareto 0.9 -1          poisson             note{category 5 and up US wind events}
    # agg Scs                 25.0         claims  2e9 xs 0  sev lognorm 100e6 cv 1.5   poisson             note{industry severe convective storm losses}
    # agg Casxol               0.5         claims 100e6 xs 0 sev lognorm   50e6 cv 0.75 poisson             note{Bermuda like casualty excess of loss book, 0.5 claims}
    # agg Noncata        1000000.00        claims            sev lognorm  50000 cv 1.0  mixed gamma 0.175   note{industry total non cat losses, all lines}
    # agg CAL             462316.42        claims            sev lognorm  40000 cv 0.5  mixed gamma 0.240   note{US statutory industry commercial auto liability, SNL 2017}
    # agg CMP             268153.90        claims            sev lognorm 100000 cv 0.5  mixed gamma 0.280   note{US statutory industry commercial multiperil (property and liability), SNL 2017}
    # agg CommProp        65087.40         claims            sev lognorm 250000 cv 1.25 mixed gamma 0.250   note{US statutory industry commercial property (fire + allied lines), SNL 2017}
    # agg Homeowners     4337346.31        claims 2500 xs 0  sev lognorm     15 cv 0.5  mixed gamma 0.240   note{US statutory industry homeowners, SNL 2017}
    # agg InlandMarine   314117.40         claims            sev lognorm  50000 cv 0.5  mixed gamma 0.350   note{US statutory industry indland marine, SNL 2017}
    # agg PPAL           5676073.30        claims            sev lognorm   5000 cv 10.0 mixed gamma 0.080   note{US statutory industry private passenger auto liability, SNL 2017}
    # agg WorkComp      2664340.53         claims            sev lognorm  15000 cv 7.0  mixed gamma 0.190   note{US statutory industry workers compensation (excluding excess), SNL 2017}
    # agg PersAuto      5676073.30         claims 3e6 xs 0   sev lognorm  50000 cv 7.0  mixed gamma 0.080   note{US statutory personal auto liability and physical damage, SNL 2017}
    # agg CommAuto      5676073.30         claims            sev lognorm  30000 cv 7.0  mixed gamma 0.080   note{US statutory personal auto liability and physical damage, SNL 2017}
