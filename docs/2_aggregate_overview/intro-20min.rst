.. _intro 20min:

``aggregate`` in Twenty Minutes
=================================

This section is a working session. Type along and you will leave with the vocabulary to model your own book: severity and frequency families, limits and layers, reinsurance, validation, portfolios, and a first capital-loaded price. It ends with the exhibit that justifies the whole approach, a head-to-head against Monte Carlo. It assumes Python and pandas basics and the ideas of frequency and severity. No knowledge of the FFT is needed.

.. _intro20 audience:

Who this is for
---------------

Pricing actuaries
    the distribution behind every increased-limits factor and aggregate feature.
Reinsurance analysts
    occurrence and aggregate programs as language primitives, with per-layer statistics built in.
Capital and ERM actuaries
    economic capital is a quantile of an aggregate distribution; here is the distribution.
Cat modelers
    frequency mixing (common shock), heavy tails, occurrence structures.
Operational-risk quants
    the loss distribution approach is a compound frequency by severity model read at the 99.9th percentile, exactly where simulation is weakest.
Students and researchers
    the collective risk model of the textbooks and the ASTIN literature, runnable and checkable.

.. _intro20 collective risk:

The collective risk model, and how ``aggregate`` computes it
------------------------------------------------------------

Total losses are :math:`S = X_1 + \cdots + X_N`: a random count :math:`N` of claims, each of random size :math:`X`, the collective risk model of the textbooks :cite:p:`Klugman2012`. The distribution of :math:`S` has no closed form outside special cases, but its Fourier transform does: discretize the severity, transform it, apply the frequency's probability generating function, and invert, a lineage running through actuarial science from :cite:t:`Heckman1983` to :cite:t:`Grubel1999`. ``aggregate`` automates that pipeline :cite:p:`Mildenhall2024`, including the unglamorous parts that make it trustworthy: choosing the discretization grid, validating moments, and warning when the answer cannot be represented well.

.. ipython:: python

    from time import perf_counter
    from aggregate import build, qd

    t0 = perf_counter()
    a = build('''
        agg Casualty
            250 claims
            1000 xs 0
            sev lognorm 100 cv 1.5
            poisson
    ''')
    print(f'built in {perf_counter() - t0:.3f}s')
    qd(a)

The result is the entire distribution: every quantile, every tail measure, deterministically, in milliseconds. Hold that build time for :ref:`intro20 head to head`.

.. _intro20 anatomy:

Anatomy of a DecL program
-------------------------

Models are written in DecL, a small declarative language. The organizing idea is that exposure is a pair, volume and coverage: how much risk, and on what terms. Volume is a claim count, an expected loss, or, as a convenience, premium times a loss ratio. Coverage is a per-occurrence limit and attachment. Around that pair sit the severity, the frequency, and the optional structure. A program can be written on one line, but anything real is clearer split across indented continuation lines:

.. code-block:: text

    agg Casualty                        keyword and name
        250 claims                      volume: count, loss, or premium at a
                                        loss ratio
        1000 xs 0                       coverage: limit xs attachment
        sev lognorm 100 cv 1.5          severity: the size-of-loss curve
        occurrence net of 250 xs 750    per-occurrence reinsurance (optional)
        poisson                         frequency
        aggregate net of 5000 xs 30000  annual reinsurance (optional)

A trailing ``approximate sgamma`` clause, which is optional and cannot be combined with occurrence reinsurance, swaps the exact convolution for a moment-matched fit when a very-high-frequency book does not need it. The whole statement above is real syntax:

.. ipython:: python

    full = build('''
        agg Anatomy
            250 claims
            1000 xs 0
            sev lognorm 100 cv 1.5
            occurrence net of 250 xs 750
            poisson
            aggregate net of 5000 xs 30000
    ''')
    qd(full.summary_df)

The clauses in detail:

exposure, volume
    ``250 claims`` (expected count), ``25000 loss`` (expected loss), or ``40000 prem at 0.65 lr`` (premium times loss ratio, a convenience for how business actually arrives).
exposure, coverage
    ``1000 xs 0``, a per-occurrence limit and attachment. Omit it for ground-up and unlimited.
severity
    any ``scipy.stats`` family, parameterized the way actuaries think, by mean and CV. See :ref:`intro20 severity`.
frequency
    ``poisson``, or a mixed or contagion family. See :ref:`intro20 frequency`.
reinsurance
    occurrence (per claim, stated before the frequency) and aggregate (per year, after it), both optional. See :ref:`intro20 reinsurance`.

The volume and coverage pair is not just pedagogy. It is the constructor's own vocabulary: :class:`Aggregate` takes ``exp_en``, ``exp_el``, ``exp_premium`` and ``exp_lr`` for volume, and ``exp_limit`` and ``exp_attachment`` for coverage.

A model you can read is a model you can review, version, email, and store. The program is data: ``a.program`` returns it, and an :class:`Underwriter` object persists whole libraries of named risks.

.. _intro20 severity:

Severity: a vocabulary, not a menu
----------------------------------

Severity takes any continuous ``scipy.stats`` family by mean and CV, plus shifting and scaling, mixtures, and empirical (discrete) data. Tail shape is a modeling choice, and it is cheap to see its consequences. Here are two books with the same mean and the same CV, differing only in family:

.. ipython:: python

    import matplotlib.pyplot as plt

    ln = build('''
        agg LN
            50 claims
            sev lognorm 100 cv 2
            poisson
    ''')
    gm = build('''
        agg GM
            50 claims
            sev gamma 100 cv 2
            poisson
    ''')

    fig, ax = plt.subplots()
    ln.density.p_total.plot(ax=ax, logy=True, label='lognormal severity')
    gm.density.p_total.plot(ax=ax, logy=True, label='gamma severity')
    ax.set(xlim=(0, 40_000), title='Same mean and CV, different tails',
           xlabel='total loss', ylabel='density (log scale)')
    @savefig intro20_tails.png scale=20
    ax.legend();

.. ipython:: python

    print(f"99.9% VaR, lognormal sev: {ln.q(0.999):10,.0f}")
    print(f"99.9% VaR, gamma sev:     {gm.q(0.999):10,.0f}")

Mixtures express heterogeneous books, here three lognormal components with weights:

.. ipython:: python

    mix = build('''
        agg Mix
            100 claims
            2000 xs 0
            sev lognorm [50 100 200] cv [1 1.5 2] wts [.5 .3 .2]
            poisson
    ''')
    qd(mix.summary_df)

Empirical severities come straight from data as discrete atoms, in units of thousands here:

.. ipython:: python

    emp = build('''
        agg FromData
            25 claims
            dsev [2 5 12 18 45] [.30 .30 .20 .15 .05]
            poisson
    ''')
    qd(emp)

.. _intro20 frequency:

Frequency: Poisson is an opinion
--------------------------------

Claim counts are usually over-dispersed: years differ for reasons shared across claims, such as weather, inflation, and the legal environment. The clause ``mixed gamma 0.4`` puts a gamma-distributed multiplier with CV 0.4 on the Poisson intensity, giving a negative binomial by another road. The effect on required capital is not subtle:

.. ipython:: python

    po = build('''
        agg Po
            100 claims
            1000 xs 0
            sev lognorm 100 cv 1.5
            poisson
    ''')
    ng = build('''
        agg Ng
            100 claims
            1000 xs 0
            sev lognorm 100 cv 1.5
            mixed gamma 0.4
    ''')
    print(f"           {'Poisson':>12s} {'mixed gamma':>12s}")
    print(f"agg CV     {po.est_cv:12.3f} {ng.est_cv:12.3f}")
    print(f"99.5% VaR  {po.q(0.995):12,.0f} {ng.q(0.995):12,.0f}")

The mixing carries through everything downstream: layers, reinsurance, and portfolio dependence, since units sharing a mixed frequency are correlated, the common-shock effect cat modelers expect.

.. _intro20 suva:

Trust, but verify: the SUVA-use workflow
----------------------------------------

The recommended working cycle is specify, update, validate, adjust, use, the SUVA-use workflow of :cite:t:`Mildenhall2024`:

Specify
    the gross compound using DecL.
Update
    the numerical approximation using the ``update`` method, performed automatically for objects created using DecL and ``build``.
Validate
    that the results are "not unreasonable" by reviewing the diagnostics; if necessary, adjust the calculation parameters and re-run ``update``.
Adjust
    the specification for reinsurance and update using the same parameters.
Use
    the output.

The order matters. Validation compares theoretical moments, computed exactly from the specification, against estimated ones, computed from the FFT output, and that comparison only exists on the gross object: once reinsurance is applied there is no closed form left to check against. Validating gross and then reinsuring on the same grid is what makes the net and ceded distributions trustworthy.

For discrete inputs the method is exact, so the error columns are zero:

.. ipython:: python

    dice = build('''
        agg Dice
            dfreq [3]
            dsev [1:6]
    ''')
    qd(dice.summary_df)

For continuous inputs the gap is discretization error, which ``aggregate`` manages by choosing a bucket size ``bs`` and grid size ``2**log2`` for you, and reporting what it chose in ``a.info``. Override with ``build(..., bs=..., log2=...)``. The choice is described in :ref:`num bucket selection`.

When something cannot be handled silently, the library refuses to guess. An infinite-variance Pareto severity makes the auto-sizer balk, so the grid must be supplied, here through an in-program ``hints{}`` clause. Even on that grid some tail mass cannot be represented, and the validator says so plainly instead of handing back a silently wrong number:

.. ipython:: python

    heavy = build('''
        agg Heavy
            10 claims
            sev 100 * pareto 1.3 - 100
            poisson
            hints{bs=0.25; log2=16;}
    ''')
    print(heavy.validation_explanation)

.. _intro20 limits:

Limits, attachments, and increased limits
-----------------------------------------

Policy terms are language, not post-processing. Expected losses by limit, and the increased limits factors that follow, come out of a loop:

.. ipython:: python

    base = None
    for lim in [500, 1000, 2000, 5000]:
        al = build(f'''
            agg ILF
                100 claims
                {lim} xs 0
                sev lognorm 100 cv 1.5
                poisson
        ''')
        base = base or al.est_m
        print(f'limit {lim:6,d}   E[S] {al.est_m:10,.0f}   '
              f'ILF {al.est_m / base:.3f}')

The layer caps each occurrence. Features that cap the year are covered in :ref:`intro20 reinsurance`.

.. _intro20 reinsurance:

Reinsurance, gross to net
-------------------------

Occurrence programs, which act per claim, and aggregate programs, which act on the whole year, are both one clause. Shares read naturally: ``50% po 300 xs 200`` is half of the 300 excess 200 layer.

.. ipython:: python

    re = build('''
        agg Book
            10 claims
            1000 xs 0
            sev lognorm 100 cv 2
            occurrence net of 50% po 300 xs 200 and 100% po 500 xs 500
            poisson
    ''')
    qd(re.summary_df)

The ``summary_df`` above switches to an economic view, gross against net, with the cession's impact on each moment. The dedicated reporting goes deeper, per stage and per layer:

.. ipython:: python

    qd(re.reins_summary_df)

An aggregate stop loss is the same idea at the year level:

.. ipython:: python

    stop = build('''
        agg Stop
            100 claims
            1000 xs 0
            sev lognorm 100 cv 1.5
            poisson
            aggregate net of 5000 xs 30000
    ''')
    qd(stop.summary_df)

.. _intro20 head to head:

The head-to-head: FFT against Monte Carlo
-----------------------------------------

Here is the honest comparison. Simulate the ``Casualty`` book from :ref:`intro20 collective risk`, Poisson(250) with lognormal severity of mean 100 and CV 1.5 capped at 1,000, and estimate the 99.9th percentile five times with 20,000 simulations each, which is 5 million severity draws per estimate:

.. ipython:: python

    import numpy as np

    m, cv, lim, lam = 100, 1.5, 1000, 250
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(m) - sigma**2 / 2
    rng = np.random.default_rng(20260611)

    def mc_q999(n_sims=20_000):
        counts = rng.poisson(lam, n_sims)
        sev = np.minimum(rng.lognormal(mu, sigma, counts.sum()), lim)
        totals = np.add.reduceat(sev, np.r_[0, counts.cumsum()[:-1]])
        return np.quantile(totals, 0.999)

    t0 = perf_counter()
    estimates = [mc_q999() for _ in range(5)]
    mc_time = perf_counter() - t0

    t0 = perf_counter()
    exact = build('''
        agg HeadToHead
            250 claims
            1000 xs 0
            sev lognorm 100 cv 1.5
            poisson
    ''').q(0.999)
    fft_time = perf_counter() - t0

    print('Monte Carlo 99.9% VaR, five runs of 20,000 simulations:')
    for e in estimates:
        print(f'   {e:10,.0f}   (error vs FFT {e / exact - 1:+.2%})')
    print(f'\nFFT          {exact:10,.0f}')
    print(f'\nMC time  {mc_time:.2f}s for five noisy answers; '
          f'FFT time {fft_time:.2f}s for one stable one.')

Each Monte Carlo estimate rests on the twenty largest of 20,000 outcomes, so run to run it wobbles by percents, which is material when it sets capital. The FFT answer is identical every run, and refreshing it after a structural change such as a new limit or a new treaty costs the same milliseconds. Simulation still earns its keep for path dependence and exotic dependence; for the workhorse compound distribution, convolution dominates.

.. _intro20 portfolio:

Portfolios, capital, and a first price
--------------------------------------

A ``port`` convolves independent units onto one grid, and frequencies can share a mixing variable for common shock. Diversification is then arithmetic, and capital-loaded pricing is one line:

.. ipython:: python

    port = build('''
        port Book
            agg Property 80 claims  500 xs 0 sev lognorm  50 cv 1.2 poisson
            agg Casualty 20 claims 2000 xs 0 sev lognorm 250 cv 2.0 mixed gamma 0.4
    ''')
    vd = port.var_dict(0.996)
    print(f"99.6% VaR stand-alone sum "
          f"{vd['Property'] + vd['Casualty']:10,.0f}")
    print(f"99.6% VaR portfolio       {vd['total']:10,.0f}")

.. ipython:: python

    # back capital with the 99.6% VaR and target a 10% return on it
    qd(port.price_pentagon(p=0.996, ROE=0.10))

``L`` is expected loss, ``M`` margin, ``P = L + M`` premium, ``Q`` capital and ``a = P + Q`` assets, plus the ratios: the standard accounting identity completed from your two inputs. Beyond this lie distortion (spectral) risk measures, calibrated pricing, and capital allocation across units. The pointers are :meth:`Portfolio.price`, the published documentation, and *Pricing Insurance Risk* :cite:p:`Mildenhall2022a`, whose framework the pricing surface implements.

.. _intro20 pandas:

Everything is a DataFrame
-------------------------

Every table ``aggregate`` produces is a pandas ``DataFrame`` and every plot is a matplotlib figure, so there is no export step. The distribution itself is a frame indexed by loss, and you slice it the way you would slice any other frame:

.. ipython:: python

    a.density_df.loc[20_000:20_020, ['loss', 'p_total', 'F', 'S']]

.. ipython:: python

    @savefig intro20_port.png scale=20
    port.plot()

.. _intro20 next:

Install and next steps
----------------------

.. code-block:: bash

    pip install aggregate

The :doc:`underwriter` guide covers the :class:`Underwriter`, the recipe library, and how a declaration documents and tests itself. The :doc:`../4_dec_Language_Reference` documents the full language, including towers, splices, zero-modified frequencies, and vectorized exposures. For the pricing and capital theory behind :ref:`intro20 portfolio`, see :cite:t:`Mildenhall2022a` and :cite:t:`Major2026`.
