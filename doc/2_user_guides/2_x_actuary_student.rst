.. _2_x_actuary_student:

.. reviewed 2022-11-10
.. reviewed 2022-12-24

Actuarial Student
====================

**Objectives:** Introduce the ``aggregate`` library for working with aggregate probability distributions in the context of actuarial society exams and university courses in (short-term) actuarial modeling.

**Audience:** Actuarial science university students and actuarial analysts.

**Prerequisites:** Familiarity with aggregate probability distributions as covered on SOA STAM, CAS MAS I, or IFOA CS-2, and basic insurance terminology from insurance company operations.

**See also:** :doc:`2_x_student` for a more basic introduction; :doc:`../2_User_Guides` for other applications.

**Contents:**

#. :ref:`Realistic Insurance Example`
#. :ref:`College and Exam Questions`
#. :ref:`Advantages of Modeling with Aggregate Distributions`
#. :ref:`actuary summary`

Realistic Insurance Example
---------------------------

**Assumptions.**
You are given the following information about a book of liability
insurance business.

1. Premium equals 2000 and the expected loss ratio equals 67.5%.
2. Ground-up severity has been fit to a lognormal distribution with a mean of 50 and CV (coefficient of variation) of 1.25.
3. All policies have a limit of 1000 with no deductible or retention.
4. Frequency is modeled using a Poisson distribution.

You model aggregate losses using the collective risk model.

**Questions.**

1. Compute the expected insured severity and expected claim count.
2. Compute the aggregate expected value, standard deviation, CV, and skewness.
3. Compute:

   1. The probability aggregate losses exceed the premium.
   2. The probability aggregate losses exceed 2500
   3. The expected value of aggregate losses limited to 2500
   4. The expected policyholder deficit in excess of 2500

**Answers.**

Build an aggregate object using simple DecL program.
The dataframe ``a01.describe`` gives the answers to questions 1 and 2. It printed and formatted automatically by ``qd(a01)``.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

    a01 = build('agg Actuary:01 '
                '2000 premium at 0.675 lr '
                '1000 xs 0 '
                'sev lognorm 50 cv 1.25 '
                'poisson'
                , bs=1/8)
    qd(a01)

The survival function ``a01.sf`` answers 3.1 and 3.2. ``qd`` is used to print with reasonable defaults. The dataframe ``a01.density_df`` computes limited expected values (levs) and expected policyholder deficit indexed by loss level, and other values. Querying it answers 3.3.

.. ipython:: python
    :okwarning:

    qd(a01.sf(2000), a01.sf(2500))
    qd(a01.density_df.loc[[2500], ['F', 'S', 'lev', 'epd']])


..  # other things to consider
    xs = a01.density_df.loc[2500, ['S', 'exgta']]
    xs = xs.prod()
    xxs = xs - 2500 * a01.density_df.loc[2500, 'S']
    lev = a01.density_df.loc[2500, 'lev']
    xs, a01.est_m - lev, xxs, xxs/a01.est_m, a01.density_df.loc[2500, 'epd']


College and Exam Questions
---------------------------

College courses and the early actuarial exams often ask purely technical questions. Using assumptions from the :ref:`Realistic Insurance Example` answer the following.

1. Compute the severity lognormal parameters mu and sigma.
2. Compute the expected insured severity and expected claim count.
3. Compute the probability the aggregate exceeds the premium using the following matched moment approximations:

   1. Normal
   2. Gamma
   3. Lognormal
   4. Shifted gamma
   5. Shifted lognormal

4. Using the ``aggregate`` and a lognormal approximation, compute:

   1. The probability losses exceed 2500
   2. The expected value of losses limited to 2500
   3. The expected value of losses in excess of 2500

The code below provides all the answers. ``mu_sigma_from_mean_cv`` computes the lognormal parameters---one of the most written macro in actuarial science! Start by applying it to the given severity parameters to answer question 1.

.. ipython:: python
    :okwarning:

    from aggregate import mu_sigma_from_mean_cv
    import pandas as pd

    print(mu_sigma_from_mean_cv(50, 1.25))

The function ``a01.approximate`` parameterizes all the requested matched moment approximations, returning frozen ``scipy.stats`` distribution objects that expose ``cdf`` methods. The :class:`Aggregate` class object ``a`` also has a ``cdf`` method. Using these functions, we can assemble a dataframe to answer question 3.

.. ipython:: python
    :okwarning:

    fz = a01.approximate('all')
    fz['agg'] = a01

    df = pd.DataFrame({k: v.sf(2000) for k, v in fz.items()}.items(),
                 columns=['Approximation', 'Value']
                ).set_index("Approximation")
    df['Error'] = df.Value / df.loc['agg', 'Value'] - 1
    qd(df.sort_values('Value'))

The function ``lognorm_lev`` computes limited expected values for the lognormal. It is used to assemble a dataframe to answer question 4.

.. ipython:: python
    :okwarning:

    from aggregate import lognorm_lev

    mu, sigma = mu_sigma_from_mean_cv(a01.agg_m, a01.agg_cv)
    lev = lognorm_lev(mu, sigma, 1, 2500)
    lev_agg = a01.density_df.loc[2500, 'lev']
    default = a01.agg_m - lev
    epd = default / a01.est_m
    default_agg = a01.est_m - lev_agg
    bit = pd.DataFrame((lev, default, lev_agg, default_agg, epd, default_agg / a01.agg_m),
                 index=pd.Index(['Lognorm LEV', 'Lognorm Default', 'Agg LEV',
                 'Agg Default', 'Lognorm EPD', 'Agg EPD'],
                 name='Item'),
                 columns=['Value'])
    qd(bit)



Advantages of Modeling with Aggregate Distributions
------------------------------------------------------

Aggregate distributions provide a powerful modeling paradigm. It separates the analysis of frequency and severity. Different datasets can be used for each. KPW list seven advantages.

1. Only the expected claim count changes with volume. The severity distribution is a characteristic of the line of business.

2. Inflation impacts ground-up severity but not claim count. The situation is more complicated when limits and deductibles apply.

3. Coverage terms impact occurrence limits and deductibles, which affect ground-up severity.

4. The impact on claims frequencies of changing deductibles is better understood.

5. Severity curves can be estimated from homogeneous data. Kaplan-Meier and related methods can adjust for censoring and truncation caused by limits and deductibles.

6. Retained, insured, ceded, and net losses can be modeled consistently.

7. Understanding properties of frequency and severity separately illuminates the shape of the aggregate.

.. _actuary summary:

Summary of Objects Created by DecL
-------------------------------------

Objects created by :meth:`build` in this guide.

.. ipython:: python
    :okwarning:
    :okexcept:

    from aggregate import pprint_ex
    for n, r in build.qshow('^Actuary:').iterrows():
        pprint_ex(r.program, split=20)


.. ipython:: python
    :suppress:

    plt.close('all')
