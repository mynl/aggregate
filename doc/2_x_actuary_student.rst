.. _2_x_actuary_student:

Actuarial Student
===========================================


**Objectives** Introduce aggregate probability distributions and the `aggregate` library for working with them in the context of exam and university courses in actuarial modeling.

**Audience** Actuarial science university student or junior analyst working in insurance.

**Prerequisites** V01 plus familiarity with aggregate probability distribution (as covered on SOA STAM, CAS MAS I, IFOA CS-2) and basic insurance terminology (insurance company operations).



Realistic Insurance Example
---------------------------

Assumptions
~~~~~~~~~~~

You are given the following information about a book of liability
insurance business.

1. Premium equals ¤2000 and the expected loss ratio equals 67.5%.
2. Ground-up severity has been fit to a lognormal distribution with a
   mean of ¤50 and a CV (coefficient of variation) of 1.25.
3. All policies have a limit of ¤1000 and no deductible or retention.
4. Frequency is modeled using a Poisson distribution.

You model aggregate losses :math:`X` using the collective risk model.

Questions
~~~~~~~~~

1. Compute the expected insured severity and expected claim count.
2. Compute the expected value, standard deviation, CV, and skewness of
   :math:`X`.
3. Compute the probability :math:`X` exceeeds the premium.
4. For :math:`X`, compute:

   1. The probability losses exceed ¤2500
   2. The expected value of lossses limited to ¤2500
   3. The expected value of losses in excess of ¤2500

::

    a = build('agg InsuranceExample '
          '2000 premium at 0.675 lr 1000 xs 0 '
          'sev lognorm 50 cv 1.25 '
          'poisson')
    print(a)

    a.sf(2000), a.sf(2500)

    # lev and epd ratio
    a.density_df.loc[[2500]]

    # epd in  ¤
    default_agg = a.agg_m - a.density_df.loc[2500, 'lev']
    default_agg


Questions (academic version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Compute the severity lognormal mu and sigma.
2. Compute the expected insured severity and expected claim count.
3. Compute the expected value, standard deviation, CV, and skewness of
   :math:`X`.
4. Compute the probability :math:`X` exceeds the premium using the
   following matched-moment approximations:

   1. normal
   2. gamma
   3. lognormal
   4. shifted gamma
   5. shifted lognormal

5. Compute the probability :math:`X` exceeeds the premium.
6. Using the :math:`X` and a lognormal approximation, compute:

   1. The probability losses exceed ¤2500
   2. The expected value of lossses limited to ¤2500
   3. The expected value of losses in excess of ¤2500


::

    from aggregate import mu_sigma_from_mean_cv
    mu_sigma_from_mean_cv(50, 1.25)

    fz = a.approximate('all')
    fz['agg'] = a

    df = pd.DataFrame({k: v.sf(2000) for k, v in fz.items()}.items(),
                 columns=['Approximation', 'Value']
                ).set_index("Approximation")
    df['Error'] = df.Value / df.loc['agg', 'Value'] - 1
    df.sort_values('Value')
    # .style.format(lambda x: f'{x:.2%}')

    from aggregate import lognorm_lev

    mu, sigma = mu_sigma_from_mean_cv(a.agg_m, a.agg_cv)
    lev = lognorm_lev(mu, sigma, 1, 2500)
    default = a.agg_m - lev
    epd = default / a.agg_m
    pd.DataFrame((lev, default, default_agg, epd, default_agg / a.agg_m),
                 index=pd.Index(['Lognorm LEV', 'Lognorm Default', 'Agg Default', 'Lognorm EPD', 'Agg EPD'], name='Item'),
                 columns=['Value']).style.format(lambda x: f'{x:.3f}')
