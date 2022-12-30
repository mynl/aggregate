.. _2_x_case_studies:

.. reviewed 2022-12-28

Case Studies
===================

**Objectives:** Using :mod:`aggregate` to reproduce the case study exhibits from the book `Pricing Insurance Risk <https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_ and build similar exhibits for your own cases.

**Audience:** Capital modeling and corporate strategy actuaries; anyone reading PIR.

**Prerequisites:** Intermediate to advanced users with a sold understanding of ``aggregate``. Familiar with PIR.

**See also:** :doc:`2_x_capital`, :doc:`2_x_strategy`.

**Contents:**

#. :ref:`PIR Case Studies`
#. :ref:`Creating a Case Study`
#. :ref:`Case Study Factory Arguments`
#. :ref:`Defining a Custom Case Study`
#. :ref:`Standard Case Study Exhibits`

**Confession:** The ``case_studies`` code is sub-optimal. Some of it is horrendous.

PIR Case Studies
--------------------

The book  `Pricing Insurance Risk
<https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_ (PIR)
presents four Case Studies that show how different methods price business.
This section shows how to reproduce all the book's exhibits for
each case and how to create new cases.

Each case describes business written by Ins Co., a one-period de novo insurer
that comes into existence at time zero, raises capital and writes business,
and pays all losses at time one. A case models two units (line, region,
operating unit, or other division) with one more risky than the other. Usually, the riskier one is reinsured. Case exhibits compare unit statistics and pricing on a gross and net basis, showing results from over a dozen different methods.

PIR includes four Case Studies.

Simple Discrete Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Simple Discrete Example Case Study, :math:`X_1` takes values 0, 8, or 10, and :math:`X_2` values 0, 1, or 90. The units are independent and sum to the portfolio loss :math:`X`. The outcome probabilities are 1/2, 1/4, and 1/4 respectively for each marginal. There are 9 possible outcomes. This type of output is typical of that produced by a catastrophe, capital, or pricing simulation model---albeit much simpler.

Tame Case Study
~~~~~~~~~~~~~~~~~~~

In the Tame Case Study, Ins Co. writes two predictable units with no
catastrophe exposure. It is an idealized, best-case risk-pool. It proxies a
portfolio of personal and commercial auto liability. It uses a
straightforward stochastic model with gamma distributions.

Aggregate reinsurance applies to the more volatile unit, with an attachment
probability 0.2 (56) and detachment probability 0.01 (69).

Catastrophe and Non-Catastrophe (CNC) Case Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Cat/Non-Cat Case Study, Ins Co. has catastrophe and non-catastrophe
exposures. The non-catastrophe unit proxies a small commercial lines
portfolio. Balancing the relative benefits of units considered to be more
stable against more volatile ones is a very common strategic problem for
insurers and reinsurers. It arises in many different guises:

-  Should a US Midwestern company expand to the East coast
   (and pick up hurricane exposure)?
-  Should an auto insurer start writing homeowners?
-  What is the appropriate mix between property catastrophe
   and non-catastrophe exposed business for a reinsurer?

The two units are independent and have gamma and lognormal distributions.

Aggregate reinsurance applies to the Cat unit, with an attachment probability
0.1(41) and detachment probability 0.005(121).

Hurricane/Severe Convective Storm (HuSCS) Case Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Hu/SCS Case Study, Ins Co. has catastrophe exposures from severe
convective storms (SCS) and, independently, hurricanes (Hu). In practice,
hurricane exposure is modeled using a catastrophe model. The Case proxies
that by using a very severe severity in place of the gross catastrophe model
event-level output. Both units are modeled using an aggregate distribution
with a Poisson frequency and lognormal severity.

Aggregate reinsurance applies to the HU unit with an occurrence attachment
probability 0.05(40) and detachment probability 0.005 (413). See REF for a
version with occurrence reinsurance.


Creating a Case Study
---------------------

Case Study exhibits are managed by the class :class:`CaseStudy` in ``aggregate.extensions.case_studies``, see :ref:`Extensions`. Here are the four steps needed to create a case study. The computations take a few minutes. The output is a set of HTML files that can be viewed in a browser. The code blocks below are provided in executable scripts described below.

1. Import ``case_studies``::

    from aggregate import build, qd
    from aggregate.extensions import case_studies as cs

2. Create a new :class:`CaseStudy` object. It is a generic container, the options are set in the next step::

    my_case = cs.CaseStudy()

3. Set the case study options. Here are the options for the PIR Tame case. The arguments are described later::

    my_case.factory(case_id='my_tame',
                    case_name='My version of PIR Tame Case',
                    case_description='Tame Case to demonstrate capabilities.',
                    a_distribution='agg A 1 claim sev gamma 50 cv 0.10 fixed',
                    b_distribution_gross='agg B 1 claim sev gamma 50 cv 0.15 fixed',
                    b_distribution_net='agg B 1 claim sev gamma 50 cv 0.15 fixed '
                                       'aggregate net of 12.90625 xs 56.171875',
                    reg_p=0.999,
                    roe=0.10,
                    d2tc=0.3,
                    f_discrete=False,
                    s_values=[.005, 0.01, 0.03],
                    gs_values=[0.029126,   0.047619,   0.074074],
                    bs=1/64,
                    log2=16,
                    padding=1)

4. To browse the exhibits execute::

    my_case.browse_exhibits()

  which opens two new browser tabs, one for the standard book exhibits and one for a set of extended exhibits.

**Details.**

Exhibit output files are stored in ``build.case_dir``, which by default is the subdirectory aggregate/cases of your home directory (``~`` on Linux, ``\users\<user name>`` on Windows, and who knows on an Apple). The book exhibits are marshaled in  ``f'{my_case.case_id}_book.html'`` and a set of extended exhibits are in ``f'{my_case.case_id}_extended.html'``. The detailed files are in a subdirectory called ``my_case.case_id``.


Case Study Factory Arguments
-------------------------------

:meth:`cs.CaseStudy().factory` takes the following arguments.

* ``case_id`` is a single word label that uniquely identifies the Case. It determines the output directory for the Case exhibits and so must be acceptable to your operating system as a directory name.

* ``case_name`` such as "Cat/Non-Cat".

* ``case_description`` is a brief description of the Case.

* ``a_distribution``, ``b_distribution_gross`` and ``b_distribution_net`` are DecL programs defining the aggregate distributions for each unit, including reinsurance on unit B. Unit names should be upper case and ideally in alphabetical order, the more volatile unit second. For example ``A`` and ``B``.

* ``reg_p`` gives the regulatory capital standard, entered as a probability of non-exceedance level. Solvency II operates at 0.995 (one in 200 years). In the US, rating agencies consider companies at 0.99 (100 years), 0.996 (250 years), 0.999 (1000 years), or higher. Corporate bond default rates impose even tighter capital standards.

* ``roe`` sets the target cost of capital. All pricing methods are calibrated to produce a return on capital of ``roe`` at the selected capital standard level. This makes them comparable.

* ``d2tc`` is the maximum allowable debt to total capital level (used for enhanced exhibits only). It is used to tranche capital into debt and equity.

* ``f_discrete`` indicates whether the distributions are discrete (Simple Discrete Example) or mixed (all others). Usually ``f_discrete=False``.

* ``s_values`` and ``gs_values`` define cat bond pricing and are used to create a blended distortion, see below.

* ``bs``, ``log2``, and ``padding`` are the usual update parameters.



PIR Case Specifications
------------------------

This section provides the arguments needed to recreate each PIR Case Study. Some of the code is used to determine the details of reinsurance. See PIR Chapter REF for more details and explanation.

The current version of ``aggregate`` uses an improved blend distortion over that shown in PIR. It is calibrated to cat bond pricing for high return periods using the following values:

  - ``s_values: [.005, 0.01, 0.03]``
  - ``gs_values: [0.029126,   0.047619,   0.074074]``

meaning bonds with a 0.5% EL have a discount spread of 2.9%, 1% EL a discount spread of 4.76% and so forth. They define the left-hand (small :math:`s`) end of a distortion function.

All Cases assume that debt to total capital limited at 30%, ``d2tc: 0.3``. This factor is only used in the extended exhibits.


Simple Discrete Example Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two flavors, one with distinct outcomes and one with two ways of obtaining the outcome 10. The latter is used in PIR to illustrate the linear natural allocation. Here are the specifications.

.. literalinclude ../../s/telos/python/aggregate_project/aggregate/extensions/discrete.py
    :language: python


.. literalinclude:: ../../aggregate/extensions/discrete.py
    :language: python




Tame Specification
~~~~~~~~~~~~~~~~~~~~~~

The first few lines calibrate the reinsurance to probability levels.

.. literalinclude ../../s/telos/python/aggregate_project/aggregate/extensions/tame.py
    :language: python


.. literalinclude:: ../../aggregate/extensions/tame.py
    :language: python


Catastrophe and Non-Catastrophe Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first few lines calibrate the reinsurance to probability levels.

.. literalinclude ../../s/telos/python/aggregate_project/aggregate/extensions/cnc.py
    :language: python


.. literalinclude:: ../../aggregate/extensions/cnc.py
    :language: python


Hurricane/Severe Convective Storm Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first few lines set up the parameters from PIR. Then the reinsurance is calibrated to probability levels. The second version has occurrence reinsurance on the Cat line with the same limit and attachment.

.. literalinclude ../../s/telos/python/aggregate_project/aggregate/extensions/hs.py
    :language: python


.. literalinclude:: ../../aggregate/extensions/hs.py
    :language: python



These snippets are provided in Python scripts in ``aggregate.extensions`` called ``discrete.py``, ``tame.py``, ``cnc.py`` and ``hs.py``. They can be run from the command line::

    python -m aggregate.extensions.tame


Precomputed versions are available at :ref:`https://www.pricinginsurancerisk.com/results`.

Defining a Custom Case Study
------------------------------

It should be obvious how to create a custom case study. The key is the DecL
for the two units. Here are ideas for some custom Case Studies illustrating
different behaviors.


Standard Case Study Exhibits
------------------------------

The next table provides a list of all the PIR exhibits and figures showing the Chapter in which they occur and the figure numbers. Not all exhibits are down for the Simple Discrete Example.

+----+---------+-----+---------------+------------------------------+
| Rf | Kind    | Ch. | Number(s)     | Description                  |
+====+=========+=====+===============+==============================+
| A  | Table   | 2   | 2.3, 2.5,     | Estimated mean, CV, skewness |
|    |         |     | 2.6, 2.7      | and kurtosis by line and in  |
|    |         |     |               | total, gross and net.        |
+----+---------+-----+---------------+------------------------------+
| B  | Figure  | 2   | 2.2, 2.4, 2.6 | Gross and net densities on a |
|    |         |     |               | linear and log scale.        |
+----+---------+-----+---------------+------------------------------+
| C  | Figure  | 2   | 2.3, 2.5, 2.7 | Bivariate densities: gross   |
|    |         |     |               | and net with gross sample.   |
+----+---------+-----+---------------+------------------------------+
| D  | Figure  | 4   | 4.9, 4.10,    | TVaR, and VaR for unlimited  |
|    |         |     | 4.11, 4.12    | and limited variables, gross |
|    |         |     |               | and net.                     |
+----+---------+-----+---------------+------------------------------+
| E  | Table   | 4   | 4.6, 4.7, 4.8 | Estimated VaR, TVaR, and EPD |
|    |         |     |               | by line and in total, gross, |
|    |         |     |               | and net.                     |
+----+---------+-----+---------------+------------------------------+
| F  | Table   | 7   | 7.2           | Pricing summary.             |
+----+---------+-----+---------------+------------------------------+
| G  | Table   | 7   | 7.3           | Details of reinsurance.      |
+----+---------+-----+---------------+------------------------------+
| H  | Table   | 9   | 9.2, 9.5, 9.8 | Classical pricing by method. |
+----+---------+-----+---------------+------------------------------+
| I  | Table   | 9   | 9.3, 9.6, 9.9 | Sum of parts (SoP)           |
|    |         |     |               | stand-alone vs. diversified  |
|    |         |     |               | classical pricing by method. |
+----+---------+-----+---------------+------------------------------+
| J  | Table   | 9   | 9.4, 9.7,     | Implied loss ratios from     |
|    |         |     | 9.10          | classical pricing by method. |
+----+---------+-----+---------------+------------------------------+
| K  | Table   | 9   | 9.11          | Comparison of stand-alone    |
|    |         |     |               | and sum of parts premium.    |
+----+---------+-----+---------------+------------------------------+
| L  | Table   | 9   | 9.12, 9.13,   | Constant CoC pricing by unit |
|    |         |     | 9.14          | for Case Study.              |
+----+---------+-----+---------------+------------------------------+
| M  | Figure  | 11  | 11.2, 11.3,   | Distortion envelope for Case |
|    |         |     | 11.4,11.5     | Study, gross.                |
+----+---------+-----+---------------+------------------------------+
| N  | Table   | 11  | 11.5          | Parameters for the six SRMs  |
|    |         |     |               | and associated distortions.  |
+----+---------+-----+---------------+------------------------------+
| O  | Figure  | 11  | 11.6, 11.7,   | Variation in insurance       |
|    |         |     | 11.8          | statistics for six           |
|    |         |     |               | distortions as *s* varies.   |
+----+---------+-----+---------------+------------------------------+
| P  | Figure  | 11  | 11.9, 11.10,  | Variation in insurance       |
|    |         |     | 11.11         | statistics as the asset      |
|    |         |     |               | limit is varied.             |
+----+---------+-----+---------------+------------------------------+
| Q  | Table   | 11  | 11.7, 11.8,   | Pricing by unit and          |
|    |         |     | 11.9          | distortion for Case Study.   |
+----+---------+-----+---------------+------------------------------+
| R  | Table   | 13  | 13.1 missing  | Comparison of gross expected |
|    |         |     |               | losses by Case,              |
|    |         |     |               | catastrophe-prone lines.     |
+----+---------+-----+---------------+------------------------------+
| S  | Table   | 13  | 13.2, 13.3,   | Constant 0.10 ROE pricing    |
|    |         |     | 13.4          | for Case Study, classical    |
|    |         |     |               | PCP methods.                 |
+----+---------+-----+---------------+------------------------------+
| T  | Figure  | 15  | 15.2 - 15.7   | Twelve plot.                 |
|    |         |     | (G/N)         |                              |
+----+---------+-----+---------------+------------------------------+
| U  | Figure  | 15  | 15.8, 15.9,   | Capital density by layer.    |
|    |         |     | 15.10         |                              |
+----+---------+-----+---------------+------------------------------+
| V  | Table   | 15  | 15.35, 15.36, | Constant 0.10 ROE pricing    |
|    |         |     | 15.37         | for Cat/Non-Cat Case Study,  |
|    |         |     |               | distortion, SRM methods.     |
+----+---------+-----+---------------+------------------------------+
| W  | Figure  | 15  | 15.11         | Loss and loss spectrums.     |
+----+---------+-----+---------------+------------------------------+
| X  | Figure  | 15  | 15.12, 15.13, | Percentile layer of capital  |
|    |         |     | 15.14         | allocations by asset level.  |
+----+---------+-----+---------------+------------------------------+
| Y  | Table   | 15  | 15.38, 15.39, | Percentile layer of capital  |
|    |         |     | 15.40         | allocations compared to      |
|    |         |     |               | distortion allocations.      |
+----+---------+-----+---------------+------------------------------+



