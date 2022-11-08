.. _2_x_agg_language:

======================
The ``agg`` Language
======================


.. _discrete:

Specifying Discrete Distributions
---------------------------------

A discrete distribution is entered as one or two equal-length rows vectors. The
first gives the outcomes and the second the probabilities. The elements are optionally
comma separated (most examples omit the commas). For example

::

    [0 9 10] [.5 .3 .2]

specifies an aggregate with a 0.5 chance of taking the value 0, 0.3
chance of 9, and 0.2 of 10. If the probabilities are omitted they are assumed
to be equal (to the reciprocal of the length of the outcomes). As a convenience
probabilities can be entered as fractions:

::

    [0 9 10] [1/3 3/10 2/10]


.. warning::
    Use binary fractions (denominator a power of two) to avoid rounding errors!

When executed, an discrete severity specification is converted into a ``dhistogram``. ``dfreq`` distributions are a particular frequency distribution.


Specifying Parametric Distributions
-----------------------------------

Parametric distributions can be specified in two different ways.

1. As ``sev DISTNAME MEAN cv CV`` where ``DISTNAME`` is the distribution name, chosen from the list below, ``MEAN`` is the expected loss and ``CV`` is the loss coefficient of
variation.
2. As ``sev SCALE * DISTNAME SHAPE`` where ``SCALE`` and ``SHAPE`` are the ``scipy.stats`` parameters. For zero parameter distributions ``SHAPE`` is omitted. Two parameter distributions are ``sev SCALE * DISTNAME SHAPE1 SHAPE2``.


Available distributions:

-  ``lognorm``: lognormal
-  ``gamma``: gamma
-  ``invgamma``: invgamma

All continuous, one parameter distributions in scipy.stat are available
by name. See below for details on using a Pareto, normal, exponential,
or beta distribution.

**Example.** Entering ``sev lognorm 10 cv 0.2`` produces a lognormal
distribution with a mean of 10 and a CV of 0.2.

When executed, a sev specification is converted into full aggregate
program form.

.. _agg:

Specifying Aggregate Distributions
----------------------------------

Aggregate distributions are specified using the ``aggregate`` language
as

::

    agg [label]
        [exposure] [limit] \
        [severity] \
        [occurrence re] \
        [frequency] \
        [aggregate re]

Throughout the programs a backslash is a newline continuation for
readability. All programs are one line long.
The words ``agg`` and ``sev`` are keywords (like ``if/then/else``),
``[label], [exposure], [limit]?`` etc. are user inputs, and the limit
clause is optional.

For example

::

       agg Auto 10 claims sev lognorm 10 cv 1.3 poisson

creates an aggregate with label Auto, an expected claim count of 10,
severity sampled from an unlimited lognormal distribution with mean 10
and CV 1.3, and a Poisson frequency distribution. The layer is unlimited
because the limit clause missing. The label must begin with a letter and
contain just letters and numbers. It can't be a language keyword, e.g.
agg, port, poisson, fixed

Exposure can be specified in three ways.

-  Stated expected loss and severity (claim count derived)
-  Premium and loss ratio and severity (expected loss and claim count
   derived)
-  Claim count times severity (expected loss derived)

For example

::

       123 claim[s]
       1000 loss
       1000 premium at 0.7 [lr]?

The first gives the expected claim count; the ``s`` on ``claims`` is
optional. The second gives the expected loss with claim counts derived
from average severity. The third gives premium and a loss ratio with
counts again derived from severity. The final ``lr`` is optional and
just used for clarity.

The first defines an expected claim count of 123. The second the
expected loss of 1000, where the claim count is derived from the average
severity. The third specifies premium of 1000 and a loss ratio of 70%.
Again, claim counts are derived from severity. The final lr\` is for
clarity and is optional. The pipe (\|) notation indicates alternative
choices.

Limit are entered as layer ``xs`` attachment or layer ``x`` attachment.

Here are four illustrative examples. The line must start with ``agg``
(no tabs or spaces first) but afterwards spacing within the spec is
ignored and can be used to enhance readability. The newline is needed.

::

       agg Example1   10  claims  30 xs 0 sev lognorm 10 cv 3.0 fixed
       agg Example2   10  claims 100 xs 0 sev 100 * expon + 10 poisson
       agg Example3 1000  loss    90 x 10 sev gamma 10 cv 6.0 mixed gamma 0.3
       agg Example4 1000  premium at 0.7 lr inf x 50 sev invgamma 20 cv 5.0 binomial 0.4


-  ``Example1`` 10 claims from the 30 x 0 layer of a lognormal severity
   with (unlimited) mean 10 and cv 3.0 and using a fixed claim count
   distribution (i.e. always exactly 10 claims).
-  ``Example2`` 10 claims from the 100 x 0 layer of an exponential
   severity with (unlimited) mean 100 shifted right by 10, and using a
   Poisson claim count distribution. The exponential has no shape
   parameters, it is just scaled. The mean refers to the unshifted
   distribution.
-  ``Example3`` 1000 expected loss from the 90 x 10 layer of a gamma
   severity with (unlimited) mean 10 and cv of 6.0 and using a
   gamma-mixed Poisson claim count distribution. The mixing distribution
   has a cv of 0.3 The claim count is derived from the **limited**
   severity.
-  ``Example4`` 700 of expected loss (1000 premium times 70 percent loss
   ratio) from an unlimited excess 50 layer of a inverse gamma
   distribution with mean of 20 and cv of 5.0 using a binomial
   distribution with p=0.4. The n parameter for the binomial is derived
   from the required claim count.

The inverse Gaussian (``ig``), ``delaporte``, ``Sichel`` and other
distributions are available as mixing distributions.

The `programs page </cases/programs>`__ provides a list of different
ways to specify an aggregate distribution using the language.

The `Aggregate Manual <https://www.mynl.com/aggregate/>`__ provides more
details.


Shifting and scaling severity
-----------------------------


Limits and attachments
----------------------

Limit are entered as layer ``xs`` attachment or simply ``x``. The
examples below show severity limited to 30, to 100, an excess layer 90 x
10, and an unlimited layer xs 50.

::

   agg Auto 10 claims  30 xs 0 sev lognorm 10 cv 1.3 poisson
   agg Auto 10 claims 100 xs 0 sev lognorm 10 cv 1.3 poisson
   agg Auto 10 claims  90 x 10 sev lognorm 10 cv 1.3 poisson
   agg Auto 10 claims inf x 50 sev lognorm 10 cv 1.3 poisson

The severity distribution is specified by name. Any ``scipy.stats``
continuous distribution with one shape parameter can be used, including
the gamma, lognormal, Pareto, Weibull etc. The exponential and normal
variables, with no shape parameters, and the beta with two shape
parameters are also available. Most distributions can be entered via
mean and CV, or specified by their shape parameters and then scaled and
shifted, using the standard ``scipy.stats`` ``scale`` and ``loc``
notations, see . Finally ``dhistogram`` and ``chistogram`` can be used
to create discrete (point mass) and continuous (ogive) empirical
distributions. Here are some examples.

+--------------------------+--------------+--------------------------+
| Code                     | Distribution | Meaning                  |
+==========================+==============+==========================+
| ``sev lognorm 10 cv 3``  | lognormal    | mean 10, cv 0.3          |
+--------------------------+--------------+--------------------------+
| `                        | lognormal    | 10\ *X*, *X*             |
| `sev 10 * lognorm 1.75`` |              | lognor                   |
|                          |              | mal(*μ* = 0, *σ* = 1.75) |
+--------------------------+--------------+--------------------------+
| ``sev                    | lognormal    | 10\ *X* + 20             |
| 10 * lognorm 1.75 + 20`` |              |                          |
+--------------------------+--------------+--------------------------+
| ``sev 10                 | lognormal    | 10\ *Y* + 50, *Y*        |
|  * lognorm 1 cv 3 + 50`` |              | lognormal mean 1 cv 3    |
+--------------------------+--------------+--------------------------+
| ``sev                    | Pareto       | Pareto, survival         |
| 100 * pareto 1.3 - 100`` |              | (100/                    |
|                          |              | (100+\ *x*))\ :sup:`1.3` |
+--------------------------+--------------+--------------------------+
| `                        | normal       | mean 100, std dev 50     |
| `sev 50 * normal + 100`` |              |                          |
+--------------------------+--------------+--------------------------+
| ``sev 5 * expon``        | exponential  | mean 5                   |
+--------------------------+--------------+--------------------------+
| ``sev 5 * uniform + 1``  | uniform      | uniform between 1 and 6  |
+--------------------------+--------------+--------------------------+
| ``sev 50 * beta 2 3``    | beta         | 50\ *Z*, *Z* beta        |
|                          |              | parameters 2, 3          |
+--------------------------+--------------+--------------------------+

The frequency is specified as follows. The expected claim count is *n*.

+-----------------+---------------------------------------------------+
| Code            | Meaning                                           |
+=================+===================================================+
| ``fixed``       | Fixed *n* claims, degenerate distribution         |
+-----------------+---------------------------------------------------+
| ``poisson``     | Poisson mean *n*                                  |
+-----------------+---------------------------------------------------+
| ``bernoulli``   | *p* = *n*                                         |
+-----------------+---------------------------------------------------+
| `               | Binomial *n*, *p* = 0.3, note mean is **not** *n* |
| `binomial 0.3`` | in this case                                      |
+-----------------+---------------------------------------------------+
| ``mixed         | Mixed Poisson. First parameter is always CV.      |
| ID 0.3 [0.1]?`` | Second varies with type.                          |
+-----------------+---------------------------------------------------+

The mixing distribution can be gamma for a negative binomial, inverse
Gaussian (``ig``), Delaporte, Sichel etc.

Limit Profiles
^^^^^^^^^^^^^^

The exposure variables can be vectors to express a *limit profile*. All
``exp_[en|prem|loss|count]`` related elements are broadcast against
one-another. For example

::

       [100 200 400 100] premium at 0.65 lr [1000 2000 5000 10000] xs 1000

expresses a limit profile with 100 of premium at 1000 x 1000; 200 at
2000 x 1000 400 at 5000 x 1000 and 100 at 10000 x 1000. In this case all
the loss ratios are the same, but they could vary too, as could the
attachments.

Mixtures
^^^^^^^^

The severity variables can be vectors to express a *mixed severity*. All
``sev_`` elements are broadcast against one-another. For example

::

   sev lognorm 1000 cv [0.75 1.0 1.25 1.5 2] wts [0.4, 0.2, 0.1, 0.1, 0.1]

expresses a mixture of five lognormals with a mean of 1000 and CVs as
indicated with weights 0.4, 0.2, 0.1, 0.1, 0.1. Equal weights can be
express as wts=[5], or the relevant number of components.

Limit Profiles and Mixtures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Limit profiles and mixtures can be combined. Each mixed severity is
applied to each limit profile component. For example

::

           ag = uw('agg multiExp [10 20 30] claims [100 200 75] xs [0 50 75]
               sev lognorm 100 cv [1 2] wts [.6 .4] mixed gamma 0.4')```

creates an aggregate with six severity subcomponents.

= ========= ============== ==========
# **Limit** **Attachment** **Claims**
= ========= ============== ==========
0 100       0              6
1 100       0              4
2 200       50             12
3 200       50             8
4 75        75             18
5 75        75             12
= ========= ============== ==========

Circumventing Products
^^^^^^^^^^^^^^^^^^^^^^

It is sometimes desirable to enter two or more lines each with a
different severity but with a shared mixing variable. For example to
model the current accident year and a run- off reserve, where the
current year is gamma mean 100 cv 1 and the reserves are larger
lognormal mean 150 cv 0.5 claims requires

::

           agg MixedPremReserve [100 200] claims \
             sev [gamma lognorm] [100 150] cv [1 0.5] \
             mixed gamma 0.4

so that the result is not the four-way exposure / severity product but
just a two-way combination. These two cases are distinguished looking at
the total weights. If the weights sum to one then the result is an
exposure / severity product. If the weights are missing or sum to the
number of severity components (i.e. are all equal to 1) then the result
is a row by row combination.

Determining Expected Claim Count
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variables are used in the following order to determine overall expected
losses.

-  If ``count`` is given it is used
-  If ``loss`` is given then count is derived from the severity
-  If ``prem[ium]`` and ``[at] 0.7 lr`` are given then the loss is
   derived and counts from severity

In addition:

-  If ``prem`` is given the loss ratio is computed
-  Claim count is conditional but severity can have a mass at zero
-  X is the GROUND UP severity, so X \| X > attachment is used and
   generates n claims **really?**

Unconditional Severity
~~~~~~~~~~~~~~~~~~~~~~

The severity distribution is conditional on a loss to the layer. For an
excess layer *y* xs *a* the severity is has distribution *X* ∣ *X*>,
where *X* is the specified severity. For a ground-up layer there is no
adjustment.

The default behavior can be over-ridden by adding ``!`` after the
severity distribution. For example

::

   agg Conditional 1 claim 10 x 10 sev lognorm 10 cv 1 fixed
   agg Unconditional 1 claim 10 x 10 sev lognorm 10 cv 1 ! fixed

produces conditional and unconditional samples from an excess layer of a
lognormal. The latter includes an approximately 0.66 chance of a claim
of zero, corresponding to *X* ≤ 10 below the attachment.


Example ``aggregate`` programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test_suite... builder html output?

