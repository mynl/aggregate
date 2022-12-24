.. _2_x_case_studies:

.. quick look 2022-12-24
.. NEEDS WORK

Case Studies
===================

**Objectives:** Recreate Case Study exhibits from PIR; create exhibits for your own portfolio.

**Audience:** Intermediate to advanced users.

**Prerequisites:** Base understanding of ``aggregate``. Familiarity with PIR.

**See also:**

**Contents:**

#. :ref:`Four Book Case Studies`

The book  `Pricing Insurance Risk <https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_  (PIR) presents four Case Studies. In this section we show how to reproduce all the exhibits shown in the book for each case and how users can create their own custom cases.

Four Book Case Studies
--------------------------------

The Cases are to help practitioners develop an intuition for how different methods price business, informing their selection of an appropriate method for an intended purpose without resorting to trial and error. The cases describe business written by Ins Co., a one-period de novo insurer that comes into existence at time zero, raises capital and writes business, and pays all losses at time one.

The Cases share several common characteristics.

* Each includes two units, one lower risk and one higher.
* Reinsurance is applied to the riskier unit.
* Total unlimited losses are calibrated to ¤100. (The symbol ¤ denotes a generic currency.)
* Losses are in ¤millions, although the actual unit is irrelevant.

For each Case Study we produce a standard set of exhibits.

.. this format omits from toc
   rubric:: Introduction to Case Studies
   :name: introduction-to-case-studies
   :class: mt-5


The Four PIR Case Studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PIR uses four Case Studies to illustrate the theory:

#.  `Simple Discrete Example <#the-simple-discrete-example>`__
#.  `Tame: two tame lines <#tame-case-study>`__
#.  `Catastrophe and Non-Catastrophe Case Study <#catastrophe-and-non-catastrophe-case-study>`__
#.  `Hurricane/Severe Convective Storm <#hurricane-and-severe-storm-case-study>`__

Simple Discrete Example
"""""""""""""""""""""""""

Ins Co. writes two units taking on loss values
*X*\ :sub:`1` = 0, 8, or 10, and *X*\ :sub:`2` = 0, 1, or
90. The units are independent and sum to the portfolio loss
*X* = *X*\ :sub:`1` + *X*\ :sub:`2`. The outcome
probabilities are 1/2, 1/4, and 1/4 respectively for each
marginal. The 9 possible outcomes, with associated
probabilities, are shown below. The output is typical of
that produced by a catastrophe, capital, or pricing
simulation model—albeit much simpler.

.. table:: Simple Discrete Example with nine possible outcomes.

   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | :math:`X_1` | :math:`X_2` | :math:`X` | :math:`\mathsf{Pr}(X_1)` | :math:`\mathsf{Pr}(X_2)` | :math:`\mathsf{Pr}(X)`  |
   +=============+=============+===========+==========================+==========================+=========================+
   | 0           | 0           | 0         | 1/2                      | 1/2                      | 1/4                     |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 0           | 1           | 1         | 1/2                      | 1/4                      | 1/8                     |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 0           | 90          | 90        | 1/2                      | 1/4                      | 1/8                     |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 8           | 0           | 8         | 1/4                      | 1/2                      | 1/8                     |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 8           | 1           | 9         | 1/4                      | 1/4                      | 1/16                    |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 8           | 90          | 98        | 1/4                      | 1/4                      | 1/16                    |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 10          | 0           | 10        | 1/4                      | 1/2                      | 1/8                     |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 10          | 1           | 11        | 1/4                      | 1/4                      | 1/16                    |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+
   | 10          | 90          | 100       | 1/4                      | 1/4                      | 1/16                    |
   +-------------+-------------+-----------+--------------------------+--------------------------+-------------------------+

Tame Case Study
""""""""""""""""


In the Tame Case Study, Ins Co. writes two predictable units
with no catastrophe exposure. We include it to demonstrate
an idealized risk-pool: it represents the best case—from Ins
Co.’s perspective. It could proxy a portfolio of personal
and commercial auto liability.

It uses a straightforward stochastic model with gamma
distributions.

The Case includes a gross and net view. Net applies
aggregate reinsurance to the more volatile unit B with an
attachment probability 0.2 (¤56) and detachment probability
0.01 (¤69).

Catastrophe and Non-Catastrophe Case Study
"""""""""""""""""""""""""""""""""""""""""""


In the Cat/Non-Cat Case Study, Ins Co. has catastrophe and
non-catastrophe exposures. The non-catastrophe unit proxies
a small commercial lines portfolio. Balancing the relative
benefits of units considered to be more stable against more
volatile ones is a very common strategic problem for
insurers and reinsurers. It arises in many different guises:

-  Should a US Midwestern company expand to the East coast
   (and pick up hurricane exposure)?
-  Should an auto insurer start writing homeowners?
-  What is the appropriate mix between property catastrophe
   and non-catastrophe exposed business for a reinsurer?

This Case uses a stochastic model similar to the Tame Case.
The two units are independent and have gamma and lognormal
distributions.

The Case includes a gross and net view. Net applies
aggregate reinsurance to the Cat unit with an attachment
probability 0.1 (¤41) and detachment probability 0.005
(¤121).

Hurricane and Severe Storm Case Study
""""""""""""""""""""""""""""""""""""""

In the Hu/SCS Case Study, Ins Co. has catastrophe exposures
from severe convective storms (SCS) and, independently,
hurricanes (Hu). In practice, hurricane exposure is modeled
using a catastrophe model. We proxy that using a very severe
lognormal distribution in place of the gross catastrophe
model event-level output. Both units are modeled by an
aggregate distribution with a Poisson frequency and
lognormal severity.

The Case includes a gross and net view. Net applies
aggregate (see Errata) reinsurance to the HU unit with an
occurrence attachment probability 0.05 (¤40) and detachment
probability 0.005 (¤413).

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
