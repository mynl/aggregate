.. _2_x_case_studies:

===================
Case Studies
===================

The book presents four standard case studies. Using the new case page, users can also create their own custom case studies and produce the standard set of exhibits by specifying the stochastic model, capital standard and cost of capital, and reinsurance. See new case instructions for more details.

Four Standard Book Case Studies
--------------------------------

The book uses four Case Studies to illustrate the theory:

* Simple Discrete Example
* Tame: two tame lines
* Catastrophe and Non-Catastrophe Case Study
* Hurricane/Severe Convective Storm

The Cases aim to help practitioners develop an intuition for how each method prices business, informing their selection of an appropriate method for an intended purpose without resorting to trial and error.

The Cases share several common characteristics.

* Each includes two units, one lower risk and one higher.
* Reinsurance is applied to the riskier unit.
* Total unlimited losses are calibrated to ¤100. (The symbol ¤ denotes a generic currency.)
* Losses are in ¤millions, although the actual unit is irrelevant.

For each Case Study we produce a standard set of exhibits. The website supplements these with some extended exhibits that vary by case.


.. rubric:: Introduction to Case Studies
   :name: introduction-to-case-studies
   :class: mt-5

The book presents `four standard case
studies <#book_case_studies>`__. Using the `new
case </cases/new>`__ page, users can also create their own
custom case studies and produce the standard set of exhibits
by specifying the stochastic model, capital standard and
cost of capital, and reinsurance. See `new case
instructions </cases/instructions>`__ for more details.

.. rubric:: Four Standard Book Case Studies
   :name: book_case_studies

The book uses four Case Studies to illustrate the theory:

-  `Simple Discrete
   Example <#the-simple-discrete-example>`__
-  `Tame: two tame lines <#tame-case-study>`__
-  `Catastrophe and Non-Catastrophe Case
   Study <#catastrophe-and-non-catastrophe-case-study>`__
-  `Hurricane/Severe Convective
   Storm <#hurricane-and-severe-storm-case-study>`__

The Cases aim to help practitioners develop an intuition for
how each method prices business, informing their selection
of an appropriate method for an intended purpose without
resorting to trial and error.

The Cases share several common characteristics.

-  Each includes two units, one lower risk and one higher.
-  Reinsurance is applied to the riskier unit.
-  Total unlimited losses are calibrated to ¤100. (The
   symbol ¤ denotes a generic currency.)
-  Losses are in ¤millions, although the actual unit is
   irrelevant.

For each Case Study we produce a `standard set of
exhibits </results/appendix>`__. The website supplements
these with some etended exhibits that vary by case.

.. rubric:: Simple Discrete Example
   :name: the-simple-discrete-example

`Results for the Discrete
Example. </results?case=discrete>`__

Ins Co. writes two units taking on loss values
*X*\ :sub:`1` = 0, 8, or 10, and *X*\ :sub:`2` = 0, 1, or
90. The units are independent and sum to the portfolio loss
*X* = *X*\ :sub:`1` + *X*\ :sub:`2`. The outcome
probabilities are 1/2, 1/4, and 1/4 respectively for each
marginal. The 9 possible outcomes, with associated
probabilities, are shown below. The output is typical of
that produced by a catastrophe, capital, or pricing
simulation model—albeit much simpler.

.. table:: Simple Discrete Example with nine possible
outcomes.

   +----------+----------+-------+----------+----------+----------+
   | **X**\   | **X**\   | **X** | *        | *        | *        |
   | :sub:`1` | :sub:`2` |       | *Pr(X**\ | *Pr(X**\ | *Pr(X)** |
   |          |          |       |  :sub:`1 |  :sub:`1 |          |
   |          |          |       | `\ **)** | `\ **)** |          |
   +==========+==========+=======+==========+==========+==========+
   | 0        | 0        | 0     | 1/2      | 1/2      | 1/4      |
   +----------+----------+-------+----------+----------+----------+
   | 0        | 1        | 1     | 1/2      | 1/4      | 1/8      |
   +----------+----------+-------+----------+----------+----------+
   | 0        | 90       | 90    | 1/2      | 1/4      | 1/8      |
   +----------+----------+-------+----------+----------+----------+
   | 8        | 0        | 8     | 1/4      | 1/2      | 1/8      |
   +----------+----------+-------+----------+----------+----------+
   | 8        | 1        | 9     | 1/4      | 1/4      | 1/16     |
   +----------+----------+-------+----------+----------+----------+
   | 8        | 90       | 98    | 1/4      | 1/4      | 1/16     |
   +----------+----------+-------+----------+----------+----------+
   | 10       | 0        | 10    | 1/4      | 1/2      | 1/8      |
   +----------+----------+-------+----------+----------+----------+
   | 10       | 1        | 11    | 1/4      | 1/4      | 1/16     |
   +----------+----------+-------+----------+----------+----------+
   | 10       | 90       | 100   | 1/4      | 1/4      | 1/16     |
   +----------+----------+-------+----------+----------+----------+

.. rubric:: Tame Case Study
   :name: tame-case-study

`Results for the Tame Case Study. </results?case=tame>`__

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

.. rubric:: Catastrophe and Non-Catastrophe Case Study
   :name: catastrophe-and-non-catastrophe-case-study

`Results for the Cat/NonCat Study. </results?case=cnc>`__

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

.. rubric:: Hurricane and Severe Storm Case Study
   :name: hurricane-and-severe-storm-case-study

`Results for the Hu/SCS Case Study. </results?case=hs>`__

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



Bodoff’s Examples
-----------------

We now show the definition above reproduces Bodoff’s “Thought experiment
1”. He considers a situation of two losses wind, *W*, and earthquake,
*Q*, where *W* and *Q* are independent, *W* takes the value 99 with
probability 20% and otherwise zero, and *Q* takes the value 100 with
probability 5% and otherwise zero. Total losses *Y* = *W* + *Q*. There
are four possibilities as shown in Table [t:bod1].

.. table:: Bodoff Thought Experiment 1

   =================== ===============
   **Event**           **Probability**
   =================== ===============
   No Loss             0.76
   *W* = 99            0.19
   *Q* = 100           0.04
   *W* = 99, *Q* = 100 0.01
   =================== ===============

**Bodoff’s Examples in ``Aggregate``**

Here are the ``Aggregate`` programs for the examples Bodoff considers.

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
