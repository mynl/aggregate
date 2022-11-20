.. _2_x_agg_language:

The ``agg`` Language
======================

**Objectives:** Introduce the ``agg`` language grammar.

**Audience:** User who wants to use it to build realistic aggregates.

**Prerequisites:** Familiar with building aggregates using ``build``. Probability theory behind aggregate distributions. Insurance and reinsurance terminology.

**See also:** :doc:`2_x_exposure`, :doc:`2_x_limits`, :doc:`2_x_severity`, :doc:`2_x_frequency`, :doc:`2_x_mixtures`, :doc:`2_x_vectorization`, :doc:`2_x_reinsurance` and :doc:`2_x_re_pricing`, and :doc:`../4_agg_Language_Reference`.

**Notation:** ``<item>`` denotes an optional term.


.. _design and purpose:

Design and Purpose
-------------------

The ``agg`` language is designed to make it easy to go from "dec page to distribution" (or, with less alliteration, from reinsurance slip to distribution). Coverage expressed concisely in words can be hard to program. Consider

    A trucking policy with a loss pick 4500, a limit of 1000, and a retention 50.

To estimate the aggregate distribution the actuary must

#. Select a suitable trucking ground-up severity curve, say lognormal with mean 50 and CV 1.75.
#. Compute the expected conditional layer severity for the layer 1000 xs 50
#. Divide severity into the loss pick to determine the expected claim count
#. Select a suitable frequency distribution, say Poisson
#. Calculate a numerical approximation to the resulting compound-Poisson aggregate distribution

The ``agg`` program takes care of many of these details based on the simple program::

    agg Trucking 4500 loss 1000 xs 50 sev lognorm 50 1.75 poisson

The program specifies the distributions selected in steps 1 and 4; these require actuarial judgment and cannot be automated. Based on this input, the ``aggregate`` package computes steps 2, 3, and 5. Its parts are explained in the next section.


.. _agg:

Specifying an Aggregate Distribution
-------------------------------------

Aggregate distributions are specified using :ref:`seven clauses <seven clauses>`, entered in the ``aggregate`` language as::

    agg label               \
        exposure <limit>    \
        severity            \
        <occurrence re>     \
        <frequency>         \
        <aggregate re>      \
        <note>

All programs are one line long.
A backslash is a newline continuation (like Python) and is used only for readability.
Horizontal white space is ignored.
The entries are as follows.


* ``agg`` is the keyword used to create an aggregate distribution. Keywords are part of the language, like ``if/then/else`` in VBA, R or Python, or ``select`` in SQL.

* ``label`` (``Trucking`` in the prior example) is a string label. It can contain letters and numbers and periods and must start with a letter. It is case sensitive. It cannot contain an underscore. It cannot be an ``agg`` language keyword. E.g., ``Motor``, ``NE.Region``, ``Unit.A`` but not ``12Line`` or ``NE_Region``.

* The ``exposure`` clause, ``4500 loss 1000 xs 50``, determines the volume of insurance, see :ref:`exposure <2_agg_class_exposure_clause>`. It optionally includes a ``layers`` :ref:`subclause <2_agg_class_layers_subclause>` (``1000 xs 50``) to set policy occurrence limits and deductibles. The exposure clause can also use the ``dfreq`` keyword REF.

* The ``severity`` clause, ``sev lognorm 50 1.75``, determines the *ground-up* severity, see :ref:`severity <2_agg_class_severity_clause>`. ``sev`` is a keyword

* ``occurrence_ re`` (omitted) specifies a per occurrence reinsurance structure. It is optional. See :ref:`reinsurance <2_agg_class_reinsurance_clause>`.

* The ``frequency`` clause, ``poisson``, specifies the frequency distribution, see :ref:`frequency <2_agg_class_frequency_clause>`.

* ``aggregate_re`` (omitted) specifies an aggregate reinsurance structure. It is optional. See :ref:`reinsurance <2_agg_class_reinsurance_clause>`.

* ``note`` (omitted) is a comment about the distribution. It is optional. See :ref:`note <2_agg_class_note_clause>`.


The package automatically computes the expected claim count from the expected loss and average severity.

The rest of this section describes the basic features of each clause; a separate chapter on each fills in the missing details.

There are two other specifications for different situations::

    agg label BUILTIN_AGG

    BUILTIN_AGG

These pull a reference distribution from the ``knowledge`` database by label, ``BUILTIN_AGG``. The first format gives ``BUILTIN_AGG`` a new label; the second uses its saved label. See the :doc:`../4_agg_Language_Reference`.



.. _2_agg_class_exposure_clause:

The ``exposure`` Clause
--------------------------

The ``exposure`` clause has two parts ``exposures <layers>``. The first specifies
the volume of insurance, the second adjusts the ground-up severity. Exposures can be specified in
four ways

-  Stated expected loss and severity (claim count derived)
-  Premium and loss ratio and severity (expected loss and claim count
   derived)
-  Claim count times severity (expected loss derived)
-  Using the ``dfreq`` keyword to directly enter the frequency distribution

For example::

       123  claims
       1000 loss
       1000 premium at 0.7 lr
       dfreq [1 2 3] [3/4 3/16 1/16]


* ``123 claims`` directly specifies the expected claim count; the last letter ``s`` on ``claims`` is optional.
* ``1000 loss`` directly specifies expected loss. The claim count is derived from average severity.
  It is typical for an actuary to estimate the loss pick and select a severity curve and then
  derive frequency.
* ``1000 premium at 0.7 lr`` directly specifies premium and a loss ratio. The claim count is again derived
  from severity. The final ``lr`` is optional and used just for clarity. Again, actuaries
  often take plan premiums and apply loss ratio picks to determine losses, rather than
  starting with a loss pick. This idiom supports that approach.
* ``dfreq [1 2 3] [3/4 3/16 1/16]`` specifies frequency outcomes and probabilities directly. It is described in `nonparametric frequency`_.

All values in the first three specifications can be :ref:`vectorized <2_x_vectorization>`.

See :doc:`2_x_exposure` for more details.

.. _2_agg_class_layers_subclause:

The Limits Subclause
~~~~~~~~~~~~~~~~~~~~~

The optional ``limits`` subclause describes policy occurrence limits and deductibles. For example::

    100 xs 0
    inf xs 100
    750 xs 250
    1 x 1

The first applies an occurrence limit of 100. The second applies a deductible of 100. The third is an excess layer, with limit 750 and retention 250. The last is also an excess layer of 1 xs 1.
``inf`` denotes infinity, for an unlimited layer. Either `xs` or `x` are acceptable.  :ref:`Multiple layers <2_x_vectorization>` can be entered at once.

.. _2_agg_class_severity_clause:

The ``severity`` Clause
-------------------------

The severity clause specifies the ground-up severity ("curve"). It is very flexible. Its design follows the ``scipy.stats`` package's specification of random variables using shape, location, and scale factors, see :ref:`probability background <5_x_probability>`. The syntax is different for parametric continuous and discrete severity curves.

Parametric Severity
~~~~~~~~~~~~~~~~~~~~~~

The two parametric specifications are::

    sev DIST_NAME MEAN cv CV
    sev DIST_NAME <SHAPE1> <SHAPE2>

where

* ``sev`` is a keyword indicating the severity specification.
* ``DIST_NAME`` is the ``scipy.stats`` distribution name, such as our favorites ``lognorm``, ``gamma``, ``pareto``, ``expon``, ``beta``, ``unif``.
* ``MEAN`` is the expected loss.
* ``CV`` is the loss coefficient of variation.
* ``SHAPE1``, ``SHAPE2`` are the shape variables.

The first form directly enters the expected ground-up severity and cv. It is available for distributions with only one shape parameter and the beta distribution. ``aggregate`` uses a formula (lognormal, gamma, beta) or numerical method to solve for the shape parameter to achieve the correct cv and then scales to the desired mean. The second form directly enters the shape variable(s). Shape parameters entered for zero parameter distributions are ignored.

``DIST_NAME`` can be any zero, one, or two shape parameter ``scipy.stats`` continuous distribution.
They have (mostly) easy to guess names.
See :doc:`2_x_severity` for a full list.

.. _nonparametric severity:

Non-Parametric Severity Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Discrete distributions (supported on a finite number of outcomes)
can be directly entered as a severity using the ``dsev`` keyword followed by
two equal-length rows vectors. The first gives the outcomes and the second the
probabilities.

::

    dsev [outcomes] <[probabilities]>

The horizontal layout is irrelevant and commas are optional.
If the ``probabilities`` vector is omitted then all probabilities are set equal to
the reciprocal of the length of the ``outcomes`` vector.
A Python-like colon notation is available for ranges.
Probabilities can be entered as fractions, but no other arithmetic operation is supported.

The five examples::

    dsev [0 9 10] [0.5 0.3 0.2]
    dsev [0 9 10]
    dsev [1:6]
    dsev [0:100:25]
    dsev [1:6] [1/4 1/4 1/8 1/8 1/8 1/8]

specify

#. A severity with a 0.5 chance of taking the value 0, 0.3 chance of 9, and 0.2 of 10.
#. Equally likely outcomes of 0, 9, or 10;
#. Equally likely outcomes 1, 2, 3, 4, 5, 6;
#. Equally likely outcomes 0, 25, 50, 100; and
#. Outcomes 1 or 2 with probability 0.25 or 3-6 with probability 0.125.

.. warning::
    Use binary fractions (denominator a power of two) to avoid rounding errors!

When executed, an discrete severity specification is converted into a ``scipy.stats`` ``histogram`` class. Internally there are discrete and continuous (ogive) histograms, sees REF.


.. _2_agg_class_frequency_clause:

The ``frequency`` Clause
--------------------------

The exposure and severity clauses determine the expected claim count. The ``frequency`` clause specifies the other particulars of the claim count distribution. As with severity, the syntax is different for parametric and non-parametric discrete distributions.

Parametric Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parametric frequency distributions are supported. Remember that the ``exposure`` clause determines the expected claim count.

* ``poisson``, no additional parameters required
* ``geometric``, no additional parameters required
* ``fixed``, no additional parameters required
* ``bernoulli``, expected claim count must be :math:`\le 1`.
* ``binomial SHAPE``, the shape determines :math:`p` and :math:`n=\mathsf{E}[N]/p`.
* ``pascal SHAPE1 SHAPE2`` (the generalized Poisson-Pascal, see REF), where ``SHAPE1``
  gives the cv and ``SHAPE2`` the number of claims per occurrence.

In addition, a :math:`G`-mixed Poisson frequency (see `mixed frequency distributions`_, remember :math:`G` must have expectation 1) can be specified using the ``mixed`` keyword, followed by the name and shape parameters of the mixing distribution::

    mixed DIST_NAME SHAPE1 <SHAPE2>

For example::

    agg 5 claims dsev [1] mixed gamma 0.16

produces a negative binomial (gamma-mixed Poisson) distribution with variance :math:`5\times (1 + 0.16^2 \times 5)`.

See :doc:`2_x_frequency` for more details.

.. _nonparametric frequency:

Non-Parametric Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An exposure clause::

    dfreq [outcomes] <[probabilities]>

directly specifies the frequency distribution. The ``outcomes`` and ``probabilities`` are specified as in `nonparametric severity`_.

.. _2_agg_class_reinsurance_clause:

The ``reinsurance`` Clauses
----------------------------

Occurrence and aggregate reinsurance can be specified in the same way as limits and deductibles.
Both clauses are optional.
The ceded or net position can be output. Layers can be stacked and can include co-participations. For example, the three programs (the last displayed over four lines):

    agg Trucking 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence net of 750 xs 250 poisson

    agg WorkComp 15000 loss 500 xs 0 sev lognorm 50 cv 1.75 poisson aggregate ceded to 50% so 2000 xs 15000

    agg Trucking 5000 loss 1000 xs 0 \
    sev lognorm 50 cv 1.75 \
    occurrence net of 50% so 250 xs 250 and 500 xs 500 poisson \
    aggregate net of 250 po 1000 xs 4000 and 5000 xs 5000

specify the following:

1. The distribution of losses to the net position on the Trucking policy after a per occurrence cession of the 750 xs 250 layer. This net position could also be written without reinsurance as

    agg Trucking 4500 loss  250 xs 50 sev lognorm 50 1.75 poisson

  All occurrence reinsurance has free and unlimited reinstatements. Running

    agg Trucking 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence ceded to 750 xs 250 poisson

  would model ceded losses.

2. The distribution of losses to an aggregate protection for the 2000 xs 15000 layer of total losses, limited to 500. The underlying business could be an SIR on a large account Workers Compensation policy, and the aggregate is a part of the insurance charge (Table L, M).

3. Back to Trucking. Now we apply two occurrence layers. The first, 250 xs 250, is only 50% placed (so stands for share of), and the second is 100% of 500 xs 500. The net of these programs flows through to aggregate layers, 250 part of of 1000 xs 4000 (25% placement), and 100% of the 5000 xs 5000 aggregate layers. The modeled outcome is net of all four layers. In this case, it is not possible to write the net of occurrence using limits and attachments.

The distributions for these models are shown  in `realistic examples`_.

See :ref:`reinsurance pricing examples <2_x_re_pricing>` more examples, including an approach to reinstatements.

.. _2_agg_class_note_clause:

The ``note`` Clause
---------------------

An optional note or comment on the distribution. Can include hints for computation::

    note{US Prems Ops, light hazard severity; for ABC account; recommend:- log2:16, bs:1/32}


Example ``aggregate`` programs
------------------------------

Here are four illustrative examples. The line must start with ``agg``
(no tabs or spaces first) but afterwards spacing within the specification is
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

See `test suite programs`_ for more examples using the built-in test suite.

From here go to
-----------------

#. :doc:`2_x_exposure`

#. :doc:`2_x_limits`

#. :doc:`2_x_severity`

#. :doc:`2_x_frequency`

#. :doc:`2_x_mixtures`

#. :doc:`2_x_vectorization`

#. :doc:`2_x_reinsurance` and :doc:`2_x_re_pricing`

#. :doc:`../4_agg_Language_Reference`

