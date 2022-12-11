.. _design and purpose:

DecL Design and Purpose
------------------------

The Dec Language, or simply DecL, is designed to make it easy to go from "Dec page to distribution" --- hence the name. An insurance policy's Declarations page spells out key coverage terms and conditions such as the limit and deductible, effective date, named insured, and covered property. A reinsurance slip performs the same functions.

Coverage expressed concisely in words on a Dec page is often incomplete and is hard to program. Consider

    A trucking policy with a premium of 5000, a limit of 1000, and a retention 50.

To estimate the distribution of aggregate loss outcomes for this policy, the actuary must:

#. Estimate the priced loss ratio on the policy to determine the loss pick (expected loss) as premium times loss ratio. Say they select 65%.
#. Select a suitable trucking ground-up severity curve, say lognormal with mean 50 and CV 1.75.
#. Compute the expected conditional layer severity for the layer 1000 xs 50.
#. Divide severity into the loss pick to determine the expected claim count.
#. Select a suitable frequency distribution, say Poisson.
#. Calculate a numerical approximation to the resulting compound-Poisson aggregate distribution

A DecL program, which can be built with ``aggregate``, takes care of many of these details. The program corresponding to the trucking policy is simply::

    agg Trucking
        5000 premium at 65% lr loss
        1000 xs 50
        sev lognorm 50 1.75
        poisson

It specifies the loss ratio and distributions selected in steps 1, 2 and 5; these require actuarial judgment and cannot be automated. Based on this input, the ``aggregate`` package computes the rest of steps 1, 3, 4, and 6. The details of the program are explained in the rest of this chapter.


Specifies a Realistic Aggregate Distribution
----------------------------------------------

The trucking example hints at the complexity of specifying a realistic insurance aggregate distribution. Abstracting the details, a complete specification has seven parts

.. _seven clauses:

1. A label
2. The exposure, optionally including occurrence limits and deductibles
3. The ground-up severity distribution
4. Occurrence reinsurance (optional)
5. The frequency distribution
6. Aggregate reinsurance (optional)
7. Additional notes (optional)

DecL follows this pattern and specifies an aggregate distribution using :ref:`seven clauses <seven clauses>`

|    agg label
|        exposure <limit>
|        severity
|        <occurrence re>
|        <frequency>
|        <aggregate re>
|        <note>

where <clause> is optional. All programs are one line long and horizontal white space is ignored. The program is built (interpreted) using the ``build`` function.
Python automatically concatenates strings between parenthesis, so it is easiest and clearest to enter a program as::

    build('agg Trucking '
          '5000 premium at 65% lr 1000 xs 50 '
          'sev lognorm 50 1.75 '
          'poisson')

The entries in this example are as follows.


* ``agg`` is the DecL keyword used to create an aggregate distribution. Keywords are part of the language, like ``if/then/else`` in VBA, R or Python, or ``select`` in SQL.

* ``Trucking`` is a string label. It can contain letters and numbers and periods and must start with a letter. It is case sensitive. It cannot contain an underscore. It cannot be a DecL keyword. E.g., ``Motor``, ``NE.Region``, ``Unit.A`` but not ``12Line`` or ``NE_Region``.

* The exposure clause is ``5000 premium at 65% lr 1000 xs 50``. It determines the volume of insurance, see :doc:`020_exposure`. It includes ``1000 xs 50``, an optional :ref:`layers subclause<2_agg_class_layers_subclause>` to set policy occurrence limits and deductibles.

* The severity clause ``sev lognorm 50 1.75`` determines the *ground-up* severity, see :ref:`severity <2_agg_class_severity_clause>`. ``sev`` is a keyword


* The ``frequency`` clause, ``poisson``, specifies the frequency distribution, see :ref:`frequency <2_agg_class_frequency_clause>`.

The occurrence re, aggregate re and note clauses are omitted. See :ref:`reinsurance <2_agg_class_reinsurance_clause>` and :doc:`090_notes`.

``build`` automatically computes the expected claim count from the premium, expected loss ratio, and average severity.

Python ``f``-strings allow variables to be passed into DecL programs, ``f'sev lognorm {x} cv {cv}``.

There are two other specifications for different situations::

    agg LABEL BUILTIN_AGG

    BUILTIN_AGG

These reference a distribution from the ``knowledge`` database.
``BUILTIN_AGG`` has the form ``agg.LABEL`` where ``agg`` identifies an aggregate object and ``LABEL`` refers to one that has already been built. For example, ``agg.Trucking`` or ``agg.Exampe1``.
The first format gives ``BUILTIN_AGG`` a new label and the second uses its existing label. See the :doc:`../../4_dec_Language_Reference`.

The rest of this Chapter describes the basic features of each clause.
