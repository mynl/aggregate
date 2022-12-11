.. _agg:

Specifying an Aggregate Distribution
-------------------------------------

Aggregate distributions are specified using :ref:`seven clauses <seven clauses>`, entered in DecL as::

    agg label               \
        exposure <limit>    \
        severity            \
        <occurrence re>     \
        <frequency>         \
        <aggregate re>      \
        <note>

All programs are one line long and horizontal white space is ignored.
A backslash is a newline continuation (like Python) and is used only for readability. Python automatically concatenates strings between parenthesis, so it is often easiest and clearest to enter a program as::

    build('agg Trucking '
          '4500 loss 1000 xs 50 '
          'sev lognorm 50 1.75 '
          'poisson')

Python ``f``-strings allow variables to be passed into DecL. The entries are as follows.


* ``agg`` is the keyword used to create an aggregate distribution. Keywords are part of the language, like ``if/then/else`` in VBA, R or Python, or ``select`` in SQL.

* ``label`` (``Trucking`` in the prior example) is a string label. It can contain letters and numbers and periods and must start with a letter. It is case sensitive. It cannot contain an underscore. It cannot be a DecL keyword. E.g., ``Motor``, ``NE.Region``, ``Unit.A`` but not ``12Line`` or ``NE_Region``.

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

