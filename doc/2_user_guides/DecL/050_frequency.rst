.. _2_x_frequency:

.. _2_agg_class_frequency_clause:

The ``frequency`` Clause
-------------------------

The exposure and severity clauses determine the expected claim count. The ``frequency`` clause specifies the other particulars of the claim count distribution. As with severity, the syntax is different for non-parametric and parametric distributions.

.. _nonparametric frequency:

Non-Parametric Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An exposure clause::

    dfreq [outcomes] <[probabilities]>

directly specifies the frequency distribution. The ``outcomes`` and ``probabilities`` are specified as in :ref:`nonparametric severity`.


**Example.**

::

    agg A dfreq [9 10 11] [.5 .25 .25] sev lognorm 50 cv 1.75


.. _parametric frequency:

Parametric Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parametric frequency distributions are supported. Remember that the exposure clause determines the expected claim count.

* ``poisson``, no additional parameters required
* ``geometric``, no additional parameters required
* ``fixed``, no additional parameters required, expected claim count must be an integer.
* ``bernoulli``, no additional parameters required, expected claim count must be :math:`\le 1`.
* ``binomial SHAPE``, the shape determines :math:`p` and :math:`n=\mathsf{E}[N]/p`.
* ``neyman SHAPE`` (nor ``neymana`` or ``neymanA``), the Neyman A
  Poisson-compound Poisson. The shape variable gives the average number of
  claimants per claim. See JKK. REF
* ``pascal SHAPE1 SHAPE2`` (the generalized Poisson-Pascal, see REF), where ``SHAPE1``
  gives the cv and ``SHAPE2`` the number of claims per occurrence.

**Example.**

::

    agg A 10 claims sev lognorm 50 cv 1.75 poisson


Mixed-Poisson Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :math:`G`-mixed Poisson frequency (see :ref:`mixed frequency distributions`), where :math:`G` has expectation 1, can be specified using the ``mixed`` keyword, followed by the name and shape parameters of the mixing distribution::

    mixed DIST_NAME SHAPE1 <SHAPE2>

``SHAPE1`` specifies cv of the mixing distribution. The following mixing distributions are supported:

* ``gamma SHAPE1`` is a gamma-Poisson, i.e., negative binomial. Since the mix mean (shape times scale) equals one
  :math:`\alpha\beta=1` and hence the mix variance equals :math:`c:=\alpha=(cv)^{-2}`, which is sometimes called the contagion. The negative binomial variance equals :math:`n(1+cn)`.
* ``delaporte SHAPE1 SHAPE2``, a shifted gamma and the second parameter equals the proportion of certain claims (which determines a minimum claim count).
* ``ig SHAPE1`` the inverse Gaussian distribution
* ``sig SHAPE1 SHAPE2`` the shifted inverse Gaussian, parameter 2 as for Delaporte.
* ``beta SHAPE1`` a beta-Poisson with mean 1 and cv ``SHAPE1``. Use with caution.
* ``sichel SHAPE1 SHAPE2`` is Sichel's (generalized inverse Gaussian) distribution with ``SHAPE2`` equal to :math:`\lambda`.

    - ``sichel.gamma SHAPE1`` is the same as Delaporte
    - ``sichel.ig SHAPE1`` is the same as a shifted inverse Gaussian.


**Example.**

::

    agg 5 claims dsev [1] mixed gamma 0.16

produces a negative binomial (gamma-mixed Poisson) distribution with variance :math:`5\times (1 + 0.16^2 \times 5)`.

.. warning::
    Fixed frequency will accept non-integer input, but will not return a distribution (it will have negative probabilities). Be careful!


Zero Modification and Zero Truncation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not yet implemented.
