.. _2_x_frequency:

.. _2_agg_class_frequency_clause:

.. reviewed 2022-12-24


The Frequency Clause
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

    agg A dfreq [1 2 3] [.5 3/8 1/8] sev lognorm 50 cv 1.75

specifies a frequency distribution with outcomes 1, 2, or 3 occurring with probabilities 0.5, 0.375, and 0.125 respectively. Probabilities can be entered as decimals or fractions.

.. _parametric frequency:

Parametric Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parametric frequency distributions are supported. Remember that the exposure clause determines the expected claim count.

* ``poisson``, no additional parameters required.
* ``geometric``, no additional parameters required.
* ``fixed``, no additional parameters required, expected claim count must be an integer.
* ``bernoulli``, no additional parameters required; expected claim count must be :math:`\le 1`.
* ``binomial SHAPE``, the shape parameter sets :math:`p` and :math:`n=\mathsf{E}[N]/p`.
* ``neyman SHAPE`` (or ``neymana`` or ``neymanA``), the Neyman A
  Poisson-compound Poisson. The shape variable gives the average number of
  claimants per claim. See JKK and :cite:t:`Consul1973`.
* ``pascal SHAPE1 SHAPE2`` (the generalized Poisson-Pascal, see REF), where ``SHAPE1``
  gives the cv and ``SHAPE2`` the number of claims per occurrence.

**Example.**

::

    agg A 100 claims sev lognorm 50 cv 0.75 poisson
    agg A 100 claims sev lognorm 50 cv 0.75 mixed gamma 0.2

specifies a Poisson frequency.  and negative binomial frequency respectively. For the latter, frequency CV equals ``(1 + .2**2 * 100) ** .5 / 10 = 0.22361``.


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

    agg A 100 claims sev lognorm 50 cv 0.75 mixed gamma 0.2

specifies a negative binomial (gamma-mixed Poisson) frequency respectively. The  variance equals :math:`100\times (1 + 0.2^2 \times 100)` and the CV equals ``(1 + .2**2 * 100) ** .5 / 10 = 0.22361``.


.. warning::
    Fixed frequency will accept non-integer input, but will not return a distribution (it will have negative probabilities). Be careful!


Zero Modification and Zero Truncation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Not yet implemented*
