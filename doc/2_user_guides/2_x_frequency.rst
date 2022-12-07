.. _2_x_frequency:

DecL: Frequency Distributions
===============================

**Objectives:**  Describe the frequency distributions available in ``aggregate``.

**Audience:** User who wants to build an aggregate with a range of frequency distributions.

**Prerequisites:** Building aggregates using ``build``. Using ``scipy.stats``. Probability theory behind discrete distributions, especially mixed-Poisson distributions and processes.

**See also:** :ref:`Severity <2_x_severity>`, :ref:`aggregate <2_x_aggregate>`, :ref:`Dec language <2_x_dec_language>`.

Specifying Frequency
---------------------

Frequency is expressed using the ``exposure`` and ``frequency`` clauses, or directly with the ``dfreq`` keyword in the ``exposure`` clause::

    agg A 10 claims sev lognorm 50 cv 1.75 poisson

    agg A 10 claims sev lognorm 50 cv 1.75 mixed gamma 0.4

    agg A dfreq [9 10 11] [.5 .25 .25] sev lognorm 50 cv 1.75


Parametric Frequency
~~~~~~~~~~~~~~~~~~~~~~~

`Parametric Frequency Distributions`_ described the following parametric frequency distributions. In this case, the ``exposure`` clause determines the expected claim count.

* ``poisson``, no additional parameters required
* ``geometric``, no additional parameters required
* ``fixed``, no additional parameters required
* ``bernoulli``, expected claim count must be :math:`\le 1`.
* ``binomial SHAPE``, the shape determines :math:`p` and :math:`n=\mathsf{E}[N]/p`.
* ``pascal SHAPE1 SHAPE2`` (the generalized Poisson-Pascal, see REF), where ``SHAPE1``
  gives the cv and ``SHAPE2`` the number of claims per occurrence.
* ``neyman SHAPE`` (nor ``neymana`` or ``neymanA``), the Neyman A Poisson-compound Poisson. The shape variable gives the average number of claimants per claim. See JKK. REF

Mixed Frequency
~~~~~~~~~~~~~~~~

In addition, a :math:`G`-mixed Poisson frequency (where the random variable :math:`G` must have expectation 1) can be specified using the ``mixed`` keyword, followed by the name and shape parameters of the mixing distribution::

    mixed DIST_NAME SHAPE1 <SHAPE2>

.. check this is true!

``SHAPE1`` specifies cv of the mixing distribution.

The following mixing distributions are supported:

* ``gamma SHAPE1`` is a gamma-Poisson, i.e., negative binomial. Since the mix mean (shape times scale) equals one
  :math:`\alpha\beta=1` and hence the mix variance equals :math:`c:=\alpha=(cv)^{-2}`, which is sometimes called the contagion. The negative binomial variance equals :math:`n(1+cn)`.
* ``delaporte SHAPE1 SHAPE2``, a shifted gamma and the second parameter equals the proportion of certain claims (which determines a minimum claim count).
* ``ig SHAPE1`` the inverse Gaussian distribution
* ``sig SHAPE1 SHAPE2`` the shifted inverse Gaussian, parameter 2 as for Delaporte.
* ``beta SHAPE1`` a beta-Poisson with mean 1 and cv ``SHAPE1``. Use with caution.
* ``sichel SHAPE1 SHAPE2`` is Sichel's (generalized inverse Gaussian) distribution with ``SHAPE2`` equal to :math:`\lambda`.

    - ``sichel.gamma SHAPE1`` is the same as Delaporte
    - ``sichel.ig SHAPE1`` is the same as a shifted inverse Gaussian.


Non-Parametric Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``dfreq`` clause.

Zero Modification and Zero Truncation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Not yet implemented.
