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

directly specifies the frequency distribution. The ``outcomes`` and ``probabilities`` are specified as in :ref:`nonparametric severity`. There is no need for a frequency clause at the end.


**Examples.**

::

    agg A dfreq [1 2 3] [.5 3/8 1/8] sev lognorm 50 cv 1.75
    agg A dfreq [1 2 3] [.5 3/8 1/8] dsev [1:11]

The first specifies a frequency distribution with outcomes 1, 2, or 3 occurring with probabilities 0.5, 0.375, and 0.125 respectively. Probabilities can be entered as decimals or fractions. The second combines a non-parametric frequency and severity.

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
  claimants per claim. See JKK and :cite:t:`Consul1973a`.
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

A frequency clause can be followed by ``zm P0`` (zero modified) or ``zt`` (zero
truncated, the special case ``P0 = 0``)::

    <freq clause> zm P0
    <freq clause> zt

These build the :math:`(a,b,1)` class: the base distribution is held fixed and
its positive-count probabilities are rescaled so that :math:`\mathsf{P}(N=0)`
takes the requested value :math:`p_0^M`,

.. math::

    p_k^M = \frac{1 - p_0^M}{1 - p_0}\, p_k, \quad k \ge 1,
    \qquad
    G^M(z) = p_0^M + \frac{1 - p_0^M}{1 - p_0}\,[G(z) - p_0].

Supported for ``poisson``, ``binomial``, ``negbin``, ``geometric`` and
``logarithmic``. Every :math:`p_0^M` in :math:`[0, 1)` is admissible: above the
natural :math:`p_0` this *inflates* the zero mass (the classic excess-zeros
model), below it *deflates* it.

.. important::

    **The exposure clause states the un-modified (base) mean, and the
    modification moves it.** Rescaling by :math:`(1-p_0^M)/(1-p_0)` changes the
    mean --- that is what a zero modification is *for* --- so
    :math:`\mathsf{E}(N)` is an **output**, not the number you typed. This is
    the parameterization used by :cite:t:`Klugman2012` §6.6,
    :cite:t:`Frees2018a` ch. 2, and R's ``actuar`` (``dzmpois(x, lambda, p0)``
    takes the base ``lambda``), so textbook problems transcribe directly.

.. ipython:: python
    :okwarning:

    zm = build('agg DecL:ZM 4 claims dsev [1] poisson zm 0.5')
    zm.frequency.base_mean, zm.n, zm.frequency.prob_eq_0

The base Poisson has mean 4; reweighting to :math:`p_0^M=0.5` leaves
:math:`\mathsf{E}(N) = 0.5 \times 4 / (1 - e^{-4}) = 2.0373`. Zero truncation
pushes the other way:

.. ipython:: python
    :okwarning:

    zt = build('agg DecL:ZT 4 claims dsev [1] poisson zt')
    zt.frequency.base_mean, zt.n

To pin the mean instead, append ``!`` --- the same *unconditional* marker used
by ``sev`` and ``dsev``, and the realized count mean is the unconditional one.
``aggregate`` then solves for the base mean whose realized :math:`\mathsf{E}(N)`
equals the exposure clause:

.. ipython:: python
    :okwarning:

    pin = build('agg DecL:ZMPin 4 claims dsev [1] poisson zm 0.5 !')
    pin.frequency.base_mean, pin.n

Use ``!`` when the exposure clause states **money** --- ``1000 loss``,
``1000 premium at 0.65 lr``, ``100 exposure at 0.05 rate`` --- since a shifted
mean would silently miss that target. Without it those forms raise a
:class:`~aggregate.constants.ZeroModifiedExposureWarning` naming the shortfall.
The plain ``n claims`` form is silent: a count in and a shifted count out is the
documented default.

Not every mean is reachable when pinning. A zero-truncated count has at least
one claim, so its mean always exceeds 1; more generally
:math:`\mathsf{E}(N) > 1 - p_0^M`. Asking for less raises, quoting the bound.

.. warning::

    No member of the :math:`(a,b,1)` class is preserved under a change of
    exposure :cite:p:`Klugman2012` §7.4 --- a sum of zero-modified counts is a
    *compound* distribution, not a zero-modified one. Zero modification is a
    per-risk, per-period model: it describes one policy that may never claim
    (duplicate coverage, a no-claims-discount incentive to self-report), or
    corrects a claims database that records no zero rows. In practice it is
    fitted at the individual-risk level in ratemaking regressions
    :cite:p:`Boucher2007`, not used as a portfolio-level aggregate frequency.
    If you need a count that scales with exposure *and* places large mass at
    zero, use a compound model instead.

**Shift helpers.** The two directions are public on
:class:`~aggregate.distributions.Frequency`:
:meth:`~aggregate.distributions.Frequency.modify_mean` maps a base mean to the
realized :math:`\mathsf{E}(N)` (closed form), and
:meth:`~aggregate.distributions.Frequency.solve_base_mean` inverts it --- what
``!`` calls internally. :meth:`~aggregate.distributions.Frequency.apply_deductible`
implements the Loss Models §8.6 thinning map from a loss count to a payment
count; see the :ref:`zero-modified Poisson/Burr example <lda zmpoisson burr>`.

