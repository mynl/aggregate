.. _2_x_exposure:

Specifying Exposure
======================


**Objectives.**  Describe the frequency distributions available in ``aggregate``.

**Audience.** User who wants to build an aggregate with a range of frequency distributions.

**Prerequisites.** Building aggregates using ``build``. Using ``scipy.stats``. Probability theory behind discrete distributions, especially mixed-Poisson distributions and processes.

**See also.** :ref:`Severity <2_x_severity>`, :ref:`aggregate <2_x_aggregate>`, :ref:`agg language <2_x_agg_language>`.



Determining Expected Claim Count
--------------------------------

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



Exposure: the Volume of Insurance
----------------------------------

:ref:`As already discussed <2_agg_class_exposure_clause>` exposure can be specified in three ways:

::

       123 claims
       1000 loss
       1000 premium at 0.7 lr

When using the ``agg`` language this choice presents no ambiguity. But if you create the
object directly, the input arguments could conflict. Here is the order in which the
exposure arguments are used:

* If ``count`` is given it is used and loss is derived from severity.
* Else if ``loss`` is given, then count is derived from the severity.
* Else if ``premium at xx lr`` are given, then the loss is derived by multiplication and counts from severity.
* In all cases, if ``premium`` is given the loss ratio is computed

.. distributions.py about line 880

In terms of ``exp_en``, ``exp_el``, ``exp_premium``, and ``exp_lr`` the first step is:

    exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)

i.e., expected losses are used if given and premium times loss ratio used if not. All these values default to 0.
At this point the object must know either loss or claim count::

    assert np.all( exp_el > 0 or exp_en > 0 )

Then

* If ``exp_en`` is input, it determines the expected claim count; expected losses determined from expected severity
* Else if ``exp_el > 0`` then it is used as expected loss and claim count determined from severity

Finally,

* If ``exp_prem > 0`` then the the loss ratio is computed
* Else if ``exp_lr > 0`` the premium is computed

Thus, if only ``exp_en`` or ``exp_loss`` is entered, the object knows loss, but not premium or loss ratio.

As usual, all of these values can be vectorized.

Conditional and Unconditional Severity
--------------------------------------

By default, claim count is conditional  on a loss to the layer, meaning that :math:`X` is the ground up severity.
Thus :math:`X \mid X > \mathit{attachment}` generates the input number of claims.
Conditional severity can have a mass at zero.

Unconditional severity with a mass at zero lowers the effective claim count.
It is specified by adding ``!`` after the severity distribution.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

    cond = build('agg Conditional   10 claims 10 x 10 sev 5 * expon   poisson')
    uncd = build('agg Unconditional 10 claims 10 x 10 sev 5 * expon ! poisson')
    qd(cond.describe)
    qd(uncd.describe)
    print(cond.sevs[0].sf(10), uncd.agg_m / cond.agg_m)


