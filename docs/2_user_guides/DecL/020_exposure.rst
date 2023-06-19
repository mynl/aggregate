.. _2_x_exposure:

.. reviewed 2022-12-24

The Exposure Clause
-------------------

The exposure clause has two parts: exposures and an optional layers sub-clause
described in :doc:`030_limits`. It specifies the volume of insurance. There
are five forms:

#.  Expected loss
#.  Premium and loss ratio
#.  Exposure and rate
#.  Claim count
#.  Using the ``dfreq`` keyword to enter the frequency distribution directly

**Examples**::

       1000 loss
       1000 premium at 0.7 lr
       5 exposure at 2000 rate
       123 claims
       dfreq [1 2 3] [3/4 3/16 1/16]


* ``1000 loss`` directly specifies expected loss. The claim count is derived
  from average severity. It is typical for an actuary to estimate the loss
  pick and select a severity curve, and then derive frequency.
* ``1000 premium at 0.7 lr`` directly specifies premium and a loss ratio.
  Expected losses equal the product. The claim count is again derived from
  severity. Actuaries often take plan premiums and apply loss ratio
  picks to determine losses, rather than starting with a loss pick. This
  idiom supports that approach.
* ``5 exposure at 2000 rate`` directly specifies exposure and a loss rate. It
  is analogous to the loss ratio form. Actuaries often know exposure and unit
  rates (per vehicle, per 100 insured value, per location). This idiom supports
  that approach.
* ``123 claims`` directly specifies the expected claim count; the last letter
  ``s`` on ``claims`` is optional, allowing ``1 claim``. Expected losses
  equal claim count times average severity.
* ``dfreq [1 2 3] [3/4 3/16 1/16]`` specifies frequency outcomes and
  probabilities directly. It is described in :ref:`nonparametric frequency`.

All values in the first three specifications can be vectorized, see :doc:`070_vectorization`.

Determining Expected Claim Count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variables are used in the following order to determine overall expected
losses.

* If ``count`` is given it is used and loss is derived from severity.
* Else if ``loss`` is given, then count is derived from the severity.
* Else if either ``pp premium at xx lr`` or ``ee exposure at rr rate`` is
  given, then the loss is derived by multiplication and counts from
  severity.
* In all cases, if ``premium`` is given the loss ratio is computed

These choices present no ambiguity when using DecL. But the input arguments
could conflict if you create the object directly.

By default, claim count is conditional on a loss to the layer, but severity can have a mass at zero. The severity can be specified to be unconditional, see :ref:`sev uncond sev`.

.. distributions.py about line 880

**Details.**

In terms of ``exp_en``, ``exp_el``, ``exp_premium``, and ``exp_lr`` the second and third steps are::

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


