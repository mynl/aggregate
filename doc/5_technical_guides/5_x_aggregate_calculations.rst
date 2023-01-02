.. _aggregate_calculations:

:class:`Aggregate` Class Calculations
======================================

**Objectives:** Describe calculations performed by the :class:`Aggregate` class.

**Audience:**

**Prerequisites:** DecL, general use of ``aggregate``, probability.

**See also:** :doc:`5_x_portfolio_calculations`, :doc:`5_x_distortions`, , :doc:`../2_user_guides/2_x_10mins`.


**Contents:**

* :ref:`Helpful References`
* :ref:`agg calcs`

Helpful References
--------------------

* :cite:t:`PIR`

.. _agg calcs:

Calculations
-------------

.. todo::

    Discussion to follow.

.. Calculations made by ``Aggregate`` objects.


  * Bucket shift (vs. more complex methdods - KPW/LDA) - not a big deal
  * Sum to 1 (to normalize or not to normalize?) huge issue: can't cumsum for levs; convol messes up
  * Left tail right tail S vs F computation of diffs
  * Cum sum as approx to ?Simpson half h rule integrals

  See :ref:`FFT Convo <2_x_fft_convolution>`.
