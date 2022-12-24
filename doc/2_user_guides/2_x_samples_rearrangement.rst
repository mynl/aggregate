.. _2_x_samples_rearrangement:

.. NEEDS WORK

Samples, Iman-Conover, and the Rearrangement Algorithm
=======================================================

**Objectives:**

**Audience:** Planning and strategy, ERM, capital modeling, risk management actuaries.

**Prerequisites:** DecL, aggregate distributions, risk measures.

**See also:** :doc:`../5_technial_guides/5_x_samples`,  :doc:`../5_technial_guides/5_x_iman_conover`, :doc:`../5_technial_guides/5_x_rearrangement``.

**Contents:**

#. :ref:`Helpful References`
#. :ref:`samp samp`
#. :ref:`samp ic`
#. :ref:`samp ra`

Helpful References
--------------------

* PIR chapter 14, 15.
* RA paper ref.
* IC paper ref.
* Mildenhall Agg Dist WP

See examples in /TELOS/Blog/agg/examples/IC_and_rearrangement.ipynb.

.. _samp samp:

Using Samples and the Switcheroo Trick
---------------------------------------

*Documentation to follow.*

There's a sneaky but effective way to add correlation. The idea is:

* Make a portfolio with independent lines as usual
* Pull a sample from each unit
* Shuffle the sample to induce the correlation you want using Iman-Conover.
  You don't have to use a normal copula.
* (Sneaky part): recompute :math:`\mathsf E[X_i \mid X]` functions with those
  from the sample.

From there, you can compute everything you need to use the natural allocation
because it works on the conditional expectations, not the actual sample. I
call it the switcheroo operation.

.. _samp ic:

Applying the Iman Conover Algorithm
---------------------------------------

In this section we implement :ref:`ic simple example` in ``aggregate``.

*Documentation to follow.*

.. _samp ra:

Applying the Re-Arrangement Algorithm
---------------------------------------

See :ref:`ra worked example`.
