.. _2_x_ir_pricing:

.. reviewed 2022-12-24
.. NEEDS WORK

Individual Risk Pricing
==========================

**Objectives:** Use ``aggregate`` to determine the loss pick and technical premium for large account structures, including aggregate charges for limited losses (called Table L and M charges in the US).

**Audience:** Individual risk large account pricing, broker, or risk retention actuary.

**Prerequisites:** DecL, underwriting and insurance terminology, aggregate distributions, risk measures.

**See also:** :doc:`2_x_re_pricing`, :doc:`DecL/080_reinsurance`. For other related examples see :doc:`2_x_problems`, especially :doc:`problems/0x0_bahnemann`.


**Contents:**

#. :ref:`Helpful references <ir references>`
#. :ref:`Modes of Large Account Analysis`
#. :ref:`ir stop loss`


The examples in this section are illustrative. ``aggregate`` gives you the gross, ceded, and net distributions and with those in hand, you are off to the analytic races. You can answer any reasonable question about a large account program.

.. _ir references:

Helpful References
--------------------

* Fisher study note, Bahnemann, :cite:t:`Fisher2019`, :cite:t:`Bahnemann2015`
* WCIRB Table L
* ISO retro rating plan
* CAS Exam 8 readings

.. Table M and Table L!
.. https://www.wcirb.com/content/california-retrospective-rating-plan
.. ISO Retro Rating Plan
.. Fisher et al case study spreadsheet...

Modes of Large Account Analysis
--------------------------------

IR pricing is based on an estimated loss pick, possibly supplemented by distribution and volatility statistics such as loss standard deviation or quantiles. ``aggregate`` can compute the loss pick for specific covers (analogous to an excess of loss treaty) and aggregate covers (analogous to a treaty with an aggregate limit and deductible). The former application is peripheral to the underlying purpose of ``aggregate``, but is very convenient nonetheless, while the latter is a showcase application.

The IR, or its risk retention vehicle, is concerned with the loss pick and understanding the impact of the insurance contract on the distribution of retained losses. IR pricing actuaries often want the full gross and retained (net, after insurance) outcome distributions.


.. _ir stop loss:

Large Deductible Stop Loss Insurance (Table L and M)
---------------------------------------------------------------

*Documentation to follow.*
