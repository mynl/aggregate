.. _2_x_case_studies:

Case Studies
===================

**Objectives:** Reproduce the case study exhibits from the book `Pricing
Insurance Risk
<https://www.wiley.com/en-us/Pricing+Insurance+Risk:+Theory+and+Practice-p-9781119755678>`_
(PIR) by Mildenhall & Major.

**Audience:** Capital modeling and corporate strategy actuaries; readers of PIR.

.. warning::

   The ``CaseStudy`` machinery and the ``Tame``/``CNC``/``HuSCS``/``Discrete``
   runner scripts have been **removed from this package** as of 1.0.0a12.

   To reproduce the book exhibits exactly as published, install the legacy
   release in an isolated environment::

         pip install aggregate==0.30.1

   The legacy ``aggregate.extensions.case_studies``, ``cnc``, ``discrete``,
   ``hs``, and ``tame`` modules and their HTML exhibit pipeline are
   preserved there. See the 0.30.1 documentation for the original
   ``CaseStudy`` workflow.


PIR Case Studies
--------------------

PIR presents four case studies that show how different methods price
business. Each describes a one-period insurer with two units (one riskier
than the other, usually reinsured), and compares unit statistics and pricing
on gross and net bases under many methods.

Simple Discrete Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Simple Discrete Example, :math:`X_1` takes values 0, 8, or 10, and
:math:`X_2` takes 0, 1, or 90 (independent; nine outcomes). Probabilities
are 1/2, 1/4, 1/4 for each marginal.

Tame Case Study
~~~~~~~~~~~~~~~~~~~

Two predictable units with no catastrophe exposure. Aggregate reinsurance
applies to the more volatile unit at an aggregate attachment.

Catastrophe and Non-Catastrophe (CNC) Case Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Catastrophe and non-catastrophe exposures. Models, e.g., whether to add a
new line. Aggregate reinsurance on the Cat unit at 0.1 attachment / 0.005
exhaustion probabilities.

Hurricane/Severe Convective Storm (HuSCS) Case Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hurricane and SCS exposures. Per-occurrence reinsurance on the Hu unit at
0.05 attachment / 0.005 exhaustion probabilities.

See PIR for the full description of each case and the published exhibits.
