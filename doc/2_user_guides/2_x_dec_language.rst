.. _2_x_dec_language:

The Dec Language
======================

**Objectives:** Introduce the Dec language (DecL) grammar.

**Audience:** User who wants to use it to build realistic aggregates.

**Prerequisites:** Familiar with building aggregates using ``build``. Probability theory behind aggregate distributions. Insurance and reinsurance terminology.

**See also:** :doc:`2_x_re_pricing`, and :doc:`../4_agg_Language_Reference`.

**Notation:** ``<item>`` denotes an optional term.


.. _design and purpose:

Design and Purpose
-------------------

The Dec language, or simply DecL, is designed to make it easy to go from "Dec page to distribution". In insurance policy's Dec, or Declarations, page spells out key coverage terms and conditions such as the limit and deductible, effective date, named insured, and covered XX. A reinsurance slip performs the same functions.

Coverage expressed concisely in words on a Dec page is hard to program. Consider

    A trucking policy with a premium of 5000, a limit of 1000, and a retention 50.

To estimate the distribution of aggregate loss outcomes the actuary must:

#. Estimate the priced loss ratio, say 65%, on the policy to determine the loss pick (expected loss) as premium times loss ratio.
#. Select a suitable trucking ground-up severity curve, say lognormal with mean 50 and CV 1.75.
#. Compute the expected conditional layer severity for the layer 1000 xs 50.
#. Divide severity into the loss pick to determine the expected claim count.
#. Select a suitable frequency distribution, say Poisson.
#. Calculate a numerical approximation to the resulting compound-Poisson aggregate distribution

The DecL program, built with ``aggregate``, takes care of many of these details. The DecL program for the trucking policy is simply::

    agg Trucking 5000 premium at 65% lr loss 1000 xs 50 sev lognorm 50 1.75 poisson

It specifies the loss ratio and distributions selected in steps 1, 2 and 5; these require actuarial judgment and cannot be automated. Based on this input, the ``aggregate`` package computes the rest of steps 1, 3, 4, and 6. The program and its parts are explained in the rest of this chapter.


Contents
---------

.. toctree::
    :maxdepth: 2

    DecL/010_Aggregate
    DecL/020_exposure
    DecL/030_limits
    DecL/040_severity
    DecL/050_frequency
    DecL/060_mixtures
    DecL/070_vectorization
    DecL/080_reinsurance
    DecL/080_notes


