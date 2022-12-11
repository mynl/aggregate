.. _2_x_vectorization:

DecL: Vectorization
=====================

**Objectives:** Describe how ``aggregate`` vectorizes exposure, limit, and severity parameters.  to catastrophe risk management, including calculation of occurrence and aggregate exceeding probability (OEP, AEP) values, and loss costs for ILWs.

**Audience:** Individual risk and reinsurance pricing actuaries, CAS Part 8 candidates.

**Prerequisites:** :ref:`Mixed severity distributions <2_x_mixtures>`, :ref:`limits <2_x_limits>`.

**See also:** :ref:`Reinsurance exposure rating <2_x_re_pricing>`.




Limit Profiles
---------------

The exposure variables can be vectors to express a *limit profile*. All
``exp_[en|prem|loss|count]`` related elements are broadcast against
one-another. For example

::

       [100 200 400 100] premium at 0.65 lr [1000 2000 5000 10000] xs 1000

expresses a limit profile with 100 of premium at 1000 x 1000; 200 at
2000 x 1000 400 at 5000 x 1000 and 100 at 10000 x 1000. In this case all
the loss ratios are the same, but they could vary too, as could the
attachments.



Limit Profiles and Mixtures
---------------------------

Limit profiles and mixtures can be combined. Each mixed severity is
applied to each limit profile component. For example

::

           ag = uw('agg multiExp [10 20 30] claims [100 200 75] xs [0 50 75]
               sev lognorm 100 cv [1 2] wts [.6 .4] mixed gamma 0.4')```

creates an aggregate with six severity subcomponents.

= ========= ============== ==========
# **Limit** **Attachment** **Claims**
= ========= ============== ==========
0 100       0              6
1 100       0              4
2 200       50             12
3 200       50             8
4 75        75             18
5 75        75             12
= ========= ============== ==========

Circumventing Products
----------------------

It is sometimes desirable to enter two or more lines each with a
different severity but with a shared mixing variable. For example to
model the current accident year and a run- off reserve, where the
current year is gamma mean 100 cv 1 and the reserves are larger
lognormal mean 150 cv 0.5 claims requires

::

           agg MixedPremReserve [100 200] claims \
             sev [gamma lognorm] [100 150] cv [1 0.5] \
             mixed gamma 0.4

so that the result is not the four-way exposure / severity product but
just a two-way combination. These two cases are distinguished looking at
the total weights. If the weights sum to one then the result is an
exposure / severity product. If the weights are missing or sum to the
number of severity components (i.e. are all equal to 1) then the result
is a row by row combination.
