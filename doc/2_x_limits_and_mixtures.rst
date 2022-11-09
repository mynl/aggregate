.. _2_x_limits_and_mixtures:

Limits, Limit Profiles, and Mixtures
=====================================


Limit Profiles
--------------

The exposure variables can be vectors to express a *limit profile*.
All ```exp_[en|prem|loss|count]``` related elements are broadcast against one-another.
For example ::

    [100 200 400 100] premium at 0.65 lr [1000 2000 5000 10000] xs 1000

expresses a limit profile with 100 of premium at 1000 x 1000; 200 at 2000 x 1000
400 at 5000 x 1000 and 100 at 10000 x 1000. In this case all the loss ratios are
the same, but they could vary too, as could the attachments.

Mixtures
--------

The severity variables can be vectors to express a *mixed severity*. All ``sev_``
elements are broadcast against one-another. For example ::

    sev lognorm 1000 cv [0.75 1.0 1.25 1.5 2] wts [0.4, 0.2, 0.1, 0.1, 0.1]

expresses a mixture of five lognormals with a mean of 1000 and CVs as indicated with
weights 0.4, 0.2, 0.1, 0.1, 0.1. Equal weights can be express as wts=[5], or the
relevant number of components.


Limit Profiles and Mixtures
---------------------------

Limit profiles and mixtures can be combined. Each mixed severity is applied to each
limit profile component. For example ::

    ag = uw('agg multiExp [10 20 30] claims [100 200 75] xs [0 50 75]
        sev lognorm 100 cv [1 2] wts [.6 .4] mixed gamma 0.4')```

creates an aggregate with six severity subcomponents.

+---+-------+------------+--------+
| # | limit | attachment | claims |
+===+=======+============+========+
| 0 | 100   |  0         |  6     |
+---+-------+------------+--------+
| 1 | 100   |  0         |  4     |
+---+-------+------------+--------+
| 2 | 200   | 50         | 12     |
+---+-------+------------+--------+
| 3 | 200   | 50         |  8     |
+---+-------+------------+--------+
| 4 |  75   | 75         | 18     |
+---+-------+------------+--------+
| 5 |  75   | 75         | 12     |
+---+-------+------------+--------+

Circumventing Products
----------------------

It is sometimes desirable to enter two or more lines each with a different severity but
with a shared mixing variable. For example to model the current accident year and a run-
off reserve, where the current year is gamma mean 100 cv 1 and the reserves are
larger lognormal mean 150 cv 0.5 claims requires ::

    agg MixedPremReserve [100 200] claims sev [gamma lognorm] [100 150] cv [1 0.5] mixed gamma 0.4

so that the result is not the four-way exposure / severity product but just a two-way
combination. These two cases are distinguished looking at the total weights. If the weights sum to
one then the result is an exposure / severity product. If the weights are missing or sum to the number
of severity components (i.e. are all equal to 1) then the result is a row by row combination.


For fixed or histogram, have to separate the parameter so they are not broad cast; otherwise
you end up with multiple lines when you intend only one



FROM THE SNIPPET

Snippet: Limits Profiles, Mixed Severity, and Frequency Mixing
--------------------------------------------------------------

The ``Aggregate`` distribution class manages creation and calculation of
aggregate distributions. It allows for very flexible creation of
aggregate distributions, including a limit profile, a mixed severity, or
both. Mixed frequency types share a mixing distribution across all
broadcast terms to ensure an appropriate inter- class correlation.

Limit profiles 2
~~~~~~~~~~~~~~~~

The exposure variables can be vectors to express a *limit profile*. All
``exp_[en|prem|loss|count]`` related elements are broadcast against
one-another. For example

::

   agg Eg1 [1000 2000 4000 1000] premium at 0.65 lr
   [1000 2000 5000 4000] xs [0 0 0 1000]
   sev lognorm 500 cv 1.25
   mixed gamma 0.6

expresses a limit profile with 1000 of premium at 1000 x 0; 2000 at 2000
x 0 4000 at 5000 x 0 and 1000 at 4000 x 1000. In this case all the loss
ratios are the same, but they could vary too.

Note that an aggregate with a mixed severity is a sum of aggregates,
with the mixture weights applied to the expected claim count. This is
analogous to the fact that :math:`\exp(a+b)=\exp(a)\exp(b)`. In terms of
compound Poisson,

.. math:: \mathsf{CP}(\lambda, \sum w_iF_i)=_d \sum_i \mathsf{CP}(w_i \lambda, F_i)

where :math:`=_d` indcates the two sides have the same distribution.

In this case we have selected a mixed frequency, using a gamma CV 0.6
mixing distribution. All of the limits share the same mixing variable.
The effect of this is shown in the report_df, comparing the independent
and mixed columns. The former adds the mixture components independently
whereas the latter uses the common mixing variable. The increase in
aggregate CV is quite marked.

.. code:: ipython3

    import aggregate as agg
    # uw appropriate for snippet
    build = agg.Underwriter(name='Mixtures', update=True, log2=16)
    build.logger_level(30)

.. code:: ipython3

    # limit profile
    eg1 = build('agg Eg1 [1000 2000 4000 1000] premium at 0.65 lr '
                '[1000 2000 5000 4000] xs [0 0 0 1000] '
                'sev lognorm 500 cv 1.25 '
                'mixed gamma 0.6')
    eg1.plot()
    eg1.report_df

Mixed severity distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The severity variables can be vectors to express a *mixed severity*. All
severity elements are broadcast against one-another. For example

::

   sev lognorm 1000 cv [0.75 1.0 1.25 1.5 2] wts [0.4, 0.2, 0.1, 0.1, 0.1]

expresses a mixture of five lognormals with a mean of 1000 and CVs as
indicated with weights 0.4, 0.2, 0.1, 0.1, 0.1. Equal weights can be
expressed using the shorthand ``wts=5``. A missing weights clause is
interpreted as giving each severity weight 1 which results in five times
the total loss.

.. code:: ipython3

    # mixed severity
    eg2 = build('agg Eg2 1000 loss sev lognorm 100 cv [0.75 1.0 1.25 1.5 2] wts [0.4, 0.2, 0.1, 0.1, 0.1] poisson')
    eg2.report_df

.. code:: ipython3

    # mixed severity with poisson frequency is the same as the sum of five independent components
    egPort = build('''
    port EgPort
        agg Unit1 400 loss sev lognorm 100 cv 0.75 poisson
        agg Unit2 200 loss sev lognorm 100 cv 1.00 poisson
        agg Unit3 100 loss sev lognorm 100 cv 1.25 poisson
        agg Unit4 100 loss sev lognorm 100 cv 1.50 poisson
        agg Unit5 100 loss sev lognorm 100 cv 2.00 poisson

    ''')
    egPort.report_df

.. code:: ipython3

    # actual frequency = total frequency x weight; wts=5 sets equal weights, here 0.2
    #
    eg2e = build('agg Eg2e 1000 loss sev lognorm 100 cv [0.75 1.0 1.25 1.5 2] wts=5 poisson')
    eg2e.report_df

.. code:: ipython3

    # missing weights set to 1 resulting in five times loss
    eg2m = build('agg Eg2m 1000 loss sev lognorm 100 cv [0.75 1.0 1.25 1.5 2] poisson')
    eg2m.report_df

Limit profiles and mixed severity 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Limit profiles and mixtures can be combined. Each mixed severity is
applied to each limit profile component. For example

::

   agg Eg3 [10 20 30] claims [100 200 75] xs [0 50 75] sev lognorm 100 cv [1 2] wts [.6 .4] mixed gamma 0.4

creates an aggregate with six severity subcomponents:

= ===== ========== ======
# limit attachment claims
= ===== ========== ======
0 100   0          6
1 100   0          4
2 200   50         12
3 200   50         8
4 75    75         18
5 75    75         12
= ===== ========== ======

.. code:: ipython3

    # limits profile and mixed severity
    eg3 = build('agg Eg3 [10 20 30] claims [100 200 75] xs [0 50 75] '
                'sev lognorm 100 cv [1 2] wts [0.6 0.4] '
                'poisson')
    display(eg3)
    display(eg3.report_df)
    eg3.plot()

Limit profiles with different severities: circumventing products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exposures with different limits may have different severity curves. In
this case, the limit profile and severity curves should all be broadcast
together, rather than broadcasting limits and severities separately and
then taking the outer product as in the previous example. To achieve
this omit the weight clause:

::

   agg Eg4 [10 10 10] claims [1000 2000 5000] xs 0 \
       sev lognorm [50 100 150] cv [0.1 0.15 0.2] \
       poisson

The interpretation is determined by the total weights. If the weights
sum to one then the severity is interpreted as a mixture, and the result
is an exposure / severity product as above. If the weights do not sum to
one, they are used to adjust the exposure. If the weights clause is
missing, then the weights are all set equal to 1 and the result is a
different severity for each limit band with the requested exposure. (If
the weights are specified and sum to the number of severity components
then they are used to adjust the expected losses. Usually, this is not
the desired behavior.) **TODO: what is wts sum to neither?**

.. code:: ipython3

    # limits profile where each limit band has a different severity curve
    eg4 = build('agg Eg4 [10 10 10] claims [1000 2000 5000] xs 0 '
                'sev lognorm [50 100 150] cv [0.1 0.15 0.2] '
                'poisson')
    eg4.report_df

.. code:: ipython3

    # adding weights that sum to the number of components adjusts expected losses
    eg4m = build('agg Eg4m [10 10 10] claims [1000 2000 5000] xs 0 '
                'sev lognorm [50 100 150] cv [0.1 0.15 0.2] wts [2 .5 .5]'
                'poisson')
    eg4m.report_df

Frequency mixing
~~~~~~~~~~~~~~~~

All severity components in an aggregate share the same frequency mixing
value, inducing correlation between the parts. For example, to model the
current accident year and prior year reserves.

::

   agg Egn [100 200] claims sev [gamma lognorm] [100 150] cv [1 0.5] mixed gamma 0.4

``Egn`` models the current accident year is gamma mean 100 cv 1 and a
run-off reserve lognormal mean 150 cv 0.5.

.. code:: ipython3

    # mixed frequency, negative binomial cv 0.4
    eg4x = build('agg Eg4x [1000 500 200 100] premium at [0.85 .75 .65 .55] lr '
                '[1000 2000 5000 10000] xs 1000 '
                'sev lognorm 100 cv .75 '
                'mixed gamma 0.4')
    eg4x.report_df

.. code:: ipython3

    # model of current AY (gamma) and reserves(lognormal) with shared gamma mixing
    eg5 = build('agg Eg5 [100 200] claims 5000 x 0 sev [gamma lognorm] [100 150] cv [1 0.5] mixed gamma 0.5',
               log2=16, bs=2.5)
    eg5.report_df

.. code:: ipython3

    # Delaporte (shifted) gamma mixing often produces more realistic output, avoiding very good years
    eg5d = build('agg Eg5d [100 200] claims 5000 x 0 sev [gamma lognorm] [100 150] cv [1 0.5] mixed delaporte 0.5 0.6',
                log2=18, bs=2.5)
    eg5d.report_df

.. code:: ipython3

    eg5.plot()

.. code:: ipython3

    eg5d.plot()

