.. _2_x_vectorization:

.. reviewed 2022-12-24

Vectorization: Limit Profiles and Mixed Severity
-------------------------------------------------------

**Prerequisites:**  Examples use ``build`` and ``qd``, and basic :class:`Aggregate` output.

Using a Limit Profile with a Mixed Severity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`065_limit_profiles` and :doc:`060_mixed_severity` can be combined. Each mixed severity is applied
to each limit profile component.

sub-components.


**Example.**

This example combines three limit bands and a severity with two mixture
components. It creates an aggregate with six severities. The ``report_df``
dataframe shows the components (transposed extract shown). The mixture weights apply to claim counts, since exposure is specified by number of expected claims.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    a11 = build('agg DecL:11 '
                '[10 20 30] claims '
                '[100 200 75] xs [0 50 75] '
                'sev lognorm 100 cv [1 2] wts [0.6 0.4] '
                'poisson')
    qd(a11)
    qd(a11.report_df.loc[['limit', 'attachment', 'freq_m',
      'agg_m', 'agg_cv']].T.iloc[:-4])


**Example.**

We can combine the mixed exponential from :ref:`med example`
with a limits profile.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    a12 = build('agg DecL:12 [20 8 4 2] claims [1e6, 2e6 5e6 10e6] xs 0 '
                     'sev [2.764e3 24.548e3 275.654e3 1.917469e6 10e6] * '
                     'expon 1 wts [0.824796 0.159065 0.014444 0.001624, 0.000071] fixed',
                     log2=18, bs=500)
    qd(a12)


The ``report_df`` shows all 20 components: 4 limits x 5 mixture components.

.. ipython:: python
    :okwarning:

    qd(a12.report_df.loc[['limit', 'attachment', 'freq_m',
      'agg_m', 'agg_cv']].T.iloc[:-4])




Circumventing Products: Modeling Multiple Units in One Aggregate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


When severity weights sum to one, the severity is treated as a mixture and all
exposure terms are broadcast against all severity terms in an outer product.

When severity weights are missing or sum to the number of severity
components (e.g., are all equal to 1) the result is an item by item
combination, circumventing the outer product.
There are two cases when this alternative is useful:

#. Two or more units each with a different severity, but with a shared mixing
   variable. For example, to model two units with expected losses 100 and
   200, one with a gamma mean 10 CV 1 severity and the other lognormal mean
   15 CV 1.5 and both share a gamma mixing variable::

      agg MixedPremReserve                     \
      [100 200] claims                         \
      sev [gamma lognorm] [10 15] cv [1 1.5]   \
      mixed gamma 0.4

   The result should be the two-way combination, not the four-way exposure and
   severity product.

#. Exposures with different limits may have different severity curves. Again,
   the limit profile and severity curves should all be broadcast together at
   once, rather than broadcasting limits and severities separately and then
   taking the outer product::

      agg Eg4                                     \
      [10 10 10] claims                           \
      [1000 2000 5000] xs 0                       \
      sev lognorm [50 100 150] cv [0.1 0.15 0.2]  \
      poisson


**Example.**

The next two examples illustrate the different behavior.

#. Two units with different limits and severities and no weights.  ``report_df`` shows only two components modeled.

.. ipython:: python
   :okwarning:

   a13 = build('agg DecL:13 '
              '[10 20] claims '
              '[1000 2000] xs 0 '
              'sev [gamma lognorm] [10 15] cv [1 1.5] '
              'mixed gamma 0.4 ')
   qd(a13)
   qd(a13.report_df.loc[['limit', 'attachment', 'freq_m',
      'agg_m', 'agg_cv']].T.iloc[:-4])


#. Adding weights results in a mixed severity, 80% for the gamma and 20% for
   lognormal. Now ``report_df`` shows that each limit band is combined with
   each severity, resulting in four modeled components.

.. ipython:: python
   :okwarning:

   a14 = build('agg DecL:14 '
              '[10 20] claims '
              '[1000 2000] xs 0 '
              'sev [gamma lognorm] [10 15] cv [1 1.5] '
              'wts [.8 .2] '
              'mixed gamma 0.4 ')
   qd(a14)
   qd(a14.report_df.loc[['limit', 'attachment', 'freq_m',
      'agg_m', 'agg_cv']].T.iloc[:-4])

