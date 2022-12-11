.. _2_x_mixtures: 

Mixed Severity Distributions
-------------------------------

**Prerequisites:**  Examples use ``build`` and ``qd``, and basic :class:`Aggregate` output.


The variables in the severity clause (scale, location, distriution ID, shape
parameters, mean and CV) can be vectors to create a **mixed severity**
distribution. All elements are broadcast against one-another. For example::

   sev lognorm 1000 cv [0.75 1.0 1.25 1.5 2] wts [0.4, 0.2, 0.1, 0.1, 0.1]

expresses a mixture of five lognormals with a mean of 1000 and CVs as
indicated with weights 0.4, 0.2, 0.1, 0.1, 0.1. Equal weights are expressed
as wts=[5], or the relevant number of components. A missing weights clause is
interpreted as giving each severity weight 1 which results in five times the
total loss. Commas in the lists are optional.

**Example.**

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

    eg = build('agg Eg '
               '1000 loss '
               '1000 xs 0 '
               'sev lognorm 100 cv [0.75 1.0 1.25 1.5 2] '
               'wts [0.4, 0.2, 0.1, 0.1, 0.1] '
               'poisson')
    qd(eg)
    qd(eg.report_df.iloc[:, :-4])

Mixed severity with Poisson frequency is the same as the sum of five independent components. The ``report_df`` shows the mixture details.

.. ipython:: python
    :okwarning:

    egPort = build('port Eg.Port '
                        'agg Unit1 400 loss 1000 xs 0 sev lognorm 100 cv 0.75 poisson '
                        'agg Unit2 200 loss 1000 xs 0 sev lognorm 100 cv 1.00 poisson '
                        'agg Unit3 100 loss 1000 xs 0 sev lognorm 100 cv 1.25 poisson '
                        'agg Unit4 100 loss 1000 xs 0 sev lognorm 100 cv 1.50 poisson '
                        'agg Unit5 100 loss 1000 xs 0 sev lognorm 100 cv 2.00 poisson ')
    qd(egPort)

Actual frequency equals total frequency times weight. Setting ``wts=5`` results in equal weights, here 0.2.

.. ipython:: python
    :okwarning:

    eg2eq = build('agg Eg.eq '
                  '1000 loss '
                  '1000 xs 0 '
                  'sev lognorm 100 cv [0.75 1.0 1.25 1.5 2] wts=5 '
                  'poisson')
    qd(eg2eq)

Missing weights set to 1 resulting in five times loss. This behavior is generally not what you want!

.. ipython:: python
    :okwarning:

    eg2m = build('agg Eg.m '
                  '1000 loss '
                  '1000 xs 0 '
                  'sev lognorm 100 cv [0.75 1.0 1.25 1.5 2] '
                  'poisson')
    qd(eg2m)


.. _med example:

Example: Mixed Exponential Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mixed exponential distribution (MED) is used by major US rating
bureaus to model severity and compute increased limits factors (ILFs).
This example explains how to create a MED in ``aggregate``. The
distribution is initially created as an ``Aggregate`` object with a degenerate
frequency identically equal to 1 claim to focus on the severity.
We then explain how frequency mixing interacts with a mixed severity.

The next table of exponential means and weights appears on slide 24 of `Li
Zhu, Introduction to Increased Limits Factors, 2011 RPM Basic Ratemaking
Workshop,
<https://www.casact.org/sites/default/files/presentation/rpm_2011_handouts_ws1-zhu.pdf>`_,
titled a “Sample of Actual Fitted Distribution”. At the time, it was a
reasonable curve for US commercial auto. We will use these means and
weights.

========== ==========
**Mean**   **Weight**
========== ==========
2,763      0.824796
24,548     0.159065
275,654    0.014444
1,917,469  0.001624
10,000,000 0.000071
========== ==========

Here the DecL to create this mixture. Note: currently, it is necessary to
enter a dummy shape parameter 1 for the exponential, even though it does not
take a shape.

.. ipython:: python
    :okwarning:

    med = build('agg MED 1 claim '
                'sev [2.764e3 24.548e3 275.654e3 1.917469e6 10e6] * '
                'expon 1 wts [0.824796 0.159065 0.014444 0.001624, 0.000071] '
                'fixed')
    qd(med)

The exponential distribution is surprisingly thick-tailed. It can be
regarded as the dividing line between thin and thick tailed distributions.
In order to achieve good accuracy, the modeling increases the number of
buckets to :math:`2^{18}` (i.e., ``log2=18``) and uses a bucket size ``bs=500``.
The dataframe ``report_df`` is a more detailed version of the audit dataframe
that includes information from ``statistics_df`` about each severity component.
(The reported claim counts are equal to the weights and cannot be interpreted
as fixed frequencies. They can be regarded as frequencies for a Poisson or
mixed Poisson.)

.. ipython:: python
    :okwarning:

    med.update(log2=18, bs=500)
    qd(med)

The middle diagnostic plot, the log density, shows the mixture components.

.. ipython:: python
    :okwarning:

    @savefig mixtures1.png
    med.plot()

The ``density_df`` dataframe includes a column ``lev``. From this we can pull out ILFs.
Zhu reports the ILF at 1M equals 1.52.

.. ipython:: python
    :okwarning:

    print(med.density_df.loc[1000000, 'lev'] / med.density_df.loc[100000, 'lev'])

    # graph of all ILFs
    base = med.density_df.loc[100000, 'lev']
    
    @savefig mixtures2.png
    ax = (med.density_df.lev / base).plot(xlim=[-100000,10.1e6], ylim=[0.9, 1.85],
                                          figsize=(3.5, 2.45))
    ax.grid(lw=.25, c='w')
    ax.set(xlabel='Limit', ylabel='ILF', title='Pure loss ILFs relative to 100K base');


Saving to the Knowledge
~~~~~~~~~~~~~~~~~~~~~~~~~~

We can save the MED severity in the knowledge and then refer to it by name.

.. ipython:: python
    :okwarning:

    build('sev COMMAUTO [2.764e3 24.548e3 275.654e3 1.917469e6 10e6] * '
          ' expon 1 wts [0.824796 0.159065 0.014444 0.001624, 0.000071]');

    lim_prof2 = build('agg LIM_PROF2 [20 8 4 2] claims [1e6, 2e6 5e6 10e6] xs 0 '
                      'sev sev.COMMAUTO fixed',
                      log2=18, bs=500)

    qd(lim_prof2)

Different Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The kind of distribution can vary across mixtures. In the following, exposure varies
for each curve, rather than using weights, see :doc:`070_vectorization`.

Using a Delaporte (shifted) gamma mixing often produces more realistic output
than a gamma, avoiding very good years.

.. ipython:: python
    :okwarning:

    dd = build('agg DD [100 200] claims '
                '5000 x 0 '
                'sev [gamma lognorm] [100 150] cv [1 0.5] '
                'mixed gamma 0.5',
                log2=16, bs=2.5)
    qd(dd.report_df)
    dd2 = build('agg DD.del [100 200] claims '
                 '5000 x 0 '
                 'sev [gamma lognorm] [100 150] cv [1 0.5] '
                 'mixed delaporte 0.5 0.6',
                log2=18, bs=2.5)
    qd(dd2.report_df)
    @savefig mix_3.png
    dd.plot()
    @savefig mix_4.png
    dd2.plot()



Severity Mixtures and Mixed Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All severity components in an aggregate share the same frequency mixing
value, inducing correlation between the parts. This is where the extra columns in
``report_df`` are used. In order to focus on the mixing and ease the computational
burden, apply a 500,000 policy limit to model a self-insured retention.
Assume a claim count of 10 claims, typical for a
small account (say, ABC). Commercial auto has parameter uncertainty CV around 25%.
The bucket size was selected by trial and error; the recommendation was 80, which
is too low.

.. ipython:: python
    :okwarning:

    med_po = build('agg ABC.Account.Po 50 claim '
                    '500000 xs 0 sev sev.COMMAUTO '
                    'poisson'
                    , bs=250)
    med_mx = build('agg ABC.Account.Po 50 claim '
                    '500000 xs 0 sev sev.COMMAUTO '
                    'mixed gamma 0.25'
                    , bs=250)
    qd(med_po)
    qd(med_mx)
    qd(med_mx.report_df.drop(['name']).iloc[:, :-2])


.. tidy up

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    plt.close('all')
