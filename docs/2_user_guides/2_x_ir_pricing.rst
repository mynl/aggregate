.. _2_x_ir_pricing:

.. reviewed 2022-12-27

Individual Risk Pricing
==========================

**Objectives:** Applications of the :class:`Aggregate` class to individual risk pricing, including LEVs, ILFs, layering, and the insurance charge and savings (Table L, M), illustrated using problems from CAS Part 8.

**Audience:** Individual risk large account pricing, broker, or risk retention
actuary.

**Prerequisites:** DecL, underwriting and insurance terminology, aggregate
distributions, risk measures.

**See also:** :doc:`2_x_re_pricing`, :doc:`DecL/080_reinsurance`. For other
related examples see :doc:`2_x_problems`, especially :doc:`problems/0x0_bahnemann`. :ref:`ir stop loss` for theoretical background.


**Contents:**

#. :ref:`ir references`
#. :ref:`ir stop loss solution`
#. :ref:`ir summary`


The examples in this section are illustrative. ``aggregate`` gives the gross,
ceded, and net distributions and with those in hand, it is possible to answer
any reasonable question about a large account program.

.. _ir references:

Helpful References
--------------------

* :cite:t:`Fisher2019`
* :cite:t:`Bahnemann2015`
* Other CAS Part 8 readings.

.. WCIRB Table L
.. ISO retro rating plan
.. CAS Exam 8 readings

.. Table M and Table L!
.. https://www.wcirb.com/content/california-retrospective-rating-plan
.. ISO Retro Rating Plan
.. Fisher et al case study spreadsheet...


.. _ir stop loss solution:

Insurance Charge and Insurance Savings in :class:`Aggregate`
-----------------------------------------------------------------

Creating a custom table of insurance charges and savings, varying with account
size, specific occurrence limit, and entry ratio (aggregate limit) is very
easy using ``aggregate``. We will make a custom function to illustrate one
solution.

First, we need a severity curve. This step is very important, and would be
customized to the state and hazard group distribution of expected losses. We
use a simple mixture of a lognormal for small claims and a Pareto for large
claims, with a mean of about 25 (work in 000s). Create it as an object in the
knowledge using :meth:`build`. The parameters are selected judgmentally.


.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    mu, sigma, shape, scale, wt = \
        -0.204573975,  1.409431871, 1.633490596, 57.96737143, 0.742942461
    mean = wt * np.exp(mu + sigma**2 / 2) + (1 - wt) * scale / (shape - 1)
    build(f'sev IR:WC '
          f'[exp({mu}) {scale}] * [lognorm pareto] [{sigma} {shape}] '
          f'+ [0 {-scale}] wts [{wt} {1-wt}]');
    print(f'Mean = {mean:.1f} in 000s')

Second, we will build the model for a large account with 350 expected claims
and an occurrence limit of 100M. This model is used to set the update
parameters. Assume a gamma mixed Poisson frequency distribution with a mixing
CV of 25% throughout. The CV could be an input parameter in a production
application.

.. ipython:: python
    :okwarning:

    a01 = build('agg IR:Base '
                '350 claims '
                '100000 xs 0 '
                'sev sev.IR:WC '
                'mixed gamma 0.25 ',
                update=False)
    qd(a01)
    qd(a01.statistics.loc['sev', [0, 1, 'mixed']])

Look at the ``aggregate_error_analysis`` to pick ``bs`` (see :ref:`10 min agg
bucket`). Use an expanded number of buckets ``log2=19`` because the mixture
includes small mean lognormal and large mean Pareto components (some trial
and error not shown).

.. ipython:: python
    :okwarning:

    err_anal = a01.aggregate_error_analysis(19)
    qd(err_anal, sparsify=False)

Select ``bs=1/4`` as the most accurate from the displayed range (``
('rel', 'm')``). Update and plot. The plot shows the impact of the occurrence
limit in the extreme right tail.


.. ipython:: python
    :okwarning:

    a01.update(approximation='exact', log2=19, bs=1/4, normalize=False)
    qd(a01)
    @savefig ir_base.png
    a01.plot()

Third, create a custom function of account size and the occurrence limit, to
produce the :class:`Aggregate` object and a small table of insurance savings
and charges. Account size is measured by the expected ground-up claim count.
It should be clear how to extend this function to include custom severity,
different mixing CVs, or produce factors for different entry ratios. The
answer is returned in a ``namedtuple``.

.. ipython:: python
    :okwarning:

    from collections import namedtuple

    def make_table(claims=360, occ_limit=100000):
        """
        Make a table of insurance charges and savings by entry ratio for
        specified account size (expected claim count) and specific
        occurrence limit.
        """
        a01 = build(f'agg IR:{claims}:{occ_limit} '
                    f'{claims} claims '
                    f'{occ_limit} xs 0 '
                     'sev sev.IR:WC '
                     'mixed gamma 0.25 '
                    , approximation='exact', log2=19, bs=1/4, normalize=False)
        er_table = np.linspace(.1, 2., 20)
        df = a01.density_df
        ix = df.index.get_indexer(er_table * a01.est_m, method='nearest')
        df = a01.density_df.iloc[ix][['loss', 'F', 'S', 'e', 'lev']]
        df['er'] = er_table
        df['charge'] = (df.e - df.lev) / a01.est_m
        df['savings'] = (df.loss - df.lev) / a01.est_m
        df['entry'] = df.loss / a01.est_m
        df = df.set_index('entry')
        df = df.drop(columns=['e',  'er'])
        df.index = [f"{x:.2f}" for x in df.index]
        df.index.name = 'r'
        Table = namedtuple('Table', ['ob', 'table_df'])
        return Table(a01, df)


Finally, apply the new function to create some tables.

#. A small account with 25 expected claims, about 621K limited losses, and a
   low 50K occurrence limit. The output shows the usual ``describe``
   diagnostics for the underlying :class:`Aggregate` object, followed by a
   small Table across different entry ratios. The Table is indexed by entry
   ratio(aggregate attachment as a proportion of limited losses) and shows
   ``loss`` the aggregate limit loss level in currency units; the cdf and sf
   at that loss level (the latter giving the probability the aggregate layer
   attaches); the limited expected value at the entry ratio ``lev``; and the
   insurance charge(``1 - lev / loss``) and savings (``r - lev / loss``).

.. ipython:: python
    :okwarning:

    tl = make_table(25, 50)
    fc = lambda x: f'{x:,.1f}' if abs(x) > 10 else f'{x:.3f}'
    qd(tl.ob)
    qd(tl.table_df, float_format=fc, col_space=8)

2. The impact of increasing the occurrence limit to 250K:

.. ipython:: python
    :okwarning:

    tl2 = make_table(25, 250)
    qd(tl2.ob)
    qd(tl2.table_df, float_format=fc, col_space=8)

3. The impact of increasing the account size to 250 expected claims, still at
   250K occurrence limit:

.. ipython:: python
    :okwarning:

    tl3 = make_table(250, 250)
    qd(tl3.ob)
    qd(tl3.table_df, float_format=fc, col_space=8)

4. Finally, increase the occurrence limit to 10M:

.. ipython:: python
    :okwarning:

    tl4 = make_table(250, 10000)
    qd(tl4.ob)
    qd(tl4.table_df, float_format=fc, col_space=8)

These Tables all behave as expected. The insurance charge decreases with
increasing expected losses (claim count) and decreasing occurrence limit.

.. _ir summary:

Summary of Objects Created by DecL
-------------------------------------

Objects created by :meth:`build` in this guide.

.. ipython:: python
    :okwarning:
    :okexcept:

    from aggregate import pprint_ex
    for n, r in build.qlist('^IR:').iterrows():
        pprint_ex(r.program, split=20)


.. ipython:: python
    :suppress:

    plt.close('all')
