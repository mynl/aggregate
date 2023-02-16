
.. _num floats:

Floating Point Arithmetic and Rounding Errors
-----------------------------------------------

The internal workings of computer floating point arithmetic can cause unexpected problems. You can read no further in this section if you promise to obey

.. warning::

    Only use a bucket size :math:`b` with an exact floating point
    representation. It must have an exact binary representation as a
    fraction :math:`a/b` where :math:`b` is a power of two.

    1/3, 1/5 and 1/10 are **not** binary floats.

For those who choose to continue, this section presents random selection of results about floats that tripped me up as I wrote ``aggregate``.

Floating point arithmetic is not associative!

.. ipython:: python

   x = .1 + (0.6 + 0.3)
   y = (0.1 + 0.6) + 0.3
   x, x.as_integer_ratio(), y, y.as_integer_ratio()

This fact can be used to create sequences with nasty accumulating errors.

.. Knuth observations.

Exercise Redux
""""""""""""""""""

Recall the exercise to compute  quantiles of a :ref:`dice roll <prob dice quantiles>`.
``aggregate`` produces the consistent results---if we look carefully and account for the foibles of floating point numbers. The case :math:`p=0.1` is easy. But the case :math:`p=1/6` appears wrong. There are two ways we can model the throw of a dice: with frequency 1 to 6 and fixed severity 1, or as fixed frequency 1 and severity 1 to 6. They give different answers. The lower quantile is wrong in the first case (it equals 1) and the upper quantile in the second (2).

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    import pandas as pd
    d = build('agg Dice dfreq [1:6] dsev [1]')
    print(d.q(0.1, 'lower'), d.q(0.1, 'upper'))
    print(d.q(1/6, 'lower'), d.q(1/6, 'upper'))
    d2 = build('agg Dice2 dfreq [1] dsev [1:6]')
    print(d2.q(1/6, 'lower'), d2.q(1/6, 'upper'))

These differences are irritating! The short answer is to adhere to the warning above.

Here's the long answer, if you want to know. Looking at the source code shows
that the quantile function is implemented as a previous or next look up on a
dataframe of distinct values of the cumulative distribution function. These
two dataframes for the different dice models are:

.. ipython:: python
    :okwarning:

    ff = lambda x: f'{x:.25g}'
    qd(d.density_df.query('p_total > 0')[['p', 'F']], float_format=ff)
    qd(d2.density_df.query('p_total > 0')[['p', 'F']], float_format=ff)
    print(f'\n{d.cdf(1):.25f} < {1/6:.25f} < 1/6 < {d2.cdf(1):.25f}')

Based on these numbers, the reported quantiles are correct. :math:`p=1/6` is strictly greater than ``d.cdf(1)`` and strictly less than ``d2.cdf(1)``, as shown in the last row! ``d`` and ``d2`` are different because the former runs through the FFT routine to convolve the trivial severity, whereas the latter does not.


Exercise
"""""""""

:math:`X` is a random variable defined on a sample space
with ten equally likely events. The event outcomes are
:math:`0,1,1,1,2,3, 4,8, 12, 25`. Compute :math:`\mathsf{VaR}_p(X)` for
all :math:`p`.

.. ipython:: python
    :okwarning:

    a = build('agg Ex.50 dfreq [1] '
              'dsev [0 1 2 3 4 8 12 25] [.1 .3 .1 .1 .1 .1 .1 .1]')
    @savefig quantile_a.png
    a.plot()
    print(a.q(0.05), a.q(0.1), a.q(0.2), a.q(0.4),
       a.q(0.4, 'upper'), a.q(0.41), a.q(0.5))
    qd(a.density_df.query('p_total > 0')[['p', 'F']],
        float_format=ff)

**Solution.** On the graph, fill in the vertical segments of the
distribution function. Draw a horizontal line at height :math:`p` and
find its intersection with the completed graph. There is a unique
solution for all :math:`p` except :math:`0.1, 0.4, 0.5,\dots, 0.9`.
Consider :math:`p=0.4`. Any :math:`x` satisfying
:math:`\mathsf{Pr}(X < x) \le 0.4 \le \mathsf{Pr}(X\le x)` is a :math:`0.4`-quantile. By
inspection the solutions are :math:`1\le x \le 2`. VaR is defined as the
lower quantile, :math:`x=1`. The :math:`0.41` quantile is :math:`x=2`.
VaRs are not interpolated in this problem specification. The loss 25 is
the :math:`p`-VaR for any :math:`p>0.9`. The apparently errant numbers for aggregate (the upper quantile at 0.1 equals 2) are explained by the float representations. The float representation of ``0.4`` is ``3602879701896397/9007199254740992`` which actually equals ``0.4000000000000000222044605``.
