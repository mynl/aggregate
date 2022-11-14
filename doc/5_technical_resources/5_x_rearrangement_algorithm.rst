.. verbatim from PIR

The Rearrangement Algorithm
===========================

The Rearrangement Algorithm (RA) is a practical and straightforward
method to determine the worst-VaR sum. The RA works by iteratively
making each marginal crossed (counter-monotonic) with the sum of the
other marginal distributions. It is easy to program and suitable for
problems involving hundreds of variables and millions of simulations.

The Rearrangement Algorithm was introduced in @Puccetti2012 and subsequently improved in
@Embrechts2013.


**Algorithm Input:** Input samples are arranged in a matrix
:math:`\tilde X = (x_{ij})` with :math:`i=1,\dots, M` rows corresponding
to the simulations and :math:`j=1,\dots, d` columns corresponding to the
different marginals. VaR probability parameter :math:`p`. Accuracy
threshold :math:`\epsilon>0` specifies convergence criterion.

**Algorithm Steps**

1. **Sort** each column of :math:`\tilde X` in descending order.
2. **Set** :math:`N := \lceil (1-p)M \rceil`.
3. **Create** matrix :math:`X` as the :math:`N\times d` submatrix of the
   top :math:`N` rows of :math:`\tilde X`.
4. **Randomly permute** rows within each column of :math:`X`.
5. **Do Loop**

   -  **Create** a new matrix :math:`Y` as follows. **For** column
      :math:`j=1,\dots,d`:

      -  **Create** a temporary matrix :math:`V_j` by deleting the
         :math:`j`\ th column of :math:`X`
      -  **Create** a column vector :math:`v` whose :math:`i`\ th
         element equals the sum of the elements in the :math:`i`\ th row
         of :math:`V_j`
      -  **Set** the :math:`j`\ th column of :math:`Y` equal to the
         :math:`j`\ th column of :math:`X` arranged to have the opposite
         order to :math:`v`, i.e., the largest element in the
         :math:`j`\ th column of :math:`X` is placed in the row of
         :math:`Y` corresponding to the smallest element in :math:`v`,
         the second largest with second smallest, etc.

   -  **Compute** :math:`y`, the :math:`N\times 1` vector with
      :math:`i`\ th element equal to the sum of the elements in the
      :math:`i`\ th row of :math:`Y`.
   -  **Compute** :math:`x` from :math:`X` similarly.
   -  **Compute** :math:`y^{\ast}:=\min(y)`, the smallest element of
      :math:`y`.
   -  **Compute** :math:`x^{\ast}:=\min(x)`.
   -  **If** :math:`y^{\ast}-x^{\ast} \ge \epsilon` **then** set
      :math:`X:=Y` and **repeat** the loop.
   -  **If** :math:`y^{\ast}-x^{\ast} < \epsilon` **then** **break**
      from the loop.

6. **Stack** :math:`Y` on top of the :math:`(M-N)\times d` submatrix of
   :math:`M-N` bottom rows of :math:`\tilde X`.
7. **Output**: The result approximates the worst :math:`\mathsf{VaR}_p`
   arrangement of :math:`\tilde X`.

Only the top :math:`N` values need special treatment; all the smaller
values can be combined arbitrarily because they aren’t included in the
worst-VaR rearrangement. Given that :math:`X` consists of the worst
:math:`1-p` proportion of each marginal, the required estimated
:math:`\mathsf{VaR}_p` is the least row sum of :math:`Y`, that is
:math:`y^{\ast}`. In implementation, :math:`x^{\ast}` can be carried
forward as the :math:`y^{\ast}` from the previous iteration and not
recomputed. The statistics :math:`x^{\ast}` and :math:`y^{\ast}` can be
replaced with the variance of the row-sums of :math:`X` and :math:`Y`
and yield essentially the same results.

@Embrechts2013 report that while there is no analytic proof the
algorithm always works, it performs very well based on examples and
tests where we can compute the answer analytically.

**Example.** Compute the worst
:math:`\mathsf{VaR}_{0.99}` of the sum of lognormal distributions with mean 10
and coefficient of variations 1, 2, and 3 by applying the Rearrangement
Algorithm to a stratified sample of :math:`N=40` observations at and
above the 99th percentile for the matrix :math:`X`.

**Solution.** The table below shows the input and
output of the Rearrangement Algorithm.


.. csv-table:: Starting :math:`X` is shown in the first three columns :math:`x_0, x_1, x_2`. The column Sum shows the row sums :math:`x_0+x_1+x_2` corresponding to a comonotonic ordering. These four columns are all sorted in ascending order. The right-hand three columns, :math:`s_0, s_1, s_2` are the output, with row sum given in the Max VaR column. The worst-case :math:`\text{VaR}_{0.99}` is the minimum of the last column, 352.8. It is 45 percent greater than the additive VaR of 242.5. Only a sample from each marginal’s largest 1 percent values is shown since smaller values are irrelevant to the calculation.
   :file: ra.csv
   :widths: 12, 12, 12, 14, 12, 12, 12, 14
   :header-rows: 1

The table illustrates the worst-case VaR may be substantially higher
than when the marginals are perfectly correlated, here 45 percent higher
at 352.8 vs. 242.5. The form of the output columns shows the two part
structure. There is a series of values up to 356 involving moderate
sized losses from each marginal with approximately the same total. The
larger values of the rearrangement are formed from a large value from
one marginal combined with smaller values from the other two.

The bold entry :math:`366.4` indicates when the comonotonic sum of
marginals exceeds the worst 0.99-VaR arrangement.

Performing the same calculation with :math:`N=1000` samples from the
largest 1 percent of each marginal produces an estimated worst VaR of
360.5.

The following code replicates this calculation in aggregate. The answer relies on random seeds and is slightly different from the table above.

.. ipython:: python
   :okwarning:

   import aggregate as agg
   import numpy as np
   import pandas as pd
   import scipy.stats as ss

   ps = np.linspace(0.99, 1, 40, endpoint=False)
   params = {i: agg.mu_sigma_from_mean_cv(10, i) for i in [1,2,3]}

   df = pd.DataFrame({f'x_{i}': ss.lognorm(params[i][1],
      scale=np.exp(params[i][0])).isf(1-ps)
      for i in [1,2,3]}, index=ps)

   df_ra = agg.rearrangement_algorithm_max_VaR(df)
   with pd.option_context('display.float_format', lambda x: f'{x:.1f}'):
       print(df_ra)

There are several important points to note about the Rearrangement
Algorithm output and the failure of subadditivity it induces. They
mirror the case :math:`d=2`.

-  The dependence structure does not have right tail dependence.
-  In Table 1, the comonotonic sum is greater than the maximum VaR sum
   for the top 40 percent observations, above 366.4. The algorithm
   output is tailored to a specific value of :math:`p` and does not work
   for other :math:`p`\ s. It produces relatively thinner tails for
   higher values of :math:`p` than the comonotonic copula.
-  The algorithm works for any non-trivial marginal distributions—it is
   universal.
-  The implied dependence structure specifies only how the larger values
   of each marginal are related; any dependence structure can be used
   for values below :math:`\mathsf{VaR}_p`.

The Rearrangement Algorithm gives a definitive answer to the question
“Just how bad could things get?” and perhaps provides a better base
against which to measure diversification effect than either independence
or the comonotonic copula. While the multivariate structure it reveals
is odd and specific to :math:`p`, it is not wholly improbable. It
pinpoints a worst-case driven by a combination of moderately severe, but
not extreme, tail event outcomes. Anyone who remembers watching their
investment portfolio during a financial crisis has seen that behavior
before! It is a valuable additional feature for any risk aggregation
software.
