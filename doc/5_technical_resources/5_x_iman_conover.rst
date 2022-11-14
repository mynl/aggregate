
The Iman-Conover Method
=======================

Here is the basic idea of the Iman-Conover method. Given samples of
:math:`n` values from two known marginal distributions :math:`X` and
:math:`Y` and a desired correlation :math:`\rho` between them, re-order
the samples to have the same rank order as a reference distribution, of
size :math:`n\times 2`, with linear correlation :math:`\rho`. Since
linear correlation and rank correlation are typically close, the
re-ordered output will have approximately the desired correlation
structure. What makes the IC method work so effectively is the existence
of easy algorithms to determine samples from reference distributions
with prescribed linear correlation structures.

Section `1.1 <#theory>`__ explains the Choleski trick for generating
multivariate reference distributions with given correlation structure.
Section `1.2 <#algorithm>`__ gives a formal algorithmic description of
the IC method.

.. _theory:

Theoretical Derivation
----------------------

Suppose that :math:`\mathsf{M}` is an :math:`n` element sample from an :math:`r`
dimensional multivariate distribution, so :math:`\mathsf{M}` is an
:math:`n\times r` matrix. Assume that the columns of :math:`\mathsf{M}` are
uncorrelated, have mean zero, and standard deviation one. Let
:math:`\mathsf{M}'` denote the transpose of :math:`\mathsf{M}`. These assumptions imply
that the correlation matrix of the sample :math:`\mathsf{M}` can be computed as
:math:`n^{-1}\mathsf{M}'\mathsf{M}`, and because the columns are independent,
:math:`n^{-1}\mathsf{M}'\mathsf{M}=\mathsf{id}`. (There is no need to scale the covariance
matrix by the row and column standard deviations because they are all
one. In general :math:`n^{-1}\mathsf{M}'\mathsf{M}` is the covariance matrix of
:math:`\mathsf{M}`.)

Let :math:`\mathsf{S}` be a correlation matrix, i.e. :math:`\mathsf{S}` is a positive
semi-definite symmetric matrix with 1’s on the diagonal and all elements
:math:`\le 1` in absolute value. In order to rule out linearly dependent
variables assume :math:`\mathsf{S}` is positive definite. These assumptions
ensure :math:`\mathsf{S}` has a Choleski decomposition

.. math:: \mathsf{S}=\mathsf{C}'\mathsf{C}

for some upper triangular matrix :math:`\mathsf{C}`, see Golub
Golub or Press et al. Set
:math:`\mathsf{T}=\mathsf{M}\mathsf{C}`. The columns of :math:`\mathsf{T}` still have mean zero, because
they are linear combinations of the columns of :math:`\mathsf{M}` which have
zero mean by assumption. It is less obvious, but still true, that the
columns of :math:`\mathsf{T}` still have standard deviation one. To see why,
remember that the covariance matrix of :math:`\mathsf{T}` is

.. math:: n^{-1}\mathsf{T}'\mathsf{T}=n^{-1}\mathsf{C}'\mathsf{M}'\mathsf{M}\mathsf{C}=\mathsf{C}'\mathsf{C}=\mathsf{S},\label{coolA}

since :math:`n^{-1}\mathsf{M}'\mathsf{M}=\mathsf{id}` is the identity by assumption. Now
:math:`\mathsf{S}` is actually the correlation matrix too because the diagonal
is scaled to one, so the covariance and correlation matrices coincide.
The process of converting :math:`\mathsf{M}`, which is easy to simulate, into
:math:`\mathsf{T}`, which has the desired correlation structure :math:`\mathsf{S}`, is
the theoretical basis of the IC method.

It is important to note that estimates of correlation matrices,
depending on how they are constructed, need not have the mathematical
properties of a correlation matrix. Therefore, when trying to use an
estimate of a correlation matrix in an algorithm, such as the
Iman-Conover, which actually requires a proper correlation matrix as
input, it may be necessary to check the input matrix does have the
correct mathematical properties.

Next we discuss how to make :math:`n\times r` matrices :math:`\mathsf{M}`, with
independent, mean zero columns. The basic idea is to take :math:`n`
numbers :math:`a_1,\dots,a_n` with :math:`\sum_i a_i=0` and
:math:`n^{-1}\sum_i a_i^2=1`, use them to form one :math:`n\times 1`
column of :math:`\mathsf{M}`, and then to copy it :math:`r` times. Finally
randomly permute the entries in each column to make them independent as
columns of random variables. Iman and Conover call the :math:`a_i`
“scores”. They discuss several possible definitions for the scores,
including scaled versions of :math:`a_i=i` (ranks) and :math:`a_i`
uniformly distributed. They note that the shape of the output
multivariate distribution depends on the scores. All of the examples in
their paper use normal scores. We will discuss normal scores here, and
consider alternatives in Section `1.4.1 <#egScore>`__.

Given that the scores will be based on normal random variables, we can
either simulate :math:`n` random standard normal variables and then
shift and re-scale to ensure mean zero and standard deviation one, or we
can use a stratified sample from the standard normal,
:math:`a_i=\Phi^{-1}(i/(n+1))`. By construction, the stratified sample
has mean zero which is an advantage. Also, by symmetry, using the
stratified sample halves the number of calls to :math:`\Phi^{-1}`. For
these two reasons we prefer it in the algorithm below.

The correlation matrix of :math:`\mathsf{M}`, constructed by randomly permuting
the scores in each column, will only be approximately equal to
:math:`\mathsf{id}` because of random simulation error. In order to correct for
the slight error which could be introduced Iman and Conover use another
adjustment in their algorithm. Let :math:`\mathsf{EE}=n^{-1}\mathsf{M}'\mathsf{M}` be the actual
correlation matrix of :math:`\mathsf{M}` and let :math:`\mathsf{EE}=\mathsf{F}'\mathsf{F}` be the
Choleski decomposition of :math:`\mathsf{EE}`, and define
:math:`\mathsf{T}=\mathsf{M}\mathsf{F}^{-1}\mathsf{C}`. The columns of :math:`\mathsf{T}` have mean zero, and
the covariance matrix of :math:`\mathsf{T}` is

.. math::

   \begin{aligned}
   n^{-1}\mathsf{T}'\mathsf{T} &=&n^{-1}\mathsf{C}'\mathsf{F}'^{-1}\mathsf{M}'\mathsf{M}\mathsf{F}^{-1}\mathsf{C} \notag  \\
   &=&\mathsf{C}'\mathsf{F}'^{-1}\mathsf{EE}\mathsf{F}^{-1}\mathsf{C} \notag   \\
   &=&\mathsf{C}'\mathsf{F}'^{-1}\mathsf{F}'\mathsf{F}\mathsf{F}^{-1}\mathsf{C} \notag  \\
   &=&\mathsf{C}' \mathsf{C} \notag  \\
   &=&\mathsf{S},\label{icCorr}\end{aligned}

and hence :math:`\mathsf{T}` has correlation matrix exactly equal to :math:`\mathsf{S}`,
as desired. If :math:`\mathsf{EE}` is singular then the column shuffle needs to
be repeated.

Now the reference distribution :math:`\mathsf{T}` with exact correlation
structure :math:`\mathsf{S}` is in hand, all that remains to complete the IC
method is to re-order the each column of the input distribution
:math:`\mathsf{X}` to have the same rank order as the corresponding column of
:math:`\mathsf{T}`.

Algorithm
---------

Here is a more algorithmic description of the IC method. The description
uses normal scores and the Choleski method to determine the reference
distribution. As we discussed above, it is possible to make other
choices in place of these and they are discussed in Section
`1.4 <#icExt>`__. We will actually present two versions of the core
algorithm. The first, called “Simple Algorithm” deals with the various
matrix operations at a high level. The second “Detailed Algorithm” takes
a more sophisticated approach to the matrix operations, including
referencing appropriate Lapack routines.
Lapack is a standard set of linear algebra functions. Software vendors
provide very high performance implementations of Lapack, many of which
are used in CPU benchmarks. Several free Windows implementations are
available on the web. The software described in the Appendix uses the
Intel Performance http://www.intel.com/software/products/perflib/. The
reader should study the simple algorithm first to understand what is
going in the IC method. In order to code a high performance
implementation you should follow the steps outlined in the detailed
algorithm. Both algorithms have the same inputs and outputs.

An :math:`n \times r` matrix :math:`\mathsf{X}` consisting of :math:`n` samples
from each of :math:`r` marginal distributions, and a desired correlation
matrix :math:`\mathsf{S}`.

The IC method does not address how the columns of :math:`\mathsf{X}` are
determined. It is presumed that the reader has sampled from the
appropriate distributions in some intelligent manner. The matrix
:math:`\mathsf{S}` must be a correlation matrix for linearly independent random
variables, so it must be symmetric and positive definite. If :math:`\mathsf{S}`
is not symmetric positive semi-definite the algorithm will fail at the
Choleski decomposition step. The output is a matrix :math:`\mathsf{T}` each of
whose columns is a permutation of the corresponding column of :math:`\mathsf{X}`
and whose approximate correlation matrix is :math:`\mathsf{S}`.

#. Make one column of scores :math:`a_i=\Phi^{-1}(i/(n+1))` for
   :math:`i=1,\dots,n` and rescale to have standard deviation one.

#. Copy the scores :math:`r` times to make the score matrix :math:`\mathsf{M}`.

#. Randomly permute the entries in each column of :math:`\mathsf{M}`.

#. Compute the correlation matrix :math:`\mathsf{EE}=n^{-1}\mathsf{M}'\mathsf{M}` of :math:`\mathsf{M}`.

#. Compute the Choleski decomposition :math:`\mathsf{EE}=\mathsf{F}'\mathsf{F}` of :math:`\mathsf{EE}`.

#. Compute the Choleski decomposition :math:`\mathsf{S}=\mathsf{C}'\mathsf{C}` of the desired
   correlation matrix :math:`\mathsf{S}`.

#. Compute :math:`\mathsf{T}=\mathsf{M}\mathsf{F}^{-1}\mathsf{C}`. The matrix :math:`\mathsf{T}` has exactly the
   desired correlation structure by Equation (`[icCorr] <#icCorr>`__).

#. Let :math:`\mathsf{Y}` be the input matrix :math:`\mathsf{X}` with each column reordered to have exactly the same rank ordering as the corresponding column of :math:`\mathsf{T}`.

#. Compute the Choleski decomposition of :math:`\mathsf{S}`, :math:`\mathsf{S}=\mathsf{C}'\mathsf{C}`, with :math:`\mathsf{C}` upper triangular. If the Choleski algorithm fails then :math:`\mathsf{S}` is not a valid correlation matrix. Flag an error and exit. Checking :math:`\mathsf{S}` is a correlation matrix in Step 1 avoids performing wasted calculations and allows the routine to exit as quickly as possible. Also check that all the diagonal entries of :math:`\mathsf{S}` are 1 so :math:`\mathsf{S}` has full rank. Again flag an error and exit if not. The Lapack routine DPOTRF can use be used to compute the Choleski decomposition. In the absence of Lapack, :math:`\mathsf{C}=(c_{ij})` can be computed recursively using

   .. math::

      c_{ij}=\frac{s_{ij}-\sum_{k=1}^{j-1}
        c_{ik}c_{jk}}{\sqrt{1-\sum_{k=1}^{j-1} c_{jk}^2}}\label{chol}

   for :math:`1\le i\le j\le n`—since all the diagonal elements of :math:`S` equal one. The empty sum :math:`\sum_0^0=0` and for :math:`j>i` the denominator of (`[chol] <#chol>`__) equals :math:`c_{ii}` and the elements of :math:`\mathsf{C}` should be calculated from left to right, top to bottom. See Wang or Herzog.

#. Let :math:`m=\lfloor n/2\rfloor` be the largest integer less than or equal to :math:`n/2` and :math:`v_i=\Phi^{-1}(i/(2m+1))` for
   :math:`i=1,\dots,m`.

#. If :math:`n` is odd set

   .. math:: \mathsf{v}=(v_m,v_{m-1},\dots,v_1,0,-v_1,\dots,-v_m)

   and if :math:`n` is even set

   .. math:: \mathsf{v}=(v_m,v_{m-1},\dots,v_1,-v_1,\dots,-v_m).

   Here we have chosen to use normal scores. Other distributions could be used in place of the normal, as discussed in Section `1.4.1 <#egScore>`__. Also note that by taking advantage of the symmetry of the normal distribution halves the number of calls to :math:`\Phi^{-1}` which is relatively computationally expensive. If multiple calls will be made to the IC algorithm then store :math:`\mathsf{v}` for use in future calls.

#. Form the :math:`n\times r` score matrix :math:`\mathsf{M}` from :math:`r` copies of the scores vector :math:`\mathsf{v}`.

#. Compute :math:`m_{xx}=n^{-1}\sum_i v_i^2`, the variance of
   :math:`\mathsf{v}`. Note that :math:`\sum_i v_i=0` by construction.

#. Randomly shuffle columns :math:`2,\dots,r` of the score matrix.

#. Compute the correlation matrix :math:`\mathsf{EE}` of the shuffled score matrix :math:`\mathsf{M}`. Each column of :math:`\mathsf{M}` has mean zero, by construction, and variance :math:`m_{xx}`. The correlation matrix is obtained by dividing each element of :math:`\mathsf{M}'\mathsf{M}` by :math:`m_{xx}`. The matrix product can be computed using the Lapack routine DGEMM. If :math:`\mathsf{EE}` is singular repeat step 6.

#. Determine Choleski decomposition :math:`\mathsf{EE}=\mathsf{F}'\mathsf{F}` of :math:`\mathsf{EE}` using the Lapack routine DPOTRF. Because :math:`\mathsf{EE}` is a correlation matrix it must be symmetric and positive definite and so is guaranteed to have a Choleski root.

#. Compute :math:`\mathsf{F}^{-1}\mathsf{C}` using the Lapack routine DTRTRS to solve the linear equation :math:`\mathsf{F}\mathsf{A}=\mathsf{C}` for :math:`\mathsf{A}`. Solving the linear equation avoids a time consuming matrix inversion and multiplication. The routine DTRTRS is optimized for upper triangular input matrices.

#. Compute the correlated scores :math:`\mathsf{T}=\mathsf{M}\mathsf{F}^{-1}\mathsf{C}=\mathsf{M}\mathsf{A}` using DGEMM. The matrix :math:`\mathsf{T}` has exactly the desired correlation structure.

#. Compute the ranks of the elements of :math:`\mathsf{T}`. Ranks are computed by indexing the columns of :math:`\mathsf{T}` as described in Section 8.4 of Press et al. Let :math:`r(k)` denote the index of the :math:`k`\ th ranked element of :math:`\mathsf{T}`.

#. Let :math:`\mathsf{Y}` be the :math:`n\times r` matrix with :math:`i`\ th
   column equal to the :math:`i`\ th column of the input matrix
   :math:`\mathsf{X}` given the same rank order as :math:`\mathsf{T}`. The re-ordering
   is performed using the ranks computed in the previous step. First
   sort the input columns into ascending order if they are not already
   sorted and then set :math:`\mathsf{Y}_{i,k}=\mathsf{X}_{i,r(k)}`.

The output of the algorithm is a matrix :math:`\mathsf{Y}` each of whose columns
is a permutation of the corresponding column of the input matrix
:math:`\mathsf{X}`. The rank correlation matrix of :math:`\mathsf{Y}` is identical to
that of a multivariate distribution with correlation matrix :math:`\mathsf{S}`.

.. _egs:

Simple Example of Iman-Conover
------------------------------

Having explained the IC method, we now give a simple example to
explicitly show all the details. The example will work with :math:`n=20`
samples and :math:`r=4` different marginals. The marginals are samples
from four lognormal distributions, with parameters
:math:`\mu=12,11,10,10` and :math:`\sigma=0.15,0.25,0.35,0.25`. The
input matrix is

.. math::

   \mathsf{X}=
   \begin{pmatrix}
   123,567  & 44,770  & 15,934  & 13,273 \\
   126,109  & 45,191  & 16,839  & 15,406 \\
   138,713  & 47,453  & 17,233  & 16,706 \\
   139,016  & 47,941  & 17,265  & 16,891 \\
   152,213  & 49,345  & 17,620  & 18,821 \\
   153,224  & 49,420  & 17,859  & 19,569 \\
   153,407  & 50,686  & 20,804  & 20,166 \\
   155,716  & 52,931  & 21,110  & 20,796 \\
   155,780  & 54,010  & 22,728  & 20,968 \\
   161,678  & 57,346  & 24,072  & 21,178 \\
   161,805  & 57,685  & 25,198  & 23,236 \\
   167,447  & 57,698  & 25,393  & 23,375 \\
   170,737  & 58,380  & 30,357  & 24,019 \\
   171,592  & 60,948  & 30,779  & 24,785 \\
   178,881  & 66,972  & 32,634  & 25,000 \\
   181,678  & 68,053  & 33,117  & 26,754 \\
   184,381  & 70,592  & 35,248  & 27,079 \\
   206,940  & 72,243  & 36,656  & 30,136 \\
   217,092  & 86,685  & 38,483  & 30,757 \\
   240,935  & 87,138  & 39,483  & 35,108
   \end{pmatrix}.

Note that the marginals are all sorted in ascending order. The algorithm
does not actually require pre-sorting the marginals but it simplifies
the last step.

The desired target correlation matrix is

.. math::

   \mathsf{S}=
   \begin{pmatrix}
   1.000 & 0.800 & 0.400 & 0.000\\
   0.800 & 1.000 & 0.300 & -0.200\\
   0.400 & 0.300 & 1.000 & 0.100\\
   0.000 & -0.200 & 0.100 & 1.000
   \end{pmatrix}.

The Choleski decomposition of :math:`\mathsf{S}` is

.. math::

   \mathsf{C}=
   \begin{pmatrix}
   1.000 & 0.800 & 0.400 & 0.000\\
   0.000 & 0.600 & -0.033 & -0.333\\
   0.000 & 0.000 & 0.916 & 0.097\\
   0.000 & 0.000 & 0.000 & 0.938\\
   \end{pmatrix}.

Now we make the score matrix. The basic scores are
:math:`\Phi^{-1}(i/21)`, for :math:`i=1,\dots,20`. We scale these by
:math:`0.868674836252965` to get a vector :math:`\mathsf{v}` with standard
deviation one. Then we combine four :math:`\mathsf{v}`\ ’s and shuffle randomly
to get

.. math::

   \mathsf{M}=
   \begin{pmatrix}
   -1.92062  & 1.22896  & -1.00860  & -0.49584 \\
   -1.50709  & -1.50709  & -1.50709  & 0.82015 \\
   -1.22896  & 1.92062  & 0.82015  & -0.65151 \\
   -1.00860  & -0.20723  & 1.00860  & -1.00860 \\
   -0.82015  & 0.82015  & 0.34878  & 1.92062 \\
   -0.65151  & -1.22896  & -0.65151  & 0.20723 \\
   -0.49584  & -0.65151  & 1.22896  & -0.34878 \\
   -0.34878  & -0.49584  & -0.49584  & -0.06874 \\
   -0.20723  & -1.00860  & 0.20723  & 0.65151 \\
   -0.06874  & 0.49584  & 0.06874  & -1.22896 \\
   0.06874  & -0.34878  & -1.22896  & 0.49584 \\
   0.20723  & 0.34878  & 0.65151  & 0.34878 \\
   0.34878  & -0.06874  & -0.20723  & 1.22896 \\
   0.49584  & -1.92062  & -0.82015  & -0.20723 \\
   0.65151  & 0.20723  & 1.92062  & -1.92062 \\
   0.82015  & 1.00860  & 1.50709  & 1.50709 \\
   1.00860  & -0.82015  & -1.92062  & 1.00860 \\
   1.22896  & 1.50709  & 0.49584  & -1.50709 \\
   1.50709  & 0.06874  & -0.06874  & 0.06874 \\
   1.92062  & 0.65151  & -0.34878  & -0.82015 \\
   \end{pmatrix}.

As described in Section `1.1 <#theory>`__, :math:`\mathsf{M}` is approximately
independent. In fact :math:`\mathsf{M}` has covariance matrix

.. math::

   \mathsf{EE}=
   \begin{pmatrix}
   1.0000  & 0.0486  & 0.0898  & -0.0960 \\
   0.0486  & 1.0000  & 0.4504  & -0.2408 \\
   0.0898  & 0.4504  & 1.0000  & -0.3192 \\
   -0.0960  & -0.2408  & -0.3192  & 1.0000 \\
   \end{pmatrix}

and :math:`\mathsf{EE}` has Choleski decomposition

.. math::

   \mathsf{F}=
   \begin{pmatrix}
   1.0000 & 0.0486 & 0.0898 & -0.0960\\
   0.0000 & 0.9988 & 0.4466 & -0.2364\\
   0.0000 & 0.0000 & 0.8902 & -0.2303\\
   0.0000 & 0.0000 & 0.0000 & 0.9391\\
   \end{pmatrix}.

Thus :math:`\mathsf{T}=\mathsf{M}\mathsf{F}^{-1}\mathsf{C}` is given by

.. math::

   \mathsf{T}=
   \begin{pmatrix}
   -1.92062  & -0.74213  & -2.28105  & -1.33232 \\
   -1.50709  & -2.06697  & -1.30678  & 0.54577 \\
   -1.22896  & 0.20646  & -0.51141  & -0.94465 \\
   -1.00860  & -0.90190  & 0.80546  & -0.65873 \\
   -0.82015  & -0.13949  & -0.31782  & 1.76960 \\
   -0.65151  & -1.24043  & -0.27999  & 0.23988 \\
   -0.49584  & -0.77356  & 1.42145  & 0.23611 \\
   -0.34878  & -0.56670  & -0.38117  & -0.14744 \\
   -0.20723  & -0.76560  & 0.64214  & 0.97494 \\
   -0.06874  & 0.24487  & -0.19673  & -1.33695 \\
   0.06874  & -0.15653  & -1.06954  & 0.14015 \\
   0.20723  & 0.36925  & 0.56694  & 0.51206 \\
   0.34878  & 0.22754  & -0.06362  & 1.19551 \\
   0.49584  & -0.77154  & 0.26828  & 0.03168 \\
   0.65151  & 0.62666  & 2.08987  & -1.21744 \\
   0.82015  & 1.23804  & 1.32493  & 1.85680 \\
   1.00860  & 0.28474  & -1.23688  & 0.59246 \\
   1.22896  & 1.85260  & 0.17411  & -1.62428 \\
   1.50709  & 1.20294  & 0.39517  & 0.13931 \\
   1.92062  & 1.87175  & -0.04335  & -0.97245 \\
   \end{pmatrix}.

An easy calculation will verify that :math:`\mathsf{T}` has correlation matrix
:math:`\mathsf{S}`, as required.

To complete the IC method we must re-order each column of :math:`\mathsf{X}` to
have the same rank order as :math:`\mathsf{T}`. The first column does not change
because it is already in ascending order. In the second column, the
first element of :math:`\mathsf{Y}` must be the 14th element of :math:`\mathsf{X}`, the
second the 20th, third 10th and so on. The ranks of the other elements
are

.. math::

   \begin{pmatrix}
   14 & 20 & 10 & 18 & 11 & 19 & 17 & 13 & 15 & 8 & 12 & 6 & 9 & 16 & 5 & 3 & 7 & 2 & 4 & 1\\
   20 & 19 & 16 & 4 & 14 & 13 & 2 & 15 & 5 & 12 & 17 & 6 & 11 & 8 & 1 & 3 & 18 & 9 & 7 & 10\\
   18 & 6 & 15 & 14 & 2 & 8 & 9 & 13 & 4 & 19 & 10 & 7 & 3 & 12 & 17 & 1 & 5 & 20 & 11 & 16\\
   \end{pmatrix}'

and the resulting re-ordering of :math:`\mathsf{X}` is

.. math::

   \mathsf{T}=
   \begin{pmatrix}
   123,567  & 50,686  & 15,934  & 16,706 \\
   126,109  & 44,770  & 16,839  & 25,000 \\
   138,713  & 57,685  & 17,620  & 19,569 \\
   139,016  & 47,453  & 35,248  & 20,166 \\
   152,213  & 57,346  & 20,804  & 30,757 \\
   153,224  & 45,191  & 21,110  & 24,019 \\
   153,407  & 47,941  & 38,483  & 23,375 \\
   155,716  & 52,931  & 17,859  & 20,796 \\
   155,780  & 49,420  & 33,117  & 27,079 \\
   161,678  & 58,380  & 22,728  & 15,406 \\
   161,805  & 54,010  & 17,265  & 23,236 \\
   167,447  & 66,972  & 32,634  & 24,785 \\
   170,737  & 57,698  & 24,072  & 30,136 \\
   171,592  & 49,345  & 30,357  & 20,968 \\
   178,881  & 68,053  & 39,483  & 16,891 \\
   181,678  & 72,243  & 36,656  & 35,108 \\
   184,381  & 60,948  & 17,233  & 26,754 \\
   206,940  & 86,685  & 25,393  & 13,273 \\
   217,092  & 70,592  & 30,779  & 21,178 \\
   240,935  & 87,138  & 25,198  & 18,821 \\
   \end{pmatrix}.

The rank correlation matrix of :math:`\mathsf{Y}` is exactly :math:`\mathsf{S}`. The
actual linear correlation is only approximately equal to :math:`\mathsf{S}`. The
achieved value is

.. math::

   \begin{pmatrix}
   1.00  & 0.85  & 0.26  & -0.11 \\
   0.85  & 1.00  & 0.19  & -0.20 \\
   0.26  & 0.19  & 1.00  & 0.10 \\
   -0.11  & -0.20  & 0.10  & 1.00 \\
   \end{pmatrix},

a fairly creditable performance given the input correlation matrix and
the very small number of samples :math:`n=20`. When used with larger
sized samples the IC method typically produces a very close
approximation to the required correlation matrix, especially when the
marginal distributions are reasonably symmetric.

.. _icExt:

Extensions of Iman-Conover
--------------------------

Following through the explanation of the IC method shows that it relies
on a choice of multivariate reference distribution. A straightforward
method to compute a reference is to use the Choleski decomposition
method Equation (`[coolA] <#coolA>`__) applied to certain independent
scores. The example in Section `1.3 <#egs>`__ used normal scores.
However nothing prevents us from using other distributions for the
scores provided they are suitably normalized to have mean zero and
standard deviation one. We explore the impact of different choices of
score distribution on the resulting multivariate distribution in Section
`1.4.1 <#egScore>`__.

Another approach to IC is to use a completely different multivariate
distribution as reference. There are several other families of
multivariate distributions, including the elliptically contoured
distribution family (which includes the normal and :math:`t` as a
special cases) and multivariate Laplace distribution, which are easy to
simulate from. We explore the impact of changing the reference
distribution in Section `1.4.2 <#egRef>`__. Note that changing scores is
actually an example of changing the reference distribution; however, for
the examples we consider the exact form of the new reference is unknown.

.. _egScore:

Alternative Scores
~~~~~~~~~~~~~~~~~~

The choice of score distribution has a profound effect on the
multivariate distribution output by the IC method. Recall that the
algorithm described in Section `1.2 <#algorithm>`__ used normally
distributed scores. We now show the impact of using exponentially and
uniformly distributed scores.

Figure `1.1 <#fig:scores>`__ shows three bivariate distributions with
identical marginal distributions (shown in the lower right hand plot),
the same correlation coefficient of :math:`0.643\pm 0.003` but using
normal scores (top left), exponential scores (top rigtht) and uniform
scores (lower left). The input correlation to the IC method was 0.65 in
all three cases and there are 1000 pairs in each plot. Here the IC
method produced bivariate distributions with actual correlation
coefficient extremely close to the requested value.

The normal scores produce the most natural looking bivariate
distribution, with approximately elliptical contours. The bivariate
distributions with uniform or exponential scores look unnatural, but it
is important to remember that if all you know about the bivariate
distribution are the marginals and correlation coefficient all three
outcomes are possible.

.. figure:: C:/SteveBase/papers/CAS_WP/FinalICExhibits/scores.pdf
   :alt: Bivariate distributions with normal, uniform and exponential
   scores.
   :name: fig:scores

   Bivariate distributions with normal, uniform and exponential scores.

.. figure:: C:/SteveBase/papers/CAS_WP/FinalICExhibits/sums.pdf
   :alt: Sum of marginals from bivariate distributions made with
   different score distributions.
   :name: fig:sums

   Sum of marginals from bivariate distributions made with different
   score distributions.

Figure `1.2 <#fig:sums>`__ shows the distribution of the sum of the two
marginals for each of the three bivariate distributions in Figure
`1.1 <#fig:scores>`__ and for independent marginals. The sum with
exponential scores has a higher kurtosis (is more peaked) than with
normal scores. As expected all three dependent sums have visibly thicker
tails than the independent sum.

Iman and Conover considered various different score distributions in
their paper. They preferred normal scores as giving more natural
looking, elliptical contours. Certainly, the contours produced using
exponential or uniform scores appear unnatural. If nothing else they
provide a sobering reminder that knowing the marginal distributions and
correlation coefficient of a bivariate distribution does not come close
to fully specifying it!

.. _egRef:

Multivariate Reference Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IC method needs some reference multivariate distribution to
determine an appropriate rank ordering for the input marginals. So far
we have discussed using the Choleski decomposition trick in order to
determine a multivariate normal reference distribution. However, any
distribution can be used as reference provided it has the desired
correlation structure. Multivariate distributions that are closely
related by formula to the multivariate normal, such as elliptically
contoured distributions and asymmetric Laplace distributions, can be
simulated using the Choleski trick.

Elliptically contoured distributions are a family which extends the
normal. For a more detailed discussion see Fang and Zhang.
The multivariate :math:`t`-distribution and
symmetric Laplace distributions are in the elliptically contoured
family. Elliptically contoured distributions must have characteristic
equations of the form

.. math:: \Phi(\mathsf{t})=\exp(i\mathsf{t}'\mathsf{m})\phi(\mathsf{t}'\mathsf{S}\mathsf{t})

for some :math:`\phi:\mathsf{R}\to\mathsf{R}`, where :math:`\mathsf{m}` is an :math:`r\times 1`
vector of means and :math:`\mathsf{S}` is a :math:`r\times r` covariance matrix
(nonnegative definite and symmetric). In one dimension the elliptically
contoured distributions coincide with the symmetric distributions. The
covariance is :math:`\mathsf{S}`, if it is defined.

If :math:`\mathsf{S}` has rank :math:`r` then an elliptically contoured
distribution :math:`\mathsf{x}` has a stochastic representation

.. math:: \mathsf{x}=\mathsf{m} + R\mathsf{T}' \mathsf{u}^{(r)}

where :math:`\mathsf{T}` is the Choleski decomposition of :math:`\mathsf{S}`, so
:math:`\mathsf{S}=\mathsf{T}'\mathsf{T}`, :math:`\mathsf{u}^{(r)}` is a uniform distribution on the
sphere in :math:`\mathsf{R}^r`, and :math:`R` is a scale factor independent of
:math:`\mathsf{u}^{(r)}`. The idea here should be clear: pick a direction on the
sphere, adjust by :math:`\mathsf{T}`, scale by a distance :math:`R` and finally
translate by the means :math:`\mathsf{m}`. A uniform distribution on a sphere
can be created as :math:`\mathsf{x}/\Vert \mathsf{x}\Vert` where :math:`\mathsf{x}` has a
multivariate normal distribution with identity covariance matrix. (By
definition, :math:`\Vert \mathsf{x}\Vert^2=\sum_i x_i^2` has a :math:`\chi^2_r`
distribution.) Uniform vectors :math:`\mathsf{u}^{(r)}` can also be created by
applying a random orthogonal matrix to a fixed vector
:math:`(1,0,\dots,0)` on the sphere. Diaconis describes a method for producing random
orthogonal matrices.

The :math:`t`-copula with :math:`\nu` degrees of freedom has a
stochastic representation

.. math:: \mathsf{x}=\mathsf{m} + \frac{\sqrt{\nu}}{\sqrt{S}}\mathsf{z}\label{tsim}

where :math:`S\sim \chi^2_{\nu}` and :math:`\mathsf{z}` is multivariate normal
with means zero and covariance matrix :math:`\mathsf{S}`. Thus one can easily
simulate from the multivariate :math:`t` by first simulating
multivariate normals and then simulating an independent :math:`S` and
multiplying.

The multivariate Laplace distribution is discussed in Kotz, Kozubowski
and Podgorski. It comes in two flavors:
symmetric and asymmetric. The symmetric distribution is also an
elliptically contoured distribution. It has characteristic function of
the form

.. math:: \Phi(\mathsf{t})=\frac{1}{1+ \mathsf{t}'\mathsf{S}\mathsf{t} / 2}\label{symLaplace}

where :math:`\mathsf{S}` is the covariance matrix. To simulate from
(`[symLaplace] <#symLaplace>`__) use the fact that :math:`\sqrt{W}\mathsf{X}`
has a symmetric Laplace distribution if :math:`W` is exponential and
:math:`\mathsf{X}` a multivariate normal with covariance matrix :math:`\mathsf{S}`.

The multivariate asymmetric Laplace distribution has characteristic
function

.. math:: \Psi(\mathsf{t})=\frac{1}{1+\mathsf{t}'\mathsf{S}\mathsf{t}/2 - i\mathsf{m}'\mathsf{t}}.\label{asymLaplace}

To simulate from (`[asymLaplace] <#asymLaplace>`__) use the fact that

.. math:: \mathsf{m} W + \sqrt{W}\mathsf{X} \label{aslsim}

has a symmetric Laplace distribution if :math:`W` is exponential and
:math:`\mathsf{X}` a multivariate normal with covariance matrix :math:`\mathsf{S}` and
means zero. The asymmetric Laplace is not an elliptically contoured
distribution.

Figure `1.3 <#fig:tCopula>`__ compares IC samples produced using a
normal copula to those produced with a :math:`t`-copula. In both cases
the marginals are normally distributed with mean zero and unit standard
deviation. The :math:`t`-copula has :math:`\nu=2` degrees of freedom. In
both figures the marginals are uncorrelated, but in the right the
marginals are not independent. The :math:`t`-copula has pinched tails,
similar to Venter’s Heavy Right Tailed copulas.

.. figure:: C:/SteveBase/papers/CAS_WP/FinalICExhibits/tCopula.pdf
   :alt: IC samples produced from the same marginal and correlation
   matrix using the normal and :math:`t` copula reference distributions.
   :name: fig:tCopula

   IC samples produced from the same marginal and correlation matrix
   using the normal and :math:`t` copula reference distributions.

.. _extAlg:

Algorithms for Extended Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Section `1.4.2 <#egRef>`__ we described how the IC method can be
extended by using different reference multivariate distributions. It is
easy to change the IC algorithm to incorporate different reference
distributions for :math:`t`-copulas and asymmetric Laplace
distributions. Follow the detailed algorithm to step 10. Then use the
stochastic representation (`[tsim] <#tsim>`__) (resp.
`[aslsim] <#aslsim>`__ for the Laplace): simulate from the scaling
distribution for each row and multiply each component by the resulting
number, resulting in an adjusted :math:`\mathsf{T}` matrix. Then complete steps
11 and 12 of the detailed algorithm.

.. _normalCopula:

Comparison With the Normal Copula Method
----------------------------------------

By the normal copula method we mean the following algorithm, described
in Wang  or Herzog.

A set of correlated risks :math:`(X_1,\dots,X_r)` with marginal
cumulative distribution functions :math:`F_i` and Kendall’s tau
:math:`\tau_{ij}=\tau(X_i,X_j)` or rank correlation coefficients
:math:`r(X_i,X_j)`.

#. Convert Kendall’s tau or rank correlation coefficient to correlation
   using

   .. math:: \rho_{ij}=\sin(\pi\tau_{ij}/2)=2\sin(\pi r_{ij}/6)

   and construct the Choleski decomposition :math:`\mathsf{S}=\mathsf{C}'\mathsf{C}` of
   :math:`\mathsf{S}=(\rho_{ij})`.

#. Generate :math:`r` standard normal variables
   :math:`\mathsf{Y}=(Y_1,\dots,Y_r)`.

#. Set :math:`\mathsf{Z}=\mathsf{Y}\mathsf{C}`.

#. Set :math:`u_i=\Phi(Z_i)` for :math:`i=1,\dots,r`.

#. Set :math:`X_i=F_i^{-1}(u_i)`.

The vectors :math:`(X_1,\dots,X_r)` form a sample from a multivariate
distribution with prescribed correlation structure and marginals
:math:`F_i`.

The Normal Copula method works because of the following theorem from
Wang.

.. container:: theorem

   [wangThm] Assume that :math:`(Z_1,\dots,Z_k)` have a multivariate
   normal joint probability density function given by

   .. math:: f(z_1,\dots,z_k)=\frac{1}{\sqrt{(2\pi)^n|\Sigma|}}\exp(-\mathsf{z}'\Sigma^{-1}\mathsf{z}/2),

   :math:`\mathsf{z}=(z_1,\dots,z_k)`, with correlation coefficients
   :math:`\Sigma_{ij}=\rho_{ij}=\rho(Z_i,Z_j)`. Let
   :math:`H(z_1,\dots,z_k)` be their joint cumulative distribution
   function. Then

   .. math:: C(u_1,\dots,u_k)=H(\Phi^{-1}(u_1),\dots,\Phi^{-1}(u_k))

   defines a multivariate uniform cumulative distribution function
   called the normal copula.

   For any set of given marginal cumulative distribution functions
   :math:`F_1,\dots,F_k`, the set of variables

   .. math::

      \label{ncm}
      X_1=F_1^{-1}(\Phi(Z_1)),\dots,X_k=F_1^{-1}(\Phi(Z_k))

   have a joint cumulative function

   .. math::

      F_{X_1,\dots,X_k}(x_1,\dots,x_k)=H(\Phi^{-1}(F_x(u_1)),\dots,
      \Phi^{-1}(F_k(u_k))

   with marginal cumulative distribution functions
   :math:`F_1,\dots,F_k`. The multivariate variables
   :math:`(X_1,\dots,X_k)` have Kendall’s tau

   .. math:: \tau(X_i,X_j)=\tau(Z_i,Z_j)=\frac{2}{\pi}\arcsin(\rho_{ij})

   and Spearman’s rank correlation coefficients

   .. math:: \text{rkCorr}(X_i,X_j)=\text{rkCorr}(Z_i,Z_j)=\frac{6}{\pi}\arcsin(\rho_{ij}/2)

In the normal copula method we simulate from :math:`H` and then invert
using (`[ncm] <#ncm>`__). In the IC method with normal scores we produce
a sample from :math:`H` such that :math:`\Phi(z_i)` are equally spaced
between zero and one and then, rather than invert the distribution
functions, we make the :math:`j`\ th order statistic from the input
sample correspond to :math:`\Phi(z)=j/(n+1)` where the input has
:math:`n` observations. Because the :math:`j`\ th order statistic of a
sample of :math:`n` observations from a distribution :math:`F`
approximates :math:`F^{-1}(j/(n+1))` we see the normal copula and IC
methods are doing essentially the same thing.

While the normal copula method and the IC method are confusingly similar
there are some important differences to bear in mind. Comparing and
contrasting the two methods should help clarify how the two algorithms
are different.

#. Theorem `[wangThm] <#wangThm>`__ shows the normal copula method
   corresponds to the IC method when the latter is computed using normal
   scores and the Choleski trick.

#. The IC method works on a given sample of marginal distributions. The
   normal copula method generates the sample by inverting the
   distribution function of each marginal as part of the simulation
   process.

#. Though the use of scores the IC method relies on a stratified sample
   of normal variables. The normal copula method could use a similar
   method, or it could sample randomly from the base normals. Conversely
   a sample could be used in the IC method.

#. Only the IC method has an adjustment to ensure that the reference
   multivariate distribution has exactly the required correlation
   structure.

#. IC method samples have rank correlation exactly equal to a sample
   from a reference distribution with the correct linear correlation.
   Normal copula samples have approximately correct linear and rank
   correlations.

#. An IC method sample must be taken in its entirety to be used
   correctly. The number of output points is fixed by the number of
   input points, and the sample is computed in its entirety in one step.
   Some IC tools (@Risk, SCARE) produce output which is in a particular
   order. Thus, if you sample the :math:`n`\ th observation from
   multiple simulations, or take the first :math:`n` samples, you will
   not get a random sample from the desired distribution. However, if
   you select random rows from multiple simulations (or, equivalently,
   if you randomly permute the rows output prior to selecting the
   :math:`n`\ th) then you will obtain the desired random sample. It is
   important to be aware of these issues before using canned software
   routines.

#. The normal copula method produces simulations one at a time, and at
   each iteration the resulting sample is a sample from the required
   multivariate distribution. That is, output from the algorithm can be
   partitioned and used in pieces.

In summary remember these differences can have material practical
consequences and it is important not to misuse IC method samples.
