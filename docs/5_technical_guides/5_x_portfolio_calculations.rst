.. _portfolio_calculations:

:class:`Portfolio` Class Calculations
------------------------------------------

A ``Portfolio`` is a collection of ``Aggregate`` objects. The class
computes the densities of each aggregate component as well as the sum,
and also computes the variables shown below.
These variables are central to many
allocation algorithms. All computations use FFTs to compute relevant
convolutions and surface integrals. :math:`X_i(a` represents recoveries
to line :math:`i` when total capital is :math:`a` and lines have equal
priority. It is given by :math:`X_i(a) = X_i(X\wedge a) /X`: when
:math:`X \le a` line :math:`i` is paid in full and :math:`X_i(a)=X_i`
and when :math:`X>a` payments are a pro rata :math:`X_i/X` share of
available assets :math:`a`. Hence expected recoveries are

.. math::

   \mathsf{E}[X_i(a)] &= \mathsf{E}[X_i(X\wedge a) / X] \\
           &= \mathsf{E}[X_i(X\wedge a) / X \mid X \le a]F(a) + \mathsf{E}[X_i(X\wedge a)/ X \mid X > a]S(a) \\
           &= \mathsf{E}[X_i\mid X \le a]F(a) + a\mathsf{E}[X_i /X \mid X > a]S(a) \\
           &= \mathsf{E}[X_i\mid X \le a]F(a) + a\alpha_i(a)S(a) \\

emphasizing the importance of knowing :math:`\mathsf{E}[X_i /X \mid X]`.

Densities are computed using FFT in :math:`O(n\log(n))` time.

.. list-table:: Variables and computational complexity by line :math:`i`
   :widths: 15 35 35 15
   :header-rows: 1

   * - Variable
     - Meaning
     - Computation
     - Complexity
   * - All lines combined
     -
     -
     -
   * - p_total
     - Density of :math:`X=\sum_i X_i`
     - FFT Convolution
     -
   * - exa_total
     - :math:`\mathsf E[\min(X,a)]=\mathsf E[X\wedge a]`
     - Direct sum :math:`\sum_{x\le a} x p(x) + aS(a)`
     - :math:`O(n)`
   * - exlea_total
     - :math:`\mathsf E[X \mid X\le a]`
     -
     -
   * - exgta_total
     - :math:`\mathsf E[X\mid X > a]`
     -
     -
   * - **By line**
     -
     -
     -
   * - (unit pmf)
     - Density of :math:`X_i` on the unit's native grid
     - FFT computation of aggregate using MGF; read via
       ``Portfolio.unit_density`` (not a ``density_df`` column)
     -
   * - exeqa_line
     - :math:`\mathsf E[X_i \mid X=a]`
     - Conv :math:`xf_i(x)`, :math:`f_{\hat i}` (first moment on the
       unit's native grid)
     - :math:`O(n\log(n))`
   * - lev_line
     - :math:`\mathsf E[\min(X_i,a)]=\mathsf E[X_i\wedge a]`
     - Native capped sums :math:`\sum_{x\le a} x p_i(x) + a S_i(a)`
     - :math:`O(n)`
   * - exlea_line
     - :math:`\mathsf E[X_i \mid X\le a]`
     - Cumsum of :math:`E(X_i \mid X=x)f_X(x)`
     - :math:`O(n)`
   * - e_line
     - :math:`\mathsf E[X_i]`
     -
     -
   * - exgta_line
     - :math:`\mathsf E[X_i \mid X > a]`
     - Conditional expectation formula
     -
   * - exi_x_line
     - :math:`\mathsf E[X_i / X]`
     - Sum using conditional expectation
     -
   * - exi_xlea_line
     - :math:`\mathsf E[X_i/X \mid X \le a]`
     - Cumsum of :math:`\mathsf E[X_i\mid X=x]f_X(x)/x`
     -
   * - exi_xgta_line
     - :math:`\mathsf E[X_i/X \mid X > a]`
     - Conditional expectation formula
     -
   * - exa_line
     - :math:`\mathsf E[X_i(a)]`
     - Conditional expectation formula
     -

**For Total, All Lines** :math:`X`

-  Density :math:`f` computed by convolving each individual line using
   FFTs.
-  :math:`F` and :math:`S` are computed from the cumulative sums of the
   density.
-  exa_total :math:`=\mathsf{E}[\min(X,a)]=\mathsf{E}[X\wedge a]`, also
   called lev_total for limited expected value, is computed as the direct
   sum :math:`\sum_{x\le a} x p(x) + aS(a)`. The direct form carries the
   grid origin, so it is exact on windowed and signed (P&L) grids where
   the legacy cumulative-survival form :math:`\sum S\cdot bs` (which
   silently integrates from 0) is not. Note exa_total= lev_total.
-  exlea\_total :math:`=\mathsf{E}[X \mid X\le a]` is the direct ratio

   .. math::

      \mathsf E[X \mid X\le a]=\frac{1}{F(a)} \sum_{x \le a} x p(x).

   When :math:`F(a)` is at or below the validation noise floor the
   conditional mean is unresolvable and the value is left NaN (an
   explicit denominator guard; the legacy code blanked a heuristic
   ``loss_max`` region instead).
-  exgta\_total :math:`=\mathsf{E}[X\mid X > a]` is computed using the relation :math:`\mathsf{E}[X] = \mathsf{E}[X\mid X \le a]F(a) + \mathsf{E}[X\mid X > a]S(a)`. Therefore

   .. math::

      \mathsf{E}[X\mid X > a] = \frac{\mathsf{E}[X]-\sum_{x\le a} x p(x)}{S(a)},

   guarded NaN where :math:`S(a)` is at or below the noise floor.


**For Individual Lines** :math:`X_i`

-  Density and distributions as for total.
-  exeqa_line :math:`=\mathsf{E}[X_i \mid X=a]=\kappa_i(a)` can be computed
   efficiently using FFTs in the case :math:`X_i` are independent.
   Without loss of generality :math:`X=X_i + \hat X_i` where
   :math:`\hat X_i` is the sum of all other lines (“not :math:`i`”). Let
   :math:`f_x(x_i, \hat x_i)` be the conditional density of
   :math:`X_i=x_i`, :math:`\hat X_i=\hat x_i` given :math:`X=x`. Thus
   :math:`f_x(x_i, \hat x_i) = f(x_i, \hat x_i) / f_X(x)` where
   :math:`f` is the bivariate density of :math:`X_i` and
   :math:`\hat X_i` and :math:`f_X` is the unconditional density of
   :math:`X`. Assuming independence between :math:`X_i` and
   :math:`\hat X_i`:

   .. math::

      \mathsf{E}[X_i \mid X=a] &= \int_0^a x_i f_a(x_i, a-x_i) dx_i\\
                 &= \frac{1}{f_X(a)} \int_0^a x_i f_i(x_i)f_{\hat i}(a-x_i) dx_i

   showing :math:`\mathsf E[X_i \mid X=a]` is the convolution of
   the functions :math:`x_i\mapsto x_i f_i(x_i)` and :math:`f_{\hat i}`.
   The convolution can be computed using FFTs. In the case
   :math:`f_X(a)` is very small these estimates may be numerically
   unreliable.
-  exlea_line :math:`=\mathsf{E}[X_i \mid X\le a]` is given by

   .. math::

      \mathsf{E}[X_i \mid X\le a] &= \mathsf{E}[\mathsf{E}(X_i \mid X\le a]\mid X) \\
             &= \int_0^a \mathsf{E}[X_i \mid X\le a, X=x]f_{\{X\mid X\le a\}}(x) dx \\
                     &=\frac{1}{F_X(a)} \int_0^a \mathsf{E}[X_i \mid X=x]f_X(x) dx \\

   can be computed for all :math:`a` using the cumulative
   sums. Care is needed when :math:`a` is so small that :math:`F(a)` is
   very small.
-  exgta_line :math:`=E(X_i \mid X \ge a)` can be computed using
   :math:`\mathsf{E}[X] = E(X_i \mid X\le a)F(a) + \mathsf{E}[X_i \mid X > a]S(a)`.
   It could also be computed with a reverse cumulative sum.
-  exi_x_line :math:`=\mathsf{E}[X_i / X]`, the unconditional average
   proportion of losses from line :math:`i` is computed as

   .. math::
      \mathsf{E}[X_i / X] &= \mathsf{E}_X[\mathsf{E}[X_i/X \mid X]] \\
             &= \mathsf{E}_X[\mathsf{E}[X_i \mid X] / X] \\
             &= \int_0^\infty \mathsf{E}[X_i \mid X=x]x^{-1} f_X(x)dx.

-  exi_xlea_line :math:`=\mathsf{E}[X_i/X \mid X \le a]` is computed
   using cumulative sums via

   .. math::

      \mathsf{E}[X_i/X \mid X \le a] = \frac{1}{F(a)}\int_0^a \mathsf{E}[X_i\mid X=x]x^{-1}f_X(x)dx.

-  exi_xgta_line :math:`=\mathsf{E}[X_i/X \mid X > a]=\alpha_i(a)` computed from
   :math:`\mathsf{E}[X_i/X]` and :math:`\mathsf{E}[X_i/X \mid X \le a]`
   as usual.
-  exa_line :math:`=\mathsf{E}[X_i(a)]` is the loss cost for line
   :math:`i` using the equal priority rule. It is computed by
   conditioning on :math:`X`

   .. math::

      \mathsf{E}[X_i(a)] &= \mathsf{E}[X_i(a] \mid X \le a)F(a) + \mathsf{E}[X_i(a] \mid X > a)S(a) \\
            &= \mathsf{E}[X_i \mid X \le a]F(a) + a\mathsf{E}[X_i/X \mid X > a]S(a)

   showing it is a simple weighted average of
   :math:`\mathsf{E}[X_i \mid X \le a]` and
   :math:`\mathsf{E}[X_i/X \mid X > a]`, both of which have already been
   computed. The implementation uses the equivalent direct-sum form
   :math:`\sum_{x\le a}\kappa_i(x)p(x) + a\sum_{x>a}\kappa_i(x)x^{-1}p(x)`,
   which avoids multiplying and dividing by :math:`F` and :math:`S` and
   carries the grid origin (exact on windowed grids). On a signed (P&L)
   grid the ratio :math:`X_i/X` is not a recovery share, so the
   share-based columns (``exi_x*_line`` and ``exa_line``) are left NaN
   there; the conditional means (``exeqa/exlea/exgta``) remain valid.

.. note::
   The second-priority recovery (``e2pri_line``) and expected
   policyholder deficit (``epd_*``) columns were removed in the 1.0
   numerics rationalization -- they had no remaining consumers. The
   stand-alone EPD is the one-liner ``(e_line - lev_line) / e_line``
   from the surviving columns, and the equal-priority version is
   ``(e_line - exa_line) / e_line``.

The upshot of these calculations is that all the required values, for
all levels of capital :math:`a` can be computed in time
:math:`O(mn\log(n))` where :math:`m` is the number of lines of business
and :math:`n` is the length of the vector used to discretize the
underlying distributions. Without using FFTs the calculations would take
:math:`O(mn^2)`. Since :math:`n` is typically in the range
:math:`2^{10}` to :math:`2^{20}` FFTs provide a huge speed-up. Using
simple simulations would be completely impractical for the delicate
calculations involved.

The calculation of
:math:`\mathsf{E}[X_i(a)] = \mathsf{E}[X_i \mid X \le a]F(a) + a\mathsf{E}[X_i/X \mid X > a]S(a)`
depends critically on the fact that the same values
:math:`\mathsf{E}[X_i \mid X=x]` and
:math:`\mathsf{E}[X_i/X \mid X > a]` are used for all values of
:math:`a`. Only the weights :math:`F(a)` and :math:`S(a)` change with
:math:`a`. As a result :math:`\mathsf{E}[X_i(a)]` can be computed in one
sweep of length :math:`n`. If different values were required for each
value of :math:`a` the complexity would jump up to
:math:`O(mn\times n^2)` (or :math:`O(mn\times n\log(n))` if it is
possible to use FFTs). This is unfortunately the situation when one line
is collateralized because the ratio of capital to collateral determines
the allocation of assets in insolvency.

Now we compute the impact of applying a distortion :math:`g` to the
underlying probabilities, i.e. discuss premium allocations.

Let :math:`\mathsf{E}_g` denote expected values with respect to the
distorted probabilities defined by :math:`g`.


.. list-table:: Variables and computational complexity by line :math:`i`, with distorted probabilities. Complexity refers to additional complexity beyond values already computed.
    :widths: 25 25 25 25
    :header-rows: 1

    * - **Variable**
      - **Meaning**
      - **Computation**
      - **Complexity**
    * - gS, gF
      - :math:`g(S(x))` and :math:`1-g(S(x))`
      -
      - :math:`O(n)`
    * - gp_total
      - Estimate of :math:`-d g(S(x))/dx`
      - Difference of :math:`g(S)`
      - :math:`O(n)`
    * - exag_total
      - :math:`\mathsf E_g[X\wedge a]`
      - Cumulative sum of :math:`g(S)`
      - :math:`O(n)`
    * - exag_line
      - :math:`\mathsf E_g[X_i(a)]`
      - See below
      - :math:`O(n)`


-  exag_total is easy to compute as the cumulative sums of :math:`g(S)`
-  exag_line is computed as

   .. math::
      \mathsf{E}_g[X_i(a)] &= \mathsf{E}\left[X_i\frac{X\wedge a}{X}g'S(X)\right] \\
      &=  \mathsf{E}\left[\mathsf{E}\left[X_i\frac{X\wedge a}{X}g'S(X)\mid X \right]\right] \\
      &=  \mathsf{E}\left[\mathsf{E}[X_i \mid X] 1_{\{X\le a\}} g'S(X) \right] +
      a \mathsf{E}\left[\frac{\mathsf{E}[X_i\mid X]}{X} 1_{\{X > a\}} g'S(X) \right] \\
      &= \int_0^a \mathsf{E}[X_i\mid X=x] g'(S(x))f_X(x)dx +
      \int_a^\infty  \mathsf{E}[X_i\mid X=x] x^{-1} g'S(x)f_X(x)dx.

   The first integral is computed as a cumulative sum of
   its terms, the second is computed as a reverse cumulative sum, both
   using ``exeqa``. This expectation can also be expressed using :math:`\beta_i(a)`.
-  If :math:`g` has a probability mass at :math:`s=0` then **how are the
   masses dealt with**?

Finally we discuss computing the impact of line specific collateral.

Computing the impact of collateral on recoveries. Computes the expected
recoveries to line :math:`X_i` when there are assets :math:`a` but line
:math:`i` has collateral :math:`c\le a`. This calculation, alas, cannot
be performed quickly using FFTs. It has to be computed mirroring the
three way split of the default zone: no default, default and line
:math:`i` just paid full collateral (which requires :math:`X_i < cx/a`
where :math:`x` is total loss), and line :math:`i` is paid its usual pro
rata proportion of assets.
