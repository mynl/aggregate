.. _2_x_severity:

.. _2_agg_class_severity_clause:

The Severity Clause
----------------------

The severity clause specifies the ground-up severity distribution, or "curve" as it is sometimes known. It is a very flexible clause. Its design follows the ``scipy.stats`` package's specification of random variables using shape, location, and scale factors, see :ref:`probability background <5_x_probability>`. The syntax is different for non-parametric discrete distributions and parametric continuous distributions.


.. _nonparametric severity:

Non-Parametric Severity Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discrete distributions (supported on a finite number of outcomes)
can be directly entered as a severity using the ``dsev`` keyword followed by
two equal-length rows vectors. The first gives the outcomes and the second the
probabilities.

::

    dsev [outcomes] <[probabilities]>

The horizontal layout is irrelevant and commas are optional.
If the ``probabilities`` vector is omitted then all probabilities are set equal to
the reciprocal of the length of the ``outcomes`` vector.
A Python-like colon notation is available for ranges.
Probabilities can be entered as fractions, but no other arithmetic operation is supported.

The five examples::

    dsev [0 9 10] [0.5 0.3 0.2]
    dsev [0 9 10]
    dsev [1:6]
    dsev [0:100:25]
    dsev [1:6] [1/4 1/4 1/8 1/8 1/8 1/8]

specify

#. A severity with a 0.5 chance of taking the value 0, 0.3 chance of 9, and 0.2 of 10.
#. Equally likely outcomes of 0, 9, or 10;
#. Equally likely outcomes 1, 2, 3, 4, 5, 6;
#. Equally likely outcomes 0, 25, 50, 100; and
#. Outcomes 1 or 2 with probability 0.25 or 3-6 with probability 0.125.

.. warning::
    Use binary fractions (denominator a power of two) to avoid rounding errors!


A ``dsev`` clause is converted by the parser into a ``dhistogram`` step distribution::

    sev dhistogram xps [outcomes] [probabilities]

In rare cases you want a continuous (ogive, piecewise linear distribution) version::

    sev chistogram xps [outcomes] [probabilities]

When executed, these are both converted into a ``scipy.stats`` ``histogram`` class.


Parametric Severity
~~~~~~~~~~~~~~~~~~~~~

A parametric distribution can be specified in two ways::

    sev DIST_NAME MEAN cv CV
    sev DIST_NAME <SHAPE1> <SHAPE2>

where

* ``sev`` is a keyword indicating the severity specification,
* ``DIST_NAME`` is the ``scipy.stats`` distribution name, see :ref:`available sev dists`,
* ``MEAN`` is the expected loss,
* ``cv`` (lowercase) is sa keyword indicating entry of the CV,
* ``CV`` is the loss coefficient of variation, and
* ``SHAPE1``, ``SHAPE2`` are the shape variables.

The first form enters the expected ground-up severity and CV directly. It is available for distributions with only one shape parameter and the beta distribution on :math:`[0,1]`. ``aggregate`` uses a formula (lognormal, gamma, beta) or numerical method (all others) to solve for the shape parameter to achieve the correct CV and then scales to the desired mean. The second form directly enters the shape variable(s). Shape parameters entered for zero parameter distributions are ignored.

**Example.** Entering ``sev lognorm 10 cv 0.2`` produces a lognormal
distribution with a mean of 10 and a CV of 0.2. Entering ``lognorm 0.2`` produces a lognormal
with :math:`\mu=0` and :math:`\sigma=0.2`. It can then be :ref:`scaled and shifted<dec shift scale>`.

``DIST_NAME`` can be any zero, one, or two shape parameter ``scipy.stats`` continuous distribution.
They have (mostly) easy to guess names:

* No shape parameters

    - ``norm``, Gaussian normal
    - ``unif``, uniform
    - ``expon``, the exponential

* One shape parameter

    - ``pareto``
    - ``lognorm``
    - ``gamma``
    - ``invgamma``
    - ``loggamma``
    - ``weibull`` WHAT?

* Two shape parameters

    - ``beta``
    - ``gengamma``, generalized gamma

See :ref:`available sev dists` for a full list.

Finally, ``dhistogram`` and ``chistogram`` can be used to create discrete
(point mass) and continuous (ogive) empirical distributions. ``chistogram``
is rarely used and ``dhistogram`` is easier to input using ``dsev``,
:ref:`nonparametric severity`.


.. _dec shift scale:

Shifting and Scaling Severity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A parametric severity clause can be transformed by scaling and location (shifting or translation) factors,
using the standard ``scipy.stats`` ``scale`` and ``loc``::

    sev SCALE * DISTNAME SHAPE + LOC
    sev SCALE * DISTNAME SHAPE - LOC

For zero
parameter distributions ``SHAPE`` is omitted. Two parameter distributions are
``sev SCALE * DISTNAME SHAPE1 SHAPE2 + LOC``.

**Examples.**

* ``sev lognorm 10 cv 3``: lognormal, mean 10, cv 0.

* ``sev 10 * lognorm 1.75``: lognormal, :math:`10X`, :math:`X \sim \mathrm{lognormal}(\mu=0,\sigma=1.75)`

* ``sev 10 * lognorm 1.75 + 20``: lognormal, :math:`10X + 20`

* ``sev 10 * lognorm 1 cv 3 + 50``: lognormal: :math:`10Y + 50`, :math:`Y\sim` lognormal mean 1, cv 3

* ``sev 100 * pareto 1.3 - 100``: Pareto, shape (:math:`\alpha`) 3, scale (:math:`\lambda`) 100

* ``sev 100 * pareto 1.3``: Single parameter Pareto for :math:`x \ge 100`, Shape (:math:`\alpha`) 3, scale (:math:`\lambda`) 100

* ``sev 50 * norm + 100``: normal, mean (location) 100, standard deviation (scale) 50. No shape parameters.

* ``sev 5 * expon``: exponential, mean (scale) 5. No shape parameters.

* ``sev 5 * uniform + 1``: uniform between 1 and 6, scale 5, location 1. No shape parameters.

* ``sev 50 * beta 2 3``: beta: :math:`50Z`, :math:`Z \sim \beta(2,3)`, shape parameters 2, 3, scale 50.

With this parameterization, the Pareto has survival function :math:`S(x)=(100 / (100 + x))^{1.3}`.

The scale and location parameters can be :doc:`vectors<070_vectorization>`.

.. warning::
    ``dsev`` severities **cannot** be shifted or scaled.
    If that is required use a Python f-string to adjust the outcomes::

        f'dsev [{{5 * outcomes + 10}}] [probabilities]'

.. warning::
    Shifting left (negative shift) must be written with space ``sev 10 * lognorm 1.5 - 10`` not
    ``sev 10 * lognorm 1.5 -10``. The lexer binds uniary minus to the number, so the latter omits the operator.


Unconditional Severity
~~~~~~~~~~~~~~~~~~~~~~~

The severity clause is ground-up and it is converted to a distribution
conditional on a loss to the layer if there is a limits sub-clause. Thus, for
an excess layer :math:`y` xs :math:`a` severity has a
distribution :math:`X \mid X > a`, where :math:`X` is specified in the
``sev`` clause. For a ground-up (or missing) layer there is no adjustment.

The default behavior can be over-ridden by adding ``!`` after the
severity distribution.


**Example.**

::

   agg Conditional 1 claim 10 x 10 sev lognorm 10 cv 1 fixed
   agg Unconditional 1 claim 10 x 10 sev lognorm 10 cv 1 ! fixed

produces conditional and unconditional samples from an excess layer of a
lognormal. The latter includes an approximately 0.66 chance of a claim
of zero, corresponding to :math:`X \le 10` below the attachment.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

    cond = build('agg Conditional   1 claim 10 x 10 sev 5 * expon   fixed')
    uncd = build('agg Unconditional 1 claim 10 x 10 sev 5 * expon ! fixed')
    qd(cond.describe)
    qd(uncd.describe)
    print(uncd.sevs[0].fz.sf(10), uncd.agg_m / cond.agg_m)

Here ``uncd.sevs[0].fz`` is ``sev 5 * expon`` ground-up.


.. _available sev dists:

``scipy.stats`` Continuous Random Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ``scipy.stats`` continuous random variable classes can be used as severity distributions. As always, with great power comes
great responsibility.

.. warning::
    The user must determine if a severity distribution is appropriate, ``aggregate`` will not check!
    Only specified zero parameter (uniform, exponential, normal) and two parameter () distributions are allowed, but **all** one parameter
    distributions will work. However, any zero parameter distribution can be called with a dummy argument, that is ignored. **Be
    careful out there!**

The information below was extracted from the `scipy help for continuous distributions <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`_. The basic list can be created by introspection---wonderful Python!


.. ipython:: python
    :okwarning:

    import scipy.stats as ss
    import pandas as pd

    ans = []
    for k in dir(ss):
        ob = getattr(ss, k)
        if str(type(ob)).find('continuous_distns') > 0:
            try:
                fz = ob()
            except TypeError as e:
                ee = e
                ans.append([k, str(e), -1, ob.a, ob.b])
            else:
                ans.append([k, 'no args fine', 0, ob.a, ob.b])

    df = pd.DataFrame(ans, columns=['dist', 'm', 'args', 'a', 'b'])
    for i in range(1,5):
        df.loc[df.m.str.find(f'{i} required')>=0, 'args'] = i

    df = df.sort_values(['args', 'dist'])
    df['params'] = ''
    df.loc[df.args > 0, 'params'] = df.loc[df.args > 0, 'm'].str.split(':').str[1]
    df = df.drop(columns='m')

    print(df.rename(columns={'dist': 'Distribution', 'args': 'Num. args',
            'a': 'Min range' , 'b': 'Max range', 'params': 'Parameters'}).\
            set_index('Distribution').to_string(float_format=lambda x: f'{x:.4g}'))

.. _list of distributions:

.. _dist alpha:

* ``alpha`` **Alpha** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html>`_). The probability density function for `alpha` is:

    .. math::

        f(x, a) = \frac{1}{x^2 \Phi(a) \sqrt{2\pi}} *
                  \exp(-\frac{1}{2} (a-1/x)^2)

    where :math:`\Phi` is the normal CDF, :math:`x > 0`, and :math:`a > 0`.

    `alpha` takes ``a`` as a shape parameter.


.. _dist anglit:

* ``anglit`` **Anglit** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anglit.html>`_). The probability density function for `anglit` is:

    .. math::

        f(x) = \sin(2x + \pi/2) = \cos(2x)

    for :math:`-\pi/4 \le x \le \pi/4`.


.. _dist arcsine:

* ``arcsine`` **Arcsine** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.arcsine.html>`_). The probability density function for `arcsine` is:

    .. math::

        f(x) = \frac{1}{\pi \sqrt{x (1-x)}}

    for :math:`0 < x < 1`.


.. _dist argus:

* ``argus`` **Argus** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.argus.html>`_). The probability density function for `argus` is:

    .. math::

        f(x, \chi) = \frac{\chi^3}{\sqrt{2\pi} \Psi(\chi)} x \sqrt{1-x^2}
                     \exp(-\chi^2 (1 - x^2)/2)

    for :math:`0 < x < 1` and :math:`\chi > 0`, where

    .. math::

        \Psi(\chi) = \Phi(\chi) - \chi \phi(\chi) - 1/2

    with :math:`\Phi` and :math:`\phi` being the CDF and PDF of a standard
    normal distribution, respectively.

    `argus` takes :math:`\chi` as shape a parameter.


.. _dist beta:

* ``beta`` **Beta** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html>`_). The probability density function for `beta` is:

    .. math::

        f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}

    for :math:`0 <= x <= 1`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `beta` takes :math:`a` and :math:`b` as shape parameters.


.. _dist betaprime:

* ``betaprime`` **Beta Prime** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betaprime.html>`_). The probability density function for `betaprime` is:

    .. math::

        f(x, a, b) = \frac{x^{a-1} (1+x)^{-a-b}}{\beta(a, b)}

    for :math:`x >= 0`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\beta(a, b)` is the beta function (see `scipy.special.beta`).

    `betaprime` takes ``a`` and ``b`` as shape parameters.


.. _dist bradford:

* ``bradford`` **Bradford** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bradford.html>`_). The probability density function for `bradford` is:

    .. math::

        f(x, c) = \frac{c}{\log(1+c) (1+cx)}

    for :math:`0 <= x <= 1` and :math:`c > 0`.

    `bradford` takes ``c`` as a shape parameter for :math:`c`.


.. _dist burr:

* ``burr`` **Burr (Type III)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr.html>`_). The probability density function for `burr` is:

    .. math::

        f(x, c, d) = c d x^{-c - 1} / (1 + x^{-c})^{d + 1}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr` takes :math:`c` and :math:`d` as shape parameters.

    This is the PDF corresponding to the third CDF given in Burr's list;
    specifically, it is equation (11) in Burr's paper. The distribution
    is also commonly referred to as the Dagum distribution. If the
    parameter :math:`c < 1` then the mean of the distribution does not
    exist and if :math:`c < 2` the variance does not exist.
    The PDF is finite at the left endpoint :math:`x = 0` if :math:`c * d >= 1`.


.. _dist burr12:

* ``burr12`` **Burr (Type XII)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr12.html>`_). The probability density function for `burr` is:

    .. math::

        f(x, c, d) = c d x^{c-1} / (1 + x^c)^{d + 1}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr12` takes ``c`` and ``d`` as shape parameters for :math:`c`
    and :math:`d`.

    This is the PDF corresponding to the twelfth CDF given in Burr's list;
    specifically, it is equation (20) in Burr's paper.


.. _dist cauchy:

* ``cauchy`` **Cauchy** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html>`_). The probability density function for `cauchy` is

    .. math::

        f(x) = \frac{1}{\pi (1 + x^2)}

    for a real number :math:`x`.


.. _dist chi:

* ``chi`` **Chi** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi.html>`_). The probability density function for `chi` is:

    .. math::

        f(x, k) = \frac{1}{2^{k/2-1} \Gamma \left( k/2 \right)}
                   x^{k-1} \exp \left( -x^2/2 \right)

    for :math:`x >= 0` and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation). :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    Special cases of `chi` are:

        - ``chi(1, loc, scale)`` is equivalent to `halfnorm`
        - ``chi(2, 0, scale)`` is equivalent to `rayleigh`
        - ``chi(3, 0, scale)`` is equivalent to `maxwell`

    `chi` takes ``df`` as a shape parameter.


.. _dist chi2:

* ``chi2`` **Chi-squared** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html>`_). The probability density function for `chi2` is:

    .. math::

        f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                   x^{k/2-1} \exp \left( -x/2 \right)

    for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation).

    `chi2` takes ``df`` as a shape parameter.

    The chi-squared distribution is a special case of the gamma
    distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
    ``scale = 2``.


.. _dist cosine:

* ``cosine`` **Cosine** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cosine.html>`_). The cosine distribution is an approximation to the normal distribution. The probability density function for `cosine` is:

    .. math::

        f(x) = \frac{1}{2\pi} (1+\cos(x))

    for :math:`-\pi \le x \le \pi`.


.. _dist crystalball:

* ``crystalball`` **Crystalball** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.crystalball.html>`_). The probability density function for `crystalball` is:

    .. math::

        f(x, \beta, m) =  \begin{cases}
                            N \exp(-x^2 / 2),  &\text{for } x > -\beta\\
                            N A (B - x)^{-m}  &\text{for } x \le -\beta
                          \end{cases}

    where :math:`A = (m / |\beta|)^m  \exp(-\beta^2 / 2)`,
    :math:`B = m/|\beta| - |\beta|` and :math:`N` is a normalisation constant.

    `crystalball` takes :math:`\beta > 0` and :math:`m > 1` as shape
    parameters.  :math:`\beta` defines the point where the pdf changes
    from a power-law to a Gaussian distribution.  :math:`m` is the power
    of the power-law tail.


.. _dist dgamma:

* ``dgamma`` **Double Gamma** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dgamma.html>`_). The probability density function for `dgamma` is:

    .. math::

        f(x, a) = \frac{1}{2\Gamma(a)} |x|^{a-1} \exp(-|x|)

    for a real number :math:`x` and :math:`a > 0`. :math:`\Gamma` is the
    gamma function (`scipy.special.gamma`).

    `dgamma` takes ``a`` as a shape parameter for :math:`a`.


.. _dist dweibull:

* ``dweibull`` **Double Weibull** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dweibull.html>`_). The probability density function for `dweibull` is given by

    .. math::

        f(x, c) = c / 2 |x|^{c-1} \exp(-|x|^c)

    for a real number :math:`x` and :math:`c > 0`.

    `dweibull` takes ``c`` as a shape parameter for :math:`c`.


.. _dist erlang:

* ``erlang`` **Erlang** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.erlang.html>`_). The Erlang distribution is a special case of the Gamma distribution, with the shape parameter `a` an integer.  Note that this restriction is not enforced by `erlang`. It will, however, generate a warning the first time a non-integer value is used for the shape parameter.

    :ref:`Refer to <dist gamma>` `gamma` for examples.


.. _dist expon:

* ``expon`` **Exponential** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html>`_). The probability density function for `expon` is:

    .. math::

        f(x) = \exp(-x)

    for :math:`x \ge 0`.


.. _dist exponnorm:

* ``exponnorm`` **Exponentially Modified Normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html>`_). The probability density function for `exponnorm` is:

    .. math::

        f(x, K) = \frac{1}{2K} \exp\left(\frac{1}{2 K^2} - x / K \right)
                  \text{erfc}\left(-\frac{x - 1/K}{\sqrt{2}}\right)

    where :math:`x` is a real number and :math:`K > 0`.

    It can be thought of as the sum of a standard normal random variable
    and an independent exponentially distributed random variable with rate
    ``1/K``.


.. _dist exponweib:

* ``exponweib`` **Exponentiated Weibull** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html>`_). The probability density function for `exponweib` is:

    .. math::

        f(x, a, c) = a c [1-\exp(-x^c)]^{a-1} \exp(-x^c) x^{c-1}

    and its cumulative distribution function is:

    .. math::

        F(x, a, c) = [1-\exp(-x^c)]^a

    for :math:`x > 0`, :math:`a > 0`, :math:`c > 0`.

    `exponweib` takes :math:`a` and :math:`c` as shape parameters:

    * :math:`a` is the exponentiation parameter,
      with the special case :math:`a=1` corresponding to the
      (non-exponentiated) Weibull distribution `weibull_min`.
    * :math:`c` is the shape parameter of the non-exponentiated Weibull law.


.. _dist exponpow:

* ``exponpow`` **Exponential Power** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponpow.html>`_). The probability density function for `exponpow` is:

    .. math::

        f(x, b) = b x^{b-1} \exp(1 + x^b - \exp(x^b))

    for :math:`x \ge 0`, :math:`b > 0`.  Note that this is a different
    distribution from the exponential power distribution that is also known
    under the names "generalized normal" or "generalized Gaussian".

    `exponpow` takes ``b`` as a shape parameter for :math:`b`.


.. _dist f:

* ``f`` **F (Snecdor F)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html>`_). The probability density function for `f` is:

    .. math::

        f(x, df_1, df_2) = \frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}
                                {(df_2+df_1 x)^{(df_1+df_2)/2}
                                 B(df_1/2, df_2/2)}

    for :math:`x > 0`.

    `f` takes ``dfn`` and ``dfd`` as shape parameters.


.. _dist fatiguelife:

* ``fatiguelife`` **Fatigue Life (Birnbaum-Saunders)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fatiguelife.html>`_). The probability density function for `fatiguelife` is:

    .. math::

        f(x, c) = \frac{x+1}{2c\sqrt{2\pi x^3}} \exp(-\frac{(x-1)^2}{2x c^2})

    for :math:`x >= 0` and :math:`c > 0`.

    `fatiguelife` takes ``c`` as a shape parameter for :math:`c`.


.. _dist fisk:

* ``fisk`` **Fisk** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisk.html>`_). The probability density function for `fisk` is:

    .. math::

        f(x, c) = c x^{-c-1} (1 + x^{-c})^{-2}

    for :math:`x >= 0` and :math:`c > 0`.

    `fisk` takes ``c`` as a shape parameter for :math:`c`.

    `fisk` is a special case of `burr` or `burr12` with ``d=1``.


.. _dist foldcauchy:

* ``foldcauchy`` **Folded Cauchy** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.foldcauchy.html>`_). The probability density function for `foldcauchy` is:

    .. math::

        f(x, c) = \frac{1}{\pi (1+(x-c)^2)} + \frac{1}{\pi (1+(x+c)^2)}

    for :math:`x \ge 0`.

    `foldcauchy` takes ``c`` as a shape parameter for :math:`c`.


.. _dist foldnorm:

* ``foldnorm`` **Folded Normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.foldnorm.html>`_). The probability density function for `foldnorm` is:

    .. math::

        f(x, c) = \sqrt{2/\pi} cosh(c x) \exp(-\frac{x^2+c^2}{2})

    for :math:`c \ge 0`.

    `foldnorm` takes ``c`` as a shape parameter for :math:`c`.


.. _dist genlogistic:

* ``genlogistic`` **Generalized Logistic** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genlogistic.html>`_). The probability density function for `genlogistic` is:

    .. math::

        f(x, c) = c \frac{\exp(-x)}
                         {(1 + \exp(-x))^{c+1}}

    for :math:`x >= 0`, :math:`c > 0`.

    `genlogistic` takes ``c`` as a shape parameter for :math:`c`.


.. _dist gennorm:

* ``gennorm`` **Generalized normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html>`_). The probability density function for `gennorm` is:

    .. math::

        f(x, \beta) = \frac{\beta}{2 \Gamma(1/\beta)} \exp(-|x|^\beta)

    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `gennorm` takes ``beta`` as a shape parameter for :math:`\beta`.
    For :math:`\beta = 1`, it is identical to a Laplace distribution.
    For :math:`\beta = 2`, it is identical to a normal distribution
    (with ``scale=1/sqrt(2)``).


.. _dist genpareto:

* ``genpareto`` **Generalized Pareto** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html>`_). The probability density function for `genpareto` is:

    .. math::

        f(x, c) = (1 + c x)^{-1 - 1/c}

    defined for :math:`x \ge 0` if :math:`c \ge 0`, and for
    :math:`0 \le x \le -1/c` if :math:`c < 0`.

    `genpareto` takes ``c`` as a shape parameter for :math:`c`.

    For :math:`c=0`, `genpareto` reduces to the exponential
    distribution, `expon`:

    .. math::

        f(x, 0) = \exp(-x)

    For :math:`c=-1`, `genpareto` is uniform on ``[0, 1]``:

    .. math::

        f(x, -1) = 1


.. _dist genexpon:

* ``genexpon`` **Generalized Exponential** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genexpon.html>`_). The probability density function for `genexpon` is:

    .. math::

        f(x, a, b, c) = (a + b (1 - \exp(-c x)))
                        \exp(-a x - b x + \frac{b}{c}  (1-\exp(-c x)))

    for :math:`x \ge 0`, :math:`a, b, c > 0`.

    `genexpon` takes :math:`a`, :math:`b` and :math:`c` as shape parameters.


.. _dist genextreme:

* ``genextreme`` **Generalized Extreme Value** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html>`_). For :math:`c=0`, `genextreme` is equal to `gumbel_r`. The probability density function for `genextreme` is:

    .. math::

        f(x, c) = \begin{cases}
                    \exp(-\exp(-x)) \exp(-x)              &\text{for } c = 0\\
                    \exp(-(1-c x)^{1/c}) (1-c x)^{1/c-1}  &\text{for }
                                                            x \le 1/c, c > 0
                  \end{cases}


    Note that several sources and software packages use the opposite
    convention for the sign of the shape parameter :math:`c`.

    `genextreme` takes ``c`` as a shape parameter for :math:`c`.


.. _dist gausshyper:

* ``gausshyper`` **Gauss Hypergeometric** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gausshyper.html>`_). The probability density function for `gausshyper` is:

    .. math::

        f(x, a, b, c, z) = C x^{a-1} (1-x)^{b-1} (1+zx)^{-c}

    for :math:`0 \le x \le 1`, :math:`a > 0`, :math:`b > 0`, :math:`z > -1`,
    and :math:`C = \frac{1}{B(a, b) F[2, 1](c, a; a+b; -z)}`.
    :math:`F[2, 1]` is the Gauss hypergeometric function
    `scipy.special.hyp2f1`.

    `gausshyper` takes :math:`a`, :math:`b`, :math:`c` and :math:`z` as shape
    parameters.


.. _dist gamma:

* ``gamma`` **Gamma** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html>`_). The probability density function for `gamma` is:

    .. math::

        f(x, a) = \frac{x^{a-1} e^{-x}}{\Gamma(a)}

    for :math:`x \ge 0`, :math:`a > 0`. Here :math:`\Gamma(a)` refers to the
    gamma function.

    `gamma` takes ``a`` as a shape parameter for :math:`a`.

    When :math:`a` is an integer, `gamma` reduces to the Erlang
    distribution, and when :math:`a=1` to the exponential distribution.

    Gamma distributions are sometimes parameterized with two variables,
    with a probability density function of:

    .. math::

        f(x, \alpha, \beta) = \frac{\beta^\alpha x^{\alpha - 1} e^{-\beta x }}{\Gamma(\alpha)}

    Note that this parameterization is equivalent to the above, with
    ``scale = 1 / beta``.


.. _dist gengamma:

* ``gengamma`` **Generalized gamma** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gengamma.html>`_). The probability density function for `gengamma` is:

    .. math::

        f(x, a, c) = \frac{|c| x^{c a-1} \exp(-x^c)}{\Gamma(a)}

    for :math:`x \ge 0`, :math:`a > 0`, and :math:`c \ne 0`.
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `gengamma` takes :math:`a` and :math:`c` as shape parameters.


.. _dist genhalflogistic:

* ``genhalflogistic`` **Generalized Half Logistic** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genhalflogistic.html>`_). The probability density function for `genhalflogistic` is:

    .. math::

        f(x, c) = \frac{2 (1 - c x)^{1/(c-1)}}{[1 + (1 - c x)^{1/c}]^2}

    for :math:`0 \le x \le 1/c`, and :math:`c > 0`.

    `genhalflogistic` takes ``c`` as a shape parameter for :math:`c`.


.. _dist genhyperbolic:

* ``genhyperbolic`` **Generalized Hyperbolic** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genhyperbolic.html>`_). The probability density function for `genhyperbolic` is:

    .. math::

        f(x, p, a, b) =
            \frac{(a^2 - b^2)^{p/2}}
            {\sqrt{2\pi}a^{p-0.5}
            K_p\Big(\sqrt{a^2 - b^2}\Big)}
            e^{bx} \times \frac{K_{p - 1/2}
            (a \sqrt{1 + x^2})}
            {(\sqrt{1 + x^2})^{1/2 - p}}

    for :math:`x, p \in ( - \infty; \infty)`,
    :math:`|b| < a` if :math:`p \ge 0`,
    :math:`|b| \le a` if :math:`p < 0`.
    :math:`K_{p}(.)` denotes the modified Bessel function of the second
    kind and order :math:`p` (`scipy.special.kn`)

    `genhyperbolic` takes ``p`` as a tail parameter,
    ``a`` as a shape parameter,
    ``b`` as a skewness parameter.


.. _dist geninvgauss:

* ``geninvgauss`` **Generalized Inverse Gaussian** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geninvgauss.html>`_). The probability density function for `geninvgauss` is:

    .. math::

        f(x, p, b) = x^{p-1} \exp(-b (x + 1/x) / 2) / (2 K_p(b))

    where `x > 0`, and the parameters `p, b` satisfy `b > 0`.
    :math:`K_p` is the modified Bessel function of second kind of order `p`
    (`scipy.special.kv`).


.. _dist gilbrat:

* ``gilbrat`` **Gilbrat** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gilbrat.html>`_). The probability density function for `gilbrat` is:

    .. math::

        f(x) = \frac{1}{x \sqrt{2\pi}} \exp(-\frac{1}{2} (\log(x))^2)

    `gilbrat` is a special case of `lognorm` with ``s=1``.


.. _dist gompertz:

* ``gompertz`` **Gompertz (Truncated Gumbel)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gompertz.html>`_). The probability density function for `gompertz` is:

    .. math::

        f(x, c) = c \exp(x) \exp(-c (e^x-1))

    for :math:`x \ge 0`, :math:`c > 0`.

    `gompertz` takes ``c`` as a shape parameter for :math:`c`.


.. _dist gumbel_r:

* ``gumbel_r`` (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html>`_). The probability density function for `gumbel_r` is:

    .. math::

        f(x) = \exp(-(x + e^{-x}))

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.


.. _dist gumbel_l:

* ``gumbel_l`` (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_l.html>`_). The probability density function for `gumbel_l` is:

    .. math::

        f(x) = \exp(x - e^x)

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.


.. _dist halfcauchy:

* ``halfcauchy`` **Half Cauchy** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfcauchy.html>`_). The probability density function for `halfcauchy` is:

    .. math::

        f(x) = \frac{2}{\pi (1 + x^2)}

    for :math:`x \ge 0`.


.. _dist halflogistic:

* ``halflogistic`` **Half Logistic** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halflogistic.html>`_). The probability density function for `halflogistic` is:

    .. math::

        f(x) = \frac{ 2 e^{-x} }{ (1+e^{-x})^2 }
             = \frac{1}{2} \text{sech}(x/2)^2

    for :math:`x \ge 0`.


.. _dist halfnorm:

* ``halfnorm`` **Half Normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfnorm.html>`_). The probability density function for `halfnorm` is:

    .. math::

        f(x) = \sqrt{2/\pi} \exp(-x^2 / 2)

    for :math:`x >= 0`.

    `halfnorm` is a special case of `chi` with ``df=1``.


.. _dist halfgennorm:

* ``halfgennorm`` **Generalized Half Normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfgennorm.html>`_). The probability density function for `halfgennorm` is:

    .. math::

        f(x, \beta) = \frac{\beta}{\Gamma(1/\beta)} \exp(-|x|^\beta)

    for :math:`x > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `gennorm` takes ``beta`` as a shape parameter for :math:`\beta`.
    For :math:`\beta = 1`, it is identical to an exponential distribution.
    For :math:`\beta = 2`, it is identical to a half normal distribution
    (with ``scale=1/sqrt(2)``).


.. _dist hypsecant:

* ``hypsecant`` **Hyperbolic Secant** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypsecant.html>`_). The probability density function for `hypsecant` is:

    .. math::

        f(x) = \frac{1}{\pi} \text{sech}(x)

    for a real number :math:`x`.


.. _dist invgamma:

* ``invgamma`` **Inverse Gamma** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html>`_). The probability density function for `invgamma` is:

    .. math::

        f(x, a) = \frac{x^{-a-1}}{\Gamma(a)} \exp(-\frac{1}{x})

    for :math:`x >= 0`, :math:`a > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `invgamma` takes ``a`` as a shape parameter for :math:`a`.

    `invgamma` is a special case of `gengamma` with ``c=-1``, and it is a
    different parameterization of the scaled inverse chi-squared distribution.
    Specifically, if the scaled inverse chi-squared distribution is
    parameterized with degrees of freedom :math:`\nu` and scaling parameter
    :math:`\tau^2`, then it can be modeled using `invgamma` with
    ``a=`` :math:`\nu/2` and ``scale=`` :math:`\nu \tau^2/2`.


.. _dist invgauss:

* ``invgauss`` **Inverse Gaussian** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html>`_). The probability density function for `invgauss` is:

    .. math::

        f(x, \mu) = \frac{1}{\sqrt{2 \pi x^3}}
                    \exp(-\frac{(x-\mu)^2}{2 x \mu^2})

    for :math:`x >= 0` and :math:`\mu > 0`.

    `invgauss` takes ``mu`` as a shape parameter for :math:`\mu`.


.. _dist invweibull:

* ``invweibull`` **Inverse Weibull** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html>`_). The probability density function for `invweibull` is:

    .. math::

        f(x, c) = c x^{-c-1} \exp(-x^{-c})

    for :math:`x > 0`, :math:`c > 0`.

    `invweibull` takes ``c`` as a shape parameter for :math:`c`.


.. _dist johnsonsb:

* ``johnsonsb`` **Johnson SB** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsb.html>`_). The probability density function for `johnsonsb` is:

    .. math::

        f(x, a, b) = \frac{b}{x(1-x)}  \phi(a + b \log \frac{x}{1-x} )

    where :math:`x`, :math:`a`, and :math:`b` are real scalars; :math:`b > 0`
    and :math:`x \in [0,1]`.  :math:`\phi` is the pdf of the normal
    distribution.

    `johnsonsb` takes :math:`a` and :math:`b` as shape parameters.


.. _dist johnsonsu:

* ``johnsonsu`` **Johnson SU** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsu.html>`_). The probability density function for `johnsonsu` is:

    .. math::

        f(x, a, b) = \frac{b}{\sqrt{x^2 + 1}}
                     \phi(a + b \log(x + \sqrt{x^2 + 1}))

    where :math:`x`, :math:`a`, and :math:`b` are real scalars; :math:`b > 0`.
    :math:`\phi` is the pdf of the normal distribution.

    `johnsonsu` takes :math:`a` and :math:`b` as shape parameters.


.. _dist kappa4:

* ``kappa4`` **Kappa 4 parameter** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa4.html>`_). The probability density function for kappa4 is:

    .. math::

        f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}

    if :math:`h` and :math:`k` are not equal to 0.

    If :math:`h` or :math:`k` are zero then the pdf can be simplified:

    h = 0 and k != 0::

        kappa4.pdf(x, h, k) = (1.0 - k*x)**(1.0/k - 1.0)*
                              exp(-(1.0 - k*x)**(1.0/k))

    h != 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*(1.0 - h*exp(-x))**(1.0/h - 1.0)

    h = 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*exp(-exp(-x))

    kappa4 takes :math:`h` and :math:`k` as shape parameters.

    The kappa4 distribution returns other distributions when certain
    :math:`h` and :math:`k` values are used.

    +------+-------------+----------------+------------------+
    | h    | k=0.0       | k=1.0          | -inf<=k<=inf     |
    +======+=============+================+==================+
    | -1.0 | Logistic    |                | Generalized      |
    |      |             |                | Logistic(1)      |
    |      |             |                |                  |
    |      | logistic(x) |                |                  |
    +------+-------------+----------------+------------------+
    |  0.0 | Gumbel      | Reverse        | Generalized      |
    |      |             | Exponential(2) | Extreme Value    |
    |      |             |                |                  |
    |      | gumbel_r(x) |                | genextreme(x, k) |
    +------+-------------+----------------+------------------+
    |  1.0 | Exponential | Uniform        | Generalized      |
    |      |             |                | Pareto           |
    |      |             |                |                  |
    |      | expon(x)    | uniform(x)     | genpareto(x, -k) |
    +------+-------------+----------------+------------------+


.. _dist kappa3:

* ``kappa3`` **Kappa 3 parameter** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa3.html>`_). The probability density function for `kappa3` is:

    .. math::

        f(x, a) = a (a + x^a)^{-(a + 1)/a}

    for :math:`x > 0` and :math:`a > 0`.

    `kappa3` takes ``a`` as a shape parameter for :math:`a`.


.. _dist ksone:

* ``ksone`` **Distribution of Kolmogorov-Smirnov one-sided test statistic** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ksone.html>`_). :math:`D_n^+` and :math:`D_n^-` are given by

    .. math::

        D_n^+ &= \text{sup}_x (F_n(x) - F(x)),\\
        D_n^- &= \text{sup}_x (F(x) - F_n(x)),\\

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `ksone` describes the distribution under the null hypothesis of the KS test
    that the empirical CDF corresponds to :math:`n` i.i.d. random variates
    with CDF :math:`F`.


.. _dist kstwo:

* ``kstwo`` **Distribution of Kolmogorov-Smirnov two-sided test statistic** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwo.html>`_). :math:`D_n` is given by

    .. math::

        D_n = \text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a (continuous) CDF and :math:`F_n` is an empirical CDF.
    `kstwo` describes the distribution under the null hypothesis of the KS test
    that the empirical CDF corresponds to :math:`n` i.i.d. random variates
    with CDF :math:`F`.


.. _dist kstwobign:

* ``kstwobign`` **Limiting Distribution of scaled Kolmogorov-Smirnov two-sided test statistic.** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwobign.html>`_). :math:`\sqrt{n} D_n` is given by

    .. math::

        D_n = \text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `kstwobign`  describes the asymptotic distribution (i.e. the limit of
    :math:`\sqrt{n} D_n`) under the null hypothesis of the KS test that the
    empirical CDF corresponds to i.i.d. random variates with CDF :math:`F`.


.. _dist laplace:

* ``laplace`` **Laplace** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html>`_). The probability density function for `laplace` is

    .. math::

        f(x) = \frac{1}{2} \exp(-|x|)

    for a real number :math:`x`.


.. _dist laplace_asymmetric:

* ``laplace_asymmetric`` (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace_asymmetric.html>`_). The probability density function for `laplace_asymmetric` is

    .. math::

       f(x, \kappa) &= \frac{1}{\kappa+\kappa^{-1}}\exp(-x\kappa),\quad x\ge0\\
                    &= \frac{1}{\kappa+\kappa^{-1}}\exp(x/\kappa),\quad x<0\\

    for :math:`-\infty < x < \infty`, :math:`\kappa > 0`.

    `laplace_asymmetric` takes ``kappa`` as a shape parameter for
    :math:`\kappa`. For :math:`\kappa = 1`, it is identical to a
    Laplace distribution.


.. _dist levy:

* ``levy`` **Levy** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy.html>`_). The probability density function for `levy` is:

    .. math::

        f(x) = \frac{1}{\sqrt{2\pi x^3}} \exp\left(-\frac{1}{2x}\right)

    for :math:`x >= 0`.

    This is the same as the Levy-stable distribution with :math:`a=1/2` and
    :math:`b=1`.


.. _dist logistic:

* ``logistic`` **Logistic** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html>`_). The probability density function for `logistic` is:

    .. math::

        f(x) = \frac{\exp(-x)}
                    {(1+\exp(-x))^2}

    `logistic` is a special case of `genlogistic` with ``c=1``.

    Remark that the survival function (``logistic.sf``) is equal to the
    Fermi-Dirac distribution describing fermionic statistics.


.. _dist loggamma:

* ``loggamma`` **Log-Gamma** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loggamma.html>`_). The probability density function for `loggamma` is:

    .. math::

        f(x, c) = \frac{\exp(c x - \exp(x))}
                       {\Gamma(c)}

    for all :math:`x, c > 0`. Here, :math:`\Gamma` is the
    gamma function (`scipy.special.gamma`).

    `loggamma` takes ``c`` as a shape parameter for :math:`c`.


.. _dist loglaplace:

* ``loglaplace`` **Log-Laplace (Log Double Exponential)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loglaplace.html>`_). The probability density function for `loglaplace` is:

    .. math::

        f(x, c) = \begin{cases}\frac{c}{2} x^{ c-1}  &\text{for } 0 < x < 1\\
                               \frac{c}{2} x^{-c-1}  &\text{for } x \ge 1
                  \end{cases}

    for :math:`c > 0`.

    `loglaplace` takes ``c`` as a shape parameter for :math:`c`.


.. _dist lognorm:

* ``lognorm`` **Log-Normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html>`_). The probability density function for `lognorm` is:

    .. math::

        f(x, s) = \frac{1}{s x \sqrt{2\pi}}
                  \exp\left(-\frac{\log^2(x)}{2s^2}\right)

    for :math:`x > 0`, :math:`s > 0`.

    `lognorm` takes ``s`` as a shape parameter for :math:`s`.


.. _dist loguniform:

* ``loguniform`` **Log-Uniform** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html>`_). The probability density function for this class is:

    .. math::

        f(x, a, b) = \frac{1}{x \log(b/a)}

    for :math:`a \le x \le b`, :math:`b > a > 0`. This class takes
    :math:`a` and :math:`b` as shape parameters.


.. _dist lomax:

* ``lomax`` **Lomax (Pareto of the second kind)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lomax.html>`_). The probability density function for `lomax` is:

    .. math::

        f(x, c) = \frac{c}{(1+x)^{c+1}}

    for :math:`x \ge 0`, :math:`c > 0`.

    `lomax` takes ``c`` as a shape parameter for :math:`c`.

    `lomax` is a special case of `pareto` with ``loc=-1.0``.


.. _dist maxwell:

* ``maxwell`` **Maxwell** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.maxwell.html>`_). A special case of a `chi` distribution,  with ``df=3``, ``loc=0.0``, and given ``scale = a``, where ``a`` is the parameter used in the Mathworld description.

    The probability density function for `maxwell` is:

    .. math::

        f(x) = \sqrt{2/\pi}x^2 \exp(-x^2/2)

    for :math:`x >= 0`.


.. _dist mielke:

* ``mielke`` **Mielke's Beta-Kappa** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mielke.html>`_). The probability density function for `mielke` is:

    .. math::

        f(x, k, s) = \frac{k x^{k-1}}{(1+x^s)^{1+k/s}}

    for :math:`x > 0` and :math:`k, s > 0`. The distribution is sometimes
    called Dagum distribution. It was already defined in, called
    a Burr Type III distribution (`burr` with parameters ``c=s`` and
    ``d=k/s``).

    `mielke` takes ``k`` and ``s`` as shape parameters.


.. _dist moyal:

* ``moyal`` **Moyal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moyal.html>`_). The probability density function for `moyal` is:

    .. math::

        f(x) = \exp(-(x + \exp(-x))/2) / \sqrt{2\pi}

    for a real number :math:`x`.


.. _dist nakagami:

* ``nakagami`` **Nakagami** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nakagami.html>`_). The probability density function for `nakagami` is:

    .. math::

        f(x, \nu) = \frac{2 \nu^\nu}{\Gamma(\nu)} x^{2\nu-1} \exp(-\nu x^2)

    for :math:`x >= 0`, :math:`\nu > 0`.

    `nakagami` takes ``nu`` as a shape parameter for :math:`\nu`.


.. _dist ncx2:

* ``ncx2`` **Non-central chi-squared** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html>`_). The probability density function for `ncx2` is:

    .. math::

        f(x, k, \lambda) = \frac{1}{2} \exp(-(\lambda+x)/2)
            (x/\lambda)^{(k-2)/4}  I_{(k-2)/2}(\sqrt{\lambda x})

    for :math:`x >= 0` and :math:`k, \lambda > 0`. :math:`k` specifies the
    degrees of freedom (denoted ``df`` in the implementation) and
    :math:`\lambda` is the non-centrality parameter (denoted ``nc`` in the
    implementation). :math:`I_\nu` denotes the modified Bessel function of
    first order of degree :math:`\nu` (`scipy.special.iv`).

    `ncx2` takes ``df`` and ``nc`` as shape parameters.


.. _dist ncf:

* ``ncf`` **Non-central F** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html>`_). The probability density function for `ncf` is:

    .. math::

        f(x, n_1, n_2, \lambda) =
            \exp\left(\frac{\lambda}{2} +
                      \lambda n_1 \frac{x}{2(n_1 x + n_2)}
                \right)
            n_1^{n_1/2} n_2^{n_2/2} x^{n_1/2 - 1} \\
            (n_2 + n_1 x)^{-(n_1 + n_2)/2}
            \gamma(n_1/2) \gamma(1 + n_2/2) \\
            \frac{L^{\frac{n_1}{2}-1}_{n_2/2}
                \left(-\lambda n_1 \frac{x}{2(n_1 x + n_2)}\right)}
            {B(n_1/2, n_2/2)
                \gamma\left(\frac{n_1 + n_2}{2}\right)}

    for :math:`n_1, n_2 > 0`, :math:`\lambda \ge 0`.  Here :math:`n_1` is the
    degrees of freedom in the numerator, :math:`n_2` the degrees of freedom in
    the denominator, :math:`\lambda` the non-centrality parameter,
    :math:`\gamma` is the logarithm of the Gamma function, :math:`L_n^k` is a
    generalized Laguerre polynomial and :math:`B` is the beta function.

    `ncf` takes ``df1``, ``df2`` and ``nc`` as shape parameters. If ``nc=0``,
    the distribution becomes equivalent to the Fisher distribution.


.. _dist nct:

* ``nct`` **Non-central Student's T** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html>`_). If :math:`Y` is a standard normal random variable and :math:`V` is an independent chi-square random variable (`chi2`) with :math:`k` degrees of freedom, then

    .. math::

        X = \frac{Y + c}{\sqrt{V/k}}

    has a non-central Student's t distribution on the real line.
    The degrees of freedom parameter :math:`k` (denoted ``df`` in the
    implementation) satisfies :math:`k > 0` and the noncentrality parameter
    :math:`c` (denoted ``nc`` in the implementation) is a real number.


.. _dist norm:

* ``norm`` **Normal (Gaussian)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`_). The probability density function for `norm` is:

    .. math::

        f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}

    for a real number :math:`x`.


.. _dist norminvgauss:

* ``norminvgauss`` **Normal Inverse Gaussian** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html>`_). The probability density function for `norminvgauss` is:

    .. math::

        f(x, a, b) = \frac{a \, K_1(a \sqrt{1 + x^2})}{\pi \sqrt{1 + x^2}} \,
                     \exp(\sqrt{a^2 - b^2} + b x)

    where :math:`x` is a real number, the parameter :math:`a` is the tail
    heaviness and :math:`b` is the asymmetry parameter satisfying
    :math:`a > 0` and :math:`|b| <= a`.
    :math:`K_1` is the modified Bessel function of second kind
    (`scipy.special.k1`).


.. _dist pareto:

* ``pareto`` **Pareto** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html>`_). The probability density function for `pareto` is:

    .. math::

        f(x, b) = \frac{b}{x^{b+1}}

    for :math:`x \ge 1`, :math:`b > 0`.

    `pareto` takes ``b`` as a shape parameter for :math:`b`.


.. _dist pearson3:

* ``pearson3`` **Pearson type III** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html>`_). The probability density function for `pearson3` is:

    .. math::

        f(x, \kappa) = \frac{|\beta|}{\Gamma(\alpha)}
                       (\beta (x - \zeta))^{\alpha - 1}
                       \exp(-\beta (x - \zeta))

    where:

    .. math::

            \beta = \frac{2}{\kappa}

            \alpha = \beta^2 = \frac{4}{\kappa^2}

            \zeta = -\frac{\alpha}{\beta} = -\beta

    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).
    Pass the skew :math:`\kappa` into `pearson3` as the shape parameter
    ``skew``.


.. _dist powerlaw:

* ``powerlaw`` **Power-function** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlaw.html>`_). The probability density function for `powerlaw` is:

    .. math::

        f(x, a) = a x^{a-1}

    for :math:`0 \le x \le 1`, :math:`a > 0`.

    `powerlaw` takes ``a`` as a shape parameter for :math:`a`.


.. _dist powerlognorm:

* ``powerlognorm`` **Power log normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlognorm.html>`_). The probability density function for `powerlognorm` is:

    .. math::

        f(x, c, s) = \frac{c}{x s} \phi(\log(x)/s)
                     (\Phi(-\log(x)/s))^{c-1}

    where :math:`\phi` is the normal pdf, and :math:`\Phi` is the normal cdf,
    and :math:`x > 0`, :math:`s, c > 0`.

    `powerlognorm` takes :math:`c` and :math:`s` as shape parameters.


.. _dist powernorm:

* ``powernorm`` **Power normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powernorm.html>`_). The probability density function for `powernorm` is:

    .. math::

        f(x, c) = c \phi(x) (\Phi(-x))^{c-1}

    where :math:`\phi` is the normal pdf, and :math:`\Phi` is the normal cdf,
    and :math:`x >= 0`, :math:`c > 0`.

    `powernorm` takes ``c`` as a shape parameter for :math:`c`.


.. _dist rdist:

* ``rdist`` **R-distribution** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rdist.html>`_). The probability density function for `rdist` is:

    .. math::

        f(x, c) = \frac{(1-x^2)^{c/2-1}}{B(1/2, c/2)}

    for :math:`-1 \le x \le 1`, :math:`c > 0`. `rdist` is also called the
    symmetric beta distribution: if B has a `beta` distribution with
    parameters (c/2, c/2), then X = 2*B - 1 follows a R-distribution with
    parameter c.

    `rdist` takes ``c`` as a shape parameter for :math:`c`.

    This distribution includes the following distribution kernels as
    special cases::

        c = 2:  uniform
        c = 3:  `semicircular`
        c = 4:  Epanechnikov (parabolic)
        c = 6:  quartic (biweight)
        c = 8:  triweight


.. _dist rayleigh:

* ``rayleigh`` **Rayleigh** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html>`_). The probability density function for `rayleigh` is:

    .. math::

        f(x) = x \exp(-x^2/2)

    for :math:`x \ge 0`.

    `rayleigh` is a special case of `chi` with ``df=2``.


.. _dist rice:

* ``rice`` **Rice** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rice.html>`_). The probability density function for `rice` is:

    .. math::

        f(x, b) = x \exp(- \frac{x^2 + b^2}{2}) I_0(x b)

    for :math:`x >= 0`, :math:`b > 0`. :math:`I_0` is the modified Bessel
    function of order zero (`scipy.special.i0`).

    `rice` takes ``b`` as a shape parameter for :math:`b`.


.. _dist recipinvgauss:

* ``recipinvgauss`` **Reciprocal Inverse Gaussian** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.recipinvgauss.html>`_). The probability density function for `recipinvgauss` is:

    .. math::

        f(x, \mu) = \frac{1}{\sqrt{2\pi x}}
                    \exp\left(\frac{-(1-\mu x)^2}{2\mu^2x}\right)

    for :math:`x \ge 0`.

    `recipinvgauss` takes ``mu`` as a shape parameter for :math:`\mu`.


.. _dist semicircular:

* ``semicircular`` **Semicircular** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.semicircular.html>`_). The probability density function for `semicircular` is:

    .. math::

        f(x) = \frac{2}{\pi} \sqrt{1-x^2}

    for :math:`-1 \le x \le 1`.

    The distribution is a special case of `rdist` with `c = 3`.


.. _dist skewcauchy:

* ``skewcauchy`` **Skew Cauchy** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewcauchy.html>`_). The probability density function for `skewcauchy` is:

    .. math::

        f(x) = \frac{1}{\pi \left(\frac{x^2}{\left(a\, \text{sign}(x) + 1
                                                   \right)^2} + 1 \right)}

    for a real number :math:`x` and skewness parameter :math:`-1 < a < 1`.

    When :math:`a=0`, the distribution reduces to the usual Cauchy
    distribution.


.. _dist skewnorm:

* ``skewnorm`` **Skew normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html>`_). The pdf is::

        skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)

  `skewnorm` takes a real number :math:`a` as a skewness parameter.
  When ``a = 0`` the distribution is identical to a normal distribution
  (`norm`).


.. _dist studentized_range:

* ``studentized_range`` (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.studentized_range.html>`_). The probability density function for `studentized_range` is:

    .. math::

         f(x; k, \nu) = \frac{k(k-1)\nu^{\nu/2}}{\Gamma(\nu/2)
                        2^{\nu/2-1}} \int_{0}^{\infty} \int_{-\infty}^{\infty}
                        s^{\nu} e^{-\nu s^2/2} \phi(z) \phi(sx + z)
                        [\Phi(sx + z) - \Phi(z)]^{k-2} \,dz \,ds

    for :math:`x ≥ 0`, :math:`k > 1`, and :math:`\nu > 0`.

    `studentized_range` takes ``k`` for :math:`k` and ``df`` for :math:`\nu`
    as shape parameters.

    When :math:`\nu` exceeds 100,000, an asymptotic approximation (infinite
    degrees of freedom) is used to compute the cumulative distribution
    function.


.. _dist t:

* ``t`` **Student's T** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html>`_). The probability density function for `t` is:

    .. math::

        f(x, \nu) = \frac{\Gamma((\nu+1)/2)}
                        {\sqrt{\pi \nu} \Gamma(\nu/2)}
                    (1+x^2/\nu)^{-(\nu+1)/2}

    where :math:`x` is a real number and the degrees of freedom parameter
    :math:`\nu` (denoted ``df`` in the implementation) satisfies
    :math:`\nu > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).


.. _dist trapezoid:

* ``trapezoid`` **Trapezoidal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trapezoid.html>`_). The trapezoidal distribution can be represented with an up-sloping line from ``loc`` to ``(loc + c*scale)``, then constant to ``(loc + d*scale)`` and then downsloping from ``(loc + d*scale)`` to ``(loc+scale)``.  This defines the trapezoid base from ``loc`` to ``(loc+scale)`` and the flat top from ``c`` to ``d`` proportional to the position along the base with ``0 <= c <= d <= 1``.  When ``c=d``, this is equivalent to `triang` with the same values for `loc`, `scale` and `c`.

  `trapezoid` takes :math:`c` and :math:`d` as shape parameters.


.. _dist triang:

* ``triang`` **Triangular** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html>`_). The triangular distribution can be represented with an up-sloping line from ``loc`` to ``(loc + c*scale)`` and then downsloping for ``(loc + c*scale)`` to ``(loc + scale)``.

  `triang` takes ``c`` as a shape parameter for :math:`c`.


.. _dist truncexpon:

* ``truncexpon`` **Truncated Exponential** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncexpon.html>`_). The probability density function for `truncexpon` is:

    .. math::

        f(x, b) = \frac{\exp(-x)}{1 - \exp(-b)}

    for :math:`0 <= x <= b`.

    `truncexpon` takes ``b`` as a shape parameter for :math:`b`.


.. _dist truncnorm:

* ``truncnorm`` **Truncated Normal** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html>`_). The standard form of this distribution is a standard normal truncated to the range [a, b] --- notice that a and b are defined over the domain of the standard normal.  To convert clip values for a specific mean and standard deviation, use::

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

  `truncnorm` takes :math:`a` and :math:`b` as shape parameters.


.. _dist tukeylambda:

* ``tukeylambda`` **Tukey-Lambda** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukeylambda.html>`_). A flexible distribution, able to represent and interpolate between the following distributions:

    - Cauchy                (:math:`lambda = -1`)
    - logistic              (:math:`lambda = 0`)
    - approx Normal         (:math:`lambda = 0.14`)
    - uniform from -1 to 1  (:math:`lambda = 1`)

    `tukeylambda` takes a real number :math:`lambda` (denoted ``lam``
    in the implementation) as a shape parameter.


.. _dist uniform:

* ``uniform`` **Uniform** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html>`_). a uniform continuous random variable


.. _dist vonmises:

* ``vonmises`` **Von-Mises (Circular)** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html>`_). The probability density function for `vonmises` and `vonmises_line` is:

    .. math::

        f(x, \kappa) = \frac{ \exp(\kappa \cos(x)) }{ 2 \pi I_0(\kappa) }

    for :math:`-\pi \le x \le \pi`, :math:`\kappa > 0`. :math:`I_0` is the
    modified Bessel function of order zero (`scipy.special.i0`).

    `vonmises` is a circular distribution which does not restrict the
    distribution to a fixed interval. Currently, there is no circular
    distribution framework in scipy. The ``cdf`` is implemented such that
    ``cdf(x + 2*np.pi) == cdf(x) + 1``.

    `vonmises_line` is the same distribution, defined on :math:`[-\pi, \pi]`
    on the real line. This is a regular (i.e. non-circular) distribution.

    `vonmises` and `vonmises_line` take ``kappa`` as a shape parameter.


.. _dist vonmises_line:

* ``vonmises_line`` (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_line.html>`_). The probability density function for `vonmises` and `vonmises_line` is:

    .. math::

        f(x, \kappa) = \frac{ \exp(\kappa \cos(x)) }{ 2 \pi I_0(\kappa) }

    for :math:`-\pi \le x \le \pi`, :math:`\kappa > 0`. :math:`I_0` is the
    modified Bessel function of order zero (`scipy.special.i0`).

    `vonmises` is a circular distribution which does not restrict the
    distribution to a fixed interval. Currently, there is no circular
    distribution framework in scipy. The ``cdf`` is implemented such that
    ``cdf(x + 2*np.pi) == cdf(x) + 1``.

    `vonmises_line` is the same distribution, defined on :math:`[-\pi, \pi]`
    on the real line. This is a regular (i.e. non-circular) distribution.

    `vonmises` and `vonmises_line` take ``kappa`` as a shape parameter.


.. _dist wald:

* ``wald`` **Wald** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wald.html>`_). The probability density function for `wald` is:

    .. math::

        f(x) = \frac{1}{\sqrt{2\pi x^3}} \exp(- \frac{ (x-1)^2 }{ 2x })

    for :math:`x >= 0`.

    `wald` is a special case of `invgauss` with ``mu=1``.


.. _dist weibull_min:

* ``weibull_min`` (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html>`_). The probability density function for `weibull_min` is:

    .. math::

        f(x, c) = c x^{c-1} \exp(-x^c)

    for :math:`x > 0`, :math:`c > 0`.

    `weibull_min` takes ``c`` as a shape parameter for :math:`c`.
    (named :math:`k` in Wikipedia article and :math:`a` in
    ``numpy.random.weibull``).  Special shape values are :math:`c=1` and
    :math:`c=2` where Weibull distribution reduces to the `expon` and
    `rayleigh` distributions respectively.


.. _dist weibull_max:

* ``weibull_max`` (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_max.html>`_). The probability density function for `weibull_max` is:

    .. math::

        f(x, c) = c (-x)^{c-1} \exp(-(-x)^c)

    for :math:`x < 0`, :math:`c > 0`.

    `weibull_max` takes ``c`` as a shape parameter for :math:`c`.


.. _dist wrapcauchy:

* ``wrapcauchy`` **Wrapped Cauchy** (`help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html>`_). The probability density function for `wrapcauchy` is:

    .. math::

        f(x, c) = \frac{1-c^2}{2\pi (1+c^2 - 2c \cos(x))}

    for :math:`0 \le x \le 2\pi`, :math:`0 < c < 1`.

    `wrapcauchy` takes ``c`` as a shape parameter for :math:`c`.


.. code to create: see blog/agg/examples/probems_and_solutions.ipynb

