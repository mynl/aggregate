.. _2_x_severity:

.. _2_agg_class_severity_clause:

.. reviewed 2022-12-24

The Severity Clause
----------------------

The severity clause specifies the ground-up severity distribution, or "severity curve" as it is sometimes known. It is a very flexible clause. Its design follows the ``scipy.stats`` package's specification of random variables using shape, location, and scale factors, see :ref:`probability background <5_x_probability>`. The syntax is different for non-parametric discrete distributions and parametric continuous distributions.


.. _nonparametric severity:

Non-Parametric Severity Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discrete distributions (supported on a finite number of outcomes)
can be directly entered as a severity using the ``dsev`` keyword followed by
two equal-length row vectors. The first gives the outcomes and the (optional) second gives the
probabilities.

::

    dsev [outcomes] <[probabilities]>

The horizontal layout is irrelevant and commas are optional.
If the ``probabilities`` vector is omitted then all probabilities are set equal to
the reciprocal of the length of the ``outcomes`` vector.
A Python-like colon notation is available for ranges.
Probabilities can be entered as fractions, but no other arithmetic operation is supported.

**Examples**::

    dsev [0 9 10] [0.5 0.3 0.2]
    dsev [0 9 10]
    dsev [1:6]
    dsev [0:100:25]
    dsev [1:6] [1/4 1/4 1/8 1/8 1/8 1/8]


* ``dsev [0 9 10] [0.5 0.3 0.2]`` is a severity with a 0.5 chance of taking the value 0, 0.3 chance of 9, and 0.2 of 10.
* ``dsev [0 9 10]`` gives equally likely outcomes of 0, 9, or 10.
* ``dsev [1:6]`` gives equally likely outcomes 1, 2, 3, 4, 5, 6. Unlike Python (but like ``pandas.DataFrame.loc``) the right-hand limit is included.
* ``dsev [0:100:25]`` gives qually likely outcomes 0, 25, 50, 100.
* ``dsev [1:6] [1/4 1/4 1/8 1/8 1/8 1/8]`` gives outcomes 1 or 2 with probability 0.25 or 3-6 with probability 0.125.

.. warning::
    Use binary fractions (denominator a power of two) to avoid rounding errors!

Details
"""""""""""

A ``dsev`` clause is converted by the parser into a ``dhistogram`` step distribution::

    sev dhistogram xps [outcomes] [probabilities]

In rare cases you want a continuous (ogive, piecewise linear distribution) version::

    sev chistogram xps [outcomes] [probabilities]

When executed, these are both converted into a ``scipy.stats`` ``histogram`` class.

Discrete severities, specified using the ``dsev`` keyword, are implemented using a ``scipy.stats`` ``rv_historgram`` object, which is actually continuous. They work by concentrating the probability in small intervals just to the left of each knot point (to make the function right continuous). Given::

    dsev [xs] [ps]

where ``xs`` and ``ps`` are the vectors of outcomes and probabilities, internally ``aggregate`` creates::

   xss = np.sort(np.hstack((xs - 2 ** -30, xs)))
   pss = np.vstack((ps1, np.zeros_like(ps1))).reshape((-1,), order='F')[:-1]
   fz_discr = ss.rv_histogram((pss, xss))

The value ``2**-30`` needs to be smaller than the bucket size resolution, i.e., enough not to “split the bucket”. The mass is to the left of the knot to make a right continuous function (the approximation ramps up before the knot). Generally histograms are downsampled, not upsampled, so this is not a restriction.

A ``dsev`` statement is translated into the more general::

    sev dhistorgram xps [xs] [ps]

where ``dhistrogram`` creates a discrete histogram (as above) and the ``xps`` keyword prefixes inputting the knots and probabilities. It is also possible to specify the input severity as a continuous histogram that is uniform on :math:`(x_k, x_{k+1}]`. The discrete probabilities are :math:`p_k=P(x_k < X \le x_{k+1})`. To create a rv_histogram variable is much easier, just use::

    sev chistorgram xps [xs] [ps]

which is translated into::

    xs2 = np.hstack((xs, xs[-1] + xs[1]))
    fz_cts = ss.rv_histogram((ps2, xs2))

The code adds an additional knot at the end to create enough differences (there are only two differences between three points). The :ref:`num robertson` uses a ``chistogram``.

The discrete method is appropriate when the distribution will be used and interpreted as fully discrete, which is the assumption the FFT method makes and the default. The continuous method is useful if the distribution will be used to create a scipy.stats rv_histogram variable. If the continuous method is interpreted as discrete and if the mean is computed as :math:`\sum_i p_k x_k`, which is appropriate for a discrete variable, then it will be under-estimated by :math:`b/2`.

Parametric Severity
~~~~~~~~~~~~~~~~~~~~~

A parametric distribution can be specified in two ways::

    sev DIST_NAME MEAN cv CV
    sev DIST_NAME <SHAPE1> <SHAPE2>

where

* ``sev`` is a keyword indicating the severity specification,
* ``DIST_NAME`` is the ``scipy.stats`` distribution name, see :ref:`available sev dists`,
* ``MEAN`` is the expected loss,
* ``cv`` (lowercase) is a keyword indicating entry of the CV,
* ``CV`` is the loss coefficient of variation, and
* ``SHAPE1``, ``SHAPE2`` are the (optional) shape variables.

The first form enters the expected ground-up severity and CV directly. It is available for distributions with only one shape parameter and the beta distribution on :math:`[0,1]`. ``aggregate`` uses a formula (lognormal, gamma, beta) or numerical method (all other one shape parameter distributions) to solve for the shape parameter to achieve the correct CV and then scales to the desired mean. The second form directly enters the shape variable(s). Shape parameters entered for zero parameter distributions are ignored.

**Example.** Entering ``sev lognorm 10 cv 0.2`` produces a lognormal
distribution with a mean of 10 and a CV of 0.2. Entering ``lognorm 0.2`` produces a lognormal
with :math:`\mu=0` and :math:`\sigma=0.2`, which can then be :ref:`scaled and shifted<dec shift scale>`.

``DIST_NAME`` can be any zero, one, or two shape parameter ``scipy.stats`` continuous distribution.
They have (mostly) easy to guess names. For example:

* Distributions with no shape parameters include:
  ``norm``, Gaussian normal; ``unif``, uniform; and ``expon``, the exponential.

* Distributions with one shape parameter include:
  ``pareto``, ``lognorm``, ``gamma``, ``invgamma``, ``loggamma``, and ``weibull_min`` the Weibull.

* Distributions with two shape parameters include:
  ``beta`` and ``gengamma``, the generalized gamma.

See :ref:`available sev dists` for a full list and :ref:`list of distributions` for details of each.

**Details.**

``dhistogram`` and ``chistogram`` create discrete
(point mass) and continuous (ogive) empirical distributions. ``chistogram``
is rarely used and ``dhistogram`` is easier to input using ``dsev``,
:ref:`nonparametric severity`.


.. _dec shift scale:

Shifting and Scaling Severity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A parametric severity clause can be transformed by scaling and location  factors,
following the ``scipy.stats`` ``scale`` and ``loc`` syntax.
Location is a shift or translation. The syntax is::

    sev SCALE * DISTNAME SHAPE + LOC
    sev SCALE * DISTNAME SHAPE - LOC


For zero parameter distributions ``SHAPE`` is omitted. Two parameter
distributions are entered ``sev SCALE * DISTNAME SHAPE1 SHAPE2 + LOC``.

**Examples.**

* ``sev lognorm 10 cv 3``: lognormal, mean 10, CV 0.

* ``sev 10 * lognorm 1.75``: lognormal, :math:`10X`, :math:`X \sim \mathrm{lognormal}(\mu=0,\sigma=1.75)`

* ``sev 10 * lognorm 1.75 + 20``: lognormal, :math:`10X + 20`

* ``sev 10 * lognorm 1 cv 3 + 50``: lognormal: :math:`10Y + 50`, :math:`Y\sim` lognormal mean 1, CV 3

* ``sev 100 * pareto 1.3 - 100``: Pareto, shape :math:`\alpha=3`, scale :math:`\lambda=100`.

* ``sev 100 * pareto 1.3``: Single parameter Pareto for :math:`x \ge 100`, Shape (:math:`\alpha`) 3, scale (:math:`\lambda`) 100

* ``sev 50 * norm + 100``: normal, mean (location) 100, standard deviation (scale) 50. No shape parameter.

* ``sev 5 * expon``: exponential, mean (scale) 5. No shape parameter.

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
    ``sev 10 * lognorm 1.5 -10``. The lexer binds uniary minus to the number, so the latter omits the operator. ``sev 10 * lognorm 1.5 + -10``, ``sev 10 * lognorm 1.5 +10`` and ``sev 10 * lognorm 1.5 + 10`` are all acceptable because there is no unary ``+``. This is a known bug and is insidious: the ``-10`` will be interpreted as a second shape parameter and ignored. You will not get the answer you expect.

.. _sev uncond sev:

Unconditional Severity
~~~~~~~~~~~~~~~~~~~~~~~

The severity clause is entered ground-up. It is converted to a distribution
conditional on a loss to the layer if there is a limits sub-clause. Thus, for
an excess layer :math:`y` xs :math:`a`, the severity used to create the aggregate has a
distribution :math:`X \mid X > a`, where :math:`X` is specified in the
``sev`` clause. For a ground-up (or missing) layer there is no adjustment.

The default behavior can be over-ridden by adding ``!`` after the
severity distribution.


**Example.**

The default behavior uses severity conditional to the layer. In this example, the conditional layer severity is 6.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    cond = build('agg DecL:Conditional '
                 '1 claim '
                 '12 xs 8 '
                 'sev 20 * uniform '
                 'fixed')
    qd(cond)

To specify unconditional severity, append ``!`` to the severity clause. The
unconditional layer severity is only 3.6 because there is just a 60% chance of
attaching the layer. In the last line, ``uncd.sevs[0].fz`` is ``sev 20 *
uniform`` ground-up.

.. ipython:: python
    :okwarning:

    uncd = build('agg DecL:Unconditional '
                 '1 claim '
                 '12 xs 8 '
                 'sev 20 * uniform ! '
                 'fixed')
    qd(uncd)
    print(uncd.sevs[0].fz.sf(8), uncd.agg_m / cond.agg_m)


.. _available sev dists:

``scipy.stats`` Continuous Random Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ``scipy.stats`` continuous random variable classes can be used as severity distributions, see :ref:`p sev dist roster` for a complete list. As always, with great power comes great responsibility.

.. warning::
    The user must determine if a severity distribution is appropriate, ``aggregate`` will not check!
    Only specified zero parameter (uniform, exponential, normal) and two parameter () distributions are allowed, but **all** one parameter
    distributions will work. However, any zero parameter distribution can be called with a dummy argument, that is ignored. **Be
    careful out there!**

