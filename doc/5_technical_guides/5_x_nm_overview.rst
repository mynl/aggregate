
.. _num overview:

Overview
----------

A Trilemma
~~~~~~~~~~~~~~~~~~

Numerical analysts face a trilemma: they can pick two of  fast, accurate, or
flexible. Simulation is flexible, but trades speed against accuracy. Fewer
simulations is faster but less accurate; many simulations improves accuracy
at the cost of speed. ``aggregate`` delivers the third trilemma option using
a fast Fourier transform (FFT) convolution algorithm to deliver speed and
accuracy but with less flexibility than simulation.

.. _num agg convo alog summary:

The ``aggregate`` Convolution Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  **Complaint**  It is the most natural thing in the world to decompose an oscillating electrical signal into sines and cosines of various frequencies via the Fourier transform - but probabilities, no. Basically, these are positive numbers adding up to 1, and what have sines and cosines to do with that? Indeed, in many applications done first by Fourier, a simpler, more understandable proof may emerge upon taking Fourier away. Still, the Fourier transform is a very effective, sometimes indispensable technical tool in much of our business here.
  :cite:p:`McKean2014bk`

This section describes the core convolution algorithm implemented in
``aggregate``. It is an application where the caveat in McKean's complaint
holds true. The use of Fourier transform methods is unnatural but very
effective. As usual, let :math:`A=X_1 + \cdots + X_N` where
severity :math:`X_i` are iid and independent of the claim count :math:`N`. We want to explicitly compute the distribution

.. math::

   \Pr(A < a) &= \sum_n \Pr(A < a \mid N=n)\Pr(N=n) \\
   &= \sum_n \Pr(X^{*n} < a)\Pr(N=n).

Here, :math:`X^{*n}` denotes the sum of :math:`n` independent copies of :math:`X`. This problem is usually hard to solve analytically.
For example, if :math:`X` is lognormal then there is no closed form
expression for the distribution of :math:`X^{*2} \sim X_1 + X_2`. However,
things are more promising in the Fourier domain. The characteristic function
of :math:`A` can be written, using independence, as

.. math::

   \phi_A(t) :&= \mathsf E[e^{itA}] \\
   &= \mathsf E[\mathsf E[ e^{itA} \mid N]] \\
   &= \mathsf E[\mathsf E[ e^{itX}]^N] \\
   &= \mathscr P_N(\phi_X(t))

where :math:`\mathscr P_N(z) = \mathsf E[z^N]` is the probability generating
function. The pgfs of most common frequency distributions are know. For
example, :math:`N` is Poisson with mean :math:`\lambda` then it is easy to
see that :math:`\mathscr P_N(z) = \exp(\lambda(z-1))`.

.. note::

    The algorithm assumes the pgf can be written an explicit function. That, in turn, implies that frequency is thin tailed (it is not subexponential). This is generally not a problem because of how insurance events are defined. Losses from a catastrophe event, which can produce a large number of claims, are combined into a single occurrence. As a result, we usually model a small and thin tailed number of events with a thick tailed severity distribution. For example, there are approximately 1.75 US landfalling hurricanes per year, and the distribution of events since 1850 is well-fit by a (thin-tailed) Poisson distribution.

Knowing the characteristic function is useful because it can be inverted to
determine the density of :math:`A`. Subject to certain terms and conditions,
described below, these arguments can be carried out in a finite discrete
setting which is helpful for two reasons. First, it makes the problem
tractable for a digital computer, and second, many :math:`A` that arise in
insurance problems that are discrete or mixed, because policy limits
introduce mass points. For these reasons, we model a discrete approximation
to :math:`A`, see also :ref:`num how agg reps a dist`.

In a discrete approximation, the **function** :math:`\phi_X(t)` is replaced by
a **vector sample** computed either by taking the discrete Fourier transform
(DFT) of a discretized approximation to the distribution of :math:`X` or by
directly sampling :math:`\phi_X`. We usually do the former, computing the DFT
using the Fast Fourier transform (FFT) algorithm. However, certain :math:`X`,
such as  stable distributions, have known characteristic functions but no
closed-form distribution or density function. In that case, a sample of the
characteristic function can be used directly. The pgf is then applied
element-by-element to the characteristic function sample vector, and the
inverse FFT routine used to obtain a discrete approximation to the aggregate
distribution. The exact rationale for this process are discussed in :ref:`num
ft convo`. The errors it introduces are discussed there and in :ref:`num
error analysis`.

In summary, the **FFT algorithm** is simply:

1. Discretize the severity cdf.
2. Apply the FFT to discrete severity.
3. Apply the frequency pgf to the FFT.
4. Apply the inverse FFT to create a discretized approximation to the aggregate distribution.

This algorithm has appeared numerous times in the literature, see :ref:`num rel act lit`. The details are laid out in :ref:`num algo details`. A plain Python implementation is presented in :ref:`10 min algo deets`.

.. _num swot:

Strengths and Weaknesses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

I've been using the FFT algorithm since Glenn Meyers explained it to me at a
COTOR meeting around 1996, and I still find it miraculous.  It is very fast
and its speed is largely independent of the expected claim count---in
contrast to simulations. The algorithm is also very accurate, both in
absolute and relative terms. It is essentially exact in many cases and
eight-plus digit precision is often easy to obtain. The algorithm works well
in almost all situations and for many use-cases it is unbeatable, including
computing:

* The distribution of aggregate losses from a portfolio with a complex limit and attachment profile, and a mixed severity.
* Ceded or net outcome distributions for an occurrence reinsurance program.
* Ceded or net outcome distributions for reinsurance contracts with variable features such as sliding commissions, swing rated programs, profit commissions, aggregate limits, see :doc:`../2_user_guides/2_x_re_pricing`.
* The distribution of retained losses net of specific and aggregate insurance, as found in a risk-retention group, see :doc:`../2_user_guides/2_x_ir_pricing`, including exact computation of Table L and Table M charges in US worker compensation ratemaking.
* The distribution of the sum of independent distributions, e.g., aggregating units such as line of business, business unit, geographic unit etc.
* The distribution of the sum of dependent units, where the dependence structure is driven by :ref:`common frequency mixing variables <5_x_probability>`.

The algorithm is particularly well-suited to compute aggregates with low claim counts and a  thick-tailed severity and where accuracy is important, such as catastrophe risk PML, AEP, and OEP points. Outcomes with low expected loss rates are hard to simulate accurately.

The FFT algorithm is not a panacea.
On the downside, its mysterious Fourier-nature presents the user with a choice of trusting in magic or a steep learning curve to understand the theory.
It relies on hard-to-select parameters and can fail spectacularly and without warning if they are not chosen judiciously. A big contribution of ``aggregate`` is to provide the user with sensible default parameters and a test of model validity, see :ref:`num error analysis`.
It does not work well for a high mean, thick-tailed frequency combined with a thick-tailed severity distribution that has an intricate distribution---an unusual situation that stresses any numerical method. However, when either frequency or severity is thin-tailed, it excels.
Finally, the ``aggergate`` implementation is limited to tracking one variable at a time. It cannot model joint distributions, such as ceded and net loss or derived quantities such as the total cession to a specific and aggregate cover, or the cession to an occurrence program with limited reinstatements. Both of these require a bivariate distribution. It *can* model the net position after specific and aggregate cessions, and ceded losses to an occurrence program with an aggregate limit. See :ref:`num extensions` for an approach to bivariate extensions.

.. _num rel act lit:

Related Actuarial Literature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The earliest reference to Fourier transforms in actuarial science I have found
is :cite:t:`Heckman1983`. They used continuous Fourier transforms to compute
aggregate distributions by numerically integrating the characteristic
function. Their analysis includes severity and frequency uncertainty, that they call contagion and mixing.

Explicit use of the FFT appears first in :cite:`Bertram1983`. It has subsequently appeared in numerous places.

:cite:t:`Buhlmann1984` compares the FFT algorithm with Panjer recursion for compound Poisson distributions. It concludes that usually FFTs can be computed in fewer operations. :cite:t:`Hurlimann1986` obtains an error bound for stop-loss premium computed with FFTs.

:cite:t:`Robertson1992` considers a quasi-FFT algorithm, using discrete-continuous adjustments to reflect a piecewise linear as opposed to a fully discrete, distribution function. These greatly complicate the analysis for little tangible benefit. We recommend using a fully discrete distribution as explained in :ref:`num how agg reps a dist`.

:cite:t:`Embrechts1993` describes the  FFT algorithm and considers Richardson extrapolation to estimate the density.

:cite:t:`WangS1998` describes the FFT algorithm, using padding to control aliasing (wrapping) error. The first edition of
:cite:t:`LM`, published in 1998, describes the algorithm, although it no longer appears in the fifth edition.
:cite:t:`Grubel1999` describes the use of exponential tilting to reduce aliasing error and :cite:t:`Grubel2000` explains how to use Richardson extrapolation to improve density estimates. Exponential tilting is the same process used in GLM exponential families to adjust the mean, and it is also used in large deviation theory.
:cite:t:`Mildenhall2005a` describes the FFT algorithm.

Approximate inversion of the Fourier transform is also possible using FFTs.
:cite:t:`Menn2006` uses of FFTs to determine densities for distributions which
have analytic MGFs but not densities, notably the class of stable
distributions. This method is shown in :ref:`num ft convo`.

:cite:t:`Kaas2008` section 3.6 presents the FFT algorithm in R.

:cite:t:`Embrechts2009a` revisits Panjer recursion compared to the FFT algorithm. It also explores exponential tilting for aliasing error. It comments "Compared to the Panjer recursion, FFT has two main advantages: It works with arbitrary frequency distributions and it is much more efficient." It concludes:

  The Panjer recursion is arguably the most widely used method to “exactly” evaluate compound distributions. However, FFT is a viable alternative: It can be applied with arbitrary frequencies and offers a tremendous timing advantage for a large number of lattice points; moreover, the use of exponential tilting—which practically rules out any aliasing effects—facilitates applications (such as evaluation of the lower tail) that were thought to be an exclusive task for recursive procedures.

More recently, :cite:t:`Papush2021`, extending :cite:t:`Papush2001`, considers the best two parameter approximation to an frequency severity convolution. It shows that the gamma provides the best fit across a wide range of synthetic examples. However, all of their examples have a bounded (hence thin tailed) severity. A simple model::

    agg 10 claims sev lognorm 2 poisson

is not best fit by a gamma.

:cite:t:`Homer2003` and :cite:t:`Mildenhall2005a` describe the use of two-dimensional FFTs to model aggregates with bivariate frequency (for two different lines) and bivariate severity (net and ceded).

.. _num other applications:

Other Applications
~~~~~~~~~~~~~~~~~~~~

The FFT algorithm is applied to model operational risk in :cite:t:`Schaller2008`, :cite:t:`Temnov2008`, :cite:t:`Luo2009`, :cite:t:`Luo2010`, and :cite:t:`Shevchenko2010`. These applications mirror the actuarial approach, using either padding or exponential tilting (exponential window) to control aliasing error. They are interesting because they include modeling with a very high expected claim counts and a very thick tailed severity (no mean). See :ref:`num truncation example`.

In finance, FFTs are used in option pricing,  :cite:t:`Carr1999`. These applications can use distributions derived from stable-:math:`\alpha` and Levy process families that have a closed for characteristic function but no analytic density. :cite:t:`Duan2012` describe more recent innovations. FFTs are also used as a general purpose convolution routine, :cite:t:`Cerny2004`

:cite:t:`Wilson2016` describes an interesting approach to accurate pairwise convolution that splits each component to limit the ratio of its most and least (non-zero) likely outcome. It provides helpful estimates for assessing relative error and determining when an FFT estimate can be trusted.


.. _num kappa:

Conditional Expectations (Kappa)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function :math:`\kappa_i(x):=\mathsf E[X_i \mid X=x]` is the basis for many of the computations in :class:`Portfolio`. It can be computed using Fourier transforms because it is a convolution. There is no loss in generality assuming :math:`X=X_1 + X_2`. For simplicity suppose :math:`(X_1, X_2)` have a bivariate density :math:`f`. Then

.. math::

   \mathsf E[X_1 \mid X=x]
   &= \int_0^x t\frac{f(t, x-t)}{f(x)}\, dt \\
   &= \frac{1}{f(x)} \int_0^x tf_1(t) f_2(x-t)\, dt

can be computed from the convolution of :math:`tf_1(t)` and :math:`f_2`. The convolution can be computed using Fourier transforms in the usual way: transform, product, inverse transform. Using FFTs and relying on the discretized version of :math:`X_i`, the algorithm becomes:

#. Compute the discrete approximation to :math:`X_{\hat i}`, the sum of all :math:`X_j`, :math:`j\not=i`, identifying distributions with their discrete approximation vectors.
#. Compute FFTs of :math:`X_{i}` and :math:`X_{\hat i}`, with optional padding.
#. Take the elementwise product of the FFTs.
#. Apply the inverse FFT and unpad if necessary.

A variable with density :math:`xf(x) / \mathsf E[X]` is called the size-bias of :math:`X`. Size-biased variables have lots of interesting applications, see :cite:t:`Arratia2019`.

The ``aggregate`` implementation computes :math:`X_{\hat i}` by dividing out the distribution of :math:`X_{i}` from the overall sum (deconvolution), where that is possible, saving computing time.
