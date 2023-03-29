
.. _num ft convo:

Fourier Transform Convolution Algorithm
----------------------------------------------


  We come now to reality. The truth is that the digital computer has totally defeated the analog computer. The input is a sequence of numbers and not a continuous function. The output is another sequence of numbers, whether it comes from a digital filter or a finite element stress analysis or an image processor. **The question is whether the special ideas of Fourier analysis still have a part to play, and the answer is absolutely yes.**  :cite:p:`Strang1986am`

The previous section quoted Strang in support of discrete models. Here we complete his quote in support of using Fourier analysis, born in application to continuous functions, in a discrete setting.

.. _num algo details:

The ``aggregate`` Convolution Algorithm: Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section expands on each step in the **FFT algorithm** presented at the end of :ref:`num agg convo alog summary` in more detail. Recall, the four steps:

1. Discretize the severity cdf.
2. Apply the FFT to discrete severity.
3. Apply the frequency pgf to the FFT.
4. Apply the inverse FFT to create a discretized approximation to the aggregate distribution.

Algorithm Objective
"""""""""""""""""""""""
Compute a discrete approximation to the aggregate distribution

.. math::
    A = X_1 + \cdots + X_N,

under the assumption that :math:`X_i` are iid like :math:`X` and :math:`N` is independent of :math:`X_i`.

Algorithm Inputs
"""""""""""""""""""""""

#. Severity distribution (cdf, sf, and moments), optionally including loss pick and occurrence reinsurance adjustments.
#. Frequency distribution (probability generating function :math:`\mathscr P(z):= \mathsf E[z^N]`, and moments).
#. Number of buckets, expressed as log base 2, :math:`n=2^\mathit{log2}`.
#. Bucket size, :math:`b`.
#. Severity calculation method: ``round``, ``forwards``, or ``backwards``
#. Discretization calculation method: ``survial``, ``distribution``, or ``both``.
#. Normalization parameter, ``True`` or ``False``.
#. Padding parameter, an integer :math:`d \ge 0`.
#. Tilt parameter, a real number :math:`\theta \ge 0`.
#. Remove "fuzz" parameter, ``True`` or ``False``

Usually either tilting or padding is applied to manage aliasing, but not both. When both are requested, tilting is applied first and the result is zero padded.

Default and Reasonable Parameter Values
""""""""""""""""""""""""""""""""""""""""

The severity and frequency distributions are required. Defaults and reasonable ranges for the other parameters are as follows. Set :math:`x_{max}=bn` to be the range of the discretized output.

3. :math:`\mathit{log2}=16`, with a reasonable range :math:`3\le\mathit{log2}\le 28-d` (on a 64-bit computer with 32GB RAM).
4. Estimating the bucket size is quite involved and is described in :ref:`num rec bucket`.
5. Severity calculation ``round``, see :ref:`num rounding default`.
6. Discretization ``survival``, see :ref:`num exact calculation`.
7. Normalization defaults to `True`. It should only be used if it is immaterial, in the sense that :math:`1-F(x_{max})` is small. See :ref:`num normalization`.
8. Padding equals 1, meaning severity is doubled in size and padded with zeros. Padding equals to 2, which quadruples the size of the severity vector, is sometimes necessary. :cite:t:`Schaller2008` report requiring padding equal to 2 in empirical tests with very high frequency and thick tailed severity.
9. Tilt equals 0 (no tilting applied). :cite:t:`Embrechts2009a` recommend :math:`\theta n \le 20`. :cite:t:`Schaller2008` section 4.2 discuss how to select :math:`\theta` to the decrease in aliasing error and impact on numerical precision. Padding is an effective way to manage aliasing, but no more so than padding most circumstances. We prefer the simpler padding approach.
10. Remove fuzz is ``True``.


Algorithm Steps
"""""""""""""""""""""

The default steps are shown next, followed by further explanation.

#. If frequency is identically zero, then :math:`(1,0,\dots)` is returned with no further calculation.
#. If frequency is identically one, then the discretized severity is returned with no further calculation.
#. If needed, estimate the bucket size, :ref:`num rec bucket`.
#. Discretize severity into a vector :math:`\mathsf p=(p_0,p_1,\dots,p_{n-1})`, see :ref:`num discretization`.  This step may include normalization.
#. Tilt severity, :math:`p_k\gets p_k e^{-k\theta}`.
#. Zero pad the vector :math:`\mathsf p` to length :math:`2^{\mathit{log2} + d}` by appending zeros, to produce :math:`\mathsf x`.
#. Compute :math:`\mathsf z:=\mathsf{FFT}(\mathsf x)`.
#. Compute :math:`\mathsf f:=\mathscr P(\mathsf z)`.
#. Compute the inverse FFT,  :math:`\mathsf y:=\mathsf{IFFT}(\mathsf f)`.
#. Take the first :math:`n` entries in :math:`\mathsf y` to obtain :math:`\mathsf a:=\mathsf y[0:n]`.
#. Aggregate reinsurance is applied :math:`\mathsf a` if applicable.


.. _num ft theory:

Theory: Why the Algorithm Works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section explains why the output output :math:`\mathsf a=(a_0,\dots,a_{m-1})` has :math:`a_k` very close to :math:`\Pr(A=kb)`.

**Fourier transforms** provide an alternative way to represent a distribution function. The [Wikipedia](https://en.wikipedia.org/wiki/Fourier_transform) article says:

    The Fourier transform of a function is a complex-valued function representing the complex sinusoids that comprise the original function. For each frequency, the magnitude (absolute value) of the complex value represents the amplitude of a constituent complex sinusoid with that frequency, and the argument of the complex value represents that complex sinusoid's phase offset. If a frequency is not present, the transform has a value of 0 for that frequency. The Fourier inversion theorem provides a synthesis process that recreates the original function from its frequency domain representation.

    Functions that are localized in the time domain have Fourier transforms that are spread out across the frequency domain and vice versa, a phenomenon known as the uncertainty principle. The critical case for this principle is the Gaussian function: the Fourier transform of a Gaussian function is another Gaussian function.

    Generalizations include the discrete-time Fourier transform (DTFT, group $Z$), the discrete Fourier transform (DFT, group $Z\pmod N$) and the Fourier series or circular Fourier transform (group = $S^1$, the unit circle being a closed finite interval with endpoints identified). The latter is routinely employed to handle periodic functions. The fast Fourier transform (FFT) is an algorithm for computing the DFT.

The Fourier transform (FT) of a distribution function :math:`F` is usually written :math:`\hat F`. The FT contains the same information as the distribution and there is a dictionary back and forth between the two, using the inverse FT.
Some computations with distributions are easier to perform using their FT, which is what makes them useful.
The FT is like exponentiation for distributions. The exponential and log
functions turn (difficult) multiplication into (easy) addition

.. math:: e^a \times e^b = e^{a+b}.

FTs turn difficult convolution of distributions (addition of the corresponding random variables) into easy multiplication of Fourier transforms. If :math:`X_i` are random variables, :math:`X=X_1+X_2`, and :math:`F_X` is the distribution of :math:`X`, then

.. math:: \widehat{F_{X_1+X_2}}(t) = \widehat{F_{X_1}}(t) \times \widehat{F_{X_2}}(t),

where the righthand side is a product of functions. Computing the distribution of a sum of random variables is complicated because you have to consider all different ways an outcome can be split, but it is easy using FTs.
Of course, this depends on it being easy to compute the FT and its inverse---which is where FFTs come in.

There are three things going on here:

#. **Fourier transform** of a function,
#. **Discrete** Fourier transform of an infinite sequence, and
#. **Fast** Fourier transform of a finite vector.

**Discrete** Fourier transforms are a discrete approximation to continuous FTs, formed by sampling at evenly spaced points. The DFT is a sequence, rather than a function. It retains the convolution property of FTs. They are sometimes called discrete cosine transforms (DCT).

The **Fast** Fourier transform refers to a very fast way to compute *finite* discrete FTs, which are applied to finite samples of FTs.
General usage blurs the distinction between discrete FTs and their computation, and uses FFT as a catchall for both.


Thus, there are four-steps from the continuous to the finite discrete computational strategy (notation explained below):

1. Analytic domain:

   .. math:: f \rightarrow \hat f \rightarrow \mathscr P\circ \hat f \rightarrow \widehat{\mathscr P\circ \hat f} =: g

2. Discrete approximation:

   .. math:: f \rightarrow f_b \rightarrow \hat{f_b} \rightarrow \mathscr P\circ \hat{f_b} \rightarrow \widehat{\mathscr P\circ \hat{f_b}} =: g_b

3. Finite discrete approximation:

   .. math:: f \rightarrow f_{b, n} \rightarrow \hat{f_{b,n}} \rightarrow \mathscr P\circ \hat{f_{b,n}} \rightarrow \widehat{\mathscr P\circ \hat{f_{b,n}}} =: g_{b, n}

4. Finite discrete approximation, periodic inversion:

   .. math:: f \rightarrow f_{b, n} \rightarrow \hat{f_{b,n}} \rightarrow \mathscr P_m\circ \hat{f_{b,n}} \rightarrow \widehat{\mathscr P_m\circ \hat{f_{b,n}}} =: g_{b,n,m}

Here is the rationale for each step.

- Step 1 to 2: **discretize** :math:`f` because we are working in a digital
  world, not an analog/analyic one (Strang quote) and because the answers are
  often not continuous distributions. Discretize at multiples of a sampling
  interval :math:`b`. The sampling rate is :math:`1/b` samples per unit. The
  sampled distribution (which no longer has a density) is

  .. math::

    f_b = \sum_k p_k\delta_{kb}.

  :math:`f_b` has Fourier transform

  .. math::

      \hat{f_b}(t) =\sum_k p_ke^{-2\pi i kb t}.

  If :math:`\hat f` is know analytically it can be sampled directly, see the stable example below. However, many relevant :math:`f` do not have analytic FTs, e.g., lognormal.
  At this point, :math:`f_b` is still defined of :math:`\mathbb R`.

- Step 2 to 3: **truncate** and take a **finite** discretization because we
  are working on a digital computer with finite memory.

  .. math::

    f_{b, n} = \sum_{k=0}^{n-1} p_k\delta_{kb}.

  Let :math:`P=nb`. Now :math:`f_{b,n}` is non-zero only on :math:`[0, P)`.
  Finite discretization combined with an assumption of :math:`P` periodicity enables the use of **FFTs** to compute
  :math:`f_{b, n} \rightarrow \hat{f_{b,n}}`.
  (In order for a Fourier series to be :math:`P`-periodic, it
  can only weight frequencies that are a multiple of :math:`1/P` since
  :math:`\exp(2\pi i (x+kP)t)=\exp(2\pi i xt)` for all integers
  :math:`k` iff :math:`\exp(2\pi i kPt)=1` iff :math:`Pt` is an
  integer. Take the integer to be 1; higher values correspond to
  aliasing. Hence Shannon-Nyquist and bandwidth limited functions etc.)
  Sampling :math:`\widehat{f_{b, n}}(t)` at :math:`t=0,1/P,\dots,(n-1)/P`, requires calculating

  .. math::
     \hat{f_{b,n}}(\tfrac{l}{P}) = \sum_k p_ke^{-2\pi i kb \tfrac{l}{P}} = \sum_k p_ke^{-\tfrac{2\pi i}{n} kl}

  **which is exactly what FFTs compute very quickly**.

- Step 3 to 4: **finite convolution**, :math:`\mathscr P_m` is computed with a
  sample of length :math:`m\ge n`, i.e., padding, to control aliasing.  We
  can also use exponential tilting (which must be done in the :math:`f`
  domain). :math:`\mathscr P_m\circ \hat{f_{b,n}}` is the application of a function to a vector, element-by-element and is easy to compute.
  :math:`\mathscr P_m\circ \hat{f_{b,n}} \rightarrow \widehat{\mathscr P_m\circ \hat{f_{b,n}}}` can be computed using FFTs, whereas inverting
  :math:`\mathscr P\circ \hat{f_{b,n}}` would usually be very
  difficult because it usually has infinite support.
  The price for using FFTs is assuming :math:`g` is :math:`P`-periodic,
  i.e., introducing aliasing error. For simplicity, assume :math:`m=n` by padding the samples in Step 2.

  Now we can use the inverse DFT to recover :math:`g` at the values
  :math:`kb`:

  .. math::


     \begin{align}
     g(kb)
     &= \sum_l \hat g(\tfrac{l}{P}) e^{2 \pi i kb \tfrac{l}{P}} \\
     &= \sum_l \hat gf(\tfrac{l}{P}) e^{\tfrac{2 \pi i}{n} kl}.
     \end{align}

  However, this is an infinite sum (step 3), and we are working with computers, so it needs to be truncated (step 4). What is

  .. math::


     \sum_{l=0}^{n-1} \hat g(\tfrac{l}{P}) e^{\tfrac{2 \pi i}{n} kl}?

  It is an inverse DFT, that FFTs compute with alacrity. What does it
  equal?

  Define :math:`g_P(x) = \sum_k g(x + kP)` to be the :math:`P`-periodic
  version of :math:`g`. If :math:`g` has finite support contained in
  :math:`[0, P)` then :math:`g_P=g`. If that is not the case there will be
  wrapping spill-over, see PICTURE.

  Now

  .. math::

     \begin{align}
     \hat g(\tfrac{l}{P})
     :&= \int_\mathbb{R} g(x)e^{-2 \pi i x \tfrac{l}{P}}dx \\
     &= \sum_k \int_{kP}^{(k+1)P} g(x)e^{-2 \pi i x \tfrac{l}{P}}dx  \\
     &= \sum_k \int_{0}^{P} g(x+kP)e^{-2 \pi i (x+kP) \tfrac{l}{P}}dx  \\
     &= \int_{0}^{P} \sum_k g(x+kP)e^{-2 \pi i x \tfrac{l}{P}}dx  \\
     &= \int_{0}^{P} g_P(x)e^{-2 \pi i x \tfrac{l}{P}}dx  \\
     &= \hat{g_P}(\tfrac{l}{P})
     \end{align}

  and therefore, arguing backwards and assuming that :math:`\hat g` is quickly
  decreasing, for large enough :math:`n`,

  .. math::


     \begin{align}
     \sum_{l=0}^{n-1} \hat{g}(\tfrac{l}{P}) e^{\tfrac{2 \pi i}{n} kl}
     &\approx\sum_l \hat{g}(\tfrac{l}{P}) e^{\tfrac{2 \pi i}{n} kl} \\
     &= \sum_l  \hat{g_P}(\tfrac{l}{P}) e^{\tfrac{2 \pi i}{n} kl}  \\
     &= g_P(kb)
     \end{align}

  Thus the partial sum we can easily compute on the left approximates :math:`g_P` and in favorable circumstances it is close to :math:`g`.


There are four sources of error in the FFT algorithm. They can be controlled by different parameters:

#. Discretization error :math:`f \leftrightarrow f_b` (really
   :math:`\hat f \leftrightarrow \hat{f_b}`): replacing the original distribution with a discretized approximation, controlled by decreasing the bucket size.
#. Truncation error :math:`\hat{f_b} \leftrightarrow \hat{f_{b, n}}`: shrinking the support of the severity distribution by right truncation, controlled by increasing the bucket size and/or increasing the number of buckets.
#. Aliasing error :math:`\widehat{\mathscr P\circ \hat{f_{b,n}}} \leftrightarrow \widehat{\mathscr P_m\circ \hat{f_{b,n}}}`:
   expect :math:`g_k` get :math:`\sum_l g_{k+ln}`: working with only finitely many frequencies in the Fourier domain which results in visible the aggregate wrapping, controlled by padding or tilting severity.
#. FFT algorithm: floating point issues, underflow and (rarely) overflow, hidden by removing numerical "fuzz" after the algorithm has run.

To summarize:

-  If we know :math:`\hat f` analyically, we can use this method to
   estimate a discrete approximation to :math:`f`. We are estimating
   :math:`f_P(kb)` not :math:`f(kb)`, so there is always aliasing error,
   unless :math:`f` actually has finite support.

-  If :math:`f` is actually discrete, the only error comes from
   truncating the Fourier series. We can make this as small as we please
   by taking enough terms in the series. This case is illustrated for
   the Poisson distribution. This method is also applied by
   ``aggregate``: the “analytic” chf is :math:`\mathscr P_N(M_X(t))`,
   where :math:`M_X(t)` is the sum of exponentials given above.

-  When :math:`\hat f` is known we have a choice between discretizing in
   the space (loss) or time domain.

-  If :math:`f` is not discrete, there is a discretization and
   potentially aliasing error. We can control the former with high
   frequency (small :math:`b`) sampling. We control the latter with
   large :math:`P=nb`, arguing for large :math:`n` or large :math:`b`
   (in conflict to managing discretization error).

Using FFT to Invert Characteristic Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The use of FFTs to recover the aggregate at the end of Step 4 is entirely generic. It can be used to invert any characteristic function. In this section we provide some of examples.

Invert a gamma distribution from a sample of its characteristic function and compare with the true density. These plots show the inversion is extremely accurate over a very wide range. The top right plot compares the log density, highlighting differences only in the extreme tails.

.. ipython:: python
    :okwarning:

    from aggregate.extensions import ft_invert
    import scipy.stats as ss
    import matplotlib.pyplot  as plt
    @savefig numfft01.png scale=20
    df = ft_invert(
             log2=6,
             chf=lambda alpha, t: (1 - 1j * t) ** -alpha,
             frz_generator=ss.gamma,
             params=[30],
             loc=0,
             scale=1,
             xmax=0,
             xshift=0,
             suptitle='Gamma distribution')


Invert a Poisson distribution with a very high mean. This is an interesting case, because we do not need space for the whole answer, just the effective range of the answer. We can use periodicity to "move" the answer to the right :math:`x` range.
This example reproduces a Poisson with mean 10,000. The standard deviation is only 100 and so the effective rate of the distribution (using the normal approximation) will be about 9500 to 10500. Thus a satisfactory approximation can be obtained with only :math:`2^{10}=1024` buckets.

.. ipython:: python
    :okwarning:

    import aggregate.extensions.ft as ft
    @savefig numfft02.png scale=20
    df = ft.ft_invert(
             log2=10,
             chf=lambda en, t: np.exp(en * (np.exp(1j * t) - 1)),
             frz_generator=ss.poisson,
             params=[10000],
             loc=0,
             scale=None, # for freq dists, scaling does not apply
             xmax=None,  # for freq dists want bs = 1, so xmax=1<<log2
             xshift=9500,
             suptitle='Poisson distribution, large mean computed in small space.')

Invert a stable distribution. Here there is more aliasing error because the distribution is so thick tailed. There is also more on the left than right because of the asymmetric beta parameter.

.. ipython:: python
    :okwarning:

    def levy_chf(alpha, beta, t):
        Φ = np.tan(np.pi * alpha / 2) if alpha != 1 else -2 / np.pi * np.log(np.abs(t))
        return np.exp(- np.abs(t) ** alpha * (1 - 1j * beta * np.sign(t) * Φ))

    df = ft_invert(
             log2=12,
             chf=levy_chf,
             frz_generator=ss.levy_stable,
             params=[1.75, 0.3],  # alpha, beta
             loc=0,
             scale=1.,
             xmax=1<<8,
             xshift=-(1<<7),
             suptitle='Stable Levy exponent $\\alpha=7/4$, '
             'slightly skewed')
    f = plt.gcf()
    ax = f.axes[1]
    @savefig numfft03.png scale=20
    ax.grid(lw=.25, c='w');


.. _num fft:

Fast Fourier Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trick with FFTs is *how* they are computed. *What* they compute is very straightforward and given by a simple matrix multiplication.

The FFT of the :math:`m\times 1` vector
:math:`\mathsf{x}=(x_0,\dots,x_{m-1})` is just another :math:`m\times 1` vector :math:`\hat{\mathsf{x}}` whose :math:`j`\ th component is

.. math:: x_j = \sum_{k=0}^{m-1} x_k\exp(-2\pi ijk/m),

where :math:`i=\sqrt{-1}`. The coefficients of :math:`\hat{\mathsf{x}}` are complex numbers. It is easy to see that :math:`\hat{\mathsf{x}}=\mathsf{F}\mathsf{x}` where

.. math::

   \mathsf{F}=
   \begin{pmatrix}
   1 & 1 & \dots & 1 \\
   1 & w & \dots & w^{m-1} \\
   1 & w^2 & \dots & w^{2(m-1)} \\
   \vdots & & & \vdots \\
   1 & w^{m-1} & \dots & w^{(m-1)^2}
   \end{pmatrix}

is a matrix of complex roots of unity and  :math:`w=\exp(-2\pi i/m)`. This shows there is nothing
inherently mysterious about an FFT. The trick is that there is a very
efficient algorithm for computing the matrix multiplication :cite:p:`Press1992a`.  Rather than taking
time proportional to :math:`m^2`, as one would expect, it can be
computed in time proportional to :math:`m\log(m)`. For large values of :math:`m`, the difference
between :math:`m\log(m)` and :math:`m^2` time is the difference between
practically possible and practically impossible.

The inverse FFT to recovers :math:`\mathsf{x}` from its transform
:math:`\hat{\mathsf{x}}`. The inverse FFT is computed using the same equation as the FFT with :math:`\mathsf F^{-1}` (matrix inverse) in place of :math:`\mathsf F`. It is easy to see that inverse equals

.. math::

   \mathsf{F}^{-1}=
   \frac{1}{m}
   \begin{pmatrix}
   1 & 1 & \dots & 1 \\
   1 & w^{-1} & \dots & w^{-(m-1)} \\
   1 & w^2 & \dots & w^{2(m-1)} \\
   \vdots & & & \vdots \\
   1 & w^{-(m-1)} & \dots & w^{-(m-1)^2}
   \end{pmatrix}.

The :math:`(j,j)` element of :math:`m\mathsf{F}\mathsf{F}^{-1}` is

.. math::
    \sum_g w^{jg}w^{-jg}= \sum_g 1 = m

and the :math:`(j,k),\ j\not=k` element is

.. math::
    \sum_g w^{jg} w^{-gk} = \sum_g w^{g(j-k)} = 0.

The inversion process can also be computed in :math:`m\log(m)` time
because the matrix equation is the same.

How does the FFT compute convolutions? Given two probability vectors for outcomes :math:`k=0,1,\dots,n-1`, say  :math:`\mathsf p=(p_0,\dots, p_{n-1})` and
:math:`\mathsf q=(q_0,\dots, q_{n-1})`, the product of the :math:`k` th elements of the FFTs equals

.. math::

    \left(\sum_g p_g w^{gk} \right)
    \left(\sum_h p_h w^{hk} \right)
    = \sum_{m=0}^{n-1}
    \left( \sum_{\substack{g, h\\ g+h\equiv m(n)}} p_g q_h \right) w^{km}

is the :math:`k` th element of the FFT of the wrapped convolution of :math:`\mathsf p` and :math:`\mathsf q`. For example, if :math:`n=4` and :math:`m=0`, the inner sum on the right equals

.. math::

    p_0 q_0 + p_1 q_3 + p_2 q_2 + p_3 q_1

which can be interpreted as

.. math::

    p_0 q_0 + p_1 q_{-1} + p_2 q_{-2} + p_3 q_{-3}

in arithmetic module :math:`n`.

In the convolution algorithm, the product of functions :math:`\widehat{F_{X_1}} \times \widehat{F_{X_2}}` is replaced by the component-by-component product of two vectors, which is easy to compute. Thus, to convolve two discrete distributions, represented as :math:`\mathsf p=(p_0,\dots,p_{m-1})` and :math:`\mathsf q=(q_0,\dots,q_{m-1})` simply

* Take the FFT of each vector, :math:`\hat{\mathsf p}=\mathsf F\mathsf p` and :math:`\hat{\mathsf q}=\mathsf F\mathsf q`
* Compute the component-by-component product :math:`\mathsf z = \hat{\mathsf p}\hat{\mathsf q}`
* Compute the inverse FFT :math:`\mathsf F^{-1}\mathsf z`.

The answer is the exact convolution of the two input distributions, except that sum values wrap around: the extreme right tail re-appears as probabilities around 0. This problem is called aliasing (the same as the wagon-wheel effect in old Westerns), but it can be addressed by padding the input vectors.

Here is a simple example of wrapping, using a compound Poisson distribution with an expected claim count of 10 and severity taking the values 0, 1, 2, 3, or 4 equally often. The aggregate has a mean of 20 and is computed using only 32 buckets. This is not enough space, and the right hand part of the distribution wraps around. The components are shown in the middle and how they combine on the right.

.. ipython:: python
    :okwarning:

    @savefig numfft04.png
    ft.fft_wrapping_illustration(ez=10, en=2)

The next figure illustrates more extreme FFT wrapping. It shows an attempt to model a compound Poisson distribution with a mean of 80 using only 32 buckets. The result is the straight line on the left. The middle plot shows the true distribution and the vertical slices of width 32 that are combined to get the total. These are shown shifted on the left. The right plot zooms into the rate ``0:32``, and shows how the wrapped components sum to the result on the left. This is a good example of how FFT methods can fail and can appear to give inexplicable results.

.. ipython:: python
    :okwarning:

    @savefig numfft05.png
    ft.fft_wrapping_illustration(ez=10, en=8)


It is not necessary to understand the details of FTs to use ``aggregate`` although they are fascinating, see for example :cite:t:`Korner2022`. In probability, the moment generating functions and characteristic function are based on FTs. They are discussed in any serious probability text.


.. _num fft routines:

FFT Routines
~~~~~~~~~~~~~~~

Computer systems offer a range of FFT routines. ``aggregate`` uses two functions from ``scipy.fft`` called :meth:`scipy.fft.rfft` and :meth:`scipy.fft.irfft`. There are similar functions in ``numpy.fft``. They are tailored to taking FFTs of vectors of real numbers (as opposed to complex numbers). The FFT routine automatically handles padding the input vector. The inverse transform returns real numbers only, so there is no need to take the real part to remove noise-level imaginary parts. It is astonishing that the whole ``aggregate`` library pivots on a single line of code::

    agg_density = rifft(\mathscr P(rfft(p)))

Obviously, a lot of work is done to marshal the input, but this line is where the magic occurs.

The FFT routines are accurate up to machine noise, of order  ``1e-16``. The noise can be positive or negative---the latter highly undesirable in probabilities. It appears random and does not accumulate undesirably in practical applications. It is best to strip out the noise, setting to zero all values with absolute value less than machine epsilon (``numpy.finfo(float).esp``). The ``remove_fuzz`` option controls this behavior. It is set ``True`` by default. CHECK SURE?

