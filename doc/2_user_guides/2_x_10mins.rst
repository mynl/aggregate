.. _2_x_10mins:

.. reviewed 2022-12-26

A Ten Minute Guide to ``aggregate``
=====================================

**Objectives:** A whirlwind introduction---don't expect to understand everything the first time, but you will see what you can achieve with the package. Follows the `pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`_ model, a *long* 10 minutes.

**Audience:** A new user.

**Prerequisites:** Python programming; aggregate distributions.  Read in conjunction with :doc:`2_x_student` or :doc:`2_x_actuary_student` practice guides.

**See also:** :doc:`../3_Reference`, :doc:`2_x_dec_language`.


**Contents:**

#. :ref:`10 min princ cl`
#. :ref:`10 min Underwriter`

    - :ref:`10 min create from decl`
    - :ref:`10 min formatting`
    - :ref:`10 min ob cr`
    - :ref:`10 min bts`

#. :ref:`10 min how`
#. :ref:`10 min Severity`
#. :ref:`10 min Aggregate`

    - :ref:`10 min creating agg`
    - :ref:`10 min quick diagnostics`
    - :ref:`10 min algo deets`
    - :ref:`10 min basic prob`
    - :ref:`10 min mixtures`
    - :ref:`10 min accessing`
    - :ref:`10 min reinsurance`

#. :ref:`10 min Distortion`
#. :ref:`10 min Portfolio`
#. :ref:`10 min est bs`

    - :ref:`10 min hyper`
    - :ref:`10 min agg bucket`
    - :ref:`10 min port bucket`

#. :ref:`10 min common`

    - :ref:`10 min info`
    - :ref:`10 min describe`
    - :ref:`10 min density_df`
    - :ref:`10 min stats`
    - :ref:`10 min report`
    - :ref:`10 min spec`
    - :ref:`10 min decl program`
    - :ref:`10 min update`
    - :ref:`10 min stats funs`
    - :ref:`10 min plot`
    - :ref:`10 min price`
    - :ref:`10 min snap`
    - :ref:`10 min approx`

#. :ref:`10 min additional`

    - :ref:`10 min conditional expected values`
    - :ref:`10 min calibrate distortions`
    - :ref:`10 min analyze distortions`
    - :ref:`10 min twelve plot`

#. :ref:`10 min extensions`
#. :ref:`10 min summary`

.. _10 min princ cl:

Principal Classes
------------------

The ``aggregate`` package makes working with aggregate probability distributions as straightforward as working with parametric distributions even though their densities rarely have closed-form expressions. It is built around five principal classes.

#. The :class:`Underwriter` class keeps track of everything in its ``knowledge`` dataframe, interprets Dec Language (DecL, pronounced like deckle, /ˈdɛk(ə)l/) programs, and acts as a helper.
#. The :class:`Severity` class models a size of loss distribution (a severity curve).
#. The :class:`Aggregate` class models a single unit of business, such as a line, business unit, geography, or operating division.
#. The :class:`Distortion` class provides a distortion function, the basis of a spectral risk measure.
#. The :class:`Portfolio` class models multiple units. It extends the functionality in :class:`Aggregate`, adding pricing, calibration, and allocation capabilities.

There is also a :class:`Frequency` class that :class:`Aggregate` derives from, but it is rarely used standalone, and a :class:`Bounds` class for advanced users.

.. _10 min Underwriter:

The :class:`Underwriter` Class
-------------------------------

The :class:`Underwriter` class is an interface into the computational functionality of ``aggregate``. It does two things:

#. Creates objects using the DecL language, and

#. Maintains a library of DecL object specifications called the knowledge. New objects are automatically added to the knowledge.

To get started, import ``build``, a pre-configured :class:`Underwriter` and :func:`qd`, a quick-display function. Import the usual suspects too, for good measure.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    import pandas as pd, numpy as np, matplotlib.pyplot as plt

Printing ``build`` reports its name, the number of objects in its knowledge, and other information about hyper-parameter default values. ``site_dir`` is where various outputs will be stored. ``default_dir`` is for internal package data. The ``build`` object loads an extensive test suite of DecL programs with over 130 entries.

.. ipython:: python
    :okwarning:

    build

.. _10 min create from decl:

Object Creation Using DecL and :meth:`build`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Underwriter class interprets DecL programs (:doc:`2_x_dec_language`). These allow severities, aggregates and portfolios to be created using standard insurance language.

For example, to build an :class:`Aggregate` using DecL and report key statistics for frequency, severity, and aggregate, needs just two commands.

.. ipython:: python
    :okwarning:

    a01 = build('agg TenM:01 100 claims 100 xs 0 sev lognorm 10 cv 1.25 poisson')
    qd(a01)


DecL is supposed to be human-readable, so I hope you can guess the meaning of the DecL code (``TenM:01`` is just a label)::

    agg TenM:01 5 claims 1000 xs 0 sev lognorm 50 cv 4 poisson

The units are 1000s of USD, EUR, or GBP.

DecL is a custom language, created to describe aggregate distributions. Alternatives are to use positional arguments or key word arguments in function calls. The former are confusing because there are so many. The latter are verbose, because of the need to specify the parameter name. DecL is a concise, expressive, flexible, and powerful alternative.

.. _10 min formatting:

Important: Formatting a DecL Program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    **All DecL programs are one line long.**

It is best to break a DecL program up to make it more readable. The fact that Python automatically concatenates strings between parenthesis makes this easy. The program above is always entered in the help as::

    a01 = build('agg TenM:01 '
                '100 claims '
                '100 xs 0 '
                'sev lognorm 10 cv 1.25 '
                'poisson')

which Python makes equivalent to::

    a01 = build('agg TenM:01 100 claims 100 xs 0 sev lognorm 10 cv 1.25 poisson')

as originally entered. **Pay attention to spaces at the end of each line!** Entering::

    a01 = build('agg TenM:01'
                '100 claims'
                '100 xs 0'
                'sev lognorm 10 cv 1.25'
                'poisson')

produces::

    a01 = build('agg TenM:01100 claims100 xs 0sev lognorm 10 cv 1.25poisson')

which results in syntax errors.

DecL includes a Python newline ``\``. All programs in the help are entered so they can be cut and pasted.

.. _10 min ob cr:

Object Creation from the Knowledge Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **knowledge** dataframe is a database of DecL programs and a parsed
dictionaries to create objects. ``build`` loads an extensive library by
default. Users can create and load their own databases, allowing them to share common parameters for

- severity (size of loss) curves,
- aggregate distributions (e.g., industry losses in major classes of business, or total catastrophe losses from major perils), and
- portfolios (e.g., an insurer's reference portfolio or educational examples like Bodoff's examples and Pricing Insurance Risk case studies).

It is indexed by object kind (severity, aggregate, portfolio) and name, and accessed as the read-only property :attr:`build.knowledge`. Here are the first five rows of the knowledge loaded by ``build``.

.. ipython:: python
    :okwarning:

    qd(build.knowledge.head(), justify="left", max_colwidth=60)

A row in the knowledge can be accessed by name using ``build``. This example models the roll of a single die.

.. ipython:: python
    :okwarning:

    print(build['B.Dice10'])

The argument ``'B.Dice10'`` is passed through to the underlying dataframe's ``getitem``.

.. _10 min create from knowledge:

A row in the knowledge can be created as a Python object using:

.. ipython:: python
    :okwarning:

    aDice = build('B.Dice10')
    qd(aDice)

The argument in this case is passed through to the method :meth:`Underwriter.build`, which first looks for ``B.Dice10`` in the knowledge. If it fails, it tries to interpret its argument as a DecL program.

The method :meth:`build.qshow` (quick show) searches the knowledge using a regex (regular expression) applied to the names, returning a dataframe of specifications. :meth:`build.qlist` (quick list) just displays them.

.. ipython:: python
    :okwarning:

    build.qlist('Dice')

The method :meth:`build.show` also searches the knowledge using a regex applied to the names, but it creates and plots each match by default. Be careful not to create too many objects! Try running::

    ans, df = build.show('Dice', return_df=True)

It returns a list ``ans`` of created objects and a dataframe ``df`` containing information about each.

.. _10 min bts:

:class:`Underwriter` Behind the Scenes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section should be skipped the first time through.

Each object has a kind property and a name property, and it can be manifest as a DecL program, a dictionary specification, or a Python class instance. The class can be updated or not updated. In detail:

1. kind equals sev for a :class:`Severity`, agg for a :class:`Aggregate`, port for a :class:`Portfolio`, and distortion for a :class:`Distortion` (dist could be distribution);
2. name is assigned to the object by the user; it is different from the Python variable name holding the object;
3. spec is a (derived) dictionary specification;
4. program is the DecL program as a text string; and
5. object is the actual Python object, an instance of a class.

:meth:`Underwriter.write` is a low-level creator function. It takes a DecL program or knowledge item name as input.

* It searches the knowledge for the argument and returns it if it finds one object. It throws an error if the name is not unique. If the name is not in the knowledge it continues.
* It calls :meth:`Underwriter.interpret_program` to pre-process the DecL and then lex and parse it one line at a time.
* It looks up occurrences of ``sev.ID``, ``agg.ID`` (``ID`` is an object name) in the knowledge and replaces them with their definitions.
* It calls :meth:`Underwriter.factory` to create any objects and update them if requested.
* It returns a list of :class:`Answer` objects, with kind, name, spec, program, and object attributes.

:meth:`Underwriter.write_file` reads a file and passes it to :meth:`Underwriter.write`. It is a convenience function.

The :meth:`Underwriter.build` method wraps the
:meth:`Underwriter.write` and provides sensible defaults to shield the user from its internal details. :math:`build` takes the following steps:

* It calls :meth:`write` with ``update=False``.
* It then estimates sensible hyper-parameters and uses them to :meth:`update` the object's discrete distribution. It tries to distinguish discrete output distributions from continuous or mixed ones.
* If the DecL program produces only one output, it strips it out of the answer returned by ``write`` and returns just that object.
* If the DecL program produces only one portfolio output (but possibly other non-portfolio objects), it returns just that.

:meth:`Underwriter.interpret_program` interprets DecL programs and matches them with the parsed specs in an ``Answer(kind, name, spec, program, object=None)`` object. It adds the result to the knowledge.

:meth:`Underwriter.factory` takes an ``Answer`` argument and updates it by creating the relevant object and updating it if ``build.update is True``.

A set of methods called :meth:`interpreter_xxx` run DecL  programs through parser for debugging purposes, but do not create any output or add anything to the knowledge.

* :meth:`Underwriter.interpreter_line` works on one line.
* :meth:`Underwriter.interpreter_file`  works on each line in a file.
* :meth:`Underwriter.interpreter_list` works on each item in a list.
* :meth:`Underwriter._interpreter_work` does the actual parsing.

.. _10 min how:

How ``aggregate`` Represents Distributions
--------------------------------------------

A distribution is represented as a discrete numerical approximation. To "know or compute a distribution" means that we have a discrete stair-step approximation to the true distribution function that jumps (is supported) only on integer multiples of a fixed bandwidth or bucket size :math:`b` (called ``bs`` in the code). The distribution is represented by :math:`b` and a vector of probabilities :math:`(p_0,p_1,\dots, p_{n-1})` with the interpretation

.. math:: \Pr(X=kb)=p_k.

All subsequent computations assume that **this approximation is the distribution**. For example, moments are estimated using

.. math:: \mathsf E[X^r] = b\,\sum_k k^r p_k.

See :ref:`num how agg reps a dist` for more details.


.. _10 min Severity:

The :class:`Severity` Class
-------------------------------

The :class:`Severity` class derives from :class:`scipy.stats.rv_continuous`, see `scipy help <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html>`_. It contains a member ``stats.rv_continuous`` variable ``fz`` that is the ground-up unlimited severity and it adds support for limits and attachments. For example, the cdf function is coded:

.. code:: python

    def _cdf(self, x, *args):
        if self.conditional:
            return np.where(x >= self.limit, 1,
                np.where(x < 0, 0,
                         (self.fz.cdf(x + self.attachment) -
                         (1 - self.pattach)) / self.pattach))
        else:
            return np.where(x < 0, 0,
                np.where(x == 0, 1 - self.pattach,
                     np.where(x > self.limit, 1,
                          self.fz.cdf(x + self.attachment, *args))))

:class:`Severity` can determine its shape parameter from a CV analytically for lognormal, gamma, inverse gamma, and inverse Gaussian distributions, and attempts to use a Newton-Raphson method to determine it for all other one-shape parameter distributions. (The CV is adjusted using the scale factor for zero parameter distributions.) Once the shape is known, it uses scaling to produce the required mean. **Warning:** The numerical methods are not always reliable.

.. fail for pareto and loggamma with 10 cv .5 for example

:class:`Severity` computes layer moments analytically for the lognormal, Pareto, and gamma, and uses numerical integration of the quantile function (``isf``) for all other distributions. These estimates can become unreliable for very thick tailed distributions. It uses ``self.fz.stats('mvs')`` and the object limit to determine if the requested moment actually exists before attempting numerical integration.

:class:`Severity` has a :meth:`plot` method that graphs the density, log density, cdf, and quantile (Lee) functions.

A :class:`Severity` can be created using DecL using any of the following five forms.

#. ``sev NAME sev.BUILDIN_ID`` is a knowledge lookup for ``BUILTIN_ID``

#. ``sev NAME DISTNAME SHAPE1 <SHAPE2>`` where ``DISTAME`` is the name of any ``scipy.stats`` continuous random variable with zero, one, or two shape parameters, see the :ref:`DecL/list of distributions`.

#. ``sev NAME SCALE * DISTNAME SHAPE1 <SHAPE2> + LOC``

#. ``sev NAME DISTNAME MEAN cv CV``

#. ``sev NAME SCALE * DISTNAME MEAN cv CV + LOC`` or ``sev NAME SCALE * DISTNAME MEAN cv CV - LOC``

Either or both of ``SCALE`` and ``LOC`` can be present. In the mean and CV form, the mean refers to the unshifted, unscaled mean, but the CV refers to the shifted and scaled CV --- because you usually want to control the overall CV.

**Example.**

``lognorm 80 cv 0.5`` results in an unshifted lognormal with mean 80 and CV 0.5.

.. ipython:: python
    :okwarning:

    s0 = build(f'sev TenM:Sev.1 '
                'lognorm 80 cv .5')
    mf, vf = s0.fz.stats(); m, v = s0.stats()
    s0.plot(figsize=(2*3.5, 2*2.45+0.15), layout='AB\nCD');
    @savefig 10min_sev0.png scale=20
    plt.gcf().suptitle(f'{s0.name}, mean {m:.2f}, CV {v**.5/m:.2f} ({mf:.2f}, {vf**.5/mf:.2f})');
    print(m,v,mf,vf)

Combining scaling, shifts, and mean/cv entry like so ``10 * lognorm 1 cv 0.5  + 70`` results in a distribution with mean ``10 * 1 + 70 = 80``, a standard deviation of ``10 * 0.5 = 5``, and a cv of ``5 / 80``.

.. ipython:: python
    :okwarning:

    s1 = build(f'sev TenM:Sev.2 '
                '10 * lognorm 1 cv .5 + 70')
    mf, vf = s1.fz.stats(); m, v = s1.stats()
    s1.plot(figsize=(2*3.5, 2*2.45+0.15), layout='AB\nCD');
    @savefig 10min_sev1.png scale=20
    plt.gcf().suptitle(f'{s1.name}, mean {m:.2f}, CV {v**.5/m:.2f} ({mf:.2f}, {vf**.5/mf:.2f})');
    print(m,v,mf,vf)

**Examples.**

This example compares the shapes of gamma, inverse Gaussian, lognormal, and inverse gamma severities with the same mean and CV. First, a short function to create the examples.

.. ipython:: python
    :okwarning:

    def plot_example(dist_name):
        s = build(f'sev TenM:{dist_name.title()} '
                  f'{dist_name} 10 cv .5')
        m, v, sk, k = s.fz.stats('mvsk')
        s.plot(figsize=(2*3.5, 2*2.45+0.15), layout='AB\nCD')
        plt.gcf().suptitle(f'{dist_name.title()}, mean {m:.2f}, '
                           f'CV {v**.5/m:.2f}, skew {sk:.2f}, kurt {k:.2f}')

Execute on the desired distributions.

.. ipython:: python
    :okwarning:

    @savefig 10min_sev2.png scale=20
    plot_example('gamma')
    @savefig 10min_sev3.png scale=20
    plot_example('invgauss')
    @savefig 10min_sev4.png scale=20
    plot_example('lognorm')
    @savefig 10min_sev5.png scale=20
    plot_example('invgamma')

**Examples.**

This example show the impact of adding a limit and attachment.
Limits and attachments determine exposure in DecL and they belong to the :class:`Aggregate` specification. DecL cannot be used to set the limit and attachment of a :class:`Severity` object. One way to apply them is to create an aggregate with a fixed frequency of one claim. By default, the severity is conditional on a loss to the layer.

.. ipython:: python
    :okwarning:

    limit, attach = 15, 5
    s2 = build(f'agg TenM:SevLayer 1 claim {limit} xs {attach} sev gamma 10 cv .5 fixed')
    m, v, sk, k = s2.sevs[0].fz.stats('mvsk')
    s2.sevs[0].plot(n=401, figsize=(2*3.5, 2*2.45+0.3), layout='AB\nCD')
    @savefig 10min_sev6.png scale=20
    plt.gcf().suptitle(f'Ground-up severity\nGround-up gamma mean {m:.2f}, CV {v**0.5/m:.2f}, skew {sk:.2f}, kurt {k:.2f}\n'
                       f'{limit} xs {attach} excess layer mean {s2.est_m:.2f}, CV {s2.est_cv:.2f}, skew {s2.est_skew:.2f}, kurt {k:.2f}');


------

A  :class:`Severity` can be created directly using ``args`` and ``kwargs``. Here is an example. It also shows the impact of making the severity unconditional (on a loss to the layer). Start by creating the conditional (default) severity and plotting it.

.. ipython:: python
    :okwarning:

    from aggregate import Severity
    s3 = Severity('gamma', attach, limit, 10, 0.5)
    s3.plot(n=401, figsize=(2*3.5, 2*2.45+0.15), layout='AB\nCD')
    m, v = s3.stats()
    @savefig 10min_sev6.png scale=20
    plt.gcf().suptitle(f'{limit} xs {attach} excess layer mean {m:.2f}, CV {v**.5/m:.2f}');

Next, create an unconditional version. The lower pdf is scaled down by the probability of attaching the layer, and the left end of the cdf shifted up by the probability of not attaching the layer. These probabilities are given by the underlying ``fz`` object's sf and cdf.

.. ipython:: python
    :okwarning:

    s4 = Severity('gamma', attach, limit, 10, 0.5, sev_conditional=False)
    s4.plot(figsize=(2*3.5, 2*2.45+0.15), layout='AB\nCD')
    m, v = s4.stats()
    @savefig 10min_sev7.png scale=20
    plt.gcf().suptitle(f'Unconditional {limit} xs {attach} excess layer mean {m:.2f}, CV {v**.5/m:.2f}');
    print(f'Probability of attaching layer {s4.fz.cdf(attach):.3f}')

------

Although :class:`Severity` accepts a weight argument, it does not actually support weighted severities. It models only one component. :class:`Aggregate` handles weighted severities by creating a separate :class:`Severity` for each component.

.. _10 min Aggregate:

The :class:`Aggregate` Class
-------------------------------

.. TODO

    * Exist in updated and non-updated state.
    * homog and inhomog multiply of built in aggs!! See Treaty 5 from Bear and Nemlick.

.. _10 min creating agg:

Creating an Aggregate Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`Aggregate` objects can be created in three ways:

#.  Generally, they are created using DecL by :meth:`Underwriter.build`, as shown in :ref:`10 min create from decl`.

#. Objects in the knowledge can be :ref:`created by name<10mins create from knowledge>`.

#. Advanced users and programmers can create :class:`Aggregate` objects directly using ``kwargs``, see :ref:`Aggregate Class`.


**Example.**

This example uses :meth:`build` to make an :class:`Aggregate` with a Poisson frequency, mean 5, and gamma severity with mean 10 and CV 1 . It includes more discussion than the example above. The line breaks improve readability but are cosmetic.

.. ipython:: python
    :okwarning:

    a02 = build('agg TenM:02 '
                '5 claims '
                'sev gamma 10 cv 1 '
                'poisson')
    qd(a02)

``qd`` displays the dataframe ``a.describe``. This example fails the aliasing validation test because the aggregate mean error is suspiciously greater than the severity. (Run with logger level 20 for more diagnostics.) However, it passes both the severity mean and aggregate mean tests.

.. _10 min quick diagnostics:

Aggregate Quick Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quick display reports a set of quick diagnostics, showing

* Exact ``E[X]`` and estimated ``Est E[X]`` frequency, severity, and aggregate statistics.
* Relative errors ``Err E[X]`` for the means.
* Coefficient of variation ``CV(X)`` and estimated CV, ``Est CV(X)``
* Skewness ``Skew(X)`` and estimated skewness, ``Est Skew(X)``
* The (log to base 2) of the number of buckets used, ``log2``
* The bucket size ``bs`` used in discretization

These statistics make it help to test if the numerical estimation is  valid. Look for a small error in the mean and close second (CV) and third (skew) moments.
The last item ``valid = True`` shows the model passes some basic validation tests. Strictly, it means the model did not fail any tests: it is not unreasonable. The test should be interpreted like a null hypothesis; you expect it to be True and are worried when it is False.

In this case, the aggregate mean error is too high because the discretization bucket size ``bs`` is too small. Update with a larger bucket.

.. ipython:: python
    :okwarning:

    a02.update(bs=1/128)
    qd(a02)


.. _10 min algo deets:

Aggregate Algorithm in Detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Here's the ``aggregate`` FFT convolution algorithm stripped down to bare essentials and coded in raw Python to show you what happens behind the curtain. The algorithm steps are:

#. Inputs

    - Severity distribution cdf. Use ``scipy.stats``.
    - Frequency distribution probability generating function. For a Poisson with mean :math:`\lambda` the PGF is :math:`\mathscr P(z) = \exp(\lambda(z - 1))`.
    - The bucket size :math:`b`. Use the value ``simple.bs``.
    - The number of buckets :math:`n=2^{log_2}`. Use the default ``log2=16`` found in ``simple.log2``.
    - A padding parameter, equal to 1 by default, from ``simple.padding``.

#. Discretize the severity cdf.
#. Apply the FFT to discrete severity with padding to size ``2**(log2 + padding)``.
#. Apply the frequency pgf to the FFT.
#. Apply the inverse FFT to create is a discretized version of the aggregate distribution and output it.

Let's recreate the following simple example. The variable names for the means and shape are for clarity. ``sev_shape`` is :math:`\sigma` for a lognormal.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    en = 50
    sev_scale = 10
    sev_shape = 0.8
    simple = build('agg Simple '
                   f'{en} claims '
                   f'sev {sev_scale} * lognorm {sev_shape} '
                   'poisson')
    qd(simple)

The next few lines of code implement the FFT convolution algorithm. Start by importing the probability distribution and FFT routines. ``rfft`` and ``irfft`` take the FFT and inverse FFT of real input.

.. ipython:: python
    :okwarning:

    import numpy as np
    from scipy.fft import rfft, irfft
    import scipy.stats as ss

Pull parameters from ``simple`` to match calculations, step 1. ``n_pad`` is the length of the padded vector used in the convolution to manage aliasing.

.. ipython:: python
    :okwarning:

    bs = simple.bs
    log2 = simple.log2
    padding = simple.padding
    n = 1 << log2
    n_pad = 1 << (log2 + padding)
    sev = ss.lognorm(sev_shape, scale=sev_scale)

Use the ``round`` method and the survival function to discretize, completing step 2.

.. ipython:: python
    :okwarning:

    xs = np.arange(0, (n + 1) * bs, bs)
    discrete_sev = -np.diff(sev.sf(xs - bs / 2))

The next line of code carries out algorithm steps 3, 4, and 5!
All the magic happens here. The forward FFT adds padding, but the answer must  be unpadded manually, with the final ``[:n]``.

.. ipython:: python
    :okwarning:

    agg = irfft( np.exp( en * (rfft(discrete_sev, n_pad) - 1) ) )[:n]

Plots to compare the two approaches. They are spot on!

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45),
        constrained_layout=True);                                    \
    ax0, ax1 = axs.flat;                                             \
    simple.density_df.p_total.plot(lw=2, label='Aggregate', ax=ax0); \
    ax0.plot(xs[:-1], agg, lw=1, label='By hand');                   \
    ax0.legend();                                                    \
    simple.density_df.p_total.plot(lw=2, label='Aggregate', ax=ax1); \
    ax1.plot(xs[:-1], agg, lw=1, label='By hand');                   \
    ax1.legend();
    @savefig 10mins_byhand.png scale=20
    ax1.set(yscale='log');

The very slight difference for small loss values arises because ``build`` removes numerical fuzz, setting values below machine epsilon (about ``2e-16``) to zero, explaining why the blue aggregate line drops off vertically on the left.



.. _10 min basic prob:

Basic Probability Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`Aggregate` object acts like a discrete probability distribution. There are properties for the aggregate and severity mean, standard deviation, coefficient of variation, and skewness, both computed exactly and numerically estimated.

.. ipython:: python
    :okwarning:

    print(a02.agg_m, a02.agg_sd, a02.agg_cv, a02.agg_skew)
    print(a02.est_m, a02.est_sd, a02.est_cv, a02.est_skew)
    print(a02.sev_m, a02.sev_sd, a02.sev_cv, a02.sev_skew)
    print(a02.est_sev_m, a02.est_sev_sd, a02.est_sev_cv, a02.est_sev_skew)

They have probability mass, cumulative distribution, survival, and quantile (inverse of distribution) functions.

.. ipython:: python
    :okwarning:

    a02.pmf(60), a02.cdf(50), a02.sf(60), a02.q(a02.cdf(60)), a02.q(0.5)

The pdf, cdf, and sf for the underlying severity are also available.

.. ipython:: python
    :okwarning:

    a02.sev.pdf(60), a02.sev.cdf(50), a02.sev.sf(60)

.. note::

    :class:`Aggregate` and :class:`Portfolio` objects need to be updated after they have been created. Updating builds out discrete numerical approximations, analogous to simulation. By default, :meth:`build` handles updating automatically.

.. warning::

    Always use bucket sizes that have an exact binary representation (integers, 1/2, 1/4, 1/8, etc.) **Never** use 0.1 or 0.2 or other numbers that do not have an exact float representation, see REF.

.. _10 min mixtures:

Mixtures
~~~~~~~~~~~~

An :class:`Aggregate` can have a mixed severity. The mixture can include different distributions, parameters, shifts, and locations.

.. ipython:: python
    :okwarning:

    a03 = build('agg TenM:03 '
                '25 claims '
                'sev [gamma lognorm invgamma] [5 10 10] cv [0.5 0.75 1.5] '
                'wts [.5 .25 .25] + [0 10 20] '
                'mixed gamma 0.5'
               , bs=1/16)
    qd(a03)

An :class:`Aggregate` can model multiple units at once, and allow them to share mixing variables. This induces correlation between the components, see the :ref:`report dataframe <10mins extra info>`. All parts of the specification can vary, including limits and attachments (not shown). This case differentiated from a mixed severity by having no weights.

.. ipython:: python
    :okwarning:

    a04 = build('agg TenM:04 '
                '[500 250 100] premium at [.8 .7 .5] lr '
                'sev [gamma lognorm invgamma] [5 10 10] cv [0.5 0.75 1.5] '
                'mixed gamma 0.5'
               , bs=1/8)
    qd(a04)


.. _10 min accessing:

Accessing Severity in an :class:`Aggregate`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The attribute :class:`Aggregate.sevs` is an array of the :class:`Severity`
objects. Usually, it contains only one distribution but when severity is a
mixture it contains one for each mixture component. It can be iterated over.
Each :class:`Severity` object wraps a ``scipy.stats`` continuous random
variable called ``fz`` that represents ground-up severity. The ``args`` are its
shape variable(s) and ``kwds`` its scale and location variables. This is
most interesting when the object has a mixed severity.

.. ipython:: python
    :okwarning:

    for s in a03.sevs:
        print(s.sev_name, s.fz.args, s.fz.kwds)

The property ``a03.sev`` is a ``namedtuple`` exposing the exact weighted pdf,
cdf, and sf of the underlying :class:`Severity` objects.

.. ipython:: python
    :okwarning:

    a03.sev.pdf(20), a03.sev.cdf(20), a03.sev.sf(20)

The component weights are proportional to ``a03.en`` and ``a03.sev.cdf`` is computed as

.. ipython:: python
    :okwarning:

    (np.array([s.cdf(20) for s in a03.sevs]) * a03.en).sum() / a03.en.sum()

The following are equal using the defaut discretization method.

.. ipython:: python
    :okwarning:

    a03.density_df.loc[20, 'F_sev'], a03.sev.cdf(20 + a03.bs/2)

.. _10 min reinsurance:

Reinsurance
~~~~~~~~~~~~~~~

:class:`Aggregate` objects can apply per occurrence and aggregate reinsurance using clauses

* ``occurrence net of [limit] xs ]attach]``
* ``occurrence net of [pct] so [limit] xs [attach]``, where ``so`` stands for "share of"
* ``occurrence ceded to [limit] xs ]attach]``
* and so forth.

**Examples.**

Gross distribution: a triangular aggregate created as the sum of two uniform distribution on 1, 2,..., 10.

.. ipython:: python
    :okwarning:

    a05g = build('agg TenM:05g dfreq [2] dsev [1:10]')
    qd(a05g)


Apply 3 xs 7 occurrence reinsurance to cap individual losses at 7. ``a05no`` is the net of occurrence distribution.

.. ipython:: python
    :okwarning:

    a05no = build('agg TenM:05no dfreq [2] dsev [1:10] '
                'occurrence net of 3 x 7')
    qd(a05no)

.. warning::

   The ``describe`` dataframe always reports gross analytic statistics (``E[X]``, ``CV(X)``, ``Skew(X)``) and the requested net or ceded estimated statistics (``Est E[X]``, ``Est CV(X)``, ``Est Skew(X)``). Look at the gross portfolio first to check computational accuracy. Net and ceded "error" report the difference to analytic gross.

Add an aggregate 4 xs 8 reinsurance cover on the net of occurrence distribution. ``a05n`` is the final net distribution.

.. ipython:: python
    :okwarning:

    a05n = build('agg TenM:05n dfreq [2] dsev [1:10] '
               'occurrence net of 3 xs 7 '
               'aggregate net of 4 xs 8')
    qd(a05n)

See :ref:`10 min plot` for plots of the different distributions.

.. _10 min Distortion:

The :class:`Distortion` Class
-------------------------------

See :doc:`../5_technical_guides/5_x_distortions` and PIR Chapter 10.5 for more information about distortions.

A :class:`Distortion` can be created using DecL.
It object has methods for ``g``, the distortion function, along with its dual ``g_dual(s)=1-g(1-s)`` and inverse ``g_inv``. The :meth:`plot` method shows ``g`` (above the diagonal) and ``g_inv`` (below).

.. ipython:: python
    :okwarning:

    d06 = build('distortion TenM:06 dual 3')
    qd(d06.g(.2), d06.g_inv(.2), d06.g_dual(0.2),
    d06.g(.8), d06.g_inv(.992), d06)
    @savefig 10mins_dist.png scale=20
    d06.plot();

The :class:`Distortion` class can create distortions from a number of parametric families.

.. ipython:: python
    :okwarning:

    from aggregate import Distortion
    Distortion.available_distortions(False, False)

Run the command::

    Distortion.test()

for graphs of samples from each available family. ``tt`` is not a distortion because it is not concave. It is included for historical reasons.

.. _10 min Portfolio:

The :class:`Portfolio` Class
-------------------------------

A :class:`Portfolio` object models a portfolio (book, block) of units (accounts, lines, business units, regions, profit centers), each represented as an :class:`Aggregate`. It uses FFTs to convolve (add) the unit distributions. By default, all the units are assumed to be independent, though there are ways to adjust this. REF. The independence assumption is not as bad as it may appear; its effect can be ameliorated by selecting units carefully and sharing mixing variables appropriately (see REF for further discussion).

:class:`Portfolio` objects have all of the attributes and methods of a :class:`Aggregate` and add methods for pricing and allocation to units.

The DecL for a portfolio is simply::

    port NAME AGG1 <AGG2> <AGG3> ...

where ``AGG1`` is an aggregate specification. Portfolios can have one or more units. The DecL can be split over multiple lines if each aggregate begins on a new line and is indented by a tab (like a Python function).

**Example.**

Here is a three-unit portfolio built using a DecL program. The line breaks and horizontal spacing are cosmetic since Python just concatenates the input.

.. ipython:: python
    :okwarning:

    p07 = build('port TenM:07 '
                'agg A '
                    '100 claims '
                    '10000 xs 0 '
                    'sev lognorm 100 cv 1.25 '
                    'poisson '
                'agg B '
                    '150 claims '
                    '2500 xs 5 '
                    'sev lognorm 50 cv 0.9 '
                    'mixed gamma .6 '
                'agg Cat '
                    '2 claims '
                    '1e5 xs 0 '
                    'sev 500 * pareto 1.8 - 500 '
                    'poisson'
               , approximation='exact', padding=2)
    qd(p07)

The portfolio units are called A, B and Cat. Printing using ``qd`` shows ``p07.describe``, which concatenates each unit's ``describe`` and adds the same statistics for the total.

* Unit A has 100 (expected) claims, each pulled from a lognormal distribution with mean of 30 and coefficient of variation 1.25 within the layer 100 xs 0 (i.e., losses are limited at 100). The frequency distribution is Poisson.
* Unit B is similar.
* The Cat unit is has expected frequency of 2 claims from the indicated limit, with severity given by a Pareto distribution with shape parameter 1.8, scale 500, shifted left by 500. This corresponds to the usual Pareto with survival function :math:`S(x) = (500 / (500 + x))^{1.8} = (1 + x / 500)^{-1.8}` for :math:`x \ge 0`.

The portfolio total (i.e., the sum of the units) is computed using FFTs to convolve (add) the unit's aggregate distributions. All computations use the same discretization bucket size; here the bucket-size ``bs=2``. See :ref:`For Portfolio Objects`.

A :class:`Portfolio` object acts like a discrete probability distribution, the same as an :class:`Aggregate`. There are properties for the mean, standard deviation, coefficient of variation, and skewness, both computed exactly and numerically estimated.

.. ipython:: python
    :okwarning:

    print(p07.agg_m, p07.agg_sd, p07.agg_cv, p07.agg_skew)
    print(p07.est_m, p07.est_sd, p07.est_cv, p07.est_skew)

They have probability mass, cumulative distribution, survival, and quantile (inverse of distribution) functions.

.. ipython:: python
    :okwarning:

    p07.pmf(12000), p07.cdf(11000), p07.sf(12000), p07.q(p07.cdf(12000)), p07.q(0.5)


The names of the units in a :class:`Portfolio` are in a list called ``p07.unit_names`` or ``p07.unit_names_ex`` including ``total``.
The :class:`Aggregate` objects in the :class:`Portfolio` can be iterated over.

.. ipython:: python
    :okwarning:

    for u in p07:
        print(u.name, u.agg_m, u.est_m)

.. _10 min est bs:

Estimating Bucket Size for Discretization
-------------------------------------------

Selecting an appropriate bucket size ``bs`` is critical to obtaining accurate results. This is a hard problem that may have hindered broad adoption of FFT-based methods.

See :doc:`../5_technical_guides/5_x_numerical_methods` for further discussion.

.. _10 min hyper:

Hyper-parameters ``log2`` and ``bs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hyper-parameters ``log2`` and ``bs`` control numerical calculations.
``log2`` equals the log to base 2 of the number of buckets used and ``bs``
equals the bucket size. These values are printed by ``qd``.

.. _10 min agg bucket:

Estimating and Testing ``bs`` For :class:`Aggregate` Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an :class:`Aggregate`, :meth:`recommend_bucket` uses a shifted lognormal
method of moments fit and takes the ``recommend_p`` percentile as the
right-hand end of the discretization. By default ``recommend_p=0.999``, but
for thick tailed distributions it may be necessary to use a value closer to
1. :meth:`recommend_bucket` also considers any limits: ideally limits are
multiples of the bucket size.

The recommended value of ``bs`` should rounded up to a binary fraction
(denominator is a power of 2) using :meth:`utilities.round_bucket`.

:class:`Aggregate` also includes two functions for assessing ``bs``,
one based on the overall error and one based on looking at each severity
component.

:meth:`Aggregate.aggregate_error_analysis` updates the object at a range of
different ``bs`` values and reports the total absolute (strictly, signed
absolute error) and relative error as well as an upper bound ``bs/2`` on
the absolute value of the discretization error. ``log2`` must be input and,
optionally, the log base 2 of the smallest bucket to model. It then models
six doublings of the input bucket. If no bucket is input, it models three
doublings up and down from the rounded :meth:`recommend_bucket` suggestion.
The output table shows:

* The actual ``(agg, m)`` and estimated ``(est, m)`` means, from the ``describe`` dataframe.
* The implied absolute ``(abs, m)``  and relative ``(rel, m)`` errors in the mean.
* ``(rel, h)`` shows the maximum relative severity discretization error, which equals ``bs / 2`` divided by the average severity.
* ``(rel, total)``, equal to the sum of ``(rel, h)`` and ``rel m``.

Thick tailed distributions can favor a large bucket size without regard to the impact on discretization; accounting for the impact of ``bs / 2`` is a countervailing force.

.. ipython:: python
    :okwarning:

    qd(a04.aggregate_error_analysis(16), sparsify=False, col_space=9)

:meth:`Aggregate.severity_error_analysis` performs a detailed error analysis of each severity component. It reports:

* The name, limit, attachment, and truncation point for each severity component.
* ``S`` the probability the component (or total losses) exceed the truncation.
* ``sum_p`` the sum of discrete probabilities, which can be :math:`<1` if ``normalize=False``.
* ``wt`` the weight of the component and ``en`` the corresponding claim count.
* ``agg_mean`` and ``agg_wt`` the aggregate mean contribution from the component (sums to the overall mean), and the each component's proportion of the total. The loss weight can differ drastically from the count weight.
* ``mean`` and ``est_mean`` the analytic and estimated severity by component and the corresponding ``abs`` and ``rel`` error.
* ``trunc_error`` the truncation error by component (tail integral) and relative truncation error.
* The ``h_error`` based on ``bs / 2`` by component, a (conservative) upper bound on discretization error and the relative error compared to the component mean.
* ``h2_adj`` and ``rel_h2_adj`` estimate a second order adjustment to the numerical mean. They give a better idea of the discretization error.

.. ipython:: python
    :okwarning:

    qd(a04.severity_error_analysis(), line_width=75)

Generally there is either discretization or truncation error. Look for one of them to dominate. Discretization error is solved with a smaller bucket; truncation with a larger. When the two conflict, add more buckets by increasing ``log2``.

.. _10 min port bucket:

Estimating and Testing ``bs`` For :class:`Portfolio` Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a :class:`Portfolio`, the right hand end of the distribution is estimated using the square root of sum of squares (proxy independent sum) of the right hand ends of each unit.

The method :meth:`port.recommend_bucket` suggests a reasonable bucket size.

.. ipython:: python
    :okwarning:

    print(p07.recommend_bucket().iloc[:, [0,3,6,10]])
    p07.best_bucket(16)

The column ``bsN`` corresponds to discretizing with 2**N buckets. The rows show suggested bucket sizes by unit and in total. For example with ``N=16`` (i.e., 65,536 buckets) the suggestion is 2.19. It is best the bucket size is a divisor of any limits or attachment points. :meth:`best_bucket` takes this into account and suggests 2.

To test ``bs``, run the tests above on each unit.

.. _10 min common:

Methods and Properties Common To :class:`Aggregate` and :class:`Portfolio` Classes
------------------------------------------------------------------------------------


:class:`Aggregate` and :class:`Portfolio` both have the following methods and properties. See :ref:`Aggregate Class` and :ref:`Portfolio Class` for full lists.

- ``info`` and  ``describe`` are dataframes with statistics and other information; they are printed with the object.

- ``density_df`` a dataframe containing estimated probability distributions and other expected value information.

- The :attr:`statistics` dataframe shows analytically computed mean, variance, CV, and sknewness for each unit and in total.

- ``report_df`` are dataframe with information to test if the numerical approximations appear valid. Numerically estimated statistics are prefaced ``est_`` or ``empirical``.

- ``log2`` and ``bs`` hyper-parameters that control numerical calculations.

- ``spec`` a dictionary containing the ``kwargs`` needed to recreate each object. For example, if ``a`` is an :class:`Aggregate` object, then :class:`Aggregate(**a.spec)` creates a new copy.

- ``spec_ex`` a dictionary that appends hyper-parameters to ``spec`` including ``log2`` and ``bs``.

- ``program`` the DecL program used to create the object. Blank if the object has been created directly. (A given object can often be created in different ways by DecL, so there is no obvious reverse mapping from the ``spec``.)

- ``renamer`` a dictionary used to rename columns of member dataframes to be more human readable.

- :meth:`update` a method to run the numerical calculation of probability distributions.

- :meth:`recommend_bucket` to recommend the value of ``bs``.
- Common statistical functions including pmf, cdf, sf, the quantile function (value at risk) and tail value at risk.

- Statistical functions: pdf, cdf, sf, quantile, value at risk, tail value at risk, and so on.

- :meth:`plot` method to visualize the underlying distributions. Plots the pmf and log pmf functions and the quantile function. All the data is contained in ``density_df`` and the plots are created using ``pandas`` standard plotting commands.

- :meth:`price` to apply a distortion (spectral) risk measure pricing rule with a variety of capital standards.

- :meth:`snap` to round an input number to the index of ``density_df``.

- :meth:`approximate` to create an analytic approximation.

- :meth:`sample` pulls samples, see :ref:`samp samp`.

.. _10 min info:

The ``info`` Dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``info`` dataframe contains information about the frequency and severity stochastic models, how the object was computed, and any reinsurance applied (none in this case).

.. ipython:: python
    :okwarning:

    print(a05n.info)
    print(p07.info)

.. _10 min describe:

The ``describe`` Dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``describe`` dataframe contains gross analytic and estimated (net or ceded) statistics. When there is no reinsurance, comparison of analytic and estimated moments provides a test of computational accuracy (first case). It should always be reviewed after updating. When there is reinsurance, empirical is net (second case).

.. ipython:: python
    :okwarning:

    qd(a05g.describe)
    with pd.option_context('display.max_columns', 15):
        print(a05n.describe)
    qd(p07.describe)

Printing with default settings shows what ``qd`` adds.


.. _10 min density_df:

The ``density_df`` Dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``density_df`` dataframe contains a wealth of information. It has ``2**log2`` rows and is indexed by the outcomes, all multiples of ``bs``. Columns containing ``p`` are the probability mass function, of the aggregate or severity.

- the aggregate and severity pmf (called ``p`` and duplicated as ``p_total`` for consistency with :class:`Portfolio` objects), log pmf, cdf and sf
- the aggregate lev (duplicated as ``exa``)
- ``exlea`` (less than or equal to ``a``) which equals :math:`\mathsf E[X\mid X\le a]` as a function of ``loss``
- ``exgta`` (greater than) which equals :math:`\mathsf E[X\mid X > a]`

In an :class:`Aggregate`, ``p`` and ``p_total`` are identical, the latter included for consistency with :class:`Portfolio` output. ``F`` and ``S`` are the cdf and sf (survival function). ``lev`` and ``exa`` are identical and equal the limited expected value at the ``loss`` level. Here are the first five rows.

.. ipython:: python
    :okwarning:

    print(a05g.density_df.shape)
    print(a05g.density_df.columns)
    with pd.option_context('display.max_columns', a05g.density_df.shape[1]):
        print(a05g.density_df.head())

The :class:`Portfolio` version is more exhaustive. It includes a variety of columns for each unit, suffixed ``_unit``, and for the complement of each unit (sum of everything but that unit) suffixed ``_ημ_unit``. The totals are suffixed ``_total``. The most important columns are ``exeqa_unit``, :ref:`10 min conditional expected values`. All the column names and a subset of ``density_df`` are shown next.

.. ipython:: python
    :okwarning:

    print(p07.density_df.shape)
    print(p07.density_df.columns)
    with pd.option_context('display.max_columns', p07.density_df.shape[1]):
        print(p07.density_df.filter(regex=r'[aipex012]_A').head())

.. _10 min stats:

The ``statistics`` Series and Dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``statistics`` dataframe shows analytically computed mean, variance, CV, and sknewness. It is indexed by

- severity name, limit and attachment,
- ``freq1, freq2, freq3`` non-central frequency moments,
- ``sev1, sev2, sev3`` non-central severity moments, and
- the mean, cv and skew(ness).

It applies to the **gross** outcome when there is reinsurance, so the results for ``a05g`` and ``a05no`` are the same.

.. ipython:: python
    :okwarning:

    oco = ['display.width', 150, 'display.max_columns', 15,
            'display.float_format', lambda x: f'{x:.5g}']
    with pd.option_context(*oco):
        print(a05g.statistics)
        print('\n')
        print(p07.statistics)

.. _10 min report:

The ``report_df`` Dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``report_df`` dataframe combines information from ``statistics`` with
estimated moments to test if the numerical approximations appear valid. It
is an expanded version of ``describe``. Numerically estimated statistics are
prefaced ``est`` or ``empirical``.

.. ipython:: python
    :okwarning:

    with pd.option_context(*oco):
        print(a05g.report_df)
        print('\n')
        print(p07.report_df)

.. _10mins extra info:

The ``report_df`` provides extra information when there is a mixed severity.

.. ipython:: python
    :okwarning:

    with pd.option_context(*oco):
        print(a03.report_df)

The dataframe shows statistics for each mixture component, columns ``0,1,2``,
their sum if they are added independently and their sum if there is a shared
mixing variable, as there is here. The common mixing induces correlation
between the mix components, acting to increases the CV and skewness, often
dramatically.

.. _10 min spec:

The ``spec`` and ``spec_ex`` Dictionaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``spec`` dictionary contains the input information needed to create each
object. For example, if ``a`` is an :class:`Aggregate`, then ``Aggregate
(**a.spec)`` creates a new copy. ``spec_ex`` appends meta-information to
``spec`` about hyper-parameters.

.. ipython:: python
    :okwarning:

    from pprint import pprint
    pprint(a05n.spec)

.. _10 min decl program:

The DecL Program
~~~~~~~~~~~~~~~~~~

The ``program`` property returns the DecL program used to create the object.
It is blank if the object was not created using DecL. The helper function :func:`pprint_ex` pretty prints a program.

.. ipython:: python
    :okwarning:

    from aggregate import pprint_ex
    pprint_ex(a05n.program, split=20)
    pprint_ex(p07.program, split=20)

.. _10 min update:

The :meth:`update` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~

After an :class:`Aggregate` or a :class:`Portfolio` object has been created it needs to be updated to populate its ``density_df`` dataframe. :meth:`build` automatically updates the objects it creates with default hyper-parameter values. Sometimes it is necessary to re-update with different hyper-parameters. The :meth:`update` method takes arguments ``log2=13``, ``bs=0``, and ``recommend_p=0.999``. The first two control the number and size of buckets. When ``bs==0`` it is estimated using the method :meth:`recommend_bucket`. If ``bs!=0`` then ``recommend_p`` is ignored.

Further control over updating is available, as described in REF.


.. _10 min stats funs:

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~~~

:class:`Aggregate` and :class:`Portfolio` objects include basic mean, CV, standard deviation, variance, and skewness statistics as attributes. Those prefixed ``agg`` are based on exact calculations:

* ``agg_m``, ``agg_cv``, ``agg_sd``, ``agg_var``, and ``agg_skew``

and prefixed ``est`` are based on the estimated numerical statistics:

* ``est_m``, ``est_cv``, ``est_sd``, ``est_var``, and ``est_skew``.

In addition, :class:`Aggregate` has similar series prefixed ``sev`` and
``est_sev`` for the exact and estimated numerical severity. These attributes
are just conveniences; they are all available in (or derivable from)
``report_df``.

:class:`Aggregate` and :class:`Portfolio` objects act like ``scipy.stats`` (continuous) frozen random variable objects and include the following statistical functions.

* :meth:`pmf` the probability mass function
* :meth:`pdf` the probability density function---broadly interpreted---defined as the pmf divided by ``bs``
* :meth:`cdf` the cumulative distribution function
* :meth:`sf` the survival function
* :meth:`q` the quantile function (left inverse cdf), also known as value at risk
* :meth:`tvar` tail value at risk function
* :meth:`var_dict` a dictionary of tail statistics by unit and in total

We aren't picky about whether the density is technically a density when the aggregate is actually mixed or discrete.
The discrete output (``density_df.p_*``) is interpreted as the distribution, so none of the statistical functions is interpolated.
For example:

.. ipython:: python
    :okwarning:

    qd(a05g.pmf(2), a05g.pmf(2.2), a05g.pmf(3), a05g.cdf(2), a05g.cdf(2.2))
    print(1 - a05g.cdf(2), a05g.sf(2))
    print(a05g.q(a05g.cdf(2)))

The last line illustrates that :meth:`q` and :meth:`cdf` are inverses. The :meth:`var_dict` function computes tail statistics for all units, return in a dictionary.

.. ipython:: python
    :okwarning:

    p07.var_dict(0.99), p07.var_dict(0.99, kind='tvar')

.. _10 min plot:

The :meth:`plot` Method
~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`plot` method provides basic visualization. There are three plots: the pdf/pmf for severity and the aggregate on the left. The middle plot shows log density for continuous distributions and the distribution function for discrete ones (selected when ``bs==1`` and the mean is < 100). The right plot shows the quantile (or VaR or Lee) plot.

The reinsurance examples below show the discrete output format. The plots show the
gross, net of occurrence, and net severity and aggregate pmf (left) and cdf
(middle), and the quantile (Lee) plot (right). The property ``a05g.figure``
returns the last figure made by the object as a convenience. You could also
use :meth:`plt.gcf`.

.. ipython:: python
    :okwarning:

    a05g.plot()
    @savefig 10min_gross.png
    a05g.figure.suptitle('Gross - discrete format');

    a05no.plot()
    @savefig 10min_no.png
    a05no.figure.suptitle('Net of occurrence');

    a05n.plot()
    @savefig 10min_noa.png
    a05n.figure.suptitle('Net of occurrence and aggregate');


Continuous distributions substitute the log density for the distribution in the middle.

.. ipython:: python
    :okwarning:

    a03.plot()
    @savefig 10min_cts.png
    a03.figure.suptitle('Continuous format');


A :class:`Portfolio` object plots the density and log density of each unit and
the total.

.. ipython:: python
    :okwarning:

    p07.plot()
    @savefig 10min_p07.png scale=20
    p07.figure.suptitle('Portfolio plot');

.. _10 min price:

The :meth:`price` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`price` method computes the risk adjusted expected value (technical price net of expenses) of losses limited by capital at a specified VaR threshold.  Suppose the 99.9%ile outcome is used to specify regulatory assets :math:`a`.

.. ipython:: python
    :okwarning:

    qd(a03.q(0.999))

The risk adjustment is specified by a spectral risk measure corresponding to an input distortion. Distortions can be built using DecL, see :ref:`10 min Distortion`. :meth:`price` applies to :math:`X\wedge a`.
It returns expected limited losses ``L``, the risk adjusted premium ``P``, the margin ``M = P - L``, the capital ``Q = a - P``, the loss ratio, leverage as premium to capital ``PQ``, and return on capital ``ROE``.

.. ipython:: python
    :okwarning:

    qd(a03.price(0.999, d06).T)


When :meth:`price` is applied to a :class:`Portfolio`, it returns the total premium and its (lifted) natural allocation to each unit, see PIR Chapter 14, along with all the other statistics in a dataframe. Losses are allocated by equal priority in default.

.. ipython:: python
    :okwarning:

    qd(p07.price(0.999, d06).df.T)

The ROE varies by unit, reflecting different consumption and cost of capital by layer. The less risky unit A runs at a higher loss ratio (cheaper insurance) but higher ROE than unit B because it consumes more expensive, equity-like lower layer capital but less capital overall (higher leverage).

.. _10 min snap:

The :meth:`snap` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`snap` rounds an input number to the index of ``density_df``. It selects the nearest element.

.. _10 min approx:

The :meth:`approximate` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`approximate` method creates an analytic approximation fit using moment matching. Normal, lognormal, gamma, shifted lognormal, and shifted gamma distributions can be fit, the last two requiring three moments. To fit all five and return a dictionary call with argument ``"all"``.

.. ipython:: python
    :okwarning:

    fzs = a03.approximate('all')
    d = pd.DataFrame({k: fz.stats('mvs') for k, fz in fzs.items()},
             index=pd.Index(['mean', 'var', 'skew'], name='stat'),
                    dtype=float)
    qd(d)


.. _10 min additional:

Additional :class:`Portfolio` Methods
---------------------------------------

.. other stuff to consider
   * stand alone pricing
   * merton perold
   * gradient
   * calibrate distortion(s)
   * apply_distortion(s)
   * analyze_distortion(s)

.. _10 min conditional expected values:

Conditional Expected Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`Portfolio` object's ``density_df`` includes a slew of values to allocate capital (please don't) or margin (please do). These all rely on what :cite:t:`Mildenhall2022a` call the :math:`\kappa` function, defined for a sum :math:`X=\sum_i X_i` as the conditional expectation

.. math::

    \kappa_i(x) = \mathsf E[X_i\mid X=x].

Notice that :math:`\sum_i \kappa_i(x)=x`, hinting at its allocation application.
See PIR Chapter 14.3 for an explanation of why :math:`\kappa` is so useful. In short, it shows which units contribute to bad overall outcomes. It is in ``density_df`` as the columns ``exeqa_unit``, read as the "expected value given X eq(uals) a".

Here are some :math:`\kappa` values and graph for ``p07``. Looking the log density plot on the right shows that unit B dominates for moderately large events, but Cat dominates for the largest events.

.. ipython:: python
    :okwarning:

    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45)); \
    ax0, ax1 = axs.flat; \
    lm = [-1000, 65000]; \
    bit = p07.density_df.filter(regex='exeqa_[ABCt]').rename(
        columns=lambda x: x.replace('exeqa_', '')).sort_index(axis=1); \
    bit.index.name = 'Loss'; \
    bit.plot(xlim=lm, ylim=lm, ax=ax0); \
    ax0.set(title=r'$E[X_i\mid X]$', aspect='equal'); \
    ax0.axhline(bit['B'].max(), lw=.5, c='C7');
    p07.density_df.filter(regex='p_[ABCt]').rename(
        columns=lambda x: x.replace('p_', '')).plot(ax=ax1, xlim=lm, logy=True);
    @savefig 10mins_exa.png scale=20
    ax1.set(title='Log density');
    bit['Pct A'] = bit['A'] / bit.index
    qd(bit.loc[:lm[1]:1024])

The thin horizontal line at the maximum value of ``exeqa_B`` (left plot) shows that :math:`\kappa_B` is not increasing. Unit B contributes more to moderately bad outcomes than Cat, but in the tail Cat dominates.

Using ``filter(regex=...)`` to select columns from ``density_df`` is a helpful idiom. The total column is labeled ``_total``. Using upper case for unit names makes them easier to select.

.. _10 min calibrate distortions:

Calibrate Distortions
~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`calibrate_distortions` method calibrates distortions to achieve requested pricing for the total loss. Pricing can be requested by loss ratio or return on capital (ROE). Asset levels can be specified in monetary terms, or as a probability of non-exceedance. To calibrate the usual suspects (constant cost of capital, proportional hazard, dual, Wang, and TVaR) to achieve a 15% return with a 99.6% capital level run:

.. ipython:: python
    :okwarning:

    p07.calibrate_distortions(Ps=[0.996], ROEs=[0.15], strict='ordered');
    qd(p07.distortion_df)
    pprint(p07.dists)

The answer is returned in the ``dist_ans`` dataframe. The requested distortions are all single parameter, returned in the ``param`` column. The last column gives the error in achieved premium. The attribute ``p07.dists`` is a dictionary with keys distortion types and values :class:`Distortion` objects. See PIR REF for more discussion.

.. _10 min analyze distortions:

Analyze Distortions
~~~~~~~~~~~~~~~~~~~~

The :meth:`analyze_distortions` method applies the distortions in ``p07.dists`` at a given capital level and summarizes the implied (lifted) natural allocations across units. Optionally, it applies a number of traditional (bullshit) pricing methods. The answer dataframe includes premium, margin, expected loss, return, loss ratio and leverage statistics for each unit and method. Here is a snippet, again at the 99.6% capital level.


.. ipython:: python
    :okwarning:

    ans = p07.analyze_distortions(p=0.996)
    print(ans.comp_df.xs('LR', axis=0, level=1).
         to_string(float_format=lambda x: f'{x:.1%}'))

.. _10 min twelve plot:

Twelve Plot
~~~~~~~~~~~~~

The :meth:`twelve_plot` method produces a detailed analysis of the behavior of a two unit portfolio. To run it, build the portfolio and calibrate some distortions. Then apply one of the distortions (to compute an augmented version of ``density_df`` with pricing information). We give two examples.

First, the case of a thin-tailed and a thick-tailed unit. Here, the thick tailed line benefits from pooling at low capital levels, resulting in negative margins to the thin-tail line in compensation. At moderate to high capital levels the total margin for both lines is positive. Assets are 12.5.  The argument ``efficient=False`` in :meth:`apply_distortion` includes extra columns in ``density_df`` that are needed to compute the plot.


.. ipython:: python
    :okwarning:

    p09 = build('port TenM:09 '
                  'agg X1 1 claim sev gamma 1 cv 0.25 fixed '
                  'agg X2 1 claim sev 0.7 * lognorm 1 cv 1.25 + 0.3 fixed'
                 , bs=1/1024)
    qd(p09)
    print(f'Asset P value {p09.cdf(12.5):.5g}')
    p09.calibrate_distortions(ROEs=[0.1], As=[12.5], strict='ordered');
    qd(p09.distortion_df)
    p09.apply_distortion('dual', efficient=False);
    fig, axs = plt.subplots(4, 3, figsize=(3 * 3.5, 4 * 2.45), constrained_layout=True)
    @savefig 10mins_twelve_p09.png
    p09.twelve_plot(fig, axs, p=0.999, p2=0.999)


There is a lot of information here. We refer to the charts as
:math:`(r,c)` for row :math:`r=1,2,3,4` and column :math:`c=1,2,3`,
starting at the top left. The horizontal axis shows the asset level in
all charts except :math:`(3,3)` and :math:`(4,3)`, where it shows
probability, and :math:`(1,3)` where it shows loss. Blue represents the
thin tailed unit, orange thick tailed and green total. When both dashed
and solid lines appear on the same plot, the solid represent
risk-adjusted and dashed represent non-risk-adjusted functions. Here is
the key.

-  :math:`(1,1)` shows density for :math:`X_1, X_2` and
   :math:`X=X_1+X_2`; the two units are independent. Both units have
   mean 1.

-  :math:`(1,2)`: log density; comparing tail thickness.

-  :math:`(1,3)`: the bivariate log-density. This plot illustrates where
   :math:`(X_1, X_2)` *lives*. The diagonal lines show :math:`X=k` for
   different :math:`k`. These show that large values of :math:`X`
   correspond to large values of :math:`X_2`, with :math:`X_1` about
   average.

-  :math:`(2,1)`: the form of :math:`\kappa_i` is clear from looking at
   :math:`(1,3)`. :math:`\kappa_1` peaks above 1.0 around :math:`x=2` and hereafter it declines to 1.0. :math:`\kappa_2` is
   monotonically increasing.

-  :math:`(2,2)`: The :math:`\alpha_i` functions. For small :math:`x`
   the expected proportion of losses is approximately 50/50, since the
   means are equal. As :math:`x` increases :math:`X_2` dominates. The
   two functions sum to 1.

-  :math:`(2,3)`: The thicker lines are :math:`\beta_i` and the thinner
   lines :math:`\alpha_i` from :math:`(2,2)`. Since :math:`\alpha_1`
   decreases :math:`\beta_1(x)\le \alpha_1(x)`. This can lead to
   :math:`X_1` having a negative margin in low asset layers. :math:`X_2`
   is the opposite.

-  :math:`(3,1)`: illustrates premium and margin determination by asset
   layer for :math:`X_1`. For low asset layers
   :math:`\alpha_1(x) S(x)>\beta_1(x) g(S(x))` (dashed above solid)
   corresponding to a negative margin. Beyond about :math:`x=1.38` the
   lines cross and the margin is positive.

-  :math:`(4,1)`: shows the same thing for :math:`X_2`. Since
   :math:`\alpha_2` is increasing, :math:`\beta_2(x)>\alpha_2(x)` for
   all :math:`x` and so all layers get a positive margin. The solid line
   :math:`\beta_2 gS` is above the dashed :math:`\alpha_2 S` line.

-  :math:`(3,2)`: the layer margin densities. For low asset layers
   premium is fully funded by loss with zero overall margin. :math:`X_2`
   requires a positive margin and :math:`X_1` a negative one, reflecting
   the benefit the thick unit receives from pooling in low layers. The
   overall margin is always non-negative. Beyond about :math:`x=1.5`,
   :math:`X_1`\ ’s margin is also positive.

-  :math:`(4,2)`: the cumulative margin in premium by asset level. Total
   margin is zero in low *dollar-swapping* layers and then increases. It
   is always non-negative. The curves in this plot are the integrals of
   those in :math:`(3,2)` from 0 to :math:`x`.

-  :math:`(3,3)`: shows stand-alone loss :math:`(1-S(x),x)=(p,q(p))`
   (dashed) and premium :math:`(1-g(S(x)),x)=(p,q(1-g^{-1}(1-p))`
   (solid, shifted left) for each unit and total. The margin is the
   shaded area between the two. Each set of three lines (solid or
   dashed) does not add up vertically because of diversification. The
   same distortion :math:`g` is applied to each unit to the stand-alone
   :math:`S_{X_i}`. It is calibrated to produce a 10 percent return
   overall. On a stand-alone basis, calculating capital by unit to the
   same return period as total, :math:`X_1` is priced to a 77.7 percent
   loss ratio and :math:`X_2` 52.5 percent, producing an average 62.7
   percent, vs. 67.6 percent on a combined basis. Returns are 37.5
   percent and 9.4 percent respectively, averaging 11.5 percent, vs 10
   percent on a combined basis, see stand-alone analysis below.

-  :math:`(4,3)`: shows the natural allocation of loss and premium to
   each unit. The total (green) is the same as :math:`(3,3)`. For each
   :math:`i`, dashed shows :math:`(p, \mathsf E[X_i\mid X=q(p)])`, i.e. the
   expected loss recovery conditioned on total losses :math:`X=q(p)`,
   and solid shows :math:`(p, \mathsf E[X_i\mid X=q(1-g^{-1}(1-p))])`, i.e. the
   natural premium allocation.
   Here the solid and dashed lines *add up* vertically: they are
   allocations of the total. Looking vertically above :math:`p`, the
   shaded areas show how the total margin at that loss level is
   allocated between lines. :math:`X_1` mostly consumes assets at low
   layers, and the blue area is thicker for small :math:`p`,
   corresponding to smaller total losses. For :math:`p` close to 1,
   large total losses, margin is dominated by :math:`X_2` and in fact
   :math:`X_1` gets a slight credit (dashed above solid). The change in
   shape of the shaded margin area for :math:`X_1` is particularly
   evident: it shows :math:`X_1` benefits from pooling and requires a
   lower overall margin.

There may appear to be a contradiction between figures :math:`(3,2)` and
:math:`(4,3)` but it should be noted that a particular :math:`p` value
in :math:`(4,3)` refers to different events on the dotted and solid
lines.

Plots :math:`(3,3)` and :math:`(4,3)` explain why the thick unit
requires relatively more margin: its shape
does not change when it is pooled with :math:`X_1`. In :math:`(3,3)` the
green shaded area is essentially an upwards shift of the orange, and the
orange areas in :math:`(3,3)` and :math:`(3,4)` are essentially the
same. This means that adding :math:`X_1` has virtually no impact on the
shape of :math:`X_2`; it is like adding a constant. This can also be
seen in :math:`(4,3)` where the blue region is almost a straight line.

Applying the same distortion on a stand-alone basis produces:

.. ipython:: python
    :okwarning:

    a = p09.stand_alone_pricing(p09.dists['dual'], p=p09.cdf(12.5))
    print(a.iloc[:8])

The lifted natural allocation (diversified pricing) is given next. These numbers
are so different than the stand-alone because X2 has to compensate X1 for the
transfer of wealth in default states. When there is a large loss, it is caused
by X2 and so X2 receives a disproportionate share of the assets in default.

.. ipython:: python
    :okwarning:

    a2 = p09.analyze_distortion('dual', ROE=0.1, p=p09.cdf(12.5))
    print(a2.pricing.unstack(1).droplevel(0, axis=0).T)

The second portfolio has been selected with two thick tailed units. A appears riskier at lower return periods and B at higher. Pricing is calibrated to a 15% ROE at a 99.6% capital level.


.. ipython:: python
    :okwarning:

    p10 = build('port TenM:10 '
                 'agg A '
                     '30 claims '
                     '1000 xs 0 '
                     'sev gamma 25 cv 1.5 '
                     'mixed delaporte 0.75 0.6 '
                 'agg B '
                     '5 claims '
                     '20000 x 20 '
                     'sev lognorm 25 cv 3.0 '
                     'poisson'
                , bs=1)
    qd(p10)
    p10.calibrate_distortions(ROEs=[0.15], Ps=[0.996], strict='ordered');
    qd(p10.distortion_df)

Apply the dual distortion and then create the twelve plot.

.. ipython:: python
    :okwarning:

    p10.apply_distortion('dual', efficient=False);
    fig, axs = plt.subplots(4, 3, figsize=(3 * 3.5, 4 * 2.45), constrained_layout=True)
    @savefig 10min_twelve_plot.png
    p10.twelve_plot(fig, axs, p=0.999995, p2=0.999999)


Applying the same distortion on a stand-alone basis produces:

.. ipython:: python
    :okwarning:

    assets = p10.q(0.996)
    a = p10.stand_alone_pricing(p10.dists['dual'], p=p10.cdf(assets))
    print(a.iloc[:8])

The lifted natural allocation (diversified pricing) is given next.

.. ipython:: python
    :okwarning:

    a2 = p10.analyze_distortion('dual', ROE=0.1, p=p10.cdf(assets))
    print(a2.pricing.unstack(1).droplevel(0, axis=0).T)


.. _10 min extensions:

Extensions
-----------

The ``extensions`` sub-package contains additional classes and functions that are either peripheral to the main project or still under development (and subject to change). Currently, ``extensions`` includes:

* ``case_studies`` for creating and managing PIR case studies (see :doc:`2_x_case_studies`).
* ``pir_figures`` for creating various exhibits and figures in PIR.
* ``figures`` for creating various other exhibits and figures.
* ``samples`` includes functions for working with samples and executing a switcheroo. Eventually, these will be integrated into :class:`Portfolio`.

.. test suite is dead...

.. _10 min summary:

Summary of Objects Created by DecL
-------------------------------------

Each of the objects created by :meth:`build` is automatically stored in the knowledge. We can list them out now.

.. ipython:: python
    :okwarning:

    from aggregate import pprint_ex
    for n, r in build.qshow('^TenM:').iterrows():
        pprint_ex(r.program, split=20)


.. ipython:: python
    :suppress:

    plt.close('all')

