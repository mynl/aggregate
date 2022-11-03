===============
6. Development
===============

The discussion section explains the *why* of design.

* About aggregate distributions
* Basic statistics of aggregate distributions
* Examples
* Computing quantiles
* auditing quantiles, Awkward discreet examples
* Purpose of the underwriter module
* Aggregates versus portfolios
* Pricing methodologies inspector missed messages



Main Features
-------------

Here are a few of the things that ``aggregate`` does well:

- Human readable input with the simple ``agg`` language
- Built in library of insurance severity curves for both catastrophe and non
  catastrophe lines
- Built in parameterization for most major lines of insurance in the US, making it
  easy to build a "toy company" based on market share by line
- Clear distinction between catastrophe and non-catastrohpe lines
- Use of Fast Fourier Transforms throughout differentiates ``aggregate`` from
  tools based on simulation
- Fast, accurate - no simulations!
- Graphics and summaries following Pandas and Matplotlib syntax
- Outputs in easy-to-manipulate Pandas dataframes

For example, to specify an aggregate distribution based on 50 claims from a lognormal
distribution with a CV of 2.0, a mean of 1000 and a Poisson frequency distribution
and plot the resulting severity and aggregate distributions enter:

::

  import aggregate as agg
  uw = agg.Underwriter()
  port = uw('agg MyAgg 50 claims sev lognorm 1000 cv 2')
  port.plot()


MyAgg is a label for the aggregate.

To create a more complex portfolio with catastrophe and non catastrophe losses:


::

  port = uw('''port MyPortfolio
    agg nonCat 10 claims 100 x 0 sev lognorm 1000 cv 2 mixed gamma 0.4
    agg cat     2 claims 1000 x 0 sev 1000 * pareto 1.8 - 1000 poisson
  ''')



Potential Applications
----------------------

- Education
     * Building intuition around how loss distribtions convolve
     * Convergence to the central limit theorem
     * Generalized distributions
     * Compound Poisson distributions
     * Mixed distributiuons
     * Tail behavior based on frequency or severity tail
     * Log concavity properties
     * Uniform, triangular to normal
     * Bernoulli to normal = life insurance
     * $P(A>x)\sim \lambda P(X>x) \sim P(M>x)$ if thick tails
     * Occ vs agg PMLs, body vs. tail. For non-cat lines it is all about correlation; for cat it is all about the tail
     * Effron's theorem
     * FFT exact for "making" Poisson, sum of normals is normal, expnentials is gamma etc.
     * Slow convergence of truncated stable to normal
     * Severity doesn't matter: difference between agg with sev and without for large claim count and stable severity
     * Small large claim split approach...attrit for small; handling without correlation??
     * Compound Poisson: CP(mixed sev) = sum CP(sev0
- Pricing small insurance portfolios on a claim by claim basis
- Analysis of default probabilities
- Allocation of capital and risk charges
- Detailed creation of marginal loss distributions that can then be
  sampled and used by other simulation software, e.g. to incorporate
  dependence structures, or in situations where it is necessary to
  track individual events, e.g. to compute gross, ceded and net bi-
  and trivariate distributions.


Practical Modeling Examples
---------------------------

* From limit profile
* Mixed severity
* Modeling $N\mid N \ge n$
* How to model 2 reinstatements

Missing Features
----------------

Here are some important things that ``aggregate`` does **not** do:

- It is strictly univariate. It is impossible to model bivariate or multivariate distributions.
  As a result ``aggregate`` is fast and accurate
- ``aggregate`` can model correlation between variables using shared mixing variables. This
  is adequate to build realistic distributions but would not be adequate for an industrial-
  strength insurance company model.


History and Applications
========================

History
-------

I have built several iterations of software to work with aggregate distributions since the first in 1997.

*  A Matlab version for CNA Re with a graphical interface. Computed aggregates by business unit and the portfolio total. Used to discover the shocking fact there was only a 53 percent chance of achieving plan...which is obvious in hindsight but was a surprise at the time.
*  A C++ version of the Matlab code called SADCo in 1997-99. This code sits behing [MALT](http://www.mynl.com/MALT/home.html).
*  Another C++ version with an implementation of the Iman Conover method to combine aggregates with correlation using the (gasp) normal copula, [SCARE](http://www.mynl.com/wp/default.html)
*  At Aon Re I worked on their simulation based tools called ALG (Aggregate Loss Generator) which simulated losses and Prime/Re which manipulated the simulations and applied various reinsurance structures. ALG used a shared mixing variables approach to correlation.
*  At Aon Re I also built related tools
	-  The Levy measure maker
	-  A simple approach to multi-year modeling based on re-scaling a base year, convolving using FFTs and tracking (and stopping) in default scenarios
*  At Aon Benfield I was involved with [ReMetric](http://www.aon.com/reinsurance/analytics-(1)/remetrica.jsp), a very sophisticated, general purpose DFA/ERM simulation tool,


Reinsurance Pricing Applications
--------------------------------

*  Excess of loss exposure rating
*  Creation of a severity curve from a limit profile

Insurance Pricing Applications
------------------------------

*  Large accounts: insurance savings and charge for WC
*  Specific and aggregate covers


Capital Modeling
----------------

*  Portfolio level probability of default, EPD, Var and TVaR statistics

Capital Allocation and Pricing
------------------------------

*  Many and varied
*  Application of distortion risk measures
*  ...



Design and Build
----------------

* Design: abstracting the business problem.
    -  Getting the right model for your problem is key.
    -  What is the problem domain? What are the principle use cases? How will the software actually be used? What is input vs. derived? What is constant vs. an account specific parameter? What is the best way to express the inputs? To view the outputs? How do you bootstrap, using simpler functionality to implement more complex? What are those key simple capabilities?

* Implementation I: mapping design to software, i.e. coding. The joy of objects.
* Implementation II: wonderful, free tools available today and the whole shareware infrastructure. I am working in Python using Jupyter, pyCharm (not quite free) and Sphinx for documentation. These are fantastic tools that make many things easy. ​People should know about the capabilities. E.g. here is the documentation automatically produced from the source code: http://www.mynl.com/aggregate/index.html plus a link to the current code on Github (which is alpha stage, i.e. not even beta yet; do not bother downloading!)

* Use and Lessons
    -  Educational lessons: convergence to the central limit theorem, mixtures vs. convolution, thick vs thin tail distributions, occurrence vs. aggregate PMLs and many more
    -  Capital allocation and distortion risk measures. I am working on several papers here, including one following from the sessions at the Spring meeting with Mango and Major. The software will be used to create all the examples. The source for the examples will be on-line so folks can try themselves....leading to...

* DIY
    -  How you can download and use the tools yourself. Some starter lessons.


Development Outline
====================

Non programming Enhancements
----------------------------
* Better sample of realistic severity curves
* Better sample of by line aggregate Blocks in agg format
* Credit modeling: what is distortion implied by bond credit curve? By cat bond pricing?
* Jon Evans note and severity
* Jed note

Short term
-----------
* Different freq dists and freq dist in exact mode, shape a, b
* Fix test cases!!
* issue with mass and bodoff1 portfolio
* Distortion that is the P/L convex envelope of a set of given points
* Errors with mass! Finite vs infinite supported distributions, lep vs ly and clin?!
* Understand output for collateral and priority!
* Output Levy measure
* Funky objects from JacodS? Simple jump examples

Medium Term
------------
* Estimate Bucket function! Auto update
* Style reports
* An about me report; better str and repr methods
* More consistent and informative reports and plots (e.g. include severity match in agg)
* Convex Hull distortion built from pricing
* Delete items easily from the database
* Save / load from non-YAML, persist the database; dict to agg language converter? Get rid of YAML dependence
* Using agg as a severity (how!)
* Name as a member in dict vs list conniptions (put up with duplication?)


Nice to have enhancements
-------------------------
* Agg limit and attachment: NO already have when you can use an agg as a severity
* How to model two reinstatements?
* $N\mid N \ge n$ distribution?



Underwriter Class
=================

(*from underwriter module*)

The Underwriter is an easy to use interface into the computational functionality of aggregate.

The Underwriter
---------------

* Maintains a default library of severity curves
* Maintains a default library of aggregate distributions corresponding to industry losses in
  major classes of business, total catastrophe losses from major perils, and other useful constructs
* Maintains a default library of portfolios, including several example instances and examples used in
  papers on risk theory (e.g. the Bodoff examples)


The library functions can be listed using

::

        uw.list()

or, for more detail

::

        uw.describe()

A given example can be inspected using ``uw['cmp']`` which returns the defintion of the database
object cmp (an aggregate representing industry losses from the line Commercial Multiperil). It can
be created as an Aggregate class using ``ag = uw('cmp')``. The Aggregate class can then be updated,
plotted and various reports run on it. In iPython or Jupyter ``ag`` returns an informative HTML
description.

The real power of Underwriter is access to the agg scripting language (see parser module). The scripting
language allows severities, aggregates and portfolios to be created using more-or-less natural language.
For example

::

        pf = uw('''
        port MyCompanyBook
            agg LineA 100 claims 100000 xs 0 sev lognorm 30000 cv 1.25
            agg LineB 150 claims 250000 xs 5000 sev lognorm 50000 cv 0.9
            agg Cat 2 claims 100000000 xs 0 sev 500000 * pareto 1.8 - 500000
        ''')

creates a portfolio with three sublines, LineA, LineB and Cat. LineA is 100 (expected) claims, each pulled
from a lognormal distribution with mean of 30000 and coefficient of variation 1.25 within the layer
100000 xs 0 (i.e. limited at 100000). The frequency distribution is Poisson. LineB is similar. Cat is jsut
2 claims from the indicated limit, with severity given by a Pareto distribution with shape parameter 1.8,
scale 500000, shifted left by 500000. This corresponds to the usual Pareto with survival function
S(x) = (lambda / (lambda + x))^1.8, x >= 0.

The portfolio can be approximated using FFTs to convolve the aggregates and add the lines. The severities
are first discretized using a certain bucket-size (bs). The port object has a port.recommend_bucket() to
suggest reasonable buckets:

>> pf.recommend_bucket()

+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
|       | bs10    | bs11   | bs12   | bs13   | bs14  | bs15  | bs16  | bs18 | bs20 |
+=======+=========+========+========+========+=======+=======+=======+======+======+
| LineA | 3,903   | 1,951  | 976    | 488    | 244   | 122   | 61.0  | 15.2 | 3.8  |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
| LineB | 8,983   | 4,491  | 2,245  | 1,122  | 561   | 280   | 140   | 35.1 | 8.8  |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
| Cat   | 97,656  | 48,828 | 24,414 | 12,207 | 6,103 | 3,051 | 1,525 | 381  | 95.4 |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+
| total | 110,543 | 55,271 | 27,635 | 13,817 | 6,908 | 3,454 | 1,727 | 431  | 108  |
+-------+---------+--------+--------+--------+-------+-------+-------+------+------+

The column bsNcorrespond to discretizing with 2**N buckets. The rows show suggested bucket sizes for each
line and in total. For example with N=13 (i.e. 8196 buckets) the suggestion is 13817. It is best the bucket
size is a divisor of any limits or attachment points, so we select 10000.

Updating can then be run as

::

    bs = 10000
    pf.update(13, bs)
    pf.report('quick')
    pf.plot('density')
    pf.plot('density', logy=True)
    print(pf)

    Portfolio name           MyCompanyBook
    Theoretic expected loss     10,684,541.2
    Actual expected loss        10,657,381.1
    Error                          -0.002542
    Discretization size                   13
    Bucket size                     10000.00
    <aggregate.port.Portfolio object at 0x0000023950683CF8>


Etc. etc.

"""
