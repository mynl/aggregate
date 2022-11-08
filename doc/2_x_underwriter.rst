.. _2_x_underwriter:

======================
The Underwriter Class
======================


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

