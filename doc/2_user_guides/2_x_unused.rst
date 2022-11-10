
Material Not Used
==================



Main Features
-------------

- Human readable input with the simple ``agg`` language
- Built in library of insurance severity curves for both catastrophe and non
  catastrophe lines
- Built in parameterization for most major lines of insurance in the US, making it
  easy to build a "toy company" based on market share by line
- Clear distinction between catastrophe and non-catastrohpe lines
- Use of Fast Fourier Transforms throughout differentiates ``aggregate`` from
  tools based on simulation
- Fast, accurate - no simulations!
- Graphics and summaries using ``pandas`` and ``matplotlib``
- Outputs in ``pandas`` dataframes



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
- Detailed creation of marginal loss distributions that can then be sampled and used by other simulation software, e.g. to incorporate dependence structures, or in situations where it is necessary to track individual events, e.g. to compute gross, ceded and net bi- and trivariate distributions.




Aggregate and Portfolio object rationalization
----------------------------------------------

Common Features in Aggregate and Portfolio classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+-----------+-----------+-------------+--------------------------+
| Function         | Aggregate | Portfolio | Underwriter |                          |
+==================+===========+===========+=============+==========================+
| audit_df         |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| cdf              |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| density_df       |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| describe         |           |           | x!          | no styling; used in code |
+------------------+-----------+-----------+-------------+--------------------------+
| pdf              |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| pmf              |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| plot             |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| price            |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| q                |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| recommend_bucket |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| report_df        |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| sf               |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| snap             |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| statistics       |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| tvar             |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| update           |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| var_dict         |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
|                  |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| spec             |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| spec_ex          |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| program          |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| list()           |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+
| describe()       |           |           |             |                          |
+------------------+-----------+-----------+-------------+--------------------------+

build(‘US.Hurricane’) –> object vs build[‘US.Hurricane’] –> answer

DataFrame elements
~~~~~~~~~~~~~~~~~~

+-----------+-----------------------+-------------------+-------------+
| Function  | Aggregate             | Portfolio         | Notes       |
+===========+=======================+===================+=============+
| audit_df  | rows = mix components | rows = lines      | Same        |
|           | or total              |                   | fun         |
|           |                       |                   | ctionality, |
|           |                       |                   | but         |
+-----------+-----------------------+-------------------+-------------+
|           | cols = fsa monents;   | cols = mcvsk; emp | different   |
|           | emp fsa; 123, mcvsk   | mcvsk; errors, Ps | index names |
+-----------+-----------------------+-------------------+-------------+
|           |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
| audit     | n/a                   | renamed version   | dropped Oct |
| (         |                       | of audit_df       | 2022        |
| property) |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           |                       | renamer and line  |             |
|           |                       | renamer           |             |
+-----------+-----------------------+-------------------+-------------+
|           |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
| d         | loss, p=p_total,      |                   |             |
| ensity_df | p_sev, log_p,         |                   |             |
|           | log_p_sev             |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           | F, F_sev, S, S_sev,   |                   |             |
|           | lev=exa, exlea, e,    |                   |             |
|           | epd                   |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           | exgta, exeqa=loss     |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
| density   | n/a                   | renamed           | dropped Oct |
| (         |                       | density_df        | 2022        |
| property) |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
| describe  | total only; fsa       | concatenates      |             |
| (         | mcvsk, errors         | agg.describes()   |             |
| property) |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           | used by html method   | adds rows for     |             |
|           |                       | total             |             |
+-----------+-----------------------+-------------------+-------------+
|           | rows fsa; cols stats  |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
| s         | all theoretical       |                   |             |
| tatistics |                       |                   |             |
| sets      |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           | EX1, 2, 3; mean, cv,  |                   |             |
|           | skew                  |                   |             |
+-----------+-----------------------+-------------------+-------------+
| stat      | by mixture component  | rows are stats    |             |
| istics_df |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           | freq_i naming         | multiindex naming |             |
+-----------+-----------------------+-------------------+-------------+
| s         | mixed and independent | n/a               |             |
| tatistics |                       |                   |             |
| _total_df |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           | same rows             |                   |             |
+-----------+-----------------------+-------------------+-------------+
| s         | concat of df and      | identical with    | Adjusted    |
| tatistics | total_df              | \_df              | Agg to have |
| (         |                       |                   | same        |
| property) |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           | returns transpose     |                   | index       |
+-----------+-----------------------+-------------------+-------------+
|           |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
| report_df | stats moments, limit, | similar, except   |             |
|           | attach                | emps and errs     |             |
+-----------+-----------------------+-------------------+-------------+
|           | by line, indep,       |                   |             |
|           | mixed; empirical and  |                   |             |
|           | error                 |                   |             |
+-----------+-----------------------+-------------------+-------------+
| r         | just mixed total col  |                   |             |
| eport_ser | from report_df        |                   |             |
| (Series)  |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
|           |                       |                   |             |
+-----------+-----------------------+-------------------+-------------+
| report    | report_df renamed not | same              | dropped Oct |
|           | styled                |                   | 2022        |
+-----------+-----------------------+-------------------+-------------+





Simple Discrete Aggregate Distributions
---------------------------------------

Aggregate Frequency and Severity Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulation algorithm for insurance losses

::

       for i = 1 to 10000
           agg = 0
           simulate number of events n
           for j = 1 to n
               simulate loss amount X
               agg = agg + X
           output agg for simulation i

-  Write :math:`A = X_1 + \cdots X_N`, :math:`X_i` and :math:`N` random
   and independent, and :math:`X_i` identically distributed
-  Model insured losses via number of claims :math:`N` the **frequency**
   and the amount :math:`X_i` of each claim, the **severity**
