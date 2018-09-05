Development Outline
====================

September 3
-----------

Non programming Enhancements
----------------------------
* Better sample of realistic severity curves
* Credit modeling: what is distortion implied by bond credit curve? By cat bond pricing?

Short term
-----------
* Distortion that is the P/L convex envelope of a set of given points
* Errors with mass! Finite vs infinite supported distributions, lep vs ly and clin?!
* Understand output for collateral and priority!
* Sev by name from examples in spec sev_a = name
* Different freq dists and freq dist in exact mode
* Integrate beta factory: Kent example should be built in: dist=beta, mean=, cv= (catch in shape from mean,cv) limit=??

Medium Term
------------
* Estimate Bucket function! Auto update
* Major paper examples? Normal with correlation
* IME review paper example?

Nice to have enhancements
-------------------------
* pf += to add a line?
    - x = Portfolio(name)
    - x.add(line='home', premium=1200, lr=.5)?!
* different ways of specfiying? .add method? add with (freq=, sev=) style
* More freq dists and shape scale specification
* occ and agg limit and attachment
* Label severity distributions to facilitate adding (user warrants labels are unique)
* Better + function combining severity distributions
* How to model two reinstatements?
* $N\mid N \ge n$ distribution?
* Split into subfiles and make a proper package
* sort beta factory

Educational Opportunities

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

## Practical Modeling Examples
* From limit profile
* Mixed severity
* Modeling $N\mid N \ge n$
* How to model 2 reinstatements

## Publication and Use
* Interest from RMIR

Open
----
- issue with mass and bodoff1 portfolio

Aug 30
------

* Examples params split into portfolios, lines and severity
* Examples is scriptable to return line or severity
* Portfolio tracks all update parameters to determine recalc
* Portfolio.update trim_df option
* User portfolios
* estimate bucket looks at limit and moments
* ability to save to YAML, optionally to user set
* Hack function to make line aggs for industry from IEE extract...very ROUGH

Aug 29
------

* removed great depenedence
* experimented with YAML persistence...tricky!
* DONE Hash of status for last run and timestamp (np.timeformat!) [not all inputs...]
* DONE Histogram mode: cts or discrete: other adj needed for cts? (Half bucket off?)
    - If you lower the bucket size then continuous will increase the mean by half the new (smaller) bucket size
* DONE trim density_df to get rid of unwanted columns
* DONE apply_distortion works on density_df and applied dist is kept for reference
* DONE fixed severity type: how to make easy
    - 'severity': {'name': 'histogram', 'xs' : (1), 'ps': (1)}
* DONE Rationalize graphics
    - See make_axis and make_pandas_axis...just needs to be propogated
* DONE created example as a class, reads examples from YAML
* DONE Include apply_distortion into density_df (easy change, easy to change back)
* removed adjust_axis_format - put in K, M, B into axis text
    - figure best format


Aug 28
-------

* added ability to recurse and use a portfolio as a severity
* deepcopy
* drawstyle='steps-post'
* distribution_factory, deleted old verison, combined in _ex
* added logging, read_log function as dataframe
* overrode + *
* removed junk from bottom:
   - old list of examples
   - qd = quick display
   - qdso with sort order on split _ index
   - qtab quick tabulate
   - cumintegralnew
   - cumintegraltest
   - pno - pre-iterator axes
   - defuzz - now in update
   - KEPT fit_distortion - ?see CAS talks, calibrate to a given - - distribution (? fit one transf to another?)

Aug 27:
------

* repr and string
* uat function into CPortfolio
* insurability_triangle
* estimate function

