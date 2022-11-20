.. _development:

*************************
Design and Development
*************************

Design Philosophy
====================

* Work at the optimal big-O order, but don't worry about speed until it becomes a problem
* Save everything until space becomes an issue
* Sensible defaults for everything
* Don't make formatting decisions for the user, use defaults
* Offer dataframe styles and renamers for dataframes but do not apply automatically
* Use sensible number formats


Development Outline (TODOs)
===============================

Non Programming Enhancements
----------------------------
* Library of realistic severity curves by line and country
* By line industry aggregate distributions in agg format from method of moments fits to historical data, per PIRC
* Credit modeling: what is distortion implied by bond credit curve? By cat bond pricing?
* Jon Evans note and severity??
* Jed note??

Short Term
-----------
* Width of printed output
* Understand output for collateral and priority!
* Output Levy measure
* Funky objects from JacodS? Simple jump examples

Medium Term
------------
* recommend_bucket function when passed infinity?
* More consistent and informative reports and plots (e.g. include severity match in agg)
* Convex Hull distortion built from pricing
* Delete items easily from the database
* Save / load from non-YAML, persist the database; dict to agg language converter? Get rid of YAML dependence
* Using agg as a severity (how!)
* Name as a member in dict vs list conniptions (put up with duplication?)

Nice to Have Enhancements
-------------------------
* How to model two reinstatements?



History
=========

I have built several iterations of software to work with aggregate distributions, the first in 1997.

*  A Matlab version for CNA Re with a graphical interface. Computed aggregates by business unit and the portfolio total. Used to discover the shocking fact there was only a 53 percent chance of achieving plan...which is obvious in hindsight but was a surprise at the time.
*  A C++ version of the Matlab code called SADCo in 1997-99. This code sits behing [MALT](http://www.mynl.com/MALT/home.html).
*  Another C++ version with an implementation of the Iman Conover method to combine aggregates with correlation using the (gasp) normal copula, [SCARE](http://www.mynl.com/wp/default.html)
*  At Aon Re I worked on their simulation based tools called ALG (Aggregate Loss Generator) which simulated losses and Prime/Re which manipulated the simulations and applied various reinsurance structures. ALG used a shared mixing variables approach to correlation.
*  At Aon Re I also built related tools
  -  The Levy measure maker
  -  A simple approach to multi-year modeling based on re-scaling a base year, convolving using FFTs and tracking (and stopping) in default scenarios
*  At Aon Benfield I was involved with [ReMetric](http://www.aon.com/reinsurance/analytics-(1)/remetrica.jsp), a very sophisticated, general purpose DFA/ERM simulation tool,

