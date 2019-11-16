
History and Applications
========================

Contents
--------

.. toctree::
   :maxdepth: 3


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


