.. _2_x_10mins:

10 minutes to aggregate
=========================

This is a test drive of aggregate. It is aimed at users who are programming interactively in Jupyter Lab or a similar REPL (read–eval–print loop) interface. Not everything will make sense the first time, but you will see what you can achieve with the package.

The two most important classes in the library are :class:`Aggregate`, which models a single unit (*unit* refers to a line of business, business unit, geography, operating division, etc.), and :class:`Portfolio`, which models multiple units. The ``Portfolio`` broadly includes all the functionality in ``Aggregate`` and adds pricing, calibration, and allocation methods. Supporting these two are :class:`Severity` that models size of loss distributions. The :class:`Underwriter` keeps track of everything and acts as a helper.

To get started, import ``build``, a pre-compiled :class:`Underwriter`.

.. ipython:: python
    :okwarning:

   from aggregate import build
   build

``build`` reports the size of its knowledge - aggregates and portfolios it knows how to construct - and other basic information.

Using the ``build`` object you can create all other objects using the agg language. Let's make an aggregate and a portfolio. The aggregate has a Poisson frequency and lognormal severity with mean 10 and cv 1. The portfolio combines to aggregates with gamma-mixed (negative binomial) frequency and gamma severity.

.. ipython:: python
    :okwarning:

    a = build('agg Example1 5 claims sev lognorm 10 cv 1 poisson')
    print(a)

    p = build('''port Port.1
        agg Unit.A 10 claims sev lognorm 10 cv 1 mixed gamma .25
        agg Unit.B 4  claims sev lognorm 20 cv 1.5 mixed gamma .3''')
    print(p)


