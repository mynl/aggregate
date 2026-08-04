.. aggregate documentation master file, created by
   sphinx-quickstart on Sat Sep  1 14:08:11 2018.

#######################
aggregate Documentation
#######################

.. toctree::
   :maxdepth: 3
   :hidden:
   :numbered:

   1_Getting_Started
   2_Aggregate_Overview
   3_Reference
   4_dec_Language_Reference
   7_bibliography


****************
Introduction
****************

:mod:`aggregate` builds approximations to compound (aggregate) probability distributions quickly and accurately.
It can be used to solve insurance, risk management, and actuarial problems using realistic models that reflect underlying frequency and severity.
It delivers the speed and accuracy of parametric distributions to situations that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution as the lognormal.
:mod:`aggregate` includes an expressive language called DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.

This help document is in four parts plus a bibliography.

#. :doc:`Getting Started <1_Getting_Started>`: get up and running: installation, :mod:`aggregate` "hello world", and a glimpse into the functionality.

#. :doc:`Aggregate Overview <2_Aggregate_Overview>`: the computation pipeline, bucket selection, and the probability theory and numerical methods behind the library.

#. :doc:`Reference <3_Reference>`: documentation for every class and function, for developers and more advanced users.

#. :doc:`Dec Language Reference <4_dec_Language_Reference>`: the Dec Language (DecL) for specifying aggregate distributions.
