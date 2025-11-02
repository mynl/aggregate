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
   2_User_Guides
   3_Reference
   4_dec_Language_Reference
   5_Technical_Guides
   6_Development
   7_bibliography



****************
Introduction
****************

:mod:`aggregate` builds approximations to compound (aggregate) probability distributions quickly and accurately.
It can be used to solve insurance, risk management, and actuarial problems using realistic models that reflect underlying frequency and severity.
It delivers the speed and accuracy of parametric distributions to situations that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution as the lognormal.
:mod:`aggregate` includes an expressive language called DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.

This help document is in six parts plus a bibliography.

.. grid:: 2 2 3 3
   :gutter: 2

   .. grid-item-card::
      :img-top: _static/gs.png
      :link: 1_Getting_Started
      :link-type: doc
      :text-align: left

      Get up and running: installation, :mod:`aggregate` "hello world", and a glimpse into the functionality.

   .. grid-item-card::
      :img-top: _static/ug.png
      :link: 2_User_Guides
      :link-type: doc
      :text-align: left

      How to solve real-world actuarial problems using :mod:`aggregate`.

   .. grid-item-card::
      :img-top: _static/api.png
      :link: 3_Reference
      :link-type: doc
      :text-align: left

      Documentation for every class and function, for developers and more advanced users.

   .. grid-item-card::
      :img-top: _static/decl.png
      :link: 4_dec_Language_Reference
      :link-type: doc
      :text-align: left

      The Dec Language (DecL) for specifying aggregate distributions.

   .. grid-item-card::
      :img-top: _static/tg.png
      :link: 5_Technical_Guides
      :link-type: doc
      :text-align: left

      Probability theory background and the numerical implementation methods employed by :mod:`aggregate`.

   .. grid-item-card::
      :img-top: _static/dev.png
      :link: 6_Development
      :link-type: doc
      :text-align: left

      Design philosophy, competing products, future development ideas, and historical perspective.
