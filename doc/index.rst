.. aggregate documentation master file, created by
   sphinx-quickstart on Sat Sep  1 14:08:11 2018.

#######################
aggregate Documentation
#######################

Introduction
=============

:mod:`aggregate` solves insurance, risk management, and actuarial problems using realistic models that reflect underlying frequency and severity.
It delivers the speed and accuracy of parametric distributions to situations that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution as the lognormal.
:mod:`aggregate` includes an expressive language called DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.

This help document is in six parts plus a bibliography.

.. Panels at https://sphinx-panels.readthedocs.io/en/latest/
   img-top-cls: pl-5 pr-5

.. panels::
   :img-top-cls: height=100px bg-success

   ---
   :img-top: _static/gs.png

   Get up and running: installation, ``aggregate`` "hello world", and a glimpse into the functionality.

   +++

   .. link-button:: 1_Getting_Started
      :type: ref
      :text: Getting Started
      :classes: btn-block btn-secondary stretched-link

   ---
   :img-top: _static/ug.png

   How to solve real-world actuarial problems using ``aggregate``.

   +++

   .. link-button:: 2_User_Guides
      :type: ref
      :text: User Guides
      :classes: btn-block btn-secondary stretched-link

   ---
   :img-top: _static/api.png


   Documentation for every class and function, for developers and more advanced users.

   +++

   .. link-button:: 3_Reference
      :type: ref
      :text: API Reference
      :classes: btn-block btn-secondary stretched-link

   ---
   :img-top: _static/decl.png


   The Dec Language (DecL)  for specifying aggregate distributions.

   +++

   .. link-button:: 4_dec_Language_Reference
      :type: ref
      :text: DecL Reference
      :classes: btn-block btn-secondary stretched-link

   ---
   :img-top: _static/tg.png


   Probability theory background and the numerical implementation methods employed by ``aggregate``.

   +++

   .. link-button:: 5_Technical_Guides
      :type: ref
      :text: Technical Guides
      :classes: btn-block btn-secondary stretched-link

   ---
   :img-top: _static/dev.png

   Design philosophy, competing products, future development ideas, and historical perspective.

   +++

   .. link-button:: 6_Development
      :type: ref
      :text: Development
      :classes: btn-block btn-secondary stretched-link


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
