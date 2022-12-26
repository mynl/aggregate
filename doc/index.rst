.. aggregate documentation master file, created by
   sphinx-quickstart on Sat Sep  1 14:08:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

.. sectnum
   :depth: 3

.. see https://shields.io/ search github

.. |downloads| image:: https://img.shields.io/pypi/dm/aggregate.svg
    :target: https://pepy.tech/project/aggregate
    :alt: Downloads

.. |stars| image:: https://img.shields.io/github/stars/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/stargazers
    :alt: Github stars

.. |forks| image:: https://img.shields.io/github/forks/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/network/members
    :alt: Github forks

.. |contributors| image:: https://img.shields.io/github/contributors/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/graphs/contributors
    :alt: Contributors

.. |version| image:: https://img.shields.io/pypi/v/aggregate.svg?label=pypi
    :target: https://pypi.org/project/aggregate
    :alt: Latest version

.. |activity| image:: https://img.shields.io/github/commit-activity/m/mynl/aggregate
   :target: https://github.com/mynl/aggregate
   :alt: Latest Version

.. |py-versions| image:: https://img.shields.io/pypi/pyversions/aggregate.svg
    :alt: Supported Python versions

.. |license| image:: https://img.shields.io/pypi/l/aggregate.svg
    :target: https://github.com/mynl/aggregate/blob/master/LICENSE
    :alt: License

.. |packages| image:: https://repology.org/badge/tiny-repos/python:aggregate.svg
    :target: https://repology.org/metapackage/python:aggregate/versions
    :alt: Binary packages

.. |doc| image:: https://readthedocs.org/projects/aggregate/badge/?version=latest
    :target: https://aggregate.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |twitter| image:: https://img.shields.io/twitter/follow/mynl.svg?label=follow&style=flat&logo=twitter&logoColor=4FADFF
    :target: https://twitter.com/SJ2Mi
    :alt: Twitter Follow

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

|  |activity| |doc| |version|
|  |py-versions| |downloads|
|  |license| |packages|  |twitter|

:mod:`aggregate` makes working with aggregate (compound) probability distributions fast, accurate, easy, and intuitive.  It can solve problems in insurance, risk management, actuarial science, and related areas, using realistic models that reflect the actual underlying frequency and severity generating processes. It includes an expressive language called DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.

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
