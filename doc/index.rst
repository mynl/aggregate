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

:mod:`aggregate` solves insurance, risk management, and actuarial problems using realistic models that reflect underlying frequency and severity.
It delivers the speed and accuracy of parametric distributions to situations that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution as the lognormal.
:mod:`aggregate` includes an expressive language called DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.

This help document is in six parts plus a bibliography.

#. :doc:`1_Getting_Started` --- Get up and running: installation, ``aggregate`` "hello world", and a glimpse into the functionality.

#. :doc:`2_User_Guides` --- How to solve real-world actuarial problems using ``aggregate``.


#. :doc:`3_Reference` --- Documentation for every class and function, for developers and more advanced users.


#. :doc:`4_dec_Language_Reference` --- The Dec Language (DecL)  for specifying aggregate distributions.


#. :doc:`5_Technical_Guides` --- Probability theory background and the numerical implementation methods employed by ``aggregate``.

#. :doc:`6_Development` --- Design philosophy, competing products, future development ideas, and historical perspective.

#. :doc:`7_bibliography`
