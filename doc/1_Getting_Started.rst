.. 2022-11-10: reviewed

*****************
Getting Started
*****************

Installation
=============

Install from PyPI ::

   pip install aggregate

See https://pypi.org/project/aggregate/. You must have ``matplotlib>=3.5`` installed, as well as the other packages listed in requirements.txt.

Source Code
===========

The source code is hosted on GitHub, https://github.com/mynl/aggregate.

Prerequisites
=============

The help assumes you know how to program in Python, understand probability, and are familiar with the concept of an aggregate distribution. Awareness of insurance terminology such as limit, attachment and deductible, and the material covered in `SOA exam STAM <https://www.soa.org/education/exam-req/edu-exam-stam-detail/>`_, `CAS exam MAS I <https://www.casact.org/exam/exam-mas-i-modern-actuarial-statistics-i>`_, or `IFOA CS-2 <https://www.actuaries.org.uk/curriculum_entity/curriculum_entity/8>`_ is helpful.

License
=======

:ref:`BSD 3<../LICENCE>`

Help Structure
===============

The help structured around **access**, **application**, **theory**, and **implementation**. There are six parts.

#. Getting Started (this document).
#. :doc:`2_User_Guides`, explaining how to **access** functionality and practical guides explaining how to **apply** it.
#. :doc:`3_Reference`: all functions, classes, methods, and properties.
#. :doc:`4_dec_Language_Reference`: syntax and grammar.
#. :doc:`5_Technical_Guides`, covering the underlying **theory** and its specific **implementation**.
#. :doc:`6_Development`, giving some history, the design philosophy, and ideas for future development.

There is also a :doc:`7_bibliography`.

Help Coding Conventions
=======================

Throughout the help, you will see input code inside code blocks such as:

::

    import pandas as pd
    pd.DataFrame({'A': [1, 2, 3]})


or:

.. ipython:: python

    import pandas as pd
    pd.DataFrame({'A': [1, 2, 3]})

The first block is a standard Python input, while in the second the ``In [1]:`` indicates the input is inside a `notebook <https://jupyter.org>`__. In Jupyter Notebooks the last line is printed and plots are shown inline.

For example:

.. ipython:: python

    a = 1
    a

is equivalent to:

::

    a = 1
    print(a)


``aggregate`` Hello World
==========================

The only object you need to import to get started is ``build``. ``qd`` (quick display) is a nice-to-have utility function that handles printing with sensible defaults.

.. ipython:: python
    :okwarning:

   from aggregate import build, qd

   build

``build`` is a :class:`Underwriter` object. It  allows you to create all other
objects and  includes a library of examples, called the knowledge.

Using ``build`` you can create an :class:`Aggregate` object using an DecL  program. For example, the program

::

    agg Eg1 dfreq [1:5] dsev [1:3]

creates an aggregate distribution called ``Eg1``. The frequency distribution is 1, 2, 3, 4, or 5, all equally likely, and the severity is 1, 2, or 3, also equally likely. The mean frequency is 3, the mean severity 2, and hence the aggregate has a mean of 6. It is built and displayed like so:

.. ipython:: python
    :okwarning:

    a = build('agg Eg1 dfreq [1:5] dsev [1:3]')
    qd(a)

:class:`Aggregate` objects act like a discrete probability distribution. There are properties for the mean, standard deviation, coefficient of variation (cv), and skewness.

.. ipython:: python
    :okwarning:

    a.agg_m, a.agg_sd, a.agg_cv, a.agg_skew

They have probability mass, cumulative distribution, survival, and quantile (inverse of distribution) functions.

.. ipython:: python
    :okwarning:

    a.pmf(6), a.cdf(5), a.sf(6), a.q(a.cdf(6)), a.q(0.5)

It is easy check some of these calculations. The probability of the minimum outcome of one equals 1/15 (1/5 for a frequency of 1 and 1/3 for a severity of 1) and the maximum outcome of 15 equals 1/1215 (1/5 for a frequency of 5 and (1/3)**5 to draw severity of 3 on each). The object returns the correct values.

.. ipython:: python
    :okwarning:

    a.pmf(1), 1/15, a.pmf(15), 1/5/3**5, 5*3**5

Creating an object automatically adds its specification to the knowledge, with name ``Eg1``. Use :attr:`build.knowledge` to view the knowledge dataframe.

.. ipython:: python
    :okwarning:

   print(build.knowledge.head())

   build.knowledge.query('name == "Eg1"')

The :doc:`2_User_Guides` contain more details and examples.
