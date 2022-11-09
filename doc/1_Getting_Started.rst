*****************
Getting Started
*****************

Installation
=============

Install from PyPI ::

   pip install aggregate

See https://pypi.org/project/aggregate/.


Source Code
===========

The source code is hosted on GitHub, https://github.com/mynl/aggregate.

Prerequisites
=============

The help assumes you know how to program in Python, understand probability, and are familiar with the concept of an aggregate distribution. Awareness of the material covered in `SOA exam STAM <https://www.soa.org/education/exam-req/edu-exam-stam-detail/>`_, `CAS exam MAS I <https://www.casact.org/exam/exam-mas-i-modern-actuarial-statistics-i>`_, or `IFOA CS-2 <https://www.actuaries.org.uk/curriculum_entity/curriculum_entity/8>`_ is helpful. The help also assumes you understand insurance terminology like limit, attachment, deductible.

License
=======

BSD 3


Help coding conventions
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


First steps with ``aggregate``
==============================

This is a short introduction to aggregate, for new users.

The only object you need to import to get started is ``build``.

.. ipython:: python
    :okwarning:

   from aggregate import build

   build

``build``
is a :class:`Underwriter` object. It  allows you to create all other objects and  includes a library of examples, called the knowledge.

Using build, you can create an :class:`Aggregate` object using a simple ``agg`` language  program. For example, the program ``agg Eg1 dfreq [1:5] dsev [1:3]`` creates an object called ``Eg1`` in the knowledge and specifies the frequency as 1, 2, 3, 4, or 5 all equally likely and the severity as 1, 2, or 3, also equally likely. The mean frequency is 3, mean severity 2, and hence the aggregate has a mean of 6. It is executed:

.. ipython:: python
    :okwarning:

    a = build('agg Eg1 dfreq [1:5] dsev [1:3]')
    a

Printing the object returns information about the frequency and severity stochastic models and how the object was computed. The last DataFrame can be accessed directly as the property ``a.describe``. Creating an object automatically adds its specification to the knowledge.

Use :attr:`build.knowledge` to view the knowledge. It is another   DataFrame.

.. ipython:: python
    :okwarning:

   build.knowledge.head()

Aggregates in the knowledge can be created by name. The next example, from the knowledge, uses the roll of a dice for the frequency and the severity. Its agg program is ``agg B.Dice14 dfreq [1:6] dsev [1:6]``.

.. ipython:: python
    :okwarning:

    a = build('B.Dice14')
    a


:class:`Aggregate` objects act like a discrete probability distribution. They have probability mass, cumulative distribution, survival, and quantile (inverse of distribution) functions. There are properties for the mean, standard deviation, coefficient of variation (cv), and skewness.

.. ipython:: python
    :okwarning:

    (a.pmf(6), a.cdf(6), a.sf(6), a.q(a.cdf(6)), a.q(0.5),
    a.agg_m, a.agg_sd, a.agg_skew)

It is easy check some of these calculations. The probability of the minimum outcome of one equals 1/36 (1/6 to roll a frequency of 1 and a severity of 1) and the maximum outcome of 36 equals 1/6**7 (1/6 to roll a frequency of 6 and (1/6)**6 to draw severity of 6 on each). The object returns the correct values.

.. ipython:: python
    :okwarning:

    a.pmf(1), 1/36, a.pmf(36), 1/6**7


The :doc:`2_User_Guide` contains more details and examples.
