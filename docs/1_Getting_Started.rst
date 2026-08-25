.. 2022-11-10: reviewed

*****************
Getting Started
*****************

Installation
=============

To install from PyPI ::

    pip install aggregate

See https://pypi.org/project/aggregate/.

Source Code
===========

The source code is hosted on GitHub, https://github.com/mynl/aggregate.

Prerequisites
=============

This help assumes you know how to program in Python, understand probability, and are familiar with the concept of an aggregate distribution. Awareness of insurance terminology such as limit, attachment and deductible, and the material covered in `SOA exam STAM <https://www.soa.org/education/exam-req/edu-exam-stam-detail/>`_, `CAS exam MAS I <https://www.casact.org/exam/exam-mas-i-modern-actuarial-statistics-i>`_, or `IFOA CS-2 <https://www.actuaries.org.uk/curriculum_entity/curriculum_entity/8>`_ is helpful.

License
=======

BSD 3.

Dependencies
=============

See pyproject.toml. Requirements are split between those needed to run the project, and a larger set needed to build the documentation. All dependencies are standard packages.

Help Parameters and Examples
================================

.. warning::
    All parameters are fabrications. They try to be realistic (or at least not materially unrealistic) but are not intended to be applied to real-world pricing. They are for educational purposes only.

Help Structure
===============

This help is in four parts.

#. Getting Started (this document).
#. :doc:`2_Aggregate_Overview`, explaining how ``aggregate`` computes and providing several examples.
#. :doc:`4_dec_Language_Reference`: syntax and grammar.
#. :doc:`3_Reference`: all functions, classes, methods, and properties.

There is also a :doc:`Bibliography<7_bibliography>`.

Help Coding Conventions
=======================

Throughout the help, you will see input code inside blocks such as:

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

A DecL statement may span several lines; statements are separated by a blank
line or by a semicolon at the end of a line, which keeps input compact.

Numbers and Units
==================

You can choose your own units. The examples include numbers interpreted in ones, thousands, and millions. Amounts are broadly calibrated to make sense in USD, EUR, and GBP.

``aggregate`` Hello World
==========================

The only object you need to import to get started is ``build``. The quick display function ``qd`` is a nice-to-have utility function that handles printing with sensible defaults. It is used extensively throughout.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

    build

``build`` is a :class:`Underwriter` object. It  allows you to create all other
objects and  includes a library of examples, called the recipe base.

Using ``build`` you can create an :class:`Aggregate` object using a :doc:`DecL program <4_dec_Language_Reference>`. For example, the program::

    agg Eg1 dfreq [1:5] dsev [1:3]

creates an aggregate distribution called ``Eg1``. The frequency distribution is 1, 2, 3, 4, or 5, all equally likely, and the severity is 1, 2, or 3, also equally likely. The mean frequency is 3, the mean severity 2, and hence the aggregate has a mean of 6. It is built and displayed like so:

.. ipython:: python
    :okwarning:

    a = build('agg Eg1 dfreq [1:5] dsev [1:3]')
    qd(a)

.. note::

   Working in JupyterLab? The cell above can be written as DecL and nothing else. See :ref:`the %%agg cell magic <agg-magic>` at the foot of this page.

The DecL program::

    agg Eg2 5 claims 1000 xs 0 sev lognorm 50 cv 4 poisson

creates a realistic insurance portfolio, with 5 expected claims, severity sampled from a 1000 xs 0 layer of a lognormal with mean 50 and CV 4 and Poisson frequency.

:class:`Aggregate` objects act like a discrete probability distribution. There are properties for the mean, standard deviation, coefficient of variation (cv), and skewness.

.. ipython:: python
    :okwarning:

    a.actual_m, a.actual_sd, a.actual_cv, a.actual_skew

They have probability mass, cumulative distribution, survival, and quantile (inverse of distribution) functions.

.. ipython:: python
    :okwarning:

    a.pmf(6), a.cdf(5), a.sf(6), a.q(a.cdf(6)), a.q(0.5)

It is easy to check some of these calculations. The probability of the minimum outcome of one equals 1/15 (1/5 for a frequency of 1 and 1/3 for a severity of 1) and the maximum outcome of 15 equals 1/1215 (1/5 for a frequency of 5 and (1/3)**5 to draw severity of 3 on each). The object returns the correct values.

.. ipython:: python
    :okwarning:

    a.pmf(1), 1/15, a.pmf(15), 1/5/3**5, 5*3**5

Creating an object automatically stores its specification as a **recipe**, with name ``Eg1``. Use :attr:`build.recipes` to view them. Each row carries the entry's program and spec plus how it describes itself, a one-line ``note`` and its ``tags``; here we show only the first few columns.

.. ipython:: python
    :okwarning:

    qd(build.recipes.iloc[:5, :9], line_width=73, max_colwidth=50, justify='left')
    qd(build.recipes.query('name == "Eg1"').iloc[:, :9], line_width=73, max_colwidth=50, justify='left')

.. _agg-magic:

The ``%%agg`` Cell Magic
=========================

In `JupyterLab <https://jupyter.org>`_ a DecL program does not have to live inside a Python string. Load the magic once per kernel::

    %load_ext aggregate.magics

and then write the program as the cell::

    %%agg
    agg Eg1 dfreq [1:5] dsev [1:3]

That is exactly ``a = build('agg Eg1 dfreq [1:5] dsev [1:3]')`` followed by ``qd(a)``, with two conveniences. The program is no longer inside quotes, so an editor still sees DecL and highlights it as DecL, and the object is bound to its own declared name as well as to ``a``, giving both ``a`` and ``Eg1``.

The load is explicit on purpose: importing a package should not put names in your notebook that you did not ask for.

A cell can declare as many objects as you like, separated by a blank line or by a semicolon at the end of a line::

    %%agg book
    agg Line1 100 claims 1000 xs 0 sev lognorm 50 cv 3 poisson

    agg Line2 200 claims 1000 xs 0 sev gamma 40 cv 1.5 poisson

    port Book agg.Line1 agg.Line2

Each declared name is bound on its own, so ``Line1``, ``Line2`` and ``Book`` are all live afterwards, and the name given to the magic, ``book`` here, is bound to a dictionary of all three. A name DecL allows but Python does not, such as ``Mack2003.Lognorm``, is reachable through that dictionary. With one output the same name is bound to the object itself. The default is ``a``, so a bare ``%%agg`` behaves like the single-object example above.

The arguments control the volume and the grid:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Argument
     - Effect
   * - ``name``
     - Positional. The variable to bind, default ``a``.
   * - ``-q``, ``--quiet``
     - Build and report what was bound, skipping the ``qd`` display. The setting for a cell declaring a dozen objects.
   * - ``-s``, ``--silent``
     - Build and print nothing at all.
   * - ``-p``, ``--plot``
     - Also call ``.plot()``, for a cell declaring a single object. Independent of the volume, so ``-s -p`` draws the figure and says nothing.
   * - ``-v``, ``--validation``
     - Display each built object's ``validation_df``, the moment against estimate audit, in place of the object summary. An object with no such frame, a recipe stub or an ``expr`` value, displays itself as before. The volume flags apply unchanged.
   * - ``--log2 N``
     - Number of buckets as a power of two. Default 0, meaning let the object choose.
   * - ``--bs X``
     - Bucket size, evaluated in the notebook namespace, so ``1/32`` and a variable both work. Default 0, meaning let the object choose.

A ``hints{}`` clause inside the program does the same job as ``--log2`` and ``--bs`` and travels with the declaration, which is usually the better place for it.

The magic is a wrapper around :meth:`Underwriter.build_many`, and :meth:`Underwriter.build` is that same method plus an unwrap, so there is no second spelling to learn for a cell that happens to declare several objects.

The :doc:`2_Aggregate_Overview` contains more details and examples.
