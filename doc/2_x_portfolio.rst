.. _2_x_portfolio:

==============================
Portfolios
==============================


14. [QS6] Creating a simple `Portfolio` using `agg` (2 line example);  [cdf, pmf, sf, q, plot, describe, statistics]


Creating an :class:`Aggregate` from the pre-loaded library

.. ipython:: python
    :okwarning:

    a, d = build.show('^B.*1$')


Creating a :class:`Portfolio`  by passing ...


Viewing data
------------

.. See the :ref:`Basics section <basics>`.

Use :attr:`build.knowledge` and :meth:`build.qshow` to view the knowledge.

.. ipython:: python
    :okwarning:

   build.knowledge.head()
   build.qshow('^E\.')

.. note::

   :meth:`DataFrame.to_numpy` does *not* include the index or column labels in the output.



``Aggregate`` and ``Portfolio`` objects both have the following attributes and functions.

* A ``density_df`` a dataframe containing the relevant probability distributions and other information.
* A ``report_df`` providing a building block summary.
* A ``plot`` method to visualize the underlying distributions.
* An ``update`` method, to trigger the numerical calculation of probability distributions.
* A ``spec`` dictionary, containing the input information needed to recreate each object. For example, if ``ag`` is an ``Aggregate`` object, then ``Aggregate(**ag.spec)`` creates a new copy.
* ``log2`` and ``bs``

Second-level attributes:

* A ``statistics_df`` and ``statistics_total_df`` dataframes with theoretically derived statistical moments (mean, variance, CV, sknewness, etc.)
* An ``audit_df`` with information to check if the numerical approximations appear valid. Numerically computed statistics are prefaced ``emp_``.



How `Underwriter` works
-----------------------


Each object two properties (kind, name) and three manifestations (spec, program, ob[ject])

1. kind: agg, sev, port, distortion
2. name of the object
3. spec: dictionary specification
4. program: text string, the aggregate program
5. object: the actual Python object, an instance of a class

``Underwriter._knowledge`` is a Pandas dataframe with row index (kind, name) mapping to (spec, program)

``.build`` wraps ``.write``

* calls write with update=False; it handles update with good defaults
* if only one output, strips it out of the dict and returns the object
* if only one port output, returns that

``.write``

* lowest level update function
* calls ``interpret_program``
* calls ``factory``
* reads program, preprocess, parse by line, expand sev.name, agg.name, create, update
* returns a list of Answer objects with  keys kind, name, spec, program, and object

``.write_file``

* Reads a file and passes to ``write``.

``.interpret_program`` (called by ``write``)

* maps the programs and specs together in an Answer(kind, name, spec, program, object=None)
* adds data to _knowledge

``.factory``

* Answer --> Answer with object created  the object and updated

``.interpreter_xxx``

* run programs through parser, for debugging purposes
* nothing created
* ``_interpreter_work`` does the actual parsing
* ``interpreter_line`` calls work on one line
* ``interpreter_file`` calls work on each line in a file
* ``interpreter_list`` calls work on each item in a list
