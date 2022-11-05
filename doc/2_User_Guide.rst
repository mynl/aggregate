==============
User Guide
==============

The User Guide describes aggregate by problem area. Each subsection
introduces a problem, such as how to compute the impact of limits and deductibles, and shows how to solve it using aggregate, generally using examples.

New users should start with :ref:`10min`.

For a high level summary of the pandas fundamentals, see :ref:`objintro` and :ref:`basics`.

Further information on any specific method can be obtained in the
:ref:`api`.

How to read the guides
----------------------

In the guide you will see input code inside code blocks such as:

::

    import pandas as pd
    pd.DataFrame({'A': [1, 2, 3]})


or:

.. ipython:: python

    import pandas as pd
    pd.DataFrame({'A': [1, 2, 3]})

The first block is a standard python input, while in the second the ``In [1]:`` indicates the input is inside a `notebook <https://jupyter.org>`__. In Jupyter Notebooks the last line is printed and plots are shown inline.

For example:

.. ipython:: python

    a = 1
    a
is equivalent to:

::

    a = 1
    print(a)



Guides
-------

.. If you update this toctree, also update the manual toctree in the
   main index.rst.template

.. toctree::
    :maxdepth: 1

    2_x_10mins
    2_x_objintro
    2_x_basics
    2_x_all_the_rest



