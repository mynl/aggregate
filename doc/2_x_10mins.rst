.. _10mins:

************************
10 minutes to aggregate
************************

This is a short introduction to aggregate, for new users.

Customarily, we import as follows:

.. ipython:: python
    :okwarning:

   import warnings
   warnings.filterwarnings("ignore")

   from aggregate import build

The ``build`` object allows you to create all other objects using the agg language.


Object creation
---------------

Creating an :class:`Aggregate` using a simple agg program

.. ipython:: python
    :okwarning:

    a = build('agg Eg1 dfreq [1:6] dsev [1:4]')
    a


Creating an :class:`Aggregate` from the pre-loaded library

.. ipython:: python
    :okwarning:

    a, d = build.show('^B.*1$')


Creating a :class:`Portfolio`  by passing ...


Viewing data
------------

See the :ref:`Basics section <basics>`.

Use :attr:`build.knowledge` and :meth:`build.qshow` to view the knowledge.

.. ipython:: python
    :okwarning:

   build.knowledge.head()
   build.qshow('^E\.')

.. note::

   :meth:`DataFrame.to_numpy` does *not* include the index or column labels in the output.

:func:`~DataFrame.describe` shows a quick statistic summary of your data:

.. ipython:: python

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
            np.random.randn(1000, 4), index=range(1000), columns=["A", "B", "C", "D"]
        )

    df = df.cumsum()

    plt.figure();
    df.plot();
    @savefig frame_plot_basic.png
    plt.legend(loc='best');


That should show a graph. But this is better.

.. jupyter-execute::

  name = 'world'
  print('hello ' + name + '!')

Base from docs.

.. jupyter-execute::

    from matplotlib import pyplot
    %matplotlib inline

    x = np.linspace(1E-3, 2 * np.pi)

    pyplot.plot(x, np.sin(x) / x)
    pyplot.plot(x, np.cos(x))
    pyplot.grid()


How did that come out?

.. jupyter-execute::

    df.head()

And that head command?
