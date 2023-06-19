.. _2_x_limits:

.. _2_agg_class_layers_subclause:

.. reviewed 2022-12-24

The Limits Sub-Clause
-----------------------

The optional ``limits`` sub-clause specifies policy occurrence limits and deductibles.

**Examples**::

    100 xs 0
    inf xs 100
    750 xs 250
    1 x 1

* ``100 xs 0`` applies an occurrence limit of 100.
* ``inf xs 100`` applies a deductible of 100 and no limit.
* ``750 xs 250`` is an excess layer, with limit 750 and deductible 250.
* ``1 x 1`` is also an excess layer of 1 xs 1.

``inf`` denotes infinity, for an unlimited layer. Both ``xs`` and ``x`` are acceptable.

:ref:`Multiple layers <2_x_vectorization>` can be entered at once using vectors.


