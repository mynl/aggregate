.. _2_x_limits:

.. _2_agg_class_layers_subclause:

The Limits Sub-Clause
-----------------------

The optional ``limits`` sub-clause describes policy occurrence limits and deductibles. For example::

    100 xs 0
    inf xs 100
    750 xs 250
    1 x 1

The first applies an occurrence limit of 100. The second applies a deductible of 100. The third is an excess layer, with limit 750 and retention 250. The last is also an excess layer of 1 xs 1.
``inf`` denotes infinity, for an unlimited layer. Either `xs` or `x` are acceptable.

:ref:`Multiple layers <2_x_vectorization>` can be entered at once using vectors.


