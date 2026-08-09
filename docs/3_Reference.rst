****************
API Reference
****************

This chapter is the generated reference for the ``aggregate`` public API,
organized by topic. Each page combines a short narrative, saying what the module
is for and where its code now lives after the 1.0.0a90 to a95 split, with the
autodoc'd classes and functions.

Read :doc:`3_reference/3_x_API_Stability` first: it says which parts of the surface below carry a stability promise and which two, :mod:`aggregate.charts` and :mod:`aggregate.exhibits`, are provisional in the sense of :pep:`411` and expected to change.

The pages then fall into three groups. The **stable public surface** is the classes and functions you import and call (``aggregate.Aggregate``, ``aggregate.build``, ``aggregate.Distortion``, …), from :doc:`3_reference/3_x_Underwriter` through :doc:`3_reference/3_x_Auxiliary`. The two **provisional modules** follow, each carrying a warning admonition of its own. The final page, :doc:`3_reference/3_x_Internal_Architecture`, documents the **internal module layout**, the concern modules and subsystems the public classes delegate to. These internal modules are not part of the supported API; they are documented because they are the map of how the library is organized.

.. toctree::
   :maxdepth: 3

   3_reference/3_x_API_Stability
   3_reference/3_x_Underwriter
   3_reference/3_x_Parser
   3_reference/3_x_Distribution
   3_reference/3_x_Portfolio
   3_reference/3_x_PnL
   3_reference/3_x_Distortion
   3_reference/3_x_Bounds
   3_reference/3_x_Utilities
   3_reference/3_x_Auxiliary
   3_reference/3_x_Charts
   3_reference/3_x_Exhibits
   3_reference/3_x_Internal_Architecture
