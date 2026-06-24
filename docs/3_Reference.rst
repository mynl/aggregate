****************
API Reference
****************

This chapter is the generated reference for the ``aggregate`` public API,
organized by topic. Each page combines a short narrative — what the module is
for and where its code now lives after the 1.0.0a90–a95 split — with the
autodoc'd classes and functions.

The first eight pages document the **public surface**: the classes and functions
you import and call (``aggregate.Aggregate``, ``aggregate.build``,
``aggregate.Distortion``, …). The final page,
:doc:`3_reference/3_x_Internal_Architecture`, documents the **internal module
layout** — the concern modules and subsystems the public classes delegate to.
These internal modules are not part of the supported API; they are documented
because they are the map of how the library is organized.

.. toctree::
   :maxdepth: 3

   3_reference/3_x_Underwriter
   3_reference/3_x_Parser
   3_reference/3_x_Distribution
   3_reference/3_x_Portfolio
   3_reference/3_x_Distortion
   3_reference/3_x_Bounds
   3_reference/3_x_Utilities
   3_reference/3_x_Auxiliary
   3_reference/3_x_Internal_Architecture
