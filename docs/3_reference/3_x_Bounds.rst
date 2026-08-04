Bounds
======

:mod:`aggregate.bounds` computes **pricing and allocation bounds**: the range
of prices (and capital allocations) consistent with a set of market or
no-arbitrage constraints, following the IME 2022 methodology (Mildenhall,
*Similar Risks Have Similar Prices*). Given a risk and a calibration, the bounds
answer "how high / low can this layer's price be?" across all admissible
distortions.

- :class:`~aggregate.bounds.Bounds`, the main driver: distortion-envelope
  pricing bounds for an :class:`~aggregate.Aggregate` or
  :class:`~aggregate.Portfolio`.
- :class:`~aggregate.bounds.AllocationBounds`, the allocation geometry
  (the achievable capital-allocation set).
- :class:`~aggregate.bounds.PricingBounds`, the price-range envelope.

These are reached by submodule import (``from aggregate.bounds import Bounds``);
they are not re-exported at the top level.

.. currentmodule:: aggregate.bounds

.. autosummary::

   Bounds
   AllocationBounds
   PricingBounds

Bounds class
------------

.. autoclass:: aggregate.bounds.Bounds
   :special-members: __init__

AllocationBounds class
----------------------

.. autoclass:: aggregate.bounds.AllocationBounds
   :special-members: __init__, __call__

PricingBounds class
-------------------

.. autoclass:: aggregate.bounds.PricingBounds
   :special-members: __init__, __call__

Module functions
----------------

.. automodule:: aggregate.bounds
   :exclude-members: Bounds, AllocationBounds, PricingBounds
