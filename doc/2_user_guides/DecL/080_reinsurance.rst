.. _2_x_reinsurance:
.. _2_agg_class_reinsurance_clause:

The ``reinsurance`` Clauses
----------------------------

**Prerequisites:**  Excess of loss reinsurance terminology.

Occurrence and aggregate reinsurance can be specified in a way similar to
limits and deductibles. Both clauses are optional. The ceded or net position
can be output. Layers can be stacked and can include co-participations. The
syntax is best illustrated with some examples.

**Examples.** 1. The DecL::

    agg Trucking
    5000 loss 1000 xs 0
    sev lognorm 50 cv 1.75
    occurrence net of 750 xs 250
    poisson

specifies the distribution of losses to the net position on the Trucking policy after a per occurrence cession of the 750 xs 250 layer. This net position could also be written without reinsurance as::

    agg Trucking
    ?? loss
    250 xs 0
    sev lognorm 50 1.75
    poisson

for some level of losses. Running::

    agg Trucking
    5000 loss 1000 xs 0
    sev lognorm 50 cv 1.75
    occurrence ceded to 750 xs 250
    poisson

models ceded losses.

2. The DecL::

    agg WorkComp
    15000 loss
    500 xs 0
    sev lognorm 50 cv 1.75
    poisson
    aggregate ceded to 50% so 2000 xs 15000

specifies the distribution of losses to an aggregate protection for the 2000 xs 15000 (the loss pick) layer of total losses, with occurrences limited to 500. The underlying business could be an SIR on a large account Workers Compensation policy, and the aggregate is a part of the insurance charge (Table L, M).

3. The DecL::

    agg Trucking 5000
    loss 1000 xs 0
    sev lognorm 50 cv 1.75
    occurrence net of 50% so 250 xs 250 and 500 xs 500
    poisson
    aggregate net of 250 po 1000 xs 4000 and 5000 xs 5000

applies two occurrence and two aggregate layers to the Trucking portfolio. The 250 xs 250 occurrence layer  is only 50% placed (``so`` stands for share of), and the second is 100% (by default) of 500 xs 500. The net of the occurrence programs flows through to aggregate layers, 250 part of 1000 xs 4000 (25% placement), and 100% share of the 5000 xs 5000 aggregate layers. The modeled outcome is net of all four layers. In this case, it is not possible to write the net of occurrence using limits and attachments.

All occurrence reinsurance has free and unlimited reinstatements.

The occurrence reinsurance clause comes after severity but before frequency, because you need to know severity but not frequency. The aggregate clause comes after frequency. If frequency is specified using ``dfreq`` the occurrence clause comes before the aggregate clause.

The options for both clauses are:

* ``ceded to`` or ``net of`` determines which losses flow out of the
  reinsurance.
* fraction ``po`` limit ``xs`` attachment describes a partial placement, e.g.,
  ``0.5 so 3 xs 2``.
* participation ``so`` limit ``xs`` attachment describes a partial placement
  by the ceded limit, e.g., ``1 po 3 xs 2``. This syntax is equivalent to
  ``0.333 so 3 xs 2``.

An unlimited cover is denoted ``inf``. Shares of unlimited covers must be expressed as shares, for obvious reasons.

Layers can be stacked using the ``and`` keyword. The initial ``net of`` or ``ceded to`` applies to all layers in the tower.


Underwriters are often interested in layering out losses from ground-up to the
policy limit. For example, a 5M limit may be layered as 250 xs 0, 250 xs 250,
500 xs 500, 1000 xs 1000, and 3000 xs 2000. A tower can be input manually::

    occurrence ceded to 250 xs 0 and 250 xs 250 and 500 xs 500  \
        and 1000 xs 1000 and 3000 xs 2000

(note the Python \ line break). However, since layering is quite common, there
is also a shorthand. They can be entered by specifying just the layer break
points using the ``tower`` keyword::

    tower [0 250 500 1000 2000 5000]

The initial 0 is optional; the tower does not have to start at 0. It does not
have to exhaust the entire policy limit. Towers can be applied to occurrence
and aggregate reinsurance.

See :ref:`reinsurance pricing<2_x_re_pricing>` for more examples, including an
approach to reinstatements.

