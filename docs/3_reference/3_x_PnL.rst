Profit and Loss
===============

.. currentmodule:: aggregate._pnl

A :class:`~aggregate.PnL` is a probability space plus named accounting
functionals: a **source** (the stochastic generator, an
:class:`~aggregate.Aggregate` or :class:`~aggregate.Portfolio`) and an ordered
list of signed **groups**, each holding consideration and obligation
:class:`~aggregate._pnl.Leg` objects. Everything else is derived: per-leg exact
distributions, per-group totals, running nets, the grand total.

The kernel is deliberately domain-agnostic. It never sees the words gross,
ceded, or net; insurance is a caller, not a special case. The DecL builders in
:mod:`aggregate._pnl_builders` translate a program into a source plus signed
groups (see :doc:`3_x_Internal_Architecture`), and the wrapped engine stays
reachable as ``pnl.engine``.

Two faces
---------

``pnl`` and ``xpnl`` are the two DecL kinds, answering two questions:

- **``pnl``, "what is my position?"** The consolidated net-in-to-net-out view:
  always one group, always the flat three-row card. Ceded economics are netted
  out and not shown, so a reinsured aggregate's default output is its net.
- **``xpnl``, "how did I get there?"** The walk from gross through each cover to
  the total: a plain multi-group :class:`PnL` (a way of building, not a new
  type) carrying the exploded ``(Step, Side)`` card and ``(Step, Side, Label)``
  stats sheet. ``peel`` books one group per reinsurance layer.

Both are built through :func:`~aggregate.build`::

    from aggregate import build
    p = build('pnl MyBook 1000 premium less agg.MyAgg')

Reading the sheets
------------------

:attr:`PnL.summary_df` is the **card**: fixed rows whose percentiles are
**marginal** quantiles of each row's own distribution. Marginal quantiles do not
add, so the card's percentile cells do not foot down the card. That is a
property of quantiles, not an error.

:attr:`PnL.economic_df` is the **sheet**: every ledger row, with ``κ`` ladder
columns that are scenario states anchored on the grand result. Column ``κq``
shows every row's conditional mean given the result lands at its ``q``-quantile,
so each column is one internally consistent state and **foots exactly**. The
header is the flag: a ``κ`` column means conditioning happened, while a ladder
with no shared source (the massive one-sweep route,
:func:`~aggregate.stack_marginal_pnls`, a stitched guaranteed-cost tower) stays
marginal under plain ``P`` headers. P&Ls are in payoff sign convention, left
tail bad, so ``κ01`` is the adverse state.

.. autosummary::

   PnL
   Leg
   Group
   stack_marginal_pnls
   LEG_KINDS

PnL and the group ledger
------------------------

.. automodule:: aggregate._pnl

Contract terms
--------------

A **contract term** is a deterministic, vectorized function of a realized loss
quantity that fills one leg of a Gross / Ceded / Net P&L. Pushed forward over
the loss distribution, 1-D on an aggregate basis and 2-D on an occurrence basis,
it makes that leg stochastic. Six features share this shape: reinstatement
premium, retro, swing, slide, profit commission, and corridor.

Five are pure single-leg maps; :class:`~aggregate.contract_terms.ReinstatementTerms`
is the one two-map feature, carrying a premium decorator and the annual-cap loss
transform. The terms objects hang off the wrapped engine
(``pnl.engine.reinstatement_terms``, ``pnl.engine.variable_terms``). They are
reached by submodule import and are not re-exported at the top level::

    from aggregate.contract_terms import SwingTerms

.. currentmodule:: aggregate.contract_terms

.. autosummary::

   ContractTerms
   RetroTerms
   SwingTerms
   SlideTerms
   ProfitCommissionTerms
   CorridorTerms
   ReinstatementTerms

.. automodule:: aggregate.contract_terms
