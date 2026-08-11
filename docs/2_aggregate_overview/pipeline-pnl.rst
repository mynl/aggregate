.. _pipeline pnl:

The P&L Computation Pipeline
============================

This section describes the profit-and-loss surface: the domain-agnostic kernel in ``_pnl.py``, the insurance builders in ``_pnl_builders.py``, and the DecL dispatch in ``underwriter.py`` that joins them. It is the companion to :ref:`pipeline aggregate`, :ref:`pipeline portfolio` and :ref:`pipeline reinsurance`. A P&L consumes what those three produce and contributes no probability machinery of its own, so this section is about accounting over an existing probability space. Class and method names are the stable anchors.

.. _pnl overview:

Overview
--------

A :class:`PnL` is a probability space plus named accounting functionals. Concretely it is two things: a **source**, the stochastic generator, already computed by an :class:`Aggregate`, a :class:`Portfolio`, or a bivariate joint; and an ordered list of **groups**, each a mini-P&L carrying a label, a ``role`` (``'sell'`` or ``'buy'``), and ordered lists of consideration and obligation **legs**. Everything else is derived: per-leg :class:`GridDistribution` rows, per-group totals and results, running nets, the grand total, and the signed exhibits.

The separation of concerns is the central idea of the design, and it is enforced by module boundary. The kernel (``_pnl.py``) is domain agnostic: it never sees the words gross, ceded, or net. All insurance semantics live in the builders (``_pnl_builders.py``) and in the DecL grammar. Insurance is a caller, not a special case. The practical consequence is that a hand-built ledger over an arbitrary source gets the whole reporting surface for free, and that the kernel's invariants can be stated without reference to reinsurance at all.

Four invariants carry the whole design.

Per atom, then group.
    Every leg function is evaluated over the source atoms; totals, results, and running nets are per-atom partial sums of the signed legs. No exhibit is ever arithmetic on marginal distributions, so covariance rides for free. "Means add, standard deviations do not" holds on the sheet automatically, because the standard deviations are computed from the summed per-atom values rather than combined from the parts.
Signed throughout.
    The group ``role`` books each leg into the holder's ledger, so ``sell`` gives ``+consideration, -obligation`` and ``buy`` gives the contra. The ``EX`` column therefore adds straight down the sheet and each row's distribution is built on the signed values. Because :math:`q_p(-X) = -q_{1-p}(X)`, the adverse tail lands where the reader expects for both roles, automatically and with no special casing.
Exact by default.
    A ``bs=0`` leg is an exact irregular :class:`GridDistribution`: group atoms by value, sum probability. Grid distributions need no equal spacing and never touch the FFT machinery, so the exact route is also the cheap one. A ``bs>0`` leg rebuckets onto a regular grid through the shared pushforward machinery and is audited in :attr:`PnL.validation_df`.
One 2-D source.
    Any number of ``is2d`` legs may read the single shared joint. An ``is2d`` leg over a 1-D source is an error; a 2-D source with 1-D legs is fine, since they read axis 0. Only one latent dimension ever exists, which is what keeps the atom set shared and the ladder conditional.

The end-to-end data flow::

    build('pnl ...' | 'xpnl ...')   parse to a spec dict
        |
        v
    Underwriter._factory            two-tier classifier -> inner Aggregate
        |                           plus _pnl_recipe. NO PnL yet: build the
        |                           engine first
        v
    build_many update loop          inner.update(log2, bs), the engine's FFT
        |
        v
    Underwriter._snapshot_pnl       dispatch on recipe kind -> one builder
        |
        v
    _pnl_builders.build_*           pick a SOURCE, declare Legs and Groups
        |                           (all insurance semantics live here)
        v
    PnL.__init__
        |
        +- _ledger_plan()           the row template: one source of truth
        +- _source_atoms()          source -> (coords, probs, shape, bs, is2d)
        +- _eval_leg() per leg      signed per-atom magnitudes
        +- _assemble_rows()         derived rows as per-atom partial sums
        |
        v
    summary_df / economic_df        the card and the footing sheet
    economic_ratios_df / legs_df    raw materials for ratio exhibits
    walk_df / evaluation_df         the margin walk, in currency and as ratios
    density_df / validation_df      per-row GDs; rebucketing audit
    evaluate()                      Cherny-Madan breakeven panel

Two faces answer two questions, and they are the same object built two ways. ``pnl`` answers "what is my position?": the consolidated net-in to net-out view, always one group, always the flat three-row card. Ceded economics are netted out and not shown, because a reinsured aggregate's default output is its net. ``xpnl`` answers "how did I get there?": the walk, gross, then each cover, then ``All``. It is a plain multi-group :class:`PnL` carrying the exploded ``(Step, Side)`` card and ``(Step, Side, Label)`` stats sheet. It is a way of building, not a new type.

A P&L is a value object. It holds signed per-atom rows, their exact moments, and the caller's labels, with no retained dependency on the engine that produced the source. The ``source`` is kept as an opaque reference and ``engine`` as a drill-down handle, but nothing on the exhibits is recomputed from either. That is why a P&L is snapshotted only after the inner engine is updated.

.. _pnl declaration:

Declaration and the two-tier classifier
---------------------------------------

The DecL statement
~~~~~~~~~~~~~~~~~~

The grammar rules are ``pnl_out`` and ``xpnl_out`` in ``decl.lark``; they share an identical body::

    pnl  NAME [as label] <premium> less <engine> [less <expenses>]
             [peel <direction>] [trailer]
    xpnl NAME [as label] <premium> less <engine> [less <expenses>]
             [peel <direction>] [trailer]

The premium head takes three forms: a fixed amount (``1000 premium``), ``inherit premium`` (copy the engine's technical premium, an error if it has none), or ``retro <collar> premium`` (a collared affine map of net account loss, so the consideration itself is stochastic). The engine is a complete stochastic object, written out or referenced: an inline ``agg NAME <body>``, an inline ``port PNAME <units>``, or an ``agg.NAME`` or ``port.NAME`` reference. The two inline forms are self-contained, which is what a program has to be to travel between sessions; a reference resolves only against the underwriter that holds the name. The expense clause carries one or more groups of ``fixed``, ``premium`` and ``loss`` basis terms.

``peel`` is accepted by the grammar on both kinds precisely so that ``pnl ... peel`` reads as a semantic error rather than a parse error: a consolidated view has no steps to peel.

Two structural refusals are worth knowing. An ``xpnl`` over a ``port.NAME`` engine is rejected, because a portfolio total has no cession structure to explode. And ``pnl`` units inside a ``port`` are rejected: book-level P&L is deferred, since a loss-sensitive consideration must be netted per unit before combining, which loses unit premium attribution.

Build the engine first, then snapshot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Underwriter._factory`` does not build a :class:`PnL`. It builds the inner pure-loss :class:`Aggregate`, since the loss structure is merged into the same spec by the parser, and attaches a recipe dict as ``inner._pnl_recipe``. ``build_many`` runs its normal update loop over the inner engine, and only then calls :meth:`Underwriter._snapshot_pnl`, which dispatches the recipe to one builder.

The ordering is forced by the value-object contract. A P&L reads per-atom values off a computed density, so there is nothing to read until the engine has run its FFT. Deferring the snapshot also means the P&L never has to hold an engine reference in order to stay correct.

The recipe carries ``kind``, ``is_tower`` (``xpnl`` or not), the resolved ``econ`` economics dict, ``expense_spec``, ``consideration`` and its label, ``loss_label``, an optional ``peel`` direction, and ``trailer_meta``, the statement's own ``note``, ``tags``, ``hints`` and ``doc``. Trailer metadata is read from the recipe rather than from the engine, because a ``port.NAME`` engine carries the portfolio's metadata, which is not the P&L's.

The two-tier classifier
~~~~~~~~~~~~~~~~~~~~~~~

The occurrence tier is classified as none, guaranteed cost, or reinstatements, and the aggregate tier as none, guaranteed cost, or feature, **independently**. Classifying them together, in a single-kind chain, once let a variable-rating feature branch shadow a reinstatements clause and mis-scope an inuring occurrence program.

Recipe kinds and where they route:

===============  ===================================================================
``kind``         Face selection in :meth:`Underwriter._snapshot_pnl`
===============  ===================================================================
``plain``        :func:`build_plain_pnl`; ``xpnl`` gets the same one-group ledger
                 presented as a one-step walk
``gcn``          :func:`build_consolidated_pnl` (pnl), :func:`build_xpnl_walk`
                 (xpnl), :func:`build_xpnl_peel` (xpnl with ``peel``)
``var``          :func:`build_variable_pnl` with ``walk=is_tower``; a retro ``xpnl``
                 errors, there being no cession to walk
``reins``        :func:`build_reinstatement_source`, then
                 :func:`build_reinstatement_pnl` with ``walk=is_tower``
``port_plain``   :func:`build_plain_pnl` over the portfolio's net-net total density
===============  ===================================================================

Ceded-premium clauses (``deposit``, ``rol``, ``rate``, ``cede``) resolve to per-side economics: ``pc_occ``, ``pc_agg``, ``c_occ``, ``c_agg``, plus ``gross`` and ``ceded``. A side that has cession layers but no premium clause books at zero ceded premium and warns once with :class:`ZeroPremiumCessionWarning`. Reinsurance presence, not economics presence, drives the face. There are two exemptions: a feature-decorated aggregate side owns its own premium slot, and a reinstated occurrence layer requires a base premium clause, an error rather than a warning, because the reinstatement schedule is defined relative to it.

.. _pnl source:

Source selection
----------------

Choosing the source is the builders' single most consequential decision, because it fixes what is measurable and therefore which ladder the sheet can carry. The rule is the simplest sufficient object: the smallest thing on which every declared leg is a function.

Guaranteed-cost marginals
~~~~~~~~~~~~~~~~~~~~~~~~~

Every perspective is read as an exact aggregate marginal off :attr:`Aggregate.reins_density_df` (see :ref:`pipeline reinsurance`), wrapped as an irregular :class:`GridDistribution` by ``_marginal_gd``:

====================  ============================  =========================================
Perspective           ``reins_density_df`` column   Used by
====================  ============================  =========================================
``gross``             ``p_agg_gross``               ``xpnl`` walk base when there is no
                                                    occurrence program
``ceded_occ``         ``p_agg_ceded_occ``           peeled and stitched occurrence rows
``net_occ``           ``p_agg_net_occ``             consolidated ``pnl`` when there is no
                                                    aggregate cover
``ceded_agg``         ``p_agg_ceded``               peeled and stitched aggregate rows
``net_agg``           ``p_agg_net``                 consolidated ``pnl`` when an aggregate
                                                    cover exists
====================  ============================  =========================================

The consolidated ``pnl`` reads the deepest net marginal: net of aggregate when an aggregate cover exists, otherwise net of occurrence. That is what comes out of the engine, and it is the whole point of the consolidated face.

When a 1-D marginal is not enough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A guaranteed-cost ``xpnl`` walk over an occurrence program cannot use any 1-D source. The ceded-occurrence aggregate is not a function of the gross aggregate: the random claim count decouples them, so two programs with the same gross aggregate can cede different amounts. A walk whose columns must foot needs both quantities on the same atom, which means a joint.

The builder therefore calls :meth:`Aggregate.occ_bivariate` with ``views=('gross', 'ceded')`` and uses ``biv.bivariate`` as the source. Construction is :func:`build_netceded_joint` in ``bivariate.py``: each claim's severity mass is placed at the exact 2-D point :math:`(x, c(x))` on the cession curve, then the univariate compound algorithm runs with the transforms swapped to 2-D, :math:`\mathrm{iFFT2}(\mathrm{freq\_pgf}(\mathrm{FFT2}(S)))`, with ``padding=1``. It is fast for the same reason the 1-D engine is fast: one FFT2 over a grid of roughly :math:`2^{20}` cells.

A subsequent aggregate cover is then a 2-D leg on the same joint, recovering :math:`g(\max(L - C, 0))`. The clip at zero matters only for off-support cells where :math:`C > L`, which carry essentially no mass but still evaluate.

Reinstatements build their own joint through :func:`build_reinstatement_source`, over :math:`(L, R)` with :math:`L` the gross loss and :math:`R` the unlimited ceded amount, so that the stochastic ceded premium :math:`D + h(R)` is measurable. Both faces of a reinstatement program ride that one joint, which is why the consolidated ``pnl`` and the ``xpnl`` walk agree exactly rather than approximately.

Grid adequacy on a joint
~~~~~~~~~~~~~~~~~~~~~~~~

The joint runs on a budget-sized common bucket size, coarser than the engine grid. ``check_joint_grid_adequacy`` warns with :class:`CoarseJointGridWarning` when a treaty kink region spans fewer than ``JOINT_KINK_MIN_BUCKETS`` (20) buckets of the joint. Two kink scales are checked: the occurrence fill width ``y = terms.limit`` on axis 1, and each finite aggregate-cover layer width ``share * limit`` resolved no better than ``max(bs_x, bs_y)``.

This warning covers the one error mode the internal bookkeeping cannot see. A kinked map on too few buckets carries a Jensen-type :math:`O(bs)` bias: mass bucketing and a nonlinear map do not commute, and the deficit accounting is blind to it because no mass is lost.

.. _pnl ledger:

The ledger, legs and groups
---------------------------

:class:`Leg`
~~~~~~~~~~~~

One declared cash flow: a label and a map over the source atoms.

==========  =========================================================================
Field       Meaning
==========  =========================================================================
``label``   The ledger row key, directly. Ledger order is declaration order, and
            duplicate labels raise
``func``    A constant, a vectorized ``f(x)`` over axis 0, or ``f(l, r)`` over the
            joint when ``is2d``
``bs``      ``0`` for the exact irregular GD; ``> 0`` for a mean-preserving rebucket
            onto a regular grid
``is2d``    ``func`` reads both axes of the shared joint
``kind``    One of ``LEG_KINDS``: ``premium``, ``loss``, ``expense``, ``recovery``,
            ``commission``
==========  =========================================================================

Magnitudes are written as the caller thinks of them; the group's ``role`` supplies the sign. ``kind`` is accounting metadata the ledger never reads: nothing in evaluation touches it, but :attr:`PnL.economic_ratios_df` needs it to separate loss from expense inside a group's obligation, and :attr:`PnL.legs_df` reports it. An unclassified obligation leg folds into ``L``, so a ledger that declares no expense legs correctly reports ``E = 0`` rather than guessing.

:class:`Group`
~~~~~~~~~~~~~~

One mini-P&L: a label, a ``role``, and ordered consideration and obligation legs, with at least one leg between them. A cession is declared conceptually, the way a reinsurer would write it, so consideration is the ceded premium and obligation is recovery plus commission, and the ``buy`` role books it as ``-premium / +recovery``. Declaring the economics once and letting the role handle the sign is what keeps a cession's ratios conventionally signed on ``economic_ratios_df`` without any special casing.

``margin_label`` names the group's own result row under ``Margin``. Only the direct (``sell``) block of a ledger that buys something reads it, and only then does it appear: the subject business names its own margin the way it already names its loss leg, defaulting to ``'Gross'``.

Expense legs
~~~~~~~~~~~~

``_resolve_expense_split`` splits a DecL ``expense`` spec into one ``(name, scalar, loss_rate)`` per expense group, where ``and``-joined terms make one group and juxtaposed groups stay separate. Within a group, ``fixed`` and ``premium`` terms are deterministic and fold into ``scalar``; a ``loss`` term is loss adjustment expense, a fraction of the actual loss.

``_expense_legs`` then builds one obligation leg per group. When the loss is an on-source observable the leg is the stochastic ``rate * x + scalar``, a scaled-loss distribution rather than a point mass. When the source is a net marginal the gross loss is not measurable there, so the loss-basis part degrades to the deterministic ``rate * E[gross loss]``. The reinstatement consolidated face does not pay that cost, because axis 0 of its joint carries the gross loss.

Colliding default leg names are de-duplicated with a ``(n)`` suffix against the names already taken.

``_ledger_plan``, one row template for every route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_ledger_plan(groups, result_name, tier_spans)`` returns the ledger as ``(label, kind, payload)`` triples, and it is the single source of truth for the row set that every evaluation route materializes. Three different routes build rows three different ways; they cannot drift, because they all walk this one plan.

Per group, in order: consideration legs, the group total consideration (only when there is more than one leg), obligation legs, the group total obligation (likewise), and the group result, which is its step delta. In a multi-group ledger a running-net row (``net through <group>``) follows each group after the first, per-group total and result rows are qualified by the group label, and the sheet closes with the grand rows: ``total consideration``, ``total obligation``, the grand result, and ``total impact``, which is the grand result against the first group's result.

Row kinds and payloads: ``'leg'`` takes ``(gi, side, li)``; ``'group_total'`` takes ``(gi, side)``; ``'group_result'`` and ``'running_net'`` take ``gi``; ``'tier_total'`` takes ``(lo, hi, side)``; ``'tier_result'`` takes ``(lo, hi)``; ``'grand_total'`` takes ``side``; ``'grand_result'`` and ``'total_impact'`` take ``None``. Payloads are hashable because ``(kind, payload)`` is the key into ``PnL._by_kind``, the structural access path the fixed card uses, which reads rows by ledger role rather than by label.

``tier_spans`` names contiguous spans of groups that earn their own subtotal block, emitted after the last group they span. A layer-peeled walk passes one span per reinsurance tier, so a peeled tower still shows the whole occurrence and whole aggregate program. A span of fewer than two groups emits nothing, mirroring the group totals: with one group the group's own rows already are the subtotal.

Every route raises on an unknown row kind rather than falling through a bare ``else``. Three such fallthroughs once silently booked an unknown kind as the impact row.

.. _pnl evaluation:

Evaluation, three routes
------------------------

Route 1, per atom, in memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default. ``_source_atoms`` reduces the source to ``(coords, probs, shape, bs, is2d)``. It resolves four shapes: a bivariate (axes 0 and 1 plus a 2-D ``density``), a :class:`GridDistribution` (``x`` and ``p``), anything exposing ``density_df`` such as an :class:`Aggregate` or :class:`Portfolio`, and a bare ``(values, probs)`` pair. ``coords`` is ``(x,)`` for a 1-D source and ``(axis0[:, None], axis1[None, :])`` for a joint, so the same ``_eval_leg`` call site serves both.

The shared probability vector is normalized to sum to exactly 1. A source density that clips a little tail mass would otherwise give a constant leg a spurious variance of :math:`c^2 \Sigma p (1 - \Sigma p)`. A P&L is a probability distribution and the clipped tail is a discretization artifact, so renormalizing is the honest move. A constant leg additionally short-circuits to ``sd = 0`` exactly in ``_moments_of``, and a break-even mean (:math:`|\mu| \le` ``VALIDATION_NOISE``) returns ``cv = nan``, because a CV is not meaningful when the mean is indistinguishable from zero.

``_assemble_rows`` then materializes the plan. Every derived row is a per-atom partial sum: a group total sums its side's signed leg values, a group result adds the two side totals, a running net is the prefix sum of group results, a tier row restricts the same sums to a group span, and ``total impact`` is ``net_after[-1] - egs[0].result_values``. Because these are vector sums over atoms rather than convolutions of marginals, the dependence between rows is exact by construction and costs nothing.

Route 2, massive one-sweep pushforward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a disk-backed :class:`MassiveBivariateDistribution` source there are no in-memory atoms. ``_init_massive`` evaluates the entire ledger in one :meth:`pushforward` band sweep. Every declared leg must carry an explicit ``bs > 0``, because the sweep scatters onto regular output grids and only the caller knows the output scale.

The critical detail is that each derived row is pushed as its own signed-sum function, never as a sum of already-bucketed legs, with the coarsest constituent ``bs``. That is what makes ``mean(result) == sum(signed leg means)`` hold exactly. Exact means and standard deviations come back from the sweep's streamed audit; skewness is read off the realized grid, because the exact third moment is not streamed. Constants stay constants and the sweep short-circuits them before touching the disk.

Route 3, marginal-stitched
~~~~~~~~~~~~~~~~~~~~~~~~~~

An internal route, not a public construction surface. Every plan row arrives from the builder as its own exact :class:`GridDistribution` plus exact mean and standard deviation, with no shared atoms. It exists for one situation: two or more peeled occurrence layers.

The reason is structural. Occurrence layers are not disjoint intervals on one axis. The ceded-occurrence aggregate is not a function of the gross aggregate, so splitting :math:`m` layers per atom would need an :math:`(m+1)`-axis joint, and the total ceded axis of the 2-D joint cannot be decomposed after the fact, because :math:`\sum_i \mathrm{ceder}(X_i)` does not determine the per-layer allocation. Each row is, however, the aggregate of a deterministic per-claim severity transform, so its marginal is exactly computable with one FFT (``_sev_transform_marginal``), and an aggregate-tier row is a plain pushforward of the aggregate subject with no FFT at all (``_agg_transform_marginal``).

The consequences are visible on the sheet and are meant to be. The ``EX`` column still foots exactly by linearity, since the default ``'linear'`` rebucketing preserves first moments, but the dispersion columns are marginal, so the ladder carries plain ``P`` headers; ``+`` composition is unavailable; and loss-basis LAE degrades to the deterministic form. One row, ``total impact``, has no law at all: its two sides ride different marginals, so it is a :class:`_DeltaRow` whose every statistic is the plain difference of two row statistics rather than a statistic of the difference.

Layer peeling picks its route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``xpnl ... peel top-down`` and ``peel bottom-up`` book one group per reinsurance layer instead of one per tier, so each layer shows its own ceded premium, commission and marginal impact, and ``net through <layer>`` builds the program up one layer at a time. Occurrence layers peel before aggregate ones, preserving the tier walk's step order; within a tier ``top-down`` introduces the highest-attaching layer first. Zero-share layers are gap fillers, structural rather than cessions, and get no step.

The two tiers are not symmetric, and that asymmetry selects the route. Aggregate layers are disjoint intervals on the one aggregate subject axis, so single-layer ceders sum identically to the cumulative ceder and each layer is simply another per-atom leg on the shared source: the kappa ladder survives and every column foots. Occurrence layers are not, for the reason given in :ref:`pnl source`. At most one occurrence step therefore takes the per-atom route (``_peel_per_atom``), and two or more take the stitched route (``_peel_stitched``).

Composition
~~~~~~~~~~~

``pnl_a + pnl_b`` concatenates two group ledgers over the same source, checked by identity or by equality of atoms and probabilities. The second ledger's groups shift right by the first ledger's count, and its tier spans shift with them. A stitched P&L cannot compose, having no shared atoms to check against.

For the genuinely independent case there is :func:`stack_marginal_pnls`, which stacks separate perspectives into one frame. Each contributes its grand-result statistics as a row and ``impacts`` add per-statistic delta rows. Means add across the stack; standard deviations and percentiles are per row. The ladder there is marginal by construction: independent perspectives share no joint, so there is nothing to condition on.

.. _pnl kappa ladder:

The scenario (kappa) ladder
---------------------------

The calculation
~~~~~~~~~~~~~~~

The scenario ladder is the P&L's one genuinely new piece of mathematics, and it is the library's kappa function applied to a ledger.

The ledger is evaluated per atom over one shared source: every row :math:`X_i` (leg, total, running net) and the grand result :math:`M = \sum_i X_i` are vectors of signed values on the same atoms :math:`\omega` with probabilities :math:`p(\omega)`. For each ladder point :math:`q` in ``PERCENTILE_LADDER`` the scenario is anchored on the margin: let :math:`m_q = \inf\{x : \Pr(M \le x) \ge q\}` be the lower :math:`q`-quantile of :math:`M`, read straight off the grand result's :class:`GridDistribution`.

Because the grid distribution's support is the set of exact result values, the event :math:`A_q = \{\omega : M(\omega) = m_q\}` is a nonempty exact slice, tested with float equality and no tolerance. Each cell of column :math:`\kappa_q` is the conditional mean of its row over that slice:

.. math::

    \kappa_q(X_i) \;=\; \mathsf E\left[X_i \mid M = m_q\right]
    \;=\; \frac{\sum_{\omega \in A_q} X_i(\omega)\, p(\omega)}
                {\sum_{\omega \in A_q} p(\omega)} .

The column is the kappa function :math:`x \mapsto \mathsf E[X_i \mid M = x]`, the same object as :class:`Portfolio`'s ``exeqa_*`` columns (see :ref:`portfolio update`), evaluated at the margin's own quantiles rather than on the whole grid.

Why it foots
~~~~~~~~~~~~

By linearity of conditional expectation every column foots exactly: legs add to totals and totals add to :math:`\kappa_q(M) = m_q`, so the grand-result cell is automatically its own marginal quantile. There is no special case for it, and no reconciliation step. That is precisely the property the card's marginal percentiles cannot have, which is why the two exhibits are deliberately different in kind.

Direction
~~~~~~~~~

Signs follow the payoff convention, left tail bad. :math:`\kappa_{01}` is the adverse state, full stop, and a loss-sensitive premium correctly shows high in the bad columns. That is deliberately reversed from the usual actuarial loss view, where the right tail is bad, and the reversal is uniform: it applies to every row of every ledger, because every row is already in payoff orientation.

Two subtleties, documented rather than engineered away
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where the margin is a monotone (decreasing) function of the underlying loss, which covers any plain book, the level sets :math:`\{M = m_q\}` and :math:`\{L = \ell_{1-q}\}` coincide, so the column is a gross-loss scenario at the complementary quantile. Where it is non-monotone, as with a swing collar or a slide or any hump, the cell is the exact mean over the full level set :math:`\{M = m_q\}`. That is well defined but subtler to read: the state is a mixture of loss levels that happen to produce the same bottom line.

A constant grand result, a fully hedged position, makes the conditioning event everything, so every cell collapses to its own ``EX``. On a 2-D source the atoms are the joint's cells :math:`(L, R)` and everything above applies unchanged.

When the ladder is marginal instead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared-source rule: scenario columns exist exactly when the ledger shares one source. The header is the flag. A :math:`\kappa` column means conditioning happened; a plain ``P`` column is a marginal quantile of that row and nothing more. Two routes keep marginal ladders: the massive one-sweep route, because conditioning needs the joint per atom and the grand-result quantiles before indicator-weighted means can accumulate, so it needs a second band sweep; and the stitched route, which has no joint at all. Positional access or ``df.filter(like='κ')`` avoids having to type the glyph.

.. _pnl reporting:

The reporting surface
---------------------

Nine objects plus the evaluation panel, on :class:`PnL`:

==========================  ============  ==================================================================
Object                      Shape         Role
==========================  ============  ==================================================================
``summary_df``              DataFrame     the card: fixed rows, marginal range percentiles
``economic_df``             DataFrame     the sheet: every ledger row, kappa or P ladder, foots
``economic_ratios_df``      DataFrame     amounts and ratios per block: raw materials, not a card
``legs_df``                 DataFrame     one row per declared leg, the only place ``kind`` surfaces
``walk_df``                 DataFrame     the margin walk in currency, at EV and in the 1-in-100 state
``evaluation_df``           DataFrame     the same walk as ratios: shares, combined, return on capital
``density_df``              OrderedDict   ``{row label: GridDistribution}``, no shared axis
``validation_df``           DataFrame     Est-against-EX audit of every ``bs > 0`` leg
``info``                    str           fixed-layout summary, one row per line, ``n/a`` where absent
``evaluate()``              DataFrame     Cherny-Madan breakeven acceptability panel
==========================  ============  ==================================================================

``summary_df``, the card
~~~~~~~~~~~~~~~~~~~~~~~~

Fixed rows that never vary with the ledger. A single group gives a flat three-row card, ``Consideration``, ``Obligation`` and ``Margin``, read from the grand references, the exact structural mirror of the flat :attr:`Aggregate.summary_df`. Multi-group gives one ``(Step, Side)`` block per step plus a closing ``'All'`` block, mirroring the per-unit blocks of :attr:`Portfolio.summary_df`::

    ('Gross',     'Consideration' | 'Obligation' | 'Margin')
    ('ceded occ', 'Consideration' | 'Obligation' | 'Margin' | 'Net')
    ('All',       'Consideration' | 'Obligation' | 'Margin' | 'Impact')

Per step, ``Margin`` is the group result and ``Net`` the running net through the step, omitted on the first step where the two coincide. A layer-peeled walk adds a three-row tier block (``'All occurrence'``, ``'All aggregate'``) after the last step it spans, with no ``Net`` row, because the running net through the tier is already the last layer's ``Net``.

Rows scale with steps, never with legs. That is the fixed-shape contract, and it is what lets the card be a card. Columns are ``EX``, ``SD``, ``CV``, ``Skew``, ``P01``, ``Median`` and ``P99``.

The percentiles here are marginal quantiles of each row's own distribution. The card answers "how big is each total", a question about range, so its percentile cells do not foot down the card. That is a property of quantiles, not an error, and it is stated in the ``_repr_html_`` intro rather than left for the reader to discover from a column that does not add up. The card is currency only: ratios live in ``economic_ratios_df``, per the reporting rule that a column carries one unit.

``_card_side_row`` resolves a side total without any extra computation, and on the massive route without extra sweep keys: with more than one leg the ``group_total`` row exists, with exactly one the leg row is the side total, and with none the total is a constant zero.

``economic_df``, the footing sheet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ledger row, in ledger order, indexed by ``_side_index``: two-level ``(Side, Label)`` for a single group, three-level ``(Step, Side, Label)`` on a tower. Columns are ``EX``, ``SD``, ``CV`` and ``Skew`` plus the full nine-point ladder. ``EX``, ``SD``, ``CV`` and ``Skew`` are row properties and stay marginal whatever the ladder does.

The ``Side`` level buckets every row: legs and totals under ``Consideration`` or ``Obligation``, and every result-flavored row (group results, running nets, the grand result, total impact) under ``Margin``. The ``Label`` level carries the presentation name.

A ledger that buys something distinguishes three margins that all used to read ``Total``: the grand result and grand totals read ``Net``, a cession's own result stays ``Total``, and the ``sell`` group's own result takes its ``margin_label``, defaulting to ``'Gross'``. Both distinctions are gated on the ledger actually containing a ``buy`` group, because only then is there anything to be direct or net of. A plain single-group ``pnl`` is one ``sell`` group whose legs are already net, and a ledger merging two sold books has no net to take, so both keep ``Total`` throughout.

The flat plan labels stay the canonical row keys everywhere else, in ``density_df``, ``validation_df`` and the sweep result keys; the index is presentation only.

``economic_ratios_df`` and ``legs_df``, the raw materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``economic_ratios_df`` reports one row per block: each group, each tier subtotal, and ``'All'`` on a multi-group ledger. It is deliberately unformatted and absent from ``qd`` and the notebook repr, because it is the frame to slice, unstack and build presentation tables from, in the spirit of :meth:`Portfolio.analyze_distortions` and its ``pricing_df``.

Columns are ``P``, ``L``, ``E``, ``C``, ``M``, ``LR``, ``ER``, ``CR``, ``E_LR``, ``E_ER``, ``E_CR``, ``P_share`` and ``M_share``. The four amounts are signed in the gross direction: consideration as booked, obligations negated. A cession's ceded premium and recovery are therefore both negative, which makes every ratio built from them come out with its conventional sign, makes the amounts add across blocks, and makes ``M == P - L - E - C`` hold identically, since it is the signed row sum. ``L``, ``M``, ``P`` and ``LR`` keep the ``pentagon.PENTAGON_STATS`` spelling so a P&L ratio frame concatenates and diffs against a pricing frame; ``E``, ``C``, ``ER`` and ``CR`` extend it, and ``Q``, ``a``, ``PQ`` and ``ROE`` have no meaning on a ledger.

``LR``, ``ER`` and ``CR`` are ratios of means, the convention of a rate filing, re-derived from the row's own amounts and never averaged from the blocks below. ``E_LR``, ``E_ER`` and ``E_CR`` are the same three as means of ratios. The pair parts company exactly when premium is random and correlated with loss, which is what a retro-rated account, or a swing, slide or profit-commission cession, is. Availability turns on whether the denominator is random, not on whether a joint happens to exist: a constant premium factors straight out of :math:`\mathsf E[X/P]`, so the mean-of-ratio columns are exact on every route and simply repeat the plain ratios; a random premium needs atoms to average over, so it reads ``nan`` on the stitched and massive routes and ``nan`` where any atom carrying probability has a vanishing premium. There is never a silent fallback to the ratio of the means.

``legs_df`` is the itemized companion: one row per declared leg, with columns ``Step``, ``Side``, ``Label``, ``kind``, ``EX`` and ``SD``, matching the ``economic_df`` level names. Derived rows are absent by design, being sums of these.

``density_df`` and ``validation_df``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``density_df`` is an ordered ``{row label: GridDistribution}`` covering the declared legs, the group results and tier results on a tower, and the grand result. It is deliberately not a single stapled frame: each row keeps its own irregular, exact grid, so there is no lossy rebucketing onto a shared axis. Its main customer is :meth:`PnL.plot`, which iterates and reads each grid distribution's Series view.

``validation_df`` audits every ``bs > 0`` leg: ``EX`` is the exact signed mean straight off the atoms, ``Est`` the mean of the rebucketed grid distribution, plus ``abs_err`` and ``rel_err``. The linear scheme matches means, so it reads very close, but it is visible. Exact (``bs = 0``) legs are absent, and there is no reaching through to the source's own validation, which may not exist.

``evaluate()``, the acceptability panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A P&L is evaluated, not priced. The question is which distortion drives the risk-adjusted margin to zero, that is, the breakeven stress the position survives, indexed by the family-agnostic Cherny-Madan ``gini_p``. Families default to ``EVAL_FAMILIES`` (``ph``, ``wang``, ``dual``, ``tvar``; ``ccoc`` is excluded).

Every margin row is evaluated, not just the grand result, so a tower reads as a story: the gross deal, each reinsurance layer as a position in its own right, and the running net after each purchase. Reading ``gini_p`` down the ``net through ...`` rows is watching what buying cover does to the deal.

A ceded layer is evaluated from the seller's side, because the buyer's margin on it is negative by construction, premium paid less recoveries received, and has no breakeven of its own. ``_row_role`` assigns the role: a group result takes its own group's role; a tier subtotal reads ``'buy'`` when every group it spans buys and ``'sell'`` when every one sells, with a mixed span reading ``'net'``; a running net and the grand result are a netting of buying against selling, so they read ``'net'`` and are evaluated as booked. That is what makes the panel a buy decision: a layer whose ``gini_p`` sits above the ``net`` row immediately over it is priced above the holder's own acceptability, so buying it lowers the net.

Only positions appear. ``total impact`` is deliberately absent: it is a margin but not a position, being the difference between two of them. Nobody holds it, so the stress it survives is not a question with an answer. The ceded program as a position is already in the sheet, under the tier subtotal rows. The panel returns tidy (long) form with ``(Step, distortion)`` rows and ``role``, ``param_name``, ``param``, ``gini_p``, ``error`` and ``status`` columns; ``.unstack('distortion')`` gives the wide comparison.

Construction introspection
~~~~~~~~~~~~~~~~~~~~~~~~~~

``construction_description`` is one paragraph, giving route, source and group count, and ``construction_explanation`` is the full story: the engine and its clauses, the economics resolution (``deposit 2000 -> pc_occ = 2000``), which source each row reads, the booking signs, whether the ladder is kappa or ``P`` and why, and any ignored clauses. Both are recorded by the builder that made the object, since the builder alone knows the why; a hand-built kernel P&L falls back to a generic structural narrative, so neither property is ever absent.

``construction_explanation`` closes with an executable replay block: the literal ``PnL(...)`` call that reproduces the object from its retained group specs and source reference. Constant legs render verbatim; function legs render as ``<fn>`` placeholders, a lambda having no literal form.

.. _pnl trust:

What makes the 2-D route trustable
----------------------------------

Three layers: runtime guards, one-line cross-checks, and one-time adjudication.

Runtime, every build
~~~~~~~~~~~~~~~~~~~~

#. Mass and deficit bookkeeping. ``build_netceded_joint`` computes ``deficit = 1 - sum(density)`` and carries it in the joint's ``meta``, the same convention as the 1-D engine's clipped-tail reporting. ``padding=1`` means no FFT wrap-around.
#. Structural correctness of the dependence. Mass cannot appear off the feasible region: each claim's atom sits exactly on the cession curve, and the one frequency PGF wraps the 2-D severity transform, so the shared claim count and the shared mixing are exact by construction rather than approximated.
#. :class:`CoarseJointGridWarning`. ``check_joint_grid_adequacy`` fires when a treaty kink region spans fewer than 20 joint buckets, the one error mode the internal bookkeeping cannot see.
#. Theory-against-realized audit on demand. The returned holder is a :class:`BivariateAggregate`, and its ``summary_df`` is the Portfolio-shaped validation frame, exact moments from ``_netceded_theory`` against realized, with noise-aware ``Err`` columns.

One line away, any time
~~~~~~~~~~~~~~~~~~~~~~~

The ``pnl`` face reads the exact 1-D marginals, so ``pnl.mean`` against the walk's ``('All', 'Margin', 'Total')`` ``EX`` bounds the joint-grid error for your actual program. Any single walk row against its ``reins_density_df`` marginal mean does the same per row. The tests do exactly this, at roughly 1e-3 relative on linear rows and 1e-2 on kinked aggregate tiers.

One-time adjudication, now pinned in tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Joint mass came out at 1 minus 3e-11, with zero off-support mass, joint marginal means equal to engine means at about 1e-5 relative, and internal accounting identities to 1e-15. A 1.6M-simulation Monte Carlo of the true process matched every leg within 2 standard errors, with the kinked-layer grid bias isolated and quantified at about 0.25% at 25 buckets per layer, which is the origin of the 20-bucket warning floor. The tests are ``test_pnl_consolidated_walk.py``, ``test_composition_matrix.py`` and ``test_reinstatement_decl.py``.

.. _pnl notes:

Notes to remember
-----------------

The kernel never sees insurance.
    Gross, ceded and net are builder vocabulary. If a change to ``_pnl.py`` needs one of those words, it belongs in ``_pnl_builders.py`` instead.
Build the engine, then snapshot the P&L.
    ``_factory`` attaches a recipe; ``_snapshot_pnl`` runs after the update loop. A P&L has no engine dependency to recompute through, so there is nothing to read until the FFT has run.
``_ledger_plan`` is the only row template.
    Three routes materialize it three ways. Any new row kind must be handled in ``_assemble_rows``, ``_init_massive``, ``_init_stitched`` and ``_side_index``, all four of which raise on an unknown kind.
The header is the flag.
    :math:`\kappa` means conditioning happened and the column foots; ``P`` means a marginal quantile of that row, and it does not. The reader never has to know which route built the object.
The card's percentiles do not foot, and that is correct.
    Marginal quantiles never add. ``summary_df`` answers "how big is each total"; ``economic_df`` is the footing sheet. Two exhibits, deliberately different in kind.
Signed in the gross direction on the ratio frame, signed by role everywhere else.
    The ledger books by ``role`` so the ``EX`` column adds down the sheet; ``economic_ratios_df`` re-signs to the gross direction so amounts add across blocks and a cession's loss ratio reads positive. Two conventions, each doing one job.
Ratio of means and mean of ratio are different numbers.
    They agree identically when premium is deterministic and part company exactly when it is random and correlated with loss, which is the whole point of a retro or a swing. ``economic_ratios_df`` reports both rather than picking one.
The occurrence tier forces the joint.
    The ceded-occurrence aggregate is not a function of the gross aggregate. Every design decision downstream, the 2-D walk, the peel route split, the marginal stitch, follows from that one fact.
Normalize the source probabilities.
    Otherwise a clipped tail gives a constant leg a variance, which is nonsense on the face of it and hard to trace back.
