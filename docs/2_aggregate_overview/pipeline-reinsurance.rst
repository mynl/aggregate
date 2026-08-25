.. _pipeline reinsurance:

The Reinsurance Reporting Surface
=================================

This section inventories every reinsurance object that produces DataFrame output and says, in plain terms, exactly what each cell computes. It is the companion to :ref:`pipeline aggregate`, which covers how the net and ceded densities are computed in :ref:`agg pipeline fft`, and to :ref:`pipeline portfolio`. The pseudo-code is deliberately arithmetic, "sum of loss times probability" rather than measure-theoretic, so that a column can be read off the screen. Method names are the stable anchors.

.. _reins overview:

Overview
--------

Reinsurance enters the pipeline at two points (see :ref:`agg pipeline fft`): occurrence cover acts on ``sev_density`` before the FFT, and aggregate cover acts on ``agg_density`` after it.

Each stage is applied by one engine, :meth:`Aggregate._apply_reins_work`, which takes a subject density and a layer list and returns a small transient frame holding the subject, net, and ceded densities on the model grid. Everything else in this section is re-assembly of those densities into reports: moments, per-layer breakdowns, exposure probabilities. The re-gridding is the mass-preserving scatter :meth:`Aggregate._rebucket_to_grid` (``reins_bucket = 'linear'`` by default, or ``'nearest'``).

The reporting surface is three public objects plus one private helper, defined on :class:`Aggregate` and mirrored end-to-end on :class:`Portfolio`:

=============================  =========  =========================================================
Object                         Shape      Role
=============================  =========  =========================================================
``reins_density_df``           DataFrame  all gcn densities, severity and aggregate, on the grid
``reins_stats_df``             DataFrame  per-layer layering view (Gross / layer.k / Ceded / Net)
``reins_summary_df``           DataFrame  per-stage economic view, the daily driver
``_reins_view_stats``          DataFrame  private; per stage, view and basis, EX against Est
                                          moments; feeds ``reins_summary_df``
``reins_kinds``                str        "None / Occurrence only / Aggregate only / both"
``reins_description``          str        human sentence, "Net of 88% share of 4,000 xs 1,000..."
``reins_occ_plot()``           figure     occurrence log-density and aggregate quantile plot
=============================  =========  =========================================================

The vocabulary used throughout:

Gross
    the very top, before any cover.
Subject
    the input to a stage. For occurrence the subject is gross; for aggregate it is the occurrence output the aggregate program sees, already net or ceded of occurrence, so calling it gross would be wrong.
Output
    the model-output view when occurrence and aggregate pass different kinds, for example ``net of`` occurrence then ``ceded to`` aggregate. It is the ``summary_df`` label for that mixed case.
gcn, fsa
    the gross / ceded / net triple, and the frequency / severity / aggregate triple.
conditional against unconditional
    only the per-layer ``layer.k`` columns of ``reins_stats_df`` are conditional on a loss reaching the layer. Every total (``Gross``, ``Ceded``, ``Net``) and the whole of ``reins_summary_df`` is unconditional.

The gross, subject, net, ceded and output words are centralized as ``REINS_LABEL_*`` constants in ``constants.py``.

.. _reins engine:

The engine: the re-gridding core
--------------------------------

Every reinsurance density in the library comes out of ``_apply_reins_work``. Given a layer list ``[(share, limit, attach), ...]`` and a subject density on the grid ``self.xs``, where ``xs[k] = k * bs``::

    ceder, netter = make_ceder_netter(layers)   # piecewise-linear loss maps
    # ceder(x)  = ceded loss when subject loss is x
    # netter(x) = x - ceder(x)

    for each grid bucket k with subject mass p_subject[k] at loss x = xs[k]:
        this loss is ceded to  loss_ceded[k] = ceder(x)
        and retained as        loss_net[k]   = netter(x)

    # both targets are off-grid; scatter their mass back onto the grid.
    # reins_bucket = 'linear' splits across the two bracketing buckets and
    # preserves E[loss] exactly; 'nearest' rounds. See _rebucket_to_grid.
    p_ceded = _rebucket_to_grid(loss_ceded, weights=p_subject)
    p_net   = _rebucket_to_grid(loss_net,   weights=p_subject)

The returned transient frame has one row per grid bucket, with columns in this order:

==============  ====================================================================
Column          Meaning
==============  ====================================================================
``loss``        bucket loss ``xs[k]``, also the index
``p_subject``   subject density, the input to this stage
``loss_net``    ``netter(xs[k])``, the exact value this bucket's loss is retained to
``loss_ceded``  ``ceder(xs[k])``, the exact value this bucket's loss is ceded to
``p_net``       net density, subject mass re-gridded through ``netter``
``p_ceded``     ceded density, subject mass re-gridded through ``ceder``
==============  ====================================================================

The key identity is a **mass split, not a per-bucket split**. ``p_net + p_ceded`` is not ``p_subject`` bucket by bucket, because a subject loss at :math:`x` sends its mass to two different buckets, its net image and its ceded image. What is preserved is total mass, since each marginal sums to :math:`\sum p_{\mathrm{subject}}`, and, under ``'linear'``, the first moment:

.. math::

    \sum \mathrm{loss} \cdot p_{\mathrm{net}}
      + \sum \mathrm{loss} \cdot p_{\mathrm{ceded}}
      = \sum \mathrm{loss} \cdot p_{\mathrm{subject}}.

The ``loss_net`` and ``loss_ceded`` columns are exact real values, the ceder and netter evaluated at the grid loss, while ``p_net`` and ``p_ceded`` are those images rebucketed back onto the grid. The frame is transient: only the rebucketed densities are kept, on ``sev_density_{gross,ceded,net}`` for occurrence and ``agg_density_{gross,ceded,net}`` for aggregate. The exact, pre-bucket moments are recomputed on demand from the retained step functions ``occ_ceder``, ``occ_netter``, ``agg_ceder`` and ``agg_netter``.

There are two callers. :meth:`apply_occ_reins` takes ``sev_density`` as the subject, sets ``sev_density_gross/net/ceded``, and points ``sev_density`` at the requested view. :meth:`apply_agg_reins` takes ``agg_density`` as the subject, sets ``agg_density_gross/net/ceded``, and re-runs the FFT on the chosen view into ``ftagg_density``.

.. _reins objects:

The reporting objects
---------------------

``reins_density_df``
~~~~~~~~~~~~~~~~~~~~

The reinsurance analogue of ``density_df``: one row per grid bucket, with consistent columns regardless of which programs are present. A missing stage contributes the no-cession values, ceded = 0 and net = subject::

    loss              = xs

    # severity level, from the occ stage; equals sev_density when no occ
    p_sev_gross       = sev_density_gross
    p_sev_ceded       = sev_density_ceded
    p_sev_net         = sev_density_net

    # aggregate of each occ severity view (a fresh FFT of each)
    p_agg_gross       = FFT_aggregate(p_sev_gross)   # the true gross aggregate
    p_agg_ceded_occ   = FFT_aggregate(p_sev_ceded)
    p_agg_net_occ     = FFT_aggregate(p_sev_net)

    # aggregate cover: subject = aggregate of the requested occ output
    p_agg_subject     = what the agg cover sees (= p_agg_gross when no occ)
    p_agg_ceded       = agg_density_ceded
    p_agg_net         = agg_density_net

Two names enforce the gross-against-subject convention. ``p_agg_gross`` is the FFT of the gross severity, which is the true gross aggregate, and ``p_agg_subject`` is the aggregate-cover input, which is the aggregate of the occurrence output. All aggregate columns come from :meth:`_fft_aggregate`, so the fixed-1 and zero-risk shortcuts are honored identically to the main path. ``reins_density_df`` is the densities-of-record frame the other objects are computed from, and :meth:`reins_occ_plot` reads from here.

``_reins_view_stats``, the EX against Est frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The per stage, view and basis moment frame, shaped like ``stats_df`` but indexed by stage, view and basis instead of by component. It is lazily built, cached in ``_reins_view_stats_cache``, and invalidated by the ``reins_bucket`` setter and on ``update``. Its only consumer is ``reins_summary_df``::

    rows:  component in {freq, sev, agg} x measure in {ex1, ex2, ex3,
                                                       mean, cv, skew}
    cols:  (stage, view, basis)
             occ stage:  view in {gross, ceded, net}
             agg stage:  view in {subject, ceded, net}
             basis    :  EX | Est

**EX**, the theoretic basis, holds the exact pre-bucket image moments, :math:`\sum g(x_s)^j \, p_{\mathrm{subject}}` with :math:`g` the identity, the ceder or the netter, with no rebucketing scatter. The occurrence aggregate row compounds the severity image moments through the frequency; the aggregate stage acts on ``p_agg_subject`` directly. The frequency reference is the gross full moments for every view, because the count is unchanged by occurrence reinsurance.

**Est**, the empirical basis, holds moments of the rebucketed density read from ``reins_density_df``. On this basis the gross frequency is left ``NaN``, mirroring ``summary_df``, and occurrence ceded and net carry the unconditional mean :math:`\mathsf E[N]` only, with cv and skewness ``NaN``, so that frequency times severity equals aggregate per view.

The EX against Est difference isolates the per-stage ``reins_bucket`` rebucketing error: ``linear`` preserves the directly rebucketed mean exactly, while ``nearest`` biases it by at most ``bs/2``. The rebucketing regression tests read this frame directly.

``reins_stats_df``, the per-layer layering view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actuarial "how is the loss layered" exhibit: per layer, empirical, on the model grid, one column per reinsurance layer plus the gross book and the ceded and net totals::

    cols:  (view, layer), view in {occ, agg}
             occ: Gross (always), layer.1 ... layer.k, Ceded, Net
             agg: layer.1 ... layer.m, Ceded, Net
                  (no agg Subject column: the subject is the occ column
                   flagged output == 1, or Gross when there is no occ program)

    rows:  a meta block, then (freq | sev | agg) x
           (ex1 | ex2 | ex3 | mean | cv | skew), where ex1 duplicates mean
           for filter(regex=...) convenience

The meta block holds:

``share``
    proportion covered.
``limit``
    for Gross, the claim-count-weighted policy limit at share 1; for occurrence Ceded, the share-placed sum of layer limits.
``attach``
    for Gross, the claim-count-weighted attachment; for occurrence Ceded, the minimum attachment.
``pr_attach``
    the ground-up probability that the underlying loss attaches the view, :math:`\Pr(X > \mathrm{exp\_attach} + \mathrm{view\_attach})`, taken from the underlying frozen severity ``fz``, not from the conditional ``sev_density``, which reads 0 at the policy cap.
``pr_detach``
    the ground-up probability that it exhausts the view, :math:`\Pr(X > \mathrm{attach} + \mathrm{limit})`. ``NaN`` for unlimited layers and for net totals.
``pr_loss``
    :math:`\Pr(\text{aggregate} > 0)` from the column's aggregate density.
``lol``
    loss on line, the expected layer aggregate loss divided by the placed limit.
``output``
    a 0/1 flag marking each stage's output view. An occurrence-plus-aggregate program carries two 1s; Gross carries it when there is no occurrence program.

Occurrence layers are conditional on a loss reaching the layer. The frequency is the penetrating count :math:`n' = \mathsf E[N] \Pr(X > \mathrm{attach})` and the severity is the unconditional layer severity divided by the same :math:`\Pr(X > \mathrm{attach})`, which leaves the layer aggregate mean :math:`n' \cdot \mathrm{sev}` equal to the unconditional :math:`\mathsf E[N]\,\mathsf E[\mathrm{ceded}]`. The ``agg`` row is the column's actual aggregate distribution, the FFT of the unconditional layer ceded severity, so its higher moments and ``pr_loss`` are exact and the layer aggregate means sum to the Ceded total. The conditioning :math:`\Pr(\mathrm{subject} > \mathrm{attach} \mid \text{policy loss})` uses the modeled ``self.sev.sf(attach)``, while the displayed ``pr_attach`` and ``pr_detach`` are absolute ground-up probabilities, a separate basis.

The Ceded and Net totals are unconditional: the same claim count as Gross and unconditional severities, so Ceded severity plus Net severity equals Gross severity. The aggregate block leaves frequency and severity all ``NaN``, because a cover on the aggregate has no per-claim frequency and severity that combine in the usual way.

Canonical ordering (Gross, layer.k, Ceded, Net, with occurrence before aggregate) is enforced by insertion order rather than by a pandas ordered Categorical, because the ``layer`` level is dynamic.

``reins_summary_df``, the per-stage economic view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The daily driver. One block per applicable stage, sharing the same eight columns as :meth:`summary_df` and mirroring its economic view::

    Occurrence block, leads with gross:
        rows (gross | ceded | net) x (freq | sev | agg)
    Aggregate block, leads with subject:
        rows (subject | ceded | net) x (agg)
        (sev is n/a; freq is degenerate, so NaN)

    cols:  EX | Est EX | Change EX | CV | Est CV | Change CV | Sk | Est Sk

``EX``, ``CV`` and ``Sk`` hold the theoretic reference, the leading view's exact pre-bucket moments, gross for occurrence and subject for aggregate, held constant down each component. That is the reference every view is measured against. ``Est *`` is the per-view model output, the rebucketed moment. ``Change`` is ``(Est - reference) / reference``, and it reads two ways off one arithmetic. On the leading gross or subject row the reference and the output are the same view, so it is the numerical validation, the rebucketing error, which is about zero under ``linear``. On the ceded and net rows it is the percentage impact of the cession on that moment. It is the same arithmetic as ``summary_df``'s single ``Change`` column.

The whole frame is unconditional, with no dividing by :math:`\Pr(\mathrm{attach})`. On the ``Est`` basis the gross frequency row is ``NaN``, mirroring ``summary_df``, and ceded and net carry the unconditional mean :math:`\mathsf E[N]` only, so frequency times severity equals aggregate per view. The ``view`` and ``component`` index labels are lower case to match the other frames. The frame is derived from ``_reins_view_stats`` and returns ``None`` when no reinsurance is configured.

A worked cross-check: in a program that is net at occurrence and net at aggregate, the occurrence ``net`` and ``agg`` ``Est`` mean equals the aggregate ``subject`` reference exactly, because the occurrence net output is the subject the aggregate cover sees.

.. _reins portfolio:

Portfolio level, end to end
---------------------------

Per-stage detail is only meaningful per unit, since units may carry different programs, so the :class:`Portfolio` objects are end to end: the final gross, ceded and net of the portfolio aggregate. The per-stage breakdown stays in each unit's ``reins_summary_df``. All three gate on "any unit cedes", reusing ``_reins_after_label``, and return ``None`` for a gross-only portfolio.

``Portfolio.reins_density_df``
    Portfolio gross, ceded and net aggregate densities, with columns ``loss``, ``p_agg_gross``, ``p_agg_ceded`` and ``p_agg_net``, formed by convolving the per-unit gcn aggregate densities under the same independent-FFT machinery the total already uses. Units without reinsurance contribute gross = ceded = net = modeled. This convolution is the one genuinely new computation at the portfolio level.
``Portfolio.reins_stats_df``
    Per-unit columns plus a portfolio ``total``, with rows ``(view, measure)`` and view in gross, ceded, net. End to end, with no per-stage or per-layer split. Total means equal the sum of unit end-to-end means per view, because means add under convolution.
``Portfolio.reins_summary_df``
    The :meth:`Portfolio.summary_df` assembly, concatenating each unit's ``reins_summary_df`` under a ``unit`` index level with the total appended. Units without reinsurance are omitted from their own blocks. The ``total`` block follows the same economic view: ``EX``, ``CV`` and ``Sk`` are the gross end-to-end moment held constant, ``Est`` is the per-view output, and ``Change`` is the program impact, which is 0 on the gross row because no exact pre-bucket reference exists for the convolved portfolio marginals.

.. _reins stats columns:

The staged reinsurance columns of ``stats_df``
----------------------------------------------

Not a separate object, but the reinsurance progression baked into the canonical ``stats_df`` during ``update_work``. Beyond ``mixed`` and ``empirical``, the staged block writes these de-fuzzed severity and aggregate moments:

===================  ==================================================================
Column               Meaning
===================  ==================================================================
``gross_empirical``  subject (gross) empirical moments, what validation checks
``after_occ``        empirical moments after the occurrence stage, before aggregate
``occ_impact``       ``after_occ / mixed``, the ratio saying what occurrence did
``agg_impact``       ``empirical / after_occ``, the ratio saying what aggregate did
``empirical``        the final object, after both stages
===================  ==================================================================

The subject, after-occurrence, after-aggregate progression is computed here, and ``error`` validates ``gross_empirical`` against ``mixed``, which is the apples-to-apples check once reinsurance has broken the theoretical against empirical correspondence.

.. _reins text helpers:

Text and plot helpers
---------------------

:attr:`reins_kinds`
    Counts which stages are present and returns one of four strings.
:attr:`reins_description`
    Walks ``occ_reins`` and ``agg_reins`` building a human sentence, "Net of 88% share of 4,000 xs 1,000 per occurrence then net of 100% share of 2,000 xs 3,000 in the aggregate." The ``net of`` and ``ceded to`` wording is the requested view, and ``reins_stats_df`` and ``reins_summary_df`` label the aggregate subject consistently with it, and with the ``output`` flag.
:meth:`reins_occ_plot`
    An occurrence log-density panel plus an aggregate quantile plot, reading the ``p_*`` columns of ``reins_density_df`` and cumulating inline for the CDF panel.

.. _reins notes:

Notes to remember
-----------------

The mass-split identity is on moments, not buckets.
    "Ceded mean plus net mean equals gross or subject mean" is the first-moment identity that ``'linear'`` rebucketing preserves exactly, and ``'nearest'`` only to ``bs/2``. It is not a per-bucket ``p_net + p_ceded == p_subject``. It surfaces as ``reins_summary_df``'s ``Change`` on the leading row.
The conditional basis is confined to the ``layer.k`` columns.
    Every total, and the whole of ``reins_summary_df``, is unconditional. Layer frequency is :math:`n \Pr(X > \mathrm{attach})` and layer severity is divided by the same probability, so the layer aggregate mean is unchanged and layer means sum to Ceded.
``pr_attach`` and ``pr_detach`` are ground-up.
    They come from the underlying frozen severity ``fz``, with each mixture component's policy attachment as the offset, not from the modeled conditional ``sev_density``, which reads 0 at the policy cap. The conditional ``self.sev.sf(attach)`` is used only for the layer count :math:`n'`, a distinct basis.
Aggregate-stage EX is exact relative to the post-occurrence subject aggregate.
    That subject already carries the occurrence rebucketing, so each stage's ``Change`` isolates that stage's own rebucketing error rather than a cumulative one.
The portfolio gcn convolution assumes unit independence.
    Consistent with the existing portfolio total, and noted in the docstring.
