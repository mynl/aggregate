.. _pipeline reinsurance:

#################################
The Reinsurance Reporting Surface
#################################

*Developer documentation for the reinsurance reporting surface as it stands at
1.0.0a19 (after the* ``reins-buckets`` *and* ``reins-reporting`` *cycles).
Inventories every reinsurance object in* ``distributions.py`` / ``portfolio.py``
*that produces DataFrame output and says, in plain terms, exactly what each cell
computes. Companion to* :ref:`pipeline aggregate` *(which covers how net/ceded
densities are computed — Phase 4, "Reinsurance brackets the FFT") and*
:ref:`pipeline portfolio`. *Line numbers drift; method names are the stable
anchors. The pseudo-code is deliberately arithmetic ("sum of loss × prob"), not
measure-theoretic — the point is to read the column off the screen.*

.. note::

   This file replaces the pre-refactor inventory (``occ_reins_df`` /
   ``agg_reins_df`` / ``reinsurance_df`` / ``reinsurance_audit_df`` /
   ``reinsurance_occ_layer_df`` / ``reinsurance_report_df``) and the
   "mapping to a proposed surface" that preceded the build. The history and the
   design reasoning live in ``dev/done/reins-reporting.md`` (including the
   "Post-plan punch-ups" addendum that records where the implementation diverged
   from the first design). What follows is the **live** surface only.

.. contents::
   :local:
   :depth: 2


********
Overview
********

Reinsurance enters the pipeline at two points (see :ref:`pipeline aggregate`,
Phase 4):

* **occurrence** cover acts on ``sev_density`` *before* the FFT;
* **aggregate** cover acts on ``agg_density`` *after* the FFT.

Each stage is applied by one engine, :meth:`Aggregate._apply_reins_work`, which
takes a *subject* density and a layer list and returns a small transient frame
holding the subject, net, and ceded densities on the model grid. Everything else
in this document is **re-assembly of those densities into reports** — moments,
per-layer breakdowns, exposure probabilities. The re-gridding is the
mass-preserving scatter :meth:`Aggregate._rebucket_to_grid`
(``reins_bucket = 'linear'`` default, or ``'nearest'``).

The reporting surface is **three public objects** (plus one private helper),
defined on ``Aggregate`` and mirrored end-to-end on ``Portfolio``:

=============================  =========  ====================================================
Object                         Shape      One-line role
=============================  =========  ====================================================
``reins_density_df``           DataFrame  all gcn densities, severity **and** aggregate, on the grid
``reins_stats_df``             DataFrame  **per-layer** layering view (Gross / layer.k / Ceded / Net)
``reins_summary_df``           DataFrame  per-stage **economic view**, the daily driver
``_reins_view_stats``          DataFrame  *(private)* per-stage/view/basis EX-vs-Est moments → feeds ``reins_summary_df``
``reinsurance_kinds()``        str        "None / Occurrence only / Aggregate only / both"
``reinsurance_description()``  str        human sentence ("Net of 88% share of 4,000 xs 1,000…")
``reinsurance_occ_plot()``     figure     occ log-density + aggregate quantile plot
=============================  =========  ====================================================

Vocabulary used throughout:

* **Gross** — the very top, before any cover.
* **Subject** — the input *to a stage*. For occurrence the subject *is* gross;
  for aggregate it is the *occurrence output the agg program sees* (already
  net/ceded of occurrence), so calling it "gross" would be wrong.
* **Output** — the model-output view when occ and agg pass *different* kinds
  (e.g. ``net of`` occ then ``ceded to`` agg); the ``describe`` label for that
  mixed case.
* **gcn** — the gross / ceded / net triple. **fsa** — freq / sev / agg.
* **conditional vs unconditional** — only the per-layer ``layer.k`` columns of
  ``reins_stats_df`` are *conditional* on a loss reaching the layer. Every
  total (``Gross`` / ``Ceded`` / ``Net``) and the whole of ``reins_summary_df``
  is **unconditional**. The gross/subject/net/ceded/output words are centralised
  as ``REINS_LABEL_*`` constants in ``constants.py``.


****************************
The engine: re-gridding core
****************************

``_apply_reins_work`` (≈ 4030)
==============================

Every reinsurance density in the library comes out of here. Given a layer list
``[(share, limit, attach), …]`` and a *subject* density on the grid ``self.xs``
(``xs[k] = k·bs``)::

    ceder, netter = make_ceder_netter(layers)      # piecewise-linear loss maps
    # ceder(x)  = ceded loss when subject loss is x
    # netter(x) = x - ceder(x)

    for each grid bucket k with subject mass p_subject[k] at loss x = xs[k]:
        this loss is ceded to  loss_ceded[k] = ceder(x)
        and retained as        loss_net[k]   = netter(x)

    # both targets are off-grid; scatter their mass back onto the grid
    # (reins_bucket = 'linear' splits across the two bracketing buckets and
    #  preserves E[loss] exactly; 'nearest' rounds — see _rebucket_to_grid)
    p_ceded = _rebucket_to_grid(loss_ceded, weights=p_subject)
    p_net   = _rebucket_to_grid(loss_net,   weights=p_subject)

Returned transient frame (one row per grid bucket), columns in order:
``loss``, ``p_subject``, ``loss_net``, ``loss_ceded``, ``p_net``, ``p_ceded``.

==============  ====================================================================
column          meaning
==============  ====================================================================
``loss``        bucket loss ``xs[k]`` (also the index)
``p_subject``   subject density (input to this stage)
``loss_net``    ``netter(xs[k])``, exact real value this bucket's loss is retained to
``loss_ceded``  ``ceder(xs[k])``, exact real value this bucket's loss is ceded to
``p_net``       net density (subject mass re-gridded through ``netter``)
``p_ceded``     ceded density (subject mass re-gridded through ``ceder``)
==============  ====================================================================

**Key identity (mass split, not per-bucket split).** ``p_net + p_ceded`` is
**not** ``p_subject`` bucket-by-bucket — a subject loss at ``x`` sends its mass
to *two different* buckets (its net image and its ceded image). What *is*
preserved is total mass (each marginal sums to ``Σ p_subject``) and, under
``'linear'``, the first moment: ``Σ loss·p_net + Σ loss·p_ceded == Σ
loss·p_subject``.

**Exact images vs rebucketed densities.** The ``loss_net`` / ``loss_ceded``
columns are *exact* real values (the ceder / netter evaluated at the grid loss);
the ``p_net`` / ``p_ceded`` columns are those images *rebucketed* back onto the
grid. The frame is **transient** — only the rebucketed densities are kept, on
``sev_density_{gross,ceded,net}`` (occ) and ``agg_density_{gross,ceded,net}``
(agg). The exact (pre-bucket) moments are recomputed on demand from the retained
step functions ``occ_ceder`` / ``occ_netter`` / ``agg_ceder`` / ``agg_netter``.
*(The old ``F_*`` CDF columns and the persistent ``occ_reins_df`` /
``agg_reins_df`` members were removed at 1.0.0a19; the debug plot cumsums the
densities inline.)*

The two callers:

* :meth:`apply_occ_reins` — subject = ``sev_density``; sets
  ``sev_density_gross/net/ceded`` and points ``sev_density`` at the requested
  view.
* :meth:`apply_agg_reins` — subject = ``agg_density``; sets
  ``agg_density_gross/net/ceded``, re-FFTs the chosen view into
  ``ftagg_density``.


*********************
The reporting objects
*********************

``reins_density_df`` (≈ 1828)
=============================

The reinsurance analogue of ``density_df``: one row per grid bucket,
**consistent columns regardless of which programs are present** (a missing stage
contributes the no-cession values: ceded = 0, net = subject)::

    loss              = xs

    # severity level (from occ stage; equal to sev_density when no occ)
    p_sev_gross       = sev_density_gross
    p_sev_ceded       = sev_density_ceded
    p_sev_net         = sev_density_net

    # aggregate of each occ severity view (fresh FFT of each, via _fft_aggregate)
    p_agg_gross       = FFT_aggregate(p_sev_gross)    # the true gross aggregate
    p_agg_ceded_occ   = FFT_aggregate(p_sev_ceded)
    p_agg_net_occ     = FFT_aggregate(p_sev_net)

    # aggregate cover: subject = aggregate of the requested occ output
    p_agg_subject     = aggregate the agg cover sees (= p_agg_gross when no occ)
    p_agg_ceded       = agg_density_ceded
    p_agg_net         = agg_density_net

The two renames vs the pre-refactor ``reinsurance_df`` enforce the gross/subject
convention::

    p_agg_gross_occ  →  p_agg_gross     # FFT of gross severity = true gross aggregate
    p_agg_gross      →  p_agg_subject   # agg-cover input = aggregate of the occ output

All aggregate columns come from :meth:`_fft_aggregate`, so the fixed-1 /
zero-risk shortcuts are honoured identically to the main path. This is the
densities-of-record (rebucketed) frame the other objects are computed from.
:meth:`reinsurance_occ_plot` reads from here.


``_reins_view_stats`` (private; ≈ 2001): the EX-vs-Est frame
============================================================

The per-stage/view/**basis** moment frame, shaped like ``stats_df`` but indexed
by stage/view/basis instead of components. Lazily built, cached in
``_reins_view_stats_cache``, invalidated by the ``reins_bucket`` setter and on
``update``. Its **only** consumer is ``reins_summary_df``.

::

    rows:  (component ∈ freq|sev|agg) × (measure ∈ ex1|ex2|ex3|mean|cv|skew)
    cols:  (stage, view, basis)
             occ stage:  view ∈ gross | ceded | net
             agg stage:  view ∈ subject | ceded | net
             basis    :  EX | Est

* **EX** ("theoretic") = exact pre-bucket image moments —
  ``Σ g(xs)^j · p_subject`` with ``g`` the identity / ceder / netter (no
  rebucketing scatter). The occ aggregate row compounds the severity image
  moments via the frequency; the agg stage acts on ``p_agg_subject`` directly.
  Frequency reference is the gross full moments for every view (the count is
  unchanged by occurrence reinsurance).
* **Est** ("empirical") = moments of the **rebucketed** density read from
  ``reins_density_df``. On this basis the **gross** frequency is left ``NaN``
  (mirroring ``describe``); occ ceded / net carry the **unconditional** mean
  ``E[N]`` only (cv / skew ``NaN``) so ``freq · sev == agg`` per view.

The EX-vs-Est difference isolates the per-stage ``reins_bucket`` rebucketing
error (``linear`` preserves the directly-rebucketed mean exactly; ``nearest``
biases it by at most ``bs/2``). The rebucketing regression tests read this frame
directly.


``reins_stats_df`` (≈ 2146): the per-layer layering view
========================================================

The actuarial "how is the loss layered" exhibit. **Per-layer**, empirical
(model-grid), one column per reinsurance layer plus the gross book and the
ceded / net totals::

    cols:  (view, layer), view ∈ occ | agg
             occ: Gross (always), layer.1 … layer.k, Ceded, Net
             agg: layer.1 … layer.m, Ceded, Net
                  (no agg Subject column — the subject is the occ column
                   flagged output == 1, or Gross when there is no occ program)

    rows:  meta block:
             share       proportion covered
             limit       Gross = claim-count-weighted policy limit (share 1);
                         occ Ceded = share-placed sum of layer limits
             attach      Gross = cc-weighted attach; occ Ceded = min attachment
             pr_attach   GROUND-UP P(underlying loss attaches the view)
                         = P(X > exp_attach + view_attach), from the underlying
                         frozen severity fz (NOT the conditional sev_density,
                         which reads 0 at the policy cap)
             pr_detach   GROUND-UP P(it exhausts the view) = P(X > attach+limit);
                         NaN for unlimited layers / net totals
             pr_loss     P(aggregate > 0) from the column's aggregate density
             lol         loss on line = expected layer aggregate loss / placed limit
             output      0/1 flag marking each stage's output view (two 1s for an
                         occ+agg program; Gross carries it when there is no occ)
           then (freq|sev|agg) × (ex1|ex2|ex3|mean|cv|skew)
                ex1 duplicates mean for filter(regex=...) convenience

**Occurrence layers are conditional** on a loss reaching the layer: frequency is
the penetrating count ``n' = E[N]·P(X > attach)`` and severity is the
unconditional layer severity divided by ``P(X > attach)`` (conditional given
attach), which leaves the layer aggregate mean ``n'·sev`` equal to the
unconditional ``E[N]·E[ceded]``. The ``agg`` row is the column's actual
aggregate distribution (FFT of the unconditional layer ceded severity), so its
higher moments and ``pr_loss`` are exact, and the layer agg means **sum to the
Ceded total**. The conditioning ``P(subject > attach | policy loss)`` uses the
modeled ``self.sev.sf(attach)``; the displayed ``pr_attach`` / ``pr_detach`` are
the absolute ground-up probabilities — a separate basis.

**The Ceded / Net totals are unconditional**: the same claim count as Gross and
unconditional severities, so ``Ceded`` sev + ``Net`` sev == ``Gross`` sev.

**The aggregate block leaves freq / sev all NaN** — a cover on the aggregate has
no per-claim frequency / severity that combine in the usual way.

Canonical ordering (Gross / layer.k / Ceded / Net, occ before agg) is enforced
by **insertion order**, not a pandas ordered Categorical (the ``layer`` level is
dynamic). This object supersedes the removed ``reinsurance_audit_df`` /
``reinsurance_occ_layer_df``; the published per-layer case-study exhibits can be
rebuilt against ``a.reins_stats_df['occ']`` directly.


``reins_summary_df`` (≈ 2461): the per-stage economic view
==========================================================

The daily driver. One block per applicable stage, sharing the **same eight
columns as** :meth:`describe`, and mirroring its **economic view**::

    Occurrence block (leads with gross):  rows (gross|ceded|net) × (freq|sev|agg)
    Aggregate  block (leads with subject): rows (subject|ceded|net) × (agg)
                                           (sev n/a; freq degenerate → NaN)

    cols:  EX | Est EX | Change EX | CV | Est CV | Change CV | Sk | Est Sk

* ``EX`` / ``CV`` / ``Sk`` hold the **theoretic reference** — the *leading*
  view's exact pre-bucket moments (``gross`` for occ, ``subject`` for agg) —
  **held constant down each component** (the reference every view is measured
  against).
* ``Est *`` is the **per-view model output** (rebucketed moment).
* ``Change = (Est − reference) / reference`` reads two ways off one arithmetic:
  on the leading (gross/subject) row the reference and output are the same view,
  so it is the **numerical validation / rebucketing error** (~0 under
  ``linear``); on the ceded / net rows it is the **% impact of the cession** on
  that moment. Same arithmetic as ``describe``'s single ``Change`` column.

**Always unconditional** (no dividing by ``P(attach)``). Frequency on the
``Est`` basis: the **gross row is NaN** (mirrors ``describe``); ceded / net
carry the unconditional mean ``E[N]`` only so ``freq · sev == agg`` per view.
``view`` / ``component`` index labels are **lower-case** to match the other
frames. Derived from ``_reins_view_stats``; returns ``None`` when no reinsurance
is configured.

A worked cross-check: in an occ-net + agg-net program, the occurrence
``net / agg`` ``Est`` mean equals the aggregate ``subject`` reference exactly —
the occ net output *is* the subject the aggregate cover sees.


******************************************
Portfolio level — end-to-end gcn (all new)
******************************************

Per-stage detail is only meaningful per unit (units may carry different
programs), so the **Portfolio objects are end-to-end** (final gross / ceded /
net of the *portfolio* aggregate); the per-stage breakdown stays in each unit's
``reins_summary_df``. All three gate on "any unit cedes" (reuse
``_reins_after_label``) and return ``None`` for a gross-only portfolio.

* **``Portfolio.reins_density_df``** — portfolio gross/ceded/net *aggregate*
  densities (cols ``loss, p_agg_gross, p_agg_ceded, p_agg_net``) by convolving
  the per-unit gcn aggregate densities under the same independent-FFT machinery
  the total already uses. Units without reinsurance contribute
  ``gross = ceded = net = modeled``. *This convolution is the one genuinely new
  computation.*
* **``Portfolio.reins_stats_df``** — per-unit columns + a portfolio ``total``,
  rows ``(view, measure)`` with ``view ∈ gross|ceded|net``. End-to-end gcn (no
  per-stage / per-layer split at the portfolio level). Total means equal the sum
  of unit end-to-end means per view (means add under convolution).
* **``Portfolio.reins_summary_df``** — the ``Portfolio.describe`` assembly:
  ``pd.concat([u.reins_summary_df for u in self] + [total], keys=names+['total'],
  names=['unit', …])``. Units without reinsurance are omitted from their own
  blocks. The ``total`` block follows the same economic view: ``EX/CV/Sk`` = the
  gross end-to-end moment held constant, ``Est`` = per-view output, ``Change`` =
  programme impact (0 on the gross row — no exact pre-bucket reference exists for
  the convolved portfolio marginals).


*********************************
``stats_df`` staged-reins columns
*********************************

Not a separate object, but the reinsurance progression baked into the canonical
``stats_df`` during ``update_work``. Beyond ``mixed`` / ``empirical`` the staged
block writes (de-fuzzed sev/agg moments):

===================  ==================================================================
column               plain meaning
===================  ==================================================================
``gross_empirical``  subject (gross) empirical moments, what validation checks
``after_occ``        empirical moments after the occurrence stage (pre-agg)
``occ_impact``       ``after_occ / mixed``     (ratio: what occ did)
``agg_impact``       ``empirical / after_occ`` (ratio: what agg did)
``empirical``        the final, after-both-stages object
===================  ==================================================================

The subject → after-occ → after-agg progression is already computed here;
``error`` validates ``gross_empirical`` vs ``mixed`` (the apples-to-apples check
once reinsurance breaks the theoretical-vs-empirical correspondence).


*********************************
Text and plot helpers (unchanged)
*********************************

* :meth:`reinsurance_kinds` — counts which stages are present → one of four
  strings.
* :meth:`reinsurance_description` — walks ``occ_reins`` / ``agg_reins`` building
  a human sentence ("Net of 88% share of 4,000 xs 1,000 per occurrence then net
  of 100% share of 2,000 xs 3,000 in the aggregate."). The ``net of`` / ``ceded
  to`` wording is the requested view; ``reins_stats_df`` / ``reins_summary_df``
  label the agg subject consistently with this (and the ``output`` flag).
* :meth:`reinsurance_occ_plot` — occ log-density plus an aggregate quantile plot,
  reading the ``p_*`` columns of ``reins_density_df`` (cumsumming inline for the
  CDF panel).


*****************
Notes to remember
*****************

* **The mass-split identity is on moments, not buckets.** "Ceded + Net mean ==
  Gross/Subject mean" is the *first-moment* identity ``'linear'`` rebucketing
  preserves exactly (``'nearest'`` only to ``bs/2``) — *not* a per-bucket
  ``p_net + p_ceded == p_subject``. It surfaces as ``reins_summary_df``'s
  ``Change`` on the leading row.
* **Conditional basis is confined to ``reins_stats_df``'s ``layer.k`` columns.**
  Every total and the whole of ``reins_summary_df`` is unconditional. Layer freq is
  ``n·P(>attach)`` and layer sev is divided by the same probability, so the
  layer aggregate mean is unchanged and layer means sum to ``Ceded``.
* **``pr_attach`` / ``pr_detach`` are ground-up.** They come from the underlying
  frozen severity ``fz`` (with each mixture component's policy attachment as the
  offset), *not* the modeled conditional ``sev_density`` (which reads 0 at the
  policy cap). The conditional ``self.sev.sf(attach)`` is used only for the layer
  count ``n'`` — a distinct basis.
* **Agg-stage EX is exact relative to the post-occ subject aggregate**, which
  already carries the occ rebucketing — so each stage's ``Change`` isolates that
  stage's own rebucketing error, not a cumulative one.
* **Portfolio gcn convolution assumes unit independence** — consistent with the
  existing portfolio total; noted in the docstring.
