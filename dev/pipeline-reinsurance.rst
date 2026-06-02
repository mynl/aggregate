.. _pipeline reinsurance:

#######################################
The Reinsurance Reporting Surface
#######################################

*Draft for the developer documentation. Inventories every reinsurance object
in* ``distributions.py`` *that produces DataFrame output, says in plain terms
exactly what each cell computes, and then maps the current surface onto the
three rationalized objects proposed in* ``dev/reins-reporting.md``. *Companion
to* :ref:`pipeline aggregate` *(which covers how net/ceded densities are
computed — Phase 4, "Reinsurance brackets the FFT") and* :ref:`pipeline
portfolio`. *Line numbers are approximate and drift; method names are the
stable anchors. The pseudo-code is deliberately arithmetic ("sum of loss ×
prob"), not measure-theoretic — the point is to read the column off the screen,
not to admire the notation.*

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
takes a *subject* density and a layer list and returns a small frame holding the
subject, net, and ceded densities on the model grid. Everything else in this
document is **re-assembly of those frames into reports** — moments, per-layer
breakdowns, expected counts. With the ``reins-buckets`` change (1.0.0a18) the
re-gridding is the mass-preserving scatter :meth:`Aggregate._rebucket_to_grid`;
nothing in the reporting layer changed shape.

Vocabulary used throughout (and the one terminology fix the plan makes):

* **Subject** — the input *to a stage*. For occurrence that is the gross
  severity; for aggregate it is the *occurrence output the agg program sees*
  (already net/ceded of occurrence), so calling it "gross" would be wrong.
* **Gross** — the very top, before any cover.
* **gcn** — the gross / ceded / net triple.
* **fsa** — the freq / sev / agg triple.

The whole surface, at a glance:

============================  =========  ====================================================
Object                        Shape      One-line role
============================  =========  ====================================================
``reinsurance_kinds()``       str        "None / Occurrence only / Aggregate only / both"
``reinsurance_description()`` str        human sentence ("Net of 88% share of 4,000 xs 1,000…")
``occ_reins_df``              DataFrame  raw occ-stage subject/net/ceded densities + loss maps
``agg_reins_df``              DataFrame  raw agg-stage subject/net/ceded densities + loss maps
``reinsurance_df``            DataFrame  all gcn densities, severity **and** aggregate
``reinsurance_audit_df``      DataFrame  per-layer moments (apply each layer alone)
``reinsurance_occ_layer_df``  DataFrame  occ layering view: layer loss, counts, conditional sev
``reinsurance_report_df``     DataFrame  mean/sd/cv/skew of every ``reinsurance_df`` density
``stats_df`` staged columns   DataFrame  subject → after-occ → after-agg moment progression
``reinsurance_occ_plot()``    figure     occ log-density + aggregate quantile plot
============================  =========  ====================================================

The DataFrame objects overlap heavily and are inconsistently shaped — three of
them (``audit``, ``report``, ``occ_layer``) are different slices of the *same*
per-layer / per-density moment computation. That redundancy is what
``reins-reporting.md`` targets; Part B maps it out.


******************************************************
Part A — the existing surface, object by object
******************************************************

The engine: ``_apply_reins_work`` (≈ 3482)
==========================================

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
    p_ceded = rebucket(loss_ceded, weights=p_subject)
    p_net   = rebucket(loss_net,   weights=p_subject)

Returned frame (one row per grid bucket), columns in this order:

============  ====================================================================
column        meaning
============  ====================================================================
``loss``      bucket loss ``xs[k]`` (also the index)
``p_subject`` subject density (input to this stage)
``F_subject`` ``p_subject`` cumulated (subject CDF)
``loss_net``  ``netter(xs[k])`` — where this bucket's loss is retained to
``loss_ceded````ceder(xs[k])``  — where this bucket's loss is ceded to
``F_net``     ``p_net`` cumulated
``F_ceded``   ``p_ceded`` cumulated
``p_net``     net density (subject mass re-gridded through ``netter``)
``p_ceded``   ceded density (subject mass re-gridded through ``ceder``)
============  ====================================================================

**Key identity (mass split, not per-bucket split).** ``p_net + p_ceded`` is
**not** ``p_subject`` bucket-by-bucket — a subject loss at ``x`` sends its mass
to *two different* buckets (its net image and its ceded image). What *is*
preserved is total mass (each marginal sums to ``Σ p_subject``) and, under
``'linear'``, the first moment: ``Σ loss·p_net + Σ loss·p_ceded == Σ
loss·p_subject``.

**Two different objects in one frame (the clarification driving the rewrite).**
The ``loss_net`` / ``loss_ceded`` columns are *exact* real values (the ceder /
netter evaluated at the grid loss); the ``p_net`` / ``p_ceded`` columns are
those images *rebucketed* back onto the grid. Mixing exact loss images and
rebucketed probabilities in one frame is the source of confusion. The
``reins-reporting`` plan separates them: the **rebucketed densities** are the
only thing stored (straight into ``reins_density_df``), while the **exact**
images are recomputed on demand from the retained ceder/netter for the
"theoretic" (pre-bucket) moments — see Part B.

**The ``F_*`` columns are vestigial.** ``F_subject`` / ``F_net`` / ``F_ceded``
are left over from the old re-gridding algorithm, which built ``F_net`` /
``F_ceded`` by ``interp1d`` of the grouped CDF and then ``np.diff``'d them to
*get* the densities. The ``reins-buckets`` scatter computes ``p_net`` /
``p_ceded`` directly, so the only remaining reader is the debug-plot CDF panel
(``filter(regex='F')``), which can ``cumsum`` inline. The plan drops them.

The two callers store this frame directly:

* :meth:`apply_occ_reins` → ``self.occ_reins_df`` (subject = ``sev_density``;
  also sets ``sev_density_gross/net/ceded`` and points ``sev_density`` at the
  requested view).
* :meth:`apply_agg_reins` → ``self.agg_reins_df`` (subject = ``agg_density``;
  sets ``agg_density_gross/net/ceded``, re-FFTs the chosen view into
  ``ftagg_density``).

So **``occ_reins_df`` / ``agg_reins_df`` are the raw per-stage frames** — but
they are redundant containers: their ``p_*`` densities already live on
``self.{sev,agg}_density_{gross,ceded,net}``, and the exact loss images can be
recomputed from the retained ceder/netter. The revised plan therefore
**removes** them (the first draft kept them private); their densities surface
through ``reins_density_df``.


``reinsurance_df`` — all gcn densities (≈ 1828)
===============================================

The reinsurance analogue of ``density_df``: one row per grid bucket, ten
columns — the gross/ceded/net densities at **both** the severity and aggregate
level::

    loss              = xs

    # severity level (from occ stage; equal to sev_density when no occ)
    p_sev_gross       = sev_density_gross
    p_sev_ceded       = sev_density_ceded
    p_sev_net         = sev_density_net

    # aggregate of each occ severity view — only when an occ program exists.
    # Each is a *fresh FFT* of the corresponding severity (re-runs the core):
    p_agg_gross_occ   = FFT_aggregate(p_sev_gross)
    p_agg_ceded_occ   = FFT_aggregate(p_sev_ceded)
    p_agg_net_occ     = FFT_aggregate(p_sev_net)

    # final aggregate, gcn of the *aggregate* program (if one exists)
    p_agg_gross       = agg_density_gross   (or agg_density if no agg program)
    p_agg_ceded       = agg_density_ceded   (or None)
    p_agg_net         = agg_density_net     (or None)

The three cases the docstring lists:

* **occ only** — the ``p_agg_*_occ`` block *is* the gcn aggregate (there is no
  separate agg program); ``p_agg_ceded/net`` are ``None``.
* **agg only** — no ``p_agg_*_occ`` block (no occ stage); ``p_sev_*`` ceded/net
  are ``None``; ``p_agg_gross/ceded/net`` carry the answer.
* **both** — ``p_agg_*_occ`` is the aggregate of each occ view (the *subject*
  the agg program sees), and ``p_agg_gross/ceded/net`` is the agg program acting
  on the requested occ output.

This is the object the plan **renames** to ``reins_density_df`` (and re-points
its one internal consumer, ``reinsurance_occ_plot``). It re-implements the FFT
core via :meth:`_fft_aggregate` — the shared helper, so the fixed-1 / zero-risk
shortcuts are honoured identically to the main path.


``reinsurance_audit_df`` — per-layer moments (≈ 1950)
=====================================================

The "what does each individual layer do" view. Built by
:meth:`_reins_audit_df_work`, separately for ``occ`` and ``agg``, then
concatenated under a ``kind`` key.

The engine applies **each layer on its own** to the gross subject density and
also keeps the *combined* (all-layers) result::

    for each layer L = (share, limit, attach) in this stage's layer list:
        _, _, df_L = _apply_reins_work([L], gross_subject_density)
        keep df_L            # this layer alone
    keep occ_reins_df / agg_reins_df    # all layers together, labelled (all, inf, 'gup')

    # then, per layer, turn its subject/net/ceded density into moments:
    for view p in {subject (p_subject), ceded (p_ceded), net (p_net)}:
        ex   = sum of  loss   · p          (mean)
        ex2  = sum of  loss^2 · p
        ex3  = sum of  loss^3 · p
        (ex, var, sd, cv, skew) = MomentWrangler(ex, ex2, ex3).stats

Result: MultiIndex **rows** ``(kind, share, limit, attach)`` — one per layer
plus a ``(…, all, inf, 'gup')`` total — × MultiIndex **columns** ``(view, stat)``
with ``view ∈ {ceded, net, subject}`` and ``stat ∈ {ex, var, sd, cv, skew}``.

Note these are moments of the **occurrence severity** density for ``kind=occ``
and of the **aggregate** density for ``kind=agg`` — the ``ex`` for occ is a
*per-claim* mean, not an annual expected loss. (``reinsurance_occ_layer_df``
below is what scales the occ block up by the claim count.)

The plan **removes** this object, folding its per-layer columns into
``reins_stats_df`` (and keeps ``_reins_audit_df_work`` as the per-layer engine).


``reinsurance_occ_layer_df`` — the occ layering view (≈ 1874)
=============================================================

A re-slice of ``reinsurance_audit_df.loc['occ']`` into the actuarial
"how is the loss layered" exhibit, scaled to **annual** (× claim count ``n``).
Per occ layer ``(share, limit, attach)``:

============  =====  ===========================================================
stat          view   plain meaning
============  =====  ===========================================================
``ex``        gcn    annual layer loss = ``n × (per-occ mean)`` for each of
                     ceded / net / subject
``cv``        gcn    coefficient of variation of the per-occ layer loss (gcn)
``en``        ceded  expected **count** of claims reaching the layer
                     = ``n``               (for the ``gup`` total row)
                     = ``n × P(X > attach)`` (for a real layer)
``severity``  ceded  expected ceded **per claim that reaches the layer**
                     = ``(per-occ ceded mean) / P(X > attach/share)``
``pct``       ceded  ceded share of total subject = ``ceded ex / subject ex``
============  =====  ===========================================================

So ``ex.ceded`` (annual ceded loss) ≈ ``en.ceded × severity.ceded`` —
frequency-to-the-layer times conditional severity-in-the-layer, the standard
layer decomposition. (The ``attach/share`` gross-up inside ``severity`` is a
known oddity — flag it when this folds into ``reins_stats_df``; the
``reinsurance_audit_df`` ``ex`` it derives from is unambiguous.)

The plan **removes** this object; its expected-count column (``n·P(X>attach)``)
becomes the freq row of ``reins_stats_df``'s ceded view.


``reinsurance_report_df`` — moments of the densities (≈ 1900)
=============================================================

The simplest summary: take every density column of ``reinsurance_df`` and
report its moments::

    for each density column c in reinsurance_df (all but `loss`):
        mean = sum of loss   · c
        cv   = sd / mean,  where sd from sum of loss^2 · c
        skew from sum of loss^3 · c
        sd   = cv × mean
    rows = [mean, cv, sd, skew]   (in that display order)

Result: rows ``[mean, cv, sd, skew]`` × columns = the nine gcn density columns
(``p_sev_gross … p_agg_net``). It is exactly the column-wise moment table of
``reinsurance_df``; the only thing it adds over ``reinsurance_audit_df`` is the
*after-occ aggregate* columns (``p_agg_*_occ``) — the aggregates of the occ
views *before* the agg program — which the per-layer audit does not carry.

The plan **removes** this object, folding the gcn totals into
``reins_stats_df`` and the daily-driver layout into ``reins_describe``.


``stats_df`` staged-reinsurance columns (meta.4, ≈ 3005)
========================================================

Not a separate object, but the reinsurance reporting baked into the canonical
``stats_df`` during ``update_work``. Beyond the usual ``mixed`` / ``empirical``
columns, the staged block writes (de-fuzzed moments, sev and agg rows):

================  ==================================================================
column            plain meaning
================  ==================================================================
``gross_empirical`` subject (gross) empirical moments — what validation checks
``after_occ``       empirical moments after the occurrence stage (pre-agg)
``occ_impact``      ``after_occ / mixed``     (ratio: what occ did)
``agg_impact``      ``empirical / after_occ`` (ratio: what agg did)
``empirical``       the final, after-both-stages object
================  ==================================================================

So the **stage progression** subject → after-occ → after-agg is already
computed and parked in ``stats_df``; ``error`` validates ``gross_empirical`` vs
``mixed`` (the only apples-to-apples check available once reinsurance breaks the
theoretical-vs-empirical correspondence). The plan reuses these directly as the
agg-fsa totals of ``reins_describe`` — no new math.


Text and plot helpers (unchanged by the plan)
==============================================

* :meth:`reinsurance_kinds` — counts which stages are present → one of four
  strings.
* :meth:`reinsurance_description` — walks ``occ_reins`` / ``agg_reins`` building
  a human sentence ("Net of 88% share of 4,000 xs 1,000 per occurrence then net
  of 100% share of 2,000 xs 3,000 in the aggregate."). The ``net of`` / ``ceded
  to`` wording is the requested view; the plan asks that ``reins_stats_df`` /
  ``reins_describe`` label the agg "subject" column **consistently** with this.
* :meth:`reinsurance_occ_plot` — occ log-density (from ``occ_reins_df``) plus an
  aggregate quantile plot (from the ``p_agg_*_occ`` columns of
  ``reinsurance_df``). Its only dependency on the renamed object is that one read
  site.


******************************************************
Part B — mapping to the proposed surface
******************************************************

.. note::

   **Implemented in 1.0.0a19.** The surface below is live. Two clarifications
   from the build: (1) ``reins_describe`` cells are the **mean** loss on the
   ``EX | Est | Change`` bases — the full ``(ex1, ex2, ex3, mean, cv, skew)``
   detail lives in ``reins_stats_df``. (2) ``EX`` is the literal pre-bucket
   image moment (``sum g(xs)^j · p_subject``) with **no** defective-mass tail
   term, so ``Change ≈ 0`` under ``'linear'`` holds tightly on the
   *directly-rebucketed* targets — occurrence **severity** rows and **all
   aggregate-stage** rows — while the occurrence **aggregate** row also carries
   the FFT grid deficit (it is the compound of the exact severity, not a direct
   rebucket). See ``tests/test_reins_reporting.py``.

``reins-reporting.md`` (target 1.0.0a19) collapses the redundant surface into
**three** coherent objects, defined at the ``Aggregate`` level and combined at
the ``Portfolio`` level (all new there). The author's inventory → fate:

==  ===============================  ===================================================
\#  current item                     fate (revised 2026-06-01)
==  ===============================  ===================================================
1   ``reinsurance_description``      unchanged
2   ``reinsurance_kinds``            unchanged
3   ``agg_reins`` tuples             unchanged
4   ``occ_reins`` tuples             unchanged
5   ``occ_reins_df``                 **remove** (densities on ``sev_density_*`` + ``reins_density_df``; exact on demand)
6   ``agg_reins_df``                 **remove** (ditto via ``agg_ceder``/``agg_netter``)
7   ``reinsurance_df``               **rename** → ``reins_density_df`` (consistent cols; ``p_agg_gross_occ→p_agg_gross``, ``p_agg_gross→p_agg_subject``)
8   ``reinsurance_audit_df``         **remove** (by-layer detail dropped)
9   ``reinsurance_occ_layer_df``     **remove** (by-layer detail dropped)
10  ``reinsurance_report_df``        **remove** → into ``reins_stats_df`` / ``reins_describe``
11  ``reinsurance_occ_plot``         unchanged (re-point reads)
–   ``_reins_audit_df_work``         **remove** (per-layer engine no longer needed)
–   ``F_*`` engine columns           **remove** (vestigial)
==  ===============================  ===================================================

The three survivors
====================

**1. ``reins_density_df``** — the rename of ``reinsurance_df``, with **consistent
columns** regardless of which programs are present (a missing stage contributes
ceded = 0, net = subject). All gcn densities (severity and aggregate-of-occ and
aggregate-cover), one row per grid bucket. Two column renames enforce the
gross/subject convention::

    p_agg_gross_occ  →  p_agg_gross     # FFT of gross severity = true gross aggregate
    p_agg_gross      →  p_agg_subject   # agg-cover input = aggregate of the occ output

*No math change*; it is the densities of record (rebucketed) that the other two
are computed from.

**2. ``reins_stats_df``** — single source of truth, shaped like ``stats_df`` but
**per stage / per view / per basis**, not per layer (the by-layer detail is
dropped — Decision 4)::

    rows:  (component ∈ freq|sev|agg) × (measure ∈ ex1|ex2|ex3|mean|cv|skew)
    cols:  (stage, view, basis), variable by what is applied
             occ stage:  view ∈ gross | ceded | net
             agg stage:  view ∈ subject | ceded | net
             basis    :  EX | Est
             EX  = exact pre-bucket moment  = sum of ceder(xs)^j · p_subject
                   (netter analogue for net; freq ceded = n · P(X > attach))
             Est = moment of the rebucketed density in reins_density_df

So the removed report objects map in by basis, not by layer:

============================  ==========================================================
old object                    becomes
============================  ==========================================================
``reinsurance_report_df``     the gcn **total** moments — now split into EX vs Est
``reinsurance_audit_df``      gone; whole-structure EX replaces per-layer moments
``reinsurance_occ_layer_df``  the ``freq`` row ``n·P(X>attach)`` (ceded count); the
                              ``severity``/``pct`` slices are dropped (compute on demand)
============================  ==========================================================

**3. ``reins_describe``** — the daily driver, **per-stage** (Decision 7): one
block per applicable stage, occ leading with **Gross**, agg leading with
**Subject**::

    Occurrence block:  rows (Gross | Ceded | Net) × (Freq | Sev | Agg)
                       Agg row = aggregate of that severity view
                       (p_agg_gross / p_agg_ceded_occ / p_agg_net_occ)
    Aggregate block:   rows (Subject | Ceded | Net) × (Agg)
                       (Sev n/a; Freq degenerate → NaN)
                       views map to p_agg_subject / p_agg_ceded / p_agg_net

    cols:  EX | Est | Change          Change = (Est − EX) / EX

``EX`` is the exact pre-bucket truth, ``Est`` the rebucketed empirical, so
``Change`` *is* the rebucketing error (≈ 0 for ``linear`` means; ≤ ``bs/2`` for
``nearest``) — the natural validator for ``reins-buckets``. Derived from
``reins_stats_df``.

Portfolio level (all new) — end-to-end gcn
==========================================

Per-stage detail is only meaningful per unit (units may carry different
programs), so the **Portfolio objects are end-to-end** (final gross / ceded /
net of the portfolio aggregate); the per-stage breakdown stays in each unit's
``reins_describe``.

* **``Portfolio.reins_density_df``** — portfolio gross/ceded/net *aggregate*
  densities by convolving the per-unit gcn aggregate densities under the same
  independence FFT the total already uses. Units without reinsurance contribute
  ``gross = ceded = net = modeled``. *This is the one genuinely new computation*
  (everything else is re-assembly).
* **``Portfolio.reins_stats_df``** — per-unit columns + a portfolio ``total``
  derived from the portfolio gcn densities, mirroring ``Portfolio.stats_df``
  (end-to-end gcn views).
* **``Portfolio.reins_describe``** — the ``Portfolio.describe`` assembly:
  ``pd.concat([u.reins_describe for u in self] + [total], keys=names+['total'])``.
  Total block (end-to-end gcn): means sum across units per view; cv/skew from the
  portfolio gcn densities.
* All three gate on "any unit cedes" (reuse ``_reins_after_label``); return
  ``None`` for a gross-only portfolio.

Things to confirm while implementing
====================================

* **The mass-split identity is on moments, not buckets.** The "Ceded + Net mean
  == Gross/Subject mean" check is the *first-moment* identity that ``'linear'``
  rebucketing preserves exactly (and ``'nearest'`` only to ``bs/2``) — *not* a
  per-bucket ``p_net + p_ceded == p_subject``. (Same point flagged in the
  ``reins-buckets`` close-out; here it surfaces as the ``Change`` column.)
* **"Subject" labelling for the agg stage** depends on the requested occ view
  (``net of`` / ``ceded to``); keep it consistent with
  ``reinsurance_description``.
* **The dropped ``reinsurance_occ_layer_df`` ``severity`` gross-up**
  (``sf(attach/share)``) looked like a bug; if a conditional-severity number is
  ever wanted in ``reins_stats_df``, use ``sf(attach)`` and confirm.
* **Agg-stage EX is exact relative to the post-occ subject aggregate**, which
  already carries the occ rebucketing — so the per-stage ``Change`` isolates each
  stage's own rebucketing error, not a cumulative one.
* **Portfolio gcn convolution assumes unit independence** — consistent with the
  existing portfolio total; note it in the docstring.
