.. _pipeline aggregate:

The Aggregate Computation Pipeline
==================================

This section describes how an :class:`Aggregate` turns a frequency model and one or more severity models into a discrete approximation of the compound distribution :math:`A = X_1 + \cdots + X_N`, using the FFT algorithm of :cite:t:`Mildenhall2024`. It follows the object from construction through grid choice, discretization, the convolution itself, and validation, and closes with the two conclusions that shaped the design: how the tail is truncated, and why a defective distribution is left defective. The implementation is in ``_aggregate.py``, reached through the ``aggregate.distributions`` facade.

.. _agg pipeline two phases:

Two phases: construction and update
-----------------------------------

An :class:`Aggregate` has two clearly separated phases.

Construction (``__init__``)
    Parse and broadcast the exposure and severity specification into a list of :class:`Severity` objects, compute theoretical (closed-form) moments, and scaffold ``stats_df``. No discretization or FFT happens here. Construction is cheap and grid-independent.
Update (``update``, then ``update_work``)
    Choose a grid, discretize the severity, run the FFT convolution, apply reinsurance, and compute empirical moments from the resulting density. This phase is the expensive, grid-dependent one, and it can be re-run at a different ``(log2, bs)`` without rebuilding the object.

The single end-to-end data flow::

    build(DecL)                    parse to a spec dict
        |
        v
    Aggregate.__init__             broadcast -> self.sevs[], theoretical
        |                          moments, stats_df[comp_*, mixed,
        |                          independent]
        v
    update(log2, bs)               pick the grid (_bs_window -> round_bucket)
        |
        v
    update_work(xs)
        |
        +- discretize()            severity CDF/SF -> PMF per component
        +- sum w_i * bed_i         claim-count-weighted severity -> sev_density
        +- picks()                 optional layer-loss-pick reweighting
        +- apply_occ_reins()       per-occurrence reinsurance on sev_density
        +- _freq_sev_convolution() z=ft(sev); ftagg=PGF(z); agg=ift(ftagg)
        +- apply_agg_reins()       aggregate reinsurance on agg_density
        +- empirical moments       de-fuzzed copy -> stats_df[empirical, error]
        |
        v
    density_df  (lazy)             p_total, F, S, lev/exa, exlea, exgta, exeqa
        |
        v
    valid / q / tvar / price       risk measures and pricing

.. _agg pipeline construction:

Construction
------------

The constructor is the most intricate part of the class. Its job is to turn a flat spec, which may hold a limit profile (a vector of layers) or a mixed severity (a weighted list of distributions) or both, into:

* ``self.sevs``, a flat ``np.array`` of :class:`Severity` objects, one per **component**, where a component is one exposure by severity cell;
* ``self.en``, ``self.attachment`` and ``self.limit``, per-component vectors;
* ``self.n``, the total expected claim count;
* ``self.stats_df``, the canonical moment frame described in :ref:`agg pipeline stats_df`;
* the headline scalars ``actual_m``, ``actual_cv``, ``actual_skew``, ``sev_m``, ``sev_cv`` and ``sev_skew``, all theoretical.

Spec capture
~~~~~~~~~~~~

The very first thing ``__init__`` does is snapshot its own arguments through ``inspect.getargvalues(frame)`` into ``self._spec``, with ``self``, ``frame`` and ``get_value`` popped. That snapshot is what ``spec``, ``json`` and persistence round-trip against, which is why the argument names are the serialization format: renaming a constructor parameter is a breaking change to saved specs.

Frequency by composition
~~~~~~~~~~~~~~~~~~~~~~~~

``self.frequency = Frequency(freq_name, ...)`` dispatches through ``Frequency.__new__`` to the right ``Frequency<Kind>`` subclass. An :class:`Aggregate` has a frequency; it is not one. The frequency object exposes ``freq_moms(n)`` for moments and ``freq_pgf(n, z)`` for the PGF applied in the FFT.

The two broadcasting arms
~~~~~~~~~~~~~~~~~~~~~~~~~

There are two structurally different paths, chosen by ``if np.sum(sev_wt) == len(sev_wt)``, that is, by whether all weights are 1.

Limit-profile arm (all weights 1)
    Exposure terms and severity terms are ``np.broadcast_arrays``-ed together and zipped once. Each row is an independent component. Used for a vector of layers sharing one severity shape, and for a single severity.
Mixture-product arm (weights not all 1)
    Exposure terms and severity terms are broadcast separately and combined as an outer product. A single ground-up mixed severity is built first (``gup_sevs``) so that, for an excess layer, the mixture weights can be re-derived from each component's survival at the attachment (``w1 = wt * sf(attach)``, then ``w /= w.sum()``). The actual layered severities (``actual_sevs``) are then built and recorded per exposure row.

Both arms funnel each component through ``_record_component``, which writes the ``comp_<r>`` column of ``stats_df`` and accumulates into the shared ``MomentAggregator``. The two arms therefore cannot drift on what a component records, only on how components are enumerated.

Exposure resolves per component under the rule that claim count trumps loss. If ``en > 0`` then ``el = en * E[X]``; otherwise if ``el > 0`` then ``en = el / E[X]``; premium and loss ratio are back-solved similarly; and an empirical-frequency sentinel ``en < 0`` resolves to ``sum(freq_a * freq_b)``. A mixture component whose layer mean underflows to ``nan`` is silently replaced by a degenerate zero severity, a ``dhistogram`` at 0, so the FFT still has something well-formed to convolve.

.. _agg pipeline stats_df:

``stats_df``, the single source of truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The frame is shaped ``MultiIndex(component, measure)`` rows by ``[comp_0 ... comp_k, mixed, independent, empirical, error]`` columns, with ``dtype=object`` so the ``('meta','name')`` string can sit alongside floats.

The rows are a ``meta`` block (name, limit, attachment, el, prem, lr, sevcv_param, mix_cv, wt) followed by ``freq``, ``sev`` and ``agg`` crossed with ``(ex1, ex2, ex3, mean, cv, skew)``. The columns are:

``comp_i``
    this component's contribution.
``mixed``
    portfolio-of-one totals with frequency mixing (``remix=True``). These are the headline theoretical moments.
``independent``
    totals without the shared mixing distribution (``remix=False``), the frequency-independent comparison.
``empirical``
    filled by ``update_work`` from the FFT output.
``error``
    noise-aware relative error of ``empirical`` against ``mixed``.

Everything else, including ``summary_df``, ``valid`` and the headline scalars, reads from ``stats_df``. It is the canonical store.

.. _agg pipeline grid:

Choosing the grid and discretizing
----------------------------------

``update(log2, bs=0)`` is the convenience wrapper. With ``bs=0`` it calls the automatic sizer ``_bs_window``, which returns ``(bs, log2, x_min)``, and builds ``xs = x_min + arange(0, 2**log2) * bs``. The sizer runs several candidate methods and records each in ``_bs_window_df``, whether or not it wins; :ref:`num bucket selection` covers it in full.

The exact-binary ``bs`` invariant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``round_bucket`` codifies a hard-won rule: only use bucket sizes with an exact binary representation. For ``bs < 1`` it returns ``1 / 2**k``, the largest power-of-two fraction at or below the raw estimate; for ``bs > 1`` it snaps to 1, 2, 5 or 10 times a power of ten. An exact binary ``bs`` means ``loss = i * bs`` is representable, ``a / bs`` is an exact integer, and ``snap`` and index lookups are exact. That eliminates a whole class of off-by-one-bucket and key-not-found bugs. It is an invariant the rest of the code relies on, not a cosmetic nicety.

``discretize``
~~~~~~~~~~~~~~

``discretize`` converts each continuous or discrete :class:`Severity` into a PMF on the bucket grid. There are two orthogonal knobs.

``sev_calc`` controls bucket placement:

``discrete``, also spelled ``round``
    The default and the recommendation. Mass at :math:`b_i` represents :math:`\Pr(b_i - bs/2 < X \le b_i + bs/2)`, so the buckets are the exact discrete support. ``adj_xs`` are the midpoints and the array is treated as fully discrete downstream.
``forward``, also spelled ``continuous``
    :math:`\Pr(b_{i-1} < X \le b_i)`, with no shift.
``backward``
    Shifted the other way.
``moment``
    Raises ``NotImplementedError``. Embrechts says it is not worth it.

``discretization_calc`` controls how each bucket probability is computed:

``survival``
    The default. Backward difference of the survival function, most accurate in the right tail, which is the tail that matters.
``distribution``
    Forward difference of the CDF, most accurate in the left tail.
``both``
    The elementwise maximum of the two, which gets the best of each because underflow drives the wrong one to zero.

Two structural choices encode the defective-distribution philosophy described in :ref:`agg defective`. First, ``adj_xs[0] = -np.inf`` always, so the first bucket absorbs all mass at and below zero, since severity may have real support there. Second, the right end uses ``xs[-1] + bs/2``, one bucket past the top, rather than ``np.inf``. Extending to infinity would guarantee probabilities summing to 1 but would dump a visible mass into the last bucket; truncating one bucket past the top avoids that spurious mass at the cost of summing slightly short, which ``normalize`` then optionally rescales. A clean tail is worth a controlled, measurable deficit.

Severity assembly
~~~~~~~~~~~~~~~~~

``update_work`` assembles the claim-count-weighted severity. Per-component frequency weights ``wts = freq_ex1 / sum(freq_ex1)`` are read straight from ``stats_df``, and ``sev_density = sum(bed_i * w_i)``. Optional ``picks`` reweighting follows. Passing ``force_severity='yes'`` is the plot-only early exit, with no FFT.

Severity is the only thing that is optionally normalized. The aggregate never is.

.. _agg pipeline fft:

The FFT core
------------

``_freq_sev_convolution(padding)`` is the whole algorithm in three lines::

    z                  = ft(self.sev_density, padding)             # severity chf
    self.ftagg_density = self.frequency.freq_pgf(self.n, z)        # apply the PGF
    self.agg_density   = real(ift(self.ftagg_density, padding))    # back to a PMF

There are two shortcuts. A zero-risk aggregate (``n == 0``) is unit mass at 0, and the FFT runs only to give ``ftagg_density`` the right shape, which is needed if the aggregate lives in a :class:`Portfolio`. A fixed frequency of 1 (``sum(en) == 1`` with ``freq == 'fixed'``) means the aggregate is the severity, so the inverse FFT is skipped entirely. That is the ``agg A 1 claim ... fixed`` path; the general ``dfreq[1]`` path does the full round trip and so carries sub-eps FFT fuzz that the shortcut path does not.

``padding`` doubles (1) or quadruples (2) the working vector to mitigate FFT aliasing and wrap-around. ``ftagg_density`` is retained because :class:`Portfolio` multiplies unit transforms to combine them under the independence copula.

Reinsurance brackets the FFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FFT core itself is reinsurance-agnostic. Cover is applied on either side of it:

* ``apply_occ_reins`` runs before the convolution, on ``sev_density``. It builds ceder and netter step functions through ``make_ceder_netter``, re-grids the net and ceded severities onto the bucket grid, and replaces ``sev_density`` with the requested view. Gross is cached as ``sev_density_gross``.
* ``apply_agg_reins`` runs after the convolution, on ``agg_density``, identically, and then re-runs the FFT on the result to keep ``ftagg_density`` consistent.

:ref:`pipeline reinsurance` covers the re-gridding and the reports built on these densities.

.. _agg pipeline moments:

Empirical moments and ``density_df``
------------------------------------

Both severity and aggregate raw moments are computed once through ``xsden_to_mwrangler``, and ``(mean, cv, skew)`` are derived from the same :class:`MomentWrangler`, so ``ex123`` and ``mcvsk`` are mutually consistent. They are written to ``stats_df['empirical']``, and ``error`` is ``_noise_aware_rel_error`` against ``mixed``.

The de-fuzzed copy is a subtlety worth preserving. The raw inverse-FFT density carries sub-machine-eps fuzz in essentially every bucket. In a plain mass sum that cancels, because mass is conserved, but raw moments weight each bucket by :math:`x^k`, so on a wide grid the far-tail fuzz is amplified by :math:`x^3` and corrupts the empirical skewness: a symmetric die's skewness drifts from about 1e-15 to about 1e-4 as ``log2`` grows. Moments are therefore computed from a throwaway ``agg_clean = where(|agg_density| < eps, 0, agg_density)`` while ``self.agg_density`` is left raw, consistent with ``ftagg_density``. The substitution is safe because the FFT is exact up to rounding and the exact aggregate has no negative density even under aliasing, so every stray value is genuinely small.

``density_df`` is a lazy property, built on first access after ``update`` and cached in ``_density_df``. Its columns are documented in the property docstring. The construction facts worth knowing:

* ``p_total`` is ``agg_density`` after ``remove_fuzz`` has zeroed :math:`|x| < \epsilon`. It is the curated view, distinct from the raw ``agg_density`` member.
* ``p_sev`` is added after ``remove_fuzz`` so the carefully computed severity is not flushed away.
* ``S = 1 - p_total.cumsum()``. Survival is best computed forwards.
* ``lev`` and ``exa`` are the left-Riemann limited expected value ``S.shift(1).cumsum() * bs``; ``exlea`` and ``exgta`` are derived from them; ``e`` is the constant ``est_m`` column; and ``exeqa = loss``, since :math:`\mathsf E[X \mid X = a] = a`.
* Several columns are deliberate duplicates (``p`` equals ``p_total``, ``exa`` equals ``lev``, ``exeqa`` equals ``loss``). They exist so a unit can be regex-inlined into a :class:`Portfolio`.

.. _agg pipeline validation:

Validation
----------

The ``valid`` property returns a :class:`Validation` flag. It short-circuits to ``REINSURANCE`` (cannot validate) or ``NOT_UPDATED`` (the empirical aggregate mean is NaN). Otherwise it:

* flags ``SEV_MEAN`` or ``AGG_MEAN`` when the relative error in :math:`\mathsf E[X]` exceeds ``eps``, where ``eps`` is ``validation_eps``, by default 1e-4;
* flags ``ALIASING`` when the aggregate-mean error is both non-trivial (greater than :math:`\epsilon^3`) and more than ``ALIASING_RATIO`` times the severity-mean error, the signature of FFT wrap-around or too small a ``bs``;
* flags CV and skewness through ``np.isclose`` at ``rtol`` of 10 ``eps`` and 100 ``eps`` respectively, with ``atol=VALIDATION_NOISE``, and only when the theoretical value itself exceeds ``VALIDATION_NOISE``. A theoretically zero skewness or CV, from a symmetric or deterministic severity, is skipped, because the FFT's estimate of a zero higher moment is grid-dependent noise with no meaningful relative error.

A pass means "not unreasonable", a failure to reject the null, rather than "correct". Type-1 error, rejecting a good model, is preferred to type-2.

:meth:`sharpen`, described in :ref:`bs sharpen`, turns the same six error terms into a score and probes neighboring grids with it.

.. _agg pipeline conclusions:

Conclusions worth keeping
-------------------------

The narrative comments that used to sit in the source encode real conclusions.

Compute survival forwards.
    Post-FFT, ``S = 1 - F`` is correct and simplest. The precision argument for backward cumulation does not apply once you have been through an FFT.
Leave no mass at the top bucket.
    Discretize to ``xs[-1] + bs`` and normalize, rather than extending to infinity. A clean tail beats a guaranteed unit sum.
Protect the severity.
    ``p_sev`` is attached after ``remove_fuzz`` so exact severity work is not zeroed.
Take the fixed-1 shortcut.
    A frequency identically 1 means the aggregate is the severity, so skipping the inverse FFT buys both speed and accuracy.
Leave moment matching unimplemented.
    On purpose. Embrechts says it is not worth it.

.. _agg defective:

Defective distributions
-----------------------

A **defective** distribution has :math:`\sum p < 1`. In this library that is a feature rather than an error, and the pipeline is built to leave it alone:

* severity may be normalized; the aggregate never is;
* ``S`` is computed forwards, as ``1 - cumsum``, so for a defective aggregate ``S`` plateaus at the deficit instead of being forced to 0, which is exactly what pricing needs, and forward and backward survival calculations then agree;
* the empirical-moment helper places the deficit mass at the implied maximum ``xs[-1] + bs`` and logs at INFO when the deficit exceeds ``VALIDATION_NOISE``, so genuine defectiveness is distinguished from floating-point dust off 1.

Whether a bucketed distribution is genuinely unbounded is a separate question, and it is not answerable from the density. It surfaces in portfolio pricing, where a distortion places mass on an unbounded tail, and is settled there by reading the spec rather than the density. See :ref:`portfolio defective unbounded`.
