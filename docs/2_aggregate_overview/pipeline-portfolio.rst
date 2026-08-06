.. _pipeline portfolio:

The Portfolio Computation Pipeline
==================================

This section describes how a :class:`Portfolio` combines its units into a total, computes the conditional expectations that drive capital allocation, applies spectral distortions to price and allocate, and rebuilds itself around an empirical sample. It closes with the conclusions that shaped the design and a note on the hard cases: defective, unbounded, and mass distortions. The implementation is in ``_portfolio.py`` and the three subsystem modules ``_portfolio_common.py``, ``_portfolio_density.py`` and ``_portfolio_sample.py``, reached through the ``aggregate.portfolio`` facade.

.. _portfolio overview:

Overview
--------

A :class:`Portfolio` is a collection of :class:`Aggregate` units plus the machinery to do five things:

#. combine them into a portfolio total under the independence copula;
#. compute the conditional expectations :math:`\kappa_i(a) = \mathsf E[X_i \mid X = a]`, the kappas, that drive capital allocation;
#. apply spectral distortions to price and allocate;
#. generate multivariate samples, optionally with an induced rank correlation by reordering, the Iman-Conover shuffle;
#. run the switcheroo, rebuilding the object around an empirical sample's kappas.

The pipeline::

    Portfolio.__init__          collect units, build stats_df (theoretical)
        |
        v
    update(log2, bs)
        |
        +- per unit: agg.update(xs)   each unit's FFT, and its reinsurance
        +- p_total = ift(prod ft(unit_i))   combine under independence
        +- unit_state[name]           native grid, pmf and padded rfft
        +- remove_fuzz()
        +- add_exa(density_df, ...)   THE KAPPA ENGINE -> exeqa_*, exa_*, ...
        +- empirical total moments    -> stats_df[empirical, error]
        |
        v
    calibrate_distortion(s)     solve the distortion shape to hit a premium
        |                       or cost-of-capital target
        v
    apply_distortion            gS, gp_total, exag_* (cached per distortion)
        |
        v
    price / pricing_at          pricing readouts (linear or lifted)
    analyze_distortion(s)

The switcheroo is a big idea, not a side path. A :class:`Portfolio` can be rebuilt around an empirical multivariate sample: rather than carrying the full joint sample everywhere, :meth:`create_from_sample` and :meth:`add_exa_sample` replace the density frame with one whose kappas :math:`\mathsf E[X_i \mid X = a]` are estimated directly from the sample. Everything about the sample's dependency structure that matters for allocation is the kappa, so once you have it, every downstream calculation runs unchanged on the sample-based object: allocation, distortion pricing, the whole augmented-frame machinery. That is the win. A high-dimensional sample collapses to a one-dimensional family of conditional means, and all manner of things simplify. The mechanics are in :ref:`portfolio sample`.

.. _portfolio construction:

Construction
------------

``__init__`` accepts a heterogeneous ``spec_list``: :class:`Aggregate` objects, spec dicts, ``(kind, spec)`` tuples, names to look up in an :class:`Underwriter`, or a single :class:`pandas.DataFrame` of samples. Each resolved unit is appended to ``agg_list`` and ``unit_names``, set as an attribute (``self.<unit> = agg``), and its theoretical moments accumulated into a ``MomentAggregator``.

``unit_names`` may not contain ``total``, which is reserved. DataFrame input is stored as ``sample_df`` and the units become discrete histogram aggregates, rounded to 8 decimal places so close values do not merge, respecting the :math:`2^{-30} \approx 10^{-9}` discrete-grid resolution. ``_build_stats_df`` mirrors ``Aggregate.stats_df`` minus the ``independent`` column, since there is no portfolio-level mixed against independent distinction; a portfolio's theoretical column is named ``total`` rather than ``mixed``.

As with :class:`Aggregate`, no FFT happens in construction.

.. _portfolio update:

Update and the kappa engine
---------------------------

``update`` resolves ``bs`` through ``_bs_window`` when ``bs=0``, builds the grid, and then, for each unit, calls that unit's own ``update``, which also applies the unit's reinsurance. It accumulates the product of the unit transforms in Fourier space, so ``p_total = real(ift(prod ftagg_density_i))`` is the independence-copula combine.

Two combine paths exist. The plain path drives every unit on the shared grid. The roll path, used for a signed (P&L) book or a windowed non-signed total, drives each unit on its own window sharing the portfolio's ``bs``, ``log2`` and ``padding``, so each unit stays internally correct, and then relabels the combined density by a single modular roll. The combine reads each unit's ``ftagg_density``, which sits at origin 0 whatever the unit's ``x_min``, so the transforms still multiply correctly.

Each unit contributes a transient ``unit_state`` entry holding its native loss grid ``xs``, its pmf ``p``, and the padded real FFT of that pmf, ``ft_p``. The state is freed as soon as ``add_exa`` returns.

Empirical total moments use the same de-fuzzed ``xsden_to_mwrangler`` convention as :class:`Aggregate`, written to ``stats_df['empirical']`` and ``error`` through ``_write_empirical_stats``.

``add_exa``, the kappa engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``add_exa`` is the heart of the :class:`Portfolio`. For each unit it computes, on the ``density_df`` grid, the money calculation

.. math::

    \kappa_i(a) = \mathsf E[X_i \mid X = a],

which falls straight out of one FFT pair because :math:`X_i` and :math:`X_{-i}` are independent. The numerator is the inverse transform of :math:`\mathrm{ft}(x \, p_i) \cdot \mathrm{ft}(p_{-i})`, where the first factor is the transform of the unit's native first-moment density :math:`x\,p_i(x)`. First moments, unlike probabilities, cannot be recovered from a rolled vector, so they are built from the true physical values and the result is relabelled onto the output window by the same single roll the combine applies to ``p_total``.

From :math:`\kappa_i` the full allocation column family follows:

``exlea_i``
    :math:`\mathsf E[X_i \mid X \le a]`.
``exgta_i``
    :math:`\mathsf E[X_i \mid X > a]`.
``exi_xgta_i``
    :math:`\mathsf E[X_i / X \mid X > a]`, alpha, the objective tail share.
``exa_i``
    :math:`\mathsf E[X_i(a)]`, the equal-priority expected loss allocated to unit :math:`i` at assets :math:`a`.
``lev_i``
    the stand-alone :math:`\mathsf E[X_i \wedge a]`.

Every cumulative column is a direct sum that carries the origin, for example :math:`\mathsf E[X \wedge a] = \sum_{x \le a} x\,p + a\,S(a)`, never ``cumsum(S) * bs``, which would silently assume the grid starts at 0. Ratio denominators carry explicit guards: where ``F`` or ``S`` sits at or below the validation noise floor the row is blanked.

On a signed (P&L) grid the equal-priority share :math:`\kappa / x` is not a recovery share, so the share-based columns are left ``NaN`` rather than divided through zero. The conditional means, ``lev_*``, and the total columns remain valid on any signed window.

.. _portfolio pricing:

Distortions and pricing
-----------------------

Calibration
~~~~~~~~~~~

:meth:`calibrate_distortion` resolves the asset level, the premium target, and the survival vector over :math:`[0, a]`, truncating at the first zero, recording ``ess_sup`` and logging when the support ends early. It asserts that ``S`` is strictly positive and weakly decreasing, then dispatches to the :class:`Distortion` subclass whose ``calibrate`` runs the Newton iteration. The per-distortion mathematics lives in ``spectral.py``.

:meth:`calibrate_distortions` inverts cost of capital to loss ratio to premium, and calibrates the standard set ``[ccoc, ph, wang, dual, tvar]``, storing them on ``self.distortions`` alongside an audit ``distortion_df``.

The augmented frame
~~~~~~~~~~~~~~~~~~~

:meth:`apply_distortion` is a thin cache keyed by distortion, invalidated on ``update`` and readable through :attr:`augmented_df` and :attr:`augmented_dfs`. The work is done by ``build_augmented``, a pure builder that returns a copy of ``density_df`` carrying the risk-adjusted quantities: ``gS``, ``gF``, ``gp_total``, and per unit ``exag_i``, the risk-adjusted allocation premium.

One :math:`O(n)` sweep serves both allocation methods and every asset level:

.. math::

    \mathrm{exag}_i(a) = \sum_{k \le a} \kappa_i(x_k)\, gp_k
                        + a \, g(S(a)) \, \mathrm{TAIL}_i(a).

The distorted atom weights :math:`gp` come from the one Choquet helper, :func:`~aggregate.spectral.choquet_weights`, and the effective :math:`g` resolves the pricing ``view`` against the portfolio's value-type role through :meth:`~aggregate.spectral.Distortion.effective_g`. The frame is truncated at the last reliable ``exeqa`` row, an FFT-noise cut, with the tail sums feeding the allocation computed on the full law first.

Linear and lifted natural allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two allocation methods differ in exactly one term, :math:`\mathrm{TAIL}_i`, the share that allocates the collapsed default states:

linear
    :math:`\mathrm{TAIL} = \mathrm{exi\_xgta}`, alpha, the objective tail share. Because the tail collapse uses objective rather than risk-adjusted probabilities, it sidesteps applying the distortion far out in the tail, which makes it much more stable. It is the linear natural allocation of *Pricing Insurance Risk* :cite:p:`Mildenhall2022a`, and it is the default: :attr:`allocation_method` is ``'linear'`` out of the box.
lifted
    :math:`\mathrm{TAIL} = \mathrm{exi\_xgtag}`, beta, the distorted tail share. The premium then uses risk-adjusted weights all the way out into the tail.

Because one sweep serves both, linear costs essentially the same as lifted; there is no performance reason to prefer either.

The difference matters at the right edge. A distortion with a mass at zero, such as constant cost of capital, on an unbounded support puts essentially all the lifted tail weight on the single last represented bucket, which is unstable and grid-dependent. The builder refuses that combination outright rather than returning an unstable answer, and the error points at ``linear``, or at certifying ``portfolio.bounded = True`` when the support really is bounded. The linear split, and every total column, depends on the tail only through :math:`g(S(a))` and is stable; with a mass on an unbounded support the linear frame is built with the beta columns blanked. See :ref:`portfolio defective unbounded`.

Readouts
~~~~~~~~

:meth:`pricing_at`
    Per-unit pricing at one asset level, which warms the cache. The column order is the pentagon, amounts then ratios: ``L M P Q a | LR PQ ROE``, with ``LR = L/P``, ``PQ = P/Q`` and ``a = P + Q``. See ``pentagon.py``. ``Q_total`` is ``a - exag_total`` exactly.
:meth:`price`
    Prices :math:`\rho(X \wedge q(p))` for one distortion, or a list or dict of them, and allocates to units. ``p`` is read as a probability when it is at most 1, converted to assets by VaR and snapped to the index, and as an asset level otherwise. ``allocation`` defaults to :attr:`allocation_method`. It returns a ``PricingResult`` carrying the per-unit frame in pentagon columns, the total price, a ``price_dict`` keyed by distortion, and ``a_reg`` and ``reg_p``.
:meth:`analyze_distortion`, :meth:`analyze_distortions`
    Single and multi-distortion exhibits built on the same machinery.
:meth:`allocation_diagnostics`
    The per-unit reconciliation, checking that :math:`\sum_i Q_i(a) = a - \mathrm{exag\_total}(a)` exactly.

.. _portfolio sample:

The sample path, or switcheroo
------------------------------

For working from an empirical multivariate sample rather than parametric units:

:meth:`add_exa_sample`
    Aligns the sample total to the bucket grid, as ``round(total/bs) * bs``, rescales the per-unit losses to preserve the total, groups by total to form :math:`\mathrm{exeqa}_i = \mathrm{mean}(X_i \mid \mathrm{total})` and ``p_total`` as the summed sample probabilities, then re-runs the ``add_exa`` derivations on that density.
:meth:`create_from_sample`
    Builds a :class:`Portfolio` from the sample, updates it to get the independent FFT density, archives that as ``independent_density_df`` and ``independent_stats_df``, then swaps in the sample-based ``density_df`` (the switcheroo) and recomputes the empirical total moments. :meth:`sample_compare` and :meth:`sample_density_compare` diff the two views.
:meth:`swap_density_df`
    Experimental. Injects marginal densities directly into a shell :class:`Portfolio` with the right unit names, recomputes the total, and runs ``add_exa``, sidestepping all :class:`Aggregate` construction.

:meth:`make_comonotonic_allocations` post-processes the kappas into a comonotonic, convex-order improvement, following Denuit and coauthors.

.. _portfolio conclusions:

Conclusions worth keeping
-------------------------

Compute survival forwards, post-FFT.
    After years of trying clever backward survival cumulations, the conclusion is ``S = 1 - F``. The precision loss is immaterial because you have already been through FFTs. The comment in ``add_exa`` quotes Eliot on the end of all our exploring.
The money calculation is one FFT pair.
    :math:`\kappa_i` exploits the independence of :math:`X_i` and :math:`X_{-i}`, and nothing else is needed.
Do not trust conditional expectations where there is no probability.
    Where ``p_total`` falls below the noise floor the estimate is unreliable, and blanking the row rather than reporting it makes a visible difference.
Layer ROE is law-invariant.
    All units share the same layer ROE, and the L'Hopital limit of :math:`(gS - S)/(1 - gS)` as :math:`S \to 1` is :math:`\mathrm{ROE}(1) = 1/g'(1) - 1`. That is what makes per-unit capital allocation well defined.
Lifted gave way to linear.
    Moving from the lifted natural allocation, which carries distortion weights into the tail and is unstable against a mass, to the linear natural allocation, which collapses the tail with objective probabilities, is the single most important conceptual evolution in this pipeline.
The last bucket knows nothing beyond the array.
    Reverse-cumulative tail fills must therefore be set explicitly. The library calls that gap the **John Major problem**.

.. _portfolio defective unbounded:

Defective, unbounded, and mass distortions
------------------------------------------

A defective total (:math:`\sum p < 1`) is handled by not normalizing, by computing ``S`` forwards, and by letting ``S`` plateau at the deficit. :meth:`calibrate_distortion` already trims ``S`` at the first zero and records ``ess_sup``.

Mass distortions, such as constant cost of capital, put weight on the survival function's jump. On a bounded support that is fine. On an unbounded support under the lifted allocation the mass lands on the last bucket, which is unstable, and the builder refuses the combination. The linear allocation avoids the problem by collapsing the tail with objective probabilities.

Boundedness is read from the spec, not from the density. Inferring it from a bucketed density is ill-posed: ``agg 100 claims dsev[10000]`` is unbounded yet has positive mass only on multiples of 10,000, so a test on "tail probabilities are positive" fails against structural zeros. The spec answers it cleanly:

.. math::

    \text{aggregate bounded} \iff
    \text{frequency bounded} \;\wedge\; \text{per-claim severity bounded}.

Frequency boundedness is a ``freq_name`` lookup. ``fixed``, ``bernoulli``, ``binomial`` and empirical ``dfreq`` are bounded; ``poisson``, ``negbin``, ``geometric``, ``logarithmic`` and all mixed-Poisson families are unbounded, since :math:`N` can be arbitrarily large. Severity is bounded when there is a finite ``exp_limit``, or a finite ``sev_ub``, or the severity is discrete with finite support (``dsev``), or the family is inherently bounded, such as beta or uniform.

That rule correctly classifies the ``dsev[10000]`` example as unbounded, not because of the severity, which is a single bounded point mass, but because the ``poisson`` frequency is unbounded. Two caveats keep it conservative. Mixtures can pair bounded with unbounded components. And there is a gap between mathematical unboundedness and numerical danger: a thin Poisson crossed with a bounded severity is mathematically unbounded yet perfectly safe under the lifted allocation. A spec rule may therefore flag safe cases, which is why the user can certify ``bounded = True`` to override it.
