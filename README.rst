|  |activity| |doc| |version|
|  |py-versions| |downloads| |stars| |forks| 
|  |license| |packages| |zenodo|

-----

aggregate: working with actuarial compound distributions
===========================================================

Purpose
-----------

``aggregate`` builds approximations to compound (aggregate) probability distributions quickly and accurately.
It can be used to solve insurance, risk management, and actuarial problems using realistic models that reflect
underlying frequency and severity. It delivers the speed and accuracy of parametric distributions to situations
that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution
as the lognormal. ``aggregate`` includes an expressive language called DecL to describe aggregate distributions
and is implemented in Python under an open source BSD-license.

Aggregate White Paper
----------------------

`Aggregate: fast, accurate, and flexible approximation of compound probability distributions <https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/aggregate-fast-accurate-and-flexible-approximation-of-compound-probability-distributions/1BF9A534D944D983B1D780C60885F065>`_ describes the ``Aggregate`` class within ``aggregate``. This paper has been published in the peer reviewed journal `Annals of Actuarial Science <https://www.cambridge.org/core/journals/annals-of-actuarial-science>`_'s Actuarial Software series.
The paper describes the purpose, implementation, and use ``Aggregate``, showing how it can be used to create and manipulate compound frequency-severity distributions.

Version History
-----------------

.. Conda Forge: https://github.com/conda-forge/aggregate-feedstock https://anaconda.org/conda-forge/aggregate/files

1.0.0a8 (in progress)
----------------------

Portfolio refactor sub-project E — stats consolidation. Six overlapping
``Portfolio`` stats frames (``statistics_df``, ``statistics``,
``report_df``, ``report``, ``audit_df``, ``make_audit_df``) collapsed
into a single canonical ``stats_df``. Public stats surface on
``Portfolio`` is now exactly three things: ``info``, ``describe``,
``stats_df`` — same shape as ``Aggregate``.

* **``stats_df``** is a ``DataFrame`` with MultiIndex on
  ``(component, measure)`` rows (``meta`` + ``freq`` + ``sev`` + ``agg``
  blocks) and columns one-per-unit + ``total`` + ``empirical`` + ``error``.
  Per-unit columns hold each ``Aggregate.stats_df['mixed']`` (the
  unit's own view); ``total`` is the portfolio-aggregate theoretical
  view (sum of each unit's ``mixed``); ``empirical`` is the post-FFT
  combined view; ``error = empirical / total - 1``.
* Column is ``total`` rather than Aggregate's ``mixed``: at the
  Portfolio level there is no mixed-vs-independent distinction
  (mixed-vs-independent is an Aggregate-only concept that strips a
  single agg's freq mixing distribution).
* **Empirical column is fully populated**:

  - ``('agg', *)`` rows — raw moments ``ex1`` / ``ex2`` / ``ex3`` plus
    ``mean`` / ``cv`` / ``skew``, computed straight from the
    portfolio-total FFT density (plain summation, no tail-mass
    correction — matches the PEG baseline numerics).
  - ``('sev', *)`` rows — raw moments and central moments,
    re-aggregated from each unit's empirical sev mean/cv/skew via a
    fresh ``MomentAggregator``. ``Aggregate.stats_df`` stores only
    empirical mean/cv/skew for sev, so the raw moments are inverted
    via ``MomentWrangler`` before being fed to the aggregator.
  - ``('meta', *)`` rows for ``limit`` / ``attachment`` / ``el`` /
    ``prem`` / ``lr`` — copied across from ``total`` with implied
    ``error = 0`` (these are factual or sums of expected values,
    no FFT analog).
  - ``('freq', *)`` rows stay ``NaN`` in ``empirical`` — frequency is
    exact (no convolution operates on it); same convention as
    ``Aggregate.stats_df``.
* **Meta totals tightened**:

  - ``total[('meta', 'attachment')]`` = ``0`` when every unit attaches
    at 0 (previously ``NaN``); ``NaN`` only when units disagree.
  - ``total[('meta', 'limit')]`` = ``max`` across units (legacy
    convention preserved).
  - ``total[('meta', 'lr')]`` = ``el / prem`` when ``prem > 0`` else
    ``NaN``.
* **``('agg', 'P99.9e')`` row dropped** — Aggregate dropped the
  estimated-99.9th-percentile row in Stage 1c+; Portfolio follows
  suit. Percentile access via ``port.q(p)`` / ``port.var_dict(p)``
  remains.
* ``describe`` and the headline ``agg_m`` / ``agg_cv`` / ``agg_skew`` /
  ``est_m`` / ``est_cv`` / ``est_skew`` now read from ``stats_df``.
  ``describe`` total row surfaces empirical sev mean/cv/skew (was
  blank before — sev empirical only existed per-unit).
* ``extensions.portfolio_pir.accounting_economic_balance_sheet`` and
  ``extensions.bodoff`` updated to read ``stats_df``. The remaining
  ``case_studies`` exhibit code keeps its old ``audit_df`` references —
  those extensions are slated for removal at 1.0 per the master plan.
* PEG regression baseline unchanged (numbers reproduce bit-identically
  at ``rtol=1e-10``).

Housekeeping in the same release block:

* ``extensions.portfolio_pir.gamma`` (the ~136-LOC conditional layer
  effectiveness γ exhibit) and its ``GammaResult`` dataclass deleted —
  both were orphaned: not called from ``make_all``, not exercised by
  any test, not referenced in any rendered doc.
* ``Underwriter.__repr__`` gains a one-line usage hint pointing at
  ``.discover(regex)`` — fills the discoverability gap left when
  ``qshow`` / ``qlist`` / ``show`` were removed in 1.0.0a1.
* Eight stale comments and docstrings across ``distributions.py`` /
  ``utilities.py`` / ``portfolio.py`` that still mentioned
  ``statistics_df`` / ``report_ser`` / ``audit_df`` (in their
  stats-consolidation sense) refreshed. Distinct
  ``reinsurance_audit_df`` / ``reinsurance_report_df`` attributes
  and the ``audit_df`` field on ``AnalyzeDistortionResult`` are
  unrelated and unchanged.

1.0.0a7
-------

Portfolio refactor sub-project D — distortion-pricing pipeline redesign.
Six related changes that together collapse ~500 LOC of pricing code into
a small cache + a single signature convention:

* **D.1 — augmented_df lazy-eval cache.** ``Portfolio.apply_distortion``
  becomes a thin cache lookup-or-build keyed on distortion name; the
  construction logic lives in a private ``_build_augmented``. Second
  calls return the cached frame (``frame_a is frame_b``).
  ``port.augmented_dfs`` is a dict view of the cache; ``port.augmented_df``
  is the clean read-side accessor (also routes through the cache).
  ``apply_distortion`` drops the ``df_in=`` (gradient path, gone),
  ``create_augmented=`` (the cache replaces it), and ``plots=``
  (uninvoked) kwargs. ``apply_distortions`` (plural) deleted. New
  ``Portfolio.pricing_at(distortion, *, p=None, a=None)`` consolidates
  the row-extraction logic that previously lived in ``price`` and
  ``analyze_distortion``.
* **D.2 — analyze_distortion(s) and calibrate_distortions collapse onto
  the cache.** Each former 100-250 LOC method becomes ~25 LOC.
  ``analyze_distortion(distortion, *, p=None, a=None)`` returns an
  ``AnalyzeDistortionResult`` dataclass with ``pricing_df`` and
  ``audit_df``. ``analyze_distortions(*, p=None, a=None,
  distortions=None)`` returns ``AnalyzeDistortionsResult`` with the
  multi-distortion exhibit (MultiIndex ``(distortion, stat)``) and a
  cache snapshot. ``analyze_distortions2`` and the list-based
  ``calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, …)`` deleted in
  favour of single-coc / single-p forms.
* **Explicit ``p=`` / ``a=`` convention** across the pricing surface
  (``pricing_at``, ``analyze_distortion``, ``analyze_distortions``,
  ``calibrate_distortions``). The legacy implicit ``p > 1 → asset``
  threshold is gone; callers state intent. Each method raises
  ``ValueError`` if both or neither is supplied.
* **D.3 — Answer → typed dataclasses.** The legacy ``Answer`` dict
  class is deleted. ``aggregate.results`` defines ``PricingResult``,
  ``PricingBoundsResult``, ``AnalyzeDistortionResult``,
  ``AnalyzeDistortionsResult``, and ``GammaResult`` (the last used by
  ``extensions.portfolio_pir``). Inline ``namedtuple`` definitions in
  ``Portfolio.price`` and ``Portfolio.pricing_bounds`` promoted to the
  same module.
* **D.4 — ordered categoricals.** ``aggregate.spectral.DISTORTION_ORDER``
  / ``DISTORTION_DTYPE`` (``ccoc, ph, wang, dual, tvar, wtdtvar, lep,
  ly, clin, tt, cll, bitvar, blend``) and
  ``aggregate.portfolio.PRICING_STAT_ORDER`` / ``PRICING_STAT_DTYPE``
  (``L, LR, M, P, PQ, Q, ROE``) bake the canonical order into the data.
  ``Portfolio.distortion_df`` ``method`` index level, ``pricing_at``
  columns, and ``analyze_distortions`` pricing_df ``distortion`` level
  are typed categoricals -- ``sort_index()`` produces the canonical
  order without ad-hoc reordering.
* **D.5 — renames.** ``Portfolio.dists`` → ``Portfolio.distortions``;
  ``Portfolio.dist_ans`` and the ``distortion_df`` property merged into
  a single ``Portfolio.distortion_df`` attribute with the trimmed 9-col
  layout (``S, L, P, PQ, Q, COC, param, std_param, error``) and index
  names ``('a', 'LR', 'method')``; ``Portfolio.limits`` →
  ``Portfolio._limits`` (internal helper).
* PEG regression baseline unchanged -- ``test_pricing`` at ``rtol=1e-8``
  reproduces the legacy ``analyze_distortions2`` exhibit bit-identically.
  The new pipeline is mathematically the same; only the API surface changed.

1.0.0a6
-------

Portfolio refactor sub-project C — distortion calibration moves to the
``Distortion`` subclasses themselves:

* ``Portfolio.calibrate_distortion`` was ~240 LOC of per-name Newton
  iterations in a giant ``if name == 'ph': ... elif name == 'wang': ...``
  switch. Each branch defined a local ``f(shape) → (residual, derivative)``
  closure and ran a hand-rolled Newton loop. That code now lives on the
  ``Distortion`` subclasses — each pricing-distortion class owns its own
  ``calibrate(S, bs, premium_target, *, ess_sup, assets, el, **kwargs)``
  method: ``PHDistortion``, ``WangDistortion``, ``DualDistortion``,
  ``TVaRDistortion`` (``max_iter=200``), ``CCoCDistortion`` (closed-form,
  no iteration), ``LYDistortion``, ``CLinDistortion``, ``LEPDistortion``,
  ``CLLDistortion``.
* ``Portfolio.calibrate_distortion`` shrinks to ~100 LOC — about half
  asset/S resolution (unchanged), about half dispatch to the subclass via
  ``Distortion._registry``. The ``tt`` (Wang-t) branch is gone — there is
  no ``TtDistortion`` subclass to host it and the branch was dead code.
  ``wtdtvar`` calibration is also dropped from the dispatcher (the
  parametrisation overload between calibration form ``(w, [p0, p1])`` and
  the standard form ``(ps, wts)`` was already broken in the constructor;
  pick a pricing distortion that calibrates cleanly instead).
* New ``Distortion`` base-class methods ``_newton_iterate(f, shape, *,
  max_iter, tol)`` and ``_finalize_calibration(shape, fx, prem, assets)``
  factor the Newton loop and the post-iteration bookkeeping (write
  ``shape`` / ``error`` / ``premium_target`` / ``assets``, log on
  non-convergence, re-run ``_build`` to refresh cached state) out of the
  per-subclass methods.
* Class attribute ``Distortion._calibration_init_shape`` is the
  per-kind starting shape used both to construct the uncalibrated
  distortion and as the Newton iteration's starting point. ``None`` on
  the base means "not calibratable through the Portfolio dispatch."
* Each subclass is now testable in isolation. New
  ``tests/test_distortion_calibrate.py`` (12 cases) exercises every
  migrated kind directly on a synthetic ``S`` vector and asserts the
  achieved premium matches the target.
* PEG regression baseline unchanged — the new subclass-based Newton
  iteration reproduces bit-identical Newton convergence.

1.0.0a5
-------

Portfolio refactor sub-project B — drop approximation and tilting paths
from ``Portfolio.update`` and ``Aggregate.update_work``:

* Removed the auto-fallback method-of-moments approximation path. The
  ``approx_freq_ge`` / ``approx_type`` / ``approximation`` kwargs are gone
  from ``Portfolio.update``; the matching ``approx_type`` /
  ``approx_freq_ge`` attrs are gone from ``Portfolio.__init__``,
  ``Portfolio.json``, and ``Portfolio.__repr__``. The
  ``'exact' if agg.n < approx_freq_ge else approx_type`` ternary is gone;
  callers always get the FFT path. The slognorm / sgamma branch in
  ``Aggregate.update_work`` (and the ``approximation`` attribute on
  ``Aggregate``) is deleted. ``Portfolio.approximate`` /
  ``Aggregate.approximate`` (the user-facing on-demand
  method-of-moments fit returning a ``scipy.stats`` frozen RV or a DecL
  spec) are unchanged.
* Removed FFT tilting (Grübel/Hermesmeier 1999) from the update pipeline:
  the ``tilt_amount`` attr is gone from ``Portfolio.__init__``, the
  ``tilt_vector`` construction block is gone from ``Portfolio.update``,
  and the ``tilt=`` parameter is removed from the ``ft`` / ``ift``
  module-level helpers in ``aggregate.utilities`` and the matching
  ``Portfolio.ft`` / ``Portfolio.ift`` wrappers. The tilt branches inside
  ``Aggregate.update_work``, ``Aggregate._freq_sev_convolution``, and
  ``Aggregate.apply_agg_reins`` are gone. Use more buckets if aliasing
  shows up — per author's standing preference.
* ``aggregate.extensions.figures.gh_example`` was the only consumer of
  tilting in the visualisation layer; it now compares the padded FFT
  result against the exact compound probability without the
  tilt-comparison loop.
* PEG regression baseline (``tests/data/peg_baseline.json``) re-captured
  against the exact FFT path. The previous baseline incidentally
  exercised slognorm — PEG's two units (n=100 and n=150) tripped the
  default ``approx_freq_ge=100`` threshold. The drift is ~5e-6 on
  ``est_m`` and ~2e-5 on pricing cells; the new contract is the
  exact-FFT result.

Portfolio refactor sub-project A — pure deletions + PIR move
(``portfolio.py`` shrinks from 6,133 → 3,707 LOC):

* Deleted ~700 LOC of dead code from ``Portfolio``: ``gradient`` (~196 LOC),
  non-spectral allocations (``merton_perold``, ``cotvar``,
  ``equal_risk_var_tvar``, ``equal_risk_epd``), the EPD / priority /
  collateral family (``analysis_priority``, ``analysis_collateral``,
  ``priority_capital_df``, ``epd_2_assets``, ``assets_2_epd`` properties
  plus their backing attrs), the ``uat`` / ``uat_differential`` /
  ``uat_interpolation_functions`` trio, ``collapse``, ``audits``,
  ``stat_renamer``, and the ``var_dict(kind='epd')`` branch.
* Stripped ``analyze_distortion_add_comps`` and
  ``analyze_distortion_plots`` (~470 LOC) — both consumed the deleted
  allocation methods. ``analyze_distortion`` keeps ``add_comps`` and
  ``plot`` parameters as no-op defaults (``add_comps=False`` now).
* Moved ~1,800 LOC of PIR-exhibit machinery to the new
  ``aggregate.extensions.portfolio_pir`` module as free functions taking
  a ``Portfolio`` as the first argument: ``premium_capital``,
  ``multi_premium_capital``, ``accounting_economic_balance_sheet``,
  ``make_all``, ``show_enhanced_exhibits``, ``set_a_p``,
  ``profit_segment_plot``, ``natural_profit_segment_plot``,
  ``density_sample``, ``biv_contour_plot``, ``twelve_plot``,
  ``short_renamer``, ``gamma``, ``stand_alone_pricing``,
  ``stand_alone_pricing_work``, ``calibrate_blends`` (with helpers
  ``check01`` / ``make_array`` / ``convex_points``), the bulk
  constructors ``from_DataFrame`` / ``from_Excel`` /
  ``from_dict_of_aggs``, and the big ``renamer`` plus
  ``premium_capital_renamer``.
* ``aggregate.extensions.case_studies`` updated to call the moved
  functions as free functions; ``aggregate.extensions.bodoff`` inlines
  the deleted ``cotvar`` lookup.

1.0.0a4
--------

Portfolio refactor sub-project 0 — PEG regression baseline:

* New regression fixture ``tests/peg.py`` exposes ``build_peg`` which
  constructs the canonical two-unit ``port PEG`` Portfolio (limit-and-
  attachment severity, three-component lognormal severity mixture per
  unit, gamma frequency mixing with different mixing CVs per unit).
* New capture script ``tests/capture_peg_baseline.py`` runs PEG through
  ``calibrate_distortions(COCs=[.15], Ps=[.995])`` and
  ``analyze_distortions2(.995)`` for the five-distortion suite
  (``ccoc``, ``ph``, ``wang``, ``dual``, ``tvar``) and writes the
  numerical baseline to ``tests/data/peg_baseline.json``.
* New test module ``tests/test_portfolio_peg_regression.py`` pins
  portfolio moments (``rtol=1e-10``), per-distortion calibration shapes
  (``rtol=1e-8``, ``|error| < 1e-5``), and every cell of the
  ``analyze_distortions2`` exhibit (120 values, ``rtol=1e-8``).
* Every subsequent Portfolio refactor sub-project (A through E) must
  reproduce these baseline numbers; the JSON is the contract.

``Aggregate`` stats consolidation — finish the job: eliminate the
``_statistics_df`` / ``_statistics_total_df`` scratch frames so ``stats_df``
is the only theoretical-moment DataFrame the class holds:

* ``Aggregate.__init__`` now pre-creates an empty ``stats_df`` (canonical
  ``MultiIndex`` rows, NaN-filled) right after ``n_components`` is known in
  each broadcasting arm, via a new ``_init_stats_df`` helper.
* ``_record_component`` writes a column of ``stats_df`` directly (no more
  intermediate row in ``_statistics_df``).
* The post-loop totals block writes ``mixed`` / ``independent`` /
  ``('meta', 'wt')`` directly into ``stats_df`` columns.
* ``('agg', 'P99.9e')`` row dropped — it had only two populated cells
  (``mixed`` and ``independent``), was read in one spot (``_limits``
  fallback when ``agg_density`` is ``None``), and is cheaply rebuildable
  on demand via ``estimate_agg_percentile``. That one read site now
  computes on the fly.
* All readers migrated: ``avg_limit`` / ``avg_attach`` / ``tot_prem`` /
  ``tot_loss``, ``self.agg_m`` / ``agg_cv`` / ``agg_skew`` / ``sev_*``,
  ``update_work`` severity weights, ``severity_error_analysis`` weights,
  ``info`` / ``_html_info_blob`` component count.
* ``_statistics_df``, ``_statistics_total_df``, and the
  ``_build_stats_df`` method are gone.
* Side benefit: ``stats_df`` row layout is now cleaner — all ``meta`` rows
  together at the top (``mix_cv`` and ``wt`` previously trailed at the
  bottom because of how the legacy scratch frames were ordered).

1.0.0a3
--------

``Aggregate`` stats consolidation: six overlapping moment DataFrames → one
``stats_df`` (breaking changes; v1.0 cleanup):

* New canonical ``Aggregate.stats_df``: single source of truth for moment
  statistics. ``MultiIndex (component, measure)`` rows (``component`` ∈
  ``{meta, freq, sev, agg}``; ``measure`` ∈ ``{mean, cv, skew, ex1, ex2,
  ex3, …}``); columns are per-component (``comp_0``, …), ``mixed``,
  ``independent``, ``empirical``, and ``error``. Built in two phases:
  theoretical content in ``__init__``, ``empirical`` and ``error`` appended
  in ``update_work`` after the FFT. Empty cells are ``NaN`` where
  meaningful (e.g. ``('freq', *) × empirical`` is undefined — the FFT
  produces one combined empirical distribution, not per-component
  empirical moments).
* Naming convention unified: ``ex1`` / ``ex2`` / ``ex3`` for raw moments
  and ``mean`` / ``cv`` / ``skew`` for derived. The legacy ``_1`` / ``_m``
  flat-column convention is gone.
* The Aggregate "stats surface" is now exactly three things — ``info`` (text
  about the Aggregate), ``describe`` (the daily-driver moment audit), and
  ``stats_df``. Removed: ``report_df``, ``report_ser``, ``statistics``,
  ``audit_df``. Privatised: ``statistics_df`` → ``_statistics_df``,
  ``statistics_total_df`` → ``_statistics_total_df``.
* ``Aggregate.describe`` rewritten to source from ``stats_df``; output
  byte-identical.
* ``Portfolio`` migrated to read ``a.stats_df['mixed']`` instead of
  ``a.report_ser`` (three lines in ``portfolio.py``). Portfolio's own
  ``statistics_df`` / ``audit_df`` / ``report_df`` are unaffected — they
  live on Portfolio, not Aggregate, and will be rationalised in Stage 2.
* Docs migrated: ~30 references to ``report_df`` / ``statistics`` /
  ``statistics_df`` across nine tutorial pages rewritten to use
  ``stats_df`` with explicit row / column accessors.

1.0.0a2
--------

Aggregate surface rationalization (breaking changes; v1.0 cleanup):

* Visible layer structure: file-level section dividers in ``distributions.py`` and a public-API block in the ``Aggregate`` class docstring document the integration surface (``report_ser``, ``statistics_df``, ``update_work``, ``agg_density``, ``ftagg_density``, ``density_df``, plus the risk-measure surface ``q`` / ``tvar`` / ``cdf`` / ``sf`` / …) that ``Portfolio`` and ``Bounds`` consume.
* FFT five-line core extracted to ``Aggregate._freq_sev_convolution``; docstring references the four-step algorithm in §2.2 of the paper. ``update_work`` reads top-to-bottom as compute-severity → occurrence reinsurance → convolution → aggregate reinsurance → audit.
* Shared inner-block of ``__init__``'s two broadcasting arms factored into ``Aggregate._record_component`` (centralises ``statistics_df`` column ordering across the limit-profile arm and the mixture-product arm).
* ``__init__`` state initialization regrouped into labelled blocks: spec passthroughs, grid + runtime config, exposure outputs, computed densities, empirical moment estimates, cached lazy functions, reinsurance state, theoretical moment tables.
* ``density_df`` property docstring expanded with a column-by-column reference table (set-by / read-by for each of 17 columns) — no behavior change.
* Aggregate methods privatised (leading underscore): ``audit_df`` → ``_audit_df``, ``statistics_total_df`` → ``_statistics_total_df``, ``limits`` → ``_limits``, ``html_info_blob`` → ``_html_info_blob``. ``aggregate/extensions/figures.py`` and ``aggregate/extensions/test_suite.py`` updated for the renames.
* ``Aggregate.more``, ``Portfolio.more``, ``Underwriter.more`` renamed to ``.help``. Backing free function in ``utilities.py`` renamed ``more`` → ``agg_help`` (prefixed so it doesn't shadow Python's builtin ``help`` at module / package level).
* ``pprogram`` / ``pprogram_html`` collapsed: dropped the ``split=20`` line-magic and the ``show=True`` side-effect print. Methods preserved — cheat sheets and Underwriter consume them.
* Historical-comment sweep across the ``Aggregate`` class: stale ``# TODO`` / ``# WHOA! WTF`` markers and a commented-out spec-dict block removed.
* Logger calls in ``distributions.py`` converted to lazy ``%s``-style formatting (extends the earlier ``utilities.py`` cleanup).
* Public surface intentionally retained after a docs audit revealed heavy tutorial usage: ``statistics``, ``statistics_df``, ``report_df``, ``report_ser``, ``info``, ``describe``, ``snap``, ``unwrap``, ``picks``, ``recommend_bucket``.

``Underwriter.build()`` return contract uniform:

* ``Underwriter.build()`` now raises ``CannotBuild`` (subclass of ``ValueError``) when a parsed spec produces no top-level object — previously returned a ``ParsedProgram`` with ``object=None`` in the named-mixed-severity edge case. The contract is now uniform: ``build → object`` always (or raises), ``build_many → list[ParsedProgram]`` always. ``CannotBuild`` is exported from the ``aggregate`` package.
* ``Underwriter.discover()`` catches ``CannotBuild`` and skips the row with a ``logger.warning`` (mirrors today's ``NotImplementedError`` handling).

Tooling:

* New ``doc-test-uv.ps1`` script: uv-managed doc build that replaces the clone-to-tmp dance in ``doc-test.ps1``. Builds in place, uses a dedicated ``.doc-venv\`` (set via ``UV_PROJECT_ENVIRONMENT``) so doc builds don't disturb the main development ``.venv``. Supports any Python via ``--python X.Y`` (uv auto-downloads if needed).

1.0.0a1
--------

Underwriter surface rationalization (breaking changes; v1.0 cleanup):

* ``Underwriter.discover(regex, kind='', plot=False, describe=False, return_objects=False, **kwargs)`` replaces ``show`` / ``qshow`` / ``qlist`` (all three removed). Default behavior is the lightweight directory view (matches today's ``qshow``); pass ``plot=True`` or ``describe=True`` to build each match.
* ``Underwriter.build_many(program, ...)`` is the explicit-batch counterpart to ``build``; ``build`` now raises ``ValueError`` when its program produces 0 or >1 top-level outputs (directing the user to ``build_many``).
* ``Underwriter.interpret_file(filename=None, where='')`` replaces ``interpret_test_file`` and absorbs ``run_test_suite``; with no arguments it runs the bundled test suite. Fixes a ``KeyError: 0`` bug from the pandas iterrows path.
* Directory rationalization: ``site_dir``, ``case_dir``, ``template_dir`` properties removed. Single new ``user_dir`` (``~/.aggregate``). ``default_dir`` is now located via ``importlib.resources.files``.
* Base data directory moved from ``~/aggregate`` to ``~/.aggregate`` (dotted convention). No fallback — existing users must ``mv ~/aggregate ~/.aggregate``.
* Constructor magic strings: ``databases='all'`` now expands to ``['default', 'user']``; ``databases='site'`` raises ``ValueError`` directing users to ``'user'``.
* Methods privatized (now leading underscore): ``write`` → ``_build_work``, ``factory`` → ``_factory``, ``safe_lookup`` → ``_safe_lookup``, ``interpret_program`` → ``_interpret_program``. ``write_from_file``, ``dir``, ``test_suite()`` method, ``run_test_suite`` deleted (all unused).
* Portfolio and case_studies internal callers switched from ``uw.write(spec)`` to ``uw.build_many(spec, update=False)`` (equivalent — same ``ParsedProgram`` list, no smart-update).
* ``ParsedProgram`` (dataclass) replaces ``Answer`` for the Underwriter parse-output type. ``Answer`` itself remains in ``utilities.py`` and continues to be used by ``Portfolio``.
* ``Underwriter.__repr__`` clarified: shows ``0 loaded (access .knowledge to read configured database(s))`` when knowledge is pending; no I/O side effect.
* Several bug fixes: ``factory`` ``ValueError`` is now actually raised; the buggy "1 port among many" return path is gone; ``__getitem__`` ``TypeError`` → ``KeyError`` chain preserved with ``from e``; ``read_database`` narrows to ``OSError`` and uses ``logger.exception``.
* Three new constants in ``constants.py``: ``USER_DIR_NAME``, ``PACKAGE_DATA_DIR``, ``TEST_SUITE_FILENAME``.
* Internal cleanup: lazy ``%s``-formatted logger calls throughout; ~130 lines of stale commented-out code removed from ``utilities.py``.

0.30.1
-------
* Confirmed support for Python 3.13 and 3.14

0.30.0
-------
* Added `comonotonic_allocations` to `Portfolio` to implement the method of Denuit, Michel, et al. "Comonotonicity and Pareto optimality, with application to collaborative insurance." Insurance: Mathematics and Economics 120 (2025): 1-16. This uses numba if available. Warning: it can be very slow without numba!

0.29.0
-------
* Portfolio analyze_distortions2 to iron out annoyances with current function but retain it for backwards compatibility.
* Portfolio calibrate_distortions2 for same reasons, args coc and reg_p.
* Spectral tvar_info_df and plot_affine for working with weighted TVaR distortions.
* Changed behavior of Distortion.random_distortion so that input number of knots *includes* mass and mean if present.
* Added random_distortion_ex(n=1, random_state=None) in Distortion class to simulate across types, extending random_distortion which is only a wtdtvar.

0.28.1
-------

* `applymap` to `map` per Pandas update.

0.28.0
-------

* Added ``standard_shape`` to Distortion and added to distortion_df created by Portfolio.calibrate_distortions.
* Updated dependencies and imports for doc build.
* Added `spectral.consistent_distortions` to create consistent family of representative distortions.

0.27.1
-------

* Fixed a bug with recommend unit in a portfolio with all fixed components. 
* Adjusted line styles in twelve plot and clarified use in doc string.
* Corrected ROE calculation of natural allocation premium when g(s) = 1.

0.27.0
~~~~~~~~~~
* Removed control over logging and just use ``logger = logging.getLogger(__name__)`` in all modules. Removed ``log_test`` function and ``LoggerManager`` class. 
* Removed ``numba`` as a requirement - huge library, hardly used. Only occurs in spectral module.
* Replaced build_docs batch file with doc-test which mirrors readthedocs process more closely.

0.26.0
~~~~~~~~~~
* ``extensions`` no longer sets ``pd.float_format`` to Engineering.
* Added ``tweedie.Tweedie`` class to ``extensions`` to compute the Tweedie class distributions for
  all valid :math:`p`. (Dangling jax dependence.)

0.25.0
~~~~~~~~~~~~
* Tweak ``extensions.ft.FourierTools``: added ``invert_simpson`` method using Simpson's rule,
  better for stable distributions. This is the method used by ``scipy.stats``.
* Bumped to 0.25 which should have done in 0.24.2 because it added new functionality
* Tidied docs
* ``knobble_fonts`` uses serif font by default in matplotlib, and sets up
  in color mode by default. 

0.24.2
~~~~~~~~~~

* Added ``Distortion.make_q`` to return the risk adjusted probabilities used
  in pricing. Same logic as ``price_ex``. Makes it easy to compute the natural
  allocation from a distortion.
* Added ``extensions.ft.FourierTools`` class, which performs direct inversion of a (continuous) Fourier transform (characteristic function)
  using FFTs. This is particularly useful for stable distributions, where the Fourier transform is known but the density is not. See examples in Section 5 of the documentation.
* Added ``make_levy_chf`` to ``extensions`` to compute the characteristic function of a Levy stable distribution.

0.24.1
~~~~~~~~~~
* Added script to build the documentation from a local clone of the repository.
* Added ``Aggregate.unwrap`` to adjust aggregates computed with too few buckets
  but enough space. It unwraps the computed aggregate by adjusting the index. This
  reverses the "wagon-wheel" effect, whereby FFTs wrap-around the end of the array.
* Vectorized ``ultilities.estimate_agg_percentile`` for use in ``Aggregate.unwrap``

0.24.0
~~~~~~~~~~
* Added state to Distortions so they can be pickled. Involved separating part of ``Distortion.__init__``
  into a new method, ``Distortion._complete_init``. This is called from ``__init__`` and ``__setstate__``.
  Ensured _complete_init refers to arguments as self.argname, not argname and set self
  variables in class ``__init__`` method.
* Fixed mixture g functions to handle input multidimensional arrays.
* Simplified ``Distortion.__repr__`` and ``Distortion.__str__``.
* Added ``Distortion.id`` to generate a unique ID depending on ``__dict__`` argument elements.
* Corrected ``g_prime`` for minimum distortion.
* Fixed biTVaR distortion to handle p1==1 by including the mass explicitly.
* Added ``Distortion.price_ex`` to combine best of price and price2 methods and improve flexibility. It sorts and summarizes if needed. Optional return formats.
* Added four numba compiled functions to Distortion for fast computation of
  g.g(1-ps.cumsum()) and g.price( kind='ask'). These are tvar_gS, bitvar_gS,
  tvar_ra (for risk adjusted expected value) and bitvar_ra. In each case the
  values are computed without any copies of the original data, making them
  far more memory efficient for very large input arrays. At the extreme,
  bitvar_ra results in a speed up of the order of 2000x in realistic
  situations, even with small (100s) input vectors. The functions are static
  members of Distortion (numba requirement). They are not parallelized
  because of the cumulative computation of S. See the file
  PyWork/Distortion-price-tester.ipynb for tests (TODO: integraete into the
  documentation.)  This addition results in numba being a required package.
* Removed dependency on ``titlecase`` package.
* Removed ``Distortion.calibrate`` method, which was not used and never tested. It lives with ``Portfolio``.

0.23.0
~~~~~~~~~~

* Added ``sample_df`` dataframe to ``Portfolio`` when created from a sample
  to store the sample. Original sample is needed in various applications.
* Added ``swap_density_df(self, new_df, padding=1)`` to ``Portfolio``.
* Fixed errors in Case Studies caused by changes in Pandas.
* Added ability to create Markdown case output, rather than HTML.
* Added beta distortion (generalizes the PH and dual)
* Updated ``np.alltrue`` to ``np.all``; updated ``NoConverge`` in ``scipy.optimize``.
* Added ``Distortion.calibrate`` to calibrate to a pricing target from input ``density_df`` (TODO: needs testing).
* Added `wtdtvar`` to ``Distortion`` to compute the weighted TVaR from p values and weights,
  masses and mean components.
* Added ``minimum`` to ``Distortion`` to create a new ``Distortion`` as the minimum of a list of input Distortions. The list is passed as shape.
* Added ``random_distortion`` to ``Distortions`` to compute a random distortion, useful
  for testing!
* Fixed ``tvar`` distortion to allow p=1 (max)
* Simplified ``Distortion.__repr__`` and ``Distortion.__str__``.
* Added `Distortion.ph``, ``.wang``, ...,  methods for common distortions, with better
  hints for parameters. All are static methods that delegate to the constructor.
* Fixed documentation build errors.

0.22.0
~~~~~~~~~~

* Created version 0.22.0, "convolation" for AAS submission

0.21.4
~~~~~~~~

* Updated requirement using ``pipreqs`` recommendations
* Color graphics in documentation
* Added ``expected_shift_reduce = 16  # Set this to the number of expected shift/reduce conflicts`` to ``parser.py``
  to avoid warnings. The conflicts are resolved in the correct way for the grammar to work.
* Issues: there is a difference between ``dfreq[1]`` and ``1 claim ... fixed``, e.g.,
  when using spliced severities. These should not  occur.


0.21.3
~~~~~~~~

* Risk progression, defaults to linear allocation.
* Added ``g_insurance_statistics`` to ``extensions`` to plot insurance statistics from a distortion ``g``.
* Added ``g_risk_appetite`` to ``extensions`` to plot risk appetite from a distortion ``g`` (value, loss ratio,
  return on capital, VaR and TVaR weights).
* Corrected Wang distortion derivative.
* Vectorized ``Distortion.g_prime`` calculation for proportional hazard
* Added ``tvar_weights`` function to ``spectral`` to compute the TVaR weights of a distortion. (Work in progress)
* Updated dependencies in pyproject.toml file.

0.21.2
~~~~~~~~

* Misc documentation updates.
* Experimental magic functions, allowing, eg. %agg [spec] to create an aggregate object (one-liner).
* 0.21.1 yanked from pypi due to error in pyproject.toml.

0.21.0
~~~~~~~~~

* Moved ``sly`` into the project for better control.  ``sly`` is a Python implementation of lex and yacc parsing tools.
  It is written by Dave Beazley. Per the sly repo on github:

  The SLY project is no longer making package-installable releases. It's fully functional, but if choose to use it,
  you should vendor the code into your application. SLY has zero-dependencies. Although I am semi-retiring the project,
  I will respond to bug reports and still may decide to make future changes to it depending on my mood.
  I'd like to thank everyone who has contributed to it over the years. --Dave

* Experimenting with a line/cell DecL magic interpreter in Jupyter Lab to obviate the
  need for ``build``.

0.20.2
~~~~~~~~~

* risk progression logic adjusted to exclude values with zero probability; graphs
  updated to use step drawstyle.

0.20.1
~~~~~~~

* Bug fix in parser interpretation of arrays with step size
* Added figures for AAS paper to extensions.ft and extensions.figures
* Validation "not unreasonable" flag set to 0
* Added aggregate_white_paper.pdf
* Colors in risk_progression

0.20.0
~~~~~~~

* ``sev_attachment``: changed default to ``None``; in that case gross losses equal
  ground-up losses, with no adjustment. But if layer is 10 xs 0 then losses
  become conditional on X > 0. That results in a different behaviour, e.g.,
  when using ``dsev[0:3]``. Ripple through effect in Aggregate (change default),
  Severity (change default, and change moment calculation; need to track the "attachment"
  of zero and the fact that it came from None, to track Pr attaching)
* dsev: check if any elements are < 0 and set to zero before computing moments
  in dhistogram
* same for dfreq; implemented in ``validate_discrete_distribution`` in distributions module
* Default ``recommend_p=0.99999`` set in constsants module.
* ``interpreter_test_suite`` renamed to ``run_test_suite`` and includes test
  to count and report if there are errors.
* Reason codes for failing validation; Aggregate.qt becomes Aggregte.explain_validation

0.19.0
~~~~~~~

* Fixed reinsurance description formatting
* Improved splice parsing to allow explicit entry of lb and ub; needed to
  model mixtures of mixtures (Albrecher et al. 2017)

0.18.0 (major update)
~~~~~~~~~~~~~~~~~~~~~~~

* Added ability to specify occ reinsurance after a built in agg; this
  allows you to alter a gross aggregate more easily.
* ``Underwriter.safe_lookup`` uses deepcopy rather than copy to avoid
  problems array elements.
* Clean up and improved Parser and grammar

    - atom -> term is much cleaner (removed power, factor; now
      managed with prcedence and assoicativity)
    - EXP and EXPONENT are right
      associative, division is not associative so 1/2/3 gives an error.
    - Still SR conflict from dfreq [ ] [  ] because it could be the
      probabilities clause or the start of a vectorized limit clause
    - Remaining SR conflicts are from NUMBER, which is used in many
      places. This is a problem with the grammar, not the parser.
    - Added more tests to the parser test suite
    - Severity weights clause must come after locations (more natural)
    - Added ability for unconditional dsev.
    - Support for splicing (see below)

* Cleanup of ``Aggregate`` class, concurrent with creating a cheat sheet

    - many documentation updates
    - ``plot_old`` deleted
    - deleted ``delbaen_haezendonck_density``; not used; not doing anything
      that isn't easy by hand. Includes dh_sev_density and dh_agg_density.
    - deleted ``fit`` as alternative name for ``approximate``
    - deleted unused fields

* Cleanup of ``Portfolio`` class, concurrent with creating a cheat sheet

    - deleted ``fit`` as alternative name for ``approximate``
    - deleted ``q_old_0_12_0`` (old quantile), ``q_temp``, ``tvar_old_0_12_0``
    - deleted ``plot_old``, ``last_a``, ``_(inverse)_tail_var(_2)``
    - deleted ``def get_stat(self, line='total', stat='EmpMean'): return self.audit_df.loc[line, stat]``
    - deleted ``resample``, was an alias for sample

* Management of knowledge in ``Underwriter`` changed to support loading
  a database after creation. Databases not loaded until needed - alas
  that includes printing the object. TODO: Consider a change?
* Frequency mfg renamed to freq_pgf to match other Frequency class methods and
  to accuractely describe the function as a probability generating function
  rather than a moment generating function.
* Added ``introspect`` function to Utilities. Used to create a cheat sheet
  for Aggregate.
* Added cheat sheets, completed for Aggregate
* Severity can now be conditional on being in a layer (see splice); managed
  adjustments to underlying frozen rv using decorators. No overhead if not
  used.
* Added "splice" option for Severity (see Albrecher et. al ch XX) and Aggregate,
  new arguments ``sev_lb`` and ``sev_ub``, each lists.
* ``Underwriter.build`` defaults update argument to None, which uses the object default.
* pretty printing: now returns a value, no tacit mode; added _html version to
  run through pygments, that looks good in Jupyter Lab.

0.17.1
~~~~~~~~

* Adjusted pyproject.toml
* pygments lexer tweaks
* Simplified grammar: % and inf now handled as part of resolving NUMBER; still 16 = 5 * 3 + 1 SR conflicts
* Reading databases on demand in Underwriter, resulting in faster object creation
* Creating and testing exsitance of subdirectories in Undewriter on demand using properties
* Creating directories moved into Extensions __init__.py
* lexer and parser as properties for Underwriter object creation
* Default ``recommend_p`` changed from 0.999 to 0.99999.
* ``recommend_bucket`` now uses ``p=max(p, 1-1e-8)`` if severity is unlimited.


0.17.0 (July 2023)
~~~~~~~~~~~~~~~~~~~~

* ``more`` added as a proper method
* Fixed debugfile in parser.py which stops installation if not None (need to
  enure the directory exists)
* Fixed build and MANIFEST to remove build warning
* parser: semicolon no longer mapped to newline; it is now used to provide hints
  notes
* ``recommend_bucket`` uses p=max(p, 1-1e-8) if limit=inf. Default increased from 0.999
  to 0.99999 based on examples; works well for limited severity but not well for unlimited severity.
* Implemented calculation hints in note strings. Format is k=v; pairs; k
  bs, log2, padding, recommend_p, normalize are recognized. If present they are used
  if no arguments are passed explicitly to ``build``.
* Added ``interpreter_test_suite()`` to ``Underwriter`` to run the test suite
* Added ``test_suite_file`` to ``Underwriter`` to return ``Path`` to ``test_suite.agg``` file
* Layers, attachments, and the reinsurance tower can now be ranges, ``[s:f:j]`` syntax

0.16.1 (July 2023)
~~~~~~~~~~~~~~~~~~~~

* IDs can now include dashes: Line-A is a legitimate date
* Include templates and test-cases.agg file in the distribution
* Fixed mixed severity / limit profile interaction. Mixtures now work with
  exposure defined by losses and premium (as opposed to just claim count),
  correctly account for excess layers (which requires re-weighting the
  mixture components). Involves fixing the ground up severity and using it
  to adjust weights first. Then, by layer, figure the severity and convert
  exposure to claim count if necessary. Cases where there is no loss in the
  layer (high layer from low mean / low vol componet) replace by zero. Use
  logging level 20 for more details.
* Added ``more`` function to ``Portfolio``, ``Aggregate`` and ``Underwriter`` classes.
  Given a regex it returns all methods and attributes matching. It tries to call a method
  with no arguments and reports the answer. ``more`` is defined in utilities
  and can be applied to any object.
* Moved work of ``qt`` from utilities into ``Aggregate``` (where it belongs).
  Retained ``qt`` for backwards compatibility.
* Parser: power <- atom ** factor to power <- factor ** factor to allow (1/2)**(3/4)
* ``random` module renamed `random_agg`` to avoid conflict with Python ``random``
* Implemented exact moments for exponential (special case of gamma) because
  MED is a common distribution and computing analytic moments is very time
  consuming for large mixtures.
* Added ZM and ZT examples to test_cases.agg; adjusted Portfolio examples to
  be on one line so they run through interpreter_file tests.

0.16.0 (June 2023)
~~~~~~~~~~~~~~~~~~~~

* Implemented ZM and ZT distributions using decorators!
* Added panjer_ab to Frequency, reports a and b values, p_k = (a + b / k) p_{k-1}. These values can be tested
  by computing implied a and b values from r_k = k p_k / p_{k-1} = ak + b; diff r_k = a and b is an easy
  computation.
* Added freq_dist(log2) option to Freq to return the frequency distribution stand-alone
* Added negbin frequency where freq_a equals the variance multiplier


0.15.0 (June 2023)
~~~~~~~~~~~~~~~~~~~~

* Added pygments lexer for decl (called agg, agregate, dec, or decl)
* Added to the documentation
* using pygments style in ``pprint_ex`` html mode
* removed old setup scripts and files and stack.md

0.14.1 (June 2023)
~~~~~~~~~~~~~~~~~~~~

* Added scripts.py for entry points
* Updated .readthedocs.yaml to build from toml not requirements.txt
* Fixes to documentation
* ``Portfolio.tvar_threshold`` updated to use ``scipy.optimize.bisect``
* Added ``kaplan_meier`` to ``utilities`` to compute product limit estimator survival
  function from censored data. This applies to a loss listing with open (censored)
  and closed claims.
* doc to docs []
* Enhanced ``make_var_tvar`` for cases where all probabilities are equal, using linspace rather
  than cumsum.

0.13.0 (June 4, 2023)
~~~~~~~~~~~~~~~~~~~~~~~

* Updated ``Portfolio.price`` to implement ``allocation='linear'`` and
  allow a dictionary of distortions
* ``ordered='strict'`` default for ``Portfolio.calibrate_distortions``
* Pentagon can return a namedtuple and solve does not return a dataframe (it has no return value)
* Added random.py module to hold random state. Incorporated into

    - Utilities: Iman Conover (ic_noise permuation) and rearrangement algorithms
    - ``Portfolio`` sample
    - ``Aggregate`` sample
    - Spectral ``bagged_distortion``

* ``Portfolio`` added ``n_units`` property
* ``Portfolio`` simplified ``__repr__``
* Added ``block_iman_conover``  to ``utilitiles``. Note tester code in the documentation. Very Nice! 😁😁😁
* New VaR, quantile and TVaR functions: 1000x speedup and more accurate. Builder function in ``utilities``.
* pyproject.toml project specification, updated build process, now creates whl file rather than egg file.

0.12.0 (May 2023)
~~~~~~~~~~~~~~~~~~~

* ``add_exa_sample`` becomes method of ``Portfolio``
* Added ``create_from_sample`` method to ``Portfolio``
* Added ``bodoff`` method to compute layer capital allocation to ``Portfolio``
* Improved validation error reporting
* ``extensions.samples`` module deleted
* Added ``spectral.approx_ccoc`` to create a ct approx to the CCoC distortion
* ``qdp`` moved to ``utilities`` (describe plus some quantiles)
* Added ``Pentagon`` class in ``extensions``
* Added example use of the Pollaczeck-Khinchine formula, reproducing examples from
  the `actuar`` risk vignette to Ch 5 of the documentation.

Earlier versions
~~~~~~~~~~~~~~~~~~

See github commit notes.

Version numbers follow semantic versioning, MAJOR.MINOR.PATCH:

* MAJOR version changes with incompatible API changes.
* MINOR version changes with added functionality in a backwards compatible manner.
* PATCH version changes with backwards compatible bug fixes.


Documentation
-------------

https://aggregate.readthedocs.io/


Where to get it
---------------

https://github.com/mynl/aggregate


Installation
------------

To install into a new ``Python>=3.10`` virtual environment::

    python -m venv path/to/your/venv``
    cd path/to/your/venv

followed by::

    \path\to\env\Scripts\activate

on Windows, or::

    source /path/to/env/bin/activate

on Linux/Unix or MacOS. Finally, install the package::

    pip install aggregate[dev]

All the code examples have been tested in such a virtual environment and the documentation will build.

To build the documentation run


Issues and Todo
-----------------

* Treatment of zero lb is not consistent with attachment equals zero.
* Flag attempts to use fixed frequency with non-integer expected value.
* Flag attempts to use mixing with inconsistent frequency distribution.

Getting started
---------------

To get started, import ``build``. It provides easy access to all functionality.

Here is a model of the sum of three dice rolls. The DataFrame ``describe`` compares exact mean, CV and skewness with the ``aggregate`` computation for the frequency, severity, and aggregate components. Common statistical functions like the cdf and quantile function are built-in. The whole probability distribution is available in ``a.density_df``.

::

  from aggregate import build, qd
  a = build('agg Dice dfreq [3] dsev [1:6]')
  qd(a)

>>>        E[X] Est E[X]    Err E[X]   CV(X) Est CV(X)   Err CV(X) Skew(X) Est Skew(X)
>>>  X
>>>  Freq     3                            0
>>>  Sev    3.5      3.5           0 0.48795   0.48795 -3.3307e-16       0  2.8529e-15
>>>  Agg   10.5     10.5 -3.3307e-16 0.28172   0.28172 -8.6597e-15       0 -1.5813e-13

::

  print(f'\nProbability sum < 12 = {a.cdf(12):.3f}\nMedian = {a.q(0.5):.0f}')

>>>  Probability sum < 12 = 0.741
>>>  Median = 10


``aggregate`` can use any ``scipy.stats`` continuous random variable as a severity, and
supports all common frequency distributions. Here is a compound-Poisson with lognormal
severity, mean 50 and cv 2.

::

  a = build('agg Example 10 claims sev lognorm 50 cv 2 poisson')
  qd(a)

>>>       E[X] Est E[X]   Err E[X]   CV(X) Est CV(X) Err CV(X)  Skew(X) Est Skew(X)
>>> X
>>> Freq    10                     0.31623                      0.31623
>>> Sev     50   49.888 -0.0022464       2    1.9314 -0.034314       14      9.1099
>>> Agg    500   498.27 -0.0034695 0.70711   0.68235 -0.035007   3.5355      2.2421

::

  # cdf and quantiles
  print(f'Pr(X<=500)={a.cdf(500):.3f}\n0.99 quantile={a.q(0.99)}')

>>> Pr(X<=500)=0.611
>>> 0.99 quantile=1727.125

See the documentation for more examples.

Dependencies
------------

See requirements.txt.

Install from source
--------------------
::

    git clone --no-single-branch --depth 50 https://github.com/mynl/aggregate.git .

    python -mvirtualenv ./venv
    # activate the virtual environment (Windows, YRMV)
    venv\Scripts\activate.bat

    # install the package
    pip install aggregate[dev]


License
-------

BSD 3 licence.

Help and contributions
-------------------------

Limited help available. Email me at help@aggregate.capital.

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome. Create a pull request on github and/or
email me.

Social media: https://www.reddit.com/r/AggregateDistribution/.


.. substitutions

.. |downloads| image:: https://img.shields.io/pypi/dm/aggregate.svg
    :target: https://pepy.tech/project/aggregate
    :alt: Downloads

.. |stars| image:: https://img.shields.io/github/stars/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/stargazers
    :alt: Github stars

.. |forks| image:: https://img.shields.io/github/forks/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/network/members
    :alt: Github forks

.. |contributors| image:: https://img.shields.io/github/contributors/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/graphs/contributors
    :alt: Contributors

.. |version| image:: https://img.shields.io/pypi/v/aggregate.svg?label=pypi
    :target: https://pypi.org/project/aggregate
    :alt: Latest version

.. |activity| image:: https://img.shields.io/github/commit-activity/m/mynl/aggregate
   :target: https://github.com/mynl/aggregate
   :alt: Latest Version

.. |py-versions| image:: https://img.shields.io/pypi/pyversions/aggregate.svg
    :alt: Supported Python versions

.. |license| image:: https://img.shields.io/pypi/l/aggregate.svg
    :target: https://github.com/mynl/aggregate/blob/master/LICENSE
    :alt: License

.. |packages| image:: https://repology.org/badge/tiny-repos/python:aggregate.svg
    :target: https://repology.org/metapackage/python:aggregate/versions
    :alt: Binary packages

.. |doc| image:: https://readthedocs.org/projects/aggregate/badge/?version=latest
    :target: https://aggregate.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10557199.svg
    :target: https://zenodo.org/records/10557199
    :alt: Zenodo DOI
