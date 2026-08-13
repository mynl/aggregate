.. _2_x_features:

What Changed in 1.0 vs 0.30.1
================================

This chapter is the user-side, topical companion to ``CHANGELOG.md``: everything
``aggregate`` 1.0 can do that 0.30.1 could not, organized by subject and shown by
worked example. The changelog is chronological and developer-facing; this
document answers *"what's new since the last release I used, and how do I use
it?"* It covers the entire ``1.0.0a*`` series, a1 through a195. Breaking changes
vs 0.30.1 are flagged in each section, and there are many: pre-1.0 was the one
chance to make the names right, and the project took it, generally without
deprecated aliases.

Maintained by ``dev/task-features.md``; the source of record for what shipped is
``CHANGELOG.md``.

.. contents::
   :local:
   :depth: 2

Coverage ledger
---------------

The maintenance contract: every major point in ``CHANGELOG.md`` is accounted for
by a row. "Under the hood" plus an em-dash means an internal change named in the
appendix but not exampled.

.. csv-table:: Coverage ledger, a1 to a195
   :header-rows: 1
   :widths: 46 8 26 20

   "Feature", "Since", "Section", "Example object"
   "``discover`` / ``build_many`` / ``interpret_file``; ``~/.aggregate`` move", "a1", "Underwriter & persistence", "``uw``"
   "``.help()`` (was ``.more``); ``CannotBuild``; surface privatization", "a2", "Underwriter & persistence / Under the hood", "—"
   "``Aggregate.stats_df`` — one canonical moment frame", "a3", "Reporting quartet", "``simple``"
   "PEG regression baseline; stats_df finish", "a4", "Under the hood", "—"
   "approximation & tilting dropped from ``update``", "a5", "Under the hood (see Pedagogy, DecL ``approximate``)", "—"
   "distortion calibration on ``Distortion`` subclasses", "a6", "Under the hood", "—"
   "pricing pipeline: ``pricing_at``, explicit ``p=``/``a=``, ``distortions``, cache", "a7", "Pricing & the pentagon", "``book``"
   "``Portfolio.stats_df``; utilities slim-down; src layout; ``*_fit`` family", "a8", "Reporting quartet / Under the hood", "``book``"
   "``Tweedie`` → ``aggregate.tweedie`` (submodule access)", "a9", "Under the hood", "—"
   "``FourierTools`` → ``aggregate.ft`` (submodule access)", "a10", "Under the hood", "—"
   "``Bounds`` redesigned — one-shot, envelope properties", "a11", "Pricing & allocation bounds", "``book``"
   "``extensions/`` removed; ``aggregate.pedagogy`` created", "a12", "Pedagogy helpers / Under the hood", "—"
   "``Distortion`` natural parameter names", "a13", "Pricing & the pentagon", "—"
   "``aggregate.style`` — matplotlib house style", "a14", "Configuration", "—"
   "``Distortion`` info / describe / stats_df / density_df quartet", "a15", "Reporting quartet", "—"
   "structured DecL parse errors; keyword word boundaries", "a16", "Better parse errors", "—"
   "``silence_warnings`` opt-in (no global mute at import)", "a16", "Configuration", "—"
   "Kusuoka atom rows in ``Distortion`` stats", "a16", "Pricing & the pentagon", "—"
   "``so``/``po`` by number; linear allocation default; ``bounded``; forwards-``S``; ``DefectiveDistributionWarning``; ``EX``/``CV``/``Sk`` headings", "a17", "New DecL elements / Reporting quartet / Pricing", "``reins``, ``defective``"
   "``reins_bucket`` linear/nearest rebucketing", "a18", "Reporting quartet", "``reins``"
   "reins reporting trio (``reins_summary_df`` / ``reins_stats_df`` / ``reins_density_df``), incl. Portfolio", "a19", "Reporting quartet", "``reins``, ``book_w_re``"
   "``occ_bivariate`` — joint (ceded, net) via 2D FFT", "a20", "New DecL elements", "``netceded``"
   "signed severity (``ssev``, signed ``dsev``); output window", "a21", "New DecL elements / Grids & windows", "``signed_d``, ``signed_s``"
   "signed ``Portfolio`` combine; ``shift - dist`` severity", "a22", "New DecL elements", "``signed_s``"
   "``pnl`` premium keyword; signed-aware ``describe`` (SD)", "a23", "New DecL elements / Reporting quartet", "``pnl``"
   "``multivariate`` / ``copula`` / ``netceded``", "a24", "New DecL elements", "``mv_indep``, ``mv_copula``, ``netceded``"
   "``hints{}`` build settings; ``note{}`` pure text", "a25", "New DecL elements / Grids & windows", "``defective``"
   "honest discrete severity — exact moments & quantiles", "a26", "Reporting quartet", "``dice``"
   "distortion flat-number DecL syntax", "a27", "New DecL elements / Pricing", "—"
   "``dsev_bucket`` linear/nearest", "a28", "Grids & windows", "—"
   "tail-thickness classification", "a29", "Tail classification", "``simple``, ``defective``"
   "config.toml; ``show_settings``; ``bucket_sizing_p`` rename; Python ≥3.11", "a30", "Configuration", "—"
   "the pentagon octet; one canonical pricing readout", "a31", "Pricing & the pentagon", "``book``"
   "legible ``Underwriter`` loading; ``to_agg``", "a32", "Underwriter & persistence", "``uw``"
   "``price_stand_alone`` restored", "a33", "Pricing & the pentagon", "``book``"
   "``distortion_df`` / ``calibration_df``; ``gini_p`` rename", "a34", "Pricing & the pentagon", "``book``"
   "empty ``Underwriter()`` default; ``to_agg`` modes & dependency order", "a35", "Underwriter & persistence", "``uw``"
   "``AllocationBounds``", "a36", "Pricing & allocation bounds", "``book``"
   "``PricingBounds``; the Gini lens", "a37", "Pricing & allocation bounds", "``book``"
   "knowledge-freeze regression harness", "a38", "Under the hood", "—"
   "keyword-only ``Underwriter``; signed Lee plot; ``density`` property", "a39", "Underwriter & persistence / Reporting quartet", "``simple``, ``signed_d``"
   "SD for zero-mean signed aggregates", "a40", "Reporting quartet", "``signed_d``"
   "``reinsurance_*`` → ``reins_*`` rename", "a41", "Reporting quartet", "``reins``"
   "fuzz-removal consolidation", "a42", "Under the hood", "—"
   "signed ``Portfolio.summary_df`` spread; ``price_pentagon``", "a43", "Reporting quartet / Pricing & the pentagon", "``book``"
   "module organization, dropped deps, lazy IPython", "a44", "Under the hood", "—"
   "``pnl`` with signed loss severity (fix)", "a45", "New DecL elements", "``pnl``"
   "exponential-tilting pedagogy", "a46", "Pedagogy helpers", "``dice``"
   "``approximate`` DecL keyword", "a47", "New DecL elements / Grids & windows", "``big``"
   "spliced unbounded severities build (fix)", "a48", "New DecL elements", "``mv_indep``"
   "portfolio combine grid: ``best_window``", "a49", "Under the hood (note in Grids & windows)", "—"
   "always-present ``approximate`` info line; self-describing fit note; window-aware plots", "a50", "Reporting quartet / Grids & windows", "``big``"
   "non-zero output window for concentrated aggregates", "a51", "Grids & windows", "``bigex``"
   "``_`` digit separators; ``of`` share synonym", "a52", "New DecL elements", "—"
   "``decl_writer``: ``format_program`` / ``spec_to_decl``; canonical ``pprogram`` / ``to_agg``", "a53", "Programs as text", "``mix_both``"
   "``value_type``; fixed-layout ``info``; ``pnl`` prem/lr meta; ``[labels]`` config", "a54", "Reporting quartet / Configuration", "``pnl``, ``book``"
   "``unit_density`` / ``unit_density_df`` / ``aligned_unit_density_df``", "a55", "Reporting quartet", "``book``"
   "``p_<unit>`` columns & EPD family removed from ``density_df``; signed objective columns", "a56", "Reporting quartet / Under the hood", "``book``"
   "one Choquet engine; ``T.*``/``M.*`` removed; ``allocation_diagnostics``; signed books price", "a57", "Pricing & the pentagon / Under the hood", "``book``"
   "convention-aware aggregate output windowing", "a58", "Grids & windows", "``bigex``"
   "single-big-jump extent floor (heavy / signed severities)", "a59", "Grids & windows", "``defective``"
   "layered thick/thin tail report (``tail_df``)", "a60", "Tail classification", "``simple``"
   "curated ``examples.agg`` library + ``build`` default", "a61", "Underwriter & persistence", "``uw``"
   "narrative tail report (``tail_description`` / ``tail_explanation``)", "a62", "Tail classification", "``defective``"
   "comprehensive scipy severity tail tables (family classifier)", "a63", "Tail classification", "—"
   "tail report wired into grid sizing", "a64", "Grids & windows", "—"
   "the grid choice made legible (``bs_window_df``, ``bs_description``)", "a65", "Grids & windows", "``simple``"
   "portfolio windowed combine 1P; single-big-jump look-through", "a66", "Under the hood", "—"
   "``Aggregate.approximate()`` un-shadowed (bug); one fit core", "a67, a68", "Under the hood", "—"
   "DecL statement separation: blank line or ``;``, no more ``\``", "a69", "New DecL elements", "—"
   "``balanced_window`` + ``Aggregate.center_window``", "a70", "Grids & windows", "—"
   "``.agg`` library rationalization (superseded by a159)", "a71", "Under the hood", "—"
   "bivariate axis sizing: measure, don't guess; measured lower edge; centred axes", "a72, a73, a75, a76", "Bivariate", "``mv_copula``"
   "``round_bucket`` ladder: no more 2.5x jumps", "a74", "Grids & windows", "—"
   "bivariate reporting surface", "a77", "Bivariate", "``mv_copula``"
   "netceded view-pairs (``grossceded``, ``grossnet``)", "a78", "Bivariate", "``netceded``"
   "shuffle-of-Min copula + ``clash`` statement", "a79", "Bivariate", "``clash``"
   "**breaking:** ``multivariate`` → ``bivariate`` (``bv``)", "a80", "New DecL elements / Bivariate", "``mv_indep``"
   "**breaking:** portfolio sub-component ``line`` → ``unit``", "a81", "Reporting quartet", "``book``"
   "consistent naming on the narrative surface; ``validation_explanation``", "a82", "Reporting quartet", "``defective``"
   "config phase 2: numerics floors + stranded sizing knobs (**breaking**)", "a83", "Configuration / Under the hood", "—"
   "**breaking:** ``describe`` → ``summary_df``; ``explain_validation`` removed", "a84", "Reporting quartet", "all"
   "**breaking:** accessor rationalization; bivariate reporting redesign; ``dependency_df``", "a85", "Reporting quartet / Bivariate", "``mv_copula``"
   "discrete bivariate severity (``dbvsev``) + discrete-freq ``bv`` forms", "a86", "Bivariate", "—"
   "infinite-variance aggregates error without an explicit ``bs``", "a87", "Grids & windows", "—"
   "DecL syntax colorer + error labels resynced with the grammar", "a88", "Under the hood", "—"
   "pedagogy figures renamed off the legacy PIR names", "a89", "Pedagogy helpers", "—"
   "``GridDistribution`` value type, adopted by Aggregate/Portfolio/Bounds", "a90, a91", "Under the hood", "—"
   "plotting subsystem: one matplotlib boundary", "a92", "Under the hood", "—"
   "distortion calibration on a single distribution (Aggregate/Portfolio parity)", "a93", "Pricing & the pentagon", "``simple``"
   "``portfolio.py`` split into a facade + three subsystems", "a94", "Under the hood", "—"
   "concern modules filled in; legacy bucket sizers retired; moment helpers", "a95, a96", "Under the hood", "—"
   "``prob_loss_assets``: free choice of capital anchor; ``price_pentagon_ex`` (the ``pla`` alias was retired at a151)", "a97", "Pricing & the pentagon", "``simple``"
   "``.help`` with finer detail control (``lod``, ``output``)", "a98", "Reporting quartet", "—"
   "distortion calibration on signed and payoff supports", "a99", "Pricing & the pentagon", "``pnl``"
   "``format_program`` spread layout (default multiline)", "a100", "Programs as text", "``mix_both``"
   "**breaking:** ``.help`` render targets; ``output`` → ``values``", "a101", "Reporting quartet", "—"
   "DecL ``payoff`` / ``loss`` orientation suffix on ``agg``", "a102", "New DecL elements / P&L", "—"
   "**breaking:** first-class ``PnL``; the in-place ``pnl`` affine removed", "a103", "P&L", "``pnl``"
   "signed additive ``PnL.summary_df`` and ``PnL.plot()``", "a104", "P&L", "``pnl``"
   "``PnL.evaluate()``: the Cherny-Madan breakeven panel", "a105", "P&L", "``pnl``"
   "reinsurance-aware Gross / Ceded / Net ``PnL`` view", "a106", "P&L", "``tower``"
   "``create_frequency()``: materialize the claim-count distribution", "a107", "Reporting quartet", "``simple``"
   "point-mass severity no longer blocks the windowed grid", "a108", "Grids & windows", "—"
   "bare unary minus on a severity (``ssev -lognorm …``)", "a109", "New DecL elements", "—"
   "return-period x-axis for quantile (Lee) plots", "a110", "Reporting quartet", "``simple``"
   "``GridDistribution`` knows its sign; Lee worker consumes it", "a111, a112", "Under the hood", "—"
   "user-facing ``summary_df`` + ``tail_df``; QA frames renamed", "a113", "Reporting quartet", "``simple``"
   "P&L expenses, ceded premium / ceding commission, GCN waterfall", "a114", "P&L", "``book_pnl``"
   "multiple ``pnl`` expense terms (``and``-joined)", "a115", "P&L", "``book_pnl``"
   "property-cat reinstatement premiums (stochastic ceded premium)", "a116, a117", "Reinsurance economics", "``reinst``"
   "variable rating: terms, engine, and DecL for swing/slide/pc/corridor", "a118, a119", "Reinsurance economics", "``swing``"
   "variable rating: ``retro`` DecL surface", "a120", "Reinsurance economics", "—"
   "the bivariate leg kernel (domain-free P&L engine)", "a121", "Under the hood", "—"
   "``PnL`` API: domain-agnostic value object", "a122", "P&L", "``pnl``"
   "``PnL`` exhibits: generic and self-describing", "a123", "P&L", "``book_pnl``"
   "DecL display labels, quoted names, expense grouping", "a124", "New DecL elements", "—"
   "``pnl`` wraps a complete engine; ``inherit premium``", "a125", "P&L", "``inh``"
   "massive (disk-backed) bivariate: out-of-core update, pyramid viz", "a126", "Bivariate", "—"
   "parallel-by-default test suite and the fast local loop", "a127", "Under the hood", "—"
   "DecL labels everywhere: interior label sites, ``LabeledMixin``", "a128", "New DecL elements", "—"
   "generic P&L: signed group ledger, builders, marginal-stack", "a129, a130, a131", "P&L", "``book_pnl``"
   "reins labels pooled into ``label_map``; canonical ``label``", "a132, a133", "New DecL elements", "—"
   "P&L punchups: kappa percentiles, summary card, 3-level tower stats", "a134", "P&L", "``book_pnl``"
   "a pure aggregate ignores economics clauses, with a warning", "a135", "P&L", "—"
   "consolidated ``pnl`` vs the ``xpnl`` walk", "a136", "P&L", "``tower``"
   "``pnl`` engine-name round-trip fix", "a137", "P&L", "—"
   "P&L faces: two-tier occ x agg classifier, composed cells, kappa walks", "a138, a139, a140, a141", "P&L", "``tower``"
   "``help`` everywhere; computed moments in summary; repr trim", "a142", "Reporting quartet", "—"
   "reinsurance premium quoted at 100% placement, scaled down", "a143", "Reinsurance economics", "``reinst``"
   "analysis drill-down classes decommissioned", "a144", "Under the hood", "—"
   "``exact_discrete`` sizing must clear a reachability guard", "a145", "Under the hood", "—"
   "Sparre-Andersen renewal frequency: ``years`` / ``wait`` / ``dwait``", "a146", "Renewal & ruin", "``renewal``"
   "layer terms on the wait clause (``wait y xs a``)", "a147", "Renewal & ruin", "—"
   "``en`` plus ``freq_df`` frequency-surface items", "a148", "Reporting quartet", "``simple``"
   "first-class-class surface sweep; ``HelpMixin``; alias retirement (**breaking**)", "a149, a150, a151", "Under the hood", "—"
   "zero-truncated / zero-modified frequency reparameterized and fixed", "a152", "Renewal & ruin", "—"
   "eventual-ruin probabilities: ``wiener_hopf``, Poisson guard", "a153", "Renewal & ruin", "``renewal``"
   "DecL round-trip surface collapsed into a mixin", "a154", "Under the hood", "—"
   "``pedagogy.ruin_example`` figure polish", "a155, a156", "Pedagogy helpers", "—"
   "``doc{{{}}}`` / ``tags{}`` trailer clauses; distortion gains a trailer", "a157", "Recipe library", "—"
   "the recipe runtime: ``Recipe``, ``build.recipe()``, ``build.recipes``", "a158", "Recipe library", "—"
   "three shipped libraries merge into ``library.agg`` (186 entries)", "a159", "Recipe library", "—"
   "the library tests itself (``tests/test_library_recipes.py``)", "a160", "Recipe library", "—"
   "library tags namespaced (``topic:`` / ``role:`` / ``check:``)", "a161", "Recipe library", "—"
   "test-impact analysis for the edit loop", "a162", "Under the hood", "—"
   "``<<decl>>``: a recipe never retypes its own program", "a163", "Recipe library", "—"
   "one ``Recipe`` class, one registry; ``knowledge`` → ``recipes`` (**breaking**)", "a164", "Recipe library", "—"
   "cookbook pages generated from ``library.agg`` into native Quarto cells", "a165", "Recipe library", "—"
   "trailer gets its own lines; ``format_program`` drops it by default", "a166", "Recipe library", "—"
   "the cookbook generator becomes ``aggregate.cookbook``", "a167", "Recipe library", "—"
   "one canonical ``recipe()`` lookup; documentation sweep", "a168", "Recipe library", "—"
   "``build('3')`` evaluates to a number; ``update=False`` finishes every kind", "a169", "Underwriter & persistence", "—"
   "the first-class-citizen contract, declared and checked", "a170", "Reporting quartet", "—"
   "**breaking:** bivariate ``tail_df`` → ``axis_support_df``; bivariate labels; ``GridDistribution.info``; **breaking:** ``PnL.gd`` → ``PnL.result``", "a171", "Bivariate / P&L / Reporting quartet", "``mv_copula``, ``pnl``"
   "``validation_df`` on bivariate and distortion; narrative pairs completed (**breaking**, narrowly); ``reins_explanation``", "a172", "Reporting quartet", "``simple``, ``reins``"
   "``discover`` rejects a mistyped filter and an unusable build option", "a173", "Underwriter & persistence", "``uw``"
   "``Frequency.__repr__``; ``Tweedie._repr_html_``; ``Copula`` gains ``HelpMixin``", "a174", "Reporting quartet / Under the hood", "``simple``"
   "DecL colorizer re-derived from the grammar (the red boxes are gone)", "a175", "Programs as text", "—"
   "``interpret_file`` reads statements, not physical lines", "a176", "Underwriter & persistence", "—"
   "``note{}`` / ``tags{}`` / ``hints{}`` bodies are inert: ``#``, ``//``, ``[``, ``]`` are free", "a177", "Programs as text", "—"
   "``library.agg`` rewritten in the canonical spread layout", "a178", "Programs as text / Recipe library", "—"
   "TVaR endpoint ``RuntimeWarning`` silenced at three sites", "a179", "Under the hood", "—"
   "``help()`` regex defaults to ``'.*'``, so a bare ``help()`` lists everything", "a180", "Reporting quartet", "``simple``"
   "``oep()``, the occurrence exceeding probability curve; exact ``sev.ppf`` / ``sev.isf``", "a181", "Reporting quartet", "``simple``"
   "``make_ceder_netter`` multi-gap knot fix", "a182", "Under the hood", "—"
   "DecL ``peel top-down`` / ``bottom-up``; **breaking:** walk step default labels", "a183", "P&L / New DecL elements", "``tower``"
   "tier subtotal rows on a peeled walk", "a184", "P&L", "``tower``"
   "**breaking:** ``Scaled`` retired; ``ratio_df`` / ``legs_df`` / ``Leg.kind``", "a185", "P&L", "``pnl``"
   "``EX_*`` → ``E_*``; the ratio gate reads the denominator", "a186", "P&L", "``retro``"
   "**breaking:** ``evaluate`` solves ``rho_g(margin) = 0``; ``Aggregate`` / ``Portfolio`` faces; ``E_consideration`` retired", "a187", "P&L / Pricing & the pentagon", "``simple``, ``tower``"
   "``evaluate`` reads a bought layer from the seller's side; ``role`` column", "a188", "P&L", "``tower``"
   "**breaking:** ``PnL`` levels ``View`` / ``Line`` → ``Side`` / ``Label``; ``Direct`` and ``Net``", "a189", "P&L", "``tower``"
   "**breaking:** ``total impact`` is not evaluated; a netted row reads ``role == 'net'``", "a190", "P&L", "``tower``"
   "**breaking:** the first step and the direct margin take their declared labels", "a191", "P&L", "``tower``"
   "``sharpen()`` audits the grid ``update`` chose; **breaking:** ``focus`` → ``center_window``", "a192", "Grids & windows", "``simple``"
   "``validation_score``; each probe row line-searches the bucket axis", "a193", "Grids & windows", "``simple``"
   "sharpen takes the best affordable grid; ``min_gain`` retired; ``log2_cap`` 20", "a194", "Grids & windows", "``dice``"
   "a grid that loses mass is disqualified; ``deficit`` / ``defective`` / ``warns``", "a195", "Grids & windows", "``simple``"

The cast of examples
--------------------

A small set of objects built once and heavily leveraged: each is the minimal
program that exercises a distinct piece of DecL surface, and the feature
sections borrow these rather than minting throwaways. (The canonical definition
brief lives in ``dev/task-features.md`` section 2.4; the live definitions are
here.)

.. ipython:: python

    from aggregate import build, qd

    # ── The cast: built once, reused throughout ────────────────────────────────
    # Clustered by theme. Each is the minimal program that exercises a distinct
    # surface; feature sections borrow these rather than minting throwaways.

    # 1. Core aggregate ─────────────────────────────────────────────────────────
    # 1.a simple — the workhorse: mixed-gamma (neg-binomial) frequency, layered lognormal
    simple = build('agg Simple 100 claims 1000 xs 0 sev lognorm 100 cv 1.0 mixed gamma 0.25')

    # 1.b defective — essentially defective tail (shifted Pareto; mass pushed to the limit)
    defective = build('agg Defective 10 claims sev 100 * pareto 1.3 - 100 poisson '
                      'hints{bs=0.25; log2=16;}')

    # 2. Discrete, exact moments ─────────────────────────────────────────────────
    dice = build('agg Dice dfreq [3] dsev [1:6]')

    # 3. Signed & P&L ────────────────────────────────────────────────────────────
    # 3.a signed_d — signed discrete (dsev with a negative atom auto-signs the agg)
    signed_d = build('agg SignedD dfreq [5] dsev [-1 2]')

    # 3.b signed_s — signed continuous via ssev; a PER-CLAIM shift (premium booked per claim)
    signed_s = build('agg SignedS 10 claim ssev 100 - lognorm 80 cv 0.025 poisson')

    # 3.c pnl — book-level premium minus loss. Contrast 3.b: SAME MEAN, DIFFERENT VARIANCE —
    #     the pnl premium is one deterministic shift; the ssev +100 is multiplied by the
    #     (random) claim count, so 3.b carries extra variance from 100*N.
    #     Since a125 a pnl wraps a COMPLETE named engine: `less agg <name> <program>`
    #     (the a23 in-place `premium - <severity>` affine was removed at a103).
    pnl = build('pnl PNL 1000 prem less agg PNL_e 10 claim sev lognorm 80 cv 0.025 poisson')

    # 4. Bivariate (was `multivariate` — renamed a80) ────────────────────────────
    # 4.a mv_indep — INDEPENDENT (shared frequency only; positive baseline corr from
    #     the shared mixing / common shock). Splicing an unbounded base works since a48.
    mv_indep = build('''
    bivariate Cat 25 claims
        agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 0.65 splice [0 250]
        agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 0.95 splice [0 300]
        mixed gamma .2
    ''')

    # 4.b mv_copula — COPULA-coupled (Gumbel, upper-tail dependence)
    mv_copula = build('''
    bivariate CatC 25 claims
        agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 0.65 splice [0 250]
        agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 0.95 splice [0 300]
        copula gumbel 0.4
        mixed gamma .2
    ''')

    # 5. Reinsurance ─────────────────────────────────────────────────────────────
    # 5.a reins — occurrence reinsurance; describe now reports gross/net.
    #     Drives reins_summary_df / reins_stats_df. (Note: the occ clause comes BEFORE freq.)
    reins = build('agg Re 10 claims 1000 xs 0 sev lognorm 100 cv 2 '
                  'occurrence net of 50% po 300 xs 200 and 100% po 500 xs 500 '
                  'poisson')

    # 5.b netceded — the SAME program as 5.a with the `netceded` prefix: the occurrence
    #     (Ceded, Net) law as a bivariate. Literally 5.a with `netceded ` prepended.
    netceded = build('netceded agg Re 10 claims 1000 xs 0 sev lognorm 100 cv 2 '
                     'occurrence net of 50% po 300 xs 200 and 100% po 500 xs 500 '
                     'poisson')

    # 6. Mixtures ────────────────────────────────────────────────────────────────
    # 6.a mix_sev — weighted MIXTURE of severities (single exposure)
    mix_sev = build('agg MixSev 100 claims 2000 xs 0 '
                    'sev lognorm [50 100 200] cv [1 1.5 2] wts [.5 .3 .2] poisson')

    # 6.b mix_exp — several EXPOSURE bands, one severity
    mix_exp = build('agg MixExp [100 200 50] claims 2000 xs 0 sev lognorm 100 cv 2 poisson')

    # 6.c mix_both — JOINT mixed severity + exposure (the gnarly one): paired exposure/limit
    #     bands AND a two-component severity mixture of shifted pareto / lognormal.
    mix_both = build('agg MixBoth [100 200 50] claims [1000 2000 5000] xs 0 '
                     'sev [200 150] * [pareto lognorm] [2.1 0.8] + [-200 0] wts [.2 .8] poisson')

    # 7. Portfolios ──────────────────────────────────────────────────────────────
    # 7.a book — a Portfolio for combine / pentagon pricing / allocation & pricing bounds
    book = build('''
    port Book
        agg A 100 claims 2000 xs 0 sev lognorm 100 cv 1.0 mixed gamma 0.5
        agg B  50 claims 4000 xs 0 sev lognorm 200 cv 2.0 mixed gamma 0.4
    ''')

    # 7.b book_w_re — the same book with occurrence reinsurance on each unit
    book_w_re = build('''
    port Book
        agg A 100 claims 2000 xs 0 sev lognorm 100 cv 1.0 occurrence net of 50% po 1000 xs 1000 mixed gamma 0.5
        agg B  50 claims 4000 xs 0 sev lognorm 200 cv 2.0 occurrence net of 50% po 2000 xs 1000 mixed gamma 0.4
    ''')

    # 8. Approximate (a47) vs exact on a non-zero window (a51) ───────────────────
    # 8.a big — very-high-frequency book via the `approximate` shortcut (sgamma fit).
    #     Note `approximate` comes AFTER the frequency clause. The info line and the
    #     self-describing fit note are a50.
    big = build('agg Big 1e6 claims dsev [1 3] poisson approximate sgamma')

    # 8.b bigex — the EXACT convolution for comparison: since a51 it resolves at
    #     bs=1 on a two-sided output window far from 0 (mean 2e6, sd ≈ 2,200)
    #     instead of wasting the whole grid on [0, 2e6).
    bigex = build('agg BigEx 1e6 claims dsev [1 3] poisson')

    # 9. Reinsurance economics (a116-a120) ───────────────────────────────────────
    # 9.a reinst — an occurrence layer with reinstatements: the ceded premium is
    #     STOCHASTIC, because reinstatement premium is driven by the recovery.
    reinst = build('agg Reinst 10000 premium at 75% lr 10000 xs 0 sev 10.808 * lognorm 1.75 '
                   'occurrence net of 24500 xs 500 deposit 1000 reinstatements [1] '
                   'mixed gamma 0.25')

    # 9.b swing — variable rating: a swing-rated aggregate layer. Same idea, other
    #     direction: the ceded premium is a function of the ceded loss.
    swing = build('agg Swing 10000 premium at 75% lr 10000 xs 0 sev 10.808 * lognorm 1.75 '
                  'mixed gamma 0.25 '
                  'aggregate net of 5000 xs 10000 swing basic 0 lcm 1.1 min 300 max 1000')

    # 10. Renewal frequency (a146) ───────────────────────────────────────────────
    # 10.a renewal — a Sparre-Andersen renewal count: claims arrive after iid waits,
    #      not on a Poisson clock. `1 year` + `wait <dist>` replaces the freq clause.
    renewal = build('agg Renewal 1 year sev gamma 2 wait 0.25 * uniform')

.. list-table:: The cast and what each member drives
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Drives
   * - ``simple``
     - reporting quartet, tail class, ``density`` property, config effects
   * - ``defective``
     - defective / mass-at-limit tail, deficit warnings, ``hints{}``
   * - ``dice``
     - discrete/exact moments, ``dsev_bucket``
   * - ``signed_d``
     - signed ``dsev``, signed-aware ``describe`` (SD vs CV)
   * - ``signed_s`` / ``pnl``
     - ``ssev``, ``shift - dist``, per-claim vs book-level premium (``pnl``)
   * - ``mv_indep`` / ``mv_copula``
     - ``bivariate``, ``copula`` (and the no-copula baseline)
   * - ``reins`` / ``netceded``
     - reins gross/net reporting; ``netceded`` occurrence bivariate
   * - ``mix_sev`` / ``mix_exp`` / ``mix_both``
     - mixed severity, mixed exposure, and the joint
   * - ``book`` / ``book_w_re``
     - combine, pentagon pricing, ``price_pentagon`` / ``price_stand_alone``,
       bounds; reins in a portfolio
   * - ``big`` / ``bigex``
     - ``approximate sgamma`` vs the exact convolution on a non-zero output
       window (a51)
   * - ``reinst`` / ``swing``
     - stochastic ceded premium: reinstatements and variable rating
   * - ``renewal``
     - Sparre-Andersen renewal frequency via the ``wait`` clause

New DecL elements
-----------------

.. _feat signed severity:

Signed severity: ``ssev`` and negative ``dsev`` atoms (a21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A profit is a negative loss, so an aggregate can live on a signed grid. The
severity family is now: ``sev``, continuous, clamps its sub-zero tail at 0
(unchanged from 0.30.1); ``dsev``, discrete, never clamps, and a negative atom
auto-signs the aggregate; ``ssev``, new, the signed continuous sibling of
``sev``. Signedness is recorded at parse time, so the analytic moments, and the
automatic two-sided output window, are right before any FFT runs.

.. ipython:: python

    qd(signed_d.summary_df)   # SD trio instead of CV — see Reporting quartet

.. ipython:: python

    # the signed Lee (quantile) plot starts cleanly at the true minimum (a39 fix)
    @savefig features_signed_d.png scale=20
    signed_d.plot()

``shift - dist``: premium minus loss per claim (a22)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant minus a distribution parses as the natural P&L reading. The cast's
``signed_s`` is ``ssev 100 - lognorm 80 cv 0.025``, premium 100 minus a lognormal
loss, *per claim*. It is exactly ``-1 * X + 100`` (reflect, then shift) and needs
``ssev`` to keep the signed support.

.. ipython:: python

    qd(signed_s.summary_df)

``pnl``: premium minus loss, once for the book (a23; reshaped a103, a125)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``pnl`` keyword collects the premium **once for the book** and subtracts the
loss aggregate: :math:`\mathrm{PnL} = \mathrm{premium} - A`. Contrast with
``signed_s``, which books premium **per claim**. They have the **same mean and
very different variance**, and the direction is the interesting part:

.. ipython:: python

    import pandas as pd
    print(f'per-claim  (signed_s): mean {signed_s.est_m:8.2f}   sd {signed_s.est_sd:8.2f}')
    print(f'book-level (pnl)     : mean {pnl.est_m:8.2f}   sd {pnl.est_sd:8.2f}')

Booking premium per claim is a **natural hedge on count risk**: premium and loss
both scale with N, so the count variance largely cancels. Write
:math:`P = 100N - S` for the per-claim form and :math:`P = 1000 - S` for the book
form, with :math:`S` the compound sum. Then

- per claim: ``Var = E[N]·Var(100−X) + Var(N)·E[100−X]² = 10(4) + 10(400) = 4,040``, sd ≈ 63.6
- book: ``Var = E[N]·Var(X) + Var(N)·E[X]² = 10(4) + 10(6,400) = 64,040``, sd ≈ 253.1

The book-level P&L carries the full ``Var(N)·E[X]²`` term because its premium is
fixed. Fixing the premium is what *creates* the count exposure, which is the
whole reason an aggregate cover is worth buying.

Since a103 a ``pnl`` is no longer an ``Aggregate`` with a shift applied in place;
it is a **first-class** :class:`PnL` object wrapping a complete named engine
(``less agg <name> <program>``, a125). Its ``summary_df`` is a signed ledger
rather than a moment table:

.. ipython:: python

    qd(pnl.summary_df)

The full P&L surface, including expenses, ceded premium and the reinsurance walk,
is :ref:`P&L: profit and loss as a first-class object <feat pnl>`.

Since a45, the *loss* severity of a ``pnl`` may itself be signed (a ``dsev`` with
a negative atom, or an ``ssev``); previously the affine path silently wrapped the
negative atoms.

``bivariate`` and ``copula``: joint laws of two perils (a24, renamed a80)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking, a80:** the keyword is **``bivariate``** (short form ``bv``). It
shipped at a24 as ``multivariate``, which promised a generality the 2D FFT engine
does not have. The object is :class:`BivariateAggregate`, and the whole surface
is covered in :ref:`Bivariate aggregates <feat bivariate>`.

One event can drive two correlated perils, and you want the joint law of the two
aggregates, not just the marginals. ``bivariate`` couples two component
severities by a copula and accumulates them with a shared outer frequency through
a 2D FFT. The component ``dfreq [0 1] [p0 p1]`` is the per-event trigger
probability. The ``copula`` clause is optional: omitted means independence, where
the only dependence is the shared (here gamma-mixed) count:

.. ipython:: python

    qd(mv_indep.summary_df)
    print(f'corr, no copula (common shock only): {mv_indep.corr:.4f}')
    print(f'corr, Gumbel tau=0.4 on top:         {mv_copula.corr:.4f}')

The realized output correlation is reported alongside the copula parameter
because they differ: compounding attenuates per-claim dependence, and shared
mixing adds common-shock dependence on top, so the independence copula still
shows positive correlation. Copulas come from the ``aggregate.copula.Copula``
factory (``normal``, ``gumbel``, ``clayton``, ``fgm``, ``independent``), each
taking its natural dependence parameter.

.. ipython:: python

    @savefig features_mv_copula.png scale=20
    mv_copula.plot()   # joint per-claim severity (left), joint aggregate (right)

The splice on the lognormal base (``splice [0 250]``) caps an unbounded family;
this crashed window sizing until the a48 fix.

.. _feat netceded:

``netceded``: the joint (ceded, net) law of a reinsured aggregate (a20, a24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefix any occurrence-reinsured ``agg`` program with ``netceded`` and the result
is the *joint* per-occurrence (Ceded, Net) law as a bivariate: the same 2D FFT
engine with a comonotone per-claim severity (per claim, :math:`(c(X), n(X))` lies
on the line :math:`c + n = X`). The margins reproduce the univariate ceded/net
aggregates and ``E[Ceded] + E[Net]`` is the gross mean; the joint law carries what
the margins cannot, the reinsurer-versus-cedent dependence:

.. ipython:: python

    qd(netceded.summary_df)
    print(f'corr(Ceded, Net) = {netceded.corr:.4f}')

``Aggregate.occ_bivariate()`` returns the same object from an already-built
reinsured aggregate (for example ``reins.occ_bivariate()``).

``hints{}``: build settings out of ``note{}`` (a25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``note{...}`` used to double as a ``key=value`` side-channel for build settings,
so any ``=`` in note prose crashed the build. Notes are now pure annotation; a
dedicated ``hints{key=value; ...}`` clause carries settings (``log2``, ``bs``,
``bucket_sizing_p``, and the rest). Explicit ``build(...)`` keyword arguments
always override hints. The cast's ``defective`` uses ``hints{bs=0.25; log2=16;}``.

Distortions in DecL: flat number list (a27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking vs 0.30.1.** The distortion form is uniformly
``distortion NAME kind n1 n2 ...``, the kind's parameters in declaration order,
no brackets. Each :class:`Distortion` subclass declares its own DecL parameter
order, so the parser no longer hard-codes kinds. Note ``ccoc`` takes the *return*
``r``, not the discount.

.. ipython:: python

    d1 = build('distortion D1 ph 0.7')
    d2 = build('distortion D2 bitvar 0.9 0.99 0.5')    # p0 p1 w1
    print(d1)
    print(d2)

Reinsurance share syntax: the number sets the meaning (a17, a52)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``so`` (share-of), ``po`` (part-of), and, since a52, plain ``of`` are synonyms;
the **number** decides. A literal percentage (``50% po 300 xs 200``) is the share
directly; a bare amount (``150 po 300 xs 200``) is an absolute amount and the
share is ``amount / limit``. In 0.30.1 ``50% po`` silently divided the percentage
by the limit. Layer validation (a18) also hard-errors on out-of-order or
overlapping layers; express a gap with a zero-share layer.

Number literals: ``_`` digit separators (a52)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``agg BIG 10_000_000 claims …`` parses exactly as Python reads ``10_000_000``.

``approximate``: method-of-moments aggregates (a47)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very-high-frequency books the exact convolution is overkill. The
``approximate sgamma | slognorm | exact`` directive (placed **after** the
frequency clause) replaces the frequency by severity convolution with a single
shifted gamma or shifted lognormal fitted to the aggregate's first three moments,
matching mean, CV and skew to about 7 significant figures. There is no special
compute path: the object is rewritten as a fixed-1-claim aggregate of the fitted
severity, so everything downstream (validation, ``pnl``, Portfolio combine) works
unchanged. Incompatible with occurrence reinsurance (rejected with a clear
error); aggregate reinsurance rides along.

.. ipython:: python

    print(big.info)    # the permanent `approximate` line + self-describing fit note (a50)

See Grids & windows for the accuracy comparison against ``bigex``, the exact
convolution.

Statement separation: a blank line or ``;`` (a69)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking.** Multi-statement DecL used to need a trailing backslash to continue
a line. Statements are now separated by a **blank line or a** ``;``, and the
backslash is gone. A program can therefore be indented naturally across lines,
which is what every multi-line example in this document relies on.

Bare unary minus on a severity (a109)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ssev -lognorm 10 cv 0.5`` negates a severity directly, without the ``-1 *``
scale factor:

.. ipython:: python

    print(build('agg U 5 claims ssev -lognorm 10 cv 0.5 poisson').est_m)

``payoff`` / ``loss`` orientation (a102)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``agg`` can be declared a payoff rather than a loss by suffixing the program.
Orientation drives the sign convention used throughout pricing, so it is declared
rather than inferred:

.. ipython:: python

    print(build('agg P 10 claims sev lognorm 100 cv 1 poisson payoff').value_type)

Labels: ``as "…"`` (a124, a128, a132, a133)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any named object, and since a128 any *interior* site (a layer, a severity
component, an exposure band), can carry a human display label with
``as "Some Label"``. Labels are presentation only: no computed value changes.
They live in one place per object (``label``) with classless interior sites
pooled into ``label_map``, reached through the ``labels`` namespace.

FYI premium on the ``claims`` / ``loss`` heads (a266)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sizing head can carry the booked premium alongside it: ``5 claims 20000
premium`` sizes the count from the claims clause and books 20,000 as
informational premium; ``500 loss 650 premium`` sizes from the loss pick and
implies the 76.9% loss ratio. The head wins, exactly as if the suffix were
absent, and the premium never touches the law; the loss ratio back-fills from
the realized expected loss, the same reconciliation the sizing head ``premium
at lr`` has always run, in the other direction. This is the natural spelling
for a book rated one way and modeled another: quote sheets carry claim counts
or loss picks plus a premium, and previously the premium had nowhere to go
short of converting the head to ``premium at lr`` by hand.

No ``and`` joins the clause. In DecL ``and`` combines terms into one item
(reinsurance layers, expense terms) and nothing is combined here; since comma
is whitespace, ``5 claims, 20000 premium`` reads naturally. Both amounts must
be scalars: a vector would broadcast into the component structure and change
the law or misreport per-component premium, which an informational clause must
never do. The suffix takes its own ``as`` label (the ``premium`` site of
``label_map``), and ``pnl … inherit premium`` reads the booked premium from an
engine that carries one. The Sparre-Andersen renewal head ``T years at r rate``
established the informational-premium concept; this extends it to the ordinary
heads.

.. ipython:: python

    fyi = build('agg Fyi 5 claims 20000 premium sev lognorm 1000 cv 2 poisson')
    qd(fyi.stats_df.loc[('meta', ['prem', 'lr']), 'mixed'])

Better parse errors (a16)
-------------------------

DecL parse failures are now structured reports with line and column, a
source-line echo, a caret marker, and "did you mean" suggestions, instead of
Lark's internal token namespace dump. The one-line summary is ``str(e)``; the
full rendered block is ``e.report.render()`` (also auto-printed by ``build``):

.. ipython:: python
    :okwarning:

    try:
        build('agg Typo 10 cliams sev lognorm 50 cv 1 poisson')
    except ValueError as e:
        print(e)

Keyword terminals also require word boundaries, so a typo like ``aggx Re …``
reports ``Unexpected 'aggx'. Did you mean: agg?`` at column 1 instead of a
misleading downstream error.

Programs as text: ``decl_writer`` (a53)
---------------------------------------

``aggregate.decl_writer`` is the structural inverse of the parser: it renders a
parsed spec back to canonical DecL instead of pretty-printing by regex.
``format_program`` accepts a program string (or a spec) and returns canonical
text, and also ``'html'``, ``'ansi'``, ``'latex'`` via ``fmt=``:

.. ipython:: python

    from aggregate.decl_writer import format_program
    print(format_program(mix_both.program))

``Aggregate.pprogram`` and ``Portfolio.pprogram`` now render this canonical form
(``self.program`` still holds the raw input), and ``Underwriter.to_agg`` exports
canonical DecL, so saved ``.agg`` files round-trip cleanly.

**Breaking vs 0.30.1:** ``utilities.decl_pprint`` is removed; use
``format_program``.

A trailer body is prose (a177)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``note{}``, ``tags{}`` and ``hints{}`` bodies used to be read as DecL before they
were read as text, which claimed four characters a human writing a note has every
right to use. ``#`` and ``//`` are the comment openers, so the statement was cut
at the note and the parse failed several clauses upstream; ``[`` and ``]`` drive
the vector collapse, which padded them with spaces, cumulatively, so regenerating
a file from its own specs grew another space every pass. All four are now free:

.. ipython:: python

    for body in ('E[loss]=85, margin 15', 'layer is 5# of limit', 'net // ceded'):
        a_ = build(f'agg Note 5 claims sev lognorm 100 cv 1 poisson note{{{body}}}',
                   update=False)
        print(repr(a_.note))

A single-line body is lifted behind a placeholder before the comment and bracket
steps and restored verbatim afterward. What a note still cannot hold is a ``}``
(the terminal ends at the first one) and a line break. That is what
``doc{{{ }}}`` is for.

The colorizer is derived from the grammar (a175)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Pygments lexer behind ``pprogram_html`` was a hand-written mirror of
``decl.lark`` that had fallen behind the language, and the visible symptom was
**red boxes in Jupyter**: ``AggLexer`` had no rule for the quoted display label
that labels-everywhere introduced, so both quotes of ``as "My Book"`` lexed as
``Token.Error``, which the ``friendly`` style draws with a red border. It is now
re-derived from the grammar, and the shipped corpora tokenize clean:

.. ipython:: python

    from pygments.token import Error
    from aggregate.decl_pygments import AggLexer
    from pathlib import Path
    import aggregate

    lib = Path(aggregate.__file__).parent / 'agg' / 'library.agg'
    n_err = sum(1 for tok, _ in AggLexer().get_tokens(lib.read_text(encoding='utf-8'))
                if tok is Error)
    print(f'library.agg Token.Error count: {n_err}   (was 170)')

Thirteen defects were fixed, and one line caused a whole class of them: every
keyword rule used ``\b`` as its word boundary, which is not DecL's. The grammar
makes ``.``, ``_``, ``:``, ``~`` and ``-`` name characters, so ``\b`` peeled
keywords off the front of longer names and ``loss-ratio`` came out as three
tokens. There is deliberately no catch-all rule, because ``Token.Error`` is the
drift signal and swallowing it would make the corpus test above vacuous.

The shipped library reads like its own documentation (a178)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``agg/library.agg`` is written in the canonical ``spread`` layout: one clause per
line at a two-space indent, ``;`` closing each statement. Median statement length
was 153 characters and 40 statements ran past 200; nothing but a note body is now
over 100. It is the same layout ``format_program`` emits and every generated
cookbook page already showed, so source and docs finally agree:

.. ipython:: python

    print(format_program('agg Ex [100 200] premium at [0.9 0.85] lr '
                         '[250 500] xs 0 sev lognorm 120 cv 12 mixed gamma 0.2'))

Indenting the clauses exposed two real defects that one long line had hidden.
Four PIR case studies carried ``hints{bs=…; log2=…}`` after their last unit,
where a ``port`` trailer does not reach, so the hints bound to the unit and were
ignored and all four built at the wrong resolution. And ``ssev <c> - <dist>``
renders as a general affine form that parses two ways, so those entries are held
back rather than grow the library's ambiguous set. Sixteen of 185 entries are
hand-written for reasons like this, tracked as ``[Unparser-Reference-Gaps]``.

A single-claim severity also reads as English again: ``1 claim``, not
``1 claims``.

The reporting quartet: ``info`` / ``describe`` / ``stats_df`` / ``density_df``
------------------------------------------------------------------------------

One stats surface everywhere (a3, a8, a15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking vs 0.30.1.** The overlapping moment frames (``report_df``,
``report_ser``, ``statistics``, ``statistics_df``, ``audit_df`` on both classes)
are gone. The stats surface is exactly three things on :class:`Aggregate`,
:class:`Portfolio` *and* :class:`Distortion` alike: ``info`` (text summary),
``describe`` (the daily-driver moment audit), ``stats_df`` (the canonical
``(component, measure)`` frame with ``ex1``/``ex2``/``ex3`` raw and
``mean``/``cv``/``skew`` derived moments, theoretical and empirical side by
side). :class:`Distortion` additionally has a ``density_df`` (the ``g`` function
and friends on a grid). The validation philosophy, every build audited against
exact moments in the declare, build, validate, trust cycle, is described in the
package paper :cite:p:`Mildenhall2024`.

.. ipython:: python

    qd(simple.summary_df)

.. ipython:: python

    qd(simple.stats_df)

Fixed-layout ``info`` and ``value_type`` (a54)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``info`` now has a fixed row layout, every row always present, same order, for
every instance, ``n/a`` where unavailable, shared across all three classes. New
rows include ``value_type`` (the loss/payoff pricing-sign convention),
``x_min``/``x_max`` (the realized window), always-present premium, expected loss,
loss ratio and ``P(loss)``, and ``bounded``/``id`` footers.

.. ipython:: python

    print(simple.info)

A :class:`Portfolio` derives ``value_type`` from its units and **rejects a mixed
loss/payoff book at construction** (no coherent sign convention):

.. ipython:: python
    :okwarning:

    try:
        build('''port Mixed
            agg A 10 claims sev lognorm 100 cv 1 poisson
            pnl B 1000 premium - 10 claim sev lognorm 80 cv 0.5 poisson''')
    except ValueError as e:
        print(e)

Exact discrete moments (a26)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discrete severities (``dsev``, ``fixed``) used to be faked through a continuous
``rv_histogram`` with sliver-width atoms, producing ``ppf(0.5) = 149.9999999992``
artifacts and quietly degrading the moments of large-valued discrete books. They
are now backed by an honest discrete RV: exact step ``cdf``/``sf``, exact
``ppf``, and **all moments computed as exact finite sums**, unlimited, limited
and layered. A fair die has mean exactly 3.5:

.. ipython:: python

    qd(dice.summary_df)   # Err columns exactly 0
    print('sev ppf(0.5) =', dice.sevs[0].fz.ppf(0.5))

Signed-aware ``describe``: SD instead of CV (a23, a40, a43)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CV = sd/mean`` is meaningless as the mean crosses 0, so for any signed object
the moment table shows an SD trio instead of CV (visible under
:ref:`signed severity <feat signed severity>` above).
The zero-mean case reports the correct SD; in 0.30.1-era logic it reconstructed
``SD = mean × CV = 0 × NaN = NaN`` (fixed a40, deriving variance from the second
moment). ``Portfolio.summary_df`` makes the CV-versus-SD choice once,
portfolio-wide, so a mixed signed/unsigned book renders one consistent frame
(a43).

.. ipython:: python

    zero_mean = build('agg ZM dfreq [3] dsev [-1 1]')   # mean 0, sd sqrt(3)
    qd(zero_mean.summary_df)

The ``density`` property and ``sev_density_df`` (a39, a21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``density`` is ``density_df.query('p_total > 0')``, the live support with the
grid's leading and trailing zero buckets dropped, which is usually what you want
to look at. The severity moved to its own frame ``sev_density_df`` (its own grid
``xs_sev``), since a windowed or signed aggregate no longer shares a grid with
its severity.

.. ipython:: python

    qd(signed_d.density)

Reinsurance reporting (a18, a19, a41)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reinsurance reporting was rebuilt end to end. Three objects on :class:`Aggregate`
*and* (new) :class:`Portfolio`:

- ``reins_summary_df``, the daily driver: per-stage gross/ceded/net with the same
  eight columns as ``describe``, where ``Change`` reads as validation error on
  the gross row and as the cession impact on the ceded and net rows;
- ``reins_stats_df``, the per-layer layering summary (occurrence layers
  conditional on attaching; meta rows for share, limit and attachment,
  attach/detach probabilities, loss-on-line);
- ``reins_density_df``, per-bucket gross/ceded/net densities with consistent
  columns.

``describe`` itself becomes an economic view under reinsurance, with Gross, Net
and Change columns:

.. ipython:: python

    qd(reins.summary_df)

.. ipython:: python

    qd(reins.reins_summary_df)

The portfolio version convolves the per-unit gross/ceded/net marginals:

.. ipython:: python

    qd(book_w_re.reins_summary_df)

Net and ceded distributions are rebucketed onto the model grid mean-preservingly
by default (``reins_bucket='linear'``, a18; ``'nearest'`` restores the historical
snap). **Breaking vs 0.30.1:** the legacy ``reinsurance_df`` /
``reinsurance_audit_df`` / ``reinsurance_report_df`` family is gone, and the
spelled-out method names are renamed: ``reinsurance_kinds`` to ``reins_kinds``,
``reinsurance_description`` to ``reins_description``, ``reinsurance_occ_plot`` to
``reins_occ_plot`` (a41; ``reins`` is the canonical short form).

Unit densities on a Portfolio (a55, a56)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking vs 0.30.1.** ``Portfolio.density_df`` no longer carries ``p_<unit>``
columns, and the EPD columns are gone (stand-alone EPD is the one-liner
``(e - lev)/e``). Unit pmfs live on the owning :class:`Aggregate` objects, on
their native grids, read through explicit accessors:

.. ipython:: python

    qd(book.unit_density_df().groupby(level='unit').head(2))   # long form, (unit, loss) index

``unit_density(unit)`` returns one unit's pmf; ``aligned_unit_density_df(grid=)``
is the display adapter that scatters unit pmfs onto a common grid (and warns when
a windowed book's view would be clipped).

Validation: noise-aware, with honest deficit warnings (a17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validation moments are tested against a definite noise floor (1e-12), so a fair
die no longer "fails skew" on floating-point dust, and ``describe`` snaps dust to
zero. A genuinely defective pmf, FFT mass off the top of the grid, sets
``Validation.DEFECTIVE`` and raises ``DefectiveDistributionWarning`` at
construction instead of surfacing silently in pricing. Since a218 the warning
fires above ``deficit_materiality`` (1e-4, the level at which the deficit can
move a price) and once per session, so a sweep that rebuilds the same shape
sixty-four times reports one fault rather than sixty-four copies of it; the
per-object verdict is always in ``valid`` and ``validation_explanation``, which
is where to look for the rest. The cast's ``defective`` (Pareto 1.3, infinite
variance) is built to show this:

.. ipython:: python

    print(defective.validation_explanation)

The frequency distribution, materialized (a107, a148)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``create_frequency()`` builds the claim-count distribution as an object in its
own right, so the frequency can be inspected and plotted like any other
aggregate. ``Frequency.freq_df`` is the diagnostic table: the count pmf ``p``
beside the mean-matched Poisson ``po_p``.

.. ipython:: python

    qd(simple.frequency.freq_df.head(8))

Reading the two columns side by side is the fastest way to see dispersion. A
``mixed gamma ν`` frequency has variance over mean exactly :math:`1 + \nu^2 n`,
so ``p`` spreads visibly wider than ``po_p``. A ``wait expon`` renewal count
reproduces ``po_p`` to the kernel noise floor, which is the Poisson process
recovered as a special case. The diagnostic is capped at mean 1000 or less: it is
a small-count eyeball tool.

Return-period quantile plots (a110, a235)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``return_period=True`` puts the Lee plot on a return-period x-axis, which is how
catastrophe results are usually read. It was ``quantile_x='return'`` until a235,
when the aggregate chart moved onto the chart IR and the return period became a
reading the document declares rather than an argument threaded down to a drawing
worker:

.. ipython:: python

    @savefig features_return_period.png scale=20
    simple.plot(return_period=True)

Reflected (survival) readings (a269)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reflect=True`` reads a probability axis as its complement, ``1 - v``. On the
Lee panel that is the exceeding probability, so the curve drawn is the survival
function, and with ``invert=True`` it is ``S(x)`` the usual way round. The
reflected axis declares a log reading that the non-exceeding probability does
not, which is the reading a log axis was invented for:

.. ipython:: python

    @savefig features_reflect.png scale=20
    simple.plot(invert=True, reflect=True, log=True)

The two probability readings compose. On a loss, whose return period is
``1 / (1 - p)``, ``reflect=True, return_period=True`` draws the curve
``return_period=True`` draws by itself. On a signed ``PnL``, whose return period
is the shortfall's ``1 / p``, the pair reads the upside tail instead. On a
distortion or a pricing envelope, whose axes are both probabilities, reflecting
both gives the dual.

``help`` on every first-class class (a98, a101, a142, a150, a180)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every first-class class answers ``help(regex)``, filtered by pattern, with
``lod`` controlling detail. **Breaking at a101:** the ``output`` argument is
``values``. At a150 nine copy-pasted implementations collapsed into a single
``HelpMixin``, so the surface is identical everywhere it appears.

.. ipython:: python

    simple.help('tvar|var$')

At a180 ``regex`` gained the default ``'.*'``, so a bare ``help()`` lists the
whole surface instead of raising ``TypeError``. Exactly two filters decide what
appears, ``regex`` and the leading-underscore skip governed by ``private``.
Inherited names are not filtered, so ``Severity.help()`` reports the
``scipy.stats.rv_continuous`` methods alongside its own.

The first-class-citizen contract (a170, a172)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"First-class citizen" stopped being a phrase in a plan and became a declaration
the build checks. Two criteria put a class on the list, and both must hold: it
can be **created in DecL**, and it flows through to the ``aggregate_api`` SPA,
which calls exactly these members on whatever it is handed.

.. ipython:: python

    from aggregate import constants as C
    print('first class :', C.FIRST_CLASS_CLASSES)
    print('near        :', C.NEAR_FIRST_CLASS)
    print('required    :', C.FCC_REQUIRED)
    print('exceptions  :', C.FCC_CONTRACT_EXCEPTIONS, C.FCC_UNPAIRED_NARRATIVES)

:class:`Bounds`, :class:`Frequency` and ``GridDistribution`` are out because they
are reached from an object, never declared. :class:`Severity` is DecL-creatable
but is a look-through onto a frozen scipy random variable rather than a compute
result, so it is exempt from the DataFrame quartet and listed separately rather
than silently omitted. Everything outside ``FCC_REQUIRED`` is optional, and
optional does not mean unconstrained: wherever one half of a ``*_description``
(short) / ``*_explanation`` (long) pair is present, the other must be too.

The declaration is checked in two places off one source,
``dev/regen_features.py`` and ``tests/test_fcc_surface.py``, so the contract
cannot break without a red test. It found four holes nobody was tracking, and
a172 closed all of them: both exception lists print empty above, which is what
finishes the item. The contract is satisfied, not excused.

``validation_df``, and the narrative pairs (a172)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``validation_df`` is a **check table**, ``Est | Ref | Err | Gate | Pass``,
carrying only what can fail. The point is that a reader who wants the verdict
should not have to know which of twenty numbers in ``summary_df`` is
load-bearing. It reached :class:`BivariateAggregate` and :class:`Distortion` at
a172:

.. ipython:: python

    qd(mv_copula.validation_df)

The distortion's four rows need **two tolerance regimes**, and that is worth
knowing. :math:`g(0) = 0` and :math:`g(1) = 1` are evaluated directly, must hold
exactly, and are gated at the config noise floor.
:math:`E[D_g] + E[D_{g^{-1}}] = 1` and :math:`g(g^{-1}(0.5)) = 0.5` come off a
trapezoidal integral on 101 points, so they carry a genuine :math:`O(h^2)`
discretization term; gating those at the noise floor would fail every kind for
the crime of being a numerical integral. They are gated at :math:`10h^2`, which
tightens automatically if the grid is refined.

.. ipython:: python

    from aggregate import Distortion
    qd(Distortion('ph', 0.5).validation_df)

**Breaking, narrowly:** ``validation_explanation`` was never a long form,
returning ``'not unreasonable'`` or ``'fails agg cv'``. That terse text now lives
on ``validation_description``, the name that describes it, and every terse
consumer reads it, so what they print is unchanged. Code that read
``validation_explanation`` for a short phrase now gets a paragraph:

.. ipython:: python

    print(simple.validation_description)
    print()
    print(simple.validation_explanation)

``reins_explanation`` is new on :class:`Aggregate` and :class:`Portfolio`.
``reins_description`` says what the program *declares*; this adds what the
cession *does*, off ``reins_summary_df``, reported per **stage**, because the
aggregate cover attaches to the occurrence net rather than the gross and one
"ceded" number across both stages double counts. It answers ``'No reinsurance.'``
on a clean book, so no caller has to check ``reins_kinds`` first:

.. ipython:: python

    print(reins.reins_explanation)
    print(simple.reins_explanation)

Every object says what it is (a171, a174)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`Frequency` had no ``__repr__`` at all and printed
``<aggregate._frequency.FrequencyPoisson object at 0x...>``. It now reports its
family, its shape parameters and ``E[N]``, omitting the mean when no owning
:class:`Aggregate` has stamped one rather than inventing a number:

.. ipython:: python

    print(repr(simple.frequency))
    print(repr(build('agg P 10 claims sev lognorm 100 cv 1 poisson').frequency))

``GridDistribution`` gained an ``info`` at a171. It is not contractual, a grid
distribution being unreachable from DecL, but the grid a quantile came off is
exactly what you want to see when a number looks wrong, and the rows are the grid
rather than the risk. Realized total mass is the row that earns its place: ``1``
for a complete distribution, less when the holder handed over a clipped or
conditional slice.

.. ipython:: python

    print(pnl.result.info)

:class:`PnL` and :class:`Distortion` also gained ``_repr_html_`` at a171, so every
first-class class now renders the same two pieces in Jupyter, an intro paragraph
and the headline frame. The P&L intro says out loud that its percentile columns
are marginal and do not foot, with ``stats_df`` named as the footing sheet.

Exact severity quantiles, and the OEP curve (a181)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Aggregate.sev`` already carried the exact continuous ``cdf``, ``sf`` and
``pdf`` of the weighted severity mixture, the documented look through past the
discretization, but it had no inverse, so there was no way to get an exact
severity quantile out of an :class:`Aggregate` at all. ``q_sev`` reads the
bucketed ``sev_density_df`` and can only return a lattice point:

.. ipython:: python

    t1 = build('agg T1 2 claims sev lognorm 1000 cv 1.31 poisson')
    print(f'sev.ppf(0.99) = {t1.sev.ppf(0.99):.6f}    exact')
    print(f'q_sev(0.99)   = {t1.q_sev(0.99):.6f}    on the lattice, bs = {t1.bs}')

A single component delegates to the :class:`Severity` ``rv_continuous`` methods,
which respect limits, attachments and splices. A mixture has no closed-form
inverse and is solved by bracketing the root with the component quantiles, a
bracket the monotonicity of the components guarantees.

That inverse is what ``oep`` needed. ``oep(agg, p)`` answers *"there is a
probability p that one or more occurrences in a year exceed x, what is x?"*,
inverting :math:`p = 1 - e^{-\lambda \Pr(L > x)}`:

.. ipython:: python

    from aggregate import oep
    qd(oep(t1, [0.5, 0.1, 0.01]))

Both return periods are reported: the occurrence one,
:math:`1/(\lambda S_{sev})`, which can be shorter than a year, and the annual
one, :math:`1/\mathrm{oep}`, which cannot. Three points are worth knowing. The
loss is computed as ``isf(-log1p(-p) / lam)`` rather than
``ppf(1 + log1p(-p) / lam)``, because the second form cancels toward 1 as ``lam``
grows. There is a ceiling, :math:`p < 1 - e^{-\lambda}`, since a year with no
occurrence has no largest loss, and it raises rather than returning ``nan``. And
Poisson frequency is required, with zero-modified Poisson refused: the thinning
argument is what makes :math:`1 - e^{-\lambda S}` hold.

Grids, buckets and windows
--------------------------

The sizing decision is inspectable (a21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``update`` runs up to several sizing methods (exact-discrete lattice, bounded
small-count, the legacy three-moment rule, windowed), records them in
``bs_window_df``, then picks. ``log2`` is a cap; pinning ``bs`` keeps full manual
control. The frame became public at a65, and
:ref:`the grid choice is legible <feat grid legible>` covers the narrative forms
built on top of it.

.. ipython:: python

    qd(bigex.bs_window_df.iloc[:, :6])

**Breaking vs 0.30.1:** the ``recommend_p`` knob is renamed ``bucket_sizing_p``
everywhere (a30), as a ``build``/``update`` keyword and as a ``hints{}`` key. No
alias.

Mean-preserving discrete bucketing: ``dsev_bucket`` (a28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Off-grid discrete atoms (an empirical sample with a non-integer ``bs``) are now
split across their two bracketing buckets so the discretized mean is exact
(``dsev_bucket='linear'``, the default; ``'nearest'`` is the historical snap,
with up to ``bs/2`` bias per atom). On-grid atoms (a die at ``bs=1``) are
unchanged.

.. ipython:: python

    off = build('agg Off dfreq [1] dsev [0.3 1.7 2.4] [.5 .3 .2]', bs=0.5)
    print(f'discretized mean {off.est_m:.6f} == exact {0.3*.5 + 1.7*.3 + 2.4*.2:.6f}')

Non-zero output windows (a51)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A concentrated aggregate, CV small enough that its whole mass sits far above 0,
now computes on a two-sided window around the mass instead of the wasteful
``[0, x_max]`` grid. The cast's ``bigex`` (mean 2e6, sd about 2,200) resolves at
``bs = 1`` on a window near the mean; on a 0-based grid it would get ``bs`` in
the hundreds:

.. ipython:: python

    qd(pd.concat([big.summary_df, bigex.summary_df], keys=['approximate sgamma', 'exact, windowed']))

Behavioral note (intended): ``q``, ``F`` and plots of a windowed aggregate live
on ``[x_min, x_max]``, not from 0. What is given up is the sub-tolerance region
below the window. Pass ``x_min=0`` to ``update`` to force the legacy 0-based
grid. Occurrence reinsurance suppresses windowing (its severity must share the
output grid); plots are window-aware (a50), so the left edge anchors at the
realized support:

.. ipython:: python

    @savefig features_bigex.png scale=20
    bigex.plot()

The portfolio combine grid was also fixed (a49, ``best_window``): the shared
``(bs, log2)`` is now resolution plus span, the finest unit bucket against the
no-wrap floor, so adding units no longer *coarsens* the grid, and all-integer
discrete books land on the lattice (``bs=1``) instead of a spurious fine
continuous ``bs``.

.. _feat grid legible:

The grid choice is legible (a58, a64, a65, a70)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sizing decision used to be a number you either trusted or overrode. It is now
a reported decision. ``bs_window_df`` shows each candidate method, whether it
applies, which was selected, and the resulting window:

.. ipython:: python

    qd(simple.bs_window_df)

``bs_description`` and ``bs_explanation`` are the short and long narrative forms
of the same decision. The :ref:`tail report <feat tail class>` feeds this: since a64 the
thick/thin classification is an *input* to sizing rather than a report written
afterwards.

Guards: infinite variance and unreachable grids (a74, a87, a145)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three sizing corrections worth knowing. ``round_bucket`` lost its 2.5x jumps
(a74), so the bucket ladder is smooth. An **infinite-variance** aggregate now
errors unless you pass an explicit ``bs`` (a87): there is no honest automatic
answer, so it asks rather than guessing. And ``exact_discrete`` sizing lost its
unconditional top priority (a145), and must clear a reachability guard before it
is chosen.

``sharpen``: auditing the grid, not just trusting it (a192 to a195)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``update`` **chooses** a grid from the analytic moments before any FFT runs.
``sharpen`` **audits** that choice afterwards: it re-updates the object on
neighbouring ``(bs, log2)`` cells, scores each against the analytic moments, and
moves when the win is worth having. It is on :class:`Aggregate` and
:class:`Portfolio`. The mechanism, the selection rule, the controls and the frame
layout are documented on the grid page,
:ref:`Auditing the choice afterwards: sharpen <bs sharpen>`; what follows is what
changed and how it looks in use.

The score came first, and it is worth having on its own. ``validation_score`` is
six terms, severity and aggregate mean, CV and skewness, read from the canonical
``stats_df['error']`` and each divided by **its own** validation tolerance. The
units are therefore tolerance: ``score <= 1`` means the object passes validation,
and ``1`` is exactly the pass boundary. Where ``valid`` says whether a line was
crossed, this says by how far, which is what makes it comparable across grids.

.. ipython:: python

    sharp = build('agg Sharp 100 claims sev lognorm 100 cv 2 poisson')
    print(f'auto-sized score: {sharp.validation_score:.4f}')

Now force it onto a grid far too small to hold the book, and probe:

.. ipython:: python

    sharp.update(log2=14, bs=1/8)
    sharp.sharpen()
    print(sharp.sharpen_description)

``sharpen_df`` is one row per cell, indexed by the offsets ``(d_bs, d_log2)``, so
the picture is one unstack away:

.. ipython:: python

    qd(sharp.sharpen_df.score.unstack('d_log2'))

Down a column is constant grid size, the resolution question; along an
anti-diagonal is constant extent. The rows are ragged because each one is a
**line search** (a193) out from the current bucket, and the selection rule (a194)
is *never pay more than you already are*. ``min_gain`` is retired and
``good_enough`` (default ``0.5``) is the only judgment knob left.

A discrete severity is a special case, and a satisfying one. When every atom is a
whole number of buckets there is nothing to search on the bucket axis, so the
bucket is **pinned** and only ``log2`` is probed, which is what a failing
discrete object actually needs, its problem being extent.

.. ipython:: python

    d_ = build('agg D6 dfreq [3] dsev [1:6]')
    d_.sharpen(good_enough=0)
    print(d_.sharpen_description)

Finally (a195), **a grid that loses mass off its top end is disqualified, however
well it scores**. A moment score cannot see lost mass and never will: on a cat
tower the winning cell sat 13 times *inside* mean tolerance while shedding
2.1e-08 of its mass out at the far tail. It is a **gate, not a penalty**, and the
threshold is ``VALIDATION_NOISE``, deliberately tighter than the materiality
floor at which ``DefectiveDistributionWarning`` fires: choosing among candidate
grids, losing nothing at all is free to insist on, while interrupting the user
is not. Three frame columns carry it:

.. ipython:: python

    qd(sharp.sharpen_df[['bs', 'log2', 'score', 'deficit', 'defective', 'selected']].head(6))

Two notes on reach. ``update(..., sharpen=True)`` opts a single update into
auto-sharpening; it is off by default and **not** turned on by ``build``, because
a probe costs eight or more extra updates. And there is no
``BivariateAggregate.sharpen`` or ``PnL.sharpen``, for reasons the grid page
gives.

**Breaking at a192:** ``Aggregate.focus`` is now ``Aggregate.center_window``. The
old name was wrong for what it does, a no-recompute re-slicer returning the
central window of ``density_df`` holding :math:`1 - p` of the mass, and it
blocked the good name. No deprecation shim.

.. _feat tail class:

Tail-thickness classification (a29)
-----------------------------------

Every :class:`Aggregate` and :class:`Portfolio` reports an ordered tail class on
a five-rung scale, ``bounded < super-exponential < exponential < subexponential <
power-law``, by deterministic family lookup, with the aggregate rung the heavier
of frequency and severity (the single-big-jump principle for
subexponential-or-heavier severities). ``bounded`` is now a derived view of the
classifier, with the certify setter ``obj.bounded = True`` as escape hatch.

.. ipython:: python

    print(simple.tail_description)

.. ipython:: python

    print(defective.tail_explanation)   # power-law, with the tail index and variance flags
    print(f'dice bounded: {dice.bounded};  defective bounded: {defective.bounded}')

``Portfolio.tail_class`` reports the worst-of across units and names the driving
units.

The layered tail report: ``tail_df`` (a60, a62, a63)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tail_class`` gives the one-word answer. ``tail_df`` gives the layered evidence
behind it, and ``tail_description`` / ``tail_explanation`` the narrative:

.. ipython:: python

    qd(simple.tail_df.head(8))
    print(simple.tail_description)

a63 added comprehensive tail tables for the scipy severity families, so the
classifier knows the analytic tail of each family rather than inferring it
numerically. That is what makes the classification trustworthy enough to drive
grid sizing (:ref:`the grid choice is legible <feat grid legible>`).

Pricing and the pentagon
------------------------

One canonical readout: the octet (a31)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every pricing method emits the same eight accounting quantities in the same
order: amounts ``L`` (loss), ``M`` (margin), ``P = L + M`` (premium), ``Q``
(capital), ``a = P + Q`` (assets), and ratios ``LR = L/P``, ``PQ = P/Q``,
``ROE = M/Q``. Stats are always columns, one row per priced entity. In 0.30.1
each method built these independently and disagreed on naming and order. The
framework is that of *Pricing Insurance Risk* :cite:p:`Mildenhall2022a` and its
capital-modeling companion :cite:p:`Major2026`.

Distortions: natural parameters (a13) and the quartet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking vs 0.30.1.** :class:`Distortion` constructors take kind-specific names
instead of the generic ``(name, shape, r0, df, …)`` slots: ``Distortion('ph',
a=0.7)``, ``Distortion('wang', lam=0.2)``, ``Distortion('dual', b=2)``,
``Distortion('tvar', p=0.99)``, ``Distortion('ccoc', r=0.1)`` (keyword-only),
``Distortion('bitvar', p0=, p1=, w1=)``. Each exposes its parameter as a
read/write property. The named kinds include the proportional-hazards and Wang
transforms :cite:p:`Wang1995,Wang1996`. Distortions carry the same ``info`` /
``describe`` / ``stats_df`` / ``density_df`` quartet as everything else (a15),
including the Kusuoka spectral-measure atoms (a16):

.. ipython:: python

    from aggregate import Distortion
    d = Distortion('bitvar', p0=0.9, p1=0.99, w1=0.5)
    qd(d.summary_df)

Calibration: ``distortion_df`` and ``calibration_df`` (a7, a34)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``calibrate_distortions(coc=, p=)`` (one point; the old list-based batch API is
gone, a7) calibrates the canonical five (``ccoc``, ``ph``, ``wang``, ``dual``,
``tvar``) and splits the result into a per-distortion receipt and the shared
target:

.. ipython:: python

    book.calibrate_distortions(coc=0.10, p=0.995)
    qd(book.distortion_df)     # param, error, gini_p, area per distortion

.. ipython:: python

    qd(book.calibration_df)    # the target once: coc, p, F(a) + the octet

**Breaking vs 0.30.1:** ``Distortion.standard_shape`` is renamed ``gini_p``
(:math:`= 2\int g - 1`); ``Portfolio.dists`` is renamed ``distortions``; pricing
methods take explicit ``p=`` *or* ``a=`` (the implicit "p greater than 1 means an
asset level" convention is gone).

Pricing a book (a7, a17, a57)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``apply_distortion`` is a cached build of the risk-adjusted frame;
``pricing_at(distortion, p=|a=)`` reads the octet at a capital level; ``price``
wraps both and returns a ``PricingResult``:

.. ipython:: python

    wang = book.distortions['wang']
    qd(book.pricing_at(wang, p=0.995))

Two allocation surfaces, one builder (a57): ``allocation='lifted'``
(risk-adjusted beta) or ``'linear'`` (objective alpha; the collapsed-tail default
since a17 for robustness with mass distortions on unbounded books, where lifted
*refuses* a mass distortion on an unbounded support rather than emitting an
unstable frame). Since a265 the whole pentagon surface honors
``allocation_method``: ``apply_distortion``, ``pricing_at``, ``pentagon_at``,
``analyze_distortion`` and ``analyze_distortions`` each take an ``allocation``
keyword defaulting to ``None``, which reads the member, so lifted is reached by
asking for it. The diagnostic layer curves live in ``allocation_diagnostics``:

.. ipython:: python

    diag = book.allocation_diagnostics(wang)
    qd(diag.filter(regex='layer_.*_total').iloc[2000:2005])

**Breaking vs 0.30.1 (a57):** the ``T.*`` and ``M.*`` column families are removed
from the augmented frame. Pricing readers use ``L = exa``, ``P = exag``,
``M = P − L``, with per-line capital computed on demand by the layer-ROE
construction; the ``efficient`` flag is gone (one frame shape); the cache is
keyed on ``(name, view, role, S_calculation, allocation)`` so variant frames
coexist. Signed (P&L) books now price (total premium via the Choquet dot
product; equal-priority per-line columns are NaN on a signed grid).

Stand-alone vs diversified (a33)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``price_stand_alone(dist, p)`` prices every unit backed by its own VaR(p) capital
and contrasts the ``sum`` with the diversified ``total``, the classic
diversification-benefit exhibit:

.. ipython:: python

    qd(book.price_stand_alone(wang, 0.995))

The ``Pentagon`` and ``price_pentagon`` (a31, a43)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pentagon_at(distortion, p=)`` returns a ``Pentagon``, an eight-vector with
named attributes that completes any soluble partial input. ``price_pentagon``
needs **no distortion**: fix the capital level (``p=`` or ``a=``) plus exactly
one target (``ROE=``, ``P=``, ``LR=``, ``M=``, ``Q=``, or ``PQ=``) and it
completes the octet by pure accounting:

.. ipython:: python

    qd(book.price_pentagon(p=0.995, ROE=0.10))   # VaR capital + cost of capital

.. ipython:: python

    qd(simple.price_pentagon(a=12500, LR=0.70))  # works on an Aggregate too

Free choice of capital anchor: ``prob_loss_assets`` (a97)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Capital can be anchored three ways, and they are the same relationship read from
different ends: pick the exceedance probability ``p``, the loss level ``L``, or
the assets ``a``, and get the other two. Pass exactly one:

.. ipython:: python

    print(simple.prob_loss_assets(p=0.99))
    print(simple.prob_loss_assets(a=20000))

``price_pentagon_ex`` extends the pentagon readout over a range of anchors. (The
short alias ``pla`` shipped at a97 and was retired at a151 under the
one-name-per-concept rule.)

Calibration on signed and payoff supports (a93, a99)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distortion calibration works on a single distribution, with :class:`Aggregate`
and :class:`Portfolio` reaching it identically (a93), and since a99 it handles
**signed** and **payoff** supports. That is what lets a P&L, which lives on both
sides of zero, be priced with the same machinery as a loss.

Pricing and allocation bounds
-----------------------------

Three classes in ``aggregate.bounds``, sharing one exact piecewise-linear hull
engine on the FFT grid (no resampling: TVaR curves are affine in
:math:`1/(1-p)` within an atom, so hulls built at CDF breakpoints are exact and
:math:`O(n)`).

``Bounds``: distortion envelopes for the total (a11, redesigned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class implementing the similar-risks bounds of :cite:t:`Mildenhall2022` is
now one-shot: construct with the object and premium, and read ``p_star``,
``min_envelope`` (a coherent :class:`Distortion`), ``max_envelope`` and
``cloud_df`` as properties:

.. ipython:: python

    import numpy as np
    from aggregate.bounds import Bounds
    P = float(book.calibration_df['P'].iloc[0])
    b = Bounds(book, P, a=float(book.q(0.995)))
    print(f'p* = {b.p_star:.4f}; min envelope: {b.min_envelope}')

``AllocationBounds``: natural-allocation ranges (a36)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given that the total is priced to P, the range of natural-allocation premiums to
each unit over *all* consistent distortions :math:`\{g : \rho_g(X) = P\}`, the
similar-risks methodology :cite:p:`Mildenhall2022` applied to allocations. The
extremes are biTVaRs; the hulls are P-independent (build once, slice any
premium):

.. ipython:: python

    ab = book.allocation_bounds(p=0.995)
    qd(ab.bounds(P))

``ab.bitvars(P)`` gives the achieving distortions, ``ab.distortion(P, unit,
bound)`` reconstructs one, and ``ab.check(P)`` reprices from first principles.

``PricingBounds`` and the Gini lens (a37)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same machinery, different question: if a reference risk X is priced to P by
*some* distortion, what is the price range of another risk Y? Any
:class:`Aggregate`, :class:`Portfolio` or pmf works as either axis:

.. ipython:: python

    pb = book.pricing_bounds(simple, p=0.995)
    qd(pb.bounds(P))

With the uniform reference (``uniform_source()``) as X, the constraint collapses
to a mean-Kusuoka-level condition, the **Gini lens**, reading the envelope gap of
Y's own TVaR curve.

Configuration
-------------

``config.toml`` (a30)
~~~~~~~~~~~~~~~~~~~~~

The hard-coded "secret bits", default ``log2``, default database, bucketing
schemes, sizing percentile, validation tolerances, are now read from an optional,
hand-editable ``~/.aggregate/config.toml``. Values layer built-in defaults, then
the config file, then ``AGGREGATE_*`` environment variables, then explicit
keyword arguments, so a call argument always wins. The surface:
``build.write_default_config()`` (fully-commented template),
``build.reload_settings()``, ``aggregate.get_settings()`` (resolved snapshot),
and the source-annotated readout:

.. ipython:: python

    build.show_settings()

Escape hatches: ``AGGREGATE_CONFIG=/path`` relocates, ``AGGREGATE_CONFIG=none``
ignores the file for reproducible runs. Unknown keys warn loudly.
**Breaking vs 0.30.1:** minimum Python is 3.11; the tunables that lived in
``aggregate.constants`` (for example ``VALIDATION_NOISE``) moved behind
``get_settings()``.

The ``[labels]`` section (a54) renames the printed ``value_type`` words
(``loss``/``payoff`` to, say, ``claim``/``P&L``) without moving any object's
role: code branches on the underlying boolean, never the label text.

Plot styling: ``aggregate.style`` (a14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library no longer touches your matplotlib settings at import (the 0.30.1
``knobble_fonts`` machinery is gone). Opt in to the house style globally with
``aggregate.style.use()``, or scoped:

.. ipython:: python

    import aggregate.style
    @savefig features_style.png scale=20
    with aggregate.style.context():
        simple.plot()

Warnings are yours (a16)
~~~~~~~~~~~~~~~~~~~~~~~~

The package no longer runs ``warnings.simplefilter('ignore')`` at import. Muting
is an explicit opt-in: ``from aggregate import silence_warnings;
silence_warnings()``, now with optional ``category=`` and ``message=`` scoping.

Underwriter and persistence
---------------------------

Construction: keyword-only, empty by default (a35, a39)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A bare ``Underwriter()`` starts **empty**, with no silent loading of the bundled
``test_suite`` (the module-level ``build`` still loads its configured databases).
The constructor is keyword-only, so ``Underwriter('test_suite')``, which used to
silently *name* the underwriter after the database you meant to load, is now a
``TypeError``. Use ``Underwriter(databases='test_suite')``.

.. ipython:: python

    from aggregate import Underwriter
    uw = Underwriter(name='demo')
    print(uw)

Legible loading (a32)
~~~~~~~~~~~~~~~~~~~~~

One explicit pipeline: ``databases=`` is the *request*; the public ``databases``
attribute reports the file ``Path`` objects actually loaded; ``load(request)``,
``reload()``, ``resolve_databases(request)`` and ``available_databases()`` are
the verbs. Globs work (``databases=['cat_*']``), resolved across the current
directory, then ``~/.aggregate``, then bundled. Every knowledge entry carries a
``source`` tag (file path or ``'session'``). **Breaking vs 0.30.1:**
``read_database(s)`` is renamed ``load``; the base data directory moved from
``~/aggregate`` to ``~/.aggregate`` (a1, no fallback); ``show``, ``qshow`` and
``qlist`` are replaced by ``discover(regex)`` (a1):

.. ipython:: python

    qd(build.discover('Dice'))

``build_many`` is the explicit batch counterpart to ``build`` (which now raises
``CannotBuild``, a ``ValueError``, rather than returning ``None``-ish objects,
a2). ``.more()`` is renamed ``.help()`` everywhere (a2).

Saving work: ``to_agg`` (a32, a35, a53)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``to_agg(path)`` writes selected knowledge entries back to a re-loadable ``.agg``
file: by default everything built this session, in dependency order (severities
and distortions before the aggregates that reference them, a35), in canonical
DecL via the unparser (a53). Modes mirror Python's: ``'x'`` (safe default,
refuses to clobber), ``'w'``, ``'a'``:

.. ipython:: python

    import tempfile, os
    uw.build('agg Demo 5 claims sev lognorm 10 cv 0.3 poisson', update=False)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'demo.agg')
        uw.to_agg(path, mode='w')
        print(open(path).read())

``discover`` refuses what it cannot honor (a173)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``build.discover(tag='role:hero')`` used to return every entry in the recipe
base. A filter that silently matches everything is the worst possible answer: it
looks like a successful query, and it denies that the tag vocabulary exists. The
tag filter was never wrong. The parameter is ``tags``, plural, and ``discover``
accepts ``**kwargs`` so it can forward build options, so the singular spelling
bound into ``kwargs`` and the lightweight directory path never read them. Nothing
was filtered because nothing was asked.

Two rules now run before any work. A kwarg close to one of ``discover``'s own
parameters is a mistyped filter and is rejected with the house ``Did you mean:``
hint; anything left over is a build option, forwarded only when ``plot``,
``describe`` or ``return_objects`` actually asks for a build.

.. ipython:: python

    for call in ("build.discover(tag='role:hero')", 'build.discover(log2=16)'):
        try:
            eval(call)
        except TypeError as e:
            print(f'{call}\n  {e}\n')

Separately, an empty filter intersection no longer raises. When ``regex`` or
``kind`` had already left zero rows the tag mask was an empty list, which pandas
does not read as a boolean mask, so ``df[[]]`` selected zero *columns* and the
lookup downstream failed:

.. ipython:: python

    print(build.discover('NoSuchName', tags='role:hero').shape)

``interpret_file`` reads statements, not lines (a176)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``interpret_file`` reported 67 parse errors on ``agg/library.agg`` and 13 on
``agg/decl-testers.agg``. Both counts are now zero, and neither file ever had
anything wrong with it: the function split raw text on **newlines** and parsed
each physical line, a model that predates multi-line statements and the
``doc{{{ }}}`` trailer. A doc body is markdown, so every line of it became a
bogus statement and every one failed. The production path has always split on
statements, which is why the same files load without complaint.

.. ipython:: python

    from pathlib import Path
    import aggregate
    lib_path = Path(aggregate.__file__).parent / 'agg' / 'library.agg'
    interp = build.interpret_file(lib_path)
    print(f'{len(interp)} statements, {(interp.error != 0).sum()} errors')
    qd(interp.head(4))

The ``preprocessed program`` and ``program`` columns collapse into one
``program`` column holding the statement: the old pair compared a preprocessed
line against its raw source, a distinction that does not exist when the unit of
work is the statement. Rows are also collected positionally rather than keyed on
the entry name, so two entries sharing a name no longer overwrite each other.

``build`` finishes every kind (a169)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two loose ends in ``build_many``'s output dispatch. A
:class:`BivariateAggregate` built with ``update=False`` came back correct and
un-updated, exactly as asked, and then logged ``Unexpected: output kind is …`` at
you, because the no-op escape hatch for ``update is False`` listed only
``(Aggregate, Portfolio)`` and a bivariate is not an :class:`Aggregate` subclass.
The branches also tested ``update is True`` and ``is False`` by identity, so
``update=1`` matched neither.

**Behavior change:** the grammar has always had a top-level ``expr`` production,
but there was no branch for it, so a bare expression raised
``ValueError: Cannot build expr objects`` after already writing an orphan entry
into the recipe base. An expression is an answer, not a declaration, so it now
evaluates to its value and is never stored:

.. ipython:: python

    print(build('3'), build('2 ** 3'), f'{build("exp(1)"):.6f}')

Note that a *bare* expression accepts a subset of arithmetic: ``1 + 2`` and a
trailing ``2 * 3`` still do not parse, because ``*`` is the severity scale
operator and ``+`` the severity shift. Since 1.0.0a268 those operators are legal
inside parentheses, where a single expression is the only reading, so
``(1 + 2)`` and ``(2 * 3)`` both evaluate:

.. ipython:: python

    print(build('(1 + 2)'), build('(2 * 3)'), build('(4 + 3*2)'))

Pedagogy helpers
----------------

The deleted ``extensions/`` package's doc-cited figure machinery lives on in
``aggregate.pedagogy`` (a12): ``bodoff_exhibit``, ``plot_twelve``,
``plot_bivariate``, ``plot_distortion_and_ins_stats``,
``plot_spectral_three_panel``, ``ClassicalPremium``, the book figures
(``plot_quantile_illustration`` and friends), ``plot_lee``, ``plot_max_min``, and
the similar-risks plots. Submodule access only: none of it is core API.

Exponential tilting (a46) returns here as pedagogy, not production. The
operational aliasing control remains padding, but
``tilted_aggregate_density(agg, log2=, bs=, tilt=)`` runs a single tilted
convolution locally for studying aliasing after Grübel and Hermesmeier
:cite:p:`Grubel1999,Grubel2000`, and ``gh_tilting_exhibit()`` reproduces their
Poisson comparison table:

.. ipython:: python

    from aggregate.pedagogy import tilted_aggregate_density
    td_ = tilted_aggregate_density(dice, log2=6, bs=1, tilt=0.05)
    print(f'tilted convolution mass: {td_.sum():.6f} (reproduces the ordinary result; '
          f'tilt=None is byte-identical)')

.. _feat pnl:

P&L: profit and loss as a first-class object
--------------------------------------------

The single largest addition since a57. In a23 ``pnl`` was a keyword that shifted
an aggregate; today it builds a :class:`PnL`, an object with its own ledger,
exhibits and reinsurance walk. The arc runs a102 to a191.

The ledger: Consideration, Obligation, Margin (a103, a122, a123)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A P&L is money in minus money out over a random state. ``summary_df`` is that
statement, not a moment table: **Consideration** is what you are paid,
**Obligation** what you owe (negative, by sign convention), **Margin** the sum.

.. ipython:: python

    qd(pnl.summary_df)

``stats_df`` breaks the same ledger out by label, and carries the **κ columns**:
``κ01 … κ99`` are conditional expectations of each line *given the total lands at
that percentile*. They answer "when the year is bad, which line is driving it?",
and they foot exactly, by construction, to the Total row.

Since a125 a ``pnl`` wraps a **complete named engine**, written
``less agg <name> <program>``. The premium can be inherited from the engine's own
technical premium rather than restated:

.. ipython:: python

    inh = build('pnl Inherit inherit premium less agg Inherit_e '
                '8000 prem at 65% lr sev lognorm 100 cv 2 poisson')
    qd(inh.summary_df)

Expenses and loss-sensitive legs (a114, a115)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expenses are ``and``-joined obligation legs. Three kinds, distinguished by what
they scale with: ``% loss expense`` rides the loss (so it is stochastic),
``% premium expense`` rides the premium, and ``fixed expense`` is a constant.

.. ipython:: python

    book_pnl = build('pnl Book 10000 premium less agg Book_e 10000 premium at 75% lr '
                     '10000 xs 0 sev 10.808 * lognorm 1.75 mixed gamma 0.25 '
                     'less 5% loss expense as LAE 100 fixed expense '
                     'and 10% premium expense')
    qd(book_pnl.economic_df)

The LAE row has the same CV and skew as the Loss row because it *is* the loss,
scaled. The expense row has zero SD. That separation is the point: a single Total
hides which legs carry risk and which merely carry cost.

``pnl`` and ``xpnl``: the net view and the walk (a136)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two objects, two questions. **``pnl``** is the consolidated **net** view: one
group, everything folded in, "what did the book earn?". **``xpnl``** is the
**walk**: gross, then each reinsurance step, then the total, "where did the
margin go?".

.. ipython:: python

    tower = build('xpnl Tower 1000 prem less agg Tower_e 1000 prem at 70% lr '
                  'sev lognorm 100 cv 2 occurrence ceded to 500 xs 500 deposit 100 poisson')
    qd(tower.summary_df)

Read it down the ``Step`` index: ``Gross`` is the book before reinsurance,
``occ 500 xs 500`` is the cover as its own signed block (you pay 100 of premium,
you recover 51 on average, so the cover costs 49 in expectation), and ``All`` is
the net result. The ``Impact`` row is the difference the step made. Each step's
rows are an affine function of a ``reins_density_df`` marginal, so the walk is
exact rather than simulated.

**Breaking at a183.** That middle step used to be called ``ceded occ``. An
undeclared cover step whose tier holds exactly **one** layer is now named by that
layer's own DecL descriptor, so a partial share reads ``agg 85% po 1500 xs 7000``
and the step name says what it covers. A multi-layer tier keeps the generic
``ceded occ`` / ``ceded agg``, because it has no single descriptor, and section
14.5 is how you see those layers separately. A layer's declared ``as`` label
still wins.

Acceptability: ``evaluate()`` (a105, corrected a187, a188, a190)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluate`` returns the breakeven panel: the level of distortion at which a
position stops being acceptable. **a187 is a correction, not an enhancement.**
The panel used to solve :math:`\rho_g(\text{obligation}) = E[\text{consideration}]`;
it now solves :math:`\rho_g(\text{margin}) = 0`, and a variable-premium position
evaluated on the old form returned a number that answered the wrong question.

The reason is worth carrying, because it says exactly when the old form was fine.
Distortion risk measures are translation-equivariant, so with a **constant**
premium :math:`P` the statements :math:`\rho_g(P - L) = 0` and
:math:`\rho_g(L) = P` are the same, and pricing the obligation against the
premium was legitimate. Once the consideration is random (swing rating, slide,
profit commission, reinstatement premium, corridor) that equivalence fails:
:math:`\rho_g` is comonotone-additive but not additive, and the margin
:math:`M = P - L` is generally not monotone in :math:`L`. Only the margin's own
pushforward answers the question. The visible symptom was a loss-sensitive P&L
with :math:`E[P] < E[L]` calibrating happily to a "premium" below its own
expected loss. **Constant-premium answers are unchanged**, which is the
regression anchor.

Three things fell out of that. The single-obligation-leg restriction is **gone**,
a margin being one random variable however many legs feed it, so expense ledgers
evaluate; so is the regular-grid restriction, because a margin pushforward
generally lands on an irregular support and the quadrature now takes a per-node
width vector. Every margin row of a ledger is evaluated, not just the grand
result. And there are new ``Aggregate.evaluate`` and ``Portfolio.evaluate`` faces
returning the same tidy frame, so panels concatenate:

.. ipython:: python

    qd(simple.evaluate(P=1.15 * simple.est_m))

``P`` defaults to ``exp_premium`` and **raises** when unset, since a premium of 0
would silently make every position unacceptable.

On a walk the panel becomes a **buy decision** (a188). A cession's margin to the
buyer is negative by construction, because you pay for cover, so the old
:math:`E[M] > 0` guard fired on every reinsurance row and six of thirteen rows on
a two-tier program came back ``NaN``. The question worth asking about a purchased
layer is what stress the **seller's** position survives, which is the buyer's
margin negated:

.. ipython:: python

    qd(tower.evaluate())

Read the ``gini_p`` column down the steps. The gross book sits at 0.681; the
layer, read from the seller's side, at 0.731; and the running net after buying it
at 0.681. A layer priced **above** the holder's own acceptability lowers the net
when bought, one below it raises the net, and the sheet now says which. The
leading ``role`` column names whose position the parameters describe, and it is
keyed off ``Group.role`` rather than a row label, so a hand-built ledger with two
sold books is not wrongly flipped.

``role`` has three values, not two (a190). A running net and the grand result buy
against selling, so they are neither side and read ``net``; a ``sell`` group's
own result reads ``sell`` and a cession's reads ``buy``; a tier subtotal takes
its span's shared role, or ``net`` if the span mixes. Numbers are unchanged,
``net`` and ``sell`` carrying the same sign. Also at a190, the ``total impact``
row is **no longer evaluated at all**: it is the grand result less the first
group's, which makes it a difference between two positions rather than a
position, and nobody holds it, so the breakeven stress it survives is not a
question with an answer. It remains an ordinary row on ``stats_df``,
``summary_df`` and ``density_df``.

A degenerate position reports ``NaN``, never ``0`` or ``inf``, with a
``DegenerateEvaluationWarning`` raised once per call naming the affected steps.
The two cases are :math:`E[M] \le 0`, acceptable at no stress, and
:math:`M \ge 0` almost surely, an arbitrage acceptable at every stress; the
``status`` column names which fired. Since a188 that warning is rare and
meaningful, firing on a layer priced below its own expected recovery rather than
on the routine fact that cover costs money.

**Breaking:** ``evaluate`` gained the leading ``role`` column, so code selecting
columns positionally needs updating (``EVAL_COLS`` is the canonical order); code
indexing a ``total impact`` block raises ``KeyError``; and
``PnL.E_consideration`` is retired along with the ``info`` row, since
``evaluate`` no longer targets a premium and ``ratio_df['P']`` is the
replacement.

Peeling a tower, layer by layer (a183, a184)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tier walk books one group per reinsurance *tier*, so a five-layer program had
one lumped ``ceded occ`` step. The DecL clause ``peel`` books one group per
**layer** instead, so every layer reports its own ceded premium, ceding
commission and marginal impact, and ``net through <layer>`` builds the program up
a layer at a time:

.. ipython:: python

    peel = build('xpnl Peel 1000 premium less agg Peel_e 1000 premium at 70% lr '
                 'sev lognorm 100 cv 2 '
                 'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 deposit 40 '
                 'poisson peel top-down')
    qd(peel.summary_df)

Occurrence layers are peeled before aggregate ones, preserving the tier walk's
step order; within a tier ``top-down`` introduces the highest-attaching layer
first and ``bottom-up`` the lowest. Zero-share gap fillers are structural, not
cessions, and get no step. The clause is a modifier on an existing statement
rather than a new object kind, because peeling is a way of building, not a new
type.

``All occurrence`` is the a184 **tier subtotal**: a tier that peels into two or
more steps gains its own block after the last step it spans, restoring the tier
line the plain walk carried. A tier that peels into one step already *is* its own
subtotal and gets nothing, which is why a one-layer-per-tier peel is byte
identical to the plain walk.

One caveat is worth knowing, and it is narrower than it sounds. Aggregate layers
are disjoint intervals on one subject, so they stay on the shared per-atom route
and the scenario κ ladder survives. **Occurrence** layers are not: the
ceded-occurrence aggregate is not a function of the gross aggregate, because the
random claim count decouples them. Two or more peeled occurrence layers therefore
route through the kernel's stitched seam, where each row's marginal is still
exactly one FFT of a transformed severity and the ``EX`` column **foots exactly**
by linearity, but the dispersion columns are marginal, so the ladder carries
plain ``P01…P99`` headers rather than κ, and ``evaluate`` and ``+`` are
unavailable.

The ledger's index: ``Side`` and ``Label`` (a189, a191)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking.** The :class:`PnL` sheets renamed their index levels. ``View`` became
**``Side``** and ``Line`` became **``Label``**, on ``stats_df``, on
``summary_df``, and as ``legs_df`` columns:

.. ipython:: python

    print('single group:', pnl.economic_df.index.names)
    print('walk        :', tower.stats_df.index.names)
    print('labels      :', tower.stats_df.index.get_level_values('Label').unique().tolist())

``Line`` assumed a line of business, which the ledger does not otherwise assume,
and the level holds presentation labels. ``Side`` was already the internal word
and this promotes it. ``Leg`` was rejected for the name because ``Leg`` is the
public class for an individual declared cash flow, and those now sit under
``Label``, so a level named ``Leg`` would be the one level with no leg names in
it.

``Total`` had been doing three jobs: a within-step subtotal, the direct result,
and the grand result. On a ledger that **buys** something they separate, gated on
the ledger actually containing a ``buy`` group, because only then is there
something to be direct *of*. The grand consideration, obligation and result read
**``Net``**; a cession's own result keeps ``Total``; and the first, sold step's
own result takes the engine's declared label, defaulting to ``Gross`` (a191
replaced a189's short lived ``Direct`` here, which never survived contact with a
labelled sheet).

a191 also made the walk read the labels you declared. The first ``Step`` is the
P&L's own ``as`` label rather than the engine's, and the default consideration
leg label is ``Premium``, capitalized, matching ``Loss``:

.. ipython:: python

    lab = build('xpnl Deal as "ABC" 1000 premium as "XYZ" less '
                'agg Deal_e as "LLL" 1000 premium at 70% lr sev lognorm 100 cv 2 poisson')
    qd(lab.stats_df.iloc[:, :4])

Migrating: first-step keys move from ``('Gross', …)`` to your own ``as`` label;
``('All', 'Margin', 'Total')`` becomes ``('All', 'Margin', 'Net')``;
``('Consideration', 'premium')`` becomes ``('Consideration', 'Premium')``; and
code that filtered legs out with ``Label not in ('Total', 'Net', 'Impact')``
needs the direct label in that set. A step's own result is still the first
``Margin`` row of its block in plan order, whatever the label reads.

Ratios in their own frame: ``ratio_df`` and ``legs_df`` (a185, a186)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking.** The ``Scaled`` column is gone, and with it ``scaled_stats_df``, the
``scale`` property and the ``scale=`` constructor argument. ``Scaled`` divided
every cell by one committed number, ``E[grand total consideration]``, which on a
walk is gross premium minus every cession, so on a two-group tower the gross
premium cell read 1.43 and the gross "loss ratio" read -0.571 against a true
0.40. The combined-ratio reading was an accident of the single-group case. Ratios
now live in their own table:

.. ipython:: python

    qd(tower.ratio_df)

The amounts are signed **in the gross direction**: consideration as booked,
obligations negated. A cession's ceded premium and recovery are therefore both
negative, which buys three things at once. The amounts **add across blocks**, so
layers sum into their tier and tiers into ``All``. ``M == P - L - E - C`` holds
identically, being the signed row sum, with ``1 - CR == M / P`` as its other
reading. And every ratio comes out with its **conventional sign**, because
numerator and denominator flip together: the cover above reads ``LR = 0.51``, not
``-0.51``, and a cover that paid back three times its premium would read
``3.10``. Ratios are re-derived from each row's own amounts, never averaged from
the blocks below.

``LR`` and ``E_LR`` are **the ratio of the means and the mean of the ratio**, and
they are reported separately because they are different numbers. When premium is
random and correlated with loss they diverge:

.. ipython:: python

    retro = build('pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium less '
                  'agg R_e 1000 loss sev lognorm 100 cv 2 poisson')
    qd(retro.ratio_df[['P', 'L', 'M', 'LR', 'E_LR']])

``LR = 0.2426`` against ``E_LR = 0.2256``, a 7% relative gap in the direction the
retro implies, since premium rises with loss and damps the per-atom ratio. With a
deterministic premium the two agree to the last bit, which is why ``tower`` above
shows them equal. a186 renamed these from ``EX_LR`` (which read as "the ``EX``
column, of ``LR``", not what it is) and fixed the availability gate, which had
asked whether a joint existed when the question is whether the **denominator is
random**: a constant premium factors straight out of ``E[X / P]``, so the mean of
the ratio is exact on any route.

``legs_df`` is the itemized companion, one row per **declared** leg. Derived rows
are absent by design, being sums of these, and it is the only place ``Leg.kind``
(one of ``premium``, ``loss``, ``expense``, ``recovery``, ``commission``)
surfaces. That kind is what makes an expense ratio possible at all, since nothing
in ``stats_df`` distinguishes a loss leg from an expense leg but the label text:

.. ipython:: python

    qd(book_pnl.legs_df)

Both frames are **raw materials**: unformatted, and deliberately absent from
``qd`` and the notebook repr, which keep rendering ``summary_df``, the
presentation-ready layer.

.. _feat bivariate:

Bivariate aggregates
--------------------

The rename, and what the object is (a80)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking, a80: ``multivariate`` is ``bivariate``** (short form ``bv``), and the
class is :class:`BivariateAggregate`. The old name promised a generality the 2D
FFT engine does not have: it is genuinely two dimensions, not n.

Axis sizing was rebuilt over a70 to a76 to *measure* rather than guess: each axis
gets an honest centred window (``balanced_window``), the lower edge comes from
the measured support instead of an artificial pin at 0, and ``round_bucket`` lost
its 2.5x jumps (a74) so the ladder is smooth.

The reporting surface (a77, a85)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three frames, each owning one question. ``summary_df`` follows the
``Portfolio.summary_df`` shape: a shared ``Freq`` block, a ``Sev`` / ``Agg``
block per component, and a ``total`` block for the genuine :math:`X + Y`.
``stats_df`` is pure per-component marginal moments. **``dependency_df``** (new
at a85) owns everything joint:

.. ipython:: python

    qd(mv_copula.dependency_df)

``tau`` is the input copula's Kendall tau, a **per-claim** property. The ``Sev``
row is the joint per-claim severity and the ``Agg`` row the realized aggregate.
They differ, and the gap is the whole story: compounding attenuates per-claim
dependence while shared frequency mixing adds common-shock dependence on top.

``clash``: two lines, one event (a79)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A clash program shares events across two lines with per-line claim counts. Here 8
events, of which 5 hit line 1 and 2 hit line 2:

.. ipython:: python

    clash = build('clash Clash 8 5 2 claims sev lognorm 50 cv 1.2 '
                  'sev lognorm 60 cv 1.5 mixed gamma 0.2')
    qd(clash.dependency_df)

Per-claim severity dependence is zero (the severities are independent given an
event), yet the aggregates are correlated: the shared event count and the shared
gamma mixing do all the work. That is exactly the clash structure, and it is
invisible if you only look at marginals.

View pairs (a78)
~~~~~~~~~~~~~~~~

:ref:`netceded <feat netceded>` has two siblings. ``grossceded`` and ``grossnet``
prefix the same reinsured program and return the corresponding joint law, so you
can ask for whichever pair the analysis needs without rebuilding.

Massive: disk-backed bivariates (a126)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 2D grid at useful resolution can exceed memory. ``update(store_dir=...)`` runs
the update out of core, backed by ``zarr`` (the ``massive`` extra), with dict
pushforwards and pyramid visualization for exploration. One caveat worth
carrying: with ``padding=0`` the wrap hides from the deficit check, so compare
means rather than trusting the deficit.

``axis_support_df``, and labels on a bivariate (a171)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking: ``BivariateAggregate.tail_df`` is ``axis_support_df``.** It was a
collision, not an analogy. ``Aggregate.tail_df`` and ``Portfolio.tail_df`` are
return-period ladders; the bivariate frame under the same name reported something
else entirely, namely where the realized mass sits on each axis. Two different
reports answering to one name is how a reader gets the wrong one.
``tail_description`` and ``tail_explanation`` keep their names, because what they
describe really is the tail.

:class:`BivariateAggregate` was also the last class carrying neither half of the
label surface, so ``label``, ``label_map``, ``labels``, ``renamer`` and
``use_labels`` now work on a bivariate, and the renamer resolves each axis to its
component :class:`Aggregate`'s label:

.. ipython:: python

    bcat = build('bivariate Cat 25 claims '
                 'agg Wind as "Windstorm" dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
                 'agg Flood as "Flooding" dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
                 'copula gumbel 0.4 poisson')
    qd(bcat.axis_support_df)
    bcat.use_labels = False
    print(bcat.axis_support_df.index.tolist())

Labels are applied at serve time in ``axis_support_df``, ``bs_window_df`` and
``stats_df``, the frames keyed by axis. The component labels come through DecL
because each unit is an ordinary ``agg``; an **object-level** label has no DecL
spelling yet, so it is set with ``label=``, a grammar gap logged as
``[Bivariate-DecL-Label]``.

Reinsurance economics: stochastic ceded premium
-----------------------------------------------

Guaranteed-cost reinsurance has a premium you know at inception. These features
share one theme: the ceded premium is **random**, because it is a function of the
ceded loss. That is what makes them require the aggregate machinery rather than a
spreadsheet.

Reinstatements (a116, a117)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Property-cat covers are paid for as they are used. A layer with
``reinstatements [1]`` carries a deposit premium plus a reinstatement premium
driven by the recovery:

.. ipython:: python

    qd(reinst.reins_summary_df)

Variable rating (a118, a119, a120)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Five features, all now expressible in DecL: **swing** (``swing basic B lcm L min
m max M``), **slide** (a sliding commission), **pc** (profit commission),
**corridor** (a retained corridor inside the layer), and **retro**. Each makes
the ceded premium, or the effective recovery, a function of the loss.

.. ipython:: python

    qd(swing.reins_summary_df)

Placement scaling (a143)
~~~~~~~~~~~~~~~~~~~~~~~~

Reinsurance premiums in a ``pnl`` or ``xpnl`` are quoted at **100% placement**
and scaled down by the fraction actually placed. Quoting at 100% and scaling once
at the end is the market convention, and it keeps a partial placement from
silently rescaling the layer terms.

Renewal frequency and ruin
--------------------------

Sparre-Andersen: the ``wait`` clause (a146, a147)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until a146 the claim count was Poisson or a mixed Poisson. It can now be a
general **renewal process**: claims arrive after iid waiting times of any law.
Write ``1 year`` in place of the count and give the waiting-time distribution:

.. ipython:: python

    qd(renewal.summary_df)

The expected count is an **output**, not an input: it falls out of the waiting law
and the horizon. a147 added layer terms on the wait clause
(``wait y xs a <dist>``), so waiting times can be capped or floored.

Zero-truncated and zero-modified frequency (a152)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``zt`` and ``zm`` were reparameterized to the textbook form and fixed; ``zt``
previously raised. The parameter is now the one the textbooks use, so a ``zt``
negative binomial matches *Loss Models* directly.

Eventual ruin: ``wiener_hopf`` (a153)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a renewal aggregate, ``wiener_hopf(rho)`` returns the eventual-ruin function
by a cepstral Wiener-Hopf factorization, at loading ``rho``:

.. ipython:: python

    wh = renewal.wiener_hopf(0.2)
    print(f'psi(0)  = {wh.ruin.iloc[0]:.4f}')
    print(f'mean claim size = {wh.mean:.4f}')
    print(f'capital for 1% ruin: u = {wh.find_u(0.01):.2f}')

``find_u`` inverts it: the capital needed to hold eventual ruin below a target.
The classical Pollaczek-Khinchine solver is still there and now carries a strict
Poisson guard, so it refuses rather than silently returning the wrong answer for
a renewal process. Defective waiting laws are refused outright.

The recipe library
------------------

The last arc before v1.0 (a157 to a168): the shipped ``.agg`` library describes,
demonstrates and tests itself.

``doc{{{ }}}`` and ``tags{ }`` (a157)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DecL trailer grew from two clauses to four. ``note{}`` is the one-line
description and ``hints{}`` the build settings, as before. New: **``tags{}``**, a
namespaced classification (``topic:``, ``role:``, ``check:``), and
**``doc{{{ }}}``**, a fenced markdown body carrying a full Problem / Solution /
Discussion / Check recipe. The fence survives markdown headings, blank lines and
Python code because it is extracted and base64-encoded *before* the comment
stripper runs. ``distortion`` gained a trailer at the same time, so distortions
are first-class in describe, test and audit like everything else.

One library (a159, a161)
~~~~~~~~~~~~~~~~~~~~~~~~

``examples.agg``, ``cookbook.agg`` and ``actuarial-severity-curves.agg`` merged
into a single **``library.agg``**, names made unique across kinds, letter
prefixes retired. It is the default database, so the out-of-the-box knowledge
base grew from 36 entries to 186.

``build.recipe()`` and ``build.recipes`` (a158, a164)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    print(f'{len(build.recipes)} entries; {build.recipes.doc.sum()} carry a full recipe')
    print(build.recipes.query('doc').index.tolist())

``build.recipes`` is the audit frame: what is documented, what is demonstrated,
what is checked. ``build.recipe(name)`` returns the entry itself. At a164 the two
objects that described one entry collapsed into one ``Recipe`` class, and
``knowledge`` was retired in favor of ``recipes``.

.. ipython:: python

    qd(build.recipes.head(8))

Tags are queryable, which is what replaced the old comment-banner table of
contents:

.. ipython:: python

    print(f"hero entries: {len(build.discover(tags='role:hero'))}")

The cookbook generates itself (a165, a167)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cookbook pages are **generated** from ``library.agg`` into native Quarto cells by
``aggregate.cookbook``. The same ``doc{{{ }}}`` body is executed by
``tests/test_library_recipes.py``, so the page you read and the test that guards
it are the same source and cannot drift. ``<<decl>>`` (a163) substitutes the
entry's own program, so a recipe never retypes the code it documents.

.. warning::

    Running a recipe executes the code in the ``.agg`` file. Treat a third-party
    ``.agg`` like a third-party Python file.

Under the hood
--------------

Internal changes, named for the record, with no new user-facing surface. Versions
refer to ``CHANGELOG.md`` sections.

- **a2** :class:`Aggregate` surface privatization (``_audit_df``, ``_limits``,
  and the rest); FFT core extracted to ``_freq_sev_convolution``.
- **a4** PEG regression baseline, the numeric contract every refactor
  sub-project had to reproduce.
- **a5** auto-approximation and FFT tilting removed from the ``update`` pipeline
  (**breaking**: the ``approx_freq_ge`` / ``approx_type`` / ``tilt_amount``
  kwargs are gone; callers always get the exact FFT). The replacements are
  design-time ``approximate`` (a47) and pedagogy tilting (a46).
- **a6** distortion calibration moved onto the :class:`Distortion` subclasses;
  the dead ``tt`` branch removed.
- **a8** ``utilities.py`` 3,753 to about 700 LOC; new ``moments`` and
  ``iman_conover`` modules; public ``*_fit`` family (``lognorm_fit``,
  ``gamma_fit``, ``sgamma_fit`` and friends) in ``distributions``; src/ package
  layout.
- **a9 / a10** ``Tweedie`` to ``aggregate.tweedie``; ``FourierTools`` to
  ``aggregate.ft`` (**breaking**: submodule access only; old
  ``aggregate.extensions.*`` import paths are gone, a12).
- **a16** per-module ``__all__``; no import-time warning muting.
- **a17** forwards-``S`` default in ``Distortion.price``; single
  ``_fft_aggregate`` core; eta-mu/second-priority surface removed;
  journey-of-discovery comments scrubbed.
- **a38** knowledge-freeze regression harness (``scripts/freeze_knowledge.py``),
  snapshot and verify all knowledge-base objects at 1e-12.
- **a42** fuzz removal consolidated into vectorized ``utilities.remove_fuzz``.
- **a44** ``make_ceder_netter`` moved to ``distributions``; unused runtime deps
  dropped (``cycler``, ``psutil``, ``ipykernel``, ``jinja2``); ``IPython``
  import made lazy (about 1s off cold import).
- **a49** portfolio combine grid: ``best_window`` (resolution plus span)
  replaces the RMS combine. See Grids and windows for the user-visible effect.
- **a55 to a57** the numerics spine: unit-density decoupling, shifted-support
  kappa with direct sums carrying the grid origin, and the single exact-discrete
  Choquet engine behind all distorted pricing. User-visible column changes are
  covered in the Reporting quartet and Pricing sections.
- **a66** portfolio windowed combine: Portfolio method-of-moments plus a
  single-big-jump look-through for the combined grid.
- **a67 to a68** ``Aggregate.approximate()`` was shadowed by a same-named
  attribute and had been uncallable; the fix, then one fit core with a symmetric
  guard and honest reflected-fit errors. The naming rule in ``CLAUDE.md`` about
  vetting a new attribute against existing methods comes from this bug.
- **a71** ``.agg`` library rationalization (the first pass, later superseded by
  the a159 merge).
- **a83** config phase 2: the last hard-coded numerics floors and stranded
  sizing knobs moved into ``aggregate.config`` (**breaking**:
  ``aggregate.constants`` no longer exposes ``FT_NOISE_FLOOR``,
  ``ALIASING_RATIO``, ``EXEQA_NOISE_FLOOR``, ``DEFICIT_MATERIALITY``).
- **a88** DecL syntax colorer and error labels resynced with the grammar.
- **a90 to a91** ``GridDistribution``, the shared discrete-grid distribution
  value type, adopted by :class:`Aggregate`, :class:`Portfolio` and
  :class:`Bounds`. Every quantile, VaR, TVaR, cdf and sf now routes through one
  implementation instead of hand-rolled searchsorted calls.
- **a92** plotting subsystem: a single matplotlib boundary.
- **a94** ``portfolio.py`` split into a facade plus three subsystems.
- **a95 to a96** shared-concern modules filled in, legacy bucket sizers retired,
  moment helpers rationalized.
- **a111 to a112** ``GridDistribution`` learned its own sign (loss vs payoff),
  and the Lee/quantile worker consumes it.
- **a121** the bivariate leg kernel: a domain-free engine for the P&L leg model,
  with insurance as a View on top.
- **a127** parallel-by-default test suite (``-n auto``) and the fast local loop.
- **a144** the ``ReinstatementAnalysis`` and ``VariableRatingAnalysis``
  drill-down classes removed; the generic P&L absorbed their work.
- **a145** the ``exact_discrete`` grid-sizing method lost unconditional top
  priority and must now clear a reachability guard.
- **a149 to a151** the first-class-class surface sweep: ``dev/FEATURES.csv``
  worked class by class, one ``help()`` implementation replacing nine copies
  (``HelpMixin``), then the breaking alias retirement. One name per concept.
- **a154** the DecL round-trip surface (``program`` / ``pprogram`` /
  ``pprogram_html``) collapsed into a third mixin.
- **a162** test-impact analysis for the edit loop (``pytest-testmon``).
- **a174** ``dev/regen_features.py`` dropped every ``_``-prefixed name from its
  introspection, which made ``dev/FEATURES.csv`` structurally blind to
  ``__repr__``, ``_repr_html_`` and the info blobs, the most duplicated part of
  the library and the part that item was about auditing. An explicit
  ``DISPLAY_MEMBERS`` allowlist lets those four through, and they are now a
  ``display`` group in the matrix with the remaining gaps written down rather
  than rediscovered. ``Tweedie._repr_html_`` was spelled ``__repr_html__`` and
  so was unreachable (dead code, not a broken display,
  ``_repr_mimebundle_`` serving the same HTML), and ``Copula`` was the one class
  carrying ``LabeledMixin`` without ``HelpMixin``.
- **a175** the guard behind the colorizer resync:
  ``tests/test_grammar_sync.py`` now tokenizes all four shipped corpora
  asserting zero ``Token.Error``, checks the token values concatenate back to
  the source byte for byte, and derives the brace-clause words and operator
  literals from the grammar, so a fifth trailer clause fails the suite. 169
  cases became 570. ``agg.sublime-syntax``, the second hand-written mirror, was
  resynced and brought under the same guard.
- **a178** ``dev/reflow_library.py`` renders each library entry through
  ``format_program(layout='spread')`` and refuses to write unless every
  statement still parses to the same ``(kind, name, spec)``. Whitespace between
  tokens is insignificant to the lexer, so that check is complete: where the
  breaks fall is taste, whether the meaning moved is not.
- **a179** three ``np.errstate`` guards for the TVaR endpoint.
  ``make_var_tvar`` pads its tail arrays with ``inf`` sentinels and selects with
  ``np.where``, which evaluates *both* branches, so at :math:`p = 1` the
  discarded branch computes ``0 * inf`` and a ``1 / 0``. The numbers were always
  right; only the reporting was noisy, and it was never specific to
  :class:`Bounds`. This does not make the library warning free: measured with
  ``pytest -W error::RuntimeWarning``, 71 failures and 4 errors became 52 and 0,
  with the remainder tracked as ``[RuntimeWarning-Census]``.
- **a182** ``make_ceder_netter`` mis-ceded on any program with three or more
  layers and two or more genuine gaps. The running top of the program was
  tracked by accumulation rather than assignment, so the test that emits the
  knot holding a cession flat across a gap silently failed from the second gap
  onward and the ceder interpolated straight across. A gap written the
  documented way, as a zero-share filler layer, keeps the attachments contiguous
  and so never triggered it, which is why this survived.
- **a184** three latent kernel problems fixed in passing: ``_assemble_rows``,
  ``_init_massive`` and ``_view_index`` each closed their row-kind dispatch with
  a bare ``else`` that *meant* ``'total_impact'``, so an unhandled kind was
  silently booked as the impact row; and ``PnL.__add__`` reconstructed from a
  fixed argument list, so any new constructor argument silently dropped on
  composition.
