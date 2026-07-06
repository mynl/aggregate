## The scenario (κ) ladder in `PnL` objects

The ledger is evaluated per atom over one shared source: every row $X_i$
(leg, total, running net) and the grand result $M=\sum_i X_i$ (the margin)
are vectors of signed values on the same atoms $\omega$ with probabilities
$p(\omega)$. For each ladder point $q \in \{0.01, 0.05, \dots, 0.99\}$ the
scenario is anchored on the **margin**: let
$m_q = \inf\{x : \Pr(M \le x) \ge q\}$ be the lower $q$-quantile of $M$.
Because the atoms carry the exact result values, the event
$A_q = \{\omega : M(\omega) = m_q\}$ is a nonempty exact slice (float
equality, no tolerance), and each cell of column $\kappa_q$ is the
conditional mean of its row over that slice:
$$
\kappa_q(X_i) \;=\; \mathsf P\left[X_i \mid M = m_q\right]
\;=\; \frac{\sum_{\omega \in A_q} X_i(\omega)\, p(\omega)}
            {\sum_{\omega \in A_q} p(\omega)} .
$$
This is the kappa function $x \mapsto \mathsf P[X_i \mid M = x]$
evaluated at the margin's own quantiles. By linearity of conditional
expectation every column **foots**: legs sum to totals and totals to
$\kappa_q(M) = m_q$, so the grand-result cell is automatically its own
marginal quantile. Signs follow the payoff convention (left tail bad), so
$\kappa_{01}$ is the adverse state and a loss-sensitive premium correctly
shows *high* in the bad columns. Where the margin is a monotone
(decreasing) function of the underlying loss — any plain book — the level
sets $\{M = m_q\}$ and $\{L = \ell_{1-q}\}$ coincide, so the column equals
a gross-loss scenario at the complementary quantile; where it is
non-monotone (a swing collar, a slide) the cell is the exact mean over the
full level set $\{M = m_q\}$, which is well-defined but subtler to read.

Notes:

1) on a 2-D source the atoms $\omega$ are the joint's cells $(L, R)$, and everything above applies unchanged
2) a fully hedged (constant) margin makes the conditioning event everything, so every cell collapses to its $\mathsf P[X_i]$.


---

Where it lives in code
-----------------------

## Where it happens — the map

The kernel (`PnL`) never builds anything; it just receives a source and flattens it into atoms. The 2-D decision and construction live one layer up, in the DecL dispatch and the builders:

- **Routing decision** — `Underwriter._snapshot_pnl`, `src/aggregate/underwriter.py:1769`: kind `gcn` + `is_tower` → `build_xpnl_walk` (:1774). The classifier that set `kind` is in `_factory`'s pnl branch (~:1130–1230).
- **The joint gets built inside the builders**, not the Underwriter:
  - GC occ walk: `_pnl_builders.py:532` — `biv = agg.occ_bivariate(views=('gross', 'ceded'))`, adequacy check at :534, and `source=biv.bivariate` goes into the `PnL`.
  - Feature-composed walk: `_pnl_builders.py:854` (same pattern inside `_build_variable_walk_occ`).
  - Reinstatements (both faces): `Aggregate.reinstatement_analysis`, `_aggregate.py:1214` — the joint rides into the analysis and thence `build_reinstatement_pnl`.
- **The joint itself**: `Aggregate.occ_bivariate` (`_aggregate.py:1028`) → `build_netceded_joint` (`bivariate.py:784`): per-claim severity mass placed at the exact 2-D point `(x, c(x))`, then `iFFT2(freq_pgf(FFT2(S)))` — the univariate compound algorithm with the transforms swapped to 2-D, `padding=1`. It's fast because it's *one* FFT2 on a ~2²⁰-cell grid — the same reason the 1-D engine is fast.
- **Kernel consumption**: `PnL.__init__` per-atom route (`_pnl.py:807`ff) via `_source_atoms` (flattens the joint's cells into shared atoms + probabilities); the κ ladder is `_scenario_ladder` (`_pnl.py:~1415`).

## What makes the 2-D trustable

Three layers — runtime guards, one-line cross-checks, and one-time adjudication:

**Runtime, every build:**

1. **Mass/deficit bookkeeping** — `build_netceded_joint` computes `deficit = 1 − Σ density` (`bivariate.py:875`) and carries it in the joint's `meta`, same convention as the 1-D engine's clipped-tail reporting. `padding=1` means no FFT wrap-around.
2. **Structural correctness of the dependence** — mass *cannot* appear off the feasible region: each claim's atom sits exactly on the cession curve, and the one frequency PGF wraps the 2-D severity transform, so the shared claim count *and* shared mixing are exact by construction, not approximated.
3. **`CoarseJointGridWarning`** (`reinstatement.py`, `check_joint_grid_adequacy`) — fires when a treaty kink region spans <20 joint buckets, the one error mode the internal bookkeeping can't see (Jensen-type O(bs) bias on kinked maps).
4. **Theory-vs-realised audit on demand** — the returned holder is a `BivariateAggregate`; its `summary_df` (`bivariate.py:2069`) is the Portfolio-shape validation frame (exact moments from `_netceded_theory` at :1730 vs realised, with noise-aware `Err` columns).

**One line away, any time:** the `pnl` face reads the *exact* 1-D marginals, so `pnl.mean` vs the walk's `('All','Margin','Total')` EX bounds the joint-grid error for your actual program; any single walk row vs its `reins_density_df` marginal mean does the same per row. The tests do exactly this (rel ~1e-3 linear rows, ~1e-2 kinked agg tiers).

**One-time adjudication (2026-07-05, now pinned in tests):** joint mass 1 − 3e-11; *zero* off-support mass; joint marginal means = engine means to ~1e-5 rel; internal accounting identities to 1e-15 (`ReinstatementAnalysis.validation_df`); and a 1.6M-simulation Monte Carlo of the true process matching every leg within 2 SE, with the kinked-layer grid bias isolated and quantified (~0.25% at 25 buckets/layer — the origin of the 20-bucket warning floor). Tests: `test_pnl_consolidated_walk.py`, `test_composition_matrix.py`, `test_reinstatement_decl.py`.

**One honest gap to note:** the reinstatement route ships its audit on the object (`pnl.analysis.validation_df`); a GC walk currently has no attached equivalent — its runtime guards are the deficit bookkeeping and the adequacy warning, and the exact-vs-realised comparison lives in the tests and the `biv.summary_df` you'd have to call yourself (the walk doesn't retain the holder; only `pnl._source` = the joint and `pnl.engine` = the Aggregate). If you want a `validation_df` on joint-sourced walks — each row's realised EX against the engine's exact marginal mean, the per-program version of what the tests assert — that's a small, well-defined addition. Say the word and it goes on the list.
