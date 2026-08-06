# Residue from the chapter 2 pipeline pages

Recorded 2026-08-06, during the five-C review of chapter 2
(`dev/doc-review-instructions.md`).

The four `docs/2_aggregate_overview/pipeline-*.rst` pages carried planning
apparatus: `Recommendations`, `Fix / reconcile`, `Optimise`, `Remove`, `Open
questions / defer`, `DECIDED (2026-05-28)` blocks, source line numbers, an
`Open items` list, and a proposed before/after consistency harness. All of it
has been stripped from the published docs, which now describe how the pipeline
works and nothing else.

Almost all of it was a **completed** backlog, verified against `src/aggregate`
before deletion:

| Item | Status |
|:--|:--|
| `aggregate_keys` dead class attribute | gone |
| Journey-of-discovery comment blocks in source | gone |
| `add_exa_details` / `add_eta_mu` / the EPD and eta-mu column family | gone |
| `loss_max` / `mult in {1,10,100}` blanking heuristic | replaced by an explicit `VALIDATION_NOISE` denominator guard, `_portfolio_density.py` |
| `_build_augmented` duplication and the `gprime1` fallback disagreement | gone with the single Choquet engine (a57) |
| `_write_empirical_stats` sev-block inversion | gone |
| Empirical-moment convention, Portfolio vs Aggregate | unified on `xsden_to_mwrangler` |
| `pricing_at` column order | now `pentagon.PENTAGON_STATS` |
| Default natural allocation | `linear` (a17); `lifted` remains an option |
| Pandas copy-on-write | enabled in `__init__.py` |
| The before/after consistency harness | built: `tests/test_baseline.py` plus `tests/baseline/` |

Three things were still live at the time of the review and are the only reason
this file exists.

- **`Portfolio.swap_density_df`** is still marked EXPERIMENTAL, and the question
  the old page raised is still open: promote it to a supported sample entry
  point, or drop it. `_portfolio.py:3880`, `_portfolio_sample.py:134`.
- **`pricing_at` still defaults to `allocation='lifted'`** (`_portfolio.py:3148`)
  while `price` defaults to `allocation_method`, which is `linear`. The two
  readouts disagree on their default unless the caller is explicit. Worth
  reconciling.
- **LEV tail-mass convention.** Whether `density_df.lev` and the empirical-moment
  tail-mass placement agree is unresolved, and belongs with the `moments.py`
  `xsden_to_meancv` vs `xsden_to_meancvskew` question.

The `PnL` page's `Open items` block was deleted outright: all eight labels
(`[Walk-Validation-DF]`, `[Massive-Kappa-Second-Sweep]`,
`[Consolidated-LAE-Off-Source]`, `[Peel-Aggregate-Tier-Only]`,
`[Ratio-Distribution]`, `[Accounting-Summary-DF]`,
`[PnL-Density-DF-Running-Nets]`, `[Portfolio-of-PnL]`) are already carried in
`dev/TODO.md`.
