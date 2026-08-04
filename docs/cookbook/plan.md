# [Cookbook] — a runnable feature cookbook for `aggregate`

Renders from ``library.agg`` ``doc{{...}}`` features.

## [Cookbook-in-Docs] — Docs Cookbook

The Cookbook in Docs is auto generated from ``Doc{{...}}`` strings in the master `library.agg` file.

Each snippet can contain a check is where "trust the tests" becomes "see it yourself." Six archetypes; each page is tagged with the ones it uses, as an HTML comment inside the callout.

- **[Check-Reconciliation]** things add up: `net + ceded = gross`; walk means foot down the sheet; allocations sum to the total; consideration = `gross − ceded premium + commission`.
- **[Check-Scaling-Sweep]** vary one knob, watch the output move as predicted — a small table or a line. The placement page *is* this: sweep the share 25/50/75/100% and see ceded premium trace a straight line.
- **[Check-Independent-Oracle]** `est_*` (empirical/FFT) vs theoretic moments; or a quick Monte Carlo for the exotic terms with no closed form.
- **[Check-Limiting-Case]** sanity bounds: `TVaR ≥ VaR`; premium ≥ EL; a distortion has `g(0)=0, g(1)=1`, concave; a swing collar clamps at min/max; zero placement ⇒ gross.
- **[Check-Round-Trip]** `format_program(build(x))` round-trips; `pnl` consolidated == `xpnl` walk to grid accuracy; two equivalent spellings agree (`rol` vs the matching `deposit`).
- **[Check-Cross-Object]** an aggregate priced alone == its allocation in a one-unit portfolio; bivariate marginals recover the univariates.
