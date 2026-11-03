Bugs found writing the `Using` Series
=====================================

1. Wrong analytic moments for a shifted severity: for sev 100 * gamma 2 + 10000 the analytic skewness is 1.387 against an exact 1.414. The grid matches the exact value, so validation fails a discretization that is correct.
2. help(values=...) runs methods: any setting other than 'none' calls every matching method that takes no arguments (utilities.py:1022). On an Underwriter that includes write_default_config() and reload().
3. Sharpen's defective-grid gate is at rounding level (1e-12): it rejects better grids. On a lognormal book with CV 1.5 it moved from bs 1 to the coarser bs 4. One sentence in sharpen_explanation also gives the wrong threshold.
4. Ambiguous names give a misleading parse error: a bare name used under two kinds becomes "Unexpected 'Mack2003.Lognorm'". reproductions.agg and library.agg share 11 names across kinds, which contradicts the error's own "Shipped library names are unique across kinds".
5. Frequency.__repr__ raises: for a dfreq or renewal count with more than one outcome it does if self.freq_a: on an array. A bare a.frequency at the end of a cell raises.
6. Missing guards:
   - poisson zm 1.5 ! builds a pmf with negative masses.
   - pollaczeck_khinchine with ρ < 0 returns ψ > 1 without complaint.
   - wiener_hopf(0) returns a wrong ψ.
7. Grid sizing:
   - log2= without bs is silently ignored for discrete severities.
   - The automatic grid for geometric zt is too short and the result is flagged defective.
   - The "raise log2 to 19" advice fails unless bs is kept.
   - The automatic window for a signed severity cuts off the left tail.
   - A Portfolio mixing light and thick units fails validation on its default grid.
8. Smaller items:
   - An inline vector of about 500 numbers raises RecursionError.
   - q(0.99) misses when 0.99 falls exactly on a jump.
   - -- dashes appear in bs_explanation, Portfolio.tail_explanation, PnL.construction_explanation and the IgnoredDecLClauseWarning text.
   - Parse errors list raw terminal names (lpar, lsqb, bang).
   - PnL.tvar's docstring says lower tail; the code returns the upper.
   - sev lognorml gives a bare AttributeError with no suggestion.


**Library findings.** The pages avoid all of these and none of them mention the bugs. I reproduced the first two myself; the rest are the writers' reports with reproductions.
1. **`PricingBounds` understates the upper bound.** In a small example the library gives [1.125, 1.125] where the correct range is [1.125, 1.375]. The standard `port.pricing_bounds({'A': port.A}, p=...)` with a finite cap is always affected. The cause is that only vertices where TVaR of X strictly increases are kept (`bounds.py`, about lines 1701 to 1708).
2. **`convex_distortion` adds a false mass at zero.** On the points (0,0), (½,0.8), (1,1) it returns a mass of 0.3 and g(¼) = 0.55; the correct value is 0.4 with no mass. `Bounds.min_envelope`, `bagged_distortion` and the EL/ROL distortions all call it.
3. **`cede` disappears when `pc` is on the same layer.** The result is identical to `pc` alone. It should either be booked or refused, the way `slide` with `cede` is.
4. **`Aggregate.price`:**
   - `p >= 1` silently gives the wrong assets instead of being read as a capital amount.
   - `ccoc` is refused on an unbounded book even at a finite `p`.
   - The docstring says an Aggregate cannot calibrate, which is no longer true.
5. **Bivariate:**
   - A component's limit is ignored when sizing the axis grid, and the resulting warning is misleading.
   - Pinning `bs` together with a per-axis `log2` can lose nearly all the mass.
   - A ragged `dbvsev` matrix gives a raw numpy error.
   - Refusing `grossnet` on an aggregate with no occurrence reinsurance names `netceded` in the message.
6. **Withdrawn 2026-11-01, not a defect: the structure chart's occurrence return-period axis counts per claim.** The 250 attachment shows 11.6, one claim in 11.6. Occurrence cover is conditional on a claim, so the axis does not depend on the claim count (author's ruling, `dev/done/plan-using-notes-wrong-answers.md`).
7. **Smaller items:**
   - `cede` is refused with `swing`, which may not be intended, and swing gets a different default step label from a fixed premium.
   - A `po` share above 100% is accepted silently, and the `po` hint is wrong for amounts of 1 or more.
   - `gini_p` is NaN for `bitvar`, `minimum` and `mixture`.
   - The DecL distortion name isn't kept on the object.
   - `Distortion.ccoc(d)` takes the discount where `Distortion('ccoc', r=...)` takes the return.
   - Refusal messages that appear on the pages contain internal text such as "decision 0", `[reins-premium]`, "is a follow-up" and `--`.
   - The docs say the Bounds classes aren't re-exported at the top level, but they are.
   - Two functions in `pedagogy.py` call methods and arguments that no longer exist.

Section 8 of the execute-plan skill should now say that a page takes its tags from the fixed list. That's an edit to a skill, which needs you; I can draft the text. I've recorded the tag list and the `/add-dir` step in memory, so later sessions use both.

On each using page: add appropriate tags and then pull the tags into the index page. that'll make it easier to cross reference.

Then, pls add new Using items. The size limit is an upper bound, not a target. Using labels, for example, should be much shorter! It is also a guideline, if you need a few extra lines, OK. But the average of longer items should be <=150. (Think Winston Churchill on rules).

* Using bivariates (independent, copula, dbvsev flavors)
* Using grossnet (and related)
* Using clash
* Using approximate (and approximation_df)
* Using distortions
* Using price (from an Agg or a Distortion)
* Using price and evaluate (on an Aggregate)
* Using occurrence re (no variable features)
* These should be **shorter** say 50-100 lines - just surgical intros to one specific thing. Use the same base portfolio for all except retro rating (which is different, it is pricing gross). The agg needs embedding in an xpnl. All tagged "variable feature" and reinsurance.
    * Using reinstatements
    * Using sliding scale commissions
    * Using swing rating
    * Using profit commissions
    * Using loss corridors
    * Using retro rating
* Using Bounds
* Using PricingBounds
* Using AllocationBounds
* Using the Structure Chart
* Using oep
* Using labels (the "as ... construction; illustrate on an xpnl)

Then rebuild and reupload.

**Found on the emergence page of Using pmir, 2026-10-30.** The warning `p_total has 0 negative values; NOT setting to zero...` comes from `_portfolio_density.py` lines 160 to 162. The guard fires on any negative value (`not np.all(p_total >= 0)`) but the count only includes values below `-eps`, so FFT rounding noise smaller than machine epsilon gives a warning that reports 0. It is harmless, and the fix is to warn only when the count is positive. The same warning appears when running block 10 of the `lattice-buckets` page.

**Found in the writing pass over the Using pages, 2026-10-30, at 1.43.1.** Library messages shown on the pages contain wording meant for a developer. None gives a wrong number, and the pages show them unchanged.

* **Internal labels.** "decision 0: no stacking" on the `profit-commission` and `swing-rating` refusals; "[reins-premium]", "[reins-single-layer]", and "the engine ceiling" on `reinstatements`.
* **Looking forward.** On `swing-rating`, "only in this release; occurrence-basis variable rating is a follow-up". On `retro-rating`, "not yet supported; retro currently requires".
* **History.** The `tweedie` refusal for a bad power ends "(It read `<mean> <p> <dispersion>` before 1.0.0a231, so an older program needs its first two numbers swapped.)". On `price-evaluate`, "reproduces the old number".
* **Verbs of a person.** On `price-evaluate`, "does not name an asset level" and "tail_behavior_df says what the tail is". On `elt`, "unit rows carry different rates".
* **`--` used as a dash.** In `bs_explanation` and `construction_explanation`, and in messages on `pnls` and `pk-wh`.

**Found on the unconditional layers page of Using aggregate, 2026-11-02, at 1.63.2.** Two small items, neither visible on the page.

* **`qd` of an `Aggregate` raises `Pandas4Warning`.** `utilities.py` line 577 calls `summary_df.fillna('')` on a float frame. Default filters hide it, so no page shows it, but a future pandas raises an error there.
* **`Aggregate.program` doubles some spaces.** `build('agg Eg dfreq [1] 3 xs 0 dsev [0:4] !')` gives `program` equal to `agg Eg dfreq  [1]  3 xs 0 dsev  [0:4]  !`, with two spaces after `dfreq`, `[1]`, `dsev` and `[0:4]`. Cosmetic.
