# Plan: guards, rulings and message text from the Using pages

`[Using-Notes-Guards-And-Messages]`. Drafted 2026-10-31 against aggregate 1.43.1 (`03aaec6`).
Two minor bumps: `[Using-Notes-Guards]` (ten guards and the five code rulings) and
`[Using-Notes-Messages]` (text only, no number moves). Agreed by the author on 2026-10-30 with
rulings on 2026-10-31 (`D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`, item 11). It
runs sixth, after `[Validation-Verdict]`, so its message text follows the verdict's.

## Rulings (author, 2026-10-31)

1. `log2` without `bs` on a discrete severity stays a cap; document it.
2. `ccoc` at a finite `p` is allowed; the refusal stays for `p = 1` and `a = inf`.
3. `cede` with `swing` stays refused, with a proper message, as `slide` with `cede` is.
4. `gini_p` for `bitvar`, `minimum`, `mixture` and the rest: **no ruling received**; the plan
   takes the recommendation, a numeric fallback from $g$, until the author says otherwise.
5. The DecL distortion statement takes an `as` clause: `distortion myd as "my distortion" ph
   0.6`; `name` stays the family, the declared label goes in `label`.
6. `Distortion.ccoc(d)` and `Distortion('ccoc', r=...)`: keyword-only `d=` or `r=`, at 2.0.

One question stays open (step 5 below): whether `Aggregate.price` refuses `p >= 1` as the review
table has it, or both `price` methods gain explicit `p=` and `a=` keywords at 2.0.

## `[Using-Notes-Guards]`

Each item: today, measured 2026-10-31; the change; the test. Line numbers are at 1.43.1.

1. **`zm` outside $[0, 1)$.** `Frequency.__init__` (`_frequency.py` 562 to 605) stores
   `freq_p0` unchecked and installs the wrappers at 601 to 605; `_zm_weight` (242 to 261) is
   negative above 1 and 0 at 1; `solve_base_mean`'s bracket error (339 to 342) is marked
   unreachable. Measured: `zm 1.5` and `zm 1` raise the bracket message; `zm -0.2` builds a
   mass of -0.2 with the verdict `not unreasonable`; `zm 1 !` builds `n = 0` and a `nan` density;
   `zm 1.5 !` builds `n = -1.157`. `zm 0` equals `zt` exactly. *Change:* at 601, inside
   `if freq_zm`, `if not 0 <= freq_p0 < 1: raise ValueError(f'zm: P(N = 0) must lie in [0, 1),
   not {freq_p0!r}')`, the `apply_deductible` template at 383 to 385. *Test:* the five inputs
   above raise, `zm 0` still builds, in `tests/test_frequency_zm.py`.
2. **`help(values=...)` calls nothing.** `agg_help` (`utilities.py` 1033; the walk 1147 to
   1192) runs `getattr(self, name)` on every match at any `values` (1156) and, for
   `values != 'none'`, calls every no-argument method (1184), which includes `update()` and
   `sharpen()`, both with no required argument, so a `help(values='all')` can resize the object.
   *Change:* classify from the class (`class_attr` at 1153: method, property, plain attribute)
   and read an instance value only for a plain attribute, or for a property when `values` asks;
   a method shows its name and signature and is never called. *Test:* a host with a method that
   raises if called passes through `help(values='all')` untouched, in `tests/test_help.py`.
3. **`Frequency.__repr__` and `info` on an array.** `_frequency.py` 671 to 673 and 641 test
   `if self.freq_a:` and format with `:.4g`; `FrequencyEmpirical` (1157 to 1158) stores arrays.
   `dfreq [1 2 3]` raises `ValueError`, `dfreq [3]` raises `TypeError`. *Change:* format an
   array as its list, scalars as today. *Test:* `repr` and `info` of `dfreq [1 2 3]`,
   `dfreq [3]` and a `renewal` count.
4. **`pollaczeck_khinchine` with $\rho \le 0$.** `_aggregate.py` 5483 to 5543 has no check;
   `ruin = 1/(1 + \rho)` everywhere, 1.25 at -0.2, 2 at -0.5, 1 at 0. *Change:* after the
   severity check at 5511, `if rho <= 0: raise ValueError(f'net profit condition requires rho >
   0, got {rho}')`, the `_ruin_paths` message at 5823 to 5825. *Test:* beside
   `tests/test_ruin.py` 82 to 83.
5. **`Aggregate.price` with `p >= 1`.** `_pricing.price` (95 to 146) takes `a_reg` from
   `var_dict(p, kind)`, and `q` lands any `p > 1` on the last bucket (`_grid_distribution.py`
   179 to 187): `p` of 1, 1.5, 100 and 1000 all price at the grid top, while the docstrings at
   `_aggregate.py` 8186 to 8194 and `_pricing.py` 106 to 115 say "if > 1 assets".
   `guard_unbounded_anchor` (`_pricing.py` 29 to 92) is not called here. `Portfolio.price`
   (`_portfolio.py` 4142 to 4148) does read `p > 1` as an asset level. *Change, per the review
   table:* `Aggregate.price` refuses `p >= 1` with the `guard_unbounded_anchor` text for
   `p == 1` and `p must be a probability below 1; pass an asset level through Portfolio.price
   or price(..., a=...)` for larger. *Question for the author:* the alternative is explicit
   `p=` and `a=` keywords on both `price` methods, retiring the dual reading of `p` in
   `Portfolio.price` (used by `tests/test_allocation_gold_standard.py` 126 to 133, `top=12.0`);
   that is a 2.0 change and not this plan's unless ruled. *Test:* `tests/test_unbounded_anchor.py`.
6. **`ccoc` at a finite `p`.** `Aggregate.apply_distortion` (`_aggregate.py` 5466 to 5471)
   refuses any `has_mass` distortion on an unbounded aggregate whatever `p`; siblings at
   `characteristic_function.py` 2855 to 2859 and the Portfolio's `_portfolio_common.py` 146 to
   153. *Change:* refuse only when the anchor is unbounded (`p == 1` or `a` infinite, the
   `guard_unbounded_anchor` reading); at a finite anchor the mass lands at $a$, which is a
   bounded problem. The note at `_reinsurance.py` 769 to 778 and the `analyze_distortions` skip
   (`_portfolio.py` 4547 to 4559) follow. *Test:* `a.price(0.99, Distortion('ccoc', r=0.1))`
   prices; `a.price(1, ...)` still refuses.
7. **`po` above 100%.** `parser.py` 2328 to 2345: a `%` literal is the share and every bare
   number is an amount divided by the limit (`6 po 5 xs 50` gives share 1.2, `150% po 100 xs
   50` gives 1.5, `0.5 po 100 xs 50` gives 0.005 with a warning that it is suspiciously small).
   Neither branch is bounded. The review's premise that numbers of 1 or more are read as
   amounts was wrong: all are. *Change:* `_validate_reins_layers` (`_reinsurance.py` 62 to
   110, the choke point its docstring names) refuses a share above 1: `share {s:.4g} on
   '{layer}' exceeds 100%; a bare number before po is an amount of the limit, a percentage is
   the share`. The stale docs sentence at `features.rst` 708 (`so` and `of` as synonyms,
   retired at a249) is corrected in the text step. *Test:* both spellings raise;
   `tests/test_paren_arithmetic.py` 88 to 98 and `tests/test_underwriter.py` 104 pass.
8. **Bivariate pinned `bs` with per-axis `log2`, discrete mode.** `bivariate.py` 1788 to 1801
   accepts the pin, 1814 to 1829 only warns when it does not cover the window, and
   `_form_severity_matrix` 2201 to 2210 scatters the atoms into `S[np.ix_(ix, iy)]` and raises
   `IndexError: index 20 is out of bounds for axis 0 with size 16` on `dbvsev [10 20 30] [10
   20]` with `bs=1, log2=(4, 4)`; the copula, clash and `dsev`-copula forms clip with a warning
   instead. *Change:* in discrete mode, when an atom index exceeds the axis size, refuse with
   the size needed: `axis 0 needs log2 >= 5 at bs 1 to hold the atom 30; log2 4 was given`
   (the lattice warning's sentence shape from `[Lattice-Grid-Sizing]`). *Test:* beside
   `tests/test_bv_discrete.py` 127.
9. **`sev lognorml`.** `SeverityScipy._build` (`_severity.py` 2856, also 1426 and 1465) does
   `getattr(ss, sev_name)` and lets the `AttributeError` out, although `_classify_sev` (1340 to
   1359) promises a clearer error. *Change:* catch it and raise `ValueError(f'unknown severity
   family {sev_name!r}; did you mean {close}?')` with `get_close_matches` over the names in
   `decl_words._SEVERITIES` plus `scipy.stats`'s continuous names (the `_did_you_mean` helper
   at `parser_errors.py` 692 to 710, cutoff 0.6), and carry the suggestion in
   `ErrorReport.suggestions` when the build came through DecL. *Test:* `lognorml` suggests
   `lognorm`; `gammma` suggests `gamma`.
10. **Unclosed bracket.** `UnderwritingLexer.preprocess` (`parser.py` 354 to 356) asserts an
    odd split count with no message; the nested path (361 to 371) does no balance check.
    `dsev [1 2 fixed` raises `AssertionError ''`. *Change:* a parse error through
    `parser_errors` with the position of the unmatched `[` or `]`, on both paths. *Test:* beside
    `tests/test_decl_parser.py` 40.
11. **`p_total has 0 negative values`.** `_portfolio_density.py` 159 to 162 tests `>= 0`
    strictly and counts below `-cut_eps`. *Change:* warn only when the count is positive.
12. **`gini_p` numeric fallback** (ruling 4 pending). `_common_init` sets `nan` (`spectral.py`
    788) and only `ccoc`, `ph`, `wang`, `dual` and `tvar` overwrite it (2282, 2379, 2483,
    2586, 2685). `bitvar` and `wtdtvar` already have the closed form in `_describe_closed_form`
    (2900 to 2925, 3097 to 3117: `p_equiv`) and do not feed it; `_ensure_grid_moments` (1364
    to 1406) has the trapezoid `p_equiv` on a 101-point grid, measured 0.33293 against the
    exact 0.33333 for PH(0.5). *Change:* `gini_p` becomes a property: the closed form where
    one exists (adding `bitvar` and `wtdtvar`), else $2\int_0^1 g - 1$ by `scipy.integrate.quad`
    on `g`, which gives 0.66 for `bitvar(0.5, 0.9, 0.4)` and 0.50000 for `beta(0.5, 1.5)`. The
    consumers at `_pricing.py` 636 to 638 and 1263 and `info` (1648 to 1652) need no change;
    the base docstring at 1161 is corrected. *Test:* each family's `gini_p` against `quad` to
    1e-8; `test_exhibits` 1980 drops `nan_ok`.
13. **The distortion `as` clause** (ruling 5). `decl.lark` 169 to 171 has no `as_label` slot
    where `agg_out` (196) and `port_out` (177) do; `distortion TDn2 as "My PH" ph 0.5` is
    `Unexpected 'as'`. The transformer (`parser.py` 785 to 796) passes `decl_spec` with
    `name = kind`; `from_spec` (`spectral.py` 697 to 723) passes a `label` key through to
    `__init__(name, shape, *, label=None)` (645), whose docstring already calls `label` "the
    DecL `as` clause"; `_init_labels` at 689 and in the eleven subclasses. *Change:*
    `distortion_out: DISTORTION name as_label ID numberl trailer` and the two combo forms
    likewise; the transformer puts the label in the spec; `name` stays the family. The spec
    snapshot moves for the distortion lines. *Test:* `tests/test_decl_labels.py` 303 to 326
    gains the `as` form; `label` is `"My PH"`, `name` is `ph`, and the writer round-trips it.
14. **`Distortion.ccoc` keyword-only** (ruling 6). The static `ccoc(d)` (`spectral.py` 1762 to
    1769) takes a discount and converts; the class takes `d=` or `r=` (2232, 2256 to 2264).
    *Change:* `Distortion.ccoc(*, d=None, r=None)` with the class's one-of check; the one
    positional caller in `src/` (`random_distortion_ex`, 1854) and the tests
    (`test_distortion_construction.py` 97 to 129, `test_distortion_quartet.py` 144 to 281,
    `test_hygiene4.py` 148, `test_numerics3_distortion.py` 84) write the keyword. A positional
    call raises `TypeError`; the CHANGELOG carries the breaking-change paragraph.
15. **`cede` with `swing`** (ruling 3). `swing` with a premium is refused at `parser.py` 2392
    to 2396 and `cede` without a premium at 2372 to 2375, so `cede 10% swing ...` gets the
    second message and `swing ... cede 10%` a parse error. *Change:* beside the slide refusal
    at 2405: `'swing' supplies the ceded premium and the commission; give it without 'cede'`.
    *Test:* beside `tests/test_variable_rating_decl.py` 271 to 283.
16. **`log2` as a cap** (ruling 1). Documented at `bs_window` (`_bucket_window.py` 911 to 914)
    and in `bucket-selection.rst` 63 and `features.rst` 1731; `Aggregate.update` has a bare
    `:param log2:` (`_aggregate.py` 4503). *Change:* the `update` docstring states the cap and
    that pinning `bs` makes `log2` exact.

**Files.** `src/aggregate/_frequency.py`, `utilities.py`, `_aggregate.py`, `_pricing.py`,
`_portfolio_common.py`, `_reinsurance.py`, `parser.py`, `decl.lark`, `bivariate.py`,
`_severity.py`, `parser_errors.py`, `_portfolio_density.py`, `spectral.py`; tests named above;
`tests/data/expected_specs.json` (regenerated: only distortion lines move);
`docs/4_dec_Language_Reference.rst` (the `as` clause on `distortion`); the grammar reference
regenerated.

**Acceptance.** Every "Test" line above; tier 3; the spec snapshot diff is the distortion lines
and nothing else; `decl-testers.agg` gains the refused programs commented and the `as` form in
their sections.

## `[Using-Notes-Messages]`

No number moves. Every string with its anchor, from the report of 2026-10-31:

- **Dashes.** `bs_explain` (`_bucket_window.py` 507, 511); `Portfolio.tail_explanation`
  (`_portfolio.py` 802, 805); the P&L construction explanation in `_pnl_builders.py` at 556,
  558, 560, 648, 650, 659, 663, 665, 793, 805, 807, 812, 821, 1014, 1053, 1648, 1650, 1660,
  1665, 1670, 1798, 1801, 1804; `IgnoredDecLClauseWarning` (`underwriter.py` 151 to 152 and
  the class docstring at `constants.py` 330). Rewritten with a colon, a comma or a full stop,
  per the house rule.
- **Internal labels.** `decision 0: no stacking` (`parser.py` 2181); `[reins-premium]` (2385);
  `[reins-single-layer]` (2137, 2198; `_pnl_builders.py` 2088); `the engine ceiling` (2200,
  2217 to 2218; `_pnl_builders.py` 2090). Each refusal gives what is refused and what to write
  instead, with the label kept only in the comment beside it. `tests/test_reinstatement_decl.py`
  156 and 163 match on the labels and are updated.
- **Looking forward or back.** `is a follow-up` (2175 to 2176), `not yet supported`
  (`underwriter.py` 1879), `before 1.0.0a231` (`parser.py` 999 to 1001), `reproduces the old
  number` (`_pricing.py` 75, 90). Each becomes a statement of what the clause does now.
- **Verbs of a person.** `does not name an asset level` and `tail_behavior_df says what the tail
  is` (`_pricing.py` 87, 92; `tests/test_unbounded_anchor.py` 45, 55 updated); `unit rows carry
  different rates` (`_loss_table.py` 886 to 887; `tests/test_elt.py` 240).
- **`grossnet` refusal.** `bivariate.py` 1089 to 1094 and 1426 say `netceded requires
  occurrence reinsurance` for `grossnet` and `grossceded` too: the message takes the view's own
  word.
- **Docstrings and docs.** `PnL.tvar` (`_pnl.py` 1726 to 1738) gives the low tail; the accessor
  reads the upper tail (payoff P&L, mean 200, `q(.99)` 819.25, `tvar(.99)` 866.67): the
  docstring is corrected to what the code does. `Aggregate.price` has `cannot calibrate`
  (`_aggregate.py` 8178 to 8179, `_pricing.py` 98 to 99, module note 11 to 12);
  `calibrate_distortions` exists at 8301: corrected. `docs/3_reference/3_x_Bounds.rst` 18 to
  19 has the Bounds classes as not re-exported; `aggregate/__init__.py` 37 re-exports all
  three: corrected. `features.rst` 708 on `so` and `of`: corrected.

**Files.** The sources above, their tests, `docs/`. Docs are pending a rebuild by the author.

**Acceptance.** `rg -- ' -- ' src/aggregate` finds no message text (code and DecL `--` excluded);
the listed test matches updated; tier 2.

## Not in this plan

- Booking `cede` beside `pc` (refused in `[Using-Notes-Wrong-Answers]`).
- Routing suggestions through `ErrorReport` for every refusal (the `[Repair-Link]` idea, pended).

## Execution log

Executed 2026-11-01 from 1.58.0. Item numbers below are those of `[Using-Notes-Guards]`.

### Rulings at execution (author, 2026-11-01)

- **Item 5, the open question:** explicit `a=` on both `price` methods. `Aggregate.price`, `AggregateCHF.price` and `Portfolio.price` take exactly one of `p` and `a`; `p > 1` raises on all three, pointing to `a=`; `p = 1` stays valid on a bounded risk. Measured before the ruling: only `tests/test_allocation_gold_standard.py` and the Using page `allocation-bounds.qmd` passed an asset level as `p`, and nothing in `src` or the API.
- **Item 6:** a direct `apply_distortion` keeps its refusal, since it has no asset level; `price` checks its own anchor and computes through the private `_set_distortion_columns`.
- **Item 12, ruling 4 that was never received:** the numeric fallback, as the plan defaulted.

### `[Using-Notes-Guards]` divergences, 1.59.0

- **Item 1.** `zm 1 !` raised `cannot convert float NaN to integer` at 1.58.0 where the plan measured a build with a `nan` density. The guard is unchanged.
- **Item 5.** The plan's message pointed to `price(..., a=...)`, which existed on neither method; the ruling makes it true. New function `_pricing.check_price_anchor`, shared by all three `price` methods. Beyond the plan: `p = 1` on an unbounded `Portfolio.price` now refuses, an infinite `a` refuses, and an `a` above the top of an unbounded risk's grid refuses, because it would snap to the top bucket and reproduce the failure the guard exists to stop.
- **Item 6.** The plan's change could not be made inside `apply_distortion`, which has no `p`. `test_mass_guard_in_builder` pins the direct refusal and passes unchanged. The Portfolio needed nothing: linear allocation already prices `ccoc` at a finite `p`, the lifted refusal concerns the whole frame's right edge and stays, and the `analyze_distortions` skip and the note in `_reinsurance.py` were already narrowed to lifted and to `a = inf`.
- **Item 7.** The comparison carries the function's `tol`, so a share computed as 1 + 1e-16 is not refused. The corpora hold no layer share above 1; a regular expression found two, both `corridor ... po`, a separate clause.
- **Item 9.** `decl_words._SEVERITIES` holds names `scipy.stats` lacks, so validity is `is_severity_family` (`cantor` or a continuous `scipy.stats` family) and the word list feeds only the suggestions. Through DecL the refusal is raised while parsing, so a typo gets the located error with `err.report.suggestions`, by a `ValueError` subclass the transformer raises and `UnderwritingParser.parse` locates: the transformer holds no source text. `_did_you_mean` filters Lark terminal names and could not take this list, so `get_close_matches` is called directly at its cutoff. Other direct calls keep the plain `unknown severity family` `ValueError` from `_scipy_family`.
- **Item 10.** A closing bracket with nothing open used to pass silently when the split count stayed odd; it is now refused. New `parser_errors.located_parse_error`. The position counts in the source as written, past comments and literals.
- **Item 12.** The closed form is `_describe_closed_form()['p_equiv']`, the existing single source, so the five `self.gini_p =` assignments in `_build` and the one in `_common_init` are gone. **Finding:** `LEPDistortion`'s closed form ignored the cap of `g` at 1, which binds for every `spread > 0`, so its `mean`, `p_equiv` and `loading` were wrong and a test against `quad` failed: 0.10869 against 0.10866 for `r0 = 0.03`, `r = 0.15`. Corrected in closed form, and the test now passes to 1e-8. Its closed-form `var`, `std` and `cv` shared the error and are dropped, which leaves them blank. `test_hygiene4::test_distortion_info_values` pinned `gini_p n/a` for `wtdtvar` and now pins 0.807; `docs/2_aggregate_overview/info-strings.rst` follows.
- **Item 13.** A distortion with no `as` has no `label` key in its spec, so the snapshot does not move at all, where the plan expected the distortion lines to. The grammar listing moved on the distortion lines only.
- **Item 14.** Only `spectral.py` and `tests/test_distortion_construction.py` called `Distortion.ccoc`. The other three test files the plan listed use `Distortion('ccoc', r=...)` and did not change. The plan did not list the Using page `distortions.qmd`, which called it.
- **Item 15.** The refusal goes ahead of the `cede` needs a premium check, not beside the slide refusal, because that check fires first when no premium is written. It is same-layer only: a `cede` on another layer beside `swing` is booked (`est_m` 2971.86 to 3021.86), which a test pins.
- **Item 3.** An array longer than six values is cut with `...`, so a long `dfreq` keeps `repr` to one line.
- **Item 2.** `cached_property` counts as a property, or `Bounds` would run its cached computations under the default listing.
- **Not applicable.** `dev/TODO.md` has no entry for this plan, and `V:/worktrees/dev-files.md` no longer exists.

### `[Using-Notes-Messages]` divergences, 1.60.0

- **Dashes, wider than the list.** The acceptance check covers all message text, so all of it was cleaned, not only the listed anchors: 64 non-docstring strings across 20 modules by a scan of the syntax tree (`ast.Constant` outside docstrings), which is now zero. A plain `rg ' -- '` cannot separate a message from a docstring: the rest of its 1,160 matches at 1.58.0 are docstrings and comments, which stay. The file written by `Underwriter`'s `# added ...` header line and the `skipping %s` log line were in the scan.
- **Also corrected in passing, because the sentence was being rewritten:** `the reference asks the object` in `_severity.py` and `the conversion reads the distribution` in `_aggregate.py` and `_portfolio.py`.
- **Labels.** `[reins-one-clause]` and the `aad` / `agglimit` copy of `the engine ceiling` were not listed and went the same way. The refusal's remedy for a tower is `Write one occurrence layer`, where the plan left the wording open. Each label stays in a comment beside its refusal. `tests/test_reinstatement_decl.py` matches the new text and gains `test_a_refusal_holds_no_internal_label`.
- **`is a follow-up`.** The two comments stating the same thing, in `parser.py` and `underwriter.py`, went with the message.
- **`Aggregate.price` docstring** (`cannot calibrate`) was rewritten at 1.59.0 with its signature; only the `_pricing` module note was left for this bump.
- **`grossnet` refusal.** New private `bivariate._joint_keyword`; the three messages take it, and a test covers the three keywords.
- **Exhibit snapshot.** The `bs_window` note text is pinned in four entries of `tests/data/exhibit_snapshots.json`, which the plan did not list. `capture_exhibit_snapshots.py` moved those four and also twelve `raw` values in the last bit, which `_assert_snapshot_close` tolerates by design, so the file was restored and only the text edited.
- **Left alone, for the author.** Bracketed design labels remain in the P&L construction explanations, such as `[Decision-Kappa-Shared-Source-Rule]` and `[Layer-Peeling-Shorthand]`. They are explanation text rather than refusals and the plan's list stops at the four refusal labels, but a reader sees them.

### `[Gini-P-Negative-Return]`, 1.60.1: a regression from 1.59.0, found by the API's suite

The plan's `gini_p` property routes the closed form through `_describe_closed_form`, which also computes moments. A premium below expected loss calibrates `ccoc` to `r < 0`, so `d < 0`, `g` is not a distortion, the variance formula goes negative, and `var ** 0.5` on a Python float returns a complex number, which `std > 0` then refuses with a `TypeError`. The old attribute was just `d`, so this path worked until 1.59.0. Tiers 2 and 3 and the `RuntimeWarning` gate were all green, because no library test calibrated to a negative `coc`; `aggregate-api`'s `test_a_gross_premium_target_calibrates_its_technical_premium` did. The fix is `_root_of_variance`, used at all seven variance roots in `spectral.py`, with tests on the constructor and through `calibrate_distortions`. A sweep of 44 shapes for the five families, in and out of range, found `gini_p` equal to the 1.58.0 assignment everywhere. `summary_df` still raises on a negative `ph` or `dual` shape, as it did at 1.58.0, from `g(0)` with a negative power; that shape is not a distortion and is left alone, while `stats_df` on it now works where 1.58.0 raised.

The new calibration test then failed the `RuntimeWarning` gate on `DualDistortion.calibrate`, which raises 0 to a negative power on its way to a failed solve. The warning and the table are the same at 1.58.0, so it is an existing finding that no test had reached, not a regression. The solve's arithmetic is guarded with `np.errstate` and a `Notes` paragraph, as `test-tiers` directs for a result that is discarded, and the `nan` row does not move.

### Ripple to the API

Measured on `aggregate-api`'s suite against 1.60.0, run read-only: 761 passed, 3 failed. One failure was the regression above, cleared by 1.60.1. The other two are this ripple, and the API owes both, since this plan does not edit that repository.

- `aggregate_api/routes/objects.py` `split_program` catches `AssertionError` from `preprocess` and raises `unbalanced [ ] in this program`; its docstring has the catch going when the library raises a `ValueError` with an `ErrorReport`, which 1.59.0 does. The catch is now dead code. The library's `ValueError` takes the same route out of `split_program`, so the build route still returns 422 (`except ValueError` at `objects.py` 1765), now with the position.
- `tests/test_multi_statement.py::test_an_unbalanced_bracket_is_a_422_not_a_500` pins the `AssertionError` and `unbalanced`, and `test_collapse_never_joins_statements` pins `unbalanced`.

No grammar keyword is new, so `decl-keywords.json` does not move.

### Using pages

No new page: the plan changes existing surfaces, so ten pages in `D:/Projects/Using/aggregate` change, each run and rendered with its sentences checked against the output, and nothing is uploaded: `distortions` (`as`, keyword-only `ccoc`, `gini_p` for every family), `help` (`values` calls nothing, the located errors for a family and a bracket), `price` (`a=`, `ccoc` at a finite level, the refusals), `allocation-bounds` (`port.price(a=a)`), `pk-wh`, `occurrence-re`, `swing-rating`, `severity-parameters`, `bivariates` and `zm-zt` (their refusals).

Pages whose source did not change hold the old message text in their last render, among them `reinstatements`, `grossnet` and the variable-feature pages, and they will pick up the new text at their next forced render. They were left alone because a forced render would also bring the seven `ValidationWarning` pages named at the end of `plan-validation-verdict.md` to a head, which is the author's call.

On the pinned `dbvsev` refusal the library first warns that the grid is short, from `_size_axes`, and then refuses, from `_form_severity_matrix`. It is harmless and left as it is; the `bivariates` page silences the warning in its refused block.
