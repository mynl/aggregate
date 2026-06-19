# Changelog

## 1.0.0a81

### Rename portfolio sub-component `line` → `unit` — **breaking**

The oldest naming wart in the library is gone. *Pricing Insurance Risk*
(Mildenhall & Major, 2022) settled on **unit** as the generic term for a
portfolio sub-component (a line of business, geography, segment, account, or
reinsurance layer all read naturally as a "unit"). The half-renamed `unit_* →
line_*` pass-throughs are deleted and `unit` is now canonical throughout; the
`line_*` names are removed outright (no deprecation alias — this is a pre-release).

- **`Portfolio` storage / accessors** `line_names`, `line_names_ex`,
  `line_name_pipe`, `line_renamer` → **`unit_names`, `unit_names_ex`,
  `unit_name_pipe`, `unit_renamer`**. Accessing `line_names` now raises
  `AttributeError`. The thin `unit_names`/`unit_names_ex` pass-through properties
  were deleted (the names now belong to the real attributes); `n_units` stays.
- **Output-frame index/column label** `name='line'` → **`name='unit'`** on every
  pricing/quantile frame (`pricing_at`, `pentagon_at`, `price`, `var_dict`,
  `Pentagon.as_frame`, the single-`Aggregate` price frame). Any downstream
  `groupby('line')` / `.loc['line']` / `index.name == 'line'` must move to
  `'unit'`.
- **Keyword arguments** `line=` / `lines=` → **`unit=` / `units=`**:
  `Portfolio.pentagon_at(unit='total')`, `Bounds(unit='total')`,
  `Pentagon.as_frame(unit=)` / `from_row(unit=)`, and the private
  `_line_capital_at` → `_unit_capital_at(units=)`.
- **`BivariateAggregate`** follows suit: ctor `lines=` → `units=`, attributes
  `line_names`/`lines`/`_line_specs` → `unit_names`/`units`/`_unit_specs`, and the
  internal DecL spec key `'lines'` → `'units'` (parser producer + `decl_writer`
  round-trip moved together).
- Swept `pedagogy.py`, `results.py`, `bounds.py`, `pentagon.py`, and the test
  suite. Matplotlib (`linewidth`, `ax.lines`), plotly line specs, the actuarial
  *rate-on-line* / *loss on line* terms, optimisation *line search*, and
  source-text *line* machinery are deliberately untouched.
- Docs reference no renamed symbols, so no `:attr:`/`:meth:` cross-refs broke;
  LOB prose in `docs/` ("line of business" → "unit") is a pending author doc pass.

## 1.0.0a80

### Rename multivariate → bivariate (MV-7) — **breaking**

The final stage of the bivariate firm-up: the code is, and will remain, strictly
two-axis, so the public surface now says so. For three or more correlated lines
the path is independent components coupled by Iman–Conover and read back as a
sample (the "switcheroo"), not a native shared-frequency `rfftn` convolution.

- **DecL keyword** `multivariate` / `mv` → **`bivariate` / `bv`**, dropped
  outright (no deprecation alias). The old words now raise a parse error with a
  "Did you mean: bivariate?" suggestion.
- **Class** `MultivariateAggregate` → **`BivariateAggregate`**;
  `__repr__` / `info` say *bivariate*. `BivariateDistribution` keeps its name.
- **Module** `aggregate.multivariate` → **`aggregate.bivariate`** (reach it as
  `from aggregate.bivariate import BivariateAggregate`); internal transformer
  kind string `mvagg` → `bvagg`; grammar rules `mv_out`/`mv_body` → `bv_out`/
  `bv_body`.
- **Config** section `[multivariate]` → **`[bivariate]`**; `MultivariateSettings`
  → `BivariateSettings`; `get_settings().multivariate` → `.bivariate`.
- Swept `decl-testers.agg` / `cookbook.agg` / `examples.agg`, the reinsurance
  user guide, `dev/info-strings.rst`, the DecL cheat sheet, the Sublime syntax,
  and the regenerated grammar reference. The `5_x_multivariate.rst` technical
  guide is **unchanged** — it documents genuinely multivariate (t-dimensional)
  *frequency* theory, not the bivariate aggregate class.
- `BivariateDistribution` and the `occ_bivariate` engine were already named
  bivariate; no change there.

This is the last stage before the `1.0.0b1` candidate: the beta's public API
*is* the v1.0 bivariate API.

## 1.0.0a79

### Bivariate modelling features: shuffle-of-Min copula + clash statement (MV-6)

Stage MV-6 of the bivariate firm-up — the two modelling wins.

- **Shuffle-of-Min copula** (`aggregate.copula.ShuffleOfMin` +
  `CopulaShuffle`). A singular copula built by cutting the unit square into `n`
  equal vertical strips, permuting them, and optionally reflecting some — the
  graph of a measure-preserving bijection. Shuffles of Min are *dense* in the
  space of copulas, so they double as a flexible non-parametric dependence
  stress-test. **Programmatic-only** (no DecL keyword): build it as
  `CopulaShuffle(perm=[...], flip=[...])` and hand it to a bivariate
  (`mv.copula = CopulaShuffle(...); mv.update()`). Its `cdf` is exactly the
  `Copula.rectangle_pmf` interface, so marginals reproduce like any copula.
  Kendall's `tau` is computed **exactly** from the permutation/flips (`n=1`
  recovers M, `tau=1`, or W, `tau=-1`); a `sample` method gives exact draws.
- **`clash` statement** — `clash NAME na nb nc claims <A limit+sev> <B limit+sev>
  <freq>`. The natural cat-clash baseline: a shared event drives two perils, each
  triggered by an independent per-event Bernoulli, and `nc` is the expected count
  of joint-trigger (clash) events. `solve_clash_model(na, nb, nc)`
  (`aggregate.multivariate`) closes the independent-trigger 2×2 table
  (`n0 = na·nb/nc`) to derive the shared event count `n` and the two triggers
  `pa = (na+nc)/n`, `pb = (nb+nc)/n`; the two components become
  `dfreq [0 1] [1-p p]` factories under the **independent** copula on the shared
  frequency. New `CLASH.2` terminal (+ `ID` exclusion); round-trips through the
  DecL unparser as the `clash` form.

Note: a clash with **heavy** components (e.g. cv 2–3 lognormals) at a **large**
shared count can exceed the 2-D memory budget — one common `bs` per axis must
resolve both the severity and the much wider aggregate, so a coarse `bs` can
under-resolve the severity. This is the existing budget tension (not a clash
bug); the MV-4 `validation` row flags it (`check: marginal mean`). Raise
`update(log2=…)` or use lighter/bounded severities.

## 1.0.0a78

### Netceded view-pairs (MV-5)

Stage MV-5 of the bivariate firm-up: the occurrence netceded decomposition
generalises from the single `(ceded, net)` pair to **any two of {gross, ceded,
net}**. The three views satisfy `ceded + net = gross`, so exactly three
unordered pairs exist, each named by one DecL prefix.

- **Two new DecL prefixes** — `grossceded` and `grossnet`, siblings of the
  existing `netceded` (priority-2 terminals with the word-boundary lookahead,
  added to the `ID` exclusion list). Each takes one ordinary `agg` carrying
  occurrence reinsurance and builds the joint per-occurrence aggregate of the
  named pair.
- **`Aggregate.occ_bivariate(views=…)`** — grows a `views=('net', 'ceded')`
  parameter (each entry one of `'gross'` / `'ceded'` / `'net'`); the override
  signature is now `occ_bivariate(views, bs, log2_x, log2_y)` (one common `bs`
  + per-axis log2), replacing the view-named `bs_ceded`/`bs_net`/`log2_ceded`/
  `log2_net`.
- **Axis-order convention fixed.** The keyword names the pair **x-then-y**, so
  `netceded` → axis 0 = Net, axis 1 = Ceded; `grossceded` → (Gross, Ceded);
  `grossnet` → (Gross, Net). **Breaking:** the previous `netceded` /
  `occ_bivariate` axis 0 was Ceded; it is now Net (gross leads when present).
- **`build_netceded_joint(views=…)`** parameterised by the view pair — `gross`
  uses the identity image map (the gross loss itself), `ceded`/`net` use
  `occ_ceder`/`occ_netter`; all three rebucket onto the common-`bs` grid via the
  linear scatter (mean-preserving). `_netceded_theory` reads the matching
  `Gross` / `Ceded` / `Net` occurrence columns of `reins_stats_df`. `info`,
  `describe`, axis labels, and the contour plot follow the chosen pair.

The validation invariant holds for every pair: each marginal reproduces the
named standalone occurrence aggregate (means exact), and `gross − net` recovers
`ceded`.

Also: docs / syntax artifacts updated for the new keywords (the language
reference `ref_include.rst`, the reinsurance user guide, `dev/info-strings.rst`,
the Sublime syntax, `decl-testers.agg`), and a latent path bug in
`parser.grammar(add_to_doc=True)` fixed (it wrote `ref_include.rst` to
`src/docs/` instead of the repo-root `docs/` under the `src/` layout).

## 1.0.0a77

### Bivariate reporting surface (MV-4)

Stage MV-4 of the bivariate firm-up: the `MultivariateAggregate` reporting now
reads like `Aggregate` / `Portfolio`.

- **`info` rebuilt to the shared catalogue.** Dropped the bespoke f-string blob
  for the `aggregate.constants.info_row` / `INFO_NA` convention: a fixed row
  catalogue, every row always present in the same order, `n/a` before `update`.
  Mirrors the Agg/Port layout (name, mode, components, copula, shared frequency,
  claim count, padding, a per-axis block — name/kind/bs/log2/x_min/x_max — then
  correlation, copula tau, tail deficit, validation, id). Documented in a new
  **Bivariate** section of `dev/info-strings.rst`.
- **`explain`** (new) — the showpiece invariant: per-axis marginal moments
  (theoretical vs realized `mean`/`cv` with relative error) and the joint tail
  deficit. The `info` `validation` row is its one-line headline (deficit gate +
  a loose marginal-mean sanity; the exact, resolution-dependent errors live in
  `explain`).
- **`bs_window_df` / `bs_description`** (new) — the realized per-axis grid
  (`kind`, `bs`, `log2`, window, `clipped`), a two-row summary (the bv *measures*
  its grid, so there is no 1-D method ladder to report).
- **`tail_df` / `tail_description`** (new) — per-axis realized support + moments
  + a `right_heavy` flag.

`describe` / `stats_df` keep their per-component moment block (theoretical vs
empirical) and joint dependence footer (correlation, copula tau).

## 1.0.0a76

### Netceded axis sizing routes through `balanced_window` (MV-3)

Stage MV-3 of the bivariate firm-up. The netceded ``(ceded, net)`` joint is now
sized by the same *measure-don't-guess* primitive as the copula axes — the
second private sizer is gone, so both regimes share one path.

- **Deleted `multivariate.size_axis`** (the per-axis moment-quantile guesser,
  `cap_log2=14`). Netceded axes are sized by `balanced_window` on the realized
  occurrence margins (`reins_density_df['p_agg_ceded_occ']` / `['p_agg_net_occ']`,
  produced in one gross pass) — pure window selection, ceded/net being
  non-negative so the grids are 0-based.
- **One common `bs`, sized from the budget.** The comonotone `(c, n)` curve
  couples the axes, so they share a single `bs`; the linear scatter rebuckets
  the gross-sampled points onto it. The common `bs` is sized to fit
  `2**total_log2` (`~ sqrt(hi_c·hi_n)/2**(total_log2/2)`), **not pinned to the
  gross `bs`** — the gross grid auto-sizes fine to resolve the cession layer
  (e.g. 0.125), which is far too fine for the 2-D grid (it blew the budget and
  lost ~55% of the mass). Sizing from the budget also makes `Aggregate.occ_bivariate`
  and the DecL `netceded` form agree regardless of the gross grid each was built
  on. When a caller pins `bs`/`log2` past the budget the wider axis is clipped
  (a warned, reported deficit).

Both private sizers (`_size_axis`, `size_axis`) are now gone; both bivariate
regimes route through `balanced_window`.

## 1.0.0a75

### Bivariate windowing: honest, centred axis measurement

Three fixes so a symmetric axis windows symmetrically (motivating case: the
mean-0 `ssev` axis of `DISCRETE.2`, which came out as `[-29, +99]`).

- **Measure each marginal on an honest grid, decoupled from the loss/payoff
  trim.** The 1-D sizer protects one tail and trims the other (a *pricing*
  convention) — for a signed axis that clipped the cheap tail, biasing the
  equal-tail measurement up. A signed marginal is now rebuilt on a *centred*
  grid (slack both sides, sized from the SBJ-aware first build's realized extent
  so a heavy tail is still covered) before `balanced_window` reads it.
  Non-negative axes are unchanged (`MultivariateAggregate._measure_marginal_window`).
- **Back the measurement depth off the FFT noise floor:** `[multivariate].window_nines`
  12 → **9**. A 2-D marginal's far tail is numerical dust below ~`1e-10`, so
  measuring equal-tail quantiles deeper read noise (and skewed a symmetric axis).
  Per-axis deficit stays ~`1e-9`, far below target.
- **Centre the measured window in the (power-of-two) grid.** The grid width is
  quantised to a power of two, so a window leaves unavoidable slack; that slack
  is now split either side instead of piled above (which left a symmetric axis
  off-centre). A non-negative axis stays clamped at 0.

Net: `DISCRETE.2` axis B is now `[-64, +64]` centred on 0 (deficit ~3e-11). The
bivariate fit rounds `bs` *up* (`round_bucket`) for guaranteed coverage, never
to nearest: the grid length is a power of two, so a window lands on a
power-of-two-wide grid whatever `bs` is — rounding `bs` down can't tighten that,
it only clips or forces a larger `log2`. The dead space is split by the centred
placement instead.

## 1.0.0a74

### `round_bucket` ladder: no more 2.5x jumps

`round_bucket` (the library-wide "nice bucket size" rounder) used a 1-2-5 decade
ladder, so a raw `bs` of 3.4 jumped to **5** (a 2.5x overshoot; the `2→5` and
`20→50` gaps). It now rounds **up** to the denser ladder `{1, 2, 4, 5, 8} * 10**k`
for `bs ≥ 1` — every consecutive gap is ≤ 2x, so 3.4 → **4**, 5.5 → 8, 9 → 10.
`bs < 1` keeps the powers-of-two ladder (binary-exact for the FFT grid; already
≤ 2x). This is a blast-radius-wide change: any auto-sized `bs ≥ 1` may now land
on a finer/closer value (4 and 8 are reachable; the overshoot is bounded < 2x).

### Bivariate: per-axis `log2` / `bs` via `(x, y)` tuples

`MultivariateAggregate.update` (and therefore `build(..., log2=…, bs=…)`) now
accepts a 2-tuple to pin the axes independently:

- `log2=(log2_x, log2_y)` pins the per-axis grid `log2` (budget = their sum);
  a scalar remains the *total* budget split automatically.
- `bs=(bs_x, bs_y)` pins the per-axis bucket size; a scalar applies to both.

This lets you explore the split the auto-sizer doesn't — e.g. for the signed
`DISCRETE.2` book at a 2²⁰ budget, the auto split `(11, 9)` (which equalizes
`bs`) leaves the hard mean-0 axis under-resolved; `build(prog, log2=(9, 11))`
gives that axis the finer grid and its sd error drops ~3x at identical memory.
Tuples pass straight through `build`; they are bivariate-only (a tuple on a 1-D
`agg`/`port` is unsupported).

## 1.0.0a73

### Bivariate windowing: use the measured lower edge (no artificial 0-pin)

Follow-up to MV-2. `_size_axes` was pinning every non-negative axis origin to
`x_min = 0`, discarding the lower edge `balanced_window` had measured. For a book
whose mass lives far from 0 (low CV — e.g. a 500-claim compound of `10 * uniform`,
mean 2500, sd 129) that stranded the mass in the upper third of the grid and
wasted ~⅔ of each axis (≈89% of the 2-D cells).

- The axis origin is now the **measured** lower edge (snapped down to `bs`):
  negative on a signed axis, positive when the mass genuinely lives far from 0,
  and 0 only when the mass reaches the origin. The sole deliberate 0-pin is a
  `pnl` axis, whose loss has a known lower bound of 0 *and* whose `_affine_axis`
  relabel assumes a 0-based loss grid (the affine owns the tight P&L window).
- The FFT working buffer length `M` is now **decoupled** from the output length
  `N`: a compound is anchored at physical 0 (non-negative severity) or wraps its
  negatives (signed), so `M` is sized to reach from `min(0, x_min)` up to the
  window top, independent of the stored window `N`. This lets a tight far-from-0
  window shrink the stored grid without aliasing the upper tail.
- Example: the `10 * uniform` axis window tightens from `[0, 5115]` to
  `[1716, 3762]` with `bs` 5 → 2 (2.5× finer resolution at the same memory),
  deficit ~1e-10. The MV-2 signed bug book is unchanged.

## 1.0.0a72

### Bivariate axis sizing: measure, don't guess (MV-2)

Stage MV-2 of the bivariate firm-up (`dev/plan-mv.md`). Fixes the motivating
aliasing bug — a signed (`ssev`) bivariate book that lost **54% of its mass** to
wrap-around because each axis was sized for its *single-event* severity, not its
*marginal* support.

- **Axis sizing is now measured, not guessed.** `MultivariateAggregate._size_axis`
  (the old moment-window guess, `cap_log2=11`) is gone. Each component's
  standalone loss marginal is run first, an equal-tail `balanced_window` (MV-1)
  reads its support straight off the realized pmf, and the per-axis
  `(bs, log2, x_min)` is read off the measured window. The total grid budget is
  a single `update(log2=...)` input (default **20** = 2²⁰ cells, config
  `[multivariate].total_log2`); the per-axis split *falls out* of the measured
  supports (allocated so the two bucket sizes come out comparable). The measured
  window always covers the deep tail, so a smaller budget coarsens `bs` rather
  than clipping support — mass is conserved regardless of budget.
- **Signed axes no longer wrap.** `update_work` now lifts the 1-D
  `_fft_aggregate` `i0`/`j0` machinery to 2-D: the per-claim severity is laid
  into the padded buffer with physical 0 at index 0 (each axis's negative
  severity buckets wrapped to the top), the shared frequency applied
  elementwise, and the finished density rolled back onto each axis's output
  window. A signed marginal gets a centred two-sided window; the all-non-negative
  grid is byte-for-byte the original zero-pad path.
- The motivating book's per-axis deficit drops from **0.54 → <1e-6**; each
  marginal mean reproduces its standalone aggregate, and the marginal sd
  converges to the standalone as the budget grows (resolution-limited, not
  biased). New `[multivariate].total_log2` config knob (default 20).

This stage touches the **copula** sizing path only; `netceded` axis sizing
(`size_axis`) is unchanged here and is rerouted in MV-3.

## 1.0.0a71

### `.agg` library rationalization

Split the bundled `src/aggregate/agg/*.agg` files into a clear shipped set and a
temporary test-scaffold set, ahead of the alpha→beta cleanup.

**Shipped at v1.0 (4):**
- `examples.agg` — the curated default `build` library (also the 20-min intro and SPA dropdown). Unchanged.
- `actuarial-severity-curves.agg` — **renamed** from `other-distributions.agg`; the severity-curve reference, cited in docs.
- `decl-testers.agg` — **promoted** from `test_decl.agg`; the DecL *language*-stress corpus (kept in sync with the pytest tree).
- `cookbook.agg` — **renamed** from `testers.agg`; the broad insurance-useful worked-examples library (DRAFT; still to be de-duplicated).

**Temporary migration scaffolding (deleted before beta), now `_`-prefixed:**
- `_test_suite.agg` (from `test_suite.agg`) — the SLY-parity regression corpus + snapshot.
- `_test_suite2.agg` (from `test_suite2.agg`) — the splice/mixed-severity extension.

**Deleted:** `spa_examples.agg` (superseded by `examples.agg` for the SPA; its
content harvested into `cookbook.agg`) and `spa_examples-old.agg` (the legacy
walkthrough that originally seeded `cookbook`).

References updated across `tests/`, `src/aggregate/config.py`
(`TEST_SUITE_FILENAME`), `config.default.toml`, and the `scripts/` defaults. New
`tests/test_agg_libraries.py` is the permanent net that every shipped *user-facing*
library loads (parses + cross-resolves), so the `_`-prefixed scaffolding can
retire safely. Docs pending a rebuild.

## 1.0.0a70

### `balanced_window` + `Aggregate.focus` (bivariate sizing foundation)

Stage MV-1 of the bivariate firm-up (`dev/plan-mv.md`). A *measure-don't-guess*
windowing primitive, pure 1-D — the foundation the bivariate axis sizing (MV-2/3)
is built on. No bivariate behaviour changes yet.

- **`aggregate.utilities.balanced_window(ser, p, bs=None)`** — given a realized
  pmf series (`index = xs`, `values = ps`) and a discarded tail mass `p`, returns
  the equal-tail window `[q(p/2), q(1 - p/2)]`, optionally snapped to `bs`. The
  *post-calc* analogue of `estimate_agg_window`: it measures the window from an
  already-computed marginal rather than guessing from moments. **Balanced** means
  equal *probability* trimmed off each tail, so a signed P&L or skewed marginal
  stays centred on its mass. Reuses `make_var_tvar` for the quantiles so the
  convention matches `Aggregate.q` (`kind='lower'`). `p` is the literal discarded
  mass (e.g. `1e-6`), not a coverage — no `>1 → nines` reading.
- **`Aggregate.focus(p=1e-6)`** — a thin, no-recompute re-slicer: runs
  `balanced_window` on the realized `p_total` and returns the central
  `density_df` slice holding `1 - p` of the mass. Does not mutate the aggregate.

## 1.0.0a69

### DecL statement separation: blank line or `;`, no more `\`

How a DecL *program* splits into individual *statements* changed. The old rule —
"one statement per line; an indented or `\`-continued line folds onto the
previous one" — is replaced by a markdown/Python hybrid:

- **A blank line** (empty or whitespace-only) separates two statements (the
  markdown paragraph model). A statement may now span as many lines, with
  whatever indentation, as you like — a multi-line `port` just needs its units
  in one paragraph (no blank line between them).
- **A `;` at the end of a line** also ends a statement (the Python model), so
  dense one-statement-per-line lists stay legal. The `;` inside
  `hints{key=value;}` / `note{...}` is untouched (those end a line with `}`).
- **Comments are transparent** — a `#` / `//` comment never separates
  statements, so you can comment out or annotate a clause inside a multi-line
  statement (e.g. a reinsurance line) and the statement stays intact. The
  corollary: a comment alone no longer separates two statements; use a blank
  line or a `;`.

**Breaking changes:**

- **The `\` line-continuation is removed.** A statement spans multiple lines for
  free, so a stray backslash is now a **lexer error** (dropped from the
  `decl.lark` `%ignore` class) rather than silently ignored — it surfaces the
  copy/paste confusion the old behavior hid.
- Two statements on adjacent lines with **neither** a blank line **nor** a `;`
  between them now fold into one statement (usually a loud parse error). Insert a
  blank line or a trailing `;`.
- `Underwriter.to_agg` now writes entries blank-line separated; files written by
  older versions that packed statements one-per-line with no separator must be
  re-exported or hand-separated to re-load.

The single owner of the rule is `UnderwritingLexer.preprocess`
(`aggregate.parser`); `decl_writer._split_statements` mirrors it. Comments are
stripped before the vector-bracket step, so a stray bracket in a comment no
longer unbalances preprocessing. The bundled `.agg` libraries and the DecL
documentation were migrated. Docs pending a rebuild.

## 1.0.0a68

### `approximate()`: one fit core, symmetric guard, honest reflected-fit errors

`Aggregate.approximate()` / `Portfolio.approximate()` and the `approximate` DecL
keyword had **two** moment-fit implementations. The method path
(`approximate_from_mcvsk`) was the unguarded one: a symmetric or left-skewed
distribution hit `sln_fit`/`sgamma_fit` with non-positive skew and got a
degenerate `(-inf, inf, 0)` → a silent `nan` distribution (e.g. a 12-dice sum,
`skew = 0`, with the default `slognorm`).

- **Single fit core.** `_approximate_sev_kwargs` is now the one place the family
  fits and guards live (generalized to all five families: `norm` / `lognorm` /
  `gamma` / `sgamma` / `slognorm`). `approximate_from_mcvsk` is a thin **output
  adapter** over it (`scipy` / `sev_kwargs` / `sev_decl` / `agg_decl` /
  `Aggregate`), so the two `approximate` surfaces and the DecL keyword share one
  implementation. No second path to drift.
- **Symmetric → normal, with a warning.** A *shifted* family (`slognorm` /
  `sgamma`) requested for a (near-)symmetric distribution now returns its normal
  limit (the mathematically correct answer) and emits a `UserWarning` from the
  interactive `.approximate()` method (pass `approx_type='norm'` to select it
  explicitly). The declarative DecL/constructor path keeps degrading silently.
- **Reflected (left-skew) fits error honestly where they can't be drawn.** A
  left-skewed fit reflects (`sev_reflect`); that has no native frozen `scipy` or
  one-line DecL form, so `output='scipy'` / `'sev_decl'` / `'agg_decl'` raise a
  clear `ValueError` pointing to `output='sev_kwargs'` / the default `Aggregate`
  object (which do reflect) or `approx_type='norm'`.
- **`approximate('all')`** stays quiet about degeneration and skips any family it
  can't represent for the given distribution/output (e.g. reflected + `scipy`),
  returning the admissible subset.

## 1.0.0a67

### Fix: `Aggregate.approximate()` method was shadowed by a same-named attribute

The `approximate=` constructor keyword added in `dev/done/plan-approximate.md`
stored its value as `self.approximate`, which **shadowed** the existing
`Aggregate.approximate()` method (the method-of-moments surrogate factory, the
parity-partner of `Portfolio.approximate`) on every instance — `agg.approximate`
was the string `'exact'`, and calling it raised `'str' object is not callable`.

- **The build-time choice is now the attribute `Aggregate.approximation`** (a
  noun), leaving `approximate()` callable as before. It is **falsey (`''`) for an
  exact freq×sev convolution** and the fit kind (`'sgamma'` / `'slognorm'`)
  otherwise, so `if a.approximation:` reads as "is this object a moment-match
  surrogate?". The `approximate` DecL keyword, the `approximate=` kwarg, and the
  spec key are unchanged; `describe` still shows the `approximate` row.
- `Portfolio` was never affected (no `approximate` attribute) and is unchanged —
  the two classes again expose the same `approximate()` method.
- **Naming-vetting rule** added to `CLAUDE.md`: a new instance attribute set in
  `__init__` silently shadows a method of the same name, so new public
  method/attribute/kwarg names must be `rg`-checked against the existing surface
  at planning time (noun for stored value, verb for action).

## 1.0.0a66

### Portfolio windowed combine 1P — Portfolio MM + single-big-jump look-through

Final task of `dev/done/plan-bucket-window-2.md` (Round 3). The portfolio
combine grid is reconciled with the a62–a65 univariate (`Aggregate`) sizer. The
old `best_window` sized the shared grid from a **linear sum of per-unit window
widths**, which overstates the bulk by `sqrt(k)` for `k` iid units (it ignores
diversification) and double-counts every unit's far-tail allowance. The combine
now mirrors the per-aggregate *bulk / extent* split:

- **Bulk from Portfolio MM.** The shared `bs`/span is sized from the **exact
  total moments** (`agg_m`, `agg_sd`, `agg_skew`; cumulants add under
  independence) fed straight into the same `estimate_agg_window` the single
  aggregate uses — never a per-unit width combine. A diversified-iid book now
  sizes `~sqrt(k)` tighter than before (the headline win).
- **One portfolio single-big-jump extent floor** (`Portfolio._single_big_jump_window`):
  a *look-through* to the units, `sbj_hi_port = agg_m + max_k(sbj_hi_k − ES_k)`
  — the heaviest unit's one big claim on the combined bulk (**max**, not sum, so
  the per-unit a59 extents are not double-counted). Self-activating: a thin /
  well-diversified total leaves the grid unmoved.
- **Resolution floor** stays `min_k bs_k` (a unit's lattice must survive), and
  `bs` is `round_bucket`-ed **once** at the top (carry raw, round once).
- **Windowed non-signed origin (Plan B).** A *concentrated* non-signed total
  whose mass clears 0 (high-frequency / tiny-cv, e.g. `Poisson(100000)`) is now
  **windowed** — the shared grid starts at `x_min > 0`, routed through the same
  roll-combine path as a signed book — instead of wasting the whole lower grid
  on a forced 0-based placement. The per-aggregate heavy-severity "Regime B"
  limitation does **not** bind the combine (each unit keeps its own 0-based
  severity grid; only the convolved *total* is relabelled).
- **`x_min` policy:** the combine ships with the algorithm's `x_min`; back-compat
  with old published grids is **not** a ship gate. The numerics-2/3
  origin-invariance (`sum_i kappa_i(x) == x`, moments, mass) holds on whatever
  grid is picked (verified in the 1P tests).
- **Reporting parity (`[bs-reporting]`).** `Portfolio.bs_window_df` gains four
  inspectable combine-candidate rows — `mm` (the live MM bulk), `rms`
  (RMS-of-windows reference; the `mm − rms` gap reads as the
  skewness/diversification adjustment), `sbj` (the look-through), and `sum` (the
  legacy linear bound) — plus the `log2_need` / `clipped` columns (parity with
  `Aggregate`). New `Portfolio.bs_explanation` (verbose grid prose) and
  `Portfolio.tail_df` (per-unit + worst-of `total`). A windowed/signed combine
  that clips the tail records `Portfolio._bs_clip` and warns once (the
  speculative phase-1 pre-pass is silenced).

**As-built note.** The review's `mm ≤ rms ≤ sum` ordering holds robustly for a
diversified light-tailed book but can legitimately invert (`mm > rms`) for a
*concentrated subexponential* total, where MM is tail/skew-aware while the RMS
reference is symmetric-normal; the tests assert the ordering only in the clean
regime, and `bs_explanation` flags an inversion rather than treating it as a bug.
The multi-driver pooled root-find (`sum_k E[N_k](1−F_k(x)) = 1−p_star`) is left
as a documented refinement; the `max_k` look-through is a safe dominant-unit
bound that the FFT doubling-padding absorbs.

## 1.0.0a65

### Bucket-selection 1A-bucket — making the grid choice legible (`[bs-reporting]`)

Fourth task of `dev/plan-univariate-bucket.md`. The bucket-grid decision is the
#1 numerical choice; this surfaces it. Pure reporting -- no change to the grid.

- **`_bs_window_df` enriched** with two derived columns: `log2_need` (the log2 a
  method's window needs at its own `bs` -- a row with `log2_need > log2` was
  capped) and `clipped` (the estimated far-tail mass dropped, on the `used` row).
- **`bs_window_df`** -- a curated, read-only public property on **`Aggregate`**
  (method / window / grid / applies / selected / note) and **`Portfolio`** (one
  row per unit + the realised `used` grid). Folds in TODO **H10**; the private
  `_bs_window_df` keeps the expert extras (`coverage`, `W`).
- **`bs_description` / `bs_explanation`** -- short and verbose narratives of the
  grid choice (`Aggregate`; `bs_description` also on `Portfolio`). The short line
  is the winning method + `(bs, log2, x_min)` + grid top + any clip; the verbose
  prose adds the aggregate tail one-liner, which methods applied and why the
  winner won, and how to widen a clipped tail. The ANSI-coloured variants are the
  module functions `aggregate.distributions.bs_describe` / `bs_explain`
  (`color=True`), mirroring the tail narrative.
- **No more double warning.** A book whose far tail is clipped at sizing time
  (item 6's `DefectiveDistributionWarning` + structured `_bs_clip`) no longer
  *also* emits the generic update-time "PMF deficit" warning -- the sizing
  warning is the same mass with actionable advice (the exact `log2` to raise to),
  so the deficit warning is suppressed when `_bs_clip` is set. The most common
  heavy book (`100 claims lognorm cv 2`) now warns once, not twice.

## 1.0.0a64

### Bucket-selection 1A-bucket — wiring the tail report into sizing (`[use-selection]`)

Third task of `dev/plan-univariate-bucket.md`. The bucket sizer (`_bs_window`)
now consults the layered tail report (`_loss_tail_classes`, `concentration`)
instead of ad-hoc geometric proxies. Six changes, each byte-stability-gated
against the full suite and `test_bucket_sizing.py`:

1. **Thickness-gated single-big-jump floor.** The SBJ extent floor now fires
   only for a genuinely **thick** (subexponential-or-heavier) tail -- the loss
   right tail for a positive severity, the reflected left tail for a signed one
   (`is_thick`). A no-op for thin tails (the MoM window already covers them),
   now explicit and cheaper.
2. **Power-law / infinite-variance: honest truncation.** An infinite-variance
   (power-law) aggregate has no finite deep quantile to size to. The old
   `recommend_bucket` fallback **re-raised** on infinite cv (a crash for an
   unlimited Pareto); it is replaced by `_reachable_bulk_high`, which sizes the
   reachable bulk to a moderate `bucket_sizing_p` coverage from the severity's
   actual quantile, **warns**, and accepts the far tail as a reported deficit --
   exact below the truncation, never normalized back in, no `alpha`-quantile
   chase.
3. **Thin-left-gated windowed left-lift (the asymmetric window).** A
   concentrated heavy-severity book (the former "Regime B", which stayed 0-based
   and clipped the tail) is now **reclaimed**: the windowed upper edge is floored
   by the single-big-jump reach, which grows the grid -- and its severity
   discretisation extent -- enough that a single heavy occurrence fits and the
   thick right tail is captured. Lifting `x_min` off 0 is gated on a thin left
   tail. Selection is relaxed so a (possibly coarser) windowed grid wins when it
   captures a reach the 0-based pick clips.
4. **Tail-aware padding / slack.** The fixed `window_pad_skew` split is replaced
   by a tail-driven one: an **asymmetric** band puts ~3/4 of the power-of-2 slack
   on the thick side (`WINDOW_SLACK_THICK`); a **symmetric** band centres, with
   the loss/payoff convention demoted to a tie-breaker. (The per-edge window
   *coverage* still follows the convention.)
5. **Concentration from the report.** The windowed-eligibility gate is now the
   conservative `concentrated` flag (`agg_cv < CONCENTRATION_CV`, ~0.1), the
   single source of truth, replacing the looser geometric `w_lo > 0` (~0.21).
   Borderline books (`cv` in ~[0.10, 0.21]) revert to the 0-based grid.
6. **Far-tail clip → warning.** The positive-tail clip is promoted from a silent
   `logger.info` to a visible `DefectiveDistributionWarning`, with a structured
   `Aggregate._bs_clip` field (reach, grid top, `log2` needed, estimated clipped
   mass via `_clipped_mass_estimate`) for the validation / bs report.

Also: an informational **`severity (net occ)` overlay row** in `tail_df`
(`occ_net_severity_row`) reporting how occurrence reinsurance reshapes the
retained per-occurrence tail (bounded when a top layer cedes 100% to infinity,
else the gross tail) -- the sizer still works on the gross severity. The
signed-padding placement was verified (a two-sided signed book's negative and
positive reaches coexist in the FFT buffer without collision).

`Aggregate.tail_report` (added experimentally in a63, same-day) is **removed**:
the ANSI `color=` option lives only on `aggregate.tail.describe_rows` /
`explain_rows`, the future terminal/HTML hook; the plain `tail_description` /
`tail_explanation` properties are the public surface.

## 1.0.0a63

### Comprehensive scipy severity tail tables (the family classifier)

Populates `aggregate.tail`'s family classifier from a reconciled survey of every
`scipy.stats` continuous distribution's tail behavior (two independent
derivations cross-checked, `C:/s/AI/notes/2026-06-17-probability-distribution-tails/integrated.md`).
A user severity from any standard scipy family now classifies exactly instead of
falling back to `UNKNOWN`:

- **`SCIPY_SEV_TAIL`** expanded to ~45 fixed-class families (super-exponential:
  `chi`, `maxwell`, `rayleigh`, `nakagami`, `rice`, `halfnorm`, `foldnorm`,
  `gompertz`, `kstwobign`, `exponpow`, …; exponential: `chi2`, `erlang`,
  `fatiguelife`, `wald`, `invgauss`, `genexpon`, `geninvgauss`, `ncx2`,
  `recipinvgauss`, `halflogistic`, `hypsecant`, `dgamma`, `genlogistic`,
  `norminvgauss`, …; subexponential: `gibrat`, `johnsonsu`, `powerlognorm`).
- **`_POWER_LAW_ALPHA`** expanded with the correct shape-slot α for `loglaplace`,
  `nct`, `halfcauchy`, `foldcauchy`, `skewcauchy`, `kappa3`, `mielke`,
  `betaprime`, `f`, `jf_skew_t`, `levy`, `alpha`, `rel_breitwigner`, `landau`.
- **Parameter-aware** families added to the classifier: `gengamma` (Weibull
  exponent `c`; `c<0` → power-law), `exponweib`, `gennorm` / `halfgennorm`
  (`β`), `dweibull`, `tukeylambda` (`λ>0` bounded / `=0` exp / `<0` power-law),
  `levy_stable` (`α<2` power-law). The shared `_weibull_shape` /
  `_family_right_class` / `_family_sides` helpers are now the single source for
  `classify_severity` and the `tail_df` per-side classes (no parallel table).
- **`_SEV_LEFT_CLASS`** records the asymmetric two-sided families whose left tail
  differs from the right (`gumbel_r`, `gumbel_l`, `loggamma`, `moyal`,
  `exponnorm`, `landau`, `crystalball`) so a signed/two-sided severity reports a
  correct per-side `tail_df`.
- **`bounded` is now robust to any finite scipy support**: `_severity_bounded`
  falls back to `fz.support()` finiteness (spec-only), so finite-support families
  not in `_BOUNDED_SCIPY_SEVS` (`argus`, `bradford`, `gausshyper`, `johnsonsb`,
  `irwinhall`, `powerlaw`, `loguniform`, `genhalflogistic`, `tukeylambda` λ>0, …)
  classify as bounded without enumeration.

The four reconciled discrepancies between the two source views (`exponpow`
right-tail = super-exponential; `genhalflogistic` bounded; `studentized_range`;
`truncnorm`) are documented in `integrated.md`. Bucket selection does not read
the tail classifier yet, so this remains report-only.

## 1.0.0a62

### Bucket-selection 1A-bucket — the narrative tail report (`[tail-narrative]`)

Second task of `dev/plan-univariate-bucket.md`. The `tail_description` (short,
aligned) and `tail_explanation` (verbose) properties now narrate the **layered**
a61 `tail_df` — support and per-side tail class, not the old single-rung
sentence. Built from one shared `Aggregate._tail_rows()` (the same `TailRow`
list behind `tail_df`), so frame and prose never drift.

- **`tail_description`** — three aligned lines (frequency / severity /
  aggregate), e.g. `aggregate tail   [0, inf), subexponential right tail; not
  concentrated (P>0=1.00)`. Also feeds `Aggregate.info()`.
- **`tail_explanation`** — bottom-up prose: the per-component severity
  breakdown and blend, the single-big-jump mechanism (or the frequency driver)
  for a thick right tail, the power-law moment failure, the heavy-left
  (signed / `pnl`) sizing note, and the concentration.
- **`Severity.tail_description`** — one line (`lognorm, [0, inf), subexponential
  right tail`), from the same `severity_tail_row`. **`Frequency.tail_description`**
  — the family count class (support depends on exposure, so the full count
  support shows only in the aggregate's `tail_df`).
- **ANSI option** — `describe_rows` / `explain_rows` take `color=True` to
  emphasise thick (subexponential-or-heavier) tail classes in bold-red for a
  TTY; the properties stay plain (so `info()` is plain).

`aggregate.tail` text builders rebuilt around the rows: `describe_row`,
`describe_rows`, `explain_rows` replace the old `TailInfo`-based `describe_lines`
/ `explain` (which carried the now-dropped log-concave / single-rung phrasing).
**Byte-stable** — narrative only; selection still does not read the report.

## 1.0.0a61

### Curated `examples.agg` example library + `build` default

A new `src/aggregate/agg/examples.agg` (Version 1) is the curated, public-facing
DecL example set: ~32 hand-picked programs (2 severities, 2 distortions, ~21
aggregates, 8 portfolios) organized A–J by what each illustrates, spanning the
DecL-capability and numerical-character axes. It is the single source for the
default `build` knowledge base, the twenty-minute intro, and the `aggregate_api`
SPA examples dropdown.

- **`build` default database changed** from `test_suite` to `examples`
  (`config.py` `BuildSettings.databases`, `config.default.toml`). Out of the
  box, `build` now loads the curated set rather than the historical reference
  suite. Set `[build] databases = ["test_suite"]` to restore the old default.
- **New `testers.agg`** — the back-room comprehensive-coverage companion (WIP
  dumping ground), seeded from the legacy feature walkthrough; pending merge
  with the `test_suite` family (TODO T1).
- **Tests:** `test_decl_unparser` now parses its `test_suite` corpus through a
  dedicated `Underwriter(databases='test_suite')` rather than the `build`
  singleton (decoupled from the default-database choice); `test_config` asserts
  the new default.
- **TODO B4** logged: zero-truncated/zero-modified frequency is broken
  (`poisson zt` raises; `zm` semantics inverted) — redesign to take the base
  mean and ship forward shift helpers. The two ZM/ZT examples are commented out
  in `examples.agg` meanwhile.
- **Companion (`aggregate_api`):** the examples route reads the bundled
  `agg/examples.agg` and folds `\`-line continuations, so multi-line `port`
  programs are captured whole.

### `tail_df` schema revision — support + per-side tail class

Reworks the a60 `tail_df` to report **support and tail shape** (and nothing
that belongs to grid selection). Per author review:

- **`min` / `max` are now the structural support** (smallest / largest
  *attainable* value; `-inf` / `inf` at an unbounded end), not a
  method-of-moments reach. So `bounded` is the self-consistent `min` and `max`
  both finite, the aggregate of a fixed 3 × dice `[1..6]` reads exactly
  `[3, 18]`, a signed `dsev` book reads its exact two-sided support, and a `pnl`
  book reads `[-inf, premium]`. The numeric grid *reach* moves to the bucket
  report (`bs_window_df`, a later task).
- **`left` / `right` thick-thin become `left_tail` / `right_tail` full tail
  classes** (`bounded` / `super-exponential` / `exponential` /
  `subexponential` / `power-law`): a finite support end is `bounded` (a hard
  boundary, no tail), an infinite end carries the family decay rung. So a
  lognorm reads `left_tail = bounded`, `right_tail = subexponential`; a `pnl`
  book reads `right_tail = bounded` (premium cap), `left_tail = subexponential`
  (the loss right tail, reflected). The sizer's thick/thin is the derived
  `is_thick(right_tail)`.
- **`concentration_p` is now `Phi(mean / sd)`** — the normal-approximation
  probability the aggregate is positive (the band clears 0), a genuine p-value
  in `(0, 1)` — replacing the a60 `1 / cv` sd-count. The conservative
  `concentrated` gate (`cv < CONCENTRATION_CV = 0.1`) is unchanged.
- **Dropped `tail_class`, `alpha`, `log_concave` columns.** None drives
  selection; the actionable power-law fact moves into `note` as
  `"power-law, alpha=1.5, infinite variance"`. Final columns: `family, min,
  max, left_tail, right_tail, bounded, concentrated, concentration_p, note`.

`aggregate.tail` API updated accordingly (`TailRow` fields; `concentration(m,
sd)`; `build_tail_rows` takes `agg_m` / `agg_sd` / `agg_reflect` / `agg_shift`).
Still **byte-stable** — selection does not read the report yet.

## 1.0.0a60

### Bucket-selection 1A-bucket — the layered thick/thin tail report (`tail_df`)

First task of `dev/plan-univariate-bucket.md` (`[tail-report]`). A new
first-class, **spec-only** report of tail shape, built bottom-up across the
layers that determine grid choice. `Aggregate.tail_df` returns a DataFrame with
one row per layer — `frequency`; one per severity mix component (`comp0` …); the
combined effective `severity` (when there is more than one component); and the
`aggregate` — carrying the structural fields the sizer reasons about:

- `min` / `max` reach (claim-space layered-loss support per component;
  method-of-moments reach for the aggregate, two-sided for a signed book, `nan`
  for an infinite-variance power-law);
- `bounded`, and `left` / `right` **thick-thin** labels (thick ⇔
  subexponential-or-heavier; a non-negative layer is thin-left; `UNKNOWN` is
  conservatively thick);
- the `tail_class` rung, power-law `alpha`, and `log_concave`;
- (aggregate row only) the conservative `concentrated` flag and its
  `concentration_p` sd-margin, using the tighter `CONCENTRATION_CV = 0.1` cut.

The report carries **two facts** for a capped heavy family — its base-family
thickness and its structural bound — so a thick base capped by a finite `limit`
/ splice is reported as effective-bounded with a `"subexponential base, capped
at L"` note. No numeric tail estimator: classification is family-lookup plus
structure (the grid the estimate would need is the very thing being chosen).

New surfaces in `aggregate.tail`: `TailRow`, `is_thick`, `thickness_label`,
`severity_support`, `concentration`, `build_tail_rows`, `tail_frame`,
`CONCENTRATION_CV`. **Pure addition — byte-stable**: nothing in selection reads
the report yet (that is the next task, `[use-selection]`).

## 1.0.0a59

### Bucket-window 1A-fix — single-big-jump extent floor (heavy / signed severities)

Second part of `dev/plan-bucket-window-2.md` (§1A-fix). The 3-moment
method-of-moments output window is blind to a tail the first three moments do
not capture. Two failure faces, one root cause, are addressed by flooring the
selected window's *extent* (not its resolution) by a single big claim on an
otherwise typical bulk — for a subexponential severity the aggregate's far tail
is `P(S>x) ≈ E[N]·P(X>x)`, so the severity is probed at the `E[N]`-adjusted
level `p** = 1 - (1-p*)/E[N]` and the extent floored at `ES - μ_X + q_X(p**)`.

- **Signed severities — correctness (catastrophic case fixed).** A signed
  severity (e.g. `100 - lognorm 10 cv 2.5`) can have positive aggregate skew
  while its reflected tail reaches far below 0. The MoM window then misses the
  reach entirely and the severity *wraps the FFT buffer* (aliasing), losing
  ~47% of the mass and returning a garbage law. The grid now always covers the
  single-big-jump reach `[sbj_lo, sbj_hi]` (width ≥ severity reach), keeping the
  bulk `bs` when the log2 budget allows and coarsening `bs` within the log2 cap
  otherwise — aliasing is corrected at any log2 (mass recovered to 1).
- **Positive heavy severities — refinement.** A heavy unlimited severity's MoM
  window under-reaches the true right tail (e.g. `5000 claims lognorm 100 cv 2`
  clips ~3.9e-7 of the priced tail at the default grid). The window now extends
  up to the single big jump **when it fits at the bulk `bs` within the requested
  `log2`** (so a larger `log2` is captured fully and finely); at a constrained
  `log2` the MoM window is kept (clipping a tiny far tail beats coarsening the
  bulk to uselessness — e.g. a 5-claim, mean-50 book whose tail reaches 47k).
  The existing `DefectiveDistribution` warning still flags a material clip.
- **`log2` honored, no silent memory growth.** The single-big-jump floor never
  grows `log2` past an explicit / hinted / default request and never coarsens a
  pinned `bs`; light / thin / bounded / concentrated and windowed books are
  byte-stable (the floor's `max`/`min` are no-ops). New
  `[discretization] sbj_tail_floor` (default `1e-14`) caps how deep the severity
  is probed (guards `q_X(p**) -> inf` for a large `E[N]` on an unbounded sev).
- **Inspectability.** `_bs_window_df` gains an `sbj` row — the grid the
  single-big-jump extent implies (origin / `bs` / `log2`, sized like every other
  method row, so it never reads NaN); the selected method's `note` records when
  the floor binds. When a positive heavy tail is clipped at a constrained
  `log2`, a `logger.info` reports the reach and suggests the `log2` that would
  capture it.

New helpers `Aggregate._single_big_jump_window` and `._severity_low_estimate`.
Byte-stability of the `test_suite.agg` snapshot and the numerics-2/3 regression
gates is preserved.

Deferred: the planned signed-only kurtosis *diagnostic* is dropped — the
true-law compound kurtosis of the motivating signed case is modest (~4.8, not
the ~737 the plan cited, which was the empirical kurtosis of the already-aliased
distribution), so a kurtosis-vs-fit test does not fire. The aliasing it was
meant to surface is now fixed at source, and a material positive-tail clip is
already flagged by `DefectiveDistribution`.

## 1.0.0a58

### Bucket-window 1A — convention-aware aggregate output windowing

First step of `dev/plan-bucket-window-2.md` (Step 1, part 1A: the
`Aggregate` sizer; part 1P, the `Portfolio` combine, follows). The automatic
output window for a concentrated aggregate (mass band clears 0) is now
symmetric in its estimator and oriented by the sign convention, and the band
is placed sensibly in the grid rather than jammed against the floor.

- **Per-edge window coverage** (`estimate_agg_window` gains `p_lo` / `p_hi`).
  A windowed book covers its *protected* edge deep (anti-clip) and trims its
  *cheap* edge shallow (anti-waste): for a loss the upper (priced right) tail
  is protected and the lower trimmed; a payoff mirrors. New
  `[discretization] window_nines_trim` (default 6) sets the trim depth;
  `window_nines` (12) remains the protected depth. Backward compatible —
  callers passing only `p` are unchanged.
- **Balanced padding** (`[discretization] window_pad_skew`, default 0.1).
  The power-of-2 slack around a windowed band is split `f = 0.5 -/+ skew`
  below the band (loss -> more room on the right, payoff -> mirror) instead of
  all above. Only the *windowed* row is rebalanced; ordinary, exact-discrete
  and bounded books keep their band-bottom origin.
- **Convention from `value_type`, with override.** The windowed skew branches
  on `_is_loss_value` (never the label string); `update(window_convention=...)`
  overrides per call. Defaults derive from the aggregate's `value_type`.
- **Relaxed windowed-selection gate** (`<` -> `<=`). A band that clears 0 now
  wins on *placement* -- reclaiming the empty `[0, x_lo)` region and balancing
  the slack -- even when `bs` is unchanged, not only when strictly finer. The
  severity-fit guard is unchanged, so the change cannot select a windowed grid
  the severity does not fit.
- **Explicit Regime-B (heavy severity) branch.** When the mass band clears 0
  but a single severity overflows the windowed extent (the benign FFT wrap is
  invalid), the book keeps the 0-based grid and a `logger.info` explains why
  (expected, not defective -- no warning). This replaces the previous silent
  fall-back.

Byte-stability: ordinary aggregates (`agg_cv > 1/z`) and heavy-severity
Regime-B books are unchanged -- the full `test_suite.agg` snapshot and the
numerics-2/3 regression gates pass untouched. The only grids that move are
genuinely windowed (Regime-A) books, which gain a centred placement.

Deferred (documented in `dev/TODO.md`): Regime-B *clip remediation* (deepening
upper coverage / growing `log2` to capture the heavy right tail, e.g. the
`5000 claims cv 2` 3.9e-7 top-bucket clip). It needs a waste/clip threshold to
separate genuinely-wasteful Regime-B books from ordinary ones that merely have
`w_lo > 0`, and so deserves its own validated pass rather than risking the 1A
byte-stability guarantee.

## 1.0.0a57

### Numerics-3 — distortion spine (one Choquet engine; linear/lifted unified)

Third plan of the numerics program
(`dev/done/plan-numerics-3-distortion.md`). All distorted pricing routes
through one exact-discrete Choquet helper; the linear and lifted
allocations become one builder; the `T.*`/`M.*` column families are gone.
Step-0 audit with measured verdicts in `dev/audit-numerics-3-findings.md`.

- **One Choquet helper.** `spectral.choquet_weights(x, p, g)` computes the
  exact distorted atom weights `gp = g(T) − g(S)` (`T = P(X ≥ x)`, the
  strict shift of `S = P(X > x)`; a pmf on a clean law) and is the *only*
  place they are computed: `Distortion.price` (both `method='dx'` and
  `'ds'` now return the identical `Σ min(x,a)·gp` value), `make_q`,
  `Aggregate.apply_distortion` and `Portfolio._build_augmented` all route
  through it. Choquet values are capped dot products carrying the grid
  origin — exact on signed, shifted and nonuniform supports. The layer
  form `x0 + Σ g(S)·Δx` survives only as an internal reconciliation
  assert.
- **Deficit policy** (new `DefectiveDistributionError`,
  `constants.DEFICIT_MATERIALITY = 1e-4`): pmf deficits at the validation
  noise floor are renormalized away; small FFT-truncation losses (already
  advertised by `DefectiveDistributionWarning`) are parked per
  `S_calculation` (forwards: top atom, backwards: bottom atom — the two
  agree to tolerance on a clean law, asserted); material deficits raise
  unless `allow_deficit=True` is passed explicitly.
- **`view × value_type` pricing axis.** `Distortion.effective_g(view,
  is_loss_value=...)` resolves the 2×2 by XOR (dual iff `bid` XOR payoff
  role), branching on the canonical `_is_loss_value` flag, never the
  configurable label strings. A payoff-role object prices as the dual of
  the matching loss object, surviving label reconfiguration.
- **Unified linear/lifted builder.** `apply_distortion(distortion, *,
  view, S_calculation, allocation='lifted', allow_deficit=False)` builds
  one column schema for both methods in one O(n) sweep across all asset
  levels: `exag_i = Σ_{k≤a} κ_i·gp + a·g(S(a))·TAIL_i` with `TAIL` = beta
  (`exi_xgtag`, lifted) or alpha (`exi_xgta`, linear). **Breaking:** the
  separate `_collapsed_exeqa` linear pricing engine is deleted;
  `Portfolio.price` reads rows of the unified frame for both methods.
  Linear **totals** are unchanged (≤ 2e-12 vs the a56 capture) but linear
  **per-line** values move: the unified formula keeps the `X = a` state at
  its true `κ(a)` and splits only the strict tail by alpha (the old
  engine merged `X ≥ a` into the collapsed atom), and per-line capital now
  uses the same layer-ROE construction as lifted (the old separate
  `rcoc` engine differed structurally). Lifted surfaces reproduce a56 to
  1e-14 (exact books) / 1e-11 (64k-row FFT books, fp order-of-ops drift);
  locked by `tests/data/numerics3_precapture.json` +
  `tests/test_numerics3_distortion.py`; the corpus baselines were
  recaptured.
- **Breaking: `T.*`/`M.*` columns removed** (`T.L/T.P/T.M/T.Q/T.LR/...`,
  `M.L/M.P/M.M/M.Q/...`) along with the `tm_renamer` property. Pricing
  readers use explicit `L = exa`, `P = exag`, `M = P − L`; per-line
  capital `Q_i(a)` is computed **on demand** by the layer-ROE
  construction (line layer margin ÷ total layer ROE, integrated; the
  layer margin is the exact first difference of `exag_i − exa_i`) inside
  `pricing_at` / `pentagon_at` / `price`, with `Σ_i Q_i(a) = a −
  exag_total(a)` reconciled. Zero-total-margin layers contribute zero
  capital (under the identity distortion per-line `Q` is 0 — there is no
  margin to allocate); fully-loss-funded layers (`gS = 1`) use the
  L'Hôpital ROE limit.
- **Breaking: mass-on-unbounded guard moved into the builder** (G6).
  `apply_distortion` / `Aggregate.apply_distortion` refuse a mass
  distortion (e.g. `ccoc`) on an unbounded support for the lifted frame —
  previously only `price(allocation='lifted')` refused, so `exag_total`
  could still build an unstable frame. `allocation='linear'` remains
  available (the collapsed default atom is bounded by construction; the
  unstable beta columns are blanked). `analyze_distortions` skips such
  members of a sweep with a `UserWarning` instead of failing the exhibit.
- **Breaking: `efficient` removed entirely** from `apply_distortion` /
  `price`; one frame shape. The diagnostic layer curves moved to the new
  explicit `Portfolio.allocation_diagnostics(distortion,
  surface='lifted'|'linear')` frame (`layer_loss/premium/margin/capital`,
  `cum_margin/cum_capital`, `layer_roe_total`, plus kappa/alpha/beta and
  `F/gF/S/gS/gp_total`); `pedagogy.plot_twelve` consumes it (the
  efficient→full cache-pop hack is gone).
- **Breaking: `apply_distortion` cache key widened** from the distortion
  name to `(name, view, role-flag, S_calculation, allocation)`, so
  bid/ask, loss/payoff, forwards/backwards and linear/lifted frames
  coexist (previously a second call with different options returned the
  stale first frame). `augmented_dfs` is keyed accordingly.
- **Aggregate surface aligned.** `Aggregate.apply_distortion(dist, *,
  view, S_calculation, allow_deficit)` writes `gS`, `gp_total` and the
  exact `exag = ρ_g(X ∧ a)`; the six surfaces (`Distortion.price` dx/ds,
  Portfolio `exag_total` / `price` both methods, Aggregate `exag` /
  `price`) agree on the total premium to machine precision.
- **Signed (P&L) books price.** The numerics-2 `NotImplementedError` is
  gone: total distorted columns (`gS/gp_total/exag_total`) are exact on
  signed windows and the signed total prices via `dot(κ, gp)`
  (additive across units); the equal-priority per-line distorted columns
  are NaN (not a recovery share on a signed grid). Homogeneous payoff
  books price through the dual automatically; mixed books still raise at
  construction (hygiene-4).
- `AllocationBounds` owns its bounded-total collapse directly (built from
  the `exi_xgta_*` columns); reproduces its baseline unchanged.
- `Pentagon.from_row` reads `L/P` and derives `M` (per-line `Q` needs the
  layer integral — use `Portfolio.pentagon_at`).
- Docs: `5_x_distortions.rst` / `5_x_portfolio_calculations.rst` /
  `2_x_10mins.rst` updated to the exact-discrete formulation (doc build
  pending, run manually).

## 1.0.0a56

### Numerics-2 — objective spine (shifted-support kappa + direct sums)

Second plan of the numerics program
(`dev/done/plan-numerics-2-objective.md`). `Portfolio.add_exa` and the
Aggregate objective columns rewritten on the exact-discrete, origin-carrying
footing; signed (P&L) books now get the objective allocation columns. Step-0
audit with measured verdicts in `dev/audit-numerics-2-findings.md`.

- **Shifted-support kappa.** `exeqa_{line}` is computed from each unit's
  **native** pmf: the first-moment density `x·p_i(x)` is built from true
  physical values and scattered into the physical-zero FFT buffer
  (first moments, unlike probabilities, cannot be recovered from a rolled
  vector), then `ift(ft_xp_i · ft_not_i) / p_total`, relabelled onto the
  output window by the same roll as the combine. Exact on negative and
  nonzero origins (brute-force-convolution tested). Per-unit FT state is
  transient within `update` (D6) — only scalars and native pmfs persist.
- **`ft_nots` single owner.** Per-line "not-line" FT products live in one
  helper: spectral division when the line's spectrum has no exact zero bins
  (measured per-bin well-conditioned even on underflowed spectra),
  prefix/suffix partial products otherwise — `O(m·M)`, replacing the legacy
  `O(m²·M)` rebuild.
- **Direct sums carrying the origin.** `exa_total/lev_total` =
  `Σ_{x≤a} x·p + a·S(a)`; `exlea/exgta/exi_xlea/exi_xgta/exa_{line}` from
  forward/reverse direct sums of `kappa·p_total`; Aggregate `lev/exa/exlea/
  exgta` likewise. `cumsum(S)·bs` and the `loss_max` / `mult ∈ {1,10,100}`
  blanking heuristic are gone; ratio denominators carry explicit
  `F/S ≤ validation-noise` guards (NaN where the conditioning event is
  unresolvable; previously unguarded division could emit `-inf`).
- **Stand-alone unit quantities from native pmfs.** `lev_{line}` is the
  exact capped native sum `Σ_{x≤a} x·p_i + a·(1−F_i(a))` and `e_{line}` the
  native mean — valid whether or not the unit window overlaps the total
  window.
- **Signed (P&L) books**: `update(add_exa=True)` now computes the objective
  columns (the warn + F/S-only fallback is removed). The equal-priority
  share `kappa/x` is not a recovery share on a signed grid (steering 6), so
  `exi_x*_{line}` and `exa_{line}` are NaN there; `apply_distortion` /
  pricing on signed books raises `NotImplementedError` until numerics-3.
- **Breaking: `p_{unit}` columns removed from `Portfolio.density_df`**
  (both combine paths). Unit pmfs live on the Aggregates — read them via
  `unit_density` / `unit_density_df` / `aligned_unit_density_df`
  (numerics-1). The sampling/switcheroo cluster (`sample`,
  `add_exa_sample`, `swap_density_df`, `make_awkward`) is mechanically
  re-sourced onto the accessors (redesign deferred to its own plan);
  `swap_density_df` still accepts a user `p_{line}` frame.
- **Breaking: EPD family removed** — `add_exa_details` (`epd_0_*`,
  `epd_1_*`, `e1xi_1gta_*`), and `Aggregate.density_df['epd']` (no
  consumers). Stand-alone EPD is the one-liner `(e − lev) / e`.
- **`add_exa` signature changed**: takes the per-unit native state
  (`{name: dict(xs, p, ft_p)}`) instead of pre-built `ft_nots`. The
  `Portfolio.ft` / `Portfolio.ift` padding-bound wrappers (only used by the
  old `add_exa`) are removed — use `aggregate.utilities.ft/ift` directly.
- Regression: key columns byte-stable (baseline harness; recaptured for the
  removed `p_{unit}` columns and the ≤2.8e-12 Aggregate `lev` drift);
  derived columns gated by new pre-change spot-checks
  (`tests/test_baseline_spotchecks.py`, measured drift ≤7.5e-12 on the
  `(e−cum)/S` cancellation, ≤2.5e-14 elsewhere). New invariant suite
  `tests/test_numerics2_objective.py` (Σκ(x)=x, Σ exa_i = exa_total,
  brute-force kappa incl. negative/positive origins, zero-spectrum
  prefix/suffix, native lev, signed exeqa vs Monte Carlo). Docs updated in
  lockstep (`5_x_portfolio_calculations.rst`, quantiles, student guide,
  10mins, samples); doc build pending.

## 1.0.0a55

### Numerics-1 — unit-density decoupling

First plan of the numerics program (`dev/done/plan-numerics-1-unit-density.md`;
target architecture in `dev/plan-numerics-0-meta.md`). Pure-additive accessors
plus migration of the display readers off the legacy
`Portfolio.density_df['p_{unit}']` columns. No compute change; no distortion
surface touched.

- **New accessors on `Portfolio`** sourcing unit pmfs from the owning
  `Aggregate` objects on their native grids:
  - `unit_density(unit, view='agg')` — one unit's pmf (`view='sev'` for the
    discretized severity), indexed by the unit's own loss grid.
  - `unit_density_df(view='agg')` — long form, `(unit, loss)` MultiIndex, with
    window-audit metadata (`bs`, `x_min`, `x_max`, represented `mass`).
  - `aligned_unit_density_df(grid='total'|'union'|'zero', *,
    allow_window_mismatch=False)` — the explicitly-named **display adapter**
    scattering unit pmfs onto a common grid (bucket-number alignment). On a
    legacy zero-origin book `grid='total'` reproduces the `p_{unit}` columns
    exactly; on a windowed book it warns that the view is clipped unless
    acknowledged. Raises if a unit was re-updated off the portfolio `bs`.
- **Display readers migrated** off `density_df['p_{unit}']`:
  `Portfolio.percentiles` (still deliberately interpolated), `_limits`, and
  `plot` (total now always plotted first — Book standard — previously only
  guaranteed for two-unit books); `pedagogy.ClassicalPremium.distribution`,
  `pedagogy.plot_bivariate`, and the density / bivariate / stand-alone-M
  panels of `pedagogy.plot_twelve` (allocation panels ride with numerics-3).
- **`p_{unit}` columns are now legacy.** The write remains (kappa in `add_exa`
  and the sampling/switcheroo cluster still read them) and is dropped in
  numerics-2 when kappa goes shifted-support. Do not write new readers.
- **Fix:** `ClassicalPremium.distribution` referenced the removed
  `Portfolio.audit_df`; empirical moments now computed directly from the pmf.
- Stale plan pointer in the signed-path `update` warning repointed to
  `dev/plan-numerics-2-objective.md`.
- Tests: `tests/test_unit_density.py` (native-grid accessors, disjoint-support
  signed book, exact legacy parity gate, windowed-clip warning, and
  stripped-frame proofs that the migrated readers no longer need `p_{unit}`).
  New DecL programs mirrored in `test_decl.agg` (section UD).

## 1.0.0a54

### Hygiene 4 — value_type, fixed-layout info strings, pnl prem/lr meta

One batch, four items (`dev/done/plan-hygiene-4.md`):

- **`Portfolio.value_type` derived from its units.** Read-only property: the
  unanimous `value_type` of the constituent aggregates. A mixed loss/payoff
  book is rejected at construction with a `ValueError` naming the offending
  units (no coherent sign convention); an empty portfolio defaults to loss.
- **Fixed-layout `info` strings** across `Aggregate` / `Portfolio` /
  `Distortion`. Every row is always present, in the same order, for every
  instance — no conditional rows; unavailable values render as `n/a`. All
  three classes share one label/value convention
  (`aggregate.constants.info_row`, 25-col label, no colon); `Distortion` was
  rewritten onto it (was indent+colon style). Row changes: Aggregate gains
  `value_type`-near-top, `x_min`/`x_max` (replacing the conditional
  `window`/`signed severity`/`severity window` block), always-present
  `premium`/`expected loss`/`loss ratio`/`P(loss)` (the `E[margin]` row is
  dropped — derivable), `bounded` and `id` footer rows; the `approximate`
  continuation line is dropped (detail stays in the note). Portfolio gains
  `value_type`, `x_min`/`x_max` (replacing `signed window`), premium rows;
  `tail`/`bounded` move to the footer; `hash` is relabelled `id`. Distortion
  drops `display name`/`strict-pricing`, renames `mu({0})`/`mu({1})` to
  `weights mean`/`weights max`, adds `kind name`/`shape`/`shape name`/
  `other params`/`area`. The full row catalogue and value enumerations are
  documented in `dev/info-strings.rst` (destined for docs; **docs pending
  rebuild**).
- **`stats_df` `('meta','prem')`/`('meta','lr')` backfilled for `pnl`.** The
  `pnl X prem - ...` form routes premium through `agg_premium`, which never
  reached the meta rows; they now backfill from it when the exposure clause
  supplied no premium. GROSS basis: under reinsurance `prem`/`lr` are the
  theoretical pre-reinsurance figures (`lr = gross el / prem`). The frozen
  numeric baseline is unaffected (no `pnl` programs in the corpus); no
  density / risk-measure numbers move.
- **`value_type` labels configurable.** New `[labels]` config section
  (`loss = "loss"`, `payoff = "payoff"`; env `AGGREGATE_VALUE_TYPE_LOSS` /
  `_PAYOFF`). Objects store the role as a private boolean `_is_loss_value`
  (loss is the anchor pole); the label is resolved at the display/parse
  boundaries only, so a relabel renames the printed word without moving any
  object's role, and future pricing code branches on the boolean, never the
  label text.

Breaking (display-level): code pinning the old `Distortion.info` format
(`Distortion: {name}`, colon rows), the Portfolio `hash` label, the Aggregate
`E[margin]` / `signed window` / `severity window` info rows, or the
conditional presence of `dsev_bucket` must be updated. Constructing a
`Portfolio` mixing loss and payoff units is now an error. The `Aggregate`
constructor and `value_type` setter now raise on an invalid `value_type`
(previously the constructor silently coerced to `'loss'`).

## 1.0.0a53

### DecL unparser + program formatter (`decl_writer`)

New `aggregate.decl_writer` module — the structural inverse of the parser
(`dev/done/plan-decl-unparser.md`). It renders a parsed spec back to canonical
DecL text instead of pretty-printing by regex, so a single function backs program
display, the `to_agg` exporter, and any future web `format` endpoint.

- **`spec_to_decl(spec, kind, name)`** — the unparser. Pure function from a raw
  transformer spec (`parsed.spec` / a knowledge entry's `pp.spec`) to canonical
  DecL. Built from clause renderers that mirror the transformer rules one-for-one
  (exposure, layers, severity incl. scale/reflect/`mean cv`/mixtures/splice/
  `dsev`/`xps`/`picks`, frequency incl. `mixed`/`zm`/`zt`, reinsurance, `pnl`,
  `port`, `multivariate`/`copula`/`netceded`, `approximate`, distortions, note/
  hints trailer).
- **`format_program(spec_or_text, *, fmt='text'|'html'|'ansi'|'latex')`** — the
  public entry. Accepts a spec, a `(kind, name, spec)` tuple, or a program string
  (which it parses first). Pure: returns a `str`, never prints. Colorization
  reuses the existing `decl_pygments.AggLexer` (no second keyword list).
- **Contract:** idempotence one step removed — `f(f(f(x))) == f(x)` with
  `f = spec_to_decl`. The whole reference corpus (`test_suite.agg` +
  `test_suite2.agg` + `test_decl.agg`) round-trips, verified by
  `tests/test_decl_unparser.py` with a numpy/inf/object-aware spec comparator.

**Breaking.** `utilities.decl_pprint` is **removed** (along with its
`pygments`/IPython plumbing in `utilities`). `Aggregate.pprogram` /
`pprogram_html` and `Portfolio.pprogram` / `pprogram_html` now render the
**canonical** form via `format_program(self.program)` (was the verbatim text with
notes stripped); `self.program` still holds the raw input. `Underwriter.to_agg`
now emits `spec_to_decl(spec)` per entry, so exported `.agg` files are canonical
(it falls back to the stored program only for `minimum`/`mixture` combinator
distortions, whose child references cannot round-trip). Docs that imported
`decl_pprint` now use `format_program`.

Also: fixed two `test_decl.agg` notes that carried `{...}` braces inside
`note{...}` (the `NOTE` terminal cannot represent `}`); they never parsed
standalone (`test_decl.agg` is a reference corpus, not runtime-loaded).

## 1.0.0a52

### Hygiene-3 batch (grammar + robustness nits)

Three small, independent nits, one version bump (`dev/done/plan-hygiene-3.md`).
No movement of the numeric baseline.

- **Underscore digit separators in DecL numbers.** The `NUMBER` terminal now
  accepts Python-style `_` group separators — `agg BIG 10_000_000 claims …` —
  with leading / trailing / doubled underscores (`_1`, `1_`, `1__0`) rejected by
  the lexer, exactly as Python's `float()` / `int()` behave. Grammar-only: the
  transformer already coerces via `float`, which strips the underscores.
- **`of` as a share synonym in reinsurance.** A reinsurance clause now accepts
  `of` alongside `so` / `po`, so `occurrence net of 90% of 6000 xs 4000` reads
  naturally. `of` is treated as *share of* (`so`): a literal percentage is the
  share directly, a bare amount is `amount / limit`. No new terminal (the `OF`
  token already existed); a single new `reins_clause` alternative.
- **Fixed an array-ambiguous truth test.** `Aggregate._sev_label` used
  `if not self.sevs:`, which raised *"truth value of an array … is ambiguous"*
  for a multi-component (weighted) severity, where `self.sevs` is an ndarray.
  Replaced with the explicit `self.sevs is None or len(self.sevs) == 0` idiom; a
  sweep of `distributions` / `portfolio` / `spectral` found no other array-valued
  truthiness tests.

Deferred: the "every public DataFrame member present (`None`) before compute"
item was moved to `dev/TODO.md` Track H (**H9**) — review found it largely
already-satisfied or aimed at members that don't exist; the genuine narrow
version needs separate scoping.

## 1.0.0a51

### Non-zero aggregate output window for high-mean / thin-tail aggregates (Plan B)

A concentrated aggregate — one whose coefficient of variation is small enough
(`agg_cv < 1/z`, `z = norm.isf(1e-WINDOW_NINES) ≈ 7`) that its whole probability
mass sits a long way above 0 — is now computed on a **two-sided output window**
far from 0 instead of the wasteful `[0, x_max]` grid. The headline case

```
agg Window 10000000 claims dsev [1 2] poisson
```

(mean 15,000,000, sd ≈ 5,000) used to build at `bs ≈ several hundred`, spending
almost all of its resolution on the empty `[0, 14.97M]`; it now resolves at
**`bs = 1`** on the window `[≈14.96M, ≈15.04M]`, with matching moments and mass.

**How it works.** This reuses the existing benign-FFT-wrap machinery built for
the negative-x / `pnl` work: the severity is laid into the period-`M·bs` FFT
buffer and the finished aggregate is relabelled onto the window by a single
modular `np.roll` of `round(x_min/bs)`. Relabelling a finished, exact array
carries no `N·s` shift term, so it is correct for random as well as fixed
frequency; the large roll and any period straddle are handled automatically by
`np.roll`'s modular semantics. The compute path was already in place — the new
work is purely the **sizing** decision in `Aggregate._bs_window`.

- **New `windowed` sizing method** (a row in the inspectable `_bs_window_df`).
  It sizes `bs` from the realised band *width* (`estimate_agg_window`), not from
  `x_max`, and is selected only when **strictly finer** than the 0-based pick —
  a self-tuning, self-limiting rule. It can only be finer when the band clears 0
  (`agg_cv < 1/z`), so **ordinary aggregates are byte-for-byte unchanged** (their
  window would include 0; the candidate never qualifies).
- **Integer lattice preserved.** For a discrete `dsev` the windowed method keeps
  the exact lattice `bs` and grows `log2` a small, bounded amount past the cap
  (`WINDOW_LOG2_GROWTH = 4`) rather than coarsening below the lattice and
  mis-placing the atoms — so the headline case lands at `bs = 1`, `log2 = 17`.
- **Severity-fit guard.** Windowing relabels only the aggregate; the severity is
  still discretised on `[0, N·bs]`. A single occurrence must fit that extent, so
  a `fixed`-1 / `approximate` object (whose one severity already sits at the
  aggregate mean) is **not** windowed — it falls back quietly to the 0-based
  grid. The `windowed` row is still recorded (marked `applies=False`) for
  inspection.
- **Occurrence reinsurance suppresses windowing.** The occ-reins severity
  rebucketing and `reins_density_df` carry the severity on the *output* grid
  (`xs == xs_sev`), which a non-zero window origin would break. A book with occ
  reins keeps the 0-based grid; **aggregate** reinsurance is unaffected
  (it operates on the aggregate, on the windowed `xs`/`x_min`).

**Behavioural note (intended).** `q` / `F` / quantiles / plots of a windowed
aggregate are defined on the window `[x_min, x_max]`, not from 0. The window
covers the probability mass to `1 − 1e-WINDOW_NINES` per edge; what is given up
is the `[0, x_min)` *axis* region (sub-tolerance far tail, low-attachment layer
losses, the from-0 severity overlay). Pass **`x_min=0` to `update`** to force the
legacy 0-based grid back. The footprint is broader than the single headline
example: **any** non-reinsured aggregate concentrated enough that its mass clears
0 (e.g. large claim counts) now resolves on a finer, non-zero-origin window.

## 1.0.0a50

### Hygiene: consistent `info`, self-describing `approximate` note, window-aware plots

Three small display/consistency fixes (no numeric baseline moves):

- **`Aggregate.info` always emits the `approximate` line.** It is now a
  permanent header line positioned *after* severity (`freq → sev → approximate`)
  and shows `exact` for an ordinary aggregate, instead of appearing only when a
  fit was active and *before* severity. The scaffold no longer reorders or drops
  the line by state. (`Portfolio.info` and `Distortion.info` audited — their
  conditionals are all feature/content driven, left as-is.)

- **The `approximate` note records the original program and fitted params.**
  Previously the note was assembled inside `Aggregate.__init__` **before**
  `self.program` was set by the build path, so it could only ever carry the fit
  moments — never *what* was approximated. The note is now kept as the user's
  pure note at construction; the fit is captured in a structured
  `self._approx_fit`, and a new `_approx_description()` renders
  `"<program>  approximated by <kind>: <family>(params), m=.. cv=.. skew=.."`
  lazily — shown indented under the `info` `approximate` line and folded into the
  note once `program` is available. Round-tripping rides on `program` (re-parsed
  on load), not the note, so the note never compounds across re-exports.

- **`Aggregate`/`Portfolio` plots are window-aware.** The linear x-limits
  (`_limits(stat='range')` and the discrete-plot left edge) are keyed on the grid
  origin: an ordinary 0-based aggregate is unchanged and a signed P&L window
  keeps its two-sided range, but a **thin-tailed output window starting above 0**
  now anchors the left edge at the realised support minimum instead of forcing 0
  (no empty `[0, x_min]` band). Existing ordinary/signed plots are unchanged; the
  new branch prepares the axes for non-zero-origin output windows. The
  severity-overlay-vs-window question is deferred (see `dev/TODO.md`).

## 1.0.0a49

### Fixed: Portfolio combine grid — `best_window` replaces the RMS combine

The portfolio auto-sizer combined its units' per-unit window choices into one
shared `(bs, log2)` grid by **root-sum-square** (`Portfolio.best_bucket`). That
scaled the wrong way — *k* identical units gave `round_bucket(b·√k)`, so
**adding units coarsened the grid** — and it ignored the integer lattice
entirely, so an all-integer discrete book got a fine continuous `bs` (e.g.
`1/4096`) spread over the full `log2=16` cap.

New `Portfolio.best_window(log2, bs_in, bucket_sizing_p)` implements the correct
**resolution + span** rule, the *max* of two independent constraints:

```
bs = round_bucket(max(min_k bs_k, W_tot / N))
```

- **resolution** `min_k bs_k` — the finest bucket any unit needs (from each
  unit's own `_bs_window`), captured in a phase-1 analytic pre-pass;
- **span** `W_tot / N` — the no-wrap floor, `W_tot = Σ_k W_k` over the
  *selected-method* support-window widths (not the padded `used`-row extent).

For a **non-signed** book the grid keeps origin 0 and `log2` is now **shrunk**
to just hold the summed support — a tiny discrete port no longer inflates to the
`log2` cap. The **signed** (P&L) path keeps its analytic origin estimate and the
`log2` cap (tight signed `log2`/origin is deferred to the output-window work).
`best_bucket` is **retained but deprecated** (`DELETE BEFORE BETA`) as a
side-by-side comparison aid; it is no longer on the live path.

**Effect on the knowledge base (9 of 146 objects move; no single aggregate
changes — the bug was purely the combine):**

- *Wins.* Discrete books size correctly: the Bodoff portfolios and
  `PIR.1.Discrete` move from an absurd fine `bs` (≈0.03–0.0001 over 65536
  buckets) to the lattice-correct `bs=1` with a shrunk `log2`.
- *Neutral.* The continuous CNC ports re-resolve to each unit's natural `bs`
  (e.g. 0.03125 → 0.125) with mean fidelity unchanged.
- *Known limitation (surfaced, not introduced).* The two heavy-tailed HuSCS
  catastrophe ports coarsen further (`bs` 20000 → 300000). This is **not** a
  combine defect: the `Hu` unit's *own standalone* `_bs_window` already sizes at
  `bs=300000`, because its `exp()·lognorm` cat severity has a `1−1e-12` window
  ~1.9e10 wide. The span faithfully refuses to wrap that mass (the old `bs=20000`
  silently truncated it). A combined-moment span gives the identical bucket, so
  there is no combine-level fix — the fix belongs to the per-unit window
  *coverage* policy for heavy tails, which is out of scope here (see
  `dev/TODO.md`). These cat books need an explicit `bs` for production use today
  regardless.

Regression tests in `tests/test_bucket_sizing.py`; the bucket-sizing decisions
for the whole knowledge base are snapshotted to `tests/data/bucket_baseline_*.csv`
(a review reference, not a hard gate) via `scripts/bucket_baseline.py`.

## 1.0.0a48

### Fixed: spliced unbounded severities crashed window sizing

Building an aggregate whose severity splices an **unbounded** base family (e.g.
`sev lognorm 40 cv .65 splice [1 100]`) crashed in grid sizing with
`ValueError: Inadmissible value passed to round_bucket, inf`.

`splice [lb ub]` records the cap in `sev_lb`/`sev_ub` and conditions
`fz.cdf/sf/isf/ppf/pdf`, so the severity correctly reports `bounded == True` — but
`fz.support()` still returned the underlying family's `(0, inf)`. The bounded-window
sizer (`_bounded_severity_window`), invoked precisely *because* the severity is
bounded, then read an infinite upper edge. `_apply_lb_ub` now also patches
`fz.support()` to the honest `[sev_lb, sev_ub]` (mirroring the reflect-shift support
patch in `_apply_reflect`). The window of existing uniform-splice aggregates
tightens slightly to the true support — a strict accuracy gain (no grid wasted over
zero-mass regions). Only the splice path is affected; unspliced severities short-
circuit before the patch.

## 1.0.0a47

### Added: `approximate` DecL keyword — method-of-moments aggregates

A new design-time directive replaces the freq × sev FFT convolution with a single
continuous severity fitted to the aggregate's first three moments — the fast
shortcut for very-high-frequency books where the exact convolution is overkill.

```
agg Big 1e6 claims sev lognorm 100 cv 2 poisson approximate sgamma
```

- `approximate exact | sgamma | slognorm` (`exact` is the inert default; omitting
  the clause is the same). `sgamma`/`slognorm` fit a **shifted gamma / shifted
  lognormal**; a **normal** is used as the symmetric (skew ≈ 0) limit, and a
  **reflected** fit handles genuinely left-skewed aggregates (verified to match
  the exact aggregate's mean/CV/skew to ~7 significant figures across all three
  skew regimes).
- **No special compute path.** The substitution happens in `Aggregate.__init__`:
  the object is rewritten as an ordinary fixed-1-claim aggregate of the fitted
  severity, so `density_df`, validation, the `pnl` affine, and the `Portfolio`
  combine all work with zero special-casing. The original program round-trips
  (it is preserved on `self.program`); the fit is summarised in `note` and shown
  in `info`.
- **Incompatible with occurrence reinsurance** (which acts pre-convolution, so the
  method-of-moments fit has nothing to bite on) — rejected with a clear error at
  parse time and in the constructor. **Aggregate reinsurance rides along**
  unchanged. Works on `pnl` too: the loss part is fitted and the premium affine
  rides along.
- Available on the `agg … claims …`, `agg … dfreq …`, and both `pnl` forms.

No core/`freeze_knowledge` impact — no existing program uses `approximate`, so all
146 knowledge-base objects are unchanged to 1e-12. Grammar changed →
`ref_include.rst` regenerated; **doc rebuild pending**.

## 1.0.0a46

### Added: exponential-tilting pedagogy (Grübel–Hermesmeier illustration)

Exponential tilting for FFT aliasing control returns as a **pedagogy helper**, not
a core feature. The production convolution stays tilt-free — padding remains the
operational aliasing control (the `tilt`/`tilt_vector` arguments dropped from the
core `ft`/`ift`/`update_work` in the 1.0 refactor are **not** restored).

- New `aggregate.pedagogy.tilted_aggregate_density(agg, *, log2, bs, padding=0,
  tilt=None, normalize=False)` runs a single tilted convolution
  (`z·e^{-θk} → rfft → freq_pgf → irfft → ·e^{+θk}`) entirely locally. With
  `tilt=None` it reproduces the ordinary untilted convolution byte-for-byte.
- New `tilt_vector(theta, n)` helper and `gh_tilting_exhibit(...)`, which
  assembles the full Grübel–Hermesmeier (1999) Poisson/Levy comparison table
  (accurate + closed-form exact + tilt sweep) in one call.
- Rewired `docs/2_user_guides/problems/010_gh_example.rst` to the new helpers
  (the old `tilt_vector=` kwarg no longer existed). **Doc rebuild pending.**

No core/`freeze_knowledge` impact — the production path is untouched (all 146
knowledge-base objects unchanged).

## 1.0.0a45

### Fixed: `pnl` with a signed loss severity (`dsev` negative atom / `ssev`)

A `pnl` (premium-minus-loss) aggregate whose **loss severity is itself signed**
— a `dsev` with a negative atom, or an `ssev` — now convolves the loss on its
genuine signed grid before the affine relabel onto the P&L window. Previously the
affine path hard-coded a **0-based** loss grid, so the loss's negative atoms
wrapped to the top of the FFT buffer: roughly half the mass was silently dropped
and the empirical moments read ±2¹⁵ grid-index garbage (e.g.
`pnl GP 5 premium - dfreq[3] dsev[-1 1]` reported `Est EX = -32763.75` instead of
`5`). It now yields the correct `P&L ∈ {2,4,6,8}` with mass 1, mean 5, sd √3.
This is a **bug fix, not a breaking change** — every ordinary (non-negative-loss)
`pnl` is byte-for-byte unchanged (all 146 frozen knowledge-base objects match to
1e-12).

Implementation: `_bs_window` already sized a correct signed loss window; it now
hands that loss origin back to `update` for the affine case (0 for an ordinary
pnl, preserving the legacy grid), and `update` builds the loss grid uniformly
from it. `_apply_agg_affine`, already origin-agnostic, relabels the signed loss
onto the tight P&L window. It also now warns (`DefectiveDistributionWarning`) when
the reverse-and-roll drops more than dust off the P&L window — surfacing the
genuinely-unrepresentable far-tail case that was previously silent.

## 1.0.0a44

### Hygiene: module organization & dependencies

- **Relocated `make_ceder_netter`** and its layer-order validator
  `_validate_reins_layers` from `utilities` to `distributions`, their only
  consumer (the occurrence/aggregate reinsurance application). Hard move, no
  deprecation alias. Direct importers should use
  `from aggregate.distributions import make_ceder_netter, _validate_reins_layers`.
- **Dropped unused runtime dependencies** `cycler`, `psutil`, `ipykernel`, and
  `jinja2` — none were imported anywhere in `src/aggregate` (`cycler` is still
  provided transitively by matplotlib; `bounds` uses stdlib `itertools.cycle`).
- **Deferred `IPython` to lazy imports** inside the two display helpers that use
  it (`decl_pprint`, `agg_help`), removing it from the top of `utilities`. Since
  `utilities` sits on the `import aggregate` path, this cuts roughly a second off
  cold import time; `IPython` is pulled in only when a display helper is actually
  called. (`Pygments` is left eager — it is ~0 ms to import and is loaded anyway
  by the `decl_pygments` lexer.)
- Confirmed **no var/tvar duplication**: `utilities.make_var_tvar` is the single
  implementation; the per-instance `Aggregate`/`Portfolio._make_var_tvar` are
  thin wrappers.

## 1.0.0a43

### Fixed: `Portfolio.describe` spread column for signed portfolios

`Portfolio.describe` reports the spread as **CV** normally and **SD** when the
portfolio is signed (any unit is a P&L / negative-support `ssev`/`dsev` unit),
mirroring `Aggregate.describe`. The choice is now made once, portfolio-wide:
CV and SD cannot be mixed in one frame, so if *any* unit is signed the whole
table — every unit block and the total — uses SD, with unsigned units forced
via the new `Aggregate._describe(force_sd=...)`. The total SD is read robustly
from the second moment (`sqrt(ex2 - mean²)`), never `mean × cv`. Previously a
mixed signed/unsigned portfolio produced a ragged frame (some unit blocks in CV,
others in SD) and the mean-zero total showed a blown-up `Est CV`. All-unsigned
portfolios are unaffected (byte-for-byte identical output).

### Added: `price_pentagon` on `Aggregate` and `Portfolio`

Complete the eight-stat pricing octet (`L, M, P, Q, a, LR, PQ, ROE`) from a
capital level plus one target — no distortion involved, pure accounting
completion against expected loss at that level:

```python
port.price_pentagon(p=0.99, ROE=0.10)   # VaR capital + cost of capital
agg.price_pentagon(a=250, LR=0.70)      # asset level + loss ratio
```

- Fix the capital level with exactly one of `p` (VaR probability) or `a` (asset
  level, snapped to the grid).
- Supply exactly one pricing target: premium `P`, cost of capital `ROE`
  (a.k.a. CoC), loss ratio `LR` — also `M`, `Q`, `PQ`. The target keywords match
  the canonical stat names (`PENTAGON_STATS`). Clear `ValueError` if not exactly
  one capital input and one target.
- Returns the canonical one-row `'total'` pentagon DataFrame
  (`PENTAGON_STATS` columns), matching the rest of the pricing family.

Thin wrapper over the existing `Pentagon` machinery — no new pricing math.
`Pentagon.solve_obj` now accepts `a=` as well as `p=` (and `p` becomes
keyword-only; it has no other callers). `Portfolio.price_ccoc` is now a thin
alias for `price_pentagon(p=p, ROE=ccoc)` (the cost-of-capital special case);
its output is unchanged.

## 1.0.0a42

### Consolidated fuzz removal into one vectorized utility

The scattered FFT round-off de-fuzz idioms are unified on a single helper,
`utilities.remove_fuzz(data, eps=None)` — accepts an ndarray or a DataFrame,
two-sided (`|x| < eps → 0`, large negatives preserved, so it is correct on
signed/P&L densities), defaulting to machine epsilon.

- Replaces the per-cell `DataFrame.map(lambda x: 0 if abs(x) < eps else x)` in
  `Portfolio.remove_fuzz` and `Aggregate.density_df` (the former now writes the
  float columns back in place; the latter reassigns) — faster, vectorized.
- Replaces four duplicated `np.where(np.abs(x) < eps, 0.0, x)` array copies
  (two in `Portfolio`, two in `Aggregate`) that fed `xsden_to_mwrangler`.
- `ft.recentering_convolution` keeps its looser `2*eps` tolerance via the
  explicit `eps=` argument.

**Numerically inert**: a freeze/check over all 146 test-suite objects matched
within `atol=1e-12`. The one intended change is the moment-fit (MMSE) path,
whose threshold tightens from a stray `1e-16` to machine `eps` (~2.22e-16); it
is not part of the frozen `describe`/`density_df` surface.

**Deliberate carve-outs** (not routed through the utility): the one-sided
`Frequency.pmf` clip (a frequency pmf has no legitimate negatives) and the
plot-cosmetic `1e-15` clip in the reinsurance occurrence plot (looser threshold
plus a `0 → nan` step). Both now carry a comment marking them as such.

## 1.0.0a41

### Renamed: public `reinsurance_*` methods → `reins_*`

The three spelled-out `Aggregate` reinsurance methods now use the `reins_`
abbreviation, matching the rest of the surface (`reins_describe`,
`reins_density_df`, `reins_stats_df`, `reins_audit_df`, `reins_df`, the
`occ_reins`/`agg_reins` layer attributes, the `reins_bucket` config key) and the
house abbreviation style (`sev`, `occ`, `agg`, `freq`, `cv`, `bs`). `reins` is
now the canonical short form for "reinsurance" in identifiers.

**Breaking, no alias** (alpha):

| Old | New |
|---|---|
| `Aggregate.reinsurance_kinds()` | `Aggregate.reins_kinds()` |
| `Aggregate.reinsurance_description()` | `Aggregate.reins_description()` |
| `Aggregate.reinsurance_occ_plot()` | `Aggregate.reins_occ_plot()` |

Pure surface rename — no behaviour, numbers, columns, grammar, spec keys, or
config/env keys change. Docs reference the new names (reference page
auto-regenerates on the next Sphinx build).

## 1.0.0a40

### Fixed: SD/variance for zero-mean signed aggregates

`describe` reported `NaN` for the standard deviation of a mean-zero signed
(P&L) aggregate — e.g. `agg A2 dfreq [3] dsev [-1 1]`, where the severity SD is
1 and the aggregate SD is √3. The stored moments were always correct
(`stats_df` carries `ex2`); only the SD/variance *derivation* was wrong. It
reconstructed `SD = mean × CV`, and `CV = SD/mean` is `NaN` at mean 0, so
`SD = 0 × NaN = NaN`. The irony: `_describe_signed` exists precisely to dodge
unstable CV at mean ≈ 0, but the SD it printed was itself built from CV.

The fix derives variance **directly from the second moment** at all four
computation sites in `distributions.py` — `var = ex2 − mean²` (theoretical) or
`MomentWrangler.central[1]` (empirical FFT), with a `max(var, 0.0)` clamp for
fp dust before `sqrt`. For any positive-mean object `ex2 − mean² == (mean·cv)²`
to fp, so **all normal aggregates are bit-for-bit unchanged**; the behavioural
change is confined to mean ≈ 0 signed objects, where SD goes `NaN → correct`.
`MomentWrangler` (whose `NaN`-at-mean-0 CV is correct) is untouched.

## 1.0.0a39

### Ergonomic tweaks: keyword-only `Underwriter`, signed Lee plot, `density` accessor

Three small quality-of-life fixes, no behavioural change to the numerics:

- **`Underwriter` is now keyword-only** (`Underwriter(*, name=, databases=, …)`).
  This blocks the easy slip of `Underwriter('test_suite')`, which used to bind
  the first positional to `name` and silently *name* the underwriter after the
  database you meant to load. Use `Underwriter(databases='test_suite')`. The
  `repr` now also reports a **`requested`** line (the load *request*
  `self._request`) directly under `knowledge`, distinct from the resolved
  `databases` actually read. All existing call sites already used keywords, so
  blast radius is zero.
- **Signed (P&L) Lee plot fix.** `Aggregate.plot`'s discrete branch anchors a
  zero-mass row just left of the support; it set `loss=0` on that row, which made
  the quantile (Lee) panel draw a spurious vertical segment from `(F=0, loss=0)`
  down to the first point on signed support. The anchor's `loss` now equals its
  own index, so the Lee plot starts cleanly at the true minimum. `Portfolio.plot`
  was unaffected (its limits are already two-sided on signed support).
- **New `density` property** on `Aggregate` and `Portfolio`:
  `density_df.query('p_total > 0')` — the "live" support of the distribution
  with the grid's leading/trailing zero-probability buckets dropped. This is
  usually what you want to inspect. Plain property (recomputed per access) so it
  never goes stale against an `update`/resample.

## 1.0.0a38

### New: knowledge-freeze regression harness (`scripts/freeze_knowledge.py`)

A standalone, dependency-free tool to snapshot and verify knowledge-base
outputs across refactors. `freeze` builds every agg/port program with default
parameters and writes each object's `describe` and filtered `density_df`
(`p_*`/`exeqa_*` columns) to parquet, plus a `_manifest.json` recording the
exact program text, database list, and library version. `check` rebuilds from
the manifest and verifies the recomputed frames match the snapshot to a tight
absolute tolerance (default 1e-12). Importable `freeze_knowledge()` /
`check_knowledge()` functions for Jupyter; argparse CLI for the shell. Default
output goes to an OS-temp dir; pass `--root` for durable storage. Output parquet
is never committed. Additive tooling only — no library code changed.

## 1.0.0a37

### New: `PricingBounds` — cross-pricing ranges and the Gini lens

`bounds.py` gains **`PricingBounds`**: given that a reference risk `X` is
priced to P by *some* distortion, the range of the price of another risk `Y`
over the whole consistent family `G_P = {g : rho_g(X) = P}` (similar-risks
paper). A distortion prices any risk by `rho_mu(Z) = int TVaR_p(Z) mu(dp)`, so
the pricing constraint is one affine condition and the extreme measures are
biTVaRs; the price range of `Y` is the vertical slice at `T_X = P` through the
convex hull of the curve `(TVaR_p(X), TVaR_p(Y))`.

- **Unmatched p-grids resolved** by the *union of breakpoints*: within the
  intersection of an X-atom and a Y-atom both TVaRs are affine in `1/(1-p)`,
  so evaluating at every union breakpoint gives the exact piecewise-linear
  curve — no resampling.
- **Shared engine.** The hull/slice query core (`bounds`, `bitvars`,
  `distortion`, `p_star`, `plot`, `check`) was factored out of
  `AllocationBounds` into a `_HullEngine` base keyed purely on the vertex
  table; both classes are now thin front ends. `AllocationBounds` behaviour is
  unchanged (its test module passes verbatim).
- **TVaR-source adapters.** Each axis is a TVaR source — a discrete risk
  (`Aggregate`/`Portfolio`/pmf `Series`, with exact breakpoints and
  first-principles repricing) or a closed-form `(T, T_inv)` pair. The uniform
  reference `uniform_source()` (`TVaR_p = (1+p)/2`) is the **Gini lens**: as
  the X-source the constraint collapses to the mean-Kusuoka-level condition
  `E_mu[p] = 2P - 1` (`PricingBounds.mean_kusuoka_level`), reading the
  envelope gap of `Y`'s own TVaR curve. Sources are symmetric across both
  axes.
- Entry point: **`Portfolio.pricing_bounds(y_sources, *, a=0, p=0,
  s_floor=1e-14, n_grid=1024)`** — `y_sources` is one risk or a list/dict;
  `a`/`p` cap `min(X, a)` and `min(Y, a)`. Query with `bounds(P)`,
  `bitvars(P)`, `distortion(P, risk, bound)`, `check(P)` (independent
  survival-hinge repricing of each `Y`, plus `total` = price of `X` = P).
- **Conditioning.** Bounded (finite `a`) is exact and well-conditioned —
  `check` reprices to ~1e-12. Unbounded, the upper price bound of a
  heavy-tailed `Y` is genuinely tail-driven and the deep-tail FFT vertices
  (`1-p` below ~1e-7) are discretization noise; cap or raise `s_floor` for
  stable answers (documented on the class).
- The module-docstring comparison table now spans all three classes
  (`Bounds` / `AllocationBounds` / `PricingBounds`).

## 1.0.0a36

### New: `AllocationBounds` — natural-allocation pricing ranges (TODO N5)

`bounds.py` gains **`AllocationBounds`**: given that the portfolio total is
priced to P, the range of natural-allocation premiums to each unit over all
consistent distortions `G_P = {g : rho_g(X) = P}` (similar-risks paper). The
extreme allocations are attained at biTVaRs; each unit's range is the vertical
slice at `T = P` through the convex hull of the curve `(TVaR_p(X), a_i(p))`.
On the discrete FFT grid that curve is *exactly piecewise linear* with
vertices at CDF breakpoints (both coordinates are affine in `1/(1-p)` within
an atom), so the hulls — built from conditional tail expectations via reverse
cumsums — are exact, O(n) per unit, and **P-independent**: construct once,
slice for any premium.

- Entry point: **`Portfolio.allocation_bounds(*, a=0, p=0, units=None,
  s_floor=1e-14)`** returns the `AllocationBounds` object; query with
  `bounds(P)`, `bitvars(P)` (achieving `(p0, p1, w1)`), `p_star(P)` (exact
  TVaR inversion), `distortion(P, unit, bound)`, `check(P)`
  (first-principles repricing audit), `na_grid(p_grid)`, `plot(P=...)`.
- **Bounded totals** via `a=` (asset level, snapped to the grid) or `p=`
  (resolves `a = q(p)`): prices `X ∧ a` with the default states `X >= a`
  collapsed to one atom carrying the *linear* natural allocation
  `a·E[X_i/X | X >= a]` (lifted NA not offered); feasible premium range
  becomes `[E[X ∧ a], a]`. The delicate tail-collapse idiom was factored
  out of `price(allocation='linear')` into a single owner,
  **`Portfolio._collapsed_exeqa(a)`**, now shared by both callers —
  `price` outputs verified byte-identical before/after the refactor.
- The `bounds.py` module docstring now contrasts `Bounds` (IME 2022:
  distortion envelopes for the total, premium fixed at construction) with
  `AllocationBounds` (allocation ranges, premium at call time).
- Diagnostics: `additivity_error` surfaces the `exeqa` noise floor inherited
  from `density_df`; `s_floor` truncates noise-dominated tail vertices.

**Removed:** the long-broken `Portfolio.pricing_bounds`
(`NotImplementedError` since 1.0.0a11) and its `PricingBoundsResult`
dataclass.

## 1.0.0a35

### A bare `Underwriter()` loads no databases by default

**Behavior change.** `Underwriter(databases=...)` now defaults to `None` —
a bare `Underwriter()` starts **empty** (loads nothing) rather than pulling the
configured `build.databases`. This makes "give me an empty underwriter" the
trivial default and removes the surprise of an ad-hoc instance silently loading
the bundled `test_suite`.

- The module-level **`build` still loads the configured `build.databases`**
  (`test_suite` by default) — it now passes that request explicitly, so
  `config.toml`'s `[build].databases` continues to control what `build` knows.
  `discover()` and the built-in examples are unaffected.
- `databases` is therefore no longer config-driven at the *constructor* level
  (only `build` reads config); `log2` / `update` still default from config.
- Nicer help: the "use the configured default" sentinel now renders as
  `<config default>` in signatures (Jupyter `?`, `inspect.signature`) instead of
  `<object object at 0x…>`.

### Fixed: `to_agg` writes entries in dependency order

`to_agg` wrote entries sorted by `(kind, name)`, so severities (`'sev'`) landed
**after** the aggregates that reference them (`'agg' < 'sev'`). Because `.agg`
files load sequentially and named references (`sev.X` / `agg.X` / `dist.X`) must
resolve as each line is parsed, a saved file containing a named-reference would
fail to re-load — breaking the advertised round-trip. Entries are now written in
dependency order (severities and distortions, then aggregates, then portfolios),
so cross-referenced books round-trip. (Deeply chained *combo* distortions that
reference each other by name may still need a manual reorder.)

### `to_agg` gains a write `mode` (`x` / `w` / `a`)

`to_agg(..., mode=...)` mirrors Python's open modes so an existing file is no
longer silently clobbered:

- `'x'` (**new default**, safe) — create a new file, raising `FileExistsError`
  if it already exists.
- `'w'` — overwrite (logged).
- `'a'` — append the selection as a new block at the end, preceded by a dated
  `# added <timestamp>` comment; the block is written in dependency order. (On
  a missing file `'a'` behaves like `'w'`.)

## 1.0.0a34

### De-crufted calibration summary (`distortion_df` / `calibration_df`)

`Portfolio.calibrate_distortions` produced a wide `distortion_df` that mixed the
*per-distortion result* with the *calibration target* — the latter constant
across all five rows, with a part-vestigial `(a, LR, method)` MultiIndex left
over from a removed batch API. Split into two clean frames.

- **`distortion_df`** is now the per-distortion receipt only: index
  `distortion` (ordered categorical, canonical `ccoc, ph, wang, dual, tvar`),
  columns `param_name, param, error, gini_p, area`. `param_name` names the
  family's parameter (`r, a, lam, b, p`); `error` is the premium miss; `gini_p`
  is the comparable normalised shape `= 2∫g−1 = p_equiv`; `area = (gini_p+1)/2
  = ∫g`.
- **`calibration_df`** (new attribute) holds the shared target once: a one-row
  frame leading with the inputs `coc, p, F(a)`, then the canonical pentagon
  octet `L, M, P, Q, a, LR, PQ, ROE`. `ROE` equals the requested `coc` — a
  built-in self-check.
- **Breaking — `Distortion.standard_shape` renamed to `Distortion.gini_p`**
  (attribute, end to end). Verified `= 2∫g−1` for every calibrated family.
- Calibration is one-point (no batch mode); the index no longer implies a batch
  that does not exist. Docs (`2_x_10mins`, `5_x_distortions`) updated to read
  `P` from `calibration_df` and describe the new columns.

## 1.0.0a33

### `Portfolio.price_stand_alone` restored (modernised)

Reinstated `Portfolio.price_stand_alone(dist, p)` — used by the "10 minutes"
guide (its absence was breaking that build). It prices every unit on a
**stand-alone** basis (each backed by its own VaR(`p`) capital, the distortion
applied to its own loss distribution) and contrasts that with the diversified
whole.

- Built on the existing pricing primitives: each unit's column comes from
  `Aggregate.price` (a unit is just an `Aggregate`), the `total` column from
  `Portfolio.pricing_at` — so stand-alone and allocated pricing share one code
  path rather than re-deriving the integral by hand.
- Every row routes through the canonical pentagon
  (`complete_pentagon`), so the readout carries the full octet
  `L, M, P, Q, a, LR, PQ, ROE` in one consistent order. The `sum` row adds the
  amounts across units and re-derives the ratios.
- Canonical orientation, matching `pricing_at`: the eight stats are the
  columns, one row per entity (units, `total`, `sum`) under a `(method, unit)`
  MultiIndex. Transpose (`a.T`) for the traditional stat-down-the-side exhibit.
- Argument checking: `p` must be a probability in `(0, 1)`; `dist` must be a
  `Distortion` or the name of a calibrated one (clear `TypeError` / `ValueError`
  / `KeyError` otherwise). NumPy-style docstring added.

## 1.0.0a32

### Legible `Underwriter` database loading

The `Underwriter` knowledge-base loading surface — historically one overloaded
`databases` attribute plus the near-identical `read_database` / `read_databases`
methods, a fragile `len(knowledge)==0` lazy-load proxy, and a DataFrame-backed
store — was rebuilt around one explicit pipeline. The resolved knowledge is
unchanged (`build` loads the same `test_suite`); only the plumbing changed.

- **Dict-backed store with provenance.** The knowledge base is now a flat
  `{(kind, name): ParsedProgram}` dict; the `(kind, name)`-indexed DataFrame is
  built on demand for `knowledge` / `discover` (now with a `source` column).
  Each entry carries a `source` tag — the originating file `Path`, or
  `'session'` for an in-session `build(...)` — answering "where did this come
  from?".
- **One request, one resolver.** The constructor `databases=` argument is the
  *request* (stored privately); the public `databases` attribute now reports the
  resolved file `Path`s **actually loaded**. A single glob-aware resolver
  handles it: `'default'` / `'user'` / `'all'` are predefined globs, anything
  else is a path or glob resolved across cwd → `~/.aggregate` → bundled
  (literals first-match-wins; globs union across all three). `databases=['cat_*']`
  now works. The removed `'site'` token is no longer special-cased.
- **Honest lazy load + clear verbs.** An explicit `_loaded` flag replaces the
  entry-count proxy. New/renamed methods: `load(request=None)` (the one load
  verb; configured-once when `None`, additive otherwise), `reload()` (reset to
  as-created and re-read), `resolve_databases(request)` (preview without
  reading), `available_databases()` (discover `.agg` files on disk).
  **Breaking:** `read_database` / `read_databases` are renamed outright to
  `load` (no deprecated aliases) — `build` was effectively the only caller.
- **Consistent error policy.** A literal file named in an explicit `load(path)`
  that is missing raises `FileNotFoundError`; the configured request and any
  empty glob only warn.
- **Save / export.** New `to_agg(path, pattern='.*', kind='all',
  source='session')` writes selected entries' DecL back to a `.agg` file
  (round-trip re-loadable), to `~/.aggregate` unless the path is absolute. The
  default `uw.to_agg('mybook')` saves everything built this session — making the
  class docstring's "persist to and from `.agg` files" claim true.
- **One preprocessing owner.** `read_database`'s ad-hoc whitespace regexes are
  gone; `UnderwritingLexer.preprocess` now owns continuation/indent folding (a
  newline followed by any tab or spaces is a continuation), covering the tabbed
  and space-indented Portfolio layouts alike.

## 1.0.0a31

### One canonical pricing readout (the "pentagon")

Every pricing method emits the same eight accounting quantities — the amounts
`L` (loss), `M` (margin), `P` (premium `= L + M`), `Q` (capital),
`a` (assets `= P + Q`) and the ratios `LR = L/P`, `PQ = P/Q`,
`ROE = M/Q`. These used to be built independently by each method, disagreeing
on naming, order, completeness and dtype. They are now a single canonical
contract owned by `aggregate.pentagon`:

- **One name, one order.** `M/Q` is always `ROE` (with `CoC` documented as
  the synonym); the canonical order is `[L, M, P, Q, a, LR, PQ, ROE]`
  (`pentagon.PENTAGON_STATS` / `PENTAGON_DTYPE`). `Portfolio.pricing_at`,
  `price`, `price_ccoc`, `analyze_distortion`, `analyze_distortions` and
  `Aggregate.price` all route through one `complete_pentagon` helper, so the
  derivation `a = P + Q; LR = L/P; …` lives in exactly one place.
- **Consistent orientation.** Stats are always columns, one row per priced
  entity; any descriptor columns lead and the pentagon octet is the trailing
  eight (`df.iloc[:, -8:]`).
- **``analyze_distortion` audit fixed.** Its `audit_df` is now a one-row frame
  in that shape — `dname`/`dshape` lead, the full octet trails. *(Shape
  change: previously a column-oriented frame that omitted ``PQ` and mixed the
  metadata into the stat index.)* `price_ccoc` now emits `ROE` instead of
  `COC`.
- **Pentagon objects (additive).** New `Portfolio.pentagon_at(distortion, p|a, line)` returns a single-row `Pentagon` — an eight-vector with named
  attributes and provenance that completes any soluble partial input via
  `Pentagon.solve` (e.g. give it `P`, `L` and `a` or `Q`, get the
  rest). No existing method changed its return type.

## 1.0.0a30

### User-editable configuration file

The "secret bits" that used to be hard-coded literals — the default `log2`,
the default database, the reinsurance / discrete-severity bucketing scheme, the
bucket-sizing percentile, the output-window coverage, and the validation
tolerances — are now read from a single, optional, hand-editable **TOML** file
at `~/.aggregate/config.toml`. Values layer lowest-to-highest as
**built-in defaults → config file → ``AGGREGATE_*` environment variables →
explicit ``build(...)`` / ``Underwriter(...)` keyword arguments**, so a call
argument always wins and two fresh installs with no file behave identically.

New surface (all on the module-level `build` and any `Underwriter`):

- `build.write_default_config()` writes an annotated, **fully-commented**
  template to `~/.aggregate/config.toml` (inert until you uncomment a key);
- `build.show_settings()` prints every setting **and its source**
  (`default` / `config` / `env`);
- `build.reload_settings()` re-reads the file/environment after an edit and
  refreshes the module `build` in place;
- `aggregate.get_settings()` returns the resolved, immutable `Settings`
  snapshot (read once per session);
- `repr(build)` / `Underwriter` info gains a `config` line reporting the
  active file and how many settings are overridden;
- escape hatches: `AGGREGATE_CONFIG=/path` relocates the file,
  `AGGREGATE_CONFIG=none` ignores it (reproducible runs). Unknown keys,
  sections, and `AGGREGATE_*` variables warn loudly rather than silently
  no-op.

The tunable defaults and the path names now live in the new leaf module
`aggregate.config`; `aggregate.constants` is slimmed to the `Validation`
flag enum, `DefectiveDistributionWarning`, and the structural reinsurance
column labels.

**Breaking changes**

- **Minimum Python is now 3.11** (the config reader uses the standard-library
  `tomllib`; no new third-party dependency). The 3.10 classifier is dropped.
- **``recommend_p`` is renamed to ``bucket_sizing_p`** everywhere — the
  `build` / `build_many` / `update` keyword, the `hints{...}` key, and
  the underlying constant. There is **no alias**; update any call sites.
- The tunable names that used to live in `aggregate.constants` (e.g.
  `VALIDATION_EPS`, `VALIDATION_NOISE`, `RECOMMEND_P`,
  `REINS_BUCKET_DEFAULT`, `DSEV_BUCKET_DEFAULT`) moved to
  `aggregate.config` and are **not** re-exported; read them from
  `get_settings()` (e.g. `get_settings().validation.noise`).
- A bare `Underwriter()` now takes its `log2` / `databases` / `update`
  defaults from the configured `[build]` section (this unifies the old
  10-vs-16 `log2` split with the module `build`); pass `databases=None`
  to load nothing.

Scope: this is Phase 1 — the `[build]`, `[discretization]`,
`[validation].eps` / `.noise`, and `[multivariate].window_nines` settings.
Plot styling (`[plotting]` / `.mplstyle` override) and the numerics-pending
validation floors (`aliasing_ratio`, `exeqa_noise_floor`, `ft_noise_floor`)
land in a later phase.

## 1.0.0a29

### Tail-thickness classification for aggregates and portfolios

Every `Aggregate` and `Portfolio` now reports an ordered tail-thickness
class on a five-rung scale — `bounded` \< `super-exponential` \<
`exponential` \< `subexponential` \< `power-law` — plus a separate
log-concavity flag. The class is derived by deterministic family lookup keyed on
the frequency and (scipy) severity families, with the aggregate rung given by
the heavier of frequency and severity (`max`): for a subexponential-or-heavier
severity the single-big-jump principle makes the aggregate inherit the severity
class; for light severity the heavier decay rate wins. New surface:

- `Aggregate.tail_class` → a `(freq, sev, agg)` named triple of
  `TailClass` rungs; `Severity.tail_class` → the component rung;
- `Aggregate.tail_description` (three aligned lines) and
  `tail_explanation` (one sentence, with the power-law tail index `alpha`
  and infinite-variance / infinite-mean flags) — both also shown in `info`;
- `Portfolio.tail_class` / `tail_description` / `tail_explanation` report
  the **worst-of** unit aggregate (correct under independence) and name the
  driving unit(s).

`bounded` is now a **derived** view of the classifier (`bounded` iff the
aggregate tail class is `BOUNDED`) on `Aggregate`, `Severity`, and
`Portfolio`; the certify setter (`obj.bounded = True`) and the lifted
natural-allocation admissibility guard are unchanged. The bounded-support tables
moved to the new leaf module `aggregate.tail` (re-exported from
`distributions` for back-compat). `tail.py` is the single source of truth.

Scope: this is Phase 1 — exact, deterministic, spec-only (`bounded` resolves
before `update()`). Unrecognised or numeric-only families (histogram, meta,
spliced) classify as `undetermined`; the numeric density-tail estimator that
will fill those in (and set the aggregate's log-concavity) is deferred to a
later Phase 2.

## 1.0.0a28

### Mean-preserving (`linear`) bucketing for discrete severities

A new `dsev_bucket` setting controls how discrete-severity atoms (`dsev` /
`dhistogram` / `fixed`) are placed onto the model grid during
discretization, mirroring the existing reinsurance `reins_bucket`:

- `'linear'` (the **default**) splits each off-grid atom's mass across its two
  bracketing buckets, so the discretized first moment equals `Σ xₖ pₖ`
  exactly (mean-preserving);
- `'nearest'` snaps each atom to its closest bucket (the historical
  behaviour), biasing the discretized mean by up to `bs/2` per atom.

This matters when atoms are off-grid — e.g. a severity given as a sample of
empirical losses with a non-integer `bs`. The common integer-atom, `bs = 1`
case (a die, fixed losses) is on-grid, where the two schemes coincide, so it is
unchanged. Pass it as a `build` / `update` keyword:

    a = build('agg Off dfreq [1] dsev [0.3 1.7 2.4] [.5 .3 .2]', bs=0.5)
    # a.dsev_bucket == 'linear'; discretized mean == 0.3*.5 + 1.7*.3 + 2.4*.2

Scope: Phase 1 covers *unlayered* discrete severities (the empirical-sample use
case). A *layered* discrete severity discretizes via the cdf-difference and so
behaves as `'nearest'` regardless of the setting. `info` shows
`dsev_bucket` when a discrete component is present. The two discrete cases in
the baseline corpus (`Sym.Dice`, `Port.Bodoff`) shift at the floating-point
floor toward exact mass placement and were re-captured.

## 1.0.0a27

### Distortion DecL syntax is a flat number list; the parser stops knowing kinds

The DecL distortion form is now uniformly `distortion NAME kind n1 n2 ...` — a
flat list of the kind's parameters, no brackets:

    distortion D ph 0.9
    distortion D bitvar 0.9 0.99 0.5      # p0 p1 w1
    distortion D power 0.01 1.0 2         # x0 x1 alpha

Previously the parser carried a hand-maintained `_distortion_spec` table that
re-encoded every kind's parameter names (duplicating what the `Distortion`
subclasses already declare) and reached into `spectral` for domain facts. That
table is gone. Each subclass now declares its DecL parameter order in a
`decl_params` class attribute, and a single `Distortion.decl_spec` maps the
number list onto the kind's natural keyword arguments — one source of truth,
and adding a distortion kind no longer touches the parser.

- `ccoc` takes the return `r` (`distortion D ccoc 0.25`), not the discount.
- `wtdtvar` (parameter *vectors*) and the `minimum` / `mixture` combinators
  (which take distortion *references*) have no flat-number form and raise a clear
  error if written that way; construct them in Python or via the combinator
  syntax. The bracketed `kind shape [list]` form is removed.

## 1.0.0a26

### Honest discrete severity — exact moments, no more `rv_histogram` hack

A truly discrete severity (`dsev`, `fixed`) used to be represented by
*abusing* `scipy.stats.rv_histogram` — a continuous, piecewise-linear-CDF
object — forced to mimic a step function by pouring each atom's mass into a
tiny `2**-d`-wide sliver to its left (sized by a float-resolution helper,
`max_log2`). It worked, but it was a representation lie: it produced quantile
artifacts (`ppf(0.5) = 149.9999999992` instead of `150`) and — because the
sliver width *scales with atom magnitude* — it quietly degraded the **moments**
of large-valued discrete books.

`SeverityDHistogram` / `SeverityFixed` now back `self.fz` with a small,
honest `_DiscreteRV`: exact right-continuous step `cdf`/`sf`, `pdf = 0`,
and exact `ppf`/`isf`/`support` (no trailing-9s artifacts; `rvs` returns
exact atoms).

- **All discrete moments are now exact**, computed as finite sums over the
  atoms — unlimited, limited, *and* layered. Previously every discrete moment
  (even the unlimited mean of a fair die) was computed by numerical
  isf-integration and came back as `3.4999999995` rather than `3.5`; layered
  discrete moments integrated a step function by quadrature, which was both
  inexact and fragile. A discrete severity never routes its moments through the
  numerical path anymore.
- **Aggregate density is unchanged** — the FFT samples `cdf`/`sf` at
  half-bucket edges, which never coincide with an atom, so the discretised
  density is bit-for-bit identical to before. The improvement is confined to
  reported moments and quantiles, which become *more* correct (the baseline /
  golden regression snapshots were re-captured to record the exact values).
- `max_log2` is now unused (kept for one release; slated for removal).

See `dev/done/plan-discrete-severity-fz.md`.

## 1.0.0a25

### `note{}` is now pure text; build settings move to `hints{}`

`note{...}` used to do double duty: free-text annotation *and* a
`key=value;` side-channel for build settings (`log2`, `bs`, …). That
overloading meant any `=` in note prose — e.g. `note{... needs x_min<=-6}` —
was mis-read as a keyword argument and crashed the build. Notes are now **pure
annotation**; a dedicated `hints{...}` clause carries build settings.

- **Syntax.** `hints{key=value; key=value}`, e.g.
  `agg A 5 claims sev lognorm 10 cv 2 poisson hints{log2=18; bs=1/64}`.
  `note{}` and `hints{}` are both optional and order-free (at most one of
  each); `hints` is allowed everywhere `note` is (agg, sev, port).
- **Caller always wins.** Explicit `build(...)` keyword arguments override
  in-program `hints` uniformly — including `recommend_p` (fixing the old
  quirk where a note's `recommend_p` overrode the caller).
- **Forgiving.** Values are inferred generically (int / float / `a/b`
  fraction / `True`/`False` / str). Unknown keys warn and are dropped;
  a duplicate key warns and the last value wins; a malformed clause warns and is
  skipped — a bad hint never crashes the build.
- **Deprecation.** A `note{}` that still looks like it carries `key=value`
  settings emits a one-time warning (the note is treated as pure text).
- **Migration.** The bundled `test_suite.agg` / `test_decl.agg` corpora moved
  their settings-in-notes into `hints{}`; built grids are unchanged. See
  `dev/plan-note-parse.md`.

## 1.0.0a24

### The `multivariate` keyword — copula-coupled bivariate aggregates

A single event can drive two correlated perils — wind *and* flood, attritional
*and* large — and you want the **joint** law of the two aggregates, not just two
marginals. `multivariate` makes that a first-class object: two component
`agg` / `pnl` severity factories whose per-claim severities are coupled by a
**copula**, accumulated by a **shared** outer frequency through a 2D FFT. It
subsumes the `1.0.0a20` `occ_bivariate` backbone into a modelled,
DecL-declared facility. See `dev/plan-multivariate.md`.

- **Syntax.** :

      multivariate Cat 25 claims
          agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
          agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
          copula gumbel 0.4          # Kendall tau = 0.4
          mixed gamma .2             # shared mixing -> common shock

  The shared count (`25 claims`) and trailing frequency own the event count;
  each component's `dfreq [0 1] [p0 p1]` is the per-event trigger probability,
  so its *aggregate* is the per-event severity `g_i = (1−p_i)δ₀ + p_i f_i`.
  The `copula` clause is **optional** — omitted (or `copula independent`)
  means the independence copula, where the only dependence is the shared count.

- **Copulas, à la ``Distortion`** (new `aggregate.copula.Copula`). A registry
  / factory hierarchy — `Copula('gumbel', 0.4)` dispatches on the name — with
  **normal** (Pearson ρ), **gumbel** (Kendall τ, upper tail), **clayton**
  (Kendall τ, lower tail), **fgm** (Spearman ρ_s), and **independent**. Each
  takes its *natural* dependence parameter and converts internally; `t` (the
  two-parameter kind) is deferred.

- **Discrete Sklar construction.** The joint per-claim severity is the copula
  rectangle mass `S[i,j] = C(G1[i],G2[j]) − C(G1[i−1],G2[j]) − C(G1[i],G2[j−1]) + C(G1[i−1],G2[j−1])` over the marginal CDF breakpoints; `S` has the
  component severities as exact marginals (atoms handled as jumps in `G`). The
  joint aggregate is `iFFT2(freq_pgf(N, FFT2(S)))` — the ordinary compound FFT
  with 1D transforms replaced by 2D, valid because `freq_pgf` is elementwise.
  Marginalising one axis reproduces that component's standalone aggregate.

- **``pnl` axes in v1.** A `pnl` component contributes its *loss* severity to
  the copula+FFT, then its premium becomes a **per-axis affine** (reflect +
  shift) applied to that tensor axis *after* the FFT — the 1D `_apply_agg_affine`
  relabel lifted to one axis. The affine commutes with marginalisation (marginal
  = the standalone `pnl`) and flips the loss-loss copula dependence to the
  correct profit-loss sign.

- **Reporting.** `MultivariateAggregate` exposes `marginals` / `moments`
  (`E[A0ⁱ A1ʲ]`) / `corr` and the properties `density_df` / `stats_df` /
  `describe` / `info`, a two-panel `plot` (joint per-claim **severity** on
  the left, joint **aggregate** on the right), and a `help` introspector. The
  realised output correlation is reported
  **alongside** the copula τ — compounding attenuates per-claim dependence, and a
  shared *mixing* frequency adds common-shock dependence on top (so even the
  independence copula gives a positive baseline correlation from the shared
  count).

- **Net/ceded as a first-class mode.** The joint per-occurrence (ceded, net)
  law of a *reinsured* aggregate is now a `MultivariateAggregate` in
  **``netceded` mode** — same 2D-FFT engine, a different (comonotone)
  per-claim severity builder. Two entry points: the DecL `netceded <agg with occurrence reinsurance>` statement, and `Aggregate.occ_bivariate()`, which
  now **returns** that object (so it gets the full `describe` / `stats_df` /
  `info` / `plot` / `help` surface, not a bare container). The axes are
  `Ceded` / `Net`; marginalising reproduces the univariate occurrence
  ceded / net aggregates, and `E[Ceded] + E[Net]` equals the gross mean.

- **Module move.** `BivariateDistribution` (the internal 2D density
  container) and the `size_axis` / `scatter_bivariate` / `build_netceded_joint`
  helpers live in `aggregate.multivariate`; the old `aggregate.bivariate`
  module is removed (import from `aggregate.multivariate`).

- New `tests/test_multivariate.py` (37 cases); DecL mirrored in
  `test_decl.agg` (section MV). The `t` copula, ≥3-variate `rfftn` path,
  and a `MultivariatePortfolio` are scoped as later stages in the plan.

## 1.0.0a23

### The `pnl` keyword — premium-minus-loss aggregates

A profit is premium minus loss, and the premium is collected **once for the
book**, not once per claim. `pnl` makes that a first-class object — a sibling
of `agg` that builds an ordinary loss aggregate and applies an
aggregate-level affine wrapper `PnL = premium − A`. It is the natural producer
of payoff-typed objects and goes anywhere an `agg` goes, so a `port` of
`pnl` lines is a book-level underwriting-result distribution (via the signed
combine landed in `1.0.0a22`). See `dev/done/plan-pnl-premium.md`.

- **Syntax.** `pnl NAME <premium> prem - <loss-exposure> <sev> <freq> …`.
  Three exposure heads: `70% lr` (binds to the stated premium,
  `E[loss] = premium·lr`), `10 claims` (frequency-driven), `85 loss`
  (expected-loss-driven). The premium **vectorises** like an `agg` exposure —
  `pnl X [100 200 100] prem - .8 lr [1000 2000 5000] xs 0 sev …` shifts by
  `Σ premium` and reports the total P&L only.
- **Once for the book, not per claim.** A constant inside `sev`/`dsev`/
  `ssev` is multiplied by the claim count; the `pnl` premium is a single
  deterministic shift. `pnl P 100 prem - 5 claims …` (mean `100 − E[A]`) is
  deliberately **not** `agg 5 claims ssev 100 - …` (mean `5·(100 − E[X])`).
- **No new numerics.** The loss FFT, its validation, and every ordinary
  aggregate are byte-for-byte unchanged (gated behind `agg_reflect` /
  `agg_shift` defaults). The affine is a pure grid relabel of the finished
  density: `mean → premium − E[A]`, `sd` unchanged, `skew → −skew`;
  `ftagg_density` is rebuilt in the combine convention so a book of `pnl`
  units convolves with no combine-side change. The P&L window is a tight,
  mass-centred two-sided window (`estimate_agg_window` on the affine moments).
- **Signed-aware ``describe`: SD instead of CV.** `CV = sd/mean` is
  meaningless as the mean → 0 (a P&L straddling break-even), so for **any**
  signed object — a `pnl` *or* a `ssev` / negative-`dsev` aggregate — the
  moment table now shows an **SD trio** (`SD | Est SD | Err SD`) instead of CV.
  This also cleans up the `1.0.0a22` signed-portfolio `describe`. Non-signed
  output is unchanged.
- **P&L readout.** `info` reports `premium` / `E[loss]` / `E[margin]` /
  `loss ratio` / `P(loss)` (`= P(PnL < 0)`, read straight off the signed
  density). `value_type` is set to `payoff` — finally giving that member a
  job (consumed by the pricing plan).
- New `tests/test_pnl.py` (15 cases); DecL mirrored in `test_decl.agg`
  (section PnLprem). Reinsurance gross/ceded-premium P&L and general aggregate
  algebra are split out to `dev/TODO-Remember.md` (items 6c / 6b).

## 1.0.0a22

### Portfolio combine on signed (profit/loss) support

Second half (`Portfolio` scope) of the negative-x work in
`dev/plan-negative-x-port.md` — the *combine*. A portfolio of independent
signed (P&L) units now aggregates correctly onto a shared signed grid, so a
book that straddles 0 is a first-class object alongside the single-unit P&L
landed in `1.0.0a21`.

- **Window-aware combine.** Each unit keeps its **own** optimal signed window
  `[x_min_k, x_max_k)`; the portfolio insists only on a shared `bs` /
  `log2` / `padding`. The FFT product is origin-at-0 because each unit's
  `ftagg_density` is independent of that unit's `x_min` (the output roll
  hits the density, never the transform), so the units multiply correctly and
  the total is placed on `[x_min_tot, ...)` by a single F2 `np.roll`. This
  replaces the old truncating `ift` (which silently dropped the wrapped
  negative tail) — `p_total` now conserves mass on signed support.
- **Driven on own grids.** Units are updated on their own signed windows
  (not a 0-based grid), so each unit object stays internally correct (no
  spurious deficit warning, right moments / `describe` / `plot`) — a strict
  improvement in instrumentation over a shared-origin drive.
- **Coarsen-to-fit bucket.** New signed-aware `Portfolio._bs_window` (a thin
  wrapper on `best_bucket`, recorded in a unit-indexed
  `Portfolio._bs_window_df`) sizes the shared grid: the summed support is
  wider than any unit's but `2**log2` is capped, so `bs` is the *coarser*
  of the RMS recommendation and the fit floor `W_tot / N` (buy the space,
  avoid aliasing). A fine-lattice unit coarsened by the shared grid surfaces
  its own per-unit deficit warning rather than failing silently.
- **density_df** `loss` / `p_total` / `p_{line}` / `F` / `S` are
  correct on signed support, and hence so are `q` / `var` / `tvar` (the
  index-agnostic `make_var_tvar` needs no change). `plot` is signed-aware
  (`_limits('range')` returns a two-sided window so the negative tail is no
  longer clipped), and `info` reports the realised signed window.
- **Pricing deferred.** Pricing / allocation columns (`add_exa` and
  everything it writes, distortion pricing, `value_type` consumption) assume
  a `loss ≥ 0` axis and are split out to
  `dev/plan-portfolio-neg-x-pricing.md`. A signed portfolio routes through
  the `add_exa=False` branch (F/S only); passing `add_exa=True` warns and
  falls back rather than emitting wrong numbers.
- The non-negative path is **byte-for-byte unchanged** — every signed path is
  gated behind `Portfolio._signed()`. The `build_many` Portfolio branch no
  longer pre-computes `best_bucket` (no back doors: `update` routes
  `bs=0` through `_bs_window` itself, mirroring the Aggregate fix). New
  `tests/test_negative_x_port.py` (14 cases); DecL mirrored in
  `test_decl.agg` (section PortPnL).
- **DecL: ``shift - dist` severity (premium minus loss).** A constant minus a
  distribution now parses as the natural P&L reading, e.g.
  `ssev 100 - lognorm 80 cv .2` — premium `100` minus a lognormal loss
  (severity mean `20`). It is exactly `-1 * X + 100` (reflect the
  distribution, then shift), composes with a scale (`100 - 2 * lognorm ...`),
  and like all reflection needs `ssev` to keep the signed support (plain
  `sev` clamps the sub-zero tail). One grammar rule (`numbers MINUS sev1`)
  \+ transformer; the unambiguous whitespace-separated minus means no existing
  program changes.

## 1.0.0a21

### Negative-support (profit/loss) severity and the output window

First half (`Aggregate` scope) of the negative-x work in
`dev/plan-negative-x-agg.md`. A *profit is a negative loss*, so an aggregate
can now live on a signed grid, making profit/loss (P&L) distributions a
first-class object.

- **Signed severity (F1).** Severity may take negative values -- a profit is a
  negative loss. The DecL severity family is now:

  - `sev` -- continuous, **clamps** its sub-zero tail at 0 (unchanged);
  - `dsev` -- discrete, **never clamps**; a negative atom (e.g.
    `dsev [-2 5] [.5 .5]`) auto-signs the aggregate, so
    `build('agg PnL 1e6 claims dsev [-1 10] [15/16 1/16] poisson')` works
    directly;
  - `ssev` -- **new**: continuous, **never clamps** -- the signed / P&L
    sibling of `sev` (e.g. `ssev 50 * norm + 10`).

  Signedness is a property of the severity declaration (`Severity.signed`),
  recorded at parse time -- which is what lets the analytic moments (and hence
  the automatic window) be correct before any FFT. The `update(..., signed=)`
  argument remains as an override. The separate `value_type` member
  (`'loss'`/`'payoff'`, default `'loss'`) records the *pricing* sign
  convention; it is **orthogonal** to `ssev` (signed does not imply payoff),
  inert for the distribution, and consumed only at the pricing layer.

- **Output window (F2).** `update(x_min=...)` places the aggregate on a window
  `[x_min, x_min + (2**log2)*bs)`; `x_min` may be negative. The default
  `x_min='auto'` resolves to `0` for an ordinary aggregate and to an
  automatic two-sided window for a signed one, so a P&L just works from
  `build` with no extra argument. The window is estimated from the analytic
  moments (new `estimate_agg_window(m, sd, skew, p)` -- reflected
  shifted-lognormal / -gamma fits with a symmetric/normal fallback; takes the
  standard deviation directly so the mean-zero case works), so a tight far-from-0
  lump (e.g. a Poisson(10^6) P&L concentrated near a *negative* mean) uses a
  small `bs` over a narrow window rather than paying for `[0, mean]`. The
  placement is a relabelling (single `np.roll` on the padded FFT buffer),
  exact for random as well as fixed frequency.

- **Bucket + window estimator.** `update` now runs up to three sizing methods
  and records them in an expert-inspectable `Aggregate._bs_window_df`, then
  selects: **exact_discrete** (a `dfreq`/`fixed` × `dsev` on an integer
  lattice has exact finite support -- `bs=1` and a minimal `log2`) \>
  **bounded_small** (a bounded severity with a small claim count -- window
  `[0, N_hi·s_max]` from a high frequency quantile) \> **moment** (the legacy
  3-moment sizing, reproduced exactly for non-negative aggregates;
  `estimate_agg_window` for signed). `log2` is a cap; pinning `bs` lets you
  keep full control of the grid.

- **Severity reporting moved to** `Aggregate.sev_density_df` (its own grid
  `xs_sev`). A windowed/signed aggregate and its severity no longer share a
  grid, so `p_sev`/`F_sev`/`S_sev`/`log_p_sev` left `density_df` for
  the new frame; `plot`, `q_sev`, `tvar_sev` and the error analysis are
  re-sourced. `q`/`tvar`/`var` work on signed support; `plot` is
  signed-aware (axes span the negative support). `info` renders discrete
  severities by their support (`atoms [-2 5]`, shortened for many) instead of a
  spurious `5 xs 0` layer, shows `window` / `value_type` / `signed severity`, and warns when the severity falls outside the output window. The
  default 0-based, non-negative path is byte-for-byte unchanged.

- Internals: `validate_discrete_distribution` gains `allow_negative`
  (`dfreq` still clamps claim counts; signed `dsev` preserves negatives);
  `SeverityDHistogram` places negative atoms correctly; signed severities use
  identity layering (no `x<0 -> 0` clamp) and raw moments. Signedness is the
  declaration only -- the `ssev` keyword is the **only** DecL change, and there
  is no `signed=` argument. New `tests/test_negative_x.py` (28 cases).

- **Deferred to the Portfolio half** (`dev/plan-negative-x-port.md`):
  portfolio combine on signed support, the full `Portfolio.density_df` column
  audit (esp. the price column / `add_exa`), and distortion/pricing
  consumption of `value_type`. The `ft.py` recentering helpers are not yet
  refactored to call the core path (follow-up).

## 1.0.0a20

### Joint (ceded, net) occurrence distribution via 2D FFT

- New `Aggregate.occ_bivariate(...)` returns a `BivariateDistribution`
  (new submodule `aggregate.bivariate`; submodule access only) holding the
  **joint** law of the aggregate occurrence ceded `C` and net `N` losses
  under an occurrence reinsurance program. The two margins are already
  available individually (`reins_density_df['p_agg_ceded_occ' | 'p_agg_net_occ']`);
  the joint law — their correlation, co-moments, reinsurer-vs-cedent
  dependency — was not, and the random claim count means it does not factor.
- The mathematics is the ordinary compound-distribution FFT with the 1-D
  transforms replaced by 2-D transforms: per claim, `(c(X), n(X))` lies on
  the line `c + n = X`, so placing the gross severity mass there builds a
  bivariate severity `S` and the joint aggregate density is
  `iFFT2(freq_pgf(n, FFT2(S)))` — valid because `freq_pgf(n, z)` is
  elementwise in `z`. Occurrence only (the aggregate-cover bivariate is
  degenerate). Per-axis bucket / window sizing is auto-derived from the
  univariate margins (with `bs_ceded` / `bs_net` / `log2_ceded` /
  `log2_net` overrides) and the net/ceded mass is scattered onto the 2-D grid
  by the active `reins_bucket` scheme.
- `BivariateDistribution` provides `.marginals()`, `.moments(max_order)`
  (mixed raw moments `E[C^i N^j]`), `.corr()` (Pearson; positive — a random
  count couples ceded and net), `.contour()`, and rich reprs. The marginals
  reproduce the univariate occurrence aggregates and the anti-diagonal `C+N`
  reproduces the gross aggregate, giving exact validation targets; auto-sizing
  generally yields a *finer* (more accurate) ceded grid than the model grid.
- New `tests/test_reins_bivariate.py` (31 cases); DecL cases added to
  `test_decl.agg` (section Z). An experimental docs subsection is pending a
  manual rebuild.

## 1.0.0a19

### Rationalized reinsurance reporting (Aggregate + Portfolio)

- Three new reinsurance objects on `Aggregate` replace the old fragmented
  surface:
  - `reins_density_df` — per-bucket gross/ceded/net densities with
    **consistent columns** regardless of which stages are configured (a
    missing stage contributes the no-cession values). Renamed from the legacy
    `reinsurance_df`: `p_agg_gross_occ → p_agg_gross` (the true gross
    aggregate) and the old `p_agg_gross` → `p_agg_subject` (the
    aggregate-cover input).
  - `reins_stats_df` — a per-layer layering summary (empirical, model-grid).
    Columns `(view, layer)` with `view` ∈ `occ|agg`: `Gross` (always),
    then per occurrence `layer.1` … and the `Ceded` / `Net` totals, then
    the aggregate layers and their `Ceded` / `Net` (no `Subject` column —
    it is the column flagged `output` below). **Occurrence layers are
    conditional** on reaching the layer: frequency is the penetrating count
    `n·P(X>attach)` and severity is the unconditional layer severity divided
    by `P(X>attach)` (so the layer aggregate mean is unchanged and aggregate
    layer means sum to `Ceded`); the `agg` row is the layer's actual FFT
    aggregate. `Ceded` / `Net` totals are unconditional (`Ceded` sev +
    `Net` sev = `Gross` sev). The aggregate block leaves `freq` / `sev`
    NaN (they don't combine). Meta rows: `share` / `limit` / `attach`
    (`Gross` = claim-count-weighted policy terms, share 1; occ `Ceded` =
    share-placed sum of limits, min attachment), `pr_attach` / `pr_detach`
    (ground-up exposure probabilities that the underlying loss attaches /
    exhausts the view — from the underlying severity `fz`, since the modeled
    severity is conditional and reads 0 at the policy cap), `pr_loss`
    (P aggregate \> 0), `lol` (loss on line = layer agg mean / placed limit),
    and `output` (0/1, marks each stage's output view). Plus
    `(freq|sev|agg, ex1|ex2|ex3|mean|cv|skew)` (`ex1` duplicates `mean`
    for `filter(regex=...)`).
  - `reins_describe` — the daily-driver per-stage summary, sharing the same
    **eight columns as** `describe` (`EX | Est EX | Change EX | CV | Est CV | Change CV | Sk | Est Sk`) and mirroring its **economic view**: `EX` /
    `CV` / `Sk` hold the *theoretic reference* — the leading view's exact
    pre-bucket moments (`Gross` for the occurrence block, `Subject` for the
    aggregate block) — held constant down each component; `Est *` is the
    per-view model output; and `Change = (Est − reference) / reference` reads
    two ways off one arithmetic: on the leading (Gross/Subject) row it is the
    numerical validation / rebucketing error (~0 under `linear`), and on the
    ceded / net rows it is the % impact of the cession on that moment. Follows
    the gross/subject convention — the occurrence block leads with **Gross**,
    the aggregate block leads with **Subject**. Frequency is reported
    *unconditionally* on the `Est` basis (mean `E[N]` only, so `freq × sev == agg` per view; cv / skew `NaN`) — consistent with `reins_stats_df`,
    whose conditional basis is confined to the per-layer `layer.k` columns; the
    leading `gross` row's `Est` frequency is left `NaN` to mirror
    `describe`. The `view` / `component` index labels are lower-case to
    match the other frames.
- New **Portfolio** reinsurance reporting (previously absent):
  `reins_density_df` / `reins_stats_df` / `reins_describe` give the
  end-to-end gross/ceded/net of the portfolio aggregate, convolving the
  per-unit gcn aggregate marginals under the existing independent-FFT
  machinery (means add: portfolio total = sum of unit means per view). All
  three return `None` when no unit cedes.
- Removed the redundant/confusing legacy objects: `reinsurance_df` (renamed),
  `reinsurance_audit_df`, `reinsurance_report_df`,
  `reinsurance_occ_layer_df`, the persistent `occ_reins_df` /
  `agg_reins_df` members, and the per-layer `_reins_audit_df_work` engine.
  The vestigial `F_*` (CDF) columns are dropped from the per-stage engine
  frame (the debug plot cumsums inline). `reinsurance_occ_plot` now reads
  from `reins_density_df`; the `occ_ceder` / `occ_netter` /
  `agg_ceder` / `agg_netter` step functions are retained for the exact
  (EX) path.
- Reinsurance reporting **labels are centralised constants** in
  `constants.py` (`REINS_LABEL_GROSS` / `SUBJECT` / `NET` / `CEDED` /
  `OUTPUT`). `describe`'s reinsurance view now leads with **Gross** (was
  "Subject") and labels the model-output column **Net** / **Ceded** / **Output**
  (the last for a mixed program, e.g. occ net of + agg ceded to — replacing the
  old "After"). The occurrence/aggregate ordering and the gross/ceded/net view
  order are canonical throughout (no longer alphabetical).
- New `tests/test_reins_reporting.py`; DecL cases added to `test_decl.agg`
  (section Y). Docs (`2_x_re_pricing.rst`, `2_x_cat.rst`) rewritten to the
  new API — pending a manual docs rebuild. `Re.Both` describe baseline
  regenerated for the Gross/Output relabel.

## 1.0.0a18

### Reinsurance rebucketing switch + layer-order validation

- New `Aggregate.reins_bucket` switch (`'linear'` default, or
  `'nearest'`) controls how net/ceded distributions are rebucketed onto
  the model grid. `'linear'` splits each off-grid value's mass across its
  two bracketing buckets, preserving the first moment **exactly**;
  `'nearest'` rounds to the closest bucket (≤ `bs/2` positional bias).
  Property + validating setter mirror `Portfolio.allocation_method` (clears
  cached reins frames on change); a new module constant
  `REINS_BUCKET_DEFAULT` and an `update`/`update_work`
  `reins_bucket=` kwarg thread it through. Reinsurance is baked in at
  `update`, so a post-build change needs a re-`update()`.
- `Aggregate._apply_reins_work` rebucket core rewritten: the old
  `groupby` → `interp1d` CDF-interpolation → `np.diff` scheme (an
  undocumented third method that did not cleanly preserve the mean, plus two
  `len(...)==1` special cases) is replaced by a vectorized `np.add.at`
  scatter (new `_rebucket_to_grid` helper). Same `reins_df` columns; the
  degenerate "all ceded → net is 0" case falls out naturally. Top-of-grid
  overflow piles into the last bucket (same mode as an aggregate deficit).
- `make_ceder_netter` now hard-errors on out-of-order or overlapping
  reinsurance layers via a new `_validate_reins_layers` check at its single
  choke point: attachments must be non-decreasing and layers must not overlap.
  Gaps are allowed — express one with a zero-share layer `0 po L xs A`.
- Baseline `Re.Both` snapshots regenerated (the only case affected; drift
  ~1e-5 relative, reflecting the more accurate mass-preserving rebucket).
  New `tests/test_reins_buckets.py`; DecL case `ReBucket` added to
  `test_decl.agg`.

## 1.0.0a17

### Refactor harness + Copy-on-Write opt-in

- New `tests/baseline/` characterisation harness for the v1.0 core-compute
  refactor: 10 deterministic cases (7 aggregates, 3 portfolios) snapshot
  `stats_df` / `describe` / `density_df` plus per-distortion
  `augmented_df` / `pricing_at` / `price()` to parquet at
  `rtol=1e-12`, with a pinned manifest recording versions + commit SHA.
  `tests/test_baseline.py` runs every case before reporting, collecting
  all divergences into one summary (see `dev/plan-baseline-harness.md`).
  Adds `pyarrow>=15` to dev extras.
- Pandas Copy-on-Write is now opted in at package import for pandas 2.x
  (pandas \>= 3.0 has CoW on as the default, so the option-setter is a
  conditional no-op there to avoid the deprecated-option warning).

Parser so/po disambiguation + mixture-arm perf guard
\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~~

- `so` (share-of) and `po` (part-of) reinsurance keywords are now
  true synonyms; the **number** sets the meaning. A literal percentage
  (`50% so 200 xs 100`, `50% po 200 xs 100`) is the share directly;
  a bare number (`5 so 10 xs 0`, `5 po 10 xs 0`) is an absolute
  amount and the share is `amount / limit`. Previously `50% po`
  divided the percentage by the limit (silent factor-of-200 error)
  and `5 so` returned the bare value as the share (out-of-range).
  Implementation: parser tracks the `%` suffix through a tiny
  `_PercentNumber` float subclass; arithmetic strips it, so an
  expression like `25 * 2` falls through as an absolute amount.
- New corpus cases `J.Re18a`..`J.Re18d` cover all four
  (keyword, percent-or-absolute) combinations and assert they collapse
  to the same `(share, limit, attach)` tuple.
- Mixture-arm `Aggregate.__init__` skips the ground-up-mixture
  `Severity` constructions when no exposure row carries a positive
  attachment — saves one Severity per mixture component on the common
  no-excess path. They were only needed for the `sf(attach)`
  re-weighting under excess covers.

### Portfolio cleanup (add_exa_details slim, swap_density_df, comments)

- `Portfolio.add_exa_details` slimmed to the still-meaningful EPD +
  reimbursement diagnostic columns (`epd_0_total`, `epd_0_{line}`,
  `epd_1_{line}`, `e1xi_1gta_*`). The legacy eta-mu /
  second-priority surface (`ημ_*`, `exeqa_ημ_*`, `e2pri_*`,
  `lev_ημ_*`, `exlea_ημ_*`, `exi_xgta_ημ_*`, `exa_ημ_*`,
  `epd_2_*`, `epd_0_ημ_*`, `epd_1_ημ_*`) and the
  `add_eta_mu()` companion method removed — they were
  `plot_twelve`-only defensive scaffolding, and `plot_twelve`
  doesn't actually read them.
- `Portfolio._build_augmented(efficient=False)` no longer computes
  `exi_xgtag_ημ_*` / `exag_ημ_*` (no consumers). `pedagogy.plot_twelve`
  no longer warms `add_exa_details(eta_mu=True)`.
- `swap_density_df` promoted from experimental method to standalone
  function in `aggregate.portfolio` (the method is now a thin shim).
  The function recomputes empirical stats via `xsden_to_mwrangler`;
  a swapped portfolio has no `mixed`/`independent` decomposition
  so those stats_df columns are left blank by design.
- `Portfolio.add_exa` / `Portfolio.update` journey-of-discovery
  comments scrubbed: commented-out alternative implementations,
  T.S. Eliot quote, `# TODO What is this crap?` markers, `Doh`
  asides, and dead chained-assignment-workaround blocks gone.
  The `ft_nots` argument of `add_exa` is now required (the
  `None`/`ημ_<line>`-fallback branch was dead since the eta-mu
  removal).

### Portfolio pricing & allocation (pentagon, linear default, ROE fix)

- `Portfolio.price` default flips to `allocation='linear'` (was
  `'lifted'`). Lifted natural allocation reads from the risk-adjusted
  `augmented_df` and is unstable on the right edge for a mass
  distortion on an unbounded support — essentially all the distortion
  weight lands on the last bucket. Linear collapses tail states with
  objective probabilities and stays bounded.
- New `Portfolio.allocation_method` member (`'linear'` /
  `'lifted'`) is the source of truth; the setter clears the
  `augmented_df` cache so the next `apply_distortion` rebuilds.
  Shown in `info`. `price(allocation=…)` still overrides on a
  one-off basis.
- New `Aggregate.bounded` / `Portfolio.bounded` property: `True`
  iff the frequency *and* every severity component is bounded
  (`fixed` / `bernoulli` / `binomial` / `empirical` frequencies
  and finite-support / layer-capped / splice-capped severities).
  Conservative — defaults to `False` whenever it cannot be proved
  `True`. Certify with `obj.bounded = True` (escape hatch for
  cases the heuristic misses).
- `Portfolio.price(allocation='lifted')` now **refuses** when the
  portfolio is unbounded and any requested distortion carries a mass
  (e.g. CCoC on Port.CNC); the error points the caller at
  `allocation='linear'` or the `bounded` override. Bounded
  portfolios (Bodoff, beta mixtures) still take lifted+CCoC unchanged.
- `Portfolio._build_augmented` de-duplicated: the total-level block
  (`exag_total`, `M.M_total`, `M.Q_total`, `M.ROE_total`,
  `roe_zero`) is now computed once with the correct L'Hôpital ROE
  fallback `ROE(1) = 1/g'(1) − 1`. Previously the `efficient=True`
  branch (the default) used `g'(1)` and disagreed with the
  `efficient=False` branch on the right edge — surfaces as numerical
  shifts on mass-distortion + tail cells (the baseline harness moves on
  Port.Bounded and PEG CCoC; non-mass distortions are unaffected).
  When `g'(1) = 0` (TVaR beyond the threshold) the limit is `+∞`
  and `M.Q_{line}/∞ = 0` falls through cleanly.
- `pricing_at` returns the pentagon order `L M P Q a | LR PQ ROE`
  (amounts then ratios; `a = P + Q` is now a first-class column,
  not a post-hoc decoration in `analyze_distortions`). The lifted
  and linear branches of `price` emit the same column shape.
  `PRICING_STAT_ORDER` / `PRICING_STAT_DTYPE` updated to match.
- Linear-branch `price`: the distortion-independent `exp_loss`
  integral (and the tail-collapse on `exeqa`) is hoisted out of the
  per-distortion loop. With *k* distortions the per-call cost drops
  from *k* full reverse-cumsum sweeps to one.
- Journey-of-discovery comments in `_build_augmented` and the linear
  `price` branch deleted; the surviving comments are short, current,
  and point at the equation numbers in PIR §14 where useful.

### Aggregate cleanups + forwards-S unification

- `Distortion.price` now defaults to `S_calculation='forwards'`
  (`S = 1 − cumsum`). Backwards is still available via the same kwarg.
  Forwards is the conservative, mass-preserving choice: under a genuine
  PMF deficit it carries the missing mass as a tail blob rather than
  silently dropping it. Aligns `Distortion.price` with the four other
  sites (`add_exa`, `_build_augmented`, `add_exa_sample`,
  `density_df.S`) that already use forwards by default.
- New `DefectiveDistributionWarning(UserWarning)` in
  `aggregate.constants`, emitted once per `update_work` when the
  aggregate PMF deficit `1 − Σp_agg` exceeds `VALIDATION_NOISE`
  (forwards and backwards `S` diverge by exactly the deficit, so the
  warning advertises the divergence at construction time rather than
  letting it surface silently in downstream pricing).
- New private `Aggregate._fft_aggregate` helper is now the single source
  of truth for the FFT-PGF-iFFT core. `_freq_sev_convolution`,
  `reinsurance_df`, and the subject-aggregate hook in `update_work`
  all delegate to it; the zero-risk and fixed-1 shortcuts live in one
  place.
- Redundant `est_*` moment writes inside `apply_occ_reins` and
  `apply_agg_reins` removed: `update_work` overwrites those fields
  immediately from the same densities using the de-fuzzed
  `xsden_to_mwrangler` worker. The unused
  `Aggregate.aggregate_keys` class attribute is also gone.

### Aggregate reinsurance reporting (Subject / Net / Change)

- `Aggregate.describe` becomes an economic view under reinsurance:
  columns are `Subject EX | <label> EX | Change EX | Subject CV | <label> CV | Change CV | Subject Sk | <label> Sk`, where `<label>`
  is `Net` (every cession passes the net), `Ceded` (every cession
  passes the ceded), or `After` (occ and agg pass different kinds).
  `Change = (after − subject) / subject` is the same column arithmetic
  as the legacy `Err` and reads either as the validation eyeball (no
  reins) or as the cession impact (under reins).
- Headings switch to the denser `EX` / `CV` / `Sk` form on both
  `Aggregate.describe` and `Portfolio.describe` (legacy
  `E[X]` / `CV(X)` / `Skew(X)` retired).
- `Portfolio.describe` now picks its column layout at the **portfolio**
  level so the unit blocks and the `total` block always agree. If
  **any** unit carries reinsurance the whole table flips to the economic
  Subject / `<label>` / Change view (the `total` Subject is the gross
  theoretical, `<label>` the realised after-reins); units with no
  cession are rendered in that layout too. With no reinsurance anywhere
  the table keeps the plain theory/empirical validation view. Previously
  the `total` block stayed in validation headings while ceding units
  used economic headings, so the columns misaligned under `pd.concat`.
  New `Portfolio._reins_after_label` aggregates the per-unit labels
  (one kind → that label, mixed → `After`); `Aggregate.describe` is
  refactored onto `Aggregate._describe(force_reins_label=...)` so the
  portfolio can impose one shared label on every unit.
- The scaffold `stats_df` columns from the previous iteration are now
  populated: `after_occ` (post-occ-reins moments, pre-agg-reins),
  `occ_impact` (after_occ / mixed), `agg_impact` (empirical /
  after_occ), and `gross_empirical` (subject empirical, via one extra
  FFT of `sev_density_gross` when occ-reins is present, free
  otherwise).
- `stats_df['error']` is now the subject-validation column:
  `gross_empirical` vs `mixed`. Under no reinsurance
  `gross_empirical == empirical` and this is exactly the legacy
  theoretical-vs-empirical column. Under reinsurance it is the only
  apples-to-apples check available (the after-reins object has no
  independent theoretical to validate against).
- `Aggregate.valid` now validates the SUBJECT under the hood and ORs
  in `Validation.REINSURANCE` when reins is present. `info` /
  `explain_validation` surface this as `reinsurance; subject not unreasonable` (or `reinsurance; subject fails ...`) so the user
  can tell whether the gross object is sound.

### Shared stats hygiene across Aggregate and Portfolio

- `stats_df` is now an all-`float64` frame on both `Aggregate` and
  `Portfolio`; the legacy `('meta','name')` string row has been removed
  (the name lives on `self.name`). All the `.astype(float)` casts at
  consumer sites are gone.
- Per-component columns are renamed from flat `comp_<i>` to the 2-D
  `e{e}.m{m}` form (exposure component × severity-mixture component).
  The limit-profile arm uses `m=0`; the mixture-product arm carries both
  indices.
- `stats_df` gains scaffold columns for the upcoming reinsurance
  reporting redesign: `after_occ`, `occ_impact`, `agg_impact`,
  `gross_empirical` (NaN-filled for now, populated when reinsurance
  reporting lands).
- `Aggregate.valid` and `Portfolio.valid` now read mean / aliasing
  signals straight off `stats_df['error']` -- single source of truth, no
  detour through `describe`. The hard-coded `eps**3` floor and `10×`
  aliasing ratio are replaced with named constants `VALIDATION_NOISE` and
  `ALIASING_RATIO` in `aggregate.constants`.
- `Portfolio.update` and `Portfolio.create_from_sample` now compute
  empirical aggregate moments via the de-fuzzed `xsden_to_mwrangler`
  worker -- the same convention `Aggregate.update_work` already uses
  (small absolute shift in `est_m`/`est_cv`/`est_skew` and downstream
  pricing -- the PEG / harness baselines move at ~1e-9 relative and are
  recaptured in this iteration).
- `Portfolio._write_empirical_stats` no longer inverts each unit's
  empirical severity `(mean, cv, skew)` back into raw moments via
  `MomentWrangler`; it reads `Aggregate.stats_df['empirical']` raw
  moments directly.
- Two new floor constants in `aggregate.constants` replace the bare
  numerics in `add_exa` / `add_exa_details`: `EXEQA_NOISE_FLOOR`
  (the `exeqa` decomposition-error truncation threshold, was `1e-4`)
  and `FT_NOISE_FLOOR` (the "build up the product" guard, was `1e-10`).

### Noise-aware validation, denoised `describe`, empirical raw moments

- `Aggregate.valid` / `Portfolio.valid` now test CV and skewness with
  `np.isclose` against a definite noise floor (`VALIDATION_NOISE = 1e-12`) instead of a relative error guarded only by `> 0`. This fixes
  spurious skew/CV failures for symmetric or low-skew distributions whose
  analytic value is exactly 0 but computes as floating-point dust -- e.g.
  `dsev [1:6]` (a fair die) no longer reports `fails sev skew, agg skew`.
- `describe` no longer displays floating-point dust: near-zero moment
  cells are snapped to 0, and the error columns fall back to absolute error
  where the theoretical value is ~0.
- `stats_df['error']` is now noise-aware (same fallback). The raw
  `empirical` and `mixed` / `total` columns retain their exact values.
- Empirical raw moments `ex1` / `ex2` / `ex3` are now populated in
  `Aggregate.stats_df['empirical']` for the `sev` and `agg` rows
  (`Portfolio` already did this).
- Empirical aggregate moments are now computed from a de-fuzzed *copy* of
  the FFT density (sub-machine-epsilon fp noise zeroed, the same
  `remove_fuzz` threshold `density_df` uses), so the stored higher
  moments -- notably skew -- are clean and grid-independent instead of
  picking up `x**3`-amplified far-tail noise. `agg_density` itself is
  left untouched as the raw FFT output.
- New `moments.ser_to_mwrangler(ser)` builds a `MomentWrangler` from a
  Series whose index is the support (e.g. `density_df.p_total`).
- `utilities.silence_warnings` now takes optional `category` / `message`
  / `module` arguments to scope what is suppressed.
- The `xsden_to_*` moment helpers now share a single public entry point
  `xsden_to_mwrangler` (returns a `MomentWrangler`), resolving the
  previous `meancv` / `meancvskew` tail-mass inconsistency and avoiding
  redundant moment passes where both raw and standardized moments are
  needed; a definitely defective distribution
  (`sum(p) < 1 - VALIDATION_NOISE`) is now logged at INFO.

## 1.0.0a16

### Distortion: atom-row stats_df, Kusuoka summary in describe

`describe` now ends with three Kusuoka-summary rows for every kind:
`mean_mass` (atom of $`\mu`$ at `p=0`), `max_mass` (atom at
`p=1`), and `interior_atoms` (boolean). The unambiguous names
replace the earlier `mass_at_0` / `mass_at_1` labelling.

`stats_df` drops those three rows and instead carries a variable-length
**atoms section** -- one row per Dirac atom of $`\mu`$, indexed
`mu_<p:.3f>`. The `closed_form` column holds the `p` value;
`D_g` holds the atom mass.

`MixtureDistortion._kusuoka_atoms` merges duplicate `p` across
members. `MinimumDistortion._kusuoka_atoms` detects atoms via two
sources: `brentq`-refined active-member transitions (mass via the
slope-jump identity $`m = s^* (g_i'(s^*) - g_j'(s^*))`$) and
boundary atoms inherited from the member active near `s = 0` /
`s = 1`.

### Portfolio.calibrate_distortion(s) cleanup

- `calibrate_distortion`: dropped the unused `df` parameter; default
  `r0` changed `0.0 → 0.05`. Docstring clarifies that `r0` is
  consumed only by the mass-at-zero kinds (`cll`, `clin`, `lep`,
  `ly`) and ignored otherwise.
- `calibrate_distortions`: dropped `r0` and `df` -- both were
  dead, since the calibrated-kind list is fixed to
  `[ccoc, ph, wang, dual, tvar]` (none take `r0`; the legacy `tt`
  kind that consumed `df` was removed earlier).
- Updated 7 `.rst` doc call sites from the legacy
  `calibrate_distortions(ROEs=[r], Ps=[p], strict='ordered')` to the
  current `calibrate_distortions(coc=r, p=p)`. Also fixed
  `port.dists[…]` → `port.distortions[…]` and `dist_ans` →
  `distortion_df` in the 10-min walkthrough prose.

### plot_twelve self-warming and bound-method fix

`pedagogy.plot_twelve` was silently relying on two preconditions the
user had to set up by hand. Now self-sufficient:

- Detects when the cached `augmented_df` is the lean
  (`efficient=True`) build (no per-line `M.M_<line>` columns), pops
  the cache entry, warms the eta-mu derivatives via
  `add_exa_details(eta_mu=True)` if needed, and rebuilds with
  `apply_distortion(distortion_name, efficient=False)`.
- Two stale `port.augmented_df.loc` / `.query` accesses (treating
  `augmented_df` as a property -- it's been a method since the
  `apply_distortion` refactor) now use the local `aug_df` variable.

### Package surface housekeeping

Each submodule declares its own `__all__`; the package `__init__.py`
is now a stack of `from .module import *` lines. Single source of
truth -- change what's public at the top level by editing the source
module, not `__init__`.

The `warnings.simplefilter('ignore')` block formerly run on package
import is gone. Library code should not mutate global state at import
time. The replacement is an explicit, opt-in helper:

``` python
from aggregate import silence_warnings
silence_warnings()    # mute warnings globally; user choice, not the
                      # library's
```

The four remaining `from .constants import *` lines (in
`distributions`, `utilities`, `spectral`, `portfolio`) were
replaced with explicit imports listing only the constants each module
actually uses.

### aggregate.parser_errors: structured DecL parse-error reports

New `aggregate.parser_errors` module turns Lark's terse parse
exceptions into structured `ErrorReport` dataclasses with line and
column, source-line echo, caret marker, friendly terminal labels, and
"did you mean..." suggestions via `difflib.get_close_matches`. With
Earley + dynamic lexer, almost every DecL parse failure surfaces as
`UnexpectedCharacters`; the formatter recovers the full mistyped word
by scanning forward through the DecL identifier character class, then
compares it against the parser-state's allowed terminal set.

The report is attached to every `build()` parse failure as
`e.report` (and `e.report.render()` gives the multi-line text
form). The wrapping `ValueError`'s `args[0]` is now a one-line
human-readable summary, so `str(e)` at the traceback tail reads
e.g. `DecL parse error at line 1, column 9: Unexpected 'cliams'. Did you mean: claims?` rather than the historical
`namespace(type='?', value='c', index=8)`. The Lark cause chain is
suppressed (`raise … from None`) so notebook tracebacks don't dump
Lark's internal `UnexpectedCharacters` frame; `e.report` carries
forward everything users actually need from it. Three opt-in recipes
(notebook print-then-raise, programmatic suggestion read, IPython
traceback hook) are documented in the "Reading Parse Errors" section
of the DecL language reference.

Long source lines are windowed around the caret with word-boundary
snap and ellipsis markers (`...` / `...`) so the marker stays
visible on a single terminal row. The rendered block uses a tight
layout: the "Did you mean" suggestion and the "Expected" list both
appear inline on the same line as the "Unexpected ..." message, with
no blank breaks.

### decl.lark: keyword terminals now require word boundaries

Every DecL keyword terminal (`AGG`, `SEV`, `PORT`, `CLAIMS`,
`MIXED`, `DISTORTION`, `FREQ`, …) now carries a negative
lookahead `(?![a-zA-Z0-9._:~\-])` mirroring the ID-continuation
character class. Without this, Lark's dynamic lexer would peel a
keyword off the front of a typo like `aggx` and continue parsing as
if the user had written `agg x`, surfacing the error several tokens
downstream at the wrong column. The lookahead forces keywords to
match only on word boundaries — same trick Python's tokenizer uses
for `def` vs `define`. Typos like `aggx Re:MFV41 …` now report
`Unexpected 'aggx'. Did you mean: agg?` at column 1 instead of a
misleading column-6 error about `Re:MFV41`.

## 1.0.0a15

### `Distortion` info / describe / stats_df / density_df quartet

`Distortion` now exposes the same four-property quartet as `Aggregate`
and `Portfolio`: `info` (multi-line summary string), `describe`
(small `(D_g, D_g_inv)` DataFrame with checks block), `stats_df`
(single-column `D_g` table with closed-form and error columns), and
`density_df` (full grid: `g, g_inv, g_dual, g_dual_inv, g_prime, g_dual_prime, kusuoka`).

All four are lazy `cached_property` — zero cost if never accessed.
Parameter setters (`d.a = 0.5`) and calibration (`_finalize_calibration`)
both route through `_build()` which invalidates the cache, so
calibrate-then-read returns fresh tables. The cache survives pickling.

Closed-form moments are surfaced where available: `ph`, `wang`,
`dual`, `tvar`, `ccoc`, `beta`, `bitvar`, `wtdtvar`, `cll`,
`clin`, `lep`. Multi-knot kinds (`minimum`, `mixture`) and the
remaining kinds (`power`, `ly`) leave the `closed_form` column as
`NaN` and rely on numeric values. The `error` column gives an
instant readout of trapezoidal-grid accuracy.

A new `_density_knots()` hook is overridden on kinked kinds so the
grid splices in TVaR/BiTVaR/WtdTVaR kinks (and the cap points for
CLL/CLin); `Distortion.plot()` now reads from `density_df`, so
the plotted curve is consistent with the tables and benefits from
the same knot splicing.

A `_kusuoka_summary()` hook returns three rows surfaced in
`stats_df` — the atoms of the Kusuoka spectral measure $`\mu`$
at `p=0` and `p=1`, and a boolean flag for interior atoms in
`\mu` (True for `tvar`, `bitvar`, `wtdtvar`, and combinations
of these).

## 1.0.0a14

### `aggregate` matplotlib house-style

New `aggregate.style` module — single source of truth for plot styling,
shared by the docs build and any forthcoming server / notebook use. The
underlying style is shipped as `aggregate/data/aggregate.mplstyle`
(color, serif, `figure.figsize = 3.5, 2.45`, `figure.dpi = 300`,
constrained layout).

To use the style in JupyterLab, at the top of any notebook:

    import aggregate.style
    aggregate.style.use()

That mutates `matplotlib.rcParams` globally for the kernel and also
sets `pd.options.display.width = 120`. Any plots from that point on
use the house style.

Variants:

    # leave pandas alone (e.g. you've already configured display.width):
    aggregate.style.use(pandas=False)

    # scoped — only for one figure, restores prior rcParams on exit:
    with aggregate.style.context():
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()

    # scoped with overrides — bigger figure for a screen demo:
    with aggregate.style.context(**{"figure.figsize": (7, 4),
                                    "figure.dpi": 100}):
        fig, ax = plt.subplots()

**One gotcha:** matplotlib's inline backend in Jupyter has its own
`figure.dpi` / `figure.figsize` defaults that it applies *after*
import. If cells imported `matplotlib` before you called `use()`,
the inline backend's defaults can sneak back. Safest pattern is to put
`import aggregate.style` / `aggregate.style.use()` as the **first**
matplotlib-touching lines in the notebook.

If figures still look wrong after that, force the inline backend's own
dpi explicitly:

    %config InlineBackend.figure_format = 'retina'   # or 'png'
    %config InlineBackend.rc = {'figure.dpi': 100}   # override for screen

Replaces `knobble_fonts` (formerly inlined in `docs/conf.py`); the
docs build now calls `aggregate.style.use()`. The B&W branch is
dropped (paperless commitment). `rc_params()` exposes the parsed
style as a dict for inspection / composition.

## 1.0.0a13

`Distortion` constructors take natural, kind-specific parameter names
instead of the generic `(name, shape, r0, df, col_x, col_y)` slots.

- New signature per kind (positional or kwarg):
  - `Distortion('ph', a=)`, `Distortion('wang', lam=)`,
    `Distortion('dual', b=)`, `Distortion('tvar', p=)`.
  - `Distortion('ccoc', d=)` *or* `Distortion('ccoc', r=)` —
    keyword-only; passing exactly one of `d` or `r` is required;
    positional `Distortion('ccoc', x)` raises `TypeError` (explicit
    over implicit). `Distortion.ccoc(d)` static factory unchanged.
  - `Distortion('bitvar', p0=, p1=, w1=)` — `w1` is the weight on the
    upper threshold `p1`.
  - `Distortion('wtdtvar', ps=, wts=)` — `ps` and `wts` are equal
    length; `wts` summing close to 1 is normalised silently, otherwise
    `ValueError`.
  - `Distortion('cll', r0=, b=)`, `Distortion('clin', r0=, slope=)`,
    `Distortion('lep', r0=, r=)`, `Distortion('ly', r0=, r=)`.
  - `Distortion('beta', a=, b=)`, `Distortion('power', x0=, x1=, alpha=)`.
  - `Distortion('minimum', distortions=)`,
    `Distortion('mixture', distortions=, wts=)`.
- Each scalar-shape subclass exposes its natural name as a read/write
  property (e.g. `d.a`, `d.p`); the writer re-runs `_build` so
  downstream cached state stays consistent.
- DecL grammar unchanged; `parser.py` translates the legacy
  `kind shape [df]` tuple to natural kwargs via a new
  `_distortion_spec` helper. Existing `distortion d1 ph 0.5`,
  `distortion d2 bitvar 0.5 [0.95 0.99]`, etc. all still parse.
- `ConvexDistortion` removed (it was a constructor, not a kind).
  Replaced by module-level `aggregate.spectral.convex_distortion(s, gs, *, display_name='')` that takes two raw arrays and returns a
  `WtdTVaRDistortion` whose piecewise-linear `g` matches the upper
  convex envelope. Companions `bagged_distortion` and
  `convex_example` are likewise module-level (not staticmethods).
- Removed: `Distortion.average_distortion`,
  `Distortion.bagged_distortion`, `Distortion.s_gs_distortion`,
  `Distortion.convex_example` staticmethods; `_plot_decorations`
  hook (presentation concern, not core behaviour).
- `power` is no longer calibratable through `Portfolio.calibrate_distortion`
  (`_calibration_init_shape` dropped; `strict_pricing=False`).
- Snapshot regression: `tests/data/distortion_g_snapshot.csv` pins
  `g` and `g_inv` at canonical parameter sets for every documented
  kind to `rtol=1e-10`.

## 1.0.0a12

`extensions/` package removed. Optional/auxiliary code consolidated into
top-level modules or migrated out:

- New: `aggregate.pedagogy` absorbs all doc-cited figure helpers
  (`adjusting_layer_losses`, `savings_charge`, `mixing_convergence`,
  `power_variance_family`, `fig_4_1`, `fig_4_5`, `fig_4_6`,
  `fig_4_8`, `fig_9_1`, `natural_scale`) plus four curated, renamed
  PIR figures: `plot_distortion_and_ins_stats` (was `fig_10_3`),
  `plot_spectral_three_panel` (was `fig_10_5`), `plot_twelve` (was
  `twelve_plot`), `plot_bivariate` (was `biv_contour_plot`). Also
  `bodoff_exhibit` (now takes `port` as first arg, not `self`).
  `ClassicalPremium` pulled in to keep `fig_9_1` working.
- New: `aggregate.pentagon` (was `extensions.pentagon`). Class plus
  the `mapper` / `make_possible_pentagons` helpers. Not re-exported
  from top-level `aggregate`; reach as
  `from aggregate.pentagon import Pentagon`.
- Deleted: `basic.py`, `samples.py`, `test_suite.py` (visual
  reporter; pytest now drives the test_suite.agg coverage),
  `bodoff.py`, `risk_progression.py`, `case_studies.py`,
  `portfolio_pir.py`, `pir_figures.py` and `figures.py`
  (cherry-picked into `pedagogy.py`; the rest deleted),
  `cnc.py` / `discrete.py` / `hs.py` / `tame.py` (PIR
  case-study runner scripts), and the entire `templates/` folder
  (HTML/Markdown scaffolding for the deleted exhibit pipeline; the
  package-data entry in `pyproject.toml` was dropped to match).
- PIR case-study reproduction: install `aggregate==0.30.1` in an
  isolated environment to get the legacy `CaseStudy` workflow. PMIR
  is a separate forward-looking project and does **not** reproduce PIR
  exhibits.
- Doc imports updated to point at `aggregate.pedagogy` (technical
  guides) and `aggregate.ft` / `aggregate.tweedie` (reference page).
  Case-studies user-guide page replaced with a redirect note. The stale
  `aggregate.extensions.ft_invert` examples in
  `5_x_nm_ft_conv_algo.rst` were removed.
- No backwards-compat shim. `from aggregate.extensions import ...` is
  gone; `from aggregate.pedagogy import ...` is the new path.

## 1.0.0a11

`Bounds` redesigned. The IME 2022 pricing-bounds class is now one-shot:
`Bounds(obj, premium, *, a=np.inf, line='total', n_p=257, n_s=513)`
runs the full computation at construction. Access `p_star`, `min_envelope`
(a coherent `Distortion`), `max_envelope` (a callable; not a Distortion
because max-of-concaves isn't concave in general), `min_envelope_hinges`
(the active `(p_lo, p_hi)` bracket at each `s`), `cloud_df`,
`weight_df` and `tvar_df` as properties.

- Accepted input types broadened: `Portfolio` (with `line=`),
  `Aggregate`, `pd.Series`, `pd.DataFrame` (first column = pmf).
- `p_star` solved with `scipy.optimize.brentq` after a dyadic coarse
  bracket on `k/256`. Adaptive p-knots densify the grid at
  `p_star ± 2^{-k}` for `k = 8..11` so the kink between the CCoC
  and TVaR regimes resolves cleanly.
- `cloud_view` → `plot_envelope`. `weight_image` → `plot_weights`.
- Renaming internal arrays to clarify the math:
  `p_knots` (TVaR thresholds, shape `(n_p,)`),
  `s_grid` (distortion eval points, shape `(n_s,)`),
  `tvar_x_p` (`TVaR_p(min(X, a))` at each knot),
  `tvar_hinges` (`min(1, s/(1-p))`, shape `(n_p, n_s)`).
- Removed: `principal_extreme_distortion_analysis`, `ped_distortion`,
  `quick_price` (uncalled), `t_mode` getter/setter and Gauss-Legendre
  branch, `add_one` flag (locked True), `make_tvar_function` (folded
  into the bounded TVaR cache), `tvar_with_bound` (ditto).
- Pedagogy helpers `similar_risks_graphs_sa`, `similar_risks_example`,
  module-level `plot_max_min`, and `plot_lee` moved to a new
  `aggregate.pedagogy` module. Not exported from top-level
  `aggregate`. `plot_max_min` and `plot_lee` previously exported
  from top-level — those exports dropped per the no-shim policy.
- `Portfolio.pricing_bounds` now raises `NotImplementedError` —
  pending rewrite against the new Bounds API. The matmul-shape bug
  reported on PEG (33977 vs 512) was a symptom of the legacy
  `Bounds.tvar_cloud` accepting a free-form `s` array; the new
  `Bounds` always uses a 513-point binary `s_grid`.
- `tests/test_bounds.py` (9 cases). Closed-form analytical pins:
  brackets `(p_star, p_hi)` at `premium = TVaR_{p_star}` carry
  weight zero, so the resulting cloud columns equal the
  `TVaR_{p_star}` distortion exactly. Arbitrary bracket reproduces
  the weighted-combination formula to `1e-10`.

## 1.0.0a10

`ft` consolidation. `FourierTools` and friends promoted from
`aggregate.extensions.ft` to top-level `aggregate.ft`. Reach for
the class via `from aggregate.ft import FourierTools` (submodule
access only, no top-level re-export — same treatment as `Tweedie`).

- The legacy procedural `ft_invert` function (~140 LOC) deleted.
  Its functionality is fully covered by the `FourierTools` class,
  which the module's own docstring already documented as the
  preferred replacement. Docs that reference `ft_invert` are stale
  and will be swept separately.
- Paper-figure helpers (`poisson_example`, `fft_wrapping_illustration`,
  `recentering_convolution`, `recentering_convolution_example`)
  retained in `aggregate.ft` for now. A future `aggregate.pedagogy`
  module will consolidate figure-generators from across the codebase
  (see CLAUDE.md TODO).
- `make_levy_chf` retained.
- Reach-back imports inside `ft.py` (`from .. import build, qd, Aggregate`)
  replaced by direct module imports — eliminates the
  partially-loaded-package fragility that drove tweedie's load-order
  dance in 1.0.0a9.
- `aggregate.tweedie`'s `FourierTools` import repathed
  (`from .extensions.ft` → `from .ft`).
- Light tidy: `FourierTools(object)` → `FourierTools`; stale
  `ft_invert` references in docstrings / assert messages / dead
  commented debug lines cleaned up.
- New `tests/test_ft.py` — small in-regression case asserting that
  `FourierTools` against a closed-form distribution
  (`scipy.stats.norm`) inverts to the analytic pdf.
- Old `from aggregate.extensions.ft import ...` will break — no
  shim per the no-backcompat policy in `CLAUDE.md`.

## 1.0.0a9 (in progress)

Tweedie consolidation. `Tweedie` class promoted from
`aggregate.extensions.tweedie` to top-level `aggregate.tweedie`.
`tweedie_convert` and `tweedie_density` moved out of
`utilities.py` into the same module; their public re-exports from
`aggregate` are unchanged. Public import path for the class:
`from aggregate.tweedie import Tweedie`.

- `Mode`, `Tweedie`, `tweedie_illustration` are NOT re-exported
  at top level — submodule access is the only path. `Tweedie` gets
  the same treatment as `Bounds` (peripheral-but-public).
- `make_test_suite` and `run_test` (interactive notebook scaffolds
  that read a CSV and `IPython.display` audit frames) deleted.
  Replaced by a small in-regression pytest module
  `tests/test_tweedie.py` covering `tweedie_convert` round-trip,
  `tweedie_density` at the mass-at-zero point, Tweedie class moments
  matching V(μ)=disp·μ^p, and the additive↔reproductive duality.
- Light tidy in the moved file: `Tweedie(object)` → `Tweedie`;
  dead commented imports / unused `Path` / `IPython.display`
  removed; `# noqa` annotation on `Aggregate` import dropped.
- `parser.py`'s lazy `tweedie_convert` import repathed from
  `.utilities` to `.tweedie`.
- Old `from aggregate.extensions.tweedie import ...` will break —
  no shim per the no-backcompat policy in `CLAUDE.md`.
- Docs (`docs/2_user_guides/DecL/100_tweedie.rst`, technical guide)
  still reference the old import path — pending a separate docs
  sweep.

## 1.0.0a8 (in progress)

Portfolio refactor sub-project E — stats consolidation. Six overlapping
`Portfolio` stats frames (`statistics_df`, `statistics`,
`report_df`, `report`, `audit_df`, `make_audit_df`) collapsed
into a single canonical `stats_df`. Public stats surface on
`Portfolio` is now exactly three things: `info`, `describe`,
`stats_df` — same shape as `Aggregate`.

- **``stats_df`** is a `DataFrame` with MultiIndex on
  `(component, measure)` rows (`meta` + `freq` + `sev` + `agg`
  blocks) and columns one-per-unit + `total` + `empirical` + `error`.
  Per-unit columns hold each `Aggregate.stats_df['mixed']` (the
  unit's own view); `total` is the portfolio-aggregate theoretical
  view (sum of each unit's `mixed`); `empirical` is the post-FFT
  combined view; `error = empirical / total - 1`.
- Column is `total` rather than Aggregate's `mixed`: at the
  Portfolio level there is no mixed-vs-independent distinction
  (mixed-vs-independent is an Aggregate-only concept that strips a
  single agg's freq mixing distribution).
- **Empirical column is fully populated**:
  - `('agg', *)` rows — raw moments `ex1` / `ex2` / `ex3` plus
    `mean` / `cv` / `skew`, computed straight from the
    portfolio-total FFT density (plain summation, no tail-mass
    correction — matches the PEG baseline numerics).
  - `('sev', *)` rows — raw moments and central moments,
    re-aggregated from each unit's empirical sev mean/cv/skew via a
    fresh `MomentAggregator`. `Aggregate.stats_df` stores only
    empirical mean/cv/skew for sev, so the raw moments are inverted
    via `MomentWrangler` before being fed to the aggregator.
  - `('meta', *)` rows for `limit` / `attachment` / `el` /
    `prem` / `lr` — copied across from `total` with implied
    `error = 0` (these are factual or sums of expected values,
    no FFT analog).
  - `('freq', *)` rows stay `NaN` in `empirical` — frequency is
    exact (no convolution operates on it); same convention as
    `Aggregate.stats_df`.
- **Meta totals tightened**:
  - `total[('meta', 'attachment')]` = `0` when every unit attaches
    at 0 (previously `NaN`); `NaN` only when units disagree.
  - `total[('meta', 'limit')]` = `max` across units (legacy
    convention preserved).
  - `total[('meta', 'lr')]` = `el / prem` when `prem > 0` else
    `NaN`.
- **``('agg', 'P99.9e')` row dropped** — Aggregate dropped the
  estimated-99.9th-percentile row in Stage 1c+; Portfolio follows
  suit. Percentile access via `port.q(p)` / `port.var_dict(p)`
  remains.
- `describe` and the headline `agg_m` / `agg_cv` / `agg_skew` /
  `est_m` / `est_cv` / `est_skew` now read from `stats_df`.
  `describe` total row surfaces empirical sev mean/cv/skew (was
  blank before — sev empirical only existed per-unit).
- `extensions.portfolio_pir.accounting_economic_balance_sheet` and
  `extensions.bodoff` updated to read `stats_df`. The remaining
  `case_studies` exhibit code keeps its old `audit_df` references —
  those extensions are slated for removal at 1.0 per the master plan.
- PEG regression baseline unchanged (numbers reproduce bit-identically
  at `rtol=1e-10`).

Housekeeping in the same release block:

- `extensions.portfolio_pir.gamma` (the ~136-LOC conditional layer
  effectiveness γ exhibit) and its `GammaResult` dataclass deleted —
  both were orphaned: not called from `make_all`, not exercised by
  any test, not referenced in any rendered doc.
- `Underwriter.__repr__` gains a one-line usage hint pointing at
  `.discover(regex)` — fills the discoverability gap left when
  `qshow` / `qlist` / `show` were removed in 1.0.0a1.
- Eight stale comments and docstrings across `distributions.py` /
  `utilities.py` / `portfolio.py` that still mentioned
  `statistics_df` / `report_ser` / `audit_df` (in their
  stats-consolidation sense) refreshed. Distinct
  `reinsurance_audit_df` / `reinsurance_report_df` attributes
  and the `audit_df` field on `AnalyzeDistortionResult` are
  unrelated and unchanged.

Utilities refactor — `aggregate/utilities.py` shrinks from 3,753 to
~700 LOC. The grab-bag module is reduced to a focused set of
cross-cutting helpers; dead code is removed; themed code moves into
new modules or back to its only caller.

- **Deletes (~1,300 LOC):**
  - `frequency_examples` / `axiter_factory` / `AxisManager`
    (~530 LOC of pedagogical scaffolding with no consumers).
  - `MomentAggregator.stats_series` (retired by Stage 1c+).
  - `test_var_tvar` (internal scaffold).
  - Plotting / formatting subsystem: `FigureManager`,
    `make_mosaic_figure`, `easy_formatter`, `knobble_fonts`,
    `style_df`, `friendly`, `GreatFormatter`, `sEngFormatter`,
    `show_fig` (~630 LOC). `aggregate` no longer touches the
    user's matplotlib settings.
  - Dead-import / alias cleanup: `html_title`, `suptitle_and_tight`,
    `ln_fit` alias.
  - Dead public helpers: `mv`, `qdp`, `introspect`,
    `get_fmts`, `sensible_jump`, `GCN` namedtuple.
  - Timer cruft and the commented `knobble_fonts(True)` call.
- **New modules:**
  - `aggregate/moments.py` — `MomentAggregator`, `MomentWrangler`,
    `xsden_to_meancv`, `xsden_to_meancvskew`.
  - `aggregate/iman_conover.py` — `iman_conover` + `ic_*`,
    `block_iman_conover`, `make_corr_matrix`,
    `random_corr_matrix`, `rearrangement_algorithm_max_VaR`.
- **Public ``*_fit`` family in ``distributions.py`:** symmetric
  `(m, cv[, skew])` → distribution-parameter cluster, all importable
  from `aggregate`: `lognorm_fit` (renamed from
  `mu_sigma_from_mean_cv`), `gamma_fit`, `beta_fit`,
  `invgamma_fit`, `invgauss_fit`, `sln_fit`, `sgamma_fit`.
  Plus `approximate_from_mcvsk` (renamed from `approximate_work`),
  `lognorm_approx`, `lognorm_lev`. The `ln_fit` alias is
  dropped — `lognorm_fit` is canonical.
  `approximate_from_mcvsk`'s gamma branch now calls
  `gamma_fit(m, cv)` — symmetric with the lognorm branch using
  `lognorm_fit`.
- **Private single-module helpers moved + privatised** (no public
  surface change beyond the underscore): `_estimate_agg_percentile`,
  `_picks_work`, `_moms_analytic` + `_partial_e` +
  `_partial_e_numeric`, `_integral_by_doubling`,
  `_logarithmic_theta` all moved into `distributions.py`.
  `_parse_note` (the merge of `parse_note` + `parse_note_ex`)
  moved into `underwriter.py`. `_short_hash` moved into
  `spectral.py`.
- `make_comonotonic_allocations` moved to `portfolio.py` as a
  public module-level function (paired with the `Portfolio`
  method of the same name). Named locally
  `make_comonotonic_allocations_work` to avoid clashing with the
  method; re-exported cleanly as `make_comonotonic_allocations`.
- `Aggregate.plot` and the `bounds.py` `FigureManager` call
  site rewritten to plain matplotlib (`plt.subplot_mosaic`,
  `plt.subplots`). `extensions/case_studies.py` mpl call sites
  converted likewise.
- `pprint` renamed to `decl_pprint` (the DecL syntax-highlighter
  helper; avoids stdlib name collision).
- Documentation: `mu_sigma_from_mean_cv` → `lognorm_fit` updated
  in `2_x_actuary_student.rst`, `2_x_re_pricing.rst`, and
  `5_x_rearrangement_algorithm.rst`. Other rst pages pending a
  full sweep.
- PEG regression baseline unchanged (numbers reproduce bit-identically
  at `rtol=1e-10`); 430 pytest cases pass.

Packaging: src/ layout. The package source moved from
`aggregate/` to `src/aggregate/`. The src layout prevents accidental
imports from the source tree when CWD is the repo root — the only way
to `import aggregate` is now via the installed (editable) package,
which makes editable installs behave identically to wheel installs.
`pyproject.toml` gains `package-dir = {"" = "src"}`; `MANIFEST.in`
grafts repathed; `docs/conf.py` `sys.path` insert updated to
`../src`; four test/capture files repathed to
`src/aggregate/agg/test_suite{,2}.agg`. No public API change; 430
pytest cases still pass.

## 1.0.0a7

Portfolio refactor sub-project D — distortion-pricing pipeline redesign.
Six related changes that together collapse ~500 LOC of pricing code into
a small cache + a single signature convention:

- **D.1 — augmented_df lazy-eval cache.** `Portfolio.apply_distortion`
  becomes a thin cache lookup-or-build keyed on distortion name; the
  construction logic lives in a private `_build_augmented`. Second
  calls return the cached frame (`frame_a is frame_b`).
  `port.augmented_dfs` is a dict view of the cache; `port.augmented_df`
  is the clean read-side accessor (also routes through the cache).
  `apply_distortion` drops the `df_in=` (gradient path, gone),
  `create_augmented=` (the cache replaces it), and `plots=`
  (uninvoked) kwargs. `apply_distortions` (plural) deleted. New
  `Portfolio.pricing_at(distortion, *, p=None, a=None)` consolidates
  the row-extraction logic that previously lived in `price` and
  `analyze_distortion`.
- **D.2 — analyze_distortion(s) and calibrate_distortions collapse onto
  the cache.** Each former 100-250 LOC method becomes ~25 LOC.
  `analyze_distortion(distortion, *, p=None, a=None)` returns an
  `AnalyzeDistortionResult` dataclass with `pricing_df` and
  `audit_df`. `analyze_distortions(*, p=None, a=None, distortions=None)` returns `AnalyzeDistortionsResult` with the
  multi-distortion exhibit (MultiIndex `(distortion, stat)`) and a
  cache snapshot. `analyze_distortions2` and the list-based
  `calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, …)` deleted in
  favour of single-coc / single-p forms.
- **Explicit ``p=`` / ``a=` convention** across the pricing surface
  (`pricing_at`, `analyze_distortion`, `analyze_distortions`,
  `calibrate_distortions`). The legacy implicit `p > 1 → asset`
  threshold is gone; callers state intent. Each method raises
  `ValueError` if both or neither is supplied.
- **D.3 — Answer → typed dataclasses.** The legacy `Answer` dict
  class is deleted. `aggregate.results` defines `PricingResult`,
  `PricingBoundsResult`, `AnalyzeDistortionResult`,
  `AnalyzeDistortionsResult`, and `GammaResult` (the last used by
  `extensions.portfolio_pir`). Inline `namedtuple` definitions in
  `Portfolio.price` and `Portfolio.pricing_bounds` promoted to the
  same module.
- **D.4 — ordered categoricals.** `aggregate.spectral.DISTORTION_ORDER`
  / `DISTORTION_DTYPE` (`ccoc, ph, wang, dual, tvar, wtdtvar, lep, ly, clin, tt, cll, bitvar, blend`) and
  `aggregate.portfolio.PRICING_STAT_ORDER` / `PRICING_STAT_DTYPE`
  (`L, LR, M, P, PQ, Q, ROE`) bake the canonical order into the data.
  `Portfolio.distortion_df` `method` index level, `pricing_at`
  columns, and `analyze_distortions` pricing_df `distortion` level
  are typed categoricals -- `sort_index()` produces the canonical
  order without ad-hoc reordering.
- **D.5 — renames.** `Portfolio.dists` → `Portfolio.distortions`;
  `Portfolio.dist_ans` and the `distortion_df` property merged into
  a single `Portfolio.distortion_df` attribute with the trimmed 9-col
  layout (`S, L, P, PQ, Q, COC, param, std_param, error`) and index
  names `('a', 'LR', 'method')`; `Portfolio.limits` →
  `Portfolio._limits` (internal helper).
- PEG regression baseline unchanged -- `test_pricing` at `rtol=1e-8`
  reproduces the legacy `analyze_distortions2` exhibit bit-identically.
  The new pipeline is mathematically the same; only the API surface changed.

## 1.0.0a6

Portfolio refactor sub-project C — distortion calibration moves to the
`Distortion` subclasses themselves:

- `Portfolio.calibrate_distortion` was ~240 LOC of per-name Newton
  iterations in a giant `if name == 'ph': ... elif name == 'wang': ...`
  switch. Each branch defined a local `f(shape) → (residual, derivative)`
  closure and ran a hand-rolled Newton loop. That code now lives on the
  `Distortion` subclasses — each pricing-distortion class owns its own
  `calibrate(S, bs, premium_target, *, ess_sup, assets, el, **kwargs)`
  method: `PHDistortion`, `WangDistortion`, `DualDistortion`,
  `TVaRDistortion` (`max_iter=200`), `CCoCDistortion` (closed-form,
  no iteration), `LYDistortion`, `CLinDistortion`, `LEPDistortion`,
  `CLLDistortion`.
- `Portfolio.calibrate_distortion` shrinks to ~100 LOC — about half
  asset/S resolution (unchanged), about half dispatch to the subclass via
  `Distortion._registry`. The `tt` (Wang-t) branch is gone — there is
  no `TtDistortion` subclass to host it and the branch was dead code.
  `wtdtvar` calibration is also dropped from the dispatcher (the
  parametrisation overload between calibration form `(w, [p0, p1])` and
  the standard form `(ps, wts)` was already broken in the constructor;
  pick a pricing distortion that calibrates cleanly instead).
- New `Distortion` base-class methods `_newton_iterate(f, shape, *, max_iter, tol)` and `_finalize_calibration(shape, fx, prem, assets)`
  factor the Newton loop and the post-iteration bookkeeping (write
  `shape` / `error` / `premium_target` / `assets`, log on
  non-convergence, re-run `_build` to refresh cached state) out of the
  per-subclass methods.
- Class attribute `Distortion._calibration_init_shape` is the
  per-kind starting shape used both to construct the uncalibrated
  distortion and as the Newton iteration's starting point. `None` on
  the base means "not calibratable through the Portfolio dispatch."
- Each subclass is now testable in isolation. New
  `tests/test_distortion_calibrate.py` (12 cases) exercises every
  migrated kind directly on a synthetic `S` vector and asserts the
  achieved premium matches the target.
- PEG regression baseline unchanged — the new subclass-based Newton
  iteration reproduces bit-identical Newton convergence.

## 1.0.0a5

Portfolio refactor sub-project B — drop approximation and tilting paths
from `Portfolio.update` and `Aggregate.update_work`:

- Removed the auto-fallback method-of-moments approximation path. The
  `approx_freq_ge` / `approx_type` / `approximation` kwargs are gone
  from `Portfolio.update`; the matching `approx_type` /
  `approx_freq_ge` attrs are gone from `Portfolio.__init__`,
  `Portfolio.json`, and `Portfolio.__repr__`. The
  `'exact' if agg.n < approx_freq_ge else approx_type` ternary is gone;
  callers always get the FFT path. The slognorm / sgamma branch in
  `Aggregate.update_work` (and the `approximation` attribute on
  `Aggregate`) is deleted. `Portfolio.approximate` /
  `Aggregate.approximate` (the user-facing on-demand
  method-of-moments fit returning a `scipy.stats` frozen RV or a DecL
  spec) are unchanged.
- Removed FFT tilting (Grübel/Hermesmeier 1999) from the update pipeline:
  the `tilt_amount` attr is gone from `Portfolio.__init__`, the
  `tilt_vector` construction block is gone from `Portfolio.update`,
  and the `tilt=` parameter is removed from the `ft` / `ift`
  module-level helpers in `aggregate.utilities` and the matching
  `Portfolio.ft` / `Portfolio.ift` wrappers. The tilt branches inside
  `Aggregate.update_work`, `Aggregate._freq_sev_convolution`, and
  `Aggregate.apply_agg_reins` are gone. Use more buckets if aliasing
  shows up — per author's standing preference.
- `aggregate.extensions.figures.gh_example` was the only consumer of
  tilting in the visualisation layer; it now compares the padded FFT
  result against the exact compound probability without the
  tilt-comparison loop.
- PEG regression baseline (`tests/data/peg_baseline.json`) re-captured
  against the exact FFT path. The previous baseline incidentally
  exercised slognorm — PEG's two units (n=100 and n=150) tripped the
  default `approx_freq_ge=100` threshold. The drift is ~5e-6 on
  `est_m` and ~2e-5 on pricing cells; the new contract is the
  exact-FFT result.

Portfolio refactor sub-project A — pure deletions + PIR move
(`portfolio.py` shrinks from 6,133 → 3,707 LOC):

- Deleted ~700 LOC of dead code from `Portfolio`: `gradient` (~196 LOC),
  non-spectral allocations (`merton_perold`, `cotvar`,
  `equal_risk_var_tvar`, `equal_risk_epd`), the EPD / priority /
  collateral family (`analysis_priority`, `analysis_collateral`,
  `priority_capital_df`, `epd_2_assets`, `assets_2_epd` properties
  plus their backing attrs), the `uat` / `uat_differential` /
  `uat_interpolation_functions` trio, `collapse`, `audits`,
  `stat_renamer`, and the `var_dict(kind='epd')` branch.
- Stripped `analyze_distortion_add_comps` and
  `analyze_distortion_plots` (~470 LOC) — both consumed the deleted
  allocation methods. `analyze_distortion` keeps `add_comps` and
  `plot` parameters as no-op defaults (`add_comps=False` now).
- Moved ~1,800 LOC of PIR-exhibit machinery to the new
  `aggregate.extensions.portfolio_pir` module as free functions taking
  a `Portfolio` as the first argument: `premium_capital`,
  `multi_premium_capital`, `accounting_economic_balance_sheet`,
  `make_all`, `show_enhanced_exhibits`, `set_a_p`,
  `profit_segment_plot`, `natural_profit_segment_plot`,
  `density_sample`, `biv_contour_plot`, `twelve_plot`,
  `short_renamer`, `gamma`, `stand_alone_pricing`,
  `stand_alone_pricing_work`, `calibrate_blends` (with helpers
  `check01` / `make_array` / `convex_points`), the bulk
  constructors `from_DataFrame` / `from_Excel` /
  `from_dict_of_aggs`, and the big `renamer` plus
  `premium_capital_renamer`.
- `aggregate.extensions.case_studies` updated to call the moved
  functions as free functions; `aggregate.extensions.bodoff` inlines
  the deleted `cotvar` lookup.

## 1.0.0a4

Portfolio refactor sub-project 0 — PEG regression baseline:

- New regression fixture `tests/peg.py` exposes `build_peg` which
  constructs the canonical two-unit `port PEG` Portfolio (limit-and-attachment severity, three-component lognormal severity mixture per
  unit, gamma frequency mixing with different mixing CVs per unit).
- New capture script `tests/capture_peg_baseline.py` runs PEG through
  `calibrate_distortions(COCs=[.15], Ps=[.995])` and
  `analyze_distortions2(.995)` for the five-distortion suite
  (`ccoc`, `ph`, `wang`, `dual`, `tvar`) and writes the
  numerical baseline to `tests/data/peg_baseline.json`.
- New test module `tests/test_portfolio_peg_regression.py` pins
  portfolio moments (`rtol=1e-10`), per-distortion calibration shapes
  (`rtol=1e-8`, `|error| < 1e-5`), and every cell of the
  `analyze_distortions2` exhibit (120 values, `rtol=1e-8`).
- Every subsequent Portfolio refactor sub-project (A through E) must
  reproduce these baseline numbers; the JSON is the contract.

`Aggregate` stats consolidation — finish the job: eliminate the
`_statistics_df` / `_statistics_total_df` scratch frames so `stats_df`
is the only theoretical-moment DataFrame the class holds:

- `Aggregate.__init__` now pre-creates an empty `stats_df` (canonical
  `MultiIndex` rows, NaN-filled) right after `n_components` is known in
  each broadcasting arm, via a new `_init_stats_df` helper.
- `_record_component` writes a column of `stats_df` directly (no more
  intermediate row in `_statistics_df`).
- The post-loop totals block writes `mixed` / `independent` /
  `('meta', 'wt')` directly into `stats_df` columns.
- `('agg', 'P99.9e')` row dropped — it had only two populated cells
  (`mixed` and `independent`), was read in one spot (`_limits`
  fallback when `agg_density` is `None`), and is cheaply rebuildable
  on demand via `estimate_agg_percentile`. That one read site now
  computes on the fly.
- All readers migrated: `avg_limit` / `avg_attach` / `tot_prem` /
  `tot_loss`, `self.agg_m` / `agg_cv` / `agg_skew` / `sev_*`,
  `update_work` severity weights, `severity_error_analysis` weights,
  `info` / `_html_info_blob` component count.
- `_statistics_df`, `_statistics_total_df`, and the
  `_build_stats_df` method are gone.
- Side benefit: `stats_df` row layout is now cleaner — all `meta` rows
  together at the top (`mix_cv` and `wt` previously trailed at the
  bottom because of how the legacy scratch frames were ordered).

## 1.0.0a3

`Aggregate` stats consolidation: six overlapping moment DataFrames → one
`stats_df` (breaking changes; v1.0 cleanup):

- New canonical `Aggregate.stats_df`: single source of truth for moment
  statistics. `MultiIndex (component, measure)` rows (`component` ∈
  `{meta, freq, sev, agg}`; `measure` ∈ `{mean, cv, skew, ex1, ex2, ex3, …}`); columns are per-component (`comp_0`, …), `mixed`,
  `independent`, `empirical`, and `error`. Built in two phases:
  theoretical content in `__init__`, `empirical` and `error` appended
  in `update_work` after the FFT. Empty cells are `NaN` where
  meaningful (e.g. `('freq', *) × empirical` is undefined — the FFT
  produces one combined empirical distribution, not per-component
  empirical moments).
- Naming convention unified: `ex1` / `ex2` / `ex3` for raw moments
  and `mean` / `cv` / `skew` for derived. The legacy `_1` / `_m`
  flat-column convention is gone.
- The Aggregate "stats surface" is now exactly three things — `info` (text
  about the Aggregate), `describe` (the daily-driver moment audit), and
  `stats_df`. Removed: `report_df`, `report_ser`, `statistics`,
  `audit_df`. Privatised: `statistics_df` → `_statistics_df`,
  `statistics_total_df` → `_statistics_total_df`.
- `Aggregate.describe` rewritten to source from `stats_df`; output
  byte-identical.
- `Portfolio` migrated to read `a.stats_df['mixed']` instead of
  `a.report_ser` (three lines in `portfolio.py`). Portfolio's own
  `statistics_df` / `audit_df` / `report_df` are unaffected — they
  live on Portfolio, not Aggregate, and will be rationalised in Stage 2.
- Docs migrated: ~30 references to `report_df` / `statistics` /
  `statistics_df` across nine tutorial pages rewritten to use
  `stats_df` with explicit row / column accessors.

## 1.0.0a2

Aggregate surface rationalization (breaking changes; v1.0 cleanup):

- Visible layer structure: file-level section dividers in `distributions.py` and a public-API block in the `Aggregate` class docstring document the integration surface (`report_ser`, `statistics_df`, `update_work`, `agg_density`, `ftagg_density`, `density_df`, plus the risk-measure surface `q` / `tvar` / `cdf` / `sf` / …) that `Portfolio` and `Bounds` consume.
- FFT five-line core extracted to `Aggregate._freq_sev_convolution`; docstring references the four-step algorithm in §2.2 of the paper. `update_work` reads top-to-bottom as compute-severity → occurrence reinsurance → convolution → aggregate reinsurance → audit.
- Shared inner-block of `__init__`'s two broadcasting arms factored into `Aggregate._record_component` (centralises `statistics_df` column ordering across the limit-profile arm and the mixture-product arm).
- `__init__` state initialization regrouped into labelled blocks: spec passthroughs, grid + runtime config, exposure outputs, computed densities, empirical moment estimates, cached lazy functions, reinsurance state, theoretical moment tables.
- `density_df` property docstring expanded with a column-by-column reference table (set-by / read-by for each of 17 columns) — no behavior change.
- Aggregate methods privatised (leading underscore): `audit_df` → `_audit_df`, `statistics_total_df` → `_statistics_total_df`, `limits` → `_limits`, `html_info_blob` → `_html_info_blob`. `aggregate/extensions/figures.py` and `aggregate/extensions/test_suite.py` updated for the renames.
- `Aggregate.more`, `Portfolio.more`, `Underwriter.more` renamed to `.help`. Backing free function in `utilities.py` renamed `more` → `agg_help` (prefixed so it doesn't shadow Python's builtin `help` at module / package level).
- `pprogram` / `pprogram_html` collapsed: dropped the `split=20` line-magic and the `show=True` side-effect print. Methods preserved — cheat sheets and Underwriter consume them.
- Historical-comment sweep across the `Aggregate` class: stale `# TODO` / `# WHOA! WTF` markers and a commented-out spec-dict block removed.
- Logger calls in `distributions.py` converted to lazy `%s`-style formatting (extends the earlier `utilities.py` cleanup).
- Public surface intentionally retained after a docs audit revealed heavy tutorial usage: `statistics`, `statistics_df`, `report_df`, `report_ser`, `info`, `describe`, `snap`, `unwrap`, `picks`, `recommend_bucket`.

`Underwriter.build()` return contract uniform:

- `Underwriter.build()` now raises `CannotBuild` (subclass of `ValueError`) when a parsed spec produces no top-level object — previously returned a `ParsedProgram` with `object=None` in the named-mixed-severity edge case. The contract is now uniform: `build → object` always (or raises), `build_many → list[ParsedProgram]` always. `CannotBuild` is exported from the `aggregate` package.
- `Underwriter.discover()` catches `CannotBuild` and skips the row with a `logger.warning` (mirrors today's `NotImplementedError` handling).

Tooling:

- New `doc-test-uv.ps1` script: uv-managed doc build that replaces the clone-to-tmp dance in `doc-test.ps1`. Builds in place, uses a dedicated `.doc-venv` (set via `UV_PROJECT_ENVIRONMENT`) so doc builds don't disturb the main development `.venv`. Supports any Python via `--python X.Y` (uv auto-downloads if needed).

## 1.0.0a1

Underwriter surface rationalization (breaking changes; v1.0 cleanup):

- `Underwriter.discover(regex, kind='', plot=False, describe=False, return_objects=False, **kwargs)` replaces `show` / `qshow` / `qlist` (all three removed). Default behavior is the lightweight directory view (matches today's `qshow`); pass `plot=True` or `describe=True` to build each match.
- `Underwriter.build_many(program, ...)` is the explicit-batch counterpart to `build`; `build` now raises `ValueError` when its program produces 0 or \>1 top-level outputs (directing the user to `build_many`).
- `Underwriter.interpret_file(filename=None, where='')` replaces `interpret_test_file` and absorbs `run_test_suite`; with no arguments it runs the bundled test suite. Fixes a `KeyError: 0` bug from the pandas iterrows path.
- Directory rationalization: `site_dir`, `case_dir`, `template_dir` properties removed. Single new `user_dir` (`~/.aggregate`). `default_dir` is now located via `importlib.resources.files`.
- Base data directory moved from `~/aggregate` to `~/.aggregate` (dotted convention). No fallback — existing users must `mv ~/aggregate ~/.aggregate`.
- Constructor magic strings: `databases='all'` now expands to `['default', 'user']`; `databases='site'` raises `ValueError` directing users to `'user'`.
- Methods privatized (now leading underscore): `write` → `_build_work`, `factory` → `_factory`, `safe_lookup` → `_safe_lookup`, `interpret_program` → `_interpret_program`. `write_from_file`, `dir`, `test_suite()` method, `run_test_suite` deleted (all unused).
- Portfolio and case_studies internal callers switched from `uw.write(spec)` to `uw.build_many(spec, update=False)` (equivalent — same `ParsedProgram` list, no smart-update).
- `ParsedProgram` (dataclass) replaces `Answer` for the Underwriter parse-output type. `Answer` itself remains in `utilities.py` and continues to be used by `Portfolio`.
- `Underwriter.__repr__` clarified: shows `0 loaded (access .knowledge to read configured database(s))` when knowledge is pending; no I/O side effect.
- Several bug fixes: `factory` `ValueError` is now actually raised; the buggy "1 port among many" return path is gone; `__getitem__` `TypeError` → `KeyError` chain preserved with `from e`; `read_database` narrows to `OSError` and uses `logger.exception`.
- Three new constants in `constants.py`: `USER_DIR_NAME`, `PACKAGE_DATA_DIR`, `TEST_SUITE_FILENAME`.
- Internal cleanup: lazy `%s`-formatted logger calls throughout; ~130 lines of stale commented-out code removed from `utilities.py`.

## 0.30.1

- Confirmed support for Python 3.13 and 3.14

## 0.30.0

- Added `comonotonic_allocations` to `Portfolio` to implement the method of Denuit, Michel, et al. "Comonotonicity and Pareto optimality, with application to collaborative insurance." Insurance: Mathematics and Economics 120 (2025): 1-16. This uses numba if available. Warning: it can be very slow without numba!

## 0.29.0

- Portfolio analyze_distortions2 to iron out annoyances with current function but retain it for backwards compatibility.
- Portfolio calibrate_distortions2 for same reasons, args coc and reg_p.
- Spectral tvar_info_df and plot_affine for working with weighted TVaR distortions.
- Changed behavior of Distortion.random_distortion so that input number of knots *includes* mass and mean if present.
- Added random_distortion_ex(n=1, random_state=None) in Distortion class to simulate across types, extending random_distortion which is only a wtdtvar.

## 0.28.1

- `applymap` to `map` per Pandas update.

## 0.28.0

- Added `standard_shape` to Distortion and added to distortion_df created by Portfolio.calibrate_distortions.
- Updated dependencies and imports for doc build.
- Added `spectral.consistent_distortions` to create consistent family of representative distortions.

## 0.27.1

- Fixed a bug with recommend unit in a portfolio with all fixed components.
- Adjusted line styles in twelve plot and clarified use in doc string.
- Corrected ROE calculation of natural allocation premium when g(s) = 1.

### 0.27.0

- Removed control over logging and just use `logger = logging.getLogger(__name__)` in all modules. Removed `log_test` function and `LoggerManager` class.
- Removed `numba` as a requirement - huge library, hardly used. Only occurs in spectral module.
- Replaced build_docs batch file with doc-test which mirrors readthedocs process more closely.

### 0.26.0

- `extensions` no longer sets `pd.float_format` to Engineering.
- Added `tweedie.Tweedie` class to `extensions` to compute the Tweedie class distributions for
  all valid $`p`$. (Dangling jax dependence.)

### 0.25.0

- Tweak `extensions.ft.FourierTools`: added `invert_simpson` method using Simpson's rule,
  better for stable distributions. This is the method used by `scipy.stats`.
- Bumped to 0.25 which should have done in 0.24.2 because it added new functionality
- Tidied docs
- `knobble_fonts` uses serif font by default in matplotlib, and sets up
  in color mode by default.

### 0.24.2

- Added `Distortion.make_q` to return the risk adjusted probabilities used
  in pricing. Same logic as `price_ex`. Makes it easy to compute the natural
  allocation from a distortion.
- Added `extensions.ft.FourierTools` class, which performs direct inversion of a (continuous) Fourier transform (characteristic function)
  using FFTs. This is particularly useful for stable distributions, where the Fourier transform is known but the density is not. See examples in Section 5 of the documentation.
- Added `make_levy_chf` to `extensions` to compute the characteristic function of a Levy stable distribution.

### 0.24.1

- Added script to build the documentation from a local clone of the repository.
- Added `Aggregate.unwrap` to adjust aggregates computed with too few buckets
  but enough space. It unwraps the computed aggregate by adjusting the index. This
  reverses the "wagon-wheel" effect, whereby FFTs wrap-around the end of the array.
- Vectorized `ultilities.estimate_agg_percentile` for use in `Aggregate.unwrap`

### 0.24.0

- Added state to Distortions so they can be pickled. Involved separating part of `Distortion.__init__`
  into a new method, `Distortion._complete_init`. This is called from `__init__` and `__setstate__`.
  Ensured `_complete_init` refers to arguments as self.argname, not argname and set self
  variables in class `__init__` method.
- Fixed mixture g functions to handle input multidimensional arrays.
- Simplified `Distortion.__repr__` and `Distortion.__str__`.
- Added `Distortion.id` to generate a unique ID depending on `__dict__` argument elements.
- Corrected `g_prime` for minimum distortion.
- Fixed biTVaR distortion to handle p1==1 by including the mass explicitly.
- Added `Distortion.price_ex` to combine best of price and price2 methods and improve flexibility. It sorts and summarizes if needed. Optional return formats.
- Added four numba compiled functions to Distortion for fast computation of
  g.g(1-ps.cumsum()) and g.price( kind='ask'). These are tvar_gS, bitvar_gS,
  tvar_ra (for risk adjusted expected value) and bitvar_ra. In each case the
  values are computed without any copies of the original data, making them
  far more memory efficient for very large input arrays. At the extreme,
  bitvar_ra results in a speed up of the order of 2000x in realistic
  situations, even with small (100s) input vectors. The functions are static
  members of Distortion (numba requirement). They are not parallelized
  because of the cumulative computation of S. See the file
  PyWork/Distortion-price-tester.ipynb for tests (TODO: integraete into the
  documentation.) This addition results in numba being a required package.
- Removed dependency on `titlecase` package.
- Removed `Distortion.calibrate` method, which was not used and never tested. It lives with `Portfolio`.

### 0.23.0

- Added `sample_df` dataframe to `Portfolio` when created from a sample
  to store the sample. Original sample is needed in various applications.
- Added `swap_density_df(self, new_df, padding=1)` to `Portfolio`.
- Fixed errors in Case Studies caused by changes in Pandas.
- Added ability to create Markdown case output, rather than HTML.
- Added beta distortion (generalizes the PH and dual)
- Updated `np.alltrue` to `np.all`; updated `NoConverge` in `scipy.optimize`.
- Added `Distortion.calibrate` to calibrate to a pricing target from input `density_df` (TODO: needs testing).
- Added `wtdtvar`` to ``Distortion` to compute the weighted TVaR from p values and weights,
  masses and mean components.
- Added `minimum` to `Distortion` to create a new `Distortion` as the minimum of a list of input Distortions. The list is passed as shape.
- Added `random_distortion` to `Distortions` to compute a random distortion, useful
  for testing!
- Fixed `tvar` distortion to allow p=1 (max)
- Simplified `Distortion.__repr__` and `Distortion.__str__`.
- Added `Distortion.ph``, ``.wang`, ..., methods for common distortions, with better
  hints for parameters. All are static methods that delegate to the constructor.
- Fixed documentation build errors.

### 0.22.0

- Created version 0.22.0, "convolation" for AAS submission

### 0.21.4

- Updated requirement using `pipreqs` recommendations
- Color graphics in documentation
- Added `expected_shift_reduce = 16  # Set this to the number of expected shift/reduce conflicts` to `parser.py`
  to avoid warnings. The conflicts are resolved in the correct way for the grammar to work.
- Issues: there is a difference between `dfreq[1]` and `1 claim ... fixed`, e.g.,
  when using spliced severities. These should not occur.

### 0.21.3

- Risk progression, defaults to linear allocation.
- Added `g_insurance_statistics` to `extensions` to plot insurance statistics from a distortion `g`.
- Added `g_risk_appetite` to `extensions` to plot risk appetite from a distortion `g` (value, loss ratio,
  return on capital, VaR and TVaR weights).
- Corrected Wang distortion derivative.
- Vectorized `Distortion.g_prime` calculation for proportional hazard
- Added `tvar_weights` function to `spectral` to compute the TVaR weights of a distortion. (Work in progress)
- Updated dependencies in pyproject.toml file.

### 0.21.2

- Misc documentation updates.
- Experimental magic functions, allowing, eg. %agg \[spec\] to create an aggregate object (one-liner).
- 0.21.1 yanked from pypi due to error in pyproject.toml.

### 0.21.0

- Moved `sly` into the project for better control. `sly` is a Python implementation of lex and yacc parsing tools.
  It is written by Dave Beazley. Per the sly repo on github:

  The SLY project is no longer making package-installable releases. It's fully functional, but if choose to use it,
  you should vendor the code into your application. SLY has zero-dependencies. Although I am semi-retiring the project,
  I will respond to bug reports and still may decide to make future changes to it depending on my mood.
  I'd like to thank everyone who has contributed to it over the years. --Dave

- Experimenting with a line/cell DecL magic interpreter in Jupyter Lab to obviate the
  need for `build`.

### 0.20.2

- risk progression logic adjusted to exclude values with zero probability; graphs
  updated to use step drawstyle.

### 0.20.1

- Bug fix in parser interpretation of arrays with step size
- Added figures for AAS paper to extensions.ft and extensions.figures
- Validation "not unreasonable" flag set to 0
- Added aggregate_white_paper.pdf
- Colors in risk_progression

### 0.20.0

- `sev_attachment`: changed default to `None`; in that case gross losses equal
  ground-up losses, with no adjustment. But if layer is 10 xs 0 then losses
  become conditional on X \> 0. That results in a different behaviour, e.g.,
  when using `dsev[0:3]`. Ripple through effect in Aggregate (change default),
  Severity (change default, and change moment calculation; need to track the "attachment"
  of zero and the fact that it came from None, to track Pr attaching)
- dsev: check if any elements are \< 0 and set to zero before computing moments
  in dhistogram
- same for dfreq; implemented in `validate_discrete_distribution` in distributions module
- Default `recommend_p=0.99999` set in constsants module.
- `interpreter_test_suite` renamed to `run_test_suite` and includes test
  to count and report if there are errors.
- Reason codes for failing validation; Aggregate.qt becomes Aggregte.explain_validation

### 0.19.0

- Fixed reinsurance description formatting
- Improved splice parsing to allow explicit entry of lb and ub; needed to
  model mixtures of mixtures (Albrecher et al. 2017)

### 0.18.0 (major update)

- Added ability to specify occ reinsurance after a built in agg; this
  allows you to alter a gross aggregate more easily.

- `Underwriter.safe_lookup` uses deepcopy rather than copy to avoid
  problems array elements.

- Clean up and improved Parser and grammar

  > - atom -\> term is much cleaner (removed power, factor; now
  >   managed with prcedence and assoicativity)
  > - EXP and EXPONENT are right
  >   associative, division is not associative so 1/2/3 gives an error.
  > - Still SR conflict from dfreq \[ \] \[ \] because it could be the
  >   probabilities clause or the start of a vectorized limit clause
  > - Remaining SR conflicts are from NUMBER, which is used in many
  >   places. This is a problem with the grammar, not the parser.
  > - Added more tests to the parser test suite
  > - Severity weights clause must come after locations (more natural)
  > - Added ability for unconditional dsev.
  > - Support for splicing (see below)

- Cleanup of `Aggregate` class, concurrent with creating a cheat sheet

  > - many documentation updates
  > - `plot_old` deleted
  > - deleted `delbaen_haezendonck_density`; not used; not doing anything
  >   that isn't easy by hand. Includes dh_sev_density and dh_agg_density.
  > - deleted `fit` as alternative name for `approximate`
  > - deleted unused fields

- Cleanup of `Portfolio` class, concurrent with creating a cheat sheet

  > - deleted `fit` as alternative name for `approximate`
  > - deleted `q_old_0_12_0` (old quantile), `q_temp`, `tvar_old_0_12_0`
  > - deleted `plot_old`, `last_a`, `_(inverse)_tail_var(_2)`
  > - deleted `def get_stat(self, line='total', stat='EmpMean'): return self.audit_df.loc[line, stat]`
  > - deleted `resample`, was an alias for sample

- Management of knowledge in `Underwriter` changed to support loading
  a database after creation. Databases not loaded until needed - alas
  that includes printing the object. TODO: Consider a change?

- Frequency mfg renamed to freq_pgf to match other Frequency class methods and
  to accuractely describe the function as a probability generating function
  rather than a moment generating function.

- Added `introspect` function to Utilities. Used to create a cheat sheet
  for Aggregate.

- Added cheat sheets, completed for Aggregate

- Severity can now be conditional on being in a layer (see splice); managed
  adjustments to underlying frozen rv using decorators. No overhead if not
  used.

- Added "splice" option for Severity (see Albrecher et. al ch XX) and Aggregate,
  new arguments `sev_lb` and `sev_ub`, each lists.

- `Underwriter.build` defaults update argument to None, which uses the object default.

- pretty printing: now returns a value, no tacit mode; added `html` version to
  run through pygments, that looks good in Jupyter Lab.

### 0.17.1

- Adjusted pyproject.toml
- pygments lexer tweaks
- Simplified grammar: % and inf now handled as part of resolving NUMBER; still 16 = 5 \* 3 + 1 SR conflicts
- Reading databases on demand in Underwriter, resulting in faster object creation
- Creating and testing exsitance of subdirectories in Undewriter on demand using properties
- Creating directories moved into Extensions \_\_init\_\_.py
- lexer and parser as properties for Underwriter object creation
- Default `recommend_p` changed from 0.999 to 0.99999.
- `recommend_bucket` now uses `p=max(p, 1-1e-8)` if severity is unlimited.

### 0.17.0 (July 2023)

- `more` added as a proper method
- Fixed debugfile in parser.py which stops installation if not None (need to
  enure the directory exists)
- Fixed build and MANIFEST to remove build warning
- parser: semicolon no longer mapped to newline; it is now used to provide hints
  notes
- `recommend_bucket` uses p=max(p, 1-1e-8) if limit=inf. Default increased from 0.999
  to 0.99999 based on examples; works well for limited severity but not well for unlimited severity.
- Implemented calculation hints in note strings. Format is k=v; pairs; k
  bs, log2, padding, recommend_p, normalize are recognized. If present they are used
  if no arguments are passed explicitly to `build`.
- Added `interpreter_test_suite()` to `Underwriter` to run the test suite
- Added `test_suite_file` to `Underwriter` to return `Path` to `test_suite.agg` file
- Layers, attachments, and the reinsurance tower can now be ranges, `[s:f:j]` syntax

### 0.16.1 (July 2023)

- IDs can now include dashes: Line-A is a legitimate date
- Include templates and test-cases.agg file in the distribution
- Fixed mixed severity / limit profile interaction. Mixtures now work with
  exposure defined by losses and premium (as opposed to just claim count),
  correctly account for excess layers (which requires re-weighting the
  mixture components). Involves fixing the ground up severity and using it
  to adjust weights first. Then, by layer, figure the severity and convert
  exposure to claim count if necessary. Cases where there is no loss in the
  layer (high layer from low mean / low vol componet) replace by zero. Use
  logging level 20 for more details.
- Added `more` function to `Portfolio`, `Aggregate` and `Underwriter` classes.
  Given a regex it returns all methods and attributes matching. It tries to call a method
  with no arguments and reports the answer. `more` is defined in utilities
  and can be applied to any object.
- Moved work of `qt` from utilities into `Aggregate` (where it belongs).
  Retained `qt` for backwards compatibility.
- Parser: power \<- atom \*\* factor to power \<- factor \*\* factor to allow (1/2)\*\*(3/4)
- `` random` module renamed `random_agg `` to avoid conflict with Python `random`
- Implemented exact moments for exponential (special case of gamma) because
  MED is a common distribution and computing analytic moments is very time
  consuming for large mixtures.
- Added ZM and ZT examples to test_cases.agg; adjusted Portfolio examples to
  be on one line so they run through interpreter_file tests.

### 0.16.0 (June 2023)

- Implemented ZM and ZT distributions using decorators!
- Added panjer_ab to Frequency, reports a and b values, p_k = (a + b / k) [p](){k-1}. These values can be tested
  by computing implied a and b values from r_k = k p_k / [p](){k-1} = ak + b; diff r_k = a and b is an easy
  computation.
- Added freq_dist(log2) option to Freq to return the frequency distribution stand-alone
- Added negbin frequency where freq_a equals the variance multiplier

### 0.15.0 (June 2023)

- Added pygments lexer for decl (called agg, agregate, dec, or decl)
- Added to the documentation
- using pygments style in `decl_pprint` html mode
- removed old setup scripts and files and stack.md

### 0.14.1 (June 2023)

- Added scripts.py for entry points
- Updated .readthedocs.yaml to build from toml not requirements.txt
- Fixes to documentation
- `Portfolio.tvar_threshold` updated to use `scipy.optimize.bisect`
- Added `kaplan_meier` to `utilities` to compute product limit estimator survival
  function from censored data. This applies to a loss listing with open (censored)
  and closed claims.
- doc to docs \[\]
- Enhanced `make_var_tvar` for cases where all probabilities are equal, using linspace rather
  than cumsum.

### 0.13.0 (June 4, 2023)

- Updated `Portfolio.price` to implement `allocation='linear'` and
  allow a dictionary of distortions

- `ordered='strict'` default for `Portfolio.calibrate_distortions`

- Pentagon can return a namedtuple and solve does not return a dataframe (it has no return value)

- Added random.py module to hold random state. Incorporated into

  > - Utilities: Iman Conover (ic_noise permuation) and rearrangement algorithms
  > - `Portfolio` sample
  > - `Aggregate` sample
  > - Spectral `bagged_distortion`

- `Portfolio` added `n_units` property

- `Portfolio` simplified `__repr__`

- Added `block_iman_conover` to `utilitiles`. Note tester code in the documentation. Very Nice! 😁😁😁

- New VaR, quantile and TVaR functions: 1000x speedup and more accurate. Builder function in `utilities`.

- pyproject.toml project specification, updated build process, now creates whl file rather than egg file.

### 0.12.0 (May 2023)

- `add_exa_sample` becomes method of `Portfolio`
- Added `create_from_sample` method to `Portfolio`
- Added `bodoff` method to compute layer capital allocation to `Portfolio`
- Improved validation error reporting
- `extensions.samples` module deleted
- Added `spectral.approx_ccoc` to create a ct approx to the CCoC distortion
- `qdp` moved to `utilities` (describe plus some quantiles)
- Added `Pentagon` class in `extensions`
- Added example use of the Pollaczeck-Khinchine formula, reproducing examples from
  the `actuar` risk vignette to Ch 5 of the documentation.

### Earlier versions

See github commit notes.

Version numbers follow semantic versioning, MAJOR.MINOR.PATCH:

- MAJOR version changes with incompatible API changes.
- MINOR version changes with added functionality in a backwards compatible manner.
- PATCH version changes with backwards compatible bug fixes.
