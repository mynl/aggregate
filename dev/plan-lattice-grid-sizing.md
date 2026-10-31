# Plan: grid sizing for lattice severities, with reinsurance, windows and the infeasible case

`[Lattice-Grid-Sizing]`. Drafted 2026-10-31 against aggregate 1.43.1 (`03aaec6`). Five minor
bumps and a test step, in order: `[Dsev-Probability-Normalization]`, `[Lattice-Views]`,
`[Window-Holds-Severity]`, `[Portfolio-Follows-Unit-Grid]`, `[Lattice-Infeasible-Warning]`,
`[Correctness-Oracle]`. Versions are assigned at execution. The evidence is in
`dev/note-lattice-grid-sizing.md` (the note) and the study under `dev/lattice-study/`, which
stays in the tree so every bump can be checked against it. It supersedes the TODO entry
`[Layered-Dsev-Lattice]` and, from `D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`,
item 1 and the items `[Portfolio-Window-Severity-Reach]` and `[Aggregate-Ceded-At-Limit]`
merged into it there.

## Goal

A discrete severity whose atoms are whole multiples of a dyadic step $h$ has an aggregate on the
lattice $h\mathbb Z$ whatever the frequency, and every reinsured view of it (gross after a layer,
ceded and net of an occurrence layer, ceded and net of an aggregate cover) is on a lattice too.
An exact grid exists: every atom of every view on a grid point, the grid just large enough to
hold the law and the severity. After this plan the sizer finds that grid whenever it fits the
`log2` cap, a one-unit Portfolio gets the same grid as its unit, and when no grid fits the cap
the build completes, the verdict contains the word, and a warning gives the `log2` to pass.

The three programs that started this, at the default cap of 16:

| Program | 1.43.1 | After this plan |
|:--|:--|:--|
| `agg T 100 claims dsev [1000 1001] fixed` | `bs` 2, `log2` 16 from 0: atom 1001 split, `fails sev cv, agg cv` | `bs` 1, `log2` 10, origin 99,543, exact |
| `port P agg A dfreq [1] dsev [3]` | `bs` 1, `log2` 1, origin 3: a grid of width 2 cannot hold the claim, `fails pmf deficit 1` | `bs` 1, `log2` 2, exact, the same grid as `agg A dfreq [1] dsev [3]` |
| `agg AWK dfreq [1:100] dsev [1000 1001]` | `bs` 2, `log2` 16, `fails sev cv, agg skew`, no reason given | `bs` 2, `log2` 16 as today, verdict `grid infeasible for this lattice ...`, and a `LatticeGridWarning`: needs `log2` 17 at `bs` 1 |

Measured on the study's 858 generated programs against an independent oracle: 699 exact today
with 68 silent wrong answers, all reinsured; 817 exact with none silent under the prototype of
this plan; 27 infeasible at the cap, every one flagged and none warned.

## Rulings

Author, 2026-10-31, recorded in the note's Decisions section:

1. **Infeasible: warn and build.** `nttagg` raises; `agg` warns, builds, and the verdict
   contains the word. The warning carries `nttagg`'s sentence with the `log2` to pass.
2. **No growth past the cap** beyond what `WINDOW_LOG2_GROWTH` already allows the exact step
   in the windowed method. The cap is the user's.
3. **A sibling warning class**, `LatticeGridWarning`, its own text, the same `warn_once` keying
   as `InfeasibleGridWarning`.
4. **The `exact_discrete` row stays sized from 0.** Laying it at its own origin is not needed
   once the `windowed` row holds the severity.
5. **The rule** (author's formulation): run the usual method; if the severities are on a
   lattice take the lattice bucket (nice divisor first); if it is finer than the usual bucket
   the lattice cannot fit the cap, keep the usual grid and warn; otherwise snap `x_min` down
   to a multiple of it and shrink `log2` by the ratio while **both** the aggregate band and the
   severity hull still fit.

Proposed in drafting, to confirm at review:

6. **`dsev` probabilities are normalized at construction.** Within the tolerance `np.isclose`
   applies today (relative 1e-5) silently; outside it normalized with a `UserWarning` where
   today `logger.error` is the only trace. No raise: the `nttagg` parser already demands an
   exact sum for its own exact arithmetic (`parser.py` 2883), and a sloppy `[.3 .3 .3]` in an
   `agg` should build.
7. **`LATTICE_DEPTH` follows the cap.** `_dyadic_lattice_step(atoms, depth=log2)` from the
   sizer; the module constant remains the default for callers without a cap.

## Current behavior and why it is wrong

The sizer `bs_window` (`src/aggregate/_bucket_window.py` 905) builds one row per method
(`moment`, `exact_discrete`, `bounded_small`, `windowed`, `sbj`) through the inner `_size`
(1006), which for a lattice step walks `_lattice_candidates(h)` and takes the first bucket whose
`need = _need_log2(x_hi - x0, b)` fits the cap. The Portfolio has a second sizer,
`port_best_window` (1795) with `_port_lattice_bucket` (1727). Seven mechanisms, measured in the
note's "The defects, by mechanism":

1. **The lattice step is taken from the unlayered gross atoms only.** `Aggregate._severity_atoms`
   (`_aggregate.py` 7005) returns `support_atoms`, stored before any layer;
   `_exact_discrete_window` (7290) takes the support of the gross sum even when `p_total` is
   the net or ceded sum; `_pooled_lattice` (`_bucket_window.py` 2227) pools the same for a
   Portfolio and for `sharpen`. A layer, an occurrence layer or an aggregate cover off the gross
   lattice sizes on the wrong step. This is where the 68 silent failures live: `agg X 1 claims
   dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed` has net 250, 500 or 1500, takes
   `bs` 1000, and returns 0, 1000 or 2000 with the verdict `reinsurance; subject not
   unreasonable`.
2. **The `windowed` row is sized on the aggregate band alone.** `_size` has no term for the
   severity, and the severity-fit guard after the row (`sev_hi < extent`, about 1298) rejects
   the row when a claim does not fit, instead of growing `log2`. For `T` the row is 100,026 to
   100,085 at `bs` 1, `log2` 6, a grid of 64 that cannot hold a claim of 1,001; rejected, the
   sizer falls to `bounded_small` from 0 at `bs` 2. The severity grid has `N = 2**log2` points
   whatever the output origin, `xs_sev = (arange(N) - i0) * bs` with physical 0 at index `i0`
   (`_severity_negative_buckets`, `_aggregate.py` 4225; `i0` is 0 for a nonnegative severity),
   so `N * bs` must span the hull of the gross severity support and 0, written $R$ below.
3. **Windowing under a cover.** The Aggregate refuses to window under occurrence reinsurance
   (1258) but not under an aggregate cover, whose ceded amount sits below the window: `3 claims
   dsev [8 10] fixed aggregate ceded to 2 xs 4` cedes 2 always and returns 12. The Portfolio
   windows a book with a reinsured unit of either kind.
4. **Off-lattice coarsening is silent.** When no candidate fits the cap, `_size` drops the
   lattice and the resolution estimate runs; the only trace is `_lattice_note`'s fragment
   `lattice 1 too fine for the log2 cap, bs 2 off it` in `bs_window_df`.
5. **The Portfolio sizer is a second implementation.** It sizes a three-moment window on the
   total at 12 nines on both edges (the Aggregate trims the low edge to 6), picks its bucket in
   `_port_lattice_bucket` with no severity term, and runs each unit's `_bs_window` only to read
   its width. Of 718 single-unit programs built alone and as a one-unit Portfolio, the two
   grids agree in 117; the laws differ materially in 127 today and in 22 under the prototype
   (18 of them infeasible books where the Portfolio takes the coarser bucket). `port P agg A
   dfreq [1] dsev [3]` takes the nice divisor 1 of the step 3 at `log2` 1 and loses all its
   mass. The combine itself is sound: whenever both grids hold the law the Portfolio reproduces
   the Aggregate to 1e-9.
6. **`LATTICE_DEPTH` is 16** (`utilities.py` 343) while the cap is a parameter.
7. **`dsev` probabilities that do not sum to 1** are kept as given (`_severity.py` 3067 logs
   an error outside `np.isclose` and does nothing inside it); the moments are sums over them,
   so a deficit is an atom at 0 the law does not have, while the discretized density is
   normalized. `[0.333333 0.222222 0.444444]` leaves 1e-6 unaccounted and, for `100 claims dsev
   [999 1000 1001]`, takes the aggregate sd from 8.75 to 13.29 and the skew from -0.02 to
   -42.6; the moment window from those moments runs from 29,648 for a law 100 wide.

## The rule, as code

For an Aggregate, in `_size`: a new argument `hold` (the hull width $R$) and

```
need = max(_need_log2(x_hi - x0, b), _need_log2(hold, b))
```

in the candidate walk and in the free-bucket branch. Every row passes `hold`; for the rows from
0 it is a no-op whenever `x_hi >= R`, which the moment and exact rows satisfy, and for the
`windowed` row it is the correction. The candidate walk is already the author's rule: the first
candidate (nice divisor, then the step) whose `need` fits the cap is "lattice bucket no finer
than the usual one, origin snapped, `log2` shrunk by the ratio"; a walk that fits nothing is
"finer than the usual bucket", and that is the moment to record the infeasibility.

$R$ for an Aggregate: `max(hi, 0) - min(lo, 0)` over the gross atoms after any layer when every
component is discrete; otherwise `_severity_high_estimate(p)` above and the signed lower
quantile below, as `_severity_negative_buckets` reads it. For a Portfolio, the largest unit's.

For a Portfolio: a one-unit book takes the unit's grid verbatim; a book of several units applies
the same walk on the total's band with the pooled step of every unit's views and $R$ the largest
unit hull, and does not window when any unit is reinsured.

## Steps

Each step is one version bump, one commit, with the study rerun and its table recorded in the
Execution log. The study measures the library as built (`sweep.py base`,
`port_vs_agg.py base`); the prototype patches in `patches.py` target 1.43.1 and are not
expected to apply after the first code bump.

### `[Dsev-Probability-Normalization]`

**Change.** In the histogram builder at `_severity.py` 3058 to 3072 (the function returning
`xs, ps` for `dhistogram` and `chistogram`), divide `ps` by its sum when the sum is positive;
inside `np.isclose(sum, 1)` silently, outside it with a `UserWarning` giving the sum, in place
of today's `logger.error`. `SeverityDHistogram` then receives probabilities that sum to 1 and
its moments are those of the law the density discretizes.

**Files.** `src/aggregate/_severity.py`; `tests/test_discrete_severity.py`.

**Acceptance.** `agg X 100 claims dsev [999 1000 1001] [0.333333 0.222222 0.444444] fixed` has
`actual_sd` within 1e-3 of 8.749 and `actual_skew` within 1e-3 of -0.0217, the values of the
same program with the probabilities at full precision; `dsev [1 2] [0.3 0.3]` builds with a
warning and its `sev_m` is 1.5. The sweep is unchanged at 699 exact (its programs already pass
full-precision probabilities).

### `[Lattice-Views]`

**Change.** One method, `Aggregate._severity_views()`, returning the atoms of every view the
sizer must keep on the lattice: the gross atoms of each component after its layer (atoms above
the attachment, `min(x - a, y)`), the ceded and the net of each occurrence layer `(share, y, a)`
in `occ_reins`, and for each aggregate cover in `agg_reins` the points `share * gross`,
`share * a` and `share * y` (the latter two reduced modulo the largest gross atom, so a cover
far above the law does not widen the lattice search). The prototype is `view_atoms` in
`dev/lattice-study/patches.py`. Then:

- `_severity_atoms` returns the concatenation of the views (so `_severity_lattice`,
  `_pooled_lattice` and `_sharpen_lattice` follow with no change of their own);
- `_exact_discrete_window` takes its step from the views and its support from the view
  `p_total` sums (gross, or net or ceded of the occurrence program, `occ_kind` is `'ceded to'`
  or `'net of'`), floored so `hi >= R`; the reachability guard keeps its reading;
- a layered `dsev` contributes its layered atoms and conditional probabilities
  (`support_atoms` stays pre-layer, as `SeverityDHistogram._build` stores it).

**Files.** `src/aggregate/_aggregate.py` (`_severity_atoms` 7005, `_severity_lattice` 7044,
`_exact_discrete_window` 7290); `tests/test_lattice_bucket.py`, `tests/test_reins_buckets.py`.

**Acceptance.** `agg X 1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed`
builds at `bs` 250 with net 250, 500 and 1500 at their probabilities. `agg X 3 claims dsev [8
10] fixed aggregate ceded to 2 xs 4` returns 2 with probability 1. Sweep: 782 exact, 42 wrong
flagged, 5 wrong silent, 27 infeasible flagged, 2 no lattice (the note's `[views]` row).

### `[Window-Holds-Severity]`

**Change.** `_size` takes `hold`; `_row` passes it; `bs_window` computes $R$ once (a new
`Aggregate._severity_hull()` returning `(lo, hi)`, discrete from the layered gross atoms,
continuous from `_severity_high_estimate` and the signed lower quantile). The severity-fit guard
on the `windowed` row becomes a note in `bs_window_df` (`sev_hull=...`) and no longer sets
`applies`. The `windowed` row is not built when `agg_reins` is set (the gate at 1258 gains
`and not agg.agg_reins`). The docstring of `_severity_high_estimate` (7049) is restated in terms
of `i0`.

**Files.** `src/aggregate/_bucket_window.py` (`_size` 1006, `_row` 1126, windowed row 1254 to
1313, selection 1315); `src/aggregate/_aggregate.py`; `tests/test_bucket_sizing.py`,
`tests/test_balanced_window.py`, `tests/test_windowed_calibrate.py`.

**Acceptance.** `T` builds at `bs` 1, `log2` 10 with `density_df.index[0] == 99543` and the
exact law (the oracle's; checked by the test as the binomial on 100,000 to 100,100). The
`windowed` row of its `bs_window_df` reads `log2` 10, not 6. Sweep, Aggregate sets only: A 224
exact of 224, B 185 of 192, C 104 of 112, D 76 of 80, E 56 of 60, F 50 of 50, with no silent
wrong answer; the Portfolio sets move at the next step. The windowed books of the existing
suites keep their grids or shrink `log2`; a grid that grows is a finding to record.

### `[Portfolio-Follows-Unit-Grid]`

**Change.** In `Portfolio.update` (`_portfolio.py` 2776 to 2800), a non-signed book of one unit
with `bs == 0` takes `(bs, log2, x_min)` from the unit's `_bs_window(log2, 0, None, p)` and sets
`_combine_x_min` when `x_min > 0`; `_build_bs_window_df` still runs so `explain` has a table,
with the unit row as the source. For several units, in `port_best_window` and
`_port_lattice_bucket`: $R$ is the largest unit hull; the lattice walk's `need` and the
resolution branch's span both hold it; `window_nonsigned` also requires that no unit has
`occ_reins` or `agg_reins`. The `(x_min_ks, x_max_ks)` signed arithmetic is untouched.

**Files.** `src/aggregate/_portfolio.py`; `src/aggregate/_bucket_window.py`
(`port_best_window` 1795, `_port_lattice_bucket` 1727); `tests/test_bucket_sizing.py`.

**Acceptance.** `port P agg A dfreq [1] dsev [3]` builds at `bs` 1, `log2` 2 with
`P(3) == 1`. `port_vs_agg.py base`: all 718 single-unit programs on the same grid alone and in
a Portfolio with total variation below 1e-9. Sweep: 817 exact, 12 wrong flagged, 0 silent, 27
infeasible flagged; set H 45 exact. `cure.py base`: the 27 infeasible programs rebuilt at the
oracle's `log2` are all exact (22 today; the 5 that are not are two-unit books this step
fixes).

### `[Lattice-Infeasible-Warning]`

**Change.** When the candidate walk in `_size` fits nothing, record on the Aggregate (and on the
Portfolio, from `_port_lattice_bucket`) a `_lattice_feasibility` dict: the step, the bucket
used, the `log2` given, and the `log2` the step needs for the selected row. Then:

- `constants.py`: `class LatticeGridWarning(UserWarning)` beside `InfeasibleGridWarning` (293).
- `_bucket_window.py`: `warn_lattice_grid(ob)` beside `warn_infeasible_grid` (326), emitted
  from the same place in `update`, keyed with `warn_once` on `(name, step, log2)`. Text, with
  `nttagg`'s sentence from `ntt.py`: `AWK: the severity atoms are on the lattice of step 1 and
  the grid needs log2 >= 17 at bs 1; log2 16 was given, so bs 2 is used and the atoms do not sit
  on the grid. Pass build(..., log2=17) or add hints{log2=17} to the declaration.`
- `_validation.py`: the reading at 330 also sets `Validation.INFEASIBLE` when
  `_lattice_feasibility` is set, and the prefix at 176 reads `grid infeasible for this lattice
  (needs log2 17 at bs 1); ...` in that case.
- `Aggregate.bs_explanation` (`_aggregate.py` 1355) and `_lattice_note` carry the same numbers.
- `utilities.py`: `_dyadic_lattice_step(..., depth=LATTICE_DEPTH)` is called with `depth=log2`
  from `bs_window`, `_port_lattice_bucket` and `_exact_discrete_window`.

**Files.** `src/aggregate/constants.py`, `src/aggregate/_bucket_window.py`,
`src/aggregate/_validation.py`, `src/aggregate/_aggregate.py`, `src/aggregate/utilities.py`;
`tests/test_lattice_bucket.py`.

**Acceptance.** `AWK` builds with one `LatticeGridWarning` whose text contains `log2=17`, a
verdict starting `grid infeasible for this lattice`, and `build(..., log2=17)` returns the exact
law with no warning. Sweep: the 27 infeasible programs class as `infeasible-warned`, 0 as
`infeasible-flagged`; nothing else moves. `agg X 1 claims dsev [1 200000] fixed` at `log2` 18
finds the step 1 (today `LATTICE_DEPTH` 16 refuses it and the resolution estimate runs).

### `[Correctness-Oracle]`

**Change.** The oracle and the program sets move into `tests/` as a helper module
`tests/lattice_oracle.py` (the enumeration, the exact rational step, the two needs, the DecL
renderer) and `tests/test_lattice_oracle.py`: a seeded subset of about 120 programs across the
eight sets in the fast suite, the full 858 and the 718-program one-unit comparison under
`@pytest.mark.slow`. Each case asserts the class the specification gives it: exact programs
have total variation below 1e-4 and no warning; infeasible programs have the verdict word and
the warning, and build exactly at the oracle's `log2`. Where a program is in `nttagg`'s scope
(`dfreq` or fixed count, nonnegative integer atoms, no layer or reinsurance) the reference is
`nttagg` with exact fractional probabilities, as `cross_check_ntt.py` does; the enumeration
elsewhere. The named examples `T`, `P`, `AWK` and the smallest reinsured failure go into
`decl-testers.agg` in their own section, mirrored by `tests/test_lattice_bucket.py` as the
existing lattice section is.

**Files.** `tests/lattice_oracle.py` (new), `tests/test_lattice_oracle.py` (new),
`src/aggregate/agg/decl-testers.agg`, `tests/test_lattice_bucket.py`.

**Acceptance.** The fast suite gains under 10 seconds; the slow set runs in about a minute
serial. `dev/lattice-study/` stays as the exploratory copy with its pickles, and the note's "How
to rerun" block records that the tests are the maintained version.

## Tests, documentation, hygiene

- Tier 1 on the files named per step; tier 2 before each commit; tier 3 at each bump, with the
  numerics gate at `[Window-Holds-Severity]` and `[Portfolio-Follows-Unit-Grid]`.
- `tests/data/expected_specs.json` does not move: no grammar change.
- `dev/TODO.md`: retire `[Layered-Dsev-Lattice]` with the version; leave
  `[Signed-Portfolio-Origin]` and add `[Windowed-Reinsured-Book]` for the five residual
  Aggregates (below).
- Using page: `D:/Projects/Using/aggregate/lattice-buckets.qmd` gains the windowed example and
  the infeasible warning; read `D:/Projects/Using/CLAUDE.md` first; render with
  `refresh.ps1 aggregate lattice-buckets`; the deploy is the author's.
- The note's "How to rerun" block is the check after each bump: `sweep.py base`,
  `port_vs_agg.py base`, `cure.py base`, `cross_check_ntt.py`.

## Not in this plan

- **Windowing a reinsured book.** Five Aggregates need a windowed grid and carry occurrence
  reinsurance or an aggregate cover (`100 claims dsev [1000 1001] occurrence net of 25% po 1500
  xs 200 fixed`: from 0 needs `log2` 19, windowed 12). The refusal to window is right today
  because the occurrence machinery carries the severity on the output grid; lifting it is
  `[Windowed-Reinsured-Book]`. With this plan they class as infeasible and warned.
- **A signed book sized from its realized origin**, `[Signed-Portfolio-Origin]`.
- **Atoms with no dyadic step** (`30% po` of integer atoms, decimal atoms, thirds): the
  continuous estimator, as today.
- **`nttagg`'s own sizing**: it pins `bs` 1 from 0 and raises below the support's `log2`; it
  stays the exact reference, not a sizing model.

## Alternatives considered

- **The `exact_discrete` row at its own origin** (`force_origin=True`), which would give a
  fixed or `dfreq` count its exact support as the window with no moment estimate. Withdrawn by
  the author: not needed once the `windowed` row holds the severity, and a prototype that added
  it as a separate selection step broke four small books through its slack placement.
- **Raising when infeasible**, as `nttagg` does. Ruled out: an `agg` is often one unit of
  something larger, and a flagged build is worth more than an exception.
- **Growing `log2` past the cap automatically.** Ruled out: the cap is the user's; the warning
  gives the number.
- **A width-combine for the Portfolio window** instead of the three-moment fit on the total.
  Not pursued here: with the one-unit case taken verbatim and the multi-unit walk holding the
  unit hull, the remaining Portfolio residuals are reinsured books, the same as the Aggregate's.

## Execution log

(filled at execution: version per step, the sweep and comparison tables after each bump,
divergences from the steps above)
