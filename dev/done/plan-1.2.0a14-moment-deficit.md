# Plan 1.2.0a14: empirical moments stop parking dust at the top of the grid

Status: revised 2026-10-12 after review, one bump, `1.2.0a14` **[Moment-Deficit-Genuine-Only]**.
Supersedes `dev/note-fuzz-removal-phantom-tail-mass.md` (2026-10-11) and closes the TODO entry
**[Fuzz-Removal-Phantom-Tail-Mass]** (`dev/TODO.md`; first diagnosis in
`dev/done/plan-a400-v1-loose-ends.md`, phase 3).

The first draft of this plan proposed two bumps and a rule that switched the deficit relocation
off whenever `normalize=True`. The review found that rule to be a large regression (section
"What the first draft got wrong" below), so the plan now lands as a single bump with two smaller
substitutions, both chosen so that the set of cases whose behavior changes is provably a subset
of the cases that are wrong today.

## Goal

`est_skew` (and `est_cv`, and the `ex1`..`ex3` rows of `stats_df`) should be the moments of the
density the FFT actually produced, cleaned of arithmetic noise and nothing else. Today two
mechanisms corrupt them, by up to a factor of ten on cases a user meets in practice (the
reproductions book's GEMAct chapter is the latest).

The acceptance standard is the author's: **no right-to-wrong transitions**. Every `est_skew` or
`est_cv` that moves must move from wrong to right or from wrong to less wrong, with the handful
of accepted exceptions named in this document and nowhere else.

## Current behavior

The aggregate empirical moments are computed at `Aggregate.update_work`
(`src/aggregate/_aggregate.py:4568`):

```python
agg_clean = remove_fuzz(self.agg_density)          # zero every |p| < 2.22e-16
_mw = xsden_to_mwrangler(self.xs, agg_clean)
```

`xsden_to_mwrangler` (`src/aggregate/moments.py:552`) computes plain sums
$\sum_k x_k^j p_k$, then adds a correction for a defective distribution:

```python
pg = 1 - den.sum()                                  # "missing" mass
ex3 = np.sum(xs**3 * den) + pg * (xs[-1] + bs)**3   # placed at the grid's right edge m
```

That is, it reports the moments of $X \wedge m$ on the reading that any shortfall is mass at or
beyond the grid edge $m$; equivalently $\mathsf{P}(X\wedge m)^j = \int_0^m j x^{j-1} S(x)\,dx$
with $S \ge p_g$ on $[0, m)$. The relocation is applied unconditionally; `VALIDATION_NOISE`
(1e-12, unchanged since a17) only gates the log message beside it.

The relocation is right when the shortfall is genuine truncation, and genuine truncation is
common: `utilities.ift` (`utilities.py:126`) slices off the padded half of the inverse FFT,
`temp[0:len(temp) >> padding]`, so with the default `padding=1` mass past the grid is **dropped,
not wrapped**, whatever `normalize` says. A `normalize=False` severity adds a second genuine
source. The relocation is wrong in the two situations where a shortfall is not truncation:

1. **The shortfall is mass `remove_fuzz` deleted in place.** The absolute cutoff 2.22e-16 zeroes
   genuine far-tail density on a tall grid (GEMAct at `log2=22`: 3.7M buckets, 2.0e-11 of mass,
   centered near x = 3e8). The wrangler moves it to the edge, x = 2.1e9, where $x^3$ makes it
   47% of the third moment.
2. **The shortfall is rounding.** The aggregate sums to one only to rounding, typically
   |1 - sum| ~ 1e-16, but with a large claim count the PGF amplifies severity rounding to ~1e-12
   (`Bruno2006.B7`, 91,000 claims: sum = 1 + 2.2e-12). On a 2^20 grid, edge^3 ~ 1e18, so 7e-16 of
   rounding adds ~770 to the third moment of ten dice, whose third moment is ~46,000 and whose
   central third moment is 0. A negative shortfall (sum above one) is relocated too, as negative
   mass at the edge, which no reading of the density supports.

Separately, the absolute cutoff itself loses genuine tail. With the relocation corrected, GEMAct
at `log2=22` reads 7.269 against 7.2841, and an ordinary `lognorm 100 cv 3` with 10 claims at
`bs=50`, `log2=16` reads 9.944 against 10.000, because the lognormal tail below 2.2e-16 is real
and is simply thrown away.

The note's diagnosis of the die cases ("noise survives the eps threshold and dominates") does not
reproduce: with the relocation removed, `remove_fuzz` gives ten dice at 2^20 exactly 0. The whole
die error is mechanism 2.

## The change, one bump

Two substitutions at the moment sites, nothing else.

### Relocate only a genuine, positive deficit

`xsden_to_mwrangler(xs, den, deficit=None)` gains a keyword `deficit`. `None` keeps today's
source, `1 - den.sum()`. Whatever its source, the deficit is then gated **one-sided**:

```python
pg = deficit if deficit is not None else 1 - den.sum()
if pg <= VALIDATION_NOISE:
    pg = 0.0
```

This is exactly the condition the `logger.info` beside it already uses (`moments.py:607`), so
relocation and log finally agree. The gate applies to every caller (severity, reinsurance,
sample, bivariate). The Notes section is rewritten to say when the relocation is right (mass the
grid could not hold) and when it is not (rounding, excess mass, mass a cleaning step deleted).

The moment sites pass the deficit measured on the **raw** density, before any cleaning:

- `Aggregate.update_work` (`_aggregate.py:4568`) and the stage helper `_moments`
  (`_aggregate.py:4684`): `deficit=1 - raw.sum()` where `raw` is the array before cleaning.
- `Portfolio.update` (`_portfolio.py:2946`) and `create_from_sample` (`_portfolio.py:376`).
  **Ordering trap:** `Portfolio.update` calls `self.remove_fuzz(log='update')` at `:2893`, which
  mutates `density_df['p_total']` in place before the moments are taken, and
  `create_from_sample` passes `remove_fuzz=True` explicitly at `:363`. The raw sum must be
  captured at `:2866` / `:2891`, right after `p_total` is written, and carried to the moment
  call. The `remove_fuzz` call itself is left where it is.

There is **no `normalize` rule**. With `normalize=True` the deficit is still genuine whenever
the padded inverse FFT dropped mass, which is the common case on a tight grid.

The set of cases that stop relocating under this rule is: negative deficits (always wrong today),
deficits at or below 1e-12 (at most 2e-4 absolute skew either way, measured on GEMAct at
`log2=18`), and mass the cleaning step deleted (always wrong, since it lived inside the grid).
Nothing that relocates today for a genuine reason stops.

### Clean with a noise floor read from the density, capped at today's cutoff

New helper `utilities.noise_floor(den)`:

```python
neg = den[den < 0]
return min(np.finfo(float).eps, -neg.min()) if len(neg) else 0.0
```

and the four moment sites above call `remove_fuzz(raw, eps=noise_floor(raw))` in place of
`remove_fuzz(raw)`. When nothing is negative the floor is 0 and the raw density is used as is.

Why the negatives: the true law is non-negative and inverse-FFT round-off is sign-indefinite, so
the largest negative value measures the noise amplitude. Measured floors on the panel run from
4e-19 to 6e-17, always below 2.22e-16, so on those cases the new cut keeps genuine tail that the
absolute cutoff threw away.

Why the cap: with `eps = min(eps_machine, maxneg)` the set of buckets zeroed is always a
**subset of today's**. The cleaning can never discard more genuine tail than it does now; below
machine epsilon it discards less. The cap is not hypothetical. Six corpus entries have a negative
bucket larger than machine epsilon (`Cc.Freq24.Binomial` has one at 4.5e-4, a single artifact
bucket at `log2=4`; `SignedLump` 7.2e-15; `WindowedSimple` 5.2e-15; `WindowedContinuous`
2.3e-16). Uncapped, the floor would zero every bucket below that value: `Cc.Freq24.Binomial`
moves 0.4% and `SignedLump` moves 0.2%. Capped, both stay where they are. This answers the
note's "one extreme value sets the floor" worry without a percentile rule.

`k` (a multiple of the measured floor) is not a parameter. With the cap in place the only
effect of `k > 1` is to discard more tail below machine epsilon; the die residual at `k = 1` is
7e-7, which is noise.

`density_df.p_total` (`_aggregate.py:1850`), `reins_density_df` and `Portfolio.remove_fuzz`
keep the plain `remove_fuzz`: they are probability views, where the absolute cutoff does no harm.

Name vetting: `deficit` as a keyword of a module function cannot shadow anything and matches the
existing vocabulary (`pmf_deficit`, `explain_validation(rv, deficit=None)`, `self._deficit`).
`noise_floor`: no callable or attribute of that name exists (`ntt.py` uses it only as a local
name in prose); `rg noise_floor src/aggregate` before adding it.

### Comments owed

Fix the two comments the TODO entry says are owed: the block above `_aggregate.py:4568` ("no
negative density even under aliasing" is false: GEMAct has 813,214 negative buckets) and the
`remove_fuzz` Notes ("safe and lossless" is false on a tall grid). The `_aggregate.py` comment
should also say why the deficit is taken from the raw array and why the floor is capped.

## Measurements

### Panel

Skewness on a ten-case panel. "clean" is `remove_fuzz` then no relocation; "raw" is the raw
density, no relocation; "floor" is the capped noise-floor cut, no relocation (identical to the
uncapped cut on every panel row, since every panel floor is below machine epsilon).

| case | theory | current `est_skew` | clean | raw | floor | 1 - sum(raw) |
|---|---:|---:|---:|---:|---:|---:|
| ten dice, `bs=1`, 2^20 | 0 | 4.875 | 0 | 1.185 | 0 | 3.3e-16 |
| `PoiDie` 10 claims, 2^20 | 0.393507 | 1.216 | 0.393507 | 0.4679 | 0.393507 | 3.3e-16 |
| `LN` 1 claim lognorm 200 cv 2, `bs=6.427`, 2^21 | 14 | 142.5 | 13.982 | 13.9996 | 13.9996 | 0 |
| GEMAct gross, `bs=500`, 2^22 | 7.2841 | 11.916 | 7.2691 | 7.2841 | 7.2823 | 0 |
| `Bruno2006.B7` | 0.0039446 | 0.0039446 | 0.0038977 | 0.0039913 | 0.0039976 | -2.2e-12 |
| `J.Re14`, 2^16 | -7.691 | -6.060 | -7.691 | -7.691 | -7.691 | 4.6e-13 |
| `CurveLognormal` | 7.875 | 7.8746 | 7.8737 | 7.8743 | 7.8743 | 0 |
| `G.Mixed10` | 566.0 | 529.5 | 448.9 | 483.7 | 483.7 | 0 |
| `Mack2003.Lognorm` | 136.37 | 130.26 | 125.91 | 128.51 | 128.51 | 0 |
| `Mata2005.Lognorm` | 2628 | 1886 | 1345.6 | 1570.0 | 1570.0 | 0 |

`J.Re14` has no theoretical skewness in `validation_df` (a net-of-reinsurance mixture); the -7.691
in its theory cell is the value every no-relocation rule gives at every grid. It is the cleanest
witness: current `est_skew` is -6.06, -2.79 and **+7.96** at `log2` 16, 18 and 20; every
no-relocation column is -7.691 at all three.

### Corpus sweep

The sweep builds every `agg` in `library.agg`, `reproductions.agg` and `_test_suite.agg` (370
recipes; 348 build without an explicit `bs`, 15 fail for unrelated reasons such as a missing
`bs` or infinite variance) and recomputes the skewness under each rule from `agg_density`. The
comparison is against `validation_df.loc['Agg', 'Sk']` on the 265 entries with a finite nonzero
theory; "worse" means the relative error grows by more than 0.1%.

| rule | movers | better | worse |
|---|---:|---:|---:|
| first draft (`deficit=0` when `normalize`, uncapped floor) | 65 | 7 | 57 |
| this plan (one-sided gate on the raw deficit, capped floor) | 16 | 3 | 12 |

The 12 "worse" under this plan, in full:

- **`Bruno2006.B7`**, 0.0039446 to 0.0039976 (1.3% on a skew of 0.004). Today's exact match is
  the relocation of a negative rounding deficit landing on the right answer by luck; the plan
  accepts this one.
- **`Mata2005.M.v3` / `.v4`, `G.Mixed10` / `11`, `Mack2003.S.v9`** (theoretical skewness in
  the hundreds or thousands). Not estimable on their grids: today's values are not stable
  (`Mata2005.Lognorm` reads 1886, 2981 and 14336 at `log2` 16, 18, 20; `M.v3` and `M.v4` differ
  by 50% today on the same law) and sit nearer theory only because mass parked at the edge
  crudely stands in for tail the grid cannot hold. The new values are lower and steadier. A
  reader of those entries will notice; the CHANGELOG says so.
- The remaining six (`WindowedContinuous`, `Pittarello2024.PLM.Gross` and `.Table1`,
  `Cc.Freq24.Binomial`, `Mack2003.S.v6`, `CurveLognormal`) move by at most 1e-3 relative, most
  by 1e-5, in both directions of "worse" that the 0.1% test can detect.

`est_cv` moves on 8 entries: the five heavy-tail entries above, `J.Re14`, and two at 1e-5.

Every other mover is a fix: `J.Re14` to -7.691, the symmetric die and random-walk shapes from
1e-13 to 1e-14, `Mata2005.M.v4` from above theory to below it with `.v3`.

### What the first draft got wrong

The draft's rule `deficit = 0 when normalize=True` assumed that with a normalized severity any
aggregate shortfall is arithmetic. The padded inverse FFT drops mass past the grid regardless of
`normalize` (see "Current behavior"), so on every tight grid the shortfall is genuine. Under the
draft rule the sweep showed 57 regressions, among them `WindowedSimple` 0.0004 to **-871**,
`WindowedContinuous` 0.0044 to **-0.39**, `Z.Ref.Signed` 1.15 to 9.55 (signed windows, where
the two-sided deficit is genuine), and the whole `Cb.Freq*` mixed-frequency block at `log2=6`
(deficits ~1e-4, errors doubling or tripling). A probe outside the corpus shows the same thing
on `agg 10 claims sev lognorm 100 cv 3 poisson`, `bs=50`: at `log2=8` the raw deficit is 1.1e-4
and today's 3.637 (theory 10) would have become 3.496.

The draft also split the work into two bumps. Step 1 alone regresses ordinary lognormal tails at
`log2=16` (the probe above: 10.0012 to 9.9439; `CurveLognormal` and the Pittarello rows show it
too) and step 2 is what recovers them, so the intermediate commit would have been a known
regression on bread-and-butter cases and no use as a bisect point.

### Residual, accepted

A `Bruno2006.B7` twin whose amplified rounding lands at +2.2e-12 rather than -2.2e-12 passes the
one-sided gate and still relocates, about 1% on that one shape. This is today's behavior, so it
is not a regression. The principled fix is for `freq_sev_convolution`
(`_aggregate_compute.py:23`) to report the mass the padding slice discards, which is the only
direct measurement of truncation, and for `update_work` to pass that as the deficit. Not worth
the plumbing now (the Portfolio path builds `p_total` from the product of transforms and would
need its own measurement); it is the next step if this ever bites, and is noted in `dev/TODO.md`
under the closing entry.

`_validation.pmf_deficit` reads `density_df['p_total']` after `Portfolio.remove_fuzz` has run,
so `Portfolio._deficit` carries the same fuzz contamination as the moments did. It feeds a
warning gated at 1e-4, where fuzz cannot matter. Noted, not fixed.

## Files

- `src/aggregate/moments.py`: `xsden_to_mwrangler` keyword and one-sided gate, Notes.
- `src/aggregate/utilities.py`: `noise_floor`; `remove_fuzz` Notes.
- `src/aggregate/_aggregate.py`: the two moment sites, the owed comment.
- `src/aggregate/_portfolio.py`: the two moment sites, raw-sum capture before `remove_fuzz`.
- `tests/test_moment_deficit.py` (new); snapshots that record `est_skew` / `est_cv`
  (`tests/data/peg_baseline.json` via `tests/capture_peg_baseline.py`, and whatever else the
  suite reports); `CHANGELOG.md`; `pyproject.toml`; `dev/TODO.md`.
- At completion: delete `dev/note-fuzz-removal-phantom-tail-mass.md` (this plan carries its
  content) and move this plan to `dev/done/` when the author says so.

## Tests

`tests/test_moment_deficit.py`, against `validation_df.loc['Agg', 'Sk']` unless stated:

- Fixes: ten dice at 2^20 (0, abs 1e-6); `PoiDie` at 2^20 (abs 1e-6); `LN` at 2^21 (rel 2e-3);
  `J.Re14` at `log2` 16, 18, 20 all equal to each other (rel 1e-4); GEMAct at 2^22 (rel 1e-3),
  marked `slow` if it costs more than a second or two.
- Guards, the three shapes that caught the first draft, each within 1e-6 relative of today's
  value: `WindowedSimple` (signed window, genuine two-sided deficit), `Cb.Freq12.IG` (mixed
  frequency on a tight grid, genuine deficit 2e-4), `Cc.Freq24.Binomial` (one artifact bucket
  at 4.5e-4, the cap must bind).
- Genuine truncation still relocates: the probe aggregate at `log2=8` has `est_m` exceeding
  $\sum x p$ by `deficit * edge` with `deficit` ~1.1e-4, so the rule the wrangler was written for
  survives with `normalize=True`.
- Unit tests of `xsden_to_mwrangler` alone: deficits of +1e-15, -1e-15 and -1e-6 are ignored; a
  deficit of 1e-6 is relocated; an explicit `deficit=` overrides `1 - den.sum()`.
- Unit tests of `noise_floor`: 0 when nothing is negative; the largest negative magnitude when
  that is below machine epsilon; machine epsilon when it is above.

## Acceptance

Tier 2 (`uv run pytest`), then tier 3 and the numerics gate at the bump
(`-m 'slow or not slow' -W error::RuntimeWarning`). Rerun the corpus sweep,
`uv run python scripts/moment_rule_sweep.py` (it builds the three corpora and compares today's
rule with the plan's in about a minute; `scripts/README.md` describes it), and read every mover
by hand, not blanket-recapture: the expected movers are the 16
listed above plus the noise-level die and random-walk shapes; any other mover is a finding and
stops the bump. The CHANGELOG entry says in one sentence that `est_skew` moves for thick-tailed
laws and very wide grids, and names B7 as the one known case that gets worse.

## Execution log, 2026-10-12

Executed as one bump, `1.2.0a14`. Every item below is a divergence from the text above, recorded when it was made.

- **`noise_floor` steps one float above the largest negative magnitude.** The plan's snippet returns `min(eps, -neg.min())`, and `remove_fuzz` cuts strictly (`|x| < eps`), so the most negative bucket itself survived the cut as a negative value. On ten dice at 2^20 that single bucket, `-4e-18` at x = 393,250, read a skewness of -1.5e-3 where the panel says 0. The panel was measured with the note's one-sided `raw < floor`, which does zero that bucket; the sweep script's `plan` column used the strict cut and so disagreed with the panel on cases the corpus does not contain. Shipped: `min(eps, nextafter(-neg.min(), inf))`. With it the panel reproduces (ten dice 0, `PoiDie` 0.393507, `LN` 13.9996, GEMAct 7.2823) and the corpus sweep is unchanged at 16 movers, 3 better, 12 worse. The sweep script's `plan` rule now calls `noise_floor` rather than restating it, so the column equals `est_skew` from this version on.
- **`J.Re14` is fixed at its default grid only.** The plan's test asked for `log2` 16, 18 and 20 to agree at -7.691. At 18 and 20 the raw deficit is +1.4e-12 and +4.5e-12, positive amplified rounding above the 1e-12 gate on a path that produces no negative buckets at all, so it still relocates and the values are today's (-2.789, +7.961). This is the "Residual, accepted" mechanism the plan already names for a `Bruno2006.B7` twin, so wrong stays wrong and nothing right became wrong. The test pins 16 at -7.691 and documents 18 and 20 as the residual: deficit above the gate, floor zero, value equal to the legacy rule. The sentence "every no-relocation column is -7.691 at all three" is true of the no-relocation columns and not of the plan's rule.
- **`Cc.Freq24.Binomial` is not within 1e-6 of today.** The plan lists it both among the six small "worse" movers and among the guards that must stay within 1e-6 of today's value. It moves 4.6e-4 relative because its deficit is -9.5e-7, negative rounding that used to be relocated as negative mass and now is not. The guard the plan wants there is that the cap binds, so the test asserts `noise_floor` returns machine epsilon on that density, that `est_skew` equals the capped recomputation, and that the uncapped cut would differ by more than 1e-3.
- **The sweep's `WindowedSimple` row disagrees with the object by 4%, on both columns.** The script's naive central-moment formula cancels catastrophically on a mean of 1.5e7; with the wrangler's arithmetic on the object's own grid the new rule is within 2e-9 of the legacy rule. The guard test uses the wrangler, not the script.
- **`create_from_sample` measures its deficit on the array in hand.** After the switcheroo `p_total` is the sample's own column, not the FFT output, so there is nothing earlier to capture; the deficit is `1 - p.sum()` of that column before the local cleaning. `Portfolio.update` captures `raw_deficit` right after `p_total` is written on both the rolled and the plain path, as specified.
- **No control list row.** `V:/worktrees/dev-files.md` no longer exists (retired 2026-09-25), so the project skill's row update is a no-op.
- **Plan and note stay in `dev/` until the author says done**, per the plan's own closing line and the house rule. The note is not deleted for the same reason: both retire together.
- **`scripts/moment_rule_sweep.py` ships in the bump.** It was untracked; the plan names it as the acceptance check and `scripts/README.md` already describes it, so it belongs to this unit.
- **The `WindowedSimple` guard routes through the wrangler.** A hand-written legacy formula in the test disagreed with the wrangler by 4% for the same reason the sweep did. `legacy_rule` in the test now calls `xsden_to_mwrangler` with the cleaned density's own shortfall as the explicit deficit, which is the old arithmetic exactly whenever that shortfall exceeds the gate; it asserts so.
- **Two library validation baselines move** (`tests/test_library_entries.py`), found by tier 1 and outside the sweep, which covers `agg` recipes only. `RawLognorm` loses `SEV_SKEW`: the flag came from the `gross_empirical` stage column, where the old `_moments` helper cleaned the severity density and read 14.151 against 14 (error 1.08e-2) while the main severity site read 13.964 on the same array; the two sites now agree and the error is 2.6e-3. A fix. `ThickThickPortfolio` gains `AGG_SKEW` on unit A (100 claims, lognorm 50 cv 3): raw deficit -3.8e-15, so nothing is truncated, and today's 3.1578 against 3.1623 is 9.4e-13 of deleted genuine tail parked at x = 5.2e6; every no-relocation reading is 3.124 to 3.127 (error 1.15e-2, tolerance 1e-2). The heavy-tail class the plan accepts for five named entries, but not named, and a shipped baseline. Reported as a finding per the plan's own rule; the author accepted it on 2026-10-12 and the baseline edit ships in the bump.
- **Exhibit canonical snapshots (52 of 295) fail and need a recapture, and the file is the author's.** The cause is a fix: three dice carried a skewness of 5.3e-14 of dust, which made the exhibit auto-format choose `si` for that column; it now reads exactly 0 and the format is `dec`. The recapture is `uv run python tests/capture_exhibit_snapshots.py`, but `tests/data/exhibit_snapshots.json` was already modified in the working tree at 16:40, after the author's `formats-raw.yaml` commit and before this session touched tests, so it holds the author's own uncommitted recapture (`1.000` to `100.0%`, `dec` to `pct`). HEAD's code passes against that modified file; this change does not. Recapturing would sweep the author's work into the bump commit, so it waited on the author, who ruled on 2026-10-12 that the combined file ships in the bump. Recaptured on top of the author's copy and the diff between the two read in full: beyond the `si` to `dec` flip, dust-level cells change sign, and the empirical `stats_df` rows of the exhibit objects move in the fifth to sixth significant figure (the largest 9e-5 relative, on the `PeelMarginal` skew), with the two moment columns that disagreed there now agreeing, the same stage-helper fix that cleared `RawLognorm`. The 1e-9 snapshot tolerance is why 52 keys failed rather than the 42 whose text changed.
- **Baseline harness and spotchecks recaptured** (`tests/baseline/capture.py`, `capture_spotchecks.py`): moves of 1.5e-11 to 8e-9 relative on the empirical `stats_df` rows and 1e-12 on `exgta`, read in the diff. The manifest recorded `1.2.0a13` because the first capture ran before the version line moved; both captures were rerun at commit time and it reads `1.2.0a14`.
