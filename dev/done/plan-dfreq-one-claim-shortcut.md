# Plan [Dfreq-One-Claim-Shortcut]: `dfreq [1]` takes the same exact path as `1 claim ... fixed`

> **Status: EXECUTED at `1.0.0a303`, 2026-08-18.** All four phases, in one
> commit. No grammar change and no spec-snapshot churn, as drafted. Execution
> notes at the foot.

## Finding (verified 2026-08-18)

The fixed-1 shortcut in `_aggregate_compute.freq_sev_convolution` (line 96)
returns `sev_density.copy()` when `sum(en) == 1 and freq_name == 'fixed'`, so
`1 claim ... fixed` is bit for bit the discretized severity. `dfreq [1]` builds
`FrequencyEmpirical` (`freq_name = 'empirical'`), misses the gate, and runs the
full `ift(freq_pgf(ft(sev)))` round trip: an identity executed numerically,
leaving machine-epsilon dust (measured ~7e-18) where the fixed spelling is
exact. The same gate is duplicated in `bivariate.py:940` (netceded joint
density). Ruling: both spellings should operate like fixed.

## Design

One predicate, owned by `Aggregate` (the fixed case needs `en`, which the
`Frequency` object does not hold): a read-only property, proposed name
`one_claim` (vetted, no collision), True when the claim count is identically 1:

- `freq_name == 'fixed'` and `sum(en) == 1` (the existing gate), or
- `isinstance(frequency, FrequencyEmpirical)` and the validated support is
  exactly `{1}` (covers `dfreq [1]` and any renewal that degenerates to one
  claim; support check, never the mean, since `dfreq [0 2] [.5 .5]` has mean 1).

`freq_sev_convolution` stops sniffing frequency semantics: replace its
`en` / `freq_name` parameters with a single `one_claim=False` boolean computed
by the caller. Internal pure function; call sites are `_aggregate.py:3992` and
`tests/test_aggregate_compute.py` only.

## Phases

1. **[Predicate]** Add `Aggregate.one_claim` (property, NumPy docstring, the
   why in Notes).
2. **[Core]** `freq_sev_convolution`: swap `en` / `freq_name` for `one_claim`;
   update the caller in `_aggregate._fft_aggregate` and the docstring.
3. **[Bivariate]** `bivariate.py:940`: replace the inline gate with
   `agg.one_claim`.
4. **[Tests-Hygiene]** Parity test: `dfreq [1]` and `1 claim ... fixed` both
   satisfy `np.array_equal(agg_density, sev_density)`; negative control
   `dfreq [0 2] [.5 .5]` does not shortcut; migrate the existing compute-test
   signatures. CHANGELOG section, version bump, one-line commit.

## Acceptance and scope

- Regression bar: every non-degenerate aggregate byte for byte unchanged
  (the shortcut only widens, the FFT path itself is untouched).
- `ftagg_density` is still computed on the shortcut path (Portfolio needs it);
  that stays.
- Out of scope: the signed / windowed branch has no shortcut today for either
  spelling and keeps that parity; the a272 dfreq mixture-weighting bug is a
  separate open item.


## Execution notes (`1.0.0a303`)

Four small departures from the plan as drafted, all recorded here.

1. **Zero probability atoms.** The plan said "the validated support is exactly
   `{1}`". `validate_discrete_distribution` makes the outcomes distinct and
   ascending but never drops a zero mass, so the test is on the atoms carrying
   positive probability rather than on the raw `freq_a` array. `dfreq [0 1]
   [0 1]` therefore shortcuts, and has a test.
2. **The property returns a real `bool`.** The fixed branch computes
   `np.sum(self.en) == 1`, which is a `np.bool_`; both branches coerce, so
   `a.one_claim is True` holds and callers can use it as a plain flag.
3. **The baseline harness moved, and was regenerated.** Not anticipated in the
   plan, and it is the clearest evidence for the regression bar: of the ten
   captured cases exactly one, `Base.DfreqOne`, moves (1e-11 relative, tail
   rows and the third moment), and the other nine rewrite byte identical
   parquet. Regenerated with `tests/baseline/capture.py` and committed
   alongside the change, per that harness's own protocol.
4. **Two documentation surfaces named the old asymmetry.** `dev/FEATURES.csv`
   gained a `frequency-api` row for `one_claim` (the auditor listed it as an
   undocumented capability), and
   `docs/2_aggregate_overview/pipeline-aggregate.rst` carried a paragraph and a
   conclusion line stating that `dfreq [1]` runs the full round trip and picks
   up fuzz. Both updated in the same commit.

The acceptance items all held: the signed and windowed branch is untouched and
keeps its no-shortcut parity for both spellings, `ftagg_density` is still
computed on the shortcut path, and the a272 dfreq mixture-weighting bug remains
a separate open item.
