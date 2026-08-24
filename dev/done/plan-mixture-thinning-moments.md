# Plan: [Mixture-Thinning-Moments]

Status: drafted 2026-08-24, awaiting author sign off. No code written yet. Three decisions are open, listed in section 7.

Source of the mathematics: `C:/S/AI/notes/aggregate-presentations/decomposing-aggs-w-mixed-sev.md`, sections "The universal marked decomposition" and "Dependence and dispersion". This plan is the implementation reading of that note.

## 1. What is wrong

A severity mixture `sev [...] * expon wts [...]` splits the claim count across its components. The library carries that split through the per-component claim count: `_aggregate.py:2711` sets `_en0 = _en * _swt`, `_record_component` hands it to `MomentAggregator.add_f1s`, and `moments.py:132` calls `freq_moms(_en0)`.

That is correct only when the thinning of `N` by weight `w` is the same distribution family with its mean scaled by `w`. It is, for Poisson and for every mixed Poisson, which is why the defect has never surfaced. It is false for fixed, binomial, Bernoulli in general, empirical (`dfreq`), renewal (`years`), Neyman A and Pascal, and it makes logarithmic raise.

Measured on `3 claims sev [1 10] * expon wts [0.9 0.1]`, severity `EX = 1.9`, `EX2 = 21.8`. "Var truth" is the compound of the parent frequency with the pooled severity, which is also what the FFT computes. "indep now" is today's `independent` column.

| frequency | Var truth | indep now | indep under this plan |
|---|---|---|---|
| poisson | 65.400 | 65.400 | 65.400 |
| mixed gamma 0.5 | 73.522 | 69.472 | 69.472 |
| delaporte, sichel | 73.522 | 69.472 | 69.472 |
| geometric | 97.890 | 81.690 | 81.690 |
| bernoulli | 8.142 | 8.430 | 8.430 |
| fixed | 54.570 | **32.700** | 59.970 |
| binomial 0.5 | 59.985 | **49.050** | 62.685 |
| neymana 2 | 87.060 | **130.800** | 76.260 |
| pascal 0.5 2 | 62.693 | **36.773** | 64.043 |
| `dfreq[1 2 6]` | 71.417 | **774.333** | 68.417 |
| logarithmic | 62.693 | **raises** | 64.043 |

For `dfreq` the damage reaches the answer itself, not just the report. `agg MED.Picks dfreq[1] sev [2764 24548 275654 1917469 10000000] * expon wts [...]` has true mean 13,989.98 and reports a theoretic mean of 12,220,435 (the unweighted sum of the scales) against an FFT mean of 2,442,979 (the equally weighted mixture), because `_aggregate.py:3798` derives the density pooling weights from the same broken per-component frequency.

## 2. The recipe

Give every claim an iid latent label `Z` with `P(Z = i) = w_i`, and let `K_i` count the claims labeled `i`. Then `K_i` given `N` is `Binomial(N, w_i)` for **every** frequency `N`, with no closure or independence assumption. Taking expectations of the binomial raw moments turns the parent's first three raw moments into the component's:

```
k1 = w m1
k2 = w(1 - w) m1 + w^2 m2
k3 = w(1 - w)(1 - 2w) m1 + 3 w^2 (1 - w) m2 + w^3 m3
```

Two properties make this the right primitive. It needs only `(m1, m2, m3)` from the parent, which every frequency can supply at its own count, so nothing is ever asked to move its mean. And it is the identity map at `w = 1`, so the limit-profile arm passes through untouched.

The component aggregate `A_i` is then the ordinary compound of `K_i` with `F_i`, which `MomentAggregator.agg_from_fs` already computes. The note also gives the cross term in closed form, with `nu = E[N]` and `v = Var(N)`:

```
Cov(A_i, A_j) = w_i w_j mu_i mu_j (v - nu),   i != j
```

so `mixed` and `independent` reconcile exactly:

```
Var(A) = sum_i Var(A_i) + sum_{i != j} Cov(A_i, A_j)
```

I verified that identity numerically against every frequency family in the table above, including `dfreq`. It reassembles the true variance to machine precision in all eleven cases. That is the acceptance criterion for the whole plan.

The sign of `v - nu` also reads directly off the table: fixed, binomial and Neyman A give negatively correlated components, Poisson gives uncorrelated, and the mixed Poissons give positively correlated. The `mixed` minus `independent` gap becomes a quantity with a meaning rather than an artifact.

## 3. What this settles

`dfreq[n]` and `n claims ... fixed` agree by construction, since both parents report `(n, n^2, n^3)` and the map is a pure function of those. No rule is needed to enforce it.

The empirical frequency keeps an immutable mean. `freq_moms` is called once per exposure row at that row's real count and never at a weighted fraction of it, so the objection to a scaling empirical is honored by the architecture rather than by a guard.

`dfreq` with a severity mixture becomes a supported pairing, so no hard error is needed for it. That matters because `dfreq[1 2]` with a mixture is a well posed model with no `fixed` equivalent to redirect a user to, and the FFT already draws it correctly.

## 4. What moves

The `mixed` column and the FFT answer do not move for any program that works today. `mixed` is computed from the pooled severity against the parent's own moments, and the FFT applies the parent PGF once to the pooled density. Both are already right.

What moves is the per-component `freq` and `agg` rows in `stats_df` and the `independent` column derived from them, and only for fixed, binomial, empirical, renewal, Neyman A, Pascal and logarithmic. For those families the current values are not a different convention, they are wrong, so every move is a correction.

The two `reins` exhibit snapshots move for the separate reason in phase E.

## 5. Phases

Each phase that changes behavior bumps the version and lands as its own commit, per the house rule.

### Phase A [Thinning-Map]

Add the map to `moments.py` as a static method on `MomentAggregator`, alongside `agg_from_fs` and `factorial_to_noncentral` which it sits with naturally. NumPy docstring with the binomial derivation in `Notes` and a pointer to the source note.

Pure addition, nothing wired up, no version bump on its own if it lands with phase B.

Tests: Poisson closure (thinning a Poisson mean `n` by `w` gives Poisson mean `wn`) on all three moments, binomial closure (thinning `Bin(r, p)` gives `Bin(r, wp)`), the mixed Poisson closure for gamma, Delaporte and Sichel mixing, `w = 1` is the identity, and `w = 0` gives the degenerate zero count.

### Phase B [Component-Moments-From-Parent]

Rewire the mixture-product arm of `Aggregate.__init__`. Per exposure row, call `freq_moms(_en)` once to get the parent triple, then derive each mixture component by the map. `_record_component` takes the frequency moment triple instead of a scalar count, and calls `ma.add_fs` rather than `ma.add_f1s`.

`tot_freq_base` then accumulates one entry per exposure row instead of one per component, which is what `get_fsa_stats(remix=True)` already wants: for a single exposure row it re-enters `freq_moms` at exactly the count the parent was built with, so the re-entry is now exact rather than approximately exact.

The zero modification path simplifies. `freq_moms` is already wrapped to return realized zm moments, and the note's derivation assumes nothing about `N` beyond iid marks, so the map applies to the realized parent directly. The per-component `_zm_base_count` and `_zm_realized_count` calls in the loop go away, replaced by one application at the row level.

The limit-profile arm is left alone. Its weights are all 1, where the map is the identity.

`_aggregate.py:3798` (`wts = freq_ex1 / freq_ex1.sum()`) needs no edit: under the map `freq_ex1` for component `(e, m)` is `w_m * nu_e`, so the ratio is already the correct pooling weight. This is worth an explicit test rather than a comment, since it is the line that currently poisons the `dfreq` density.

Tests: the reassembly identity of section 2 across all eleven families; `dfreq[n]` equals `n claims ... fixed` on the full `stats_df`; the `MED.Picks` program reaches 13,989.98 theoretic against 13,986.79 FFT; `logarithmic` with a mixture stops raising; the mixed column and the `est_*` values are unchanged for Poisson and the mixed Poissons.

### Phase C [Mixture-Covariance-Row]

Optional, and I recommend it. Add the closed-form covariance total to `stats_df` so `mixed` and `independent` visibly reconcile instead of differing for reasons the reader has to reconstruct. One row, `('agg', 'cov')`, zero for Poisson by construction.

Update the `stats_df` docstring to say what the two columns now mean: `independent` is the sum of the true component marginals with their dependence dropped, and the gap to `mixed` is the covariance row.

### Phase D [Empirical-Profile-Guard]

An exposure profile with more than one row is not a thinning. Each row is a separate risk with its own frequency, and the library's model for the whole is one frequency at the summed count against the pooled severity. That is what the FFT computes, confirmed on `1 claim [5 50] xs 0 sev 100 * expon fixed`, where the FFT variance 850.382 matches `mixed` and not `independent` (256.295).

An empirical frequency cannot be that summed-count parent, so this pairing is genuinely ill posed, and it is the one place a hard error belongs. Today it fails silently and badly: `dfreq[1] [5 50] xs 0 sev 100 * expon` reports a `mixed` variance of `nan` and an FFT mean of 22.112 against a correct 44.224, because `freq_pgf` ignores the count the same way `freq_moms` does.

This is the decision in section 7.

### Phase E [Reinsurance-Layer-Thinning]

`_reinsurance.py:1118` calls `freq_moms(n * pr)` for the count of claims reaching a layer. This is a thinning, so the map is the right tool and the call becomes the map applied at `pr`. Independent of the mixture work and wrong today for any empirical frequency: the `EX.Re` fixture reports 1.5 claims into a `10 xs 10` layer where `1.5 * P(X > 10) = 1.0` is correct.

Moves the two `reins` exhibit snapshots, which need re-capturing, and is a moved number for the API.

### Phase F [Release-Hygiene]

`library.agg:247` tells the reader to build a mixed exponential by folding it into an `agg` with `dfreq[1]`. That advice becomes correct when phase B lands; until then it points at the defect. Add a note to `CommAutoMixedExponentialSev` recording the 13,990 mean so the entry carries its own check.

`CHANGELOG.md` section, `dev/TODO.md`, `dev/FEATURES.csv` if the public surface moved, and this plan to `dev/done/`.

## 6. Tests

The reassembly identity is the spine. One parametrized case per frequency family asserting that the mixed variance equals the sum of the component variances plus the closed-form covariance, to machine precision. It fails today for five families and passes for the rest.

Add the mixed exponential to `decl-testers.agg` under the severity section, since it is the case that surfaced this and it round trips.

Regenerate `tests/data/expected_specs.json` only if the grammar moves, which it should not.

## 7. Decisions I need

**7.1 Phase D, the empirical plus multi-row profile guard.** Hard error, or leave it computing a `nan` variance and half the correct mean? I recommend the error, with a message that names the reason (the profile model is one frequency at the summed count, which a stated pmf cannot be) and points at `fixed` or `poisson`. Note this is the only place I now think an error is warranted; the mixture case the discussion started from does not need one.

**7.2 Phase C, the covariance row.** Worth a `stats_df` row, or does the reconciliation live in the docstring only? A new row is a visible change for any consumer that iterates the index, the API included.

**7.3 Scope of phase E.** The reinsurance thinning is a real wrong number independent of everything else here. Land it in this plan, or split it into its own so the mixture work does not carry a moved exhibit snapshot?

## 8. What this plan does not touch

The wait mixtures inside `_build_renewal_frequency` route their weights through `Severity` rather than through counts, so I believe they are unaffected. I have not verified that and will before phase B lands.

`Portfolio` aggregates units through the same `MomentAggregator` but with `add_fs`, supplying its own frequency moments, so it is already on the phase B interface.

## 9. Execution notes (1.0.0a316, 2026-08-24)

Executed in full. The author ruled the three section 7 decisions on 2026-08-24: 7.1 hard error, 7.2 no covariance row because a covariance is a matrix and the docs should say it is computable, 7.3 fix the excess claim count in `reins_stats_df`.

Verified before and after. The reassembly identity of section 2 holds to machine precision for all twelve frequency families tested (poisson, fixed, binomial, bernoulli, geometric, logarithmic, neymana, pascal, gamma, delaporte and sichel mixing, and two `dfreq` forms). Before the change it held only for the Poisson and mixed Poisson rows. `dfreq[n]` and `n claims ... fixed` now agree on every `stats_df` cell that describes the distribution.

Divergences from the plan as drafted:

1. **Phase C shrank to documentation**, per ruling 7.2. The reconciliation identity, the closed-form covariance and the sign rule live in the `_init_stats_df` docstring, which says explicitly that the matrix is not stored.
2. **`MomentAggregator.add_f1s` was deleted**, which the plan did not anticipate. It became unused once both arms supply frequency moments explicitly, and its contract ("accumulate a count, derive the moments from the stored frequency") is precisely the invariant that proved false. Leaving it would have left the defect available to the next caller.
3. **`_record_component` gained a `base` parameter** alongside the moment triple. `add_f1s` used to accumulate `tot_freq_base` as a side effect of taking a count; with moments passed in, the base contribution has to travel separately so `get_fsa_stats(remix=True)` can still re-enter `freq_moms` at the count the frequency was built with.
4. **The zero modification moved from per component to per exposure row**, which the plan implied without stating the consequence: a zm or zt frequency combined with a severity mixture reports different component moments and a different expected loss than before. The old code applied `_zm_base_count` to each weighted count, so the shift was applied once per component; the shift belongs to `N`, and thinning applies to the realized parent.
5. **`Frequency.carries_own_count`** is the detection mechanism for phase D. The plan did not name one. It is a class flag next to `supports_zm`, true only on `FrequencyEmpirical` and so inherited by `FrequencyRenewal`.
6. **Phase F needed no `library.agg` edit.** `CommAutoMixedExponentialSev` already records the 13,990 mean, and the `MixedExponentialSev` note that tells the reader to fold the severity into an `agg` with `dfreq[1]` is correct as written once this lands. The author was editing that file concurrently, so it was left alone.
7. **All phases landed in one commit** rather than one per phase. The guard and the reinsurance count are consequences of the same insight, and splitting them would have required hunk-level staging of `_aggregate.py`.
8. **`dev/TODO.md` was left unstaged.** It carried uncommitted author edits at the time.

Suite: the baseline at `e973377` was green. After the change the only failures are in `test_library_entries.py`, `test_agg_libraries.py`, `test_recipe.py` and `test_feasibility.py`, all traced to concurrent author edits to `library.agg` (two named entries no longer exist, and one tolerance failure reproduces byte for byte with the source files reverted to HEAD). The two `reins` exhibit snapshots were re-captured; the diff is fourteen lines, all of them the `layer.1` frequency moments moving to the exact `Binomial(N, 2/3)` values.
