# [ZT-ZM-Frequency-Fix] — reparameterize zero-truncated / zero-modified frequency

## Context

`aggregate` currently treats the exposure-resolved claim count as the **realized**
`E[N]` and back-solves an unmodified base mean so that the post-modification mean
equals it (`Frequency._solve_n_base`, `src/aggregate/_frequency.py:218`). Three
problems follow.

1. **It is broken.** `_solve_n_base` brackets the root at `a=0`, where
   `(1 - p0) x / (1 - prn_eq_0(x))` evaluates `0/0`. Verified against the current
   tree:

   | program | result |
   |---|---|
   | `4 claims dsev [1] poisson zt` | `ValueError: function value at x=0.0 is NaN` |
   | `4 claims dsev [1] poisson zm .01` | same NaN (zero-*deflation* half) |
   | `4 claims dsev [1] poisson zm .5` | works, base λ = 7.997 |
   | `4 claims dsev [1] poisson zm .95` | works, base λ = **80** |

   So *every* `zt` fails, and so does every `zm` that reduces the zero mass. The
   singularity is removable (the limit is `1 - p0`), but the last row shows the
   design smell: modelling a mean-4 aggregate with 95% zeros forces the FFT to
   carry a base convolution out to ~80 claims.

2. **The feasible region is not rectangular.** `E[N^M] ∈ (1 - p0M, ∞)`, so
   `agg X 0.5 claims … poisson zt` has no solution at all — a ZT-Poisson mean is
   always > 1. Under the textbook parameterization every `p0M ∈ [0, 1)` is
   admissible.

3. **It contradicts every source.** Loss Models (`Klugman2012` §6.6), Loss Data
   Analytics (`Frees2018a` ch. 2) and R's `actuar` (`dzmpois(x, lambda, p0)`) all
   parameterize as *(base parameters, p0M) in, mean out*. Loss Models §7.4 is
   explicit that no (a,b,1) member is reproductive under exposure, and §9.2 that
   the quantity held fixed across periods is **p0M, not the mean**. ZT/ZM are
   per-risk constructs; summing them across policies gives a compound
   distribution, not a ZM one.

The practical consequence is that the three textbook exercises the docs promise
cannot be expressed. `docs/2_user_guides/problems/0x0_loss_data_analytics.rst:521`
states LDA 5.5.4 as "zero-modified Poisson with λ=3 and p0M = 0.5" and then
`.. todo:: Implement ZT and ZM!`; `0x0_loss_models.rst` has two more;
`docs/2_user_guides/DecL/050_frequency.rst:99` is an empty section reading "Not
yet implemented"; two `examples.agg` lines are commented out.

**Outcome:** the exposure clause sets the *base* mean and `zt` / `zm p0` shift it
forward in closed form (no solver, always feasible), with an opt-in `!` marker
restoring the mean-pinning behaviour for anyone who wants it.

## Decisions taken

- **Default = base-mean-forward.** The exposure-resolved count is the un-modified
  base mean. `zt` / `zm p0` apply the (a,b,1) reweighting forward; the realized
  `E[N]` falls out and is reported as `Aggregate.n`.
- **`!` = pin the mean.** `poisson zm 0.5 !` solves for the base mean that makes
  the realized `E[N]` (or `E[S]`, for the monetary forms) equal the exposure
  clause — i.e. today's behaviour, now opt-in. `!` is already the DecL
  *unconditional* marker (`decl.lark:291,411,434`), and the realized count mean is
  the unconditional one, so the sense is consistent rather than overloaded.
- **Warning fires on the monetary exposure forms only.** `loss` /
  `premium at lr` / `exposure at rate` state a money target that a bare `zm`/`zt`
  will miss, so warn and name the `!` fix. The `n claims` form stays silent —
  count in, shifted count out is the documented default and the common case.
- **`!` with no solution is a hard error**, quoting the feasibility bound
  (`E[N] > 1 - p0M`). No silent fallback.
- **Breaking change**, called out in `CHANGELOG.md`: existing `poisson zm 0.5`
  programs change answer (mean 4 → ≈ 2.07). Acceptable at `1.0.0a*`, and `zt` was
  wholly broken anyway.
- **Not in scope:** the extended (a,b,1) members (ETNB with −1 < r < 0, Sibuya).
  No documented demand.

## Names (vetted — `rg` over `src/aggregate`, all currently free)

| name | kind | role |
|---|---|---|
Strict verb-for-action methods against noun attributes, per CLAUDE.md.

| name | kind | role |
|---|---|---|
| `Frequency.base_mean` | attribute (noun) | the un-modified mean in force. **Renames the existing `unmodified_mean`** (only 2 occurrences, both in `_frequency.py`) — one canonical name for the concept. |
| `Frequency.modify_mean(base_mean=None)` | method (verb) | forward map: realized `E[N]` for a base mean (defaults to the one in force). |
| `Frequency.solve_base_mean(target_mean)` | method (verb) | inverse map: base mean hitting a target realized mean. Raises with the feasibility bound when none exists. |
| `Frequency.apply_deductible(survival)` | method (verb) | Loss Models §8.6: `N^L → N^P` under a deductible with survival `v`. Returns the shifted `(base_mean, p0M)`. |
| `freq_pin_mean` | spec key / `Aggregate` kwarg | `True` when `!` was given. |
| `ZeroModifiedExposureWarning` | warning class | monetary-form target miss. |

## Changes

**Math (`src/aggregate/_frequency.py`).** Replace `_solve_n_base` /
`_install_zm_wrappers` with the closed-form (a,b,1) reweighting. For base mean
`m`, natural zero mass `p0 = prn_eq_0(m)` and modified mass `p0M`:

```
c        = (1 - p0M) / (1 - p0)
G^M(z)   = p0M + c (G(z) - p0)
E[N^M]   = c · E[N],  and likewise for the higher raw moments
```

`freq_pgf(n, z)` and `freq_moms(n)` keep their signatures but now interpret `n`
as the **base** mean throughout — pure functions, no solver, no captured state.
`solve_base_mean` is the only place `brentq` survives, reached only under `!`.

**Wiring (`src/aggregate/_aggregate.py`).** `self.n = ma.tot_freq_1`
(`:2086`) already comes out of the ZM-wrapped `freq_moms`, so it becomes the
realized mean for free. The two sites that feed a mean *into* `freq_pgf` must
switch to the base mean: the FFT path (`:3192`) and `freq_pmf` (`:3412`). The
`n != en` logger warning at `:3418` will otherwise fire on every `zm` build —
suppress it for the modified case. Under `!`, resolve the base via
`solve_base_mean` before the moment aggregation.

**Grammar / round-trip.** `decl.lark:266-267` gains the optional marker
(`freq ZM expr "!"` → `freq_zm_pin`, `freq ZT "!"` → `freq_zt_pin`); watch for
Earley ambiguity against the `sev "!"` rules and disambiguate on clause position
if it appears. Transformer at `parser.py:1001-1010` sets `freq_pin_mean`;
`decl_writer.py:342-344` must emit the `!` back so programs round-trip.
Regenerate `docs/4_agg_language_reference/ref_include.rst` via
`grammar(add_to_doc=True)`.

**Reporting.** `Frequency.info` / `tail_explanation`
(`_frequency.py:311,370`) gain the base mean alongside the realized `E[N]`, so
the shift is visible on inspection.

## Verification

1. **Unit** — new `tests/test_frequency_zm.py`: closed-form pgf/moments against
   hand-computed (a,b,1) values for poisson / negbin / binomial / geometric /
   logarithmic; `modified_mean` ∘ `solve_base_mean` round-trips; `zt` works at
   every mean including `< 1`; `zm p0` works on **both** sides of the natural
   `p0` (the currently-broken deflation half); infeasible `!` raises.
2. **Regression** — `tests/test_create_frequency.py:31` (`poisson zm 0.3`) and
   `tests/test_pgf_polynomial.py:94` change expected values; update with the
   hand-computed numbers, not with whatever the code emits.
3. **Test suite / DecL** — `_test_suite.agg:76-82` already carries six `zm`/`zt`
   lines; add `!` variants and re-check `tests/data/expected_specs.json` for the
   new spec key. Per house rule, mirror any new pytest DecL programs into
   `src/aggregate/agg/decl-testers.agg`.
4. **Worked examples (the real acceptance test)** — LDA 5.5.4 must now be
   expressible and correct: `N^L` ~ ZM-Poisson(λ=3, p0M=0.5), Burr(α=3, θ=50,
   γ=1), deductible 30; compute `v = S(30)`, apply `deductible_shift`, and check
   `E(N^P)` / `Var(N^P)` against the published answer. Same for Loss Models 9.11
   (ZM binomial, m=3, q=0.3, p0M=0.4) via Panjer recursion. Replace the three
   `.. todo:: Implement ZT and ZM!` blocks with the executed solutions.
5. **Gate** — `uv run pytest -m 'slow or not slow'` (with `UV_LINK_MODE=copy`),
   full and parallel, before the commit. Docs are **not** rebuilt in the loop;
   note them as pending.

## Release hygiene (one commit)

Bump `pyproject.toml` to `1.0.0a152`; `CHANGELOG.md` section describing the new
semantics and flagging the breaking change; tick `[ZT-ZM-Frequency-Fix]` in
`dev/TODO.md` (and the ZT/ZM item under `[Doc-Gaps]`); regenerate
`dev/FEATURES.csv` via `dev/regen_features.py`; fill
`docs/2_user_guides/DecL/050_frequency.rst`; un-comment the two `examples.agg`
lines (`:191-194`). Single commit, one-line subject:
`[ZT-ZM-Frequency-Fix] a152: …`.

## Bibliography

All three keys verified present in `C:/s/TELOS/Biblio/uber-library.bib`:
`Klugman2012` (Loss Models 4th ed.), `Frees2018a` (Loss Data Analytics), and
`Boucher2007` (Risk Classification for Claim Counts, NAAJ 11(4)) — added by the
author during this session, confirmed at `:67885`. Cite `Boucher2007` in the
`050_frequency.rst` section for the practical-usage note: ZT/ZM are fitted at the
individual-risk level in ratemaking regressions, not used as portfolio-level
aggregate frequencies — which is the substantive reason the base-mean
parameterization is the right default.
