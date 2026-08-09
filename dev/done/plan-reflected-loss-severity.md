# [Reflected-Loss-Severity]

Allow a reflected severity under the plain `sev` keyword, clamping the negative part at zero the way every other negative-support severity already does, with a warning that names the clamped mass and points at `ssev`. Fix the splice-versus-moment defect that the investigation turned up on the way.

Status: EXECUTED at 1.0.0a230. Judgment calls answered by the author in §7; see §8 for what execution changed relative to the plan.

## 1. Motivation

`sev 10 - lognorm 1.5 splice [0 10]` is rejected today. The refusal is not a grammar restriction: `sev` and `ssev` share the same `sev` nonterminal (`src/aggregate/decl.lark:456-457`), so the arithmetic parses either way. It is a transformer guard:

```
src/aggregate/parser.py:1665
    if sev.get("sev_reflect", False):
        raise ValueError("DecL: a reflected (signed) severity needs 'ssev', not 'sev' ...")
```

The guard fires on *any* reflection, without looking at whether the reflected support actually reaches below zero. That is inconsistent with how the library already treats other negative-support severities under plain `sev`, all of which build, clamp at zero, and return correct clamped moments:

| declaration | `Severity.signed` | `fz.support()` | `moms()[0]` | exact |
|---|---|---|---|---|
| `sev 10 * norm + 5` | `False` | `(-inf, inf)` | 6.977966 | `E[(5+10Z)+]` = 6.977966 |
| `sev lognorm 5 cv 1 - 10` | `False` | `(-10, inf)` | 0.634385 | est_m 0.634381 |
| `sev 10 - lognorm 1.5` | rejected | | | |

The clamp is deliberate and tested (`tests/test_negative_x.py::test_ssev_not_clamped_vs_sev`, which asserts that `sev 10 * norm` clamps while `ssev 10 * norm` does not). The reflected form is the one shape singled out for refusal, and there is no reason for it: `10 - X` on a spliced `X` is often a perfectly ordinary bounded loss.

Routing through `ssev` is not a satisfying workaround. It works numerically (see §2), but `Severity.signed` is set unconditionally by the reflection (`_severity.py:1447`) even when the support never goes below zero, which puts an ordinary loss on the signed code path: identity layering, so an occurrence layer is silently ignored, plus the two-sided grid, two-sided quantiles, and the signed `Portfolio` combine.

## 2. Verified current behavior

Everything below was measured against the working tree at `1.0.0a229`, not inferred.

**`ssev` does not swap the loss convention.** The loss/payoff convention is the orientation suffix (`decl.lark:96-98`), which sets `value_type`. `ssev` sets `Severity.signed`, documented as orthogonal at `parser.py:1680-1681`. Confirmed: `build('agg T 5 claims ssev 10 - lognorm 1.5 splice [0 10] poisson').value_type` is `'loss'`.

**Splice runs before reflection, and composes correctly.** `_severity.py:1089-1096`, note at `1370-1373`. For `ssev 10 - lognorm 1.5 splice [0 10]` the window applies to the underlying lognormal, so `Y = 10 - (X | 0 <= X <= 10)` has `fz.support()` exactly `(0.0, 10.0)`, `_severity_negative_buckets` is 0, and `xs[0]` is 0, so the grid wastes nothing on a negative half. The FFT answer is right: aggregate severity mean 8.31148 against the exact `10 - E[X | X <= 10] = 10 - 1.68852 = 8.31148`.

**The analytic moments of that same severity are wrong.** `sevs[0].sev1` is 6.919783, which is `10 - 3.080217`, the *unspliced* lognormal mean. Cause: `_apply_lb_ub` (`_severity.py:1333-1373`) patches `cdf`, `sf`, `pdf`, `ppf`, `isf`, and `support` on the frozen RV, but not `moment`, and `_apply_reflect` reads `Z.moment(k)` at `_severity.py:1434`. `_apply_signed` has the same defect at `_severity.py:1470-1474`, so a spliced `ssev` gets unspliced moments with or without a reflection. Consequences: `explain_validation` reports `fails sev mean, agg mean` on a correct build, and the automatic bucket window is sized from wrong moments.

**Layering a signed severity is silently wrong.** `_apply_signed` makes attachment and limit identity (docstring at `_severity.py:1463-1465`), but the aggregate still sizes its grid to the limit. `1 claims 8 xs 1 ssev 10 - lognorm 1.5 splice [0 10]` returns 5.677 against a true layered mean of 7.002, with no warning. Out of scope here; recorded in §9.

## 3. Design

One rule. Under plain `sev` a reflected body is built like any other severity whose support reaches below zero: the existing layered-loss transform clamps `x < 0` to `0` (`_apply_layer_attachment`, always runs, docstring at `_severity.py:1395-1398`), the severity stays `signed = False`, and layers, attachments, and occurrence reinsurance all work normally. Under `ssev` nothing changes.

When the reflected support does reach below zero, warn once, name the mass being clamped, and suggest `ssev`. Then clamp and continue. The extreme case follows from the rule rather than being special-cased: `sev -lognorm 1.5` has support `(-inf, 0]`, so the whole law clamps and the severity is a point mass at zero, with a warning. That is the author's stated intent.

The warning is reflection-specific, not support-specific. See judgment call J1 in §7.

## 4. Phases

### 4a. [Reflect-Under-Sev] accept the declaration

**`src/aggregate/parser.py`, `sev_clause_sev` (1663-1675).** Delete the raise. `sev_signed` stays unset, so the spec carries `sev_reflect=True, sev_signed=False`, a combination that is new but well formed.

**`src/aggregate/_severity.py`, `Severity.__init__` (1088-1115).** Collapse the three-way dispatch into splice, then optional reflect, then signed-or-layered:

```python
self._apply_lb_ub()
if not self.signed:
    self._validate_moments()      # targets describe the pre-reflection base
if self.sev_reflect:
    self._apply_reflect()
if self.signed:
    self._apply_signed()
else:
    self._compute_attachment_probs()
    self._apply_layer_attachment()
```

Two things move. `_apply_reflect` drops its `self.signed = True` line (`_severity.py:1447`); `ssev` keeps setting `signed` through the constructor kwarg and a `dsev` with a negative atom keeps setting it in `_build`, so the only case that changes is the new one. `_validate_moments` moves ahead of the reflection because it reads `self.fz.stats('mv')` (`_severity.py:1526`) and the user's `mean cv` target describes the base `X`, not `10 - X`. It has no dependency on `_compute_attachment_probs`, so the move is safe for the existing path. This ordering also keeps §4c honest: patching `fz.stats` with the rest of the reflected methods would otherwise make `_validate_moments` compare a base target against reflected stats and emit spurious warnings.

**`src/aggregate/_severity.py`, `moms()` path 2 (1657-1662).** Add `and not self.sev_reflect` to the analytic-shortcut condition. This is required, not defensive: `sev -lognorm 1.5` has `sev_loc` 0, `sev_lb` 0, `sev_ub` inf and `sev_name` `'lognorm'`, so it satisfies the shortcut exactly and would return the *unreflected* closed-form moments. Today the case cannot arise because reflection implies signed, and path 0 returns first.

The numerical path needs no change and is already correct for the clamped reflected law. Traced for the fully-negative case: `_compute_attachment_probs` sets `moment_pattach = fz.sf(0) = 0`, so `_numerical_moms` gets `upper == lower == 0` and takes the zero-width short circuit at `_severity.py:826`; `detachment` is infinite so the `dma * lower` term at `_severity.py:898` (which would be `inf * 0`) is skipped; `conditional` divides by `pattach = 1`. Result `(0, 0, 0)`, which is right. A severity that is identically zero does build an aggregate today (checked with `sev dhistogram xps [0] [1]`: `bs` 1.0, `log2` 1, `est_m` 0.0), so there is no structural blocker, but the continuous route reaches the bucket window differently and §4e tests it explicitly.

### 4b. [Reflected-Clamp-Warning] say so

New `ReflectedSeverityClampWarning(UserWarning)` in `src/aggregate/constants.py`, alongside the existing six, with the house docstring shape and the closing "Subclasses `UserWarning` so Python's default warning filter shows it (not the logger, which is silent by default)." Add to `__all__`.

Raised from `Severity.__init__` immediately after `_apply_reflect`, when `not self.signed` and the reflected support lower edge is below zero (tolerance `VALIDATION_NOISE` from `src/aggregate/moments.py:15`, which reads `get_settings().validation.noise`, not a loose epsilon). Emitted through `warn_once` (`constants.py:409`), keyed on `(sev_name, shift, sev_lb, sev_ub)` so distinct declarations each warn once rather than the first one masking the rest.

Message shape, one line, no internal blanks, "did you mean" concatenated onto the message per house style:

```
Severity 'lognorm': reflected support [-inf, 10] reaches below 0, so 'sev' clamps 6.24% of the mass to 0. Did you mean 'ssev'?
```

The clamped mass is `fz.cdf(0)` on the reflected law, which is one call and worth having in the message because it is the number that tells the user whether they care.

### 4c. [Splice-Moment-Defect] fix the moments

The defect is §2's third finding: `moment` is not among the methods `_apply_lb_ub` swaps, so both `_apply_reflect` (`_severity.py:1434`) and `_apply_signed` (`_severity.py:1470-1474`) read unspliced raw moments off the frozen RV.

Fix in the two readers rather than by patching `moment` on the frozen RV. Add one private helper that computes the first three raw non-central moments by integrating the already-patched `isf` over `(0, 1)`, reusing `_safe_integrate` exactly as `_numerical_moms` does, and use it only when a splice is active:

- unspliced (`sev_lb == 0 and sev_ub == np.inf`): keep `Z.moment(k)`, which is exact and closed form for the scipy families, and keep the binomial expansion in `_apply_reflect`. No behavior change.
- spliced: integrate. Correct where the current code is wrong, at the cost of quadrature accuracy on a case that has no closed form anyway.

This is judgment call J6 in §7; the alternative is to always integrate, which is simpler code but gives up exactness on the common unspliced path.

While here, patch `fz.stats` alongside the other reflected methods in `_apply_reflect` so an internal caller does not silently read base-law statistics. Two callers exist today: `_numerical_moms:815-818` uses it only for a finiteness check, and `_validate_moments:1526`, which §4a moves ahead of the reflection precisely so this patch is safe. Note that `fz.stats` is internal; the user-facing `Severity.stats()` routes through `_stats` to `moms()` and is unaffected.

### 4d. [Reflect-Unparser-Round-Trip] keep the keyword

`src/aggregate/decl_writer.py:304` currently emits `ssev` whenever `sev_signed` **or** `sev_reflect` is set. Left alone, `sev 10 - lognorm 1.5` would round-trip to `ssev 10 - lognorm 1.5` and silently change semantics. Drop the `sev_reflect` disjunct so the keyword follows `sev_signed` alone, and update the comment at `decl_writer.py:301-303`, which explains the old rule. Covered by `test_decl_unparser` (444 corpus cases).

### 4e. [Reflect-Corpus-And-Tests]

New lines in `src/aggregate/agg/decl-testers.agg`, under the signed-severity section next to the existing `ssev` examples, each with a `note{}` and each required to round-trip:

- `sev 10 - lognorm 1.5 splice [0 10]`, bounded reflection, no clamp, no warning
- `sev 100 - lognorm 80 cv .2`, small clamped tail, warns; mirrors the existing `ssev 100 - lognorm 80 cv .2` at `library.agg:739`
- `sev -lognorm 10 cv 0.5`, fully clamped, degenerate point mass at zero, warns
- `sev -lognorm 2 cv .5 + 5`, tight-binding unary minus, partly clamped; mirrors `UM.Shift` at `decl-testers.agg:625`
- `ssev lognorm 1.5 splice [0 10]`, the unreflected spliced signed case that §4c fixes

`decl-testers.agg` feeds `test_decl_unparser` and `test_grammar_sync` but not the frozen `expected_specs.json` snapshot, which is driven by `_test_suite.agg` alone (`tests/test_decl_parser.py:24`), so no snapshot churn.

New cases in `tests/test_negative_x.py`, where `test_ssev_not_clamped_vs_sev` already lives:

- `test_sev_reflected_bounded_no_clamp`: `signed is False`, `fz.support()` is `(0, 10)`, `moms()[0]` is 8.31148 to 1e-6, and no warning is raised
- `test_sev_reflected_clamps_and_warns`: `pytest.warns(ReflectedSeverityClampWarning)`, `moms()[0]` equals `E[Y+]` from an independent quadrature
- `test_sev_reflected_all_negative_degenerate`: `moms()` is `(0, 0, 0)`, the aggregate builds, `est_m` is 0
- `test_sev_reflected_layer`: an occurrence layer on the non-signed reflected severity matches an independent quadrature, in contrast to the signed case
- `test_sev_reflected_round_trip`: the spec unparses to `sev`, not `ssev`
- `test_ssev_splice_moments`: `ssev 10 - lognorm 1.5 splice [0 10]` has `sev1` 8.31148, and `explain_validation` no longer reports a mean failure
- `test_ssev_splice_moments_no_reflect`: `ssev lognorm 1.5 splice [0 10]` has `sev1` 1.68852

Regression watch: `test_ssev_not_clamped_vs_sev` must stay green unchanged, since `sev 10 * norm` behavior is untouched.

### 4f. [Reflect-Docs-And-Release]

The grammar is unchanged, so no `ref_include.rst` regeneration.

Prose that states the "reflection requires `ssev`" rule needs updating. `docs/2_user_guides` and `docs/5_technical_guides` are generated from the monograph, so edit the `.qmd` at `C:/s/AI/aggregate-monograph`, never the RST. Grep both trees for the rule before deciding which pages move.

Release hygiene per the standing rules: version bump, `CHANGELOG.md` section, `dev/TODO.md`, `dev/FEATURES.csv` via `dev/regen_features.py` (a new exported warning class changes the public surface), and move this plan to `dev/done/`. One commit, subject `[Reflected-Loss-Severity] a230: sev accepts a reflected body, clamps at zero with a warning; spliced-signed moments fixed`.

Gates: tier 2 `uv run pytest` during the work, then tier 3 `uv run pytest -m 'slow or not slow'` at the bump, plus the numerics gate `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`, which is warranted here because §4c touches moment integration.

## 5. What does not change

`ssev` semantics, the orientation suffix and `value_type`, the silent clamp on `sev 10 * norm` and friends, the `wait` clause rejection of reflected bodies (`parser.py:1868-1871`, see J4), and layering on a signed severity, which stays out of scope and silently wrong (§9).

## 6. Risks

The dispatch restructure in §4a touches every `Severity` construction, so a mistake there is broad rather than local. The mitigation is that the restructure is behavior-preserving by construction for the two existing branches: the signed path runs `_apply_lb_ub`, `_apply_reflect`, `_apply_signed` in the same order, and the plain path runs the same four steps with `_validate_moments` moved earlier, which it does not depend on.

Moving to numerical moments for spliced signed severities (§4c) changes numbers that some snapshot or tolerance may pin. They are currently wrong numbers, so any test that breaks is a test that was pinning the defect, but the breakage should be read case by case rather than re-baselined wholesale.

## 7. Judgment calls for the author

**J1. Warning scope: reflection-only, or any negative support under `sev`?** Recommend reflection-only. A support-based rule would be more principled, and its blast radius is small: one corpus line (`_test_suite.agg:90`, `sev 100 * norm +500`) and one test line (`tests/test_negative_x.py:139`, `sev 10 * norm`) would newly warn, while the two spliced `norm` lines at `library.agg:1072` and `library.agg:1076` are windowed to `[0, 40]` and `[0, 35]` and would stay quiet. The argument against is not blast radius but intent: `tests/test_negative_x.py::test_ssev_not_clamped_vs_sev` documents the silent clamp on `sev 10 * norm` as deliberate, so widening the rule reverses a decision rather than filling a gap. Better taken later on its own merits. ==> yes, only warn if reflection AND if implied support goes negative.

**J2. Warning name.** `ReflectedSeverityClampWarning`. Checked against the existing six in `constants.py` and against the public surface for collisions. ==> OK

**J3. Warning frequency.** `warn_once` keyed on `(sev_name, shift, sev_lb, sev_ub)`, so a portfolio of ten similar units warns once but two genuinely different declarations both warn. The alternative is a plain `warnings.warn` every time, which is noisier but harder to miss in a loop. ==> agree

**J4. Should `wait` clauses now accept a non-signed reflected body?** Recommend no, keep the rejection at `parser.py:1868-1871`. A clamped reflection puts an atom at zero waiting time, which the renewal machinery would read as simultaneous claims. ==> agree, to complicated for waits.

**J5. Degenerate all-clamped case: warning or error?** Decided by the author already: warning, build the point mass at zero, move on. Recorded here so it is not relitigated.

**J6. Spliced moments: exact-when-unspliced plus numerical-when-spliced, or always numerical?** Recommend the former, per §4c. It preserves today's closed-form exactness on the common path and confines quadrature to the case that never had a closed form.

## 8. Execution notes (what the plan got wrong)

Five things surfaced during execution that the plan did not anticipate. All are in the shipped code.

**The plan forgot to stop `_apply_reflect` populating the moments.** §4a said the numerical path was already correct for the clamped reflected law, which is true, but `_apply_reflect` set `sev1`/`sev2`/`sev3` unconditionally and `moms()` path 1 returns those whenever they are set and the layer is trivial. So the clamped case never reached the numerical path: `sev -lognorm 10 cv 0.5` reported a mean of -10 for a severity that is identically 0. Fixed by setting the raw moments only on the signed path, where ``Y`` itself is the answer.

**`_validate_moments` mis-handled the reflection shift.** For a reflected severity `sev_loc` is the shift in `shift - X`, not an additive location, so adding it to the target compared 180 against an achieved 80 and printed `WARNING target mean ... not close` on every correct `sev 100 - lognorm 80 cv .2`. Moving the call ahead of the reflection (which the plan did specify) was necessary but not sufficient; the loc term also has to be dropped. Covered by `test_sev_reflected_no_spurious_moment_validation`.

**The `approximate sgamma` / `slognorm` left-skew fitter was relying on the old coupling.** `_approximate_sev_kwargs` returns `sev_reflect=True` for a left-skewed aggregate and never set `sev_signed`, because reflection used to force it. Decoupling them left the fitted severity clamping at zero, which is exactly what an approximation matching three moments must not do; `tests/test_approximate.py::test_left_skew_reflect_matches_exact` caught it. The fix applies the module's own low-quantile test to the reflected law, which is what its docstring already promised.

**`fz.stats` is deliberately left unpatched**, against §4c's "while here" note. The justification there does not survive contact: the one internal caller asks only about finiteness, and finiteness is invariant under reflection, so the patch would change no behavior. Worse, `_apply_lb_ub` does not swap `stats` either, so a reflected-but-unspliced `stats` would look fixed without being fixed. A comment in `_apply_reflect` records the reasoning.

**J6 was never marked.** Executed on the plan's recommendation (exact `fz.moment` when unspliced, quadrature when spliced), which leaves every unspliced result bit-identical. `test_ssev_unspliced_moments_stay_closed_form` pins that.

Docs needed no edits. Nothing in `docs/` or the monograph asserted the rejection; the closest statement, features.rst on `shift - dist` needing `ssev` "to keep the signed support", is still true. The grammar is unchanged, so no `ref_include.rst` regeneration, and `dev/FEATURES.csv` is unchanged because the new warning class lives in `constants`, not on a first-class class surface.

Unrelated pre-existing failure seen while running the features gate: `dev/check_features_rst.py` block 80 dies on a `UnicodeEncodeError` printing `κ` to a cp1252 console. It reproduces on a clean checkout and has nothing to do with this work.

## 9. Out of scope, worth a separate item

Layering a signed severity is silently wrong (§2, fourth finding). `_apply_signed` makes attachment and limit identity while the aggregate still sizes its grid to the limit, so `8 xs 1 ssev ...` returns a number that is neither the layered loss nor the unlayered one, with no warning. The minimum fix is a warning when a layer clause meets a signed severity; the real fix is layering on signed support, which `dev/plan-negative-x-agg.md` §6 already defers. Note that this plan reduces the blast radius: the bounded reflected case that used to require `ssev` can now be declared with `sev`, where layering works.
