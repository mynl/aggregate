# Plan: the Tsallis distortion, distortions from a Gini index, and an extended calibration set

`[Gini-Rename]`, `[Tsallis-Distortion]`, `[Distortion-From-Gini]`, `[Extended-Calibration-Set]`. Drafted 2026-11-06 against aggregate 2.0.0 (commit `1fe40eb8`). **Status: ruled 2026-11-06, executing (see "Execution log").** Four phases in the order below, one pre-release bump each (`2.1.0aN`, house rule 2026-11-06). No grammar change.

## Goal

1. **`gini_p` becomes `gini`**, as the `Distortion` property and as the column in every frame and exhibit, with no alias left behind (author, 2026-11-06, accepting the break of the API stability promise because nothing outside the author's own work reads it yet).
2. A new standard distortion family, `tsallis`, the Beta(α, 1) mixture of TVaRs from the note `D:/Projects/Scratch/08-tsallis.qmd`, built and called like `ph` or `dual`: `Distortion('tsallis', alpha=3)` in Python, `distortion D tsallis 3` in DecL, with closed forms, a Kusuoka density and a `calibrate` method.
3. The Gini index $2\int_0^1 g - 1$ on every distortion as `gini`, in closed form wherever the family has one.
4. `Distortion.from_gini(kind, gini)` for any one-parameter family, with an informative `ValueError` for a family with more than one parameter.
5. `calibrate_distortions(..., extended=False)`; `extended=True` calibrates the standard set with `tsallis` between `wang` and `dual`.

## The Tsallis family

For $\alpha > 0$ the Kusuoka measure is Beta$(\alpha, 1)$ on the tail probability $s$, density $\alpha s^{\alpha-1}$ (in the monograph's convention $p = 1 - s$, Beta$(1, \alpha)$ on $p$, density $\alpha(1-p)^{\alpha-1}$). Integrating $\min(u/s, 1)$ against it gives

$$
g_\alpha(u) = \frac{\alpha u - u^\alpha}{\alpha - 1},\qquad g_1(u) = u(1 - \log u),
$$

the second the limit at $\alpha = 1$. Facts the implementation uses, all proved in the note:

- $g_\alpha'(u) = \alpha(1 - u^{\alpha-1})/(\alpha - 1)$, $g_1'(u) = -\log u$. The slope at 0 is $\alpha/(\alpha-1)$ for $\alpha > 1$ and infinite for $\alpha \le 1$; the slope at 1 is 0. $g_\alpha''(u) = -\alpha u^{\alpha-2}$.
- $\int_0^1 g_\alpha = (\alpha+2)/(2(\alpha+1))$, so the Gini index is $1/(\alpha+1)$ and the family with Gini $G$ is $\alpha = 1/G - 1$.
- No mass at zero and no Kusuoka atoms. $\alpha \to 0$ is the maximum, $\alpha \to \infty$ the mean. $\alpha = 2$ is `dual(2)`; $\alpha = 1/2$ is $2\sqrt u - u$.
- Submultiplicative for every $\alpha$; the interval condition holds exactly for $0 < \alpha \le 2$.
- $D_g$, the law on $[0, 1]$ with distribution function $g$ that `describe` reports, has raw moments $\mathsf P X^k = \int_0^1 k x^{k-1}(1 - g_\alpha(x))\,dx = \alpha/((k+1)(k+\alpha))$, valid at $\alpha = 1$ too (derived for this plan; the test below checks it against quadrature).

**Numerically stable forms.** With $\beta = \alpha - 1$ and $L = \log u$,

$$
g_\alpha(u) = u\left(1 - \frac{\operatorname{expm1}(\beta L)}{\beta}\right),\qquad g_\alpha'(u) = -\alpha\,\frac{\operatorname{expm1}(\beta L)}{\beta},
$$

with the $\beta = 0$ branch $u(1 - L)$ and $-L$. These have no cancellation as $\alpha \to 1$, give $u\alpha/(\alpha - 1)$ when $u^\alpha$ underflows for large $\alpha$, and give $g = 1$ for $u > 0$ as $\alpha \to 0$. At $u = 0$: $g = 0$; $g' = \alpha/\beta$ for $\beta > 0$ and $+\infty$ otherwise.

## Current behavior, 2.0.0

- **Registry.** `Distortion` subclasses register by `kind` in `Distortion._registry` (`spectral.py:612`), in declaration order. `available_distortions(pricing, strict)` (`spectral.py:941`) filters on the flags `documented`, `pricing_ok`, `strict_pricing` and `has_mass_default`. `available_distortions()` returns `ccoc, ph, wang, dual, tvar`, and `tests/test_distortion_construction.py:405` asserts it equals `_pricing.DEFAULT_CALIBRATION_DISTORTIONS` (`_pricing.py:25`). `decl_words` lists `available_distortions(pricing=False)` (`decl_words/__init__.py:225`). A distortion kind is an `ID` in `decl.lark:170`, not a reserved word.
- **The Gini index already exists** as `Distortion.gini_p` (`spectral.py:1231`): `_describe_closed_form()['p_equiv']` where the family supplies it, otherwise adaptive quadrature of `g` (`epsabs = epsrel = 1e-11`). Closed forms today: `ccoc` ($d$), `ph` ($(1-a)/(1+a)$), `wang` ($2\Phi(\lambda/\sqrt2) - 1$), `dual` ($(b-1)/(b+1)$), `tvar` ($p$), `bitvar`, `wtdtvar`, `beta` ($(b-a)/(a+b)$), `cll`, `clin`, `lep`. Quadrature: `ly`, `power`, `minimum`, `mixture`.
- **Where `gini_p` appears**, counted 2026-11-06 (135 occurrences in 29 live files): the property and its docstrings (`spectral.py`); the `calibration_df` columns (`_pricing.py:706`); the evaluation columns `_pricing.EVAL_COLS` (`_pricing.py:1091`) and the rows built at `_pricing.py:1332`; docstrings in `_aggregate.py`, `_portfolio.py`, `_pnl.py`, `results.py`; the pricing exhibits (`exhibits/_pricing.py`, `exhibits/_core.py`), including the `gini_p` block name and caption; the format keys in `formats/formats-insurer.yaml:54` and `formats/formats-raw.yaml:108`; comments in `library.agg:268` and `decl-testers.agg:742`, and a `note{}` at `decl-testers.agg:751`, which is in the spec snapshot; eleven test files and `tests/data/exhibit_snapshots.json` (27); six doc pages under `docs/`. The stats and describe tables of a distortion already have a row named `gini` beside `p_equiv` (`spectral.py:1518`, `1642`), which the rename leaves as it is.
- **Gini to parameter** exists only as `p_to_parameters(p)` (`spectral.py:4300`), a dict for the five standard families (`ccoc` as the discount $d$), and `consistent_distortions(p)` (`spectral.py:4316`), which builds those five.
- **Calibration.** `Distortion.calibrate_set(names=...)` (`spectral.py:2231`) loops over the subclass `calibrate` methods; `ph`, `wang`, `dual` use Newton with an analytic derivative (`_newton_iterate`) and `_finalize_calibration`. `calibrate_distortions` has four entry points, all forwarding to `_pricing.calibrate_distortions` (`_pricing.py:819`): `Aggregate` (`_aggregate.py:8479`), `Portfolio` (`_portfolio.py:3565`), both defaulting `names=_pricing.DEFAULT_CALIBRATION_DISTORTIONS`, and `AggregateCHF` (`characteristic_function.py:2520`), defaulting `names=None`.
- **The envelope chart** draws only the families in `charts/_emit_bounds.py:47` `CALIBRATED`, so a calibrated `tsallis` would be dropped silently.
- **The note** registers a session class with `documented = False`, `pricing_ok = False`, bisection `g_inv` and no `calibrate`; it is the prototype, not the implementation.

## The change

### [Gini-Rename] `gini_p` becomes `gini`

A mechanical rename in its own bump, first, so the later phases are written against the new name and the rename's diff is reviewed alone.

- `Distortion.gini_p` becomes `Distortion.gini`, same body and docstring. No alias and no deprecation shim: `gini_p` raises `AttributeError` afterwards.
- Every column, block, caption, format key and docstring listed under "Where `gini_p` appears" says `gini`: `calibration_df`, `EVAL_COLS` and the evaluation frames, the exhibit block and its caption, both `formats-*.yaml` keys, the docs pages, the `library.agg` and `decl-testers.agg` comments and `note{}`.
- Tests updated; `tests/data/exhibit_snapshots.json` and `tests/data/expected_specs.json` recaptured, and their diffs read: the only change in either is `gini_p` to `gini`.
- Historical text keeps `gini_p`: `CHANGELOG.md` entries before this one and the plans in `dev/done/`.
- The CHANGELOG entry says plainly that this is a breaking rename of a stable-class property and of frame columns, made deliberately inside a minor series.
- `rg -w gini_p src tests docs` returns nothing afterwards.

### [Tsallis-Distortion] the family

`class TsallisDistortion(Distortion)` in `spectral.py`, **declared between `WangDistortion` and `DualDistortion`**, so registry order puts it between `wang` and `dual` in `available_distortions(pricing=True, strict=False)`, in `available_distortions(pricing=False)` and in the DecL distortion words.

- Class attributes: `kind = 'tsallis'`, `med_name = 'Tsallis'`, `long_name = 'Tsallis (Beta(alpha, 1) mixture of TVaRs)'`, `documented = True`, `pricing_ok = True`, `strict_pricing = False` (ruling `[Tsallis-Strict-Flag]`), `param_name = 'alpha'`, `_calibration_init_shape = 3.0` (Gini 1/4; `calibrate_set` constructs from it).
- `alpha` property with setter, as `PHDistortion.a`. `_label_default` returns `f'Tsallis({self.shape:.3g})'`.
- `_build` raises `ValueError` unless `alpha` is finite and positive, and sets no mass.
- `g`, `g_prime` in the stable forms above. `g_inv` by vectorized bisection on $[0, 1]$ to 60 halvings ($g$ is continuous and strictly increasing), with the closed forms at $\alpha = 2$ ($1 - \sqrt{1-y}$) and $\alpha = 1/2$ ($(1 - \sqrt{1-y})^2$).
- `_describe_closed_form`: mean, var, std, cv, skew from $\mathsf P X^k = \alpha/((k+1)(k+\alpha))$, `p_equiv` $= 1/(\alpha+1)$, `loading` $= 1/(2(\alpha+1))$.
- `_kusuoka_atoms` returns `[]`; `_kusuoka_density(p)` returns $\alpha(1-p)^{\alpha-1}$ on $[0, 1)$, guarded as `PHDistortion._kusuoka_density` is.
- `calibrate`: `brentq` in $\theta = 1/(\alpha + 1)$ on $(10^{-9}, 1 - 10^{-9})$, solving $\int g_{1/\theta - 1}(S)\,dx = P$ with `_quad`, then `_finalize_calibration`. The premium is increasing in $\theta$, from the expected loss at $\theta \to 0$ to $\int \mathbf 1\{S > 0\}\,dx$ at $\theta \to 1$, so a bracketed solve always converges and needs no derivative; the Newton derivative in $\alpha$ is singular at $\alpha = 1$. A target outside the bracket is a failed calibration as for the other families: shape at the nearer end, residual recorded, the existing warning logged.
- Docstring Notes: the Kusuoka construction, the closed forms, Gini $1/(\alpha+1)$, the entropy reading $g_\alpha[X] = \mathsf PX + \xi_\alpha(X)$ (the cumulative residual Tsallis entropy), the interval condition range, and the special cases.
- DecL: `distortion D tsallis 3` works through `decl_spec`'s `param_name` fallback. One program per form (`tsallis 3`, `as "..." tsallis 0.5`) appended to `decl-testers.agg`, and the spec snapshot recaptured.
- Prose that enumerates the families gains `tsallis`: the `Distortion` class Notes (`spectral.py:597`), the `available_distortions` docstring (`spectral.py:961`), `docs/3_reference/3_x_Distortion.rst` (the autosummary and the family paragraph at line 38).

### [Distortion-From-Gini] the Gini index and construction from it

- **The Gini property** is `gini` after `[Gini-Rename]`. This phase adds a closed form for `mixture`, the weighted sum of its members' `gini` (the index is linear in $g$), beside the `tsallis` closed form of the previous phase. `ly`, `power` and `minimum` stay on quadrature, accurate to about $10^{-11}$.
- **The constructor.** A classmethod `Distortion.from_gini(kind, gini, *, label=None)` returning a new distortion of `kind` with that Gini index. Each one-parameter subclass supplies a classmethod hook `_params_from_gini(gini)` returning its natural keyword arguments:

  | kind | parameter | from Gini $G$ |
  |---|---|---|
  | `ccoc` | `d` | $G$ |
  | `ph` | `a` | $(1-G)/(1+G)$ |
  | `wang` | `lam` | $\sqrt2\,\Phi^{-1}((1+G)/2)$ |
  | `tsallis` | `alpha` | $1/G - 1$ |
  | `dual` | `b` | $(1+G)/(1-G)$ |
  | `tvar` | `p` | $G$ |

  The base hook returns `None`, and `from_gini` then raises `ValueError` naming the kind and its parameters (`decl_params`, or "distortion references" for `minimum` and `mixture`). That covers `bitvar`, `wtdtvar`, `beta`, `power`, `minimum`, `mixture`, and `cll`, `clin`, `lep`, `ly` (ruling `[From-Gini-Research-Families]`). $G$ outside the open interval $(0, 1)$ raises `ValueError` for every kind.
- **Single source.** `p_to_parameters(p)` and `consistent_distortions(p)` are rewritten over the hooks, with identical output (`p_to_parameters` still returns the `ccoc` discount $d$). Both gain `extended=False`, which adds `tsallis` in calibration-set order (ruling `[Consistent-Distortions-Extended]`).

### [Extended-Calibration-Set] `extended=` on `calibrate_distortions`

- `_pricing.EXTENDED_CALIBRATION_DISTORTIONS = ('ccoc', 'ph', 'wang', 'tsallis', 'dual', 'tvar')`, beside the default constant.
- `extended=False`, keyword only, on `_pricing.calibrate_distortions`, `Aggregate.calibrate_distortions`, `Portfolio.calibrate_distortions` and `AggregateCHF.calibrate_distortions`. `names` defaults to `None` on all four (it is already `None` on `AggregateCHF`), resolved in `_pricing` to the extended or the default constant. `names` given together with `extended=True` raises `ValueError` (ruling `[Extended-Names-Conflict]`). Every call that passes neither is unchanged.
- `charts/_emit_bounds.CALIBRATED` gains `'tsallis': 'Tsallis'` between `wang` and `dual`, so the envelope draws it when present.
- Verify, and fix if needed, that the pricing exhibits and `CalibrationResult` handle a sixth family: they pivot on the `distortion` index level, so they are expected to need nothing.
- The acceptability panels (`_pricing.EVAL_FAMILIES`, `EVAL_FAMILIES_ANCHORED`) are not changed; adding `tsallis` there is a separate decision.

## Rulings

- **[Gini-Name]** Ruled by the author, 2026-11-06: rename `gini_p` to `gini` everywhere and drop `gini_p`, with no alias; the constructor is `from_gini`. The break of the stability promise is accepted because nothing outside the author's own work reads the name yet.

- **[Gini-Rename-Version]** Ruled by the author, 2026-11-06: the four phases stay in the `2.1.0aN` series, a deliberate exception to the house rule that a breaking change retargets the series to the next major, since nothing outside the author's own work reads `gini_p` yet.

The four below were open with recommendations, and the author accepted every recommendation as the ruling, 2026-11-06:

- **[Tsallis-Strict-Flag]** Recommended: `strict_pricing = False`. `available_distortions()` and every default output stay the five standard families, and `tsallis` enters a calibration only through `extended=True` or `names=`. With `True` the default list grows to six and `test_distortion_construction.py:405` changes.
- **[From-Gini-Research-Families]** Recommended: `cll`, `clin`, `lep` and `ly` raise, since each has `r0` as well as its shape. The alternative is a numerical solve for the shape at a given `r0`.
- **[Extended-Names-Conflict]** Recommended: `names` with `extended=True` raises. The alternative, `extended` ignored, hides a mistake.
- **[Consistent-Distortions-Extended]** Recommended: `extended=False` on `p_to_parameters` and `consistent_distortions` too, rather than the note's suggestion of adding `tsallis` to them unconditionally, which would change their output.

## Names vetted

2026-11-06, `rg` over `src/aggregate` and `tests`: `tsallis`, `TsallisDistortion`, `from_gini`, `_params_from_gini`, `EXTENDED_CALIBRATION_DISTORTIONS` and a `gini` attribute or method find nothing. `gini` exists only as a row label in a distortion's stats and describe tables, which is not an attribute. A frame that carries both a `gini` column and a `gini` row does not occur. `alpha` is also `PowerDistortion`'s parameter, a different class. The class name follows its siblings (`PHDistortion`, `WangDistortion`) rather than the `Base<Kind>` prefix rule in `CLAUDE.md`, which applies to new taxonomies and the deferred Distortion cleanup; mixing the two forms inside one taxonomy is worse than either.

## Files

- `[Gini-Rename]`: the 29 files under "Where `gini_p` appears", and the two snapshots recaptured.
- `src/aggregate/spectral.py`: `TsallisDistortion`; `from_gini` and `_params_from_gini` on the base and the six classes; the `mixture` closed form; `p_to_parameters`, `consistent_distortions`; docstrings that list the families.
- `src/aggregate/_pricing.py`: `EXTENDED_CALIBRATION_DISTORTIONS`; `extended` and the `names` resolution in `calibrate_distortions`.
- `src/aggregate/_aggregate.py`, `src/aggregate/_portfolio.py`, `src/aggregate/characteristic_function.py`: `extended` on each `calibrate_distortions`, with docstrings.
- `src/aggregate/charts/_emit_bounds.py`: `CALIBRATED`.
- `src/aggregate/agg/decl-testers.agg`, `tests/data/expected_specs.json` (recaptured).
- `tests/test_distortion_tsallis.py` (new); additions to `tests/test_distortion_construction.py` and `tests/test_distortion_calibrate.py`. Rerun `tests/capture_distortion_snapshot.py` if it iterates the registry, and read its diff.
- `docs/3_reference/3_x_Distortion.rst`; `dev/FEATURES.csv` via `dev/regen_features.py`.
- Using page: a Tsallis and `from_gini` section in `D:/Projects/Using/aggregate/distortions.qmd`, or a new `tsallis.qmd`, whichever `D:/Projects/Using/CLAUDE.md` favors.

## Tests

0. **Rename.** `Distortion('ph', 0.5).gini` is $1/3$ and `gini_p` raises `AttributeError`; `calibration_df` and an evaluation frame have a `gini` column and no `gini_p`; the existing tests pass with the new name.
1. **Closed form.** $g_\alpha$ against `quad` of the Kusuoka integral $\int_0^1 \min(u/s, 1)\,\alpha s^{\alpha-1}\,ds$ at $\alpha \in \{0.25, 0.5, 1, 1.5, 2, 3, 7\}$ and $u \in \{0.05, 0.3, 0.7\}$, within $10^{-9}$.
2. **Continuity at 1.** $g$ and $g'$ at $\alpha = 1 \pm 10^{-9}$ against $\alpha = 1$, within $10^{-8}$; $\alpha = 10^{6}$ and $10^{-6}$ finite, near the mean and the maximum.
3. **Special cases.** $\alpha = 2$ equals `dual(2)`; $\alpha = 1/2$ equals $2\sqrt u - u$.
4. **Gini.** `gini` $= 1/(\alpha+1)$, and quadrature of `g` agrees within $10^{-10}$.
5. **Derivative and inverse.** `g_prime` against a central difference; `g(g_inv(y)) = y` within $10^{-12}$.
6. **Describe.** The closed-form moments $\alpha/((k+1)(k+\alpha))$ against quadrature; `validation_df` passes.
7. **Calibration on the uniform.** For $X \sim U(0,1)$, $\rho_g(X) = \int g = (1 + G)/2$, so calibrating any of the six one-parameter families to premium $(1+G)/2$ on a fine uniform grid returns `from_gini(kind, G)`. One test, six families, three values of $G$.
8. **Calibration on a book.** `calibrate_distortions(..., names=['tsallis'])` on a fixture `Aggregate` hits the premium within the existing tolerance.
9. **DecL.** `distortion D tsallis 3` and `distortion D as "T" tsallis 0.5` build and round-trip.
10. **From Gini.** Round trip `from_gini(k, G).gini == G` within $10^{-12}$ for the six; `ValueError` for each multi-parameter kind, naming its parameters, and for $G \in \{0, 1, -0.1, 1.5\}$.
11. **Single source.** `p_to_parameters(p)` and `consistent_distortions(p)` unchanged from 2.0.0 at three `p`; `extended=True` adds `tsallis` between `wang` and `dual`.
12. **Extended.** `calibrate_distortions(..., extended=True)` on an `Aggregate`, a `Portfolio` and an `AggregateCHF` returns the six families in order; the default returns the five, unchanged; `names` with `extended=True` raises.
13. **Defaults unchanged.** `available_distortions() == DEFAULT_CALIBRATION_DISTORTIONS`; `tsallis` is in `available_distortions(pricing=True, strict=False)` and in the DecL words.
14. **Envelope.** The envelope chart of an extended calibration has a Tsallis curve.

## Acceptance check

- The tests above pass. Tier 2 passes throughout; tier 3 passes at each bump, with `-W error::RuntimeWarning`, since every phase is numerical.
- `Distortion('tsallis', alpha=3).gini` is exactly `0.25`, and `Distortion.from_gini('tsallis', 0.25)` returns it.
- After `[Gini-Rename]`, `rg -w gini_p src tests docs` is empty and the snapshot diffs contain only the rename.
- `consistent_distortions(0.125, extended=True)['tsallis']` is order 7, and its `g(0.1)` equals $(0.7 - 0.1^7)/6$.
- Every existing output that does not ask for `tsallis` or `extended` is unchanged: the distortion and spec snapshots move only by the added entries.

## Ripple

- The DecL distortion words gain `tsallis`. `V:/dev/aggregate-api/web/src/decl-keywords.json` mirrors parser terminal labels, not distortion kinds, so it needs nothing; an app list of distortion families fed from `decl_words` picks it up.
- The note `08-tsallis.qmd` can drop its session class and use the library's once this lands; that edit belongs to the note's project.
- **`[Gini-Rename]` breaks these readers of `gini_p` outside this repository.** Five Using pages (`D:/Projects/Using/aggregate/allocation-bounds.qmd`, `distortions.qmd`, `price-evaluate.qmd`, `price.qmd`, `pricing-bounds.qmd`) are edited, rendered and committed in that repository by the executing session, per section 8 of the `execute-plan` skill. The rest are the author's to edit and re-render: the KOG-blog post `posts/notes/2026-10-27-Pricing-occurrence-reinsurance-with-an-annual-aggregate-deductible/index.qmd`; and five files in `D:/Projects/Scratch` (`08-tsallis.qmd`, `geometry_of_distortion_risk_measures.qmd`, three scripts under `tmp/`). `V:/dev/aggregate-api` mentions `gini_p` only in a retired plan, so the app needs nothing.
- The auto-memory note `show-gini-p-for-distortions` names `gini_p`; it is updated to `gini` when the rename lands.

## Execution log

Executed 2026-11-06 from 2.1.0a1 (`e6cc9bee`). The review found the plan's facts current, with the drift recorded below.

### Divergences

- **[Review-Drift]** `gini_p` occurs 130 times in 29 live files, not 135. The default-list assertion is at `test_distortion_construction.py:408`, not 405. `tests/capture_distortion_snapshot.py` does not iterate the registry, so it needs no rerun. Two live dev notes, `dev/issues-arising-from-using-notes.md` and `dev/task-features.md`, mention `gini_p` and are left as written, with `dev/deferred/`.
- **[DecL-Words-Sorted]** The DecL distortion words are `sorted(available_distortions(pricing=False))` (`decl_words/__init__.py:225`), so `tsallis` lands alphabetically there, not between `wang` and `dual`. Test 13 checks membership only.

- **[Gini-Column-Home]** The `gini` column lives in `distortion_df`, built in `_pricing._calibration_frames`, not in `calibration_df`, which holds the pentagon. Test 0 checks `distortion_df` and `evaluation_df`.
- **[Gini-Pinning-Test]** `rg -w gini_p src tests docs` returns one test, `test_gini_replaces_gini_p_with_no_alias` in `tests/test_distortion_construction.py`, which has to spell the old name to pin its absence. Nothing else matches.
- **[Cheat-Sheet-Rename]** Two tracked cheat sheets, `cheat-sheets/Distortion_Cheat_Sheet.tex` and `cheat-sheets/PnL_Cheat_Sheet.tex`, contain `gini\_p`; the plan's list does not include them, and both are renamed. Their PDFs are the author's to rebuild. `docs/2_aggregate_overview/info-strings.rst` also contains `\mathrm{gini\_p}`, which a search for `gini_p` does not match; it is renamed, and its simple table keeps its column alignment.
- **[Exhibit-Snapshot-Ulp]** A fresh `capture_exhibit_snapshots.py` also moved six raw values of the `CalibrationAggregate` calibrate exhibit in the last bit (for example `0.166127844532068` to `0.16612784453206797`), deterministic between reruns on this machine and unrelated to the rename. `_assert_snapshot_close` compares raw floats at relative `1e-9` for exactly this reason. The committed snapshot is HEAD's with `gini_p` replaced, checked against the fresh capture to within `1e-9`, so its diff is the rename alone.

### [Gini-Rename], 2.1.0a2

Renamed in 29 files: the property, the info row (`INFO_LABEL_WIDTH` padding kept, so the `test_hygiene4.py` row moved two spaces), the two frame column lists, the exhibit block and captions, the format keys, docstrings, docs, the `library.agg` and `decl-testers.agg` comments and `note{}`, and the tests, whose renamed function names collide with nothing. `expected_specs.json` moved by the one `note{}`. `dev/regen_features.py` reported no change. The auto-memory note became `show-gini-for-distortions`. Gate: tier 3 with `-W error::RuntimeWarning`, which subsumes plain tier 3: 11,554 passed, 3 skipped.

### [Tsallis-Distortion], 2.1.0a3

`TsallisDistortion` sits between `WangDistortion` and `DualDistortion` as specified, with the flags, closed forms, Kusuoka density, `calibrate` and docstring Notes of the plan. New tests in `tests/test_distortion_tsallis.py` (plan tests 1 to 6, 8, 9 and 13); `tsallis` joined the parametrized lists of `test_distortion_construction.py`, `test_distortion_calibrate.py` and `test_distortion_decl.py`. `decl-testers.agg` gained `distortion W.tts tsallis 3` and `distortion W.ttsh as "T" tsallis 0.5`, and the spec snapshot moved by those two entries only.

- **[Distortion-Dtype]** `spectral.DISTORTION_ORDER`, the ordered categorical behind the `distortion` index of `distortion_df` and `evaluation_df`, listed a fixed set of kinds without `tsallis`, so `calibrate_distortions(..., names=['tsallis'])` returned a `NaN` index. Under `[Extended-Calibration-Set]` the frames were to need nothing, since they pivot on that level. `tsallis` joins the order between `wang` and `dual`; no existing frame or snapshot moves.
- **[Tsallis-Overflow-Branch]** The plan's stable form `u(1 - expm1(beta L)/beta)` overflows for `alpha < 1` once `beta L` passes about 709, at `alpha` near 0 and `u` near the smallest float, which a survival vector in a calibration reaches. `_g_alpha` writes `g = u - t/beta` with `t = u expm1(beta L)`, and switches to `t = exp(alpha L) - u` above `beta L = 700`, where there is no cancellation. `g_prime` returns `inf` there, the correct float, under an `over` guard.
- **[Tsallis-Inverse-In-Log]** Bisection on `u` to 60 halvings has absolute precision about 9e-19, which misses test 5's `1e-12` for a small `alpha` and `y` (at `alpha = 0.25`, `y = 1e-3`, the root is near 3e-13). `g_inv` bisects on `log u` over `[-708, 0]` with 64 halvings instead, a relative precision near 4e-17, keeping the closed forms at `alpha` 2 and 1/2. Below `g(e^{-708})` it returns 0: at `alpha = 0.001` the root of `g(u) = 0.1` is near 1e-1000, which no float holds.
- **[Tsallis-Static-Factory]** `Distortion.tsallis(alpha)` joins the static factories `ph`, `wang` and `dual`; the plan does not include it. No collision: `tsallis` is otherwise only a registry key.
- **[Tsallis-First-Gate]** The first tier 3 run failed three cases. `tests/test_decl_words.py` keeps one constructor example per DecL distortion word in `DISTORTION_EXAMPLES`, which needed a `tsallis` entry. The failed-calibration test read `caplog` bare and saw nothing in the full run, because five test modules call `logging.disable(logging.CRITICAL)` at import and every xdist worker imports them all; it now uses `caplog.at_level(..., logger='aggregate.spectral')` as its siblings do, which lifts the disable. The second run passed 11,620 and failed only `test_bivariate.py::test_mv_explain_flags_clipped_book`, the known `[Bivariate-Gate-Flake]` under `-W error::RuntimeWarning`, which passed when rerun alone under the same flag and touches no distortion code.
- **[Tsallis-Clip]** `g`, `g_prime` and `g_inv` clip their argument to `[0, 1]`, since `log` of a rounding negative from a survival function would otherwise be `nan` with a warning under the numerics gate.
