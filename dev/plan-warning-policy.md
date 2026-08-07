# [Warning-Policy] Signal, not noise

## Context

The rendered reproductions book (`docs/reproductions/_book/`) carries about **250 stderr lines** across 17 pages. They are not spread thin: **208 `Warning` lines come from 9 source lines**, plus ~41 `logger.warning` lines from one more. Two chapters produce 72% of the total.

| Chapter | Lines | What it is doing |
|---|---|---|
| `Mack2003` | 85 | infinite-mean Pareto, 56 `build(normalize=False)` in a nested loop, a 120k-edge spliced `chistogram` |
| `Mata2005` | 64 | one `build()` per `sf(a, mean)` inside four nested comprehensions, run twice |
| `Liu2026` | 42 + 39 logger | Type-I Pareto everywhere |
| `BenRached2024` | 11 | Lévy severity, no finite mean |
| `Venter1983` | 5 | `chistogram` on a non-uniform edge grid |
| `Homer2003`, `Bruno2006` | 1 each | a `1.6e-12` deficit, i.e. nothing |

The counts are the problem, not the warnings. `Mata2005` prints the **same two messages** 64 times. `Mack2003` prints 25 identical copies of a scipy warning whose text names its own fix. The genuinely interesting facts, that Mack's Pareto splice has no finite mean and that BenRached's Lévy severity loses 12.6% of its mass, are buried in the repetition rather than surfaced by it.

Four independent causes, each with a clean fix:

1. **`DefectiveDistributionWarning` (95)** fires at every `update()` on `deficit > VALIDATION_NOISE` (`1e-12`) and embeds the deficit in the message text, so Python's own once-per-location dedup never engages. Sixteen of the 95 are below `1e-4`, three are below `1e-7`, and one is `1.6e-12`. Meanwhile the library's own pricing code already treats `1e-4` (`validation.deficit_materiality`) as the line between numerical truncation and an economic problem. Construction warns three orders of magnitude tighter than pricing cares about.
2. **`RuntimeWarning: Bin widths are not constant` (30)** is scipy's, from `rv_histogram.__init__`, and only fires because two `aggregate` call sites leave `density` at its `None` default. The message states its own fix.
3. **`RuntimeWarning: invalid value encountered` (44)** is `inf - inf` and `0 * inf` on moments that genuinely do not exist. `NaN` is the correct answer; numpy's report adds nothing. Already tracked as **[RuntimeWarning-Census]** in `dev/TODO.md`.
4. **Type-I Pareto (39 `IntegrationWarning` + 41 logger lines)**, not in the original report but a third of the noise. `_partial_e` has an analytic branch only for the shifted (Lomax) form, so every `sev {xm} * pareto {a}` logs a warning and falls back to `quad` from 0 to infinity on a heavy tail.

Intended outcome: a defective distribution is announced **once**, a law priced through a distortion while materially defective is announced **once** (that is where forwards and backwards `S` actually diverge), every object carries its own verdict silently in `validation`, and the three benign families disappear entirely.

Author decisions taken during planning: material threshold plus once-per-session plus a price-time warning; once keyed per session per category; no new opt-out surface (`silence_warnings` already covers it); finish the whole `[RuntimeWarning-Census]`; add the analytic Type-I Pareto branch; record the deficit as a new `Validation.DEFECTIVE` flag; on the book side fix only the chapter's own warning and leave the legitimate large deficits visible.

---

## Deliverable 1 — `[Warning-Policy]`, bump to `1.0.0a218`

### 1a. A once-per-session helper

New in `src/aggregate/constants.py`, beside the six warning classes (it is the import-graph leaf and needs only stdlib `warnings`). Add all three names to `constants.__all__`, which is star-exported by `src/aggregate/__init__.py:36`, so they land on `aggregate.*`.

```python
def warn_once(message, category, *, key=None, stacklevel=3)
def reset_warn_once()                    # public; clears the registry
def warn_once_isolated()                 # context manager
```

- Module-level `set()` of keys. `key` defaults to `category`, so "once per session, per category" is the default behaviour; an explicit key splits one category into several independently-once channels (used in 1c).
- The single emission appends one sentence, not a block: `... (further occurrences this session are suppressed; each object records its own deficit in validation).` Keep it on the message line, no internal blank lines.
- `warn_once_isolated()` both **bypasses** the registry (every call warns) and **restores** it on exit (nothing inside consumes the session budget). One context manager covers both existing internal needs:
  - `src/aggregate/_bucket_window.py:1132` — the speculative per-unit pre-pass already wraps `catch_warnings` + `simplefilter('ignore', DefectiveDistributionWarning)`. Without isolation it would silently spend the budget before the real combine runs.
  - `src/aggregate/_bucket_window.py:1685` — `sharpen`'s `_cell` uses `catch_warnings(record=True)` + `simplefilter('always')` and **counts** what fires into `row['warnings']` / `row['warns']`. It needs every cell to warn.

Vetted against the existing surface: no `warn_once` / `reset_warn_once` / `warn_once_isolated` anywhere in `src/aggregate`; `warn_degenerate` (`_pricing.py:820`) is the naming precedent. No new instance attribute, so nothing can shadow a method.

Note while here: `ZeroModifiedExposureWarning` (`constants.py:243`) is missing from `constants.__all__`. Add it.

### 1b. Raise the construction threshold to materiality

`src/aggregate/_aggregate.py:3523` and `src/aggregate/_portfolio.py:2365`: change `deficit > VALIDATION_NOISE` to `deficit > DEFICIT_MATERIALITY` and route through `warn_once`. Align the three bivariate sites (`bivariate.py:455`, `:779`, `:951`), which currently use a hard-coded `1e-8`, onto the same constant. The sizer's clip warning (`_bucket_window.py:874`) is a *reach* test, not a deficit test, so its trigger is unchanged, but it too goes through `warn_once`.

**Do not move the `sharpen` gate.** `_bucket_window.py:1657` keeps `deficit > VALIDATION_NOISE`: the probe is choosing among candidate grids, where insisting on losing no mass at all is free, whereas the warning is about interrupting the user. This breaks a stated invariant, so two places must be corrected to say the gate is deliberately tighter than the warning: `docs/2_aggregate_overview/bucket-selection.rst:226` ("The threshold is the library's own `VALIDATION_NOISE`, the level at which `DefectiveDistributionWarning` already fires, so a cell rejected here is exactly one that would warn when you used it") and the comment at `_bucket_window.py:1517`.

### 1c. A price-time warning at the single choke point

`choquet_weights` (`src/aggregate/spectral.py:112`) is the one gate every pricing route passes: `Distortion.price` (`spectral.py:1914`, `:1920`, `:1956`), `_portfolio_common.py:152`, `_aggregate.py:4109`. Its three-band logic at `spectral.py:210-242` already distinguishes dust, parked truncation, and a material deficit; today a material deficit with `allow_deficit=True` only writes `logger.debug`.

Split that branch: park quietly up to `DEFICIT_MATERIALITY`, and above it `warn_once` with `key='defective-pricing'` naming the deficit, the parking direction, and that the two directions differ by exactly the deficit. The construction-time call uses `key='defective-construction'`, so the two are independently once and neither swallows the other. Same warning **class**, so `silence_warnings(DefectiveDistributionWarning)` still covers both and the class inventory is unchanged.

`stacklevel` from inside `choquet_weights` lands on library internals; pick it so the reported frame is the user's `price` / `apply_distortion` call.

### 1d. `Validation.DEFECTIVE`

- Add `DEFECTIVE = auto()` to the `Validation` flag (`constants.py:172`). It is a genuine failure, so it is deliberately **not** in the `.passes` set (`constants.py:194`): a defective object flips `valid` to `False` and `qd` says so.
- Set it in `valid_aggregate` (`_validation.py:73`) from `1 - float(np.sum(agg.agg_density))` and in `valid_portfolio` (`_validation.py:197`) from `1 - float(np.sum(port.density_df['p_total']))`, both against `DEFICIT_MATERIALITY`.
- `explain_validation` (`_validation.py:27`) gains an optional `deficit=None` so the short form can read `pmf deficit 4.141e-01` rather than a bare flag name. Report it **first** among failures: a deficit makes every moment comparison below it uninformative, the same argument the function already applies to mean-before-CV.
- `validation_explanation` (`_validation.py:453`) gains a paragraph next to the existing `ALIASING` and `REINSURANCE` ones.

Per the author's choice this is the flag route only: **no** new `Aggregate.deficit` / `.defective` public properties, so no new names to collide with the existing surface.

### 1e. `rv_histogram`

Pass `density=True` explicitly at `src/aggregate/_severity.py:1957` (`SeverityCHistogram`, `aps = ps / np.diff(xss)` is already a density) and `:2030` (`SeverityMeta`, whose bins are `bs*1e-7`, `bs/2`, `bs, bs, ...` and so **never** constant, meaning every `meta` severity warns today). `density=True` is exactly what scipy assumes when `density is None`, so this is bit-for-bit unchanged output.

### 1f. Docs and hygiene for this bump

`DefectiveDistributionWarning` docstring rewritten for the new policy; `utilities.silence_warnings` (`utilities.py:38`) docstring documents the chosen opt-out recipe and points at `reset_warn_once`; `config.default.toml` `deficit_materiality` comment now also names it as the warning threshold; `docs/2_aggregate_overview/features.rst:886` and `:1245`; `CHANGELOG.md` `## 1.0.0a218`.

---

## Deliverable 2 — `[RuntimeWarning-Census]`, bump to `1.0.0a219`

Closes the `dev/TODO.md` item of the same name (line 501).

**Measure first, then fix.** Run `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning` to get the *current* list. The TODO's census (`spectral.py` 1156/2304/2399/3607/3847/3856, `moments.py` 193/284/285/299/315, `pedagogy.py` 1571/1711, `_aggregate_compute.py:330`) predates several changes, and exploration surfaced further candidates that are unverified until measured: `spectral.py` 2337/2344/2432/2508/2647/2794/3561/3644/3877 (`g_prime` / `calibrate` boundary evaluation, `0 ** negative`, `0/0`, `log(0)`), `bounds.py:700`, `iman_conover.py:421`, `copula.py:358`, `moments.py` 395/411.

Decide each on its merits, exactly as the TODO states: an `np.errstate` guard where the discarded branch is genuinely unreachable, a real fix where the `NaN` can reach a result. Follow the `[TVaR-Endpoint-Noise]` (`a179`, commit `c588c05`) pattern — the guard plus a `Notes` paragraph saying why the arithmetic never reaches the answer:

```python
with np.errstate(invalid='ignore'):
    return (f1 * s1,
            f1 * s2 + (f2 - f1) * s1 ** 2,
            f1 * s3 + f3 * s1 ** 3 + 3 * (f2 - f1) * s1 * s2
            + (- 3 * f2 + 2 * f1) * s1 ** 3)
```

The four `moments.py` sites the book actually hits (193, 284/285, 299, 315) are the clear-cut case: an infinite severity moment makes these `inf - inf` / `0 * inf`, and `NaN` is the correct report for an undefined moment.

Thirty `np.errstate` sites already exist (`_grid_distribution.py`, `_portfolio_density.py`, `_portfolio_common.py`, `_aggregate.py:1220-1287`, ...) — match their idiom rather than inventing one.

Once the count is zero, add the gate. Prefer `[tool.pytest.ini_options] filterwarnings = ["error::RuntimeWarning", ...]` in `pyproject.toml` with explicit `ignore` entries for any third-party residue; if third-party noise makes that brittle, document `-W error::RuntimeWarning` as the release-gate invocation in the CLAUDE.md testing section instead. That call belongs at execution time, on what the measurement shows.

---

## Deliverable 3 — `[Pareto-Type-I-Analytic]`, bump to `1.0.0a220`

Its own bump because it **moves numbers** (quadrature to exact), so it needs its own bisect point.

`_partial_e` (`src/aggregate/_severity.py:59`) handles the shifted Pareto (`scale=λ, loc=-λ`, so `λ + loc == 0`) analytically and sends everything else to `_partial_e_numeric` with a `logger.warning`. The Type-I / single-parameter form (`scale=λ, loc=0`, support `[λ, ∞)`) is the common case in `Liu2026` and hits the fallback every time.

Add the branch before the existing `if λ + loc != 0` test at `_severity.py:130`. For `S(x) = (λ/x)^α`, `f(x) = α λ^α x^{-α-1}`:

```
∫_λ^a x^k f(x) dx = α λ^α (a^{k-α} − λ^{k-α}) / (k − α),     k ≠ α
                  = α λ^α log(a/λ),                          k = α
```

Checks: `k = 0` gives `1 − (λ/a)^α`; `a = ∞` with `k < α` gives `α λ^k / (α − k)`, so `k = 1` is `αλ/(α−1)`, the Type-I mean; `k ≥ α` gives `inf`, which is correct and is what feeds the `moments.py` `NaN`s guarded in Deliverable 2. Return zeros for `a ≤ λ` alongside the existing `a == 0` early return at `_severity.py:85`.

Also harden the fallback, which genuinely shifted Paretos still use:

- `_partial_e_numeric` (`_severity.py:45`) integrates from `float(fz.support()[0])`, not `0`. Quadrature over a flat dead region `[0, λ)` is most of why `quad` reports "probably divergent".
- `logger.warning` at `_severity.py:131` becomes `logger.debug`: falling back is a routine code path, not a problem, and it is 41 uncontrolled stderr lines in the book. Same for `_severity.py:54`.
- Wrap the residual `quad` in `catch_warnings` suppressing `IntegrationWarning`, with a comment saying the caller reports non-convergence through the existing `temp[1] > 1e-4` check.

`_partial_e_numeric` stays as the auditing reference its docstring describes, and becomes the test oracle below.

---

## Deliverable 4 — the book (no bump; author's to commit)

Per the author's choice, only the chapter's own warning:

- `docs/reproductions/Mack2003.qmd:132` — `(edges / U) ** -ALPHA` with `edges[0] == 0.0`. The enclosing `np.where` discards it and line 133 sets `S[0] = 1.0`, so it is harmless. Start the grid above 0 or wrap in `np.errstate`, following the precedent already in the book at `Bear1990.qmd:54`: a targeted filter with a comment saying why.

**Not** doing: no `warning: false` in `_quarto.yml` or on any chunk. `Mack2003`'s infinite-mean Pareto and `BenRached2024`'s 12.6% Lévy deficit warn for real reasons the prose already discusses; under once-per-session they collapse to one line per chapter, which is the right outcome. The book is not re-rendered here.

Noted, not fixed: `_quarto.yml` lists `Homer2003.qmd` twice, `Bear1990.html` was never built, and absolute `T:\worktrees\...` and `C:\Users\...\ipykernel_53636\...` paths leak into published pages.

---

## Tests

**Breaking change to catch first.** Once-per-session defeats `pytest.warns` across tests in one process. Add an autouse fixture to `tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def _reset_warn_once():
    reset_warn_once()
```

Affected today: `tests/test_negative_x.py:195` (`pytest.warns(DefectiveDistributionWarning)`) and `tests/test_reins_bivariate.py:293` (`match='clipped'`). Also confirm the first still warns under the raised threshold — its deficit must clear `1e-4`, not merely `1e-12`. `tests/test_pnl.py:217` (`simplefilter('error', ...)`, expecting silence) is unaffected. `tests/test_validation.py` has 6 `Validation` references to reconcile with the new member.

New coverage:

- `warn_once`: fires once, second call silent, `reset_warn_once` re-arms, distinct keys are independently once, `warn_once_isolated` both bypasses and restores.
- Threshold: a `3e-6` deficit is silent, a `3e-3` deficit warns once; `_bucket_window`'s `sharpen` still counts a warning per probed cell.
- Price time: a materially defective law priced through a distortion warns once even when construction already warned.
- `Validation.DEFECTIVE` set / cleared, `valid` flips, `explain_validation` and `validation_explanation` render the deficit.
- `chistogram` on a non-uniform edge grid and a `meta` severity emit **no** `RuntimeWarning`; `rv_histogram` output is unchanged versus the current `density=None` path.
- Type-I Pareto: analytic vs `_partial_e_numeric` for `α > 3` (all moments finite), vs `fz.stats` for `a = ∞`, the `k == α` log branch, `a ≤ λ`, and `k ≥ α` returning `inf`.

Per the standing rule, any new DecL programs used in tests get appended to `src/aggregate/agg/decl-testers.agg` under the matching section, and must round-trip.

---

## Verification

1. **Edit loop** — `uv run pytest -n0 --dist no --testmon-forceselect` (all three flags are load-bearing).
2. **Per deliverable** — `.venv/Scripts/python.exe -m pytest` (the full fast suite; note `UV_PROJECT_ENVIRONMENT=.doc-venv` is ambient, so a bare `uv run` resolves to the wrong venv).
3. **Per bump** — `uv run pytest -m 'slow or not slow'`, ~6 minutes. Deliverable 3 moves Type-I Pareto numbers, so expect baseline movement there and re-baseline deliberately, not reflexively.
4. **Census gate** — `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning` before and after Deliverable 2. The TODO records the a179 starting point of 52 failed / 0 errors; the target is 0 / 0.
5. **End to end, the actual point of the exercise** — re-run the two worst chapters headless and count. Expected: `Mata2005` 64 lines to 1, `Mack2003` 85 to roughly 2 (one construction deficit, one clip), `Liu2026` 81 to 0, `Venter1983` 5 to 0, `Homer2003` / `Bruno2006` 1 to 0, `BenRached2024` 11 to 1. Grand total ~250 to under 10. Do this with a small script over the `.qmd` code cells rather than a full Quarto render — the doc build is slow and is the author's to run.
6. **Smoke** — `build('agg Dice dfreq [3] dsev [1:6]')` then `qd`, plus one deliberately under-gridded build to see exactly one warning and a `DEFECTIVE` validation line.

## Release hygiene

Three bumps, three commits, each a coherent unit carrying its code change, `pyproject.toml`, its `CHANGELOG.md` section, and any `dev/TODO.md` edit. One-line subjects, no body, no trailers:

```
[Warning-Policy] a218: defective warns once per session at the materiality floor, and again when a defective law is actually priced
[RuntimeWarning-Census] a219: guard the benign numpy boundary arithmetic; -W error::RuntimeWarning is clean
[Pareto-Type-I-Analytic] a220: closed-form partial moments for the single-parameter Pareto replace divergent quadrature
```

`dev/TODO.md`: close `[RuntimeWarning-Census]` at a219. `[v1-Journey-Philosophy]` (line 322) already promises a statement that the user manages warnings — this plan is what that paragraph will describe, so leave a pointer. The `Mack2003.qmd` edit does not bump and is left uncommitted for the author.
