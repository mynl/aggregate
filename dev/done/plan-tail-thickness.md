# tail-thickness — ordered tail classification for Aggregate & Portfolio

> **STATUS: Phase 1 DONE (2026-06-04), shipped in 1.0.0a29.** New leaf module
> `src/aggregate/tail.py` (TailClass IntEnum with sentinel UNKNOWN, TailInfo /
> TailClasses structs, `classify_frequency` / `classify_severity` / `combine` /
> `aggregate_tail_info`, text builders, and the moved `_BOUNDED_*` tables).
> `Aggregate` / `Severity` / `Portfolio` `.bounded` rewritten as derived views;
> `.tail_class` / `.tail_description` / `.tail_explanation` added; `.info`
> integration on both. Tests in `tests/test_tail.py` (46, all green; full suite
> 993 green). DecL examples appended to `test_decl.agg` TAIL section.
>
> **Open-decision C resolved:** option 1 — `UNKNOWN` is a sentinel off the order
> that poisons `combine`; `bounded` is decided structurally so it is unaffected.
>
> **Phase 2 (deferred, NOT built):** `estimate_tail_from_density` (mean-excess
> slope, log-log-S alpha, discrete log-concavity) for histogram / meta / spliced
> / unknown families and the aggregate `log_concave` (currently `None`). Heavy
> mixed-Poisson refinement (PIG / Sichel / Neyman-A) and `invgauss` are the
> watch items. Everything below the rule is the original plan, kept for Phase 2.

New feature; bumps the `a*` version. Standalone (no dependency on the reins
plans).

> ## Critical review & revisions (2026-06-04) — read FIRST; OVERRIDES the body
>
> Reviewed against the current tree. The math base is sound; the corrections
> below are about phasing, integration safety, and stale references. Headline
> recommendation: **ship a deterministic family-lookup base (Phase 1) and defer
> the numeric density estimator (Phase 2)** — that is where the author expects
> iteration, and it must not block a testable, exact-by-construction core.
>
> ### A. Stale line numbers (all refs in the body are pre-refactor; corrected)
> - `Aggregate.bounded` → **`distributions.py:1815`** (not 1680).
> - `Severity.bounded` → **`distributions.py:7358`** (not 5342).
> - `_BOUNDED_FREQS` → **`:1718`**, `_BOUNDED_SCIPY_SEVS` → **`:1722`** (not 1583).
> - `Aggregate.info` → **`distributions.py:3592`** (not 2622).
> - `Portfolio.bounded` → **`portfolio.py:857`**; `_certified_bounded` init at
>   **`portfolio.py:311`**.
> - lifted-NA guard → **`portfolio.py:3046`** (not 2636); message at `:3054`.
> - `Portfolio.info` → **`portfolio.py:898`** (not 891), and it already prints a
>   `bounded` line at **`:903`** — the new tail line slots in beside it.
>
> ### B. PHASE THE WORK (main punch-up)
> **Phase 1 — deterministic core (this iteration, the "solid base"):**
> exact family-lookup classifiers (`classify_frequency`, `classify_severity`),
> the `combine` max-rule, the `TailClass` enum + `log_concave` from the tables,
> derived `bounded`, the `TailInfo` struct, `tail_description` /
> `tail_explanation` / `.info` integration, Portfolio worst-of, and the full
> equivalence test-suite. **No density required** (see invariant D).
> **Phase 2 — numeric estimator (next iteration):** `estimate_tail_from_density`
> (mean-excess slope, log-log-S α, discrete log-concavity) for histogram / meta
> / spliced / unknown families and the aggregate's `log_concave`. This is the
> tail-region-sensitive, iterate-against-real-data part; keeping it out of
> Phase 1 lets the base be exact and regression-testable.
>
> In Phase 1, families not in the tables resolve to **`TailClass.UNKNOWN`** (see
> C), `log_concave=None`, and the text reads "undetermined (numeric estimate
> pending)". Phase 2 fills these in without changing the Phase-1 surface.
>
> ### C. OPEN DECISION for the author — how should UNKNOWN sort?
> A 5-rung `IntEnum` used with `max()` cannot cleanly absorb "don't know yet".
> Three options; I recommend **(1)**:
> 1. **Add a sentinel `UNKNOWN` that is NOT on the order and poisons `combine`**
>    (any `max` involving UNKNOWN → UNKNOWN; the rung is reported as
>    "undetermined"). Honest: never fabricate a thickness we have not
>    established. `bounded` is unaffected (it is decided structurally, invariant
>    D), so the lifted-NA guard keeps working even when the rung is UNKNOWN.
> 2. Default unknown families to a **conservative `SUBEXPONENTIAL`** with a
>    "provisional" note. Risk: silently labels an unknown as moderately heavy.
> 3. Phase 1 simply **omits** the rung for unknown families (property returns
>    `None`); no sentinel. Cleanest type-wise but makes `combine`/worst-of
>    special-case `None`.
> Pick one before coding; it shapes the enum and `combine`.
>
> ### D. INVARIANT — `bounded` must stay spec-only (no density)
> Today `bounded` is a pure-spec property, resolvable **before** `update()`, and
> the lifted-NA guard (`portfolio.py:3046`) relies on that. The derivation
> `bounded ⇔ tail_class.agg == BOUNDED` is fine **only if** the BOUNDED
> determination never touches the numeric fallback. It does not: rung 0 is set
> by the structural test (finite family / atom / finite `limit` / finite
> `sev_ub` / `_certified_bounded`), which runs *before* family lookup and long
> before any density estimate. Make this an explicit invariant and **test that
> `a.bounded` and `port.bounded` are correct on an un-`update()`-d object.**
> Corollary: `classify_*` must order checks certified → structural-bounded →
> family-lookup → (Phase 2) numeric, and return at the first hit.
>
> ### E. Multi-component severity (gap in the body)
> `self.sevs` is a list (mixed/broadcast severities). `classify_severity` takes
> ONE `Severity`; `Aggregate.tail_class.sev` must be the **max (thickest) over
> components**, with `log_concave = all(component lc)`. This is consistent with
> today's `bounded = all(s.bounded ...)`: `max rung == 0 ⇔ every component
> bounded`. Spell this out; add a mixed-severity test (e.g. lognorm + pareto
> components → POWER_LAW).
>
> ### F. `tail_class` return type — define one struct
> Have a single internal `_tail_info()` build a small frozen dataclass/namedtuple
> `TailInfo(freq, sev, agg: TailClass; freq_lc, sev_lc, agg_lc: bool|None;
> alpha: float|None; flags: dict)`. Then `tail_class` → `(freq, sev, agg)` view,
> `bounded` → `agg == BOUNDED`, and `tail_description`/`tail_explanation` are
> formatters over the same struct (no recompute, no drift).
>
> ### G. scipy parameter-mapping risk (test per family)
> The param-aware rules read `sev_a`/`sev_b`, but the scipy→`sev_a` mapping is
> family-specific and a classic bug source: `gamma` shape, `weibull_min` `c`,
> `lognorm` σ, `beta` (a,b) all land in `sev_a`/`sev_b`, but the **power-law α**
> lives in different shape slots (`pareto`/`lomax` `b`, `genpareto` `c`=ξ,
> `burr` `c`,`d`, `t` `df`). Build a small per-family α-extraction table and
> **unit-test α against scipy** for each power-law family rather than assuming a
> uniform slot. `sev_name` can be a non-str (meta/copy wrap an object) — guard
> `isinstance(sev_name, str)` before any dict lookup, matching `Severity.bounded`.
>
> ### H. Module boundary / imports
> `tail.py` must stay a **leaf**: classifiers take objects and read attributes
> (`freq_name`, `sev_name`, `sev_a`, `limit`, `sev_ub`, `_certified_bounded`,
> `sev_density`, `agg_density`) **duck-typed** — no import of `distributions` /
> `portfolio`, so `distributions` can import `tail` without a cycle. Moving the
> two `_BOUNDED_*` frozensets is safe: **no test references them** and no other
> module imports them today; still, re-export them from `distributions` (`from
> .tail import _BOUNDED_FREQS, _BOUNDED_SCIPY_SEVS`) so any external `:1718`-era
> import keeps working.
>
> ### I. Math nits (small, fix in the tables)
> - `weibull_min` with **c == 1 is the exponential** → also `log_concave=True`
>   (the body marks lc only for c>1). Boundary, but be consistent.
> - `invgauss → EXPONENTIAL` is defensible (semi-heavy, `e^{-cx}x^{-3/2}`) but
>   borderline; tag it a Phase-2 numeric-refinement watch item alongside the
>   mixed-Poisson families.
> - The `max` combine rule is a sound engineering approximation under the
>   analytic-pgf caveat already stated; keep that caveat in `tail_explanation`
>   wording for the heavy-mixing cases.
>
> ### J. Compaction
> Recommend **compacting before implementing Phase 1.** This thread now carries
> the full `dsev_bucket` build plus a floating-point digression; none of it is
> needed for tail-thickness, and Phase 1 is a clean, self-contained build the
> revised plan fully specifies.

## Context

Knowing how heavy a distribution's tail is drives modelling choices — bucket
sizing, validation tolerance, and whether the lifted natural allocation is even
admissible. Today the only tail-shape signal is `bounded` (`Aggregate.bounded`
`distributions.py:1680`, `Severity.bounded` `:5342`, `Portfolio.bounded`
`portfolio.py:850`, backed by `_BOUNDED_FREQS` / `_BOUNDED_SCIPY_SEVS` `:1583`).
We want a richer, ordered tail-thickness label for frequency, severity, and the
resulting aggregate, surfaced in `.info`, plus a human explanation of how the
aggregate class arises.

**Dependency direction (per author):** the new tail classifier is the **single
source of truth**, and `bounded` is **derived from it** (`bounded ⇔ tail class
== BOUNDED`), not the reverse. The existing bounded-detection logic
(`_BOUNDED_FREQS`, `_BOUNDED_SCIPY_SEVS`, finite-limit / splice checks) is
**moved into** the new classifier; the three `.bounded` properties become thin
derived views. The certify-override (`_certified_bounded`, used to admit lifted
NA on a capped fat-tailed book) is preserved by having the classifier honour it
at the top.

Decisions locked with the author:
- **Sortable 5-rung scale** by tail-decay rate, plus a **separate `log_concave`
  boolean** surfaced in the text:
  `BOUNDED(0) < SUPER_EXPONENTIAL(1) < EXPONENTIAL(2) < SUBEXPONENTIAL(3) <
  POWER_LAW(4)`. Log-concavity is a structural density property, not a strict
  point on the thickness order, so it is a flag, not a rung (Poisson/Normal are
  log-concave & super-exponential; Gamma(k≥1) is log-concave but exponential-
  tailed; Lognormal is not log-concave → subexponential).
- **Engine**: exact family lookup keyed by `freq_name` / scipy `sev_name`
  (deterministic), with a **numeric fallback** estimating from the computed
  density tail for histograms / meta / spliced / unknown families (and to set the
  aggregate's `log_concave`, which a random sum does not inherit analytically).
- **Portfolio** reports the **worst-of** unit aggregate thickness (correct under
  independence) and names the driver.
- `.bounded` becomes a **derived** view of the classifier (rung 0); the bounded
  detection logic moves into `tail.py`.

## Math

**Per-component classification** (rung, `log_concave`):

- **Frequency**: `fixed/bernoulli/binomial/empirical → BOUNDED`;
  `poisson → SUPER_EXPONENTIAL` (log-concave);
  `negbin/geometric/pascal/logarithmic/delaporte → EXPONENTIAL` (geometric-type
  tail); mixed-Poisson families (`ig/sig/sichel*/neymana/beta/gamma`) classified
  by the **mixing law** — default `EXPONENTIAL` with a watch note (PIG/Sichel
  mixing can be heavier; the numeric fallback can refine).
- **Severity** (scipy `sev_name`, param-aware):
  `norm → SUPER_EXPONENTIAL` (lc); `expon → EXPONENTIAL` (lc);
  `gamma → EXPONENTIAL`, lc iff shape `sev_a ≥ 1`;
  `weibull_min` (c = `sev_a`): `c>1 → SUPER_EXPONENTIAL` (lc),
  `c==1 → EXPONENTIAL`, `c<1 → SUBEXPONENTIAL`;
  `lognorm → SUBEXPONENTIAL`; `invgauss → EXPONENTIAL`;
  `pareto/genpareto(ξ>0)/lomax/burr/fisk/loglogistic/t/cauchy/invweibull/frechet
  → POWER_LAW` (report α from parameters; flag α<2 ⇒ infinite variance, α≤1 ⇒
  infinite mean — the alpha-stable-ish extreme).
  The classifier **owns the bounded test**: finite-support scipy family
  (`_BOUNDED_SCIPY_SEVS`), histogram / fixed atom, or a finite layer `exp_limit`
  / splice `sev_ub` ⇒ `BOUNDED` — lifted out of the current `Severity.bounded`.
- **Aggregate combine rule**: `agg_rung = max(freq_rung, sev_rung)`. Exact under
  the cases we support: when severity is subexponential or heavier (rung ≥ 3) the
  single-big-jump principle gives `P(S>x) ~ E[N]·P(X>x)`, so the aggregate
  inherits the severity class (and `max` returns it, since light frequencies are
  rung ≤ 2); when severity is light, the compound tail is set by the heavier of
  severity-decay and frequency-decay (`max` again). Caveat: valid while the
  frequency pgf is analytic at 1 (all standard freqs); genuinely heavy mixing is
  the watch item. Aggregate `log_concave` is **not** inherited from a random sum
  → set it from the numeric test on `agg_density`.
- **Portfolio aggregate**: `max` over unit aggregate rungs — under independence
  the tail of a sum is governed by the thickest summand (subexponential closure;
  and for the light/exponential rungs the convolution decay rate equals the
  slowest = thickest). Confirms the author's "worst-of" intuition.

## Engine

New module **`src/aggregate/tail.py`** (submodule access only, per CLAUDE.md).
The single source of truth — the bounded tables (`_BOUNDED_FREQS`,
`_BOUNDED_SCIPY_SEVS`) move here from `distributions.py`.

- `class TailClass(IntEnum)` — the 5 rungs; `__str__` → human labels
  ("bounded", "super-exponential", …); ordered so `max()` works.
- `FREQ_TAIL: dict[str, tuple[TailClass, bool]]` and a param-aware
  `classify_frequency(frequency) -> (TailClass, bool)`.
- `SCIPY_SEV_TAIL` table + `classify_severity(severity) -> (TailClass, bool)`:
  honours a `_certified_bounded` override first (→ `BOUNDED`), then the bounded
  test (finite family / atom / finite limit / splice ub → `BOUNDED`), then family
  lookup (param-aware via `sev_a` / `sev_b`), then numeric fallback on
  `severity.sev_density`.
- `combine(freq_cls, sev_cls) -> TailClass` (the `max` rule).
- `estimate_tail_from_density(xs, S) -> (TailClass, bool, dict)` — numeric
  fallback using classic tail diagnostics on the **reliable** tail region (90th
  – ~(1−1e−8) pctile, excluding fp-noise / deficit buckets):
  - **mean-excess function** slope — constant → exponential; linear-increasing →
    power-law (slope → α); → ∞ sublinearly → subexponential; → 0 / bounded →
    super-exponential / bounded;
  - **log-log S slope** → α for power-law confirmation;
  - **discrete log-concavity** test `p_k² ≥ p_{k-1}·p_{k+1}` over the bulk.
- `describe_lines(...)` and `explain(...)` text builders.

## API additions

`Aggregate`:
- `.tail_class` (property) → `(freq, sev, agg)` `TailClass` values via the
  classifiers + `combine`; the authoritative computation.
- `.bounded` **rewritten as derived**: `return self.tail_class.agg ==
  TailClass.BOUNDED`. The setter keeps its current contract — it sets the
  `_certified_bounded` override, which the classifier now consumes — so existing
  certify behaviour and the lifted-NA guard (`portfolio.py:2636`) are unchanged.
  `Severity.bounded` likewise derives from `classify_severity`.
- `.tail_description` (property) → 3 lines, e.g.
  `frequency   log-concave, super-exponential (poisson)` /
  `severity    subexponential (lognorm)` /
  `aggregate   subexponential`.
- `.tail_explanation` (property) → sentence, e.g. "Log-concave (super-
  exponential) Poisson frequency × subexponential lognormal severity ⇒
  subexponential aggregate (single big jump: P(S>x) ≈ E[N]·P(X>x))."
- `.info` (`distributions.py:2622`): append the 3 `tail_description` lines.

`Portfolio`:
- `.tail_class` → worst-of unit aggregate `TailClass` (`max`).
- `.bounded` **rewritten as derived**: `tail_class == BOUNDED` (keeping the
  `_certified_bounded` override and setter as today).
- `.tail_description` → the worst-of aggregate line.
- `.tail_explanation` → per-unit aggregate classes + named driver unit(s).
- `.info` (`portfolio.py:891`): append the aggregate tail line.

## Files

- `src/aggregate/tail.py` — new: enum, **the bounded tables moved here**,
  classifiers, combine rule, numeric estimator, text builders.
- `src/aggregate/distributions.py` — add `Aggregate.tail_class`,
  `.tail_description`, `.tail_explanation`; one line into `info`. **Rewrite**
  `Aggregate.bounded` (`:1680`) and `Severity.bounded` (`:5342`) as derived
  views of the classifier (keep the `_certified_bounded` override + setters);
  remove `_BOUNDED_FREQS` / `_BOUNDED_SCIPY_SEVS` (now in `tail.py`, import if
  still referenced). Reuse `frequency.freq_name`, `self.sevs` / `Severity`,
  `sev_density`, `agg_density` / `density_df`.
- `src/aggregate/portfolio.py` — add `Portfolio.tail_class`,
  `.tail_description`, `.tail_explanation`; one line into `info`. **Rewrite**
  `Portfolio.bounded` (`:850`) as derived (worst-of `tail_class`), keeping the
  certify setter. Reuse `agg_list`.

## Verification

- New `tests/test_tail.py` covering representative builds:
  `poisson × lognorm → subexponential`; `poisson × gamma → exponential`;
  `poisson × pareto → power-law` (and α reported, infinite-variance flag for
  α<2); `fixed × uniform → bounded`; `poisson × uniform (bounded sev) →
  super-exponential` (frequency-driven); `negbin × expon → exponential`; a
  `dhistogram` / spliced severity exercising the numeric fallback; `weibull_min`
  with c>1 / c=1 / c<1 spanning three rungs.
- Portfolio: a 2-unit port (thin + heavy) → `tail_description` == heavy unit's
  class; driver named; `bounded` honoured.
- **Derived-bounded equivalence**: for every case assert `obj.bounded ==
  (obj.tail_class.agg == BOUNDED)` on Aggregate/Portfolio and `sev.bounded ==
  (classify_severity(sev) == BOUNDED)`; assert the existing `bounded`-dependent
  tests / lifted-NA guard still pass; assert the certify setter (`bounded =
  True`) flips both `bounded` and the reported tail class to BOUNDED.
- Assert `TailClass` ordering and that `max()` reproduces the combine rule.
- `.info` smoke for an Aggregate and a Portfolio.
- `uv run pytest` green (`UV_LINK_MODE=copy`); sync any DecL into
  `src/aggregate/agg/test_decl.agg`.

## Close-out

- Bump `pyproject.toml` to the next `1.0.0a*`.
- README.rst bullet: tail-thickness classification (`tail_description` /
  `tail_explanation`, `.info` integration, Portfolio worst-of; `bounded` now
  derived from the classifier). Lifted-NA admissibility ties to `.bounded`.

## Open / watch

- Heavy **mixed-Poisson** frequencies (PIG / Sichel / Neyman-A): the default
  `EXPONENTIAL` is provisional; the numeric fallback or a literature-backed table
  refinement can promote them. Flagged, not blocking.
- Numeric fallback is tail-region sensitive — guard against fp-noise / aggregate
  deficit buckets (reuse the reliable-tail window logic already used for TVaR /
  validation), and note that lognormal-vs-Pareto separation needs a long grid.
- α estimate for power-law is best-effort from parameters when known, numeric
  otherwise.
