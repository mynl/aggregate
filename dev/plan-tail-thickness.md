# tail-thickness — ordered tail classification for Aggregate & Portfolio

New feature; bumps the `a*` version. Standalone (no dependency on the reins
plans).

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
