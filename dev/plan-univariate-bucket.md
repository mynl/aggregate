# Plan univariate-bucket (1A-bucket) — tail intelligence, usage, and reporting

> **What this is.** The next univariate bucket-selection plan, successor to the
> 1A / 1A-fix work that landed in `plan-bucket-window-2.md` (a58/a59). Where 1A
> built the *mechanism* (windowed placement, single-big-jump floor), this plan
> builds the *intelligence and the explanation*: a first-class thick/thin tail
> report for frequency, severity, and aggregate; wiring it into `bs`/`x_min`
> selection; and making the decision legible (a curated public `bs_window_df`
> and narrative `bs_description` / `bs_explanation`).
>
> **Naming.** Tasks are named in brackets — `[tail-report]`, `[tail-narrative]`,
> `[use-selection]`, `[bs-reporting]` — and the round-2 review items likewise
> (`[freq-tail]`, `[thick-thin]`, …). The braces mark a trackable unit.
>
> **This plan owns `dev/bucket-selection.rst`** — every task that changes
> behaviour or surfaces keeps that document current (it is destined for
> `docs/5_technical_guides/`).
>
> **Relationship to `plan-bucket-window-2.md`.** That plan's 1A (the `Aggregate`
> sizer, a58) and 1A-fix (the single-big-jump floor, a59) have landed. Its **1P**
> (the `Portfolio` windowed combine) is *still open there*; we return to it after
> this plan, and it inherits this plan's reporting surfaces (`tail_df`,
> `tail_description` / `tail_explanation`, `bs_window_df`, `bs_description` /
> `bs_explanation`) at the `Portfolio` level.
>
> Tasks land one at a time, each bumping `1.0.0a*`, updating `CHANGELOG.md` /
> `dev/TODO.md`, and keeping `bucket-selection.rst` current.

---

## Round-2 input — issues, decisions, status

Captured from the author's 2026-06-17 reviews; each row tracked to closure.

| task | issue | decision | status |
|---|---|---|---|
| **`[recommend-bucket]`** | `recommend_bucket` is a legacy one-shot sizer. | **Replace** (not just reconcile) with a new sizer taking `log2` (and possibly `x_min`) as explicit args, then retire `recommend_bucket`. | Tracked — `dev/TODO.md` **W10** + §Also-noticed. |
| **`[freq-tail]`** | Frequency needs its own tail report. | Yes — first-class layer (`min` can be `1` for zero-truncated; heavy mixing can *drive* the agg tail). | **LANDED (a60)** — `frequency` row in `tail_df`. |
| **`[thick-thin]`** | Stop thinking "left / right tail" as separate objects. | Thickness is the property; each side is **thick / thin**. The bs decision basis is `{left thick/thin, right thick/thin, concentrated}`. Finer rung / `alpha` kept underneath only for quantile **magnitude**. **Thick ⇔ subexponential-or-heavier** (lognorm, power-law); thin ⇔ exponential-or-lighter. | **LANDED (a60)** — `is_thick` / `thickness_label`, `left` / `right` columns. |
| **`[splice-exposure]`** | Splice vs limit/attach. | **Severity owns both** splicing *and* `limit xs attach` (confirmed: `comp.limit` / `comp.attachment` live on `Severity`). So there is **no separate Exposure layer** — limit/attach is just part of each component severity, and blending the mix components is the "exposure rationalization". **Build order: mix components → combined effective severity → aggregate → bucket** (one FFT on the blended sev, *not* per-component aggregates), because the sizer is tail-aggressive: a small but thick component still asserts itself in the blend. The report is still a **DataFrame with a `component` column**. Empirical sevs are bounded (easy). | **LANDED (a60)** — per-`comp` rows + combined `severity` row; occ-re row deferred to `[use-selection]`. |
| **`[signed-padding]`** | Is FFT padding "in the middle" for a signed sev? | Code read (`_freq_sev_convolution`): positives at the bottom of the period, negatives wrapped to the top, `M−N` zeros **between** — already "in the middle". | **Open (verify)** — `[use-selection]` §padding; targeted test + sufficiency check when both tails heavy. |
| **`[clip-warning]`** | The clip message is a silent `logger.info`. | Promote to a visible **warning** + structured clipped-mass field. **Do it with the rest** of `[use-selection]`, not as an early standalone. | **Folded** — `[use-selection]`. |
| **`[reins-gross]`** | Reinsurance and the window. | **Size on GROSS, ignore reinsurance.** Aggregate re ignored; occ-re is a reported overlay row, never a sizing input. Gross is needed to report re impact anyway. | **Partly landed (a60)** — `tail_df` sizes/reports on gross; the **occ-re overlay row** is deferred to `[use-selection]` (lands with the gross-sizing wiring, where it has a sizing decision to annotate). |

Resolved open questions this round: concentration cutoff, thick/thin cut,
Exposure-vs-Severity structure, tail-aware slack formula (see each task).

---

## Why now

Grid choice is the single most consequential numerical decision in the library,
and it is subtle. Today the logic is correct but **opaque** — it reads only the
`bounded` bit of the rich `aggregate.tail` classifier, and its only window is the
private `_bs_window_df`. Two consequences:

1. **The sizer is under-informed.** The `sbj` floor reimplements the
   single-big-jump principle `tail.py` already *names*, unwired; power-law sevs
   fall back to `recommend_bucket`; the windowed left-lift is taken on faith
   rather than on a thin-left determination.
2. **The user is under-served.** A narrative of *why this grid* is, in the
   author's words, "a major selling point", sharing its technique with the
   forthcoming validation report.

This plan closes both.

---

## `[tail-report]` — the layered thick/thin tail report

> **LANDED (a60, schema revised a61).** `Aggregate.tail_df` (spec-only), the
> `TailRow` / `build_tail_rows` / `tail_frame` machinery in `aggregate.tail`, the
> thick/thin cut (`is_thick`), claim-space `severity_support`, structural
> aggregate support (`_agg_support`, exact for bounded books, affine-mapped for
> `pnl`), and the conservative `concentration` (`CONCENTRATION_CV = 0.1`) all
> shipped, byte-stable, with ~14 tests in `tests/test_tail.py`.
>
> **a61 schema (author review):** the report is **support + tail shape**, nothing
> from grid selection. `min`/`max` are the **structural support** (attainable
> bounds, `±inf` at an open end), not a reach estimate; `bounded` is derived
> `min`/`max`-finite; `left_tail`/`right_tail` carry the **full per-side tail
> class** (`bounded` at a finite end, else the family decay rung), not thick/thin;
> `concentration_p = Φ(mean/sd)` (P band clears 0). Dropped `tail_class`,
> `alpha`, `log_concave` columns (the power-law fact lives in `note`). The reach
> estimate moves to `bs_window_df`.
>
> **Deferred to `[use-selection]`:** the occurrence-reinsurance overlay row
> (layer 4 below) — it reports a *net* impact with no sizing consequence, so it
> lands with the gross-sizing wiring rather than as a standalone reported row
> now. The `[q-left-combine]` rule is resolved in `build_tail_rows` (positive ⇒
> `bounded` left; signed `ssev`/`dsev` ⇒ reflected severity left; affine `pnl` ⇒
> loss right tail, premium-capped right).

### Layered model (`[freq-tail]`, `[splice-exposure]`, `[reins-gross]`)

Tail shape is built **bottom-up**, spec-only (pre-`update`, like today's
`tail_class`). A `mixed` book has several severity *mix components*; the report
tracks each. Motivating example (must handle):

```
agg TAILTEST [20 30 40] claims [inf inf 1000] xs [100 0 0]
    sev [gamma lognorm lognorm] [100 100 100] cv [1 3 1.3] mixed gamma .5
```

Verified build: 3 components — `comp0` gamma (`inf xs 100`, unbounded), `comp1`
lognorm cv 3 (`inf xs 0`, unbounded → thick right), `comp2` lognorm cv 1.3
(`1000 xs 0`, **bounded** by its limit); `mixed gamma .5` frequency. Layers:

1. **Frequency.** Own row: `min` (`0`, or `1` zero-truncated, or a `dfreq`/
   `fixed` atom), `max` (`inf`/bounded), thin left (counts `>= 0`), right
   thickness from the family. **Heavy mixing matters** (`mixed gamma`, negbin,
   PIG, Sichel, Neyman-A): heavier-than-Poisson right tails can *drive* the agg
   (`driver = 'frequency'`).
2. **Severity, per mix component.** From the `Severity` object, which owns the
   distribution **and its layer** (`limit` / `attachment`) **and any splice**
   (a splice sets a lower bound, an upper bound, or both — part of what the sev
   *is*; reported as-is, **no** unspliced-vs-spliced split). A finite `limit`
   makes the component **bounded right** (`comp2`); an `attachment` truncates the
   left (`comp0`). Empirical/histogram severities are bounded.
3. **Combined effective severity.** The exposure-weighted blend of the
   components → the single severity that feeds the FFT. Right tail driven by the
   thickest component (`comp1`, subexponential), even though `comp2` is capped.
4. **Occurrence-reinsurance overlay (reported, not sized).** Occ re is one more
   overlay on the *combined* severity — its own row. Aggregate re is ignored.
   Per `[reins-gross]` the window is sized on the **gross** combined severity;
   the re row is reported so the user sees re's impact, never drives `bs`/`x_min`.
5. **Aggregate.** Frequency ⊗ gross combined severity → the agg row, plus the
   agg-only concentration fields.

So the report is a **DataFrame with a `component` column** (one row per component
at the severity layer, plus single rows for freq, combined sev, occ-re, agg).

**Build order (`[splice-exposure]`): components → combined sev → aggregate →
bucket.** We size on the *blended* aggregate, not on per-component aggregates
combined afterward — the sizer is deliberately tail-aggressive, so a small but
thick component still asserts itself in the blend and gets covered.

### Fields & decision basis (`[thick-thin]`, a61 schema)

Per row, structural support + per-side tail class, plus (agg only) concentration:

| field | meaning |
|---|---|
| `min` | smallest **attainable** value (`-inf` at an open left end). |
| `max` | largest **attainable** value (`inf` at an open right end). |
| `left_tail` | lower-tail class: `bounded` at a finite left end, else the family decay rung. |
| `right_tail` | upper-tail class: `bounded` at a finite right end, else the family decay rung. |
| `bounded` | derived: `min` and `max` both finite. |
| `concentrated` | agg only: a *conservative* "band clears 0" flag — see below. |
| `concentration_p` | agg only: `Φ(mean/sd)` = P(aggregate > 0) under a normal approx. |
| `note` | power-law `alpha` + failing moment, capped-base, driver. |

**The bs decision turns on the *thickness* of each side (`is_thick(right_tail)` /
`is_thick(left_tail)`, thick ⇔ subexponential-or-heavier) and `concentrated`:**

- *right thick* ⇒ MoM upper edge under-reaches ⇒ apply the `sbj` floor.
- *right thin* (incl. `bounded`) ⇒ MoM/normal upper quantile suffices.
- *left thin* (`bounded` / light) ⇒ safe to lift `x_min` (windowed left-lift trustworthy).
- *left thick* (signed / left-spliced / `pnl`) ⇒ `x_min` must cover the reach (correctness).
- *concentrated* ⇒ mass band clears 0 ⇒ windowed placement eligible.

(`alpha`, `log_concave`, and a single `tail_class` rung were dropped from the
frame — none drives selection; `alpha`/infinite-moments live in `note`, and the
narrative `tail_description` keeps the log-concavity prose.)

**Thick ⇔ subexponential-or-heavier** (subexponential *and* power-law; lognorm is
thick); **thin ⇔ exponential-or-lighter**. The finer rung (`… POWER_LAW` +
`alpha`) is retained underneath, used **only** for tail-quantile *magnitude* in
`[use-selection]` (e.g. power-law `q_X(p) ∝ (1-p)^{-1/alpha}`), never for the
structural choice.

**Concentration is conservative.** A book is `concentrated` only when its
`cv` is comfortably small — `cv < ~0.1` (the band sits ≳ 10 sd above 0), tighter
than the current `1/z ≈ 0.14`. Rationale: lifting `x_min` when the band does
*not* really clear 0 clips left-tail mass, so **err toward not-windowing when
marginal**. (Large-`E[N]` Poisson books have tiny `cv` and window comfortably;
the borderline `cv ∈ [0.10, 0.14]` books revert to the 0-based grid — a tracked
behaviour change, regression-checked.) `concentration_p` reports the margin
(e.g. `1/cv`, the sd-count the band clears 0) for transparency; it is not a
separate gate.

### Combine rules

- **Right (aggregate).** `agg_right = max(freq, sev)` (today's `combine`) — exact
  under single big jump; `alpha` propagates.
- **Left (aggregate).** *New.* Positive severity ⇒ thin (downward large
  deviation, no single big jump). Signed / left-spliced ⇒ mirror of the right
  argument. **`[q-left-combine]`** formalize + test.
- **Blend across components.** `min = min_i`, `max = max_i`,
  `right = thickest_i`, `left = thickest_left_i`, `bounded = all_i`. Mirrors
  `_combine_severities` (already does this for the right rung + `alpha`).

### Non-family severities — structural, not numeric (`[q-phase2]` resolved: no numerics)

**No numeric tail estimator.** It cannot drive the sizer anyway — estimating a
tail numerically needs a discretized grid, but the grid is the thing the tail
estimate is meant to *choose* (chicken-and-egg). So `tail.py` stays family-lookup
+ structure, and we **carry two facts, not one**:

- **base-family thickness** — the un-layered shape (e.g. a Lévy / stable base is
  thick), and
- **structural bound** — any finite `limit` / splice cap / attachment.

The **effective** classification (what the sizer uses) combines them: a thick
base **capped at `L`** is *effective-bounded* — cover `[0, L]`, finite and
correct — while the base heaviness still informs *resolution within* `[0, L]`.
So a limited Lévy sizes as bounded (right) without any numeric fit, and the
report shows both ("thick base, capped at L → effective bounded"). Histograms /
empirical are bounded; splices are owned by the severity (§above). The only
residual is a **genuinely unknown *and* unlimited** scipy family → treat
**conservatively as thick** (size wide); never guess.

> **Family tables populated (a63).** Rather than wait for cases to appear, the
> `aggregate.tail` family tables were filled out from a reconciled survey of
> *every* `scipy.stats` continuous distribution's tail behaviour (two independent
> derivations cross-checked → `integrated.md` in the 2026-06-17 notes folder).
> `SCIPY_SEV_TAIL` (~45 fixed-class families), `_POWER_LAW_ALPHA` (correct
> shape-slot α per family), the parameter-aware set (`gengamma`, `gennorm`,
> `dweibull`, `exponweib`, `tukeylambda`, `levy_stable`, …), `_SEV_LEFT_CLASS`
> (asymmetric two-sided left tails), and a `fz.support()`-finiteness fallback in
> `_severity_bounded` mean almost no standard family hits the `UNKNOWN`
> conservative-thick path now. Single source: `_weibull_shape` /
> `_family_right_class` / `_family_sides` feed both `classify_severity` and the
> `tail_df` per-side classes.

### Data structure

Extend `aggregate.tail`: a per-row `TailRow` (the five fields + rung/`alpha`) and
a builder assembling the layered DataFrame via `frequency_tail`,
`severity_tail` (per component + blend), `aggregate_tail`. Surface
the assembled frame as a public **`tail_df`** property (the `_df` convention,
parallel to `bs_window_df`) with a `component` column. Keep `TailInfo` /
`TailClass` / `alpha` as the right-tail core; `tail_df` composes them with
`min`/`max`/`left`/concentration. `Aggregate.tail_class` / `.bounded` stay as
derived views (back-compat).

---

## `[tail-narrative]` — the narrative tail report

> **LANDED (a62).** `tail_description` (short) / `tail_explanation` (verbose) now
> narrate the layered a61 `tail_df`, built from one shared
> `Aggregate._tail_rows()` so frame and prose never drift. `Severity` and
> `Frequency` carry one-line `tail_description` too. ANSI emphasis via
> `describe_rows`/`explain_rows(..., color=True)` (a self-contained bold-red on
> thick rungs — the DecL `format`-program colorizer lexes DecL, so it is not a
> fit for prose). `Portfolio` keeps its existing worst-of text (the layered
> port-level narrative is part of 1P). +7 narrative tests; byte-stable.

Per the house naming rule, **two** read-only text properties, extending today's
same-named ones to the layered content — **keep both** (`tail_description` is
short, `tail_explanation` verbose), on `Frequency`, `Severity`, `Aggregate` (and
`Portfolio` later):

- **`tail_description`** — the short, aligned summary (a62, per-side **tail
  class**, not thick/thin):

  ```
  frequency tail   poisson, count [0, inf), super-exponential right tail
  severity tail    lognorm, [0, inf), subexponential right tail
  aggregate tail   [0, inf), subexponential right tail; not concentrated (P>0=1.00)
  ```

- **`tail_explanation`** — the verbose prose (per-component breakdown, *why* the
  aggregate inherits the severity right tail, the single-big-jump sentence, the
  heavy-left signed/`pnl` note, the driver, the concentration).

Notes: the **ANSI colour/emphasis option** is a small self-contained rung
emphasis (`color=True`), not the DecL `format`-program colorizer (which lexes
DecL syntax). How much enters the default `qd`/`info` need not be decided now
(today's `tail_description` already feeds `info()`).

---

## `[use-selection]` — using the report in bs selection

> **LANDED (a64).** All six items wired into `_bs_window`, plus the occ-re
> overlay row (`occ_net_severity_row`) and the signed-padding verification
> (`test_signed_two_sided_reach_no_collision`). The "Regime B" heavy-severity
> book is now reclaimed by the thin-left-gated upper floor (item 3); the old
> `test_window_regime_b_*` is replaced by `test_window_heavy_severity_reclaimed_via_sbj_floor`,
> and the convention-mirror test split into symmetric (mirrors) /asymmetric
> (skews to thick) cases. Borderline `cv ∈ [0.10, 0.21]` books revert to the
> 0-based grid (item 5). Next: `[bs-reporting]`.

Each item is a self-contained change to `_bs_window`, byte-stability-gated
(full suite + `test_suite.agg` snapshot; light / thin / bounded / windowed books
unchanged). Sized on the **gross combined severity** (`[reins-gross]`).

1. **Gate `sbj` on thickness.** Apply the single-big-jump floor only for a
   **thick** right tail (signed: thick left). A no-op for thin tails today, but
   explicit, cheaper, and it stops the floor firing where the MoM window is
   already exact.
2. **Power-law / infinite-variance: honest truncation, no normalization**
   (`[q-powerlaw-quantile]` resolved). A power-law tail has no finite deep
   quantile to size to, so do **not** chase it (no `alpha`-quantile sizing, no
   `recommend_bucket` fudge). Size the *reachable bulk* well, then **accept the
   truncation**: every aggregate point at `p` below the truncation is **exact**;
   the missing tail is left as a visible **deficit** — **we do NOT normalize**
   (the shortfall is reported, never smeared back in) — and a **warning** is
   issued. This is the existing `DefectiveDistribution` / `normalize=False`
   philosophy applied deliberately. (`alpha` is still reported in the tail
   report; it just doesn't drive sizing.)
3. **Thin-left-gated windowed left-lift (the asymmetric window).** The headline
   case (`T5`, target `[~400k, ~1.2M]`): floor the `windowed` upper edge by
   `sbj_hi`, **gate the `x_min` lift on a thin-left determination** from the
   report (only trust lifting `x_min` when the left tail is genuinely thin), and
   relax the selection gate so a windowed grid wins when it captures a reach the
   0-based pick clips (not only when no coarser). The `windowed`↔`sbj`
   unification flagged at the end of `plan-bucket-window-2.md`.
4. **Tail-aware padding / slack (`[q-slack]` resolved).** Replace the fixed
   `window_pad_skew` split with:
   - **Either edge hard-bounded** (a `pnl` premium = hard upper; bounded support;
     exact-discrete edge) ⇒ **snap to the hard bound** — but to a *clean* grid
     value: round the bound up (out) to the nearest wholesome binary `bs`-multiple
     via `round_bucket`, never a silly number like `13.6`, `14.1`, or `1/7`. Put
     all slack on the soft side.
   - **Symmetric tails** (thick/thick or thin/thin) ⇒ **centre** the band (split
     slack evenly).
   - **Asymmetric** (thick/thin) ⇒ a hardwired **≈3/4 toward the thick side,
     1/4 the thin side** (the "right look").
   The loss/payoff convention skew is demoted to a **tie-breaker**, used only
   when the tails are symmetric.
5. **Concentration from the report.** Replace the inline `cv < 1/z` gate with the
   conservative `concentrated` field (`cv < ~0.1`, `[tail-report]`) — the single
   source of truth; note the borderline-book regression above.
6. **Clip → warning (`[clip-warning]`).** Promote the positive-tail clip
   `logger.info` to a visible `warnings.warn` (a `DefectiveDistribution`-style
   category) reporting the reach and the `log2` that would capture it, and expose
   the clipped-mass estimate as a structured field for the validation report and
   `bs_explanation`. Lands **with** this task, not early.

### Padding placement for signed books (`[signed-padding]`)

Strictly FFT-internals, tracked per the author. Non-negative book: grid
`[0, N*bs]`, `x_max` implicit, the `M = N << padding` zero-pad sits *above* the
support — correct. Signed book: the support straddles 0, so the pad should sit
"in the middle". A read of `_freq_sev_convolution` says it **already does**
(positives at indices `0…`, negatives at `M-i0…M-1`, padding zeros between). The
deliverable is a **verification** — a targeted test that a signed book's two
aggregate tails do not collide in the buffer, plus a padding-*sufficiency* check
when both tails are heavy (else widen `padding` or warn) — not a fix unless it
disproves the read.

---

## `[bs-reporting]` — make the choice legible

1. **`_bs_window_df` — the complete decision journey.** Keep it private and add
   *as many columns/rows as needed* to record the journey, not just the outcome:
   why a method did/didn't apply (the severity-fit reason, not just `False`), how
   far the `sbj` floor bound, the clipped-mass estimate, a compact tail-report
   summary, `log2` budget vs need.
2. **`bs_window_df` — curated public view.** A read-only property culling
   `_bs_window_df` to the user-facing columns (method, window, grid, applies,
   selected, one-line note), on both `Aggregate` and **`Portfolio`** (folds in
   TODO **H10**; keep `_bs_window_df` for experts).
3. **`bs_description` / `bs_explanation` — the narrative.** Per the house naming
   rule, the short summary (`bs_description` — the chosen grid in a line: method,
   `(bs, log2, x_min)`, any clip) and the verbose prose (`bs_explanation` — what
   the book is from the tail narrative, which methods ran, why the winner won,
   what is clipped and how to widen it). ANSI option as in `[tail-narrative]`.
   The "#1 decision, made legible" surface, sharing its helper with the upcoming
   validation report.

---

## Open questions (remaining)

- *(none)* — `[q-left-combine]` resolved in a60 (`_aggregate_left_tail`):
  positive ⇒ thin; signed `ssev`/`dsev` ⇒ reflected combined-severity left;
  affine `pnl` ⇒ the loss's right tail. Tested by
  `test_tail_df_signed_reach_is_two_sided`.

**Resolved this round:** narrative naming (`tail_description` /
`tail_explanation` kept, both; `bs_description` / `bs_explanation`; structured
`tail_df`); `info`/`qd` surface (decide later, not now); concentration cutoff
(`cv < ~0.1`, conservative); thick/thin cut (thick ⇔ subexponential-or-heavier);
Exposure-vs-Severity (Severity owns limit/attach + splice — no separate Exposure
layer); build order (components → combined sev → agg → bucket); slack formula
(hard-bound-rounded / centre / 3-1); **`[q-phase2]`** — *no numeric estimator*
(family tables + base-class/structural-bound; conservative-thick for a genuinely
unknown-and-unlimited family; the chicken-and-egg with grid sizing rules numerics
out anyway); **`[q-powerlaw-quantile]`** — *honest truncation, no normalization,
warn* (exact below the truncation, deficit reported, never smeared).

---

## Also noticed (lower priority)

- **`[recommend-bucket]` replacement.** Eventually *replaced* (not reconciled) by
  a new sizer taking `log2` (and possibly `x_min`) as explicit args. Its last job
  — the infinite-variance fallback — is removed by `[use-selection]` item 2
  (honest truncation + no-normalize + warn handles infinite variance directly),
  clearing the way. Tracked in `dev/TODO.md` (**W10**).
- **Where the narratives live.** `tail_description` / `tail_explanation` /
  `bs_description` / `bs_explanation` integration with `info()` / `qd` /
  `_repr_html_`, and a shared narrative helper with the forthcoming validation
  report.
- **Performance / caching.** The tail report is spec-only; cache and invalidate
  on spec change, like the other pre-`update` derived views.

---

## Sequencing

1. **`[tail-report]`** — the layered structure + combine rules + thick/thin +
   conservative concentration. Pure addition (no selection change) → byte-stable.
2. **`[tail-narrative]`** — `tail_description` / `tail_explanation` text + ANSI.
   Pure addition.
3. **`[use-selection]`** — wire into selection, item by item. The asymmetric
   window (3), tail-aware slack (4), the concentration tightening (5), and the
   clip warning (6) are the behaviour-changing ones; ship with the fullest
   regression net.
4. **`[bs-reporting]`** — `_bs_window_df` journey, public `bs_window_df`,
   `bs_description` / `bs_explanation`. Can interleave with `[use-selection]` for
   the journey columns it produces.

Then return to `plan-bucket-window-2.md` **1P** (Portfolio combine), which
consumes `[tail-report]` / `[bs-reporting]` at the portfolio level.
