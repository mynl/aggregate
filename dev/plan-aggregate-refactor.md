# Plan: Aggregate pipeline refactor

> Status: **draft for discussion**, not yet started. Captures the design
> discussion of 2026-05-28. No code changed yet. Companion to the
> current-state description in `dev/pipeline-aggregate.rst`. Move to
> `dev/done/` when complete.
>
> **Splitting advice (read first).** This plan is large but the items are
> interdependent — the reinsurance reporting redesign drives the `stats_df`
> schema, which drives `valid`, which is why `stats_df` goes all-float. So keep
> it as **one plan** for now. The two natural carve-outs, *if* we decide to
> stage the work, are **Group 1 (reins reporting)** — the largest, most
> self-contained piece — and **Group 4 (so/po parsing)** — purely a
> parser/grammar concern. Everything in Groups 2–3 is small, local hygiene that
> can land incrementally.

---

## 1. Context & rationale (the discussion)

### 1.1 The reinsurance reporting problem

When reinsurance is present the object is **after-reinsurance everywhere except
its theoretical moments**. "After reinsurance" is *not* a fixed gross/ceded/net
(gcn) triple — a clause can pass along **either the ceded or the net** (`net of`
vs `ceded to`), and the stages can **mix kinds** (e.g. net-of at the occurrence
stage and ceded-to at the aggregate stage; reinsurance is very flexible). So the
realised density is simply "whatever the clauses pass along". (`so`/`po` clause
parsing is its own item — Group 4.)

Concretely: `agg_density`, `q`, `tvar`, `sf`, `price`, the whole `density_df` are
all after-reinsurance, but the `mixed`/theoretical column of `stats_df` — and the
`agg_m/cv/skew`, `sev_m/cv/skew` scalars derived from it — stay **subject**
(pre-reinsurance / gross). So `describe` pairs a subject theoretical column
against an after-reinsurance empirical column and reports an `error` that is
really the reinsurance impact masquerading as numerical error. The display is
"plain wrong".

### 1.2 Why we cannot just net the theoretical moments — the snake eats its tail

- **After-occurrence severity (net or ceded) is an independent analytic
  object.** `Y = f(X)` for the piecewise-linear netter/ceder `f`, and
  `E[Y^k] = Σ_c w_c ∫ f(x)^k f_c(x) dx` — a 1-D integral against the known
  severity, summed over mixture components, frequency unchanged. Independent of
  the FFT ⇒ *can* validate.
- **Aggregate net has no independent theoretical, by construction.**
  `E[A_net^k] = ∫ (a − c(a))^k dF_A(a)` needs `F_A`, and the only `F_A` we have is
  the FFT output. Any "theoretical" net-agg moment is therefore identical to the
  empirical net-agg moment computed from that same FFT density. There is nothing
  independent to validate it against.

### 1.3 The resolution: validate subject, report after-reinsurance

This **matches the documented workflow** ("build gross/subject → validate → build
the reinsured view, trusting it"). The current `REFACTOR` behaviour
(`valid → REINSURANCE`, do nothing) actually breaks that promise — there is no
machine check that the subject object validated. So "make sure subject validates"
restores intended behaviour.

- **Validation** runs on the **subject** (subject theoretical `mixed` vs subject
  empirical FFT) — always apples-to-apples, independent of the cession.
- **Reporting** shows the **subject → after-occ → after-agg** progression plus
  the **`Change`** of whatever is passed along, as a % difference from subject.
  The reinsurance step itself is a deterministic transform: it needs reporting,
  not statistical validation.

---

## 2. Decisions: made vs open

| # | Item | Decision |
|---|------|----------|
| D1 | `describe`: keep the 8-column shape, relabel to Subject / Net-or-Ceded-or-After / Change | **decided** |
| D2 | 3rd & 6th columns become **`Change`** = `(after − subject)/subject` (% change, raw fraction — same arithmetic as the old `Err`) | **decided** |
| D3 | `stats_df`: drop the `('meta','name')` row so the frame is all-float | **decided** |
| D4 | `stats_df`: keep `comp_` and `independent` (internal SSoT, completeness over width) | **decided** |
| D5 | `stats_df`: add `after_occ` / `after_agg` staged columns + per-stage impact | **decided (shape TBD in impl)** |
| D6 | `valid` reads **only** from `stats_df` (single source of truth), incl. mean/aliasing | **decided** |
| D7 | Magic numbers in `valid` (`eps**3`, `10×`) → `VALIDATION_NOISE`-based constants | **decided** |
| D8 | Portfolio empirical moments adopt **Aggregate's** `xsden_to_mwrangler` (de-fuzzed) convention | **decided** |
| D9 | Delete `aggregate_keys` (dead) | **decided** |
| D10 | Delete journey-of-discovery comments (pith preserved in the .rst) | **decided** |
| D11 | First/theoretical column is named **`Subject`** (the reins term for the book a treaty applies to); display label only — internal column stays `mixed` | **decided** (was O1) |
| D12 | Validate the **subject** under the hood — condition met: ≤ one FFT of the already-computed gross severity (occ case), zero extra (agg-only) | **decided** (was O2) |
| D13 | Remove the redundant `est_*` writes inside `apply_occ/agg_reins` (final `est_*` from `update_work` are net; the reins-stage writes are dead) | **decided** (was O3) |
| D14 | **Leave `update` / `update_work` as-is** — the `xs` argument exists so `Portfolio.update` can drive every unit onto one shared grid | **decided** (was O4) |
| D15 | `xsden_to_meancv` / `meancvskew` tail-mass — **closed**, resolved by the `moments.py` consolidation onto the shared `xsden_to_mwrangler` worker | **done** (was O6) |
| D16 | **Name `stats_df` component columns `e{e}.m{m}`** (exposure component × severity-mixture component), replacing the flat `comp_n` | **decided** (wire in during the `__init__` refactor) |
| O5 | Reconcile forwards/backwards `S` conventions (incl. `Distortion.price` backwards) | **open (investigate)** |

---

## 3. Work items

### Group 1 — Reinsurance reporting (the through-line)

**1.1 `describe`: relabel, same 8-column shape.** When reinsurance is present,
the moment table becomes an *economic* view instead of a *validation* view.
Headings use the denser `EX` / `CV` / `Sk` form (not `E[X]` / `CV(X)` /
`Skew(X)`):

```
 Subject EX | Net EX | Change  |  Subject CV | Net CV | Change  |  Subject Sk | Net Sk
```

- col 1 (sourced from `mixed`) → **`Subject`** (pre-reinsurance / gross
  theoretical).
- col 2 (sourced from `empirical`) → the after-reinsurance value. Header is
  **`Net`** or **`Ceded`** when the object is consistently one kind (clear), else
  **`After`** (when occ/agg mix kinds, e.g. net-of-occ then ceded-to-agg). The
  final object is one density; per-stage detail lives in `stats_df`.
- col 3 (the old `Err` slot) → **`Change`** = `(after − subject) / subject`, a
  **% change displayed as a raw fraction** (e.g. `-0.30`, *not* formatted as a
  percent). Continuity with the old view: `Change` is the same arithmetic as the
  old `Err` relative-error column `(new − old)/old` — with no reinsurance it is
  ≈0 (the validation eyeball); with reinsurance it is the reinsurance impact
  (e.g. `-0.30` = the reinsured view sits 30% below subject). So `describe` keeps
  one arithmetic, two readings.
- Skew stays 2-column (`Subject Sk | Net Sk`) as today (skew estimate is noisy);
  a `Change` column for skew is optional, not required.
- Files: `Aggregate.describe` (`distributions.py`). Source everything from
  `stats_df` (see 1.2). The non-reins path keeps the theoretical/empirical/error
  meaning but adopts the denser `EX`/`CV`/`Sk` headings for consistency — note
  this touches any doctests that match `E[X]` etc., so sweep them.

**1.2 `stats_df`: all-float, staged columns.**

- **Drop the `('meta','name')` row.** It is the only non-numeric cell; the name
  already lives on `self.name`. Removing it makes `stats_df` a `float64` frame
  and deletes every `.astype(float)` cast scattered across `__init__`,
  `update_work`, and `portfolio.py`. (D3.)
- **Keep `comp_*` and `independent`** — internal SSoT; completeness beats width
  here. (D4.)
- **Add staged columns** reflecting the pipeline `Subject → after occ → after
  agg`:
  - `mixed` stays = subject (gross) theoretical. `independent` unchanged.
  - `after_occ` = empirical moments after the occurrence stage (present only
    when occ reins exists).
  - `empirical` = the **final** object moments (== after_agg). Keep the name
    `empirical` for the final realised view.
  - per-stage **impact** columns (ratios relative to the prior position):
    `occ_impact = after_occ / mixed`, `agg_impact = empirical / after_occ`.
  - `error` repurposed under reins to the **gross** validation
    (`gross_empirical` vs `mixed`); with no reins, `gross_empirical == empirical`
    so `error` is exactly today's theoretical-vs-empirical column. Degenerates
    cleanly.
  - `gross_empirical` column appears only if O2 (validate-gross) is adopted.
- The exact column set is finalised during implementation; the staging *shape*
  (subject → after_occ → after_agg + impacts) is the decided part. (D5.)
- Files: `_init_stats_df`, `_STATS_ROW_INDEX` (drop name row), `update_work`
  (write staged empirical), `describe`, and `portfolio.py` readers (which do
  `astype(float)` and read `('meta','name')` — sweep both).

**1.3 Validate the subject under the hood. [D12 — decided]**
Even under reins, compute the **subject** (gross) empirical agg and validate it
against `mixed`. Cost is acceptable — at most one FFT, on a density we already
have:
- agg-only reins: `agg_density_gross` already *is* the subject agg (pre-agg-reins
  FFT output) — **zero** extra FFT, just take its moments.
- occ reins (± agg): one FFT of the already-computed `sev_density_gross` → subject
  agg. `reinsurance_df` already does exactly this (`p_agg_gross_occ`) — reuse it.
Keep the public `valid = REINSURANCE` / "n/a" for the reinsured object (the net
agg genuinely cannot be independently validated), but surface "subject validates:
yes/no" in `explain_validation` / `info`.

**1.4 Naming: the theoretical column is `Subject`. [D11 — decided]**
"Subject" is the precise reinsurance term — the subject premium/loss is the book
a treaty applies to. Use it as the display label in `describe` and in docs. The
internal `stats_df` column can stay `mixed` (it is referenced widely); `Subject`
is the display name.

### Group 2 — Validation & stats hygiene

**2.1 `valid` reads only from `stats_df` (SSoT). [D6]**
Today mean/aliasing read from `describe` (already snapped/noise-adjusted) while
CV/skew read from `stats_df`. Move *all* reads to `stats_df`; `describe` becomes
purely a display concern. Files: `Aggregate.valid`, `Portfolio.valid`
(mirror the change).

**2.2 Magic numbers → named constants. [D7]**
The ALIASING test's `eps**3` floor and `10×` ratio are heuristics fitted to old
failing docs examples. Re-express via `VALIDATION_NOISE` and a named ratio
constant — consistent with the de-fuzzing philosophy already applied to
`agg_density`. Files: `valid` (both classes), `constants.py`.

**2.3 All-float `stats_df`.** See 1.2 (D3) — listed here too because it removes
the `dtype=object` casts that this group's validation code currently pays.

**2.4 Portfolio empirical-moment convention. [D8]**
`Portfolio.update` uses a plain `Σ p·xᵏ` "to match the PEG baseline" and does
**not** de-fuzz; `Aggregate.update_work` uses `xsden_to_mwrangler` on a de-fuzzed
copy. **Aggregate is the way to go** — make Portfolio match (de-fuzzed copy +
`xsden_to_mwrangler`). This will move the PEG baseline numbers slightly; update
the baseline deliberately in the same commit (see harness). Files:
`Portfolio.update`, `create_from_sample` (the duplicated moment block).

**2.5 Tail-mass consistency. [D15 — done]** `xsden_to_meancv` /
`xsden_to_meancvskew` / `xsden_to_noncentral` now all delegate to the shared
`xsden_to_mwrangler` worker, so they share one tail-mass convention — the old
inconsistency is **resolved**. Any remaining reconciliation with the
`density_df.lev` left-Riemann tail is folded into O5.

### Group 3 — Pipeline structure

**3.1 Leave `update` / `update_work` as-is. [D14 — decided]**
`update_work(xs, …)` takes `xs` **not** because of a 2016 mistake but because
`Portfolio.update` calls `agg.update_work(xs, …)` to force every unit onto **one
shared grid**. That coupling justifies both the `xs` argument and the two-method
split: `update(log2, bs)` is the standalone convenience that *builds* `xs`;
`update_work(xs)` is the grid-driven worker the Portfolio reuses. **No change** —
the split earns its keep. (Recorded because it was raised as a removal candidate;
the decision is to keep it.)

**3.2 Extract the FFT convolution helper. [decided]**
`reinsurance_df` re-implements the `ft → PGF → ift` core (including the fixed-1
shortcut) to recompute gross/ceded/net aggregates. Factor the 5-line core into a
small private helper and call it from both `_freq_sev_convolution` and
`reinsurance_df` (and from the gross-validation hook in 1.3 if adopted).

**3.3 Redundant `est_*` writes in reins methods. [D13 — decided: delete]**
Clarification of the earlier finding: the **final** `est_*` (after
`update_work`) are **net** — computed from the net `sev_density` / `agg_density`.
The `est_*` assignments **inside** `apply_occ_reins` / `apply_agg_reins` are
**redundant duplicates** (overwritten by `update_work`, and using the older
`xsden_to_meancv(skew)` helper rather than the de-fuzzed `xsden_to_mwrangler`).
Nothing reads `est_*` between the reins call and the recompute. **Delete them.**
Low risk.

**3.4 Forwards/backwards `S`. [OPEN — O5, investigate]**
`density_df` and `_build_augmented` use forwards `S = 1 − cumsum`; `add_exa_sample`
offers both; **`Distortion.price` uses backwards (per user)**. Reconcile to one
canonical `S` (or document why a given site needs backwards). Investigate
`spectral.py:Distortion.price` first.

**3.5 Double Severity construction in the mixture arm. [perf, low priority]**
In the mixture-product arm of `__init__` (weights ≠ 1), Severity objects are
built **twice**: once as `gup_sevs` (ground-up, used only to derive mixture
weights from `sf(attach)`), then again as `actual_sevs` (layered) for **each**
exposure row. For `E` exposure rows × `C` mixture components that is `C + E·C`
Severity constructions, each running a scipy setup + moment integration. Fine for
small specs; only worth revisiting if large limit-profile × mixture specs become
a use case. Possible fix: reuse/cache, or derive layer moments from the gup
severities without rebuilding. Low priority.

**3.6 Delete `aggregate_keys`. [D9]** Class attribute (`distributions.py` ≈1639),
defined and never referenced anywhere in `src`. Dead.

**3.7 Delete journey-of-discovery comments. [D10]** Their pith is preserved in
`dev/pipeline-aggregate.rst` ("Journey of discovery — distilled"). Remove the
narrative blocks and large commented-out alternatives during the touch of each
method.

**3.8 Component column naming: `e{e}.m{m}`. [D16]**
Broadcast components currently land in `stats_df` as flat `comp_0`, `comp_1`, … —
uninformative. They are really a **2-D grid: exposure component × severity-mixture
component**. Name them accordingly — `e0.m0`, `e0.m1`, `e1.m0`, … — where:
- `e{e}` = the **exposure** component, one per `(claims|premium, limit xs attach)`
  row (the *outer* broadcast / outer loop of the mixture-product arm);
- `m{m}` = the **severity-mixture** component, one per weighted severity in the
  mixture (the *inner* loop).
The mixture-product arm already loops exposure-outer / mixture-inner, so the
`(e, m)` indices fall straight out — wire this in during the `__init__` refactor
(alongside 3.5 and the `stats_df` schema work in 1.2). The limit-profile arm
(all weights 1, single severity) is simply `e{e}.m0`. **Implementation gotcha:**
the component columns are currently selected by `startswith('comp_')` in the
post-loop totals block and in `update_work` (the freq-weight vector) — replace
those with a robust identifier (a stored list, or an `e\d+\.m\d+` regex) so the
rename doesn't break them. Dotted labels are fine (consistent with the `T.M_*`
augmented columns).

### Group 4 — Reinsurance clause parsing (`so` / `po`)

**4.1 `so` (share of) / `po` (part of) disambiguation. [parser]**
They are **synonyms** as keywords, but the *number* determines meaning:

- `50% so 200 xs 100` or `50% po 200 xs 100` → **pro-rata share** of 0.5 of the
  `200 xs 100` layer. ⇒ `share = 0.50`, `limit = 200`, `attach = 100`.
- `100 po 200 xs 100` → an **absolute part-of** amount ⇒ `share = 100 / 200 =
  0.50`, `limit = 200`, `attach = 100`.

So the parser must branch on whether the leading quantity is a **percentage**
(→ use directly as share) or an **absolute amount** (→ `share = amount / limit`).
Both forms ultimately produce the `(share, limit, attach)` tuple consumed by
`make_ceder_netter`. Files: `decl.lark` (grammar may already accept both; the
**transformer** must detect `%`), `parser.py` reins-clause handling. Add tests
for both forms of both keywords. *Primarily a parser concern — natural split-out
candidate if Group 1 is tackled first.*

---

## 4. Before/after consistency harness (cross-ref)

The full proposal lives in `dev/pipeline-portfolio.rst`. For this plan the
relevant point: **capture the deterministic golden baseline from `REFACTOR` HEAD
before starting**, with reinsurance cases (occ-only, agg-only, occ+agg with
*mixed* kinds, `so` vs `po`, pro-rata vs part-of) explicitly in the corpus,
because Groups 1 and 4 change exactly those outputs. Items that intentionally
move numbers (2.4 Portfolio convention; any tail-mass change) update the baseline
deliberately, in the same commit, called out.

---

## 5. Suggested sequencing

1. **Harness first** — capture golden baseline (so everything below is safe).
2. **2.3 / 1.2 schema (+ 3.8 component renaming)** — drop the name row, make
   `stats_df` all-float, add the staged-column scaffold (NaN-filled), and rename
   the component columns `e{e}.m{m}`. Mechanical; unblocks the rest.
3. **2.1 / 2.2 valid** — route `valid` through `stats_df` only; name the magic
   constants. Small, high-confidence.
4. **3.2 FFT helper + 3.6 / 3.7 deletions** — local cleanups.
5. **1.1 describe + 1.2 staged empirical writes** — the visible reins-reporting
   payoff (Subject / Net-or-Ceded-or-After / Change, denser `EX` headings).
6. **1.3 validate-subject** — reuses the 3.2 FFT helper.
7. **2.4 Portfolio convention** — coordinated with a deliberate baseline update.
8. **Group 4 (so/po)** — independent; can be done any time, ideally with its own
   baseline cases.
9. **List items** (O5 forwards/backwards, 3.5 perf) — as capacity allows.

---

## 6. Future ideas

> Post-refactor, larger enhancements. Not scheduled; recorded so the design
> reasoning is not lost. Both items below are **the same machinery** seen from
> two directions — a movable, possibly-negative "window of interest" on the FFT
> grid.

### FI-1 — Negative `xs` (profit / loss distributions)

**Value.** Allow severity (and hence the aggregate) to take **negative** values,
so a single deal's profit/loss distribution can be modelled and several deals
convolved into a total-P&L distribution. This is one of the most-wished-for
enhancements; getting it done would be a big step.

**The proposed shift trick — and its catch.** The natural idea is: pick a shift
`s` with `X + s ≥ 0`, bucket `X + s` as an ordinary non-negative distribution,
and subtract `s` back out when displaying / computing moments. This works
perfectly **for the severity**. The catch is the aggregate:

```
A = X₁ + … + X_N          shift each Xᵢ by s
A_shifted = Σ (Xᵢ + s) = A + N·s
```

The per-claim shift is multiplied by the **random** claim count `N`. So
`A_shifted` is `A + N·s`, **not** `A + (constant)`, and you cannot recover `A`'s
distribution by subtracting a constant. (In FT terms the shift `e^{iωs}` sits
*inside* the frequency PGF — e.g. Poisson `exp(λ(e^{iωs}φ_X − 1))` — so it cannot
be factored out afterwards.) The de-shift is clean **only when frequency is
deterministic** (`fixed` N, including the `fixed`-1 case). For random frequency
the shift trick fails. *This is the "harder than expected" the author rightly
feared.*

**Where the shift *does* work: the Portfolio combine.** At the **portfolio**
level — convolving unit aggregates that have *already* been computed — there is
no random frequency: the total is a deterministic sum of the unit aggregates, so
shifting unit `i` by `sᵢ` shifts the total by `Σ sᵢ` (a constant) and de-shifts
cleanly. So negative values are *easy* at the portfolio-combine step and only
hard *within* a unit (the freq × sev convolution with random `N`). This narrows
where the difficulty actually lives — and matches the author's original framing
of the idea as a portfolio-level operation.

**The general solution: a windowed / circular FFT + reindex.** The FFT already
computes *circular* convolution on a grid of period `P = N·bs` (× padding).
A negative value `−v` is simply congruent to `P − v` — it lives at the **high end
of the array**. So negatives need no shift at all; they need the grid to be
*interpreted* on a circle and then **cut at the right place** so the (contiguous)
support is displayed as a monotone range that may start below 0. This is exactly
FI-2's machinery, and it handles random frequency because it never relies on a
constant shift.

**Impact on the Aggregate pipeline** (component by component):

- **Grid / `recommend_bucket` / `update`.** Today `xs = arange(0, 2^log2)·bs`
  starts at 0. Negative support needs a notion of *where 0 is* — either an
  explicit window offset `xs = (arange(N) − k)·bs`, or the circular
  interpretation with a display reindex. `recommend_bucket` must size for the
  **range** `max − min`, not just an upper percentile, and choose the window
  placement.
- **`discretize`.** The `adj_xs[0] = −inf` convention (dump all mass ≤ 0 into
  bucket 0) is exactly wrong for a P&L severity — negative mass must land in
  negative buckets. The left-edge `−inf` belongs only at the true left end of
  the window. This is a real change.
- **`_freq_sev_convolution` (the FFT core) — unchanged.** Circular convolution
  and the PGF application are agnostic to where 0 is. The `fixed`-1 and zero-risk
  shortcuts still hold. The maths does not change; only the *interpretation* of
  the input/output grid does.
- **`density_df` — one new step, then agnostic.** The only structural change is
  to **roll/reindex** the density to a monotone, real-valued `xs` (cut the
  circle where the support is contiguous, labels possibly negative) *before*
  building the frame. After that, `F = p.cumsum()`, `S = 1 − F`, `lev`, `q`,
  `tvar` all just need monotone `xs` — they work **unchanged**. This is the
  elegant part.
- **Moments.** `xsden_to_mwrangler` already takes `xs` as an argument, so once
  `xs` carries the real (negative-inclusive) labels, moments are correct. The
  de-fuzz is fine; the tail-mass-at-`xs[-1]+bs` convention needs a rethink under
  the circular interpretation (where the "deficit" is no longer simply a right
  tail).
- **Pricing / distortions / reinsurance — out of scope initially.** `epd`, LEV,
  "capital = a − P", and reinsurance layer functions all assume losses ≥ 0 and
  go semantically murky with negatives. Phase 1 should target the
  **distribution + risk-measure surface** (density, moments, `q`, `tvar`) and
  explicitly defer (or flag) the insurance-pricing overlay.

**Existing seed.** `Aggregate.unwrap(p, audit)` already finds the effective
support `[L, R]` from the `p` / `1−p` quantiles and reindexes the density on the
assumption it fits in one period — i.e. it *already does the reindex* for the
single-wrap case. FI-1 is largely "make that placement a first-class input
rather than a post-hoc repair."

**Priority: low — likely a long-term pipe dream.** The within-unit
random-frequency case is the genuine blocker, and it also forces the
`discretize` change above. *If* pursued, prefer the **windowed-FFT + reindex**
route (it generalises to random frequency and reuses `unwrap`) over the shift
trick; a plausible phase 1 is distribution + risk measures only, with the
fixed-frequency and **portfolio-combine** cases first (where the shift agrees,
giving a cross-check), and pricing / reinsurance deferred.

### FI-2 — Integrated aliasing handling & a movable window of interest

**The insight (from the `ft_invert` / `FourierTools` work).** A wrapped FFT is
**not necessarily an error**. The transform is periodic with period `P = N·bs`;
if the aggregate support is longer than `P` *and self-overlaps*, that is true
aliasing and the answer is corrupt. But if it merely **wraps once** (support
fits in one period but straddles the `0`/`P` boundary), it is fully recoverable
by **reindexing** — rolling `density_df` so the contiguous support is displayed.
Most users do not know this and panic at any wrap.

**Proposal.**

- Promote *window placement* to a first-class pipeline concept (the offset / cut
  point), instead of always starting at 0.
- **Detect single-wrap vs genuine overlap** and, in the single-wrap case,
  auto-unwrap (with a clear, non-alarming flag) rather than just raising the
  `ALIASING` validation flag. Reserve the hard failure for true self-overlap.
- Expose window controls so the user can deliberately move the window of
  interest.

**Relationship to FI-1.** Identical machinery: negative support is just "the
window of interest starts below 0", and single-wrap aliasing is "the window of
interest is rolled off the right edge". Implementing the movable window delivers
both. Prototypes already exist in `Aggregate.unwrap` and in `ft.py`
(`FourierTools`, the `recentering_convolution` examples) — the work is porting
that insight into the core `update` path and teaching `density_df` /
`recommend_bucket` / `valid` about a non-zero window origin.

**Education / docs.** Whatever ships, document the "a wrapped FT that only wraps
once is fine — here is how to unwrap it" point prominently; it is a recurring
source of user confusion.
