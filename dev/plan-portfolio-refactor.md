# Plan: Portfolio pipeline refactor

> Status: **draft for discussion**, not yet started. Captures the design
> discussion of 2026-05-28. No code changed yet. Companion to the current-state
> description in `dev/pipeline-portfolio.rst`, and sibling to
> `dev/plan-aggregate-refactor.md`. IDs (D1…, O1…) are **local to this
> document**. Move to `dev/done/` when complete.
>
> **Splitting advice (read first).** Like the aggregate plan, keep this as **one
> plan** — the distortion/pricing items share the `augmented_df` schema and the
> pentagon `PricingResult`, and the stats/validation items mirror the aggregate
> changes one-for-one. Natural carve-outs *if* staging: **Group 1
> (distortion/pricing/allocation)** — the through-line and largest — and the
> **pandas Copy-on-Write** item (D13), which is **library-wide** and should be
> coordinated with the aggregate work, not done twice.

---

## 1. Context & rationale (the discussion)

### 1.1 What a Portfolio does

Combine `Aggregate` units into a total under the **independence copula**, compute
the **kappas** `κᵢ(a) = E[Xᵢ | X=a]` (one FFT pair per line, exploiting
independence of `Xᵢ` and `X₋ᵢ`) that drive capital allocation, apply **spectral
distortions** to price and allocate, **generate samples** (optionally with an
induced rank-correlation by reordering — Iman–Conover "shuffle"), and run the
**switcheroo**.

### 1.2 The switcheroo (a big idea, not a side path)

Rebuild the object around an empirical multivariate **sample** by **replacing the
density frame with one whose kappas are estimated directly from the sample**. The
sample's dependency structure that matters for allocation *is* the kappa, so the
full high-dimensional sample collapses to a one-dimensional family of conditional
means and every downstream calculation (allocation, distortion pricing, the whole
`augmented_df` machinery) runs unchanged. "All manner of things simplify." This
should be treated as first-class, tested, and preserved through the refactor.

### 1.3 Lifted vs linear natural allocation — the crux

- **Lifted** reads the allocation from the distorted `augmented_df` using the
  **risk-adjusted** weights all the way into the tail. With a **mass** distortion
  (e.g. CCoC) on an **unbounded** support, essentially all the distortion weight
  lands on the last bucket → unstable, grid-dependent.
- **Linear** collapses states ≥ `a` to the asset level using **objective**
  probabilities, then prices the layers below `a`. It never applies the
  distortion far out in the tail → stable. The per-line allocation is already a
  single vectorised reverse sweep; with the objective leg hoisted out of the
  distortion loop, linear costs ≈ lifted.

Decision: **linear is the default** (lifted stays optional), and a lifted NA on
an unbounded + mass case is **refused** rather than returned unstable.

### 1.4 Two findings from the read (drive the work)

- **`_build_augmented` duplication + ROE-fallback bug.** The `efficient` and full
  branches recompute the same total-level block; their fallback ROE where
  `M.Q_total==0` (`gS==1`) **disagrees** — efficient uses `g'(1)`, full uses
  `1/g'(1) − 1`. The L'Hôpital limit of `(gS−S)/(1−gS)` as `S→1` is
  `1/g'(1) − 1`, so the **default (efficient) branch looks wrong** in the tail.
- **Empirical moments diverge from Aggregate.** `Portfolio.update` uses a plain
  `Σ p·xᵏ` ("PEG baseline") with no de-fuzz; Aggregate uses `xsden_to_mwrangler`
  on a de-fuzzed copy. Aggregate is the agreed convention.

---

## 2. Decisions: made vs open  *(2026-05-28; IDs local to this doc)*

| # | Item | Decision |
|---|------|----------|
| D1 | **Linear is the default** allocation; lifted stays available as an option | **decided** |
| D2 | **Refuse** a *lifted* NA on an *unbounded* distribution with a *mass* distortion (raise, point at linear); delete the "ugly" mass/tail-support code | **decided** |
| D3 | **Boundedness from the spec** (`freq bounded ∧ sev bounded`) where provable, with a user `bounded=True` **certify override** (fallback `bounded=False`) | **decided** |
| D4 | Portfolio empirical moments adopt **Aggregate's** `xsden_to_mwrangler` on a **de-fuzzed copy** | **decided** |
| D5 | Delete the `_write_empirical_stats` **sev-block inversion** — read `stats_df['empirical'][('sev','ex1..3')]` from each unit directly | **decided** |
| D6 | `pricing_at` → **pentagon** order `L M P Q a \| LR PQ ROE` (include `a`); this row *is* the canonical `PricingResult` | **decided** |
| D7 | **`allocation_method` member** (`'linear'`\|`'lifted'`, default `linear`) drives `pricing_at`/`price`/`apply_distortion`/`augmented_df`; a **property setter invalidates the `augmented_df` cache** on change; shown in `info`. **Delete `price`** if then redundant (check consistency first!) | **decided** |
| D8 | **`S` forwards by default** everywhere, **offer `backwards`** consistently. Under a genuine deficit (`Σp<1`) the two legitimately diverge by the deficit — emit `DefectiveDistributionWarning` (subclass of `UserWarning`). See aggregate plan D17 | **decided** |
| D9 | **`Distortion.price`** (spectral.py) → return a **`PricingResult`** + default **forwards** + docstring update (keep backwards reachable). Folds into the D17 cross-module flip | **decided** |
| D10 | `valid` reads **only** from `stats_df` (single source of truth), mean/aliasing included | **decided** |
| D11 | **No magic numbers** — name/share `eps**3`, `10×`, `exeqa_err<1e-4`, `\|ft\|<1e-10`; **root out** the `loss_max` length-bucketed `mult∈{1,10,100}` machinery for a principled rule | **decided** |
| D12 | **`_build_augmented`**: compute the total-level block **once** with the correct `gprime1 = 1/g'(1) − 1`; `efficient` gates only the extra `M.*`/η-μ columns | **decided** (verify ROE-fallback against a mass distortion first) |
| D13 | **Adopt pandas Copy-on-Write** — single switch `pd.options.mode.copy_on_write = True` in `aggregate/__init__.py`; fix the fallout once, library-wide. Cross-ref aggregate plan D18 | **decided** |
| D14 | **Drop unneeded `augmented_df` columns** (intermediate `M.*` the readouts never consume) | **decided** |
| D15 | **`add_exa_details`/`add_eta_mu`/EPD/η-μ**: slim to *exactly* what `plot_twelve` consumes (not everything), move to / gate behind `pedagogy` | **decided** (need the exact column audit) |
| D16 | **Hoist** the distortion-independent `exp_loss` out of the linear-allocation distortion loop | **decided** |
| D17 | **`swap_density_df` → standalone function** that swaps marginal densities and **recomputes stats via the `xsden_…` family** (no `mixed`/`independent` — a swapped object has no freq/sev decomposition) | **decided** (was O1) |

*(No open items: the former O2 "tolerance tiers" is not a decision but the testing method — it lives in the harness plan, `dev/plan-baseline-harness.md`.)*

---

## 3. Work items

### Group 1 — Distortion, pricing & allocation (the through-line)

**1.1 Linear default + lifted optional. [D1]**
Flip the commented `DeprecationWarning`; `allocation='linear'` becomes the
default in `price` / `pricing_at`. Keep `'lifted'` working. Files:
`Portfolio.price`, `pricing_at`.

**1.2 Refuse lifted on unbounded + mass; delete the ugly support code. [D2]**
When `allocation='lifted'`, the distortion has a mass, and the support is
unbounded (D3), **raise** with a message pointing at `linear`. Remove the
mass/tail-fill hacks that currently try to limp through this case. Files:
`price`, `_build_augmented` (the tail-fill / truncation around the mass).

**1.3 Boundedness from the spec + certify override. [D3]**
Add a `bounded` property/flag. `aggregate bounded ⟺ frequency bounded AND
severity bounded`:
- frequency: `fixed`/`bernoulli`/`binomial`/empirical `dfreq` bounded;
  poisson/negbin/geometric/logarithmic/mixed-poisson unbounded.
- severity: finite `exp_limit` or `sev_ub`, or discrete/finite-support, or a
  bounded family (beta/uniform).
Auto-detect where provable (handles `dsev[10000]` → unbounded via the Poisson
frequency); fall back to `bounded=False`; let the user set `bounded=True`.
Conservative by design — the override is the escape hatch. Mixtures: bounded
iff all components bounded. Files: a helper on `Aggregate`/`Portfolio` reading
the spec; consumed by 1.2.

**1.4 `pricing_at` pentagon order. [D6]**
Reorder to `L M P Q a | LR PQ ROE` (amounts then ratios; include `a = P+Q`,
`LR=L/P`, `PQ=P/Q`; see `pentagon.py`) and bake into the `PricingResult` /
`CategoricalDtype`. Files: `pricing_at`, `PRICING_STAT_ORDER` /
`PRICING_STAT_DTYPE`, `price`, `analyze_distortion(s)`.

**1.4b `allocation_method` as a member, not an argument. [D7]**
`linear` vs `lifted` affects not only `pricing_at` but `apply_distortion` /
`augmented_df` and its cache. Rather than thread an argument through all of them,
make it a **member**: ``self.allocation_method ∈ {'linear','lifted'}``, default
``'linear'``, exposed as a **property whose setter invalidates the
``_augmented_dfs`` cache** (the augmented frame differs by method). Show the
mode in ``info``. All of ``pricing_at`` / ``price`` / ``apply_distortion`` /
``augmented_df`` read ``self.allocation_method``. *Alternative* if it turns out
to change often: key the cache by ``(distortion_name, method)`` instead of
invalidating — but member + invalidate is simpler and "doesn't change that
often". If ``price`` becomes a thin wrapper under this, **delete it**.
**Subtlety to resolve in impl:** lifted's ``augmented_df`` is a full curve (read
at any ``a``); the linear allocation as written is **``a``-specific** (the
tail-collapse sets ``S[a]=0``), so a cached "linear augmented_df" must either
drop the collapse (full-curve form) or be keyed by ``a`` too.
Files: ``apply_distortion``, ``augmented_df``, ``_build_augmented``,
``pricing_at``, ``price``, ``info``.

**1.5 `_build_augmented` de-dup + ROE-fallback fix. [D12]**
Compute `gS`/`gp_total`, per-line `exag`/`exi_xgtag`, and the total-level block
(`exag_total`, `M.M_total`, `M.Q_total`, `M.ROE_total`, `roe_zero`) **once**,
with `gprime1 = 1/g'(1) − 1`. `efficient=True` then only *skips* the extra
per-line `M.*` marginals and the η-μ columns. **Verify** the ROE-fallback claim
against a mass distortion (it can change `T.Q` in the `gS==1` tail) — prime
harness case. Files: `_build_augmented`.

**1.6 `Distortion.price` → `PricingResult` + forwards. [D9]**  *(spectral.py)*
Currently returns a bare namedtuple and defaults `S_calculation='backwards'`.
Change to: (a) return a **`PricingResult`** (the pentagon row — single
distribution, no allocation, so just the total), (b) default **forwards**, (c)
keep `backwards` as an option and **rewrite the docstring** (which currently
advocates backwards) to: forwards is the default for consistency, backwards
remains available and is more reliable for *thin-tailed* risks (forward
`1−cumsum` loses tail detail). Files: `spectral.Distortion.price`. *Cross-module
item — note it lives in `spectral.py`, not `portfolio.py`.*

**1.7 Drop unneeded `augmented_df` columns + hoist `exp_loss`. [D14, D16]**
With `efficient=True` trimming, also drop intermediate `M.*` columns nothing
reads. In the linear branch, compute the distortion-independent `exp_loss`
**once**, not per distortion. Files: `_build_augmented`, `price` (linear branch).

### Group 2 — Stats, moments & validation hygiene (mirrors Aggregate)

**2.1 Empirical-moment convention → Aggregate's. [D4]**
Replace `Portfolio.update`'s plain `Σ p·xᵏ` with `xsden_to_mwrangler` on a
**de-fuzzed copy** (same treatment as `Aggregate.update_work`; this is aggregate
plan D8). Coordinate the resulting baseline shift deliberately (harness). Files:
`Portfolio.update`, `create_from_sample` (duplicated moment block).

**2.2 Delete the sev-block inversion. [D5]**
`_write_empirical_stats` reconstructs `(ex1,ex2,ex3)` from `(mean,cv,skew)` via
`MomentWrangler.central` — unnecessary now that Aggregate stores empirical sev
raw moments. Read `a.stats_df['empirical'][('sev','ex1..3')]` directly. Fold in
with 2.1. Files: `_write_empirical_stats`.

**2.3 `valid` from `stats_df` only. [D10]**
Move mean/aliasing reads off `describe` and onto `stats_df` (CV/skew already do);
mirrors the aggregate change. Files: `Portfolio.valid`.

**2.4 No magic numbers; kill the `mult` machinery. [D11]**
Name/share `eps**3`, the `10×` aliasing ratio, `exeqa_err < 1e-4`, the
`|ft| < 1e-10` "build up the product" guard. **Root out** the `loss_max`
`mult∈{1,10,100}`-by-frame-length blanking (in both `add_exa` and
`add_exa_details`) — replace with a principled rule (e.g. blank where
`F < k·eps`). Files: `add_exa`, `add_exa_details`, `_build_augmented`,
`update`, `constants.py`.

### Group 3 — Structure & cleanup

**3.1 `S`-convention unification. [D8]**
`add_exa` (`1-F`), `_build_augmented` (forwards), `add_exa_sample` (both), and
`Distortion.price` (backwards) should all default to **forwards** and accept a
uniform `backwards` option. Document one canonical `S`. Files: those four sites.

**3.2 `add_exa_details` slim + quarantine. [D15]**
Audit exactly which columns `pedagogy.plot_twelve` (via its helper) consumes —
it is **not** everything `add_exa_details` produces. Provide only those; move the
EPD + η-μ surface into `pedagogy` (or gate behind an explicit opt-in) so the core
`density_df` is lean. Files: `add_exa_details`, `add_eta_mu`, `pedagogy.py`.

**3.3 `swap_density_df` → standalone function. [D17]**
Promote it from EXPERIMENTAL method to a supported standalone **function** that
swaps in marginal densities, recombines, runs `add_exa`, **and knocks out fresh
stats via the `xsden_…` family** — the one wrinkle. A swapped object has **no**
`mixed`/`independent` columns (no frequency/severity decomposition), so the stats
are just the empirical moments from `xsden_to_mwrangler`/`ser_to_mwrangler`.
Reuse (don't duplicate) `update`'s `ft_nots` recombination. Files:
`swap_density_df` (→ function), `xsden_to_mwrangler` / `ser_to_mwrangler`.

**3.4 Pandas Copy-on-Write. [D13]**
Enable `pd.options.mode.copy_on_write`. CoW is the pandas-3.0 default and removes
the chained-assignment `FutureWarning`s (likely most of the 77) and defensive
copies. The wide `df['x'] = …` builders are mostly CoW-clean already; fix the
fallout. **Library-wide — coordinate with the aggregate work, do it once.**

**3.5 Delete journey-of-discovery comments.**
Pith preserved in `dev/pipeline-portfolio.rst`. Remove the narrative blocks and
commented-out alternatives as each method is touched.

### Group 4 — Samples & switcheroo (mostly preserve + test)

**4.1 Make the switcheroo first-class + tested. [from 1.2 context]**
The sample path (`create_from_sample` / `add_exa_sample`) and `Portfolio.sample`
(with `desired_correlation` shuffle) are correct but lightly tested. Add the
sample-built Portfolio to the harness corpus (Group/§4) so the kappa-replacement
survives the refactor. Files: tests; light touch on `add_exa_sample` only if the
`S`-unification (3.1) touches its `forwards/backwards` branch.

---

## 4. Before/after consistency harness (cross-ref)

The harness has its **own plan with the concrete DecL corpus:
`dev/plan-baseline-harness.md`** (the narrative proposal is in
`dev/pipeline-portfolio.rst`). For this plan:

- **Shared baseline with aggregate.** One harness, one golden baseline captured
  from `REFACTOR` HEAD before any of this starts.
- **Corpus is small on purpose** (max-digit precision, deterministic): ~3–5
  aggregates composing **2 Portfolios** (one parametric PEG-like, one
  sample-built). Snapshot `stats_df`, `describe`,
  `density_df.filter('p_|exeqa_')` + `loss`/`S`, then **two distortions — one
  with a mass (CCoC), one without (PH/Wang)** — capturing `gS`/`gp_total`/
  `exag_*` under **both `lifted` and `linear`**, plus `q`/`tvar` and the
  `analyze_distortions` frames.
- **Initial goal: VERY VERY close** (start `rtol≈1e-12`; introduce a looser tier
  only where an `interp1d`/reins-step/root-solve forces it). [O2]
- Items that **intentionally** move numbers (2.1 empirical convention; 1.5 ROE
  fix) update the baseline deliberately, in the same commit, called out.

---

## 5. Suggested sequencing

1. **Harness first** — execute `dev/plan-baseline-harness.md`: capture the shared
   golden baseline (covers aggregate too) before any refactor edits.
2. **3.4 pandas CoW** — library-wide; do early so subsequent edits are written
   CoW-clean and the warning noise is gone.
3. **2.x stats/validation hygiene** — empirical convention (2.1) + sev-block
   delete (2.2) + `valid` SSoT (2.3) + magic numbers (2.4). Mirrors aggregate;
   coordinate so both modules land together. (2.1 moves the baseline — planned.)
4. **1.5 `_build_augmented` de-dup + ROE fix** — verify against a mass distortion;
   this is where a real number may change, so do it under the harness.
5. **1.4 pentagon `pricing_at` + `linear`/`lifted` arg; 1.7 drop cols / hoist** —
   the pricing-surface reshape.
6. **1.1 linear default + 1.2 refuse lifted + 1.3 bounded detection** — the
   allocation policy, on top of the reshaped surface.
7. **1.6 `Distortion.price`** — spectral.py; `PricingResult` + forwards.
8. **3.1 `S` unification, 3.2 `add_exa_details` slim, 3.3 `swap_density_df`,
   3.5 comment deletion** — cleanup once the behaviour-bearing work is green.
9. **4.1** — switcheroo tests (folded into the harness corpus).

---

## 6. Future ideas

> Post-refactor, larger enhancements. Recorded so the reasoning is not lost.

### FI-1 — `Portfolio.pricing_bounds` rewrite

Currently raises `NotImplementedError` (CLAUDE.md): it was wired to the legacy
`Bounds.tvar_cloud` API and passed the dense `density_df.S` as the s-grid (a
shape mismatch downstream). The new `Bounds` uses a fixed 513-point binary
`s_grid`; aligning `pricing_bounds` requires deciding how to interpolate the
cloud's distortion values onto the per-unit `exeqa_*` columns at portfolio
resolution. Defer until the design is settled (the author wants to think it
through). **The author wants to be reminded about this periodically** — surface
it whenever portfolio pricing / bounds work comes up.

### FI-2 — Negative `xs` at the portfolio-combine level

The easy half of aggregate FI-1: convolving **already-computed** unit aggregates
is a deterministic sum, so a constant shift de-shifts cleanly (no random
frequency at the combine step). So profit/loss units could be summed at the
portfolio level even while the within-unit random-frequency case stays hard.
Cross-ref `dev/plan-aggregate-refactor.md` §6 FI-1. Low priority; depends on the
aggregate windowing work.
