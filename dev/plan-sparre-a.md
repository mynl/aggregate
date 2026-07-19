# [Renewal-Frequency-Wait-Clause] + [Empirical-PGF-Horner-Dispatch]

Status: not yet executed (planned 2026-07-19). On completion move to
`dev/done/` and bump the version (plan-based change).

## Context

The `sparre` notes library (`C:/s/AI/notes/sparre-andersen`, esp. `theory.md` §§1–5, 7, 11 and
`src/sparre/renewal.py`) computes the renewal count N(1) for iid waiting times W by a
Plancherel/no-inverse-FFT method: `{N(t) ≥ k} = {S_k ≤ t}`, so `P(N=k) = F^{*k}(t) − F^{*(k+1)}(t)`,
and each `F^{*k}(t)` is a frequency-domain inner product of `p̂^k` with the transform of the tilted
indicator of `[0, t]` — two forward rffts total, one vector multiply + one dot per k. Goal: let
`aggregate` model the claim-generation process as a general Sparre-Andersen renewal process (any
waiting-time law, not just exponential ⇒ Poisson), via a new DecL surface:

```
agg LABEL
    NUMBER year[s] [at NUMBER rate]     <- new exposure form (T = years)
    <severity clause, unchanged>
    [occurrence reinsurance]
    wait <severity-expression>          <- full sev mini-language: 0.1*expon, mixtures,
  or dwait [outcomes] [probs] [!]          splice, !, sev.NAME; dwait mirrors dsev
    [agg reinsurance]  [note{...}]
```

Equivalences (correctness anchors): `10 claims … poisson` ≡ `10 years … wait expon`
≡ `1 years … wait 0.1 * expon` (rate 10/yr for 1 year).

Second, related change: `FrequencyEmpirical.freq_pgf` (`_frequency.py:587-588`) currently does
`freq_b @ np.power(z, freq_a.reshape(-1,1))` — an `n_atoms × len(z)` complex matrix (memory-heavy,
poorly conditioned). Replace with a dispatching polynomial evaluator: **Horner** for dense supports,
**sorted-gap square-and-multiply** for sparse ones. The renewal count pgf (dense 0..kmax) reuses it.
Note the FFT pipeline already does only ONE inverse FFT (`freq_sev_convolution`,
`_aggregate_compute.py:91-94` applies `freq_pgf` to `FFT(sev)` then a single `ift`) — the
`Σ pᵢ IFFT(FFT(X)^i) = IFFT(Σ pᵢ FFT(X)^i)` identity is already in place; only the *polynomial
evaluation* changes.

**Cost model / dispatch rule**: with sorted exponents `k₁<…<k_K`, gaps `Δᵢ = kᵢ − kᵢ₋₁`: Horner
costs `k_max` vector fused multiply-adds; incremental binary exponentiation costs `Σᵢ 2⌈log2 Δᵢ⌉`
vector multiplies. Both are O(len(z)) per op, so compare the two integers and pick the smaller
(ties → Horner: better conditioned, single accumulator). Dense `0..K` ⇒ Horner; sparse
`[0 1 2 1000]` ⇒ powers. Implement squaring explicitly — do NOT rely on `np.power` (complex pow
falls back to exp/log for large exponents).

## Decisions made with the author (settled)

- `at NUMBER rate` ⇒ `exp_premium = years × rate`, **informational** (feeds PnL gross premium).
  Exposure/count comes solely from the wait law over T years. Also store `exp_rate` for byte-exact
  writer round-trip.
- **Strict pairing**: `years` exposure ⇔ `wait`/`dwait` clause, enforced at the grammar level
  (separate `agg_body` alternative); `2 years … poisson` and `10 claims … wait expon` fail to parse.
- **Flat `wait_*` spec keys** mirroring `sev_*` as `Aggregate.__init__` kwargs.
- Canonical name **`renewal`**: `freq_name='renewal'`, class `FrequencyRenewal`.
- Negative wait mass: collapses into the 0 bucket + warning stating the resulting P(W=0); all mass
  ≤ 0 ⇒ error. Zero-wait mass p0 = P(W ≤ 0) (atoms at 0, e.g. `dwait [0 …]`, plus collapsed
  negative mass): exact geometric-batch factor-out — remove the atom, **renormalize the remaining
  pmf by (1−p0)** (a true splice defect survives as the genuine conditional defect), compute count
  M for the conditional law `W|W>0`, then `N = Σ_{i=1}^{M+1} Gᵢ − 1`, `G ~ Geom{1,2,…}`,
  `P(G=g)=(1−p0)p0^{g−1}`. The (M+1)-th batch is the run of zero-waits riding at the last epoch
  ≤ T (equivalently the leading zeros at time 0) — it always counts; dropping it (the plan's
  original `N = Σ_{i≤M} Gᵢ`) is WRONG (check case `dwait [0 1] [.5 .5]`, T=1: exact
  P(N=m) = m/2^{m+1}, boundary-dropped version gives 2^{−m}). Composed pmf:
  `P(N=n) = Σ_m P(M=m)·C(n,m)·(1−p0)^{m+1}·p0^{n−m}`, support n ≥ m; p0=0 ⇒ identity; formula
  unchanged for defective waits (the zeros before the terminating draw still count, and their run
  length is independent of the terminator type). (Continuous rounding mass in `(0, h/2)` stays in
  bucket 0 of the *transform*, as in `sparre` — it is O(h) discretization noise, not a batch atom.)
- Defective waits (`wait expon splice [0 1.1] !`): total mass < 1 = terminating renewal process;
  count pmf still proper. Verified `_severity.py:1101` `_apply_lb_ub` ALWAYS renormalizes splices
  (`!` only affects layer attachment), so defectiveness is a grid-level post-pass (below), not a
  `Severity` change.
- Exact-lattice `dwait`: when atoms and T are commensurable, use the exact step and FULL-bucket
  readout at T (atom exactly at T belongs to `P(S_k ≤ T)`); continuous waits use the half-bucket
  endpoint correction (`ind[n1] *= 0.5 + (T − n1·bs)/bs`).

## Discretization: REUSE, not a new discretizer

`Aggregate.discretize` (`_aggregate.py:2451-2572`) is nearly pure already: build `adj_xs` edges
from `(xs, bs, sev_calc, signed)`, then per-component `diff(cdf)` / `-diff(sf)` / max-of-both,
optional normalize, optional linear scatter for discrete atoms. Plan:

- **Extract the body into a module-level kernel** `discretize_severities(sevs, xs, bs, *, i0=0,
  sev_calc='discrete', discretization_calc='survival', normalize=True, dsev_bucket=None,
  rebucket=None) -> list[ndarray]` in `_aggregate_compute.py` (the established home for pure
  kernels, cf. `freq_sev_convolution`). `Aggregate.discretize` becomes a thin delegate passing
  `self.sevs`, `self.xs_sev`, `self.bs`, `self.i0`, `self.dsev_bucket`,
  `self._rebucket_to_grid`. **Byte-for-byte regression-neutral** — the existing suite is the guard.
- **The wait path calls the same kernel** with the wait grid `xs_W = arange(n)·h` (h from the
  sizing rules below), `sev_calc='discrete'` (the rounding scheme — identical math to `sparre`'s
  `round_discretize`: bucket j = mass of `[(j−½)h, (j+½)h)`, first bucket from −inf), and
  `normalize=False` — a wait pmf is *deliberately* short on the grid (mass beyond T is dropped; it
  cannot contribute to any `F^{*k}(T)`, theory §2).
- Wait-specific work is then three tiny **post-passes on the pmf vector**, not new discretization:
  1. `p0` split: `p0 = component.cdf(0)` (atoms at 0 + collapsed negative mass); subtract from
     bucket 0; warn (with the P(W=0) value) when negative mass was collapsed; error if `p0` is the
     whole mass.
  2. Defective window (unconditional splice only): build the component `Severity` WITHOUT
     `lb/ub` (else `_apply_lb_ub` renormalizes) and zero out buckets outside `[lb, ub]` — the
     retained mass is the defect. Conditional splices pass `lb/ub` into `Severity` as normal and
     need nothing here.
  3. Truncate beyond T (`pm[n1+1:] = 0`).
- Mixtures: weight-combine the per-component post-passed pmfs (and p0, defect) with `wait_wt`.

## Wait-grid sizing, kmax, and frequency moments

**A-priori inputs** (exact, pre-discretization): per-component `Severity.moms()` gives conditional
wait mean μ and sd σ; mixture-combine. Analytic anchors — elementary renewal theorem
`E N(T) ≈ T/μ`, second-order `E N(T) = T/μ + (σ²−μ²)/(2μ²) + o(1)`, renewal CLT
`N(T) ≈ Normal(T/μ, T·σ²/μ³)` — used for sizing and sanity diagnostics, never as the reported
moments.

**kmax** (series length): `kmax = ⌈T/μ + z·√(T·σ²/μ³) + z⌉`, z=10, from the *discretized*
conditional moments (recomputed after discretization so splices/truncation are respected); for
defective mass q<1, cap by `⌈log(1e-17)/log(q)⌉` and take the min. Warn when kmax > ~1e5 (cost is
`kmax` vector multiply-adds on the count grid — still fast, but worth surfacing).

**Bucket size h** (three constraints, take the tightest):
1. Coverage: `m·h > T` with ~25% headroom (`n1 ≈ ¾·m`); tilt (θL = 20) damps wrap-around, so no
   more than that is needed. Lattice must hit T exactly: `h = T/n1`, integer `n1`.
2. Shape resolution: `h ≤ μ/κ`, κ = 64 (resolve the wait law near its scale; critical when μ ≪ T,
   i.e. many claims per period). Use σ as a secondary floor only when σ ≥ μ/10 (near-deterministic
   continuous waits otherwise blow up log2; truly discrete waits take the exact-lattice path).
3. Accuracy: with the half-bucket correction the per-convolution error is O(h²) accumulating
   ~linearly in k, so total ≈ O(kmax·h²); require `kmax·h² ≤ tol` (tol ~ 1e-9) ⇒ `h ≤ √(tol/kmax)`.

Then `log2 = ⌈log2(T·4/3 / h)⌉`, floor 16, cap 24 with a warning; recompute `h = T/n1`. dwait
exact-lattice detection (atoms and T commensurable via Fraction/gcd with tolerance) overrides all
of this with the exact step. **Diagnostic, explicit opt-in** (no magic):
`FrequencyRenewal.convergence_check()` recomputes the count pmf at h/2 and reports max |Δp_k| —
a Richardson-style check, also used by the tests.

**`_renewal_bs_df`** — the selection logic is captured in a DataFrame, mirroring the aggregate
`_bs_window_df` idiom (`_bucket_window.py` `_row`/`_size` style): one row per constraint —
`coverage` (m·h > T·4/3), `shape` (μ/κ), `accuracy` (√(tol/kmax)), `exact_lattice` (dwait
override), `log2_cap` — with columns for the implied `bs`/`log2`/`n1`, whether the row is
feasible, and a `selected` marker on the binding constraint; plus the final `bs`, `log2`, `n1`,
`kmax`, `p0`, `defect`, and estimated count error `~kmax·h²`. Built inside `wait_grid`, stored as
`FrequencyRenewal._renewal_bs_df`, surfaced on the owning `Aggregate` (property delegating to
`self.frequency`, populated whenever `freq_name == 'renewal'`).

**Frequency moments and downstream sizing** (all inherited/automatic once `freq_a=0..kmax`,
`freq_b=pN` are set):
- `freq_moms(n)` = exact sums `Σ kʲ p_k` (inherited from `FrequencyEmpirical`), independent of n.
  The frequency object is built at `_aggregate.py:1587`, *before* `MomentAggregator` consumes
  `freq_moms` at `:1734` — so `stats_df` theoretical freq/agg moments, `self.n = ma.tot_freq_1`
  (`:2005`), CV/skew, and `explain_validation` all work unchanged and are exact-to-discretization.
- Expected count: the `exp_en = -1` sentinel (same as `dfreq`) makes the exposure loop derive
  `_en = Σ freq_a·freq_b` (`_aggregate.py:1802-1804`) = E N(T) over the whole T years; `el`,
  premium, lr follow.
- **Aggregate** grid sizing (`_bs_window` → `_bucket_window.bs_window`): the moment row consumes
  `self.frequency.freq_moms(self.n)` (`_aggregate.py:4593`) — real numbers, works unchanged. With
  `'renewal'` added to the `_exact_discrete_window` carve-out (`:4505`), a renewal × integer-lattice
  dsev also gets exact-discrete window sizing (count support bounded by kmax). The count itself
  wraps in a `GridDistribution` for q/tvar via the inherited `freq_pmf` machinery.

**Realized frequency = a dfreq (confirmed design).** The moment `_build` finishes,
`FrequencyRenewal` *is* an empirical frequency: `freq_a = 0..kmax`, `freq_b = pN` is exactly the
`dfreq [k…] [p_k…]` representation, and every downstream consumer (pgf evaluation, moments, the
`exp_en=-1` en derivation, count support, `freq_pmf`) runs the ordinary dfreq/empirical logic —
the renewal computation only ever *produces* that vector pair. Two views, kept distinct:
- **Model round-trip** (`spec` / `decl_writer`): keeps the `years … wait` clause, so a rebuilt
  object recomputes pN from the wait law (the model is the wait distribution, not its realization).
- **Realized count** (`_count_program` / `create_frequency()`): the renewal branch emits the
  materialized `dfreq [0:kmax] [pN…]` program — the realized frequency as a first-class object
  running through the usual dfreq machinery, no recomputation of the renewal kernel.

## Stages

### Stage 0 — kernel extraction + `src/aggregate/_renewal.py`

1. Extract `discretize_severities` from `Aggregate.discretize` into `_aggregate_compute.py`;
   delegate; full suite green (regression-neutral refactor).
2. New leaf module `src/aggregate/_renewal.py` (numpy + the kernel import only):
   - `renewal_count_pmf(pm, bs, T, *, lattice=False, z=10.0, tilt_total=20.0, kmax=None) -> (k, pN)`
     — port of `sparre/renewal.py` generalized t=1→T, operating on an already-discretized pmf:
     tilt `exp(−θx)`, `θ = tilt_total/(m·bs)`; two forward rffts; Plancherel vector `c` with rfft
     weights `[1,2,…,2,1]`; loop `v *= p̂`, `F[k] = Re(v @ c)`; `pN = clip(−diff(F), 0, None)`.
     Readout weight: half-bucket (continuous) vs full-bucket (`lattice=True`). Require `m·bs > T`.
   - `geometric_batch_compose(pmf_M, p0) -> pmf_N` — boundary-batch-corrected accumulation
     `P(N=n) = Σ_m pmf_M[m]·C(n, m)·(1−p0)^{m+1}·p0^{n−m}` (i.e. `N = Σ_{i≤M+1} Gᵢ − 1`;
     O(kmax²), exact; p0=0 ⇒ identity).
   - `wait_grid(mu, sigma, T, atoms=None) -> (bs, log2, lattice)` — the sizing rules above,
     building `_renewal_bs_df` as a side product.
   - Orchestrator `wait_count_pmf(components, weights, T, ...)`: kernel-discretize each component
     on the common grid, post-passes (p0 split / defective window mask / truncate at T),
     weight-combine, renormalize by (1−p0), kmax, `renewal_count_pmf`,
     `geometric_batch_compose` if p0>0.
3. Tests `tests/test_renewal.py` (no DecL): expon→Poisson(λT); gamma(a, rate λ) →
   `P(N≥k) = gammainc(k·a, λT)`; uniform(0,1) T=1 → `P(N=k) = k/(k+1)!`; inverse Gaussian closed
   form; deterministic `pm=[0,1]` (atom at x=1) bs=1 T=3 → N≡3 (lattice boundary); Richardson
   halve-h convergence (O(h²) observed); defective q<1 (proper pmf, terminating tail);
   `geometric_batch_compose` vs brute-force enumeration for W ∈ {0, 1}; direct-vs-factored
   cross-check (kernel run with the 0-atom left in ≈ factored composition, ~1e-12 — the atom
   convolves exactly at lattice index 0, so both routes model clustering identically).

### Stage 1 — [Empirical-PGF-Horner-Dispatch] (independent of Stage 0)

- `evaluate_pgf_polynomial(atoms, weights, z)` in `_aggregate_compute.py`. Guard: non-integer atoms
  → legacy matrix path verbatim (fractional dfreq outcomes parse today; must not change). Integer
  atoms → dispatch per the cost rule: (a) dense Horner over scattered coefficients 0..kmax (one
  fused multiply-add per degree, O(len(z)) memory); (b) sorted-gap square-and-multiply (explicit
  squaring loop). Handle scalar and 1-D real/complex `z` (callers: `freq_sev_convolution`,
  `freq_pmf` `_aggregate.py:3306`, `bivariate.py:871/1920`, `_aggregate_compute_massive.py:405`,
  `ft.py:298`).
- `FrequencyEmpirical.freq_pgf` (`_frequency.py:587-588`) → one-line delegation. ZM wrappers
  (`_install_zm_wrappers`, `_frequency.py:232-255`) wrap the bound method — compose unchanged.
- Tests `tests/test_pgf_polynomial.py`: dense (`dfreq [0:5]`) and sparse (`dfreq [1 1000 100000]`)
  vs the legacy matrix expression at ~1e-13 on rfft-shaped complex z; scalar/1-D shape-dtype
  parity; end-to-end `build()` density regression dense+sparse; a ZM empirical case.

### Stage 2 — Grammar, terminals, mirrors, transformer (parse-only, testable standalone)

1. `decl.lark`:
   - `agg_body` (`:77-80`): add
     `| exposures_years layers sev_clause occ_reins wait_clause agg_reins approx_clause orientation -> agg_body_renewal`.
   - New rules:
     `exposures_years: numbers YEARS as_label -> exposures_years | numbers YEARS as_label AT numbers RATE -> exposures_years_rate`;
     `wait_clause: WAIT sev as_label -> wait_clause_wait | dwait as_label -> wait_clause_dwait`;
     `dwait: DWAIT doutcomes dprobs -> dwait_main | dwait "!" -> dwait_unconditional`.
   - Terminals `WAIT.2`, `DWAIT.2`, `YEARS.2: /(?:years|year)(?![a-zA-Z0-9._:~\-])/` (standard
     boundary lookahead); add `wait|dwait|years|year` to the ID-exclusion alternation (`:688`).
     `AT`/`RATE` reused. Earley-safe: YEARS keyword-disjoint from CLAIMS/LOSS/PREMIUM/EXPOSURE at
     the head; WAIT/DWAIT disjoint from FREQ/MIXED/ZM/ZT/AGGREGATE in the freq slot; the reused
     `sev` subtree already terminates before following keyword clauses.
2. Mirrors (enforced by `tests/test_grammar_sync.py`): `decl_pygments.py` (dwait → discrete group
   ~:119; wait/years/year → keyword group ~:131-136); `parser_errors._TERMINAL_LABELS` entries for
   WAIT/DWAIT/YEARS.
3. `parser.py` transformer:
   - `exposures_years` → `{'exp_years': T, '_exposure_label': …}` (scalar only — reject vectors);
     `exposures_years_rate` adds `{'exp_rate': rate, 'exp_premium': T*rate}`.
   - `_sev_to_wait(d)` helper: rename `sev_*` → `wait_*`; drop `name/note/hints/label` from a
     `sev.NAME` lookup; reject `sev_pick_*` ("picks not meaningful on a wait") and
     `sev_signed`/`sev_reflect`.
   - `wait_clause_wait`/`wait_clause_dwait` (+ `'_wait_label'`, new `_INTERIOR_LABEL_KEYS` site
     `'wait'`, `parser.py:407-411`); `dwait_main` → `{'wait_name': 'dhistogram', 'wait_xs',
     'wait_ps'}`; `dwait_unconditional` → `wait_conditional=False`. dwait prob-sum policy: without
     `!` renormalize with warning if `|Σp − 1| > 1e-6`; with `!` accept Σp ≤ 1, error above 1.
   - `agg_body_renewal` → merged fragments + `{'freq_name': 'renewal', 'exp_en': -1}` (empirical
     sentinel, exactly like `dfreq`, `parser.py:1620-1628`).
4. Tests: `tests/test_renewal_decl.py` — exact spec dicts for plain / `at rate` / dwait(±`!`) /
   mixture+splice / `wait sev.NAME` / labels; strict-pairing parse FAILURES (`2 years … poisson`,
   `10 claims … wait expon`); `yearly` still a valid ID. Corpus: new section in
   `src/aggregate/agg/_test_suite.agg` + mirrored lines in `src/aggregate/agg/decl-testers.agg`;
   re-capture `tests/data/expected_specs.json` (`uv run python tests/capture_sly_snapshot.py`),
   verify the diff is additive only.

### Stage 3 — `FrequencyRenewal` + `Aggregate` wiring

1. `_frequency.py`: `class FrequencyRenewal(FrequencyEmpirical)`, `freq_name='renewal'`
   (auto-registers, `:185-192`). Constructed DIRECTLY (bypasses factory dispatch, `__new__` `:204`)
   with keyword payload `wait_components` (list of `(Severity, wt, lb, ub, conditional)`) and
   `years`; stash payload before `super().__init__` (which calls `_build`). `_build` delegates to
   `_renewal.wait_count_pmf`, sets `freq_a = arange(len(pN))`, `freq_b = pN`, then inherited
   empirical validation. `freq_moms`/`freq_pgf` (→ Horner, dense) inherited. Diagnostics stored
   (`wait_bs`, `wait_log2`, `wait_p0`, `wait_defect`, `kmax`, `years`, `_renewal_bs_df`) for
   repr/info; add `convergence_check()`. `supports_zm=False`. Add to `__all__` (`:19-39`).
2. `_aggregate.py`:
   - `__init__` signature (`:1400-1412`): add `exp_years=0.0`, `exp_rate=0.0`, and 13 `wait_*`
     kwargs (`wait_name=''`, `wait_a=np.nan`, `wait_b=0.0`, `wait_mean=0.0`, `wait_cv=0.0`,
     `wait_loc=0.0`, `wait_scale=0.0`, `wait_xs=None`, `wait_ps=None`, `wait_wt=1.0`,
     `wait_lb=0.0`, `wait_ub=np.inf`, `wait_conditional=True`). `_spec` capture is automatic
     (`:1514-1517`). Name-collision vetting done: zero hits for `wait*`/`exp_years`/`renewal` on
     the public surface.
   - Before the `Frequency(...)` call at `:1587`: if `freq_name == 'renewal'` validate pairing
     (`exp_years > 0` and wait spec present; wait/years with any other freq_name → ValueError),
     build wait components with a small explicit broadcast loop (mirroring `:1784` but no exposure
     product, no layers, `exp_attachment=None`; `lb/ub` passed to `Severity` only when
     conditional — withheld for the defective post-pass otherwise), construct
     `FrequencyRenewal(...)`. Store `self.exp_years`/`self.exp_rate` near `:1598`.
   - **Arm-1 reorder fix**: move the `_en < 0` sentinel block (`:1801-1804`) ABOVE the premium/lr
     reconciliation (`:1796-1799`) so `years at rate` records per-component lr correctly.
     Regression-neutral: no existing path has `_en<0` together with `_pr>0` or `_lr>0`.
   - Carve-outs: `:699` count-support and `:4505` `_exact_discrete_window` →
     `in ('empirical', 'renewal')`; `_count_program` (`:3761-3775`) → renewal branch emits the
     realized `dfreq [0:kmax] [pN…]` program (the dfreq-conversion view above), NOT the wait
     clause. `tail.py:261-280` `FREQ_TAIL` → `'renewal'` (bounded, like empirical).
     `@`-inhomogeneous builtin scaling (`parser.py:1841-1848`) is a frequency no-op for renewal →
     warn. Add the `Aggregate` property surfacing `_renewal_bs_df`.
3. Tests `tests/test_renewal_agg.py` (end-to-end `build()`): the three-way equivalence above vs
   `poisson` (density ~1e-8); gamma-wait count sf vs regularized gamma; uniform T=1 → `k/(k+1)!`;
   `3 years … dwait [1]` → N≡3; `dwait [1 2]` T=2 vs enumeration (full-bucket rule); clustering
   `dwait [0 1] [.5 .5]` vs enumeration; defective splice-`!`; mixture wait; `wait sev.W`;
   `1 year at 500 rate` → `exp_premium==500`, lr = el/500, and a `pnl … inherit premium` smoke;
   renewal unit inside `port`; `approximate` MoM path; `create_frequency()` returns the realized
   dfreq whose pmf matches `frequency.freq_b` exactly; `_renewal_bs_df` structure test (one row
   per constraint, exactly one `selected`, exact-lattice row selected for commensurable dwait);
   `explain_validation()` clean (renewal theoretical moments are the pmf's own, so exact).

### Stage 4 — `decl_writer.py` round-trip

- `_render_freq` (`:325`): return `''` for `'renewal'` (wait clause renders in the freq slot;
  never add to `_FREQ_WORDS`).
- `_render_exposure` (`:357`): renewal branch FIRST (a `years at rate` spec carries `exp_premium`
  and would otherwise mis-render `premium at lr`): `{years} year|years[ at {exp_rate} rate]`.
- New `_render_wait`: view-dict rename `wait_*`→`sev_*`, reuse `_render_dist` prefixed `wait `;
  dhistogram → `dwait [xs] [ps][ !]`; wait label site.
- `_render_agg` (`:695-705`, and the pnl engine block `:759-767`): freq slot =
  `_render_freq(spec) or _render_wait(spec)`.
- Coverage automatic via `tests/test_decl_unparser.py` over the Stage-2 corpus lines.

### Stage 5 — docs & release hygiene

- Regenerate `docs/4_agg_language_reference/ref_include.rst` via the `grammar(add_to_doc=True)`
  hook — CAUTION: under the src/ layout it writes to `src/docs/...`; copy to the real `docs/`
  path. Keep clause-reference `.rst` prose in lockstep; note "docs pending rebuild" — do NOT run
  the full doc build.
- `dev/FEATURES.csv` row(s) for the renewal frequency / wait clause (+ introspection cross-check);
  `CHANGELOG.md` new `## 1.0.0a<next>` section; `pyproject.toml` version bump; `dev/TODO.md` entry
  `[Renewal-Frequency-Wait-Clause]`; move `dev/plan-sparre-a.md` → `dev/done/` at close.
- Gate: `uv run pytest -m 'slow or not slow'`.

## Verification (end-to-end)

1. Stage-0/1 unit suites green standalone (closed-form renewal benchmarks; pgf-dispatch parity);
   the discretize-kernel extraction lands green on the full suite before any wait code touches it.
2. Smoke: `build('agg T 10 years sev lognorm 100 cv 1 wait expon')` vs
   `build('agg P 10 claims sev lognorm 100 cv 1 poisson')` — compare `qd`, means, `q(0.99)`;
   `explain_validation()` clean.
3. `2 years dsev [1] wait expon splice [0 1.1] !` — count-only process, terminating tail visible.
4. `FrequencyRenewal.convergence_check()` on the benchmark cases shows O(h²) behavior.
5. Full parallel suite `-m 'slow or not slow'`; watch the peg/baseline suites for last-ulp Horner
   drift (inspect before re-capturing anything; call out in run summary).

## Risks / open items

- **Horner reordering vs frozen baselines**: summation-order changes can move densities ~1e-16;
  peg suites are the sentinel. Inspect, don't blind-recapture.
- **Tiny-μ waits** (μ ≪ T ⇒ huge kmax): sizing rules cap log2 at 24 and warn on kmax > ~1e5;
  documented cost, not a blocker.
- **Large collapsed negative mass** changes semantics materially (big P(W=0) batches) — warning
  states the resulting batch probability.
- Delayed/equilibrium (stationary) first wait is a documented future extension, not in scope.
- Ruin/Wiener-Hopf machinery (theory.md §10) explicitly out of scope.
