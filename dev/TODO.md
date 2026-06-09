# TODO

> The v1.0 backlog, organized into **tracks** with mnemonic codes. Two tables
> first (tracks; priorities + dependencies), then the details, then post-v1.0
> ideas. Snappy entries only — details live in the plan files (`dev/`,
> `dev/done/`) and the git log. What's landed is in `CHANGELOG.md`.
>
> **Phase tags:** `[A]` alpha = must finish before cutting `1.0.0b1`.
> `[B]` early beta = fine just after the alpha→beta cut, does not block it. 
>
> **Last updated: 2026-06-08** — current version 1.0.0a47.

---

## Track codes (alphabetical)

| Code | Track |
|------|-------|
| **B** | Bugs & investigations |
| **D** | Docs & packaging |
| **F** | Features (approximation, pricing, config) |
| **H** | Hygiene (module organization & dependencies) |
| **M** | Multivariate |
| **N** | Numerics & pricing core (incl. signed / negative-x) |
| **T** | Tests (suite consolidation) |
| **W** | Windows & plotting |

## Priorities & dependencies

**Critical path (the spine):** `N1 → N2 → N3 → N4 → N5`, plus `N6` (validation)
alongside. **Start at `N1`.**

| Status | ID | Item | Phase | Depends on | Parallel-safe with |
|:--|----|------|:-----:|------------|--------------------|
|    | N1 | Aggregate `update` — base `density_df` calcs (+ neg-x) | A | — | H*, B*, F1–F3, D3, D5, W1 |
|    | N2 | Portfolio `update` — kappa / exa / `add_exa` (+ neg-x) | A | N1 | (same) |
|    | N3 | Portfolio apply-distortion calcs — columns, masses, `plot_twelve` | A | N2 | (same) |
|    | N4 | Bounds numerics read-through | A | N2 | runs after the spine settles |
| ✅ | N5 | `Portfolio.pricing_bounds` rewrite → `allocation_bounds` (a36) | A | N4 | — |
| ✅ | N5b | `PricingBounds` — cross-pricing ranges + Gini lens (a37) | A | N5 | — |
|    | N6 | Validation-calc review | A | — | independent of N1–N5 |
|    | N7 | Negative-x deferred follow-ups | B | N1–N3 | — |
|    | N8 | Input guards & semantic consistency (3, ex-README) | A | — | everything |
| ✅ | H1 | Relocate `make_ceder_netter` (a44) | A | — | everything |
| ✅ | H2 | Dedupe `make_var_tvar` (a44, non-issue) | A | — | everything |
| ✅ | H3 | Import-dependency audit (a44) | A | — | everything |
|    | H4 | Docstring style sweep → NumPy | A | — | everything |
|    | H5 | `pedagogy` figure-generator migrations | B | — | everything |
| ✅ | H6 | Underwriter database-loading rewrite | A | — | everything |
|    | B1 | `10000 xs 0 lognorm` "sum sev<1" warning | A | — | everything |
|    | B2 | "ugly histogram with spikes" (reconstruct) | A | — | everything |
| ✅ | F1 | `approximate` DecL keyword — MoM aggregates | A | — | H*, B*, N* |
| ✅ | F2 | G&H tilting DIY (pedagogy) | A | — | H*, B*, N* |
| ✅ | F3 | `pricing_at = P + Q` / Pentagon | A | — | H*, B* |
|    | F4 | Gross → Subject in `describe` | A | — | H*, B* |
|    | F5 | DecL `of` synonym | B | — | — |
|    | F6 | Gross/ceded-premium reinsurance P&L | B | — | — |
|    | F7 | PMIR best-bucket + manual kappa | B | — | pairs T3 |
|    | F8 | Config **Phase 2** (plotting/style, env, floors) | B | — | ties W1–W3 |
|    | W1 | Support-aware window bounds | A | — | N*, H*, B* |
|    | W2 | Window bounds for bivariate | B | W1 | ties M |
|    | W3 | Plot severity outside the agg window | B | — | — |
|    | M1 | Multivariate later stages (2–5) | B | — | — |
|    | M2 | Multivariate punch-up (sizing/coverage) | B | — | ties W2 |
|    | T1 | Merge the three `.agg` libraries into one | A | — | — |
|    | T2 | Rationalize tests / library coupling | A | T1 | after code churn |
|    | T3 | Switcheroo `Port.Sample` regression case | B | — | pairs F7 |
| ✅ | T4 | Knowledge-freeze regression harness (`scripts/freeze_knowledge.py`, a38) | B | — | N* (verifies them) |
|    | D1 | New README + split CHANGELOG | A | — | docs parallel |
|    | D2 | v1.0 Journey + statements of philosophy | A | — | docs parallel |
|    | D3 | Grammar reference from `decl.lark` | A | — | ready now |
|    | D4 | Tail-descriptor docs + tests | A | tail (shipped) | docs parallel |
|    | D5 | Doc gaps (errors, ZT/ZM, splice, site refs…) | A | — | ready now |
|    | D6 | API docstring coverage / rendering | A | H4 | docs parallel |
|    | D7 | Reinsurance case-study docs rewrite | B | N2–N3 | — |
|    | D8 | PUNCHUP `pedagogy` + integrate docs | B | H5 | — |
|    | D9 | Reinsurance structure diagrams | B | — | — |

**Status:** blank = untouched, X Done, P in Progress

**DOD mapping:** Docs updated → **D**; Numerical calculations checked +
negative-x methods → **N** (+ bug **B**); test suite trimmed → **T**.

---

## Track N — Numerics & pricing core  `[A]` (critical path)

> Negative-x is folded in here: it focuses on the same code and the same
> calculations. All "review numerics" work is **zero blast radius** (internal,
> no API change, ideally identical numbers — faster compute + trim unused
> `density_df` columns) **except** where the negative-x angle deliberately
> extends behaviour to signed support. **Separate the base `density_df` calcs
> from the apply-distortion calcs — both have a negative-x angle.**

- [ ] **N1 `[A]` Aggregate `update` — base `density_df` calcs.** Severity
  numerical-moments review (#26) + the main compute loop, the P / M / Q
  (Pentagon) elements (#27), with the **negative-x** angle on the Aggregate grid
  (#31). Includes **BUG: `.plot` draws a baseline from 0 to the first `xs`**,
  wrong for signed support *(new, 2026-06-05)*. The severity-moments and
  agg-loop pieces can split into parallel sub-tasks.
- [ ] **N2 `[A]` Portfolio `update`.** Computation of **kappas, exa / exeqa,
  `add_exa`** with the negative-x angle (#28 + #9). Trace
  `Portfolio.update → add_exa` end-to-end (every column, the `shift(-1)` tail,
  the `loss_max` blanking heuristic → wants a principled `F < k·eps` rule).
  Audit **what `plot_twelve` actually needs** (#29) — it constrains the column
  trim. **needs N1.**
- [ ] **N3 `[A]` Portfolio apply-distortion calcs.** Which columns;
  **handling masses (currently black-magic / a cluster)**; signed /
  `value_type` distortion pricing (`dev/plan-portfolio-neg-x-pricing.md`:
  `add_exa` column audit, distortion pricing, `value_type` consumption — signed
  books warn + fall back to F/S-only until this lands); `plot_twelve` impact;
  **trim unused `density_df` columns**. **needs N2.**
- [ ] **N4 `[A]` Bounds numerics read-through** (#30 + #10) — `bounds.py`
  (IME 2022, 513-point binary `s_grid`) end-to-end. **needs N2; prereq for N5.**
- [x] **N5 `[A]` `Portfolio.pricing_bounds` rewrite** (#32 + #11) — **done
  1.0.0a36** as `Portfolio.allocation_bounds` / `bounds.AllocationBounds`
  (`dev/done/plan-allocation-bounds.md`): exact convex-hull slicing of the
  `(TVaR_p, a_i(p))` curve, no `s_grid` interpolation needed. Bounded
  totals (`a=` / `p=`, linear NA, tail collapse) landed in the same
  release; the collapse idiom was factored into
  `Portfolio._collapsed_exeqa`, shared with `price(allocation='linear')`.
- [x] **N5b `[A]` `PricingBounds` — cross-pricing ranges** — **done 1.0.0a37**
  (`dev/done/plan-pricing-bounds.md`): given `X` priced to P, the range of the
  price of another risk `Y` over the consistent family, by slicing the convex
  hull of `(TVaR_p(X), TVaR_p(Y))` (union-of-breakpoints vertices, exact). The
  `AllocationBounds` hull/slice core was extracted into a shared `_HullEngine`
  base; TVaR-source adapters make either axis a risk or a closed-form pair, so
  the uniform reference gives the Gini mean-Kusuoka-level lens. Entry point
  `Portfolio.pricing_bounds`.
- [ ] **N6 `[A]` Validation-calc review** (#49) — audit the algorithm vs the
  published *Aggregate* paper; make docs match the actual algo; finish "all
  switches → config" (`eps`/`noise` already moved in a30); fix the false-positive
  *agg-mean-error ≫ sev-error / aliasing* failure (try larger `bs`; revisit the
  too-tight tolerance). Independent of N1–N5.
- [ ] **N7 `[B]` Negative-x deferred follow-ups** (#3) — two-sided deficit
  split; `ft.py` recentering helpers → call the core path; re-home
  `estimate_agg_window` → `utilities.py`; occ-reins on a signed severity grid;
  DecL keyword for `signed` / `value_type`. **needs N1–N3.**
- [x] **N7a `[A]` Signed-loss `pnl`** — **done 1.0.0a45**
  (`dev/done/plan-pnl-signed-severity.md`): a `pnl` whose loss severity is itself
  signed (`dsev` negative atom / `ssev`) now convolves the loss on its genuine
  signed grid before the affine relabel (was hard-coded to a 0-based grid →
  half the mass dropped, ±2¹⁵ empirical-moment garbage). `_bs_window` hands the
  loss origin to `update` for the affine case (0 for ordinary pnl, byte-for-byte);
  `_apply_agg_affine` now warns on material P&L-window mass drop. The single-unit
  case is fixed here; the **book**-level shared combine was fixed separately in
  **N7b** below. Knowledge-base freeze: all 146 objects match to 1e-12.
- [x] **N7b `[A]` Portfolio combine grid (`best_window`)** — **done 1.0.0a49**
  (`dev/done/plan-bucket-combine.md`): replaced the root-sum-square
  `best_bucket` combine (coarsened with unit count; ignored the integer lattice)
  with the resolution + span rule `bs = round_bucket(max(min_k bs_k, W_tot/N))`,
  shrinking `log2` for discrete books. Fixes the headline signed-combine bug
  (two `bs=1` units no longer give `bs=2`) and the all-integer `bs≈1/4096`
  mis-size. `best_bucket` kept as a comparison aid, flagged DELETE BEFORE BETA.
- [ ] **N7c `[A]` Heavy-tail per-unit window coverage** *(surfaced by N7b,
  2026-06-08)* — the per-aggregate `_bs_window` sizes its window to `1−1e-12`
  coverage (`WINDOW_NINES`), which for a heavy-tailed cat severity
  (`exp()·lognorm`, e.g. PIR `Hu`) yields an astronomically wide window
  (~1.9e10) and a coarse `bs` (~300000) **even standalone** — the aggregate mean
  lands in bucket 0 and the FFT moments are unusable without a manual `bs`. Cap
  or temper the coverage for genuinely heavy tails (or fall back to a limit /
  moment window). Independent of the combine; the combine (N7b) only surfaces it
  on the HuSCS ports. These cat books need an explicit `bs` today regardless.
- [ ] **N8 `[A]` Input guards & semantic consistency** *(ported from README,
  2026-06-06)* — three small correctness/guard items:
  - Treatment of zero `lb` is not consistent with attachment equals zero.
  - Flag attempts to use **fixed** frequency with a non-integer expected value.
  - Flag attempts to use **mixing** with an inconsistent frequency distribution.

  The latter two are input-validation guards; the first is a layer/attachment
  semantics fix. Pairs with **N6** (validation) but distinct (guards, not the
  validation-calc algorithm).

---

## Track H — Hygiene (module organization & dependencies)  `[A]` (parallel)

- [x] **H1 `[A]`** Relocate `make_ceder_netter` → `distributions` (a44) — hard move
  to its only consumer; its validator `_validate_reins_layers` moved alongside it
  (only `make_ceder_netter` and the tests use it); test import retargeted.
  `dev/done/plan-hygiene-1.md`.
- [x] **H2 `[A]`** Dedupe var/tvar (a44) — **non-issue, closed.** `utilities.make_var_tvar`
  is the single implementation; `Aggregate`/`Portfolio._make_var_tvar` are thin
  per-instance wrappers (caller/callee, not duplication). No code change.
  (Incidental: a likely member-crossing bug in `Aggregate.tvar_sev` was flagged
  for a separate look — see B-track.) `dev/done/plan-hygiene-1.md`.
- [x] **H3 `[A]`** Import-dependency audit (a44) — dropped unused runtime deps
  `cycler`, `psutil`, `ipykernel`, `jinja2`; deferred `IPython` to lazy imports
  inside `decl_pprint`/`agg_help` to cut ~1s off `import aggregate` (Pygments
  left eager — ~0ms and pulled by the decl_pygments lexer anyway).
  `dev/done/plan-hygiene-1.md`.
- [ ] **H4 `[A]`** Docstring sweep `iman_conover.py` / `moments.py` (and pockets
  elsewhere) Sphinx `:param:` → NumPy style; public surface first (#18). Feeds **D6**.
- [ ] **H5 `[B]`** `pedagogy.py` migrations: figure generators out of `ft.py` /
  `tweedie.py` so those stay API-focused (#19).
- [x] **H7 `[A]`** Public `reinsurance_*` methods → `reins_*` (a41) — surface made
  uniformly abbreviated (`reins_kinds`/`reins_description`/`reins_occ_plot`);
  `reins` canonized as the short form. `dev/done/plan-reins-rename.md`.
- [x] **H8 `[A]`** Consolidate fuzz removal → `utilities.remove_fuzz` (a42) — one
  vectorized two-sided helper replaces the per-cell `DataFrame.map` lambda (×2)
  and four `np.where(abs<eps)` copies; `ft` keeps `2*eps` via arg; MMSE stray
  `1e-16`→`eps`. Freeze/check on 146 objects: all match @1e-12.
  `dev/done/plan-remove-fuzz.md`.
- [x] **H9 `[A]`** `price_pentagon` on `Aggregate`/`Portfolio` (a43) — complete the
  pricing octet from a capital level (`p`/`a`) + one target (P/roe/lr/M/Q/pq) via
  the Pentagon; no new math. `solve_obj` gained `a=`; `price_ccoc` now a thin
  alias. `dev/done/plan-price-pentagon.md`.
- [x] **H6 `[A]` Underwriter database loading rewrite** — done in **1.0.0a32**
  + **1.0.0a35** (`dev/done/plan-databases.md`). The a32 rewrite plus the a35
  follow-on where a fresh `Underwriter` loads **no** databases (rather than
  `default`) — commit `637febca`. Loading made legible: dict-backed store (with a
  DataFrame view) + `source` provenance; one glob-aware resolver; honest
  `_loaded` flag; single `load(request=None)` verb + `reload` (reset to
  as-created), `resolve_databases` (preview), `available_databases` (discover);
  `databases` reports loaded paths; one preprocessing owner
  (`UnderwritingLexer.preprocess`); **`to_agg(path, pattern, kind, source)`**
  save/export (to user dir unless absolute). Renamed outright (no
  `read_database(s)` aliases).

---

## Track B — Bugs & investigations  `[A]` (small, parallel)

- [ ] **B1 `[A]`** `10000 xs 0 lognorm 120 cv 1.5` triggers a "sum sev < 1"
  warning — is it firing for the **agg, not the sev**? (#43)
- [ ] **B2 `[A]`** "ugly continuous histogram with small spikes" — recover what
  this was about; is it the `linear`/`nearest` discretization, and was it only
  ever a stats/moment concern? (#48 — *forgotten; reconstruct on sight*)
- [ ] **B4 `[A]`** `Aggregate.tvar_sev` (`distributions.py`) member-crossing: tests
  `self._var_tvar_function is None` but assigns `self._sev_var_tvar_function` and
  returns `self._var_tvar_function['tvar']` — the `sev`/non-`sev` members look
  swapped. Flagged during the H2 review (`plan-hygiene-1.md`); confirm and fix.
- [x] **B3 `[A]`** Zero-mean signed aggregate SD/var reported `NaN` (a40) — SD was
  rebuilt as `mean*cv` (nan at mean 0); now derived from `ex2 - mean^2` /
  `MomentWrangler.central`. `dev/done/plan-signed-sd.md`.
- [x] **B3b `[A]`** `Portfolio.describe` CV/SD mixing (a43, no bump) — spread
  choice now portfolio-wide: any signed unit forces SD across all unit blocks +
  total (via `Aggregate._describe(force_sd=...)`); total SD from `ex2-mean^2`.
  All-unsigned output byte-identical. NB surfaced a **pre-existing** signed-
  *combine* gap (FFT total var > independent sum of unit vars; identical under
  the old CV path) — tracked under **N2** / `plan-portfolio-neg-x-pricing.md`.

---

## Track F — Features (approximation, pricing, config)

- [x] **F1 `[A]` `approximate` DecL keyword — MoM aggregates** (#33) — **done
  1.0.0a47** (`dev/done/plan-approximate.md`). Design-time `approximate
  exact | sgamma | slognorm` directive: rewrites the aggregate at construction to
  a fixed-1-claim aggregate of a continuous severity fitted to its first three
  moments (shifted gamma / shifted lognormal; normal at the symmetric limit;
  reflected fit for left skew). No special compute path — `density_df`,
  validation, the `pnl` affine, and the `Portfolio` combine all just work.
  Incompatible with occurrence reinsurance; aggregate reinsurance rides along.
  `tests/test_approximate.py`. (The older idea of a tail-classifier-driven family
  pick inside the legacy `.approximate()` *method* is superseded by this explicit,
  self-documenting keyword.)
- [x] **F2 `[A]` G&H tilting DIY** (#34) — **done 1.0.0a46**
  (`dev/done/plan-gh-tilt.md`). The `ft` exponential **tilt** (Grübel–Hermesmeier
  aliasing reduction) removed in `6de20f2` is reborn as a self-contained
  **pedagogy** illustration, not a core method (the production convolution stays
  tilt-free — padding is the operational aliasing control). Added
  `pedagogy.tilted_aggregate_density` (+ `tilt_vector`, `gh_tilting_exhibit`) and
  rewired `docs/.../010_gh_example.rst`. `tilt=None` reproduces the untilted
  convolution byte-for-byte; `tests/test_pedagogy_tilt.py`.
- [x] **F3 `[A]` `pricing_at` = P + Q** (#63 + #64) — **done v1.0.0a31**
  (`dev/done/plan-pentagon.md`). Canonical `pentagon.py` contract
  (`PENTAGON_STATS`/`complete_pentagon`); all emitters routed through it;
  `analyze_distortion` audit fixed; additive `Portfolio.pentagon_at` →
  `Pentagon` object output.
- [ ] **F4 `[A]` Gross → Subject in `describe`** (#46) — relabel the first
  describe column to *Subject* for agg-only covers (no occ reins). Uses the
  `REINS_LABEL_*` constants.
- [ ] **F5 `[B]` DecL `of`** (#44) — `of` in place of / alongside `po` / `so`;
  maybe spell them out.
- [ ] **F6 `[B]` Gross/ceded-premium reinsurance P&L** (#5) — extend `pnl` with
  both premium legs (`plan-pnl-premium.md` §9).
- [ ] **F7 `[B]` PMIR best-bucket + manual kappa** (#47) — port the best-bucket
  and clever manual kappa calc. Pairs **T3**.
- [ ] **F8 `[B]` Config Phase 2** — `dev/plan-config.md`: `[plotting]` +
  `.mplstyle` override, the rest of the env matrix, and the numerics-pending
  floors (`aliasing_ratio`, `exeqa_noise_floor`, `ft_noise_floor`). Ties Track W.
- [x] **F9 `[A]` `Portfolio.price_stand_alone`** — **done v1.0.0a33**. Restored
  the deleted `stand_alone_pricing` (renamed to sort with `price`/`price_ccoc`):
  per-unit stand-alone pricing via `Aggregate.price` + total via `pricing_at`,
  canonical pentagon orientation (stats = columns), arg-checking, docstring.
  Unblocks the 10-min guide (clears **doc-fix G4**).

---

## Track W — Windows (range for output) & plotting

- [ ] **W1 `[A]`** Support-aware window bounds (#6) — use `fz.support()`
  endpoints; for finite frequency the support is exactly `[N·loc, N·ub]` (exact
  window, not a MoM guess).
- [ ] **W2 `[B]`** Window bounds for bivariate/multivariate per-axis sizing (#7).
  **needs W1**; ties Track M.
- [ ] **W3 `[B]`** Plot severity outside the aggregate window (#8) — inset,
  broken axis, or separate figure when grids don't overlap (`info` already warns).
- [x] **W4 `[A]`** Signed (P&L) Lee-plot artifact (a39) — `Aggregate.plot`'s
  discrete zero-anchor row set `loss=0`, drawing a spurious vertical segment to
  the first point on signed support; anchor `loss` now equals its index.
- [x] **W5 `[A]`** Window-aware plot x-limits (a50) — `Aggregate`/`Portfolio`
  `_limits(stat='range')` and the discrete left edge now key on the grid origin
  (ordinary 0-based and signed unchanged; a thin-tailed window `x_min > 0`
  anchors at the realised support min). See `dev/done/plan-hygiene-2.md`.
- [ ] **W6 `[B]`** Plot: revisit the severity overlay vs the aggregate output
  window — the severity is drawn on its own `sev_density_df` grid, which need not
  match the (windowed) aggregate axis, so for a non-zero-origin / thin-tailed
  aggregate the severity curve may fall outside or be mis-scaled against the
  windowed aggregate. Decide *whether* and *how* to show it (overlay on the
  aggregate window, inset, or separate). Surfaced by W5; pairs with **W3**.

### Ergonomic tweaks (a39)

- [x] Keyword-only `Underwriter(*, …)` (kills `Underwriter('db')` silently naming
  the underwriter); `repr` gains a `requested` line for `self._request`.
- [x] New `density` property on `Aggregate`/`Portfolio` =
  `density_df.query('p_total > 0')` (the live support).

---

## Track M — Multivariate `[B]` (early beta)

- [ ] **M1 `[B]`** Multivariate later stages (#4) — Stage 2 `t` copula; Stage 3
  reporting/plot polish; Stage 4 ≥3-variate `rfftn` shared frequency; Stage 5
  `MultivariatePortfolio` / `netceded` DecL form (`dev/done/plan-multivariate.md`).
- [ ] **M2 `[B]`** Multivariate punch-up — axis sizing / coverage reconciliation
  (`dev/plan-multivariate-punchup.md`). Ties **W2**.

---

## Track T — Tests (suite consolidation)  `[A]` (DOD: test suite trimmed)

> The crux is **how the current tests use / interact with the test libraries**
> (e.g. `conftest` parametrizing every line of `test_suite.agg`, the SLY
> snapshot regression) — consolidating the data is only half the job; the test
> code's coupling to it is the other half.

- [ ] **T1 `[A]`** Merge the three `.agg` libraries (`test_suite`,
  `test_suite2`, `test_decl`) into **one** without breaking tests (#50).
- [ ] **T2 `[A]`** Rationalize tests — needed vs no-longer-needed; untangle and
  re-wire how the suite *consumes* the single library without losing
  effectiveness (#51). **needs T1.**
- [ ] **T3 `[B]`** Switcheroo harness `Port.Sample` regression case (#12) — guards
  the kappa-replacement path. Pairs **F7**.

---

## Track D — Docs & packaging

- [ ] **D1 `[A]`** New `README.md` for the stable-v1.0 audience (what / who /
  install / one-liner DecL) (#39, #13, #14).
  - [x] **Split done (2026-06-06):** `README.rst` → markdown `README.md`
    (intro / install / getting-started / badges) **+ `CHANGELOG.md`** (full
    version history, keyed by version); `dev/PROGRESS.md` removed, subsumed by
    `CHANGELOG.md` + git log; `pyproject.toml` `readme` repointed.
  - [ ] Remaining: rewrite the README body for the stable-v1.0 audience (it is
    currently the verbatim moved content).
- [ ] **D2 `[A]`** v1.0 intro / "Journey" page **+ statements of philosophy**
  (user manages logging / warnings / matplotlib; the distribution **is** `p_i`
  at `x_i`, no jump detection; `qd` is the doc-only fixed-font exception); cover
  the v1.0 shift (linear allocation default, bounded detection, forwards-`S`,
  pentagon columns, `DefectiveDistributionWarning`) (#15 + DOD).
- [ ] **D3 `[A]`** Grammar reference from `decl.lark` / `grammar(add_to_doc=True)`
  — `docs/4_agg_language_reference/` still describes the SLY-era grammar (#17).
  *No code dependency — ready now.*
- [ ] **D4 `[A]`** Tail-descriptor docs **+ tests** (bounded / log-concave /
  super-exp / exp / sub-exp for freq **and** sev) and the bounded/unbounded
  indicator (#41, #42). Aligns with tail Phase 2.
- [ ] **D5 `[A]`** Doc gaps: custom errors (#58); syntax checker / better error
  reporting (#59); stale "site" database refs (#60); ZT/ZM zero-truncation/
  modification (#61); splice examples (#62). *No code dependency — ready now.*
- [ ] **D6 `[A]`** API docstring coverage / rendering — every public function/
  class carries a NumPy-style docstring that renders in the API reference (the
  "Docs updated" DOD bullet, doc side of **H4**).
- [ ] **D7 `[B]`** Reinsurance case-study docs rewrite — `bahnemann`,
  `enterprise risk`, `other_misc`: rebuild per-layer exhibits from
  `reins_stats_df`, verify vs published (#16). **needs N2–N3** (stable numerics).
- [ ] **D8 `[B]`** PUNCHUP `pedagogy` and integrate with docs; possible minor
  renamings (#40). **needs H5.**
- [ ] **D9 `[B]`** Reinsurance structure diagrams (PMIR code?) (#45).

---

## Related plans

- `dev/plan-portfolio-neg-x-pricing.md` → **N3**.
- `dev/plan-config.md` (Phase 2) → **F8**.
- `dev/plan-multivariate-punchup.md` → **M2**.
- `dev/done/` → shipped: tail-thickness (**F1**, **D4**), config Phase 1,
  pentagon pricing contract (**F3**, `plan-pentagon.md`), database loading
  (**H6**, `plan-databases.md`), multivariate stages 0–1, etc.

---

## Post-v1.0 ideas

- [ ] **Multi-resolution portfolio combine** (#20) — the real fix for the coarse
  shared-`bs` deficit (a22 residual #4): compute each unit on its own `bs`,
  decimate onto the shared grid before the Fourier product. Deficit accepted /
  surfaced for now.
- [ ] **General premium/loss algebra in DecL (v2.0)** (#21) — constant
  aggregates, full aggregate arithmetic (`agg.A - agg.B`, `agg.A + c`); `pnl`
  covers the common case for v1.0.
- [ ] **DecL colorization** (#22) — design parked 2026-05-27
  (`dev/tentative-plan-decl-colorization.md`); payoff mostly Sphinx-docs
  identity. Wait for a clearer use case.
