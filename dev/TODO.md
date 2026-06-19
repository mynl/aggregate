# TODO

> The v1.0 backlog, organized into **tracks** with mnemonic codes. The priorities
> table first, then the details per track, then post-v1.0 ideas. Snappy entries
> only — details live in the plan files (`dev/`, `dev/done/`) and the git log.
> **What's landed is in `CHANGELOG.md`** and is removed from here.
>
> **Phase tags:** `[A]` alpha = must finish before cutting `1.0.0b1`.
> `[B]` early beta = fine just after the alpha→beta cut, does not block it.
>
> **Last updated: 2026-06-19** — Cleanup pass: all shipped items pruned (they
> live in `CHANGELOG.md`); priorities table reconciled against the track sections
> and rebuilt; under-specified items moved to a flagged tail group. Current focus:
> **Track M** (bivariate firm-up, `dev/plan-mv.md`, MV-1…7) — beta-blocking.

---

## Track codes (alphabetical)

| Code | Track |
|------|-------|
| **B** | Bugs & investigations |
| **D** | Docs & packaging |
| **F** | Features (approximation, pricing) |
| **H** | Hygiene (module organization & dependencies) |
| **M** | Multivariate → bivariate |
| **N** | Numerics & pricing core (incl. signed / negative-x) |
| **T** | Tests (suite consolidation) |
| **W** | Windows & plotting |

## NEW STEVE MANUAL ENTRIES

H: Portfolio class should use unit not line. All refs to line replaced with unit! **carefully!**
H: build.knowledge source col: just store the db name and not whole path in source. Yes, ambiguous. Prefix ~/ if from user's .aggregate store, full path if not built-in or home dir.
N: distributions with no variance should refuse to estimate bs. how is that being done ATM? I think in the past it was an error. Now you can package hints with the program there are no excuses for not providing bs,log2.

N/D: find examnple where padding has an impact





## Priorities & dependencies

**Critical path (the spine):** the `N` spine is mostly shipped — `N2`, `N3`,
`N5`, `N5b` landed (a36–a57). What remains: **`N6`** (validation-calc review)
and **`N8`** (input guards). **Current active block: `M`** (bivariate firm-up)
— beta-blocking.

| Status | ID | Item | Phase | Depends on |
|:--|----|------|:-----:|------------|
|   | N6 | Validation-calc review | A | — |
|   | N8 | Input guards & semantic consistency (3, ex-README) | A | — |
|   | H4 | Docstring style sweep → NumPy | A | — |
|   | H5 | `pedagogy` figure-generator migrations | B | — |
|   | B4 | ZT/ZM frequency broken + add shift helpers | A | — |
|   | W2 | Window bounds for bivariate | B | M |
|   | W10 | Retire `recommend_bucket` | A | — (W9 shipped) |
|   | M | **Bivariate firm-up (MV-1…7, `plan-mv.md`)** — beta-blocking | A | — |
|   | T1 | Merge the three `.agg` libraries into one | A | — |
|   | T2 | Rationalize tests / library coupling | A | T1 |
|   | T3 | Switcheroo `Port.Sample` regression case | B | — |
|   | D1 | New README body for stable-v1.0 audience | A | — |
|   | D2 | v1.0 Journey + statements of philosophy | A | — |
|   | D3 | Grammar reference from `decl.lark` | A | — |
|   | D4 | Tail-descriptor docs + tests | A | — |
|   | D5 | Doc gaps (errors, ZT/ZM, splice, site refs…) | A | — |
|   | D6 | API docstring coverage / rendering | A | H4 |
|   | D7 | Reinsurance case-study docs rewrite | B | (N2–N3 shipped) |
|   | D8 | PUNCHUP `pedagogy` + integrate docs | B | H5 |
|   | D10 | Single keyword source of truth (`decl.lark` → mirrors) | B | — |
|   | D11 | Sphinx docs → master `uber-library.bib` | B | — |
|   | D12 | Cheat-sheet tweaks once UI settles | B | — (hold to beta) |
| **— unclear / under-specified (revisit before scheduling) —** |
|   | H9 | Public DataFrame members present (`None`) before compute | A | — (needs scoping) |
|   | F7 | PMIR best-bucket + manual kappa | B | — |
|   | W3 | Plot severity outside the agg window | B | — (approach TBD) |
|   | D9 | Reinsurance structure diagrams | B | — |

**Status:** blank = untouched, X Done, P in Progress.

**DOD mapping:** Docs updated → **D**; Numerical calculations checked +
negative-x methods → **N** (+ bug **B**); test suite trimmed → **T**.

---

## Track N — Numerics & pricing core  `[A]` (critical path)

> Negative-x is folded in here: it focuses on the same code and the same
> calculations. All "review numerics" work is **zero blast radius** (internal,
> no API change, ideally identical numbers — faster compute + trim unused
> `density_df` columns) **except** where the negative-x angle deliberately
> extends behaviour to signed support. **Separate the base `density_df` calcs
> from the apply-distortion calcs — both have a negative-x angle.** (The
> apply-distortion side, `N2`/`N3`/`N5`, shipped a36–a57.)

- [ ] **N6 `[A]` Validation-calc review** (#49) — audit the algorithm vs the
  published *Aggregate* paper; make docs match the actual algo; finish "all
  switches → config" (`eps`/`noise` already moved in a30); fix the false-positive
  *agg-mean-error ≫ sev-error / aliasing* failure (try larger `bs`; revisit the
  too-tight tolerance). Independent of N8.
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

- [ ] **H4 `[A]`** Docstring sweep `iman_conover.py` / `moments.py` (and pockets
  elsewhere) Sphinx `:param:` → NumPy style; public surface first (#18). Feeds **D6**.
- [ ] **H5 `[B]`** `pedagogy.py` migrations: figure generators out of `ft.py` /
  `tweedie.py` so those stay API-focused (#19).
- [ ] **H9 `[A]` Public DataFrame members present (`None`) before compute** —
  *needs scoping* (deferred at a52). Goal: reading a documented `*_df` member on
  a freshly constructed (not-yet-`update`d) object never raises `AttributeError`.
  Caveats found during review: the scope list in the old plan named several
  non-members (`reins_audit_df`, public `reins_df`/`report_df`/`statistics_df`/
  `bs_window_df` don't exist); the real properties (`density_df`,
  `sev_density_df`) already raise an *informative* `ValueError('Update … first')`,
  not `AttributeError`, and `reins_*_df` already return `None`; `augmented_df` is
  a parameterised method (can't return `None`). So the genuine work is narrow —
  enumerate the real members per class (`Aggregate`/`Portfolio`/`Distortion`/
  `Underwriter`) and decide informative-`ValueError`-vs-`None` before touching
  anything.

---

## Track B — Bugs & investigations  `[A]` (small, parallel)

- [ ] **B4 `[A]`** **Zero-truncated / zero-modified frequency is broken.**
  `poisson zt` raises `function value at x=0.0 is NaN; solver cannot continue`
  for every parameterization; `zm` builds but the *semantics* are wrong. The
  current design takes the **post-modification** mean and inverts to find the
  base mean — the "figuring" step is fragile and the wrong contract. Redesign:
  the user inputs the **un-truncated/un-modified base mean** and we apply the
  ZT/ZM shift forward (no solver). **Ship helper functions** that do the
  mean/parameter shift for the user (both directions, documented). Until then
  the two ZM/ZT examples are commented out in `examples.agg`. Pairs with D5
  (ZT/ZM docs). *(new, 2026-06-17)*

---

## Track F — Features (approximation, pricing)

- [ ] **F7 `[B]` PMIR best-bucket + manual kappa** (#47) — port the best-bucket
  and clever manual kappa calc. Pairs **T3**. *(vague — recover the PMIR code
  first.)*

---

## Track W — Windows (range for output) & plotting

- [ ] **W2 `[B]`** Window bounds for bivariate/multivariate per-axis sizing (#7).
  Ties Track M (consumed by `plan-mv.md` §5).
- [ ] **W3 `[B]`** Plot severity outside the aggregate window (#8) — inset,
  broken axis, or separate figure when grids don't overlap (`info` already warns).
  *(approach undecided.)*
- [ ] **W10 `[A]`** Retire `recommend_bucket` — replace the legacy one-shot
  sizer with a new (TBD) function that takes `log2` (and possibly `x_min`) as
  explicit arguments, then remove `recommend_bucket`. W9's honest-truncation path
  (`[use-selection]` item 2: accept truncation, no-normalize, warn) removed its
  last real job (the infinite-variance fallback), so this follows the shipped W9.
  See `dev/done/plan-univariate-bucket.md` (`[recommend-bucket]`).

---

## Track M — Multivariate → **bivariate** `[A]` (now beta-blocking)

> **Re-scoped & promoted to `[A]` (2026-06-19, `dev/plan-mv.md`).** The "firm up
> multivariate" block is the **last major work before `1.0.0b1`** and now *blocks*
> the beta (author: no post-beta API scope creep — land the whole bivariate
> surface in v1.0). `dev/plan-mv.md` is the fully-staged execution plan (MV-1…7).
> Decisions: rename `multivariate`/`mv` → `bivariate`/`bv` (dropped outright,
> no synonyms); ≥3-variate `rfftn` **killed** (use Iman–Conover + switcheroo);
> **`t` copula dropped** (flaky + footgun); add shuffle-of-Min + a `clash`
> statement; sizing reframed to *measure-don't-guess* (`balanced_window` on the
> realized marginals, `update(log2=)` budget); netceded → three occurrence
> view-pairs (`netceded`/`grossceded`/`grossnet`).

- [ ] **M (the block) `[A]`** — execute `dev/plan-mv.md` stages **MV-1…MV-7**
  (`balanced_window`+`focus` → measure-don't-guess copula sizing → netceded
  one-bs sizing → reporting surface → netceded view-pairs → shuffle-of-Min+clash
  → rename to bivariate). Each a version bump; all seven block `1.0.0b1`.
  Execution cadence: one stage per iteration, with a review + commit between each.
  - [x] **MV-1** `[a70]` — `balanced_window(ser, p, bs=None)` (utilities) +
    `Aggregate.focus(p)`; pure 1-D, no bv change. `tests/test_balanced_window.py`.
  - [x] **MV-2** `[a72]` — measure-don't-guess copula axis sizing (deleted
    `_size_axis`; standalone-marginal + `balanced_window`, `update(log2=)` total
    budget, `[multivariate].total_log2`) + signed 2-D compound (`i0`/`j0`
    wrap-and-roll in `update_work`). 54%-deficit book → <1e-6.

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
  install / one-liner DecL) (#39, #13, #14). The `README.rst`→`README.md` +
  `CHANGELOG.md` split is done (2026-06-06); **remaining:** rewrite the README
  body for the stable-v1.0 audience (currently the verbatim moved content).
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
  `reins_stats_df`, verify vs published (#16). N2–N3 numerics now stable.
- [ ] **D8 `[B]`** PUNCHUP `pedagogy` and integrate with docs; possible minor
  renamings (#40). **needs H5.**
- [ ] **D10 `[B]`** Single keyword source of truth — derive `decl_pygments.AggLexer`,
  `parser_errors._TERMINAL_LABELS`, and the web app's `decl-keywords.json` from
  the `decl.lark` terminals (five hand-maintained mirrors today). Independent of
  the unparser (`decl_writer` reuses `AggLexer` as-is, shipped a53); ties the
  parked DecL colorization (#22).
- [ ] **D11 `[B]`** Transition the Sphinx docs' bibliography to the master
  `C:/s/TELOS/Biblio/uber-library.bib` (per the CLAUDE.md "Citations and
  bibliography" standing order, added 2026-06-11). The docs currently use an
  older biblio file with slightly different keys — sweep the `.rst` citations,
  map old keys → uber keys, and point the docs' bibtex config at the master
  file (or an exported subset) so the docs and the `dev/*.qmd` artifacts cite
  identically.
- [ ] **D12 `[B]` Cheat-sheet tweaks once the UI settles** — the six class / DecL
  cheat sheets (`cheat-sheets/`) were rebuilt for the v1.0 API on the
  tectonic + `make.ps1` / `combine.ps1` build with auto-`\aggversion` stamping
  from `pyproject.toml` (see the dir's `README.rst` "instructions for Claude").
  **Revisit at the alpha→beta cut (`1.0.0b1`)**, when the API has stabilized:
  re-run `introspect` per class, reconcile any renames/removals, and apply
  pending wording/layout tweaks (incl.\ whether to densify DecL pages 2–3).
  *(flagged 2026-06-18; author wants this held until the first beta.)*
- [ ] **D9 `[B]`** Reinsurance structure diagrams (PMIR code?) (#45).
  *(under-specified — confirm source/scope.)*

---

## Related plans

- **Numerics program** (`dev/plan-numerics-0-meta.md` + `-1`…`-4`) → the N-track;
  numerics-1/2/3 shipped (a55–a57 → N2/N3); numerics-4 (windowed combine/
  bivariate) → W2/M-track.
- `dev/plan-config.md` (Phase 2) → **F8**.
- `dev/plan-mv.md` → **Track M** (bivariate firm-up, MV-1…7). Absorbs & replaces
  the deleted `plan-multivariate-punchup.md` (former M2).
- `dev/done/` → shipped plans (tail-thickness, config Phase 1, pentagon, database
  loading, bucket-window 1A/1P, allocation/pricing bounds, decl-unparser, …).

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
- [ ] **F6 `[B]` Gross/ceded-premium reinsurance P&L** (#5) — extend `pnl` with
  both premium legs (`plan-pnl-premium.md` §9).
- [ ] extend reinsurance clauses to allow net of 50% of 500 xs 500 at .3 rol or 3000 ceded or .25 ros (rate on subject = quota share)
- [ ] **F8 `[B]` Config Phase 2 for graphics** — `dev/plan-config.md`: `[plotting]` +
  `.mplstyle` override, the rest of the env matrix, and the numerics-pending
  floors (`aliasing_ratio`, `exeqa_noise_floor`, `ft_noise_floor`). Ties Track W.
