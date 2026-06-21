# TODO

> The v1.0 backlog, organized into **tracks** with mnemonic codes. The priorities
> table first, then the details per track, then post-v1.0 ideas. Snappy entries
> only — details live in the plan files (`dev/`, `dev/done/`) and the git log.
> **What's landed is in `CHANGELOG.md`** and is removed from here.
>
> **Phase tags:** `[A]` alpha = must finish before cutting `1.0.0b1`.
> `[B]` early beta = fine just after the alpha→beta cut, does not block it.
>
> **Last updated: 2026-06-21** — Cleanup pass: pruned all shipped/done items
> (they live in `CHANGELOG.md`) — the completed Steve-manual entries, Track M
> (bivariate, shipped a70–a80), Track F, and the done `W2` / `T1`; dropped `H9`
> and `F7`. Next focus: `N6` / `N8` (numerics + guards) and Track D docs toward
> `1.0.0b1`.

---

## Track codes (alphabetical)

| Code | Track |
|------|-------|
| **B** | Bugs & investigations |
| **D** | Docs & packaging |
| **H** | Hygiene (module organization & dependencies) |
| **N** | Numerics & pricing core (incl. signed / negative-x) |
| **T** | Tests (suite consolidation) |
| **W** | Windows & plotting |
| **β** | Pre-beta scaffold retirement (delete at the alpha→beta cut) |

## Priorities & dependencies

**Critical path (the spine):** the `N` spine is mostly shipped — `N2`, `N3`,
`N5`, `N5b` landed (a36–a57). What remains on the spine: **`N6`**
(validation-calc review) and **`N8`** (input guards), plus Track D docs.

| Status | ID | Item | Phase | Depends on |
|:--|----|------|:-----:|------------|
|   | N6 | Validation-calc review | A | — |
|   | N8 | Input guards & semantic consistency (3, ex-README) | A | — |
|   | H4 | Docstring style sweep → NumPy | A | — |
|   | H5 | `pedagogy` figure-generator migrations | B | — |
|   | B4 | ZT/ZM frequency broken + add shift helpers | A | — |
|   | W10 | Retire `recommend_bucket` | A | — (W9 shipped) |
|   | T2 | Rationalize tests / library coupling | A | — |
|   | T3 | Switcheroo `Port.Sample` regression case | B | — |
|   | D1 | New README body for stable-v1.0 audience | A | — |
|   | D2 | v1.0 Journey + statements of philosophy | A | — |
|   | D3 | Grammar reference from `decl.lark` | A | — |
|   | D4 | Tail-descriptor docs + tests | A | — |
|   | D5 | Doc gaps (errors, ZT/ZM, splice, site refs…) | A | — |
|   | D6 | API docstring coverage / rendering | A | H4 |
|   | D7 | Reinsurance case-study docs rewrite | B | (N2–N3 shipped) |
|   | D8 | PUNCHUP `pedagogy` + integrate docs | B | H5 |
|   | D11 | Sphinx docs → master `uber-library.bib` | B | — |
|   | D12 | Cheat-sheet tweaks once UI settles | B | — (hold to beta) |
| **— unclear / under-specified (revisit before scheduling) —** |
|   | W3 | Plot severity outside the agg window | B | — (approach TBD) |
|   | D9 | Reinsurance structure diagrams | B | — |
| **— β pre-beta scaffold retirement (do at the b1 cut; see Track β below) —** |
|   | β1 | Delete `_test_suite.agg` + `_test_suite2.agg` (SLY-parity scaffold) | A | b1 |
|   | β2 | Retire their dependents (snapshot, fixtures, config, scripts, docs) | A | β1 |
|   | β3 | Confirm `test_agg_libraries.py` is the surviving net for shipped libs | A | β1 |

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
  published *Aggregate* paper; make docs match the actual algo; the "all
  switches → config" sub-goal is **done** (`eps`/`noise` a30; `aliasing_ratio` /
  `exeqa_noise_floor` / `deficit_materiality` a83); fix the false-positive
  *agg-mean-error ≫ sev-error / aliasing* failure (try larger `bs`; revisit the
  too-tight tolerance — now an `aliasing_ratio` config edit). Independent of N8.
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

---

## Track B — Bugs & investigations  `[A]` (small, parallel)

- [ ] **B5 `[B]` Investigate `bs_describe` (the "bs_describe wart").** Flagged
  during `plan-line-to-unit` (author: "not sure about that!" — and "keep
  bothering me about it"). Scope TBD — review the `bs_describe`/`bs_explain`
  module functions in `distributions.py` (the `color=` workers behind
  `bs_description`/`bs_explanation`): purpose, the local-`line` accumulator, and
  whether they still earn their place / want reshaping. **Reconcile with
  `dev/done/plan-consistent-naming.md` §3** — the *deferred* bs-worker rename (the
  `bs_describe`/`bs_explain` verb workers shadow the noun properties by one
  letter; that plan parks renaming them to non-homonyms like `_format_bs_grid`).
  Do these two together so the wart isn't fixed twice or lost. *(new,
  2026-06-19; standing reminder — surface it periodically until scoped.)*
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

## Track W — Windows (range for output) & plotting

- [ ] **W3 `[B]`** Plot severity outside the aggregate window (#8) — inset,
  broken axis, or separate figure when grids don't overlap (`info` already warns).
  *(approach undecided.)*
- [ ] **W10 `[A]`** Retire `recommend_bucket` — replace the legacy one-shot
  sizer with a new (TBD) function that takes `log2` (and possibly `x_min`) as
  explicit arguments, then remove `recommend_bucket`. Its last real job (the
  infinite-variance fallback) is gone — `_bs_window` now raises
  `InfiniteVarianceError` instead of guessing a grid (a87) — so nothing left
  depends on it. See `dev/done/plan-univariate-bucket.md` (`[recommend-bucket]`).

---

## Track T — Tests (suite consolidation)  `[A]` (DOD: test suite trimmed)

> The crux is **how the current tests use / interact with the test libraries**
> (e.g. `conftest` parametrizing every line of `test_suite.agg`, the SLY
> snapshot regression) — consolidating the data is only half the job; the test
> code's coupling to it is the other half.

- [ ] **T2 `[A]`** Rationalize tests — needed vs no-longer-needed; untangle and
  re-wire how the suite *consumes* the single library without losing
  effectiveness (#51).
- [ ] **T3 `[B]`** Switcheroo harness `Port.Sample` regression case (#12) — guards
  the kappa-replacement path.

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

## Track β — Pre-beta scaffold retirement  `[A]` (do at the b1 cut)

> The `.agg` libraries were split at a71 into a **shipped** set (`examples`,
> `actuarial-severity-curves`, `decl-testers`, `cookbook`) and a **temporary**
> SLY-parity scaffold (`_test_suite.agg`, `_test_suite2.agg`). The scaffold has
> done its job (proving the Lark parser matches the retired SLY parser); delete
> it and its dependents at the alpha→beta cut. The surviving net is
> `tests/test_agg_libraries.py` (shipped libraries parse + cross-resolve), so
> deleting the scaffold loses no live coverage.

- [ ] **β1 `[A]` Delete the scaffold `.agg` files** — `src/aggregate/agg/_test_suite.agg`
  and `_test_suite2.agg`.
- [ ] **β2 `[A]` Retire the scaffold's dependents**, all of which only exist to
  exercise it:
  - SLY snapshot: `tests/data/expected_specs.json` + `tests/capture_sly_snapshot.py`.
  - `tests/test_decl_parser.py` (parametrizes every `_test_suite` line vs the snapshot).
  - `tests/test_splice_suite.py` (reads `_test_suite2.agg`).
  - `tests/conftest.py` — the `test_suite_lines` / `underwriter` fixtures (and any
    test still using the `underwriter` fixture).
  - `src/aggregate/config.py` `TEST_SUITE_FILENAME`; `Underwriter.test_suite_file`
    property; `Underwriter.interpret_file`'s default-to-test-suite behaviour.
  - `scripts/freeze_knowledge.py` / `scripts/bucket_baseline.py` —
    `DEFAULT_DATABASES = ("_test_suite",)`.
  - `src/aggregate/data/config.default.toml` — the `_test_suite` databases line.
  - `docs/4_dec_Language_Reference.rst` — the "Test Suite Programs" section
    (`literalinclude` of `_test_suite.agg`); repoint or drop.
- [ ] **β3 `[A]` Confirm the surviving net** — `tests/test_agg_libraries.py` covers
  the shipped libraries (optionally extend it from parse-only to a build smoke
  test). Decide whether `decl-testers.agg` needs its own permanent parse harness
  once the `_test_suite` snapshot is gone (it currently rides `test_decl_unparser`
  + the mirrored pytest cases).

---

## Related plans

- **Numerics program** (`dev/done/plan-numerics-0-meta.md` + `-1`…`-4`) → the
  N-track, **complete**: numerics-1/2/3 shipped (a55–a57 → N2/N3); numerics-4
  (windowed combine/bivariate) delivered via `plan-mv` MV-2/MV-3 (a72/a76).
- `dev/done/plan-config.md` (Phase 1 + Phase 2) → config, shipped a30 / a83.
- `dev/done/plan-mv.md` → bivariate firm-up (MV-1…7, shipped a70–a80). Absorbs &
  replaces the deleted `plan-multivariate-punchup.md`.
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
- [x] **DecL colorization** (#22) — **rejected 2026-06-19**
  (`dev/done/plans-considered-and-rejected.md`): payoff is aesthetic only and the
  headline wins are structurally weak (Jupyter `_repr_html_` doesn't fire from
  IPython tracebacks; Sphinx `pygments_style` is global). If docs identity ever
  becomes a priority, do only the minimal palette + Pygments-style slice.
- [ ] **F6 `[B]` Gross/ceded-premium reinsurance P&L** (#5) — extend `pnl` with
  both premium legs (`plan-pnl-premium.md` §9).
- [ ] extend reinsurance clauses to allow net of 50% of 500 xs 500 at .3 rol or 3000 ceded or .25 ros (rate on subject = quota share)
