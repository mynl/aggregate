# Task — maintain `docs/new_material/features.qmd`

> **What this is.** A *repeatable task*, not a one-shot plan. Invoke it by saying
> **"execute task-features"** (or "bring features up to date"). Each run
> reconciles `docs/new_material/features.qmd` against the current development state and adds /
> updates whatever is missing, then verifies the whole notebook executes clean.
>
> **Resume model.** There is no "step N done" — every run starts from the
> coverage ledger inside `features.qmd` and the `CHANGELOG.md` feed, computes the
> gap, and closes it. The first run is a long job (the whole 1.0.0a series); every
> run after that is a small incremental top-up tied to the latest `a*` / `b*`
> release.
>
> **Ownership and edit flow.** `features.qmd` is the **single source of truth
> for its own content**, and the author edits it directly — prose, examples,
> cast, structure. Author edits are authoritative and survive runs. A task run
> is **additive and repair-only**: it (a) adds sections / ledger rows for new
> CHANGELOG items, (b) repairs cells broken by API drift, (c) keeps the ledger
> consistent. It never rewrites or deletes author prose; if a restructure seems
> warranted, it is *proposed* in the run summary, not performed. Git is the
> reconciliation layer — the author commits their edits, the run's changes
> arrive as a reviewable diff. This spec governs the *process*; the doc governs
> the *content*.

---

## 1. Purpose and audience

`features.qmd` is the **user-side, topical companion to `CHANGELOG.md`**.

- `CHANGELOG.md` is **developer-facing and chronological** — "what landed, in
  what order, with what breaking-change notes."
- `features.qmd` is **user-facing and topical/educational** — "here is what
  `aggregate` can now do that 0.30.1 could not, organised by subject, shown by
  worked example." It is the answer to a user who asks *"what's new since the last
  release I used, and how do I use it?"*

It covers **everything added on `REFACTOR`, i.e. the entire `1.0.0a*` series since
0.30.1** (and, going forward, the `1.0.0b*` series). It is a Quarto-markdown
notebook the author opens in **Jupyter Lab** (jupytext pairs `.qmd` ↔ notebook;
`jupytext` ships in the `notebook` extra). It lives in `docs/new_material/`
awaiting integration; it is a project artifact, **not** part of the
Sphinx/readthedocs tree *today*.

**Destiny and scale.** This document is expected to graduate into the Sphinx docs
at release as a **"What changed in 1.0 vs 0.30"** chapter. Write it to that
standard from the start: match the level of detail of the existing docs (the
10-minutes guide and the feature chapters are the register to aim for), and
expect a substantial document — **tens of pages** when rendered. "Prose is
tight" (§2.1) governs density, not total length; there is a lot of material and
it should all be here.

---

## 2. The deliverable: shape of `features.qmd`

### 2.1 Format and conventions

- **Quarto markdown** (`.qmd`) with a YAML front-matter header and
  ```` ```{python} ```` code cells. jupytext reads it as a notebook.
- Front matter (kernel `python3`, a title, `jupyter: python3`). Keep it minimal.
- **Runnable source only — no embedded outputs are committed.** The notebook is a
  live teaching document the author runs in Jupyter Lab; output (DataFrames,
  plots) stays out of git so diffs are clean and never churn on numeric drift.
  The task's job is to guarantee every cell *runs clean* (§4), not to freeze its
  output.
- Display with the library's own quick-display: `from aggregate import build, qd`
  then `qd(obj)` / `qd(df)`. Use `obj.plot()` where a picture earns its place.
- **US spelling** in all prose and comments. `reins` is the canonical short form
  for "reinsurance"; lean to full words for any *new* prose identifiers but keep
  house abbreviations (`sev`, `occ`, `agg`, `freq`, `cv`, `bs`).
- Prose is tight: no filler, one or two illustrative examples per feature, the
  "why it matters" in a sentence. For depth, calibrate to the existing Sphinx
  docs (the 10-minutes guide is the register) — this doc is headed there (§1).
  Tight density, not short length: the full document will run to tens of pages.
- **Citations** follow the standing order in `CLAUDE.md` ("Citations and
  bibliography"): `bibliography:` + `csl:` lines in the YAML, `@Key` cites
  found by searching `C:/s/TELOS/Biblio/uber-library.bib` (never invented),
  and a `## References` / `::: {#refs}` block at the end.

### 2.2 Top-level section layout

> **Provisional.** This layout is a *starting hypothesis*, not a fixed contract.
> The real grouping should fall out of the first full pass through the changes —
> expect sections to split, merge, or reorder as the material dictates. Revise
> this list (and this spec) once the first run has explored the surface.

```
0.  YAML front matter + one-paragraph intro (what this doc is; points at CHANGELOG)
1.  Coverage ledger                       ← the contract table (§2.3)
2.  The cast of examples                  ← canonical objects, built once (§2.4)
3.  New DecL elements                     ← examples first
4.  Better parse errors (Lark report)
5.  Programs as text                      ← decl_writer: format_program, canonical pprogram / to_agg (a53)
6.  The reporting quartet                 ← info / describe / stats_df / density_df
7.  Grids, buckets & windows              ← bs/log2 sizing, output windows (a49/a51), dsev_bucket, hints{}
8.  Tail-thickness classification
9.  Pricing & the pentagon
10. Pricing & allocation bounds
11. Configuration (config.toml)
12. Underwriter & persistence
13. Pedagogy helpers
14. Under the hood                        ← appendix: internal changes, named only
```

**"Examples first, then the new properties"** (author's instruction): §2 builds a
small cast of objects up front; the feature sections (§3 on) *reuse* those objects
to demonstrate new properties and methods rather than minting a throwaway object
per feature.

### 2.3 The coverage ledger (the incremental-update contract)

Immediately after the intro, `features.qmd` carries a **coverage ledger** — a
markdown table (inside a raw markdown cell) that maps every CHANGELOG item to its
treatment in the doc:

| Feature | Since | Section | Example object |
|---|---|---|---|
| `ssev` / signed severity | a21 | New DecL elements | `signed_s` |
| `pnl` premium | a23 | New DecL elements | `signed_s` |
| `multivariate` / copula / netceded | a24 | New DecL elements | `mv_copula`, `mv_indep`, `netceded` |
| `price_pentagon` | a43 | Pricing & the pentagon | `book` |
| fuzz consolidation | a42 | Under the hood | — |
| … | | | |

This table **is** the maintenance state. The contract is **morally complete,
not slavishly so**: every *major point* in `CHANGELOG.md` must be accounted for
by a row, but the row granularity is a judgment call. The default unit is one
row per logical feature — usually a `###` sub-heading, but merge trivially
related sub-items into one row, and split a mega-entry (the early `a8`/`a9`
"in progress" entries, the `a17` core refactor) into the several features it
actually contains. The test is a reader's, not an auditor's: *could a returning
0.30.1 user find every change that matters to them via this table?* A row whose
**Section** is "Under the hood" and **Example** is "—" is an internal change
that is named but not exampled (§2.5). The reconciliation in §3 is: *CHANGELOG
items not accounted for by any ledger row are the work list.*

### 2.4 The cast of examples

A small set of objects built once in §2 and **heavily leveraged** thereafter.
There is real art in choosing them: each should be the *minimal* program that
exercises a distinct piece of DecL surface, and as many feature sections as
possible should borrow an existing cast member rather than mint a throwaway.
Add a new member only when none of the existing ones can illustrate a feature,
and say why in a comment.

> **The cast's canonical home is `features.qmd` §2 itself** (since the first
> run landed, 2026-06-11). This section keeps the *design brief* only —
> clustering, minimality, the reuse rules above — not a second copy of the
> code. To change the cast, edit `features.qmd` §2 directly; there is no
> copy-through step.
>
> **The run has license to edit the cast in `features.qmd`** — fix API drift
> and parse errors, tidy names and comments, and tune programs to be more
> educational / illustrative — reporting the edits in the run summary.
> Compositional judgment calls (drop a member, change what a member is *for*)
> deserve a note to the author but need not block the run.

Members are **clustered** (`n.a`, `n.b`, …) so closely-related examples — and the
features they illustrate — sit together. Renumber freely as the set evolves.
The current roster (see `features.qmd` §2 for the live definitions): `simple`,
`defective`, `dice`, `signed_d`, `signed_s`, `pnl`, `mv_indep`, `mv_copula`,
`reins`, `netceded`, `mix_sev`, `mix_exp`, `mix_both`, `book`, `book_w_re`,
`big`, `bigex`.

#### Pending dependencies (do not block the whole run on these)

**Currently none.** The original three (the `mv_*` splice crash, the `bigex`
window infeasibility, the wrong `approximate` info note) were resolved by a48,
a51 and a50 respectively; the full cast was smoke-tested green against a57 on
2026-06-11.

The mechanism stands for future use: when a cast member depends on work that has
not landed, **tag its cell pending** and **exclude it from the must-pass
verification gate (§4)**; drop the tag when the dependency ships so the cell
becomes required again.

### 2.5 Internal changes — the "Under the hood" appendix

Pure-internal / dev-only changes (no new user-facing API) are **named in a short
appendix**, one line each with the version and a pointer to CHANGELOG — *not*
exampled. This honours "cover everything since 0.30.1" without contriving
examples for plumbing. Examples of internal items: fuzz consolidation (a42),
module organisation / dropped deps / lazy IPython (a44), the knowledge-freeze
harness (a38), config-file *plumbing* (a30 — though its user-facing surface,
`show_settings` / layering, **is** exampled in §9).

---

## 3. The procedure (each `execute task-features` run)

### 3.0 Pre-flight

- Confirm the worktree is on `REFACTOR`.
- `uv sync --all-extras` (the verification in §4 needs a Jupyter kernel;
  `ipykernel` arrives transitively via `jupyterlab`). **Never a single
  `--extra`**: `uv sync` is an exact sync and prunes the extras you did not
  select, which deletes the `massive` / `viz` / `numba` packages and breaks the
  bivariate suites. `UV_LINK_MODE=copy` is set for the harness; in a plain shell
  `$env:UV_LINK_MODE = "copy"`.
- Run against **`.venv`**, not `.doc-venv`. An ambient
  `UV_PROJECT_ENVIRONMENT=.doc-venv` makes a bare `uv run` resolve to the docs
  environment, where `jupytext` may not be installed. Be explicit:
  `.venv/Scripts/python.exe -m jupytext …`.
- Note the current version in `pyproject.toml` and skim recent `git log` —
  the author may have landed things ahead of the plans (standing workflow rule).
- Check `git status` / `git diff` for **uncommitted author edits to
  `features.qmd`**. They are authoritative — work around them additively, never
  revert them, and note in the run summary that the run started from an edited
  working copy.

### 3.1 Build the work list

1. Parse `CHANGELOG.md`: collect every `## 1.0.0aN` (and `b*`) heading and its
   `###` sub-headings — this is the authoritative feed of what shipped.
   **First run must read the full file, including `a1`–`a20`**, not only the
   recent entries.
2. Read the coverage ledger (§2.3) from `features.qmd` (empty on the first run).
3. **Gap = CHANGELOG items with no reconciled ledger row** (new since last run),
   plus any ledger row whose example has drifted from the current API (caught in
   §4 if not spotted here).

### 3.2 Close the gap

For each work item:

- **Classify** user-facing vs internal. Heuristic: does it add or change
  something a user *writes or calls* (DecL keyword, method, property, argument,
  config key, error behaviour)? → user-facing. Otherwise (refactor, dep change,
  internal harness, performance, module move) → internal.
- **User-facing** → place it in the right section (§2.2), write one or two worked
  examples, **reusing a cast object** where possible. If it needs an object none
  of the cast provides, add a cast member in §2 and note why.
- **Internal** → add a one-line row to the "Under the hood" appendix (§2.5).
- **Update the coverage ledger** so the item now has a row.

Keep the cast small and the examples leveraged — resist one-object-per-feature
sprawl. Cross-link sections rather than re-introducing an object.

**Additive and repair-only** (the ownership rule in the header): closing the
gap means *adding* coverage and *repairing* broken cells. Author prose and
examples already in the doc are not rewritten, reorganized, or deleted — if the
run believes a restructure would improve the doc, it says so in the run summary
and waits.

### 3.3 Housekeeping touches

- Where a feature was a **breaking change** vs 0.30.1 (e.g. `recommend_p` →
  `bucket_sizing_p`, `reinsurance_*` → `reins_*`, distortion flat-number syntax,
  `Underwriter` keyword-only / empty-by-default), say so plainly in that
  section — this doc is also where a returning user learns what to change.

---

## 4. Verification (the done-gate)

Every run ends by **executing the whole notebook headlessly** and requiring it to
complete with no cell raising:

```
.venv/Scripts/python.exe -m jupytext --to ipynb --execute \
    docs/new_material/features.qmd --output <temp>.ipynb
```

(`jupytext --execute` runs each cell through the kernel; a non-zero exit means a
cell raised.) The executed `.ipynb` is a **throwaway** — discard it, never commit
it. Outputs are not stored in the `.qmd`.

If a cell fails, the usual cause is API drift since the example was written — fix
the example to the current API (that *is* part of keeping the doc honest), or, if
the feature itself regressed, stop and report it rather than papering over it.

**Done when:** every CHANGELOG item has a ledger row; the notebook executes
end-to-end clean; the cast is still small and leveraged.

---

## 5. Release-hygiene integration

This task is **part of release hygiene**, not a separately versioned change:

- `features.qmd` is a doc-only artifact, so updating it **does not itself bump the
  `1.0.0*` version** (per the repo's standing rule: doc-only edits don't bump).
- Instead, **each `a*` / `b*` release that adds a user-facing feature should run
  this task as part of closing the iteration**, so `features.qmd` lands in step
  with the CHANGELOG entry — the same close-the-iteration discipline as updating
  `CHANGELOG.md` and `dev/TODO.md`.
- No `dev/done/` move applies (this is a standing task, not a consumable plan).

---

## 5a. Run history

- **First run, 2026-06-11, at a57.** Covered a1 to a57; 57-row ledger; §4 gate
  passed. Seed classification retained in §6.1 below.
- **Catch-up run, 2026-07-29, at a168.** The task had not been run since the
  first, so the gap was 111 releases. Ledger extended to **168 rows**. Five new
  top-level sections were added for subsystems that did not exist at a57: P&L
  (§14), Bivariate (§15), Reinsurance economics (§16), Renewal and ruin (§17),
  the Recipe library (§18); Under the hood became §19. Substantial repair was
  needed as well, because the doc had drifted badly against the API:
  `describe` → `summary_df` (a84), `multivariate` → `bivariate` (a80),
  `explain_validation()` → `validation_explanation` (a82/a84),
  `BivariateAggregate.corr()` → property (a85), and the whole `pnl` cast member,
  whose in-place affine form was removed at a103 and replaced by the
  engine-wrapping form at a125. **One substantive error was corrected, not just
  a rename:** old §3.3 claimed the per-claim (`ssev`) premium form carried
  *more* variance than the book-level `pnl` form. It carries far less
  (sd 63.6 vs 253.1), because premium booked per claim scales with N and hedges
  the count risk. Verified against both the FFT and the closed-form
  decomposition. §4 gate passed: 70 code cells, 0 errors.
  Both files then moved from `dev/` to `docs/new_material/`.

## 6. First-run note (completed 2026-06-11, at a57)

The first execution covered the entire `a1`–`a57` series; `features.qmd` now
exists with a 57-row ledger and passed the §4 gate. Subsequent runs are the
small top-up per release. The seed classification below is **retained for
reference** (it records where each item was placed); the live state is the
ledger in `features.qmd`.

### 6.1 Seed classification (a21–a57, verified against CHANGELOG)

User-facing (get worked examples):

- **a21** negative-support severity, `ssev`, output window → §3 / §6
- **a22** signed `Portfolio` combine; `shift - dist` severity → §3 / §6
- **a23** `pnl` premium keyword; signed-aware `describe` → §3 / §6
- **a24** `multivariate` / `copula` / `netceded` → §3
- **a25** `hints{}` vs `note{}` → §3 / §7
- **a27** distortion flat-number DecL syntax (breaking) → §3
- **a28** `dsev_bucket` linear/nearest → §7
- **a29** tail-thickness classification → §8
- **a30** config.toml layering, `show_settings` (user surface) → §11
- **a31** the pentagon octet; one canonical pricing readout → §9
- **a32** legible `Underwriter` loading; `to_agg` (breaking renames) → §12
- **a33** `price_stand_alone` restored → §9
- **a34** `distortion_df` / `calibration_df`; `gini_p` rename → §9
- **a35** empty `Underwriter()` by default; `to_agg` modes → §12
- **a36** `AllocationBounds` → §10
- **a37** `PricingBounds`, the Gini lens → §10
- **a39** keyword-only `Underwriter`, signed Lee plot, `density` property → §12 / §6
- **a40** SD for zero-mean signed aggregates → §6
- **a41** `reinsurance_*` → `reins_*` (breaking) → §3 or §6 (reins reporting)
- **a43** `price_pentagon`; signed `Portfolio.describe` spread → §9 / §6
- **a45** `pnl` with signed loss severity (bug fix, enables the case) → §3
- **a46** exponential-tilting pedagogy → §13
- **a47** `approximate` DecL keyword (note: comes *after* the freq clause) → §3
- **a48** spliced unbounded severities build (fix; enables the `mv_*` cast splices) → §3 / §7
- **a50** always-present `approximate` info line; self-describing fit note; window-aware plots → §6
- **a51** non-zero output window for concentrated aggregates; `x_min=0` to `update`
  restores the legacy 0-based grid (behavioral note: `q`/`F`/plots live on the
  window) → §7, demonstrated by `bigex`
- **a52** `_` digit separators in DecL numbers; `of` as a share synonym in reins → §3
- **a53** `format_program` / `spec_to_decl`; `pprogram` and `to_agg` now emit
  canonical DecL (breaking: `decl_pprint` removed) → §5
- **a54** `value_type` on `Aggregate`/`Portfolio` (mixed loss/payoff books rejected);
  fixed-layout `info` rows (breaking, display-level); `pnl` prem/lr in `stats_df`
  meta; configurable `[labels]` → §6 / §11
- **a55** `unit_density` / `unit_density_df` / `aligned_unit_density_df` accessors → §6
- **a56** breaking: `p_<unit>` columns and the EPD family removed from
  `density_df` (use the a55 accessors; EPD is the one-liner `(e − lev)/e`);
  signed books get the objective columns → §6
- **a57** breaking: `T.*`/`M.*` columns and `efficient` removed from
  `apply_distortion`/`price`; new `allocation_diagnostics`; signed (P&L) books
  price; deficit policy (`allow_deficit`) → §9

Internal (named in "Under the hood"):

- **a38** knowledge-freeze regression harness
- **a42** fuzz-removal consolidation (`utilities.remove_fuzz`)
- **a44** module organisation, dropped deps, lazy IPython
- **a49** portfolio combine grid: `best_window` replaces the RMS combine — mostly
  internal, but worth a one-line user note in §7 (discrete books now land on the
  integer lattice; adding units no longer coarsens the grid)
- **a55/a56/a57** numerics spine mechanics (shifted kappa, direct sums, one
  Choquet engine) — internal; only the breaking column changes and new
  surfaces above are exampled

**a1–a20 are not yet classified** — the first run must read them in `CHANGELOG.md`
and slot them in (the negative-x backbone, parser-error promotion, the
distributions/portfolio core refactor, etc., live there).
