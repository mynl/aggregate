# Task — maintain `dev/features.qmd`

> **What this is.** A *repeatable task*, not a one-shot plan. Invoke it by saying
> **"execute task-features"** (or "bring features up to date"). Each run
> reconciles `dev/features.qmd` against the current development state and adds /
> updates whatever is missing, then verifies the whole notebook executes clean.
>
> **Resume model.** There is no "step N done" — every run starts from the
> coverage ledger inside `features.qmd` and the `CHANGELOG.md` feed, computes the
> gap, and closes it. The first run is a long job (the whole 1.0.0a series); every
> run after that is a small incremental top-up tied to the latest `a*` / `b*`
> release.

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
`jupytext` ships in the `notebook` extra). It lives in `dev/` — it is a project
artifact, **not** part of the Sphinx/readthedocs tree *today*.

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

> **This block is the canonical definition of the cast.** The `execute` step
> copies it verbatim into `features.qmd` §2. To change the cast, **edit it here**
> and it flows through on the next run — do not diverge the two.
>
> **The run has full license to edit this block** — fix API drift and parse
> errors, tidy names and comments, and tune programs to be more educational /
> illustrative. The cast is expected to evolve and settle over time; what is
> most useful long-term is hard to call up front, so improve it whenever an
> improvement is clear, and report the edits in the run summary. Compositional
> judgment calls (drop a member, change what a member is *for*) deserve a note
> to the author but need not block the run.

Members are **clustered** (`n.a`, `n.b`, …) so closely-related examples — and the
features they illustrate — sit together. Renumber freely as the set evolves.

```python
from aggregate import build, qd

# ── The cast: built once, reused throughout ────────────────────────────────
# Clustered by theme. Each is the minimal program that exercises a distinct
# surface; feature sections borrow these rather than minting throwaways.

# 1. Core aggregate ─────────────────────────────────────────────────────────
# 1.a simple — the workhorse: mixed-gamma (neg-binomial) frequency, layered lognormal
simple = build('agg Simple 100 claims 1000 xs 0 sev lognorm 100 cv 1.0 mixed gamma 0.25')

# 1.b defective — essentially defective tail (shifted Pareto; mass pushed to the limit)
defective = build('agg Defective 10 claims sev 100 * pareto 1.3 - 100 poisson '
                  'hints{bs=0.25; log2=16;}')

# 2. Discrete, exact moments ─────────────────────────────────────────────────
dice = build('agg Dice dfreq [3] dsev [1:6]')

# 3. Signed & P&L ────────────────────────────────────────────────────────────
# 3.a signed_d — signed discrete (dsev with a negative atom auto-signs the agg)
signed_d = build('agg SignedD dfreq [5] dsev [-1 2]')

# 3.b signed_s — signed continuous via ssev; a PER-CLAIM shift (premium booked per claim)
signed_s = build('agg SignedS 10 claim ssev 100 - lognorm 80 cv 0.025 poisson')

# 3.c pnl — book-level premium minus loss. Contrast 3.b: SAME MEAN, DIFFERENT VARIANCE —
#     the pnl premium is one deterministic shift; the ssev +100 is multiplied by the
#     (random) claim count, so 3.b carries extra variance from 100*N. The qmd section
#     should pull this contrast apart explicitly.
pnl = build('pnl PNL 1000 premium - 10 claim sev lognorm 80 cv 0.025 poisson')

# 4. Multivariate ────────────────────────────────────────────────────────────
# 4.a mv_indep — INDEPENDENT (shared frequency only; positive baseline corr from
#     the shared mixing / common shock). Splicing an unbounded base works since a48.
mv_indep = build('''
multivariate Cat 25 claims
    agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 0.65 splice [0 250]
    agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 0.95 splice [0 300]
    mixed gamma .2
''')

# 4.b mv_copula — COPULA-coupled (Gumbel, upper-tail dependence)
mv_copula = build('''
multivariate CatC 25 claims
    agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 0.65 splice [0 250]
    agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 0.95 splice [0 300]
    copula gumbel 0.4
    mixed gamma .2
''')

# 5. Reinsurance ─────────────────────────────────────────────────────────────
# 5.a reins — occurrence reinsurance; describe now reports gross/net.
#     Drives reins_describe / reins_stats_df. (Note: the occ clause comes BEFORE freq.)
reins = build('agg Re 10 claims 1000 xs 0 sev lognorm 100 cv 2 '
              'occurrence net of 50% po 300 xs 200 and 100% po 500 xs 500 '
              'poisson')

# 5.b netceded — the SAME program as 5.a with the `netceded` prefix: the occurrence
#     (Ceded, Net) law as a bivariate. Literally 5.a with `netceded ` prepended.
netceded = build('netceded agg Re 10 claims 1000 xs 0 sev lognorm 100 cv 2 '
                 'occurrence net of 50% po 300 xs 200 and 100% po 500 xs 500 '
                 'poisson')

# 6. Mixtures ────────────────────────────────────────────────────────────────
# 6.a mix_sev — weighted MIXTURE of severities (single exposure)
mix_sev = build('agg MixSev 100 claims 2000 xs 0 '
                'sev lognorm [50 100 200] cv [1 1.5 2] wts [.5 .3 .2] poisson')

# 6.b mix_exp — several EXPOSURE bands, one severity
mix_exp = build('agg MixExp [100 200 50] claims 2000 xs 0 sev lognorm 100 cv 2 poisson')

# 6.c mix_both — JOINT mixed severity + exposure (the gnarly one): paired exposure/limit
#     bands AND a two-component severity mixture of shifted pareto / lognormal.
#     The qmd should spend time pulling apart exactly what this builds.
mix_both = build('agg MixBoth [100 200 50] claims [1000 2000 5000] xs 0 '
                 'sev [200 150] * [pareto lognorm] [2.1 0.8] + [-200 0] wts [.2 .8] poisson')

# 7. Portfolios ──────────────────────────────────────────────────────────────
# 7.a book — a Portfolio for combine / pentagon pricing / allocation & pricing bounds
book = build('''
port Book
    agg A 100 claims 2000 xs 0 sev lognorm 100 cv 1.0 mixed gamma 0.5
    agg B  50 claims 4000 xs 0 sev lognorm 200 cv 2.0 mixed gamma 0.4
''')

# 7.b book_w_re — the same book with occurrence reinsurance on each unit
book_w_re = build('''
port Book
    agg A 100 claims 2000 xs 0 sev lognorm 100 cv 1.0 occurrence net of 50% po 1000 xs 1000 mixed gamma 0.5
    agg B  50 claims 4000 xs 0 sev lognorm 200 cv 2.0 occurrence net of 50% po 2000 xs 1000 mixed gamma 0.4
''')

# 8. Approximate (a47) vs exact on a non-zero window (a51) ───────────────────
# 8.a big — very-high-frequency book via the `approximate` shortcut (sgamma fit).
#     Note `approximate` comes AFTER the frequency clause. The info line and the
#     self-describing fit note are a50.
big = build('agg Big 1e6 claims dsev [1 3] poisson approximate sgamma')

# 8.b bigex — the EXACT convolution for comparison: since a51 it resolves at
#     bs=1 on a two-sided output window far from 0 (mean 2e6, sd ≈ 2,200)
#     instead of wasting the whole grid on [0, 2e6).
bigex = build('agg BigEx 1e6 claims dsev [1 3] poisson')
```

| Member | Drives |
|---|---|
| `simple` | reporting quartet, tail class, `density` property, config effects |
| `defective` | defective / mass-at-limit tail, deficit warnings, `hints{}` |
| `dice` | discrete/exact moments, `dsev_bucket` |
| `signed_d` | signed `dsev`, signed-aware `describe` (SD vs CV) |
| `signed_s` / `pnl` | `ssev`, `shift - dist`, per-claim vs book-level premium (`pnl`) |
| `mv_indep` / `mv_copula` | `multivariate`, `copula` (and the no-copula baseline) |
| `reins` / `netceded` | reins gross/net reporting; `netceded` occurrence bivariate |
| `mix_sev` / `mix_exp` / `mix_both` | mixed severity, mixed exposure, and the joint |
| `book` / `book_w_re` | combine, pentagon pricing, `price_pentagon` / `price_stand_alone`, bounds; reins in a portfolio |
| `big` / `bigex` | `approximate sgamma` vs the exact convolution on a non-zero output window (a51) |

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
- `uv sync --extra notebook` (the verification in §4 needs a Jupyter kernel;
  `ipykernel` arrives transitively via `jupyterlab`). `UV_LINK_MODE=copy` is set
  for the harness; in a plain shell `$env:UV_LINK_MODE = "copy"`.
- Note the current version in `pyproject.toml` and skim recent `git log` —
  the author may have landed things ahead of the plans (standing workflow rule).

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
uv run jupytext --to ipynb --execute dev/features.qmd --output <temp>.ipynb
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

## 6. First-run note

The first execution is large (the entire `a1`–`a57` series) and will surface
judgement calls — which features share a cast object, how deep each example goes,
where the user-facing/internal line falls. **Expect feedback that edits both this
spec and the output `.qmd`.** Once tuned, subsequent runs are a small top-up per
release. Treat the initial classification below as a seed, not gospel.

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
