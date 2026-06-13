# Task — maintain the three introductions (`dev/intro-*.qmd`)

> **What this is.** A *repeatable task*, sister to `dev/task-features.md`.
> Invoke it by saying **"execute task-intros"** (or "bring the intros up to
> date"). Each run verifies the three introduction notebooks execute clean
> against the current API, repairs drift, and — when the library's story has
> materially changed — refreshes the pitch.
>
> **Ownership and edit flow.** Same git-mediated model as `features.qmd`: the
> `.qmd` files are the source of truth for their content and the author may
> edit them directly; runs arrive as reviewable diffs. One deliberate
> difference: **the pitch voice belongs to the assistant.** The author's
> theory: *the sausage maker is not a good sausage salesperson* — the author
> is too close to the features to rank them for a stranger. So where
> `task-features` runs are strictly additive/repair-only, an intros run **may
> propose pitch rewrites** (reordering the sell, swapping a hero example,
> cutting a beloved-but-niche feature) — as a diff the author reviews, with
> the reasoning in the run summary. Author edits to *facts and examples*
> always survive; the *salesmanship* is contestable.

---

## 1. Purpose and audience

Three pure user-facing introductions for someone **new to `aggregate`** — or,
equivalently, written as if 0.30.1 never existed. No changelog talk, no "new
in 1.0", no version archaeology: present tense, current API, the library as a
stranger first meets it.

- **`intro-1min.qmd`** — the elevator pitch. One screen: the problem, the
  promise, one irresistible example, the install line.
- **`intro-5min.qmd`** — the demo. The core loop (**declare → build →
  validate → trust**) plus the two hooks that land hardest (insurance
  vocabulary built in; exactness you can check).
- **`intro-20min.qmd`** — the working session. A structured tour a
  practitioner can type along with: DecL anatomy, the severity/frequency
  vocabulary, layers, reinsurance, the simulation comparison, portfolios and
  capital, where to go next.

**Pattern, not content.** Modeled on the existing ten-minute guide
(`docs/2_user_guides/2_x_10mins.rst`) and its own models, pandas' "10 minutes
to pandas" and the NumPy quickstart: an **Objectives / Audience /
Prerequisites** header block, a contents list (20-minute only), code-first
cells with one-sentence lead-ins, no wall-of-prose. The *content* is written
fresh — these are sales documents first, tutorials second.

### 1.1 Audience map (who is being pitched)

Primary: **P&C actuaries and risk managers**. The pitch should make each of
these readers feel seen within the first minute of their respective document:

1. **Pricing actuary** — excess layers, ILFs, treaty terms; the distribution
   behind every increased-limits factor.
2. **Reinsurance / treaty analyst** — occurrence and aggregate programs,
   net/ceded views, per-layer statistics (CAS's reinsurance-pricing
   standardization push is the zeitgeist).
3. **Capital / ERM actuary** — economic capital is a quantile of an aggregate
   distribution (SOA ERM syllabus framing); spectral risk measures and
   capital allocation.
4. **Cat modeler** — frequency mixing / common shock, occurrence covers,
   heavy tails.
5. **Operational-risk quant (banking)** — the Basel LDA *is* a compound
   frequency×severity model evaluated at the 99.9th percentile, exactly where
   Monte Carlo is weakest.
6. **Academic / student** — the collective risk model of Klugman *Loss
   Models* and the ASTIN literature, runnable; exam practice; reproducible
   papers.

Secondary: insurtech data scientists and anyone with a "random sum of random
amounts" problem.

### 1.2 The sales argument (the spine all three share)

1. **The problem is universal.** Premium, reserve risk, capital, reinsurance
   value — all are questions about the distribution of a random sum
   `S = X₁ + … + X_N`. Closed forms essentially never exist.
2. **The incumbent answers are bad in the tail.** Moment matching is wrong
   where the money is; simulation is slow, noisy precisely at the quantiles
   that drive capital, and must be rerun for every structural tweak.
3. **`aggregate` gives the whole distribution, fast and essentially exact**
   (FFT convolution; machine-precision-grade accuracy at parametric speed),
   with built-in validation against exact moments — *trust but verify* is a
   first-class feature, not an afterthought.
4. **It speaks insurance.** Limits, attachments, shares, occurrence and
   aggregate reinsurance, mixed exposures are language primitives (DecL), so
   a model is a short, readable, auditable document — not five hundred lines
   of simulation code.

The one-minute doc is points 1–3 with one example. The five-minute doc adds
point 4 and a self-check. The twenty-minute doc demonstrates all four,
including a head-to-head against Monte Carlo.

**House framings (author-fixed, use verbatim):**

- The cycle is **declare → build → validate → trust** — validation is a named
  stage, consistent with the Annals of Actuarial Science package paper
  (`Mildenhall2024`), which the five- and twenty-minute docs cite.
- DecL anatomy is presented as **exposure = (volume, coverage)**: volume =
  claim count | expected loss | premium × loss ratio (convenience); coverage
  = limit xs attachment. This mirrors the `Aggregate` constructor's `exp_*`
  arguments (`exp_en`/`exp_el`/`exp_premium`/`exp_lr` +
  `exp_limit`/`exp_attachment`). The twenty-minute doc carries the full
  multi-line anatomy diagram including the optional occurrence and aggregate
  reinsurance clauses and the `approximate` shortcut.

---

## 2. The deliverables

### 2.1 Format and conventions

- Quarto markdown (`.qmd`), YAML front matter (`title`, `jupyter: python3`),
  ```` ```{python} ```` cells. jupytext pairs them as notebooks; `quarto
  render` also works.
- **Runnable source only — no committed outputs** (same rule as
  `features.qmd`).
- Display via `from aggregate import build, qd`; plots where a picture sells
  (the five- and twenty-minute docs should each have at least one).
- **US spelling**; house abbreviations (`sev`, `occ`, `agg`, `freq`, `cv`,
  `bs`) used and briefly glossed on first use.
- Realistic numbers a practitioner recognizes (limits like 1000 xs 0,
  loss ratios near 0.65, ROE near 0.10) — no toy 3-and-7 examples except
  where exactness is the point (dice).
- Each document is **self-contained** (its own imports and builds) and ends
  by pointing at the next-longer one, then the published docs.
- **Citations** follow the standing order in `CLAUDE.md` ("Citations and
  bibliography"): `bibliography:` + `csl:` lines in the YAML, `@Key` cites
  found by searching `C:/s/TELOS/Biblio/uber-library.bib` (never invented),
  and a `## References` / `::: {#refs}` block on pages that cite. Keep the
  one-minute page cite-free (an elevator pitch has no footnotes); the five-
  and twenty-minute pages cite the collective-risk-model and pricing-theory
  anchors.
- Length discipline: the names are honest. One minute reads in one minute
  (~one screen, ≤2 code cells); five minutes is ~8–12 cells; twenty minutes
  is a long session but still a session (~25–40 cells).

### 2.2 Content rules

- **Zero release-history content.** Never "new", "now", "recently", "since
  0.30"; no CHANGELOG or version references. (That is `features.qmd`'s job.)
- Current API only; every cell must execute (§4).
- Sell benefits, then show mechanism: each section leads with what the reader
  gets, demonstrates it, and only then (twenty-minute doc only) sketches how
  it works.
- The Monte Carlo comparison (twenty-minute doc) must be **fair and
  reproducible**: same model both ways, seeded RNG, honest about what MC does
  well; the point is tail-quantile noise and re-run cost, not a strawman.
- Distortion pricing / capital allocation appear as a *teaser with a
  pointer*, not a course — `price_pentagon` is the right altitude for an
  intro.

---

## 3. The procedure (each `execute task-intros` run)

1. **Pre-flight** — as `task-features` §3.0 (branch, `uv sync --extra
   notebook`, note version, check for uncommitted author edits to the
   `intro-*.qmd` files — author edits are authoritative).
2. **Verify** all three notebooks execute (§4). Repair API drift.
3. **Re-read as a salesperson** (cheap, every run): does the pitch still lead
   with the library's strongest material? If a recently landed feature
   belongs in the sell (rare — the intros surf the stable core), or an
   example has been superseded by something more compelling, propose the
   rewrite as a diff with reasoning. Do not churn the pitch for taste alone;
   stability is a virtue in entry documents.
4. **Run summary** — repairs made, pitch changes proposed/applied, anything
   flagged for the author.

---

## 4. Verification (the done-gate)

All three must execute headlessly with no cell raising:

```
uv run jupytext --to ipynb --execute dev/intro-1min.qmd  --output <temp>.ipynb
uv run jupytext --to ipynb --execute dev/intro-5min.qmd  --output <temp>.ipynb
uv run jupytext --to ipynb --execute dev/intro-20min.qmd --output <temp>.ipynb
```

Executed `.ipynb` files are throwaways. (`quarto render` is an equivalent
gate where quarto is installed; jupytext is the one guaranteed by the
`notebook` extra.)

**Done when:** all three execute clean; lengths honor their names; the pitch
spine (§1.2) is intact across the trio.

---

## 5. Release-hygiene integration

- Doc-only artifacts: **no version bump**, no CHANGELOG entry.
- Re-verify whenever a release touches the public API the intros use
  (`build`, `qd`, `describe`, `q`/`tvar`, `plot`, reins reporting,
  `price_pentagon`).
- **Destiny:** candidates for the Sphinx user guide at release (alongside the
  ten-minute guide, which they deliberately do not duplicate — the ten-minute
  guide is a class-by-class reference tour; these are pitch-first). No
  `dev/done/` move; standing task.
