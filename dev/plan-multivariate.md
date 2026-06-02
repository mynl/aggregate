# Plan: integrated multivariate aggregate distributions

> **Status:** DRAFT for iteration (2026-06-02). Sibling plan:
> [`plan-negative-x-agg.md`](plan-negative-x-agg.md) (+ `plan-negative-x-port.md`)
> — **build negative-x first**;
> multivariate reuses its signed-axis / window-sizing machinery. This plan
> generalises the just-shipped `occ_bivariate` / `bivariate.py` (1.0.0a20) into a
> first-class multivariate facility with its own DecL, classes, and reporting,
> and is expected to **subsume** the bivariate special case.

---

## 1. Motivation

`occ_bivariate` (a20) proved the core trick: the joint law of two dependent
aggregate quantities is an ordinary compound-distribution FFT with the 1D
transforms replaced by ND transforms, because `freq_pgf(n, z)` is elementwise in
`z` and therefore applies unchanged to a multi-dimensional transform. We now want
this as a *modelled object*, declared in DecL, with marginals, a correlation
structure, joint moments, `info` / `describe` / `density_df` / `stats_df`, and
plots — mirroring `Aggregate` / `Portfolio`.

Two user-facing entry forms, both grounded in the author's examples:

```
# (1) Several correlated lines sharing one claim-count process
multivariate AL.GL 25 claims
    agg AL dfreq [0 1] [.2 .8] sev lognorm 20 cv .8
    agg GL dfreq [0 1] [.9 .1] sev lognorm 60 cv 1.2
    poisson                       # Poisson count -> lines INDEPENDENT

multivariate AL.GL.2 25 claims
    agg AL dfreq [0 1] [.2 .8] sev lognorm 20 cv .8
    agg GL dfreq [0 1] [.9 .1] sev lognorm 60 cv 1.2
    mixed gamma .2                # shared mixing -> lines CORRELATED

# (2) Joint ceded/net of one occurrence-reinsured line (replaces occ_bivariate)
netceded RE.test.1 40 claims
    agg dfreq [1] 1000 xs 0 lognorm 140 cv .9 occurrence net of 90% po 600 xs 400
    mixed ig .4
```

End-state we want to keep in view (not in the first cut):

```
multivariate Complex 25 claims
    agg Unit1 <spec1>
    agg Unit2 <spec2>
    copula <normal|t|clayton|...> <params>
    <freq spec>
```

— the two aggregate *outputs* run through a copula to produce the bucketed joint
severity that feeds the ND-FFT.

---

## 2. The unifying insight: two joint-severity constructions, one backbone

Every case is: **build a joint per-claim severity tensor `S` on an ND grid →
`density = iFFTn(freq_pgf(N, FFTn(S)))` → report.** What differs is *only how `S`
is built*:

The author frames this as **two families** of joint severity, both feeding the
same frequency convolution and both handled **seamlessly by one
`MultivariateAggregate` class** (it dispatches on the construction mode; nothing
downstream of `S` differs):

- **(i) independent component severities** (units *a*, *b*, … built separately)
  combined by outer product, where the *aggregate* dependence is created by the
  shared frequency (Poisson ⇒ independent, mixed-Poisson ⇒ correlated);
- **(ii) a complex / dependent joint severity** built directly — the ceded/net
  comonotone split **or** two unit outputs coupled by a **copula** — then the
  same frequency convolution.

| Construction | Family | Used by | Per-claim joint severity `S` | Dependence source |
|---|---|---|---|---|
| **Independent-coordinate (outer product)** | (i) | `multivariate` | `S = g₁ ⊗ g₂ ⊗ … ⊗ g_k`, where `g_i` is line *i*'s discretised severity **including its mass at 0** (the `dfreq [0 1] [p₀ p₁]` zero-inflation). `FFTn(S) = ⊗ᵢ FFT(gᵢ)`. | The **shared frequency**: Poisson ⇒ superposition ⇒ independent; mixed-Poisson (shared `G`) ⇒ common shock ⇒ correlated. |
| **Comonotone scatter (anti-diagonal)** | (ii) | `netceded` | One gross loss `X` splits to `(c(X), n(X))` on the line `c+n=X`; scatter gross mass there (`scatter_bivariate`). | Per-claim comonotonicity of `c,n` **plus** the count (exactly `occ_bivariate`). |
| **Copula coupling** | (ii) | `copula` clause | couple the component severities / aggregate outputs through a fitted copula, re-bucket to the joint `S`. | The copula (a peer of the ceded/net split, not a special case). |

**Why the outer product is correct for `multivariate`.** A single claim makes
line *i* occur independently w.p. `p_i` (the `[0 1][p₀ p₁]` Bernoulli) with
severity `f_i`; "neither or both" is allowed. So the per-claim vector is
`(B₁X₁, …, B_kX_k)` with independent coordinates → `S` is the outer product of the
marginal per-claim severities `g_i = (1−p_i)δ₀ + p_i f_i`. These `g_i` are exactly
what each inner `agg` already discretises. Under a Poisson outer count the
superposition theorem makes the lines independent Poisson compounds; a shared
mixing distribution couples them — matching the author's stated behaviour
precisely (Poisson ⇒ independent, `mixed` ⇒ correlated).

This means **`multivariate` is even simpler than `netceded`**: it's an outer
product of 1D severity FFTs, then the shared PGF. `netceded` is the scatter case
we already implemented. The new `multivariate.py` provides **all three severity
builders** (outer-product, comonotone-scatter, copula) over one ND-FFT backbone,
selected by a construction-mode flag and otherwise sharing every downstream step;
`BivariateDistribution`/`occ_bivariate` become a 2-D `netceded` view (kept as a
thin compatibility wrapper, or deprecated in favour of `netceded` — see §7). The
copula builder is **designed as a peer now** (so the class shape and grammar slot
don't need repainting) and **implemented in a later stage**.

---

## 3. Current state (grounding)

- **Bivariate prototype:** `bivariate.py` — `BivariateDistribution`,
  `size_axis` (per-axis window sizing), `scatter_bivariate` (anti-diagonal
  scatter). `Aggregate.occ_bivariate` (`distributions.py`, after
  `reinsurance_occ_plot`) orchestrates the scatter case. **Lesson already
  banked:** `freq_pgf` is elementwise *mathematically* but
  `FrequencyEmpirical.freq_pgf` (backs `dfreq`) is a matmul over the support —
  must `z.ravel()` then `.reshape(z.shape)` for any ND `z`. This applies to every
  ND transform here.
- **Grammar / parser:** `decl.lark` top-level `answer: sev_out | agg_out |
  port_out | distortion_out | expr`; `port_out: PORT name note agg_list` and
  `agg_list` is a list of `agg_out` (`decl.lark:42–45`). The transformer
  (`parser.py`) returns `('agg', name, spec)` per agg and `('port', name,
  {'spec': agg_list, ...})`; `_factory` (`underwriter.py:409`) dispatches on
  `kind` to build `Aggregate` / `Portfolio`. **We extend exactly these seams.**
- **Frequency PGFs:** `distributions.py:984–1407` — each family's `freq_pgf`;
  mixing families (`gamma`/`ig`/`sichel`/…) implement the shared-mixing transform
  we rely on for correlation.
- **Portfolio combine:** `portfolio.py:1692–1702` multiplies per-line FFTs (the
  independent-sum case); the multivariate object instead *keeps* the joint tensor
  rather than collapsing to a total.

---

## 4. DecL design

### 4.1 New top-level statements
Two new keywords, `multivariate` (alias `mv`?) and `netceded`, parsed into new
`kind`s. Grammar additions (mirroring `port_out` / `agg_out_full`):

```lark
answer: ...
      | mv_out          -> answer_mv

mv_out: MULTIVARIATE name exposures mv_body freq note     -> mv_out_multivariate
      | NETCEDED        name exposures agg_out freq note   -> mv_out_netceded

mv_body: mv_body agg_out   -> mv_body_cons
       | agg_out           -> mv_body_one
```

Terminals (priority 2, with the same `(?![namechar])` lookahead as the others):

```lark
MULTIVARIATE.2: /(?:multivariate|mv)(?![a-zA-Z0-9._:~\-])/
NETCEDED.2:     /netceded(?![a-zA-Z0-9._:~\-])/
```

and add both words to the `ID` negative-lookahead exclusion list (`decl.lark:311`)
and to `parser.py`'s keyword handling.

**Shape rationale:**
- The outer `exposures … freq` is the **shared count** (e.g. `25 claims … mixed
  gamma .2`) — same nonterminals `agg_out_full` already uses, so we reuse the
  exposure/frequency transformer code verbatim.
- The body is a list of ordinary `agg_out` lines. For `multivariate` they are
  **severity factories**: restricted to the zero-inflated `dfreq [0 1] [p₀ p₁]`
  form (claim count 0 or 1) so each contributes a per-claim `g_i`. For
  `netceded` the single inner `agg` carries the occurrence reinsurance whose
  ceded/net split defines the two axes.

### 4.2 Validation (in the transformer / new classes)
- `multivariate`: each inner agg must be a `dfreq [0 1] […]` (Bernoulli)
  severity factory — reject other frequencies with a clear DecL error (the outer
  `freq` owns the count). Severities may be mixed / spliced / scaled / shifted
  (incl. negative support → signed axis, via negative-x).
- `netceded`: exactly one inner agg, and it **must** carry an `occurrence`
  reinsurance clause (else there's nothing to split). Error otherwise.
- Cap dimensionality (dense ND grid): warn/refuse beyond ~3–4 axes in the first
  cut (memory). 2-D and 3-D are the realistic targets.

### 4.3 `kind` / factory
`_factory` (`underwriter.py:409`) gains:
```python
elif kind == 'mvagg':
    obj = MultivariateAggregate(**spec)   # spec carries sub-specs + shared freq + mode
```
`spec` mirrors the port shape: `{'name', 'exposures'/'en', 'freq…', 'lines':
[inner agg specs], 'mode': 'multivariate'|'netceded', 'note'}`. `build()` /
`build_many()` need no change beyond the new kind flowing through.

---

## 5. New classes

### 5.1 `MultivariateAggregate` (new `multivariate.py`)
Mirrors `Aggregate`'s public quartet. Holds:
- the shared `Frequency` (built from the outer freq spec),
- a list of inner `Aggregate`-style **severity factories** (mode
  `multivariate`) *or* one occ-reinsured `Aggregate` (mode `netceded`),
- per-axis grids/buckets via the unified window sizer (`size_axis` →
  shared helper; signed axes when a line is P&L),
- the joint density tensor `S` and `density` (ND ndarray).

Methods (named to match 1D counterparts):
- `update(log2=…, bs=…, …)` / `update_work(...)` — build `S` (outer product or
  scatter), `FFTn`, `freq_pgf(N, z.ravel()).reshape`, `iFFTn`, store `density`.
  Per-axis sizing + the negative-x offset per axis.
- `density_df` — tidy/long frame: a MultiIndex of axis values (`L1, L2, …`) with
  the joint `p`, plus marginal columns `p_L1`, `p_L2`, … (axis sums). For 2-D
  this is the current `BivariateDistribution.density`/marginals, reshaped to the
  house DataFrame style.
- `stats_df` — per-line marginal moments (mean/cv/skew, theoretical vs
  empirical, reusing `xsden_to_mwrangler` on each axis-sum) **plus** the joint
  block: covariance / correlation matrix and mixed moments `E[Lᵢ Lⱼ]`.
- `describe` — economic-view summary per line + correlation matrix.
- `info` — shape, axes/buckets/windows, shared-freq description, dependence
  source (Poisson⇒independent / mixed⇒correlated), tail deficit per axis.
- marginals/moments/corr — generalise `BivariateDistribution.marginals/moments/
  corr` to ND (the 2-D code already exists; lift it here).
- plots — 2-D contour (existing), pairwise contour grid for ≥3-D, marginal
  overlays.

**Validation showpiece:** each marginal of a `multivariate` reproduces the
corresponding standalone `Aggregate` (the line's own compound under the shared
count's marginal law); each `netceded` margin reproduces the univariate
occurrence ceded/net aggregate (exactly the a20 validation). These give exact
test targets.

### 5.2 `MultivariatePortfolio`
Mirrors `Portfolio`: a collection of `MultivariateAggregate` units (or a
portfolio whose lines are themselves correlated). Provides the same quartet at
the portfolio level. The portfolio *total* across independent units is the usual
FFT product (`portfolio.py:1698`); the novel content is preserving / reporting
the joint dependence and per-unit marginals. **Likely a later stage** — land
`MultivariateAggregate` first.

### 5.3 Relationship to `bivariate.py`
- Move/extend the 2-D code into `multivariate.py` (ND `size_axis`,
  `scatter_bivariate` → `scatter_comonotone`, outer-product builder,
  `BivariateDistribution` → `MultivariateDistribution` with a 2-D convenience
  view).
- Keep `occ_bivariate` working as a thin wrapper that constructs the `netceded`
  case (back-compat), or deprecate it pointing at `netceded` DecL. **Decision in
  §7.**

---

## 6. Math notes

- **Shared-mixing correlation.** With outer count `N` mixed-Poisson (mixing `G`),
  the joint transform `freq_pgf(N, ⊗ᵢ FFT(gᵢ))` introduces dependence through `G`
  exactly as the mixing PGF dictates; under a pure Poisson it factorises into
  independent compounds (superposition). No special-casing — the existing
  `freq_pgf` families deliver the right joint law. Verify numerically:
  `corr(mixed) > corr(Poisson) ≈ 0`.
- **Empirical-freq ND caveat:** `freq_pgf(n, z.ravel()).reshape(z.shape)` for
  `dfreq`/empirical outer counts (banked from a20).
- **Memory.** Dense ND grid is `∏ Nᵢ`. 2-D `2¹⁰×2¹⁰ ≈ 10⁶` fine; 3-D `2⁸³ ≈
  1.6×10⁷` OK; beyond that, dense is infeasible → copula/sampling fallback
  (future). Per-axis auto-sizing keeps each `Nᵢ` as small as accuracy allows.
- **Signed axes.** A P&L line → that axis uses the negative-x offset `i0_k`
  (per-axis roll). This is the concrete negative-x dependency.

---

## 7. Open questions for the author

1. **`occ_bivariate` fate:** keep as a back-compat wrapper over `netceded`, or
   deprecate with a pointer? (It shipped only in a20, so churn cost is low.)
2. **Keyword:** `multivariate` only, or also short alias `mv`? `netceded`
   spelling (vs `net_ceded` / `ceded_net`)?
3. **Inner-agg restriction for `multivariate`:** enforce `dfreq [0 1]` strictly,
   or allow general `dfreq` (claim can produce >1 of a line)? The author leaned
   "restrict to dfreq … only 0/1" — confirm we hard-error otherwise.
4. **Dimensionality cap** for the first cut (2-D only? 2–3-D? configurable with a
   memory warning?).
5. **Reporting shape of `density_df`** for ND — tidy/long MultiIndex (scales to
   ND, my default) vs wide 2-D matrix (nice for bivariate, doesn't generalise).
6. **Copula clause** — confirm deferred to a later stage (design the grammar slot
   now so we don't repaint, implement later).
7. **Class names:** `MultivariateAggregate` / `MultivariatePortfolio` (mirrors
   existing) — confirm; and the container `MultivariateDistribution`.

---

## 8. Staged implementation

1. **Stage 0 — depends on negative-x** (signed axes, unified window sizer).
2. **Stage 1 — `multivariate` (independent-coordinate) DecL + class.** Grammar +
   transformer + `kind='mvagg'`; `MultivariateAggregate` with outer-product `S`,
   ND-FFT, marginals/moments/corr, `density_df`/`stats_df`/`info`. Poisson
   (independent) and `mixed` (correlated) verified against standalone-agg
   marginals.
3. **Stage 2 — `netceded` DecL** routed through the same class (scatter builder),
   reproducing `occ_bivariate`; reconcile/retire `occ_bivariate` (§7.1).
4. **Stage 3 — reporting polish & plots** (describe, pairwise contours,
   correlation matrix), docs subsection (extends the a20 "Joint Distribution"
   section in `2_x_re_pricing.rst`; new user-guide page for `multivariate`).
5. **Stage 4 — `MultivariatePortfolio`.**
6. **Stage 5 (future) — copula clause.**

---

## 9. Testing

`tests/test_multivariate.py`:
- **Marginals reproduce standalone aggs.** `multivariate AL.GL … poisson` →
  each axis-sum matches `build('agg AL …')` / `build('agg GL …')` (means exact,
  cv at matched grid — same caveat as a20's linear-rebucketing finding).
- **Independence under Poisson:** `corr ≈ 0` (within numerical tol) for the
  Poisson outer count; **correlation under shared mixing:** `corr(mixed) >
  corr(poisson)` and matches the analytic common-shock value where derivable.
- **`netceded` reproduces `occ_bivariate`** bit-for-bit (regression bridge):
  same program both ways → same joint density.
- **Outer-product correctness:** 2-line fixed-count case vs a hand-built
  Kronecker convolution (closed form).
- **Empirical outer count** (`dfreq`) exercises the `ravel/reshape` path in ND.
- **Signed axis:** one P&L line (negative severity) → correct signed marginal
  (depends on negative-x).
- DecL programs appended to `src/aggregate/agg/test_decl.agg` under a new
  section.

Run `uv run pytest` (`$env:UV_LINK_MODE='copy'`); no docs build in-loop.

---

## 10. Risks / watch

- **Memory blow-up** beyond 2–3 axes — enforce the cap, auto-size each axis.
- **ND `freq_pgf`** elementwise assumption — `ravel/reshape` everywhere; add a
  guard/test for each frequency family used as the outer count.
- **Marginal-vs-grid accuracy** — the a20 linear-rebucketing second-moment
  finding recurs per axis; document that means match on any grid, cv at the
  matched grid.
- **DecL ambiguity** — new keywords must not capture identifiers; reuse the
  established lookahead + `ID` exclusion discipline; add parse-only tests.
- **Scope creep into copula** — keep it a designed-but-unimplemented slot.
