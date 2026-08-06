# [Paper-Reproductions] — candidate assessment

## Execution status

The six top-ranked candidates have been written up. Five are reproductions; one is not.

| Key | Document | Outcome |
|---|---|---|
| Venter1983 | `Venter1983.qmd` | **Reproduced.** Exhibit 3, all three columns. Severity moments (18,198 / 2.6600 / 3.6746) confirm the piecewise-linear transcription. `bs=500` matches the recursive column to 5e-5; refining to `bs=5` matches the characteristic-function column to 7e-5. The transformed gamma **is** `scipy.stats.gengamma(r, alpha, scale=1/lambda)`, matching all 34 rows to 5e-5. Exhibit 3's printed aggregate mean 250,600 is a scan artefact of 250,000. |
| Mack2003 | `Mack2003.qmd` | **Reproduced.** Table 1 (26 cells, all within 8e-4). "American Pareto" = `lomax`; "Gamma a=0.1" = shape. Theorem 2 verified end to end: the spliced lognormal gives `b(2v)/b(v) = 1.200000` exactly for `v >= u*`, and 1.271 below. Threshold u\* = 5.186 E(X), paper's "u ~ 5 E(X)". |
| Mata2005 | `Mata2005.qmd` | **Reproduced, with a finding.** Driven by the paper's own Table 1 the pipeline reproduces Tables 2, 3, 4, 5, 11, 13, 14 to the last printed digit. But **Table 1's LEVs are not the exact lognormal LEVs**: 0.04% high at $250k drifting to 0.24% low at $5M, in both years, while `aggregate` matches the closed form to 8 figures. Conclusions unaffected (third decimal). |
| Bruno2006 | `Bruno2006.qmd` | **Reproduced.** Tables 6, 7 and 8, every value exact at the printed six decimals. Table 7 (lambda=91,000, 92,832 convolutions) runs in **under 2 seconds** against the paper's reported 1,950. Table 8's Poisson/generalized-Pareto mixture is not a built-in `mixed` family; computed p(n) by quadrature and passed as `dfreq`. |
| Bodoff2017 | `Bodoff2017.qmd` | **Reproduced.** Exhibit 4, every cell. Two inferred model choices confirmed by the fit: the severity is the **two-parameter** Pareto (`lomax`), and Exhibit 3's "$25M / p=100%" row is a **global cap on the XPL branch**, not a limits-profile row. |
| Jin2016 | `Jin2016.qmd` | **NOT a reproduction.** See the corrected assessment below. |

Recorded for whoever picks up the remaining candidates: check whether the paper's own tabulated inputs are exact before blaming a mismatch on `aggregate` (Mata2005), and put every LEV that will be **compared** with another on a **common grid** (Mack2003).

---

**Status:** a curated assessment of eighteen references for suitability as `aggregate` reproductions /
examples. Each paper is rated on how well its worked example(s) map onto what
`aggregate` does: compound (frequency × severity) distributions via FFT, severity
layers/limits, reinsurance, stop-loss, risk measures, distortion pricing, and
capital allocation.

Two kinds of fit:

- **Model paper** — an insurance structure with explicit parameters. A true
  reproduction: build it in DecL, match the published numbers.
- **Method paper** — a competing numerical technique for the aggregate
  distribution. Here `aggregate` is the *exact ground truth* the paper's
  approximation is measured against; the fit depends on whether the paper gives
  concrete input parameters **and** tabulated output to compare to.

Source full-text was read from the archivum extracts
(`…/archivum/full-text/<hash>/…pdftotext.md`); Venter (1983) was read from the
author-supplied PDF and added to the archivum store. There is an aggregate API that will return the pdf given <<tag>>: http://192.168.4.43:9124/view/<<Mildenhall1998>>. Tags are always AuthorYear format.

Reproductions should follow the pattern

1. Context and problem
2. Paper solution and gold-standard
3. ``aggregate`` reproduction
4. Comments if any

A simple reproduction can be very short. Something unusual or unfamiliar to actuaries may need more context.

---

## Ranked summary

| Tier | Key | Fit | One-line reason |
|---|---|---|---|
| 1 | **Venter1983** | Strong | PCAS classic; full 3-method aggregate table (Poisson × discrete sev), two exact columns `aggregate` reproduces directly + transformed-gamma (= scipy `gengamma`) it can also build. |
| 1 | **Mack2003** | Strong | Collective-risk exposure rating, Pareto-tail splicing, Riebesell α=0.737, loss-elimination curves — core engine. |
| 1 | **Mata2005** | Strong | Complete XOL experience/exposure-rating example; lognormal σ=2.29, trend, limit profile, layer-split trend. |
| 1 | **Bruno2006** | Strong | Explicit discrete severity + Poisson(λ); tabulated PMF/CDF (Tables 6–8) to ~6 dp — gold benchmark. |
| ~~1~~ 3 | **Jin2016** | ~~Strong~~ **Weak** | **Corrected on reading.** The univariate half publishes *no* output tables (figures plus K-S statistics on simulated samples); the FFT columns are all in the *bivariate* half, whose common-shock count structure `aggregate` cannot express. |
| 1 | **Bodoff2017** | Strong | XPL exposure rating (Pareto θ=50k α=1.5, Bernoulli limit). Thin example; *not* the Bodoff already in the repo. |
| 2 | **Berens1997** | Moderate→Strong | Surcharge tables for multi-year aggregate limits; native single-year pieces + a carry-forward wrapper. |
| 2 | **Clark2005** | Moderate | Stop-loss + RMK allocation; compound distribution native, allocation is post-processing. |
| 2 | **Kang2019** | Moderate | Heavy-tail VaR/TVaR benchmark (Pareto-II 3.5,3.5); severity-only, no compounding. |
| 2 | **Lau1984** | Moderate | Moment example with explicit aggregate moments + MPY=$265,640; severity form must be reverse-fit. |
| 3 | **Goffard2020** | Weak | Orthogonal-polynomial stop-loss; test cases are graphs only, one error table. |
| 3 | **Nadarajah2016** | Weak | Critique of Jin2016 over parameter sweeps; no concrete output tables. |
| 3 | **Vernic1999** | Weak | Genuinely bivariate compound; `aggregate` reproduces marginals only. |
| 3 | **Bakar2022** | Weak | Severity-only distribution family; no compounding (aggregate's value-add unused). |
| 3 | **Provost2005** | Weak | General-statistics moment approximation; no insurance content. |
| 3 | **Hilbe2014** | Weak | Count *regression* textbook; reference for frequency math, not a reproduction. |
| — | **Lindsay2000** | n/a | No full-text extract available; generic moment-mixture statistics — likely Tier 3. |
| — | **Braithwaite1997** | n/a | ECO / excess-of-policy-limits pricing in clash treaties. **Not assessed** — surfaced only as the conceptual pair to Bodoff2017; read it before ranking. |

**Recommended first reproductions:** Venter1983 and Mack2003 (model papers,
on-theme), then Bruno2006 and Jin2016 (exact benchmarks with input + output
tables).

---

## Tier 1 — strong targets

### Venter1983 — *Transformed beta and gamma distributions and aggregate losses* (PCAS)
- **Topic:** introduces the transformed gamma (= generalized gamma) and
  transformed beta families (generalizing gamma, Weibull, Pareto, Burr,
  loglogistic, F). Uses the transformed gamma to model the **aggregate**
  distribution by matching three moments; the transformed beta adds parameter
  uncertainty on the scale λ.
- **Worked examples (explicit parameters):**
  1. **Hospital professional liability**, losses limited to \$1M/occurrence:
     aggregate mean \$219,316, CV 1.550, skewness 2.510, P(no loss)=.123.
     Positive part: mean 250,000, CV 1.409, skewness 2.344 → transformed gamma
     r=0.2478, α=1.470, λ=1.144×10⁻⁶. Full law
     Pr(L≤x)=.123+.877·G(x;0.2478,1.470,1.144×10⁻⁶). Excess ratio at \$1M =
     .0728 → excess expected losses \$18,200 (conditional), \$16,000 (entire).
  2. **Parameter-uncertainty extension:** industry loss ratios .505/.750/1.001/
     1.357 → transformed beta r=0.2478, s=2.597, α=1.470, β=1,288,500. Excess
     ratio at \$1M rises to .1348 → \$33,700 (vs \$18,200).
  3. **Exhibit 3 — the reproduction target.** A full comparative aggregate-loss
     table over \$25k–\$850k in \$25k steps for three methods: characteristic
     function (exact), recursive Adelson/Panjer (exact), and transformed gamma
     (approx). Assumptions: **frequency Poisson λ=13.7376**; **severity** a
     piecewise-linear CDF + discrete PDF at \$500 intervals (severity mean
     18,198, CV 2.6600, skewness 3.6746); aggregate mean 250,000, CV 0.7667,
     skewness 1.0744; transformed-gamma fit r=0.5613125, α=1.8300318,
     λ=1/417896.414.
- **Replication fit:** **Strong.** The two exact columns are exactly what
  `aggregate`'s FFT computes — build Poisson(13.7376) × the discrete severity
  and the cumulative-probability / excess-ratio columns should match to the
  paper's 4 dp. The transformed-gamma column is also reproducible because the
  transformed gamma **is** scipy `gengamma`, so `aggregate` can build the
  moment-matched approximation too and show all three side by side.
- **What you'd reproduce:** Exhibit 3 Part 1 (the 34-row, 3-method table). Bonus
  extension: the hospital example with its point mass at zero and the
  transformed-beta parameter-uncertainty loading (\$18,200 → \$33,700).
- **Caveats:** the severity is a slightly intricate piecewise-linear + discrete
  construction that must be transcribed; the zero-mass / conditional-positive
  handling in example 1 needs the `p(no loss)` mixture.

### Mack2003 — *Exposure-rating in liability reinsurance* (Mack & Fackler)
- **Topic:** exposure rating for XL liability via Riebesell increased-limits,
  grounded in the collective risk model and Pareto tails.
- **Examples / parameters:** Riebesell z=20% ⇒ Pareto α=0.737; lognormal σ=1.8
  baseline spliced with a Pareto tail above u≈5·E(X); loss-elimination ratio
  r(a)=(a/v)^0.263; Table 1 threshold values for exponential/gamma/lognormal/
  Pareto; mixture giving observed tail α≈α+β≈2.0 (with β=1.3 sum-insured shape).
- **Fit:** **Strong** — collective model + XL layers + Pareto tails + exposure
  curves are squarely `aggregate`'s domain; severity splicing + layer pricing.
- **Reproduce:** Table 1 thresholds; the §7 exposure curve (premium vs limit)
  via LEV; the §8 mixture tail parameter.
- **Caveats:** α<1 Pareto has infinite mean (truncate by limit); the paper gives
  the exposure curve, not a full aggregate distribution — add a frequency to go
  end-to-end.

### Mata2005 — *Improved experience rating of XOL treaties using exposure rating*
- **Topic:** layer-by-layer experience/exposure rating with trend and exposure
  adjustment.
- **Parameters:** lognormal μ=9.31 σ=2.29 (2005), μ=8.93 (2000); 8% unlimited
  severity trend; limit profile \$250k–\$5M with premium weights; layers
  (\$250k xs \$250k, \$500k xs \$500k, \$4M xs \$1M, …); 60% ground-up ELR;
  layer trend split into frequency/severity components; exposure-adjustment
  factors (e.g. 2.0× for \$4M xs \$1M).
- **Fit:** **Strong** — self-contained, fully numeric; exercises LEV, limits,
  layers, ILFs.
- **Reproduce:** Tables 1–5 and 9–14 (LEV across limits/layers, layer trend
  decomposition, exposure adjustment).
- **Caveats:** the paper trends aggregate losses deterministically; `aggregate`
  reproduces the LEV/exposure pieces, the trending is a post-step.

### Bruno2006 — *A new method for evaluating the distribution of aggregate claims*
- **Topic / method:** thresholded direct convolution (error bound 10⁻⁶);
  competitor to FFT/Panjer.
- **Examples:** explicit 14-point discrete severity (Table 1) × Poisson(λ) for
  λ=104.8…5004.8; **tabulated PMF and CDF** at ~23 quantiles (Table 6,
  λ=504.8: μ=21,586, σ=1,048), Table 7 (λ=91,000), Table 8 (Poisson–
  generalized-Pareto mixture, μ=10.23). Timing tables vs Panjer/De Pril/FFT.
- **Fit:** **Strong** — both inputs and outputs are explicit; `aggregate`'s FFT
  should match the tabulated CDF to ≥6 dp.
- **Reproduce:** Tables 6/7/8 CDF columns.
- **Caveats:** discrete integer severity; Poisson / mixed-Poisson only.

### Jin2016 — *Moment-based density approximations for aggregate losses* (Jin, Provost, Ren)
- **Topic / method:** gamma-polynomial moment approximation, univariate +
  bivariate, benchmarked against FFT.
- **Examples (univariate 1–6):** Poisson(3)×γ(3,2), Poisson(15)×IG(3,2),
  NB(10,5)×γ(3,2), NB(10,5)×IG(3,2), Poisson(3)×Pareto(20,100), NB(3,2)×
  Pareto(20,100). All six build in one line of DecL.
- **Fit:** ~~**Strong**~~ **Weak. This entry was wrong; corrected after reading
  the paper. See `Jin2016.qmd` for the evidence.**
- **What the univariate half actually publishes:** figures, and nothing else.
  Tables 1–6 are Kolmogorov–Smirnov statistics between a *simulated* empirical
  distribution and the approximation fitted to that sample's own moments, so they
  are not properties of the model and not reproducible (no seed is given).
  **Exactly one univariate number is checkable**, Example 1's base gamma
  (2.684349, 7.056877), and `aggregate` reproduces it to six figures.
- **What the bivariate half publishes:** 96 genuine reference values in the FFT
  columns of Tables 7, 9, 11, 13, 15, 18, 21, 23.
- **Why `aggregate` cannot claim them:** the dependence is Hesselager common-shock
  *counts*, N = N₀+N₁ and W = N₀+N₂ with independent severities. `aggregate`'s
  `bivariate` is a shared outer frequency with **copula-coupled severities**.
  Thinning the total count shows the required trigger co-occurrence λ₀/λ is
  exactly the Fréchet *lower* bound p_X + p_Y − 1, so the only copula that fixes
  the triggers is the countermonotone one, which then also forces the severities
  to be countermonotone when Hesselager needs them independent. Structural
  mismatch, not a parameterization difficulty. Marginals alone do not rescue it:
  the common shock moves the joint CDF up to 4.5 points from the independent
  product.
- **Verified anyway:** a twelve-line direct 2-D transform over `aggregate`'s
  severity discretization reproduces Tables 7 and 11 to 1.3e-4 and 3.5e-4, the
  residual falling monotonically under grid refinement. The paper's FFT column is
  right; only the route is missing.
- **Feature that would unlock it:** a common-shock count mode for `bivariate`
  (shared N₀ plus private N₁, N₂, severities independent throughout). Would turn
  96 published values into a regression suite and give `aggregate` a
  count-driven dependence beside its copula-driven one.
- **Defect noted:** bivariate Example 6 reads "N₀ ~ NB(5), N₁ ~ NB(6),
  N₂ ~ NB(7)", one parameter for a two-parameter family. β = 5 is the obvious
  reading from the neighbouring examples but it is a guess.
- **Conversions worth keeping:** IG(η, θ) = `{θ} * invgauss {η/θ}`;
  Pareto(γ, δ) with density γδ^γ/(u+δ)^(γ+1) = `{δ} * lomax {γ}`;
  NB(r, β) = gamma-mixed Poisson, mean rβ, **mixing CV 1/√r**, independent of β.

### Bodoff2017 — *An Actuarial Model of Excess of Policy Limits Losses*
- **Not** the Bodoff already reproduced in the repo (that is **Bodoff 2007**,
  "Capital Allocation by Percentile Layer", `docs/5_technical_guides/
  5_x_bodoff.rst`). Distinct key, no duplication.
- **Example:** Pareto(θ=50,000, α=1.5), 9-point policy-limit profile, Bernoulli
  XPL effectiveness p=0.99; Exhibit 4 layer loss % with/without XPL loading
  (e.g. \$5M xs \$5M: 0.031% → 0.068%).
- **Fit:** **Strong** — novel and reproducible and very easy in DecL, but thin (single severity, no
  frequency given). Pairs conceptually with Braithwaite (ECO/XPL).

---

## Tier 2 — good, with a caveat

### Berens1997 — *Reinsurance contracts with a multi-year aggregate limit*
- **Examples:** Poisson(30)×lognormal(mean 150k, sd 37.5k) and NegBin×Pareto
  (q=1.5) across layers \$150k xs \$150k…\$600k; annual aggregate 2–3× expected;
  clash variant (5% two-claimant, \$300k cap) → 160.9% surcharge. Real-data
  example \$8M xs \$2M.
- **Fit:** **Moderate→Strong.** Single-year layer pieces are native; the
  multi-year **rolling** aggregate limit is path-dependent and needs a wrapper
  loop over independent annual builds with cumulative capping.
- **Reproduce:** the NB/Pareto surcharge tables (Exhibit III); good "extensions"
  story for the carry-forward.

### Clark2005 — *Reinsurance Applications for the RMK Framework*
- **Examples:** three independent lognormal lines (premiums 1,250/1,875/2,150,
  CVs 0.50/0.50/1.00), aggregate stop-loss 20 pts xs 80% LR; profit-commission
  and capital-allocation variants.
- **Fit:** **Moderate.** `aggregate` gives the exact compound distribution and
  stop-loss premium natively; the RMK co-measure **allocation** is post-hoc
  scenario logic, not built in. Clean Portfolio + distortion/allocation demo.
- **Reproduce:** Example 1 stop-loss cost and per-line contribution.

### Kang2019 — *Moment-based density approximation for heavy-tailed distributions*
- **Example:** Pareto-II(α=3.5, β=3.5) VaR/TVaR at κ=0.8/0.9/0.99/0.999 (e.g.
  VaR₀.₉₉=9.547, TVaR₀.₉₉=14.765), with exact values; plus real Danish-fire and
  auto-claims data.
- **Fit:** **Moderate.** The "aggregate" here is severity-only (frequency=1) —
  no compounding. Valuable as a heavy-tail VaR/TVaR accuracy benchmark and a
  custom-severity showcase.

### Lau1984 — *An Effective Approach for Estimating the Aggregate Loss …*
- **Example:** severity moments (μ=1700, …), frequency moments (μ=40.5, …),
  resulting aggregate moments (μ=68,850, skew 1.196, kurt 3.970), MPY at
  α=0.01 = \$265,640 via Pearson-curve fitting.
- **Fit:** **Moderate.** Only moments are given, not the severity form — must
  reverse-fit a distribution (ambiguous). Best framed as "`aggregate` computes
  the exact quantile vs Lau's Pearson approximation."

---

## Tier 3 — weak as reproductions

- **Goffard2020** — orthogonal-polynomial stop-loss. Test cases (e.g.
  Poisson(2)×Gamma, Poisson(4)×Pareto) are **graphs only**; just one Laplace-
  inversion error table. Reproducible only by digitizing figures; the
  Pascal×Gamma(1,·) closed form is the one clean exact check.
- **Nadarajah2016** — critique of Jin2016 across 8k–160k parameter sweeps;
  reports "% of cases method X wins", **no concrete output tables**. Use only to
  validate the exact PDFs (its Eqs 8–11).
- **Vernic1999** — genuinely **bivariate** compound (correlated/bivariate
  Poisson, trinomial frequency). `aggregate` is univariate → marginals only,
  not the joint dependence that is the paper's whole point.
- **Bakar2022** — severity-only density-hazard family (E-LN etc.) fit to
  indemnity/auto/fire data (e.g. E-LN auto: k=1.463, μ=7.342, σ=1.194). No
  compounding, so `aggregate`'s value-add is unused; at best a custom-severity
  cookbook entry.
- **Provost2005** — general-statistics moment approximation (distance in a cube,
  Wilks' Λ). No insurance content, no compound models. Not a target.
- **Hilbe2014** — textbook on count **regression** (Poisson/NB/PIG with
  covariates). `aggregate` uses count *distributions*, not regression. A
  reference for frequency math, not a reproduction.

---

## Already harvested — severity curves in the shipped library

Section F of `src/aggregate/agg/actuarial-severity-curves.agg` already carries
reusable `sev` objects named `<CitationKey>.<Curve>`, transcribed from this
assessment, for six of the papers:

| Key | Objects |
|---|---|
| Berens1997 | `Lognorm`, `Pareto` |
| Mack2003 | `Lognorm`, `RiebesellPareto`, `AmericanPareto` |
| Mata2005 | `Lognorm` |
| Bodoff2017 | `Pareto` |
| Jin2016 | `Gamma`, `Pareto` |
| Kang2019 | `Pareto` |

So a reproduction of any of those six starts from a named severity rather than
from scratch. The library is **not** in the default `build` databases
(`build.databases` is `('examples',)`), so load it explicitly, and give the
heavy members a limit — `Bodoff2017.Pareto` and `Mack2003.RiebesellPareto` have
no finite variance (α=1.5, α=0.737) and raise `InfiniteVarianceError` unbounded:

```python
uw = Underwriter(databases='actuarial-severity-curves')
uw.build('agg t 1 claim 1000000 xs 0 sev.Bodoff2017.Pareto fixed')   # mean 78,178
uw.build('agg v 3 claims sev.Jin2016.Gamma poisson')                 # mean 18, thin: no limit needed
```

The Tier-1 targets with **nothing** harvested yet are **Venter1983**
(piecewise-linear CDF + discrete PDF at \$500 intervals, Poisson λ=13.7376) and
**Bruno2006** (14-point discrete severity), both of which need the severity
transcribed first.

## Notes / housekeeping for the author

- **Citation keys are confirmed.** All eighteen keys above exist verbatim in
  `C:/s/TELOS/Biblio/uber-library.bib` (checked 2026-07-27, one entry each):
  `Venter1983`, `Berens1997`, `Braithwaite1997`, `Clark2005`, `Goffard2020`,
  `Vernic1999`, `Mack2003`, `Mata2005`, `Bodoff2017`, `Hilbe2014`, `Jin2016`,
  `Nadarajah2016`, `Lau1984`, `Bruno2006`, `Bakar2022`, `Provost2005`,
  `Lindsay2000`, `Kang2019`. Cite them directly — no re-verification needed, and
  nothing needs adding via archivum.
- **Venter1983 added.** PDF copied into the archivum store at
  `…/archivum/docs/42/4280F0882F_1983_Venter_transformed beta gamma
  distributions aggregate losses.pdf`, replacing a broken `.crdownload` link. (A
  full-text extract under `…/full-text/42/` will appear when archivum next
  processes it.)
- **Lindsay2000** has no full-text extract in `…/full-text/25/`; not assessed.
