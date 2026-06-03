# Plan: integrated multivariate aggregate distributions (copula-first)

> **Status:** REVISED DRAFT (2026-06-03), after negative-x agg (1.0.0a21), the
> signed Portfolio combine (1.0.0a22) and `pnl` (1.0.0a23). The prerequisite
> signed-axis / window machinery is **now shipped**, so this plan is no longer
> gated on negative-x. **Author decisions (2026-06-03):**
>
> 1. **v1 headline = the bivariate copula.** A copula couples the two
>    **per-claim severities**, which feed the shared-frequency **2D** FFT (the
>    `occ_bivariate` backbone) — the "one event, two correlated perils" model.
>    *Not* output-coupling. When a copula is used the object is **bivariate**.
> 2. **Copulas implemented à la `Distortion`** (a `_registry` + `__init_subclass__`
>    + `__new__` dispatch + `param_name`). Ship **normal, gumbel, clayton, fgm**
>    in v1; **t** (the only two-parameter kind: `rho`, `df`) deferred.
> 3. **Components may be `agg` or `pnl` — both in v1.** A `pnl` axis is the
>    per-claim loss severity through the copula+FFT, then a **per-axis affine**
>    (reflect + premium shift) applied to that axis of the joint tensor (§6).
> 4. **The FFT backbone is dimension-aware:** `rfft2` for the bivariate (copula)
>    case, `rfftn` only for the general ≥3-variate shared-frequency case (later).
> 5. **Two new modules:** **`copula.py`** (the `Copula` hierarchy) and
>    **`multivariate.py`** (the `MultivariateAggregate` class + builders). The
>    a20 `bivariate.py` is **subsumed**: its container (`BivariateDistribution`)
>    and helpers (`size_axis`, `scatter_bivariate`) move into `multivariate.py`;
>    `Aggregate.occ_bivariate` repoints its import; `bivariate.py` is reduced to a
>    thin back-compat re-export (or removed, updating its importers).
> 6. **Ships as `1.0.0a24`** (bump `pyproject.toml`; README; move plan to
>    `dev/done` when complete).
>
> The plan is **not** time-boxed to one night — do it properly. This subsumes the
> a20 `occ_bivariate` / `bivariate.py` into a first-class facility with its own
> DecL, class, and reporting.

---

## 1. Motivation

`occ_bivariate` (a20) proved the backbone: the joint law of two dependent
aggregate quantities is an ordinary compound-distribution FFT with the 1D
transforms replaced by 2D transforms, because `freq_pgf(n, z)` is *elementwise*
in `z` and so applies unchanged to a 2D transform. We now want this as a
**modelled object**, declared in DecL, where the per-claim dependence between
two perils is set by a **copula**:

```
# one event drives two correlated perils; gumbel upper-tail dependence
multivariate Cat 25 claims
    agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
    agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
    copula gumbel 0.4                 # Kendall tau = 0.4
    mixed gamma .2                    # shared mixing (count) -> extra common shock
```

Per event a claim triggers Wind w.p. `.7` and Flood w.p. `.5` (the `[0 1]`
Bernoulli zero-inflation), with severities coupled by the copula; `N` such
events accumulate to the **joint** aggregate `(A_Wind, A_Flood)`. Marginalising
either axis reproduces that line's standalone `Aggregate` — the exact validation
target.

---

## 2. The construction (bivariate, copula on per-claim severities)

Everything is: **build the joint per-claim severity `S` on a 2D grid →
`density = iFFT2(freq_pgf(N, FFT2(S)))` → report.** Only the *severity builder*
is new versus `occ_bivariate` (which scatters gross mass onto the comonotone
anti-diagonal `c+n=X`); here `S` is built from a **copula**.

### 2.1 Per-claim marginals (each component)
Each component is a **severity factory**: an inner `agg` restricted to the
zero-inflated `dfreq [0 1] [p0 p1]` Bernoulli form, so its per-claim severity is

```
g_i = (1 - p_i) delta_0  +  p_i f_i
```

(`f_i` the line's conditional severity; the atom at 0 is "peril i not triggered
this event"). These `g_i` are exactly what the inner `Aggregate` already
discretises (`sev_density` including the zero bucket). Severities may be mixed /
spliced / scaled / shifted, including signed (P&L) — that axis then carries the
negative-x offset / per-axis affine (v1; §6).

### 2.2 Joint per-claim severity via the copula (discrete Sklar)
With marginal per-claim CDFs `G_i = cumsum(g_i)` (which have a **jump at 0** from
the zero-inflation) and a copula CDF `C(u, v)`, the joint per-claim mass on the
2D grid is the **rectangle probability**

```
S[i, j] = C(G1[i], G2[j]) - C(G1[i-1], G2[j]) - C(G1[i], G2[j-1]) + C(G1[i-1], G2[j-1])
```

(`G[-1] := 0`). This is exact for the given discrete marginals and any copula —
atoms (the mass at 0) are handled automatically as jumps in `G`. The marginals
of `S` are `g1, g2` by construction, so each aggregate marginal reproduces the
standalone line. The copula sets the **per-claim** peril dependence (e.g. gumbel
upper-tail: both perils large together).

`C(u, v)` is evaluated once on the outer grid of CDF breakpoints
(`n_c x n_n`), then differenced — `O(n_c n_n)`. The grids are the **severity**
axes (coarse, ~2^8–2^10), *not* the aggregate axes, so this is cheap for the
closed-form copulas and ~2–3s for the Gaussian CDF at 512^2 (perf watch §8).

### 2.3 The 2D compound FFT (reuse `occ_bivariate`'s backbone)
Identical to `occ_bivariate` with `scatter_bivariate` swapped for the copula
builder, and with the **dimension-aware** transform (`rfft2` here):

```python
z      = sfft.rfft2(S, s=(n_c<<pad, n_n<<pad))
ftagg  = self.frequency.freq_pgf(self.n, z.ravel()).reshape(z.shape)   # ravel/reshape: a20 lesson
density = np.real(sfft.irfft2(ftagg, s=...))[:n_c, :n_n]
```

The fixed-1 / zero-risk shortcuts carry over. The shared frequency adds a second
layer of dependence (Poisson ⇒ count-only; `mixed` ⇒ common-shock on top of the
copula) — no special-casing, the existing `freq_pgf` families deliver the joint
law.

---

## 3. The `Copula` class hierarchy (à la `Distortion`)

New module **`src/aggregate/copula.py`** (submodule access only; no top-level
re-export, per CLAUDE.md). Mirror the `Distortion` machinery (`spectral.py:314`):
`_registry`, `__init_subclass__` (register by `kind`), `__new__` (dispatch on the
name string), a `param_name` class attr for the single-parameter kinds, and an
`__init__` override for the two-parameter `t`.

```python
class Copula:
    _registry: dict[str, type] = {}
    kind: str = ''
    param_name: str | None = None     # 'tau' | 'rho' | 'rho_s'; None for t
    long_name: str = ''
    # factory: Copula('gumbel', tau=0.4) or Copula('gumbel', 0.4)
    def C(self, u, v): ...             # copula CDF, vectorised over arrays
    def tau(self): ...                 # Kendall's tau (for reporting)
    # _build(): convert the natural param to the internal theta/rho/alpha
```

### 3.1 The five copulas — natural parameter and CDF

| kind | natural param | internal | `C(u, v)` | dependence |
|---|---|---|---|---|
| **normal** | Pearson `rho` ∈ (−1, 1) | `rho` | `Φ_ρ(Φ⁻¹u, Φ⁻¹v)` (scipy bivariate-normal CDF) | full range, no tail dep |
| **gumbel** | Kendall `tau` ∈ [0, 1) | `θ = 1/(1−τ)` ≥ 1 | `exp(−((−ln u)^θ + (−ln v)^θ)^{1/θ})` | upper-tail |
| **clayton** | Kendall `tau` ∈ [0, 1) | `θ = 2τ/(1−τ)` > 0 | `(u^{−θ} + v^{−θ} − 1)^{−1/θ}` | lower-tail |
| **fgm** | Spearman `rho_s` ∈ [−1/3, 1/3] | `α = 3·rho_s` ∈ [−1, 1] | `u v (1 + α(1−u)(1−v))` | weak only |
| **t** *(Stage 2)* | `rho` + `df` | `rho`, `ν` | `t_{ρ,ν}(t⁻¹_ν u, t⁻¹_ν v)` (bivariate-t CDF) | both tails |

- **normal**: evaluate `Φ_ρ` via `scipy.stats.multivariate_normal.cdf` over the
  breakpoint grid (verified ~2.8s @ 512²; an optional Drezner–Wesolowsky vectorised
  bivariate-normal CDF is a clean later optimisation). Natural param **rho**
  (linear correlation of the normal scores); offer `from_tau` (ρ = sin(πτ/2)).
- **gumbel / clayton / fgm**: closed-form `C`, vectorised numpy, instant.
- **t**: the only two-parameter kind — overrides `__init__(self, name='t', *, rho, df)`.
  The bivariate-t CDF is the at-risk piece (scipy `multivariate_t.cdf` is
  slow/version-flaky); land behind the other four with a numerical (Genz)
  fallback if needed.

### 3.2 DecL slot
A `copula` clause in the `multivariate` body (see §4). The kind is an ID, the
param(s) are `numbers`; the natural-parameter meaning is per-kind in the class.

---

## 4. DecL design

### 4.1 New statement + copula clause
```lark
answer: ...
      | mv_out          -> answer_mv

mv_out: MULTIVARIATE name exposures mv_body copula_clause freq note  -> mv_out_copula
//     (the ≥3-variate shared-frequency, no-copula form is a later stage)

mv_body: mv_body agg_out   -> mv_body_cons
       | agg_out           -> mv_body_one    // exactly TWO for the copula form (validated)

copula_clause: COPULA ID numbers          -> copula_one_param   // gumbel/clayton/fgm/normal
             | COPULA ID numbers numbers   -> copula_two_param   // t: rho df
```

Terminals (priority 2, same `(?![namechar])` lookahead discipline), added to the
`ID` negative-lookahead exclusion list and the parser keyword handling:

```lark
MULTIVARIATE.2: /(?:multivariate|mv)(?![a-zA-Z0-9._:~\-])/
COPULA.2:       /copula(?![a-zA-Z0-9._:~\-])/
```

- Outer `exposures … freq` is the **shared count** (reuses `agg_out_full`'s
  exposure/frequency nonterminals + transformer verbatim).
- Body = the component severity factories (exactly **two** for the copula form).
- The copula kind ID is resolved by `Copula(kind, *params)` in the transformer.

### 4.2 Validation (transformer / class)
- copula form: **exactly two** components; each a `dfreq [0 1] […]` Bernoulli
  severity factory (`agg` or `pnl`; hard-error on any other frequency — the outer
  `freq` owns the count); copula kind in the registry; param count matches the
  kind (1; t is deferred).
- A `pnl` component uses the dfreq form `pnl A <premium> prem - dfreq [0 1] [p0 p1]
  sev …` (the existing `pnl_out_dfreq` production); its premium becomes the
  per-axis affine (§6). Components may also be signed via `ssev` / negative
  `dsev`.

### 4.3 `kind` / factory
`_factory` (`underwriter.py`) gains `kind == 'mvagg'` → `MultivariateAggregate(**spec)`.
`spec` mirrors the port shape: `{'name', 'exposures'/'en', 'freq…', 'lines':
[two inner agg specs], 'copula': Copula(...), 'mode': 'copula', 'note'}`.
`build()` / `build_many()` need no change beyond the new kind flowing through.

---

## 5. Class: `MultivariateAggregate` (new `multivariate.py`)

Bivariate in v1. **`multivariate.py` subsumes `bivariate.py`:** move
`BivariateDistribution` (which already provides `marginals`, `moments`
(`E[CⁱNʲ]`), `corr`, `contour`, `_repr_html_`) and the helpers `size_axis` /
`scatter_bivariate` into the new module; `MultivariateAggregate` *is* the
modelled object and holds (or owns) the joint-density container. Repoint
`Aggregate.occ_bivariate` and update the `tests/test_reins_bivariate.py` /
any `from aggregate.bivariate import …` to the new module; leave a thin
`bivariate.py` re-export only if a smoke check shows external importers.
`MultivariateAggregate` holds:
- the shared `Frequency` (from the outer freq spec),
- the two inner `Aggregate` severity factories (built to get `g_i`, the per-claim
  `sev_density`; each also gives the **standalone aggregate** validation target),
- the `Copula` instance,
- per-axis grids/buckets, the joint severity `S`, and the joint `density`.

Methods (named to match the 1D counterparts):
- `update` / `update_work` — build `g1, g2` → `S` (copula §2.2) → `rfft2` →
  `freq_pgf(N, z.ravel()).reshape` → `irfft2` → store `density`. Per-axis sizing
  via the **shipped** `estimate_agg_window` / `size_axis` (so a signed axis sizes
  two-sided); record the per-axis deficit.
- `density_df` — house-style frame: axis grids + joint `p` + marginal columns
  `p_<line1>`, `p_<line2>` (axis sums). 2-D reshape of `BivariateDistribution`.
- `stats_df` — per-line marginal moments (theoretical vs empirical via
  `xsden_to_mwrangler` on each axis-sum) **plus** the joint block: `Cov`, Pearson
  `corr`, the realised Kendall τ, and `E[A1 A2]`. (Realised output corr ≠ the
  copula param — compounding attenuates it — report both.)
- `describe` — per-line summary + correlation. **Signed-aware:** when an axis is
  a `pnl` / signed, reuse the a23 **SD-not-CV** logic (`_describe_signed`).
- `info` — shape, axes/buckets/windows, shared-freq description, copula kind +
  natural param + implied τ, per-axis tail deficit.
- `contour` / pairwise plot — reuse `BivariateDistribution.contour`.

**Validation showpiece:** each marginal reproduces the corresponding standalone
`Aggregate` (means exact; cv at the matched grid — the a20 linear-rebucketing
caveat recurs per axis). A comonotone-limit copula (e.g. gumbel τ→1) should
approach the `occ_bivariate` anti-diagonal; the independence copula (fgm α=0, or
gumbel τ=0) factorises `S = g1 ⊗ g2`.

---

## 6. Components as `agg` or `pnl` (both v1)

A `pnl` component contributes its **loss** severity `g_i` to the copula+FFT
exactly like an `agg` (the aggregate-level affine never touches `sev_density`).
Its premium is a **per-axis affine** (reflect + shift) applied to axis *i* of the
joint `density` *after* the 2D FFT — the 1D `_apply_agg_affine` relabel lifted to
one tensor axis (reverse + roll along axis *i*, relabel that axis's grid to the
tight P&L window via the shipped `_pnl_window`). Because the affine is a pure
per-axis grid relabel it commutes with marginalisation, so marginal *i* = the
standalone `pnl` and marginal *j* is unchanged; the copula's loss-loss dependence
becomes the correct profit-loss sign once the axis is reflected. The inner
`Aggregate` already carries `_agg_reflect` / `_agg_shift`, so the orchestrator
just reads them and applies the per-axis relabel. Implemented **in v1**,
alongside the plain-`agg` path.

---

## 7. Staged implementation

1. **Stage 1a — `copula.py`.** The `Copula` hierarchy + **normal, gumbel,
   clayton, fgm** (Distortion-style registry/dispatch), with CDF + `tau()` unit
   tests. Self-contained, lands first.
1. **Stage 1b — `multivariate.py` + DecL (the headline, v1).** Subsume
   `bivariate.py` (container + helpers, repoint `occ_bivariate`). `multivariate`
   statement + `copula` clause + `kind='mvagg'`; `MultivariateAggregate` with the
   §2.2 copula severity builder over the §2.3 `rfft2` backbone;
   marginals/moments/corr, `density_df`/`stats_df`/`info`/`contour`. **Both `agg`
   and `pnl` axes** (§6, per-axis affine). Poisson (count-only) and `mixed` (extra
   common shock) verified against standalone-agg/pnl marginals. Bump to
   `1.0.0a24`, README, move plan to `dev/done`.
2. **Stage 2 — the `t` copula** (bivariate-t CDF, numerical Genz fallback if
   scipy balks).
3. **Stage 3 — reporting polish & plots** (pairwise contours, correlation +
   τ block, docs subsection extending the a20 "Joint Distribution" section).
4. **Stage 4 — ≥3-variate shared-frequency path** (no copula; outer-product `S`
   via `rfftn`, the dimension-aware general case; memory cap §8).
5. **Stage 5 (later) — `MultivariatePortfolio`** and reconciling/retiring
   `occ_bivariate` against a `netceded` DecL form.

---

## 8. Risks / watch

- **Grid feasibility (the real gotcha).** The joint is `n_c x n_n`; the
  **severity** axes are coarse (≈2^8–2^10) so this is fine, but the per-axis
  sizing must keep each `n_i` small while covering the support. The aggregate
  axes are never multiplied out. Cap and auto-size per axis.
- **Gaussian-copula CDF perf** — ~2.8s @ 512² via scipy; acceptable for v1.
  Optional Drezner–Wesolowsky vectorised bivariate-normal CDF later. The
  closed-form copulas are instant.
- **t-copula CDF** — the one MEDIUM-HIGH risk; deferred to Stage 3 with a Genz
  numerical fallback.
- **ND `freq_pgf` elementwise** — `ravel/reshape` everywhere (a20 lesson); the
  bivariate path inherits this from `occ_bivariate`. Add a per-family guard/test
  for any frequency used as the shared count.
- **Realised vs specified dependence** — the output aggregate correlation ≠ the
  copula parameter (compounding attenuates it); report both the copula τ and the
  realised joint τ/ρ so this is never confused.
- **Marginal-vs-grid accuracy** — the a20 linear-rebucketing second-moment
  finding recurs per axis: means match on any grid, cv at the matched grid.
- **DecL ambiguity** — new keywords must not capture identifiers; reuse the
  established lookahead + `ID` exclusion discipline; add parse-only tests.
- **Memory blow-up** beyond 2 axes — only relevant once Stage 4 (`rfftn`) lands;
  enforce the ~3–4 axis cap there.

---

## 9. Testing (`tests/test_multivariate.py`)

- **Copula CDF unit tests** — each kind: `C(u,0)=C(0,v)=0`, `C(u,1)=u`,
  `C(1,v)=v`, monotone, and the natural-param→τ identities (gumbel `τ=1−1/θ`,
  clayton `τ=θ/(θ+2)`, fgm `ρ_s=α/3`); the `Copula('gumbel', tau=.4)` /
  positional / `__new__` dispatch (mirrors the Distortion factory tests).
- **Marginals reproduce standalone aggs** — `multivariate … copula … poisson` →
  each axis-sum matches `build('agg Wind …')` / `build('agg Flood …')` (means
  exact; cv at matched grid).
- **Independence copula factorises** — fgm α=0 (or gumbel τ=0) ⇒ `S = g1 ⊗ g2`
  and `corr ≈ count-only`.
- **Comonotone limit** — gumbel τ→1 approaches the `occ_bivariate` anti-diagonal
  concentration.
- **Dependence ordering** — realised joint corr increases with the copula param;
  `corr(mixed) > corr(poisson)` for the same copula (extra common shock).
- **Empirical outer count** (`dfreq`) exercises the `ravel/reshape` ND path.
- **Signed / `pnl` axis** (v1) — one `pnl` component → correct signed marginal
  (= the standalone `pnl`), and the realised joint corr flips sign vs the
  loss-loss copula dependence.
- DecL programs appended to `src/aggregate/agg/test_decl.agg` under a new section.

Run `uv run pytest` (`UV_LINK_MODE=copy`); no docs build in-loop.

---

## 10. What this plan dropped vs the 2026-06-02 draft

- **Stage 0 (negative-x) removed** — shipped (a21/a22/a23); signed axes now reuse
  existing machinery.
- **Copula promoted from "Stage 5, future" to the v1 headline**, restricted to
  bivariate per the author, coupling **per-claim severities** (not outputs).
- **`pnl` axes are v1, not a later stage** — the per-axis affine on the joint
  tensor (§6); v1 combines `agg` *and* `pnl` components.
- **`bivariate.py` is subsumed into `multivariate.py`** (container + helpers
  moved, `occ_bivariate` repointed) — the a20 facility keeps working as the
  comonotone-scatter peer, just from the new home. A `netceded` DecL form is
  Stage 5.
- **Outer-product / shared-frequency generalisation and `MultivariatePortfolio`
  pushed to Stages 4–5** — the dimension-aware `rfftn` path is the >2 case.
- **`t` copula deferred to Stage 2** (bivariate-t CDF risk).
