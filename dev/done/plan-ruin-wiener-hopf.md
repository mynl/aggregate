# [Ruin-Wiener-Hopf] — PK Poisson guard, `Aggregate.wiener_hopf`, pedagogy `ruin_example`

## Context

`Aggregate.pollaczeck_khinchine` (PK) computes the eventual-ruin function psi(u) via the
classical geometric-compound formula. Its docstring says "Assumes frequency is Poisson"
but it never checks — it silently returns the compound-Poisson answer for any frequency.
Separately, the author's notes module `C:/s/AI/notes/ruin-probabilities/ruin.py` contains
a cepstral Wiener-Hopf solver for the Sparre-Andersen renewal model: psi(u) computed from
the random walk with per-claim step `Y = X - cW` (X severity, W inter-claim wait, c premium
rate), via Spitzer/cepstral factorization — exactly the case the library's renewal
(`years ... wait ...`) frequencies describe. This plan: (1) make PK raise for non-Poisson
frequency, (2) port the cepstral kernel as `Aggregate.wiener_hopf` with an interface
matching PK, (3) keep the specialized `plot_ruin_surplus_paths` (PIR Fig 9.1) but add a
**general** pedagogy function `ruin_example(agg, ...)` modeled on the notes' example
builder — psi + simulated surplus paths + summary — that takes any poisson or renewal
`Aggregate` and auto-dispatches PK/WH.

**Author decisions (asked and answered):**
- Name: **`wiener_hopf`** (correct spelling; fix the `Weiner` typo in plan-for-v1.md in passing).
- Return: **shared named tuple** for both methods — clears PK's standing
  `TODO: Should return a named tuple`. A namedtuple IS a tuple so all positional
  unpackings keep working.
- Defective (terminating) wait laws (`wait_defect > 0`, e.g. `dwait ... !`): **refuse with
  ValueError** (killed-walk semantics deferred).
- `plot_ruin_surplus_paths` **stays** (specialized); the general Aggregate-first function
  is new. Its `port` argument (answered from docs/5_technical_guides/5_x_pk.rst:245-256):
  the `PZTest` portfolio — units named exactly `Limit1`/`Limit10`, 0.1 claims,
  `sev lognorm 50000 cv 10`, 1M/10M xs 0 limits, poisson, `bs=500, log2=18, padding=1`.
  Document this requirement in its docstring.
- **Carry the notes' documentation over.** ruin.py's docstrings and inline comments
  (horizon heuristics, LIL funnel, rounding-for-aperiodicity, wrap diagnostics) and
  ruin.qmd's derivation are the source material for the new docstrings — port them
  generously, don't re-derive thinly.

## Step 0 — Bookkeeping at execution start

- The author will clear the tree before execution (one code job at a time). Verify with
  `git status` / `git log`; determine the release number as pyproject current + 1
  (expected **a153**, after the in-flight a152 commits).
- Save this plan as `dev/plan-ruin-wiener-hopf.md`; move to `dev/done/` at close.

## Step 1 — Kernel `ruin_cepstral` in `src/aggregate/_renewal.py`

- Add `from scipy.fft import fft, ifft` (complex transforms required by the phase unwrap —
  the rfft-based `utilities.ft/ift` cannot be used). Module stays a leaf.
- Add `'ruin_cepstral'` to `__all__` (line 40).
- Append the kernel verbatim from the notes module with a NumPy docstring whose Notes
  section ports the ruin.qmd derivation: Spitzer identity (`log(1-phi_Y)` splits into
  plus/minus factors with disjoint cepstral support), z=1 regularization by dividing out
  `1 - z^{-1}` (a pure minus factor, so plus quefrencies untouched; `Gt[0] = -mean_y` is
  the l'Hopital limit, positive under net profit, and zero winding makes `unwrap` close),
  support separation `cplus[1:M//2]`, geometric ladder compounding done in log space:

```python
def ruin_cepstral(fy, mean_y):
    M = len(fy)
    k = np.arange(M)
    G = 1.0 - fft(fy)
    d = 1.0 - np.exp(2j * np.pi * k / M)
    Gt = np.empty(M, dtype=complex)
    Gt[1:] = G[1:] / d[1:]
    Gt[0] = -mean_y
    logGt = np.log(np.abs(Gt)) + 1j * np.unwrap(np.angle(Gt))
    c_ = np.real(ifft(logGt))
    cplus = np.zeros(M)
    cplus[1:M // 2] = c_[1:M // 2]
    log_phi_M = cplus.sum() - fft(cplus)
    pmf_M = np.real(ifft(np.exp(log_phi_M)))
    psi = 1.0 - np.cumsum(pmf_M)
    return pmf_M, psi
```

Inputs: `fy` = wrapped step pmf of Y on the circular lattice (negative steps at wrapped
indices), `mean_y` = `signed @ fy` in bucket units, must be < 0. Returns `(pmf_M, psi)`
with `psi[j] = P(eventual ruin | u = j buckets)`.

## Step 2 — Shared plumbing in `src/aggregate/_aggregate.py`

Module level, before `class Aggregate` (~line 423):

- `RuinFunction = namedtuple('RuinFunction', ['ruin', 'find_u', 'mean', 'density'])`
  — `density` holds the method's u-grid density vector: the integrated-severity
  (equilibrium) density `dfi` for PK; the pmf of the all-time maximum for WH.
- `_ruin_find_u(ruin, kind)` — factor PK's two `find_u` closures (currently
  `_aggregate.py:3807-3820`, moved verbatim: `kind='index'` snaps, else interpolates).
- Import `ruin_cepstral` from `._renewal` (leaf module, no cycle;
  `discretize_severities` is already imported).

## Step 3 — PK guard + named tuple (`_aggregate.py:3755-3822`)

- Top of body, before the `sev_density` check:

```python
fname = getattr(self.frequency, 'freq_name', '')
if fname != 'poisson':
    hint = (' -- for a renewal (wait-clause) frequency use wiener_hopf'
            if fname == 'renewal' else '')
    raise ValueError(f'pollaczeck_khinchine assumes a Poisson frequency, '
                     f'got {fname!r}{hint}')
```

  Strict `'poisson'` only — mixed Poissons (gamma/delaporte/...) refused too.
- Replace the closure block with `find_u = _ruin_find_u(ruin, kind)`; return
  `RuinFunction(ruin, find_u, mean, dfi)`. Delete the TODO line; docstring gains the
  Raises note and names `RuinFunction`.

## Step 4 — `Aggregate.wiener_hopf` (insert directly after PK)

Signature: `wiener_hopf(self, rho, kind='index', log2=None)` — `rho` has PK's
margin-to-loss semantics; premium rate `c = (1 + rho) * E[X] / E[W]`.

Guards (in order):
1. `freq_name != 'renewal'` → ValueError (hint: poisson → use `pollaczeck_khinchine`).
2. `self.sev_density is None` → ValueError (must update first, like PK).
3. `self.i0` nonzero → ValueError (signed severity grid unsupported).
4. `self.frequency.wait_defect > 1e-12` → ValueError (defective/terminating wait law).
5. After building fy: `mean_y >= 0` → ValueError (net profit fails on the grid; raise rho).

Computation:
- Grid: half-grid `n = 2**log2` (default `log2 = self.log2`, so the u-grid equals the
  severity grid `arange(n) * bs` exactly; larger `log2` allowed for heavy tails,
  smaller refused); wrapped circle `M = 2n`.
- Severity: **reuse** `self.sev_density_df.p_sev` (already-discretized rounding pmf) —
  do not rediscretize; `mean = sum(bit * bit.index)` as in PK. (Rounding keeps the mean
  correct to O(bs^2) and makes the lattice walk aperiodic — port the notes' comment.)
- `E[W]`: exact unconditional mixture mean `wait_weights @ [sev.moms()[0] ...]` — exact
  precisely because the defect guard ensures unconditional windows lose no mass.
- Wait law on the monetary grid: factor as a **private helper**
  `Aggregate._discretize_wait_pmf(self, c, n)` returning `fcw` (length n, time buckets of
  width `bs/c`) so `pedagogy.ruin_example` can reuse it for simulation. Implementation
  reuses `discretize_severities(components, xsw, bs/c, sev_calc='discrete',
  discretization_calc='survival', normalize=False)` following `_renewal.wait_count_pmf`
  (lines 583-602) **including** unconditional-window masking but **without** T-truncation
  and **without** p0 removal — the walk needs the whole W law (zero-wait atoms ride along;
  bucket 0 = `P(W <= h/2)` matches the notes' rounding exactly, verified against
  `_aggregate_compute.py:213-333`). Weight-combine with `wait_weights`.
- Wrapped step pmf: `X[:len(bit)] = bit`; `Wn[0] = fcw[0]`, `Wn[M-n+1:] = fcw[1:][::-1]`;
  `fy = real(ifft(fft(X) * fft(Wn)))` (scipy complex fft); `mean_y = signed @ fy` where
  `signed = where(k < n, k, k - M)`.
- `pmf_max, psi = ruin_cepstral(fy, mean_y)`; wrap diagnostic: `warnings.warn` if
  `psi[n-1] > 1e-6`, advising a larger `log2` (heavy-tail X needs a wider grid before
  wrap-around contamination — port the notes' three-diagnostics discussion into Notes).
- Return `RuinFunction(pd.Series(psi[:n], index=arange(n)*bs), _ruin_find_u(ruin, kind),
  mean, pmf_max[:n])`.

Docstring: NumPy style, Notes drawing on ruin.qmd (Sparre-Andersen model, ruin only at
claim instants, Lindley walk `Y = X - cW`, cepstral factorization, grid conventions and
diagnostics); cite Embrechts-Kluppelberg-Mikosch as PK does, plus Spitzer.

## Step 5 — Pedagogy: dispatch + keep the specialized plot

Module-level dispatch helper in `src/aggregate/pedagogy.py`:

```python
def _ruin_function(ag, margin, *, kind, padding=1):
    fname = getattr(ag.frequency, 'freq_name', '')
    if fname == 'poisson':
        return ag.pollaczeck_khinchine(margin, kind=kind, padding=padding)
    if fname == 'renewal':
        return ag.wiener_hopf(margin, kind=kind)
    raise ValueError(f'no eventual-ruin solver for a {fname!r} frequency: need '
                     f'poisson (pollaczeck_khinchine) or renewal (wiener_hopf)')
```

- Route the three existing PK call sites through it (`ClassicalPremium.illustrate` :786,
  `plot_ruin_surplus_paths` :844, `natural_scale` :893) — behavior-identical for the
  poisson doc portfolios, and makes them renewal-capable/fail-loud for free.
- `plot_ruin_surplus_paths` otherwise **stays as-is** (PIR Fig 9.1). Upgrade its docstring
  to document the required `port`: the `PZTest` build from 5_x_pk.rst — units named
  exactly `Limit1` and `Limit10` (0.1 claims, lognorm 50000 cv 10, 1M/10M xs 0, poisson,
  bs=500, log2=18), calibration premium 110, margin 0.1 hard-coded.
- Double-compute cleanup: compute the outer psi with `padding=2` (what `illustrate` used
  internally) and pass `K=find_us[unit_name](p_default)` to `illustrate` instead of
  `p=p_default` — K numerically identical to today's, one PK per unit instead of two.

## Step 5b — New general `pedagogy.ruin_example(agg, ...)`

Port of the notes' `ruin_example` (the newer version, incl. LIL funnel / `t_plot` /
`show_default_times`; drop `ruin_example_old`), taking an `Aggregate` instead of raw
scipy distributions. Name `ruin_example` vetted — no hits in src/tests.

Signature:
`ruin_example(agg, rho, u0, *, log2=None, n_sims=100_000, n_plot=50, n_steps=None,
t_plot=None, show_default_times=False, seed=None)` → `(summary, fig)` as in the notes.

Mapping from the notes version:
- `dist_x, dist_w, c, bs, log2n` → derived from `agg`: psi via `_ruin_function(agg, rho,
  kind='index')` (poisson → PK, renewal → WH, else ValueError); `bs = agg.bs`;
  `c = (1 + rho) * E[X] / E[W]`.
- `E[W]`/`Var[W]` (needed for c, trend, LIL rate `sigma2 = (Var X + (EX/EW)^2 Var W)/EW`,
  horizon heuristics): poisson → `W ~ expon(1/lambda)`, `lambda = agg.n` (annual count),
  so `EW = 1/lambda`, `VarW = 1/lambda^2`; renewal → exact mixture moments from
  `wait_weights @ [sev.moms() ...]`. Severity moments from the discretized pmf (consistent
  with the psi computation).
- Simulation sampling (walk at claim instants, chunked to cap memory ~2e7 variates —
  port that logic and its comments verbatim):
  - severity: sample the discretized `sev_density_df.p_sev` grid (precedent:
    `ClassicalPremium.illustrate` samples `density_df`), via `rng.choice`/searchsorted on
    the cdf — works for every severity form.
  - waits: poisson → `rng.exponential(scale=EW)` (exact); renewal → sample the
    `_discretize_wait_pmf(c, n)` grid (times `= bucket * bs/c`) — works for dwait
    mixtures/splices with no per-component logic. Judgment call, flag in run summary:
    grid sampling makes the simulated and cepstral models identical, so the sim-vs-exact
    comparison in the summary is apples-to-apples (no discretization gap).
- Ported intact with their clever comments: auto `n_steps` (horizon from
  `|E Y| n - 3 sd sqrt(n) = u_safe` with `u_safe` read off the exact psi, cap 50,000),
  auto `t_plot` (residual ruin ~1e-3), pre/post-claim interleaved jump paths, expected
  trend, LIL funnel `±sqrt(2 sigma2 t ln ln t)`, ruin-time rug, title with sim-vs-exact
  rates, and the summary DataFrame (same rows: c, u0, means, cv2 waiting, theta, mean
  step, sigma2, psi(0)/psi(u0) exact, psi(u0) simulated, ruins/trials/se, horizons,
  mean ruin time, grid size, bs, grid top, mass-beyond-grid diagnostic).
- Docstring: full NumPy docstring built from the notes' (which is already good), plus the
  dispatch rule and the Aggregate-derived parameter mapping.

## Step 6 — Doc lockstep edit

`docs/5_technical_guides/5_x_pk.rst:215`: `... sev 4 * pareto 5 - 4 fixed` → `... poisson`
(PK numbers identical — it reads only severity; the `qd(a)` audit table re-renders with
Poisson stats, expected). New-feature docs (a `ruin_example` demo section) deferred —
docs rebuild is the author's, outside the loop; note "docs pending rebuild" in the run
summary.

## Step 7 — Tests: new `tests/test_ruin.py`

Conventions per `tests/test_renewal_agg.py` (module functions, `build`, autouse
quiet-warnings fixture). Force `matplotlib.use('Agg')` at module top for the plotting
test. Module-scope fixtures: `po_agg = build('agg RuPo 10 claims sev gamma 2 poisson',
bs=1/16, log2=12)` and `re_agg = build('agg RuRe 10 years sev gamma 2 wait expon',
bs=1/16, log2=12)` — the same model (test_renewal_agg.py:36 documents the equivalence).

1. **PK guard**: poisson works; `fixed` and `binomial 0.5` raise (match
   `'assumes a Poisson'`); renewal raises with the `wiener_hopf` hint.
2. **WH guards**: poisson raises (match `'pollaczeck_khinchine'`); defective
   `agg RuDef 2 years dsev [1] dwait [1 2] [.4 .5] !` raises (match `'defective'`).
3. **WH ≈ PK, exponential waits**: identical u-grid indexes; psi compared at
   u = 0, 4, 8, 16, 32 with `abs=5e-3` (PK's integrated-distribution build is O(bs)-
   biased, WH rounding O(bs^2); tighten after observing actuals); `psi(0) ≈ 1/(1+rho)`.
4. **Exponential-severity closed form, gamma waits** (genuinely non-Poisson):
   `agg RuExp 5 years sev 10 * expon wait gamma 2`, `bs=1/4, log2=12`. For expon severity
   under ANY wait law, `psi(u) = psi(0) * exp(-R u)` with `R = beta * (1 - psi(0))`,
   `beta = 1/10`. Test log-linearity of the slope between u = 20/60/100 (`rel=1e-3`) and
   the slope against R (`rel=1e-2`).
5. **Net-profit violation**: `wiener_hopf(-0.2)` raises (match `'net profit'`).
6. **Named tuple**: both methods return `RuinFunction`; fields accessible by name AND
   position; `find_u` callable and in-range.
7. **Dispatch**: `pedagogy._ruin_function` routes poisson→PK, renewal→WH, raises on
   `fixed` (no plotting exercised).
8. **ruin_example smoke**: renewal agg, small `n_sims` (~2000), `n_plot=10`, fixed seed;
   assert `(summary, fig)` returned, key summary rows present, simulated psi(u0) within
   ~4 standard errors of exact, `plt.close(fig)`.

Also append the new DecL programs (RuPo, RuRe, RuDef, RuExp, RuFix) to
`src/aggregate/agg/decl-testers.agg` under the matching sections (round-trip requirement).

## Step 8 — Release hygiene (one coherent commit, expected a153)

1. `pyproject.toml` → next version per Step 0.
2. `CHANGELOG.md` new section: WH method + kernel, PK strict guard (**flag as breaking**
   for non-Poisson PK callers incl. mixed Poissons), shared `RuinFunction` (positional
   unpack unchanged), pedagogy dispatch + new `ruin_example`, doc example fixed→poisson.
3. `dev/FEATURES.csv`: hand-add the `wiener_hopf` row next to `pollaczeck_khinchine`,
   then `uv run python dev/regen_features.py` must exit 0 (it audits, not regenerates).
4. `dev/TODO.md` (~line 476): amend `[Renewal-Frequency-Wait-Clause]` — ruin/Wiener-Hopf
   no longer out of scope; delayed/equilibrium first wait remains out.
5. `plan-for-v1.md:69-71`: tick all three boxes (PK poisson audit → Step 3; Wiener-Hopf
   extension → Step 4; pedagogy w-h version → Step 5b); fix `Weiner` → `Wiener`.
6. Move `dev/plan-ruin-wiener-hopf.md` → `dev/done/`.
7. Commit (one line, no body): `[Ruin-Wiener-Hopf] a153: Aggregate.wiener_hopf renewal
   ruin via cepstral factorization; PK poisson guard; shared RuinFunction; pedagogy
   ruin_example + dispatch`.

## Verification

- `uv run pytest tests/test_ruin.py -n0` first (observe actual tolerances; tighten test 3
  if actuals allow), then `uv run pytest tests/test_renewal_agg.py tests/test_fcc_surface.py`,
  then the full fast suite `uv run pytest` as the gate.
- Smoke: `build('agg R 10 years sev gamma 2 wait expon').wiener_hopf(.2)` — psi(0) near
  1/1.2; `ruin_example` on the notes' Case-1 analog (expon sev, expon waits, c=1.2, u0=5):
  exact PK closed form `(1/1.2) exp(-5/6)` should reproduce.
- `uv run python dev/regen_features.py` exits 0.

## Risks / notes

- Tree cleared before execution (author confirmed); Step 0 re-checks version regardless.
- PK guard is breaking for any downstream non-Poisson PK use; nothing in-repo does
  (docs/tests audited).
- Wait tail beyond the grid top is dropped (matches the notes module); the
  `Gt[0] = -mean_y` override absorbs the residual — documented in Notes, no guard.
- psi left raw (not clipped/monotonized), same as PK — kernel fidelity.
- Mixture/spliced wait discretization on the shared monetary grid is new relative to the
  single-scipy-dist notes module; single-component paths covered by tests 3-4; a
  mixed-wait test is a nice-to-have if the DecL wait grammar supports mixtures — check at
  execution.
- `ruin_example` wait sampling from the discretized grid (renewal case) is a flagged
  judgment call (see Step 5b) — exact `.rvs` per mixture component is the alternative if
  the author prefers continuous paths.

## Critical files

- `src/aggregate/_aggregate.py` — PK guard (3755-3822), new `wiener_hopf` +
  `_discretize_wait_pmf`, `RuinFunction`, `_ruin_find_u`
- `src/aggregate/_renewal.py` — `ruin_cepstral` kernel, `__all__`, scipy.fft import
- `src/aggregate/pedagogy.py` — `_ruin_function` dispatch, new `ruin_example`;
  call sites 786, 844, 893; `plot_ruin_surplus_paths` docstring
- `tests/test_ruin.py` (new), `src/aggregate/agg/decl-testers.agg`
- `docs/5_technical_guides/5_x_pk.rst:215`
- Hygiene: `pyproject.toml`, `CHANGELOG.md`, `dev/FEATURES.csv`, `dev/TODO.md`,
  `plan-for-v1.md`, `dev/plan-ruin-wiener-hopf.md`
