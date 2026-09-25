# Plan: [Cantor-Severity] the Cantor distribution as a first-class DecL severity

Status: reviewed with the author 2026-09-25; the four design questions at the
end are DECIDED as marked and the plan is written to those decisions. Not yet
approved for implementation.

## Goal

Make the (generalized) Cantor distribution available as a named DecL severity,
on the same footing as `uniform`:

```
agg C1  1 claim sev cantor              fixed        # standard middle-thirds
agg C2  1 claim sev cantor 0.5          fixed        # remove middle half
agg C3  1 claim sev 3 * cantor + 5      fixed        # scale and shift
agg C4 10 claims sev cantor poisson                  # as a real severity
```

with a working `cdf`, `sf`, `ppf`, `isf`, `rvs`, exact moments, and full
participation in the FFT pipeline (layers, mixtures, splices, reinsurance all
come along for free once the frozen distribution behaves). Extra credit, a
shape parameter equal to the proportion removed from the middle at each step.

This is deliberately a little illusionary: the Cantor law is singular
continuous, floats only see a countable dyadic skeleton, and the discretized
aggregate is what it always is, a lattice approximation. The point is that the
approximation is controlled and the exact structure (moments, cdf values,
self-similarity) is available analytically. See "Numerical honesty" below.

## Current state

- Nothing named `cantor` exists in `src/aggregate`. Working prototypes live in
  `hacks/`:
  - `hacks/cantor_cdf.py`: correctly rounded cdf of the standard middle-thirds
    Cantor measure, scalar (exact integer ternary digits) and vectorized
    (uint64) versions. Slower than the pmf construction but human-fast.
  - `hacks/cantor.py`: stratified and deterministic sampling via the inverse
    map (binary digits of p become ternary digits of x), with a byte lookup
    table.
  - `hacks/cantor_fft.py`: `cantor_fft_pmf(m)`, the exact level-m
    discretization: mass `2**-m` at each of the `2**m` ternary lattice points
    with digits {0, 2}, `bucket_size = 2/3**m`, already zero-padded for FFT.
- `Severity` (in `_severity.py`) dispatches construction through a registry:
  `_classify_sev(sev_name, sev_xs)` maps a constructor call to a `sev_kind`
  string, and `__new__` picks the registered subclass (`SeverityScipy`,
  `SeverityDHistogram`, `SeverityCHistogram`, `SeverityFixed`, `SeverityMeta`,
  `SeverityCopy`). Unknown string names fall through to `SeverityScipy`, whose
  `getattr(scipy.stats, name)` raises. This registry is the intended extension
  point; `dhistogram` is the model to follow, as requested.
- The DecL grammar needs NO change. `sev0: ids ...` accepts any identifier, so
  `sev cantor 0.5`, `sev 3 * cantor + 5`, weights, picks, splices, and `!` all
  parse today; the parser puts the single parameter in `sev_a`
  (`sev0_one_param`) and mean/cv in `sev_mean`/`sev_cv` (`sev0_mean_cv`).
- `Aggregate` discretizes severities by sf/cdf differences
  (`_aggregate_compute.discretize_severities`, default
  `discretization_calc='survival'`). No pdf is required on the default path,
  which matters because the Cantor law has none.

## Mathematics

Fix the removed proportion $c \in [0, 1)$ and the kept-piece length
$a = (1-c)/2 \in (0, 1/2]$. The generalized Cantor distribution is the law of

$$X = (1-a) \sum_{i \ge 1} B_i\, a^{i-1}, \qquad B_i \ \text{iid Bernoulli}(1/2),$$

supported on the Cantor set obtained by repeatedly deleting the open middle
proportion $c$ of each interval. Equivalently, the self-similar identity

$$X \overset{d}{=} aX + (1-a)B, \qquad B \sim \text{Bernoulli}(1/2) \perp X.$$

Special cases: $c = 1/3$ is the standard middle-thirds Cantor distribution;
$c = 0$ (so $a = 1/2$) is exactly uniform on $[0,1]$; $c \to 1$ degenerates to
Bernoulli(1/2). For $c > 0$ the law is singular continuous: no atoms, no
density, cdf a devil's staircase.

**Moments.** Writing $m_n = \mathsf{P}X^n$, the self-similar identity gives the
closed recursion

$$m_n = \frac{1}{2(1 - a^n)} \sum_{k=0}^{n-1} \binom{n}{k} a^k\, m_k\, (1-a)^{n-k},
\qquad m_0 = 1,$$

hence $m_1 = 1/2$, $m_2 = 1/(2(1+a))$,
$\operatorname{Var} X = (1-a)/(4(1+a))$ ($= 1/8$ at $a = 1/3$, $= 1/12$ at
$a = 1/2$, uniform, as it must), skewness $0$ by the symmetry
$X \overset{d}{=} 1 - X$. All moments are exact rational-in-$a$ quantities,
computed by an $O(n^2)$ recursion.

**CV parameterization (extra credit).** $\mathrm{cv}^2 = (1-a)/(1+a)$, so
$a = (1-\mathrm{cv}^2)/(1+\mathrm{cv}^2)$ and the attainable range is
$\mathrm{cv} \in (1/\sqrt{3}, 1)$: uniform at one end, Bernoulli at the other.
`sev cantor 10 cv 0.8` therefore has an analytic solution, like `lognorm`.

**CDF.** Iterate on $x \in [0,1]$: if $x < a$, emit binary digit 0 and set
$x \leftarrow x/a$; if $x > 1-a$, emit digit 1 and set
$x \leftarrow (x - (1-a))/a$; if $x$ lands in the gap, emit digit 1 and stop.
$F(x)$ is the binary number so produced. At most 53 iterations resolve a
float64 (each level contributes one bit); vectorized with masks this is a
short numpy loop, comparable in spirit to the hack's uint64 version but valid
for every $a$, not just $a = 1/3$.

**Quantile.** The inverse map reads the binary digits $b_i$ of $p$ and returns
$x = (1-a)\sum b_i a^{i-1}$ (Horner over at most
$\lceil 53 / \log_2(1/a) \rceil \le 53$ digits). Exact, fast, vectorizable;
this is the hack's `_cantor_map` generalized. On a gap (dyadic $p$) the two
binary expansions of $p$ give the two gap endpoints; use the scipy convention
$q(p) = \inf\{x : F(x) \ge p\}$, the left endpoint. `rvs` is then exact
inverse-transform sampling for free.

**Characteristic function (extra credit).** The digit sum gives the classical
infinite product, truncated geometrically fast:

$$\varphi(t) = e^{it/2} \prod_{k \ge 1} \cos\!\big((1-a)\,a^{k-1}\, t / 2\big),$$

the standard $c = 1/3$ case being $e^{it/2}\prod \cos(t/3^k)$. Useful for
`FourierTools` experiments; cheap to include.

## Numerical honesty

Two facts make the float story respectable rather than hand-wavy:

- $F$ is Holder continuous with exponent $\log 2 / \log(1/a)$ ($\approx 0.631$
  at $a = 1/3$). A grid coordinate perturbed by float error $\epsilon$ moves
  the cdf by at most $O(\epsilon^{0.63})$: about $10^{-9}$ for
  $\epsilon \approx 10^{-14}$. Discretization by cdf differences is therefore
  insensitive to a non-representable bucket size.
- The level-m truncation $X_m = (1-a)\sum_{i \le m} B_i a^{i-1}$ is EXACTLY a
  discrete uniform on $2^m$ points with dyadic masses. Whenever $1/a$ is an
  integer $q$ (i.e. $c = 1 - 2/q$), those points are integer multiples of
  $q^{-m}$: the atoms are $(q-1)\sum b_i q^{m-i} / q^m$. So each
  reciprocal-integer shape has a natural bucket family
  $\mathrm{bs} = \mathrm{scale}/q^m$: ternary for the standard $c = 1/3$
  ($q = 3$), BINARY for $c = 1/2$ ($q = 4$), binary again for the uniform
  limit $c = 0$ ($q = 2$). `hacks/cantor_fft.py` builds the $q = 3$ case
  exactly; it is the natural test oracle and the exact-bucket story below.

## Design

### [cantor-module] `src/aggregate/cantor.py` (new)

Follow the `tweedie.py` precedent: a small self-contained module, submodule
access only (`from aggregate.cantor import cantor`), nothing added to the
top-level namespace. Contents:

- **`CantorGen(scipy.stats.rv_continuous)`** with one shape parameter `c`,
  the proportion removed, and frozen convenience instance
  `cantor = CantorGen(a=0, b=1, name='cantor', shapes='c')`. Implemented
  methods:
  - `_argcheck`: $0 \le c < 1$ (c = 0 allowed and yields uniform exactly).
  - `_cdf` / `_sf`: the vectorized digit iteration above ($S = 1 - F$; the
    symmetry $F(x) + F(1-x) = 1$ makes `_sf(x) = _cdf(1-x)` an option).
  - `_ppf` / `_isf`: the vectorized inverse digit map.
  - `_munp(n, c)`: the exact moment recursion (all integer moments).
  - `_stats`: mean 1/2, var $(1-a)/(4(1+a))$, skew 0, excess kurtosis from
    `_munp` (delegating `'k'` to `_munp` is fine).
  - `_pdf`: returns `np.nan` with a docstring stating the law is singular
    continuous and has no density (a plot's density panel goes blank rather
    than crashing; the default aggregate discretization never calls it).
  - `_rvs`: inherited generic inverse transform through our `_ppf`, exact.
  scipy's frozen machinery then provides `loc`/`scale` semantics identical to
  `uniform`: `cantor(1/3, loc=5, scale=3)` is $3X + 5$.
- **`cantor_pmf(log2_points, q=3)`**: port of `hacks/cantor_fft.py`
  generalized to any integer base $q \ge 2$ (shape $c = 1 - 2/q$), returning
  `(xs, ps)` for the exact level-m discretization: atoms
  `(q-1) * j / q**m` for digit words $j$ over $\{0, 1\}$, masses `2**-m`.
  Consumable as `Severity('dhistogram', sev_xs=xs, sev_ps=ps)` or a `dsev`
  program. The recursion is base-agnostic (prepend digit 0, prepend digit
  $q-1$); only the $q = 3$ default existed in the hack. For shapes with
  non-integer $1/a$ the atoms share no lattice and the function does not
  apply; documented as such.
- **`cantor_bs(m, c=1/3, scale=1.0)`**: the natural bucket size
  `scale / q**m` with $q = 2/(1-c)$, raising a clear `ValueError` when $q$ is
  not an integer (no shared lattice exists for that shape). This is the
  memory aid for the bucket-size recipe; see [ternary-buckets].
- **`cantor_chf(t, c=1/3)`**: the truncated product formula, for
  `FourierTools`. (Extra credit, cheap.)

The correctly rounded ternary machinery in `hacks/cantor_cdf.py` is NOT
ported: it is $c = 1/3$ only and the general digit iteration agrees with it to
a few ulp, which is far below the Holder-bounded discretization sensitivity.
It survives as the test oracle (open question 1).

### [severity-wiring] `SeverityCantor` in `_severity.py`

- `_classify_sev`: add `if sev_name == 'cantor': return 'cantor'` before the
  scipy fallthrough.
- `class SeverityCantor(Severity)` with `sev_kind = 'cantor'`, following the
  `SeverityDHistogram` pattern. `_build` logic:
  - shape resolution, in order: `sev_a` given, use it (validated
    $0 \le c < 1$ with a clear message pointing at scaling for `sev cantor 10`
    style mistakes); else `sev_cv > 0`, analytic
    $c = 1 - 2a$, $a = (1-\mathrm{cv}^2)/(1+\mathrm{cv}^2)$, with a range
    check on cv; else default $c = 1/3$.
  - scale/loc: `sev_mean > 0` implies `sev_scale = 2 * sev_mean` (base mean is
    1/2), mirroring the mean-driven scaling of the scipy path; otherwise use
    `sev_scale` / `sev_loc` as given (the `3 * cantor + 5` route).
  - `self.fz = cantor(c, loc=sev_loc, scale=sev_scale)`; write the resolved
    `sev_a` back onto self for introspection and unparsing.
  - set `sev1, sev2, sev3` from `self.fz.moment(1..3)` (exact via `_munp`
    plus scipy's loc/scale binomial), so the no-layer case takes the `moms()`
    histogram fast path. Layered cases fall to `_numerical_moms`, whose
    isf-space quadrature meets our exact `_ppf`; the integrand is monotone and
    bounded so quad converges (verify no `IntegrationWarning` leaks; the
    RuntimeWarning gate stays clean).
- `SeverityCantor.natural_bs(m)`: instance convenience delegating to
  `cantor_bs(m, c=self.sev_a, scale=self.sev_scale)`, so the recipe is one
  autocomplete away on the object you are holding. Deliberately two names
  for one implementation: `cantor_bs` is the free function (named for
  discovery in the module), `natural_bs` the bound method (naming it
  `cantor_bs` on a Cantor object would be redundant). No `Severity.info`
  changes: info strings stay uniform across severity kinds (author decision,
  2026-09-25).
- Naming vetted per house rule: `rg -i cantor src/aggregate` is empty today;
  `cantor` collides with nothing in scipy.stats, DecL keywords, or the class
  surface. New public names: DecL/`sev_name` string `cantor`, class
  `SeverityCantor` with method `natural_bs`, module `aggregate.cantor` with
  `CantorGen`, `cantor`, `cantor_pmf`, `cantor_bs`, `cantor_chf`
  (`natural_bs` and `bs` do not collide with anything on `Severity` or its
  parents; `rg` confirms at implementation time).

### [decl-surface] grammar, highlighting, corpus

- Grammar: no change. No new token, so `parser_errors.py` and
  `test_grammar_sync` are untouched.
- `decl_pygments.py`: add `'cantor'` to the one-shape-parameter word list so
  it highlights like `lognorm`.
- `aggregate/agg/test_suite.agg`: add lines to the severity category, e.g.
  bare `sev cantor`, `sev cantor 0.5`, `sev 3 * cantor 0.5 + 5`, and a
  mean/cv form. Regenerate `tests/data/expected_specs.json` with
  `uv run python tests/capture_spec_snapshot.py` and review the diff: ONLY
  the new lines may appear.
- `decl_writer.py`: expected to handle `cantor` generically (named severity
  with `sev_a`); confirmed by the unparser round-trip suite picking up the
  new corpus lines. Fix only if round-trip fails.

### [ternary-buckets] bucket-size guidance, the binary-fraction rule break

The standing guidance is that `bs` should be a binary fraction. For a Cantor
severity with reciprocal-integer kept fraction ($q = 2/(1-c)$ an integer) the
natural grids are base-$q$: with `bs = scale/q**m` the level-m cylinder sets
align with buckets, and cdf-difference discretization reproduces the exact
level-m masses (up to float noise bounded by the Holder estimate). For the
standard $c = 1/3$ that is ternary `bs = scale/3**m`, the rule break; for
$c = 1/2$ it is `bs = scale/4**m`, binary after all.

The author will not remember this recipe, so it must live where it is
tripped over, not where it must be recalled (decided 2026-09-25):

- **`cantor_bs` / `natural_bs`** (defined in [cantor-module] and
  [severity-wiring] above): the executable form of the recipe, one
  autocomplete away.
- **Two library.agg entries** (`src/aggregate/agg/library.agg`), the primary
  home: a standard middle-thirds example and a $c = 1/2$ example, each
  carrying `hints{bs=...; log2=...}` with the natural bucket size and a
  `note{...}` explaining the natural lattice in a sentence and naming
  `aggregate.cantor.cantor_bs` as the way to recompute it. The hints
  machinery is confirmed suitable: `bs` and `log2` are on the underwriter's
  `_HINT_KEYS` allow-list, `_resolve_hints` merges caller-wins (an explicit
  `update(bs=...)` still overrides), and `_coerce_hint_value` parses `a/b`
  fractions, so the entries write `hints{bs=1/59049; ...}` (m = 10 ternary;
  `1/59049` coerces to the identical float as `1/3**10`) and
  `hints{bs=1/4096; ...}` (m = 6, $q = 4$), matching the existing
  `hints{bs=1/4; log2=13}` style in library.agg. Entry names and note prose
  follow the surrounding library style.
- **Docstrings and the docs example** (see [docs-lockstep]): default binary
  `bs` is CORRECT (atomless law, continuous cdf) merely blurry at the finest
  scales; the natural `bs` is exact-at-level-m and the pretty choice for
  pictures of self-similarity.
- NOT in `recommend_bucket`: the sizer stays severity-agnostic. The general
  idea (a severity-declared natural-lattice family the sizer snaps to, which
  would serve `dhistogram` gcd lattices too) is parked in `dev/TODO.md` as
  `[Natural-Lattice-Snap]`, not built here.
- NOT in `Severity.info`: info strings stay consistent across kinds.

Also verify nothing in the update path *enforces* or *rounds to* binary `bs`
when the user passes `bs` explicitly (recommend_bucket only suggests). If an
assert or silent rounding exists, surface it, do not work around it.

### [docs-lockstep] docs touches, pending rebuild

Grep the `.rst` tree for the severity-name lists (wherever `dhistogram` /
`chistogram` / the scipy zoo are enumerated) and add `cantor` in lockstep. Do
not build the docs; note in the CHANGELOG entry that docs are pending a
rebuild. A worked example page (self-similar aggregate pictures via
`cantor_pmf` and ternary `bs`) is out of scope here; candidate for
`pedagogy.py` later, noted in `dev/TODO.md`.

## Files touched

| File | Change |
|---|---|
| `src/aggregate/cantor.py` | new: `CantorGen`, `cantor`, `cantor_pmf`, `cantor_bs`, `cantor_chf` |
| `src/aggregate/_severity.py` | `SeverityCantor` (+ `natural_bs`), `_classify_sev` branch |
| `src/aggregate/decl_pygments.py` | add `cantor` to one-shape list |
| `src/aggregate/agg/test_suite.agg` | new severity corpus lines |
| `src/aggregate/agg/library.agg` | two Cantor entries with `hints{}` + `note{}` |
| `dev/TODO.md` | park `[Natural-Lattice-Snap]` idea |
| `tests/data/expected_specs.json` | regenerated snapshot |
| `tests/test_cantor.py` | new unit + end-to-end tests |
| `docs/...` | severity-name mentions, lockstep only |
| `pyproject.toml`, `CHANGELOG.md`, `dev/TODO.md` | bump, entry, note |

## Stages and commits

Two version bumps, each its own commit, per house rules:

1. **[cantor-module]** `a346`: the module plus `tests/test_cantor.py` unit
   tests. Self-contained, no behavior change elsewhere.
2. **[severity-wiring]** `a347`: `SeverityCantor` with `natural_bs`,
   classification, pygments, corpus lines, snapshot regen, the two
   library.agg entries, docs lockstep, end-to-end tests, `dev/TODO.md` note,
   CHANGELOG, move this plan to `dev/done/` when the author says done.

## Acceptance checks

Unit (module level, `tests/test_cantor.py`):

- cdf oracle: at $c = 1/3$, agree with `hacks/cantor_cdf.cantor_cdf_scalar`
  (copied into the test file as a fixture oracle) to within a few ulp on a
  mixed grid of uniform draws, ternary rationals, gap points, endpoints,
  subnormals.
- cdf at $c = 0$ is the identity (uniform), exactly.
- symmetry $F(x) + F(1-x) = 1$; monotonicity; $F(a) = 1/2 = F(1-a)$ flats.
- quantile: $F(q(p)) = p$ for non-dyadic $p$; $q$ agrees with
  `hacks/cantor.cantor_midpoints` values at $c = 1/3$ up to the gap-endpoint
  convention; `rvs` sample mean/var within Monte Carlo tolerance.
- moments: `_munp` vs the closed forms $1/2$, $1/(2(1+a))$, var $1/8$ at
  $a = 1/3$, $1/12$ at $a = 1/2$; vs exact moments of the level-m atoms from
  `cantor_pmf` (geometric convergence in m); skew 0.
- `cantor(1/3, loc=5, scale=3).mean() == 6.5` and friends: loc/scale flows.

End-to-end (DecL level):

- `build('agg C 1 claim sev cantor fixed', ...)` with binary `bs`: aggregate
  mean $1/2$, var $1/8$ to discretization tolerance; `explain_validation`
  clean.
- same with ternary `bs = 1/3**m`: discretized severity equals `cantor_pmf`
  masses to float noise (the exact-bucket claim).
- `sev 3 * cantor 0.5 + 5`: mean $3 \cdot 1/2 + 5$, var $9(1-a)/(4(1+a))$.
- a Poisson-frequency compound runs and validates; a layered
  (`occurrence net of ...`) variant produces finite, warning-free moments.
- unparser round-trips the new corpus lines; snapshot diff contains only the
  added lines.
- the two library.agg entries build from the knowledge base with no explicit
  `bs` (the hints supply it); the standard entry's discretized severity
  equals `cantor_pmf(m)` masses, the $c = 1/2$ entry's equals
  `cantor_pmf(m, q=4)` masses; `cantor_bs` / `natural_bs` reproduce the
  hinted values exactly.

Gate: tier 2 (`uv run pytest`) at each bump; tier 3 plus the RuntimeWarning
numerics gate at the close, since this touches quadrature.

## Decided questions

Settled with the author 2026-09-25; each resolution is the one marked.

1. **Correctly rounded cdf.** Decided: ship only the general-shape digit
   iteration; keep `hacks/cantor_cdf.py` as the test oracle (scalar version
   copied into the test file). The alternative, porting the exact ternary
   version and special-casing $c = 1/3$, doubles the code for a few ulp that
   the Holder bound says cannot matter downstream.
2. **`cantor_pmf` exact lattice utility.** Decided: ship it in the module
   ($c = 1/3$ only, documented), because it is the test oracle for the
   exact-bucket acceptance check anyway and the natural tool for
   self-similarity pictures. No DecL hook for it.
3. **Export surface.** Decided: submodule only, like `Tweedie`
   (`from aggregate.cantor import cantor`); the DecL name `sev cantor ...`
   works regardless and is the real user surface.
4. **Remembering the bucket-size recipe.** Decided: `cantor_bs` /
   `natural_bs` helpers plus two library.agg entries carrying
   `hints{bs=...; log2=...}` and an explanatory `note{}` that names
   `cantor_bs`. Explicitly rejected: special-casing `recommend_bucket`
   (severity-agnostic sizer stays clean; general snap idea parked in
   `dev/TODO.md` as `[Natural-Lattice-Snap]`) and a `Severity.info` line
   (info strings stay consistent across severity kinds).

5. **Shape parameterization.** Decided: single shape `c` = proportion
   removed, default $1/3$, $c = 0$ giving uniform exactly, plus the analytic
   cv route (range $(1/\sqrt{3}, 1)$). Alternative parameterizations (kept
   fraction $a$, similarity dimension) are one-liners away and not worth a
   second knob.
