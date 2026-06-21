# plan-bv-discrete — discrete bivariate severity (`dbvsev`) + discrete frequency for `bv`

New DecL feature; bumps the `a*` version. Goal: let a `bivariate` (`bv`) object
be declared **directly from discrete data** — a shared discrete frequency
(`dfreq`) and/or a discrete bivariate severity (`dbvsev`) — exactly the way
`agg NAME dfreq [...] [...] dsev [...] [...]` declares a discrete univariate
aggregate. One new keyword (`DBVSEV`), a small grammar addition, and a third
construction `mode='discrete'` in `BivariateAggregate`.

> **STATUS: DONE (1.0.0a86).** Shipped: `dbvsev` keyword (dense / dense-uniform
> / sparse), the four `bv` forms, `mode='discrete'` on `BivariateAggregate`,
> nesting-aware preprocessor, unparser support, and `tests/test_bv_discrete.py`.
> Note: the validation surface uses the as-built `summary_df` / `stats_df` /
> `dependency_df` (there is no `validation_df`); component names are `X` / `Y`.
> Design settled with the author 2026-06-20:
> - **Dense contingency-table form** `dbvsev [xs] [ys] [matrix]` (Option A),
>   whitespace-native (no commas — that was a typo in the original ask), missing
>   probability matrix ⇒ uniform over the lattice. Clincher: `dbvsev` is for
>   **small pedagogical / exact test cases**; anything larger is built via the
>   constructor or piped in as numpy arrays through an f-string.
> - **Sparse-triples form ALSO supported** `dbvsev [[x y p] [x y p] …]`; the
>   grammar **auto-detects** dense vs sparse from the first argument (a flat
>   `[numbers]` list ⇒ dense; a bracket-of-brackets ⇒ sparse). Both lower onto the
>   same internal joint matrix `S`.
> - **`pnl` axes are NOT supported in discrete mode** — a per-axis premium shift
>   raises a clear error (decided; see Implementation).

## Context

`BivariateAggregate` (in `bivariate.py`) today builds the joint law one of two
ways (`mode`):

- **`mode='copula'`** (the `bivariate` DecL keyword): the body is **exactly two**
  `agg`/`pnl` components. Each becomes a 1-claim zero-inflated `Aggregate` whose
  aggregate density is the per-event severity `g_i`; the two are coupled by a
  `Copula` into the joint per-claim severity matrix
  `S = copula.rectangle_pmf(G0, G1)` (`bivariate.py:1089`), then run through the
  shared 2-D compound FFT in `update_work` (`bivariate.py:1057`).
- **`mode='netceded'`**: one reinsured aggregate split into a view-pair joint.

The **only** thing that distinguishes the joint construction is how `S` is
formed. A discrete bivariate severity supplies `S` **directly** as a contingency
table on an explicit lattice, so it slots into the same 2-D FFT with almost no
new machinery — `update_work` uses the given `S` instead of calling
`rectangle_pmf`.

The shared outer frequency is independent of the severity path: it is already
either an `exposures … freq` head (continuous count + distribution) or, as in
`agg`, a `dfreq` clause (discrete empirical frequency). `dfreq` is already a
grammar nonterminal (`decl.lark:222`, transformer `parser.py:979`) returning
`{freq_name:'empirical', freq_a, freq_b, exp_en:-1}`; it is reused unchanged.

DecL lists are whitespace-separated (`numberl: numberl expr`); there is **no
comma token** and the DSL is deliberately comma-free. `dbvsev` keeps that.

## The four forms (one exists, three new)

Frequency source × severity source is a 2×2; form 1 already ships:

| # | Frequency | Severity | DecL | Status |
|---|---|---|---|---|
| 1 | `exposures … freq` | two `agg`/`pnl` + `copula` | `bv N 5 claims agg A … agg B … copula gumbel 0.3` | **exists** |
| 2 | `dfreq` | two `agg`/`pnl` + `copula` | `bv N dfreq [1 2] [.7 .3] agg A … agg B … copula …` | **new** ("dfreq with agg/agg") |
| 3 | `exposures … freq` | `dbvsev` | `bv N 5 claims dbvsev [xs] [ys] [matrix] mixed gamma 0.1` | **new** ("EXPOSURE with dbvsev") |
| 4 | `dfreq` | `dbvsev` | `bv N dfreq [1 2] [.7 .3] dbvsev [xs] [ys] [matrix]` | **new** (the headline) |

`dfreq` replaces the whole `exposures … freq` head (it carries both the count and
the distribution), exactly as in `agg_out_dfreq` (`decl.lark:68`).

Form 3 takes the **full frequency vocabulary** after `dbvsev`, just as form 1
does after the copula clause: a bare count defaults to Poisson, or you name any
distribution —
```
bv N 5 claims dbvsev [xs] [ys] [matrix]                 # Poisson(5) shared freq
bv N 5 claims dbvsev [xs] [ys] [matrix] poisson         # explicit
bv N 5 claims dbvsev [xs] [ys] [matrix] mixed gamma 0.1 # mixed-Poisson, cv 0.1
```
The trailing `freq` parses unambiguously: `dbvsev` ends with the matrix
(`[[…]]`) or, in the uniform/sparse forms, a bracketed list, and `freq` begins
with a keyword (`poisson` / `mixed` / `negbin` / …) or is absent.

**`clash` is not a separate FFT mode — it lowers to `mode='copula'`.** The
`clash` statement (`decl.lark:126`, solver `solve_clash_model`) derives the
shared event count and two per-event Bernoulli triggers from the (na, nb, nc)
claim counts, then **builds two `dfreq [0 1] [1-p p]` components coupled by the
independent copula** — i.e. it constructs an ordinary copula-mode `bvagg`. There
is therefore **no "discrete clash" to add**: clash already produces discrete
(Bernoulli) component severities via the copula path. `dbvsev` is the orthogonal
case — the joint per-claim matrix given *directly*, not derived from triggers.

## DecL — `dbvsev` syntax & semantics

Two equivalent surface forms; the grammar auto-detects which (see below). Both
build the same internal joint per-claim matrix `S`.

### Dense (contingency table) — `dbvsev [xs] [ys] [matrix]`

**Row index = X, column index = Y.** This is the crucial reading:

```
dbvsev [10 20]        # x outcomes:  x[0]=10, x[1]=20            (n_x = 2 ROWS)
       [1 2 3]        # y outcomes:  y[0]=1,  y[1]=2,  y[2]=3    (n_y = 3 COLS)
       [[.4 .1 .0]    # ROW 0  ->  X = x[0] = 10:  P(10,1)=.4  P(10,2)=.1  P(10,3)=0
        [.2 .1 .2]]   # ROW 1  ->  X = x[1] = 20:  P(20,1)=.2  P(20,2)=.1  P(20,3)=.2
#  matrix[i][j] = P(X = xs[i], Y = ys[j]);  shape = (n_x, n_y) = (2, 3);  sum = 1
```

So **row `i` fixes `X = xs[i]` and ranges over all `Y` values across its
columns** — yes, "row 0 = x=x[0] paired with the y-values" is exactly right. The
matrix has one row per `x` outcome and one column per `y` outcome.

- `matrix` is the `n_x × n_y` **joint** per-claim probability table. This is the
  literal 2-D analogue of `dsev [outcomes] [probs]` (outcomes → a pair of axis
  lists; pmf → a matrix) and the canonical contingency-table representation.
- **`dbvsev [xs] [ys]`** (matrix omitted) ⇒ **uniform** `1 / (n_x · n_y)` over the
  full lattice — same default as `dsev`/`dfreq` with empty `dprobs`
  (`parser.py:971,981`).
- `xs` / `ys` accept the existing `doutcomes` forms incl. ranges (`[0:2]`,
  `[0:10:5]`), so `dbvsev [0:2] [0:2]` is legal.

### Sparse (triples) — `dbvsev [[x y p] [x y p] …]`

A list of `(x, y, prob)` atoms — the long/tidy encoding of the same law:

```
dbvsev [[10 1 .4] [10 2 .1] [20 1 .2] [20 2 .1] [20 3 .2]]
#        x  y  p   ...                                       sum p = 1
```

Lowered to the dense `S` internally: `xs = sorted(unique x)`, `ys = sorted(unique
y)`, `S[index(x), index(y)] += p` (collisions summed). Atoms must land on the
implied cross-product lattice (they do by construction). Compact for a handful of
co-occurrence scenarios; the dense form is clearer for a small full table.

### Both

- **Marginals fall out as row/column sums** of `S` — the bivariate
  "marginalise → standalone aggregate" invariant is exact by construction.
- Probabilities are renormalised to sum 1 (warn if off by > `validation` noise),
  mirroring `dsev`.

### Grammar additions (`decl.lark`)

New keyword token (priority form, like `DSEV`/`DFREQ`):
```
DBVSEV.2: /dbvsev(?![a-zA-Z0-9._:~\-])/
```
Add `dbvsev` to the `ID` negative-lookahead reserved-word list (`decl.lark:436`,
beside `dsev`/`dfreq`).

Severity rule (reuses `doutcomes`); dense + sparse + uniform:
```
dbvsev: DBVSEV doutcomes doutcomes dprob_matrix   -> dbvsev_dense
      | DBVSEV doutcomes doutcomes                -> dbvsev_dense_uniform
      | DBVSEV dtriple_list                       -> dbvsev_sparse

dprob_matrix: "[" dmatrix_rows "]"
dmatrix_rows: dmatrix_rows drow   -> dmatrix_rows_cons
            | drow                -> dmatrix_rows_one
drow: "[" numberl "]"             -> drow

dtriple_list: "[" dtriples "]"
dtriples: dtriples dtriple        -> dtriples_cons
        | dtriple                 -> dtriples_one
dtriple: "[" expr expr expr "]"   -> dtriple
```

**Auto-detection is unambiguous and falls out of the grammar** — no lookahead
flag needed. `doutcomes` is `"[" numberl "]"`, and `numberl` is *numbers*, not
brackets. So:
- dense (`DBVSEV [a b c] [d e f] …`) — the first two groups are flat numeric
  lists, parseable as `doutcomes`;
- sparse (`DBVSEV [[x y p] …]`) — the first (and only) group's content starts
  with `[`, which **cannot** parse as `doutcomes` (no numbers inside), so only
  `dbvsev_sparse` matches.

For any concrete input exactly one production matches (Earley + dynamic lexer).
`dprob_matrix` (rows of any length) and `dtriple_list` (rows of exactly 3) never
compete: they sit in different productions reachable only after the dense
`doutcomes doutcomes` prefix or directly after `DBVSEV` respectively.

New `bv_out` productions (`decl.lark`, beside the existing copula forms):
```
bv_out: …                                                                  // existing
      | BIVARIATE name dfreq bv_body copula_clause trailer   -> bv_out_copula_dfreq
      | BIVARIATE name exposures dbvsev freq trailer         -> bv_out_discrete
      | BIVARIATE name exposures dbvsev trailer              -> bv_out_discrete_nofreq
      | BIVARIATE name dfreq dbvsev trailer                  -> bv_out_discrete_dfreq
```

### Transformer (`parser.py`)

All three `dbvsev` productions normalise to the **same** partial spec
`{"dbv_xs": xs, "dbv_ys": ys, "dbv_S": matrix}` (parallel to `dsev_main`
returning `sev_*`), so `BivariateAggregate` sees one shape regardless of surface
form:
- `drow(c)` → `np.asarray(numberl)`; `dmatrix_rows_*` accumulate rows;
  `dprob_matrix(c)` → 2-D `np.array`.
- `dbvsev_dense(c)` → builds `S` from the matrix; validates
  `matrix.shape == (len(xs), len(ys))`, entries ≥ 0, renormalises to sum 1 with a
  warning if off by > `validation` noise (mirror `dsev`).
- `dbvsev_dense_uniform(c)` → `np.full((n_x, n_y), 1/(n_x*n_y))`.
- `dtriple(c)` → `(x, y, p)`; `dtriples_*` accumulate; `dbvsev_sparse(c)` →
  `xs = sorted(unique x)`, `ys = sorted(unique y)`, scatter
  `S[i, j] += p` (sum collisions), then the same validate/renormalise.
- `bv_out_discrete*` build `("bvagg", name, spec)` with `mode="discrete"`,
  `**exposures`/`**dfreq`/`**freq` for the shared frequency, and the `dbv_*`
  keys; `bv_out_copula_dfreq` is the existing copula builder with the frequency
  from `dfreq` instead of `exposures … freq`.

## Implementation — `BivariateAggregate` (`bivariate.py`)

Add **`mode='discrete'`**. The elegant minimal change: discrete mode reuses
*all* of copula mode except the formation of `S`.

- **`__init__`**: accept `dbv_xs`, `dbv_ys`, `dbv_S` kwargs. In discrete mode,
  build the two **marginal units** as 1-claim discrete aggregates from the row /
  column sums of `S`:
  `g0 = dbv_S.sum(axis=1)` on `dbv_xs`, `g1 = dbv_S.sum(axis=0)` on `dbv_ys`,
  each as `Aggregate(... dfreq [1] [1] dsev [xs_i] [g_i])` (a 1-claim `dsev`
  aggregate whose aggregate density **is** the per-claim severity `g_i`). These
  units give (a) the per-axis sizing inputs and (b) the standalone validation
  targets — exactly the role the copula-mode units play. Store `self._S_given =
  dbv_S` and the lattice. Build `self.frequency` / `self.en` from the shared
  `freq_*` / `exp_*` kwargs as today.
- **`update_work`** (`bivariate.py:1057`): when `self._S_given is not None`, use
  it as `S` directly instead of `S = self.copula.rectangle_pmf(G0, G1)`; the rest
  (`_lay_signed_2d`, `rfft2`, `freq_pgf`, `irfft2`, roll, deficit, marginal
  theory cache) is unchanged. `self.copula` is `None` in discrete mode (info /
  repr already handle `copula is None`, e.g. `bivariate.py:1287`, `:1336`).
- **Sizing**: the per-axis grid (`bs`, window, `log2` split) is driven by the two
  marginal units through the existing measure-don't-guess machinery
  (`dev/plan-mv.md` §5). Because the units are `dsev` aggregates, their grids are
  integer-lattice exact (the `dsev` bucket scheme), so `S` lands on the lattice
  with no rebucketing — the discrete analogue of the 1-D `dsev` exactness.
- **Validation**: `validation_df` (the per-axis marginal-reproduction frame; note
  its rename from `explain` is tracked in `dev/plan-name-rationalization.md`)
  should show **exact** marginal means/CVs in discrete mode, since the unit
  marginals are *defined* as the row/column sums of `S`.
- **`info` / repr**: a discrete `bv` reports `kind = discrete` (vs a copula name);
  the `copula_tau` row is `n/a` (already the `copula is None` branch).
- **`pnl` axes NOT supported (decided).** Discrete mode is loss/loss only. If a
  spec carries a per-axis `pnl` affine (`agg_reflect` / `agg_shift` / premium)
  alongside `dbvsev`, raise a clear `ValueError` ("pnl / premium axes are not
  supported with dbvsev; use the copula form `bv … agg/pnl … agg/pnl …`"). The
  `dbvsev` grammar has no premium slot, so this only guards a hand-built spec.

No change to `mode='netceded'`. `clash` is unaffected (it lowers to
`mode='copula'`, see The four forms).

## test_suite.agg + tests

- Add a category-N-style block to `agg/test_suite.agg` exercising every new form:
  form 4 headline (`bv … dfreq … dbvsev …`), form 3 (`bv … N claims dbvsev … freq`),
  form 2 (`bv … dfreq … agg/agg copula …`), the **sparse** form
  (`dbvsev [[x y p] …]`), the uniform-default `dbvsev [xs] [ys]` (no matrix), and a
  `dbvsev [0:2] [0:2]` range case. Each line is auto-exercised by the parse +
  snapshot tests.
- `tests/test_bv_discrete.py` (new):
  - an **independence** check — a rank-1 `S = outer(p_x, p_y)` reproduces the
    `copula independent` joint of the same marginals (densities match to FFT
    tolerance);
  - **marginal reproduction exact** — `validation_df['mean_error']` /
    `['cv_error']` ≈ 0 for a discrete `bv` (tighter than the copula path);
  - a **dependence** case — a non-separable `S` (e.g. near-comonotone on a 3×3)
    gives the hand-computable joint moments / correlation;
  - **dense == sparse** — the same law written both ways builds an identical `S`
    / density;
  - **uniform default** — `dbvsev [xs] [ys]` equals the explicit uniform matrix;
  - **shape / normalisation guards** — wrong matrix shape errors; an un-normalised
    matrix (and sparse probs) warn and renormalise;
  - **pnl rejected** — a hand-built discrete spec carrying a premium shift raises
    the clear `ValueError`.

## Files

- `src/aggregate/decl.lark` — `DBVSEV` token; reserved-word list; `dbvsev`
  (dense / dense-uniform / sparse) + `dprob_matrix` / `dmatrix_rows` / `drow` +
  `dtriple_list` / `dtriples` / `dtriple` rules; four new `bv_out` productions.
- `src/aggregate/parser.py` — `drow` / `dmatrix_rows_*` / `dprob_matrix` /
  `dtriple` / `dtriples_*` / `dbvsev_dense` / `dbvsev_dense_uniform` /
  `dbvsev_sparse`; `bv_out_copula_dfreq`, `bv_out_discrete`,
  `bv_out_discrete_nofreq`, `bv_out_discrete_dfreq`.
- `src/aggregate/bivariate.py` — `mode='discrete'` in `__init__`
  (marginal-unit build + store `S`), the `update_work` `S` override, info/repr
  `kind=discrete`.
- `src/aggregate/agg/test_suite.agg` — example lines for the new forms (dense,
  sparse, uniform, range, forms 2–4).
- `tests/test_bv_discrete.py` (new) — the cases above.
- `pyproject.toml` / `CHANGELOG.md` — version bump + feature note.
- Docs: a short `dbvsev` subsection in the bivariate user guide (note pending a
  manual rebuild per CLAUDE.md; do not build docs in the loop).

## Verification

- `build('bv Dice dfreq [1 2 3] [.5 .3 .2] dbvsev [0 1 2] [0 5 10] [[…]]')`
  returns a `BivariateAggregate` with `mode='discrete'`; `qd` shows the joint
  `summary_df`; marginals reproduce the standalone `dsev` aggregates exactly.
- Independence `S = outer(p_x, p_y)` matches the `copula independent` build of the
  same marginals.
- Full suite green (`UV_LINK_MODE=copy uv run pytest`), including the parse +
  snapshot tests over the new `test_suite.agg` lines; the `test_style.py` trio
  remains the only (pre-existing) failure.
- `rg dbvsev` finds the keyword wired through grammar, transformer, class, suite,
  tests, and docs; no stray references.

## Close-out

- Bump `pyproject.toml` `1.0.0a*`; `CHANGELOG.md` feature section (new `dbvsev`
  keyword, the three new `bv` forms, uniform-matrix default).
- `dev/TODO.md`: mark the item landed; move this plan to `dev/done/`.

## Considered & rejected / deferred

- **Comma-separated literals** (`[[a b],[c d]], …`) — rejected: DecL is
  deliberately comma-free; the commas in the original ask were a typo.
- **A standalone `dbvsev NAME …` severity object** (à la `sev NAME dsev …`,
  `sev_out`) — out of scope; `dbvsev` only appears inside a `bv` declaration for
  now. Add later if a reusable joint-severity object is wanted.
- **`pnl` axes in discrete mode** — deferred; raises a clear error (see
  Implementation). The headline use case is loss/loss; premium axes stay on the
  copula form.

*(The sparse-triples form is now **in scope**, not deferred — see DecL syntax.)*

## Open questions

- **`agg/agg` body with a `dbvsev`-style direct matrix override?** Not proposed —
  the copula path already covers continuous/agg components. Flagged only to
  confirm we are not also wiring a matrix into copula mode.
