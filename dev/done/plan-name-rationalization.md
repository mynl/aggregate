# plan-name-rationalization — finish the accessor naming convention

New-ish refactor; bumps the `a*` version. Standalone (no dependency on config or
bv plans). Goal: make the public accessor surface on the four main classes
(`Aggregate`, `Portfolio`, `Distortion`, `BivariateAggregate`) obey the two
established rules **uniformly**, finishing what the a82/a84 sweeps started:

1. **Every DataFrame-returning member is `<noun>_df`** and is a property (or a
   parameterized method only when it genuinely takes arguments).
2. **Every narrative/text accessor is a noun-named property** returning a string
   (`tail_description` / `tail_explanation` / `bs_description` /
   `validation_explanation` are the template) — never a verb, never a method when
   it takes no arguments.

See the two memories: [[aggregate-df-naming-convention]] and
[[no-deprecated-aliases-pre-1.0]]. **No deprecated aliases** — clean breaks, this
is pre-1.0. Sweep `src` + `tests` + `docs` for each rename.

> **STATUS: DONE (1.0.0a85).** Landed A1, A3, B4, C5, D, E1, and the section-F
> bivariate reporting redesign; E2/E3 and E4 deferred as noted. Full suite green
> (style-test trio is the only pre-existing failure). Inventory captured 2026-06-20 from a full pass over the
> four main classes. Author rulings 2026-06-20: **C5** rename `trim_df()` →
> `trim_density_df()` *and* the `update(trim_df=…)` kwarg → `trim_density_df=…`;
> **D** convert `corr` / `marginals` to plain noun properties (no `_df` — neither
> returns a DataFrame); **E4** (`.shape`) deferred; **E2/E3** deferred (possible
> out-of-repo readers, gate on confirmation). **A2 superseded — bivariate
> reporting redesign (section F):** `explain` is *deleted* (not renamed),
> `summary_df` is restructured to the exact `Portfolio.summary_df` shape, and a
> new `dependency_df` owns the joint-structure scalars. In-scope batch: A1, A3,
> B4, C5, D, E1, **F**.

## Context

The `describe` → `summary_df` rename (a84) fixed the headline `_df` violation but
left siblings inconsistent: `reins_describe` is still a `describe`-verb property
returning a df, and `BivariateAggregate.explain` is a verb-named property
returning a df. Separately the narrative surface is *almost* all
noun-properties, but `reins_kinds` is a no-arg **method** returning reins text.
A `tvar_info_df` is a no-arg **method** where its siblings are cached properties.

`Aggregate.reins_description` is **already a property** (the a82 narrative sweep
landed it) — it is *not* on the fix list; the method-shaped reins narrative that
remains is `reins_kinds` (B4).

## A. DataFrame members breaking `<noun>_df` / property

| # | Member | Class(es) | Problem | Fix | Call sites |
|---|---|---|---|---|---|
| A1 | `reins_describe` | Aggregate, Portfolio | property→df, but `describe` verb (leftover from a84) | **`reins_summary_df`** | ~56 (src/tests/docs) |
| A2 | `explain` | BivariateAggregate | ~~property→df, verb-named, no `_df`~~ | **SUPERSEDED → section F** (deleted, not renamed) | — |
| A3 | `tvar_info_df` | Distortion (base `spectral.py:903`; WtdTVaR `:2743`) | **method** returning a df, no args — siblings `info` / `summary_df` / `stats_df` / `density_df` are all `cached_property` | make it a **property** (name already correct); convert `self.tvar_info_df()` call at `spectral.py:2820` | 1 src |

Notes:
- A1: the partner `reins_density_df` / `reins_stats_df` are already correct, so
  `reins_summary_df` slots in cleanly beside them. The private worker
  `_reins_describe` / `_reins_describe_block` cache attrs keep their names
  (internal). Watch the `constants.py` comment that lists `reins_describe`.
- A2: superseded by section F — the bivariate reporting surface is redesigned
  rather than `explain` being renamed in place.
- A3: the base `Distortion` quartet is `cached_property`; make `tvar_info_df`
  `cached_property` too for symmetry (it is pure given the distortion). The
  WtdTVaR override at `:2743` must match (property), and its internal call updates.

## B. Narrative/text method that should be a property

| # | Member | Class | Problem | Fix | Call sites |
|---|---|---|---|---|---|
| B4 | `reins_kinds` | Aggregate | **method** returning the reins-kinds text ("None" / "Occurrence only" / …), no args → should be a noun property like every other narrative accessor | make it a **property** | 2 src (`distributions.py:4579,5680` call `.reins_kinds()`) + 1 doc (`2_x_re_pricing.rst:201,235`, `:meth:` role + `print(a.reins_kinds())`) |

Note: `reins_kinds` is already a noun, so only the `@property` decorator + the
two internal `()` call-site drops + the doc (`:meth:`→`:attr:`, drop `()`) change.
Fix the docstring typos while there ("desciption", "Aggergate").

## C. Misleading `_df` on a mutator (DECIDED)

| # | Member | Class | Problem | Fix |
|---|---|---|---|---|
| C5 | `trim_df()` | Portfolio | a **mutator**: drops columns from `self.density_df` in place, returns `None`; `_df` falsely signals "returns a df" | **`trim_density_df()`** (method) |

The method is `update`-internal only (`portfolio.py:2637`, behind the
`update(trim_df=…)` kwarg) — no external callers — so the rename is contained.
**Open:** does the `update(trim_df=…)` **kwarg** (`portfolio.py:2403,2428,2636`)
rename to `trim_density_df=…` in lockstep? Recommend **yes** (one consistent
name); confirm.

## D. No-arg methods → properties (DECIDED, with a naming snag)

Convert both to **properties** (cheap pure no-arg accessors):
- `BivariateAggregate.corr()` — **returns a scalar** Pearson correlation
  (`bivariate.py:1175` → `self.bivariate.corr()`).
- `BivariateAggregate.marginals()` — **returns a 2-tuple** of marginal density
  arrays (`bivariate.py:1159`).

`moments(max_order=3)` correctly stays a method (takes an argument).
`Portfolio.augmented_dfs` — **no change** (a property returning a *dict* of
frames; the plural `_dfs` is correct for a collection).

**No `_df` (DECIDED 2026-06-20).** Neither returns a DataFrame, so neither takes
the `_df` suffix — they become plain noun properties keeping today's returns:
- **`corr`** — property, scalar float (Pearson correlation).
- **`marginals`** — property, 2-tuple of marginal density arrays.

(`_df` is reserved for DataFrame-returning members; applying it here would
violate the very rule this plan enforces.)

## E. Back-compat shims (separate cleanup category — not convention fixes)

Legacy compatibility cruft found in the same pass. Removal is a clean-break win
but several have *external* (out-of-repo, e.g. PMIR) readers, so each needs a
"nothing outside reads it" confirmation.

| # | Shim | Location | In-repo callers | Risk |
|---|---|---|---|---|
| E1 | `_BOUNDED_FREQS` / `_BOUNDED_SCIPY_SEVS` re-export | `distributions.py:56` (`# noqa: F401`) | `tests/test_tail.py` imports them via `distributions` | **Low** — repoint test (+ any external) to `aggregate.tail`, delete the re-export |
| E2 | `_kind_immutable` property → `self._name` | `spectral.py:854` | none (only its def) | external dispatch site (comment) — confirm no outside reader, then delete |
| E3 | `_RegistryProperty` / `_available_distortions_` / `_kinds_ordered` | `spectral.py:583,859–867` | none in-repo | same external caveat as E2 |
| E4 | `self.shape = self._distortions` legacy reads | `spectral.py:2856,3057` | — | ⚠️ **DEFERRED (author 2026-06-20).** `.shape` is the core distortion-parameter attribute, read/written pervasively across `spectral.py` (stores rho/alpha/…); only these two Minimum/Mixture assignments are flagged legacy. A real fix is a `.shape`→`.param` rename across the whole distortion hierarchy — its own project, not this plan. |

## F. Bivariate reporting redesign (supersedes A2; author-directed 2026-06-20)

The `BivariateAggregate` reporting surface today is a muddle: a verb-named
`explain` property duplicates validation content; `summary_df` bolts a dependence
footer (`joint` row) onto a per-component theoretical summary; and `stats_df`
carries a joint dependence block (`cov`/`corr`/`copula_tau`/`E[A0A1]`). A
bivariate is essentially a **2-unit portfolio + a dependency structure**, so the
surface should split cleanly along that seam.

**F1 — delete `explain`.** Its only content is per-axis theory-vs-empirical
mean/cv — exactly what `Portfolio.summary_df`'s validation columns already carry.
`_explain_oneline` (the `info` `validation` row, `bivariate.py:1407`) repoints to
the new `summary_df` error columns. Call sites: `bivariate.py:1418` (in
`_explain_oneline`), tests `test_bivariate.py:287,723,771`.

**F2 — `summary_df` → exact `Portfolio.summary_df` shape.** One `Freq/Sev/Agg`
row block per component (X, Y) + a `total` block (today's `joint` row *becomes*
`total`, and `total` is the genuine X+Y aggregate — the anti-diagonal sum of the
joint density — with its own theory/empirical/error, not a dependence footer).
Columns are the Portfolio validation view (`EX | Est EX | Err EX | CV | Est CV |
Err CV | Sk | Est Sk`), SD-substituted when signed, mirroring
`Portfolio.summary_df`. Absorbs `explain`'s content. **All dependence leaves
`summary_df`.**

**F3 — strip the dependence block from `stats_df`.** Drop the `joint` column's
`cov` / `corr` / `copula_tau` / `E[A0A1]` rows (`bivariate.py:1227-1241`);
`stats_df` reverts to pure per-component marginal moments.

**F4 — new `dependency_df` (property).** Owns the joint-structure scalars. **Lean
by ruling** — only `cov`, linear `corr`, and the input copula `tau`; no rank
corr / Kendall on the realised grid (expensive — O(n²)+ on a 2^log2 grid), no
higher central comoments (cheap user-side numpy from `moments(3)` / `_S`). Two
rows × three columns:

| | cov | corr | tau |
|---|---|---|---|
| **Sev** | per-claim joint `self._S` (the copula severity matrix, already cached at `bivariate.py:1090`) | from `_S` | copula input `self.copula.tau()` |
| **Agg** | from `moments(2)` (realised aggregate joint) | from `moments(2)` | — (realised tau skipped) |

The Sev row is a cheap double-sum over the already-built `_S` against the
severity grids — no new copula machinery (author confirmed `_S` is always the
fft2 input in copula mode). In `netceded` mode there is no copula / no `_S`: the
Sev dependence is the deterministic ceded/net split of one severity — compute
`cov`/`corr` from the ceded/net severity arrays directly, `tau` = n/a.

Call-site sweep: `summary_df` / `stats_df` readers in `bivariate.py`, the
`_repr_html_`, tests (`test_bivariate.py`, `test_reins_bivariate.py`), and any
`.rst`/`.qmd` doc that prints a bivariate summary. New `dependency_df` needs its
own smoke test + a row in the bivariate test module.

## Proposed batch

- **In scope (decided):** A1, A3, B4, C5, D, **F**, E1. Each non-F item =
  rename/decorator change + call-site sweep (`src`+`tests`+`docs`) + docstring
  cross-ref updates, no alias. F = the bivariate reporting redesign.
- **Resolved:** D = plain noun properties (no `_df`); C5 kwarg renames in
  lockstep (`trim_density_df=…`); A2 superseded by F.
- **Shims:** E1 do now (low risk); **E2/E3 deferred** (pending "no external
  reader" confirmation); **E4 deferred** to a separate `.shape`→`.param` project.

## Verification

- `rg` finds zero `reins_describe`, `\.explain\b` (bv), `\.tvar_info_df\(`,
  `\.reins_kinds\(` outside changelog/history; the new names resolve.
- `hasattr(obj, 'summary_df')`-style smoke for each renamed member; the
  Distortion cache-invalidation list (`spectral.py:1007`) updated if a
  `cached_property` name changes.
- Full suite green (`UV_LINK_MODE=copy uv run pytest`); style-test trio remains
  the only (pre-existing) failure.
- Docs: grep the `.rst` tree for each old name; update `:meth:`↔`:attr:` roles
  and drop `()` where a method became a property. Note docs pending manual
  rebuild per CLAUDE.md.

## Close-out

- Bump `pyproject.toml` `1.0.0a*`; `CHANGELOG.md` section listing every rename as
  breaking; update the two naming memories if the rules sharpen.
- Move to `dev/done/` when the in-scope batch ships.

## Open questions

1. **C5 kwarg** — rename `update(trim_df=…)` → `trim_density_df=…` with the
   method? Recommend yes.
2. **E2/E3** — any out-of-repo reader of `_kind_immutable` /
   `_available_distortions_` / `_kinds_ordered`? (gate their deletion)
3. **A2 name** — `validation_df` good, or prefer `marginal_validation_df` /
   `reproduction_df` (it validates that marginals reproduce the standalones)?
