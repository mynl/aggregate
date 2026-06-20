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

> **STATUS: NOT STARTED — plan for review.** Inventory captured 2026-06-20 from a
> full pass over the four main classes. Author rulings 2026-06-20: **C5** rename
> `trim_df()` → `trim_density_df()`; **D** convert `corr` / `marginals` to
> properties and `augmented_dfs` stays as-is; **E4** (`.shape`) deferred. Items
> A1–A3, B4, C5, D are the in-scope batch. **One open naming snag:** `corr`
> returns a *scalar* and `marginals` a *tuple* — neither is a DataFrame, so the
> proposed `corr_df` / `marginals_df` would contradict the `_df`-⇒-DataFrame rule
> (see D and Open questions). The `update(trim_df=…)` kwarg and shims E1/E2/E3
> also still need a ruling.

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
| A2 | `explain` | BivariateAggregate | property→**df**, verb-named *and* no `_df` (docstring: returns a DataFrame of per-axis mean/cv theory-vs-empirical, not text) | **`validation_df`** | 1 src (`bivariate.py:1418`) + 3 tests (`test_bivariate.py:287,723,771`) |
| A3 | `tvar_info_df` | Distortion (base `spectral.py:903`; WtdTVaR `:2743`) | **method** returning a df, no args — siblings `info` / `summary_df` / `stats_df` / `density_df` are all `cached_property` | make it a **property** (name already correct); convert `self.tvar_info_df()` call at `spectral.py:2820` | 1 src |

Notes:
- A1: the partner `reins_density_df` / `reins_stats_df` are already correct, so
  `reins_summary_df` slots in cleanly beside them. The private worker
  `_reins_describe` / `_reins_describe_block` cache attrs keep their names
  (internal). Watch the `constants.py` comment that lists `reins_describe`.
- A2: `validation_df` mirrors `validation_explanation` (the narrative) and the
  `info` "validation" row the docstring references. Confirm `validation_df`
  doesn't collide with anything on `BivariateAggregate` (it does not today).
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

## Proposed batch

- **In scope (decided):** A1, A2, A3, B4, C5, D. Each = rename/decorator change +
  call-site sweep (`src`+`tests`+`docs`) + docstring cross-ref updates, no alias.
- **Resolve before/at implementation:** the D `corr`/`marginals` `_df` snag
  (D-keep vs D-df); whether the `update(trim_df=…)` kwarg renames with C5.
- **Shims:** E1 do now (low risk); E2/E3 pending "no external reader"
  confirmation; **E4 deferred** to a separate `.shape`→`.param` project.

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
