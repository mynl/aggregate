# plan-pentagon — one canonical, complete, consistent pricing readout

> **STATUS: PLAN FOR REVIEW (2026-06-05).** No code yet. Investigation done;
> design below. Maps to TODO **F3** (`pricing_at = P + Q` / Pentagon
> integration) and **F7** (output Pentagon-like objects). Phase 1 is
> alpha-tagged, zero-blast-radius (same numbers, consistent columns); Phase 2
> (Pentagon objects + `pentagon.py` cleanup) is early beta.

## The problem

The pricing methods all emit the same eight accounting quantities —
`L, M, P, Q, a` (loss, margin, premium, capital, assets) and the ratios
`LR = L/P`, `PQ = P/Q`, `ROE = M/Q` — as the column (or row) index of a
DataFrame. But **each method builds that set independently**, and they disagree
on order, naming, completeness, and dtype. Evidence (current tree):

| Emitter | Order as written | `M/Q` name | Ordered categorical? | Complete? |
|---|---|---|:--:|---|
| `pentagon.Pentagon.index` (`:69`) | `L,P,M,a,Q,LR,PQ,COC` | `COC` | n/a (Series) | 8 |
| `portfolio.PRICING_STAT_ORDER` (`:153`) | `L,M,P,Q,a,LR,PQ,ROE` | `ROE` | dtype defined | 8 |
| `portfolio.pricing_at` (`:2805`) | → `PRICING_STAT_ORDER` | `ROE` | **yes** | 8 |
| `portfolio.price` lifted (`:3137`) | builds `L,M,P,Q`, reindex | `ROE` | plain reindex | 8 |
| `portfolio.price_ccoc` (`:3253`) | `L,P,M,Q,a,LR,PQ,COC` | `COC` | no | 8 |
| `portfolio.analyze_distortion` audit (`:3298`) | `a,L,P,M,Q,LR,ROE,dname,dshape` | `ROE` | no | **no PQ**, +metadata |
| `portfolio.analyze_distortions` (`:3358`) | `L,LR,M,P,PQ,Q,ROE,a` (alpha) | `ROE` | **dropped by `.T`** | 8 |
| `distributions.Aggregate.price` (`:6583`) | `line,L,P,M,Q,a,LR,PQ,ROE` | `ROE` | no | 8 |
| `results.PricingResult` docstring (`:80`) | `…,LR,PQ,COC` | `COC` (!) | — | code emits `ROE` → drift |
| `results.AnalyzeDistortionResult` docstring (`:36`) | `L,LR,M,P,PQ,Q,ROE` | `ROE` | — | order ≠ actual audit_df |

Concrete defects this causes:
- **Two names for one quantity.** `M/Q` is `ROE` in `portfolio`, `COC` in
  `pentagon` / `price_ccoc` / one docstring. (Same number; issuer reads it as
  cost of capital, the buyer/equity reads it as return on equity.)
- **Five column orders.** A user concatenating or comparing two readouts gets
  silently misaligned or reordered columns.
- **Inconsistent completeness.** `analyze_distortion`'s audit drops `PQ` and
  mixes the stats with `dname`/`dshape` metadata in the same index.
- **The categorical is used in exactly one method.** `pricing_at` builds the
  ordered `CategoricalIndex`; `analyze_distortions` then **transposes and loses
  it** (the code comment at `:3358` admits this and re-applies a plain order).
- **The derivation `a=P+Q; LR=L/P; PQ=P/Q; ROE=M/Q` is hand-copied** in
  `pricing_at`, `price` (lifted), `price_ccoc`, and `Aggregate.price` — four
  chances to drift.
- **`Pentagon.solve()` already does the completion job** (any soluble triple →
  all 8, with identity checks) but **no emitter routes through it**, and it
  carries its own inconsistencies (lowercase `lr/pq/coc` kwargs vs uppercase
  `LR/PQ/COC` attrs; a `# TODO fragile` order coupling at `:420`; a redundant
  line in the `('M','a','Q')` case at `:273`).

## The idea (your two asks, made concrete)

1. **Ordered categorical index, defined once.** A single canonical
   `PENTAGON_STATS` list + `CategoricalDtype` that *every* readout uses for its
   stat axis (column index, or row index after a transpose — re-applied so it
   survives). The mechanism already works in `pricing_at`; this generalizes and
   centralizes it.
2. **One "complete the pentagon" entry point.** All emitters feed their known
   subset (in practice they already hold `L, P, M, Q` from the augmented row)
   into a single helper that fills `a` + the ratios and stamps the canonical
   categorical columns. The general "give me `P`, `L`, and `a` *or* `Q`, derive
   the rest" case is exactly `Pentagon.solve()` — we expose and reuse it instead
   of re-deriving inline.

## Decisions to confirm (recommendations in **bold**)

- **D1 — name for `M/Q`.** Recommend **`ROE`** for emitted columns (matches your
  prompt's list and 5 of 7 code sites), with **`CoC` documented as the
  synonym**. Reconcile `pentagon.py`, `price_ccoc`, and the `PricingResult`
  docstring (all currently `COC`) onto it. *(This mirrors the earlier
  `roe→ccoc` consolidation for distortion kinds; here the public accounting
  column stays `ROE`.)*
- **D2 — canonical order.** Recommend **`[L, M, P, Q, a, LR, PQ, ROE]`** — the
  accounting reading order (`L+M=P`, then `P+Q=a`, then the three ratios). This
  is today's `PRICING_STAT_ORDER`; `pentagon.Pentagon.index` (`L,P,M,a,Q,…`)
  becomes the outlier we align.
- **D3 — home of the contract.** Recommend **`pentagon.py`** (the accounting
  authority; a numpy/pandas-only leaf, so `portfolio`/`distributions` import it
  with no cycle). `PRICING_STAT_ORDER`/`PRICING_STAT_DTYPE` in `portfolio.py`
  are replaced by an import. Per CLAUDE.md `pentagon` stays submodule-access
  only — fine, this is an internal contract, not a top-level re-export.
- **D4 — single vs grouped axis.** Recommend a **single ordered categorical**
  (simplest, matches your ask). Note an *option*: a 2-level column index
  splitting "flows" (`L,M,P,Q,a`) from "ratios" (`LR,PQ,ROE`); deferred unless
  you want it.
- **D5 — how far to push Pentagon objects (Phase 2).** Recommend: DataFrames
  stay the primary multi-line / multi-distortion output (a `Pentagon` is
  scalar), **but the scalar total-level readouts become `Pentagon` objects** —
  starting with `analyze_distortion`'s audit (which is scalar and today is the
  most broken). Add `Portfolio.pentagon_at(p|a, distortion, line='total')` and
  `Pentagon.from_row(...)` as the object-flavored "easy constructor."

## Design

### The canonical contract (Phase 1)
In `pentagon.py`:
- `PENTAGON_STATS = ['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']` (per D1/D2).
- `PENTAGON_DTYPE = pd.CategoricalDtype(PENTAGON_STATS, ordered=True)`.
- `CORE_STATS = ['L', 'M', 'P', 'Q', 'a']`, `RATIO_STATS = ['LR', 'PQ', 'ROE']`
  for callers that want the split.
- `apply_pentagon_columns(df, *, axis='columns')` — stamp the categorical on the
  chosen axis in canonical order (survives transposes; fixes the
  `analyze_distortions` drop).

### The completion helper (Phase 1)
- `complete_pentagon(df)` — vectorized. Given a frame carrying the known core
  columns (the emitters all have `L, P, M, Q`), compute `a = P + Q` and the
  ratios, return canonical ordered categorical columns. One implementation,
  replacing the four hand-copied blocks.
- For genuinely partial input (only some of `L,M,P,Q,a` known), defer to
  `Pentagon.solve(**subset)` per row. These frames are tiny (a few lines × a few
  distortions), so a row-wise fallback is cheap.
- The "from `P, L`, and `a` or `Q`" workflow you described **is**
  `Pentagon.solve(L=…, P=…, a=…)` / `(…, Q=…)` — already implemented by the
  case-match; we expose it cleanly and make its output canonical.

### Pentagon object outputs (Phase 2)
- `Pentagon.from_row(aug_row, line='total')` — pull `L=exa[g]_{line}`,
  `P=exag_{line}`, `M=T.M_{line}`, `Q=T.Q_{line}`; solve; carry optional
  `distortion`/`shape` as **attributes**, not index rows.
- `Portfolio.pentagon_at(*, p=None, a=None, distortion, line='total')` → a
  `Pentagon` (the scalar analogue of `pricing_at`).
- `analyze_distortion` returns the total readout **as a `Pentagon`** (its
  `audit_df` becomes `pentagon.as_frame()`); `dname`/`dshape` move to attributes
  — fixes the missing-`PQ` and stats-vs-metadata-mixing defects.
- `pentagon.py` cleanups: unify `lr/pq/coc` kwargs ↔ `LR/PQ/COC` attrs (pick one
  casing, with the `ratios()`/`solve()` signatures matching the chosen stat
  names); remove the `# TODO fragile` order coupling (`:420`) by building from
  `PENTAGON_STATS`; drop the redundant `('M','a','Q')` line (`:273`); NumPy
  docstrings; optional `solve_frame`.

### Emitters routed through the contract
`pricing_at` (reference, minimal change), `price` (lifted + linear branches),
`price_ccoc`, `Aggregate.price`, `analyze_distortion`, `analyze_distortions`
(re-apply the categorical after the transpose). Update `results.py` docstrings
so `PricingResult` (currently `COC`) and `AnalyzeDistortionResult` match the
emitted contract.

## Files
- `src/aggregate/pentagon.py` — **new** contract constants + `complete_pentagon`
  / `apply_pentagon_columns`; Phase 2 object constructors + internal cleanups.
- `src/aggregate/portfolio.py` — replace `PRICING_STAT_ORDER`/`_DTYPE` (`:153`)
  with imports; route `pricing_at`, `price`, `price_ccoc`, `analyze_distortion`,
  `analyze_distortions` through the contract.
- `src/aggregate/distributions.py` — route `Aggregate.price` (`:6542`) through
  the contract.
- `src/aggregate/results.py` — docstrings to match (kill the COC/ROE drift);
  Phase 2: `AnalyzeDistortionResult` holds a `Pentagon`.

## Phases
- **Phase 1 `[A]` (zero blast radius).** Contract + `complete_pentagon`; the
  D1–D3 naming/order/home decisions; route all emitters; fix docstrings.
  **Same numbers**, consistent canonical columns everywhere.
- **Phase 2 `[B]`.** Pentagon-object outputs (audit → `Pentagon`,
  `pentagon_at`, `from_row`); `pentagon.py` internal reconciliation +
  vectorized `solve_frame`.

## Verification
- **Golden values unchanged:** snapshot each emitter's output before/after;
  assert numerically identical (Phase 1 is presentation-only).
- **Column contract:** every emitter's stat axis equals `PENTAGON_STATS` in
  order with `PENTAGON_DTYPE` (incl. after the `analyze_distortions` transpose).
- **Completeness:** every readout has all eight stats (fixes `analyze_distortion`
  missing `PQ`); metadata (`dname`/`dshape`) no longer in the stat axis.
- **Pentagon round-trip:** `Pentagon.solve` from each soluble triple of a known
  `(L,P,a)` reproduces the consistent pricing (extend `Pentagon.test_cases`);
  `make_possible_pentagons()` unchanged.
- `uv run pytest` green; `uv run ruff check src` clean.

## Open / watch
- `pentagon.py` `solve` is a big `match` over named triples — Phase 2 cleanup
  should keep that readable; the vectorized `complete_pentagon` only needs the
  common all-core-known path, so the `match` stays the general/scalar engine.
- If D5 grows (per-line Pentagons, what-if re-solving in the UI), revisit
  whether `pricing_at` should optionally return a dict of `Pentagon`s keyed by
  line — deferred.
