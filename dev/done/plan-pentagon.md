# plan-pentagon — one canonical, complete, consistent pricing readout

> **STATUS: DONE (2026-06-06, v1.0.0a31).** Executed in one pass; 1009 tests
> green, ruff clean. Maps to TODO **F3**
> (`pricing_at = P + Q` / Pentagon integration) and **F7** (output Pentagon-like
> objects). Executing both phases in **one pass** (per user). **Same numbers**,
> consistent canonical columns; the only intentional shape change is
> `analyze_distortion`'s `audit_df` (now a one-row frame; see D6). Folds in the
> review Q&A: orientation standard (D6 — all descriptors first, octet trailing
> `[-8:]`), no return-type changes (D5 revised), `Pentagon`-as-value-object (D7).

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

## Decision confirmed by user 

- **D1 — name for `M/Q`.** Use **`ROE`** for emitted columns (matches your
  prompt's list and 5 of 7 code sites), with **`CoC` documented as the
  synonym**. Reconcile `pentagon.py`, `price_ccoc`, and the `PricingResult`
  docstring (all currently `COC`) onto it. *(This mirrors the earlier
  `roe→ccoc` consolidation for distortion kinds; here the public accounting
  column stays `ROE`.)* Note: ROE is an ex post view, what return was achieved; COC is a prospective pricing view. Our presentation as a PnL leads to the ROE ex post view. Note the confusion between capital and equity - which involves capital tranching. For aggregate all capital is equity. 
- **D2 — canonical order.** **`[L, M, P, Q, a, LR, PQ, ROE]`** — the
  accounting reading order (`L+M=P`, then `P+Q=a`, then the three ratios). This
  is today's `PRICING_STAT_ORDER`; `pentagon.Pentagon.index` (`L,P,M,a,Q,…`) is adjusted to comport with this order. It is canonical throughout the project.  
- **D3 — home of the contract.**  **`pentagon.py`** (the accounting
  authority; a numpy/pandas-only leaf, so `portfolio`/`distributions` import it
  with no cycle). `PRICING_STAT_ORDER`/`PRICING_STAT_DTYPE` in `portfolio.py`
  are replaced by an import. Per CLAUDE.md `pentagon` stays submodule-access
  only — fine, this is an internal contract, not a top-level re-export.
- **D4 — single vs grouped axis.** **Single ordered categorical**
  (simplest, matches your ask). Note an *option*: a 2-level column index
  splitting "flows" (`L,M,P,Q,a`) from "ratios" (`LR,PQ,ROE`); deferred unless
  you want it.
- **D5 — how far to push Pentagon objects (Phase 2).** *Revised in review,
  see D6/D7 below.* **No existing DataFrame-returning method changes its return
  type.** `Pentagon` objects are *additive only*: a new
  `Portfolio.pentagon_at(p|a, distortion, line='total')` accessor and a
  `Pentagon.from_row(...)` constructor. The single-row readouts that are *broken*
  today (notably `analyze_distortion`'s audit) are fixed *in place* as plain
  frames (D6); the `Pentagon` object is the engine they call and a convenience
  the user can reach for, not a forced return type.

## Review Q&A (2026-06-06)

**Q1. Why does `analyze_distortion` (singular) return a *column* when everything
else returns a *row*?**
No good reason — it's an ad-hoc build, and it's the one readout we should
straighten. At `portfolio.py:3298` it hand-assembles a *vertical* frame:
`pd.DataFrame({'value': [a, L, P, M, Q, LR, ROE, dname, dshape]},
index=['a','L','P','M','Q','LR','ROE','dname','dshape'])`. So the eight
accounting stats run *down* a row index, there's a single `value` column, **and
the two metadata fields (`dname`, `dshape`) are mixed into the same stat index**
(which is also why it silently drops `PQ`). Every other emitter does the
opposite: stats on the **column** axis, one row per priced entity (line). The
singular is simply the lone hand-rolled single-row case that never got the
`pricing_at` treatment.

**D6 — one orientation standard for every readout.** Stats live on the
**column** axis as a single canonical 8-wide block (`PENTAGON_STATS`,
`PENTAGON_DTYPE`); each priced entity is a **row**. **All descriptor columns
(keys *and* metadata) come first; the pentagon octet is always the trailing
eight columns** — so it reads naturally left-to-right (who/what, then the
numbers) and is extractable with a fixed `df.iloc[:, -8:]` regardless of how
many descriptors precede it. (Revised from the earlier "keys prepend / metadata
append" — putting the octet in the *middle* defeats the clean `[-8:]` grab.)

| Readout | row(s) | descriptor columns (prepended) | pentagon block (trailing `[-8:]`) |
|---|---|---|---|
| `pricing_at`, `price`, `analyze_distortions` | per line | — (`line` is the index) | `L M P Q a LR PQ ROE` |
| `Aggregate.price` | one (`total`) | — (`line` is the index) | same |
| `analyze_distortion` (fixed) | one (`total`, `line` index) | `dname, dshape` | same |

This makes `analyze_distortion`'s audit a **one-row frame in the same shape as
every other readout** (a row, not a column) — its `PQ` reappears, the
descriptors `dname`/`dshape` lead, the octet is the trailing `[-8:]`, and
concatenating/diffing two readouts is finally safe.

**Q2. Should methods that currently return DataFrames start returning
`Pentagon`s instead?** **No — agreed.** Silently changing a return type is an
API break and a footgun for every notebook and downstream call. All current
df-returning methods keep returning frames. `Pentagon` is reached through *new*
names only (`pentagon_at`, `from_row`). This supersedes the original D5.

**Q3 / D7 — what is the `Pentagon` object *for*, and should it become a
DataFrame?** The object earns its place as the **single-row accounting record
(an 8-vector) + completion engine**: it (a) *completes a partial input* — give it any soluble
triple (e.g. `P, L`, and `a` *or* `Q`) and `solve()` fills the rest with
identity checks; (b) offers named attribute access (`.L`, `.Q`, `.ROE`); (c)
carries provenance (`distortion`, `shape`) as **attributes**, not index rows.
A DataFrame row does none of those.

It should **not** subclass `DataFrame` (pandas subclassing is brittle —
`_constructor`/`__finalize__` plumbing, surprising slice semantics). Instead
`Pentagon` *has* a frame: it stays a small value object and **emits** canonical
output via `as_series()` / `as_frame()` (both already exist; we just make them
canonical-ordered per D2). Crucially, the single-row `solve()` logic and the
vectorized `complete_pentagon` (Phase 1) are **the same accounting identities**
expressed two ways — so every df-emitter completes its rows *through* this code.
That is what makes `pentagon.py` visibly load-bearing rather than a curiosity:
the multi-row frames and the single-row object share one source of truth for both the
identities and the canonical column order.

**Q4 — the bar for "done."** A reader should never wonder why `pentagon.py`
exists. After this work: the constants and identities live there and *every*
pricing readout imports them; the partial-completion workflow
(`Pentagon.solve` / `pentagon_at`) is the natural, documented way to turn a
known subset into the full octet; and the module's name matches its job
(the accounting authority for the L–M–P–Q–a pentagon).

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

### `analyze_distortion` fixed in place (bug fix)
Per D6, rebuild `audit_df` as a **one-row frame** (a row, not a column): index
`['total']` (name `line`), descriptor columns `dname, dshape` first, then the
trailing eight `PENTAGON_STATS` (`PQ` restored), routed through
`complete_pentagon` like every other emitter. It changes the audit frame's
*shape* (now row-oriented, +`PQ`, descriptors off the stat axis) but keeps it a
DataFrame — no return-type change to the method.

### Pentagon object outputs (additive only)
New names; nothing existing changes return type (D5 revised / Q2).
- `Pentagon.from_row(aug_row, line='total', *, distortion=None)` — pull
  `L=exa[g]_{line}`, `P=exag_{line}`, `M=T.M_{line}`, `Q=T.Q_{line}`; solve;
  carry optional `distortion`/`shape` as **attributes**, not index rows.
- `Portfolio.pentagon_at(*, p=None, a=None, distortion, line='total')` → a
  `Pentagon` (the single-row, object-flavored analogue of `pricing_at`; the natural
  home for the "complete a partial input" workflow).
- `Pentagon` stays a value object that *has* a frame (D7): `as_series()` /
  `as_frame()` emit canonical-ordered output; it does **not** subclass
  `DataFrame`.
- `pentagon.py` cleanups: unify `lr/pq/coc` kwargs ↔ `LR/PQ/COC` attrs (pick one
  casing, with the `ratios()`/`solve()` signatures matching the chosen stat
  names); remove the `# TODO fragile` order coupling (`:420`) by building from
  `PENTAGON_STATS`; drop the redundant `('M','a','Q')` line (`:273`); NumPy
  docstrings; optional vectorized `solve_frame`.

### Emitters routed through the contract
`pricing_at` (reference, minimal change), `price` (lifted + linear branches),
`price_ccoc`, `Aggregate.price`, `analyze_distortion`, `analyze_distortions`
(re-apply the categorical after the transpose). Update `results.py` docstrings
so `PricingResult` (currently `COC`) and `AnalyzeDistortionResult` match the
emitted contract.

## Files
- `src/aggregate/pentagon.py` — **new** contract constants + `complete_pentagon`
  / `apply_pentagon_columns`; object constructors (`from_row`) + internal cleanups.
- `src/aggregate/portfolio.py` — replace `PRICING_STAT_ORDER`/`_DTYPE` (`:153`)
  with imports; route `pricing_at`, `price`, `price_ccoc`, `analyze_distortion`,
  `analyze_distortions` through the contract.
- `src/aggregate/distributions.py` — route `Aggregate.price` (`:6542`) through
  the contract.
- `src/aggregate/results.py` — docstrings to match (kill the COC/ROE drift) and
  document `AnalyzeDistortionResult.audit_df`'s new canonical one-row shape.
  `AnalyzeDistortionResult` keeps returning a frame; no Pentagon in the return
  type.

## Execution (one pass — Phase 1 + 2 together, per user)
Both phases land in a single change set:
1. **Contract** in `pentagon.py`: `PENTAGON_STATS`, `PENTAGON_DTYPE`,
   `CORE_STATS`/`RATIO_STATS`, `apply_pentagon_columns`, `complete_pentagon`;
   D1–D3 naming/order/home; the **D6 orientation standard**.
2. **Route all emitters** through the contract; fix `results.py` docstrings.
3. **`analyze_distortion`** audit rebuilt as a one-row frame (D6).
4. **Additive Pentagon accessors**: `Pentagon.from_row`,
   `Portfolio.pentagon_at`; `pentagon.py` internal reconciliation (kwarg/attr
   casing, drop `# TODO fragile` coupling and the redundant `('M','a','Q')`
   line, NumPy docstrings, optional vectorized `solve_frame`).

**Same numbers everywhere.** The only intentional shape change is
`analyze_distortion`'s `audit_df` (now a one-row canonical frame, `PQ` restored,
`dname`/`dshape` leading); flag it in the README/close-out notes.

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
  common all-core-known path, so the `match` stays the general single-row engine.
- If D5 grows (per-line Pentagons, what-if re-solving in the UI), revisit
  whether `pricing_at` should optionally return a dict of `Pentagon`s keyed by
  line — deferred.
