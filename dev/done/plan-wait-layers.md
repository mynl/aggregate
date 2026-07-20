# [Wait-Clause-Layers] — `y xs a` layers on renewal wait distributions

**Status: EXECUTED 2026-07-20, shipped as 1.0.0a147.** One deviation from the
plan as written: the cap atom exactly on the lattice exposed a readout-convention
issue (sums `S_k` landing exactly at `T` were counted at half weight — `min(W,
0.5)` waits at `T = 10` gave `EN = 19.5`), so snapped grids use the
closed-interval readout like the exact-lattice path; `convergence_check` keeps
it at h/2 (O(h) continuous convergence there, documented). Also recorded: `wait
expon [0 .5]` is a shape-vector mixture, NOT a window — the window is
`wait expon splice [0 .5] !`.

## Context

1.0.0a146 shipped the Sparre-Andersen renewal frequency (`years` / `wait` / `dwait`,
plan `dev/done/plan-sparre-a.md`). The author now wants the sev-style layer
transform available on the wait clause: for a wait law `W`,

- **conditional** (default, no `!`): `W' = ((W − a) | W > a) ∧ y` — short raw
  waits conditioned away, cap atom at `y` of mass `P(W > a+y)/P(W > a)`;
- **unconditional** (`!`): `W' = min((W − a)+, y)` — raw waits `≤ a` collapse to
  a zero-wait atom of mass `P(W ≤ a)` → **simultaneous-claim clusters**, plus
  the cap atom at `y`.

Author decision (2026-07-20): **conditional default, `!` = unconditional** —
DecL-consistent with the sev mini-language the wait clause reuses.

**Why this is cheap:** `Severity` already implements the full layer transform —
`_apply_layer_attachment` (`_severity.py:1143`) builds `min((X−a)+, y)` with both
conditioning flavors, including the unconditional **atom at 0** (`cdf(0) = 1 −
pattach`, `_severity.py:334-358`). `Severity._cdf/_sf` route through the layered
wrappers (`_severity.py:1339-1343`) and `moms()` returns layered moments
(`_severity.py:1376`). So the a146 kernel consumes a layered wait with **zero
changes**: `wait_count_pmf` already reads `p0 = sev.cdf(0.0)` per component and
recomposes clusters via `geometric_batch_compose`; grid sizing uses `sev.moms()`.
The work is grammar + plumbing + one numerics refinement (cap atom on-grid).

## Design decisions

- **Syntax:** `wait y xs a <dist> [!]` — layer term before the distribution,
  mirroring the exposures layer clause (`100 claims 30 xs 20 sev ...`). The
  `XS.2` terminal exists (`decl.lark:661`) and `xs` is already in the ID
  exclusion list (`decl.lark:715`) — no lexer work.
- **Spec keys / kwargs:** `wait_attachment` (default `None`) and `wait_limit`
  (default `np.inf`), matching sev's `exp_attachment`/`exp_limit` semantics
  (`None` vs `0` distinction preserved). Name vetting done at planning time:
  `rg` shows neither name exists anywhere in `src/aggregate` — no collisions on
  `Aggregate` / `Frequency` / `Severity` surfaces.
- **Splice + layer on the same wait: rejected** with a clear `ValueError`
  (transformer AND `_build_renewal_frequency`, so the programmatic API is
  covered). Can be relaxed later if ever wanted.
- **Layers on `dwait`: not supported** — write the clamped outcomes directly.
  The `dwait` grammar alternative is untouched.
- **Vector layers broadcast** like every other `wait_*` term (they join the
  `np.broadcast_arrays` list) — a mixture of differently-layered waits, weights
  via `wait_wt` as usual. Free, no special casing.
- **Cap-atom accuracy:** a finite `y` is a genuine atom in the layered wait law.
  Snap the wait grid so `y` lies exactly on the lattice when `{y, T}` are
  commensurable (Fraction-gcd, reusing the `_lattice_step` idiom); otherwise
  document the O(h/2) placement smear (~1e-5 count error at log2 22 — same
  scale as the known uniform-endpoint artifact).
- **Component tuple unchanged:** layered components pass `(sev_w, 0.0, np.inf,
  True)` — the layer lives *inside* the Severity; the orchestrator's
  defective-window mask (`conditional=False` tuple flag) remains exclusively the
  splice-`!` path. `wait_conditional` maps to `sev_conditional` on the layered
  branch instead of triggering the window mask.
- Interaction check (verified in planning): unconditional layered ⇒
  `sev.cdf(0) = P(W ≤ a)` feeds the existing `p0s` collection; `cdf(0⁻) = 0` so
  no negative-mass warning; `moms()/q_pos` yields exactly the conditional
  positive-wait moments used for sizing. `bed[0] = max(bed[0] − p0_, 0)` split
  works identically to the shipped dwait-with-zero path.

## Stages

### Stage 1 — Grammar + transformer
- `src/aggregate/decl.lark` (`wait_clause`, ~line 274): add the layered
  alternative:
  ```
  wait_clause: WAIT sev as_label                        -> wait_clause_wait
             | WAIT numbers XS numbers sev as_label     -> wait_clause_layer
             | dwait as_label                           -> wait_clause_dwait
  ```
- `src/aggregate/parser.py`: `wait_clause_layer(self, c)` — unpack
  `(_wait, limit, _xs, attach, sev, as_label)` (same `limit xs attach` order as
  `layers_xs`, `parser.py:1841`), run `_sev_to_wait(sev)`, then set
  `wait_limit`/`wait_attachment`; raise `ValueError` if the fragment carries a
  non-default splice (`wait_lb != 0` or `wait_ub != inf`). `_sev_to_wait` itself
  is unchanged (layers never appear inside a `sev` subtree). Prob policy and
  `_wait_label` handling identical to `wait_clause_wait`.
- No `parser_errors.py` / `decl_pygments.py` changes (`xs` already labeled and
  colorized).

### Stage 2 — Aggregate plumbing
- `src/aggregate/_aggregate.py` `__init__` (~line 1408): add
  `wait_attachment=None, wait_limit=np.inf` kwargs + docstring entries; pass to
  `_build_renewal_frequency`.
- `_build_renewal_frequency` (~line 2090): add both to the broadcast list.
  Branch order: `dhistogram` (unchanged) → **layered** (attachment not None or
  finite limit): `Severity(_wn, _watt, _wlim, ..., sev_conditional=wait_conditional)`,
  tuple `(sev_w, 0.0, np.inf, True)`; reject layer+splice here too → splice
  branches (unchanged). A layered `dhistogram` raises (grammar can't produce it;
  guards the programmatic API).
- `_frequency_program` renewal branch strips by `wait_` prefix — covers the new
  keys automatically; verify only.

### Stage 3 — Hard-atom grid snap (`src/aggregate/_renewal.py`)
- `wait_grid(..., hard_atoms=None)`: after continuous sizing picks `h_cand`,
  if `hard_atoms` are commensurable with `T` (`_lattice_step(hard_atoms, T)` →
  `step`), refine `bs = step / 2**j` for the smallest `j` with `bs ≤ h_cand`;
  then `n1 = T/bs` is exactly integer and every hard atom sits on the lattice.
  Record a `hard_atom_snap` row in `bs_df` (feasible/selected flags like the
  existing rows). This is *not* the coarse exact-lattice override — the grid
  stays fine (continuous part needs it); it is phase-aligned only.
- `wait_count_pmf`: collect finite cap points (`sev.limit` where
  `sev.detachment < inf`, layered components both flavors) and pass as
  `hard_atoms`. `convergence_check`'s h/2 override still divides `step` evenly.

### Stage 4 — Writer + corpus
- `src/aggregate/decl_writer.py` `_render_wait` (~line 359): when `wait_limit`
  present/finite (gate like `_render_layers` gates on `exp_limit`), emit
  `wait {limit} xs {attachment} {dist}`; `!` already rides through the
  `sev_conditional` view key.
- Corpus: +3 programs in `src/aggregate/agg/_test_suite.agg` Y section
  (conditional layer, unconditional layer w/ clusters, layered mixture),
  blank-line separated; mirror in `src/aggregate/agg/decl-testers.agg` AB
  section (`;`-terminated). Re-capture `tests/data/expected_specs.json`
  (additive-only — verify diff).

### Stage 5 — Tests
- `tests/test_renewal_agg.py` — exact closed forms via expon memorylessness:
  - `wait inf xs a expon μ` (conditional) ≡ plain `wait expon μ` — identical
    counts to kernel noise (~2e-9); the flagship exact test.
  - `wait inf xs a expon μ !` ≡ `geometric_batch_compose(plain counts,
    p0 = 1 − e^{−a/μ})` — direct cluster-formula cross-check.
  - `a=None/0, y=inf` layered spec ≡ plain wait — degenerate regression.
  - Cap-heavy: expon with mean ≫ y, small `y` commensurable with `T` ⇒ count
    ≈ deterministic ⌊T/y⌋ (validates cap-atom snap; compare vs `dwait [y]`).
  - Mixture broadcast (vector attachment) builds and foots to 1.
- `tests/test_renewal_decl.py`: exact spec dicts both flavors; splice+layer
  `ValueError`; `wait 2 xs 1 dwait ...` fails to parse; label round-trip.
- `tests/test_renewal.py`: `wait_grid` hard-atom snap — bs divides step, `y`
  on-lattice, `bs_df` row present; incommensurable falls back cleanly.
- Writer round-trip through the unparser suite.

### Stage 6 — Release hygiene
- `pyproject.toml` → 1.0.0a147; `CHANGELOG.md` section; `dev/TODO.md`;
  `dev/FEATURES.csv` (Aggregate kwarg surface note, re-run introspection
  cross-check); grammar ref regen
  (`uv run python -c "from aggregate.parser import grammar; grammar(add_to_doc=True)"` —
  writes `docs/4_agg_language_reference/ref_include.rst` directly); update
  memory (`project_sparre_renewal_plan.md` gains the layers note).
- Commits per stage, one-line subjects, `[Wait-Clause-Layers]` prefix.

## Verification
- Edit loop: `uv run pytest tests/test_renewal.py tests/test_renewal_decl.py
  tests/test_renewal_agg.py` plus the writer/parse suites touched.
- Smoke: `build('agg T 10 years sev lognorm 100 cv 2 wait 2 xs 0.25 expon 1')` —
  inspect `a.frequency.wait_p0` (0 conditional, `1−e^{−0.25}` with `!`),
  `_renewal_bs_df` shows the snap row; compare mean counts vs plain expon.
- Gate before declaring done: full `uv run pytest -m 'slow or not slow'`
  (expected ~2470+ passed).
