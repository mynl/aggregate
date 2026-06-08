# Plan A — fix the Portfolio combine grid (resolution + span, replace RMS)

## Status

Drafted, **not executed**. Split out of the former `dev/plan-bucket-sizing.md`
(now deleted) — this is the cheap, safe, high-value half: it fixes the headline
portfolio-combine bug with no change to the convolution core or the severity
grid. **Land this first.** The deeper, bivariate-relevant half (non-zero
aggregate *output* windows for high-mean / thin-tail cases) is
`dev/plan-bucket-window.md` (Plan B) and lands after this.

The bucket machinery took a lot of tuning to reach its current state — change it
carefully. All steps land together (one version bump).

## Motivation (the bug)

```python
port = build('port A agg A1 dfreq[4] dsev[3] agg A2 dfreq[3] dsev[-1 1]')
port.bs    # -> 2   (absurd; both units want bs=1)
```

Reproduced 2026-06-08: the port builds at `bs=2, log2=16`; both units are driven
onto that shared `bs=2` even though A1 is a point mass at 12 and A2 has integer
support `[-3, 3]` — both want `bs=1`.

**Root cause** (`Portfolio.best_bucket`, `portfolio.py:1700-1714`): the per-unit
recommended buckets are combined by **root-sum-square**, then `round_bucket`:

```python
bs = sum([a.recommend_bucket(log2, p=bucket_sizing_p) ** 2 for a in self]) ** 0.5
return round_bucket(bs)
```

Both units recommend `bs ≈ 1`. RMS gives `√(1²+1²) = 1.4142`, and `round_bucket`
rounds **aggressively up** — `round_bucket(1.4142)` hits the `rbs == 1 → 2.0`
branch (`utilities.py:243-245`), so `bs = 2`. RMS scales the *wrong* way: *k*
identical units → `round_bucket(b·√k)`, i.e. adding units *coarsens* the grid.
The author already flagged it — `TODO: Is this really the best approach?!` at
`portfolio.py:1704`.

This particular case routes through the **signed** combine path (A2 has a
negative atom, so `Portfolio._signed()` is `True`); there `best_bucket`'s RMS
value `bs_best` wins over the span floor in phase 2 (`portfolio.py:1818-1828`).
The **non-signed** path (`portfolio.py:1804`) calls `best_bucket` directly with
no span floor at all. Both paths are wrong; both are fixed here.

## The correct rule (replaces RMS)

A portfolio combine grid must satisfy two **independent** constraints; the right
rule is their **max**, never an RMS:

1. **Resolution** — the finest bucket any unit needs: `min_k bs_k`, where `bs_k`
   is each unit's *natural selected* `bs` from its own `_bs_window` (the
   per-unit window estimator), captured in a **phase-1 pre-pass**. A finer grid
   is strictly better provided it still fits — there is no "over-cost" from one
   unit forcing the portfolio finer.
2. **Span fit** — the summed support must fit in `N = 2**log2` buckets without
   wrapping: `W_tot / N`, where `W_tot = Σ_k W_k` and **`W_k` is the selected
   *method's* support window width** (`x_max − x_min` of the winning method row),
   *not* the padded `used`-row grid extent (`N·bs`, which is power-of-2 inflated
   and would needlessly re-coarsen).

→ `bs = round_bucket(max(min_k bs_k, W_tot / N))`.

- Motivating case: `max(1, 6/65536) = 1`. ✅ (A1 width 0, A2 width 6.)
- Wide continuous portfolio (k fat-tailed lines): `W_tot/N ≈ k×` a single
  line's bs and dominates, so bs coarsens as it should. ✅ (RMS *under*-sizes
  this — `√k` not `k` — and only ever "worked" via `round_bucket` rounding up.)

**Origin is unchanged in Plan A:** `x_min = 0` for a non-signed portfolio; the
existing signed-origin logic (`portfolio.py:1829-1836`) is kept as-is. Non-zero
output windows for high-mean *non-signed* books are Plan B.

## Design points (the enhancements folded in)

1. **Width source = selected-method support window**, not the `used` padded grid.
   Read it from each unit's `_bs_window_df`: the row with `selected == True`
   (its `x_max − x_min`), not the `used` row.
2. **Resolution = `min_k` of each unit's natural selected `bs`** from the
   phase-1 pass — *not* the post-combine `unit.bs` (which has already been driven
   onto the shared grid; in the motivating case the units read `bs=2.0` *after*
   the combine, masking their natural `1`).
3. **The non-signed path must grow a phase-1 loop.** Today it is a bare
   `best_bucket()` call (`portfolio.py:1804`) and never runs per-unit
   `_bs_window`, so it has no per-unit bs/widths to take the `max` over. It must
   mirror the signed path's per-unit pre-pass (`portfolio.py:1807-1816`).
4. **Span-fit vs. FFT padding.** `W_tot/N` is the bare no-wrap floor; the combine
   then applies `padding` (doubling) for headroom. Keep the bare floor (matches
   the current signed phase-2 intent) but **state it explicitly** in the
   docstring so the reliance on padding is documented, not accidental.
5. **`best_bucket` → `best_window`.** Add a new `Portfolio.best_window(log2,
   bs_in, bucket_sizing_p)` that returns the full decision `(bs, log2, x_min)`
   *and* builds `self._bs_window_df` (the unit-row table), via the phase-1
   pre-pass + the `max(resolution, span)` rule. Wire it into **both** combine
   paths and `update`. **Keep `best_bucket` (RMS) for now** as a side-by-side
   comparison aid, but mark it **`DELETE BEFORE BETA`** in its docstring and
   logger — it is no longer on the live path.
6. **Baseline must contain the target cases.** `bucket_baseline.py` loads only
   `test_suite`, which may not include the motivating signed port or a tiny
   discrete port. Add a small fixed **probe set** (the motivating port; a tiny
   all-integer discrete port) so the before/after diff actually contains the rows
   this change targets.

## What the modern machinery already gives us (no change needed)

- **`Aggregate._bs_window`** (`distributions.py:6158`) is the real per-aggregate
  sizer: it runs `moment` / `exact_discrete` / `bounded_small`, records each in
  `_bs_window_df`, and **already shrinks `log2` below the cap** for an
  `exact_discrete` window (`_size`, `distributions.py:6243-6249`). Plan A
  *consumes* its per-unit output; it does not touch it.
- **`Aggregate.update`** already routes through `_bs_window` for every aggregate
  (`distributions.py:4403`), so single aggregates are correct today. The bug is
  purely the Portfolio combine.

## Plan (staged — all lands together)

### Step 1 — capture the current sizing (do first, no code change)

`scripts/bucket_baseline.py` is written. Run from the repo root:

```
python scripts/bucket_baseline.py        # -> tests/data/bucket_baseline_{summary,windows}.csv
```

Then extend it (or a sibling probe block) with the fixed probe set so the
baseline holds the target rows:

- `port A agg A1 dfreq[4] dsev[3] agg A2 dfreq[3] dsev[-1 1]` (motivating, signed);
- a tiny all-integer discrete port (e.g. two `dfreq/dsev` units), to exercise the
  `log2 < 16` propagation.

These CSVs are a **one-time working reference** for this change, **not**
go-forward test infrastructure — do not wire a pytest around them. They live
under `tests/data/` for convenience and may be removed once the change is
reviewed.

### Step 2 — `best_window`: the resolution + span combine

- Add `Portfolio.best_window(log2, bs_in, bucket_sizing_p)`:
  - **phase-1 pre-pass** over `self.agg_list`: each unit's `_bs_window(log2, 0,
    None, bucket_sizing_p)` → capture natural `bs_k` and the selected-method
    support window `W_k` (from its `_bs_window_df` `selected` row);
  - `resolution = min_k bs_k`; `span = (Σ_k W_k) / N`;
  - `bs = float(bs_in) if bs_in > 0 else round_bucket(max(resolution, span))`;
  - build `self._bs_window_df` (unit rows + `used` row) via the existing
    `_build_bs_window_df` (`portfolio.py:1733`);
  - return `(bs, log2, x_min)` with `x_min = 0` (non-signed) or the existing
    signed origin estimate (signed).
- Wire `best_window` into `Portfolio._bs_window` (replace the `best_bucket` RMS
  term in phase 2, `portfolio.py:1826`) **and** the non-signed branch
  (`portfolio.py:1803-1805`), **and** the `bs == 0` branch of `update`
  (`portfolio.py:1889-1891`).
- Keep `best_bucket` (RMS) intact but flagged `DELETE BEFORE BETA`.

### Step 3 — discrete favouring + log2 propagation

- Confirm the per-unit `exact_discrete` minimal `log2` **propagates** through
  `best_window` (the combine should not re-inflate a tiny discrete port to the
  `log2=16` cap). The selected `log2` is per-unit; the combine should carry the
  max of the units' needed `log2` (still ≤ cap), not the cap itself.
- Confirm discrete units only leave `bs=1` when the integer support genuinely
  exceeds `N` at `bs=1` (large `dsev` atoms) — the `_bs_window` `_size` already
  enforces this; Plan A must not undo it.

### Step 4 — re-baseline and review

Re-run `scripts/bucket_baseline.py`; diff against the Step-1 snapshot. Walk every
changed row and confirm each move is an improvement (smaller/sensible `bs`,
smaller `log2`), not a regression.

## Verification

- **`uv run pytest`** — must stay green. None of the suite is expected to pin
  `bs`/`log2`; if any does, that is a finding to record.
- **`scripts/freeze_knowledge.py`** freeze-before / check-after — changing
  `bs`/`log2` on affected **ports** *will* move densities, so this is **not** an
  all-match. It becomes the **second review surface**: which ports moved, and is
  the finer/coarser grid the reason.
- **`scripts/bucket_baseline.py` diff** — the **primary** review surface for the
  sizing decisions themselves.
- **New permanent test `tests/test_bucket_sizing.py`** (small, committed —
  consistent with how a47 shipped `tests/test_approximate.py`):
  - the motivating `port` → `bs == 1`;
  - a tiny all-integer discrete `port` → `log2 < 16`;
  - a multi-line fat-tailed `port` still coarsens (span floor dominates, `bs`
    sensibly large) — guards against the rule under-sizing.

## Housekeeping (standing rules)

- Bump `pyproject.toml` `1.0.0a*` (behaviour change; one bump for the whole thing).
- `CHANGELOG.md`: describe the new `best_window` combine rule (resolution + span,
  replacing RMS), note `best_bucket` is retained-but-deprecated, and call out
  that some **ports** change grid (improvement — baseline diff as evidence).
- `dev/TODO.md`: one line under the appropriate track; mark on landing.
- `tests/test_bucket_sizing.py`: the committed targeted asserts above.
- `tests/data/bucket_baseline_*.csv`: one-time reference (commit alongside or
  delete once reviewed — author's call). Do **not** wire a pytest around them.
- **Doc review (not necessarily change):** `docs/2_user_guides/2_x_10mins.rst`
  references `best_bucket` and its recommendation table. `best_bucket` itself is
  unchanged (kept for comparison), so its explicit example stays valid; but the
  narrative around the *realised* port `bs` may shift. Flag for the author's doc
  rebuild; keep `.rst` edits in lockstep if the number moves. (Do **not** build
  docs in the iteration loop.)
- Move this plan to `dev/done/plan-bucket-combine.md` on landing.

## Out of scope

- **Non-zero aggregate *output* windows for non-signed high-mean / thin-tail
  cases** — that is Plan B (`dev/plan-bucket-window.md`), the bivariate enabler.
  Plan A keeps non-signed origins at 0.
- **The severity grid.** It always contains physical 0 on its lattice
  (`xs_sev = (arange(N) − i0)·bs`); this is an invariant and is untouched.
- The FFT/convolution core, the `exact_discrete` math itself (only its `log2`
  propagation), and `round_bucket`'s rounding table.

## Resolved (author decisions)

- **Resolution operator** — `min_k selected_bs` is optimal; the only proviso is
  that span-fit passes. No "over-cost" concern.
- **`best_bucket` → `best_window`** — new method carries all the info and is
  wired in; `best_bucket` kept for side-by-side comparison, flagged
  **DELETE BEFORE BETA**.
- **Baseline CSVs** — one-time reference, **not** wired into the suite.
- **Staging** — land it all together (no separate Step-1 commit).
