# Plan — signed-severity "bounded" window: kill the `int(inf)` overflow and the silent ignored-layer trap

> **Status: DRAFT — not executed.** A focused robustness fix in the bucket/window
> sizer plus a UX guard for layers on signed severities. Touches core grid-sizing
> code (`_bucket_window.py`, `_aggregate.py`, `_severity.py`), so the regression
> bar is "ordinary aggregates byte-for-byte unchanged."

---

## Reproduction

```python
build('agg NT 50 claims 25000 xs 0 ssev -lognorm 200 cv 10 + 180 mixed ig .4', bs=5)
# OverflowError: cannot convert float infinity to integer
#   _bucket_window.py:527  need = int(np.ceil(np.log2(max(span / bs + 1.0, 1.0))))
```

## What actually happens (confirmed empirically)

The crash is one symptom of **three** distinct defects, in increasing depth:

### Defect 1 — `_size` overflows on a non-finite window (the crash itself)
`_bucket_window.py:527` computes `need = int(np.ceil(np.log2(...)))` with
`span = +inf`, so `int(inf)` raises. Two aggravating details:
- The user **pinned `bs=5`**, so `need` is discarded one line later
  (`:528` `if bs_in > 0: l2 = log2`). The crash is in **dead arithmetic** —
  `need` is never used on the pinned-bs path.
- Even when `need` *is* used, a non-finite `span` should never reach `int()`.

### Defect 2 — `_bounded_severity_window` claims a window it doesn't have
`_aggregate.py:4413`. The method gates on `all(s.bounded for s in self.sevs)`,
which is **True** here: `bounded ≡ tail_class == BOUNDED`, satisfied by the
finite layer `exp_limit = 25000`. But the severity is **signed**, so the lower
edge (`:4431`) reads `s.fz.support()[0]`, and the reflect patch
(`_severity.py:1209`, `support = (d - hi, d - lo)`) gives
`fz.support() = (180 − ∞, 180 − 0) = (−inf, 180)`. So the method returns
`(−inf, 4.2e6)` — a "bounded" window with an unbounded edge. Confirmed:

```
bounded flag: [True]   signed flag: [True]   fz.support(): [(-inf, 180.0)]
_bounded_severity_window -> (-inf, 4226213.10)
```

The same hazard for the **upper** edge is already documented and patched in the
splice path (`_severity.py:1129–1136`: "`fz.support()` reporting the underlying
(0, inf) … `_bounded_severity_window` reads an infinite upper edge"). This is its
**lower-edge, signed-severity twin**, and it is unguarded.

### Defect 3 — the layer is **half-applied** on a reflected/signed severity
This is the real root cause (author insight). A layer `y xs a` means
`min(y, max(X − a, 0))`; for `25000 xs 0` on `X = 180 − lognorm` that is
`min(25000, max(X, 0)) = [0, 180]` — a *bounded, non-negative* severity. That is
exactly what an ordinary `sev` does (it "deliberately clamps `x<0 → 0`",
`plan-negative-x-agg.md:38`). But the build dispatch (`_severity.py:986`) checks
`sev_reflect` **first**:

```python
if self.sev_reflect:        # -lognorm ... + 180
    _apply_lb_ub(); _apply_reflect(); _apply_signed()   # identity — NO clamp
elif self.signed:           # ssev / negative-atom dsev
    _apply_lb_ub(); _apply_signed()                     # NO clamp
else:                       # ordinary sev
    ... _apply_layer_attachment()                       # the clamp lives here
```

So a **reflected** base (`_apply_reflect` sets `signed=True`, `:1210`) takes the
no-clamp path under *either* `sev` or `ssev` — the clamp is **unimplemented for
reflected/signed severities**, the case `plan-negative-x-agg.md §6` explicitly
deferred (the `_apply_signed` docstring cites it). **Yet the layer's metadata is
still recorded** (`self.limit = 25000` in `__init__`, hence
`bounded == tail_class == BOUNDED == True`). The layer is therefore **half-
applied**: bounding *bookkeeping* set, bounding *transform* skipped. That lie —
`bounded=True`/`limit=25000` over an actually-`(-inf, 180)` support — is what
feeds `_bounded_severity_window` the `(-inf, finite)` it chokes on (Defect 2).

Note the math: applying the clamp would make this **non-negative**, so `xs 0`
*destroys* the signed part entirely (the degenerate fully-clipping corner). A
*negative* attachment (`25000 xs -1000` → `max(X+1000, 0)`) is the meaningful
general case that keeps the signed region down to −1000. The real deferred
feature is "layered signed severity with arbitrary attachment."

For completeness, falling back to the moment window (Defect-2 fix alone) does
**not** rescue this particular program — the raw `180 − lognorm(cv 10)`
(σ_log ≈ 2.15) reaches ~14.5M buckets over 50 claims, so it builds *defective*:

```
# _bounded_severity_window patched to None -> moment window:
est_m = -185,602,242   bs=5  log2=16   x_min=-185,928,840
# + DefectiveDistributionWarning: PMF deficit 0.977   + "negative reach exceeds grid; clipping"
```

— which is honest (a genuinely unbounded signed severity), but is exactly why the
**semantic** decision (apply the clamp vs reject the clause) matters more than the
window fallback.

---

## Fix design

### Must-fix (robustness — no opaque crash, ever)

- **F1 — guard `_size` against a non-finite window** (`_bucket_window.py:_size`,
  ~`:516–544`).
  - F1a: when `bs_in > 0`, **skip the `need` computation entirely** (it is
    unused — `l2 = log2`). Cheapest fix; removes the crash on the pinned-bs path.
  - F1b: when computing `need`, treat a non-finite `span`/`bs` ratio as "needs
    more than the cap" rather than calling `int(inf)` — i.e. fall through to the
    coarsen branch or skip the row. Belt-and-suspenders for the *un*-pinned path
    (a non-finite window can still arrive there).

- **F2 — `_bounded_severity_window` returns `None` on a non-finite edge**
  (`_aggregate.py:4434`-ish, after `s_max, s_min = …`). A severity whose
  computed `s_min`/`s_max` is not finite is **not** boundedly windowable; return
  `None` so the sizer falls back to the moment window (the documented "caller
  selects this only when at least as tight" contract already tolerates `None`).
  This is the principled fix at the right layer and mirrors the existing
  upper-edge guard intent.

### The semantics — the real decision (Defect 3; pick one, see D1)

The half-applied layer must be resolved one of three ways:

- **F3-A — implement the clamp** for reflected/signed severities: apply
  `min(limit, max(X − attach, 0))` on the reflected base. `25000 xs 0` → bounded
  `[0,180]`; `bounded`/`limit` become *true*; `_bounded_severity_window` works
  unchanged. This **lifts deferred §6** (`plan-negative-x-agg.md`) — real scope,
  with grid/reporting ripples (post-`max(·,0)` the result is non-negative, so it
  composes as an ordinary bounded severity; a negative attachment keeps it
  signed). Most correct, biggest.
- **F3-B — reject the contradiction** *(lean for this plan)*: a layer/limit clause
  on a reflected/signed severity raises a clean build error ("layering (`xs` /
  limit) on a signed/reflected severity is not supported — drop the layer, or
  model the clamp explicitly"). Honest, small, matches today's documented stance.
  The user is told immediately instead of getting a lie or garbage.
- **F3-C — make the metadata honest**: keep no-clamp, but do **not** set
  `limit`/`bounded`/`attachment` from a layer clause on a signed/reflected base
  (strip them post-dispatch). `_bounded_severity_window` then correctly returns
  `None`, the moment window is used — no crash, honestly unbounded (still
  defective for this cv-10 program, but truthfully so). Silent about the dropped
  clause.

### Considered and deferred (do **not** do here)

- **Changing `bounded` / `tail_class` semantics** globally so a signed severity
  with non-finite reflected support is never `BOUNDED`. Correct in principle but
  broad: `tail_class` feeds many consumers (`tail.py`, plotting, descriptions).
  F2 (and F3-B/C) handle the *window* without re-classing the severity wholesale.
- **Promoting the 97.7%-deficit case to a hard refusal** (an
  `InfiniteVarianceError`-style raise when signed negative reach ≫ grid). The
  existing `DefectiveDistributionWarning` + negative-reach-clip already fire;
  escalating them is a separate policy call (see decision D2).

---

## Open decisions (author)

- **D1 — the layer-on-signed semantics (F3-A / F3-B / F3-C).** This is the
  central decision. *Lean: **F3-B*** (reject the contradictory clause cleanly) for
  this plan, with **F3-A** (implement the clamp — the genuinely "correct" reading)
  scheduled with the deferred §6 layered-signed-severity work where the grid and
  reporting implications are handled together. F3-C is the minimal "let it build
  honestly" option if you'd rather not raise. Confirm before building.
- **D2 — should a severe signed deficit ever hard-refuse (outcome B)?** Matches
  the `InfiniteVarianceError` philosophy (refuse rather than return garbage), but
  needs a threshold and is a behavior change beyond this bug. Recommend deferring
  to its own item unless you want it now.
- **D3 — F1a vs F1b vs both.** Recommend **both**: F1a removes the crash on the
  reported (pinned-bs) path; F1b hardens the general path. Small, independent.

---

## Implementation order

1. **F2** (`_bounded_severity_window` → `None` on non-finite edge) — the root
   correctness fix; rebuild the repro to confirm it no longer routes through
   `bounded_small`.
2. **F1a/F1b** (`_size` guards) — defense in depth; a non-finite window from any
   *other* method can no longer overflow.
3. **F3** (the chosen A/B/C semantics) — pending D1. B/C are small; A is the
   deferred §6 feature and would graduate to its own plan.
4. Regression sweep: `uv run pytest` green, with attention to bucket-sizing and
   signed/negative-x suites.

## Tests

Extend the bucket-sizing / signed suites (`tests/test_bucket_sizing.py`,
`tests/test_negative_x.py`):
- **No crash:** the repro program builds without `OverflowError` (asserts the
  specific failure is gone).
- **F2 unit:** `_bounded_severity_window` returns `None` for a signed severity
  with non-finite `fz.support()` (and still returns a finite window for a genuine
  non-signed bounded layer — the regression guard).
- **F1 unit:** `_size`-level — a synthetic `(x_lo=-inf, x_hi=finite, bs_in>0)`
  yields `l2 = log2` with no exception.
- **F3 (per D1):** B — a layer/limit clause on a reflected/signed severity raises
  the matched error; C — it builds with honest (unbounded) metadata and the layer
  clause dropped; A — `25000 xs 0 ssev -lognorm 200 cv 10 + 180` builds bounded on
  `[0,180]`. In every case a layer on a **non-signed** severity is unchanged
  (clamps as before).
- **No regression:** a normal bounded layer (`1000 xs 0 sev …`, non-signed)
  selects `bounded_small` exactly as before (snapshot a couple of
  `_bs_window_df` rows).

## Housekeeping

Plan-based code change → bump `1.0.0a*` in `pyproject.toml`; add a `CHANGELOG.md`
section (robustness: signed-severity bounded-window overflow fixed; ignored-layer
warning); move this plan to `dev/done/` and tick `dev/TODO.md` at close.
