# Plan: [Closed-Support-Window] the grid must contain the support it was sized for

Status: specified and ruled by the author 2026-09-25, ready to execute. One
version bump. One open question, marked below, on how far to carry the fix.

## Goal

A bucket grid sized to cover a window `[0, A_hi]` must actually contain `A_hi`.
Today, in two places, it can end exactly on it, which silently deletes the top
half bucket of mass. Make the realized extent strictly exceed the window in
every branch that claims to cover it.

The user-visible symptom, and the reason this was written:

```
>>> a = build('agg C 1 claim sev cantor fixed')
>>> a.est_m, a.validation_description
(0.4997557401074740, 'fails sev mean, agg mean')      # exact answer is 0.5
```

A default build of a perfectly ordinary bounded severity fails validation out of
the box. After the change it returns `0.5` exactly and validates. `uniform` and
`beta` on `[0, 1]` improve in the same breath.

## Current behavior

### The counting convention is already right, in three places

`_need_log2` (`src/aggregate/_bucket_window.py`, around line 583) counts buckets
**inclusively**, which is correct for a closed support: a window of width `span`
at step `bs` needs `span/bs + 1` grid points, not `span/bs`.

```python
ratio = span / bs + 1.0
return int(np.ceil(np.log2(max(ratio, 1.0))))
```

Two further call sites compute the same quantity inline and also carry the
`+ 1.0` (around lines 1252 and 1273), and the bivariate sizer independently uses
the same convention as `round_bucket(widths[i] / ((1 << L_i) - 1))` (around lines
1630 and 1715). So the house convention is settled and appears four times.

### Two "coarsen to fit" fallbacks forget it

Inside `_size` (around line 893) the flow is: pick a `bs`, ask `_need_log2` how
much `log2` that needs, and if it needs more than the cap, coarsen `bs` so it
fits. The coarsening step is the bug.

```python
bs = round_bucket(W / N0) if W > 0 else 1.0     # line 929: first pass
need = _need_log2(span, bs)                     # inclusive, correct
if need <= log2:
    l2 = min(log2, max(need, 1))                # fine: extent > span, strictly
elif ...
elif np.isfinite(span):
    bs = round_bucket(span / N0)                # line 954: THE DEFECT
    l2 = log2
```

The fallback re-runs the same **exclusive** division that produced the `bs` whose
`need` just overflowed the cap. `round_bucket` rounds up to a binary rung, so
when `span / N0` already sits exactly on a rung the "coarsened" `bs` is
bit-identical to the one before it, `log2` is pinned at the cap, and the realized
extent is exactly `span`. Nothing was coarsened and the grid ends on the support
top.

The identical pattern appears a second time around line 1256, on the signed
subject-floor path, whose own comment says the grid **must** cover
`[sbj_lo, sbj_hi]` at any `log2` because otherwise "the FFT *wraps* and corrupts
the whole law (the LNS 47% mass-loss / aliasing failure)":

```python
need = int(np.ceil(np.log2(max(span / keep_bs + 1.0, 1.0))))   # inclusive
if need <= log2:
    _apply_floor(x0, floor_hi, keep_bs, max(need, sel_l2))
else:
    bs_f = round_bucket(span / (1 << log2))                    # exclusive
```

So a correctness-critical path is guarded by an inclusive test and repaired by an
exclusive one.

### What the deleted mass costs

The grid stops at `A_hi`, and with the default `sev_calc='discrete'` the last
bucket is centered there, so it covers only to `A_hi - bs/2`. The mass in
`(A_hi - bs/2, A_hi]` is dropped and `normalize=True` then spreads the shortfall
proportionally over the whole support, which drags the mean down.

The size of the loss is `sf(A_hi - bs/2)`, and how big that is depends on how
thick the severity is against its own upper endpoint:

| severity on `[0, 1]` | mass lost at `bs = 2**-16` | relative mean error |
|---|---|---|
| `beta 2 3` | about `1e-12` | `2.8e-15` |
| `uniform` | `7.6e-6` | `7.6e-6` |
| `cantor` (middle thirds) | `6.0e-4` | `4.9e-4` |
| `cantor 0.5` (middle half) | `2.4e-3` | `2.0e-3` |

A smooth density vanishing at the endpoint hides it. A uniform pays `O(bs)`. The
Cantor law pays `O(bs**alpha)` with `alpha = log 2 / log 3` about `0.631`,
because its distribution function is Holder with that exponent and not Lipschitz,
and that is what pushes it past the `1e-4` validation threshold. The defect is
general; Cantor is merely the first severity thick enough at its support end to
make it visible.

### Who it bites

`round_bucket` returns binary rungs and `N0 = 2**log2`, so `span / N0` lands
exactly on a rung if and only if **`span` is itself a power of two**. That is
rare for a book measured in currency and universal for a bounded severity on its
natural scale: `span = 1` for `uniform`, `beta`, `cantor`, and any `X * dist` on
the unit interval. `CurveBeta` in the shipped library is `100000 * beta 2 5`,
whose `span / N0` of `1.5259` is not a rung, so it already over-covers and is
untouched.

## The change

Divide by the inclusive count in both fallbacks, matching `_need_log2` and the
bivariate sizer.

```python
# _size, around line 954
- bs = round_bucket(span / N0)
+ bs = round_bucket(span / (N0 - 1))

# signed subject-floor, around line 1256
- bs_f = round_bucket(span / (1 << log2))
+ bs_f = round_bucket(span / ((1 << log2) - 1))
```

Line 929, the first pass, is deliberately **not** changed. When its `bs` yields
`need <= log2` the grid already covers strictly, because `need` carries the `+ 1`.
Only the branch that ignores `need` is wrong.

Update the `_size` docstring, which currently says `bs` is "a resolution
`round_bucket(W / 2**cap)`", to state the convention: the realized extent must
strictly exceed the window, so the coarsening divides by `2**cap - 1`, one grid
point per bucket boundary plus the closing one.

## Evidence already gathered

Both edits were applied to a scratch copy and measured, then reverted. Nothing in
this section needs re-deriving; it needs confirming.

- **Full suite green, unchanged.** `uv run pytest -m 'slow or not slow'` gave
  `5209 passed`, the same count as without the change. No test moves.
- **No shipped entry moves.** All 197 `library.agg` entries were built with and
  without the change and compared on `(bs, log2)`. **Zero differ.**
- **The target case is fixed.** `agg C 1 claim sev cantor fixed` goes from
  `bs = 2**-16, extent 1.0, est_m 0.499755740107474, "fails sev mean, agg mean"`
  to `bs = 2**-15, extent 2.0, est_m 0.500000000000000, "not unreasonable"`.
- **Three neighbors improve, none regress.** On the same default build,
  `uniform` goes `0.499996185273630` to exactly `0.5`, `beta 2 3` goes
  `0.399999999999999` to `0.4`, `cantor 0.5` goes `0.499021526418787` to exactly
  `0.5`. Unbounded and compound cases are untouched:
  `10 claims sev cantor poisson` keeps `bs = 0.000976562`, and
  `10 claims 1000 xs 0 sev lognorm 100 cv 2 poisson` keeps `bs = 0.25`.

The cost is grid efficiency, not accuracy. A bounded severity on a power-of-two
span now uses half its buckets, giving up one bit of resolution. The FFT length
is unchanged, so there is **no speed cost**. Doubling `log2` instead would keep
the resolution and double the transform, which is the wrong trade here.

## Open question for the author

**Does the `Portfolio` sizer get the same treatment?** Around line 1693 the
portfolio path has the same shape, an exclusive `span = W_ext / N_cap` feeding
`round_bucket`, with an inclusive `need = ceil(log2((x_hi - x_min)/bs + 1.0))`
capped by `min(log2, ...)` just below it. Measured, it does **not** bite today:
`W_ext` is an already-extended moment window, so a two-unit portfolio of unit
severities lands at extent 4 for a span of 2 and a Cantor pair at extent 8, both
comfortably clear. Two defensible answers, and the plan is written to the first:

1. **Fix it too, for consistency** (recommended). The pattern is identical and
   leaving one instance behind is how this bug survives the next refactor. It
   currently changes nothing, so the risk is nil.
2. **Leave it, and add a comment** saying the window is pre-extended so the
   exclusive division is safe here.

If the author does not rule, take option 1.

## What this does NOT fix

Say so in the CHANGELOG rather than implying the hole is closed.

- **A pinned `bs`.** On the `bs_in > 0` path `_size` states that "the user pinned
  bs, so the grid is theirs and the cap `log2` is honored verbatim", and `need`
  is never consulted. `build(prog, bs=1/65536, log2=16)` against a unit support
  still ends exactly at 1.0 and still deletes that half bucket. The user asked
  for that grid, so honoring it is defensible, but the mass loss is silent.
- **A genuinely unbounded severity.** There the residual past the grid top is
  real tail, not a half bucket, and truncating it is the intended behavior. The
  `DefectiveDistributionWarning` already reports it.

Both point at the same follow-on, deliberately **out of scope here** and to be
filed in `dev/TODO.md` as `[Bounded-Residual-Lump]`: for a severity whose support
end lies within one bucket of the grid top, add `sf(top edge)` into the last
bucket instead of letting `normalize` spread it over the whole support. The error
becomes `mass * 3*bs/4` rather than `mass * A_hi`, about `7e-9` instead of
`4.9e-4` for Cantor, and it costs nothing. It makes the discretization honest
whatever grid it is handed, where this plan makes the grid honest. Neither
subsumes the other.

## Alternatives considered and rejected

- **Grow `log2` by one instead of coarsening `bs`.** Keeps the resolution, but
  doubles the FFT and exceeds a cap the caller asked for. The `grow_cap`
  machinery is precedent for exceeding the cap, but it exists to stop atoms being
  mis-placed off a lattice, which is correctness. Spending a doubling on one bit
  of resolution is not the same trade.
- **Guard the widening on `sf(top edge) > tolerance`.** Proposed before the root
  cause was found, and unnecessary once it was: `round_bucket` already makes the
  change self-limiting to exactly the pathological case, so a probe would add a
  severity evaluation to the sizer for no benefit.
- **Relax the validation threshold, or teach validation about discretization
  error.** Rejected outright. The mean really is wrong by `4.9e-4`; the gate is
  doing its job.
- **Let the severity declare a natural lattice for the sizer to snap to.** That
  is `[Natural-Lattice-Snap]` in `dev/TODO.md` and it does not fix this: the
  lattice path runs through the same `need` and coarsen branches, so it inherits
  the defect rather than curing it, and auto-snapping would make every Cantor
  compound about eight times more expensive for no accuracy gain.

## Files touched

| File | Change |
|---|---|
| `src/aggregate/_bucket_window.py` | two (or three, per the open question) exclusive divisions become inclusive; `_size` docstring states the convention |
| `tests/test_bucket_sizing.py` | new cases pinning the invariant |
| `dev/TODO.md` | file `[Bounded-Residual-Lump]` as the follow-on |
| `pyproject.toml`, `CHANGELOG.md` | bump to `1.0.0a348`, entry |

## Stages and commits

One phase, one bump, one commit: `[Closed-Support-Window] a348`. There is no
sensible way to split a two-line change, and the tests belong with it.

## Acceptance checks

New cases in `tests/test_bucket_sizing.py`:

- **The invariant, stated directly.** For each of
  `sev uniform`, `sev beta 2 3`, `sev cantor`, `sev cantor 0.5` at
  `1 claim ... fixed` with no explicit grid, assert
  `a.bs * (1 << a.log2) > a.sevs[0].fz.support()[1]`, strictly. This is the
  property the plan is named for and it is what would have caught the bug.
- **The moments that were wrong are now exact.** `est_m` equals `0.5`, `0.4`,
  `0.5`, `0.5` respectively to `1e-13`, and each reports
  `validation_description == 'not unreasonable'`.
- **Scaled supports are untouched.** `1 claim sev 100000 * beta 2 5 fixed` keeps
  `bs = 2`, confirming the change only bites on a power-of-two span.
- **Unbounded and compound grids are untouched.**
  `10 claims sev cantor poisson` keeps `bs = 0.000976562` and
  `10 claims 1000 xs 0 sev lognorm 100 cv 2 poisson` keeps `bs = 0.25`.
- **The library does not move.** Build every `library.agg` entry and assert the
  `(bs, log2)` pairs are unchanged from the values recorded in the Evidence
  section above. A moved entry is a finding, not a re-capture.

Gate: tier 3, `uv run pytest -m 'slow or not slow'`, expected `5209 passed` with
no re-captured baselines. This change touches numerics, so also run the
`RuntimeWarning` gate, `-W error::RuntimeWarning`, and note that
`test_every_library_entry_builds[agg:RenewalDeterministicWait]` fails it today
for an unrelated reason recorded in `dev/done/plan-cantor-LIB.md` finding F3.

If any baseline or snapshot does move, **stop and report it** rather than
re-capturing: the measurement above says nothing should, so a diff means the
change is doing more than this plan describes.

## Execution log, 2026-09-25, landed as `1.0.0a348`

Executed from `a347` with a clean tree. Every fact the plan asserts about the
code was re-verified first and all of it held: lines 929, 954, 1256 and 1693
read exactly as quoted, and the symptom reproduced to the digit
(`est_m = 0.499755740107474`, `fails sev mean, agg mean`).

**The open question was taken as option 1**, per the plan's own instruction to
do so absent a ruling. The portfolio sizer got the same treatment.

Three recorded divergences:

1. **A fourth site changed, not three.** Immediately below the portfolio
   `span = W_ext / N_cap` sits the signed wrap-safety floor,
   `span = max(span, max_k W_k / N_cap)`, which computes the identical
   quantity. Fixing only the first would have left a `max()` comparing an
   inclusive count against an exclusive one, which is worse than either
   convention alone, so both moved together.
2. **Each division is guarded as `max(N - 1, 1)`.** `underwriter.py` requires
   `log2 > 0`, but `Portfolio` (`_portfolio.py`, around line 2444) accepts
   `log2 >= 0`, and at `log2 = 0` the new denominator is zero. The guard costs
   nothing and the old `if N_cap else W_ext` it replaces was dead, since
   `1 << 0` is `1` and never falsy.
3. **"The library does not move" was run as a measurement, not committed as a
   test.** All 197 entries were built on `a347`, the change applied, and all 197
   rebuilt and compared on `(bs, log2)`: zero differ, confirming the plan's
   evidence. It is not a committed test because the vehicle for one is stale and
   building one properly is more machinery than this fix warrants. The captured
   `tests/data/bucket_baseline_summary.csv` holds 146 rows against today's 197,
   no test reads it, and `bvagg` entries carry a list-valued `bs` and no `log2`
   while four entries refuse to build at all. **For the author:** a live
   `(bs, log2)` pin over the shipped library, folded into the existing
   `test_every_library_entry_builds` so it costs no extra builds, is worth
   having and is the net that would have caught this class of bug. It is not
   filed in `dev/TODO.md` because it is a test-infrastructure call rather than a
   library change.

Gates, both run at the bump:

- `uv run pytest -m 'slow or not slow'` — **5220 passed** in 121 s. That is the
  plan's predicted `5209` plus the 11 new cases. No baseline or snapshot moved.
- `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning` — 1 failed,
  5219 passed. The one failure is
  `test_every_library_entry_builds[agg:RenewalDeterministicWait]`, exactly the
  pre-existing unrelated failure the plan predicts, recorded as finding F3 in
  `dev/done/plan-cantor-LIB.md`.

All four acceptance measurements landed on the predicted values: the invariant
holds strictly for `uniform`, `beta 2 3`, `cantor` and `cantor 0.5`; their
`est_m` are exactly `0.5`, `0.4`, `0.5`, `0.5` and all four report "not
unreasonable"; `100000 * beta 2 5` keeps `bs = 2`; `10 claims sev cantor
poisson` keeps `bs = 0.000976562` and the lognorm compound keeps `bs = 0.25`.
