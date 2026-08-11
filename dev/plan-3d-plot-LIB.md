# plan-3d-plot, the LIB half: review and execution notes

Written by the LIB agent, 2026-08-11, against
`aggregate_api/dev/plan-3d-plot.md` (the canonical copy; LIB's is a symlink).
Companion to whatever the API agent writes for its half. Read section 2 first
if you only want to know what the wire actually carries.

**Verdict on the plan: in order, and executed.** Section 5 is the LIB half and
all seven items are done, in two version bumps. Everything below is either a
statement of what shipped or a place where the plan and the code disagree and
the code is deliberate. There are five of those and they are collected in
section 3, none of them large, two of them corrections to numbers the plan
asserts.

---

## 1. What landed

| plan item | shipped in | where |
|---|---|---|
| 5.3 the density clip | `1.0.0a257` `[Joint-Density-Clip]` | `bivariate.py`, `tests/test_bivariate_density_clip.py` |
| 5.1 block coordinates | `1.0.0a258` `[Joint-Surface-Contract]` | `charts/_emit_bivariate.py` |
| 5.2 window before reduction | same | same |
| 5.4 the section 2.2 fields | same | `charts/ir.py` `SurfaceData` |
| 5.5 exact marginals and moments | same | same |
| 5.6 encodings | same | `charts/ir.py` `SurfaceZBlock`, `encode_z_block`, `decode_z_block` |
| 5.7 the test programs | same | `tests/test_chart_surface_pilot.py` fixtures, with the note carried |

Two bumps rather than one because 5.3 is independent of the format work,
touches a different module and is a correction rather than a feature, which is
what the plan's own section 6 says about it.

`CHART_IR_VERSION` stays at **2**, per the plan's section 2.4 phase one. The
array form is still emitted beside the lattice form. Dropping it is phase two
and is the thing that bumps to 3.

### Things the plan worried about that turned out to be free

- **Question 6, the `load_chart_doc` seam.** Already closed: the reader landed
  at `a252`, before this work started, so "the emitter and the reader land
  together" was satisfied by extending the reader in the same commit. The
  nested `SurfaceZBlock` is rebuilt by `_load_surface`, and the round trip is
  asserted over a real surface document, hash **and** object equality.
- **Oversight condition 4, "chartdoc baselines and fixtures regenerate".**
  Not on the LIB side. There are two chartdoc baselines, `agg.png` and
  `distortion.png`, and no bivariate one; no fixture in `tests/data/` mentions
  the surface. `BivariateAggregate.plot()` goes through `plots.plot_bivariate`
  and not through the chart IR, so it draws exactly what it drew before. The
  only picture that moves is the one served through `build_chart_doc`, which
  is the app's, and it moves because it is now windowed and correctly
  labeled.
- **The suite.** 3,964 fast tests pass, and the slow bivariate suites pass.
  No number anywhere else in the library moves: the de-fuzz floor was chosen
  to reproduce the old absolute constant on a normalized joint (see 3.5).

---

## 2. The wire contract as built

This is what `series[].surface` carries now. The API and the SPA should code
against this section, not against the plan's 2.2, where the two disagree.

```
surface: {
  # the older form, still emitted (phase one)
  x: [...], y: [...], z: [[...], ...],

  # the lattices, normative
  x0: float, dx: float, nx: int,
  y0: float, dy: float, ny: int,
  edge: "left",

  # what the grid is a reduction of
  bs: [bs_x, bs_y],          # the fine bucket size per axis
  k:  [kx, ky],              # the block factor; dx == bs[0] * k[0]

  # what was kept
  window: {p: 4.0, x: [lo, hi], y: [lo, hi], kept: 0.9998...},

  # the references, off the whole distribution
  marginals: {x: [nx values], y: [ny values]},
  moments:   {mean: [mx, my]},
  deficit:   float,

  # z again, encoded
  z_block: {dtype: "f32b64", order: "yx", data: "<base64>",
            peak: float, decades: float}   # peak/decades only for u16log12b64
}
```

**Every value is a mass per display cell.** `z`, `z_block` and **both
marginals**. Not a density. The consumer divides by `dx * dy` once, for `z`,
and by `dx` or `dy` for a marginal. This is the plan's decision 1 applied
consistently; see 3.2, where the plan's own wording slips.

**`edge` is `"left"`.** The coordinate is the low edge of the cell it labels,
which spans `[x_i, x_i + dx)`. The vocabulary is `("left", "mid")`; there is
no `"right"`, because a right-edge convention labels a cell with a coordinate
no point in it reaches, which is how the old bias got in.

**`window.x` and `window.y` are the outer edges of the outer cells**, that is
`[x0, x0 + nx * dx]`, not the coordinates of the first and last cells. Under a
left-edge convention those differ by one cell at the top and the outer edge is
the honest box.

**`window.kept` is `z.sum() / density.sum()`**, the share of the mass the grid
*placed* that is inside the box. `deficit` separately reports the mass the
construction never placed. On the four test surfaces the two live 5 to 7
decades apart, so the distinction is bookkeeping rather than arithmetic, but
they are different facts and are reported separately.

**`marginals` are cropped on their own axis only.** `marginals.x` sums to
slightly more than `window.kept`, because it is the object's real marginal
over the x crop and includes the mass sitting outside the *y* crop. That is
the point: a marginal integrated off the boxed joint is the marginal of a
truncated distribution, which is a different and worse curve. Asserted cell by
cell in the suite (`marginals.x >= z.sum(axis=0)` everywhere).

### The emitter's arguments

`build_chart_doc(bv, 'joint_surface', window=..., detail=..., encoding=...)`.
No change was needed in `build_chart_doc`, which already forwards `**options`.

| argument | default | meaning |
|---|---|---|
| `window` | `4.0` | keep `q(10 ** -window)` to `q(1 - 10 ** -window)` of each marginal, measured on the fine lattice **before** the reduction. `0` or `None` keeps the whole grid |
| `detail` | `128` | target cells per axis, honored as a ceiling (one exception, 3.4) |
| `encoding` | `'f32b64'` | `f32b64`, `f64b64`, `u16log12b64`, or `json` |

`display_log2` is **gone**, replaced by `detail`. Nothing in either repo passed
it (the app's `display_log2_for` is the unrelated density-frame binner in
`serializers.py`).

LIB validates: an unknown encoding raises, `detail` below 8 raises. It does not
cap `window` or `detail` above; that is the API's job, and
`AGGAPI_MAX_CHART_DETAIL` is where it belongs.

### The encodings

`SURFACE_DTYPES = ('f32b64', 'f64b64', 'u16log12b64')`. Little-endian, always,
so the bytes do not depend on the machine and the document hash is portable.
Base64 applied last. No in-payload compression.

`u16log12b64` differs from the plan's formula in one way and it matters: **code
0 is reserved for an exact zero.** The decode is

```
value = 0                                                 if code == 0
value = peak * 10 ** ((code - 1) / 65534 * decades - decades)   otherwise
```

The plan's formula spreads all 65,536 codes over the twelve decades, which
gives an exact zero no code of its own and decodes it as `peak * 1e-12`. A
real joint density is 14% to 59% exact zeros, so that is a floor of spurious
mass under most of the grid. Reserving code 0 costs one level out of 65,536:
measured worst relative error on live cells is **2.108e-4**, which is the
2.1e-4 the plan quotes, to four figures.

`encoding='json'` emits **no** `z_block` at all; the plain arrays are the
payload. That is what "the current behavior, kept for one release as a
fallback" means in practice, and it avoids a third copy of the same grid in
one document.

### Measured, on the plan's four surfaces, at the defaults

Reproduce with `dev/prototypes/joint-surface/make-surface-data.py` in the API
repo, or the equivalent four programs in `tests/test_chart_surface_pilot.py`.

| surface | fine grid | display | dx, dy | k | kept | f32 base64 | canonical JSON |
|---|---|---|---|---|---|---|---|
| Clayton | 512 x 2048 | 75 x 78 | 2, 4 | (2, 8) | 0.99980 | 30.5 kB | 160 kB |
| Indep | 64 x 16384 | 31 x 116 | 2, 8 | (1, 2) | 0.99987 | 18.7 kB | 102 kB |
| IndepFreq | 512 x 2048 | 82 x 108 | 32, 40 | (4, 4) | 0.99981 | 46.1 kB | 245 kB |
| IndepSigned | 1024 x 1024 | 66 x 126 | 40, 32 | (8, 4) | 0.99964 | 43.3 kB | 231 kB |

`kept` lands inside the plan's predicted 99.96% to 99.998% on all four.

The canonical JSON column is the **dual emission** cost: roughly two thirds of
it is the nested `z` array that phase two deletes. `u16log12b64` halves the
base64 column.

The zero snap fires on Clayton's y (window opens 13 above an origin the
distribution reaches, window 312 wide, 13 < 5% of 312) and not on its x (opens
at 8.0 against a 7.5 threshold, a genuine near miss), not on either axis of
`IndepSigned`, and not on `Indep`'s x, whose fine lattice was measured up from
48 so there is no zero on it to reach. All four branches of the rule are
exercised by the fixtures.

---

## 3. Where the code and the plan disagree

Five. Two are corrections to numbers the plan states; three are choices the
plan left implicit and I had to make. All are deliberate.

### 3.1 The `z_block` name, and why the encoded block is not called `z`

The plan's 2.3 spells the encoded block `z: {dtype, order, data}`. Phase one
cannot have that, because phase one is defined as additive and `z` is already
the nested array a current consumer reads. Changing what `z` means is exactly
the breaking change phase one exists to avoid.

So the field is **`z_block`**, beside `z`. At phase two, when `x`, `y` and `z`
go, the block can either keep the name or be renamed to `z`; that is a free
choice at a version bump and I have no opinion. Flagging it because it is the
one place where an API or SPA reader written from the plan's text alone would
look for the wrong key.

### 3.2 The marginals are masses, and the plan says densities

Plan 2.2 asks for "the exact marginal **densities** on the display lattice",
while decision 1 says the wire carries mass and the client divides once.
Emitting one field as a density beside a `z` in mass is how the prototype
ended up with a marginal and a conditional differing by a factor of 1024, the
bug the plan's own 4.1 describes. So `marginals` are masses per display cell,
like everything else, and the consumer divides by `dx` (or `dy`) when it wants
a curve to stand beside a density.

One rule for the whole document beats matching the plan's adjective.

### 3.3 Acceptance criterion "every row reads 0.000 buckets" is not reachable, and 5.1 is still fixed

Plan 8.1 says the block-coordinate item is done "when every row reads `0.000
buckets`" in the probe's bias table. It will not, ever, and the criterion
should be restated. Here is why, because the reasoning is the useful part.

A display cell holds `k` atoms of the fine lattice. Labeling it with any
single coordinate throws away their spread, so a mean taken against the
display lattice cannot equal the mean taken against the fine one unless the
label is the block's mass-weighted centroid, which is not a lattice at all and
which plan 2.2.1 rightly forbids. Only the *sign and size* of the residual
change:

| convention | bias | on the four surfaces |
|---|---|---|
| right edge (before) | up to `+1` display bucket | +0.38, +0.44, +0.47, **+0.95** |
| left edge (now) | between `-1` and `0` buckets | -0.006 to -0.45 |

The measured left-edge biases, at the default window: Clayton -0.26 / -0.45,
Indep -0.006 / -0.27, IndepFreq -0.39 / -0.39, IndepSigned -0.44 / -0.37. Part
of each is the window rather than the labeling (the tail outside the box has
real weight), which the probe's table conflates.

**What was actually wrong, and is now right**, is the part that made numbers
wrong rather than approximate:

- `Indep`'s y support was reported as starting at **508** on a distribution
  supported from 0. It now starts at 0.
- the bias was **unbounded in the direction that mattered**, because it grew
  with the block factor and the block factor grows with the axis. On `Indep`'s
  y, 128 fine cells to a block, the reported mean was 508 against a true 23.8.

Proposed replacement for the criterion, which is what the suite asserts:

1. the display grid's first coordinate **is** the fine lattice's first
   coordinate, at three reduction factors (`test_display_grid_starts_where_the_fine_lattice_does`);
2. the display-grid mean sits in `(-1, 0]` display buckets of the fine mean,
   never above it (`test_display_mean_bias_is_bounded_by_one_bucket`);
3. a consumer that wants a mean reads `moments`, which is exact and is in the
   document for exactly this reason.

Point 3 is the real answer and the plan's 5.5 already contains it. It is worth
saying out loud in the SPA half: **do not integrate the picture to get a
mean.** The document carries the mean.

### 3.4 `detail=128` on `Indep`'s y gives 116 cells, not 232, and the floor can beat the ceiling

Plan 3.2 and 8.1 both quote "232 cells" for the windowed `Indep` y axis. 232
is the count of **fine** cells in that window (928 / 4). Reducing them to a
128 target by a power of two gives **116**. The contrast the plan is drawing,
232-ish against 8, survives intact; only the number moves.

The choice behind it: `detail` is honored as a **ceiling**, so `k` is the
smallest power of two with `ceil(span / k) <= detail`. The alternative, the
smallest count not *below* the target, would have given 232 and would make
`AGGAPI_MAX_CHART_DETAIL` mean nothing, since a cap of 256 could then serve
511 cells.

One exception, found by testing rather than by design: **`MIN_CELLS`
outranks `detail`**. Powers of two do not reach every count. A 528-cell window
reaches 33, 17, 9, 5, so `detail=8` has nothing to land on and the choice is 9
cells or 5, of which 5 is a grid with no spacing to interpolate on. The floor
takes it. The overshoot is bounded by `2 * MIN_CELLS - 1 = 15` cells whatever
`detail` was, so it cannot reach a payload budget, and it can only happen at a
target within a factor of two of the floor. The API should still validate
`detail >= 16` if it wants the cap to be exact, or simply report the realized
`nx`, `ny` and `k`, which is what the plan already tells the client to do.

### 3.5 The de-fuzz floor is relative to the mass, not to the peak

Plan 5.3 proposes `eps * density.max()`. I used `1e-15 * abs(density).sum()`.

The plan's diagnosis of the *predicate* is right and is implemented as
written. Its diagnosis of the *threshold* is not. A 2-D FFT accumulates
round-off in proportion to the total it sums, not to the tallest cell it
produces. On a joint normalized to one that is about `eps`, which is where the
absolute `1e-15` has always sat and why it has always worked. Anchoring to the
peak instead would clip at `2e-17` on a peaked grid, keeping dust, and at
`2e-24` on a flat one, keeping everything, and it would move every existing
number in the library for no gain. Anchoring to the sum holds the depth still,
which is what the plan wants, and reproduces today's behavior exactly on a
normalized joint, which is why nothing in the suite moves.

The plan's claim that the same `1e-15` "sits 12.4 decades below the peak on
one grid and 10.7 on another" is true and is a fact about the peak, not about
the noise.

The related tidy the plan asks for is done: both sites go through one helper
that takes the density semantics. `utilities.remove_fuzz` is untouched, since
its two-sidedness is correct for the frames it serves.

---

## 4. For the API and the SPA half

Nothing here needs LIB to move again; it is a list of what to code against.

1. **Read `z_block`, not `z`**, and expect `x0/dx/nx` rather than the arrays.
   Both forms are present for one release. See 3.1 for the name.
2. **Divide by the cell area once**, at the decode, and remember the marginals
   are masses too (3.2). The plan's 4.1 order of operations is right and the
   only change is that step 2 applies to three arrays, not one.
3. **Do not compute a mean off the display grid.** Read `moments.mean`. See
   3.3, and use it as the check the plan asks for in 5.5: if the consumer's
   own arithmetic disagrees with `moments` by more than a display bucket,
   something is wrong on the consumer's side.
4. **Report the realized grid.** `nx`, `ny` and `k` say what was delivered;
   `window.p` says what depth was asked for. A `detail` near the floor can
   come back slightly over (3.4).
5. **The reading strip's precision follows `z_block.dtype`**, which is the
   plan's answer 5, unchanged: seven figures under `f32b64`, four under
   `u16log12b64`, and say the height is quantized when it is.
6. **`window=0` is the whole grid** and is the escape hatch for anyone who
   wants what the old document carried. The default is now 4, so the served
   picture changes the moment the API syncs, whether or not the API passes the
   parameter. That is intended: on `Indep` the old picture spent 99.99% of its
   y axis on an empty tail.
7. **The base64 payload is deterministic**, so ETags behave. Two builds of the
   same object give byte-identical canonical JSON, asserted in the suite.

---

## 5. What LIB has not done, and does not think it should

- **No `CHART_IR_VERSION` bump.** Phase two, when it comes, is a one-line
  deletion in the emitter plus dropping three fields from `SurfaceData`, and
  the bump goes with it. Someone has to decide when the SPA has moved.
- **No `heatmap` work.** Plan 4.1 correctly notes the app has two call sites
  reading `series.surface`. Both are app side. LIB emits one surface panel and
  the second reader is the SPA's problem, which the plan already assigns
  there.
- **No `Portfolio` to bivariate route.** Plan 6 lists it as a prerequisite
  that is not part of the plan and is post 1.0. Still true, still not started.
  The emitter's availability predicate keys on the object type and the
  presence of an in-memory joint, and on nothing else, so nothing here changes
  when it lands (plan answer 3).
- **No massive path.** A disk-backed bivariate still has no surface chart.
  The predicate excludes it, as before, and the plan puts it out of scope.
- **No fixture move.** Plan 5.7 says the four test programs should carry their
  note when they move upstream. Two of them (`Indep`, `IndepSigned`) are now
  LIB fixtures and carry it; `Clayton` and `IndepFreq` add nothing the two do
  not already exercise, so they stay in the prototype rather than costing the
  suite two more bivariate builds. The note about only `Indep` having
  independent marginals is on the `indep` fixture's docstring, where anyone
  writing an independence test will hit it.
