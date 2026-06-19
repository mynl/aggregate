# Plan: firm up multivariate.py → bivariate, for the beta candidate

> **STATUS: READY TO EXECUTE (2026-06-19), rev 3.** The last major block before a
> `1.0.0b1` candidate. All design decisions are made (review 2026-06-18/19);
> §9 is the staged execution plan. No open design questions remain — only the
> tuning called out in §10. Move to `dev/done/` when complete.

---

## 0. Thesis

`multivariate.py` works but is the least finished major class: it predates the
1A/1P bucket-window machinery (W5–W8, shipped a51–a66), so it sizes its 2-D axes
with a crude private path and **aliases badly**. This block:

1. **Renames the truth: multivariate → bivariate** (the code is, and will remain,
   strictly two-axis — §2).
2. **Routes axis sizing through the shipped `_bs_window`, under a 2-D memory
   budget** — fixes the aliasing (§5).
3. **Completes the reporting surface** (`bs_*` / `tail_*` / `explain`) and aligns
   vocabulary with the siblings (§4).
4. **Generalises netceded to any view-pair** of {gross, ceded, net} (§3).
5. **Adds the wanted modelling features**: the `t` copula, a shuffle-of-Min
   copula, and **clash parameterisation** of the two component triggers (§7).

The motivating bug (author, 2026-06-18):

```python
p = build('''mv DISCRETE.2
    200 claims
    agg A dfreq[1] ssev uniform - .3
    agg B dfreq[1] ssev uniform - .5
    mixed gamma .5
''')
```

→ grid 512×512, **tail deficit 0.544** (axis-A mass 0.456): wrap-around aliasing.
Root cause: `MultivariateAggregate._size_axis` (`:567`) caps at `log2 ≤ 11` and
ignores the support-aware / signed two-sided / tail-intelligent sizing the
univariate `_bs_window` now does (a signed `ssev` axis especially needs the
two-sided window).

---

## 1. Backlog folded in (from `dev/TODO.md`)

| ID | Item | Phase | Disposition here |
|----|------|:-----:|------------------|
| **M1** | MV stages 2–5 (`t`; reporting; ≥3-variate `rfftn`; `MultivariatePortfolio` / `netceded` DecL) | `[B]` | **`t` → dropped** (§7.1); netceded DecL → **in** (§3); reporting → **in** (§4); **≥3-variate `rfftn` → killed** (§2); `MultivariatePortfolio` → defer |
| **M2** | Multivariate punch-up — axis sizing / coverage reconciliation | `[B]` | **absorbed into §5; plan file deleted** (one sizing notion) |
| **W2** | Window bounds for bivariate/multivariate per-axis sizing | `[B]` | the per-axis windowing primitive — §5 |
| W1 | Support-aware window bounds (`fz.support()`) | `[A]` | inherited from whatever the 1-D path does; not owned here |

The former `dev/plan-multivariate-punchup.md` (M2 starter) is **absorbed into §5
and deleted** (2026-06-19): its premise ("two private sizing paths that disagree")
is resolved by deleting both and routing every axis through `balanced_window` on
the realized marginals. Its `[multivariate]` config-knob table (`default_log2` /
`cap_log2` / `quantile`) evaporates; `window_nines` already shipped.

---

## 2. Rename multivariate → bivariate  *(DECIDED)*

**Decision (author).** No plans beyond bivariate. ≥3-variate isn't practical
natively; the real path for ≥3 is **independent components → Iman–Conover →
read back as a sample → switcheroo**, not an `rfftn` shared-frequency convolution.
Bivariate already buys the two wins worth having: the **netceded** decomposition
and **two lines with clash** (§7).

**So:**
- Class `MultivariateAggregate` → **`BivariateAggregate`**. The result container
  `BivariateDistribution` keeps its name (it *is* bivariate; no clash).
- DecL keyword `multivariate` / `mv` → **`bivariate` / `bv`**. **Drop `mv` /
  `multivariate` outright — no deprecation cycle** (author: we're pre-beta;
  rename clean). Sweep `test_decl.agg`, docs, the cheat sheet in the same change.
- **Kill Stage 4** (`rfftn`, ≥3-variate shared frequency) from the roadmap; the
  class stays hardcoded 2-axis. Replace the deferred-stage note in the docstring
  with the "for ≥3 use Iman–Conover + switcheroo" pointer.
- `info` / `__repr__` say "bivariate", not "MultivariateAggregate".

Churn: class rename (tests, `_factory` `kind=='mvagg'` → `'bvagg'` or keep the
internal kind string and only rename the public class), grammar terminal + the
`ID` exclusion list, `test_decl.agg`, docs, the DecL + (new) Bivariate cheat
sheet. Do the rename **once, late** (§9) so it lands after the surface settles.

---

## 3. netceded → any view-pair of {gross, ceded, net}  *(DIRECTION DECIDED)*

**Finding.** Today `netceded agg_out` (decl.lark:114) is a clean **prefix on any
existing agg with occ reinsurance** — anonymous (reuses the agg's identity),
builds the (ceded, net) comonotone joint only. This is *different in spirit from*
`mv NAME …` (which needs its own name + builds components).

**Decision (author).** Keep the clean prefix-on-existing-agg form — do **not**
move it under `mv NAME …` (that would force a new name and stop you repurposing
an existing agg). Add three sibling keywords, one per view-pair, with
**keyword reading order = (x-axis, y-axis)**:

| keyword | x-axis (axis0) | y-axis (axis1) | maps | `reins_density_df` cols |
|---|---|---|---|---|
| `netceded` *(exists)* | net | ceded | `occ_netter` × `occ_ceder` | `p_agg_net_occ`, `p_agg_ceded_occ` |
| `grossceded` *(new)* | gross | ceded | identity × `occ_ceder` | `p_agg_gross`, `p_agg_ceded_occ` |
| `grossnet` *(new)* | gross | net | identity × `occ_netter` | `p_agg_gross`, `p_agg_net_occ` |

Rationale: the cession map is comonotone on `c + n = g`, so any *two* of the
three determine the third — only three unordered pairs exist, named by one
keyword each. "Can't do all three" is exactly this pick-2-of-3 (2 axes). A "few
more terms for Lark" but straightforward. **Convention:** the keyword names the
pair x-then-y, so `netceded` → x = net, y = ceded; gross leads when present.
(This pins today's `netceded` to x = net, y = ceded — verify against the current
Ceded=axis0 labelling and flip if needed; this is the firm-up moment.) All three
views are **occurrence** reinsurance; aggregate-reins columns (`p_agg_subject` /
`p_agg_ceded` / `p_agg_net`) are out of scope.

Implementation: `build_netceded_joint` already scatters gross mass onto two
maps; parameterise *which two of the three* maps it uses (gross = identity, which
also means the gross axis can skip the scatter — it's already on the gross grid).
`_netceded_theory` reads the matching two of the three `reins_stats_df` views.
Axis labels, `info`, `describe`, `density_df`, `plot` titles follow the chosen
pair instead of hardcoded "Ceded"/"Net". `Aggregate.occ_bivariate()` grows a
`views=('net','ceded')` parameter passing through the same way.

Back-compat: `netceded` keeps its current meaning and the anonymous form.

---

## 4. Reporting consistency — `info` / `describe` / `stats_df` / `density_df` /
`plot` + the `bs_*` / `tail_*` family

**Correction to rev 1.** `Aggregate.describe`, `density_df`, `info` are all
`@property` (distributions.py:6316 / 2622 / 4398); `BivariateAggregate` already
exposes these as properties — **there is no method-vs-property mismatch.** Dropped.

**But (author):** there is real work *rewriting the existing methods*, not just
adding new ones. MV's `info` and `describe` were hand-rolled and **do not look
like** the Agg/Port ones. The work:

1. **`info` — rebuild to the shared convention.** Agg/Port/Distortion `info` use
   `aggregate.constants.info_row(label, value)` + `INFO_NA` for not-yet-available
   values, with a **fixed row catalogue** documented in `dev/info-strings.rst`
   (same rows, same order, every time). MV's `info` is a bespoke f-string blob
   instead. Rebuild it on `info_row` / `INFO_NA`, mirroring the Port row order
   (name, value_type, object count, bs, log2, padding, x_min/x_max per axis,
   then the tail/bounded/last-update/id footer), with the bivariate-specific
   rows (the two axes' windows, the copula + realised correlation, the deficit).
   **Read `Aggregate.info` / `Portfolio.info` (distributions.py:4398,
   portfolio.py:1130) and `dev/info-strings.rst` first** — they are the template.
   Add a bivariate section to `dev/info-strings.rst`.
2. **`describe` — decide the content.** Settle what rows/columns the bivariate
   `describe` carries so it reads like the 1-D one: per-component moment block
   (the daily-driver `EX`/`CV`/`Sk` shape Agg uses) + the joint footer (corr,
   copula tau). Align the `stats_df` basis labels (`theoretical`/`empirical`) and
   `describe` columns to the 1-D wording. This is a deliberate design pass on the
   *display*, not just a rename.
3. **Add the bucket/tail surfaces** the siblings got in a60–a65 and MV lacks:
   - `bs_window_df` / `bs_description` / `bs_explanation` — per axis (a dict keyed
     by line name, or the two axes' frames stacked with an axis level).
   - `tail_df` / `tail_description` — per axis.
4. **Add `explain`** (validation) whose headline check is the showpiece
   invariant: *each marginal reproduces the standalone aggregate*.
5. **`plot`** — already a 2-panel contour honouring `FIG_W/H`; once §5 lands the
   right panel uses the honest windowed grid (today it can render the aliased one).

`density_df` stays a 2-D matrix frame (intentionally different from the tall 1-D
frame) — document the difference rather than force-match.

---

## 5. The bucketing fix — *measure*, don't guess  *(DESIGN DECIDED)*

**Finding.** Two private sizing paths, neither aware of the shipped machinery:
copula `_size_axis` (`:567`, moment window, `cap_log2=11`) and netceded module
`size_axis` (`:73`, empirical quantile, `cap_log2=14`). Both **guess** the
window from moments/quantiles up front, the way a 1-D `Aggregate` is forced to.
Hence the 54% deficit on the signed-`ssev` book.

### 5.1 The reframe — pre-calc vs post-calc sizing (author, the key idea)

An **`Aggregate` must size pre-calc**: it picks `(bs, log2, x_min)` *before* the
FFT, guessing the answer's extent from moments (that's what `_bs_window` does, as
well as a guess can be done). A **`bv` is in a privileged position**: it can run
the underlying aggregate(s) **first**, look at the *realized* marginals, and
**focus** the 2-D grid on exactly the support that matters. The bv never guesses
— it **measures**. This is the unifying principle for *both* regimes.

So the sizing question collapses to **window selection on a realized pmf**, for
which we add one primitive:

```
balanced_window(ser, prob_budget) -> (lo, hi)        # ser: index=xs, values=ps
```

Given a realized marginal pmf and a tail probability `p`, return the window
**`[q(p/2), q(1 − p/2)]`** (the two quantiles, snapped to `bs`) — i.e. trim `p/2`
of the mass off **each** tail, keeping `1 − p` centred. **Balanced** = equal tail
probability each side (DECIDED), so a signed P&L axis stays centred on its mass;
for a skewed axis this trims more *value* off the heavy side but equal
*probability*. This is the post-calc analogue of the 1A per-edge coverage
(`estimate_agg_window` `p_lo`/`p_hi`, `window_nines_trim`) run on an actual
density rather than moments — **extract/share that machinery, don't re-derive**.
The argument is the discarded tail mass `p` (small, e.g. `1e-6`); `1 − p` is the
mass kept.

### 5.2 Two regimes (differ only in the bs constraint)

- **Genuine bv severity (independence / copula).** `S` is built by diff/diff
  (discrete-Sklar rectangle) — the bv does all the work, and the two severities
  are independent, so **each axis is free to take its own `bs`** and sizes like a
  standalone agg.
- **Reinsurance (netceded).** The severity lives on the comonotone curve
  `(c(g), n(g))`, **sampled at the gross `bs`** (no choice). `c` and `n` *can* be
  rebucketed to different `bs` (the linear scatter already does this, preserving
  the first moment), but **for 1.0.0 we keep one `bs`** (the gross `bs`) across
  the axes. So netceded sizing is *purely window selection* — and we already know
  the windows, because **one gross-agg pass yields all three occurrence views at
  the common gross `bs`** in `reins_density_df`: `p_agg_gross` (gross),
  `p_agg_ceded_occ` (ceded), `p_agg_net_occ` (net). `balanced_window` runs on the
  two columns the chosen view-pair selects. The other three columns
  (`p_agg_subject` / `p_agg_ceded` / `p_agg_net`) are **aggregate** reinsurance —
  out of scope here (occurrence only). Custom per-axis `bs` is a post-1.0 option.

### 5.3 The budget falls out (author) — it isn't searched

The total budget is a **user input**, exactly like `log2` for the univariate
build: a total-`log2` on `update` (default **20**, ≈ 2²⁰ cells), raise/lower at
will (memory is quadratic, so up only a little). The **split** is then a
*measurement*, not a probe:

1. Run each marginal aggregate (cheap, standalone).
2. `balanced_window` on each realized marginal at the prob budget → each axis's
   required support → each axis's `log2` need. **The split `log2_A + log2_B` just
   falls out of "how much space each marginal's support needs."**
3. If the sum **fits** the total budget: done (possibly with headroom).
4. If it **exceeds** the budget: tighten the prob budget (focus harder — the
   `balanced_window` lever) until it fits; copula may *also* coarsen an axis `bs`;
   netceded can only tighten/clip (its `bs` is pinned to gross) → **clip + warn**,
   recorded in `bs_window_df`'s `clipped` column.

This replaces the rev-2 "probe 10+10 / 9+11 / 11+9 and pick the best" search: we
*compute* the split from measured support instead. (The probe is the fallback
mental model only if the measured split is ambiguous; expect minor tuning on real
books, but the mechanism is measurement, not search.)

### 5.4 `Aggregate.focus(p)` — IN for v1 (author, DECIDED)

The same primitive applied *after* an ordinary `Aggregate.update` re-windows a
computed aggregate to a tighter tail level — `agg.focus(p)` re-slices the density
to `[q(p/2), q(1 − p/2)]`. **Ship it in v1**, but keep it **boringly simple — no
YELLs**: a thin public method over `balanced_window`, re-slicing the existing
`density_df` / grid (no recompute), with a clear docstring; nothing clever. It's
useful beyond the bv (tighten any computed agg's display window) and falls out
for free once `balanced_window` exists.

### 5.5 Cleanup + validation

Delete `_size_axis` and module `size_axis`; route both regimes through
`balanced_window` + the run-marginals-first flow; expose per-axis `bs_window_df` /
`tail_df` (§4). **Validation:** re-run the bug book, assert per-axis deficit at
the 1-D tolerance and each marginal `(mean, cv)` matches the standalone aggregate.

---

## 6. (merged into §5)

The rev-2 "memory-cap / budget-allocation" content now lives in §5.3 (the budget
is a `update(log2=)` input; the split falls out of measured support). Kept as a
numbering placeholder so §7.x references below stay stable.

---

## 7. Modelling features to add

### 7.1 `t` copula  *(DROPPED — author)*
Considered and **cut.** Rationale (author): the bivariate-t CDF is flaky/slow,
**and** it's a footgun — a study found insurance data is "surprisingly normal-
copula"; offering `t` mostly tempts users to pick 2 dof and do something
statistically silly. The existing **normal** copula covers the realistic case;
gumbel/clayton cover deliberate tail dependence. No `t`.

### 7.2 Shuffle-of-Min copula  *(IN — author interest; starter impl in hand)*
A **shuffle of Min** (shuffle of the comonotone copula M): not continuous, but
academically interesting and dense — **any copula can be approximated by a
shuffle-of-Min**. Discrete by nature, which suits the discrete-Sklar rectangle
machinery.

**Decision (author): no DecL clause — programmatic only.** The author has a
working starter `ShuffleOfMin` (perm + optional flip; `cdf` / `S` / `sample`) —
see Appendix A. Its `cdf(u, v)` is **exactly** the interface the discrete-Sklar
builder consumes (`Copula.rectangle_pmf` differences a copula CDF on the
breakpoint grid). Integration is a thin `Copula` subclass: register `kind =
'shuffle'`, hold a `ShuffleOfMin`, expose `C(u, v) = som.cdf(u, v)` and a `tau()`
(derive from the perm/flip, or estimate). Constructed in Python and handed to
`BivariateAggregate(copula=…)`; no grammar work. OPEN 7a resolved.

### 7.3 Clash parameterisation of the two triggers  *(IN — marquee win; math in hand)*
Today each component is hand-written `dfreq [0 1] [p0 p1]` (the per-event
Bernoulli trigger). **Add: derive the two triggers (and the shared event count)
from interpretable claim counts** — `na` (A only), `nb` (B only), `nc` (clash:
both A and B). The author's solver (Appendix B, `solve_clash_model`) closes the
math: it is the **independent-trigger 2×2 table**. The missing cell is
`n0 = na·nb / nc` (the independence constraint `nc·n0 = na·nb`), so

```
n  = na + nb + nc + n0          # derived shared event count
pa = (na + nc) / n              # P(A triggers this event)  → dfreq [0 1] [1-pa pa]
pb = (nb + nc) / n              # P(B triggers this event)  → dfreq [0 1] [1-pb pb]
```

The dependence comes entirely from the **shared frequency + independent
per-event triggers** (the copula is **independent**); clash is the expected
count of joint-trigger events `nc`. This is the natural baseline cat-clash model
and is clean to build: the clash statement supplies `(na, nb, nc)`, the solver
derives `n` / `pa` / `pb`, and the two components become the independent-copula
Bernoulli factories — no hand-tuned probabilities.

Key consequence: a clash bivariate **derives** its shared event count from
`(na, nb, nc)` rather than taking an explicit `<count> claims` exposure — so it's
nearly its own statement shape (or a `clash` clause that *replaces* the exposure).

**DecL surface (author, DECIDED).** A dedicated `clash` statement:

```
clash NAME na nb nc claims <A limit + sev clause> <B limit + sev clause> <freq clause>
```

Exposure is **only** via the three `claims` counts `(na, nb, nc)` — no other
exposure form on a clash statement. The solver (App. B) turns them into the
shared event count `n` and the two triggers `pa` / `pb`; the two components are
built from the given limit/severity clauses wrapped in the solved
`dfreq [0 1] [1-p p]` factories, coupled by the **independent** copula; the
`<freq clause>` is the shared outer frequency (e.g. `mixed gamma .2` for
common-shock on top). v1 is **independent triggers only** (matches the solver);
a non-independent copula layered on the clash triggers is a later extension, not
v1. New Lark terminal `CLASH.2`; added to the `ID` exclusion list.

---

## 8. The validation invariant (throughout)

Every change preserves: **each marginal reproduces the standalone aggregate**
(means exact; cv at the matched grid). It is the `explain` headline (§4) and the
acceptance test for §3 (view-pairs), §5 (sizing), §7 (new copulas/clash).

---

## 9. Staged execution plan

Seven stages, each a self-contained version bump with its own tests; ordered so
every stage rests on a settled foundation. `tests/test_multivariate.py` (plus
`test_decl.agg` MV section) grows with each. Run `uv run pytest`
(`UV_LINK_MODE=copy`) per stage; no docs build in-loop. Knowledge-freeze
(`scripts/freeze_knowledge.py`) before/after each numeric stage.

**Version numbers `[~a70]…` are indicative, not pinned.** Repo is on `a68`; an
unrelated DecL fix takes `a69`, so MV work starts ~`a70` and each stage takes the
next number at execution time (other interleaving work may shift them further).
**Execution cadence:** one stage per iteration, with a **review + commit between
each** (each stage is a clean version-bump boundary).

### Stage MV-1 — `balanced_window` primitive + `Aggregate.focus` `[~a70]`
*Foundation; pure 1-D, no bv yet.*
- Add `balanced_window(ser, p)` → `[q(p/2), q(1−p/2)]` snapped to `bs`
  (utilities or distributions). Factor it out of / share with the 1A per-edge
  coverage (`estimate_agg_window` `p_lo`/`p_hi`); do **not** duplicate.
- Add `Aggregate.focus(p)` — thin public re-slicer over the computed
  `density_df`/grid (no recompute), simple docstring (§5.4).
- **Tests:** window holds `1−p`, equal-tail balance, snapping; `focus` round-trips
  mass to tolerance; signed/skewed margin stays centred.
- **DoD:** primitive + `focus` shipped; 1-D only; no MV change yet.

### Stage MV-2 — measure-don't-guess axis sizing (the bug fix) `[~a71]`
*The core correctness stage; copula mode.*
- Delete `MultivariateAggregate._size_axis`. New flow in `update`: run each inner
  marginal aggregate (cheap), `balanced_window` on each realized marginal,
  derive per-axis `log2` need, split under the `update(log2=…)` total budget
  (default 20; copula axes free `bs`); over budget → tighten `p` / coarsen `bs`.
- **Tests:** the bug book (`mv DISCRETE.2 … ssev …`) deficit < 1e-6 per axis;
  each marginal `(mean, cv)` matches the standalone aggregate at the 1-D
  tolerance; signed two-sided window verified; budget input honoured + override.
- **DoD:** the 54%-deficit book is clean; copula sizing is measured, not guessed.

### Stage MV-3 — netceded one-bs window selection `[~a72]`
*Netceded sizing onto the same primitive.*
- Delete module `size_axis`. Size netceded axes via `balanced_window` on the
  chosen `reins_density_df` view columns at the gross `bs` (window-only; `bs`
  pinned). Clip + warn when gross-`bs` × budget can't cover; record in
  `bs_window_df` `clipped`.
- **Tests:** netceded marginals match `reins_density_df['p_agg_*_occ']` (rebucketed)
  to tolerance; clip path warns + reports honestly; comonotone identity holds.
- **DoD:** both regimes route through `balanced_window`; both private sizers gone.

### Stage MV-4 — reporting surface `[~a73]`
*Mechanical now that grids are honest (§4).*
- Rebuild `info` on `info_row`/`INFO_NA` to the shared catalogue (template:
  `Aggregate.info`, `Portfolio.info`, `dev/info-strings.rst`); add a bivariate
  section to `dev/info-strings.rst`. Settle `describe` content (per-component
  `EX`/`CV`/`Sk` + joint footer); reconcile `stats_df` basis labels. Add per-axis
  `bs_window_df`/`bs_description`/`bs_explanation`, `tail_df`/`tail_description`,
  and `explain` (headline: marginal-reproduces-standalone).
- **Tests:** `info` row catalogue present + ordered; `explain` flags a deliberately
  under-budgeted book; `bs_window_df`/`tail_df` shape per axis.
- **DoD:** the bv reporting reads like Agg/Port.

### Stage MV-5 — netceded view-pairs `[~a74]`
*Grammar + the two new keywords (§3).*
- Add `grossceded` / `grossnet` terminals (priority 2, lookahead) + `ID`
  exclusion; parameterise `build_netceded_joint` by view-pair (gross = identity,
  skip scatter); `_netceded_theory` reads the matching `reins_stats_df` views;
  axis labels / `info` / `describe` / `density_df` / `plot` follow the pair.
  `Aggregate.occ_bivariate(views=…)`. Fix today's `netceded` axis order to the
  (x=net, y=ceded) convention.
- **Tests:** each keyword parses + builds; marginals match the named views;
  `occ_bivariate(views=…)` parity; `test_decl.agg` lines added.
- **DoD:** all three occurrence view-pairs available, named, labelled.

### Stage MV-6 — copulas: shuffle-of-Min + clash `[~a75]`
*The modelling wins (§7.2, §7.3).*
- **Shuffle-of-Min:** `Copula` subclass `kind='shuffle'` wrapping a
  `ShuffleOfMin` (App. A — improve/replace as needed), `C(u,v)=som.cdf`, `tau()`;
  programmatic only (no DecL). Tests: independence/comonotone limits, marginals
  reproduce, dense-approximation sanity.
- **Clash:** `solve_clash_model` (App. B — improve/replace) + new `clash NAME na
  nb nc claims <A sev> <B sev> <freq>` statement (`CLASH.2` terminal, `ID`
  exclusion, transformer → solved triggers + independent copula + shared freq).
  Tests: solver identities; built clash bv marginals reproduce; `corr(mixed) >
  corr(poisson)` (common shock); parse-only tests.
- **DoD:** shuffle-of-Min usable in Python; `clash` statement builds end-to-end.

### Stage MV-7 — rename multivariate → bivariate `[~a76]`
*Identity churn, last so it lands once over a settled surface (§2).*
- `MultivariateAggregate` → `BivariateAggregate`; keyword `multivariate`/`mv` →
  `bivariate`/`bv` (**dropped outright**, no synonyms); internal kind string;
  `info`/`__repr__` say "bivariate". Sweep `test_decl.agg`, docs, cheat sheets;
  kill the Stage-4 (`rfftn`) note → Iman–Conover + switcheroo pointer.
- **Tests:** full suite green under the new names; no stale `mv`/`multivariate`.
- **DoD:** the public surface tells the truth; this *is* the v1.0 bivariate API.

**Beta gate (author, DECIDED): ALL seven stages block `1.0.0b1`.** Rationale:
**minimal / no API scope creep post-beta, as a matter of philosophy** — land the
whole bivariate surface now so the beta's public API *is* the v1.0 API. MV-1
through MV-4 are correctness + consistency; MV-5/6 are the modelling features;
MV-7 is honesty. This is the genuine "last major block before the beta candidate."

Per the release rules each stage bumps `pyproject.toml` + adds a `CHANGELOG.md`
section; update `dev/TODO.md` (M1/M2/W2) as stages land; move this plan to
`dev/done/` after MV-7.

---

## 10. Open decisions, collected

**No open design decisions remain.** What's left is implementation tuning,
resolved by code + benchmarking, not by a choice:

- **netceded ceded/net window** — `bs` pinned to gross, so tighten `p` / clip +
  warn when it can't cover; record honestly in `bs_window_df` (`clipped`). The
  one numeric case to watch (Stage MV-3).
- **measured split fine-tuning** — the `log2_A + log2_B` allocation falls out of
  the realized supports (§5.3); minor tuning expected on real books (Stage MV-2).

(Resolved: **§5a** "balanced" = equal-tail-prob, `[q(p/2), q(1−p/2)]`; **§5b**
`Aggregate.focus(p)` ships in v1, kept simple — both DECIDED 2026-06-19.)

(Resolved across revs: **`mv`/`multivariate` dropped outright**, rename →
bivariate, no deprecation cycle (§2); ≥3-variate killed (Iman–Conover +
switcheroo instead); netceded = prefix form + three keywords `netceded` /
`grossceded` / `grossnet`, **keyword order = (x, y) axis** (§3); `info` rebuilt
to the `info_row` catalogue + `describe` content settled, not just a rename (§4);
sizing reframed — **a bv measures, it doesn't guess**: run the marginals first,
then `balanced_window` on each realized pmf → the `log2_A + log2_B` split **falls
out** of measured support; total budget is a `update(log2=)` **input** (default
20); copula axes free `bs`, netceded pinned to gross `bs` (window-only) (§5);
**`t` copula dropped** (flaky + footgun; normal copula suffices) — scope is
**normal / gumbel / clayton / fgm + shuffle-of-Min** (no DecL, App. A) + clash
(independent-trigger solver, `clash NAME na nb nc claims …` statement, App. B)
(§7); **beta gate = everything blocks `1.0.0b1`**, no post-beta API scope
creep (§9).)

---

## 11. Risks / watch

- **Square-law memory** — the 2²⁰ budget (`update(log2=)` input) is load-bearing;
  the split is *measured* from the realized marginals (§5.3), but benchmark the
  fit-vs-budget fallback (tighten prob budget / coarsen bs / clip) on real books.
- **`balanced_window` on signed/skewed axes** — equal-tail-prob trim keeps a P&L
  axis centred (the `ssev` bug); verify it behaves on a heavily skewed margin and
  reuses the 1A per-edge coverage rather than re-deriving it.
- **netceded `bs` pinning** — gross-`bs` coupling means netceded can only tighten
  the *window*, not coarsen `bs`; clip+warn must be honest in `bs_window_df`. The
  main numeric case to get right.
- **clash** — the math is settled (App. B); the residual risk is the new `clash`
  statement in the grammar (terminal, `ID` exclusion, parse-only tests).
- **`.focus()` scope** — if shipped (OPEN 5b), it's public API; lock its contract
  pre-beta so it's not revisited after.
- **Marginal-reproduces-standalone** — the invariant gating every change (§8).

---

## 12. Cross-references

- `dev/done/plan-multivariate.md` — the Stage-1 build (copula + netceded).
- ~~`dev/plan-multivariate-punchup.md`~~ — M2 starter, **absorbed into §5 and
  deleted** (2026-06-19).
- `dev/done/plan-univariate-bucket.md` / `plan-bucket-window-2.md` — the 1A/1P
  machinery §5 reuses (`_bs_window` log2-cap, `bs_window_df`, `tail_df`).
- `dev/TODO.md` — M1, M2, W2 (and W1).
- Iman–Conover + switcheroo (the ≥3-variate path that replaces native `rfftn`).

---

## Appendix A — `ShuffleOfMin` starter (author, 2026-06-18)

**Example only — free to improve or replace** (author). Programmatic-only (§7.2).
`cdf(u, v)` plugs straight into `Copula.rectangle_pmf`. To wrap as a `Copula`
subclass: register `kind='shuffle'`, hold an instance, `C(u, v) = som.cdf(u, v)`;
derive/estimate `tau()`.

```python
import numpy as np


class ShuffleOfMin:
    """Shuffle of Min copula on the unit square.

    Built from M(u, v) = min(u, v) by cutting [0, 1] into n equal
    vertical strips, permuting them by `perm`, and optionally flipping
    (reflecting) strips flagged in `flip`. The mass lives on n segments
    of slope +/- 1, the graph of a measure-preserving bijection S.
    """

    def __init__(self, perm, flip=None):
        # perm[i] = destination slot of the i-th source strip
        self.perm = np.asarray(perm, dtype=int)
        self.n = self.perm.size
        # flip[i] = True reflects strip i (slope -1 instead of +1)
        self.flip = (np.zeros(self.n, bool) if flip is None
                     else np.asarray(flip, bool))
        self.w = 1.0 / self.n  # common strip width

        # Source interval on u-axis for each strip: [i*w, (i+1)*w]
        self.u0 = np.arange(self.n) * self.w
        # Destination interval on v-axis, placed by the permutation
        self.v0 = self.perm * self.w

    def S(self, u):
        """Apply the measure-preserving map V = S(U)."""
        u = np.asarray(u, float)
        i = np.clip((u / self.w).astype(int), 0, self.n - 1)  # strip index
        local = u - self.u0[i]                                # offset in strip
        local = np.where(self.flip[i], self.w - local, local) # reflect if flagged
        return self.v0[i] + local

    def cdf(self, u, v):
        """Copula C(u, v) = mass of S-graph inside [0, u] x [0, v]."""
        u = np.asarray(u, float)[..., None]   # broadcast over strips
        v = np.asarray(v, float)[..., None]
        # Source sub-interval of each strip lying left of u
        a = np.clip(u - self.u0, 0.0, self.w)
        # For each strip, the v-extent of its image segment, clipped to [0, v]
        lo = np.where(self.flip, self.v0 + self.w - a, self.v0)
        hi = np.where(self.flip, self.v0 + self.w,     self.v0 + a)
        overlap = np.clip(np.minimum(hi, v) - np.maximum(lo, 0.0), 0.0, None)
        # Only strips actually reached by u contribute their overlap
        contrib = np.where(a > 0, overlap, 0.0)
        return contrib.sum(axis=-1)

    def sample(self, m, rng=None):
        """Exact draws: U uniform, V = S(U)."""
        rng = np.random.default_rng(rng)
        u = rng.random(m)
        return np.column_stack([u, self.S(u)])
```

---

## Appendix B — clash solver (author, 2026-06-18)

**Example only — free to improve or replace** (author). The independent-trigger
2×2 table (§7.3). Given `(na, nb, nc)`, derives the shared event count `n` and the
two per-event trigger probabilities `pa`, `pb` → `dfreq [0 1] [1-pa pa]` /
`[1-pb pb]`, independent copula.

```python
from typing import NamedTuple


class ClashSolution(NamedTuple):
    """Solution for the independent-event clash model."""

    n: float
    na: float
    nb: float
    nc: float
    n0: float
    pa: float
    pb: float


def solve_clash_model(na: float, nb: float, nc: float) -> ClashSolution:
    """Solve for n, pa, and pb from A-only, B-only, and clash counts.

    The observed cells are:
        na = A claim, no B claim
        nb = B claim, no A claim
        nc = A and B claim

    The missing cell is:
        n0 = no A claim, no B claim

    Independence implies:
        nc * n0 = na * nb
    """
    if nc <= 0:
        raise ValueError("nc must be positive for a finite non-degenerate solution.")

    if na < 0 or nb < 0:
        raise ValueError("na and nb must be non-negative.")

    n0 = na * nb / nc
    n = na + nb + nc + n0

    pa = (na + nc) / n
    pb = (nb + nc) / n

    return ClashSolution(n=n, na=na, nb=nb, nc=nc, n0=n0, pa=pa, pb=pb)
```
