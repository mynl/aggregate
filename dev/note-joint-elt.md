---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Note: the joint distribution of two units of an ELT

Status: note, 2026-10-28. Not a plan; no code changed. `[ELT-Joint-Units]`
(the deferred extension of that name in `dev/done/plan-elt.md`) was drafted
as a plan on 2026-10-28 and pended the same day: the need is a visualization
of the bivariate distribution of two units, and the planned DecL route grew
too large for that need. This note keeps what was decided and learned, for
whoever takes it up. The univariate side landed separately as
`[ELT-Shared-Event-Rollup]` (`dev/plan-elt-shared-rollup.md`).

## The model

An ELT aggregate declares one row per event and unit,
`agg X elt unit A [...] elt unit B [...]`. Within one occurrence of event
$j$, the pair $(X_{Uj}, X_{Vj})$ of two named units is coupled by Burke's
one-factor Gaussian copula [@Burke2016; eltq spec section 4A,
`V:/dev/eltq/secondary-uncertainty-mathematical-specification.md`;
@Mildenhall2026a]. Occurrences are a Poisson process, so

$$
(A_U, A_V)=\sum_{k=1}^{N}(X_{U,k}, X_{V,k}),
\qquad N\sim\operatorname{Poisson}(\Lambda_{UV}),
$$

with $\Lambda_{UV}$ the total rate of events exposing $U$ or $V$. An event
exposing only $U$ puts $V$ at zero; an event exposing neither drops out
(Poisson thinning). Each axis's marginal is the unit's own one-row law, so
the annual marginal of $A_U$ is exactly unit $U$'s aggregate built alone.
Under Burke's coupling the pair's joint law does not depend on the other
units' rows, so only the pair's rows need go to eltq.

$A_U+A_V$ is **not** the univariate total, which uses the roll-up
(`[ELT-Shared-Event-Rollup]`). The two agree in the mean only; on the clean
two-unit case of the roll-up plan the roll-up SD of an event total is 45
against Burke's 42.92.

## Decisions taken while it was a plan

Author, 2026-10-28. They bind a future plan unless revisited.

- **Coupling:** Burke's, never the roll-up or Homer's pro rata allocation
  (an ad hoc rule: under it the pair is $(s_UX_R, s_VX_R)$, perfectly
  comonotone within an event, and a unit can receive more than its own
  maximum).
- **Syntax:** `bivariate unit U V <agg_out>` (and `bv unit U V ...`), where
  `<agg_out>` is an inline ELT aggregate or the rename form
  `agg NEW [as "label"] agg.Base`, as for `grossnet`. The bivariate takes the
  inner agg's name; `note`, `hints` and `tags` are lifted as
  `_bv_out_viewpair` (`src/aggregate/parser.py:1391`) lifts them. No new
  reserved word: `unit` is reserved already, and the ID terminal excludes it
  (`decl.lark:1047`), so `BIVARIATE UNIT ...` is token-disjoint from every
  existing `bv_out` alternative.
- **Refusals:** occurrence or aggregate reinsurance on the inner agg (they
  act on the event total; per-unit terms are `[ELT-Unit-Terms]`), a unit the
  table does not contain (listing those it does), $U=V$, a `yelt` body, a body
  that is not a loss table.
- **Large grids** go through the existing massive path.

Proposed, not ruled: a quoted unit name (`bivariate unit "Personal Lines"
Commercial`), because a unit read from a file can be any text
(`_loss_table.py:606`) while a DecL name must be an identifier.

## A working route today, from Python

No library change is needed to see the joint distribution. eltq's
`occurrence_pair_joint` output is exactly what `BivariateAggregate`'s
discrete mode takes; its docstring points there. Checked 2026-10-28 with
aggregate 1.18.2 and eltq 2.14.0, synthetic rows:

```python
import numpy as np, eltq
from aggregate.bivariate import BivariateAggregate

# event, unit, rate, mean, max, sdi, sdc (ELT6)
rows = [('1', 'A', .10, 30, 300, 20, 10), ('1', 'B', .10, 20, 200, 15, 10),
        ('2', 'A', .05, 80, 500, 40, 30), ('2', 'B', .05, 60, 400, 30, 30),
        ('2', 'C', .05, 9, 90, 3, 3), ('3', 'A', .20, 15, 100, 10, 5),
        ('4', 'B', .15, 25, 150, 10, 10), ('5', 'C', .30, 50, 100, 10, 10)]
ev, un, rate, mean, mx, sdi, sdc = map(np.array, zip(*rows))
k = np.isin(un, ['A', 'B'])                      # the pair's rows only
pmf, xu, xv, meta = eltq.occurrence_pair_joint(
    ev[k].astype(object), un[k].astype(object),
    rate=rate[k].astype(float), mean_loss=mean[k].astype(float),
    sd_indep=sdi[k].astype(float), sd_corr=sdc[k].astype(float),
    max_loss=mx[k].astype(float),
    units=('A', 'B'), bucket_size=5.0, coupling='burke')
bv = BivariateAggregate('AB', mode='discrete', dbv_xs=xu, dbv_ys=xv,
                        dbv_S=pmf / meta.mass, exp_en=meta.mass,
                        freq_name='poisson')
bv.update(log2=(8, 8), bs=(5.0, 5.0))
bv.plot(log=True)
```

Observed: the pair joint was `(101, 81)` with mass 0.5 and occurrence
correlation 0.41, in 0.01 s; `validation_df` filled; `plot(log=True)` showed
the shared-event dependence (the linear-scale default is one flat color,
because the atom at zero, $e^{-0.5}=0.61$, swamps it). The axes are labeled
`X` and `Y` (hard-coded in `_init_discrete`, `bivariate.py:1473`). The grid
means were 9.975 and 8.733 against the model's 10.0 and 8.75, about 0.25%
low at a bucket of 5 against maxima up to 500; not investigated (candidates:
eltq's conditional bucketing, or the factor rule's truncation of the common
factor's tails).

## What a DecL route would take

From the pended plan, with file references as of 1.18.2.

- **Underwriter.** `Underwriter._factory` (`underwriter.py:1653`) refuses
  any non-`agg` kind whose spec contains a loss table (`carries_loss_table`,
  `_loss_table.py:224`), so every bivariate over an ELT is refused today. A
  route intercepts a `bvagg` of a new mode before that refusal.
- **Rows.** `_lower_elt` (`_loss_table.py:744`) validates the sources into
  local arrays; only ELT6 keeps them (on `EltModel`). Factor the reading and
  validation into a function returning the combined rows; ELT3, ELT4 and
  `no2u` reach eltq with zero SDs (ELT3 maximum set to the mean), so one
  path serves every form.
- **Construction without a grid**: $\Lambda_{UV}$, each axis's closed-form
  moments (`eltq.moments_arrays` on the unit's rows), the pair total's
  moments (`eltq.multiunit_moments(coupling='burke')` on the pair's rows).
- **Update**: one common bucket $h$ (eltq takes a scalar), proposed as the
  larger of the two standalone unit buckets from the existing ELT sizer;
  annual axes long enough to hold the occurrence support, so nothing is
  truncated; then the discrete machinery, in memory or with `store_dir`.
- **Discrete mode gaps**: axis names hard-coded `X`, `Y`; per-axis `log2`
  not capped by the total budget in discrete mode (`_size_axes`,
  `bivariate.py:1664`); no guard in `_form_severity_matrix` (:2099) against
  an atom index beyond the grid (an `IndexError` today).
- **Writer**: `_render_bvagg` (`decl_writer.py:1242`) keys off `mode`.
- **Ambiguity test**: `_NAME_RE` in `tests/test_grammar_ambiguity.py:53`
  would take `unit` as the statement name of `bv unit ...`.

## Limits, in the order they bind

1. **eltq's pair array is dense, in memory, on full support.**
   `_occurrence_run` passes `grid_length=None`, so an axis spans
   $\lceil\max_jM_{Uj}/h\rceil+1$ buckets: $2^{14}$ per axis is 2 GiB,
   $2^{16}$ is 32 GiB, before aggregate receives it. The massive path
   handles the annual FFT, not this array. One unit with a large exposed
   value and a small mean forces a coarse $h$ or an impossible array.
   Upstream asks: a windowed pair joint with an overflow policy
   (`[Eltq-Windowed-Pair-Joint]`) and per-axis buckets
   (`[Eltq-Pair-Axis-Buckets]`).
2. **eltq's run time at realistic size is unmeasured.** Each shared event
   costs $Q=32$ outer products of conditional unit laws, and eltq's windowed
   multi-unit route measured one to three orders of magnitude over budget on
   2026-10-27. Measure first: synthetic tables of $10^3$ to $10^5$ events,
   two and four units, axes of $2^{10}$ to $2^{14}$.
3. **The annual grid.** In memory up to `BivariateSettings.total_log2 = 20`
   by default (`config.py:282`). Massive is opt-in by `store_dir` only
   (`bivariate.py:1939`); the kernel takes the occurrence severity already
   in memory and streams it, with the annual grid on disk; $2^{13}$ to
   $2^{16}$ per axis is its practical range (`dev/done/plan-bv.md:213`). An
   automatic switch to massive would need a default directory and would
   write tens of GB unasked; refusing with the `store_dir=` call that would
   succeed is simpler.

## References

::: {#refs}
:::
