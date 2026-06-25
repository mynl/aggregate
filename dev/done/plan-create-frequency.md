# Plan — `create_frequency()` on Aggregate and Portfolio

> **Status: EXECUTED (1.0.0a107).** `Aggregate.create_frequency()` /
> `Portfolio.create_frequency()` shipped, sharing `Aggregate._count_program`
> (static spec→DecL helper). Scope widened during execution: the spec edit
> strips **layers and both occurrence + aggregate reinsurance** as well as
> severity — the original "two nodes" (severity + exposure) would have corrupted
> the count whenever a layer/reins clause was present (`occurrence net of
> 50 xs 0` nets a `dsev [1]` unit to 0). Portfolio re-parses its *own* program
> for the unit specs (units don't retain individual programs) and pairs each
> with its object's resolved `n`. Verified across poisson / mixed gamma /
> delaporte / sichel / zm / dfreq and exposure-derived counts; `zt` parent
> builds still fail upstream (tracked B4, unrelated). No dependency on
> `plan-summary-tail-tables.md` (that plan leaves Freq percentiles blank; this
> one is how a user materializes the count distribution when they want it).

---

## Why

The engine carries frequency only as a **PGF** (applied in the Fourier domain) —
it never materializes the claim-count distribution. So there is no `q` / `tvar` /
`cdf` / percentiles for the count, for any family. Users sometimes want exactly
that ("what's the 1-in-100 claim count?", "plot the count distribution"). The
cheapest, most honest way to give it: build the marginal as a **real first-class
object** they already know how to use.

## What

`Aggregate.create_frequency()` returns a new `Aggregate` whose aggregate
distribution *is* this object's claim-count distribution — built via the
`dsev [1]` trick: same frequency clause, severity a point mass at 1. N claims each
of size 1 sum to N, so the resulting `agg_density[k] = P(N = k)` and every
inherited method (`q`, `tvar`, `cdf`, `sf`, `pmf`, `plot`, `density_df`,
`summary_df`, `tail_df`, …) just works on the count.

```python
fa = agg.create_frequency()     # an Aggregate that IS the count distribution
fa.q([0.01, 0.5, 0.99])         # count percentiles
fa.tvar(0.99)                   # tail count
fa.plot()                       # the count distribution, plotted
```

`Portfolio.create_frequency()` returns a **Portfolio** built from one
`create_frequency()` unit per constituent aggregate (each unit is that unit's
count distribution). The portfolio total is then the **total claim count across
all units** — a useful object in its own right.

```python
pf = port.create_frequency()    # Portfolio of per-unit count dists
pf.summary_df                   # per-unit + total count moments/percentiles
```

## How (implementation)

Build a **new object through the normal API** (`build` / the underwriter) from a
DecL program that keeps the frequency and replaces the severity with `dsev [1]`.
Going through the front door means windowing, `bs`, and `log2` selection all
happen automatically — no grid special-casing here (see *Large counts* below).

Rendering that program: we already have the spec→DecL unparser —
`decl_writer.spec_to_decl` / `format_program` (backs `pprogram`). So the clean
path is **spec-level, not string surgery**:

1. recover the **raw transformer spec** (`parsed.spec`) — re-parse `self.program`
   if the object doesn't retain it (note: `spec_to_decl` wants the raw
   transformer spec, *not* the dense `Aggregate._spec` constructor dict);
2. edit two nodes — severity → point mass `dsev [1]`; exposure → the **resolved
   expected count `self.n` as `… claims`** (see subtlety below);
3. `spec_to_decl(edited)` → DecL → `build`.

Name the result e.g. `f'{self.name}.freq'`.

Portfolio: `Portfolio(f'{self.name}.freq', [a.create_frequency() for a in
self.agg_list])`, then `update()`. The portfolio total is the total claim count.

## Subtleties (the "mixtures" worry)

- **Exposure-derived counts — the real trap.** When the count is *derived* from
  severity (`500 loss …`, or a `premium at lr` / limit profile), swapping the
  severity changes `n` (`500 loss` with mean-50 sev = 10 claims; with `dsev [1]`
  = 500 claims). So **collapse exposure to the resolved `self.n claims`** rather
  than re-rendering the original exposure clause. `self.n` is the total expected
  count even for profiles (sum of per-band λ) and mixed frequency.
- **Severity mixtures are a non-issue** — we discard severity wholesale, so a
  `wts` severity mixture simply vanishes. Nothing to render, nothing to get
  wrong.
- **Frequency mixing must be preserved.** The contagion/mixing clause (`mixed
  gamma c`, ZM, etc.) is part of the *frequency*, not the severity — keep it
  verbatim in the rendered program so the count distribution (e.g. neg-binomial)
  is reproduced.

## Large counts — not a caveat

High exposure (`n` ~ `1e7`) just produces a high-mean count distribution.
Building through the API means the **windowing / bucket machinery handles it
automatically** — a non-zero output window, sane `bs`/`log2` — exactly as for any
high-mean aggregate. No cap, no special path.

The returned object is a *snapshot* — rebuild if the parent changes (same
contract as any built object).

## Acceptance

- `Aggregate.create_frequency()` returns a built `Aggregate` whose mean/var/skew
  match `freq_moms(n)` (validation: its `validation_df` is clean).
- `Portfolio.create_frequency()` returns a built `Portfolio`; its `total` mean
  equals the sum of unit `n`s.
- Works across families incl. mixed (mixed gamma/IG, Sichel, Delaporte) and ZM.
- NumPy-style docstrings; version bump + CHANGELOG entry.
