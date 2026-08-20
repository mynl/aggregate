# Plan [PnL-Reinsurance-Pricing]: the technical premium ladder

> **Status: EXECUTED at LIB `1.0.0a306`, 2026-08-20.** All four phases landed
> as planned, with no divergences: L1 `_pnl_layer_ratios`, L2 `_pnl_technical`
> (plus `_pnl_layer_premium`, split out of L2 so the existing-clause resolution
> and the `rate` refusal carry their own docstring), L3 the signature on
> `_program.pnl_program` and both delegating methods including the stale
> `Portfolio` docstring, L4 sixteen cases rather than the nine sketched (the
> extras: the deposit rounding joint, an unusable net ratio, the portfolio
> refusal, and `derive premium` surviving with no cover to pay for). The
> section 3 worked example reproduces byte for byte and digit for digit.
> Corpus programs added to `agg/decl-testers.agg`, DP block, both halves.
>
> One thing the plan did not anticipate, recorded for the next reader: the
> P&L's booked loss is **not** the same float as the engine's `est_m` the
> ladder priced off. They differ by about 2e-10 relative, the P&L meaning its
> own grid distribution over the net density, which is grid arithmetic and
> nothing to do with the pricing. So the identity is asserted against the loss
> the P&L booked, where it holds to 1e-12, and the two are compared separately
> at 1e-9. Asserting the identity against `est_m` would have looked like a
> pricing error at the 1e-8 level and invited a loosened tolerance.
>
> Line anchors LIB `1.0.0a305`, API `1.0.0a112`. Companion
> document: API `dev/plan-pnl-button-punchup.md`, which owns the app half and
> is this plan's first consumer. The two are independent in one direction: this
> lands and is useful alone from a notebook, and the app half degrades to what
> it does today if this never lands.
>
> **v3 replaces v2's quotation form** on the author's ruling of 2026-08-20:
> every ceded premium is written as a **deposit**, a currency amount. v2 chose
> between a `rate` (against the P&L premium) and a `rol` (against the layer
> limit) by the layer's attachment probability. Review found that the `rate`
> form makes the booked premium depend on itself wherever a premium clause is
> already present, and that the probability the switch read means different
> things on the two tiers. The ruling removes both at once: a deposit is
> `share x amount`, so it references nothing, and there is no switch left to
> misread. v2's sections 5 and 9 collapse; the arithmetic of section 2 is
> untouched.
>
> **v2 replaced v1's construction** on the author's ruling of 2026-08-19.
> v1 sized the booked premium straight off a loss ratio and then asked which
> expected loss to use, gross or net. The answer is that the question was
> wrong: build the premium from the bottom up instead, net technical premium
> plus the cost of each cover, then gross up once for expenses.
>
> **Origin**: the app's PnL button writes a P&L over a reinsured engine and the
> library immediately warns about it (`ZeroPremiumCessionWarning`, "price the
> cover to silence this"). The immediate ask is a demo convenience: put
> reasonable numbers in front of the reader, who will then adjust them.

## The design in one paragraph

`pnl_program` learns to build a premium the way a book is actually priced. The
net technical premium is the net expected loss over a **net combined ratio**;
each reinsurance layer's premium is its own expected ceded loss over **its own
combined ratio**; the technical premium is their sum; and the booked premium is
that grossed up once for expenses, `P = TP / (1 - ER)`. Each layer premium is
written into the program as a deposit, a currency amount quoted at 100 percent
placement. The per layer ratios are passed as scalars or as one value per
layer, so the same function that writes a demo today takes real market rates
tomorrow, and the net ratio is the slot a distortion's implied technical
premium ratio fills. Every input is already public on `reins_stats_df`, no new
frame is computed, and the loss structure is untouched.

## 1. What is already in place

Verified on `1.0.0a305`, this box, 2026-08-19 and 2026-08-20.

| Fact | Where |
|---|---|
| Ceded premium resolves as `share x deposit`, `share x rol x limit`, `share x rate x gross_premium` | `underwriter.py:1290`, `_resolve_reins_economics` |
| Every form is quoted at 100 percent placement and scaled down by `share` | same branch, and its docstring says so |
| `deposit` is the only form referencing neither the premium nor the limit | read off the same branch |
| `gross_premium` for the `rate` base is the P&L's stated consideration, resolved before economics see it | `underwriter.py:1477`, `1515` |
| `derive premium` resolves through `derive_consideration(expense_spec, technical, name)`, `(T + F) / (1 - r)`, which does **not** see ceded premium | `_pnl_builders.py:200`, called at `underwriter.py:2627` |
| Per layer, both tiers: `share`, `limit`, `attach`, `pr_attach`, `pr_detach`, `pr_loss`, `lol` | `reins_stats_df.loc['meta']` |
| Per layer expected ceded loss, **share adjusted** | `reins_stats_df.loc[('agg', 'mean')]` |
| `lol` is **not** share adjusted: it is `mean / (share x limit)` | measured: a `50% po 100 xs 100` layer reports mean 88.400 against the full line's 176.800, and `lol` 1.76800 for both |
| The frame is lazy and cached: 105 ms first call on a three layer object, 4 microseconds after | measured |
| `est_m` on an engine declared net of a cession is the **net** expected loss | measured, 742.198761 on the section 3 example |
| A cession with no premium clause books at zero and warns once per side | `underwriter.py:1526` |
| The warning fires for an `Aggregate` engine and not for a `Portfolio` engine's units | measured |
| The resolved per layer economics ride at `pnl.engine._pnl_recipe['econ']`, keys `pc_occ_by_layer` / `pc_agg_by_layer`; `pnl._pnl_recipe` is `None` | measured, the read path for the section 6 test |
| A premium clause on a bare `agg` is stripped from the built object with `IgnoredDecLClauseWarning` but **survives in the program text**, so `_require_program`'s re-parsed spec carries it while `ob.spec` does not | measured |

Nothing here needs a new query. The whole construction is arithmetic on one
frame the object builds on demand.

## 2. The construction

With `E_net` the net expected loss, `E_j` the expected ceded loss of layer `j`
(share adjusted, as the frame reports it), `c_net` the net combined ratio and
`c_j` layer `j`'s:

```
net technical premium   T_net = E_net / c_net          (or the engine's stated premium, section 2.1)
layer premium           Q_j   = E_j   / c_j
technical premium       TP    = T_net + sum_j Q_j
booked premium          P     = TP / (1 - ER)          via derive_consideration, section 4
```

**The identity that makes this the right construction.** The P&L books `P`,
pays expenses `ER x P`, pays ceded premium `sum_j Q_j`, and carries the net
loss, so

```
margin = P - ER x P - sum_j Q_j - E_net = TP - sum_j Q_j - E_net = T_net - E_net
```

The underwriting margin is exactly the net margin, `(1 - c_net) x T_net`, and
it does not move when a reinsurance ratio moves. Raising a layer's price raises
the booked premium by exactly the extra cost and leaves the book's own result
alone, which is what a reader turning these knobs should see and is what v1's
construction got wrong.

Combined ratio is the author's word for the ratio of expected loss to premium
at whatever level of the tower it is applied. On a technical premium there is
no expense load in it, so on the net it is a loss ratio; the name is chosen for
what it becomes when market rates arrive.

### 2.1 When the engine states a premium

An engine declared `1000 premium at 0.65 lr` states its own technical premium
`T`. It wins over the convention, as it does today: `T_net = T` and `c_net` is
unused. Two consequences worth writing down.

- With **no priced cession**, `P = derive_consideration(expense_spec, T)` is
  exactly what the `derive premium` sentinel resolves to at build, so the
  program still says `derive premium` and the linkage is kept. Nothing about
  today's output moves.
- With **priced cessions**, `TP` is `T` plus the ceded premiums, which the
  sentinel cannot know, so the number is written out.

## 3. Worked, and verified

`agg TR 10 claims sev lognorm 100 cv 2 occurrence net of 100 xs 100 poisson
aggregate net of 500 xs 1000`, every combined ratio at 0.90, `ER = 0.25`.
Both layers are placed 100 percent, so `share` is 1 and each deposit is its
layer premium unchanged.

```
E_net 742.198761,  E_occ 176.800443,  E_agg 81.009664
T_net = 742.198761/0.9 = 824.665290
Q_occ = 176.800443/0.9 = 196.444937 -> deposit 196        (rounded, section 6)
Q_agg =  81.009664/0.9 =  90.010738 -> deposit 90.01
TP    = 824.665290 + 196 + 90.01 = 1110.675290
P     = 1110.675290/0.75 = 1480.900387 -> 1481
```

Emitted:

```
pnl TR_PnL
  1481 premium
  less
    agg TR
      10 claims
      sev lognorm 100 cv 2
      occurrence net of
        100 xs 100 deposit 196
      poisson
      aggregate net of
        500 xs 1000 deposit 90.01
  less
    0.25 premium expenses
```

Built, `economic_df` reads net premium 1194.99, net loss 742.198761, expense
370.25, **margin 82.541239**, and the arithmetic
`P x (1 - ER) - sum_j s_j d_j - E_net` predicts 82.541239 exactly. The ideal
`T_net - E_net` is 82.466529; the 0.0747 between them is the rounding of `P`
to whole units and nothing else, since every deposit was rounded **before**
`TP` was formed. No `ZeroPremiumCessionWarning` is raised.

That exactness is the improvement the deposit form buys. v2's rate form left a
residual its test could only bound; here the test asserts equality.

## 4. The signature

```python
def pnl_program(ob, loss_ratio=0.70, expense_ratio=0.25, *,
                net_combined_ratio=None,
                occ_combined_ratio=None,
                agg_combined_ratio=None):
```

`net_combined_ratio` is the switch as well as a value: `None`, the default,
is today's function exactly, so no existing caller's text moves and no
behavior needs a deprecation. A number engages the ladder of section 2. There
is no separate boolean, because a caller who wants the ladder always wants the
cessions priced and a caller who does not, passes nothing.

`occ_combined_ratio` and `agg_combined_ratio` each accept:

- `None`, the default, meaning `net_combined_ratio`, so one number prices the
  whole tower (the author's "default to all the same");
- a scalar, applied to every layer on that tier;
- a sequence, one value per layer of that tier, index aligned with the cession
  list as declared, ascending attachment. A length mismatch is a `ValueError`
  naming both counts, in the house style of `_cession_spec`'s messages.

The sequence form is the point of the exercise. It is how market rates arrive:
one number per layer, read off a quote sheet rather than off a convention.

`loss_ratio` keeps its meaning and its default, and is consulted only when
`net_combined_ratio` is `None`.

**`working_attach` is gone**, with the quotation form it selected. Nothing
replaces it: there is one premium form and no threshold to set.

## 5. The deposit, and the share

Once `Q_j` is known it is written as a deposit on layer `j`:

```
d_j = Q_j / s_j
```

and the resolver returns `share x deposit = s_j x d_j = Q_j` exactly. That
division is the whole of the share algebra, and it is there because the two
sides quote from different ends: the frame's `E_j` is **share adjusted**, the
loss of the fraction actually placed, while the DecL clause is quoted at
**100 percent placement** and scaled down at resolution. Writing `Q_j` itself
would price a half placed layer at half of what it should be.

The 100 percent line deposit is therefore share invariant, which is also what
makes it the number a reader would read off a quote sheet:

```
d_j = E_j / (c_j s_j) = lol_j x y_j / c_j
```

using `lol_j = E_j / (s_j y_j)` from the frame. A `50% po 100 xs 100` layer and
the same layer at 100 percent write the identical deposit, and their resolved
premiums differ by exactly the share. Measured: both write 196, and the half
placed layer resolves to `pc_occ` 98.0. This is the one place an implementation
is likely to go wrong, so a test asserts it directly.

There is no circularity anywhere in this, and unlike v2 that is now true
unconditionally rather than only for the layers the function itself prices: a
deposit references neither `P` nor anything derived from it.

The value goes into the parallel spec lists the writer already reads,
`spec['occ_reins_premium'][j] = ('deposit', d_j)` (`decl_writer.py:604`), so
nothing in the render path changes.

## 6. Rounding, and the round trip

`_fmt_num` renders whatever float it is handed, and an unrounded deposit prints
sixteen digits into a program a reader is meant to keep and edit, the ugliness
`_round_consideration` (`_program.py:710`) already exists to prevent for the
consideration. A deposit is a currency amount of exactly the same kind, so it
rounds through **the same function**: whole units above 100, two decimals at or
below. One rounding rule for every currency figure the program states, which is
one convention fewer than v2 carried.

**Order matters.** Each `d_j` is rounded **first**, `TP` is then formed from the
rounded deposits, and `P` is rounded last. The program is then internally
consistent to within the rounding of `P` alone, and the residual is exactly
predictable rather than merely small.

The contract test builds the emitted program, reads `pc_occ_by_layer` and
`pc_agg_by_layer` off `pnl.engine._pnl_recipe['econ']`, and asserts each equals
`s_j x d_j` exactly, and that the reported margin equals
`P x (1 - ER) - sum_j s_j d_j - E_net` exactly. If `derive_consideration` or the
resolver ever moves, that test fails here rather than the library silently
mispricing.

## 7. Scope

**In**: an `Aggregate` engine, both tiers, any number of layers, partial
placements.

**Out, each for a reason**:

- **A `Portfolio` engine.** Its units can carry cessions, and pricing them
  needs its own ruling. The library does not warn about them today (measured),
  so nothing is left half done by leaving them alone, and `xpnl` refuses a
  portfolio anyway, so the app half never reaches this case. The ladder itself
  extends to a book cleanly when it is wanted: sum the units' cessions.
- **Ceding commission, reinstatements, variable features.** Each has its own
  premium interaction and none is needed to put reasonable numbers in front of
  a reader.
- **A layer that already carries a premium clause** is **left untouched**, the
  rule that keeps this from overwriting terms the author wrote. A `deposit` or
  `rol` clause resolves without reference to `P`, so its premium enters `TP` at
  the resolved amount and the identity of section 2 still holds. A `rate`
  clause does not, and it is the case the v3 ruling was made against; see
  decision D1.

## 8. Phases

**L1. The ratio resolution.** A helper turning
`(None | scalar | sequence, layer count)` into a per layer list, with the
length check and its message. Small, pure, tested alone, and the piece both
tiers share.

**L2. The ladder.** `_pnl_technical(ob, ...)` in `_program.py`, returning
`(P, occ_deposit_list, agg_deposit_list)`: reads one frame, applies section 2,
then section 5, rounding per section 6. Calls `derive_consideration` rather
than recomputing the gross up. Note the seam recorded in section 1: a
pre-existing premium clause is on the **re-parsed** spec `_require_program`
returns, not on `ob.spec`, so the D1 check reads the former. NumPy docstring
carrying the identity of section 2 and the share algebra of section 5 in Notes,
because those are the two parts a reader will otherwise get wrong.

**L3. The signature and the wiring.** The three arguments, the `None` path
proved inert, the spec keys, the `derive premium` rule of section 2.1, and the
docstring. The delegating methods (`_aggregate.py:959`, `_portfolio.py:1100`)
grow the same signature and forward. **While in `_portfolio.py`**: its
`pnl_program` docstring is stale, claiming the emitted text is `less port.NAME`
and that "the grammar has no inline portfolio engine". Both have been false
since `1.0.0a216`; the units are written out inline (measured). Fix it in the
same edit.

**L4. Tests**, in `tests/test_derived_programs.py`:

1. `net_combined_ratio=None` emits exactly what it emits today, for a stated
   premium engine and a sized one, with and without reinsurance, so the default
   is provably inert;
2. the section 3 worked example reproduces, text and numbers;
3. the round trip and the margin identity of section 6, **exactly**, for a
   sized premium and for a stated one;
4. the identity holds under a changed layer ratio: move `occ_combined_ratio`,
   `P` moves by exactly the extra cost, margin does not;
5. a sequence of per layer ratios prices each layer at its own, and a wrong
   length raises;
6. a `50% po` layer writes the same deposit as the full line layer and resolves
   to exactly the share of it, the section 5 invariance;
7. a layer that already carries a `deposit` or `rol` clause keeps it, and its
   resolved premium still enters `TP`;
8. a layer carrying a `rate` clause raises, per D1;
9. no `ZeroPremiumCessionWarning` is raised when building a priced program.

## 9. Open questions

- **D2**: whether `net_combined_ratio` should eventually take a `Distortion`
  as well as a number, resolving the implied technical premium ratio itself.
  Named here because it is the destination, and because it argues for the
  argument being a ratio rather than a loss ratio in name. Out of scope now.

**Closed.** v3's D1, a layer whose source program already carries a **`rate`**
premium clause, is settled by the author 2026-08-20: **refuse**, a `ValueError`
naming the layer and telling the caller to restate it as a deposit, or to drop
it and let the ladder price it. Its resolved amount is `s_j x r_j x P` and `P`
is what the ladder is computing, so it cannot enter `TP` at a known amount the
way a `deposit` or `rol` clause can, and the alternatives were to solve a
circularity the ruling had just removed or to break the margin identity
silently. The case is live, not hypothetical: a bare `agg` accepts the clause,
warns that it is ignoring it, and `pnl_program` reproduces it verbatim into a
program where it **does** resolve against `P` (measured).

v2's D1 (one ratio for the whole tower) is settled by the author's
"default to all the same". v2's D3 (`pr_attach` means different things on the
two tiers, and one threshold on it inverts the classification on a realistic
tower: a `900 xs 600` occurrence layer reads `pr_attach` 0.0203 while paying in
18.3 percent of years, against a `500 xs 1500` aggregate layer at 0.1247 paying
in 12.5 percent) is dissolved by the v3 ruling, which removes the threshold.
