# Plan: the `pnl` keyword — premium-minus-loss aggregates

> **Status:** READY (v1, next up — "quick hit"). Ships as **1.0.0a23**. Builds
> directly on the landed signed-support work: the Aggregate negative-x / output-
> window machinery (`dev/done/plan-negative-x-agg.md`, 1.0.0a21) and the signed
> Portfolio combine (`dev/plan-negative-x-port.md`, 1.0.0a22). **No new
> numerics** — `pnl` is a thin affine wrapper over the existing aggregate build
> path.
>
> Speculative extensions (gross/ceded-premium reinsurance views, general
> aggregate algebra) are **split out to `dev/TODO-Remember.md`** items 6b/6c so
> this plan stays a tight v1.

---

## 1. Motivation

A **profit is premium minus loss**, and the premium is collected **once for the
book**, not once per claim. The existing severity-level minus
(`ssev 100 - lognorm 80 cv .2`) puts the constant on the *severity*, so it is
multiplied by the claim count:

```
A = Σᵢ₌₁ᴺ (100 − Xᵢ) = 100·N − Σ Xᵢ        # 100 counted N times — NOT a book P&L
```

That severity form is **correct and unchanged** — it legitimately means "a fixed
cost / credit of 100 *per claim* plus a (scaled) severity", and it is exactly the
`dfreq[1]` (single-claim, severity-is-total-loss) special case of what we want
here. We are **not** touching it.

What's missing is the **once-for-the-book** form:

```
PnL = Premium − A,    A = Σᵢ₌₁ᴺ Xᵢ          # premium subtracted once
```

`pnl` provides it as a first-class object that goes anywhere an `agg` goes
(including as a `port` unit), so a book of `pnl` lines is a book-level
underwriting-result distribution via the signed combine we already built.

---

## 2. Syntax

A `pnl` declaration is a sibling of `agg`. It **always** states a premium and
**always** subtracts a loss aggregate — the surface mirrors the equation
`PnL = premium − loss`:

```
pnl NAME  <premium> prem  -  <loss-exposure> <layers?> <sev> <occ-reins?> <freq> <agg-reins?> <note?>
```

The part after `prem -` is an ordinary `agg` body; only the **exposure** head is
specialised so the premium can drive a loss ratio. Three exposure forms:

```
pnl B1 1000 prem - 70% lr  sev lognorm 100 cv 2 poisson      # loss = 1000·0.70 = 700; margin 300
pnl B2  100 prem - 10 claims sev lognorm 8 cv 1.5 poisson    # loss = 10-claim compound; prem independent
pnl B3  100 prem - 85 loss  sev lognorm 8 cv 1.5 poisson     # E[loss] = 85; margin 15
```

- **`r lr` / `N% lr`** — the **only** form that reaches back to the stated
  premium: `E[loss] = premium · lr`. One number (the premium) is both the amount
  subtracted and the loss-ratio basis. `lr` is simply that multiplier (we do
  **not** read any expense / combined-ratio meaning into it). (Reads better than
  `agg`'s `1000 prem at 0.7 lr`; we deliberately do **not** retrofit `agg`.)
- **`N claims`** — frequency-driven loss; the premium is independent.
- **`N loss`** — expected-loss-driven; the premium is independent.

Mental-model rule to document, one line:

> A constant inside `sev`/`dsev`/`ssev` is **per claim** (× frequency); the
> premium on a `pnl` line is subtracted **once** for the book.

### 2a. Vectorized premium (mirrors `agg`)

The premium **vectorizes exactly like an `agg` exposure** — a vector of
per-component premiums alongside vector layers / severity weights. We surface
**only the total `pnl`**, not a per-component breakdown:

```
pnl X [100 200 100] prem - .8 lr [1000 2000 5000] xs 0 sev lognorm 500 cv 2 poisson
```

reads exactly as the matching `agg` would: three components with premiums
`100/200/100`, each losing `80%` (`lr` applies component-wise to derive per-
component claim counts, as `exposures_premium_lr` already does), with limits
`1000/2000/5000`. Then:

- **shift `b = Σ premium` = 400** (the single deterministic amount subtracted);
- **loss `A`** = the usual mixture/weighted compound the `agg` body builds;
- **`PnL = 400 − A`**, mean `= 400 − Σ(premiumᵢ·lr) = 400 − 320 = 80` (i.e.
  `Σ premiumᵢ·(1 − lr)`).

So the premium vector does the same double duty per component (basis for `lr`)
and is summed for the shift. No per-component `pnl` object — total only. (If a
per-component P&L is ever wanted, that's the general-algebra item 6c, not v1.)

### 2b. Grammar sketch

Mirror `agg_out` (`decl.lark:51`) with a premium head and a bare-`lr` exposure:

```
pnl_out: PNL name numbers PREMIUM MINUS pnl_exposures layers sev_clause occ_reins freq agg_reins note
       | PNL name numbers PREMIUM MINUS pnl_exposures dfreq  layers sev_clause occ_reins      agg_reins note   // dfreq variant, cf. agg_out_dfreq

pnl_exposures: numbers CLAIMS   -> pnl_exp_claims
             | numbers LOSS     -> pnl_exp_loss
             | numbers LR       -> pnl_exp_lr        // NEW: bare lr, binds to the pnl premium
```

`PNL` is a new keyword terminal (`pnl`), added to the lexer keyword set and the
`ID` negative-lookahead exclusion list (`decl.lark:313`), priority alongside
`AGG`. `PREMIUM` and `MINUS` already exist. The `MINUS` sits between
`numbers PREMIUM` and `pnl_exposures`, so it is unambiguous (no existing
production has a number-premium followed by a minus then an exposure). `numbers`
already carries vectors, so 2a needs no extra grammar.

### 2c. Transformer

`pnl_out` builds the **same spec** an `agg_out` would for the loss body, then
records an aggregate-level affine transform and the premium:

- resolve the loss exposure:
  - `pnl_exp_claims` → `exposures_claims`;
  - `pnl_exp_loss` → `exposures_loss`;
  - `pnl_exp_lr` → synthesize `exposures_premium_lr` with `premium = <the pnl
    premium vector>` and `lr = <the bare lr>` (reuses the existing per-component
    claim-count derivation unchanged).
- attach `agg_premium = <premium vector>`, `agg_reflect = True`,
  `agg_shift = Σ premium`, `value_type = 'payoff'` to the spec.

The entire loss-model machinery (layers, sev, occ/agg reins, freq, mixing) is
reused verbatim; `pnl` only adds the premium + affine wrapper.

---

## 3. Semantics — the aggregate affine primitive

`PnL = premium − A` is a deterministic **relabelling of the aggregate grid**, not
a new convolution. It is the general primitive `a·A + b` restricted to what we
need: **reflection (`a = −1`) + additive shift (`b = Σ premium`)**. Magnitude
scaling (`|a| ≠ 1`) stays on the *severity* (homogeneous, already correct, and
would otherwise rescale `bs`).

Applied at the **end** of `update` (after the FFT produces the loss density on
its grid `xs`):

- **density:** reflect the density array and relabel `x ↦ Σprem − x` — exactly
  the signed F2 flip+roll already implemented for `sev_reflect` / windowed
  output, applied to the *aggregate* instead of the severity. The result lands on
  a signed grid straddling 0 (an underwriting loss is `PnL < 0`).
- **moments (closed form):**
  - `mean → Σprem − E[A]`
  - `sd   → sd(A)`  (shift / reflection don't change spread)
  - `skew → −skew(A)`  (reflection flips the sign)
  These flow into `stats_df` so `describe` / validation read correctly.

`value_type = 'payoff'` is set (profit is "more is better"). This finally gives
that inert member a job and is precisely what
`dev/plan-portfolio-neg-x-pricing.md` expects to consume — `pnl` is the natural
*producer* of payoff-typed objects, that plan is the *consumer*. (Open Q5:
confirmed.)

---

## 4. Implementation sketch

Ships as **1.0.0a23** (bump `pyproject.toml`; README bullet).

- **`decl.lark`** — `PNL` terminal; `pnl_out` + `pnl_exposures` productions (§2b).
- **`parser.py`** — `pnl_out` / `pnl_exp_*` transformers (§2c): reuse the agg
  spec builder, synthesize the `premium_lr` exposure for the bare-lr case, attach
  `agg_premium` / `agg_reflect` / `agg_shift` / `value_type`.
- **`distributions.py`** — `Aggregate` gains the affine fields (`agg_shift`,
  `agg_reflect`, default `0` / `False` so ordinary aggregates are untouched),
  applied at the end of `update_work` as a grid relabel (reusing the existing
  reflection/roll helpers), with the analytic moment adjustment into `stats_df`.
  `_signed()` already returns `True` when the grid straddles 0, so signed
  plotting / quantile / portfolio-combine paths just work. Signed-aware
  `describe` per §5.
- **`underwriter.build` / `build_many`** — route the `pnl` kind like `agg`
  (it returns an `Aggregate`); the Portfolio combine needs **no** change (a `pnl`
  unit is an ordinary signed `Aggregate` to it).

Net new code is small: a keyword, two productions + transformers, the affine tail
of `update_work`, and the describe swap. **Hard constraint:** with
`agg_shift = 0` and `agg_reflect = False` (every non-`pnl` aggregate) the path is
byte-for-byte unchanged.

---

## 5. Display — signed-aware `describe` (+ a P&L readout)  ⭐ Open Q4

Two parts. The first **also retro-fixes the 1.0.0a22 signed portfolio describe**,
which today shows a meaningless huge `Est CV` for a mean-near-zero P&L
(`CV = sd / mean` blows up as `mean → 0`).

### 5a. SD instead of CV when signed (general fix)

CV is unstable and meaningless when the mean can be ~0. So:

- When the object is **signed** (`_signed()` — grid straddles 0), the `describe`
  moment table swaps the **CV trio for an SD trio**:

  ```
  EX | Est EX | Err EX | SD | Est SD | Err SD | Sk | Est Sk
  ```

  `SD` is finite and informative regardless of the mean.
- When **not signed**, columns are unchanged (`CV …`) — byte-identical for every
  existing aggregate / portfolio.
- One detection (`_signed()`), one swap; applies uniformly to
  `Aggregate.describe`, `Portfolio.describe`, and `pnl` (a signed Aggregate).
- `Err SD` reuses the existing noise-aware relative error
  (`_noise_aware_rel_error`), but SD is rarely ~0 so it stays well-behaved.

This is the right home for the issue you flagged: **when signed, report SD not
CV.** (It cleans up the portfolio P&L describe we already shipped.)

### 5b. P&L summary readout (proposal)

For `pnl` specifically, append a compact one-glance actuarial summary (either a
small block under the moment table or in `info`). Proposed fields:

| field        | meaning / formula                                      |
|--------------|--------------------------------------------------------|
| `premium`    | the stated premium (Σ of the vector)                   |
| `E[loss]`    | `E[A]`                                                  |
| `E[margin]`  | `premium − E[A]`                                        |
| `lr`         | implied loss ratio `E[A] / premium`                    |
| `P(loss)`    | `P(PnL < 0) = P(A > premium)` = loss agg `S(premium)`   |

`E[margin]` and `P(loss)` are the headline numbers an underwriter wants;
`P(loss)` reuses the loss aggregate's survival function at the premium, so it's
essentially free. **Proposal for sign-off:** ship 5a (SD/CV) in v1 as it is both
necessary and small; ship 5b as the P&L block in v1 if cheap, else immediately
after. Risk readout note: profit risk is the **left** tail (`var(0.01)` = the
1-in-100 bad year); the signed `var`/`tvar` read the grid directly, the
payoff-vs-loss *orientation* only bites at pricing (deferred plan).

---

## 6. Portfolio composition

Because a `pnl` line is just a signed `Aggregate`, a `port` of `pnl` lines is the
**book-level underwriting P&L** — built by the signed combine landed in 1.0.0a22,
with each line's premium baked in. It mixes freely with ordinary `agg`
(pure-loss) units. This **subsumes** the "premium as its own fixed `dfreq[1]
dsev[premium]` portfolio line" trick: that trick is the manual engine; `pnl` is
the sugar that carries premium with the loss model on one line.

```
port Book
    pnl Cat   2000 prem - 55% lr sev lognorm 500 cv 3 poisson
    pnl Prop  5000 prem - 65% lr sev lognorm 100 cv 2 poisson
    agg Admin ...                      # a pure cost/loss line, no premium
```

→ total margin, diversification across lines, `P(total < 0)`.

---

## 7. Testing

- **mean/sd/skew:** `pnl X 1000 prem - 70% lr …` → mean `300`, sd = sd of the
  loss agg, skew = −(loss skew). Cross-check all three exposure forms
  (`lr` / `claims` / `loss`).
- **vector premium (§2a):** `pnl X [100 200 100] prem - .8 lr [1000 2000 5000]
  xs 0 sev …` → shift `400`, mean `80` (`Σ premiumᵢ(1−lr)`); equals the
  scalar-premium total of the same components.
- **equivalence:** `pnl X P prem - <body>` matches a hand-built reflected+shifted
  `agg X <body>` (same loss density relabelled `P − x`); and matches the
  portfolio trick (fixed `dfreq[1] dsev[P]` line + reflected loss line) to a
  tolerance.
- **per-claim vs once:** `pnl X 100 prem - 5 claims …` (mean `100 − E[A]`) is
  **not** `agg X 5 claims ssev 100 - <sev> …` (mean `5·(100−E[X])`) — assert the
  distinction so the two forms can't be confused.
- **`P(PnL<0)`** matches the loss aggregate's `S(premium)`.
- **describe (§5a):** a signed object's describe shows SD columns (not CV) and is
  finite for a mean-zero P&L; a non-signed object's describe is unchanged.
- **portfolio:** a 2-line `pnl` book combines correctly (means add, variances add
  under independence); mixes with an `agg` line.
- **regression:** every non-`pnl` build is byte-for-byte unchanged
  (`agg_shift=0`, `agg_reflect=False`); non-signed describe unchanged.
- DecL mirrored in `test_decl.agg` (a `PnLprem` section); tests in
  `tests/test_pnl.py`. `uv run pytest` green (`UV_LINK_MODE=copy`).

---

## 8. v1 scope / non-goals

- **One premium, one loss view.** v1 subtracts the single stated premium from
  whatever loss view the agg body emits (gross / ceded / net, per its reins
  clause). **It is the user's job to make the premium consistent with that loss
  view** (e.g. supply a net-of-reins premium when the body is `... net of ...`).
  v1 does **not** model ceded premium — that's TODO 6c.
- **`lr` is just the multiplier** (`E[loss] = premium · lr`); no expense /
  combined-ratio semantics.
- **Magnitude scaling stays on severity** (homogeneous); the aggregate affine is
  reflect + shift only.
- **Total only** for vector premium — no per-component `pnl` (TODO 6c).

---

## 9. Speculative extensions — moved to TODO

To keep v1 tight, the forward-looking ideas live in `dev/TODO-Remember.md`:

- **6c — gross/ceded-premium reinsurance P&L.** `pnl B 1000 gross prem 200 ceded
  prem - <agg with reins>`: net premium = gross − ceded, and the agg's existing
  gross/ceded/net **loss** views become parallel gross/ceded/**net P&L** views
  (the cedent's *and* reinsurer's underwriting result from one line) — three
  affine transforms of the three loss legs. Powerful, but an accounting layer on
  top of the core `pnl`.
- **6b — general aggregate algebra.** First-class constant aggregates and full
  `agg.A ± agg.B ± c` arithmetic (and hence per-component `pnl`). A v2.0
  "aggregate algebra" layer; `pnl` covers the common premium-minus-loss case
  without it.

---

## 10. Open questions — resolved

1. **Keyword spelling** → `pnl`. ✅
2. **`lr` meaning** → just the multiplier, not a "loss ratio" with expense
   semantics. ✅ (§2, §8)
3. **Vector premium** → yes, mirrors `agg`; total `pnl` only (§2a). ✅
4. **`describe`** → §5: SD-not-CV when signed (also fixes the a22 portfolio
   describe) + a P&L summary block (premium / E[loss] / margin / lr / P(loss)).
   ✅ (proposal above, 5b for sign-off)
5. **`value_type` pricing** → yes, `pnl` sets `payoff`; it is exactly what the
   pricing plan consumes. ✅
