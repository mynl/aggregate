# Plan — first-class `PnL`: orientation suffix, a clean veneer, and *evaluation*

> **Progress (staged execution):**
> - **Stage A — DecL `payoff`/`loss` orientation suffix — DONE (1.0.0a102).**
>   `agg … poisson payoff` sets `value_type` only (no reflect/shift); prices
>   through the dual via the existing `_is_loss_value` / `_canonical_loss_frame`
>   path. Grammar (`decl.lark` `orientation` rule + `PAYOFF` terminal), parser
>   (`orientation_*`), unparser round-trip (`decl_writer._render_orientation`),
>   tests (`tests/test_negative_x.py`), corpus (`decl-testers.agg` `Orient.*`).
> - **Stage B — first-class `PnL` veneer; in-place affine removed — DONE (1.0.0a103).**
>   `PnL` composition over a pure-loss `.agg`; `make_pnl`; `build('pnl')` → `PnL`;
>   derived `pnl_df` (no FFT/window); affine ripped out of `Aggregate` /
>   `_bucket_window` / `tail` / `bivariate`. **Portfolios & bivariates of `pnl`
>   are deferred** (book-level / joint P&L needs net-then-combine, which loses
>   per-unit attribution) and now raise `NotImplementedError`; a payoff book uses
>   `agg … payoff` orientation. Constant **and** callable consideration supported.
> - **Stage C — signed additive `summary_df` + `PnL.plot()` — DONE (1.0.0a104).**
>   `summary_df` = Consideration / Obligation / Margin (EX adds; SD not CV;
>   constant consideration certain, callable carries SD/Sk; buying flips both
>   signs). `plot()` = Margin density + distribution, break-even line, no sev
>   panel (`plot_pnl` compositor).
> - **Stage D — Cherny–Madan `evaluate` panel — DONE (1.0.0a105).**
>   `PnL.evaluate()` calibrates `g(loss-version) = held consideration` over the
>   full support (reuses `Distortion.calibrate_set`); returns the acceptability
>   panel (ph/wang/dual/tvar, ccoc excluded) with breakeven `param` + `gini_p`
>   (monotone in profit). Constant consideration only.
> - Stage E (reins-aware GCN view, DecL swing/slide builder) and the deferred
>   `PortPnL` — pending.
>
> **Status: DRAFT — partially executed (Stages A, B, C, D landed).** Re-architects the `pnl` concept. **`pnl` is
> new and nothing external depends on it — this is a free *internal* change**, not
> a compat-constrained migration; the return type of `build('pnl …')` changes
> (`Aggregate` → `PnL`) and we update our own consumers freely. Version bump,
> `CHANGELOG.md`, full suite green before any commit.
>
> **Goal (acceptance):** (1) any aggregate's loss/payoff **orientation** is
> explicitly declarable in DecL; (2) a `PnL` is a thin, *honest* veneer over an
> `Aggregate` — no more "loss moments + payoff density" bastard; (3) a `PnL` is
> **evaluated** (Cherny–Madan breakeven), not priced; (4) it has its own
> `.plot()` and a **signed, additive** Consideration/Obligation/Margin
> `summary_df` (SD not CV; no severity panel); (5) it is **reinsurance-aware** —
> one consideration → a net P&L, three → the Gross/Ceded/Net view.
>
> **v1.0 scope:** `PnL` over an **`Aggregate`** — the `payoff`/`loss` suffix, the
> veneer (`.plot()`, signed summary, `evaluate` panel), **constant *and*
> function-valued consideration** (functions passed by hand in Python — no DecL
> *builder* for swings/slides in 1.0), and **reins-awareness incl. GCN**. A
> per-unit **`PortPnL`** comes nearly free off the `Portfolio` combine and is
> in-scope **if the veneer lands clean**. **Deferred (post-v1.0):** the DecL
> swing/slide grammar, book-level `PortPnL` consideration, named Cherny–Madan
> families (TODO #23), and an independent/dependent `Aggregate` consideration leg.

---

## 0. The model — three layers, currently fused into one object

Today `pnl` fuses three orthogonal things and then can't decide which it is
(`agg_m` reports the **loss** 900; `density_df`/`q` report the reflected
**payoff** 100; pricing treats it as a payoff). The fix is to *name the layers*:

| Layer | What it is | Sign | Operation |
|---|---|---|---|
| **X** — risky primitive | an `Aggregate`; loss (default) or `payoff` | own convention; mean any sign | **priced** (distortion → risk-adjusted value) |
| **P** — consideration | a scalar | — | **input** (held position) **or output** (a quote) |
| **Net** — the P&L | `P − X` (X loss) or `P + X` (X payoff) | **always payoff** | **evaluated** (derived; never priced) |

**"A P&L is always payoff" is a definition.** It's in the name - a PROFIT and LOSS 
is often used  as a synonym for payoff convention. More net money is
always better: X loss ⇒ `net = P − X` (more loss → less net); X payoff ⇒
`net = P + X` (more X → more net). Either way the net is more-is-better. What is
*free* is **X's** orientation; the net's payoff-ness is forced.

**The combine sign follows X's orientation — there is one knob, not two.**
Long/short is rarely a helpful construct: long X is short -X but what is X. That said...
Long/short is *not* an independent direction: writing/selling a payoff means you
**owe** a good outcome, which is a **loss to you**, so a "short payoff" is simply
`X` oriented as a loss. The same RV `max(S−K,0)` is a payoff if you hold it and a
loss if you wrote it — that *is* orientation. Hence:

```
net = P − X   if X is a loss
net = P + X   if X is a payoff
```

`make_pnl(P=…)` therefore takes **no `sign` argument** — it reads `X.value_type`.
In DecL the `prem -` / `prem +` spelling must agree with the orientation
(`-` with `loss`, the dominant case; `+` with `payoff`); it is a readability
echo of the orientation, not a second degree of freedom. Long/short and
buy/sell are expressed by *which orientation you give X*, which keeps the whole
conic long/short space reachable through the single knob.

---

## 1. Evaluation, not pricing — the Cherny–Madan breakeven

You never price the net; you **price X** (apply a distortion to the risky leg).
"Evaluating a P&L" = find the distortion parameter that drives the risk-adjusted
net to **zero** — the breakeven stress the position survives. The net is always
payoff, so evaluate it with `gdual`; with translation (`g(P+Y)=P+g(Y)`) and
duality (`g(−X) = −gdual(X)`) both directions reduce to the **same** equation:

```
X loss,    net = P − X:   gdual(P − X) = P − g(X)   = 0  ⟹  P = g(X)
X payoff,  net = P + X:   gdual(P + X) = P − g(−X)  = 0  ⟹  P = g(−X)   (−X is the loss)
```

Both: **`P = g(loss-version of the risky leg)`** — which is *exactly* the
`ρ_g(S) = premium_target` equation `Distortion.calibrate_set` already solves.
So evaluation reuses the calibration solver with **`premium_target = the held
P`** (an input) over the **full** support (`a = ess_sup`; no asset cap, no `coc`
inversion). That is why a `PnL` exposes *evaluate*, not the `coc` `price`
methods — and why `ccoc` (which needs an asset level) is **not** in the evaluate
panel.

### 1.1 `gini_p` is the family-agnostic acceptability index

> **Terminology.** "Acceptability index" (Cherny–Madan) = the largest stress
> level at which a position's risk-adjusted value stays ≥ 0 — bigger ⇒ more
> acceptable / profitable. **Do not abbreviate it "AI"** (overloaded); write
> "acceptability index" / "acceptability panel" in full.

The raw breakeven parameter is family-specific (`wang λ` ≠ `tvar p`), but
`gini_p = 2∫g − 1 ∈ [0, 1]` is comparable and **monotone in loading** (0 at the
identity, 1 at maximal stress). At breakeven a more profitable position survives
a larger stress ⇒ larger `gini_p`. So **breakeven `gini_p` is the single
Cherny–Madan acceptability score** (`@Cherny2009a`), positively correlated with
profitability; the per-family params are its spellings. The `calibrate_set`
family loop therefore yields an **acceptability panel** in one common currency
for free. Edge cases saturate meaningfully: a guaranteed loser at `gini_p = 0`,
a deal acceptable at any stress at the family maximum.

> **Citations.** `@Cherny2009a` — Cherny & Madan, *New Measures for Performance
> Evaluation* (key confirmed present in `uber-library.bib`). They define **four**
> families: **MINVAR** `1−(1−u)^{1+x}` (min of `1+x` copies), **MAXVAR**
> `u^{1/(1+x)}`, and the compositions **MINMAXVAR** / **MAXMINVAR** (load both
> tails; MINMAXVAR is the conic-finance workhorse). Map to us: **MINVAR *is* our
> `dual`** (`b = 1+x`) — and with integer `b = n`, `dual` prices the loss as
> `E[max(X_1..X_n)]` (worst of `n` draws), the mirror of MINVAR's min-of-payoffs.
> So the `dual` row *is* **AIMINVAR**; a `ph`/power row maps to **AIMAXVAR**
> (parameter map to confirm); **MINMAXVAR** is the one kind worth *adding* to
> surface **AIMINMAXVAR**. The panel already *is* their indices, in `gini_p`
> currency. (Names are theirs; we report in full words, not the "AI…" prefix.)

---

## 2. Architecture — `PnL` is **has-a** `Aggregate` (composition)

```
PnL
  .agg            -> Aggregate X   (UNTOUCHED: density_df = the OBLIGATION; honest moments/plot of X)
  .consideration  -> Consideration (the C leg; SIGNED — see below)
  .pnl_df         -> derived NET   (consideration ± obligation; computed on the fly, cached)
```

The combine sign is **read from `X.value_type`** (§0), not stored as a free
parameter: `net = consideration − X` for a loss, `consideration + X` for a
payoff. **Consideration is signed** — `+` when received, `−` when paid — so a
bought position is just a negative consideration; the convention never flips.

**Construction — one path, the DecL `pnl` is sugar over it:**

```python
pnl = agg.make_pnl(consideration=...)     # no sign arg (reads X.value_type).
                                          # consideration = signed number | increasing function f(X)
# build('pnl 100 prem - <body>')   ==   build('<body>').make_pnl(consideration=100)
re_pnl = agg_re.make_pnl(gross=..., ceded=...)   # reins-bearing agg -> GCN PnL (§3.4)
re_net = agg_re.make_pnl(consideration=...)      # one premium on a reins agg -> NET-only P&L
# port_pnl = port.make_pnl(premiums={...})       # PortPnL -> POST-v1.0 (§3.2)
```

**Consideration is a number _or_ a function — *both in v1.0*** (the
function form is what makes a P&L interesting: swing/retro premium, profit
commission, reinstatements — so the `summary_df` Consideration row carries a real
SD/CV/Sk, not just a constant). Variable consideration is almost always a
**function of the loss**, not an independent RV, so it lives on the *same* `X` and
needs no joint-distribution model. Three rungs:

1. **constant** (signed scalar — today's `pnl`). *v1.0.*
2. **increasing function of `X`** — `Consideration = f(X)`, comonotone with the
   loss, applied bucket-wise on the shared grid, so `Margin = f(X) − X` is a pure
   1-D transform (**no joint model, no FFT**). *v1.0 — but the function is passed
   **by hand** (a Python callable / array); the `make_ceder_netter` layer machinery
   is the natural builder and we demo it by hand. A **DecL swing/slide grammar is
   deferred** (post-v1.0).*
3. **independent / dependent `Aggregate`** — only if a real case demands it
   (a dependence assumption; defer to the bivariate machinery). *Post-v1.0.*

- **`Aggregate` stays pure.** Drop the in-place `_apply_agg_affine` mutation of
  the core density from the `pnl` update path. `X.density_df` = **the obligation**
  (the loss), with honest moments and plot, full stop.
- **The net is `pnl_df`, computed on the fly (cached) — not a second stored
  density.** `density_df` consistently means *the obligation* (on `.agg`); the
  net = *consideration ± obligation* is a cheap deterministic transform of it
  (`_pnl_window` two-sided grid + reflect/shift, or the `Consideration=f(X)` map), so it
  is derived lazily, keeping **one** source of truth. `PnL`'s distribution-like
  methods (`q`/`cdf`/`mean`/plot, the `Margin` row) read `pnl_df`. No
  schizophrenia: **two honest frames** — `pnl.agg.density_df` is the obligation X
  (900); `pnl.pnl_df` is the net (100).
- **Small override surface** — exactly: forced-payoff net, `summary_df` (§3),
  plot-without-sev (§3), `evaluate` (§1). Everything else delegates to `.agg`.
- **One construction path** (per the block above): `agg.make_pnl(consideration=…)`
  is canonical (no sign arg — reads `X.value_type`); `pnl …` DecL builds `X` then
  calls it, so `pnl 100 prem + agg.PAYOFF` and
  `build('agg PAYOFF … payoff').make_pnl(consideration=100)` coincide.

**Why composition over subclass:** subclassing `Aggregate` inherits ~6k lines
(`update`, reins, windowing) that must then be policed against leaking the wrong
identity — fragile-base-class risk for no gain, since we *want* most behavior
unchanged and only a handful of methods to re-skin.

---

## 3. Reporting

### 3.1 `summary_df` — signed, additive P&L rows (SD, not CV)

A `PnL` summary **breaks with accounting a little**: every row is its **signed
contribution to the P&L** (loss sign convention), so the rows **add** —
`Consideration + Obligation = Margin` — and it shows the **SD** spread trio, *not*
CV, because the margin sits near zero where CV is meaningless (reuse
`_describe_signed`). Same 8-column shape as an `agg`:

```
                  EX     SD
Consideration    +100     0     paid/received at INCEPTION  (+ received, − paid)
Obligation        −90    ..     the risky leg's CONTRIBUTION (− a loss borne, + a payoff held)
Margin            +10    ..     = Consideration + Obligation
```

A sold-cover loss shows Consideration `+100`, Obligation `−90` (the loss subtracts
from P&L), Margin `+10`. **Buying flips both signs**: Consideration `< 0` (you paid
at inception) *and* Obligation `> 0` (you hold the payoff) — e.g. paid 100 for a
payoff worth ~90. One convention, fully additive, no relabeling.

"**Consideration**" deliberately denotes *the amount changing hands at inception*
— clearer than the overloaded "premium". **No Freq/Sev/Agg detail here** — that
lives on `pnl.agg.summary_df` (the honest `X` table). Clean split: `pnl.agg` = the
risky primitive; `pnl` = the position's signed P&L.

### 3.2 `PortPnL` — the multi-line P&L *(cheap; v1.0 if the veneer lands clean)*

> **Mostly free — reconsidered.** A `PortPnL` reuses almost everything: the
> obligation combine `Σ X_i` *is* the existing `Portfolio` FFT, and the total net
> is just `Σ C_i + (signed) Σ X_i` = `make_pnl(port_total, consideration = Σ C_i)`.
> So it's a consideration **vector** + a stacked signed summary over the
> `Portfolio` that already exists. Include it in v1.0 if the `Aggregate` veneer
> generalizes cleanly; the **only** genuinely new piece — splitting a *book-level*
> consideration back to units — is an allocation problem and stays deferred.

A `port` whose units are all `pnl`s is a **`PortPnL`**: it stacks each unit's
signed Consideration/Obligation/Margin and adds a **total** row — the exact
analogue of `Portfolio.summary_df` stacking per-unit Freq/Sev/Agg + total. No new
DecL keyword: the all-`pnl` unit set drives the type, as a mixed book's role is
inferred from its units today. (Class name `PortPnL`, mirroring the
`Aggregate`/`Portfolio` pair; `pnl` stays the only `pnl`-family keyword.)

**Consideration — per-unit *or* top-level.** Per-unit (`make_pnl(premiums={unit:
…})`) is the cheap v1.0 path; a single **book-level** consideration against the
total is the deferred bit (splitting it to units = allocation, reuse the portfolio
allocation machinery). **Evaluation is top-level** (the book's breakeven), but
each unit's `PnL` evaluates **standalone** too — the book index *and* a per-unit
acceptability profile from the same objects.

### 3.3 `PnL.plot()` — required, no severity panel

A `PnL` **has its own `.plot()`** (required, not delegated). The net is an
**affine of the aggregate**, not a compound of a severity, so it has **no sev
panel** — that severity/`d/dx` density-derivative panel belongs to an `Agg`. It
shows the **Margin** density + distribution; for the GCN form (§3.4) it overlays
the Gross/Ceded/Net legs. When `X` carries reinsurance we **plot what comes out**
of `X` (gross/ceded/net already baked into the obligation). The bare `X` is still
plotted via `pnl.agg.plot()`.

### 3.4 Reins-aware `PnL`, and the Gross / Ceded / Net view — *additive*, signed

**`PnL` is reinsurance-aware (v1.0).** On a reins-bearing `Agg`:

- **one consideration** (`make_pnl(consideration=…)`) → a **net-only** P&L against
  the net loss (`agg`'s net obligation — what comes out);
- **two/three premiums** (`make_pnl(gross=Pg, ceded=Pc[, net=…])`) → the full
  **Gross/Ceded/Net** three-way view.

A reins-bearing `Agg` already carries gross/ceded/net loss distributions, all
**comonotone** (deterministic functions of the one gross loss). So the GCN rows
**add** — `Net = Gross + Ceded` — once consideration carries its sign:

```
              Consideration     Obligation     Margin
Gross         +Pg               L_gross        Pg − L_gross
Ceded         −Pc               −R (recovery)  R − Pc          (PAY premium, GET recovery)
Net (=G+C)    Pg − Pc           L_gross − R    (Pg−Pc) − (L_gross−R)
```

"Ceded is negative relative to gross" is literal: the ceded leg's consideration
(`−Pc`, you pay) **and** obligation (`−R`, recovery offsets loss) are negative, so
the legs sum to Net. This is the **additive, signed-rows** exhibit (the accountant
wish from the signed-calibration discussion, finally falling out) — and it stays
1-D (comonotone on the shared grid, no joint model). `net=` overrides the derived
`Pg − Pc` when the retained premium is stated directly. Summary and plot render
the three legs; `Net` is the headline.

> **Implementation dependency (confirm early).** GCN needs the `Agg`'s
> **gross / ceded / net** loss distributions retrievable as *separate* frames
> after `update()`. The reins machinery *computes* all three (`reins_density_df`
> etc.); verify they're retained (or cheaply recomputable) before committing the
> three-way exhibit — the one real unknown in the v1.0 scope.

---

## 4. DecL grammar

1. **Orientation suffix on any `agg`** — `agg … poisson payoff` sets X to the
   payoff role (default `loss`). Pure orientation: flips pricing to `gdual`, **no
   reflection** (an asset-return / direct-payoff primitive). Keywords **`payoff`**
   and **`loss`** (loss rarely used but available for explicitness). *No* `payout`
   alias (house rule: one canonical name; matches the existing `value_type`
   token). Placed **last, immediately before the `note`/hints trailer** — after
   `freq`/reins/`approximate` — so it cannot collide with the `loss` *exposure*
   keyword (`numbers LOSS`, which sits in the exposure head).
2. **Direct creation is preserved.** `pnl NAME <consideration> premium ± <body>
   [payoff|loss]` still builds a `PnL` in one line (now the veneer, not an
   `Aggregate`-with-affine). **Defaults reproduce today's behavior**: `premium -`
   and `loss` orientation → `net = consideration − X`, X a loss. `prem +` is the
   new alternative; the body may be a `payoff`-tagged agg or an `agg.NAME`
   reference. The trailing orientation suffix (point 1) is the same keyword.
3. `pnl` exposures (`claims`/`loss`/`lr`) unchanged.

---

## 5. Internal updates & tests (no external compat constraint)

**`pnl` is new — nothing external depends on it, so we change it freely.** The
return type of `build('pnl …')` becomes `PnL` (was an `Aggregate`-with-affine);
the only work is updating **our own** consumers — not a delicate migration.

- Update every `pnl` / `_apply_agg_affine` / `_agg_affine_active` consumer:
  `tests/test_pnl.py`, the `PnL*` / `H4.Pay*` corpus in `decl-testers.agg`,
  `summary_df` / `_describe_signed`, plotting, the bivariate `pnl` component path,
  and the **signed-calibration** entry (`_pricing.calibrate_distortions` reads
  `_is_loss_value`; `PnL.evaluate` now front-doors that for held P&Ls).
- **Drop the in-place affine** from the `Aggregate` update path: `pnl`'s reflect/
  shift moves to `PnL.pnl_df` (derived). A plain `Aggregate` no longer carries
  `_agg_reflect`/`_agg_shift`; confirm nothing else relied on them.
- **`value_type` setter** stays (`a.value_type = 'payoff'`); the DecL suffix is
  the supported path. Reconcile with `test_hygiene4` / `test_numerics3`.
- New `tests/test_pnl_evaluate.py`: breakeven reproduces `g(X) = P`; `gini_p`
  monotone vs consideration; the acceptability panel; orientation × sold/bought
  (signed consideration); **function-valued consideration** (Consideration row has
  SD/CV); summary rows reconcile and add; GCN `Net = Gross + Ceded`; `PnL.plot()`
  runs and has no sev panel. Mirror DecL as `PNL2.*` in `decl-testers.agg`.

---

## 6. Out of scope / future

- **Bid/ask breakeven — *not pursued*.** The counterparty side is just the
  orientation flip (the sign of `X`), which the `payoff`/`loss` knob already gives;
  no separate bid/ask machinery and the terminology isn't adopted.
- **General affine clause.** `pnl` is sugar for `consideration + sign·X`; a fully general
  `c + s·X` modifier on any agg is the eventual endpoint but is **not** pulled in
  here — wait for a real use case.
- **No change to the FFT/convolution core**, the distortion subclass math, or
  the landed signed-calibration pricing surface.

---

## 7. Decisions (author-confirmed) & remaining questions

**Confirmed (v1.0 — `PnL` over `Aggregate`):**
- **`pnl` is new; nothing depends on it** → a free internal change (return type
  `Aggregate` → `PnL`), not a compat-constrained migration. ✓
- `density_df` = **the obligation** (loss), on `.agg`; the **net** is `pnl_df`,
  computed **on the fly** (cached), single source of truth. ✓
- The combine **sign follows `X.value_type`** — `make_pnl(consideration=…)` takes
  **no `sign` arg**; long/short is X's orientation (one knob). ✓
- `consideration` accepts a **signed number _or_ an increasing function `f(X)`**,
  **both v1.0** — the function passed **by hand** (Python callable/array; demo
  swings via `make_ceder_netter`). A **DecL swing/slide builder is post-v1.0.** ✓
- **Reins-aware** — one consideration → net-only P&L; `gross=, ceded=[, net=]` →
  the additive **GCN** three-way (`Net = Gross + Ceded`), comonotone, 1-D. ✓
- `pnl.evaluate` uses the **default distortion families** (panel); `ccoc`
  excluded (no asset level). ✓
- **`PnL.plot()` is required** (its own method); no sev panel (that's the `Agg`
  `d/dx` panel); Margin density + distribution, GCN overlays the three legs. ✓
- **Direct `pnl …` creation preserved** — `pnl NAME C premium ± <body> [payoff|
  loss]`; defaults `premium -` + `loss` reproduce today's behavior. ✓
- **`payoff`/`loss` postfix keywords**, last token before `note`/hints. ✓
- `summary_df` = **signed, additive** Consideration / Obligation / Margin rows
  (`Consid + Oblig = Margin`); shows **SD, not CV** (near-zero margin); a bought
  position is Consideration `< 0` *and* Obligation `> 0`. No Freq/Sev/Agg here. ✓
- The breakeven **acceptability panel** *is* the Cherny–Madan indices in `gini_p`
  currency; never abbreviate "acceptability index" as "AI". ✓
- **Bid/ask not pursued** — it's just the orientation flip (sign of `X`); no
  separate machinery. ✓

**v1.0 if the veneer lands clean (cheap, reconsidered):**
- **`PortPnL`** (§3.2) per-unit — reuses the `Portfolio` combine + a consideration
  vector + a stacked signed summary. Include it unless the `Aggregate` veneer
  fights back; only **book-level consideration allocation** stays deferred.

**Deferred (post-v1.0):**
- **DecL swing/slide grammar** for function-valued consideration (the function
  itself ships, passed by hand).
- **Book-level `PortPnL` consideration** (allocation).
- **Named Cherny–Madan families** (MINMAXVAR/…) — TODO #23.
- **Independent/dependent `Aggregate` consideration** leg.

**Remaining (non-blocking):** confirm the reins gross/ceded/net frames are
retrievable post-`update()` (§3.4 implementation dependency) before building GCN.
