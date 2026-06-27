# Shared design appendix — legs model, dimensionality, per-layer economics, naming

> **Status: DRAFT — shared foundation for three plans.** Referenced by
> `plan-pnl-expenses-ceded-premium.md` (Phase 1), `plan-reinstatements.md` (Phase 2), and
> `plan-variable-rating.md` (Phase 3). It holds the cross-cutting design decisions so the three
> plans do not duplicate (or drift on) them. Locked with the author 2026-06-27.

The end state: **expenses, ceded premium, ceding commission, and the six variable-rating
features (reinstatement premium, retro, swing, slide, profit commission, corridor) are all
the same object** — a deterministic function pushed forward over the loss distribution,
writing one leg of a Gross/Ceded/Net P&L. This appendix names that object once.

---

## 1. The legs model (the backbone)

A `PnL` is an **ordered set of legs**. Each leg is:

| Field | Values |
|---|---|
| `perspective` | `gross` / `ceded` / `net` (net is derived: `gross ⊕ ceded`) |
| `kind` | `consideration` (premium) / `obligation` (loss or expense) |
| `value` | a **scalar** (deterministic) or a **`GridDistribution`** (stochastic) |

Payoff-sign convention (premium received `+`, paid `−`; loss a negative obligation; recovery,
commission, profit commission `+`). **Rows add and columns add — in expectation.** UW per
perspective `= Σ consideration + Σ obligation` (signs already baked in). The exhibit is:

| | Premium (consid.) | Loss (oblig.) | Expense (oblig.) | UW (margin) |
|---|---|---|---|---|
| **Gross** | `P_G` | `−L` | `−E_G` | `P_G − L − E_G` |
| **Ceded** | `−CededPrem` | `+A` | `+C + PC` | `A + C + PC − CededPrem` |
| **Net** | `P_G − CededPrem` | `−(L − A)` | `−(E_G − C − PC)` | rows add |

where `A` = ceded loss (recovery), `C` = ceding commission, `PC` = profit commission, `E_G` =
gross expense. **Net loss `= L − A`; net expense `= E_G − C − PC`** — i.e. ceding commission
and profit commission **book in the expense column** (a credit / negative expense), *not*
against consideration (author decision). When `C`/`PC` are stochastic this makes the expense
leg a **distribution** (`cv(expense) > 0`), which is exactly slide / PC.

**Which legs can be distributions** (the entire variable-feature surface — three stochastic
slots):

| Leg | Made stochastic by |
|---|---|
| ceded premium (consideration) | reinstatement premium, swing |
| expense (obligation) | slide, profit commission (and `cede` is its deterministic case) |
| ceded loss (obligation) | corridor (and the reinstatement annual cap) |
| gross premium (consideration) | retro (account-level, rating clause) |

`[pnl-share]` (Phase 1) builds the table by **iterating legs**, agnostic to scalar-vs-distribution
and to which slot is filled. Every later feature is then *additive*: it fills a leg, it does not
reshape the table.

---

## 2. Dimensionality — set by the reinsurance *basis*, not the feature

The cost of a feature is **1-D or 2-D according to whether ceded loss is a deterministic
function of gross loss**, which is a property of the layer's basis, *uniform across features*:

- **Aggregate basis → 1-D.** `ceded = c(L_gross)` deterministic; the 1-D gross density gives
  ceded, net, and their (degenerate) joint. Any leg `φ(L, ceded) = ψ(L)` is a **1-D pushforward
  of the gross density**.
- **Occurrence basis → 2-D.** The recovery `R` is *not* a function of `L_gross` (the claim-count
  split matters), so `(L, R)` has a genuine 2-D joint (FFT2 `occ_bivariate`); any leg combining
  the two needs it.

Consequences:

- **Means always add and are 1-D.** Only the **SD / CV / percentile** rows of the occurrence
  GCN need the joint. ("Means add but SDs do not.")
- **Reinstatement premium is *always* 2-D** because reinstatements are intrinsically an
  occurrence construct. Every other feature is 2-D on an occurrence layer, **1-D on an aggregate
  layer**. Retro on a book with no occurrence reinsurance is the cleanest 1-D case.
- **The engine must ship both paths** — `pushforward_1d(gross_density, φ)` and
  `pushforward_2d(joint, φ)` — returning the same `GridDistribution` leg. φ and target-leg are
  feature-specific; the **basis selects the path**.
- **`[one-variable-occurrence-layer]` (the 2-D ceiling).** Two occurrence layers with
  independent variable pricing need `(L, R₁, R₂)` = 3-D. So **at most one occurrence layer may
  carry any variable feature** (reinstatements, swing, slide, pc, corridor); other layers must be
  plain or aggregate. Aggregate variable features stay 1-D in gross and may stack freely. This is
  a foundation rule, not a reinstatement quirk — Phase 2's `[reins-single-layer]` is its first
  instance.

---

## 3. Per-layer economics matrix (composition + mutual exclusion)

A priced layer is a **deterministic pipeline** of slots, applied in accounting order; each slot
reads that layer's own recovery `R` (or `A`), so multiple filled slots on **one** layer stay
2-D. The slots:

| Order | Slot | Fills leg | Options (choose ≤ 1 unless noted) |
|---|---|---|---|
| 1 | **loss transform** | ceded loss `A` | `corridor` (optional); reinstatement annual cap (implicit) |
| 2 | **premium** | ceded premium | `deposit` \| `rol` \| `rate` (fixed) — **or** `swing` (variable) |
| 3 | **commission** | expense (credit) | `cede` (fixed) — **or** `slide` (variable) |
| 4 | **profit comm.** | expense (credit) | `pc` (optional) |

Reinstatement premium = a **fixed premium slot** (`deposit`/`rol`/`rate`) **+** the
`reinstatements` decorator (it varies the *already-chosen* base premium). Retro is **not** in
this matrix — it sets **gross** premium at the account level (rating clause), Phase 3.

**Validation (errors):**

- premium slot: more than one of `deposit`/`rol`/`rate`/`swing` → error; `swing` **replaces**
  `deposit`/`rol`/`rate`.
- commission slot: `cede` and `slide` together → error (`slide` replaces `cede`).
- `reinstatements` requires exactly one fixed premium clause as its base and **no** `swing`.
- cross-layer: `[one-variable-occurrence-layer]` (§2).
- `rate` (% of gross premium) is a run-time error if the block has no gross premium.
- `rol` requires a finite limit.

---

## 4. Loss basis per feature (defaults; nameable where ambiguous)

φ reads a loss quantity; default to the natural one, force an explicit basis only where the
denominator is genuinely ambiguous:

| Feature | Loss basis | Denominator (for ratios) |
|---|---|---|
| Reinstatement prem. | occurrence recovery `R` (capped) | — |
| Retro | net account loss (after inuring reins) | — |
| Swing | ceded loss to the layer | — |
| Slide | **ceded** loss ratio | ceded premium |
| PC | ceded loss ratio | ceded premium |
| Corridor | ceded loss ratio | ceded premium |

Note (author): for a **quota share** the cession rate is 100% (scaled by percent placed), so the
ceded and subject loss ratios coincide — `slide` on a QS reads the subject LR by construction.

---

## 5. Naming / keyword set (rationalized)

New DecL terminals (everything else reuses existing `at`, `and`, `premium`, `loss`, `po`, `xs`):

```
deposit  rol  rate  cede                      # ceded premium + commission (Phase 1)
expense(s)                                     # gross expense (Phase 1)
retro  swing  basic  lcm  min  max            # variable premium (Phase 3)
slide                                          # variable commission (Phase 3)
pc  after                                      # profit commission (Phase 3)
corridor                                       # variable ceded loss (Phase 3)
```

plus the bounded number-words `one…five` (from Phase 2). Rationalization decisions:

- **`deposit` only** for fixed ceded premium — **no `min` synonym** (frees `min`/`max` for the
  swing/retro collar exclusively, removing the clash).
- **`corridor` reuses `po`/`xs`** (a loss-ratio layer: `corridor 50% po 30% xs 20%`) — **no
  `from`/`to`**.
- **`slide` reuses `at`/`and`** (anchor pairs, §Phase 3).
- **`cede`** kept (house-short); watch the `cede`/`ceded` (direction keyword) proximity in the
  lexer pass. Swap to `commission` only if it bites.
- Expense base is **explicit in all three forms**: `25% premium expenses` / `25% loss expenses`
  / `2000 fixed expenses`.

---

## 6. The shared engine seam

- **`[engine]`** (built in Phase 2): `pushforward_1d` + `pushforward_2d` →
  `GridDistribution`; `transformed_moments` (exact moments on the source grid). Generic over an
  arbitrary vectorized φ; **not** reinstatement-named. **This replaces `PnL`'s comonotone
  net-distribution relabel** (`pnl_df` / `_frame_from`, `_pnl.py:160`) once a leg is
  loss-sensitive: a saturating XOL makes φ many-to-one (must *bin*, not *sort*), and the
  occurrence split makes the net result non-single-valued in `L` (→ 2-D). Monotone on a quota
  share, so the relabel survives there; XOL needs the engine. (Phase 1 plan, decision 7.)
- **`ContractTerms` taxonomy**: each feature is a frozen dataclass owning its φ, loss basis, and
  target leg. `ReinstatementTerms` (Phase 2); `RetroTerms`, `SwingTerms`, `SlideTerms`,
  `ProfitCommissionTerms`, `CorridorTerms` (Phase 3). Subclass prefix form per CLAUDE.md if a
  base class emerges (`ContractTerms*`).
- **`[pnl-share]`** (Phase 1): the leg-iterating table builder both `PnL` and the analysis call.

Execution order across plans: **Phase 1 → Phase 2 (reinstatements) → Phase 3.**
