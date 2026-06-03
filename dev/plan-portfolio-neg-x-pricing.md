# Plan (DRAFT): pricing & allocation on signed (P&L) support — `add_exa`, distortions, `value_type`

> **Status:** DRAFT — split out of `plan-negative-x-port.md` at its rev-1
> refresh. **Do not start until the combine iteration lands** (signed
> `loss`/`p_total`/`p_{line}`/`F`/`S` working, signed VaR/TVaR working). This plan
> covers everything that prices, allocates, or assumes `loss ≥ 0`.
>
> Author flags carried over: **the price column is the priority and must be
> *checked*, not assumed.** This is a **discovery-driven** plan — its central
> deliverable is a column-by-column audit produced by actually running a
> signed-support portfolio, not a guessed list.

---

## 1. Why this is separate

Combining densities on signed support is a clean deterministic convolution (the
combine plan). Pricing is not: distortions, conditional expectations, and capital
allocation all encode the actuarial **"loss, more-is-worse"** orientation and a
**non-negative** loss axis in dozens of places. Getting `p_total`/`F`/`S` right
does **not** make `add_exa` right. The two cadences should not be coupled.

---

## 2. The `add_exa` / `density_df` column audit ⚠ (primary deliverable)

`add_exa` (`portfolio.py:2122`) and `add_exa_details` (`:2251`) write the
allocation/pricing columns. The audit must enumerate **every** column they (and
`update`) emit and classify each as:

- **(a) already valid on signed support** — purely label-indexed cumulative
  quantities computed from the signed grid (`F`, `S`, and `p_*`, already handled
  in the combine plan);
- **(b) needs the signed grid but otherwise fine** — formula is correct once
  `loss` carries signed values and the FFT presentation is rolled;
- **(c) genuinely assumes `loss ≥ 0`** and needs re-derivation.

**Method (discovery):** build a small signed P&L portfolio (e.g. two `ssev`
lognormal-minus-shift lines, plus one ordinary loss line so the book straddles 0),
run `update(add_exa=True)` with the signed path *temporarily* enabled, and
inspect each column against a brute-force / simulation reference. Produce the
table **column → assumption → signed-support verdict → fix**.

Known suspects to scrutinise (verdicts TBD by the audit, not pre-judged):

- `exeqa_{line}` — `ift(ft(loss · p_i) · ft_nots[i]) / p_total` (`:2194–2197`).
  The `loss` weight is now signed; the `ft_nots` convolution must use the **same
  origin-at-0 / single-roll** convention as the combine plan (the F2 step), and
  the division by `p_total` must align bucket-for-bucket on the *rolled* grid.
  This is the **priority column** the author flagged.
- `exa_{line}` / `exa_total` — conditional-expectation / allocation integrals;
  check the direction of the tail integral when `loss < 0`.
- `lev` (limited expected value), `epd` (expected policyholder deficit) — both are
  built around a non-negative loss/limit ladder; semantics on a P&L need defining
  (is "deficit" measured from 0, from the mean, from a capital level?).
- The `loss_max` blanking heuristic and any `shift(-1, fill_value=...)` on the
  tail — these assume the heavy mass sits at the **high** end; on signed support
  the informative tail can be at either end.
- Any `loss`-weighted sum / cumulative that presumes positivity (`exi_x*`,
  `e2pri_*`, priority/`epd` ladders in `add_exa_details`).
- `_limits` (`:2027`) and the plot ranges — already partly addressed by the
  Aggregate `info`/window work; confirm at the portfolio level.

Tracked in `dev/TODO-Remember.md` (item 6, pricing half).

---

## 3. Distortion pricing & `value_type`

The Aggregate work introduced `value_type ∈ {loss, payoff}` (default `loss`) as an
**inert** member. Make it bite at the pricing layer:

- Distortions assume the **loss** orientation ("more is worse", price off `S`).
- A `payoff` object (asset / "more is better") must be reflected — negate the
  variable (or apply the distortion's dual) before pricing, then map results back.
  Decide one canonical mechanism (negation vs dual) and apply it consistently in
  `Distortion.price` / the augmented-df pricing path (`_build_augmented`,
  `portfolio.py:2595`).
- Open semantic questions to settle (possibly a short research note):
  - What do TVaR / distortion premium / capital allocation **mean** on a P&L whose
    support straddles 0? (Allocation of *what* — economic capital relative to 0,
    or to the mean?)
  - Does the lifted-natural-allocation admissibility story (`bounded` + mass,
    `portfolio.py:2636`) change when support straddles 0? The combine plan keeps
    `bounded` semantics unchanged; confirm the guard still reads correctly.
  - How `Distortion.price`'s forward/backward `S` handling behaves on signed `S`
    (it is monotone in `loss`, so the integral is well-defined, but the sign
    conventions of `exa`/premium need re-checking).
- Whether `value_type` gets a **DecL keyword** (deferred from the Aggregate plan)
  — decide here since this is where it first does anything.

---

## 4. Testing (sketch — firm up during audit)

- **`payoff` == negated `loss`:** a `payoff`-typed object prices as the negation
  of the equivalent `loss` object (round-trip consistency of `value_type`).
- **Allocation adds up:** `Σ_line exa_{line} == exa_total` on signed support at
  every loss level (the fundamental allocation identity), validated against the
  combine plan's `p_{line}` marginals.
- **`exeqa` vs simulation:** `exeqa_{line}(a)` matches a Monte-Carlo
  `E[X_i | X ≈ a]` on a signed book at several `a`, including `a < 0`.
- **Snapshot:** once columns are fixed, capture a signed-support `density_df`
  pricing snapshot for a small P&L portfolio.
- **Non-negative regression:** existing pricing tests/baselines unchanged.

---

## 5. Files (anticipated)

- `src/aggregate/portfolio.py` — re-enable the signed `add_exa` path (remove the
  combine plan's warn+fallback), fix the audited columns, signed `_limits`.
- `src/aggregate/spectral.py` — `value_type`-aware `Distortion.price` (or its
  dual) if that's where reflection lands.
- `src/aggregate/distributions.py` — make `value_type` actually consumed; possibly
  a DecL keyword (with `parser.py` / `decl.lark`).
- Tests + `dev/TODO-Remember.md` close-out of #6's pricing half.

---

## 6. Open questions

1. Reflection mechanism for `payoff`: negate-the-variable vs distortion-dual —
   pick one canonical path.
2. Meaning of capital allocation / EPD on a straddling P&L (reference point: 0,
   mean, or a chosen capital level?).
3. `value_type` DecL keyword: yes/no and spelling.
4. Whether any of `lev`/`epd` are simply **undefined** on signed support and
   should be blanked rather than re-derived.
