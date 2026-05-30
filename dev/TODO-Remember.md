# TODO / Remember

> Working list of pending work and things to think about, post v1.0 core-compute
> refactor. As items here move into the codebase, they migrate to PROGRESS.md
> (which expands as this shrinks).
>
> **Last updated: 2026-05-30** — after meta.8 closed the v1.0 plan.

---

## Deep dives (understand before changing)

1. **Portfolio update numerical mechanics** — the full `Portfolio.update` →
   `add_exa` split. Why this shape (per-line agg via `Aggregate.update_work`,
   then FFT recombine for `p_total`, then `add_exa` for the conditional
   expectations)? Trace every column written by `add_exa` and confirm the
   numerical convention is consistent end-to-end (forwards `S`, the
   `shift(-1, fill_value=...)` tail handling, the `loss_max` blanking
   heuristic with `mult ∈ {1, 10, 100}` — a hangover from the legacy code,
   probably wants a principled `F < k·eps` rule per portfolio plan D11).
   Goal: a docstring-grade understanding so the next bug or extension lands
   safely.

2. **Bounds numerical mechanics.** The `Bounds` class (IME 2022 methodology,
   `bounds.py`) — fixed 513-point binary `s_grid`, distortion clouds, weight
   frames. Today's `Bounds` works in isolation but `Portfolio.pricing_bounds`
   is `NotImplementedError` because aligning it to the new `s_grid` is
   non-obvious (see item below). Read end-to-end so the pricing-bounds rewrite
   can proceed from understanding, not pattern-matching.

---

## Parked (post-v1.0) — from `dev/done/plan-meta.md`

3. **Aggregate FI-1: negative `xs` / windowed FFT.** Allow severity (and
   therefore aggregate) to take negative values, so per-deal profit/loss
   distributions convolve. Severity is easy via shift; aggregate is hard at
   random `N` because `A_shifted = A + N·s` rather than `A + constant`. Clean
   for `fixed`-N (including `fixed`-1). Most-wished-for enhancement.

4. **Aggregate FI-2: integrated aliasing + movable window.** Same machinery
   as FI-1 from another angle — a movable, possibly-negative window of
   interest on the FFT grid.

5. **Portfolio FI-1: `Portfolio.pricing_bounds` rewrite.** Raises
   `NotImplementedError` as of 1.0.0a11. The old wiring assumed the dense
   `density_df.S` was the s-grid; new `Bounds` uses a fixed 513-point binary
   `s_grid` and the per-unit `exeqa_*` columns need to be interpolated onto
   it. Defer until the design is settled. **Author wants periodic reminders.**

6. **Portfolio FI-2: negative `xs` at the portfolio-combine level.** Easier
   half of #3 — combining already-computed unit aggregates is a deterministic
   sum, so a constant shift de-shifts cleanly. Doable without solving the
   within-unit random-frequency problem.

7. **Switcheroo harness case.** Add a `Port.Sample` case (hand-built or
   seeded sample) to the baseline once Portfolio sample work next surfaces,
   to keep the kappa-replacement path under regression.

---

## Docs / packaging

8. **New README** for the v1.0 launch — currently `README.rst` reads as a
   running release-notes draft for the refactor iterations. Rewrite for the
   stable v1.0 audience: what `aggregate` is, who it's for, install + the
   one-liner DecL example, pointers to docs.

9. **New CHANGELOG file** — extract the existing `README.rst` iteration notes
   into a proper `CHANGELOG.rst` (or `.md`) keyed by version, so the README
   can shed its release-notes role.

10. **Docs intro for v1.0.** A short orienting page at the top of the Sphinx
    docs that explains the v1.0 shift (linear default allocation, bounded
    detection, `allocation_method` member, forwards-`S` default, pentagon
    pricing columns, `DefectiveDistributionWarning`). Replace any legacy
    framing.

11. **Docs reference SLY-era grammar.** `docs/4_agg_language_reference/`
    describes the grammar in SLY's `@_` form. Should `include`
    `aggregate/decl.lark` directly, or call
    `aggregate.parser.grammar(add_to_doc=True)` which writes
    `docs/4_agg_language_reference/ref_include.rst`.

12. **Docstring style sweep.** `iman_conover.py` and `moments.py` use Sphinx
    `:param x:` style with many empty parameter slots; the rest of the
    codebase is NumPy style (per CLAUDE.md). Convert in one dedicated pass
    (~25 docstrings in moments + iman_conover, plus pockets elsewhere).
    Public surface first, private helpers second.

13. **`pedagogy.py` migrations.** Move the remaining figure generators in:
    `ft.py`'s `poisson_example`, `fft_wrapping_illustration`,
    `recentering_convolution`, `recentering_convolution_example`;
    `tweedie.py`'s `tweedie_illustration`. Goal: keep `ft.py` / `tweedie.py`
    focused on the API.

---

## Deferred designs

14. **DecL colorization** — `dev/tentative-plan-decl-colorization.md`. Design
    ready but parked (2026-05-27): IPython tracebacks don't call
    `_repr_html_`, so the payoff is mostly Sphinx-docs identity. Wait for a
    clearer use case.

---

## Done since this file opened

*(append as items land)*

- (2026-05-30) `xsden_to_meancv` / `xsden_to_meancvskew` tail-mass
  inconsistency — resolved as a side effect of meta.3 / D8: both now route
  through `xsden_to_mwrangler`, which places the tail mass at `xs[-1] + bs`
  identically. Was item #8 on the CLAUDE.md TODO list.
