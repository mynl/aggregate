# TODO

> Two lists: (1) must-do before v1.0 ships, (2) ideas and post-v1.0
> considerations. Snappy entries only; details live in the plan files
> (`dev/`, `dev/done/`) and the git log (one commit per a-iteration).
> What's landed is in `PROGRESS.md`.
>
> **Last updated: 2026-06-04** — current version 1.0.0a28.

---

## DOD

- [ ] Docs updated
    - [ ] new features
    - [ ] new grammar
    - [ ] new syntax
    - [ ] removed features
    - [ ] Journey for v1.0
    - [ ] statements of philosophy: user manages logging, user manages warnings and matplotlib setup; the distribution **is** pi at xi (not a cts approx thereunto - can't detect jumps); user manages fancy pd.DataFrame formatting - ``qd`` is an exception for docs fixed font print
- [ ] Numerical calculations checked
- [ ] Numerical methods updated for negative xs
- [ ] Sensible test suite trimmed for go-forward development


- [ ] NUMERICS: Review Severity numerical moments calc - it works but can it be improved - change with zero blast radius
- [ ] NUMERICS: Review Aggregate main computation loop - the P. M. Q. etc elements
- [ ] NUMERICS: Review Portfolio numerics... (split from interface) (ditto)
- [ ] NUMERICS: what does plot_twelve actually need
- [ ] NUMERICS: Review Bounds numerics
- [ ] NUMERICS: implications of negative xs
- [ ] NUMERICS: Portfolio.pricing_bounds (possible duplicate)
- [ ] **alpha**? Ability to do approximate somehow (and pick gamma lognormal based on tail) (again, since functin removedoality removed)
- [ ] **alpha**?Pedagogy to reproduce G & H tilting example (since that was removed)
- [ ] **alpha**?move make_ceder_netter  from ultilities to distributions - why is it where it is?
- [ ] **alpha**? why utilities make_var_tvar and a version in distributions
- [ ] **alpha**? Review all import dependencies for small non-standard packages. eg cycler in graphics?

- [ ] New README and split out CHANGELOG.md
- [ ] PUNCHUP pedagogy and integrate with docs - possible minor renamings. OK to push into beta
- [ ] DOCS: tail descriptor testing and docs (bounded, log concave, super expon, expon, sub expon) for f and s
- [ ] DOCS:  bounded / unbounded indicator
- [ ] 10000 xs 0 lognorm 120 cv 1.5 triggers sum sev < 1 warning? --> is it for agg not sev
- [ ] DECL: ``of`` in place or as well as ``po`` / ``so`` (and or spell them out)
- [ ] Reins structure diagrams? Use PMIR code?
- [ ] Gross -> subject for agg only in desc.
- [ ] Port pmir code w best bucket and the clever manual kappa calculation? (minor add, ok beta)
- [ ] Can we avoid the ugly cts histogram with small spikes if we use linear/nearest? Was it all about stats? WHAT IS THIS??? I forget...
- [ ] IMPORTANT: review validation calc; all switches config; docs reflect actual algo (compare published Aggregate paper); validation fix: fails agg mean error >> sev, possible aliasing; try larger bs which is often a bit annoying and maybe too tight tolerance.
- [ ] built in test_suite library - combine the three libraries we have into one - with out breaking tests (see next)
- [ ] **TESTS** inspect tests for needed / no longer needed. ratioalize without decreasing effectiveness. Integrate better with new single test_suite. 


## 1. TODO before v1.0 ships

### Features added from my personal list

- [ ] Docs: custom errors
- [ ] Docs: syntax checker and better error reporting
- [ ] Docs: check for refs to "site" database (2_user_guides/2_x_case_studies.html)
- [ ] Docs: check and examples for ZT and ZM (zero truncation...)
- [ ] Docs: splice examples
- [ ] pricing_at method needs a = P + Q and consistent Pentagon output
- [ ] ?better Pentagon integration? Output to Pentagon?

### Features in flight

- [x] (1) **Tail-thickness classifier.** Implementation in flight (`tail.py`,
   `tests/test_tail.py` in the working tree); plan in
   `dev/done/plan-tail-thickness.md`.
- [ ] (2) **Portfolio negative-x pricing half.** `dev/plan-portfolio-neg-x-pricing.md`
   (DRAFT): the `add_exa` column audit, distortion pricing, `value_type`
   consumption. Signed books warn + fall back to F/S-only until this lands.
- [ ] (3) **Negative-x deferred follow-ups.** Two-sided deficit split; `ft.py`
   recentering helpers → call the core path; re-home `estimate_agg_window` to
   `utilities.py`; occ-reins on a signed severity grid; DecL keyword for
   `signed`/`value_type`.
- [ ] (4) **Multivariate later stages.** Stage 2 `t` copula; Stage 3 reporting/plot
   polish; Stage 4 ≥3-variate `rfftn` shared frequency; Stage 5
   `MultivariatePortfolio` / `netceded` DecL form. See
   `dev/done/plan-multivariate.md`. Better implementation of window.
- [ ] (5) **Gross/ceded-premium reinsurance P&L.** Extend `pnl` with both premium
   legs so the gross/ceded/net loss views become parallel P&L views. Split
   out of `plan-pnl-premium.md` §9.

### Windows and plotting

- [ ] (6) **Support-aware window bounds.** Use `fz.support()` lower/upper endpoints
   in the window estimator; for finite frequency the aggregate support is
   exactly `[N·loc, N·ub]` — an exact window instead of a MoM guess.
- [ ] (7) **Window bounds for bivariate.** Apply the window estimation machinery to
   the bivariate/multivariate per-axis sizing.
- [ ] (8) **Plot severity outside the aggregate window.** When the windowed
   aggregate and severity grids don't overlap, how to render severity in
   `plot` — inset, broken axis, or separate figure? `info` already warns.

### Numerics and pricing deep dives

- [ ] (9) **Portfolio update numerics.** Trace `Portfolio.update` → `add_exa`
   end-to-end (every column, the `shift(-1)` tail handling, the `loss_max`
   blanking heuristic that wants a principled `F < k·eps` rule).
- [ ] (10) **Bounds numerics.** Read `bounds.py` (IME 2022, 513-point binary
    `s_grid`) end-to-end before attempting the `pricing_bounds` rewrite.
- [ ] (11) **`Portfolio.pricing_bounds` rewrite.** `NotImplementedError` since a11;
    needs `exeqa_*` interpolated onto the new `s_grid`. **Author wants
    periodic reminders.**

### Testing

- [ ] (12) **Switcheroo harness case.** Add a `Port.Sample` case to the baseline when
    Portfolio sample work next surfaces, to regression-guard the
    kappa-replacement path.

### Docs and packaging

- [ ] (13) **New README.** Rewrite `README.rst` for the stable v1.0 audience (what,
    who, install, one-liner DecL example); it currently reads as release
    notes.
- [ ] (14) **CHANGELOG file.** Extract the README iteration notes into a proper
    `CHANGELOG.rst` keyed by version.
- [ ] (15) **Docs intro for v1.0.** Short orienting page on the v1.0 shift (linear
    allocation default, bounded detection, forwards-`S`, pentagon columns,
    `DefectiveDistributionWarning`); replace legacy framing.
- [ ] (16) **Reinsurance case-study docs rewrite.** Three `docs/2_user_guides/problems`
    case studies (bahnemann, enterprise risk, other_misc) were left as
    migration notes after a19; rebuild their per-layer exhibits directly from
    `reins_stats_df` and verify against published references.
- [ ] (17) **Grammar reference docs.** `docs/4_agg_language_reference/` still
    describes the SLY-era grammar; switch to including `decl.lark` /
    `grammar(add_to_doc=True)` output.
- [ ] (18) **Docstring style sweep.** Convert `iman_conover.py` / `moments.py` (and
    pockets elsewhere) from Sphinx `:param:` style to NumPy style; public
    surface first.
- [ ] (19) **`pedagogy.py` migrations.** Move the remaining figure generators out of
    `ft.py` and `tweedie.py` so those modules stay API-focused.

---

## 2. Ideas to consider

- [ ] (20) **Multi-resolution portfolio combine.** The real fix for the coarse
    shared-`bs` deficit (a22 residual #4): compute each unit on its own `bs`,
    decimate onto the shared grid before the Fourier product. Out of scope for
    now; the deficit is accepted and surfaced.
- [ ] (21) **General premium/loss algebra in DecL (v2.0).** Constant aggregates,
    full aggregate arithmetic (`agg.A - agg.B`, `agg.A + c`); `pnl` covers the
    common case for v1.0.
- [ ] (22) **DecL colorization.** Design parked 2026-05-27
    (`dev/tentative-plan-decl-colorization.md`); payoff is mostly Sphinx-docs
    identity. Wait for a clearer use case.
