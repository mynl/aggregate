# Plans considered and rejected

A standing log of design ideas that were drafted, weighed, and **deliberately not
done** — kept so the reasoning survives and the same idea isn't re-proposed cold.
Each entry records what was proposed, why it was rejected, and what we do instead.
Append new entries at the top.

---

## Unified DecL colorization (rejected 2026-06-19)

**Origin.** `dev/tentative-plan-decl-colorization.md` (drafted, then DEFERRED
2026-05-27; retired here). One canonical `palette.py` dict feeding four renderers
of DecL: a Pygments style for Sphinx docs code blocks, an ANSI renderer for
`ErrorReport.render()` in a TTY, an HTML `_repr_html_` box for `ErrorReport` in
Jupyter, and the existing matplotlib style. Required promoting `style.py` to a
5-file package, a new Pygments `Style` class + `pyproject` entry point, a
hex→xterm-256 ANSI module, an HTML span module, `ErrorReport` changes, ~8 tests,
and a docs page.

**Why rejected.** The whole payoff is **aesthetic, not load-bearing** — nothing
computes differently — and the three "wins" are structurally weaker than they look:

1. **Jupyter (the surface everyone pictures) doesn't fire automatically.**
   IPython's traceback formatter calls `__str__`, not `_repr_html_`, so the styled
   HTML box only appears if the user *manually* surfaces `e.report`. The most-built
   renderer has the least automatic payoff.
2. **Docs (the only always-on win) has a scoping caveat the plan didn't budget.**
   Sphinx's `pygments_style` is **global** and the AggLexer emits ordinary token
   types (`Keyword`, `Number`, …), so a custom `AggregateStyle` restyles *every*
   code block in the docs (Python included), not just `.. code-block:: agg`.
   Scoping the palette to DecL alone is *more* work than the plan assumed, not less.
3. **The one clean win is marginal.** ANSI coloring of `ErrorReport` only shows on
   a rare event (a parse error, in a TTY), coloring a one-line caret + suggestion.

Against that: a package refactor, an entry point, two permanent color sources
(`palette.py` deliberately *not* unified with `aggregate.mplstyle`), plus drift to
fix (`render()` was rewritten since drafting). Cost/benefit is upside-down.

**What we do instead.** Nothing — DecL keeps the matplotlib styling it already has;
`ErrorReport.render()` stays plain text. **If docs identity ever becomes a stated
priority**, do only the *minimal slice* (palette dict + Pygments style + entry
point, one or two files — no package promotion, no ANSI/HTML, no `ErrorReport`
changes), and decide the global-restyle question deliberately at that point.

---

## Matplotlib graphics into `config.toml` (rejected 2026-06-19)

**Origin.** Phase 2 of `dev/plan-config.md`. Phase 1 (the config machinery —
`~/.aggregate/config.toml`, the defaults→file→env→kwargs cascade, `[build]` /
`[discretization]` / `[validation]` settings) shipped at a30. Phase 2 was to add
a `[plotting]` section enumerating `fig_w` / `fig_h`, `font_size`, `legend_font`,
`plot_face_color`, `figure_bg_color`, plus a `style = "my.mplstyle"` override
resolved against `~/.aggregate`, repointing the `FIG_*` / `FONT_SIZE` / colour
constants out of `constants.py` into `get_settings().plotting.*`.

**Why rejected.**

1. **It duplicates matplotlib's own mechanism.** Nearly every proposed key *is*
   an rcParam: `fig_w`/`fig_h` → `figure.figsize`, `font_size` → `font.size`, the
   face/background colours → `axes.facecolor` / `figure.facecolor`. Enumerating
   them as first-class TOML keys creates two places to set the same value and an
   ambiguous precedence against any loaded mplstyle. matplotlib already solves
   user restyling with `mplstyle` / `rcParams` / `matplotlibrc`.
2. **Real refactor cost with a footgun for little gain.** The `FIG_*` constants
   live in *default-argument expressions* (`def plot(..., figsize=(FIG_W, FIG_H))`),
   evaluated at import. Moving them to `get_settings()` forces converting every
   such signature to `figsize=None` + resolve-in-body across every plot method —
   mechanical, broad, and a classic half-done-default-arg hazard.
3. **Cosmetic and low-urgency.** The config feature's value (unifying `log2`,
   databases, bucket types, validation tolerances; killing the `constants.py`
   junk drawer) was already delivered in Phase 1. The plotting surface added
   surface area and a maintenance overlap without a matching payoff.

**What we do instead.** The plotting constants (`FIG_W` / `FIG_H`, `FONT_SIZE`,
`LEGEND_FONT`, `PLOT_FACE_COLOR`, `FIGURE_BG_COLOR`) **stay module-level in
`constants.py`** (which is also why they were allowed to stay there in Phase 1 —
the default-argument constraint). The bundled `aggregate.mplstyle` remains
package data loaded via `importlib.resources`; users who want to restyle figures
use matplotlib's native `mplstyle` / `rcParams` directly. `aggregate` exposes
**no** plotting-specific config surface.

> Note: this rejects only the *graphics* half of plan-config Phase 2. The
> remaining Phase 2 item — the numerics-pending validation floors
> (`aliasing_ratio`, `exeqa_noise_floor`, `ft_noise_floor`), now unblocked by the
> completed numerics review — is still live in `dev/plan-config.md`.
