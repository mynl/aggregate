# New material

New documentation that needs integrating into the docs.

These pages were drafted in `dev/` while the code they describe was being
written, and were moved here on 2026-07-29 so they sit beside the tree they
belong to. They are finished prose, not scratch notes, but none of them is
wired into a `toctree` yet, so nothing links to them and Sphinx does not build
them as part of the site.

### Sphinx pages (`.rst`)

| File | Subject | Likely home |
|---|---|---|
| `bucket-selection.rst` | Automatic grid selection: `bs`, `log2`, `x_min` | `docs/5_technical_guides/` |
| `info-strings.rst` | The `info` string contract shared by the first-class classes | `docs/5_technical_guides/` or the API reference |
| `pipeline-aggregate.rst` | The `Aggregate` computation pipeline | `docs/5_technical_guides/` |
| `pipeline-portfolio.rst` | The `Portfolio` computation pipeline | `docs/5_technical_guides/` |
| `pipeline-reinsurance.rst` | The reinsurance reporting surface | `docs/5_technical_guides/` |
| `tests.rst` | A friendly guide to testing `aggregate` | `docs/` top level, or `dev/` facing |
| `underwriter.rst` | The `Underwriter` and recipes | `docs/2_user_guides/` |

### Markdown pages (`.md`)

| File | Subject | Likely home |
|---|---|---|
| `reinstatements.md` | Property catastrophe reinsurance with reinstatements, the math | `docs/5_technical_guides/`, or cookbook §5 |
| `distribution-types.md` | Laws, types, and the `scipy.stats` shape / loc / scale paradigm | `docs/2_user_guides/`, or cookbook §2 |
| `task-features.md` | **Not a page.** The maintenance spec for `features.qmd` | stays with `features.qmd`; do not publish |

`task-features.md` is a process document, not reader-facing material. It travels
with `features.qmd` because it defines the coverage-ledger contract, the cast
design brief and the execution gate that keep that notebook honest. If
`features.qmd` graduates into the docs, this file goes back to `dev/`, or is
deleted once the notebook stops being maintained.

`distribution-types.md` carries Quarto YAML front matter (`bibliography:`,
`csl:`), so it is really a Quarto page with a `.md` extension. Both use `$…$`
math, which MyST renders through `dollarmath` (already enabled) but which needs
checking against the `.rst` math role if either converts.

### Quarto pages (`.qmd`)

| File | Subject | Likely home |
|---|---|---|
| `intro-1min.qmd` | `aggregate` in one minute | a landing page, or the README |
| `intro-5min.qmd` | `aggregate` in five minutes: declare, build, validate, trust | `docs/2_user_guides/`, or the cookbook front matter |
| `intro-20min.qmd` | `aggregate` in twenty minutes | `docs/2_user_guides/` |
| `features.qmd` | What changed in 1.0 vs 0.30.1, by subject, with worked examples | its own chapter, per §1 of `task-features.md` |

`features.qmd` is the big one: a runnable notebook covering the whole `1.0.0a*`
series, a1 through a168, with a 168-row coverage ledger reconciling it against
`CHANGELOG.md`. It is **current as of a168** and every cell executes clean
(70 code cells, 0 errors, 2026-07-29). Re-verify with the §4 gate in
`task-features.md` before publishing, since it drifts as the API moves.

The three intros are a graduated set and read as one sequence, so place them
together. They are Quarto, not Sphinx, which is a decision in itself: either
they render alongside `docs/cookbook/` under Quarto, or they convert to `.rst`
to join the Sphinx tree. `intro-20min.qmd` also pins an absolute interpreter
path in its front matter (`wd-python`, pointing into this worktree's `.venv`),
which has to go before it moves anywhere shared.

The "likely home" column is a suggestion, not a decision. Integrating a page
means picking its section, adding it to that section's `toctree` or
`_quarto.yml`, and checking its cross-references resolve against the current
API.

## Before this directory joins a build

`docs/conf.py` does **not** exclude `new_material/`, and both `.rst` and `.md`
are live source suffixes there (`myst_parser` is enabled). A Sphinx build will
therefore pick up the seven `.rst` files, the two `.md` pages and this README as
orphan documents and warn about each one. The `.qmd` files are not affected,
since `.qmd` is not a Sphinx source suffix. Either add `'new_material'` to
`exclude_patterns` in
`docs/conf.py` while the material waits, or integrate the pages and delete this
directory. The same fix is already pending for `docs/cookbook/`.
