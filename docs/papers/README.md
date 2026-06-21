# aggregate Example Library

Standing instructions for the example library — a collection of small, focused,
**executable** examples for `aggregate`. Each example states a problem, its setup
and assumptions, the DecL and Python that solve it, and the output — computed
live, not pasted.

Two flavors share one format:

- **Cookbook examples** illustrate a feature or technique.
- **Paper reproductions** reproduce a published worked example — the same atom
  plus a citation. A published example is an a priori problem with a known
  answer, and `aggregate` matches it; that's the strongest evidence the library
  is correct and fast.

> A cookbook example *becomes* a paper reproduction by adding a source citation
> and the Reproduction section. There is one taxonomy, not two.

This file is the process; `_template.qmd` is the starting point for each post.

## Layout

> This directory is `example_library/` (renamed from `papers/`). It is a
> standalone Quarto **website**, modeled on a blog: posts live under `posts/`.

```
example_library/
  _quarto.yml            # the Quarto website project
  index.qmd              # landing page; auto-lists the posts
  README.md              # these instructions
  _template.qmd          # copy to start a new post
  example_library.agg    # shared DecL knowledge base (assumptions, defined once)
  posts/
    <slug>/
      index.qmd          # one example
      img/               # cropped original exhibits (you supply), if any
```

One post per example. If a paper yields several examples, use one post with a
`##` section per example. Use short, lowercase, hyphenated slugs; for paper
reproductions, author + year is ideal (`wang-2000`).

## Workflow

For a **cookbook example**: copy `_template.qmd` to `posts/<slug>/index.qmd`,
delete the Reproduction section and any citations, and fill it in.

For a **paper reproduction**:

1. **You** create `posts/<slug>/`, drop the source PDF in it, and tell me
   (a) which example/exhibit to reproduce and (b) the BibTeX **citation key**.
2. **I** confirm the key exists in `C:/s/TELOS/Biblio/uber-library.bib`
   (read-only; I `rg` for the exact key, never guess). If missing, I list it in
   the run summary for you to add via `archivum`, then cite once it exists.
3. **I** read the PDF, transcribe the published model and results exactly, and
   fill the template — set the `subtitle` to name the source, add `paper` to
   `categories`, and fill the Reproduction section.
4. **You** supply any cropped original exhibits as PNGs in `<slug>/img/`.

## The atom

One example = one `index.qmd` with front-matter metadata:

```yaml
title: "..."
subtitle: "Reproducing Wang (2000), Table 1"   # paper reproductions only
description: "one line for the listing page"
date: 2026-06-21
categories: [paper, reinsurance]               # tags drive listing + filtering
```

Treat the front-matter as a future single source of truth: clean metadata now
keeps the door open to auto-generating index pages, filtered chapters, or a
test manifest later.

## Reuse of assumptions — the shared `.agg` + Underwriter pattern

Shared assumptions live **once** in `example_library.agg` as named DecL objects
(severities, frequencies, books). Each post loads them into its own
`Underwriter` and references them by name:

```python
from aggregate import Underwriter, qd
uw = Underwriter(databases=None)     # start empty
uw.load('example_library.agg')       # resolved from the project root (execute-dir: project)
a = uw.build('agg MyBook agg.SampleBook occurrence net of 50 xs 50')   # reference, rename, scale, layer
```

DecL references (`sev.Name`, `agg.Name`), renaming, scaling, and adding builtins
are native — this is what the knowledge base is *for*. Change an assumption in
`example_library.agg` and every post that references it updates; no retyping, no
drift.

**Principle — DRY in the source, explicit in the render.** Reuse the *definition*
by name, but `qd()` the referenced object on the page so the reader still sees
the assumptions even though you didn't retype them:

```python
qd(uw.build('agg.SampleBook'))   # show what the shared object actually is
```

Keep `example_library.agg` small and curated — only genuinely shared objects.
One-off models stay inline in their post.

## House style

Match the existing reproductions (`docs/2_user_guides/problems/*.rst`,
`docs/5_technical_guides/5_x_bodoff.rst`); only the markup changes (qmd, not rst).

- **Display.** Use `qd(...)` for every frame/series — never bare `print` or a raw
  repr. Pass `accuracy=` when more digits help.
- **Build, don't hand-roll.** Express the model in DecL via `uw.build('...')`;
  drop to Python only for post-processing, comparison frames, and plots.
- **Voice.** Plain, declarative, US spelling (behavior, not behaviour). State the
  result plainly when it matches — no hedging.
- **Naming.** Full words for new identifiers (soft default; house abbreviations
  stay). `reins` is canonical — never expand to `reinsurance`.
- **Output density.** Tight rendered blocks: no gratuitous blank lines;
  concatenate short labels onto the line they describe.
- **Severity choice.** Thick / unlimited → `slognorm`; bounded / thin →
  `sgamma`; if in doubt, `slognorm`.
- **Math.** Inline `$...$`, display `$$...$$`; define notation on first use.

## Citations (paper reproductions)

- Bibliography and CSL are set once in `_quarto.yml` (master library +
  Journal of Risk and Uncertainty style); posts inherit them.
- Cite inline with `@Key` (textual) or `[@Key]` (parenthetical); group with
  `[@Key1; @Key2]`. End a citing post with `## References` over an empty
  `::: {#refs}` div.
- Keys follow `AuthorYYYY[a-z]`. I `rg` the bib for the exact key; missing keys
  go in the run summary for you to add via `archivum` — I never fabricate a key
  or edit the library.

## Images

You supply cropped original-paper exhibits; I transcribe numbers.

- **Numeric tables → markdown / a `published` column.** Reproduces cleanly, diffs
  in git, renders everywhere.
- **Figures / busy exhibits → cropped PNG** in the post's `img/`, referenced with
  a citing caption:
  ```markdown
  ![Original results table, @Wang2000.](img/wang2000_table1.png){width=80%}
  ```
- **What I can/can't do with PDFs.** I can read a PDF and transcribe any value
  exactly. I cannot mouse-crop a region; I *can* render pages or pull embedded
  images programmatically (`pymupdf`), but crops are approximate — your hand
  crops are cleaner, so that's the default.
- Paste only what's needed for comparison, and attribute via the caption.

## Cross-referencing

- Give each example a stable, slug-based section id (`#sec-tail-allocation`)
  decoupled from any ordering number, so reordering never breaks a link.
- Quarto resolves cross-file `@sec-`/`@tbl-`/`@fig-` refs only in the **website
  build**, not in a standalone single-file render. So: render one post for fast
  iteration; render the site for working cross-references and the bibliography.

## Getting examples into the Sphinx (`rst`) docs

The example library is a **standalone Quarto website** you host yourself; the
main Sphinx docs **link to it**. When you also want an example to live *in* the
Sphinx tree, there are two conversion paths — both verified to work. The `qmd`
is always the single source of truth; never maintain a hand-edited parallel rst.

**1. Automated static snapshot (recommended).** Quarto renders a post to rst
with code, *captured* output, and extracted figures:

```
uv run quarto render posts/<slug>/index.qmd --to rst
```

This produces clean rst — `.. code:: python` blocks, literal output blocks, and
`.. image::` for plots (extracted to an `index_files/` folder). Treat the rst as
a **generated artifact**: never hand-edit it, and regenerate on rebuild so it
can't drift from the qmd. It does *not* re-execute under Sphinx (so the Sphinx
build needs no `aggregate` env), and it is *not* re-verified there — the Quarto
site build is the verification. Quarto wraps each cell in a `.. container:: cell`
div; harmless in Sphinx, and easy to strip in post if you want barer output.

**2. Live ipython-directive port.** To have Sphinx *re-execute* the example
(matching the existing docs, always current), port it to `.. ipython:: python`
using the map below. More faithful and self-verifying, but more maintenance, and
the cited keys must also exist in the Sphinx bib (`docs/extract.bib` /
`docs/books.bib`), not just `uber-library.bib`.

| qmd | rst |
|---|---|
| ` ```{python} ` … ` ``` ` | `.. ipython:: python` + `:okwarning:`, body indented |
| `# Title` / `## H2` / `### H3` | `===` / `---` / `~~~` underlines |
| `![alt](img/x.png){width=80%}` | `.. image:: img/x.png` + `:width: 800` |
| a figure-producing cell | add `@savefig name.png scale=20` before the plotting line |
| `@Key` / `[@Key]` | `:cite:t:` `` `Key` `` / `:cite:p:` `` `Key` `` |
| `$x$` / `$$…$$` | `:math:` `` `x` `` / `.. math::` |

## Building

Render with the project env so the Jupyter kernel has `aggregate`:

```
uv sync --extra notebook      # once: provides jupyter
uv run quarto preview         # live reload while authoring
uv run quarto render          # build the whole site to _site/
```

`execute-dir: project` (set in `_quarto.yml`) means `uw.load('example_library.agg')`
resolves from the project root for every post. `freeze: auto` caches executed
output so unchanged posts don't re-run.

## Checklist per example

- [ ] Slug folder under `posts/`; clean front-matter (title, description,
      date, categories).
- [ ] Setup loads the shared `example_library.agg`; shared references `qd()`-ed.
- [ ] Model expressed in DecL via `uw.build(...)`; `qd` used for all output.
- [ ] **Paper:** citation key confirmed (or flagged for archivum); subtitle set;
      `paper` category; published vs aggregate shown side by side and matching.
- [ ] Original exhibits (if any) cropped into `img/` with citing captions.
- [ ] `## References` + `::: {#refs}` present iff the post cites.
- [ ] Renders clean under `uv run quarto preview`.
