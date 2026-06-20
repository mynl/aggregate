# Reproducing published worked examples

Standing instructions for adding reproductions of published actuarial examples
to the `aggregate` documentation. The point of these pages is *proof*: a
published example is an a priori problem with a known answer, and `aggregate`
reproduces that answer. Matching the literature is the strongest evidence the
library is correct and fast.

This file is the process; `_template.qmd` is the starting point for each page.

## Workflow (how we collaborate)

1. **You** create a subfolder of `docs/papers/`, drop the source PDF in it, and
   tell me (a) which example or exhibit to reproduce and (b) the BibTeX **citation
   key**.
2. **I** confirm the key exists in the master library
   (`C:/s/TELOS/Biblio/uber-library.bib` — read-only; I `rg` for the exact key,
   never guess). If it is missing I list it in the run summary for you to add via
   `archivum`, then cite once the key exists.
3. **I** read the PDF, transcribe the published model and results exactly, and
   fill out `_template.qmd` into `<subfolder>/index.qmd`.
4. **You** supply any cropped original-paper exhibits as PNGs in
   `<subfolder>/img/` (see [Images](#images)).
5. **I** emit the Sphinx `.rst` companion (see [Two outputs](#two-outputs-qmd-and-rst)).

## Folder layout

Each paper is self-contained:

```
docs/papers/
  README.md              # these instructions
  _template.qmd          # copy to start a new example
  <paper-slug>/          # one folder per paper, e.g. wang-2000/
    <paper>.pdf          # the source PDF (you drop in)
    index.qmd            # the reproduction (authored from the template)
    index.rst            # emitted for the Sphinx build
    img/                 # cropped original exhibits (you supply)
```

Use a short, lowercase, hyphenated `<paper-slug>` — author + year is ideal
(`wang-2000`, `bahnemann-2015`). If one paper yields several examples, use one
`index.qmd` with a `##` section per example rather than many folders.

## The template sections

`_template.qmd` carries the agreed skeleton. What goes in each:

| Section | Content |
|---|---|
| **Overview** | The business problem, the source paper (cite with `@Key`), and exactly what we reproduce. One short paragraph. |
| **Published Model** | Frequency, severity, exposure, contract terms, assumptions — transcribed *exactly* as the paper states them. Use a markdown table for parameters; paste the original spec image if it clarifies. |
| **Aggregate Implementation** | Executable DecL inside `build('...')`, plus any Python. This is the heart of the page — keep the DecL clean and readable. |
| **Reproduction** | Published values vs. aggregate values, side by side. Transcribe the paper's numbers into a `published` column and put aggregate's beside it. Comparison is visual — **no asserts** on the page. |
| **Distributional Analysis** | Full gross, ceded, and net distributions — the value `aggregate` adds beyond the published point answers. |
| **Extensions** | Additional structures, sensitivities, risk measures, or allocations the paper did not compute but the model now makes free. |
| **Conclusions** | What the example demonstrates about `aggregate`. |

## House style

Match the existing reproductions (`docs/2_user_guides/problems/*.rst`,
`docs/5_technical_guides/5_x_bodoff.rst`). The prose is the same; only the
markup changes (qmd, not rst).

- **Imports and display.** Open with `from aggregate import build, qd`. Use
  `qd(...)` ("quick display") for every frame/series — never bare `print` or a
  raw repr. Pass `accuracy=` when more digits help the comparison.
- **Build, don't hand-roll.** Express the model as DecL in `build('...')`
  whenever the language can say it. Drop to Python only for post-processing,
  comparison frames, and plots.
- **Voice.** Plain, declarative, US spelling (behavior, not behaviour). State
  what is reproduced and confirm the match plainly — e.g. "This table is
  identical to the table in the paper." No hedging when it matches.
- **Naming.** Full words for new identifiers (soft default; house abbreviations
  stay). `reins` is canonical — never expand it to `reinsurance`.
- **Output density.** Keep rendered blocks tight: no gratuitous internal blank
  lines, concatenate short labels onto the line they describe.
- **Severity choice.** When approximating a severity, follow the house rule:
  thick / unlimited → `slognorm`; bounded / thin → `sgamma`; if in doubt,
  `slognorm`.
- **Math.** Inline `$...$`, display `$$...$$`. Define notation the first time it
  appears, as the paper does.

## Citations

Per the project standing order, authored pages cite the master library.

- Every `index.qmd` YAML carries:
  ```yaml
  bibliography: C:/s/TELOS/Biblio/uber-library.bib
  csl: C:/s/TELOS/Biblio/journal-of-risk-and-uncertainty.csl
  ```
- Cite inline with `@Key` (textual: "as @Wang2000 shows") or `[@Key]`
  (parenthetical); group with `[@Key1; @Key2]`.
- End every citing page with a `## References` heading over an empty
  `::: {#refs}` div.
- Keys follow `AuthorYYYY[a-z]`. I always `rg` the bib for the exact key; if
  it's missing it goes in the run summary for you to add via `archivum` — I do
  not fabricate keys or edit the library.

## Images

You supply cropped original-paper exhibits; I transcribe numbers.

- **Numeric tables → markdown.** I transcribe the paper's published values into a
  native markdown table (or a `published` column in the comparison frame). This
  reproduces cleanly, diffs in git, and renders in both HTML and Sphinx.
- **Figures / busy exhibits → cropped PNG.** Put a hand-cropped PNG in the
  example's `img/` folder; reference it with a caption that cites the source:
  ```markdown
  ![Original results table, @Wang2000.](img/wang2000_table1.png){width=80%}
  ```
  This mirrors `010_gh_example.rst`, where `aggregate` regenerates the figure and
  the cropped original sits beside it for visual proof.
- **What I can and can't do with PDFs.** I can read the PDF (rendered pages) and
  transcribe any table or value exactly. I cannot mouse-select and crop a region;
  I *can* render pages or pull embedded images programmatically (`pymupdf`), but
  crops are approximate and need iteration — your hand crops are cleaner, so
  that's the default.
- **Licensing.** Paste only what's needed for comparison (one table/figure), and
  always attribute via the caption citation.

## Two outputs: qmd and rst

The page is authored once in `.qmd` and also emitted as `.rst` for the existing
Sphinx build.

- **qmd** is the source of truth. Executable cells are fenced ` ```{python} `;
  Quarto runs them live (jupyter engine) so the published-vs-aggregate match is
  computed at render time, not pasted.
- **rst** is mechanically derived for Sphinx, using the `ipython` directive that
  the existing docs use. Register the new page in a Sphinx `toctree` (simplest:
  add it under `docs/2_user_guides/2_x_problems.rst`).

Conversion map (qmd → rst):

| qmd | rst |
|---|---|
| ` ```{python} ` … ` ``` ` | `.. ipython:: python` + `:okwarning:`, body indented |
| `# Title` / `## H2` / `### H3` | `===` / `---` / `~~~` underlines |
| `![alt](img/x.png){width=80%}` | `.. image:: img/x.png` + `:width: 800` |
| a figure-producing cell | add `@savefig name.png scale=20` before the plotting line |
| `@Key` | `:cite:t:` `` `Key` `` |
| `[@Key]` | `:cite:p:` `` `Key` `` |
| `$x$` / `$$…$$` | `:math:` `` `x` `` / `.. math::` |

Two reconciliations to watch when emitting rst:

1. **Bibliography.** The qmd cites `uber-library.bib`; the Sphinx build resolves
   keys from `docs/extract.bib` / `docs/books.bib`. If a key isn't in the Sphinx
   bib yet, it must be added there too — flag it.
2. **Figures.** Sphinx needs a real `@savefig` line and the `savefig`/`img`
   paths the existing docs use.

## Building

- **qmd → HTML** (executes aggregate live; needs the notebook extra for jupyter):
  ```
  uv run quarto render docs/papers/<paper-slug>/index.qmd
  ```
- **rst → Sphinx**: built by the author with the normal `docs` build — **do not**
  run the full doc build in an iteration loop (it's slow; see the project
  CLAUDE.md). Keep rst in lockstep with the qmd and note that a rebuild is
  pending.

## Checklist per example

- [ ] Citation key confirmed in `uber-library.bib` (or flagged for archivum).
- [ ] Published model transcribed exactly; parameters in a markdown table.
- [ ] DecL in `build('...')` reads cleanly; `qd` used for all output.
- [ ] Reproduction shows published vs. aggregate side by side and they match.
- [ ] Distributional analysis covers gross / ceded / net where applicable.
- [ ] Original exhibits (if any) cropped into `img/` with citing captions.
- [ ] `## References` + `::: {#refs}` present.
- [ ] `.rst` emitted, registered in a toctree, bib keys reconciled, rebuild noted.
