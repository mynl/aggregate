# docs/flow

Information-flow diagrams for `aggregate`, organized along the SUVA-Use path: **S**pecify, **U**pdate, **V**alidate, **A**djust, **U**se.

## How this directory is put together

Every diagram lives in its own `_*.md` fragment: a heading, a paragraph or two of prose, and one fenced mermaid block. The `.qmd` files are assemblies that pull fragments in with `{{< include >}}` and add nothing but front matter, part headings and citations. Reordering the flow means reordering include lines, never moving diagram source.

The leading underscore does three jobs at once. Quarto skips `_*` files when rendering a project, so fragments never render as standalone pages. It marks them as parts rather than documents. And it groups them together in a directory listing.

Fragments use the Quarto fence form:

    ```{mermaid}
    %%| label: fig-something
    %%| fig-cap: "Caption."
    flowchart LR
        ...
    ```

That one form works in all three renderers this repo cares about. Quarto treats it as a diagram cell and uses the `%%|` lines for the figure label and caption. Writedown accepts both ```mermaid and ```{mermaid}, and the `%%|` lines are ordinary mermaid comments so they are ignored. MyST reads ```{directive} as directive syntax, so the same block is a `mermaid` directive in Sphinx once `sphinxcontrib-mermaid` is installed.

## The fragments, in SUVA-Use order

| Fragment | What it shows |
|---|---|
| `_key-node-kinds.md` | The four node colors. No diagram, read it first |
| `_suva-use-cycle.md` | The five stages and the two loops back into Update |
| `_one-line.md` | DecL to engine to `density_df` to every answer |
| `_specify-decl-to-object.md` | Parser, recipe, knowledge base, live object |
| `_update-two-outputs.md` | `density_df` and the two halves of `stats_df` |
| `_update-grid-choice.md` | The bucket and window sizer, `bs_window_df`, `sharpen` |
| `_validate-derived-views.md` | `validation_df`, `summary_df`, the two tail objects |
| `_adjust-reinsurance.md` | Gross, ceded and net as three distributions |
| `_use-frames-consumers.md` | Pandas, charts and exhibits all read the frames |
| `_use-charts.md` | Emitter to `ChartDoc` to renderer |
| `_use-exhibits.md` | An exhibit names a frame, plus `Perspective` |
| `_use-pricing-inputs.md` | Distribution, capital rule, target, family |
| `_use-pricing-calibration.md` | `calibration_df` and `distortion_df` |
| `_use-pricing-augmented.md` | The one new frame pricing adds |
| `_use-pricing-readouts.md` | Six readouts off `augmented_df` |
| `_use-pricing-pentagon.md` | The accounting path, no distortion involved |
| `_use-pricing-agg-vs-port.md` | Why the two classes price differently |
| `_use-pricing-whole.md` | The pricing half on one page |
| `_core-whole.md` | Specify through Adjust on one page |
| `_style-block.md` | Page CSS for the Quarto wrappers. Not prose, do not include it in a fragment |

## The assemblies

| File | Audience |
|---|---|
| `flow.qmd` | The full document, every fragment, SUVA-Use ordered |
| `flow-overview.qmd` | The talk version: cycle, one line, key, core |
| `flow-pricing.qmd` | The Use stage for pricing, on its own |

Render one with:

    quarto render docs/flow/flow.qmd

Each produces a self-contained HTML file with mermaid inlined, so the output travels on its own.

## Adding a diagram

Write a new `_*.md` fragment with a `##` heading and one mermaid block, give the block a `fig-` label so it can be cross-referenced, then add one include line to whichever assemblies should carry it. Nothing else needs touching.

Keep the node colors consistent with `_key-node-kinds.md`: blue for declared inputs, orange for work, green for stored frames, purple for derived views. The `classDef` lines are repeated in each fragment on purpose, so that every fragment renders correctly on its own.

## Sphinx

These files are excluded from the Sphinx build as standalone pages, the same way `4_agg_language_reference/ref_include.rst` is. To pull a fragment into an `.rst` page:

    .. include:: ../flow/_suva-use-cycle.md
       :parser: myst_parser.sphinx_

The `:parser:` option is supported by the installed Sphinx, and `myst_parser` is already an enabled extension. Rendering the diagrams rather than showing their source additionally needs `sphinxcontrib-mermaid` in the `dev` extra and in `extensions` in `docs/conf.py`. Until that lands, the mermaid blocks in an included fragment will raise an unknown-directive error, so hold off on adding the includes.
