Introduction
==============

This folder contains **reproductions**: worked examples using ``aggregate`` to reproduce figures, tables, and results from published papers.

The pages are assembled into a Quarto book. ``_quarto.yml`` defines the chapter order and the two output formats, ``index.qmd`` is the preface, and ``references.qmd`` collects the bibliography once for the whole book. Build with ``quarto render`` from this folder; output lands in ``_book``. This file and ``plan-examples.md`` are working notes and are deliberately not chapters.

Each page follows the same shape: context and problem, the paper's own solution and gold standard, the ``aggregate`` reproduction, and comments. A page named for a paper that turned out **not** to be reproducible says so and gives the evidence.

Actuarial papers
-------------------

* `Venter1983.qmd` — transformed beta and gamma; Exhibit 3, all three columns. The bucket size selects which of Venter's two exact methods you get.
* `Mack2003.qmd` — Riebesell's increased limits rule under the collective risk model; Table 1 and the threshold above which the rule is exact.
* `Mata2005.qmd` — excess trend and exposure adjustment by layer; Tables 1 to 14, and a discrepancy in the paper's own Table 1.
* `Bruno2006.qmd` — thresholded direct convolution; Tables 6, 7 and 8 exact at six decimals, including 92,832 convolutions in under two seconds.
* `Homer2003.qmd` — the bivariate FFT; all three examples, every published cell. The best fit in the folder: each construction is a shipped DecL form (`netceded`, `dbvsev`, per-axis `bs`).
* `Bodoff2017.qmd` — excess of policy limits losses as a Bernoulli policy limit; Exhibit 4.
* `Bear1990.qmd` — adjustable features and loss sharing provisions; Table I, six treaties, six DecL clauses. Reproduces the paper's own lognormal arithmetic exactly and its collective risk column by FFT.
* `Jin2016.qmd` — **not a reproduction.** Why the univariate half publishes nothing checkable and the bivariate half needs a common-shock count mode. Read alongside `Homer2003.qmd`: Jin cites Clark and Homer for its FFT benchmark, and the two sit on opposite sides of the feature boundary.

Rare-event and numerical-accuracy papers
-------------------

* `Asmussen2016a.qmd`
* `BenAmar2023.qmd`
* `BenRached2024.qmd`
* `Liu2026.qmd`
* `Wilson2016a.qmd`
* `Wilson2017.qmd`
* `Commentary.qmd` — a summary across the first three.

Candidate assessment
-------------------

`plan-examples.md` rates eighteen references for suitability, records which have been executed, and carries the corrections that executing them produced.

Provenance
-------------------

`severity-curves.qmd` is the reference record for the ten published severity curves shipped in ``library.agg``: which paper and which table each came out of, what the paper uses it for, and every transcription decision behind the DecL. It also carries the selection rationale for the eighteen unattributed ``Curve*`` reference distributions.
