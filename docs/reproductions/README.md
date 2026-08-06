Introduction
==============

This folder contains **reproductions**: worked examples using ``aggregate`` to reproduce figures, tables, and results from published papers.

Each page follows the same shape: context and problem, the paper's own solution and gold standard, the ``aggregate`` reproduction, and comments. A page named for a paper that turned out **not** to be reproducible says so and gives the evidence.

Actuarial papers
-------------------

* `Venter1983.qmd` — transformed beta and gamma; Exhibit 3, all three columns. The bucket size selects which of Venter's two exact methods you get.
* `Mack2003.qmd` — Riebesell's increased limits rule under the collective risk model; Table 1 and the threshold above which the rule is exact.
* `Mata2005.qmd` — excess trend and exposure adjustment by layer; Tables 1 to 14, and a discrepancy in the paper's own Table 1.
* `Bruno2006.qmd` — thresholded direct convolution; Tables 6, 7 and 8 exact at six decimals, including 92,832 convolutions in under two seconds.
* `Bodoff2017.qmd` — excess of policy limits losses as a Bernoulli policy limit; Exhibit 4.
* `Jin2016.qmd` — **not a reproduction.** Why the univariate half publishes nothing checkable and the bivariate half needs a common-shock count mode.

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
