## Adjust: add the reinsurance

The discipline of this stage: start gross, validate the gross, then cede. A cession fitted on top of an unvalidated gross model buries two errors in one number, and you cannot tell them apart afterwards.

Reinsurance is a second computation on the same declaration, producing a parallel set of frames. Gross, ceded and net are three separate distributions rather than a decomposition, which is why they get their own frames instead of extra columns on `density_df`. A gross premium less a net premium compares two books; it does not decompose one.

```{mermaid}
%%| label: fig-adjust-reins
%%| fig-cap: "Reinsurance produces parallel frames, not extra columns, because the three views are three distributions."
flowchart TD
    dens["density_df"]
    spec["occurrence and aggregate<br/>reinsurance clauses"]
    eng(["engine, reinsurance stage"])

    rdens["reins_density_df<br/>gross, ceded, net columns"]
    rstats["reins_stats_df"]
    rsumm["reins_summary_df"]
    rprice["reins_price_df<br/>needs a distortion"]
    rviews["reins_views<br/>which views exist"]
    rdesc["reins_description<br/>reins_explanation<br/>reins_kinds"]
    stats["stats_df<br/>after_occ, occ_impact,<br/>agg_impact, gross_empirical"]

    spec --> eng
    eng --> dens
    eng --> rdens
    rdens --> rstats
    rstats --> rsumm
    rdens --> rprice
    rdens --> rviews
    spec --> rdesc
    eng --> stats

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class spec cIn;
    class eng cEn;
    class dens,rdens,stats cFr;
    class rstats,rsumm,rprice,rviews,rdesc cVw;
```

The moment store carries the cession story too, in `after_occ`, `occ_impact`, `agg_impact` and `gross_empirical`. That is how `validation_df` switches from a validation reading, theoretical against realized, to an economic reading, gross against net, without changing shape. The same eight columns, read two ways.

Adjusting is not only reinsurance. Retuning the grid, changing the exposure, adding premium and expense so a P&L can be formed: all of it lands here, and all of it sends you back through Update and Validate before you Use anything.
