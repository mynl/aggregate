## Validate: what hangs off the two frames

Validation asks one question: did the FFT reproduce the moments the declaration promised? If the first three moments agree, the aggregate is not unreasonable. If they do not, the grid is wrong and no amount of downstream work will fix it.

Everything in this diagram is a derived view. None of it recomputes the distribution.

```{mermaid}
%%| label: fig-validate
%%| fig-cap: "Derived views are cheap. The two frames above them are not."
flowchart TD
    dens["density_df"]
    stats["stats_df"]
    spec["declaration"]

    val["validation_df<br/>moment QA, theoretical vs realized"]
    summ["summary_df<br/>Mean SD CV Skew P01 Median P99"]
    tail["tail_df / tail_periods_df<br/>return period ladder"]
    tb["tail_behavior_df<br/>tail class, both sides"]
    tdesc["tail_description<br/>tail_explanation"]
    q["q, tvar, cdf, sf, var_dict<br/>via GridDistribution"]

    stats --> val
    stats --> summ
    dens -->|"percentiles via q, q_sev"| summ
    dens --> q
    q --> tail
    spec -->|"classify freq and sev, no grid needed"| tb
    tb --> tdesc

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class spec cIn;
    class dens,stats cFr;
    class val,summ,tail,tb,tdesc,q cVw;
```

Two details worth knowing.

`summary_df` is the only frame fed from both sides, moments from `stats_df` and percentiles from the grid, so before `update` it degrades gracefully to the analytic moments with the percentile columns blank.

There are two tail objects and they have different sources. `tail_df` is a reach question, how far out the numbers go, read off the grid. `tail_behavior_df` and `tail_description` are a shape question, how the tail decays, classified analytically from the frequency and severity declarations. The classifier needs no grid at all, so it is available the moment you have specified.
