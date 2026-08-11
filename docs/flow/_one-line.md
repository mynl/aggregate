## The whole library in one line

The organizing claim: `density_df` is the key output. One frame, computed once, and almost every number the library reports is a read off it or off the moment store beside it.

```{mermaid}
%%| label: fig-one-line
%%| fig-cap: "The elevator version. Everything else is detail hung on this line."
flowchart LR
    decl["DecL program"] --> obj["Aggregate<br/>or Portfolio"]
    obj --> eng(["engine<br/>update()"])
    eng --> dens["density_df"]
    dens --> ans["every answer:<br/>moments, quantiles, TVaR,<br/>prices, allocations, charts"]

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class decl,obj cIn;
    class eng cEn;
    class dens cFr;
    class ans cVw;
```
