## Use: how a chart gets made

Charts go through an intermediate representation. An emitter reads the frames and builds a `ChartDoc`, a plain data object of panels, series and marks with no matplotlib in it. A renderer then draws it. That split is what lets the same picture come out of a notebook, a browser and a paper.

```{mermaid}
%%| label: fig-charts
%%| fig-cap: "obj.plot() is one line into this path, asking for the object's primary chart."
flowchart LR
    frames[("frames")]
    emit(["emitter<br/>one per object kind"])
    doc["ChartDoc<br/>Panel, ChartSeries, Mark, ChartAxis"]
    rend(["renderer<br/>plot_chartdoc"])
    fig["figure"]
    other["any other renderer<br/>browser, tex"]

    frames --> emit
    emit --> doc
    doc --> rend
    rend --> fig
    doc --> other

    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    classDef cOut fill:#fbeef2,stroke:#a3466a,color:#111;
    class frames cFr;
    class emit,rend cEn;
    class doc cVw;
    class fig,other cOut;
```

Registered charts: `agg`, `port`, `severity`, `distortion`, `pnl`, `reins`, `envelope` for pricing bounds, and the bivariate joint surface. `available_charts(obj)` says which of them an object can serve.
