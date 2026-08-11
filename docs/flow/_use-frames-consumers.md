## Use: three ways to consume the frames

This is the diagram to show people. The point is the fan at the bottom: charting and exhibits never talk to the engine. They talk to the frames, which is why a new chart or a new exhibit costs nothing in compute and cannot disagree with the table beside it.

```{mermaid}
%%| label: fig-consumers
%%| fig-cap: "The expensive half is on top and happens once. The cheap half is underneath and happens constantly."
flowchart TD
    eng(["engine"])

    subgraph frames["the frames: computed once, by update()"]
        direction LR
        dens["density_df<br/>the distribution"]
        stats["stats_df<br/>the moment store"]
    end

    subgraph vws["named views: derived on read"]
        direction LR
        summ["summary_df"]
        val["validation_df"]
        tail["tail_df"]
        tb["tail_behavior_df"]
        bsw["bs_window_df"]
        rein["reins frames"]
        econ["economic_df<br/>PnL only"]
    end

    subgraph consume["three consumers"]
        direction LR
        pand["your own pandas<br/>the frames are public"]
        charts["charts<br/>build_chart_doc"]
        exh["exhibits<br/>build_exhibit"]
    end

    qd["qd()<br/>the console readout"]
    fig["matplotlib figure"]
    tbl["formatted table<br/>notebook, web, docs"]

    eng --> dens
    eng --> stats
    dens --> summ
    stats --> summ
    stats --> val
    dens --> tail
    dens --> rein
    dens --> econ

    summ --> pand
    summ --> charts
    summ --> exh
    dens --> charts
    tail --> exh
    val --> exh
    tb --> exh
    bsw --> exh
    rein --> exh
    econ --> exh

    charts --> fig
    exh --> tbl
    summ --> qd
    val --> qd

    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    classDef cOut fill:#fbeef2,stroke:#a3466a,color:#111;
    class eng cEn;
    class dens,stats cFr;
    class summ,val,tail,tb,bsw,rein,econ cVw;
    class pand,charts,exh,qd cVw;
    class fig,tbl cOut;
```

The asymmetry is deliberate and worth stating out loud when you teach this. Update is expensive and happens once. Use is cheap and happens constantly. A user who understands that stops re-running `update` to get a different picture.
