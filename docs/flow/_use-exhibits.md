## Use: how an exhibit gets made

Exhibits are the table side of the same idea, and the binding is even more direct: an exhibit names a frame. The summary exhibit is registered as the name `summary`, the title `Summary`, and the attribute `summary_df`. That is the whole registration.

```{mermaid}
%%| label: fig-exhibits
%%| fig-cap: "An exhibit is a named frame plus a perspective. Nothing is recomputed."
flowchart LR
    frames[("named frames<br/>summary_df, stats_df, tail_df")]
    reg[("exhibit registry<br/>name to frame, per class")]
    persp["Perspective<br/>RAW or INSURER"]
    build(["build_exhibit"])
    exhibit["Exhibit<br/>blocks, captions, formats, flags"]
    tbl["rendered table"]

    frames --> build
    reg --> build
    persp --> build
    build --> exhibit
    exhibit --> tbl

    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    classDef cOut fill:#fbeef2,stroke:#a3466a,color:#111;
    class frames,reg cFr;
    class build cEn;
    class exhibit,persp cVw;
    class tbl cOut;
```

`Perspective` is the only genuine translation in the layer. RAW serves the frame as the library computes it. INSURER serves the same numbers relabeled and rearranged for someone reading an insurance submission rather than a distribution. Where no translation exists, INSURER equals RAW by the default rule and nothing extra has to be written.

Registered exhibits: `summary`, `stats`, `validation`, `tail`, `reins`, `bs_window`, `sharpen`, `tail_behavior`, `economic`, `economic_ratios`, `economic_waterfall`, `dependency`. `available_exhibits(obj)` says which of them an object can serve, which depends on its class, on whether it has been updated, and for `sharpen` on whether the grid probe has run.
