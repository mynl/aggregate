## The core, whole

Specify, Update, Validate and Adjust on one page. Three bands: the work, the stored state, and the free reads.

```{mermaid}
%%| label: fig-core-whole
%%| fig-cap: "The single arrow out to augmented_df is where Use picks up."
flowchart TD
    decl["DecL program"]
    kb["knowledge base<br/>build by name"]
    spec["spec: Aggregate or Portfolio"]

    subgraph engine_box["the engine"]
        sizer(["bucket and window sizer"])
        eng(["update: discretize, FFT, PGF, invert"])
        reins(["reinsurance stage"])
        sharp(["sharpen, opt in"])
    end

    subgraph frames["the frames"]
        dens["density_df"]
        stats["stats_df"]
        sev["sev_density_df"]
        rdens["reins_density_df"]
        bswdf["bs_window_df"]
        shdf["sharpen_df"]
    end

    subgraph views["derived views"]
        summ["summary_df"]
        val["validation_df"]
        tail["tail_df"]
        tb["tail_behavior_df"]
        rstats["reins_stats_df"]
        rsumm["reins_summary_df"]
        narr["descriptions and explanations"]
    end

    aug["augmented_df<br/>the Use stage"]

    decl --> spec
    kb --> spec
    spec --> stats
    spec --> sizer
    spec --> tb
    stats --> sizer
    sizer --> bswdf
    sizer --> eng
    eng --> sev
    eng --> dens
    eng --> reins
    reins --> rdens
    dens --> stats
    sharp --> eng
    sharp --> shdf
    dens --> summ
    stats --> summ
    stats --> val
    dens --> tail
    rdens --> rstats
    rstats --> rsumm
    bswdf --> narr
    shdf --> narr
    tb --> narr
    dens --> aug

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class decl,kb,spec cIn;
    class sizer,eng,reins,sharp cEn;
    class dens,stats,sev,rdens,bswdf,shdf cFr;
    class summ,val,tail,tb,rstats,rsumm,narr,aug cVw;
```
