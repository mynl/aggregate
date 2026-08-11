## Update: the engine has two outputs, not one

`density_df` is the distribution. `stats_df` is the moment store, and it fills from two directions: the theoretical columns are written at construction, straight from the declaration, before any FFT runs. The empirical columns are written afterwards by reading the grid back. The `error` column is the gap between them, and that gap is the whole basis of the Validate stage.

```{mermaid}
%%| label: fig-two-outputs
%%| fig-cap: "The two arrows into stats_df are the point. Theoretical from the declaration, empirical from the density, meeting in one frame so they can be compared."
flowchart TD
    spec["declaration<br/>frequency, severity, exposure, layers"]
    stats["stats_df"]
    thcols["theoretical columns<br/>mixed, independent, per component"]
    sizer(["bucket and window sizer"])
    grid["grid: bs, log2, x_min"]
    eng(["engine: update_work()<br/>discretize severity, FFT,<br/>apply PGF, invert"])
    sev["sev_density_df"]
    dens["density_df<br/>loss, p_total, F, S"]
    empcols["empirical columns<br/>empirical, error"]

    spec --> thcols
    thcols --> stats
    thcols -->|"mean, cv, skew"| sizer
    sizer --> grid
    grid --> eng
    spec --> eng
    eng --> sev
    sev --> dens
    eng --> dens
    dens -->|"read the grid back"| empcols
    empcols --> stats

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    class spec,thcols,empcols,grid cIn;
    class sizer,eng cEn;
    class stats,sev,dens cFr;
```

Note where the grid comes from. The sizer runs first and consumes the theoretical moments the declaration produced, so the analytic half of `stats_df` is upstream of the density, not a report on it.
