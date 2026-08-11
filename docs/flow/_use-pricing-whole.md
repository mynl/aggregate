## Use: the pricing picture, whole

```{mermaid}
%%| label: fig-pricing-whole
%%| fig-cap: "Calibration is optional, the augmented frame is not."
flowchart TD
    dens["density_df"]

    subgraph calib["calibration, optional"]
        coc["coc, p or a"]
        cal(["calibrate_distortions"])
        caldf["calibration_df"]
        distdf["distortion_df"]
    end

    decldist["a declared Distortion<br/>from DecL"]
    dists[("Distortion objects")]

    apply(["apply_distortion"])
    aug["augmented_df<br/>cached per option set"]

    subgraph reads["readouts"]
        pat["pricing_at"]
        pr["price"]
        ads["analyze_distortions"]
        diag["allocation_diagnostics"]
    end

    pent["price_pentagon, price_ccoc<br/>no distortion needed"]
    rein["reins_price_df<br/>price the cession"]
    bnds["Bounds<br/>the envelope over distortions"]

    dens --> cal
    coc --> cal
    cal --> caldf
    cal --> distdf
    cal --> dists
    decldist --> dists
    dists --> apply
    dens --> apply
    apply --> aug
    aug --> pat
    aug --> pr
    aug --> ads
    aug --> diag
    dens --> pent
    dists --> rein
    dists --> bnds
    dens --> bnds

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class dens,aug cFr;
    class coc,decldist cIn;
    class cal,apply cEn;
    class caldf,distdf,dists,pat,pr,ads,diag,pent,rein,bnds cVw;
```

The engine answers what can happen and how likely it is. The distortion answers what that is worth. They meet in one frame, and every price, margin, allocation and return in the library is a row of it.
