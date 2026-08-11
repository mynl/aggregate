## Use: readouts off the augmented frame

```{mermaid}
%%| label: fig-readouts
%%| fig-cap: "Six readouts, one frame. None of them recompute anything."
flowchart TD
    aug["augmented_df"]
    dens["density_df"]

    pat["pricing_at<br/>pentagon row per unit at p or a"]
    pr["price<br/>PricingResult: df, price,<br/>price_dict, a_reg, reg_p"]
    ad["analyze_distortion<br/>pricing_df + audit_df"]
    ads["analyze_distortions<br/>pricing_df across families<br/>+ augmented_dfs snapshot"]
    psa["price_stand_alone<br/>each unit on its own"]
    diag["allocation_diagnostics"]
    bod["bodoff<br/>layer allocation, no distortion"]

    aug --> pat
    aug --> pr
    aug --> ad
    aug --> ads
    aug --> psa
    aug --> diag
    dens --> bod

    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class aug,dens cFr;
    class pat,pr,ad,ads,psa,diag,bod cVw;
```

`analyze_distortions` is the loop version: it walks the calibrated set, warms the cache for each, and returns one `pricing_df` with the families side by side. That single frame answers which distortion to use, and it exists because all of them sit on the same density.

`bodoff` is drawn detached on purpose. Layer allocation needs no distortion at all, so it hangs off the density rather than the augmented frame.
