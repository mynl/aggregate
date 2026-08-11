## Use: the augmented frame

Pricing adds exactly one new frame, `augmented_df`, and everything priced is a read off it.

```{mermaid}
%%| label: fig-augmented
%%| fig-cap: "One frame per option set, all asset levels, cached until the density changes."
flowchart TD
    dens["density_df"]
    dist["a Distortion"]
    opts["options:<br/>view ask or bid,<br/>allocation lifted or linear,<br/>S_calculation"]

    apply(["apply_distortion<br/>one O(n) sweep"])
    cache[("augmented_dfs cache<br/>keyed by name, view, role,<br/>S_calculation, allocation")]
    aug["augmented_df<br/>density_df plus gS, gp_total,<br/>exag_total, exag per unit"]

    dens --> apply
    dist --> apply
    opts --> apply
    apply --> cache
    cache --> aug

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    class dens,aug,cache cFr;
    class dist,opts cIn;
    class apply cEn;
```

Three facts about this frame.

One sweep serves all asset levels. You do not build a new frame to price at a different `a`, you read a different row, which is why pricing feels instant after the first call.

The cache key includes the options, so `lifted` and `linear` coexist rather than overwrite. Any `update` clears the cache, because a new density invalidates every distorted read off it. That is the Adjust stage reaching forward: cede, and every price you had computed is gone, correctly.

The two allocations differ in exactly one term, the tail share used to split the last layer: the distorted share for lifted, the objective share for linear. Everything else in the two frames is identical.
