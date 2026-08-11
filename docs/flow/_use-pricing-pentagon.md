## Use: pricing with no distortion in sight

Two of the pricing entry points are pure accounting. They complete the identity between loss, premium, margin, capital and return, given a capital level and one target. No distortion is involved and nothing is calibrated.

```{mermaid}
%%| label: fig-pentagon
%%| fig-cap: "No distortion, nothing calibrated, no risk measure. Arithmetic on the identity."
flowchart LR
    dens["density_df"]
    L["expected loss at a"]
    one["exactly one target:<br/>P, M, Q, LR, PQ or ROE"]
    pent(["Pentagon.solve"])
    row["the octet<br/>L M P Q a, LR PQ ROE"]

    dens --> L
    L --> pent
    one --> pent
    pent --> row

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class dens cFr;
    class L,one cIn;
    class pent cEn;
    class row cVw;
```

`price_pentagon`, `price_ccoc` and `prob_loss_assets` are all this diagram. Keeping them visibly separate from the distortion path is useful when teaching, because it makes clear which questions need a risk measure and which are arithmetic.
