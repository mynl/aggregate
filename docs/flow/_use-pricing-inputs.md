## Use: pricing needs three inputs, not one

A distortion on its own does not price a book. Pricing needs three things beyond the distribution, and separating them is the clearest way to explain the API.

```{mermaid}
%%| label: fig-pricing-inputs
%%| fig-cap: "Separating the three is the clearest way to explain the pricing API."
flowchart LR
    dens["density_df<br/>the distribution"]
    cap["a capital rule<br/>p, or an asset level a"]
    tgt["a target<br/>cost of capital, or a premium"]
    fam["a distortion family<br/>ccoc, ph, wang, dual, tvar"]
    price["a price"]

    dens --> price
    cap --> price
    tgt --> price
    fam --> price

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cOut fill:#fbeef2,stroke:#a3466a,color:#111;
    class dens cFr;
    class cap,tgt,fam cIn;
    class price cOut;
```

Calibration is the step that turns a target into a distortion. Once you have a distortion you no longer need the target, which is why the two frames it produces are separate: `calibration_df` records what you asked for, `distortion_df` records what you got.
