## Use: Aggregate and Portfolio price differently

This is the most common source of confusion, and it deserves a slide of its own.

```{mermaid}
%%| label: fig-agg-vs-port
%%| fig-cap: "One unit has nothing to allocate. That single fact explains the whole difference."
flowchart TD
    subgraph aggside["Aggregate: one distribution"]
        adens["density_df"]
        aap(["apply_distortion"])
        acols["gS, gp_total, exag<br/>appended to density_df"]
        aprice["price(p, g)<br/>applies a given distortion"]
        adens --> aap --> acols --> aprice
    end

    subgraph portside["Portfolio: many units, one total"]
        pdens["density_df<br/>plus exa, exeqa, exi_xgta"]
        pcal(["calibrate_distortions"])
        pap(["apply_distortion"])
        paug["augmented_df<br/>a separate cached frame"]
        pprice["price, pricing_at,<br/>analyze_distortions"]
        pdens --> pcal --> pap
        pdens --> pap --> paug --> pprice
    end

    note["to calibrate an Aggregate,<br/>wrap it in a one line Portfolio"]
    aprice -.-> note
    note -.-> pcal

    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class adens,pdens,paug,acols cFr;
    class aap,pap,pcal cEn;
    class aprice,pprice,note cVw;
```

An `Aggregate` writes its distorted columns straight onto its own `density_df`, because with one unit there is nothing to allocate. A `Portfolio` builds a separate `augmented_df`, because allocation across units is the entire point and it needs the conditional expectation columns that only a portfolio density carries.

An `Aggregate` cannot calibrate. It applies distortions you give it. Calibration is a portfolio operation, so the recipe for calibrating a single line is to make it a one line portfolio.
