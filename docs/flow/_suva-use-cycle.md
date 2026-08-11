## SUVA-Use

Five stages, in order, and the order matters. **S**pecify in DecL, **U**pdate to compute, **V**alidate the result, **A**djust the declaration, then **U**se the object for whatever you built it for.

The stages are not equal in cost. Specify is free. Update is the expensive one, and it is the only stage that computes anything. Validate, Adjust and Use are all reads and re-declarations around that single computation.

Adjust is the stage people skip, and it is the one that saves the day. Start gross, get the gross numbers right, and only then add the reinsurance. A cession fitted on top of a gross model you have not validated hides two errors inside one number.

```{mermaid}
%%| label: fig-suva-use
%%| fig-cap: "The SUVA-Use cycle. Adjust loops back into Update, which is why the grid and the validation have to be re-checked after every change to the declaration."
flowchart LR
    S["Specify<br/>write the DecL"]
    U1["Update<br/>choose the grid, run the FFT"]
    V["Validate<br/>do the moments agree?"]
    A["Adjust<br/>add reinsurance, retune the grid"]
    U2["Use<br/>price, allocate, chart, tabulate"]

    S --> U1 --> V --> A --> U2
    A -.->|"re-declare, then recompute"| U1
    V -.->|"grid is wrong, resize and rerun"| U1

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class S,A cIn;
    class U1 cEn;
    class V,U2 cVw;
```

Two dotted returns, and they mean different things. Validate sends you back when the grid is wrong, which is a numerics problem: same declaration, different `bs` or `log2`. Adjust sends you back when the declaration itself changes, which is a modeling decision. Keeping them separate in your head is worth the effort, because the fix is different in each case.
