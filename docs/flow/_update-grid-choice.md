## Update: how the grid gets chosen

The sizer lives inside the engine but deserves its own picture, because it decides what the density can and cannot see. Everything downstream inherits that decision, so a grid that clips the tail produces a distribution that is wrong in a way no later stage can repair.

`sharpen` sits outside the loop. It is opt in, it re-runs the engine over the eight neighboring `(bs, log2)` cells, and it moves to a better one on a large win. It is the only genuine cycle in the library.

```{mermaid}
%%| label: fig-grid-choice
%%| fig-cap: "Dotted edges are the opt-in feedback path. Everything else in the library is acyclic."
flowchart TD
    thcols["theoretical moments<br/>from stats_df"]
    sizer(["bs_window: run the candidate methods<br/>moment, exact_discrete, bounded_small"])
    bswdf["bs_window_df<br/>one row per method, which was selected"]
    bsnarr["bs_description<br/>bs_explanation"]
    grid["grid: bs, log2, x_min"]
    eng(["engine"])
    dens["density_df"]
    sharp(["sharpen()<br/>on demand, probes 8 neighbor cells"])
    shdf["sharpen_df<br/>one row per probed cell"]
    shnarr["sharpen_description<br/>sharpen_explanation<br/>sharpen_program"]

    thcols --> sizer
    sizer --> bswdf
    sizer --> grid
    bswdf --> bsnarr
    grid --> eng
    eng --> dens
    dens -.->|"aliasing and moment error say the grid is wrong"| sharp
    sharp -->|"re-run update per cell"| eng
    sharp --> shdf
    shdf --> shnarr
    sharp -.->|"move to the winning cell"| grid

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class thcols,grid cIn;
    class sizer,eng,sharp cEn;
    class dens cFr;
    class bswdf,shdf,bsnarr,shnarr cVw;
```

`bs_window_df` is the receipt: one row per candidate method, which window each proposed, and which one was selected. When Validate fails, that frame is the first place to look.
