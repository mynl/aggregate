## Use: calibration, target in and distortions out

Calibration is one point: one cost of capital at one capital level. Every fitted family hits the same premium at that one point and they part company everywhere else, which is exactly the comparison the exercise is for.

```{mermaid}
%%| label: fig-calibration
%%| fig-cap: "What you asked for and what you got, kept in separate frames."
flowchart TD
    dens["density_df"]
    coc["coc target"]
    pa["p or a<br/>the capital level"]
    names["families to fit"]
    view["reins_view<br/>gross, ceded or net"]

    cal(["calibrate_distortions"])

    caldf["calibration_df<br/>one row, shared by all families:<br/>coc, p, F(a), then L M P Q a LR PQ ROE"]
    distdf["distortion_df<br/>one row per family:<br/>param_name, param, error, gini_p, area"]
    dists[("distortions<br/>the fitted objects, by name")]

    dens --> cal
    coc --> cal
    pa --> cal
    names --> cal
    view --> cal
    cal --> caldf
    cal --> distdf
    cal --> dists

    classDef cIn fill:#eef3fb,stroke:#4667a3,color:#111;
    classDef cEn fill:#fdf1e3,stroke:#b8762a,color:#111;
    classDef cFr fill:#eaf6ee,stroke:#2f7a4a,color:#111;
    classDef cVw fill:#f6f0fa,stroke:#7a4f9e,color:#111;
    class dens cFr;
    class coc,pa,names,view cIn;
    class cal cEn;
    class caldf,distdf,dists cVw;
```

`distortion_df` carries `gini_p`, the normalized shape that makes families with incomparable raw parameters comparable, so a Wang shape and a PH shape can be read on one scale.

A distortion does not have to be calibrated. You can declare one in DecL and hand it straight to `apply_distortion`, skipping this diagram entirely.

`reins_view` matters when the book cedes. The portfolio total already is its net view, since the units convolved are the units as built with cessions applied, so `reins_view='gross'` is the one that says something new.
