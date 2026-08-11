## Reading the diagrams

Four kinds of node appear throughout, and the coloring is consistent across every diagram in this set.

| Kind | Color | What it is | Examples |
|---|---|---|---|
| Input | blue | Declared, not computed | DecL text, the spec, the grid parameters |
| Engine | orange | Does work, costs time | the bucket sizer, the FFT, the reinsurance stage |
| Frame | green | Stored state, computed once | `density_df`, `stats_df`, `reins_density_df` |
| View | purple | Derived on read, cheap | `summary_df`, `validation_df`, `tail_df`, every narrative string |

The practical consequence: green nodes are what `update` costs you and what a cache holds. Purple nodes are free, so there is no reason to hoard them in a variable.
