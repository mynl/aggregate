# [BS-Window-Dtypes]

Executed at `1.0.0a275`. Recorded for posterity: the bug is small, the shape of
it is not, and it can recur anywhere a reporting frame is built by transposing.

## The finding

Every column of `Aggregate.bs_window_df` came out `object`: `applies`, `x_min`,
`x_max`, `W`, `bs`, `log2`, `coverage`, `note`. The cause was one line in
`_bucket_window.py`:

```python
df = pd.DataFrame(rows).T
```

`rows` is a dict of dicts, one per sizing method, each mixing bool, float, int
and str. `pd.DataFrame(rows)` puts **methods in the columns and fields in the
index**, so every column is mixed and the whole block types as `object`; the
transpose then carries that dtype across wholesale, because pandas does not
re-infer per column on a transpose.

The tell was which columns worked. `selected`, `log2_need` and `clipped` are the
three assigned **after** the transpose, by ordinary column assignment, and they
were the only three correctly typed.

## Why it mattered

`bs_window` is a passthrough exhibit, so the frame's dtypes are what the served
table sees. Every object column reached the IR as `dtype='string'`,
`align='l'`, `wrap=True`, `raw=False`, with cells carrying `raw=None`. So the
numeric columns were left aligned, wrapped, and carried **no raw values**, which
is what an interactive grid sorts and filters on. `log2` had no
`BS_WINDOW_FORMATS` entry (as an int it never needed one) and so rendered
completely unformatted.

The Portfolio frame was already correct, and was the model for the fix:
`port_build_bs_window_df` builds from a *list* of row dicts via
`pd.DataFrame(rows).set_index('unit')`, never transposed, so per column
inference runs.

## What landed

1. `_bucket_window.py`, the aggregate sizer: `pd.DataFrame.from_dict(rows,
   orient='index')` in place of the transpose.
2. Both `log2` columns, aggregate and portfolio, cast to nullable `Int64`. An
   exponent reads as an integer, and both can be genuinely absent (a non-applies
   `sbj` row records no grid; `_need` returns NaN on a degenerate window), so
   plain `int64` would not hold. Without the cast the published dtype would
   depend on the book: `int64` on most, `float64` on one with an n/a row.
3. `bivariate.py`, `bs_window_df` and `axis_support_df`: the identical
   transpose, same fix. `bs_window_df` feeds the same exhibit, so fixing only
   the aggregate would have left that leaf broken.

Untouched, and deliberately: `_portfolio.py` `tail_behavior_df` transposes
Series carrying `INFO_NA` sentinels, so its object dtype is by design;
`bivariate.py` `dependency_df` is all float, so its transpose is harmless.

## The snapshot diff, read deliberately

12 of 116 exhibit snapshots moved, all in `bs_window` and `dependency`. Column
metadata went `('string', 'l', raw=None)` to the real dtype, right aligned, with
raw values. Visible text moved in two places, both improvements: `log2_need`
from `6.000` to `6`, and the bivariate `axis_support_df` floats from raw repr
(`19.499999999999996`, `0.7297909280734887`) to formatted (`19.50`, `0.72979`),
since a string column had been bypassing formatting altogether. Metadata
(captions, notes, head, foot, level counts) identical everywhere; no keys added
or removed.

## Left open

The bivariate `clipped` column is a **bool flag** while the aggregate `clipped`
is an **estimated mass**, and `BS_WINDOW_FORMATS` carries one `'.2e'` for the
name. So the bivariate flag renders `0.00e+00` where it should read `False`.
Pre-existing and unchanged by this work (verified against the old object dtype
frame), but now visible as a genuine dtype conflict rather than two strings.
Fixing it means either renaming one of the two columns or scoping the format per
class, which is a naming call for the author.

## The general lesson

`pd.DataFrame(dict_of_dicts).T` is never right for a mixed-type reporting frame.
Build it index-oriented, or from a list of row dicts, so pandas infers per
column. Anything assigned after the fact will look correct and mask the problem.
