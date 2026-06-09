

```python
$env:UV_LINK_MODE = "copy"
uv run --extra dev --extra notebook jupyter lab

or the explicit two-step:

$env:UV_LINK_MODE = "copy"
uv sync --extra dev --extra notebook
uv run jupyter lab
```

Comprehensive remove fuzz improvements. what is done for Portfolio (which columns)? Mask approach seems best. This is done in Aggregate and Portfolio - but only called as a function in Portfolio. 

```python
df.mask(df.abs() < eps, 0.0)
```


1. Test Pricing Bounds and Pricing Allocation; impacts on gini p
2. Test mv, pnl, ssev etc. (new decl terms)
3. docs/2_user_guides/2_x_re_pricing.rst (~lines 210–216) listing removed by-layer names like reinsurance_audit_df  
4.  `decl_pprint` is a bit mystifying 
5. port with no neg sev should have a None _bs_window_df

***
### Nits and Gnats
- [x] (a50) info for Agg needs to be line by line consistent (freq, sev, then approx); report none/na as appropriate
**hygene-3**
- [ ]    File "T:\worktrees\aggregate_REFACTOR\src\aggregate\distributions.py", line 2138, in _sev_label
    if not self.sevs:
           ^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

### Juggling Balls

- [ ] freeze testing - see ipynb - seems to work... 

***

### Build Docs  / Run Tests
```python
# run all tests
uv run pytest -q 2>&1 | tail -40 (24s)

# html doc from REFACTOR
$log = "T:\tmp\build-log-$(Get-Date -Format yyyyMMdd-HHmmss).log"
.\doc-test-uv.ps1 -Lenient 2>&1 | Tee-Object -FilePath $log

# text mode 
 $log = "T:\tmp\build-log-text-$(Get-Date -Format yyyyMMdd-HHmmss).log"
.\doc-test-uv.ps1 -Text -Lenient -OutputDir T:\doc-diff\agg-doc-diff\text 2>&1 | Tee-Object -FilePath $log
```


### Other uv tricks
```python
# Launch the UberShell with the aggregate_hacks (aggfz) plugin discoverable,
# running inside the aggregate project venv so the editable `aggregate` is
# importable. Just run:  .\uber-agg.ps1
#
# Nothing is installed into the project venv or pyproject: uv builds an
# ephemeral, cached overlay (uber_shell + this repo's hacks/ package) on top of
# the project environment. `aggregate` imports lazily on the first `aggfz`
# command and stays resident, so freeze/check cycles skip repeated cold imports.
$env:UV_LINK_MODE = "copy"
$here = $PSScriptRoot
uv run --project "$here" --with uber_shell --with-editable "$here\hacks" uber
```

***

This takes a weirdly long time to compute. The individual aggs don't? 

```python
port = build('port Test2 '
            'agg A 1 claim sev gamma 100 cv .3 fixed '
            'agg B 1 claim sev lognorm 50 cv .5 fixed '
             'agg C 1 claim sev 100 * pareto 2.1 - 100 fixed '
             , bs=1, log2=16
        )
port.plot()
port

# fast?
a1 = build( 'agg A 1 claim sev gamma 100 cv .3 fixed ')
a1

a2 = build('agg B 1 claim sev lognorm 50 cv .5 fixed')
a2

a3 = build('agg C 1 claim sev 100 * pareto 2.1 - 100 fixed ')
a3
```

---

Pricing Bounds
=================

Read the following for background. It contains a way to determine the range of natural allocations to each unit consistent with a given price for the total. This goes via determining the graphs (TvaR(X), NA to Xi) which is done by the first function. that function appears to work quite well. Then there is the next step of determining the convex hulls of the graphs. that step the proposal looks a bit long  and im not sure it makes the most of what we know about the input data (which will come from Portfolio.density_df in the obvs. way). This is all heading towards re-writing Portfolio.pricing_bounds. There is probably overlap with bounds.py too. 

The ask is this: I want a clean, fast and efficient implmentation of a pricing_bounds function. I think ideally it returns a function of the total premium which when called on P returns the upper and lower bounds by unit, along with information about the bitvars that achieve each. I'm open to exact shape and form. I'd like to try this out in a hackathon. I want you to 1) digest, 2) confirm/clarify understanding, 3) sketch your approach, using or not the code below, 4) implement your code in a function(s) in ./hacks/pb.py that i can play around with (import pb; ans = pb.<do work>(port); etc.). Once we are all set on that, it can be integrated into bounds.py and/or portfolio.py. 

Good to go?

---


Background: in my paper on similar risks  i identify all distortions g that price a given risk X to a price P. the idea is that the set of such g is the closed convex hull of bitvars that achieve P and these are easy to identify. there is a p* st TVaR(p*)(X) = P. Then given any p1 < p < p2 we can find a w so that the p1, p2 bitvar with weight w (usually on p2 by my convention in Aggregate) achieves P. These are the extreme points of G_P := { g | g(X) = P }. Now, i am interested in determining the range of natural allocations in this sense. X = sum X_i is a decomposition into a finite number of units i. For g we have the natural alloc premium to unit i which equals P(X_i g'(S(X))) (plus modification for non-diff). Since everything is linear the NA to a weighted g is the weighted NA, so the NA of unit i is the co conv of the bitvar allocs. thus to determine the range of NAs we just need to calc all the bitvar NAs. The extreme points will also be boundary points - so the extreme na occurs at a bitvar. 

--> 

Yes -- for a scalar natural allocation, I concur, with one qualification.

Let $T(p)=\operatorname{TVaR}_p(X)$ and let $a_i(p)$ be the TVaR natural allocation to unit $i$. A pricing distortion represented by a Kusuoka measure $\mu$ has

$$  
\rho_\mu(X)=\int T(p)\mu(dp), \qquad A_i(\mu)=\int a_i(p)\mu(dp).  
$$

The constraint $\rho_\mu(X)=P$ is one affine constraint on probability measures. Your paper uses exactly the corresponding hyperplane/simplex geometry: the extreme consistent measures are the pure $p^_$ TVaR and two-point TVaR mixtures with $p_1\le p^_\le p_2$, with weight chosen to hit $P$. The proof uses Dubins/Bauer to reduce extrema of affine functionals to those extreme points.

The qualification is important: every scalar allocation extremum occurs at a bitvar, but not every bitvar allocation is necessarily a boundary point of the allocation image. The affine allocation map can collapse extreme measures into interior allocation vectors. For any fixed unit $i$, however, $\min A_i$ and $\max A_i$ occur at bitvars. More generally, any linear allocation score $\sum_i c_i A_i$ is optimized by a bitvar.

The Clever Reduction

For unit $i$, the bitvar allocation from $p_1,p_2$ is just the height at $T=P$ of the chord joining the two points $(T(p_1),a_i(p_1))$ and $(T(p_2),a_i(p_2))$:

$$  
B_i(p_1,p_2)=a_i(p_1)+{P-T(p_1)\over T(p_2)-T(p_1)}{a_i(p_2)-a_i(p_1)}.  
$$

Therefore, the allocation range for unit $i$ is the vertical slice at $T=P$ through the convex hull of the parametric curve

$$  
C_i={(T(p),a_i(p)):0\le p\le 1}.  
$$

So we do not need a 2D sweep over all $(p_1,p_2)$. Compute the lower and upper convex envelopes of the curve $a_i$ as a function of $T$. The lower envelope at $P$ gives the minimum allocation, and the upper envelope at $P$ gives the maximum allocation. The edge of the hull that crosses $T=P$ identifies the two optimal TVaR levels $p_1,p_2$ and the weight $w$.

Practical Algorithm

Compute a grid $p_j$, including $0$, $1$, $p^*$, and all important CDF breakpoints. For each $p_j$, compute $T_j=\operatorname{TVaR}_{p_j}(X)$ and $a_{ij}=\operatorname{NA}_i(\operatorname{TVaR}_{p_j})$. Sort the points $(T_j,a_{ij})$ by $T_j$. For each unit $i$, run a monotone-chain convex hull algorithm on those 2D points. Intersect the lower and upper hulls with the vertical line $T=P$. If the relevant hull edge joins $j,k$, then the optimal bitvar is $(p_j,p_k,w)$ with

$$  
w={P-T_j\over T_k-T_j}.  
$$

This changes the calculation from an $O(n^2)$ pair sweep to roughly $O(n\log n)$, or $O(n)$ after sorting. For an Aggregate-style discrete distribution, using the CDF breakpoints should often be exact for the discretized distribution, because TVaR and the canonical TVaR allocation vary linearly with one another inside a probability atom.

Dual View

The same method is the LP dual in geometric form. The lower allocation is

$$  
\sup_{\alpha,\beta}{\alpha+\beta P:\alpha+\beta T(p)\le a_i(p)\ \text{for all }p},  
$$

and the upper allocation replaces $\le$ by $\ge$ and takes the corresponding infimum. The optimizing bitvar endpoints are the contact points of the supporting line. In a smooth continuous case, those contacts satisfy the equal-slope condition $a_i'(p)/T'(p)=\beta$ at both endpoints, but the hull method is more robust numerically.

For joint allocation geometry, use the convex hull of $(T(p),a_1(p),\ldots,a_d(p))$ and slice at $T=P$. For scalar ranges by unit, the 2D hull per unit is the clean solution.

Implementation to find TVaR, Allocations

```python
def tvar_na_grid_from_kappa(
    df,
    p_grid,
    loss_col="loss",
    prob_col="p_total",
    kappa_prefix="exeqa_",
    endpoint_atol=None,
    endpoint_rtol=0.0,
):
    """
    Compute TVaR natural allocations over a grid of p values.

    The input DataFrame has aggregate loss in loss_col, probability mass in
    prob_col, and conditional allocation columns kappa_i = P(X_i | X=x)
    whose names start with kappa_prefix.

    For p with VaR atom k, the allocation is

        [sum_{j>k} prob_j kappa_j + (F_k - p) kappa_k] / (1 - p).

    At p=1, the allocation is kappa at the largest positive-mass loss.
    """

    # Use a tight endpoint tolerance, only for floating-point endpoint noise.
    if endpoint_atol is None:
        endpoint_atol = 64 * np.finfo(float).eps

    # Identify allocation columns.
    kappa_cols = [c for c in df.columns if c.startswith(kappa_prefix)]
    if not kappa_cols:
        raise ValueError(f"No columns start with {kappa_prefix!r}.")

    # Collapse duplicate loss levels, summing probability and prob-weighted kappas.
    d0 = df[[loss_col, prob_col, *kappa_cols]].copy()

    weighted_kappa = {
        c: d0[prob_col].to_numpy(dtype=float) * d0[c].to_numpy(dtype=float)
        for c in kappa_cols
    }

    grouped = (
        d0.assign(**weighted_kappa)
          .groupby(loss_col, sort=True)
          .sum()
    )

    x = grouped.index.to_numpy(dtype=float)
    prob_raw = grouped[prob_col].to_numpy(dtype=float)
    kappa_num = grouped[kappa_cols].to_numpy(dtype=float)

    # Tolerate tiny negative probability noise, but not real negative mass.
    prob_tol = endpoint_atol * max(1.0, np.abs(prob_raw).sum())
    if np.any(prob_raw < -prob_tol):
        raise ValueError("prob_col contains negative probabilities beyond tolerance.")

    prob_raw = np.where(np.abs(prob_raw) <= prob_tol, 0.0, prob_raw)

    # Keep only positive-mass loss levels. Ess sup is then the last retained x.
    keep = prob_raw > 0.0
    if not np.any(keep):
        raise ValueError("No positive probability mass.")

    x = x[keep]
    prob_raw = prob_raw[keep]
    kappa_num = kappa_num[keep]

    # Recover conditional kappas at each aggregate loss level.
    kappa = np.divide(
        kappa_num,
        prob_raw[:, None],
        out=np.zeros_like(kappa_num),
        where=prob_raw[:, None] > 0.0,
    )

    # Normalize probabilities, then force the final CDF value to be exactly 1.
    # prob = prob_raw / prob_raw.sum()
    prob = prob_raw
    cdf = np.cumsum(prob)
    cdf[-1] = 1.0

    # Right-tail sums excluding the current VaR atom.
    prob_kappa = prob[:, None] * kappa
    tail_incl = np.cumsum(prob_kappa[::-1], axis=0)[::-1]
    tail_excl = np.vstack([
        tail_incl[1:],
        np.zeros((1, len(kappa_cols)), dtype=float),
    ])

    # Validate and clip p only for endpoint floating-point noise.
    p_raw = np.asarray(p_grid, dtype=float)

    if np.any(p_raw < -endpoint_atol) or np.any(p_raw > 1.0 + endpoint_atol):
        raise ValueError("p_grid contains values outside [0, 1] beyond tolerance.")

    p = np.clip(p_raw, 0.0, 1.0)

    out = np.empty((len(p), len(kappa_cols)), dtype=float)

    # Handle p=1, including endpoint noise such as 1 - 1e-16 or 1 + 1e-16.
    is_one = np.isclose(
        p,
        1.0,
        atol=endpoint_atol,
        rtol=endpoint_rtol,
    )

    # Main TVaR allocation calculation for p < 1.
    mask = ~is_one
    pp = p[mask]

    if pp.size:
        k = np.searchsorted(cdf, pp, side="left")

        # Defensive guard against any remaining CDF endpoint roundoff.
        k = np.minimum(k, len(cdf) - 1)

        atom_weight = cdf[k] - pp
        atom_weight = np.maximum(atom_weight, 0.0)

        numerator = tail_excl[k] + atom_weight[:, None] * kappa[k]
        denominator = 1.0 - pp

        out[mask] = numerator / denominator[:, None]

    # TVaR_1 is ess sup; allocate by kappa at largest positive-mass loss.
    out[is_one] = kappa[np.argmax(x)]

    return pd.DataFrame(
        out,
        index=pd.Index(p, name="p"),
        columns=kappa_cols,
    )
```


now we have df with cols exeqa_total and \_i (= tvar(p) and ai(p) p = index). what do you recommend as the best way to do the next step and actually get the slices range at P? as a function of P with an anticiaption we may want it as a function of P?

OTHER MODEL PROPOSAL

```python

def _cross(o, a, b):
    """
    Return the signed cross product for the turn o -> a -> b.
    """

    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _collapse_same_x(x, y, p, side, x_atol):
    """
    Collapse duplicate or near-duplicate x values.

    For a lower hull, keep the smallest y at each x.
    For an upper hull, keep the largest y at each x.
    """

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    p = p[order]

    out = []
    start = 0

    while start < len(x):
        stop = start + 1

        while stop < len(x) and np.isclose(x[stop], x[start], atol=x_atol, rtol=0.0):
            stop += 1

        yy = y[start:stop]

        if side == "lower":
            j = start + int(np.argmin(yy))
        else:
            j = start + int(np.argmax(yy))

        out.append((x[j], y[j], p[j]))
        start = stop

    return out


def _monotone_hull(x, y, p, side, x_atol=None, hull_atol=None):
    """
    Compute the lower or upper monotone hull of points (x, y).

    The returned frame has columns price, allocation, and p. Consecutive
    points define the hull edges used for vertical slicing.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(p)
    x = x[finite]
    y = y[finite]
    p = p[finite]

    if x_atol is None:
        x_atol = 256 * np.finfo(float).eps * max(1.0, np.max(np.abs(x)))

    if hull_atol is None:
        hull_atol = 256 * np.finfo(float).eps

    points = _collapse_same_x(x, y, p, side=side, x_atol=x_atol)
    hull = []

    for point in points:
        while len(hull) >= 2:
            turn = _cross(hull[-2], hull[-1], point)

            scale = max(
                1.0,
                abs((hull[-1][0] - hull[-2][0]) * (point[1] - hull[-2][1])),
                abs((hull[-1][1] - hull[-2][1]) * (point[0] - hull[-2][0])),
            )
            tol = hull_atol * scale

            if side == "lower" and turn <= tol:
                hull.pop()
            elif side == "upper" and turn >= -tol:
                hull.pop()
            else:
                break

        hull.append(point)

    return pd.DataFrame(hull, columns=["price", "allocation", "p"])


def build_na_hulls(
    na_df,
    price_col="exeqa_total",
    unit_prefix="exeqa_",
    unit_cols=None,
    x_atol=None,
    hull_atol=None,
):
    """
    Build lower and upper allocation hulls for all units.

    na_df is indexed by p and contains price_col = TVaR_p(X) and allocation
    columns a_i(p). The hulls can then be sliced repeatedly for any target
    price P.
    """

    if unit_cols is None:
        unit_cols = [
            c for c in na_df.columns
            if c.startswith(unit_prefix) and c != price_col
        ]

    p = na_df.index.to_numpy(dtype=float)
    price = na_df[price_col].to_numpy(dtype=float)

    hulls = {}

    for col in unit_cols:
        allocation = na_df[col].to_numpy(dtype=float)

        hulls[col] = {
            "lower": _monotone_hull(
                price,
                allocation,
                p,
                side="lower",
                x_atol=x_atol,
                hull_atol=hull_atol,
            ),
            "upper": _monotone_hull(
                price,
                allocation,
                p,
                side="upper",
                x_atol=x_atol,
                hull_atol=hull_atol,
            ),
        }

    return hulls


def _slice_hull(hull, prices, price_atol=None):
    """
    Evaluate a piecewise-linear hull at target prices.

    The returned columns identify the allocation value and the two TVaR
    levels p1, p2, with weight w2 on p2.
    """

    x = hull["price"].to_numpy(dtype=float)
    y = hull["allocation"].to_numpy(dtype=float)
    p = hull["p"].to_numpy(dtype=float)

    prices = np.asarray(prices, dtype=float)

    if price_atol is None:
        price_atol = 256 * np.finfo(float).eps * max(1.0, np.max(np.abs(x)))

    if np.any(prices < x[0] - price_atol) or np.any(prices > x[-1] + price_atol):
        raise ValueError("Target price lies outside the hull price range.")

    prices = np.clip(prices, x[0], x[-1])

    if len(x) == 1:
        value = np.full_like(prices, y[0], dtype=float)
        p1 = np.full_like(prices, p[0], dtype=float)
        p2 = np.full_like(prices, p[0], dtype=float)
        w2 = np.ones_like(prices, dtype=float)

        return pd.DataFrame({
            "allocation": value,
            "p1": p1,
            "p2": p2,
            "w2": w2,
        })

    edge = np.searchsorted(x, prices, side="right") - 1
    edge = np.clip(edge, 0, len(x) - 2)

    x1 = x[edge]
    x2 = x[edge + 1]
    y1 = y[edge]
    y2 = y[edge + 1]

    denom = x2 - x1

    w2 = np.divide(
        prices - x1,
        denom,
        out=np.zeros_like(prices, dtype=float),
        where=np.abs(denom) > price_atol,
    )
    w2 = np.clip(w2, 0.0, 1.0)

    value = (1.0 - w2) * y1 + w2 * y2

    return pd.DataFrame({
        "allocation": value,
        "p1": p[edge],
        "p2": p[edge + 1],
        "w2": w2,
    })


def slice_na_hulls(hulls, prices, price_atol=None):
    """
    Slice all unit hulls at one or more target prices.

    Returns a tidy DataFrame indexed by price and unit. The min and max
    columns give the natural allocation range. The p1, p2, and w2 columns
    identify the bitvars that achieve each bound.
    """

    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    records = []

    for unit, pair in hulls.items():
        lower = _slice_hull(pair["lower"], prices, price_atol=price_atol)
        upper = _slice_hull(pair["upper"], prices, price_atol=price_atol)

        for j, price in enumerate(prices):
            records.append({
                "price": price,
                "unit": unit,
                "min": lower.loc[j, "allocation"],
                "max": upper.loc[j, "allocation"],
                "width": upper.loc[j, "allocation"] - lower.loc[j, "allocation"],
                "min_p1": lower.loc[j, "p1"],
                "min_p2": lower.loc[j, "p2"],
                "min_w2": lower.loc[j, "w2"],
                "max_p1": upper.loc[j, "p1"],
                "max_p2": upper.loc[j, "p2"],
                "max_w2": upper.loc[j, "w2"],
            })

    return (
        pd.DataFrame.from_records(records)
          .set_index(["price", "unit"])
          .sort_index()
    )
    
```

