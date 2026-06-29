
> **SUPERSEDED 2026-06-29** — moved to `dev/done`; superseded by `dev/plan-bivariate-legs.md` (see its consolidation note). Realized as Phase 2 (a116–a117).

# Specification: Property Catastrophe Reinstatement Analysis

## 1. Purpose

Extend `aggregate` to model property catastrophe occurrence reinsurance with paid and free reinstatements and to produce complete gross, ceded, and net underwriting exhibits.

The feature should calculate the probability distributions of:

- gross premium, loss, and underwriting result;
- ceded premium, loss, and underwriting result;
- net premium, loss, and underwriting result;
- reinstatement premium;
- unlimited occurrence ceded loss.

The central actuarial feature is that ceded premium is stochastic. The cedant pays a fixed deposit premium at inception and additional reinstatement premium when catastrophe recoveries consume reinstatement capacity. Consequently, net premium is stochastic and dependent on gross and ceded losses.

The implementation should calculate one canonical bivariate aggregate distribution using FFT2 and obtain all other distributions by deterministic pushforward transformations. No additional convolution is required after the canonical joint distribution has been computed.

The initial implementation should support one reinstatement basis: either one occurrence layer or one programme-level annual aggregate cap and reinstatement premium function applied to total unlimited occurrence recoveries. Independently reinstated layers require a higher-dimensional state and are outside the initial scope.

## 2. Existing `aggregate` functionality

The `REFACTOR` branch already provides the essential numerical foundation.

For an occurrence reinsurance programme, the gross severity $X$ is mapped deterministically into any two of:

- gross: $X$;
- ceded: $c(X)$;
- net: $n(X)=X-c(X)$.

The function `build_netceded_joint` constructs a bivariate per-occurrence severity by scattering the gross severity probability mass onto the selected pair of image coordinates. A shared frequency distribution is then applied through FFT2 to produce the annual joint aggregate distribution.

For the gross/ceded view, the canonical construction is therefore based on the per-occurrence vector
$$
\bigl(X,c(X)\bigr).
$$
If the annual claim count is $N$, the resulting annual vector is
$$
(L,R)=
\left(
\sum_{i=1}^N X_i,
\sum_{i=1}^N c(X_i)
\right),
$$
where:

- $L$ is annual gross loss;
- $R$ is annual ceded loss to the occurrence programme before any reinstatement or annual aggregate limitation.

The existing FFT2 machinery computes the joint distribution of $(L,R)$ while preserving the dependence arising from both occurrence severity and the shared random event count.

The reinstatement extension should treat the distribution of $(L,R)$ as its single probabilistic source of truth.

## 3. Basic occurrence reinsurance model

Let annual gross catastrophe loss be
$$
L=
\sum_{i=1}^N X_i,
$$
where $N$ is the annual occurrence count and the $X_i$ are identically distributed occurrence severities.

For a single full-share occurrence layer of $y$ xs $a$, unlimited ceded loss from occurrence $i$ is
$$
C_i=
y\wedge(X_i-a)_+.
$$
The unlimited annual occurrence recovery is
$$
R=
\sum_{i=1}^N C_i.
$$
The word “unlimited” means that each occurrence is subject to the occurrence limit $y$, but no annual restriction has yet been imposed on the number or total amount of occurrence recoveries.

For a more general occurrence programme, $c(x)$ is the existing piecewise-linear cession function produced by `make_ceder_netter`, and
$$
R=
\sum_{i=1}^N c(X_i).
$$
The reinstatement analysis requires only the joint annual distribution of $(L,R)$ and does not otherwise depend on the form of $c$.

## 4. Standard reinstatement model

Suppose the treaty has:

- occurrence limit $y$;
- $m$ full reinstatements;
- fixed deposit premium $D$;
- reinstatement premium function $h(R)$.

For standard equal-price reinstatements charged at 100% of an original rate on line $r$, the actual ceded loss is
$$
A(R)=
R\wedge(m+1)y,
$$
and reinstatement premium is
$$
\mathrm{RP}(R)=
r(R\wedge my).
$$
The original limit and $m$ reinstated limits provide total annual ceded-loss capacity of $(m+1)y$. Only $my$ of capacity can be reinstated, so reinstatement premium stops increasing once unlimited occurrence recoveries reach $my$.

Partial losses consume reinstatement capacity pro rata. A recovery of $0.25y$ uses $0.25y$ of reinstatement capacity; it does not consume an indivisible reinstatement event.

Under automatic reinstatements pro rata as to amount and not as to time, annual recovery and annual reinstatement premium depend on $R$ but not on the chronological ordering of the individual occurrences.

## 5. General reinstatement premium function

The implementation should not restrict reinstatement premium to
$$
\mathrm{RP}(R)=
r(R\wedge my).
$$
Instead, it should accept a general function
$$
h:[0,\infty)\longrightarrow[0,\infty),
$$
with
$$
\mathrm{RP}(R)=
h(R).
$$
For conventional treaty terms, $h$ should be:

- nonnegative;
- nondecreasing;
- piecewise linear;
- constant once all paid reinstatement capacity has been consumed;
- normally satisfy $h(0)=0$.

The mathematical transformation does not strictly require monotonicity, but the public reinstatement API should normally validate that a supplied function is nondecreasing. An escape hatch may permit an arbitrary callable for advanced users.

The actual recovery function may initially remain
$$
A(R)=
R\wedge Y,
$$
where
$$
Y=
(m+1)y
$$
is the annual ceded-loss capacity.

Separating the recovery function $A$ from the premium function $h$ is important. Two contracts may provide the same recovery capacity but charge different reinstatement premiums.

### 5.1 Reinstatement schedules

Let reinstatement tranche $j$ have width $w_j$ and price multiplier $\alpha_j$ relative to the base rate on line $r$. The cumulative reinstatement premium is
$$
h(R)=
r\sum_{j=1}^k
\alpha_j
\left[
(R-b_j)_+\wedge w_j
\right],
$$
where $b_j$ is the amount of unlimited recovery consumed before tranche $j$ begins.

For equal full-limit reinstatements,
$$
w_j=
y,
\qquad
b_j=
(j-1)y.
$$
A schedule consisting of:

- one free reinstatement;
- one reinstatement at 50%;
- three reinstatements at 100%;

has multipliers
$$
(\alpha_1,\ldots,\alpha_5)=
(0,0.5,1,1,1).
$$
Its premium function is
$$
h(R)=
r\left[
0.5\bigl((R-y)_+\wedge y\bigr)
+
\sum_{j=3}^5
\bigl((R-(j-1)y)_+\wedge y\bigr)
\right].
$$
The first reinstatement is free and contributes no premium. The second contributes premium at half the base rate, and the final three contribute premium at the full rate.

### 5.2 Suggested helper object

Introduce an immutable reinstatement-terms object, provisionally:

```python
@dataclass(frozen=True)
class ReinstatementTerms:
    limit: float
    rates: tuple[float, ...]
    rate_on_line: float
    deposit_premium: float
```

Here:

- `limit` is $y$;
    
- `rates[j]` is the price multiplier $\alpha_{j+1}$ for reinstatement $j+1$;
    
- `len(rates)` is $m$;
    
- `rate_on_line` is $r$;
    
- `deposit_premium` is $D$.
    

Derived properties should include:

```python
terms.n_reinstatements
terms.reinstatement_capacity       # m * y
terms.total_recovery_capacity      # (m + 1) * y
terms.maximum_reinstatement_premium
```

Methods should include:

```python
terms.recovery(r)
terms.reinstatement_premium(r)
terms.ceded_premium(r)
```

A lower-level alternative constructor should support general tranche widths:

```python
ReinstatementTerms.from_tranches(
    recovery_limit=...,
    widths=[...],
    rates=[...],
    rate_on_line=...,
    deposit_premium=...,
)
```

A callable override may be supported:

```python
ReinstatementTerms.from_callable(
    recovery_limit=...,
    premium_function=...,
    deposit_premium=...,
)
```

The object should evaluate NumPy arrays without Python loops.

## 6. Gross, ceded, and net accounting variables

Let $P_G$ denote fixed gross premium and $D$ fixed ceded deposit premium.

For every annual outcome $(L,R)$, define actual ceded loss  
$$  
A=  
A(R),  
$$  
and reinstatement premium  
$$  
\mathrm{RP}=  
h(R).  
$$

### 6.1 Gross position

Gross premium is fixed:  
$$  
P_{\mathrm{gross}}=  
P_G.  
$$  
Gross loss is  
$$  
L_{\mathrm{gross}}=  
L.  
$$  
Gross underwriting result is  
$$  
U_{\mathrm{gross}}=  
P_G-L.  
$$

### 6.2 Ceded position

Ceded premium paid by the cedant is  
$$  
P_{\mathrm{ceded}}=  
D+h(R).  
$$  
Ceded loss recovery is  
$$  
L_{\mathrm{ceded}}=  
A(R).  
$$  
For an underwriting exhibit viewed from the cedant’s transaction with the reinsurer, ceded underwriting result should be defined as premium paid less recovery received:  
$$  
U_{\mathrm{ceded}}=  
D+h(R)-A(R).  
$$  
This sign convention makes ceded underwriting result a cost to the cedant. It is also the reinsurer’s underwriting result before expenses and investment income.

### 6.3 Net position

Net premium retained by the cedant is  
$$  
P_{\mathrm{net}}=  
P_G-D-h(R).  
$$  
Net loss retained by the cedant is  
$$  
L_{\mathrm{net}}=  
L-A(R).  
$$  
Net underwriting result is  
$$  
U_{\mathrm{net}}=  
P_G-D-h(R)-L+A(R).  
$$  
Equivalently,  
$$  
U_{\mathrm{net}}=  
U_{\mathrm{gross}}-U_{\mathrm{ceded}}.  
$$

### 6.4 Pointwise accounting identities

The following identities must hold at every $(L,R)$ grid point:  
$$  
P_{\mathrm{gross}}=  
P_{\mathrm{ceded}}+P_{\mathrm{net}},  
$$  
$$  
L_{\mathrm{gross}}=  
L_{\mathrm{ceded}}+L_{\mathrm{net}},  
$$  
and  
$$  
U_{\mathrm{gross}}=  
U_{\mathrm{ceded}}+U_{\mathrm{net}}.  
$$  
These are stronger than moment checks and should be tested before any rebucketing.

## 7. Examples

Consider a 100 xs 100 occurrence treaty with one reinstatement at 100%, deposit premium 10, and rate on line 10%. Thus,  
$$  
y=  
100,  
\qquad  
m=  
1,  
\qquad  
D=  
10,  
\qquad  
r=  
0.10.  
$$  
The recovery and premium functions are  
$$  
A(R)=  
R\wedge200,  
$$  
and  
$$  
h(R)=  
0.10(R\wedge100).  
$$

### 7.1 One event

Suppose the only occurrence loss is 175. Unlimited ceded recovery is  
$$  
R=  
100\wedge(175-100)=  
75.  
$$  
Actual ceded loss is  
$$  
A=  
75.  
$$  
Reinstatement premium is  
$$  
\mathrm{RP}=  
0.10(75)=  
7.5.  
$$  
Total ceded premium is  
$$  
P_{\mathrm{ceded}}=  
10+7.5=  
17.5.  
$$  
Ceded underwriting result is  
$$  
U_{\mathrm{ceded}}=  
17.5-75=  
-57.5.  
$$

### 7.2 Two events

Suppose occurrence losses are 250 and 150. Unlimited occurrence recoveries are 100 and 50, so  
$$  
R=  
150.  
$$  
Actual recovery is  
$$  
A=  
150.  
$$  
Reinstatement premium is capped at  
$$  
\mathrm{RP}=  
0.10(100)=  
10.  
$$  
Total ceded premium is 20 and ceded underwriting result is  
$$  
U_{\mathrm{ceded}}=  
20-150=  
-130.  
$$

### 7.3 Exhaustion after three or more events

Suppose occurrence losses are 200, 110, and 200. Unlimited occurrence recoveries are 100, 10, and 100, so  
$$  
R=  
210.  
$$  
Actual recovery is capped at  
$$  
A=  
200.  
$$  
Reinstatement premium remains  
$$  
\mathrm{RP}=  
10.  
$$  
Total ceded premium is 20 and ceded underwriting result is  
$$  
U_{\mathrm{ceded}}=  
20-200=  
-180.  
$$

## 8. Canonical computational architecture

The implementation should have two stages.

### Stage 1: calculate the canonical joint distribution

Use the existing FFT2 occurrence-bivariate machinery to calculate  
$$  
p_{ij}=  
P{L=l_i,\ R=r_j}.  
$$  
The per-occurrence severity is the deterministic image  
$$  
X\longmapsto  
\bigl(X,c(X)\bigr).  
$$  
The annual joint density is obtained through the existing shared-frequency FFT2 calculation.

No reinstatement assumptions should enter this FFT2 calculation. The resulting object represents the unlimited occurrence programme and may be reused for multiple reinstatement structures and pricing alternatives.

### Stage 2: deterministic pushforwards

At every joint grid point $(l_i,r_j)$, evaluate all accounting variables:

```python
actual_ceded_loss = recovery(r_j)
rp = premium_function(r_j)

gross_premium = gross_premium_constant
gross_loss = l_i
gross_uw = gross_premium_constant - l_i

ceded_premium = deposit_premium + rp
ceded_loss = actual_ceded_loss
ceded_uw = ceded_premium - ceded_loss

net_premium = gross_premium_constant - ceded_premium
net_loss = l_i - ceded_loss
net_uw = net_premium - net_loss
```

Each scalar variable is then obtained by summing or scattering the joint probability masses $p_{ij}$ according to its transformed value.

## 9. Pushforward and diagonal summarisation

### 9.1 Why another FFT is not appropriate

After FFT2 has produced the joint density of $(L,R)$, quantities such as  
$$  
L-A(R)  
$$  
or  
$$  
P_G-D-h(R)-L+A(R)  
$$  
are deterministic, generally nonlinear functions of both coordinates.

The two components are not independent, so their marginal distributions cannot be convolved. Another FFT would be incorrect unless the full joint characteristic function were transformed analytically for the specific mapping, which is neither general nor necessary.

The correct calculation is a pushforward of the joint probability matrix.

### 9.2 Computational complexity

Suppose the joint density has shape  
$$  
(n_L,n_R).  
$$  
Every scalar output requires inspecting each nonzero joint cell at least once, so direct pushforward complexity is  
$$  
O(n_Ln_R).  
$$  
For square grids with $n_L=n_R=n$, this is  
$$  
O(n^2).  
$$  
With axis sizes between $2^{10}$ and $2^{12}$, the joint matrix contains approximately:

|Axis size|Number of cells|
|--:|--:|
|$2^{10}=1{,}024$|$1.05$ million|
|$2^{11}=2{,}048$|$4.19$ million|
|$2^{12}=4{,}096$|$16.78$ million|

These sizes are entirely practical for vectorised NumPy operations. The FFT2 itself and storage of the dense matrix are likely to dominate or be comparable to the pushforward cost.

A single pass over 1–17 million cells should ordinarily be quick. The implementation should begin with NumPy, not Numba.

### 9.3 Avoid materialising unnecessary full coordinate matrices

The implementation should use broadcasting:

```python
L = grid_l[:, None]
R = grid_r[None, :]
```

Then calculate transformed arrays as needed:

```python
A = recovery(R)
RP = premium_function(R)
net_loss = L - A
net_uw = gross_premium - deposit_premium - RP - net_loss
```

`A` and `RP` depend only on the $R$ axis and need shape `(1, n_R)`. There is no reason to allocate them at full matrix size.

Variables involving $L$ will broadcast to `(n_L, n_R)`.

### 9.4 Efficient histogramming

For a uniform output grid with origin $z_0$ and bucket size `bs`, transformed values can be converted to fractional bucket coordinates:  
$$  
u=  
\frac{z-z_0}{\mathrm{bs}}.  
$$  
For nearest rebucketing, obtain integer indices and use:

```python
np.bincount(index.ravel(), weights=density.ravel(), minlength=n_out)
```

For linear rebucketing, calculate lower and upper indices and corresponding weights, then use two `np.bincount` calls:

```python
out = np.bincount(k0.ravel(), weights=(p * w0).ravel(), minlength=n_out)
out += np.bincount(k1.ravel(), weights=(p * w1).ravel(), minlength=n_out)
```

`np.bincount` should generally be faster than `np.add.at` for dense one-dimensional output indexing.

The calculation should process all desired transformed variables either:

- sequentially, to minimise memory; or
    
- in small groups that share intermediate arrays.
    

The code should not retain nine full transformed matrices simultaneously.

### 9.5 Exact diagonal collapse where possible

When:

- $L$ and $R$ share a common bucket size;
    
- the transformation is affine with integer-grid coefficients;
    
- no nonlinear cap or premium function intervenes;
    

a diagonal summation can be performed by exact integer index arithmetic.

For the general reinstatement case, $A(R)$ and $h(R)$ create piecewise-affine mappings. The same integer-index approach may still work within each piece when all breakpoints align with the grid, but a generic linear scatter is simpler and more robust.

### 9.6 Numba policy

Do not require Numba for v1.0.

Implement the pushforward using NumPy broadcasting and `np.bincount`, benchmark it on grids from $2^{10}\times2^{10}$ through $2^{12}\times2^{12}$, and only add a compiled path if profiling demonstrates a material bottleneck.

A Numba implementation may become useful if:

- many reinstatement alternatives are evaluated repeatedly against the same joint distribution;
    
- arbitrary Python callables prevent vectorisation;
    
- sparse or chunked processing is introduced;
    
- multiple outputs are fused into one memory pass.
    

The public API should not depend on Numba, and numerical results from any accelerated implementation must match the NumPy reference path.

### 9.7 Chunking

A $4096\times4096$ float64 matrix occupies approximately 134 MB. A full transformed float64 matrix of the same size consumes another 134 MB. Several simultaneous arrays can therefore create unnecessary memory pressure.

The pushforward implementation should support row chunks:

```python
for rows in row_chunks:
    l = grid_l[rows, None]
    values = transform(l, r)
    accumulate(values, density[rows])
```

Chunking preserves $O(n^2)$ time while bounding temporary memory. A reasonable default chunk size can be chosen from a memory budget.

For small grids, the unchunked vectorised path may be faster. The implementation can select automatically.

## 10. Generic bivariate pushforward API

Add a generic scalar pushforward method to `BivariateDistribution`.

Suggested interface:

```python
def pushforward(
    self,
    function,
    *,
    bs=None,
    log2=None,
    window=None,
    scheme="linear",
    name=None,
    value_type=None,
    chunk_size=None,
):
    ...
```

The function should accept broadcast-compatible axis arrays:

```python
function(x, y) -> ndarray
```

The result should be a one-dimensional grid-distribution object compatible with the distributions currently used by `Aggregate` and `PnL`.

The method should:

1. determine or accept an output window;
    
2. determine or accept a bucket size;
    
3. evaluate the function in chunks;
    
4. scatter probability mass onto the output grid;
    
5. retain total probability;
    
6. calculate any deficit or clipped mass;
    
7. store metadata describing the source axes and transformation;
    
8. expose the resulting density, CDF, survival function, moments, and quantiles.
    

A more efficient internal method should accept a known transformation structure and precomputed one-axis functions, but the public generic method is valuable beyond reinstatements.

Suggested internal form:

```python
_pushforward_values(
    value_builder,
    *,
    grid,
    scheme,
    chunk_size,
)
```

## 11. Reinstatement analysis object

Introduce a dedicated result object, provisionally:

```python
class ReinstatementAnalysis:
    ...
```

The object should contain:

```python
analysis.source
analysis.terms
analysis.gross_premium
analysis.distributions
analysis.exhibit
analysis.audit
analysis.metadata
```

### 11.1 `source`

`source` is the canonical `BivariateDistribution` for $(L,R)$.

Axis names should be explicit:

```python
source.axis_names == ("gross_loss", "unlimited_ceded_loss")
```

### 11.2 `terms`

`terms` is the immutable `ReinstatementTerms` object or another object satisfying the same protocol:

```python
terms.recovery(r)
terms.reinstatement_premium(r)
terms.deposit_premium
```

### 11.3 `distributions`

Store named one-dimensional distributions:

```python
{
    "gross_premium": ...,
    "gross_loss": ...,
    "gross_uw": ...,
    "ceded_premium": ...,
    "ceded_loss": ...,
    "ceded_uw": ...,
    "net_premium": ...,
    "net_loss": ...,
    "net_uw": ...,
    "reinstatement_premium": ...,
    "unlimited_ceded_loss": ...,
}
```

Fixed gross premium may be represented as a degenerate one-point distribution. That keeps the downstream reporting API uniform.

### 11.4 Suggested construction

```python
analysis = agg.reinstatement_analysis(
    gross_premium=...,
    terms=ReinstatementTerms(
        limit=...,
        rates=(...),
        rate_on_line=...,
        deposit_premium=...,
    ),
    percentiles=(0.90, 0.95, 0.99, 0.995, 0.996, 0.999),
)
```

The method should:

1. require an updated `Aggregate`;
    
2. require occurrence reinsurance;
    
3. build or reuse the gross/ceded bivariate distribution;
    
4. apply the reinstatement transformations;
    
5. return a `ReinstatementAnalysis`.
    

The bivariate source should be cached independently of the terms so multiple reinstatement schedules can be evaluated without repeating FFT2.

A useful separation is:

```python
joint = agg.occ_bivariate(views=("gross", "ceded"))
analysis = joint.reinstatement_analysis(
    gross_premium=...,
    terms=...,
)
```

and the convenience wrapper:

```python
analysis = agg.reinstatement_analysis(...)
```

## 12. Relationship to `PnL`

`PnL` currently represents a consideration against a one-dimensional risky leg, with net payoff obtained by a deterministic transform of that risky leg.

The reinstatement problem is conceptually a bivariate P&L because:

- gross loss depends on $L$;
    
- ceded recovery depends on $R$;
    
- reinstatement premium depends on $R$;
    
- net underwriting result depends jointly on $L$ and $R$.
    

The existing `PnL` class should not be forced to carry the joint distribution directly unless `PnL` is intentionally generalized to multivariate obligations.

The cleaner v1.0 architecture is:

- `BivariateDistribution` owns generic two-dimensional pushforwards;
    
- `ReinstatementAnalysis` owns contract interpretation and GCN reporting;
    
- each resulting one-dimensional underwriting distribution may use the same reporting conventions and distribution structures as `PnL`;
    
- common statistical and plotting utilities should be factored out rather than duplicated.
    

In particular, `PnL` correctly reports SD rather than CV for a margin near zero. The reinstatement exhibit should follow the same rule for underwriting result.

A later development could introduce a general concept such as:

```python
JointPnL
```

or:

```python
BivariatePnL
```

but reinstatements do not require that abstraction to be completed for v1.0.

## 13. Exhibit specification

The headline exhibit, `summary_df` should have columns:

- Gross;
    
- Ceded;
    
- Net;
    
- Impact

- Percentage Impact.
    

Rows should be:

1. Premium;
    
2. CV(Premium);
    
3. Loss;
    
4. CV(Loss);
    
5. Underwriting Result;
    
6. SD(Underwriting Result);
    
7. percentile rows at 90%, 95%, 99%, 99.5%, 99.6%, and 99.9%.
    

Suggested display:

|Measure|Gross|Ceded|Net|Impact|Pct Impact|
|---|--:|--:|--:|--:|--:|
|Premium||||||
|CV(Premium)||||||
|Loss||||||
|CV(Loss)||||||
|Underwriting Result||||||
|SD(Underwriting Result)||||||
|90.0%||||||
|95.0%||||||
|99.0%||||||
|99.5%||||||
|99.6%||||||
|99.9%||||||

### 13.1 Premium row

Report mean premium:

- Gross: $P_G$;
    
- Ceded: $P(D+h(R))$;
    
- Net: $P(P_G-D-h(R))$;
    
- Impact: Net minus Gross, or equivalently negative Ceded.
    

Since premium is a cost/retention amount rather than a risk measure, the Impact convention should be clearly labelled. Recommended:  
$$  
\text{Impact}_{\mathrm{premium}}=  
\text{Net}-\text{Gross}=  
-\text{Ceded}.  
$$

### 13.2 CV(Premium) row

Report:  
$$  
\operatorname{CV}(P)=  
\frac{\operatorname{SD}(P)}  
{P(P)}  
$$  
when the mean is materially nonzero.

Gross premium has CV zero because it is fixed.

Ceded and net premium generally have nonzero CV because reinstatement premium is stochastic.

## The Impact column should report:  
$$  
\operatorname{CV}(P_{\mathrm{net}})-
\operatorname{CV}(P_{\mathrm{gross}})=  
\operatorname{CV}(P_{\mathrm{net}})  
$$  
since the gross premium is fixed and has zero CV. A ratio is not useful because the gross CV is zero.

### 13.3 Loss row

Report mean loss (note, write $P(X)$ for expectation of $X$ wrt probability $P$)

- Gross: $PL$;
    
- Ceded: $PA(R)$;
    
- Net: $P[L-A(R)]$;
    
- Impact:  
    $$  
    \text{Impact}_{\mathrm{loss}}=  
    \text{Gross}-\text{Net}=  
    \text{Ceded}.  
    $$  
    A positive impact represents reduction in retained expected loss.
    

### 13.4 CV(Loss) row

Report gross, ceded, and net loss CV.

## Recommended impact:  
$$  
\text{Impact}_{\mathrm{CV(loss)}}=  
\operatorname{CV}(L) -
\operatorname{CV}(L-A(R)).  
$$  
A positive value means reinsurance reduced the retained loss CV.

A relative impact may also be exposed in a detailed table:  
$$  
1-  
\frac{\operatorname{CV}(L-A(R))}  
{\operatorname{CV}(L)}.  
$$

### 13.5 Underwriting-result row

Report mean underwriting result:

- Gross: $P[P_G-L]$;
    
- Ceded: $P[D+h(R)-A(R)]$;
    
- Net: $P[P_G-D-h(R)-L+A(R)]$.
    

## With the selected ceded sign convention:  
$$  
U_{\mathrm{gross}}=  
U_{\mathrm{ceded}}+U_{\mathrm{net}}.  
$$  
Recommended impact:  
$$  
\text{Impact}_{\mathrm{UW}}=  
P[U_{\mathrm{net}}] -
P[U_{\mathrm{gross}}].  
$$  
This value is the negative expected ceded underwriting result:  
$$  
\text{Impact}_{\mathrm{UW}}=  
-P[U_{\mathrm{ceded}}].  
$$

### 13.6 SD(Underwriting Result) row

Report SD rather than CV because expected underwriting result may be zero or negative.

## Recommended impact:  
$$  
\text{Impact}_{\mathrm{SD(UW)}}=  
\operatorname{SD}(U_{\mathrm{gross}}) -
\operatorname{SD}(U_{\mathrm{net}}).  
$$  
A positive value represents a reduction in underwriting-result volatility.

Also expose the relative reduction:  
$$  
1-  
\frac{\operatorname{SD}(U_{\mathrm{net}})}  
{\operatorname{SD}(U_{\mathrm{gross}})}.  
$$

### 13.7 Percentile rows

The percentile rows require an unambiguous adverse-direction convention.

Underwriting result is a payoff: low values are adverse. Rows labelled 90%, 95%, and so forth are more naturally interpreted as upper percentiles of an adverse loss variable.

Define underwriting loss:  
$$  
W=  
-U.  
$$  
Then:

- gross underwriting loss:  
    $$  
    W_{\mathrm{gross}}=  
    L-P_G;  
    $$
    
- ceded underwriting loss:  
    $$  
    W_{\mathrm{ceded}}=  
    A(R)-D-h(R);  
    $$
    
- net underwriting loss:  
    $$  
    W_{\mathrm{net}}=  
    L-A(R)-P_G+D+h(R).  
    $$
    

The percentile rows should report upper percentiles of $W$:  
$$  
q_p(W),  
\qquad  
p\in  
{0.90,0.95,0.99,0.995,0.996,0.999}.  
$$  
This convention makes higher values uniformly worse and aligns the percentile presentation with standard loss and capital reporting.

Equivalent lower underwriting-result quantiles may be shown in labels or tooltips:  
$$  
q_p(-U)=  
-q_{1-p}^{+}(U),  
$$  
subject to the package’s discrete quantile convention.

## Recommended percentile impact:  
$$  
\text{Impact}_p=  
q_p(W_{\mathrm{gross}})

q_p(W_{\mathrm{net}}).  
$$  
A positive value means the reinsurance improves the adverse underwriting outcome at percentile $p$.

The Ceded column should report $q_p(W_{\mathrm{ceded}})$, but users should not expect percentile additivity:  
$$  
q_p(W_{\mathrm{gross}})  
\ne  
q_p(W_{\mathrm{ceded}})  
+  
q_p(W_{\mathrm{net}})  
$$  
in general.

The exhibit should state that mean accounting values add, but standard deviations and percentiles do not.

## 14. Detailed statistics table

In addition to the headline exhibit, provide a machine-readable long-form table (`stats_df`) with fields such as:

```text
basis
measure
statistic
value
impact_absolute
impact_relative
```

Suggested bases:

```text
gross
ceded
net
```

Suggested measures:

```text
premium
loss
uw_result
uw_loss
reinstatement_premium
unlimited_ceded_loss
```

Suggested statistics:

```text
mean
sd
cv
skew
q_0.90
q_0.95
q_0.99
q_0.995
q_0.996
q_0.999
```

The headline exhibit can then be a formatted view of this canonical table.

## 15. Exact moments versus displayed-density moments

Pushforward rebucketing may slightly alter second and higher moments (`validation_df`). The implementation should distinguish:

1. moments calculated directly from transformed values on the source joint grid;
    
2. moments calculated from the rebucketed one-dimensional display distribution.
    

For each transformed variable $Z=f(L,R)$, direct moments are:  
$$  
PZ=  
\sum_{i,j} p_{ij}f(l_i,r_j),  
$$  
$$  
PZ^2=  
\sum_{i,j} p_{ij}f(l_i,r_j)^2.  
$$  
Hence,  
$$  
\operatorname{Var}(Z)=  
PZ^2-(PZ)^2.  
$$  
These source-grid moments should drive the principal exhibit because they avoid additional rebucketing error.

The rebucketed distribution should drive quantiles, plotting, and survival probabilities.

The analysis object should report differences between direct and rebucketed moments as a numerical audit.

## 16. Numerical validation and audit

The analysis should expose an audit table containing at least the following checks.

### 16.1 Probability mass

For the canonical bivariate distribution:  
$$  
\sum_{i,j}p_{ij}\approx1.  
$$  
For every pushed-forward distribution:  
$$  
\sum_k p_k\approx1.  
$$

### 16.2 Premium identity

At the moment level:  
$$  
P[P_{\mathrm{gross}}]\approx  
P[P_{\mathrm{ceded}}]+P[P_{\mathrm{net}}].  
$$

### 16.3 Loss identity

At the moment level:  
$$  
PL\approx  
P[A(R)]+P[L-A(R)].  
$$

### 16.4 Underwriting identity

At the moment level:  
$$  
P[U_{\mathrm{gross}}]\approx  
P[U_{\mathrm{ceded}}]+P[U_{\mathrm{net}}].  
$$

### 16.5 Variance identities using covariance

Because gross equals ceded plus net:  
$$  
\operatorname{Var}(L)=  
\operatorname{Var}(A)  
+  
\operatorname{Var}(L-A)  
+  
2\operatorname{Cov}(A,L-A).  
$$  
The corresponding underwriting-result identity is:  
$$  
\operatorname{Var}(U_{\mathrm{gross}})=  
\operatorname{Var}(U_{\mathrm{ceded}})  
+  
\operatorname{Var}(U_{\mathrm{net}})  
+  
2\operatorname{Cov}(U_{\mathrm{ceded}},U_{\mathrm{net}}).  
$$  
These provide useful checks on the direct joint-grid calculations.

### 16.6 Function validation

Check that:

- recovery is finite and nonnegative;
    
- recovery does not exceed total contractual capacity;
    
- reinstatement premium is finite and nonnegative;
    
- a standard reinstatement premium function is nondecreasing;
    
- the supplied function evaluates vectorially or can be safely vectorized;
    
- breakpoints and rates are internally consistent.
    

## 17. Scope and restrictions

### 17.1 Exact supported case

The initial implementation is exact when actual ceded loss and reinstatement premium are functions of total unlimited annual occurrence recovery $R$:  
$$  
A=  
A(R),  
\qquad  
\mathrm{RP}=  
h(R).  
$$  
This includes:

- one reinstated occurrence layer;
    
- one programme-level reinstatement structure applied to total occurrence cessions;
    
- arbitrary free and paid reinstatement schedules;
    
- arbitrary increasing premium functions of $R$;
    
- partial reinstatements;
    
- annual aggregate recovery caps.
    

### 17.2 Independently reinstated layers

Suppose a programme contains layers indexed by $j$, with unlimited annual recoveries $R_j$. If each layer has its own reinstatement terms, then:  
$$  
A=  
\sum_j A_j(R_j),  
$$  
and  
$$  
\mathrm{RP}=  
\sum_j h_j(R_j).  
$$  
In general, these quantities cannot be determined from:  
$$  
R=  
\sum_jR_j.  
$$  
The exact state is:  
$$  
(L,R_1,\ldots,R_k).  
$$  
FFT2 is insufficient for an arbitrary independently reinstated tower.

The initial API must therefore reject or explicitly decline independent layer-level reinstatements unless there is only one reinstated basis.

It must not silently apply programme-level reinstatements to a tower whose legal terms apply separately by layer.

### 17.3 Time-dependent and optional reinstatements

The proposed model assumes annual outcomes are order invariant. It does not initially support:

- reinstatement premium pro rata as to time;
    
- optional or discretionary reinstatements;
    
- delayed reinstatement;
    
- nonpayment termination conditions;
    
- reinstatement terms depending on event date;
    
- event-count rather than amount-based restoration;
    
- peril- or territory-specific reinstatement restrictions.
    

Such contracts require a chronological state model rather than a pushforward from annual $R$ alone.

## 18. Plotting

Useful plots should include:

### 18.1 Canonical gross/unlimited-ceded contour

Generic `plot` method should show the joint density of $(L,R)$ with overlays for:

- $R=my$, where reinstatement premium reaches its maximum for equal rates;
    
- $R=(m+1)y$, where recovery reaches its annual maximum;
    
- any additional premium-function breakpoints.
    
Bivariate plots are best shown using log density. 

### 18.2 Reinstatement functions

Plot:

- actual ceded loss $A(R)$;
    
- reinstatement premium $h(R)$;
    
- total ceded premium $D+h(R)$;
    
- ceded underwriting result $D+h(R)-A(R)$.
    

### 18.3 Gross and net underwriting distributions

Overlay gross and net underwriting-loss survival curves or return-period curves.

### 18.4 Impact curve

## Plot:  
$$  
q_p(W_{\mathrm{gross}})

q_p(W_{\mathrm{net}})  
$$  
against return period or percentile.

## 19. Performance strategy

The expected performance profile is:

1. FFT2 calculation of the canonical $(L,R)$ distribution;
    
2. one or more $O(n_Ln_R)$ pushforward scans;
    
3. one-dimensional sorting, cumulative sums, and quantile extraction.
    

For axis sizes up to $2^{12}$, NumPy should be adequate.

Implementation order:

1. build a clear NumPy reference implementation;
    
2. use broadcasting and `np.bincount`;
    
3. add chunking to control temporary memory;
    
4. benchmark realistic models;
    
5. fuse transformations only if profiling warrants it;
    
6. consider Numba only after profiling.
    

The most valuable optimization is likely to calculate all direct moments in one chunked pass while constructing only those one-dimensional distributions required for quantiles and plots.

## 20. Proposed files and responsibilities

### `src/aggregate/bivariate.py`

Add:

- generic scalar `pushforward`;
    
- chunked histogram utility;
    
- direct transformed-moment utility;
    
- optional transformation metadata.
    

Possible methods:

```python
BivariateDistribution.pushforward(...)
BivariateDistribution.transformed_moments(...)
BivariateDistribution.map_statistics(...)
```

### `src/aggregate/reinstatement.py`

New module containing:

```python
ReinstatementTerms
ReinstatementAnalysis
make_reinstatement_schedule
validate_reinstatement_function
```

The module should not contain FFT code.

### `src/aggregate/_aggregate.py` or the appropriate Aggregate concern module

Add convenience method:

```python
Aggregate.reinstatement_analysis(...)
```

This method should build or retrieve the gross/ceded canonical bivariate distribution and delegate to `ReinstatementAnalysis`.

### `src/aggregate/_pnl.py`

Factor out any reusable one-dimensional P&L statistics and presentation helpers.

Do not make `PnL` responsible for the bivariate calculation unless a broader multivariate-P&L redesign is undertaken.

### Tests

Add:

```text
tests/test_reinstatement_terms.py
tests/test_reinstatement_pushforward.py
tests/test_reinstatement_analysis.py
tests/test_reinstatement_exhibit.py
```

## 21. Minimum test cases

### 21.1 Deterministic one-event cases

Test the examples above using fixed frequency one and discrete severity atoms.

### 21.2 Multiple small losses

Use 100 occurrences of 101 followed by one occurrence of 200 for a 100 xs 100 layer with one reinstatement. Confirm:

- unlimited recovery $R=200$;
    
- actual recovery $A=200$;
    
- reinstatement premium equals the maximum;
    
- ordering does not affect annual results.
    

### 21.3 Free and differently priced reinstatements

Test schedule:  
$$  
(0,0.5,1,1,1).  
$$  
Check values immediately before, at, and after every breakpoint.

### 21.4 Accounting identities

For every occupied joint grid cell, assert premium, loss, and underwriting identities before rebucketing.

### 21.5 Distribution identities

Check means and covariance decompositions after pushforward.

### 21.6 Order invariance

For a small enumerated frequency/severity model, explicitly enumerate event sequences and verify that annual results depend only on $R$ under standard assumptions.

### 21.7 Grid invariance

Run at multiple bucket sizes and verify:

- probability mass;
    
- direct transformed means;
    
- convergence of SDs and quantiles;
    
- bounded rebucketing error.
    

### 21.8 Comparison with simulation

For a nontrivial Poisson/lognormal catastrophe model, compare FFT2 results with a large Monte Carlo simulation for:

- mean reinstatement premium;
    
- mean actual ceded loss;
    
- mean net underwriting result;
    
- SD of net underwriting result;
    
- selected percentiles.
    

## 22. Recommended initial public API

```python
terms = ReinstatementTerms(
    limit=100.0,
    rates=(0.0, 0.5, 1.0, 1.0, 1.0),
    rate_on_line=0.10,
    deposit_premium=10.0,
)

analysis = agg.reinstatement_analysis(
    gross_premium=150.0,
    terms=terms,
    percentiles=(0.90, 0.95, 0.99, 0.995, 0.996, 0.999),
)
```

Access:

```python
analysis.exhibit
analysis.statistics
analysis.audit
analysis.distributions["net_uw"]
analysis.distributions["net_loss"]
analysis.source
analysis.terms
```

Alternative construction for custom terms:

```python
terms = ReinstatementTerms.from_callable(
    recovery_limit=600.0,
    deposit_premium=10.0,
    premium_function=my_increasing_function,
)
```

## 23. Design conclusions

The correct probabilistic object is the joint annual distribution of gross loss and unlimited occurrence ceded loss:  
$$  
(L,R).  
$$  
FFT2 should be used once to construct this joint distribution.

Actual ceded recovery, reinstatement premium, gross/ceded/net premium, loss, and underwriting result are deterministic functions of $(L,R)$. Their distributions should be produced by $O(n_Ln_R)$ pushforward operations, not by further convolution.

The reinstatement premium should be modelled as a general increasing function of unlimited occurrence recovery. Piecewise-linear treaty schedules should be first-class helpers, while arbitrary vectorized callables should remain possible.

A dedicated `ReinstatementAnalysis` should own the contract transformation, distributions, accounting identities, actuarial exhibit, and numerical audit. `BivariateDistribution` should supply the general numerical pushforward mechanism. Existing `PnL` conventions should be reused for signed underwriting-result reporting, especially the use of SD rather than CV for underwriting margins.

The design gives `aggregate` an exact and unusually complete solution to a longstanding actuarial problem: the joint effect of occurrence protection, finite reinstatement capacity, stochastic reinstatement premium, and the resulting dependence between net premium and net loss.

## 24. Open design questions / decisions

1. Should the Ceded underwriting-result column use the cedant-cost convention  
    $D+\mathrm{RP}-A$, as specified here, or the cedant-benefit convention  
    $A-D-\mathrm{RP}$? The former produces the clean identity  
    $U_{\mathrm{gross}}=U_{\mathrm{ceded}}+U_{\mathrm{net}}$.
    Decision: use the payoff sign convention through out. Ceded premium is negative (it is an expense) and ceded recoveries are positive. Exhibits are shown with signs so exhibits just "add" (contrary to common accounting views, but much clearer and unambiguous IMO.)
2. Should the percentile rows display underwriting loss $-U$, so larger values are adverse, or display lower-tail underwriting-result quantiles directly? Underwriting loss gives the clearest 90%–99.9% exhibit.
    Decision: percentile rows show "bad" outcome = context sensitive. 
3. Should `deposit_premium` default to `rate_on_line * limit`, or always be explicit? Explicit is safer because quoted deposit premium and reinstatement rate on line need not be identical.
    Decision: `deposit_premium` entered in currency units, rate on line is computed. 
4. Should a supplied custom premium function be required to be vectorized, or should the implementation accept scalar callables and apply `np.vectorize` as a convenience? Requiring vectorisation is cleaner and faster.
	Decision: require vectorized, ValueError if not.     
5. Should the generic `BivariateDistribution.pushforward` be public in v1.0, or initially remain an internal primitive exposed only through `ReinstatementAnalysis`?
    Decision: public - this is a wow factor feature. 

 