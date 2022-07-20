## A Two Step Underwriting Risk Model


This example considers how economic value is earned over time as reserve uncertainty resolves in a two step steady-state book.

The current accident year is modeled as new business plus one run off year. All loss payments occur at $t=2$.

At inception $t=0$ there is expected claim count uncertainty, through a mixing variable $G$, and full severity uncertainty. $G$ has mean $1$. Conditional on $G$ the claim count has a Poisson distribution with mean $Gn$. The mixing variable controls macro-unknowns such as the weather, inflation and level of economic activity that affect all insureds. It drives correlation.

The value of $G=g$ becomes known through the first period and at $t=1$ claim count uncertainty reduces to Poisson but with mean given by $gn$ rather than the prior expectation $n$. Reserves are set from this distribution at $t=1$.

Reserves are paid at $t=2$ as a realization of the reserve distribution booked at $t=1$.

The current calendar year $X$ is modeled as $X_0$ new business with $G$-mixed uncertainty and  reserves $X_{-1}$ from the prior year with a known $G=g$ state variable. For convenience assume the book is steady-state with no volume changes in prior periods. The model is summarized in \cref{tab:reserve-two-resolution}.

|    Time    | Description                     | Loss Model                                                                  |
|:----------:|:--------------------------------|:----------------------------------------------------------------------------|
|     0      | Period $y$ new business written | $X_0=C_1 + \cdots + C_N$,  independent severity $C_i$ and $N$ a $G$-mixed Poisson      |
| | |
| $0$ to $1$ | Macro variables revealed        | $G=g$ revealed, claim count $N$ distributed Poisson$(ng)$            |
| | |
|    $1$     |                                 | $X_{-1}=C_1 + \cdots + C_N$, $N$ Poisson with known mean $gn$               |
| | |
| $1$ to $2$ | Final outcome revealed          | All claims settled and paid                                                 |
| | |
|    $2$     |                                 | $X_{-1}=x$                                                                  |

Table: Stochastic model for resolution of claim uncertainty. \label{tab:reserve-two-resolution}


NEED BETTER NOTATION.

A claim $C_i$ is modeled as a layer from a distribution with mean $m_i$ and unconditional uncertainty $v_i$.

\Cref{tab:basic-loss-statistics-runoff} shows resulting volatilities in a reasonably parameterized example.


<!-- insert base statistics etc.  -->

{% include "Reserve_specification.md" %}

### Earning Margins

<!-- see material from reserve_runoff -->

\footnotesize

{{ margin_earned }}

Table: Margin is earned in each period is determined by the natural allocation. Beginning margin shows cumulative margin available at each point in time.

\normalsize

\Cref{tab:margin-earned-runoff} shows how economic income is earned.  The difference in margin with the following period is earned during the period. The bulk of the margin is earned in the first period, as expected. It has the greatest resolution of uncertainty. This is particularly true for short-tailed, catastrophe exposed lines. Removing the claim count uncertainty has a relatively small impact. Setting up accurate case reserves will be important. Obviously the more accurate case reserves the less uncertainty and margin is deferred into period 4 when claims are paid. At $t=4$ all claims are paid and closed. In steady state the total margin earned equals the margin embedded in the current accident year.

In this case reserve duration is exactly four years. There are no payments made before then. The risk tenor, defined as the ratio of cumulative beginning period margins to one year margin is {{risk_tenor}}.

In run-off the margin required for the reserves at best estimate is {{best_estimate_margin}} vs. {{going_concern_margin}} as a going-concern.

{{ year_end_option_analysis }}

