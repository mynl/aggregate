## Economics of Reserve Run-Off

This example considers how economic value is earned over time as reserve uncertainty resolves in a steady state book.

The current accident year is modeled as new business plus three run off years. All loss payments occur at $t=4$.

At inception $t=0$ there is expected claim count uncertainty, through a mixing variable $G$, and full severity uncertainty. $G$ has mean $1$. Conditional on $G$ the claim count has a Poisson distribution with mean $Gn$. The mixing variable controls macro-unknowns such as the weather, inflation and level of economic activity that affect all insureds. It drives correlation.

The value of $G=g$ becomes known through the first period and at $t=1$ claim count uncertainty reduces to Poisson but with mean given by $gn$ rather than the prior expectation $n$.

Claim reporting occurs during the second period and by $t=2$ the actual number of claims is known as a Poisson draw with mean $gn$. Only amount uncertainty remains.

Case reserves are set during the third period. The ultimate value of each claim is modeled as a distribution with a uncertainty, measured by the coefficient of variation, $v_{res}$ around its case reserve. By $t=3$ all case reserves are known and the remaining variability is the reserving error.

During the fourth period all claims are settled and paid and the final result becomes known.

The current calendar year $X$ is modeled as $X_0$ new business with full uncertainty, $X_{-1}$ reserves from the prior year with known state variable, $X_{-2}$ from the second prior year with known claim count etc., and $X=X_0+X_{-1}+X_{-2}+X_{-3}$. For convenience assume the book is steady-state with no volume changes in prior periods. The model is summarized in \cref{tab:reserve-resolution}.

|    Time    | Description                     | Loss Model                                                                  |
|:----------:|:--------------------------------|:----------------------------------------------------------------------------|
|     0      | Period $y$ new business written | $X_0=C_1 + \cdots + C_N$,  independent severity $C_i$ and $N$ a $G$-mixed Poisson      |
| | |
| $0$ to $1$ | Macro variables revealed        | $G=g$ revealed, claim count $N$ distributed Poisson$(ng)$            |
| | |
|    $1$     |                                 | $X_{-1}=C_1 + \cdots + C_N$, $N$ Poisson                                    |
| | |
| $1$ to $2$ | Claim count revealed            | $N=n$ revealed and all claims reported, no information about severity    |
| | |
|    $2$     |                                 | $X_{-2}=C_1 + \cdots + C_n$                                                 |
| | |
| $2$ to $3$ | Best-estimate case reserves set | Unbiased case reserves $C'_i$ set; payments distributed with $v_{res}$ about mean |
| | |
|    $3$     |                                 | $X_{-3}=C'_1 + \cdots + C'_n$                                               |
| | |
| $3$ to $4$ | Final outcome revealed          | All claims settled and paid                                                 |
| | |
|    $4$     |                                 | $X_{-4}=x$                                                                  |

Table: Stochastic model for resolution of claim uncertainty. \label{tab:reserve-resolution}


A claim $C_i$ is modeled as a layer from a distribution with mean $m_i$ and unconditional uncertainty $v_i$. The reserve outcome distribution $C'_i$ has conditional mean $C_i=c_i$ and uncertainty $v_{res}$.
The reserve uncertainty, which measures variability of the final settlement around case estimates, is much smaller than the unconditional severity uncertainty.

In aggregate, setting reserves collapses risk from $C_1+\cdots + C_n$ to $C'_1+\cdots + C'_n$. The unconditional variance is $\mathsf{Var}(\sum C_i)=\sum (m_i v_i)^2 = n(mv)^2$ if claim distributions are identical. The conditional variance given $C_i=c_i$ becomes $\mathsf{Var}(\sum C'_i)=v_{res}^2 \sum c_i^2$. Approximate $\sum c_i^2$ with $\sum\mathsf{E}[C_i^2]=\sum m_i(1+v_i^2)$.
In the identically distributed case the uncertainty of total liability decreases from
$v/\sqrt{n}$  to roughly $v_{res}\sqrt{(1+v^2)/n}\approx  v_{res}v/\sqrt{n}$, i.e. by a factor of $v_{res}$. The general case can be reduced to the identically distributed case by considering severity to be a mixture.

\Cref{tab:basic-loss-statistics-runoff} shows resulting volatilities in a reasonably parameterized example.



<!-- insert base statistics etc.  -->

{% include "Reserve_specification.md" %}

### Earning Margins

The total margin shown in \cref{tab:premium-capital-detail-runoff} is the one year cost of bearing total risk $X$. It covers the emergence of information about the current year, one-year re-estimation risk on prior years and payment risk. For each prior year the margin in the table is the natural allocation of the total. If the prior years were prospective lines of business then the loss plus margin would be the market premium, rather than the market value of liability shown here. The latter is the statement value under market accounting, whereas expected incurred losses is the objective statement value.

The last two columns of \cref{tab:premium-capital-detail-runoff} show totals for the beginning period reserves ($\eta\nu$X0 = all but X0) and end of period reserves that have to be established before the next accounting period. The latter determine the capital requirement for InsCo in the next period.

Premium for new business covers all the costs of risk transfer, to ultimate. Therefore it is equal to expected losses plus the total margin, across all prior years. The income is economically allocated to prior years as shown in the allocated margin row.

Under Objective accounting, where liabilities are held without a risk margin, premium funds the allocated margins for each line since there is no carried accrual. Income for the current period is fully recognized at the end of the period but can be thought of as allocated to prior years.

Under Market accounting only the allocated margin is earned by each prior year, with the remainder funding the risk margin carried forward in reserves. Accounting changes how income is recognized but not the total.

Policyholder asset for older years do depend on accounting because they are increased by risk margins. As a result, shareholder funds are decreased. Thus to premium to achieve a target return will vary. At the same time, adding risk loads increases shareholder risk because there is the possibility that assets will be inadequate to book the risk load. In banking this risk is very material because market prices can vary significantly compared to fundamental value: spread expansion on CDS was a major difficulty for several institutions during the 2008 Financial Crisis, for example. Reserve risk margins will generally be more stable, though could exhibit considerable volatility for certain lines like mass torts. Adding risk loads increases the probability of insolvency from $S(a)$ to $g(S(a))$.

The magnitude of the difference between policyholder assets with and without a risk load is determined by the allocated margin earnings, which provides a distinctive **profit signature** for each portfolio. The reduction in shareholder funds is given by the cumulative sum of the cumulative sum of individual risk loads. The more deferred the resolution of reserve uncertainty the larger the difference. It can be regarded as a risk tenor of the reserves and compared to the payment tenor.

Premium depends how the reserve risk is managed. In steady-state the cost of bearing the run-off risk will be as shown in \cref{tab:premium-capital-detail-runoff}, assuming losses emerge at plan or higher or lower in a bad or good year. In run-off the cost will be higher because the reserves will not be diversified against new prospective business. In a competitive market it is plausible that premiums will be pushed down to the going-concern rate, possibly with insurers holding higher than regulator assets. The latter is observed in the market but we don't know if the former occurs or not. There is a loss in value to reserve holders in the going-concern model, so they would not accept paying for run-off premium but getting less coverage. In general there is no canonical premium. The situation is analogous to rolling-over bank financing. At each point there is a risk financing will not be available, but it is generally ignored...until it becomes a huge issue!


\footnotesize

{{ margin_earned }}

Table: Margin is earned in each period is determined by the natural allocation. Beginning margin shows cumulative margin available at each point in time.

\normalsize

\Cref{tab:margin-earned-runoff} shows how economic income is earned.  The difference in margin with the following period is earned during the period. The bulk of the margin is earned in the first period, as expected. It has the greatest resolution of uncertainty. This is particularly true for short-tailed, catastrophe exposed lines. Removing the claim count uncertainty has a relatively small impact. Setting up accurate case reserves will be important. Obviously the more accurate case reserves the less uncertainty and margin is deferred into period 4 when claims are paid. At $t=4$ all claims are paid and closed. In steady state the total margin earned equals the margin embedded in the current accident year.

In this case reserve duration is exactly four years. There are no payments made before then. The risk tenor, defined as the ratio of cumulative beginning period margins to one year margin is {{risk_tenor}}.

In run-off the margin required for the reserves at best estimate is {{best_estimate_margin}} vs. {{going_concern_margin}} as a going-concern.

{{ year_end_option_analysis }}


