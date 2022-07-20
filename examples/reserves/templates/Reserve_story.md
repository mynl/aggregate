<!-- main page for allocation functions -->

## Example Story For Reserves

### Set Up and Standing Assumptions

Up to this point InsCo has been a one period entity, created *de novo* at the beginning of the period and extinguished at the end. Now extend this concept by allowing InsCo to carry forward reserve liabilities and supporting assets from business written in prior periods. At the beginning of each new period InsCo has the option of continuing as a going-concern and writing new business, or becoming a run-off entity. This section analyzes InsCo's decision.

Reserves are an evaluation of claim liabilities at a point in time before final settlement. Simultaneously with reserve evaluations InsCo must pass accounting solvency and regulatory capital tests. Introducing reserves is inextricably linked with a notion of *time*: when and how frequently are reserves, solvency and capital evaluated? Until now there has been no explicit notion of time in the analysis.
The distortion function $g$ introduced in Ch. 4 prices a bond with a probability $s$ of default over one time period. In a multi-period model $g$ will need to be applied multiple times in order to determine the risk load to ultimate.

\footnotesize

The ability to avoid specifying the length of a period  has been enabled by the assumption of a zero risk free rate of interest. Since interest represents a rate of payment per unit of time it also requires an explicit temporal framework. The analysis will continue to assume a zero risk free rate.

\normalsize

In a one period model, where losses are paid at the end of the period, it is clear that a risk measure must apply to the paid loss random variable.
In a multi-period model there are different ways to define which variable should be used to evaluate risk. In Solvency II the capital standard is applied to the end of period market value of liabilities. Since there isn't a market value for most insurance liabilities, this quantity is estimated as reserves plus a risk margin that is calculated as the cost of capital times regulatory capital over the run-off period. The UK (England REF) requires risk be evaluated using ultimate variability at each point in time. Since the sum of one year views must reconcile to ultimate risk in a steady state book these two must be approximately equal, albeit with the option to push recognized volatility into the future in Solvency II.

The examples in this section assume that the regulator uses the ultimate distribution of losses at each point in time when evaluating risk. This view is consistent with the fact that InsCo has an option not to re-capitalize to the regulator standard at each evaluation point. It also reflects the inherent difficulties in modeling a future distribution of market values. The extent it possibly over-states risk can be offset by selecting a lower probability threshold in the regulator risk measure.

Specifically, the regulatory and valuation assumptions are as follows.

1. All risky liabilities have a fair market price given by the distortion $g$ and associated risk measure $\rho$, per Ch2. 4 and 9. The market value of a risk $X$, whether paid or booked as a liability at the end of the period, is $\rho(X)=\int g(S(t))\,dt$. If $X$ is a sum of parts then there is a natural allocation to each part, REF.
As a result of this assumption InsCo faces costly capital.
Previously the analysis considered risky cash flow rather than risky liabilities.

2. Regulation requires InsCo holds $\VaR$ capital based on the *objective* distribution of *ultimate* loss. The calculation uses expected value, reflecting the full range of outcomes, and with a reduction for claims paying ability. It is MAYBE consistent with the UK but not Solvency II. If InsCo fails the capital standard the regulator can place it into rehabilitation, conservation or liquidation, i.e. existing management is usurped. The example takes $p={{p}}$.

4. There are three accounting views: statutory, objective and economic that differ in how they value liabilities.
    a. **Statutory** accounting takes an objective point estimate and does not reduce payments to reflect available assets. It proxies US statutory and GAAP accounting.
    b. **Objective** accounting follows the capital standard: it uses expected value against objective probabilities, reflecting the full range of outcomes, and with a reduction for claims paying ability.
    c. Market accounting takes market value of liabilities. It differs from objective accounting by using risk adjusted probabilities, which produce an explicit risk load. Market accounting proxies IFRS.

5. There is no difference in the carrying value of an asset across the three accounting views. All assets are invested in a risk free instrument, the numeraire, that pays a zero interest rate. The market value of an asset always remains equal to its original statement value.

6. Liability holders have equal priority in default.

7. All policies have the same effective date and a one period policy term. There is no unearned premium reserve at the end of a period. The only liability carried forward is an unpaid loss reserve.

8. Capital markets are efficient and InsCo investors do not have a liquidity constraint.

In addition to solvency regulation InsCo also faces standard financial accounting thresholds. If its assets fall below the financial reporting expected value of liabilities it becomes insolvent and goes into liquidation. Since this is a weaker test than the capital standard it never becomes part of the analysis.

As a matter of nomenclature, *new business* refers to policies that are written and become effective during a period. It encompasses both renewals of existing policies and new-new policies. The model is not addressing questions of new-new business performing differently than renewal business.

In the example the reserves and new business are independent for computational convenience. However, the calculus does not require independence.

If expected payments from reserves plus new business equals the amount of new business written then the portfolio is in steady state. However, this is not necessary. Paid losses are only relevant to determine the probability of different decisions, and are not included in the model until the last section.

$X_{0}$ represents business written in the first period from $\tau=0$ to $\tau=1$. $X_{-1}$ represents business written in the prior period from $\tau=-1$ to $\tau=0$. As usual $X=X_{-1}+X_{0}$ is total liabilities and $a$ represents assets.

The **market value** of InsCo defined as the market value of assets less market value of liabilities. Since assets are held at market value in a risk free account it is simply $\mathit{MV}(X, a) = a - \rho(X\wedge a)$.
The economic gain or loss for investors from a change to InsCo assets or liabilities is defined as the change in market value minus the change in assets invested.
If there is a change $X$ to $X'$ and $a$ to $a'$ then it is given by
\begin{align}
EGL
&= (a'-\rho(X'\wedge a'))- (a-\rho(X\wedge a)) -(a'-a) \nonumber \\
&= \rho(X\wedge a) - \rho(X'\wedge a'). \label{eq:egl}
\end{align}
Changes in economics are driven solely by their impact on insured claim liabilities. An economic gain or loss for investors is exactly offset by an economic loss or gain for insureds.

The simple form of \cref{eq:egl} relies on the fact assets are invested in a risk free instrument so $a$ does not change from its initial value. It also relies on there being no frictional costs to investors of holding assets within InsCo. With asset risk or frictional costs there could a loss of value immediately on investment. Practitioners speak of insurers as capital roach motels: capital can check-in but it can never check out. We are assuming an efficient capital market without such frictions.

The economic gain or loss from a small change $da$ in assets with no change in liabilities, i.e. an increase in capital, is
\begin{equation}
\mathit{EGL} = \int_0^a g(S(t))dt - \int_0^{a'} g(S(t))dt = -g(S(a))da < 0. \label{eq:never-recap}
\end{equation}
by \cite{eq:egl}.
Investors will *never* voluntarily add assets to InsCo with no change in $X$ because it is a gift to insureds. On the other hand they will *always* benefit from decreasing assets, but insureds will never voluntarily submit to such a reduction.
Here there is a critical distinction between new business and reserves: new business has a choice of insurer in a competitive market whereas claimant liability holders, *reserve holders*, are locked into InsCo.
In a transparent market it is not possible to extract excess profits, beyond the cost of capital, from new business because insureds would not purchase at a price above the fair market value. However, it is possible to do so from reserve holders because they are unable to unilaterally move their claim to another insurer. The insurer has latitude to change capitalization for claimants provided it complies with regulation. In particular it has the right to dividend excess capital, above the regulator standard, to owners resulting in a transfer of market value from reserve holders to shareholder owners. Extracting this value creates an incentive for InsCo to continue as a going-concern.

The rest of this section lays out a detailed example.

<!-- insert base statistics etc.  -->
{% include "Reserve_specification.md" %}

Expected one period payments are {{ est_paid }} and expected end of period assets are {{ est_ye_assets }}.


### Earning Margins

<!-- see material from reserve_runoff -->

\footnotesize

{{ margin_earned }}

Table: Margin is earned in each period is determined by the natural allocation. Beginning margin shows cumulative margin available at each point in time.

\normalsize

\Cref{tab:margin-earned-runoff} shows how economic income is earned.  The difference in margin with the following period is earned during the period. The bulk of the margin is earned in the first period, as expected. It has the greatest resolution of uncertainty. This is particularly true for short-tailed, catastrophe exposed lines. Removing the claim count uncertainty has a relatively small impact. Setting up accurate case reserves will be important. Obviously the more accurate case reserves the less uncertainty and margin is deferred into period 4 when claims are paid. At $t=4$ all claims are paid and closed. In steady state the total margin earned equals the margin embedded in the current accident year.

In this case reserve duration is exactly four years. There are no payments made before then. The risk tenor, defined as the ratio of cumulative beginning period margins to one year margin is {{risk_tenor}}.

In run-off the margin required for the reserves at best estimate is {{best_estimate_margin}} vs. {{going_concern_margin}} as a going-concern.

### Detailed Results

\footnotesize

{{ year_end_option_analysis }}

Table: Detailed results \label{tab:detailed-results}

Column or Symbol                        | Definition                                         |
:---------------------------------------|:---------------------------------------------------|
$a$ | Assets at the end of the prior period, brought forwards
$S_X(a)$                      | $Pr(X > a)$                                        |
$S_{X_{-1}}(a)$               | $Pr(X_{-1} > a)$                                      |
$E(X_{-1}\wedge a)$           | Expected value payments to $X_{-1}$ limited at $a$, stand-alone basis |
$\rho(X_{-1}\wedge a)$        | Market value payments to $X_{-1}$ limited at $a$, stand-alone basis |
$E(X_{-1}(a))=\bar S_0(a)$    | Expected value of recoveries by $X_{-1}$ within $X$ with assets $a$ |
$P(X_{-1}(a))=\bar P^a_0$     | Market value of recoveries by $X_{-1}$ within $X$ with assets $a$ |
$E(X_{0}(a))$                 | Expected value of recoveries by $X_{0}$ within $X$ with assets $a$ |
$P(X_{0}(a))$                 | Market value of recoveries by $X_{0}$ within $X$ with assets $a$ |
$E(X\wedge a)=\bar S(a)$      | Expected value payments to $X$,limited at $a$      |
$\rho(X\wedge a)=\bar P(a)$   | Market value payments to $X$,   limited at $a$     |
$ro$                          | Stand-alone view, liabilities $X_{-1}$ |
$gc$                          | Going-concern view, liabilities $X=X_{-1}+X_{0}$ include new business |
$\Delta  Q_{ro}(a)$           | Change in equity paid in, run-off              |
$\Delta  \mathit{MV}_{ro}(a)$ | Change in market value for re-capping, run-off |
$\mathit{EGL}_{ro}(a)$        | Economic gain or loss from re-capping, run-off  |
$\Delta  Q_{gc}(a)$           | Change in equity paid in as a going-concern        |
$\Delta  \mathit{MV}_{gc}(a)$ | Change in market value for re-capping as a going-concern |
$\mathit{EGL}_{gc}(a)$        | Economic gain or loss from re-capping as a going-concern |

Table: Detailed result table key \label{detailed-results-key}

\normalsize

The column header shows assets $a$ carried forward by InsCo to back the reserve liability at the end of the prior period. The table shows a range of $a$ around the required run-off asset level  $a_{ro}:=\mathit{VaR}_{p}(X_{-1})={{a_x0}}$. The last column shows the going-concern capital $a_{gc}:=\mathit{VaR}_{p}(X)={{a_x}}$ including $X_{0}$.
$\rho(X_{-1}\wedge a)$ is run-off premium.
$P(X_{-1}(a))$ is the natural allocation going-concern market value premium for $X_{-1}$ as part of $X$.
$\rho(X\wedge a)$ is total premium as a going-concern.
All premiums are computed with the same distortion but generally with different $Q$-measures between the run-off and run-off scenarios.
The market value of insurance payments reflects actual assets, whereas financial accounting reserve expectations may not. <!-- Thus the market value of InsCo is greater than its accounting value. It has a price to book ratio greater than 1. -->

InsCo's market value can be computed on a run-off basis, with liabilities $X_{-1}$, or on a going-concern basis, with liabilities $X$. $\Delta\mathit{MV}$ represents the change in market value from a strategic decision.

$\Delta Q(a)$ represents capital cash calls or dividends paid to owners as the result of strategic decisions made about renewal and run-off. A positive value represents a capital injection, required to maintain the minimum capital level, and a negative value corresponds to a dividend payable to owners to lower excess capital to the regulated level. InsCo faces costly capital and will never hold more than the regulator required level.

Going into the new period InsCo has two options.

1. Enter run-off, adding no new premium.
    a. If $a<a_{ro}$, InsCo would be subject to some level of regulatory supervision unless it re-capitalized to the regulatory level. If $a<E[X_{-1}]$ InsCo is technically insolvent and could be forced into rehabilitation, conservation or liquidation by the regulator without management action. This case is not considered separately.
    b. If $a>a_{ro}$ then InsCo can dividend excess capital back to owners and reconsider its options.
2. Continue as a going-concern, writing new business, $X_{0}$. It is subject to minimum capital requirements and must re-capitalize to $a_{gc}$.

In order to decide between these options InsCo computes the economic gain or loss from each and selects the one with the greatest *EGL* provided it is positive. If *EGL* is negative for all options the company enters run-off and accepts regulatory strictures associated with failing the capital standard.

In reality InsCo would have a range of new business it could write. It would select the one with the greatest positive *EGL* by repeating the following analysis for each.

This process mirrors a Lloyd's reinsurance to close transaction, where one year of account is reinsured into a new syndicate.

#### Case 1: Run-Off {-}

In run-off, InsCo requires assets $a_{ro}$ and owners face a call (dividend if $<0$) of $\Delta Q_{ro}(a) = a-a_{ro}$ to re-capitalize (return excess capital). It is relatively unlikely that assets are inadequate and investors face a call, see the left two columns of \cref{tab:detailed-results}. The third column corresponds to assets exactly equal to $a_{ro}$ and so there is  no capital adjustment. In all columns to the right they receive a dividend.

When it takes no action, InsCo has a run-off market value of $\mathit{MV}_{ro}(a) = a-\rho(X_{-1}\wedge a)$. If ending assets are low relative to unlimited liabilities the market value is very small, but it is always positive unless $\Pr(X_{-1}<a_{ro})=0$, i.e. if the owner's residual value is certain to be zero. If InsCo re-capitalizes the market value becomes $\mathit{MV}_{ro}(a_{ro})$.
Owners will decide to re-capitalize if there is a positive economic gain or loss from their investment $\mathit{EGL}_{ro}(a)=P(X_{-1}\wedge a) - \rho(X_{-1}\wedge a_{ro}) \ge 0$.

As we have already seen in \cref{eq:never-recap} re-capitalizing with no change to liabilities will alway destroy market value. Adding capital is a donation to claimants, increasing the value of their claim and destroying investor value. For asset values close to $a_{ro}$ the loss is quite small, but it increases rapidly for lower asset levels. An investment could still be undertaken to avoid costs of financial distress generated by regulator supervision, where investor appointed managers are replaced with regulator appointees who may have non-commercial incentives but these considerations are outside the model.

In case 1.b, where ending assets $a>a_{ro}$, InsCo can dividend excess capital back to owners and become a run-off entity. Again, in the absence of other reasons, it will always choose do so. Extenuating reasons could include reputation and marketing benefits or if InsCo faces inefficient capital markets for raising new capital under financial distress. Higher premium for a more well capitalized company is **not** a reason because premiums already explicitly incorporate capital adequacy.

InsCo can also consider going-concern options.

#### Case 2: Going-Concern {-}

As a going-concern InsCo requires assets $a_{gc}$, which will typically be larger than $a_{ro}$ by at least the volume of new business written.
The new assets will come from two sources: premium from new business and a capital contribution from current investors. Existing investors pay the fair market premium for $X_{-1}$, combine it with $X_{0}$, add assets to the regulated level, and collect new premium. They create an entity with a higher market value which they could sell to recoup their investment if desired.
They will undertake the investment if the increase in market value it produces is greater than the amount of their investment.

Since the capital market is efficient there is no need to distinguish between new and existing investors. Existing investors engineer InsCo so it can write $X_{-1}+X_{0}$ and then sell it or retain it. Or InsCo owners can bring on new investors by selling a pro rata portion of the going-concern. Or new investors can be brought in *pari passu* with existing owners during the transaction. In an efficient market there are no liquidity constraints and the order of these transactions is irrelevant. Owners will be able to obtain the necessary financing to re-capitalize and sell if doing so creates market value.
Since all valuation is at the fair market price there is no dilution between new and existing investors. The model does not allow imperfections such as a control premium or asymmetric information between management, existing and new investors, for example.

Adding $X_{0}$ results in total liabilities $X=X_{-1}+X_{0}$ and regulatory asset level $a_{gc}$. The natural allocation gives the fair market value of insurance cash flows, i.e. premium, $P(X_i(a_{gc}))$ for each line. The new entity has market value $\mathit{MV}_{gc}(a_{gc})=a_{gc}-\rho(X\wedge a_{gc})={{mv_gc}}$, shown in the right-hand most column. Note this single market value is computed for $X$, as opposed to run-off $X_{-1}$.

To be fair to new insureds InsCo must bring forward assets equal to the market value of assumed reserve liabilities, $P(X_{-1}(a_{gc}))$, and increase capital. $P(X_{-1}(a_{gc}))$ is same market value that would apply if new business with a distribution $X_{-1}$ were written alongside $X_{0}$.
InsCo owners invest $\Delta Q_{gc}(a) = a_{gc}-P(X_{0}(a_{gc}))-a$ to capitalize InsCo to write new business, where  $P(X_{0}(a_{gc}))$ is the market value of new business.
The increase is funded out of ending assets $a$ and new investment by existing InsCo owners. If there are excess ending assets they are returned to original investors as a buy-out dividend payment.
The funding constraint implies $a_{gc}=P(X_{-1}(a_{gc}))+P(X_{0}(a_{gc}))+\mathit{MV}_{gc}(a_{gc})$.

By \cref{eq:egl} the economic gain or loss from these transactions is
\begin{align*}
\mathit{EGL}_{gc}(a)
% &= \Delta_{gc}\mathit{MV}_{gc}(a) - \Delta Q_{gc}(a) \\
%&= (a_{gc} - \rho(X\wedge a_{gc})) - (a - \rho(X_{-1}\wedge a)) - (a_{gc} - P(X_{0}(a_{gc}))-a)  \\
%&= - \rho(X\wedge a_{gc}) + \rho(X_{-1}\wedge a) + P(X_{0}(a_{gc}))   \\
&=  \rho(X_{-1}\wedge a) - P(X_{-1}(a_{gc}))   \\
\end{align*}
the difference between the market value of reserves supported by ending assets and the market value of reserves supported by regulatory capital within a portfolio with new business.
The gain from the investment exactly equals the loss in  market value to claimants caused by a relative reduction in their capital adequacy. In general the term on the right is positive.
InsCo investors make the investment to create the going-concern if $\mathit{EGL}_{gc}(a)>\max(0, \mathit{EGL}_{ro}(a))$: i.e. if it is positive and is a better option than going into run-off. In this case the break-even asset level is $a={{break_even}}$ and going-concern dominates run-off.

Comparing the going-concern and run-off *EGL*s shows they both have the same form: market value of reserve liabilities with current assets minus market value with regulated assets.
\begin{gather*}
\mathit{EGL}_{ro}(a) = \rho(X_{-1}\wedge a) - \rho(X_{-1}\wedge a_{ro}) \\
\mathit{EGL}_{gc}(a) = \rho(X_{-1}\wedge a) - P(X_{-1}(a_{sa})) \\
a_{ro} < a < a_{gc}  \\
{{a_x0}} < {{est_ye_assets}} < {{a_x}} \\
P(X_{-1}(a_{gc})) < \rho(X_{-1}\wedge a_{ro}) < \rho(X_{-1}\wedge a) \\
{{mvp_gc}} < {{mvp_ro}} < {{mvp_a}}
\end{gather*}
The last four lines show the normal ordering of these variables in a steady state when paid losses are close to the expected value and reserves are less volatile than new business The relevant values are pulled from  the Results table.

In run-off, the re-capitalizing *EGL* is positive only if it results in a dividend because assets are greater than regulated assets. This is likely to occur be because starting assets are calibrated to the more volatile going-concern book. By continuing as a going-concern InsCo captures the efficiencies of pooling reserves with new business, and so $P(X_{-1}(a_{gc}))={{mvp_gc}}$ is generally less than $\rho(X_{-1}\wedge a_{ro})={{mvp_ro}}$.  The lower value in the going-concern is driven by the claims from higher volatility new business eroding the claimants' position. The dynamics are illustrated in \cref{fig:reserve-illustration-lev}. Thus under normal circumstances the company will continue as a going-concern. Ending assets below {{break_even}} will put the company into run-off but this implies very high paid loss for the year.

![Market value of reserve holder claims in a run-off and going-concern entity. \label{fig:reserve-illustration-lev} ](img/ch12-reserve-illustration.png)

In both run-off and going-concern InsCo reduces the value of the reserve holder claims by reducing assets to the regulatory amount: a regulator-sanctioned haircut for claimants. Since the reserve holders are tied into InsCo they have no recourse. The analogous action for newly written business is impossible: it corresponds to switch-and-bait, selling at a premium corresponding to a higher asset level and delivering a lower one. If InsCo attempted that it would fail to write the business. <!-- Duplication but OK? Important point -->

### Timing Issues and Comparison with Solvency II

There are some important timing differences in timing and the modeled quantities in the model presented here and Solvency II.

Solvency II capital is calculated as the value at risk of one year market values. It requires estimating the distribution of reserves book one year out, i.e. modeling a variable with no objectively determined value, an obvious problem.

The approach here uses an estimate of the distribution of ultimate payments from current reserves, which is an objective quantity. It implicitly relies on the distribution of paid losses during the year, another objective quantity that can be modeled. Paid losses relate to a specific time frame.

