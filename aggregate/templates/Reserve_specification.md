### Specification of Risks for {{scenario_name}} Example

#### Portfolio Program {-}

The portfolio is generated using the following `Aggregate` specification.

\footnotesize

```
{{program}}

```

\normalsize

Line $Xm1=X_{-1}$ represents reserves and $X0=X_{0}$ new business.

#### Basic Loss Statistics

{{ basic_loss_statistics }}

Table: Loss statistics for new business and prior year reserves. \label{tab:basic-loss-statistics-{{scenario_name}}}

Statistics are shown on an unlimited basis, i.e. assuming unlimited InsCo assets.

New business is more volatile than reserves because it allows for uncertainty in claim occurrence as well as ultimate payment amount. In practice a prospective book may have an estimated CV between 20 and 40 percent. Reserves often have estimated CVs below 10 percent.

![Liability densities and log densities (right).](img/{{scenario_name}}-distributions.png)

### Pricing Information

#### Distortion Calibration {-}

\footnotesize

{{ distortion_calibration }}

\normalsize

In the table  Total assets are $a={{a_x}}$. $\iota$ is the input average investor cost of capital across the entire portfolio and $\nu=1/(1+\iota)$, see Ch. 4. The premium is $\rho(X\wedge a)$ when the insurer has assets $a$. Levg shows the premium to surplus ratio and $\bar Q(a)$ initial capital. Expected losses, EL, are reduced to reflect available assets.

The summary confirms pricing is calibrated to produce a {{ROE}} percent ROE at $p={{p}}$-VaR capital for the total portfolio.

#### Distortion Information {-}
Parameters for distortions producing a {{ROE}} percent ROE at $p={{p}}$-VaR capital.

\footnotesize

{{ distortion_information }}

\normalsize

The example uses a {{dist}} distortion.

#### Premium and Capital Detail {-}

\footnotesize

{{ premium_capital }}

Table: Insurance metrics by line on a run-off (ro) and going-concern (gc) basis. Rows 6, 7, 8 only apply to prospective business and are shown for comparison purposes only. \label{tab:premium-capital-{{scenario_name}}}

\normalsize

\Cref{tab:premium-capital-detail-{{scenario_name}}} shows objective and risk adjusted values for each line at the required {{p}} value at risk asset levels. The amounts are all natural allocation values. All by-line quantities sum to the total shown. Rows 6-8 compute one-period returns for each line.


#### Accounting and Economic Balance Sheets {-}

\footnotesize

{{ accounting_economic_balance_sheets }}

Table: Premium and capital detail. \label{tab:accounting_economic_balance_sheets-{{scenario_name}}}

\normalsize

The objective and market values track from \cref{tab:premium-capital-detail-{{scenario_name}}}.
The Difference column is the difference between Market and Objective, which corresponds to the risk margin.
Equity increases from statutory to objective because of limited liability, but decreases to market because of risk margins. The latter are far more material than the former for a well capitalized company, so there is a more material difference between Market and Objective equity than between Objective and Statutory.

