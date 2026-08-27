# New Examples To Merge In With Library.Agg

## Group 1: The Capstone Example (suitable for demoing ot reinsurance actuaries)  

```decl
sev Capstone.Sev (250 / exp(1/2)) * lognorm 1 
  note{Severity curve for the capstone example, lognormal with mean 250 and sigma=1.}
  tags{topic:capstone, role:advanced};
  
agg Gross 
  [2000 5000 10000 3000 1000] premium as GWP at [.95 .95 .9 .85 .75] lr 
  [500  1000 5000 10000 5000] xs [0 0 0 0 1000]
  sev.Capstone.Sev 
  mixed gamma 0.125
  note{Capstone gross portfolio entered from a limit/attachment profile by premium and loss ratio. DRY: refer to severity by name.}
  tags{topic:capstone, role:advanced};

agg Exposure.Rating
    agg.Gross 
    occurrence ceded to 
        tower[0 250 500 1000 2000 5000 10000]
  note{Exposure rating layering of Capstone example into 250 xs 0, 250 xs 250, 500 xs 500, 1 xs 1, 3 xs 2, and 5 xs 5 using the tower keyword. This is an intermediate step to view the layer exposure rating.}
  tags{topic:capstone, role:advanced};

agg Selected.Losses
  [2000 5000 10000 3000 1000] premium as GWP at [.95 .95 .9 .85 .75] lr 
  [500  1000 5000 10000 5000] xs [0 0 0 0 1000]
  sev sev.Capstone.Sev
      picks [500 1000 2000 5000 10000] [13500 2250 1250 500 50] 
  mixed gamma 0.125
  note{Aggregate built using and adjusted Capstone severity curve to achieve desired blended experience and exposure rating loss picks by layer.}
  tags{topic:capstone, role:advanced};

agg Test.Loss.Picks
  agg.Selected.Losses
  occurrence ceded to 
    tower[0 500 1000 2000 5000 10000]
  note{Throw-away test view to confirm adjusted severity curve achieves desired loss picks by layer.}
  tags{topic:capstone, role:advanced};

agg Capstone.XOL 
  agg.Selected.Losses
  occurrence net of
    50% po 500 xs 500 deposit 3000 as "500x500" and
    90% po 1000 xs 1000 deposit 1750 as "1x1" and
    95% po 3000 xs 2000 rol 25% as "3x2" and
    5000 xs 5000 rol 3% as "5x5" 
  note{Apply XOL program terms to Capstone using adjusted severity curve.}
  tags{topic:capstone, role:advanced};

xpnl Capstone.XOL.PnL
  derive premium 
  less
    agg.Capstone.XOL
  less
    25% premium expenses as "G&A"
  note{PnL (profit and loss) view for Capstone with the XOL program only.}
  tags{topic:capstone, role:advanced};

agg Capstone.FullProgram
 agg.Capstone.XOL
  aggregate net of 
    75% po inf xs 0 rate 100% cede 32.5% as "QS" 
  note{Apply quota share on the net book to the Capstone account model, in addition to the XOL program.}
  tags{topic:capstone, role:advanced};

xpnl Capstone.PnL
  derive premium 
  less
    agg.Capstone.FullProgram
  less
    25% premium expenses as "G&A"
  peel bottom-up
  note{PnL (profit and loss) view for Capstone by program and layer.}
  tags{topic:capstone, role:advanced};

xpnl Capstone.PC
  derive premium 
  less
    agg Capstone.Slide agg.Capstone.XOL
      aggregate net of 
        75% po inf xs 0 rate 100% slide 0.48 at 0.42 and 0.38 at 0.62 and 0.18 at 0.82 as "QS w Slide" 
  less
      25% premium expenses as "G&A"
  note{Final Capstone model adding a sliding scale commission to the quota share. The XOL program is pulled in by reference, ensuring all terms stay consistent. Look at Capstone.XOL.PnL to calibrate the slide: loss ratios net of XOL are broadly between 42% and 83%. Peel is not available in this case, but it would add nothing new.}
  tags{topic:capstone, role:advanced};
   
grossnet
  agg.Capstone.XOL
  note{Bivariate distribution of gross and net of XOL program losses for Capstone. Also see: grossceded and netceded.}
  tags{topic:capstone, role:advanced};
```

## Group 2: renewal frequency (`years`, `wait`, `dwait`)

```decl
# ----------------------------------------------------------------------
# Renewal frequency (Sparre-Andersen)
# A `years` exposure paired with a waiting-time law. The wait clause takes
# the whole severity mini-language, written on the time axis.
# ----------------------------------------------------------------------

agg RenewalExponentialWait
  16 years
  dsev[1]
  wait expon
  note{Renewal frequency: claims arrive over a ten year exposure with exponential waiting times of mean one, which reproduces 10 claims sev lognorm 100 cv 1 poisson exactly. The wait clause takes the whole severity mini-language, so a waiting time is written like any other distribution, on the time axis.}
  tags{topic:frequency, role:advanced};

agg RenewalGammaWait
  5 years
  dsev[1]
  wait gamma 2
  note{Erlang waiting times, shape two and mean two, over a five year exposure, giving an expected count of 2.25. Non-exponential waits are what separate a renewal count from a Poisson one.}
  tags{topic:frequency, role:advanced};

agg RenewalDeterministicWait
  3 years
  dsev [1]
  dwait [1]
  note{Deterministic unit waits over three years, so the count is exactly three: the arrival falling on the horizon counts. dwait mirrors dsev, taking outcomes and probabilities.}
  tags{topic:frequency, role:advanced};

agg RenewalDefectiveWait
  2 years
  dsev [1]
  dwait [1 2] [0.4 0.5] !
  note{A defective, terminating renewal process: the wait probabilities sum to 0.9, so with probability 0.1 no further claim ever arrives and the expected count falls to 1.06. The trailing ! is what permits the defect.}
  tags{topic:frequency, role:advanced};

agg RenewalLayeredWait
  5 years
  dsev[1]
  wait 2 xs 1 expon
  note{The layered wait form applies the severity layer transform to the waiting time, here conditional on the wait exceeding one and capped two above it, lifting the expected count to 5.58. Add ! for the unconditional reading, where waits below the attachment collapse to a zero-wait cluster.}
  tags{topic:frequency, role:advanced};
```

All five are canonical. `RenewalExponentialWait` is verified identical to `10 claims ... poisson` (n, mean and cv agree to machine precision), which is what makes it the right first entry.

## Group 3: reinsurance economics and variable features (`cede`, `rate`, `free`, `no`)

```decl 
xpnl CedingCommission 
  inherit premium
  less
    agg CedingCommission.Book
      10000 premium at 0.7 lr
      2000 xs 0
      sev lognorm 150 cv 2.5
      occurrence net of
        1750 xs 250 rate 0.325 cede 0.1 as "Occ Re"
      mixed gamma 0.25
  less
    2000 fixed expense as Expenses
  note{Ceded premium as a rate on the gross premium, with a 10% ceding commission returned on it. rate is the third premium form beside deposit and rol, and cede is only valid alongside a premium; the layer is entered gross of commission.}
  tags{topic:reinsurance, role:advanced};

#[needs UNPARSER_EXEMPT]
xpnl ReinstatementTreaty                              
  10000 premium
  less
    agg ReinstatementTreaty.Book
      10000 premium at 0.9 lr
      20000 xs 0
      sev 1500 * lomax 1.5  
      occurrence net of
        18000 xs 2000 rol 0.275 reinstatement 1 at 50% and 1 at 100% as "Cat"
      mixed gamma 0.25
  note{The treaty-language reinstatement schedule: one free reinstatement then two at the full base rate on line, so the cover runs to four annual limits. It builds the same schedule as the explicit multiplier list [0.5 1].}
  tags{topic:reinsurance, role:advanced};

# [needs UNPARSER_EXEMPT]
xpnl NoReinstatementTreaty                              
  10000 premium
  less
    agg ReinstatementTreaty.Book
      10000 premium at 0.9 lr
      20000 xs 0
      sev 1500 * lomax 1.5  
      occurrence net of
        18000 xs 2000 rol 0.275 no reinstatements as "Cat"
      mixed gamma 0.25
  note{Zero reinstatements: a single annual limit, recovery capped there and no reinstatement premium, so the ceded premium stays deterministic. This is distinct from omitting the clause, which leaves the cover free and unlimited.}
  tags{topic:reinsurance, role:advanced};

# [needs UNPARSER_EXEMPT]
xpnl ReinstatementNumberWords                                 
  10000 premium
  less
    agg ReinstatementNumberWords.Book
      10000 premium at 0.9 lr
      20000 xs 0
      sev 1500 * lomax 1.5  
      occurrence net of
        18000 xs 2000 rol 0.275 reinstatements one free and three at 100% as "Cat"
      mixed gamma 0.25
  note{Reinstatement counts written as number words rather than digits. The words one to five are read as counts only in this position, so they stay ordinary identifiers everywhere else.}
  tags{topic:reinsurance, role:advanced};

agg VariableFeature.Base
  10000 premium at 0.9 lr
  2000 xs 0
  sev lognorm 150 cv 2.5
  mixed gamma 0.25

xpnl SwingRatedAggCover
  10000 premium
  less
    agg SwingRated agg.VariableFeature.Base
      aggregate net of
        5000 xs 8000 swing basic 300 lcm 1.1 min 1000 max 3000 as "Agg XOL"
  note{Swing rating: the ceded premium is a collared affine function of ceded loss, clip(basic + lcm * ceded, min, max). Swing replaces the deposit, rol or rate clause rather than decorating it.}
  tags{topic:reinsurance, role:advanced};


pnl RetroRatedAccount
  retro basic 500 lcm 1.05 min 6000 max 13000 premium
  less
    agg RetroRated agg.VariableFeature.Base
  note{Retrospective rating: the account-level clause that varies the gross premium with the account loss, replacing the fixed premium head. Retro currently requires a book with no inuring reinsurance, the one-dimensional case where net account loss equals gross loss. xpnl not available for retro rating.}
  tags{topic:reinsurance, role:advanced};

xpnl SlidingScaleCommission
  12000 premium
  less
    agg SlidingScale agg.VariableFeature.Base
      aggregate net of
        80% po inf xs 0 rate 100% slide 0.32 at 0.6 and 0.22 at 0.7 and 0.17 at 0.8 as "QS+slide"
  less
    22.5% premium expense as "Expenses"
  note{Sliding scale commission: the ceding commission slides down the anchors as the ceded loss ratio rises, interpolating between them and flat outside. Slide replaces the cede clause and needs a base premium to form the ratio.}
  tags{topic:reinsurance, role:advanced};

xpnl ProfitCommission
  inherit premium
  less
    agg ProfitCommission.Book
      10000 premium at 0.9 lr
      2000 xs 0
      sev lognorm 150 cv 2.5
      mixed gamma 0.25
      aggregate net of
        80% po inf xs 0 rate 100% pc 0.35 after 0.1 as "QS + P/C"
  note{Profit commission: the reinsurer returns 35% of the profit remaining after a 10% expense allowance, so the credit is share * (1 - ceded loss ratio - allowance) of ceded premium, floored at zero. Net of expense calculation.}
  tags{topic:reinsurance, role:advanced};

xpnl LossCorridor
  inherit premium
  less
    agg LossCorridor.Book
      10000 premium at 0.7 lr
      2000 xs 0
      sev lognorm 150 cv 2.5
      mixed gamma 0.25
      aggregate net of
        80% po inf xs 0 rate 80% corridor 0.5 po 0.4 xs 0.5 as "QS+Corridor"
  less 
    20% premium expense as "Expenses"
  note{Loss corridor: the cedant takes back half of the ceded losss in a ratio band running 40 points excess of 50%, so the recovery flattens across the corridor and resumes above it. The band is written with the layer words po and xs. Net quota share with no cede.}
  tags{topic:reinsurance, role:advanced};


```

## Group 5: bivariate (`grossnet`, `grossceded`, `clash`, `dbvsev`)

```decl
agg Agg.w.Occ.Re 
8 claims
sev 300 * beta 2 3
occurrence net of
  50% po 250 xs 50
poisson

grossnet
  agg.Agg.w.Occ.Re
  note{The joint distribution of gross and net for an excess of loss program. The three view prefixes name the axes in order: netceded, grossceded and grossnet.}
  tags{topic:bivariate, topic:reinsurance, role:advanced};

grossceded
  agg.Agg.w.Occ.Re
  note{The joint distribution of gross and ceded for the same program, the view that shows how much of a bad gross year the treaty actually pays.}
    tags{topic:bivariate, topic:reinsurance, role:advanced};

netceded
  agg.Agg.w.Occ.Re
  note{The joint distribution of gross and ceded for the same program, the view that shows how much of a bad gross year the treaty actually pays.}
    tags{topic:bivariate, topic:reinsurance, role:advanced};

clash BivariateClash
  8 5 2 claims
  sev lognorm 50 cv 0.6
  sev lognorm 60 cv 0.8
  mixed gamma 0.2
  note{A shared-event clash model: 8 claims are A only, 5 are B only, and 2 are both, and the solver derives the shared frequency and the per-event triggers, so the two components become Bernoulli factories on one common count.}
  tags{topic:bivariate, role:advanced};

bivariate BivariateDiscreteJoint
  dfreq [1 5 20] [0.5 0.25 0.25]
  dbvsev [1 2 3] [1 2 3] [[3/16 1/8 0] [1/8 1/8 1/8] [0 1/8 3/16]]
  note{Discrete bivariate severity, the two-dimensional analogue of dsev: the joint per-claim probability matrix is given directly on an explicit lattice, row index the X outcome and column index the Y outcome. The marginals reproduce the standalone dsev compounds exactly.}
  tags{topic:bivariate, role:advanced};

# [needs UNPARSER_EXEMPT]
bivariate BivariateDiscreteSparse                            
  5 claims
  dbvsev [[10 1 0.4] [10 2 0.1] [20 1 0.2] [20 2 0.15] [1 50 .15]]
  fixed
  note{The sparse specification of dbvsev: outcome, outcome and probability triples, which reads better than a dense matrix when most of the joint lattice is empty. It expands to the dense form on clicking Reformat, which is why it does not round-trip. Enjoy on log-scale!}
  tags{topic:bivariate, role:advanced};
```

`BivariateClash` is the slowest of the whole set at **4.7s**, so it wants `SLOW_ENTRIES`.

## Group 6: the small forms

```decl
# [needs UNPARSER_EXEMPT]
agg ExposureRatedPolicy                            
  10000 exposure at 0.125 rate
  250 xs 0
  sev lognorm 40 cv 1.5
  poisson
  note{The exposure-rated exposure clause: expected loss is exposure times rate, 10_000 at 0.125 for a loss pick of 12_500, and the claim count is derived from the severity. The spec records it exactly as premium at lr does, so the spelling does not survive a reformat.}
  tags{topic:aggregate, role:intermediate};

agg EqualWeightMixture
  10 claims
  500 xs 0
  sev lognorm [20 40 100] cv [0.5 0.6 0.7] wts=3
  mixed gamma 0.2
  note{The equal-weight shorthand: wts=3 gives three components of weight one third each, a shorthand for the explicit list. The severity mean is their equally weighted average, before the 500 limit.}
  tags{topic:severity, role:intermediate};

agg PayoffPrimitive
  dfreq [3]
  dsev [-13 -8 -3 5 8]
  payoff
  note{The payoff orientation suffix marks the variable as an asset-return primitive rather than a loss, so it prices through the dual distortion. loss is the twin suffix, stating the ordinary convention explicitly.}
  tags{topic:economics, role:advanced};

# [needs UNPARSER_EXEMPT]
agg TowerLimitProfile                                        
  1 claim
  tower [0 100 250 500]
  sev lognorm 100 cv 3
  fixed
  note{A tower written in the policy limit position rather than the reinsurance one, expanding to the three consecutive layers 100 xs 0, 150 xs 100 and 250 xs 250.}
  tags{topic:aggregate, role:intermediate};

# XXXXX
agg SplicedDisjointSegments
  1 claim
  sev lognorm 10 cv 1 splice [2 20]
  fixed
  note{Conditional on the severity in [2, 20].}
  tags{topic:severity, role:advanced};

agg UnconditionalDiscreteSeverity
  5 claims
  4 xs 2
  dsev [1 2 3 4 5 6] !
  poisson
  note{The trailing ! makes the severity unconditional, so the five claims are ground-up claims and the two die faces below the attachment contribute nothing. Conditional is the default, where the count is of claims reaching the layer; the severity mean falls from 2.5 to 1.667 here.}
  tags{topic:severity, role:advanced};

agg PascalFrequency
  10 claims
  dsev[1] 
  pascal 0.8 3
  note{A two-parameter frequency: the Pascal, a Poisson stopped sum of negative binomials, taking a contagion and a claims-per-occurrence parameter.}
  tags{topic:frequency, role:advanced};

agg DelaporteMixedFrequency
  10 claims
  dsev[1]
  mixed delaporte 0.65 0.25
  note{A two-parameter mixing law: the Delaporte is a shifted gamma mixed Poisson, taking the mixing cv and the shift. sig and sichel take two parameters in the same position.}
  tags{topic:frequency, role:advanced};

# [needs UNPARSER_EXEMPT] ISSUE
distortion MixtureDistortion mixture dist.PHDistortion dist.DualDistortion wts [0.6 0.4]
  note{A weighted mixture of two distortions, itself a distortion. Weights are only meaningful for mixture: the other combinators, minimum among them, take the child list alone.}
  tags{topic:spectral, role:advanced};                       
```

`MixtureDistortion` must sit **after** `PHDistortion` and `DualDistortion` in the file, since builtin references resolve by sequential load. The keyword is `wts`, not `weights`.

## Group 7: expense and pnl related 

```decl

xpnl PnL.Expenses
  10000 premium as "GWP"
  less
    agg Book.Agg.1
      6500 loss
      1000 xs 0
      sev lognorm 100 cv 2
      mixed ig 0.25
  less
    500 fixed expense as Fees
    15% premium expense as Commission
    10% loss expense as LAE
note{Explicitly specify premium in pnl clause. List of expenses appear as separate line items.}
tags{topic:economics, role:advanced}

xpnl  PnL.Inherit
  inherit premium as "GWP"
  less
    agg Book.Agg.2
      10000 premium at 0.65 lr 
      1000 xs 0
      sev lognorm 100 cv 2
      mixed ig 0.25
  less
    500 fixed expense and 25% premium expense as "Total Expenses" 
note{Inherit premium from the aggregate (must be present). Combined expenses with "and" with label at the end do not appear split out in the PnL statement.}
tags{topic:economics, role:advanced}


xpnl PnL.Derive
  derive premium as "GWP"
  less
    agg Book.Agg.3
      10000 premium as "Technical Premium" at 0.65 lr 
      1000 xs 0
      sev lognorm 100 cv 2
      occurrence net of 
        750 xs 250 deposit 4000 cede 32.5% as "Occ Re"
      mixed ig 0.25
  less
    2000 fixed expense as "Overhead"
    25% premium expense as "Commission"
note{Derive the premium from the aggregate technical premium by grossing up for expenses.}
tags{topic:economics, role:advanced}

```


## On `UNPARSER_EXEMPT`

Yes, with one correction to the wording. It is not about the **formatting** round-tripping. The test compares the *flattened* statement, so indentation and line breaks are normalized away before the comparison and can never move it. What fails is that **the source spelling is not recoverable from the spec**: the parser evaluates or expands the clause and keeps only the result, so `format_program` renders the result rather than what you wrote.

Your three guesses are all in the set, plus several more. The full catalogue, from the 27 currently failing plus these seven:

| cause                              | example                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------- |
| arithmetic evaluated at parse      | `ph 2/3` → `0.6666666666666666`                                            |
| array shorthand expanded           | `dsev [1:6]` → `[1 2 3 4 5 6]`; `[1:3] claims` → `[1 2 3] claims`          |
| tower expanded                     | `tower [0 100 250 500]` → `[100 150 250] xs [0 100 250]`                   |
| treaty language collapsed          | `1 free and 2 at 100%` → `[0 1 1]`                                         |
| number words to digits             | `one free and three` → `[0 1 1 1]`                                         |
| digit separators dropped           | `1_000` → `1000`                                                           |
| percentages to decimals            | `70% lr` → `0.7 lr`                                                        |
| sparse to dense                    | `dbvsev [[10 1 .4] ...]` → the full matrix                                 |
| synonym spellings sharing one spec | `1000 exposure at 0.05 rate` → `1000 premium at 0.05 lr`                   |
| named references not retained      | `minimum` / `mixture` distortion combinators: the unparser raises outright |

The practical consequence, which matters more than the green test: `dev/done/reflow_library.py` would **rewrite** an unexempted entry into the canonical spelling, deleting the exact thing the entry exists to demonstrate. Exempting `ClaimCountRange` is not test hygiene, it is what keeps `[1:3]` in the file.

Two things fell out of this that are yours to judge:

- **`exposure at rate` and `premium at lr` produce byte-identical specs.** Both write `exp_premium`, `exp_lr` and `exp_el`; the transformer methods differ only in variable names. So the exposure spelling is a pure synonym that can never round-trip, and an exposure base is recorded as a premium for anything downstream reading `exp_premium`.
- **The parse error for `weights` suggests itself back.** `distortion X mixture dist.A dist.B weights [...]` reports `Unexpected 'weights'. Did you mean: weights?` The Expected list does carry `wts`, so the fix is there, but the "did you mean" is echoing the input token rather than proposing the correction.

Coverage tally: your six variable-rating entries close 12 of the 34 missing terminals, these 25 close another 14. The remaining 8 are `EXPENSES`, `FIXED`, `DERIVE`, `PEEL`, `XPNL` (group 1, which your variable work will pick up) and `BUILTIN_SEV`, `BUILTIN_PORT`, `INHOMOG_MULTIPLY` (group 3, the secret feature).
---

## Execution notes (folded into library.agg at 1.0.0a327, 2026-08-27)

All 43 entries compiled, built and landed, in the groups above. Divergences
from the file as written:

**Renames.** The capstone entries took a `Capstone.` prefix so the group
reads as one worked example and stops squatting generic global names:
`Gross` to `Capstone.Gross`, `Exposure.Rating` to `Capstone.ExposureRating`,
`Selected.Losses` to `Capstone.SelectedLosses`, `Test.Loss.Picks` to
`Capstone.LossPicksTest`. `Agg.w.Occ.Re` became `BivariateOccReBase` (the
letter-prefix test rejects `Agg.`, and the name collided with its own views,
below). `PnL.Expenses` / `PnL.Inherit` / `PnL.Derive` became `PnLExpenses` /
`PnLInherit` / `PnLDerive` (same test rejects `PnL.`), matching the existing
`PnLSimple` family.

**The view statements needed names.** An unnamed `grossnet agg.X` names its
bvagg `X`, so the three views of one base collide with the base and with each
other under the unique-names rule. Each view now wraps the base through a
named derived agg: `Capstone.GrossNet`, and `BivariateGrossNet` /
`BivariateGrossCeded` / `BivariateNetCeded` over `BivariateOccReBase`.

**A duplication found and resolved.** The existing inline `netceded
BivariateNetCeded` entry was byte-for-byte the same model as `Agg.w.Occ.Re`
(8 claims, `300 * beta 2 3`, `50% po 250 xs 50`). It was replaced by the DRY
base plus the three views, keeping its name on the netceded view. The
`netceded` note here was a copy of the grossceded one and was rewritten.

**Content corrections.** `ReinstatementTreaty` was written `1 at 50% and 1
at 100%` while its note described one free and two at par ([0 1 1]); the decl
now reads `1 free and 2 at 100%`, which also demonstrates `free` as the group
header promises. `NoReinstatementTreaty`'s inner book was named
`ReinstatementTreaty.Book` by copy-paste and is now its own.
`RenewalExponentialWait` was `16 years` against a note describing ten; it is
`10 years` (matching the Poisson identity to 1e-8, and 2.3s instead of 8.1s).
`ExposureRatedPolicy`'s note claimed a loss pick of 12_500 from 10_000 at
0.125; it is 1_250. `BivariateDiscreteJoint`'s fractions were respelled as
decimals so the entry round-trips.

**SplicedDisjointSegments rewritten.** As drafted (`splice [2 20]`) it
duplicated `SevSpliced`. The two-list form pairs each mixture COMPONENT with
its own bounds, so the disjoint-segments entry is now `sev lognorm [10 10]
cv [1 1] wts [0.7 0.3] splice [2 10] [5 20]`: weight 0.7 on [2, 5], 0.3 on
[10, 20], zero in the gap, verified numerically. FINDING: the one-component
two-list form (`sev lognorm 10 cv 1 splice [2 10] [5 20]`) silently builds
an improper law (mass 0.23 in the gap, total under 0.65); it wants a
parse-time guard on the component/bounds length mismatch.

**Delaporte is honestly DEFECTIVE.** The shifted-gamma mixing pushes 1.1e-4
of count mass off the grid. FINDING: it cannot be hinted away because the
discrete-severity update path sizes its own grid and ignores both a caller
`log2` and a `hints{log2=...}` clause. Entered in `VALIDATION_BASELINE`
beside `InverseGaussianMixed`; `Capstone.SelectedLosses` entered under the
picks cause with all six moment flags.

**Test consequences.** Nineteen new entries measured over a second and
joined `SLOW_ENTRIES` (the capstone chain is 1.2s to 4.4s per link). The
first nested `dbvsev [[...]]` entries flip `preprocess` step 3 onto its
depth-aware path, which does not pad brackets the way the whole-file fast
path does, so `test_library_is_written_in_the_canonical_layout` now compares
whitespace-normalized text, and a `test_recipe` as-read assertion that
passed only on that padding was corrected. The 27 pre-existing non-canonical
entries this exposed (the "27 currently failing" above) are in
`UNPARSER_EXEMPT` under the cause table, except `dfreq[1]` spacing and the
`hints`-before-trailer ordering, fixed in place, and `MED.Exposure` /
`MED.WithPicks`, renamed `CommAuto.Exposure` / `CommAuto.WithPicks` for the
letter-prefix rule (the trailer test that pinned `agg.MED.WithPicks` moved
with it).

**Out of scope, noted for the author.** The `weights` did-you-mean echo
(above) stands. `test_split_limit_policy_prices_the_per_accident_limit`
fails on this tree at 1.7e-5 against a 1e-9 tolerance; verified identical
with the a326 library.agg, so it predates this merge and is unrelated.
