I want to

1) build in expenses: build in gross expenses to pnl, get expenses wired through out the gcn and other pnl exhibits.
2) Add ceded premium to reinsurance clauses
3) add in variable-features and (adjust plan-reinstatements as needed) support retro rating, swings, slides and profit commissions.

I want all these in v1. I believe that will then be all for v1. (Famous last words.)

The pnl becomes Premium - Loss - Expenses = UW. The AI (acceptance index) still works on the uw result - no changes needed. So i think that sets us up to do everything.

Pls feel free to split into two or possibly sub-plans as you see best for execution. (1-2) and (3) looks like a good split IMO. Keep plan-reinstatements.

1) Add expenses

```python
pnl EXPENSES
   10000 premium
   less
   85% lr
       5000 xs 0
       sev lognorm 50 cv 3
       mixed ig .25
   # expenses are optional (0 if missing), base is EXPLICIT, clause placed LAST:
   25% premium expenses    # fraction of premium ('%' => not dollars)
   # or  25% loss expenses    (fraction of expected loss)
   # or  2000 fixed expenses  ('expense'/'expenses' both accepted)
```


2) Add premium and ceding commission (cede) info to reins

```python
occurrence (or aggregate) net of [ceded to] [program]
    # doesn't matter what you request - prem is for the ceded leg
    deposit [$$]  # deposit (initial) premium in currency
                  # NO `min` synonym (freed for the swing/retro collar) -- deposit only
    # or
    rol [%]    # % is rate on line, only if there is a limit
               # actual premium = share x rol * limit
    # or
    rate %     # applied to gross premium, run time error if no premium on computations
    # optional but only valid with a premium
    cede 24%   # cede as percent of ceded premium; premium entered gross
and ...    # one price per layer; error to have > 1 variable price.
    ...
```


3) Variable features

Features Supported
===================

| Feature      | What varies       | Inputs                              |
|:-------------|:------------------|:------------------------------------|
| Reinst. Prem | ceded premium     | see other plan and below            |
| Retro rating | ceded premium     | basic, lcm, min and max, loss basis |
| Swing        | ceded prem        | similar to retro but for reins      |
| Slide        | ceding commission | low, high, factor                   |
| PC           | Expenses          | base, x after y allowance           |
| Corridor     | Ceded loss        | from $$ to $$ share %%              |

Swings and retro are very directly analogous: retro applies to a whole account, swings to reinsurance. PC are often combined with other terms. RPs we have already discussed.

Examples
=========

First note, retro rating is different

```python
pnl RETRO
    # retro rating prices the whole book, possibly net of inuring reinsurance
    # this is for large account pricing
    # in this case we put the  info in the rating clause, prefix with retro to flag
    # what is coming
   retro basic 3000 lcm 1.1 min 3500 max 8000 premium
   # prem = max(3500, min(8000, 3000 + 1.1 L)), L = net loss out of aggregate
   # min = basic if no min, basic and lcm required; max=inf if missing
   # the retro clause REPLACES `<numbers> premium` (it sets gross premium variably)
   less
   5000 loss
        5000 xs 0
        sev lognorm 50 cv 3
        occurrence net of 95% po 3500 xs 0
        # reins occ program just shaping the net book - no premium
        # retro just  looks at the price for the net part.
        # occ reins cost is usually a pass through.
    mixed gamma 0.5
   # perspective is insurer quoting account; these are the insurer's expenses
   # (fixed and variable both allowed); clause placed LAST:
   200 fixed expenses
```

The remainder of the terms apply to reinsurance.


```python
pnl REINSTATEMENT_PREMIUM
   10000 premium
   less
   77.5% lr
       sev lognorm 50 cv 3
       occurrence net of
           95% po 100 xs 100
                rol 18%
                # no cede on cat
                reinstatements 1 free and 1 at 50% and two at 100%
                # reinstatements [0 .5 1 1]
       poisson
   25% premium expenses

# can apply to occ or agg
# reins version of retro rating
pnl SWING
   10000 premium
   less
   6000 loss
        5000 xs 0
        sev lognorm 50 cv 3
        occurrence net of 95% po 3500 xs 0
            swing 3000 basic 1.1 lcm 3500 min 8000 max # same as retro clause
            # swing replaces rate, rol, deposit clause
            # no cede
    mixed gamma 0.5
   3000 fixed expenses


# can apply to occ or agg
pnl SLIDE
    10000 premium
    less
    7000 loss
        5000 xs 0
        sev lognorm 50 cv 3
    mixed gamma 0.5
    aggregate net of 95% po inf xs 0
        rate 100%  # quota share
        slide [min_comm min_LR] [rate lr] [rate lr] max_comm
        # eg [.19 .8] [.25 .7] [.5 .6] .45 pays a min of 19% above 80% LR, then .25 for each point below .8
        # down to .7, then .5 for each point below .6 lR with an overall max of 45%
        # can apply to occ or acc
        # slide replaces cede - error to have both
        # could possibly use layer and attach language?
    28% premium expenses

# can apply to occ or agg
pnl PC
    10000 premium
    less
    7000 loss
        5000 xs 0
        sev lognorm 50 cv 3
        occurrence net of
            95% po 4500 xs 500
                deposit 1800
                pc 25% after 20%  # for loss L pc pays max(0, 0.25(1 - L/premium - 0.2))
    mixed gamma 0.5
    2000 fixed expenses

# can apply to occ or agg
pnl CORRIDOR
    10000 premium
    less
    7000 loss
        5000 xs 0
        sev lognorm 50 cv 3
        occurrence net of
            95% po 4500 xs 500
                deposit 2200
                cede 30%
                corridor 50% po a xs b and ... # a and b are ceded loss ratios entered as 0.2 or 20%
    mixed gamma 0.5
    2000 fixed expenses
```

Other Examples
===============


```python
pnl REINSTATEMENT_PREMIUM
   10000 premium
   less
   77.5% lr
       sev lognorm 50 cv 3
       occurrence net of
           95% po 100 xs 100
                rol 18%  # note this moved relative to original
                # no cede on cat
                reinstatements 1 free and 1 at 50% and two at 100%
       poisson
       aggregate net of
            85% po 1500 xs 7000
   20.5% premium expense
```
