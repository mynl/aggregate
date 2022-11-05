.. _all_the_rest:



Tutorial 1: Creating, updating, plotting and inspecting an ``Aggregate`` object
-------------------------------------------------------------------------------


How Expected Claim Count is determined
--------------------------------------

* en determines en
* prem x loss ratio -> el
* severity x en -> el

* always have en and el; may have prem and exp_lr
* if prem then exp_lr computed
* if exp_lr then premium computed

* el is determined using np.where(el==0, prem*exp_lr, el)
* if el==0 then el = freq * sev
* assert np.all( el>0 or en>0 )

* call with el (or prem x exp_lr) (or n) expressing a mixture, with the same severity
* call with el expressing lines of business with an array of severities
* call with single el and array of sevs expressing a mixture; [] broken down by weights

Conditional and Unconditional Severity
--------------------------------------

* n is the CONDITIONAL claim count
* X is the GROUND UP severity, so X | X > attachment is used and generates n claims



2. How-To Guides
================

**How to guides** provide step-by-step instructions to solve specific problems. They assume the material covered in the tutorials.

Contents


Limit Profiles
--------------

The exposure variables can be vectors to express a *limit profile*.
All ```exp_[en|prem|loss|count]``` related elements are broadcast against one-another.
For example ::

    [100 200 400 100] premium at 0.65 lr [1000 2000 5000 10000] xs 1000

expresses a limit profile with 100 of premium at 1000 x 1000; 200 at 2000 x 1000
400 at 5000 x 1000 and 100 at 10000 x 1000. In this case all the loss ratios are
the same, but they could vary too, as could the attachments.

Mixtures
--------

The severity variables can be vectors to express a *mixed severity*. All ``sev_``
elements are broadcast against one-another. For example ::

    sev lognorm 1000 cv [0.75 1.0 1.25 1.5 2] wts [0.4, 0.2, 0.1, 0.1, 0.1]

expresses a mixture of five lognormals with a mean of 1000 and CVs as indicated with
weights 0.4, 0.2, 0.1, 0.1, 0.1. Equal weights can be express as wts=[5], or the
relevant number of components.


Limit Profiles and Mixtures
---------------------------

Limit profiles and mixtures can be combined. Each mixed severity is applied to each
limit profile component. For example ::

    ag = uw('agg multiExp [10 20 30] claims [100 200 75] xs [0 50 75]
        sev lognorm 100 cv [1 2] wts [.6 .4] mixed gamma 0.4')```

creates an aggregate with six severity subcomponents.

+---+-------+------------+--------+
| # | limit | attachment | claims |
+===+=======+============+========+
| 0 | 100   |  0         |  6     |
+---+-------+------------+--------+
| 1 | 100   |  0         |  4     |
+---+-------+------------+--------+
| 2 | 200   | 50         | 12     |
+---+-------+------------+--------+
| 3 | 200   | 50         |  8     |
+---+-------+------------+--------+
| 4 |  75   | 75         | 18     |
+---+-------+------------+--------+
| 5 |  75   | 75         | 12     |
+---+-------+------------+--------+

Circumventing Products
----------------------

It is sometimes desirable to enter two or more lines each with a different severity but
with a shared mixing variable. For example to model the current accident year and a run-
off reserve, where the current year is gamma mean 100 cv 1 and the reserves are
larger lognormal mean 150 cv 0.5 claims requires ::

    agg MixedPremReserve [100 200] claims sev [gamma lognorm] [100 150] cv [1 0.5] mixed gamma 0.4

so that the result is not the four-way exposure / severity product but just a two-way
combination. These two cases are distinguished looking at the total weights. If the weights sum to
one then the result is an exposure / severity product. If the weights are missing or sum to the number
of severity components (i.e. are all equal to 1) then the result is a row by row combination.


For fixed or histogram, have to separate the parameter so they are not broad cast; otherwise
you end up with multiple lines when you intend only one


Jupyter Workbooks
-----------------

Further topics are explored in a series of Jupyter Lab notebooks.

1. Basic Reinsurance
2. Dice aggregates
3. Discrete aggregates
4. Limit profiles and mixed severity LPMS
5. Mixed exponentials
6. `Tweedie family <file:///C:/S/TELOS/Python/aggregate_project/examples/snippets/Tweedie.html>`_.
