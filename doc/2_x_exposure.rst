.. _2_x_exposure:

Specifying exposure
======================

Marked change.

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


Determining Expected Claim Count
---------------------------------

Variables are used in the following order to determine overall expected losses.

* If ``count`` is given it is used
* If ``loss`` is given then count is derived from the severity
* If ``prem[ium]`` and ``[at] 0.7 lr`` are given then the loss is derived and counts from severity

In addition:

* If ``prem`` is given the loss ratio is computed
* Claim count is conditional but severity can have a mass at zero
* X is the GROUND UP severity, so X | X > attachment is used and generates n claims **really?**

Conditional and Unconditional Severity
--------------------------------------

* n is the CONDITIONAL claim count
* X is the GROUND UP severity, so X | X > attachment is used and generates n claims


The severity distribution is conditional on a loss to the layer. For an excess layer $y$ xs $a$ the severity is has distribution $X\mid X>$, where $X$ is the specified severity. For a ground-up layer there is no adjustment.

The default behavior can be over-ridden by adding `!` after the severity distribution. For example

::

    agg Conditional 1 claim 10 x 10 sev lognorm 10 cv 1 fixed
    agg Unconditional 1 claim 10 x 10 sev lognorm 10 cv 1 ! fixed


produces conditional and unconditional samples from an excess layer of a lognormal. The latter includes an approximately 0.66 chance of a claim of zero, corresponding to :math:`X\le 10` below the attachment.

Jupyter Workbooks
-----------------

Further topics are explored in a series of Jupyter Lab notebooks.

1. Basic Reinsurance
2. Dice aggregates
3. Discrete aggregates
4. Limit profiles and mixed severity LPMS
5. Mixed exponentials
