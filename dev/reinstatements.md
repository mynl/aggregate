# Property Catastrophe Reinsurance with Reinstatements

## Introduction

Property catastrophe reinsurance is commonly written as occurrence excess-of-loss cover. A treaty described as $y$ xs $a$ pays the part of each catastrophe loss exceeding the occurrence attachment $a$, subject to an occurrence limit $y$.

Let annual gross catastrophe loss be
$$
G=X_1+\cdots+X_N,
$$
where $N$ is the annual number of catastrophe occurrences and the $X_i$ are independent and identically distributed occurrence severities.

Ignoring reinstatement limitations, the ceded loss from occurrence $i$ under a $y$ xs $a$ treaty is
$$
C_i=y\wedge(X_i-a)_+,
$$
where $x_+=\max(x,0)$ and $x\wedge z=\min(x,z)$.

The annual sum of occurrence recoveries before applying any reinstatement or annual aggregate limitation is
$$
R=C_1+\cdots+C_N.
$$
Thus, $R$ is the aggregate loss to an unlimited occurrence programme: each occurrence is limited to $y$, but there is initially no restriction on the number of occurrences to which the treaty can respond.

## Rate on line

The rate on line, abbreviated ROL, is the treaty premium divided by the occurrence limit. If the rate on line is $r$, the original treaty premium is
$$
P=ry.
$$
For example, a premium of 10 for a limit of 100 corresponds to a rate on line of
$$
r=\frac{10}{100}=10\%.
$$
Rate on line expresses premium as a proportion of the limit supplied. It does not by itself determine the expected loss ratio or profitability of the treaty.

## Reinstatements

Suppose the treaty provides $m$ paid reinstatements, each pro rata as to amount but not as to time.

A full reinstatement supplies additional capacity equal to one occurrence limit $y$. The original limit and the $m$ reinstated limits therefore provide total annual recovery capacity of
$$
(m+1)y.
$$
Partial recoveries consume reinstatement capacity pro rata. A partial recovery does not use an indivisible reinstatement event. For example, a recovery of $0.25y$ consumes $0.25y$ of reinstatement capacity.

The actual annual recovery is therefore
$$
A=R\wedge(m+1)y.
$$
For recovery modelling, an occurrence treaty with $m$ full automatic reinstatements is equivalent to the unlimited occurrence programme $R$, subject to an annual aggregate limit of $(m+1)y$.

The reinstatement premium applies only to replacement capacity. The maximum amount that can be reinstated is $my$, rather than $(m+1)y$. If each reinstatement is charged at 100% of the original rate on line, total reinstatement premium is
$$
\mathrm{RP}=r(R\wedge my).
$$
The first $my$ of occurrence recoveries therefore generates reinstatement premium at rate $r$. Recoveries between $my$ and $(m+1)y$ consume the final reinstated capacity but do not generate further reinstatement premium.

More generally, if reinstatements are charged at $q$ times the original premium rate, where $q=1$ denotes a 100% reinstatement, then
$$
\mathrm{RP}=qr(R\wedge my).
$$
The formulas assume that all reinstatements have the same price, are automatic, are pro rata as to amount, and are not pro rata as to time.

## Example 1: one occurrence

Consider a 100 xs 100 treaty with one paid reinstatement at 100%. Let
$$
y=100,\qquad a=100,\qquad m=1,\qquad r=10\%.
$$
The original premium is $ry=10$. Suppose there is one occurrence with gross loss
$$
X_1=175.
$$
The unlimited occurrence recovery is
$$
R=C_1=100\wedge(175-100)=75.
$$
The actual treaty recovery is
$$
A=75\wedge200=75,
$$
and the reinstatement premium is
$$
\mathrm{RP}=0.10(75\wedge100)=7.5.
$$
The recovery of 75 is fully reinstated. The treaty therefore retains a full occurrence limit of 100 for a subsequent event, while 25 of the possible reinstatement capacity remains unused.

## Example 2: two occurrences

Retain the same treaty and suppose the annual occurrence losses are
$$
X_1=250,\qquad X_2=150.
$$
The occurrence recoveries before the annual aggregate limitation are
$$
C_1=100,\qquad C_2=50,
$$
so that
$$
R=100+50=150.
$$
The actual annual recovery is
$$
A=150\wedge200=150.
$$
The reinstatement premium is
$$
\mathrm{RP}=0.10(150\wedge100)=10.
$$
The first occurrence exhausts the original limit and triggers the full reinstatement premium. The second occurrence consumes 50 of the reinstated limit. Because no further reinstatement is available, the second recovery does not generate additional reinstatement premium.

## Example 3: three or more occurrences exhausting all capacity

Again retain the same treaty and suppose the annual occurrence losses are
$$
X_1=200,\qquad X_2=110,\qquad X_3=200.
$$
The unlimited occurrence recoveries are
$$
C_1=100,\qquad C_2=10,\qquad C_3=100,
$$
and hence
$$
R=210.
$$
The actual recovery is limited to
$$
A=210\wedge200=200.
$$
The reinstatement premium is
$$
\mathrm{RP}=0.10(210\wedge100)=10.
$$
The first occurrence exhausts the original limit and uses the complete reinstatement entitlement. The second occurrence consumes 10 of the reinstated limit, leaving 90 available. The third occurrence would produce an occurrence recovery of 100, but only 90 of annual capacity remains. The treaty therefore pays 90 and is exhausted.

## Joint distribution of recovery and reinstatement premium

Both actual recovery and reinstatement premium are nondecreasing functions of the same random variable $R$:
$$
A(R)=R\wedge(m+1)y,
$$
and
$$
\mathrm{RP}(R)=r(R\wedge my).
$$
Consequently, $A$ and $\mathrm{RP}$ are comonotonic. Their joint distribution is concentrated on a piecewise-linear curve.

For $0\le R\le my$,
$$
(A,\mathrm{RP})=(R,rR).
$$
A plot of $\mathrm{RP}$ against $A$ therefore increases with slope $r$ from $(0,0)$ to
$$
(my,rmy).
$$
For $my<R\le(m+1)y$,
$$
(A,\mathrm{RP})=(R,rmy),
$$
so the graph then runs horizontally from
$$
(my,rmy)
$$
to
$$
((m+1)y,rmy).
$$
When $R>(m+1)y$, both quantities remain fixed at
$$
(A,\mathrm{RP})=((m+1)y,rmy).
$$
The joint distribution therefore has a point mass at the terminal point of size
$$
P\{R\ge(m+1)y\}.
$$
If reinstatement premium is rescaled as $\mathrm{RP}/r$, the graph of $\mathrm{RP}/r$ against $A$ initially has slope 1, runs from $(0,0)$ to $(my,my)$, and then runs horizontally to $((m+1)y,my)$.

## Distribution of net recovery

From the cedant's perspective, reinstatement premium is a payment. The economically relevant quantity is therefore the net recovery
$$
B=A-\mathrm{RP}.
$$
Since both $A$ and $\mathrm{RP}$ are deterministic nondecreasing functions of $R$, the net recovery is also a deterministic function of $R$:
$$
B(R)
=
\bigl(R\wedge(m+1)y\bigr)
-
r(R\wedge my).
$$
Equivalently,
$$
B(R)=
\begin{cases}
(1-r)R, & 0\le R\le my,\\
R-rmy, & my<R\le(m+1)y,\\
(m+1)y-rmy, & R>(m+1)y.
\end{cases}
$$
Provided $0\le r<1$, the function $B(R)$ is increasing. Its slope is $1-r$ while recoveries are generating reinstatement premium, then increases to 1 once all reinstatement premium has been incurred, and finally falls to zero when the treaty is exhausted.

The distribution of $B$ therefore follows directly from the distribution of $R$ by this piecewise-linear transformation. In particular, $B$ has an upper atom of size
$$
P\{R\ge(m+1)y\}
$$
at the maximum net recovery
$$
B_{\max}=(m+1)y-rmy.
$$
For the treaty with one reinstatement, $m=1$, so
$$
B_{\max}=2y-ry.
$$
The annual net recovery is independent of the ordering of the occurrence losses under the stated assumptions. Event order affects the timing of recoveries and reinstatement-premium payments, but not their ultimate annual difference.

## References

1. Anderson, R. R. and Dong, W. (1998), “Pricing Catastrophe Reinsurance with Reinstatement Provisions,” *Casualty Actuarial Society Forum*, Spring 1998.

2. Sanders, D. E. A. (1995), “An Introduction to Catastrophe Excess of Loss Reinsurance,” *Casualty Actuarial Society Forum*, Fall 1995.

3. Clark, D. R. (2014), *Basics of Reinsurance Pricing*, Casualty Actuarial Society and Society of Actuaries study note.

4. Mata, A. J. (2000), “Pricing Excess of Loss Reinsurance with Reinstatements,” *ASTIN Bulletin*, 30(2), 349–368.

5. Hürlimann, W. (2005), “Excess of Loss Reinsurance with Reinstatements Revisited,” *ASTIN Bulletin*, 35(1), 211–238.

