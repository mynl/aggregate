.. _q aep oep:

Occurrence and Aggregate Probable Maximal Loss
-------------------------------------------------

Probable maximal loss (PML)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Probable maximal loss** or **PML** and the related **maximum
foreseeable loss** (MFL) originated in fire underwriting in the early
1900s. The PML estimates the largest loss that a building is likely to
suffer from a single fire if all critical protection systems function as
expected. The MFL estimates the largest fire loss likely to occur if
loss-suppression systems fail. For a large office building, the PML
could be a total loss to 4 to 6 floors, and the MFL could be a total
loss within four walls, assuming a single structure burns down.
:cite:t:`McGuinness1969` discusses PMLs.

Today, PML is used to quantify potential catastrophe losses. Catastrophe
risk is typically managed using reinsurance purchased on an occurrence
basis and covering all losses from a single event. Therefore insurers
are interested in the annual frequency of events greater than an
attachment threshold, leading to the occurrence PML, now known as occurrence exceeding probabilities.

Occurrence Exceeding Probability (OEP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To describe **occurrence PMLs**, we need to specify the stochastic model
used to generate events. It is standard to use a homogeneous Poisson
process, with a constant event intensity :math:`\lambda` per year. The
number of events in time :math:`t` has a Poisson distribution with mean
:math:`\lambda t`. If :math:`X` is the severity distribution (size of
loss conditional on an event) then the number of events per year above
size :math:`x` has Poisson distribution with mean :math:`\lambda S(x)`.
Therefore the probability of one or more events causing loss :math:`x`
or more is 1 minus the probability that a
Poisson\ :math:`(\lambda S(x))` random variable equals zero, which
equals :math:`1-e^{-\lambda S(x)}`. The :math:`n` **year occurrence
PML**, :math:`\mathsf{PML}_{n, \lambda}(X)=\mathsf{PML}_{n, \lambda}`,
is the smallest loss :math:`x` so that the probability of one or more
events causing a loss of :math:`x` or more in a year is at least
:math:`1/n`. It can be determined by solving
:math:`1-e^{-\lambda S(\mathsf{PML}_{n, \lambda})}=1/n`, giving

.. math::

   S(\mathsf{PML}_{n, \lambda})&=\frac{1}{\lambda}\log\left( \frac{n}{n-1}\right) \\
   \implies \mathsf{PML}_{n, \lambda} &= q_X\left( 1 -\frac{1}{\lambda}\log\left( \frac{n}{n-1}\right) \right)

(if :math:`S(x)=s` then :math:`F(x)=1-s` and
:math:`x=q_X(1-s)=\mathsf{VaR}_{1-s}(X)`). Thus, the occurrence PML is
a quantile of severity at an adjusted probability level, where the
adjustment depends on :math:`\lambda`.

Converting to non-exceedance probabilities, if :math:`p=1-1/n` (close to
1) then :math:`n/(n-1)=1/p` and we obtain a relationship between the
occurrence PML and severity VaR:

.. math::

   \mathsf{PML}_{n, \lambda} = q_X\left( 1 +\frac{\log(p)}{\lambda} \right)
   =\mathsf{VaR}_{1+\log(p)/\lambda}(X)

Catastrophe models output a sample of :math:`N` loss events, each with
an associated annual frequency :math:`\lambda_i` and an expected loss
:math:`x_i`, :math:`i=1,\dots,N`. Each event is assumed to have a
Poisson occurrence frequency distribution. The associated severity
distribution is concentrated on the set :math:`\{x_1,\dots,x_N\}` with
:math:`\mathsf{Pr}(X=x_i)=\lambda_i/\lambda`, where
:math:`\lambda=\sum_i \lambda_i` is the expected annual event frequency.
It is customary to fit or smooth :math:`X` to get a continuous
distribution, resulting in unique quantiles.


Return Periods
~~~~~~~~~~~~~~~~~

VaR points are often quoted by **return period**, such as a 100 or 250
year loss, rather than by probability level. By definition, the
exceedance probability :math:`\mathsf{Pr}(X > \mathsf{VaR}_p(X))` of
:math:`p`-VaR is less than or equal to :math:`1-p`, meaning at most a
:math:`1-p` probability per year. If years are independent, then the
average waiting time to an exceedance is at least :math:`1/(1-p)`. (The
waiting time has a geometric distribution, with parameter :math:`p`. Let
:math:`q=1-p`. The average wait time is
:math:`q + 2pq + 3p^2q+\cdots=q(1+2p+3p^2+\cdots)=1/q`.)

Standard return periods and their probability representation are shown
below.

+----------------+----------------+----------------+------------------+
| **VaR          | **Exceedance   | **Return       |                  |
| threshold**    | probability**  | Period**       | **Applications** |
+================+================+================+==================+
| :math:`p`      | :math:`1-p`    | :math:`1/(1-p)`|                  |
+----------------+----------------+----------------+------------------+
| 0.99           | 0.01           | 100 years      |                  |
+----------------+----------------+----------------+------------------+
| 0.995          | 0.005          | 200 years      | Solvency 2       |
+----------------+----------------+----------------+------------------+
| 0.996          | 0.004          | 250 years      | AM Best, S&P,    |
|                |                |                | RBC              |
+----------------+----------------+----------------+------------------+
| 0.999          | 0.001          | 1,000 years    |                  |
+----------------+----------------+----------------+------------------+

In a Poisson model, the waiting time between events with a frequency of :math:`\lambda` has an exponential distribution with mean :math:`1/\lambda`. Thus, an event with frequency 0.01 is often quoted as having a 100 year return period. Notice, however, the distinction between the chances of no events in a year and the waiting time until the next event. If :math:`\lambda` is large, say 12 (one event per month on average), the chances of no events in a year equals
:math:`\exp(-12)=6.1\times 10^{-6}` is vs. a one-month return period. For small :math:`\lambda` there is very little difference between the two since the probability of one or more events equals :math:`1-\exp(-\lambda)\approx \lambda`.

To reiterate the definition above, when :math:`X` represents aggregate annual losses, the statement
:math:`x=\mathsf{VaR}_{0.99}(X)`, :math:`p=0.99` means

- :math:`x` is the smallest loss for which :math:`X\le x` with an annual probability of at least :math:`0.99`, or
- :math:`x` is the smallest loss with an annual probability at most :math:`0.01` of being exceeded.

Aggregate Exceeding Probability (AEP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Severity VaR (quantile) and occurrence PML are distinct but related concepts.
However, **aggregate PML** or **aggregate exceeding probability** is
often used as a synonym for aggregate VaR, i.e., VaR of the aggregate
loss distribution..

Let :math:`A` equal the annual aggregate loss random variable. :math:`A`
has a compound Poisson distribution with expected annual frequency
:math:`\lambda` and severity random variable :math:`X`. :math:`X` is
usually thick tailed. Then, as we explain shortly,

.. math::

   \mathsf{VaR}_p(A) \approx \mathsf{VaR}_{1-(1-p)/\lambda}(X).

This equation is a relationship between aggregate and
severity VaRs.

We can sometimes estimate aggregate VaRs in terms of occurrence PMLs with no
simulation. For large :math:`n` and a thick tailed :math:`X` occurrence
PMLs and aggregate VaRs contain the same information—there is not more
information in the aggregate, as is sometimes suggested. The
approximation follows from the equation

.. math::

   \mathsf{Pr}(X_1+\cdots +X_n >x) \to n\mathsf{Pr}(X>x)\ \text{as}\ x\to\infty

for all :math:`n`, which holds when :math:`X` is
sufficiently thick tailed. See :cite:t:`Embrechts1997`, Corollary 1.3.2 for the
details.
