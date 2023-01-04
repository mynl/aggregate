
.. _p sev irrel:

When Is Severity Irrelevant?
-------------------------------

In some cases the actual form of the severity distribution is essentially
irrelevant to the shape of the aggregate distribution. Consider an aggregate
with a :math:`G`-mixed Poisson frequency distribution. If the expected claim
count :math:`n` is large and if the severity is tame (roughly tame means
bounded or has a log concave density; a policy with a limit has a tame
severity; unlimited workers compensation or cat losses may not be tame) then
particulars of the severity distribution diversify away in the aggregate.
Moreover, the variability from the Poisson claim count component also
diversifies away, and the shape of the aggregate distribution converges to the
shape of the frequency mixing distribution
:math:`G`. Another way of saying the same thing is that the normalized
distribution of aggregate losses (aggregate losses divided by expected
aggregate losses) converges in distribution to :math:`G`.

We can prove these assertions using moment generating functions. Let
:math:`X_n` be a sequence of random variables with distribution
functions :math:`F_n` and let :math:`X` another random variable with
distribution :math:`F`. If :math:`F_n(x)\to F(x)` as :math:`n\to\infty`
for every point of continuity of :math:`F` then we say :math:`F_n`
converges weakly to :math:`F` and that :math:`X_n` converges in
distribution to :math:`F`.

Convergence in distribution is a relatively weak form of convergence. A
stronger form is convergence in probability, which means for all
:math:`\epsilon>0` :math:`\mathsf{Pr}(|X_n-X|>\epsilon)\to 0` as
:math:`n\to\infty`. If :math:`X_n` converges to :math:`X` in probability
then :math:`X_n` also converges to :math:`X` in distribution. The
converse is false. For example, let :math:`X_n=Y` and :math:`X=1-Y` be
binomial 0/1 random variables with :math:`\mathsf{Pr}(Y=1)=\mathsf{Pr}(X=1)=1/2`. Then
:math:`X_n` converges to :math:`X` in distribution. However, since
:math:`\mathsf{Pr}(|X-Y|=1)=1`, :math:`X_n` does not converge to :math:`X` in
probability.

It is a fact that :math:`X_n` converges to :math:`X` if the MGFs
:math:`M_n` of :math:`X_n` converge to the MFG of :math:`M` of :math:`X`
for all :math:`t`: :math:`M_n(t)\to M(t)` as :math:`n\to\infty`. See
:cite:t:`feller71` for more details. We can now prove the
following result.

.. container:: prop

   **Proposition.** Let :math:`N` be a :math:`G`-mixed Poisson distribution with mean
   :math:`n`, :math:`G` with mean 1 and variance :math:`c`, and let
   :math:`X` be an independent severity with mean :math:`x` and variance
   :math:`x(1+\gamma^2)`. Let :math:`A=X_1+\cdots+X_N` and :math:`a=nx`.
   Then :math:`A/a` converges in distribution to :math:`G`, so

   .. math:: \mathsf{Pr}(A/a < \alpha) \to \mathsf{Pr}(G < \alpha)

   as :math:`n\to\infty`. Hence

   .. math:: \sigma(A/a) = \sqrt{c + \frac{x(1+\gamma^2)}{a}}\to\sqrt{c}.

   *Proof.* We know

   .. math:: M_A(\zeta)=  M_G(n(M_X(\zeta)-1))

   and so using Taylor’s expansion we can write

   .. math::

      \lim_{n\to\infty} M_{A/a}(\zeta)
      &= \lim_{n\to\infty} M_A(\zeta/a)  \\
      &= \lim_{n\to\infty} M_G(n(M_X(\zeta/nx)-1))  \\
      &= \lim_{n\to\infty} M_G(n(M_X'(0)\zeta/nx+R(\zeta/nx)))  \\
      &= \lim_{n\to\infty} M_G(\zeta+nR(\zeta/nx)))  \\
      &= M_G(\zeta)

   for some remainder function :math:`R(t)=O(t^2)`. Note that the
   assumptions on the mean and variance of :math:`X` guarantee
   :math:`M_X'(0)=x=\mathsf{E}[X]` and that the remainder term in Taylor’s
   expansion actually is :math:`O(t^2)`. The second part is trivial.

The proposition implies that if the frequency distribution is actually a
Poisson, so the mixing distribution :math:`G=1` with
probability 1, then the loss ratio distribution of a very large book
will tend to the distribution concentrated at the expected, hence the
expression that “with no parameter risk the process risk completely
diversifies away.”

The next figure illustrate the proposition, showing how aggregates change
shape as expected counts increase.

.. ipython:: python
    :okwarning:

    from aggregate.extensions import mixing_convergence
    @savefig tr_prob_convg.png scale=20
    mixing_convergence(0.25, 0.5)

On the top, :math:`G=1` and the claim count is Poisson. Here the scaled
distributions get more and more concentrated about the expected value
(scaled to 1.0). Notice that the density peaks (left) are getting further
apart as the claim count increases. The distribution (right) is converging
to a Dirac delta step function at 1.

On the bottom, :math:`G` has a gamma distribution
with variance :math:`0.0625` (asymptotic CV of 25%). The density peaks are getting closer, converging to the mixing gamma. The scaled
aggregate distributions converge to :math:`G` (thick line, right).

It is also interesting to compute the correlation between :math:`A` and
:math:`G`. We have

.. math::

   \mathsf{cov}(A,G)
   &= \mathsf{E}[AG]-\mathsf{E}[A]\mathsf{E}[G]  \\
   &= \mathsf{E}\mathsf{E}[AG \mid G] - nx  \\
   &= \mathsf{E}[nxG^2] - nx  \\
   &= nxc,

and therefore

.. math:: \mathsf{corr}(A,G)=nxc/\sqrt{nx\gamma + n(1+cn)}\sqrt{c}\to 1

as :math:`n\to\infty`.

The proposition shows that in some situations severity is irrelevant to
large books of business. However, it is easy to think of examples where
severity is very important, even for large books of business. For
example, severity becomes important in excess of loss reinsurance when
it is not clear whether a loss distribution effectively exposes an
excess layer. There, the difference in severity curves can amount to the
difference between substantial loss exposure and none. The proposition
does *not* say that any uncertainty surrounding the severity
distribution diversifies away; it is only true when the severity
distribution is known with certainty. As is often the case with risk
management metrics, great care needs to be taken when applying general
statements to particular situations!

