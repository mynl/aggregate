.. _re loss picks:

Adjusting Layer Loss Picks
===========================

**Objectives:** Describe how to adjust severity distributions to achieve selected loss picks by layer in an excess of loss reinsurance or layered insurance program.

**Audience:** Reinsurance and large account actuaries.

**Prerequisites:** DecL, general use of ``aggregate``, probability.

**See also:**


Reinsurance actuaries apply experience and exposure rating to excess of
loss programs. Experience rating trends and develops layer losses to
estimate loss costs. Exposure rating starts with a (ground-up) severity
curve. In the US, these are often published by a rating agency (ISO,
NCCI). It then applies a limit profit and uses difference of ILFs with a
ground up loss ratio to estimate layer losses. The actuary then selects
a loss cost by layer based on the two methods. When the selection is
different from the exposure rate, the actuary no longer has a
well-defined stochastic model for the business. In this section we show
how to adjust the severity curve to match the selected loss picks by
layer. The adjusted curve can then be used in a stochastic model that
will replicate the layer loss selections.

Layer severity equals the integral of the survival function and layer
expected losses equals layer frequency times severity. The easiest way
to adjust a single layer is to scale the frequency. The simple approach
fails when there are multiple layers because higher layer frequency
impacts lower layers. We are led to adjust the survival function in each
layer to hit the all selected layer loss picks. The method described
next creates a legitimate, non-increasing survival function and retains
its continuity properties whenever possible. It is easy to *select*
inconsistent layer losses which produces negative probabilities or
values greater than 1. When such inconsistencies occur the selections
must be altered.

Here is the layer adjustment process. Adjustments to higher layers impact
all lower layers because they change the probability of limit losses.
The approach is to start from the top-most layer, figure its adjustment,
and then take the impact of that adjustment into account on the next
layer down, and so forth. The adjusted severity curve to maintain the
shape of the curve and it continuity properties, conditional on a loss
in each layer.

To make these ideas rigorous requires a surprising amount of notation.
Define

-  Specify layer attachment points
   :math:`0=a_0 < a_1 < a_2 < \cdots < a_n` and corresponding layer
   limits :math:`y_i = a_i - a_{i-1}` for :math:`i=1,2, \dots, n`. The
   layers are :math:`l_i` excess :math:`a_{i-1}`.
-  :math:`l_i = \mathsf{LEV}(a_i) - \mathsf{LEV}(a_{i-1}) = \int_{a_{i-1}}^{a_i} S(x)dx = \mathsf{E}[ (X-a_{i-1})^+ \wedge y_i ]`
   equals the unconditional expected layer loss (per ground-up claim).
-  :math:`p_i = \Pr(a_{i-1} < X \le a_i) = S(a_{i-1}) - S(a_i)` equals
   the probability of a loss *in the layer*, excluding the mass at the
   limit.
-  :math:`e_i = y_iS(a_i)` equals the part of :math:`l_i` from full
   limit losses.
-  :math:`f_i = a_{i-1}p_i`
-  :math:`m_i = \int_{a_{i-1}}^{a_i} xdF(x) - f_i = \int_{a_{i-1}}^{a_i} (x-a_{i-1})dF(x) = l_i - e_i`
   equals the part of :math:`l_i` from losses in the layer.
-  :math:`t_i` are selected unconditional expected losses by layer.
   :math:`t_i=l_i` resutls in no adjustment. :math:`t_i` is computed by
   dividing the layer loss pick by the expected number of ground-up
   claims.

Integration by parts gives

.. math::
   \begin{align}
   \int_{a_{i-1}}^{a_i} S(x)dx
   &= xS(x)\,\big\vert_{a_{i-1}}^{a_i} + \int_{a_{i-1}}^{a_i} x dF(x) \\
   %&= a_iS(a_i) - a_{i-1}S(a_{i-1}) + \int_{a_{i-1}}^{a_i} x dF(x) \\
   &= a_iS(a_i) + \int_{a_{i-1}}^{a_i} (x - a_{i-1}) dF(x) \\
   &= e_i + m_i.
   \end{align}

These quantities are illustrated in the next figure.

.. ipython:: python
    :okwarning:

    from aggregate.extensions.figures import adjusting_layer_losses
    @savefig picks.png
    adjusting_layer_losses();

There is no adjustment to :math:`S` for :math:`x\ge a_n`. In the top
layer, adjust to :math:`\tilde S(x) = S(a_n) + w_n(S(x) - S(a_n))`, so

.. math::
   \begin{align}
   t_n
   &= \int_{a_{n-1}}^{a_n} \tilde S(x)dx \\
   &= S(a_n)y_n + w_n(l_n - e_n) \\
   &= \omega_n y_n + w_nm_n \\
   \implies w_n &= \frac{t_n - \omega_n y_n}{m_n},
   \end{align}

where :math:`\omega_n=S(a_n)`. Set
:math:`\omega_i = \omega_{i+1} + w_{i+1} p_{i+1}` and
:math:`\tilde S(x) = \omega_i + w_n(S(x) - S(a_n))` in the :math:`i`\ th
layer. We can compute all the weights by proceeding down the tower:

.. math::
   \begin{align}
   t_i
   &= \int_{a_{i-1}}^{a_i} \tilde S(x)dx \\
   &= \omega_i y_i + w_i(l_i - e_i) \\
   \implies w_i &= \frac{t_i - \omega_i y_i}{m_i}.
   \end{align}

:math:`\tilde S` is continuous is :math:`S` is because of the definition
of :math:`\omega` at the layer boundaries. When :math:`x=a_{i-1}`,
:math:`\tilde S(a_{i-1}) = \omega_i + w_i(S(a_{i-1}) - S(a_i)) = \omega_i + w_ip_i = \omega_{i=1}`.

The function ``utilities.picks_work`` computes the adjusted severity. In
debug mode, it returns useful layer information. A severity can be
adjusted on-the-fly by ``Aggregate`` using the ``picks`` keyword after
the severity specification and before any occurrence reinsurance.

