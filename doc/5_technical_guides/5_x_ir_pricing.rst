
.. _ir stop loss:

Self-Insurance Plan Stop-Loss Insurance
------------------------------------------

Self-insurance plans often purchase per occurrence (specific) insurance, to
limit the amount from any one loss that flows into the plan, and aggregate
stop-loss insurance, to limit their aggregate liability over all occurrences
in a year. Retro rating plans need to estimate the **insurance charge** for
the aggregate cover. It is a function of the expected loss, the specific loss
limit, and the aggregate retention. They sometimes also want to know
the **insurance savings**, a credit for losses below a minimum. Tables
tabulating insurance savings and charges are called Table L (California) or
Table M (rest of the US).

Let :math:`X` denote unlimited severity, :math:`N` annual frequency, :math:`l`
the occurrence limit and :math:`a` the aggregate retention of limited losses.
The distribution of gross aggregate losses is given by

.. math::
    A_g := X_1 + \cdots + X_N.

Aggregate losses retained by the plan, reflecting the specific but not the
aggregate insurance, are a function of :math:`l` and :math:`n:=\mathsf E
[N]` the expected ground-up claim count, with distribution

.. math::
    A(n, l) := (X_1 \wedge l) + \cdots + (X_N \wedge l).

Aggregate limits are expressed in terms of the **entry ratio** :math:`r`,
which we define as the ratio

.. math::

    r = \frac{a}{\mathsf E[A(n,l)]}

of the aggregate limit to expected losses net of specific insurance.
(Per Fisher, this is the definition used by Table REF). Therefore, the aggregate
retention equals

.. math::

    a = r\mathsf E[A(n, l)] = rn\mathsf E[X_1 \wedge l].

The insurance charge

.. math::

    \phi(r):&= \frac{\mathsf E\left[A(n, l) 1_{A(n, l) > r\mathsf E[A(n,l)]}\right]}{\mathsf E[A(n,l)]} \\
    &=\frac{\mathsf E\left[A(n, l) \mid A(n, l) > r\mathsf E[A(n,l)\right] S_{(n, l)}(r\mathsf E[A(n,l)])}{\mathsf E[A(n,l)]}

where :math:`S_{(n, l)}(\cdot)` is the survival function of :math:`A(n,l)`.
The aggregate protection loss cost equals :math:`\phi(r)\mathsf E[A(n,l)]`. The insurance
savings equals

.. math::

    \psi(r):&= \frac{\mathsf E\left[A(n, l) 1_{A(n, l) \le r\mathsf E[A(n,l)]}\right]}{\mathsf E[A(n,l)]} \\
     &= \frac{\mathsf E\left[A(n, l) \mid A(n, l) \le r\mathsf E[A(n,l)\right] F_{A(n, l)}(r\mathsf E[A(n,l)])}{\mathsf E[A(n,l)]}.

where :math:`F_{(n, l)}(\cdot)` is the cdf of :math:`A(n,l)`.

With this notation, a retro program with maximum entry ratio :math:`r_1` and minimum :math:`r_0`
has a net insurance charge (ignoring expenses and the loss conversion factor) equal to

.. math::

    (\phi(r_1) - \psi(r_0)) n\mathsf E[X_1 \wedge l].

The charge and savings are illustrated below. Losses are scaled by expected
(limited) losses in the figure and so the area under the blue curve equal 1.
The graph is the Lee diagram, plotting :math:`x` against :math:`F(x)`.

.. ipython:: python
    :okwarning:

    from aggregate.extensions.figures import savings_charge
    @savefig ir_savings_exp.png scale=20
    savings_charge();

The figure makes the put-call parity relationship, savings plus 1 equals entry
plus charge obvious:

.. math::
    \psi(r) + 1 = r + \phi(r).

Remember :math:`r` is the area under the horizontal line because the width of
the plot equals 1. Taking :math:`r=1` in put-call parity shows
that :math:`\psi(1)=\phi(1)`: at expected losses, the savings equals the
charge.
