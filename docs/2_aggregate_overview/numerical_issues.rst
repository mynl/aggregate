.. _num numerical issues:

Numerical Issues
=================

This section collects artifacts that appear in output, look like bugs, and are not. Grid selection itself, how ``bs``, ``log2`` and ``x_min`` are chosen, is covered in :ref:`num bucket selection`.

.. _num reins ripple:

Ripple in a reinsurance net density
------------------------------------

Cede a share of a layer that is not a binary fraction and the net density above the attachment picks up a visible ripple.

.. ipython:: python

    import matplotlib.pyplot as plt
    from aggregate import build
    from aggregate.constants import FIG_W, FIG_H

    a = build('agg ONE dfreq[1] sev 10 * uniform occurrence net of 55.2% po 6 xs 4')
    fig, ax = plt.subplots(figsize=(2 * FIG_W, FIG_H), layout='constrained')
    a.density.loc[4:4.01, 'p_total'].plot(ax=ax, lw=2, c='C2', label='linear (default)')
    a.reins_bucket = 'nearest'
    a.update()
    a.density.loc[4:4.01, 'p_total'].plot(ax=ax, lw=.5, c='C3', label='nearest')
    ax.set(xlabel='net loss', ylabel='p_total',
           title='Net density just above the attachment')
    @savefig numerical_ripple.png scale=20
    ax.legend(loc='lower right', ncols=2);

Severity is uniform on :math:`[0,10]` and there is exactly one claim, so the net law is known in closed form: density :math:`0.1` on :math:`[0,4]`, then :math:`0.1/0.448 = 0.2232` on :math:`(4, 6.688]`. Below the attachment the computed density is flat to :math:`5\times 10^{-12}`. Above it, the ``linear`` line ripples by 4.7% peak to peak.

That is not floating-point noise, which would be eleven orders of magnitude smaller. It is deterministic aliasing. A cession moves loss off the lattice, so the net and ceded distributions have to be placed back on it, under the scheme named by :attr:`~aggregate.distributions.Aggregate.reins_bucket`. On the layer the net map has slope :math:`1 - 0.552 = 0.448`, so consecutive gross grid points land :math:`0.448\,\mathrm{bs}` apart, a stride of 0.448 buckets. The default ``'linear'`` scheme splits each off-grid value's mass between its two bracketing buckets, which is a tent kernel of half width ``bs``, and sampling a train of tents at a non-integer bucket stride :math:`d` leaves harmonics of relative size :math:`\mathrm{sinc}^2(n/d)`. Here they are 0.0090 and 0.0050, predicting a peak deviation of :math:`2(0.0090 + 0.0050) = 2.8\%` against a measured 2.9%. Both alias against the one-bucket sampling, the first to a period of 4.3 buckets, which is the wave in the plot, the second to 2.2 buckets, the wiggle riding on it. Below the attachment the map is the identity, images land exactly on grid points, and there is nothing to split.

Switching the scheme off makes it worse. ``'nearest'`` rounds instead of splitting, a box kernel rather than a tent, so its harmonics decay like :math:`\mathrm{sinc}` rather than :math:`\mathrm{sinc}^2`. It also gives up the exact first moment.

=========  ===============  ==============================
scheme     plateau ripple   computed mean (exact 4.0064)
=========  ===============  ==============================
linear     4.7e-02          4.00640000000000
nearest    4.5e-01          4.00640000104904
=========  ===============  ==============================

What removes the ripple is a net slope that is a binary fraction, so the images land on grid points. Round in decimal is not the criterion: a 60% net share has slope 0.4, which repeats in binary, and ripples more than 55.2% does.

===========  ==========  ===============
net share    net slope   plateau ripple
===========  ==========  ===============
50%          1/2         2.3e-12
75%          1/4         2.8e-13
87.5%        1/8         1.4e-13
55.2%        0.448       4.7e-02
60%          0.4         8.0e-02
===========  ==========  ===============

None of which matters, because the ripple is local and averages to zero, so it does not accumulate. Against the exact net distribution function, the largest error is 2.8e-05 under ``'linear'``, below the 5.4e-05 mass of a single bucket, and the mean is exact to 1e-14. Quantiles, TVaR, distortion prices and allocations all read the distribution function, so none of them sees it. Leave the default alone. If the density itself is the exhibit, plot the distribution function instead, or smooth over the 2.2 bucket period.

For the record, :attr:`~aggregate.distributions.Aggregate.reins_bucket` accepts ``'linear'`` (default) or ``'nearest'``, as a :func:`~aggregate.underwriter.build` keyword, as an attribute followed by ``update()``, as ``discretization.reins_bucket`` in settings, or from the ``AGGREGATE_REINS_BUCKET`` environment variable. :attr:`~aggregate.distributions.Aggregate.dsev_bucket` is the sibling that places discrete severity atoms.
