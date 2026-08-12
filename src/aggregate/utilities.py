import inspect
import itertools
import logging
import sys
from numbers import Number

import numpy as np
import pandas as pd
import re
import scipy.fft as sft

# IPython is deliberately NOT imported at module scope: `from IPython.display
# import ...` costs ~1s and utilities.py sits on the `import aggregate` path
# (distributions/portfolio/everything import it). The only consumer left is
# `agg_help`, so it is imported lazily inside that function instead. (Program
# pretty-printing/colorization moved to aggregate.decl_writer at 1.0.0a53.)

from ._grid_distribution import make_var_tvar


logger = logging.getLogger(__name__)

__all__ = [
    'ft', 'ift',
    'subsets',
    'remove_fuzz',
    'round_bucket',
    'nice_multiple',
    'qd', 'mv',
    'oep',
    'balanced_window',
    'kaplan_meier', 'kaplan_meier_np',
    'agg_help', 'explain_validation', 'introspect',
    'silence_warnings',
]


def silence_warnings(category=Warning, message='', module=''):
    """Suppress matching warnings in the current process.

    Convenience for notebook / REPL users who'd rather not see numpy / scipy
    chatter while exploring. With no arguments it silences *all* warnings;
    narrow it by passing a ``category``, a ``message`` regex, or a ``module``
    regex. For example, the benign boundary-evaluation noise from the
    distortion g-functions is all :class:`RuntimeWarning`::

        from aggregate.utilities import silence_warnings
        silence_warnings(RuntimeWarning)                  # just those
        silence_warnings(message='divide by zero')        # even narrower
        silence_warnings()                                # everything

    Parameters
    ----------
    category : type[Warning], default :class:`Warning`
        Warning class to ignore. The default :class:`Warning` matches every
        category; pass e.g. ``RuntimeWarning`` or ``DeprecationWarning`` to
        scope it.
    message : str, default ''
        Regex matched against the *start* of the warning text; '' matches all.
    module : str, default ''
        Regex matched against the issuing module's ``__name__``; '' matches all.

    Notes
    -----
    Appends an ``ignore`` rule via :func:`warnings.filterwarnings`. Unlike the
    blunter ``warnings.simplefilter('ignore')`` it does **not** discard
    existing filters. The effect is process-wide and is an explicit
    end-user-only convenience -- never call it from library code.

    This is also the opt-out for a *deliberately* clipped grid: reading
    ``E[X and a]`` off a one-claim build in a loop, say, where the missing
    far tail is beside the point. The library already reports each condition
    once per session (:func:`aggregate.constants.warn_once`), so a sweep of
    sixty-four builds costs one line, but silencing that line outright is a
    filter away::

        silence_warnings(DefectiveDistributionWarning)     # the whole class

    The inverse is :func:`aggregate.constants.reset_warn_once`, which re-arms
    every once-per-session warning for a session that has moved on to a
    different book.
    """
    import warnings
    warnings.filterwarnings('ignore', message=message, category=category, module=module)


def ft(z, padding):
    """
    fft with padding
    padding = n makes vector 2^n as long
    n=1 doubles (default)
    n=2 quadruples

    :param z:
    :param padding: = 1 doubles
    :return:
    """
    locft = sft.rfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
    # valeus per https://stackoverflow.com/questions/71706387/finding-fft-gives-keyerror-aligned-pandas
    zt = z
    if type(zt) != np.ndarray:
        zt = zt.to_numpy()
    # padding handled by the ft routine
    # temp = np.hstack((z, np.zeros_like(z)))
    return locft(zt, len(z) << padding)


def ift(z, padding):
    """
    ift that strips out padding

    :param z:
    :param padding:
    :return:
    """
    locift = sft.irfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
    if type(z) != np.ndarray:
        temp = locift(z.to_numpy())
    else:
        temp = locift(z)
    # unpad
    if padding != 0:
        temp = temp[0:len(temp) >> padding]
    return temp


def remove_fuzz(data, eps=None):
    """Zero entries with ``|x| < eps`` (machine-epsilon FFT/round-off noise).

    Vectorized replacement for the scattered de-fuzz idioms. Accepts an ndarray
    or a DataFrame; for a DataFrame only the float64 columns are touched and a
    new frame is returned (callers that need in-place semantics assign the
    result back). Two-sided: large negatives are preserved (unlike a one-sided
    ``x < eps`` clip), so it is correct on signed / P&L densities.

    Parameters
    ----------
    data : numpy.ndarray | pandas.DataFrame
        Array or frame to clean. The input is never mutated.
    eps : float, optional
        Threshold; defaults to ``np.finfo(float).eps`` (~2.22e-16).

    Returns
    -------
    numpy.ndarray | pandas.DataFrame
        Same type as ``data``, a copy with sub-eps entries set to ``0.0``.

    Notes
    -----
    The raw inverse-FFT density carries sub-machine-epsilon fuzz (tiny +/-
    values) in essentially every bucket. In a plain mass sum this cancels, but
    moment calculations weight each bucket by ``x**k``, so far-tail fuzz at
    large ``x`` is amplified and corrupts the empirical skew. Zeroing
    ``|x| < eps`` is safe and lossless: the exact aggregate has no genuine
    density below machine epsilon.

    The one-sided ``Frequency.pmf`` clip and the plot-cosmetic ``1e-15`` clip in
    the reinsurance occurrence plot are deliberate carve-outs that do NOT route
    through this helper -- they have different (one-sided / looser) semantics.
    """
    if eps is None:
        eps = np.finfo(float).eps
    if isinstance(data, pd.DataFrame):
        out = data.copy()
        float_cols = out.select_dtypes(include=['float64']).columns
        out[float_cols] = out[float_cols].mask(out[float_cols].abs() < eps, 0.0)
        return out
    return np.where(np.abs(data) < eps, 0.0, data)


#: Mantissa ladder for ``bs >= 1``: the "nice" bucket sizes within each decade.
#: Consecutive ratios (1, 2, 4, 5, 8, 10) are all <= 2, so rounding *up* to the
#: next rung overshoots the requested ``bs`` by less than 2x -- no 2.5x jumps
#: (the old 2 -> 5 / 20 -> 50 gaps). ``10`` rolls into the next decade's ``1``.
_BUCKET_LADDER = (1, 2, 4, 5, 8, 10)


def round_bucket(bs):
    """Round ``bs`` *up* to a "nice" bucket size.

    The grid step ``bs`` is chosen to **cover** the support, so this rounds up
    (never down) to the next nice value. Two regimes, both with <= 2x rung gaps
    so the overshoot is bounded below 2x:

    - ``bs >= 1`` -- the smallest member of the decade ladder
      ``{1, 2, 4, 5, 8} * 10**k`` that is ``>= bs`` (so 3.4 -> 4, not 5; 5.5 ->
      8; 9 -> 10). See :data:`_BUCKET_LADDER`.
    - ``bs < 1`` -- the smallest power of two ``>= bs`` (``..., 1/4, 1/2, 1``).
      Kept binary-exact: a sub-unit ``bs`` divides the FFT grid, and powers of
      two are exact in floating point (0.2 / 0.4 / 0.8 are not), while still
      honouring the <= 2x rule.

    There is deliberately no "round to nearest" mode. Rounding *down* would
    leave ``bs`` failing to cover the support, and it cannot tighten a grid
    anyway: the grid length is a power of two, so a power-of-two ``bs`` rounded
    down just forces a larger ``log2`` to recover coverage (more memory) or
    clips the tail.

    Parameters
    ----------
    bs : float
        Raw (un-rounded) bucket size; must be finite and non-zero.

    Returns
    -------
    float
        The rounded bucket size, ``>= bs``.

    Examples
    --------
    Test cases: ::

        test_cases = [1, 1.1, 2, 2.5, 4, 5, 5.5, 8.7, 9.9, 10, 13,
                      15, 20, 50, 100, 99, 101, 200, 250, 400, 457,
                        500, 750, 1000, 2412, 12323, 57000, 119000,
                        1e6, 1e9, 1e12, 1e15, 1e18, 1e21]
        for i in test_cases:
            print(i, round_bucket(i))
        for i in test_cases:
            print(1/i, round_bucket(1/i))
    """
    if bs == 0 or np.isinf(bs):
        raise ValueError(f'Inadmissible value passed to round_bucket, {bs}')

    if bs >= 1:
        # smallest {1,2,4,5,8}*10**k >= bs (round up within the decade)
        exp = int(np.floor(np.log10(bs)))
        base = 10.0 ** exp
        m = bs / base                       # mantissa in [1, 10)
        for rung in _BUCKET_LADDER:
            if m <= rung * (1 + 1e-9):       # tolerance for fp dust on exact rungs
                return float(rung * base)
        return 10.0 * base                   # m ~ 10 -> next decade (defensive)

    # bs < 1: smallest power of two >= bs (binary-exact small buckets). 1/bs > 1,
    # so floor it and take the largest power of two <= that via int.bit_length;
    # the reciprocal is the smallest power of two >= bs.
    n = int(1.0 / bs)
    return 1.0 / (1 << (n.bit_length() - 1))


def value_type_role(v):
    """Map a ``value_type`` token to the is-loss boolean role.

    Parameters
    ----------
    v : str
        Either a canonical token (``'loss'`` / ``'payoff'``) or the
        currently-configured label (``settings.labels``).

    Returns
    -------
    bool
        ``True`` for the loss convention, ``False`` for payoff.

    Raises
    ------
    ValueError
        If ``v`` is outside the accepted pair.

    Notes
    -----
    The role is the canonical, never-reconfigured spec field
    (``_is_loss_value``); the label strings are display/spelling settings
    (``[labels]`` in the config). Read at call time, not import time, so a
    :func:`~aggregate.config.reload_settings` is honoured.
    """
    from .config import get_settings
    labels = get_settings().labels
    if v in ('loss', labels.loss):
        return True
    if v in ('payoff', labels.payoff):
        return False
    raise ValueError(
        f"value_type must be {labels.loss!r} or {labels.payoff!r}, not {v!r}")


# Logger configuration is controlled by the user of the package, not the package itself.


def subsets(x):
    """
    all non empty subsets of x, an interable
    """
    return list(itertools.chain.from_iterable(
        itertools.combinations(x, n) for n in range(len(x) + 1)))[1:]


# new graphics methods
def nice_multiple(mx):
    """
    Suggest a nice multiple for an axis with scale 0 to mx. Used by the MultipleLocator in discrete plots,
    where you want an integer multiple. Return 0 to let matplotlib figure the answer. Real issue is stopping
    multiples like 2.5.

    :param mx:
    :return:
    """
    m = mx / 6
    if m < 0:
        return 0

    m = mx // 6
    m = {3: 2, 4: 5, 6: 5, 7: 5, 8: 10, 9: 10}.get(m, m)
    if m < 10:
        return m

    # punt back to mpl for larger values
    return 0



def qd(*argv, accuracy=3, align=True, trim=True, ff=None, **kwargs):
    """
    Generic printer for a list of aggregate-related objects.

    Dataframes handled in text with reasonable defaults. For use in documentation.

    :param: argv: list of objects to print
    :param: accuracy: number of decimal places to display
    :param: align: legacy alignment flag (currently no-op; was used by removed engineering formatter)
    :param: trim: legacy trailing-zero trim flag (currently no-op)
    :param: ff: if not None, use this function to format floats, or 'basic', or 'binary'
    :kwargs: passed to pd.DataFrame.to_string for dataframes only. e.g., pass dict of formatters by column.
    """
    from .distributions import Aggregate
    from .portfolio import Portfolio
    from ._pnl import PnL
    from .results import CalibrationResult, EvaluationResult
    if ff is None:
        ff = lambda x: f'{x:.5g}'
    elif ff == 'basic':
        ff = lambda x: f'{x:.1%}' if x < 1 else f'{x:12,.0f}'
    elif ff == 'int_ratio':
        def format_function(x):
            ir = np.round(x, 13).as_integer_ratio()
            return f'{int(x)}' if x in [0, 1] else f'  {ir[0]}/{ir[1]}'

        ff = format_function
    # split output
    for x in argv:
        if isinstance(x, (Aggregate, Portfolio)):
            # Headline risk view: a one-line text intro (which now carries the
            # validation result inline, mirroring ``_repr_html_``) and the
            # moments-and-percentiles ``summary_df``. The return-period
            # ``tail_df`` is served on demand.
            print(x._text_info_blob())
            qd(x.summary_df.fillna(''), accuracy=accuracy, **kwargs)
        elif isinstance(x, PnL):
            # P&L headline: the repr then the fixed summary card (marginal
            # range percentiles; the footing sheet is economic_df).
            print(repr(x))
            qd(x.summary_df.fillna(''), accuracy=accuracy, **kwargs)
        elif isinstance(x, CalibrationResult):
            # the receipt reads as the two frames it always was: the per
            # family shapes, then the one target they share
            qd(x.distortion_df, accuracy=accuracy, **kwargs)
            qd(x.calibration_df, accuracy=accuracy, **kwargs)
        elif isinstance(x, EvaluationResult):
            qd(x.evaluation_df, accuracy=accuracy, **kwargs)
        elif isinstance(x, pd.DataFrame):
            # 100 line width matches rtd html format
            args = {'line_width': 100,
                    'max_cols': 35,
                    'max_rows': 25,
                    'float_format': ff,
                    # needs to be larger for text output
                    # 'max_colwidth': 10,
                    'sparsify': True,
                    'justify': None
                    }
            args.update(kwargs)
            print()
            print(x.to_string(**args))
            # print(x.to_string(formatters={c: f for c in x.columns}))
        elif isinstance(x, pd.Series):
            args = {'max_rows': 25,
                    'float_format': ff,
                    'name': True
                    }
            args.update(kwargs)
            print()
            print(x.to_string(**args))
        elif isinstance(x, int):
            print(x)
        elif isinstance(x, Number):
            print(ff(x))
        else:
            print(x)


def mv(x, y=None):
    """
    Nice display of mean and variance for Aggregate or Portfolios or
    entered values.

    R style function, no return value.

    :param x: Aggregate or Portfolio or float
    :param y: float, if x is a float
    :return: None
    """
    from .distributions import Aggregate
    from .portfolio import Portfolio
    if y is None and isinstance(x, (Aggregate, Portfolio)):
        print(f'mean     = {x.actual_m:.6g}')
        print(f'variance = {x.actual_var:.7g}')
        print(f'std dev  = {x.actual_sd:.6g}')
    else:
        print(f'mean     = {x:.6g}')
        print(f'variance = {y:.7g}')
        print(f'std dev  = {y**.5:.6g}')


def balanced_window(ser, p, bs=None):
    """Equal-tail window ``[q(p/2), q(1 - p/2)]`` of a realized pmf.

    The *post-calc* analogue of :func:`aggregate._bucket_window.estimate_agg_window`:
    where that places a window from method-of-moments fits *before* the FFT (a
    guess), this measures the window directly from an already-computed marginal
    pmf. A privileged consumer -- a bivariate aggregate, which runs its inner
    marginals first, or :meth:`Aggregate.center_window` on a finished aggregate -- can
    therefore *measure* the support that matters rather than guess it.

    The window trims ``p / 2`` of the probability mass off **each** tail and
    keeps the central ``1 - p``. **Balanced** means equal *probability* each
    side, not equal value: a signed P&L marginal stays centred on its mass, and
    a skewed marginal trims more value off its heavy side but the same
    probability either way. The kept window always contains at least ``1 - p``
    of the mass (each discarded tail is at most ``p / 2``).

    Parameters
    ----------
    ser : pandas.Series
        A realized pmf: index = outcomes (``xs``), values = probabilities. The
        index must be unique and monotonic increasing and the values should sum
        to ~1 (the usual ``density_df.query('p_total > 0').p_total`` shape). The
        equal-tail accounting assumes the values are non-negative.
    p : float
        Total discarded tail mass, split equally between the two tails (``p / 2``
        each). Small, e.g. ``1e-6``; ``1 - p`` is the mass kept. Must satisfy
        ``0 < p < 1``. Unlike the coverage ``p`` of ``estimate_agg_window`` this
        is the *complementary* (discarded) mass, so there is no ``> 1 -> nines``
        reading here -- pass the literal tail mass.
    bs : float, optional
        Bucket size to snap the window to. The lower edge is floored and the
        upper edge ceiled to a multiple of ``bs`` so the window aligns with (and
        slightly contains) the grid. When ``None`` (default) the edges are
        returned exactly as the quantiles -- already grid points when ``ser``
        comes from a bucketed ``density_df``.

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper window edges. Both are **lower** quantiles
        (:func:`~aggregate._grid_distribution.make_var_tvar` ``q_lower``): ``lo = q(p/2)``,
        ``hi = q(1 - p/2)``, matching :meth:`Aggregate.q` ``kind='lower'``.

    Notes
    -----
    Reuses :func:`~aggregate._grid_distribution.make_var_tvar` for the quantiles rather than re-deriving a
    cumulative lookup, so the convention matches ``Aggregate.q`` exactly. With
    lower quantiles on both edges the mass strictly below ``lo`` is ``< p/2`` and
    the mass strictly above ``hi`` is ``<= p/2``, so the kept mass is
    ``>= 1 - p`` -- the guarantee the docstring promises.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f'p must be in (0, 1); got {p}.')
    qf = make_var_tvar(ser)
    lo = float(qf.q_lower(p / 2.0))
    hi = float(qf.q_lower(1.0 - p / 2.0))
    if bs:
        lo = np.floor(lo / bs) * bs
        hi = np.ceil(hi / bs) * bs
    return lo, hi


def oep(agg, p, *, freq=0):
    """Occurrence exceeding probability curve of a Poisson aggregate.

    Given an annual probability ``p``, return the loss ``x`` such that there is
    a probability ``p`` that one or more occurrences in a year exceed ``x``, or
    equivalently that the year's largest occurrence exceeds ``x``. OEP points
    are also called occurrence PML points.

    Parameters
    ----------
    agg : Aggregate
        The aggregate supplying the severity and, unless ``freq`` overrides it,
        the expected claim count. Its frequency must be Poisson, and not zero
        modified. Need not have been updated: only the input severity and the
        claim count are read, never ``density_df``.
    p : float or array_like of float
        Annual probability level(s), each in ``(0, 1)`` and strictly below the
        ceiling ``1 - exp(-lam)``. Typically small, 0.01 or less. Order is
        preserved, not sorted.
    freq : float, default 0
        Expected annual claim count to use in place of ``agg.n``. The default
        ``0`` means use ``agg.n``; any positive value overrides it. Negative
        and non-finite values raise. Supplying ``freq`` rescales the curve but
        does not waive the Poisson requirement.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``p``, with columns

        ``loss``
            The occurrence PML ``x``, exact, not snapped to the ``bs`` grid.
        ``S_sev``, ``F_sev``
            ``Pr(L > x)`` and the severity percentile ``Pr(L <= x)`` of that
            loss, read back from ``loss``. ``F_sev`` is the level you would
            hand to :meth:`Aggregate.q_sev`; ``S_sev`` is the usable handle far
            out in the tail, where ``F_sev`` rounds to 1.
        ``oep``
            The achieved annual probability. Equals the index ``p`` to machine
            precision for a continuous severity.
        ``occurrence_return_period``
            ``1 / (lam * S_sev)``, the average gap between occurrences
            exceeding ``x``. Can be shorter than a year.
        ``annual_return_period``
            ``1 / oep``, the average gap between years containing such an
            occurrence. Always at least 1.

    Raises
    ------
    TypeError
        If ``agg`` is not an :class:`Aggregate`.
    ValueError
        If the frequency is not Poisson or is zero modified, if ``freq`` is
        negative or non-finite, or if any ``p`` falls outside ``(0, 1)`` or
        above the ceiling.

    See Also
    --------
    Aggregate.sev : the exact severity functions this is built on.
    Aggregate.q_sev : the grid-snapped severity quantile.

    Notes
    -----
    **Derivation.** Fix a threshold :math:`x` and keep only the occurrences
    exceeding it. Thinning a Poisson stream leaves a Poisson stream, so those
    occurrences are Poisson with rate :math:`\\lambda\\,P(L>x)`. The chance a
    year contains at least one is one minus the chance it contains none,

    .. math:: \\mathrm{OEP}(x) = 1 - e^{-\\lambda P(L>x)}.

    Inverting for :math:`x` given :math:`p` gives what the reference below
    calls the exceedance quantile :math:`\\mathrm{EQ}`,

    .. math:: x = S_L^{-1}\\left(\\frac{-\\ln(1-p)}{\\lambda}\\right)
              = q_L\\left(1 + \\frac{\\ln(1-p)}{\\lambda}\\right),

    which is the formula in the catastrophe modeling user guide. This function
    is named for the forward map because that is the standing industry usage:
    an "OEP point" is a loss.

    **The two forms are not numerically equal.** ``1 + ln(1-p)/lam`` cancels
    toward 1 as ``lam`` grows, losing digits before the quantile is even
    evaluated. At ``lam = 100`` and ``p = 1e-4`` the two routes already differ
    by 6e-12 relative. This function computes ``s = -log1p(-p) / lam`` and
    calls ``isf(s)``, which is accurate throughout.

    **The ceiling.** Solving requires ``-ln(1-p)/lam <= 1``, that is
    ``p <= 1 - exp(-lam)``, the probability of one or more occurrences. Above
    it no solution exists: a year with no occurrence has no largest loss, so no
    threshold is exceeded. At ``lam = 2`` the ceiling is 0.8646647. The ceiling
    itself is excluded as well, since there the only answer is the infimum of
    the severity support, which every threshold is exceeded above.

    **Two return periods, one threshold.** They are not the same number and the
    difference is the usual source of confusion. The occurrence return period
    counts occurrences and can dip below a year; the annual return period
    counts the years that contain one and is at least 1. They agree when ``x``
    is large and diverge as ``x`` falls.

    **Exact, not bucketed.** The loss comes from :meth:`Aggregate.sev`, the
    continuous input severity, rather than :meth:`Aggregate.q_sev`, which snaps
    to the ``bs`` lattice. Where the severity has an atom or a gap in support,
    for instance a limited severity whose quantile lands on the limit for a
    whole range of ``s``, the achieved ``oep`` column falls below the requested
    ``p``. That is reported rather than hidden.

    References
    ----------
    Mildenhall, S. J., *Return Period Confusions Clarified*, 2026.
    https://blog.mynl.com/posts/notes/2026-07-16-Return-Period-Confusions-Clarified/

    Examples
    --------
    ::

        from aggregate import build, qd, oep
        a = build('agg Cat 2 claims sev lognorm 1648.7212707 cv 1.3108324 poisson')
        qd(oep(a, [0.001, 0.01, 0.05, 0.5]))

    gives losses 26853.227, 13119.408, 7021.788 and 1483.772.
    """
    from ._aggregate import Aggregate

    if not isinstance(agg, Aggregate):
        raise TypeError(f'oep requires an Aggregate; got {type(agg).__name__}.')

    fname = getattr(agg.frequency, 'freq_name', '')
    if fname != 'poisson':
        raise ValueError(f'oep requires Poisson frequency; {agg.name} has '
                         f'freq_name={fname!r}.')
    if getattr(agg.frequency, 'freq_zm', False):
        raise ValueError(f'oep requires plain Poisson frequency; {agg.name} is zero '
                         'modified, which breaks the 1 - exp(-lam S) derivation.')

    freq = float(freq)
    if freq < 0 or not np.isfinite(freq):
        raise ValueError(f'freq must be a non-negative finite float, 0 to use agg.n; '
                         f'got {freq}.')
    lam = freq if freq > 0 else float(agg.n)
    if lam <= 0:
        raise ValueError(f'oep needs a positive expected claim count; {agg.name} has '
                         f'n={agg.n}. Pass freq= to supply one.')

    p = np.atleast_1d(np.asarray(p, dtype=float))
    bad = p[~((p > 0.0) & (p < 1.0))]
    if bad.size:
        raise ValueError(f'p must be in (0, 1) exclusive; got {bad.tolist()}.')
    # S(x) < 1 caps the attainable annual probability below Pr(N >= 1)
    ceiling = -np.expm1(-lam)
    bad = p[p >= ceiling]
    if bad.size:
        raise ValueError(f'p must be below the ceiling 1 - exp(-lam) = {ceiling:.7g} '
                         f'for lam = {lam:g}; got {bad.tolist()}. A year with no '
                         'occurrence has no largest loss, so nothing is exceeded.')

    # severity exceedance probability implied by p; log1p keeps small p exact
    s = -np.log1p(-p) / lam
    loss = np.asarray(agg.sev.isf(s), dtype=float)
    # read S back from the loss rather than reusing s, so an atom or a gap in
    # the severity support shows up as an achieved oep below the requested p
    s_sev = np.asarray(agg.sev.sf(loss), dtype=float)
    achieved = -np.expm1(-lam * s_sev)
    with np.errstate(divide='ignore', invalid='ignore'):
        occ_rp = 1.0 / (lam * s_sev)
        ann_rp = 1.0 / achieved
    return pd.DataFrame({
        'loss': loss,
        'S_sev': s_sev,
        'F_sev': 1.0 - s_sev,
        'oep': achieved,
        'occurrence_return_period': occ_rp,
        'annual_return_period': ann_rp,
    }, index=pd.Index(p, name='p'))


def kaplan_meier(df, loss='loss', closed='closed'):
    """
    Compute Kaplan Meier Product limit estimator based on a sample
    of losses in the dataframe df. For each loss you know the current
    evaluation in column ``loss`` and a 0/1 indicator for open/closed
    in ``closed``.

    The output dataframe has columns

    * index x_i, size of loss
    * open - the number of open events of size x_i (open claim with this size)
    * closed - the number closed at size x_i
    * events - total number of events of size x_i
    * n - number at risk at x_i
    * s - probability of suriviving past x_i = 1 - closed / n
    * pl - cumulative probability of surviving past x_i

    See ipython workbook kaplan_meier.ipynb for a check against lifelines
    and some kaggle data (telco customer churn,
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
    https://towardsdatascience.com/introduction-to-survival-analysis-the-kaplan-meier-estimator-94ec5812a97a

    :param df: dataframe of data
    :param loss: column containing loss amount data
    :param closed: column indicating if the obervation is a closed claim (1) or open (0)
    :return: dataframe as described above
    """

    df = df[[loss, closed]].rename(columns={loss: 'loss', closed: 'closed'}).copy()
    df['open'] = 1 - df.closed
    df = df.sort_values(['loss', 'closed'], ascending=[False, True]).reset_index(drop=True)

    df = df.groupby(['loss', 'closed']).count()
    # c has index loss amount and closed indicator, and column number of observations
    c = df.unstack(1)
    # total number of observables at each loss event size
    c['t'] = c.sum(1)
    # total number at risk at each event size
    c['n'] = c.t[::-1].cumsum()
    # better column names
    c.columns = ['open', 'closed', 'events', 'n']
    #
    c = c.fillna(0)
    # prob of surviving past each observed event size
    c['s'] = 1 -  c.closed / c.n
    # KM product estimator
    c['pl'] = c.s.cumprod()
    return c


def kaplan_meier_np(loss, closed):
    """
    Feeder to kaplan_meier where loss is np array  of loss amounts and
    closed a same sized array of 0=open, 1=closed indicators.
    """
    df = pd.DataFrame({'loss': loss, 'closed': closed})
    return kaplan_meier(df)


def _in_jupyter():
    """True iff running under a Jupyter (ZMQ) kernel, without importing IPython.

    Probes ``sys.modules`` rather than importing IPython: the import costs ~1s
    and ``utilities`` sits on the ``import aggregate`` path, so we never force it.
    If IPython was never imported we are certainly not in a notebook. A live
    ``ZMQInteractiveShell`` is notebook / lab / qtconsole (all render ANSI);
    ``TerminalInteractiveShell`` or ``None`` is not Jupyter.
    """
    mod = sys.modules.get('IPython')
    if mod is None:
        return False
    shell = mod.get_ipython()
    return shell is not None and type(shell).__name__ == 'ZMQInteractiveShell'


def _help_target(fmt):
    """Resolve the :func:`agg_help` render target to ``'text'``, ``'ansi'`` or ``'html'``.

    ``auto`` (the default) picks ``ansi`` under a Jupyter kernel and ``text``
    otherwise. It deliberately never resolves to ``html`` -- ANSI colors with a
    consistent monospace font are preferred even in JupyterLab; ``html`` is
    reachable only by asking for it explicitly. An explicit ``fmt`` always wins.
    """
    if fmt not in ('auto', 'text', 'ansi', 'html'):
        raise ValueError(
            f"fmt must be 'auto', 'text', 'ansi', or 'html'; got {fmt!r}")
    if fmt != 'auto':
        return fmt
    return 'ansi' if _in_jupyter() else 'text'


def agg_help(self, regex='.*', lod='terse', values='none', private=False, fmt='auto'):
    """
    Investigate ``self`` for public names matching ``regex`` and display each
    one's documentation and (optionally) its value or no-argument call result.

    Module-level free function backing the ``.help(regex, ...)`` method on
    :class:`Aggregate`, :class:`Portfolio`, :class:`Underwriter`,
    :class:`Severity`, :class:`Distortion`, :class:`Bounds`, :class:`PnL`, and
    the bivariate classes. Named ``agg_help`` (not ``help``) to avoid shadowing
    Python's builtin ``help`` at module / package scope. Fka ``more``.

    Parameters
    ----------
    self : object
        The instance to introspect.
    regex : str, default '.*'
        Regular expression; names matching it (via :func:`re.search`) are shown.
        The default matches every name, so a bare call lists the whole surface.
    lod : {'terse', 'short', 'all'}, default 'terse'
        Level of *documentation* detail per match:

        * ``'terse'`` -- name (and method signature) only, no docstring;
        * ``'short'`` -- the first few lines of the docstring;
        * ``'all'`` -- the full docstring.
    values : {'none', 'short', 'all'}, default 'none'
        How much of each name's *value* to display -- the attribute value, or a
        method's no-argument call result (methods needing arguments are
        skipped):

        * ``'none'`` -- show no values (names + docstrings only);
        * ``'short'`` -- show values, but a :class:`pandas.DataFrame` or
          :class:`pandas.Series` is truncated to ``.head(5)``;
        * ``'all'`` -- show values in full.
    private : bool, default False
        When ``False`` (the default) names beginning with an underscore
        (``_private`` and ``__dunder__``) are skipped -- the public surface
        only. Pass ``True`` to include them.
    fmt : {'auto', 'text', 'ansi', 'html'}, default 'auto'
        Render *target*:

        * ``'auto'`` -- ``ansi`` under a Jupyter kernel, ``text`` in a plain
          terminal / REPL (never ``html`` -- ask for it explicitly);
        * ``'text'`` -- plain ``print`` (no color, no IPython import);
        * ``'ansi'`` -- ``text`` with the header line colorized via ANSI escapes
          (renders as color in JupyterLab and in a color terminal);
        * ``'html'`` -- the rich Markdown / IPython display path (Jupyter only).

    Notes
    -----
    ``lod``, ``values`` and ``fmt`` are three orthogonal axes: ``lod`` governs
    the docstring detail, ``values`` how much of the value / call result, and
    ``fmt`` the render target. The default ``lod='terse', values='none'`` is a
    bare public-name listing (``private=False``). Documentation is shown for
    methods and properties only (a plain field carries no useful docstring).

    Two filters, and only two, decide which names appear: ``regex`` (default
    ``'.*'``, i.e. everything) and the leading-underscore skip governed by
    ``private``. Nothing else is excluded, so the listing spans inherited names
    too: on :class:`Severity`, whose base is ``scipy.stats.rv_continuous``, a
    bare call reports the inherited ``cdf`` / ``ppf`` / ``rvs`` family alongside
    the ``aggregate`` surface.
    """
    if lod not in ('terse', 'short', 'all'):
        raise ValueError(f"lod must be 'terse', 'short', or 'all'; got {lod!r}")
    if values not in ('none', 'short', 'all'):
        raise ValueError(
            f"values must be 'none', 'short', or 'all'; got {values!r}")
    target = _help_target(fmt)   # validates fmt; raises ValueError on a bad value

    short_doc_lines = 4   # 'short' lod: first few lines of the docstring
    head_n = 5            # 'short' values: rows of a DataFrame/Series to show

    # Select the three emit primitives once, before the walk, so the branch on
    # render target lives in one place rather than at every call site.
    if target == 'html':
        # IPython imported lazily to keep it off the `import aggregate` path
        # (it is ~1s to import); see module note above. Only the html path needs
        # it -- text / ansi stay dependency-free.
        from IPython.display import Markdown, display

        def emit_header(is_method, name, sig):
            tag = 'Callable' if is_method else 'Attribute'
            display(Markdown(f'### {tag}: {name}{sig}\n'))

        def emit_doc(doc):
            display(Markdown(doc))

        def emit_value(value):
            display(value)

        def emit_error(name, msg):
            display(Markdown(f'### Error: {name}\n'))
            print(msg)
    else:
        # text / ansi both print plain; ansi colorizes only the header line, with
        # a tiny set of escape constants (no Markdown->ANSI engine, no new dep).
        bold, dim, accent, reset = (
            ('\x1b[1m', '\x1b[2m', '\x1b[36m', '\x1b[0m')
            if target == 'ansi' else ('', '', '', ''))

        def emit_header(is_method, name, sig):
            tag = 'Callable' if is_method else 'Attribute'
            print(f'{dim}{tag}:{reset} {bold}{accent}{name}{reset}{sig}')

        def emit_doc(doc):
            print(doc)

        def emit_value(value):
            print(value)

        def emit_error(name, msg):
            print(f'Error: {name}')
            print(msg)

    for name in dir(self):
        if not private and name.startswith('_'):
            continue
        if not re.search(regex, name):
            continue
        # classify off the *class* so a raising property does not abort the walk
        class_attr = getattr(type(self), name, None)
        is_property = isinstance(class_attr, property)
        try:
            ob = getattr(self, name)
        except Exception as e:  # noqa: BLE001 - report, don't propagate
            emit_error(name, f'{type(e).__name__}: {e}')
            continue

        is_method = callable(ob) and not is_property

        # header, with the bound signature for methods
        sig = ''
        if is_method:
            try:
                sig = str(inspect.signature(ob))
            except (TypeError, ValueError):
                pass
        emit_header(is_method, name, sig)

        # documentation (methods and properties only), governed by lod
        if lod != 'terse' and (is_method or is_property):
            doc = inspect.getdoc(ob if is_method else class_attr) or ''
            if doc:
                if lod == 'short':
                    doc = '\n'.join(doc.split('\n')[:short_doc_lines])
                emit_doc(doc)

        # value / no-arg call result, governed by values
        if values != 'none':
            if is_method:
                try:
                    value = ob()
                except Exception:  # noqa: BLE001 - needs args or has side effects
                    continue
            else:
                value = ob
            if values == 'short' and isinstance(value, (pd.DataFrame, pd.Series)):
                emit_value(value.head(head_n))
            else:
                emit_value(value)


def introspect(ob):
    """
    Discover the non-private methods and properties of an object ``ob``.

    Used to build the class cheat sheets: it walks ``dir(ob)``, classifies each
    public name as a method, property, or plain field, and records its type,
    value, call signature, and docstring. The result is a :class:`pandas.DataFrame`
    sorted by classification then name, ready to export and arrange by category.

    Parameters
    ----------
    ob : object
        Any instance. Typically an :class:`Underwriter`, :class:`Aggregate`,
        :class:`Portfolio`, :class:`Severity`, or :class:`Distortion`.

    Returns
    -------
    pandas.DataFrame
        One row per public name with columns ``name``, ``kind`` (``method``,
        ``property``, ``field``, or ``error``), ``value``, ``type``,
        ``signature``, ``help`` (first docstring line), and ``length``.

    Notes
    -----
    Accessing a property can raise — e.g. a :class:`Portfolio` property that
    assumes the object has been updated. A bare ``getattr`` in the loop would
    then abort the whole introspection (the historical reason this function
    "would not run" on a Portfolio). Each access is therefore guarded: a name
    whose access raises is reported with ``kind='error'`` and the exception text
    in ``value``, so introspection always completes and surfaces the offender
    instead of dying on it. Classification (property vs. field) is read from the
    *class* via ``getattr(type(ob), name)`` so it does not depend on the instance
    access succeeding.
    """
    names = [i for i in dir(ob) if i[0] != '_']
    rows = []
    for name in names:
        class_attr = getattr(type(ob), name, None)
        is_property = isinstance(class_attr, property)
        # guarded access: a raising property must not abort the whole walk
        try:
            g = getattr(ob, name)
        except Exception as e:  # noqa: BLE001 - report, don't propagate
            rows.append([name, 'error', f'{type(e).__name__}: {e}', '', '', '', 0])
            continue

        value = ''
        type_str = ''
        signature = ''
        help_str = ''
        length = 0
        if callable(g) and not is_property:
            kind = 'method'
            try:
                signature = str(inspect.signature(g))
            except (TypeError, ValueError):
                signature = ''
            help_str = (g.__doc__ or '').strip().split('\n')[0]
        else:
            kind = 'property' if is_property else 'field'
            value = str(g)
            type_str = str(type(g))
            try:
                length = len(g)
            except TypeError:
                length = 0

        rows.append([name, kind, value, type_str, signature, help_str, length])

    df = pd.DataFrame(
        rows,
        columns=['name', 'kind', 'value', 'type', 'signature', 'help', 'length'])
    df = df.sort_values(['kind', 'length', 'name']).reset_index(drop=True)
    return df


# explain_validation relocated to _validation.py (Phase 1b); re-exported
# here for back-compat (mirrors make_var_tvar in _grid_distribution).
from ._validation import explain_validation  # noqa: F401

    
