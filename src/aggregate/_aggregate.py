"""The Aggregate compound-distribution class and its support functions.

Extracted from ``distributions.py`` (Phase 1, kind split). Imported through the
``distributions`` facade so every existing import path keeps working.
"""

from collections import namedtuple
from collections.abc import Iterable
import hashlib
import json
import inspect
import logging
import math
import warnings
import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.fft as sfft
from scipy.integrate import quad
import scipy.stats as ss
from scipy import interpolate
from scipy.optimize import NoConvergence  # noqa
from textwrap import fill
from .constants import (DefectiveDistributionWarning,
                        INFO_NA, info_row,
                        InfiniteVarianceError,
                        REINS_LABEL_GROSS, REINS_LABEL_NET,
                        REINS_LABEL_CEDED, REINS_LABEL_OUTPUT,
                        Validation)
from .config import get_settings
from .moments import (MomentAggregator, MomentWrangler,
                      xsden_to_mwrangler,
                      xsden_to_meancv, xsden_to_meancvskew,
                      _noise_aware_rel_error, _snap_noise)
from .utilities import (ft, ift,
                        round_bucket,
                        balanced_window,
                        agg_help, remove_fuzz, value_type_role)
from ._grid_distribution import GridDistribution, return_period_map, period_to_p
from ._labeled import LabeledMixin
from .decl_writer import format_program, spec_to_decl
import aggregate.random_agg as ar
from .spectral import choquet_weights
from . import tail as _tail
from .tail import TailClass

from ._fits import (_approximate_sev_kwargs, approximate_from_mcvsk)
from ._frequency import Frequency
from ._severity import Severity
# Phase 1b shared concerns (leaf/near-leaf; never import back into _aggregate).
from ._bucket_window import (
    WINDOW_NINES, WINDOW_LOG2_GROWTH, WINDOW_NINES_TRIM, WINDOW_PAD_SKEW,
    WINDOW_SLACK_THICK, BUCKET_SIZING_P, SBJ_TAIL_FLOOR,
    _estimate_agg_percentile, estimate_agg_window, bs_describe, bs_explain,
)
from . import _bucket_window
from ._validation import VALIDATION_NOISE, ALIASING_RATIO, explain_validation
from . import _validation
from . import _reinsurance
from ._aggregate_compute import freq_sev_convolution
from . import _pricing

logger = logging.getLogger(__name__)

__all__ = [
    'Aggregate',
]

#: Default return-period ladder for the summary ``tail_df`` (overridable via
#: ``tail_df(periods=...)``). The Solvency II ``1-in-200`` (99.5%) and US
#: capital-adequacy / rating ``1-in-250`` (99.6%) anchors are both included and
#: highlighted in the HTML rendering.
DEFAULT_RETURN_PERIODS = (2, 5, 10, 25, 50, 100, 200, 250, 500, 1000)

#: Key percentiles carried by the summary ``summary_df`` (low / median / high).
SUMMARY_PERCENTILES = (0.01, 0.50, 0.99)

#: Relative floor below which ``CV = SD / E[X]`` is left blank in ``summary_df``:
#: the mean is treated as indistinguishable from zero when ``|E[X]| < tol * SD``
#: (a signed / near-break-even position), where ``CV`` is meaningless. ``SD`` is
#: always reported.
CV_MEAN_REL_TOL = 1e-3


def value_type_label(is_loss_value):
    """Render the is-loss boolean role as the configured label string.

    Parameters
    ----------
    is_loss_value : bool
        ``True`` for the loss convention, ``False`` for payoff.

    Returns
    -------
    str
        ``settings.labels.loss`` or ``settings.labels.payoff``.
    """
    labels = get_settings().labels
    return labels.loss if is_loss_value else labels.payoff


def return_period_frame(q, tvar, mean, is_loss_value, periods=None):
    """Build a return-period / exceedance table from quantile and TVaR functions.

    Shared by :meth:`Aggregate.tail_df` and :meth:`Portfolio.tail_df` so the
    aggregate and portfolio-total tables read one implementation.

    Parameters
    ----------
    q, tvar : callable
        ``q(p)`` (VaR) and ``tvar(p)`` (TVaR) at a non-exceedance probability.
    mean : float
        ``E[X]`` -- the leverage denominator and ``xsVaR`` reference.
    is_loss_value : bool
        Orientation passed to :func:`period_to_p` (loss -> upper tail, payoff
        -> downside).
    periods : array_like of float, optional
        Return-period ladder. Defaults to :data:`DEFAULT_RETURN_PERIODS`.

    Returns
    -------
    pandas.DataFrame
        Indexed by return period ``T``; columns ``p | VaR | TVaR | xsVaR |
        VaR/Mean``; ``E[X]`` carried in ``.attrs['mean']``.
    """
    periods = DEFAULT_RETURN_PERIODS if periods is None else periods
    T = np.atleast_1d(np.asarray(periods, dtype=float))
    p = np.atleast_1d(period_to_p(T, is_loss_value))
    mean = float(mean)
    var = np.array([q(float(pi)) for pi in p], dtype=float)
    tv = np.array([tvar(float(pi)) for pi in p], dtype=float)
    leverage = (var / mean if abs(mean) > VALIDATION_NOISE
                else np.full_like(var, np.nan))
    # Integer return periods read cleanly as ``200`` not ``200.0``.
    idx = pd.Index([int(t) if float(t).is_integer() else t for t in T], name='T')
    df = pd.DataFrame(
        {
            'p': p,
            'VaR': var,
            'TVaR': tv,
            'xsVaR': var - mean,
            'VaR/Mean': leverage,
        },
        index=idx,
    )
    df.attrs['mean'] = mean
    return df


def max_log2(x):
    """
    Return the largest power of two d so that (x + 2**-d) - x == 2**-d, with d <= 30.
    Used in dhistogram severity types to determine the size of the step.
    """
    d = min(30, -np.log2(np.finfo(float).eps) - np.ceil(np.log2(x)) - 1)
    if (x + 2 ** -d) - x != 2 ** -d:
        raise ValueError('max_log2 failed')
    return d


def _picks_work(attachments, layer_loss_picks, xs, sev_density, n=1, sf=None, debug=False):
    """
    Adjust the layer unconditional expected losses to target. You need int xf(x)dx, but
    that is fraught when f is a mixed distribution. So we only use the int S version.
    ``fz`` was initially a frozen continuous distribution; but adjusted to sf function
    and dropped need for pdf function.

    See notes for how the parts are defined. Notice that::

        np.allclose(p.layers.v - p.layers.f, p.layers.l - p.layers.e)

    is true.

    :param attachments: array of layer attachment points, in ascending order (bottom to top). a[0]>0
    :param layer_loss_picks: Target means. If ``len(layer_loss_picks)==len(attachments)`` then the bottom layer, 0 to a[0],
      is added. Can be input as unconditional layer severity (i.e., :math:`\\mathbb{E}[(X-a)^+\\wedge y]`) or as the
      layer loss pick (i.e., :math:`\\mathbb{E}[(X-a)^+\\wedge y]'times n` where *n* is the number of ground-up (to the
      insurer) claims. Multiplying and dividing by :math:`S(a)` shows this equals conditional severity in the layer
      times the number of claims in the layer.) Actuaries usually estimate the loss pick to the layer in pricing. When
      called from :class:`Aggregate` the number of ground up claims is known.
    :param en: ground-up expected claims. Target is divided by ``en``.
    :param xs: x values for discretization
    :param sev_density: Series of existing severity density from Aggregate.
    :param sf: cdf function for the severity distribution.
    :param debug: if True, return debug information (layers, density with adjusted probs, audit
      of layer expected values.
    """

    # want xs, attachments, and sev_density to be numpy arrays
    xs = np.array(xs)
    attachments = np.array(attachments)
    # target is the unconditional layer expected loss, E[(X-a)^+ ^ y]
    target = np.array(layer_loss_picks) / n
    # print(n, layer_loss_picks, target)
    sev_density = np.array(sev_density)
    # figure bucket size
    bs = xs[1] - xs[0]

    # dataframe of adjusted probabilties, starts here
    density = pd.DataFrame({'x': xs, 'p': sev_density}).set_index('x', drop=False)
    fill_value = max(0, 1. - density.p.sum())
    density['S'] = density.p.shift(-1, fill_value=fill_value)[::-1].cumsum()

    # numerical integrals - these match
    layers = pd.DataFrame(columns=['a', 'lev', 'int_fdx', 'aS', 'S'], index=range(1, 1+len(attachments)),
                          dtype=float)
    for i, x in enumerate(attachments):
        ix = density.loc[0:x-bs, 'S'].sum() * bs
        ix2 = density.loc[0:x-bs, ['x', 'p']].prod(axis=1).sum()
        layers.loc[i+1, :] = [x, ix, ix2, x * density.loc[x, 'S'] if x < np.inf else 0.0, density.loc[x, 'S']]

    # prob of loss in layer
    layers['p'] = layers.S.shift(1, fill_value=1) - layers.S
    # unconditional expected loss in layer
    layers['l'] = layers.lev - layers.lev.shift(1, fill_value=0)
    layers.index.name = 'layer'
    # bottom of layer
    layers['a_bottom'] = layers.a.shift(1, fill_value=0)
    # width of layer
    layers['y'] = layers.a - layers.a_bottom
    # e = rectangle to right in int S computation
    layers['e'] = layers.S * layers.y
    # f = rectangle below attachment in int xf computation
    layers['f'] = layers.p * layers.a_bottom
    # these are two versions of m (unconditional)
    # m-bit: int S - e == int xf - f
    layers['m'] = layers.l - layers.e
    # int f dx in layer
    layers['v'] = layers.f + layers.m
    # and conditional vertical loss in layer
    layers['v_c'] = layers.v / layers.p
    layers = layers[['a_bottom', 'a', 'y', 'lev', 'S', 'p', 'l', 'v', 'v_c', 'm', 'e', 'f']]

    # add weights w and offsets=omega, computed from the top layer down
    layers['t'] = target
    layers['w'] = 0.0
    layers['ω'] = 0.0

    # this computation leaves the tail unchanged and uses the same "adjust the curve" method
    # in all layers
    ω = layers.loc[len(layers), 'S']
    for i in layers.index[::-1]:
        layers.loc[i, 'w'] = (layers.loc[i, 't'] - ω * layers.loc[i, 'y']) / layers.loc[i, 'm']
        layers.loc[i, 'ω'] = ω
        ω += layers.loc[i, 'p'] * layers.loc[i, 'w']

    # adjusted S: bins -> layer number; add in offsets
    density['bin'] = pd.cut(density.x, np.hstack((0, layers.a.values)), include_lowest=True, right=True)
    # layer description returned by cut to layer number in layers
    mapper = {i:j+1 for j, i in enumerate(density.bin.unique())}
    density['layer'] = density.bin.map(mapper.get)

    density['ω'] = density.layer.map(layers.ω).astype(float)
    # S(a_n-1)
    density['Sa'] = density.layer.map(layers.S).astype(float)
    density['w'] = density.layer.map(layers.w).astype(float)

    density['S_adj'] = np.minimum(1, density.ω + (density.S - density.Sa) * density.w)
    # no change in the tail
    density.loc[attachments[-1]:, 'S_adj'] = density.loc[attachments[-1]:, 'S']
    # adj probs as difference of S
    density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
    achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
    # display(achieved)
    if abs(achieved.iloc[0] - target[0]) > 1e-3:
        # issues with hitting 1
        logger.warning(f'achieved[0] = {achieved.iloc[0]} != target[0] = {target[0]}')
        # take top right corner off
        if target[0] > attachments[0]:
            raise ValueError(f'target[0] = {target[0]} > first attachment[0] = {attachments[0]} which is impossible.')
        s0 = 2 * (attachments[0] - target[0]) / (1 - layers.loc[1, 'ω'])
        # snap to index
        s0 = bs * np.round(s0 / bs, 0)
        # convert to probability
        s = attachments[0] - s0
        density.loc[0:s, 'S_adj'] = 1.0
        temp = np.array(density.loc[s+bs:attachments[0]].index)
        wts = (temp - s) / s0
        density.loc[s+bs:attachments[0], 'S_adj'] = 1 - wts + layers.loc[1, 'ω'] * wts
        # update
        density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
        achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
        logger.warning(f'Revised layer 1 achieved = {achieved.iloc[0]}')

    density['diff S'] = density['S'] - density['Sa']

    if debug is False:
        return density['p_adj'].values

    # data frame of layer statistics from input density
    exact = None
    if sf is not None:
        logger.warning('sf passed in; computing exact layer statistics')
        exact = pd.DataFrame(columns=['a', 'lev', 'aS', 'S'],
                             index=range(1, 1+len(attachments)), dtype=float)
        for i, x in enumerate(attachments):
            ix = quad(sf, 0, x)
            # check error is small
            assert ix[1] < 1e-6
            sf_ = sf(x)
            exact.loc[i+1, :] = [x, ix[0], x * sf_ if x < np.inf else 0.0, sf_]

    if exact is None:
        l = layers.l
        ln = 'layers'
    else:
        l = exact.lev - exact.lev.shift(1, fill_value=0)
        ln = 'exact'

    t = pd.concat((l,
                   density.groupby(density.layer.shift(-1)).apply(lambda g: g['S'].sum() * bs),
                   achieved,
                   ), keys=[ln, 'computed', 'adj'], axis=1)
    t.loc['sum'] = t.sum()
    Picks = namedtuple('picks', ['layers', 'exact', 'density', 'audit'])
    return Picks(layers=layers, exact=exact, density=density, audit=t)


def _integral_by_doubling(func, x0, err=1e-8):
    r"""
    Compute :math:`\int_{x_0}^\infty f` as the sum

    .. math::

        \int_{x_0}^\infty f = \sum_{n \ge 0} \int_{2^nx_0}^{2^{n+1}x_0} f

    Caller should check the integral actually converges.

    :param func: function to be integrated.
    :param x0: starting x value
    :param err: desired accuracy: stop when incremental integral is <= err.
    """
    ans = 0.
    counter = 0
    # from to
    f, t = x0, 2 * x0
    last_int = 10
    while last_int > err:
        s = quad(func, f, t)
        if s[1] > err:
            raise ValueError(
                f'Questionable integral numeric convergence, err {s[1]:.4g}\n'
                f'f={f}, t={t}, x0={x0}, counter={counter}')
        last_int = s[0]
        ans += s[0]
        f, t = t, 2 * t
        counter += 1
        if counter > 96:
            raise ValueError(f'counter = {counter} and error = {err}')
    return -ans


# ---------------------------------------------------------------------------
# Stats DataFrame helpers
# ---------------------------------------------------------------------------
# ``Aggregate.stats_df`` is the canonical (component, measure) × view
# DataFrame holding theoretical + empirical moments. ``MomentAggregator``
# emits its per-component / totals statistics as flat names like ``freq_1``,
# ``agg_m``; ``_flat_col_to_stats_index`` maps each to the
# ``(component, measure)`` tuple used by the ``stats_df`` row MultiIndex.

_STATS_META_NAMES = frozenset({
    'name', 'limit', 'attachment', 'el', 'prem', 'lr', 'sevcv_param',
    'mix_cv', 'wt',
})


_STATS_MEASURE_MAP = {'1': 'ex1', '2': 'ex2', '3': 'ex3', 'm': 'mean'}


def _flat_col_to_stats_index(col):
    """Map a flat ``MomentAggregator`` moment name to ``(component, measure)``.

    Examples: ``'freq_1' → ('freq', 'ex1')``, ``'agg_m' → ('agg', 'mean')``,
    ``'limit' → ('meta', 'limit')``.

    Used to bridge the flat moment names emitted by
    :meth:`MomentAggregator.get_fsa_stats` / ``column_names()`` to the
    canonical ``(component, measure)`` MultiIndex used by ``stats_df``.
    """
    if col in _STATS_META_NAMES:
        return ('meta', col)
    comp, _, measure = col.partition('_')
    if comp in ('freq', 'sev', 'agg'):
        return (comp, _STATS_MEASURE_MAP.get(measure, measure))
    raise ValueError(f'Cannot map column {col!r} for stats_df build.')


# Canonical row MultiIndex for ``stats_df`` — written directly in __init__
# (component columns) and the post-loop totals block (``mixed`` /
# ``independent``). All ``meta`` rows up top, then freq/sev/agg moment blocks.
# The frame is all-float: ``self.name`` is already an attribute, no need for
# a ``('meta','name')`` string row that would force ``dtype=object``.
_STATS_ROW_INDEX = pd.MultiIndex.from_tuples(
    [
        ('meta', 'limit'), ('meta', 'attachment'),
        ('meta', 'el'), ('meta', 'prem'), ('meta', 'lr'),
        ('meta', 'sevcv_param'), ('meta', 'mix_cv'), ('meta', 'wt'),
        ('freq', 'ex1'), ('freq', 'ex2'), ('freq', 'ex3'),
        ('freq', 'mean'), ('freq', 'cv'), ('freq', 'skew'),
        ('sev', 'ex1'), ('sev', 'ex2'), ('sev', 'ex3'),
        ('sev', 'mean'), ('sev', 'cv'), ('sev', 'skew'),
        ('agg', 'ex1'), ('agg', 'ex2'), ('agg', 'ex3'),
        ('agg', 'mean'), ('agg', 'cv'), ('agg', 'skew'),
    ],
    names=['component', 'measure'],
)


class Aggregate(LabeledMixin):
    """Compound (aggregate) probability distribution.

    Implements the FFT-based algorithm of Mildenhall (2024): discretize
    severity, FFT, apply frequency PGF, inverse FFT. See
    ``_freq_sev_convolution`` for the five-line core; ``update_work`` for the
    orchestration (severity prep → occurrence reinsurance → convolution →
    aggregate reinsurance → audit). Validation by theoretical-vs-empirical
    moment comparison (paper §4.7) lives in the ``empirical`` and ``error``
    columns of ``stats_df`` written at the end of ``update_work``.

    Construction is via the ``__init__`` arguments below, or — more usually —
    via :func:`build` parsing DecL.

    **Public surface that Portfolio and Bounds depend on.** Three stats
    surfaces — ``info`` for text, ``summary_df`` for the daily risk view,
    and ``stats_df`` for everything else — plus the compute and risk-measure
    surface:

    Stats / display
        - ``info``: one-screen textual summary (frequency, severity, layer,
          grid, validation flag). Not stats.
        - ``summary_df``: 3-row Freq / Sev / Agg at-a-glance moments + key
          percentiles. The daily-driver headline. ``tail_df`` is the companion
          return-period table; ``validation_df`` the moment-error QA frame.
        - ``stats_df``: ``MultiIndex (component, measure)`` × per-component
          / ``mixed`` / ``independent`` / ``empirical`` / ``error``.
          Single source of truth for Aggregate moments — see the property
          for the row / column reference.
        - ``stats_df``: ``MultiIndex (component, measure)`` × per-component
          / ``mixed`` / ``independent`` / ``empirical`` / ``error``.
          Single source of truth for Aggregate moments — see the property
          for the row / column reference.

    Data attributes
        - ``agg_density``: empirical PMF on the bucket grid (set by
          ``update_work``; consumed by Portfolio).
        - ``ftagg_density``: FT of the aggregate density (consumed by
          Portfolio's copula combine).
        - ``density_df``: per-bucket density / CDF / risk-measure frame.
        - ``n``: total frequency.
        - ``name``, ``program``, ``note``: spec metadata.
        - ``bs``, ``log2``, ``xs``: discretization grid.

    Methods
        - ``update``, ``update_work``: trigger / drive the compute.
        - ``q``, ``q_sev``, ``tvar``, ``tvar_sev``, ``cdf``, ``sf``, ``pdf``,
          ``pmf``, ``var_dict``: risk-measure surface.
        - ``sample``: draw from the discretised aggregate.
        - ``price``: distortion-based pricing.
        - ``approximate``, ``entropy_fit``: parametric fits to the FFT output.
        - ``apply_distortion``, ``pollaczeck_khinchine``: distortion / ruin.
        - ``plot``: single plotting entry point.
        - ``snap``, ``picks``, ``unwrap``,
          ``aggregate_error_analysis``, ``severity_error_analysis``: utilities.

    Methods / attributes with a leading underscore are internal —
    ``_init_stats_df``, ``_record_component``, ``_freq_sev_convolution``,
    ``_apply_reins_work``, ``_limits``, ``_html_info_blob``,
    ``_grid_distribution``, … . The legacy ``audit_df`` / ``report_df`` /
    ``report_ser`` / ``statistics`` / ``statistics_df`` /
    ``statistics_total_df`` surface has been removed; consult ``stats_df``
    instead.
    """

    # ================================================================
    # Public read-only properties: spec, density frame, reinsurance frames
    # ================================================================

    @property
    def spec(self):
        """
        Get the dictionary specification, but treat as a read only
        property

        :return:
        """
        return self._spec

    @property
    def spec_ex(self):
        """
        All relevant info.

        :return:
        """
        return {'type': type(self), 'spec': self.spec, 'bs': self.bs, 'log2': self.log2,
                'sevs': len(self.sevs)}

    def _tail_info(self):
        """Build the :class:`~aggregate.tail.TailInfo` for this aggregate.

        Spec-only (frequency name + severity families); touches no computed
        density, so it is valid *before* ``update()``. A ``_certified_bounded``
        override short-circuits to a BOUNDED result so the certify contract and
        the lifted-natural-allocation guard are preserved.

        Returns
        -------
        aggregate.tail.TailInfo
        """
        if getattr(self, '_certified_bounded', False):
            return _tail.TailInfo(
                freq=TailClass.BOUNDED, sev=TailClass.BOUNDED, agg=TailClass.BOUNDED,
                freq_lc=None, sev_lc=None, agg_lc=None, alpha=None,
                driver='certified', flags={'certified': True})
        return _tail.aggregate_tail_info(self.frequency, self.sevs)

    @property
    def tail_class(self):
        """The ``(freq, sev, agg)`` :class:`~aggregate.tail.TailClass` triple.

        The authoritative tail-thickness computation: frequency and severity
        rungs by deterministic family lookup, the aggregate rung by the
        ``max`` combine rule. ``.bounded`` and the tail text are derived from
        this. Returns a :class:`~aggregate.tail.TailClasses` namedtuple with
        ``.freq`` / ``.sev`` / ``.agg`` fields.
        """
        return self._tail_info().classes

    @property
    def bounded(self) -> bool:
        """Whether the aggregate has bounded support.

        Derived view: ``True`` iff the aggregate tail class is
        :attr:`~aggregate.tail.TailClass.BOUNDED`, i.e. the frequency *and*
        every severity component is bounded. Frequencies in
        :data:`~aggregate.tail._BOUNDED_FREQS` (``fixed``, ``bernoulli``,
        ``binomial``, ``empirical``) are bounded; mixed Poisson / negbin / etc.
        are not. A severity is bounded when it is a histogram, a fixed atom, a
        bounded scipy family, or carries a finite layer ``exp_limit`` or splice
        ``sev_ub``. Conservative: ``False`` whenever boundedness cannot be
        proved from the spec. Set ``self.bounded = True`` to certify (e.g. a
        fat-tailed scipy severity with a large layer cap the heuristic misses).
        """
        return self._tail_info().agg == TailClass.BOUNDED

    @bounded.setter
    def bounded(self, value: bool) -> None:
        if value is not True and value is not False:
            raise ValueError('bounded must be True (certify) or False (reset)')
        self._certified_bounded = bool(value)

    @property
    def tail_description(self) -> str:
        """Three aligned lines summarising the frequency, severity, and aggregate tails.

        Short narrative over the layered :attr:`tail_behavior_df` report -- per-layer
        support and per-side tail class, plus the aggregate concentration. E.g.::

            frequency tail           poisson, count [0, inf), super-exponential right tail
            severity tail            lognorm, [0, inf), subexponential right tail
            aggregate tail           [0, inf), subexponential right tail; not concentrated (cv=1.5)

        The verbose form is :attr:`tail_explanation`; the lines are also appended
        to :meth:`info`.
        """
        return '\n'.join(_tail.describe_rows(self._tail_rows()))

    @property
    def tail_explanation(self) -> str:
        """Verbose prose over the layered tail report (the per-component story).

        Walks the book bottom-up -- frequency, the severity components and their
        blend, the aggregate -- naming the single-big-jump mechanism (or the
        frequency driver) for a thick right tail, any power-law moment failure,
        and the concentration. Derived from :attr:`tail_behavior_df`.
        """
        return _tail.explain_rows(self._tail_rows(), self._tail_info())

    @property
    def bs_window_df(self) -> 'pd.DataFrame':
        """Curated, read-only view of the bucket/window decision (``[bs-reporting]``).

        One row per sizing method that ran (``moment``, ``exact_discrete``,
        ``bounded_small``, ``windowed``, ``sbj``) plus the realized ``used``
        grid, culled to the user-facing columns: whether the method ``applies``,
        whether it was ``selected``, the method window (``x_min`` / ``x_max``),
        the grid (``bs`` / ``log2``), the log2 the window ``needs`` at that ``bs``,
        any estimated ``clipped`` far-tail mass, and a one-line ``note``.

        The complete decision journey (extra columns, coverage strings) stays on
        the private :attr:`_bs_window_df` for experts. Returns ``None`` before
        :meth:`update`. See :attr:`bs_description` / :attr:`bs_explanation` for the
        narrative.
        """
        if self._bs_window_df is None:
            return None
        cols = ['applies', 'selected', 'x_min', 'x_max', 'bs', 'log2',
                'log2_need', 'clipped', 'note']
        return self._bs_window_df.reindex(columns=cols).copy()

    @property
    def bs_description(self) -> str:
        """One-line summary of the chosen bucket grid (``[bs-reporting]``).

        The winning method and the realized ``(bs, log2, x_min)`` with the grid
        top, plus a clip note when the far tail is truncated. The verbose form is
        :attr:`bs_explanation`; the ANSI-coloured variant is
        ``aggregate.distributions.bs_describe(agg, color=True)``.
        """
        return bs_describe(self)

    @property
    def bs_explanation(self) -> str:
        """Verbose prose explaining the bucket-grid choice (``[bs-reporting]``).

        What the book is (the aggregate tail one-liner), which methods applied
        and which won and why, the realized grid, and any far-tail clip with how
        to widen it. The ANSI-coloured variant is
        ``aggregate.distributions.bs_explain(agg, color=True)``.
        """
        return bs_explain(self)

    def _sev_label(self) -> str:
        """Short severity family label for tail text (the family, or ``'N components'``)."""
        if self.sevs is None or len(self.sevs) == 0:
            return ''
        if len(self.sevs) == 1:
            name = getattr(self.sevs[0], 'sev_name', '')
            return name if isinstance(name, str) else 'severity'
        return f'{len(self.sevs)} components'

    @property
    def tail_behavior_df(self) -> pd.DataFrame:
        """The layered tail report -- support and tail class per layer -- as a DataFrame.

        One row per layer, bottom-up -- ``frequency``; one per severity mix
        component (``comp0`` ...); the combined effective ``severity`` (only when
        there is more than one component); and the ``aggregate``. Columns:
        ``min`` / ``max`` **structural support** (``-inf`` / ``inf`` at an
        unbounded end), ``left_tail`` / ``right_tail`` (the per-side tail class --
        ``bounded`` at a finite end, else the family decay rung), ``bounded`` (the
        support finite both ends), and (aggregate row only) the conservative
        ``concentrated`` flag and ``cv = sd / mean``. A
        ``note`` carries power-law ``alpha`` / infinite-moment and capped-base
        annotations.

        Spec-only -- built from the family classifier, the structural support,
        and the pre-computed moments, so it is valid *before* :meth:`update`. The
        numeric grid *reach* lives in the bucket report (``bs_window_df``), not
        here. See :mod:`aggregate.tail` and ``dev/bucket-selection.rst``.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``component``.
        """
        return _tail.tail_frame(self._tail_rows())

    def _tail_rows(self):
        """The layered tail report as a list of :class:`~aggregate.tail.TailRow`.

        Single source for :attr:`tail_behavior_df`, :attr:`tail_description`,
        and :attr:`tail_explanation`. Spec-only -- valid before :meth:`update`.
        """
        freq_min, freq_max, freq_zt = self._frequency_count_support()
        return _tail.build_tail_rows(
            self.frequency, self.sevs,
            freq_min=freq_min, freq_max=freq_max, freq_zero_truncated=freq_zt,
            agg_m=self.agg_m, agg_sd=self.agg_sd,
            occ_reins=self.occ_reins,
        )

    def _frequency_count_support(self):
        """Spec-only ``(min, max, zero_truncated)`` claim-count support.

        The count magnitude lives on the aggregate (the exposure ``n``), not on
        the bare :class:`Frequency`, so it is resolved here. Returns ``(n, n)``
        for a ``fixed`` count, ``(0, 1)`` for ``bernoulli``, the atom range for
        an ``empirical`` count, and ``(0, inf)`` otherwise -- with the third
        element flagging a genuine zero-truncated (``zm``, ``p0 == 0``) count.
        """
        fname = getattr(self.frequency, 'freq_name', '')
        n = float(self.n) if self.n else 0.0
        if fname == 'fixed':
            return n, n, False
        if fname == 'bernoulli':
            return 0.0, 1.0, False
        if fname == 'empirical':
            atoms = getattr(self.frequency, 'freq_a', None)
            if atoms is not None and len(atoms):
                return float(np.min(atoms)), float(np.max(atoms)), False
        zt = False
        lo = 0.0
        if getattr(self.frequency, 'freq_zm', False):
            p0 = getattr(self.frequency, 'freq_p0', None)
            if p0 is not None and float(p0) == 0.0:
                lo, zt = 1.0, True
        return lo, np.inf, zt

    @property
    def reins_bucket(self) -> str:
        """Rebucketing scheme for reinsurance net/ceded distributions.

        ``'linear'`` (default) splits each off-grid net/ceded value's mass
        across its two bracketing grid buckets, preserving the first moment
        exactly; ``'nearest'`` rounds to the closest bucket (≤ ``bs/2``
        positional bias). Mirrors :meth:`Portfolio.allocation_method`.

        Reinsurance is baked in during :meth:`update`, so a change after
        ``build`` requires a re-``update()`` to take effect. The setter
        clears the cached reinsurance frames so they rebuild on next access.
        """
        return self._reins_bucket

    @reins_bucket.setter
    def reins_bucket(self, value: str) -> None:
        if value not in ('linear', 'nearest'):
            raise ValueError(
                f"reins_bucket must be 'linear' or 'nearest', not {value!r}")
        if value != getattr(self, '_reins_bucket', None):
            self._reins_bucket = value
            self._reins_density_df = None
            self._reins_stats_df = None
            self._reins_view_stats_cache = None
            self._reins_describe = None

    @property
    def dsev_bucket(self) -> str:
        """Scheme for placing discrete-severity atoms onto the model grid.

        Applies to ``dsev`` / ``dhistogram`` / ``fixed`` (point-mass)
        severities; continuous severities are already exact via the
        cdf-difference and ignore this. Sibling of :attr:`reins_bucket`.

        ``'linear'`` (default) splits each off-grid atom's mass across its two
        bracketing grid buckets ``k``, ``k+1`` with weights ``1-f``, ``f``
        (``f = x/bs - k``), so the discretized first moment equals
        ``Σ xₖ pₖ`` **exactly**; ``'nearest'`` snaps each atom to its closest
        bucket (the historical behaviour, with up to ``bs/2`` positional bias).
        On-grid atoms (``f == 0``, e.g. integer atoms with ``bs == 1``) give
        identical results under both schemes.

        Discretization happens in :meth:`update`, so a change after ``build``
        requires a re-``update()`` to take effect. The setter clears the cached
        density frames so they rebuild on next access.

        Notes
        -----
        Phase 1 covers **unlayered** discrete severities (the empirical-sample
        use case). A *layered* discrete severity (``a xs b dsev ...``)
        discretizes via the standard cdf-difference and so behaves as
        ``'nearest'`` regardless of this setting.
        """
        return self._dsev_bucket

    @dsev_bucket.setter
    def dsev_bucket(self, value: str) -> None:
        if value not in ('linear', 'nearest'):
            raise ValueError(
                f"dsev_bucket must be 'linear' or 'nearest', not {value!r}")
        if value != getattr(self, '_dsev_bucket', None):
            self._dsev_bucket = value
            self._density_df = None
            self._sev_density_df = None

    def _severity_in_window(self):
        """Whether the severity support overlaps the output window ``[x_min, x_max]``.

        Returns ``True`` (the common case) when at least one severity bucket
        with positive mass falls inside the aggregate output grid. ``False``
        flags the divergent case (e.g. a high-claim-count aggregate windowed
        far from 0 while the per-claim severity sits near 0), where
        ``density_df.p_sev`` is all zeros and the severity must be read from
        ``sev_density_df`` instead.
        """
        if self.sev_density is None or self.xs_sev is None:
            return True
        nz = self.sev_density > 0
        if not np.any(nz):
            return True
        s_lo = float(self.xs_sev[nz][0])
        s_hi = float(self.xs_sev[nz][-1])
        return not (s_hi < self.x_min or s_lo > self.x_max)

    def _sev_density_on_output_grid(self):
        """Severity density mapped from ``xs_sev`` onto the output grid ``xs``.

        The severity is discretised on ``xs_sev`` (physical 0 at index ``i0``);
        the output grid is ``xs = x_min + bs*arange``. This places each severity
        bucket at its physical location on the output grid, zero where the
        severity falls outside the output window. With no offset (``i0 == 0``
        and ``x_min == 0``) it returns ``self.sev_density`` unchanged.

        Returns
        -------
        np.ndarray
            Severity density aligned to ``self.xs`` (length ``len(xs)``).

        Notes
        -----
        Output index ``k`` (physical ``x_min + k*bs``) corresponds to severity
        index ``k + i0 + j0`` where ``j0 = round(x_min/bs)`` -- because the
        severity bucket ``s`` sits at physical ``(s - i0)*bs`` and we need
        ``(s - i0)*bs == x_min + k*bs``. When the severity is far from the
        output window (e.g. a tight P&L window around a large negative mean) the
        overlap is empty and the result is all zeros -- the severity simply is
        not in the displayed window.
        """
        sev = self.sev_density
        if sev is None:
            return sev
        N = len(self.xs)
        j0 = int(round(self.x_min / self.bs)) if self.bs else 0
        shift = self.i0 + j0
        if shift == 0:
            return sev
        out = np.zeros(N)
        k = np.arange(N)
        src = k + shift
        valid = (src >= 0) & (src < N)
        out[valid] = np.asarray(sev)[src[valid]]
        return out

    @property
    def density(self):
        """The "live" part of :attr:`density_df` — rows with positive total mass.

        Returns ``density_df.query('p_total > 0')``: the actual support of the
        aggregate, dropping the leading and trailing zero-probability buckets of
        the FFT grid. This is usually what you want to *see*. It is recomputed on
        each access (a plain property, not cached) because the underlying frame
        can be rebuilt by ``update``.
        """
        return self.density_df.query('p_total > 0')

    @property
    def density_df(self):
        """Per-bucket density / distribution / risk-measure frame.

        Built lazily on first access after ``update``. Stored in
        ``self._density_df``; treat as read-only.

        Columns (in construction order):

        ================  =====================================  =========================
        Column            Set from                               Read by
        ================  =====================================  =========================
        ``loss``          ``self.xs`` (also the index)           ``plot``, ``q``, bounds
        ``p_total``       ``self.agg_density``                   Portfolio (when Aggregate
                                                                  is in a port), bounds,
                                                                  ``plot``, user code
        ``p``             alias of ``p_total``                   Portfolio API compat
        ``log_p``         ``np.log(p)``                          ``plot`` log scale
        ``F``             ``p.cumsum()``                         ``q``, ``var``, ``tvar``
        ``S``             ``1 - p_total.cumsum()``               ``q``, ``tvar``, ``plot``
        ``lev``           ``cumsum(loss·p) + loss·S``            pricing
        ``exa``           alias of ``lev``                       Portfolio API compat
        ``exlea``         ``cumsum(loss·p) / F``                 pricing
        ``e``             ``self.est_m`` (constant column)       ``exgta``
        ``exgta``         ``(e - cumsum(loss·p)) / S``           pricing
        ``exeqa``         ``loss`` (since ``E[X|X=a] = a``)      Portfolio API compat
        ================  =====================================  =========================

        ``lev``, ``exlea`` and ``exgta`` are direct sums that carry the
        window origin ``x0`` (numerics-2): on a windowed or signed grid
        ``E[X ∧ a] = Σ_{x≤a} x·p + a·S(a)``, never ``cumsum(S)·bs`` (which
        silently assumes the grid starts at 0). Ratio denominators are
        guarded: ``exlea`` (``exgta``) is NaN where ``F`` (``S``) is at or
        below the validation noise floor. The ``epd`` column was removed
        at numerics-2 (no consumers; ``max(0, e - lev)/e`` if needed).

        Duplicated columns (``p == p_total``, ``exa == lev``, ``exeqa == loss``) are
        intentional: Portfolio's ``filter(regex='p_<name>')`` / ``exeqa_*`` /
        ``exa_*`` patterns require these names exist on the unit's frame so the
        unit can be inlined into a portfolio.

        :return: DataFrame indexed by ``loss``, columns as tabulated above.
        """
        if self._density_df is None:
            # really should have one of these anyway...
            if self.agg_density is None:
                raise ValueError('Update Aggregate before asking for density_df')

            # really convenient to have p=p_total to be consistent with Portfolio objects
            self._density_df = pd.DataFrame(dict(loss=np.asarray(self.xs, dtype=float),
                                                 p_total=self.agg_density))
            self._density_df = self._density_df.set_index('loss', drop=False)
            self._density_df['p'] = self._density_df.p_total
            # remove the fuzz, same method as Portfolio.remove_fuzz
            self._density_df = remove_fuzz(self._density_df)

            # Severity columns (p_sev/F_sev/S_sev/log_p_sev) now live on their
            # own native grid in ``sev_density_df`` -- on a windowed/signed grid
            # the severity (near 0) and the aggregate (windowed) no longer share
            # a grid, so forcing the severity onto the aggregate index is at best
            # partial. See ``sev_density_df`` and ``info``'s outside-window note.

            # reindex
            self._density_df = self._density_df.set_index('loss', drop=False)
            # guard log of 0 / fp-fuzz negatives (cosmetic display column only;
            # values unchanged: log(0) = -inf, log(<0) = nan as before).
            with np.errstate(divide='ignore', invalid='ignore'):
                self._density_df['log_p'] = np.log(self._density_df.p)

            # generally acceptable for F, by construction
            self._density_df['F'] = self._density_df.p.cumsum()

            # Update 2021-01-28: S is best computed forwards
            self._density_df['S'] = 1 - self._density_df.p_total.cumsum()

            # LEV and the conditional means by direct sums that carry the
            # window origin (numerics-2): E[X∧a] = Σ_{x≤a} x·p + a·S(a),
            # valid on windowed and signed grids where cumsum(S)·bs is not.
            loss_v = self._density_df['loss'].to_numpy()
            p_v = self._density_df['p_total'].to_numpy()
            F_v = self._density_df['F'].to_numpy()
            S_v = self._density_df['S'].to_numpy()
            cum_xp = np.cumsum(loss_v * p_v)
            self._density_df['lev'] = cum_xp + loss_v * S_v
            self._density_df['exa'] = self._density_df['lev']
            # explicit denominator guards (F/S at or below the validation
            # noise floor cannot support a conditional mean)
            tol = VALIDATION_NOISE
            with np.errstate(divide='ignore', invalid='ignore'):
                exlea = cum_xp / F_v
            exlea[F_v <= tol] = np.nan
            self._density_df['exlea'] = exlea

            # expected value
            self._density_df['e'] = self.est_m  # np.sum(self._density_df.p * self._density_df.loss)
            with np.errstate(divide='ignore', invalid='ignore'):
                exgta = (self.est_m - cum_xp) / S_v
            exgta[S_v <= tol] = np.nan
            self._density_df['exgta'] = exgta
            self._density_df['exeqa'] = self._density_df.loss  # E(X | X=a) = a(!) included for symmetry was exa

        return self._density_df

    @property
    def sev_density_df(self):
        """Per-bucket severity density / distribution on the **severity** grid.

        The severity lives on its own grid ``xs_sev`` (physical 0 at index
        ``i0``), which equals the aggregate grid only on the default 0-based
        case. On a windowed / signed aggregate the severity (near 0) and the
        aggregate (windowed, possibly far from 0) genuinely occupy different
        grids, so the severity reporting columns live here -- indexed by the
        severity's own loss -- rather than in ``density_df``. This frame is
        always correct regardless of the aggregate output window.

        Columns
        -------
        loss : severity grid ``xs_sev`` (also the index).
        p_sev : ``self.sev_density``.
        log_p_sev : ``log(p_sev)`` (``-inf`` at zero buckets).
        F_sev : ``p_sev.cumsum()``.
        S_sev : ``1 - p_sev.cumsum()``.

        Returns
        -------
        DataFrame indexed by severity ``loss``.
        """
        if self._sev_density_df is None:
            if self.sev_density is None:
                raise ValueError('Update Aggregate before asking for sev_density_df')
            df = pd.DataFrame(dict(loss=np.asarray(self.xs_sev, dtype=float),
                                   p_sev=self.sev_density))
            df = df.set_index('loss', drop=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                if df.p_sev.dtype == np.dtype('O'):
                    df['log_p_sev'] = np.nan
                else:
                    df['log_p_sev'] = np.log(df.p_sev)
            df['F_sev'] = df.p_sev.cumsum()
            df['S_sev'] = 1 - df.p_sev.cumsum()
            self._sev_density_df = df
        return self._sev_density_df

    # ================================================================
    # Reinsurance reporting (rationalized; see dev/reins-reporting.md)
    # ================================================================

    @property
    def reins_density_df(self):
        """Per-bucket gross / ceded / net densities under reinsurance.

        One row per model-grid bucket (``loss = k * bs``); the empirical,
        FFT-ready rebucketed densities. Columns are **always present**
        regardless of which stages are configured; a missing stage
        contributes the no-cession values (ceded mass at 0, net = subject).

        Columns
        -------
        loss : grid (also the index).
        p_sev_gross, p_sev_ceded, p_sev_net : occurrence-level severity
            views. With no occurrence cover ``p_sev_ceded`` is a point mass
            at 0 and ``p_sev_net == p_sev_gross``.
        p_agg_gross, p_agg_ceded_occ, p_agg_net_occ : aggregate of each
            occurrence severity view. ``p_agg_gross`` is the *true* gross
            aggregate (``_fft_aggregate`` of the gross severity).
        p_agg_subject, p_agg_ceded, p_agg_net : aggregate-cover views.
            ``p_agg_subject`` is the aggregate input to the aggregate cover
            (= aggregate of the requested occurrence output); it equals
            ``p_agg_gross`` only when there is no occurrence cover.

        Notes
        -----
        All aggregate columns route through ``_fft_aggregate`` so the
        zero-risk / fixed-1 shortcuts apply consistently. Renamed from the
        legacy ``reinsurance_df``: ``p_agg_gross_occ -> p_agg_gross`` and the
        old ``p_agg_gross`` (the agg-cover input) ``-> p_agg_subject``.
        Returns ``None`` when no reinsurance is configured.
        """
        return _reinsurance.reins_density_df(self)

    def reins_occ_plot(self, axs=None, **kwargs):
        """
        Plots for occurrence reinsurance: occurrence log density and aggregate
        quantile plot. Reads the gross/ceded/net views from ``reins_density_df``.

        :param kwargs: Lee-panel options forwarded to the quantile worker --
               notably ``quantile_x='return'`` and ``max_return_period``.
        """
        from .plots import plot_reins_occ
        return plot_reins_occ(self, axs=axs, **kwargs)

    def occ_bivariate(self, views=('net', 'ceded'), bs=None,
                      log2_x=None, log2_y=None):
        """Joint law of two of the occurrence {gross, ceded, net} aggregates via 2D FFT.

        Computes the *joint* distribution of two aggregate occurrence views --
        e.g. ceded ``C = sum c(X_i)`` and net ``N = sum n(X_i)`` -- where each
        gross claim ``X_i`` is split deterministically by the occurrence cession
        map. The two are **not** deterministic functions of one another -- the
        random claim count decouples them -- so the joint law carries genuine
        information (their correlation, co-moments, reinsurer-vs-cedent
        dependency) beyond the two univariate margins already in
        :attr:`reins_density_df`. The three views satisfy ``ceded + net = gross``,
        so any *two* determine the third; the pick of which two is ``views``.

        Parameters
        ----------
        views : (str, str), default ``('net', 'ceded')``
            The ``(x, y)`` axis view pair, each one of ``'gross'`` / ``'ceded'``
            / ``'net'``. Axis 0 (x) is ``views[0]``, axis 1 (y) is ``views[1]``.
            The default ``('net', 'ceded')`` matches the DecL ``netceded`` form;
            ``('gross', 'ceded')`` matches ``grossceded`` and ``('gross', 'net')``
            matches ``grossnet``.
        bs : float, optional
            Bucket-size override (a single common ``bs`` for both axes). Default:
            one common ``bs`` sized from the budget (coarser than the gross
            bucket -- the gross grid is far finer than a 2-D grid affords).
        log2_x, log2_y : int, optional
            Axis-0 / axis-1 log2 grid lengths (grid has ``1 << log2`` points).
            Default: measured from the two views' occurrence aggregate margins
            via :func:`~aggregate.utilities.balanced_window`.

        Returns
        -------
        BivariateAggregate
            A first-class joint object in ``netceded`` mode, with the joint
            ``density``, the two axis grids (``axis_xs``), ``marginals`` /
            ``moments`` (``E[X^i Y^j]``) / ``corr`` / ``summary_df`` / ``stats_df``
            / ``info`` and a two-panel ``plot`` (comonotone per-claim severity
            and joint aggregate). Equivalent to the DecL ``netceded`` /
            ``grossceded`` / ``grossnet`` ``<agg>`` prefix forms.

        Raises
        ------
        ValueError
            If the object carries no occurrence reinsurance, or has not been
            updated (no severity densities present).

        Notes
        -----
        **Occurrence only.** Any *aggregate* reinsurance on the object is
        ignored: this is the joint law of the per-occurrence ceded / net
        aggregates. (The aggregate-cover bivariate is degenerate -- at the
        aggregate level ceded and net are deterministic functions of the
        aggregate, supported on a curve -- so it is out of scope.)

        **Method.** Per claim, ``(c(X), n(X))`` lies on the line ``c + n = X``.
        Placing the gross severity mass at ``(c(x_k), n(x_k))`` (rebucketed onto
        the 2D grid by the active :attr:`reins_bucket` scheme) builds the
        bivariate severity ``S``. The joint aggregate density is
        ``iFFT2(freq_pgf(n, FFT2(S)))`` -- exactly the univariate
        :meth:`_fft_aggregate` with the 1D transforms replaced by 2D transforms,
        valid because ``freq_pgf(n, z)`` is elementwise in ``z``. The zero-risk
        (``n == 0``) and fixed-count-one shortcuts mirror ``_fft_aggregate``.
        Marginalising the result over one axis recovers the corresponding
        univariate occurrence ceded / net aggregate (exact validation targets;
        see the ``Cross-check`` notes in ``dev/done/reins-bivariate.md``).

        This wraps the object as a :class:`aggregate.bivariate.BivariateAggregate`
        in ``netceded`` mode (the engine is :func:`aggregate.bivariate.build_netceded_joint`).
        """
        from .bivariate import BivariateAggregate

        mv = BivariateAggregate(
            self.name, mode='netceded', nc_agg=self, nc_views=views,
            nc_kwargs=dict(bs=bs, log2_x=log2_x, log2_y=log2_y))
        # build eagerly so preconditions (occ reins present, object updated)
        # raise here, and the returned object is ready to query.
        mv.update()
        return mv

    def variable_rating_analysis(self, *, percentiles=None):
        """Variable-rating analysis for this account's single feature.

        Pairs the stored :class:`~aggregate.contract_terms.ContractTerms`
        (a ``swing`` / ``slide`` / ``pc`` / ``corridor`` DecL feature, attached by
        the underwriter) with this aggregate's **gross** loss density and returns a
        :class:`~aggregate.variable_rating.VariableRatingAnalysis` (the leg
        distributions and the Gross / Ceded / Net exhibit). The aggregate is built
        gross of the decorated layer; the analysis applies the layer ceder, so the
        feature's one stochastic leg is a 1-D pushforward of the gross density
        (decision 0, aggregate basis).

        Parameters
        ----------
        percentiles : tuple of float, optional
            Adverse-tail levels for the summary; default the analysis default.

        Returns
        -------
        VariableRatingAnalysis
        """
        from .variable_rating import VariableRatingAnalysis
        terms = getattr(self, 'variable_terms', None)
        if terms is None:
            raise ValueError(
                f"{self.name}: no variable-rating feature attached (set by a DecL "
                "swing / slide / pc / corridor clause).")
        grid = self.density_df.index.to_numpy()
        density = self.density_df['p'].to_numpy()
        kw = {} if percentiles is None else {'percentiles': percentiles}
        return VariableRatingAnalysis(
            grid, density, terms,
            gross_premium=getattr(self, 'variable_gross_premium'),
            ceded_premium=getattr(self, 'variable_ceded_premium', 0.0),
            layer=getattr(self, 'variable_layer', None),
            gross_expense=getattr(self, 'variable_gross_expense', 0.0),
            commission=getattr(self, 'variable_commission', 0.0), **kw)

    def reinstatement_analysis(self, gross_premium=None, terms=None, *,
                               percentiles=None, agg_ceded_premium=0.0,
                               gross_expense=0.0, occ_commission=0.0,
                               agg_commission=0.0, bs=None,
                               log2_x=None, log2_y=None):
        """Stochastic-ceded reinstatement analysis for this occurrence layer.

        Pairs a :class:`~aggregate.reinstatement.ReinstatementTerms` with the
        ``(gross loss, unlimited ceded recovery)`` joint of this aggregate and
        returns a :class:`~aggregate.reinstatement.ReinstatementAnalysis` (the
        eleven leg distributions and the GCN / summary / audit / tail exhibits).
        The reinstatement annual cap ``(m+1) y`` lives in ``terms.recovery`` --
        the source feeds an **unlimited** ``R`` -- so the occurrence layer must
        carry *no* annual aggregate cap of its own; a genuine subsequent
        ``aggregate net of`` cover is allowed (it is a deterministic pushforward
        over the same joint).

        Parameters
        ----------
        gross_premium : float, optional
            Fixed gross premium ``P_G``. Defaults to ``self.exp_premium`` (the
            DecL ``premium`` clause) if set, else required.
        terms : ReinstatementTerms, optional
            The reinstatement basis. Defaults to ``self.reinstatement_terms`` (set
            by a DecL ``reinstatements`` clause) if present, else required.
        percentiles : tuple of float, optional
            Adverse-tail levels for the summary table.
        bs, log2_x, log2_y : optional
            Forwarded to :meth:`occ_bivariate` for the joint grid.

        Returns
        -------
        ReinstatementAnalysis

        Raises
        ------
        ValueError
            If the object carries no occurrence reinsurance, the occurrence layer
            is not a single layer ([reins-single-layer]), or ``gross_premium`` /
            ``terms`` are missing and have no default.
        """
        from .reinstatement import ReinstatementAnalysis, SUMMARY_PERCENTILES
        if self.occ_reins is None:
            raise ValueError(
                'reinstatement_analysis requires occurrence reinsurance; none '
                'configured on this aggregate.')
        if len(self.occ_reins) != 1:
            raise ValueError(
                f'reinstatements require a single occurrence layer; found '
                f'{len(self.occ_reins)} ([reins-single-layer]). A reinstated '
                'layer cannot share the occurrence tier with other variable '
                'layers (the engine ceiling is the 2-D (L, R) joint).')
        if terms is None:
            terms = getattr(self, 'reinstatement_terms', None)
        if terms is None:
            raise ValueError(
                'reinstatement_analysis needs terms= (a ReinstatementTerms), or '
                'a DecL reinstatements clause that sets self.reinstatement_terms.')
        if gross_premium is None:
            # the gross premium stashed by a DecL ``pnl ... reinstatements``
            # build (the consideration), then the exposure ``premium`` clause.
            gross_premium = getattr(self, 'reinstatement_gross_premium', None)
        if gross_premium is None:
            gross_premium = getattr(self, 'exp_premium', None)
        if gross_premium is None or gross_premium == 0:
            raise ValueError(
                'reinstatement_analysis needs gross_premium= (no premium clause '
                'on the aggregate to default from).')
        biv = self.occ_bivariate(views=('gross', 'ceded'), bs=bs,
                                 log2_x=log2_x, log2_y=log2_y)
        # decision 3: a subsequent aggregate cover is a deterministic pushforward
        # of the SAME joint via the net-of-occurrence loss L - A(R). Build its
        # ceder g (the recovery map) so the analysis can populate the agg-tier
        # waterfall columns; R stays unlimited (the cap lives only in terms).
        agg_recovery = None
        if self.agg_reins is not None:
            ceder, _netter = _reinsurance.make_ceder_netter(self.agg_reins)
            agg_recovery = ceder
        kw = {} if percentiles is None else {'percentiles': percentiles}
        return ReinstatementAnalysis(biv.bivariate, terms, gross_premium,
                                     joint_aggregate=biv, agg_recovery=agg_recovery,
                                     agg_ceded_premium=agg_ceded_premium,
                                     gross_expense=gross_expense,
                                     occ_commission=occ_commission,
                                     agg_commission=agg_commission, **kw)

    # ----- reinsurance stats: exact (EX) vs rebucketed (Est) -------------

    @staticmethod
    def _reins_moments6_from_raw(e1, e2, e3):
        """``(ex1, ex2, ex3, mean, cv, skew)`` from raw moments."""
        return _reinsurance.reins_moments6_from_raw(e1, e2, e3)

    def _reins_exact_image_raw(self, image_fn, p_subject):
        """Raw moments ``E[g(X)^j]``, ``j=1..3``, of an exact loss image
        ``g = image_fn`` weighted by ``p_subject``: the pre-bucket truth
        ``sum g(xs)^j * p_subject`` with **no** rebucketing scatter.

        Notes
        -----
        Unlike :func:`xsden_to_mwrangler` (the ``Est`` basis), no
        defective-mass tail term is added: that convention places lost mass
        at the implied max loss ``xs[-1] + bs``, which is right for a gross
        aggregate but wrong for a ceded image (where the lost mass maps to the
        capped cession). The EX basis is therefore the literal exact moment of
        the on-grid subject; the EX-vs-Est difference reflects the rebucketing
        scatter (plus, for a gross aggregate carried through an FFT, the grid
        deficit -- negligible on an adequate grid).
        """
        return _reinsurance.reins_exact_image_raw(self, image_fn, p_subject)

    def _reins_agg6_from_sev_raw(self, s1, s2, s3):
        """Compound exact severity raw moments into aggregate
        ``(ex1, ex2, ex3, mean, cv, skew)`` via the frequency."""
        return _reinsurance.reins_agg6_from_sev_raw(self, s1, s2, s3)

    def _reins_density6(self, p):
        """``(ex1, ex2, ex3, mean, cv, skew)`` of a density on the grid."""
        return _reinsurance.reins_density6(self, p)

    @property
    def _reins_view_stats(self):
        """Per-stage reinsurance moments on two bases: exact and rebucketed.

        Internal frame feeding :meth:`reins_summary_df` (which surfaces the
        ``EX`` exact vs ``Est`` rebucketed comparison as its ``Change``
        column). The public per-layer summary is :meth:`reins_stats_df`.
        Shaped like ``stats_df`` but indexed by stage / view / basis instead
        of component columns.

        Rows
        ----
        ``MultiIndex (component, measure)`` with ``component in
        {freq, sev, agg}`` and ``measure in {ex1, ex2, ex3, mean, cv, skew}``.

        Columns
        -------
        ``MultiIndex (stage, view, basis)``:

        * ``stage`` ``occ`` (when ``occ_reins``) and/or ``agg`` (when
          ``agg_reins``);
        * ``view`` ``gross|ceded|net`` for ``occ``; ``subject|ceded|net`` for
          ``agg``;
        * ``basis`` ``EX`` (exact, pre-bucket) or ``Est`` (rebucketed).

        Notes
        -----
        **EX** ("theoretic"). For the occurrence stage the exact severity
        moments are ``E[g(X)^j] = sum g(xs)^j * p_sev_gross`` with ``g`` the
        identity / ``occ_ceder`` / ``occ_netter``; the aggregate row compounds
        those via the frequency. For the aggregate stage the exact aggregate
        moments are ``E[g(S)^j] = sum g(xs)^j * p_agg_subject`` with ``g`` the
        identity / ``agg_ceder`` / ``agg_netter`` (no compounding -- the cover
        acts on the aggregate directly). Frequency: the ``EX`` reference is the
        gross full moments for every view (the count is unchanged by occurrence
        reinsurance). On the ``Est`` (model-output) basis the gross view is left
        ``NaN`` (mirroring ``validation_df``, which never re-estimates the input
        frequency); occ ceded / net carry the *unconditional* mean ``E[N]`` only
        (so ``freq * sev == agg`` per view), cv / skew ``NaN``. The aggregate
        stage has no sev rows and a degenerate freq row (all ``NaN``).
        Columns are ordered occ before agg, views gross/subject, ceded, net,
        and ``EX`` before ``Est``.

        **Est** ("empirical") reads the rebucketed densities from
        :meth:`reins_density_df` through ``xsden_to_mwrangler``. The
        difference EX vs Est isolates the per-stage ``reins_bucket``
        rebucketing error (``linear`` preserves the mean exactly; ``nearest``
        biases it by at most ``bs/2``).

        Lazily built; invalidated by the ``reins_bucket`` setter and on
        ``update``. Returns ``None`` when no reinsurance is configured.
        """
        return _reinsurance.reins_view_stats(self)

    @property
    def reins_stats_df(self):
        """Per-layer reinsurance layering summary (empirical, model-grid).

        A layering analysis with one column per reinsurance layer plus the
        gross book and the ceded / net totals.

        Columns
        -------
        ``MultiIndex (view, layer)`` -- ``view`` is ``occ`` / ``agg``:

        * occurrence: ``Gross`` (the gross book -- always present), then, when
          ``occ_reins``, ``layer.1`` ... ``layer.k`` (one per layer) and
          ``Ceded`` / ``Net`` totals.
        * aggregate (when ``agg_reins``): ``layer.1`` ... ``layer.m``,
          ``Ceded`` / ``Net``. There is **no aggregate ``Subject`` column** --
          the subject (the occurrence output the aggregate cover applies to)
          is the column flagged ``('meta', 'output') == 1`` in the occurrence
          block, or ``Gross`` when there is no occurrence program.

        Rows
        ----
        ``MultiIndex (component, measure)``:

        * ``meta``: ``share`` (proportion covered), ``limit``, ``attach``,
          ``pr_attach`` and ``pr_detach`` -- the **ground-up exposure
          probabilities** that the underlying loss attaches / fully exhausts
          the view, ``P(X > exp_attach + view_attach)`` and ``P(X > exp_attach
          + view_attach + view_limit)``. These come from the **underlying**
          severity (``self.sevs[i].fz``), not the modeled ``sev_density``,
          which is conditional (claims to the policy layer) and would report 0
          at the policy cap. ``pr_detach`` is ``NaN`` for unlimited layers /
          net totals. ``pr_loss`` (``P(aggregate > 0)`` from the column's
          aggregate density), ``lol`` (loss on line = expected layer aggregate
          loss / placed limit), and ``output`` (``0/1`` flag marking each
          stage's output view -- two 1s for an occ+agg program; ``Gross``
          carries the 1 when there is no occurrence program). ``Gross`` carries
          the claim-count-weighted policy ``limit`` / ``attach`` (``share`` 1);
          the occurrence ``Ceded`` total carries the share-placed sum of layer
          limits and the minimum attachment. (The layer ``freq`` ``n'`` uses
          the *conditional* ``P(subject > attach | policy loss)`` -- the model
          count is claims to the policy -- a separate basis from the absolute
          ``pr_attach``.)
        * ``('freq'|'sev'|'agg', ex1|ex2|ex3|mean|cv|skew)`` -- moments
          (``ex1`` duplicates ``mean`` for easy ``filter(regex=...)`` access).

        Notes
        -----
        **Occurrence layers are conditional** on a loss reaching the layer:
        the frequency is the expected penetrating count ``n' = E[N] * P(X >
        attach)`` and the severity is the unconditional layer severity divided
        by ``P(X > attach)`` (conditional given attach), which leaves the layer
        aggregate mean ``n' * sev`` equal to the unconditional ``E[N] *
        E[ceded]``. The ``agg`` row is the column's actual aggregate
        distribution (FFT of the unconditional layer ceded severity), so its
        higher moments and ``pr_loss`` are exact.

        **The occurrence ``Ceded`` / ``Net`` columns are unconditional**
        totals: the same claim count as ``Gross`` and unconditional
        severities, so ``Ceded`` sev + ``Net`` sev == ``Gross`` sev. Aggregate
        layer means sum to the ``Ceded`` aggregate mean.

        **The aggregate block leaves ``freq`` and ``sev`` all ``NaN``** -- a
        cover on the aggregate has no per-claim frequency / severity that
        combine in the usual way.

        Lazily built; invalidated by the ``reins_bucket`` setter and on
        ``update``. Returns ``None`` when no reinsurance is configured.
        """
        return _reinsurance.reins_stats_df(self)

    @property
    def reins_summary_df(self):
        """Per-stage reinsurance summary -- the daily driver.

        Mirrors the **economic view** of :attr:`validation_df`: compare the theoretic
        reference (the leading view -- ``Gross`` for occurrence, ``Subject`` for
        aggregate) against the model output of each view. One block per
        applicable stage; each block is a ``view x component`` table sharing the
        **same eight columns as** :attr:`validation_df`:

        * ``EX`` / ``Est EX`` / ``Change EX`` -- the **theoretic reference** mean
          (constant down each component), the per-view model-output mean, and
          ``(Est - reference) / reference``;
        * ``CV`` / ``Est CV`` / ``Change CV`` -- the same three for the CV;
        * ``Sk`` / ``Est Sk`` -- reference and model-output skew (no change
          column).

        ``Change`` carries two readings off one arithmetic: on the leading
        (Gross / Subject) row the reference and the model output are the same
        view, so it is the **numerical validation / rebucketing error** (~0 under
        ``linear``); on the ceded / net rows it is the **% impact of the
        cession** on that moment. This is the per-view, per-component analogue of
        the single ``Change`` column in :attr:`validation_df`.

        Layout (per the gross/subject convention; ``view`` and ``component``
        labels are lower-case to match the other frames):

        * **Occurrence block** -- leads with ``gross`` (the reference): rows
          ``(gross|ceded|net) x (freq|sev|agg)``.
        * **Aggregate block** -- leads with ``subject`` (the reference): rows
          ``(subject|ceded|net) x (agg)`` (sev not applicable; freq
          degenerate -> ``NaN``).

        Frequency is reported on the ``Est`` (model-output) basis
        *unconditionally* (mean ``E[N]`` only, so ``freq * sev == agg`` within a
        view; cv / skew ``NaN``) -- consistent with :meth:`reins_stats_df`. The
        leading ``gross`` row's ``Est`` frequency is left ``NaN`` to mirror
        :attr:`validation_df` exactly. (Only the per-layer ``layer.k`` columns of
        :meth:`reins_stats_df` are *conditional*; ``reins_summary_df`` is always
        unconditional.)

        Index is ``MultiIndex (stage, view, component)``. Derived from
        :meth:`reins_stats_df` / the internal view-stats frame. Returns ``None``
        when no reinsurance is configured.
        """
        return _reinsurance.reins_summary_df(self)

    def _reins_describe_block(self, stage, views, comps):
        """One :meth:`reins_summary_df` block: theoretic reference vs model output
        by view x component, mirroring the eight-column :attr:`validation_df` layout.

        The ``EX`` / ``CV`` / ``Sk`` columns hold the **theoretic reference** --
        the leading view's exact (pre-bucket) moments: ``Gross`` for the
        occurrence block, ``Subject`` for the aggregate block. They are therefore
        constant down each component (the same reference is compared against
        every view). ``Est *`` is the per-view model output (the rebucketed,
        model-grid moment). ``Change`` is ``(Est - reference) / reference``: on
        the leading (Gross/Subject) row it degenerates to the rebucketing /
        validation error (~0 under ``linear``); on the ceded / net rows it reads
        as the % impact of the cession on that moment.
        """
        return _reinsurance.reins_describe_block(self, stage, views, comps)

    def rescale(self, scale, kind='homog'):
        """
        Return a rescaled Aggregate object - used to compute derivatives.

        All need to be safe multiplies because of array specification there is an array that is not a numpy array

        TODO have parser return numpy arrays not lists!

        :param scale:  amount of scale
        :param kind:  homog of inhomog

        :return:
        """
        spec = self._spec.copy()

        def safe_scale(sc, x):
            """
            if x is a list wrap it

            :param x:
            :param sc:
            :return: sc x
            """

            if type(x) == list:
                return sc * np.array(x)
            else:
                return sc * x

        nm = spec['name']
        spec['name'] = f'{nm}:{kind}:{scale}'
        if kind == 'homog':
            # do NOT scale en... that is inhomog
            # do scale EL etc. to keep the count the same
            spec['exp_el'] = safe_scale(scale, spec['exp_el'])
            spec['exp_premium'] = safe_scale(scale, spec['exp_premium'])
            spec['exp_attachment'] = safe_scale(scale, spec['exp_attachment'])
            spec['exp_limit'] = safe_scale(scale, spec['exp_limit'])
            spec['sev_loc'] = safe_scale(scale, spec['sev_loc'])
            # note: scaling the scale takes care of the mean, so do not double count
            # default is 0. Can't ask if array is...but if array have to deal with it
            if (type(spec['sev_scale']) not in (int, float)) or spec['sev_scale']:
                spec['sev_scale'] = safe_scale(scale, spec['sev_scale'])
            else:
                spec['sev_mean'] = safe_scale(scale, spec['sev_mean'])
            if spec['sev_xs']:
                spec['sev_xs'] = safe_scale(scale, spec['sev_xs'])
        elif kind == 'inhomog':
            # just scale up the volume, including en
            spec['exp_el'] = safe_scale(scale, spec['exp_el'])
            spec['exp_premium'] = safe_scale(scale, spec['exp_premium'])
            spec['exp_en'] = safe_scale(scale, spec['exp_en'])
        else:
            raise ValueError(f'Inadmissible option {kind} passed to rescale, kind should be homog or inhomog.')
        return Aggregate(**spec)

    # ================================================================
    # Construction: __init__ and its component-recording helper
    # ================================================================

    def __init__(self, name, exp_el=0.0, exp_premium=0.0, exp_lr=0.0, exp_en=0.0, exp_attachment=None, exp_limit=np.inf,
                 sev_name='', sev_a=np.nan, sev_b=0.0, sev_mean=0.0, sev_cv=0.0, sev_loc=0.0, sev_scale=0.0,
                 sev_xs=None, sev_ps=None, sev_wt=1.0, sev_lb=0.0, sev_ub=np.inf, sev_conditional=True,
                 sev_signed=False, sev_reflect=False,
                 sev_pick_attachments=None, sev_pick_losses=None,
                 occ_reins=None, occ_kind='', occ_reins_label=None,
                 freq_name='', freq_a=0.0, freq_b=0.0, freq_zm=False, freq_p0=np.nan,
                 agg_reins=None, agg_kind='', agg_reins_label=None,
                 reins_bucket=None, dsev_bucket=None,
                 value_type='loss',
                 approximate='exact',
                 display_label=None, label_map=None,
                 note='', hints=''):
        """
        The :class:`Aggregate` distribution class manages creation and calculation of aggregate distributions.
        It allows for very flexible creation of Aggregate distributions. Severity
        can express a limit profile, a mixed severity or both. Mixed frequency types share
        a mixing distribution across all broadcast terms to ensure an appropriate inter-
        class correlation.

        :param name:            name of the aggregate
        :param exp_el:          expected loss or vector
        :param exp_premium:     premium volume or vector  (requires loss ratio)
        :param exp_lr:          loss ratio or vector  (requires premium)
        :param exp_en:          expected claim count per segment (self.n = total claim count)
        :param exp_attachment:  occurrence attachment; None indicates no limit clause, which is treated different
                                from an attachment of zero.
        :param exp_limit:       occurrence limit
        :param sev_name:        severity name or sev.BUILTIN_SEV or meta.var agg or port or similar or vector or matrix
        :param sev_a:           scipy stats shape parameter
        :param sev_b:           scipy stats shape parameter
        :param sev_mean:        average (unlimited) severity
        :param sev_cv:          unlimited severity coefficient of variation
        :param sev_loc:         scipy stats location parameter
        :param sev_scale:       scipy stats scale parameter
        :param sev_xs:          xs and ps must be provided if sev_name is (c|d)histogram, xs are the bucket break points
        :param sev_ps:          ps are the probability densities within each bucket; if buckets equal size no adjustments needed
        :param sev_wt:          weight for mixed distribution
        :param sev_lb:          lower bound for severity (length of sev_lb must equal length of sev_ub and weights)
        :param sev_ub:          upper bound for severity
        :param sev_conditional: if True, severity is conditional, else unconditional.
        :param sev_signed:      if True the severity is signed (never clamps its
                                negative support; a profit is a negative loss).
                                Set by the ``ssev`` DecL keyword. Orthogonal to
                                ``value_type``. ``dsev`` with a negative atom
                                self-signs regardless of this flag.
        :param sev_pick_attachments:  if not None, a list of attachment points to define picks
        :param sev_pick_losses:  if not None, a list of losses by layer
        :param occ_reins:       layers: share po layer xs attach or XXXX
        :param occ_kind:        ceded to or net of
        :param freq_name:       name of frequency distribution
        :param freq_a:          cv of freq dist mixing distribution
        :param freq_b:          claims per occurrence (delaporte or sig), scale of beta or lambda (Sichel)
        :param freq_zm:         True/False zero modified flag
        :param freq_p0:         if freq_zm, provides the modified value of p0; default is nan
        :param agg_reins:       layers
        :param agg_kind:        ceded to or net of
        :param value_type:      ``'loss'`` (default) or ``'payoff'``; the DecL
                                orientation suffix sets ``'payoff'`` (more is
                                better) for an asset-return / direct-payoff
                                primitive. Inert for the distribution, consumed
                                at the pricing layer (the dual distortion). A
                                premium-minus-loss position is the separate
                                :class:`PnL` veneer (see ``make_pnl``), not a
                                value on the aggregate.
        :param approximate:     ``'exact'`` (default), ``'sgamma'`` or ``'slognorm'``.
                                When not ``'exact'``, the freq x sev convolution is
                                replaced at construction by a single continuous
                                severity fitted to the aggregate's first three
                                moments (method of moments: shifted gamma / shifted
                                lognormal, with normal as the symmetric limit and a
                                reflected fit for left skew), carried on a fixed
                                frequency of 1. The original program is preserved in
                                ``note``. Incompatible with occurrence reinsurance
                                (which acts pre-convolution); aggregate reinsurance
                                rides along unchanged. Set by the ``approximate``
                                DecL keyword. The chosen kind is exposed on the
                                instance as the **attribute** ``self.approximation``
                                (``''`` when exact, else the kind) -- a distinct name
                                from the ``approximate()`` *method* (the MoM-surrogate
                                factory) so the two do not collide. See
                                dev/done/plan-approximate.md.
        :param display_label:   optional human display label (the DecL ``as
                                "..."`` clause); ``None`` falls back to ``name``.
                                Presentation only -- repr / exhibit titles prefer
                                it over ``name`` -- never an identity / reference
                                target. See dev/plan-decl-labels.md.
        :param occ_reins_label: optional per-occurrence-layer display labels (the
                                reins-clause ``as`` clause); a list parallel to
                                ``occ_reins`` (entries ``None`` where unlabeled), or
                                ``None``. Pooled into ``label_map['occ_reins']`` as
                                a sparse ``{layer_index: label}`` dict (read via
                                ``a.labels.occ_reins``); names the cession group /
                                leg rows in a P&L ledger.
        :param agg_reins_label: optional per-aggregate-layer display labels,
                                parallel to ``agg_reins`` (see ``occ_reins_label``;
                                pooled into ``label_map['agg_reins']``).
        :param note:            free-text note, from a ``note{...}`` clause
        :param hints:           raw ``hints{...}`` build-settings string
            (``key=value;`` form). Pure annotation here; the underwriter
            parses it on build (caller-supplied ``build()`` kwargs win).
        """

        # have to be ready for inputs to be in a list, e.g. comes that way from Pandas via Excel
        def get_value(v):
            if isinstance(v, list):
                return v[0]
            else:
                return v

        # class variables
        self.name = get_value(name)
        # for persistence, save the raw called spec via inspect; must call before
        # creating any other local variables.
        frame = inspect.currentframe()
        self._spec = dict(inspect.getargvalues(frame).locals)
        for n in ['frame', 'get_value', 'self']:
            if n in self._spec: self._spec.pop(n)

        # Method-of-moments approximation (the ``approximate`` DecL keyword). When
        # not ``'exact'`` the requested freq x sev aggregate is replaced, right
        # here at construction, by a single continuous severity fitted to its
        # first three moments and carried on a fixed frequency of 1. The fit needs
        # the theoretical aggregate moments, so build a throwaway exact copy
        # (analytic moments only -- no FFT) to read them, then rewrite the local
        # construction variables before the frequency/severity setup below. The
        # original spec was just captured into ``self._spec`` (so the object still
        # round-trips), and any ``pnl`` affine / aggregate reinsurance rides along
        # on the rewritten object unchanged. See dev/done/plan-approximate.md.
        #
        # Stored as ``self.approximation`` -- a *noun* attribute -- deliberately
        # NOT ``self.approximate``, which would shadow the ``approximate()``
        # method (the MoM-surrogate factory, the parity-partner of
        # ``Portfolio.approximate``) on every instance. Falsey (``''``) for an
        # exact freq x sev convolution, else the fit kind, so ``if
        # a.approximation:`` reads as "is this object a moment-match surrogate?".
        # The DecL keyword and the ``approximate=`` kwarg / spec key are
        # unchanged; ``_spec`` maps ``'' -> 'exact'`` on the way out (round-trip).
        self.approximation = '' if approximate == 'exact' else approximate
        # Structured record of the method-of-moments fit, populated below when
        # ``approximate != 'exact'``. Kept here (not folded into ``note``) so the
        # human-readable description can be composed *lazily* -- the original
        # ``program`` text is set by the build path only after ``__init__``
        # returns, so it is not yet visible at construction. See
        # ``_approx_description`` and the note finalisation in ``Underwriter``.
        self._approx_fit = None
        if approximate not in ('exact', 'sgamma', 'slognorm'):
            raise ValueError(
                f"approximate must be 'exact', 'sgamma' or 'slognorm', "
                f"not {approximate!r}")
        if approximate != 'exact':
            if occ_reins is not None:
                raise ValueError(
                    f"{self.name}: approximate is incompatible with occurrence "
                    "reinsurance (the method-of-moments fit bypasses the "
                    "per-occurrence convolution); use aggregate reinsurance instead.")
            _orig = Aggregate(**{**self._spec, 'approximate': 'exact'})
            _m, _cv, _sk = _orig.agg_m, _orig.agg_cv, _orig.agg_skew
            _fit = _approximate_sev_kwargs(_m, _cv, _sk, approximate)
            # frequency -> fixed 1; exposure -> a single deterministic claim
            freq_name, freq_a, freq_b, freq_zm, freq_p0 = 'fixed', 0.0, 0.0, False, np.nan
            exp_en, exp_el, exp_premium, exp_lr = 1, 0.0, 0.0, 0.0
            exp_attachment, exp_limit = None, np.inf
            occ_reins, occ_kind = None, ''
            # severity -> the fitted continuous distribution
            sev_name = _fit['sev_name']
            sev_a = _fit.get('sev_a', np.nan)
            sev_b = 0.0
            sev_mean, sev_cv = 0.0, 0.0
            sev_loc = _fit['sev_loc']
            sev_scale = _fit['sev_scale']
            sev_xs = sev_ps = None
            sev_wt = 1.0
            sev_signed = bool(_fit.get('sev_signed', False))
            sev_reflect = bool(_fit.get('sev_reflect', False))
            # Record the fit for a lazy, program-aware description (rendered in
            # ``info`` and folded into ``note`` once ``self.program`` is set).
            # ``note`` is deliberately left as the user's pure note here.
            self._approx_fit = dict(
                kind=approximate, sev_name=sev_name, sev_a=sev_a,
                sev_loc=sev_loc, sev_scale=sev_scale, m=_m, cv=_cv, skew=_sk)

        logger.debug(
            f'Aggregate.__init__ | creating new Aggregate {self.name}')
        # Composition: an Aggregate *has* a frequency model, not *is* one.
        # ``Frequency(...)`` dispatches via ``__new__`` to the correct
        # ``Frequency<Kind>`` subclass.
        self.frequency = Frequency(
            get_value(freq_name), get_value(freq_a), get_value(freq_b),
            get_value(freq_zm), get_value(freq_p0))
        # Spec pass through from constructor arguments
        self.note = note
        # Exposure premium / loss ratio, retained for the P&L path: a ``pnl``
        # wrapping this engine reads ``exp_premium`` as the *technical* premium
        # (``inherit premium``), and a Portfolio accumulates it across units. 0.0
        # for a claims / loss exposure or an approximated aggregate (no premium).
        # Held as passed (scalar or per-component vector); consumers reduce with
        # ``np.sum``. See dev/plan-pnl-engine-source.md.
        self.exp_premium = exp_premium
        self.exp_lr = exp_lr
        # Raw `hints{...}` settings string; consumed by the underwriter build
        # path (caller-wins merge), retained here for round-tripping / repr.
        self.hints = hints
        self.program = ''  # can be set externally
        self.occ_reins = occ_reins
        self.occ_kind = occ_kind
        self.agg_reins = agg_reins
        self.agg_kind = agg_kind
        # Object-level display label + interior label_map (exposure / layer /
        # severity clause / cessions). Presentation only -- repr / exhibit titles
        # and the exhibit ``renamer`` read these; ``name`` stays the identity /
        # reference handle. See dev/done/plan-labels.md ([DecL-Labels-Everywhere]).
        self._init_labels(display_label=display_label, label_map=label_map)
        # The per-layer cession labels arrive as spec lists parallel to
        # ``occ_reins`` / ``agg_reins`` (the unparser round-trips those keys)
        # but POOL into ``label_map`` as sparse ``{layer_index: label}`` dicts
        # -- read ``a.labels.occ_reins[0]`` -- so ``labels`` is the complete
        # interior-label surface: one home, no parallel label attributes.
        for site, layer_labels in (('occ_reins', occ_reins_label),
                                   ('agg_reins', agg_reins_label)):
            if layer_labels:
                d = {i: lbl for i, lbl in enumerate(layer_labels)
                     if lbl is not None}
                if d:
                    self.label_map[site] = d
        self.sev_pick_attachments = sev_pick_attachments
        self.sev_pick_losses = sev_pick_losses

        # Grid + runtime config (set by update / update_work)
        self.figure = None
        self.xs = None
        self.bs = 0
        self.log2 = 0
        self.padding = 0
        # Signed-support / output-window state (set by update_work). The
        # defaults reproduce the non-negative, zero-based grid exactly:
        #   x_min == 0   -> output window starts at 0 (no output roll)
        #   i0 == 0      -> severity has no negative buckets (no input roll)
        #   xs_sev == xs -> severity and output share the grid
        # See dev/plan-negative-x-agg.md (F1 negative-x, F2 output window).
        self.x_min = 0.0
        self.x_max = None
        self.i0 = 0           # index of physical 0 in the severity array
        self.xs_sev = None    # severity discretisation grid (may differ from xs)
        self._bs_window_df = None   # inspectable bucket/window estimator summary
        self._bs_clip = None        # structured far-tail clip report (item 6) or None
        self._bs_raw = None         # pre-dyadic-round bs (unset for the multi-method agg sizer)
        # F1 opt-in: when True the severity keeps its negative support (the
        # layering clamp ``x<0 -> 0`` is bypassed). Default False preserves the
        # established non-negative behaviour. Opt-in wiring is pending a design
        # decision (DecL keyword vs flag vs dsev auto-detect).
        self._signed_sev = False
        # Sign convention: how the variable is read. Inert for the
        # distribution itself; consumed at the pricing/distortion layer
        # (actuarial loss orientation). ``pnl`` sets ``payoff``. See plan §5.5.
        # Canonical role boolean; the label string is resolved at display time
        # (value_type_label) so a [labels] config relabel never moves the role.
        self._is_loss_value = value_type_role(value_type)
        self.validation_eps = get_settings().validation.eps
        self.sev_calc = ""
        self.discretization_calc = ""
        self.normalize = ""

        # Exposure / mixture outputs (filled by broadcasting below)
        self.en = None   # per-component frequency (e.g. for a limit profile)
        self.n = 0       # total frequency
        self.attachment = None
        self.limit = None
        self.sevs = None

        # Computed densities (set by update_work)
        self.sev_density = None
        self.agg_density = None
        self.ftagg_density = None
        self.fzapprox = None
        self._density_df = None
        self._sev_density_df = None

        # Empirical moment estimates (set by update_work; consumed by q / tvar)
        self.est_m = 0
        self.est_cv = 0
        self.est_sd = 0
        self.est_var = 0
        self.est_skew = 0
        self.est_sev_m = 0
        self.est_sev_cv = 0
        self.est_sev_sd = 0
        self.est_sev_var = 0
        self.est_sev_skew = 0

        # Cached lazy functions (built on demand)
        self._valid = None
        # GridDistribution views over the aggregate and severity grids; own the
        # var/tvar kernel cache. Rebuilt (set None -> lazily) when update runs.
        self._dist = None
        self._sev_dist = None
        self._cdf = None
        self._pdf = None
        self._sev = None

        # Reinsurance state (set by apply_occ_reins / apply_agg_reins).
        # The exact (EX) reporting path reads the ceder/netter step
        # functions retained here; the per-stage reins frames
        # (``reins_density_df``, ``reins_stats_df``, ``reins_summary_df``)
        # are rebuilt lazily and cached in the underscore members below.
        self.occ_netter = None
        self.occ_ceder = None
        self.agg_netter = None
        self.agg_ceder = None
        self.sev_density_ceded = None
        self.sev_density_net = None
        self.sev_density_gross = None
        self.agg_density_ceded = None
        self.agg_density_net = None
        self.agg_density_gross = None
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_view_stats_cache = None
        self._reins_describe = None
        # rebucketing scheme for reins net/ceded distributions; set the backing
        # field directly (the setter clears the caches just initialised above)
        self._reins_bucket = reins_bucket if reins_bucket is not None else get_settings().discretization.reins_bucket
        if self._reins_bucket not in ('linear', 'nearest'):
            raise ValueError(
                f"reins_bucket must be 'linear' or 'nearest', not {self._reins_bucket!r}")
        # rebucketing scheme for discrete-severity atoms (dsev/dhistogram/fixed);
        # set the backing field directly (no caches to clear at construction)
        self._dsev_bucket = dsev_bucket if dsev_bucket is not None else get_settings().discretization.dsev_bucket
        if self._dsev_bucket not in ('linear', 'nearest'):
            raise ValueError(
                f"dsev_bucket must be 'linear' or 'nearest', not {self._dsev_bucket!r}")

        # ``stats_df`` is pre-created inside each broadcasting arm below once
        # ``n_components`` is known; see ``_init_stats_df``.
        ma = MomentAggregator(self.frequency.freq_moms)

        # overall freq CV with common mixing
        mix_cv = self.frequency.freq_a

        # broadcast arrays: force answers all to be arrays (?why only these items?!)
        if not isinstance(exp_el, Iterable):
            exp_el = np.array([exp_el])
        if not isinstance(sev_wt, Iterable):
            sev_wt = np.array([sev_wt])
        if not isinstance(sev_lb, Iterable):
            sev_lb = np.array([sev_lb])
        if not isinstance(sev_ub, Iterable):
            sev_ub = np.array([sev_ub])

        # counter to label components
        r = 0
        # broadcast together and create container for the severity distributions
        if np.sum(sev_wt) == len(sev_wt):
            # do not perform the exp / sev product, in this case
            # broadcast all exposure and sev terms together
            exp_el, exp_premium, exp_lr, en, attachment, limit, \
                sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, \
                sev_wt, sev_lb, sev_ub = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit,
                                    sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                                    sev_wt, sev_lb, sev_ub)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            all_arrays = zip(exp_el, exp_premium, exp_lr, en, attachment, limit,
                             sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                             sev_wt, sev_lb, sev_ub)
            self.en = en
            self.attachment = attachment
            self.limit = limit
            # these all have the same length because have been broadcast
            n_components = len(exp_el)
            logger.debug('Aggregate.__init__ | Broadcast/align: exposures + severity = %d exp = '
                         '%d sevs = %d componets', len(exp_el), len(sev_a), n_components)
            self.sevs = np.empty(n_components, dtype=type(Severity))
            # limit-profile arm: weights all 1, single severity per exposure
            # row → mixture component ``m`` is trivially 0; ``e`` indexes the
            # broadcast exposure rows.
            self._init_stats_df([f'e{e_idx}.m0' for e_idx in range(n_components)])

            # perform looping creation of severity distribution
            # in this case wts are all 1, so no need to broadcast
            for _el, _pr, _lr, _en, _at, _y, _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _swt, _slb, _sub in all_arrays:
                assert _swt==1, 'Expect weights all equal to 1'

                # WARNING: note sev_xs and sev_ps are NOT broadcast
                self.sevs[r] = Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                        _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect)
                sev1, sev2, sev3 = self.sevs[r].moms()

                # input claim count trumps input loss
                if _en > 0:
                    _el = _en * sev1
                elif _el > 0:
                    _en = _el / sev1
                # neither of these options can be triggered, by a dfreq dsev, for example.

                # if premium compute loss ratio, if loss ratio compute premium
                if _pr > 0:
                    _lr = _el / _pr
                elif _lr > 0:
                    _pr = _el / _lr

                # for empirical freq claim count entered as -1
                if _en < 0:
                    _en = np.sum(self.frequency.freq_a * self.frequency.freq_b)
                    _el = _en * sev1

                # scale for the mix - OK because we have split the exposure and severity components
                _pr *= _swt
                _el *= _swt
                # _lr *= _swt  ?? seems wrong
                _en *= _swt

                self._record_component(self._comp_cols[r], ma, _at, _y, _scv,
                                       _en, _el, _pr, _lr,
                                       mix_cv, sev1, sev2, sev3)
                r += 1

        else:
            # perform exp / sev product; but there is only one severity distribution
            # it could be a mixture - in which case we need to convert to en input (not loss)
            # and potentially re-weight for excess covers.
            # broadcast exposure terms (el, epremium, en, lr, attachment, limit) and sev terms (sev_) separately
            # then we take an "outer product" of the two parts...
            exp_el, exp_premium, exp_lr, en, attachment, limit = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit)
            sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt, sev_lb, sev_ub = \
                np.broadcast_arrays(sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                                    sev_wt, sev_lb, sev_ub)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            exp_arrays = [exp_el, exp_premium, exp_lr, en, attachment, limit]
            sev_arrays = [sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_lb, sev_ub]
            n_components = len(exp_el) * len(sev_name)
            self.en = np.empty(n_components, dtype=float)
            self.attachment = np.empty(n_components, dtype=float)
            self.limit = np.empty(n_components, dtype=float)
            # all broadcast arrays have the same length, hence:
            logger.debug(
                f'Aggregate.__init__ | Broadcast/product: exposures x severity = {len(exp_el)} x {len(sev_name)} '
                f'=  {n_components}')
            self.sevs = np.empty(n_components, dtype=type(Severity))
            # mixture-product arm: outer exposure × inner severity-mixture
            # gives a 2-D component grid; labels carry both indices.
            _n_exp = len(exp_el)
            _n_mix = len(sev_name)
            self._init_stats_df([
                f'e{e_idx}.m{m_idx}'
                for e_idx in range(_n_exp)
                for m_idx in range(_n_mix)
            ])

            # Ground-up mixture components are needed only to reweight the
            # mixture under an excess-of attachment (the ``sf(_at)`` call
            # below). Skip the constructions entirely when no exposure row
            # carries a positive attachment — saves one ``Severity`` per
            # mixture component on the common no-excess path.
            need_gup = any(
                _at is not None and _at > 0 for _at in attachment
            )
            gup_sevs = []
            if need_gup:
                for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, _swt in zip(*sev_arrays, sev_wt):
                    gup_sevs.append(Severity(_sn, 0, np.inf, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                             _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect))

            # perform looping creation of severity distribution
            for e_idx, (_el, _pr, _lr, _en, _at, _y) in enumerate(zip(*exp_arrays)):
                # adjust weights for excess coverage
                sev_wt0 = sev_wt.copy()
                # attachment can be None, and that needs to percolate through to Severity
                if _at is not None and _at > 0:
                    w1 = sev_wt0 * np.array([s.sf(_at) for s in gup_sevs])
                    sev_wt0 = w1 / w1.sum()

                # store actual sevs in a group (all are also appended to self.sevs) so we can compute the expected value
                # weight still irrelevant; but pull in layer and attaches which must vary for it to be meaningful
                actual_sevs = []
                for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, _swt in zip(*sev_arrays, sev_wt):
                    actual_sevs.append(Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                                _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect))

                # now we need to figure the severity across the mixture for this particular layer and  attach
                moms = []
                for s in actual_sevs:
                    # just return the first moment
                    moms.append(s.moms())

                # component mean (corresponding to the outside loop) can now be computed
                component_mean = (np.nan_to_num(np.array([m[0] for m in moms])) * sev_wt0).sum()

                # figure claim count if not entered, for the group (at this point we have not weighted down)
                # this forces subsequent calcuations to use (correct) en weighting even if premium or loss are
                # entered
                logger.info('%s xs %s, component_mean = %s, %s',
                            _y, _at, component_mean, [m[0] for m in moms])
                if _en == 0:
                    _en = _el / component_mean

                # for cases where a mixture component has no losses in the layer
                # usually because of underflow.
                zero = None

                # break up the total claim count into parts and add sevs to self.sevs
                # need the first variables for sev statistics
                for m_idx, (_sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, s, _swt, (sev1, sev2, sev3)) in \
                        enumerate(zip(*sev_arrays, actual_sevs, sev_wt0, moms)):

                    # store the severity
                    if np.isnan(sev1):
                        if zero is None:
                            zero = Severity('dhistogram', 0, np.inf, 0, 0, 0, 0, 0, 0, [0], [1], 0, np.inf, 0, False)
                        # replace this component with the zero distribution
                        # ignore the (small) weights that are being ignored
                        self.sevs[r] = zero
                        _sn = 'dhistogram'
                        logger.info('%s xs %s on %s x (%s, %s, %s, %s, %s) + %s '
                                    ' | %s < X le %s '
                                    'component has sev=(%s, %s, %s), '
                                    ' weight = %s; replacing with zero.',
                                    _y, _at, _ssc, _at, _sm, _scv, _sa, _sb, _sloc,
                                    _slb, _sub, sev1, sev2, sev3, _swt)
                        sev1 = sev2 = sev3 = 0.0
                    else:
                        self.sevs[r] = s

                    # input claim count, figure total loss for the component
                    if _en > 0:
                        _el = _en * sev1
                    elif _en < 0:
                        # for empirical freq claim count entered as -1
                        _en = np.sum(self.frequency.freq_a * self.frequency.freq_b)
                        _el = _en * sev1
                    else:
                        logger.info('%s xs %s on %s x (%s, %s, %s, %s, %s) + %s '
                                    ' | %s < X le %s has '
                                    '_en = %s. Adjusting el to 0.',
                                    _y, _at, _ssc, _at, _sm, _scv, _sa, _sb, _sloc,
                                    _slb, _sub, _en)
                        _el = 0.

                    # if premium compute loss ratio, if loss ratio compute premium
                    if _pr > 0:
                        _lr = _el / _pr
                    elif _lr > 0:
                        _pr = _el / _lr

                    # scale for the mix - OK because we have split the exposure and severity components
                    _pr0 = _pr * _swt
                    _el0 = _el * _swt
                    _en0 = _en * _swt

                    self._record_component(f'e{e_idx}.m{m_idx}', ma, _at, _y, _scv,
                                           _en0, _el0, _pr0, _lr,
                                           mix_cv, sev1, sev2, sev3)

                    self.en[r] = _en0
                    self.attachment[r] = _at
                    self.limit[r] = _y

                    r += 1

        # average exp_limit and exp_attachment — weighted by per-component
        # frequency mean. Sourced from the stats_df columns populated by the
        # broadcast loop above. ``stats_df`` is now all-float, so the casts
        # that this block used to need are gone.
        _comp_cols = self._comp_cols
        _comp_limit = self.stats_df.loc[('meta', 'limit'), _comp_cols]
        _comp_attach = self.stats_df.loc[('meta', 'attachment'), _comp_cols]
        _comp_freq = self.stats_df.loc[('freq', 'ex1'), _comp_cols]
        avg_limit = float(np.sum(_comp_limit * _comp_freq) / ma.tot_freq_1)
        avg_attach = float(np.sum(_comp_attach * _comp_freq) / ma.tot_freq_1)

        # store answer for total
        tot_prem = float(self.stats_df.loc[('meta', 'prem'), _comp_cols].sum())
        tot_loss = float(self.stats_df.loc[('meta', 'el'), _comp_cols].sum())
        # GROSS basis: ``tot_loss`` here is the theoretical loss before
        # ``update_work`` applies any reinsurance, so ``lr`` is a gross loss
        # ratio -- do not recompute it against a net/ceded loss.
        if tot_prem > 0:
            lr = tot_loss / tot_prem
        else:
            lr = np.nan

        # Write the post-loop totals directly into ``stats_df``: per-component
        # weights, then ``mixed`` and ``independent`` columns (theoretical
        # moments + meta).
        freq_ex1 = self.stats_df.loc[('freq', 'ex1'), _comp_cols]
        self.stats_df.loc[('meta', 'wt'), _comp_cols] = (freq_ex1 / ma.tot_freq_1).values
        # mixed and independent totals
        _flat_names = MomentAggregator.column_names()
        for _col, _remix in (('mixed', True), ('independent', False)):
            for _flat, _val in zip(_flat_names, ma.get_fsa_stats(total=True, remix=_remix)):
                self.stats_df.loc[_flat_col_to_stats_index(_flat), _col] = _val
            self.stats_df.loc[('meta', 'limit'), _col] = avg_limit
            self.stats_df.loc[('meta', 'attachment'), _col] = avg_attach
            self.stats_df.loc[('meta', 'sevcv_param'), _col] = 0
            self.stats_df.loc[('meta', 'el'), _col] = tot_loss
            self.stats_df.loc[('meta', 'prem'), _col] = tot_prem
            self.stats_df.loc[('meta', 'lr'), _col] = lr
            self.stats_df.loc[('meta', 'mix_cv'), _col] = (
                float(mix_cv) if np.isscalar(mix_cv) else np.nan
            )
            self.stats_df.loc[('meta', 'wt'), _col] = float(
                self.stats_df.loc[('meta', 'wt'), _comp_cols].sum()
            )

        self.n = ma.tot_freq_1
        # Pull the headline moments off the canonical stats_df mixed column.
        _mixed = self.stats_df['mixed']
        self.agg_m = float(_mixed[('agg', 'mean')])
        self.agg_cv = float(_mixed[('agg', 'cv')])
        self.agg_skew = float(_mixed[('agg', 'skew')])
        # variance and sd come up in exam questions. Derive them directly from
        # the second moment (var = ex2 - mean^2), NOT as mean*cv: at mean 0 the
        # CV is legitimately nan, which would poison sd = mean*cv -> nan for a
        # signed (P&L) aggregate whose sd is perfectly well defined. The clamp
        # absorbs fp dust when a symmetric ex2 - mean^2 lands slightly negative.
        self.agg_var = max(float(_mixed[('agg', 'ex2')]) - self.agg_m ** 2, 0.0)
        self.agg_sd = math.sqrt(self.agg_var)
        # severity exact moments
        self.sev_m = float(_mixed[('sev', 'mean')])
        self.sev_cv = float(_mixed[('sev', 'cv')])
        self.sev_skew = float(_mixed[('sev', 'skew')])
        self.sev_var = max(float(_mixed[('sev', 'ex2')]) - self.sev_m ** 2, 0.0)
        self.sev_sd = math.sqrt(self.sev_var)

    def _init_stats_df(self, comp_cols):
        """Pre-create the empty ``stats_df`` (NaN-filled).

        Called from each broadcasting arm of ``__init__`` once the per-
        component column labels are known. Columns are:

        * the broadcast components, named ``e{e}.m{m}`` where ``e`` is the
          exposure component (one per ``(claims|premium, limit xs attach)``
          row) and ``m`` is the severity-mixture component (one per weighted
          severity); the limit-profile arm always uses ``m=0``.
        * ``mixed`` / ``independent``: theoretical (subject / gross) totals.
        * ``empirical``: post-FFT empirical moments (the final, possibly
          after-reinsurance object).
        * ``after_occ``: empirical moments after the occurrence-reinsurance
          stage (populated in meta.4; scaffold here, NaN-filled).
        * ``occ_impact`` / ``agg_impact``: ``after_occ / mixed`` and
          ``empirical / after_occ`` ratios (scaffold).
        * ``gross_empirical``: subject empirical (the reinsurance validation
          hook from §1.3 of the plan; scaffold).
        * ``error``: noise-aware relative error of ``empirical`` vs
          ``mixed``.

        ``_record_component`` writes each component column inside the
        broadcast loop; the post-loop block writes ``mixed`` /
        ``independent``; ``update_work`` writes ``empirical`` and ``error``
        after the FFT; ``after_occ`` / ``occ_impact`` / ``agg_impact`` /
        ``gross_empirical`` are populated in meta.4 (reins reporting).

        ``stats_df`` is now an all-``float64`` frame: ``self.name`` lives on
        the attribute, so no string row is needed.
        """
        self._comp_cols = list(comp_cols)
        cols = self._comp_cols + [
            'mixed', 'independent', 'after_occ', 'empirical',
            'occ_impact', 'agg_impact', 'gross_empirical', 'error',
        ]
        self.stats_df = pd.DataFrame(
            np.nan, index=_STATS_ROW_INDEX, columns=cols, dtype=float,
        )

    def _record_component(self, col, ma, attach, layer, scv, en, el, prem, lr, mix_cv,
                          sev1, sev2, sev3):
        """Accumulate this component into ``ma`` and write its ``stats_df`` column.

        Called once per component from each of the two broadcasting arms of
        ``__init__``: the limit-profile arm (all weights == 1) and the
        mixture-product arm. Centralises which ``stats_df`` per-component
        column gets written so the two arms cannot drift apart.

        Parameters
        ----------
        col : str
            Per-component column label in ``stats_df``, of the form
            ``e{e}.m{m}`` (exposure × severity-mixture). Limit-profile arm
            uses ``m=0``.
        ma : MomentAggregator
            Accumulator collecting freq, sev, agg moments across all components.
        attach, layer : float
            Per-component attachment and layer height.
        scv : float
            Severity CV parameter for this component.
        en, el, prem, lr : float
            Per-component frequency, expected loss, premium, loss ratio, already
            scaled by the mixture weight by the caller.
        mix_cv : float
            Overall mixing-distribution CV (constant across rows).
        sev1, sev2, sev3 : float
            First three raw severity moments for this component.
        """
        ma.add_f1s(en, sev1, sev2, sev3)
        moments = ma.get_fsa_stats(total=False)
        # Write this component's data directly into the canonical ``stats_df``
        # column. Maps MA's flat moment names to the ``(component, measure)``
        # MultiIndex via ``_flat_col_to_stats_index``. ``('meta', 'wt')`` is
        # filled in by the post-loop block (it depends on total frequency).
        # ``mix_cv`` is only meaningful for true mixed-Poisson frequencies; for
        # ``dfreq`` (where ``freq_a`` is the discrete pmf array) it is NaN.
        self.stats_df.loc[('meta', 'limit'), col] = layer
        self.stats_df.loc[('meta', 'attachment'), col] = attach
        self.stats_df.loc[('meta', 'el'), col] = el
        self.stats_df.loc[('meta', 'prem'), col] = prem
        self.stats_df.loc[('meta', 'lr'), col] = lr
        self.stats_df.loc[('meta', 'sevcv_param'), col] = scv
        self.stats_df.loc[('meta', 'mix_cv'), col] = (
            float(mix_cv) if np.isscalar(mix_cv) else np.nan
        )
        for flat, val in zip(MomentAggregator.column_names(), moments):
            self.stats_df.loc[_flat_col_to_stats_index(flat), col] = val

    # ================================================================
    # Repr / info / help — string and HTML representations
    # ================================================================

    # ``display_name`` / ``_title_name`` come from ``LabeledMixin`` (the shared
    # label surface); ``name`` stays the identity handle. See dev/plan-labels.md.

    def __repr__(self):
        """
        String version of _repr_html_
        :return:
        """
        return f'{self.display_name}, {super(Aggregate, self).__repr__()}'

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        s = [self.info]
        with pd.option_context('display.width', 200,
                               'display.max_columns', 15,
                               'display.float_format', lambda x: f'{x:,.5g}'):
            # get it on one row
            s.append(str(self.summary_df))
        # s.append(super().__repr__())
        return '\n'.join(s)

    def help(self, regex, lod='short', values='short', fmt='auto'):
        """
        Lookup help on methods and properties matching ``regex``.

        Thin wrapper over :func:`aggregate.utilities.agg_help` — the free
        function is prefixed to avoid shadowing Python's builtin ``help`` at
        module / package scope. Three orthogonal axes: ``lod``
        (``'terse'|'short'|'all'``) controls how much docstring is shown;
        ``values`` (``'none'|'short'|'all'``) how much of each value or
        no-argument call result (a ``DataFrame`` / ``Series`` is headed to 5
        rows under ``'short'``); ``fmt`` (``'auto'|'text'|'ansi'|'html'``) the
        render target (``auto`` = ANSI in Jupyter, plain text in a terminal).
        """
        agg_help(self, regex, lod=lod, values=values, fmt=fmt)

    def _approx_description(self):
        """One-line description of the method-of-moments fit, or ``''`` if none.

        Renders the *original* program together with the fitted family and its
        parameters, e.g.::

            <program>  approximated by sgamma: gamma(loc=.., scale=.., a=..), m=.. cv=.. skew=..

        Composed lazily (not at construction) because the original ``program``
        text is assigned by the build path only after ``__init__`` returns; the
        fit itself is captured in :attr:`_approx_fit`. Returns ``''`` for an
        ordinary (``exact``) aggregate, or when no fit was recorded.

        Returns
        -------
        str
        """
        fit = getattr(self, '_approx_fit', None)
        if not fit:
            return ''
        prog = self.pprogram if self.program else self.name
        parts = [f"loc={fit['sev_loc']:.6g}", f"scale={fit['sev_scale']:.6g}"]
        a = fit.get('sev_a')
        if a is not None and np.isfinite(a):
            parts.append(f"a={a:.6g}")
        params = ', '.join(parts)
        return (f"{prog}  approximated by {fit['kind']}: {fit['sev_name']}({params}), "
                f"m={fit['m']:.6g} cv={fit['cv']:.6g} skew={fit['skew']:.6g}")

    def _spec_hash(self):
        """Display-only 8-hex id of the canonical spec.

        Computed on the fly from ``self._spec`` (machine-independent md5,
        first 8 hex, matching ``Distortion.id()``); no stored attribute, no
        timestamp.
        """
        blob = json.dumps(self._spec, sort_keys=True, default=str)
        return hashlib.md5(blob.encode('utf-8')).hexdigest()[:8].upper()

    @property
    def info(self):
        """Fixed-layout multi-line summary string.

        Every row is always present, in the same order, for every
        ``Aggregate``; a value that is not (yet) available -- e.g. the grid
        block before ``update`` -- renders as ``n/a``. The row catalogue and
        value enumerations are documented in ``dev/info-strings.rst``. Shares
        the label/value convention (:func:`aggregate.constants.info_row`)
        with ``Portfolio`` and ``Distortion``.
        """
        updated = self.bs > 0
        n_sev = len(self.sevs)
        if n_sev == 1:
            sv = self.sevs[0]
            sev_desc = f'{sv.long_name}, {sv.support_description}.'
        else:
            sev_desc = f'{n_sev} components'
        if updated:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1 / self.bs)}'
        else:
            bss = INFO_NA
        # premium / expected loss / loss ratio: populated when a premium is
        # known (the DecL exposure clause states one) and the object is updated.
        # ``P(loss)`` is a P&L concept and lives on the :class:`PnL` veneer.
        prem = float(self.stats_df.loc[('meta', 'prem'), 'mixed'])
        e_loss = None
        p_loss = INFO_NA
        if updated and self.agg_density is not None:
            e_loss = float(self.est_m)
        rows = [
            ('aggregate object name', self.name),
            ('value_type', self.value_type),
            ('claim count', f'{self.n:,.3f}'),
            ('frequency distribution', self.frequency.freq_name),
            ('severity distribution', sev_desc),
            ('approximate', getattr(self, 'approximation', '') or 'exact'),
            ('bs', bss),
            ('log2', self.log2 if updated else INFO_NA),
            ('padding', self.padding if updated else INFO_NA),
            ('sev_calc', self.sev_calc if updated else INFO_NA),
            ('dsev_bucket', self.dsev_bucket),
            ('normalize', self.normalize if updated else INFO_NA),
            ('x_min', f'{self.x_min:,.6g}' if updated else INFO_NA),
            ('x_max', f'{self.x_max:,.6g}'
             if updated and self.x_max is not None else INFO_NA),
            ('premium', f'{prem:,.6g}' if prem > 0 else INFO_NA),
            ('expected loss', f'{e_loss:,.6g}' if e_loss is not None else INFO_NA),
            ('loss ratio', f'{e_loss / prem:.1%}'
             if prem > 0 and e_loss is not None else INFO_NA),
            ('P(loss)', p_loss),
            ('validation_eps', self.validation_eps),
            ('reinsurance', self.reins_kinds.lower()),
            ('occurrence reinsurance', self._reins_description('occ').lower()),
            ('aggregate reinsurance', self._reins_description('agg').lower()),
            ('validation', self.validation_explanation),
        ]
        s = [info_row(label, value) for label, value in rows]
        # Tail report summary (frequency / severity / aggregate -- support and
        # per-side tail class); spec-only, so available before update.
        s.extend(_tail.describe_rows(self._tail_rows()))
        s.append(info_row('bounded', self.bounded))
        s.append(info_row('id', self._spec_hash()))
        return '\n'.join(s)

    @property
    def validation_explanation(self):
        """
        Long-narrative explanation of the validation result (str).

        The consistent narrative surface, mirroring ``tail_explanation`` /
        ``bs_explanation``. Validation is computed if needed.
        """
        return _validation.validation_explanation(self)

    def _html_info_blob(self):
        """Short HTML intro for ``_repr_html_`` -- identity, grid, *and a
        validation flag only when the object fails*.

        The headline tables (``summary_df`` / ``tail_df``) carry the risk view;
        this blob is the one-glance context. Validation is **silent on pass**
        (a clean or cleanly-reinsured subject says nothing) and surfaces a red
        block only on a genuine failure -- the same convention as :meth:`qd`.
        """
        s = [f'<h3>Aggregate object: {self._title_name}</h3>']
        s.append(f'<p>{self.frequency.freq_name} frequency distribution.')
        n = len(self.sevs)
        if n == 1:
            sv = self.sevs[0]
            s.append(f'Severity {sv.long_name} distribution, {sv.support_description}.')
        else:
            s.append(f'Severity with {n} components.')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.</p>')
        if self.agg_density is not None and not self._validation_passes():
            s.append('<p>Validation: <div style="color: #f00; font-weight:bold;">fails</div><pre>\n'
                     f'{self.validation_explanation}</pre></p>')
        return '\n'.join(s)

    def _text_info_blob(self) -> str:
        """Short plain-text intro (the text twin of :meth:`_html_info_blob`).

        Object identity, the frequency / severity families, and the realised
        grid -- the one-glance context :meth:`qd` prints above the headline
        ``summary_df`` / ``tail_df``. **No validation line** (the caller flags
        a failure separately, staying silent on a pass).
        """
        s = [f'Aggregate object: {self._title_name}',
             f'{self.frequency.freq_name} frequency distribution.']
        n = len(self.sevs)
        if n == 1:
            sv = self.sevs[0]
            s.append(f'Severity {sv.long_name} distribution, {sv.support_description}.')
        else:
            s.append(f'Severity with {n} components.')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.')
        return '\n'.join(s)

    def _validation_passes(self) -> bool:
        """Whether the object clears validation (clean *or* cleanly reinsured).

        ``True`` for a ``NOT_UNREASONABLE`` result and for ``REINSURANCE`` (the
        subject validated and reinsurance makes the moment audit n/a). Used by
        the display surfaces (:meth:`_html_info_blob`, :meth:`qd`) to stay silent
        on a pass and only flag a genuine failure.
        """
        r = self.valid
        return bool(r == Validation.NOT_UNREASONABLE or (r & Validation.REINSURANCE))

    def _repr_html_(self):
        """HTML view: short intro, the ``summary_df`` headline, the ``tail_df``
        return-period table, and a validation flag only on failure.
        """
        fmt = lambda x: f'{x:,.5g}'
        out = [self._html_info_blob(),
               '<h4>Summary</h4>',
               self.summary_df.to_html(float_format=fmt, na_rep='')]
        td = self.tail_df()
        if td is not None:
            out.append('<h4>Tail &mdash; return period (exact, not simulated)</h4>')
            out.append(td.to_html(float_format=fmt, na_rep=''))
        return '\n'.join(out)

    # ================================================================
    # Discretization, snap, update, FFT convolution
    # The 5-line FFT core (Mildenhall 2024, §2.2) lives in
    # ``_freq_sev_convolution`` below.
    # ================================================================

    def _signed_severity(self):
        """Whether the aggregate has signed (negative-support) severity.

        Returns ``True`` iff any component severity is signed -- declared with
        the ``ssev`` keyword (continuous) or a ``dsev`` with a negative atom,
        both recorded on ``Severity.signed`` at construction.

        This is the *convolution-grid* gate: it drives the negative-x severity
        layout (``i0``), the two-sided window in :meth:`_bs_window`, and
        ``_signed_sev``. It is **not** the public display/combine gate -- a
        ``pnl`` aggregate has an ordinary non-negative loss severity (so this is
        ``False``) but is signed at the aggregate level via the affine wrapper;
        see :meth:`_signed`.

        Returns
        -------
        bool

        Notes
        -----
        Signedness is a *parse-time property of the severity*, not a runtime
        flag -- which is what lets the analytic moments (and hence the auto
        window) be correct before any FFT, and is why there is **no**
        ``signed=`` override on ``update``: a built ``Severity`` is already
        clamped or not, and flipping it would require a deep rebuild. To change
        signedness, change the declaration (``sev`` / ``dsev`` / ``ssev``).
        See ``dev/plan-negative-x-agg.md``.
        """
        return any(getattr(s, 'signed', False)
                   for s in (self.sevs if self.sevs is not None else []))

    def _signed(self):
        """Whether the aggregate is signed (straddles 0) for display / combine.

        ``True`` when the severity itself reaches below 0
        (:meth:`_signed_severity`) -- a ``ssev`` continuous severity or a
        ``dsev`` with a negative atom. This is the gate read by plotting,
        two-sided quantiles, the signed-aware ``validation_df`` (SD instead of CV),
        and the :class:`Portfolio` combine. (A premium-minus-loss position is
        now the separate :class:`PnL` veneer, not a signed aggregate.)

        Returns
        -------
        bool
        """
        return self._signed_severity()

    def _severity_negative_buckets(self, bs):
        """Number of negative buckets the severity reaches (index of physical 0).

        Returns ``i0`` such that the severity discretisation grid
        ``xs_sev = (arange(N) - i0) * bs`` places physical 0 at index ``i0``
        and covers the severity's left tail. ``0`` for any severity supported
        on ``[0, inf)`` -- which keeps the non-negative path byte-for-byte
        unchanged. For a severity that reaches below 0 (e.g. ``norm``, a
        shifted distribution, or a ``dsev`` with negative atoms) it is the
        number of buckets from 0 down to the severity's effective lower
        endpoint.

        Parameters
        ----------
        bs : float
            Bucket size.

        Returns
        -------
        int
            ``max(0, ceil(-lo / bs))`` where ``lo`` is the smallest effective
            lower endpoint over the severity mixture components. A small
            tolerance keeps floating-point dust at exactly 0 from spuriously
            triggering signed mode.

        Notes
        -----
        ``lo`` is taken as the ``1e-12`` lower quantile of each *signed*
        component (a signed Severity has identity layering, so its ``ppf`` is
        the raw lower quantile), mirroring how the upper grid edge is sized from
        a high quantile. Any mass below the leftmost bucket is dropped and
        absorbed by the (optional) renormalisation in ``discretize`` -- at the
        ``1e-12`` level this is negligible.
        """
        # Only signed severities reach below 0; an unsigned component keeps the
        # clamp-at-0 layering (no negative buckets). i0 == 0 unless the
        # aggregate is signed, so the non-negative path is untouched.
        if not getattr(self, '_signed_sev', False):
            return 0
        if self.sevs is None or len(self.sevs) == 0:
            return 0
        los = []
        for sev in self.sevs:
            if not getattr(sev, 'signed', False):
                continue
            try:
                lo = float(sev.ppf(1e-12))   # identity layering -> raw quantile
            except Exception:  # pragma: no cover - defensive
                lo = 0.0
            if not np.isfinite(lo):
                lo = 0.0
            los.append(lo)
        lo = min(los) if los else 0.0
        if lo >= 0:
            return 0
        i0 = int(np.ceil(-lo / bs - 1e-9))
        # The severity's negative reach must fit inside the grid.
        N = len(self.xs)
        if i0 >= N:
            logger.warning(
                '%s: severity negative reach (%d buckets) exceeds grid size '
                '%d; clipping. Increase log2 or bs for a signed aggregate.',
                self.name, i0, N)
            i0 = N - 1
        return i0

    def discretize(self, sev_calc, discretization_calc, normalize):
        """
        Discretize the severity distributions and weight.

        ``sev_calc`` describes how the severity is discretize, see `Discretizing the Severity Distribution`_. The
        options are discrete=round, forward, backward or moment.

        ``sev_calc='continuous'`` (same as forward, kept for backwards compatibility) is used when
        you think of the resulting distribution as continuous across the buckets
        (which we generally don't). The buckets are not shifted and so :math:`Pr(X=b_i) = Pr( b_{i-1} < X \\le b_i)`.
        Note that :math:`b_{i-1}=-bs/2` is prepended.

        We use the discretized distribution as though it is fully discrete and only takes values at the bucket
        points. Hence, we should use `sev_calc='discrete'`. The buckets are shifted left by half a bucket,
        so :math:`Pr(X=b_i) = Pr( b_i - b/2 < X \\le b_i + b/2)`.

        The other wrinkle is the righthand end of the range. If we extend to np.inf then we ensure we have
        probabilities that sum to 1. But that method introduces a probability mass in the last bucket that
        is often not desirable (we expect to see a smooth continuous distribution, and we get a mass). The
        other alternative is to use endpoint = 1 bucket beyond the last, which avoids this problem but can leave
        the probabilities short. We opt here for the latter and normalize (rescale).

        ``discretization_calc`` controls whether individual probabilities are computed using backward-differences of
        the survival function or forward differences of the distribution function, or both. The former is most
        accurate in the right-tail and the latter for the left-tail of the distribution. We are usually concerned
        with the right-tail, so prefer `survival`. Using `both` takes the greater of the two esimates giving the best
        of both worlds (underflow makes distribution zero in the right-tail and survival zero in the left tail,
        so the maximum gives the best estimate) at the expense of computing time.

        Sensible defaults: sev_calc=discrete, discretization_calc=survival, normalize=True.

        :param sev_calc:  discrete=round, forward, backward, or continuous
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :return:
        """

        # Severity is discretised on ``xs_sev`` (physical 0 at index i0), which
        # equals ``self.xs`` on the default 0-based grid. ``i0 > 0`` means the
        # severity reaches below 0 (signed mode).
        xs_sev = self.xs_sev if self.xs_sev is not None else self.xs
        signed = self.i0 > 0

        if sev_calc == 'discrete' or sev_calc == 'round':
            # adj_xs = np.hstack((xs_sev - self.bs / 2, np.inf))
            # mass at the end undesirable. can be put in with reinsurance layer in spec
            # note the first bucket is negative
            adj_xs = np.hstack((xs_sev - self.bs / 2, xs_sev[-1] + self.bs / 2))
        elif sev_calc == 'forward' or sev_calc == 'continuous':
            adj_xs = np.hstack((xs_sev, xs_sev[-1] + self.bs))
        elif sev_calc == 'backward':
            adj_xs = np.hstack((xs_sev[0] - self.bs, xs_sev))  # , np.inf))
        elif sev_calc == 'moment':
            raise NotImplementedError(
                'Moment matching discretization not implemented. Embrechts says it is not worth it.')
            #
            # adj_xs = np.hstack((xs_sev, np.inf))
        else:
            raise ValueError(
                f'Invalid parameter {sev_calc} passed to discretize; options are discrete, continuous, or raw.')

        if not signed:
            # Non-negative severity: the first bucket must include all mass at
            # and below 0. Capture the whole left tail from -inf, exactly as
            # before (byte-for-byte unchanged on the default path).
            adj_xs[0] = -np.inf
        # Signed mode: the leftmost bucket is treated identically to every
        # other bucket (a finite ``xs_sev[0] - bs/2`` edge, set above) -- no
        # -inf catch. Any residual mass below it (<= 1e-12 by construction of
        # i0) is dropped and absorbed by the optional renormalisation below.

        # bed = bucketed empirical distribution. A signed Severity carries
        # identity layering, so its own cdf/sf are already un-clamped; an
        # unsigned component keeps the clamp-at-0 layering. So per-component
        # ``sev.cdf`` / ``sev.sf`` are correct on the signed grid with no special
        # casing (an unsigned component simply contributes 0 to negative
        # buckets). Occurrence reinsurance on a signed severity is out of scope
        # (plan §6).
        beds = []
        for sev in self.sevs:
            if (self.dsev_bucket == 'linear'
                    and sev.sev_kind in ('dhistogram', 'fixed')
                    and sev.exp_attachment is None):
                # Unlayered discrete severity: place the atoms on the grid with
                # the mean-preserving linear scatter, so the discretized first
                # moment equals Σ xₖ pₖ exactly. The default cdf-difference path
                # (below) snaps each atom to its nearest bucket (== 'nearest'),
                # biasing the mean by up to bs/2 per atom when atoms are off-grid
                # (empirical samples, non-integer bs). On-grid atoms give f == 0
                # so this reduces to nearest -- the dice / integer-bs case is
                # unchanged. ``sev.fz`` is the _DiscreteRV holding the validated,
                # sorted atoms (xk) and masses (pk). ``exp_attachment is None``
                # is the truly-unlayered test (a discrete sev always has a finite
                # ``detachment`` = max atom, so a ``== np.inf`` test never fires).
                # Layered discrete severities fall through to the cdf-diff path
                # (Phase 1; see dsev_bucket). The bed is indexed on ``xs_sev``,
                # so the scatter origin is the severity grid origin ``xs_sev[0]``
                # (== -i0·bs), NOT the output-window origin ``x_min`` -- the two
                # differ when the output window is forced wider than the atom
                # support (e.g. an explicit ``x_min`` below the smallest atom).
                appx = self._rebucket_to_grid(sev.fz.xk, sev.fz.pk,
                                              scheme='linear', origin=xs_sev[0])
            elif discretization_calc == 'both':
                # see comments: we rescale each severity...
                appx = np.maximum(np.diff(sev.cdf(adj_xs)), -np.diff(sev.sf(adj_xs)))
            elif discretization_calc == 'survival':
                appx = -np.diff(sev.sf(adj_xs))
                # beds.append(appx / np.sum(appx))
            elif discretization_calc == 'distribution':
                appx = np.diff(sev.cdf(adj_xs))
                # beds.append(appx / np.sum(appx))
            else:
                raise ValueError(
                    f'Invalid options {discretization_calc} to double_diff; options are density, survival or both')
            if normalize:
                beds.append(appx / np.sum(appx))
            else:
                beds.append(appx)
        return beds

    def snap(self, x):
        """
        Snap value x to the index of density_df, i.e., as a multiple of self.bs.

        :param x:
        :return:
        """
        ix = self.density_df.index.get_indexer([x], 'nearest')[0]
        return self.density_df.iloc[ix, 0]

    @property
    def value_type(self):
        """Sign convention for the variable: ``'loss'`` or ``'payoff'``.

        Records how the aggregate should be read -- actuarial **loss**
        ("more is worse", the default) vs. **payoff** / asset ("more is
        better"). It is **inert for the distribution itself**: density,
        moments, quantiles, deficit and plotting do not depend on it.
        It is consumed only when applying distortions / pricing (specified
        in the Portfolio plan and downstream pricing work), where a
        ``'payoff'`` object is negated / the dual distortion applied.

        The role is stored as a boolean (``_is_loss_value``); this getter
        returns the **configured label** for the role (``[labels]`` in the
        config, defaults ``'loss'`` / ``'payoff'``). The setter accepts the
        canonical tokens or the configured labels. Pricing code must branch
        on ``_is_loss_value``, never on the label text.

        See ``dev/plan-negative-x-agg.md`` §5.5.
        """
        return value_type_label(self._is_loss_value)

    @value_type.setter
    def value_type(self, v):
        self._is_loss_value = value_type_role(v)
        # The cached GridDistributions carry the orientation, so a post-build
        # role change must drop them; the next accessor rebuilds with the new
        # is_loss_value. (Until now only update() reset these caches.)
        self._dist = None
        self._sev_dist = None

    def make_pnl(self, consideration=None, *, gross=None, ceded=None,
                 expense_spec=None, gcn_economics=None, consideration_label=None,
                 loss_label=None):
        """Wrap this aggregate as the risky leg of a :class:`PnL` position.

        Object sugar delegating to the builders in
        :mod:`aggregate._pnl_builders`. Two construction modes:

        * **plain** -- ``make_pnl(consideration=C)``: a one-group ``sell``
          ledger over this aggregate's own density (for a reinsurance-bearing
          aggregate that is the *net* density -- what comes out of the
          aggregate).
        * **group ledger** -- ``make_pnl(gross=Pg, ceded=Pc)`` on a
          reinsurance-bearing aggregate: the per-atom Gross / Ceded / Net
          **group ledger** (:func:`~aggregate._pnl_builders.build_gcn_pnl`) --
          an aggregate-only cession books as a real ``buy`` group over the
          gross marginal; an occurrence program books over the net-of-occ
          marginal with the occ economics as constants (its risk transfer is
          the ``xpnl`` exhibit).

        Parameters
        ----------
        consideration : float, array-like, or callable, optional
            The plain-mode consideration, **signed** (``+`` received, ``-``
            paid); a vector sums to one book amount; a callable ``f(x)`` is a
            loss-sensitive consideration applied bucket-wise. Mutually
            exclusive with ``gross``/``ceded``.
        gross, ceded : float, optional
            The gross premium received and ceded premium paid (positive
            magnitudes) for the group-ledger view.
        consideration_label : str, optional
            Name for the premium leg (the DecL premium ``as`` clause);
            defaults to ``'consideration'`` (plain) / ``'premium'`` (ledger).
        loss_label : str, optional
            Name for the loss leg (the DecL engine ``as`` clause); defaults to
            ``'loss'``.

        Returns
        -------
        PnL
            **Always** a :class:`~aggregate.PnL` value object; the ledger form
            carries its resolved cession economics as ``pnl.economics``.

        Notes
        -----
        ``build('pnl NAME C premium less <body>')`` is sugar for
        ``build('agg NAME <body>').make_pnl(consideration=C)``.
        """
        from ._pnl_builders import build_plain_pnl, build_gcn_pnl
        if gross is not None or ceded is not None:
            if gross is None or ceded is None:
                raise ValueError(
                    'a Gross/Ceded/Net PnL needs both gross= and ceded= premiums.')
            if consideration is not None:
                raise ValueError(
                    'pass either consideration= or gross=/ceded=, not both.')
            if self.agg_reins is None and self.occ_reins is None:
                raise ValueError(
                    'the Gross/Ceded/Net view requires reinsurance on the risky '
                    'leg; the aggregate carries no occurrence / aggregate treaty.')
            return build_gcn_pnl(
                self, gross=gross, ceded=ceded, gcn_economics=gcn_economics,
                expense_spec=expense_spec,
                consideration_label=consideration_label,
                loss_label=loss_label, name=self.name,
                display_label=self.display_label)
        if consideration is None:
            raise ValueError(
                'PnL needs a consideration= (or gross=/ceded= for the '
                'Gross/Ceded/Net view).')
        return build_plain_pnl(
            self, consideration=consideration,
            consideration_label=consideration_label, loss_label=loss_label,
            expense_spec=expense_spec, name=self.name,
            display_label=self.display_label)

    def update(self, log2=16, bs=0, bucket_sizing_p=BUCKET_SIZING_P, debug=False,
               x_min='auto', x_max=None, window_convention=None, **kwargs):
        """
        Convenience function, delegates to update_work. Avoids having to pass xs. Also
        aliased as easy_update for backward compatibility.

        :param log2:
        :param bs:
        :param bucket_sizing_p: p value passed to the moment-window bucket sizer. If > 1 converted to 1 - 10**-p.
        :param debug:
        :param x_min: lower edge of the output window. ``'auto'`` (default)
          resolves to ``0`` for an ordinary non-negative aggregate (today's
          behaviour) and to ``None`` (automatic two-sided window) for a signed
          aggregate -- so a P&L declared with ``ssev`` / negative ``dsev`` just
          works from ``build``. ``None`` forces the automatic window; a number
          forces that origin (snapped to a multiple of ``bs``; may be negative).
        :param x_max: upper edge of the output window; informational, the grid
          length is fixed by ``log2``. Currently unused when ``x_min`` is given
          explicitly (the window is ``[x_min, x_min + (2**log2)*bs)``).
        :param window_convention: ``'loss'`` / ``'payoff'`` to override the sign
          convention orienting the automatic windowed placement (per-edge
          coverage and padding skew). ``None`` (default) derives it from
          ``value_type``. See ``dev/plan-bucket-window-2.md`` §1A.
        :param kwargs:  passed through to update
        :return:

        Signedness is carried by the severity declaration (``ssev`` /negative
        ``dsev``), not by an argument here -- see ``_signed``.
        """
        # Unified bucket + window estimator: runs the candidate sizing methods
        # (moment / exact_discrete / bounded_small), records them in the
        # inspectable ``self._bs_window_df``, and selects (see ``_bs_window``).
        # ``x_min='auto'`` lets the selected method choose the origin (0 for a
        # non-negative aggregate, a negative origin only when signed); a number
        # forces it; ``log2`` is a cap. The legacy non-negative ``moment`` path
        # reproduces the legacy 3-moment bucket sizing exactly, so ordinary
        # aggregates are unchanged.
        x_min_arg = None if (isinstance(x_min, str) and x_min == 'auto') else x_min
        bs, log2, x_min = self._bs_window(log2, bs, x_min_arg, bucket_sizing_p,
                                          window_convention=window_convention)
        N = 1 << log2
        # ``x_min`` is the convolution origin chosen by ``_bs_window``: 0 for an
        # ordinary aggregate; a negative origin when the severity is signed
        # (``ssev`` / negative-``dsev``), so the loss convolves on its genuine
        # signed grid.
        xs = x_min + np.arange(0, N, dtype=float) * bs
        return self.update_work(xs, debug=debug, x_min=x_min, x_max=x_max,
                                **kwargs)

    def update_work(self, xs, padding=1, sev_calc='discrete',
                    discretization_calc='survival', normalize=True, force_severity=False,
                    reins_bucket=None, dsev_bucket=None, debug=False, x_min=0, x_max=None):
        """
        Compute a discrete approximation to the aggregate density via FFT.

        See discretize for sev_calc, discretization_calc and normalize.

        Empirical-moment note: the aggregate raw moments -- and hence the
        empirical CV/skew shown in ``stats_df`` and ``validation_df`` -- are taken
        from a de-fuzzed *copy* of the FFT density (values below machine
        epsilon zeroed). Without this, sub-eps floating-point fuzz in far-tail
        buckets is amplified by ``x**3`` in the third moment and corrupts the
        empirical skew on wide grids: a symmetric distribution's skew can
        drift from ~1e-15 to ~1e-4 as log2 grows, purely from buckets the
        distribution never reaches. The fuzz is safe to drop because the FFT
        is exact up to rounding and the exact aggregate has no negative density
        even under aliasing, so any stray value is small. ``self.agg_density``
        is deliberately left as the raw FFT output (consistent with
        ``ftagg_density``); only the moment computation sees the cleaned copy.
        See the inline comment at the moment computation for full detail.

        Quick simple test with log2=13 update took 5.69 ms and _eff took 2.11 ms. So quicker
        but not an issue unless you are doing many buckets or aggs.

        :param xs: range of x values used to discretize
        :param padding: for FFT calculation
        :param sev_calc:  discrete=round, forward, backward, or continuous
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :param force_severity: make severities for plotting even when only the aggregate is requested
        :param reins_bucket: optional override of the net/ceded rebucketing scheme
               ('linear' or 'nearest'); defaults to the current ``self.reins_bucket``.
        :param dsev_bucket: optional override of the discrete-severity atom
               placement scheme ('linear' or 'nearest'); defaults to the current
               ``self.dsev_bucket``. See :attr:`dsev_bucket`.
        :param debug: run reinsurance in debug model if True.
        :param x_min: ``None`` requests an automatic two-sided output window
          (NYI in this stage -- treated as the grid origin ``xs[0]``);
          otherwise informational (the grid origin is read from ``xs[0]``).
        :param x_max: informational upper window edge.
        :return:
        """
        self._density_df = None  # invalidate
        self._sev_density_df = None
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_view_stats_cache = None
        self._reins_describe = None
        self._dist = None
        self._sev_dist = None
        self._valid = None
        self.sev_calc = sev_calc
        self.discretization_calc = discretization_calc
        self.normalize = normalize
        self.padding = padding
        if reins_bucket is not None:
            # validating setter; takes effect for the reins applied below
            self.reins_bucket = reins_bucket
        if dsev_bucket is not None:
            # validating setter; takes effect for the discretization below
            self.dsev_bucket = dsev_bucket
        self.xs = xs
        # bs is the grid step; xs[1]-xs[0] (not xs[1]) so a signed/offset grid
        # whose origin xs[0] != 0 still reports the correct bucket size.
        self.bs = xs[1] - xs[0]
        self.log2 = int(np.log2(len(xs)))
        # Output-window origin: the physical value at output index 0. For the
        # default 0-based grid this is 0 and all the offset machinery below is
        # inert (i0 == 0, j0 == 0 -> no rolls; xs_sev == xs).
        self.x_min = float(xs[0])
        self.x_max = float(xs[-1])

        # F1 -- negative-support severity. Resolve the signed opt-in (auto for
        # discrete severities with negative atoms; explicit flag otherwise),
        # then determine the severity's negative reach as a whole number of
        # buckets ``i0`` (the index of physical 0 in the severity array). Zero
        # for any severity supported on [0, inf), so the non-negative path is
        # untouched. The severity is then discretised on its own grid
        # ``xs_sev`` (physical 0 at index i0), which may differ from the output
        # grid ``xs`` (e.g. a tight far-from-0 output window).
        self._signed_sev = self._signed_severity()
        self.i0 = self._severity_negative_buckets(self.bs)
        self.xs_sev = (np.arange(len(xs), dtype=float) - self.i0) * self.bs

        # claim-count weighted severity vector (always computed; FFT is the only path)
        freq_ex1 = self.stats_df.loc[('freq', 'ex1'), self._comp_cols].values
        wts = freq_ex1 / freq_ex1.sum()
        if self.en.sum() == 0:
            self.en = freq_ex1
        self.sev_density = np.zeros_like(xs)
        beds = self.discretize(sev_calc, discretization_calc, normalize)
        for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
            self.sev_density += temp * w

        # adjust for picks if necessary
        if self.sev_pick_attachments is not None:
            logger.warning('Adjusting for picks.')
            self.sev_density = self.picks(self.sev_pick_attachments, self.sev_pick_losses)

        if force_severity == 'yes':
            # only asking for severity (used by plot)
            return

        # deal with per occ reinsurance
        if self.occ_reins is not None:
            if self.sev_density_gross is not None:
                # re-applying reins on an already-updated object: restore
                # gross sev so apply_occ_reins is idempotent
                self.sev_density = self.sev_density_gross
            self.apply_occ_reins(debug)

        self._freq_sev_convolution(padding)
        if self.n > 0:
            # zero-risk case has no aggregate to reinsure
            self.apply_agg_reins(debug)

        # Empirical severity moments from the discretised distribution.
        # Compute the raw moments once and derive (mean, cv, skew) from the
        # same MomentWrangler, so the ex123 rows and the mcvsk values are
        # mutually consistent.
        if self.sev_density is not None:
            # severity lives on xs_sev (== xs on the default grid)
            _mw = xsden_to_mwrangler(self.xs_sev, self.sev_density)
            sev_ex1, sev_ex2, sev_ex3 = _mw.noncentral
            self.est_sev_m, self.est_sev_cv, self.est_sev_skew = _mw.mcvsk
            # var/sd straight off the wrangler (var = central[1]), never via
            # mean*cv -- see the theoretical site above for the mean-0 rationale.
            self.est_sev_var = max(float(_mw.central[1]), 0.0)
            self.est_sev_sd = math.sqrt(self.est_sev_var)
        else:
            sev_ex1 = sev_ex2 = sev_ex3 = np.nan
            self.est_sev_m = np.nan
            self.est_sev_cv = np.nan
            self.est_sev_skew = np.nan
            self.est_sev_var = np.nan
            self.est_sev_sd = np.nan

        # Empirical aggregate moments from the FFT output.
        #
        # WHY a de-fuzzed *copy*: the raw inverse-FFT density carries
        # sub-machine-epsilon "fuzz" (tiny +/- values) in essentially every
        # bucket. In the plain mass sum this cancels (mass is conserved), but
        # the raw moments weight each bucket by ``x**k``, so on a wide grid the
        # far-tail fuzz at large ``x`` is amplified by ``x**3`` and corrupts
        # the empirical skew -- e.g. a symmetric die's skew drifts from ~1e-15
        # to ~1e-4 as log2 grows, purely from fuzz at buckets the distribution
        # never reaches. The fuzz is genuine fp noise: the FFT is exact up to
        # rounding and the exact aggregate has no negative density even under
        # aliasing (aliasing only wraps *positive* mass), so every stray value
        # is small and zeroing ``|x| < eps`` is safe and lossless. We do this
        # on a throwaway copy and deliberately leave ``self.agg_density`` as
        # the raw output (kept consistent with ``ftagg_density``); the curated
        # view ``density_df.p_total`` applies the identical ``remove_fuzz``
        # separately. (We cannot source the moments from ``density_df.p_total``
        # here: building ``density_df`` needs ``est_m``, computed just below.)
        agg_clean = remove_fuzz(self.agg_density)
        _mw = xsden_to_mwrangler(self.xs, agg_clean)
        agg_ex1, agg_ex2, agg_ex3 = _mw.noncentral
        self.est_m, self.est_cv, self.est_skew = _mw.mcvsk
        # var/sd straight off the wrangler (var = central[1]), never via
        # mean*cv -- correct for a mean-0 signed aggregate (see above).
        self.est_var = max(float(_mw.central[1]), 0.0)
        self.est_sd = math.sqrt(self.est_var)

        # Write empirical and error columns into the canonical stats_df.
        # This is the validation showpiece of Mildenhall 2024, §4.7:
        # theoretical (``mixed`` column) vs. empirical (FFT output).
        self.stats_df.loc[('sev', 'ex1'),  'empirical'] = sev_ex1
        self.stats_df.loc[('sev', 'ex2'),  'empirical'] = sev_ex2
        self.stats_df.loc[('sev', 'ex3'),  'empirical'] = sev_ex3
        self.stats_df.loc[('sev', 'mean'), 'empirical'] = self.est_sev_m
        self.stats_df.loc[('sev', 'cv'),   'empirical'] = self.est_sev_cv
        self.stats_df.loc[('sev', 'skew'), 'empirical'] = self.est_sev_skew
        self.stats_df.loc[('agg', 'ex1'),  'empirical'] = agg_ex1
        self.stats_df.loc[('agg', 'ex2'),  'empirical'] = agg_ex2
        self.stats_df.loc[('agg', 'ex3'),  'empirical'] = agg_ex3
        self.stats_df.loc[('agg', 'mean'), 'empirical'] = self.est_m
        self.stats_df.loc[('agg', 'cv'),   'empirical'] = self.est_cv
        self.stats_df.loc[('agg', 'skew'), 'empirical'] = self.est_skew

        # Defective-distribution check. The aggregate FFT loses mass off the
        # right end of the grid when log2 is too small. A genuine deficit
        # (above VALIDATION_NOISE, well clear of fp dust) makes forwards and
        # backwards S diverge by exactly the deficit in Distortion.price,
        # so surface it loudly at construction time rather than have one
        # answer silently differ from another downstream.
        #
        # When the sizer already issued a far-tail *clip* warning
        # (``self._bs_clip`` set -- the positive single-big-jump reach did not
        # fit the log2 budget), that warning is the same mass with actionable
        # advice (the exact log2 to raise to), so we do not double-warn here.
        deficit = 1.0 - float(np.sum(self.agg_density))
        if deficit > VALIDATION_NOISE and self._bs_clip is None:
            warnings.warn(
                f'{self.name}: aggregate PMF deficit {deficit:.3e} '
                f'(Σp = 1 − {deficit:.3e} < 1); forwards and backwards '
                f'S diverge by the deficit (forwards > backwards).',
                DefectiveDistributionWarning, stacklevel=2)

        # Staged reinsurance reporting -- §1.2 of the aggregate refactor plan.
        #
        # ``empirical`` above is the final (after-occ + after-agg) realised
        # view. To express the Subject -> after-occ -> after-agg progression
        # we also need the subject (gross) empirical moments and -- when
        # reinsurance is present -- the intermediate after-occ moments.
        # Validation continues to use the subject vs theoretical comparison,
        # which is the only apples-to-apples check available under reins.
        has_occ = self.occ_reins is not None
        has_agg = self.agg_reins is not None

        # Subject severity density: when occ-reins applied, the pre-reins
        # severity is preserved on ``sev_density_gross``; otherwise the
        # current ``sev_density`` IS gross.
        subject_sev = self.sev_density_gross if has_occ else self.sev_density
        # After-occ severity is whatever the occ stage passed along
        # (= ``sev_density`` post ``apply_occ_reins``); identical to subject
        # severity when there is no occ stage.
        after_occ_sev = self.sev_density

        # Subject aggregate: with occ-reins we need one extra FFT of the
        # gross severity (the "validate the subject" hook in §1.3); with no
        # occ-reins the pre-agg-reins density already encodes gross
        # (``agg_density_gross`` when has_agg, else the final
        # ``agg_density``).
        if has_occ:
            subject_agg, _ = self._fft_aggregate(subject_sev, padding)
        elif has_agg:
            subject_agg = self.agg_density_gross
        else:
            subject_agg = self.agg_density

        # After-occ aggregate (pre-agg-reins): when an agg stage exists,
        # ``apply_agg_reins`` stored the pre-stage density in
        # ``agg_density_gross``; with only occ-reins the final
        # ``agg_density`` IS the after-occ density; with no reins there is
        # no separate stage to report.
        if has_agg:
            after_occ_agg = self.agg_density_gross
        elif has_occ:
            after_occ_agg = self.agg_density
        else:
            after_occ_agg = None

        # De-fuzzed moment helper: same |x| < eps zeroing the main
        # empirical block uses (see WHY comment above), wrapped so we can
        # reuse it on subject / after-occ densities.
        def _moments(arr, grid):
            # ``grid`` is xs_sev for severity densities, xs for aggregates
            # (they coincide on the default 0-based grid).
            if arr is None:
                return (np.nan,) * 6
            mw = xsden_to_mwrangler(grid, remove_fuzz(arr))
            return (*mw.noncentral, *mw.mcvsk)

        sub_sev_mom = _moments(subject_sev, self.xs_sev)
        sub_agg_mom = _moments(subject_agg, self.xs)
        self._write_stage_moments('gross_empirical', sub_sev_mom, sub_agg_mom,
                                  copy_freq_from='empirical')

        if has_occ or has_agg:
            aft_sev_mom = _moments(after_occ_sev, self.xs_sev)
            aft_agg_mom = _moments(after_occ_agg, self.xs)
            self._write_stage_moments('after_occ', aft_sev_mom, aft_agg_mom,
                                      copy_freq_from='empirical')

        # Per-stage impact ratios (after / before). 1.0 means no impact;
        # written only when the corresponding stage exists, so consumers can
        # detect "stage absent" by NaN.
        if has_occ:
            self.stats_df['occ_impact'] = (
                self.stats_df['after_occ'] / self.stats_df['mixed'])
        if has_agg:
            self.stats_df['agg_impact'] = (
                self.stats_df['empirical'] / self.stats_df['after_occ'])

        # ``error`` is the SUBJECT validation: gross_empirical vs mixed.
        # With no reinsurance gross_empirical == empirical and this is
        # exactly the legacy theoretical-vs-empirical column. With reins it
        # is the only apples-to-apples check (the after-reins object has no
        # independent theoretical to validate against).
        self.stats_df['error'] = _noise_aware_rel_error(
            self.stats_df['gross_empirical'], self.stats_df['mixed'])

        # invalidate stored functions
        self._cdf = None

    def _fft_aggregate(self, sev_density, padding):
        """Run one FFT convolution: severity density -> aggregate density.

        Single source of truth for the FFT-PGF-iFFT core (Mildenhall 2024,
        §2.2). Used by ``_freq_sev_convolution`` (the main per-update path),
        by ``update_work`` for the subject (gross) aggregate when occ-reins
        is present, and by ``reins_density_df`` to compute gross/ceded/net
        aggregates from the corresponding severities. The zero-risk and
        fixed-1 shortcuts live here so every caller sees them consistently.

        Parameters
        ----------
        sev_density : np.ndarray
            Discretised severity on ``self.xs_sev`` (gross, net, or ceded);
            ``sev_density[j]`` is the mass at physical ``(j - i0) * bs``.
        padding : int
            FFT padding factor passed to ``ft`` / ``ift``.

        Returns
        -------
        agg_density : np.ndarray
            Aggregate density on the output grid ``self.xs``.
        ftagg_density : np.ndarray
            FT of the aggregate (padded length). Callers that don't need
            this (e.g. ``reins_density_df``) discard it.

        Notes
        -----
        Two paths, selected by whether any offset is active:

        - **Default (``i0 == 0`` and output window origin ``x_min == 0``).**
          The original ``ft`` / ``freq_pgf`` / ``ift`` path, unchanged and
          byte-for-byte identical to prior releases.
        - **Signed / windowed (F1 + F2).** Negatives live at the top of the
          padded length-``M = N << padding`` FFT buffer (period ``M*bs``); the
          severity is laid in with physical 0 at index 0 (``i0`` negative
          buckets wrapped to the top). After the FFT the result is *relabelled*
          onto the output window by a single ``np.roll`` of ``-round(x_min/bs)``
          and the first ``N`` buckets kept. Relabelling a finished, exact array
          carries no ``N·s`` shift term, so this is correct for random as well
          as fixed frequency (the key F2 clarification, plan §2). Exact when the
          aggregate support width ``W < M*bs``; a window narrower than the
          support shows up as a two-sided deficit.
        """
        # Thin wrapper over the extracted pure kernel (Phase 2A); all the
        # self-state it needs is passed explicitly so the core is testable
        # without a full update(). See aggregate._aggregate_compute.
        return freq_sev_convolution(
            sev_density, self.frequency.freq_pgf, self.n,
            N=len(self.xs), bs=self.bs, i0=self.i0, x_min=self.x_min,
            en=self.en, freq_name=self.frequency.freq_name, padding=padding)

    def _write_stage_moments(self, col, sev_mom, agg_mom, copy_freq_from=None):
        """Write a moment tuple into a single ``stats_df`` column.

        Helper to keep the staged-empirical writes in ``update_work`` tidy.
        Each moment tuple is the six values ``(ex1, ex2, ex3, mean, cv,
        skew)`` returned by ``xsden_to_mwrangler``.

        Parameters
        ----------
        col : str
            Destination column label (``empirical``, ``after_occ``,
            ``gross_empirical``, ...).
        sev_mom, agg_mom : tuple of float
            Six-tuple raw + central moments for the sev and agg rows.
        copy_freq_from : str or None
            If set, mirror the freq mean/cv/skew rows from another column
            (freq is unchanged by either reinsurance stage).
        """
        _measures = ('ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew')
        for measure, value in zip(_measures, sev_mom):
            self.stats_df.loc[('sev', measure), col] = value
        for measure, value in zip(_measures, agg_mom):
            self.stats_df.loc[('agg', measure), col] = value
        if copy_freq_from is not None:
            src = self.stats_df[copy_freq_from]
            for measure in ('mean', 'cv', 'skew'):
                self.stats_df.loc[('freq', measure), col] = src[('freq', measure)]

    def _freq_sev_convolution(self, padding):
        """Compute the aggregate density by FFT convolution (Mildenhall 2024, §2.2).

        Thin wrapper that routes the (post-occ-reins) ``sev_density`` through
        ``_fft_aggregate`` and writes ``self.agg_density`` /
        ``self.ftagg_density``. The FFT core, zero-risk and fixed-1 shortcuts
        all live in ``_fft_aggregate``.

        Parameters
        ----------
        padding : int
            Padding factor passed to ``ft`` / ``ift`` to mitigate FFT aliasing
            (see Mildenhall 2024, §2.3.2).

        Notes
        -----
        Per-occurrence reinsurance is applied to ``sev_density`` *before* this
        method is called; aggregate reinsurance is applied to ``agg_density``
        *after*. The FFT here is unaware of either.
        """
        self.agg_density, self.ftagg_density = self._fft_aggregate(
            self.sev_density, padding)

    # ================================================================
    # Validation (paper §4.7), unwrap, picks, freq_pmf
    # ================================================================

    @property
    def valid(self):
        """
        Check if the model appears valid. An answer of True means the model is "not unreasonable".
        It does not guarantee the model is valid. On the other hand,
        False means it is definitely suspect. (The interpretation is similar to the null hypothesis
        in a statistical test).
        Called and reported automatically by qd for Aggregate objects.

        Checks the relative errors (from the canonical ``stats_df``) for:

        * severity mean < eps
        * severity cv < 10 * eps
        * severity skew < 100 * eps (skewness is more difficult to estimate)
        * aggregate mean < eps and < ``ALIASING_RATIO`` * severity mean
          relative error (larger values indicate possible aliasing — i.e.
          that ``bs`` is too small).
        * aggregate cv < 10 * eps
        * aggregate skew < 100 * esp

        The default uses eps = 1e-4 relative error. This can be changed by
        setting the ``validation_eps`` variable.

        All reads come from ``stats_df`` -- the single source of truth -- not
        ``validation_df`` (display).

        The CV and skew tests are applied only when the theoretical value is
        finite and its magnitude exceeds ``VALIDATION_NOISE`` -- a
        theoretically-zero skew (symmetric severity) or CV (deterministic
        severity) is skipped, because the FFT's empirical estimate of a zero
        higher moment is grid-dependent noise with no meaningful relative
        error. When the test applies, ``np.isclose`` with relative tolerance
        ``10*eps`` (CV) / ``100*eps`` (skew, harder to estimate) measures
        agreement.

        The ALIASING test silences itself when the agg-mean relative error
        is itself below ``VALIDATION_NOISE`` (genuine numerical dust, not
        aliasing) -- this replaces the old ``eps ** 3`` floor that was fitted
        to the default ``eps`` value.

        Run with logger level 20 (info) for more information on failures.

        A Type 1 error (rejecting a valid model) is more likely than Type 2 (failing to reject an invalide one).

        :return: True (interpreted as not unreasonable) if all tests are passed, else False.

        """
        return _validation.valid_aggregate(self)

    def unwrap(self, p=1e-7, audit=True):
        """
        Unwrap self created with log2 that is too small to contain the answer.

        :param p: Percentile threshold. The estimated p and 1-p quantiles are
            used to determine the effective support [L, R]. R-L must fit in the
            space available, i.e., R-L <= N * self.bs.
        :param audit: If audit, return comparison of empirical moments of shifted
            answer with a.agg_m etc. analytic moments.
        :return: Unwrap named tuple containing fields y the density as a Series,
            mode of shifting/unwrapping, prob_captured the probability in the
            effective support (which should be close to 1), L, R the boundary of the
            effective support.
        """
        # figure bounds from method of moments estimates
        m, cv, skew = self.agg_m, self.agg_cv, self.agg_skew
        sc = self.bs
        L, R = _estimate_agg_percentile(m, cv, skew, p=(p, 1 - p))
        # snap to grid in both cases (can't use self.snap because outside index!)
        L = int(np.round(L / sc, 0))
        R = int(np.round(R / sc, 0))

        # number of buckets
        N = 1 << self.log2

        # is the request reasonable?
        # enough space condition: R - L <= N
        assert R - L <= N, f'{R=} - {L=} = {R-L=} > {N=}, not enough space'

        # how many "blocks" to the right are we?
        l = L // N
        r = R // N

        # extract aliased density
        y = self.density_df.p_total.values

        # there are now two cases: dist fits within one block or wraps over two
        # if it falls over more than two that is an error
        if l == r:
            # no unwrapping, range lies in one block
            # just shift index to right by correct number of chunks
            # locate correct left hand edge, index created below
            L = (L // N) * N
            # method reporting
            mode = 'Shift only'  # \n{L=}'
        elif l == r - 1:
            # must wrap answer into one block and shift
            # figure location of extreme points as remainders
            rem_r = R % N       # right hand end in fft-wrapped coords
            rem_l = L % N
            # by math this will always be true (see blog post)
            assert rem_l >= rem_r
            # unwrap amount
            roll_forward = N - (rem_l + rem_r) // 2
            y = np.roll(y, roll_forward)
            # shifted index, factoring in unwrap
            L = (L // N + 1) * N - roll_forward
            # method reporting
            mode = 'Shift and wrap'  # \n{roll_forward=}, {L=}'
        else:
            # see blog post
            print(f'Should not occur: {l=}, {r=}')

        # align with index and create answer
        i = np.arange(L, L + N, dtype=float) * sc
        ans = pd.Series(y, index=i)
        # apply scale to L and R now to match ans
        L *= sc
        R *= sc
        # document proportion of probability in selected range
        prob_captured = ans[L:R].sum()
        # package results
        Unwrap = namedtuple('Unwrap', 'y, mode, prob_captured, L, R, audit_df')
        if audit:
            em, ecv, eskew = xsden_to_meancvskew(ans.index, ans)
            audit_df = pd.DataFrame(
                {'m': [m, em],
                 'cv': [cv, ecv],
                 'skew': [skew, eskew]},
                index=['actual', 'rewrapped'])

        else:
            audit_df = None
        ans = Unwrap(ans, mode, prob_captured, L, R, audit_df)
        return ans

    def picks(self, attachments, layer_loss_picks, debug=False):
        """
        Adjust the computed severity to hit picks targets in layers defined by a.
        Delegates work to :func:`_picks_work`. See that function for details.

        """
        # always want to work off gross severity
        if self.sev_density_gross is not None:
            logger.info('Using GROSS severity in picks')
            sd = self.sev_density_gross
        else:
            sd = self.sev_density
        return _picks_work(attachments, layer_loss_picks, self.xs, sd, n=self.n,
                          sf=self.sev.sf, debug=debug)

    def freq_pmf(self, log2):
        """
        Return the frequency probability mass function (pmf) computed using 2**log2 buckets.
        Uses self.en to compute the expected frequency. The :class:`Frequency` does not
        know the expected claim count, so this is a method of :class:`Aggregate`.

        """
        n = 1 << log2
        z = np.zeros(n)
        z[1] = 1
        fz = ft(z, 0)
        fz = self.frequency.freq_pgf(self.en, fz)
        dist = ift(fz, 0)
        # remove fuzz -- intentionally ONE-SIDED (zeroes negatives too); a
        # frequency pmf has no legitimate negatives, so this is NOT the shared
        # two-sided ``remove_fuzz`` utility.
        dist[dist < np.finfo(float).eps] = 0
        if not np.allclose(self.n,  self.en):
            logger.warning('Frequency.pmf | n %s != en %s; using en', self.n, self.en)
        return dist

    # ================================================================
    # Reinsurance application: occ pre-FFT, agg post-FFT
    # ================================================================

    def _rebucket_to_grid(self, values, mass, scheme=None, origin=None):
        """Scatter off-grid ``mass`` at target ``values`` onto the model grid.

        The model grid is ``self.xs == bs * arange`` with ``xs[0] == 0``, so
        the (fractional) grid index of a value ``v`` is ``v / bs``. Used to
        place reinsurance net/ceded values back on the grid after the cession
        map moves them off it, and to place discrete-severity atoms on the grid
        during discretization (see :meth:`discretize`).

        Parameters
        ----------
        values : ndarray
            Target loss values (net/ceded points, or severity atoms).
        mass : ndarray
            Probability mass to redistribute, aligned with ``values``.
        scheme : {'linear', 'nearest'}, optional
            Placement scheme. Defaults to :attr:`reins_bucket` (so the
            reinsurance call sites are unchanged); the discrete-severity call
            site passes :attr:`dsev_bucket`.
        origin : float, optional
            Physical value at output index 0 -- the grid origin used to map a
            value to its (fractional) bucket index ``(v - origin) / bs``.
            Defaults to :attr:`x_min` (the output-window origin), correct for
            the reinsurance call sites. The discrete-severity call site passes
            the *severity* grid origin ``xs_sev[0] == -i0·bs``, which differs
            from ``x_min`` when the output window is forced wider than the
            severity support (e.g. an explicit ``x_min`` below the smallest
            atom).

        Returns
        -------
        ndarray
            Probability vector on ``self.xs`` (length ``self.n``).

        Notes
        -----
        Two schemes:

        - ``'nearest'`` rounds each value to its closest bucket. Full mass
          lands in one bucket, with up to ``bs/2`` positional bias.
        - ``'linear'`` splits each value's mass between its two bracketing
          buckets ``k`` and ``k+1`` with weights ``1-f`` and ``f`` where
          ``f = v/bs - k``. Because ``(1-f)·k·bs + f·(k+1)·bs == v``, the
          first moment is preserved **exactly**; both schemes preserve total
          mass (``Σp == Σmass``).

        Values at or beyond the top of the grid (``xs[-1]``) pile into the
        last bucket -- the same overflow mode as an aggregate deficit, and
        surfaced the same way. Negative targets (should not occur for valid
        cessions) clip into bucket 0.
        """
        bs = self.bs
        n = len(self.xs)
        if origin is None:
            origin = self.x_min
        # Grid index of a value v is (v - origin) / bs; origin == 0 on the
        # default grid recovers the original v / bs.
        scaled = (np.asarray(values, dtype=float) - origin) / bs
        out = np.zeros(n)
        if scheme is None:
            scheme = self.reins_bucket
        if scheme == 'nearest':
            idx = np.clip(np.round(scaled).astype(int), 0, n - 1)
            np.add.at(out, idx, mass)
        else:  # 'linear' -- mass split preserves E[X] exactly
            k = np.clip(np.floor(scaled).astype(int), 0, n - 1)
            f = np.clip(scaled - k, 0.0, 1.0)
            kp1 = np.clip(k + 1, 0, n - 1)
            np.add.at(out, k, mass * (1 - f))
            np.add.at(out, kp1, mass * f)
        return out

    def _apply_reins_work(self, reins_list, base_density, debug=False):
        """
        Actually do the work. Called by apply_reins and reins_audit_df.
        Only needs self to get limits, which it must guess without q (not computed
        at this stage). Does not need to know if occ or agg reins,
        only that the correct base_density is supplied.

        :param reins_list:
        :param kind: occ or agg, for debug plotting
        :param debug:
        :return: ceder, netter,
        """
        return _reinsurance.apply_reins_work(self, reins_list, base_density, debug)

    def apply_occ_reins(self, debug=False):
        """
        Apply the entire occ reins structure and save output
        For by layer detail create reins_audit_df
        Makes sev_density_gross, sev_density_net and sev_density_ceded, and updates sev_density to the requested view.

        Not reflected in statistics df.

        :param debug: More verbose.
        :return:
        """
        return _reinsurance.apply_occ_reins(self, debug)

    def apply_agg_reins(self, debug=False, padding=1):
        """
        Apply the entire agg reins structure and save output.
        For by layer detail create reins_audit_df.
        Makes agg_density_gross, agg_density_net and agg_density_ceded, and
        updates agg_density to the requested view.

        Not reflected in statistics df: the post-reins empirical moments
        (``est_*`` and the ``stats_df['empirical']`` column) are written by
        ``update_work`` from the same density updated here.

        :return:
        """
        return _reinsurance.apply_agg_reins(self, debug, padding)

    @property
    def reins_description(self):
        """
        Short narrative description of the reinsurance (str).

        The consistent narrative surface, mirroring ``tail_description`` /
        ``bs_description``: returns the ``kind='both', width=0`` text. For the
        parameterized form (a single kind, or wrapped to a width) use the
        private worker :meth:`_reins_description`.
        """
        return self._reins_description(kind='both', width=0)

    def _reins_description(self, kind='both', width=0):
        """
        Text description of the reinsurance (parameterized worker).

        :param kind: both, occ, or agg
        :param width: width of text for textwrap.fill; omitted if width==0
        """
        return _reinsurance.reins_description(self, kind, width)

    @property
    def reins_kinds(self):
        """Text description of kinds of reinsurance applied.

        Returns
        -------
        str
            One of ``'None'``, ``'Occurrence only'``, ``'Aggregate only'``, or
            ``'Occurrence and aggregate'``.
        """
        return _reinsurance.reins_kinds(self)

    # ================================================================
    # Distortion, ruin theory, plotting
    # ================================================================

    def apply_distortion(self, dist, *, view='ask', S_calculation='forwards',
                         allow_deficit=False):
        r"""
        Apply distortion to the aggregate density; appends ``gS``,
        ``gp_total`` and ``exag`` columns to ``density_df``.

        Routes through the exact-discrete Choquet helper
        (:func:`~aggregate.spectral.choquet_weights`):
        ``exag(a) = rho_g(X ∧ a) = Σ_{x≤a} x·gp + a·g(S(a))``, a direct
        sum carrying the origin -- valid on windowed and signed supports
        (the old ``cumsum(gS)·bs`` idiom assumed a zero-origin grid). The
        effective ``g`` resolves ``view`` × the object's value-type role
        (:meth:`~aggregate.spectral.Distortion.effective_g`), so a
        payoff-role aggregate prices through the dual automatically.

        Parameters
        ----------
        dist : Distortion
            The distortion to apply.
        view : {'ask', 'bid'}
            Pricing view; composes with the value-type role by XOR.
        S_calculation : {'forwards', 'backwards'}
            Deficit-parking direction; see
            :func:`~aggregate.spectral.choquet_weights`.
        allow_deficit : bool
            Explicit truncation policy for a materially defective pmf.
            Default False raises
            :class:`~aggregate.constants.DefectiveDistributionError`.

        Raises
        ------
        ValueError
            For a mass distortion on an unbounded support: the mass lands
            on the last represented bucket, which is a different bounded
            problem, not an approximation. Certify ``self.bounded = True``
            if the support is in fact bounded.
        """
        if self.agg_density is None:
            logger.warning('You must update before applying a distortion ')
            return
        if getattr(dist, 'has_mass', False) and not self.bounded:
            raise ValueError(
                f'mass distortion ({dist.display_name}) on an unbounded aggregate: '
                f'the mass lands on the last represented bucket, a '
                f'different bounded problem. Certify `bounded = True` if '
                f'the support is in fact bounded.')

        g, _, _ = dist.effective_g(view, is_loss_value=self._is_loss_value)
        x = self.density_df.loss.to_numpy()
        p = self.density_df.p_total.to_numpy()
        w = choquet_weights(x, p, g, S_calculation=S_calculation,
                            allow_deficit=allow_deficit)
        self.density_df['gS'] = w.gS
        self.density_df['gp_total'] = w.gp
        # exag(a) = rho(X ∧ a); the strict-tail gp sum telescopes to g(S(a))
        self.density_df['exag'] = np.cumsum(x * w.gp) + x * w.gS

    def pollaczeck_khinchine(self, rho, cap=0, excess=0, stop_loss=0, kind='index', padding=1):
        """
        Return the Pollaczeck-Khinchine Capital function relating surplus to eventual probability of ruin.
        Assumes frequency is Poisson.

        See Embrechts, Kluppelberg, Mikosch 1.2, page 28 Formula 1.11

        TODO: Should return a named tuple.

        :param rho: rho = prem / loss - 1 is the margin-to-loss ratio
        :param cap: cap = cap severity at cap, which replaces severity with X | X <= cap
        :param excess:  excess = replace severity with X | X > cap (i.e. no shifting)
        :param stop_loss: stop_loss = apply stop loss reinsurance to cap, so  X > stop_loss replaced
          with Pr(X > stop_loss) mass
        :param kind:
        :param padding: for update (the frequency tends to be high, so more padding may be needed)
        :return: ruin vector as pd.Series and function to lookup (no interpolation if
          kind==index; else interp) capitals
        """

        if self.sev_density is None:
            raise ValueError("Must recalc before computing Cramer Lundberg distribution.")

        bit = self.sev_density_df.p_sev.copy()
        if cap:
            idx = np.searchsorted(bit.index, cap, 'right')
            bit.iloc[idx:] = 0
            bit = bit / bit.sum()
        elif excess:
            # excess may not be in the index...
            idx = np.searchsorted(bit.index, excess, 'right')
            bit.iloc[:idx] = 0
            bit = bit / bit.sum()
        elif stop_loss:
            idx = np.searchsorted(bit.index, stop_loss, 'left')
            xsprob = bit.iloc[idx + 1:].sum()
            bit.iloc[idx] += xsprob
            bit.iloc[idx + 1:] = 0
        mean = np.sum(bit * bit.index)

        # integrated F function
        fi = bit.shift(-1, fill_value=0)[::-1].cumsum()[::-1].cumsum() * self.bs / mean
        # difference = probability density
        dfi = np.diff(fi, prepend=0)
        # use loc FFT, with wrapping
        fz = ft(dfi, padding)
        mfz = 1 / (1 - fz / (1 + rho))
        f = ift(mfz, padding)
        f = np.real(f) * rho / (1 + rho)
        f = np.cumsum(f)
        ruin = pd.Series(1 - f, index=bit.index)

        if kind == 'index':
            def find_u(p):
                idx = len(ruin) - ruin[::-1].searchsorted(p, 'left')
                return ruin.index[idx]
        else:
            def find_u(p):
                below = len(ruin) - ruin[::-1].searchsorted(p, 'left')
                above = below - 1
                q_below = ruin.index[below]
                q_above = ruin.index[above]
                p_below = ruin.iloc[below]
                p_above = ruin.iloc[above]
                q = q_below + (p - p_below) / (p_above - p_below) * (q_above - q_below)
                return q

        return ruin, find_u, mean, dfi  # , ruin2

    # for backwards compatibility
    cramer_lundberg = pollaczeck_khinchine

    def plot(self, axd=None, xmax=0, **kwargs):
        """
        Basic plot with severity and aggregate, linear and log plots and Lee plot.

        :param xmax: Enter a "hint" for the xmax scale. E.g., if plotting gross and net you want all on
               the same scale. Only used on linear scales?
        :param axd:
        :param kwargs: Lee-panel options forwarded to the quantile worker --
               notably ``quantile_x='return'`` (plot the Lee panel against log
               return period instead of ``p``) and ``max_return_period``; plus
               ``figsize`` for the canvas.
        :return:
        """
        from .plots import plot_aggregate
        return plot_aggregate(self, axd=axd, xmax=xmax, **kwargs)

    def _limits(self, stat='range', kind='linear', zero_mass='include'):
        """
        Suggest sensible plotting limits for kind=range, density, etc., same as Portfolio.

        Should optionally return a locator for plots?

        Called by ploting routines. Single point of failure!

        Must work without ``q`` function when not yet computed.

        :param stat:  range or density (for y axis)
        :param kind:  linear or log (this is the y-axis, not log of range...that is rarely plotted)
        :param zero_mass:  include exclude, for densities
        :return:
        """

        # fudge l/r factors
        def f(x):
            fl, fr = 0.02, 1.02
            return [-fl * x, fr * x]

        # lower bound for log plots
        eps = 1e-16

        # if not computed
        # GOTCHA: if you call q and it fails because not agg_density then q is set to {}
        # which is not None
        if self.agg_density is None:
            # No FFT output yet; estimate the 0.999 quantile from the theoretical
            # mixed-total agg moments.
            try:
                p999 = _estimate_agg_percentile(self.agg_m, self.agg_cv, self.agg_skew, 0.999)
            except ValueError:
                p999 = np.inf
            return f(p999)

        if stat == 'range':
            p = 0.999 if kind == 'linear' else 0.99999
            hi = self.q(p)
            # Window-aware x-limits keyed on the grid origin (``xs[0]``):
            #  * origin < 0  -- signed P&L: mass can sit anywhere on the real
            #    line, so use a *two-sided* quantile range (``f(hi)`` would be
            #    reversed/clipped when hi < 0);
            #  * origin > 0  -- thin-tailed output window: anchor the left edge
            #    at the realised support minimum, not 0, so the empty
            #    ``[0, x_min]`` band isn't drawn;
            #  * origin == 0 -- ordinary non-negative aggregate: unchanged.
            if self.xs is not None and self.xs[0] < 0:
                lo = self.q(1 - p)
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            if self.xs is not None and self.xs[0] > 0:
                lo = float(self.density['loss'].min())
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            return f(hi)

        elif stat == 'density':
            # for density need to divide by bs
            mx = self.agg_density.max() / self.bs
            mxx0 = self.agg_density[1:].max() / self.bs
            if kind == 'linear':
                if zero_mass == 'include':
                    return f(mx)
                else:
                    return f(mxx0)
            else:
                return [eps, mx * 1.5]
        else:
            # if you fall through to here, wrong args
            raise ValueError('Inadmissible stat/kind passsed, expected range/density and log/linear.')

    # ================================================================
    # Display reports, diagnostics, queries, risk measures, pricing
    # ================================================================

    @property
    def pprogram(self):
        """Canonical DecL program text, rendered from the parsed spec.

        Derived by re-parsing :attr:`program` and rendering through
        :func:`aggregate.decl_writer.format_program` (the inverse of the
        parser). It is canonical, not verbatim --- equivalent programs share one
        form. Rendered in the default ``spread`` layout (each clause on its own
        two-space-indented line); call ``format_program(self.program,
        layout='terse')`` for the single-line form. For the raw input as supplied
        to ``build`` use :attr:`program`. An object built programmatically (empty
        ``program``) returns ``''``.
        """
        return format_program(self.program, fmt='text')

    @property
    def pprogram_html(self):
        """Syntax-highlighted DecL program for IPython / Jupyter display."""
        return format_program(self.program, fmt='html')

    @staticmethod
    def _count_program(spec, n, name):
        """Render the DecL for a claim-count distribution from a parsed spec.

        Builds a minimal ``agg`` program that keeps the frequency clause verbatim
        and replaces the severity with a point mass at 1 (``dsev [1]``). Because N
        claims each of size 1 sum to N, the resulting aggregate density is exactly
        the claim-count distribution ``P(N = k)``. Shared by
        :meth:`create_frequency` and :meth:`Portfolio.create_frequency`.

        Parameters
        ----------
        spec : dict
            A **raw transformer spec** (``parser.parse(...)[2]``), *not* the
            dense ``Aggregate._spec`` constructor dict.
        n : float
            The resolved total expected count (``Aggregate.n``).
        name : str
            Name for the rendered ``agg``.

        Returns
        -------
        str
            A canonical, single-line DecL ``agg`` program for the count
            distribution.

        Notes
        -----
        Only the *frequency* keys are carried over; severity, layers
        (``exp_limit`` / ``exp_attachment``) and both reinsurance clauses are
        dropped. Those reshape severity-per-claim or the aggregate total but
        never the *number* of claims, and carrying them through against a
        ``dsev [1]`` severity would corrupt the count (e.g.
        ``occurrence net of 50 xs 0`` would net every unit point mass to 0).

        The exposure is collapsed to the resolved expected count ``n`` as
        ``<n> claims`` rather than re-rendering the original exposure clause:
        when the count is *derived* from severity (``500 loss ...``, a
        ``premium at lr`` or limit profile), swapping the severity would change
        the count. ``n`` is the correct total expected count even for profiles
        and mixed frequency. An empirical (``dfreq``) frequency already *is* the
        count distribution, so its outcome/probability vectors are kept as the
        ``dfreq`` head and no ``claims`` exposure is synthesized.

        Frequency mixing / contagion (``mixed gamma c``, ``zm`` / ``zt``, etc.)
        is part of the frequency and is preserved verbatim.
        """
        # Keep only the frequency clause; drop severity, layers and reinsurance.
        new = {'name': name}
        for key in ('freq_name', 'freq_a', 'freq_b', 'freq_zm', 'freq_p0'):
            if key in spec:
                new[key] = spec[key]
        # An empirical (dfreq) frequency is itself the count distribution and
        # renders as the exposure head; everything else collapses to the
        # resolved expected count.
        if spec.get('freq_name') != 'empirical':
            new['exp_en'] = n
        # Point-mass severity: N unit claims sum to N.
        new['sev_name'] = 'dhistogram'
        new['sev_xs'] = [1.0]
        new['sev_ps'] = [1.0]
        return spec_to_decl(new, kind='agg', name=name)

    def _frequency_program(self, name):
        """Render the count-distribution DecL by re-parsing :attr:`program`.

        The raw transformer spec is recovered by re-parsing :attr:`program`
        (``spec_to_decl`` wants the transformer spec, not the dense
        ``Aggregate._spec`` constructor dict), then handed to
        :meth:`_count_program` with the resolved :attr:`n`.

        Raises
        ------
        ValueError
            If the object was built programmatically and carries no
            :attr:`program` to re-parse.
        """
        if not self.program:
            raise ValueError(
                f'create_frequency requires a DecL program to re-parse; '
                f'aggregate {self.name!r} was built programmatically (empty '
                f'program).')
        from .underwriter import build
        _kind, _name, spec = build.parser.parse(self.program)
        return self._count_program(spec, self.n, name)

    def create_frequency(self):
        """Materialize this object's claim-count distribution as an ``Aggregate``.

        The engine carries frequency only as a PGF (applied in the Fourier
        domain), so there is no ``q`` / ``tvar`` / ``cdf`` / percentiles for the
        count itself. This builds the marginal count distribution as a
        first-class :class:`Aggregate` (via the ``dsev [1]`` point-mass trick) so
        every inherited method works on the count.

        Returns
        -------
        Aggregate
            A built aggregate named ``f'{self.name}.freq'`` whose aggregate
            density *is* this object's claim-count distribution: ``agg_density[k]
            = P(N = k)``. Use ``.q``, ``.tvar``, ``.cdf``, ``.plot``,
            ``.density_df``, ``.summary_df`` etc. on it directly.

        Examples
        --------
        >>> fa = a.create_frequency()
        >>> fa.q([0.01, 0.5, 0.99])   # count percentiles
        >>> fa.tvar(0.99)             # tail count
        >>> fa.plot()                 # the count distribution, plotted

        Notes
        -----
        Built through the normal ``build`` front door, so grid windowing, ``bs``
        and ``log2`` selection all happen automatically -- a high-mean count
        (large exposure) needs no special handling. The returned object is a
        *snapshot*: rebuild it if the parent changes. See
        :meth:`_frequency_program` for what is kept and dropped from the spec.
        """
        from .underwriter import build
        return build(self._frequency_program(f'{self.name}.freq'))

    @property
    def validation_df(self):
        """Moment-vs-estimate table for Freq / Sev / Agg (the QA view).

        The validation frame: it proves the FFT reproduced the analytic
        moments ("if the first three moments match, the aggregate is *not
        unreasonable*"). Surfaced on demand and in :meth:`qd` when the object
        *fails* validation; the daily-driver headline is :attr:`summary_df`.
        Three-row Freq / Sev / Agg frame.

        Two display modes, same 8-column shape and same column arithmetic:

        * **No reinsurance** -- validation view. Columns are theoretical
          ``EX | Est EX | Err EX | CV | Est CV | Err CV | Sk | Est Sk``.
          ``Err`` is the noise-aware relative error of empirical vs
          theoretical.
        * **With reinsurance** -- economic view. Columns become ``Gross EX
          | <label> EX | Change EX | Gross CV | <label> CV | Change CV |
          Gross Sk | <label> Sk``, where ``Gross`` is the theoretical
          before any cover and ``<label>`` is the model output -- ``Net``
          (all covers net of), ``Ceded`` (all ceded to), or ``Output``
          (mixed, occ and agg passing different kinds). ``Change = (output
          - gross) / gross`` -- arithmetically the same column as ``Err``
          (so the eyeball degenerates cleanly to the validation view when
          reins is absent), but now read as the % change driven by the
          cession. Labels are the ``REINS_LABEL_*`` constants.

        Sources from the canonical ``self.stats_df``: ``mixed`` for
        Gross, ``empirical`` for the realised (model-output) view.
        """
        return self._describe()

    @staticmethod
    def _cv_or_nan(mean, sd):
        """``CV = SD / E[X]``, blanked (``NaN``) when the mean is ~0.

        ``CV`` is meaningless near a zero mean (a signed / near-break-even
        position), so it is left blank when ``|mean| < CV_MEAN_REL_TOL * sd``
        -- the mean is then indistinguishable from zero at the scale of the
        spread. ``SD`` is always reported by the caller; only ``CV`` blanks.
        """
        mean = float(mean)
        sd = float(sd)
        if not (np.isfinite(mean) and np.isfinite(sd)):
            return np.nan
        if abs(mean) < CV_MEAN_REL_TOL * sd:
            return np.nan
        if mean == 0.0:
            return np.nan
        return sd / mean

    @property
    def summary_df(self):
        """At-a-glance risk view -- moments + key percentiles, Freq / Sev / Agg.

        The daily-driver headline (the lead frame in :meth:`qd` and
        :meth:`_repr_html_`). The compound-model identity made legible: each row
        answers a different question -- count risk (``Freq``), single-claim
        severity (``Sev``), total loss (``Agg``) -- and the percentiles trace
        where the tail comes from (a heavy ``Agg`` skew you can see is inherited
        from ``Sev``). The *validation* moment-error table is now
        :attr:`validation_df`; the tail-behavior classifier is
        :attr:`tail_behavior_df`.

        **Index** ``Freq`` / ``Sev`` / ``Agg`` (the ``X`` index).

        **Columns** ``E[X] | SD | CV | Skew | p0.01 | p0.50 | p0.99``.

        - ``SD`` and ``CV`` are **both always present** (stable layout).
          ``CV = SD / E[X]`` is blank when ``|E[X]|`` is ~0 relative to ``SD``
          (a signed / near-break-even position -- see :meth:`_cv_or_nan`); ``SD``
          never blanks. ``Skew`` is well defined even at mean 0, so it stays.
        - Moments are the analytic (theoretical) moments from
          :attr:`stats_df`, so the ``Freq`` × ``Sev`` = ``Agg`` mean identity is
          exact.
        - Percentiles come from the FFT grid (exact, not simulated): ``Agg`` via
          :meth:`q`, ``Sev`` via :meth:`q_sev` (mixtures included, already on the
          grid). They populate only **after** :meth:`update`.

        **Frequency-row percentiles are blank** by design: frequency is carried
        as a PGF (``freq_pgf``), applied in the Fourier domain -- the engine
        never materializes a count distribution, so there is nothing to take a
        quantile of. The Freq row still carries ``E[X] / SD / CV / Skew``
        (PGF-exact), which is what that row is for (count volatility). To get the
        count distribution as a first-class object, use
        :meth:`create_frequency`, then ``.q(...)`` / ``.tvar(...)`` on it.

        Returns
        -------
        pandas.DataFrame
            Three-row Freq / Sev / Agg frame, ``E[X]`` carried in ``.attrs``.
        """
        st = self.stats_df['mixed']
        rows = ['Freq', 'Sev', 'Agg']
        comps = ['freq', 'sev', 'agg']
        means = [float(st[(c, 'mean')]) for c in comps]
        cvs = [float(st[(c, 'cv')]) for c in comps]
        sds = [m * cv if np.isfinite(cv) else np.nan for m, cv in zip(means, cvs)]
        skews = [float(st[(c, 'skew')]) for c in comps]
        df = pd.DataFrame(
            {
                'E[X]': means,
                'SD': sds,
                'CV': [self._cv_or_nan(m, sd) for m, sd in zip(means, sds)],
                'Skew': skews,
            },
            index=rows,
        )
        df.index.name = 'X'
        # Percentiles from the realised grid (exact, not simulated); Freq blank
        # by design (PGF, no materialized count distribution); pre-update blank.
        pcols = [f'p{p:.2f}' for p in SUMMARY_PERCENTILES]
        for pc in pcols:
            df[pc] = np.nan
        if self.agg_density is not None:
            for p, pc in zip(SUMMARY_PERCENTILES, pcols):
                df.loc['Sev', pc] = self.q_sev(p)
                df.loc['Agg', pc] = self.q(p)
        for c in ('E[X]', 'SD', 'Skew', *pcols):
            df[c] = _snap_noise(df[c])
        df.attrs['mean'] = means[-1]
        return df

    def tail_df(self, periods=None):
        """Return-period / exceedance table for the aggregate (the centerpiece).

        The language of reinsurance submissions, cat-model output, and
        Solvency II / rating-agency capital. **Aggregate-only** (tail risk is a
        property of the total), so it complements :attr:`summary_df`
        ("made of") with "how bad does it get". The tail numbers -- including the
        1-in-1000 TVaR -- come from the FFT grid, **exact, not simulated** (no
        Monte-Carlo wobble).

        **Index** the return period ``T`` (default ladder
        :data:`DEFAULT_RETURN_PERIODS`; pass ``periods=`` to override). The
        1-in-200 (99.5%, Solvency II) and 1-in-250 (99.6%, US capital-adequacy /
        rating) rows are highlighted in the HTML rendering.

        **Columns** ``p | VaR | TVaR | xsVaR | VaR/Mean``.

        - ``p`` non-exceedance probability for the row.
        - ``VaR = q(p)`` -- the quoted number.
        - ``TVaR = tvar(p)`` -- the priced number; adjacent to ``VaR`` so the
          VaR-to-TVaR gap (tail fatness) reads at a glance.
        - ``xsVaR = VaR - E[X]`` -- capital, the excess of VaR over expected.
        - ``VaR/Mean`` -- leverage.

        Parameters
        ----------
        periods : array_like of float, optional
            Return-period ladder. Defaults to :data:`DEFAULT_RETURN_PERIODS`.

        Returns
        -------
        pandas.DataFrame or None
            Indexed by return period ``T``; ``E[X]`` carried in ``.attrs``.
            ``None`` before :meth:`update` (the realised grid is not yet built).

        Notes
        -----
        Loss objects (``is_loss_value``) map ``T = 1 / (1 - p)`` (the upper
        tail); payoff / P&L objects map ``T = 1 / p`` so the table reads off the
        downside -- the shared :func:`period_to_p`. Downside *TVaR* for a signed
        payoff position is refined in the P&L veneer (see ``dev/plan-pnl-*``).
        """
        if self.agg_density is None:
            return None
        return return_period_frame(
            self.q, self.tvar, self.est_m, self._is_loss_value, periods)

    def _describe(self, force_reins_label=None, force_sd=False):
        """Build the ``validation_df`` frame, optionally forced into reins view.

        Parameters
        ----------
        force_reins_label : str or None
            When ``None`` (the default, used by the ``validation_df`` property)
            the column format is chosen from this unit's own reinsurance:
            the economic Gross/Net/Ceded/Output view if a treaty is
            present, else the plain theory/empirical validation view.

            When a non-``None`` label is supplied, the economic view is
            forced and that label is used for the after-reins column,
            regardless of this unit's own cession. ``Portfolio.validation_df``
            passes a portfolio-wide label here so that every unit block —
            including units with no reinsurance — shares one column
            layout and aligns with the ``total`` block.
        force_sd : bool, default False
            Force the **SD** spread trio (instead of CV) even when this unit
            is not itself signed. A signed unit always uses SD; this flag lets
            ``Portfolio.validation_df`` push the whole table into SD when *any* unit
            is signed, so the unit blocks and the ``total`` block share one
            column layout (CV and SD cannot be mixed in one frame).

        Returns
        -------
        pandas.DataFrame
            Three-row Freq / Sev / Agg frame; see :attr:`validation_df`.
        """
        if self._signed() or force_sd:
            return self._describe_signed(force_reins_label)
        st = self.stats_df['mixed']
        rlabel = force_reins_label if force_reins_label is not None \
            else self._reins_after_label()
        df = pd.DataFrame(
            {
                'EX': [st[('freq', 'mean')], st[('sev', 'mean')], st[('agg', 'mean')]],
                'CV': [st[('freq', 'cv')],   st[('sev', 'cv')],   st[('agg', 'cv')]],
                'Sk': [st[('freq', 'skew')], st[('sev', 'skew')], st[('agg', 'skew')]],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'
        emp = self.stats_df['empirical']
        post_update = pd.notna(emp.get(('agg', 'mean'), np.nan))
        if post_update:
            # Realised (after-reins, or = subject if no reins) middle column.
            mid_label = rlabel or 'Est'
            df.loc['Sev', f'{mid_label} EX'] = emp[('sev', 'mean')]
            df.loc['Agg', f'{mid_label} EX'] = emp[('agg', 'mean')]
            change_label = 'Change' if rlabel else 'Err'
            df.loc[:, f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} CV'] = emp[('sev', 'cv')]
            df.loc['Agg', f'{mid_label} CV'] = emp[('agg', 'cv')]
            df.loc[:, f'{change_label} CV'] = _noise_aware_rel_error(
                df[f'{mid_label} CV'], df['CV'])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = emp[('sev', 'skew')]
            df.loc['Agg', f'{mid_label} Sk'] = emp[('agg', 'skew')]
            ordered = [
                'EX', f'{mid_label} EX', f'{change_label} EX',
                'CV', f'{mid_label} CV', f'{change_label} CV',
                'Sk', f'{mid_label} Sk',
            ]
            df = df[ordered]
        # First-column label: under reinsurance the theoretical is the
        # ``Gross`` view (top of step 1, before any cover); without reins keep
        # the legacy ``EX``/``CV``/``Sk`` headings (no rename necessary).
        if rlabel:
            df = df.rename(columns={
                'EX': f'{REINS_LABEL_GROSS} EX',
                'CV': f'{REINS_LABEL_GROSS} CV',
                'Sk': f'{REINS_LABEL_GROSS} Sk'})
        # snap floating-point dust to 0 in moment-value columns for
        # display (e.g. the skew of a symmetric severity); NaN preserved.
        # Change/Err columns retain their numeric dust (they are the
        # validation eyeball).
        for c in df.columns:
            if ' EX' in c or ' CV' in c or ' Sk' in c or c in ('EX', 'CV', 'Sk'):
                if not (c.startswith('Err ') or c.startswith('Change ')):
                    df[c] = _snap_noise(df[c])
        return df

    def _describe_signed(self, force_reins_label=None):
        """``validation_df`` for a signed aggregate -- **SD** trio instead of CV.

        Same 8-column shape and column arithmetic as :meth:`_describe`, but the
        ``CV`` trio is replaced by an ``SD`` trio. The coefficient of variation
        ``CV = sd / mean`` is unstable and meaningless when the mean can be ~0
        (a signed aggregate straddling 0), so for any signed object -- a
        ``ssev`` / negative-``dsev`` aggregate -- the spread is reported as the
        standard deviation, which is finite and informative regardless of the
        mean. (This also cleans up the 1.0.0a22 signed-portfolio validation_df.)

        The Freq / Sev / Agg rows are in their native (signed) frame; the
        theoretical (Gross) column is sourced from the loss ``stats_df``.

        Parameters
        ----------
        force_reins_label : str or None
            As in :meth:`_describe`.

        Returns
        -------
        pandas.DataFrame
        """
        st = self.stats_df['mixed']
        emp = self.stats_df['empirical']
        rlabel = force_reins_label if force_reins_label is not None \
            else self._reins_after_label()

        # Theoretical (loss stats_df). The signed severity already straddles 0,
        # so the moments are in their native frame -- no display transform.
        freq_sd = st[('freq', 'mean')] * st[('freq', 'cv')]
        sev_sd = self.sev_sd
        agg_sd = self.agg_sd
        df = pd.DataFrame(
            {
                'EX': [st[('freq', 'mean')], st[('sev', 'mean')],
                       st[('agg', 'mean')]],
                'SD': [freq_sd, sev_sd, agg_sd],
                'Sk': [st[('freq', 'skew')], st[('sev', 'skew')],
                       st[('agg', 'skew')]],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'
        post_update = pd.notna(emp.get(('agg', 'mean'), np.nan))
        if post_update:
            mid_label = rlabel or 'Est'
            # Empirical: Freq/Sev from the stats_df; Agg from the scalars.
            emp_freq_sd = emp[('freq', 'mean')] * emp[('freq', 'cv')]
            df.loc['Sev', f'{mid_label} EX'] = self.est_sev_m
            df.loc['Agg', f'{mid_label} EX'] = self.est_m
            change_label = 'Change' if rlabel else 'Err'
            df.loc[:, f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} SD'] = self.est_sev_sd
            df.loc['Agg', f'{mid_label} SD'] = self.est_sd
            df.loc['Freq', f'{mid_label} SD'] = emp_freq_sd
            df.loc[:, f'{change_label} SD'] = _noise_aware_rel_error(
                df[f'{mid_label} SD'], df['SD'])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = self.est_sev_skew
            df.loc['Agg', f'{mid_label} Sk'] = self.est_skew
            ordered = [
                'EX', f'{mid_label} EX', f'{change_label} EX',
                'SD', f'{mid_label} SD', f'{change_label} SD',
                'Sk', f'{mid_label} Sk',
            ]
            df = df[ordered]
        if rlabel:
            df = df.rename(columns={
                'EX': f'{REINS_LABEL_GROSS} EX',
                'SD': f'{REINS_LABEL_GROSS} SD',
                'Sk': f'{REINS_LABEL_GROSS} Sk'})
        for c in df.columns:
            if ' EX' in c or ' SD' in c or ' Sk' in c or c in ('EX', 'SD', 'Sk'):
                if not (c.startswith('Err ') or c.startswith('Change ')):
                    df[c] = _snap_noise(df[c])
        return df

    def _reins_after_label(self):
        """Heading for the model-output column in ``validation_df``.

        ``Net`` when every cession passes the net; ``Ceded`` when every
        cession passes the ceded; ``Output`` when occ and agg pass
        different kinds (e.g. ``net of occ then ceded to agg`` -- a mixed
        output). Returns ``None`` when no reinsurance is configured (legacy
        validation-view headings apply).
        """
        return _reinsurance.reins_after_label(self)

    def _severity_lattice(self):
        """Integer-lattice step of the severity, or ``None`` if not on a lattice.

        Returns the gcd of the (integer) severity atoms -- the natural bucket
        size, since an aggregate of lattice-valued severities is itself on that
        lattice **regardless of the frequency** (e.g. Poisson x ``dsev [1:10]``
        is integer-valued, so ``bs=1``). ``None`` for any continuous component.
        This is what lets the window estimator pick a coarse, exact ``bs`` and
        shrink ``log2`` to fit, instead of defaulting to a fine ``bs`` over the
        full ``2**log2`` buckets.

        Returns
        -------
        float or None
            ``gcd`` of the integer atoms (1 for ``dsev [1:n]``; 5 for atoms
            ``[0 5 10]``), or ``None`` if any severity component is continuous /
            non-integer.
        """
        atoms = []
        for s in (self.sevs if self.sevs is not None else []):
            a = getattr(s, 'support_atoms', None)
            if a is None:
                return None
            atoms.append(np.asarray(a, dtype=float))
        if not atoms:
            return None
        allx = np.concatenate(atoms)
        if not np.allclose(allx, np.round(allx), atol=1e-9):
            return None
        ints = np.abs(np.round(allx).astype(np.int64))
        ints = ints[ints != 0]
        if len(ints) == 0:
            return 1.0
        g = int(np.gcd.reduce(ints))
        return float(g) if g > 0 else 1.0

    def _severity_high_estimate(self, p):
        """Upper extent (~``p`` quantile) of the per-occurrence severity mixture.

        Used by ``_bs_window``'s ``windowed`` guard. The windowed method only
        relabels the finished *aggregate*; the severity is still discretised on
        ``xs_sev = [0, N*bs]`` (physical 0 at index 0). So windowing is valid
        only when a single severity fits in the windowed grid extent -- true for
        a genuine compound (many small claims summing far above 0), false for a
        ``fixed``-1 / ``approximate`` object whose one severity already sits at
        the aggregate mean.

        Parameters
        ----------
        p : float
            Coverage (e.g. ``1 - 1e-12``) for the method-of-moments percentile
            on an unbounded severity.

        Returns
        -------
        float
            A (conservative) upper bound on the severity support: the analytic
            method-of-moments high quantile from the severity moments, capped by
            any finite policy/support limit. A **degenerate (point-mass)**
            severity returns its atom location directly (see Notes). ``inf`` if
            it cannot be estimated (forces the windowed guard to fail safe -> no
            windowing).

        Notes
        -----
        The method-of-moments quantile slightly *overstates* a bounded
        severity's reach, which biases the guard toward **not** windowing --
        the safe direction (a false reject merely keeps the legacy 0-based
        grid; a false accept would corrupt the severity discretisation).

        A degenerate severity (standard deviation zero -- a ``dsev [k]`` point
        mass, e.g. the count distribution materialized by
        :meth:`create_frequency`) has no spread, so its skewness is ``0/0 =
        NaN`` and the MoM fit returns NaN -- which would silently fail the
        ``np.isfinite`` windowed guard and force a high-mean concentrated
        aggregate onto the coarse 0-based grid (the *textbook* windowing case,
        defeated). Such a severity's high extent is simply the atom location
        (the mean), returned directly. The test is on the standard deviation
        against :data:`VALIDATION_NOISE`, so a machine-noise ``cv`` still counts
        as a point mass.
        """
        try:
            sev_m = float(self.stats_df['mixed'][('sev', 'mean')])
            sev_cv = float(self.stats_df['mixed'][('sev', 'cv')])
            sev_sk = float(self.stats_df['mixed'][('sev', 'skew')])
            if sev_m * sev_cv <= VALIDATION_NOISE:
                # Degenerate point mass: zero spread, the MoM fit is undefined.
                hi = sev_m
            else:
                hi = float(_estimate_agg_percentile(sev_m, sev_cv, sev_sk, p))
        except (ValueError, KeyError):
            hi = np.inf
        lim = (float(self.limit.max())
               if self.limit is not None and len(self.limit) else np.inf)
        return min(hi, lim) if np.isfinite(lim) else hi

    def _severity_low_estimate(self, tail):
        """Lower (``~tail`` quantile) reach of the severity mixture.

        The signed counterpart of :meth:`_severity_high_estimate`: how far
        *below* zero a single occurrence reaches at lower-tail probability
        ``tail``. Used by :meth:`_single_big_jump_window` to size the negative
        extent (and grid width) of a signed aggregate so the severity cannot
        wrap the FFT buffer (the aliasing failure mode).

        Parameters
        ----------
        tail : float
            Lower-tail probability (e.g. ``1e-14``); the reach is ``ppf(tail)``.

        Returns
        -------
        float
            The minimum component ``ppf(tail)`` -- a conservative (most
            negative) bound on the severity's lower support. ``0.0`` for a
            non-signed severity (no negative reach). ``-inf`` if a component
            quantile cannot be evaluated.

        Notes
        -----
        Bracketing the mixture quantile below by the minimum of the component
        quantiles is the safe (wider) direction for grid sizing: it can only
        widen the window, never clip the severity's negative tail.
        """
        if not self._signed_severity() or self.sevs is None:
            return 0.0
        los = []
        for s in self.sevs:
            if not getattr(s, 'signed', False):
                continue
            try:
                los.append(float(s.fz.ppf(tail)))
            except Exception:  # pragma: no cover - defensive
                los.append(-np.inf)
        return float(min(los)) if los else 0.0

    def _loss_tail_classes(self):
        """Loss-space ``(left_tail, right_tail)`` rungs for the bucket sizer.

        The single-big-jump floor and the tail-aware slack split key off the
        *loss* convolution's tail thickness -- the grid is sized on the loss
        FFT. Returns the aggregate's per-side decay rungs from the shared tail
        report (:meth:`_tail_rows`), oriented in loss space.

        Returns
        -------
        (left_tail, right_tail) : tuple of aggregate.tail.TailClass
            The loss-space aggregate decay rungs. Spec-only (valid before
            :meth:`update`); fed to :func:`aggregate.tail.is_thick`.
        """
        agg = self._tail_rows()[-1]
        return agg.left_tail, agg.right_tail

    def _single_big_jump_window(self, p_star):
        """Single-big-jump extent floor for a heavy / signed severity.

        For a subexponential severity the aggregate's far tail is dominated by a
        single large claim on an otherwise typical bulk:
        ``P(S > x) ~ E[N]·P(X > x)``. So to cover the aggregate to ``p_star``
        the *severity* must be probed at the deeper level
        ``p** = 1 - (1 - p_star)/E[N]`` -- one claim reaches there, the other
        ``E[N]-1`` are typical (this is **not** ``N·q_X``, which would assume
        *every* claim is huge and wildly over-size). The single-big-jump extent
        replaces one typical claim (mean ``mu_X``) on the bulk (aggregate mean
        ``ES``) by one big claim::

            sbj_hi = ES - mu_X + q_X_hi(p**)      # one big claim up
            sbj_lo = ES - mu_X + q_X_lo(p**)      # one big claim down (signed)

        Parameters
        ----------
        p_star : float
            Aggregate coverage to guarantee (e.g. ``1 - 1e-12``); ``> 1`` is
            read as a number of nines.

        Returns
        -------
        (sbj_lo, sbj_hi) : tuple of float, or None
            The single-big-jump window edges (``sbj_lo == 0`` for a non-signed
            severity). ``None`` when ``E[N]``, the severity mean, or the
            ``p**`` quantile is unavailable (e.g. no finite variance) -- the
            caller then keeps the existing window / moment-sizer path.

        Notes
        -----
        ``p**`` deepens with ``E[N]``; at ``E[N]=5000`` and
        ``p_star = 1 - 1e-12``, ``1 - p** = 2e-16`` is past double precision and
        ``q_X(p**) -> inf`` for an unbounded severity. The lower tail
        ``1 - p**`` is therefore floored at ``SBJ_TAIL_FLOOR``
        (``discretization.sbj_tail_floor``) and ``q_X_hi`` is capped by any
        finite policy limit (via :meth:`_severity_high_estimate`).
        """
        en = float(self.n)
        if not (np.isfinite(en) and en >= 1.0):
            return None
        try:
            sev_m = float(self.stats_df['mixed'][('sev', 'mean')])
        except (KeyError, ValueError, TypeError):
            return None
        es = float(self.agg_m)
        if not (np.isfinite(sev_m) and np.isfinite(es)):
            return None
        # p** with the author's numerical-depth guard: 1 - p** = (1-p*)/E[N]
        # floored at SBJ_TAIL_FLOOR so q_X(p**) stays finite for an unbounded
        # severity.
        p_star = float(np.where(p_star > 1, 1.0 - 10.0 ** -p_star, p_star))
        tail = max((1.0 - p_star) / en, SBJ_TAIL_FLOOR)
        p2 = 1.0 - tail
        q_hi = self._severity_high_estimate(p2)
        if not np.isfinite(q_hi):
            return None
        sbj_hi = es - sev_m + q_hi
        if self._signed_severity():
            q_lo = self._severity_low_estimate(tail)
            sbj_lo = (es - sev_m + q_lo) if np.isfinite(q_lo) else q_lo
        else:
            sbj_lo = 0.0
        if not np.isfinite(sbj_lo):
            return None
        return float(sbj_lo), float(sbj_hi)

    def _clipped_mass_estimate(self, grid_top):
        """Estimate the aggregate mass above ``grid_top`` via the single big jump.

        When the far right tail does not fit the grid (item 6), the clipped
        aggregate mass is dominated by the single-big-jump mechanism: a grid top
        of ``grid_top`` corresponds to one big claim of size
        ``grid_top - (ES - mu_X)`` on an otherwise typical bulk, so
        ``P(S > grid_top) ~ E[N]·P(X > grid_top - ES + mu_X)``. Summed (the mix
        is additive in the survival), capped at 1.

        Parameters
        ----------
        grid_top : float
            The realized grid's upper edge.

        Returns
        -------
        float
            The estimated clipped mass in ``[0, 1]``, or ``nan`` if it cannot be
            formed (no finite ``E[N]`` / severity mean).
        """
        en = float(self.n)
        try:
            sev_m = float(self.stats_df['mixed'][('sev', 'mean')])
        except (KeyError, ValueError, TypeError):
            return np.nan
        es = float(self.agg_m)
        if not (np.isfinite(en) and en >= 1.0
                and np.isfinite(sev_m) and np.isfinite(es)):
            return np.nan
        x_claim = grid_top - es + sev_m
        sevs = self.sevs if self.sevs is not None else []
        sf = 0.0
        for s in sevs:
            try:
                sf += float(s.fz.sf(x_claim))
            except Exception:  # pragma: no cover - defensive
                return np.nan
        n_comp = max(len(sevs), 1)
        return float(min(en * sf / n_comp, 1.0))

    def _exact_discrete_window(self):
        """Exact aggregate support for a fully-discrete ``dfreq``/``fixed`` x ``dsev``.

        Returns ``(A_min, A_max, bs_lattice)`` when the frequency is discrete-
        finite (``dfreq`` -> ``empirical``, or ``fixed``) **and** every severity
        component is a discrete histogram on an integer lattice; otherwise
        ``None``. The aggregate then takes values exactly on the integer lattice
        and its support is finite and exactly computable.

        Notes
        -----
        With claim counts ``N`` (atoms, min ``N_min`` max ``N_max``, possibly
        including 0) and severity atoms ``s_min … s_max``:

        - ``A_max = max(N_max·s_max, N_min·s_max)`` (the sum is maximised by the
          largest atom repeated; over ``N`` the extreme is at ``N_max`` if
          ``s_max>0`` else ``N_min``); include ``0`` if ``0`` is a count atom.
        - ``A_min = min(N_max·s_min, N_min·s_min)`` symmetrically; include ``0``
          if ``0`` is a count atom.

        ``bs_lattice`` is 1 for integer atoms (the common case).
        """
        freq = self.frequency
        if freq.freq_name == 'empirical':
            n_atoms = np.asarray(freq.freq_a, dtype=float)
            has_zero = bool(np.any(n_atoms == 0))
        elif freq.freq_name == 'fixed':
            n_atoms = np.array([float(self.n)])
            has_zero = (self.n == 0)
        else:
            return None
        if self.sevs is None or len(self.sevs) == 0:
            return None
        atoms = []
        for s in self.sevs:
            a = getattr(s, 'support_atoms', None)
            if a is None:
                return None    # a non-discrete component -> not exact
            atoms.append(np.asarray(a, dtype=float))
        s_all = np.concatenate(atoms)

        def _allint(x):
            return bool(np.allclose(x, np.round(x), atol=1e-9))

        if not (_allint(n_atoms) and _allint(s_all)):
            return None
        s_min, s_max = float(s_all.min()), float(s_all.max())
        n_min, n_max = float(n_atoms.min()), float(n_atoms.max())
        hi = max(n_max * s_max, n_min * s_max)
        lo = min(n_max * s_min, n_min * s_min)
        if has_zero:
            hi = max(hi, 0.0)
            lo = min(lo, 0.0)
        return float(lo), float(hi), 1.0

    def _bounded_severity_window(self, p):
        """Window for a bounded severity with a (possibly small) claim count.

        Returns ``(A_lo, A_hi)`` using the severity's bounded support and a high
        frequency quantile ``N_hi`` (from the analytic frequency moments), or
        ``None`` if the severity is not bounded. For a small claim count this
        bound is tight; for a large count the law of large numbers concentrates
        the aggregate far inside ``[0, N_hi·s_max]`` and the moment window is
        tighter -- the caller (``_bs_window``) only selects this method when it
        is at least as tight as the moment window.
        """
        if self.sevs is None or len(self.sevs) == 0:
            return None
        if not all(getattr(s, 'bounded', False) for s in self.sevs):
            return None
        s_his, s_los = [], []
        for s in self.sevs:
            hi = s.limit if np.isfinite(s.limit) else float(s.fz.support()[1])
            lo = float(s.fz.support()[0]) if getattr(s, 'signed', False) else 0.0
            s_his.append(hi)
            s_los.append(lo)
        s_max, s_min = max(s_his), min(s_los)
        f1, f2, f3 = self.frequency.freq_moms(self.n)
        fsd = float(np.sqrt(max(f2 - f1 * f1, 0.0)))
        zN = ss.norm.isf(1 - p)
        n_hi = f1 + zN * fsd
        return float(min(0.0, n_hi * s_min)), float(n_hi * s_max)

    def _bs_window(self, log2, bs_in, x_min_in, bucket_sizing_p,
                   window_convention=None):
        """Decide ``(bs, log2, x_min)`` for ``update`` and build ``_bs_window_df``.

        Orchestrator delegated to :func:`_bucket_window.bs_window`; see there
        for the full method / selection documentation. Runs the sizing methods,
        populates ``self._bs_window_df``, and returns the chosen grid.
        """
        return _bucket_window.bs_window(self, log2, bs_in, x_min_in,
                                        bucket_sizing_p, window_convention)

    def aggregate_error_analysis(self, log2, bs2_from=None, **kwargs):
        """
        Analysis of aggregate error across a range of bucket sizes. If ``bs2_from
        is None`` size a starting bs from the analytic moment window
        (``estimate_agg_window``) and scan plus/minus 3 doublings. Note: if the
        distribution does not have a second moment, you must enter bs2_from.

        :param log2:
        :param bs2_from: lower bound on bs to use, in log2 terms; estimated from
          the analytic moment window if not input.
        :param kwargs: passed to ``update``

        """
        # copy of self, updating alters the internal state of an object
        cself = Aggregate(**self.spec)

        if bs2_from is None:
            if cself.agg_cv == np.inf:
                raise ValueError('Distribution must have variance to guess bucket size. '
                                 'Input bs2_from')
            # ``recommend_bucket`` retired (W10); size a starting bs from the
            # analytic 3-moment output window (``estimate_agg_window``) instead.
            _, _, _w = estimate_agg_window(
                self.agg_m, self.agg_m * self.agg_cv, self.agg_skew)
            bs = round_bucket(_w / (1 << log2))
            bs2 = int(np.log2(bs))
            bss = 2. ** np.arange(bs2 - 3, bs2 + 4)
        else:
            bss = 2. ** np.arange(bs2_from, bs2_from + 7)

        # analytic aggregate mean
        m = cself.agg_m
        # aggregate analysis
        agg_ans = []
        for bs in bss:
            cself.update(bs=bs, log2=log2, **kwargs)
            agg_ans.append([bs, m, cself.est_m,
                            cself.est_m - m, cself.est_m / m - 1])

        agg_df = pd.DataFrame(agg_ans,
                              columns=['bs', 'agg_m', 'est_m',
                                       'abs_m', 'rel_m', ])
        m = cself.sev_m
        agg_df['rel_h'] = agg_df.bs / 2 / m
        agg_df['rel_total'] = agg_df.rel_h * np.sign(agg_df.rel_m) + agg_df.rel_m
        agg_df = agg_df.set_index('bs')
        agg_df.columns = agg_df.columns.str.split('_', expand=True)
        agg_df.columns.names = ['view', 'stat']
        return agg_df

    def severity_error_analysis(self, sev_calc='round', discretization_calc='survival',
                                normalize=True):
        """
        Analysis of severity component errors, uses the current bs in self.
        Gives detailed, component by component, error analysis of severities.
        Includes discretization error (bs large relative to mean) and
        truncation error (tail integral large).

        Total S shows the aggregate not severity. Generally about self.n * (1 - sum_p)
        (per Feller).

        """
        truncation_point = self.bs * (1 << self.log2)
        wts = self.en / self.n
        beds = self.discretize(sev_calc=sev_calc,
                               discretization_calc=discretization_calc,
                               normalize=normalize)
        sev_ans = []
        total_row = len(self.sevs)
        for i, (s, wt, en, bed) in enumerate(zip(self.sevs, wts, self.en, beds)):
            # exact theoretical sev mean from the canonical stats_df: the
            # per-component column label is ``e{e}.m{m}`` (exposure × sev-
            # mixture), enumerated in order by both broadcasting arms.
            label = self._comp_cols[i]
            m = self.stats_df.loc[('sev', 'ex1'), label]
            if len(self.sevs) == 1:
                i = self.name
            # estimated
            em, _ = xsden_to_meancv(self.xs, bed)
            sev_ans.append([s.long_name,
                            s.limit, s.attachment,
                            truncation_point,
                            s.sf(truncation_point), bed.sum(),
                            wt, en,
                            m, 0,
                            m, em
                            ])
        # the total
        m = self.sev_m
        # attachment is None if the limit clause is missing
        min_attach = np.where(self.attachment==None, 0., self.attachment).min()
        sev_ans.append(['total',
                        self.limit.max(), min_attach,
                        truncation_point,
                        self.sf(truncation_point), self.sev_density_df.p_sev.sum(),
                        1, self.n,
                        m, 0.,
                        m, self.est_sev_m
                        ])

        sev_df = pd.DataFrame(sev_ans,
                              columns=['name',
                                       'limit', 'attachment',
                                       'trunc',
                                       'S', 'sum_p',
                                       'wt', 'en',
                                       'agg_mean', 'agg_wt',
                                       'mean', 'est_mean'
                                       ],
                              index=range(total_row + 1))
        sev_df['agg_mean'] *= sev_df['en']
        sev_df['agg_wt'] = sev_df['agg_mean'] / \
                           sev_df.loc[0:total_row - 1, 'agg_mean'].sum()
        sev_df['abs'] = sev_df['est_mean'] - sev_df['mean']
        sev_df['rel'] = sev_df['abs'] / sev_df['mean']
        sev_df['trunc_error'] = \
            [_integral_by_doubling(s.sf, truncation_point) for s in self.sevs] + \
            [_integral_by_doubling(self.sev.sf, truncation_point)]
        sev_df['rel_trunc_error'] = sev_df.trunc_error / sev_df['mean']
        sev_df['h_error'] = self.bs / 2
        sev_df['rel_h_error'] = self.bs / 2 / sev_df['mean']

        # compute discretization_err_2 (was a separate function in development)
        xs = np.hstack((self.xs - self.bs / 2, self.xs[-1] + self.bs / 2))
        ans = []
        for s in self.sevs:
            # density at xs
            f = s.pdf(xs)
            # derv of f = -S''
            df = np.gradient(f, self.bs)
            # integral to quadratic adjustment term approx to S
            ans.append(np.sum(df) * self.bs ** 3 / 24)
        ans = pd.Series(ans)

        sev_df['h2_adj'] = np.hstack((ans, 0.))
        sev_df.loc[total_row, 'h2_adj'] = \
            sev_df.loc[0:total_row - 1, ['wt', 'h2_adj']].prod(1).sum()
        sev_df['rel_h2_adj'] = sev_df['h2_adj'] / sev_df['mean']

        return sev_df

    def q(self, p, kind='lower'):
        """
        Return quantile function of density_df.p_total.

        Definition 2.1 (Quantiles)
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] ≥ α} is the lower α-quantile of X
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] > α} is the upper α-quantile of X.

        ``kind=='middle'`` has been removed.

        :param p:
        :param kind: 'lower' or 'upper'.
        :return:
        """

        if kind == 'middle' and getattr(self, 'middle_warning', 0) == 0:
            self.middle_warning = 1

        if kind == 'middle':
            kind = 'lower'

        assert kind in ['lower', 'upper'], 'kind must be lower or upper'

        return self._grid_distribution().q(p, kind)

    # for consistency with scipy
    ppf = q

    def _grid_distribution(self):
        """The :class:`GridDistribution` view over the aggregate ``p_total`` grid.

        Lazily built and cached on first use (and after :meth:`update`, which
        resets the cache to ``None``). Built on the **full** contiguous ``bs``
        grid (zero-mass buckets included): the var/tvar kernel filters to the
        positive-mass subset internally (:meth:`GridDistribution._funcs`), so
        ``q``/``tvar`` are unchanged, while the width-summing ``lev`` /
        ``cdf`` / ``sf`` need the full grid to match the ``exa`` / ``add_exa``
        convention -- dropping the empty low buckets (where ``S == 1``) would
        make ``lev`` undercount. (Pre-1.0.0a97 this filtered ``p_total > 0`` up
        front, which silently broke ``lev`` on the subset grid; the filter moved
        inside the kernel where it belongs.)
        """
        if self._dist is None:
            self._dist = GridDistribution.from_series(
                self.density_df.p_total, bs=self.bs, name=self.name,
                is_loss_value=self._is_loss_value)
        return self._dist

    def _sev_grid_distribution(self):
        """The :class:`GridDistribution` view over the severity ``p_sev`` grid.

        Severity has its own grid (``sev_density_df``); this is the discretised
        *output* severity PMF (the uniform ``bs``-grid fed to the FFT), distinct
        from the input ``self.fz``. Lazily built and cached.

        Carries the **aggregate's** orientation (``self._is_loss_value``), not an
        intrinsic loss role: the severity curve shares the aggregate's Lee panel
        and must spread the same tail, so a payoff aggregate draws its severity
        with the payoff convention too. The objective accessors that read this GD
        (:meth:`sev_q`, :meth:`sev_tvar`) are sign-agnostic, so the role only
        affects :meth:`GridDistribution.return_period`.
        """
        if self._sev_dist is None:
            ser = self.sev_density_df.query('p_sev > 0').p_sev
            self._sev_dist = GridDistribution.from_series(
                ser, bs=self.bs, name=f'{self.name} sev',
                is_loss_value=self._is_loss_value)
        return self._sev_dist

    def focus(self, p=1e-6):
        """Return the central window of ``density_df`` holding ``1 - p`` of the mass.

        A thin, no-recompute re-slicer over the finished aggregate: it runs
        :func:`~aggregate.utilities.balanced_window` on the realized
        ``p_total`` and returns the rows of :attr:`density_df` in the equal-tail
        window ``[q(p/2), q(1 - p/2)]`` -- ``p/2`` of the mass trimmed off each
        tail, ``1 - p`` kept and centred. Useful for tightening the display
        window of any computed aggregate (and the post-calc primitive the
        bivariate axis sizing is built on).

        Parameters
        ----------
        p : float, default 1e-6
            Total discarded tail mass, split equally between the two tails. Must
            satisfy ``0 < p < 1``. The literal discarded mass, not a coverage --
            see :func:`~aggregate.utilities.balanced_window`.

        Returns
        -------
        pandas.DataFrame
            The slice ``density_df.loc[lo:hi]`` (a view onto the existing frame;
            no recomputation). All columns are preserved.

        Notes
        -----
        Edges are snapped to the bucket size ``bs`` so the window aligns with
        the grid. Does not mutate the aggregate -- ``density_df`` is unchanged.
        """
        if self.density_df is None:
            raise ValueError('Must update before calling focus.')
        ser = self.density_df.query('p_total > 0').p_total
        lo, hi = balanced_window(ser, p, bs=self.bs)
        return self.density_df.loc[lo:hi]

    def q_sev(self, p):
        """
        Compute quantile of severity distribution, returning element in the index.
        Very similar code to q, but only lower quantiles.

        :param p:
        :return:
        """

        return self._sev_grid_distribution().q(p, 'lower')

    def tvar_sev(self, p):
        """
        TVaR of severity - now available for free!

        added June 2023

        Fixed 1.0.0a91: previously this read the *aggregate* var/tvar cache
        (``_var_tvar_function['tvar']`` built from ``p_total``) rather than the
        severity grid, so it returned the aggregate TVaR. It now correctly uses
        the severity grid (``sev_density_df.p_sev``) -- the numbers move.
        """
        return self._sev_grid_distribution().tvar(p)

    def tvar(self, p, kind=''):
        """
        Updated June 2023, 0.13.0

        Compute the tail value at risk at threshold p

        Definition 2.6 (Tail mean and Expected Shortfall)
        Assume E[X−] < ∞. Then
        x¯(α) = TM_α(X) = α^{−1}E[X 1{X≤x(α)}] + x(α) (α − P[X ≤ x(α)])
        is α-tail mean at level α the of X.
        Acerbi and Tasche (2002)

        We are interested in the right hand exceedence [?? note > vs ≥]
        α^{−1}E[X 1{X > x(α)}] + x(α) (P[X ≤ x(α)] − α)

        McNeil etc. p66-70 - this follows from def of ES as an integral
        of the quantile function

        q is exact quantile (most of the time)
        q1 is the smallest index element (bucket multiple) greater than or equal to q

        tvar integral is int_p^1 q(s)ds = int_q^infty xf(x)dx = q + int_q^infty S(x)dx
        we use the last approach. np.trapz approxes the integral. And the missing piece
        between q and q1 approx as a trapezoid too.

        :param p:
        :param kind:
        :return:
        """
        if kind != '' and getattr(self, 'c', None) is None:
            logger.warning('kind is no longer used in TVaR, new method equivalent to kind=tail but much faster. '
                           'Argument kind will be removed in the future.')
            self.c = 1

        if kind == 'inverse':
            logger.warning('kind=inverse called...??!!')

        assert self.density_df is not None, 'Must recompute prior to computing tail value at risk.'

        return self._grid_distribution().tvar(p)

    def sample(self, n, replace=True):
        """
        Draw a sample of n items from the aggregate distribution. Wrapper around
        pd.DataFrame.sample.


        """

        if self.density_df is None:
            raise ValueError('Must update before sampling.')
        return self.density_df[['loss']].sample(n=n, weights=self.density_df.p_total,
                                                replace=replace, random_state=ar.RANDOM,
                                                ignore_index=True)

    @property
    def sev(self):
        """
        Make exact sf, cdf and pdfs and store in namedtuple for use as sev.cdf etc.
        """
        if self._sev is None:
            SevFunctions = namedtuple('SevFunctions', ['cdf', 'sf', 'pdf'])
            if len(self.sevs) == 1:
                self._sev = SevFunctions(cdf=self.sevs[0].cdf, sf=self.sevs[0].sf,
                                         pdf=self.sevs[0].pdf)
            else:
                # multiple severites, needs more work
                wts = np.array([i.sev_wt for i in self.sevs])
                # for non-broadcast weights the sum is n = number of components; rescale
                if wts.sum() == len(self.sevs):
                    wts = self.stats_df.loc[('freq', 'ex1'), self._comp_cols].values
                wts = wts / wts.sum()

                # tried a couple of different approaches here and this is as fast as any
                def _sev_cdf(x):
                    return np.sum([wts[i] * self.sevs[i].cdf(x) for i in range(len(self.sevs))], axis=0)

                def _sev_sf(x):
                    return np.sum([wts[i] * self.sevs[i].sf(x) for i in range(len(self.sevs))], axis=0)

                def _sev_pdf(x):
                    return np.sum([wts[i] * self.sevs[i].pdf(x) for i in range(len(self.sevs))], axis=0)

                self._sev = SevFunctions(cdf=_sev_cdf, sf=_sev_sf, pdf=_sev_pdf)
        return self._sev

    def cdf(self, x, kind='previous'):
        """
        Return cumulative probability distribution at x using kind interpolation.

        2022-10 change: kind introduced; default was linear

        :param x: loss size
        :return:
        """
        if self._cdf is None:
            self._cdf = interpolate.interp1d(self.xs, self.agg_density.cumsum(), kind=kind,
                                             bounds_error=False, fill_value='extrapolate')
        # 0+ converts to float
        return 0. + self._cdf(x)

    def sf(self, x):
        """
        Return survival function using linear interpolation.

        :param x: loss size
        :return:
        """
        return 1 - self.cdf(x)

    def pdf(self, x):
        """
        Probability density function, assuming a continuous approximation of the bucketed density.

        :param x:
        :return:
        """
        if self._pdf is None:
            self._pdf = interpolate.interp1d(self.xs, self.agg_density, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
        return self._pdf(x) / self.bs

    def pmf(self, x):
        """
        Probability mass function, treating aggregate as discrete
        x must be in the index (?)

        """
        if self.density_df is None:
            raise ValueError("Must update before computing probabilities!")

        try:
            return self.density_df.loc[x, 'p_total']
        except KeyError:
            return 0.0
            # raise KeyError(f'Value {x} must be in index for probability mass function.')

    def json(self):
        """
        Write spec to json string.

        :return:
        """
        return json.dumps(self._spec)

    def approximate(self, approx_type='slognorm', output='scipy'):
        """
        Create an approximation to self using method of moments matching.

        Compare to Portfolio.approximate which returns a single sev fixed freq agg, this
        returns a scipy dist by default.

        Use case: exam questions with the normal approacimation!

        :param approx_type: norm, lognorm, slognorm (shifted lognormal), gamma, sgamma. If 'all'
            then returns a dictionary of each admissible approx (families that
            cannot be represented for this distribution/output are skipped).
        :param output: scipy - frozen scipy.stats continuous rv object;
          sev_decl - DecL program for severity (to substituate into an agg ; no name)
          sev_kwargs - dictionary of parameters to create Severity
          agg_decl - Decl program agg T 1 claim sev_decl fixed
          any other string - created Aggregate object
        :return: as above.

        A shifted family (``slognorm`` / ``sgamma``) requested for a symmetric
        distribution degenerates to its normal limit and emits a ``UserWarning``
        (pass ``approx_type='norm'`` to select the normal explicitly). A
        left-skewed distribution is fitted by reflection, which has no native
        frozen ``scipy`` / one-line DecL form -- ``output='scipy'`` /
        ``'sev_decl'`` / ``'agg_decl'`` raise ``ValueError`` for it; use
        ``output='sev_kwargs'`` or the default Aggregate object.
        """
        # Prefer empirical moments (post-update) over theoretical (pre-update).
        emp = self.stats_df['empirical']
        if pd.notna(emp.get(('agg', 'mean'), np.nan)):
            m, cv, skew = (emp[('agg', 'mean')], emp[('agg', 'cv')], emp[('agg', 'skew')])
        else:
            mixed = self.stats_df['mixed']
            m, cv, skew = (mixed[('agg', 'mean')], mixed[('agg', 'cv')], mixed[('agg', 'skew')])
        note = f'frozen version of {self.name}'

        def _one(kind, warn):
            nm = f'{kind[0:4]}.{self.name[0:5]}'
            return approximate_from_mcvsk(m, cv, skew, nm, f'agg {nm} 1 claim sev ',
                                          note, kind, output, warn_degenerate=warn)

        if approx_type == 'all':
            # Survey: stay quiet about degeneration, and skip a family that
            # cannot be represented for this distribution/output (e.g. a
            # reflected fit requested as a frozen scipy rv).
            out = {}
            for kind in ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']:
                try:
                    out[kind] = _one(kind, warn=False)
                except ValueError:
                    continue
            return out
        return _one(approx_type, warn=True)

    def entropy_fit(self, n_moments, tol=1e-10, verbose=False):
        """
        Find the max entropy fit to the aggregate based on n_moments fit.
        The constant is added (sum of probabilities constraint), for two
        moments there are n_const = 3 constrains.

        Based on discussions with, and R code from, Jon Evans

        Run ::

            ans = obj.entropy_fit(2)
            ans['ans_df'].plot()

        to compare the fits.

        :param n_moments: number of moments to match
        :param tol:
        :param verbose:
        :return:
        """
        # sum of probs constraint
        n_constraints = n_moments + 1

        # don't want to mess up the object...
        xs = self.xs.copy()
        p = self.agg_density.copy()
        # de-fuzz before the moment fit (threshold standardized on machine eps)
        p = remove_fuzz(p)
        p = p / np.sum(p)
        p1 = p.copy()

        mtargets = np.zeros(n_constraints)
        for i in range(n_constraints):
            mtargets[i] = np.sum(p)
            p *= xs

        parm1 = np.zeros(n_constraints)
        x = np.array([xs ** i for i in range(n_constraints)])

        probs = np.exp(-x.T @ parm1)
        machieved = x @ probs
        der1 = -(x * probs) @ x.T

        er = 1
        iters = 0
        while er > tol:
            iters += 1
            try:
                parm1 = parm1 - inv(der1) @ (machieved - mtargets)
            except np.linalg.LinAlgError:
                print('Singluar matrix')
                print(der1)
                return None
            probs = np.exp(-x.T @ parm1)
            machieved = x @ probs
            der1 = -(x * probs) @ x.T
            er = (machieved - mtargets).dot(machieved - mtargets)
            if verbose:
                print(f'Error: {er}\nParameter {parm1}')
        ans = pd.DataFrame(dict(xs=xs, agg=p1, fit=probs))
        ans = ans.set_index('xs')
        return dict(params=parm1, machieved=machieved, mtargets=mtargets, ans_df=ans)

    def var_dict(self, p, kind='lower', snap=False):
        """
        Make a dictionary of value at risks for the aggregate, mirrors Portfolio.var_dict.
        Here is just marshals calls to the appropriate var or tvar function.

        Allows the price function to run consistently with Portfolio version.

        Example Use: ::

            for p, arg in zip([.996, .996, .996, .985], ['var', 'lower', 'upper', 'tvar']):
                print(port.var_dict(p, arg,  snap=True))

        :param p:
        :param kind: var (defaults to lower), upper, lower, tvar
        :param snap: snap tvars to index
        :return:
        """
        if kind == 'var': kind = 'lower'
        if kind == 'tvar':
            d = {self.name: self.tvar(p)}
        else:
            d = {self.name: self.q(p, kind)}
        if snap and kind == 'tvar':
            d = {self.name: self.snap(d[self.name])}
        return d

    def price(self, p, g, kind='var'):
        """
        Price using regulatory and pricing g functions, mirroring Portfolio.price.
        Unlike Portfolio, cannot calibrate. Applying specified Distortions only.
        If calibration is needed, embed Aggregate in a one-line Portfolio object.

        Compute E_price (X wedge E_reg(X) ) where E_price uses the pricing distortion and E_reg uses
        the regulatory distortion.

        Regulatory capital distortion is applied on unlimited basis: ``reg_g`` can be:

        * if input < 1 it is a number interpreted as a p value and used to determine VaR capital
        * if input > 1 it is a directly input  capital number
        * d dictionary: Distortion; spec { name = dist name | var, shape=p value a distortion used directly

        ``pricing_g`` is  { name = ph|wang and shape=}, if shape (lr or roe not allowed; require calibration).

        if ly, must include ro in spec

        :param p: a distortion function spec or just a number; if >1 assets, if <1 a prob converted to quantile
        :param kind: var lower upper tvar
        :param g:  pricing distortion function
        :return:
        """
        # Thin delegator to the shared single-distribution pricing concern.
        return _pricing.price(self, p, g, kind)

    def price_pentagon(self, *, p=None, a=None, P=None, M=None, Q=None,
                       LR=None, PQ=None, ROE=None):
        """Complete the pricing octet at a capital level given one target.

        Fix the capital level with exactly one of ``p`` (a VaR probability,
        ``a = self.q(p)``) or ``a`` (an asset level, snapped to the grid), then
        supply exactly one pricing target -- premium ``P``, cost of capital
        ``ROE`` (a.k.a. CoC), or a loss ratio via ``LR`` (equivalently ``M``,
        ``Q`` or ``PQ``). Returns the canonical one-row (``'total'``) pentagon
        ``DataFrame`` (columns :data:`~aggregate.pentagon.PENTAGON_STATS`).
        The target keywords match the canonical stat names.

        Pure accounting completion against the object's expected loss at the
        chosen capital level -- **no distortion is involved** (contrast
        :meth:`price`, which prices with a :class:`Distortion`). The triple
        ``{L, a, target}`` is solved by :meth:`Pentagon.solve`.

        Parameters
        ----------
        p : float, optional
            VaR probability fixing the capital level; mutually exclusive with ``a``.
        a : float, optional
            Asset level fixing the capital; mutually exclusive with ``p``.
        P, M, Q, LR, PQ, ROE : float, optional
            Exactly one pricing target -- premium, margin, capital, loss ratio,
            premium-to-capital, or cost of capital (``M/Q``).

        Returns
        -------
        pandas.DataFrame
            One ``'total'`` row, eight canonical pentagon columns.

        Raises
        ------
        ValueError
            If not exactly one of ``p``/``a`` is given, or not exactly one
            pricing target is supplied.
        """
        # Thin delegator to the shared single-distribution pricing concern.
        return _pricing.price_pentagon(
            self, p=p, a=a, P=P, M=M, Q=Q, LR=LR, PQ=PQ, ROE=ROE)

    def price_ccoc(self, ccoc, *, p):
        """Price at a constant cost of capital ``ccoc`` and VaR level ``p``.

        No distortion is involved -- a thin alias for
        ``self.price_pentagon(p=p, ROE=ccoc)`` returning the canonical one-row
        (``'total'``) pentagon ``DataFrame``. Parity with
        :meth:`Portfolio.price_ccoc`.
        """
        return _pricing.price_ccoc(self, ccoc, p=p)

    def prob_loss_assets(self, *, p=None, L=None, a=None):
        """Given any one of ``p``, ``L``, ``a``, return the consistent triple.

        Free choice over the capital anchor: pass exactly one of the VaR
        probability ``p``, the limited expected loss ``L = E[min(X, a)]``, or
        the asset level ``a`` -- any one determines the other two. Thin
        delegator to
        :meth:`~aggregate._grid_distribution.GridDistribution.prob_loss_assets`
        over this aggregate's ``p_total`` grid (the single ``lev`` source).
        Aliased :meth:`pla`.

        Returns
        -------
        ProbLossAssets
            Namedtuple ``(p, L, a)``, grid-snapped and mutually consistent
            (``L == lev(a)``, ``p == cdf(a)``).
        """
        return self._grid_distribution().prob_loss_assets(p=p, L=L, a=a)

    pla = prob_loss_assets

    def price_pentagon_ex(self, *, p=None, a=None, L=None,
                          M=None, P=None, Q=None, LR=None, PQ=None, ROE=None):
        """Complete the pricing octet over the full pentagon vocabulary.

        The full-power front door over :meth:`price_pentagon`: free over the
        capital anchor (``p``, ``a``, **or** the limited expected loss ``L``)
        and accepting any soluble pentagon configuration, warning when an
        accounting-determined ``L`` does not reconcile with ``E[min(X, a)]``.
        Thin delegator to the shared single-distribution pricing concern; see
        :func:`aggregate._pricing.price_pentagon_ex` for the full contract.

        Returns
        -------
        pandas.DataFrame
            One ``'total'`` row: a leading ``p`` column then the eight canonical
            pentagon stats.
        """
        return _pricing.price_pentagon_ex(
            self, p=p, a=a, L=L, M=M, P=P, Q=Q, LR=LR, PQ=PQ, ROE=ROE)

    def calibrate_distortions(self, coc, *, p=None, a=None, kind='lower',
                              names=_pricing.DEFAULT_CALIBRATION_DISTORTIONS):
        """Calibrate the standard pricing distortion set to a cost-of-capital target.

        The ``Aggregate`` counterpart of :meth:`Portfolio.calibrate_distortions`
        -- calibration to **this** distribution (the aggregate is its own
        total), with no per-unit allocation (that stays a ``Portfolio``
        concern). Calibrating directly here means no more wrapping a single
        ``Aggregate`` in a one-unit ``Portfolio`` to obtain a calibrated
        distortion set.

        Parameters
        ----------
        coc : float
            Target cost of capital ``COC = (P - L) / Q``.
        p : float, optional
            Probability at which the calibration applies; converted to an asset
            level via ``self.q(p, kind)``. Exactly one of ``p`` or ``a``.
        a : float, optional
            Asset level; snapped to the grid. Exactly one of ``p`` or ``a``.
        kind : {'lower', 'upper'}, optional
            VaR kind when ``p`` is provided. Default ``'lower'``.

        Returns
        -------
        pandas.DataFrame
            ``distortion_df`` (also stored on ``self.distortion_df``): one row
            per distortion in ``[ccoc, ph, wang, dual, tvar]``. The shared
            calibration target is stored once on ``self.calibration_df`` and the
            calibrated objects on ``self.distortions`` keyed by name -- same
            schema as :meth:`Portfolio.calibrate_distortions`.

        Notes
        -----
        The expected loss anchoring the premium target is computed on the full
        aggregate grid (the ``E[min(X, a)]`` / ``add_exa`` convention), matching
        a one-unit ``Portfolio``'s ``exa_total``.

        ``names`` selects the distortion families to calibrate (default the
        standard set); signed / payoff supports are handled transparently (see
        :func:`aggregate._pricing.calibrate_distortions`).
        """
        return _pricing.calibrate_distortions(self, coc, p=p, a=a, kind=kind,
                                              names=names)
