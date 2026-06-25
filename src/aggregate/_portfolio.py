from copy import deepcopy
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy import interpolate
from textwrap import fill
import warnings

from .constants import (DefectiveDistributionWarning,
                        FIG_H, FIG_W, INFO_NA, info_row,
                        REINS_LABEL_OUTPUT, Validation)
from .config import get_settings
from .distributions import (Aggregate, Severity, WINDOW_NINES, BUCKET_SIZING_P,
                            _flat_col_to_stats_index, approximate_from_mcvsk,
                            estimate_agg_window, value_type_label)

# Resolved once per session from config (see aggregate.config). VALIDATION_NOISE
# is the absolute dust floor used throughout validation; ALIASING_RATIO is the
# agg-vs-sev mean-error multiple for the ALIASING flag; EXEQA_NOISE_FLOOR is the
# exeqa_err floor below which a bucket's conditional decomposition is reliable.
VALIDATION_NOISE = get_settings().validation.noise
ALIASING_RATIO = get_settings().validation.aliasing_ratio
EXEQA_NOISE_FLOOR = get_settings().validation.exeqa_noise_floor

__all__ = ['Portfolio', 'make_awkward']
from .results import (AnalyzeDistortionResult, AnalyzeDistortionsResult,
                      PricingResult)
from .spectral import Distortion, DISTORTION_DTYPE
from . import tail as _tail
from .tail import TailClass
from .pentagon import (PENTAGON_STATS, PENTAGON_DTYPE, complete_pentagon,
                       Pentagon)
from .moments import (MomentAggregator, xsden_to_mwrangler,
                      _noise_aware_rel_error, _snap_noise)
from .decl_writer import format_program
from .utilities import (ft, ift,
                        round_bucket,
                        agg_help, explain_validation,
                        remove_fuzz as remove_fuzz_util)
from ._grid_distribution import GridDistribution
from ._aggregate import return_period_frame, SUMMARY_PERCENTILES
from . import _pricing
from . import _reinsurance
from . import _bucket_window
from . import _validation
from . import _portfolio_density as _density
from . import _portfolio_sample as _smpl
from . import _portfolio_common as _common
from ._portfolio_common import check01, make_array, convex_points
from ._portfolio_sample import (make_comonotonic_allocations,
                                make_comonotonic_allocations_work,
                                swap_density_df)


# fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# matplotlib.rcParams['legend.fontsize'] = 'xx-small'
logger = logging.getLogger(__name__)



# Canonical column order/dtype for pricing exhibits — the single source of
# truth lives in ``pentagon.py`` (the accounting authority). These module-level
# aliases preserve the historical names used throughout this file.
PRICING_STAT_ORDER = PENTAGON_STATS
PRICING_STAT_DTYPE = PENTAGON_DTYPE


# Canonical row MultiIndex for ``Portfolio.stats_df``. Parallels
# ``aggregate.distributions._STATS_ROW_INDEX`` (meta + freq + sev + agg
# moment blocks). Kept as its own constant so future Portfolio-only
# rows (e.g. between-unit copula moments) do not bleed into
# Aggregate's surface. All-float: ``self.name`` lives on the attribute.
_PORT_STATS_ROW_INDEX = pd.MultiIndex.from_tuples(
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


class Portfolio(object):
    """
    Portfolio creates and manages a portfolio of Aggregate objects each modeling one
    unit of business. Applications include

    - Model a book of insurance
    - Model a large account with several sub units
    - Model a reinsurance portfolio or large treaty

    """

    def __init__(self, name, spec_list, uw=None):
        """
        Create a new :class:`Portfolio` object.

        :param name: The name of the portfolio. No spaces or underscores.
        :param spec_list: A list of

           1. dictionary: Aggregate object dictionary specifications or
           2. Aggregate: An actual aggregate objects or
           3. tuple (type, dict) as returned by uw['name'] or
           4. string: Names referencing objects in the optionally passed underwriter
           5. a single DataFrame: empirical samples (the total column, if present, is ignored);
              a p_total column is used for probabilities if present

        :returns:  new :class:`Portfolio` object.
        """
        self.name = name
        self.agg_list = []
        self.unit_names = []
        self._valid = None
        self.sample_df = None
        logger.debug(f'Portfolio.__init__| creating new Portfolio {self.name}')
        # logger.debug(f'Portfolio.__init__| creating new Portfolio {self.name} at {super(Portfolio, self).__repr__()}')
        ma = MomentAggregator()
        max_limit = 0
        if len(spec_list) == 1 and isinstance(spec_list[0], pd.DataFrame):
            # create from samples...slightly different looping behavior
            logger.info('Creating from sample DataFrame')
            spec_list = spec_list[0]
        if isinstance(spec_list, pd.DataFrame):
            spec_list = spec_list.copy().astype(float)
            if 'p_total' not in spec_list:
                logger.info('Adding p_total column to DataFrame with equal probs')
                spec_list['p_total'] = np.repeat(1 / len(spec_list), len(spec_list))
            # it is helpful to know what sample the object is created
            self.sample_df = spec_list

        for spec in spec_list:
            if isinstance(spec, Aggregate):
                # directly passed in an agg object
                a = spec
                agg_name = spec.name
            elif isinstance(spec, str) and isinstance(spec_list, pd.DataFrame):
                if spec not in ['total', 'p_total']:
                    # hack: issue: close values mess up the discrete distribution
                    # 2**-30 = 9.313225746154785e-10 approx 1e-9, so to ensure we don't
                    # have any merging issues in the discrete distribution, we round
                    # to 8 decimal places.
                    temp = spec_list[[spec, 'p_total']]
                    temp['rounded'] = np.round(temp[spec].astype(float), 8)
                    s = temp.groupby('rounded').sum()
                    a = Aggregate(name=spec,
                                  exp_en=1,
                                  sev_name='dhistogram', sev_xs=s.index.values, sev_ps=s.p_total.values,
                                  freq_name='fixed')
                    agg_name = spec
                else:
                    a = None
            elif isinstance(spec, str):
                # look up object in uw return actual instance
                # uw.build_many(spec, update=False) parses or looks up by name
                # and returns a list[ParsedProgram] with `.object` populated but
                # not smart-updated — Portfolio handles its own update later.
                if uw is None:
                    raise ValueError('Must pass valid Underwriter instance to create aggs by name')
                try:
                    a_out = uw.build_many(spec, update=False)
                except Exception as e:
                    logger.error('Item %s not found in your underwriter', spec)
                    raise e
                # a is a disct (kind, name) -> (obj or spec, program) pair. Portfolios are ?always created so
                # here, spec is the name
                assert len(a_out) == 1
                # remember, the thing you make must be called a as part of the loop
                a = a_out[('agg', spec)][0]
                assert isinstance(a, Aggregate)
                agg_name = a.name
            elif isinstance(spec, tuple):
                # uw returns type, spec
                assert spec[0] == 'agg'
                a = Aggregate(**spec[1])
                agg_name = spec[1]['name']
            elif isinstance(spec, dict):
                a = Aggregate(**spec)
                agg_name = spec['name'][0] if isinstance(spec['name'], list) else spec['name']
            else:
                raise ValueError(f'Invalid type {type(spec)} passed to Portfolio, expect Aggregate, str or dict.')

            if a is not None:
                # deals with total in DataFrame intput mode
                self.agg_list.append(a)
                self.unit_names.append(agg_name)
                self.__setattr__(agg_name, a)
                mixed = a.stats_df['mixed']
                ma.add_fs(mixed[('freq', 'ex1')], mixed[('freq', 'ex2')], mixed[('freq', 'ex3')],
                          mixed[('sev',  'ex1')], mixed[('sev',  'ex2')], mixed[('sev',  'ex3')])
                max_limit = max(max_limit, np.max(np.array(a.limit)))

        self.unit_names_ex = self.unit_names + ['total']
        self.unit_name_pipe = "|".join(self.unit_names_ex)
        for n in self.unit_names:
            # unit names cannot equal total
            if n == 'total':
                raise ValueError('Line names cannot equal total, it is reserved for...total')

        # value_type: derived, the unanimous sign-convention role of the
        # units. Mixed loss/payoff units have no coherent "more is worse /
        # more is better" reading, so they are rejected here, before any
        # expensive update. Empty portfolio -> neutral default loss.
        roles = {a._is_loss_value for a in self.agg_list}
        if len(roles) == 2:
            loss_label, payoff_label = value_type_label(True), value_type_label(False)
            loss_units = [a.name for a in self.agg_list if a._is_loss_value]
            payoff_units = [a.name for a in self.agg_list if not a._is_loss_value]
            raise ValueError(
                f"Portfolio {self.name!r}: mixed value_type across units -- "
                f"{loss_label}: {loss_units}, {payoff_label}: {payoff_units}. "
                f"A portfolio cannot mix {loss_label} and {payoff_label} units.")
        self._is_loss_value = roles.pop() if roles else True

        # Canonical ``stats_df``: per-unit columns + ``mixed`` +
        # ``empirical`` + ``error``. Mirror of ``Aggregate.stats_df``
        # shape, minus the ``independent`` column (an
        # Aggregate-frequency-mixing concept with no portfolio-level
        # analog). ``empirical`` and ``error`` start NaN; ``update``
        # populates them after the FFT.
        self._build_stats_df(ma, max_limit)
        # future storage
        self.density_df = None
        self.independent_density_df = None
        self._augmented_dfs: dict[str, pd.DataFrame] = {}
        self._last_applied_distortion_name: str | None = None
        # default natural-allocation method (linear; see ``allocation_method``)
        self._allocation_method: str = 'linear'
        self._certified_bounded: bool = False
        self.independent_stats_df = None
        self.padding = 0
        # GridDistribution view over the portfolio total grid; owns the var/tvar
        # kernel cache. Rebuilt (set None -> lazily) when the density changes.
        self._dist = None
        self._cdf = None
        self._pdf = None
        self.bs = 0
        self.log2 = 0
        self.ex = 0
        self.last_update = 0
        self.hash_rep_at_last_update = ''
        self._distortion = None
        # portfolio reinsurance reporting caches (end-to-end gcn); rebuilt
        # lazily, invalidated on update. See dev/reins-reporting.md.
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_describe = None
        self.sev_calc = ''
        self._remove_fuzz = 0
        self.discretization_calc = ''
        self.normalize = None
        self._unit_renamer = None
        # if created by uw it stores the program here
        self.program = ''
        self.distortions = None
        self.distortion_df = None
        self.calibration_df = None
        self.figure = None
        # bucket/window reporting (set by best_window / update)
        self._bs_window_df = None
        self._bs_clip = None            # portfolio far-tail clip record or None
        self._bs_raw = None             # pre-dyadic-round bs (auto-size only) or None
        self._combine_x_min = None      # windowed combine origin (Plan B) or None

        # for consistency with Aggregates
        self.agg_m = self.stats_df.loc[('agg', 'ex1'), 'total']
        self.agg_cv = self.stats_df.loc[('agg', 'cv'), 'total']
        self.agg_skew = self.stats_df.loc[('agg', 'skew'), 'total']
        # variance and sd come up in exam questions
        self.agg_sd = self.agg_m * self.agg_cv
        self.agg_var = self.agg_sd * self.agg_sd
        # these are set when the object is updated
        self.est_m = self.est_cv = self.est_skew = self.est_sd = self.est_var = 0

        self.validation_eps = get_settings().validation.eps

    def help(self, regex, lod='short', values='short', fmt='auto'):
        """
        Lookup help on methods and properties matching ``regex``.

        Three orthogonal axes: ``lod`` (``'terse'|'short'|'all'``) controls how
        much docstring is shown; ``values`` (``'none'|'short'|'all'``) how much
        of each value or no-argument call result (a ``DataFrame`` / ``Series``
        is headed to 5 rows under ``'short'``); ``fmt``
        (``'auto'|'text'|'ansi'|'html'``) the render target (``auto`` = ANSI in
        Jupyter, plain text in a terminal). See
        :func:`aggregate.utilities.agg_help`.
        """
        agg_help(self, regex, lod=lod, values=values, fmt=fmt)

    def add_exa_sample(self, sample, S_calculation='forwards'):
        """Compute a sample-based ``density_df`` with ``E[X_i | X]`` from a sample.

        Thin wrapper over :func:`aggregate._portfolio_sample.add_exa_sample`
        (the dependence subsystem -- the switcheroo's exeqa-from-sample kernel).
        Returns a frame shaped like ``density_df`` for swapping in. See the free
        function for the full algorithm.
        """
        return _smpl.add_exa_sample(self, sample, S_calculation=S_calculation)

    @staticmethod
    def create_from_sample(name, sample_df, bs, log2=16, **kwargs):
        """
        Create from a multivariate sample, update with bs, execute switcheroo,
        and return new Portfolio object.

        OED: switcheroo, n. a change of position or an exchange, esp. one intended
        to surprise or deceive; a reversal or turn-about; spec. an unexpected change
        or ‘twist’ in a story. Also attributive, reversible, reversed.

        """
        logger.info(f'Creating Porfolio {name} from sample_df. Handles adding total and converting to floats.')
        port = Portfolio(name, sample_df)
        logger.info(f'Updating with bs={bs}, log2={log2}, remove_fuzz=True')
        port.update(bs=bs, log2=log2, remove_fuzz=True, **kwargs)
        # archive the original density_df
        port.independent_density_df = port.density_df.copy()
        # execute switeroo
        logger.info('Creating exa_sample and executing switcheroo')
        port.density_df = port.add_exa_sample(sample_df)
        # update total stats — snapshot the pre-switcheroo stats_df and
        # recompute the empirical agg-total moments from the new density_df.
        logger.info('Updating total statistics (WARNING: these are now empirical)')
        port.independent_stats_df = port.stats_df.copy()
        # Same de-fuzzed ``xsden_to_mwrangler`` convention as
        # ``Portfolio.update`` and ``Aggregate.update_work``.
        _xs = port.density_df['loss'].values
        _p = port.density_df['p_total'].values
        _p_clean = remove_fuzz_util(_p)
        _mw = xsden_to_mwrangler(_xs, _p_clean)
        _ex1, _ex2, _ex3 = _mw.noncentral
        port.est_m, port.est_cv, port.est_skew = _mw.mcvsk
        port.ex = port.est_m
        port.est_sd = port.est_m * port.est_cv
        port.est_var = port.est_sd ** 2
        port._write_empirical_stats(_ex1, _ex2, _ex3)
        # return new created object
        logger.info('Returning new Portfolio object')
        return port

    def create_frequency(self):
        """Materialize the per-unit and total claim-count distributions.

        Builds a new :class:`Portfolio` with one count-distribution unit per
        constituent aggregate (each the output of
        :meth:`Aggregate.create_frequency`). The portfolio total is then the
        **total claim count across all units** -- a useful object in its own
        right -- and every per-unit row is that unit's count distribution.

        Returns
        -------
        Portfolio
            A built portfolio named ``f'{self.name}.freq'`` whose units are the
            per-unit count distributions and whose total is the total count.
            Use e.g. ``.summary_df`` for per-unit and total count moments /
            percentiles.

        Examples
        --------
        >>> pf = port.create_frequency()
        >>> pf.summary_df            # per-unit + total count moments

        Notes
        -----
        The portfolio's own :attr:`program` is re-parsed to recover each unit's
        raw transformer spec (portfolio units do not retain individual
        programs), and each is paired with its built object's resolved ``n``.
        The per-unit count lines are assembled into a ``port`` DecL program and
        built through the normal ``build`` front door, so grid windowing, ``bs``
        and ``log2`` for the total are sized automatically. The returned object
        is a *snapshot*: rebuild it if the parent changes.

        Raises
        ------
        ValueError
            If the portfolio was built programmatically and carries no
            :attr:`program` to re-parse.
        """
        if not self.program:
            raise ValueError(
                f'create_frequency requires a DecL program to re-parse; '
                f'portfolio {self.name!r} was built programmatically (empty '
                f'program).')
        from .underwriter import build
        name = f'{self.name}.freq'
        _kind, _name, spec = build.parser.parse(self.program)
        n_by_name = {a.name: a.n for a in self.agg_list}
        units = [Aggregate._count_program(sub_spec, n_by_name[sub_name],
                                          f'{sub_name}.freq')
                 for _k, sub_name, sub_spec in spec['spec']]
        program = f'port {name}\n' + '\n'.join('\t' + u for u in units)
        return build(program)

    def sample_compare(self, ax=None):
        """Compare the sample-based portfolio total to the independent
        marginal sum.

        Compares the ``empirical`` agg-total moments from the
        post-switcheroo ``stats_df`` against the pre-switcheroo
        snapshot stored in ``independent_stats_df``.
        """
        from .plots import plot_sample_compare
        return plot_sample_compare(self, ax=ax)

    def sample_density_compare(self, fuzz=0):
        """
        Compare from density_df
        """
        bit = pd.concat((self.independent_density_df.filter(regex='p_'),
                         self.density_df.filter(regex='p_total|exeqa_')),
                        axis=1, keys=['independent', 'sample'])
        bit = bit.loc[(bit[('independent', 'p_total')] > fuzz) + (bit[('sample', 'p_total')] > fuzz)]
        bit[('', 'difference')] = bit[('independent', 'p_total')] - bit[('sample', 'p_total')]
        return bit

    def allocation_bounds(self, *, a=0, p=0, units=None, s_floor=1e-14):
        """
        Natural-allocation premium ranges by unit, as a function of the
        total premium.

        Constructs an :class:`~aggregate.bounds.AllocationBounds` object:
        for any total premium P within the feasible range it returns the
        lower/upper bound on each unit's natural allocation over all
        distortions pricing the total to P, together with the achieving
        biTVaR distortions. Construction is P-independent (exact convex
        hulls of the ``(TVaR_p, a_i(p))`` curves); evaluating at a premium
        is a cheap slice.

        With assets specified (``a`` or ``p``) the total is bounded at
        ``X ∧ a``: default states ``X >= a`` collapse to a single atom
        carrying the *linear* natural allocation ``a · E[X_i/X | X >= a]``
        (the collapse is built inside ``AllocationBounds`` from the
        ``exi_xgta_*`` columns); the feasible premium range becomes
        ``[E[X ∧ a], a]``. With neither given the total is unbounded.

        Replaces the pre-1.0 ``pricing_bounds`` method (removed at
        1.0.0a36).

        Parameters
        ----------
        a : float, default 0
            Asset level, snapped to the loss grid. ``0`` means
            unspecified.
        p : float, default 0
            Probability level: resolves ``a = q(p)`` when ``a`` is
            unspecified. Both zero gives the unbounded total.
        units : list of str, optional
            Unit names to include. Default: all of ``unit_names``.
        s_floor : float, default 1e-14
            Tail-probability floor below which curve vertices are dropped
            as FFT noise; see :class:`~aggregate.bounds.AllocationBounds`.

        Returns
        -------
        AllocationBounds
            Call it (or its ``bounds`` method) with one or more premiums::

                ab = port.allocation_bounds(p=0.995)
                ab.bounds([1200, 1300])    # (P, unit) -> lower/upper/width
                ab.bitvars(1200)           # achieving (p0, p1, w1)
        """
        from .bounds import AllocationBounds
        if a == 0 and p == 0:
            a = np.inf
        elif a == 0:
            a = self.q(p)
        else:
            a = self.snap(a)
        return AllocationBounds(self, a=a, units=units, s_floor=s_floor)

    def pricing_bounds(self, y_sources, *, a=0, p=0, s_floor=1e-14, n_grid=1024):
        """
        Price ranges of other risks consistent with pricing this total.

        Constructs a :class:`~aggregate.bounds.PricingBounds` object with this
        Portfolio's total as the reference risk ``X``: for any premium P
        within the feasible range it returns the lower/upper bound on the
        price of each risk in ``y_sources`` over all distortions pricing the
        total to P, together with the achieving biTVaR distortions.

        With assets specified (``a`` or ``p``) both ``X`` and the ``y_sources``
        are priced capped at ``min(., a)`` — the well-conditioned regime
        (deep-tail vertices of unbounded heavy-tailed risks are FFT noise and
        make the upper bound grid-sensitive; cap, or raise ``s_floor``).

        Parameters
        ----------
        y_sources : source or list/dict of sources
            The risk(s) ``Y`` to price.  Each may be an ``Aggregate``,
            ``Portfolio``, pmf ``Series``, ``'uniform'``, or a TVaR source;
            a dict supplies explicit names.
        a : float, default 0
            Asset level, snapped to the loss grid.  ``0`` means unspecified.
        p : float, default 0
            Probability level: resolves ``a = q(p)`` when ``a`` is
            unspecified.  Both zero gives the unbounded total.
        s_floor : float, default 1e-14
            Tail-probability floor below which curve vertices are dropped;
            see :class:`~aggregate.bounds.PricingBounds`.
        n_grid : int, default 1024
            Fallback p-grid size for all-closed-form sources.

        Returns
        -------
        PricingBounds
            ::

                pb = port.pricing_bounds(layer_agg, p=0.99)
                pb.bounds([1200, 1300])    # (P, risk) -> lower/upper/width
                pb.bitvars(1200)           # achieving (p0, p1, w1)
        """
        from .bounds import PricingBounds
        if a == 0 and p == 0:
            a = np.inf
        elif a == 0:
            a = self.q(p)
        else:
            a = self.snap(a)
        return PricingBounds(self, y_sources, a=a, s_floor=s_floor,
                             n_grid=n_grid)

    @property
    def distortion(self):
        return self._distortion

    def remove_fuzz(self, df=None, eps=0, force=False, log=''):
        """
        remove fuzz at threshold eps. if not passed use np.finfo(float).eps.

        Apply to self.density_df unless df is not None

        Only apply if self.remove_fuzz or force
        :param eps:
        :param df:  apply to dataframe df, default = self.density_df
        :param force: do regardless of self.remove_fuzz
        :return:
        """

        if df is None:
            df = self.density_df
        if eps == 0:
            eps = np.finfo(float).eps

        if self._remove_fuzz or force:
            logger.debug(f'Portfolio.remove_fuzz | Removing fuzz from {self.name} dataframe, caller {log}')
            # ``remove_fuzz`` returns a de-fuzzed copy; write the float columns
            # back *in place* so the mutation reaches ``self.density_df`` (the
            # default ``df``). Rebinding the local ``df`` would silently no-op.
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = remove_fuzz_util(df, eps)[float_cols]

    def __repr__(self):
        """
        Goal unmbiguous
        :return:
        """
        # return str(self.to_dict())
        # this messes up when port = self has been enhanced...

        # cannot use ex, etc. because object may not have been updated
        return f'{self.name} at {super().__repr__()}'

    def _validation_passes(self) -> bool:
        """Whether the portfolio clears validation (clean *or* cleanly reinsured).

        Display twin of :meth:`Aggregate._validation_passes` -- lets
        :meth:`_repr_html_` and :meth:`qd` stay silent on a pass and flag only a
        genuine failure.
        """
        r = self.valid
        return bool(r == Validation.NOT_UNREASONABLE or (r & Validation.REINSURANCE))

    def _text_info_blob(self) -> str:
        """Short plain-text intro for :meth:`qd` -- identity, unit count, grid.

        The portfolio twin of :meth:`Aggregate._text_info_blob`; no validation
        line (the caller flags a failure separately).
        """
        _n = len(self.agg_list)
        _s = '' if _n == 1 else 's'
        s = [f'Portfolio object: {self.name}',
             f'Portfolio contains {_n} aggregate component{_s}.']
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1 / self.bs)}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.')
        return '\n'.join(s)

    def _repr_html_(self):
        """HTML view: short intro, the ``summary_df`` headline, the ``tail_df``
        return-period table, and a validation flag only on failure.
        """
        _n = len(self.agg_list)
        _s = '' if _n == 1 else 's'
        s = [f'<h3>Portfolio object: {self.name}</h3>',
             f'<p>Portfolio contains {_n} aggregate component{_s}.']
        if self.bs > 0:
            s.append(f'Updated with bucket size {self.bs:.6g} and log2 = {self.log2}.</p>')
        else:
            s.append('</p>')
        if self.density_df is not None and not self._validation_passes():
            s.append('<p>Validation: <div style="color: #f00; font-weight:bold;">fails</div>'
                     f'<pre>\n{self.validation_explanation}</pre></p>')
        fmt = lambda x: f'{x:,.5g}'
        out = ['\n'.join(s),
               '<h4>Summary</h4>',
               self.summary_df.to_html(float_format=fmt, na_rep='')]
        td = self.tail_df()
        if td is not None:
            out.append('<h4>Tail &mdash; return period (exact, not simulated)</h4>')
            out.append(td.to_html(float_format=fmt, na_rep=''))
        return '\n'.join(out)

    def __str__(self):
        """ Default behavior """
        ex = float(self.stats_df.loc[('agg', 'mean'), 'total'])
        empex_raw = self.stats_df.loc[('agg', 'mean'), 'empirical']
        if pd.isna(empex_raw):
            empex = np.nan
            isupdated = False
        else:
            empex = float(empex_raw)
            isupdated = True

        s = [f'Portfolio object         {self.name:s}',
             f'Theoretic expected loss  {ex:,.1f}',
             f'Estimated expected loss  {empex:,.1f}',
             f'Error                    {empex / ex - 1:.6g}'
             ]

        s.append(
             f'Updated                  {isupdated}'
        )

        if self.bs > 0:
            if self.bs > 1:
                s.append(f'bs                       {self.bs}')
            else:
                s.append(f'bs                       1 / {int(1/self.bs)}')
            s.append(f'log2                     {self.log2}')
            s.append(f'validation_eps           {self.validation_eps}')
            s.append(f'padding                  {self.padding}')
            s.append(f'sev_calc                 {self.sev_calc}')
            s.append(f'normalize                {self.normalize}')
            s.append(f'remove_fuzz              {self._remove_fuzz}')
            s.append(f'distortion               {repr(self._distortion)}')

        if isupdated:
            s.append('')
            with pd.option_context('display.width', 140, 'display.float_format', lambda x: f'{x:,.5g}'):
                # get it on one row
                s.append(str(self.summary_df))
        # s.append(super(Portfolio, self).__repr__())
        return '\n'.join(s)

    def __hash__(self):
        """
        hashing behavior
        :return:
        """
        return hash(repr(self.__dict__))

    def __iter__(self):
        """
        make Portfolio iterable: for each x in Portfolio

        :return:
        """
        return iter(self.agg_list)

    def __getitem__(self, item):
        """
        allow Portfolio[slice] to return bits of agg_list

        :param item:
        :return:
        """
        if type(item) == str:
            return self.agg_list[self.unit_names.index(item)]
        return self.agg_list[item]

    @property
    def tail_class(self) -> TailClass:
        """The portfolio's worst-of aggregate :class:`~aggregate.tail.TailClass`.

        Under independence the tail of a sum is governed by the thickest
        summand (subexponential closure; and for the lighter rungs the
        convolution decay rate equals the slowest = thickest), so this is the
        ``max`` over the unit aggregate rungs. ``UNKNOWN`` on any unit poisons
        the result (it is not treated as a thickness). A ``_certified_bounded``
        override forces :attr:`~aggregate.tail.TailClass.BOUNDED`.
        """
        if getattr(self, '_certified_bounded', False):
            return TailClass.BOUNDED
        worst = TailClass.BOUNDED
        for a in self.agg_list:
            c = a.tail_class.agg
            if c == TailClass.UNKNOWN:
                return TailClass.UNKNOWN
            worst = max(worst, c)
        return worst

    @property
    def bounded(self) -> bool:
        """Whether every unit's aggregate has bounded support.

        Derived view: ``True`` iff :attr:`tail_class` is
        :attr:`~aggregate.tail.TailClass.BOUNDED`, i.e. each unit is bounded
        (correct under independence). Set ``self.bounded = True`` to certify the
        portfolio bounded (e.g. when a unit's auto-detection is conservatively
        ``False`` but the modeller knows the support is capped).
        """
        return self.tail_class == TailClass.BOUNDED

    @bounded.setter
    def bounded(self, value: bool) -> None:
        if value is not True and value is not False:
            raise ValueError('bounded must be True (certify) or False (reset)')
        self._certified_bounded = bool(value)

    def _tail_driver(self):
        """Return ``(worst_class, [driver unit name(s)])`` for the portfolio tail.

        The driver unit(s) are those whose aggregate rung equals the worst-of
        rung. Returns an empty driver list when the worst class is UNKNOWN or
        the portfolio is certified bounded.
        """
        worst = self.tail_class
        if worst in (TailClass.UNKNOWN,) or getattr(self, '_certified_bounded', False):
            return worst, []
        drivers = [a.name for a in self.agg_list if a.tail_class.agg == worst]
        return worst, drivers

    @property
    def tail_description(self) -> str:
        """One aligned line: the portfolio's worst-of aggregate tail class.

        E.g. ``aggregate tail           subexponential (driver: B)``. Derived
        from :attr:`tail_class`.
        """
        worst, drivers = self._tail_driver()
        phrase = _tail.tail_class_label(worst)
        if drivers:
            phrase += f' (driver: {", ".join(drivers)})'
        return f'{"aggregate tail":<25}{phrase}'

    @property
    def tail_explanation(self) -> str:
        """Sentence: per-unit aggregate tail classes and the named driver unit(s).

        Derived from each unit's :attr:`Aggregate.tail_class`.
        """
        worst, drivers = self._tail_driver()
        parts = [f'{a.name}: {_tail.tail_class_label(a.tail_class.agg)}'
                 for a in self.agg_list]
        units = '; '.join(parts)
        if worst == TailClass.UNKNOWN:
            return (f'Portfolio aggregate tail undetermined (a unit is an '
                    f'unrecognised or numeric-only family). Units -- {units}.')
        driver_txt = (f', driven by {", ".join(drivers)}' if drivers else '')
        return (f'Portfolio aggregate tail is {_tail.tail_class_label(worst)} '
                f'(worst-of under independence{driver_txt}). Units -- {units}.')

    @property
    def bs_window_df(self) -> 'pd.DataFrame':
        """Curated, read-only view of the portfolio combine grid (``[bs-reporting]``).

        Delegated to :func:`_bucket_window.port_bs_window_df`. Returns ``None``
        before the grid is sized. See :attr:`bs_description` / :attr:`bs_explanation`.
        """
        return _bucket_window.port_bs_window_df(self)

    @property
    def bs_description(self) -> str:
        """One-line summary of the shared portfolio combine grid (``[bs-reporting]``).

        The realised ``(bs, log2, x_min)`` and grid ``x_max`` of the resolution +
        window-width combine (``best_window``); ``'portfolio grid not sized yet'``
        before the grid is built.
        """
        df = getattr(self, '_bs_window_df', None)
        if df is None or 'used' not in df.index:
            return 'portfolio grid not sized yet (call update())'
        u = df.loc['used']
        top = float(u['x_min']) + (1 << int(u['log2'])) * float(u['bs'])
        txt = (f'portfolio grid: bs={float(u["bs"]):g}, log2={int(u["log2"])}, '
               f'x_min={float(u["x_min"]):g} (x_max={top:g})')
        clip = getattr(self, '_bs_clip', None)
        if clip is not None:
            cm = clip.get('clipped_mass', float('nan'))
            cm_txt = f'~{cm:.3g}' if np.isfinite(cm) else 'a sliver'
            txt += f'; clips {cm_txt} of the tail (raise log2 to {int(clip["need_log2"])})'
        return txt

    @property
    def bs_explanation(self) -> str:
        """Verbose prose explaining the portfolio combine grid (``[bs-reporting]``).

        Walks the grid choice in the reporting template: the portfolio's per-side
        tail classes and log2 budget; the per-unit tails; the recommended
        **window width** and the candidate widths it was chosen from (portfolio
        method of moments, RMS-of-units, sum-of-units, single big jump); the
        raw-to-dyadic ``bs`` rounding; any natural support bounds and the total
        concentration; the realised ``x_min`` / ``x_max``; and a closing log2
        suggestion when a far-tail clip occurred (raise). ``'portfolio grid not
        sized yet'`` before :meth:`update`.

        Notes
        -----
        "Window width" (the realised ``W = x_max - x_min`` and the ``mm`` / ``rms``
        / ``sum`` / ``sbj`` candidates) replaces the older "span" wording -- the
        candidates are all window widths the combine chooses between.
        """
        df = getattr(self, '_bs_window_df', None)
        if df is None or 'used' not in df.index:
            return 'Portfolio grid not sized yet (call update()).'
        u = df.loc['used']
        log2 = int(u['log2'])
        bs = float(u['bs'])
        x_min = float(u['x_min'])
        x_max = float(u['x_max'])
        W = float(u['W']) if 'W' in u.index and np.isfinite(float(u['W'])) \
            else x_max - x_min

        def _as_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float('nan')

        td = self.tail_behavior_df
        tot = td.loc['total']
        parts = [f'The portfolio tail is {tot["left_tail"]} left / '
                 f'{tot["right_tail"]} right. Log2 is {log2}.']

        unit_rows = [ix for ix in td.index if ix != 'total']
        if unit_rows:
            unit_txt = ' and '.join(
                f'{ix} {td.loc[ix, "left_tail"]}/{td.loc[ix, "right_tail"]}'
                for ix in unit_rows)
            parts.append(f'The unit tails are: {unit_txt}.')

        if {'mm', 'rms', 'sum', 'sbj'}.issubset(df.index):
            parts.append(
                f'The recommended window width {W:g} is based on portfolio method '
                f'of moments {float(df.loc["mm", "W"]):g}, RMS(units) '
                f'{float(df.loc["rms", "W"]):g}, sum(units) '
                f'{float(df.loc["sum", "W"]):g}, and single big jump of '
                f'{float(df.loc["sbj", "W"]):g}.')

        raw = getattr(self, '_bs_raw', None)
        if raw is not None:
            parts.append(
                f'The window produces a raw bs {raw:g} which dyadically rounds to '
                f'{bs:g} producing a final {W:g} window width.')

        lo, hi = _as_float(tot['min']), _as_float(tot['max'])
        if tot['left_tail'] == 'bounded' and np.isfinite(lo):
            parts.append(f'It has a natural lower support bound of {lo:g}.')
        if tot['right_tail'] == 'bounded' and np.isfinite(hi):
            parts.append(f'It has a natural upper support bound of {hi:g}.')

        conc_flag, conc_cv = _tail.concentration(float(self.agg_m), float(self.agg_sd))
        if conc_flag and conc_cv is not None and np.isfinite(conc_cv):
            parts.append(f'The distribution is concentrated with a CV of {conc_cv:g}.')

        parts.append(f'The recommended x_min is {x_min:g} resulting in '
                     f'x_max of {x_max:g}.')

        clip = getattr(self, '_bs_clip', None)
        if clip is not None:
            cm = clip.get('clipped_mass', float('nan'))
            cm_txt = f'~{cm:.3g}' if np.isfinite(cm) else 'a sliver'
            parts.append(
                f'The combined support exceeds the grid ({cm_txt} of the mass '
                f'clipped, a reported deficit not normalized); the analysis '
                f'suggests increasing log2 to {int(clip["need_log2"])}.')
        return ' '.join(parts)

    @property
    def tail_behavior_df(self) -> 'pd.DataFrame':
        """Per-unit aggregate tail rows plus a worst-of ``total`` (``[bs-reporting]``).

        One row per unit -- that unit's aggregate-level
        :attr:`Aggregate.tail_behavior_df` row (support ``min`` / ``max``, ``left_tail`` /
        ``right_tail`` classes, ``bounded``, ``concentrated`` / ``cv``) -- and a
        ``total`` row carrying the portfolio worst-of decay (computed *per side*)
        and the total-moment concentration. Spec-only for the per-unit rows
        (valid before :meth:`update`). See :attr:`tail_description` /
        :attr:`tail_explanation` for the narrative.

        Notes
        -----
        The ``total`` row mixes two notions of support **by design**: per-unit
        rows show *structural* support (``inf`` at an unbounded end), while the
        ``total`` ``min`` / ``max`` show the *realised* combine grid extent (the
        ``used`` row of :attr:`bs_window_df`, always finite) -- the grid is the
        operative support for the portfolio total. Because that grid is finite on
        both ends, the per-side tail classes are **not** derived from ``min`` /
        ``max`` finiteness; each side's class is the worst-of (thickest) over the
        units' per-side rungs. So a non-negative book reports a ``bounded`` left
        tail (every unit's left is bounded) while the right side carries the
        heaviest unit's right-tail decay. ``min`` / ``max`` stay ``INFO_NA``
        before the grid is sized.

        Returns
        -------
        pandas.DataFrame
            Indexed by unit name, with a final ``total`` row.
        """
        rows = {a.name: a.tail_behavior_df.loc['aggregate'] for a in self.agg_list}
        df = pd.DataFrame(rows).T
        worst = self.tail_class
        conc_flag, conc_cv = _tail.concentration(float(self.agg_m), float(self.agg_sd))
        total = pd.Series(INFO_NA, index=df.columns, dtype=object)

        def _worst_side(col):
            """Worst-of (thickest) per-side rung over the unit rows; UNKNOWN poisons."""
            classes = [_tail.tail_class_from_label(v) for v in df[col]]
            if any(c == TailClass.UNKNOWN for c in classes):
                return _tail.tail_class_label(TailClass.UNKNOWN)
            return _tail.tail_class_label(max(classes, default=TailClass.BOUNDED))

        if 'left_tail' in df.columns:
            total['left_tail'] = _worst_side('left_tail')
        if 'right_tail' in df.columns:
            total['right_tail'] = _worst_side('right_tail')
        if 'bounded' in df.columns:
            total['bounded'] = (worst == TailClass.BOUNDED)
        if 'concentrated' in df.columns:
            total['concentrated'] = bool(conc_flag)
        if 'cv' in df.columns and conc_cv is not None:
            total['cv'] = float(conc_cv)
        if 'note' in df.columns:
            total['note'] = 'portfolio worst-of (independence)'
        # Complete min/max on the total row from the realised combine grid (the
        # bs_window_df 'used' row) -- available only after update; leave INFO_NA
        # otherwise so the total support reads honestly as not-yet-sized.
        bwdf = self.bs_window_df
        if bwdf is not None and 'used' in bwdf.index:
            if 'min' in df.columns:
                total['min'] = float(bwdf.loc['used', 'x_min'])
            if 'max' in df.columns:
                total['max'] = float(bwdf.loc['used', 'x_max'])
        df.loc['total'] = total
        return df

    @property
    def allocation_method(self) -> str:
        """Natural-allocation method: ``'linear'`` (default) or ``'lifted'``.

        Drives :meth:`price` (and downstream readouts) when no explicit
        ``allocation=`` is passed. Set this once on the portfolio rather
        than threading the argument through every call. The setter
        invalidates the ``augmented_df`` cache because the lifted and
        linear frames differ on the right edge.
        """
        return self._allocation_method

    @allocation_method.setter
    def allocation_method(self, value: str) -> None:
        if value not in ('linear', 'lifted'):
            raise ValueError(
                f"allocation_method must be 'linear' or 'lifted', not {value!r}")
        if value != self._allocation_method:
            self._augmented_dfs.clear()
            self._allocation_method = value

    @property
    def value_type(self):
        """Sign convention for the portfolio: ``'loss'`` or ``'payoff'``.

        Derived, not user-set: the unanimous
        :attr:`~aggregate.distributions.Aggregate.value_type` of the units
        (mixed books are rejected at construction; an empty portfolio
        defaults to loss). Read-only — there is no setter, unlike
        ``Aggregate``. It is **inert for the distribution itself**: density,
        moments, quantiles and allocation do not depend on it. It is consumed
        only by distortion / pricing, where a payoff portfolio is negated /
        the dual distortion applied. Returns the configured label for the
        role (``[labels]`` in the config); pricing code must branch on
        ``_is_loss_value``, never on the label text.
        """
        return value_type_label(self._is_loss_value)

    @property
    def info(self):
        """Fixed-layout multi-line summary string.

        Every row is always present, in the same order, for every
        ``Portfolio``; a value that is not (yet) available -- e.g. the grid
        block before ``update`` -- renders as ``n/a``. The row catalogue is
        documented in ``dev/info-strings.rst``. Shares the label/value
        convention (:func:`aggregate.constants.info_row`) with ``Aggregate``
        and ``Distortion``.
        """
        updated = self.bs > 0
        if updated:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1/self.bs)}'
        else:
            bss = INFO_NA
        # Realized grid window, read off the density index (the signed path
        # may start below 0; the non-signed grid starts at 0).
        if self.density_df is not None:
            x_min = f'{float(self.density_df.index[0]):,.6g}'
            x_max = f'{float(self.density_df.index[-1]) + self.bs:,.6g}'
        else:
            x_min = x_max = INFO_NA
        # premium / expected loss / loss ratio: populated when the units
        # carry a premium. Expected loss is empirical (est_m); for a payoff
        # book est_m is E[PnL], so E[loss] = premium - E[PnL].
        prem = float(self.stats_df.loc[('meta', 'prem'), 'total'])
        e_loss = None
        if updated:
            if self._is_loss_value:
                e_loss = float(self.est_m)
            elif prem > 0:
                e_loss = prem - float(self.est_m)
        h = self.hash_rep_at_last_update
        rows = [
            ('portfolio object name', self.name),
            ('value_type', self.value_type),
            ('aggregate objects', f'{len(self.unit_names):d}'),
            ('allocation_method', self.allocation_method),
            ('bs', bss),
            ('log2', self.log2 if updated else INFO_NA),
            ('padding', self.padding if updated else INFO_NA),
            ('sev_calc', self.sev_calc if updated else INFO_NA),
            ('normalize', self.normalize if updated else INFO_NA),
            ('x_min', x_min),
            ('x_max', x_max),
            ('premium', f'{prem:,.6g}' if prem > 0 else INFO_NA),
            ('expected loss', f'{e_loss:,.6g}' if e_loss is not None else INFO_NA),
            ('loss ratio', f'{e_loss / prem:.1%}'
             if prem > 0 and e_loss is not None else INFO_NA),
        ]
        s = [info_row(label, value) for label, value in rows]
        # Footer: tail, bounded, last update, id -- matching Aggregate.info
        # (tail near the end; ``id`` was labelled ``hash``).
        s.append(self.tail_description)
        s.append(info_row('bounded', self.bounded))
        s.append(info_row('last update',
                          self.last_update if self.last_update != 0 else INFO_NA))
        s.append(info_row('id', f'{h:x}' if isinstance(h, int) else INFO_NA))
        return '\n'.join(s)

    def _reins_after_label(self):
        """Portfolio-wide heading for the after-reins column in ``validation_df``.

        Aggregates the per-unit :meth:`Aggregate._reins_after_label`
        across the book. Returns ``None`` when **no** unit carries
        reinsurance (the legacy theory/empirical validation view
        applies). When exactly one cession kind appears across all
        ceding units returns that label (``Net`` / ``Ceded``);
        otherwise returns ``Output`` — the umbrella label for mixed output —
        so the whole table can share one column layout.
        """
        labels = {lbl for a in self
                  if (lbl := a._reins_after_label()) is not None}
        if not labels:
            return None
        if len(labels) == 1:
            return labels.pop()
        return REINS_LABEL_OUTPUT

    @property
    def validation_df(self):
        """Theoretic-and-empirical moment-error stats (the QA view).

        The validation frame: theoretical moments vs the realised FFT estimate,
        with noise-aware relative errors. Surfaced on demand and in :meth:`qd`
        when a unit (or the total) *fails* validation; the daily-driver headline
        is :attr:`summary_df`.

        Reads from the canonical ``stats_df``: theoretical moments from
        the ``total`` column, empirical from ``empirical``, errors from
        ``error``. The output shape mirrors ``Aggregate.validation_df`` — one
        ``Freq``/``Sev``/``Agg`` row block per unit + ``total``.

        Two display modes, chosen at the **portfolio** level so the unit
        blocks and the ``total`` block always share one column layout:

        * **No unit has reinsurance** -- validation view, columns
          ``EX | Est EX | Err EX | CV | Est CV | Err CV | Sk | Est Sk``.
        * **Any unit has reinsurance** -- economic view, columns
          ``Subject EX | <label> EX | Change EX | ...`` where ``<label>``
          is the portfolio-wide :meth:`_reins_after_label`. Every unit —
          including those with no cession — is rendered in this layout
          (forced via ``Aggregate._describe``) so the table aligns.

        The spread column is **CV** normally, but **SD** when the portfolio is
        signed (any unit is a P&L / negative-support ``ssev``/``dsev`` unit),
        since CV is unstable near a zero mean. The choice is portfolio-wide —
        CV and SD cannot be mixed in one frame — so every unit block is forced
        to SD (via ``Aggregate._describe(force_sd=...)``) and the total SD is
        read robustly from the second moment, not ``mean * cv``.
        """
        _total = self.stats_df['total']
        emp = self.stats_df['empirical']
        rlabel = self._reins_after_label()

        # Spread column: CV by default, **SD** when the portfolio is signed.
        # CV = sd / mean is unstable/meaningless when a mean can be ~0 (a P&L
        # unit straddling break-even), so a signed portfolio reports the
        # standard deviation instead. CV and SD cannot be mixed in one frame,
        # so the choice is made ONCE here, at the portfolio level: if ANY unit
        # is signed (``self._signed()``) the whole table -- every unit block
        # and the total -- switches to SD, with the units forced via
        # ``Aggregate._describe(force_sd=...)``. SD is taken robustly from the
        # second moment, ``sqrt(ex2 - mean**2)``, never ``mean * cv``.
        use_sd = self._signed()
        spread = 'SD' if use_sd else 'CV'

        def _theo_spread(comp):
            if not use_sd:
                return float(_total[(comp, 'cv')])
            m = float(_total[(comp, 'mean')])
            return float(np.sqrt(max(float(_total[(comp, 'ex2')]) - m * m, 0.0)))

        def _emp_spread(comp):
            if not use_sd:
                return float(emp[(comp, 'cv')])
            m = float(emp[(comp, 'mean')])
            return float(np.sqrt(max(float(emp[(comp, 'ex2')]) - m * m, 0.0)))

        df = pd.DataFrame(
            {
                'EX': [float(_total[('freq', 'ex1')]),
                       float(_total[('sev',  'ex1')]),
                       float(_total[('agg',  'ex1')])],
                spread: [_theo_spread('freq'),
                         _theo_spread('sev'),
                         _theo_spread('agg')],
                'Sk': [float(_total[('freq', 'skew')]),
                       float(_total[('sev',  'skew')]),
                       float(_total[('agg',  'skew')])],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'

        # Post-update? Empirical agg moments live in stats_df['empirical'].
        # After the punch-up: portfolio-level sev empirical is also
        # populated (via MomentAggregator off per-unit empirical sev);
        # surface it in the validation_df table too. Under reinsurance the
        # ``total`` column is the (gross) Subject view and ``empirical``
        # the realised after-reins view, exactly mirroring Aggregate.
        emp_agg_m = emp.get(('agg', 'mean'), np.nan)
        if pd.notna(emp_agg_m):
            mid_label = rlabel or 'Est'
            change_label = 'Change' if rlabel else 'Err'
            df.loc['Sev', f'{mid_label} EX'] = float(emp[('sev', 'mean')])
            df.loc['Agg', f'{mid_label} EX'] = float(emp_agg_m)
            df[f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} {spread}'] = _emp_spread('sev')
            df.loc['Agg', f'{mid_label} {spread}'] = _emp_spread('agg')
            df[f'{change_label} {spread}'] = _noise_aware_rel_error(
                df[f'{mid_label} {spread}'], df[spread])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = float(emp[('sev', 'skew')])
            df.loc['Agg', f'{mid_label} Sk'] = float(emp[('agg', 'skew')])
            df = df[['EX', f'{mid_label} EX', f'{change_label} EX',
                     spread, f'{mid_label} {spread}', f'{change_label} {spread}',
                     'Sk', f'{mid_label} Sk']]
        # Subject-column label under reinsurance; plain headings otherwise.
        if rlabel:
            df = df.rename(columns={
                'EX': 'Subject EX', spread: f'Subject {spread}',
                'Sk': 'Subject Sk'})
        # snap floating-point dust to 0 in the moment-value columns for
        # display (e.g. the skew of a symmetric unit); NaN preserved.
        # Change/Err columns retain their numeric dust.
        for c in df.columns:
            if (' EX' in c or ' CV' in c or ' SD' in c or ' Sk' in c
                    or c in ('EX', 'CV', 'SD', 'Sk')) \
                    and not (c.startswith('Err ') or c.startswith('Change ')):
                df[c] = _snap_noise(df[c])

        # Force every unit block into the portfolio-wide layout so the
        # concat aligns (units with no cession render in reins view too
        # when ``rlabel`` is set; unsigned units render in SD view via
        # ``force_sd`` when the portfolio is signed).
        t1 = [a._describe(force_reins_label=rlabel, force_sd=use_sd)
              for a in self] + [df]
        t2 = [a.name for a in self] + ['total']
        df = pd.concat(t1, keys=t2, names=['unit', 'X'])
        return df

    @property
    def summary_df(self):
        """At-a-glance risk view -- moments + key percentiles, per unit + total.

        The portfolio analogue of :attr:`Aggregate.summary_df` and the
        daily-driver headline. A ``MultiIndex (unit, X)`` block per unit (each
        unit's own :attr:`Aggregate.summary_df` -- Freq / Sev / Agg moments and
        percentiles) plus a ``total`` block carrying the **Agg row only** (a
        portfolio has no single Freq / Sev). The ``total`` Agg row is the
        portfolio's "what's my number" line: ``E[X] / SD / CV / Skew`` from the
        canonical ``stats_df`` total, and ``p0.01 / p0.50 / p0.99`` from the
        realised portfolio grid (:meth:`q`).

        The moment-error QA frame is now :attr:`validation_df`; the return-period
        table is :attr:`tail_df`; the tail-behavior classifier is
        :attr:`tail_behavior_df`.

        Returns
        -------
        pandas.DataFrame
            ``MultiIndex (unit, X)`` rows; columns ``E[X] | SD | CV | Skew |
            p0.01 | p0.50 | p0.99``.
        """
        blocks = [a.summary_df for a in self]
        keys = [a.name for a in self]
        # ``total`` block: the Agg row only (no portfolio Freq / Sev).
        m = float(self.agg_m)
        sd = float(self.agg_sd)
        total = pd.DataFrame(
            {
                'E[X]': [m],
                'SD': [sd],
                'CV': [Aggregate._cv_or_nan(m, sd)],
                'Skew': [float(self.agg_skew)],
            },
            index=pd.Index(['Agg'], name='X'),
        )
        pcols = [f'p{p:.2f}' for p in SUMMARY_PERCENTILES]
        for p, pc in zip(SUMMARY_PERCENTILES, pcols):
            total[pc] = self.q(p) if self.density_df is not None else np.nan
        for c in ('E[X]', 'SD', 'Skew', *pcols):
            total[c] = _snap_noise(total[c])
        total.attrs['mean'] = m
        blocks.append(total)
        keys.append('total')
        df = pd.concat(blocks, keys=keys, names=['unit', 'X'])
        df.attrs['mean'] = m
        return df

    def tail_df(self, periods=None):
        """Return-period / exceedance table -- per unit plus the portfolio total.

        The portfolio analogue of :meth:`Aggregate.tail_df`. A leading ``unit``
        index level: each unit's aggregate return-period table
        (:meth:`Aggregate.tail_df`) stacked under its name, plus a ``total``
        block computed from the realised portfolio grid (:meth:`q` / :meth:`tvar`
        / :meth:`est_m`). Columns ``p | VaR | TVaR | xsVaR | VaR/Mean``; the
        numbers are exact (FFT grid, not simulated).

        Per-unit *contribution* to the total tail (TVaR allocation) is
        allocation / pricing territory -- see :meth:`price` -- not here.

        Parameters
        ----------
        periods : array_like of float, optional
            Return-period ladder. Defaults to :data:`DEFAULT_RETURN_PERIODS`.

        Returns
        -------
        pandas.DataFrame or None
            ``MultiIndex (unit, T)`` rows. ``None`` before :meth:`update`.
        """
        if self.density_df is None:
            return None
        blocks, keys = [], []
        for a in self:
            ut = a.tail_df(periods)
            if ut is not None:
                blocks.append(ut)
                keys.append(a.name)
        total = return_period_frame(
            self.q, self.tvar, self.est_m, self._is_loss_value, periods)
        blocks.append(total)
        keys.append('total')
        df = pd.concat(blocks, keys=keys, names=['unit', 'T'])
        df.attrs['mean'] = float(self.est_m)
        return df

    # ================================================================
    # Reinsurance reporting (end-to-end gcn; see dev/reins-reporting.md)
    # ================================================================

    def _reins_unit_views(self, a):
        """End-to-end gross / ceded / net aggregate densities for one unit.

        Returns a dict ``{'gross', 'ceded', 'net'}`` of densities on the
        unit's grid. A non-ceding unit contributes ``gross == net ==
        modeled`` and a point mass at 0 for ``ceded``. For a ceding unit the
        views are read from its :meth:`Aggregate.reins_density_df`: when an
        aggregate cover is present the final aggregate-cover net/ceded; else
        the occurrence net/ceded. The ``ceded`` view of a unit carrying
        *both* covers is the final (aggregate-stage) cession; the per-stage
        chain lives in the unit's own ``reins_summary_df``.
        """
        zero = np.zeros_like(a.xs, dtype=float)
        zero[0] = 1.0
        rd = a.reins_density_df
        if rd is None:
            modeled = np.asarray(a.agg_density, dtype=float)
            return {'gross': modeled, 'ceded': zero, 'net': modeled}
        g = rd['p_agg_gross'].to_numpy()
        if a.agg_reins is not None:
            return {'gross': g, 'ceded': rd['p_agg_ceded'].to_numpy(),
                    'net': rd['p_agg_net'].to_numpy()}
        if a.occ_reins is not None:
            return {'gross': g, 'ceded': rd['p_agg_ceded_occ'].to_numpy(),
                    'net': rd['p_agg_net_occ'].to_numpy()}
        return {'gross': g, 'ceded': zero, 'net': g}

    @property
    def reins_density_df(self):
        """Portfolio end-to-end gross / ceded / net aggregate densities.

        Convolves the per-unit end-to-end gross / ceded / net aggregate
        marginals (see :meth:`_reins_unit_views`) under the same independent
        FFT machinery used for the portfolio total, producing three
        portfolio-level marginal aggregate distributions. Units without
        reinsurance contribute their modeled density to ``gross`` and ``net``
        and nothing (point mass at 0) to ``ceded``.

        Columns ``loss``, ``p_agg_gross``, ``p_agg_ceded``, ``p_agg_net``
        (always present). Returns ``None`` when **no** unit cedes.

        Notes
        -----
        Assumes unit independence -- consistent with the existing portfolio
        total. The three portfolio marginals are *separate* distributions
        (portfolio gross / ceded / net loss); they no more satisfy
        ``gross = net (+) ceded`` than the unit-level views do.
        """
        if self._reins_after_label() is None:
            return None
        if self._reins_density_df is not None:
            return self._reins_density_df
        xs = self.density_df['loss'].to_numpy()
        ft_gross = ft_ceded = ft_net = None
        for a in self.agg_list:
            uv = self._reins_unit_views(a)
            fg = ft(uv['gross'], self.padding)
            fc = ft(uv['ceded'], self.padding)
            fn = ft(uv['net'], self.padding)
            if ft_gross is None:
                ft_gross, ft_ceded, ft_net = fg, fc, fn
            else:
                ft_gross = ft_gross * fg
                ft_ceded = ft_ceded * fc
                ft_net = ft_net * fn
        df = pd.DataFrame({'loss': xs}, index=pd.Index(xs, name='loss'))
        df['p_agg_gross'] = np.real(ift(ft_gross, self.padding))
        df['p_agg_ceded'] = np.real(ift(ft_ceded, self.padding))
        df['p_agg_net'] = np.real(ift(ft_net, self.padding))
        self._reins_density_df = df
        return self._reins_density_df

    @property
    def reins_stats_df(self):
        """Per-unit and portfolio-total end-to-end gross / ceded / net moments.

        Rows are ``MultiIndex (view, measure)`` with ``view in
        {gross, ceded, net}`` and ``measure in {ex1, ex2, ex3, mean, cv,
        skew}`` (aggregate-level; there is no per-stage split at the portfolio
        level). Columns are the unit names plus ``total``. Unit columns read
        each unit's end-to-end marginals; ``total`` reads the convolved
        portfolio marginals in :meth:`reins_density_df`. Returns ``None`` when
        no unit cedes.
        """
        if self._reins_after_label() is None:
            return None
        if self._reins_stats_df is not None:
            return self._reins_stats_df
        views = ['gross', 'ceded', 'net']
        measures = ['ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew']
        row_index = pd.MultiIndex.from_product(
            [views, measures], names=['view', 'measure'])

        def moments6(xs, p):
            mw = xsden_to_mwrangler(xs, np.asarray(p, dtype=float))
            return (*mw.noncentral, *mw.mcvsk)

        cols = {}
        for a in self.agg_list:
            uv = self._reins_unit_views(a)
            s = pd.Series(np.nan, index=row_index)
            for v in views:
                for m, val in zip(measures, moments6(a.xs, uv[v])):
                    s[(v, m)] = val
            cols[a.name] = s
        rdp = self.reins_density_df
        xs = rdp['loss'].to_numpy()
        s = pd.Series(np.nan, index=row_index)
        for v, col in [('gross', 'p_agg_gross'), ('ceded', 'p_agg_ceded'),
                       ('net', 'p_agg_net')]:
            for m, val in zip(measures, moments6(xs, rdp[col].to_numpy())):
                s[(v, m)] = val
        cols['total'] = s
        self._reins_stats_df = pd.DataFrame(cols)
        return self._reins_stats_df

    @property
    def reins_summary_df(self):
        """Portfolio end-to-end reinsurance loss summary.

        One block per unit plus a ``total`` block, concatenated with
        ``unit`` / ... keys (the :attr:`validation_df` assembly pattern). Each
        block is a ``view x component`` table of **mean loss** on the
        eight :attr:`Aggregate.validation_df` columns (``EX | Est EX | Change EX |
        CV | Est CV | Change CV | Sk | Est Sk``) for the unit's own per-stage
        cession (from :meth:`Aggregate.reins_summary_df`); units without
        reinsurance are omitted from their own blocks. The ``total`` block is
        the end-to-end gross / ceded / net portfolio aggregate moments from
        :meth:`reins_stats_df`; the ``EX`` / ``CV`` / ``Sk`` reference is the
        gross end-to-end moment held constant down the three views, ``Est`` is
        the per-view output and ``Change = (output - gross) / gross`` reads as
        the % impact of the programme (0 on the gross row -- no exact pre-bucket
        reference exists for the convolved portfolio marginals).

        Total means equal the sum of the unit end-to-end means per view
        (means add under convolution). Returns ``None`` when no unit cedes.
        """
        if self._reins_after_label() is None:
            return None
        if self._reins_describe is not None:
            return self._reins_describe
        blocks = []
        keys = []
        for a in self:
            rdesc = a.reins_summary_df
            if rdesc is not None:
                blocks.append(rdesc)
                keys.append(a.name)
        # total block: end-to-end gcn, eight columns matching the unit blocks.
        # Reference (EX/CV/Sk) is the gross end-to-end moment, held constant down
        # each view (the economic view of validation_df): Est is the per-view output
        # and Change = (output - gross) / gross. The gross row compares gross to
        # gross, so its Change is 0 (no exact pre-bucket reference exists for the
        # convolved portfolio marginals); the ceded / net rows read as the %
        # impact of the whole reinsurance programme.
        rs = self.reins_stats_df
        gross_mean = float(rs.loc[('gross', 'mean'), 'total'])
        gross_cv = float(rs.loc[('gross', 'cv'), 'total'])
        gross_sk = float(rs.loc[('gross', 'skew'), 'total'])
        rows = []
        idx = []
        for v in ['gross', 'ceded', 'net']:
            mean = float(rs.loc[(v, 'mean'), 'total'])
            cv = float(rs.loc[(v, 'cv'), 'total'])
            sk = float(rs.loc[(v, 'skew'), 'total'])
            rows.append([
                gross_mean, mean, float(_noise_aware_rel_error(mean, gross_mean)),
                gross_cv, cv, float(_noise_aware_rel_error(cv, gross_cv)),
                gross_sk, sk])
            idx.append(('total', v, 'agg'))
        total_block = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx, names=['stage', 'view', 'component']),
            columns=_reinsurance.REINS_DESCRIBE_COLS)
        blocks.append(total_block)
        keys.append('total')
        self._reins_describe = pd.concat(blocks, keys=keys, names=['unit'])
        return self._reins_describe

    @property
    def spec(self):
        """
        Get the dictionary specification.

        :return:
        """
        d = dict()
        d['name'] = self.name
        d['spec_list'] = [a._spec for a in self.agg_list]
        return d

    @property
    def spec_ex(self):
        """
        All relevant info.

        :return:
        """
        return {'type': type(self), 'spec': self.spec, 'bs': self.bs, 'log2': self.log2,
                'aggs': len(self.agg_list)}

    def json(self, stream=None):
        """
        write object as json

        :param    stream:
        :return:  stream or text
        """

        args = dict()
        args["bs"] = self.bs
        args["log2"] = self.log2
        args["padding"] = self.padding
        args["distortion"] = repr(self._distortion)
        args["sev_calc"] = self.sev_calc
        args["remove_fuzz"] = self._remove_fuzz
        args["last_update"] = str(self.last_update)
        args["hash_rep_at_last_update"] = str(self.hash_rep_at_last_update)

        d = self.spec
        d['args'] = args

        logger.debug(f'Portfolio.json| dummping {self.name} to {stream}')
        s = json.dumps(d)  # , default_flow_style=False, indent=4)
        logger.debug(f'Portfolio.json | {s}')
        if stream is None:
            return s
        else:
            return stream.write(s)

    def save(self, filename='', mode='a'):
        """
        persist to json in filename; if none save to user.json

        :param filename:
        :param mode: for file open
        :return:
        """
        if filename == "":
            filename = Path.home() / 'agg/user.json'
            filename.parent.mkdir(parents=True, exist_ok=True)

        with filename.open(mode=mode, encoding='utf-8') as f:
            self.json(stream=f)
            logger.debug(f'Portfolio.save | {self.name} saved to {filename}')

    def __add__(self, other):
        """
        Add two portfolio objects INDEPENDENT sum (down road can look for the same severity...)

        :param other:
        :return:
        """
        assert isinstance(other, Portfolio)
        new_spec = []
        for a in self.agg_list:
            c = deepcopy(a._spec)
            c['name'] = c['name']
            new_spec.append(c)
        for a in other.agg_list:
            c = deepcopy(a._spec)
            c['name'] = c['name']
            new_spec.append(c)

        return Portfolio(f'({self.name}) + ({other.name})', new_spec)

    def __rmul__(self, other):
        """
        new = other * self; treat as scale change

        :param other:
        :return:
        """

        assert other > 0

        new_spec = []
        for a in self.agg_list:
            new_spec.append(deepcopy(a._spec))

        for d in new_spec:
            # d is a dictionary agg spec, need to adjust the severity
            s = d['severity']
            if 'mean' in s:
                s['mean'] *= other
            elif 'scale' in s:
                s['scale'] *= other
            else:
                raise ValueError(f"Cannot adjust {s['name']} for scale")

        return Portfolio(f'{other} x {self.name}', new_spec)

    def __mul__(self, other):
        """
        new = self * other, other integer, sum of other independent copies

        :param other:
        :return:
        """

        assert isinstance(other, int)

        new_spec = []
        for a in self.agg_list:
            new_spec.append(deepcopy(a._spec))

        for d in new_spec:
            # d is a dictionary agg spec, need to adjust the frequency
            # TODO better freq dists; deal with Bernoulli where n=log<1
            d['frequency']['n'] *= other

        return Portfolio(f'Sum of {other} copies of {self.name}', new_spec)

    def snap(self, x):
        """
        snap value x to the index of density_df

        :param x:
        :return:
        """
        ix = self.density_df.index.get_indexer([x], 'nearest')[0]
        return self.density_df.iloc[ix, 0]

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

        if kind == 'middle':
            # logger.warning(f'kind=middle is deprecated, replacing with kind=lower')
            kind = 'lower'

        assert kind in ['lower', 'upper'], 'kind must be lower or upper'

        return self._grid_distribution().q(p, kind)

    def _grid_distribution(self):
        """The :class:`GridDistribution` view over the portfolio ``p_total`` grid.

        Lazily built and cached on first use (invalidated to ``None`` whenever
        the density changes). Built on the **full** contiguous ``bs`` grid
        (zero-mass buckets included): the var/tvar kernel filters to the
        positive-mass subset internally (:meth:`GridDistribution._funcs`), so
        ``q``/``tvar`` are unchanged, while the width-summing ``lev`` /
        ``cdf`` / ``sf`` need the full grid to match the ``exa_total`` /
        ``add_exa`` convention -- dropping the empty low buckets (where
        ``S == 1``) would make ``lev`` undercount. Replaces the old
        ``_var_tvar_function`` dict and the mutating ``_make_var_tvar`` wrapper.
        (Pre-1.0.0a97 this filtered ``p_total > 0`` up front, which silently
        broke ``lev`` on the subset grid; the filter moved inside the kernel.)
        """
        if self._dist is None:
            self._dist = GridDistribution.from_series(
                self.density_df.p_total, bs=self.bs, name=self.name,
                is_loss_value=self._is_loss_value)
        return self._dist

    def cdf(self, x):
        """
        distribution function

        :param x:
        :return:
        """
        if self._cdf is None:
            # Dec 2019: kind='linear' --> kind='previous'
            self._cdf = interpolate.interp1d(self.density_df.loss, self.density_df.F, kind='previous',
                                             bounds_error=False, fill_value='extrapolate')
        return 0. + self._cdf(x)

    def sf(self, x):
        """
        survival function

        :param x:
        :return:
        """
        return 1 - self.cdf(x)

    def pdf(self, x):
        """
        probability density function, assuming a continuous approximation of the bucketed density
        :param x:
        :return:
        """
        if self._pdf is None:
            self._pdf = interpolate.interp1d(self.density_df.loss, self.density_df.p_total, kind='linear',
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

    def var(self, p):
        """
        value at risk = alias for quantile function

        :param p:
        :return:
        """
        return self.q(p)

    def tvar(self, p, kind=''):
        """
        Compute the tail value at risk at threshold p. Revised June 2023.

        Really this function returns ES, CVaR, but in modern terminology
        this is called TVaR.

        Definition 2.6 (Tail mean and Expected Shortfall)
        Assume E[X−] < ∞. Then
        x¯(α) = TM_α(X) = α^{−1}E[X 1{X≤x(α)}] + x(α) (α − P[X ≤ x(α)])
        is α-tail mean at level α the of X.
        Acerbi and Tasche (2002)

        McNeil etc. p66-70 - this follows from def of ES as an integral
        of the quantile function


        :param p:
        :param kind: No longer neeed as the new method is exact (equals the old
        tail) and about 1000x faster.
        :return:
        """

        if kind != '' and getattr(self, 'tvar-warning', 0) == 0:
            logger.warning('kind is no longer used in TVaR, new method equivalent to kind=tail but much faster. '
                           'Argument kind will be removed in the future.')
            setattr(self, 'tvar-warning', 1)

        if kind == 'inverse':
            logger.warning('kind=inverse called...??!!')

        assert self.density_df is not None, 'Must recompute prior to computing tail value at risk.'

        return self._grid_distribution().tvar(p)

    def tvar_threshold(self, p, kind):
        """
        Find the value pt such that TVaR(pt) = VaR(p) using Bisection method.
        Will fail if p=0 because signs are the same.
        """
        return self._grid_distribution().tvar_threshold(p, kind)

    def as_severity(self, limit=np.inf, attachment=0, conditional=False):
        """
        Convert portfolio into a severity without recomputing.

        Throws an error if self not updated.

        :param limit:
        :param attachment:
        :param conditional:
        :return:
        """
        if self.density_df is None:
            raise ValueError('Must update prior to converting to severity')
        return Severity(sev_name=self, sev_a=self.log2, sev_b=self.bs,
                        exp_attachment=attachment, exp_limit=limit, sev_conditional=conditional)

    def approximate(self, approx_type='slognorm', output='scipy'):
        """
        Create an approximation to self using method of moments matching.

        Returns a dictionary specification of the portfolio aggregate_project.
        If updated uses empirical moments, otherwise uses theoretic moments

        :param approx_type: norm | lognorm | gamma | slognorm | sgamma | all
        :param output: return a dict or agg language specification
        :return:

        A shifted family (``slognorm`` / ``sgamma``) requested for a symmetric
        portfolio total degenerates to its normal limit and emits a
        ``UserWarning`` (pass ``approx_type='norm'`` for the normal explicitly).
        A left-skewed total is fitted by reflection, which has no native frozen
        ``scipy`` / one-line DecL form; use ``output='sev_kwargs'`` or the
        default Aggregate object for that case. Mirrors
        :meth:`Aggregate.approximate`.
        """
        emp_mean = self.stats_df.loc[('agg', 'mean'), 'empirical']
        if pd.isna(emp_mean):
            # not updated — use theoretical moments from the total column
            m = float(self.stats_df.loc[('agg', 'mean'), 'total'])
            cv = float(self.stats_df.loc[('agg', 'cv'), 'total'])
            skew = float(self.stats_df.loc[('agg', 'skew'), 'total'])
        else:
            # use empirical (post-FFT) moments matched to the computed aggregate
            m = float(emp_mean)
            cv = float(self.stats_df.loc[('agg', 'cv'), 'empirical'])
            skew = float(self.stats_df.loc[('agg', 'skew'), 'empirical'])
        note = f'frozen version of {self.name}'

        def _one(kind, warn):
            nm = f'{kind[0:4]}.{self.name[0:5]}'
            return approximate_from_mcvsk(m, cv, skew, nm, f'agg {nm} 1 claim sev ',
                                          note, kind, output, warn_degenerate=warn)

        if approx_type == 'all':
            # Survey: quiet about degeneration; skip a family that cannot be
            # represented for this distribution/output (e.g. reflected + scipy).
            out = {}
            for kind in ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']:
                try:
                    out[kind] = _one(kind, warn=False)
                except ValueError:
                    continue
            return out
        return _one(approx_type, warn=True)

    def _single_big_jump_window(self, p_star):
        """Portfolio single-big-jump extent floor by look-through to the units.

        Delegated to :func:`_bucket_window.port_single_big_jump_window`.
        """
        return _bucket_window.port_single_big_jump_window(self, p_star)

    def best_window(self, log2=16, bs_in=0, bucket_sizing_p=BUCKET_SIZING_P):
        """Decide the portfolio combine grid by Portfolio MM + SBJ look-through.

        Delegated to :func:`_bucket_window.port_best_window`; see there for the
        full bulk / resolution / SBJ-floor documentation. Populates
        :attr:`_bs_window_df` and returns ``(bs, log2, x_min)``.
        """
        return _bucket_window.port_best_window(self, log2, bs_in, bucket_sizing_p)

    def _signed(self):
        """Whether any unit has signed (negative-support) severity.

        The portfolio is *signed* iff at least one component aggregate is
        signed -- declared with ``ssev`` (continuous) or a ``dsev`` with a
        negative atom; see :meth:`Aggregate._signed`. This mirrors the
        Aggregate gate so the non-signed path stays byte-for-byte identical:
        every signed-support code path in :meth:`update` (window coarsening,
        the F2 present roll, the ``add_exa`` fallback) is reached **only**
        when this returns ``True``.

        Returns
        -------
        bool
        """
        return any(a._signed() for a in self.agg_list)

    def _build_bs_window_df(self, rows, bs, log2, x_min, cand, resolution, W_ext):
        """Build the unit-indexed bucket/window summary for the combine.

        Delegated to :func:`_bucket_window.port_build_bs_window_df`.
        """
        return _bucket_window.port_build_bs_window_df(
            self, rows, bs, log2, x_min, cand, resolution, W_ext)

    def _bs_window(self, log2, bs_in, bucket_sizing_p=BUCKET_SIZING_P):
        """Decide ``(bs, log2, x_min)`` for the portfolio combine grid.

        Thin forwarder to :meth:`best_window`, delegated to
        :func:`_bucket_window.port_bs_window`.
        """
        return _bucket_window.port_bs_window(self, log2, bs_in, bucket_sizing_p)

    def update(self, log2, bs, remove_fuzz=False,
               sev_calc='discrete', discretization_calc='survival', normalize=True, padding=1,
               trim_density_df=False, add_exa=True, force_severity=True, bucket_sizing_p=BUCKET_SIZING_P,
               debug=False):
        """

        TODO: currently debug doesn't do anything...

        Create density_df, performs convolution. optionally adds additional information if ``add_exa=True``
        for allocation and priority analysis

        num buckets and max loss from bucket size

        Aggregate reinsurance in parser has replaced the aggregate_cession_function (a function of a Portfolio object
        that adjusts individual unit densities; applied after unit aggs created but before creating not-units;
        actual statistics do not reflect impact.) Agg re by unit is now applied in the Aggregate object.

        TODO: consider aggregate covers at the portfolio level...Where in parse - at the top!


        :param log2:
        :param bs: bucket size
        :param remove_fuzz: remove machine noise elements from FFT
        :param sev_calc: how to calculate the severity, discrete (point masses as xs) or continuous (uniform between xs points)
        :param discretization_calc:  survival or distribution (accurate on right or left tails)
        :param normalize: if true, normalize the severity so sum probs = 1. This is generally what you want; but
        :param padding: for fft 1 = double, 2 = quadruple
        :param trim_density_df: remove unnecessary columns from density_df before returning
        :param add_exa: run add_exa to append the objective allocation columns needed for pricing
        :param force_severity: force computation of severities for aggregate components even when approximating
        :param bucket_sizing_p: percentile to use for bucket recommendation.
        :param debug: if True, print debug information
        :return:
        """
        self._valid = None # reset valid flag

        if log2 <= 0:
            raise ValueError('log2 must be >= 0')
        self.log2 = log2
        # Grid sizing routes through ``best_window`` (Portfolio MM bulk + SBJ
        # look-through). ``self._combine_x_min`` records the windowed origin for
        # a *non-signed* total whose mass clears 0 (Plan B): when set, ``update``
        # takes the same roll-combine path as a signed book, placing the shared
        # grid at ``x_min > 0`` instead of forcing the legacy 0-based grid.
        signed = self._signed()
        self._combine_x_min = None
        if signed:
            bs, log2, _x_min_est = self._bs_window(log2, bs, bucket_sizing_p)
            self.log2 = log2
            self.bs = bs
        elif bs == 0:
            # Non-signed auto-size: Portfolio MM bulk + SBJ look-through
            # (best_window), which shrinks log2 to just hold the extent -- a tiny
            # discrete book no longer inflates to the log2 cap. A concentrated
            # total that clears 0 is windowed (x_min > 0, Plan B).
            bs, log2, x_min_est = self.best_window(log2, 0, bucket_sizing_p)
            self.log2 = log2
            self.bs = bs
            if x_min_est > 0:
                self._combine_x_min = float(x_min_est)
            logger.info(f'bs=0 entered, setting bs={bs:.6g}, log2={log2}, '
                        f'x_min={x_min_est:.6g} via best_window (Portfolio MM '
                        f'+ SBJ look-through).')
        else:
            self.bs = bs
        self.padding = padding
        self.sev_calc = sev_calc
        self._remove_fuzz = remove_fuzz
        self.discretization_calc = discretization_calc
        self.normalize = normalize

        if self.hash_rep_at_last_update == hash(self):
            # this doesn't work
            logger.warning(f'Nothing has changed since last update at {self.last_update}')
            return

        self._dist = None
        # density changes invalidate the augmented_df cache
        self._augmented_dfs = {}
        self._last_applied_distortion_name = None
        # invalidate reinsurance reporting caches
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_describe = None

        # Per-unit state for the kappa construction in ``add_exa``: the
        # unit's native grid / pmf plus the padded FT of its pmf. Captured
        # at combine time (it cannot be reconstructed after rebasing) and
        # **transient within this call** — freed before ``update`` returns
        # (meta D6); only scalars (``agg.x_min``) and the native pmfs
        # persist, on the Aggregate objects themselves.
        unit_state = {}

        # Build the grid and the per-unit densities, accumulating their
        # product in Fourier space to get ``p_total``.
        N = 1 << log2
        # The roll-combine path serves both a signed (P&L) book and a windowed
        # non-signed total (Plan B, ``self._combine_x_min`` set by best_window).
        use_roll = signed or self._combine_x_min is not None
        if use_roll:
            # ---- roll combine on a shared (signed or windowed) grid --------
            # Drive each unit on its OWN window [x_min_k, ...) sharing the
            # portfolio bs/log2/padding, so the unit object stays internally
            # correct (no false deficit, right moments, right summary_df/plot --
            # plan 2c). The combine reads each unit's ftagg_density, which is
            # origin-at-0 *regardless* of the unit's x_min (the output roll hits
            # the density, never ftagg), so the units' FFTs still multiply
            # correctly here -- and a heavy unit severity that cannot itself
            # window stays on its own 0-based grid, so the portfolio windowing
            # is not blocked by it (the per-aggregate Regime-B limitation does
            # not bind the combine).
            ft_all = None
            x_mins = []
            for agg in self.agg_list:
                agg.update(log2=log2, bs=self.bs, padding=self.padding,
                           sev_calc=sev_calc,
                           discretization_calc=discretization_calc,
                           normalize=normalize, force_severity=force_severity,
                           x_min='auto', bucket_sizing_p=bucket_sizing_p, debug=debug)
                # de-fuzz the native pmf when requested — same convention
                # as the de-fuzzed density_df the kappa numerator used to
                # read (the first-moment weighting amplifies far-tail dust)
                p_unit = (remove_fuzz_util(agg.agg_density)
                          if self._remove_fuzz else agg.agg_density)
                unit_state[agg.name] = dict(
                    xs=agg.xs, p=p_unit, ft_p=agg.ftagg_density)
                x_mins.append(agg.x_min)
                if ft_all is None:
                    ft_all = np.copy(agg.ftagg_density)
                else:
                    ft_all *= agg.ftagg_density
            # Realised portfolio origin. Signed: the support min of the
            # independent sum is Sigma x_min_k; floored by min_k x_min_k so no
            # per-unit marginal wraps (safety floor on the plan's sum-of-
            # windows). Windowed non-signed (Plan B): the Portfolio MM windowed
            # origin from best_window (each unit sits at its own 0-origin; the
            # *total* mass is what clears 0). Snapped to bs.
            if signed:
                x_min_tot = float(min(sum(x_mins), min(x_mins)))
            else:
                x_min_tot = float(self._combine_x_min)
            x_min_tot = float(np.round(x_min_tot / self.bs) * self.bs)
            j0_tot = int(round(x_min_tot / self.bs))
            xs = x_min_tot + np.arange(N, dtype=float) * self.bs
            self.density_df = pd.DataFrame(index=xs)
            self.density_df['loss'] = xs
            # F2 present step: full-length irfft then a single roll placing
            # x_min_tot at output index 0, keep N. ``ift(., 0)`` returns the
            # whole length-M = N<<padding buffer (origin-at-0, negatives
            # wrapped to the top); the truncating ``ift(., padding)`` used on
            # the non-signed path would silently drop that wrapped tail.
            self.density_df['p_total'] = np.roll(ift(ft_all, 0), -j0_tot)[:N]
            self._signed_window = (x_min_tot, x_min_tot + N * self.bs)
        else:
            # ---- non-signed path (combine unchanged, byte-for-byte) -------
            # Use self.bs (resolved above): build_many now passes bs through
            # as 0 => auto, so the grid must read the resolved bucket, not the
            # raw parameter.
            MAXL = N * self.bs
            xs = np.linspace(0, MAXL, N, endpoint=False)
            self.density_df = pd.DataFrame(index=xs)
            self.density_df['loss'] = xs
            ft_all = None
            for agg in self.agg_list:
                agg.update_work(xs, self.padding, sev_calc, discretization_calc,
                                normalize, force_severity, debug=debug)
                # de-fuzz as for the signed path above (legacy parity: the
                # kappa numerator read the de-fuzzed p_{unit} columns)
                p_unit = (remove_fuzz_util(agg.agg_density)
                          if self._remove_fuzz else agg.agg_density)
                unit_state[agg.name] = dict(
                    xs=xs, p=p_unit, ft_p=agg.ftagg_density)
                if ft_all is None:
                    ft_all = np.copy(agg.ftagg_density)
                else:
                    ft_all *= agg.ftagg_density
            self.density_df['p_total'] = np.real(ift(ft_all, self.padding))

        self.remove_fuzz(log='update')

        # Objective allocation columns (kappa + direct sums) — now valid on
        # signed (P&L) windows too (the shifted-support kappa carries unit
        # origins; share-based columns are blanked per steering 6).
        if add_exa:
            self.add_exa(self.density_df, unit_state)
        else:
            # at least want F and S to get quantile functions
            self.density_df['F'] = np.cumsum(self.density_df.p_total)
            self.density_df['S'] = 1 - self.density_df.F
        # D6: the padded FT state is transient — drop it before returning.
        del unit_state, ft_all

        # Mass-conservation check on the rolled grid (signed or windowed
        # non-signed): a window narrower than the combined support shows up as a
        # deficit (the wrapped tail would be truncated by the final [:N]).
        # Surface it loudly, like the Aggregate path, and record it as the
        # portfolio far-tail clip for the bs report.
        if use_roll:
            deficit = 1.0 - float(np.sum(self.density_df['p_total']))
            if deficit > VALIDATION_NOISE:
                top = float(x_min_tot + N * self.bs)
                self._bs_clip = dict(
                    reach=float('nan'), grid_top=top, log2=int(log2),
                    bs=float(self.bs), need_log2=int(log2 + 1),
                    clipped_mass=float(deficit))
                if self._bs_window_df is not None \
                        and 'used' in self._bs_window_df.index:
                    self._bs_window_df.loc['used', 'clipped'] = float(deficit)
                kind = 'signed' if signed else 'windowed'
                warnings.warn(
                    f'{self.name}: portfolio PMF deficit {deficit:.3e} '
                    f'(Σp = 1 − {deficit:.3e} < 1); the {kind} window is '
                    f'narrower than the combined support -- raise log2 or '
                    f'widen the grid.', DefectiveDistributionWarning,
                    stacklevel=2)

        # Empirical portfolio-total agg moments from the FFT output, via
        # ``xsden_to_mwrangler`` on a de-fuzzed copy -- mirrors
        # ``Aggregate.update_work`` so both modules share one moment
        # convention (meta.3 D4/D8). The de-fuzz zeroes |p| < eps before the
        # ``x**k`` weighting, otherwise far-tail fp noise gets amplified to
        # spurious skew on wide grids.
        _xs = self.density_df['loss'].values
        _p = self.density_df['p_total'].values
        _p_clean = remove_fuzz_util(_p)
        _mw = xsden_to_mwrangler(_xs, _p_clean)
        _ex1, _ex2, _ex3 = _mw.noncentral
        self.est_m, self.est_cv, self.est_skew = _mw.mcvsk
        self.ex = self.est_m
        self.est_sd = self.est_m * self.est_cv
        self.est_var = self.est_sd ** 2
        self._write_empirical_stats(_ex1, _ex2, _ex3)

        self.last_update = np.datetime64('now')
        self.hash_rep_at_last_update = hash(self)
        if trim_density_df:
            self.trim_density_df()
        # invalidate stored functions
        self._dist = None
        self._cdf = None

    def _build_stats_df(self, ma, max_limit):
        """Construct ``stats_df`` from the ``MomentAggregator`` after init.

        Adapted from ``Aggregate._init_stats_df`` + post-loop totals
        block. Per-unit columns hold each ``Aggregate.stats_df['mixed']``
        (the unit's view of its own theoretical moments); the ``total``
        column holds portfolio totals from the ``MomentAggregator``
        (``remix=False`` — the running totals across units).

        Portfolio's column is ``total`` rather than Aggregate's ``mixed``
        because there is no portfolio-level mixed-vs-independent analog
        (mixed-vs-independent is an Aggregate-only concept that strips
        a single agg's freq mixing distribution). Columns are therefore:
        per-unit + ``total`` + ``empirical`` + ``error``.

        ``empirical`` and ``error`` are left as NaN here; ``update``
        populates them after the FFT.
        """
        cols = list(self.unit_names) + [
            'total', 'after_occ', 'empirical',
            'occ_impact', 'agg_impact', 'gross_empirical', 'error',
        ]
        self.stats_df = pd.DataFrame(
            np.nan, index=_PORT_STATS_ROW_INDEX, columns=cols, dtype=float,
        )

        # Per-unit columns: copy each agg's ``stats_df['mixed']`` into the
        # matching unit column. (Both frames are now all-float; the legacy
        # ``meta/name`` row that was excluded by the ``in self.stats_df.index``
        # guard is gone, but the guard is harmless.)
        for a in self.agg_list:
            a_mixed = a.stats_df['mixed']
            for idx, val in a_mixed.items():
                if idx in self.stats_df.index:
                    self.stats_df.loc[idx, a.name] = val

        # Portfolio totals: ``total`` = running totals across units
        # (``remix=False``, preserves each agg's freq mixing).
        unit_cols = list(self.unit_names)
        _flat_names = MomentAggregator.column_names()

        def _collect(meta_key):
            return self.stats_df.loc[('meta', meta_key), unit_cols]

        tot_el = float(_collect('el').sum())
        tot_prem = float(_collect('prem').sum())
        tot_lr = (tot_el / tot_prem) if tot_prem else np.nan
        # Attachment: portfolio total only makes sense if all units
        # share the same attachment (commonly 0); else NaN. Limit at
        # portfolio level is the max across units (legacy convention).
        _attaches = _collect('attachment').values
        if len(_attaches) and np.all(_attaches == _attaches[0]):
            tot_attach = float(_attaches[0])
        else:
            tot_attach = np.nan

        stats = ma.get_fsa_stats(total=True, remix=False)
        for flat, val in zip(_flat_names, stats):
            self.stats_df.loc[_flat_col_to_stats_index(flat), 'total'] = val
        self.stats_df.loc[('meta', 'limit'), 'total'] = max_limit
        self.stats_df.loc[('meta', 'attachment'), 'total'] = tot_attach
        self.stats_df.loc[('meta', 'el'), 'total'] = tot_el
        self.stats_df.loc[('meta', 'prem'), 'total'] = tot_prem
        self.stats_df.loc[('meta', 'lr'), 'total'] = tot_lr

    def _write_empirical_stats(self, agg_ex1, agg_ex2, agg_ex3):
        """Populate the ``empirical`` and ``error`` columns of ``stats_df``.

        Called from ``update`` (and ``create_from_sample`` after the
        switcheroo) once the FFT has run. Three sources feed it:

        - ``meta`` rows: limit / attachment / prem / el / lr have no
          empirical analog at the portfolio level — they are factual
          (limit, attach) or sums of expected values (prem, el, lr).
          Copy the ``total`` values across with implied ``error = 0``.
        - ``sev`` rows: combine each unit's empirical sev moments
          (``a.stats_df['empirical']``) with the unit's theoretical
          frequency via a fresh ``MomentAggregator``. Yields portfolio-
          level empirical sev mean / cv / skew + raw moments.
        - ``agg`` rows: ``ex1`` / ``ex2`` / ``ex3`` are the raw moments
          ``agg_ex1`` / ``agg_ex2`` / ``agg_ex3`` already computed from
          the portfolio-total FFT density; ``mean`` / ``cv`` / ``skew``
          are ``est_m`` / ``est_cv`` / ``est_skew``.

        ``('freq', *)`` empirical rows stay NaN — frequency is exact (no
        FFT applies to it) and the Aggregate convention is the same.
        """
        # meta: copy from total, error = 0 implicit.
        for meta_key in ('limit', 'attachment', 'el', 'prem', 'lr'):
            self.stats_df.loc[('meta', meta_key), 'empirical'] = (
                self.stats_df.loc[('meta', meta_key), 'total']
            )

        # sev: re-aggregate per-unit (theoretical freq, empirical sev)
        # through a fresh MomentAggregator. Aggregate stores empirical sev
        # raw moments directly (``a.stats_df['empirical']`` rows
        # ``('sev','ex1'..'ex3')``), so we read them straight in -- no
        # (mean, cv, skew) → (ex1, ex2, ex3) inversion needed.
        # ``get_fsa_stats`` returns 18 entries: freq f1/f2/f3/m/cv/sk, sev
        # s1/s2/s3/m/cv/sk, agg a1/a2/a3/m/cv/sk; we use the sev block
        # (indices 6..11).
        ma_emp = MomentAggregator()
        for a in self.agg_list:
            mixed = a.stats_df['mixed']
            emp = a.stats_df['empirical']
            ma_emp.add_fs(
                float(mixed[('freq', 'ex1')]),
                float(mixed[('freq', 'ex2')]),
                float(mixed[('freq', 'ex3')]),
                float(emp[('sev', 'ex1')]),
                float(emp[('sev', 'ex2')]),
                float(emp[('sev', 'ex3')]),
            )
        emp_stats = ma_emp.get_fsa_stats(total=True, remix=False)
        sev_block = [('sev', 'ex1'), ('sev', 'ex2'), ('sev', 'ex3'),
                     ('sev', 'mean'), ('sev', 'cv'), ('sev', 'skew')]
        for idx, val in zip(sev_block, emp_stats[6:12]):
            self.stats_df.loc[idx, 'empirical'] = val

        # agg: raw moments + mean/cv/skew from the FFT density.
        self.stats_df.loc[('agg', 'ex1'),  'empirical'] = agg_ex1
        self.stats_df.loc[('agg', 'ex2'),  'empirical'] = agg_ex2
        self.stats_df.loc[('agg', 'ex3'),  'empirical'] = agg_ex3
        self.stats_df.loc[('agg', 'mean'), 'empirical'] = self.est_m
        self.stats_df.loc[('agg', 'cv'),   'empirical'] = self.est_cv
        self.stats_df.loc[('agg', 'skew'), 'empirical'] = self.est_skew

        # error: noise-aware diff vs the ``total`` column — relative error,
        # falling back to absolute where the theoretical value is ~0 (e.g.
        # the skew of a symmetric unit). For meta rows where empirical ==
        # total the result is 0 (or NaN where total is NaN).
        self.stats_df['error'] = _noise_aware_rel_error(
            self.stats_df['empirical'], self.stats_df['total'])

    @property
    def valid(self):
        """
        Check if the model appears valid. See documentation for Aggregate.valid.

        An answer of True does not guarantee the model is valid, but
        False means it is definitely suspect. (Similar to the null hypothesis in a statistical test).
        Called and reported automatically by qd for Aggregate objects.

        Checks the relative errors (from ``stats_df['error']``) for:

        * severity mean < eps
        * severity cv < 10 * eps
        * severity skew < 100 * eps (skewness is more difficult to estimate)
        * aggregate mean < eps and < 2 * severity mean relative error (larger values
          indicate possibility of aliasing and that ``bs`` is too small).
        * aggregate cv < 10 * eps
        * aggregate skew < 100 * esp

        eps = 1e-3 by default; change in ``validation_eps`` attribute.

        The CV and skew tests are applied only when the theoretical value is
        finite and its magnitude exceeds ``VALIDATION_NOISE`` -- a
        theoretically-zero skew (symmetric total) or CV is skipped, because
        the FFT's empirical estimate of a zero higher moment is grid-
        dependent noise with no meaningful relative error. When the test
        applies, ``np.isclose`` with relative tolerance ``10*eps`` (CV) /
        ``100*eps`` (skew) measures agreement.

        :return: True if all tests are passed, else False.

        """
        return _validation.valid_portfolio(self)

    @property
    def validation_explanation(self):
        """
        Long-narrative explanation of the validation result (str).

        The consistent narrative surface, mirroring ``tail_explanation`` /
        ``bs_explanation``.
        """
        return _validation.validation_explanation(self)

    def trim_density_df(self):
        """
        Trim out unwanted columns from density_df

        :return:
        """
        self.density_df = self.density_df.drop(
            self.density_df.filter(regex='^e_|^exi_xlea|^[a-z_]+ημ').columns,
            axis=1
        )

    @property
    def pprogram(self):
        """Canonical DecL program text, rendered from the parsed spec.

        Re-parses :attr:`program` and renders through
        :func:`aggregate.decl_writer.format_program` (the inverse of the
        parser). Rendered in the default ``spread`` layout: a ``port`` head line,
        each unit two-space-indented, and each unit's clauses one level deeper.
        Canonical rather than verbatim. Call ``format_program(self.program,
        layout='terse')`` for the historical single-line-per-unit form. A
        portfolio built programmatically (empty ``program``) returns ``''``.
        """
        return format_program(self.program, fmt='text')

    @property
    def pprogram_html(self):
        """Syntax-highlighted DecL program for IPython / Jupyter display."""
        return format_program(self.program, fmt='html')

    def _limits(self, stat='range', kind='linear', zero_mass='include'):
        """
        Suggest sensible plotting limits for kind=range, density, .. (same as Aggregate).

        Should optionally return a locator for plots?

        Called by ploting routines. Single point of failure!

        Must work without ``q`` function when not yet computed.

        :param stat:  range or density or logy (for log density/survival function...ensure consistency)
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

        # if not computed have no business asking for limits
        assert self.density_df is not None

        if stat == 'range':
            p = 0.999 if kind == 'linear' else 1 - 1e-10
            hi = self.q(p)
            # Window-aware x-limits keyed on the grid origin
            # (``density_df.index[0]``); mirrors the Aggregate ``_limits``:
            #  * origin < 0  -- signed (P&L) book: two-sided quantile range
            #    (``f(hi)`` would clip the negative tail or reverse when hi < 0);
            #  * origin > 0  -- thin-tailed output window: anchor the left edge
            #    at the realised support minimum, not 0;
            #  * origin == 0 -- ordinary non-negative book: unchanged.
            if self.density_df.index[0] < 0:
                lo = self.q(1 - p)
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            if self.density_df.index[0] > 0:
                lo = float(self.density['loss'].min())
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            return f(hi)
        elif stat == 'density':
            # total + native unit pmfs (numerics-1)
            pmfs = [self.density_df.p_total] + \
                   [self.unit_density(unit) for unit in self.unit_names]
            mx = max(float(s.max()) for s in pmfs)
            mxx0 = max(float(s.iloc[1:].max()) for s in pmfs)
            if kind == 'linear':
                if zero_mass == 'include':
                    return f(mx)
                else:
                    return f(mxx0)
            else:
                return [eps, mx * 1.5]
        elif stat == 'logy':
            pmfs = [self.density_df.p_total] + \
                   [self.unit_density(unit) for unit in self.unit_names]
            mx = min(1, max(float(s.max()) for s in pmfs))
            return [1e-12, mx * 2]
        else:
            # if you fall through to here, wrong args
            raise ValueError('Inadmissible stat/kind passsed, expected range/density and log/linear.')

    @property
    def density(self):
        """The "live" part of :attr:`density_df` — rows with positive total mass.

        Returns ``density_df.query('p_total > 0')``: the actual support of the
        portfolio, dropping the leading and trailing zero-probability buckets of
        the FFT grid. This is usually what you want to *see*. It is recomputed on
        each access (a plain property, not cached) because ``density_df`` can be
        reassigned by ``update`` or by sampling.
        """
        return self.density_df.query('p_total > 0')

    # ================================================================
    # Unit (native-grid) density accessors -- numerics-1.
    # Unit pmfs live on each Aggregate's own grid; these are the only
    # ways to read them from the Portfolio. The legacy
    # ``density_df['p_{unit}']`` columns were removed at numerics-2
    # (dev/done/plan-numerics-2-objective.md).
    # ================================================================

    def unit_density(self, unit, view='agg'):
        """Native-grid pmf of one unit, read from the owning :class:`Aggregate`.

        The unit's pmf lives on the unit's **own** loss grid (which on a
        windowed or signed book differs from the portfolio total grid).
        This accessor is the supported source for per-unit densities; the
        legacy ``density_df['p_{unit}']`` total-grid columns were removed
        at numerics-2 (use :meth:`aligned_unit_density_df` for an
        explicitly-labelled total-grid display view).

        Parameters
        ----------
        unit : str
            Unit name; one of :attr:`unit_names`.
        view : {'agg', 'sev'}
            ``'agg'`` reads the unit's aggregate pmf
            (``Aggregate.density_df.p_total`` on grid ``xs``); ``'sev'``
            reads the discretized severity
            (``Aggregate.sev_density_df.p_sev`` on grid ``xs_sev``).

        Returns
        -------
        pandas.Series
            pmf named ``p_{unit}``, indexed by the unit's native loss grid.
        """
        if unit not in self.unit_names:
            raise KeyError(
                f'unknown unit {unit!r}; expected one of {self.unit_names}')
        agg = self[unit]
        if view == 'agg':
            ser = agg.density_df['p_total'].copy()
        elif view == 'sev':
            ser = agg.sev_density_df['p_sev'].copy()
        else:
            raise ValueError(f"view must be 'agg' or 'sev', not {view!r}")
        ser.name = f'p_{agg.name}'
        return ser

    def unit_density_df(self, view='agg'):
        """Long-form frame of all unit pmfs, each on its native grid.

        One block per unit, concatenated with a ``(unit, loss)``
        MultiIndex. Each block carries the unit's pmf plus the
        window-audit metadata needed to reason about grid alignment.

        Parameters
        ----------
        view : {'agg', 'sev'}
            As in :meth:`unit_density`.

        Returns
        -------
        pandas.DataFrame
            Index ``(unit, loss)``; columns ``unit, loss, p, F, S, bs,
            x_min, x_max, mass`` where ``x_min`` / ``x_max`` are the unit's
            grid edges and ``mass`` is the unit's represented probability
            ``p.sum()`` (less than 1 on a too-narrow window).
        """
        blocks = []
        for agg in self.agg_list:
            if view == 'agg':
                src = agg.density_df
                df = src[['loss', 'p_total', 'F', 'S']].rename(
                    columns={'p_total': 'p'}).copy()
            elif view == 'sev':
                src = agg.sev_density_df
                df = src[['loss', 'p_sev', 'F_sev', 'S_sev']].rename(
                    columns={'p_sev': 'p', 'F_sev': 'F', 'S_sev': 'S'}).copy()
            else:
                raise ValueError(f"view must be 'agg' or 'sev', not {view!r}")
            df['unit'] = agg.name
            df['bs'] = agg.bs
            df['x_min'] = float(df['loss'].iloc[0])
            df['x_max'] = float(df['loss'].iloc[-1])
            df['mass'] = float(df['p'].sum())
            blocks.append(df[['unit', 'loss', 'p', 'F', 'S', 'bs',
                              'x_min', 'x_max', 'mass']])
        out = pd.concat(blocks)
        out = out.set_index(['unit', 'loss'], drop=False)
        return out

    def aligned_unit_density_df(self, grid='total', *,
                                allow_window_mismatch=False):
        """Unit pmfs scattered onto a common grid — a **display adapter**.

        Reindexes each unit's native pmf onto the requested grid. The
        result is a labelled presentation artifact, never compute input:
        on a windowed book a unit's grid can extend beyond the requested
        grid, and the off-grid buckets are silently dropped (the method
        warns; see below).

        Alignment is by bucket number ``round(loss / bs)`` — the Portfolio
        dictates a common ``bs`` to its units, asserted here — so
        physically identical points always land together regardless of how
        each grid's floats were built.

        Parameters
        ----------
        grid : {'total', 'union', 'zero'}
            ``'total'`` — the portfolio ``density_df`` grid (for a legacy
            zero-origin book this reproduces the ``p_{unit}`` columns
            exactly); ``'union'`` — the union of the unit grids (never
            drops a bucket); ``'zero'`` — the bucket lattice from
            ``min(0, lowest unit x_min)`` through the highest unit
            ``x_max`` (always contains the origin).
        allow_window_mismatch : bool, keyword only
            When unit buckets fall outside the requested grid the method
            warns unless this is ``True`` (acknowledging the view is a
            window-clipped artifact).

        Returns
        -------
        pandas.DataFrame
            Indexed by the requested grid's loss values, one ``p_{unit}``
            column per unit; zero where a unit has no bucket.
        """
        bs = self.bs
        for agg in self.agg_list:
            if not np.isclose(agg.bs, bs, rtol=1e-12, atol=0):
                raise ValueError(
                    f'unit {agg.name!r} has bs={agg.bs} != portfolio '
                    f'bs={bs}; it was re-updated off the portfolio grid')
        # per-unit pmfs on the integer bucket lattice
        unit_j = {}
        unit_p = {}
        for agg in self.agg_list:
            ser = agg.density_df['p_total']
            unit_j[agg.name] = np.round(
                ser.index.to_numpy() / bs).astype(np.int64)
            unit_p[agg.name] = ser.to_numpy()
        if grid == 'total':
            target_x = self.density_df.index.to_numpy()
            target_j = np.round(target_x / bs).astype(np.int64)
        elif grid == 'union':
            target_j = None
            for j in unit_j.values():
                target_j = j if target_j is None else np.union1d(target_j, j)
            target_x = target_j * bs
        elif grid == 'zero':
            j_lo = min(0, min(int(j[0]) for j in unit_j.values()))
            j_hi = max(int(j[-1]) for j in unit_j.values())
            target_j = np.arange(j_lo, j_hi + 1, dtype=np.int64)
            target_x = target_j * bs
        else:
            raise ValueError(
                f"grid must be 'total', 'union' or 'zero', not {grid!r}")
        out = pd.DataFrame(index=pd.Index(target_x, name='loss'))
        mismatched = []
        for agg in self.agg_list:
            j, p = unit_j[agg.name], unit_p[agg.name]
            inside = np.isin(j, target_j)
            if not inside.all():
                dropped = float(np.abs(p[~inside]).sum())
                mismatched.append(
                    f'{agg.name} ({int(np.sum(~inside))} buckets, '
                    f'|p| {dropped:.3e} dropped)')
            col = np.zeros(len(target_j))
            col[np.searchsorted(target_j, j[inside])] = p[inside]
            out[f'p_{agg.name}'] = col
        if mismatched and not allow_window_mismatch:
            warnings.warn(
                f'{self.name}: unit window(s) extend beyond the {grid!r} '
                f'grid -- {"; ".join(mismatched)}. This view is a clipped '
                f'display artifact; pass allow_window_mismatch=True to '
                f'acknowledge.', stacklevel=2)
        return out

    def plot(self, axd=None, figsize=(2 * FIG_W, FIG_H)):
        """
        Defualt plot of density, survival functions (linear and log)

        :param axd: dictionary with plots A and B for density and log density
        :param figsize: figure size used by ``plt.subplot_mosaic`` if ``axd`` is not provided
        :return:
        """
        from .plots import plot_portfolio
        return plot_portfolio(self, axd=axd, figsize=figsize)

    def scatter(self, marker='.', s=5, alpha=1, figsize=(10, 10), diagonal='kde', **kwargs):
        """
        Create a scatter plot of marginals against one another, using pandas.plotting scatter_matrix.

        Designed for use with samples. Plots exeqa columns


        """
        from .plots import plot_scatter
        return plot_scatter(self, marker=marker, s=s, alpha=alpha,
                            figsize=figsize, diagonal=diagonal, **kwargs)

    def add_exa(self, df, unit_state):
        r"""Add the objective (conditional-expectation) allocation columns to ``df``.

        Thin wrapper over :func:`aggregate._portfolio_density.add_exa` -- the
        independent-sum ``exeqa_*`` / ``exa_*`` kernel (the density subsystem).
        Extends ``df`` in place and returns it. See the free function for the
        full column list and the shifted-support kappa notes.
        """
        return _density.add_exa(self, df, unit_state)

    def calibrate_distortions(self, coc, *, p=None, a=None, kind='lower',
                              names=_pricing.DEFAULT_CALIBRATION_DISTORTIONS):
        """
        Calibrate the standard pricing distortion set to a cost-of-capital target.

        Parameters
        ----------
        coc : float
            Target cost of capital ``COC = (P - L) / Q``.
        p : float, optional
            Probability at which the calibration applies; converted to asset
            level via ``self.q(p, kind)``. Exactly one of ``p`` or ``a`` must
            be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.
        kind : {'lower', 'upper'}, optional
            VaR kind when ``p`` is provided. Default ``'lower'``.

        Returns
        -------
        pandas.DataFrame
            The per-distortion calibration receipt, ``distortion_df`` (also
            stored on ``self.distortion_df``): one row per distortion in
            ``[ccoc, ph, wang, dual, tvar]``, index named ``distortion`` (an
            ordered categorical, canonical sort), columns
            ``[param_name, param, error, gini_p, area]``. ``param`` is the raw
            shape (``param_name`` says what it is per family); ``error`` is the
            premium miss; ``gini_p`` is the comparable normalised shape
            ``= 2∫g−1 = p_equiv`` (TVaR-equivalent level); ``area = (gini_p+1)/2
            = ∫g``.

        Notes
        -----
        The shared calibration *target* — identical across all five rows — is
        not repeated here; it is stored once on ``self.calibration_df`` as a
        one-row frame: the inputs ``coc, p, F(a)`` lead, then the canonical
        pentagon octet ``L, M, P, Q, a, LR, PQ, ROE``. ``ROE`` there equals
        ``coc`` (a free self-check). The calibrated distortion objects are on
        ``self.distortions`` keyed by name.

        Calibration is one-point (one ``coc`` at one ``p``/``a``). This replaces
        both the legacy batch
        ``calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, ...)`` and
        ``calibrate_distortions2(coc, reg_p)``.

        ``names`` selects the distortion families to calibrate (default the
        standard set).
        """
        return _pricing.calibrate_distortions(self, coc, p=p, a=a, kind=kind,
                                              names=names)

    def apply_distortion(self, distortion, *, view='ask', S_calculation='forwards',
                         allocation='lifted', allow_deficit=False):
        """
        Apply ``distortion`` and return the resulting augmented_df.

        Results are cached on ``self._augmented_dfs`` keyed by
        ``(name, view, role, S_calculation, allocation)`` so bid/ask,
        loss/payoff, forwards/backwards and linear/lifted frames coexist
        (the ``role`` slot is the canonical value-type flag
        ``_is_loss_value``, never the configurable label string). A second
        call with the same key is an O(1) dict lookup; the returned
        DataFrame is the same object (``is``-identical) as the prior call.

        Parameters
        ----------
        distortion : Distortion or str
            A ``Distortion`` instance, or the name of a previously calibrated
            distortion (looked up in ``self.distortions``).
        view : {'ask', 'bid'}
            Pricing view. Composes with the portfolio's value-type role by
            XOR to select ``g`` or ``g_dual``; see
            :meth:`~aggregate.spectral.Distortion.effective_g`.
        S_calculation : {'forwards', 'backwards'}
            Deficit-parking direction for the exact-discrete tail; see
            :func:`~aggregate.spectral.choquet_weights`. Equivalent on a
            clean (normalized) law.
        allocation : {'lifted', 'linear'}
            Tail-share choice for the per-unit ``exag_*`` columns: lifted
            uses the distorted tail share ``exi_xgtag_*`` (beta), linear
            the objective ``exi_xgta_*`` (alpha). Identical column schema;
            the total columns do not depend on the choice.
        allow_deficit : bool
            Explicit truncation policy for a materially defective total
            (``1 - sum(p_total)`` above the validation noise floor).
            Default False raises ``DefectiveDistributionError``.

        Returns
        -------
        pandas.DataFrame
            The cached ``augmented_df`` for this key.

        Notes
        -----
        The actual construction lives in ``_build_augmented``. The cache is
        invalidated whenever ``update`` is called (the underlying density
        changes).
        """
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        name = distortion.name
        key = (name, view, self._is_loss_value, S_calculation, allocation)
        if key not in self._augmented_dfs:
            self._augmented_dfs[key] = self._build_augmented(
                distortion, view=view, S_calculation=S_calculation,
                allocation=allocation, allow_deficit=allow_deficit)
        self._distortion = distortion
        self._last_applied_distortion_name = name
        return self._augmented_dfs[key]

    def augmented_df(self, distortion):
        """
        Return the cached augmented_df for ``distortion`` (building it on demand).

        Identical to ``apply_distortion(distortion)`` with default kwargs --
        provided as the clean read-side accessor.
        """
        return self.apply_distortion(distortion)

    @property
    def augmented_dfs(self):
        """
        The augmented_df cache as a dict keyed by
        ``(distortion_name, view, role, S_calculation, allocation)``.

        Read-only view -- mutate via ``apply_distortion`` (insert) or
        ``update`` (clear).
        """
        return self._augmented_dfs

    def pricing_at(self, distortion, *, p=None, a=None, allocation='lifted'):
        """Pentagon pricing readout per unit at probability ``p`` or asset ``a``.

        Warms the augmented_df cache for ``distortion`` and pulls the
        ``L M P Q a | LR PQ ROE`` row at the requested asset level.

        Parameters
        ----------
        distortion : Distortion or str
            Passed through to ``apply_distortion``.
        p : float, optional
            Probability; converted to asset level via ``self.q(p)``. Exactly
            one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.
        allocation : {'lifted', 'linear'}
            Tail-share choice for the per-unit premium allocation; passed
            through to :meth:`apply_distortion`.

        Returns
        -------
        pandas.DataFrame
            Rows indexed by unit (units + 'total'), columns
            ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']``. Per-unit
            ``a = P + Q`` (allocated assets); on ``total`` it equals the
            requested portfolio asset level.

        Notes
        -----
        ``L = exa``, ``P = exag``, ``M = P - L``; per-unit capital ``Q`` is
        computed on demand by the layer-ROE construction
        (:meth:`_unit_capital_at` -- no persistent per-unit ``Q`` column).
        Total ``Q`` uses the exact row identity ``a - exag_total``.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'pricing_at requires exactly one of p= (probability) '
                'or a= (asset level).')
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        if a is None:
            a = self.q(p)
        else:
            a = self.snap(a)
        aug = self.apply_distortion(distortion, allocation=allocation)
        if a in aug.index:
            row = aug.loc[a]
        else:
            logger.warning(
                f'pricing_at: asset level {a} not in augmented_df.index; using last row.')
            row = aug.iloc[-1]
            a = float(row['loss'])
        units = list(self.unit_names_ex)
        out = pd.DataFrame(
            index=units,
            columns=['L', 'M', 'P', 'Q'],
            dtype=float,
        )
        out.index.name = 'unit'
        unit_q = self._unit_capital_at(aug, a, distortion)
        for unit in self.unit_names:
            out.loc[unit, 'L'] = row[f'exa_{unit}']
            out.loc[unit, 'P'] = row[f'exag_{unit}']
            out.loc[unit, 'Q'] = unit_q[unit]
        out.loc['total', 'L'] = row['exa_total']
        out.loc['total', 'P'] = row['exag_total']
        out['M'] = out.P - out.L
        # exact total Q = a - exag_total beats the layer-by-layer sum,
        # which can drift by a few buckets in the tail.
        out.loc['total', 'Q'] = a - row['exag_total']
        # fill a + ratios and stamp the canonical categorical (pentagon.py)
        out = complete_pentagon(out)
        return out

    def pentagon_at(self, distortion, *, p=None, a=None, unit='total',
                    allocation='lifted'):
        """Single-unit pentagon as a :class:`~aggregate.pentagon.Pentagon` object.

        The object-flavored analogue of :meth:`pricing_at`: returns one fully
        solved :class:`Pentagon` (an eight-vector with named attributes and
        provenance) for ``unit`` at probability ``p`` or asset level ``a``,
        rather than a DataFrame of all units. The natural entry point for the
        "complete a partial input" workflow — the returned object carries the
        accounting identities and can be re-solved.

        Parameters
        ----------
        distortion : Distortion or str
            Passed through to :meth:`apply_distortion`.
        p : float, optional
            Probability; converted to asset level via ``self.q(p)``. Exactly
            one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index.
        unit : str, default 'total'
            Which unit to read (``'total'`` for the portfolio total).
        allocation : {'lifted', 'linear'}
            Tail-share choice for the per-unit premium allocation; passed
            through to :meth:`apply_distortion`.

        Returns
        -------
        Pentagon
            Fully solved, with ``.distortion`` / ``.shape`` provenance attached.

        Notes
        -----
        Reads the same augmented-distortion row as :meth:`pricing_at`; the
        total ``Q`` uses the exact ``a - exag_total`` (matching ``pricing_at``),
        per-unit ``Q`` the on-demand layer-ROE construction
        (:meth:`_unit_capital_at`), so the two agree.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'pentagon_at requires exactly one of p= (probability) '
                'or a= (asset level).')
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        if a is None:
            a = self.q(p)
        else:
            a = self.snap(a)
        aug = self.apply_distortion(distortion, allocation=allocation)
        if a in aug.index:
            row = aug.loc[a]
        else:
            row = aug.iloc[-1]
            a = float(row['loss'])
        peg = Pentagon(obj=self)
        L = row[f'exa_{unit}']
        P = row[f'exag_{unit}']
        if unit == 'total':
            # exact total Q, matching pricing_at
            Q = a - row['exag_total']
        else:
            Q = self._unit_capital_at(aug, a, distortion, units=[unit])[unit]
        # L, P, Q are the three independent amounts; M = P - L, a = P + Q follow.
        peg.solve(L=L, P=P, Q=Q)
        peg.distortion = distortion
        peg.shape = getattr(distortion, 'shape', None)
        return peg

    def _build_augmented(self, dist, *, view='ask', S_calculation='forwards',
                         allocation='lifted', allow_deficit=False):
        r"""Construct an augmented_df from ``self.density_df`` under ``dist``.

        Thin wrapper over :func:`aggregate._portfolio_common.build_augmented`
        (the common exeqa numerics -- the apply-distortion / allocation engine,
        agnostic to FFT-vs-sample origin). ``apply_distortion`` writes the
        returned frame into ``self._augmented_dfs``.
        """
        return _common.build_augmented(
            self, dist, view=view, S_calculation=S_calculation,
            allocation=allocation, allow_deficit=allow_deficit)

    def _unit_capital_at(self, aug, a, dist, *, view='ask', units=None):
        r"""Per-unit allocated capital ``Q_i(a)`` by the layer-ROE construction.

        Thin wrapper over :func:`aggregate._portfolio_common.unit_capital_at`.
        """
        return _common.unit_capital_at(self, aug, a, dist, view=view, units=units)

    def allocation_diagnostics(self, distortion, *, surface='lifted',
                               view='ask', S_calculation='forwards'):
        r"""Layer-curve diagnostic frame for a distorted portfolio.

        Thin wrapper over
        :func:`aggregate._portfolio_common.allocation_diagnostics`. See the free
        function for the full column list.
        """
        return _common.allocation_diagnostics(
            self, distortion, surface=surface, view=view,
            S_calculation=S_calculation)

    def var_dict(self, p, kind='lower', total='total', snap=False):
        """
        make a dictionary of value at risks for each unit and the whole portfolio.

         Returns: {unit : var(p, kind)} and includes the total as self.name unit

        Example:

            for p, arg in zip([.996, .996, .996, .985], ['var', 'lower', 'upper', 'tvar']):
                print(port.var_dict(p, arg,  snap=True))

        :param p:
        :param kind: var (defaults to lower), upper, lower, tvar
        :param total: name for total: total=='name' gives total name self.name
        :param snap: snap tvars to index
        :return:
        """
        if kind == 'var':
            kind = 'lower'

        if kind == 'tvar':
            d = {a.name: a.tvar(p) for a in self.agg_list}
            d['total'] = self.tvar(p)
        else:
            d = {a.name: a.q(p, kind) for a in self.agg_list}
            d['total'] = self.q(p, kind)
        if total != 'total':
            d[self.name] = d['total']
            del d['total']
        if snap and kind == 'tvar':
            d = {k: self.snap(v) for k, v in d.items()}
        return d

    def price(self, p, distortion=None, *, allocation=None, view='ask'):
        """Price the total under a distortion and allocate to units.

        ``rho(X ∧ q(p))`` for a single distortion (or a dict / list of
        distortions). ``p`` is a probability if ``p ≤ 1`` (converted to
        assets via VaR and snapped to the index) and an asset level
        otherwise. ``allocation`` defaults to :attr:`allocation_method`
        (``'linear'`` out of the box); pass ``'lifted'`` to override.

        Both methods read rows of the same unified augmented frame
        (:meth:`apply_distortion`); the only difference is the tail
        share allocating the collapsed default states -- objective
        ``alpha`` (linear) vs distorted ``beta`` (lifted). Lifted is
        unstable on the right edge for distortions with a mass on an
        unbounded support; the builder **refuses** that combination and
        points at ``'linear'``.

        Parameters
        ----------
        p : float
            VaR probability when ``p <= 1``; asset level when ``p > 1``.
        distortion : Distortion | list | dict | None
            One distortion, several, or ``None`` to use
            :attr:`self.distortions`.
        allocation : {'linear', 'lifted', None}, optional
            ``None`` (default) reads :attr:`allocation_method`.
        view : {'ask', 'bid'}
            Pricing view.

        Returns
        -------
        PricingResult
            Per-unit ``df`` (pentagon columns), the total price scalar,
            ``price_dict`` keyed by distortion, and ``a_reg`` / ``reg_p``.
        """
        if allocation is None:
            allocation = self._allocation_method
        if allocation not in ('lifted', 'linear'):
            raise ValueError(
                f"allocation must be 'lifted' or 'linear', not {allocation!r}")

        if isinstance(distortion, Distortion):
            distortion = {str(distortion): distortion}
        elif isinstance(distortion, list):
            distortion = {str(d): d for d in distortion}
        elif distortion is None:
            assert self.distortions is not None, 'Must pass a distortion or calibrate distortions prior to calling'
            distortion = self.distortions

        # figure regulatory assets; applied to unlimited losses
        if p > 1:
            a_reg = self.snap(p)
            reg_p = self.cdf(a_reg)
        else:
            a_reg = self.q(p)
            reg_p = p

        dfs = {}
        price = {}
        last_price = 0
        for k, v in distortion.items():
            logger.info(f'Executing for {k}, {allocation}')
            aug_df = self.apply_distortion(v, view=view, allocation=allocation)
            if a_reg in aug_df.index:
                aug_row = aug_df.loc[a_reg]
                a_eff = a_reg
            else:
                logger.warning('Regulatory assets not in augmented_df. Using last.')
                aug_row = aug_df.iloc[-1]
                a_eff = float(aug_row['loss'])

            df = pd.DataFrame(
                index=pd.Index(list(self.unit_names_ex), name='unit'),
                columns=['L', 'M', 'P', 'Q'],
                dtype=float,
            )
            unit_q = self._unit_capital_at(aug_df, a_eff, v, view=view)
            for unit in self.unit_names:
                df.loc[unit, 'L'] = aug_row[f'exa_{unit}']
                df.loc[unit, 'P'] = aug_row[f'exag_{unit}']
                df.loc[unit, 'Q'] = unit_q[unit]
            df.loc['total', 'L'] = aug_row['exa_total']
            df.loc['total', 'P'] = aug_row['exag_total']
            df['M'] = df.P - df.L
            # exact total capital, immune to layer-sum drift
            df.loc['total', 'Q'] = a_eff - aug_row['exag_total']
            df = complete_pentagon(df)
            price[k] = last_price = df.loc['total', 'P']
            dfs[k] = df.sort_index()

        df = pd.concat(dfs.values(), keys=dfs.keys(), names=['distortion', 'unit'])
        return PricingResult(df, last_price, price, a_reg, reg_p)

    def price_stand_alone(self, dist, p):
        """
        Price each unit on a stand-alone basis and compare to the diversified whole.

        Every unit is priced *as if it were the only unit in the book*: its
        capital standard is its own VaR at level ``p`` (no diversification
        credit), and the distortion ``dist`` is applied to its own loss
        distribution. This is contrasted with the ``total`` column — the whole
        portfolio priced together at the portfolio VaR(``p``) — and with the
        ``sum`` column, the simple sum of the stand-alone parts. The gap between
        ``sum`` and ``total`` is the diversification benefit.

        Each unit's stand-alone row is produced by :meth:`Aggregate.price`
        (a unit is just an :class:`Aggregate`); the ``total`` row by
        :meth:`pricing_at`. Both route through the canonical pentagon
        (:func:`~aggregate.pentagon.complete_pentagon`), so every row carries
        the full octet ``L, M, P, Q, a, LR, PQ, ROE`` derived in one place.

        Parameters
        ----------
        dist : Distortion or str
            The (already calibrated) pricing distortion, or the name of one in
            :attr:`distortions` (populated by :meth:`calibrate_distortions`).
        p : float
            Probability in ``(0, 1)``; the VaR capital standard for each
            stand-alone unit and for the total.

        Returns
        -------
        pandas.DataFrame
            Canonical pentagon orientation: the eight statistics ``L, M, P, Q,
            a, LR, PQ, ROE`` are the columns; rows are one per priced entity —
            the units, plus ``total`` (diversified whole) and ``sum`` (sum of
            the stand-alone parts) — under a ``(method, unit)`` MultiIndex, where
            ``method`` is the distortion's string form. Transpose for the
            traditional stat-down-the-side exhibit (``a.T``), matching
            ``analyze_distortion(...).pricing_df.T``.

        Raises
        ------
        TypeError
            If ``dist`` is neither a :class:`Distortion` nor a string, or ``p``
            is not numeric.
        ValueError
            If ``p`` is not in ``(0, 1)``, or a distortion name is requested
            when no distortions have been calibrated.
        KeyError
            If a distortion name is not found in :attr:`distortions`.

        Notes
        -----
        The amounts (``L, M, P, Q, a``) add across units, so the ``sum`` row is
        their column-wise total; the ratios (``LR, PQ, ROE``) do not add and are
        re-derived from the summed amounts via ``complete_pentagon``.
        """
        # ---- validate p -------------------------------------------------
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError(f'p must be a probability in (0, 1), got {p!r}.')
        if not 0 < p < 1:
            raise ValueError(f'p must be a probability in (0, 1), got {p}.')

        # ---- resolve the distortion ------------------------------------
        if isinstance(dist, str):
            if not self.distortions:
                raise ValueError(
                    f'No calibrated distortions on this Portfolio; cannot look '
                    f'up {dist!r}. Pass a Distortion instance, or call '
                    f'calibrate_distortions(...) first.')
            try:
                dist = self.distortions[dist]
            except KeyError:
                raise KeyError(
                    f'Distortion {dist!r} not found; available: '
                    f'{sorted(self.distortions)}.')
        elif not isinstance(dist, Distortion):
            raise TypeError(
                f'dist must be a Distortion or the name of a calibrated '
                f'distortion, got {type(dist).__name__}.')

        # ---- per-unit stand-alone pentagons (each at its OWN VaR(p)) ----
        parts = pd.concat([ag.price(p, dist) for ag in self.agg_list])

        # ---- the whole book priced together at the portfolio VaR(p) -----
        total = self.pricing_at(dist, p=p).loc[['total']]

        # ---- sum of the stand-alone parts: amounts add, ratios re-derive --
        sop = parts[['L', 'M', 'P', 'Q']].sum().to_frame('sum').T
        sop.index.name = 'unit'
        sop = complete_pentagon(sop)

        # ---- assemble in canonical orientation: stats are the columns, one
        # row per entity (units, then total, then sum), tagged by method ----
        exhibit = pd.concat([parts, total, sop])[list(PENTAGON_STATS)]
        exhibit.index.name = 'unit'
        exhibit.columns.name = 'stat'
        return pd.concat({str(dist): exhibit}, names=['method'])

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

        Pure accounting completion against the portfolio's total expected loss
        at the chosen capital level -- **no distortion is involved** (contrast
        :meth:`price`, which calibrates and applies a :class:`Distortion`). The
        triple ``{L, a, target}`` is solved by :meth:`Pentagon.solve`.
        Generalizes :meth:`price_ccoc` (the cost-of-capital special case).

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
        return _pricing.price_pentagon(
            self, p=p, a=a, P=P, M=M, Q=Q, LR=LR, PQ=PQ, ROE=ROE)

    def price_ccoc(self, ccoc, *, p):
        """
        Convenience function to price with a constant cost of captial equal ``ccoc``
        at VaR level ``p``. Does not invoke a Distortion. Returns the standard
        canonical pentagon DataFrame (one ``'total'`` row, columns
        :data:`~aggregate.pentagon.PENTAGON_STATS`).

        Thin alias for :meth:`price_pentagon` with the cost-of-capital target::

            self.price_pentagon(p=p, ROE=ccoc)
        """
        return _pricing.price_ccoc(self, ccoc, p=p)

    def prob_loss_assets(self, *, p=None, L=None, a=None):
        """Given any one of ``p``, ``L``, ``a``, return the consistent triple.

        Free choice over the capital anchor: pass exactly one of the VaR
        probability ``p``, the limited expected loss ``L = E[min(X, a)]``, or
        the asset level ``a`` -- any one determines the other two. Thin
        delegator to
        :meth:`~aggregate._grid_distribution.GridDistribution.prob_loss_assets`
        over the portfolio total ``p_total`` grid (the single ``lev`` source,
        matching the ``exa_total`` / ``add_exa`` convention). Aliased
        :meth:`pla`.

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
        capital anchor (``p``, ``a``, **or** the total limited expected loss
        ``L``) and accepting any soluble pentagon configuration, warning when an
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

    def analyze_distortion(self, distortion, *, p=None, a=None, kind='lower'):
        """
        Pricing readout for ``distortion`` at probability ``p`` or asset level ``a``.

        Parameters
        ----------
        distortion : Distortion or str
            A ``Distortion`` instance, or the name of a previously calibrated
            distortion (looked up in ``self.distortions``).
        p : float, optional
            Probability; converted to asset level via ``self.q(p, kind)``.
            Exactly one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.
        kind : {'lower', 'upper'}
            Type of VaR (only relevant when ``p`` is provided).

        Returns
        -------
        AnalyzeDistortionResult
            Holds the per-unit pricing DataFrame (from :meth:`pricing_at`)
            and a one-row ``audit_df`` for the total: descriptor columns
            ``dname``, ``dshape`` first, then the canonical pentagon octet
            (:data:`~aggregate.pentagon.PENTAGON_STATS`) as the trailing eight.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'analyze_distortion requires exactly one of p= (probability) '
                'or a= (asset level).')
        if isinstance(distortion, str):
            distortion = self.distortions[distortion]
        if a is None:
            a_cal = self.q(p, kind)
        else:
            a_cal = self.snap(a)
        pricing_df = self.pricing_at(distortion, a=a_cal)
        # one-row audit, same orientation as every other readout: descriptors
        # (dname/dshape) lead, the pentagon octet is the trailing [-8:].
        audit_df = pd.DataFrame(
            {'dname': distortion.name, 'dshape': distortion.shape,
             'L': pricing_df.loc['total', 'L'],
             'M': pricing_df.loc['total', 'M'],
             'P': pricing_df.loc['total', 'P'],
             'Q': pricing_df.loc['total', 'Q']},
            index=pd.Index(['total'], name='unit'),
        )
        audit_df = complete_pentagon(audit_df)
        return AnalyzeDistortionResult(
            distortion=distortion,
            pricing_df=pricing_df,
            audit_df=audit_df,
        )

    def analyze_distortions(self, *, p=None, a=None, distortions=None):
        """
        Pricing readout for a set of distortions at probability ``p`` or asset ``a``.

        Parameters
        ----------
        p : float, optional
            Probability; converted to asset level via ``self.q(p)``. Exactly
            one of ``p`` or ``a`` must be provided.
        a : float, optional
            Asset level; snapped to the index. Exactly one of ``p`` or ``a``
            must be provided.
        distortions : dict[str, Distortion], optional
            The distortions to analyse. Defaults to ``self.distortions`` (populated
            by :meth:`calibrate_distortions`).

        Returns
        -------
        AnalyzeDistortionsResult
            ``pricing_df`` is the concatenated exhibit with MultiIndex
            ``(distortion, stat)`` on rows and unit names on columns;
            ``stat`` runs over ``['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE', 'a']``.
            ``augmented_dfs`` is a snapshot of the cache for the analysed
            distortions.

        Notes
        -----
        Replaces both the legacy ``analyze_distortions(a=0, p=0, ...)`` and
        ``analyze_distortions2(p, dists=None)``. The output shape matches the
        legacy ``analyze_distortions2``: rows are ``(distortion, stat)``,
        columns are unit names.

        A mass distortion on an unbounded portfolio cannot build the
        lifted frame (the mass lands on the last represented bucket); such
        members of the sweep are skipped with a ``UserWarning`` -- price
        them explicitly with ``price(..., allocation='linear')``.
        """
        if (p is None) == (a is None):
            raise ValueError(
                'analyze_distortions requires exactly one of p= (probability) '
                'or a= (asset level).')
        distortions = distortions or self.distortions
        if not distortions:
            raise ValueError(
                'No distortions to analyse. Pass distortions=, or call '
                'calibrate_distortions first.')
        if a is None:
            a_cal = self.q(p)
        else:
            a_cal = self.snap(a)
        per_dist = {}
        for name, d in distortions.items():
            # a mass distortion on an unbounded support cannot build the
            # lifted frame (numerics-3 G6); skip it from the sweep with a
            # visible warning rather than failing the whole exhibit.
            if getattr(d, 'has_mass', False) and not self.bounded:
                warnings.warn(
                    f'analyze_distortions: skipping {name} -- mass '
                    f'distortion on an unbounded portfolio (lifted frame '
                    f'refused). Price it explicitly with '
                    f"allocation='linear'.")
                continue
            # pricing_at returns units × canonical pentagon columns; transpose
            # so stats are rows and units are columns. The transpose drops the
            # categorical column dtype, so work in plain string labels here and
            # reapply the canonical stat order/dtype after concat.
            exhibit = self.pricing_at(d, a=a_cal).T
            exhibit.index = exhibit.index.astype(str)
            # 'a' row: P + Q per unit, rescaled so totals sum to a_cal.
            a_row = exhibit.loc['P'] + exhibit.loc['Q']
            a_row = a_row * a_cal / a_row['total']
            exhibit.loc['a'] = a_row
            # canonical stat order (pentagon.py), trailing octet semantics
            per_dist[name] = exhibit.reindex(PENTAGON_STATS)
        if not per_dist:
            raise ValueError(
                'analyze_distortions: nothing to price -- every requested '
                'distortion is a mass distortion on an unbounded portfolio.')
        pricing_df = pd.concat(
            per_dist.values(),
            keys=per_dist.keys(),
            names=['distortion', 'stat'],
        )
        # bake the canonical distortion order into level 0 and the canonical
        # stat order/dtype into level 1 of the index (survives the transpose).
        pricing_df.index = pricing_df.index.set_levels(
            pricing_df.index.levels[0].astype(DISTORTION_DTYPE), level='distortion')
        pricing_df.index = pricing_df.index.set_levels(
            pricing_df.index.levels[1].astype(PENTAGON_DTYPE), level='stat')
        # snapshot only the distortions analysed (default-key frames; the
        # cache key is (name, view, role, S_calculation, allocation))
        augmented_dfs = {
            n: frame for (n_, view_, role_, sc_, alloc_), frame
            in self._augmented_dfs.items()
            for n in distortions
            if n_ == n and view_ == 'ask' and sc_ == 'forwards'
            and alloc_ == 'lifted'
        }
        return AnalyzeDistortionsResult(
            distortions=dict(distortions),
            pricing_df=pricing_df,
            augmented_dfs=augmented_dfs,
        )


    @property
    def unit_renamer(self):
        """
        plausible defaults for nicer looking names

        replaces . or : with space and capitalizes (generally don't use . because it messes with
        analyze distortion....

        leaves : alone

        converts X1 to tex

        converts XM1 to tex with minus (for reserves)

        :return:
        """
        def rename(ln):
            # guesser ...
            if ln == 'total':
                return 'Total'
            if ln.find('.') > 0:
                return ln.replace('.', ' ').title()
            if ln.find(':') > 0:
                return ln.replace(':', ' ').title()
            # numbered units
            ln = re.sub('([A-Z])m([0-9]+)', r'$\1_{-\2}$', ln)
            ln = re.sub('([A-Z])([0-9]+)', r'$\1_{\2}$', ln)
            return ln

        if self._unit_renamer is None:
            self._unit_renamer = { ln: rename(ln) for ln in self.unit_names_ex}

        return self._unit_renamer

    def nice_program(self, wrap_col=90):
        """
        return wrapped version of port program
        :return:
        """
        return fill(self.program, wrap_col, subsequent_indent='\t\t', replace_whitespace=False)


    def bodoff(self, *, p=0.99, a=0):
        """
        Determine Bodoff layer asset allocation at asset level a or
        VaR percentile p, one of which must be provided. Uses formula
        14.42 on p. 284 of Pricing Insurance Risk.

        :param p: VaR percentile
        :param a: asset level
        :return: Bodoff layer asset allocation by unit
        """

        return _common.bodoff(self, p=p, a=a)

    def sample(self, n, replace=True, desired_correlation=None, keep_total=True):
        """
        Pull multivariate sample. Apply Iman Conover to induce correlation if required.

        Thin wrapper over :func:`aggregate._portfolio_sample.sample` (the
        dependence subsystem).
        """
        return _smpl.sample(self, n, replace=replace,
                            desired_correlation=desired_correlation,
                            keep_total=keep_total)

    @property
    def n_units(self):
        return len(self.unit_names)

    def make_comonotonic_allocations(self, max_loss=-1):
        """
        Make comonotonic version of kappas using Denuit's alogorithm.

        Pass in upper bound max_loss, or use self.q(1) by default.
        """
        if max_loss <= 0: max_loss = self.q(1)
        df = self.density_df.filter(regex='loss|p_total|exeqa_[^t]').loc[:max_loss]
        assert df is not None, 'Object must be updated to compute allocations.'
        s_grid = df.loss.to_numpy()
        pdf_s = df.p_total.to_numpy()
        bit = df.iloc[:, 2:]
        kappa = bit.T.to_numpy()
        # do the work
        kappa_tilde = make_comonotonic_allocations_work(s_grid, pdf_s, kappa)
        # add to extract and return
        dfnew = df.copy()
        for c, k in zip(bit, kappa_tilde):
            dfnew[f'{c}_t'] = k
        return dfnew

    def swap_density_df(self, new_df, padding=1):
        """Thin shim around :func:`swap_density_df` (module-level function).

        Kept for callers that already use the method form; new code
        should call the standalone function so the dependency on a
        pre-existing Portfolio object is explicit.
        """
        swap_density_df(self, new_df, padding=padding)


def make_awkward(log2, scale=False):
    """
    Decompose a uniform random variable on range(2**log2) into two parts
    using Eamonn Long's base 4 method.

    Usage: ::

        awk = make_awkward(16)
        awk.aligned_unit_density_df().cumsum().plot()
        awk.density_df.filter(regex='exeqa_[AB]|loss').plot()

    """
    n = 1 << (log2 // 2)
    sc = 1 << log2
    xs = [int(bin(i)[2:], 4) for i in range(n)]
    ys = [2 * i for i in xs]
    ps = [1 / n] * n
    if scale is True:
        xs = np.array(xs) / sc
        ys = np.array(ys) / sc

    A = Aggregate('A', exp_en=1, sev_name='dhistogram', sev_xs=xs, sev_ps=ps,
                      freq_name='empirical', freq_a=np.array([1]), freq_b=np.array([1]))
    B = Aggregate('B', exp_en=1, sev_name='dhistogram', sev_xs=ys, sev_ps=ps,
                      freq_name='empirical', freq_a=np.array([1]), freq_b=np.array([1]))
    awk = Portfolio('awkward', [A, B])
    awk.update(log2, 1/sc if scale else 1, remove_fuzz=True, padding=0)
    return awk


