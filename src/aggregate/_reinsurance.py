"""Reinsurance ceder/netter construction (Aggregate-only concern).

Extracted from ``distributions.py`` / ``_aggregate.py`` (Phase 1b, shared concerns). A leaf/near-leaf: it never imports ``_aggregate``/``_portfolio`` (takes plain data / a distribution object), which is what lets Portfolio reuse it in P4.
"""

import logging
from textwrap import fill

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import NoConvergence  # noqa

from .constants import (REINS_LABEL_GROSS, REINS_LABEL_NET,
                        REINS_LABEL_CEDED, REINS_LABEL_OUTPUT)
from .moments import (MomentAggregator, MomentWrangler, xsden_to_mwrangler,
                      _noise_aware_rel_error, _snap_noise)
from .utilities import ft

logger = logging.getLogger(__name__)

# Column layout shared with ``summary_df`` (see ``Aggregate._describe``): exact
# (``EX``) value, rebucketed (``Est``) value, and ``Change`` for the mean and CV;
# skew omits the change column (it is the hardest moment to estimate).
# ``EX``/``Est`` here are the reins bases, not theory/empirical.
REINS_DESCRIBE_COLS = ['EX', 'Est EX', 'Change EX',
                       'CV', 'Est CV', 'Change CV',
                       'Sk', 'Est Sk']


def _validate_reins_layers(reins_list, tol=1e-9):
    """
    Validate that a reinsurance program is entered bottom-up and non-overlapping.

    ``make_ceder_netter`` builds the ceder by walking layers in list order and
    tracking a running height; out-of-order or overlapping layers silently
    produce wrong cessions. Enforce the contract with a hard error here, at the
    single choke point used by every reinsurance path.

    Parameters
    ----------
    reins_list : list of (share, limit, attach)
        Layer tuples; ``limit`` may be ``np.inf`` for an unlimited top layer.
    tol : float
        Tolerance for the non-overlap comparison (absorbs float noise).

    Raises
    ------
    ValueError
        If attachments are not non-decreasing, or two layers overlap.

    Notes
    -----
    Rules over ``[(share, limit, attach), ...]``:

    - attachments must be **non-decreasing**;
    - layers must **not overlap**: ``attach_{i+1} >= attach_i + limit_i - tol``.

    Gaps between layers are allowed -- represent a gap with a zero-share layer
    ``0 po L xs A``. Zero-share entries still must respect ordering. Single-layer
    programs are trivially valid.
    """
    prev_attach = None
    prev_top = None
    for (share, limit, attach) in reins_list:
        if prev_attach is not None:
            if attach < prev_attach - tol:
                raise ValueError(
                    'Reinsurance layers must be entered bottom-up (lowest '
                    f'attachment first); got attachment {attach} after '
                    f'{prev_attach}. Enter layers in ascending order and use '
                    '"0 po L xs A" to represent a gap.')
            if attach < prev_top - tol:
                raise ValueError(
                    f'Reinsurance layers overlap: layer attaching at {attach} '
                    f'starts below the top ({prev_top}) of the preceding layer. '
                    'Layers must be non-overlapping; use "0 po L xs A" to '
                    'represent a gap.')
        prev_attach = attach
        prev_top = attach + (0 if np.isinf(limit) else limit)


def make_ceder_netter(reins_list, debug=False):
    """
    Build the netter and ceder functions. It is applied to occ_reins and agg_reins,
    so should be stand-alone.

    The reinsurance functions are piecewise linear functions from 0 to inf with
    kinks as needed to express the ceded loss as a function of subject (gross) loss.

    The entries in ``reins_list`` are tuples (share of, limit, attach) where share of is the
    percentage share, between 0 and 1.

    For example, if ``reins_list = [(1, 10, 0), (0.5, 30, 20)]`` the program is 10 x 10 and
    15 part of 30 x 20 (share=0.5). This requires nodes at 0, 10, 20, 50, and inf.

    It is easiest to make the ceder function. Ceded loss at subject loss at x equals
    the sum of the limits below x plus the cession to the layer in which x lies. The
    variable ``base`` keeps track of the layer, ``h`` of the sum (height) of lower layers.
    ``xs`` tracks the knot points, ``ys`` the values.

    ::

         Break (xs)   Ceded (ys)
              0            0
             10            0
             20           10
             50           25
            inf           25


    For example:
    ::

        %%sf 1 2

        c, n, x, y = make_ceder_netter([(1, 10, 10), (0.5, 30, 20), (.25, np.inf, 50)], debug=True)

        xs = np.linspace(0,250, 251)
        ys = c(xs)

        ax0.plot(xs, ys)
        ax0.plot(xs, xs, ':C7')
        ax0.set(title='ceded')

        ax1.plot(xs, xs-ys)
        ax1.plot(xs, xs, 'C7:')
        ax1.set(title='net')

    :param reins_list: a list of (share of, limit, attach), e.g., (0.5, 3, 2) means 50% share of 3x2
        or, equivalently, 1.5 part of 3 x 2. It is better to store share rather than part
        because it still works if limit == inf.
    :param debug: if True, return layer function xs and ys in addition to the interpolation functions.
    :return: netter and ceder functions; optionally debug information.
    """
    # hard error on out-of-order / overlapping layers (single choke point)
    _validate_reins_layers(reins_list)
    # poor mans inf
    INF = 1e99
    h = 0
    base = 0
    xs = [0]
    ys = [0]
    for (share, y, a) in reins_list:
        # part of = share of times limit
        if np.isinf(y):
            y = INF
        p = share * y
        if a > base:
            # moved to new layer, write out left-hand knot point
            xs.append(a)
            ys.append(h)
        # increment height
        h += p
        # write out right-hand knot points
        xs.append(a + y)
        ys.append(h)
        # update left-hand end
        base += (a + y)
    # if not at infinity, stay flat from base to end
    if base < INF:
        xs.append(np.inf)
        ys.append(h)
    ceder = interpolate.interp1d(xs, ys)
    netter = lambda x: x - ceder(x)
    if debug:
        return ceder, netter, xs, ys
    else:
        return ceder, netter


# ====================================================================
# Apply engine: ceder/netter convolution onto the model grid
# ====================================================================

def apply_reins_work(agg, reins_list, base_density, debug=False):
    """
    Actually do the work. Called by apply_reins and reins_audit_df.
    Only needs ``agg`` to get limits, which it must guess without q (not computed
    at this stage). Does not need to know if occ or agg reins,
    only that the correct base_density is supplied.

    :param agg: the owning :class:`Aggregate` (read-through only).
    :param reins_list:
    :param base_density: subject (gross) density on ``agg.xs``.
    :param debug:
    :return: ceder, netter, reins_df
    """
    ans = make_ceder_netter(reins_list, debug)
    if debug:
        # debug xs and ys are the knot points of the interpolation function; good for plotting
        ceder, netter, xs, ys = ans
    else:
        ceder, netter = ans
    # assemble df for answers
    reins_df = pd.DataFrame(
        {'loss': agg.xs, 'p_subject': base_density}).set_index('loss', drop=False)
    reins_df['loss_net'] = netter(reins_df.loss)
    reins_df['loss_ceded'] = ceder(reins_df.loss)
    # Rebucket the off-grid net/ceded values back onto the uniform model
    # grid (agg.xs == bs * arange, xs[0] == 0) via the selected
    # ``reins_bucket`` scheme. See _rebucket_to_grid for the mass identity.
    p_subject = np.asarray(base_density, dtype=float)
    p_net = agg._rebucket_to_grid(reins_df['loss_net'].to_numpy(), p_subject)
    p_ceded = agg._rebucket_to_grid(reins_df['loss_ceded'].to_numpy(), p_subject)
    reins_df['p_net'] = p_net
    reins_df['p_ceded'] = p_ceded
    # F_* columns were vestigial from the old interp1d-of-CDF algorithm;
    # the scatter computes p_net / p_ceded directly. The debug CDF panel
    # below cumsums inline.
    reins_df = reins_df[['loss', 'p_subject', 'loss_net',
                         'loss_ceded', 'p_net', 'p_ceded']]

    if debug is False:
        return ceder, netter, reins_df

    logger.debug('making re graphs.')
    # quick debug; need to know kind=occ|agg here. Throwaway debug plot
    # bound to ephemeral internals -- stays out of the plots subsystem (see
    # the boundary test: only top-level matplotlib is forbidden); matplotlib
    # is reached lazily through the Layer-0 canvas helper.
    from .plots import make_mosaic
    f, axd = make_mosaic('AB\nCD', figsize=(12, 9))
    xlim = agg._limits()
    # scale??
    x = np.linspace(0, xlim[1], 201)
    y = ceder(x)
    n = x - y
    nxs = netter(x)

    ax = axd['A']
    ax.plot(x, y, 'o')
    ax.plot(x, y)
    ax.plot(x, x, lw=.5, c='C7')
    ax.set(aspect='equal', xlim=xlim, ylim=xlim,
           xlabel='Subject', ylabel='Ceded',
           title=f'Subject and ceded\nMax ceded loss {y[-1]:,.1f}')

    ax = axd['B']
    ax.plot(x, nxs, 'o')
    ax.plot(x, n)
    ax.plot(x, x, lw=.5, c='C7')
    ax.set(aspect='equal', ylim=xlim,
           xlabel='Subject', ylabel='Net',
           title=f'Subject and net\nMax net loss {n[-1]:,.1f}')

    ax = axd['C']
    cdf = reins_df[['p_subject', 'p_net', 'p_ceded']].cumsum()
    cdf.columns = ['F_subject', 'F_net', 'F_ceded']
    cdf.plot(xlim=xlim, ax=ax)
    ax.set(title='Subject, net and ceded\ndistributions')
    ax.legend()

    ax = axd['D']
    reins_df.filter(regex='p_').plot(xlim=xlim, drawstyle='steps-post', ax=ax)
    ax.set(title='Subject, net and ceded\ndensities')
    ax.legend()

    return ceder, netter, reins_df


def apply_occ_reins(agg, debug=False):
    """
    Apply the entire occ reins structure and save output
    For by layer detail create reins_audit_df
    Makes sev_density_gross, sev_density_net and sev_density_ceded, and updates sev_density to the requested view.

    Not reflected in statistics df.

    :param agg: the owning :class:`Aggregate` (mutated in place).
    :param debug: More verbose.
    :return:
    """
    # generic function makes netter and ceder functions
    if agg.occ_reins is None:
        return
    logger.info('running apply_occ_reins')
    occ_ceder, occ_netter, occ_reins_df = apply_reins_work(agg, agg.occ_reins, agg.sev_density, debug)
    # Retain the ceder/netter step functions for the exact (EX) reporting
    # path; the rebucketed densities go straight onto the gcn members and
    # into ``reins_density_df`` (no persistent per-stage frame).
    agg.occ_ceder = occ_ceder
    agg.occ_netter = occ_netter
    agg.sev_density_gross = agg.sev_density
    agg.sev_density_net = occ_reins_df['p_net'].to_numpy()
    agg.sev_density_ceded = occ_reins_df['p_ceded'].to_numpy()
    if agg.occ_kind == 'ceded to':
        agg.sev_density = agg.sev_density_ceded
    elif agg.occ_kind == 'net of':
        agg.sev_density = agg.sev_density_net
    else:
        raise ValueError(f'Unexpected kind of occ reinsurance, {agg.occ_kind}')
    # ``est_sev_*`` is written by ``update_work`` from the post-reins
    # severity (the same density set above) using ``xsden_to_mwrangler``
    # -- no intermediate write needed here.


def apply_agg_reins(agg, debug=False, padding=1):
    """
    Apply the entire agg reins structure and save output.
    For by layer detail create reins_audit_df.
    Makes agg_density_gross, agg_density_net and agg_density_ceded, and
    updates agg_density to the requested view.

    Not reflected in statistics df: the post-reins empirical moments
    (``est_*`` and the ``stats_df['empirical']`` column) are written by
    ``update_work`` from the same density updated here.

    :param agg: the owning :class:`Aggregate` (mutated in place).
    :return:
    """
    # generic function makes netter and ceder functions
    if agg.agg_reins is None:
        return
    logger.info('Applying aggregate reinsurance for %s', agg.name)

    agg_ceder, agg_netter, agg_reins_df = apply_reins_work(agg, agg.agg_reins, agg.agg_density, debug)
    # Retain the ceder/netter for the exact (EX) reporting path; the
    # rebucketed densities go onto the gcn members / ``reins_density_df``.
    agg.agg_ceder = agg_ceder
    agg.agg_netter = agg_netter
    agg.agg_density_gross = agg.agg_density
    agg.agg_density_net = agg_reins_df['p_net'].to_numpy()
    agg.agg_density_ceded = agg_reins_df['p_ceded'].to_numpy()
    if agg.agg_kind == 'ceded to':
        agg.agg_density = agg.agg_density_ceded
    elif agg.agg_kind == 'net of':
        agg.agg_density = agg.agg_density_net
    else:
        raise ValueError(f'Unexpected kind of agg reinsurance, {agg.agg_kind}')

    # update ft of agg
    agg.ftagg_density = ft(agg.agg_density, padding)


# ====================================================================
# Reinsurance reporting (rationalized; see dev/reins-reporting.md)
# ====================================================================

def reins_density_df(agg):
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
    if agg.occ_reins is None and agg.agg_reins is None:
        logger.warning('Asking for reins_density_df, but no reinsurance specified. Returning None.')
        return None

    if agg._reins_density_df is None:
        xs = agg.xs
        has_occ = agg.occ_reins is not None
        has_agg = agg.agg_reins is not None
        # point mass at 0 == the "no cession" density (ceded 0 w.p. 1)
        zero = np.zeros_like(xs, dtype=float)
        zero[0] = 1.0

        # --- severity (occurrence-level) views -----------------------
        sev_gross = np.asarray(
            agg.sev_density_gross if agg.sev_density_gross is not None
            else agg.sev_density, dtype=float)
        sev_ceded = np.asarray(agg.sev_density_ceded, dtype=float) if has_occ else zero
        sev_net = np.asarray(agg.sev_density_net, dtype=float) if has_occ else sev_gross
        df = pd.DataFrame({
            'loss': xs,
            'p_sev_gross': sev_gross,
            'p_sev_ceded': sev_ceded,
            'p_sev_net': sev_net,
        }, index=pd.Index(xs, name='loss'))

        # --- aggregate of each occurrence severity view --------------
        # p_agg_gross is the TRUE gross aggregate (FFT of gross sev).
        agg_gross, _ = agg._fft_aggregate(sev_gross, agg.padding)
        df['p_agg_gross'] = agg_gross
        if has_occ:
            logger.info('Computing aggregates with gcn severities')
            agg_ceded_occ, _ = agg._fft_aggregate(sev_ceded, agg.padding)
            agg_net_occ, _ = agg._fft_aggregate(sev_net, agg.padding)
            df['p_agg_ceded_occ'] = agg_ceded_occ
            df['p_agg_net_occ'] = agg_net_occ
        else:
            df['p_agg_ceded_occ'] = zero
            df['p_agg_net_occ'] = agg_gross

        # --- aggregate-cover views -----------------------------------
        # subject = the aggregate input to the agg cover = aggregate of
        # the requested occ output (== p_agg_gross when no occ stage).
        if has_agg:
            df['p_agg_subject'] = np.asarray(agg.agg_density_gross, dtype=float)
            df['p_agg_ceded'] = np.asarray(agg.agg_density_ceded, dtype=float)
            df['p_agg_net'] = np.asarray(agg.agg_density_net, dtype=float)
        else:
            df['p_agg_subject'] = np.asarray(agg.agg_density, dtype=float)
            df['p_agg_ceded'] = zero
            df['p_agg_net'] = np.asarray(agg.agg_density, dtype=float)

        agg._reins_density_df = df

    return agg._reins_density_df


# ----- reinsurance stats: exact (EX) vs rebucketed (Est) -------------

def reins_moments6_from_raw(e1, e2, e3):
    """``(ex1, ex2, ex3, mean, cv, skew)`` from raw moments."""
    mw = MomentWrangler()
    mw.noncentral = (e1, e2, e3)
    return (e1, e2, e3, *mw.mcvsk)


def reins_exact_image_raw(agg, image_fn, p_subject):
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
    g = np.asarray(image_fn(agg.xs), dtype=float)
    p = np.asarray(p_subject, dtype=float)
    return (float(np.sum(g * p)),
            float(np.sum(g * g * p)),
            float(np.sum(g * g * g * p)))


def reins_agg6_from_sev_raw(agg, s1, s2, s3):
    """Compound exact severity raw moments into aggregate
    ``(ex1, ex2, ex3, mean, cv, skew)`` via the frequency."""
    f1, f2, f3 = agg.frequency.freq_moms(agg.n)
    a1, a2, a3 = MomentAggregator.agg_from_fs(f1, f2, f3, s1, s2, s3)
    return (a1, a2, a3, *MomentAggregator.static_moments_to_mcvsk(a1, a2, a3))


def reins_density6(agg, p):
    """``(ex1, ex2, ex3, mean, cv, skew)`` of a density on the grid."""
    mw = xsden_to_mwrangler(agg.xs, np.asarray(p, dtype=float))
    return (*mw.noncentral, *mw.mcvsk)


def reins_view_stats(agg):
    """Per-stage reinsurance moments on two bases: exact and rebucketed.

    Internal frame feeding :func:`reins_summary_df` (which surfaces the
    ``EX`` exact vs ``Est`` rebucketed comparison as its ``Change``
    column). The public per-layer summary is :func:`reins_stats_df`.
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
    ``NaN`` (mirroring ``summary_df``, which never re-estimates the input
    frequency); occ ceded / net carry the *unconditional* mean ``E[N]`` only
    (so ``freq * sev == agg`` per view), cv / skew ``NaN``. The aggregate
    stage has no sev rows and a degenerate freq row (all ``NaN``).
    Columns are ordered occ before agg, views gross/subject, ceded, net,
    and ``EX`` before ``Est``.

    **Est** ("empirical") reads the rebucketed densities from
    :func:`reins_density_df` through ``xsden_to_mwrangler``. The
    difference EX vs Est isolates the per-stage ``reins_bucket``
    rebucketing error (``linear`` preserves the mean exactly; ``nearest``
    biases it by at most ``bs/2``).

    Lazily built; invalidated by the ``reins_bucket`` setter and on
    ``update``. Returns ``None`` when no reinsurance is configured.
    """
    if agg.occ_reins is None and agg.agg_reins is None:
        return None
    if agg._reins_view_stats_cache is not None:
        return agg._reins_view_stats_cache

    measures = ['ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew']
    components = ['freq', 'sev', 'agg']
    row_index = pd.MultiIndex.from_product(
        [components, measures], names=['component', 'measure'])
    nan6 = (np.nan,) * 6

    rd = reins_density_df(agg)
    data = {}  # (stage, view, basis) -> Series over row_index

    def put(stage, view, basis, freq6, sev6, agg6):
        s = pd.Series(np.nan, index=row_index)
        for comp, six in zip(components, (freq6, sev6, agg6)):
            for m, v in zip(measures, six):
                s[(comp, m)] = v
        data[(stage, view, basis)] = s

    # ---- occurrence stage -------------------------------------------
    if agg.occ_reins is not None:
        p_gross = rd['p_sev_gross'].to_numpy()
        n = agg.n
        # Frequency is unchanged by occurrence reinsurance, so the gross
        # full moments are the theoretic reference (``EX``) for every view.
        # The model-output (``Est``) frequency is reported *unconditionally*
        # -- no division by ``P(attach)`` -- so ``freq * sev == agg`` within
        # each view; only the mean is meaningful (the per-view count is the
        # gross count), so cv / skew stay ``NaN``. The gross ``Est``
        # frequency is left ``NaN`` to mirror ``summary_df`` exactly (the
        # validation view never re-estimates the input frequency).
        f1, f2, f3 = agg.frequency.freq_moms(n)
        freq_gross = (f1, f2, f3,
                      *MomentAggregator.static_moments_to_mcvsk(f1, f2, f3))
        freq_est_uncond = (np.nan, np.nan, np.nan, f1, np.nan, np.nan)

        # EX: exact image moments of the gross severity, compounded; the
        # frequency reference is the gross full moments for every view.
        identity = lambda x: x
        for view, image_fn in [('gross', identity),
                               ('ceded', agg.occ_ceder),
                               ('net', agg.occ_netter)]:
            s1, s2, s3 = reins_exact_image_raw(agg, image_fn, p_gross)
            put('occ', view, 'EX',
                freq_gross,
                reins_moments6_from_raw(s1, s2, s3),
                reins_agg6_from_sev_raw(agg, s1, s2, s3))

        # Est: moments of the rebucketed densities; unconditional freq
        # (gross frequency left NaN to mirror summary_df).
        for view, scol, acol, freq6 in [
                ('gross', 'p_sev_gross', 'p_agg_gross', nan6),
                ('ceded', 'p_sev_ceded', 'p_agg_ceded_occ', freq_est_uncond),
                ('net', 'p_sev_net', 'p_agg_net_occ', freq_est_uncond)]:
            put('occ', view, 'Est',
                freq6,
                reins_density6(agg, rd[scol].to_numpy()),
                reins_density6(agg, rd[acol].to_numpy()))

    # ---- aggregate stage --------------------------------------------
    if agg.agg_reins is not None:
        p_subject = rd['p_agg_subject'].to_numpy()
        identity = lambda x: x
        for view, image_fn in [('subject', identity),
                               ('ceded', agg.agg_ceder),
                               ('net', agg.agg_netter)]:
            a1, a2, a3 = reins_exact_image_raw(agg, image_fn, p_subject)
            put('agg', view, 'EX', nan6, nan6,
                reins_moments6_from_raw(a1, a2, a3))
        for view, acol in [('subject', 'p_agg_subject'),
                          ('ceded', 'p_agg_ceded'),
                          ('net', 'p_agg_net')]:
            put('agg', view, 'Est', nan6, nan6,
                reins_density6(agg, rd[acol].to_numpy()))

    # Canonical column order: occ before agg; within a stage the views
    # gross/subject, ceded, net (never alphabetical); EX before Est.
    ordered = []
    if agg.occ_reins is not None:
        for view in ('gross', 'ceded', 'net'):
            for basis in ('EX', 'Est'):
                ordered.append(('occ', view, basis))
    if agg.agg_reins is not None:
        for view in ('subject', 'ceded', 'net'):
            for basis in ('EX', 'Est'):
                ordered.append(('agg', view, basis))
    out = pd.DataFrame(data)[ordered]
    out.columns = pd.MultiIndex.from_tuples(
        ordered, names=['stage', 'view', 'basis'])
    agg._reins_view_stats_cache = out
    return agg._reins_view_stats_cache


def reins_stats_df(agg):
    """Per-layer reinsurance layering summary (empirical, model-grid).

    A layering analysis with one column per reinsurance layer plus the
    gross book and the ceded / net totals. See :class:`Aggregate` for the
    full column / row documentation (carried on the delegating property).

    Lazily built; invalidated by the ``reins_bucket`` setter and on
    ``update``. Returns ``None`` when no reinsurance is configured.
    """
    if agg.occ_reins is None and agg.agg_reins is None:
        return None
    if agg._reins_stats_df is not None:
        return agg._reins_stats_df

    measures = ['ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew']
    components = ['freq', 'sev', 'agg']
    meta_rows = ['share', 'limit', 'attach', 'pr_attach', 'pr_detach',
                 'pr_loss', 'lol', 'output']
    row_index = pd.MultiIndex.from_tuples(
        [('meta', m) for m in meta_rows]
        + [(c, m) for c in components for m in measures],
        names=['component', 'measure'])
    nan6 = (np.nan,) * 6
    rd = reins_density_df(agg)
    xs = agg.xs
    n = agg.n
    data = {}

    def col(share=np.nan, limit=np.nan, attach=np.nan, pr_attach=np.nan,
            pr_detach=np.nan, pr_loss=np.nan, lol=np.nan, output=0.0,
            freq6=nan6, sev6=nan6, agg6=nan6):
        s = pd.Series(np.nan, index=row_index)
        for mk, mv in zip(meta_rows, (share, limit, attach, pr_attach,
                                      pr_detach, pr_loss, lol, output)):
            s[('meta', mk)] = mv
        for comp, six in zip(components, (freq6, sev6, agg6)):
            for m, v in zip(measures, six):
                s[(comp, m)] = v
        return s

    def moments6(raw):
        """(ex1, ex2, ex3, mean, cv, skew) from raw moments ``raw``."""
        return (*raw, *MomentAggregator.static_moments_to_mcvsk(*raw))

    def _raw3(density):
        d = np.asarray(density, dtype=float)
        return (float(np.sum(xs * d)),
                float(np.sum(xs * xs * d)),
                float(np.sum(xs * xs * xs * d)))

    def _pr_pos(density):
        """P(loss > 0) = 1 - mass in the zero bucket."""
        return float(1.0 - np.asarray(density, dtype=float)[0])

    def _sf(a):
        try:
            return float(agg.sev.sf(a))
        except Exception:  # pragma: no cover - exotic severities
            return np.nan

    def _ge(density, t):
        """P(loss >= t) from a density, ``NaN`` if ``t`` is not finite.
        Used for the *aggregate*-level attach/detach probabilities (the
        modeled aggregate density is exact)."""
        if not np.isfinite(t):
            return np.nan
        return float(np.sum(np.asarray(density, dtype=float)[xs >= t]))

    # Claim-count weights and the ground-up severity survival, used for the
    # occurrence attach/detach probabilities. The modeled ``sev_density``
    # is *conditional* (claims to the policy layer, ``n`` the conditional
    # count), so the unconditional exposure probabilities -- P(a ground-up
    # claim's subject loss exceeds a threshold) -- come from the underlying
    # frozen severity ``fz`` with each component's policy attachment as the
    # offset. The limited ``agg.sev.sf`` would report 0 at the policy cap
    # by definition; the ground-up ``fz`` gives the true detachment prob.
    en = np.asarray(agg.en, dtype=float)
    ws = (en / en.sum() if en.sum() > 0
          else np.full(len(agg.sevs), 1.0 / max(len(agg.sevs), 1)))

    def _gsf(t, inclusive=False):
        """Weighted P(subject loss > t) (``>= t`` if ``inclusive``).

        ``subject_i = layer(X_i; attach_i, limit_i)``, so subject > t iff
        the ground-up ``X_i > attach_i + t`` (and t below the cap)."""
        try:
            tot = 0.0
            for i, sev in enumerate(agg.sevs):
                lim = float(sev.limit)
                if t < lim or (inclusive and t <= lim):
                    tot += ws[i] * float(sev.fz.sf(float(sev.attachment) + t))
            return float(tot)
        except Exception:  # pragma: no cover - exotic severities
            return np.nan

    def _gross_detach():
        """Weighted P(policy detaches) = P(ground-up >= attach + limit)
        over finite-limit components; ``NaN`` if every component unlimited."""
        fin = [i for i, sev in enumerate(agg.sevs)
               if np.isfinite(float(sev.limit))]
        if not fin:
            return np.nan
        try:
            return float(sum(
                ws[i] * float(agg.sevs[i].fz.sf(
                    float(agg.sevs[i].attachment) + float(agg.sevs[i].limit)))
                for i in fin))
        except Exception:  # pragma: no cover - exotic severities
            return np.nan

    def _lol(mean, placed_limit):
        """Loss on line: expected layer aggregate loss / placed limit."""
        if placed_limit is None or np.isnan(placed_limit):
            return np.nan
        if np.isinf(placed_limit):
            return 0.0
        return mean / placed_limit if placed_limit > 0 else np.nan

    def _layer_ceded(ceder, subject):
        """Rebucketed ceded density: subject mass through one layer."""
        return agg._rebucket_to_grid(
            np.asarray(ceder(xs), dtype=float),
            np.asarray(subject, dtype=float))

    # Full gross frequency (count unchanged by reinsurance).
    f1, f2, f3 = agg.frequency.freq_moms(n)
    freq_full = moments6((f1, f2, f3))

    # Claim-count-weighted gross policy limit / attachment (mixtures);
    # ``en`` / ``ws`` were set with the ground-up survival helpers above.
    lim = np.asarray(agg.limit, dtype=float)
    att = np.asarray(agg.attachment, dtype=float)
    if en.sum() > 0:
        gross_limit = float(np.average(lim, weights=en))
        gross_attach = float(np.average(att, weights=en))
    else:  # zero-risk fallback
        gross_limit = float(np.mean(lim))
        gross_attach = float(np.mean(att))

    # Which view carries each stage's output=1 flag.
    occ_out = (REINS_LABEL_NET if agg.occ_kind == 'net of'
               else REINS_LABEL_CEDED) if agg.occ_reins is not None else None
    agg_out = (REINS_LABEL_NET if agg.agg_kind == 'net of'
               else REINS_LABEL_CEDED) if agg.agg_reins is not None else None
    # Gross is the aggregate subject (output=1) only when there is an
    # aggregate program but no occurrence program.
    gross_output = 1.0 if (agg.agg_reins is not None
                           and agg.occ_reins is None) else 0.0

    # ----- gross book (always present) -----
    p_sev_gross = rd['p_sev_gross'].to_numpy()
    gross_agg6 = reins_density6(agg, rd['p_agg_gross'].to_numpy())
    data[('occ', REINS_LABEL_GROSS)] = col(
        share=1.0, limit=gross_limit, attach=gross_attach,
        # exposure probabilities from the underlying ground-up severity
        pr_attach=_gsf(0.0),            # P(a ground-up claim hits the policy)
        pr_detach=_gross_detach(),      # P(it exhausts the policy limit)
        pr_loss=_pr_pos(rd['p_agg_gross'].to_numpy()),
        lol=_lol(gross_agg6[3], gross_limit),
        output=gross_output,
        freq6=freq_full,
        sev6=reins_density6(agg, p_sev_gross),
        agg6=gross_agg6)

    # ----- occurrence layering (conditional layers; unconditional totals) -----
    if agg.occ_reins is not None:
        p_gross = rd['p_sev_gross'].to_numpy()
        for k, (s, y, a) in enumerate(agg.occ_reins, 1):
            ceder_k, _ = make_ceder_netter([(s, y, a)])
            ceded_k = _layer_ceded(ceder_k, p_gross)
            # Conditioning for the layer freq / sev is relative to the
            # *policy* claims (the modeled count ``n``): P(subject > a |
            # policy loss) = agg.sev.sf(a). The displayed pr_attach /
            # pr_detach are the absolute ground-up exposure probabilities.
            pr = _sf(a)
            u1, u2, u3 = _raw3(ceded_k)              # raw (conditional on policy)
            if pr and not np.isnan(pr):              # condition on the layer
                cond = (u1 / pr, u2 / pr, u3 / pr)
            else:
                cond = (np.nan, np.nan, np.nan)
            lf = agg.frequency.freq_moms(n * pr)     # conditional count n'
            agg_k, _ = agg._fft_aggregate(ceded_k, agg.padding)
            agg6 = reins_density6(agg, agg_k)
            data[('occ', f'layer.{k}')] = col(
                share=s, limit=y, attach=a,
                pr_attach=_gsf(a),
                pr_detach=_gsf(a + y, inclusive=True) if np.isfinite(y) else np.nan,
                pr_loss=_pr_pos(agg_k),
                lol=_lol(agg6[3], s * y),
                freq6=moments6(lf), sev6=moments6(cond), agg6=agg6)
        placed = float(sum(s * y for (s, y, _a) in agg.occ_reins))
        min_attach = float(min(a for (_s, _y, a) in agg.occ_reins))
        ceded_agg6 = reins_density6(agg, rd['p_agg_ceded_occ'].to_numpy())
        data[('occ', REINS_LABEL_CEDED)] = col(
            limit=placed, attach=min_attach,
            pr_attach=_gsf(min_attach),   # P(any ceding) = P(hit lowest layer)
            pr_loss=_pr_pos(rd['p_agg_ceded_occ'].to_numpy()),
            lol=_lol(ceded_agg6[3], placed),
            output=1.0 if occ_out == REINS_LABEL_CEDED else 0.0,
            freq6=freq_full,
            sev6=reins_density6(agg, rd['p_sev_ceded'].to_numpy()),
            agg6=ceded_agg6)
        data[('occ', REINS_LABEL_NET)] = col(
            pr_attach=_gsf(0.0),          # P(any subject loss retained)
            pr_loss=_pr_pos(rd['p_agg_net_occ'].to_numpy()),
            output=1.0 if occ_out == REINS_LABEL_NET else 0.0,
            freq6=freq_full,
            sev6=reins_density6(agg, rd['p_sev_net'].to_numpy()),
            agg6=reins_density6(agg, rd['p_agg_net_occ'].to_numpy()))

    # ----- aggregate layering (freq/sev left NaN -- they don't combine) -----
    if agg.agg_reins is not None:
        p_subject = rd['p_agg_subject'].to_numpy()
        pr_subject = _pr_pos(p_subject)              # P(subject > 0)

        def _pr_agg(a):
            return float(np.sum(p_subject[xs > a]))

        for k, (s, y, a) in enumerate(agg.agg_reins, 1):
            ceder_k, _ = make_ceder_netter([(s, y, a)])
            ceded_k = _layer_ceded(ceder_k, p_subject)
            agg6 = reins_density6(agg, ceded_k)
            data[('agg', f'layer.{k}')] = col(
                share=s, limit=y, attach=a, pr_attach=_pr_agg(a),
                pr_detach=_ge(p_subject, a + y), pr_loss=_pr_pos(ceded_k),
                lol=_lol(agg6[3], s * y), agg6=agg6)
        placed = float(sum(s * y for (s, y, _a) in agg.agg_reins))
        min_attach = float(min(a for (_s, _y, a) in agg.agg_reins))
        ceded_agg6 = reins_density6(agg, rd['p_agg_ceded'].to_numpy())
        data[('agg', REINS_LABEL_CEDED)] = col(
            limit=placed, attach=min_attach, pr_attach=pr_subject,
            pr_loss=_pr_pos(rd['p_agg_ceded'].to_numpy()),
            lol=_lol(ceded_agg6[3], placed),
            output=1.0 if agg_out == REINS_LABEL_CEDED else 0.0,
            agg6=ceded_agg6)
        data[('agg', REINS_LABEL_NET)] = col(
            pr_attach=pr_subject,
            pr_loss=_pr_pos(rd['p_agg_net'].to_numpy()),
            output=1.0 if agg_out == REINS_LABEL_NET else 0.0,
            agg6=reins_density6(agg, rd['p_agg_net'].to_numpy()))

    ordered = list(data.keys())  # gross, occ block, agg block (insertion)
    out = pd.DataFrame(data)[ordered]
    out.columns = pd.MultiIndex.from_tuples(ordered, names=['view', 'layer'])
    agg._reins_stats_df = out
    return agg._reins_stats_df


def reins_summary_df(agg):
    """Per-stage reinsurance summary -- the daily driver.

    Mirrors the **economic view** of ``summary_df``. See :class:`Aggregate`
    for the full column / row documentation (carried on the delegating
    property). Derived from :func:`reins_view_stats`. Returns ``None`` when
    no reinsurance is configured.
    """
    if agg.occ_reins is None and agg.agg_reins is None:
        return None
    if agg._reins_describe is not None:
        return agg._reins_describe
    blocks = []
    if agg.occ_reins is not None:
        blocks.append(reins_describe_block(
            agg, 'occ', ['gross', 'ceded', 'net'], ['freq', 'sev', 'agg']))
    if agg.agg_reins is not None:
        blocks.append(reins_describe_block(
            agg, 'agg', ['subject', 'ceded', 'net'], ['agg']))
    agg._reins_describe = pd.concat(blocks)
    return agg._reins_describe


def reins_describe_block(agg, stage, views, comps):
    """One :func:`reins_summary_df` block: theoretic reference vs model output
    by view x component, mirroring the eight-column ``summary_df`` layout.

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
    rs = reins_view_stats(agg)
    ref_view = views[0]  # Gross (occ) / Subject (agg) -- theoretic reference
    rows = []
    idx = []
    for view in views:
        for comp in comps:
            def _ref_est(measure):
                ref = float(rs.loc[(comp, measure), (stage, ref_view, 'EX')])
                est = float(rs.loc[(comp, measure), (stage, view, 'Est')])
                return ref, est
            ref_m, est_m = _ref_est('mean')
            ref_cv, est_cv = _ref_est('cv')
            ref_sk, est_sk = _ref_est('skew')
            rows.append([
                ref_m, est_m, float(_noise_aware_rel_error(est_m, ref_m)),
                ref_cv, est_cv, float(_noise_aware_rel_error(est_cv, ref_cv)),
                ref_sk, est_sk,
            ])
            idx.append((stage, view, comp))
    mi = pd.MultiIndex.from_tuples(idx, names=['stage', 'view', 'component'])
    df = pd.DataFrame(rows, index=mi, columns=REINS_DESCRIBE_COLS)
    # Snap fp dust to 0 in the value columns (skew of a symmetric view);
    # the Change columns keep their dust as the rebucketing eyeball.
    for c in df.columns:
        if not c.startswith('Change'):
            df[c] = _snap_noise(df[c])
    return df


# ====================================================================
# Reinsurance narrative
# ====================================================================

def reins_description(agg, kind='both', width=0):
    """
    Text description of the reinsurance (parameterized worker).

    :param kind: both, occ, or agg
    :param width: width of text for textwrap.fill; omitted if width==0
    """
    ans = []
    if agg.occ_reins is not None and kind in ['occ', 'both']:
        ans.append(agg.occ_kind)
        ra = []
        for (s, y, a) in agg.occ_reins:
            if np.isinf(y):
                ra.append(f'{s:,.0%} share of unlimited xs {a:,.0f}')
            else:
                if s == y:
                    ra.append(f'{y:,.0f} xs {a:,.0f}')
                else:
                    ra.append(f'{s:,.0%} share of {y:,.0f} xs {a:,.0f}')
        ans.append(' and '.join(ra))
        ans.append('per occurrence')
    if agg.agg_reins is not None and kind in ['agg', 'both']:
        if len(ans):
            ans.append('then')
        ans.append(agg.agg_kind)
        ra = []
        for (s, y, a) in agg.agg_reins:
            if np.isinf(y):
                ra.append(f'{s:,.0%} share of unlimited xs {a:,.0f}')
            else:
                if s == y:
                    ra.append(f'{y:,.0f} xs {a:,.0f}')
                else:
                    ra.append(f'{s:,.0%} share of {y:,.0f} xs {a:,.0f}')
        ans.append(' and '.join(ra))
        ans.append('in the aggregate.')
    if len(ans):
        # capitalize
        s = ans[0]
        s = s[0].upper() + s[1:]
        ans[0] = s
        reins = ' '.join(ans)
    else:
        reins = 'No reinsurance'
    if width:
        reins = fill(reins, width)
    return reins


def reins_kinds(agg):
    """Text description of kinds of reinsurance applied.

    Returns
    -------
    str
        One of ``'None'``, ``'Occurrence only'``, ``'Aggregate only'``, or
        ``'Occurrence and aggregate'``.
    """
    n = 1 if agg.occ_reins is not None else 0
    n += 2 if agg.agg_reins is not None else 0
    if n == 0:
        return "None"
    elif n == 1:
        return 'Occurrence only'
    elif n == 2:
        return 'Aggregate only'
    else:
        return 'Occurrence and aggregate'


def reins_after_label(agg):
    """Heading for the model-output column in ``summary_df``.

    ``Net`` when every cession passes the net; ``Ceded`` when every
    cession passes the ceded; ``Output`` when occ and agg pass
    different kinds (e.g. ``net of occ then ceded to agg`` -- a mixed
    output). Returns ``None`` when no reinsurance is configured (legacy
    validation-view headings apply).
    """
    kinds = []
    if agg.occ_reins is not None:
        kinds.append(agg.occ_kind)
    if agg.agg_reins is not None:
        kinds.append(agg.agg_kind)
    if not kinds:
        return None
    uniq = set(kinds)
    if uniq == {'net of'}:
        return REINS_LABEL_NET
    if uniq == {'ceded to'}:
        return REINS_LABEL_CEDED
    return REINS_LABEL_OUTPUT
