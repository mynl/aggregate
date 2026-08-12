"""``aggregate.exhibits`` -- business exhibits over greater_tables IR.

.. warning::

   **Provisional module, in the sense of PEP 411.** ``aggregate.exhibits`` is
   additive to the 1.0 release and is **not part of the 1.0 API contract**.
   Its API may change in a minor release with no deprecation period, unlike
   the stable core (:class:`~aggregate.Aggregate`,
   :class:`~aggregate.Portfolio`, :class:`~aggregate.PnL`,
   :class:`~aggregate.Severity`, :class:`~aggregate.Distortion`,
   :class:`~aggregate.Underwriter`, :func:`~aggregate.build` and the DecL
   grammar), which does carry the usual promise. Exhibit names, block
   structure, captions, row flags and the :class:`Perspective` vocabulary may
   all move. The module is deliberately public rather than underscore
   prefixed: use it, and report what does not fit. That feedback is how it
   graduates to stable in a later minor release. See
   :doc:`/3_reference/3_x_API_Stability`.

The library owns meaning, the app owns arrangement. An *exhibit* is a small
list of presentation ready tables (greater_tables ``TableDoc`` IR blocks)
derived from the first class citizen frames (``summary_df``, ``tail_df``,
``economic_df``, ...), carrying the business knowledge that would otherwise
leak into a client: captions, row emphasis, raw moment drops, relabeling. The
test for placement: if deleting the web app would destroy knowledge an actuary
would want in a notebook, that knowledge belongs here.

**The RAW invariant: a RAW block is exactly one public frame.** One block per
frame, named for the attribute that holds it, in the frame's own orientation,
with no split, no dropped rows and no rearrangement. A RAW block carries a
caption saying what the frame *is*, and the column formats for units the frame
cannot carry itself. Nothing else. **INSURER is the only perspective that may
restructure**: it re-orients, splits one frame into several blocks, merges,
drops rows and re-captions to say what the frame *means*. Ruling
``[Perspective-May-Restructure]``: the **block list itself** may differ between
perspectives, not merely the content of each block, so ``meta['blocks']`` is a
property of the (exhibit, perspective) pair and a client must not assume parity
across the two. Both halves are swept in ``tests/test_exhibits.py``.

The invariant is also a forcing function, which is its real value: it says what
a new exhibit owes. A RAW block with no frame behind it means the **frame** is
what is missing, and this package is not the place to invent one. The
``economic_waterfall`` blocks were the last exception and were promoted to
:attr:`~aggregate.PnL.walk_df` and :attr:`~aggregate.PnL.evaluation_df` at
``1.0.0a253`` rather than exempted.

**Dependencies point inward.** This package imports from the core; the core
never imports it, so nothing here touches an existing class and nothing here
can destabilize or delay 1.0. Work continues as attention allows and 1.0 ships
whether or not it is finished; anything incomplete, in particular any exhibit
meta-language, is explicitly post-1.0. Deliberately NOT star imported in
``aggregate/__init__.py`` (the ``Tweedie`` / ``Pentagon`` precedent). Reach it
with ``from aggregate import exhibits``, then
``exhibits.summary(obj, 'insurer')``.

Layout, mirroring :mod:`aggregate.plots` (``dev/plan-exhibits.md``):

- :mod:`._core` -- the machinery, class agnostic: :class:`Perspective`,
  :class:`Exhibit`, the singledispatch registry, the frame and IR stages, and
  the translation helpers more than one class shares.
- ``_aggregate`` / ``_portfolio`` / ``_pnl`` / ``_bivariate`` / ``_distortion``
  -- one module per class, holding **that class's business translation**. This
  is where you go to edit how an exhibit reads.
- The **manifest** at the foot of this file -- every plain passthrough
  exhibit, one line each, via :func:`register_simple_exhibit`.

Adding an exhibit is one of three sizes. A passthrough over an existing frame
is a manifest line. A new insurer translation of an existing exhibit is a
function in the class module. A wholly new multi block exhibit declares its
generic function in ``_core`` and registers builders in the class module.

Nothing at module scope imports greater_tables. The IR conversion step
(:func:`build_exhibit`, :meth:`Exhibit.to_payload`) imports it at the point of
use, so ``import aggregate`` and ``import aggregate.exhibits`` both stay free
of it even though it is a plain dependency (since 1.0.0a229). That greater_tables
is a plain dependency rather than an extra is a fact about installation, chosen
so the exhibit surface never raises ``ImportError`` on a supported interpreter.
It is not a stability promise, and it does not move this module into the 1.0
contract.
"""

from .._aggregate import Aggregate
from .._pnl import PnL
from .._portfolio import Portfolio
from ..bivariate import BivariateAggregate
from ..spectral import Distortion

from ._core import (
    CAPITAL_ANCHOR_PERIODS, EXHIBITS, Exhibit, INCLUDE_RAW, MAX_ROWS,
    MEASURE_FORMATS, Perspective, RAW_MOMENT_MEASURES,
    available_exhibits, build_exhibit, exhibit_frames, register_simple_exhibit,
    dependency, economic, economic_ratios, economic_waterfall, reins, stats,
    summary, tail, validation,
    pricing_allocate, pricing_calibrate, pricing_evaluate,
    _perspectives_sharpen, _perspectives_updated,
)

# Per-class business translation. Imported for their registration side
# effects, which is the whole point: importing the package wires the registry.
# ``_pricing`` is the same idea over the result objects rather than over a
# first class class: what a calibration dispatches on is its result.
from . import (_aggregate, _portfolio, _pnl, _bivariate,  # noqa: F401
               _distortion, _pricing)
from ._pricing import (  # noqa: F401
    CALIBRATION_FORMATS, DISTORTION_FORMATS, EVALUATION_FORMATS,
    PENTAGON_FORMATS, STAT_SLICES, STAT_SLICE_FORMATS, STAT_SLICE_TITLES,
)

__all__ = [
    'Perspective', 'Exhibit', 'EXHIBITS',
    'available_exhibits', 'exhibit_frames', 'build_exhibit',
    'register_simple_exhibit',
    'CAPITAL_ANCHOR_PERIODS', 'RAW_MOMENT_MEASURES',
    'summary', 'tail', 'stats', 'validation', 'reins',
    'economic', 'economic_ratios', 'economic_waterfall', 'dependency',
    'bs_window', 'sharpen', 'tail_behavior', 'SHARPEN_FORMATS',
    'pricing_calibrate', 'pricing_allocate', 'pricing_evaluate',
    'PENTAGON_FORMATS', 'CALIBRATION_FORMATS', 'DISTORTION_FORMATS',
    'EVALUATION_FORMATS', 'STAT_SLICES', 'STAT_SLICE_FORMATS',
    'STAT_SLICE_TITLES',
]


# --- the passthrough manifest -----------------------------------------------
# Every exhibit that serves one frame with no business translation, declared
# in one line. These register no insurer override, so INSURER equals RAW by
# the default rule; adding an override in a class module changes only that
# (exhibit, type) pair. Exhibits with a translation are declared in ``_core``
# and registered in their class module instead.
#
# Every line carries a **caption** (1.0.0a226, [Loss-Lab-Round-3] phase D).
# Without one a passthrough arrived as a bare table, and the client that
# wanted prose wrote its own, which is how a frame's description came to have
# three possible sources that could disagree. The captions here say what the
# frame *is*; the insurer overrides in the class modules say what it *means*
# for the business, and replace these where they exist.
#
# One frame does not have one description across five classes, so an exhibit
# registered for several is declared once per class group. ``summary_df`` is
# the clear case: count risk / severity / total loss on an Aggregate, the
# three ledger rows on a PnL, moments of g on a Distortion.

_LOSS_FCC = [Aggregate, Portfolio]

# ---- summary ---------------------------------------------------------------
register_simple_exhibit(
    'summary', 'Summary', 'summary_df', _LOSS_FCC,
    formatters=MEASURE_FORMATS,
    caption='Headline moments and key percentiles by component: count risk '
            '(Freq), single claim severity (Sev) and total loss (Agg), one '
            'block per unit on a portfolio. Percentiles are exact grid '
            'values. Frequency percentiles are blank by design, because '
            'frequency enters through its PGF and no count distribution is '
            'ever materialized.')
register_simple_exhibit(
    'summary', 'Summary', 'summary_df', [PnL],
    formatters=MEASURE_FORMATS,
    caption='The ledger in three rows: what was received (Consideration), '
            'what is owed (Obligation) and what is left (Margin), each with '
            'its moments and key percentiles.')
register_simple_exhibit(
    'summary', 'Summary', 'summary_df', [Distortion],
    caption='Moments of the distortion g and of its dual, each against its '
            'closed form where one exists, with the error between them.')
register_simple_exhibit(
    'summary', 'Summary', 'summary_df', [BivariateAggregate],
    caption='Reference against realized moments for the shared frequency, '
            'each marginal, and their total.')

# ---- stats -----------------------------------------------------------------
register_simple_exhibit(
    'stats', 'Statistics', 'stats_df', [Aggregate, Portfolio, PnL],
    caption='The canonical moment store: a meta block (limit, attachment, '
            'expected loss, premium, loss ratio, severity CV) over the '
            'moments at each stage of the calculation, from the analytic mix '
            'through any cession to the realized empirical grid, with the '
            'error between the last two.')
register_simple_exhibit(
    'stats', 'Statistics', 'stats_df', [Distortion],
    caption='Moments of the distortion, computed against closed form where '
            'one exists, with the error between them.')
register_simple_exhibit(
    'stats', 'Statistics', 'stats_df', [BivariateAggregate],
    caption='Theoretical against realized moments, one column per axis.')

# ---- validation ------------------------------------------------------------
register_simple_exhibit(
    'validation', 'Validation', 'validation_df', _LOSS_FCC,
    caption='Moment QA: the reference moment against the realized FFT '
            'estimate, with noise aware relative errors, for the frequency, '
            'severity and aggregate of each unit. Errors of this size are '
            'discretization, not model error.')
register_simple_exhibit(
    'validation', 'Validation', 'validation_df', [PnL],
    caption='Ledger QA: each declared amount against the realized estimate, '
            'with absolute and relative error.')
register_simple_exhibit(
    'validation', 'Validation', 'validation_df',
    [Distortion, BivariateAggregate],
    caption='Identity checks: the realized value against its reference, the '
            'gate it has to clear, and whether it cleared it.')

# ---- tail and economics ----------------------------------------------------
register_simple_exhibit(
    'tail', 'Return periods', 'tail_df', _LOSS_FCC,
    caption='Return period ladder read off the realized grid: VaR (the '
            'quoted number), TVaR (the priced number), excess VaR over the '
            'mean (the capital), and VaR to mean leverage.')
register_simple_exhibit(
    'economic', 'Economics', 'economic_df', [PnL],
    caption='The full ledger by side and label in currency units, then the '
            'kappa columns: what each line comes to when the book as a whole '
            'lands at that percentile. Read down for the ledger, across for '
            'a scenario.')

#: Diagnostics, the app's "More" material. Both need the realized grid.
bs_window = register_simple_exhibit(
    'bs_window', 'Grid sizing', 'bs_window_df',
    [Aggregate, Portfolio, BivariateAggregate],
    predicate=_perspectives_updated,
    caption='How the grid was chosen: the candidate windows, which one '
            'applied, and the bucket size and log2 that follow from it. A '
            'clipped row is a window that did not fit and was cut to the '
            'grid, which is where aliasing comes from.')
#: The probe's own audit reads in scientific notation, deliberately: the
#: ``u_`` columns are relative errors against the analytic moments and run
#: from about 1e-7 to a few percent, so at a fixed ``.4f`` a good cell and a
#: perfect cell both print ``0.0000``, which is exactly the comparison the
#: table exists to support. ``score`` is the number that decides, and gets the
#: digits to separate two cells that are close.
SHARPEN_FORMATS = {
    'score': '.5f', 'extent': ',.0f', 'x_min': ',.0f', 'bs': ',.4g',
    'u_sev_mean': '.2e', 'u_sev_cv': '.2e', 'u_sev_skew': '.2e',
    'u_agg_mean': '.2e', 'u_agg_cv': '.2e', 'u_agg_skew': '.2e',
    'aliasing': '.4f', 'deficit': '.2e', 'seconds': '.3f',
}

sharpen = register_simple_exhibit(
    'sharpen', 'Grid probe', 'sharpen_df', _LOSS_FCC,
    predicate=_perspectives_sharpen, formatters=SHARPEN_FORMATS,
    caption='Every grid the last probe tried, one row per cell walked, with '
            'its working: the realized bucket size and log2, the six '
            'normalized moment errors that make up the score, the aliasing '
            'ratio and the validation verdict. Steps are offsets from the '
            'grid the object started on, and the line search makes the walk '
            'ragged, so a cell it never reached is blank rather than bad.')


@sharpen.insurer.register(Aggregate)
@sharpen.insurer.register(Portfolio)
def _sharpen_insurer(obj, blocks):
    """Lead with the picture: the score grid, then the walk that produced it.

    Ruling ``[Sharpen-Grid-Is-A-Reading]``: ``score`` is a **column** on
    ``sharpen_df`` used for deciding, not a second fact, so the grid is a
    reading of a published frame rather than a frame of its own. That is
    exactly what an INSURER block is for, and this is the smallest exercise
    of ``[Perspective-May-Restructure]``: one block raw, two here.

    The unstack is the whole translation. A reader comparing grids wants the
    two step axes on the two axes of a table, not twenty columns of working
    with the deciding number buried among them.
    """
    (walk_name, walk, walk_kw), = blocks
    grid = walk['score'].unstack('d_log2')
    return [
        ('score_grid', grid,
         dict(formatters={c: '.5f' for c in grid.columns}, caption=(
             'Every cell the probe walked, scored: steps in bucket size down '
             'and steps in log2 across, from the grid the object started on. '
             'Lower is better, and the blanks are where the line search '
             'stopped rather than cells that failed.'))),
        (walk_name, walk, dict(walk_kw, caption=(
            'The same walk with its working, one row per grid tried. The six '
            'u_ columns are relative errors against the analytic moments and '
            'are what the score is built from; aliasing and the validation '
            'verdict say whether a cell was sound as well as accurate.'))),
    ]


tail_behavior = register_simple_exhibit(
    'tail_behavior', 'Tail behavior', 'tail_behavior_df',
    [Aggregate, Portfolio], predicate=_perspectives_updated,
    caption='What each tail looks like, as opposed to how far out it reaches: '
            'the realized support, the tail class on each side, whether the '
            'law is bounded, and the coefficient of variation. A different '
            'question from the return period ladder, which is the tail '
            'exhibit.')
