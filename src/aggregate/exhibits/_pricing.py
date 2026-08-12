"""Exhibit treatments for the pricing result objects.

The three leaves of the Pricing pane, ``pricing.calibrate`` /
``pricing.allocate`` / ``pricing.evaluate``, registered on
:class:`~aggregate.results.CalibrationResult` and
:class:`~aggregate.results.EvaluationResult` rather than on the objects those
were computed from. Ruling ``[Pricing-Keyed-On-Result]`` (author, 2026-08-11):
a calibration is a **calculation**, not stored state, and giving it a type is
what lets it dispatch through the same ``singledispatch`` registry as every
other exhibit with no framework change at all.

Two things follow, and both are contracts rather than conveniences.

**The RAW invariant survives.** A RAW block still names an attribute on the
dispatched object that returns a real public frame; the result carries
``distortion_df``, ``calibration_df``, ``pricing_df`` and ``reins_price_df``,
the last two computed on first access. Computing a frame on demand is an
efficiency question. Whether a public frame stands behind the block is the
contract question, and one does.

**The block list depends on the source as well as the perspective.** A
``Portfolio`` calibration spreads its target across units, a reinsured
``Aggregate`` across the views of its cession, and an ``Aggregate`` with
neither has one distribution and nothing to spread. That is ordinary Python
inside one registered builder, not a second dispatch mechanism.

The formats and the captions here were the app's until this landed. They are
statements about what the library's own numbers mean and how they read, so
they belong on this side of the wire (``dev/plan-pricing-exhibits.md``).
"""

import pandas as pd

from ..pentagon import PENTAGON_STATS, complete_pentagon
from ..results import CalibrationResult, EvaluationResult
from ._core import (
    pricing_allocate, pricing_calibrate, pricing_evaluate,
    register_simple_exhibit,
)

__all__ = [
    'PENTAGON_FORMATS', 'CALIBRATION_FORMATS', 'DISTORTION_FORMATS',
    'EVALUATION_FORMATS', 'STAT_SLICES', 'STAT_SLICE_FORMATS',
    'STAT_SLICE_TITLES',
]

#: Money to two decimals and the three ratios in their own units. Two decimals
#: rather than a magnitude aware choice: money is money at every scale, and a
#: table whose decimal count moves with the book is harder to read across than
#: one that is slightly over precise in places. ``PQ`` is a ratio and reads as
#: one, to three places; ``LR`` and ``ROE`` are percentages.
PENTAGON_FORMATS = {
    **{stat: ',.2f' for stat in ('L', 'M', 'P', 'Q', 'a')},
    'LR': '.1%', 'PQ': '.3f', 'ROE': '.1%',
}

#: The one row calibration target: its three input descriptors, then the octet.
#: ``p`` and ``F(a)`` are probabilities out at the fifth place, where the
#: difference between 0.99 and 0.99001 is the difference between what was asked
#: for and what the grid could deliver.
CALIBRATION_FORMATS = {'coc': '.1%', 'p': '.5f', 'F(a)': '.5f',
                       **PENTAGON_FORMATS}

#: The per family receipt. ``error`` is the premium miss, which is a residual
#: and reads in scientific notation for the same reason the probe's errors do:
#: at a fixed number of decimals a good fit and an exact fit both print zero,
#: and that comparison is the column's whole purpose.
DISTORTION_FORMATS = {'param': '.4f', 'error': '.2e', 'gini_p': '.4f',
                      'area': '.4f'}

#: The acceptability panel. ``gini_p`` is the index a reader compares across
#: families and down a walk, so it gets the digits to separate two close rows.
EVALUATION_FORMATS = {'param': '.4f', 'gini_p': '.4f', 'error': '.2e'}

#: The stats a portfolio allocation is read one slice at a time under INSURER,
#: with the title each slice carries. Four of the eight: the amounts ``L``,
#: ``M`` and ``Q`` and the asset level are in ``pricing_df`` whole, and the
#: comparison a reader makes across distortions is between prices and the
#: ratios that describe them.
STAT_SLICES = ('LR', 'P', 'PQ', 'ROE')

#: Slice titles, moved upstream from the app's ``PRICE_TITLE``.
STAT_SLICE_TITLES = {
    'LR': 'Loss ratio', 'P': 'Premium', 'PQ': 'Premium / capital',
    'ROE': 'Return on capital',
}

#: A slice's columns are units, not stats, so its format is uniform over the
#: frame rather than per column.
STAT_SLICE_FORMATS = {'LR': '.1%', 'P': ',.2f', 'PQ': '.3f', 'ROE': '.1%'}


# --- pricing.calibrate ------------------------------------------------------

register_simple_exhibit(
    'pricing.calibrate', 'Calibrated distortions', 'distortion_df',
    [CalibrationResult], formatters=DISTORTION_FORMATS,
    caption='One row per distortion family, each fitted to the same premium '
            'target: the family\'s own natural parameter, how far the fitted '
            'premium missed the target, and the two comparable readings of '
            'the shape. gini_p normalizes across families (it is the '
            'TVaR equivalent level), so it is the column to read down when '
            'asking which families agree; area is the same fact restated as '
            'the integral of g.')


# --- pricing.allocate -------------------------------------------------------

def _calibration_block(result):
    """The one row target, as its own block.

    Leads the allocation on the two sources that have units or views to spread
    over, and **is** the allocation on the one that has neither. An aggregate
    with no cession holds one distribution, so the calibration row is already
    the whole story and there is nothing further to allocate; saying so with
    the same block the other shapes lead with is better than serving an empty
    table or refusing the leaf.
    """
    return ('calibration_df', result.calibration_df,
            {'formatters': CALIBRATION_FORMATS,
             'caption': 'The shared calibration target: the cost of capital '
                        'asked for, the asset level it was struck at and the '
                        'probability that level sits at, then the pentagon '
                        'the three of them determine. Every family in the '
                        'set was fitted to this one premium.'})


@pricing_allocate.register(CalibrationResult)
def _allocate_raw(result):
    """RAW blocks, by what the calibration was made on.

    Three shapes, one registered builder: dispatching on ``type(_source)``
    here is ordinary Python and does not need a second registry. Each block
    still names a public attribute of the result and serves that frame whole,
    in its own orientation, so the invariant reads the same over a result as
    over an object.
    """
    if getattr(result._source, 'agg_list', None) is not None:
        return [
            _calibration_block(result),
            ('pricing_df', result.pricing_df,
             {'caption': 'The calibrated set allocated across the units of '
                         'the book, at the calibration asset level: the '
                         'eight pentagon statistics down the rows, units and '
                         'the total across. Each distortion is a different '
                         'answer to how the total premium should be shared '
                         'out, computed from the same total.'}),
        ]
    if getattr(result._source, 'reins_views', None):
        return [
            ('reins_price_df', result.reins_price_df,
             {'formatters': PENTAGON_FORMATS,
              'caption': 'Every view of the cession priced with every '
                         'calibrated family: gross, ceded and net are three '
                         'separate distributions, so each is a price in its '
                         'own right rather than a share of one. The ceded '
                         'row is what the risk measure says the cession is '
                         'worth to whoever writes it.'}),
        ]
    return [_calibration_block(result)]


def _star_the_basis(frame, basis):
    """Mark the calibrated view in the index, ``gross`` to ``gross*``.

    One row of the table was fitted and the rest were priced with the set that
    fit came out of, which is the single most important thing a reader of this
    table needs to know and the one thing the numbers cannot say themselves.
    """
    view = frame.index.get_level_values('view')
    starred = [f'{v}*' if v == basis else v for v in view]
    return frame.set_axis(
        pd.MultiIndex.from_arrays(
            [frame.index.get_level_values('distortion'), starred],
            names=frame.index.names),
        axis=0)


def _difference_rows(block, basis, views, distortion, names):
    """``basis less view`` rows for one distortion, ratios recomputed.

    The amounts difference and the ratios do **not**: a loss ratio of a
    difference is not the difference of two loss ratios, so the four amounts
    are differenced and :func:`complete_pentagon` re-derives ``a`` and the
    three ratios from them. The result is the allowance for reinsurance in the
    rate, read as a position in its own right, and its ``LR`` is the loss ratio
    the cover is being bought at.
    """
    rows, index = [], []
    for view in views:
        if view == basis or view not in block.index:
            continue
        rows.append([block.loc[basis, stat] - block.loc[view, stat]
                     for stat in ('L', 'M', 'P', 'Q')])
        index.append((distortion, f'{basis} less {view}'))
    if not rows:
        return None
    return complete_pentagon(pd.DataFrame(
        rows, columns=['L', 'M', 'P', 'Q'],
        index=pd.MultiIndex.from_tuples(index, names=names)))


@pricing_allocate.insurer.register(CalibrationResult)
def _allocate_insurer(result, blocks):
    """The buyer's reading of an allocation.

    On a book, the four ratio and price slices a reader actually compares,
    one block each, units across. On a cession, the whole program views only,
    the calibrated one starred, and the differences appended.
    """
    source = result._source
    if getattr(source, 'agg_list', None) is not None:
        return [blocks[0], *_stat_slices(blocks[1][1])]
    if getattr(source, 'reins_views', None):
        return _reins_insurer(result, blocks[0])
    return blocks


def _stat_slices(pricing_df):
    """One block per stat, distortions down and units across.

    ``pricing_df`` stacks eight statistics for every distortion into one tall
    frame, which is the honest raw shape and an unreadable comparison. A
    reader asking which distortion to use compares one statistic at a time
    across the same units, which is what these four blocks are.
    """
    out = []
    for stat in STAT_SLICES:
        if stat not in pricing_df.index.get_level_values('stat'):
            continue
        slice_ = pricing_df.xs(stat, level='stat')
        title = STAT_SLICE_TITLES[stat]
        out.append((f'stat_{stat}', slice_, {
            'float_format': STAT_SLICE_FORMATS[stat],
            'caption': f'{title} ({stat}) by distortion, per unit and in '
                       f'total, at the calibration asset level. The rows are '
                       f'the same book priced five ways; the spread between '
                       f'them is what the choice of risk measure is worth.'}))
    return out


def _reins_insurer(result, raw_block):
    """The whole program views, the calibrated one starred, the differences.

    ``ceded`` and ``ceded occ`` come out. A ceded price is what the layer is
    worth to the party writing it, which is a reinsurer's reading, and this
    perspective is the cedent's. The cedent's reading of the same cession is
    the difference between two of its own programs, which is appended instead.
    Ruling ``[Difference-Is-A-Perspective]``.
    """
    _name, frame, kw = raw_block
    basis = result.reins_view or 'net'
    program = [v for v in getattr(result._source, 'reins_views', [])
               if not v.startswith('ceded')]
    kept = frame[frame.index.get_level_values('view').isin(program)]
    # Distortion by distortion, so each family reads as one small table:
    # its views, then what the difference between them is worth. Appending
    # every difference row at the foot instead would put the comparison a
    # reader is making several rows away from the numbers it is made from.
    pieces = []
    for distortion in kept.index.get_level_values('distortion').unique():
        block = kept.xs(distortion, level='distortion', drop_level=False)
        pieces.append(block)
        if basis not in block.index.get_level_values('view'):
            continue
        differences = _difference_rows(
            block.droplevel('distortion'), basis, program, distortion,
            kept.index.names)
        if differences is not None:
            pieces.append(differences)
    kept = pd.concat(pieces)
    caption = (
        f'Distortions fitted to the {basis} basis at the calibration anchor, '
        f'then applied unchanged to the other whole program views. The '
        f'starred row is the calibrated one. A "less" row is the difference '
        f'between two programs: the implied allowance for reinsurance in the '
        f'rate, and its loss ratio is the rate the cover is being bought at. '
        f'It is not the price of the cession, which is what a reinsurer would '
        f'charge for the same layer and is the ceded row of the raw view.')
    return [('reins_price_df', _star_the_basis(kept, basis),
             dict(kw, caption=caption, formatters=PENTAGON_FORMATS))]


# --- pricing.evaluate -------------------------------------------------------

register_simple_exhibit(
    'pricing.evaluate', 'Breakeven acceptability', 'evaluation_df',
    [EvaluationResult], formatters=EVALUATION_FORMATS,
    caption='The distortion in each family whose risk adjusted margin is '
            'exactly zero: the most stress this position survives. One row '
            'block per position measured, so a walk reads down the steps and '
            'a book reads across its units.')


@pricing_evaluate.insurer.register(EvaluationResult)
def _evaluate_insurer(result, blocks):
    """What the panel means, over what it is.

    Background: Cherny and Madan (2009), *New measures for performance
    evaluation*. The caption names the three things a reader has to know to
    use the table, and the third is the one people get wrong: a blank row is
    two opposite situations and the ``status`` column is what tells them
    apart.
    """
    block_name, frame, kw = blocks[0]
    anchor = ('measured against the whole distribution'
              if result.a is None
              else f'measured with assets of {result.a:,.0f} behind it')
    premium = ('' if result.premium is None
               else f' The consideration is {result.premium:,.2f}.')
    caption = (
        f'The breakeven stress, {anchor}. gini_p is the family agnostic '
        f'acceptability index, so it compares across families and, for a '
        f'walk, down the steps: a layer whose figure sits above the net row '
        f'over it is priced above the holder\'s own acceptability, and buying '
        f'it lowers the net.{premium} A blank row is one of two opposite '
        f'situations and the status column says which: a position that is '
        f'unacceptable at any stress (its expected margin is not positive), '
        f'or one that cannot lose and is therefore acceptable at every '
        f'stress. Cherny and Madan (2009) is the reference.')
    return [(block_name, frame, dict(kw, caption=caption))]
