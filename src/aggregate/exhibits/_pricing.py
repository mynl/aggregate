"""Exhibit treatments for the pricing result objects.

The leaves of the Pricing pane, ``pricing.calibrate`` /
``pricing.stand_alone`` / ``pricing.allocate`` / ``pricing.evaluate``,
registered on
:class:`~aggregate.results.CalibrationResult` and
:class:`~aggregate.results.EvaluationResult` rather than on the objects those
were computed from. Ruling ``[Pricing-Keyed-On-Result]`` (author, 2026-08-11):
a calibration is a **calculation**, not stored state, and giving it a type is
what lets it dispatch through the same ``singledispatch`` registry as every
other exhibit with no framework change at all.

**Stand-alone prices the parts; allocate splits the whole.** Ruling
``[Standalone-Prices-The-Parts, Allocate-Splits-The-Whole]`` (author,
2026-08-14). The two are different questions and both are worth asking, so
they are two leaves rather than one table with a footnote. Sum the
stand-alone parts and compare with the whole and you have read the
diversification story; decompose the whole into the parts and you have read
the allocation. A ``Portfolio``'s ``pricing_df`` was an allocation all along
(``analyze_distortions`` documents its ``allocation=`` parameter as the tail
share choice for the per unit premium split) and stays put; the aggregate
side of the old ``pricing.allocate`` was stand-alone pricing under the wrong
name and moves.

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

from ..pentagon import complete_pentagon
from .._pricing import STAND_ALONE_SUM, STAND_ALONE_TOTAL
from ..results import CalibrationResult, EvaluationResult
from ._core import (
    pricing_allocate, pricing_evaluate, pricing_stand_alone,
    register_simple_exhibit,
)

__all__ = [
    'STAT_SLICES', 'STAT_SLICE_FORMATS', 'STAT_SLICE_TITLES',
]

# The pricing vocabulary is read from the format sheets like every other
# column's (``aggregate/formats/formats-raw.yaml``): the pentagon octet as
# money, ``LR`` / ``ROE`` / ``coc`` as ratios, ``PQ`` as a multiple to three
# places, ``p`` and ``F(a)`` as probabilities out at the fifth place where
# what was asked for and what the grid could deliver part company, and
# ``error`` as a residual, so a good fit and an exact fit do not both print
# zero. The one thing that stays here is the stat slice below, because its
# format is a property of the block's shape rather than of a column's meaning.

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
    [CalibrationResult],
    caption='One row per distortion family, each fitted to the same premium '
            'target: the family\'s own natural parameter, how far the fitted '
            'premium missed the target, and the two comparable readings of '
            'the shape. gini_p normalizes across families (it is the '
            'TVaR equivalent level), so it is the column to read down when '
            'asking which families agree; area is the same fact restated as '
            'the integral of g.')


# --- shared ------------------------------------------------------------------

def _calibration_block(result):
    """The one row target, as its own block.

    Leads the leaf on the sources that have parts to price or spread over, and
    **is** the stand-alone story on the one that has neither. An aggregate with
    no cession holds one distribution, so the calibration row is already the
    whole story: it has one part, which is the whole. Saying so with the same
    block the other shapes lead with is better than serving an empty table or
    refusing the leaf.
    """
    return ('calibration_df', result.calibration_df,
            {'caption': 'The shared calibration target: the cost of capital '
                        'asked for, the asset level it was struck at and the '
                        'probability that level sits at, then the pentagon '
                        'the three of them determine. Every family in the '
                        'set was fitted to this one premium.'})


# --- pricing.stand_alone -----------------------------------------------------

@pricing_stand_alone.register(CalibrationResult)
def _stand_alone_raw(result):
    """RAW blocks, by what the calibration was made on.

    Dispatching on ``type(_source)`` here is ordinary Python and does not need
    a second registry. Each block still names a public attribute of the result
    and serves that frame whole, in its own orientation, so the invariant reads
    the same over a result as over an object.
    """
    if getattr(result._source, 'agg_list', None) is not None:
        return [
            _calibration_block(result),
            ('stand_alone_df', result.stand_alone_df,
             {'caption': 'Each unit priced as its own distribution with the '
                         'same fitted families, at the book\'s calibrated '
                         'asset level; the sum of those prices against the '
                         'book priced whole at the same level. The gap is '
                         'what pooling is worth under that family. These are '
                         'separate prices rather than shares of one, so they '
                         'do not foot, which is the difference between this '
                         'table and the allocation.'}),
        ]
    if getattr(result._source, 'reins_views', None):
        return [
            ('reins_price_df', result.reins_price_df,
             {'caption': 'Every view of the cession priced with every '
                         'calibrated family: gross, ceded and net are three '
                         'separate distributions, so each is a price in its '
                         'own right rather than a share of one. The ceded '
                         'row is what the risk measure says the cession is '
                         'worth to whoever writes it.'}),
        ]
    return [_calibration_block(result)]


@pricing_stand_alone.insurer.register(CalibrationResult)
def _stand_alone_insurer(result, blocks):
    """The buyer's reading of stand-alone prices.

    On a book, the diversification benefit appended per family. On a cession,
    the whole program views only, the calibrated one starred, and the
    differences appended.
    """
    if getattr(result._source, 'agg_list', None) is not None:
        return [blocks[0], _book_benefit(result, blocks[1])]
    if getattr(result._source, 'reins_views', None):
        return _reins_insurer(result, blocks[0])
    return blocks


def _book_benefit(result, raw_block):
    """Append ``sum of parts less total`` per family: what pooling is worth.

    The buyer's restructure of a frame whose rows are all measurements. The
    sum and the total ride in RAW because both are facts (one is arithmetic on
    measurements, the other is a measurement); the difference between them is
    the reading, and a reading belongs to a perspective
    (``[Difference-Is-A-Perspective]``).

    Interleaved family by family rather than appended at the foot, the
    ``_reins_insurer`` pattern: the comparison a reader is making should sit
    next to the numbers it is made from.
    """
    _name, frame, kw = raw_block
    names = frame.index.names
    pieces = []
    for distortion in frame.index.get_level_values('distortion').unique():
        block = frame.xs(distortion, level='distortion', drop_level=False)
        pieces.append(block)
        difference = _difference_rows(
            block.droplevel('distortion'), STAND_ALONE_SUM,
            [STAND_ALONE_TOTAL], distortion, names)
        if difference is not None:
            pieces.append(difference)
    caption = (
        'Each unit priced as its own distribution with the same fitted '
        'families, at the book\'s calibrated asset level, then the sum of '
        'those prices against the book priced whole at that level. The '
        '"less" row is the diversification benefit: what the book saves by '
        'being one book rather than two, under that family. It is positive '
        'for every concave family, because a sum of parts capped at the '
        'book\'s assets is at least the book capped there and the risk '
        'measure is sub-additive. Its capital column is the mirror of its '
        'premium column by construction, both rows standing behind the same '
        'assets, so the reading is the premium and the margin.')
    return ('stand_alone_df', pd.concat(pieces),
            dict(kw, caption=caption))


# --- pricing.allocate -------------------------------------------------------

@pricing_allocate.register(CalibrationResult)
def _allocate_raw(result):
    """RAW blocks for the decomposition of one premium.

    Reached only where :func:`~aggregate.exhibits._core._perspectives_allocation`
    says there are parts to split across, so there is no "nothing to allocate"
    branch here: that case does not serve this leaf at all.
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
    return [
        ('natural_allocation_df', result.natural_allocation_df,
         {'caption': _allocation_caption(result)}),
    ]


def _allocation_caption(result):
    """What the occurrence allocation is, and what it is not.

    Three things a reader has to know and cannot read off the numbers: that
    this is one premium decomposed rather than two prices compared, which
    machinery produced the split, and how far the joint's grid sat from the
    one the fit was struck on.
    """
    frame = result.natural_allocation_df
    gaps = frame.attrs.get('rho_gap') or {}
    sizing = frame.attrs.get('joint_sizing') or {}
    worst = max((abs(v) for v in gaps.values() if v == v), default=None)
    grid = ''
    if sizing:
        exact = ('on the exact lattice' if sizing.get('exact_lattice')
                 else f'at bs = {sizing["bs"]:,.6g}')
        grid = (f' The split is computed on the joint {exact}, '
                f'2**{sizing["log2"][0]} by 2**{sizing["log2"][1]} cells, '
                f'carrying a deficit of {sizing["deficit"]:.1e}.')
    gap = ''
    if worst is not None:
        gap = (f' The joint\'s gross price differs from the fine 1-D reading '
               f'by at most {worst:,.2f} across the families (rho_gap), which '
               f'is reported rather than absorbed: the fractions are computed '
               f'on the joint\'s grid and applied to the calibrated premium.')
    return (
        'The calibrated gross premium split across the occurrence program on '
        'one consistent basis. Each family\'s distorted view of the gross '
        'distribution sets the weights; the kappa curve off the joint says '
        'what ceded and net each earn under them; ceded plus net foot to '
        'gross exactly. This is neither view priced on its own, which is the '
        'stand-alone table, nor the difference of two such prices: it is one '
        'price decomposed, so the ceded row here is the cedent\'s allocated '
        'cost of the program rather than a reinsurer\'s quote for it. The '
        'reading is unlimited, so there is no capital column.'
        + grid + gap)


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
    one block each, units across.

    On an occurrence program, **nothing changes**. The allocation is already
    the cedent's one basis reading: the ceded row is what the cession costs
    the cedent out of its own premium rather than what a reinsurer would
    charge, so there is nothing to drop and nothing to star, and there are no
    difference rows because the whole table is a decomposition already. RAW
    and INSURER are identical here, which is the default rule doing its job
    rather than an omission (plan decision 8; revisit only if a REINSURER
    perspective ever arrives, which would read the ceded row differently).
    """
    if getattr(result._source, 'agg_list', None) is not None:
        return [blocks[0], *_stat_slices(blocks[1][1])]
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
             dict(kw, caption=caption))]


# --- pricing.evaluate -------------------------------------------------------

register_simple_exhibit(
    'pricing.evaluate', 'Breakeven acceptability', 'evaluation_df',
    [EvaluationResult],
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
