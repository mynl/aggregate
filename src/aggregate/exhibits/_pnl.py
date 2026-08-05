"""Exhibit treatments for :class:`~aggregate._pnl.PnL`.

The accounting family: ``economic`` (the ledger sheet) and
``economic_ratios``, plus the ``stats`` override that explains an absent
engine. Since [PnL-Economic-Frames] a P&L's ``stats_df`` is its engine's
moment store, so ``stats`` is the ordinary treatment rather than a special
case; the ledger has its own exhibit under its own name.
"""

from .._pnl import PnL
from ._core import economic_ratios, stats, _stats_insurer_moment_store


@stats.insurer.register(PnL)
def _stats_insurer_pnl(obj, blocks):
    """The engine's moment store, or an explanation of why there is none.

    Since [PnL-Economic-Frames] a P&L's ``stats_df`` is its engine's moment
    store, so the treatment is the ordinary one. A hand-built kernel P&L
    carries no engine and the frame is empty; rather than serve a blank
    table with no explanation, the caption says what happened. The ledger
    itself is the ``economic`` exhibit.
    """
    block_name, df, kw = blocks[0]
    if df.empty:
        caption = ('This P&L was built from grids rather than from a '
                   'declared book, so it carries no stochastic engine and '
                   'has no moment store to report. Its accounting view is '
                   'the economic exhibit.')
        return [(block_name, df, dict(kw, caption=caption))]
    return _stats_insurer_moment_store(obj, blocks)


# ``economic`` itself is a plain passthrough over ``economic_df`` and is
# declared in the manifest; only its insurer translation will live here
# ([Exhibits-Economic-Insurer]). ``economic_ratios`` carries two blocks, so
# it needs a builder.

@economic_ratios.register(PnL)
def _economic_ratios_frames(obj):
    """The per block amounts and ratios, plus the itemized declared legs."""
    return [('economic_ratios_df', obj.economic_ratios_df, {}),
            ('legs_df', obj.legs_df, {})]
