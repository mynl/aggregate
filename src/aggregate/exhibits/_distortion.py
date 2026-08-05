"""Exhibit treatments for :class:`~aggregate.spectral.Distortion`.

Only the check-table emphasis on its structural identity frame; everything
else a Distortion serves is a passthrough declared in the manifest.
"""

from ..spectral import Distortion
from ._core import validation, _check_table_emphasis


@validation.insurer.register(Distortion)
def _validation_insurer_distortion(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Structural identity checks with gates and verdicts; failing '
               'checks are emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_check_table_emphasis(df)))]
