"""Typed return values for ``Portfolio`` pricing methods.

Replaces the legacy ``Answer`` dict (a glorified ``dict`` with attribute
access). Each public ``Portfolio`` method that previously returned an
``Answer`` returns a frozen dataclass declared here.

D.2 introduces ``AnalyzeDistortionResult`` and ``AnalyzeDistortionsResult``
to back the rewritten ``analyze_distortion(s)`` methods. D.3 extends with
``PricingResult`` and ``PricingBoundsResult``, completing the sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .spectral import Distortion


@dataclass
class AnalyzeDistortionResult:
    """Return type for :meth:`Portfolio.analyze_distortion`.

    Attributes
    ----------
    distortion : Distortion
        The pricing distortion this row pertains to.
    pricing_df : pandas.DataFrame
        Per-line pricing readout at the chosen asset level, indexed by
        line (units + ``'total'``), columns ``['L', 'LR', 'M', 'P', 'PQ',
        'Q', 'ROE']``. Lifted from :meth:`Portfolio.pricing_at`.
    audit_df : pandas.DataFrame
        Calibration audit values (``a``, ``LR``, ``ROE``, ``L``, ``P``,
        ``Q``, ``dname``, ``dshape``) at the total level.
    """

    distortion: 'Distortion'
    pricing_df: pd.DataFrame
    audit_df: pd.DataFrame


@dataclass
class AnalyzeDistortionsResult:
    """Return type for :meth:`Portfolio.analyze_distortions`.

    Attributes
    ----------
    distortions : dict[str, Distortion]
        The distortions analysed, keyed by name.
    pricing_df : pandas.DataFrame
        Concatenated per-distortion exhibit, MultiIndex
        ``(distortion, stat)`` on rows, line names on columns. ``stat``
        runs over ``['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE', 'a']``.
    augmented_dfs : dict[str, pandas.DataFrame]
        Snapshot of the Portfolio's augmented_df cache at the time of
        the call -- the per-distortion DataFrames the pricing was read
        from. Keyed by distortion name.
    """

    distortions: dict[str, 'Distortion']
    pricing_df: pd.DataFrame
    augmented_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
