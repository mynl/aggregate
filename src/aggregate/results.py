"""Typed return values for ``Portfolio`` pricing methods.

Replaces the legacy ``Answer`` dict (a glorified ``dict`` with attribute
access). Each public ``Portfolio`` method that previously returned an
``Answer`` or an inline ``namedtuple`` returns a dataclass declared here.

Currently defined:

- ``AnalyzeDistortionResult`` — single-distortion pricing readout.
- ``AnalyzeDistortionsResult`` — multi-distortion exhibit.
- ``PricingResult`` — :meth:`Portfolio.price`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
        line (units + ``'total'``); columns are the canonical pentagon octet
        ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']`` (see
        :data:`aggregate.pentagon.PENTAGON_STATS`). Lifted from
        :meth:`Portfolio.pricing_at`.
    audit_df : pandas.DataFrame
        One-row total-level audit: descriptor columns ``dname``, ``dshape``
        first, then the canonical pentagon octet as the trailing eight columns
        (so ``audit_df.iloc[:, -8:]`` is the octet).
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
        ``(distortion, stat)`` on rows, line names on columns. ``stat`` is an
        ordered categorical over the canonical pentagon octet
        ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']``
        (:data:`aggregate.pentagon.PENTAGON_STATS`).
    augmented_dfs : dict[str, pandas.DataFrame]
        Snapshot of the Portfolio's augmented_df cache at the time of
        the call -- the per-distortion DataFrames the pricing was read
        from. Keyed by distortion name.
    """

    distortions: dict[str, 'Distortion']
    pricing_df: pd.DataFrame
    augmented_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class PricingResult:
    """Return type for :meth:`Portfolio.price`.

    Attributes
    ----------
    df : pandas.DataFrame
        Per-(distortion, line) pricing readout. MultiIndex
        ``(distortion, unit)`` on rows; columns are the canonical pentagon
        octet ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']``
        (:data:`aggregate.pentagon.PENTAGON_STATS`). ``M/Q`` is named ``ROE``
        (cost of capital, ``CoC``, is the synonym).
    price : float
        Total premium for the last distortion applied (back-compat with
        the legacy single-distortion case).
    price_dict : dict[str, float]
        Premium keyed by distortion name.
    a_reg : float
        Regulatory asset level used in the calculation.
    reg_p : float
        Corresponding probability ``self.cdf(a_reg)``.
    """

    df: pd.DataFrame
    price: float
    price_dict: dict[str, float]
    a_reg: float
    reg_p: float


