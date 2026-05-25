"""Typed return values for ``Portfolio`` pricing methods.

Replaces the legacy ``Answer`` dict (a glorified ``dict`` with attribute
access). Each public ``Portfolio`` method that previously returned an
``Answer`` or an inline ``namedtuple`` returns a dataclass declared here.

Currently defined:

- ``AnalyzeDistortionResult`` — single-distortion pricing readout.
- ``AnalyzeDistortionsResult`` — multi-distortion exhibit.
- ``PricingResult`` — :meth:`Portfolio.price`.
- ``PricingBoundsResult`` — :meth:`Portfolio.pricing_bounds`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .bounds import Bounds
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


@dataclass
class PricingResult:
    """Return type for :meth:`Portfolio.price`.

    Attributes
    ----------
    df : pandas.DataFrame
        Per-(distortion, line) pricing readout. MultiIndex
        ``(distortion, unit)`` on rows; columns include ``L``, ``P``,
        ``M``, ``Q``, ``a``, ``LR``, ``PQ``, ``COC``.
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


@dataclass
class PricingBoundsResult:
    """Return type for :meth:`Portfolio.pricing_bounds`.

    Attributes
    ----------
    bounds : Bounds
        The underlying ``Bounds`` object (with ``cloud_df`` / ``weight_df``
        attached) used to compute the natural allocation cloud.
    allocs : pandas.DataFrame
        Natural allocations across the bound cloud (fast path).
    stats : pandas.DataFrame
        Summary statistics of the natural allocations across the cloud.
    comp : pandas.DataFrame or None
        Per-line min/mean/max comparison of slow-path natural allocations
        with the fast-path allocations. ``None`` if the slow path was not
        run.
    allocs_slow : pandas.DataFrame or None
        Per-(pl, pu) bi-TVaR allocations from the slow path. ``None`` if
        the slow path was not run.
    p_star : float
        Calibrating ``p*`` for the bi-TVaR family at the input premium.
    """

    bounds: 'Bounds'
    allocs: pd.DataFrame
    stats: pd.DataFrame
    comp: pd.DataFrame | None
    allocs_slow: pd.DataFrame | None
    p_star: float
