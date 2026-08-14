"""Typed return values for the pricing methods.

Replaces the legacy ``Answer`` dict (a glorified ``dict`` with attribute
access). Each public method that previously returned an ``Answer``, an inline
``namedtuple`` or a bare ``DataFrame`` returns a dataclass declared here.

Currently defined:

- ``AnalyzeDistortionResult``: single-distortion pricing readout.
- ``AnalyzeDistortionsResult``: multi-distortion exhibit.
- ``PricingResult``: :meth:`Portfolio.price`.
- ``CalibrationResult``: :meth:`Aggregate.calibrate_distortions` and its
  ``Portfolio`` twin.
- ``EvaluationResult``: :meth:`Aggregate.evaluate` and its ``Portfolio`` and
  ``PnL`` twins.

Every result carries ``_source``, the object it was computed from, and borrows
that object's identity through :class:`SourcedMixin`. Two reasons, and the
second is the load-bearing one. A frame alone cannot say what it is about, so a
reader holding one three cells later has lost the book it came from. And an
exhibit dispatches on ``type(obj)``, so serving a **calculation** rather than a
stored frame needs a type to dispatch on: that is what makes
``pricing.calibrate`` / ``pricing.allocate`` / ``pricing.evaluate`` ordinary
registrations rather than a new channel through the exhibit machinery
(``dev/plan-pricing-exhibits.md``, ``[Pricing-Keyed-On-Result]``).

Naming note. The suffix form (``CalibrationResult``, not ``ResultCalibration``)
is deliberate here and is the one place the house ``Base<Kind>`` prefix rule is
not followed: these five are siblings of three names that shipped long before
the rule was written down, and one family spelled two ways reads worse than one
family spelled against the rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .spectral import Distortion


class SourcedMixin:
    """Identity borrowed from the object a result was computed from.

    The four names an exhibit reads off a dispatched object (``name``,
    ``label``, ``_title_name``, ``_relabel``) delegated to ``_source``, so a
    served pricing exhibit is titled ``Calibrated distortions: BasicBook``
    rather than ``Calibration: CalibrationResult``, and its unit axis carries
    the source's labels.

    A result with no source (hand constructed, or unpickled from an older
    release) answers ``None`` and relabels to the identity rather than
    raising: the frames are still the point, and losing the title is not worth
    an exception.

    Mixin, not a base class: the results are dataclasses with no shared state
    beyond ``_source`` and the ``<Role>Mixin`` idiom is what the house rules
    reserve for exactly this (``CLAUDE.md``, naming conventions).
    """

    #: Set as a dataclass field by each host; declared here so the delegation
    #: is total even on a result built without one.
    _source = None

    @property
    def name(self):
        """The source object's identity handle, or ``None``."""
        return getattr(self._source, 'name', None)

    @property
    def label(self):
        """The source object's resolved display label, or ``None``."""
        return getattr(self._source, 'label', None)

    @property
    def _title_name(self):
        """Exhibit-title form of the source; the class name with no source."""
        return getattr(self._source, '_title_name', type(self).__name__)

    def _relabel(self, df):
        """Apply the source's label renamer to a served frame.

        The identity when there is no source, or when the source does not
        carry the label surface.
        """
        relabel = getattr(self._source, '_relabel', None)
        return df if relabel is None else relabel(df)


@dataclass
class AnalyzeDistortionResult(SourcedMixin):
    """Return type for :meth:`Portfolio.analyze_distortion`.

    Attributes
    ----------
    distortion : Distortion
        The pricing distortion this row pertains to.
    pricing_df : pandas.DataFrame
        Per-unit pricing readout at the chosen asset level, indexed by
        unit (units + ``'total'``); columns are the canonical pentagon octet
        ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']`` (see
        :data:`aggregate.pentagon.PENTAGON_STATS`). Lifted from
        :meth:`Portfolio.pricing_at`.
    audit_df : pandas.DataFrame
        One-row total-level audit: descriptor columns ``dname``, ``dshape``
        first, then the canonical pentagon octet as the trailing eight columns
        (so ``audit_df.iloc[:, -8:]`` is the octet).
    _source : Portfolio, optional
        The book this was computed from; see :class:`SourcedMixin`.
    """

    distortion: 'Distortion'
    pricing_df: pd.DataFrame
    audit_df: pd.DataFrame
    _source: object = None


@dataclass
class AnalyzeDistortionsResult(SourcedMixin):
    """Return type for :meth:`Portfolio.analyze_distortions`.

    Attributes
    ----------
    distortions : dict[str, Distortion]
        The distortions analysed, keyed by name.
    pricing_df : pandas.DataFrame
        Concatenated per-distortion exhibit, MultiIndex
        ``(distortion, stat)`` on rows, unit names on columns. ``stat`` is an
        ordered categorical over the canonical pentagon octet
        ``['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']``
        (:data:`aggregate.pentagon.PENTAGON_STATS`).
    augmented_dfs : dict[str, pandas.DataFrame]
        Snapshot of the Portfolio's augmented_df cache at the time of
        the call -- the per-distortion DataFrames the pricing was read
        from. Keyed by distortion name.
    _source : Portfolio, optional
        The book this was computed from; see :class:`SourcedMixin`.
    """

    distortions: dict[str, 'Distortion']
    pricing_df: pd.DataFrame
    augmented_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
    _source: object = None


@dataclass
class PricingResult(SourcedMixin):
    """Return type for :meth:`Portfolio.price`.

    Attributes
    ----------
    df : pandas.DataFrame
        Per-(distortion, unit) pricing readout. MultiIndex
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
    _source : Portfolio, optional
        The book this was computed from; see :class:`SourcedMixin`.
    """

    df: pd.DataFrame
    price: float
    price_dict: dict[str, float]
    a_reg: float
    reg_p: float
    _source: object = None


@dataclass
class CalibrationResult(SourcedMixin):
    """Return type for ``calibrate_distortions`` on ``Aggregate`` and ``Portfolio``.

    The receipt of one calibration: which families were fitted, to what
    target, on which distribution, and what the allocation of that target
    looks like. The first three are computed by the call and stored; the
    fourth is computed on first access and cached, because it is a second
    sweep over the book that a caller who only wanted the shapes should not
    pay for.

    Attributes
    ----------
    distortions : dict[str, Distortion]
        The calibrated distortion objects, keyed by family name.
    distortion_df : pandas.DataFrame
        The per family receipt: ``param_name``, ``param``, ``error``,
        ``gini_p``, ``area``, indexed by family.
    calibration_df : pandas.DataFrame
        The shared one-row target: ``coc``, ``p``, ``F(a)`` then the canonical
        pentagon octet.
    coc : float
        The cost of capital target the families were fitted to. When the
        caller stated a loss ratio (``lr=``) this is the cost of capital that
        loss ratio implies at the resolved anchor.
    lr : float or None
        The loss ratio target as the caller stated it, or ``None`` when the
        caller stated ``coc``.
    p : float
        The resolved exceedance probability at the asset level, ``F(a)``.
    a : float
        The resolved asset level, snapped to the grid.
    anchor : {'p', 'a'}
        Which of the two the caller fixed. The other is derived, and the
        derived frames re-anchor on the one the caller named, so a sweep over
        views holds fixed what the caller held fixed.
    kind : str
        VaR kind used to resolve ``a`` from ``p``.
    names : tuple of str
        The families requested.
    reins_view : str or None
        Which of a cession's distributions the fit was made on; ``None`` is
        the object's own.
    _source : Aggregate or Portfolio, optional
        The object calibrated; see :class:`SourcedMixin`.

    Notes
    -----
    **The allocation frames are lazy, and that is what keeps the RAW exhibit
    invariant intact.** A RAW block is exactly one public frame named for the
    attribute holding it, and an exhibit built on a calculation would
    ordinarily have to invent one. Here it does not: :attr:`pricing_df` and
    :attr:`reins_price_df` are public attributes of this object that return
    real frames, so the invariant reads the same over a result as it does over
    an ``Aggregate``. That they are computed on demand rather than at
    construction is an efficiency question, not a contract question.

    Which one exists depends on what was calibrated. A ``Portfolio`` allocates
    across units (:attr:`pricing_df`); a reinsured ``Aggregate`` allocates
    across views (:attr:`reins_price_df`); an ``Aggregate`` with no cession
    has one distribution and therefore nothing to spread, and its allocation
    story is the degenerate single row of :attr:`calibration_df`.
    """

    distortions: dict[str, 'Distortion']
    distortion_df: pd.DataFrame
    calibration_df: pd.DataFrame
    coc: float
    p: float
    a: float
    anchor: str = 'p'
    lr: float = None
    kind: str = 'lower'
    names: tuple = ()
    reins_view: str = None
    _source: object = None
    _frames: dict = field(default_factory=dict, init=False, repr=False,
                          compare=False)

    @property
    def _anchor_kwargs(self):
        """``{'p': ...}`` or ``{'a': ...}``: the anchor as the caller stated it.

        Re-anchoring the derived frames on the keyword the caller used, rather
        than on the resolved pair, is what makes a sweep mean what the caller
        meant. ``p=`` holds the *threshold* fixed and lets each view or unit
        find its own capital; ``a=`` holds the capital fixed. The two are the
        same number on the calibrated distribution and different numbers
        everywhere else, which is the whole point of the distinction.
        """
        return {self.anchor: getattr(self, self.anchor)}

    @property
    def pricing_df(self):
        """The calibrated set allocated across the units of a ``Portfolio``.

        :meth:`Portfolio.analyze_distortions` at the calibration anchor with
        this result's own distortions: ``(distortion, stat)`` rows, units
        across. Computed on first access and cached.

        Allocation is always on the book's own total, which already is its net
        view; a calibration made on ``reins_view='gross'`` therefore reads as
        the gross-calibrated set applied to the net book, which is a
        deliberate reading and not a mixed basis by accident. See
        :meth:`Portfolio.analyze_distortions`, which refuses the other views
        by name.

        Raises
        ------
        AttributeError
            When the source is not a ``Portfolio``. There is no allocation
            across units when there are no units.
        """
        if 'pricing_df' not in self._frames:
            if not hasattr(self._source, 'analyze_distortions'):
                raise AttributeError(
                    f'{type(self._source).__name__} has no units to allocate '
                    'across, so a calibration on it carries no pricing_df; '
                    'the allocation story is calibration_df.')
            self._frames['pricing_df'] = self._source.analyze_distortions(
                distortions=self.distortions, **self._anchor_kwargs).pricing_df
        return self._frames['pricing_df']

    @property
    def stand_alone_df(self):
        """Every unit of a ``Portfolio`` priced alone, against the book whole.

        ``(distortion, unit)`` rows over the units, then ``sum of parts`` and
        ``total``; the canonical pentagon octet across. Every row prices at the
        calibration's own asset level, so the ``a`` column is constant.
        Computed on first access and cached.

        The counterpart to :attr:`pricing_df`, and the other half of the pair
        the pricing pane exists to put on screen. That frame splits one premium
        across the units, so its rows foot. This prices each unit as its own
        distribution with the same fitted families, so its rows do not, and the
        gap between ``sum of parts`` and ``total`` is what pooling is worth
        under that family (``[Standalone-Prices-The-Parts,
        Allocate-Splits-The-Whole]``).

        A calibration struck on a cession view reads here as that set applied
        to each unit's own distribution, exactly as :attr:`pricing_df` reads as
        it applied to the net book: a deliberate reading, not a mixed basis by
        accident.

        Raises
        ------
        AttributeError
            When the source is not a ``Portfolio``. There are no parts to price
            separately when there are no units.
        """
        if 'stand_alone_df' not in self._frames:
            if getattr(self._source, 'agg_list', None) is None:
                raise AttributeError(
                    f'{type(self._source).__name__} has no units to price '
                    'separately, so a calibration on it carries no '
                    'stand_alone_df; the stand-alone story is '
                    'calibration_df, or reins_price_df on a cession.')
            from ._pricing import stand_alone_price_df
            self._frames['stand_alone_df'] = stand_alone_price_df(
                self._source, self.distortions, self.a)
        return self._frames['stand_alone_df']

    @property
    def reins_price_df(self):
        """The calibrated set applied to every view of a cession.

        :meth:`Aggregate.reins_price_df` at the calibration anchor with this
        result's own distortions: ``(distortion, view)`` rows, the canonical
        pentagon octet across. Computed on first access and cached.

        Raises
        ------
        AttributeError
            When the source carries no cession, so there are no views to
            spread the target over.
        """
        if 'reins_price_df' not in self._frames:
            if not getattr(self._source, 'reins_views', None):
                raise AttributeError(
                    f'{getattr(self._source, "name", "the source")} carries no '
                    'cession, so a calibration on it has no views to price; '
                    'the allocation story is calibration_df.')
            self._frames['reins_price_df'] = self._source.reins_price_df(
                self.distortions, **self._anchor_kwargs)
        return self._frames['reins_price_df']


@dataclass
class EvaluationResult(SourcedMixin):
    """Return type for ``evaluate`` on ``Aggregate``, ``Portfolio`` and ``PnL``.

    The Cherny and Madan breakeven acceptability panel plus the position it
    was measured on: a panel read without its premium and its asset anchor is
    a table of shapes with nothing to hold them to.

    Attributes
    ----------
    evaluation_df : pandas.DataFrame
        The panel: ``(Step, distortion)`` rows, columns ``role`` /
        ``param_name`` / ``param`` / ``gini_p`` / ``error`` / ``status``.
        Named to match :attr:`PnL.evaluation_df`, the frame the waterfall
        exhibit already publishes under that name.
    premium : float or None
        The consideration the position was measured against. ``None`` on a
        ``PnL``, where every ledger row carries its own and no single number
        stands for them.
    reins_view : str or None
        Which of a cession's distributions was evaluated; ``None`` is the
        object's own.
    p : float or None
        The exceedance probability at the asset anchor, or ``None`` when the
        position was evaluated unlimited (the whole distribution).
    a : float or None
        The asset anchor, or ``None`` when unlimited.
    names : tuple of str
        The families reported.
    _source : Aggregate, Portfolio or PnL, optional
        The object evaluated; see :class:`SourcedMixin`.

    Notes
    -----
    ``p`` and ``a`` both ``None`` is the unanchored reading: the position is
    measured against the whole distribution, which is the unlimited case and
    the historical behavior. An anchored panel measures ``P - min(X, a)``, the
    position as an obligation with assets behind it, which is what closes the
    round trip against a calibration at the same anchor.
    """

    evaluation_df: pd.DataFrame
    premium: float = None
    reins_view: str = None
    p: float = None
    a: float = None
    names: tuple = ()
    _source: object = None
